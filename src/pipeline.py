from dataclasses import dataclass

import torch
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
)
from transformers import pipeline as hf_pipeline

DEPTH_MODEL = "Intel/dpt-hybrid-midas"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_IMAGE_ENCODER = "models/image_encoder"


@dataclass(frozen=True)
class BackendConfig:
    base_model: str
    pipeline_cls: type
    controlnets: dict[str, str]
    ip_adapter_subfolder: str
    ip_adapter_weight: str
    vae_repo: str | None  # None = use the base model's built-in VAE
    default_max_size: int


BACKENDS: dict[str, BackendConfig] = {
    "sdxl": BackendConfig(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls=StableDiffusionXLControlNetPipeline,
        controlnets={
            "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "canny": "diffusers/controlnet-canny-sdxl-1.0",
        },
        ip_adapter_subfolder="sdxl_models",
        ip_adapter_weight="ip-adapter_sdxl_vit-h.safetensors",
        vae_repo="madebyollin/sdxl-vae-fp16-fix",  # stock SDXL VAE NaNs in fp16
        default_max_size=1024,
    ),
    "sd15": BackendConfig(
        base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        pipeline_cls=StableDiffusionControlNetPipeline,
        controlnets={
            "depth": "lllyasviel/control_v11f1p_sd15_depth",
            "canny": "lllyasviel/control_v11p_sd15_canny",
        },
        ip_adapter_subfolder="models",
        ip_adapter_weight="ip-adapter_sd15.safetensors",
        vae_repo=None,
        default_max_size=512,
    ),
}
DEFAULT_BACKEND = "sdxl"

NEGATIVE_PROMPT_DEFAULT = "blurry, low quality, distorted"

PRESETS: dict[str, dict[str, float]] = {
    "follow content closely": {"ip_adapter_weight": 0.5, "controlnet_scale": 0.85},
    "balanced":               {"ip_adapter_weight": 0.8, "controlnet_scale": 0.6},
    "maximum style":          {"ip_adapter_weight": 1.1, "controlnet_scale": 0.35},
}
DEFAULT_PRESET = "balanced"

LOW_VRAM_THRESHOLD_BYTES = 10 * 1024**3


def _detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _is_low_vram() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory < LOW_VRAM_THRESHOLD_BYTES


def _instant_style_scale(backend: str, weight: float):
    """Per-backend IP-Adapter scale targeting the U-Net blocks that carry
    style information rather than semantic content.

    SDXL: InstantStyle's published mapping inserts the adapter only at the
    deep style blocks (down block 2 attn 1, up block 0 attn 1).

    SD1.5: the SD1.5 U-Net has a different block layout and the per-block
    dict format isn't reliably accepted by every diffusers release for
    SD1.5 IP-Adapter. Apply a flat scalar instead — the result is plain
    IP-Adapter on SD1.5 (slightly more semantic bleed than InstantStyle
    proper), which is an acceptable trade for the lighter backend.
    """
    if backend == "sdxl":
        return {
            "down": {"block_2": [0.0, weight]},
            "up": {"block_0": [0.0, weight, 0.0]},
        }
    return float(weight)


class StylePipeline:
    """Lazy-loaded ControlNet + InstantStyle pipeline.

    One pipeline is built per ControlNet variant on first use, then cached.
    The ``backend`` selects between the SDXL flagship and a lighter SD1.5
    path for older / smaller hardware (see ``BACKENDS``).
    """

    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose from {sorted(BACKENDS)}."
            )
        self.backend = backend
        self.config = BACKENDS[backend]
        self.device, self.dtype = _detect_device()
        self.low_vram = _is_low_vram()
        self._pipes: dict = {}
        self._depth = None
        self._canny = CannyDetector()

    def _load(self, variant: str):
        if variant in self._pipes:
            return self._pipes[variant]
        if variant not in self.config.controlnets:
            raise ValueError(f"Unknown ControlNet variant: {variant}")

        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnets[variant], torch_dtype=self.dtype
        )

        kwargs = dict(
            controlnet=controlnet,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )
        if self.config.vae_repo is not None:
            kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.config.vae_repo, torch_dtype=self.dtype
            )

        pipe = self.config.pipeline_cls.from_pretrained(
            self.config.base_model, **kwargs
        )
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=self.config.ip_adapter_subfolder,
            weight_name=self.config.ip_adapter_weight,
            image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER,
        )

        if self.low_vram:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)

        self._pipes[variant] = pipe
        return pipe

    def _control_image(self, content: Image.Image, variant: str) -> Image.Image:
        if variant == "depth":
            if self._depth is None:
                # Keep the depth model off-GPU when VRAM is tight; it is small
                # enough that CPU inference adds only a couple of seconds.
                depth_device = "cpu" if self.low_vram else self.device
                self._depth = hf_pipeline(
                    "depth-estimation",
                    model=DEPTH_MODEL,
                    device=depth_device,
                )
            return self._depth(content)["depth"].convert("RGB")
        return self._canny(content).convert("RGB")

    def generate(
        self,
        content: Image.Image,
        style: Image.Image,
        prompt: str,
        controlnet_variant: str,
        ip_adapter_weight: float,
        controlnet_scale: float,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        negative_prompt: str,
    ) -> Image.Image:
        pipe = self._load(controlnet_variant)
        pipe.set_ip_adapter_scale(
            _instant_style_scale(self.backend, float(ip_adapter_weight))
        )

        control_image = self._control_image(content, controlnet_variant)

        # MPS does not implement the Generator API; fall back to a CPU generator,
        # which still seeds the MPS kernels deterministically.
        if seed is None:
            generator = None
        else:
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            image=control_image,
            ip_adapter_image=style,
            controlnet_conditioning_scale=float(controlnet_scale),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=content.width,
            height=content.height,
            generator=generator,
        )
        return result.images[0]
