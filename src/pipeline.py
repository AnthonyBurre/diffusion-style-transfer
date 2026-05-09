import torch
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from transformers import pipeline as hf_pipeline

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNETS = {
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
}
DEPTH_MODEL = "Intel/dpt-hybrid-midas"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl_vit-h.safetensors"
IP_ADAPTER_IMAGE_ENCODER = "models/image_encoder"

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


def _instant_style_scale(weight: float) -> dict:
    """InstantStyle inserts the IP-Adapter only at the deep blocks that carry
    style information, keeping the semantic blocks untouched."""
    return {
        "down": {"block_2": [0.0, weight]},
        "up": {"block_0": [0.0, weight, 0.0]},
    }


class StylePipeline:
    """Lazy-loaded SDXL + ControlNet + InstantStyle pipeline.

    One pipeline is built per ControlNet variant on first use, then cached.
    """

    def __init__(self) -> None:
        self.device, self.dtype = _detect_device()
        self.low_vram = _is_low_vram()
        self._pipes: dict[str, StableDiffusionXLControlNetPipeline] = {}
        self._depth = None
        self._canny = CannyDetector()

    def _load(self, variant: str) -> StableDiffusionXLControlNetPipeline:
        if variant in self._pipes:
            return self._pipes[variant]
        if variant not in CONTROLNETS:
            raise ValueError(f"Unknown ControlNet variant: {variant}")

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNETS[variant], torch_dtype=self.dtype
        )
        vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=self.dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT,
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
        pipe.set_ip_adapter_scale(_instant_style_scale(float(ip_adapter_weight)))

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
