import os
from dataclasses import dataclass

import torch
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
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
    lcm_lora: str  # distillation LoRA used by fast mode
    has_safety_checker: bool = False


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
        lcm_lora="latent-consistency/lcm-lora-sdxl",
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
        lcm_lora="latent-consistency/lcm-lora-sdv1-5",
        has_safety_checker=True,
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

DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
# Fast mode (LCM-LoRA) samples in a handful of steps. Guidance <= 1 disables
# classifier-free guidance, halving the U-Net work per step; the negative
# prompt is ignored as a result.
FAST_STEPS = 6
FAST_GUIDANCE_SCALE = 1.0


def sampling_defaults(fast: bool) -> tuple[int, float]:
    """(steps, guidance scale) defaults for normal or fast mode."""
    if fast:
        return FAST_STEPS, FAST_GUIDANCE_SCALE
    return DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE

# Below this much device memory the pipeline CPU-offloads instead of placing the
# whole model graph on-device, so it fits — and stops swapping to death — on
# small machines. CUDA looks at dedicated VRAM; MPS shares unified system RAM, so
# we look at total RAM there (an 8 GB Apple Silicon machine cannot hold SDXL on
# the GPU at all, and even SD1.5 will thrash without offloading).
CONSTRAINED_CUDA_VRAM_BYTES = 12 * 1024**3
CONSTRAINED_SYSTEM_RAM_BYTES = 16 * 1024**3


def _detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def total_memory_bytes(device: str) -> int | None:
    """Device memory for ``cuda``; total unified/system RAM for ``mps``/``cpu``.

    ``None`` where system RAM can't be queried (``os.sysconf`` is POSIX-only).
    """
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory
    if not hasattr(os, "sysconf"):
        return None
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _is_memory_constrained(device: str) -> bool:
    """Whether to CPU-offload rather than place the whole pipeline on-device."""
    if device == "cuda":
        return total_memory_bytes(device) < CONSTRAINED_CUDA_VRAM_BYTES
    if device == "mps":
        return total_memory_bytes(device) < CONSTRAINED_SYSTEM_RAM_BYTES
    return False  # plain CPU: nothing to offload to


def _ip_adapter_scale(backend: str, weight: float):
    """Per-backend IP-Adapter scale.

    SDXL: InstantStyle's "style only" mapping — the adapter is injected at
    up block 0 attn 1, the block that carries style. (InstantStyle's other
    block, down block 2, carries layout; ControlNet already supplies the
    layout here, so injecting the style image's layout would fight it.)

    SD1.5: InstantStyle publishes no block mapping for the SD1.5 U-Net, so a
    flat scalar is applied — plain IP-Adapter, with more semantic bleed from
    the style image than InstantStyle on SDXL.
    """
    if backend == "sdxl":
        return {"up": {"block_0": [0.0, weight, 0.0]}}
    return float(weight)


class StylePipeline:
    """Lazy-loaded ControlNet + InstantStyle pipeline.

    The base pipeline is built once on first use; ControlNet variants are
    loaded on demand and swapped into it, so switching depth/canny doesn't
    load a second copy of the base model. The ``backend`` selects between
    SDXL and the lighter SD1.5 path (see ``BACKENDS``); ``fast`` adds an
    LCM-LoRA for few-step sampling.
    """

    def __init__(self, backend: str = DEFAULT_BACKEND, fast: bool = False) -> None:
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose from {sorted(BACKENDS)}."
            )
        self.backend = backend
        self.fast = fast
        self.config = BACKENDS[backend]
        self.device, self.dtype = _detect_device()
        self.memory_constrained = _is_memory_constrained(self.device)
        self._pipe = None
        self._variant: str | None = None
        self._controlnets: dict[str, ControlNetModel] = {}
        self._depth = None
        self._canny = CannyDetector()

    @property
    def label(self) -> str:
        """Backend name plus mode, used as the output filename prefix."""
        return f"{self.backend}-lcm" if self.fast else self.backend

    def _controlnet(self, variant: str) -> ControlNetModel:
        if variant not in self.config.controlnets:
            raise ValueError(f"Unknown ControlNet variant: {variant}")
        if variant not in self._controlnets:
            self._controlnets[variant] = ControlNetModel.from_pretrained(
                self.config.controlnets[variant], torch_dtype=self.dtype
            )
        return self._controlnets[variant]

    def _load(self, variant: str):
        controlnet = self._controlnet(variant)
        if self._pipe is None:
            self._pipe = self._build(controlnet)
        elif self._variant != variant:
            if not self.memory_constrained:
                self._pipe.controlnet.to("cpu")  # free VRAM for the incoming one
            self._pipe.controlnet = controlnet
            # Offload hooks are per-component; re-placing reattaches them to
            # the new ControlNet (the enable_* calls clear the old hooks first).
            self._place(self._pipe)
        self._variant = variant
        return self._pipe

    def _build(self, controlnet: ControlNetModel):
        kwargs = dict(
            controlnet=controlnet,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )
        if self.config.has_safety_checker:
            # SD1.5's NSFW checker false-positives on ordinary inputs (it
            # blacked out the smoke test's shapes) and costs ~1.2 GB of memory.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
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
        if self.fast:
            pipe.load_lora_weights(self.config.lcm_lora)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        self._place(pipe)
        return pipe

    def _place(self, pipe) -> None:
        """Put ``pipe`` on the device, or CPU-offload it on small machines."""
        if self.memory_constrained:
            # Keep peak memory low so the pipeline fits a small GPU / 8 GB Mac
            # instead of OOMing (SDXL) or swapping until the OS kills it (SD1.5).
            if self.backend == "sdxl":
                # SDXL's ~9 GB of fp16 weights can't sit on a small device at
                # once; stream them submodule-by-submodule. Slow, but it runs.
                pipe.enable_sequential_cpu_offload(device=self.device)
            else:
                # SD1.5 fits with whole-model offload: only the active model is
                # resident, avoiding the per-submodule streaming cost.
                pipe.enable_model_cpu_offload(device=self.device)
            # Tile the VAE decode to cap its peak-memory spike at high resolution.
            # NB: attention slicing is intentionally NOT enabled — it swaps the
            # UNet attention processors and clobbers IP-Adapter's, breaking
            # stylisation ('tuple' object has no attribute 'shape').
            pipe.vae.enable_tiling()
        else:
            pipe.to(self.device)

    def _control_image(self, content: Image.Image, variant: str) -> Image.Image:
        if variant == "depth":
            if self._depth is None:
                # Keep the depth model off-GPU when VRAM is tight; it is small
                # enough that CPU inference adds only a couple of seconds.
                depth_device = "cpu" if self.memory_constrained else self.device
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
            _ip_adapter_scale(self.backend, float(ip_adapter_weight))
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
