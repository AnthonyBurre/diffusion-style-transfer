# Diffusion-based artistic style transfer

A Gradio web app for image-to-image artistic style transfer using diffusion models. Sibling project to [`image-style-transfer`](https://github.com/AnthonyBurre/image-style-transfer) - same UX, fundamentally different approach and system requirements (GPU + ~15 GB of model weights).

Stable Diffusion XL (SDXL) is comfortable with ~12 GB VRAM and takes tens of seconds per image even on a recent GPU. This project assumes a CUDA-capable GPU (or Apple Silicon MPS as a slower fallback).

## Run with Docker (CUDA)

**Linux, or Windows via WSL2 - both with an NVIDIA GPU.** `--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), which is supported on Linux and inside WSL2 (NVIDIA exposes CUDA into the WSL2 VM). Docker Desktop on macOS runs containers inside a VM with no GPU passthrough.

```shell
docker build -t style-transfer-diffusion .
docker run --rm --gpus all -p 7860:7860 \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  style-transfer-diffusion
```

The `-v` mount points the container at the host Hugging Face cache; see [Model cache behaviour](#model-cache-behaviour) for why.

## Run on the host

If Docker isn't an option (macOS, or any host without the NVIDIA Container Toolkit), run directly on the host.

```shell
python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt          # add --extra-index-url https://download.pytorch.org/whl/cu124 on CUDA hosts (Linux or Windows)
python -m src.app
```

On Apple Silicon the default PyPI index is correct and MPS is detected automatically.

## Background

My last image style transfer project produced some interesting results using three statistic-matching methods (Magenta, Gatys, StyTr²), but failed to fully realize the potential of style transfer. Diffusion methods are the next approach to investigate.

For the mechanics of forward/reverse diffusion:

| Title | Author | Content |
|---|---|---|
| [Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model) | Wikipedia | General overview of the model family with pointers to the major variants and original papers. |
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | Niels Rogge and Kashif Rasul | Walkthrough of a minimal diffusion model implementation in PyTorch line by line. |
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Lilian Weng | Very mathematical long-form derivation of the forward/reverse process, score matching, and DDPM/DDIM sampling. |



### Conditioning

Stable Diffusion is primarily a text-to-image model: at every denoising step the network reads a CLIP-style text embedding and steers toward an image consistent with the prompt. Image conditioning was added later as adapter modules. ControlNet adds a parallel branch that consumes a spatial control signal - a depth map or an edge map - and injects it into the U-Net's residual blocks. IP-Adapter (and its style-specialised variant InstantStyle) extracts CLIP image features from a reference image and feeds them through learned cross-attention layers. Both are translators that convert images into the same kind of conditioning vector the network already knows how to listen to.

In this pipeline three conditioning signals - the optional text prompt, the depth map of the content image, and the InstantStyle features of the style image - pull on every one of the ~30 denoising steps simultaneously.

A useful side effect of generating rather than editing: tonal extremes survive. The statistic-matching methods in the sibling project (Magenta, Gatys, StyTr²) match Gram matrices and channel-wise mean/std, both invariant to absolute pixel intensity, so a true black in the content tends to get re-normalised away regardless of whether the style image also has true blacks. Diffusion has no feature-statistic loss in the loop - a pixel value of 0 is as easy to produce as any other, and the base model has seen enough museum-scraped paintings during training to know how deep blacks behave under different brushwork.

### Limits on out-of-distribution inputs

Content images can be wildly out-of-distribution without much issue. ControlNet does not pass the raw content image to the model - it passes a depth map or edge map, and depth/edges are universal features. Whatever the content shows, its depth map looks like a depth map, and the model just sees "structure roughly here, here, and here".

Style is the OOD-sensitive direction. IP-Adapter relies on CLIP, a generic visual–semantic encoder trained on ~400 M image–text pairs. For mainstream art-historical styles - impressionism, watercolour, oil painting, ink wash, anime, cyberpunk, the modal Artstation aesthetic - CLIP has rich representations and the diffusion model has seen plenty of training examples; results are good. For genuinely novel styles - an unknown contemporary illustrator, an obscure 19th-century engraver, an idiosyncratic personal style - IP-Adapter still produces an output, but it is the closest approximation the base model can assemble from styles it already knows. Closing that gap is the job of per-style LoRA fine-tuning (see [Roadmap](#roadmap)).

## Method

Single inference path: **SDXL base + InstantStyle (style conditioning) + ControlNet-Depth (content conditioning)**.

- **Base model**: `stabilityai/stable-diffusion-xl-base-1.0`. SDXL outperforms SD 1.5 for stylisation fidelity and has the most current adapter ecosystem. Paired with the community-fixed VAE `madebyollin/sdxl-vae-fp16-fix` because SDXL's stock VAE produces NaNs in fp16.
- **Style conditioning**: [InstantStyle](https://github.com/InstantStyle/InstantStyle), an IP-Adapter variant explicitly tuned to disentangle style from semantic content. Without it, plain IP-Adapter tends to copy *objects* from the style image, not just texture/colour.
- **Content conditioning**: ControlNet-Depth (`diffusers/controlnet-depth-sdxl-1.0`). Depth maps preserve overall composition without copying low-level patterns from the content image. Maps are produced by `Intel/dpt-hybrid-midas` (DPT-Hybrid), the balanced quality/speed choice from `controlnet-aux`. ControlNet-Canny is offered as an alternative when the user wants edges preserved more literally.

A diffusion sampling loop (~30 steps) generates the final image.

## Hardware

| Setup | VRAM | Per-image latency | Notes |
|---|---|---|---|
| 8 GB GPU | 8 GB | ~2 min | Sequential CPU offload kicks in automatically (10 GB VRAM threshold) |
| 12+ GB GPU | 12 GB | ~15–30 s | Recommended baseline |
| 24 GB GPU | 24 GB | ~10 s | Comfortable; can run higher resolutions / bigger batches |
| Apple Silicon (MPS) | unified | ~45–90 s on M2/M3 Max | Works via `diffusers` MPS backend |
| CPU only | n/a | 5+ min | Technically works, not recommended |

Disk: **~15 GB** for cached models on first run.


## Architecture

- `src/app.py` - Gradio `Interface`. Inputs: content image, style image, optional prompt, ControlNet variant (depth/canny), advanced controls (see [Parameters](#parameters)). Output: stylised image.
- `src/pipeline.py` - Builds the `StableDiffusionXLControlNetPipeline`, loads InstantStyle weights, applies the depth/canny preprocessor, runs inference. Attention runs on torch 2.x SDPA - no xFormers required on either CUDA or MPS. **Lazy-loaded** - first call triggers ~15 GB of Hugging Face Hub downloads and a few seconds of CUDA init.
- `src/image_utils.py` - PIL preprocessing (EXIF orientation, RGB convert, resize so dimensions are multiples of 8 for the VAE).

### Model cache behaviour

All weights live in the standard Hugging Face Hub cache (`$HF_HOME` or `~/.cache/huggingface/hub`). The Docker image deliberately does **not** bake them in (would push the image past ~20 GB). First-run download takes 5–15 minutes depending on network. In production, mount the host cache as a volume so subsequent container starts are instant.

## Parameters

The GUI exposes named presets which map to fixed combinations of IP-Adapter weight and ControlNet conditioning scale. Raw knobs sit behind a collapsible **Advanced** panel.

### Presets

| Preset | IP-Adapter weight | ControlNet conditioning scale |
|---|---|---|
| Follow content closely | 0.5 | 0.85 |
| Balanced *(default)* | 0.8 | 0.6 |
| Maximum style | 1.1 | 0.35 |

### Advanced

| Parameter | Range | Default | Notes |
|---|---|---|---|
| Content conditioning | depth / canny | depth | Depth preserves composition; canny preserves edges literally. |
| IP-Adapter weight | 0.0–1.5 | preset-driven (0.8) | Higher = more style. Applied via InstantStyle's deep-block-only schedule. |
| ControlNet conditioning scale | 0.0–1.5 | preset-driven (0.6) | Higher = stricter content adherence. |
| Max output side (px) | 512–1024 | 1024 | SDXL is trained at 1024; below ~768 quality drops. Above 1024 needs tiling. |
| Inference steps | 10–60 | 30 | Diminishing returns past 30–40. |
| Guidance scale | 1.0–15.0 | 5.0 | Lower than typical text-to-image since image conditioning already pulls hard. |
| Seed | int, -1 = random | -1 | Reproducibility. |
| Negative prompt | text | `"blurry, low quality, distorted"` | Editable, including down to empty. |

## Roadmap

In no particular order:

1. **Per-style LoRA fine-tuning** - a separate training script that trains a LoRA on 10–50 images of a target style, saved into `loras/<style-name>.safetensors`, selectable from the UI. The highest-fidelity path when the goal is matching a specific artist or hand.
2. **Refiner stage** - SDXL ships a refiner model that improves fine detail. Adds ~3 GB but visibly better edges/textures.
3. **Img2img mode** - instead of pure ControlNet conditioning, use the content image as the starting latent (`StableDiffusionXLImg2ImgPipeline` + IP-Adapter). Different tradeoff: more content fidelity, less stylistic freedom.
4. **Hosted inference fallback** - for users without a GPU, allow pointing at a Replicate / Modal / fal.ai endpoint instead of running the pipeline locally.
