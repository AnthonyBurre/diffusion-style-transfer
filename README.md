# Diffusion-based artistic style transfer

A Gradio web app for image-to-image artistic style transfer using diffusion models. Sibling project to [`image-style-transfer`](https://github.com/AnthonyBurre/image-style-transfer) — same UX, fundamentally different approach and system requirements (GPU + ~15 GB of model weights).

Stable Diffusion XL (SDXL) needs ~12 GB VRAM and tens of seconds per image even on a recent GPU. This project assumes a CUDA-capable GPU (or Apple Silicon MPS as a slower fallback).

## Run with Docker (CUDA)

**Linux host with an NVIDIA GPU only.** `--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), which is Linux-only — Docker Desktop on macOS and Windows runs containers inside a VM that has no GPU passthrough. 

```shell
docker build -t style-transfer-diffusion .
docker run --rm --gpus all -p 7860:7860 \
  -v $HOME/.cache/huggingface:/app/.cache/huggingface \
  style-transfer-diffusion
```

## Run on the host

On a Mac (including Apple Silicon), skip Docker and use the host instructions below; MPS is detected automatically.

```shell
.venv/bin/python -m src.app
```

## Background

My last image style transfer project produced some interesting results using three statistic-matching methods (Magenta, Gatys, StyTr²), but failed to fully realize the potential of style transfer. Diffusion methods are the next approach to investigate.


### Forward and reverse diffusion

A diffusion model is trained on a corruption process. Take a real image, add a tiny bit of Gaussian noise, add another tiny bit, repeat for ~1000 steps; by the end the image is indistinguishable from pure noise. A neural network is then trained on the inverse task: given a noisy image at step *t*, predict the noise that was added on the way in. With hundreds of millions of image–text pairs as training data, the network ends up implicitly modelling the statistical structure of natural images.

Inference runs the process in reverse. Start from pure Gaussian noise, ask the network "what noise is in this?", subtract a small fraction of the prediction, and repeat for ~30–50 steps. Each step nudges the sample toward something the network considers plausible, so the noise gradually resolves into a coherent image. Nothing is being looked up or copied — the network has learned what natural images look like well enough that it can sculpt noise into one.

### Conditioning

The original conditioning signal is text. Stable Diffusion, DALL-E, Imagen and similar systems are primarily text-to-image models: at every denoising step the network reads a CLIP-style text embedding and steers the prediction toward an image consistent with the prompt.

Image conditioning was added later as adapter modules on top of already-trained text-to-image models. ControlNet adds a parallel branch that consumes a spatial control signal — a depth map or an edge map — and injects it into the U-Net's residual blocks. IP-Adapter (and its style-specialised variant InstantStyle) extracts CLIP image features from a reference image and feeds them through learned cross-attention layers. Mechanically, both are translators that convert images into the same kind of conditioning vector the network already knows how to listen to.

In this pipeline three conditioning signals — the optional text prompt, the depth map of the content image, and the InstantStyle features of the style image — pull on every one of the ~30 denoising steps simultaneously.

### Why this preserves absolute tonality

The statistic-matching methods in the sibling project (Magenta, Gatys, StyTr²) treat the content image as a starting point and edit its features until the feature statistics match the style image. The losses involved match feature statistics — Gram matrices and channel-wise mean/std — which are by construction invariant to absolute pixel intensity, so a true black in the content tends to get re-normalised away regardless of whether the style image also has true blacks.

Diffusion does not edit, it generates. The output is sculpted from noise under the influence of the conditioning signals, with no feature-statistic loss in the loop. A pixel value of 0 is as easy to produce as any other value, and the base model has seen enough museum-scraped paintings during training to know how deep blacks behave under different brushwork. Even with a fairly generic style prompt, tonal extremes survive.

### Limits on out-of-distribution inputs

Content images can be wildly out-of-distribution without much issue. ControlNet does not pass the raw content image to the model — it passes a depth map or edge map, and depth/edges are universal features. Whatever the content shows, its depth map looks like a depth map, and the model just sees "structure roughly here, here, and here".

Style is the OOD-sensitive direction. IP-Adapter relies on CLIP, a generic visual–semantic encoder trained on ~400 M image–text pairs. For mainstream art-historical styles — impressionism, watercolour, oil painting, ink wash, anime, cyberpunk, the modal Artstation aesthetic — CLIP has rich representations and the diffusion model has seen plenty of training examples; results are good. For genuinely novel styles — an unknown contemporary illustrator, an obscure 19th-century engraver, an idiosyncratic personal style — IP-Adapter still produces an output, but it is the closest approximation the base model can assemble from styles it already knows. Closing that gap is the job of per-style LoRA fine-tuning (see Roadmap): a small set of additional weights trained on 10–50 examples extends the model's repertoire to include a target style at the same level it knows impressionism.

## Method

Single inference path: **SDXL base + InstantStyle (style conditioning) + ControlNet-Depth (content conditioning)**.

- **Base model**: `stabilityai/stable-diffusion-xl-base-1.0`. SDXL outperforms SD 1.5 for stylisation fidelity and has the most current adapter ecosystem.
- **Style conditioning**: [InstantStyle](https://github.com/InstantStyle/InstantStyle), an IP-Adapter variant explicitly tuned to disentangle style from semantic content. Without it, plain IP-Adapter tends to copy *objects* from the style image, not just texture/colour.
- **Content conditioning**: ControlNet-Depth (`diffusers/controlnet-depth-sdxl-1.0`). Depth maps preserve overall composition without copying low-level patterns from the content image. ControlNet-Canny is offered as an alternative when the user wants edges preserved more literally.

A diffusion sampling loop (~30 steps) generates the final image. An optional text prompt acts as a third conditioning signal (e.g. "oil painting", "ink wash").

## Hardware

| Setup | VRAM | Per-image latency | Notes |
|---|---|---|---|
| 8 GB GPU | 8 GB | ~2 min | Requires `pipe.enable_sequential_cpu_offload()` |
| 12+ GB GPU | 12 GB | ~15–30 s | Recommended baseline |
| 24 GB GPU | 24 GB | ~10 s | Comfortable; can run higher resolutions / bigger batches |
| Apple Silicon (MPS) | unified | ~45–90 s on M2/M3 Max | Works via `diffusers` MPS backend |
| CPU only | n/a | 5+ min | Technically works, not recommended |

Disk: **~15 GB** for cached models on first run.


## Architecture

- `src/app.py` — Gradio `Interface`. Inputs: content image, style image, optional prompt, ControlNet variant (depth/canny), advanced controls (IP-Adapter weight, ControlNet conditioning scale, num inference steps, guidance scale, seed). Output: stylised image.
- `src/pipeline.py` — Builds the `StableDiffusionXLControlNetPipeline`, loads InstantStyle weights, applies the depth/canny preprocessor, runs inference. **Lazy-loaded** — first call triggers ~15 GB of Hugging Face Hub downloads and a few seconds of CUDA init.
- `src/image_utils.py` — PIL preprocessing (EXIF orientation, RGB convert, resize so dimensions are multiples of 8 for the VAE).

### Model cache behaviour

All weights live in the standard Hugging Face Hub cache (`$HF_HOME` or `~/.cache/huggingface/hub`). The Docker image deliberately does **not** bake them in (would push the image past ~20 GB). First-run download takes 5–15 minutes depending on network. In production, mount the host cache as a volume so subsequent container starts are instant.

The volume mount avoids re-downloading ~15 GB on every container start.

## Defaults and rationale

Decisions that were close calls, recorded so the rationale doesn't have to be reconstructed at implementation time.

- **Depth estimator: DPT-Hybrid.** `controlnet-aux` ships MiDaS, DPT and ZoeDepth. DPT-Hybrid is the balanced choice on quality vs. speed; ZoeDepth gives metric depth at higher cost, MiDaS is faster but coarser.
- **VAE: `madebyollin/sdxl-vae-fp16-fix`.** SDXL's stock VAE produces NaNs in fp16, and the pipeline runs fp16 by default for speed and VRAM. The community-fixed VAE is a drop-in replacement with no quality loss.
- **Attention backend: torch 2.x SDPA.** Scaled-dot-product attention is built into recent PyTorch and is sufficient on both CUDA and MPS. xFormers is no longer required and is not installed unless a benchmark on the target hardware shows a meaningful win.
- **Output resolution: 1024×1024.** SDXL is trained at this resolution; below ~768 quality drops noticeably, and above 1024 needs tiling. User-overridable from the UI.
- **UI surface: presets in front, raw knobs behind.** The default UI exposes 2–3 named presets (*follow content closely* / *balanced* / *maximum style*) which map to fixed combinations of IP-Adapter weight and ControlNet conditioning scale. IP-Adapter weight, conditioning scale, step count, guidance scale and seed are accessible behind a collapsible **Advanced** panel.
- **Negative prompt: hard-coded default, user-overridable.** SDXL responds well to a generic negative prompt; the default is `"blurry, low quality, distorted"`. The textbox is editable, including down to empty.

## Roadmap

In no particular order:

1. **Per-style LoRA fine-tuning** — a separate training script that trains a LoRA on 10–50 images of a target style, saved into `loras/<style-name>.safetensors`, selectable from the UI. The highest-fidelity path when the goal is matching a specific artist or hand.
2. **Refiner stage** — SDXL ships a refiner model that improves fine detail. Adds ~3 GB but visibly better edges/textures.
3. **Img2img mode** — instead of pure ControlNet conditioning, use the content image as the starting latent (`StableDiffusionXLImg2ImgPipeline` + IP-Adapter). Different tradeoff: more content fidelity, less stylistic freedom.
4. **Hosted inference fallback** — for users without a GPU, allow pointing at a Replicate / Modal / fal.ai endpoint instead of running the pipeline locally.
