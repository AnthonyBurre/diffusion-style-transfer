# Diffusion-based artistic style transfer

A Gradio web app for image-to-image artistic style transfer using diffusion models. Sibling project to [`image-style-transfer`](https://github.com/AnthonyBurre/image-style-transfer) - same UX, fundamentally different approach and system requirements (GPU + ~15 GB of model weights).

Stable Diffusion XL (SDXL) is comfortable with ~12 GB VRAM and takes tens of seconds per image even on a recent GPU. A CUDA GPU is the smoothest path, but smaller machines are supported too: a lighter SD 1.5 backend, a few-step `--fast` mode, and automatic CPU offload all trade speed or fidelity for lower memory (see [Hardware](#hardware)).

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

If Docker isn't an option, run directly on the host with [uv](https://docs.astral.sh/uv/):

```shell
uv sync --extra cuda      # Linux / Windows with an NVIDIA GPU (CUDA 12.6 wheels)
uv sync --extra cpu       # CPU-only — or Apple Silicon: the macOS wheel ships MPS
uv run python -m src.app              # add -b sd15 and/or --fast for smaller hardware
```

On Apple Silicon, `--extra cpu` installs the standard macOS arm64 PyTorch wheel, which includes Metal (MPS) support; the device is detected automatically.

## Headless / batch processing

The CLI processes (content, style) pairs without launching the web UI. With no arguments it walks the cartesian product of `examples/content/` × `examples/style/` and writes results to `examples/output/`:

```shell
uv run python -m src.cli                                  # examples/content × examples/style → examples/output
uv run python -m src.cli -c my.jpg -s ref.jpg -o out.webp # single pair to a single file
uv run python -m src.cli -v canny -p "maximum style"      # variant + preset overrides
uv run python -m src.cli -b sd15                          # lighter SD 1.5 backend (older / smaller hardware)
uv run python -m src.cli --fast                           # LCM-LoRA: ~6 steps instead of 30
uv run python -m src.cli --smoke-test -b sd15             # verify the pipeline runs here (tiny synthetic inputs)
```

`--smoke-test` runs a single small generation on in-code synthetic images (no `-c`/`-s`/`-o` needed) and prints the device, memory mode, and per-step timing — a quick way to confirm the pipeline works on your hardware, and how slow it'll be, before committing to a real batch. The first run still downloads the chosen backend's weights.

Every UI knob is exposed as a flag, see `uv run python -m src.cli --help`. Outputs are named `<backend>-<content>_X_<style>.webp` (e.g. `sdxl-…`, `sd15-…`, or `sdxl-lcm-…` with `--fast`), the same convention used when downloading from the Gradio app.

### Backends

1. `--backend sdxl` (default) is the flagship: SDXL + InstantStyle + ControlNet, ~12 GB VRAM, native 1024 px.
2. `--backend sd15` swaps in Stable Diffusion 1.5 with the SD1.5 ControlNet v1.1 + IP-Adapter SD1.5 weights — ~3-4 GB VRAM, native 512 px (auto-applied as `--max-size` default), and much less work per image (a quarter of the pixels on a smaller U-Net). The trade is lower stylisation fidelity and more semantic bleed from the style image: the SD1.5 path applies a flat IP-Adapter scale rather than InstantStyle's per-block targeting used on SDXL.

### Fast mode

`--fast` (CLI and app) works with either backend. It loads the matching [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl) (`lcm-lora-sdxl` / `lcm-lora-sdv1-5`, ~100-400 MB) and switches to the LCM scheduler, so sampling takes ~6 steps instead of 30. It also defaults guidance to 1.0, which disables classifier-free guidance and halves the U-Net work per step. The negative prompt has no effect in this mode. Expect somewhat softer detail than a full 30-step run. It's the best option on slow hardware, and a quick way to preview settings before a full render.

## Background

My last image style transfer project produced some interesting results using three statistic-matching methods (Magenta, Gatys, StyTr²), but failed to fully realize the potential of style transfer. Diffusion methods are the next approach to investigate.

For the mechanics of forward/reverse diffusion:

| Title | Author | Content |
|---|---|---|
| [Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model) | Wikipedia | General overview of the model family with pointers to the major variants and original papers. |
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | Niels Rogge and Kashif Rasul | Walkthrough of a minimal diffusion model implementation in PyTorch line by line. |
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Lilian Weng | Very mathematical long-form derivation of the forward/reverse process, score matching, and DDPM/DDIM sampling. |



### Conditioning

Stable Diffusion is primarily a text-to-image model: at every denoising step the network reads a CLIP-style text embedding and steers toward an image consistent with the prompt. Image conditioning was added later as adapter modules, and the two used here enter the network differently. ControlNet is a trainable copy of the U-Net's encoder that consumes a spatial control signal - a depth map or an edge map - and adds its outputs to the U-Net's skip connections, so it constrains *where* things go. IP-Adapter extracts CLIP image features from a reference image and feeds them through extra cross-attention layers alongside the text, so it acts like an image-valued prompt that steers *what things look like*. InstantStyle restricts IP-Adapter to the U-Net blocks that carry style.

In this pipeline three conditioning signals - the optional text prompt, the depth map of the content image, and the InstantStyle features of the style image - pull on every one of the ~30 denoising steps simultaneously.


## Method

Default path: **SDXL base + InstantStyle (style conditioning) + ControlNet-Depth (content conditioning)**. The SD 1.5 backend has the same structure with SD1.5 weights and plain IP-Adapter (see [Backends](#backends)).

- **Base model**: `stabilityai/stable-diffusion-xl-base-1.0`. SDXL outperforms SD 1.5 for stylisation fidelity and has the most current adapter ecosystem. Paired with the community-fixed VAE `madebyollin/sdxl-vae-fp16-fix` because SDXL's stock VAE produces NaNs in fp16.
- **Style conditioning**: [InstantStyle](https://github.com/InstantStyle/InstantStyle), an IP-Adapter variant explicitly tuned to disentangle style from semantic content. Without it, plain IP-Adapter tends to copy *objects* from the style image, not just texture/colour.
- **Content conditioning**: ControlNet-Depth (`diffusers/controlnet-depth-sdxl-1.0`). Depth maps preserve overall composition without copying low-level patterns from the content image. Maps are produced by `Intel/dpt-hybrid-midas` (DPT-Hybrid), the balanced quality/speed choice from `controlnet-aux`. ControlNet-Canny is offered as an alternative when the user wants edges preserved more literally.

A diffusion sampling loop (30 steps, or ~6 with `--fast`) generates the final image.

## Hardware

Any of these can run the pipeline; they differ in how long you wait. CPU offload switches on automatically below 12 GB of VRAM (CUDA) or 16 GB of unified memory (Apple Silicon), trading speed for fitting in memory. Use `--smoke-test` to see the per-step time on your own machine.

| Setup | What to expect |
|---|---|
| CUDA GPU, 12+ GB | SDXL runs fully on-device; the smoothest experience. |
| CUDA GPU, < 12 GB | Offload kicks in. SDXL streams weights and is slow; `-b sd15` and/or `--fast` help a lot. |
| Apple Silicon, 16+ GB | Runs on-device via MPS. |
| Apple Silicon, 8 GB (e.g. base M2) | Both backends complete via offload, but slowly: SD 1.5 at 512 px / 30 steps measured ~55 min per image, and SDXL longer. `--fast` cuts the step count ~5×. |
| CPU only | Works in fp32; slowest option. |

Disk: **~15 GB** of cached models for SDXL on first run, less for SD 1.5.


## Architecture

- `src/app.py` - Gradio UI. Backend and `--fast` are fixed at launch. Inputs side-by-side on top; controls (preset, ControlNet variant, advanced sliders) lower-left; output lower-right. Saves each result to a tempdir as `<backend>-<content>_X_<style>.webp` and returns the path so the browser download has a meaningful name.
- `src/cli.py` - Headless batch driver. Cartesian product over `examples/content/` × `examples/style/` by default; same naming convention as the app. Every UI knob is a flag.
- `src/pipeline.py` - Backend configs, and `StylePipeline`, which builds the ControlNet pipeline for the chosen backend, loads IP-Adapter (plus the LCM-LoRA in fast mode), applies the depth/canny preprocessor, and runs inference. The base pipeline is built once; switching depth/canny swaps only the ControlNet. Attention runs on torch 2.x SDPA - no xFormers required on either CUDA or MPS. **Lazy-loaded** - the first generation triggers the Hugging Face Hub downloads.
- `src/image_utils.py` - PIL preprocessing (EXIF orientation, RGB convert, resize so dimensions are multiples of 8 for the VAE) and the shared `output_filename()` helper.
- `src/_quiet.py` - Shared warning-filter setup, imported first by both entrypoints to silence known-harmless upstream noise before the pipeline loads.
- `tests/` - Fast no-download unit tests for the preprocessing helpers and pipeline config (`uv run pytest`, ~4 s). The live model path is exercised separately by `python -m src.cli --smoke-test`.

### Model cache behaviour

All weights live in the standard Hugging Face Hub cache (`$HF_HOME` or `~/.cache/huggingface/hub`). The Docker image deliberately does **not** bake them in (would push the image past ~20 GB). First-run download takes 5-15 minutes depending on network. In production, mount the host cache as a volume so subsequent container starts are instant.

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
| IP-Adapter weight | 0.0-1.5 | preset-driven (0.8) | Higher = more style. On SDXL, applied only to InstantStyle's style block; on SD 1.5, applied uniformly. |
| ControlNet conditioning scale | 0.0-1.5 | preset-driven (0.6) | Higher = stricter content adherence. |
| Max output side (px) | 512-1024 | 1024 (SDXL) / 512 (SD 1.5) | Each backend is trained at its default; far below it quality drops. |
| Inference steps | 1-60 | 30 (6 with `--fast`) | Diminishing returns past 30-40; fast mode works in 4-8. |
| Guidance scale | 1.0-15.0 | 5.0 (1.0 with `--fast`) | Lower than typical text-to-image since image conditioning already pulls hard. 1.0 disables CFG. |
| Seed | int, -1 = random | -1 | Reproducibility. |
| Negative prompt | text | `"blurry, low quality, distorted"` | Editable, including down to empty. Ignored when guidance is 1.0. |

## Roadmap

In no particular order:

1. **Per-style LoRA fine-tuning** - a separate training script that trains a LoRA on 10-50 images of a target style, saved into `loras/<style-name>.safetensors`, selectable from the UI. The highest-fidelity path when the goal is matching a specific artist or hand.
2. **Refiner stage** - SDXL ships a refiner model that improves fine detail. Adds ~3 GB but visibly better edges/textures.
3. **Img2img mode** - instead of pure ControlNet conditioning, use the content image as the starting latent (`StableDiffusionXLImg2ImgPipeline` + IP-Adapter). Different tradeoff: more content fidelity, less stylistic freedom.
4. **Hosted inference fallback** - for users without a GPU, allow pointing at a Replicate / Modal / fal.ai endpoint instead of running the pipeline locally.
