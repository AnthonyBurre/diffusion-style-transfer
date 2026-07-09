"""Headless CLI: stylise (content, style) image pairs and write the result(s).

The bare invocation ``python -m src.cli`` batch-processes every image in
``examples/content`` against every image in ``examples/style`` through the
InstantStyle + ControlNet pipeline, writing ``.webp`` files into
``examples/output/`` using the same filename convention as the Gradio app.

Each of ``-c`` / ``-s`` may be a file or a directory; with directory inputs
the cartesian product of (content, style) pairs is processed. ``-o`` is
either an output file (single-pair only) or an output directory.

``--backend`` selects the diffusion base model: ``sdxl`` (default, ~12 GB
VRAM) or ``sd15`` (lighter, faster, for older / smaller hardware).

Sibling of ``src.app`` (the Gradio UI); the two share the pipeline and
preprocessing modules but not dispatch code, so UI changes can't ripple
into the CLI.
"""
from . import _quiet  # noqa: F401  -- installs warning filters before pipeline import

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .image_utils import output_filename, prepare_content_image, prepare_style_image
from .pipeline import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_PRESET,
    NEGATIVE_PROMPT_DEFAULT,
    PRESETS,
    StylePipeline,
    total_memory_bytes,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# --smoke-test uses a deliberately tiny, fixed config so it is a fast "does it
# run on this machine?" probe rather than a quality render.
SMOKE_TEST_STEPS = 4
SMOKE_TEST_SIZE = 256


def _resolve_inputs(path):
    """Return a sorted list of image paths for a file-or-directory argument."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in _IMAGE_EXTS)
        if not files:
            sys.exit(f"error: no image files in {p}")
        return files
    sys.exit(f"error: path not found: {p}")


def _save(result, out_path):
    if out_path.suffix.lower() == ".webp":
        result.save(out_path, format="webp", lossless=True)
    else:
        result.save(out_path)


def run(args):
    contents = _resolve_inputs(args.content)
    styles = _resolve_inputs(args.style)

    out = Path(args.output)
    is_file_output = out.suffix.lower() in _IMAGE_EXTS
    total = len(contents) * len(styles)

    if is_file_output and total > 1:
        sys.exit(
            f"error: file output ({out}) requires exactly one content × one style; "
            f"got {len(contents)} × {len(styles)}"
        )

    if not is_file_output:
        out.mkdir(parents=True, exist_ok=True)

    preset = PRESETS[args.preset]
    ip_adapter_weight = (
        args.ip_adapter_weight if args.ip_adapter_weight is not None
        else preset["ip_adapter_weight"]
    )
    controlnet_scale = (
        args.controlnet_scale if args.controlnet_scale is not None
        else preset["controlnet_scale"]
    )
    seed = None if args.seed is None or args.seed < 0 else args.seed

    max_size = (
        args.max_size if args.max_size is not None
        else BACKENDS[args.backend].default_max_size
    )

    pipeline = StylePipeline(backend=args.backend)
    mem_gb = total_memory_bytes(pipeline.device) / 1024**3
    sys.stderr.write(
        f"device={pipeline.device}  memory={mem_gb:.0f} GB  backend={args.backend}  "
        f"offload={'on' if pipeline.memory_constrained else 'off'}\n"
    )
    if pipeline.memory_constrained and args.backend == "sdxl":
        sys.stderr.write(
            "warning: SDXL on a memory-constrained device streams weights via "
            "sequential CPU offload — expect minutes per image; --backend sd15 "
            "is far faster here.\n"
        )
    sys.stderr.flush()

    # Prepared once: each style is reused across every content image.
    prepared_styles = [
        (p, prepare_style_image(Image.open(p))) for p in styles
    ]

    failures = 0
    i = 0
    for content_path in contents:
        content_img = prepare_content_image(Image.open(content_path), max_size=max_size)
        for style_path, style_img in prepared_styles:
            i += 1
            label = f"{content_path.stem} × {style_path.stem}"
            sys.stderr.write(f"[{i}/{total}] {label}\n")
            sys.stderr.flush()

            out_path = out if is_file_output else out / output_filename(
                content_path.stem, style_path.stem, backend=args.backend
            )
            try:
                result = pipeline.generate(
                    content=content_img,
                    style=style_img,
                    prompt=args.prompt,
                    controlnet_variant=args.controlnet_variant,
                    ip_adapter_weight=ip_adapter_weight,
                    controlnet_scale=controlnet_scale,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                    negative_prompt=args.negative_prompt,
                )
                _save(result, out_path)
            except Exception as exc:  # noqa: BLE001 - one bad pair shouldn't abort the batch
                failures += 1
                sys.stderr.write(f"  FAILED {label}: {type(exc).__name__}: {exc}\n")
                sys.stderr.flush()
                continue

            if not out_path.exists() or out_path.stat().st_size == 0:
                failures += 1
                sys.stderr.write(f"  FAILED {label}: no output file written\n")
                sys.stderr.flush()
                continue

            print(str(out_path.resolve()))

    if failures:
        sys.exit(f"{failures} of {total} pair(s) failed.")


def _synthetic_images() -> tuple[Image.Image, Image.Image]:
    """Two tiny in-code images so ``--smoke-test`` needs nothing on disk.

    Content gets clear shapes/edges (so depth and canny both have structure to
    follow); style gets bold colour bands (so the IP-Adapter has a palette and
    texture to transfer).
    """
    content = Image.new("RGB", (SMOKE_TEST_SIZE, SMOKE_TEST_SIZE), (40, 60, 90))
    d = ImageDraw.Draw(content)
    d.ellipse((48, 48, 208, 208), fill=(220, 200, 160), outline=(15, 15, 15), width=4)
    d.rectangle((96, 120, 160, 232), fill=(120, 80, 60), outline=(0, 0, 0), width=3)
    d.line((0, 205, SMOKE_TEST_SIZE, 175), fill=(0, 0, 0), width=5)

    style = Image.new("RGB", (SMOKE_TEST_SIZE, SMOKE_TEST_SIZE))
    s = ImageDraw.Draw(style)
    bands = [(231, 76, 60), (241, 196, 15), (46, 204, 113), (52, 152, 219), (155, 89, 182)]
    band_h = SMOKE_TEST_SIZE // len(bands)
    for i, colour in enumerate(bands):
        s.rectangle((0, i * band_h, SMOKE_TEST_SIZE, (i + 1) * band_h), fill=colour)
    return content, style


def _smoke_test(args) -> None:
    """Run one tiny generation on synthetic inputs to verify the pipeline works
    on this machine. Exits non-zero with a clear message on any failure."""
    import time

    content = prepare_content_image(_synthetic_images()[0], max_size=SMOKE_TEST_SIZE)
    style = prepare_style_image(_synthetic_images()[1])

    pipeline = StylePipeline(backend=args.backend)
    mem_gb = total_memory_bytes(pipeline.device) / 1024**3
    preset = PRESETS[DEFAULT_PRESET]
    sys.stderr.write(
        f"smoke-test: device={pipeline.device}  memory={mem_gb:.0f} GB  "
        f"backend={args.backend}  offload={'on' if pipeline.memory_constrained else 'off'}\n"
        f"smoke-test: {SMOKE_TEST_SIZE}px synthetic inputs, {args.controlnet_variant} "
        f"conditioning, {SMOKE_TEST_STEPS} steps "
        f"(first run downloads the {args.backend} weights)\n"
    )
    sys.stderr.flush()

    start = time.perf_counter()
    try:
        result = pipeline.generate(
            content=content,
            style=style,
            prompt="",
            controlnet_variant=args.controlnet_variant,
            ip_adapter_weight=preset["ip_adapter_weight"],
            controlnet_scale=preset["controlnet_scale"],
            steps=SMOKE_TEST_STEPS,
            guidance_scale=5.0,
            seed=0,
            negative_prompt=NEGATIVE_PROMPT_DEFAULT,
        )
    except Exception as exc:  # noqa: BLE001 - report clearly and signal failure
        sys.stderr.write(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}\n")
        if pipeline.memory_constrained and args.backend == "sdxl":
            sys.stderr.write("hint: SDXL is heavy here — retry with `--smoke-test -b sd15`.\n")
        sys.exit(1)

    elapsed = time.perf_counter() - start
    out_path = Path(tempfile.mkdtemp()) / f"selftest-{args.backend}.webp"
    result.save(out_path, format="webp", lossless=True)
    sys.stderr.write(
        f"SMOKE TEST PASSED in {elapsed:.0f}s "
        f"(~{elapsed / SMOKE_TEST_STEPS:.1f}s/step); wrote {out_path}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Stylise (content, style) image pairs and write the results. "
                    "Each of -c/-s may be a single image or a directory of images; "
                    "with directory inputs every (content, style) pair is processed. "
                    "For the web UI, run `python -m src.app` instead.",
    )
    parser.add_argument(
        "-c", "--content", default="examples/content",
        help="content image, or directory of content images (default: examples/content)",
    )
    parser.add_argument(
        "-s", "--style", default="examples/style",
        help="style image, or directory of style images (default: examples/style)",
    )
    parser.add_argument(
        "-o", "--output", default="examples/output",
        help="output image path (.png/.jpg/.webp/…) for single-pair, "
             "or output directory for batch (default: examples/output)",
    )
    parser.add_argument(
        "-b", "--backend", default=DEFAULT_BACKEND, choices=sorted(BACKENDS),
        help=f"diffusion backend: 'sdxl' is the flagship (~12 GB VRAM), "
             f"'sd15' is the lightweight option for older / smaller hardware "
             f"(~4 GB VRAM, faster, lower fidelity) (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "-v", "--controlnet-variant", default="depth", choices=["depth", "canny"],
        help="content conditioning signal (default: depth)",
    )
    parser.add_argument(
        "-p", "--preset", default=DEFAULT_PRESET, choices=list(PRESETS),
        help=f"named preset for ip-adapter-weight + controlnet-scale (default: {DEFAULT_PRESET})",
    )
    parser.add_argument("--prompt", default="", help="optional text prompt")
    parser.add_argument(
        "--ip-adapter-weight", type=float, default=None,
        help="override preset's IP-Adapter weight (0.0-1.5)",
    )
    parser.add_argument(
        "--controlnet-scale", type=float, default=None,
        help="override preset's ControlNet conditioning scale (0.0-1.5)",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="max output side in px (default: 1024 for sdxl, 512 for sd15)",
    )
    parser.add_argument("--steps", type=int, default=30, help="diffusion inference steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="seed; -1 = random")
    parser.add_argument(
        "--negative-prompt", default=NEGATIVE_PROMPT_DEFAULT,
        help="negative prompt (default matches the Gradio UI)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="verify the pipeline runs on this machine: one tiny generation on "
             "in-code synthetic images (ignores -c/-s/-o). First run downloads "
             "the chosen backend's weights; pair with '-b sd15' on small hardware.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
