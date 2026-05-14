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
from pathlib import Path

from PIL import Image

from .image_utils import output_filename, prepare_content_image, prepare_style_image
from .pipeline import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_PRESET,
    NEGATIVE_PROMPT_DEFAULT,
    PRESETS,
    StylePipeline,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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

    # Prepared once: each style is reused across every content image.
    prepared_styles = [
        (p, prepare_style_image(Image.open(p))) for p in styles
    ]

    i = 0
    for content_path in contents:
        content_img = prepare_content_image(Image.open(content_path), max_size=max_size)
        for style_path, style_img in prepared_styles:
            i += 1
            sys.stderr.write(f"[{i}/{total}] {content_path.stem} × {style_path.stem}\n")
            sys.stderr.flush()

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

            out_path = out if is_file_output else out / output_filename(
                content_path.stem, style_path.stem, backend=args.backend
            )
            _save(result, out_path)
            print(str(out_path.resolve()))


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
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
