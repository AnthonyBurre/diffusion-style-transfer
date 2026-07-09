"""Fast, no-download unit tests for the image preprocessing helpers."""
from PIL import Image

from src.image_utils import (
    output_filename,
    prepare_content_image,
    prepare_style_image,
)


def _img(w, h, mode="RGB", colour=(120, 120, 120)):
    return Image.new(mode, (w, h), colour)


def test_content_downscales_longest_side_to_max():
    out = prepare_content_image(_img(2000, 1000), max_size=1024)
    assert max(out.size) <= 1024


def test_content_dims_are_multiples_of_8():
    # Multiple-of-8 alignment is required for the diffusion VAE.
    w, h = prepare_content_image(_img(1003, 777), max_size=1024).size
    assert w % 8 == 0 and h % 8 == 0


def test_content_does_not_upscale_small_images():
    # 100x80 is already under max; only the round-down-to-8 applies (100 -> 96).
    out = prepare_content_image(_img(100, 80), max_size=1024)
    assert out.size == (96, 80)


def test_content_converts_to_rgb():
    out = prepare_content_image(_img(64, 64, mode="L", colour=128), max_size=64)
    assert out.mode == "RGB"


def test_style_caps_longest_side():
    out = prepare_style_image(_img(2000, 1500), max_size=512)
    assert max(out.size) == 512


def test_style_leaves_small_image_unchanged():
    out = prepare_style_image(_img(300, 200), max_size=512)
    assert out.size == (300, 200)


def test_style_converts_to_rgb():
    assert prepare_style_image(_img(64, 64, mode="L", colour=128)).mode == "RGB"


def test_output_filename_convention():
    assert output_filename("katy", "ty", backend="sd15") == "sd15-katy_X_ty.webp"
    assert output_filename("a", "b", backend="sdxl") == "sdxl-a_X_b.webp"
