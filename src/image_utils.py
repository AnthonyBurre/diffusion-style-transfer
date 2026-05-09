from PIL import Image, ImageOps


def prepare_content_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Apply EXIF rotation, convert to RGB, scale longest side to ``max_size``,
    and round both dimensions down to a multiple of 8 for the SDXL VAE.

    The returned dimensions drive the diffusion output resolution.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_size / max(w, h))
    w, h = int(w * scale), int(h * scale)
    w, h = w - (w % 8), h - (h % 8)
    return image.resize((w, h), Image.LANCZOS)


def prepare_style_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Apply EXIF rotation, convert to RGB, and scale longest side to ``max_size``.

    The style image only feeds the IP-Adapter's CLIP image encoder, which
    resizes shortest-edge to 224 and center-crops to 224x224. 512 long-side
    leaves ~2x headroom for the center crop on non-square uploads while
    avoiding a wasteful LANCZOS pass over multi-MP source pixels. No
    multiple-of-8 alignment is needed since this image never feeds the VAE.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def output_filename(content_stem: str, style_stem: str) -> str:
    """Output filename convention shared by ``src.app`` (Gradio) and ``src.cli``."""
    return f"sdxl-{content_stem}_X_{style_stem}.webp"
