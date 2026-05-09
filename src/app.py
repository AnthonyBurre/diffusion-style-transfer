import warnings

# We only use canny + depth conditioning, dont need mediapipe
warnings.filterwarnings("ignore", message="The module 'mediapipe' is not installed")
# Emitted from inside controlnet_aux, await an upstream release
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
# Internal to the library
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_")

import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from .image_utils import output_filename, prepare_content_image, prepare_style_image
from .pipeline import (
    DEFAULT_PRESET,
    NEGATIVE_PROMPT_DEFAULT,
    PRESETS,
    StylePipeline,
)

PREVIEW_HEIGHT = 320

_pipeline = StylePipeline()


def _apply_preset(name: str) -> tuple[float, float]:
    cfg = PRESETS[name]
    return cfg["ip_adapter_weight"], cfg["controlnet_scale"]


def stylise(
    content_path,
    style_path,
    prompt,
    controlnet_variant,
    ip_adapter_weight,
    controlnet_scale,
    max_size,
    steps,
    guidance_scale,
    seed,
    negative_prompt,
):
    if content_path is None or style_path is None:
        raise gr.Error("Both content and style images are required.")

    content_img = prepare_content_image(Image.open(content_path), max_size=int(max_size))
    style_img = prepare_style_image(Image.open(style_path))

    result = _pipeline.generate(
        content=content_img,
        style=style_img,
        prompt=prompt,
        controlnet_variant=controlnet_variant,
        ip_adapter_weight=ip_adapter_weight,
        controlnet_scale=controlnet_scale,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=int(seed) if seed is not None and int(seed) >= 0 else None,
        negative_prompt=negative_prompt,
    )

    name = output_filename(Path(content_path).stem, Path(style_path).stem)
    out_path = Path(tempfile.mkdtemp()) / name
    result.save(out_path, format="webp", lossless=True)
    return str(out_path)


def build_ui() -> gr.Blocks:
    default = PRESETS[DEFAULT_PRESET]
    with gr.Blocks(title="Diffusion-based style transfer") as demo:
        gr.Markdown("# Diffusion-based artistic style transfer")
        with gr.Row():
            with gr.Column(scale=1):
                content = gr.Image(
                    type="filepath", 
                    label="Content",
                    height=PREVIEW_HEIGHT
                )
            with gr.Column(scale=1):
                style = gr.Image(
                    type="filepath", 
                    label="Style",
                    height=PREVIEW_HEIGHT
                )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="oil painting, ink wash, watercolour, ...",
                )
                controlnet_variant = gr.Radio(
                    choices=["depth", "canny"],
                    value="depth",
                    label="Content conditioning",
                )
                preset = gr.Radio(
                    choices=list(PRESETS),
                    value=DEFAULT_PRESET,
                    label="Preset",
                )
                with gr.Accordion("Advanced", open=False):
                    ip_adapter_weight = gr.Slider(
                        0.0, 1.5, value=default["ip_adapter_weight"], step=0.05,
                        label="IP-Adapter weight",
                    )
                    controlnet_scale = gr.Slider(
                        0.0, 1.5, value=default["controlnet_scale"], step=0.05,
                        label="ControlNet conditioning scale",
                    )
                    max_size = gr.Slider(
                        512, 1024, value=1024, step=64,
                        label="Max output side (px)",
                    )
                    steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")
                    guidance_scale = gr.Slider(
                        1.0, 15.0, value=5.0, step=0.5, label="Guidance scale",
                    )
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    negative_prompt = gr.Textbox(
                        value=NEGATIVE_PROMPT_DEFAULT, label="Negative prompt",
                    )
                run = gr.Button("Stylise", variant="primary")
            with gr.Column(scale=1):
                output = gr.Image(type="filepath", label="Output", interactive=False)

        preset.change(_apply_preset, preset, [ip_adapter_weight, controlnet_scale])
        run.click(
            stylise,
            inputs=[
                content, style, prompt, controlnet_variant,
                ip_adapter_weight, controlnet_scale, max_size,
                steps, guidance_scale, seed, negative_prompt,
            ],
            outputs=output,
        )
    return demo


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
