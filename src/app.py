import warnings

# We only use canny + depth conditioning, dont need mediapipe
warnings.filterwarnings("ignore", message="The module 'mediapipe' is not installed")
# Emitted from inside controlnet_aux, await an upstream release
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
# Internal to the library
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_")

import gradio as gr

from .image_utils import prepare_content_image, prepare_style_image
from .pipeline import NEGATIVE_PROMPT_DEFAULT, StylePipeline

PRESETS: dict[str, dict[str, float]] = {
    "follow content closely": {"ip_adapter_weight": 0.5, "controlnet_scale": 0.85},
    "balanced":               {"ip_adapter_weight": 0.8, "controlnet_scale": 0.6},
    "maximum style":          {"ip_adapter_weight": 1.1, "controlnet_scale": 0.35},
}
DEFAULT_PRESET = "balanced"

_pipeline = StylePipeline()


def _apply_preset(name: str) -> tuple[float, float]:
    cfg = PRESETS[name]
    return cfg["ip_adapter_weight"], cfg["controlnet_scale"]


def stylise(
    content,
    style,
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
    if content is None or style is None:
        raise gr.Error("Both content and style images are required.")

    content_img = prepare_content_image(content, max_size=int(max_size))
    style_img = prepare_style_image(style)

    return _pipeline.generate(
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


def build_ui() -> gr.Blocks:
    default = PRESETS[DEFAULT_PRESET]
    with gr.Blocks(title="Diffusion-based style transfer") as demo:
        gr.Markdown("# Diffusion-based artistic style transfer")
        with gr.Row():
            with gr.Column():
                content = gr.Image(type="pil", label="Content")
                style = gr.Image(type="pil", label="Style")
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
            with gr.Column():
                output = gr.Image(type="pil", label="Output")

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
