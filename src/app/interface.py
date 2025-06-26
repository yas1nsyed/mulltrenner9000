import gradio as gr
from ..process.process import display_result
import os


def main_interface():
    with gr.Blocks(title = "🗑️ Mülltrenner9000 🤖") as demo:
        gr.Markdown("# 🗑️ Mülltrenner9000 🤖")
        gr.Markdown("🗑️ This app classifies and segments trash for proper recycling according to german mülltrennung rules. 🤖")


        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Segmented Output")
        input_image.change(fn=display_result, inputs=input_image, outputs=output_image)
    return demo