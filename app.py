import gradio as gr

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 🗑️ Mülltrenner9000 🤖")
        gr.Markdown("🗑️ This app classifies and segments trash for proper recycling. 🤖")


        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Segmented Output")
        # input_image.change(fn=predict, inputs=input_image, outputs=output_image)
    demo.launch()


if __name__ == "__main__":
    main()
