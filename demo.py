"""
Gradio demo interface for the TinyStories GPT model.
Loads the most recent checkpoint and allows users to generate text from prompts.
"""

import os
import torch
import tiktoken
import gradio as gr
from contextlib import nullcontext

from src.tiny_stories.model import GPT
from src.tiny_stories.config import BabyModelCPUConfig


def load_model(out_dir="out", device=None):
    """Load the most recent model checkpoint."""

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:1"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model on {device}...")

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = BabyModelCPUConfig()
    if "config" in checkpoint:
        # Update config with saved values
        saved_config = checkpoint["config"]
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    model = GPT(config)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Model loaded successfully from {ckpt_path}")
    print(f"Training iteration: {checkpoint.get('iter_num', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")

    return model, device, config


def setup_tokenizer():
    """Setup the tokenizer (using GPT-2 encoding by default)."""
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


model, device, config = load_model()
encode, decode = setup_tokenizer()

device_type = "cuda" if "cuda" in device else "cpu"
dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
ptdtype = dtype_map.get(config.dtype, torch.float32)
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


def generate_text_streaming(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=200,
):
    """Generate text from a prompt using the loaded model, streaming tokens as they're generated."""

    if not prompt:
        yield "Please enter a prompt!"
        return

    start_ids = encode(prompt)
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            for step in range(int(max_new_tokens)):
                # Crop context if needed
                idx_cond = (
                    idx
                    if idx.size(1) <= config.context_window
                    else idx[:, -config.context_window :]
                )

                logits, _ = model(idx_cond)

                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_int = int(top_k)
                    v, _ = torch.topk(logits, min(top_k_int, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, idx_next), dim=1)

                # Check if we've generated the end-of-text token
                if idx_next.item() == config.eot_token_id:
                    generated_text = decode(idx[0].tolist()[:-1])
                    yield generated_text
                    break

                generated_text = decode(idx[0].tolist())
                yield generated_text


with gr.Blocks(title="TinyStories GPT Demo") as demo:
    gr.Markdown(
        """
        # TinyStories GPT Model Demo

        Generate stories and text using a GPT model trained on the TinyStories dataset.
        Enter a prompt and adjust the generation parameters to control the output.
        """
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Once upon a time...",
                lines=3,
                value="Once upon a time",
            )

            with gr.Accordion("Generation Parameters", open=True):
                max_tokens_slider = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max New Tokens",
                    info="Maximum number of tokens to generate",
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, Lower = more deterministic",
                )

                top_k_slider = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Top-k",
                    info="Only sample from top k tokens (0 = disabled)",
                )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text", lines=15, interactive=False
            )

    gr.Markdown(
        """
        ### Tips:
        - Start with simple prompts like "Once upon a time" or "A little girl"
        - Lower temperature (~0.5-0.7) for more coherent stories
        - Higher temperature (~1.0-1.5) for more creative/random text
        - Adjust top-k to control vocabulary diversity
        """
    )

    # Connect the generate button with streaming
    generate_btn.click(
        fn=generate_text_streaming,
        inputs=[
            prompt_input,
            max_tokens_slider,
            temperature_slider,
            top_k_slider,
        ],
        outputs=output_text,
    )

    # Also allow Enter key to generate with streaming
    prompt_input.submit(
        fn=generate_text_streaming,
        inputs=[
            prompt_input,
            max_tokens_slider,
            temperature_slider,
            top_k_slider,
        ],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7870)
