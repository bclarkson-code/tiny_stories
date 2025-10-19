# TinyStories GPT

A GPT model implementation trained on the TinyStories dataset - a collection of simple, child-friendly short stories. This project provides a complete pipeline for training and generating text with a transformer-based language model.

## Overview

This implementation includes:
- GPT model architecture with configurable parameters
- Training pipeline with gradient accumulation and mixed precision support
- Data preparation and tokenization using tiktoken (GPT-2 tokenizer)
- Interactive Gradio demo for text generation
- Support for both CPU and GPU training
- WandB integration for experiment tracking

## Installation

This project requires Python 3.12 or higher and uses `uv` for package management.

```bash
# Clone the repository
git clone https://github.com/bclarkson-code/tiny_stories.git
cd tiny_stories

# Install dependencies using uv
uv sync
```

### Dependencies

The main dependencies include:
- PyTorch (>=2.9.0) - Deep learning framework
- tiktoken (>=0.12.0) - Tokenization
- Gradio (>=5.49.1) - Interactive demo interface
- WandB (>=0.22.2) - Experiment tracking (optional)
- tqdm, numpy, requests

## Data Preparation

Before training, you need to download and tokenize the TinyStories dataset.

```bash
# Download and tokenize both train and validation datasets
uv run python -m src.tiny_stories.data.prepare
```

This script will:
1. Download the TinyStories dataset from HuggingFace (train: ~2.1M stories, valid: ~22K stories)
2. Tokenize the text using the GPT-2 tokenizer
3. Save tokenized data as pickle files in [src/tiny_stories/data/](src/tiny_stories/data/)
   - `train_tokens.pkl` - Training data
   - `valid_tokens.pkl` - Validation data

The tokenization process uses parallel processing to speed up the operation.

## Training

The training script supports three different configurations:

### Training on GPU

For training on a single GPU (recommended):

```bash
# Train from scratch with baby_gpu config (default)
uv run python train.py --config baby_gpu

# Resume training from checkpoint
uv run python train.py --config baby_gpu --resume
```

**BabyModelGPU Configuration:**
- 6 layers, 6 attention heads, 384 embedding dimensions (~8.8M parameters)
- Batch size: 64, Context window: 512 tokens
- Learning rate: 1e-3 with cosine decay
- Max iterations: 10,000
- Device: CUDA
- Mixed precision: bfloat16 (if supported) or float16

### Training on CPU

For training on CPU (slower, mainly for testing):

```bash
# Train from scratch with baby_cpu config
uv run python train.py --config baby_cpu

# Resume training from checkpoint
uv run python train.py --config baby_cpu --resume
```

**BabyModelCPU Configuration:**
- 4 layers, 4 attention heads, 256 embedding dimensions (~3.4M parameters)
- Batch size: 8, Context window: 512 tokens
- Learning rate: 1e-3
- Max iterations: 5,000
- Device: MPS (for Mac) or CPU
- Dtype: float32

### Full GPT-2 Configuration

For training a full GPT-2 sized model (requires significant compute):

```bash
uv run python train.py --config gpt2
```

**GPT-2 Configuration:**
- 12 layers, 12 attention heads, 768 embedding dimensions (~124M parameters)
- Batch size: 12 with gradient accumulation (effective batch size: 480)
- Context window: 1024 tokens
- Max iterations: 600,000

### Training Options

- `--config`: Choose from `baby_gpu` (default), `baby_cpu`, or `gpt2`
- `--resume`: Resume training from the most recent checkpoint in the output directory

### Monitoring Training

Training logs are printed to the console showing:
- Iteration number
- Loss values (train and validation)
- Time per iteration
- TFLOPS (floating point operations per second)

If WandB logging is enabled in the config, metrics will also be logged to Weights & Biases.

Checkpoints are saved to the `out/` directory whenever validation loss improves.

## Running the Demo

After training (or with a pre-trained checkpoint), you can launch the interactive Gradio demo:

```bash
uv run python demo.py
```

The demo will:
1. Load the most recent checkpoint from `out/ckpt.pt`
2. Automatically detect and use available hardware (CUDA > MPS > CPU)
3. Launch a web interface at `http://localhost:7870`

### Demo Features

- **Prompt input**: Enter a starting prompt for story generation
- **Max New Tokens**: Control the length of generated text (10-500 tokens)
- **Temperature**: Adjust randomness (0.1 = deterministic, 2.0 = very random)
- **Top-k**: Limit sampling to top k tokens (helps control coherence)

The interface streams tokens as they're generated, providing real-time feedback.

## Project Structure

```
tiny_stories/
├── src/tiny_stories/
│   ├── model.py           # GPT model architecture
│   ├── config.py          # Training configurations
│   ├── dataset.py         # Data loading utilities
│   ├── tokeniser.py       # Tokenization helpers
│   └── data/
│       └── prepare.py     # Data download and preparation
├── train.py               # Training script
├── demo.py                # Gradio demo interface
├── sample.py              # Command-line sampling script
├── bench.py               # Benchmarking utilities
└── pyproject.toml         # Project dependencies
```

## Model Architecture

The GPT model implements a decoder-only transformer with:
- Multi-head causal self-attention
- Feed-forward networks with GELU activation
- Layer normalization
- Dropout for regularization
- Rotary positional embeddings (optional)

The model is compiled with PyTorch 2.0's `torch.compile()` for improved performance (enabled by default in GPU configs).

## Tips

- Start with the `baby_gpu` config for quick experimentation (~30 minutes on a modern GPU)
- Use lower temperature (0.5-0.7) for more coherent stories in the demo
- Monitor validation loss to prevent overfitting
- The model performs best with simple, child-friendly prompts like "Once upon a time"

## License

This project uses the TinyStories dataset from [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories).

## Acknowledgments

- TinyStories dataset by Ronen Eldan and Yuanzhi Li
- Repo ~stolen from~ inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
