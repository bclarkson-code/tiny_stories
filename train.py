"""
This training script runs on a single GPU.

To run:
$ python train.py --config baby_gpu --resume
$ python train.py --config gpt2
"""

import os
import time
import math
import itertools
import argparse
from contextlib import nullcontext

import torch

from tiny_stories.model import GPT
from tiny_stories.data.prepare import Split
from tiny_stories.config import load_config, ConfigType
from tiny_stories.dataset import load_dataloaders
import logging

log_config = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(log_config)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train GPT model on TinyStories dataset")
parser.add_argument(
    "--config",
    type=str,
    default="baby_gpu",
    choices=["gpt2", "baby_cpu", "baby_gpu"],
    help="Configuration to use (default: baby_gpu)"
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from checkpoint"
)
args = parser.parse_args()

config_type = ConfigType(args.config)
config = load_config(config_type)

# Override init_from based on resume flag
if args.resume:
    config.init_from = "resume"
    logger.info(f"Resume flag set - will attempt to resume training from {config.out_dir}")
else:
    config.init_from = "scratch"
    logger.info("Starting training from scratch")

# various inits, derived attributes, I/O setup
device = config.device
tokens_per_iter = (
    config.gradient_accumulation_steps * config.batch_size * config.context_window
)
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
logger.info(f"Using device: {device}")
device = torch.device(device)

# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[config.dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
logger.info("Loading data...")
train_dl, valid_dl = load_dataloaders(config)

step = 0
best_val_loss = 1e9

# are we starting from scratch or resuming
if config.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = GPT(config)
elif config.init_from == "resume":
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    model = GPT(config)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)

    step = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type
)
if config.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for dl, split in zip([train_dl, valid_dl], [Split.TRAIN, Split.VALID]):
        losses = torch.zeros(config.eval_iters)
        eval_step = 0
        for batch in dl:
            if eval_step >= config.eval_iters:
                break
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with ctx:
                _, loss = model(inputs, labels)
            losses[eval_step] = loss.item()
            eval_step += 1
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# logging
if config.wandb_log:
    import wandb
    wandb.init(project=config.wandb_project, config=config)

# training loop
if config.use_inifite_dataloader:
    # loop over our dataset forever
    train_dl = itertools.cycle(train_dl)

train_dl = iter(train_dl)
batch = next(train_dl)  # fetch the very first batch
inputs = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

start_time = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model
running_tflops = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(step) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if step % config.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses[Split.TRAIN]:.4f}, val loss {losses[Split.VALID]:.4f}"
        )
        if config.wandb_log:
            wandb.log(
                {
                    "iter": step,
                    "train/loss": losses[Split.TRAIN],
                    "val/loss": losses[Split.VALID],
                    "lr": lr,
                    "tflops": running_tflops,
                },
                step=step,
            )
        if losses[Split.VALID] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses[Split.VALID]
            if step > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": step,
                    "best_val_loss": best_val_loss,
                    "config": config.to_dict(),
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))
    if step == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(inputs, labels)
            loss = (
                loss / config.gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        batch = next(train_dl)
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    end_time = time.time()
    difference = end_time - start_time
    start_time = end_time
    if step % config.log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        step_loss = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            tflops = raw_model.estimate_tflops(
                config.batch_size * config.gradient_accumulation_steps, difference
            )
            running_tflops = tflops if running_tflops == -1.0 else 0.9 * running_tflops + 0.1 * tflops
        wandb.log({'train/step_loss': step_loss, "train/step_time":difference, "train/tflops": running_tflops}, step=step)
        logging.info(
            f"iter {step}: loss {step_loss:.4f}, time {difference*1000:.2f}ms, tflops {running_tflops:.2f}"
        )
    step += 1
    local_iter_num += 1

    # termination conditions
    if step > config.max_iters:
        break
