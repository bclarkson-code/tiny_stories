"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from tiny_stories.config import Config


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        assert config.embedding_dim % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.embedding_dim, config.embedding_dim, bias=config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_heads
        self.n_embd = config.embedding_dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.context_window, config.context_window)
                ).view(1, 1, config.context_window, config.context_window),
            )

    def forward(self, x: torch.Tensor):
        batch_size, sequence_len, embedding_dim = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        key = key.view(
            batch_size, sequence_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)
        query = query.view(
            batch_size, sequence_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :sequence_len, :sequence_len] == 0,  # type: ignore
                float("-inf"),
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(batch_size, sequence_len, embedding_dim)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()  # type: ignore
        self.c_fc = nn.Linear(
            config.embedding_dim, 4 * config.embedding_dim, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.embedding_dim, config.embedding_dim, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()  # type: ignore
        self.ln_1 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()  # type: ignore
        assert config.vocab_size is not None
        assert config.context_window is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.embedding_dim),
                wpe=nn.Embedding(config.context_window, config.embedding_dim),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=LayerNorm(config.embedding_dim, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # type: ignore
        return n_params  # type: ignore

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # type: ignore
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        device = idx.device
        _, n_tokens = idx.size()
        assert n_tokens <= self.config.context_window, (
            f"Cannot forward sequence of length {n_tokens}, block size is only {self.config.context_window}"
        )
        pos = torch.arange(0, n_tokens, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # type: ignore # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # type: ignore # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore
        for block in self.transformer.h:  # type: ignore
            x = block(x)  # type: ignore
        x = self.transformer.ln_f(x)  # type: ignore

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [  # type: ignore
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args: dict[str, bool] = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,  # type: ignore
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_tflops(self, fwdbwd_per_iter: int, dt: float):
        """estimate TFLOPS (trillion floating point operations per second)"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layers,
            cfg.n_heads,
            cfg.embedding_dim // cfg.n_heads,
            cfg.context_window,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # calculate achieved TFLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        tflops = flops_achieved / 1e12  # convert to TFLOPS
        return tflops

    @torch.no_grad()  # type: ignore
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_window
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_window
                else idx[:, -self.config.context_window :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
