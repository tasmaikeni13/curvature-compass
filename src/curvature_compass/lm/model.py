"""Modern LLaMA transformer decoder with RMSNorm, RoPE, SwiGLU, and bias-free projections."""
from dataclasses import asdict
import math
import numpy as np
import jax
import jax.numpy as jnp
from .config import ModelConfig, digest


def precompute_rope_freqs(head_dim: int, seq_len: int, base: float = 10000.0):
    theta = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, theta)
    emb = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    cos = cos[None, :x.shape[1], None, :].astype(x.dtype)
    sin = sin[None, :x.shape[1], None, :].astype(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class Decoder:
    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config
        d, m = config.width, config.mlp_width
        self.layout = [("token_embedding", (config.padded_vocab_size, d))]
        for i in range(config.layers):
            self.layout.extend((f"block.{i}.{name}", shape) for name, shape in [
                ("input_norm.weight", (d,)),
                ("q_proj.weight", (d, d)),
                ("k_proj.weight", (d, d)),
                ("v_proj.weight", (d, d)),
                ("o_proj.weight", (d, d)),
                ("post_attention_norm.weight", (d,)),
                ("gate_proj.weight", (d, m)),
                ("up_proj.weight", (d, m)),
                ("down_proj.weight", (m, d)),
            ])
        self.layout.extend([("final_norm.weight", (d,))])
        self.offsets, self.parameter_count = [], 0
        for _, shape in self.layout:
            count = math.prod(shape)
            self.offsets.append((self.parameter_count, self.parameter_count + count))
            self.parameter_count += count
        self.fingerprint = digest({"model": asdict(config), "layout": self.layout})

    def initialize(self, seed):
        rng = np.random.default_rng(seed)
        flat = np.empty(self.parameter_count, dtype=np.float32)
        for (name, shape), (start, stop) in zip(self.layout, self.offsets):
            if len(shape) == 1:
                flat[start:stop] = 1.0
            else:
                is_residual = name.endswith(("o_proj.weight", "down_proj.weight"))
                std = .02 / math.sqrt(2 * self.config.layers) if is_residual else .02
                flat[start:stop] = rng.normal(0, std, stop - start).astype(np.float32)
        return flat

    def unpack(self, flat):
        return {name: flat[start:stop].reshape(shape)
                for (name, shape), (start, stop) in zip(self.layout, self.offsets)}

    def decay_mask(self):
        mask = np.zeros(self.parameter_count, dtype=np.float32)
        for (_, shape), (start, stop) in zip(self.layout, self.offsets):
            if len(shape) >= 2:
                mask[start:stop] = 1
        return mask

    def logits(self, flat, tokens, compute_dtype="float32", attention="reference", rematerialize=False):
        p, c = self.unpack(flat), self.config
        if tokens.shape[1] > c.sequence_length:
            raise ValueError("Sequence exceeds model sequence length capacity")
        dtype = jnp.bfloat16 if compute_dtype == "bfloat16" else flat.dtype
        precision = jax.lax.Precision.DEFAULT if compute_dtype == "bfloat16" else jax.lax.Precision.HIGHEST
        head_dim = c.width // c.heads
        cos, sin = precompute_rope_freqs(head_dim, tokens.shape[1], getattr(c, "rope_theta", 10000.0))

        def linear(x, w):
            return jnp.matmul(x.astype(dtype), w.astype(dtype), precision=precision).astype(flat.dtype)

        def rms_norm(x, weight):
            variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            return (x.astype(jnp.float32) * jax.lax.rsqrt(variance + getattr(c, "rms_norm_eps", 1e-5))).astype(flat.dtype) * weight

        def block(x, weights):
            a = rms_norm(x, weights["input_norm.weight"])
            q = linear(a, weights["q_proj.weight"]).reshape(*a.shape[:-1], c.heads, head_dim)
            k = linear(a, weights["k_proj.weight"]).reshape(*a.shape[:-1], c.heads, head_dim)
            v = linear(a, weights["v_proj.weight"]).reshape(*a.shape[:-1], c.heads, head_dim)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if attention in ("flash", "pallas"):
                try:
                    from .attention import flash_attention
                    value = flash_attention(q, k, v, causal=True).astype(flat.dtype)
                except (ImportError, AttributeError):
                    attention_mode = "reference"
                else:
                    attention_mode = attention
            else:
                attention_mode = attention
            if attention_mode == "sdpa":
                with jax.default_matmul_precision("float32" if compute_dtype == "float32" else "bfloat16"):
                    value = jax.nn.dot_product_attention(q.astype(dtype), k.astype(dtype), v.astype(dtype), is_causal=True, implementation="xla").astype(flat.dtype)
            elif attention_mode not in ("flash", "pallas"):
                scores = jnp.einsum("bthd,bshd->bhts", q.astype(dtype), k.astype(dtype), precision=precision).astype(flat.dtype) / math.sqrt(head_dim)
                causal = jnp.arange(tokens.shape[1])[:, None] >= jnp.arange(tokens.shape[1])[None, :]
                scores = jnp.where(causal, scores, -jnp.inf)
                weights_attn = jax.nn.softmax(scores, axis=-1)
                value = jnp.einsum("bhts,bshd->bthd", weights_attn.astype(dtype), v.astype(dtype), precision=precision).astype(flat.dtype)
            x = x + linear(value.reshape(*x.shape), weights["o_proj.weight"])
            a = rms_norm(x, weights["post_attention_norm.weight"])
            gate = linear(a, weights["gate_proj.weight"])
            up = linear(a, weights["up_proj.weight"])
            hidden = jax.nn.silu(gate) * up
            x = x + linear(hidden, weights["down_proj.weight"])
            return x

        x = p["token_embedding"][tokens]
        execute = jax.checkpoint(block) if rematerialize else block
        for i in range(c.layers):
            prefix = f"block.{i}."
            x = execute(x, {name[len(prefix):]: value for name, value in p.items() if name.startswith(prefix)})
        x = rms_norm(x, p["final_norm.weight"])
        logits = linear(x, p["token_embedding"].T)
        return jnp.where(jnp.arange(c.padded_vocab_size) < c.vocab_size, logits, -1e9)

    def loss_sum(self, flat, batch, **execution):
        x, y, mask = batch
        logits = self.logits(flat, x, **execution)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, y[..., None], axis=-1)[..., 0]
        return -jnp.sum(selected * mask), jnp.sum(mask)

    def loss(self, flat, batch, **execution):
        total, count = self.loss_sum(flat, batch, **execution)
        return total / jnp.maximum(count, 1)

    def training_loss(self, flat, batch, accumulation_steps=1, **execution):
        if accumulation_steps == 1:
            return self.loss(flat, batch, **execution)
        micro = tuple(x.reshape(accumulation_steps, x.shape[0] // accumulation_steps, *x.shape[1:]) for x in batch)
        totals, counts = jax.lax.scan(lambda _, b: (None, self.loss_sum(flat, b, **execution)), None, micro)[1]
        return jnp.sum(totals) / jnp.maximum(jnp.sum(counts), 1)
