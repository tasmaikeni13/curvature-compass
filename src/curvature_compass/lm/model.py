"""A small GPT decoder with explicit reference attention and matrix precision."""
from dataclasses import asdict
import math
import numpy as np
import jax
import jax.numpy as jnp
from .config import ModelConfig, digest


class Decoder:
    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config
        d, m = config.width, config.mlp_width
        self.layout = [("token_embedding", (config.padded_vocab_size, d)),
                       ("position_embedding", (config.sequence_length, d))]
        for i in range(config.layers):
            self.layout.extend((f"block.{i}.{name}", shape) for name, shape in [
                ("ln1.weight", (d,)), ("ln1.bias", (d,)),
                ("qkv.weight", (d, 3 * d)), ("qkv.bias", (3 * d,)),
                ("attention.weight", (d, d)), ("attention.bias", (d,)),
                ("ln2.weight", (d,)), ("ln2.bias", (d,)),
                ("mlp_in.weight", (d, m)), ("mlp_in.bias", (m,)),
                ("mlp_out.weight", (m, d)), ("mlp_out.bias", (d,))])
        self.layout.extend([("final_ln.weight", (d,)), ("final_ln.bias", (d,))])
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
            if name.endswith("bias"):
                flat[start:stop] = 0
            elif "ln" in name and name.endswith("weight"):
                flat[start:stop] = 1
            else:
                std = .02 / math.sqrt(2 * self.config.layers) if name.endswith(("attention.weight", "mlp_out.weight")) else .02
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
            raise ValueError("Sequence exceeds position embedding capacity")
        dtype = jnp.bfloat16 if compute_dtype == "bfloat16" else flat.dtype
        precision = jax.lax.Precision.DEFAULT if compute_dtype == "bfloat16" else jax.lax.Precision.HIGHEST

        def linear(x, w, b=None):
            value = jnp.matmul(x.astype(dtype), w.astype(dtype), precision=precision).astype(flat.dtype)
            return value if b is None else value + b

        def norm(x, weight, bias):
            centered = x - jnp.mean(x, axis=-1, keepdims=True)
            return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5) * weight + bias

        def block(x, weights):
            a = norm(x, weights["ln1.weight"], weights["ln1.bias"])
            qkv = linear(a, weights["qkv.weight"], weights["qkv.bias"])
            q, k, v = [z.reshape(*z.shape[:-1], c.heads, c.width // c.heads) for z in jnp.split(qkv, 3, axis=-1)]
            if attention == "sdpa":
                with jax.default_matmul_precision("float32" if compute_dtype == "float32" else "bfloat16"):
                    value = jax.nn.dot_product_attention(q.astype(dtype), k.astype(dtype), v.astype(dtype), is_causal=True, implementation="xla").astype(flat.dtype)
            else:
                scores = jnp.einsum("bthd,bshd->bhts", q.astype(dtype), k.astype(dtype), precision=precision).astype(flat.dtype) / math.sqrt(c.width // c.heads)
                causal = jnp.arange(tokens.shape[1])[:, None] >= jnp.arange(tokens.shape[1])[None, :]
                scores = jnp.where(causal, scores, -jnp.inf)
                weights_attn = jax.nn.softmax(scores, axis=-1)
                value = jnp.einsum("bhts,bshd->bthd", weights_attn.astype(dtype), v.astype(dtype), precision=precision).astype(flat.dtype)
            x = x + linear(value.reshape(*x.shape), weights["attention.weight"], weights["attention.bias"])
            a = norm(x, weights["ln2.weight"], weights["ln2.bias"])
            a = jax.nn.gelu(linear(a, weights["mlp_in.weight"], weights["mlp_in.bias"]), approximate=False)
            return x + linear(a, weights["mlp_out.weight"], weights["mlp_out.bias"])

        x = p["token_embedding"][tokens] + p["position_embedding"][:tokens.shape[1]]
        execute = jax.checkpoint(block) if rematerialize else block
        for i in range(c.layers):
            prefix = f"block.{i}."
            x = execute(x, {name[len(prefix):]: value for name, value in p.items() if name.startswith(prefix)})
        x = norm(x, p["final_ln.weight"], p["final_ln.bias"])
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
