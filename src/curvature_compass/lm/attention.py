"""High-performance FlashAttention subsystem targeting TPU v4 via Pallas with CPU fallback.

Exposes:
- `flash_attention`: Unified public API with automatic backend detection.
- `pallas_flash_attention`: Explicit Pallas FlashAttention kernel with safe CPU fallback.
- `reference_attention`: Reference scaled dot-product attention via standard JAX/XLA.
- `is_tpu_available`: Diagnostic check for active TPU hardware.
- `is_pallas_tpu_available`: Diagnostic check for Pallas TPU kernel availability.
- `pad_sequence_to_tile`: Utility to pad sequence lengths to TPU block size 128.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp

__all__ = [
    "flash_attention",
    "pallas_flash_attention",
    "reference_attention",
    "is_tpu_available",
    "is_pallas_tpu_available",
    "pad_sequence_to_tile",
]


def is_tpu_available() -> bool:
    """Return True if JAX has initialized with an active TPU backend."""
    try:
        if jax.default_backend() != "tpu":
            return False
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


def is_pallas_tpu_available() -> bool:
    """Return True if TPU backend is available AND Pallas TPU flash_attention ops are importable."""
    if not is_tpu_available():
        return False
    try:
        from jax.experimental.pallas.ops.tpu import flash_attention as _pallas_fa  # noqa: F401
        return True
    except Exception:
        return False


def pad_sequence_to_tile(x: jnp.ndarray, seq_axis: int = 1, tile_size: int = 128) -> Tuple[jnp.ndarray, int]:
    """Pad tensor `x` along `seq_axis` to the next multiple of `tile_size` (default 128).
    
    Returns:
        padded_x: Tensor padded with zeros along `seq_axis`.
        pad_len: Number of padded elements added.
    """
    seq_len = x.shape[seq_axis]
    rem = seq_len % tile_size
    if rem == 0:
        return x, 0
    pad_len = tile_size - rem
    pad_width = [(0, 0)] * x.ndim
    pad_width[seq_axis] = (0, pad_len)
    return jnp.pad(x, pad_width), pad_len


def reference_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    layout: str = "BTHD",
) -> jnp.ndarray:
    """Reference scaled dot-product attention using `jax.nn.dot_product_attention`.
    
    Args:
        q, k, v: Input query, key, value tensors.
        causal: Whether to apply causal autoregressive masking.
        sm_scale: Softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        layout: 'BTHD' (batch, seq, heads, dim) or 'BHTD' (batch, heads, seq, dim).
    
    Returns:
        Attention output tensor matching the query shape and layout.
    """
    if layout == "BHTD":
        q_in = jnp.swapaxes(q, 1, 2)
        k_in = jnp.swapaxes(k, 1, 2)
        v_in = jnp.swapaxes(v, 1, 2)
    elif layout == "BTHD":
        q_in, k_in, v_in = q, k, v
    else:
        raise ValueError(f"Unsupported layout: {layout}. Expected 'BTHD' or 'BHTD'.")

    head_dim = q_in.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = jax.nn.dot_product_attention(
        q_in,
        k_in,
        v_in,
        is_causal=causal,
        scale=sm_scale,
        implementation="xla",
    )

    if layout == "BHTD":
        out = jnp.swapaxes(out, 1, 2)
    return out


def pallas_flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    block_sizes: Any = None,
    fallback_on_cpu: bool = True,
    layout: str = "BTHD",
    mesh: Optional[Any] = None,
) -> jnp.ndarray:
    """Execute high-performance FlashAttention targeting TPU v4 via Pallas.
    
    When running under a CPU runtime or when Pallas TPU lowering is unsupported,
    transparently falls back to `reference_attention` if `fallback_on_cpu=True`.
    
    Args:
        q, k, v: Input tensors (BTHD or BHTD).
        causal: Whether to apply causal autoregressive masking.
        sm_scale: Softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        block_sizes: Optional Pallas `BlockSizes` instance.
        fallback_on_cpu: If True, fall back to reference attention on CPU. If False, raise RuntimeError.
        layout: 'BTHD' or 'BHTD'.
        mesh: Optional SPMD device mesh for shard_map wrapping.
    
    Returns:
        Attention output tensor matching the input shape and layout.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if not is_pallas_tpu_available():
        if fallback_on_cpu:
            return reference_attention(q, k, v, causal=causal, sm_scale=sm_scale, layout=layout)
        raise RuntimeError("Pallas FlashAttention kernel requires TPU backend with Pallas support.")

    try:
        from jax.experimental.pallas.ops.tpu import flash_attention as pallas_fa

        if layout == "BTHD":
            seq_len = q.shape[1]
            q_pad, pad_len = pad_sequence_to_tile(q, seq_axis=1, tile_size=128)
            k_pad, _ = pad_sequence_to_tile(k, seq_axis=1, tile_size=128)
            v_pad, _ = pad_sequence_to_tile(v, seq_axis=1, tile_size=128)
            # Transpose from (B, T, H, D) to (B, H, T, D) for Pallas kernel
            q_k = jnp.swapaxes(q_pad, 1, 2)
            k_k = jnp.swapaxes(k_pad, 1, 2)
            v_k = jnp.swapaxes(v_pad, 1, 2)
        elif layout == "BHTD":
            seq_len = q.shape[2]
            q_pad, pad_len = pad_sequence_to_tile(q, seq_axis=2, tile_size=128)
            k_pad, _ = pad_sequence_to_tile(k, seq_axis=2, tile_size=128)
            v_pad, _ = pad_sequence_to_tile(v, seq_axis=2, tile_size=128)
            q_k, k_k, v_k = q_pad, k_pad, v_pad
        else:
            raise ValueError(f"Unsupported layout: {layout}. Expected 'BTHD' or 'BHTD'.")

        if mesh is not None:
            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P

            @functools.partial(
                shard_map,
                mesh=mesh,
                in_specs=(P("data", None, None, None), P("data", None, None, None), P("data", None, None, None)),
                out_specs=P("data", None, None, None),
                check_rep=False,
            )
            def _sharded_fa(q_in, k_in, v_in):
                return pallas_fa.flash_attention(
                    q_in,
                    k_in,
                    v_in,
                    causal=causal,
                    sm_scale=sm_scale,
                    block_sizes=block_sizes,
                )

            out_k = _sharded_fa(q_k, k_k, v_k)
        else:
            out_k = pallas_fa.flash_attention(
                q_k,
                k_k,
                v_k,
                causal=causal,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
            )

        if layout == "BTHD":
            out = jnp.swapaxes(out_k, 1, 2)
            if pad_len > 0:
                out = out[:, :seq_len, :, :]
        else:
            out = out_k
            if pad_len > 0:
                out = out[:, :, :seq_len, :]

        return out
    except Exception:
        if fallback_on_cpu:
            return reference_attention(q, k, v, causal=causal, sm_scale=sm_scale, layout=layout)
        raise


def flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    layout: str = "BTHD",
    mesh: Optional[Any] = None,
) -> jnp.ndarray:
    """Public unified FlashAttention entrypoint with automatic backend detection.
    
    Dispatches to Pallas TPU kernel on TPU devices and transparently falls back
    to `jax.nn.dot_product_attention` on CPU.
    """
    return pallas_flash_attention(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        fallback_on_cpu=True,
        layout=layout,
        mesh=mesh,
    )
