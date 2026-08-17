# SPDX-License-Identifier: Apache-2.0
"""gfx1x cached-BF16 fallback for block-scaled W8A8 linear layers.

Adapted from AlexKGwyn/ds4-vllm-public. gfx1151 has no native FP8 matrix
cores, so repeatedly lowering the generic block-scaled FP8 Triton GEMM is a
poor decode path. This module dequantizes each FP8 weight to BF16 once, keeps
that copy for the process lifetime, and routes the GEMM through vLLM's ROCm
unquantized dispatcher (including its skinny-GEMM path).

The caller controls activation through VLLM_GFX1X_W8A8_BF16. The separate
VLLM_GFX1X_W8A8_BF16_DIRECT flag lets the patched linear kernel pass the
original BF16 activation directly, avoiding a lossy FP8 quantize/dequantize
round trip. Both flags are latched by the patched caller before this module is
imported and are enabled only for the DeepSeek V4 gfx1151 TP2 profile. TP1
keeps them disabled because a full-model BF16 duplicate exceeds its memory
headroom.
"""

import os

import torch


_BF16_WEIGHT_CACHE: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
_BF16_PREFILL = os.environ.get("VLLM_GFX1X_W8A8_BF16_PREFILL", "1") != "0"
_MAX_COLD_CACHE_ROWS = 32


def _weight_cache_key(weight: torch.Tensor) -> tuple[int, tuple[int, ...]]:
    return weight.data_ptr(), tuple(weight.shape)


def cached_bf16_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    rows: int,
) -> torch.Tensor | None:
    """Return a cached dequantized BF16 weight, or ``None`` for stock fallback.

    A cold cache is populated only by a small-M decode call. Large-M startup
    profiling therefore measures the original FP8 path and does not allocate
    temporary FP32/BF16 copies while vLLM is sizing its KV cache. Launchers pin
    the DeepSeek KV pool explicitly so the post-profile BF16 cache has room.
    """
    key = _weight_cache_key(weight)
    cached = _BF16_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached if rows <= _MAX_COLD_CACHE_ROWS or _BF16_PREFILL else None
    if rows > _MAX_COLD_CACHE_ROWS:
        return None

    if weight.ndim != 2 or len(block_size) != 2:
        raise ValueError("block-scaled W8A8 expects a 2D weight and two block sizes")

    block_n, block_k = (int(value) for value in block_size)
    out_features, in_features = weight.shape
    if out_features % block_n or in_features % block_k:
        raise ValueError(
            "weight dimensions must be divisible by the block-scaled group shape"
        )

    scale = weight_scale.float()
    expected_scales = (out_features // block_n) * (in_features // block_k)
    if scale.numel() != expected_scales:
        raise ValueError(
            f"expected {expected_scales} W8A8 weight scales, got {scale.numel()}"
        )

    dequantized = weight.float().reshape(
        out_features // block_n,
        block_n,
        in_features // block_k,
        block_k,
    )
    scale = scale.reshape(
        out_features // block_n,
        1,
        in_features // block_k,
        1,
    )
    cached = (dequantized * scale).reshape(out_features, in_features)
    cached = cached.to(torch.bfloat16).contiguous()
    _BF16_WEIGHT_CACHE[key] = cached
    return cached


def w8a8_block_bf16_direct(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor | None:
    """Run a block-scaled linear directly from the original BF16 activation."""
    rows = activation.numel() // activation.shape[-1]
    bf16_weight = cached_bf16_weight(weight, weight_scale, block_size, rows)
    if bf16_weight is None:
        return None

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    return rocm_unquantized_gemm_impl(activation, bf16_weight)


def w8a8_block_fp8_bf16(
    quantized_activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Fallback BF16 GEMM when the caller has already quantized activation."""
    rows, in_features = quantized_activation.shape
    bf16_weight = cached_bf16_weight(weight, weight_scale, block_size, rows)
    if bf16_weight is None:
        return None

    block_k = int(block_size[1])
    activation = quantized_activation.float().reshape(
        rows, in_features // block_k, block_k
    )
    scale = activation_scale.float().reshape(rows, in_features // block_k, 1)
    activation = (activation * scale).reshape(rows, in_features)
    activation = activation.to(torch.bfloat16)

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    return rocm_unquantized_gemm_impl(activation, bf16_weight).to(output_dtype)
