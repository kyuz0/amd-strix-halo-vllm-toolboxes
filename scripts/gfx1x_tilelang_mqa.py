# SPDX-License-Identifier: Apache-2.0
"""TileLang sparse-indexer MQA kernels for DeepSeek V4 on ROCm gfx1x.

Adapted from AlexKGwyn/ds4-vllm-public's ``ds4_tl_indexer.py``:
https://github.com/AlexKGwyn/ds4-vllm-public

The TileLang matrix multiply operates on BF16 because gfx1151 lacks native FP8
matrix-core dot products. FP8 remains the storage format. Unlike the source
implementation, the paged path explicitly converts vLLM's 16x16 shuffled ROCm
cache pages back to logical token-major rows before invoking TileLang.
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


_PREFILL_KERNELS = {}
_DECODE_KERNELS = {}
_CONTEXT_LENS_TENSOR = None
_CONTEXT_LENS_VERSION = None
_CONTEXT_LENS_HOST = None

_KV_BUCKET = 512
_PREFILL_KV_BUCKET = 8192
_DECODE_BLOCK_N = 64
_DECODE_THREADS = 256
_CACHE_TILE = 16


def _build_prefill(
    seq_len,
    seq_len_kv,
    heads,
    index_dim,
    block_n=128,
    num_stages=2,
    threads=256,
):
    dtype, accum, index_type = "bfloat16", "float32", "int32"

    @T.prim_func
    def main(
        index_q: T.Tensor((seq_len * heads, index_dim), dtype),
        index_k: T.Tensor((seq_len_kv, index_dim), dtype),
        index_k_scale: T.Tensor((seq_len_kv,), accum),
        weights: T.Tensor((seq_len, heads), accum),
        cu_seq_len_ks: T.Tensor((seq_len,), index_type),
        cu_seq_len_ke: T.Tensor((seq_len,), index_type),
        logits: T.Tensor((seq_len, seq_len_kv), accum),
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            iq = T.alloc_shared([heads, index_dim], dtype)
            ik = T.alloc_shared([block_n, index_dim], dtype)
            k_scale = T.alloc_fragment([block_n], accum)
            scores = T.alloc_fragment([block_n, heads], accum)
            reduced = T.alloc_fragment([block_n], accum)
            weight = T.alloc_fragment([heads], accum)
            k_min = T.alloc_var(index_type)
            k_max = T.alloc_var(index_type)
            k_min = T.min(cu_seq_len_ks[bx], seq_len_kv)
            k_max = T.min(cu_seq_len_ke[bx], seq_len_kv)
            T.copy(index_q[bx * heads, 0], iq)
            T.copy(weights[bx, 0], weight)
            for nb in T.Pipelined(
                T.ceildiv(k_max - k_min, block_n), num_stages=num_stages
            ):
                T.copy(index_k[k_min + nb * block_n, 0], ik)
                T.copy(index_k_scale[k_min + nb * block_n], k_scale)
                T.gemm(
                    ik,
                    iq,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bn, head in T.Parallel(block_n, heads):
                    scores[bn, head] = (
                        T.max(scores[bn, head], 0)
                        * weight[head]
                        * k_scale[bn]
                    )
                T.reduce_sum(scores, reduced, dim=-1, clear=True)
                for bn in T.Parallel(block_n):
                    logits[bx, k_min + nb * block_n + bn] = reduced[bn]

    return main


def _fit_prefill_config(heads, index_dim, budget=56 * 1024):
    remaining = budget - heads * index_dim * 2
    for num_stages in (2, 1):
        for block_n in (128, 64, 32, 16):
            if block_n * index_dim * 2 * num_stages <= remaining:
                return block_n, num_stages
    return 16, 1


def _prefill_kernel(seq_len, seq_len_kv, heads, index_dim):
    block_n, num_stages = _fit_prefill_config(heads, index_dim)
    key = (seq_len, seq_len_kv, heads, index_dim, block_n, num_stages)
    if key not in _PREFILL_KERNELS:
        _PREFILL_KERNELS[key] = tilelang.compile(
            _build_prefill(
                seq_len,
                seq_len_kv,
                heads,
                index_dim,
                block_n=block_n,
                num_stages=num_stages,
            ),
            out_idx=[6],
        )
    return _PREFILL_KERNELS[key]


def fp8_mqa_logits_tilelang(
    q,
    k_fp8,
    scale,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
):
    """Compute non-paged sparse-indexer logits with a TileLang BF16 GEMM."""
    num_queries, num_heads, head_dim = q.shape
    num_keys = k_fp8.shape[0]
    bucketed_keys = (
        (num_keys + _PREFILL_KV_BUCKET - 1) // _PREFILL_KV_BUCKET
    ) * _PREFILL_KV_BUCKET

    q_bf16 = q.to(torch.bfloat16).reshape(num_queries * num_heads, head_dim)
    k_bf16 = k_fp8.to(torch.bfloat16).contiguous()
    scales = scale.reshape(-1).float().contiguous()
    if bucketed_keys != num_keys:
        k_bf16 = F.pad(k_bf16, (0, 0, 0, bucketed_keys - num_keys))
        scales = F.pad(scales, (0, bucketed_keys - num_keys))

    logits = _prefill_kernel(
        num_queries, bucketed_keys, num_heads, head_dim
    )(
        q_bf16.contiguous(),
        k_bf16,
        scales,
        weights.float().contiguous(),
        cu_seqlen_ks.int().contiguous(),
        cu_seqlen_ke.int().contiguous(),
    )[:, :num_keys]
    key_indices = torch.arange(num_keys, device=logits.device)[None, :]
    valid = (key_indices >= cu_seqlen_ks[:, None]) & (
        key_indices < cu_seqlen_ke[:, None]
    )
    return logits.masked_fill(~valid, float("-inf"))


def _build_decode(
    seq_len_kv,
    heads,
    index_dim,
    block_n=_DECODE_BLOCK_N,
    threads=_DECODE_THREADS,
):
    dtype, accum = "bfloat16", "float32"

    @T.prim_func
    def main(
        index_q: T.Tensor((heads, index_dim), dtype),
        index_k: T.Tensor((seq_len_kv, index_dim), dtype),
        index_k_scale: T.Tensor((seq_len_kv,), accum),
        weights: T.Tensor((heads,), accum),
        logits: T.Tensor((seq_len_kv,), accum),
    ):
        with T.Kernel(T.ceildiv(seq_len_kv, block_n), threads=threads) as bx:
            iq = T.alloc_shared([heads, index_dim], dtype)
            ik = T.alloc_shared([block_n, index_dim], dtype)
            k_scale = T.alloc_fragment([block_n], accum)
            scores = T.alloc_fragment([block_n, heads], accum)
            reduced = T.alloc_fragment([block_n], accum)
            weight = T.alloc_fragment([heads], accum)
            base = bx * block_n
            T.copy(index_q[0, 0], iq)
            T.copy(weights[0], weight)
            T.copy(index_k[base, 0], ik)
            T.copy(index_k_scale[base], k_scale)
            T.gemm(
                ik,
                iq,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for bn, head in T.Parallel(block_n, heads):
                scores[bn, head] = (
                    T.max(scores[bn, head], 0)
                    * weight[head]
                    * k_scale[bn]
                )
            T.reduce_sum(scores, reduced, dim=-1, clear=True)
            for bn in T.Parallel(block_n):
                logits[base + bn] = reduced[bn]

    return main


def _decode_kernel(seq_len_kv, heads, index_dim):
    key = (seq_len_kv, heads, index_dim)
    if key not in _DECODE_KERNELS:
        _DECODE_KERNELS[key] = tilelang.compile(
            _build_decode(seq_len_kv, heads, index_dim), out_idx=[4]
        )
    return _DECODE_KERNELS[key]


def _context_lens_host(context_lens):
    """Copy shared scheduler context lengths to the host once per mutation."""
    global _CONTEXT_LENS_TENSOR, _CONTEXT_LENS_VERSION, _CONTEXT_LENS_HOST
    if (
        context_lens is _CONTEXT_LENS_TENSOR
        and context_lens._version == _CONTEXT_LENS_VERSION
    ):
        return _CONTEXT_LENS_HOST
    host = context_lens.reshape(-1).tolist()
    _CONTEXT_LENS_TENSOR = context_lens
    _CONTEXT_LENS_VERSION = context_lens._version
    _CONTEXT_LENS_HOST = host
    return host


def _logical_cache_values(cache, num_pages, block_size, head_dim, fp8_dtype):
    """Undo vLLM's 16x16 ROCm cache shuffle into token-major rows."""
    values = cache[..., : block_size * head_dim].view(dtype=fp8_dtype)
    if block_size == 1:
        return values.reshape(num_pages, block_size, head_dim)
    if block_size % _CACHE_TILE or head_dim % _CACHE_TILE:
        raise ValueError(
            "gfx1x TileLang indexer requires cache block size and head "
            f"dimension divisible by {_CACHE_TILE}; got {block_size}x{head_dim}"
        )
    return (
        values.view(
            num_pages,
            block_size // _CACHE_TILE,
            head_dim // _CACHE_TILE,
            _CACHE_TILE,
            _CACHE_TILE,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(num_pages, block_size, head_dim)
    )


def fp8_paged_mqa_logits_tilelang(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    max_model_len,
):
    """Compute paged sparse-indexer logits with a TileLang BF16 GEMM."""
    from vllm.platforms import current_platform
    from vllm.utils.math_utils import cdiv

    batch_size, next_n, num_heads, head_dim = q.shape
    block_size = kv_cache.shape[1]
    context_lengths = _context_lens_host(context_lens)
    if len(context_lengths) < batch_size:
        raise ValueError(
            f"received {len(context_lengths)} context lengths for {batch_size} rows"
        )
    if max(context_lengths[:batch_size], default=0) > max_model_len:
        raise ValueError("context length exceeds max_model_len")

    bucketed_lengths = [
        ((length + _KV_BUCKET - 1) // _KV_BUCKET) * _KV_BUCKET
        for length in context_lengths[:batch_size]
    ]
    width = max(max(bucketed_lengths, default=0), _KV_BUCKET)
    rows = batch_size * next_n
    logits = torch.full(
        (rows, width),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    q_bf16 = q.to(torch.bfloat16)
    flat_cache = kv_cache.view(-1, block_size * (head_dim + 4))
    scale_offset = block_size * head_dim
    fp8_dtype = current_platform.fp8_dtype()

    for batch in range(batch_size):
        seq_len = context_lengths[batch]
        if seq_len <= 0:
            continue
        num_pages = cdiv(seq_len, block_size)
        if num_pages > block_tables.shape[1]:
            raise ValueError("context length exceeds the supplied block table")
        physical_pages = block_tables[batch, :num_pages]
        cache = flat_cache[physical_pages]
        values = _logical_cache_values(
            cache, num_pages, block_size, head_dim, fp8_dtype
        ).reshape(num_pages * block_size, head_dim)[:seq_len]
        scales = (
            cache[..., scale_offset:]
            .view(dtype=torch.float32)
            .reshape(num_pages * block_size)[:seq_len]
        )
        values = values.to(torch.bfloat16).contiguous()
        scales = scales.contiguous()
        bucketed_len = bucketed_lengths[batch]
        if bucketed_len != seq_len:
            values = F.pad(values, (0, 0, 0, bucketed_len - seq_len))
            scales = F.pad(scales, (0, bucketed_len - seq_len))
        kernel = _decode_kernel(bucketed_len, num_heads, head_dim)

        for speculative_index in range(next_n):
            row = batch * next_n + speculative_index
            row_logits = kernel(
                q_bf16[batch, speculative_index].reshape(
                    num_heads, head_dim
                ).contiguous(),
                values,
                scales,
                weights[row].float().reshape(num_heads).contiguous(),
            )
            causal_end = seq_len - next_n + speculative_index + 1
            if causal_end > 0:
                logits[row, :causal_end] = row_logits[:causal_end]
    return logits
