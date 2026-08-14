"""Portable DeepSeek V4 FP8 kernels for ROCm gfx1x.

gfx1151 has no native FP8 matrix-core dot product.  Letting Triton or AITER
lower a raw-FP8 ``tl.dot`` either fails or produces pathological LLVM code.
These build-time patches retain FP8 storage but convert operands to BF16 for
the dot product on gfx1x only.
"""

from pathlib import Path


SPARSE_MARKER = "# PATCHED: gfx1x portable sparse-indexer MQA"
LINEAR_MARKER = "# PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot"


PAGED_CODE = r'''

# PATCHED: gfx1x portable sparse-indexer MQA
@triton.jit
def _gfx1x_portable_fp8_paged_mqa_kernel(
    q_ptr,
    kv_ptr,
    kv_scale_ptr,
    weights_ptr,
    block_tables_ptr,
    query_offsets_ptr,
    context_limits_ptr,
    out_ptr,
    kv_row_stride,
    kv_scale_row_stride,
    block_table_row_stride,
    out_row_stride,
    num_blocks,
    max_model_len,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    SHUFFLED: tl.constexpr,
):
    """BF16-dot paged MQA with the layout used by vLLM's cache writer."""
    row = tl.program_id(0)
    logical_page = tl.program_id(1)
    query_offset = tl.load(query_offsets_ptr + row)
    if logical_page * BLOCK_SIZE > query_offset:
        return

    batch = row // NEXT_N
    physical_page = tl.load(
        block_tables_ptr + batch * block_table_row_stride + logical_page
    )
    physical_page = tl.minimum(tl.maximum(physical_page, 0), num_blocks - 1)

    token = tl.arange(0, BLOCK_SIZE)
    dim = tl.arange(0, HEAD_DIM)
    head = tl.arange(0, NUM_HEADS)
    position = logical_page * BLOCK_SIZE + token
    context_limit = tl.load(context_limits_ptr + row)
    valid = (position < context_limit) & (position <= query_offset)

    query = tl.load(
        q_ptr
        + row * NUM_HEADS * HEAD_DIM
        + head[:, None] * HEAD_DIM
        + dim[None, :]
    ).to(tl.bfloat16)

    if SHUFFLED:
        cache_offsets = (
            physical_page * kv_row_stride
            + (token[:, None] // BLOCK_TILE_SIZE)
            * BLOCK_TILE_SIZE
            * HEAD_DIM
            + (token[:, None] % BLOCK_TILE_SIZE) * HEAD_TILE_SIZE
            + (dim[None, :] // HEAD_TILE_SIZE)
            * BLOCK_TILE_SIZE
            * HEAD_TILE_SIZE
            + (dim[None, :] % HEAD_TILE_SIZE)
        )
    else:
        cache_offsets = (
            physical_page * kv_row_stride
            + token[:, None] * HEAD_DIM
            + dim[None, :]
        )
    key = tl.load(kv_ptr + cache_offsets).to(tl.bfloat16)

    scores = tl.maximum(tl.dot(key, tl.trans(query)), 0.0)
    weights = tl.load(weights_ptr + row * NUM_HEADS + head).to(tl.float32)
    logits = tl.sum(scores * weights[None, :], axis=1)
    scale = tl.load(kv_scale_ptr + physical_page * kv_scale_row_stride + token)
    logits *= scale
    logits = tl.where(valid, logits, float("-inf"))
    tl.store(
        out_ptr + row * out_row_stride + position,
        logits,
        mask=position < max_model_len,
    )


def gfx1x_portable_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Portable gfx1x paged sparse-indexer logits with shuffled-cache support."""
    from vllm.utils.math_utils import cdiv

    batch_size, next_n, num_heads, head_dim = q.shape
    num_blocks, block_size = kv_cache.shape[:2]
    cache_flat = kv_cache.view(num_blocks, block_size * (head_dim + 4))
    cache_values = cache_flat[:, : block_size * head_dim].view(
        current_platform.fp8_dtype()
    )
    cache_scales = cache_flat[:, block_size * head_dim :].view(torch.float32)

    q_bf16 = q.to(torch.bfloat16).reshape(-1, num_heads, head_dim).contiguous()
    weights_f32 = weights.to(torch.float32).reshape(-1, num_heads).contiguous()

    context_lens_i32 = context_lens.to(device=q.device, dtype=torch.int32)
    if context_lens_i32.dim() == 1:
        context_limits = context_lens_i32[:, None].expand(batch_size, next_n)
        query_offsets = context_lens_i32[:, None] - next_n + torch.arange(
            next_n, device=q.device, dtype=torch.int32
        )[None, :]
    else:
        context_limits = context_lens_i32.reshape(batch_size, next_n)
        query_offsets = context_limits - 1
    context_limits = context_limits.reshape(-1).contiguous()
    query_offsets = query_offsets.reshape(-1).contiguous()

    num_logical_pages = min(
        cdiv(max_model_len, block_size), block_tables.shape[1]
    )
    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    _gfx1x_portable_fp8_paged_mqa_kernel[
        (batch_size * next_n, num_logical_pages)
    ](
        q_bf16,
        cache_values,
        cache_scales,
        weights_f32,
        block_tables,
        query_offsets,
        context_limits,
        out,
        cache_values.stride(0),
        cache_scales.stride(0),
        block_tables.stride(0),
        out.stride(0),
        num_blocks,
        max_model_len,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_TILE_SIZE=16,
        HEAD_TILE_SIZE=16,
        SHUFFLED=block_size > 1,
        num_warps=4,
        num_stages=1,
    )
    return out
'''


PREFILL_CODE = r'''

@triton.jit
def _gfx1x_portable_fp8_mqa_prefill_kernel(
    q_ptr,
    k_ptr,
    scales_ptr,
    weights_ptr,
    starts_ptr,
    ends_ptr,
    out_ptr,
    num_keys,
    stride_q_m: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_n: tl.int64,
    stride_k_d: tl.constexpr,
    stride_w_m: tl.int64,
    stride_w_h: tl.constexpr,
    stride_o_m: tl.int64,
    stride_o_n: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BF16-dot prefill MQA avoiding unsupported gfx1x raw-FP8 tl.dot."""
    row = tl.program_id(0)
    head = tl.arange(0, NUM_HEADS)[:, None]
    dim = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q_ptr
        + row * stride_q_m
        + head * stride_q_h
        + dim[None, :] * stride_q_d
    ).to(tl.bfloat16)
    weights = tl.load(
        weights_ptr + row * stride_w_m + head * stride_w_h
    ).to(tl.float32)
    start = tl.maximum(tl.load(starts_ptr + row), 0)
    end = tl.minimum(tl.load(ends_ptr + row), num_keys)

    for base in tl.range(0, num_keys, BLOCK_N):
        key_index = base + tl.arange(0, BLOCK_N)
        valid = (key_index >= start) & (key_index < end) & (key_index < num_keys)
        key = tl.load(
            k_ptr
            + dim[:, None] * stride_k_d
            + key_index[None, :] * stride_k_n,
            mask=valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(query, key)
        scale = tl.load(scales_ptr + key_index, mask=valid, other=0.0)
        scores *= scale[None, :]
        scores = tl.maximum(scores, 0.0) * weights
        logits = tl.sum(scores, axis=0)
        tl.store(
            out_ptr + row * stride_o_m + key_index * stride_o_n,
            logits,
            mask=valid,
        )


def gfx1x_portable_fp8_mqa_logits(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Portable gfx1x non-paged sparse-indexer logits."""
    num_queries, num_heads, head_dim = q.shape
    num_keys = k_fp8.shape[0]
    out = torch.full(
        (num_queries, num_keys),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    _gfx1x_portable_fp8_mqa_prefill_kernel[(num_queries,)](
        q,
        k_fp8,
        scale.reshape(-1),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        num_keys,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )
    return out
'''


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_sparse_indexer_mqa(path: Path) -> bool:
    """Patch DeepSeek V4's sparse-indexer MQA dispatch for gfx1x."""
    source = path.read_text()
    if SPARSE_MARKER in source:
        return False

    source = _replace_once(
        source,
        "    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950\n"
        "else:\n"
        "    _ON_GFX942 = False\n"
        "    _ON_GFX950 = False\n",
        "    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950, on_gfx1x\n"
        "else:\n"
        "    _ON_GFX942 = False\n"
        "    _ON_GFX950 = False\n\n"
        "    def on_gfx1x() -> bool:\n"
        "        return False\n",
        "ROCm platform import",
    )
    source = _replace_once(
        source,
        "\n\n@functools.lru_cache\ndef paged_mqa_logits_module():",
        PAGED_CODE + "\n\n@functools.lru_cache\ndef paged_mqa_logits_module():",
        "paged portable-kernel insertion",
    )
    source = _replace_once(
        source,
        "    if rocm_aiter_ops.is_enabled():\n"
        "        aiter_paged_mqa_logits_module = paged_mqa_logits_module()\n",
        "    if rocm_aiter_ops.is_enabled() and not on_gfx1x():\n"
        "        aiter_paged_mqa_logits_module = paged_mqa_logits_module()\n",
        "paged AITER gfx1x gate",
    )
    source = _replace_once(
        source,
        "    else:\n"
        "        return fp8_paged_mqa_logits_torch(\n"
        "            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len\n"
        "        )\n",
        "    else:\n"
        "        if on_gfx1x():\n"
        "            return gfx1x_portable_fp8_paged_mqa_logits(\n"
        "                q_fp8,\n"
        "                kv_cache_fp8,\n"
        "                weights,\n"
        "                context_lens,\n"
        "                block_tables,\n"
        "                max_model_len,\n"
        "            )\n"
        "        return fp8_paged_mqa_logits_torch(\n"
        "            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len\n"
        "        )\n",
        "paged gfx1x dispatch",
    )
    source = _replace_once(
        source,
        "\n\n@functools.lru_cache\ndef mqa_logits_module():",
        PREFILL_CODE + "\n\n@functools.lru_cache\ndef mqa_logits_module():",
        "prefill portable-kernel insertion",
    )
    source = _replace_once(
        source,
        "    k_fp8, scale = kv\n\n"
        "    # Temporarily route gfx942 to the vendored ROCm/aiter#3257 workaround.\n",
        "    k_fp8, scale = kv\n\n"
        "    if on_gfx1x():\n"
        "        return gfx1x_portable_fp8_mqa_logits(\n"
        "            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke\n"
        "        )\n\n"
        "    # Temporarily route gfx942 to the vendored ROCm/aiter#3257 workaround.\n",
        "prefill gfx1x dispatch",
    )
    path.write_text(source)
    return True


def patch_block_scaled_fp8_linear(path: Path) -> bool:
    """Use BF16 matrix-core dot for vLLM's block-scaled FP8 GEMM on gfx1x."""
    source = path.read_text()
    if LINEAR_MARKER in source:
        return False

    source = _replace_once(
        source,
        "    # Meta-parameters\n"
        "    BLOCK_SIZE_M: tl.constexpr,\n",
        "    # Meta-parameters\n"
        "    USE_BF16_DOT: tl.constexpr,\n"
        "    BLOCK_SIZE_M: tl.constexpr,\n",
        "block-FP8 kernel constexpr",
    )
    source = _replace_once(
        source,
        "        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n",
        "        # PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot\n"
        "        if USE_BF16_DOT:\n"
        "            dot = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))\n"
        "        else:\n"
        "            dot = tl.dot(a, b)\n"
        "        accumulator += dot * a_s[:, None] * b_s[None, :]\n",
        "block-FP8 dot implementation",
    )
    source = _replace_once(
        source,
        "    from vllm.platforms.rocm import on_gfx1250\n",
        "    from vllm.platforms.rocm import on_gfx1250, on_gfx1x\n",
        "gfx helper import",
    )
    source = _replace_once(
        source,
        "    def grid(META):\n"
        "        return (\n",
        "    config = dict(config)\n"
        "    config[\"USE_BF16_DOT\"] = on_gfx1x()\n\n"
        "    def grid(META):\n"
        "        return (\n",
        "block-FP8 gfx1x dispatch",
    )
    path.write_text(source)
    return True


def patch_dsv4_gfx1x(root: Path = Path(".")) -> list[Path]:
    """Apply all validated gfx1x DeepSeek V4 kernel patches below *root*."""
    targets = (
        (
            root / "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",
            patch_sparse_indexer_mqa,
        ),
        (
            root
            / "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
            patch_block_scaled_fp8_linear,
        ),
    )
    changed = []
    for path, patcher in targets:
        if path.exists() and patcher(path):
            changed.append(path)
    return changed
