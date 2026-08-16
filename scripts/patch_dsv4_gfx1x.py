"""DeepSeek V4 compatibility patches for ROCm gfx1x.

The sparse-indexer dispatch uses a separately packaged TileLang implementation
on gfx1x.  The block-scaled linear patch retains FP8 storage while converting
the matrix operands to BF16 for the dot product, because gfx1151 has no native
FP8 matrix-core dot product.
"""

from pathlib import Path


SPARSE_MARKER = "# PATCHED: gfx1x TileLang sparse-indexer MQA"
LINEAR_MARKER = "# PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot"


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_sparse_indexer_mqa(path: Path) -> bool:
    """Route DeepSeek V4 sparse-indexer MQA to TileLang on gfx1x."""
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
        "    aiter_paged_mqa_logits_module = None\n",
        "    if on_gfx1x():\n"
        "        # PATCHED: gfx1x TileLang sparse-indexer MQA\n"
        "        from vllm.v1.attention.ops.gfx1x_tilelang_mqa import (\n"
        "            fp8_paged_mqa_logits_tilelang,\n"
        "        )\n\n"
        "        return fp8_paged_mqa_logits_tilelang(\n"
        "            q_fp8,\n"
        "            kv_cache_fp8,\n"
        "            weights,\n"
        "            context_lens,\n"
        "            block_tables,\n"
        "            max_model_len,\n"
        "        )\n\n"
        "    aiter_paged_mqa_logits_module = None\n",
        "paged gfx1x dispatch",
    )
    source = _replace_once(
        source,
        "    aiter_mqa_logits_module = None\n",
        "    if on_gfx1x():\n"
        "        k_fp8, scale = kv\n"
        "        from vllm.v1.attention.ops.gfx1x_tilelang_mqa import (\n"
        "            fp8_mqa_logits_tilelang,\n"
        "        )\n\n"
        "        return fp8_mqa_logits_tilelang(\n"
        "            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke\n"
        "        )\n\n"
        "    aiter_mqa_logits_module = None\n",
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
        "    assert len(block_size) == 2\n",
        "    from vllm.platforms.rocm import on_gfx1x\n\n"
        "    assert len(block_size) == 2\n",
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
    """Apply all gfx1x DeepSeek V4 patches below *root*."""
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
