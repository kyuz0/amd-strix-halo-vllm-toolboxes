"""DeepSeek V4 compatibility patches for ROCm gfx1x.

The sparse-indexer dispatch uses a separately packaged TileLang implementation
on gfx1x.  The block-scaled linear patch retains FP8 storage while converting
the matrix operands to BF16 for the dot product, because gfx1151 has no native
FP8 matrix-core dot product. A model-scoped cached-BF16 path additionally
avoids repeating that dequantization and the activation FP8 round trip during
decode.
"""

from pathlib import Path


SPARSE_MARKER = "# PATCHED: gfx1x TileLang sparse-indexer MQA"
LINEAR_MARKER = "# PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot"
CACHED_LINEAR_MARKER = "# PATCHED: gfx1x cached-BF16 W8A8 linear"


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


def patch_cached_bf16_w8a8_linear(path: Path) -> bool:
    """Use a cached BF16 weight and skinny GEMM for gfx1x W8A8 decode."""
    source = path.read_text()
    if CACHED_LINEAR_MARKER in source:
        return False

    source = _replace_once(
        source,
        "import torch\n",
        "import os\n\nimport torch\n",
        "cached-BF16 environment import",
    )
    source = _replace_once(
        source,
        "from .ScaledMMLinearKernel import (\n"
        "    Int8ScaledMMLinearLayerConfig,\n"
        ")\n\n\n"
        "class TritonInt8ScaledMMLinearKernel",
        "from .ScaledMMLinearKernel import (\n"
        "    Int8ScaledMMLinearLayerConfig,\n"
        ")\n\n\n"
        "def _on_gfx1x() -> bool:\n"
        "    if not current_platform.is_rocm():\n"
        "        return False\n"
        "    from vllm.platforms.rocm import on_gfx1x\n\n"
        "    return on_gfx1x()\n\n\n"
        "_GFX1X_W8A8_BF16 = False\n"
        "_GFX1X_W8A8_BF16_DIRECT = False\n"
        "_GFX1X_DIRECT_DIAGNOSTIC = [False, False]\n\n\n"
        "def refresh_gfx1x_w8a8_env() -> None:\n"
        "    global _GFX1X_W8A8_BF16, _GFX1X_W8A8_BF16_DIRECT\n"
        "    _GFX1X_W8A8_BF16 = (\n"
        "        os.environ.get(\"VLLM_GFX1X_W8A8_BF16\") == \"1\"\n"
        "        and _on_gfx1x()\n"
        "    )\n"
        "    _GFX1X_W8A8_BF16_DIRECT = (\n"
        "        _GFX1X_W8A8_BF16\n"
        "        and os.environ.get(\"VLLM_GFX1X_W8A8_BF16_DIRECT\") == \"1\"\n"
        "    )\n\n\n"
        "refresh_gfx1x_w8a8_env()\n\n\n"
        "class TritonInt8ScaledMMLinearKernel",
        "cached-BF16 environment policy",
    )
    source = _replace_once(
        source,
        "    def apply_block_scaled_mm(\n"
        "        self,\n"
        "        A: torch.Tensor,\n",
        "    def apply_weights(\n"
        "        self,\n"
        "        layer: torch.nn.Module,\n"
        "        x: torch.Tensor,\n"
        "        bias: torch.Tensor | None = None,\n"
        "        **kwargs,\n"
        "    ) -> torch.Tensor:\n"
        "        # PATCHED: gfx1x cached-BF16 W8A8 linear\n"
        "        if _GFX1X_W8A8_BF16_DIRECT and x.dtype == torch.bfloat16:\n"
        "            from .gfx1x_w8a8_bf16 import w8a8_block_bf16_direct\n\n"
        "            params = self._get_layer_params(layer)\n"
        "            weight_scale = (\n"
        "                params.weight_scale\n"
        "                if params.weight_scale_inv is None\n"
        "                else params.weight_scale_inv\n"
        "            )\n"
        "            input_2d = x.view(-1, x.shape[-1])\n"
        "            output = w8a8_block_bf16_direct(\n"
        "                input_2d,\n"
        "                params.weight,\n"
        "                weight_scale,\n"
        "                list(self.weight_group_shape),\n"
        "            )\n"
        "            if output is not None:\n"
        "                if not _GFX1X_DIRECT_DIAGNOSTIC[0]:\n"
        "                    _GFX1X_DIRECT_DIAGNOSTIC[0] = True\n"
        "                    print(\n"
        "                        f\"[gfx1x_w8a8] BF16 direct path active \"\n"
        "                        f\"(M={input_2d.shape[0]}, N={params.weight.shape[0]})\",\n"
        "                        flush=True,\n"
        "                    )\n"
        "                if bias is not None:\n"
        "                    output = output + bias\n"
        "                output_shape = [*x.shape[:-1], params.weight.shape[0]]\n"
        "                return output.to(dtype=self.config.out_dtype).view(*output_shape)\n\n"
        "        if _GFX1X_W8A8_BF16_DIRECT and not _GFX1X_DIRECT_DIAGNOSTIC[1]:\n"
        "            _GFX1X_DIRECT_DIAGNOSTIC[1] = True\n"
        "            rows = x.numel() // x.shape[-1]\n"
        "            print(\n"
        "                f\"[gfx1x_w8a8] BF16 direct path deferred (M={rows})\",\n"
        "                flush=True,\n"
        "            )\n"
        "        return super().apply_weights(layer, x, bias, **kwargs)\n\n"
        "    def apply_block_scaled_mm(\n"
        "        self,\n"
        "        A: torch.Tensor,\n",
        "cached-BF16 direct linear override",
    )
    source = _replace_once(
        source,
        "    from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
        "        w8a8_triton_block_scaled_mm,\n"
        "    )\n\n"
        "    return w8a8_triton_block_scaled_mm(\n",
        "    if _GFX1X_W8A8_BF16:\n"
        "        from .gfx1x_w8a8_bf16 import w8a8_block_fp8_bf16\n\n"
        "        output = w8a8_block_fp8_bf16(\n"
        "            qx, weight, x_scale, weight_scale, block_size, output_dtype\n"
        "        )\n"
        "        if output is not None:\n"
        "            return output\n\n"
        "    from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
        "        w8a8_triton_block_scaled_mm,\n"
        "    )\n\n"
        "    return w8a8_triton_block_scaled_mm(\n",
        "cached-BF16 quantized-input fallback",
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
        (
            root
            / "vllm/model_executor/kernels/linear/scaled_mm/triton.py",
            patch_cached_bf16_w8a8_linear,
        ),
    )
    changed = []
    for path, patcher in targets:
        if path.exists() and patcher(path):
            changed.append(path)
    return changed
