import ast
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_dsv4_gfx1x  # noqa: E402


SPARSE_SOURCE = '''\
import functools

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX942, _ON_GFX950
else:
    _ON_GFX942 = False
    _ON_GFX950 = False


def fp8_paged_mqa_logits_torch(*args):
    pass


@functools.lru_cache
def paged_mqa_logits_module():
    pass


def rocm_fp8_paged_mqa_logits(
    q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
    schedule_metadata, max_model_len,
):
    aiter_paged_mqa_logits_module = None
    # The pinned vLLM revision initializes scheduling data before dispatch.
    batch_size, next_n = q_fp8.shape[:2]
    block_size = kv_cache_fp8.shape[1]

    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        return None
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


def fp8_mqa_logits_torch(*args):
    pass


@functools.lru_cache
def mqa_logits_module():
    pass


def rocm_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()
'''


LINEAR_SOURCE = '''\
@triton.jit
def _w8a8_triton_block_scaled_mm(
    A,
    B,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    for k in range(1):
        a = A
        b = B
        a_s = 1
        b_s = 1
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]


def w8a8_triton_block_scaled_mm(A, B, block_size):
    assert len(block_size) == 2

    config = {"BLOCK_SIZE_M": 64}

    def grid(META):
        return (
            1,
        )
'''


class DeepSeekV4Gfx1xPatchTests(unittest.TestCase):
    def test_sparse_patch_routes_both_mqa_paths_only_on_gfx1x(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rocm_aiter_mla_sparse.py"
            path.write_text(SPARSE_SOURCE)

            self.assertTrue(patch_dsv4_gfx1x.patch_sparse_indexer_mqa(path))
            patched = path.read_text()

            self.assertIn(patch_dsv4_gfx1x.SPARSE_MARKER, patched)
            self.assertIn(
                "if on_gfx1x():\n"
                "        # PATCHED: gfx1x TileLang sparse-indexer MQA",
                patched,
            )
            self.assertIn(
                "return fp8_paged_mqa_logits_tilelang(", patched
            )
            self.assertIn("return fp8_mqa_logits_tilelang(", patched)
            self.assertNotIn("_gfx1x_portable_fp8_paged_mqa_kernel", patched)
            self.assertNotIn("_gfx1x_portable_fp8_mqa_prefill_kernel", patched)
            ast.parse(patched)

            self.assertFalse(patch_dsv4_gfx1x.patch_sparse_indexer_mqa(path))
            self.assertEqual(path.read_text(), patched)

    def test_block_scaled_linear_patch_preserves_native_non_gfx1x_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fp8_utils.py"
            path.write_text(LINEAR_SOURCE)

            self.assertTrue(
                patch_dsv4_gfx1x.patch_block_scaled_fp8_linear(path)
            )
            patched = path.read_text()

            self.assertIn(patch_dsv4_gfx1x.LINEAR_MARKER, patched)
            self.assertIn(
                "from vllm.platforms.rocm import on_gfx1x", patched
            )
            self.assertIn('config["USE_BF16_DOT"] = on_gfx1x()', patched)
            self.assertIn(
                "dot = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))", patched
            )
            self.assertIn("else:\n            dot = tl.dot(a, b)", patched)
            ast.parse(patched)

            self.assertFalse(
                patch_dsv4_gfx1x.patch_block_scaled_fp8_linear(path)
            )
            self.assertEqual(path.read_text(), patched)

    def test_patch_fails_closed_when_upstream_anchor_changes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fp8_utils.py"
            path.write_text("def changed_upstream():\n    pass\n")
            with self.assertRaisesRegex(RuntimeError, "expected one anchor"):
                patch_dsv4_gfx1x.patch_block_scaled_fp8_linear(path)

    def test_tilelang_module_is_parseable_and_handles_shuffled_cache(self):
        module_path = ROOT / "scripts/gfx1x_tilelang_mqa.py"
        source = module_path.read_text()

        ast.parse(source)
        self.assertIn("AlexKGwyn/ds4-vllm-public", source)
        self.assertIn("tilelang.compile", source)
        self.assertIn("def fp8_mqa_logits_tilelang(", source)
        self.assertIn("def fp8_paged_mqa_logits_tilelang(", source)
        self.assertIn("def _logical_cache_values(", source)
        self.assertIn(".permute(0, 1, 3, 2, 4)", source)
        self.assertIn("_KV_BUCKET = 512", source)
        self.assertIn("_PREFILL_KV_BUCKET = 8192", source)

    def test_ubuntu_image_copies_helpers_before_running_main_patcher(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        helper_copy = (
            "COPY scripts/patch_dsv4_gfx1x.py /opt/vllm/patch_dsv4_gfx1x.py"
        )
        tilelang_copy = (
            "COPY scripts/gfx1x_tilelang_mqa.py "
            "/opt/vllm/vllm/v1/attention/ops/gfx1x_tilelang_mqa.py"
        )
        run_patch = "RUN python /opt/vllm/patch_strix.py"
        self.assertIn(helper_copy, dockerfile)
        self.assertIn(tilelang_copy, dockerfile)
        self.assertLess(dockerfile.index(helper_copy), dockerfile.index(run_patch))
        self.assertLess(dockerfile.index(tilelang_copy), dockerfile.index(run_patch))
        self.assertIn(patch_dsv4_gfx1x.SPARSE_MARKER.removeprefix("# "), dockerfile)
        self.assertIn(patch_dsv4_gfx1x.LINEAR_MARKER.removeprefix("# "), dockerfile)
        self.assertIn("assert importlib.util.find_spec('tilelang')", dockerfile)


if __name__ == "__main__":
    unittest.main()
