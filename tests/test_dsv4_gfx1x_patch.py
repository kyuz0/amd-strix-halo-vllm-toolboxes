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
    k_fp8, scale = kv

    # Temporarily route gfx942 to the vendored ROCm/aiter#3257 workaround.
    if _ON_GFX942 and rocm_aiter_ops.is_enabled():
        return None
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


def w8a8_triton_block_scaled_mm(A, B):
    from vllm.platforms.rocm import on_gfx1250

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
                "if rocm_aiter_ops.is_enabled() and not on_gfx1x():", patched
            )
            self.assertIn(
                "return gfx1x_portable_fp8_paged_mqa_logits(", patched
            )
            self.assertIn("return gfx1x_portable_fp8_mqa_logits(", patched)
            self.assertIn("SHUFFLED=block_size > 1", patched)
            self.assertIn(
                "(token[:, None] // BLOCK_TILE_SIZE)", patched
            )
            self.assertIn("key = tl.load(kv_ptr + cache_offsets).to(tl.bfloat16)", patched)
            self.assertEqual(patched.count(".to(tl.bfloat16)"), 4)
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

    def test_ubuntu_image_copies_helper_before_running_main_patcher(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        helper_copy = (
            "COPY scripts/patch_dsv4_gfx1x.py /opt/vllm/patch_dsv4_gfx1x.py"
        )
        run_patch = "RUN python /opt/vllm/patch_strix.py"
        self.assertIn(helper_copy, dockerfile)
        self.assertLess(dockerfile.index(helper_copy), dockerfile.index(run_patch))
        self.assertIn(patch_dsv4_gfx1x.SPARSE_MARKER.removeprefix("# "), dockerfile)
        self.assertIn(patch_dsv4_gfx1x.LINEAR_MARKER.removeprefix("# "), dockerfile)


if __name__ == "__main__":
    unittest.main()
