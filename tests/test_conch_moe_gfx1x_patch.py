import ast
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_conch_moe_gfx1x  # noqa: E402


OPT_FLAGS_SOURCE = '''\
# isort: off
# fmt: off
from dataclasses import dataclass

import triton
from triton_kernels.target_info import get_cdna_version, get_rdna_version
from triton_kernels.tensor import FP4
import torch
from .opt_flags_details import opt_flags_amd, opt_flags_nvidia
from triton_kernels.tensor import bitwidth


@dataclass
class OptFlags:
    block_m: int


def make_default_opt_flags_amd(
    lhs_dtype, rhs_dtype, precision_config, constraints,
):
    is_cdna4 = get_cdna_version() == 4
    block_m = 32
    m = 512
    if is_cdna4 and m >= 512:
        block_m = 128
    elif get_rdna_version() in (3, 4) and m >= 512:
        block_m = 64
    block_n = 256
    block_k = 128
    num_warps = 2
    num_stages = 2
    grid_size = 1
    if grid_size > 40:
        split_k = 1
    else:
        n_cu = torch.cuda.get_device_properties(0).multi_processor_count
        split_k = max(1, n_cu // grid_size)
    # AMD-specific
    target_kernel_kwargs = {"waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    epilogue_subtile = constraints.get('epilogue_subtile', None)


def make_opt_flags():
    backend = triton.runtime.driver.active.get_current_target().backend
    return backend
'''

class ConchMoeGfx1xPatchTests(unittest.TestCase):
    def test_patch_is_opt_in_and_matches_alex_production_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt_flags.py"
            path.write_text(OPT_FLAGS_SOURCE)

            self.assertTrue(patch_conch_moe_gfx1x.patch_opt_flags(path))
            patched = path.read_text()

            self.assertIn(patch_conch_moe_gfx1x.MARKER, patched)
            self.assertIn('os.environ.get("VLLM_GFX1X_MOE_TUNE")', patched)
            expected_defaults = (
                '"block_m": int(os.environ.get("VLLM_GFX1X_MOE_BM", "16"))',
                '"block_n": int(os.environ.get("VLLM_GFX1X_MOE_BN", "32"))',
                '"block_k": int(os.environ.get("VLLM_GFX1X_MOE_BK", "256"))',
                '"num_warps": int(os.environ.get("VLLM_GFX1X_MOE_NW", "2"))',
                '"num_stages": int(os.environ.get("VLLM_GFX1X_MOE_NS", "2"))',
                '"waves_per_eu": int(os.environ.get("VLLM_GFX1X_MOE_WPE", "1"))',
            )
            for expected in expected_defaults:
                self.assertIn(expected, patched)
            ast.parse(patched)

            self.assertFalse(patch_conch_moe_gfx1x.patch_opt_flags(path))
            self.assertEqual(path.read_text(), patched)

    def test_tuning_is_limited_to_skinny_rdna_mxfp4(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt_flags.py"
            path.write_text(OPT_FLAGS_SOURCE)
            patch_conch_moe_gfx1x.patch_opt_flags(path)
            patched = path.read_text()

            self.assertIn("get_rdna_version() in (3, 4)", patched)
            self.assertIn("and block_m < 128", patched)
            self.assertIn("and bitwidth(lhs_dtype) == 16", patched)
            self.assertIn("and bitwidth(rhs_dtype) == 4", patched)
            self.assertIn("and precision_config.weight_scale is not None", patched)
            self.assertIn('target_kernel_kwargs = {\n            "waves_per_eu"', patched)

    def test_patch_fails_closed_when_conch_layout_moves(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt_flags.py"
            path.write_text("def changed_upstream():\n    pass\n")
            with self.assertRaisesRegex(RuntimeError, "expected one anchor"):
                patch_conch_moe_gfx1x.patch_opt_flags(path)

    def test_image_applies_patch_after_installing_conch_and_vllm(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        copy_patch = (
            "COPY scripts/patch_conch_moe_gfx1x.py "
            "/opt/patch_conch_moe_gfx1x.py"
        )
        install_requirements = (
            "python -m pip install -c /tmp/rocm-stack.constraints "
            "-r requirements/rocm.txt"
        )
        install_vllm = "python -m pip install --no-deps /tmp/dist/*.whl"
        run_patch = "python /opt/patch_conch_moe_gfx1x.py"
        self.assertIn(copy_patch, dockerfile)
        self.assertIn(run_patch, dockerfile)
        self.assertLess(
            dockerfile.index(install_requirements), dockerfile.index(run_patch)
        )
        self.assertLess(dockerfile.index(install_vllm), dockerfile.index(run_patch))


if __name__ == "__main__":
    unittest.main()
