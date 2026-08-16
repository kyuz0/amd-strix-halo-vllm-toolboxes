import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import models  # noqa: E402
import verify_vllm_compat  # noqa: E402


MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"

LOCAL_PATCH_PATHS = (
    "vllm/_aiter_ops.py",
    "vllm/models/deepseek_v4/amd/model.py",
    "vllm/models/deepseek_v4/amd/rocm.py",
    "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",
    "vllm/v1/attention/ops/gfx1x_tilelang_mqa.py",
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
    "vllm/v1/executor/ray_executor_v2.py",
    "vllm/model_executor/layers/fused_moe/oracle/unquantized.py",
    "vllm/platforms/rocm.py",
    "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
    "aiter/jit/__init__.py",
    "flash_attn/flash_attn_interface.py",
    "amdsmi_interface.py",
    "aiter_meta/csrc/include/ck_tile/vec_convert.h",
)


class NativeDSparkIntegrationTests(unittest.TestCase):
    def test_model_uses_native_dspark_block_speculation(self):
        config = models.MODEL_TABLE[MODEL_ID]
        speculative = config["speculative_config"]
        self.assertEqual(speculative["method"], "dspark")
        self.assertEqual(speculative["num_speculative_tokens"], 7)
        self.assertEqual(speculative["draft_sample_method"], "greedy")
        self.assertTrue(speculative["disable_padded_drafter_batch"])
        self.assertTrue(speculative["enforce_eager"])
        self.assertEqual(config["warmup"]["prompt_tokens"], 2048)
        self.assertEqual(config["valid_tp"], [1, 2])
        self.assertNotIn("dspark_mtp", config)

    def test_image_pins_audited_stable_and_runs_compatibility_guard(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        self.assertIn(
            "ARG VLLM_REF=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
            dockerfile,
        )
        self.assertIn(
            "COPY scripts/verify_vllm_compat.py /opt/vllm/verify_vllm_compat.py",
            dockerfile,
        )
        self.assertIn("python /opt/vllm/verify_vllm_compat.py", dockerfile)
        self.assertNotIn("patch_dspark_mtp.py", dockerfile)
        self.assertNotIn("dspark_mtp.py", dockerfile)

    def test_manifest_tracks_native_interfaces_and_is_in_the_image(self):
        manifest = (ROOT / "docs" / "VLLM_PATCH_MANIFEST.md").read_text()
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        for path in verify_vllm_compat.REQUIRED_NATIVE_INTERFACES:
            with self.subTest(path=path):
                self.assertIn(path, manifest)
        self.assertIn("6e448d0ea9bf3d88d898b65449ca6dc2aec170ac", manifest)
        self.assertIn(
            "COPY docs/VLLM_PATCH_MANIFEST.md /opt/VLLM_PATCH_MANIFEST.md",
            dockerfile,
        )

    def test_manifest_tracks_every_local_patch_path(self):
        manifest = (ROOT / "docs" / "VLLM_PATCH_MANIFEST.md").read_text()
        for path in LOCAL_PATCH_PATHS:
            with self.subTest(path=path):
                self.assertIn(path, manifest)

    def test_compatibility_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for relative_path, anchors in (
                verify_vllm_compat.REQUIRED_NATIVE_INTERFACES.items()
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(anchors))

            verify_vllm_compat.verify_native_interfaces(root)
            first_path = root / next(iter(verify_vllm_compat.REQUIRED_NATIVE_INTERFACES))
            first_path.write_text("upstream changed\n")
            with self.assertRaisesRegex(RuntimeError, "Unsupported vLLM source layout"):
                verify_vllm_compat.verify_native_interfaces(root)


if __name__ == "__main__":
    unittest.main()
