import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import launcher_features  # noqa: E402
import models  # noqa: E402


MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"


class LauncherFeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = models.MODEL_TABLE[MODEL_ID]

    def test_deepseek_defaults_to_native_dspark7_greedy_and_automatic_warmup(self):
        speculative = self.config["speculative_config"]
        self.assertEqual(speculative["method"], "dspark")
        self.assertEqual(speculative["num_speculative_tokens"], 7)
        self.assertEqual(speculative["draft_sample_method"], "greedy")
        self.assertTrue(speculative["disable_padded_drafter_batch"])
        self.assertTrue(speculative["enforce_eager"])
        self.assertEqual(self.config["warmup"]["prompt_tokens"], 2048)
        self.assertEqual(self.config["valid_tp"], [1, 2])

    def test_deepseek_reserves_room_for_post_profile_bf16_weight_cache(self):
        flags = self.config["extra_flags"]
        index = flags.index("--kv-cache-memory-bytes")
        self.assertEqual(flags[index + 1], "6442450944")

    def test_deepseek_uses_conservative_tp2_prefill_budget(self):
        flags = self.config["extra_flags"]
        index = flags.index("--max-num-batched-tokens")
        self.assertEqual(flags[index + 1], "512")

    def test_speculative_args_are_compact_and_can_be_disabled(self):
        args = launcher_features.speculative_config_args(self.config, True)
        self.assertEqual(args[0], "--speculative-config")
        self.assertEqual(json.loads(args[1]), self.config["speculative_config"])
        self.assertNotIn(" ", args[1])
        self.assertEqual(
            launcher_features.speculative_config_args(self.config, False), []
        )

    @patch("launcher_features.subprocess.Popen")
    @patch("launcher_features.get_warmup_script")
    def test_warmup_helper_is_spawned_with_parent_and_model_context(
        self, get_script, popen
    ):
        get_script.return_value = Path("/opt/vllm_warmup.py")
        launcher_features.launch_automatic_warmup(
            MODEL_ID, "8000", self.config, {"TRITON_CACHE_DIR": "/cache"}
        )
        args, kwargs = popen.call_args
        self.assertEqual(args[0][-1], "/opt/vllm_warmup.py")
        self.assertEqual(kwargs["env"]["VLLM_WARMUP_MODEL"], MODEL_ID)
        self.assertEqual(kwargs["env"]["VLLM_WARMUP_PORT"], "8000")
        self.assertEqual(
            kwargs["env"]["VLLM_WARMUP_PROMPT_TOKENS"], "2048"
        )
        self.assertIn("VLLM_WARMUP_PARENT_PID", kwargs["env"])

    def test_both_launchers_expose_and_execute_the_features(self):
        for launcher in ("start_vllm.py", "start_vllm_cluster.py"):
            with self.subTest(launcher=launcher):
                source = (ROOT / "scripts" / launcher).read_text()
                self.assertIn("Speculative Decoding:", source)
                self.assertIn("Automatic Warmup:", source)
                self.assertIn("speculative_config_args", source)
                self.assertIn("launch_automatic_warmup", source)

    def test_runtime_image_contains_the_warmup_helper(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        self.assertIn(
            "COPY scripts/launcher_features.py /opt/launcher_features.py",
            dockerfile,
        )
        self.assertIn(
            "COPY scripts/vllm_warmup.py /opt/vllm_warmup.py", dockerfile
        )


if __name__ == "__main__":
    unittest.main()
