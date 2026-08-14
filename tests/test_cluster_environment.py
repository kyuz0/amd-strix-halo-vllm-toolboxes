import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cluster_manager  # noqa: E402
import models  # noqa: E402


AITER_UNSET = "unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR"


class ClusterEnvironmentTests(unittest.TestCase):
    def test_image_does_not_bake_model_specific_aiter_policy(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        env_block = dockerfile.rsplit("ENV VIRTUAL_ENV=", 1)[1].split("\n\n", 1)[0]
        self.assertNotIn("VLLM_ROCM_USE_AITER=", env_block)
        self.assertNotIn("VLLM_ROCM_USE_AITER_LINEAR=", env_block)

    def test_default_and_deepseek_model_environments_are_explicit(self):
        inherited = {
            "VLLM_ROCM_USE_AITER": "stale",
            "VLLM_ROCM_USE_AITER_LINEAR": "stale",
        }
        inherited.update(models.get_model_env({}))
        self.assertEqual(inherited["VLLM_ROCM_USE_AITER"], "0")
        self.assertEqual(inherited["VLLM_ROCM_USE_AITER_LINEAR"], "0")

        deepseek = models.get_model_env(
            models.MODEL_TABLE["deepseek-ai/DeepSeek-V4-Flash-0731"]
        )
        self.assertEqual(deepseek["VLLM_ROCM_USE_AITER"], "1")
        self.assertEqual(deepseek["VLLM_ROCM_USE_AITER_LINEAR"], "0")

    def test_both_launchers_apply_the_selected_model_environment(self):
        expected = "env.update(models.get_model_env(config))"
        for launcher in ("start_vllm.py", "start_vllm_cluster.py"):
            with self.subTest(launcher=launcher):
                source = (ROOT / "scripts" / launcher).read_text()
                self.assertIn(expected, source)

    @patch("cluster_manager.subprocess.run")
    def test_head_ray_daemon_starts_without_aiter_policy(self, run):
        self.assertTrue(cluster_manager.setup_head_node("192.168.100.1"))
        script = run.call_args.kwargs["input"].decode()
        self.assertIn(AITER_UNSET, script)
        self.assertLess(script.index(AITER_UNSET), script.index("ray start --head"))

    @patch("cluster_manager.subprocess.run")
    def test_worker_ray_daemon_starts_without_aiter_policy(self, run):
        self.assertTrue(
            cluster_manager.setup_worker_node(
                "192.168.100.2", "192.168.100.1", "vllm-therock-gfx1151-dev"
            )
        )
        script = run.call_args.kwargs["input"].decode()
        self.assertIn(AITER_UNSET, script)
        self.assertLess(script.index(AITER_UNSET), script.index("ray start --address="))

    def test_legacy_cluster_script_also_sanitizes_both_ray_daemons(self):
        script = (ROOT / "scripts" / "configure_cluster.sh").read_text()
        self.assertEqual(script.count(AITER_UNSET), 2)


if __name__ == "__main__":
    unittest.main()
