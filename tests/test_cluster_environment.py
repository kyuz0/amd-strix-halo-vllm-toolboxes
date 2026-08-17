import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cluster_manager  # noqa: E402
import models  # noqa: E402
import patch_strix  # noqa: E402


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
        self.assertEqual(deepseek["VLLM_GFX1X_W8A8_BF16"], "1")
        self.assertEqual(deepseek["VLLM_GFX1X_W8A8_BF16_DIRECT"], "1")
        self.assertEqual(deepseek["VLLM_GFX1X_MOE_TUNE"], "1")
        self.assertEqual(deepseek["VLLM_GFX1X_RADIX_TOPK"], "1")

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

    def test_strix_patch_refreshes_cached_policy_after_ray_applies_worker_env(self):
        source = """\
class RayWorkerProc:
    def initialize_worker(self, env_vars):
        for key, value in env_vars.items():
            os.environ[key] = value

        self.local_rank = 0
"""
        with patch("patch_strix.Path.read_text", return_value=source), patch(
            "patch_strix.Path.write_text"
        ) as write_text:
            executor = Path("vllm/v1/executor/ray_executor_v2.py")
            patch_strix.patch_ray_executor_aiter_env(executor)
            patched = write_text.call_args.args[0]

        refresh = "rocm_aiter_ops.refresh_env_variables()"
        w8a8_refresh = "refresh_gfx1x_w8a8_env()"
        self.assertEqual(patched.count(refresh), 1)
        self.assertEqual(patched.count(w8a8_refresh), 1)
        self.assertLess(
            patched.index('os.environ[key] = value'), patched.index(refresh)
        )
        self.assertLess(patched.index(refresh), patched.index(w8a8_refresh))
        self.assertLess(patched.index(w8a8_refresh), patched.index("self.local_rank = 0"))
        self.assertLess(patched.index(refresh), patched.index("self.local_rank = 0"))


if __name__ == "__main__":
    unittest.main()
