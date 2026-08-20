import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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

        config = models.MODEL_TABLE["deepseek-ai/DeepSeek-V4-Flash-0731"]
        deepseek_tp1 = models.get_model_env(config, 1)
        deepseek_tp2 = models.get_model_env(config, 2)
        for deepseek in (deepseek_tp1, deepseek_tp2):
            self.assertEqual(deepseek["VLLM_ROCM_USE_AITER"], "1")
            self.assertEqual(deepseek["VLLM_ROCM_USE_AITER_LINEAR"], "0")
            self.assertEqual(deepseek["VLLM_GFX1X_MOE_TUNE"], "1")
            self.assertEqual(deepseek["VLLM_GFX1X_RADIX_TOPK"], "1")
        self.assertEqual(deepseek_tp1["VLLM_GFX1X_W8A8_BF16"], "0")
        self.assertEqual(deepseek_tp1["VLLM_GFX1X_W8A8_BF16_DIRECT"], "0")
        self.assertEqual(deepseek_tp2["VLLM_GFX1X_W8A8_BF16"], "1")
        self.assertEqual(deepseek_tp2["VLLM_GFX1X_W8A8_BF16_DIRECT"], "1")

    def test_both_launchers_apply_the_selected_model_environment(self):
        expected = "model_env = models.get_model_env(config, current_tp)"
        for launcher in ("start_vllm.py", "start_vllm_cluster.py"):
            with self.subTest(launcher=launcher):
                source = (ROOT / "scripts" / launcher).read_text()
                self.assertIn(expected, source)
                self.assertIn("env.update(model_env)", source)

    def test_benchmark_launchers_resolve_model_environment_by_tp(self):
        expected = {
            "run_vllm_bench.py": "models.get_model_env(MODEL_TABLE[model], tp_size)",
            "vllm_cluster_bench.py": (
                "models.get_model_env(MODEL_TABLE[model], CLUSTER_TP)"
            ),
            "find_max_context.py": "models.get_model_env(config, tp_size)",
        }
        for filename, call in expected.items():
            with self.subTest(filename=filename):
                source = (ROOT / "benchmarks" / filename).read_text()
                self.assertIn(call, source)

    def test_strix_patch_disables_aiter_sampler_on_gfx1x(self):
        source = """\
def _skip_aiter_sampler_on_gfx1250() -> bool:
    # Lazy ROCm-only import; keeps arch detection out of import time on CUDA/CPU.
    from vllm.platforms.rocm import on_gfx1250

    return on_gfx1250()

enabled = (
    rocm_aiter_ops.is_enabled()
    and not _skip_aiter_sampler_on_gfx1250()  # TODO (JPVILLAM): Enable
)
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "topk_topp_sampler.py"
            path.write_text(source)
            self.assertTrue(patch_strix.patch_gfx1x_aiter_sampler(path))
            patched = path.read_text()
            self.assertIn(patch_strix.GFX1X_AITER_SAMPLER_MARKER, patched)
            self.assertIn("from vllm.platforms.rocm import on_gfx1x", patched)
            self.assertIn("and not _skip_aiter_sampler_on_gfx1x()", patched)
            self.assertNotIn("on_gfx1250", patched)
            self.assertFalse(patch_strix.patch_gfx1x_aiter_sampler(path))

    @patch("cluster_manager.subprocess.run")
    def test_head_ray_daemon_starts_without_aiter_policy(self, run):
        self.assertTrue(cluster_manager.setup_head_node("192.168.100.1"))
        script = run.call_args.kwargs["input"].decode()
        self.assertIn(AITER_UNSET, script)
        self.assertLess(script.index(AITER_UNSET), script.index("ray start --head"))

    def test_auto_transport_resolves_to_roce_without_active_ib_hca(self):
        with patch("cluster_manager.detect_ib_hca", return_value=None), patch(
            "cluster_manager.detect_roce_hca", return_value="irdma0:1"
        ):
            self.assertEqual(cluster_manager.resolve_transport("auto"), "roce")

    def test_auto_transport_resolves_to_ethernet_without_any_rdma(self):
        with patch("cluster_manager.detect_ib_hca", return_value=None), patch(
            "cluster_manager.detect_roce_hca", return_value=None
        ):
            self.assertEqual(cluster_manager.resolve_transport("auto"), "ethernet")

    def test_auto_transport_resolves_to_infiniband_with_active_ib_hca(self):
        with patch("cluster_manager.detect_ib_hca", return_value="mlx4_0:1"):
            self.assertEqual(cluster_manager.resolve_transport("auto"), "infiniband")

    def test_auto_transport_prefers_infiniband_over_roce(self):
        with patch("cluster_manager.detect_ib_hca", return_value="mlx4_0:1"), patch(
            "cluster_manager.detect_roce_hca", return_value="irdma0:1"
        ):
            self.assertEqual(cluster_manager.resolve_transport("auto"), "infiniband")

    def test_transport_env_ethernet_disables_ib(self):
        env = cluster_manager.transport_env("ethernet", "eth0")
        self.assertEqual(env["NCCL_IB_DISABLE"], "1")
        self.assertNotIn("NCCL_IB_HCA", env)
        self.assertNotIn("NCCL_IB_GID_INDEX", env)
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eth0")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "eth0")

    def test_transport_env_roce_uses_gid_index_one(self):
        env = cluster_manager.transport_env("roce", "eth0")
        self.assertEqual(env["NCCL_IB_DISABLE"], "0")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "1")
        self.assertNotIn("NCCL_PROTO", env)

    def test_transport_env_infiniband_pins_hca_and_gid_zero(self):
        with patch("cluster_manager.detect_ib_hca", return_value="mlx4_0:1"):
            env = cluster_manager.transport_env("infiniband", "eth0")
        self.assertEqual(env["NCCL_IB_DISABLE"], "0")
        self.assertEqual(env["NCCL_IB_HCA"], "mlx4_0:1")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "0")
        self.assertEqual(env["NCCL_PROTO"], "LL")
        self.assertEqual(env["NCCL_ALGO"], "Ring")

    def test_transport_env_infiniband_honors_vllm_ib_hca_override(self):
        with patch.dict("os.environ", {"VLLM_IB_HCA": "mlx5_0:1"}), patch(
            "cluster_manager.detect_ib_hca", return_value="mlx4_0:1"
        ):
            env = cluster_manager.transport_env("infiniband", "eth0")
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5_0:1")

    def test_transport_env_roce_honors_vllm_roce_hca_override(self):
        with patch.dict("os.environ", {"VLLM_ROCE_HCA": "usb4_rdma0:1"}), patch(
            "cluster_manager.active_rdma_ports",
            return_value=[("irdma0:1", "Ethernet"), ("usb4_rdma0:1", "Ethernet")],
        ), patch("cluster_manager.detect_roce_hca", return_value="irdma0:1"):
            env = cluster_manager.transport_env("roce", "thunderbolt0")
        self.assertEqual(env["NCCL_IB_HCA"], "usb4_rdma0:1")

    def test_transport_env_roce_pins_hca_with_multiple_active_rails(self):
        with patch(
            "cluster_manager.active_rdma_ports",
            return_value=[("irdma0:1", "Ethernet"), ("usb4_rdma0:1", "Ethernet")],
        ), patch("cluster_manager.detect_roce_hca", return_value="irdma0:1"):
            env = cluster_manager.transport_env("roce", "eth0")
        self.assertEqual(env["NCCL_IB_HCA"], "irdma0:1")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "1")

    def test_transport_env_roce_leaves_hca_unset_with_single_rail(self):
        with patch(
            "cluster_manager.active_rdma_ports",
            return_value=[("irdma0:1", "Ethernet")],
        ):
            env = cluster_manager.transport_env("roce", "eth0")
        self.assertNotIn("NCCL_IB_HCA", env)
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "1")

    def test_warn_multi_rdma_is_silent_with_less_than_two_ports(self):
        captured = io.StringIO()
        with patch(
            "cluster_manager.active_rdma_ports", return_value=[("irdma0:1", "Ethernet")]
        ), redirect_stdout(captured):
            cluster_manager.warn_multi_rdma("roce")
        self.assertEqual(captured.getvalue(), "")

    def test_warn_multi_rdma_lists_candidates_and_effective_hca(self):
        captured = io.StringIO()
        with patch(
            "cluster_manager.active_rdma_ports",
            return_value=[
                ("mlx4_0:1", "InfiniBand"),
                ("irdma0:1", "Ethernet"),
            ],
        ), patch("cluster_manager.detect_ib_hca", return_value="mlx4_0:1"), patch.dict(
            "os.environ", {}
        ), redirect_stdout(captured):
            cluster_manager.warn_multi_rdma("infiniband")
        text = captured.getvalue()
        self.assertIn("2 active RDMA ports detected", text)
        self.assertIn("mlx4_0:1", text)
        self.assertIn("irdma0:1", text)
        self.assertIn("VLLM_IB_HCA", text)

    def test_warn_multi_rdma_roce_mentions_vllm_roce_hca(self):
        captured = io.StringIO()
        with patch(
            "cluster_manager.active_rdma_ports",
            return_value=[("irdma0:1", "Ethernet"), ("usb4_rdma0:1", "Ethernet")],
        ), redirect_stdout(captured):
            cluster_manager.warn_multi_rdma("roce")
        self.assertIn("VLLM_ROCE_HCA", captured.getvalue())

    def test_transport_script_exports_reference_detect_helper_for_ib(self):
        exports = cluster_manager.transport_script_exports("infiniband")
        self.assertIn("export NCCL_IB_HCA=$(", exports)
        self.assertIn("export NCCL_IB_GID_INDEX=0", exports)
        self.assertIn("export NCCL_IB_DISABLE=0", exports)
        self.assertIn("export NCCL_SOCKET_IFNAME=$RDMA_IFACE", exports)
        exports_eth = cluster_manager.transport_script_exports("ethernet")
        self.assertIn("export NCCL_IB_DISABLE=1", exports_eth)
        self.assertNotIn("NCCL_IB_GID_INDEX", exports_eth)
        self.assertNotIn("NCCL_IB_HCA", exports_eth)

    def test_transport_script_exports_ib_honors_vllm_ib_hca_override(self):
        with patch.dict("os.environ", {"VLLM_IB_HCA": "mlx5_0:1"}):
            exports = cluster_manager.transport_script_exports("infiniband")
        self.assertIn("export NCCL_IB_HCA=mlx5_0:1", exports)
        self.assertNotIn("_detect_rdma_hca", exports)

    def test_transport_script_exports_roce_pins_rail_when_multiple_active(self):
        with patch(
            "cluster_manager.active_rdma_ports",
            return_value=[("irdma0:1", "Ethernet"), ("usb4_rdma0:1", "Ethernet")],
        ):
            exports = cluster_manager.transport_script_exports("roce")
        self.assertIn("export NCCL_IB_HCA=$(", exports)
        self.assertIn("_detect_rdma_hca 'Ethernet'", exports)

    @patch("cluster_manager.subprocess.run")
    def test_worker_ray_script_emits_selected_transport(self, run):
        with patch.dict(
            "os.environ", {"VLLM_CLUSTER_TRANSPORT": "infiniband"}
        ), patch("cluster_manager.detect_ib_hca", return_value="mlx4_0:1"):
            cluster_manager.setup_worker_node(
                "192.168.100.2", "192.168.100.1", "vllm-therock-gfx1151-dev"
            )
        script = run.call_args.kwargs["input"].decode()
        self.assertIn("transport=infiniband", script)
        self.assertIn("export NCCL_IB_GID_INDEX=0", script)
        self.assertIn("export NCCL_IB_DISABLE=0", script)

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
