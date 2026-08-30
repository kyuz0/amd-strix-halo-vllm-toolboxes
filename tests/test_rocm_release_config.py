import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RocmReleaseConfigTests(unittest.TestCase):
    def test_dockerfile_defaults_to_rocm_10_stable_gfx1151_stack(self):
        dockerfile = (ROOT / "Dockerfile.ubuntu-repoamd").read_text()
        self.assertIn(
            "ARG ROCM_WHL=https://stable.repo.amd.com/rocm/whl-next/",
            dockerfile,
        )
        self.assertIn("ARG TORCH_VERSION=2.11.0", dockerfile)
        self.assertIn("ARG ROCM_VERSION=10.0.0", dockerfile)
        self.assertIn("libamd_smi.so.27", dockerfile)
        self.assertNotIn("libamd_smi.so.26", dockerfile)

    def test_workflow_resolves_complete_stack_and_stable_vllm(self):
        workflow = (
            ROOT / ".github" / "workflows" / "build-ubuntu-stable.yml"
        ).read_text()
        self.assertIn(
            "ROCM_WHL: https://stable.repo.amd.com/rocm/whl-next/",
            workflow,
        )
        self.assertIn('<<< "$DEV_LIST"', workflow)
        self.assertIn("grep -E '^v[0-9]+\\.[0-9]+\\.[0-9]+$'", workflow)
        self.assertIn(
            "github.event.inputs.vllm_ref == '' && "
            "github.event.inputs.image_tag == ''",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
