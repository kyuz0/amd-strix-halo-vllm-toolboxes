#!/usr/bin/env python3
"""Fail-closed source audit for version-sensitive native vLLM interfaces.

This is not a runtime correctness test. It prevents an automatic stable-vLLM
update from silently building after the native DSpark or DeepSeek integration
has moved underneath the local gfx1151 patches. The required human and GPU
checks are recorded in docs/VLLM_PATCH_MANIFEST.md.
"""

from pathlib import Path


REQUIRED_NATIVE_INTERFACES = {
    "vllm/config/speculative.py": (
        'DSparkModelTypes = Literal["dspark"]',
        'def use_dspark(self) -> bool:',
        'self.method == "dspark"',
    ),
    "vllm/models/deepseek_v4/__init__.py": (
        "from .amd.dspark import",
        "DSparkDeepseekV4ForCausalLM",
    ),
    "vllm/models/deepseek_v4/amd/dspark.py": (
        "class DSparkDeepseekV4ForCausalLM",
        "config.dspark_target_layer_ids",
        "DeepseekV4DecoderLayer",
        "has_own_lm_head = False",
    ),
    "vllm/models/deepseek_v4/amd/model.py": (
        "aux_hidden_states: list[torch.Tensor]",
        "if (idx + 1) in self.aux_hidden_state_layers:",
        "aux_hidden_states.append(aux_recon.mean(dim=1))",
    ),
    "vllm/v1/worker/gpu/spec_decode/__init__.py": (
        'speculative_config.method == "dspark"',
        "DSparkSpeculator",
    ),
    "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py": (
        "class DSparkSpeculator(DFlashSpeculator):",
        "def _sample_sequential(",
        "load_dspark_model",
    ),
    "vllm/v1/worker/gpu/spec_decode/dspark/utils.py": (
        "def load_dspark_model(",
        'raise NotImplementedError("DSpark does not support pipeline parallelism.")',
        'draft_model, "has_own_lm_head"',
    ),
    "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py": (
        'getattr(hf_config, "dspark_target_layer_ids", None)',
        "layer_ids = [i + 1 for i in dspark_layer_ids]",
    ),
    "vllm/v1/attention/backends/mla/sparse_swa.py": (
        "self.is_dspark = spec_config is not None and spec_config.use_dspark()",
    ),
}


def verify_native_interfaces(root: Path = Path(".")) -> None:
    failures = []
    for relative_path, anchors in REQUIRED_NATIVE_INTERFACES.items():
        path = root / relative_path
        if not path.exists():
            failures.append(f"missing path: {relative_path}")
            continue
        source = path.read_text()
        for anchor in anchors:
            if anchor not in source:
                failures.append(f"{relative_path}: missing interface {anchor!r}")

    if failures:
        details = "\n - ".join(failures)
        raise RuntimeError(
            "Unsupported vLLM source layout. Audit the candidate release against "
            "docs/VLLM_PATCH_MANIFEST.md before rebasing the compatibility guard:\n"
            f" - {details}"
        )


if __name__ == "__main__":
    verify_native_interfaces()
    print("Verified native vLLM DSpark/DeepSeek compatibility interfaces.")
