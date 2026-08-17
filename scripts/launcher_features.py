"""Shared optional serve features for single-host and Ray launchers."""

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OPT_WARMUP_SCRIPT = Path("/opt/vllm_warmup.py")
DEFAULT_COLLECTIVE_PROFILE_DIR = Path.home() / ".cache" / "vllm" / "profiles"


def speculative_config_args(config, enabled):
    """Return the vLLM CLI arguments for a model's speculative configuration."""
    speculative_config = config.get("speculative_config")
    if not enabled or speculative_config is None:
        return []
    return [
        "--speculative-config",
        json.dumps(speculative_config, separators=(",", ":"), sort_keys=True),
    ]


def collective_profiler_args(enabled, profile_dir=None):
    """Return an opt-in multi-rank Torch profiler configuration.

    Profiling remains dormant until the API's ``/start_profile`` route is
    called. Shape recording identifies the real TP collective tensors; stacks
    and memory tracking stay off to keep the controlled run reasonably small.
    """
    if not enabled:
        return []
    directory = Path(profile_dir or DEFAULT_COLLECTIVE_PROFILE_DIR).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "profiler": "torch",
        "torch_profiler_dir": str(directory),
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_use_gzip": True,
        "ignore_frontend": True,
    }
    return [
        "--profiler-config",
        json.dumps(config, separators=(",", ":"), sort_keys=True),
    ]


def get_warmup_script():
    """Resolve the installed image path, with a checkout fallback for tests."""
    if OPT_WARMUP_SCRIPT.exists():
        return OPT_WARMUP_SCRIPT
    return SCRIPT_DIR / "vllm_warmup.py"


def launch_automatic_warmup(model_id, port, config, env):
    """Start a best-effort helper that waits for the API and warms real requests."""
    warmup = config.get("warmup")
    if warmup is None:
        return None

    helper_env = env.copy()
    helper_env.update(
        {
            "VLLM_WARMUP_MODEL": model_id,
            "VLLM_WARMUP_PORT": str(port),
            "VLLM_WARMUP_PROMPT_TOKENS": str(
                warmup.get("prompt_tokens", 0)
            ),
            "VLLM_WARMUP_PARENT_PID": str(os.getpid()),
            "VLLM_WARMUP_READY_TIMEOUT": str(
                warmup.get("ready_timeout_seconds", 1200)
            ),
        }
    )
    return subprocess.Popen(
        [sys.executable, str(get_warmup_script())],
        env=helper_env,
        close_fds=True,
    )
