#!/usr/bin/env python3
"""Add opt-in gfx1x MXFP4 MoE tuning to conch-triton-kernels.

DeepSeek V4 uses conch's ``matmul_ogs`` path for its MXFP4 experts.  The
generic AMD heuristic is not tuned for the skinny matrices used during decode
on gfx1151.  This patch retains the stock path unless
``VLLM_GFX1X_MOE_TUNE=1`` is present in the worker environment.

The initial defaults mirror the production settings published by
AlexKGwyn/ds4-vllm-public.  Every tile parameter remains independently
overridable so the profile can be A/B tested without rebuilding the image.
"""

from __future__ import annotations

from pathlib import Path


MARKER = "# PATCHED: opt-in gfx1x MXFP4 MoE tuning"
OPT_FLAGS_TARGET = Path(
    "third_party/triton_kernels/matmul_ogs_details/opt_flags.py"
)


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_opt_flags(path: Path) -> bool:
    """Patch one installed conch ``opt_flags.py`` in place."""
    source = path.read_text()
    if MARKER in source:
        return False

    source = _replace_once(
        source,
        "from dataclasses import dataclass\n\nimport triton\n",
        "from dataclasses import dataclass\n"
        "import os\n\n"
        "import triton\n",
        "environment import",
    )
    source = _replace_once(
        source,
        "from triton_kernels.tensor import bitwidth\n\n\n@dataclass\n",
        "from triton_kernels.tensor import bitwidth\n\n\n"
        f"{MARKER}\n"
        "_GFX1X_MOE_CONFIG = None\n\n\n"
        "def _gfx1x_moe_config():\n"
        "    # Resolve lazily: Ray applies the final worker environment after\n"
        "    # importing parts of vLLM, but before the first model execution.\n"
        "    global _GFX1X_MOE_CONFIG\n"
        "    if _GFX1X_MOE_CONFIG is None:\n"
        "        if os.environ.get(\"VLLM_GFX1X_MOE_TUNE\") != \"1\":\n"
        "            _GFX1X_MOE_CONFIG = False\n"
        "        else:\n"
        "            _GFX1X_MOE_CONFIG = {\n"
        "                \"block_m\": int(os.environ.get(\"VLLM_GFX1X_MOE_BM\", \"16\")),\n"
        "                \"block_n\": int(os.environ.get(\"VLLM_GFX1X_MOE_BN\", \"32\")),\n"
        "                \"block_k\": int(os.environ.get(\"VLLM_GFX1X_MOE_BK\", \"256\")),\n"
        "                \"num_warps\": int(os.environ.get(\"VLLM_GFX1X_MOE_NW\", \"2\")),\n"
        "                \"num_stages\": int(os.environ.get(\"VLLM_GFX1X_MOE_NS\", \"2\")),\n"
        "                \"waves_per_eu\": int(os.environ.get(\"VLLM_GFX1X_MOE_WPE\", \"1\")),\n"
        "            }\n"
        "            print(\n"
        "                f\"[gfx1x_moe] enabled {_GFX1X_MOE_CONFIG}\",\n"
        "                flush=True,\n"
        "            )\n"
        "    return _GFX1X_MOE_CONFIG\n\n\n"
        "@dataclass\n",
        "gfx1x configuration helper",
    )
    source = _replace_once(
        source,
        "    # AMD-specific\n"
        "    target_kernel_kwargs = {\"waves_per_eu\": 0, \"matrix_instr_nonkdim\": 16, \"kpack\": 1}\n"
        "    epilogue_subtile = constraints.get('epilogue_subtile', None)\n",
        "    # AMD-specific\n"
        "    target_kernel_kwargs = {\"waves_per_eu\": 0, \"matrix_instr_nonkdim\": 16, \"kpack\": 1}\n\n"
        "    # DeepSeek V4's decode MoE uses skinny BF16 x MXFP4 matmuls.\n"
        "    # Keep generic AMD behavior for every other dtype, architecture,\n"
        "    # large-M tile, and whenever the explicit model policy is off.\n"
        "    gfx1x_moe = _gfx1x_moe_config()\n"
        "    if (\n"
        "        gfx1x_moe\n"
        "        and get_rdna_version() in (3, 4)\n"
        "        and block_m < 128\n"
        "        and bitwidth(lhs_dtype) == 16\n"
        "        and bitwidth(rhs_dtype) == 4\n"
        "        and precision_config.weight_scale is not None\n"
        "    ):\n"
        "        block_m = gfx1x_moe[\"block_m\"]\n"
        "        block_n = gfx1x_moe[\"block_n\"]\n"
        "        block_k = gfx1x_moe[\"block_k\"]\n"
        "        num_warps = gfx1x_moe[\"num_warps\"]\n"
        "        num_stages = gfx1x_moe[\"num_stages\"]\n"
        "        target_kernel_kwargs = {\n"
        "            \"waves_per_eu\": gfx1x_moe[\"waves_per_eu\"],\n"
        "            \"kpack\": 1,\n"
        "        }\n\n"
        "    epilogue_subtile = constraints.get('epilogue_subtile', None)\n",
        "AMD kernel configuration",
    )
    path.write_text(source)
    return True


def installed_target() -> Path:
    from importlib.util import find_spec

    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed vllm package")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_root / OPT_FLAGS_TARGET


def main() -> None:
    target = installed_target()
    if not target.exists():
        raise RuntimeError(f"missing conch-triton-kernels target: {target}")
    changed = patch_opt_flags(target)
    print(f"{'Patched' if changed else 'Already patched'} {target}")


if __name__ == "__main__":
    main()
