import sys
import re
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 0: csrc/spinloop.cpp (clang-compatible mwaitx include)
    # spinloop.cpp includes <mwaitxintrin.h> directly. ROCm clang rejects that
    # with a hard #error ("Never use <mwaitxintrin.h> directly; include
    # <x86intrin.h> instead."). GCC tolerates it, so vLLM upstream CI never sees
    # the break — but this toolbox builds vLLM with CC/CXX=ROCm clang (for ABI
    # alignment with PyTorch), so the spinloop target fails to compile.
    # <x86intrin.h> is the umbrella header accepted by both compilers and still
    # exposes the MONITORX/MWAITX intrinsics. Guarded so it no-ops once vLLM
    # fixes it upstream or removes the file.
    p_spinloop = Path('csrc/spinloop.cpp')
    if p_spinloop.exists():
        txt = p_spinloop.read_text()
        if '#include <mwaitxintrin.h>' in txt:
            txt = txt.replace('#include <mwaitxintrin.h>',
                              '#include <x86intrin.h>')
            p_spinloop.write_text(txt)
            print(" -> Patched csrc/spinloop.cpp (mwaitxintrin.h -> x86intrin.h for clang)")

    # NOTE: the former Patch 1 / Patch 1.5 (comment out `import amdsmi`, stub it with a
    # MagicMock, and force the GCN arch to gfx1151) are GONE.
    #
    # They existed because "the actual amdsmi library doesn't work on Strix Halo APUs in
    # containers". That is no longer true: the amdsmi python bindings ship with ROCm
    # (/opt/rocm/share/amd_smi) and work fine on gfx1151 -- they were simply never
    # installed, so `from amdsmi import ...` failed and everything had to be stubbed.
    # The Dockerfile now installs them, and vLLM resolves the arch and device name on its
    # own (verified: _query_gcn_arch_from_amdsmi() -> 'gfx1151', device name ->
    # AMD_Radeon_8060S).
    #
    # Keeping the MagicMock was also actively harmful: a mock call never raises, so vLLM's
    # `try: <amdsmi> except: <fallback>` helpers never reached their fallback and the mock
    # leaked into real use. vLLM >= 0.25.1 calls get_device_total_memory() from
    # get_batch_defaults() at engine-config time and does `device_memory >= 70*GiB`, which
    # raised `TypeError: '>=' MagicMock vs int` and crashed startup for EVERY model.
    #
    # The one genuine bug left is in amdsmi itself (it reports the 512 MiB BIOS VRAM
    # carveout as the VRAM total on APUs); that is worked around in scripts/patch_amdsmi.py,
    # at the layer where the bug actually is, so vLLM needs no memory patch at all.

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1151, disable unsafe gfx1x ops)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()
        
        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform", 
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend only the central AITER capability gate. v0.26.0 used on_mi3xx();
        # v0.27.0rc1 changed it to get_cdna_version() > 2. Support both layouts,
        # scope the override to gfx1151, and fail the build if neither known anchor
        # is present instead of printing a false-success message.
        aiter_gate_variants = (
            (
                "        from vllm.platforms.rocm import on_mi3xx\n\n"
                "        return on_mi3xx()\n",
                "        from vllm.platforms.rocm import on_gfx1151, on_mi3xx\n\n"
                "        return on_mi3xx() or on_gfx1151()\n",
            ),
            (
                "        from vllm.platforms.rocm import get_cdna_version\n\n"
                "        return get_cdna_version() > 2\n",
                "        from vllm.platforms.rocm import get_cdna_version, on_gfx1151\n\n"
                "        return get_cdna_version() > 2 or on_gfx1151()\n",
            ),
        )
        if not any(new in txt for _, new in aiter_gate_variants):
            matches = [(old, new) for old, new in aiter_gate_variants if old in txt]
            if len(matches) != 1:
                raise RuntimeError(
                    "Unsupported vLLM is_aiter_found_and_supported() layout; "
                    "refusing to build without the gfx1151 AITER gate"
                )
            old, new = matches[0]
            txt = txt.replace(old, new, 1)

        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n',
                r'\1 False\n',
                txt, count=1, flags=re.DOTALL
            )

        # NOTE: the former "disable AITER RMSNorm" patch (is_rmsnorm_enabled) was REMOVED.
        # vLLM >=0.26.0 deleted that method; AITER RMSNorm is now gated solely by the
        # IrOpPriorityConfig bypass in Patch 5 (`rms_norm = ["aiter"]+default` -> `default`
        # on gfx1x). Re-adding an _aiter_ops gate here would be a silent no-op. (v0.26.0 audit.)

        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n', 
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n', 
                txt, count=1, flags=re.DOTALL
            )
            
        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1151 AITER gate, FP8 linear/MoE safety)")

    # Patch 2.5: DeepSeek V4 has two private FP8 linear fast paths which check the
    # broad AITER toggle instead of the FP8-linear capability gate. That bypasses
    # Patch 2 on gfx1151, preshuffles the weights, and eventually calls AITER's
    # unsupported float8_e4m3fn quantizer. Route both decisions through
    # is_linear_fp8_enabled(); on gfx1151 Patch 2 forces that gate off, leaving
    # the broad AITER toggle available to the sparse-indexer helpers while the
    # linear layers use vLLM's Triton fallback.
    dsv4_linear_gates = (
        (
            Path("vllm/models/deepseek_v4/amd/model.py"),
            "self._gateup = rocm_aiter_ops.is_enabled()",
            "self._gateup = rocm_aiter_ops.is_linear_fp8_enabled()",
        ),
        (
            Path("vllm/models/deepseek_v4/amd/rocm.py"),
            "if not rocm_aiter_ops.is_enabled():",
            "if not rocm_aiter_ops.is_linear_fp8_enabled():",
        ),
    )
    for path, old_gate, new_gate in dsv4_linear_gates:
        if not path.exists():
            continue
        txt = path.read_text()
        if new_gate not in txt:
            if txt.count(old_gate) != 1:
                raise RuntimeError(
                    f"Unsupported DeepSeek V4 AITER linear gate in {path}; "
                    "refusing to build without the gfx1151 Triton fallback"
                )
            txt = txt.replace(old_gate, new_gate, 1)
            path.write_text(txt)
        print(f" -> Patched {path} (AITER FP8 linear gate respected)")

    # Patch 3.5: unquantized.py (Hard-block AITER MoE forced override on gfx1x)
    p_unquant = Path('vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    if p_unquant.exists():
        txt = p_unquant.read_text()
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                'if envs.is_set("VLLM_ROCM_USE_AITER")',
                'from vllm.platforms.rocm import on_gfx1x\n    if envs.is_set("VLLM_ROCM_USE_AITER")'
            )
            txt = txt.replace(
                'if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:',
                'if getattr(on_gfx1x, "__call__", lambda: False)() or not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:'
            )
            p_unquant.write_text(txt)
            print(" -> Patched unquantized.py (Blocked AITER MoE override on gfx1x)")

    # Patch 5: custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()

        # RMSNorm off AITER on gfx1x via the IrOpPriorityConfig list — the only live anchor
        # in vLLM >=0.26.0. The former legacy `custom_ops.append("+rms_norm")` variants
        # (vLLM <0.19, and the 0.19+ compilation_config form) were REMOVED: dead anchors in
        # the versions we build, and the 0.19+ string now only appears in an unrelated SP/PP
        # path in config/vllm.py that must NOT be touched. (v0.26.0 audit.)
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )

        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (IrOpPriorityConfig rms_norm bypassed on gfx1x)")

    # Patch 6: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (duplicate pattern bypass)
    p_fusion = Path('vllm/compilation/passes/fusion/rocm_aiter_fusion.py')
    if p_fusion.exists():
        txt = p_fusion.read_text()
        if "skip_duplicates=True" not in txt:
            txt = re.sub(
                r"(pm\.register_replacement\s*\((?:(?!\bpm\.register_replacement\b).)*?)pm_pass(\s*[\),])", 
                r"\1pm_pass, skip_duplicates=True\2", 
                txt, flags=re.DOTALL
            )
            p_fusion.write_text(txt)
            print(" -> Patched vllm/compilation/passes/fusion/rocm_aiter_fusion.py (skip_duplicates)")

    # NOTE: the former "Triton AttrsDescriptor repr" patch was REMOVED — the AttrsDescriptor
    # class no longer exists in Triton >=3.6.0 (deleted upstream), so the patch was a
    # permanent silent no-op on the triton we ship. (v0.26.0 / triton-3.6.0 audit.)

    # Patch 7: aiter JIT path fix — aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 8: flash_attn_interface.py — make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    hard_import_bare = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"
    
    def _patch_flash_interface(fa_iface):
        txt = fa_iface.read_text()
        if hard_import_bare not in txt or "except (ImportError" in txt:
            return False
        # Detect indentation of the original import line
        m = re.search(r'^( *)' + re.escape(hard_import_bare), txt, re.MULTILINE)
        if not m:
            return False
        indent = m.group(1)
        original_line = indent + hard_import_bare
        soft_import = (
            f"{indent}try:\n"
            f"{indent}    {hard_import_bare}\n"
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError, RuntimeError):\n"
            f"{indent}    flash_attn_gpu = None"
        )
        txt = txt.replace(original_line, soft_import)
        fa_iface.write_text(txt)
        print(f" -> Patched {fa_iface} (aiter import made resilient)")
        return True

    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                _patch_flash_interface(fa_iface)
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            _patch_flash_interface(fa_iface)

    # NOTE: the former "Triton MoE on gfx11xx" cap-bump patch was REMOVED. vLLM >=0.26.0
    # already enables the OAI Triton MoE kernels on gfx11xx via the ROCm on_gfx1x()/on_gfx9()
    # gate (_triton_kernel_moe_supports_current_device); the old `< (11, 0)` cap is gone from
    # oracle/mxfp4.py, and the only remaining `(11, 0)` match sits in a CUDA-only branch of
    # gpt_oss_triton_kernels_moe.py where our replace mis-fired (inert on ROCm, but a wrong
    # edit to the CUDA gate). Dropped entirely. (v0.26.0 audit.)

    # Patch 11: RocmPlatform.is_integrated_gpu override (smart UMA detection)
    # Upstream vLLM PR #35356 (merged 2026-04-13) added Platform.is_integrated_gpu()
    # and made MemorySnapshot.measure() use psutil.virtual_memory().available for free
    # memory on UMA devices (hipMemGetInfo ignores OS-reclaimable memory there -> free
    # is over-reported -> over-allocation into swap; vLLM #35313, this repo's #65). But
    # the override was implemented for CUDA only; RocmPlatform inherited the base False,
    # so AMD APUs never took that path. HIP reports integrated=1 for gfx APUs (Strix
    # Halo/Point, MI300A) and torch surfaces it as is_integrated (verified =1 on
    # gfx1151), so mirror CudaPlatformBase. Replaces the former ROCM-21812 GTT/sysfs
    # +8GiB heuristic (removed above): the 50% APU VRAM clamp it worked around is fixed
    # in current ROCm (full GTT total is reported). No-ops once vLLM ships a ROCm
    # override upstream.
    p_rocm_isint = Path('vllm/platforms/rocm.py')
    if p_rocm_isint.exists():
        txt = p_rocm_isint.read_text()
        if 'def is_integrated_gpu' not in txt and 'class RocmPlatform(Platform):' in txt:
            method = (
                '    @classmethod\n'
                '    def is_integrated_gpu(cls, device_id: int = 0) -> bool:\n'
                '        # AMD APUs (Strix Halo/Point, MI300A) are UMA: HIP reports\n'
                '        # integrated=1, surfaced by torch as is_integrated. Mirrors\n'
                '        # CudaPlatformBase so MemorySnapshot uses the psutil free-memory\n'
                '        # path on unified memory (avoids over-allocation / swap).\n'
                '        try:\n'
                '            return bool(torch.cuda.get_device_properties(device_id).is_integrated)\n'
                '        except Exception:\n'
                '            return False\n\n'
            )
            head, sep, tail = txt.partition('class RocmPlatform(Platform):')
            tail = tail.replace('    @classmethod\n', method + '    @classmethod\n', 1)
            txt = head + sep + tail
            p_rocm_isint.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (RocmPlatform.is_integrated_gpu UMA override)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
