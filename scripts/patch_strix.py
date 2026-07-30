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

    # Patch 1: vllm/platforms/__init__.py (amdsmi monkey patch — PROVEN working for 5 months)
    # Comment out real amdsmi imports and replace with pass stubs.
    # The actual amdsmi library doesn't work on Strix Halo APUs in containers.
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (amdsmi disabled, is_rocm forced True)")

    # Patch 1.5: vllm/platforms/rocm.py (amdsmi mock + force gfx1151)
    # Replace real amdsmi with a minimal mock that returns proper values for
    # the specific functions vLLM calls, so device memory queries fall through
    # to the torch.cuda fallback instead of returning a MagicMock.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add amdsmi mock header if not already present
        if 'sys.modules["amdsmi"] = _AmdsmiMock()' not in txt:
            header = (
                'import sys\n'
                'from unittest.mock import MagicMock\n'
                'class _AmdsmiMock:\n'
                '    def amdsmi_get_processor_handles(self):\n'
                '        raise RuntimeError("amdsmi not available on Strix Halo")\n'
                '    def amdsmi_get_gpu_memory_total(self, handle, mem_type):\n'
                '        raise RuntimeError("amdsmi not available on Strix Halo")\n'
                '    def amdsmi_get_gpu_gcn_arch(self, handle):\n'
                '        return "gfx1151"\n'
                '    def __getattr__(self, name):\n'
                '        return MagicMock()\n'
                'sys.modules["amdsmi"] = _AmdsmiMock()\n'
            )
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (_AmdsmiMock + forced gfx1151)")

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()
        
        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform", 
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend is_aiter_found_and_supported
        if "or on_gfx1x()" not in txt:
            txt = txt.replace("import on_mi3xx", "import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            
        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n', 
                r'\1 False\n', 
                txt, count=1, flags=re.DOTALL
            )
            
        # Disable AITER RMSNorm on gfx1x (CUDA Graph hang)
        if "is_rmsnorm_enabled" in txt:
            txt = re.sub(
                r'(def is_rmsnorm_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._RMSNORM_ENABLED)\n', 
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n', 
                txt, count=1, flags=re.DOTALL
            )
            
        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n', 
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n', 
                txt, count=1, flags=re.DOTALL
            )
            
        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support, FP8 linear empty, MoE disabled)")

    # Patch 3: rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms.rocm import on_mi3xx", "from vllm.platforms.rocm import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            p_fa.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

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
        
        # Legacy vLLM < 0.19 fallback
        if "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")" in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not getattr(self, 'on_gfx1x', lambda: False)():\n            custom_ops.append(\"+rms_norm\")"
            )
        
        # Modern vLLM 0.19+ struct (compilation_config.custom_ops)
        elif "compilation_config.custom_ops.append(\"+rms_norm\")" in txt:
            if "if not getattr(self, \"on_gfx1x\", lambda: False)():" not in txt:
                txt = re.sub(
                    r'(\s+)compilation_config\.custom_ops\.append\("\+rms_norm"\)',
                    r'\1if not getattr(self, "on_gfx1x", lambda: False)():\n\1    compilation_config.custom_ops.append("+rms_norm")',
                    txt
                )
                
        # Modern vLLM 0.19.2rc1+ IrOpPriorityConfig bypass
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )
            
        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops & IrOpPriorityConfig rms_norm bypassed on gfx1x)")

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

    # Patch 7: Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):", 
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

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
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError):\n"
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

    # Patch 9: Allow Triton MoE kernels on gfx11xx (Strix Halo)
    # vLLM recently capped MXFP4 Triton MoE kernels to < (11, 0) which excludes RDNA3.5 (11.x)
    for p_triton in [
        Path('vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py'),
        Path('vllm/model_executor/layers/fused_moe/oracle/mxfp4.py')
    ]:
        if p_triton.exists():
            txt = p_triton.read_text()
            if "cap.minor) < (11, 0)" in txt:
                txt = txt.replace("cap.minor) < (11, 0)", "cap.minor) < (12, 0)")
            if "capability() < (11, 0)" in txt:
                txt = txt.replace("capability() < (11, 0)", "capability() < (12, 0)")
            p_triton.write_text(txt)
            print(f" -> Patched {p_triton} (Triton MoE on gfx11xx)")

    # Patch 11: moe_wna16.py — fix tp_size AttributeError on RoutedExperts
    # vLLM refactored FusedMoE and moved tp_size out of RoutedExperts (the
    # weight container) into FusedMoEConfig. But moe_wna16_weight_loader still
    # accesses `layer.tp_size` which crashes AWQ MoE models (Qwen3.5 etc.)
    # that fall back from AWQMoeMarlin to the WNA16 path.
    # Fix: use get_tp_group().world_size which is always available.
    # https://github.com/vllm-project/vllm/issues/45403
    p_moe_wna16 = Path('vllm/model_executor/layers/quantization/moe_wna16.py')
    if p_moe_wna16.exists():
        txt = p_moe_wna16.read_text()
        if 'layer.tp_size' in txt:
            txt = txt.replace('layer.tp_size', 'get_tp_group().world_size')
            p_moe_wna16.write_text(txt)
            print(" -> Patched moe_wna16.py (layer.tp_size -> get_tp_group().world_size)")

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
