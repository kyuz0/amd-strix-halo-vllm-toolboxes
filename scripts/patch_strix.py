import sys
import re
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

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

    # Patch 1.5: vllm/platforms/rocm.py (MagicMock amdsmi + force gfx1151)
    # Prepend MagicMock so any remaining amdsmi references in rocm.py silently succeed.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add MagicMock header if not already present
        if 'sys.modules["amdsmi"] = MagicMock()' not in txt:
            header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (MagicMock amdsmi + forced gfx1151)")

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

    # Patch 10: ROCM-21812 APU VRAM Dynamic Margin Patch
    # Explanation: ROCm nightly builds introduced a 50% APU VRAM clamp to prevent
    # OOM kernel panics on headless hosts. This broke vLLM large model loading.
    # This patch intercepts PyTorch memory bounds and dynamically proxies the 
    # real amdgpu hardware GTT limits, minus a strict 8GB OS safety margin.
    # By symmetrically carving the OS margin from the top of the GTT ceiling, 
    # vLLM's memory profiler allocates flawlessly while guaranteeing the OS stays alive,
    # regardless of the specific GTT allocation size on the host.
    # Ref: https://github.com/ROCm/rocm-systems/pull/5113
    # TODO: Remove this patch block entirely once PR #5113 merges and is 
    # incorporated into the ROCm nightly tarballs used by this toolbox.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if "_patched_mem_info" not in txt:
            mem_patch = '''
# --- ROCM-21812 VRAM DYNAMIC PATCH ---
import torch
import glob
import os

try:
    _orig_mem_info = torch.cuda.mem_get_info
    _orig_get_dev_prop = torch.cuda.get_device_properties

    class MockCudaDeviceProperties:
        def __init__(self, prop, override_total):
            self._prop = prop
            self.total_memory = override_total
        def __getattr__(self, name):
            return getattr(self._prop, name)
        def __dir__(self):
            return dir(self._prop)

    def _patched_mem_info(device=None):
        free, total = _orig_mem_info(device)
        try:
            # On APUs, ROCm clamps total to 50% limit. We need the real GTT limits.
            if total < 70 * 1024**3: 
                drm_cards = glob.glob('/sys/class/drm/card*/device/mem_info_gtt_total')
                if drm_cards:
                    card_dir = os.path.dirname(drm_cards[0])
                    with open(os.path.join(card_dir, 'mem_info_gtt_total'), 'r') as f:
                        gtt_total = int(f.read().strip())
                    with open(os.path.join(card_dir, 'mem_info_gtt_used'), 'r') as f:
                        gtt_used = int(f.read().strip())
                    
                    # Symmetrically carve 8GB off the TOP of the device perfectly.
                    safe_ceiling = gtt_total - (8 * 1024**3)
                    
                    real_total = safe_ceiling
                    real_free = max(0, safe_ceiling - gtt_used)
                    
                    total = max(total, real_total)
                    free = real_free
        except Exception as e:
            pass
        return int(free), int(total)

    def _patched_get_dev_prop(device=None):
        prop = _orig_get_dev_prop(device)
        free, total = _patched_mem_info(device)
        if hasattr(prop, 'total_memory') and prop.total_memory < total:
            return MockCudaDeviceProperties(prop, total)
        return prop

    torch.cuda.mem_get_info = _patched_mem_info
    torch.cuda.get_device_properties = _patched_get_dev_prop
except Exception:
    pass
# ---------------------------
'''
            txt = mem_patch + txt
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (ROCM-21812 APU VRAM Dynamic Margin)")

    # Patch 11: vllm/transformers_utils/gguf_utils.py
    # extract_vision_config_from_gguf only special-cases Gemma3, and
    # maybe_patch_hf_config_from_gguf only wires the vision tower when
    # model_type is gemma3{,_text}. Add projector branches for nemotron_v2_vl
    # (Radio, used by NanoNemotronVL) and pixtral (Mistral3), and plumb the
    # vision_config into the parent for mistral3/nemotron_h{,_moe} so an
    # mmproj.gguf sibling is actually exposed at load time.
    p_gguf = Path('vllm/transformers_utils/gguf_utils.py')
    if p_gguf.exists():
        txt = p_gguf.read_text()

        # Branch in extract_vision_config_from_gguf
        if '"nemotron_v2_vl"' not in txt and 'projector_type == VisionProjectorType.GEMMA3' in txt:
            txt = txt.replace(
                '    if projector_type == VisionProjectorType.GEMMA3:\n'
                '        # Gemma3 doesn\'t use the vision pooling head (multihead attention)\n'
                '        # This is a vLLM-specific parameter used in SiglipVisionTransformer\n'
                '        config_params["vision_use_head"] = False\n'
                '        logger.info("Detected Gemma3 projector, disabling vision pooling head")',
                '    if projector_type == VisionProjectorType.GEMMA3:\n'
                '        config_params["vision_use_head"] = False\n'
                '        logger.info("Detected Gemma3 projector, disabling vision pooling head")\n'
                '    elif projector_type == "nemotron_v2_vl":\n'
                '        # Nemotron-3 Nano Omni: C-RADIOv4-H + 2-layer MLP projector. The vision\n'
                '        # tower has its own pooling so we disable the SigLIP head here too.\n'
                '        config_params["vision_use_head"] = False\n'
                '        logger.info("Detected nemotron_v2_vl projector (Nano Omni / Radio)")\n'
                '    elif projector_type == "pixtral":\n'
                '        # Mistral3 Pixtral: 48-layer ViT, image up to 1540px, patch 14.\n'
                '        # The projector handles its own break tokens; no SigLIP pooling head.\n'
                '        config_params["vision_use_head"] = False\n'
                '        logger.info("Detected pixtral projector (Mistral3)")'
            )

        # Patch maybe_patch_hf_config_from_gguf to also wire Mistral3 / NemotronH.
        # Strategy: keep the existing Gemma3 branch verbatim; append additional
        # branches that run only when model_type matches and a vision_config was
        # successfully extracted.
        if "is_mistral3 = hf_config.model_type" not in txt and "is_gemma3 = hf_config.model_type" in txt:
            txt = txt.replace(
                '        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")\n'
                '        if vision_config is not None and is_gemma3:\n'
                '            new_hf_config = Gemma3Config(\n'
                '                text_config=text_config,\n'
                '                vision_config=vision_config,\n'
                '                architectures=["Gemma3ForConditionalGeneration"],\n'
                '            )\n'
                '            hf_config = new_hf_config\n',
                '        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")\n'
                '        is_mistral3 = hf_config.model_type in ("mistral3", "ministral3")\n'
                '        is_nemotron_h_omni = hf_config.model_type in (\n'
                '            "nemotron_h_moe", "nemotron_h", "nemotron_v2_vl",\n'
                '        )\n'
                '        if vision_config is not None and is_gemma3:\n'
                '            new_hf_config = Gemma3Config(\n'
                '                text_config=text_config,\n'
                '                vision_config=vision_config,\n'
                '                architectures=["Gemma3ForConditionalGeneration"],\n'
                '            )\n'
                '            hf_config = new_hf_config\n'
                '        elif vision_config is not None and is_mistral3:\n'
                '            # Mistral3 keeps text_config under .text_config and the vision\n'
                '            # tower under .vision_config. Side-load both onto the parent.\n'
                '            try:\n'
                '                hf_config.vision_config = vision_config\n'
                '                hf_config.architectures = ["Mistral3ForConditionalGeneration"]\n'
                '                logger.info("Patched HF config for Mistral3 GGUF (Pixtral mmproj)")\n'
                '            except Exception as _e:\n'
                '                logger.warning("Mistral3 mmproj wiring failed: %s", _e)\n'
                '        elif vision_config is not None and is_nemotron_h_omni:\n'
                '            # NanoNemotronVL expects vision_config under .vision_config and\n'
                '            # registers the multimodal head from architectures.\n'
                '            try:\n'
                '                hf_config.vision_config = vision_config\n'
                '                if not getattr(hf_config, "architectures", None):\n'
                '                    hf_config.architectures = ["NemotronH_Nano_Omni_Reasoning_V3"]\n'
                '                logger.info("Patched HF config for Nemotron-Omni GGUF (nemotron_v2_vl mmproj)")\n'
                '            except Exception as _e:\n'
                '                logger.warning("Nemotron-Omni mmproj wiring failed: %s", _e)\n'
            )

        p_gguf.write_text(txt)
        print(" -> Patched vllm/transformers_utils/gguf_utils.py (Patch_11: nemotron_v2_vl + pixtral GGUF projector)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
