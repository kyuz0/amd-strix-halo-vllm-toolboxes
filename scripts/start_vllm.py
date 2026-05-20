#!/usr/bin/env python3
import sys
import os
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

# Add benchmarks dir to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
OPT_DIR = Path("/opt")


# Check /opt first (Container), then local fallback for results file location
if (OPT_DIR / "run_vllm_bench.py").exists():
    sys.path.append(str(OPT_DIR))
else:
    sys.path.append(str(BENCH_DIR))
    # Also ensure current script dir is in path for local 'models' import if not already
    sys.path.append(str(SCRIPT_DIR))

try:
    import models
    MODEL_TABLE = models.MODEL_TABLE
    MODELS_TO_RUN = models.MODELS_TO_RUN
except ImportError:
    print("Error: Could not import models.py config.")
    sys.exit(1)

if (OPT_DIR / "max_context_results.json").exists():
    RESULTS_FILE = OPT_DIR / "max_context_results.json"
else:
    RESULTS_FILE = BENCH_DIR / "max_context_results.json"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

def detect_gpus():
    """Detects AMD GPUs via rocm-smi or /dev/dri."""
    try:
        # Try rocm-smi first
        res = subprocess.run(["rocm-smi", "--showid", "--csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            count = res.stdout.count("GPU")
            if count > 0: return count
    except: pass
    
    # Fallback to /dev/dri/render*
    try:
        return len(list(Path("/dev/dri").glob("renderD*")))
    except:
        return 1

def get_discovered_models():
    """
    Returns all models defined in models.py that are compatible with the current hardware constraints.
    """
    gpu_count = detect_gpus()
    compatible_models = []
    
    for m in MODELS_TO_RUN:
        if m in MODEL_TABLE:
            valid_tps = MODEL_TABLE[m].get("valid_tp", [1])
            min_required = min(valid_tps)
            if min_required <= gpu_count:
                compatible_models.append(m)
                
    return compatible_models

# Refresh the list of models to run based on what we found
MODELS_TO_RUN = get_discovered_models()

def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Please install it (apt-get install dialog).")
        sys.exit(1)

def get_verified_config(model_id, tp_size, max_seqs):
    """
    Reads max_context_results.json to find the best verified configuration.
    Returns dict: {'ctx': int, 'util': float}
    """
    default_config = {
        "ctx": "auto",
        "util": float(models.GPU_UTIL),
    }
    
    if not RESULTS_FILE.exists():
        return default_config

    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        # Filter for Model + TP + Sequences
        matches = [r for r in data 
                  if r["model"] == model_id 
                  and r["tp"] == tp_size 
                  and r["max_seqs"] == max_seqs 
                  and r["status"] == "success"]
        
        if not matches:
            # Fallback 1: Try finding match with SAME TP but ANY Sequences (e.g. 1) to get base context?
            # Actually, safer to fallback to default or try finding nearest sequence?
            # Let's try finding exact match first. If fail, return default.
            return default_config
            
        # Sort by Util desc, then Context desc
        # We prefer higher utilization if available (performance), as long as it is verified success
        matches.sort(key=lambda x: (float(x["util"]), x["max_context_1_user"]), reverse=True)
        
        best = matches[0]
        return {
            "ctx": best["max_context_1_user"],
            "util": float(best["util"])
        }
        
    except Exception as e:
        return default_config

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def nuke_vllm_cache():
    """Removes vLLM cache directory to fix potential graph/incompatibility issues."""
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            print(f"Clearing vLLM cache at {cache}...", end="", flush=True)
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            print(" Done.")
            time.sleep(1)
        except Exception as e:
            print(f" Failed: {e}")

def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]

    # Validate stringly-typed fields here so a typo surfaces with the
    # offending model_id rather than as an opaque vllm argparse error.
    _VALID_ENUMS = {
        "tokenizer_mode": {"auto", "mistral", "slow", "custom"},
        "config_format": {"auto", "mistral", "hf"},
        "load_format": {
            "auto", "mistral", "safetensors", "npcache", "dummy",
            "tensorizer", "runai_streamer", "sharded_state", "gguf",
            "bitsandbytes", "pt",
        },
    }
    for _f, _valid in _VALID_ENUMS.items():
        _v = config.get(_f)
        if _v and _v not in _valid:
            raise ValueError(
                f"models.py {model_id!r}: {_f}={_v!r} not in {sorted(_valid)}"
            )

    # Static Config
    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1

    # Defaults
    current_tp = min(gpu_count, max_tp)
    # Seed concurrency from the entry's max_num_seqs. Speculative-decode entries
    # rely on this >= 4 to have a batch to amortize the draft cost.
    current_seqs = int(config.get("max_num_seqs", 1))
    
    # Initial Lookup
    verified = get_verified_config(model_id, current_tp, current_seqs)
    current_ctx = verified["ctx"]
    current_util = verified["util"]
    
    clear_cache = True  # Default ON: stale graphs from version upgrades cause crashes
    use_eager = config.get("enforce_eager", False) # Default to model config, usually False
    attn_backends = ["Triton", "ROCm (CK)", "AITER"]
    current_attn_backend = "Triton" # Default to Triton

    # Per-model attention backend override.
    _attn_alias = {"TRITON_ATTN": "Triton", "ROCM_ATTN": "ROCm (CK)", "AITER": "AITER"}
    _attn_override = config.get("attention_backend")
    if _attn_override:
        if _attn_override not in _attn_alias:
            raise ValueError(
                f"models.py {model_id!r}: attention_backend={_attn_override!r} "
                f"not in {list(_attn_alias)}"
            )
        current_attn_backend = _attn_alias[_attn_override]
    
    name = model_id.split("/")[-1]
    
    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        
        menu_args = [
            "--clear", "--backtitle", f"AMD Strix Halo vLLM Launcher (GPUs: {gpu_count})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "22", "65", "9",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx} (Verified)",
            "4", f"GPU Utilization:      {current_util} (Verified)",
            "5", f"Attention Backend:    {current_attn_backend}",
            "6", f"Erase vLLM Cache:     {cache_status}",
            "7", f"Force Eager Mode:     {eager_status}",
            "8", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False # Back/Cancel
        
        if choice == "1":
            # TP Selection
            new_tp = run_dialog([
                "--title", "Tensor Parallelism",
                "--rangebox", f"Set TP Size (1-{max_tp})", "10", "40", "1", str(max_tp), str(current_tp)
            ])
            if new_tp: 
                new_tp_int = int(new_tp)
                if new_tp_int != current_tp:
                    current_tp = new_tp_int
                    # RE-CALCULATE Config
                    verified = get_verified_config(model_id, current_tp, current_seqs)
                    current_ctx = verified["ctx"]
                    current_util = verified["util"]
            
        elif choice == "2":
            # Max Seqs Selection
            new_seqs = run_dialog([
                "--title", "Concurrent Requests",
                "--menu", "Select Max Concurrent Requests:", "12", "40", "4",
                "1", "1 (Latency Focus)",
                "4", "4 (Balanced)",
                "8", "8 (Throughput)",
                "16", "16 (Max Load)"
            ])
            if new_seqs:
                current_seqs = int(new_seqs)
                # RE-CALCULATE Config based on new concurrency
                verified = get_verified_config(model_id, current_tp, current_seqs)
                current_ctx = verified["ctx"]
                current_util = verified["util"]

        elif choice == "3":
            # Configured Length Override
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox", f"Override verified limit ({current_ctx}):", "10", "40", str(current_ctx)
            ])
            if new_ctx: current_ctx = int(new_ctx)

        elif choice == "4":
             # Util Override
             pass 

        elif choice == "5":
            # Cycle Attention Backend
            idx = attn_backends.index(current_attn_backend)
            current_attn_backend = attn_backends[(idx + 1) % len(attn_backends)]

        elif choice == "6":
            # Toggle Cache
            if not clear_cache:
                # Enabling it -> Show Warning
                warn_msg = (
                    "WARNING: Erasing the vLLM cache will remove the compiled compute graphs.\n\n"
                    "This is useful if you are experiencing crashes, 'invalid graph' errors,\n"
                    "or have switched vLLM versions recently.\n\n"
                    "However, the next startup will take longer as graphs are re-compiled.\n\n"
                    "Are you sure you want to enable this?"
                )
                confirm = run_dialog([
                    "--title", "Erase Cache Warning", 
                    "--yesno", warn_msg, "12", "60"
                ])
                
                # If confirm is not None (exit 0), it is YES.
                if confirm is not None:
                     clear_cache = True
            else:
                # Disabling it -> No warning needed
                clear_cache = False
             
        elif choice == "7":
            # Toggle Eager Mode
            use_eager = not use_eager
             
        elif choice == "8":
            # Launch
            break
            
    # Build Command
    subprocess.run(["clear"])
    
    if clear_cache:
        nuke_vllm_cache()
    
    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-num-seqs", str(current_seqs),
        "--max-model-len", str(current_ctx),
        "--gpu-memory-utilization", str(current_util),
        "--dtype", "auto"
    ]

    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")

    # Optional structured fields. .get() keeps existing entries unaffected.
    if (q := config.get("quantization")):
        cmd.extend(["--quantization", q])
    if (n := config.get("served_model_name")):
        cmd.extend(["--served-model-name", n])
    if (tm := config.get("tokenizer_mode")):
        cmd.extend(["--tokenizer-mode", tm])
    if (cf := config.get("config_format")):
        cmd.extend(["--config-format", cf])
    if (lf := config.get("load_format")):
        cmd.extend(["--load-format", lf])
    if (ho := config.get("hf_overrides")):
        cmd.extend(["--hf-overrides", json.dumps(ho)])
    if (sc := config.get("speculative_config")):
        cmd.extend(["--speculative-config", json.dumps(sc)])
    if (ea := config.get("extra_args")):
        cmd.extend([str(a) for a in ea])
    
    # Env Vars
    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    
    if current_attn_backend == "AITER":
        env["VLLM_ROCM_USE_AITER"] = "1"
        cmd.extend(["--attention-backend", "ROCM_ATTN"])
    elif current_attn_backend == "ROCm (CK)":
        if "VLLM_ROCM_USE_AITER" in env:
            del env["VLLM_ROCM_USE_AITER"]
        cmd.extend(["--attention-backend", "ROCM_ATTN"])
    else: # Triton
        if "VLLM_ROCM_USE_AITER" in env:
            del env["VLLM_ROCM_USE_AITER"]
        cmd.extend(["--attention-backend", "TRITON_ATTN"])

    env.update(config.get("env", {}))

    # ViT attention on gfx1151: the default falls to TORCH_SDPA (flash_attn's
    # Triton-AMD subpackage isn't available) which produces NaN/Inf embeddings
    # and collapses the LM into an endless '!' stream. TRITON_ATTN uses vLLM's
    # own Triton ViT wrapper and is numerically healthy. No-op for LM-only.
    cmd.extend(["--mm-encoder-attn-backend", "TRITON_ATTN"])

    
    print("\n" + "="*60)
    print(f" Launching: {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx} | Util={current_util}")
    print(f" Backend:   {current_attn_backend}")
    if clear_cache:
        print(f" Action:    Clearing vLLM Cache (~/.cache/vllm)")
        
    # Variables that represent the custom environment overrides for models
    custom_env = config.get("env", {})
    if custom_env:
        print("\n --- Environment Variables ---")
        for k, v in custom_env.items():
            print(f" export {k}={v}")
            
    print(f"\n Command:   {' '.join(cmd)}")
    print("="*60 + "\n")
    
    os.execvpe("vllm", cmd, env)

def main():
    check_dependencies()
    gpu_count = detect_gpus()
    
    while True:
        # Build Model Menu
        menu_items = []
        for i, m_id in enumerate(MODELS_TO_RUN):
            name = m_id.split("/")[-1]
            # Pre-calc verified ctx for 'default' TP to show in menu? 
            # Or just show names. Just names is cleaner.
            config = MODEL_TABLE[m_id]
            menu_items.extend([str(i), name])
            
        choice = run_dialog([
            "--clear", "--backtitle", f"AMD Strix Halo vLLM Launcher (GPUs: {gpu_count})",
            "--title", "Select Model",
            "--menu", "Choose a model to serve:", "20", "60", "10"
        ] + menu_items)
        
        if not choice:
            subprocess.run(["clear"])
            print("Selection cancelled.")
            sys.exit(0)
            
        configure_and_launch(int(choice), gpu_count)

if __name__ == "__main__":
    main()
