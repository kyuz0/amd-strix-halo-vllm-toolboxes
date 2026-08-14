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

ATTENTION_BACKENDS = (
    "TRITON_ATTN",
    "ROCM_ATTN",
    "ROCM_AITER_UNIFIED_ATTN",
)

DEFAULT_TRITON_CACHE_DIR = Path.home() / ".cache" / "triton"


def get_triton_cache_dir():
    """Return a host-persistent per-user Triton cache path."""
    configured = os.getenv("TRITON_CACHE_DIR")
    if configured and configured != "/opt/triton_cache":
        return Path(configured).expanduser()
    return DEFAULT_TRITON_CACHE_DIR

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
    model_ctx_default = MODEL_TABLE[model_id].get("ctx", "auto")
    default_config = {
        "ctx": model_ctx_default,
        "util": 0.90 # Safe default
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
        # Cap util to 0.90 max. Due to recent changes in vLLM/ROCm UMA
        # available memory calculations (psutil.virtual_memory().available vs hipMemGetInfo),
        # 0.95 often leads to OOM at startup on Strix Halo APUs.
        util = float(best["util"])
        return {
            "ctx": best["max_context_1_user"],
            "util": min(0.90, util)
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

def reset_compiled_caches():
    """Clear vLLM, Triton, and AITER compiled caches for a cold rebuild."""
    caches = [
        ("vLLM", Path.home() / ".cache" / "vllm"),
        ("Triton", get_triton_cache_dir()),
        ("AITER", Path.home() / ".aiter"),
    ]

    for label, cache in caches:
        if not cache.exists():
            print(f"{label} cache not present at {cache}; skipping.")
            continue

        try:
            print(f"Clearing {label} cache at {cache}...", end="", flush=True)
            # Keep the cache root itself in case the user mounted it explicitly.
            for entry in cache.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            print(" Done.")
        except Exception as e:
            print(f" Failed: {e}")

    time.sleep(1)

def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    
    # Static Config
    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1
    
    # Defaults
    current_tp = min(gpu_count, max_tp)
    current_seqs = 1 # Default to 1 concurrent user/request for stability
    
    # Initial Lookup
    verified = get_verified_config(model_id, current_tp, current_seqs)
    current_ctx = verified["ctx"]
    current_util = verified["util"]
    
    clear_cache = False  # Keep compiled kernels on normal launches; reset only when needed
    use_eager = config.get("enforce_eager", False) # Default to model config, usually False
    configured_attn_backend = config.get("attention_backend", "TRITON_ATTN")
    attention_backend_locked = configured_attn_backend is None
    if attention_backend_locked:
        current_attn_backend = config.get(
            "attention_backend_label", "Model-specific"
        )
        attn_backends = [current_attn_backend]
    else:
        if configured_attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention backend {configured_attn_backend!r} "
                f"for {model_id}"
            )
        attn_backends = list(ATTENTION_BACKENDS)
        current_attn_backend = configured_attn_backend
    current_extra_flags = list(config.get("extra_flags", []))  # Copy so edits don't mutate config
    
    name = model_id.split("/")[-1]
    
    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        
        extra_flags_display = ' '.join(current_extra_flags) if current_extra_flags else '(none)'
        # Truncate display for menu readability
        extra_flags_short = (extra_flags_display[:40] + '...') if len(extra_flags_display) > 43 else extra_flags_display

        menu_args = [
            "--clear", "--backtitle", f"AMD Strix Halo vLLM Launcher (GPUs: {gpu_count})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "24", "70", "10",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx} (Verified)",
            "4", f"GPU Utilization:      {current_util} (Verified)",
            "5", f"Attention Backend:    {current_attn_backend}",
            "6", f"Reset Compiled Caches:{cache_status:>5}",
            "7", f"Force Eager Mode:     {eager_status}",
            "8", f"Extra vLLM Flags:     {extra_flags_short}",
            "9", "LAUNCH SERVER"
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
            if attention_backend_locked:
                run_dialog([
                    "--title", "Attention Backend",
                    "--msgbox",
                    f"{current_attn_backend}\n\n"
                    "This model provides its own attention implementation, so "
                    "start-vllm does not pass --attention-backend.",
                    "10", "68"
                ])
            else:
                # Cycle Attention Backend
                idx = attn_backends.index(current_attn_backend)
                current_attn_backend = attn_backends[(idx + 1) % len(attn_backends)]

        elif choice == "6":
            # Toggle Cache
            if not clear_cache:
                # Enabling it -> Show Warning
                warn_msg = (
                    "This performs a cold rebuild by clearing:\n\n"
                    "~/.cache/vllm - vLLM/torch.compile graphs\n"
                    "  Reset after vLLM/PyTorch changes or graph/compile errors.\n\n"
                    "~/.cache/triton - Triton kernels and autotuning results\n"
                    "  Reset after Triton/PyTorch/ROCm/backend changes or kernel errors.\n\n"
                    "~/.aiter - AITER JIT kernels\n"
                    "  Reset after AITER/ROCm/backend changes or AITER JIT errors.\n\n"
                    "Keep this disabled for normal launches: rebuilding can take minutes.\n\n"
                    "Reset all compiled caches on the next launch?"
                )
                confirm = run_dialog([
                    "--title", "Reset Compiled Caches",
                    "--yesno", warn_msg, "21", "76"
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
            # Edit Extra vLLM Flags
            current_str = ' '.join(current_extra_flags)
            new_flags = run_dialog([
                "--title", "Extra vLLM Flags",
                "--inputbox",
                "Edit extra flags (space-separated, passed directly to vllm serve).\n"
                "Clear the field to remove all extra flags.",
                "12", "70", current_str
            ])
            if new_flags is not None:  # None = cancelled
                current_extra_flags = new_flags.split() if new_flags.strip() else []
              
        elif choice == "9":
            # Launch
            break
            
    # Build Command
    subprocess.run(["clear"])
    
    if clear_cache:
        reset_compiled_caches()
    
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

    # Extra vLLM flags (from models.py defaults + user edits)
    if current_extra_flags:
        cmd.extend(current_extra_flags)
    
    # Env Vars
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(get_triton_cache_dir())
    # Attention backend selection is independent from the broad AITER operator
    # toggle. Model-specific env overrides (notably DeepSeek V4) are applied below.
    env.pop("VLLM_ROCM_USE_AITER", None)

    if not attention_backend_locked:
        cmd.extend(["--attention-backend", current_attn_backend])

    env.update(config.get("env", {}))

    print("\n" + "="*60)
    print(f" Launching: {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx} | Util={current_util}")
    print(f" Backend:   {current_attn_backend}")
    if clear_cache:
        print(" Action:    Resetting vLLM, Triton, and AITER compiled caches")
    if current_extra_flags:
        print(f" Extras:    {' '.join(current_extra_flags)}")
        
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
            "--clear", "--backtitle", f"AMD R9700 vLLM Launcher (GPUs: {gpu_count})",
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
