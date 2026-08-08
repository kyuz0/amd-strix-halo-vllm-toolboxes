#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, argparse, shutil
from pathlib import Path

try:
    import bench_utils
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import bench_utils


# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

try:
    import models
except ImportError:
    # If running locally and models.py is in ../scripts?
    # Or if running in /opt where models.py is alongside.
    # We will try adding current dir to path just in case
    sys.path.append(os.getcwd())
    try:
        import models
    except ImportError:
        # Fallback for local structure: assuming this is in benchmarks/ and models is in scripts/
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        import models

# Import from shared config
MODEL_TABLE = models.MODEL_TABLE
MODELS_TO_RUN = models.MODELS_TO_RUN
GPU_UTIL = models.GPU_UTIL
OFF_NUM_PROMPTS = models.OFF_NUM_PROMPTS
OFF_FORCED_OUTPUT = models.OFF_FORCED_OUTPUT
DEFAULT_BATCH_TOKENS = models.DEFAULT_BATCH_TOKENS

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("~/vllm_benchmark_results").expanduser()
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# =========================
# UTILS
# =========================

def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    try:
        # Using rocm-smi --showid to list GPUs. 
        # Output format: "GPU[0] : Device Name: ..."
        res = subprocess.run(["rocm-smi", "--showid"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return 1 # Force return 1 for Strix Halo APU
        else:
            log("rocm-smi failed, defaulting to 1 GPU (Hardcoded Fallback)")
            return 1
    except Exception as e:
        log(f"Error detecting GPUs: {e}, defaulting to 1 GPU")
        return 1

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9", 
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None



def get_model_args(model, tp_size, overrides=None):
    config = MODEL_TABLE.get(model, {})
    overrides = overrides or {}
    
    # Allow per-model GPU utilization override
    util = overrides.get("gpu_util", config.get("gpu_util", GPU_UTIL))
    max_seq_override = overrides.get("max_num_seqs", config.get("max_num_seqs", "32"))

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", str(util),
        "--dtype", "auto",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", str(max_seq_override)
    ]
    
    if "ctx" in overrides:
        cmd.extend(["--max-model-len", str(overrides.get("ctx"))])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    return cmd

def run_throughput(model, tp_size, backend_name="Default", output_dir=RESULTS_DIR, extra_env=None, overrides=None):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    overrides = overrides or {}
    
    model_safe = model.replace("/", "_")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    tag = overrides.get("tag", "").strip()
    tag_suffix = f"_{tag}" if tag else ""
    output_file = output_dir_path / f"{model_safe}_tp{tp_size}{tag_suffix}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP {model} (TP={tp_size} | {backend_name})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    # Retrieve Model-Specific Batch Tokens
    batch_tokens = str(overrides.get("max_tokens", MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)))

    log(f"START {model} (TP={tp_size} | {backend_name}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    vllm_path = shutil.which("vllm") or "vllm"
    cmd = ["python", "-W", "ignore", vllm_path, "bench", "throughput"] + get_model_args(model, tp_size, overrides)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # Explicitly set Attention Backend for every run
    if backend_name == "AITER-Attn":
        cmd.extend(["--attention-backend", "ROCM_AITER_FA"])
    elif backend_name == "ROCm-Attn":
        cmd.extend(["--attention-backend", "ROCM_ATTN"])
    else:
        cmd.extend(["--attention-backend", "TRITON_ATTN"])

    cmd.extend(["--mm-encoder-attn-backend", "TRITON_ATTN"])

    # ENV Setup: Global + Model Specific
    env = os.environ.copy()
    env.pop("VLLM_ROCM_USE_AITER", None)
    
    # Inject model specific env vars (e.g. for AWQ)
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)
    
    # Extra Env
    if extra_env:
        env.update(extra_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except: 
        log(f"ERROR: Failed {model} [{backend_name}]")


def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'Tag':<15} | {'Triton':<8} | {'ROCm':<8} | {'AITER':<8}")
    print("-" * 103)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        name_cell = m.split('/')[-1]
        
        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue
            
            prefix = f"{msafe}_tp{tp}"
            
            tags = set()
            for p in (RESULTS_DIR / "triton").glob(f"{prefix}*_throughput.json"):
                name_part = p.name[len(prefix):-len("_throughput.json")]
                tag = name_part.lstrip("_")
                tags.add(tag)
                
            for p in (RESULTS_DIR / "rocm").glob(f"{prefix}*_throughput.json"):
                name_part = p.name[len(prefix):-len("_throughput.json")]
                tag = name_part.lstrip("_")
                tags.add(tag)
                
            for p in (RESULTS_DIR / "aiter").glob(f"{prefix}*_throughput.json"):
                name_part = p.name[len(prefix):-len("_throughput.json")]
                tag = name_part.lstrip("_")
                tags.add(tag)
                
            if not tags:
                tags.add("") # Default empty tag if no files found
                
            for tag in sorted(list(tags)):
                tag_suffix = f"_{tag}" if tag else ""
                
                # Default
                try: 
                    p1 = (RESULTS_DIR / "triton") / f"{prefix}{tag_suffix}_throughput.json"
                    if p1.exists():
                        d1 = json.loads(p1.read_text())
                        val1 = f"{d1.get('tokens_per_second', 0):.1f}"
                    else:
                        val1 = "N/A"
                except: val1 = "N/A"
                
                # ROCm
                try:
                    p2 = (RESULTS_DIR / "rocm") / f"{prefix}{tag_suffix}_throughput.json"
                    if p2.exists():
                        d2 = json.loads(p2.read_text())
                        val2 = f"{d2.get('tokens_per_second', 0):.1f}"
                    else:
                        val2 = "N/A"
                except: val2 = "N/A"

                # AITER
                try:
                    p3 = (RESULTS_DIR / "aiter") / f"{prefix}{tag_suffix}_throughput.json"
                    if p3.exists():
                        d3 = json.loads(p3.read_text())
                        val3 = f"{d3.get('tokens_per_second', 0):.1f}"
                    else:
                        val3 = "N/A"
                except: val3 = "N/A"

                display_tag = tag if tag else "(Default)"
                print(f"{name_cell:<40} | {tp:<2} | {display_tag:<15} | {val1:<8} | {val2:<8} | {val3:<8}")
                
    print("-" * 103)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM High-Concurrency Throughput Benchmark Suite")
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument("--tui", action="store_true", help="Launch interactive configuration UI")
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} AMD GPU(s)")
    log("NOTE: Running Peak Throughput Benchmark. This simulates high-concurrency batching to saturate hardware bandwidth.")
    log("This does NOT represent single-request user generation speed (Concurrency=1).")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    selected_models = MODELS_TO_RUN
    
    if args.tui:
        # TUI Model Selection
        checklist_args = [
            "--clear", "--backtitle", "AMD vLLM Benchmark Launcher",
            "--title", "Model Selection",
            "--checklist", "Select models to benchmark:", "20", "65", "10"
        ]
        
        for m in MODELS_TO_RUN:
            m_name = m.split("/")[-1]
            # All selected "on" by default
            checklist_args.extend([m, m_name, "on"])
            
        choice = bench_utils.run_dialog(checklist_args)
        
        if choice is None:
            subprocess.run(["clear"])
            print("Cancelled by user.")
            sys.exit(0)
            
        # Parse space-separated quoted output from dialog checklist
        import shlex
        selected_models = [m for m in shlex.split(choice)]
        
        if not selected_models:
            subprocess.run(["clear"])
            print("No models selected. Exiting.")
            sys.exit(0)

    kill_vllm()
    for tp in valid_tp_args:
        for m in selected_models:
            overrides = {}
            if args.tui:
                config = MODEL_TABLE.get(m, {})
                default_seqs = config.get("max_num_seqs", "32")
                default_tokens = config.get("max_tokens", DEFAULT_BATCH_TOKENS)
                default_util = config.get("gpu_util", GPU_UTIL)
                default_ctx = config.get("ctx", "auto")
                
                form_args = [
                    "--clear", "--backtitle", f"AMD vLLM Benchmark Configuration (TP: {tp})",
                    "--title", f"Tune Parameters: {m.split('/')[-1]}",
                    "--form", "Edit the options below. Leave tag empty for no suffix.",
                    "15", "70", "5",
                    "Max Concurrent Seqs:", "1", "1",  str(default_seqs), "1", "25", "15", "0",
                    "Max Batched Tokens:", "2", "1", str(default_tokens), "2", "25", "15", "0",
                    "GPU Utilization (0-1):", "3", "1", str(default_util), "3", "25", "15", "0",
                    "Max Context Length:", "4", "1", str(default_ctx), "4", "25", "15", "0",
                    "Filename Tag (Optional):", "5", "1", "", "5", "25", "15", "0"
                ]
                
                form_res = bench_utils.run_dialog(form_args)
                if form_res is None:
                    subprocess.run(["clear"])
                    print(f"Skipping {m} (TP={tp}) due to user cancellation.")
                    continue
                    
                lines = form_res.splitlines()
                if len(lines) >= 5:
                    overrides["max_num_seqs"] = lines[0].strip()
                    overrides["max_tokens"] = lines[1].strip()
                    overrides["gpu_util"] = lines[2].strip()
                    
                    ctx_val = lines[3].strip()
                    if ctx_val and ctx_val.lower() != "auto":
                        overrides["ctx"] = ctx_val
                        
                    overrides["tag"] = lines[4].strip()
            
            # 1. Triton Attention (explicit)
            run_throughput(m, tp, "Triton-Attn", RESULTS_DIR / "triton", overrides=overrides)
            
            # 2. ROCm Attention 
            # We force this via CLI argument --attention-backend ROCM_ATTN below
            # No specific env vars needed if forcing backend.
            rocm_env = {}
            print(f"[DEBUG] Forcing ROCm Env: {rocm_env} + CLI: --attention-backend ROCM_ATTN")
            run_throughput(m, tp, "ROCm-Attn", RESULTS_DIR / "rocm", rocm_env, overrides=overrides)
            
            # 3. AITER Attention
            aiter_env = {}
            print("[DEBUG] Forcing CLI: --attention-backend ROCM_AITER_FA")
            run_throughput(m, tp, "AITER-Attn", RESULTS_DIR / "aiter", aiter_env, overrides=overrides)
            
    print_summary(valid_tp_args)
