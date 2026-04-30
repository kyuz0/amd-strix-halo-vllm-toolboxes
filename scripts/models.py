# Per-model launcher config.
#
# Required: max_num_seqs (str), max_tokens (str). Optional: trust_remote (bool),
# valid_tp (list[int], default [1]), enforce_eager (bool), env (dict).
#
# Optional vllm-serve passthrough (read with .get() so older entries are
# unaffected): quantization, served_model_name, tokenizer_mode, config_format,
# load_format, hf_overrides (dict), speculative_config (dict, see
# SpeculativeMethod in vllm/config/speculative.py for valid `method` values),
# attention_backend (TRITON_ATTN/ROCM_ATTN/AITER), extra_args (list[str]),
# notes (str). The stringly-typed enum fields are validated by start_vllm.py
# before exec.

# Shared extra_args groups so multi-entry families don't drift.
_QWEN3_TOOL_ARGS = [
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
]

_MISTRAL_TOOL_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser", "mistral",
    "--reasoning-parser", "mistral",
]

_NEMOTRON_OMNI_TOOL_ARGS = [
    "--reasoning-parser", "nemotron_v3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--video-pruning-rate", "0.5",
    "--media-io-kwargs", '{"video":{"num_frames":512,"fps":1}}',
    "--limit-mm-per-prompt", '{"image":4,"video":1,"audio":2}',
]


MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MAD uses 131k tokens. We scale to 32k for 32GB VRAM safety.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    "google/gemma-4-26B-A4B-it": {
        "trust_remote": False,
        "enforce_eager": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    "google/gemma-4-31B-it": {
        "trust_remote": False,
        "enforce_eager": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },
    # 2. GPT-OSS 20B (MXFP4)
    # MAD Row 0 uses 8192. We match this exactly.
    "openai/gpt-oss-20b": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },

    "openai/gpt-oss-120b": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },

    "Qwen/Qwen3.6-35B-A3B": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "enforce_eager": True,
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

    "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2], # Too big for single GPU
        "enforce_eager": True,
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

    "cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit": {
        "trust_remote": True,
        "valid_tp": [2], # Too big for single GPU
        "enforce_eager": True,
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

    "cyankiwi/MiniMax-M2.7-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [2],
        "enforce_eager": False,
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

    # Mistral-Medium-3.5-128B (Mistral3 dense + Pixtral vision) paired with the
    # official Mistral-Medium-3.5-128B-EAGLE FP8 draft. The mistral tokenizer/
    # config/load formats are required for the Tekken tokenizer + tool-calling.
    "mistralai/Mistral-Medium-3.5-128B": {
        "trust_remote": False,
        "valid_tp": [1],
        "enforce_eager": True,
        "max_num_seqs": "4",
        "max_tokens": "32768",
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "speculative_config": {
            "method": "eagle",
            "model": "mistralai/Mistral-Medium-3.5-128B-EAGLE",
            "num_speculative_tokens": 3,
        },
        "extra_args": _MISTRAL_TOOL_ARGS,
    },

    # NVIDIA Nemotron-3 Nano Omni (NemotronH_Nano_Omni_Reasoning_V3 ->
    # nano_nemotron_vl). Hybrid Mamba2-Transformer MoE with vision (Radio),
    # audio (Parakeet) and video (EVS). Patch_11 wires the nemotron_v2_vl
    # GGUF projector. Audio extras are installed by install_deps.sh + Dockerfile.
    "unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF": {
        "trust_remote": True,
        "valid_tp": [1],
        "enforce_eager": True,
        "quantization": "gguf",
        "max_num_seqs": "8",
        "max_tokens": "131072",
        "extra_args": _NEMOTRON_OMNI_TOOL_ARGS,
    },

    # Qwen3.6-27B GGUF + native MTP head. TP=1 only on hybrid GDN until
    # vllm-project/vllm#41190 closes. Pre-quant GGUFs typically drop the MTP
    # tensors; falls back to non-MTP if the head isn't found.
    "unsloth/Qwen3.6-27B-GGUF": {
        "trust_remote": True,
        "valid_tp": [1],
        "enforce_eager": True,
        "quantization": "gguf",
        "max_num_seqs": "8",
        "max_tokens": "131072",
        "speculative_config": {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        },
        "extra_args": _QWEN3_TOOL_ARGS,
    },

    # AWQ-INT4 safetensors variant of Qwen3.6-27B with the MTP head intact.
    # Use this when the GGUF + MTP path above can't find the MTP tensors.
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": {
        "trust_remote": True,
        "valid_tp": [1],
        "enforce_eager": True,
        "max_num_seqs": "8",
        "max_tokens": "131072",
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "speculative_config": {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        },
        "extra_args": _QWEN3_TOOL_ARGS,
    },

}

MODELS_TO_RUN = list(MODEL_TABLE.keys())

# Hardware / Global Defaults
GPU_UTIL = "0.90"
OFF_NUM_PROMPTS = 200 # Increased for Strix Halo (Steady State Saturation)
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"
