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

    "ayysasha/MiniMax-M2.7-AWQ-G32-STRIX-2H": {
        "trust_remote": True,
        "valid_tp": [2],
        "enforce_eager": False,
        "env": {"VLLM_USE_TRITON_AWQ": "1"},
        "ctx": "196608",
        "max_num_seqs": "64",
        "max_tokens": "16384"
    },

}

MODELS_TO_RUN = list(MODEL_TABLE.keys())

# Hardware / Global Defaults
GPU_UTIL = "0.90"
OFF_NUM_PROMPTS = 200 # Increased for Strix Halo (Steady State Saturation)
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"
