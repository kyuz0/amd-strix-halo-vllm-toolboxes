DEFAULT_MODEL_ENV = {
    "VLLM_ROCM_USE_AITER": "0",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
}


def get_model_env(config):
    """Return explicit per-serve defaults plus model-specific overrides."""
    return DEFAULT_MODEL_ENV | config.get("env", {})


MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MAD uses 131k tokens. We scale to 32k for 32GB VRAM safety.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ]
    },

    # EXPERIMENTAL — FP8 (W8A8) via @leonyurko's Strix Halo Triton kernels (#67).
    # The "env" VLLM_STRIX_FP8_TRITON=1 opts this model into the patched fp8_triton
    # path (default-off; without it FP8 uses stock torch._scaled_mm). The kernels
    # require VLLM_ROCM_USE_AITER=0 + enforce_eager. Correctness-verified on gfx1151,
    # not yet benchmarked.
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic": {
        "trust_remote": False,
        "valid_tp": [1],
        "enforce_eager": True,
        "env": {"VLLM_STRIX_FP8_TRITON": "1", "VLLM_ROCM_USE_AITER": "0"},
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ]
    },

    "google/gemma-4-26B-A4B-it": {
        "trust_remote": False,
        "enforce_eager": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
        ]
    },

    "google/gemma-4-31B-it": {
        "trust_remote": False,
        "enforce_eager": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
        ]
    },
    # 2. GPT-OSS 20B (MXFP4)
    # MAD Row 0 uses 8192. We match this exactly.
    "openai/gpt-oss-20b": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "8192",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "openai",
            "--reasoning-parser", "openai_gptoss",
        ]
    },
    
    "openai/gpt-oss-120b": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "8192",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "openai",
            "--reasoning-parser", "openai_gptoss",
        ]
    },

    # DeepSeek V4 Flash owns its ROCm sparse-MLA attention path. Enable broad
    # AITER only for the tested sparse-indexer MQA-logits helper; gfx1151 cannot
    # use AITER's FP8 linear kernels. Keep eager mode as part of the validated
    # launch recipe.
    "deepseek-ai/DeepSeek-V4-Flash-0731": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "enforce_eager": True,
        # The ROCm DeepSeek-V4 model hardwires its own sparse MLA backend
        # (ROCM_FLASHMLA_SPARSE_DSV4); generic --attention-backend is inapplicable.
        "attention_backend": None,
        "attention_backend_label": "ROCM_FLASHMLA_SPARSE_DSV4 (model-specific)",
        "env": {
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_LINEAR": "0",
        },
        "ctx": "262144",
        "max_num_seqs": "1",
        "max_tokens": "256",
        # vLLM v0.27.1 has a native AMD DSpark speculator. It reuses DFlash's
        # block-parallel machinery, then applies DSpark's sequential Markov head.
        # Keep this explicit so both launchers expose a clean on/off toggle.
        "speculative_config": {
            "method": "dspark",
            "num_speculative_tokens": 5,
            "disable_padded_drafter_batch": True,
            "enforce_eager": True,
        },
        # Runs after the OpenAI API becomes ready. The tiny request exercises
        # decode/DSpark; the longer request warms chunked prefill/TileLang kernels.
        "warmup": {
            "prompt_tokens": 2048,
            "ready_timeout_seconds": 1200,
        },
        "extra_flags": [
            "--kv-cache-dtype", "fp8",
            "--block-size", "256",
            "--max-num-batched-tokens", "256",
            "--logprobs-mode", "processed_logprobs",
        ]
    },

    "Qwen/Qwen3.6-35B-A3B": {
        "trust_remote": True,
        "valid_tp": [1],
        # Verified end-to-end on gfx1151. Explicit backend selection works while
        # the broad AITER toggle remains off, avoiding the unsupported sampler.
        "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
        "env": {
            "VLLM_ROCM_USE_AITER": "0",
            "VLLM_ROCM_USE_AITER_LINEAR": "0",
        },
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    # Verified end-to-end on gfx1151 with unified AITER.
    "LiquidAI/LFM2.5-1.2B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
        "env": {
            "VLLM_ROCM_USE_AITER": "0",
            "VLLM_ROCM_USE_AITER_LINEAR": "0",
        },
        "ctx": "128000",
        "max_num_seqs": "64",
        "max_tokens": "16384",
    },

    # Muse Glimmer is a BF16 multimodal model with alternating 2k sliding and
    # full-attention layers. vLLM does not yet have a native model class, so use
    # its Transformers implementation. Unified AITER supports its GQA shape,
    # 128-d head size, sliding windows, and multimodal-prefix attention.
    "meta-models/Muse-Glimmer-30B": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
        "env": {
            "VLLM_ROCM_USE_AITER": "0",
            "VLLM_ROCM_USE_AITER_LINEAR": "0",
        },
        "ctx": "131072",
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--model-impl", "transformers",
        ]
    },

    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1], 
        "enforce_eager": True, 
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },  

    "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2], # Too big for single GPU
        "enforce_eager": True, 
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    "cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit": {
        "trust_remote": True,
        "valid_tp": [2], # Too big for single GPU
        "enforce_eager": True, 
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    "cyankiwi/MiniMax-M2.7-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [2],
        "enforce_eager": True,
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "minimax_m2",
            "--reasoning-parser", "deepseek_r1",
        ]
    },

    "ayysasha/MiniMax-M2.7-AWQ-G32-STRIX-2H": {
        "trust_remote": True,
        "valid_tp": [2],
        "enforce_eager": True,
        "ctx": "131072",
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "minimax_m2",
            "--reasoning-parser", "deepseek_r1",
        ]
    },

}

MODELS_TO_RUN = list(MODEL_TABLE.keys())

# Hardware / Global Defaults
GPU_UTIL = "0.90"
OFF_NUM_PROMPTS = 200 # Increased for Strix Halo (Steady State Saturation)
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"
