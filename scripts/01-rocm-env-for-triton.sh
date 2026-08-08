# Required for Strix Halo / RDNA3.5 on vLLM
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export VLLM_TARGET_DEVICE=rocm
export VLLM_USE_TRITON_AWQ=1
# MIOpen's exhaustive kernel search hangs on gfx1151 for some conv shapes
# (notably vision-encoder conv stems). FAST uses heuristics instead.
export MIOPEN_FIND_MODE=FAST

export PYTHONNOUSERSITE=1
