# vLLM and ROCm patch manifest

This is the upgrade ledger for the Ubuntu `repo.amd` toolbox built by
`Dockerfile.ubuntu-repoamd`. Read and update it before changing any pinned
ROCm, PyTorch, AITER, TileLang, RDMA-core, or vLLM revision.

The current build inputs are:

| Component | Pin |
|---|---|
| ROCm wheels | `7.14.0` |
| PyTorch | `2.11.0+rocm7.14.0` |
| AITER | `v0.1.19` |
| vLLM audited baseline | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| rdma-core | `v62.0` |
| Target GPU | `gfx1151` / Strix Halo |

The patchers are intentionally anchor-based and mostly fail closed. In addition,
`scripts/verify_vllm_compat.py` checks native upstream interfaces that local
launch policy depends on but does not patch. A failed anchor or compatibility
guard means upstream changed: inspect the new source and decide whether to
remove, rebase, or replace the local behavior. Never weaken a check merely to
make an upgrade build.

The GitHub workflow resolves the latest stable tag when no manual `vllm_ref` is
provided; the Dockerfile fallback is the audited version above. A newer stable
tag passing the source guard is only build compatibility, not GPU validation.
Update this baseline after completing the runtime matrix below.

## Upgrade procedure

1. Resolve the candidate vLLM ref to a full commit and record it above and in
   `Dockerfile.ubuntu-repoamd`.
2. Apply every patcher to a clean checkout of that exact commit. Review the
   resulting diff path by path; a successful string replacement is not proof
   that the surrounding semantics are unchanged.
3. Search upstream for the marker, issue, class, and behavior named in each
   table below. Remove local code when upstream provides the same behavior.
4. Run `python -m unittest discover -s tests -v`, `git diff --check`, and Python
   compilation checks.
5. Build the image with the GitHub Actions development workflow. Do not treat a
   local source test as an image or GPU validation.
6. On gfx1151, validate cold and warm starts for:
   - single-host DeepSeek V4 with speculation disabled;
   - single-host DeepSeek V4 with native DSpark-5;
   - TP=2 over Ethernet;
   - TP=2 over RDMA/RoCE;
   - at least one non-DeepSeek model with the normal AITER policy.
7. For DSpark, record accepted tokens per target step, acceptance rate, output
   quality, warm decode throughput, and behavior at the configured maximum
   concurrency. A server that merely stays up is not a performance validation.
8. Verify that `~/.cache/triton` and `~/.aiter` persist on both Ray nodes and
   that automatic warmup completes after the API becomes ready.

## DeepSeek V4 gfx1151 and TileLang

Owned by `scripts/patch_dsv4_gfx1x.py` and
`scripts/gfx1x_tilelang_mqa.py`.

| Upstream path | Local behavior | Update audit |
|---|---|---|
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | Routes prefill and paged sparse-indexer MQA logits to the local TileLang implementation on gfx1x. Other GPUs retain upstream dispatch. | Check whether upstream has a gfx11/gfx1151 sparse-indexer path, whether cache layout or function signatures changed, and whether speculative verification still passes `next_n > 1` correctly. |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | Keeps FP8 storage but converts gfx1x block-scaled matrix operands to BF16 for `tl.dot`; gfx1151 has no native FP8 matrix-core dot. | Check the block-scaled GEMM implementation, config construction, scale semantics, and any new upstream RDNA fallback. Remove when upstream avoids raw FP8 dot on gfx1x. |
| new `vllm/v1/attention/ops/gfx1x_tilelang_mqa.py` | TileLang BF16 sparse-indexer GEMM, paged-cache de-shuffle, bounded KV buckets, and speculative-row causal bounds. Adapted from AlexKGwyn/ds4-vllm-public. | Compare with the current upstream sparse indexer and the source project's latest `ds4_tl_indexer.py`. Revalidate page layout, query shape, FP8 scale interpretation, context bucketing, and `next_n`. |

Build markers:

- `PATCHED: gfx1x TileLang sparse-indexer MQA`
- `PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot`

Tests: `tests/test_dsv4_gfx1x_patch.py`.

## Native DSpark / DFlash block speculation

The model is `deepseek-ai/DeepSeek-V4-Flash-0731`. The launcher passes native
vLLM configuration with `method=dspark`, five speculative tokens, unpadded
draft batches, and eager draft execution. DSpark reuses vLLM's DFlash
block-parallel machinery, then samples left-to-right with its trained Markov
head. There is deliberately no local DSpark model implementation or vLLM patch.

This differs from the older
[AlexKGwyn/ds4-vllm-public](https://github.com/AlexKGwyn/ds4-vllm-public/tree/71a73d0c1ad42a51e8d4da7b3585a217917a4637)
port, which patched an earlier vLLM snapshot before native AMD DeepSeek DSpark
landed. The useful launch policy and warmup behavior were retained; the old
drafter and invasive integration patches were not.

`scripts/verify_vllm_compat.py` checks every native path below during the image
build. These are dependencies, not locally modified files.

| Native upstream path | Dependency | Update audit |
|---|---|---|
| `vllm/config/speculative.py` | Recognizes `method=dspark`, treats it as block-parallel drafting, validates block size, and constructs a draft config from weights embedded in the target checkpoint. | Check the method name, config fields, `num_speculative_tokens` validation, draft TP behavior, and whether a model path becomes required. |
| `vllm/models/deepseek_v4/__init__.py` | Registers `DSparkDeepseekV4ForCausalLM` from the AMD implementation on ROCm. | Check platform dispatch and model-registry names. Never fall back to the NVIDIA implementation on gfx1151. |
| `vllm/models/deepseek_v4/amd/dspark.py` | Native AMD DeepSeek DSpark backbone, Markov head, checkpoint `mtp.*` remapping, shared embeddings/head declarations, MHC dispatch, and per-draft-layer sliding-window KV insertion. | Recheck weight names and quantized Markov heads, target-layer count, number of draft layers, FP8/FP4 scales, MHC fused/unfused behavior, TP collectives, RoPE, and AMD attention/cache dtypes. |
| `vllm/models/deepseek_v4/amd/model.py` | Emits mean-pooled, post-MHC auxiliary hidden states for the configured target layers. | Recheck one-based layer conversion, fused/unfused reconstruction, tuple return shape, pipeline-parallel restrictions, and interaction with the local AITER-linear gate patch. |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | Selects `DSparkSpeculator` for `method=dspark`. | Check worker/model-runner selection; confirm the runtime is not falling through to legacy `EagleProposer` or serial MTP. |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | Runs one block-parallel draft, then applies sequential Markov sampling. | Recheck anchor/position convention, query count, greedy and probabilistic paths, reduced-vocab mapping, graph capture, batch padding, and accepted-token metrics. |
| `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | Loads the draft from the target checkpoint and explicitly aliases target embeddings and LM head. | Recheck `has_own_embed_tokens`/`has_own_lm_head`, quant config replacement, pipeline-parallel rejection, and model wrappers. A broken alias can produce invalid draft logits without breaking target-only serving. |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | Converts checkpoint `dspark_target_layer_ids` to the target model's one-based auxiliary-layer interface. | Recheck layer-number semantics against the checkpoint config and `amd/model.py`. |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | Marks DSpark sparse sliding-window attention so its within-block non-causal semantics and cache metadata are built correctly. | Recheck non-causal block masks, slot mappings, cache groups, block size, and TP=2 behavior. |

The local TileLang sparse-indexer patch is still exercised during target
verification. At upgrades, check its `next_n` row bounds and shuffled-cache
layout together with native DSpark; validating each component separately is not
enough.

The old port's custom `amd/dspark_mtp.py`, target-capture patch, generic
proposer replay patch, forced KV grouping, `_dspark_kv` hook, custom
Thunderbolt all-reduce, and scheduler diagnostics are intentionally absent.
Reintroduce none of them unless the native path is proven deficient and the
specific change is separately documented and tested.

Tests: `tests/test_native_dspark_integration.py` and
`tests/test_launcher_features.py`.

## Automatic warmup

Owned by `scripts/launcher_features.py` and `scripts/vllm_warmup.py`.

Immediately before `execvpe`, each launcher starts a best-effort helper. The
helper watches the unchanged server PID, waits for `/v1/models`, discovers the
actual served model ID, then sends:

1. a tiny 12-token decode request to exercise decode and speculative kernels;
2. a configurable longer prompt (2048 approximate tokens for DeepSeek V4) to
   exercise chunked prefill and TileLang indexer kernels.

The helper cannot make server startup fail. It exits if the server process
disappears or readiness times out. The launcher TUI exposes a per-launch toggle.
Warmup runs only on the head API; a TP request executes on both Ray ranks and
therefore warms both GPUs. Compiled caches remain host-persistent.

At the next update, verify the OpenAI route, model discovery response, chat
template behavior, first-request compile log, cache locations on both nodes,
and whether vLLM has gained a native post-start request warmup facility.

## General Strix/AITER vLLM patcher

Owned by `scripts/patch_strix.py`.

| Upstream/package path | Local behavior | Update audit |
|---|---|---|
| `vllm/_aiter_ops.py` | Allows the central AITER capability on gfx1151 while disabling unsupported FP8 linear and fused-MoE operators. | Recheck AITER capability APIs and every gfx1151 operator independently. Do not equate central availability with universal operator support. |
| `vllm/models/deepseek_v4/amd/model.py` and `vllm/models/deepseek_v4/amd/rocm.py` | Routes private DeepSeek FP8-linear decisions through `is_linear_fp8_enabled()`. | Remove when upstream respects the capability gate. Ensure this composes with native DSpark auxiliary-hidden-state capture. |
| `vllm/v1/executor/ray_executor_v2.py` | Refreshes AITER's cached environment after Ray applies final worker variables. | Check actor initialization and environment-copy order. Remove if AITER no longer snapshots environment values at import/class initialization. |
| `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` | Prevents a broad AITER environment override from forcing unsupported MoE on gfx1x. | Check new MoE oracle and environment override semantics. |
| `vllm/platforms/rocm.py` | Keeps AITER RMSNorm out of the gfx1x operator priority list. | Revalidate RMSNorm correctness and graph capture; remove when upstream supports gfx1151. |
| `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | Registers replacements with duplicate skipping. | Check upstream pattern-manager behavior; remove when duplicate registration is handled. |
| installed `aiter/jit/__init__.py` | Adds `~/.aiter/jit` to the package search path. | Check AITER's JIT output/import path; remove when upstream imports its cache correctly. |
| installed `flash_attn/flash_attn_interface.py` | Makes the AITER flash-attention import soft so Triton attention remains usable after an AITER failure. | Check the current flash-attention package layout and fallback behavior. |

Comments inside `patch_strix.py` also record removed historical patches. Preserve
those notes during upgrades: they prevent obsolete workarounds from being
accidentally resurrected.

## Non-vLLM build-time patches

| Patcher/path | Local behavior | Update audit |
|---|---|---|
| `scripts/patch_amdsmi.py` -> installed `amdsmi_interface.py` | On APUs, uses the larger GTT/unified pool instead of the 512 MiB BIOS VRAM carveout. | Check ROCm/rocm-systems issues 8419 and 8476 and compare amdsmi, HIP, KFD, and host available memory. Remove after upstream reports the unified pool correctly. |
| `scripts/patch_aiter_headers.py` -> `aiter_meta/csrc/include/ck_tile/vec_convert.h` | Replaces unsupported packed CDNA FP8 instructions with scalar RDNA fallbacks and supplies the needed FP4 types/conversions. | Diff against AITER's current header, compile representative JIT modules on gfx1151, and remove once upstream carries correct RDNA implementations. |
| rdma-core `v62.0` overlay | Replaces Ubuntu 24.04's rdma-core 50/libirdma provider, which produced RCCL `Unknown completion` errors with the host irdma driver. | Run the two-rank RCCL all-reduce over RoCE. Upgrade only as a matched libibverbs/provider set; do not mix provider ABIs. |

## Legacy image note

The old Fedora `Dockerfile` has a separate, experimental
`scripts/patch_fp8_kernels.py` path and external
`leonyurko/vllm-fp8-strix-halo-kernel-support` modules. That is not part of the
Ubuntu stable image documented above. If the legacy image is changed, audit its
patch stack separately rather than assuming this manifest covers it.
