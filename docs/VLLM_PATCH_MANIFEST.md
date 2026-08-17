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
| vLLM stable fallback | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vLLM validated development baseline | `v0.27.2rc1.dev16` / `79f3183f86b89c3bda05d467041bf3ef9ef60426` |
| conch-triton-kernels | `1.2.1` (from vLLM `requirements/rocm.txt`) |
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
   - single-host DeepSeek V4 with native DSpark K7 greedy drafting;
   - TP=2 over Ethernet;
   - TP=2 over RDMA/RoCE;
   - at least one non-DeepSeek model with the normal AITER policy.
7. For DSpark, record accepted tokens per target step, acceptance rate, output
   quality, warm decode throughput, and behavior at the configured maximum
   concurrency. A server that merely stays up is not a performance validation.
8. Verify that `~/.cache/triton` and `~/.aiter` persist on both Ray nodes and
   that automatic warmup completes after the API becomes ready.
9. For the cached-BF16 W8A8 path, record target-only throughput before testing
   DSpark, confirm TP1 leaves it disabled and TP2 logs
   `[gfx1x_w8a8] BF16 direct path active` on every rank, compare output quality
   against the stock path, and record peak host memory. The BF16 copy changes
   numerics and is allocated after vLLM memory profiling.
10. For the MXFP4 MoE profile, confirm `[gfx1x_moe] enabled` on every TP rank,
    compare fixed-prompt target and speculative decode with the gate both off
    and on, and run output-quality and maximum-concurrency checks. Do not infer
    TP=2 behavior from the single-rank kernel result recorded below.
11. For radix top-k, compare its output exactly against the stable reference on
    random, tied, `-inf`-masked, bounded-prefill, and decode-shaped inputs. Then
    run long-context recall with the gate both off and on; kernel agreement at
    synthetic shapes is not a substitute for model recall.

## DeepSeek V4 gfx1151, TileLang, sampling, and cached-BF16 W8A8

Owned by `scripts/patch_strix.py`, `scripts/patch_dsv4_gfx1x.py`, and
the packaged `scripts/gfx1x_tilelang_mqa.py` and
`scripts/gfx1x_radix_topk.py`, and
`scripts/gfx1x_w8a8_bf16.py` helpers.

| Upstream path | Local behavior | Update audit |
|---|---|---|
| `vllm/v1/sample/ops/topk_topp_sampler.py` | Extends upstream's gfx1250 AITER-sampler exclusion to all gfx1x/RDNA. DeepSeek can keep broad AITER enabled for separate sparse-indexer helpers while normal sampling uses the native path. This replaces the old `--logprobs-mode processed_logprobs` workaround and avoids its unconditional full-vocabulary `log_softmax`. | Check whether upstream now excludes gfx1151 or provides a validated RDNA sampler. Revalidate ordinary top-k/top-p sampling with DeepSeek's broad AITER toggle enabled; remove the patch once upstream provides the same gate. |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | Routes prefill and paged sparse-indexer MQA logits to the local TileLang implementation on gfx1x. When the separate radix gate is enabled, both prefill and decode top-k use the deterministic local kernel; disabled, unavailable, and non-gfx1x cases retain upstream `top_k_per_row_*`. | Check whether upstream has a gfx11/gfx1151 sparse-indexer path, whether cache layout or function signatures changed, whether speculative verification still passes `next_n > 1` correctly, and whether top-k row-bound and output-index contracts changed. |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | Keeps FP8 storage but converts gfx1x block-scaled matrix operands to BF16 for `tl.dot`; gfx1151 has no native FP8 matrix-core dot. | Check the block-scaled GEMM implementation, config construction, scale semantics, and any new upstream RDNA fallback. Remove when upstream avoids raw FP8 dot on gfx1x. |
| `vllm/model_executor/kernels/linear/scaled_mm/triton.py` | When the DeepSeek-only environment gate is enabled on gfx1x, bypasses activation FP8 quantization and routes block-scaled linear calls to a cached BF16 weight. If the cache is unavailable, the original quantized-input and Triton paths remain intact. | Recheck `TritonFp8BlockScaledMMKernel`, `FP8BlockParams`, `weight_group_shape`, bias/cast/reshape ordering, Ray environment timing, and the stock fallback. Remove when upstream provides a performant RDNA block-FP8 or cached dequantized path. |
| `vllm/v1/executor/ray_executor_v2.py` | Refreshes the latched W8A8 environment gate after Ray copies model variables into an already-created worker, beside the existing AITER refresh. | Check when actor imports occur relative to `initialize_worker(env_vars)`. Both TP ranks must log the active path; printed driver environment alone is insufficient. |
| new `vllm/v1/attention/ops/gfx1x_tilelang_mqa.py` | TileLang BF16 sparse-indexer GEMM, paged-cache de-shuffle, bounded KV buckets, and speculative-row causal bounds. Adapted from AlexKGwyn/ds4-vllm-public. | Compare with the current upstream sparse indexer and the source project's latest `ds4_tl_indexer.py`. Revalidate page layout, query shape, FP8 scale interpretation, context bucketing, and `next_n`. |
| new `vllm/v1/attention/ops/gfx1x_radix_topk.py` | Deterministic radix threshold selection plus ordered integer compaction. It emits local prefill indices or global decode indices already ascending, with `-1` padding, without atomics or full-row sort scratch. Adapted from AlexKGwyn/ds4-vllm-public commit `95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff`. | Compare against the stable reference for both histogram modes and every served top-k. Recheck Triton histogram lowering, float-to-key ordering, tie behavior, row bounds, output strides, `-inf` handling, and maximum context. Remove if upstream gains deterministic performant ROCm selection. |
| new `vllm/model_executor/kernels/linear/scaled_mm/gfx1x_w8a8_bf16.py` | Caches each block-dequantized BF16 weight, sends small-M decode through `rocm_unquantized_gemm_impl`/gfx1x skinny GEMM, and optionally reuses warm weights for prefill. Adapted from AlexKGwyn/ds4-vllm-public. | Revalidate scale orientation and dtype, weight layout, skinny-GEMM dispatch, cache lifetime, temporary FP32 peak, BF16 cache size, output quality, and cold/warm behavior. Never infer end-to-end speed from the source project's per-kernel claim. |

Build markers:

- `PATCHED: gfx1x TileLang sparse-indexer MQA`
- `PATCHED: gfx1x deterministic radix top-k`
- `PATCHED: gfx1x block-FP8 GEMM uses BF16 tl.dot`
- `PATCHED: gfx1x cached-BF16 W8A8 linear`
- `PATCHED: disable AITER sampler on gfx1x`

The DeepSeek model profile always enables `VLLM_GFX1X_RADIX_TOPK=1`. Its W8A8
weight-cache policy is TP-aware: TP1 sets `VLLM_GFX1X_W8A8_BF16=0` and
`VLLM_GFX1X_W8A8_BF16_DIRECT=0`; TP2 sets both to `1`. A TP1 process holds the
entire 156+ GiB target and DSpark weights, so duplicating block-FP8 weights as
BF16 exhausted the 188 GiB device during startup warmup. TP2 shards both the
model and BF16 copies across hosts and retains the measured decode benefit.
Callers must resolve the environment through `models.get_model_env(config,
tp_size)` instead of reading `config["env"]` directly.

Setting `VLLM_GFX1X_RADIX_TOPK=0` restores upstream sparse-indexer top-k.
Setting the W8A8 base flag to `0` is the full cached-weight rollback; setting
only the direct flag to `0` retains the cached-BF16 quantized-input fallback.
The launcher also pins the per-rank KV pool to 6 GiB with
`--kv-cache-memory-bytes 6442450944`. Revalidate that pin whenever model
weights, context policy, cache layout, TP size, or concurrency changes. The
model profile no longer passes `--logprobs-mode processed_logprobs`; the direct
gfx1x sampler gate preserves the native fallback without changing logprob
semantics or paying for a full-vocabulary log-softmax on every decode step.

Tests: `tests/test_dsv4_gfx1x_patch.py`.

Initial gfx1151 validation on the 192 GiB `gh2` host used the same ROCm,
PyTorch, Triton, and validated vLLM development baseline listed above. The
radix kernel matched its stable reference exactly for random 128K rows, exact
ties, `-inf` masks, underfilled rows, bounded/local prefill indices, repeated
launches, and both histogram implementations. Component timings for top-512
selection were:

| Shape | Radix | Stable two-sort | Speedup |
|---|---:|---:|---:|
| K7-like decode, `8 x 128K` | `494.5 us` | `1105.0 us` | `2.23x` |
| prefill, `32 x 228K` | `1022.2 us` | `6085.0 us` | `5.95x` |
| prefill, `128 x 228K` | `3393.7 us` | `29355.2 us` | `8.65x` |

A disposable TP=1 server then loaded the exact patched vLLM baseline with
native K7 DSpark, logged `[gfx1x_topk] deterministic radix path active`, and
recovered the exact needle from a 33,732-token neutral-filler prompt while
generating 256 tokens. This validates the enabled single-rank integration; it
does not replace the required TP=2, gate-off A/B, deeper recall, or benchmark
matrix before promotion.

## gfx1x MXFP4 MoE tile profile

Owned by `scripts/patch_conch_moe_gfx1x.py`. The patch is applied after vLLM's
ROCm requirements install `conch-triton-kernels`, because the target is the
vendored conch module inside the installed vLLM package rather than the source
tree used to build the wheel.

| Installed upstream path | Local behavior | Update audit |
|---|---|---|
| `vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py` | Adds an explicit gfx1x BF16-by-MXFP4 skinny-M profile: `BM=16`, `BN=32`, `BK=256`, two warps, two stages, and one wave per EU. It is enabled only by `VLLM_GFX1X_MOE_TUNE=1`; every value is independently overridable. Other dtypes, unscaled weights, large-M work, non-RDNA3/4 targets, and a disabled gate retain conch's stock selector. | Recheck the conch version and patch anchors, `make_default_opt_flags_amd`, RDNA target detection, bitwidth/scale semantics, `target_kernel_kwargs`, and whether upstream has gained a gfx1151 decode profile. Compare generated kernels and output quality before carrying the values forward. |

The initial profile is adapted from
[AlexKGwyn/ds4-vllm-public](https://github.com/AlexKGwyn/ds4-vllm-public/tree/71a73d0c1ad42a51e8d4da7b3585a217917a4637),
but it is deliberately narrower: the local gate requires the exact scaled
BF16-by-MXFP4 case and does not import that project's unrelated routing,
target-query, or custom all-reduce changes.

The DeepSeek model profile enables `VLLM_GFX1X_MOE_TUNE=1`. To roll back this
optimization for a manual serve, set it to `0`. Optional overrides are
`VLLM_GFX1X_MOE_BM`, `VLLM_GFX1X_MOE_BN`, `VLLM_GFX1X_MOE_BK`,
`VLLM_GFX1X_MOE_NW`, `VLLM_GFX1X_MOE_NS`, and `VLLM_GFX1X_MOE_WPE`.

Initial controlled result on the 192 GiB gfx1151 `gh2` host, using the
development baseline above, TP=1, K7 greedy DSpark, a fixed prompt, 300 output
tokens, two warmups, and five measured requests:

| Profile | TTFT | Decode throughput |
|---|---:|---:|
| stock conch selector | `1.499 s` | `11.281 +/- 0.007 tok/s` |
| local MXFP4 profile | `0.778 s` | `21.766 +/- 0.016 tok/s` |

Memoizing conch target, backend, and CU-count queries was also tested with this
same profile. It measured `21.765 +/- 0.015 tok/s`, versus
`21.766 +/- 0.016 tok/s` without memoization, so the extra patch is not shipped.

This is a kernel/profile A/B, not a complete validation. A deterministic chat
sanity request produced a coherent answer and fixed-prompt DSpark draft
acceptance remained comparable to the baseline, but TP=2, long-context,
multi-request, benchmark-recall, and broader accuracy checks are still required
before promotion.

Build marker: `PATCHED: opt-in gfx1x MXFP4 MoE tuning`.

Tests: `tests/test_conch_moe_gfx1x_patch.py`.

## Native DSpark / DFlash block speculation

The model is `deepseek-ai/DeepSeek-V4-Flash-0731`. The launcher passes native
vLLM configuration with `method=dspark`, seven speculative tokens, greedy draft
sampling, unpadded draft batches, and eager draft execution. K7 greedy matches
DeepSeek's official 0731 vLLM recipe; the unpadded/eager settings remain local
gfx1151 launch policy. DSpark reuses vLLM's DFlash block-parallel machinery,
then samples left-to-right with its trained Markov head. There is deliberately
no local DSpark model implementation or vLLM patch.

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

## TP collective profiling

The cluster launcher exposes an opt-in `TP Collective Profile` toggle. It uses
vLLM's native Torch profiler rather than patching RCCL or the communicator. The
profiler is configured with tensor-shape recording, but stack and memory
tracking disabled, and remains dormant until the operator calls:

```bash
curl -X POST http://127.0.0.1:8000/start_profile
# Send one controlled inference request.
curl -X POST http://127.0.0.1:8000/stop_profile
vllm-collective-report
```

Each Ray rank writes its own compressed trace under
`~/.cache/vllm/profiles`. Run `vllm-collective-report` on both hosts, or pass
explicit trace paths, to summarize actual all-reduce/all-gather tensor shapes
and RCCL/NCCL GPU-kernel durations. Profiling is deliberately disabled by
default because Torch tracing materially perturbs latency. At an update,
recheck `ProfilerConfig`, the `/start_profile` and `/stop_profile` routes,
multi-rank trace fan-out, Chrome-trace shape keys, and RCCL kernel event names.

Tests: `tests/test_launcher_features.py` and
`tests/test_collective_report.py`.

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
