# SPDX-License-Identifier: Apache-2.0
# Adapted from AlexKGwyn/ds4-vllm-public commit
# 95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff.
"""Deterministic row-wise top-k with ascending output for gfx1x sparse attention.

Replaces vLLM's atomic ``top_k_per_row_prefill`` and
``top_k_per_row_decode`` calls on the explicitly enabled gfx1x path. The
Triton kernel selects the top ``k`` entries of each row under the total order
``(value descending, column index ascending)`` and writes them **already sorted
ascending by column index**, ``-1``-padded at the tail.

Determinism
-----------
No floating-point accumulation and no atomic anywhere. The kernel computes

  1. a threshold bracket ``[lo, hi]`` plus a tie budget ``need``, by radix
     selection over integer histograms (integer addition is associative and
     commutative, so reduction order cannot change the counts), and
  2. the output slot of each selected element as an exact integer prefix count
     taken in column order.

Both are pure functions of the input row, so the output is bit-identical run to
run, launch to launch, and under any thread/block schedule. See ``DESIGN.md``.

Conventions follow ``gfx1x_tilelang_mqa.py``: a model-scoped environment gate
and — the point of the bucketing discipline there — **no shape-dependent
compile key**. Every size is a runtime argument marked
``do_not_specialize``, so at most two kernel variants are ever built (prefill
with row bounds, decode without) and no decode step can trigger a JIT stall.
"""

import os

import torch
import triton
import triton.language as tl

# The DeepSeek model profile enables this. The integration layer retains
# upstream top_k_per_row_* whenever it is disabled or unavailable.
_ENABLED = os.environ.get("VLLM_GFX1X_RADIX_TOPK", "0") == "1"

# Radix configuration.
#   "hist" -> 8-bit digits, 256 bins, tl.histogram, <= 4 refinement levels
#   "sum"  -> 4-bit digits,  16 bins, masked tl.sum, <= 8 refinement levels
# "sum" exists because tl.histogram's lowering is the only part of this kernel
# whose ROCm-backend support is not obvious from the Python source. It is slower
# but uses only reductions every backend implements. Both emit identical output;
# The gfx1151 validation checks that both modes emit identical output.
_HIST_MODE = os.environ.get("VLLM_GFX1X_RADIX_TOPK_HIST", "hist")

# Elements per tile of the streaming loop. 2048 fp32 = 8 KiB per iteration; with
# num_warps=8 (512 lanes at wave64) that is 4 elements/lane, which holds the
# tl.cumsum in the emit pass to 11 shuffle steps.
_TILE = 2048
_NUM_WARPS = 8
_PAD_TILE = 1024  # >= any topk we serve (512); granularity of the -1 tail fill

_I32_MIN = -2147483648


# --------------------------------------------------------------------------- #
# key transform
# --------------------------------------------------------------------------- #
@triton.jit
def _okey(v):
    """Order-preserving float32 -> int32 key.

    Signed int32 comparison on the result reproduces IEEE-754 order on the
    input: ``key(a) < key(b) <=> a < b`` for all non-NaN ``a``, ``b``.

    ``-0.0`` is normalised to ``+0.0`` first, so the two tie and the tie is then
    broken on column index, which is the documented contract. (torch's own
    behaviour here is already self-inconsistent: its CUDA radix path orders
    ``-0.0`` strictly below ``+0.0`` for wide rows while its bitonic path, taken
    for rows <= 4096, treats them as equal. Neither can arise from
    ``sum(relu(q.k) * w) * scale``, whose accumulator starts at ``+0.0``.)

    ``-inf`` maps to the smallest key of any non-NaN input, which is what lets
    the emit pass drop it with a single comparison.
    """
    v = tl.where(v == 0.0, 0.0, v)  # -0.0 == 0.0 is true, so this normalises it
    b = v.to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _digit_hist(d, cand, BINS: tl.constexpr, USE_HIST: tl.constexpr):
    """Counts of digits ``d`` over the lanes where ``cand`` is true."""
    if USE_HIST:
        h = tl.histogram(d, BINS, mask=cand)
    else:
        h = tl.zeros([BINS], tl.int32)
        bidx = tl.arange(0, BINS)
        for j in tl.static_range(BINS):
            c = tl.sum((cand & (d == j)).to(tl.int32))
            h = tl.where(bidx == j, c, h)
    return h


# --------------------------------------------------------------------------- #
# kernel — one workgroup per row
# --------------------------------------------------------------------------- #
@triton.jit(
    do_not_specialize=[
        "width",
        "topk",
        "logit_stride_row",
        "out_stride_row",
    ]
)
def _topk_rows_kernel(
    logits_ptr,  # [n_rows, width] fp32, last dim contiguous
    out_ptr,  # [n_rows, topk]  int32, last dim contiguous
    ks_ptr,  # [n_rows] int32 (row scan start, inclusive) or unused
    ke_ptr,  # [n_rows] int32 (row scan end, exclusive)   or unused
    logit_stride_row,
    out_stride_row,
    width,
    topk,
    HAS_KS: tl.constexpr,
    HAS_KE: tl.constexpr,
    LOCAL_INDEX: tl.constexpr,
    TILE: tl.constexpr,
    BINS: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    LEVELS: tl.constexpr,
    USE_HIST: tl.constexpr,
    PAD_TILE: tl.constexpr,
):
    r = tl.program_id(0)
    base = logits_ptr + r.to(tl.int64) * logit_stride_row
    obase = out_ptr + r.to(tl.int64) * out_stride_row

    # ---- per-row scan bounds ------------------------------------------------
    # Columns outside [ks, ke) are already -inf (every logits producer masks
    # them), so honouring the bounds is a work saving, not a semantic change --
    # and it also makes this kernel independent of that masking.
    raw_ks = tl.full((), 0, tl.int32)
    col_lo = tl.full((), 0, tl.int32)
    col_hi = width.to(tl.int32)
    if HAS_KS:
        raw_ks = tl.load(ks_ptr + r).to(tl.int32)
        col_lo = tl.maximum(raw_ks, 0)
    if HAS_KE:
        col_hi = tl.minimum(tl.load(ke_ptr + r).to(tl.int32), col_hi)
    col_hi = tl.maximum(col_hi, 0)
    col_lo = tl.minimum(col_lo, col_hi)

    row_start = tl.full((), 0, tl.int32)
    if LOCAL_INDEX:
        row_start = raw_ks

    # ---- radix selection ----------------------------------------------------
    # Invariant across levels:
    #   n_gt = #{ j : v_j != -inf and key_j >  hi }               (< topk)
    #   need = topk - n_gt                                        (> 0)
    #   cand = { j : v_j != -inf and lo <= key_j <= hi },  |cand| >= need
    # ulo/uhi hold the bracket in the biased-unsigned domain (key + 2**31) as
    # int64 scalars, so the digit prefix accumulates without int32 overflow.
    key_min = tl.full((), -2147483647, tl.int32) - 1  # int32 min, built safely
    key_max = tl.full((), 2147483647, tl.int32)
    ulo = tl.full((), 0, tl.int64)
    uhi = tl.full((), 4294967295, tl.int64)
    lo = key_min
    hi = key_max
    n_gt = tl.full((), 0, tl.int32)
    # An empty window selects nothing: force need = 0 so n_sel = 0 and the tail
    # fill below writes -1 across the whole output row.
    empty = col_lo >= col_hi
    need = tl.where(empty, 0, topk.to(tl.int32))
    done = empty.to(tl.int32)

    # Declared before the level loop so the value carried out of each `if` has
    # an unambiguous type; only ever read inside the block that writes them.
    hist = tl.zeros([BINS], tl.int32)
    m_cnt = tl.full((), 0, tl.int32)
    kmin = key_max
    kmax = key_min

    for level in tl.static_range(LEVELS):
        if done == 0:
            shift = 32 - RADIX_BITS * (level + 1)
            hist = tl.zeros([BINS], tl.int32)
            m_cnt = tl.full((), 0, tl.int32)
            kmin = key_max
            kmax = key_min

            for t in tl.range(col_lo, col_hi, TILE):
                offs = t + tl.arange(0, TILE)
                inr = offs < col_hi
                v = tl.load(base + offs, mask=inr, other=float("-inf"))
                key = _okey(v)
                cand = inr & (v != float("-inf")) & (key >= lo) & (key <= hi)
                m_cnt += tl.sum(cand.to(tl.int32))
                kmin = tl.minimum(kmin, tl.min(tl.where(cand, key, key_max)))
                kmax = tl.maximum(kmax, tl.max(tl.where(cand, key, key_min)))
                d = (key >> shift) & (BINS - 1)
                if shift == 32 - RADIX_BITS:
                    # Top digit carries the sign bit; flip it so unsigned digit
                    # order matches signed key order.
                    d = d ^ (BINS // 2)
                hist += _digit_hist(d, cand, BINS, USE_HIST)

            if m_cnt <= need:
                # Every remaining candidate is selected. At level 0 this is the
                # "row has fewer than topk finite entries" case; deeper down it
                # is "the winning bin is exactly the tie budget".
                need = m_cnt
                done = tl.full((), 1, tl.int32)
            elif kmin == kmax:
                # All candidates carry the same value: no further digit can
                # separate them, so the tie resolves purely on column index.
                ulo = kmin.to(tl.int64) + 2147483648
                uhi = ulo
                lo = kmin
                hi = kmin
                done = tl.full((), 1, tl.int32)
            else:
                bidx = tl.arange(0, BINS)
                cum_incl = tl.cumsum(hist, axis=0)  # #cand with digit <= b
                n_above = m_cnt - cum_incl  # #cand with digit >  b
                # n_above is non-increasing in b, so this predicate flips once.
                dstar = tl.min(tl.where(n_above < need, bidx, BINS))
                pick = bidx == dstar
                cum_above = tl.sum(tl.where(pick, n_above, 0))
                n_gt += cum_above
                need -= cum_above
                ulo = ulo | (dstar.to(tl.int64) << shift)
                uhi = ulo | ((tl.full((), 1, tl.int64) << shift) - 1)
                lo = (ulo - 2147483648).to(tl.int32)
                hi = (uhi - 2147483648).to(tl.int32)

    n_sel = n_gt + need

    # ---- ordered emit -------------------------------------------------------
    # Walk the row in ascending column order. For column j let
    #   A(j) = #{ i < j : key_i >  hi }          (unconditionally selected)
    #   E(j) = #{ i < j : lo <= key_i <= hi }    (the tie band)
    # j is selected iff key_j > hi, or it is in the band and E(j) < need. Its
    # output slot is A(j) + min(E(j), need) -- its rank in column order among
    # the selected set. Both are exact integer prefix sums, so the slot map is a
    # bijection onto [0, n_sel) that no scheduling choice can perturb, and the
    # output is ascending by construction: no second sort.
    a_run = tl.full((), 0, tl.int32)
    e_run = tl.full((), 0, tl.int32)
    for t in tl.range(col_lo, col_hi, TILE):
        offs = t + tl.arange(0, TILE)
        inr = offs < col_hi
        v = tl.load(base + offs, mask=inr, other=float("-inf"))
        key = _okey(v)
        fin = inr & (v != float("-inf"))
        above = fin & (key > hi)
        band = fin & (key >= lo) & (key <= hi)
        ai = above.to(tl.int32)
        bi = band.to(tl.int32)
        a_abs = a_run + tl.cumsum(ai, axis=0) - ai
        e_abs = e_run + tl.cumsum(bi, axis=0) - bi
        sel = above | (band & (e_abs < need))
        slot = a_abs + tl.minimum(e_abs, need)
        tl.store(obase + slot, (offs - row_start).to(tl.int32), mask=sel)
        a_run += tl.sum(ai)
        e_run += tl.sum(bi)

    # ---- -1 tail ------------------------------------------------------------
    # Disjoint from every slot written above, so no ordering constraint.
    for p in tl.range(0, topk, PAD_TILE):
        s = p + tl.arange(0, PAD_TILE)
        tl.store(obase + s, -1, mask=(s >= n_sel) & (s < topk))


# --------------------------------------------------------------------------- #
# python wrapper
# --------------------------------------------------------------------------- #
def _radix_config():
    if _HIST_MODE == "sum":
        return 16, 4, 8, False
    return 256, 8, 4, True


def topk_indices_ascending(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-``topk_tokens`` columns of each row of ``logits``, ascending.

    Selection order is ``(value descending, column index ascending)``. ``-inf``
    columns are never selected. The result is sorted ascending by column index
    with ``-1`` padding at the tail -- i.e. exactly what ``_topk_indices_torch``
    followed by ``_canonicalize_topk_indices_`` produces today.

    ``row_starts`` (``cu_seqlen_ks``) makes the returned indices *local* to each
    row's valid range, matching the CUDA ``top_k_per_row_prefill`` contract.
    ``row_ends`` (``cu_seqlen_ke``) is optional and only bounds the scan; it is
    correctness-neutral because those columns are already ``-inf``.

    ``out`` lets the caller write straight into ``topk_indices_buffer``, which
    also removes the ``.copy_()`` the current path pays.
    """
    assert logits.dim() == 2, f"expected [rows, width] logits, got {tuple(logits.shape)}"
    assert logits.dtype == torch.float32, f"expected fp32 logits, got {logits.dtype}"
    rows, width = logits.shape

    if out is None:
        out = torch.empty((rows, topk_tokens), dtype=torch.int32, device=logits.device)
    else:
        assert tuple(out.shape) == (rows, topk_tokens), (
            f"out {tuple(out.shape)} != ({rows}, {topk_tokens})"
        )
        assert out.dtype == torch.int32
    if rows == 0:
        return out
    if width == 0 or topk_tokens == 0:
        out.fill_(-1)
        return out

    assert logits.stride(-1) == 1, "topk kernel needs contiguous logits rows"
    assert out.stride(-1) == 1, "topk kernel needs contiguous output rows"

    ks = row_starts
    ke = row_ends
    if ks is not None:
        ks = ks.to(device=logits.device, dtype=torch.int32).contiguous()
        assert ks.numel() == rows, f"row_starts {ks.numel()} != rows {rows}"
    if ke is not None:
        ke = ke.to(device=logits.device, dtype=torch.int32).contiguous()
        assert ke.numel() == rows, f"row_ends {ke.numel()} != rows {rows}"

    bins, radix_bits, levels, use_hist = _radix_config()
    unused = out  # never dereferenced when the matching HAS_* flag is False

    _topk_rows_kernel[(rows,)](
        logits,
        out,
        ks if ks is not None else unused,
        ke if ke is not None else unused,
        logits.stride(0),
        out.stride(0),
        width,
        int(topk_tokens),
        HAS_KS=ks is not None,
        HAS_KE=ke is not None,
        LOCAL_INDEX=ks is not None,
        TILE=_TILE,
        BINS=bins,
        RADIX_BITS=radix_bits,
        LEVELS=levels,
        USE_HIST=use_hist,
        PAD_TILE=_PAD_TILE,
        num_warps=_NUM_WARPS,
    )
    return out


# --------------------------------------------------------------------------- #
# reference — the contract, spelled out (tests only; slow)
# --------------------------------------------------------------------------- #
def _okey_torch(v: torch.Tensor) -> torch.Tensor:
    """Torch twin of :func:`_okey`."""
    v = torch.where(v == 0.0, torch.zeros_like(v), v)
    b = v.contiguous().view(torch.int32)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


def topk_indices_ascending_reference(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    """Definitional reference for :func:`topk_indices_ascending`.

    Written on the same integer key transform the kernel uses, so the tie rule
    (lower column index wins) and the ``-0.0`` normalisation are stated once.
    """
    rows, width = logits.shape
    dev = logits.device
    out = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=dev)
    for r in range(rows):
        lo = 0 if row_starts is None else max(0, int(row_starts[r]))
        hi = width if row_ends is None else min(width, int(row_ends[r]))
        hi = max(hi, 0)
        lo = min(lo, hi)
        if hi <= lo:
            continue
        row = logits[r, lo:hi]
        finite = row != float("-inf")
        if not bool(finite.any()):
            continue
        cols = torch.arange(lo, hi, device=dev, dtype=torch.int64)[finite]
        key = _okey_torch(row)[finite].to(torch.int64)
        # Stable sort on the negated integer key: equal keys keep input order,
        # and input order is ascending column index -> ties take the lower one.
        order = torch.argsort(-key, stable=True)
        sel = cols[order][:topk_tokens]
        sel, _ = torch.sort(sel)
        if row_starts is not None:
            sel = sel - int(row_starts[r])
        out[r, : sel.numel()] = sel.to(torch.int32)
    return out


# --------------------------------------------------------------------------- #
# drop-in for vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
# --------------------------------------------------------------------------- #
def topk_indices_torch_fallback(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """The current production two-sort path, verbatim, for A/B and fallback."""
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.sort(logits, dim=-1, descending=True, stable=True)
    values = values[:, :k]
    indices = indices[:, :k].to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    if row_starts is not None:
        starts = row_starts.to(dtype=torch.int32).view(-1, 1)
        indices = torch.where(indices < 0, indices, indices - starts)
    if k != topk_tokens:
        padded = torch.full(
            (logits.shape[0], topk_tokens),
            -1,
            dtype=torch.int32,
            device=logits.device,
        )
        padded[:, :k] = indices
        indices = padded
    sentinel = torch.iinfo(indices.dtype).max
    sortable = torch.where(indices >= 0, indices, sentinel)
    ordered = torch.sort(sortable, dim=-1, stable=True).values
    return torch.where(ordered == sentinel, -1, ordered)


def select_topk(
    logits: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entry point used by ``rocm_aiter_mla_sparse``."""
    if not _ENABLED or not logits.is_cuda:
        res = topk_indices_torch_fallback(logits, topk_tokens, row_starts)
        if out is not None:
            out.copy_(res)
            return out
        return res
    return topk_indices_ascending(
        logits, topk_tokens, row_starts=row_starts, row_ends=row_ends, out=out
    )
