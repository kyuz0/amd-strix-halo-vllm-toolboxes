#!/usr/bin/env python3
"""RULER-style long-context recall probe for a live vLLM OpenAI endpoint.

Synthetic needle-in-a-haystack style checks sized to an exact context target.
When the server exposes /tokenize, prompts are corrected before generation;
otherwise (and additionally) the authoritative size comes from
``usage.prompt_tokens`` on each completion, with one grow-and-resend round
when a prompt lands under --min-token-frac (default 0.97) of the requested
context length. A "32k PASS" therefore really means a ~32k prompt was billed.
Complements the manifest's upgrade-procedure recall requirement
(docs/VLLM_PATCH_MANIFEST.md) with something runnable against any host.

Tasks:
  niah           one needle (magic number for a key) in neutral filler
  niah_multikey  several keyed needles, only the queried one matters
  vt             variable-tracking chain (VAR A = n; VAR B = A; ... )
  cwe            common-words extraction: 10 planted words among noise

Every task runs once per requested context length at a fixed needle placement
(50%). Scoring is deterministic (temperature 0). The JSON report records the
tokenized length actually sent; client-side timeouts are reported as ERROR
with guidance rather than silently counted as model failures.

Exit code: 0 when every task passes, 1 otherwise.

Examples:
  python3 benchmarks/recall_probe.py --base-url http://127.0.0.1:8000/v1 \
      --lengths 8192 32768
  python3 benchmarks/recall_probe.py --tasks niah vt --lengths 131072 \
      --output results/recall.json
"""

import argparse
import json
import random
import re
import sys

import requests

FILLER = (
    "The grass is green. The sky is blue. The sun is bright. The air is clean. "
    "The road is long. The wind is calm. The lake is cold. The trees are tall. "
)

NEEDLE_FMT = "One of the special magic numbers for {key} is: {value}."
NIAH_Q = (
    "What is the special magic number for {key} mentioned in the provided text?"
)

VT_SEED_VALUES = [
    "72513", "48290", "91034", "36572", "51847",
    "80293", "24615", "63708", "97421", "15836",
]

# Deliberately disjoint from the filler vocabulary above.
CWE_VOCAB = [
    "stone", "cloud", "ember", "harbor", "meadow", "lantern",
    "cinder", "orchid", "thistle", "anchor", "willow", "quartz", "fable",
    "beacon",
]
CWE_PASS_MIN = 8


def _post(base_url, path, payload, api_key, request_timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(
        base_url.rstrip("/") + path,
        json=payload,
        headers=headers,
        timeout=request_timeout,
    )


# vLLM serves /tokenize at the server root even when the OpenAI API sits
# under /v1; older builds exposed neither path. Cache whichever URL answers.
_TOKENIZE_URL = None


def _tokenize_url_candidates(base_url):
    base = base_url.rstrip("/")
    root = re.sub(r"/v1/?$", "", base)
    candidates = [base + "/tokenize", root + "/tokenize"]
    if _TOKENIZE_URL:
        candidates.insert(0, _TOKENIZE_URL)
    seen, ordered = set(), []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def count_tokens(base_url, model, api_key, request_timeout, text):
    """Tokenize via the server so context claims are exact, not estimated.

    Returns None when no tokenize endpoint answers; callers then fall back
    to post-hoc sizing from usage.prompt_tokens.
    """
    global _TOKENIZE_URL
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for url in _tokenize_url_candidates(base_url):
        try:
            r = requests.post(
                url,
                json={"model": model,
                      "messages": [{"role": "user", "content": text}]},
                headers=headers,
                timeout=request_timeout,
            )
            r.raise_for_status()
        except Exception:
            continue
        _TOKENIZE_URL = url
        body = r.json()
        count = body.get("count")
        if count is None:
            count = len(body.get("tokens") or [])
        return int(count)
    return None


def server_has_tokenize(base_url, model, api_key):
    return count_tokens(base_url, model, api_key, 15, "ping") is not None


def _stable_task_offset(name):
    # PYTHONHASHSEED-randomized hash() would make runs unreproducible.
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 100000


def _filler(units):
    return FILLER * max(0, int(units))


class Tasks:
    @staticmethod
    def niah(seed):
        rng = random.Random(seed)
        key = f"word-{rng.randrange(10**6):06d}"
        value = str(rng.randrange(10**7)).zfill(7)
        needle = NEEDLE_FMT.format(key=key, value=value)

        def make(place_frac, filler_units):
            cut = filler_units * place_frac
            return (
                _filler(cut) + needle + _filler(filler_units - cut),
                value,
            )

        question = NIAH_Q.format(key=key)
        return make, question, lambda resp, exp: exp in resp

    @staticmethod
    def niah_multikey(seed):
        rng = random.Random(seed)
        wanted_idx = rng.randrange(4)
        pairs = []
        for i in range(4):
            k = f"key-{rng.randrange(10**6):06d}"
            v = str(rng.randrange(10**7)).zfill(7)
            pairs.append((k, v))

        def make(place_frac, filler_units):
            per = filler_units // 4
            chunks = []
            for i, (k, v) in enumerate(pairs):
                cut = per * place_frac
                chunks.append(
                    _filler(cut) + NEEDLE_FMT.format(key=k, value=v)
                    + _filler(per - cut)
                )
            return "".join(chunks), pairs[wanted_idx][1]

        question = NIAH_Q.format(key=pairs[wanted_idx][0])
        return make, question, lambda resp, exp: exp in resp

    @staticmethod
    def vt(seed):
        rng = random.Random(seed)
        chain_len = 16
        names = [f"VAR{i}" for i in range(chain_len)]
        head_value = VT_SEED_VALUES[rng.randrange(len(VT_SEED_VALUES))]
        block_lines = [f"VAR {names[0]} = {head_value}."]
        block_lines += [f"VAR {n} = {p}." for p, n in zip(names, names[1:])]
        block = "\n".join(block_lines)

        def make(place_frac, filler_units):
            cut = filler_units * place_frac
            return (
                _filler(cut) + "\n" + block + "\n" + _filler(filler_units - cut),
                head_value,
            )

        question = (
            f"Follow the variable assignments in the provided text and report "
            f"the numeric value that {names[-1]} refers to."
        )
        return make, question, lambda resp, exp=head_value: exp in resp

    @staticmethod
    def cwe(seed):
        rng = random.Random(seed)
        words = rng.sample(CWE_VOCAB, 10)

        def make(place_frac, filler_units):
            gap = max(1, filler_units // (len(words) + 2))
            pieces = []
            idx = 0
            for u in range(filler_units):
                pieces.append(FILLER)
                if idx < len(words) and u == gap * (idx + 1):
                    pieces.append(f"The {words[idx]} remembers the old song. ")
                    idx += 1
            return "".join(pieces), words

        question = (
            "Ten uncommon words were deliberately placed throughout the "
            "provided text, each followed by the phrase 'remembers the old "
            "song'. List those ten words as a comma-separated list."
        )

        def score(resp, expected):
            low = resp.lower()
            return sum(1 for w in expected if w in low)

        return make, question, score


TASKS = {
    "niah": Tasks.niah,
    "niah_multikey": Tasks.niah_multikey,
    "vt": Tasks.vt,
    "cwe": Tasks.cwe,
}


def run_one(base_url, model, api_key, request_timeout, max_tokens,
            task_name, target_tokens, seed, min_frac, thinking,
            use_tokenize, cal, place_frac=0.5):
    offset = _stable_task_offset(task_name)
    make, question, scorer = TASKS[task_name](seed + target_tokens + offset)

    def build(units):
        text, expected = make(place_frac, units)
        return (f"{text}\n\nQuestion: {question}\nAnswer:", expected)

    def send(full):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": full}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if not thinking:
            # Reasoning would burn the token budget before the answer;
            # recall probes want direct retrieval.
            payload["chat_template_kwargs"] = {"thinking": False}
        r = _post(base_url, "/chat/completions", payload,
                  api_key, request_timeout)
        r.raise_for_status()
        return r.json()

    # Calibrated filler sizing: tokens per FILLER unit, learned from the
    # first completion's usage and reused for every later row in this run.
    if cal.get("tokens_per_unit"):
        est_units = max(64, int(target_tokens * 0.99 / cal["tokens_per_unit"]))
    else:
        est_units = max(64, int(target_tokens * min_frac) // 32)
    full, expected = build(est_units)

    if use_tokenize:
        # Correct the estimate before paying for a prefill.
        tok = count_tokens(base_url, model, api_key, request_timeout, full)
        tries = 0
        while tok is not None and tries < 8:
            cal["tokens_per_unit"] = max(tok, 1) / max(est_units, 1)
            if tok < target_tokens * min_frac:
                est_units = int(est_units * (target_tokens * 0.99) / max(tok, 1)) + 8
            elif tok > target_tokens * 1.03:
                est_units = max(64, int(est_units * (target_tokens * 0.99) / tok))
            else:
                break
            full, expected = build(est_units)
            tok = count_tokens(base_url, model, api_key, request_timeout, full)
            tries += 1

    row = {
        "task": task_name,
        "requested_context": target_tokens,
    }

    def absorb(body):
        """Record usage + answer; returns prompt_tokens (0 if absent)."""
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if prompt_tokens and est_units:
            cal["tokens_per_unit"] = prompt_tokens / est_units
        content = (body["choices"][0]["message"]["content"] or "")
        return prompt_tokens, content

    try:
        body = send(full)
        prompt_tokens, content = absorb(body)
        resized = None
        if prompt_tokens and not (min_frac * target_tokens
                                  <= prompt_tokens <= target_tokens * 1.03):
            # One corrective resend: grow when under target, shrink when over.
            resized = "grow" if prompt_tokens < min_frac * target_tokens else "shrink"
            est_units = max(
                64, int(est_units * (target_tokens * 0.99) / prompt_tokens))
            full, expected = build(est_units)
            body = send(full)
            prompt_tokens, content = absorb(body)
        row["tokenized_context"] = prompt_tokens
        if resized:
            row["resized"] = resized
        row["response_excerpt"] = content[:200]
        if task_name == "cwe":
            found = scorer(content, expected)
            row["found"] = found
            row["of"] = len(expected)
            row["status"] = "PASS" if found >= CWE_PASS_MIN else "FAIL"
        else:
            hit = scorer(content, expected)
            row["expected"] = expected
            row["status"] = "PASS" if hit else "FAIL"
    except requests.exceptions.Timeout:
        row["status"] = "ERROR"
        row["tokenized_context"] = None
        row["error"] = (
            f"client timeout after {request_timeout}s - raise "
            "--request-timeout; this is not scored as a model failure"
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "ERROR"
        row["tokenized_context"] = None
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def main():
    p = argparse.ArgumentParser(description="RULER-style long-context recall probe")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None,
                   help="served model id; defaults to first entry of /v1/models")
    p.add_argument("--api-key", default=None)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=sorted(TASKS))
    p.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768],
                   help="context targets in tokens (must fit max_model_len)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--request-timeout", type=float, default=900,
                   help="per-request client timeout in seconds")
    p.add_argument("--min-token-frac", type=float, default=0.97,
                   help="/tokenize must confirm at least this fraction")
    p.add_argument("--thinking", action="store_true",
                   help="leave the server's thinking mode enabled; default "
                        "disables it per request so the budget reaches the answer")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", default=None, help="write the JSON report here")
    args = p.parse_args()

    model = args.model
    if not model:
        r = requests.get(args.base_url.rstrip("/") + "/models", timeout=30)
        r.raise_for_status()
        model = r.json()["data"][0]["id"]
        print(f"discovered model: {model}")

    use_tokenize = server_has_tokenize(args.base_url, model, args.api_key)
    if not use_tokenize:
        print("no /tokenize on this server; sizing verified post-hoc via "
              "usage.prompt_tokens (one corrective resend when outside target)")

    cal = {"tokens_per_unit": None}
    results = []
    for target in args.lengths:
        for task in args.tasks:
            row = run_one(args.base_url, model, args.api_key,
                          args.request_timeout, args.max_tokens,
                          task, target, args.seed, args.min_token_frac,
                          args.thinking, use_tokenize, cal)
            results.append(row)
            brief = row.get("response_excerpt") or row.get("error", "")
            extra = ""
            if task == "cwe" and "found" in row:
                extra = f" found={row['found']}/{row['of']}"
            print(f"[{row['status']:>5}] {task:>13} @{target:>8}: "
                  f"tok={row['tokenized_context']}{extra} | {brief[:70]}")

    ok = all(r["status"] == "PASS" for r in results)
    report = {
        "model": model,
        "base_url": args.base_url,
        "min_token_frac": args.min_token_frac,
        "passed": ok,
        "results": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report: {args.output}")
    print("RECALL PROBE:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
