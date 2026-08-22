#!/usr/bin/env python3
"""Speculative-decoding acceptance meter for a live vLLM OpenAI endpoint.

Snapshots the Prometheus /metrics endpoint, drives a fixed burst of chat
completions, snapshots again, and reports the delta of every
``vllm:spec_decode_*`` counter -- overall accepted/draft ratio plus the
per-position acceptance curve. This is the number every kernel or launch-policy
change must hold steady against: a faster engine that quietly loses draft
acceptance is a net loss.

Counter semantics differ slightly across vLLM versions (some count the bonus
token in drafts, some do not), so raw deltas are always printed alongside any
derived ratio. Per-position counters only advance at position k when all
earlier draft positions were accepted, so ``per_pos[k] / per_pos[1]``
approximates P(at least k draft tokens accepted).

Exit code: 0 normally, including when speculative decoding appears disabled
(no spec_decode metrics found -- guidance printed). Exit 2 on transport errors.

Example:
  python3 benchmarks/spec_acceptance.py --base-url http://127.0.0.1:8000/v1 \
      --num-requests 8 --concurrency 4 --output results/acceptance.json
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

COUNTER_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+eE0-9.]+)\s*$"
)
SPEC_PREFIX = "vllm:spec_decode_"
POS_RE = re.compile(r"_per_pos(\d+)$")


def _metrics_url(base_url):
    # /metrics is served at the server root even when the OpenAI API sits
    # under /v1, so accept either form of --base-url.
    return re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/metrics"


def snapshot_spec_counters(base_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(_metrics_url(base_url), headers=headers, timeout=30)
    r.raise_for_status()
    totals = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        m = COUNTER_RE.match(line.strip())
        if not m:
            continue
        name, _, value = m.group(1), m.group(2), m.group(3)
        if not name.startswith(SPEC_PREFIX):
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        # Aggregate across label sets (e.g. per-model labels).
        totals[name] = totals.get(name, 0.0) + v
    return totals


def drive_burst(base_url, model, api_key, prompt, num_requests, concurrency,
                max_tokens, request_timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def one(i):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt}\n(variation {i})"}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        r = requests.post(base_url.rstrip("/") + "/chat/completions",
                          json=payload, headers=headers,
                          timeout=request_timeout)
        r.raise_for_status()
        body = r.json()
        # Count billed completion tokens rather than visible content:
        # reasoning-mode servers can return null content while still
        # generating (and drafting) the full budget.
        usage = body.get("usage") or {}
        return int(usage.get("completion_tokens") or 0)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        outs = list(pool.map(one, range(num_requests)))
    return outs


def fmt(n):
    return f"{n:.0f}"


def report(before, after, output_path):
    keys = sorted(set(before) | set(after))
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}
    changed = {k: v for k, v in deltas.items() if abs(v) > 0}

    print("\n-- spec_decode counter deltas over the measurement window --")
    for k in sorted(changed):
        print(f"{k[len(SPEC_PREFIX):]:<44} {fmt(changed[k]):>12}")

    pos = {}
    other = {}
    for k, v in changed.items():
        base = k[len(SPEC_PREFIX):]
        m = POS_RE.search(base)
        if m:
            pos[int(m.group(1))] = v
        else:
            other[base] = v

    drafts = next((v for n, v in other.items() if "draft_tokens" in n), None)
    accepted = next((v for n, v in other.items() if "accepted_tokens" in n
                     and "per_pos" not in n), None)

    if not changed:
        print("no spec_decode counters moved during the burst; either "
              "speculative decoding is off, the burst was too small "
              "(raise --num-requests/--max-tokens), or metrics are stale")
    elif drafts is None and accepted is None and not pos:
        print("spec_decode counters present but no recognized "
              "draft/accepted tokens counters; inspect the deltas above")

    if drafts and accepted is not None and drafts > 0:
        print(f"\noverall accepted/draft = {accepted / drafts:.4f} "
              f"({fmt(accepted)} / {fmt(drafts)}; ratio includes any "
              f"version-specific bonus-token accounting)")
    if pos:
        first = pos.get(1, 0.0)
        print("\ndraft-position acceptance curve (pos k reached only if "
              "positions < k were accepted):")
        for k in sorted(pos):
            rel = f"{pos[k] / first:6.3f}" if first else "   n/a"
            print(f"  pos {k}: {fmt(pos[k]):>10}   rel_to_pos1={rel}")

    if output_path:
        doc = {"deltas": changed, "draft_tokens": drafts,
               "accepted_tokens": accepted, "per_position": pos}
        with open(output_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"\nreport: {output_path}")


def main():
    p = argparse.ArgumentParser(description="speculative-decode acceptance meter")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None,
                   help="served model id; defaults to first entry of /v1/models")
    p.add_argument("--api-key", default=None)
    p.add_argument("--prompt", default=(
        "Write a detailed step-by-step recipe for baking sourdough bread, "
        "including timings, temperatures, and common failure modes."))
    p.add_argument("--num-requests", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256,
                   help="generation cap; keep high enough that drafting has room")
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--output", default=None, help="write the JSON report here")
    args = p.parse_args()

    try:
        before = snapshot_spec_counters(args.base_url, args.api_key)
        if not before:
            print("no vllm:spec_decode_* metrics found on this server.")
            print("If you expected speculative decoding here, check that the ")
            print("server was launched with a --speculative-config and that ")
            print("/metrics exposes spec-decode counters on this version.")
            return 0

        model = args.model
        if not model:
            r = requests.get(args.base_url.rstrip("/") + "/models", timeout=30)
            r.raise_for_status()
            model = r.json()["data"][0]["id"]
            print(f"discovered model: {model}")

        lens = drive_burst(args.base_url, model, args.api_key, args.prompt,
                           args.num_requests, args.concurrency,
                           args.max_tokens, args.request_timeout)
        print(f"burst complete: {len(lens)} requests, "
              f"{sum(lens)} completion tokens total")

        after = snapshot_spec_counters(args.base_url, args.api_key)
        report(before, after, args.output)
        return 0
    except requests.exceptions.RequestException as exc:
        print(f"ERROR talking to {args.base_url}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
