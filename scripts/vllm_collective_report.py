#!/usr/bin/env python3
"""Summarize collective tensor shapes from vLLM Torch-profiler traces."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path


COLLECTIVE_RE = re.compile(
    r"all.?reduce|all.?gather|reduce.?scatter|broadcast",
    re.IGNORECASE,
)
GPU_COLLECTIVE_RE = re.compile(
    r"nccl|rccl",
    re.IGNORECASE,
)
DTYPE_BYTES = {
    "bool": 1,
    "byte": 1,
    "char": 1,
    "uint8": 1,
    "int8": 1,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "short": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "int": 4,
    "int32": 4,
    "double": 8,
    "float64": 8,
    "long": 8,
    "int64": 8,
}


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _first_shape(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(value, list):
        if value and all(isinstance(item, int) for item in value):
            return tuple(value)
        for item in value:
            shape = _first_shape(item)
            if shape is not None:
                return shape
    return None


def _first_dtype(value):
    if isinstance(value, list):
        for item in value:
            dtype = _first_dtype(item)
            if dtype:
                return dtype
        return "unknown"
    text = str(value or "unknown").split("::")[-1].lower()
    return text


def _tensor_bytes(shape, dtype):
    item_size = DTYPE_BYTES.get(dtype)
    if shape is None or item_size is None:
        return None
    return math.prod(shape) * item_size


def summarize(paths):
    calls = defaultdict(lambda: {"count": 0, "durations": []})
    kernels = defaultdict(list)
    for path in paths:
        with _open_trace(path) as trace_file:
            data = json.load(trace_file)
        for event in data.get("traceEvents", []):
            name = str(event.get("name", ""))
            duration = float(event.get("dur", 0.0))
            args = event.get("args") or {}
            category = str(event.get("cat", "")).lower()
            if GPU_COLLECTIVE_RE.search(name) and (
                "kernel" in category or "stream" in args
            ):
                kernels[name].append(duration)
            if not COLLECTIVE_RE.search(name):
                continue
            shape = _first_shape(
                args.get("Input Dims", args.get("Input dims"))
            )
            if shape is None:
                continue
            dtype = _first_dtype(
                args.get("Input type", args.get("Input Type"))
            )
            key = (name, shape, dtype, _tensor_bytes(shape, dtype))
            calls[key]["count"] += 1
            calls[key]["durations"].append(duration)
    return calls, kernels


def _human_bytes(value):
    if value is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path.home() / ".cache" / "vllm" / "profiles"],
    )
    args = parser.parse_args()
    traces = []
    for path in args.paths:
        if path.is_dir():
            traces.extend(path.rglob("*.pt.trace.json*"))
        elif path.is_file():
            traces.append(path)
    if not traces:
        parser.error("no *.pt.trace.json[.gz] files found")

    calls, kernels = summarize(sorted(set(traces)))
    print(f"Traces: {len(set(traces))}")
    print("\nCollective call shapes (CPU launch duration, not fabric latency):")
    print(f"{'count':>7}  {'bytes':>10}  {'dtype':>10}  {'shape':>22}  operation")
    for (name, shape, dtype, size), stats in sorted(
        calls.items(), key=lambda item: (item[0][3] or -1, item[0][0])
    ):
        print(
            f"{stats['count']:7d}  {_human_bytes(size):>10}  {dtype:>10}  "
            f"{str(shape):>22}  {name}"
        )

    print("\nRCCL/NCCL GPU kernels (actual device duration):")
    print(f"{'count':>7}  {'mean us':>10}  {'max us':>10}  kernel")
    for name, durations in sorted(kernels.items()):
        print(
            f"{len(durations):7d}  {sum(durations) / len(durations):10.1f}  "
            f"{max(durations):10.1f}  {name}"
        )


if __name__ == "__main__":
    main()
