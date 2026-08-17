import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vllm_collective_report  # noqa: E402


class CollectiveReportTests(unittest.TestCase):
    def test_extracts_collective_shape_and_gpu_kernel_duration(self):
        trace = {
            "traceEvents": [
                {
                    "name": "c10d::allreduce_",
                    "cat": "cpu_op",
                    "dur": 4.5,
                    "args": {
                        "Input Dims": [[[2, 7168]]],
                        "Input type": [["c10::BFloat16"]],
                    },
                },
                {
                    "name": "ncclDevKernel_AllReduce_Sum_bf16",
                    "cat": "kernel",
                    "dur": 31.25,
                    "args": {"stream": 7},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rank0.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as trace_file:
                json.dump(trace, trace_file)
            calls, kernels = vllm_collective_report.summarize([path])

        self.assertEqual(len(calls), 1)
        key = next(iter(calls))
        self.assertEqual(key[1], (2, 7168))
        self.assertEqual(key[2], "bfloat16")
        self.assertEqual(key[3], 2 * 7168 * 2)
        self.assertEqual(calls[key]["count"], 1)
        self.assertEqual(
            kernels["ncclDevKernel_AllReduce_Sum_bf16"], [31.25]
        )


if __name__ == "__main__":
    unittest.main()
