#!/usr/bin/env python3
"""Best-effort post-start kernel warmup for the local vLLM OpenAI API.

The helper is spawned immediately before the launcher execs vLLM. It waits for
``/v1/models``, then sends a tiny request to exercise decode/speculation and an
optional longer request to exercise chunked prefill and the TileLang indexer.
It never changes server health or makes startup fail.

Adapted from ``host/ds4-vllm-warmup.py`` in AlexKGwyn/ds4-vllm-public commit
71a73d0c1ad42a51e8d4da7b3585a217917a4637.
"""

import json
import os
import time
import urllib.error
import urllib.request


PORT = int(os.environ.get("VLLM_WARMUP_PORT", "8000"))
FALLBACK_MODEL = os.environ.get("VLLM_WARMUP_MODEL", "")
PROMPT_TOKENS = int(os.environ.get("VLLM_WARMUP_PROMPT_TOKENS", "0"))
PARENT_PID = int(os.environ.get("VLLM_WARMUP_PARENT_PID", "0"))
READY_TIMEOUT = int(os.environ.get("VLLM_WARMUP_READY_TIMEOUT", "1200"))
BASE_URL = f"http://127.0.0.1:{PORT}"


def log(message):
    print(f"[vllm-warmup] {message}", flush=True)


def parent_is_alive():
    if PARENT_PID <= 0:
        return True
    try:
        os.kill(PARENT_PID, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_ready():
    deadline = time.monotonic() + READY_TIMEOUT
    models_url = f"{BASE_URL}/v1/models"
    while time.monotonic() < deadline and parent_is_alive():
        try:
            with urllib.request.urlopen(models_url, timeout=3) as response:
                payload = json.loads(response.read())
            models = payload.get("data", [])
            if models:
                return models[0].get("id") or FALLBACK_MODEL
            return FALLBACK_MODEL
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(3)
    return None


def prompt_of(approx_tokens):
    # Deliberately repetitive structure but changing values: inexpensive to
    # generate, tokenizes predictably enough, and cannot be optimized to an
    # empty prompt by a chat template.
    line_count = max(1, approx_tokens // 14)
    return "\n".join(
        f"Line {i}: id={i} value={(i * 7919) % 100003} tag={chr(65 + i % 6)}."
        for i in range(line_count)
    )


def request(model, prompt, max_tokens, timeout):
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read())
    return result.get("usage", {}).get("prompt_tokens", 0)


def main():
    model = wait_ready()
    if not model:
        log("server did not become ready; skipping")
        return

    started = time.monotonic()
    try:
        log("warming decode and speculative kernels")
        request(model, "Say ACK.", 12, 600)
    except Exception as exc:
        log(f"tiny request skipped: {str(exc)[:120]}")

    if PROMPT_TOKENS > 0 and parent_is_alive():
        try:
            log(f"warming prefill kernels at approximately {PROMPT_TOKENS} tokens")
            actual = request(
                model,
                prompt_of(PROMPT_TOKENS) + "\nReply: ACK",
                4,
                1800,
            )
            log(f"prefill request used {actual} prompt tokens")
        except Exception as exc:
            log(f"prefill request skipped: {str(exc)[:120]}")

    log(f"complete in {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main()
