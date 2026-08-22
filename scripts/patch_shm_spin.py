"""Shorten vLLM's shm-broadcast reader busy-loop default (busy_loop_s 1 s -> 2 ms).

On TP>=2 serving, every process that reads the scheduler's shared-memory
MessageQueue (EngineCore workers, API processes) busy-loops via sched_yield for
up to ``busy_loop_s`` -- default 1 second -- after its last read, before
falling back to the event-driven zmq poller wait:

    class SpinCondition:
        def __init__(..., busy_loop_s: float = 1):
            ...
        def wait(...):
            if current_time <= self.last_read + self.busy_loop_s:
                sched_yield()
            else:
                ... poller.poll(...) ...

Decode-step IPC gaps are milliseconds, so under steady load the idle path
never engages and N-1 cores burn at boost clocks between steps. On Strix Halo
(gfx1151) the CPU and GPU share one SoC power/thermal budget, so those pinned
cores directly steal frequency and thermal headroom from inference.

Fix: lower the reader-side default from 1 s to 2 ms. Reads within 2 ms of the
previous one still take the sched_yield fast path; anything slower parks in
the zmq poller instead of spinning a core. The writer side hardcodes
``busy_loop_s = 0`` and is unaffected. This mirrors the runtime hotfix shipped
by the MiaAI-Lab DSpark two-node recipe (their issue #79).

Upstream status: vllm-project/vllm v0.27.1 still defaults to 1 second.
Delete this script once upstream shortens the default or makes the
busy-loop window configurable.
"""

import sys
from pathlib import Path

MARKER = "PATCHED: gfx1151 shm reader spin"

# SpinCondition's reader constructor default, verbatim in v0.27.1 and in the
# June-2026 TheRock snapshots (8-space indent inside __init__).
_ANCHOR = "        busy_loop_s: float = 1,"
_REPLACEMENT = "        busy_loop_s: float = 0.002,  # " + MARKER


def patch_file(path) -> bool:
    """Patch shm_broadcast.py at `path`. Returns True when a change was made.

    Idempotent: returns False without touching the file when MARKER is
    already present. Fail-closed: raises SystemExit when the anchor is
    missing, because a silent no-op would ship images with the spin intact.
    """
    path = Path(path)
    txt = path.read_text()

    if MARKER in txt:
        print(" -> shm_broadcast.py already patched (idempotent no-op)")
        return False

    if _ANCHOR not in txt:
        print(
            " -> SpinCondition's 'busy_loop_s: float = 1' default not found; "
            "the spin fix was NOT applied. If upstream shortened or made the "
            "busy-loop configurable this script is obsolete -- verify and drop "
            "it; otherwise update the anchor for the new shape.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    path.write_text(txt.replace(_ANCHOR, _REPLACEMENT, 1))
    print(
        " -> Patched SpinCondition busy_loop_s 1 -> 0.002 "
        "(readers park in the zmq poller after 2 ms instead of spinning 1 s)"
    )
    return True


def main() -> int:
    # In both image Dockerfiles this script runs from /opt/vllm, so the target
    # is the source tree rooted beside it. An explicit argv override exists
    # for offline testing against a fixture checkout.
    default_target = (
        Path(__file__).resolve().parent
        / "vllm/distributed/device_communicators/shm_broadcast.py"
    )
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target
    if not target.is_file():
        print(f" -> {target} does not exist", file=sys.stderr)
        return 1
    patch_file(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
