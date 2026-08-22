import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_shm_spin  # noqa: E402


# Fixture mirrors the real SpinCondition constructor shape around the anchor
# (v0.27.1 and TheRock snapshots), with enough surrounding context to catch
# accidental whole-file rewrites.
SPIN_CONDITION_SOURCE = '''\
class SpinCondition:
    """Spin quickly while reads are frequent, then idle on a zmq poller."""

    def __init__(
        self,
        is_reader: bool,
        context: zmq.Context,
        notify_address: str,
        busy_loop_s: float = 1,
    ):
        self.is_reader = is_reader

        if is_reader:
            # Time of last shm buffer read
            self.last_read = time.monotonic()

            # Time to keep busy-looping on the shm buffer before going idle
            self.busy_loop_s = busy_loop_s
            self.poller = zmq.Poller()
        else:
            self.last_read = 0
            self.busy_loop_s = 0

    def wait(self, timeout_ms=None):
        current_time = time.monotonic()
        if current_time <= self.last_read + self.busy_loop_s:
            sched_yield()
        else:
            events = dict(self.poller.poll(timeout=timeout_ms))
'''


class ShmSpinPatchTests(unittest.TestCase):
    def test_patch_shortens_reader_default_and_keeps_writer_side(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shm_broadcast.py"
            path.write_text(SPIN_CONDITION_SOURCE)

            self.assertTrue(patch_shm_spin.patch_file(path))
            patched = path.read_text()

            self.assertIn(patch_shm_spin.MARKER, patched)
            self.assertIn("busy_loop_s: float = 0.002", patched)
            # Only the reader-side default changes.
            self.assertNotIn("busy_loop_s: float = 1,", patched)
            self.assertIn("self.busy_loop_s = 0\n", patched)
            # Surrounding code is untouched.
            self.assertIn("if current_time <= self.last_read + self.busy_loop_s:", patched)

    def test_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shm_broadcast.py"
            path.write_text(SPIN_CONDITION_SOURCE)

            self.assertTrue(patch_shm_spin.patch_file(path))
            once = path.read_text()

            self.assertFalse(patch_shm_spin.patch_file(path))
            self.assertEqual(path.read_text(), once)

    def test_patch_fails_closed_on_anchor_drift(self):
        drifted = SPIN_CONDITION_SOURCE.replace(
            "busy_loop_s: float = 1,", "busy_loop_s: float = 0.5,"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shm_broadcast.py"
            path.write_text(drifted)

            with self.assertRaises(SystemExit) as ctx:
                patch_shm_spin.patch_file(path)
            self.assertEqual(ctx.exception.code, 1)
            # Nothing was written.
            self.assertEqual(path.read_text(), drifted)


if __name__ == "__main__":
    unittest.main()
