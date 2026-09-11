"""Exclusive per-output locks for multi-worker inference."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def output_lock(filename):
    """Yield whether this worker owns the output; always release its lock.

    Existing outputs are skipped. A lock left by a killed process must only be
    removed after confirming its job is no longer running.
    """
    output = Path(filename)
    lock = Path(str(output) + ".lock")
    if output.exists():
        yield False
        return
    try:
        handle = lock.open("x")
    except FileExistsError:
        yield False
        return
    handle.close()
    try:
        yield not output.exists()
    finally:
        lock.unlink(missing_ok=True)
