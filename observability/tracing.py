import time
from contextlib import contextmanager


@contextmanager
def trace_span(name: str, sink=None):
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        msg = {"span": name, "elapsed_sec": round(elapsed, 6)}
        if sink is not None:
            sink(msg)
        else:
            print(f"[trace] {name}: {elapsed:.4f}s")
