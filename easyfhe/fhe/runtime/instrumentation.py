import atexit
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict

import easyfhe as torch


def _sync_device():
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "synchronize"):
        torch.cpu.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class OpStats:
    count: int = 0
    total_time: float = 0.0

    @property
    def avg_time(self):
        return self.total_time / self.count if self.count else 0.0


class NullInstrumentation:
    def run(self, op_name, impl, *args, **kwargs):
        return impl(*args, **kwargs)


class OpInstrumentation:
    def __init__(
        self,
        *,
        count=False,
        time_ops=False,
        sync=False,
        print_at_exit=False,
        include=None,
    ):
        self.count_ops = count
        self.time_ops = time_ops
        self.sync = sync
        self.include = None if include is None else set(include)
        self.records: Dict[str, OpStats] = {}
        self._atexit_registered = False
        if print_at_exit:
            self.register_atexit()

    def run(self, op_name, impl, *args, **kwargs):
        if self.include is not None and op_name not in self.include:
            return impl(*args, **kwargs)
        if self.sync:
            _sync_device()
        start = time.perf_counter() if self.time_ops else None

        result = impl(*args, **kwargs)

        if self.sync:
            _sync_device()
        elapsed = time.perf_counter() - start if start is not None else 0.0

        if self.count_ops or self.time_ops:
            record = self.records.setdefault(op_name, OpStats())
            record.count += 1
            record.total_time += elapsed

        return result

    def summary(self, sort_by="total_time", limit=None):
        rows = []
        for op_name, record in self.records.items():
            rows.append((op_name, record.count, record.total_time, record.avg_time))

        if sort_by == "count":
            rows.sort(key=lambda row: row[1], reverse=True)
        elif sort_by == "name":
            rows.sort(key=lambda row: row[0])
        else:
            rows.sort(key=lambda row: row[2], reverse=True)

        if limit is not None:
            rows = rows[:limit]
        return rows

    def print_summary(self, sort_by="total_time", limit=None):
        rows = self.summary(sort_by=sort_by, limit=limit)
        if not rows:
            return
        total_time = sum(row[2] for row in rows)
        print("\nFHE Operation Profile:")
        print(f"{'op':32s} {'count':>8s} {'total(s)':>12s} {'avg(ms)':>12s}")
        for op_name, count, total, avg in rows:
            print(f"{op_name:32s} {count:8d} {total:12.6f} {avg * 1000:12.3f}")
        print(f"Total profiled time: {total_time:.6f}s")

    def register_atexit(self):
        if not self._atexit_registered:
            atexit.register(self.print_summary)
            self._atexit_registered = True
        return self


def instrumentation_from_options(options):
    if options is None:
        return NullInstrumentation()
    enabled = (
        getattr(options, "count_ops", False)
        or getattr(options, "time_ops", False)
        or getattr(options, "auto_sync", False)
    )
    if not enabled:
        return NullInstrumentation()
    return OpInstrumentation(
        count=getattr(options, "count_ops", False) or getattr(options, "time_ops", False),
        time_ops=getattr(options, "time_ops", False),
        sync=getattr(options, "auto_sync", False),
        print_at_exit=getattr(options, "count_ops", False) or getattr(options, "time_ops", False),
    )


@contextmanager
def profile(ctx, *, sync=False, include=None):
    previous = getattr(ctx, "instrumentation", NullInstrumentation())
    profiler = OpInstrumentation(count=True, time_ops=True, sync=sync, include=include)
    ctx.instrumentation = profiler
    try:
        yield profiler
    finally:
        ctx.instrumentation = previous
