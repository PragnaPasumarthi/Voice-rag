"""
Latency Analytics Tracker
Measures P50/P70/P100 latency across the pipeline.
"""
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimerContext:
    """Context manager for timing operations."""
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_ms: float = 0.0
    _tracker: Optional["LatencyTracker"] = field(default=None, repr=False)
    _label: str = ""

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        if self._tracker:
            self._tracker._record(self._label, self.elapsed_ms)
        return self.elapsed_ms

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()


class LatencyTracker:
    """
    Thread-safe latency tracker for pipeline benchmarks.
    
    Tracks:
    - Per-stage latencies (STT, chunking, retrieval, LLM, guardrails)
    - End-to-end latencies
    - P50, P70, P90, P100 percentiles
    - Mean, min, max, std dev
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: Dict[str, List[float]] = {}
        self._e2e_latencies: List[float] = []

    def start(self, label: str = "e2e") -> TimerContext:
        """Start a new timer context."""
        ctx = TimerContext(_tracker=self, _label=label)
        ctx.start()
        return ctx

    def _record(self, label: str, elapsed_ms: float):
        with self._lock:
            if label not in self._records:
                self._records[label] = []
            self._records[label].append(elapsed_ms)
            if label == "e2e":
                self._e2e_latencies.append(elapsed_ms)

    def record(self, label: str, elapsed_ms: float):
        """Manually record a latency value."""
        self._record(label, elapsed_ms)

    def get_stats(self, label: str = "e2e") -> Dict[str, Any]:
        """Get latency statistics for a given label."""
        with self._lock:
            if label == "e2e":
                data = self._e2e_latencies
            else:
                data = self._records.get(label, [])

            if not data:
                return {"label": label, "count": 0}

            arr = np.array(data)
            return {
                "label": label,
                "count": len(data),
                "p50": float(np.percentile(arr, 50)),
                "p70": float(np.percentile(arr, 70)),
                "p90": float(np.percentile(arr, 90)),
                "p100": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
            }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tracked labels."""
        with self._lock:
            labels = list(self._records.keys()) + ["e2e"]

        stats = {}
        for label in set(labels):
            stats[label] = self.get_stats(label)
        return stats

    def get_percentile_report(self) -> str:
        """Generate a human-readable latency report."""
        stats = self.get_stats("e2e")
        if stats["count"] == 0:
            return "No latency data recorded yet."

        report = [
            "=" * 50,
            "  LATENCY REPORT",
            "=" * 50,
            f"  Samples:  {stats['count']}",
            f"  P50:      {stats['p50']:.1f} ms",
            f"  P70:      {stats['p70']:.1f} ms",
            f"  P90:      {stats['p90']:.1f} ms",
            f"  P100:     {stats['p100']:.1f} ms",
            f"  Mean:     {stats['mean']:.1f} ms",
            f"  Min:      {stats['min']:.1f} ms",
            f"  Max:      {stats['max']:.1f} ms",
            f"  Std Dev:  {stats['std']:.1f} ms",
            "=" * 50,
        ]

        # Add per-stage breakdown
        for label in sorted(self._records.keys()):
            if label != "e2e":
                stage_stats = self.get_stats(label)
                if stage_stats["count"] > 0:
                    report.append(
                        f"  [{label}] P50={stage_stats['p50']:.1f}ms  "
                        f"Mean={stage_stats['mean']:.1f}ms"
                    )

        return "\n".join(report)

    def clear(self):
        """Reset all recorded data."""
        with self._lock:
            self._records.clear()
            self._e2e_latencies.clear()

    def benchmark(
        self,
        func,
        n_queries: int = 50,
        warmup: int = 5,
        label: str = "e2e",
    ) -> Dict[str, Any]:
        """Run a function n_queries times and collect latency stats."""
        # Warmup
        for _ in range(warmup):
            try:
                func()
            except Exception:
                pass

        # Benchmark
        for _ in range(n_queries):
            timer = self.start(label)
            try:
                func()
            finally:
                timer.stop()

        return self.get_stats(label)
