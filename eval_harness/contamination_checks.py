from typing import Iterable, Set


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def is_contaminated(train_sources, benchmark_sources, threshold: float = 0.2):
    """
    Infer contamination risk from benchmark-source protection coverage.

    train_sources is expected to be the configured exclusion list
    (sources blocked from training). Higher overlap with benchmark_sources
    means safer protection, so contamination risk is the inverse.
    """
    protected = set(train_sources or [])
    benchmarks = set(benchmark_sources or [])
    if not benchmarks:
        return False, 1.0
    coverage = len(protected & benchmarks) / len(benchmarks)
    contaminated = coverage < threshold
    return contaminated, coverage
