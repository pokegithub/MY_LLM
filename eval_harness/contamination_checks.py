from typing import Iterable, Set


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def benchmark_exclusion_coverage(
    excluded_sources,
    benchmark_sources,
) -> float:
    """Return how much of the benchmark source list is explicitly excluded."""
    protected = set(excluded_sources or [])
    benchmarks = set(benchmark_sources or [])
    if not benchmarks:
        return 1.0
    return len(protected & benchmarks) / len(benchmarks)


def has_low_benchmark_exclusion_coverage(
    excluded_sources,
    benchmark_sources,
    threshold: float = 0.2,
):
    """
    Flag weak source-level benchmark exclusion coverage.

    This is not content contamination detection. It does not compare examples,
    hashes, text spans, or generated outputs. It only checks whether configured
    excluded source IDs cover the benchmark source IDs.
    """
    coverage = benchmark_exclusion_coverage(excluded_sources, benchmark_sources)
    return coverage < threshold, coverage


def is_contaminated(train_sources, benchmark_sources, threshold: float = 0.2):
    """Backward-compatible alias for source-exclusion coverage checks."""
    return has_low_benchmark_exclusion_coverage(
        train_sources,
        benchmark_sources,
        threshold=threshold,
    )
