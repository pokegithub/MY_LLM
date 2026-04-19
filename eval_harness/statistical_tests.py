import math


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bootstrap_ci(xs, alpha: float = 0.05):
    # Lightweight approximation to avoid heavy dependencies.
    m = mean(xs)
    s = std(xs)
    n = max(len(xs), 1)
    z = 1.96 if alpha == 0.05 else 1.64
    half = z * s / math.sqrt(n)
    return m - half, m + half
