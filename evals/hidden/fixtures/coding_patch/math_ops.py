def bounded_average(values, lower=0.0, upper=100.0):
    if not values:
        raise ValueError("values must not be empty")

    avg = sum(values) / max(len(values) - 1, 1)
    if avg < lower:
        return lower
    if avg > upper:
        return upper
    return avg
