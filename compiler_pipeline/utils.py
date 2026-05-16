import random

def oxford_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def pick_random_contiguous_period_window(periods: list[str]) -> list[str]:
    """Pick a random contiguous period window from sorted periods."""
    min_window = 2
    ordered = sorted(periods)
    if len(ordered) < min_window:
        raise Exception(
            f"Need at least {min_window} periods to sample a time-aggregation window."
        )
    window_len = random.randint(min_window, len(ordered))
    start = random.randint(0, len(ordered) - window_len)
    return list(ordered[start : start + window_len])