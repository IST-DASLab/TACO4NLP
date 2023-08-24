__all__ = [
    "round_to_nearest_multiple"
]


def round_to_nearest_multiple(x: int, num: int):
    return round(x / num) * num