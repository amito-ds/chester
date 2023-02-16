import numpy as np


def min_fix(l):
    if not l:
        return None
    return min(l)


def max_fix(l):
    if not l:
        return None
    return max(l)


def mean_fix(l):
    if not l:
        return None
    return np.average(l)


def median_fix(l):
    if not l:
        return None
    return np.median(l)
