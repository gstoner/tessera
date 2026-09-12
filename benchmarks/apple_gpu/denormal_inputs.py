"""Bit-pattern boundary corpus shared by host and owning-Metal checks."""
import numpy as np


def operands(random_count=131072):
    edges = np.array([0, 1, 2, 3, 4, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff,
                      0x800000, 0x800001, 0xffffff, 0x1000000, 0x3effffff,
                      0x3f000000, 0x3f000001, 0x3f7fffff, 0x3f800000, 0x3f800001,
                      0x40000000, 0x7f7fffff, 0x7f800000, 0x7f800001, 0x7fc00000], np.uint32)
    edges = np.concatenate([edges, edges | np.uint32(0x80000000)])
    a, b = np.repeat(edges, len(edges)), np.tile(edges, len(edges))
    rng = np.random.default_rng(746)
    a = np.concatenate([a, rng.integers(0, 2**32, random_count, dtype=np.uint32)])
    b = np.concatenate([b, rng.integers(0, 2**32, random_count, dtype=np.uint32)])
    return a.view(np.float32), b.view(np.float32)


def flush(values):
    bits = values.view(np.uint32).copy()
    tiny = (bits & np.uint32(0x7f800000)) == 0
    bits[tiny] &= np.uint32(0x80000000)
    return bits.view(np.float32)
