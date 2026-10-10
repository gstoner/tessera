"""Float64 scale-adjoint oracle. Diagnostic only; never a production lowering."""
import numpy as np


def scale_adjoint(a, b, sa, sb, dy, *, scale_k, scale_n,
                  batching=None, transpose_b=False):
    """Return original-storage scale cotangents, reducing shared batch axes.

    Each K group retains its own unscaled partial product. No ordinary GEMM
    transpose can replace these reductions when K/N groups differ.
    """
    a, b, sa, sb, dy = (np.asarray(x).astype(np.float64)
                        for x in (a, b, sa, sb, dy))
    if transpose_b:
        b = b.swapaxes(-1, -2)
    lhs_mapped = batching in {"shared_rhs_rows", "independent_rhs"}
    rhs_mapped = batching in {"shared_lhs", "independent_rhs"}
    prefix = dy.shape[:-2]
    dsa, dsb = np.zeros_like(sa), np.zeros_like(sb)
    n, k = b.shape[-1], a.shape[-1]
    columns = np.arange(n) // scale_n
    for batch in np.ndindex(prefix):
        aa, ss = (a[batch], sa[batch]) if lhs_mapped else (a, sa)
        bb, tt = (b[batch], sb[batch]) if rhs_mapped else (b, sb)
        da = dsa[batch] if lhs_mapped else dsa
        db = dsb[batch] if rhs_mapped else dsb
        for group in range((k + scale_k - 1) // scale_k):
            lo, hi = group * scale_k, min(k, (group + 1) * scale_k)
            partial = aa[:, lo:hi] @ bb[lo:hi, :]
            weighted = dy[batch] * partial
            da[:, group] += np.sum(weighted * tt[group, columns], axis=1)
            per_column = np.sum(weighted * ss[:, group, None], axis=0)
            np.add.at(db[group], columns, per_column)
    return dsa, dsb
