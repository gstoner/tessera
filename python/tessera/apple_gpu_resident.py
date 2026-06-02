"""Pre-decode warmup — ResidentWeights cache.

Project-3 follow-on (2026-06-01) to the single-cb decoder roadmap.
Phase 3 in ``docs/audit/backend/apple/APPLE_AUDIT.md``: upload
LLM weights to device once and reuse across decode steps instead of
re-uploading every iteration.

Pattern in current per-step code:

    for step in range(num_steps):
        # Every step uploads ALL weights AND the activation:
        wq_dev = device_tensor(Wq)
        wk_dev = device_tensor(Wk)
        ...
        x_dev = device_tensor(X_step)
        # run decoder layer ...
        # All ts_dev_free'd at end of step.

Real LLM inference: weights are fixed across steps; only the
activation (and KV cache) changes. The ResidentWeights cache makes
that explicit:

    cache = ResidentWeights()
    # Once, before the decode loop:
    cache.weight("Wq", Wq)
    cache.weight("Wk", Wk)
    cache.weight("Wv", Wv)
    cache.weight("Wo", Wo)
    cache.weight("gamma", gamma)
    cache.weight("Theta", Theta)

    for step in range(num_steps):
        x_dev = cache.activation("x", X_step)  # re-uploaded each step
        with batched_session() as s:
            n = rmsnorm_enc(s, x_dev, cache["gamma"], rows=..., cols=D)
            q = bmm_enc(s, n, cache["Wq"], batch=1, M=..., N=D, K=D)
            ...
        # No ts_dev_free at end of step — weights persist.

    cache.free()  # or use as a context manager

The headline win is **N step-wise host→device transfers eliminated**
where N = number of weights. For a 7-weight Llama attention block,
that's ~7 host→device round-trips removed per decode step.
"""

from __future__ import annotations


import numpy as np

from .apple_gpu_batched import DeviceTensor, device_tensor


class ResidentWeights:
    """A keyed cache of device-resident tensors.

    Two access modes:

    * :meth:`weight` / :meth:`__getitem__` — *persistent* tensors.
      Uploaded once on first ``weight()`` call; subsequent
      ``weight(name, arr)`` calls with the SAME ``arr`` are no-ops
      (the cached tensor is returned). Free on :meth:`free` or
      context-manager exit.
    * :meth:`activation` — *re-uploaded* tensors. The cache allocates
      the device buffer once (when shape/dtype first seen for that
      ``name``); each subsequent call uploads new host bytes into
      the SAME device buffer. Eliminates per-step alloc/free churn
      for the activation argument.

    Use as a context manager to guarantee all tracked tensors are
    released::

        with ResidentWeights() as cache:
            cache.weight("Wq", Wq)
            for step in range(N):
                x = cache.activation("x", X_step)
                ...  # decode
        # All weights + activation buffers freed here.
    """

    def __init__(self) -> None:
        self._weights: dict[str, DeviceTensor] = {}
        self._activations: dict[str, DeviceTensor] = {}
        # Track the host-side numpy id we uploaded for each weight so
        # the second weight(name, arr) with the same arr is a true
        # no-op AND a different arr triggers a clear error (rather
        # than silently keeping the old GPU bytes).
        self._weight_host_ids: dict[str, int] = {}
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ResidentWeights":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.free()

    def free(self) -> None:
        """Release every tracked device tensor. Idempotent — calling
        again is a no-op."""
        for t in self._weights.values():
            t.free()
        for t in self._activations.values():
            t.free()
        self._weights.clear()
        self._activations.clear()
        self._weight_host_ids.clear()
        self._closed = True

    def __del__(self) -> None:
        # Best-effort cleanup if the user forgot a free() / with-block.
        if not self._closed:
            try:
                self.free()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Persistent weights                                                 #
    # ------------------------------------------------------------------ #

    def weight(self, name: str, arr: np.ndarray) -> DeviceTensor:
        """Upload ``arr`` to device under ``name`` if not already
        present. Returns the cached :class:`DeviceTensor`. Calling
        again with the SAME numpy array is a no-op (returns the
        cached tensor). Calling with a DIFFERENT array under the
        same name raises — re-keying a weight is the caller's
        responsibility (use :meth:`replace_weight` to be explicit)."""
        if self._closed:
            raise RuntimeError("ResidentWeights already freed")
        if name in self._weights:
            if id(arr) != self._weight_host_ids[name]:
                raise ValueError(
                    f"ResidentWeights: name {name!r} already bound to a "
                    f"different host array (id={self._weight_host_ids[name]} "
                    f"vs {id(arr)}). Use replace_weight() to rebind, or "
                    f"choose a different name.")
            return self._weights[name]
        dev = device_tensor(arr)
        self._weights[name] = dev
        self._weight_host_ids[name] = id(arr)
        return dev

    def replace_weight(self, name: str, arr: np.ndarray) -> DeviceTensor:
        """Replace an existing weight binding with new host bytes.
        Frees the old device tensor and uploads the new one. Use
        this when weights legitimately change between decode loops
        (e.g., a new adapter swapped in)."""
        if self._closed:
            raise RuntimeError("ResidentWeights already freed")
        if name in self._weights:
            self._weights[name].free()
            del self._weights[name]
            del self._weight_host_ids[name]
        return self.weight(name, arr)

    def __getitem__(self, name: str) -> DeviceTensor:
        """Return the cached weight tensor for ``name``. Raises
        :class:`KeyError` if no weight is bound. Activations are
        retrieved via :meth:`activation` instead — different lifetime
        contract."""
        if self._closed:
            raise RuntimeError("ResidentWeights already freed")
        if name not in self._weights:
            raise KeyError(
                f"no weight named {name!r}; bound: "
                f"{sorted(self._weights)}")
        return self._weights[name]

    def __contains__(self, name: str) -> bool:
        return name in self._weights

    def weight_names(self) -> tuple[str, ...]:
        return tuple(self._weights.keys())

    # ------------------------------------------------------------------ #
    # Activations                                                        #
    # ------------------------------------------------------------------ #

    def activation(self, name: str, arr: np.ndarray) -> DeviceTensor:
        """Upload ``arr`` into the activation buffer for ``name``.
        On first call, allocates a device buffer of the right size.
        On subsequent calls with the SAME shape/dtype, re-uses the
        device buffer (just uploads new bytes). On a shape/dtype
        change, raises — activations should be a stable-shape stream
        (e.g., decoder input for each token at a fixed S/D)."""
        if self._closed:
            raise RuntimeError("ResidentWeights already freed")
        nbytes = int(arr.nbytes)
        existing = self._activations.get(name)
        if existing is not None:
            if existing.nbytes != nbytes:
                raise ValueError(
                    f"activation {name!r}: shape/dtype changed "
                    f"({existing.nbytes} bytes → {nbytes} bytes). "
                    f"Activations should be stable-shape; allocate a "
                    f"fresh cache or use a different name.")
            existing.upload(arr)
            return existing
        dev = device_tensor(arr)
        self._activations[name] = dev
        return dev

    def activation_names(self) -> tuple[str, ...]:
        return tuple(self._activations.keys())

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def total_resident_bytes(self) -> int:
        """Sum of bytes across all tracked device tensors. Useful
        for back-of-envelope memory accounting."""
        return (sum(t.nbytes for t in self._weights.values()) +
                sum(t.nbytes for t in self._activations.values()))


__all__ = [
    "ResidentWeights",
]
