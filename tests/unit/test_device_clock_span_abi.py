"""The device-clock span ABI (sync WSL-TIMING-ADMISSION-2026-09-26).

`--tessera-device-clock-span` appends one span-buffer argument to a native
kernel; the storage package records it as ABI kind ``clock_span``. That image
is the instrumented member of a calibration pair, launched raw by a recorder.
These tests pin that the kind is admitted only as the single trailing
argument and that it can never reach a production tensor binding.
"""

from __future__ import annotations

import inspect

import pytest

from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage
from tessera.compiler.native_gpu_tensor import TensorSpec, validate_tensor_signature


def _package(abi: tuple[str, ...]) -> NativeGPUStoragePackage:
    draft = NativeGPUStoragePackage('rocm', 'gfx1151', 'k', 'sizer', abi, 'arena', b'img',
                                    b'lib', 'c', 'l', '')
    return NativeGPUStoragePackage(**{**draft.__dict__, 'binding_digest': draft._digest()})


def test_clock_span_is_admitted_only_as_the_single_trailing_argument() -> None:
    _package(('pointer', 'index', 'clock_span')).validate()
    for bad in (('clock_span', 'pointer'), ('pointer', 'clock_span', 'index'),
                ('pointer', 'clock_span', 'clock_span')):
        with pytest.raises(ValueError, match='single trailing argument'):
            _package(bad).validate()
    with pytest.raises(ValueError, match='unsupported native storage ABI'):
        _package(('pointer', 'span')).validate()


def test_an_instrumented_abi_cannot_be_tensor_bound() -> None:
    """The tensor contract describes the clean kernel; the extra span
    argument makes the instrumented ABI disagree, so binding refuses it."""
    spec = TensorSpec.__new__(TensorSpec)
    object.__setattr__(spec, 'name', 'x')
    signature = inspect.Signature([inspect.Parameter('x', inspect.Parameter.POSITIONAL_ONLY)])
    with pytest.raises(ValueError, match='argument count'):
        validate_tensor_signature(('pointer', 'clock_span'), signature, [spec], (1, 1, 1), (1, 1, 1))
