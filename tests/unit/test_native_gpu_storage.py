"""Immutable native companion/image binding and raw ABI validation."""
from dataclasses import asdict, replace
import pytest
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage, BoundNativeGPUStorage


def package():
    p = NativeGPUStoragePackage('nvidia', 'sm_120', 'entry', 'size', ('pointer', 'index'),
                                'module {}', b'image', b'host', 'c' * 64, 'd' * 64, '')
    return NativeGPUStoragePackage(**{**asdict(p), 'binding_digest': p._digest()})


@pytest.mark.parametrize('field,value', [('image', b'other'), ('host_library', b'other'),
    ('abi', ('index', 'pointer')), ('entry', 'other'), ('sizer', 'other'),
    ('chip', 'sm_90'), ('arena_ir', 'changed')])
def test_modified_pair_is_rejected(field, value):
    with pytest.raises(ValueError, match='binding'):
        replace(package(), **{field: value}).validate()


def test_serialized_pair_requires_pinned_identity():
    original = package()
    assert NativeGPUStoragePackage.from_json(original.to_json(), expected_digest=original.binding_digest) == original
    with pytest.raises(ValueError, match='pinned identity'):
        NativeGPUStoragePackage.from_json(original.to_json(), expected_digest='0' * 64)


@pytest.mark.parametrize('arguments', [(1,), (True, 1), (0, 1), (1, -1), (1, 1 << 63), (1, 1.5)])
def test_raw_abi_rejects_invalid_arguments(arguments):
    import ctypes as ct
    bound = object.__new__(BoundNativeGPUStorage)
    bound.package = package()
    bound._types = (ct.c_void_p, ct.c_int64)
    with pytest.raises(ValueError):
        bound._arguments(arguments)


def test_native_image_escapes_preserve_ptx_and_binary_bytes():
    from tessera.compiler.native_gpu_storage import _decode_image
    assert _decode_image(r".version 9.0\n\t\22quoted\22\00\FF\\") == b'.version 9.0\n\t"quoted"\x00\xff\\'
    with pytest.raises(ValueError, match='escape'):
        _decode_image(r"\q")
