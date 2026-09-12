"""Immutable native companion/image binding and raw ABI validation."""
from dataclasses import asdict, replace
import pytest
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage, BoundNativeGPUStorage


def test_gpu_serialization_toolkit_is_explicit_and_cannot_inject_passes(tmp_path):
    from tessera.compiler.native_gpu_storage import _binary_pass
    toolkit = tmp_path / 'cuda-13.4'
    toolkit.mkdir()
    alias = tmp_path / 'cuda'
    alias.symlink_to(toolkit, target_is_directory=True)
    assert _binary_pass(alias) == f'gpu-module-to-binary{{toolkit={toolkit}}}'
    assert _binary_pass(None) == 'gpu-module-to-binary'
    invalid = tmp_path / 'bad},canonicalize'
    invalid.mkdir()
    with pytest.raises(ValueError, match='plain absolute path'):
        _binary_pass(invalid)


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


def test_gpu_storage_refuses_unconsumed_denormal_policy(tmp_path):
    from tessera.compiler.native_gpu_storage import build_native_gpu_storage
    for backend, chip in [('nvidia','sm_120'), ('rocm','gfx1151')]:
        with pytest.raises(ValueError, match='Apple arena consumer'):
            build_native_gpu_storage('module attributes {tessera.denormal_mode = "gradual"} {}',
                                     compiler=tmp_path/'missing', llvm_bin=tmp_path, backend=backend, chip=chip)
