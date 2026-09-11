import numpy as np
import pytest
from benchmarks.record_dtype_arithmetic import check, emit, oracle, samples


@pytest.mark.parametrize('dtype', ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_integer_oracle_wraps_and_probe_has_no_poison_flags(dtype):
    a,b = samples(dtype, 1)
    bits = a.dtype.itemsize * 8
    for op in ('add','sub','mul'):
        expected = oracle(a,b,op)
        assert check(expected,expected) == 0
        unsigned = expected.view(f'uint{bits}')
        exact = {'add':int(a[1])+int(b[1]), 'sub':int(a[1])-int(b[1]), 'mul':int(a[1])*int(b[1])}[op]
        assert int(unsigned[1]) == exact % (1 << bits)
    assert 'nsw' not in emit(dtype) and 'nuw' not in emit(dtype)


def test_probe_comparison_preserves_signed_zero_and_nan_semantics():
    x = np.array([0., -0., np.nan, np.inf], np.float32)
    assert check(x,x) == 0
    assert check(x, np.array([-0.,-0.,np.nan,np.inf],np.float32)) == 1
    assert check(x, np.array([0.,-0.,1.,np.inf],np.float32)) == 1


def test_fp8_probe_uses_bytes_and_explicit_rounding():
    source = emit('fp8_e4m3', 2)
    assert 'llvm.load %ap : !llvm.ptr<1> -> vector<2xi8>' in source
    assert 'arith.extf %xlow : vector<2xf8E4M3FN> to vector<2xf32>' in source
    assert 'arith.truncf %addv : vector<2xf32> to vector<2xf8E4M3FN>' in source
    assert 'arith.divf' in source


def test_division_oracle_truncates_signed_integers_without_float_rounding():
    a = np.array([-7, 7, -(2**63)+1], np.int64)
    b = np.array([3, -3, 3], np.int64)
    np.testing.assert_array_equal(oracle(a,b,'div'), [-2,-2,-3074457345618258602])
    assert 'arith.divsi' in emit('int64')
    assert 'arith.divui' in emit('uint64')


@pytest.mark.parametrize('dtype', ['fp8_e4m3', 'fp8_e5m2'])
def test_fp8_samples_cover_every_pair_of_storage_encodings(dtype):
    a, b = samples(dtype, 1)
    pairs = a.view(np.uint8).astype(np.uint32) * 256 + b.view(np.uint8)
    assert len(np.unique(pairs)) == 65536


def test_bool_probe_names_logic_and_normalizes_storage():
    a, b = samples('bool', 1)
    np.testing.assert_array_equal(oracle(a,b,'sub'), a != b)
    source = emit('bool',2)
    assert 'arith.cmpi ne' in source and 'arith.cmpi eq' in source
    assert 'arith.extui' in source


@pytest.mark.parametrize('dtype', ['complex64','complex128'])
def test_complex_probe_keeps_imaginary_components(dtype):
    a,b = samples(dtype,2)
    assert np.any(oracle(a,b,'mul').imag != 0)
    assert check(a, a.conjugate()) > 0
    assert '%l1mul1' in emit(dtype,2)
