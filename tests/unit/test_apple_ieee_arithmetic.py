"""Run exactly the emitted integer algorithm against host IEEE f32 arithmetic."""
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from benchmarks.apple_gpu.denormal_inputs import operands
from benchmarks.record_dtype_arithmetic import check


def test_emitted_integer_mul_div_against_ieee_oracle(tmp_path):
    compiler = shutil.which('clang++') or ('/usr/lib/llvm-23/bin/clang++' if Path('/usr/lib/llvm-23/bin/clang++').is_file() else None)
    if not compiler:
        pytest.skip('host C++ compiler required')
    header = Path('src/transforms/lib/AppleIEEEArithmetic.h').read_text()
    code = header.split('R"MSL(', 1)[1].split(')MSL"', 1)[0]
    source = '''#include <cstdint>
#include <cstring>
using uint = uint32_t; using ulong = uint64_t;
template<class T, class U> T as_type(U x) { T out; static_assert(sizeof(T)==sizeof(U)); std::memcpy(&out,&x,sizeof(out)); return out; }
''' + code + '''
extern "C" void run(const float *a,const float *b,float *m,float *d,float *s,float *t,int n) {
 for (int i=0;i<n;++i) { m[i]=tessera_ieee_mul(a[i],b[i]); d[i]=tessera_ieee_div(a[i],b[i]); s[i]=tessera_ieee_add(a[i],b[i]); t[i]=tessera_ieee_sub(a[i],b[i]); }
}
'''
    cpp, library = tmp_path/'oracle.cpp', tmp_path/'oracle.so'
    cpp.write_text(source)
    subprocess.run([compiler,'-std=c++17','-O2','-fPIC','-shared',str(cpp),'-o',str(library)],check=True,capture_output=True)
    native = ct.CDLL(str(library)).run
    native.argtypes = [ct.c_void_p]*6 + [ct.c_int]
    a,b = operands()
    m,d,s,t = [np.empty_like(a) for _ in range(4)]
    native(a.ctypes.data,b.ctypes.data,m.ctypes.data,d.ctypes.data,s.ctypes.data,t.ctypes.data,len(a))
    with np.errstate(all='ignore'):
        assert check(m,a*b) == 0
        assert check(d,a/b) == 0
        assert check(s,a+b) == 0
        assert check(t,a-b) == 0
