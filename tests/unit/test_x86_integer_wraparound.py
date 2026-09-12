"""The scalar oracle must match non-saturating VNNI without C++ overflow UB."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_reference_vnni_wraparound_is_defined_under_ubsan(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++ compiler unavailable')
    source = tmp_path / 'probe.cpp'
    source.write_text(r'''
#include <cstdint>
#include <cstring>
#include <vector>
extern "C" void tessera_x86_reference_gemm_u8s8_s32(const uint8_t*, const int8_t*, int32_t*, int, int, int, int);
int main() {
  const int k = 70001;
  std::vector<uint8_t> a(k, 255);
  for (int value : {-128, 127}) {
    std::vector<int8_t> b(k, value);
    for (int beta : {0, 2, -2}) {
      int32_t out = INT32_MAX;
      int64_t exact = int64_t(255) * value * k + int64_t(INT32_MAX) * beta;
      uint32_t expected = uint32_t(exact), actual;
      tessera_x86_reference_gemm_u8s8_s32(a.data(), b.data(), &out, 1, 1, k, beta);
      std::memcpy(&actual, &out, sizeof(actual));
      if (actual != expected) return 1;
    }
  }
}
''')
    kernel = ROOT / 'src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_vnni_gemm_int8.cpp'
    binary = tmp_path / 'probe'
    subprocess.run([compiler, '-O2', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
                    '-I'+str(ROOT/'src/compiler/layout_algebra/include'), str(source), str(kernel), '-o', str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)
