"""Optional NVTX3 ranges for synchronous benchmark calls, with run/artifact IDs."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile


class Ranges:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tessera-nvtx-')
        root = Path(self.temp.name)
        source = root / 'ranges.c'
        source.write_text('#include <nvtx3/nvToolsExt.h>\nvoid push(const char *s){nvtxRangePushA(s);}\nvoid pop(void){nvtxRangePop();}\n')
        include = Path(os.environ.get('CUDA_HOME','/usr/local/cuda-13.3')) / 'include'
        try:
            subprocess.run(['cc','-shared','-fPIC','-I'+str(include),str(source),'-ldl','-o',str(root/'ranges.so')],check=True,capture_output=True)
            self.lib = ctypes.CDLL(str(root/'ranges.so'))
            self.lib.push.argtypes, self.lib.push.restype = [ctypes.c_char_p], None
            self.lib.pop.argtypes, self.lib.pop.restype = [], None
        except BaseException:
            self.temp.cleanup()
            raise

    @contextmanager
    def mark(self, label):
        self.lib.push(label.encode())
        try:
            yield
        finally:
            self.lib.pop()

    def close(self):
        self.temp.cleanup()
