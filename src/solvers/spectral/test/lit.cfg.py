import os

import lit.formats

config.name = "TesseraSpectral"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

# Make ts-spectral-opt (from the build dir) and FileCheck (from LLVM) findable
# by the RUN lines.  BUILD_DIR points at the CMake binary dir; the inherited
# PATH is preserved so a caller can also just export it directly.
tool_dirs = []
build_dir = os.environ.get('BUILD_DIR', '')
if build_dir:
    tool_dirs += [build_dir, os.path.join(build_dir, 'bin')]
config.environment['PATH'] = os.pathsep.join(
    tool_dirs + [os.environ.get('PATH', '')])
