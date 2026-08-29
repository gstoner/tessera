import lit.formats
import os

config.name = "TesseraEBM"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

# Put the freshly built driver on the suite's PATH. `%PATH%` alone was a
# substitution the RUN lines never used, so the suite silently depended on the
# caller having exported the tool directory. BUILD_DIR is supplied by the
# check-* target; the bare build dir is where the standalone driver links.
tool_dirs = []
build_dir = os.environ.get('BUILD_DIR', '')
if build_dir:
    tool_dirs += [build_dir, os.path.join(build_dir, 'bin')]
config.substitutions.append(('%PATH%', os.pathsep.join(tool_dirs)))
config.environment['PATH'] = os.pathsep.join(
    tool_dirs + [os.environ.get('PATH', '')])
