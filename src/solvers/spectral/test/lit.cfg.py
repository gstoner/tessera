
import os

config.name = "TesseraSpectral"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

tool_dirs = [os.path.join(os.environ.get('BUILD_DIR',''), 'bin')]
config.substitutions.append(('%PATH%', os.pathsep.join(tool_dirs)))
