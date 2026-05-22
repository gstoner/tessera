
import os, lit
config.name = "TESSERA-CPX-v1.1"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.substitutions.append(('%cpx_opt', 'tessera-cpx-opt'))
