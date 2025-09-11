
import os
config.name = "TESSERA-CPX"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'RUN')
config.substitutions.append(('%cpx_opt', 'tessera-cpx-opt'))
