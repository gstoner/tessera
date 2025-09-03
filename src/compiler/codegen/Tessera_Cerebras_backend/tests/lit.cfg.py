import os
import lit.formats

config.name = "tessera-cerebras"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.test']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'RUN')

# Assume tool is on PATH after build
config.substitutions.append(('%tessera_cerebras_opt', 'tessera-cerebras-opt'))
