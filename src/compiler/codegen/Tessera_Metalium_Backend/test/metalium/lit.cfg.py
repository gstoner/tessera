# -*- Python -*-
# Minimal lit config for Tessera Metalium backend tests.

import os
import lit.formats

config.name = "TesseraMetalium"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir', '.test']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

# Look up tools from PATH
tools = ['mlir-opt', 'FileCheck', 'metalium-codegen-demo', 'tessera-metalium-opt']
for tool in tools:
    path = lit.util.which(tool, os.environ.get('PATH', ''))
    if path:
        config.environment[tool.upper().replace('-', '_')] = path
    else:
        config.available_features.discard(tool) if hasattr(config, 'available_features') else None

# Provide REQUIRES-style features (soft)
config.available_features = set([
    'tessera_metalium_dialect',
    'tessera_metalium_passes',
    'metalium_codegen_demo',
])

config.available_features.add('tessera_metalium_opt')
