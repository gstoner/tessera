# -*- Python -*-
# Lit configuration for Tessera CLI integration tests.

import os
import lit.formats

config.name         = "tessera-cli-lit"
config.test_format  = lit.formats.ShTest(True)
config.suffixes     = ['.mlir', '.txt', '.sh']
config.test_source_root = os.path.dirname(__file__)

# Directory containing the built tessera-* binaries.
# Set by -Dtessera_tools_dir=... or inferred from PATH.
tools_dir = lit_config.params.get(
    'tessera_tools_dir',
    os.path.join(os.path.dirname(__file__), '../../../../..', 'build', 'bin'))

config.environment['PATH'] = tools_dir + os.pathsep + os.environ.get('PATH', '')

# Substitutions available in test files.
config.substitutions.append(('%tessera_src',
    os.path.join(os.path.dirname(__file__), '..', '..')))
config.substitutions.append(('%tools_dir', tools_dir))

# Skip tests that require the real toolchain
config.available_features.add('shell')
if os.path.isfile(os.path.join(tools_dir, 'tessera-opt')):
    config.available_features.add('tessera-tools')
