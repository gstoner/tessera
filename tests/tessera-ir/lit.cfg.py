
# -*- Python -*-
import os, lit.formats
config.name = "Tessera-IR-Verify-Canonicalize (src layout)"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')
