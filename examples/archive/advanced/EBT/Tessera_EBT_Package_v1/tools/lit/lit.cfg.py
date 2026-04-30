import os
config.name = "Tessera-EBT"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.check']
config.test_source_root = os.path.dirname(__file__)
