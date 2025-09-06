# -*- Python -*-
import os
config.name = "tessera-cli-lit"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.txt']
config.test_source_root = os.path.dirname(__file__)
