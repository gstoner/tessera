import lit.formats
config.name = "tessera-rocm"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.txt']
