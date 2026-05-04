import lit.formats

config.name = "tessera-nvidia"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir", ".txt"]
