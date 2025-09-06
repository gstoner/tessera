import os, lit.formats
config.name='Tessera-EBT-v2'
config.test_format=lit.formats.ShTest(True)
config.suffixes=['.mlir','.check','.tir.mlir']
config.test_source_root=os.path.dirname(__file__)
