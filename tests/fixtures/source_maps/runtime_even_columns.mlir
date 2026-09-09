// Native runtime-shaped view contract: all rows, every other column.
// Extents come from the input descriptor; there is no Python shape bucket.
module {
  func.func @even_columns(%x: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes {tessera.autodiff = "reverse"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %rows = tensor.dim %x, %zero : tensor<?x?xf32>
    %cols = tensor.dim %x, %one : tensor<?x?xf32>
    %padded = arith.addi %cols, %one : index
    %count = arith.divui %padded, %two : index
    %view = tensor.extract_slice %x[0, 0] [%rows, %count] [1, 2]
      : tensor<?x?xf32> to tensor<?x?xf32>
    return %view : tensor<?x?xf32>
  }
}
