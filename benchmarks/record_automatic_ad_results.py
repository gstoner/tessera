"""Native AD-generated dynamic public result fixture (2 or 4 elements)."""

def source():
    return 'module {\n func.func @choose(%x: tensor<4xf32>) -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {\n  %z = arith.constant 0 : index\n  %two = arith.constant 2 : index\n  %four = arith.constant 4 : index\n  %zero = arith.constant 0.0 : f32\n  %v = tensor.extract %x[%z] : tensor<4xf32>\n  %cond = arith.cmpf ogt, %v, %zero : f32\n  %n = arith.select %cond, %two, %four : index\n  %slice = tensor.extract_slice %x[0][%n][1] : tensor<4xf32> to tensor<?xf32>\n  %sq = "tessera.mul"(%slice,%slice) : (tensor<?xf32>,tensor<?xf32>) -> tensor<?xf32>\n  return %sq : tensor<?xf32>\n }\n}\n'
