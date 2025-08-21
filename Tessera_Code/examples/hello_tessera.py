
from tessera import Graph, Tensor, ops
g = Graph()
a = Tensor((2,3)); b = Tensor((3,4))
c = ops.matmul(a,b)
print("matmul shape:", c.shape)
