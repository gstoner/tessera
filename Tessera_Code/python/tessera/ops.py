
from .graph import Tensor

def matmul(A: Tensor, B: Tensor) -> Tensor:
    return Tensor((A.shape[0], B.shape[1]), dtype=A.dtype, device=A.device)

def conv2d(x: Tensor, w: Tensor, stride=(1,1), padding=(0,0), layout="nhwc") -> Tensor:
    return Tensor(x.shape, dtype=x.dtype, device=x.device)
