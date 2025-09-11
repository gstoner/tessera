# Tessera API Reference

## Table of Contents
1. [Core Types](#core-types)
2. [Tensor Operations](#tensor-operations)
3. [Kernel Programming](#kernel-programming)
4. [Distributed Computing](#distributed-computing)
5. [Compilation and JIT](#compilation-and-jit)
6. [Memory Management](#memory-management)
7. [Numerical Types](#numerical-types)
8. [Autodiff](#autodiff)
9. [Optimization](#optimization)
10. [Utilities](#utilities)

## Core Types

### Tensor

```python
class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.
    
    Parameters:
        data: Array-like data or shape tuple
        dtype: Data type (default: f32)
        device: Device placement (default: "cuda:0")
        requires_grad: Enable gradient computation (default: False)
        layout: Memory layout (default: "row_major")
    """
    
    def __init__(
        self,
        data: Union[ArrayLike, Shape],
        dtype: DType = f32,
        device: str = "cuda:0",
        requires_grad: bool = False,
        layout: str = "row_major"
    )
    
    # Properties
    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor shape"""
    
    @property
    def dtype(self) -> DType:
        """Data type"""
    
    @property
    def device(self) -> str:
        """Device placement"""
    
    @property
    def grad(self) -> Optional['Tensor']:
        """Gradient tensor"""
    
    @property
    def requires_grad(self) -> bool:
        """Whether gradient is computed"""
    
    # Methods
    def to(self, device: str) -> 'Tensor':
        """Move tensor to device"""
    
    def detach(self) -> 'Tensor':
        """Detach from computation graph"""
    
    def backward(self, gradient: Optional['Tensor'] = None):
        """Compute gradients"""
    
    def item(self) -> Number:
        """Get scalar value"""
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy array"""
    
    def cuda(self) -> 'Tensor':
        """Move to CUDA device"""
    
    def cpu(self) -> 'Tensor':
        """Move to CPU"""
```

### DType

```python
# Numerical data types
f32 = DType("float32")      # 32-bit float
f16 = DType("float16")      # 16-bit float
bf16 = DType("bfloat16")    # Brain float 16
f64 = DType("float64")      # 64-bit float

# Extended precision types
fp8_e4m3 = DType("fp8_e4m3")  # 8-bit float (4 exp, 3 mantissa)
fp8_e5m2 = DType("fp8_e5m2")  # 8-bit float (5 exp, 2 mantissa)
fp6 = DType("fp6")            # 6-bit float
fp4 = DType("fp4")            # 4-bit float

# Integer types
int8 = DType("int8")         # 8-bit integer
int16 = DType("int16")       # 16-bit integer
int32 = DType("int32")       # 32-bit integer
int64 = DType("int64")       # 64-bit integer
uint8 = DType("uint8")       # Unsigned 8-bit

# Boolean
bool = DType("bool")         # Boolean type
```

## Tensor Operations

### Creation Operations

```python
def zeros(shape: Shape, dtype: DType = f32, device: str = "cuda:0") -> Tensor:
    """Create tensor filled with zeros"""

def ones(shape: Shape, dtype: DType = f32, device: str = "cuda:0") -> Tensor:
    """Create tensor filled with ones"""

def full(shape: Shape, fill_value: Number, dtype: DType = f32, device: str = "cuda:0") -> Tensor:
    """Create tensor filled with value"""

def arange(start: Number, end: Number, step: Number = 1, dtype: DType = f32, device: str = "cuda:0") -> Tensor:
    """Create tensor with range of values"""

def linspace(start: Number, end: Number, steps: int, dtype: DType = f32, device: str = "cuda:0") -> Tensor:
    """Create tensor with linearly spaced values"""

def randn(shape: Shape, dtype: DType = f32, device: str = "cuda:0", generator: Optional[Generator] = None) -> Tensor:
    """Create tensor with normal distribution"""

def rand(shape: Shape, dtype: DType = f32, device: str = "cuda:0", generator: Optional[Generator] = None) -> Tensor:
    """Create tensor with uniform distribution"""

def randint(low: int, high: int, shape: Shape, dtype: DType = int32, device: str = "cuda:0") -> Tensor:
    """Create tensor with random integers"""
```

### Mathematical Operations

```python
# Arithmetic
def add(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise addition"""

def sub(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise subtraction"""

def mul(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise multiplication"""

def div(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise division"""

def pow(x: Tensor, exponent: Union[Tensor, Number]) -> Tensor:
    """Element-wise power"""

# Reduction operations
def sum(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Sum reduction"""

def mean(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Mean reduction"""

def max(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Maximum reduction"""

def min(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Minimum reduction"""

def var(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Variance reduction"""

def std(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Standard deviation"""

# Trigonometric
def sin(x: Tensor) -> Tensor:
    """Sine"""

def cos(x: Tensor) -> Tensor:
    """Cosine"""

def tan(x: Tensor) -> Tensor:
    """Tangent"""

def sinh(x: Tensor) -> Tensor:
    """Hyperbolic sine"""

def cosh(x: Tensor) -> Tensor:
    """Hyperbolic cosine"""

def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent"""

# Exponential and logarithm
def exp(x: Tensor) -> Tensor:
    """Exponential"""

def log(x: Tensor) -> Tensor:
    """Natural logarithm"""

def log2(x: Tensor) -> Tensor:
    """Base-2 logarithm"""

def log10(x: Tensor) -> Tensor:
    """Base-10 logarithm"""

def sqrt(x: Tensor) -> Tensor:
    """Square root"""

def rsqrt(x: Tensor) -> Tensor:
    """Reciprocal square root"""
```

### Linear Algebra Operations

```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication"""

def gemm(a: Tensor, b: Tensor, c: Optional[Tensor] = None, 
         alpha: float = 1.0, beta: float = 0.0,
         trans_a: bool = False, trans_b: bool = False) -> Tensor:
    """General matrix multiplication: alpha * A @ B + beta * C"""

def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product"""

def einsum(equation: str, *tensors: Tensor) -> Tensor:
    """Einstein summation"""

def transpose(x: Tensor, dim0: int = -2, dim1: int = -1) -> Tensor:
    """Transpose dimensions"""

def permute(x: Tensor, dims: Tuple[int, ...]) -> Tensor:
    """Permute dimensions"""

def reshape(x: Tensor, shape: Shape) -> Tensor:
    """Reshape tensor"""

def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten tensor"""

# Decompositions
def cholesky(x: Tensor) -> Tensor:
    """Cholesky decomposition"""

def svd(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Singular value decomposition"""

def qr(x: Tensor) -> Tuple[Tensor, Tensor]:
    """QR decomposition"""

def eig(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Eigenvalue decomposition"""
```

### Neural Network Operations

```python
# Activation functions
def relu(x: Tensor) -> Tensor:
    """Rectified linear unit"""

def gelu(x: Tensor) -> Tensor:
    """Gaussian error linear unit"""

def silu(x: Tensor) -> Tensor:
    """Sigmoid linear unit (Swish)"""

def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation"""

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation"""

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Log softmax"""

# Normalization
def layer_norm(x: Tensor, normalized_shape: Shape, 
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    """Layer normalization"""

def batch_norm(x: Tensor, running_mean: Tensor, running_var: Tensor,
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """Batch normalization"""

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Root mean square normalization"""

# Convolution
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1) -> Tensor:
    """2D convolution"""

# Pooling
def max_pool2d(input: Tensor, kernel_size: Tuple[int, int],
               stride: Optional[Tuple[int, int]] = None,
               padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """2D max pooling"""

def avg_pool2d(input: Tensor, kernel_size: Tuple[int, int],
               stride: Optional[Tuple[int, int]] = None,
               padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """2D average pooling"""

# Dropout
def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout regularization"""

# Loss functions
def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error loss"""

def cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None,
                  reduction: str = "mean") -> Tensor:
    """Cross entropy loss"""
```

## Kernel Programming

### Kernel Decorator

```python
@kernel(
    tile_shape: Optional[Tuple[int, ...]] = None,
    shared_memory: int = 0,
    num_warps: int = 4,
    autotune: bool = False,
    autotune_configs: Optional[List[Dict]] = None,
    target: str = "cuda"
)
def kernel_function(...):
    """
    Decorator for kernel functions.
    
    Parameters:
        tile_shape: Tile dimensions
        shared_memory: Shared memory size in bytes
        num_warps: Number of warps per block
        autotune: Enable autotuning
        autotune_configs: Autotuning configurations
        target: Target architecture
    """
```

### Tile Operations

```python
class tile:
    """Tile-level operations for kernel programming"""
    
    @staticmethod
    def linear_id() -> int:
        """Get linear tile ID"""
    
    @staticmethod
    def thread_id() -> Tuple[int, int, int]:
        """Get 3D thread ID"""
    
    @staticmethod
    def block_id() -> Tuple[int, int, int]:
        """Get 3D block ID"""
    
    @staticmethod
    def warp_id() -> int:
        """Get warp ID"""
    
    @staticmethod
    def lane_id() -> int:
        """Get lane ID within warp"""
    
    @staticmethod
    def group_id() -> int:
        """Get tile group ID"""
    
    @staticmethod
    def alloc_shared(shape: Shape, dtype: DType, 
                    swizzle: Optional[str] = None,
                    alignment: int = 16) -> Tensor:
        """Allocate shared memory"""
    
    @staticmethod
    def alloc_register(shape: Shape, dtype: DType) -> Tensor:
        """