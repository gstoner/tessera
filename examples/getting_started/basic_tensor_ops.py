#!/usr/bin/env python3
"""Basic Tessera tensor operations example."""

import tessera as tsr
from tessera import Tensor

@tsr.function
def add_tensors(
    x: Tensor["B", "S", "D"],
    y: Tensor["B", "S", "D"]
) -> Tensor["B", "S", "D"]:
    """
    Add two tensors with shape polymorphism.
    Shapes B, S, D can be symbolic or dynamic.
    """
    return x + y

def main():
    print("🌟 Tessera Basic Tensor Operations")
    print("=" * 40)

    # Create tensors with symbolic shapes
    x = tsr.randn([4, "S", 512])   # Batch=4, seq=S, hidden=512
    y = tsr.randn([4, "S", 512])

    # Perform basic addition
    z = add_tensors(x, y)
    print(f"Addition result shape: {z.shape}")

    # Try a few other standard ops
    normed = tsr.nn.rms_norm(z, weight=tsr.ones([512]))
    print(f"RMSNorm output shape: {normed.shape}")

    activated = tsr.nn.swiglu(z, 
                              W_gate=tsr.randn([512, 2048]),
                              W_up=tsr.randn([512, 2048]),
                              W_down=tsr.randn([2048, 512]))
    print(f"SwiGLU output shape: {activated.shape}")

if __name__ == "__main__":
    main()