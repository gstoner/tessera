#!/usr/bin/env python3
"""Basic Tessera tensor operations example.

Walks through the canonical Tessera surface:
  * `@tessera.jit` to compile a function
  * `tessera.Tensor[...]` shape-annotation syntax
  * Replicated tensor factories (`tessera.randn`, `tessera.ones`)
  * `tessera.nn` functional wrappers (RMSNorm, SwiGLU)

Runs on the CPU reference path — no accelerator required.
"""

import tessera


@tessera.jit
def add_tensors(
    x: tessera.Tensor["B", "S", "D"],
    y: tessera.Tensor["B", "S", "D"],
) -> tessera.Tensor["B", "S", "D"]:
    """Elementwise add. Shapes B/S/D are symbolic and bound at call time."""
    return tessera.ops.add(x, y)


def main():
    print("Tessera Basic Tensor Operations")
    print("=" * 40)

    B, S, D, FF = 4, 16, 512, 2048

    # Replicated DistributedArray factories — single-rank ergonomic shortcuts
    x = tessera.randn((B, S, D), dtype="fp32")
    y = tessera.randn((B, S, D), dtype="fp32")

    # 1. Compiled elementwise add
    z = add_tensors(x.numpy(), y.numpy())
    print(f"add result shape: {z.shape}")

    # 2. RMSNorm with a learnable scale
    weight = tessera.ones((D,), dtype="fp32")
    normed = tessera.nn.rms_norm(z, weight=weight.numpy())
    print(f"rms_norm output shape: {normed.shape}")

    # 3. SwiGLU MLP block (D -> FF -> D)
    W_gate = tessera.randn((D, FF), dtype="fp32").numpy()
    W_up = tessera.randn((D, FF), dtype="fp32").numpy()
    W_down = tessera.randn((FF, D), dtype="fp32").numpy()
    activated = tessera.nn.swiglu(normed, W_gate, W_up, W_down)
    print(f"swiglu output shape: {activated.shape}")


if __name__ == "__main__":
    main()
