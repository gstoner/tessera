#!/usr/bin/env python3
import numpy as np
from tessera_kan import KANLinear, bspline_params

def main():
    B,I,O = 32, 8, 16
    x = np.random.randn(B, I).astype(np.float32)
    layer = KANLinear(I, O, **bspline_params(k=3, grid_min=-2.0, grid_max=2.0, grid_size=16))
    y = layer(x)
    print("y shape:", y.shape, "mean:", float(y.mean()))

if __name__ == "__main__":
    main()