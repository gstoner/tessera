<!-- MERGE-START: P3D_HowTo -->
# How‑To: Using P3D in Tessera

## 1) Define your model (Python)
```python
import tessera as tsr

@tsr.function
def p3d_block(x: tsr.Tensor["B","C","D","H","W"]) -> tsr.Tensor:
    y = tsr.ops.conv3d(x, weight, bias, stride=(1,1,1), dilation=(1,1,1), padding=(1,1,1), bc="periodic")
    levels = tsr.ops.pyramid_build(y, factors=[(2,2,2),(4,4,4)])
    ctx   = tsr.ops.global_context(levels[0])
    up    = tsr.ops.upsample3d(ctx, size=(x.shape[2], x.shape[3], x.shape[4]), mode="trilinear")
    return up
```

## 2) Compile & run
```bash
tessera-opt p3d_model.mlir -tessera-autotune-p3d -tessera-lower-p3d | tessera-compile --target=cuda --arch=sm_90 -o p3d.kbin
```

## 3) Distributed 3D domain
Use Tessera mesh + neighbors/halo to shard `[D,H,W]` with automatic halo exchange.

## 4) Validation
- Compare against reference CFD fields (MSE, spectrum, invariants).
- Use Tessera profiler + roofline to ensure compute/memory balance.
<!-- MERGE-END: P3D_HowTo -->
