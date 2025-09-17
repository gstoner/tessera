<!-- MERGE_START: ATLAS_HOWTO -->
# How-To: Use Atlas Memory in Tessera (v0.1) — 2025-09-17

```mlir
// Pseudo IR sketch
%mem = atlas.memory.create (1024, 256, 4) : !atlas.memory
atlas.optimizer.set %mem, "muon" lr=1e-3
%Kf = atlas.feature.map %K, "poly" {degree = 3} : tensor<?x256xf16> -> tensor<?xMxf16>
%Qf = atlas.feature.map %Q, "poly" {degree = 3} : tensor<?x256xf16> -> tensor<?xMxf16>
%mem2 = atlas.memory.update %mem, %Kf, %V, 4096 : !atlas.memory, tensor<?xMxf16>, tensor<?xNxf16> -> !atlas.memory
%Y    = atlas.memory.read %mem2, %Qf : !atlas.memory, tensor<?xMxf16> -> tensor<?xNxf16>
```

### Compile
```
tessera-opt model.mlir -tessera-atlas | tessera-compile --target=nvidia.sm90
```

### Tips
- Start with `W=4k–16k`, poly degree 2–4.
- Use FP32 accumulators for optimizer state; store mem in BF16.
<!-- MERGE_END: ATLAS_HOWTO -->