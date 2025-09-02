# Tessera API Reference - Volume 3
## Intermediate Representation (IR) & Target IR

### 1. Graph IR
- Nodes for parameters, constants, matmul, softmax【18†source】

Example Node:
```mlir
Node(id=4, op="matmul", inputs=[0,1], transpose_b=True, shape=["B","H","S","S"])
```

---

### 2. Schedule IR
- Defines tiling, fusion, memory plan【18†source】

```mlir
ScheduleIR {
  fused_kernels: [FusedKernel(name="flash_attention_fused", operations=[...])]
}
```

---

### 3. Tile IR
- Warp/thread mapping, shared/register allocation【18†source】

```mlir
TileMMA(
    a=RegisterTile("q_reg"),
    b=RegisterTile("k_reg"),
    c=RegisterTile("scores_reg"),
    instruction=MMAInstruction(shape=(16,8,16), dtype=(bf16,bf16,f32))
)
```

---

### 4. Target IR Dialect
- KernelOp, LaunchOp, PTXInstrOp【20†source】

```mlir
tessera_target.kernel @flash_attention_kernel(...) config = #tessera_target.kernel_config<grid=[128,64], block=[256,1], shared=49152, regs=128>
```

---

### 5. Target-Specific Optimizations

#### NVIDIA Hopper (SM_90)
```mlir
%result = tessera_target.tensor_core "wgmma" (%A_desc, %B_desc, %C_acc)
```

#### AMD RDNA3
```mlir
%result = tessera_target.hip_instr "v_wmma_f32_16x16x16_f16" (%A_wave, %B_wave, %C_wave) arch="gfx1100"
```

---

### 6. Integration with LLVM/MLIR
- Lowering patterns【23†source】
- Conversion to NVVM/ROCm ops【23†source】
