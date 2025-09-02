# Tessera API Reference - Index

Welcome to the **Tessera Programming Model API Reference**.  
This documentation is split into four volumes for clarity and modularity.

---

## 📑 Volumes

1. [Volume 1: Frontend & Type System](Tessera_API_Vol1_Frontend_and_TypeSystem.md)  
   - Python and Rust APIs  
   - Shape-polymorphic type system  
   - Numerical policies  
   - Effects & determinism  

2. [Volume 2: Operations](Tessera_API_Vol2_Operations.md)  
   - Normalization (RMSNorm, LayerNorm)  
   - Activations (SwiGLU, GELU, Swish)  
   - Attention mechanisms (FlashAttention, MLA, MQA, GQA)  
   - Positional encodings (RoPE, ALiBi, etc.)  
   - Embeddings (Casted, Learned, Quantized)  
   - Distributed ops  

3. [Volume 3: IR & Target](Tessera_API_Vol3_IR_and_Target.md)  
   - Graph IR, Schedule IR, Tile IR  
   - Target IR dialect and lowering passes  
   - Platform-specific optimizations (CUDA, AMD, Intel)  
   - Autotuning and profiling support  

4. [Volume 4: Runtime & Deployment](Tessera_API_Vol4_Runtime_and_Deployment.md)  
   - Tessera runtime engine  
   - Host-device orchestration  
   - Autotuning & profiling APIs  
   - Production deployment (C/C++ APIs, CMake integration)  
   - Example models and usage patterns  

---

## 🔍 How to Use
- Start with **Volume 1** to understand the programming model.  
- Use **Volume 2** as the operator reference for implementing models.  
- Refer to **Volume 3** for compiler IR and backend integration.  
- Consult **Volume 4** for deployment, profiling, and production workflows.  

---

## 📦 Download
This index links to the volumes included in the packaged zip:

- [Tessera_API_Reference_Expanded.zip](Tessera_API_Reference_Expanded.zip)
