# SM120 checkpoint bias-gradient package proof

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.

Super-Bear NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0 executes exact-shaped f32 checkpoint backward
through Graph -> Schedule -> Tile -> NVIDIA Target/NVVM/PTX and a checked
eleven-buffer/seven-dimension ABI. The fourth result is [B,Hq,Sq,Sk] bias
gradient. Each element has one deterministic writer; its derivative omits
the extra Q.K scale used by dQ/dK.

Six rows cover full/causal, Sq<Sk and Sq>Sk, grouped query, batch two,
and independent D/Dv. The independent float64 oracle checks forward O/LSE
and all four gradients. Host outputs start as NaN. Resident outputs are
NaN-poisoned before warmup/timing and checked after the event window.
Maximum absolute gradient error: 1.43987783e-07.
Existing three-gradient and new four-gradient packages remain tested.

Three samples, 100 launches per resident window, and 20 warmup launches.
Resident CUDA-event window medians: 0.011120–0.029601 ms.
Host package end-to-end medians: 2.028543–14.347226 ms.
CUDA windows include driver submission gaps. Host end-to-end includes
staging, allocation, synchronization and output copy. No speedup or selector
promotion is claimed.

400 focused tests passed in host WSL with matching rebuilt compiler/bridge.
JSON records actual GPU UUID/driver, source fingerprints, compiler/bridge
hashes, Schedule digest, PTX image hash and raw timing samples.

Remaining: automatic paired-AD bias export, JIT wrt/gradient ordering,
private resident tape bias-gradient ownership, broader composed graphs,
and sibling-backend native execution. This explicit checkpoint package proof
does not close those requirements.

Reproduce from the repository root on the owning SM120 host:
source .build-sm120-w1-1/validation-env.sh
.venv/bin/python benchmarks/nvidia/benchmark_checkpoint_bias_gradient.py --output packet.json
