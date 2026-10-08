# Native-bound frontend attention argument order on SM120

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-ATTENTION-ARGUMENT-ORDER-2026-10-02.

Exact device: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0.

The frontend's argument indices are not necessarily Q/K/V/bias roles.
Before this fix, attention(k,bias,v,q) requested argument indices (1,3)
for wrt=(bias,q), while the native checkpoint exported canonical gradients.
The old wrapper selected dK/dBias and capture assumed canonical inputs.

Native paired-AD checkpoint export now derives the input permutation from
typed block arguments. Native Schedule hashes that mapping into its contract,
and replay rejects changes. Package projection retains the checked mapping.
Capture accepts the original frontend argument order, including keyword-bias
placement; backward maps wrt indices to the physical gradient roles.

28 exact-device rows cover canonical and two permuted biased signatures plus
an unbiased attention(v,q,k) signature; full/causal, GQA, batch two, unequal
D/Dv, and Sq<Sk/Sq>Sk. Expected capture/input and gradient mappings are
specified independently of the compiler's metadata. Caller input mutation,
repeated backward and the independent float64 oracle pass.
Maximum absolute gradient error: 2.55680632e-07.

452 focused tests pass, including Schedule hash mutation, invalid/out-of-range/
duplicate/boolean mappings, capture arity and keyword bias, and exact-device
permuted capture/gradient checks. Both compiler tools were rebuilt in WSL.

Three capture/backward wall samples per row are recorded independently.
They include allocation, module loading/copying and synchronization. They are
not device kernel time and support no speedup claim. Existing checkpoint
CUDA-event windows remain a separate evidence packet.

Remaining: aliasing/composed producer graphs, broadcast-bias reduction,
permuted JVP/higher derivatives, broader dtype/layout and sibling native
consumers. This proves the isolated native attention VJP compile API,
not universal JIT dispatch or any non-NVIDIA physical schedule.

Reproduce from the owning repository root:
    source .build-sm120-w1-1/validation-env.sh
    .venv/bin/python benchmarks/nvidia/benchmark_attention_argument_order.py --output packet.json
