# gfx1201 native replay cache — 2026-10-08

Owner E2E-REAL-6. Sync ROCM-NATIVE-REPLAY-CACHE-2026-10-08.
Isolated follow-up to draft PR895; paired with the gfx1151 owning packet.

Tajasaurus reports RX 9070 XT / gfx1201 through the live runtime and rocminfo.
The recorder requires an explicit architecture and rejects a live mismatch
before packaging. The packet binds compiler, source, replay cache, ancestry
verifier and recorder SHA256 values. All four Python source hashes match
the candidate committed in this worktree.

Six native GPU profiles pass independent float64 numerical references before
timing. Seven alternating A/B pairs retain warm images; only native ancestry
replay memoization changes. Images, symbols and compiler/toolchain fingerprints
match every arm. Each unary candidate drops from three subprocesses to one;
version queries remain zero. Artifact validation, replay comparison and ABI
projection still execute per package call.

| Profile | Uncached package ms | Cached package ms |
| --- | ---: | ---: |
| softmax 3x17 | 44.032 | 16.894 |
| softmax 5x300 | 43.851 | 16.606 |
| mean reduction 2x3x5 | 43.031 | 16.628 |
| mean reduction 4x33x7 | 43.000 | 16.920 |
| fp16 matmul 16x16x16, control | 39.226 | 38.977 |
| fp16 matmul 17x31x19, control | 39.227 | 39.099 |

These measure warm-image package wall time, not kernel performance. Matmul
controls retain three subprocesses; this change does not cache their ancestry.
Maximum numerical error across these profiles is 1.063e-7.

Owning focused tests: 50 passed, 9 skipped. The test log warns that
pytest-timeout is absent; timeout enforcement is not claimed. Compiler and
runtime are the existing LLVM 23.1.1 snapshot recorded in the packet, not a
fresh all-source build. Raw recorder and test logs are retained losslessly as
.log.gz files.

Sibling assessment: gfx1151 has its own six-profile numerical and timing
packet. x86 keeps its existing cache; NVIDIA and Apple do not enter this ROCm
path. Wider envelopes, matmul ancestry caching, discovery overhead and generic
batching/transpose closure remain open.

## Main synchronization identity guard

After merged PR894, replay identity also explicitly includes the working
directory, even if the resolved compiler/library bytes are unchanged. A host
regression proves one replay per directory and reuse within a directory.
The packet above retains its original measured source hashes; it predates
this identity guard and is not a fresh benchmark of the synchronized head.
The native passes, images and ABI remain unchanged.
