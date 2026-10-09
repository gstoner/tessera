# Native scaled JVP host-frame preparation

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SCALED-MAP-AXIS-INTEGRATION-20261008. Follow-up in PR902.

## Architecture and route

Live gfx1201 / RX 9070 XT, checked in the recorder before compilation.
The existing typed scaled-product Graph -> Schedule -> Tile -> ROCm Target ->
LLVM/native-image program owns primal and tangent arithmetic. This change
moves positive-stride host frame materialization below the Python binding into
the checked C++ byte packer. It changes neither the GPU schedule nor image ABI.
Backing capacity is checked before cold frontend certification can read a
borrowed view. Tangents require fp32 storage without an implicit conversion.
Cold independent frontend numerical certification remains an oracle.

The packer collapses contiguous suffixes, copies rows in blocks, and computes
outer offsets once per row. Dense C/F span checks use logical bytes while still
walking the backing owner and rejecting forged allocation bounds. Positive
strides, overflow, source/destination disjointness and destination capacity
checks remain enforced.

## Current-source proof

- 368 WSL host storage, frame, map and diagnostic/pass registry tests pass.
- 36 exact-gfx1201 new frame and existing map tests pass with no hardware skips.
  The owning pytest installation reports its existing unknown timeout config.
- Four exact-RTX5070 attention JVP regression tests pass with current jit.py.
  This does not establish NVIDIA scaled-product derivative parity.
- Shared movement replay: 52 gfx1151 pass/four architecture-family skips and
  48 gfx1201 pass/six architecture-family skips. This proves shared span
  admission for those existing routes, not gfx1151 FP8 WMMA support.
- Whole Python Ruff plus new fixture/recorder Ruff passes; mypy ratchet stays 0.
- delivery.json verifies all nine final packet source hashes against this tree.
- Native HIP translation-unit build and earlier 42-case row-retune regression
  logs are retained as historical intermediate checks, not final-source proof.

## Matched benchmark and remaining cost

The diagnostic control loads native_jvp from frozen commit
747c3364e4c20a55c57b112c3032a08838ee60c1. Both arms bind the same owner and
reuse an identical compiled program/image package. The reference method is
recorded beside the final JSON. Python compact-copy preparation is permitted
only in this diagnostic arm. Candidate warm calls forbid that compaction;
both arms forbid compilation. Numerical checks precede and follow timing.

Nine profiles cover KN/NK single, nested and Cartesian maps, two composed
ragged products and a 200x129x1536 composed product. Each uses seven rotating
paired windows of 32 completed public calls. The final median paired
native/reference ratios are 1.0234–1.0451: native preparation remains about
2.3–4.5% slower. The original per-element packer had a 1.4175 large-case ratio;
row-block packing reduced that large-case gap to about 4.5%. Initial/rows JSON
files retain their distinct source and runtime hashes as historical experiments.
They are not current-source certification.

Resident native-program events exclude host preparation/transfers and use
windows greater than 20ms. They are a separate scope; no event/wall speedup,
default GPU schedule promotion, or generic AD closure is claimed. Current
source, loaded runtime, compiler tools, package and image hashes are in the
final packet.

Reproduce on the owning gfx1201 host: extract jit.py from frozen commit
747c3364e4c20a55c57b112c3032a08838ee60c1 with git show into a scratch file,
then run benchmark_native_scaled_jvp_frame.py with --reference-jit pointing
to that file and --output pointing to a scratch JSON. Use the owning validation
environment and freshly built current native program runtime.

Generic scaled_matmul batching/transpose closure, nonzero output axes, dynamic
maps and wider storage derivatives remain open. No primitive coverage state
or zero-open assertion changes.
