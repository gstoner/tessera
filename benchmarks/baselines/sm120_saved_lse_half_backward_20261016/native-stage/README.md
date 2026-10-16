# Frozen native half-backward compiler/kernel stage

Source: 9b2159e65581e9f6df01d3f39d08e99a33704e6f. Exact tree/tool/provider and
owning RTX5070 identities are in native-stage-witness.json.gz. Source/tools
remain unchanged through both gates; the checked source is clean.

- Native AD/Graph/Schedule/Tile and registry/packet/audit lane: 639 passed,
  zero skips/failures.
- Owning device lane: 167 passed, zero skips/failures. Includes 96 new typed
  direct forward/backward kernel cases and 71 existing saved-forward/FP32
  checkpoint cases. These counts are separate scopes and are not added.
- All 33 generated documentation views regenerate; AST graph refresh completes.
  Existing archived-source extractor warnings are retained in graphify.log.gz.

The new kernel matrix covers FP16/BF16, primary half/FP32 output, optional
FP32 LSE seeds, broadcast bias gradients, causal/noncausal, batching/MHA/GQA,
Sq=1, Sk=129 and Sq>Sk. Independent FP64 references use identical stored
inputs. Protected stores and poisoned saved O check role widths and the
native FP32 probability row correction.

This stage invokes compiler-generated images through the CUDA driver. It is
not proof of the unfinished typed public backward descriptor, retained owner,
standalone public replay, performance promotion or complete five-slice goal.
No new timings are recorded. The source bound to these receipts must not be
relabelled when later packaging/runtime changes land. The executing helper and
raw JUnit/logs are losslessly compressed with hashes in the manifest.
