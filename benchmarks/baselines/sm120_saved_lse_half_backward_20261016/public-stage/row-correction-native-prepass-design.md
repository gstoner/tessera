# Follow-up after current PR synchronization

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1; no implementation or promotion yet.

Source inspection confirms dQ computes rowStats per gradient dimension, dK recomputes it per key/dimension/query/grouped head, and broadcast dBias recomputes it per logical score. Saved-LSE dV already avoids it. For the measured medium profile this is source reuse factor1040 without bias /1169 with bias relative to28 unique rows; it is not a counter measurement or expected speedup. Source attribution JSON pins file bytes and locations.

Investigate native Schedule lowering of one checkpoint-backward Graph into a per-row FP32 correction producer and gradient consumer. A28-row profile needs112 bytes of native-owned correction storage. Preserve exact current P*dP accumulation order, masking, scale, GQA mapping, LSE seed subtraction and dtype contracts. Do not substitute dot(dO, rounded half O), a Python producer or an unregistered CUDA template.

Schedule/Tile must declare both producer and consumer, image entry identities, checked workspace size/lifetime and their dependency. Prefer one compiled PTX module with two explicit entries if the native packaging contract can represent both correctly; otherwise extend the native program member contract. Both require verified compiler projection rather than Python entry reconstruction. The CUDA owner must allocate bounded row scratch and submit producer before consumer on its stream; record producer/consumer event windows and their combined device/public cost. Skip correction production for V-only activity.

Exercise all current339 cases plus source/ABI tampering, row scratch overflow/capacity/lifetime, half/F32 primary, both half storages, seed/bias/causal/trace-at-one, and delayed producer writes. Compare candidate/control interleaved on identical stored inputs. Reuse existing default route until numerical and paired timing evidence warrants selection. Assess Apple/ROCm/x86 plans without transferring CUDA schedules or evidence. Finish and merge current typed public-owner PR before implementing this follow-up, keeping one PR at a time.

## Selection and latency experiments

Keep the existing in-kernel correction as the short-profile control. The extra prepass launch can outweigh avoided work at Sq3/Sk5, so measure both small and long-key rows before selecting any rule. A native Schedule selection must encode its policy and checked envelope; Python should not reconstruct phases or choose hidden alternate kernels. Compare explicit two-launch and native captured-graph execution as separate latency domains, preserving source images and launch geometry. Persistent/cooperative execution needs its own exact-device progress/barrier proof and is not assumed.

Host overhead is independent: execute_family already caches compiled JSON/certificates by source, active roles and compiler identity, but still reconstructs RuntimeArtifact and hashes/passes provenance per invocation. Profile call counts on a completed source to establish which work remains on the warm path; do not infer a speedup from source inspection. Preserve tool-selection invalidation, immutable source identities, per-invocation dtype/shape/stream/capacity checks and explicit lifetime.

Current causal contract uses q+max(Sk-Sq,0); the independent oracle matches it. The current profile does not establish fully masked-row behavior for new mask/window envelopes. A prepass must preserve this declared rule and acquire separate proof before widening masks.
