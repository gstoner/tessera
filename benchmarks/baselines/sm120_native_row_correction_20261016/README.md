# SM120 native attention row correction

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-NATIVE-ROW-CORRECTION-20261016.

This packet follows the original typed Graph/native AD product through verified
Schedule/Tile, NVIDIA/LLVM native code, the checked C++ owner and public replay.
Each subpacket retains its own measured source/tool witnesses and proof scope.

- [contract](contract/README.md): private typed operation and initial verifier proof.
- [projection](projection/README.md): native AD projection and source-role lineage.
- [schedule_target](schedule_target/README.md): native row producer and device proof.
- [consumer](consumer/README.md): explicit residual backward and compact activity.
- [prepared_owner](prepared_owner/README.md): diagnostic three-image native owner.
- [thread_review](thread_review/README.md): exact Tile producer/consumer thread ABI correction.
- [public_program](public_program/README.md): portable VJP v3, original JIT and public A/B.

The public prepass is opt-in; tiny regressions remain. Dynamic/layout/composition
and sibling-backend execution remain follow-ups. This packet does not establish
completion of the full original five-slice goal.
