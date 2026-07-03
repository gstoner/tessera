//===- LoweringUtils.h - Shared Apple lowering helpers ---------*- C++ -*-===//
//
// Audit 2026-06-10 (CODE_AUDIT_2026_06_10 §4) — the `extractPtr` /
// `ensureExternalDecl` helpers were copy-pasted byte-for-byte across ~18 Apple
// Tile→Target lowering passes, then hoisted into `tessera::apple`.
//
// Workstream A1 (COMPILER_REFACTOR_PLAN §3, Workstream A) — the same helpers
// were *also* duplicated in the x86 backend. They now live once in
// `tessera::common` (Tessera/Common/Lowering.h); this header re-exports them
// into `tessera::apple` via `using` declarations, so the ~18 unqualified call
// sites inside `tessera::apple::{anonymous}` resolve unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_APPLE_LOWERINGUTILS_H
#define TESSERA_TARGET_APPLE_LOWERINGUTILS_H

#include "Tessera/Common/Lowering.h"

namespace tessera {
namespace apple {

using ::tessera::common::ensureExternalDecl;
using ::tessera::common::extractPtr;

} // namespace apple
} // namespace tessera

#endif // TESSERA_TARGET_APPLE_LOWERINGUTILS_H
