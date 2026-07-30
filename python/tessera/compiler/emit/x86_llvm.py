"""Compatibility import for the renamed portable-C x86 candidate.

The implementation never emitted LLVM IR.  Import from
``tessera.compiler.emit.x86_c`` in new code.
"""

from tessera.compiler.emit.x86_c import (  # noqa: F401
    X86CEmitter,
    X86CRunner,
    X86GenericCCandidate,
    host_supports_x86_64_v4,
)

__all__ = [
    "X86CEmitter",
    "X86CRunner",
    "X86GenericCCandidate",
    "host_supports_x86_64_v4",
]
