"""Test-only text model for cache driver doubles; never a codegen route."""
import hashlib
import re

#: Host-side scaffolding TileToROCM leaves around the directive in a scheduled
#: Target IR module. None of it reaches the HSACO (the generator emits
#: the kernel from the directive; the host function is not serialized). Any
#: other operation fails closed: dropping an op we have not audited could drop
#: something the binary depends on.
_SHAPE_FREE_SCAFFOLD_OPS = frozenset({
    "module", "func.func", "return", "func.return", "arith.constant",
    "arith.index_cast", "bufferization.to_buffer", "bufferization.to_tensor",
    "memref.alloc", "memref.extract_aligned_pointer_as_index", "llvm.inttoptr",
})

_OP_LINE_RE = re.compile(r'^\s*(?:%[^=\n]+=\s*)?"?([A-Za-z_][\w.]*)')
_MODULE_HEADER_RE = re.compile(r"^module(?: attributes \{.*\})? \{$")
_NAME_ATTR_RE = re.compile(r'(?:(?<=\{)|(?<=, ))name = "([^"\\]*)"')

# Exact-Tile memo: one Tile text -> its shape-free Target IR module. Saves the
# Tile -> Target run on an exact repeat; a new shape pays that run (~20 ms),
# never the binary compile of an identity already in `_cache`.



def project_reference(target_ir: str, *, family: str, directive: str) -> str:
    """Project one audited ROCm Target IR module onto its kernel identity.

    Returns a Target IR module holding only the module header and the one
    directive, with the directive's ``name`` replaced by a symbol derived from
    everything else in that module. Static extents live only in the host
    scaffolding (the function signature and ``arith.constant`` launch
    arguments), which this drops; every directive attribute -- storage,
    accumulator, kind, axis, keepdims, layout, ``inner_is_one``, exp/ftz/NaN
    policy, arch -- is kept verbatim, so each is in the cache key. The binary is
    then compiled from exactly this text, so the key covers the binary's input
    by construction rather than by an audit of what a generator reads.
    """
    lines = [line for line in target_ir.splitlines() if line.strip()]
    if not lines or not _MODULE_HEADER_RE.match(lines[0].strip()):
        raise RuntimeError("ROCm shape-free kernel identity requires a single top-level Target IR module")
    header = lines[0].strip()
    if family in {"paged_kv", "moe_dispatch"} and (
        len(re.findall(r"(?m)^\s*llvm\.func @", target_ir)) != 1
        or len(re.findall(r"(?m)^\s*llvm\.return\b", target_ir)) != 1
    ):
        raise RuntimeError(f"ROCm {family} Target IR needs one checked LLVM wrapper")
    directive_lines: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "}":
            continue
        match = _OP_LINE_RE.match(line)
        name = match.group(1) if match else ""
        if name.startswith("tessera_rocm."):
            directive_lines.append(stripped)
        elif family in {"paged_kv", "moe_dispatch"} and name in {"llvm.func", "llvm.return"}:
            # Native Schedule/Tile emits one host wrapper around the direct
            # directive. Its shape-bound signature and replay contract do not
            # enter the GPU image; the exact directive below is the code input.
            continue
        elif name not in _SHAPE_FREE_SCAFFOLD_OPS:
            raise RuntimeError(
                f"ROCm shape-free kernel identity cannot drop unaudited Target IR operation {name or stripped!r}"
            )
    if len(directive_lines) != 1:
        raise RuntimeError(f"ROCm shape-free kernel identity requires exactly one {directive} directive")
    line = directive_lines[0]
    if family == "matmul":
        # Schedule ancestry is checked against the replayed Tile artifact and
        # remains in the launch descriptor. The generator does not read this
        # provenance-only attribute; leaving it on the directive would make
        # each runtime shape a distinct image despite identical GPU code.
        schedule_hash = re.findall(
            r', tessera\.schedule_hash = "([0-9a-f]{64})"', line
        )
        if len(schedule_hash) != 1:
            raise RuntimeError(
                "ROCm matmul shape-free identity requires one Schedule hash"
            )
        line = re.sub(
            r', tessera\.schedule_hash = "[0-9a-f]{64}"', "", line
        )
    if not (line.startswith(directive + " {") and line.endswith("}")):
        raise RuntimeError(f"ROCm shape-free kernel identity requires one attribute-only {directive} directive")
    if len(_NAME_ATTR_RE.findall(line)) != 1:
        raise RuntimeError(f"ROCm {directive} directive must carry exactly one kernel name")
    anonymous = _NAME_ATTR_RE.sub('name = ""', line)
    identity = hashlib.sha256(
        "\x1f".join(("tessera.rocm_shape_free_kernel.v1", family, header, anonymous)).encode()
    ).hexdigest()
    symbol = f"tessera_rocm_{family}_{identity[:16]}"
    named = _NAME_ATTR_RE.sub('name = "' + symbol + '"', line)
    return header + "\n  " + named + "\n}\n"
