"""Logical dense-storage 2:4 matrices lowered to internal sparse Schedule IR.

This is a compiler producer and device-proof surface, consumed by the opt-in sparse capture adapter.
The fourth output contains one validity word per lane/tile. Consumers MUST reject
all results if any word is zero; no pruning or implicit sparsification is allowed.
"""
from itertools import combinations


def sparse_logical_schedule_ir(m: int, n: int, k: int, dtype: str, *, accum: str = "f32", output_storage: str | None = None, rhs_dtype: str | None = None, integer_bits: int = 8) -> str:
    """Build tiled A[M,K] @ B[K,N], with explicit accumulator storage and validity output.

    A and B use ordinary contiguous row-major storage. Packing and sparse index
    selection execute in the compiled kernel. Each wave owns one 16x16 tile and
    loops over K in steps of 32; no register payload is supplied by Python.

    **B MUST BE GATHERED COLUMN-MAJOR. This is an ISA requirement, not a
    preference.** RDNA4 ISA 7.12, sparse matrices: "When the A-matrix is a 4:2
    sparse matrix, the corresponding B-matrix must be (K x N), and loaded in
    column-major order."

    It is satisfied here by the addressing below: `%bcol` is a per-lane COLUMN
    (`col0 + low`) held fixed across the gather, while the row term walks K, so
    every one of the 16 elements in `%bv` shares a column and differs in k --
    `bi = (k-varying) * n + bcol`. A change that made the gather contiguous
    (stride 1, walking N) would be faster and WRONG, and would produce silently
    incorrect sparse results rather than an error.

    Stated 2026-09-20 because it was previously implicit in the index
    arithmetic and nothing tested it (docs/backends/rocm/wmma-fragment-layout.md
    10j.5). Gated by `test_sparse_b_gather_is_column_major`.
    """
    if any(type(x) is not int or x <= 0 for x in (m, n, k)) or m % 16 or n % 16 or k % 32:
        raise ValueError("sparse logical matrices require positive M/N multiples of 16 and K of 32")
    types = {"float16":"f16", "bfloat16":"bf16", "int8":"i8", "uint8":"i8", "float8_e4m3fn":"f8E4M3FN", "float8_e5m2":"f8E5M2"}
    if dtype not in types:
        raise ValueError("sparse logical matrices require supported half/FP8/signed/unsigned-i8 storage")
    rhs_dtype = dtype if rhs_dtype is None else rhs_dtype
    fp8_types = {"float8_e4m3fn", "float8_e5m2"}
    if rhs_dtype != dtype and not ((dtype in fp8_types and rhs_dtype in fp8_types) or (dtype in {"int8","uint8"} and rhs_dtype in {"int8","uint8"})):
        raise ValueError("mixed sparse operands require two FP8 or two integer formats")
    e = types[dtype]
    be = types[rhs_dtype]
    integer = e == "i8"
    if type(integer_bits) is not int or integer_bits not in (4,8) or (integer_bits != 8 and not integer):
        raise ValueError("sparse integer width requires integer operands and 4 or 8 bits")
    packed = e

    if integer and accum == "f32":
        raise ValueError("signed int8 sparse multiplication requires i32 accumulation")
    output = "i32" if integer else "f32"
    destination = output if output_storage is None else output_storage
    if destination not in ({"i32"} if integer else ({"f32", e} if e in {"f16", "bf16"} else {"f32"})):
        raise ValueError("sparse output storage must match operands or accumulation")
    bits = "i16" if e in {"f16", "bf16"} else "i8"
    if accum not in ({"i32"} if integer else ({"f32", e} if bits == "i16" else {"f32"})):
        raise ValueError("sparse accumulator disagrees with its input format")
    zero = "0" if integer else "0.0"
    tiles = (m // 16) * (n // 16)
    lines = [f'''module attributes {{gpu.container_module}} {{
  gpu.module @sparse {{
    gpu.func @probe(%a: memref<{m*k}x{e}>, %b: memref<{k*n}x{be}>,
                    %out: memref<{m*n}x{destination}>, %status: memref<{tiles*32}xi32>) kernel
        attributes {{gpu.known_block_size = array<i32: 32, 1, 1>}} {{''']
    def emit(s):
        lines.append("      " + s)
    constants = sorted(set([0, 1, 2, 3, 4, 8, 16, 20, 32, k, n, n // 16] + list(range(17))))
    for x in constants:
        emit(f"%c{x} = arith.constant {x} : index")
    emit(f"%zbits = arith.constant 0 : {bits}")
    emit("%z32 = arith.constant 0 : i32")
    emit("%one32 = arith.constant 1 : i32")
    emit("%yes = arith.constant true")
    emit("%no = arith.constant false")
    if integer_bits == 4:
        for side, dt in (("a",dtype),("b",rhs_dtype)):
            lo, hi = (-8,7) if dt == "int8" else (0,15)
            emit(f"%{side}min = arith.constant {lo} : i8")
            emit(f"%{side}max = arith.constant {hi} : i8")
    emit(f"%za = arith.constant dense<{zero}> : vector<8x{e}>")
    emit(f"%zb = arith.constant dense<{zero}> : vector<16x{be}>")
    emit(f"%zc = arith.constant dense<{zero}> : vector<8x{accum}>")
    emit("%lane = gpu.thread_id x")
    emit("%tile = gpu.block_id x")
    emit(f"%tm = arith.divui %tile, %c{n//16} : index")
    emit(f"%tn = arith.remui %tile, %c{n//16} : index")
    emit("%row0 = arith.muli %tm, %c16 : index")
    emit("%col0 = arith.muli %tn, %c16 : index")
    emit("%low = arith.remui %lane, %c16 : index")
    emit("%half = arith.divui %lane, %c16 : index")
    emit("%arow = arith.addi %row0, %low : index")
    emit(f"%abase = arith.muli %arow, %c{k} : index")
    emit("%bcol = arith.addi %col0, %low : index")
    emit("%half8 = arith.muli %half, %c8 : index")
    emit(f"%loop:2 = scf.for %kk = %c0 to %c{k} step %c32 iter_args(%acc = %zc, %valid = %yes) -> (vector<8x{accum}>, i1) {{")
    idx, valid = "%z32", "%valid"
    avalues: list[str] = []
    for reg in range(4):
        off = (reg // 2) * 16 + (reg % 2) * 4
        emit(f"%g{reg}a = arith.addi %kk, %half8 : index")
        emit(f"%g{reg}b = arith.addi %g{reg}a, %c{off} : index")
        emit(f"%g{reg} = arith.addi %abase, %g{reg}b : index")
        vals, zeros = [], []
        for j in range(4):
            name = f"g{reg}v{j}"
            emit(f"%{name}i = arith.addi %g{reg}, %c{j} : index")
            emit(f"%{name} = memref.load %a[%{name}i] : memref<{m*k}x{e}>")
            if integer_bits == 4:
                pred = "s" if dtype == "int8" else "u"
                emit(f"%{name}lo = arith.cmpi {pred}ge, %{name}, %amin : i8")
                emit(f"%{name}hi = arith.cmpi {pred}le, %{name}, %amax : i8")
                emit(f"%{name}range = arith.andi %{name}lo, %{name}hi : i1")
                emit(f"%{name}valid = arith.andi {valid}, %{name}range : i1")
                valid = f"%{name}valid"
            if integer:
                emit(f"%{name}z = arith.cmpi eq, %{name}, %zbits : i8")
            else:
                emit(f"%{name}bits = arith.bitcast %{name} : {e} to {bits}")
                emit(f"%{name}z = arith.cmpi eq, %{name}bits, %zbits : {bits}")
            vals.append("%" + name)
            zeros.append("%" + name + "z")
        first, second, code, found = vals[0], vals[1], "%z32", "%no"
        for p, (i, j) in enumerate(combinations(range(4), 2)):
            other = [x for x in range(4) if x not in (i, j)]
            prefix = f"p{reg}_{p}"
            emit(f"%{prefix}ok = arith.andi {zeros[other[0]]}, {zeros[other[1]]} : i1")
            emit(f"%{prefix}code = arith.constant {(i | j << 2) << (4*reg)} : i32")
            emit(f"%{prefix}x = arith.select %{prefix}ok, {vals[i]}, {first} : {e}")
            emit(f"%{prefix}y = arith.select %{prefix}ok, {vals[j]}, {second} : {e}")
            emit(f"%{prefix}idx = arith.select %{prefix}ok, %{prefix}code, {code} : i32")
            emit(f"%{prefix}found = arith.ori %{prefix}ok, {found} : i1")
            first, second, code, found = (f"%{prefix}{s}" for s in ("x", "y", "idx", "found"))
        avalues.extend((first, second))
        emit(f"%idx{reg} = arith.ori {idx}, {code} : i32")
        emit(f"%valid{reg} = arith.andi {valid}, {found} : i1")
        idx, valid = f"%idx{reg}", f"%valid{reg}"
    emit(f"%av = vector.from_elements {', '.join(avalues)} : vector<8x{e}>")
    bvalues = []
    # Inverse B fragment mapping from the RDNA4 ISA: row = half*8 +
    # floor(register/4)*16 + (register%4)*2 + element half.
    for j in range(16):
        off = (j // 8) * 16 + j % 8
        if off not in constants:
            emit(f"%bo{j} = arith.constant {off} : index")
        c = f"%c{off}" if off in constants else f"%bo{j}"
        emit(f"%br{j}a = arith.addi %kk, %half8 : index")
        emit(f"%br{j} = arith.addi %br{j}a, {c} : index")
        emit(f"%bi{j}a = arith.muli %br{j}, %c{n} : index")
        emit(f"%bi{j} = arith.addi %bi{j}a, %bcol : index")
        emit(f"%b{j} = memref.load %b[%bi{j}] : memref<{k*n}x{be}>")
        if integer_bits == 4:
            pred = "s" if rhs_dtype == "int8" else "u"
            emit(f"%b{j}lo = arith.cmpi {pred}ge, %b{j}, %bmin : i8")
            emit(f"%b{j}hi = arith.cmpi {pred}le, %b{j}, %bmax : i8")
            emit(f"%b{j}range = arith.andi %b{j}lo, %b{j}hi : i1")
            emit(f"%b{j}valid = arith.andi {valid}, %b{j}range : i1")
            valid = f"%b{j}valid"
        bvalues.append(f"%b{j}")
    emit(f"%bv = vector.from_elements {', '.join(bvalues)} : vector<16x{be}>")
    signs = f', a_signed = {str(dtype == "int8").lower()}, b_signed = {str(rhs_dtype == "int8").lower()}' if integer else ""
    av, bv = "%av", "%bv"
    signs += f", integer_bits = {integer_bits} : i64" if integer else ""
    emit(f'%result = schedule.sparse_mma {av}, {bv}, %acc, {idx} {{arch = "gfx1201"{signs}}} : vector<8x{packed}>, vector<16x{packed if integer_bits == 4 else be}>, vector<8x{accum}> -> vector<8x{accum}>')
    emit(f"scf.yield %result, {valid} : vector<8x{accum}>, i1")
    emit("}")
    emit("%orowbase = arith.addi %row0, %half8 : index")
    if accum != output:
        emit(f"%wide = arith.extf %loop#0 : vector<8x{accum}> to vector<8xf32>")
    result = "%loop#0" if accum == output else "%wide"
    for j in range(8):
        emit(f"%orow{j} = arith.addi %orowbase, %c{j} : index")
        emit(f"%oi{j}a = arith.muli %orow{j}, %c{n} : index")
        emit(f"%oi{j} = arith.addi %oi{j}a, %bcol : index")
        emit(f"%o{j} = vector.extract {result}[{j}] : {output} from vector<8x{output}>")
        stored = f"%o{j}"
        if destination != output:
            emit(f"%cast{j} = arith.truncf %o{j} : {output} to {destination}")
            stored = f"%cast{j}"
        emit(f"memref.store {stored}, %out[%oi{j}] : memref<{m*n}x{destination}>")
    emit("%si0 = arith.muli %tile, %c32 : index")
    emit("%si = arith.addi %si0, %lane : index")
    emit("%status_value = arith.select %loop#1, %one32, %z32 : i32")
    emit(f"memref.store %status_value, %status[%si] : memref<{tiles*32}xi32>")
    emit("gpu.return")
    lines.append("    }\n  }\n}\n")
    return "\n".join(lines)
