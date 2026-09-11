"""GPU snapshot marking and exclusive sweeping of fixed generation slots.

Numeric or opaque-byte slots are bounded and nonmoving, with up to 32 explicit
reference edges. Snapshot marking reads private copies while the single writer
mutates the active graph; final remark/sweep excludes writers and readers.
This is not a CPython allocator or concurrent sweeping collector.
"""

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from .native_gpu_storage import build_native_gpu_storage, _run, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def emit_pool(slots, width, mode, *, payload_dtype="fp32", references=2):
    if (
        type(slots) is not int
        or payload_dtype not in ("fp32", "int8")
        or type(references) is not int
        or not 1 <= references <= 32
        or type(width) is not int
        or not 1 <= slots <= 256
        or not 1 <= width
        or slots * width > 262144
        or mode not in ("allocate", "collect", "graph", "mark", "collect_seeded")
    ):
        raise ValueError("GPU heap pool requires bounded slots/width and a known operation")
    specs: list[TensorSpec | IndexSpec] = [
        TensorSpec("state", "int64", (slots, 3), True),
        TensorSpec("roots", "int64", (slots,), True),
        TensorSpec("edges", "int64", (slots, 2 * references), True),
    ]
    if mode == "allocate":
        specs += [
            TensorSpec("payload", payload_dtype, (slots, width), True),
            TensorSpec("input", payload_dtype, (width,)),
            TensorSpec("status", "int64", (3,), True),
            IndexSpec("length", 0, width),
        ]
    elif mode == "graph":
        specs += [
            TensorSpec("root_input", "int64", (slots,)),
            TensorSpec("edge_input", "int64", (slots, 2 * references)),
        ]
    else:
        specs += [TensorSpec("marks", "int64", (slots,), True), TensorSpec("status", "int64", (3,), True)]
    if mode == "collect_seeded":
        specs += [TensorSpec("seed", "int64", (slots,)), TensorSpec("seed_status", "int64", (3,))]
    specs += [IndexSpec("scratch", 1, 1)]
    args = ", ".join("%" + s.name + ": " + ("!llvm.ptr<1>" if isinstance(s, TensorSpec) else "index") for s in specs)
    lines = [
        "module {",
        "gpu.module @native_tape {",
        f"gpu.func @product({args}) kernel attributes {{known_block_size = array<i32: 1, 1, 1>}} {{",
        "%marker = memref.alloca(%scratch) : memref<?xf32>",
        '"tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()',
    ]
    for name, value, ty in [
        ("z", 0, "index"),
        ("one", 1, "index"),
        ("N", slots, "index"),
        ("W", width, "i64"),
        ("zero", 0, "i64"),
        ("unit", 1, "i64"),
        ("invalid", -1, "i64"),
        ("limit", (1 << 31) - 1, "i64"),
    ]:
        lines.append(f"%{name} = arith.constant {value} : {ty}")
    lines += ["%true = arith.constant true", "%false = arith.constant false"]
    serial = 0

    def load(base, index, ty="i64"):
        nonlocal serial
        n = f"v{serial}"
        serial += 1
        lines.extend(
            [
                f"%{n}p = llvm.getelementptr %{base}[{index}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {ty}",
                f"%{n} = llvm.load %{n}p : !llvm.ptr<1> -> {ty}",
            ]
        )
        return "%" + n

    def store(base, index, value, ty="i64"):
        nonlocal serial
        n = f"p{serial}"
        serial += 1
        lines.extend(
            [
                f"%{n} = llvm.getelementptr %{base}[{index}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {ty}",
                f"llvm.store {value}, %{n} : {ty}, !llvm.ptr<1>",
            ]
        )

    def offsets(prefix):
        lines.extend(
            [
                f"%{prefix} = arith.index_cast %i : index to i64",
                f"%{prefix}3 = arith.constant 3 : i64",
                f"%{prefix}4 = arith.constant {2 * references} : i64",
                f"%{prefix}base = arith.muli %{prefix}, %{prefix}3 : i64",
                f"%{prefix}len = arith.addi %{prefix}base, %unit : i64",
                f"%{prefix}alive = arith.addi %{prefix}len, %unit : i64",
                f"%{prefix}edge = arith.muli %{prefix}, %{prefix}4 : i64",
            ]
        )

    if mode == "graph":
        lines += ["scf.for %i = %z to %N step %one {", "%ix = arith.index_cast %i : index to i64"]
        value = load("root_input", "%ix")
        store("roots", "%ix", value)
        lines += [
            f"%edgecount = arith.constant {2 * references} : index",
            "scf.for %j = %z to %edgecount step %one {",
            "%ji = arith.index_cast %j : index to i64",
            f"%stride = arith.constant {2 * references} : i64",
            "%base = arith.muli %ix, %stride : i64",
            "%offset = arith.addi %base, %ji : i64",
        ]
        value = load("edge_input", "%offset")
        store("edges", "%offset", value)
        lines += ["}", "}"]
    elif mode == "allocate":
        lines += ["%found = scf.for %i = %z to %N step %one iter_args(%slot = %invalid) -> i64 {"]
        offsets("a")
        epoch = load("state", "%abase")
        alive = load("state", "%aalive")
        lines += [
            f"%free = arith.cmpi eq, {alive}, %zero : i64",
            f"%room = arith.cmpi ult, {epoch}, %limit : i64",
            "%unused = arith.cmpi eq, %slot, %invalid : i64",
            "%both = arith.andi %free, %room : i1",
            "%take = arith.andi %both, %unused : i1",
            "%next = arith.select %take, %a, %slot : i64",
            "scf.yield %next : i64",
            "}",
            "%has = arith.cmpi ne, %found, %invalid : i64",
            "scf.if %has {",
            "%three = arith.constant 3 : i64",
            f"%four = arith.constant {2 * references} : i64",
            "%base = arith.muli %found, %three : i64",
            "%lenptr = arith.addi %base, %unit : i64",
            "%aliveptr = arith.addi %lenptr, %unit : i64",
        ]
        epoch = load("state", "%base")
        lines += [
            f"%generation = arith.addi {epoch}, %unit : i64",
            "%payloadbase = arith.muli %found, %W : i64",
            "scf.for %j = %z to %length step %one {",
            "%ji = arith.index_cast %j : index to i64",
            "%destination = arith.addi %payloadbase, %ji : i64",
        ]
        value = load("input", "%ji", "f32" if payload_dtype == "fp32" else "i8")
        store("payload", "%destination", value, "f32" if payload_dtype == "fp32" else "i8")
        lines += ["}", "%size = arith.index_cast %length : index to i64"]
        store("state", "%base", "%generation")
        store("state", "%lenptr", "%size")
        store("roots", "%found", "%generation")
        lines += ["%edgebase = arith.muli %found, %four : i64"]
        for offset in range(2 * references):
            lines += [
                f"%eo{offset} = arith.constant {offset} : i64",
                f"%ei{offset} = arith.addi %edgebase, %eo{offset} : i64",
            ]
            store("edges", f"%ei{offset}", "%invalid" if offset % 2 == 0 else "%zero")
        store("state", "%aliveptr", "%unit")
        store("status", "%zero", "%zero")
        store("status", "%unit", "%found")
        lines += ["%two = arith.constant 2 : i64"]
        store("status", "%two", "%generation")
        lines += ["} else {"]
        store("status", "%zero", "%unit")
        store("status", "%unit", "%invalid")
        lines += ["}"]
    else:
        lines += ["%valid = scf.for %i = %z to %N step %one iter_args(%ok = %true) -> i1 {"]
        offsets("c")
        epoch = load("state", "%cbase")
        alive = load("state", "%calive")
        root = load("roots", "%c")
        length = load("state", "%clen")
        lines += [
            f"%live = arith.cmpi eq, {alive}, %unit : i64",
            f"%dead = arith.cmpi eq, {alive}, %zero : i64",
            "%flag = arith.ori %live, %dead : i1",
            f"%nogcroot = arith.cmpi eq, {root}, %zero : i64",
            f"%rootepoch = arith.cmpi eq, {root}, {epoch} : i64",
            "%liveroot = arith.andi %rootepoch, %live : i1",
            "%rootok = arith.ori %nogcroot, %liveroot : i1",
            f"%sizeok = arith.cmpi ule, {length}, %W : i64",
            f"%epochpositive = arith.cmpi sgt, {epoch}, %zero : i64",
            f"%epochbounded = arith.cmpi ule, {epoch}, %limit : i64",
            "%epochvalid = arith.andi %epochpositive, %epochbounded : i1",
            "%epochok = arith.ori %dead, %epochvalid : i1",
            "%flagsize = arith.andi %flag, %sizeok : i1",
            "%recordok = arith.andi %flagsize, %epochok : i1",
            "%start = arith.andi %recordok, %rootok : i1",
        ]
        previous = "%start"
        if mode == "collect_seeded":
            status = load("seed_status", "%zero")
            seed = load("seed", "%c")
            lines += [
                f"%seedok = arith.cmpi eq, {status}, %zero : i64",
                f"%seedzero = arith.cmpi eq, {seed}, %zero : i64",
                f"%seedone = arith.cmpi eq, {seed}, %unit : i64",
                "%seedlive = arith.andi %seedone, %live : i1",
                "%seedvalid = arith.ori %seedzero, %seedlive : i1",
                "%seedstatus = arith.andi %seedvalid, %seedok : i1",
                "%seedstart = arith.andi %start, %seedstatus : i1",
            ]
            previous = "%seedstart"
        for e in range(references):
            lines += [
                f"%e{e}off = arith.constant {2 * e} : i64",
                f"%e{e}idx = arith.addi %cedge, %e{e}off : i64",
                f"%e{e}genidx = arith.addi %e{e}idx, %unit : i64",
            ]
            target = load("edges", f"%e{e}idx")
            gen = load("edges", f"%e{e}genidx")
            lines += [
                f"%edge{e}ok = scf.if %live -> i1 {{",
                f"%empty{e} = arith.cmpi eq, {target}, %invalid : i64",
                f"%checked{e} = scf.if %empty{e} -> i1 {{",
                f"%emptygen{e} = arith.cmpi eq, {gen}, %zero : i64",
                f"scf.yield %emptygen{e} : i1",
                "} else {",
                f"%bound{e} = arith.constant {slots} : i64",
                f"%inside{e} = arith.cmpi ult, {target}, %bound{e} : i64",
                f"%targetok{e} = scf.if %inside{e} -> i1 {{",
                f"%targetbase{e} = arith.muli {target}, %c3 : i64",
                f"%targetlen{e} = arith.addi %targetbase{e}, %unit : i64",
                f"%targetalive{e} = arith.addi %targetlen{e}, %unit : i64",
            ]
            te = load("state", f"%targetbase{e}")
            ta = load("state", f"%targetalive{e}")
            lines += [
                f"%teq{e} = arith.cmpi eq, {te}, {gen} : i64",
                f"%talive{e} = arith.cmpi eq, {ta}, %unit : i64",
                f"%edgegood{e} = arith.andi %teq{e}, %talive{e} : i1",
                f"scf.yield %edgegood{e} : i1",
                "} else {scf.yield %false : i1}",
                f"scf.yield %targetok{e} : i1",
                "}",
                f"scf.yield %checked{e} : i1",
                "} else {scf.yield %true : i1}",
                f"%joined{e} = arith.andi {previous}, %edge{e}ok : i1",
            ]
            previous = f"%joined{e}"
        lines += [
            f"%allok = arith.andi %ok, {previous} : i1",
            "scf.yield %allok : i1",
            "}",
            "scf.if %valid {",
            "scf.for %i = %z to %N step %one {",
            "%ix = arith.index_cast %i : index to i64",
        ]
        r = load("roots", "%ix")
        if mode == "collect_seeded":
            prior = load("seed", "%ix")
            lines += [f"%union = arith.ori {r}, {prior} : i64"]
            r = "%union"
        lines += [f"%rooted = arith.cmpi ne, {r}, %zero : i64", "%mark = arith.select %rooted, %unit, %zero : i64"]
        store("marks", "%ix", "%mark")
        lines += [
            "}",
            "scf.for %round = %z to %N step %one {",
            "scf.for %i = %z to %N step %one {",
            "%ix = arith.index_cast %i : index to i64",
        ]
        m = load("marks", "%ix")
        lines += [
            f"%marked = arith.cmpi ne, {m}, %zero : i64",
            "scf.if %marked {",
            f"%four = arith.constant {2 * references} : i64",
            "%eb = arith.muli %ix, %four : i64",
        ]
        for e in range(references):
            lines += [f"%off{e} = arith.constant {e * 2} : i64", f"%ei{e} = arith.addi %eb, %off{e} : i64"]
            target = load("edges", f"%ei{e}")
            lines += [f"%exists{e} = arith.cmpi ne, {target}, %invalid : i64", f"scf.if %exists{e} {{"]
            store("marks", target, "%unit")
            lines += ["}"]
        lines += ["}", "}", "}", "%reclaimed = scf.for %i = %z to %N step %one iter_args(%freed = %zero) -> i64 {"]
        offsets("s")
        alive = load("state", "%salive")
        marked = load("marks", "%s")
        lines += [
            f"%live = arith.cmpi eq, {alive}, %unit : i64",
            f"%unmarked = arith.cmpi eq, {marked}, %zero : i64",
            "%collect = arith.constant false" if mode == "mark" else "%collect = arith.andi %live, %unmarked : i1",
            "%next = scf.if %collect -> i64 {",
        ]
        store("state", "%salive", "%zero")
        lines += [
            "%increment = arith.addi %freed, %unit : i64",
            "scf.yield %increment : i64",
            "} else {scf.yield %freed : i64}",
            "scf.yield %next : i64",
            "}",
        ]
        store("status", "%zero", "%zero")
        store("status", "%unit", "%reclaimed")
        lines += ["} else {", "%bad = arith.constant 2 : i64"]
        store("status", "%zero", "%bad")
        store("status", "%unit", "%zero")
        lines += ["}"]
    lines += ["gpu.return", "}", "}", "}"]
    return attach_tensor_contract("\n".join(lines), specs, grid=(1, 1, 1), block=(1, 1, 1)), tuple(specs)


@dataclass(frozen=True)
class GPUHeapPoolKernel:
    slots: int
    width: int
    mode: str
    compiler: Path
    package: NativeGPUStoragePackage
    payload_dtype: str = "fp32"
    references: int = 2

    def validate(self):
        self.package.validate()
        if hashlib.sha256(self.compiler.read_bytes()).hexdigest() != self.package.compiler_digest:
            raise ValueError("GPU pool compiler identity changed")
        source, specs = emit_pool(
            self.slots, self.width, self.mode, payload_dtype=self.payload_dtype, references=self.references
        )
        replay = _run(
            self.compiler,
            "--allow-unregistered-dialect",
            "--tessera-tile-buffer-reuse",
            "--tessera-tile-buffer-arena",
            "--canonicalize",
            source=source,
        )
        if replay != self.package.arena_ir:
            raise ValueError("GPU heap pool disagrees with native replay")
        return specs

    def bind(self):
        specs = self.validate()
        return generate_tensor_binding(
            self.package,
            inspect.Signature([inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs]),
        )


def materialize_pool(slots, width, mode, *, compiler, llvm_bin, backend, chip, payload_dtype="fp32", references=2):
    source, _ = emit_pool(slots, width, mode, payload_dtype=payload_dtype, references=references)
    package = build_native_gpu_storage(
        source, compiler=Path(compiler), llvm_bin=Path(llvm_bin), backend=backend, chip=chip
    )
    result = GPUHeapPoolKernel(slots, width, mode, Path(compiler), package, payload_dtype, references)
    result.validate()
    return result
