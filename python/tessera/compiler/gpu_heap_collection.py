"""GPU snapshot marking and exclusive sweeping of fixed generation slots.

Numeric or opaque-byte slots are bounded and nonmoving, with up to 32 explicit
reference edges. Snapshot marking reads private copies while the single writer
mutates the active graph; final remark/sweep excludes writers and readers.
State columns are generation, byte/element length, and lifecycle (0 free,
1 live, 2 logically retired). Incremental sweep retires a complete unreachable
cohort before reclaiming a range; retired edges are not traversed and allocation
cannot reuse retired slots. This is not a CPython allocator or concurrent sweeper.
"""

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from .native_gpu_storage import build_native_gpu_storage, _run, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding


def emit_pool(slots, width, mode, *, payload_dtype="fp32", references=2):
    protocol_mode = mode
    atomic = mode.startswith("atomic_") and mode.removeprefix("atomic_") in ("allocate", "allocate_marked", "graph_checked", "graph_incremental", "mark_begin", "mark_step", "retire_marked", "reclaim_pinned", "pin", "unpin", "inspect")
    if atomic:
        mode = mode.removeprefix("atomic_")
    if (
        type(slots) is not int
        or payload_dtype not in ("fp32", "int8")
        or type(references) is not int
        or not 1 <= references <= 32
        or type(width) is not int
        or not 1 <= slots <= 256
        or not 1 <= width
        or slots * width > 262144
        or mode not in ("allocate", "collect", "graph", "mark", "collect_seeded", "collect_slice", "graph_checked", "retire", "reclaim", "pin", "unpin", "allocate_marked", "graph_incremental", "mark_begin", "mark_step", "retire_marked", "reclaim_pinned", "inspect")
    ):
        raise ValueError("GPU heap pool requires bounded slots/width and a known operation")
    specs: list[TensorSpec | IndexSpec] = [
        TensorSpec("state", "int64", (slots, 3), True),
        TensorSpec("roots", "int64", (slots,), True),
        TensorSpec("edges", "int64", (slots, 2 * references), True),
    ]
    if mode in ("allocate", "allocate_marked"):
        specs += [
            TensorSpec("payload", payload_dtype, (slots, width), True),
            TensorSpec("input", payload_dtype, (width,)),
            TensorSpec("status", "int64", (3,), True),
            IndexSpec("length", 0, width),
        ]
    elif mode in ("graph", "graph_checked", "graph_incremental"):
        specs += [
            TensorSpec("root_input", "int64", (slots,)),
            TensorSpec("edge_input", "int64", (slots, 2 * references)),
        ]
        if mode == "graph_incremental":
            specs += [TensorSpec("marks", "int64", (slots,), True)]
        if mode in ("graph_checked", "graph_incremental"):
            specs += [TensorSpec("status", "int64", (3,), True)]
    elif mode in ("pin", "unpin"):
        specs += [TensorSpec("pins", "int64", (slots,), True),
                  TensorSpec("status", "int64", (3,), True),
                  IndexSpec("slot", 0, slots - 1), IndexSpec("generation", 1, (1 << 31) - 1)]
    else:
        specs += [TensorSpec("marks", "int64", (slots,), True), TensorSpec("status", "int64", (3,), True)]
    if mode in ("collect_seeded", "collect_slice"):
        specs += [TensorSpec("seed", "int64", (slots,)), TensorSpec("seed_status", "int64", (3,))]
    if mode == "collect_slice":
        specs += [IndexSpec("sweep_begin", 0, slots), IndexSpec("sweep_end", 0, slots)]
    if mode == "reclaim_pinned":
        specs += [TensorSpec("pins", "int64", (slots,))]
    if mode == "allocate_marked":
        specs += [TensorSpec("marks", "int64", (slots,), True)]
    if mode == "mark_step":
        specs += [IndexSpec("budget", 1, slots)]
    if mode == "inspect":
        specs += [TensorSpec("state_out", "int64", (slots, 3), True),
                  TensorSpec("roots_out", "int64", (slots,), True),
                  TensorSpec("edges_out", "int64", (slots, 2 * references), True),
                  TensorSpec("marks_out", "int64", (slots,), True)]
    if atomic:
        specs += [TensorSpec("gate", "int64", (1,), True)]
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
        ("retired", 2, "i64"),
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

    def shade(index):
        nonlocal lines
        value = load("marks", index)
        tag = f"white{serial}"
        lines += [f"%{tag} = arith.cmpi eq, {value}, %zero : i64", f"scf.if %{tag} {{"]
        store("marks", index, "%unit")
        lines += ["}"]

    def copy_graph():
        nonlocal lines
        lines += ["scf.for %i = %z to %N step %one {", "%ix = arith.index_cast %i : index to i64"]
        value = load("root_input", "%ix")
        if mode == "graph_incremental":
            lines += [f"%hasroot = arith.cmpi ne, {value}, %zero : i64", "scf.if %hasroot {"]
            shade("%ix")
            lines += ["}"]
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
        if mode == "graph_incremental":
            lines += ["%pair = arith.constant 2 : i64", "%part = arith.remui %ji, %pair : i64",
                      "%targetpart = arith.cmpi eq, %part, %zero : i64",
                      f"%hastarget = arith.cmpi ne, {value}, %invalid : i64",
                      "%triple = arith.constant 3 : i64", "%sb = arith.muli %ix, %triple : i64",
                      "%sl = arith.addi %sb, %unit : i64", "%sa = arith.addi %sl, %unit : i64"]
            alive = load("state", "%sa")
            lines += [f"%source_live = arith.cmpi eq, {alive}, %unit : i64",
                      "%target_exists = arith.andi %targetpart, %hastarget : i1",
                      "%shade_target = arith.andi %target_exists, %source_live : i1", "scf.if %shade_target {"]
            shade(value)
            lines += ["}"]
        store("edges", "%offset", value)
        lines += ["}", "}"]

    if atomic:
        lines += ["%reservation = llvm.cmpxchg %gate, %zero, %unit acq_rel acquire : !llvm.ptr<1>, i64",
                  "%acquired = llvm.extractvalue %reservation[1] : !llvm.struct<(i64, i1)>",
                  "scf.if %acquired {"]
    if mode == "inspect":
        for base, count in (("state", slots * 3), ("roots", slots), ("edges", slots * 2 * references), ("marks", slots)):
            lines += [f"%{base}_count = arith.constant {count} : index",
                      f"scf.for %i = %z to %{base}_count step %one {{",
                      "%ix = arith.index_cast %i : index to i64"]
            value = load(base, "%ix")
            store(base + "_out", "%ix", value)
            lines += ["}"]
        store("status", "%zero", "%zero")
    elif mode == "graph":
        copy_graph()
    elif mode in ("pin", "unpin"):
        lines += ["%ix = arith.index_cast %slot : index to i64",
                  "%requested = arith.index_cast %generation : index to i64",
                  "%three = arith.constant 3 : i64", "%base = arith.muli %ix, %three : i64",
                  "%len = arith.addi %base, %unit : i64", "%alive = arith.addi %len, %unit : i64"]
        gen, length, live, pins = load("state", "%base"), load("state", "%len"), load("state", "%alive"), load("pins", "%ix")
        lines += [f"%same = arith.cmpi eq, {gen}, %requested : i64",
                  f"%is_live = arith.cmpi eq, {live}, %unit : i64",
                  f"%is_retired = arith.cmpi eq, {live}, %retired : i64",
                  "%present = arith.ori %is_live, %is_retired : i1",
                  f"%sizeok = arith.cmpi ule, {length}, %W : i64",
                  f"%pinroom = arith.cmpi ult, {pins}, %limit : i64",
                  f"%haspin = arith.cmpi sgt, {pins}, %zero : i64",
                  "%lifetime = arith.andi %same, " + ("%is_live" if mode == "pin" else "%present") + " : i1",
                  "%bounded = arith.andi %sizeok, " + ("%pinroom" if mode == "pin" else "%haspin") + " : i1",
                  "%valid = arith.andi %lifetime, %bounded : i1", "scf.if %valid {"]
        lines += [f"%next = arith.{'addi' if mode == 'pin' else 'subi'} {pins}, %unit : i64"]
        store("pins", "%ix", "%next")
        store("status", "%zero", "%zero")
        store("status", "%unit", length)
        store("status", "%retired", gen)
        lines += ["} else {", "%bad = arith.constant 2 : i64"]
        store("status", "%zero", "%bad")
        lines += ["}"]
    elif mode in ("allocate", "allocate_marked"):
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
        if mode == "allocate_marked":
            store("marks", "%found", "%unit")
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
        root = load("root_input" if mode in ("graph_checked", "graph_incremental") else "roots", "%c")
        length = load("state", "%clen")
        lines += [
            f"%live = arith.cmpi eq, {alive}, %unit : i64",
            f"%dead = arith.cmpi eq, {alive}, %zero : i64",
            f"%pending = arith.cmpi eq, {alive}, %retired : i64",
            "%notlive = arith.ori %pending, %dead : i1",
            "%flag = arith.ori %live, %notlive : i1",
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
        if mode in ("graph_incremental", "mark_step", "retire_marked"):
            mark = load("marks", "%c")
            lines += [f"%markzero = arith.cmpi eq, {mark}, %zero : i64",
                      f"%markblack = arith.cmpi eq, {mark}, %retired : i64",
                      f"%markbounded = arith.cmpi ule, {mark}, %retired : i64"]
            if mode == "retire_marked":
                lines += ["%finished = arith.ori %markzero, %markblack : i1",
                          "%rootmarked = arith.ori %nogcroot, %markblack : i1",
                          "%blacklive = arith.andi %markblack, %live : i1",
                          "%marklive = arith.ori %markzero, %blacklive : i1",
                          "%closedroot = arith.andi %rootmarked, %finished : i1",
                          "%markvalid = arith.andi %closedroot, %marklive : i1"]
            else:
                lines += ["%marklive = arith.ori %markzero, %live : i1",
                          "%markvalid = arith.andi %markbounded, %marklive : i1"]
            lines += ["%markedstart = arith.andi %start, %markvalid : i1"]
            previous = "%markedstart"
        if mode in ("collect_seeded", "collect_slice"):
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
            target = load("edge_input" if mode in ("graph_checked", "graph_incremental") else "edges", f"%e{e}idx")
            gen = load("edge_input" if mode in ("graph_checked", "graph_incremental") else "edges", f"%e{e}genidx")
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
            if mode == "retire_marked":
                target_mark = load("marks", target)
                lines += [f"%targetblack{e} = arith.cmpi eq, {target_mark}, %retired : i64",
                          f"%sourcewhite{e} = arith.xori %markblack, %true : i1",
                          f"%closededge{e} = arith.ori %sourcewhite{e}, %targetblack{e} : i1"]
            lines += [
                f"%teq{e} = arith.cmpi eq, {te}, {gen} : i64",
                f"%talive{e} = arith.cmpi eq, {ta}, %unit : i64",
                f"%edgegood{e} = arith.andi %teq{e}, %talive{e} : i1",
                (f"%edgeclosed{e} = arith.andi %edgegood{e}, %closededge{e} : i1\nscf.yield %edgeclosed{e} : i1"
                 if mode == "retire_marked" else f"scf.yield %edgegood{e} : i1"),
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
        ]
        if mode in ("graph_checked", "graph_incremental"):
            copy_graph()
            store("status", "%unit", "%zero")
        elif mode in ("mark_begin", "mark_step", "retire_marked", "reclaim_pinned"):
            if mode == "mark_begin":
                lines += ["scf.for %i = %z to %N step %one {", "%ix = arith.index_cast %i : index to i64"]
                root = load("roots", "%ix")
                lines += [f"%rooted = arith.cmpi ne, {root}, %zero : i64", "%initial = arith.select %rooted, %unit, %zero : i64"]
                store("marks", "%ix", "%initial")
                lines += ["}"]
            elif mode == "mark_step":
                lines += ["%scanned = scf.for %i = %z to %N step %one iter_args(%count = %z) -> index {",
                          "%ix = arith.index_cast %i : index to i64"]
                mark = load("marks", "%ix")
                lines += [f"%grey = arith.cmpi eq, {mark}, %unit : i64", "%room = arith.cmpi ult, %count, %budget : index",
                          "%visit = arith.andi %grey, %room : i1", "%next = scf.if %visit -> index {"]
                store("marks", "%ix", "%retired")
                lines += [f"%stride = arith.constant {2 * references} : i64", "%eb = arith.muli %ix, %stride : i64"]
                for e in range(references):
                    lines += [f"%off{e} = arith.constant {2 * e} : i64", f"%ei{e} = arith.addi %eb, %off{e} : i64"]
                    target = load("edges", f"%ei{e}")
                    lines += [f"%exists{e} = arith.cmpi ne, {target}, %invalid : i64", f"scf.if %exists{e} {{"]
                    shade(target)
                    lines += ["}"]
                lines += ["%inc = arith.addi %count, %one : index", "scf.yield %inc : index",
                          "} else {scf.yield %count : index}", "scf.yield %next : index", "}",
                          "%scanned64 = arith.index_cast %scanned : index to i64"]
                store("status", "%retired", "%scanned64")
            else:
                lines += ["scf.for %i = %z to %N step %one {"]
                offsets("f")
                alive = load("state", "%falive")
                if mode == "retire_marked":
                    mark = load("marks", "%f")
                    lines += [f"%white = arith.cmpi eq, {mark}, %zero : i64",
                              f"%live = arith.cmpi eq, {alive}, %unit : i64",
                              "%change = arith.andi %white, %live : i1"]
                else:
                    pins = load("pins", "%f")
                    lines += [f"%unpinned = arith.cmpi eq, {pins}, %zero : i64",
                              f"%dead = arith.cmpi eq, {alive}, %retired : i64",
                              "%change = arith.andi %unpinned, %dead : i1"]
                lines += ["scf.if %change {"]
                store("state", "%falive", "%retired" if mode == "retire_marked" else "%zero")
                lines += ["}", "}"]
            if mode == "mark_step":
                lines += ["%remaining = scf.for %i = %z to %N step %one iter_args(%count = %zero) -> i64 {",
                          "%ix = arith.index_cast %i : index to i64"]
                mark = load("marks", "%ix")
                lines += [f"%grey = arith.cmpi eq, {mark}, %unit : i64", "%delta = arith.select %grey, %unit, %zero : i64",
                          "%next = arith.addi %count, %delta : i64", "scf.yield %next : i64", "}"]
                store("status", "%unit", "%remaining")
            else:
                store("status", "%unit", "%zero")
        else:
            lines += ["scf.for %i = %z to %N step %one {", "%ix = arith.index_cast %i : index to i64"]
            r = load("roots", "%ix")
            if mode in ("collect_seeded", "collect_slice"):
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
            lines += ["}", "}", "}"]
            if mode == "collect_slice":
                # Logically retire the entire unreachable cohort before reclaiming
                # individual slots. Otherwise an unswept dead cycle would contain
                # dangling edges into an earlier batch and fail the next remark.
                lines += ["scf.for %i = %z to %N step %one {"]
                offsets("r")
                alive = load("state", "%ralive")
                marked = load("marks", "%r")
                lines += [f"%r_live = arith.cmpi eq, {alive}, %unit : i64",
                          f"%r_unmarked = arith.cmpi eq, {marked}, %zero : i64",
                          "%r_dead = arith.andi %r_live, %r_unmarked : i1",
                          "scf.if %r_dead {"]
                store("state", "%ralive", "%retired")
                lines += ["}", "}"]
            lines += ["%reclaimed = scf.for %i = %z to %N step %one iter_args(%freed = %zero) -> i64 {"]
            offsets("s")
            alive = load("state", "%salive")
            marked = load("marks", "%s")
            lines += [
                (f"%live = arith.cmpi eq, {alive}, %unit : i64" if mode == "retire" else
                 f"%live = arith.cmpi ne, {alive}, %zero : i64"),
                f"%unmarked = arith.cmpi eq, {marked}, %zero : i64",
                ("%collect = arith.constant false" if mode == "mark" else
                 f"%collect = arith.cmpi eq, {alive}, %retired : i64" if mode == "reclaim" else
                 "%collect = arith.andi %live, %unmarked : i1"),
                "%next = scf.if %collect -> i64 {",
            ]
            if mode == "collect_slice":
                lines += ["%after_begin = arith.cmpi uge, %i, %sweep_begin : index",
                          "%before_end = arith.cmpi ult, %i, %sweep_end : index",
                          "%in_slice = arith.andi %after_begin, %before_end : i1",
                          "scf.if %in_slice {"]
            store("state", "%salive", "%retired" if mode == "retire" else "%zero")
            if mode == "collect_slice":
                lines += ["}"]
            lines += [
                ("%delta = arith.select %in_slice, %unit, %zero : i64" if mode == "collect_slice" else "%delta = arith.constant 1 : i64"),
                "%increment = arith.addi %freed, %delta : i64",
                "scf.yield %increment : i64",
                "} else {scf.yield %freed : i64}",
                "scf.yield %next : i64",
                "}",
            ]
            store("status", "%unit", "%reclaimed")
        store("status", "%zero", "%zero")
        lines += ["} else {", "%bad = arith.constant 2 : i64"]
        store("status", "%zero", "%bad")
        store("status", "%unit", "%zero")
        lines += ["}"]
    if atomic:
        lines += ["%released = llvm.atomicrmw xchg %gate, %zero release : !llvm.ptr<1>, i64",
                  "} else {", "%busy = arith.constant 3 : i64"]
        store("status", "%zero", "%busy")
        lines += ["}"]
    lines += ["gpu.return", "}", "}", "}"]
    from .heap_barrier_contract import attach_heap_contract
    source = attach_tensor_contract("\n".join(lines), specs, grid=(1, 1, 1), block=(1, 1, 1))
    return attach_heap_contract(source, slots, width, references, protocol_mode), tuple(specs)


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
        from .heap_barrier_contract import read_heap_contract
        contract = read_heap_contract(self.package.arena_ir)
        if (contract["slots"], contract["width"], contract["references"], contract["mode"]) != (
            self.slots, self.width, self.references, self.mode
        ):
            raise ValueError("GPU heap pool disagrees with protocol replay")
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
