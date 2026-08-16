"""Canonical Graph -> Schedule -> launch-Tile handoff for E2E-REAL-3."""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .graph_ir import GraphIRModule


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


@dataclass(frozen=True)
class ScheduledMatmulArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    a_name: str
    b_name: str
    output_name: str
    m: int
    n: int
    k: int
    a_dtype: str
    b_dtype: str
    output_dtype: str
    storage: str
    accum: str
    macro_tile_m: int
    macro_tile_n: int
    schedule_digest: str

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        if len(re.findall(r"(?m)^\s*%[^=]+ = schedule\.matmul\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled matmul artifact requires one scheduled SSA operation")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled matmul artifact requires one durable schedule record")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("scheduled matmul artifact has incomplete Schedule digest identity")
        if self.tile_ir.count("tile.matmul_kernel") != 1:
            raise ValueError("scheduled matmul artifact requires exactly one Tile launch op")
        if "tessera.matmul" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("scheduled matmul Tile artifact must not retain Graph or Schedule ops")
        hashes = _HASH_RE.findall(self.tile_ir)
        if hashes != [self.schedule_digest]:
            raise ValueError("scheduled matmul Tile artifact has a stale schedule digest")
        for name, value in (
            ("tessera.macro_tile_m", self.macro_tile_m),
            ("tessera.macro_tile_n", self.macro_tile_n),
        ):
            if not re.search(rf"{re.escape(name)} = {value} : i64", self.tile_ir):
                raise ValueError(f"scheduled matmul Tile artifact has stale {name}")
        if digest_text(self.schedule_ir) == self.tile_digest:
            raise ValueError("Schedule and Tile artifacts must be distinct boundary outputs")


def lower_scheduled_matmul(
    module: GraphIRModule,
    *,
    target: str,
) -> ScheduledMatmulArtifact:
    """Lower one bounded Graph matmul through the production C++ boundaries."""

    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled matmul lowering requires production tessera-opt")

    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled matmul lowering did not preserve one schedule digest")

    (
        compiler_target,
        architecture,
        function_name,
        a_name,
        b_name,
        output_name,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        output_dtype,
        storage,
        accum,
        macro_tile_m,
        macro_tile_n,
    ) = contract
    artifact = ScheduledMatmulArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=compiler_target,
        architecture=architecture,
        function_name=function_name,
        a_name=a_name,
        b_name=b_name,
        output_name=output_name,
        m=m,
        n=n,
        k=k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        output_dtype=output_dtype,
        storage=storage,
        accum=accum,
        macro_tile_m=macro_tile_m,
        macro_tile_n=macro_tile_n,
        schedule_digest=hashes[0],
    )
    artifact.validate()
    return artifact


def supports_scheduled_matmul(module: GraphIRModule, *, target: str) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if len(module.functions) != 1:
        raise ValueError("scheduled matmul requires one Graph function")
    function = module.functions[0]
    if len(function.body) != 1 or len(function.result_types) != 1:
        raise ValueError("scheduled matmul requires one Graph operation and result")
    op = function.body[0]
    if op.op_name != "tessera.matmul" or len(op.operands) != 2:
        raise ValueError("scheduled matmul requires one tessera.matmul")
    if op.kwargs.get("transposeA", False) or op.kwargs.get("transposeB", False):
        raise ValueError("scheduled matmul does not support transpose")
    args = {arg.name: arg for arg in function.args}
    a_name, b_name = (value.removeprefix("%") for value in op.operands)
    if a_name not in args or b_name not in args:
        raise ValueError("scheduled matmul operands must be function arguments")
    try:
        a_shape = tuple(int(value) for value in args[a_name].ir_type.shape)
        b_shape = tuple(int(value) for value in args[b_name].ir_type.shape)
        out_shape = tuple(int(value) for value in function.result_types[0].shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduled matmul requires static shapes") from exc
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        raise ValueError("scheduled matmul requires rank-2 tensors")
    m, k = a_shape
    kb, n = b_shape
    if min(m, n, k) <= 0 or kb != k or out_shape != (m, n):
        raise ValueError("scheduled matmul shapes do not form MxK @ KxN -> MxN")
    a_dtype = args[a_name].ir_type.dtype
    b_dtype = args[b_name].ir_type.dtype
    output_dtype = function.result_types[0].dtype
    output_name = op.result
    if not output_name:
        if len(function.return_values) != 1:
            raise ValueError("scheduled matmul requires one named Graph result")
        output_name = function.return_values[0].removeprefix("%")
    if target == "x86" and (a_dtype, b_dtype, output_dtype) == ("fp32", "fp32", "fp32"):
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "x86",
            "zen5-avx512",
            "f32",
            "f32",
            16,
            16,
        )
    elif target == "rocm_gfx1151" and (a_dtype, b_dtype, output_dtype) == (
        "fp16",
        "fp16",
        "fp32",
    ):
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "rocm",
            "gfx1151",
            "f16",
            "f32",
            32,
            64,
        )
    elif target == "apple_gpu" and (a_dtype, b_dtype, output_dtype) == (
        "fp32",
        "fp32",
        "fp32",
    ):
        # Apple GPU has no rank-2 f32 cooperative-matrix GEMM; the shared launch
        # contract is consumed as a batch-1 MPS BMM.  The 16x16 macro-tile is a
        # logical default that the Apple package records as an explicit drop,
        # matching the C++ getMatmulSchedule default for this target.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "apple_gpu",
            "apple7",
            "f32",
            "f32",
            16,
            16,
        )
    elif target == "apple_gpu" and (a_dtype, b_dtype, output_dtype) == (
        "fp16",
        "fp16",
        "fp32",
    ):
        # Apple7+ simdgroup_matrix GEMM: f16 storage, f32 accumulation.  This is
        # a compiler-emitted MSL route (not delegated MPS), so the 32x32 macro
        # tile is honored by the emitter.  Matches the C++ getMatmulSchedule
        # apple7 f16 branch.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "apple_gpu",
            "apple7",
            "f16",
            "f32",
            32,
            32,
        )
    else:
        raise ValueError("unsupported target dtype contract for scheduled matmul")
    return (
        compiler_target,
        architecture,
        function.name,
        a_name,
        b_name,
        output_name,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        output_dtype,
        storage,
        accum,
        macro_tile_m,
        macro_tile_n,
    )


def find_tessera_opt() -> Path | None:
    for name in ("TESSERA_OPT", "TESSERA_OPT_BIN"):
        if configured := os.environ.get(name):
            path = Path(configured).expanduser()
            return path if path.is_file() else None
    root = Path(__file__).resolve().parents[3]
    if selected_build := os.environ.get("TESSERA_BUILD_DIR"):
        build = Path(selected_build).expanduser()
        if not build.is_absolute():
            build = root / build
        path = build / "tools/tessera-opt/tessera-opt"
        return path if path.is_file() else None
    for path in (
        root / "build/tools/tessera-opt/tessera-opt",
        root / "build-rocm-ci-local/tools/tessera-opt/tessera-opt",
    ):
        if path.is_file():
            return path
    found = shutil.which("tessera-opt")
    return Path(found) if found else None


def run_tessera_opt(tool: Path, source: str, option: str) -> str:
    result = subprocess.run(
        [str(tool), "-", option],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"scheduled compiler boundary {option} failed: "
            + (result.stderr.strip() or str(result.returncode))
        )
    return result.stdout


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
