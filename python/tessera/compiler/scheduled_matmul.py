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
    bias_name: str | None = None
    residual_name: str | None = None
    activation: str = "none"
    dynamic_m: bool = False
    dynamic_n: bool = False
    dynamic_k: bool = False

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
        matmul_kernels = self.tile_ir.count("tile.matmul_kernel")
        macro_cta_kernels = self.tile_ir.count("tessera_nvidia.macro_cta_matmul")
        typed_mmas = len(re.findall(r"(?m)^\s*%[^=]+ = tile\.mma\b", self.tile_ir))
        if self.target == "nvidia_sm120":
            producers = int(bool(matmul_kernels)) + int(bool(typed_mmas)) + int(bool(macro_cta_kernels))
            if producers != 1 or matmul_kernels > 1 or typed_mmas > 1 or macro_cta_kernels > 1:
                raise ValueError(
                    "SM120 scheduled artifact requires exactly one typed MMA, "
                    "macro-CTA, or deferred Tile matmul producer"
                )
        elif matmul_kernels != 1:
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
        if self.activation not in {"none", "relu", "gelu", "silu"}:
            raise ValueError("scheduled matmul artifact has an unsupported activation")
        if self.bias_name is not None and 'bias = true' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its bias epilogue")
        if self.residual_name is not None and 'residual = true' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its residual epilogue")
        if self.activation != "none" and f'activation = "{self.activation}"' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its activation epilogue")
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
        bias_name,
        residual_name,
        activation,
        dynamic_m,
        dynamic_n,
        dynamic_k,
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
        bias_name=bias_name,
        residual_name=residual_name,
        activation=activation,
        dynamic_m=dynamic_m,
        dynamic_n=dynamic_n,
        dynamic_k=dynamic_k,
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
    if op.op_name != "tessera.matmul" or not 2 <= len(op.operands) <= 4:
        raise ValueError("scheduled matmul requires one tessera.matmul")
    if op.kwargs.get("transposeA", False) or op.kwargs.get("transposeB", False):
        raise ValueError("scheduled matmul does not support transpose")
    args = {arg.name: arg for arg in function.args}
    a_name, b_name = (value.removeprefix("%") for value in op.operands[:2])
    if a_name not in args or b_name not in args:
        raise ValueError("scheduled matmul operands must be function arguments")
    def extent(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    a_shape = tuple(extent(value) for value in args[a_name].ir_type.shape)
    b_shape = tuple(extent(value) for value in args[b_name].ir_type.shape)
    out_shape = tuple(extent(value) for value in function.result_types[0].shape)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        raise ValueError("scheduled matmul requires rank-2 tensors")
    m_value, k_value = a_shape
    kb_value, n_value = b_shape
    dynamic_m = m_value is None
    dynamic_n = n_value is None
    dynamic_k = k_value is None
    dynamic = dynamic_m or dynamic_n or dynamic_k or None in b_shape or None in out_shape
    raw_bounds = op.kwargs.get("shape_bounds")
    if dynamic:
        if target not in {"nvidia_sm120", "rocm_gfx1151"} or not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 3:
            raise ValueError(
                "dynamic scheduled matmul requires an SM120 or gfx1151 "
                "shape_bounds=[M,N,K] contract"
            )
        try:
            m, n, k = (int(value) for value in raw_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("scheduled matmul shape_bounds must be positive integers") from exc
    else:
        m, n, k = int(m_value), int(n_value), int(k_value)
    compatible = lambda value, expected: value is None or value == expected
    if (
        min(m, n, k) <= 0
        or not compatible(m_value, m)
        or not compatible(k_value, k)
        or not compatible(kb_value, k)
        or not compatible(n_value, n)
        or len(out_shape) != 2
        or not compatible(out_shape[0], m)
        or not compatible(out_shape[1], n)
    ):
        raise ValueError("scheduled matmul shapes do not form MxK @ KxN -> MxN")
    a_dtype = args[a_name].ir_type.dtype
    b_dtype = args[b_name].ir_type.dtype
    output_dtype = function.result_types[0].dtype
    activation = str(op.kwargs.get("activation", "none"))
    if activation not in {"none", "relu", "gelu", "silu"}:
        raise ValueError("scheduled matmul has an unsupported activation")
    bias_value = op.kwargs.get("bias")
    residual_value = op.kwargs.get("residual")
    bias_name = bias_value.removeprefix("%") if isinstance(bias_value, str) else None
    residual_name = (
        residual_value.removeprefix("%") if isinstance(residual_value, str) else None
    )
    if bias_value not in {None, False} and bias_name is None:
        raise ValueError("scheduled matmul bias must name a Graph argument")
    if residual_value not in {None, False} and residual_name is None:
        raise ValueError("scheduled matmul residual must name a Graph argument")
    expected_operands = [f"%{a_name}", f"%{b_name}"]
    if bias_name is not None:
        expected_operands.append(f"%{bias_name}")
    if residual_name is not None:
        expected_operands.append(f"%{residual_name}")
    if op.operands != expected_operands:
        raise ValueError(
            "scheduled matmul epilogue operands must follow A/B/bias/residual ABI order"
        )
    if bias_name is not None:
        bias = args.get(bias_name)
        if bias is None or tuple(bias.ir_type.shape) != (str(n),) or bias.ir_type.dtype != "fp32":
            raise ValueError("scheduled matmul bias must be an fp32 [N] argument")
    if residual_name is not None:
        residual = args.get(residual_name)
        if (
            residual is None
            or tuple(residual.ir_type.shape) != (str(m), str(n))
            or residual.ir_type.dtype != "fp32"
        ):
            raise ValueError("scheduled matmul residual must be an fp32 [M,N] argument")
    if (
        target != "nvidia_sm120"
        and (bias_name is not None or residual_name is not None or activation != "none")
    ):
        raise ValueError("scheduled fused matmul is currently NVIDIA-owned")
    output_name = op.result
    if not output_name:
        if len(function.return_values) != 1:
            raise ValueError("scheduled matmul requires one named Graph result")
        output_name = function.return_values[0].removeprefix("%")
    function_name = function.name
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
    elif (
        target == "nvidia_sm120"
        and a_dtype == b_dtype
        and a_dtype in {"fp16", "bf16"}
        and output_dtype in {"fp32", "fp16"}
    ):
        # SM120 consumes the same canonical Schedule/Tile boundary as the
        # other native backends.  The NVIDIA package owns the physical
        # m16n8k16 MMA lowering; this contract records only the shared launch
        # envelope and keeps the schedule decision content-addressed.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "nvidia_sm120",
            "sm_120",
            "f16" if a_dtype == "fp16" else "bf16",
            "f32",
            128,
            128,
        )
        # Schedule->Tile replaces the Graph tensor wrapper with this explicit
        # raw-pointer launch entry.  Keeping the suffix here makes the
        # package descriptor name the same canonical kernel that Tile IR
        # carries, rather than a Python-side substitute.
        fused = bias_name is not None or residual_name is not None or activation != "none"
        reduced = output_dtype == "fp16"
        suffix = (
            f"_fused_{storage}_{activation}_b{int(bias_name is not None)}"
            f"_r{int(residual_name is not None)}"
            if fused or reduced
            else ""
        )
        if reduced:
            suffix += "_outf16"
        function_name = f"{function.name}{suffix}" + (
            "_macro_kernel"
            if _uses_sm120_macro_cta(m, n, k, storage, accum)
            else "_kernel"
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
        bias_name,
        residual_name,
        activation,
        dynamic_m,
        dynamic_n,
        dynamic_k,
    )


def _uses_sm120_macro_cta(
    m: int, n: int, k: int, storage: str, accum: str
) -> bool:
    """Mirror the measured target-owned 32x32/four-warp admission contract.

    SuperBear's retained WSL pruning packet found staging/barrier overhead at
    small sizes and timing variance through 33.6M FLOPs.  Every measured
    67.1M+ case is both low-variance and materially faster. WSL evidence is
    deliberately ineligible for the global performance registry, but it is
    sufficient to keep the explicit scheduled route from selecting an
    unproven crossover while bare-metal selector evidence remains open.
    """
    work = 2 * m * n * k
    return (
        m >= 32
        and n >= 32
        and k >= 16
        and storage in {"f16", "bf16"}
        and accum == "f32"
        and work >= 67_108_864
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
