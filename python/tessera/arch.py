"""
tessera.arch - Differentiable architecture search foundation.

This module provides small, deterministic Python-side building blocks that mirror
the planned Graph IR and Schedule IR DNAS concepts. They are intentionally light:
full autodiff-aware lowering is a later compiler feature, but these objects give
tests, docs, and examples one stable surface to converge on.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class ArchitectureSearchError(ValueError):
    """Raised when an architecture-search object is malformed."""


@dataclass
class Parameter:
    """Architecture parameter vector, kept separate from model weights."""

    size: int
    init: float = 0.0
    name: str = "alpha"
    dtype: str = "fp32"
    kind: str = "arch"
    values: List[float] = field(init=False)
    grad: List[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ArchitectureSearchError("architecture Parameter size must be > 0")
        if self.dtype != "fp32":
            raise ArchitectureSearchError("architecture parameters must remain fp32")
        if self.kind not in ("arch", "weight", "cost"):
            raise ArchitectureSearchError("parameter kind must be arch, weight, or cost")
        self.values = [float(self.init) for _ in range(self.size)]
        self.grad = [0.0 for _ in range(self.size)]

    def set(self, values: Sequence[float]) -> None:
        if len(values) != self.size:
            raise ArchitectureSearchError(
                f"expected {self.size} architecture logits, got {len(values)}"
            )
        self.values = [float(v) for v in values]

    def argmax(self) -> int:
        return max(range(self.size), key=lambda i: self.values[i])

    def zero_grad(self) -> None:
        self.grad = [0.0 for _ in range(self.size)]

    def set_grad(self, grad: Sequence[float]) -> None:
        if len(grad) != self.size:
            raise ArchitectureSearchError(
                f"expected {self.size} gradient values, got {len(grad)}"
            )
        self.grad = [float(g) for g in grad]

    def step(self, lr: float, *, clip_norm: Optional[float] = None) -> None:
        if lr < 0.0:
            raise ArchitectureSearchError("learning rate must be >= 0")
        grad = list(self.grad)
        if clip_norm is not None:
            norm = math.sqrt(sum(g * g for g in grad))
            if norm > clip_norm > 0.0:
                scale = clip_norm / norm
                grad = [g * scale for g in grad]
        self.values = [v - lr * g for v, g in zip(self.values, grad)]


def collect_parameters(obj: Any, *, kind: Optional[str] = None) -> List[Parameter]:
    """Recursively collect architecture/search parameters from Python objects."""

    found: List[Parameter] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(value, Parameter):
            if kind is None or value.kind == kind:
                found.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    visit(obj)
    return found


def arch_parameters(obj: Any) -> List[Parameter]:
    return collect_parameters(obj, kind="arch")


def validate_backward_wrt(wrt: str) -> str:
    """Validate autodiff partition names used by ``backward(wrt=...)``."""

    if wrt not in ("weights", "arch", "cost", "all"):
        raise ArchitectureSearchError(
            "backward(wrt=...) must be one of weights, arch, cost, or all"
        )
    return wrt


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0.0:
        raise ArchitectureSearchError("temperature must be > 0")
    scaled = [float(x) / temperature for x in logits]
    m = max(scaled)
    exp = [math.exp(x - m) for x in scaled]
    denom = sum(exp)
    return [x / denom for x in exp]


class Softmax:
    """Differentiable softmax relaxation over architecture logits."""

    def __init__(self, alpha: Parameter, *, temperature: float = 1.0) -> None:
        self.alpha = alpha
        self.temperature = float(temperature)

    def __call__(self) -> List[float]:
        return _softmax(self.alpha.values, self.temperature)


class GumbelSoftmax(Softmax):
    """Seedable Gumbel-Softmax relaxation."""

    def __init__(
        self,
        alpha: Parameter,
        *,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(alpha, temperature=temperature)
        self.seed = seed

    def __call__(self) -> List[float]:
        rng = random.Random(self.seed)
        noisy = []
        for value in self.alpha.values:
            u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
            g = -math.log(-math.log(u))
            noisy.append(value + g)
        return _softmax(noisy, self.temperature)


class STEOneHot:
    """Straight-through one-hot surface for discrete specialization tests."""

    def __init__(self, alpha: Parameter) -> None:
        self.alpha = alpha

    def __call__(self) -> List[float]:
        idx = self.alpha.argmax()
        return [1.0 if i == idx else 0.0 for i in range(self.alpha.size)]


class HardConcrete:
    """Hard-concrete style gate expectation used for sparse edge penalties."""

    def __init__(self, alpha: Parameter, *, temperature: float = 2.0) -> None:
        self.alpha = alpha
        self.temperature = float(temperature)

    def __call__(self) -> List[float]:
        return [
            min(1.0, max(0.0, 1.2 * p - 0.1))
            for p in _softmax(self.alpha.values, self.temperature)
        ]


def weighted_sum(values: Sequence[float], gate: Sequence[float]) -> float:
    """Weighted sum helper for scalar candidate outputs."""

    if len(values) != len(gate):
        raise ArchitectureSearchError("weighted_sum requires matching value/gate lengths")
    return sum(float(v) * float(g) for v, g in zip(values, gate))


def switch(candidates: Sequence[Any], gate: Sequence[float], *, hard: bool = False) -> Any:
    """Return either a hard candidate or a soft `(candidate, weight)` list."""

    if len(candidates) != len(gate):
        raise ArchitectureSearchError("switch requires matching candidate/gate lengths")
    if hard:
        idx = max(range(len(gate)), key=lambda i: gate[i])
        return candidates[idx]
    return list(zip(candidates, [float(g) for g in gate]))


@dataclass
class MixedOp:
    """Search-space node containing candidate ops and architecture logits."""

    candidates: Sequence[Any]
    relax: str = "gumbel"
    temperature: float = 5.0
    name: str = "mixed_op"
    seed: Optional[int] = None
    alpha: Parameter = field(init=False)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ArchitectureSearchError("MixedOp requires at least one candidate")
        self.alpha = Parameter(len(self.candidates), name=f"{self.name}.alpha")

    def gates(self) -> List[float]:
        if self.relax == "gumbel":
            return GumbelSoftmax(self.alpha, temperature=self.temperature, seed=self.seed)()
        if self.relax == "softmax":
            return Softmax(self.alpha, temperature=self.temperature)()
        if self.relax == "hard_concrete":
            return HardConcrete(self.alpha, temperature=self.temperature)()
        if self.relax == "ste":
            return STEOneHot(self.alpha)()
        raise ArchitectureSearchError(f"unknown relaxation {self.relax!r}")

    def choice(self) -> int:
        return self.alpha.argmax()


@dataclass(frozen=True)
class CostFeatures:
    """Feature vector used by analytical and learned hardware-cost models."""

    flops: float
    bytes_moved: float
    params: float = 0.0
    tiles: float = 1.0
    seq_len: float = 1.0
    sm_arch: str = "auto"
    bandwidth_gbps: float = 2000.0
    peak_tflops: float = 312.0


@dataclass(frozen=True)
class HardwareCost:
    latency_ms: float
    energy: float
    memory_bytes: float

    def __iter__(self):
        yield self.latency_ms
        yield self.energy
        yield self.memory_bytes


class AnalyticalCostModel:
    """Fast differentiable-style proxy for latency, energy, and memory."""

    def predict(self, features: CostFeatures) -> HardwareCost:
        if features.bandwidth_gbps <= 0 or features.peak_tflops <= 0:
            raise ArchitectureSearchError("hardware feature bandwidth/peak must be > 0")
        compute_ms = features.flops / (features.peak_tflops * 1e12) * 1e3
        memory_ms = features.bytes_moved / (features.bandwidth_gbps * 1e9) * 1e3
        latency = max(compute_ms, memory_ms)
        energy = 0.5 * compute_ms + 0.2 * memory_ms
        return HardwareCost(
            latency_ms=latency,
            energy=energy,
            memory_bytes=features.bytes_moved + 4.0 * features.params,
        )


class LearnedSurrogateCostModel:
    """Small online linear surrogate for latency, energy, and memory."""

    def __init__(self, *, lr: float = 1e-6) -> None:
        if lr <= 0.0:
            raise ArchitectureSearchError("surrogate learning rate must be > 0")
        self.lr = float(lr)
        self._weights: Dict[str, List[float]] = {
            "latency_ms": [0.0] * 7,
            "energy": [0.0] * 7,
            "memory_bytes": [0.0] * 7,
        }
        self.num_updates = 0

    @staticmethod
    def _vector(features: CostFeatures) -> List[float]:
        return [
            1.0,
            features.flops / 1e12,
            features.bytes_moved / 1e9,
            features.params / 1e9,
            features.tiles / 1e3,
            features.seq_len / 1e4,
            features.bandwidth_gbps / 1e3,
        ]

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def predict(self, features: CostFeatures) -> HardwareCost:
        x = self._vector(features)
        return HardwareCost(
            latency_ms=max(0.0, self._dot(self._weights["latency_ms"], x)),
            energy=max(0.0, self._dot(self._weights["energy"], x)),
            memory_bytes=max(0.0, self._dot(self._weights["memory_bytes"], x)),
        )

    def update(self, features: CostFeatures, measured: HardwareCost) -> None:
        x = self._vector(features)
        targets = {
            "latency_ms": measured.latency_ms,
            "energy": measured.energy,
            "memory_bytes": measured.memory_bytes,
        }
        for name, target in targets.items():
            pred = self._dot(self._weights[name], x)
            err = pred - float(target)
            self._weights[name] = [
                w - self.lr * err * xi for w, xi in zip(self._weights[name], x)
            ]
        self.num_updates += 1

    def update_from_autotuner(self, autotuner: Any, *, arch_name: str = "auto") -> int:
        """Fit from a BayesianAutotuner-like object exposing workload/results."""

        count = 0
        workload = getattr(autotuner, "workload")
        for result in getattr(autotuner, "results"):
            cfg = result.config
            bytes_moved = 2.0 * (
                workload.M * workload.K + workload.K * workload.N + workload.M * workload.N
            )
            features = CostFeatures(
                flops=float(workload.flops()),
                bytes_moved=bytes_moved,
                tiles=max(1.0, (workload.M / cfg.tile_m) * (workload.N / cfg.tile_n)),
                sm_arch=arch_name,
            )
            measured = HardwareCost(
                latency_ms=result.latency_ms,
                energy=max(result.latency_ms * 0.1, 0.0),
                memory_bytes=bytes_moved,
            )
            self.update(features, measured)
            count += 1
        return count


def extract_cost_features(
    target: Any,
    *,
    schedule: Optional[Mapping[str, Any]] = None,
) -> CostFeatures:
    """Extract DNAS hardware-cost features from a model, mapping, or feature object."""

    if isinstance(target, CostFeatures):
        return target
    if hasattr(target, "cost_features") and callable(target.cost_features):
        features = target.cost_features()
        if isinstance(features, CostFeatures):
            return features
        target = features
    elif hasattr(target, "features") and callable(target.features):
        features = target.features()
        if isinstance(features, CostFeatures):
            return features
        target = features

    data: Mapping[str, Any]
    if isinstance(target, Mapping):
        data = target
    else:
        data = {
            "flops": getattr(target, "flops", 0.0),
            "bytes_moved": getattr(target, "bytes_moved", 0.0),
            "params": getattr(target, "params", 0.0),
            "tiles": getattr(target, "tiles", 1.0),
            "seq_len": getattr(target, "seq_len", 1.0),
            "sm_arch": getattr(target, "sm_arch", "auto"),
            "bandwidth_gbps": getattr(target, "bandwidth_gbps", 2000.0),
            "peak_tflops": getattr(target, "peak_tflops", 312.0),
        }

    tiles = float(data.get("tiles", 1.0))
    if schedule:
        numeric_knobs = [
            float(v) for v in schedule.values() if isinstance(v, (int, float))
        ]
        if numeric_knobs:
            tiles = max(1.0, sum(numeric_knobs) / len(numeric_knobs))

    return CostFeatures(
        flops=float(data.get("flops", 0.0)),
        bytes_moved=float(data.get("bytes_moved", 0.0)),
        params=float(data.get("params", 0.0)),
        tiles=tiles,
        seq_len=float(data.get("seq_len", 1.0)),
        sm_arch=str(data.get("sm_arch", "auto")),
        bandwidth_gbps=float(data.get("bandwidth_gbps", 2000.0)),
        peak_tflops=float(data.get("peak_tflops", 312.0)),
    )


def hw_cost(
    target: Any,
    *,
    schedule: Optional[Mapping[str, Any]] = None,
    cost_model: Optional[Any] = None,
) -> HardwareCost:
    """Predict latency, energy, and memory for DNAS objectives."""

    model = cost_model or AnalyticalCostModel()
    return model.predict(extract_cost_features(target, schedule=schedule))


def measure(
    target: Any,
    *,
    reps: int = 5,
    metric: Sequence[str] = ("latency", "energy", "memory"),
    schedule: Optional[Mapping[str, Any]] = None,
) -> HardwareCost:
    """Deterministic measurement placeholder for DNAS/autotuner integration.

    The production runtime should replace this with on-device measurements.
    Keeping the same return type lets cost surrogates and tests depend on a
    stable API while compiler/runtime plumbing matures.
    """

    if reps <= 0:
        raise ArchitectureSearchError("measure reps must be > 0")
    cost = hw_cost(target, schedule=schedule)
    requested = set(metric)
    return HardwareCost(
        latency_ms=cost.latency_ms if "latency" in requested else 0.0,
        energy=cost.energy if "energy" in requested else 0.0,
        memory_bytes=cost.memory_bytes if "memory" in requested or "mem" in requested else 0.0,
    )


@dataclass
class ScheduleSpace:
    """Discrete Schedule IR knob space with architecture logits per knob."""

    knobs: Mapping[str, Sequence[Any]]
    temperature: float = 1.0
    alpha: Dict[str, Parameter] = field(init=False)

    def __post_init__(self) -> None:
        if not self.knobs:
            raise ArchitectureSearchError("ScheduleSpace requires at least one knob")
        self.alpha = {
            name: Parameter(len(values), name=f"schedule.{name}.alpha")
            for name, values in self.knobs.items()
        }

    def gates(self) -> Dict[str, List[float]]:
        return {
            name: Softmax(param, temperature=self.temperature)()
            for name, param in self.alpha.items()
        }

    def current(self, *, hard: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, values in self.knobs.items():
            gate = Softmax(self.alpha[name], temperature=self.temperature)()
            if hard or not all(isinstance(v, (int, float)) for v in values):
                out[name] = values[max(range(len(gate)), key=lambda i: gate[i])]
            else:
                out[name] = weighted_sum([float(v) for v in values], gate)
        return out


def argmax(mixed_ops: Mapping[str, MixedOp]) -> Dict[str, int]:
    """Freeze mixed ops to discrete candidate indexes."""

    return {name: op.choice() for name, op in mixed_ops.items()}


def schedule_argmax(space: ScheduleSpace) -> Dict[str, Any]:
    """Freeze a schedule search space to hard knob choices."""

    return space.current(hard=True)


def specialize(mixed_ops: Mapping[str, MixedOp], choices: Mapping[str, int]) -> Dict[str, Any]:
    """Return selected candidates for a frozen architecture choice map."""

    selected: Dict[str, Any] = {}
    for name, choice in choices.items():
        if name not in mixed_ops:
            raise ArchitectureSearchError(f"unknown MixedOp {name!r}")
        op = mixed_ops[name]
        if choice < 0 or choice >= len(op.candidates):
            raise ArchitectureSearchError(
                f"choice {choice} out of range for MixedOp {name!r}"
            )
        selected[name] = op.candidates[choice]
    return selected


def deterministic_alpha_all_reduce(
    rank_values: Sequence[Sequence[float]],
    *,
    op: str = "mean",
) -> List[float]:
    """Deterministically reduce architecture logits/gradients across ranks."""

    if not rank_values:
        raise ArchitectureSearchError("alpha all-reduce requires at least one rank")
    width = len(rank_values[0])
    if width == 0:
        raise ArchitectureSearchError("alpha vectors must be non-empty")
    for vec in rank_values:
        if len(vec) != width:
            raise ArchitectureSearchError("all alpha vectors must have the same length")

    if op not in ("sum", "mean"):
        raise ArchitectureSearchError("deterministic alpha all-reduce supports sum or mean")

    out: List[float] = []
    for i in range(width):
        total = 0.0
        c = 0.0
        for vec in rank_values:
            y = float(vec[i]) - c
            t = total + y
            c = (t - total) - y
            total = t
        out.append(total / len(rank_values) if op == "mean" else total)
    return out


@dataclass
class BilevelStepPlan:
    """Declarative plan for DNAS weight/architecture update sequencing."""

    inner_steps: int = 1
    arch_every: int = 1
    gradient: str = "unrolled"

    def __post_init__(self) -> None:
        if self.inner_steps < 1:
            raise ArchitectureSearchError("inner_steps must be >= 1")
        if self.arch_every < 1:
            raise ArchitectureSearchError("arch_every must be >= 1")
        if self.gradient not in ("unrolled", "implicit", "ste"):
            raise ArchitectureSearchError("gradient must be unrolled, implicit, or ste")

    def update_kind(self, step: int) -> str:
        return "arch" if step % self.arch_every == 0 else "weights"
