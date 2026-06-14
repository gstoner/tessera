"""Guided diffusion utilities, starting with Contrastive Gradient Guidance.

CGG is a test-time score-composition method: run a reference denoiser and one
or more favored/unfavored denoiser pairs, then guide the sampler with

    s_ref + sum_i gamma_i(t) * (s_favored_i - s_unfavored_i).

This module is deliberately forward-only. Objective-gradient guidance can build
on the same ``DenoiseOutput`` and schedule contracts later, but CGG itself does
not require a reward model or a backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np


ArrayLike = Any
PredictionType = Literal["score", "noise", "x0"]


class GuidanceError(ValueError):
    """Raised when guided diffusion contracts are internally inconsistent."""


class DenoiserProtocol(Protocol):
    """Schedule-aware denoiser callable used by guidance and samplers."""

    def __call__(
        self,
        x_t: ArrayLike,
        t: int,
        condition: Any,
        schedule: "DiffusionSchedule",
    ) -> "DenoiseOutput":
        ...


@dataclass(frozen=True)
class DiffusionSchedule:
    """Minimal variance-preserving schedule metadata for guided sampling.

    ``alpha_bar[t]`` is the cumulative signal power at integer timestep ``t``.
    Timesteps are interpreted as descending from high noise to low noise.
    """

    alpha_bar: Sequence[float]
    # Validated float64 cache of ``alpha_bar``, set in __post_init__ via
    # object.__setattr__ (frozen dataclass). Declared so the type checker sees it.
    _alpha_bar: "np.ndarray" = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.alpha_bar, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise GuidanceError("alpha_bar must be a 1-D sequence with at least two entries")
        if not np.all((values > 0.0) & (values <= 1.0)):
            raise GuidanceError("alpha_bar values must be in (0, 1]")
        object.__setattr__(self, "_alpha_bar", values)

    @property
    def num_steps(self) -> int:
        return int(self._alpha_bar.size)

    def alpha(self, t: int) -> float:
        self._check_t(t)
        return float(np.sqrt(self._alpha_bar[int(t)]))

    def sigma(self, t: int) -> float:
        self._check_t(t)
        return float(np.sqrt(max(1.0 - self._alpha_bar[int(t)], 0.0)))

    def _check_t(self, t: int) -> None:
        if not (0 <= int(t) < self.num_steps):
            raise GuidanceError(f"timestep {t} outside [0, {self.num_steps})")

    def score_from_noise(self, noise_pred: ArrayLike, t: int) -> np.ndarray:
        sigma = max(self.sigma(t), 1.0e-12)
        return -np.asarray(noise_pred, dtype=np.float64) / sigma

    def noise_from_score(self, score: ArrayLike, t: int) -> np.ndarray:
        return -self.sigma(t) * np.asarray(score, dtype=np.float64)

    def x0_from_noise(self, x_t: ArrayLike, noise_pred: ArrayLike, t: int) -> np.ndarray:
        alpha = max(self.alpha(t), 1.0e-12)
        return (np.asarray(x_t, dtype=np.float64) - self.sigma(t) * np.asarray(noise_pred)) / alpha

    def noise_from_x0(self, x_t: ArrayLike, x0_pred: ArrayLike, t: int) -> np.ndarray:
        sigma = max(self.sigma(t), 1.0e-12)
        return (np.asarray(x_t, dtype=np.float64) - self.alpha(t) * np.asarray(x0_pred)) / sigma

    def score_from_x0(self, x_t: ArrayLike, x0_pred: ArrayLike, t: int) -> np.ndarray:
        return self.score_from_noise(self.noise_from_x0(x_t, x0_pred, t), t)


@dataclass(frozen=True)
class DenoiseOutput:
    """Normalized denoiser result.

    At least one of ``score``, ``noise_pred``, or ``x0_pred`` must be supplied.
    CGG normalizes all outputs to a score before composition.
    """

    score: ArrayLike | None = None
    noise_pred: ArrayLike | None = None
    x0_pred: ArrayLike | None = None
    prediction_type: PredictionType = "score"
    timestep: int | None = None
    model_id: str = "model"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_score(self, x_t: ArrayLike, t: int, schedule: DiffusionSchedule) -> np.ndarray:
        if self.score is not None:
            return np.asarray(self.score, dtype=np.float64)
        if self.noise_pred is not None:
            return schedule.score_from_noise(self.noise_pred, t)
        if self.x0_pred is not None:
            return schedule.score_from_x0(x_t, self.x0_pred, t)
        raise GuidanceError(f"DenoiseOutput {self.model_id!r} has no score, noise_pred, or x0_pred")

    def as_noise(self, x_t: ArrayLike, t: int, schedule: DiffusionSchedule) -> np.ndarray:
        if self.noise_pred is not None:
            return np.asarray(self.noise_pred, dtype=np.float64)
        if self.score is not None:
            return schedule.noise_from_score(self.score, t)
        if self.x0_pred is not None:
            return schedule.noise_from_x0(x_t, self.x0_pred, t)
        raise GuidanceError(f"DenoiseOutput {self.model_id!r} has no score, noise_pred, or x0_pred")


Gamma = float | Callable[[int, DiffusionSchedule], float]


@dataclass(frozen=True)
class ContrastivePair:
    """One CGG favored/unfavored score pair."""

    favored: DenoiserProtocol
    unfavored: DenoiserProtocol | Literal["base"] = "base"
    gamma: Gamma = 0.75
    name: str = "preference"
    enabled_timestep_range: tuple[int, int] | None = None

    def gamma_at(self, t: int, schedule: DiffusionSchedule) -> float:
        value = self.gamma(t, schedule) if callable(self.gamma) else self.gamma
        return float(value)

    def enabled_at(self, t: int) -> bool:
        if self.enabled_timestep_range is None:
            return True
        lo, hi = self.enabled_timestep_range
        return int(lo) <= int(t) <= int(hi)


@dataclass(frozen=True)
class GuidanceSafety:
    """Numerical guardrails for contrastive score deltas."""

    gamma_min: float = 0.0
    gamma_max: float = 2.0
    max_delta_norm: float | None = None
    max_delta_ratio: float | None = None


@dataclass(frozen=True)
class GuidanceResult:
    guided_score: np.ndarray
    base_output: DenoiseOutput
    component_outputs: dict[str, DenoiseOutput]
    deltas: dict[str, np.ndarray]
    scales: dict[str, float]
    metadata: dict[str, Any]


class ContrastiveScoreGuidance:
    """Contrastive Gradient Guidance score composition."""

    def __init__(
        self,
        pairs: ContrastivePair | Sequence[ContrastivePair],
        *,
        safety: GuidanceSafety | None = None,
    ) -> None:
        if isinstance(pairs, ContrastivePair):
            pairs = (pairs,)
        self.pairs = tuple(pairs)
        if not self.pairs:
            raise GuidanceError("ContrastiveScoreGuidance requires at least one pair")
        self.safety = safety or GuidanceSafety()

    def apply(
        self,
        base_denoiser: DenoiserProtocol,
        x_t: ArrayLike,
        t: int,
        condition: Any,
        schedule: DiffusionSchedule,
        *,
        base_output: DenoiseOutput | None = None,
    ) -> GuidanceResult:
        x_arr = np.asarray(x_t, dtype=np.float64)
        base = base_output or base_denoiser(x_arr, t, condition, schedule)
        base_score = base.as_score(x_arr, t, schedule)
        guided = np.array(base_score, dtype=np.float64, copy=True)

        components: dict[str, DenoiseOutput] = {"base": base}
        deltas: dict[str, np.ndarray] = {}
        scales: dict[str, float] = {}
        clipped: dict[str, dict[str, float]] = {}
        disabled: list[str] = []

        for pair in self.pairs:
            if not pair.enabled_at(t):
                disabled.append(pair.name)
                scales[pair.name] = 0.0
                deltas[pair.name] = np.zeros_like(base_score)
                continue

            favored = pair.favored(x_arr, t, condition, schedule)
            if pair.unfavored == "base":
                unfavored = base
            else:
                unfavored = pair.unfavored(x_arr, t, condition, schedule)
            fav_score = favored.as_score(x_arr, t, schedule)
            unfav_score = unfavored.as_score(x_arr, t, schedule)
            self._check_shape(pair.name, base_score, fav_score, "favored")
            self._check_shape(pair.name, base_score, unfav_score, "unfavored")

            gamma_raw = pair.gamma_at(t, schedule)
            gamma = float(np.clip(gamma_raw, self.safety.gamma_min, self.safety.gamma_max))
            delta = fav_score - unfav_score
            delta, clip_info = self._clip_delta(delta, base_score)
            if gamma != gamma_raw or clip_info:
                clipped[pair.name] = {"gamma_raw": gamma_raw, "gamma": gamma, **clip_info}

            guided = _score_combine(guided, delta, gamma=gamma)
            components[f"{pair.name}.favored"] = favored
            components[f"{pair.name}.unfavored"] = unfavored
            deltas[pair.name] = delta
            scales[pair.name] = gamma

        return GuidanceResult(
            guided_score=guided,
            base_output=base,
            component_outputs=components,
            deltas=deltas,
            scales=scales,
            metadata={
                "base_norm": _norm(base_score),
                "guided_norm": _norm(guided),
                "clipped": clipped,
                "disabled": tuple(disabled),
            },
        )

    @staticmethod
    def _check_shape(name: str, base: np.ndarray, other: np.ndarray, role: str) -> None:
        if other.shape != base.shape:
            raise GuidanceError(
                f"contrastive pair {name!r} {role} score shape {other.shape} "
                f"does not match base shape {base.shape}"
            )

    def _clip_delta(
        self,
        delta: np.ndarray,
        base_score: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        info: dict[str, float] = {}
        limit = self.safety.max_delta_norm
        if self.safety.max_delta_ratio is not None:
            ratio_limit = self.safety.max_delta_ratio * max(_norm(base_score), 1.0e-12)
            limit = ratio_limit if limit is None else min(limit, ratio_limit)
        if limit is None:
            return delta, info
        delta_norm = _norm(delta)
        if delta_norm <= limit or delta_norm <= 1.0e-12:
            return delta, info
        scale = float(limit / delta_norm)
        info.update({"delta_norm": delta_norm, "delta_limit": float(limit), "delta_scale": scale})
        return delta * scale, info


@dataclass(frozen=True)
class SamplerStep:
    timestep: int
    state: np.ndarray
    score_norm: float
    noise_norm: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GuidedSamplerResult:
    final: np.ndarray
    trajectory: tuple[SamplerStep, ...]
    guidance: tuple[GuidanceResult | None, ...]


class GuidedDiffusionSampler:
    """Deterministic DDIM-style sampler for guided score experiments."""

    def __init__(self, schedule: DiffusionSchedule) -> None:
        self.schedule = schedule

    def sample(
        self,
        x_init: ArrayLike,
        denoiser: DenoiserProtocol,
        *,
        condition: Any = None,
        guidance: ContrastiveScoreGuidance | None = None,
        timesteps: Sequence[int] | None = None,
    ) -> GuidedSamplerResult:
        if timesteps is None:
            timesteps = tuple(range(self.schedule.num_steps - 1, 0, -1))
        x = np.asarray(x_init, dtype=np.float64)
        steps: list[SamplerStep] = []
        guidance_results: list[GuidanceResult | None] = []

        for t in timesteps:
            base = denoiser(x, int(t), condition, self.schedule)
            if guidance is not None:
                gr = guidance.apply(
                    denoiser, x, int(t), condition, self.schedule, base_output=base
                )
                score = gr.guided_score
            else:
                gr = None
                score = base.as_score(x, int(t), self.schedule)
            noise = self.schedule.noise_from_score(score, int(t))
            x0 = self.schedule.x0_from_noise(x, noise, int(t))
            prev_t = max(int(t) - 1, 0)
            x = self.schedule.alpha(prev_t) * x0 + self.schedule.sigma(prev_t) * noise
            steps.append(
                SamplerStep(
                    timestep=int(t),
                    state=x.copy(),
                    score_norm=_norm(score),
                    noise_norm=_norm(noise),
                    metadata={} if gr is None else gr.metadata,
                )
            )
            guidance_results.append(gr)

        return GuidedSamplerResult(
            final=x,
            trajectory=tuple(steps),
            guidance=tuple(guidance_results),
        )


class ObjectiveGradientGuidance:
    """Placeholder contract for look-ahead objective-gradient guidance.

    This is intentionally not implemented in the CGG milestone: it requires
    gradients through ``x0_pred`` and objective functions, unlike score-only CGG.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "ObjectiveGradientGuidance is reserved for the look-ahead gradient "
            "guidance milestone; CGG v1 is forward-only score composition."
        )


def _norm(x: ArrayLike) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1)))


def _score_combine(base: np.ndarray, delta: np.ndarray, *, gamma: float) -> np.ndarray:
    # Lazy import avoids a package-init cycle: tessera imports this module before
    # the ops namespace is fully constructed.
    from tessera import ops as _ops

    return np.asarray(_ops.score_combine(base, delta, gamma=gamma), dtype=np.float64)


__all__ = [
    "ContrastivePair",
    "ContrastiveScoreGuidance",
    "DenoiseOutput",
    "DenoiserProtocol",
    "DiffusionSchedule",
    "GuidanceError",
    "GuidanceResult",
    "GuidanceSafety",
    "GuidedDiffusionSampler",
    "GuidedSamplerResult",
    "ObjectiveGradientGuidance",
    "SamplerStep",
]
