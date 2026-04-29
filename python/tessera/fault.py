"""Fault handling and injection helpers for Tessera.

The production runtime will wire these policies to distributed launchers,
communicators, watchdogs, and checkpoint/restart. The Python layer records the
same contracts for tests, documentation, and local orchestration glue.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Optional


VALID_FAILURE_POLICIES = ("drain_then_resume", "fail_fast", "pause_for_manual")
VALID_PREEMPT_ACTIONS = ("checkpoint_then_exit", "partial_flush_then_exit", "fail_fast")


class FaultPolicyError(ValueError):
    """Raised when fault-tolerance policies are invalid."""


@dataclass(frozen=True)
class FailurePolicy:
    """Failure handling policy attached to a function."""

    policy: str = "drain_then_resume"
    max_retries: int = 3
    checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.policy not in VALID_FAILURE_POLICIES:
            raise FaultPolicyError(f"policy must be one of {VALID_FAILURE_POLICIES}")
        if self.max_retries < 0:
            raise FaultPolicyError("max_retries must be >= 0")

    def to_ir_attr(self) -> str:
        ckpt = "true" if self.checkpoint else "false"
        return (
            "{tessera.fault = {"
            f'policy = "{self.policy}", max_retries = {self.max_retries}, '
            f"checkpoint = {ckpt}}}"
        )


@dataclass(frozen=True)
class PreemptPolicy:
    """Policy for scheduler preemption signals."""

    grace_s: int = 30
    action: str = "checkpoint_then_exit"

    def __post_init__(self) -> None:
        if self.grace_s <= 0:
            raise FaultPolicyError("grace_s must be > 0")
        if self.action not in VALID_PREEMPT_ACTIONS:
            raise FaultPolicyError(f"action must be one of {VALID_PREEMPT_ACTIONS}")

    def to_ir_attr(self) -> str:
        return f'{{tessera.preempt = {{grace_s = {self.grace_s}, action = "{self.action}"}}}}'


@dataclass(frozen=True)
class FaultEvent:
    """A controlled injected fault for tests and staging."""

    kind: str
    target: object = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "metadata": dict(self.metadata)}


_FAULT_STACK: list[FaultEvent] = []


def on_failure(
    fn: Optional[Callable] = None,
    *,
    policy: str = "drain_then_resume",
    max_retries: int = 3,
    checkpoint: bool = True,
) -> Callable:
    """Decorator that attaches a failure policy to a step/function."""

    cfg = FailurePolicy(policy=policy, max_retries=max_retries, checkpoint=checkpoint)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__tessera_failure_policy__ = cfg
        return wrapper

    return decorator(fn) if fn is not None else decorator


def on_preempt(
    fn: Optional[Callable] = None,
    *,
    grace_s: int = 30,
    action: str = "checkpoint_then_exit",
) -> Callable:
    """Decorator that attaches scheduler preemption handling metadata."""

    cfg = PreemptPolicy(grace_s=grace_s, action=action)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__tessera_preempt_policy__ = cfg
        return wrapper

    return decorator(fn) if fn is not None else decorator


@contextlib.contextmanager
def inject(fault: Optional[str] = None, **kwargs) -> Iterator[FaultEvent]:
    """Inject a controlled fault in tests or staging."""

    kind = fault or kwargs.pop("kind", None)
    if not kind:
        if "drop_device" in kwargs:
            kind = "drop_device"
        elif "network_partition" in kwargs:
            kind = "network_partition"
        else:
            kind = "custom"
    target = kwargs.pop("target", kwargs.get("drop_device", kwargs.get("network_partition")))
    event = FaultEvent(kind=str(kind), target=target, metadata=dict(kwargs))
    _FAULT_STACK.append(event)
    try:
        yield event
    finally:
        if _FAULT_STACK and _FAULT_STACK[-1] is event:
            _FAULT_STACK.pop()


def active_faults() -> tuple[FaultEvent, ...]:
    return tuple(_FAULT_STACK)


__all__ = [
    "FailurePolicy",
    "FaultEvent",
    "FaultPolicyError",
    "PreemptPolicy",
    "active_faults",
    "inject",
    "on_failure",
    "on_preempt",
]
