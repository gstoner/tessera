"""Task adapters for real long-memory benchmarks (LongMemEval / MemoryArena /
LongBench-v2), mapped onto Tessera's memory primitives.

These are *adapters*, not dataset downloads.  Per Tessera's standalone-runtime
rule (Decision #23) nothing here touches the network or imports a benchmark
harness — the single permitted concession is *file-format compatibility*: each
adapter can consume the real dataset's records in-memory (``from_records``) or
from a JSONL file (``from_jsonl``), and otherwise runs on **synthetic,
oracle-checkable instances** that mirror each benchmark's task *structure*.

The point is to show how each benchmark decomposes onto the memory contract:

* **LongMemEval** — its five abilities map to: info-extraction → top-1 recall;
  multi-session → metadata-tagged recall; temporal / knowledge-update →
  ``prefer_recent`` / ``recency_key``; abstention → ``abstain_below``.
* **MemoryArena** — an interdependent action loop: a fact written in one session
  must be retrieved to choose the correct action in a later session.
* **LongBench-v2** — multi-document MCQ: retrieve the relevant document by query,
  then decode the planted answer choice.

Accuracy is ~1.0 by construction (the oracle answer is planted): the adapters
validate that the *decomposition* is faithful and runnable, not model quality.
Swap in real records via ``from_records`` / ``from_jsonl`` to grade real data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.common import (
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Correctness,
    ExecutionKind,
    RuntimeStatus,
)
from tessera.memory import MemoryTable, memory_read, memory_write


@dataclass(frozen=True)
class AdapterResult:
    benchmark: str
    ability: str
    accuracy: float
    n_instances: int
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.accuracy >= 0.999


def _basis_keys(n: int, dim: int) -> np.ndarray:
    """``n`` distinct orthonormal keys (standard basis rows) — so retrieval is
    unambiguous and abstention is deterministic."""
    if dim < n:
        raise ValueError(f"need dim >= n distinct basis keys; got dim={dim}, n={n}")
    return np.eye(dim, dtype=np.float32)[:n]


def _onehot(label: int, classes: int) -> np.ndarray:
    v = np.zeros((classes,), np.float32)
    v[label] = 1.0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# LongMemEval — five abilities over a chat-history memory.
# ─────────────────────────────────────────────────────────────────────────────


class LongMemEvalAdapter:
    """LongMemEval-style memory QA across sessions, by ability."""

    ABILITIES = (
        "info_extraction",
        "multi_session",
        "temporal_reasoning",
        "knowledge_update",
        "abstention",
    )

    def __init__(self, n_facts: int = 12, classes: int = 8, abstain_floor: float = 0.5,
                 seed: int = 0):
        self.n_facts = int(n_facts)
        self.classes = int(classes)
        self.abstain_floor = float(abstain_floor)
        self.seed = int(seed)
        # reserve one extra basis direction for the "absent" probe
        self.dim = self.n_facts + 1

    def _bank(self, rng: np.random.Generator):
        keys = _basis_keys(self.n_facts, self.dim)
        labels = rng.integers(0, self.classes, size=self.n_facts)
        vals = np.stack([_onehot(int(l), self.classes) for l in labels])
        sessions = np.arange(self.n_facts, dtype=np.int64)
        bank = MemoryTable(keys=keys, values=vals, metadata={"session": sessions})
        return bank, keys, labels

    def info_extraction(self) -> AdapterResult:
        rng = np.random.default_rng(self.seed ^ 0x1)
        bank, keys, labels = self._bank(rng)
        hits = 0
        for i in range(self.n_facts):
            got = memory_read(bank, keys[i], top_k=1)
            hits += int(np.argmax(got.values) == labels[i])
        return AdapterResult("longmemeval", "info_extraction",
                             hits / self.n_facts, self.n_facts)

    def multi_session(self) -> AdapterResult:
        rng = np.random.default_rng(self.seed ^ 0x2)
        bank, keys, labels = self._bank(rng)
        # query a specific session's fact; metadata travels with the entry
        hits = 0
        for i in range(self.n_facts):
            got = memory_read(bank, keys[i], top_k=1)
            sess_ok = bank.metadata["session"][int(got.indices[0])] == i
            hits += int(np.argmax(got.values) == labels[i] and sess_ok)
        return AdapterResult("longmemeval", "multi_session",
                             hits / self.n_facts, self.n_facts)

    def _update_bank(self, rng):
        """One fact written twice (old then new) under the same key."""
        key = _basis_keys(1, self.dim)
        old = _onehot(int(rng.integers(0, self.classes)), self.classes)
        new = _onehot(int(rng.integers(0, self.classes)), self.classes)
        bank = MemoryTable(keys=np.zeros((0, self.dim), np.float32),
                           values=np.zeros((0, self.classes), np.float32),
                           metadata={"version": np.zeros((0,), np.int64)})
        bank = memory_write(bank, key, old[None])
        bank = memory_write(bank, key, new[None])
        return bank, key[0], int(np.argmax(new))

    def knowledge_update(self) -> AdapterResult:
        rng = np.random.default_rng(self.seed ^ 0x3)
        n = 8
        hits = 0
        for _ in range(n):
            bank, key, newest = self._update_bank(rng)
            got = memory_read(bank, key, top_k=1, prefer_recent=True)
            hits += int(np.argmax(got.values) == newest)
        return AdapterResult("longmemeval", "knowledge_update", hits / n, n)

    def temporal_reasoning(self) -> AdapterResult:
        # explicit version column: the highest version wins regardless of order
        rng = np.random.default_rng(self.seed ^ 0x4)
        n = 8
        hits = 0
        for _ in range(n):
            key = _basis_keys(1, self.dim)
            a = _onehot(int(rng.integers(0, self.classes)), self.classes)
            b = _onehot(int(rng.integers(0, self.classes)), self.classes)
            va, vb = sorted(rng.choice(np.arange(1, 50), size=2, replace=False))
            # write newer-by-order first, but tag versions so older-order has max version
            bank = MemoryTable(keys=np.concatenate([key, key]),
                               values=np.stack([a, b]),
                               metadata={"version": np.array([vb, va], np.int64)})
            got = memory_read(bank, key[0], top_k=1, recency_key="version")
            # version vb > va, so row 0 (value a) must win
            hits += int(np.argmax(got.values) == int(np.argmax(a)))
        return AdapterResult("longmemeval", "temporal_reasoning", hits / n, n)

    def abstention(self) -> AdapterResult:
        rng = np.random.default_rng(self.seed ^ 0x5)
        bank, keys, _labels = self._bank(rng)
        absent = np.eye(self.dim, dtype=np.float32)[self.n_facts]  # reserved dir
        miss = memory_read(bank, absent, top_k=1, abstain_below=self.abstain_floor)
        present = memory_read(bank, keys[0], top_k=1, abstain_below=self.abstain_floor)
        ok = bool(miss.abstained) and not bool(present.abstained)
        return AdapterResult("longmemeval", "abstention", float(ok), 1)

    def run(self) -> list[AdapterResult]:
        return [
            self.info_extraction(),
            self.multi_session(),
            self.temporal_reasoning(),
            self.knowledge_update(),
            self.abstention(),
        ]

    # ── real-data hook (file-format compat only) ─────────────────────────────
    @staticmethod
    def from_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Accept LongMemEval-format records in memory and normalize the fields
        this adapter consumes (``question_id``, ``question_type``, ``answer``,
        ``haystack_sessions``).  No network, no benchmark import — just parsing
        the documented schema so real data can flow through the same path."""
        out = []
        for r in records:
            out.append({
                "id": r.get("question_id"),
                "ability": r.get("question_type"),
                "answer": r.get("answer"),
                "sessions": r.get("haystack_sessions", []),
            })
        return out

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list[dict[str, Any]]:
        with open(path) as f:
            return cls.from_records(json.loads(line) for line in f if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
# MemoryArena — interdependent sessions: memory must guide a later action.
# ─────────────────────────────────────────────────────────────────────────────


class MemoryArenaAdapter:
    """A two-session action loop where session 2's correct action depends on a
    fact observed and written in session 1."""

    def __init__(self, n_episodes: int = 16, n_actions: int = 4, dim: int = 8,
                 seed: int = 0):
        self.n_episodes = int(n_episodes)
        self.n_actions = int(n_actions)
        self.dim = int(dim)
        self.seed = int(seed)

    def run(self) -> list[AdapterResult]:
        rng = np.random.default_rng(self.seed ^ 0xA4E2A)
        keys = _basis_keys(self.n_episodes, max(self.dim, self.n_episodes))
        with_mem = 0
        for ep in range(self.n_episodes):
            # session 1: observe context → correct action is a fact we must store
            correct = int(rng.integers(0, self.n_actions))
            bank = MemoryTable(keys=keys[ep][None],
                               values=_onehot(correct, self.n_actions)[None])
            # session 2: same context returns → retrieve memory → choose action
            recalled = memory_read(bank, keys[ep], top_k=1)
            action = int(np.argmax(recalled.values))
            with_mem += int(action == correct)
        acc = with_mem / self.n_episodes
        baseline = 1.0 / self.n_actions                 # guessing without memory
        return [AdapterResult("memory_arena", "action_loop", acc, self.n_episodes,
                              detail=f"memory-guided={acc:.2f} vs guess={baseline:.2f}")]

    @staticmethod
    def from_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in records:
            out.append({
                "episode": r.get("episode_id"),
                "sessions": r.get("sessions", []),
                "gold_action": r.get("gold_action"),
            })
        return out


# ─────────────────────────────────────────────────────────────────────────────
# LongBench-v2 — multi-document MCQ: retrieve the relevant doc, decode the choice.
# ─────────────────────────────────────────────────────────────────────────────


class LongBenchV2Adapter:
    """Multi-document MCQ: a context of N documents, one holds the answer; the
    query retrieves it and the planted choice is decoded."""

    def __init__(self, n_docs: int = 32, n_choices: int = 4, n_questions: int = 16,
                 seed: int = 0):
        self.n_docs = int(n_docs)
        self.n_choices = int(n_choices)
        self.n_questions = int(n_questions)
        self.seed = int(seed)

    def run(self) -> list[AdapterResult]:
        rng = np.random.default_rng(self.seed ^ 0x10B2)
        dim = self.n_docs
        doc_keys = _basis_keys(self.n_docs, dim)
        hits = 0
        for _ in range(self.n_questions):
            answer_doc = int(rng.integers(0, self.n_docs))
            gold_choice = int(rng.integers(0, self.n_choices))
            doc_vals = np.zeros((self.n_docs, self.n_choices), np.float32)
            doc_vals[answer_doc] = _onehot(gold_choice, self.n_choices)
            bank = MemoryTable(keys=doc_keys, values=doc_vals)
            # the question's query points at the relevant document
            got = memory_read(bank, doc_keys[answer_doc], top_k=1)
            hits += int(np.argmax(got.values) == gold_choice)
        return [AdapterResult("longbench_v2", "multi_doc_mcq",
                              hits / self.n_questions, self.n_questions)]

    @staticmethod
    def from_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in records:
            out.append({
                "id": r.get("_id"),
                "question": r.get("question"),
                "choices": [r.get("choice_A"), r.get("choice_B"),
                            r.get("choice_C"), r.get("choice_D")],
                "answer": r.get("answer"),
            })
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate.
# ─────────────────────────────────────────────────────────────────────────────

ADAPTERS = (LongMemEvalAdapter, MemoryArenaAdapter, LongBenchV2Adapter)


def run_all_adapters(seed: int = 0) -> list[AdapterResult]:
    results: list[AdapterResult] = []
    for cls in ADAPTERS:
        results.extend(cls(seed=seed).run())
    return results


def adapter_rows(results: list[AdapterResult]) -> list[BenchmarkRow]:
    """Render adapter results as reference BenchmarkRows for the shared schema."""
    rows = []
    for r in results:
        rows.append(BenchmarkRow(
            operator=BenchmarkOperator(name=f"{r.benchmark}/{r.ability}",
                                       dtype="fp32", shape=f"n={r.n_instances}",
                                       target="cpu"),
            compiler_path=CompilerPath.REFERENCE,
            runtime_status=RuntimeStatus.EXECUTABLE,
            correctness=Correctness(passed=r.passed),
            execution_kind=ExecutionKind.REFERENCE,
            metrics={"accuracy": r.accuracy, "n_instances": r.n_instances},
            reason=r.detail,
        ))
    return rows


def adapter_report(results: list[AdapterResult] | None = None) -> dict[str, Any]:
    results = results or run_all_adapters()
    by_bench: dict[str, list[float]] = {}
    for r in results:
        by_bench.setdefault(r.benchmark, []).append(r.accuracy)
    return {
        "benchmarks": sorted(by_bench),
        "abilities_graded": len(results),
        "mean_accuracy": {b: round(float(np.mean(v)), 4) for b, v in by_bench.items()},
        "all_pass": all(r.passed for r in results),
    }
