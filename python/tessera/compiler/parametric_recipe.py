"""Opt-in, rank/prune-only native optimization before shape elaboration.

A recipe owns both the original parametric MLIR oracle and one native optimized
program. Bucket reports share that program's digest. They are not executable
specializations and cannot authorize a runtime/arbiter promotion.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .graph_ir import GraphIRModule, unresolved_element_type_diagnostics
from .presburger import PresburgerSystem, attach_presburger_system


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=128)
def _optimize(text: str, executable: str, tool_digest: str) -> str:
    # Include binary contents in cache identity: a rebuilt compiler is a new
    # producer even if its path is unchanged.
    result = subprocess.run(
        [executable, '--tessera-symdim-equality', '--canonicalize', '--cse',
         '-mlir-print-debuginfo'], input=text, text=True, capture_output=True,
        timeout=60, check=True,
    )
    return result.stdout


@dataclass(frozen=True)
class BucketRank:
    bindings: tuple[tuple[str, int], ...]
    recipe_digest: str
    retained: bool
    reason: str
    promotion_eligible: bool = False


@dataclass(frozen=True)
class ParametricRecipe:
    oracle_mlir: str
    optimized_mlir: str
    system: PresburgerSystem | None
    symbols: tuple[str, ...]
    tool_digest: str
    digest: str

    def rank_buckets(self, buckets: Sequence[Mapping[str, int]]) -> tuple[BucketRank, ...]:
        """Prune only complete integer witnesses that violate shape constraints.

        Missing bindings and unknown proof domains are rejected rather than
        silently compared as if they were instances of this recipe.
        """
        ranks = []
        for bucket in buckets:
            if set(bucket) != set(self.symbols):
                raise ValueError('bucket must bind exactly the recipe symbols')
            if any(type(v) is not int or v <= 0 for v in bucket.values()):
                raise ValueError('bucket dimensions must be positive integers')
            accepted = self.system is None or self.system.check_witness(bucket) is True
            ranks.append(BucketRank(tuple(sorted(bucket.items())), self.digest, accepted,
                                    'constraint witness satisfied' if accepted else 'constraint witness rejected'))
        return tuple(ranks)


def prepare_recipe(module: GraphIRModule, *, tessera_opt: str,
                   system: PresburgerSystem | None = None) -> ParametricRecipe:
    """Optimize a parseable symbolic recipe once with existing native passes.

    No new pass registry or alternate Python execution/lowering is introduced.
    The caller explicitly opts into this analysis tier; normal JIT execution
    retains its existing authority and oracle.
    """
    if len(module.functions) != 1:
        raise ValueError("parametric rank tier requires exactly one function")
    candidate = copy.deepcopy(module)
    problems = unresolved_element_type_diagnostics(candidate)
    if problems:
        raise ValueError('; '.join(f'{p.code}: {p.message}' for p in problems))
    symbols = set(system.symbols if system else ())
    for function in candidate.functions:
        if any(key in function.fn_attrs for key in ('tessera.dim_bindings', 'tessera.nonlinear_shape_guards')):
            raise ValueError('rank tier does not yet check nonlinear or legacy string bindings')
        if system is not None:
            existing = function.fn_attrs.get('tessera.presburger_constraints')
            if existing is not None and existing != system.to_mlir_attr():
                raise ValueError('supplied Presburger system differs from the recipe carrier')
            attach_presburger_system(function, system)
        elif 'tessera.presburger_constraints' in function.fn_attrs:
            raise ValueError('supply the typed Presburger system for bucket witness checking')
        for argument in function.args:
            for name in argument.dim_names:
                if not name.isdecimal():
                    if not name.isidentifier():
                        raise ValueError('recipe requires named symbolic dimensions')
                    symbols.add(name)
        # Every dynamic argument dimension must have a corresponding name;
        # otherwise two buckets cannot be tied to a common parametric program.
        for argument in function.args:
            if '*' in argument.ir_type.shape:
                raise ValueError('parametric recipes require ranked arguments')
            if '?' in argument.ir_type.shape:
                if len(argument.dim_names) != len(argument.ir_type.shape):
                    raise ValueError('dynamic recipe dimensions require dim_names')
    oracle = candidate.to_mlir(canonical=True)
    executable = str(Path(tessera_opt).resolve(strict=True))
    tool_digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    optimized = _optimize(oracle, executable, tool_digest)
    identity = json.dumps({'oracle': _digest(oracle), 'optimized': _digest(optimized),
                           'presburger': system.digest if system else None,
                           'tool': tool_digest, 'symbols': sorted(symbols)}, sort_keys=True)
    return ParametricRecipe(oracle, optimized, system, tuple(sorted(symbols)),
                            tool_digest, _digest(identity))
