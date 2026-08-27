"""Rank-local MPI transport for DIST-NATIVE-1.

This adapter is intentionally separate from the one-process/multi-device
NCCL/RCCL adapter.  Every Python process owns exactly one rank and one local
array.  Admission binds the declared world size, rank ordering, host/local-rank
ownership, communicator digest, and collective sequence before issuing data
transport.  ``mpi4py`` is optional; its absence is a hard unavailable state,
never a mock fallback.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class MPIRankTopology:
    world_size: int
    rank: int
    rank_hosts: tuple[str, ...]
    local_ranks: tuple[int, ...]
    mesh_axes: tuple[tuple[str, int], ...]
    library_version: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tessera.collective.mpi_rank_topology.v1",
            "backend": "mpi",
            "world_size": self.world_size,
            "rank": self.rank,
            "rank_order": list(range(self.world_size)),
            "rank_hosts": list(self.rank_hosts),
            "local_ranks": list(self.local_ranks),
            "mesh_axes": dict(self.mesh_axes),
            "library_version": self.library_version,
            "digest": self.digest,
        }


class MPIRankAdapter:
    """One process-rank bound to an MPI communicator.

    The initial executable envelope is contiguous ``float32``, SUM reductions,
    and axis-zero movement.  Unsupported dtype, shape, axis, operation, world,
    topology digest, or cross-rank issue order fails closed.
    """

    backend = "mpi"
    rank_local = True

    def __init__(
        self,
        *,
        expected_world_size: int,
        mesh_axes: dict[str, int] | None = None,
        communicator: Any | None = None,
        expected_communicator_digest: str | None = None,
        _mpi: Any | None = None,
    ) -> None:
        mpi_module: Any
        if _mpi is None:
            try:
                mpi_module = importlib.import_module("mpi4py.MPI")
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "MPI rank transport requires a loadable mpi4py/MPI runtime"
                ) from exc
        else:
            mpi_module = _mpi
        self._mpi = mpi_module
        self._comm = (
            mpi_module.COMM_WORLD if communicator is None else communicator
        )
        world = int(self._comm.Get_size())
        rank = int(self._comm.Get_rank())
        expected_error = ""
        try:
            expected_world = int(expected_world_size)
        except (TypeError, ValueError, OverflowError) as exc:
            expected_world = 0
            expected_error = f"MPI expected world size is invalid: {exc}"
        axes_error = ""
        axes: dict[str, int] = {}
        try:
            axes = {
                str(name): int(extent)
                for name, extent in dict(mesh_axes or {"dp": world}).items()
            }
            if any(not name or extent < 1 for name, extent in axes.items()):
                axes_error = "MPI mesh axes require names and positive extents"
            product = 1
            for extent in axes.values():
                product *= extent
            if not axes_error and product != world:
                axes_error = (
                    "MPI mesh extent product must equal communicator world size"
                )
        except (TypeError, ValueError, OverflowError) as exc:
            axes_error = f"MPI mesh axes are invalid: {exc}"
        host = str(mpi_module.Get_processor_name())
        library_version = " ".join(
            str(mpi_module.Get_library_version()).split()
        )
        topology_intent = (
            expected_world, tuple(sorted(axes.items())), axes_error,
            host, library_version, expected_communicator_digest, expected_error,
        )
        topology_intents = tuple(self._comm.allgather(topology_intent))
        if len(topology_intents) != world:
            raise RuntimeError("MPI communicator topology admission is incomplete")
        authority = topology_intent[:3] + topology_intent[4:]
        if any(item[:3] + item[4:] != authority for item in topology_intents):
            raise RuntimeError(
                "MPI ranks disagree on expected world, mesh, MPI library, or "
                "communicator digest"
            )
        if expected_error:
            raise ValueError(expected_error)
        if expected_world < 2 or world != expected_world:
            raise RuntimeError(
                f"MPI communicator world_size={world} does not match required "
                f"world_size={expected_world} >= 2"
            )
        if rank < 0 or rank >= world:
            raise RuntimeError("MPI communicator returned an invalid process rank")
        if axes_error:
            raise ValueError(axes_error)
        hosts = tuple(str(item[3]) for item in topology_intents)
        local_ranks = tuple(
            sum(1 for earlier in hosts[:index] if earlier == owner)
            for index, owner in enumerate(hosts)
        )
        common = {
            "schema": "tessera.collective.mpi_communicator.v1",
            "backend": "mpi",
            "world_size": world,
            "rank_order": list(range(world)),
            "rank_hosts": list(hosts),
            "local_ranks": list(local_ranks),
            "mesh_axes": axes,
            "library_version": library_version,
        }
        communicator_digest = _digest(common)
        if (
            expected_communicator_digest is not None
            and communicator_digest != expected_communicator_digest
        ):
            raise RuntimeError("MPI communicator/topology digest mismatch")
        self.topology = MPIRankTopology(
            world_size=world,
            rank=rank,
            rank_hosts=hosts,
            local_ranks=local_ranks,
            mesh_axes=tuple(sorted((str(k), int(v)) for k, v in axes.items())),
            library_version=library_version,
            digest=communicator_digest,
        )
        self._sequence = 0
        self._artifact_admissions = 0
        self._subgroups: dict[tuple[int, ...], MPISubgroupAdapter | None] = {}

    @property
    def rank(self) -> int:
        return self.topology.rank

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    @property
    def communicator_digest(self) -> str:
        return self.topology.digest

    @property
    def mesh_axes(self) -> dict[str, int]:
        return dict(self.topology.mesh_axes)

    @property
    def rank_order(self) -> tuple[int, ...]:
        return tuple(range(self.world_size))

    def capability_snapshot(self) -> dict[str, Any]:
        return self.topology.to_dict()

    def admit_artifact(
        self, artifact_digest: str, expected_communicator_digest: str
    ) -> None:
        artifact_text = str(artifact_digest)
        communicator_text = str(expected_communicator_digest)
        record = (
            self._artifact_admissions,
            artifact_text,
            communicator_text,
            self.communicator_digest,
        )
        records = tuple(self._comm.allgather(record))
        if len(records) != self.world_size or any(item != record for item in records):
            raise RuntimeError(
                "MPI ranks disagree on collective artifact or communicator identity"
            )
        if len(artifact_text) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_text
        ):
            raise ValueError("MPI collective artifact digest must contain 64 hex characters")
        if len(communicator_text) != 64 or any(
            char not in "0123456789abcdef" for char in communicator_text
        ):
            raise ValueError(
                "MPI collective communicator digest must contain 64 hex characters"
            )
        if communicator_text != self.communicator_digest:
            raise RuntimeError("MPI collective artifact communicator digest mismatch")
        self._artifact_admissions += 1

    def for_subgroup(
        self, ranks: tuple[int, ...]
    ) -> "MPISubgroupAdapter | None":
        local_error = ""
        try:
            normalized = tuple(int(rank) for rank in ranks)
            if (
                len(normalized) < 2
                or len(set(normalized)) != len(normalized)
                or any(rank < 0 or rank >= self.world_size for rank in normalized)
            ):
                local_error = "MPI subgroup requires distinct in-range ranks"
        except (TypeError, ValueError, OverflowError) as exc:
            normalized = ()
            local_error = f"MPI subgroup membership is invalid: {exc}"
        record = (
            self._sequence,
            "subgroup",
            normalized,
            local_error,
            self.communicator_digest,
        )
        records = tuple(self._comm.allgather(record))
        if len(records) != self.world_size or any(item != record for item in records):
            raise RuntimeError(
                "MPI ranks disagree on collective sequence or subgroup membership"
            )
        if local_error:
            raise ValueError(local_error)
        self._sequence += 1
        if normalized == self.rank_order:
            return MPISubgroupAdapter(self, self, normalized)
        if normalized in self._subgroups:
            return self._subgroups[normalized]
        color = 1 if self.rank in normalized else self._mpi.UNDEFINED
        key = normalized.index(self.rank) if self.rank in normalized else self.rank
        communicator = self._comm.Split(color=color, key=key)
        if self.rank not in normalized:
            self._subgroups[normalized] = None
            return None
        inner = MPIRankAdapter(
            expected_world_size=len(normalized),
            mesh_axes={"subgroup": len(normalized)},
            communicator=communicator,
            _mpi=self._mpi,
        )
        wrapper = MPISubgroupAdapter(self, inner, normalized)
        self._subgroups[normalized] = wrapper
        return wrapper

    def _admit(
        self,
        kind: str,
        value: Any,
        *,
        options: tuple[Any, ...] = (),
        artifact_context: tuple[Any, ...] = (),
        expected_dtype: str | None = None,
    ) -> np.ndarray:
        local_error = ""
        try:
            source = np.asarray(value)
            shape = tuple(int(v) for v in source.shape)
            dtype = str(source.dtype)
        except Exception as exc:  # every rank must reach the admission collective
            source = np.asarray([], dtype=np.float32)
            shape = ()
            dtype = "invalid"
            local_error = f"MPI rank transport could not materialize its input: {exc}"
        record = (
            self._sequence, kind, tuple(options), tuple(artifact_context),
            shape, dtype, local_error,
            self.communicator_digest,
        )
        records = self._comm.allgather(record)
        if len(records) != self.world_size or any(item != record for item in records):
            raise RuntimeError(
                "MPI ranks disagree on collective sequence, kind, options, "
                "dtype, shape, or communicator"
            )
        if local_error:
            raise ValueError(local_error)
        if source.dtype != np.float32:
            raise ValueError("MPI rank transport v1 requires float32 storage")
        if expected_dtype not in {None, "f32", "float32"}:
            raise ValueError(
                f"MPI rank transport artifact requires unsupported dtype {expected_dtype!r}"
            )
        self._sequence += 1
        return np.ascontiguousarray(source)

    def all_reduce(
        self,
        value: Any,
        *,
        op: str = "sum",
        _artifact_context: tuple[Any, ...] = (),
        _expected_dtype: str | None = None,
    ) -> np.ndarray:
        try:
            op_value, option_error = str(op), ""
        except Exception as exc:
            op_value, option_error = "", f"MPI all_reduce operation is invalid: {exc}"
        source = self._admit(
            "all_reduce",
            value,
            options=(op_value, option_error),
            artifact_context=_artifact_context,
            expected_dtype=_expected_dtype,
        )
        if option_error:
            raise ValueError(option_error)
        if op_value != "sum":
            raise ValueError("MPI rank transport v1 supports SUM reductions only")
        output = np.empty_like(source)
        self._comm.Allreduce(source, output, op=self._mpi.SUM)
        return output

    def all_gather(
        self,
        value: Any,
        *,
        axis: int = 0,
        _artifact_context: tuple[Any, ...] = (),
        _expected_dtype: str | None = None,
    ) -> np.ndarray:
        try:
            axis_value, option_error = int(axis), ""
        except (TypeError, ValueError, OverflowError) as exc:
            axis_value, option_error = 0, f"MPI all_gather axis is invalid: {exc}"
        source = self._admit(
            "all_gather",
            value,
            options=(axis_value, option_error),
            artifact_context=_artifact_context,
            expected_dtype=_expected_dtype,
        )
        if option_error:
            raise ValueError(option_error)
        if axis_value != 0:
            raise ValueError("MPI rank transport v1 all_gather requires axis=0")
        if source.ndim < 1:
            raise ValueError("MPI all_gather requires a rank-one-or-greater tensor")
        gathered = np.empty((self.world_size,) + source.shape, dtype=np.float32)
        self._comm.Allgather(source, gathered)
        return gathered.reshape((self.world_size * source.shape[0],) + source.shape[1:])

    def reduce_scatter(
        self,
        value: Any,
        *,
        axis: int = 0,
        op: str = "sum",
        _artifact_context: tuple[Any, ...] = (),
        _expected_dtype: str | None = None,
    ) -> np.ndarray:
        op_error = ""
        try:
            op_value = str(op)
        except Exception as exc:
            op_value = ""
            op_error = f"MPI reduce_scatter operation is invalid: {exc}"
        try:
            axis_value, option_error = int(axis), ""
        except (TypeError, ValueError, OverflowError) as exc:
            axis_value, option_error = 0, f"MPI reduce_scatter axis is invalid: {exc}"
        source = self._admit(
            "reduce_scatter", value,
            options=(axis_value, op_value, option_error, op_error),
            artifact_context=_artifact_context,
            expected_dtype=_expected_dtype,
        )
        if option_error:
            raise ValueError(option_error)
        if op_error:
            raise ValueError(op_error)
        if axis_value != 0 or op_value != "sum":
            raise ValueError("MPI rank transport v1 reduce_scatter requires axis=0 and SUM")
        if source.ndim < 1 or source.shape[0] % self.world_size:
            raise ValueError("MPI reduce_scatter axis 0 must divide evenly across ranks")
        shape = (source.shape[0] // self.world_size,) + source.shape[1:]
        output = np.empty(shape, dtype=np.float32)
        self._comm.Reduce_scatter_block(source, output, op=self._mpi.SUM)
        return output

    def all_to_all(
        self,
        value: Any,
        *,
        scatter_axis: int = 0,
        gather_axis: int = 0,
        _artifact_context: tuple[Any, ...] = (),
        _expected_dtype: str | None = None,
    ) -> np.ndarray:
        option_error = ""
        try:
            scatter_value = int(scatter_axis)
            gather_value = int(gather_axis)
        except (TypeError, ValueError, OverflowError) as exc:
            scatter_value = gather_value = 0
            option_error = f"MPI all_to_all axis is invalid: {exc}"
        source = self._admit(
            "all_to_all", value,
            options=(scatter_value, gather_value, option_error),
            artifact_context=_artifact_context,
            expected_dtype=_expected_dtype,
        )
        if option_error:
            raise ValueError(option_error)
        if scatter_value != 0 or gather_value != 0:
            raise ValueError("MPI rank transport v1 all_to_all requires both axes=0")
        if source.ndim < 1 or source.shape[0] % self.world_size:
            raise ValueError("MPI all_to_all axis 0 must divide evenly across ranks")
        chunk = source.shape[0] // self.world_size
        send = source.reshape((self.world_size, chunk) + source.shape[1:])
        receive = np.empty_like(send)
        self._comm.Alltoall(send, receive)
        return receive.reshape(source.shape)

    def collective_permute(
        self,
        value: Any,
        *,
        pairs: tuple[tuple[int, int], ...],
        _artifact_context: tuple[Any, ...] = (),
        _expected_dtype: str | None = None,
    ) -> np.ndarray:
        pair_error = ""
        try:
            canonical_pairs = tuple(
                (int(sender), int(target)) for sender, target in pairs
            )
        except (TypeError, ValueError, OverflowError) as exc:
            canonical_pairs = ()
            pair_error = f"MPI collective_permute peer map is invalid: {exc}"
        source = self._admit(
            "collective_permute",
            value,
            options=(canonical_pairs, pair_error),
            artifact_context=_artifact_context,
            expected_dtype=_expected_dtype,
        )
        if pair_error:
            raise ValueError(pair_error)
        sources = tuple(sender for sender, _ in canonical_pairs)
        targets = tuple(target for _, target in canonical_pairs)
        if not canonical_pairs or len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError("MPI collective_permute requires a non-empty one-to-one peer map")
        if any(peer < 0 or peer >= self.world_size for peer in (*sources, *targets)):
            raise ValueError("MPI collective_permute peer is outside the communicator")
        destination = next((target for sender, target in canonical_pairs if sender == self.rank), self._mpi.PROC_NULL)
        origin = next((sender for sender, target in canonical_pairs if target == self.rank), self._mpi.PROC_NULL)
        output = np.zeros_like(source)
        self._comm.Sendrecv(source, dest=destination, recvbuf=output, source=origin)
        return output

    def barrier(self) -> None:
        records = self._comm.allgather((self._sequence, "barrier"))
        if any(item != records[0] for item in records):
            raise RuntimeError("MPI ranks disagree on barrier sequence")
        self._sequence += 1
        self._comm.Barrier()


class MPISubgroupAdapter:
    """Global-rank view of one communicator derived from a bound MPI world."""

    backend = "mpi"
    rank_local = True

    def __init__(
        self,
        parent: MPIRankAdapter,
        inner: MPIRankAdapter,
        ranks: tuple[int, ...],
    ) -> None:
        self._parent = parent
        self._inner = inner
        self.rank_order = ranks
        self.world_size = len(ranks)
        self.rank = parent.rank
        self.mesh_axes = {"subgroup": len(ranks)}
        self.communicator_digest = _digest(
            {
                "schema": "tessera.collective.mpi_subgroup.v1",
                "parent_digest": parent.communicator_digest,
                "rank_order": list(ranks),
                "inner_digest": inner.communicator_digest,
            }
        )

    def _kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        context = tuple(kwargs.pop("_artifact_context", ()))
        return {
            **kwargs,
            "_artifact_context": (*context, self.communicator_digest),
        }

    def all_reduce(self, value: Any, **kwargs: Any) -> np.ndarray:
        return self._inner.all_reduce(value, **self._kwargs(kwargs))

    def all_gather(self, value: Any, **kwargs: Any) -> np.ndarray:
        return self._inner.all_gather(value, **self._kwargs(kwargs))

    def reduce_scatter(self, value: Any, **kwargs: Any) -> np.ndarray:
        return self._inner.reduce_scatter(value, **self._kwargs(kwargs))

    def all_to_all(self, value: Any, **kwargs: Any) -> np.ndarray:
        return self._inner.all_to_all(value, **self._kwargs(kwargs))

    def collective_permute(self, value: Any, **kwargs: Any) -> np.ndarray:
        pairs = tuple(kwargs.pop("pairs"))
        local = {rank: index for index, rank in enumerate(self.rank_order)}
        pair_error = ""
        try:
            local_pairs = tuple((local[source], local[target]) for source, target in pairs)
        except KeyError as exc:
            local_pairs = ()
            pair_error = (
                f"MPI collective_permute peer {exc.args[0]!r} is outside its subgroup"
            )
        context = tuple(kwargs.pop("_artifact_context", ()))
        return self._inner.collective_permute(
            value,
            pairs=local_pairs,
            **self._kwargs(
                {
                    **kwargs,
                    "_artifact_context": (*context, pair_error),
                }
            ),
        )


def mpi_rank_adapter(
    *,
    expected_world_size: int,
    mesh_axes: dict[str, int] | None = None,
    communicator: Any | None = None,
    expected_communicator_digest: str | None = None,
) -> MPIRankAdapter:
    return MPIRankAdapter(
        expected_world_size=expected_world_size,
        mesh_axes=mesh_axes,
        communicator=communicator,
        expected_communicator_digest=expected_communicator_digest,
    )


__all__ = ["MPIRankAdapter", "MPIRankTopology", "mpi_rank_adapter"]
