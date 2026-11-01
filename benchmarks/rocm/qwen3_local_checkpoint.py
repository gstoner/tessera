"""Read pinned Qwen3-8B checkpoint tensors from a local Hugging Face snapshot.

Reference/diagnostic only. The ranged-HTTP loader in
``benchmark_rocm_nvfp4_checkpoint`` serves layer 0; whole-model quality needs
every layer, so this reads the downloaded snapshot instead. It depends on
NumPy and ml_dtypes alone, so both the Torch reference environment and the
native measurement environment can import it.

Integrity: a snapshot file is a symlink into the cache's ``blobs`` directory,
and Hugging Face names each LFS blob by its SHA-256. Every shard is hashed and
compared with that name before a tensor is read, so a truncated or substituted
shard fails closed instead of producing plausible weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

BF16_REPO = "Qwen/Qwen3-8B"
BF16_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
NVFP4_REPO = "nvidia/Qwen3-8B-NVFP4"
NVFP4_REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
PINS = {BF16_REPO: BF16_REVISION, NVFP4_REPO: NVFP4_REVISION}
MLP_ROLES = ("gate_proj", "up_proj", "down_proj")

_DTYPES = {
    "U8": np.dtype(np.uint8),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F32": np.dtype(np.float32),
    "BF16": np.dtype(ml_dtypes.bfloat16),
}
_SHA = re.compile("[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_dir(cache_root: Path, repo: str, revision: str) -> Path:
    """Return the pinned snapshot directory, refusing an unpinned revision."""
    if PINS.get(repo) != revision:
        raise ValueError(f"{repo}@{revision} is not a pinned quality-gate revision")
    path = Path(cache_root) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"pinned snapshot {repo}@{revision} is not downloaded under {cache_root}")
    return path


@dataclass
class LocalSnapshot:
    """One pinned repository snapshot with lazily verified Safetensors shards."""

    repo: str
    revision: str
    root: Path

    def __post_init__(self) -> None:
        index_path = self.root / "model.safetensors.index.json"
        raw = index_path.read_bytes()
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        self.weight_map: dict[str, str] = json.loads(raw)["weight_map"]
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}
        self.shard_sha256: dict[str, str] = {}

    @classmethod
    def open(cls, cache_root: Path, repo: str) -> "LocalSnapshot":
        revision = PINS[repo]
        return cls(repo, revision, snapshot_dir(cache_root, repo, revision))

    def verify_shard(self, shard: str) -> str:
        if shard in self.shard_sha256:
            return self.shard_sha256[shard]
        path = self.root / shard
        blob = Path(os.path.realpath(path))
        if not blob.is_file() or _SHA.fullmatch(blob.name) is None:
            raise RuntimeError(f"{self.repo}/{shard} is not a content-addressed cache blob")
        actual = _sha256_file(blob)
        if actual != blob.name:
            raise RuntimeError(f"{self.repo}/{shard} content hash {actual} differs from its blob name")
        self.shard_sha256[shard] = actual
        return actual

    def _header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard not in self._headers:
            self.verify_shard(shard)
            with (self.root / shard).open("rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                if not 0 < size < 1 << 28:
                    raise RuntimeError(f"{self.repo}/{shard} has an implausible header size {size}")
                header = json.loads(handle.read(size))
            self._headers[shard] = (header, 8 + size)
        return self._headers[shard]

    def tensor(self, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a copied tensor and its provenance record."""
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"{self.repo} has no tensor {name!r}") from exc
        header, data_start = self._header(shard)
        spec = header[name]
        try:
            dtype = _DTYPES[spec["dtype"]]
        except KeyError as exc:
            raise ValueError(f"unsupported Safetensors dtype {spec['dtype']!r} for {name}") from exc
        start, stop = map(int, spec["data_offsets"])
        expected = int(np.prod(spec["shape"], dtype=np.int64)) * dtype.itemsize
        if stop - start != expected:
            raise RuntimeError(f"{name} byte span {stop - start} differs from shape {spec['shape']}")
        mapped = np.memmap(self.root / shard, dtype=np.uint8, mode="r", offset=data_start + start, shape=(expected,))
        raw = np.array(mapped)  # copy out of the map before the file handle closes
        del mapped
        values = raw.view(dtype).reshape(tuple(spec["shape"]))
        record = {
            "repo": self.repo,
            "revision": self.revision,
            "shard": shard,
            "shard_sha256": self.shard_sha256[shard],
            "storage": spec,
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }
        return values, record


def load_layer_mlp(cache_root: Path, layer: int, *, bf16: LocalSnapshot | None = None,
                   nvfp4: LocalSnapshot | None = None) -> dict[str, Any]:
    """Load one layer's gate/up/down as NVFP4 storage plus the BF16 originals."""
    from tessera.compiler import rocm_nvfp4_ingest as ingest

    if type(layer) is not int or layer < 0:
        raise ValueError("layer must be a non-negative integer")
    bf16 = bf16 or LocalSnapshot.open(cache_root, BF16_REPO)
    nvfp4 = nvfp4 or LocalSnapshot.open(cache_root, NVFP4_REPO)
    projections, originals, sources = [], [], []
    for role in MLP_ROLES:
        prefix = f"model.layers.{layer}.mlp.{role}"
        weight, weight_src = nvfp4.tensor(prefix + ".weight")
        scale, scale_src = nvfp4.tensor(prefix + ".weight_scale")
        global_, global_src = nvfp4.tensor(prefix + ".weight_scale_2")
        original, original_src = bf16.tensor(prefix + ".weight")
        if weight_src["storage"]["dtype"] != "U8" or scale_src["storage"]["dtype"] != "F8_E4M3" \
                or global_src["storage"]["dtype"] != "F32":
            raise ValueError(f"{prefix} does not have the NVFP4 storage contract")
        if original_src["storage"]["dtype"] != "BF16" or original.ndim != 2:
            raise ValueError(f"{prefix} BF16 original must be a rank-2 BF16 matrix")
        n, k = original.shape
        if k % 32 or weight.shape != (n, k // 2) or scale.shape != (n, k // 16) or global_.size != 1:
            raise ValueError(f"{prefix} NVFP4 storage does not match its BF16 shape")
        projections.append(ingest.NVFP4Projection(role, weight, scale, float(global_.reshape(()))))
        originals.append(original)
        sources.append({"role": role, "weight": weight_src, "weight_scale": scale_src,
                        "weight_scale_2": global_src, "bf16": original_src})
    return {
        "layer": layer,
        "projections": projections,
        "bf16": originals,
        "source": {"bf16_index_sha256": bf16.index_sha256, "nvfp4_index_sha256": nvfp4.index_sha256,
                   "tensors": sources},
    }
