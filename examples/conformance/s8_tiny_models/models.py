"""Importable S8 tiny-model conformance examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import tessera as ts


ArrayTree = dict[str, Any]


@dataclass(frozen=True)
class TinyModelSpec:
    model_id: str
    family: str
    init: Callable[[], ArrayTree]
    forward: Callable[[ArrayTree, ArrayTree], np.ndarray]
    loss: Callable[[ArrayTree, ArrayTree], Any]
    sample_batch: Callable[[int], ArrayTree]
    required_surfaces: tuple[str, ...]
    compile_inputs: tuple[np.ndarray, ...]
    expected_output_shape: tuple[int, ...]
    expected_grad_shapes: dict[str, tuple[int, ...]]
    compile_fn: Callable[..., Any] | None = None


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _linear_params(in_dim: int, out_dim: int, seed: int) -> ArrayTree:
    rng = _rng(seed)
    return {
        "w": rng.normal(0.0, 0.15, size=(in_dim, out_dim)).astype(np.float64),
        "b": np.zeros((out_dim,), dtype=np.float64),
    }


def _project(features: np.ndarray, params: ArrayTree) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim > 1:
        x = arr.reshape(arr.shape[0], -1).mean(axis=0, keepdims=True)
    else:
        x = arr.reshape(1, -1)
    return ts.ops.add(ts.ops.matmul(x, params["w"]), params["b"])


def _mse_to_target(y: np.ndarray, target: np.ndarray):
    err = ts.ops.sub(y, target)
    return ts.ops.mean(ts.ops.mul(err, err))


def init_diffusion_dit() -> ArrayTree:
    params = _linear_params(12, 4, 8)
    params["kernel"] = np.ones((2, 1, 3), dtype=np.float32) / 3.0
    return params


def batch_diffusion_dit(batch_size: int = 1) -> ArrayTree:
    x = np.linspace(-1.0, 1.0, batch_size * 2 * 6, dtype=np.float32).reshape(batch_size, 2, 6)
    noise = ts.rng.normal(ts.rng.RNGKey.from_seed(81), x.shape).astype(np.float32) * 0.01
    return {"x": x, "target": np.zeros((1, 4), dtype=np.float64), "noise": noise}


def forward_diffusion_dit(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    denoise_in = ts.ops.add(batch["x"], batch["noise"])
    h = ts.nn.conv1d(denoise_in, params["kernel"], padding=1, groups=2)
    h = ts.nn.group_norm(h, num_groups=2)
    return _project(h, params)


def loss_diffusion_dit(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_diffusion_dit(params, batch), batch["target"])


def init_xlstm_recurrent() -> ArrayTree:
    return _linear_params(2, 2, 17)


def batch_xlstm_recurrent(batch_size: int = 3) -> ArrayTree:
    xs = np.linspace(-0.4, 0.5, batch_size * 2, dtype=np.float64).reshape(batch_size, 2)
    return {"x": xs, "target": np.zeros((1, 2), dtype=np.float64)}


def forward_xlstm_recurrent(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    def step(carry, item):
        gate = ts.ops.sigmoid_safe(ts.ops.add(carry, item))
        candidate = ts.ops.tanh(ts.ops.add(carry, item))
        next_carry = ts.ops.add(ts.ops.mul(gate, carry), ts.ops.mul(ts.ops.sub(1.0, gate), candidate))
        return next_carry, next_carry

    final, _ = ts.scan(step, np.zeros(2, dtype=np.float64), batch["x"])
    return _project(final, params)


def loss_xlstm_recurrent(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_xlstm_recurrent(params, batch), batch["target"])


def init_mamba_ssm() -> ArrayTree:
    params = _linear_params(2, 2, 23)
    params.update({
        "A": np.array([0.8, 0.6], dtype=np.float64),
        "B": np.array([0.4, -0.2], dtype=np.float64),
        "C": np.array([1.0, 0.5], dtype=np.float64),
    })
    return params


def batch_mamba_ssm(batch_size: int = 4) -> ArrayTree:
    xs = np.linspace(-0.2, 0.7, batch_size * 2, dtype=np.float64).reshape(batch_size, 2)
    return {"x": xs, "target": np.zeros((1, 2), dtype=np.float64)}


def forward_mamba_ssm(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    def step(state, item):
        state = ts.ops.tanh(ts.ops.add(ts.ops.mul(params["A"], state), ts.ops.mul(params["B"], item)))
        return state, ts.ops.mul(state, params["C"])

    final, ys = ts.scan(step, np.zeros(2, dtype=np.float64), batch["x"])
    return _project(ts.ops.add(final, ts.ops.mean(ys, axis=0)), params)


def loss_mamba_ssm(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_mamba_ssm(params, batch), batch["target"])


def init_hyena_fnet_spectral() -> ArrayTree:
    params = _linear_params(4, 2, 31)
    params["filter"] = np.array([0.5, 0.25, -0.25, 0.125], dtype=np.float64)
    return params


def batch_hyena_fnet_spectral(batch_size: int = 4) -> ArrayTree:
    del batch_size
    return {
        "x": np.linspace(0.0, 1.0, 4, dtype=np.float64),
        "target": np.zeros((1, 2), dtype=np.float64),
    }


def forward_hyena_fnet_spectral(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    spectrum = ts.ops.fft(batch["x"] * params["filter"])
    features = np.real(spectrum).astype(np.float64)
    return _project(features, params)


def loss_hyena_fnet_spectral(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_hyena_fnet_spectral(params, batch), batch["target"])


def init_linformer_cosformer() -> ArrayTree:
    return _linear_params(12, 3, 43)


def batch_linformer_cosformer(batch_size: int = 1) -> ArrayTree:
    tokens = np.arange(batch_size * 3 * 4, dtype=np.float32).reshape(batch_size, 3, 4) / 10.0
    return {"x": tokens, "target": np.zeros((1, 3), dtype=np.float64)}


def forward_linformer_cosformer(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    attn = ts.nn.multi_head_attention(batch["x"], batch["x"], batch["x"], num_heads=2)
    return _project(attn, params)


def loss_linformer_cosformer(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_linformer_cosformer(params, batch), batch["target"])


def init_griffin_megalodon() -> ArrayTree:
    params = _linear_params(4, 2, 53)
    params["gate"] = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    return params


def batch_griffin_megalodon(batch_size: int = 3) -> ArrayTree:
    xs = np.linspace(-0.3, 0.8, batch_size * 4, dtype=np.float64).reshape(batch_size, 4)
    return {"x": xs, "target": np.zeros((1, 2), dtype=np.float64)}


def forward_griffin_megalodon(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    def step(state, item):
        state = ts.ops.add(ts.ops.mul(params["gate"], state), item)
        return state, state

    final, ys = ts.scan(step, np.zeros(4, dtype=np.float64), batch["x"])
    return _project(ts.ops.add(final, ts.ops.mean(ys, axis=0)), params)


def loss_griffin_megalodon(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_griffin_megalodon(params, batch), batch["target"])


def init_jepa() -> ArrayTree:
    return _linear_params(4, 2, 67)


def batch_jepa(batch_size: int = 2) -> ArrayTree:
    context = np.array([[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]], dtype=np.float64)[:batch_size]
    target = np.array([[0.3, 0.6], [0.2, 0.5]], dtype=np.float64)[:1]
    return {"x": context, "target": target}


def forward_jepa(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    pooled = ts.ops.mean(batch["x"], axis=0)
    return _project(pooled, params)


def loss_jepa(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_jepa(params, batch), batch["target"])


def init_titans_atlas_memory() -> ArrayTree:
    params = _linear_params(2, 2, 79)
    params["keys"] = np.eye(2, dtype=np.float64)
    params["values"] = np.array([[0.5, -0.25], [0.25, 0.75]], dtype=np.float64)
    return params


def batch_titans_atlas_memory(batch_size: int = 1) -> ArrayTree:
    query = np.array([[1.0, 0.2]], dtype=np.float64)[:batch_size]
    new_key = np.array([[0.3, 0.7]], dtype=np.float64)
    new_value = np.array([[0.1, 0.9]], dtype=np.float64)
    return {
        "query": query,
        "new_key": new_key,
        "new_value": new_value,
        "target": np.zeros((1, 2), dtype=np.float64),
    }


def forward_titans_atlas_memory(params: ArrayTree, batch: ArrayTree) -> np.ndarray:
    table = ts.memory.MemoryTable(params["keys"], params["values"])
    table = ts.memory.memory_write(table, batch["new_key"], batch["new_value"], max_entries=3)
    read = ts.memory.memory_read(table, batch["query"], top_k=2).values
    return _project(read, params)


def loss_titans_atlas_memory(params: ArrayTree, batch: ArrayTree):
    return _mse_to_target(forward_titans_atlas_memory(params, batch), batch["target"])


def compile_mlp_slice(A, B):
    return ts.ops.relu(ts.ops.matmul(A, B))


def compile_attention_slice(A, B):
    return ts.ops.softmax(ts.ops.matmul(A, B))


def compile_recurrent_slice(A, B):
    h = ts.ops.tanh(ts.ops.matmul(A, B))
    return ts.ops.add(h, A)


def _compile_inputs(m: int = 2, k: int = 3, n: int = 2) -> tuple[np.ndarray, ...]:
    return (
        np.arange(m * k, dtype=np.float32).reshape(m, k) / 10.0,
        np.ones((k, n), dtype=np.float32) * 0.25,
    )


def manifest() -> tuple[TinyModelSpec, ...]:
    return (
        TinyModelSpec(
            "tiny_diffusion_dit",
            "diffusion/DiT",
            init_diffusion_dit,
            forward_diffusion_dit,
            loss_diffusion_dit,
            batch_diffusion_dit,
            ("S2", "S4", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15"),
            _compile_inputs(2, 3, 4),
            (1, 4),
            {"w": (12, 4), "b": (4,)},
            compile_mlp_slice,
        ),
        TinyModelSpec(
            "tiny_xlstm_recurrent",
            "xLSTM/recurrent",
            init_xlstm_recurrent,
            forward_xlstm_recurrent,
            loss_xlstm_recurrent,
            batch_xlstm_recurrent,
            ("S2", "S3", "S4", "S5", "S7", "S8", "S10", "S11", "S12", "S14"),
            _compile_inputs(2, 2, 2),
            (1, 2),
            {"w": (2, 2), "b": (2,)},
            compile_recurrent_slice,
        ),
        TinyModelSpec(
            "tiny_mamba_ssm",
            "Mamba/SSM",
            init_mamba_ssm,
            forward_mamba_ssm,
            loss_mamba_ssm,
            batch_mamba_ssm,
            ("S2", "S5", "S7", "S8", "S10", "S11", "S12", "S14"),
            _compile_inputs(2, 2, 2),
            (1, 2),
            {"w": (2, 2), "b": (2,), "A": (2,), "B": (2,), "C": (2,)},
            compile_recurrent_slice,
        ),
        TinyModelSpec(
            "tiny_hyena_fnet_spectral",
            "Hyena/FNet/spectral",
            init_hyena_fnet_spectral,
            forward_hyena_fnet_spectral,
            loss_hyena_fnet_spectral,
            batch_hyena_fnet_spectral,
            ("S2", "S7", "S8", "S10", "S11", "S12", "S14"),
            _compile_inputs(2, 4, 2),
            (1, 2),
            {"w": (4, 2), "b": (2,), "filter": (4,)},
            compile_mlp_slice,
        ),
        TinyModelSpec(
            "tiny_linformer_cosformer",
            "Linformer/cosFormer",
            init_linformer_cosformer,
            forward_linformer_cosformer,
            loss_linformer_cosformer,
            batch_linformer_cosformer,
            ("S2", "S5", "S6", "S7", "S8", "S10", "S11", "S12", "S14", "S15"),
            _compile_inputs(2, 3, 3),
            (1, 3),
            {"w": (12, 3), "b": (3,)},
            compile_attention_slice,
        ),
        TinyModelSpec(
            "tiny_griffin_megalodon",
            "Griffin/Megalodon",
            init_griffin_megalodon,
            forward_griffin_megalodon,
            loss_griffin_megalodon,
            batch_griffin_megalodon,
            ("S2", "S5", "S7", "S8", "S10", "S11", "S12", "S14"),
            _compile_inputs(2, 4, 2),
            (1, 2),
            {"w": (4, 2), "b": (2,), "gate": (4,)},
            compile_recurrent_slice,
        ),
        TinyModelSpec(
            "tiny_jepa",
            "JEPA",
            init_jepa,
            forward_jepa,
            loss_jepa,
            batch_jepa,
            ("S2", "S3", "S7", "S8", "S10", "S11", "S12", "S14", "S15"),
            _compile_inputs(2, 4, 2),
            (1, 2),
            {"w": (4, 2), "b": (2,)},
            compile_mlp_slice,
        ),
        TinyModelSpec(
            "tiny_titans_atlas_memory",
            "Titans/Atlas memory",
            init_titans_atlas_memory,
            forward_titans_atlas_memory,
            loss_titans_atlas_memory,
            batch_titans_atlas_memory,
            ("S2", "S3", "S7", "S8", "S10", "S11", "S12", "S14", "S15"),
            _compile_inputs(2, 2, 2),
            (1, 2),
            {"w": (2, 2), "b": (2,), "keys": (2, 2), "values": (2, 2)},
            compile_attention_slice,
        ),
    )


def surfaces_covered() -> frozenset[str]:
    return frozenset(surface for spec in manifest() for surface in spec.required_surfaces)
