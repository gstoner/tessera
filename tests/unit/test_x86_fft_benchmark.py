from __future__ import annotations

import json

from scipy import fft as scipy_fft

from benchmarks.spectral import benchmark_x86_fft as bench


def test_x86_fft_comparison_emits_forward_and_inverse_rows(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(bench.rt, "_x86_elementwise_available", lambda: True)
    monkeypatch.setattr(
        bench.rt,
        "_x86_fft_c2c_rows",
        lambda values, inverse, np: (
            scipy_fft.ifft(values, axis=-1, workers=1) * values.shape[-1]
            if inverse
            else scipy_fft.fft(values, axis=-1, workers=1)
        ).astype(np.complex64),
    )
    monkeypatch.setattr(bench, "_measure", lambda call, warmup, repeats: 1.0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_x86_fft.py",
            "--sizes",
            "64",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ],
    )

    assert bench.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["transform"] for row in rows] == ["c2c_forward", "c2c_inverse"]
    assert all(row["max_abs_error"] == 0.0 for row in rows)
