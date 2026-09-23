#!/usr/bin/env python3
"""Exact-output, one-lever gfx1201 folded-prefill codegen ablations.

This is an experiment, not a production selector. The complete-wave fast path
hoists output and activation bases and uses 32-bit row offsets; ragged tiles
retain the production epilogue unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import emit_mxfp4_folded_prefill_hip
from benchmarks.rocm import ablate_gfx1201_folded_b_cache as shared
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base


_EPILOGUE_START = "\n#pragma unroll\n  for (int j = 0; j < 2; ++j) {"

_FAST_EPILOGUE = r'''
  // The CTA-wide condition avoids divergent barrier behavior; the branch is
  // after all K steps. The 32-bit product is bounded by N <= 2^23 and the
  // largest relative row offset (63), well below signed 32-bit range.
  if (m0 + 256 <= M && n0 + 64 <= N && N <= 8388607) {
    const long row_base = m0 + wm * 64;
    const float *Asb = As + row_base;
    __bf16 *Ob = O + row_base * N + n0 + wn * 32 + col;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned char exponent = Ref[n0 + wn * 32 + j * 16 + col];
      const float row_scale = exponent ? __builtin_bit_cast(float, (unsigned int)exponent << 23) : 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int relative_row = i * 16 + half + e;
          const float partial = acc[i][j][e];
          const float activation_scale = Asb[relative_row];
          const float combined_scale = row_scale * activation_scale;
          float scaled = partial * combined_scale;
          if (__builtin_expect(
                  !__builtin_isfinite(combined_scale) || combined_scale == 0.0f,
                  0)) {
            if (partial == 0.0f && __builtin_isfinite(activation_scale))
              scaled = 0.0f;
            else
              scaled = (float)((double)partial * (double)row_scale *
                               (double)activation_scale);
          }
          const int relative_offset = relative_row * (int)N + j * 16;
          Ob[relative_offset] = (__bf16)scaled;
        }
    }
  } else {
'''


def epilogue_source() -> str:
    source = emit_mxfp4_folded_prefill_hip(full_k64=True)
    head, marker, tail = source.rpartition(_EPILOGUE_START)
    if not marker or not tail.endswith("\n}\n"):
        raise RuntimeError("folded epilogue template changed; review ablation")
    return head + "\n" + _FAST_EPILOGUE + marker + tail[:-2] + "  }\n}\n"


def staging_source() -> str:
    """Hoist 64-bit A/B bases and use bounded 32-bit relative row offsets."""
    source = emit_mxfp4_folded_prefill_hip(full_k64=True)
    changes = (
        (
            "  for (long kb = 0; kb < K; kb += 64) {",
            "  if (K > 8388607) __builtin_trap();\n"
            "  const int Ki = (int)K;\n"
            "  for (long kb = 0; kb < K; kb += 64) {\n"
            "    const unsigned char *Ab = A + m0 * K + kb;\n"
            "    const unsigned char *Bb = B + n0 * K + kb;",
        ),
        (
            "value = *reinterpret_cast<const copy_u32x4 *>(A + safe * K + kb + off);",
            "const int relative_row = (int)(safe - m0);\n"
            "        value = *reinterpret_cast<const copy_u32x4 *>(Ab + relative_row * Ki + off);",
        ),
        (
            "value = *reinterpret_cast<const copy_u32x4 *>(B + safe * K + kb + off);",
            "const int relative_row = (int)(safe - n0);\n"
            "        value = *reinterpret_cast<const copy_u32x4 *>(Bb + relative_row * Ki + off);",
        ),
    )
    for old, new in changes:
        if source.count(old) != (1 if "for (long kb" in old else 2):
            raise RuntimeError("folded staging template changed; review ablation")
        source = source.replace(old, new)
    return source


def barrier_source() -> str:
    """Probe Radiance's stricter K-step scheduling barrier without other edits."""
    source = emit_mxfp4_folded_prefill_hip(full_k64=True)
    old = "__builtin_amdgcn_sched_barrier(6);"
    if source.count(old) != 1:
        raise RuntimeError("folded K-step barrier template changed; review ablation")
    return source.replace(old, "__builtin_amdgcn_sched_barrier(0);")


def unconditional_kstep_source() -> str:
    """Also make all four K16 compute steps unconditional within a K64 slab."""
    source = emit_mxfp4_folded_prefill_hip(full_k64=True)
    old = "step < 4 && kb + step * 16 < K"
    if source.count(old) != 1:
        raise RuntimeError("folded K16 loop template changed; review ablation")
    return source.replace(old, "step < 4")


def benchmark(
    radiance_module: Path, radiance_revision: str, tessera_opt: Path,
    *, variant_kind: str, warmup: int = 6, trials: int = 11, iterations: int = 12,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("epilogue ablation requires selected gfx1201")
    if os.environ.get("RADIANCE_MXFP4_WPERM") != "1":
        raise ValueError("matched Radiance requires fragment-order WPERM=1")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("epilogue ablation requires HIP")
    if variant_kind not in (
        "epilogue", "staging", "barrier", "full_k64",
    ):
        raise ValueError(f"unsupported codegen ablation: {variant_kind}")
    source = {
        "epilogue": epilogue_source,
        "staging": staging_source,
        "barrier": barrier_source,
        "full_k64": unconditional_kstep_source,
    }[variant_kind]()
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    payload = shared._compile_variant(source)
    radiance = base._load_radiance(radiance_module)
    rows: list[dict[str, object]] = []
    for case in (
        base.Case("prefill", 256, 5120, 8704),
        base.Case("prefill", 1024, 17408, 5120),
    ):
        inputs = base._logical_inputs(case)
        exact = base._tessera_engine(hip, case, inputs, 3, None, None)
        baseline, folded = folded_bench.folded_engine(
            hip, case, inputs, 3, tessera_opt=tessera_opt,
        )
        variant = shared._variant_engine(
            hip, case, inputs, folded, payload, source_sha,
            variant_name=f"tessera_{variant_kind}_ablation",
            ablation=(
                {
                    "epilogue": "full_wave_epilogue_base_hoist_only",
                    "staging": "A_B_32bit_relative_address_only",
                    "barrier": "K_step_sched_barrier_mode_0_only",
                    "full_k64": "unconditional_K16_compute_steps_only",
                }[variant_kind]
            ),
        )
        independent = base._radiance_engine(hip, radiance, case, inputs, 3)
        engines = (exact, baseline, variant, independent)
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sampled_rows, sampled_cols, reference = base._sampled_exact_reference(
                case, inputs,
            )
            np.testing.assert_array_equal(
                outputs["tessera"][np.ix_(sampled_rows, sampled_cols)],
                reference,
            )
            if not folded.lossless:
                raise RuntimeError("epilogue ablation inputs must fold losslessly")
            for name, output in outputs.items():
                np.testing.assert_array_equal(
                    output.view(np.uint16), outputs["tessera"].view(np.uint16),
                    err_msg=f"{case.label}: {name} changed BF16 output",
                )
            samples = base._measure_interleaved(
                hip, list(engines), warmup=warmup, trials=trials,
                iterations=iterations,
            )
            for engine in engines:
                rows.append({
                    "case": case.label,
                    "engine": engine.name,
                    "median_ms": statistics.median(samples[engine]),
                    "samples_ms": samples[engine],
                    "output_sha256": hashlib.sha256(
                        outputs[engine.name].view(np.uint8),
                    ).hexdigest(),
                    "metadata": engine.metadata,
                })
        finally:
            for engine in reversed(engines):
                engine.close()
    return {
        "schema": "tessera.rocm.gfx1201_folded_codegen_ablation.v1",
        "variant": variant_kind,
        "device": base._selected_device_name(hip),
        "architecture": rt._rocm_live_arch(),
        "source_revision": base._git_revision(),
        "benchmark_sha256": base._sha256(Path(__file__)),
        "generator_sha256": base._sha256(
            shared.ROOT / "python/tessera/compiler/rocm_mxfp4_folded.py"
        ),
        "variant_source_sha256": source_sha,
        "radiance_revision": radiance_revision,
        "radiance_binary_sha256": base._sha256(radiance_module),
        "radiance_wperm": 1,
        "selected": "tessera_folded",
        "method": "alternating_interleaved_hip_events",
        "phase_attribution_admissible": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("epilogue", "staging", "barrier", "full_k64"),
        required=True,
    )
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = benchmark(
        args.radiance_module, args.radiance_revision, args.tessera_opt,
        variant_kind=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
