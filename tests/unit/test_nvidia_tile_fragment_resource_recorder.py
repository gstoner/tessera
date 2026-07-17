from benchmarks.nvidia.record_tile_fragment_resources import (
    parse_resource_usage,
    parse_sass_instruction_families,
)


def test_parse_tile_fragment_cubin_resources() -> None:
    text = """Resource usage:
 Common:
  GLOBAL:0
 Function tile_matmul:
  REG:42 STACK:0 SHARED:3072 LOCAL:0 CONSTANT[0]:944 TEXTURE:0
"""
    assert parse_resource_usage(text) == {
        "tile_matmul": {
            "registers_per_thread": 42,
            "stack_bytes": 0,
            "static_shared_memory_bytes": 3072,
            "local_bytes": 0,
        }
    }


def test_parse_tile_fragment_sass_families_by_kernel() -> None:
    text = """
        Function : f16_tile
        /*0700*/ HMMA.16816.F32 R4, R8, R26, R4 ;
        Function : fp8_tile
        /*03c0*/ QMMA.16832.F32.E4M3.E4M3 R8, R8, R12, RZ ;
        Function : fp4_tile
        /*0170*/ OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R16, R16, R14 ;
"""
    assert parse_sass_instruction_families(text) == {
        "f16_tile": ["HMMA.16816.F32"],
        "fp8_tile": ["QMMA.16832.F32.E4M3.E4M3"],
        "fp4_tile": ["OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"],
    }
