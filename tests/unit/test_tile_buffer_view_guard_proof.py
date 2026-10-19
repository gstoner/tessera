"""Native scalar-view proof must preserve masking under unproved conditions."""
import hashlib

import pytest

from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def guarded_tile(tool, condition):
    expressions = {
        "and": "%ok = arith.andi %in_bounds, %other : i1",
        "nested_and": "%first = arith.andi %in_bounds, %other : i1\n"
                      "%ok = arith.andi %other, %first : i1",
        "or": "%ok = arith.ori %in_bounds, %other : i1",
        "negated": "%ok = arith.xori %in_bounds, %true : i1",
        "select": "%ok = arith.select %other, %in_bounds, %true : i1",
        "wrong_bound": "%wide = arith.cmpi ult, %row, %six : i64\n"
                       "%ok = arith.andi %wide, %other : i1",
        "else": "%ok = arith.andi %in_bounds, %other : i1",
    }
    accesses = """
      %va = tile.load_element %av[%row, %zero] : !tile.buffer_view<f16, "read", "row_major"> -> f16
      %vb = tile.load_element %bv[%row, %zero] : !tile.buffer_view<f16, "read", "row_major"> -> f16
      %sum = arith.addf %va, %vb : f16
      tile.store_element %sum, %ov[%row, %zero] : f16, !tile.buffer_view<f16, "write", "row_major">
    """
    guarded = ("scf.if %ok { } else {" + accesses + "}" if condition == "else"
               else "scf.if %ok {" + accesses + "}")
    text = '''module {
      llvm.func @proof_row(%a: !llvm.ptr, %b: !llvm.ptr, %out: !llvm.ptr, %row: i64, %cols: i64) {
        %zero = arith.constant 0 : i64
        %rows = arith.constant 3 : i64
        %columns = arith.constant 5 : i64
        %six = arith.constant 6 : i64
        %true = arith.constant true
        %av = tile.buffer_view %a, %rows, %columns, %columns : (!llvm.ptr, i64, i64, i64) -> !tile.buffer_view<f16, "read", "row_major">
        %bv = tile.buffer_view %b, %rows, %columns, %columns : (!llvm.ptr, i64, i64, i64) -> !tile.buffer_view<f16, "read", "row_major">
        %ov = tile.buffer_view %out, %rows, %columns, %columns : (!llvm.ptr, i64, i64, i64) -> !tile.buffer_view<f16, "write", "row_major">
        %in_bounds = arith.cmpi ult, %row, %rows : i64
        %other = arith.cmpi eq, %cols, %columns : i64
        ''' + expressions[condition] + "\n" + guarded + '''
        llvm.return
      }
    }'''
    printed = invoke(tool, text)
    assert printed.returncode == 0, printed.stderr
    function = "\n".join(line[2:] for line in printed.stdout.splitlines()[1:-1]).rstrip()
    digest = hashlib.sha256(function.encode()).hexdigest()
    seal = "a" * 64
    return ('module {\n tile.row_region_kernel attributes {name = "proof_row", arch = "sm_120", '
            'rows = 3 : i64, columns = 5 : i64, storage = f16, workgroup_size = 128 : i64, '
            f'schema = 6 : i64, source_hash = "{seal}", artifact_hash = "{seal}", body_hash = "{digest}"'
            '} {\n' + function + "\n}\n}")


@pytest.mark.parametrize("condition,expected_guards", [
    ("and", 1), ("nested_and", 1), ("or", 4), ("negated", 4),
    ("select", 4), ("wrong_bound", 4), ("else", 4)])
def test_only_proved_enclosing_conditions_remove_memory_guards(
        compiler_toolchain, condition, expected_guards):
    tool = compiler_toolchain.require_tessera_opt("--lower-tile-to-nvidia")
    result = invoke(tool, guarded_tile(tool, condition), "--lower-tile-to-nvidia=sm=120")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("scf.if") == expected_guards
    assert result.stdout.count("llvm.load") == 2
    assert result.stdout.count("llvm.store") == 1
