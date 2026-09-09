"""Joint physical storage requires a dominating exact SSA volume guard."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def source(guard='ule', product='%n, %m', placement='inside'):
    allocation = '%tmp = memref.alloc(%n, %m) : memref<?x?xf32>'
    return f'''module attributes {{tessera.native_result_program = true}} {{
      func.func @joint(%shape: memref<2xindex>, %out: memref<1xf32> {{tessera.result_shape = 2 : i64}}, %length: memref<1xi64>) {{
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %cap = arith.constant 64 : index
        %a = memref.load %shape[%z] : memref<2xindex>
        %b = memref.load %shape[%one] : memref<2xindex>
        %n = arith.minui %a, %cap : index
        %m = arith.minui %b, %cap : index
        %volume = arith.muli {product} : index
        %fits = arith.cmpi {guard}, %volume, %cap : index
        {allocation if placement == 'before' else ''}
        scf.if %fits {{
          {allocation if placement == 'inside' else ''}
          scf.yield
        }} else {{
          {allocation if placement == 'else' else ''}
          scf.yield
        }}
        %zero64 = arith.constant 0 : i64
        memref.store %zero64, %length[%z] : memref<1xi64>
        return
      }}
    }}'''


def lower(text):
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler required')
    return run_tessera_opt(tool, text, '--tessera-native-tape-to-gpu=status-buffer=true')


def test_exact_joint_guard_reduces_the_physical_capacity():
    output = lower(source())
    assert 'memref<256xi8>' in output


@pytest.mark.parametrize('kwargs', [dict(guard='ugt'), dict(product='%n, %n'),
                                    dict(placement='before'), dict(placement='else')])
def test_unrelated_or_nondominating_volume_cannot_authorize_storage(kwargs):
    with pytest.raises(RuntimeError): lower(source(**kwargs))
