from tessera.compiler.graph_dataflow import GraphDataflow
from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IROp, tensor_ir_type


F32_4 = tensor_ir_type((4,), "fp32")


def _op(result: str, name: str, operands: list[str], *, aliasing: str = "none") -> IROp:
    return IROp(
        result=result,
        op_name=name,
        operands=operands,
        operand_types=[str(F32_4)] * len(operands),
        result_type=str(F32_4),
        inferred_type=F32_4,
        kwargs={"tessera.aliasing": aliasing},
    )


def test_shape_alias_liveness_and_activity_are_one_snapshot():
    view = _op("view", "tessera.reshape", ["%x"], aliasing="operand_0")
    dead = _op("dead", "tessera.add", ["%y", "%y"])
    out = _op("out", "tessera.add", ["%view", "%view"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", F32_4), IRArg("y", F32_4)],
        result_types=[F32_4],
        body=[view, dead, out],
        return_values=["%out"],
    )
    analysis = GraphDataflow(fn).recompute()

    assert analysis.fact("%out").shape == ("4",)
    assert analysis.may_alias("%x", "%view")
    assert not analysis.may_alias("%view", "%out")
    assert analysis.fact("%dead").live is False
    assert analysis.is_active(view)
    assert not analysis.is_active(dead)


def test_unknown_producer_and_stale_snapshot_fail_closed():
    mystery = _op("m", "tessera.unregistered_external", ["%x"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", F32_4)],
        result_types=[F32_4],
        body=[mystery],
        return_values=["%m"],
    )
    analysis = GraphDataflow(fn).recompute()
    assert analysis.fact("%m").alias_roots is None
    assert analysis.may_alias("%x", "%m")

    fn.body.append(_op("later", "tessera.add", ["%m", "%m"]))
    assert analysis.valid is False
    assert analysis.fact("%x").alias_roots is None
    assert analysis.is_active(mystery)


def test_stop_gradient_is_an_explicit_activity_barrier():
    before = _op("before", "tessera.mul", ["%x", "%x"])
    stop = _op("stop", "tessera.stop_gradient", ["%before"])
    out = _op("out", "tessera.add", ["%stop", "%stop"])
    fn = GraphIRFunction(
        "f",
        args=[IRArg("x", F32_4)],
        result_types=[F32_4],
        body=[before, stop, out],
        return_values=["%out"],
    )
    analysis = GraphDataflow(fn).recompute()
    assert analysis.is_active(out)
    assert analysis.is_active(stop)
    assert not analysis.is_active(before)
