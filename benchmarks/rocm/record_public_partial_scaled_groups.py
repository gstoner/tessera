"""Correctness-gated public/native baseline for static trailing K32 groups."""
import argparse,json
from pathlib import Path
from benchmarks.rocm import record_public_independent_scaled_primal as base
from tests.unit.test_public_partial_scaled_groups import partial_case,oracle

def case(mask,tb=False,encoded=False,prefix=(2,3),jvp=False,seed=1007,ta=False):
    if tuple(prefix)!=(2,3):raise ValueError("recorder requires its explicit static prefix")
    return partial_case(mask,ta,tb,encoded,37,jvp,seed=seed)

def measure():
    original_case,original_expected=base.case,base.expected
    base.case,base.expected=case,oracle
    try:packet=base.measure()
    finally:base.case,base.expected=original_case,original_expected
    packet["partial_recorder_sha256"]=base.digest(__file__)
    packet["partial_fixture_sha256"]=base.digest("tests/unit/test_public_partial_scaled_groups.py")
    packet["limitations"]=["static M3 N5 K37 and prefix 2x3","aligned K32 scale blocks with a partial trailing group",
        "FP32 scale derivatives only","no speedup or selector promotion","no sibling-device proof"]
    for row in packet["rows"]:row["shape_mnk"]=[3,5,37]
    return packet

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(),indent=2)+"\n")
