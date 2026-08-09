"""Guard signed Graph/attention verifier bounds under MLIR 23.

Signless ``I64Attr`` value accessors may use an unsigned native type.  A
verifier must inspect the ``IntegerAttr`` (or its ``...Attr().getInt()``
accessor) before comparing against a signed lower bound.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "src/compiler/ir/TesseraOps.cpp",
    ROOT / "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
)
RAW_LOWER_BOUND = re.compile(
    r"\b(?P<accessor>get[A-Z][A-Za-z0-9_]*)\(\)\s*(?:<|<=)\s*-?\d+"
)


def test_signed_i64_lower_bounds_do_not_use_raw_value_accessors() -> None:
    violations: list[str] = []
    for source in SOURCES:
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            for match in RAW_LOWER_BOUND.finditer(line):
                accessor = match.group("accessor")
                # RankedTensorType::getRank() is a signed structural query, not
                # a generated signless-integer attribute accessor.
                if accessor in {"getInt", "getRank"} or accessor.endswith("Attr"):
                    continue
                violations.append(f"{source.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not violations, (
        "signed lower-bound checks must read IntegerAttr.getInt(); raw MLIR 23 "
        "value accessors can wrap negatives:\n" + "\n".join(violations)
    )
