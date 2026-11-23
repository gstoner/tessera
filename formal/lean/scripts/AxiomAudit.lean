-- LEAN-PILOT-1 axiom audit. Run with `lake env lean scripts/AxiomAudit.lean`
-- after `lake build`. tests/unit/test_formal_lean_pilot.py parses this output
-- and allows only Lean's three standard axioms (propext, Classical.choice,
-- Quot.sound); `sorryAx` or any project axiom fails the gate.
import TesseraFormal

#print axioms TesseraFormal.VJPExp.vjpExp_eq_pullback
#print axioms TesseraFormal.VJPExp.vjpExp_hasDerivAt
#print axioms TesseraFormal.PagedKV.page_in_bounds
#print axioms TesseraFormal.PagedKV.offset_in_bounds
#print axioms TesseraFormal.PagedKV.reconstruct
#print axioms TesseraFormal.PagedKV.slot_map_injective
#print axioms TesseraFormal.PagedKV.physical_in_bounds
#print axioms TesseraFormal.PagedKV.physical_injective
#print axioms TesseraFormal.Softmax.softmax_shift
