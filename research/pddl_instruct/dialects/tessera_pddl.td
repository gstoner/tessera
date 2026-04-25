
// MERGE-START: tessera_pddl.td
//===- tessera_pddl.td - Tessera PDDL Dialect ------------------*- tablegen -*-===//
// Minimal PDDL dialect: domains, predicates, actions, states, plans.
include "mlir/IR/OpBase.td"

def TesseraPDDL_Dialect : Dialect {
  let name = "t_pddl";
  let cppNamespace = "tessera::pddl";
}

def PDDLString : StrAttr;
def NameAttr : StrAttr;

// Types
def TPDLStateType : TypeDef<TesseraPDDL_Dialect, "State"> {
  let summary = "Symbolic state as a set of ground atoms";
}
def TPDLDomainType : TypeDef<TesseraPDDL_Dialect, "Domain"> { let summary = "PDDL domain handle"; }
def TPDLProblemType : TypeDef<TesseraPDDL_Dialect, "Problem"> { let summary = "PDDL problem handle"; }

// Ops
def PDDL_DomainOp : Op<TesseraPDDL_Dialect, "domain", [NoSideEffect]> {
  let summary = "PDDL domain container";
  let arguments = (ins PDDLString:$text);
  let results = (outs TPDLDomainType:$domain);
  let assemblyFormat = "$text attr-dict";
}

def PDDL_ProblemOp : Op<TesseraPDDL_Dialect, "problem", [NoSideEffect]> {
  let summary = "PDDL problem container";
  let arguments = (ins PDDLString:$text);
  let results = (outs TPDLProblemType:$problem);
  let assemblyFormat = "$text attr-dict";
}

def PDDL_PlanOp : Op<TesseraPDDL_Dialect, "plan", [NoSideEffect]> {
  let summary = "Ground plan and optional trace";
  let arguments = (ins PDDLString:$plan_json, PDDLString:$trace_jsonl);
  let results = (outs);
  let assemblyFormat = "$plan_json ($trace_jsonl)? attr-dict";
}
// MERGE-END: tessera_pddl.td
