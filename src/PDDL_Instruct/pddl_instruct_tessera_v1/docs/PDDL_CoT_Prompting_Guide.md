<!-- MERGE-START: PDDL_CoT_Prompting_Guide.md -->
# PDDL‑Instruct Logical CoT Prompting Guide

We use three families of prompts. Replace bracketed blocks with domain/problem content.

## 1) Prove‑As‑You‑Go (default)
```
System: You are a symbolic planner. Use PDDL logic. Only emit valid actions.
User:
[DOMAIN]
[PROBLEM]

At each step:
1) State the current known state (facts).
2) For your chosen action a_t:
   - List preconditions and verify each against the state with true/false.
   - If any false, do NOT apply; propose an alternative.
3) If applicable == true, apply effects (add/del) and show the new state.
4) Stop when all goal conditions hold.
Emit JSON for each step using the schema:
{"step":k,"action":"(act args)","applicable":true|false,"reason":[...],"effects":{"add":[...],"del":[...]}}.
```

## 2) Plan‑Then‑Prove
First output a candidate plan as a list of actions. Then iterate stepwise proof.

## 3) Critique‑Repair (validator feedback)
Inject validator errors as input and ask for the smallest patch to fix the first failing step.
<!-- MERGE-END: PDDL_CoT_Prompting_Guide.md -->
