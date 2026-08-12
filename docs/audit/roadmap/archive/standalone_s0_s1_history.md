---
last_updated: 2026-08-11
audit_role: reference
---

# Standalone compiler S0/S1 history

This archived note preserves the scope and registry decisions formerly embedded
in `docs/audit/standalone_primitive_coverage.md`. Current status lives in the
generated sections of that dashboard and in `docs/audit/generated/`.

## S0 scope decision (2026-05-10)

The data pipeline is in scope. Tessera owns native dataset, batching,
data-sharding, and tokenization surfaces. `tf.data`, `torch.utils.data`, Grain,
Tiktoken, Hugging Face Tokenizers, and SentencePiece are reference
vocabularies, not runtime dependencies.

The training step, functional optimizers and losses, checkpointing,
custom-primitive authoring, AOT export, and the persistent compilation cache are
also in scope.

## S1 registry result

S1 established the primitive registry and its contract axes. Completion meant
that the registry, generators, and drift tests existed; it did not mean every
primitive had a native kernel on every target.

The initial implementation imported existing operators from `OP_SPECS`, joined
registered VJP/JVP rules, distinguished reference-only from Graph-lowered
operations, and introduced the S-series/model-family classification. Those
mechanisms have since evolved into the live generated registry and compiler
foundation summaries.

## Historical reading rule

Use this file only for scope provenance. Do not use it for counts, target
support, open work, or sequencing.
