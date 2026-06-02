# Domain Audit

Domain audit tracking lives here. This theme covers GA/EBM, attention variants,
CorrDiff/SciML, sharding, autodiff crosscuts, and other domain-driven compiler
requirements.

## Current Truth

- Start with [DOMAIN_AUDIT.md](DOMAIN_AUDIT.md) for the consolidated domain
  audit: GA/EBM, attention, CorrDiff/SciML, sharding, autodiff, and archived
  source material.
- Domain roadmaps are implementation context, not generated status truth.
- GA/EBM, attention, CorrDiff, and sharding docs remain useful as historical
  design sources, but current primitive/support status should be checked through
  generated dashboards.
- Generated status should be read from:
  - [../generated/support_table.md](../generated/support_table.md)
  - [../generated/s_series_status.md](../generated/s_series_status.md)
  - [../generated/tsol_coverage.md](../generated/tsol_coverage.md)
  - [../generated/effect_lattice_audit.md](../generated/effect_lattice_audit.md)

## Open Items

- Keep domain scope locks separate from backend execution claims.
- Route domain feature readiness through coverage/compiler/backend proof gates.
- Preserve GA/EBM and attention roadmap provenance while avoiding duplicate
  current-status claims.

## Archived Source Material

- [attention_variants_plan.md](archive/attention_variants_plan.md)
- [corrdiff_compiler_split_evaluation.md](archive/corrdiff_compiler_split_evaluation.md)
- [ebm_scope_lock.md](archive/ebm_scope_lock.md)
- [ga6_autodiff_plan.md](archive/ga6_autodiff_plan.md)
- [ga_ebm_roadmap.md](archive/ga_ebm_roadmap.md)
- [ga_scope_lock.md](archive/ga_scope_lock.md)
- [sharding_partial_audit.md](archive/sharding_partial_audit.md)
- [source_base_review_2026_05_17.md](archive/source_base_review_2026_05_17.md)
