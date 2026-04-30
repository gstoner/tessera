# Tessera Research

This folder holds experimental research prototypes that are useful to keep close
to the main repo, but are not part of the production source tree or default
build.

Research projects should stay here when they are:

- exploring an interface, compiler idea, model pattern, or verification method
- useful as source material for future examples or production components
- too experimental to live under `src/`, `python/`, or `tools/`

When a research project becomes part of the product surface, promote it into the
appropriate canonical folder and leave a short note in that project's README.

## Active Projects

- `pddl_instruct/` - symbolic/verifier-guided reasoning and structured
  chain-of-thought experiments for Tessera.
- `sandbox_compilers/` - small frontend-to-backend compiler experiments for
  TileScript-style DSLs and backend codegen sketches.

## Archive

Stale research snapshots should move to `research/archive/`, not into
`src/archive/`, unless they were previously production source.
