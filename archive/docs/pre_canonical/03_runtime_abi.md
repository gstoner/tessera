---
status: Archive
classification: Archive
last_updated: 2026-04-26
---

# 03_runtime_abi.md
# Tessera Runtime ABI (Archived Stub)

> Archived pre-canonical stub. Use `docs/spec/RUNTIME_ABI_SPEC.md` as the only normative runtime ABI reference.

## 1. Overview
A thin C ABI enabling host toolchains to load TileIR modules, allocate device memory, launch tile graphs, and interop with vendor drivers.

## 2. Versioning
- Semantic version: MAJOR.MINOR.PATCH; breaking changes bump MAJOR.

## 3. Core Interfaces
- `tsrContextCreate/Destroy`
- `tsrModuleLoad/Unload`
- `tsrMemAlloc/Free/Copy`
- `tsrTileGraphCreate/Launch/Synchronize`
- `tsrEventCreate/Record/Wait/Destroy`

## 4. Error Model
- Return codes + extended diagnostic buffer.

## 5. Scheduling Contracts
- Deterministic order within a tile; inter‑tile ordering by explicit deps.
