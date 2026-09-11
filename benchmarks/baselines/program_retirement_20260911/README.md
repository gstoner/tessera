# Program retirement and declared slots — 2026-09-11

Super-Bear RTX 5070 SM120 and Princess-Luna Radeon 8060S gfx1151, both WSL.
Each packet records its compiler and recorder digest and the two native SSD
artifact identities. No timing or promotion claim.

The recorder uploads an opt-in declared-slot self-cycle and verifies that GPU
collection retains it. This covers hook-free Python slot descriptors, not
opaque CPython/C-extension heaps or concurrent host mutation.

It then captures and differentiates two resident SSD frames through public
`vjp`, closes derivative reader scopes, and retires the whole program. Bound
context synchronization is replaced with an assertion throughout capture,
backward and program retirement. Query-driven frame retirement completes
before module unloading is admitted to off-thread workers. Completion requires
both binding owners to release their native modules. A 30-second observation
limit is a test bound, not a cancellation guarantee for the driver.

Reproduce with the owning backend environment and
`record_program_retirement.py --backend nvidia|rocm --compiler <tool>
--output <packet>`. Unit tests additionally exercise live-reader refusal,
partial-retirement retry and failed-unload retention. General traced AD,
concurrent sweeping and arbitrary heap discovery remain open.
