# App Note Addendum (v1.3)

## New in v1.3
- **Tile Queue dialect**: `tessera.queue.{create,push,pop}` + `!queue.token` lets the schedule lowering express real producer/consumer ordering and analyze occupancy.
- **LSE shape checks**: `lse.save` verifies rows match; failing test included.
- **PTX body layout for `tcgen05.mma`**: Inline-asm contains a schematic PTX block gated for `sm_100`. Replace with real PTX when available.

## Migration
- Replace prior implicit token threading with queue ops in Schedule lowering.
- Add `-tessera-queue` dialect registration to your toolchain.
