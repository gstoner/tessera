# How the RDNA GPU Scheduler Works (Micro Engine Scheduler)

Synthesis of AMD's *Micro Engine Scheduler Specification* (April 2024, 54 pp).
Source-grounded; section page numbers in parentheses. The raw cleaned text is in
[`sections/`](sections/), the structured command surface in [`api.json`](api.json).

The **MES** is the on-GPU firmware scheduler that decides *which application work
runs on the shader complex, when, and for how long*. It runs on a dedicated
RS64 micro-processor in the GPU frontend and is driven by the kernel-mode driver
(KMD) over a ring buffer of API commands.

## 1. What it has to do (p5)

Two hard requirements: **fair, efficient** sharing of the GPU, and **multiple
priority levels**. Concretely, four priority levels (mirroring the Microsoft GPU
scheduling spec):

| Level | Guarantee | Example workloads |
|-------|-----------|-------------------|
| **Real time** | Lowest possible launch latency; *may infinitely starve* lower levels | VR compositor, "super-wet" ink, True Audio |
| **Focus** | Large share, but must not infinitely starve Normal | Desktop compositor, video post, foreground app |
| **Normal** | Majority of GPU time when no Real-time work | Typical app work without user focus |
| **Low** | Runs only when all higher levels are idle (but Normal guarantees it forward progress) | Compression, encryption, Folding@home |

## 2. The two-level hardware model (p6–7)

The scheduler's whole job is mapping those requirements onto the hardware, which
schedules in **two levels**:

```
                         KMD  ──API commands──▶  MES ring buffer
                                                    │
   ┌──────────────────────────── GPU frontend ──────┼─────────────────────────┐
   │  3 micro-processors: scheduler (MES/RS64), compute fw, gfx fw            │
   └────────────────────────────────────────────────┼─────────────────────────┘
                                                     │ (1) FW scheduling:
                                                     │     map user queues → HW queues
                                                     ▼
   GFX pipe ─┬─ Queue Mgr (arbitrates N HW queues) ──┐
   CS pipe 0 ┼─ Queue Mgr ────────────────────────────┤ (2) HW scheduling:
   CS pipe 1 ┼─ Queue Mgr ────────────────────────────┤     QM picks one "ready"
   ...       ┴─ Queue Mgr ────────────────────────────┘     HW queue → "connected"
                                                     │
                                                     ▼
                            shared pool of ALUs (3D/CS shader complex)
```

- **Level 1 (firmware, MES):** decides *which application (user) queues are
  mapped onto the limited hardware queues* across the pipes.
- **Level 2 (Queue Manager hardware):** of the mapped HW queues on a pipe, the QM
  picks one *ready* queue and runs it on the shader complex — the **connected
  queue**. The MES does not pick the connected queue directly; it *influences* the
  QM's choice via knobs: **HW queue priority, quantum, connection priority**.
- Each pipe is an independent launch path, so up to `#pipes` connected queues can
  run concurrently over one **shared ALU pool** (GFX + compute).

## 3. Queue terminology & state machine (p10–11)

- **User queue** — an application's linear command stream (ring buffer of draws /
  dispatches). Analogous to a CPU thread. *Cannot run on its own.*
- **Hardware queue** — a HW descriptor holding a user queue's state (ring base,
  rd/wr ptrs). Limited in number (each pipe has a fixed set).
- **Mapping** — load user-queue state into a HW queue. **Unmapping** — save it
  back to memory; *only possible after preemption*.
- **Connected queue** — the HW queue the QM currently runs on the shader complex.

```
        ┌────────────┐   map (load state)    ┌────────────────────┐
        │  Unmapped  │ ─────────────────────▶ │ Mapped &           │
        │ (in memory)│ ◀───────────────────── │ disconnected       │
        └────────────┘  unmap (after preempt) └────────────────────┘
                                                  ▲          │ QM connects
                                       QM disconnects        │ (only if pending work)
                                       (quantum/idle)        ▼
                                              ┌────────────────────┐
                                              │ Mapped & connected │  ← runs on shaders
                                              └────────────────────┘
```

Only **mapped+connected** queues with **pending work** execute.

## 4. Round-robin scheduling (p12)

The baseline (all queues equal priority):

- The scheduler keeps a **scheduling context**: a queue list per unique
  `(queue_type ∈ {GFX, Compute, DMA}) × priority_level` pair → **12 queue lists**
  total (3 types × 4 levels). Plus per-queue/process info (MQD pointers, VMIDs,
  resources).
- It maps as many user queues onto HW queues as possible. On mapping, it programs
  a **quantum** (min run time) into the HW queue.
- **Not oversubscribed** (`#user ≤ #HW queues`): map *all* user queues, equal
  quantum each. The QM connects each in turn for its configured time; if a
  connected queue goes idle early, the QM connects the next ready queue.
- **Oversubscribed** (`#user > #HW queues`): map as many as fit, then
  **gradually unmap** them on **quantum expiry or idle** to rotate in queues from
  the next process.
- **Discovering new work on unmapped queues:** **aggregated doorbells** — SW
  writing one tells the MES an unmapped queue has work, so it can map it ASAP by
  priority. If unavailable, the MES **polls write-pointer memory** of unmapped
  queues (only under oversubscription). This is the *event-driven* core of the
  design. (Fallback path: `MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE`.)

## 5. Priority enforcement — the HW levers (p13–15)

Round-robin alone can't express priority. The MES composes these hardware
features:

| Lever | What it does |
|-------|--------------|
| **Mid-command-buffer preemption** | Preempt at submission / dispatch / threadgroup / instruction boundary — finer granularity = lower latency but more saved/restored state |
| **Wave limiting** | Cap waves a queue may issue, throttling competitors ("wave" = a group of shader threads) |
| **Pipe priority** | Each connected queue asserts a pipe priority; shader HW uses it to pick next work |
| **Dispatch tunneling** | A high-priority dispatch *immediately* disables other queues' work (a per-queue property) |
| **Queue quantum** | Both QM and FW: QM connects/disconnects by HW-queue quantum; under oversubscription FW unmaps on quantum expiry |
| **Queue connection priority** | Per-HW-queue value the QM uses to pick the next queue to connect |
| **Compute-unit reservation** | Carve out CUs exclusively for one queue when launch latency is critical |

How they combine per level (p15):

- **Real time:** stays mapped once created (**max 4 RT queues, ≤1 per pipe**);
  **CU reservation** (or **wave limiting** on competitors) to grab CUs fast;
  **highest** connection priority and shader-type priority.
- **Focus vs Normal:** same connection priority, but Focus gets a **larger
  quantum**, **higher pipe priority**, and Normal queues get a **heavier
  preemption level**; FW may **unmap long-running Normal** queues on other pipes
  so Focus work can launch.
- **Low/Idle:** runs only after all higher levels have been idle for some time.

**Gangs:** queues are grouped into *gangs* sharing a global priority level, each
with its own `gang_quantum`; a *process* owns gangs across priorities (-7..+7)
with a `process_quantum`. Quanta are in **100 ns units** (see `ADD_QUEUE`).

## 6. Firmware architecture (p8–9)

Five components on the RS64 processor:

1. **Scheduler APIs** — driver→FW commands (queue create/destroy/suspend/priority).
2. **Scheduler context** — the state: HW-resource state (HQD = mapped queue/type/
   time; VMID = mapped process; GDS partition owner) + process scheduling state
   (per-level process list, grace period, normalband %, `has_ready_queues`;
   per-process gang lists + quantum + running-time carryover; per-gang queue list
   + quantum).
3. **API processor** — applies incoming APIs to the context.
4. **Core scheduler** — reads context, decides actions (map/unmap/suspend), applies
   them. This is the §4–5 algorithm.
5. **Interrupt handler** — fields HW interrupts.

Interrupt sources the RS64 sees: `ME0 Pipe0` (gfx), `ME1/ME2 Pipe0-3` (8 compute
pipes), MES packet FIFO (new API data), HW-queue message (QM), software, timer,
unprivileged-access, external (non-gfx blocks).

## 7. The driver↔firmware API (p16–17, full set in `api.json`)

- KMD submits commands to the **MES ring buffer**. Every command is a fixed
  **64-DWORD** frame (`API_FRAME_SIZE_IN_DWORDS = 64`).
- Header: `type:4` (1 = Scheduling), `opcode:8` (`MES_SCH_API_OPCODE`), `dwsize:8`.
- Completion: each command carries `api_status` = a **fence addr + value**; MES
  writes the value to that address to signal the KMD the command was processed.

The 19 scheduling opcodes (0–18) and the `MISC` sub-commands are catalogued with
purpose + page + C union in [`api.json`](api.json). The ones that matter most for
understanding the model:

| Opcode | Command | Role |
|-------:|---------|------|
| 0 | `SET_HW_RSRC` | First init API — describes the HW resources to MES |
| 1 | `SET_SCHEDULING_CONFIG` | Grace periods / priority bands (Windows `DxgkDdiSetPriorityBands`) |
| 2 | `ADD_QUEUE` | Register a user queue (process/gang/quantum/priority/MQD/doorbell/VMID…) |
| 3 | `REMOVE_QUEUE` | Unregister a user queue |
| 5 | `SET_GANG_PRIORITY_LEVEL` | Change a gang's global priority |
| 6 / 7 | `SUSPEND` / `RESUME` | Pause / resume a gang or queue |
| 8 | `RESET` | Hang detection & recovery |
| 10 | `CHANGE_GANG_PRORITY` *(sic)* | Per-gang priority change |
| 14 | `MISC` | Non-scheduling ops (register r/w, GART invalidation, wait-reg-mem, shader debugger, unmapped-queue notify…) |
| 15 | `UPDATE_ROOT_PAGE_TABLE` | Swap a process's page-table base |
| 17 | `SET_SE_MODE` | Power-gate a shader engine |
| 18 | `SET_GANG_SUBMIT` | Pair two queues for co-submission |

## Why this matters for a Tessera ROCm backend

This is the layer **beneath** anything the compiler emits — it explains the
runtime contract the generated kernels live inside, and it bounds what a backend
can assume:

- **Concurrency model.** "Up to `#pipes` connected queues over a shared ALU pool"
  is the real co-residency story behind multi-stream / multi-kernel overlap.
  Persistent-kernel and producer/consumer scheduling assumptions (cf. the CDNA3
  wave-specialization spine in `ROCM_AUDIT.md`) sit *on top of* MES queue
  mapping, not instead of it.
- **Preemption granularity** (submission / dispatch / threadgroup / instruction)
  is a real cost the runtime pays; **`is_long_running`** queues and the Focus-vs-
  Normal unmap behavior bear directly on how a long FA/GEMM kernel coexists with
  other work.
- **Quantum & gangs** (100 ns units, `process/gang_quantum`, `SET_GANG_SUBMIT`)
  are the knobs a launcher would set when it eventually registers a HIP launcher
  into the runtime's GPU-launch bridge (BACKEND_AUDIT.md G7).
- **`ADD_QUEUE` is the concrete queue-creation ABI** (doorbell offset, MQD addr,
  VMID, GDS/GWS, trap handler) — the field list a Tessera runtime queue
  abstraction would have to populate to schedule work on real silicon.

> The MES spec carries no RDNA gfx-version tag and does not name a specific ISA;
> it cross-references the *RDNA3 ISA Reference Guide* (p7). Treat the queue/pipe
> counts and exact knobs as **family-level**, confirmed against the actual part
> (gfx1151 / Strix Halo) at bring-up.
