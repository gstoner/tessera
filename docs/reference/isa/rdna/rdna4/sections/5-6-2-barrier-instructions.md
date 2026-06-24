# 5.6.2. Barrier Instructions

> RDNA4 ISA — pages 59–59

S_NOP.

All waves in work-groups have these barriers:

Work-group Barrier
      A barrier that is completed when each of the waves in the work-group have signaled or terminated.

Trap Barrier
      Same as the work-group-barrier, but exclusively for use by the trap handler. If the user shader tries to
      signal or wait on it, the operation ignored.

5.6.1. Barrier State
A Barrier consists of two counts:
     • Member count: how many waves must signal in order to complete the barrier. Initialized to the work-
       group size for the workgroup and trap barriers.
     • Signaled count: how many waves have already signaled the barrier. This resets to zero when the first wave
       of the work-group is created and after barrier completes.

When the last wave signals a barrier making it complete (signaledCount == memberCount), the barrier
broadcasts to the work-group that the barrier is complete and then resets signaledCount=0. Keep in mind that
trap and work-group barriers can only complete once they’re valid. When a work-group is launched, all
signalCounts are set to zero, the trap and work-group barrier’s memberCount is set to the number of waves in
the work-group.

Waves have these bits of state related to barriers:
     • barrierComplete: records if the work-group barrier has previously completed but wave has not yet waited
       on it. This bit is used by S_BARRIER_WAIT: if the bit is 1 when S_BARRIER_WAIT executes, the
       S_BARRIER_WAIT executes immediately and the bit is reset to zero; if the bit is zero then the wave waits
       until the barrier completes (and then the bit is reset to zero when the wave executes S_BARRIER_WAIT).
     • trapBarrierComplete: same, but for the trap barrier

This wave state (excluding the trap barrier state) can be read with S_GETREG_B32 STATE_PRIV, and written
with S_SETREG_B32 STATE_PRIV only while in the trap handler (PRIV=1). The barrier state can be read with
S_GET_BARRIER_STATE but cannot be written.

5.6.2. Barrier Instructions
Barriers are controlled using: S_BARRIER_SIGNAL{_ISFIRST}, S_BARRIER_WAIT, and
S_GET_BARRIER_STATE. Barrier instructions can use M0 or an inline constant to refer to the barrier number,
except that for S_BARRIER_WAIT can only take an inline constant.

Reference to the work-group and trap barriers can only be made with inline constants.

                                              Table 24. Barrier ID’s
Instruction       Barrier               Description
Bar#
-2                Trap Barrier          Barrier dedicated for use by the trap handler
