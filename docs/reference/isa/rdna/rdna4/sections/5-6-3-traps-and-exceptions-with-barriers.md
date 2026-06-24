# 5.6.3. Traps and Exceptions with Barriers

> RDNA4 ISA — pages 60–60

Instruction     Barrier                Description
Bar#
-1              Work-group barrier     Barrier to synchronize all waves in a work-group

                                             Table 25. Barrier Instructions
Instruction               Args   Description
S_BARRIER_SIGNAL          Bar#   Signal barrier <Bar#>. Bar# can be an inline constant or M0.
S_BARRIER_SIGNAL_                When M0 is used, it supplies the barrierID# (M0[4:0]). When an inline-constant supplies
ISFIRST                          the barrierID#, the memberCount is not set. Setting MemberCnt <= SignalCnt is considered
                                 illegal and may cause the barrier to not complete.

                                 S_BARRIER_SIGNAL_ISFIRST:
                                 Same as above, but also return SCC to the wave: 1 if it was the first wave to signal the
                                 barrier in this round, else return zero. Workgroup barriers return SCC=0 or =1 for each
                                 s_barrier_signal_isfirst; Uses KMcnt to track when SCC has returned for workgroups.
                                 SCC should only be read after barrier is complete (after an S_BARRIER_WAIT).
S_BARRIER_WAIT            Bar#   Wait for a barrier to complete. Bar# must be an inline constant.
S_GET_BARRIER_STATE Bar#, Get barrier state and return it to an SGPR.
                    SDST SDST = { 0, signalCnt[6:0], 5’b0, memberCnt[6:0], 3’b0, valid }.

                                 Uses KMcnt to track completion (inc/dec by 1) Bar# is from M0[4:0] or an inline constant.

5.6.3. Traps and Exceptions with Barriers
Traps and exceptions may occur while barriers are in use. They behave normally: the wave waits for idle and
jumps to the trap handler. If a wave is waiting at S_BARRIER_WAIT when an exception occurs, the wave saves
the PC of the S_BARRIER_WAIT and jumps to the trap handler. When the shader returns from the trap
handler, the S_BARRIER_WAIT is re-executed.

5.6.4. Context Switching
It is up to the trap handler to save and restore wave and barrier-unit state. Each wave knows if its barrier has
completed or not. The trap handler begins with a S_BARRIER_SIGNAL & wait (trap barrier). This ensures that
any previous S_BARRIER_SIGNAL from the user shader has completed and reported back barrier-complete if
the barrier had completed. The trap handler can then read out and save barrier unit state (the signal-count),
and restore it later when the context is restored via a series of S_BARRIER_SIGNALs.

To save context, if a wave is part of a work-group, the wave’s user-barrier state (barrierComplete) must be
saved & restored. The context save must first ensure all waves have entered their trap handler (or exited) and
that the user’s barrier operations are all done (idle). This is accomplished by using the trap-barrier.

One wave is nominated to save the barrier unit state using the barrier’s "signal_isfirst" capability. This wave
reads the barrier state per unit (excluding the trap barrier) with S_GET_BARRIER_STATE and saves it off. On
context restore, the same wave restores the state of each barrier unit before any of the waves return to the user
shader.
