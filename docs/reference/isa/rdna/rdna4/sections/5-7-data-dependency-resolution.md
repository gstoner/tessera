# 5.7. Data Dependency Resolution

> RDNA4 ISA — pages 61–61

5.7. Data Dependency Resolution
Shader hardware can resolve most non-memory data dependencies, but memory and export dependencies
require explicit handling by the shader program. This section describes the rules shader programmers must
follow in order to get correct results from their shader program. The programmer is rarely required to insert
waitstates (S_NOP or unrelated instructions).

The S_WAIT_*CNT instructions are used to resolve data hazards.

Each wave has counters that count the number of outstanding memory instructions per wave. These counters
increment when an instruction is issued and decrement when the instructions completes. These allow the
shader writer to schedule long-latency instructions, execute unrelated work and specify when results of long-
latency operations are needed. The shader uses the appropriate S_WAIT_*CNT instruction to wait for the
named counter to be less than or equal to a certain value, to know that some or all of the previously issued
instructions have completed. While waiting, the wave is effectively asleep (inactive, unable to issue
instructions).

Simple S_WAIT_*CNT Example

    GLOBAL_LOAD_B32 V0, V[4:5], 0x0     // load memory[ {V5, V4} ] into V0, LOADcnt++
    GLOBAL_LOAD_B32 V1, V[4:5], 0x8     // load memory[ {V5, V4} +8 ] into V1, LOADcnt++
    S_WAIT_LOADcnt <= 1                 // wait for first global_load to have completed
    V_MOV_B32   V9, V0                  // move V0 into V9

Another Example

      S_LOAD s0, ...
      S_LOAD s1, ...
      S_WAIT_KMcnt <= 0
      S_MOV_B32 s2, s0     // out of order

      DS_param_load v0
      DS_param_load v1
      EXP v2
      EXP v3
      S_WAIT_EXPcnt <= 1    // higher values don't work but also don't need zero
      V_MOV_B32 v4, v0

Instructions of a given type return in order (except SMEM which return out-of-order), but instructions of
different types can complete out of order. For example, both exports and LDS-param-load instructions use
EXPcnt but they can return out of order. From vector memory, samples stay in order with samples (from one
wave) but are unordered with respect to loads, stores and BVH ops. Export-grants return in-order within a
family of targets but out-of-order between different families (where a family is: MRT’s and Z, positions,
primitive data, dual-source-blends).

                                       Table 26. Data Dependency Instructions
