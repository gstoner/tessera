# 5.4. Send Message Types

> RDNA4 ISA — pages 56–56

These are ILLEGAL within a clause                                These are LEGAL within a clause of any type, but not
                                                                 as the first instruction
Instructions of a different type than those of the clause type   S_DELAY_ALU
S_CLAUSE                                                         S_NOP
S_ENDPGM                                                         S_SETPRIO*
SALU                                                             S_VERSION
Branch/Jump                                                      S_WAIT_EVENT
S_SENDMSG*                                                       S_WAIT_*CNT, S_WAIT_IDLE
S_SLEEP, S_SLEEP_VAR                                             S_ICACHE_INV
S_SETHALT
DS_PARAM_LOAD, DS_DIRECT_LOAD
Export
VALU: FP64 ops

S_TRAP is legal within a clause, even as the first instruction after S_CLAUSE. S_TRAP ends the clause.

Note: V_S_* instructions (Pseudo-scalar trans) are VALU ops and may be used in a VALU clause.

If the first instruction in a VALU clause has EXEC==0, then the clause is ignored and instructions are issued as
if there were no clause. If the VALU clause starts with EXEC!=0 but EXEC becomes zero in the middle of the
clause, the clause continues until the last instruction of the specified clause.

If an S_DELAY_ALU is needed before starting a clause, the order must be:

      S_DELAY_ALU // must not come immediately after S_CLAUSE - that inst declares clause type
      S_CLAUSE
      <first instruction in clause>

If the first instruction after S_CLAUSE is skipped (e.g. due to EXEC==0, or VMEM-load skipped due to EXEC==0
and LOADcnt==0) then a clause is not started. Subsequent instructions within what would have been the clause
are not skipped and are still executed but individually, not as part of a clause.

5.3.1. Clause Breaks
The following conditions can break a clause:

 1. VALU exception (trap) breaks a VALU clause
 2. Host commands to wave (halt, resume, single step, etc) breaks all active clauses.
    Context-save breaks clauses of affected waves.
    If a wave halts or is killed, its clauses are ended.
 3. Any action that causes a wave to jump to its trap handler breaks clause (includes context-save).
    A wave entering HALT (including for host-initiated single-step) may break clauses.

5.4. Send Message Types
S_SENDMSG is used to send messages to fixed function hardware, the host, or to request that a value be
returned to the wave. S_SENDMSG encodes the message type in the SIMM16 field and the message payload in
M0. S_SENDMSG_RTN_B* encodes the message type in the SSRC0 field (does not read an SGPR), the payload (if
any) in M0, and the destination SGPR in SDST. S_SENDMSG_RTN_B* instructions return data to the shader:
