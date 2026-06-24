# 5.2.1. Clause Breaks

> RDNA3.5 ISA — pages 50–51

  • SMEM
  • VALU

May also be in a clause ("clause internal instructions"):
  • S_DELAY_ALU is legal inside a clause (internal) but is pointless.
      ◦ S_DELAY_ALU must not occur within a VALU clause.
  • S_NOP and S_SLEEP may be used inside a clause, but the first instruction of the clause must be the clause-
    type instruction (ALU, memory).

Cannot be in a clause:
  • Instructions of a different type those of the clause type are illegal
  • S_CLAUSE
  • S_ENDPGM
  • SALU, Export, branch, message, GDS, lds_param_load, lds_direct_load
  • S_WAITCNT, S_WAIT_IDLE, S_WAIT_DEPCTR

If the first instruction in a VALU clause has EXEC==0, then the clause is ignored and instructions are issued as
if there were no clause. If the VALU clause starts with EXEC!=0 but EXEC becomes zero in the middle of the
clause, the clause continues until the last instruction of the specified clause.

If an S_DELAY_ALU is needed before starting a clause, the order must be:

      S_DELAY_ALU // must not come immediately after S_CLAUSE - that inst declares clause type
      S_CLAUSE
      <first instruction in clause>

If the first instruction after S_CLAUSE is skipped (e.g. due to EXEC==0, or VMEM-load skipped due to EXEC==0
and VMcnt==0) then then a clause is not started. Subsequent instructions within what would have been the
clause that are not skipped and are still executed but individually, not as part of a clause.

5.2.1. Clause Breaks
The following conditions can break a clause:

 1. VALU exception (trap) breaks a VALU clause
 2. Host commands to wave (halt, resume, single step, etc) breaks all active clauses.
    Context-save breaks clauses of affected waves.
    This allows the host to read and write SGPRs & VGPRs while debugging. If clauses were not broken by host
    commands, the GPRs could not be read from waves other than the one currently in a clause.
    If a wave halts or is kill, its clauses are ended.
 3. Any action that cause a wave to jump to its trap handler breaks clause (includes context-save).
    A wave entering HALT (including for host-initiated single-step) may break clauses.

5.3. Send Message Types
S_SENDMSG is used to send messages to fixed function hardware, the host, or to request that a value be
returned to the wave. S_SENDMSG encodes the message type in the SIMM16 field and the message payload in

M0. S_SENDMSG_RTN encodes the message type in the SSRC0 field (does not read an SGPR), the payload (if
any) in M0, and the destination SGPR in SDST.

Completion is tracked with LGKMcnt.

The table below lists the messages that can be generated using the S_SENDMSG command.

S_SENDMSG_RTN_B* instructions return data to the shader: increment LGKMcnt by 2, and then decrement by
1 when the messages goes out, and by another 1 when the data returns. This allows the user to simply use
"s_waitcnt LGKMcnt==0" to wait for the data to be returned.

All message codes not listed are reserved (illegal).

                                           Table 15. S_SENDMSG Messages
Message                SIMM16       Payload
                       [7:0]
Reserved               0x00         Reserved
Interrupt              0x01         Software-generated interrupt. M0[23:0] carries user data. ID’s are also sent (wave_id,
                                    cu_id, etc.)
HS TessFactor          0x02         Indicates HS tessellation factor is all zero or one for all patches in this HS work-group.
                                    Data from M0[0]: 0 = "all threads have tess factor of zero", 1 = "all threads have a tess
                                    factor of one". This message is optional, but do not send more than once or from any
                                    shader stage other than HS.
Dealloc VGPRs          0x03         Deallocate all VGPRs for this wave, allowing another wave to allocate these VGPRs
                                    before this wave ends. Use only when next instruction is S_ENDPGM. Typically used
                                    when a shader is waiting memory-write-acknowledgments before ending.
GS alloc req           0x09         Request GS space in parameter cache. M0[9:0] = number of vertices, M0[22:12] =
                                    number of primitives. Response: a GS-alloc response to non-zero requests (broadcast to
                                    work-group).

S_SENDMSG_RTN is used to send messages that return a value to the wave. The instruction specifies which
SGPR receives the data in SDST field. The message is encoded in SSRC0 (in the instruction field, not in an
SGPR).

                                         Table 16. S_SENDMSG_RTN Messages
Message                SSRC0        Payload
Get Doorbell ID        0x80         Get the doorbell ID associated with this wave.
                                    (does not exist for ME0. Return 0x0bad. Also returns 0x0bad for invalid pipeID or
                                    queueID).
Get Draw ID            0x81         Get the Draw or dispatch ID associated with this wave.
Get TMA                0x82         Get the Trap Memory Address: [31:0] or [63:0] depending on the request size.
Get REALTIME           0x83         Get the value of the constant frequency (REFCLK) time counter: [31:0] or [63:0]
                                    depending on the request size.
Save wave              0x84         Used in context switching in indicate this wave is ready to be context saved.
                                    Only the trap handler can send this message (user shaders have this converted to
                                    MSG_ILLEGAL_RTN).
Get TBA                0x85         Gets the Trap Base Address [31:0] or [63:0] depending on request size
MSG_ILLEGAL _RTN       0xFF         Illegal message with data return to wave
