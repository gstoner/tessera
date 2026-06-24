# 5.3. Send Message Types

> RDNA3 ISA — pages 50–50

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
Message                SIMM16      Payload
                       [7:0]
Reserved               0x00        Reserved
Interrupt              0x01        Software-generated interrupt. M0[23:0] carries user data. ID’s are also sent (wave_id,
                                   cu_id, etc.)
HS TessFactor          0x02        Indicates HS tessellation factor is all zero or one for all patches in this HS work-group.
                                   Data from M0[0]: 1 = "all are zero or one". This message is optional, but do not send
                                   more than once or from any shader stage other than HS.
Dealloc VGPRs          0x03        Deallocate all VGPRs for this wave, allowing another wave to allocate these VGPRs
                                   before this wave ends. Use only when next instruction is S_ENDPGM. Typically used
                                   when a shader is waiting memory-write-acknowledgments before ending.
GS alloc req           0x09        Request GS space in parameter cache. M0[9:0] = number of vertices, M0[22:12] =
                                   number of primitives. Response: a GS-alloc response to non-zero requests (broadcast to
                                   work-group).

S_SENDMSG_RTN is used to send messages that return a value to the wave. The instruction specifies which
SGPR receives the data in SDST field. The message is encoded in SSRC0 (in the instruction field, not in an
SGPR).

                                       Table 16. S_SENDMSG_RTN Messages
Message                SSRC0       Payload
Get Doorbell ID        0x80        Get the doorbell ID associated with this wave.
                                   (does not exist for ME0. Return 0x0bad. Also returns 0x0bad for invalid pipeID or
                                   queueID).
Get Draw ID            0x81        Get the Draw or dispatch ID associated with this wave.
Get TMA                0x82        Get the Trap Memory Address: [31:0] or [63:0] depending on the request size.
