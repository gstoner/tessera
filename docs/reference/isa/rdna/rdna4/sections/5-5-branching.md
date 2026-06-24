# 5.5. Branching

> RDNA4 ISA — pages 57–57

increment KMcnt by 2, and then decrement by 1 when the messages goes out, and by another 1 when the data
returns. This allows the user to use "S_WAIT_KMCNT==0" to wait for the data to be returned.

Completion is tracked with KMcnt.

The table below lists the messages that can be generated using the S_SENDMSG command.
All message codes not listed are reserved (illegal).

                                             Table 21. S_SENDMSG Messages
Message                SIMM16         Payload
                       [7:0]
Reserved               0x00           Reserved
INTERRUPT              0x01           Software-generated interrupt. M0[23:0] carries user data. ID’s are also sent (wave_id,
                                      wgp_id, etc.)
HS_TESSFACTOR          0x02           Indicates HS tessellation factor is all zero or one for all patches in this HS work-group.
                                      Data from M0[0]: 0 = "all threads have tess factor of zero", 1 = "all threads have a tess
                                      factor of one". This message is optional, but do not send more than once or from any
                                      shader stage other than HS.
DEALLOC_VGPRS          0x03           Deallocate all VGPRs and scratch memory for this wave, allowing another wave to
                                      allocate these VGPRs before this wave ends. Typically used immediately before
                                      S_ENDPGM when a shader is waiting for memory-write-acknowledgments before
                                      ending.
GS_ALLOC_REQ           0x09           Request GS space in parameter cache. M0[9:0] = number of vertices, M0[22:12] =
                                      number of primitives. Response: a GS-alloc response to non-zero requests (broadcast to
                                      work-group).

S_SENDMSG_RTN is used to send messages that return a value to the wave. The instruction specifies which
SGPR receives the data in SDST field. The message is encoded in SSRC0 (in the instruction field, not in an
SGPR).

                                          Table 22. S_SENDMSG_RTN Messages
Message                        SIMM16 Payload
                               [7:0]
RTN_GET_DOORBELL               0x80         Get the doorbell ID associated with this wave.
RTN_GET_DDID                   0x81         Get the Draw or dispatch ID associated with this wave.
RTN_GET_TMA                    0x82         Get the Trap Memory Address: [31:0] or [63:0] depending on the request size.
RTN_GET_REALTIME               0x83         Get the value of the constant frequency (REFCLK) time counter: [31:0] or [63:0]
                                            depending on the request size.
RTN_SAVE_WAVE                  0x84         Used in context switching to indicate this wave is ready to be context saved.
                                            Only the trap handler can send this message (user shaders have this converted
                                            to MSG_ILLEGAL_RTN).
RTN_GET_TBA                    0x85         Gets the Trap Base Address [31:0] or [63:0] depending on request size
RTN_GET_SE_HW_ID               0x87         Gets the logical shader-engine IDs, returned as:
                                            data[3:0] = SE_ID; data[11:8] = AID_ID
RTN_ILLEGAL_MSG                0xFF         Illegal message with data return to wave

5.5. Branching
Branching is done using one of the following scalar ALU instructions and affects the entire wave, not just
individual work-items. "SIMM16" is a sign-extended 16 bit integer constant, treated as a DWORD offset for
