# MESAPI_MISC__WAIT_REG_MEM

> Micro Engine Scheduler ISA — pages 44–44

                       •   inv_range_size - invalidation range size
                 Note: If inv_range_va_start = 0 or inv_range_size = 0, then MES scheduler invalidates entire
                 range.

MESAPI_MISC__QUERY_STATUS
                 The KMD uses this to trigger an interrupt from KIQ.
             struct QUERY_STATUS
             {
                   uint32_t context_id;
             };

                       •   context_id - Value is copied to CONTEXT_ID in QueryStatus PM4 packet

MESAPI_MISC__READ_REG
                 Perform register read on request of the KMD.
             struct READ_REG
             {
                   uint32_t reg_offset;
                   uint64_t buffer_addr;
                   union
                   {
                           struct
                           {
                               uint32_t read64Bits : 1;
                               uint32_t reserved : 31;
                           }bits;
                           uint32_t all;
                   }option;
             };

                       •   reg_offset - Offset of the register

                       •   buffer_addr – MC address to which MES scheduler writes the register value

                       •   read64Bits - Control bit to enable 64-bit register read (0 = 32-bit, 1 = 64-bit)

MESAPI_MISC__WAIT_REG_MEM
                 The KMD uses this API to request for the MES to wait on specific register values.
             enum WRM_OPERATION
             {

April 2024                                                                                                      44
