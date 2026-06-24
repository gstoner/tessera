# MES_SCH_API_MISC

> Micro Engine Scheduler ISA — pages 42–42

                   uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                       •   gang_context_addr - Gang context address of master queue

                       •   slave_gang_context_addr - Gang context address of slave queue

                       •   gang_context_array_index - Gang context array index of master queue. Valid only when
                           use_rs64mem_for_proc_gang_ctx is set in mes_sch_api_set_hw_rsrc

                       •   slave_gang_context_array_index - Gang context array index of slave queue. Valid only
                           when use_rs64mem_for_proc_gang_ctx is set in mes_sch_api_set_hw_rsrc

MES_SCH_API_MISC
                 This API contains miscellaneous non-scheduling functionalities. Each functionality has a sub-
                 opcode and corresponding structures.
             union MESAPI__MISC
             {
                   struct
                   {
                           union MES_API_HEADER     header;
                           enum MESAPI_MISC_OPCODE opcode;
                           struct MES_API_STATUS    api_status;

                           union
                           {
                                struct WRITE_REG write_reg;
                                struct INV_GART inv_gart;
                                struct QUERY_STATUS query_status;
                                struct READ_REG read_reg;
                                struct WAIT_REG_MEM wait_reg_mem;
                                struct SET_SHADER_DEBUGGER set_shader_debugger;
                                enum MES_AMD_PRIORITY_LEVEL queue_sch_level;
                                uint32_t data[MISC_DATA_MAX_SIZE_IN_DWORDS];
                           };
                           uint64_t                 timestamp;
                           uint32_t                 doorbell_offset;
                           uint32_t                 os_fence;
                   };

                   uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];

April 2024                                                                                                   42
