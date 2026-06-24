# MES_SCH_API_SET_GANG_SUBMIT

> Micro Engine Scheduler ISA — pages 41–41

                           uint64_t cpg_ctxt_sync_fence_addr;
                           uint32_t cpg_ctxt_sync_fence_value;

                           /* log_seq_time - Scheduler logs the switch seq start/end ts in the IH cookies */
                           union
                           {
                                struct
                                {
                                     uint32_t log_seq_time : 1;
                                     uint32_t reserved : 31;
                                };
                                uint32_t uint32_all;
                           };
                           struct MES_API_STATUS api_status;
                   };

                   uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                       •   new_se_mode – New SE mode to be applied

MES_SCH_API_SET_GANG_SUBMIT
                 The KMD uses this API to pair two queues together for the purpose of gang submission. MES
                 scheduler will guarantee that the paired queues will always be mapped at the same time.
             struct SET_GANG_SUBMIT
             {
                   uint64_t gang_context_addr;
                   uint64_t slave_gang_context_addr;
                   uint32_t gang_context_array_index;
                   uint32_t slave_gang_context_array_index;
             };

             union MESAPI__SET_GANG_SUBMIT
             {
                   struct
                   {
                           union MES_API_HEADER      header;
                           struct MES_API_STATUS     api_status;
                           struct SET_GANG_SUBMIT    set_gang_submit;
                   };

April 2024                                                                                              41
