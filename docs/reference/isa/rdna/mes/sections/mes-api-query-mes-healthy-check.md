# MES_API_QUERY_MES__HEALTHY_CHECK

> Micro Engine Scheduler ISA — pages 37–37

                       •   proc_ctx_array_size_addr - Memory address where MES will write process context array
                           size

                       •   gang_ctx_array_size_addr - Memory address where MES will write gang context array
                           size

MES_API_QUERY_MES__HEALTHY_CHECK
                 The KMD uses this API to check if MES is running and responding.
             struct MES_API_QUERY_MES__HEALTHY_CHECK
             {
                   uint64_t        healthy_addr;
             };

                       •   healthy_addr – Not used. Currently, MES firmware writes fence to the memory to notify
                           KMD that MES is not hang

MES_SCH_API_PROGRAM_GDS
                 The KMD uses this API to request MES for GDS programming for the target process. GDS
                 registers are programmed when VMID is allocated. If VMID is already allocated, registers will be
                 programmed before API returns.
             union MESAPI__PROGRAM_GDS
             {
                   struct
                   {
                           union MES_API_HEADER         header;
                           uint64_t                     process_context_addr;
                           uint32_t                     gds_base;
                           uint32_t                     gds_size;
                           uint32_t                     gws_base;
                           uint32_t                     gws_size;
                           uint32_t                     oa_mask;
                           struct MES_API_STATUS        api_status;
                           uint64_t                     timestamp;
                           uint32_t                     process_context_array_index;
                   };

                   uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                       •   process_context_addr - Memory where process specific information is saved. Scheduler
                           owns the format of this memory. The size of the process context is defined in the

April 2024                                                                                                  37
