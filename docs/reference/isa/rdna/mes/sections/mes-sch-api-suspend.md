# MES_SCH_API_SUSPEND

> Micro Engine Scheduler ISA — pages 29–29

MES_SCH_API_SET_GANG_PRIORITY_LEVEL
                 This API is not currently supported.

MES_SCH_API_SUSPEND
                 When MES_SCH_API_SET_HW_RSRC.legacy_sch_mode is set, the KMD uses this API to
                 suspend a single queue to prevent it from being scheduled for a single engine in Windows OS
                 preemption.

                 (Used in the following DDIs in Windows: DxgkDdiSuspendContext, DxgkDdiPreemptCommand.)
             union MESAPI__SUSPEND
             {
                    struct
                    {
                        union MES_API_HEADER             header;
                        /* false - suspend all gangs; true - specific gang */
                        struct
                        {
                             uint32_t                      suspend_all_gangs : 1;
                             uint32_t                      reserved : 31;
                        };
                        /* gang_context_addr is valid only if suspend_all = false */

                        uint64_t gang_context_addr;

                        uint64_t                           suspend_fence_addr;
                        uint32_t                           suspend_fence_value;

                        struct MES_API_STATUS              api_status;

                        union
                        {
                             uint32_t return_value; // to be removed
                         uint32_t sch_id;               //keep the old return_value temporarily for
             compatibility
                        };
                        uint32_t                         doorbell_offset;
                        uint64_t                         timestamp;
                        enum MES_QUEUE_TYPE              legacy_uq_type;
                        enum MES_AMD_PRIORITY_LEVEL legacy_uq_priority_level;
                        uint32_t                         gang_context_array_index;

April 2024                                                                                                29
