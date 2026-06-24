# Flags

> Micro Engine Scheduler ISA — pages 30–30

                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    gang_context_addr - Gang context address for target queue to be suspended

                    •    suspend_fence_addr – MC address for suspend completion fence

                    •    suspend_fence_value - Suspend fence ID

                    •    doorbell_offset - Doorbell offset for target queue to be suspended. Only used if no flag is
                         set

                    •    gang_context_array_index - Gang context array index for target queue to be suspended.
                         Valid only when MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is set
                 The following fields are only valid for Windows OS preemption.

                    •    return_value - Obsolete

                    •    sch_id –Scheduler ID for target engine to be suspended

                    •    legacy_uq_type – Queue type for target engine to be suspended (GFX/compute/SDMA)

                    •    legacy_uq_priority_level – Priority level to be suspended

Flags
                    •    suspend_all_gangs – Not currently supported

MES_SCH_API_RESUME
                 The KMD uses this API to resume a single queue suspended by MES_SCH_API_SUSPEND, or
                 resume scheduling after reset.
                 (Used in the following DDIs in Windows OS: DxgkDdiResumeContext,
                 DxgkDdiResumeHwEngine.)
             union MESAPI__RESUME
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         /* false - resume all gangs; true - specified gang */
                         struct
                         {
                              uint32_t                   resume_all_gangs : 1;
                              uint32_t                   reserved : 31;
                         };

April 2024                                                                                                    30
