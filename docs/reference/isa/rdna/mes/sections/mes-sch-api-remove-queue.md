# MES_SCH_API_REMOVE_QUEUE

> Micro Engine Scheduler ISA — pages 26–26

                    •    is_video_blit_queue - Indicates the queue is a video blit queue

MES_SCH_API_AMD_LOG

                 Copy MES_SCH_CONTEXT to AMGLOG specified memory location for TDR analysis.

             union MESAPI_AMD_LOG
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         uint64_t                        p_buffer_memory;
                         uint64_t                        p_buffer_size_used;
                         struct MES_API_STATUS           api_status;
                         uint64_t                        timestamp;
                    };
                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    p_buffer_memory - Pointer to amdlog buffer

                    •    p_buffer_size_used - Not used, buffer size is equal to sizeof(struct MES_SCH_CONTEXT)

MES_SCH_API_REMOVE_QUEUE
                 The KMD uses this API to remove a user queue from the scheduler's internal structure.
                 If the queue being removed is the last queue in the gang, all information related to the gang is
                 removed from the scheduler context.
                 If the removed gang is the last in the process, the process information is removed from the
                 scheduler context.
             union MESAPI__REMOVE_QUEUE
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         uint32_t                        doorbell_offset;
                         uint64_t                        gang_context_addr;
                         struct
                         {

April 2024                                                                                                     26
