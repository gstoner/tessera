# Flags

> Micro Engine Scheduler ISA — pages 31–32

                         /* valid only if resume_all_gangs = false */
                         uint64_t                       gang_context_addr;

                         struct MES_API_STATUS          api_status;
                         uint32_t                       doorbell_offset;
                         uint64_t                       timestamp;
                         uint32_t                       gang_context_array_index;
                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    gang_context_addr - Gang context address for target queue to be resumed. Valid only if
                         resume_all_gangs = 0

                    •    gang_context_array_index – Gang context array index for target queue to be resumed.
                         Valid only if resume_all_gangs = 0 and
                         MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx = 1

Flags
                    •    resume_all_gangs - Resume all scheduling. Meant to be called after an engine reset

MES_SCH_API_RESET
                 In Windows, the KMD uses this API for hang detection and reset. The MES scheduler returns a
                 list of doorbell offsets of hung queues. If the list is empty, no hangs are detected.

                 Used in the following Windows DDIs; DxgkDdiQueryEngineStatus, DxgkDdiResetEngine,
                 DxgkDdiResetHwEngine

                 The KMD can also use this API to reset kernel queues by setting reset_legacy_gfx flag.
             union MESAPI__RESET
             {
                    struct
                    {
                         union MES_API_HEADER           header;
                         struct
                         {
                         uint32_t                 reset_queue_only : 1; // Only reset the queue given
             by doorbell_offset (not entire gang)
                         uint32_t                hang_detect_then_reset : 1; // Hang detection first
             then reset any queues that are hung
                             uint32_t                   hang_detect_only : 1; // Only do hang detection (no
             reset)

April 2024                                                                                                    31

                         uint32_t                     reset_legacy_gfx : 1; // Reset HP and LP kernel
             queues not managed by MES
                         uint32_t                use_connected_queue_index : 1; // Fallback to use
             conneceted queue index when CP_CNTX_STAT method fails (gfx pipe 0)
                            uint32_t                  use_connected_queue_index_p1 : 1; // For gfx pipe 1
                            uint32_t                  reserved : 26;
                       };
                       uint64_t                       gang_context_addr;
                       /* valid only if reset_queue_only = true */
                       uint32_t                       doorbell_offset;
                       /* valid only if hang_detect_then_reset = true */
                       uint64_t                       doorbell_offset_addr;
                       enum MES_QUEUE_TYPE            queue_type;
                       //valid only if reset_legacy_gfx = true
                       uint32_t pipe_id_lp;
                       uint32_t queue_id_lp;
                       uint32_t vmid_id_lp;
                       uint64_t mqd_mc_addr_lp;
                       uint32_t doorbell_offset_lp;
                       uint64_t wptr_addr_lp;
                       uint32_t pipe_id_hp;
                       uint32_t queue_id_hp;
                       uint32_t vmid_id_hp;
                       uint64_t mqd_mc_addr_hp;
                       uint32_t doorbell_offset_hp;
                       uint64_t wptr_addr_hp;
                       struct MES_API_STATUS          api_status;
                       uint32_t                       active_vmids;
                       uint64_t                       timestamp;
                       uint32_t                       gang_context_array_index;

                       uint32_t                       connected_queue_index;
                       uint32_t                       connected_queue_index_p1;
                  };
                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                  •    gang_context_addr - Obsolete

                  •    doorbell_offset – Doorbell offset of the queue. Only valid when reset_queue_only = 1

April 2024                                                                                               32
