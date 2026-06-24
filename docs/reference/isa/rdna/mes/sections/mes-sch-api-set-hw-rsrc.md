# MES_SCH_API_SET_HW_RSRC

> Micro Engine Scheduler ISA — pages 18–20

MES_SCH_API_SET_HW_RSRC
                 This is the first API that KMD submits to MES during initialization.

                 It provides list of hardware resources (hardware queues, virtual memory ID (VMID), etc.) to be
                 managed by the scheduler and configuration flags (OS dependent features, workaround, etc.).

             enum { MAX_COMPUTE_PIPES = 8 };
             enum { MAX_GFX_PIPES           = 2 };
             enum { MAX_SDMA_PIPES          = 2 };

             enum MES_AMD_PRIORITY_LEVEL
             {
                    AMD_PRIORITY_LEVEL_LOW           = 0,
                    AMD_PRIORITY_LEVEL_NORMAL        = 1,
                    AMD_PRIORITY_LEVEL_MEDIUM        = 2,
                    AMD_PRIORITY_LEVEL_HIGH          = 3,
                    AMD_PRIORITY_LEVEL_REALTIME      = 4,

                    AMD_PRIORITY_NUM_LEVELS
             };

             union MESAPI_SET_HW_RESOURCES
             {
                    struct
                    {
                        union MES_API_HEADER       header;
                        uint32_t                                 vmid_mask_mmhub;
                        uint32_t                                 vmid_mask_gfxhub;
                        uint32_t                                 gds_size;
                        uint32_t                                 paging_vmid;
                        uint32_t                                 compute_hqd_mask[MAX_COMPUTE_PIPES];
                        uint32_t                                 gfx_hqd_mask[MAX_GFX_PIPES];
                        uint32_t                                 sdma_hqd_mask[MAX_SDMA_PIPES];
                        uint32_t                                 aggregated_doorbells[AMD_PRIORITY_NUM_LEVELS];
                        uint64_t                                 g_sch_ctx_gpu_mc_ptr;
                        uint64_t                                 query_status_fence_gpu_mc_ptr;
                        uint32_t                                 gc_base[MES_MAX_HWIP_SEGMENT];

April 2024                                                                                                  18

                       uint32_t                                mmhub_base[MES_MAX_HWIP_SEGMENT];
                       uint32_t                                osssys_base[MES_MAX_HWIP_SEGMENT];
                       struct MES_API_STATUS       api_status;
                       union
                       {
                            struct
                            {
                                 uint32_t disable_reset : 1;
                                 uint32_t use_different_vmid_compute : 1;
                                 uint32_t disable_mes_log      : 1;
                                 uint32_t apply_mmhub_pgvm_invalidate_ack_loss_wa : 1;
                                 uint32_t apply_grbm_remote_register_dummy_read_wa : 1;
                                 uint32_t second_gfx_pipe_enabled : 1;
                                 uint32_t enable_level_process_quantum_check : 1;
                                 uint32_t legacy_sch_mode : 1;
                                 uint32_t disable_add_queue_wptr_mc_addr : 1;
                                 uint32_t enable_mes_event_int_logging : 1;
                                 uint32_t enable_reg_active_poll : 1;
                                 uint32_t use_disable_queue_in_legacy_uq_preemption : 1;
                                 uint32_t send_write_data : 1;
                                 uint32_t os_tdr_timeout_override : 1;
                                 uint32_t use_rs64mem_for_proc_gang_ctx : 1;
                                 uint32_t use_add_queue_unmap_flag_addr : 1;
                                 uint32_t enable_mes_sch_stb_log : 1;
                                 uint32_t reserved : 15;
                            };
                            uint32_t uint32_all;
                       };
                       uint32_t                                 oversubscription_timer;
                       uint64_t                                 doorbell_info;
                       uint64_t                                 event_intr_history_gpu_mc_ptr;
                       uint64_t                                 timestamp;
                       uint32_t                                 os_tdr_timeout_in_sec;
                  };
                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                  •    vmid_mask_gfxhub – Bit mask of VMIDs in GC hub that are available for scheduler to
                       manage. Each bit position indicates the availability of the corresponding VMID, e.g., 0x6

April 2024                                                                                                  19

                 means VMID 1 and 2 are available

             •   vmid_mask_mmhub – Obsolete

             •   gds_size – Size of the global data storage (GDS) on the chip

             •   paging_vmid – VMID that driver assigns to paging process (excluded from
                 vmid_mask_gfxhub)

             •   compute_hqd_mask – Per pipe bit mask of compute hardware queue descriptors (HQD)
                 that are managed by scheduler. Each bit position indicates the availability of
                 corresponding compute HQD on the particular pipe, e.g., 0x3 means compute HQD 0
                 and 1 of the pipe are available

             •   gfx_hqd_mask - Per pipe bit mask of graphics (GFX) HQDs that are managed by
                 scheduler. Each bit position indicates the availability of corresponding GFX HQD on the
                 particular pipe, e.g., 0x3 means GFX queue 0 and 1 of the pipe are available

             •   sdma_hqd_mask – Per pipe bit mask of SDMA HQDs that are managed by scheduler.
                 Each bit position indicates the availability of corresponding SDMA HQD on the particular
                 pipe, e.g., 0x3 means SDMA queue 0 and 1 of the pipe are available

             •   aggregated_doorbells – Offsets of aggregated doorbells. Value of 0XFFFFFFFF indicates
                 invalid offset

             •   g_sch_ctx_gpu_mc_ptr – Obsolete

             •   query_status_fence_gpu_mc_ptr – MC address of query_status packet fence memory.

             •   gc_base – HWIP base for GC block

             •   mmhub_base – HWIP base for MM block

             •   ossys_base – HWIP base for OSSYS block

             •   oversubscription_timer – Duration in micro-second of timer when oversubscription
                 happens. Scheduler wakes up to check if any unmapped queue has new work when timer
                 is up

             •   doorbell_info – Debug only. Memory to hold aggregated doorbell counter

             •   event_intr_history_gpu_mc_ptr – Debug only. MC address to hold MES
                 event/interrupt/API history log

             •   os_tdr_timeout_in_sec – Unmap timeout value in seconds. The driver is able to use this
                 to overwrite the default unmap time out value of 2 seconds. Only valid when
                 os_tdr_timeout_override is set

April 2024                                                                                          20
