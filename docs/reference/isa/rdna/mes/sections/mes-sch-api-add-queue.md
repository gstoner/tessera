# MES_SCH_API_ADD_QUEUE

> Micro Engine Scheduler ISA — pages 22–24

MES_SCH_API_ADD_QUEUE
                 The KMD uses this API to add a use queue into the scheduler's internal structure to schedule it
                 on GPU hardware.
             union MESAPI__ADD_QUEUE
             {
                    struct
                    {
                        union MES_API_HEADER            header;
                        uint32_t                        process_id;
                        uint64_t                        page_table_base_addr;
                        uint64_t                        process_va_start;
                        uint64_t                        process_va_end;
                        uint64_t                        process_quantum;
                        uint64_t                        process_context_addr;
                        uint64_t                        gang_quantum;
                        uint64_t                        gang_context_addr;
                        uint32_t                        inprocess_gang_priority;
                        enum MES_AMD_PRIORITY_LEVEL gang_global_priority_level;
                        uint32_t                        doorbell_offset;
                        uint64_t                        mqd_addr;
                     uint64_t                           wptr_addr;    //From MES_API_VERSION 2, mc addr is
             expected for wptr_addr
                        uint64_t                        h_context;
                        uint64_t                        h_queue;
                        enum MES_QUEUE_TYPE             queue_type;
                        uint32_t                        gds_base;
                        uint32_t                        gds_size;
                        uint32_t                        gws_base;
                        uint32_t                        gws_size;
                        uint32_t                        oa_mask;
                        uint64_t                        trap_handler_addr;
                        uint32_t                        vm_context_cntl;
                        struct
                        {
                             uint32_t paging        : 1;
                             uint32_t debug_vmid    : 4;
                             uint32_t program_gds : 1;
                             uint32_t is_gang_suspended : 1;
                             uint32_t is_tmz_queue : 1;

April 2024                                                                                                  22

                            uint32_t map_kiq_utility_queue : 1;
                            uint32_t is_kfd_process : 1;
                            uint32_t trap_en : 1;
                            uint32_t is_aql_queue : 1;
                            uint32_t skip_process_ctx_clear : 1;
                            uint32_t map_legacy_kq : 1;
                            uint32_t exclusively_scheduled : 1;
                            uint32_t is_long_running : 1;
                            uint32_t is_dwm_queue : 1;
                            uint32_t is_video_blit_queue : 1;
                            uint32_t reserved     : 14;
                       };
                       struct MES_API_STATUS         api_status;
                       uint64_t                      tma_addr;
                       uint32_t                      sch_id;
                       uint64_t                      timestamp;
                       uint32_t                      process_context_array_index;
                       uint32_t                      gang_context_array_index;
                       uint32_t                      pipe_id;      //used for mapping legacy kernel queue
                       uint32_t                      queue_id;
                       uint32_t                      alignment_mode_setting;
                     uint64_t                        unmap_flag_addr; //Used for letting driver know queue
             is unmapped, mc addr is expected
                  };
                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                  •    process_id – Process ID that appears in the IH Cookie as pasid. The KMD assigns unique
                       process ID to each process

                  •    page_table_base_addr – Page table base address of the process, and is programmed in
                       VM_CONTEXTx_PAGE_TABLE_BASE_LO/HI registers

                  •    process_va_start – Starting VA that’s covered by the process’s page table. Programmed
                       in VM_CONTEXTx_PAGE_TABLE_START_LO/HI

                  •    process_va_end – End VA that’s covered by the process’s page table. Programmed in
                       VM_CONTEXTx_PAGE_TABLE_END_LO/HI

                  •    process_quantum – Measured in 100ns units. Indicates the minimum time a process is
                       allowed to run on the GPU

April 2024                                                                                              23

             •   process_context_addr – The memory where process specific information is saved. The
                 scheduler owns the format of content saved in this memory. The size of the process
                 context is defined in mes_api_def.h

             •   gang_quantum – Measured in 100ns units. Indicates the minimum amount of time a gang
                 runs on the GPU

             •   gang_context_addr – memory where gang specific information is saved. Scheduler owns
                 the format of content saved in this memory. The size of this memory is defined in the
                 mes_api_def.h

             •   inprocess_gang_priority – The priority number assigned to the gang relative to other
                 gangs within the same process

             •   gang_global_priority_level – The global priority level assigned to the gang. All queues
                 within a gang share this priority level

             •   doorbell_offset – The doorbell offset (DWORD offset, i.e bits[27:2]) assigned to the
                 queue

             •   mqd_addr – The MC address of queue's MQD memory

             •   wptr_addr – If MES_SCH_API_SET_HW_RSRC.disable_add_queue_wptr_mc_addr is set,
                 GPUVA of wptr poll memory. Else, it’s the MC address of wptr poll memory

             •   h_context – OS handle of the context

             •   h_queue – OS handle of the queue

             •   queue_type – GFX/compute/SDMA

             •   gds_base/size – GDS base/size

             •   gws_base/size – GWS base/size

             •   oa_mask – OA mask

             •   trap_handler_addr – CWSR trap handler GPU VA

             •   tma_addr – CWSR TMA GPU VA

             •   vm_context_cntl – Programmed in VM_CONTEXTx_CNTL

             •   sch_id – The scheduler ID of the engine node belonging to the queue

             •   timestamp – The CPU time stamp of when driver submits this packet to the ring. Used
                 for debugging only.

             •   process_context_array_index – The index of the process context array in scheduler's

April 2024                                                                                              24
