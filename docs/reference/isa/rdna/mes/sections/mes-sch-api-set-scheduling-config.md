# MES_SCH_API_SET_SCHEDULING_CONFIG

> Micro Engine Scheduler ISA — pages 27–27

                            uint32_t reserved01                   : 1;
                            uint32_t unmap_kiq_utility_queue      : 1;
                            uint32_t preempt_legacy_gfx_queue : 1;
                            uint32_t unmap_legacy_queue           : 1;
                            uint32_t reserved                     : 28;
                       };
                       struct MES_API_STATUS          api_status;
                       uint32_t                       pipe_id;
                       uint32_t                       queue_id;
                       uint64_t                       tf_addr;
                       uint32_t                       tf_data;
                       enum MES_QUEUE_TYPE            queue_type;
                       uint64_t                       timestamp;
                       uint32_t                       gang_context_array_index;
                  };
                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                  •    doorbell_offset – Doorbell offset [DWORD offset, bits [27:2]] of the queue to be
                       removed

                  •    gang_context_addr – The gang’s context address that maintains the info of all queues
                       belonging to that gang

                  •    pipe/queue_id – Used to remove a kernel queue (i.e., queues are managed by KMD); pipe
                       ID/queue ID of the kernel queue being removed

                  •    tf_addr/data – Trailing fence address and value for OS preemption

                  •    queue_type – Gfx/compute/SDMA

                  •    gang_context_array_index – Index of the gang context array in scheduler's local memory;
                       valid only when MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is true

Flags
                  •    unmap_kiq_utility_queue – Obsolete

                  •    preempt_legacy_gfx_queue – Indicates that this is for OS preemption

                  •    unmap_legacy_queue – Indicates that this is for kernel queue

MES_SCH_API_SET_SCHEDULING_CONFIG
              Corresponds to Windows DDI DxgkDdiSetProrityBands.

              Sets up process quantum and other related information during bootup for each priority band.
              The MES scheduler uses this information for scheduling decisions.

April 2024                                                                                                27
