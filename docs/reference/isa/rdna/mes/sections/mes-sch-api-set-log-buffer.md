# MES_SCH_API_SET_LOG_BUFFER

> Micro Engine Scheduler ISA — pages 34–35

MES_SCH_API_SET_LOG_BUFFER
                 The KMD uses this API to save log buffer information passed from Windows OS DDI
                 DxgkDdiSetSchedulingLogBuffer.
             union MESAPI__SET_LOGGING_BUFFER
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         /* There are separate log buffers for each queue type */
                         enum MES_QUEUE_TYPE             log_type;
                         /* Log buffer GPU Address */
                         uint64_t                        logging_buffer_addr;
                         /* number of entries in the log buffer */
                         uint32_t                        number_of_entries;
                         /* Entry index at which CPU interrupt needs to be signalled */
                         uint32_t                        interrupt_entry;

                         struct MES_API_STATUS           api_status;
                         uint64_t                        timestamp;
                         uint32_t                        vmid;
                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    log_type - Target engine type for this log buffer update (each engine has its own log
                         buffer)

                    •    logging_buffer_addr – GPU virtual address of log buffer

                    •    number_of_entries - Log buffer size

                    •    interrupt_entry - When number of entries logged in the log buffer reaches this log entry
                         index, it raises an interrupt to KMD/OS. The interrupt is meant to give OS advanced
                         warning of when the existing log buffer is going to be filled up so that it can allocate a
                         new log buffer

MES_SCH_API_CHANGE_GANG_PRORITY
                 In the Windows use-case, this API corresponds to DDI
                 DxgkDDiSetContextSchedulingProperties. The Windows OS changes user queue quantum to
                 reflect changes in the owning process's status. For example, when a user’s mouse focus changes

April 2024                                                                                                       34

                 from one process to another.
             union MESAPI__CHANGE_GANG_PRIORITY_LEVEL
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         uint32_t                        inprocess_gang_priority;
                         enum MES_AMD_PRIORITY_LEVEL gang_global_priority_level;
                         uint64_t                        gang_quantum;
                         uint64_t                        gang_context_addr;
                         struct MES_API_STATUS           api_status;
                         uint32_t                        doorbell_offset;
                         uint64_t                        timestamp;
                         uint32_t                        gang_context_array_index;
                         struct
                         {
                              uint32_t                   queue_quantum_scale              : 2;
                              uint32_t                   queue_quantum_duration           : 8;
                              uint32_t                   apply_quantum_all_processes : 1;
                              uint32_t                   reserved                         : 21;
                         };
                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    inprocess_gang_priority - Gang priority within a process, not used in current FW

                    •    gang_global_priority_level - Overall gang priority level, lower priority gangs tend to get
                         preempted for high priority gangs during scheduling

                    •    gang_quantum - Quantum provided by Windows OS, usually 2ms, queue is considered
                         "expired" after its quantum runs out

                    •    doorbell_offset – Obsolete

                    •    gang_context_array_index – index of the gang context array in scheduler's local memory;
                         valid only when MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is set

                    •    queue_quantum_scale – Used by Windows OS

                    •    queue_quantum_duration – Used by Windows OS

                    •    apply_quantum_all_processes – Used by Windows OS

April 2024                                                                                                      35
