# MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE

> Micro Engine Scheduler ISA — pages 46–46

                       struct
                       {
                           uint32_t single_memop : 1;     // SQ_DEBUG.single_memop
                           uint32_t single_alu_op : 1; // SQ_DEBUG.single_alu_op
                           uint32_t reserved : 30;
                       }flags;
                       uint32_t u32All;
                  };
                  uint32_t spi_gdbg_per_vmid_cntl;
                  uint32_t tcp_watch_cntl[4]; // TCP_WATCHx_CNTL
                  uint32_t trap_en;
             };

                   •   single_memop - SINGLE_MEMOP setting in SQ_DEBUG register

                   •   single_alu_op - SINGLE_ALU_OP setting in SQ_DEBUG register

                   •   process_context_addr - Memory where process specific context is saved

                   •   spi_gdbg_per_vmid_cntl - Setting for SPI_GDBG_PER_VMID_CNTL register

                   •   tcp_watch_cntl[4] - Setting for TCP_WATCHx_CNTL registers

                   •   trap_en - TRAP_EN setting in SQ_SHADER_TBA_HI register

MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE
              KMD uses this API as a workaround for aggregate doorbell. Meant to be called when an
              unmapped queue has a new submission. Notifies MES that target priority level has new work
              and MES will try to schedule queues of this level.
             enum MES_AMD_PRIORITY_LEVEL queue_sch_level;

                   •   queue_sch_level - Target priority level that has new work

MESAPI_MISC__NOTIFY_TO_UNMAP_PROCESSES
              The KMD uses this API to request the MES to unmap queues for all processes.

April 2024                                                                                           46
