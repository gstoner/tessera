# MES_SCH_API_SET_DEBUG_VMID

> Micro Engine Scheduler ISA — pages 38–39

                           mes_api_def.h

                       •   gds_base - GDS base address. Programming for GDS_VMIDx_BASE register

                       •   gds_size - GDS aperture size. Programming for GDS_VMIDx_SIZE register

                       •   gws_base - GWS base. Programming for BASE field in GDS_GWS_VMIDx register

                       •   gws_size - GWS size. Programming for SIZE field in GDS_GWS_VMIDx register

                       •   oa_mask - Bit mask representing the alloc counters allocated VMID can use.
                           Programming for GDS_OA_VMIDx register

                       •   process_context_array_index - Processes context array index for target process. Valid
                           only when MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is set

MES_SCH_API_SET_DEBUG_VMID
                 The KMD uses this API to set up the page table for a process that requests debug VMID for
                 tools like Radeon GPU Profiler (RGP).
                 The user mode driver can request debug VMID, and KMD/MES will allocate a VMID for this
                 process. The page table base registers for this allocated debug VMID will be programed to this
                 process's page table base.
             union MESAPI__SET_DEBUG_VMID
             {
                   struct
                   {
                           union MES_API_HEADER          header;
                           struct MES_API_STATUS         api_status;
                           union
                           {
                                struct
                                {
                                    uint32_t use_gds    : 1;
                                    uint32_t operation : 2;
                                    uint32_t reserved   : 29;
                                }flags;
                                uint32_t u32All;
                           };
                           uint32_t                        reserved;
                           uint32_t                        debug_vmid;
                           uint64_t                        process_context_addr;
                           uint64_t                        page_table_base_addr;
                           uint64_t                        process_va_start;
                           uint64_t                        process_va_end;

April 2024                                                                                                    38

                         uint32_t                        gds_base;
                         uint32_t                        gds_size;
                         uint32_t                        gws_base;
                         uint32_t                        gws_size;
                         uint32_t                        oa_mask;
                         uint64_t                        output_addr; // output addr of the acquired vmid
             value
                         uint64_t                        timestamp;
                         uint32_t                        process_vm_cntl;
                         enum MES_QUEUE_TYPE             queue_type;
                         uint32_t                        process_context_array_index;
                         uint32_t                        alignment_mode_setting;
                  };

                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                     •   debug_vmid - The VMID reserved as the debug VMID, used when operation flag =
                         DEBUG_VMID_OP_RELEASE (2)

                     •   process_context_addr – Memory where process specific context is saved. Scheduler
                         owns the format of this memory. The size of the process context is defined in the
                         mes_api.def.h, this is for the process that requests the debug VMID

                     •   page_table_base_addr – page table base address of the process

                     •   process_va_start - Starting address of the process's VA space

                     •   process_va_end - Ending address of the process's VA space

                     •   gds_base/size – GDS base/size

                     •   gws_base/size – GWS base/size

                     •   oa_mask – OA mask

                     •   output_addr – MES scheduler writes the allocated debug VMID value to this address for
                         driver to read. This is used when operation flag = DEBUG_VMID_OP_ALLOCATE (1)

                     •   process_vm_cntl - Not used

                     •   queue_type – gfx/compute/SDMA

                     •   process_context_array_index – Index of the process context array in scheduler's local
                         memory; valid only when
                         MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is set

                     •   alignment_mode_setting – alignment mode setting to be programmed in
                         SH_MEM_CONFIG

April 2024                                                                                                  39
