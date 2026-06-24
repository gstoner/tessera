# Flags

> Micro Engine Scheduler ISA — pages 25–25

                 local memory; valid only when
                 MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is True

             •   gang_context_array_index – The index of the gang context array in scheduler's local
                 memory; valid only when
                 MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is True

             •   pipe_id – Used to map a kernel queue; the Pipe ID of the kernel queue

             •   queue_id – Used to map a kernel queue; the Queue ID of the kernel queue

             •   alignment_mode_setting – The shader alignment mode to be programmed in
                 SH_MEM_CONFIG

             •   unmap_flag_addr – The MC address for queue unmap status memory. Only valid when
                 MES_SCH_API_SET_HW_RSRC. use_add_queue_unmap_flag_addr is set

Flags

             •   paging – The queue belonging to the paging process

             •   debug_vmid – Process requires the debug vmid (used by RGP (Radeon GFX Profiling)
                 tool

             •   program_gds – Process uses GDS

             •   is_gang_suspended – A queue's context in suspended state to prevent scheduling of a
                 queue

             •   is_tmz_queue – Obsolete

             •   map_kiq_utility_queue – Obsolete

             •   is_kfd_process – Queue belonging to the KFD process

             •   trap_en – Enables trap for shader debugger

             •   is_aql_queue – The AQL queue

             •   map_legacy_kq – The kernel queue

             •   exclusively_scheduled – Supports cooperative launch

             •   is_long_running – Indicates that the queue has a long running compute job

             •   is_dwm_queue – Indicates that the queue belongs to the DWM process

April 2024                                                                                             25
