# Flags

> Micro Engine Scheduler ISA — pages 21–21

Flags
             •   disable_reset – Disable MES automatic hang detection

             •   use_different_vmid_compute – Scheduler assigns different VMIDs for GFX and compute
                 of the same process

             •   disable_mes_log– Disables MSFT GPU hardware scheduling log

             •   apply_mmhub_pgvm_invalidate_ack_loss_wa – Obsolete

             •   apply_grbm_remote_register_dummy_read_wa – Obsolete

             •   second_gfx_pipe_enabled – Enables 2nd GFX pipe

             •   enable_level_process_quantum_check – Enable an optimization that jumps out of the
                 scheduling loop to handle an API event

             •   legacy_sch_mode – Set to 1 on the older OSes that do not understand or support the
                 GPU hardware scheduling.

             •   disable_add_queue_wptr_mc_addr – If set to 1, the scheduler uses part of memory queue
                 descriptor (MQD) memory for wptr poll memory. Otherwise, scheduler use the address
                 passed in ADD_QUEUE API (see MES_SCH_API_ADD_QUEUE for details)

             •   enable_mes_event_int_logging – Debug only. Enables MES internal event/interrupt/API
                 logging

             •   enable_reg_active_poll – Controls how the scheduler polls queue's active bit. 1: poll HQD
                 register; 0: poll MQD memory

             •   use_disable_queue_in_legacy_uq_preemption – Set to 1 to allow the scheduler to use
                 disable_queue bit in MQD for OS preemption

             •   send_write_data – Set to 1 for the scheduler to send a write_date packet to write a fence
                 following each KIQ packet

             •   os_tdr_timeout_override – Enables unmap timeout overwrite

             •   use_rs64mem_for_proc_gang_ctx – Enables scheduler optimization that puts the process
                 context and gang context into the MES scheduler local memory

             •   use_add_queue_unmap_flag_addr – If set to 1, the scheduler uses MC address passed in
                 MES_SCH_API_ADD_QUEUE for queue unmap status. Else, scheduler will use the MQD
                 memory

             •   enable_mes_sch_stb_log – Enables MES to log into Smart Trace Buffer

April 2024                                                                                           21
