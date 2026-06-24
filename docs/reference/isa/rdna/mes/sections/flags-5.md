# Flags

> Micro Engine Scheduler ISA — pages 33–33

                •   doorbell_offset_addr – MC address of memory that holds doorbell offset array. MES
                    scheduler populates this array with offsets for queues that are hung

                •   queue_type - Indicates which engine MES should reset/hang detect
                    (GFX/compute/SDMA)

                •   active_vmids - Workaround to indicate which VMIDs are currently active for
                    CP_CNTX_STAT hang detect method

                •   gang_context_array_index - Obsolete

                •   connected_queue_index - Workaround to indicate which queue is currently connected
                    on GFX Pipe 0. Valid only when use_connected_queue_index = 1

                •   connected_queue_index_p1 - Workaround to indicate which queue is currently
                    connected on GFX Pipe 1. Valid only when use_connected_queue_index_p1 = 1

             The following fields are only valid when reset_legacy_gfx is set and are used in Windows:

                •   pipe_id_lp - Pipe ID for low priority GFX Kernel queue

                •   queue_id_lp - Queue ID for low priority GFX Kernel queue

                •   vmid_id_lp - VMID for low priority GFX Kernel queue

                •   mqd_mc_addr_lp - MQD MC address for low priority GFX Kernel queue

                •   doorbell_offset_lp - Doorbell offset for low priority GFX Kernel queue

                •   wptr_addr_lp - Write pointer poll memory address for low priority GFX Kernel queue

                •   pipe_id_hp - Pipe ID for high priority GFX Kernel queue

                •   queue_id_hp - Queue ID for high priority GFX Kernel queue

                •   vmid_id_hp - VMID for high priority GFX Kernel queue

                •   mqd_mc_addr_hp - MQD MC address for high priority GFX Kernel queue

                •   doorbell_offset_hp - Doorbell offset for high priority GFX Kernel queue

                •   wptr_addr_hp - Write pointer poll memory address for high priority GFX Kernel queue
Flags
                •   reset_queue_only - Reset single queue with no hang detection

                •   hang_detect_then_reset - Performs hang detection, and reset all hung queues. Return
                    doorbell offsets of all hung queues

                •   hang_detect_only - Perform hang detection only. Returns doorbell offsets of all hung
                    queues

                •   reset_legacy_gfx – Resets legacy GFX queue

                •   No flag set - Obsolete. The driver is expected to set one of the above flags

April 2024                                                                                               33
