# Interrupt history

> Micro Engine Scheduler ISA — pages 49–49

                       •   event_log_id – events that MES scheduler sends to CP; Defined in enum
                           MES_SCH_EVT_LOG_ID

                       •   doorbell_offset – doorbell offset of the queue for which the event is sent

                       •   time_before_call – GPU timestamp at which scheduler sends event to CP

                       •   time_after_call – GPU timestamp at which CP finishes processing the event

                       •   status – if CP processing event is successful or not, 1: success, 0: otherwise

                       •   queue_type – queue type (gfx/compute/sdma) of the queue for which the event is sent

Interrupt history

                 Each entry in interrupt_history array has the following format:
             struct MES_SCH_INTR_HIST_INFO
             {
                   enum MES_SCH_INTR_ID intr_id;
                   uint64_t                    time_trace;
                   struct MES_SCH_INTR_CB_DATA intr_callback;
             };
             enum MES_SCH_INTR_ID
             {
                   MES_INTR_ME_0 = 0,
                   MES_INTR_ME_1 = 1,
                   MES_INTR_PACKET = 2,
                   MES_INTR_TIMER = 3,
                   MES_INTR_AGGREAGATE_DOORBELL = 4
             };

                       •   intr_id – interrupt ID defined in enum MES_SCH_INTR_ID

                       •   time_trace – GPU timestamp at which MES scheduler receives the interrupt

                       •   intr_callback – Interrupt call back data defined in struct MES_SCH_INTR_CB_DATA
                           below

             struct MES_SCH_INTR_CB_DATA
             {
                   union
                   {

April 2024                                                                                                   49
