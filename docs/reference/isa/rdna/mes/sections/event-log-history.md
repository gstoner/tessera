# Event log history

> Micro Engine Scheduler ISA — pages 48–48

                   };
             };

                       •   api_id – indicates which API command of this entry

                       •   time_before_call – GPU timestamp when MES scheduler starts processing this API
                           command

                       •   timer_after_call – GPU timestamp when MES scheduler finishes processing this API
                           command

                       •   error_code – error code for certain APIs if API processing encounters error. Error code is
                           defined in mes_sch_context.h

                       •   status – 1: API processing is successful; 0: otherwise

Event log history

                 Each entry in event_log_hisotry array has the following format:

             struct MES_SCH_EVT_LOG_HIST_INFO
             {
                   enum MES_SCH_EVT_LOG_ID event_log_id;
                   uint32_t                      doorbell_offset;
                   uint64_t                      time_before_call;
                   uint64_t                      time_after_call;
                   struct
                   {
                           uint32 status : 1;
                           uint32 queue_type : 2;
                           uint32 reserved : 29;
                   };
             };

             enum MES_SCH_EVT_LOG_ID
             {
                   MES_EVT_LOG_MAP_QUEUE = 0,
                   MES_EVT_LOG_UNMAP_QUEUE = 1,
                   MES_EVT_LOG_QUERY_STATUS = 2,
                   MES_EVT_LOG_UNMAP_RESET_QUEUE = 3
             };

April 2024                                                                                                      48
