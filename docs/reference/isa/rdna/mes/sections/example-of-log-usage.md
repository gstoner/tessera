# Example of log usage

> Micro Engine Scheduler ISA — pages 50–51

                       struct
                       {
                           uint32_t                enc_inter : 5;
                           uint32_t                intr_pipe_id : 2;
                           uint32_t                intr_queue_id : 3;
                           uint32_t                reserved1 : 1;
                           uint32_t                action_id : 4;
                           uint32_t                enc_inter_valid : 1;
                           uint32_t                reserved2 : 12;
                           uint32_t                vmid : 4;
                       } inter_encode;
                       uint32_t inter_enc;
                  };
                  union
                  {
                       struct
                       {
                           uint64_t                intr_data : 62;
                           uint64_t                intr_pipe_id : 2;
                       } inter_data_pipe;
                       struct
                       {
                           uint64_t                doorbell_offset : 26;
                           uint64_t                reserved3 : 6;
                           uint64_t                data : 32;
                       } fence;
                       uint64_t inter_data;
                       uint64_t inter_addr;
                  };
             };

Example of log usage

              When KMD reports MES API timeout error message, one may use MES log to understand the
              failure.

              For example, one of the most common MES API timeout error is message 3 timeout. From enum
              MES_SCH_API_OPCODE defined in mes_api_def.h, 3 is MES_SCH_API_REMOVE_QUEUE.
              KMD issues this API to request MES scheduler to remove a user queue. There may be multiple
              reasons of this API failure. From MES log, one can find the most recent entry in api_history array

April 2024                                                                                                50

                 which has api_id MES_API_REMOVE_QUEUE (3). Then, from the error_code (see below), one
                 can check the reason of the error.
             enum MES_SCH_API_REMOVEQUEUE_ERRCODE
             {
                   API_REMOVEQUEUE_NOERROR = 0,
                   API_REMOVEQUEUE_UNMAP_FAIL = 1,
                   API_REMOVEQUEUE_HQDQUEUE_MAP_MISMATCH = 2,
                   API_REMOVEQUEUE_CLEANUP_FAIL = 3,
                   API_REMOVEQUEUE_QUEUE_NOT_FOUND = 4,
                   API_REMOVEQUEUE_NULL_GANG = 5,
             };

                 If error_code is 1, it means when MES scheduler requests CP to unmap the queue, CP failed the
                 unmap request. This usually means the queue being unmapped is in a hang state. As the next
                 debugging step, one need to look for the reason why the queue is hang. In this scenario, in the
                 most recent entry in event_log_history array with event_log_id
                 MES_EVT_LOG_UNAMP_QUEUE, one would see the status bit is 0, which means unmap failure
                 and doorbell_offset field tells which queue has triggered this error.

                 If error_code is not 1, it means error in either MES scheduler firmware or in driver. For example,
                 3 means when MES scheduler cleans up its internal structure, it encounters some issue; 5 means
                 KMD has passed a null gang in the API command.

April 2024                                                                                                    51
