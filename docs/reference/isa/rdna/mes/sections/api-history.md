# API history

> Micro Engine Scheduler ISA — pages 47–47

Scheduler log
                 As described in previous sections, MES scheduler firmware interacts with kernel mode driver
                 and CP block. Events between MES scheduler and KMD, MES scheduler and CP are of interests
                 to understand system state when it comes to debugging MES issues.

                 To use MES log, KMD needs to allocate log buffer memory and passes GPU address of the log
                 buffer memory to MES scheduler in API MES_SCH_API_SET_HW_RSRC.

                 MES log format is defined in the following structure.

             struct MES_EVT_INTR_HIST_LOG
             {
                   struct MES_SCH_INTR_HIST_INFO       interrupt_history[MES_SCH_MAX_NUM_MES_INTR_ENTRY];
                 struct MES_SCH_EVT_LOG_HIST_INFO
             event_log_history[MES_SCH_MAX_NUM_MES_EVT_LOG_ENTRY];
                   struct MES_SCH_API_HIST_INFO        api_history[MES_SCH_MAX_NUM_API_CALL_ENTRY];
                   uint32                              interrupt_history_index;
                   uint32                              event_log_history_index;
                   uint32                              api_history_index;
             };

                 It contains three arrays, api_history is for events from KMD to MES scheduler, event_log_history
                 is for events from MES scheduler to CP and interrupt_history is for interrupt events from CP to
                 MES scheduler. These arrays are updated in a circular buffer fashion and each array has an index
                 which always points to the entry in the array that will be updated next.

API history

                 Each entry in api_history array has the following format:
             struct MES_SCH_API_HIST_INFO
             {
                   enum MES_SCH_API_CALL_ID api_id;
                   uint64_t                   time_before_call;
                   uint64_t                   time_after_call;
                   uint32_t                   error_code;
                   struct
                   {
                       uint32 status : 1;
                       uint32 reserved : 31;

April 2024                                                                                                  47
