# MES API

> Micro Engine Scheduler ISA — pages 16–17

MES API
             This section describes MES API usage. The kernel mode driver (KMD) communicates with the
             Micro Engine Scheduler (MES) firmware by submitting API commands to the MES queue ring
             buffer.

                  •    Some API’s fields are for debug purposes which are not enabled by default. These fields
                       have Debug Only in their descriptions

MES API format
                  •    MES scheduler APIs are defined in mes_api_def.h

                  •    Each API has length 64 DWORDS as defined in enum {API_FRAME_SIZE_IN_DWORDS = 64}

             The following format is applicable to all APIs:
             union MESAPI__APINAME
             {
                 struct
                 {
                     union MES_API_HEADER            header;
                     //API specific info
                        struct MES_API_STATUS        api_status;
                        uint64_t                     timestamp;
                  };
                  uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

             Each API contains its specific information and three common fields: header, timestamp and
             api_status:

             union MES_API_HEADER
             {
                  struct
                  {
                        uint32_t type       : 4;     /* 0 - Invalid; 1 - Scheduling; 2-15 - Reserved*/
                        uint32_t opcode     : 8;   /* API command defined in MES_SCH_API_OPCODE enum */
                        uint32_t dwsize     : 8;   /* Size in DWORD of the API command including header */
                        uint32_t reserved : 12;
                  };
                  uint32_t u32All;
             };

April 2024                                                                                                16

                 Opcode defines all supported MES APIs:
             enum MES_SCH_API_OPCODE
             {
                    MES_SCH_API_SET_HW_RSRC = 0,
                    MES_SCH_API_SET_SCHEDULING_CONFIG = 1,
                    MES_SCH_API_ADD_QUEUE = 2,
                    MES_SCH_API_REMOVE_QUEUE = 3,
                    MES_SCH_API_PERFORM_YIELD = 4,
                    MES_SCH_API_SET_GANG_PRIORITY_LEVEL = 5,
                    MES_SCH_API_SUSPEND = 6,
                    MES_SCH_API_RESUME = 7,
                    MES_SCH_API_RESET = 8,
                    MES_SCH_API_SET_LOG_BUFFER = 9,
                    MES_SCH_API_CHANGE_GANG_PRORITY = 10,
                    MES_SCH_API_QUERY_SCHEDULER_STATUS = 11,
                    MES_SCH_API_PROGRAM_GDS = 12,
                    MES_SCH_API_SET_DEBUG_VMID = 13,
                    MES_SCH_API_MISC = 14,
                    MES_SCH_API_UPDATE_ROOT_PAGE_TABLE = 15,
                    MES_SCH_API_AMD_LOG = 16,
                    MES_SCH_API_SET_SE_MODE = 17,
                    MES_SCH_API_SET_GANG_SUBMIT = 18,
                    MES_SCH_API_MAX = 0xFF
             };

                 The api_status in each API command contains fence address and fence value that the KMD
                 inserts. MES firmware writes the fence value to the given address to notify the KMD that the
                 API has been processed by scheduler.
             struct MES_API_STATUS
             {
                    uint64_t api_completion_fence_addr;
                    uint64_t api_completion_fence_value;
             };

April 2024                                                                                                 17
