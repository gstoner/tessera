# MES_SCH_API_QUERY_SCHEDULER_STATUS

> Micro Engine Scheduler ISA — pages 36–36

MES_SCH_API_QUERY_SCHEDULER_STATUS
                 The KMD uses this API to query status/info from MES firmware.
             enum MES_API_QUERY_MES_OPCODE
             {
                    MES_API_QUERY_MES__GET_CTX_ARRAY_SIZE,
                    MES_API_QUERY_MES__CHECK_HEALTHY,
                    MES_API_QUERY_MES__MAX,
             };

             union MESAPI__QUERY_MES_STATUS
             {
                    struct
                    {
                         union MES_API_HEADER              header;
                         enum MES_API_QUERY_MES_OPCODE     subopcode;
                         struct MES_API_STATUS             api_status;
                         uint64_t                          timestamp;
                         union
                         {
                              struct MES_API_QUERY_MES__CTX_ARRAY_SIZE     ctx_array_size;
                              struct MES_API_QUERY_MES__HEALTHY_CHECK      healthy_check;
                              uint32_t data[QUERY_MES_MAX_SIZE_IN_DWORDS];
                         };
                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    subopcode - Changes functionality based on what MES_API_QUERY_MES_OPCODE is
                         used

MES_API_QUERY_MES__GET_CTX_ARRAY_SIZE
                 The KMD uses this to query MES internal structure size.
             struct MES_API_QUERY_MES__CTX_ARRAY_SIZE
             {
                   uint64_t      proc_ctx_array_size_addr;
                   uint64_t      gang_ctx_array_size_addr;
             };

April 2024                                                                                       36
