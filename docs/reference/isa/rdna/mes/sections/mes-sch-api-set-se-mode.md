# MES_SCH_API_SET_SE_MODE

> Micro Engine Scheduler ISA — pages 40–40

MES_SCH_API_UPDATE_ROOT_PAGE_TABLE
                 The KMD uses this API to change page table base of a process.
             union MESAPI__UPDATE_ROOT_PAGE_TABLE
             {
                   struct
                   {
                           union MES_API_HEADER          header;
                           uint64_t                      page_table_base_addr;
                           uint64_t                      process_context_addr;
                           struct MES_API_STATUS         api_status;
                           uint64_t                      timestamp;
                           uint32_t                      process_context_array_index;
                   };

                   uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                       •   page_table_base_addr - Page table base address

                       •   process_context_addr – Memory where process specific context is saved

                       •   process_context_array_index – Index of the process context array in scheduler's local
                           memory; valid only when
                           MES_SCH_API_SET_HW_RSRC.use_rs64mem_for_proc_gang_ctx is set

MES_SCH_API_SET_SE_MODE
                 The API allows the driver to turn off the second shader engine.
             enum MES_SE_MODE
             {
                 MES_SE_MODE_INVALID = 0,
                 MES_SE_MODE_SINGLE_SE = 1,
                 MES_SE_MODE_DUAL_SE = 2,
                 MES_SE_MODE_LOWER_POWER = 3,
             };

             union MESAPI__SET_SE_MODE
             {
                   struct
                   {
                           union MES_API_HEADER header;
                           /* the new SE mode to apply*/
                           MES_SE_MODE new_se_mode;
                           /* the fence to make sure the ItCpgCtxtSync packet is completed */

April 2024                                                                                                    40
