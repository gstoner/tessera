# MES_SCH_API_PERFORM_YIELD

> Micro Engine Scheduler ISA — pages 28–28

             union MESAPI__SET_SCHEDULING_CONFIG
             {
                    struct
                    {
                         union MES_API_HEADER            header;
                         uint64_t                        grace_period_other_levels[AMD_PRIORITY_NUM_LEVELS];
                         /* Default quantum for scheduling across processes within a priority band. */
                         uint64_t                        process_quantum_for_level[AMD_PRIORITY_NUM_LEVELS];

                         /* Default grace period for processes that preempt each other within a priority
             band.*/
                     uint64_t
             process_grace_period_same_level[AMD_PRIORITY_NUM_LEVELS];

                     /* For normal level this field specifies the target GPU percentage in situations
             when it's starved by the high level.
                             Valid values are between 0 and 50, with the default being 10.*/
                         uint32_t                        normal_yield_percent;

                         struct MES_API_STATUS           api_status;
                         uint64_t                        timestamp;
                    };

                    uint32_t max_dwords_in_api[API_FRAME_SIZE_IN_DWORDS];
             };

                    •    grace_period_other_levels - Grace period when preempting another priority band for this
                         priority band. The value for idle priority band is ignored, as it never preempts other bands

                    •    process_quantum_for_level - Default quantum for scheduling across processes within a
                         priority band

                    •    process_grace_period_same_level - Default grace period for processes that preempt each
                         other within a priority band

                    •    normal_yield_percent - For normal level this field specifies the target GPU percentage in
                         situations when it's starved by the high level. Valid values are between 0 and 50, with
                         the default being 10
                 Note: In current fw, only relevant quantum is process_quantum_for_level, other fields are not
                 used in scheduling/

MES_SCH_API_PERFORM_YIELD
                 This API is not currently supported.

April 2024                                                                                                     28
