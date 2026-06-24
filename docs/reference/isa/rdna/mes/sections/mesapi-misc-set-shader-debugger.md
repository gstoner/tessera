# MESAPI_MISC__SET_SHADER_DEBUGGER

> Micro Engine Scheduler ISA — pages 45–45

                   WRM_OPERATION__WAIT_REG_MEM,
                   WRM_OPERATION__WR_WAIT_WR_REG,

                   WRM_OPERATION__MAX,
             };

             struct WAIT_REG_MEM
             {
                   enum WRM_OPERATION op;
                 // only function = equal_to_the_reference_value and mem_space = register_space
             supported for now
                   uint32_t reference;
                   uint32_t mask;
                   uint32_t reg_offset1;
                   uint32_t reg_offset2;
             };

                       •   op - WRM_OPERATION opcode

                       •   WRM_OPERATION__WAIT_REG_MEM (0) - MES will tight loop on reg_offset1 until it
                           equals reference value

                       •   WRM_OPERATION__WR_WAIT_WR_REG (1) - MES will first write reference to
                           reg_offset1, then it will poll reg_offset2 until it equals reference value

                       •   reference - Reference value to poll (op = 0), or reference value to poll/write (op = 1).

                       •   mask - Mask off comparison bits

                       •   reg_offset1 - Register to poll (op = 0), or target register to write to (op = 1)

                       •   reg_offset2 - Register to poll (op = 1)

MESAPI_MISC__SET_SHADER_DEBUGGER
                 This API enables shader debugger register programming.
                 The MES also clears the process context if the process has not been added.

                 The shader debugger settings are saved to the process context.

                 Registers are programmed whenever a compute queue belonging to the process is mapped.
                 Registers are restored to their default settings when process has no compute queues mapped.
             struct SET_SHADER_DEBUGGER
             {
                   uint64_t process_context_addr;
                   union
                   {

April 2024                                                                                                        45
