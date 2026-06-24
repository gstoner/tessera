# HW architecture overview

> Micro Engine Scheduler ISA — pages 6–7

HW architecture overview
             The scheduler firmware’s main role is to map the scheduling requirement on to the HW
             architecture. Therefore, it is required to understand the HW architecture to understand how
             scheduling firmware achieves the scheduling requirements on the AMD GPUs.

             The following diagram describes the high-level HW architecture and execution flow to
             schedule/run an application queue.

             Key highlights of HW architecture can be summarized as follows.

                •   The GPU frontend has three micro-processors meant to execute scheduling, compute
                    and gfx firmware

                •   There are multiple GFX and Compute pipes where each pipe contains a queue mgr that
                    arbitrates a certain number of HW queues attached to that pipe

                •   There are two levels of scheduling:

                        o   First level of scheduling is at firmware, where firmware decides the applications
                            queues that should be mapped onto the available hardware queues on various
                            pipes

April 2024                                                                                               6

                         o   Second level of scheduling is in the Queue Manager HW where it selects one of
                             the ready hardware queue and runs it on the shader complex. Although the
                             second level of scheduling is done by Queue manager hardware, scheduler FW is
                             able to influence the Queue manager’s hardware queue selection and execution
                             via various knobs such as hardware queue priority, quantum etc.

                 •   Queue manager’s arbitration logic selects a HW queue and runs it on the shader
                     complex. The mapped hardware queue selected for execution is called a “connected
                     queue”

                 •   Each pipe provides an independent path to launch a queue’s work inside 3D/CS complex.
                     So potentially there could be #pipes worth of “connected queues” running in parallel

                 •   There is a shared pool of ALUs for GFX and compute work

             Refer to RDNA3 Instruction Set Architecture Reference Guide for additional information.

April 2024                                                                                              7
