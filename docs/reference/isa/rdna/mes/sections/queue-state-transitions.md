# Queue state transitions

> Micro Engine Scheduler ISA — pages 11–11

Queue state transitions
             This diagram describes the possible queue states and triggers for the transitions.

             Based on this illustration, a queue could be in one of the following states:

                •   Unmapped
                    The user queue has not been initialized into a hardware queue and it solely exists in
                    memory.

                •   Mapped & disconnected
                    The user queue has been initialized into a hardware queue but is currently not connected
                    to the shader subsystem so is not able to execute.

                •   Mapped and connected
                    The user queue has been initialized into a hardware queue and is connected to the
                    shader subsystem. Only connected queues are able to request and launch their work on
                    the shader resources. Only queues with pending work are allowed to connect.

             The GPUSCH implementation can be explained in two steps where first we go into the round
             robin scheduling and secondly we look at how different levels of queue priority are
             implemented.

April 2024                                                                                                  11
