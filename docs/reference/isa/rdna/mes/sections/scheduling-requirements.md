# Scheduling requirements

> Micro Engine Scheduler ISA — pages 5–5

Scheduling requirements
             At a high-level, the scheduling requirements can be summarized as:
                •     Fair and efficient scheduling of the application’s work on the GPU
                •     Implementation of multiple priority levels for a variety of user scenarios

             These high-level requirements can also be described from a user scenario perspective:

                •     Applications with the same priority level should get the equal amount of the GPU
                      execution time

                •     Applications with the user focus (for e.g. compositor) should receive larger GPU time, but
                      not infinitely starve the Normal priority level

                •     Real time work such VR, Super-Wet ink or True audio should run immediately and can
                      infinitely starve work in the lower priority levels

                •     Low-priority work such as OneDrive, photo enhancement, compression or
                      Folding@home should only run when all higher priority levels are idle

             Scheduler implements the above stated requirements via 4 levels of queue prioritization.

              Level                  Scheduling expectation             What runs here

              Real time              Lowest possible launch latency.    VR compositor, Super wet ink, True
                                                                        audio next.
              Focus                  Provides no forward progress       Desktop compositor, Video post
                                     guarantee for the lower levels.    processing, foreground app’s work.
              Normal                 Gets majority of GPU execution     Typical work from the application
                                     time in the absence of Real time   that does not have the user focus
                                     work.
              Low                    Ensures forward progress for the All background work with no strict
                                     Normal level work.               deadline requirements for e.g. file
                                                                      compression, encryption etc.

             This scheduling behavior mirrors Microsoft specifications for GPU scheduling. The requirements
             are captured in the Microsoft GPU scheduling specification and are not explained further.

April 2024                                                                                                5
