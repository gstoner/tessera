# Queue prioritization

> Micro Engine Scheduler ISA — pages 13–15

             to be informed when an unmapped queue has new work.

             This is achieved using aggregated doorbells. Aggregated doorbells are special doorbells that are
             written by SW when it submits work to an unmapped queue. Write to an aggregated doorbell
             informs the scheduler of new work to an unmapped queue. The scheduler then uses this
             notification to map the queue as soon as possible, based on the queue’s priority relative to the
             other work. When aggregated doorbells are not available or used by the SW, scheduler start to
             periodically polls the write pointer memory of the unmapped queues to discover if they have
             new work. This is only done when there is a queue-over subscription as all user queues could
             not be mapped on to the limited hardware queues.

             This flowchart shows the event driven scheduling design and how scheduler handles these
             events to implement a basic round robin scheduling of the user queues.

Queue prioritization
             The scheduler maps as many user queues as possible to the available HW queues. Once the HW
             queues are over-subscribed, the scheduler starts to round robin the user queues onto the
             available HW queues.

             This basic round robin scheme falls short when it comes to executing work of varying priority

April 2024                                                                                               13

             levels. The scheduler uses a combination of various hardware prioritization features to
             implement the desired scheduling behavior for each priority level.

             Before discussing the scheduler’s usage of these prioritization features, it’s useful to discuss the
             various hardware prioritization features available for scheduler’s use:

             •     Mid command buffer preemption
                   Queue preemption is the most fundamental feature that is employed in various
                   prioritization scenarios to achieve the desired quality of service. Preemption can be issued
                   at several different work boundaries that affects the latency and the amount of state that
                   gets saved or restored. For example, compute work can be preempted at a submission,
                   dispatch, thread group or at a shader instruction boundary. The preemption latency and
                   amount of saved or restored states will vary based on the preemption granularity.

             •     Wave limiting
                   This method reduces the workload from other queues by limiting the number of waves
                   that can be issued. “Wave” represents a group of shader threads.

             •     Pipe priority
                   Connected queues on each pipe asserts a pipe priority to the shader HW. The shader HW
                   uses this priority to select and launch upcoming work based on pipe priority.

             •     Dispatch tunneling
                   The method immediately disables the work from other queues when a dispatch from a
                   high-priority queue is executed. The ability to tunnel dispatches is configured as a queue-
                   property.

             •     Queue quantum
                   Quantum is implemented by both queue manager hardware and scheduler firmware. The
                   queue manager connects and disconnects queues based on the quantum configured in the
                   hardware queue by the scheduler firmware.
                   During queue oversubscription, the scheduler firmware un-maps the queue once its
                   quantum has expired to allow mapping of other unmapped user queues on the hardware
                   queues.

             •     Queue connection priority
                   The queue connection priority is specified for each hardware queue and is used by the
                   queue manager hardware to select the next hardware queue that will be connected to the
                   pipe.

             •     Compute unit reservation
                   This method allows a certain number of compute units to be carved out and only made
                   available for a particular queue. This method is used in scenarios where the machine
                   utilization launch latency is critical.

             The scheduler uses a combination of the described methods to achieve the desired prioritization
             in the presence of workload from queues with different priorities.

April 2024                                                                                                  14

             The following table lists how various methods are employed in different scenarios:

              Ready work to run         Expected scheduling                How scheduler achieves it
                                        behavior
              1.   Real time compute    1. Real time priority queue        Real time prioritization
                   queue                   runs without any delays         1. Real time queue once created stays
              2.   Focus gfx queue      2. Once Real time queue is            mapped(max 4 RT queues allowed i.e. max
              3.   Normal priority         idle, Focus queue will start       1 RT queue/pipe)
                   compute queues          to execute.                     2. A certain # of Compute units are reserved
              4.   Idle Compute queue   3. Once Focus queue has               for the Real time queue. Certain Real time
                                           executed for a configured          queues will use Wave limiting instead of
                                           amount of time, the Normal         Compute unit reservation to quickly get
                                           queue will execute for a           their work to execute.
                                           certain period of time.         3. Highest queue connection priority
                                        4. Once all Real time, Focus       4. Highest shader type priority
                                           and Normal queues have
                                           nothing else to execute,
                                                                           Focus and Normal prioritization
                                           only then the Idle queue will
                                           execute                         1. Focus queue is mapped as the same
                                                                              connection priority as Normal queue.
                                                                           2. Focus queue has a larger quantum relative
                                                                              to the Normal queue.
                                                                           3. Focus queues have higher pipe priority.
                                                                           4. Scheduler firmware may also unmap
                                                                              Normal queues on other pipes when they
                                                                              have long running shaders that prevent the
                                                                              Focus work from being able to launch on
                                                                              the compute units.
                                                                           5. Normal queues get preempted with a
                                                                              higher level of preemption than the Focus
                                                                              queues.

                                                                           Idle prioritization
                                                                               Executes when all queue in the higher
                                                                               priority levels have been idle for some time.

April 2024                                                                                                       15
