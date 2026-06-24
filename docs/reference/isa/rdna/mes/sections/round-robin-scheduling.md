# Round robin scheduling

> Micro Engine Scheduler ISA — pages 12–12

Round robin scheduling
             Round robin scheduling refers to the vanilla round robin scheduling where queues from all
             applications have the same priority, and the scheduler is expected to provide an equal amount of
             gpu time to each application.

             Schedules achieves this by:

                •   maintaining a database of queues from all applications

                •   mapping them on to available hardware queues based on their scheduling turn. The
                    database referred above is the scheduling context that contains queue list for each
                    unique pair of queue type(GFX, Compute, DMA) and priority level.

             There are 12 queue lists in total maintained inside the scheduler context.

             Scheduler context also contains queue or process specific information such as MQD pointers,
             VMIDs or any special resources allocated to the queue or the process. Various APIs from the
             driver result in queue and process information to be updated inside the scheduler context.

             Any updates to the scheduler context are then acted upon by the scheduler by performing
             certain scheduling actions such as queue map or unmap.

             AMD GPU has certain number of pipes, and each pipe has a fixed set of hardware queues. The
             user queues must be mapped onto the hardware queues to execute their work. Since there are
             limited number of hardware queues, the scheduler will attempt to map as many user queues on
             the hardware queues as possible.

             When a user queue is mapped on the hardware queue, the scheduler configures a quantum that
             the queue must run. Once the quantum has expired, the queue manager will connect the next
             hardware queue on the same pipe.

             When the hardware queues are not over-subscribed (#user queues <= #hardware queues), the
             scheduler will map all user queues on the hardware queues and configure equal quantum for all
             queues.

             This allows the queue manager to “connect” each hardware queue for an equal amount of
             configure time. It is possible that a “connected queue” may go idle before its quantum has
             expired, in which case the queue manager will connect the next hardware queue that has ready
             work to execute.

             When the hardware queues are over-subscribed (#user queues > #hardware queues), the
             scheduler will map as many queues possible on the available HW queues and will unmap them
             gradually upon quantum expiry or when they go idle to map the queues from the next process.

             To ensure that the limited number of hardware queues are used in best way possible, the
             scheduler only maps user queues with outstanding work to execute. This requires the scheduler

April 2024                                                                                                12
