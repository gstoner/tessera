# Scheduler FW architecture

> Micro Engine Scheduler ISA — pages 8–9

Scheduler FW architecture
                 The scheduler firmware architecture can be decomposed into following key components:

             1. Scheduler APIs
                These are the commands sent by the driver to inform scheduler of the events such as queue
                creation, destruction, suspension, or any changes to its priority. Each API is described later under
                APIs section.

             2. Scheduler context
                Data structures where scheduler maintains application, queue state or any other scheduling state
                or configuration.

                 Scheduler context is the state that API processor and Core scheduler thread works on. The
                 scheduler context consists of:

                 HW resource state

                 •     HQD State – Current Queue mapped, queue type, scheduled time.

                 •     VMID State - Current process mapped

                 •     GDS State – Current process using the GDS partition.

                 Process scheduling state

                 •     Scheduling level state - process list, grace period, normalband percentage,
                       has_ready_queues

                 •     Process state – Gang list for each context priority(-7/+7), processquantum, running time
                       carryover

                 •     Per Gang state – Queuelist, running time carryover, gang quantum.

             3. API processor
                Processes the APIs submitted by the driver and modifies the scheduler state if required.

             4. Core Scheduler
                Looks at the scheduler state, decide next set of scheduling actions and applies them.
                For example, mapping a queue when it is created, or suspending as required. The scheduling
                algorithm is described in a dedicated section later in this document.

             5. Interrupt Handler
                Handles interrupts from various internal HW blocks.
                For example, interrupt handlers reads the API data from the fetcher or collects the busy, idle
                state of various hardware queues.

April 2024                                                                                                    8

             These are the main types of interrupts that RS64 processer will receive:

                              Interrupt                                   Description
                               source

             ME0 Pipe0                                     Gfx pipe

             ME1 Pipe0/1/2/3                               First 4 compute pipes

             ME2 Pipe0/1/2/3                               Other 4 compute pipes

             MES packet fifo                               Indicates new data in the MES queues

             Hardware queue Message interrupt              QueueManager interrupts

             Software interrupt                            Caused by MES fw itself

             Timer interrupt                               Used for Timer expiration

             Unprivileged access                           Unprivileged access of MES registers

             External interrupt                            From Non-gfx blocks

April 2024                                                                                        9
