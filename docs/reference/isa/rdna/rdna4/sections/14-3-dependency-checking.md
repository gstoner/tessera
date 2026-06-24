# 14.3. Dependency Checking

> RDNA4 ISA — pages 171–171

data. This data is passed on to subsequent pixel shaders.

Every vertex shader must output at least one position vector (x, y, z; w is optional) to the POS0 target. The last
position export must have the DONE bit set to 1.

14.3. Dependency Checking
Export instructions are executed by the hardware in two phases. First, the instruction is selected to be
executed, and EXPCNT is incremented by 1. At this time, the wave has made a request to export data, but the
data has not been exported yet. Later, when the export actually occurs the EXEC mask and VGPR data is read
and the data is exported, and finally EXPcnt is decremented.

Use S_WAIT_EXPCNT to prevent the shader program from overwriting EXEC or the VGPRs holding the data to
be exported before the export operation has completed.

Multiple export instructions can be outstanding at one time. Exports of the same type (for example: position)
are completed in order, but exports of different types can be completed out of order. If the STATUS register’s
SKIP_EXPORT bit is set to one, the hardware treats all EXPORT instructions as if they were NOPs.
