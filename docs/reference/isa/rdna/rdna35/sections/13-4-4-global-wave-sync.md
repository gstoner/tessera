# 13.4.4. Global Wave Sync

> RDNA3.5 ISA — pages 147–149

13.4.4. Global Wave Sync
"Global Wave Sync" allows the waves running in different thread-groups, including across different CU’s and
SE’s to synchronize through barriers and semaphores.

The Global Wave Sync (GWS) unit contains 64 sync resources that are allocated by the Command Processor to
applications (VM_ID’s). These sync resources can be configured to act as counting semaphores or barriers.

  • GWS registers must be configured before use via GRBM reg writes: gds_gws_resource_cntl,
    gds_gws_resource
  • GDS_GWS_RESOURCE: Flag, Counter (number of waves at resource), type, head_{queue, valid, flag}
  • GDS_GWS_VMID: Per-VMID register identifying the range of GWS resources owned by each VMID (base &
    size)

The GWS contains 64 sync resources, each of which contains the following state:
  • 1-bit state flag: 0 or 1 - used to separate even & odd passes, distinguish entering waves from leaving.
  • a 12-bit counter - unsigned int
  • 1 byte Type: Semaphore or Barrier
  • Head-of-queue + valid + flag (13 bits)
  • Tail of Queue + flag (12 bits)
  • FIFO - holds full wave-id and a 1-bit flag

When used by the shader, M0 supplies the "resource_base[5:0]" which is used to virtualize the resources.

The resource offset comes from the GDS/GWS instruction’s "offset0[5:0]" field and is added to M0 and also to a
base-address per VMID to get the final resource ID. Resource ID’s are clamped to the range owned by this
VMID. If clamping occurs, the GWS returns a NACK which causes the wave to rewind the PC and halt.

  • GWS_resource_id = (GDS_GWS_VMID.BASE(vmid) + M0[21:16] + offset0[5:0]) % 64

                                               Table 63. GWS Instructions
Opcode                 Description
GWS_INIT                Initialize GWS resource
(uint vsrc0, u8 offset0
)                       Initialize the global wave sync resource specified by the virtualized resource id OFFSET0[5:0] with a
                        total wave count. This is most often intended to initialize a barrier resource for use by a later
                        ds_gws_barrier to synchronize all waves associated with this resource, but is not type specific and
                        can also be used to initialize a semaphore with an initial wave release count. The total wave count
                        is provided by the lane of vsrc associated with the first active thread based on the current EXEC
                        thread mask, interpreted as a 32-bit integer value.
                        The resource id is also offset by the value of M0[21:16], allowing virtualization of global wave sync
                        resource ids between draw contexts or based on other shader initialization state.
                        This is primarily to be used via the GRBM.
                        Operation:
                        //Initialize GWS_RESOURCE for later gws commands:
                        rid = (M0[21:16] + OFFSET0[5:0]) % 64
                        GWS_RESOURCE[rid].counter = vsrc.lane[find_first(EXEC)].u
                        GWS_RESOURCE[rid].flag = 0
                        return //release calling wave immediately

Opcode                 Description
GWS_SEMA_V             Semaphore: Increment resource counter
(u8 offset0)
                       For the global wave sync resource specified by the virtualized resource id OFFSET0[5:0], releases
                       one wave, immediately if already queued at this semaphore or once one arrives. Sets the resource
                       to semaphore type.
                       Operation:
                       //Release waves queued by ds_gws_sema_p instructions:
                       rid = (M0[21:16] + OFFSET0[5:0]) % 64
                       GWS_RESOURCE[rid].counter++
                       GWS_RESOURCE[rid].type = SEMAPHORE
                       return //release calling wave immediately
GWS_SEMA_BR             Semaphore Bulk Release
(uint vsrc0, u8 offset0
)                       For the global wave sync resource specified by the virtualized resource id OFFSET0[5:0], releases
                        the number of waves specified as a 32-bit integer in the first active lane of vsrc, immediately if
                        already queued at this semaphore or as they arrive. Sets the resource to semaphore type.
                        Operation: //Release waves queued by ds_gws_sema_p instructions:
                        rid = (M0[21:16] + OFFSET0[5:0]) % 64
                        release_count = vsrc.lane[find_first(EXEC)].u
                        GWS_RESOURCE[rid].counter += release_count
                        GWS_RESOURCE[rid].type = SEMAPHORE
                        return //release calling wave immediately
GWS_SEMA_P             Semaphore acquire (wait)
(u8 offset0 )
                       Queues this wave until the global wave sync resource specified by the virtualized resource id
                       OFFSET0[5:0] indicates that it should be released, which may be immediately if another wave has
                       already issued a ds_gws_sema_v or ds_gws_sema_br instruction to the resource. Sets the resource
                       to semaphore type.
                       Operation:
                       //Queue this wave until released:
                       rid = (M0[21:16] + OFFSET0[5:0]) % 64
                       GWS_RESOURCE[rid].type = SEMAPHORE
                       while (GWS_RESOURCE[rid].counter <= 0)
                       WAIT_IN_QUEUE
                       GWS_RESOURCE[rid].counter--
                       return //release calling wave
GWS_SEMA_              Semaphore release all waves waiting at a semaphore
RELEASE_ALL
(u8 offset0)           Operation:
                       //Release waves queued by ds_gws_sema_p instructions:
                       rid = (M0[21:16] + OFFSET0[5:0]) % 64
                       release_count = the number of waves currently enqueued at the semaphore
                       GWS_RESOURCE[rid].counter += release_count
                       GWS_RESOURCE[rid].type = SEMAPHORE
                       return //release calling wave immediately
                       This is typically used via the GRBM.

Opcode                  Description
GWS_BARRIER             Barrier wait
(uint vsrc0, u8 offset0
)                       Creates a global barrier for all waves associated with the global wave sync resource specified by a
                        virtualized resource id OFFSET0[5:0], which causes all waves issuing a ds_gws_barrier on the same
                        resource id to wait until a previously specified count of waves have also issued. Sets the resource to
                        barrier type. This provides functionality similar to an s_barrier instruction for local waves, but
                        allows synchronization of waves running on different compute units.

                        The wave count for completion of the barrier is initially provided by a ds_gws_init instruction.
                        Each subsequent ds_gws_barrier instruction may then provide the total wave count value for a
                        following ds_gws_barrier instruction. The total wave count minus one is provided by the lane of
                        vsrc associated with the first active thread based on the current EXEC thread mask, interpreted as a
                        32-bit integer value.

                        Operation:
                        //On entry: GWS_RESOURCE[rid].counter previously initialized
                        rid = (M0[21:16] + OFFSET0[5:0]) % 64
                        count_next = vsrc.lane[find_first(EXEC)].u
                        GWS_RESOURCE[rid].type = BARRIER
                        GWS_RESOURCE[rid].counter--
                        flag = GWS_RESOURCE[rid].flag
                        if (GWS_RESOURCE[rid].counter <= 0) //last wave in group
                          GWS_RESOURCE[rid].flag ^= 1 //release enqueued waves
                          GWS_RESOURCE[rid].counter = count_next //init for next barrier
                        return //release calling wave

                        // Enqueue waves which enter until the last enters and releases them
                        while (1)
                         if (GWS_RESOURCE[rid].type == BARRIER && GWS_RESOURCE[rid].flag != flag)
                        return //release calling wave
                        The description of "flag" above is a bit simplistic. Basically, every wave which enters is tagged with the
                        current GWS_RESOURCE.flag value. When the barrier condition is met, all waves with that flag value are
                        released, and GWS_RESOURCE.flag is inverted so any incoming waves are tagged with the opposite value
                        of flag.
