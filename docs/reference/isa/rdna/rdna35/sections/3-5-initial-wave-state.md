# 3.5. Initial Wave State

> RDNA3.5 ISA — pages 37–39

       S_SENDMSG_RTN_B64 S[2:3] REALTIME
       S_WAITCNT LGKMcnt == 0

3.5. Initial Wave State
Before a wave begins execution, some of the state registers including SGPRs and VGPRs are initialized with
values derived either from state data, dynamic or derived data (e.g. interpolants or unique per-wave data). The
values are derived from register state and dynamic wave-launch state.

Note that some of this state is common across all waves in a draw call, and other state is unique per wave.

This section describes what state is initialized per shader stage. Note that as usual in this spec, the shader
stages refer to hardware shader stages and these often are not identical to software shader stages.

State initialization is controlled by state registers which are defined in other documentation.

3.5.1. EXEC initialization
Normally, EXEC is initialized with the mask of which threads are active in a wave. There are, however, cases
where the EXEC mask is initialized to zero indicating that this wave should do no work and exit immediately.
These are referred to as "Null waves" (EXEC==0) and exit immediately after starting execution.

3.5.2. FLAT_SCRATCH Initialization
Waves that have scratch memory space allocated to them are initialized with their FLAT_SCRATCH register
having a pointer to the address in global memory. Waves without scratch have this initialized to zero.

3.5.3. SGPR Initialization
SGPRs are initialized based on various SPI_PGM_RSRC* or COMPUTE_PGM_* register settings. Note that only
the enabled values are loaded, and they are packed into consecutive SGPRs, skipping over disabled values
regardless of the number of user-constants loaded. No SGPRs are skipped for alignment.

The tables below show how to control which values are initialized prior to shader launch.

3.5.3.1. Pixel Shader (PS)
                                                 Table 8. PS SGPR Load
SGPR Order            Description                                        Enable
First 0..32 of        User data registers                                SPI_SHADER_PGM_RSRC2_PS.user_sgpr
then                  {bc_optimize, prim_mask[14:0], lds_offset[15:0]}   N/A
then                  {ps_wave_id[9:0], ps_wave_index[5:0]}              SPI_SHADER_PGM_RSRC2_PS.wave_cnt_en

SGPR Order            Description                                           Enable
then                  Provoking Vtx Info:                                   SPI_SHADER_PGM_RSRC1_PS .
                      {prim15[1:0], prim14[1:0], …, prim0[1:0]}             LOAD_PROVOKING_VTX

    PS_wave_index is (se_id[1:0] * GPU__GC__NUM_PACKER_PER_SE + packer_id).

    PS_wave_id is an index value which is incremented for every wave. There is a separate counter per
    packer, so the combination of { ps_wave_id, ps_wave_index } forms a unique ID for any wave on the
    chip. The wave-id counter wraps at SPI_PS_MAX_WAVE_ID.

3.5.3.2. Geometry Shader (GS)
ES and GS are launched as a combined wave, of type GS. The shader is initialized as a GS wave type, with the PC
pointing to the ES shader and with GS user-SGPRs preloaded, along with a memory pointer to more GS user
SGPRs. The shader executes to the ES program first, then upon completion executes the GS shader. Once the ES
shader completes, it may re-use the SGPRs which contain ES user data and the GS shader address.

The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero).

State registers:
    • SPI_SHADER_PGM_{LO,HI}_ES : address of the GS shader
    • SPI_SHADER_PGM_RSRC1: resources of combined ES + GS shader
        ◦ GS_VGPR_COMP_CNT = # of GS VGPRs to load (2 bits)
    • SPI_SHADER_PGM_RSRC2: resources of combined ES + GS shader
        ◦ VGPR_COMP_CNT = # of VGPRs to load (2 bits)
       ◦ OC_LDS_EN
    • SPI_SHADER_PGM_RSRC{3,4}: resources of combined ES + GS shader

                                                     Table 9. GS SGPR Load
SGPR #       GS with FAST_LAUNCH != 2           GS with FAST_LAUNCH == 2         Enable
0            GS Program Address [31:0]          GS Program Address [31:0]        automatically loaded
             comes from:                        comes from:
             SPI_SHADER_PGM_LO_GS               SPI_SHADER_PGM_LO_GS
1            GS Program Address [63:32]         GS Program Address [63:32]       automatically loaded
             comes from:                        comes from:
             SPI_SHADER_PGM_HI_GS               SPI_SHADER_PGM_HI_GS
2            {1’b0, gsAmpPrimPerGrp[8:0], 32’h0                                  Must not be overwritten, in some cases listed
             1’b0, esAmpVertPerGrp[8:0],                                         below.
             ordered_wave_id[11:0]}
3            { TGsize[3:0],                     { TGsize[3:0],                   automatically loaded.
             WaveInGroup[3:0], 8’h0,            WaveInGroup[3:0], 24’h0 }
             gsInputPrimCnt[7:0],
             esInputVertCnt[7:0] }
4            Off-chip LDS base [31:0]           { TGID_Y[15:0],                  SPI_SHADER_PGM_RSRC2_GS.oc_lds_en
                                                TGID_X[15:0] }
5            { 17’h0, attrSgBase[14:0] }        { TGID_Z[15:0], 1’b0,            -
                                                attrSgBase[14:0] }
6                         SPI is loading flat_scratch[63:0] at this time         -
7                                                                                -

SGPR #           GS with FAST_LAUNCH != 2            GS with FAST_LAUNCH == 2        Enable
8 - (up to)      User data registers of GS           User data registers of GS shader SPI_SHADER_PGM_RSRC2_GS.user_sgpr
39               shader

When stream-out is used, SGPR[2] must not be modified or overwritten any time before the final stream out is
issued (GDS ordered count with 'done' = 1). This is because the pipeline reset sequence which hardware
automatically executes reads SGPR to fabricate a GDS-ordered-count instruction and relies on this value.

3.5.3.3. Front End Shader (HS)
LS and HS are launched as a combined wave, of type HS. The shader is initialized as an HS wave type, with the
PC pointing to the LS shader and with HS user-SGPRs preloaded, along with a memory pointer to more HS user
SGPRs. The shader executes to the LS program first, then upon completion executes the HS shader. Once the
LS shader completes, it may re-use the SGPRs which contain LS user data and the HS shader address.

The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero).

Other registers:
    • SPI_SHADER_PGM_{LO,HI}_LS : address of the LS shader
    • SPI_SHADER_PGM_RSRC1: resources of combined LS + HS shader
        ◦ LS_VGPR_COMP_CNT = # of LS VGPRs to load (2 bits)
    • SPI_SHADER_PGM_RSRC{2,3,4}: resources of combined LS + HS shader

                                                     Table 10. HS (LS) SGPR Load
SGPR #                    Description                                     Enable
0                         HS Program Address Low ([31:0])                 SPI_SHADER_USER_DATA_LO_HS
1                         HS Program Address High ([63:32])               SPI_SHADER_USER_DATA_HI_HS
2                         Off-chip LDS base [31:0]                        automatically loaded
3                         {first_wave[0], lshs_TGsize[6:0],               automatically loaded
                          lshs_PatchCount[7:0], HS_vertCount[7:0],
                          LS_vertCount[7:0]}
4                         TF buffer base [17:0]                           automatically loaded
5                         { 27’b0, wave_id_in_group[4:0] }                SPI_SHADER_PGM_RSRC2_HS.scratch_en
8 - (up to) 39            User data registers of HS shader                SPI_SHADER_PGM_RSRC2_HS.user_sgpr

3.5.3.4. Compute Shader (CS)
                                                        Table 11. CS SGPR Load
SGPR Order            Description                                           Enable
First 0.. 16 of       User data registers                                   COMPUTE_PGM_RSRC2.user_sgpr
then                  work_group_id0[31:0]                                  COMPUTE_PGM_RSRC2.tgid_x_en
then                  work_group_id1[31:0]                                  COMPUTE_PGM_RSRC2.tgid_y_en
then                  work_group_id2[31:0]                                  COMPUTE_PGM_RSRC2.tgid_z_en
then                  {first_wave, 6’h00, wave_id_in_group[4:0], 2’h0,      COMPUTE_PGM_RSRC2.tg_size_en
                      ordered_append_term[11:0], work-
                      group_size_in_waves[5:0]}
TTMP4,5               0
