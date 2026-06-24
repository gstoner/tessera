# 3.5.3. SGPR Initialization

> RDNA4 ISA — pages 41–43

3.5. Initial Wave State
Before a wave begins execution, some of the state registers including SGPRs and VGPRs are initialized with
values derived either from state data, dynamic or derived data (e.g. interpolants or unique per-wave data). The
values are derived from register state and dynamic wave-launch state.

Note that some of this state is common across all waves in a draw call, and other state is unique per wave.

This section describes what state is initialized per shader stage. Note that as usual in this spec, the shader
stages refer to hardware shader stages and these often are not identical to software shader stages.
Shader state that is not explicitly listed as initialized in this section is not initialized. This includes LDS,
VGPRs and SGPRs not listed here as initialized.

3.5.1. EXEC initialization
Normally, EXEC is initialized with the mask of which threads are active in a wave. There are, however, cases
where the EXEC mask is initialized to zero indicating that this wave should do no work and exit immediately.
These are referred to as "Null waves" (EXEC==0) and exit immediately after starting execution.

3.5.2. SCRATCH_BASE Initialization
Waves that have scratch memory space allocated to them are initialized with their SCRATCH_BASE register
having a pointer to the address in global memory. Waves without scratch have this initialized to zero.

3.5.3. SGPR Initialization
SGPRs are initialized based on various SPI_PGM_RSRC* or COMPUTE_PGM_* register settings. Note that only
the enabled values are loaded, and they are packed into consecutive SGPRs, skipping over disabled values
regardless of the number of user-constants loaded. No SGPRs are skipped for alignment.

The tables below show how to control that values are initialized prior to shader launch.

3.5.3.1. Pixel Shader (PS)
                                                  Table 9. PS SGPR Load
SGPR Order            Description                                        Enable
First 0..32 of        User data registers                                SPI_SHADER_PGM_RSRC2_PS.user_sgpr
then                  {bc_optimize, prim_mask[14:0], lds_offset[15:0]}   N/A
then                  {ps_wave_id[9:0], ps_wave_index[5:0]}              SPI_SHADER_PGM_RSRC2_PS.wave_cnt_en
then                  Provoking Vtx Info:                                SPI_SHADER_PGM_RSRC1_PS .
                      {prim15[1:0], prim14[1:0], …, prim0[1:0]}          LOAD_PROVOKING_VTX

   PS_wave_index is (se_id[1:0] * NUM_PACKER_PER_SE + packer_id).

   PS_wave_id is an index value which is incremented for every wave. There is a separate counter per
   packer, so the combination of { ps_wave_id, ps_wave_index } forms a unique ID for any wave on the

      chip. The wave-id counter wraps at SPI_PS_MAX_WAVE_ID.

3.5.3.2. Geometry Shader (GS)
ES and GS are launched as a combined wave, of type GS. The shader is initialized as a GS wave type, with the PC
pointing to the ES shader and with GS user-SGPRs preloaded, along with a memory pointer to more GS user
SGPRs. The shader executes to the ES program first, then upon completion executes the GS shader. Once the ES
shader completes, it may re-use the SGPRs that contain ES user data and the GS shader address.

The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero).

State registers:
    • SPI_SHADER_PGM_{LO,HI}_ES : address of the GS shader
    • SPI_SHADER_PGM_RSRC1: resources of combined ES + GS shader
        ◦ GS_VGPR_COMP_CNT = # of GS VGPRs to load (2 bits)
    • SPI_SHADER_PGM_RSRC2: resources of combined ES + GS shader
        ◦ VGPR_COMP_CNT = # of VGPRs to load (2 bits)
        ◦ OC_LDS_EN
    • SPI_SHADER_PGM_RSRC{3,4}: resources of combined ES + GS shader

                                                Table 10. GS SGPR Load
SGPR #        GS with FAST_LAUNCH != 2      GS with FAST_LAUNCH == 2     Enable
0             GS Program Address [31:0]     GS Program Address [31:0]    automatically loaded
              comes from:                   comes from:
              SPI_SHADER_PGM_LO_GS          SPI_SHADER_PGM_LO_GS
1             GS Program Address [63:32]    GS Program Address [63:32]   automatically loaded
              comes from:                   comes from:
              SPI_SHADER_PGM_HI_GS          SPI_SHADER_PGM_HI_GS
2             {1’b0, gsAmpPrimPerGrp[8:0], 32’h0                         automatically loaded
              1’b0, esAmpVertPerGrp[8:0],
              ordered_wave_id[11:0]}
3             { TGsize[3:0],                { TGsize[3:0],               automatically loaded.
              WaveInGroup[3:0], 8’h0,       WaveInGroup[3:0], 24’h0 }
              gsInputPrimCnt[7:0],
              esInputVertCnt[7:0] }
4             Off-chip LDS base [31:0]      { TGID_Y[15:0],              SPI_SHADER_PGM_RSRC2_GS.oc_lds_en
                                            TGID_X[15:0] }
5             { 17’h0, attrSgBase[14:0] }   { TGID_Z[15:0], 1’b0,        -
                                            attrSgBase[14:0] }
6-7           -                             -                            -
8 - (up to)   User data registers of GS     User data registers of GS shader SPI_SHADER_PGM_RSRC2_GS.user_sgpr
39            shader

3.5.3.3. Front End Shader (HS)
LS and HS are launched as a combined wave, of type HS. The shader is initialized as an HS wave type, with the
PC pointing to the LS shader and with HS user-SGPRs preloaded, along with a memory pointer to more HS user
SGPRs. The shader executes to the LS program first, then upon completion executes the HS shader. Once the

LS shader completes, it may re-use the SGPRs that contain LS user data and the HS shader address.

The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero).

Other registers:
    • SPI_SHADER_PGM_{LO,HI}_LS : address of the LS shader
    • SPI_SHADER_PGM_RSRC1: resources of combined LS + HS shader
        ◦ LS_VGPR_COMP_CNT = # of LS VGPRs to load (2 bits)
    • SPI_SHADER_PGM_RSRC{2,3,4}: resources of combined LS + HS shader

                                                Table 11. HS (LS) SGPR Load
SGPR #               Description                                    Enable
0                    HS Program Address Low ([31:0])                SPI_SHADER_USER_DATA_LO_HS
1                    HS Program Address High ([63:32])              SPI_SHADER_USER_DATA_HI_HS
2                    Off-chip LDS base [31:0]                       automatically loaded
3                    {first_wave[0], lshs_TGsize[6:0],              automatically loaded
                     lshs_PatchCount[7:0], HS_vertCount[7:0],
                     LS_vertCount[7:0]}
4                    TF buffer base [17:0]                          automatically loaded
5                    { 27’b0, wave_id_in_group[4:0] }               SPI_SHADER_PGM_RSRC2_HS.scratch_en
8 - (up to) 39       User data registers of HS shader               SPI_SHADER_PGM_RSRC2_HS.user_sgpr

3.5.3.4. Compute Shader (CS)
Compute shaders initialize both user-SGPRs as well as trap-temp SGPRs.

                                                  Table 12. CS SGPR Load
SGPR #             Description                                       Enable
First 0.. 16 of    User data registers                               COMPUTE_PGM_RSRC2.user_sgpr
then               {14’h0, ordered_append_term[11:0],                COMPUTE_PGM_RSRC2.tg_size_en
                   work_group_size_in_waves[5:0]}
TTMP6              Unused
TTMP7              workgroup Grid {Z[15:0], Y[15:0]}                 Loaded if GridYZvalid==1
TTMP8              {DebugMark[0], GridYZvalid[0],                    All CS waves init this: DebugMark=0; GridYZvalid
                   waveIDinGroup[4:0], dispatchIndex[24:0] }         indicates if GridY and GridZ are both valid or if
                                                                     neither is valid. dispatchIndex is from
                                                                     compute_dispatch_pkt_addr_lo[24:0]
TTMP9              work_group Grid X[31:0]                           All CS waves init this.

Work-group Grid X, Y, Z are in units of "workgroup-dimension". Every wave in a given workgroup has the same
work-group Grid X, Y and Z values. Finding the thread position within the grid is accomplished by:

       thread_id_x = work_group_grid_X * work_group_dim_x + thread_idx_in_workgroup_X (VGPR0)
       (similar for Y and Z if exists)
       work_group_dim_x/y/z is a compile-time constant.

    • If the shader references either GridY or GridZ, then both are initialized and GridYZvalid=1;
