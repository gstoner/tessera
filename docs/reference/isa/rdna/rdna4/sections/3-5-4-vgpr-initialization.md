# 3.5.4. VGPR Initialization

> RDNA4 ISA — pages 44–46

    if neither is used by the shader then neither is initialized and GridYZvalid=0.
       ◦ If GridY is valid but GridZ is not, GridZ receives the value of zero.
  • If a wave has no scratch space allocated, SCRATCH_BASE == 0
  • "wave-id-in-work-group" is a 6 bit index (not x,y,z).
  • The dispatch-index is a zero-based index of the 64-byte dispatch packet in the AQL user queue
  • The trap handler can determine the dispatch grid dimensions by looking up into in the dispatch packet,
    referenced by the dispatch-index pointer.

3.5.4. VGPR Initialization
The section shows the VGPRs that may be initialized prior to wave launch. COMPUTE_PGM_RSRC* or
SPI_SHADER_PGM_RSRC* control registers can select a reduced set per shader stage.

Stage      HS (+LS) combined              GS (+ES)                         GS (+ES)                       GS (+ES)
                                          ES is DS combined                ES is VS combined              Fast Launch
                                                                                                          combined
VGPR0      HS: Patch ID                   GS:                              GS:                            X, Y, Z
                                          [7:0] = rel index vertex 0       [7:0] = rel index vertex 0
                                          [8] = edge flag for index 0      [8] = edge flag for index 0
                                          [16:9] = rel index vertex 1      [16:9] = rel index vertex 1
                                          [17] = edge flag for index 1     [17] = edge flag for index 1
                                          [25:18] = rel index vertex 2     [25:18] = rel index vertex 2
                                          [26] = edge flag for index 2     [26] = edge flag for index 2
                                          [31:27] = gs instance ID         [31:27] = gs instance ID
VGPR1      HS:                            GS:                              GS:                            unused
           [7:0] = rel patch ID (0..255) Primitive ID                      Primitive ID
           [12:8] = control point ID
VGPR2      LS:                            unused                           GS and adjacent prim type:     unused
           index of current vertex                                         [7:0] = rel index vertex 3
           within vertex buffer                                            [8] = edge flag for index 3
                                                                           [16:9] = rel index vertex 4
                                                                           [17] = edge flag for index 4
                                                                           [25:18] = rel index vertex 5
                                                                           [26] = edge flag for index 5
VGPR3      LS: Instance ID                ES: U[fp32]                      ES: vertex index               unused

VGPR4      unused                         ES: V[fp32]                      ES: instance ID                unused

VGPR5      unused                         ES: Rel Patch ID                 unused                         unused

VGPR6      unused                         ES: Patch ID                     unused                         unused

Stage       CS
VGPR0       {2’b0, Z[9:0], Y[9:0], X[9:0]}
            X, Y, and Z work-item position within the workgroup. Work-item 0 is at (0,0,0).

3.5.4.1. Pixel Shader VGPR Input Control
Pixel Shader VGPR input loading is quite a bit more complicated. There is a CAM in SPI that maps VS outputs
to PS inputs. Of the PS inputs which need loading, they are loaded in this order:

                   I persp sample            I linear sample            X float
                   J persp sample            J linear sample            Y float
                   I persp center            I linear center            Z float
                   J persp center            J linear center            W float
                   I persp centroid          I linear centroid          Facedness
                   J persp centroid          J linear centroid          Ancillary: RTA, ISN, PT,
                   I/W                       Line stipple               eye-id
                   J/W                                                  Sample mask
                   1/W                                                  X/Y fixed

Two registers (SPI_PS_INPUT_ENA and SPI_PS_INPUT_ADDR) control the enabling of IJ calculations and
specifying of VGPR initialization for PS waves. SPI_PS_INPUT_ENA is used to determine what gradients are
enabled for setup, whether per-pixel Z is enabled, what terms are calculated and/or passed through the
barycentric logic, and what is loaded into VGPR for PS. SPI_PS_INPUT_ADDR can be used to manipulate the
VGPR destination of terms that are enabled by INPUT_ENA, typically providing a way to maintain consistent
VGPR addressing when terms are disabled in INPUT_ENA. It is valid to set a bit in ADDR when the
corresponding bit in ENA is not set, but if the ENA bit is set then the corresponding bit in ADDR must also be
set.

The two Pixel Staging Register (PSR) control registers contain an identical set of fields and consist of the
following:

             Field Name                IJ / VGPR Terms                  BITS        VGPR Dest with Full
                                                                                    Load
             PERSP_SAMPLE_ENA          PERSP_SAMPLE I                   32          VGPR0
                                       PERSP_SAMPLE J                   32          VGPR1
             PERSP_CENTER_ENA          PERSP_CENTER I                   32          VGPR2
                                       PERSP_CENTER J                   32          VGPR3
             PERSP_CENTROID_ENA        PERSP_CENTROID I                 32          VGPR4
                                       PERSP_CENTROID J                 32          VGPR5
             PERSP_PULL_MODEL_ENA PERSP_PULL_MODEL I/W                  32          VGPR6
                                       PERSP_PULL_MODEL J/W             32          VGPR7
                                       PERSP_PULL_MODEL 1/W             32          VGPR8
             LINEAR_SAMPLE_ENA         LINEAR_SAMPLE I                  32          VGPR9
                                       LINEAR_SAMPLE J                  32          VGPR10
             LINEAR_CENTER_ENA         LINEAR_CENTER I                  32          VGPR11
                                       LINEAR_CENTER J                  32          VGPR12
             LINEAR_CENTROID_ENA       LINEAR_CENTROID I                32          VGPR13
                                       LINEAR_CENTROID J                32          VGPR14
             LINE_STIPPLE_TEX_ENA      LINE_STIPPLE_TEX                 32          VGPR15
             POS_X_FLOAT_ENA           POS_X_FLOAT                      32          VGPR16
             POS_Y_FLOAT_ENA           POS_Y_FLOAT                      32          VGPR17
             POS_Z_FLOAT_ENA           POS_Z_FLOAT                      32          VGPR18
             POS_W_FLOAT_ENA           POS_W_FLOAT                      32          VGPR19
             FRONT_FACE_ENA            FRONT_FACE                       32          VGPR20
             ANCILLARY_ENA             RTA_Index[28:16],                29          VGPR21
                                       Sample_Num[11:8],
                                       Eye_id[7],
                                       VRSrateY[5:4],
                                       VRSrateX[3:2],
                                       Prim Typ[1:0]

             Field Name                    IJ / VGPR Terms                     BITS    VGPR Dest with Full
                                                                                       Load
             SAMPLE_COVERAGE_ENA           SAMPLE_COVERAGE                     16      VGPR22
             POS_FIXED_PT_ENA              Position {Y[16], X[16]}             32      VGPR23

The above table shows VGPR destinations for PS when all possible terms are enabled. If PS_INPUT_ADDR ==
PS_INPUT_ENA, then PS VGPRs pack towards VGPR0 as terms are disabled, as shown in the table below:

Field Name                             ENA        ADDR       IJ / VGPR Terms                    VGPR Dest
PERSP_SAMPLE_ENA                       1          1          PERSP_SAMPLE I                     VGPR0
                                                             PERSP_SAMPLE J                     VGPR1
PERSP_CENTER_ENA                       1          1          PERSP_CENTER I                     VGPR2
                                                             PERSP_CENTER J                     VGPR3
PERSP_CENTROID_ENA                     0          0          PERSP_CENTROID I                   X
                                                             PERSP_CENTROID J                   X
PERSP_PULL_MODEL_ENA                   0          0          PERSP_PULL_MODEL I/W               X
                                                             PERSP_PULL_MODEL J/W               X
                                                             PERSP_PULL_MODEL 1/W               X
LINEAR_SAMPLE_ENA                      0          0          LINEAR_SAMPLE I                    X
                                                             LINEAR_SAMPLE J                    X
LINEAR_CENTER_ENA                      0          0          LINEAR_CENTER I                    X
                                                             LINEAR_CENTER J                    X
LINEAR_CENTROID_ENA                    0          0          LINEAR_CENTROID I                  X
                                                             LINEAR_CENTROID J                  X
LINE_STIPPLE_TEX_ENA                   0          0          LINE_STIPPLE_TEX                   X
POS_X_FLOAT_ENA                        1          1          POS_X_FLOAT                        VGPR4
POS_Y_FLOAT_ENA                        1          1          POS_Y_FLOAT                        VGPR5
POS_Z_FLOAT_ENA                        0          0          POS_Z_FLOAT                        X
POS_W_FLOAT_ENA                        0          0          POS_W_FLOAT                        X
FRONT_FACE_ENA                         0          0          FRONT_FACE                         X
ANCILLARY_ENA                          0          0          Ancil Data                         X
SAMPLE_COVERAGE_ENA                    0          0          SAMPLE_COVERAGE                    X
POS_FIXED_PT_ENA                       0          0          Position {Y[16], X[16]}            X

However, if PS_INPUT_ADDR != PS_INPUT_ENA then the VGPR destination of enabled terms can be
manipulated. An example is this is shown in the table below:

Field Name                             ENA        ADDR       IJ / VGPR Terms                    VGPR Dest
PERSP_SAMPLE_ENA                       1          1          PERSP_SAMPLE I                     VGPR0
                                                             PERSP_SAMPLE J                     VGPR1
PERSP_CENTER_ENA                       1          1          PERSP_CENTER I                     VGPR2
                                                             PERSP_CENTER J                     VGPR3
PERSP_CENTROID_ENA                     0          1          PERSP_CENTROID I                   VGPR4 skipped
                                                             PERSP_CENTROID J                   VGPR5 skipped
PERSP_PULL_MODEL_ENA                   0          1          PERSP_PULL_MODEL I/W               VGPR6 skipped
                                                             PERSP_PULL_MODEL J/W               VGPR7 skipped
                                                             PERSP_PULL_MODEL 1/W               VGPR8 skipped
LINEAR_SAMPLE_ENA                      0          0          LINEAR_SAMPLE I                    X
                                                             LINEAR_SAMPLE J                    X
