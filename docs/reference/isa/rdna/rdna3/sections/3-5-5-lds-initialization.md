# 3.5.5. LDS Initialization

> RDNA3 ISA — pages 42–42

Field Name                             ENA   ADDR    IJ / VGPR Terms                     VGPR Dest
LINEAR_CENTROID_ENA                    0     0       LINEAR_CENTROID I                   X
                                                     LINEAR_CENTROID J                   X
LINE_STIPPLE_TEX_ENA                   0     0       LINE_STIPPLE_TEX                    X
POS_X_FLOAT_ENA                        1     1       POS_X_FLOAT                         VGPR4
POS_Y_FLOAT_ENA                        1     1       POS_Y_FLOAT                         VGPR5
POS_Z_FLOAT_ENA                        0     0       POS_Z_FLOAT                         X
POS_W_FLOAT_ENA                        0     0       POS_W_FLOAT                         X
FRONT_FACE_ENA                         0     0       FRONT_FACE                          X
ANCILLARY_ENA                          0     0       Ancil Data                          X
SAMPLE_COVERAGE_ENA                    0     0       SAMPLE_COVERAGE                     X
POS_FIXED_PT_ENA                       0     0       Position {Y[16], X[16]}             X

However, if PS_INPUT_ADDR != PS_INPUT_ENA then the VGPR destination of enabled terms can be
manipulated. An example is this is shown in the table below:

Field Name                             ENA   ADDR    IJ / VGPR Terms                     VGPR Dest
PERSP_SAMPLE_ENA                       1     1       PERSP_SAMPLE I                      VGPR0
                                                     PERSP_SAMPLE J                      VGPR1
PERSP_CENTER_ENA                       1     1       PERSP_CENTER I                      VGPR2
                                                     PERSP_CENTER J                      VGPR3
PERSP_CENTROID_ENA                     0     1       PERSP_CENTROID I                    VGPR4 skipped
                                                     PERSP_CENTROID J                    VGPR5 skipped
PERSP_PULL_MODEL_ENA                   0     1       PERSP_PULL_MODEL I/W                VGPR6 skipped
                                                     PERSP_PULL_MODEL J/W                VGPR7 skipped
                                                     PERSP_PULL_MODEL 1/W                VGPR8 skipped
LINEAR_SAMPLE_ENA                      0     0       LINEAR_SAMPLE I                     X
                                                     LINEAR_SAMPLE J                     X
LINEAR_CENTER_ENA                      0     0       LINEAR_CENTER I                     X
                                                     LINEAR_CENTER J                     X
LINEAR_CENTROID_ENA                    0     1       LINEAR_CENTROID I                   VGPR9 skipped
                                                     LINEAR_CENTROID J                   VGPR10 skipped
LINE_STIPPLE_TEX_ENA                   0     1       LINE_STIPPLE_TEX                    VGPR11 skipped
POS_X_FLOAT_ENA                        1     1       POS_X_FLOAT                         VGPR12
POS_Y_FLOAT_ENA                        1     1       POS_Y_FLOAT                         VGPR13
POS_Z_FLOAT_ENA                        0     0       POS_Z_FLOAT                         X
POS_W_FLOAT_ENA                        0     0       POS_W_FLOAT                         X
FRONT_FACE_ENA                         0     0       FRONT_FACE                          X
ANCILLARY_ENA                          0     0       Ancil Data                          X
SAMPLE_COVERAGE_ENA                    0     0       SAMPLE_COVERAGE                     X
POS_FIXED_PT_ENA                       0     0       Position {Y[16], X[16]}             X

3.5.5. LDS Initialization
Only pixel shader (PS) waves have LDS pre-initialized with data before the wave launches. For PS wave, LDS is
preloaded with vertex parameter data that can be interpolated using barycentrics (I and J) to compute per-pixel
parameters.
