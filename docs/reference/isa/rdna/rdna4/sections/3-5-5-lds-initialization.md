# 3.5.5. LDS Initialization

> RDNA4 ISA — pages 47–47

Field Name                             ENA   ADDR    IJ / VGPR Terms                     VGPR Dest
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
parameters. This data may be loaded before the wave launches, or after launch. When it is loaded after wave
launch, DS_PARAM_LOAD instruction stall until the parameter data is loaded into LDS.
