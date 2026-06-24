# 3.5.5. LDS Initialization

> RDNA3.5 ISA — pages 43–43

Field Name                               ENA   ADDR   IJ / VGPR Terms                    VGPR Dest
ANCILLARY_ENA                            0     0      Ancil Data                         X
SAMPLE_COVERAGE_ENA                      0     0      SAMPLE_COVERAGE                    X
POS_FIXED_PT_ENA                         0     0      Position {Y[16], X[16]}            X

3.5.5. LDS Initialization
Only pixel shader (PS) waves have LDS pre-initialized with data before the wave launches. For PS wave, LDS is
preloaded with vertex parameter data that can be interpolated using barycentrics (I and J) to compute per-pixel
parameters.
