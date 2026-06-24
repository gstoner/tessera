# 10.1.1. Texture Fault Enable (TFE) and LOD Warning Enable (LWE)

> RDNA3 ISA — pages 105–105

Instruction Fields
DMASK          4           Data VGPR enable mask: 1 .. 4 consecutive VGPRs
                           Loads: defines which components are returned: 0=red,1=green,2=blue,3=alpha
                           Stores: defines which components are written with data from VGPRs (missing components get 0).
                           Enabled components come from consecutive VGPRs.
                           E.G. DMASK=1001 : Red is in VGPRn and alpha in VGPRn+1.

                           For D16 loads, DMASK indicates which components to return;
                           For D16 stores, the DMASK the mask indicates which components to store but has restrictions:
                           Data is read out of consecutive VGPRs: LSB’s of VDATA, then MSB’s of VDATA then LSB’s
                           of VDATA+1 and last if needed MSB’s of VDATA+1. This is regardless of which DMASK bits
                           are set, only how many bits are set. The position of the DMASK bits controls which components
                           are written in memory.
                           If DMASK==0, the TA overrides DMASK=1 and puts zeros in VGPR followed by LWE status if exists. TFE
                           status is not generated since the fetch is dropped.
                           For IMAGE_GATHER4* instructions, DMASK indicates which component (RGBA), and the
                           number of VGPRs to use is determined automatically by hardware (4 VGPRs when D16=0, and 2
                           VGPRs when D16=1).
GLC            1           Group Level Coherent.
                           Atomics:
                           1 = return the memory value before the atomic operation is performed.
                           0 = do not return anything.
DLC            1           Device Level Coherent. Controls behavior of L1 cache (GL1).
SLC            1           System Level Coherent.
TFE            1           Texel Fault Enable for PRT (Partially Resident Textures). When set, fetch may return a NACK that
                           causes a VGPR write into DST+1 (first GPR after all fetch-dest gprs).
LWE            1           LOD Warning Enable. When set to 1, a texture fetch may return "LOD_CLAMPED = 1", and causes
                           a VGPR write into DST+1 (first GPR after all fetch-dest gprs). LWE only works for sampler ops;
                           LWE is ignored for non-sampler ops.
D16            1           VGPR-Data-16bit. On loads, convert data in memory to 16-bit format before storing it in VGPRs.
                           For stores, convert 16-bit data in VGPRs to the memory format before going to memory. Whether
                           the data is treated as float or int is decided by NFMT. Allowed only with these opcodes:

                             • IMAGE_SAMPLE*
                             • IMAGE_GATHER4
                             • IMAGE_LOAD
                             • IMAGE_LOAD_MIP
                             • IMAGE_STORE
                             • IMAGE_STORE_MIP
NSA            1           Non-Sequential Address
                           When NSA=0, the image addresses must be in sequential VGPRs starting at 'VADDR'.
                           When NSA=1, the instruction encoding allows up to 5 address components to be specified
                           separately by using an additional instruction DWORD.
ADDR1-4        4x8         Four 8-bit VGPR address fields, used by NSA. The "VADDR" field provides ADDR0.

10.1.1. Texture Fault Enable (TFE) and LOD Warning Enable (LWE)
This is related to "Partially Resident Textures".

When either of these bits are set in the instruction, any texture fetch may return one extra VGPR after all of the
data-return VGPRs. This data is returned uniquely to each thread and indicates the error / warning status of
that thread.
