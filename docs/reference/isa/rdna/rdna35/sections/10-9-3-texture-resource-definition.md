# 10.9.3. Texture Resource Definition

> RDNA3.5 ISA — pages 120–120

NSA and A16:
    • A16=0, MIMG-NSA specifies 5 groups of consecutive VGPRs: node_pointer, ray_extent, ray_origin, ray_dir
      and ray_inv_dir.
    • A16=1, MIMG-NSA specifies 4 groups. In the above set, ray_dir and ray_inv_dir are packed into 3 VGPRs.

When using A16=1 mode, ray-dir and ray-inv-dir share the same vgprs and ADDR4 is unused.

10.9.3. Texture Resource Definition
The T# used with these instructions is different from other image instructions.

                                              Table 54. BVH Resource Definition
Field            Bits        Size         Data
Base Address     39:0        40           Base address of the BVH texture 256 byte aligned
Reserved         54:40       15           Set to zero
Box growing      62:55       8            Number of ULPs to be added during ray-box test, encoded as unsigned integer
amount
Box sorting      63          1            Whether the ray-box test result need to be sorted
enable
Size             105:64      42           Number of nodes minus 1 in the BVH texture used to enforce bounds checking
Reserved         118:106     13           Set to zero
Pointer Flags    119         1            0: Do not use pointer flags or features supported by point flags
                                          1: Utilize pointer flags to enable HW winding, backface cull, opaque/non-opaque
                                          culling and primitive type-based culling.
triangle_return 120          1            0: Return data for triangle tests are
_mode                                     {0: t_num, 1: t_denom, 2: triangle_id, 3: hit_status}
                                          1: Return data for triangle tests are
                                          {0: t_num, 1: t_denom, 2: I_num, 3: J_num}
llc_stream or    122:121     2            0: use the LLC for load/store if enabled in Mtype
unused                                    1: use the LLC for load, bypass for store/atomics (store/atomics probe-invalidate)
                                          2: Reserved
                                          3: bypass the LLC for all ops
big_page         123         1            Describes resource page usage
                                          0 : No page size override.
                                          1 : Indicates when a whole resource is only using pages that are >= 64kB in size.
Type             127:124     4            Set to 0x8

Barycentrics
The ray-tracing hardware is designed to support computation of barycentric coordinates directly in hardware.
This uses the "triangle_return_mode" in the table in the previous section (T# descriptor).

                                              Table 55. Ray Tracing Return Mode
DWORD      Return Mode =0                                                 Return Mode = 1
           Field Name            Type                                     Field Name              Type
0          t_num                 float32                                  t_num                   float32
1          t_denom               float32                                  t_denom                 float32
2          triangle_id           uint32                                   I_num                   float32
3          hit_status            uint32 (boolean value)                   J_num                   float32
