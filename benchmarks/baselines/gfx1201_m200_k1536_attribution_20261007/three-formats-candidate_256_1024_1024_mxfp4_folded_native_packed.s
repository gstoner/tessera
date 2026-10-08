
/tmp/tmpxyai24ub.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <packed_folded_w4a8>:
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v3, 4, v0       // 000000001b00: ca220080 02020084
	v_lshrrev_b32_e32 v1, 2, v0                                // 000000001b08: 32020082
	s_mov_b32 s10, ttmp9                                       // 000000001b0c: be8a0075
	s_ashr_i32 s11, ttmp9, 31                                  // 000000001b10: 860b9f75
	v_and_b32_e32 v74, 0xc0, v0                                // 000000001b14: 369400ff 000000c0
	s_lshl_b64 s[24:25], s[10:11], 6                           // 000000001b1c: 8498860a
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b20: bf870009
	v_dual_mov_b32 v10, s25 :: v_dual_and_b32 v3, 48, v3       // 000000001b24: ca240019 0a0206b0
	v_mul_u32_u24_e32 v5, 0x50, v1                             // 000000001b2c: 160a02ff 00000050
	s_clause 0x4                                               // 000000001b34: bf850004
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b38: f4002080 f80000d8
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b40: f4004400 f80000c8
	s_load_b64 s[6:7], s[0:1], 0x8                             // 000000001b48: f4002180 f8000008
	s_load_b64 s[4:5], s[0:1], 0x30                            // 000000001b50: f4002100 f8000030
	s_load_b64 s[8:9], s[0:1], 0x80                            // 000000001b58: f4002200 f8000080
	v_and_b32_e32 v70, 15, v0                                  // 000000001b60: 368c008f
	v_or_b32_e32 v82, 16, v74                                  // 000000001b64: 38a49490
	v_or_b32_e32 v104, 32, v74                                 // 000000001b68: 38d094a0
	v_add_nc_u32_e32 v72, v5, v3                               // 000000001b6c: 4a900705
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b70: 320a0081
	v_and_b32_e32 v6, 0xcf, v0                                 // 000000001b74: 360c00ff 000000cf
	s_mov_b32 s12, ttmp7                                       // 000000001b7c: be8c0073
	v_or_b32_e32 v7, v104, v70                                 // 000000001b80: 380e8d68
	s_ashr_i32 s13, ttmp7, 31                                  // 000000001b84: 860d9f73
	v_and_b32_e32 v75, 8, v5                                   // 000000001b88: 36960a88
	v_mul_u32_u24_e32 v5, 0x50, v6                             // 000000001b8c: 160a0cff 00000050
	v_or_b32_e32 v6, v82, v70                                  // 000000001b94: 380c8d52
	s_lshl_b64 s[14:15], s[12:13], 8                           // 000000001b98: 848e880c
	v_dual_mov_b32 v4, v2 :: v_dual_and_b32 v71, 32, v0        // 000000001b9c: ca240102 044600a0
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000001ba4: bf870193
	v_or_b32_e32 v73, v75, v5                                  // 000000001ba8: 38920b4b
	v_mul_u32_u24_e32 v6, 0x50, v6                             // 000000001bac: 160c0cff 00000050
	v_mul_u32_u24_e32 v5, 0x50, v7                             // 000000001bb4: 160a0eff 00000050
	v_or_b32_e32 v11, s14, v1                                  // 000000001bbc: 3816020e
	v_or_b32_e32 v130, 48, v74                                 // 000000001bc0: 390494b0
	s_wait_kmcnt 0x0                                           // 000000001bc4: bfc70000
	s_lshr_b64 s[10:11], s[2:3], 5                             // 000000001bc8: 858a8502
	v_or_b32_e32 v76, v6, v75                                  // 000000001bcc: 38989706
	v_or_b32_e32 v77, v5, v75                                  // 000000001bd0: 389a9705
	v_mul_lo_u32 v13, s3, v11                                  // 000000001bd4: d72c000d 02021603
	v_mad_co_u64_u32 v[5:6], null, s2, v11, v[3:4]             // 000000001bdc: d6fe7c05 040e1602
	s_mul_u64 s[10:11], s[10:11], s[18:19]                     // 000000001be4: aa8a120a
	v_or_b32_e32 v7, v130, v70                                 // 000000001be8: 380e8d82
	v_dual_mov_b32 v35, v2 :: v_dual_and_b32 v8, 47, v0        // 000000001bec: ca240102 230800af
	v_or_b32_e32 v9, s24, v1                                   // 000000001bf4: 38120218
	s_add_nc_u64 s[20:21], s[8:9], s[10:11]                    // 000000001bf8: a9940a08
	s_lshr_b64 s[10:11], s[2:3], 4                             // 000000001bfc: 858a8402
	s_mul_i32 s11, s2, s15                                     // 000000001c00: 960b0f02
	v_mul_u32_u24_e32 v7, 0x50, v7                             // 000000001c04: 160e0eff 00000050
	v_mul_u32_u24_e32 v8, 0x50, v8                             // 000000001c0c: 161010ff 00000050
	v_or_b32_e32 v14, 64, v11                                  // 000000001c14: 381c16c0
	v_add3_u32 v6, v13, v6, s11                                // 000000001c18: d6550006 002e0d0d
	v_add_co_u32 v66, vcc_lo, s20, v9                          // 000000001c20: d7006a42 02021214
	s_delay_alu instid0(valu_dep_1)                            // 000000001c28: bf870001
	v_add_co_ci_u32_e64 v67, null, s21, v10, vcc_lo            // 000000001c2c: d5207c43 01aa1415
	v_add_co_u32 v79, vcc_lo, s6, v5                           // 000000001c34: d7006a4f 02020a06
	v_mul_lo_u32 v16, s3, v14                                  // 000000001c3c: d72c0010 02021c03
	s_wait_alu depctr_va_vcc(0)                                // 000000001c44: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, s7, v6, vcc_lo              // 000000001c48: d5207c50 01aa0c07
	v_mov_b32_e32 v6, v2                                       // 000000001c50: 7e0c0302
	v_or_b32_e32 v78, v7, v75                                  // 000000001c54: 389c9707
	v_or_b32_e32 v15, v75, v8                                  // 000000001c58: 381e114b
	v_mad_co_u64_u32 v[7:8], null, s2, v14, v[3:4]             // 000000001c5c: d6fe7c07 040e1c02
	v_or_b32_e32 v13, 0x80, v11                                // 000000001c64: 381a16ff 00000080
	v_and_b32_e32 v5, 3, v0                                    // 000000001c6c: 360a0083
	v_alignbit_b32 v10, v10, v9, 4                             // 000000001c70: d616000a 0212130a
	s_lshr_b32 s12, s3, 4                                      // 000000001c78: 850c8403
	v_or_b32_e32 v11, 0xc0, v11                                // 000000001c7c: 381616ff 000000c0
	v_or_b32_e32 v162, 16, v71                                 // 000000001c84: 39448e90
	v_dual_mov_b32 v37, v2 :: v_dual_add_nc_u32 v88, 0x5000, v15// 000000001c88: ca200102 25581eff 00005000
	v_add3_u32 v14, v16, v8, s11                               // 000000001c94: d655000e 002e1110
	v_mul_lo_u32 v16, s3, v13                                  // 000000001c9c: d72c0010 02021a03
	v_mad_co_u64_u32 v[8:9], null, s2, v13, v[3:4]             // 000000001ca4: d6fe7c08 040e1a02
	v_mul_lo_u32 v13, s12, v10                                 // 000000001cac: d72c000d 0202140c
	v_mad_co_u64_u32 v[5:6], null, s10, v10, v[5:6]            // 000000001cb4: d6fe7c05 0416140a
	s_lshr_b32 s12, s25, 4                                     // 000000001cbc: 850c8419
	v_add_co_u32 v81, vcc_lo, s6, v7                           // 000000001cc0: d7006a51 02020e06
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cc8: bf88ff9e
	s_mul_i32 s10, s10, s12                                    // 000000001ccc: 960a0c0a
	v_bfe_u32 v7, v0, 1, 1                                     // 000000001cd0: d6100007 02050300
	v_add3_u32 v9, v16, v9, s11                                // 000000001cd8: d6550009 002e1310
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce0: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s7, v14, vcc_lo             // 000000001ce4: d5207c53 01aa1c07
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cec: bf88ff9e
	v_add3_u32 v6, v13, v6, s10                                // 000000001cf0: d6550006 002a0d0d
	v_mul_lo_u32 v13, s3, v11                                  // 000000001cf8: d72c000d 02021603
	v_mad_co_u64_u32 v[10:11], null, s2, v11, v[3:4]           // 000000001d00: d6fe7c0a 040e1602
	v_mad_co_u64_u32 v[3:4], null, s18, v7, v[1:2]             // 000000001d08: d6fe7c03 04060e12
	v_add_co_u32 v84, vcc_lo, s6, v8                           // 000000001d10: d7006a54 02021006
	v_lshlrev_b64_e32 v[5:6], 7, v[5:6]                        // 000000001d18: 3e0a0a87
	s_wait_alu depctr_va_vcc(0)                                // 000000001d1c: bf88ff9d
	v_add_co_ci_u32_e64 v85, null, s7, v9, vcc_lo              // 000000001d20: d5207c55 01aa1207
	s_add_nc_u64 s[2:3], s[8:9], s[24:25]                      // 000000001d28: a9821808
	v_add3_u32 v1, v13, v11, s11                               // 000000001d2c: d6550001 002e170d
	v_add_co_u32 v86, vcc_lo, s6, v10                          // 000000001d34: d7006a56 02021406
	v_and_or_b32 v0, v0, 60, v5                                // 000000001d3c: d6570000 04157900
	v_mad_co_u64_u32 v[4:5], null, s19, v7, v[4:5]             // 000000001d44: d6fe7c04 04120e13
	s_wait_alu depctr_va_vcc(0)                                // 000000001d4c: bf88ff9d
	v_add_co_ci_u32_e64 v87, null, s7, v1, vcc_lo              // 000000001d50: d5207c57 01aa0207
	v_mov_b32_e32 v7, v2                                       // 000000001d58: 7e0e0302
	v_add_co_u32 v5, vcc_lo, s4, v0                            // 000000001d5c: d7006a05 02020004
	s_wait_alu depctr_va_vcc(0)                                // 000000001d64: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s5, v6, vcc_lo               // 000000001d68: d5207c06 01aa0c05
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d70: bf88ff9e
	v_add_co_u32 v0, vcc_lo, s2, v3                            // 000000001d74: d7006a00 02020602
	s_wait_alu depctr_va_vcc(0)                                // 000000001d7c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s3, v4, vcc_lo               // 000000001d80: d5207c01 01aa0803
	v_add_co_u32 v68, vcc_lo, 0x43, v5                         // 000000001d88: d7006a44 02020aff 00000043
	v_mov_b32_e32 v5, v2                                       // 000000001d94: 7e0a0302
	v_or_b32_e32 v12, v162, v70                                // 000000001d98: 38188da2
	s_wait_alu depctr_va_vcc(0)                                // 000000001d9c: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v6, vcc_lo               // 000000001da0: d5207c45 01aa0c80
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2             // 000000001da8: ca100102 03040102
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_4)// 000000001db0: bf870243
	v_mul_u32_u24_e32 v12, 0x50, v12                           // 000000001db4: 161818ff 00000050
	v_mov_b32_e32 v6, v2                                       // 000000001dbc: 7e0c0302
	v_dual_mov_b32 v8, v2 :: v_dual_mov_b32 v39, v2            // 000000001dc0: ca100102 08260102
	v_mov_b32_e32 v9, v2                                       // 000000001dc8: 7e120302
	v_or_b32_e32 v12, v12, v75                                 // 000000001dcc: 3818970c
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v41, v2           // 000000001dd0: ca100102 22280102
	v_dual_mov_b32 v36, v2 :: v_dual_mov_b32 v59, v2           // 000000001dd8: ca100102 243a0102
	s_delay_alu instid0(valu_dep_3)                            // 000000001de0: bf870003
	v_dual_mov_b32 v38, v2 :: v_dual_add_nc_u32 v89, 0x5000, v12// 000000001de4: ca200102 265818ff 00005000
	v_dual_mov_b32 v61, v2 :: v_dual_mov_b32 v40, v2           // 000000001df0: ca100102 3d280102
	v_dual_mov_b32 v63, v2 :: v_dual_mov_b32 v58, v2           // 000000001df8: ca100102 3f3a0102
	v_dual_mov_b32 v65, v2 :: v_dual_mov_b32 v60, v2           // 000000001e00: ca100102 413c0102
	v_dual_mov_b32 v27, v2 :: v_dual_mov_b32 v62, v2           // 000000001e08: ca100102 1b3e0102
	v_dual_mov_b32 v29, v2 :: v_dual_mov_b32 v64, v2           // 000000001e10: ca100102 1d400102
	v_dual_mov_b32 v31, v2 :: v_dual_mov_b32 v26, v2           // 000000001e18: ca100102 1f1a0102
	v_dual_mov_b32 v33, v2 :: v_dual_mov_b32 v28, v2           // 000000001e20: ca100102 211c0102
	v_dual_mov_b32 v51, v2 :: v_dual_mov_b32 v30, v2           // 000000001e28: ca100102 331e0102
	v_dual_mov_b32 v53, v2 :: v_dual_mov_b32 v32, v2           // 000000001e30: ca100102 35200102
	v_dual_mov_b32 v55, v2 :: v_dual_mov_b32 v50, v2           // 000000001e38: ca100102 37320102
	v_dual_mov_b32 v57, v2 :: v_dual_mov_b32 v52, v2           // 000000001e40: ca100102 39340102
	v_dual_mov_b32 v19, v2 :: v_dual_mov_b32 v54, v2           // 000000001e48: ca100102 13360102
	v_dual_mov_b32 v21, v2 :: v_dual_mov_b32 v56, v2           // 000000001e50: ca100102 15380102
	v_dual_mov_b32 v23, v2 :: v_dual_mov_b32 v18, v2           // 000000001e58: ca100102 17120102
	v_dual_mov_b32 v25, v2 :: v_dual_mov_b32 v20, v2           // 000000001e60: ca100102 19140102
	v_dual_mov_b32 v43, v2 :: v_dual_mov_b32 v22, v2           // 000000001e68: ca100102 2b160102
	v_dual_mov_b32 v45, v2 :: v_dual_mov_b32 v24, v2           // 000000001e70: ca100102 2d180102
	v_dual_mov_b32 v47, v2 :: v_dual_mov_b32 v42, v2           // 000000001e78: ca100102 2f2a0102
	v_dual_mov_b32 v49, v2 :: v_dual_mov_b32 v44, v2           // 000000001e80: ca100102 312c0102
	v_dual_mov_b32 v11, v2 :: v_dual_mov_b32 v46, v2           // 000000001e88: ca100102 0b2e0102
	v_dual_mov_b32 v13, v2 :: v_dual_mov_b32 v48, v2           // 000000001e90: ca100102 0d300102
	v_dual_mov_b32 v15, v2 :: v_dual_mov_b32 v10, v2           // 000000001e98: ca100102 0f0a0102
	v_dual_mov_b32 v17, v2 :: v_dual_mov_b32 v12, v2           // 000000001ea0: ca100102 110c0102
	v_mov_b32_e32 v14, v2                                      // 000000001ea8: 7e1c0302
	v_mov_b32_e32 v16, v2                                      // 000000001eac: 7e200302
	s_lshl_b64 s[22:23], s[18:19], 1                           // 000000001eb0: 84968112
	s_mov_b64 s[26:27], 0                                      // 000000001eb4: be9a0180
	global_load_u8 v103, v[0:1], off                           // 000000001eb8: ee04007c 00000067 00000000
	global_load_u8 v109, v[66:67], off                         // 000000001ec4: ee04007c 0000006d 00000042
	global_load_d16_u8 v102, v[0:1], off                       // 000000001ed0: ee07807c 00000066 00000000
	global_load_d16_hi_u8 v102, v[66:67], off                  // 000000001edc: ee08407c 00000066 00000042
	v_add_co_u32 v90, vcc_lo, v79, s26                         // 000000001ee8: d7006a5a 0200354f
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef0: bf88ff9d
	v_add_co_ci_u32_e64 v91, null, s27, v80, vcc_lo            // 000000001ef4: d5207c5b 01aaa01b
	v_add_co_u32 v94, s2, v81, s26                             // 000000001efc: d700025e 02003551
	s_wait_alu depctr_va_sdst(0)                               // 000000001f04: bf88f19f
	v_add_co_ci_u32_e64 v95, null, s27, v83, s2                // 000000001f08: d5207c5f 000aa61b
	global_load_b128 v[90:93], v[90:91], off                   // 000000001f10: ee05c07c 0000005a 0000005a
	s_clause 0x1                                               // 000000001f1c: bf850001
	global_load_b32 v110, v[68:69], off offset:-67             // 000000001f20: ee05007c 0000006e ffffbd44
	global_load_b32 v111, v[68:69], off offset:-3              // 000000001f2c: ee05007c 0000006f fffffd44
	global_load_b128 v[94:97], v[94:95], off                   // 000000001f38: ee05c07c 0000005e 0000005e
	v_add_co_u32 v98, s3, v84, s26                             // 000000001f44: d7000362 02003554
	v_add_co_u32 v105, s4, v86, s26                            // 000000001f4c: d7000469 02003556
	s_wait_alu depctr_va_sdst(0)                               // 000000001f54: bf88f19f
	v_add_co_ci_u32_e64 v99, null, s27, v85, s3                // 000000001f58: d5207c63 000eaa1b
	v_add_co_ci_u32_e64 v106, null, s27, v87, s4               // 000000001f60: d5207c6a 0012ae1b
	v_add_co_u32 v0, vcc_lo, v0, s22                           // 000000001f68: d7006a00 02002d00
	s_clause 0x1                                               // 000000001f70: bf850001
	global_load_b128 v[98:101], v[98:99], off                  // 000000001f74: ee05c07c 00000062 00000062
	global_load_b128 v[105:108], v[105:106], off               // 000000001f80: ee05c07c 00000069 00000069
	s_wait_alu depctr_va_vcc(0)                                // 000000001f8c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s23, v1, vcc_lo              // 000000001f90: d5207c01 01aa0217
	s_barrier_signal -1                                        // 000000001f98: be804ec1
	s_barrier_wait 0xffff                                      // 000000001f9c: bf94ffff
	v_add_co_u32 v68, s2, 0x200, v68                           // 000000001fa0: d7000244 020288ff 00000200
	s_wait_alu depctr_va_sdst(0)                               // 000000001fac: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v69, s2                  // 000000001fb0: d5207c45 000a8a80
	s_add_nc_u64 s[26:27], s[26:27], 64                        // 000000001fb8: a99ac01a
	s_wait_loadcnt 0x8                                         // 000000001fbc: bfc00008
	v_sub_nc_u32_e32 v112, v109, v103                          // 000000001fc0: 4ce0cf6d
	s_wait_loadcnt 0x6                                         // 000000001fc4: bfc00006
	v_cmp_eq_u16_e64 s2, 0, v102.l                             // 000000001fc8: d43a0002 0202cc80
	v_cmp_eq_u16_e32 vcc_lo, v102.h, v102.l                    // 000000001fd0: 7c74cde6
	s_delay_alu instid0(valu_dep_3)                            // 000000001fd4: bf870003
	v_cmp_ne_u32_e64 s3, 2, v112                               // 000000001fd8: d44d0003 0202e082
	v_cmp_ne_u32_e64 s4, 3, v112                               // 000000001fe0: d44d0004 0202e083
	s_wait_alu depctr_va_vcc(0)                                // 000000001fe8: bf88ff9d
	v_cndmask_b32_e64 v134, 0, 0x3c383000, vcc_lo              // 000000001fec: d5010086 01a9fe80 3c383000
	v_cndmask_b32_e64 v135, 0, 0x4c484440, vcc_lo              // 000000001ff8: d5010087 01a9fe80 4c484440
	v_cmp_ne_u32_e32 vcc_lo, 1, v112                           // 000000002004: 7c9ae081
	v_cmp_ne_u32_e64 s5, 4, v112                               // 000000002008: d44d0005 0202e084
	s_wait_loadcnt 0x5                                         // 000000002010: bfc00005
	ds_store_b128 v72, v[90:93]                                // 000000002014: db7c0000 00005a48
	v_cmp_ne_u32_e64 s6, 5, v112                               // 00000000201c: d44d0006 0202e085
	v_cmp_ne_u32_e64 s7, 6, v112                               // 000000002024: d44d0007 0202e086
	s_wait_alu depctr_va_vcc(0)                                // 00000000202c: bf88ff9d
	v_cndmask_b32_e32 v92, 0x44403c38, v135, vcc_lo            // 000000002030: 02b90eff 44403c38
	v_cndmask_b32_e32 v93, 0x34302800, v134, vcc_lo            // 000000002038: 02bb0cff 34302800
	v_cmp_ne_u32_e64 s8, 7, v112                               // 000000002040: d44d0008 0202e087
	v_cmp_ne_u32_e64 s9, 8, v112                               // 000000002048: d44d0009 0202e088
	v_cmp_ne_u32_e64 s10, 9, v112                              // 000000002050: d44d000a 0202e089
	s_wait_alu depctr_va_sdst(0)                               // 000000002058: bf88f19f
	v_cndmask_b32_e64 v92, 0x3c383430, v92, s3                 // 00000000205c: d501005c 000eb8ff 3c383430
	v_cndmask_b32_e64 v93, 0x2c282000, v93, s3                 // 000000002068: d501005d 000ebaff 2c282000
	v_cmp_ne_u32_e64 s11, 10, v112                             // 000000002074: d44d000b 0202e08a
	v_cmp_ne_u32_e64 s12, 11, v112                             // 00000000207c: d44d000c 0202e08b
	v_cmp_ne_u32_e64 s13, 12, v112                             // 000000002084: d44d000d 0202e08c
	v_cndmask_b32_e64 v92, 0x34302c28, v92, s4                 // 00000000208c: d501005c 0012b8ff 34302c28
	v_cndmask_b32_e64 v93, 0x24201800, v93, s4                 // 000000002098: d501005d 0012baff 24201800
	s_wait_loadcnt 0x2                                         // 0000000020a4: bfc00002
	ds_store_b128 v72, v[94:97] offset:5120                    // 0000000020a8: db7c1400 00005e48
	v_lshrrev_b32_e32 v118, 1, v110                            // 0000000020b0: 32ecdc81
	v_lshrrev_b32_e32 v127, 5, v111                            // 0000000020b4: 32fede85
	v_cndmask_b32_e64 v92, 0x2c282420, v92, s5                 // 0000000020b8: d501005c 0016b8ff 2c282420
	v_cndmask_b32_e64 v93, 0x1c181000, v93, s5                 // 0000000020c4: d501005d 0016baff 1c181000
	v_lshrrev_b32_e32 v119, 5, v110                            // 0000000020d0: 32eedc85
	v_lshrrev_b32_e32 v128, 9, v111                            // 0000000020d4: 3300de89
	v_lshrrev_b32_e32 v120, 9, v110                            // 0000000020d8: 32f0dc89
	v_cndmask_b32_e64 v92, 0x24201c18, v92, s6                 // 0000000020dc: d501005c 001ab8ff 24201c18
	v_cndmask_b32_e64 v93, 0x14100800, v93, s6                 // 0000000020e8: d501005d 001abaff 14100800
	v_lshrrev_b32_e32 v129, 13, v111                           // 0000000020f4: 3302de8d
	s_and_b32 vcc_lo, s13, s12                                 // 0000000020f8: 8b6a0c0d
	v_lshrrev_b32_e32 v121, 13, v110                           // 0000000020fc: 32f2dc8d
	v_cndmask_b32_e64 v92, 0x1c181410, v92, s7                 // 000000002100: d501005c 001eb8ff 1c181410
	v_cndmask_b32_e64 v93, 0xc080400, v93, s7                  // 00000000210c: d501005d 001ebaff 0c080400
	v_lshrrev_b32_e32 v124, 25, v110                           // 000000002118: 32f8dc99
	v_lshrrev_b32_e32 v131, 17, v111                           // 00000000211c: 3306de91
	v_lshlrev_b32_e32 v117, 3, v110                            // 000000002120: 30eadc83
	v_cndmask_b32_e64 v92, 0x14100c08, v92, s8                 // 000000002124: d501005c 0022b8ff 14100c08
	v_cndmask_b32_e64 v93, 0x6040200, v93, s8                  // 000000002130: d501005d 0022baff 06040200
	v_lshrrev_b32_e32 v122, 17, v110                           // 00000000213c: 32f4dc91
	v_lshrrev_b32_e32 v132, 21, v111                           // 000000002140: 3308de95
	v_lshrrev_b32_e32 v113, 8, v110                            // 000000002144: 32e2dc88
	v_cndmask_b32_e64 v92, 0xc080604, v92, s9                  // 000000002148: d501005c 0026b8ff 0c080604
	v_cndmask_b32_e64 v93, 0x3020100, v93, s9                  // 000000002154: d501005d 0026baff 03020100
	v_lshrrev_b32_e32 v114, 24, v110                           // 000000002160: 32e4dc98
	v_lshrrev_b32_e32 v115, 8, v111                            // 000000002164: 32e6de88
	v_lshrrev_b32_e32 v116, 24, v111                           // 000000002168: 32e8de98
	v_cndmask_b32_e64 v92, 0x6040302, v92, s10                 // 00000000216c: d501005c 002ab8ff 06040302
	v_cndmask_b32_e64 v93, 0x2010000, v93, s10                 // 000000002178: d501005d 002abaff 02010000
	v_lshrrev_b32_e32 v123, 21, v110                           // 000000002184: 32f6dc95
	v_lshrrev_b32_e32 v126, 1, v111                            // 000000002188: 32fcde81
	v_lshrrev_b32_e32 v133, 25, v111                           // 00000000218c: 330ade99
	v_cndmask_b32_e64 v92, 0x3020201, v92, s11                 // 000000002190: d501005c 002eb8ff 03020201
	v_cndmask_b32_e64 v93, 0x1000000, v93, s11                 // 00000000219c: d501005d 002ebaff 01000000
	v_and_b32_e32 v118, 56, v118                               // 0000000021a8: 36ececb8
	v_and_b32_e32 v127, 56, v127                               // 0000000021ac: 36fefeb8
	v_and_b32_e32 v119, 56, v119                               // 0000000021b0: 36eeeeb8
	v_cndmask_b32_e64 v94, 0x2010100, v92, s12                 // 0000000021b4: d501005e 0032b8ff 02010100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021c0: bf88ff9e
	v_cndmask_b32_e32 v92, 0, v93, vcc_lo                      // 0000000021c4: 02b8ba80
	v_and_b32_e32 v128, 56, v128                               // 0000000021c8: 370100b8
	v_and_b32_e32 v120, 56, v120                               // 0000000021cc: 36f0f0b8
	v_and_b32_e32 v129, 56, v129                               // 0000000021d0: 370302b8
	v_cndmask_b32_e64 v93, 0x1000000, v94, s13                 // 0000000021d4: d501005d 0036bcff 01000000
	v_and_b32_e32 v121, 56, v121                               // 0000000021e0: 36f2f2b8
	v_and_b32_e32 v124, 56, v124                               // 0000000021e4: 36f8f8b8
	v_and_b32_e32 v131, 56, v131                               // 0000000021e8: 370706b8
	v_lshlrev_b32_e32 v125, 3, v111                            // 0000000021ec: 30fade83
	v_and_b32_e32 v122, 56, v122                               // 0000000021f0: 36f4f4b8
	v_and_b32_e32 v132, 56, v132                               // 0000000021f4: 370908b8
	v_lshrrev_b64 v[94:95], v117, v[92:93]                     // 0000000021f8: d73d005e 0202b975
	v_lshlrev_b16 v102.h, 4, v110.l op_sel:[0,0,1]             // 000000002200: d7384066 0202dc84
	v_and_b16 v102.l, 0x80, v110.l                             // 000000002208: d7620066 0202dcff 00000080
	v_lshlrev_b16 v103.l, 4, v110.h op_sel:[0,1,0]             // 000000002214: d7381067 0202dc84
	v_and_b16 v103.h, 0x80, v110.h op_sel:[0,1,1]              // 00000000221c: d7625067 0202dcff 00000080
	v_lshlrev_b16 v109.l, 4, v111.l                            // 000000002228: d738006d 0202de84
	v_and_b16 v109.h, 0x80, v111.l op_sel:[0,0,1]              // 000000002230: d762406d 0202deff 00000080
	v_lshlrev_b16 v110.l, 4, v111.h op_sel:[0,1,0]             // 00000000223c: d738106e 0202de84
	v_and_b16 v110.h, 0x80, v111.h op_sel:[0,1,1]              // 000000002244: d762506e 0202deff 00000080
	v_lshlrev_b16 v111.l, 4, v113.l                            // 000000002250: d738006f 0202e284
	v_and_b16 v111.h, 0x80, v113.l op_sel:[0,0,1]              // 000000002258: d762406f 0202e2ff 00000080
	v_and_b32_e32 v123, 56, v123                               // 000000002264: 36f6f6b8
	v_lshlrev_b16 v112.l, 4, v114.l                            // 000000002268: d7380070 0202e484
	v_and_b16 v112.h, 0x80, v114.l op_sel:[0,0,1]              // 000000002270: d7624070 0202e4ff 00000080
	v_and_b32_e32 v126, 56, v126                               // 00000000227c: 36fcfcb8
	v_lshlrev_b16 v113.l, 4, v115.l                            // 000000002280: d7380071 0202e684
	v_and_b16 v113.h, 0x80, v115.l op_sel:[0,0,1]              // 000000002288: d7624071 0202e6ff 00000080
	v_lshlrev_b16 v114.l, 4, v116.l                            // 000000002294: d7380072 0202e884
	v_and_b32_e32 v133, 56, v133                               // 00000000229c: 370b0ab8
	v_and_b16 v114.h, 0x80, v116.l op_sel:[0,0,1]              // 0000000022a0: d7624072 0202e8ff 00000080
	v_lshrrev_b64 v[95:96], v118, v[92:93]                     // 0000000022ac: d73d005f 0202b976
	v_lshrrev_b64 v[115:116], v127, v[92:93]                   // 0000000022b4: d73d0073 0202b97f
	v_lshrrev_b64 v[96:97], v119, v[92:93]                     // 0000000022bc: d73d0060 0202b977
	v_lshrrev_b64 v[116:117], v128, v[92:93]                   // 0000000022c4: d73d0074 0202b980
	s_wait_loadcnt 0x1                                         // 0000000022cc: bfc00001
	ds_store_b128 v72, v[98:101] offset:10240                  // 0000000022d0: db7c2800 00006248
	v_lshrrev_b64 v[97:98], v120, v[92:93]                     // 0000000022d8: d73d0061 0202b978
	v_lshrrev_b64 v[117:118], v129, v[92:93]                   // 0000000022e0: d73d0075 0202b981
	s_wait_loadcnt 0x0                                         // 0000000022e8: bfc00000
	ds_store_b128 v72, v[105:108] offset:15360                 // 0000000022ec: db7c3c00 00006948
	v_lshrrev_b64 v[98:99], v121, v[92:93]                     // 0000000022f4: d73d0062 0202b979
	v_lshrrev_b64 v[105:106], v124, v[92:93]                   // 0000000022fc: d73d0069 0202b97c
	v_lshrrev_b64 v[118:119], v131, v[92:93]                   // 000000002304: d73d0076 0202b983
	v_lshrrev_b64 v[99:100], v122, v[92:93]                    // 00000000230c: d73d0063 0202b97a
	v_lshrrev_b64 v[106:107], v125, v[92:93]                   // 000000002314: d73d006a 0202b97d
	v_lshrrev_b64 v[119:120], v132, v[92:93]                   // 00000000231c: d73d0077 0202b984
	v_lshrrev_b64 v[100:101], v123, v[92:93]                   // 000000002324: d73d0064 0202b97b
	v_lshrrev_b64 v[107:108], v126, v[92:93]                   // 00000000232c: d73d006b 0202b97e
	v_lshrrev_b64 v[120:121], v133, v[92:93]                   // 000000002334: d73d0078 0202b985
	v_and_b16 v102.h, 0x80, v102.h op_sel:[0,1,1]              // 00000000233c: d7625066 0202ccff 00000080
	v_and_b16 v103.l, 0x80, v103.l                             // 000000002348: d7620067 0202ceff 00000080
	v_and_b16 v109.l, 0x80, v109.l                             // 000000002354: d762006d 0202daff 00000080
	v_and_b16 v110.l, 0x80, v110.l                             // 000000002360: d762006e 0202dcff 00000080
	v_and_b16 v90.l, 0x80, v111.l                              // 00000000236c: d762005a 0202deff 00000080
	v_and_b16 v90.h, 0x80, v112.l op_sel:[0,0,1]               // 000000002378: d762405a 0202e0ff 00000080
	v_and_b16 v91.l, 0x80, v113.l                              // 000000002384: d762005b 0202e2ff 00000080
	v_and_b16 v91.h, 0x80, v114.l op_sel:[0,0,1]               // 000000002390: d762405b 0202e4ff 00000080
	v_or_b16 v92.l, v102.h, v94.l op_sel:[1,0,0]               // 00000000239c: d763085c 0202bd66
	v_or_b16 v92.h, v102.l, v95.l op_sel:[0,0,1]               // 0000000023a4: d763405c 0202bf66
	v_or_b16 v90.l, v90.l, v96.l                               // 0000000023ac: d763005a 0202c15a
	v_or_b16 v93.l, v111.h, v97.l op_sel:[1,0,0]               // 0000000023b4: d763085d 0202c36f
	v_or_b16 v93.h, v103.l, v98.l op_sel:[0,0,1]               // 0000000023bc: d763405d 0202c567
	v_or_b16 v94.l, v103.h, v99.l op_sel:[1,0,0]               // 0000000023c4: d763085e 0202c767
	v_or_b16 v90.h, v90.h, v100.l op_sel:[1,0,1]               // 0000000023cc: d763485a 0202c95a
	v_or_b16 v94.h, v112.h, v105.l op_sel:[1,0,1]              // 0000000023d4: d763485e 0202d370
	v_or_b16 v95.l, v109.l, v106.l                             // 0000000023dc: d763005f 0202d56d
	v_or_b16 v95.h, v109.h, v107.l op_sel:[1,0,1]              // 0000000023e4: d763485f 0202d76d
	v_or_b16 v91.l, v91.l, v115.l                              // 0000000023ec: d763005b 0202e75b
	v_or_b16 v96.l, v113.h, v116.l op_sel:[1,0,0]              // 0000000023f4: d7630860 0202e971
	v_or_b16 v96.h, v110.l, v117.l op_sel:[0,0,1]              // 0000000023fc: d7634060 0202eb6e
	v_or_b16 v97.l, v110.h, v118.l op_sel:[1,0,0]              // 000000002404: d7630861 0202ed6e
	v_or_b16 v91.h, v91.h, v119.l op_sel:[1,0,1]               // 00000000240c: d763485b 0202ef5b
	v_or_b16 v97.h, v114.h, v120.l op_sel:[1,0,1]              // 000000002414: d7634861 0202f172
	v_cndmask_b16 v92.l, v92.l, 0, s2                          // 00000000241c: d65d005c 0009015c
	v_cndmask_b16 v92.h, v92.h, 0, s2                          // 000000002424: d65d485c 0009015c
	v_cndmask_b16 v90.l, v90.l, 0, s2                          // 00000000242c: d65d005a 0009015a
	v_cndmask_b16 v93.l, v93.l, 0, s2                          // 000000002434: d65d005d 0009015d
	v_cndmask_b16 v93.h, v93.h, 0, s2                          // 00000000243c: d65d485d 0009015d
	v_cndmask_b16 v94.l, v94.l, 0, s2                          // 000000002444: d65d005e 0009015e
	v_cndmask_b16 v90.h, v90.h, 0, s2                          // 00000000244c: d65d485a 0009015a
	v_cndmask_b16 v94.h, v94.h, 0, s2                          // 000000002454: d65d485e 0009015e
	v_cndmask_b16 v95.l, v95.l, 0, s2                          // 00000000245c: d65d005f 0009015f
	v_cndmask_b16 v95.h, v95.h, 0, s2                          // 000000002464: d65d485f 0009015f
	v_cndmask_b16 v91.l, v91.l, 0, s2                          // 00000000246c: d65d005b 0009015b
	v_cndmask_b16 v97.h, v97.h, 0, s2                          // 000000002474: d65d4861 00090161
	v_cndmask_b16 v91.h, v91.h, 0, s2                          // 00000000247c: d65d485b 0009015b
	v_cndmask_b16 v97.l, v97.l, 0, s2                          // 000000002484: d65d0061 00090161
	v_cndmask_b16 v96.h, v96.h, 0, s2                          // 00000000248c: d65d4860 00090160
	v_cndmask_b16 v96.l, v96.l, 0, s2                          // 000000002494: d65d0060 00090160
	v_lshlrev_b16 v97.h, 8, v97.h op_sel:[0,1,1]               // 00000000249c: d7385061 0202c288
	v_and_b16 v91.h, 0xff, v91.h op_sel:[0,1,1]                // 0000000024a4: d762505b 0202b6ff 000000ff
	v_lshlrev_b16 v97.l, 8, v97.l                              // 0000000024b0: d7380061 0202c288
	v_and_b16 v96.h, 0xff, v96.h op_sel:[0,1,1]                // 0000000024b8: d7625060 0202c0ff 000000ff
	v_lshlrev_b16 v96.l, 8, v96.l                              // 0000000024c4: d7380060 0202c088
	v_and_b16 v91.l, 0xff, v91.l                               // 0000000024cc: d762005b 0202b6ff 000000ff
	v_lshlrev_b16 v95.h, 8, v95.h op_sel:[0,1,1]               // 0000000024d8: d738505f 0202be88
	v_and_b16 v95.l, 0xff, v95.l                               // 0000000024e0: d762005f 0202beff 000000ff
	v_lshlrev_b16 v94.h, 8, v94.h op_sel:[0,1,1]               // 0000000024ec: d738505e 0202bc88
	v_and_b16 v90.h, 0xff, v90.h op_sel:[0,1,1]                // 0000000024f4: d762505a 0202b4ff 000000ff
	v_lshlrev_b16 v94.l, 8, v94.l                              // 000000002500: d738005e 0202bc88
	v_and_b16 v98.l, 0xff, v93.h op_sel:[0,1,0]                // 000000002508: d7621062 0202baff 000000ff
	v_lshlrev_b16 v98.h, 8, v93.l op_sel:[0,0,1]               // 000000002514: d7384062 0202ba88
	v_and_b16 v90.l, 0xff, v90.l                               // 00000000251c: d762005a 0202b4ff 000000ff
	v_lshlrev_b16 v99.l, 8, v92.h op_sel:[0,1,0]               // 000000002528: d7381063 0202b888
	v_and_b16 v99.h, 0xff, v92.l op_sel:[0,0,1]                // 000000002530: d7624063 0202b8ff 000000ff
	v_or_b16 v93.h, v91.h, v97.h op_sel:[1,1,1]                // 00000000253c: d763585d 0202c35b
	v_or_b16 v93.l, v96.h, v97.l op_sel:[1,0,0]                // 000000002544: d763085d 0202c360
	v_or_b16 v92.h, v91.l, v96.l op_sel:[0,0,1]                // 00000000254c: d763405c 0202c15b
	v_or_b16 v92.l, v95.l, v95.h op_sel:[0,1,0]                // 000000002554: d763105c 0202bf5f
	v_or_b16 v91.h, v90.h, v94.h op_sel:[1,1,1]                // 00000000255c: d763585b 0202bd5a
	v_or_b16 v91.l, v98.l, v94.l                               // 000000002564: d763005b 0202bd62
	v_or_b16 v90.h, v90.l, v98.h op_sel:[0,1,1]                // 00000000256c: d763505a 0202c55a
	v_or_b16 v90.l, v99.h, v99.l op_sel:[1,0,0]                // 000000002574: d763085a 0202c763
	s_cmp_lg_u64 s[26:27], 0x400                               // 00000000257c: bf11ff1a 00000400
	ds_store_b128 v72, v[90:93] offset:20480                   // 000000002584: db7c5000 00005a48
	s_wait_dscnt 0x0                                           // 00000000258c: bfc60000
	s_barrier_signal -1                                        // 000000002590: be804ec1
	s_barrier_wait 0xffff                                      // 000000002594: bf94ffff
	ds_load_2addr_b64 v[90:93], v73 offset1:2                  // 000000002598: d9dc0200 5a000049
	ds_load_2addr_b64 v[94:97], v88 offset1:2                  // 0000000025a0: d9dc0200 5e000058
	ds_load_2addr_b64 v[98:101], v89 offset1:2                 // 0000000025a8: d9dc0200 62000059
	ds_load_2addr_b64 v[105:108], v76 offset1:2                // 0000000025b0: d9dc0200 6900004c
	ds_load_2addr_b64 v[109:112], v77 offset1:2                // 0000000025b8: d9dc0200 6d00004d
	ds_load_2addr_b64 v[113:116], v78 offset1:2                // 0000000025c0: d9dc0200 7100004e
	ds_load_2addr_b64 v[117:120], v73 offset0:4 offset1:6      // 0000000025c8: d9dc0604 75000049
	ds_load_2addr_b64 v[121:124], v88 offset0:4 offset1:6      // 0000000025d0: d9dc0604 79000058
	ds_load_2addr_b64 v[125:128], v89 offset0:4 offset1:6      // 0000000025d8: d9dc0604 7d000059
	ds_load_2addr_b64 v[131:134], v76 offset0:4 offset1:6      // 0000000025e0: d9dc0604 8300004c
	ds_load_2addr_b64 v[135:138], v77 offset0:4 offset1:6      // 0000000025e8: d9dc0604 8700004d
	ds_load_2addr_b64 v[139:142], v78 offset0:4 offset1:6      // 0000000025f0: d9dc0604 8b00004e
	s_wait_dscnt 0xa                                           // 0000000025f8: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[90:91], v[94:95], v[2:9]// 0000000025fc: cc464002 1c0abd5a
	s_wait_dscnt 0x9                                           // 000000002604: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[90:91], v[98:99], v[34:41]// 000000002608: cc464022 1c8ac55a
	s_wait_dscnt 0x8                                           // 000000002610: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[105:106], v[94:95], v[58:65]// 000000002614: cc46403a 1ceabd69
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[105:106], v[98:99], v[26:33]// 00000000261c: cc46401a 1c6ac569
	s_wait_dscnt 0x7                                           // 000000002624: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[109:110], v[94:95], v[50:57]// 000000002628: cc464032 1ccabd6d
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[109:110], v[98:99], v[18:25]// 000000002630: cc464012 1c4ac56d
	s_wait_dscnt 0x6                                           // 000000002638: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[113:114], v[94:95], v[42:49]// 00000000263c: cc46402a 1caabd71
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[113:114], v[98:99], v[10:17]// 000000002644: cc46400a 1c2ac571
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[92:93], v[96:97], v[2:9]// 00000000264c: cc464002 1c0ac15c
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[92:93], v[100:101], v[34:41]// 000000002654: cc464022 1c8ac95c
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[107:108], v[96:97], v[58:65]// 00000000265c: cc46403a 1ceac16b
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[107:108], v[100:101], v[26:33]// 000000002664: cc46401a 1c6ac96b
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[111:112], v[96:97], v[50:57]// 00000000266c: cc464032 1ccac16f
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[111:112], v[100:101], v[18:25]// 000000002674: cc464012 1c4ac96f
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[115:116], v[96:97], v[42:49]// 00000000267c: cc46402a 1caac173
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[115:116], v[100:101], v[10:17]// 000000002684: cc46400a 1c2ac973
	s_wait_dscnt 0x4                                           // 00000000268c: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[117:118], v[121:122], v[2:9]// 000000002690: cc464002 1c0af375
	s_wait_dscnt 0x3                                           // 000000002698: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[117:118], v[125:126], v[34:41]// 00000000269c: cc464022 1c8afb75
	s_wait_dscnt 0x2                                           // 0000000026a4: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[131:132], v[121:122], v[58:65]// 0000000026a8: cc46403a 1ceaf383
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[131:132], v[125:126], v[26:33]// 0000000026b0: cc46401a 1c6afb83
	s_wait_dscnt 0x1                                           // 0000000026b8: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[135:136], v[121:122], v[50:57]// 0000000026bc: cc464032 1ccaf387
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[135:136], v[125:126], v[18:25]// 0000000026c4: cc464012 1c4afb87
	s_wait_dscnt 0x0                                           // 0000000026cc: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[139:140], v[121:122], v[42:49]// 0000000026d0: cc46402a 1caaf38b
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[139:140], v[125:126], v[10:17]// 0000000026d8: cc46400a 1c2afb8b
	v_wmma_f32_16x16x16_fp8_fp8 v[2:9], v[119:120], v[123:124], v[2:9]// 0000000026e0: cc464002 1c0af777
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[119:120], v[127:128], v[34:41]// 0000000026e8: cc464022 1c8aff77
	v_wmma_f32_16x16x16_fp8_fp8 v[58:65], v[133:134], v[123:124], v[58:65]// 0000000026f0: cc46403a 1ceaf785
	v_wmma_f32_16x16x16_fp8_fp8 v[26:33], v[133:134], v[127:128], v[26:33]// 0000000026f8: cc46401a 1c6aff85
	v_wmma_f32_16x16x16_fp8_fp8 v[50:57], v[137:138], v[123:124], v[50:57]// 000000002700: cc464032 1ccaf789
	v_wmma_f32_16x16x16_fp8_fp8 v[18:25], v[137:138], v[127:128], v[18:25]// 000000002708: cc464012 1c4aff89
	v_wmma_f32_16x16x16_fp8_fp8 v[42:49], v[141:142], v[123:124], v[42:49]// 000000002710: cc46402a 1caaf78d
	v_wmma_f32_16x16x16_fp8_fp8 v[10:17], v[141:142], v[127:128], v[10:17]// 000000002718: cc46400a 1c2aff8d
	s_cbranch_scc1 64997                                       // 000000002720: bfa2fde5 <packed_folded_w4a8+0x3b8>
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000002724: f4002180 f8000058
	v_or_b32_e32 v78, s14, v74                                 // 00000000272c: 389c940e
	v_or_b32_e32 v76, s24, v70                                 // 000000002730: 38988c18
	v_mov_b32_e32 v1, s25                                      // 000000002734: 7e020219
	v_mov_b32_e32 v67, s15                                     // 000000002738: 7e86020f
	s_mov_b32 s5, 0                                            // 00000000273c: be850080
	v_or_b32_e32 v80, v75, v78                                 // 000000002740: 38a09d4b
	v_or_b32_e32 v0, v76, v71                                  // 000000002744: 38008f4c
	v_mov_b32_e32 v81, s15                                     // 000000002748: 7ea2020f
	v_mov_b32_e32 v79, s15                                     // 00000000274c: 7e9e020f
	v_mov_b32_e32 v77, s25                                     // 000000002750: 7e9a0219
	v_or_b32_e32 v66, 7, v80                                   // 000000002754: 3884a087
	v_cmp_gt_u64_e64 s3, s[18:19], v[0:1]                      // 000000002758: d45c0003 02020012
	s_delay_alu instid0(valu_dep_2)                            // 000000002760: bf870002
	v_cmp_gt_u64_e32 vcc_lo, s[16:17], v[66:67]                // 000000002764: 7cb88410
	s_wait_kmcnt 0x0                                           // 000000002768: bfc70000
	s_and_b32 s4, s6, 15                                       // 00000000276c: 8b048f06
	s_and_b32 s2, vcc_lo, s3                                   // 000000002770: 8b02036a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002774: bf88ff9e
	s_cmp_eq_u64 s[4:5], 0                                     // 000000002778: bf108004
	s_cselect_b32 s10, -1, 0                                   // 00000000277c: 980a80c1
	s_wait_alu depctr_sa_sdst(0)                               // 000000002780: bf88ff9e
	s_and_b32 s2, s10, s2                                      // 000000002784: 8b02020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002788: bf88ff9e
	s_xor_b32 s2, s2, -1                                       // 00000000278c: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002790: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002794: be842002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002798: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 00000000279c: 8d05047e
	s_cbranch_execz 138                                        // 0000000027a0: bfa5008a <packed_folded_w4a8+0xecc>
	v_cmp_gt_i64_e64 s2, s[16:17], v[80:81]                    // 0000000027a4: d4540002 0202a010
	v_or_b32_e32 v68, 1, v80                                   // 0000000027ac: 3888a081
	v_mov_b32_e32 v69, v81                                     // 0000000027b0: 7e8a0351
	v_or_b32_e32 v71, 2, v80                                   // 0000000027b4: 388ea082
	v_mov_b32_e32 v72, v81                                     // 0000000027b8: 7e900351
	v_or_b32_e32 v85, 3, v80                                   // 0000000027bc: 38aaa083
	s_wait_alu depctr_va_sdst(0)                               // 0000000027c0: bf88f19f
	v_cndmask_b32_e64 v70, 0, v81, s2                          // 0000000027c4: d5010046 000aa280
	v_cmp_gt_i64_e64 s4, s[16:17], v[68:69]                    // 0000000027cc: d4540004 02028810
	v_cndmask_b32_e64 v69, 0, v80, s2                          // 0000000027d4: d5010045 000aa080
	v_cmp_gt_i64_e64 s2, s[16:17], v[71:72]                    // 0000000027dc: d4540002 02028e10
	v_mov_b32_e32 v86, v81                                     // 0000000027e4: 7eac0351
	v_or_b32_e32 v87, 5, v80                                   // 0000000027e8: 38aea085
	v_mov_b32_e32 v88, v81                                     // 0000000027ec: 7eb00351
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f0: bf88f19f
	v_cndmask_b32_e64 v83, 0, v68, s4                          // 0000000027f4: d5010053 00128880
	v_lshlrev_b64_e32 v[68:69], 2, v[69:70]                    // 0000000027fc: 3e888a82
	v_cndmask_b32_e64 v70, 0, v71, s2                          // 000000002800: d5010046 000a8e80
	v_cndmask_b32_e64 v71, 0, v81, s2                          // 000000002808: d5010047 000aa280
	v_cndmask_b32_e64 v84, 0, v81, s4                          // 000000002810: d5010054 0012a280
	v_cmp_gt_i64_e64 s2, s[16:17], v[85:86]                    // 000000002818: d4540002 0202aa10
	v_add_co_u32 v68, s4, s6, v68                              // 000000002820: d7000444 02028806
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002828: bf870214
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]                    // 00000000282c: 3e8c8c82
	v_lshlrev_b64_e32 v[72:73], 2, v[83:84]                    // 000000002830: 3e90a682
	s_wait_alu depctr_va_sdst(0)                               // 000000002834: bf88f19f
	s_delay_alu instid0(valu_dep_4)                            // 000000002838: bf870004
	v_cndmask_b32_e64 v83, 0, v85, s2                          // 00000000283c: d5010053 000aaa80
	v_or_b32_e32 v85, 4, v80                                   // 000000002844: 38aaa084
	v_cndmask_b32_e64 v84, 0, v81, s2                          // 000000002848: d5010054 000aa280
	v_add_co_ci_u32_e64 v69, null, s7, v69, s4                 // 000000002850: d5207c45 00128a07
	v_add_co_u32 v89, s2, s6, v70                              // 000000002858: d7000259 02028c06
	v_add_co_u32 v72, s4, s6, v72                              // 000000002860: d7000448 02029006
	s_wait_alu depctr_va_sdst(0)                               // 000000002868: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s7, v71, s2                 // 00000000286c: d5207c5a 000a8e07
	v_cmp_gt_i64_e64 s2, s[16:17], v[85:86]                    // 000000002874: d4540002 0202aa10
	v_add_co_ci_u32_e64 v73, null, s7, v73, s4                 // 00000000287c: d5207c49 00129207
	v_lshlrev_b64_e32 v[70:71], 2, v[83:84]                    // 000000002884: 3e8ca682
	v_cmp_gt_i64_e64 s4, s[16:17], v[87:88]                    // 000000002888: d4540004 0202ae10
	v_or_b32_e32 v83, 6, v80                                   // 000000002890: 38a6a086
	v_mov_b32_e32 v84, v81                                     // 000000002894: 7ea80351
	s_wait_alu depctr_va_sdst(0)                               // 000000002898: bf88f19f
	v_cndmask_b32_e64 v85, 0, v85, s2                          // 00000000289c: d5010055 000aaa80
	v_cndmask_b32_e64 v86, 0, v81, s2                          // 0000000028a4: d5010056 000aa280
	v_cndmask_b32_e64 v87, 0, v87, s4                          // 0000000028ac: d5010057 0012ae80
	v_cmp_gt_i64_e64 s2, s[16:17], v[83:84]                    // 0000000028b4: d4540002 0202a610
	v_cndmask_b32_e64 v88, 0, v81, s4                          // 0000000028bc: d5010058 0012a280
	v_add_co_u32 v91, s4, s6, v70                              // 0000000028c4: d700045b 02028c06
	s_wait_alu depctr_va_sdst(0)                               // 0000000028cc: bf88f19f
	v_add_co_ci_u32_e64 v92, null, s7, v71, s4                 // 0000000028d0: d5207c5c 00128e07
	v_lshlrev_b64_e32 v[70:71], 2, v[85:86]                    // 0000000028d8: 3e8caa82
	v_lshlrev_b64_e32 v[84:85], 2, v[87:88]                    // 0000000028dc: 3ea8ae82
	v_cndmask_b32_e64 v86, 0, v83, s2                          // 0000000028e0: d5010056 000aa680
	v_cndmask_b32_e64 v87, 0, v81, s2                          // 0000000028e8: d5010057 000aa280
	v_cmp_gt_i64_e64 s2, s[16:17], v[66:67]                    // 0000000028f0: d4540002 02028410
	v_add_co_u32 v93, s4, s6, v70                              // 0000000028f8: d700045d 02028c06
	s_wait_alu depctr_va_sdst(0)                               // 000000002900: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s7, v71, s4                 // 000000002904: d5207c5e 00128e07
	s_delay_alu instid0(valu_dep_3)                            // 00000000290c: bf870003
	v_cndmask_b32_e64 v67, 0, v67, s2                          // 000000002910: d5010043 000a8680
	v_cndmask_b32_e64 v66, 0, v66, s2                          // 000000002918: d5010042 000a8480
	v_lshlrev_b64_e32 v[70:71], 2, v[86:87]                    // 000000002920: 3e8cac82
	v_add_co_u32 v83, s2, s6, v84                              // 000000002924: d7000253 0202a806
	s_wait_alu depctr_va_sdst(0)                               // 00000000292c: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s7, v85, s2                 // 000000002930: d5207c54 000aaa07
	v_lshlrev_b64_e32 v[66:67], 2, v[66:67]                    // 000000002938: 3e848482
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 00000000293c: bf8701b4
	v_add_co_u32 v85, s2, s6, v70                              // 000000002940: d7000255 02028c06
	s_wait_alu depctr_va_sdst(0)                               // 000000002948: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s7, v71, s2                 // 00000000294c: d5207c56 000a8e07
	v_add_co_u32 v87, s2, s6, v66                              // 000000002954: d7000257 02028406
	s_wait_alu depctr_va_sdst(0)                               // 00000000295c: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s7, v67, s2                 // 000000002960: d5207c58 000a8607
	s_clause 0x7                                               // 000000002968: bf850007
	global_load_b32 v70, v[68:69], off                         // 00000000296c: ee05007c 00000046 00000044
	global_load_b32 v71, v[72:73], off                         // 000000002978: ee05007c 00000047 00000048
	global_load_b32 v72, v[89:90], off                         // 000000002984: ee05007c 00000048 00000059
	global_load_b32 v73, v[91:92], off                         // 000000002990: ee05007c 00000049 0000005b
	global_load_b32 v66, v[93:94], off                         // 00000000299c: ee05007c 00000042 0000005d
	global_load_b32 v67, v[83:84], off                         // 0000000029a8: ee05007c 00000043 00000053
	global_load_b32 v68, v[85:86], off                         // 0000000029b4: ee05007c 00000044 00000055
	global_load_b32 v69, v[87:88], off                         // 0000000029c0: ee05007c 00000045 00000057
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029cc: bf88ff9e
	s_or_saveexec_b32 s4, s5                                   // 0000000029d0: be842205
	v_lshlrev_b64_e32 v[118:119], 2, v[80:81]                  // 0000000029d4: 3eeca082
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029d8: bf88ff9e
	s_xor_b32 exec_lo, exec_lo, s4                             // 0000000029dc: 8d7e047e
	s_cbranch_execz 15                                         // 0000000029e0: bfa5000f <packed_folded_w4a8+0xf20>
	s_wait_loadcnt 0x3                                         // 0000000029e4: bfc00003
	s_delay_alu instid0(valu_dep_1)                            // 0000000029e8: bf870001
	v_add_co_u32 v66, s2, s6, v118                             // 0000000029ec: d7000242 0202ec06
	s_wait_loadcnt 0x2                                         // 0000000029f4: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 0000000029f8: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s7, v119, s2                // 0000000029fc: d5207c43 000aee07
	global_load_b128 v[70:73], v[66:67], off                   // 000000002a04: ee05c07c 00000046 00000042
	s_wait_loadcnt 0x1                                         // 000000002a10: bfc00001
	global_load_b128 v[66:69], v[66:67], off offset:16         // 000000002a14: ee05c07c 00000042 00001042
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002a20: 8c7e047e
	v_cmp_gt_i64_e64 s2, s[18:19], v[0:1]                      // 000000002a24: d4540002 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000002a2c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a30: bf870121
	v_cndmask_b32_e64 v84, 0, v0, s2                           // 000000002a34: d5010054 000a0080
	v_cndmask_b32_e64 v83, 0, v1, s2                           // 000000002a3c: d5010053 000a0280
	v_add_co_u32 v154, s2, s20, v84                            // 000000002a44: d700029a 0202a814
	s_wait_alu depctr_va_sdst(0)                               // 000000002a4c: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002a50: bf8700c2
	v_add_co_ci_u32_e64 v155, null, s21, v83, s2               // 000000002a54: d5207c9b 000aa615
	global_load_u8 v83, v[154:155], off                        // 000000002a5c: ee04007c 00000053 0000009a
	s_wait_loadcnt 0x0                                         // 000000002a68: bfc00000
	v_lshlrev_b32_e32 v84, 23, v83                             // 000000002a6c: 30a8a697
	v_mul_f32_e32 v83, v70, v84                                // 000000002a70: 10a6a946
	s_delay_alu instid0(valu_dep_1)                            // 000000002a74: bf870001
	v_cmp_class_f32_e64 s2, v83, 0x198                         // 000000002a78: d47e0002 0201ff53 00000198
	v_mul_f32_e32 v83, v2, v83                                 // 000000002a84: 10a6a702
	s_xor_b32 s2, s2, -1                                       // 000000002a88: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a8c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002a90: be842002
	s_cbranch_execnz 3328                                      // 000000002a94: bfa60d00 <packed_folded_w4a8+0x4398>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002a9c: 8c7e047e
	v_mul_f32_e32 v2, v71, v84                                 // 000000002aa0: 1004a947
	s_delay_alu instid0(valu_dep_1)                            // 000000002aa4: bf870001
	v_cmp_class_f32_e64 s2, v2, 0x198                          // 000000002aa8: d47e0002 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 000000002ab4: 10040503
	s_xor_b32 s2, s2, -1                                       // 000000002ab8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002abc: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002ac0: be842002
	s_cbranch_execnz 3334                                      // 000000002ac4: bfa60d06 <packed_folded_w4a8+0x43e0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ac8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002acc: 8c7e047e
	v_mul_f32_e32 v3, v72, v84                                 // 000000002ad0: 1006a948
	s_delay_alu instid0(valu_dep_1)                            // 000000002ad4: bf870001
	v_cmp_class_f32_e64 s2, v3, 0x198                          // 000000002ad8: d47e0002 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 000000002ae4: 10060704
	s_xor_b32 s2, s2, -1                                       // 000000002ae8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002aec: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002af0: be842002
	s_cbranch_execnz 3340                                      // 000000002af4: bfa60d0c <packed_folded_w4a8+0x4428>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002af8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002afc: 8c7e047e
	v_mul_f32_e32 v4, v73, v84                                 // 000000002b00: 1008a949
	s_delay_alu instid0(valu_dep_1)                            // 000000002b04: bf870001
	v_cmp_class_f32_e64 s2, v4, 0x198                          // 000000002b08: d47e0002 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 000000002b14: 10080905
	s_xor_b32 s2, s2, -1                                       // 000000002b18: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b1c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b20: be842002
	s_cbranch_execnz 3346                                      // 000000002b24: bfa60d12 <packed_folded_w4a8+0x4470>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b2c: 8c7e047e
	v_mul_f32_e32 v5, v66, v84                                 // 000000002b30: 100aa942
	s_delay_alu instid0(valu_dep_1)                            // 000000002b34: bf870001
	v_cmp_class_f32_e64 s2, v5, 0x198                          // 000000002b38: d47e0002 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 000000002b44: 100a0b06
	s_xor_b32 s2, s2, -1                                       // 000000002b48: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b4c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b50: be842002
	s_cbranch_execnz 3352                                      // 000000002b54: bfa60d18 <packed_folded_w4a8+0x44b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b5c: 8c7e047e
	v_mul_f32_e32 v6, v67, v84                                 // 000000002b60: 100ca943
	s_delay_alu instid0(valu_dep_1)                            // 000000002b64: bf870001
	v_cmp_class_f32_e64 s2, v6, 0x198                          // 000000002b68: d47e0002 0201ff06 00000198
	v_mul_f32_e32 v66, v7, v6                                  // 000000002b74: 10840d07
	s_xor_b32 s2, s2, -1                                       // 000000002b78: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b7c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b80: be842002
	s_cbranch_execnz 3358                                      // 000000002b84: bfa60d1e <packed_folded_w4a8+0x4500>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b8c: 8c7e047e
	v_mul_f32_e32 v6, v68, v84                                 // 000000002b90: 100ca944
	s_delay_alu instid0(valu_dep_1)                            // 000000002b94: bf870001
	v_cmp_class_f32_e64 s2, v6, 0x198                          // 000000002b98: d47e0002 0201ff06 00000198
	v_mul_f32_e32 v7, v8, v6                                   // 000000002ba4: 100e0d08
	s_xor_b32 s2, s2, -1                                       // 000000002ba8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bac: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002bb0: be842002
	s_cbranch_execnz 3364                                      // 000000002bb4: bfa60d24 <packed_folded_w4a8+0x4548>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002bbc: 8c7e047e
	v_mul_f32_e32 v6, v69, v84                                 // 000000002bc0: 100ca945
	s_delay_alu instid0(valu_dep_1)                            // 000000002bc4: bf870001
	v_cmp_class_f32_e64 s2, v6, 0x198                          // 000000002bc8: d47e0002 0201ff06 00000198
	v_mul_f32_e32 v6, v9, v6                                   // 000000002bd4: 100c0d09
	s_xor_b32 s2, s2, -1                                       // 000000002bd8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bdc: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002be0: be842002
	s_cbranch_execnz 3370                                      // 000000002be4: bfa60d2a <packed_folded_w4a8+0x4590>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002be8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002bec: 8c7e047e
	s_load_b64 s[8:9], s[0:1], 0xa8                            // 000000002bf0: f4002200 f80000a8
	v_mul_lo_u32 v8, s19, v80                                  // 000000002bf8: d72c0008 0202a013
	v_mul_lo_u32 v69, s18, v81                                 // 000000002c00: d72c0045 0202a212
	v_mad_co_u64_u32 v[67:68], null, s18, v80, 0               // 000000002c08: d6fe7c43 0202a012
	v_lshlrev_b64_e32 v[160:161], 1, v[0:1]                    // 000000002c10: 3f400081
	v_or_b32_e32 v70, 0x400000, v83                            // 000000002c14: 388ca6ff 00400000
	v_bfe_u32 v71, v2, 16, 1                                   // 000000002c1c: d6100047 02052102
	v_or_b32_e32 v0, 0x400000, v2                              // 000000002c24: 380004ff 00400000
	v_cmp_u_f32_e64 s1, v6, v6                                 // 000000002c2c: d4180001 02020d06
	v_or_b32_e32 v164, 1, v75                                  // 000000002c34: 39489681
	v_or_b32_e32 v163, 2, v75                                  // 000000002c38: 39469682
	v_add3_u32 v68, v68, v69, v8                               // 000000002c3c: d6550044 04228b44
	v_bfe_u32 v69, v83, 16, 1                                  // 000000002c44: d6100045 02052153
	v_or_b32_e32 v8, s14, v82                                  // 000000002c4c: 3810a40e
	v_or_b32_e32 v167, 3, v75                                  // 000000002c50: 394e9683
	v_or_b32_e32 v165, 4, v75                                  // 000000002c54: 394a9684
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 000000002c58: 3e868681
	v_add3_u32 v1, v69, v83, 0x7fff                            // 000000002c5c: d6550001 03fea745 00007fff
	v_add3_u32 v69, v71, v2, 0x7fff                            // 000000002c68: d6550045 03fe0547 00007fff
	v_or_b32_e32 v86, v8, v75                                  // 000000002c74: 38ac9708
	v_or_b32_e32 v168, 5, v75                                  // 000000002c78: 39509685
	v_or_b32_e32 v166, 6, v75                                  // 000000002c7c: 394c9686
	s_wait_kmcnt 0x0                                           // 000000002c80: bfc70000
	v_add_co_u32 v67, s0, s8, v67                              // 000000002c84: d7000043 02028608
	s_delay_alu instid0(valu_dep_1)                            // 000000002c8c: bf870001
	v_add_co_ci_u32_e64 v68, null, s9, v68, s0                 // 000000002c90: d5207c44 00028809
	v_cmp_u_f32_e64 s0, v83, v83                               // 000000002c98: d4180000 0202a753
	v_or_b32_e32 v169, 7, v75                                  // 000000002ca0: 39529687
	v_mov_b32_e32 v9, s15                                      // 000000002ca4: 7e12020f
	v_mov_b32_e32 v87, s15                                     // 000000002ca8: 7eae020f
	v_or_b32_e32 v94, v164, v8                                 // 000000002cac: 38bc11a4
	s_wait_alu depctr_va_sdst(0)                               // 000000002cb0: bf88f19f
	v_cndmask_b32_e64 v1, v1, v70, s0                          // 000000002cb4: d5010001 00028d01
	v_add_co_u32 v72, s0, v67, v160                            // 000000002cbc: d7000048 02034143
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc4: bf88f19f
	v_add_co_ci_u32_e64 v73, null, v68, v161, s0               // 000000002cc8: d5207c49 00034344
	v_cmp_u_f32_e64 s0, v2, v2                                 // 000000002cd0: d4180000 02020502
	v_or_b32_e32 v88, v163, v8                                 // 000000002cd8: 38b011a3
	v_or_b32_e32 v82, v167, v8                                 // 000000002cdc: 38a411a7
	global_store_d16_hi_b16 v[72:73], v1, off                  // 000000002ce0: ee09407c 00800000 00000048
	v_or_b32_e32 v70, v165, v8                                 // 000000002cec: 388c11a5
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf0: bf88f19f
	v_cndmask_b32_e64 v0, v69, v0, s0                          // 000000002cf4: d5010000 00020145
	v_add_co_u32 v2, s0, v67, s22                              // 000000002cfc: d7000002 02002d43
	s_wait_alu depctr_va_sdst(0)                               // 000000002d04: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s23, v68, s0                // 000000002d08: d5207c43 00028817
	v_bfe_u32 v68, v3, 16, 1                                   // 000000002d10: d6100044 02052103
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d18: bf8701a3
	v_add_co_u32 v84, s0, v2, v160                             // 000000002d1c: d7000054 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000002d24: bf88f19f
	v_add_co_ci_u32_e64 v85, null, v67, v161, s0               // 000000002d28: d5207c55 00034343
	s_delay_alu instid0(valu_dep_3)                            // 000000002d30: bf870003
	v_add3_u32 v1, v68, v3, 0x7fff                             // 000000002d34: d6550001 03fe0744 00007fff
	v_or_b32_e32 v68, 0x400000, v3                             // 000000002d40: 388806ff 00400000
	v_cmp_u_f32_e64 s0, v3, v3                                 // 000000002d48: d4180000 02020703
	global_store_d16_hi_b16 v[84:85], v0, off                  // 000000002d50: ee09407c 00000000 00000054
	s_wait_alu depctr_va_sdst(0)                               // 000000002d5c: bf88f19f
	v_cndmask_b32_e64 v0, v1, v68, s0                          // 000000002d60: d5010000 00028901
	v_add_co_u32 v2, s0, v2, s22                               // 000000002d68: d7000002 02002d02
	v_bfe_u32 v1, v4, 16, 1                                    // 000000002d70: d6100001 02052104
	s_wait_alu depctr_va_sdst(0)                               // 000000002d78: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v67, s0                 // 000000002d7c: d5207c03 00028617
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002d84: bf870193
	v_add_co_u32 v90, s0, v2, v160                             // 000000002d88: d700005a 02034102
	v_add3_u32 v1, v1, v4, 0x7fff                              // 000000002d90: d6550001 03fe0901 00007fff
	v_or_b32_e32 v67, 0x400000, v4                             // 000000002d9c: 388608ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002da4: bf88f19f
	v_add_co_ci_u32_e64 v91, null, v3, v161, s0                // 000000002da8: d5207c5b 00034303
	v_cmp_u_f32_e64 s0, v4, v4                                 // 000000002db0: d4180000 02020904
	v_bfe_u32 v4, v5, 16, 1                                    // 000000002db8: d6100004 02052105
	v_or_b32_e32 v68, v168, v8                                 // 000000002dc0: 388811a8
	global_store_d16_hi_b16 v[90:91], v0, off                  // 000000002dc4: ee09407c 00000000 0000005a
	s_wait_alu depctr_va_sdst(0)                               // 000000002dd0: bf88f19f
	v_cndmask_b32_e64 v1, v1, v67, s0                          // 000000002dd4: d5010001 00028701
	v_add_co_u32 v2, s0, v2, s22                               // 000000002ddc: d7000002 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000002de4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s0                  // 000000002de8: d5207c03 00020617
	v_add3_u32 v0, v4, v5, 0x7fff                              // 000000002df0: d6550000 03fe0b04 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002dfc: bf8701a3
	v_add_co_u32 v92, s0, v2, v160                             // 000000002e00: d700005c 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000002e08: bf88f19f
	v_add_co_ci_u32_e64 v93, null, v3, v161, s0                // 000000002e0c: d5207c5d 00034303
	v_or_b32_e32 v4, 0x400000, v5                              // 000000002e14: 38080aff 00400000
	v_cmp_u_f32_e64 s0, v5, v5                                 // 000000002e1c: d4180000 02020b05
	global_store_d16_hi_b16 v[92:93], v1, off                  // 000000002e24: ee09407c 00800000 0000005c
	v_bfe_u32 v1, v66, 16, 1                                   // 000000002e30: d6100001 02052142
	s_wait_alu depctr_va_sdst(0)                               // 000000002e38: bf88f19f
	v_cndmask_b32_e64 v0, v0, v4, s0                           // 000000002e3c: d5010000 00020900
	v_add_co_u32 v2, s0, v2, s22                               // 000000002e44: d7000002 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000002e4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s0                  // 000000002e50: d5207c03 00020617
	v_add3_u32 v1, v1, v66, 0x7fff                             // 000000002e58: d6550001 03fe8501 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e64: bf870003
	v_add_co_u32 v98, s0, v2, v160                             // 000000002e68: d7000062 02034102
	v_or_b32_e32 v4, 0x400000, v66                             // 000000002e70: 380884ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e78: bf88f19f
	v_add_co_ci_u32_e64 v99, null, v3, v161, s0                // 000000002e7c: d5207c63 00034303
	v_cmp_u_f32_e64 s0, v66, v66                               // 000000002e84: d4180000 02028542
	v_or_b32_e32 v66, v166, v8                                 // 000000002e8c: 388411a6
	v_or_b32_e32 v8, v169, v8                                  // 000000002e90: 381011a9
	global_store_d16_hi_b16 v[98:99], v0, off                  // 000000002e94: ee09407c 00000000 00000062
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea0: bf88f19f
	v_cndmask_b32_e64 v1, v1, v4, s0                           // 000000002ea4: d5010001 00020901
	v_add_co_u32 v2, s0, v2, s22                               // 000000002eac: d7000002 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s0                  // 000000002eb8: d5207c03 00020617
	v_bfe_u32 v4, v7, 16, 1                                    // 000000002ec0: d6100004 02052107
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002ec8: bf8701a3
	v_add_co_u32 v96, s0, v2, v160                             // 000000002ecc: d7000060 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed4: bf88f19f
	v_add_co_ci_u32_e64 v97, null, v3, v161, s0                // 000000002ed8: d5207c61 00034303
	s_delay_alu instid0(valu_dep_3)                            // 000000002ee0: bf870003
	v_add3_u32 v0, v4, v7, 0x7fff                              // 000000002ee4: d6550000 03fe0f04 00007fff
	v_or_b32_e32 v4, 0x400000, v7                              // 000000002ef0: 38080eff 00400000
	v_cmp_u_f32_e64 s0, v7, v7                                 // 000000002ef8: d4180000 02020f07
	global_store_d16_hi_b16 v[96:97], v1, off                  // 000000002f00: ee09407c 00800000 00000060
	v_mov_b32_e32 v1, s15                                      // 000000002f0c: 7e02020f
	v_or_b32_e32 v7, 0x400000, v6                              // 000000002f10: 380e0cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f18: bf88f19f
	v_cndmask_b32_e64 v4, v0, v4, s0                           // 000000002f1c: d5010004 00020900
	v_bfe_u32 v0, v6, 16, 1                                    // 000000002f24: d6100000 02052106
	v_add_co_u32 v2, s0, v2, s22                               // 000000002f2c: d7000002 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000002f34: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s0                  // 000000002f38: d5207c03 00020617
	s_delay_alu instid0(valu_dep_3)                            // 000000002f40: bf870003
	v_add3_u32 v5, v0, v6, 0x7fff                              // 000000002f44: d6550005 03fe0d00 00007fff
	v_or_b32_e32 v0, 7, v86                                    // 000000002f50: 3800ac87
	v_add_co_u32 v100, s0, v2, v160                            // 000000002f54: d7000064 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000002f5c: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v3, v161, s0               // 000000002f60: d5207c65 00034303
	v_add_co_u32 v2, s0, v2, s22                               // 000000002f68: d7000002 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000002f70: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s0                  // 000000002f74: d5207c03 00020617
	v_cmp_gt_u64_e64 s0, s[16:17], v[0:1]                      // 000000002f7c: d45c0000 02020010
	v_cndmask_b32_e64 v0, v5, v7, s1                           // 000000002f84: d5010000 00060f05
	v_add_co_u32 v102, s1, v2, v160                            // 000000002f8c: d7000166 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000002f94: bf88f19f
	v_add_co_ci_u32_e64 v103, null, v3, v161, s1               // 000000002f98: d5207c67 00074303
	s_and_b32 s1, s0, s3                                       // 000000002fa0: 8b010300
	global_store_d16_hi_b16 v[100:101], v4, off                // 000000002fa4: ee09407c 02000000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb0: bf88ff9e
	s_and_b32 s1, s10, s1                                      // 000000002fb4: 8b01010a
	global_store_d16_hi_b16 v[102:103], v0, off                // 000000002fb8: ee09407c 00000000 00000066
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fc4: bf88ff9e
	s_xor_b32 s1, s1, -1                                       // 000000002fc8: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fcc: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000002fd0: be822001
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fd4: bf88ff9e
	s_xor_b32 s4, exec_lo, s2                                  // 000000002fd8: 8d04027e
	s_cbranch_execz 133                                        // 000000002fdc: bfa50085 <packed_folded_w4a8+0x16f4>
	v_mov_b32_e32 v95, v9                                      // 000000002fe0: 7ebe0309
	v_cmp_gt_i64_e64 s1, s[16:17], v[86:87]                    // 000000002fe4: d4540001 0202ac10
	v_mov_b32_e32 v89, v9                                      // 000000002fec: 7eb20309
	v_mov_b32_e32 v83, v9                                      // 000000002ff0: 7ea60309
	v_mov_b32_e32 v71, v9                                      // 000000002ff4: 7e8e0309
	v_cmp_gt_i64_e64 s2, s[16:17], v[94:95]                    // 000000002ff8: d4540002 0202bc10
	v_mov_b32_e32 v69, v9                                      // 000000003000: 7e8a0309
	s_wait_alu depctr_va_sdst(0)                               // 000000003004: bf88f19f
	v_cndmask_b32_e64 v1, 0, v87, s1                           // 000000003008: d5010001 0006ae80
	v_cndmask_b32_e64 v0, 0, v86, s1                           // 000000003010: d5010000 0006ac80
	v_cmp_gt_i64_e64 s1, s[16:17], v[88:89]                    // 000000003018: d4540001 0202b010
	v_mov_b32_e32 v67, v9                                      // 000000003020: 7e860309
	v_cndmask_b32_e64 v3, 0, v95, s2                           // 000000003024: d5010003 000abe80
	v_cndmask_b32_e64 v2, 0, v94, s2                           // 00000000302c: d5010002 000abc80
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000003034: 3e000082
	v_cmp_gt_i64_e64 s2, s[16:17], v[82:83]                    // 000000003038: d4540002 0202a410
	s_wait_alu depctr_va_sdst(0)                               // 000000003040: bf88f19f
	v_cndmask_b32_e64 v5, 0, v89, s1                           // 000000003044: d5010005 0006b280
	v_cndmask_b32_e64 v4, 0, v88, s1                           // 00000000304c: d5010004 0006b080
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000003054: 3e040482
	v_add_co_u32 v0, s1, s6, v0                                // 000000003058: d7000100 02020006
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000003060: bf870233
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000003064: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 000000003068: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s1                   // 00000000306c: d5207c01 00060207
	v_add_co_u32 v2, s1, s6, v2                                // 000000003074: d7000102 02020406
	v_cndmask_b32_e64 v7, 0, v83, s2                           // 00000000307c: d5010007 000aa680
	v_cndmask_b32_e64 v6, 0, v82, s2                           // 000000003084: d5010006 000aa480
	s_wait_alu depctr_va_sdst(0)                               // 00000000308c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s1                   // 000000003090: d5207c03 00060607
	v_cmp_gt_i64_e64 s1, s[16:17], v[70:71]                    // 000000003098: d4540001 02028c10
	v_add_co_u32 v105, s2, s6, v4                              // 0000000030a0: d7000269 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000030a8: bf88f19f
	v_add_co_ci_u32_e64 v106, null, s7, v5, s2                 // 0000000030ac: d5207c6a 000a0a07
	v_cmp_gt_i64_e64 s2, s[16:17], v[68:69]                    // 0000000030b4: d4540002 02028810
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000030bc: 3e080c82
	v_cndmask_b32_e64 v7, 0, v71, s1                           // 0000000030c0: d5010007 00068e80
	v_cndmask_b32_e64 v6, 0, v70, s1                           // 0000000030c8: d5010006 00068c80
	v_cmp_gt_i64_e64 s1, s[16:17], v[66:67]                    // 0000000030d0: d4540001 02028410
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d8: bf88f19f
	v_cndmask_b32_e64 v108, 0, v69, s2                         // 0000000030dc: d501006c 000a8a80
	v_cndmask_b32_e64 v107, 0, v68, s2                         // 0000000030e4: d501006b 000a8880
	v_add_co_u32 v109, s2, s6, v4                              // 0000000030ec: d700026d 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f4: bf88f19f
	v_add_co_ci_u32_e64 v110, null, s7, v5, s2                 // 0000000030f8: d5207c6e 000a0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000003100: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[107:108]                    // 000000003104: 3e0cd682
	v_cndmask_b32_e64 v108, 0, v67, s1                         // 000000003108: d501006c 00068680
	v_cndmask_b32_e64 v107, 0, v66, s1                         // 000000003110: d501006b 00068480
	v_cmp_gt_i64_e64 s1, s[16:17], v[8:9]                      // 000000003118: d4540001 02021010
	v_add_co_u32 v111, s2, s6, v4                              // 000000003120: d700026f 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000003128: bf88f19f
	v_add_co_ci_u32_e64 v112, null, s7, v5, s2                 // 00000000312c: d5207c70 000a0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[107:108]                    // 000000003134: 3e08d682
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000003138: bf8701d4
	v_cndmask_b32_e64 v108, 0, v9, s1                          // 00000000313c: d501006c 00061280
	v_cndmask_b32_e64 v107, 0, v8, s1                          // 000000003144: d501006b 00061080
	v_add_co_u32 v113, s1, s6, v6                              // 00000000314c: d7000171 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000003154: bf88f19f
	v_add_co_ci_u32_e64 v114, null, s7, v7, s1                 // 000000003158: d5207c72 00060e07
	v_lshlrev_b64_e32 v[6:7], 2, v[107:108]                    // 000000003160: 3e0cd682
	v_add_co_u32 v107, s1, s6, v4                              // 000000003164: d700016b 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000316c: bf88f19f
	v_add_co_ci_u32_e64 v108, null, s7, v5, s1                 // 000000003170: d5207c6c 00060a07
	s_delay_alu instid0(valu_dep_3)                            // 000000003178: bf870003
	v_add_co_u32 v115, s1, s6, v6                              // 00000000317c: d7000173 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	v_add_co_ci_u32_e64 v116, null, s7, v7, s1                 // 000000003188: d5207c74 00060e07
	s_clause 0x7                                               // 000000003190: bf850007
	global_load_b32 v4, v[0:1], off                            // 000000003194: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 0000000031a0: ee05007c 00000005 00000002
	global_load_b32 v6, v[105:106], off                        // 0000000031ac: ee05007c 00000006 00000069
	global_load_b32 v7, v[109:110], off                        // 0000000031b8: ee05007c 00000007 0000006d
	global_load_b32 v0, v[111:112], off                        // 0000000031c4: ee05007c 00000000 0000006f
	global_load_b32 v1, v[113:114], off                        // 0000000031d0: ee05007c 00000001 00000071
	global_load_b32 v2, v[107:108], off                        // 0000000031dc: ee05007c 00000002 0000006b
	global_load_b32 v3, v[115:116], off                        // 0000000031e8: ee05007c 00000003 00000073
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031f4: bf88ff9e
	s_and_not1_saveexec_b32 s2, s4                             // 0000000031f8: be823004
	s_cbranch_execz 28                                         // 0000000031fc: bfa5001c <packed_folded_w4a8+0x1770>
	s_wait_loadcnt 0x3                                         // 000000003200: bfc00003
	v_add_co_u32 v0, s1, v74, s14                              // 000000003204: d7000100 02001d4a
	s_wait_loadcnt 0x2                                         // 00000000320c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000003210: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s1                   // 000000003214: d5207c01 00041e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000321c: bf870122
	v_add_co_u32 v0, s1, v0, v75                               // 000000003220: d7000100 02029700
	s_wait_alu depctr_va_sdst(0)                               // 000000003228: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, v1, s1                    // 00000000322c: d5207c01 00060280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003234: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000003238: 3e000082
	v_add_co_u32 v0, s1, s6, v0                                // 00000000323c: d7000100 02020006
	s_wait_alu depctr_va_sdst(0)                               // 000000003244: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003248: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, s1                   // 00000000324c: d5207c01 00060207
	global_load_b128 v[4:7], v[0:1], off offset:64             // 000000003254: ee05c07c 00000004 00004000
	s_wait_loadcnt 0x1                                         // 000000003260: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:80             // 000000003264: ee05c07c 00000000 00005000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003270: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003274: 8c7e027e
	global_load_u8 v67, v[154:155], off                        // 000000003278: ee04007c 00000043 0000009a
	s_wait_loadcnt 0x0                                         // 000000003284: bfc00000
	v_lshlrev_b32_e32 v69, 23, v67                             // 000000003288: 308a8697
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000328c: bf870091
	v_mul_f32_e32 v67, v4, v69                                 // 000000003290: 10868b04
	v_cmp_class_f32_e64 s1, v67, 0x198                         // 000000003294: d47e0001 0201ff43 00000198
	v_mul_f32_e32 v67, v58, v67                                // 0000000032a0: 1086873a
	s_xor_b32 s1, s1, -1                                       // 0000000032a4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000032ac: be822001
	s_cbranch_execnz 2953                                      // 0000000032b0: bfa60b89 <packed_folded_w4a8+0x45d8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000032b8: 8c7e027e
	v_mul_f32_e32 v4, v5, v69                                  // 0000000032bc: 10088b05
	s_delay_alu instid0(valu_dep_1)                            // 0000000032c0: bf870001
	v_cmp_class_f32_e64 s1, v4, 0x198                          // 0000000032c4: d47e0001 0201ff04 00000198
	v_mul_f32_e32 v4, v59, v4                                  // 0000000032d0: 1008093b
	s_xor_b32 s1, s1, -1                                       // 0000000032d4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032d8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000032dc: be822001
	s_cbranch_execnz 2959                                      // 0000000032e0: bfa60b8f <packed_folded_w4a8+0x4620>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000032e8: 8c7e027e
	v_mul_f32_e32 v5, v6, v69                                  // 0000000032ec: 100a8b06
	s_delay_alu instid0(valu_dep_1)                            // 0000000032f0: bf870001
	v_cmp_class_f32_e64 s1, v5, 0x198                          // 0000000032f4: d47e0001 0201ff05 00000198
	v_mul_f32_e32 v5, v60, v5                                  // 000000003300: 100a0b3c
	s_xor_b32 s1, s1, -1                                       // 000000003304: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003308: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000330c: be822001
	s_cbranch_execnz 2965                                      // 000000003310: bfa60b95 <packed_folded_w4a8+0x4668>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003314: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003318: 8c7e027e
	v_mul_f32_e32 v6, v7, v69                                  // 00000000331c: 100c8b07
	s_delay_alu instid0(valu_dep_1)                            // 000000003320: bf870001
	v_cmp_class_f32_e64 s1, v6, 0x198                          // 000000003324: d47e0001 0201ff06 00000198
	v_mul_f32_e32 v6, v61, v6                                  // 000000003330: 100c0d3d
	s_xor_b32 s1, s1, -1                                       // 000000003334: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003338: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000333c: be822001
	s_cbranch_execnz 2971                                      // 000000003340: bfa60b9b <packed_folded_w4a8+0x46b0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003344: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003348: 8c7e027e
	v_mul_f32_e32 v7, v0, v69                                  // 00000000334c: 100e8b00
	s_delay_alu instid0(valu_dep_1)                            // 000000003350: bf870001
	v_cmp_class_f32_e64 s1, v7, 0x198                          // 000000003354: d47e0001 0201ff07 00000198
	v_mul_f32_e32 v7, v62, v7                                  // 000000003360: 100e0f3e
	s_xor_b32 s1, s1, -1                                       // 000000003364: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003368: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000336c: be822001
	s_cbranch_execnz 2977                                      // 000000003370: bfa60ba1 <packed_folded_w4a8+0x46f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003374: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003378: 8c7e027e
	v_mul_f32_e32 v0, v1, v69                                  // 00000000337c: 10008b01
	s_delay_alu instid0(valu_dep_1)                            // 000000003380: bf870001
	v_cmp_class_f32_e64 s1, v0, 0x198                          // 000000003384: d47e0001 0201ff00 00000198
	v_mul_f32_e32 v0, v63, v0                                  // 000000003390: 1000013f
	s_xor_b32 s1, s1, -1                                       // 000000003394: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003398: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000339c: be822001
	s_cbranch_execnz 2983                                      // 0000000033a0: bfa60ba7 <packed_folded_w4a8+0x4740>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000033a8: 8c7e027e
	v_mul_f32_e32 v1, v2, v69                                  // 0000000033ac: 10028b02
	s_delay_alu instid0(valu_dep_1)                            // 0000000033b0: bf870001
	v_cmp_class_f32_e64 s1, v1, 0x198                          // 0000000033b4: d47e0001 0201ff01 00000198
	v_mul_f32_e32 v60, v64, v1                                 // 0000000033c0: 10780340
	s_xor_b32 s1, s1, -1                                       // 0000000033c4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033c8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000033cc: be822001
	s_cbranch_execnz 2989                                      // 0000000033d0: bfa60bad <packed_folded_w4a8+0x4788>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000033d8: 8c7e027e
	v_mul_f32_e32 v1, v3, v69                                  // 0000000033dc: 10028b03
	s_delay_alu instid0(valu_dep_1)                            // 0000000033e0: bf870001
	v_cmp_class_f32_e64 s1, v1, 0x198                          // 0000000033e4: d47e0001 0201ff01 00000198
	v_mul_f32_e32 v1, v65, v1                                  // 0000000033f0: 10020341
	s_xor_b32 s1, s1, -1                                       // 0000000033f4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033f8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000033fc: be822001
	s_cbranch_execnz 2995                                      // 000000003400: bfa60bb3 <packed_folded_w4a8+0x47d0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003404: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003408: 8c7e027e
	v_mul_lo_u32 v58, s19, v86                                 // 00000000340c: d72c003a 0202ac13
	v_mul_lo_u32 v61, s18, v87                                 // 000000003414: d72c003d 0202ae12
	v_mad_co_u64_u32 v[2:3], null, s18, v86, 0                 // 00000000341c: d6fe7c02 0202ac12
	v_bfe_u32 v62, v67, 16, 1                                  // 000000003424: d610003e 02052143
	v_or_b32_e32 v63, 0x400000, v67                            // 00000000342c: 387e86ff 00400000
	v_cmp_u_f32_e64 s1, v67, v67                               // 000000003434: d4180001 02028743
	v_or_b32_e32 v64, 0x400000, v4                             // 00000000343c: 388008ff 00400000
	v_cmp_u_f32_e64 s2, v1, v1                                 // 000000003444: d4180002 02020301
	v_add3_u32 v62, v62, v67, 0x7fff                           // 00000000344c: d655003e 03fe873e 00007fff
	v_mov_b32_e32 v59, s15                                     // 000000003458: 7e76020f
	v_add3_u32 v3, v3, v61, v58                                // 00000000345c: d6550003 04ea7b03
	v_bfe_u32 v61, v4, 16, 1                                   // 000000003464: d610003d 02052104
	v_or_b32_e32 v58, s14, v104                                // 00000000346c: 3874d00e
	s_wait_alu depctr_va_sdst(0)                               // 000000003470: bf88f19f
	v_cndmask_b32_e64 v62, v62, v63, s1                        // 000000003474: d501003e 00067f3e
	v_or_b32_e32 v63, 0x400000, v5                             // 00000000347c: 387e0aff 00400000
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003484: 3e040481
	v_add3_u32 v61, v61, v4, 0x7fff                            // 000000003488: d655003d 03fe093d 00007fff
	v_or_b32_e32 v108, v58, v75                                // 000000003494: 38d8973a
	v_mov_b32_e32 v109, s15                                    // 000000003498: 7eda020f
	v_or_b32_e32 v116, v164, v58                               // 00000000349c: 38e875a4
	v_or_b32_e32 v110, v163, v58                               // 0000000034a0: 38dc75a3
	v_add_co_u32 v2, s1, s8, v2                                // 0000000034a4: d7000102 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s9, v3, s1                   // 0000000034b0: d5207c03 00060609
	v_cmp_u_f32_e64 s1, v4, v4                                 // 0000000034b8: d4180001 02020904
	v_or_b32_e32 v104, v167, v58                               // 0000000034c0: 38d075a7
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000034c8: bf870002
	v_cndmask_b32_e64 v4, v61, v64, s1                         // 0000000034cc: d5010004 0006813d
	v_add_co_u32 v106, s1, v2, v160                            // 0000000034d4: d700016a 02034102
	s_wait_alu depctr_va_sdst(0)                               // 0000000034dc: bf88f19f
	v_add_co_ci_u32_e64 v107, null, v3, v161, s1               // 0000000034e0: d5207c6b 00074303
	v_add_co_u32 v2, s1, v2, s22                               // 0000000034e8: d7000102 02002d02
	v_bfe_u32 v61, v5, 16, 1                                   // 0000000034f0: d610003d 02052105
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 0000000034fc: d5207c03 00060617
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003504: bf870193
	v_add_co_u32 v112, s1, v2, v160                            // 000000003508: d7000170 02034102
	v_add3_u32 v61, v61, v5, 0x7fff                            // 000000003510: d655003d 03fe0b3d 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000351c: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000003520: bf870143
	v_add_co_ci_u32_e64 v113, null, v3, v161, s1               // 000000003524: d5207c71 00074303
	v_cmp_u_f32_e64 s1, v5, v5                                 // 00000000352c: d4180001 02020b05
	v_or_b32_e32 v64, v165, v58                                // 000000003534: 388075a5
	s_wait_alu depctr_va_sdst(0)                               // 000000003538: bf88f19f
	v_cndmask_b32_e64 v5, v61, v63, s1                         // 00000000353c: d5010005 00067f3d
	v_add_co_u32 v2, s1, v2, s22                               // 000000003544: d7000102 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000354c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 000000003550: d5207c03 00060617
	v_bfe_u32 v61, v6, 16, 1                                   // 000000003558: d610003d 02052106
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003560: bf8701a3
	v_add_co_u32 v114, s1, v2, v160                            // 000000003564: d7000172 02034102
	s_wait_alu depctr_va_sdst(0)                               // 00000000356c: bf88f19f
	v_add_co_ci_u32_e64 v115, null, v3, v161, s1               // 000000003570: d5207c73 00074303
	s_delay_alu instid0(valu_dep_3)                            // 000000003578: bf870003
	v_add3_u32 v61, v61, v6, 0x7fff                            // 00000000357c: d655003d 03fe0d3d 00007fff
	v_or_b32_e32 v63, 0x400000, v6                             // 000000003588: 387e0cff 00400000
	v_cmp_u_f32_e64 s1, v6, v6                                 // 000000003590: d4180001 02020d06
	s_clause 0x2                                               // 000000003598: bf850002
	global_store_d16_hi_b16 v[106:107], v62, off               // 00000000359c: ee09407c 1f000000 0000006a
	global_store_d16_hi_b16 v[112:113], v4, off                // 0000000035a8: ee09407c 02000000 00000070
	global_store_d16_hi_b16 v[114:115], v5, off                // 0000000035b4: ee09407c 02800000 00000072
	v_bfe_u32 v5, v7, 16, 1                                    // 0000000035c0: d6100005 02052107
	v_or_b32_e32 v6, 0x400000, v7                              // 0000000035c8: 380c0eff 00400000
	v_or_b32_e32 v62, v168, v58                                // 0000000035d0: 387c75a8
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d4: bf88f19f
	v_cndmask_b32_e64 v4, v61, v63, s1                         // 0000000035d8: d5010004 00067f3d
	v_add_co_u32 v2, s1, v2, s22                               // 0000000035e0: d7000102 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000035e8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 0000000035ec: d5207c03 00060617
	v_add3_u32 v5, v5, v7, 0x7fff                              // 0000000035f4: d6550005 03fe0f05 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003600: bf8701a3
	v_add_co_u32 v120, s1, v2, v160                            // 000000003604: d7000178 02034102
	s_wait_alu depctr_va_sdst(0)                               // 00000000360c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, v3, v161, s1               // 000000003610: d5207c79 00074303
	v_cmp_u_f32_e64 s1, v7, v7                                 // 000000003618: d4180001 02020f07
	v_or_b32_e32 v7, 0x400000, v0                              // 000000003620: 380e00ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003628: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_3)// 00000000362c: bf8701d2
	v_cndmask_b32_e64 v5, v5, v6, s1                           // 000000003630: d5010005 00060d05
	v_add_co_u32 v2, s1, v2, s22                               // 000000003638: d7000102 02002d02
	v_bfe_u32 v6, v0, 16, 1                                    // 000000003640: d6100006 02052100
	s_wait_alu depctr_va_sdst(0)                               // 000000003648: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 00000000364c: d5207c03 00060617
	v_add_co_u32 v124, s1, v2, v160                            // 000000003654: d700017c 02034102
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000365c: bf8701a3
	v_add3_u32 v6, v6, v0, 0x7fff                              // 000000003660: d6550006 03fe0106 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000366c: bf88f19f
	v_add_co_ci_u32_e64 v125, null, v3, v161, s1               // 000000003670: d5207c7d 00074303
	v_cmp_u_f32_e64 s1, v0, v0                                 // 000000003678: d4180001 02020100
	s_wait_alu depctr_va_sdst(0)                               // 000000003680: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003684: bf8701d1
	v_cndmask_b32_e64 v0, v6, v7, s1                           // 000000003688: d5010000 00060f06
	v_add_co_u32 v2, s1, v2, s22                               // 000000003690: d7000102 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003698: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 00000000369c: d5207c03 00060617
	v_bfe_u32 v6, v60, 16, 1                                   // 0000000036a4: d6100006 0205213c
	v_add_co_u32 v122, s1, v2, v160                            // 0000000036ac: d700017a 02034102
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b4: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000036b8: bf870193
	v_add_co_ci_u32_e64 v123, null, v3, v161, s1               // 0000000036bc: d5207c7b 00074303
	v_add3_u32 v6, v6, v60, 0x7fff                             // 0000000036c4: d6550006 03fe7906 00007fff
	v_or_b32_e32 v7, 0x400000, v60                             // 0000000036d0: 380e78ff 00400000
	v_cmp_u_f32_e64 s1, v60, v60                               // 0000000036d8: d4180001 0202793c
	s_clause 0x2                                               // 0000000036e0: bf850002
	global_store_d16_hi_b16 v[120:121], v4, off                // 0000000036e4: ee09407c 02000000 00000078
	global_store_d16_hi_b16 v[124:125], v5, off                // 0000000036f0: ee09407c 02800000 0000007c
	global_store_d16_hi_b16 v[122:123], v0, off                // 0000000036fc: ee09407c 00000000 0000007a
	v_or_b32_e32 v60, v166, v58                                // 000000003708: 387875a6
	v_or_b32_e32 v58, v169, v58                                // 00000000370c: 387475a9
	s_wait_alu depctr_va_sdst(0)                               // 000000003710: bf88f19f
	v_cndmask_b32_e64 v0, v6, v7, s1                           // 000000003714: d5010000 00060f06
	v_add_co_u32 v4, s1, v2, s22                               // 00000000371c: d7000104 02002d02
	v_bfe_u32 v2, v1, 16, 1                                    // 000000003724: d6100002 02052101
	s_wait_alu depctr_va_sdst(0)                               // 00000000372c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v3, s1                  // 000000003730: d5207c05 00060617
	v_mov_b32_e32 v3, s15                                      // 000000003738: 7e06020f
	v_add_co_u32 v126, s1, v4, v160                            // 00000000373c: d700017e 02034104
	v_add3_u32 v6, v2, v1, 0x7fff                              // 000000003744: d6550006 03fe0302 00007fff
	v_or_b32_e32 v2, 7, v108                                   // 000000003750: 3804d887
	s_wait_alu depctr_va_sdst(0)                               // 000000003754: bf88f19f
	v_add_co_ci_u32_e64 v127, null, v5, v161, s1               // 000000003758: d5207c7f 00074305
	v_add_co_u32 v4, s1, v4, s22                               // 000000003760: d7000104 02002d04
	v_or_b32_e32 v7, 0x400000, v1                              // 000000003768: 380e02ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003770: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s1                  // 000000003774: d5207c05 00060a17
	v_cmp_gt_u64_e64 s1, s[16:17], v[2:3]                      // 00000000377c: d45c0001 02020410
	global_store_d16_hi_b16 v[126:127], v0, off                // 000000003784: ee09407c 00000000 0000007e
	v_cndmask_b32_e64 v1, v6, v7, s2                           // 000000003790: d5010001 000a0f06
	v_add_co_u32 v128, s2, v4, v160                            // 000000003798: d7000280 02034104
	s_wait_alu depctr_va_sdst(0)                               // 0000000037a0: bf88f19f
	v_add_co_ci_u32_e64 v129, null, v5, v161, s2               // 0000000037a4: d5207c81 000b4305
	s_and_b32 s2, s1, s3                                       // 0000000037ac: 8b020301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037b0: bf88ff9e
	s_and_b32 s2, s10, s2                                      // 0000000037b4: 8b02020a
	global_store_d16_hi_b16 v[128:129], v1, off                // 0000000037b8: ee09407c 00800000 00000080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037c4: bf88ff9e
	s_xor_b32 s2, s2, -1                                       // 0000000037c8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037cc: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 0000000037d0: be842002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037d4: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 0000000037d8: 8d05047e
	s_cbranch_execz 133                                        // 0000000037dc: bfa50085 <packed_folded_w4a8+0x1ef4>
	v_mov_b32_e32 v117, v59                                    // 0000000037e0: 7eea033b
	v_cmp_gt_i64_e64 s2, s[16:17], v[108:109]                  // 0000000037e4: d4540002 0202d810
	v_mov_b32_e32 v111, v59                                    // 0000000037ec: 7ede033b
	v_mov_b32_e32 v105, v59                                    // 0000000037f0: 7ed2033b
	v_mov_b32_e32 v65, v59                                     // 0000000037f4: 7e82033b
	v_cmp_gt_i64_e64 s4, s[16:17], v[116:117]                  // 0000000037f8: d4540004 0202e810
	v_mov_b32_e32 v63, v59                                     // 000000003800: 7e7e033b
	s_wait_alu depctr_va_sdst(0)                               // 000000003804: bf88f19f
	v_cndmask_b32_e64 v1, 0, v109, s2                          // 000000003808: d5010001 000ada80
	v_cndmask_b32_e64 v0, 0, v108, s2                          // 000000003810: d5010000 000ad880
	v_cmp_gt_i64_e64 s2, s[16:17], v[110:111]                  // 000000003818: d4540002 0202dc10
	v_mov_b32_e32 v61, v59                                     // 000000003820: 7e7a033b
	v_cndmask_b32_e64 v3, 0, v117, s4                          // 000000003824: d5010003 0012ea80
	v_cndmask_b32_e64 v2, 0, v116, s4                          // 00000000382c: d5010002 0012e880
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000003834: 3e000082
	v_cmp_gt_i64_e64 s4, s[16:17], v[104:105]                  // 000000003838: d4540004 0202d010
	s_wait_alu depctr_va_sdst(0)                               // 000000003840: bf88f19f
	v_cndmask_b32_e64 v5, 0, v111, s2                          // 000000003844: d5010005 000ade80
	v_cndmask_b32_e64 v4, 0, v110, s2                          // 00000000384c: d5010004 000adc80
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000003854: 3e040482
	v_add_co_u32 v0, s2, s6, v0                                // 000000003858: d7000200 02020006
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000003860: bf870233
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000003864: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s2                   // 00000000386c: d5207c01 000a0207
	v_add_co_u32 v2, s2, s6, v2                                // 000000003874: d7000202 02020406
	v_cndmask_b32_e64 v7, 0, v105, s4                          // 00000000387c: d5010007 0012d280
	v_cndmask_b32_e64 v6, 0, v104, s4                          // 000000003884: d5010006 0012d080
	s_wait_alu depctr_va_sdst(0)                               // 00000000388c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s2                   // 000000003890: d5207c03 000a0607
	v_cmp_gt_i64_e64 s2, s[16:17], v[64:65]                    // 000000003898: d4540002 02028010
	v_add_co_u32 v131, s4, s6, v4                              // 0000000038a0: d7000483 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a8: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s7, v5, s4                 // 0000000038ac: d5207c84 00120a07
	v_cmp_gt_i64_e64 s4, s[16:17], v[62:63]                    // 0000000038b4: d4540004 02027c10
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000038bc: 3e080c82
	v_cndmask_b32_e64 v7, 0, v65, s2                           // 0000000038c0: d5010007 000a8280
	v_cndmask_b32_e64 v6, 0, v64, s2                           // 0000000038c8: d5010006 000a8080
	v_cmp_gt_i64_e64 s2, s[16:17], v[60:61]                    // 0000000038d0: d4540002 02027810
	s_wait_alu depctr_va_sdst(0)                               // 0000000038d8: bf88f19f
	v_cndmask_b32_e64 v134, 0, v63, s4                         // 0000000038dc: d5010086 00127e80
	v_cndmask_b32_e64 v133, 0, v62, s4                         // 0000000038e4: d5010085 00127c80
	v_add_co_u32 v135, s4, s6, v4                              // 0000000038ec: d7000487 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f4: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s7, v5, s4                 // 0000000038f8: d5207c88 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000003900: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[133:134]                    // 000000003904: 3e0d0a82
	v_cndmask_b32_e64 v134, 0, v61, s2                         // 000000003908: d5010086 000a7a80
	v_cndmask_b32_e64 v133, 0, v60, s2                         // 000000003910: d5010085 000a7880
	v_cmp_gt_i64_e64 s2, s[16:17], v[58:59]                    // 000000003918: d4540002 02027410
	v_add_co_u32 v137, s4, s6, v4                              // 000000003920: d7000489 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000003928: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s7, v5, s4                 // 00000000392c: d5207c8a 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[133:134]                    // 000000003934: 3e090a82
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000003938: bf8701d4
	v_cndmask_b32_e64 v134, 0, v59, s2                         // 00000000393c: d5010086 000a7680
	v_cndmask_b32_e64 v133, 0, v58, s2                         // 000000003944: d5010085 000a7480
	v_add_co_u32 v139, s2, s6, v6                              // 00000000394c: d700028b 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000003954: bf88f19f
	v_add_co_ci_u32_e64 v140, null, s7, v7, s2                 // 000000003958: d5207c8c 000a0e07
	v_lshlrev_b64_e32 v[6:7], 2, v[133:134]                    // 000000003960: 3e0d0a82
	v_add_co_u32 v133, s2, s6, v4                              // 000000003964: d7000285 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000396c: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s7, v5, s2                 // 000000003970: d5207c86 000a0a07
	s_delay_alu instid0(valu_dep_3)                            // 000000003978: bf870003
	v_add_co_u32 v141, s2, s6, v6                              // 00000000397c: d700028d 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000003984: bf88f19f
	v_add_co_ci_u32_e64 v142, null, s7, v7, s2                 // 000000003988: d5207c8e 000a0e07
	s_clause 0x7                                               // 000000003990: bf850007
	global_load_b32 v4, v[0:1], off                            // 000000003994: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 0000000039a0: ee05007c 00000005 00000002
	global_load_b32 v6, v[131:132], off                        // 0000000039ac: ee05007c 00000006 00000083
	global_load_b32 v7, v[135:136], off                        // 0000000039b8: ee05007c 00000007 00000087
	global_load_b32 v0, v[137:138], off                        // 0000000039c4: ee05007c 00000000 00000089
	global_load_b32 v1, v[139:140], off                        // 0000000039d0: ee05007c 00000001 0000008b
	global_load_b32 v2, v[133:134], off                        // 0000000039dc: ee05007c 00000002 00000085
	global_load_b32 v3, v[141:142], off                        // 0000000039e8: ee05007c 00000003 0000008d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f4: bf88ff9e
	s_and_not1_saveexec_b32 s4, s5                             // 0000000039f8: be843005
	s_cbranch_execz 28                                         // 0000000039fc: bfa5001c <packed_folded_w4a8+0x1f70>
	s_wait_loadcnt 0x3                                         // 000000003a00: bfc00003
	v_add_co_u32 v0, s2, v74, s14                              // 000000003a04: d7000200 02001d4a
	s_wait_loadcnt 0x2                                         // 000000003a0c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000003a10: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s2                   // 000000003a14: d5207c01 00081e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003a1c: bf870122
	v_add_co_u32 v0, s2, v0, v75                               // 000000003a20: d7000200 02029700
	s_wait_alu depctr_va_sdst(0)                               // 000000003a28: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, v1, s2                    // 000000003a2c: d5207c01 000a0280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003a34: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000003a38: 3e000082
	v_add_co_u32 v0, s2, s6, v0                                // 000000003a3c: d7000200 02020006
	s_wait_alu depctr_va_sdst(0)                               // 000000003a44: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003a48: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, s2                   // 000000003a4c: d5207c01 000a0207
	global_load_b128 v[4:7], v[0:1], off offset:128            // 000000003a54: ee05c07c 00000004 00008000
	s_wait_loadcnt 0x1                                         // 000000003a60: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:144            // 000000003a64: ee05c07c 00000000 00009000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003a74: 8c7e047e
	global_load_u8 v61, v[154:155], off                        // 000000003a78: ee04007c 0000003d 0000009a
	s_wait_loadcnt 0x0                                         // 000000003a84: bfc00000
	v_lshlrev_b32_e32 v63, 23, v61                             // 000000003a88: 307e7a97
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003a8c: bf870091
	v_mul_f32_e32 v61, v4, v63                                 // 000000003a90: 107a7f04
	v_cmp_class_f32_e64 s2, v61, 0x198                         // 000000003a94: d47e0002 0201ff3d 00000198
	v_mul_f32_e32 v61, v50, v61                                // 000000003aa0: 107a7b32
	s_xor_b32 s2, s2, -1                                       // 000000003aa4: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003aa8: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003aac: be842002
	s_cbranch_execnz 2585                                      // 000000003ab0: bfa60a19 <packed_folded_w4a8+0x4818>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ab4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003ab8: 8c7e047e
	v_mul_f32_e32 v4, v5, v63                                  // 000000003abc: 10087f05
	s_delay_alu instid0(valu_dep_1)                            // 000000003ac0: bf870001
	v_cmp_class_f32_e64 s2, v4, 0x198                          // 000000003ac4: d47e0002 0201ff04 00000198
	v_mul_f32_e32 v4, v51, v4                                  // 000000003ad0: 10080933
	s_xor_b32 s2, s2, -1                                       // 000000003ad4: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ad8: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003adc: be842002
	s_cbranch_execnz 2591                                      // 000000003ae0: bfa60a1f <packed_folded_w4a8+0x4860>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ae4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003ae8: 8c7e047e
	v_mul_f32_e32 v5, v6, v63                                  // 000000003aec: 100a7f06
	s_delay_alu instid0(valu_dep_1)                            // 000000003af0: bf870001
	v_cmp_class_f32_e64 s2, v5, 0x198                          // 000000003af4: d47e0002 0201ff05 00000198
	v_mul_f32_e32 v5, v52, v5                                  // 000000003b00: 100a0b34
	s_xor_b32 s2, s2, -1                                       // 000000003b04: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b08: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b0c: be842002
	s_cbranch_execnz 2597                                      // 000000003b10: bfa60a25 <packed_folded_w4a8+0x48a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b18: 8c7e047e
	v_mul_f32_e32 v6, v7, v63                                  // 000000003b1c: 100c7f07
	s_delay_alu instid0(valu_dep_1)                            // 000000003b20: bf870001
	v_cmp_class_f32_e64 s2, v6, 0x198                          // 000000003b24: d47e0002 0201ff06 00000198
	v_mul_f32_e32 v6, v53, v6                                  // 000000003b30: 100c0d35
	s_xor_b32 s2, s2, -1                                       // 000000003b34: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b38: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b3c: be842002
	s_cbranch_execnz 2603                                      // 000000003b40: bfa60a2b <packed_folded_w4a8+0x48f0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b48: 8c7e047e
	v_mul_f32_e32 v7, v0, v63                                  // 000000003b4c: 100e7f00
	s_delay_alu instid0(valu_dep_1)                            // 000000003b50: bf870001
	v_cmp_class_f32_e64 s2, v7, 0x198                          // 000000003b54: d47e0002 0201ff07 00000198
	v_mul_f32_e32 v7, v54, v7                                  // 000000003b60: 100e0f36
	s_xor_b32 s2, s2, -1                                       // 000000003b64: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b68: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b6c: be842002
	s_cbranch_execnz 2609                                      // 000000003b70: bfa60a31 <packed_folded_w4a8+0x4938>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b78: 8c7e047e
	v_mul_f32_e32 v0, v1, v63                                  // 000000003b7c: 10007f01
	s_delay_alu instid0(valu_dep_1)                            // 000000003b80: bf870001
	v_cmp_class_f32_e64 s2, v0, 0x198                          // 000000003b84: d47e0002 0201ff00 00000198
	v_mul_f32_e32 v0, v55, v0                                  // 000000003b90: 10000137
	s_xor_b32 s2, s2, -1                                       // 000000003b94: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b98: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b9c: be842002
	s_cbranch_execnz 2615                                      // 000000003ba0: bfa60a37 <packed_folded_w4a8+0x4980>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ba4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003ba8: 8c7e047e
	v_mul_f32_e32 v1, v2, v63                                  // 000000003bac: 10027f02
	s_delay_alu instid0(valu_dep_1)                            // 000000003bb0: bf870001
	v_cmp_class_f32_e64 s2, v1, 0x198                          // 000000003bb4: d47e0002 0201ff01 00000198
	v_mul_f32_e32 v52, v56, v1                                 // 000000003bc0: 10680338
	s_xor_b32 s2, s2, -1                                       // 000000003bc4: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bc8: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003bcc: be842002
	s_cbranch_execnz 2621                                      // 000000003bd0: bfa60a3d <packed_folded_w4a8+0x49c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003bd8: 8c7e047e
	v_mul_f32_e32 v1, v3, v63                                  // 000000003bdc: 10027f03
	s_delay_alu instid0(valu_dep_1)                            // 000000003be0: bf870001
	v_cmp_class_f32_e64 s2, v1, 0x198                          // 000000003be4: d47e0002 0201ff01 00000198
	v_mul_f32_e32 v1, v57, v1                                  // 000000003bf0: 10020339
	s_xor_b32 s2, s2, -1                                       // 000000003bf4: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bf8: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003bfc: be842002
	s_cbranch_execnz 2627                                      // 000000003c00: bfa60a43 <packed_folded_w4a8+0x4a10>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003c08: 8c7e047e
	v_mul_lo_u32 v50, s19, v108                                // 000000003c0c: d72c0032 0202d813
	v_mul_lo_u32 v53, s18, v109                                // 000000003c14: d72c0035 0202da12
	v_mad_co_u64_u32 v[2:3], null, s18, v108, 0                // 000000003c1c: d6fe7c02 0202d812
	v_bfe_u32 v54, v61, 16, 1                                  // 000000003c24: d6100036 0205213d
	v_or_b32_e32 v55, 0x400000, v61                            // 000000003c2c: 386e7aff 00400000
	v_cmp_u_f32_e64 s2, v61, v61                               // 000000003c34: d4180002 02027b3d
	v_or_b32_e32 v56, 0x400000, v4                             // 000000003c3c: 387008ff 00400000
	v_cmp_u_f32_e64 s4, v1, v1                                 // 000000003c44: d4180004 02020301
	v_add3_u32 v54, v54, v61, 0x7fff                           // 000000003c4c: d6550036 03fe7b36 00007fff
	v_mov_b32_e32 v51, s15                                     // 000000003c58: 7e66020f
	v_add3_u32 v3, v3, v53, v50                                // 000000003c5c: d6550003 04ca6b03
	v_bfe_u32 v53, v4, 16, 1                                   // 000000003c64: d6100035 02052104
	v_or_b32_e32 v50, s14, v130                                // 000000003c6c: 3865040e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c70: bf88f19f
	v_cndmask_b32_e64 v54, v54, v55, s2                        // 000000003c74: d5010036 000a6f36
	v_or_b32_e32 v55, 0x400000, v5                             // 000000003c7c: 386e0aff 00400000
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003c84: 3e040481
	v_add3_u32 v53, v53, v4, 0x7fff                            // 000000003c88: d6550035 03fe0935 00007fff
	v_or_b32_e32 v134, v50, v75                                // 000000003c94: 390c9732
	v_mov_b32_e32 v135, s15                                    // 000000003c98: 7f0e020f
	v_or_b32_e32 v142, v164, v50                               // 000000003c9c: 391c65a4
	v_or_b32_e32 v136, v163, v50                               // 000000003ca0: 391065a3
	v_add_co_u32 v2, s2, s8, v2                                // 000000003ca4: d7000202 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000003cac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s9, v3, s2                   // 000000003cb0: d5207c03 000a0609
	v_cmp_u_f32_e64 s2, v4, v4                                 // 000000003cb8: d4180002 02020904
	v_or_b32_e32 v130, v167, v50                               // 000000003cc0: 390465a7
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003cc8: bf870002
	v_cndmask_b32_e64 v4, v53, v56, s2                         // 000000003ccc: d5010004 000a7135
	v_add_co_u32 v132, s2, v2, v160                            // 000000003cd4: d7000284 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000003cdc: bf88f19f
	v_add_co_ci_u32_e64 v133, null, v3, v161, s2               // 000000003ce0: d5207c85 000b4303
	v_add_co_u32 v2, s2, v2, s22                               // 000000003ce8: d7000202 02002d02
	v_bfe_u32 v53, v5, 16, 1                                   // 000000003cf0: d6100035 02052105
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003cfc: d5207c03 000a0617
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d04: bf870193
	v_add_co_u32 v138, s2, v2, v160                            // 000000003d08: d700028a 02034102
	v_add3_u32 v53, v53, v5, 0x7fff                            // 000000003d10: d6550035 03fe0b35 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003d1c: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000003d20: bf870143
	v_add_co_ci_u32_e64 v139, null, v3, v161, s2               // 000000003d24: d5207c8b 000b4303
	v_cmp_u_f32_e64 s2, v5, v5                                 // 000000003d2c: d4180002 02020b05
	v_or_b32_e32 v56, v165, v50                                // 000000003d34: 387065a5
	s_wait_alu depctr_va_sdst(0)                               // 000000003d38: bf88f19f
	v_cndmask_b32_e64 v5, v53, v55, s2                         // 000000003d3c: d5010005 000a6f35
	v_add_co_u32 v2, s2, v2, s22                               // 000000003d44: d7000202 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003d4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003d50: d5207c03 000a0617
	v_bfe_u32 v53, v6, 16, 1                                   // 000000003d58: d6100035 02052106
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d60: bf8701a3
	v_add_co_u32 v140, s2, v2, v160                            // 000000003d64: d700028c 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000003d6c: bf88f19f
	v_add_co_ci_u32_e64 v141, null, v3, v161, s2               // 000000003d70: d5207c8d 000b4303
	s_delay_alu instid0(valu_dep_3)                            // 000000003d78: bf870003
	v_add3_u32 v53, v53, v6, 0x7fff                            // 000000003d7c: d6550035 03fe0d35 00007fff
	v_or_b32_e32 v55, 0x400000, v6                             // 000000003d88: 386e0cff 00400000
	v_cmp_u_f32_e64 s2, v6, v6                                 // 000000003d90: d4180002 02020d06
	s_clause 0x2                                               // 000000003d98: bf850002
	global_store_d16_hi_b16 v[132:133], v54, off               // 000000003d9c: ee09407c 1b000000 00000084
	global_store_d16_hi_b16 v[138:139], v4, off                // 000000003da8: ee09407c 02000000 0000008a
	global_store_d16_hi_b16 v[140:141], v5, off                // 000000003db4: ee09407c 02800000 0000008c
	v_bfe_u32 v5, v7, 16, 1                                    // 000000003dc0: d6100005 02052107
	v_or_b32_e32 v6, 0x400000, v7                              // 000000003dc8: 380c0eff 00400000
	v_or_b32_e32 v54, v168, v50                                // 000000003dd0: 386c65a8
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd4: bf88f19f
	v_cndmask_b32_e64 v4, v53, v55, s2                         // 000000003dd8: d5010004 000a6f35
	v_add_co_u32 v2, s2, v2, s22                               // 000000003de0: d7000202 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003de8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003dec: d5207c03 000a0617
	v_add3_u32 v5, v5, v7, 0x7fff                              // 000000003df4: d6550005 03fe0f05 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e00: bf8701a3
	v_add_co_u32 v144, s2, v2, v160                            // 000000003e04: d7000290 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000003e0c: bf88f19f
	v_add_co_ci_u32_e64 v145, null, v3, v161, s2               // 000000003e10: d5207c91 000b4303
	v_cmp_u_f32_e64 s2, v7, v7                                 // 000000003e18: d4180002 02020f07
	v_or_b32_e32 v7, 0x400000, v0                              // 000000003e20: 380e00ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003e28: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_3)// 000000003e2c: bf8701d2
	v_cndmask_b32_e64 v5, v5, v6, s2                           // 000000003e30: d5010005 000a0d05
	v_add_co_u32 v2, s2, v2, s22                               // 000000003e38: d7000202 02002d02
	v_bfe_u32 v6, v0, 16, 1                                    // 000000003e40: d6100006 02052100
	s_wait_alu depctr_va_sdst(0)                               // 000000003e48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003e4c: d5207c03 000a0617
	v_add_co_u32 v148, s2, v2, v160                            // 000000003e54: d7000294 02034102
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e5c: bf8701a3
	v_add3_u32 v6, v6, v0, 0x7fff                              // 000000003e60: d6550006 03fe0106 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003e6c: bf88f19f
	v_add_co_ci_u32_e64 v149, null, v3, v161, s2               // 000000003e70: d5207c95 000b4303
	v_cmp_u_f32_e64 s2, v0, v0                                 // 000000003e78: d4180002 02020100
	s_wait_alu depctr_va_sdst(0)                               // 000000003e80: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003e84: bf8701d1
	v_cndmask_b32_e64 v0, v6, v7, s2                           // 000000003e88: d5010000 000a0f06
	v_add_co_u32 v2, s2, v2, s22                               // 000000003e90: d7000202 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003e98: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003e9c: d5207c03 000a0617
	v_bfe_u32 v6, v52, 16, 1                                   // 000000003ea4: d6100006 02052134
	v_add_co_u32 v146, s2, v2, v160                            // 000000003eac: d7000292 02034102
	s_wait_alu depctr_va_sdst(0)                               // 000000003eb4: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003eb8: bf870193
	v_add_co_ci_u32_e64 v147, null, v3, v161, s2               // 000000003ebc: d5207c93 000b4303
	v_add3_u32 v6, v6, v52, 0x7fff                             // 000000003ec4: d6550006 03fe6906 00007fff
	v_or_b32_e32 v7, 0x400000, v52                             // 000000003ed0: 380e68ff 00400000
	v_cmp_u_f32_e64 s2, v52, v52                               // 000000003ed8: d4180002 02026934
	s_clause 0x2                                               // 000000003ee0: bf850002
	global_store_d16_hi_b16 v[144:145], v4, off                // 000000003ee4: ee09407c 02000000 00000090
	global_store_d16_hi_b16 v[148:149], v5, off                // 000000003ef0: ee09407c 02800000 00000094
	global_store_d16_hi_b16 v[146:147], v0, off                // 000000003efc: ee09407c 00000000 00000092
	v_or_b32_e32 v52, v166, v50                                // 000000003f08: 386865a6
	v_or_b32_e32 v50, v169, v50                                // 000000003f0c: 386465a9
	s_wait_alu depctr_va_sdst(0)                               // 000000003f10: bf88f19f
	v_cndmask_b32_e64 v0, v6, v7, s2                           // 000000003f14: d5010000 000a0f06
	v_add_co_u32 v4, s2, v2, s22                               // 000000003f1c: d7000204 02002d02
	v_bfe_u32 v2, v1, 16, 1                                    // 000000003f24: d6100002 02052101
	s_wait_alu depctr_va_sdst(0)                               // 000000003f2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v3, s2                  // 000000003f30: d5207c05 000a0617
	v_mov_b32_e32 v3, s15                                      // 000000003f38: 7e06020f
	v_add_co_u32 v150, s2, v4, v160                            // 000000003f3c: d7000296 02034104
	v_add3_u32 v6, v2, v1, 0x7fff                              // 000000003f44: d6550006 03fe0302 00007fff
	v_or_b32_e32 v2, 7, v134                                   // 000000003f50: 38050c87
	s_wait_alu depctr_va_sdst(0)                               // 000000003f54: bf88f19f
	v_add_co_ci_u32_e64 v151, null, v5, v161, s2               // 000000003f58: d5207c97 000b4305
	v_add_co_u32 v4, s2, v4, s22                               // 000000003f60: d7000204 02002d04
	v_or_b32_e32 v7, 0x400000, v1                              // 000000003f68: 380e02ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003f70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 000000003f74: d5207c05 000a0a17
	v_cmp_gt_u64_e64 s2, s[16:17], v[2:3]                      // 000000003f7c: d45c0002 02020410
	global_store_d16_hi_b16 v[150:151], v0, off                // 000000003f84: ee09407c 00000000 00000096
	v_cndmask_b32_e64 v1, v6, v7, s4                           // 000000003f90: d5010001 00120f06
	v_add_co_u32 v152, s4, v4, v160                            // 000000003f98: d7000498 02034104
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa0: bf88f19f
	v_add_co_ci_u32_e64 v153, null, v5, v161, s4               // 000000003fa4: d5207c99 00134305
	s_and_b32 s3, s2, s3                                       // 000000003fac: 8b030302
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fb0: bf88ff9e
	s_and_b32 s3, s10, s3                                      // 000000003fb4: 8b03030a
	global_store_d16_hi_b16 v[152:153], v1, off                // 000000003fb8: ee09407c 00800000 00000098
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fc4: bf88ff9e
	s_xor_b32 s3, s3, -1                                       // 000000003fc8: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fcc: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003fd0: be842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fd4: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000003fd8: 8d05047e
	s_cbranch_execz 133                                        // 000000003fdc: bfa50085 <packed_folded_w4a8+0x26f4>
	v_mov_b32_e32 v143, v51                                    // 000000003fe0: 7f1e0333
	v_cmp_gt_i64_e64 s3, s[16:17], v[134:135]                  // 000000003fe4: d4540003 02030c10
	v_mov_b32_e32 v137, v51                                    // 000000003fec: 7f120333
	v_mov_b32_e32 v131, v51                                    // 000000003ff0: 7f060333
	v_mov_b32_e32 v57, v51                                     // 000000003ff4: 7e720333
	v_cmp_gt_i64_e64 s4, s[16:17], v[142:143]                  // 000000003ff8: d4540004 02031c10
	v_mov_b32_e32 v55, v51                                     // 000000004000: 7e6e0333
	s_wait_alu depctr_va_sdst(0)                               // 000000004004: bf88f19f
	v_cndmask_b32_e64 v1, 0, v135, s3                          // 000000004008: d5010001 000f0e80
	v_cndmask_b32_e64 v0, 0, v134, s3                          // 000000004010: d5010000 000f0c80
	v_cmp_gt_i64_e64 s3, s[16:17], v[136:137]                  // 000000004018: d4540003 02031010
	v_mov_b32_e32 v53, v51                                     // 000000004020: 7e6a0333
	v_cndmask_b32_e64 v3, 0, v143, s4                          // 000000004024: d5010003 00131e80
	v_cndmask_b32_e64 v2, 0, v142, s4                          // 00000000402c: d5010002 00131c80
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000004034: 3e000082
	v_cmp_gt_i64_e64 s4, s[16:17], v[130:131]                  // 000000004038: d4540004 02030410
	s_wait_alu depctr_va_sdst(0)                               // 000000004040: bf88f19f
	v_cndmask_b32_e64 v5, 0, v137, s3                          // 000000004044: d5010005 000f1280
	v_cndmask_b32_e64 v4, 0, v136, s3                          // 00000000404c: d5010004 000f1080
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000004054: 3e040482
	v_add_co_u32 v0, s3, s6, v0                                // 000000004058: d7000300 02020006
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000004060: bf870233
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000004064: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 000000004068: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s3                   // 00000000406c: d5207c01 000e0207
	v_add_co_u32 v2, s3, s6, v2                                // 000000004074: d7000302 02020406
	v_cndmask_b32_e64 v7, 0, v131, s4                          // 00000000407c: d5010007 00130680
	v_cndmask_b32_e64 v6, 0, v130, s4                          // 000000004084: d5010006 00130480
	s_wait_alu depctr_va_sdst(0)                               // 00000000408c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s3                   // 000000004090: d5207c03 000e0607
	v_cmp_gt_i64_e64 s3, s[16:17], v[56:57]                    // 000000004098: d4540003 02027010
	v_add_co_u32 v156, s4, s6, v4                              // 0000000040a0: d700049c 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000040a8: bf88f19f
	v_add_co_ci_u32_e64 v157, null, s7, v5, s4                 // 0000000040ac: d5207c9d 00120a07
	v_cmp_gt_i64_e64 s4, s[16:17], v[54:55]                    // 0000000040b4: d4540004 02026c10
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000040bc: 3e080c82
	v_cndmask_b32_e64 v7, 0, v57, s3                           // 0000000040c0: d5010007 000e7280
	v_cndmask_b32_e64 v6, 0, v56, s3                           // 0000000040c8: d5010006 000e7080
	v_cmp_gt_i64_e64 s3, s[16:17], v[52:53]                    // 0000000040d0: d4540003 02026810
	s_wait_alu depctr_va_sdst(0)                               // 0000000040d8: bf88f19f
	v_cndmask_b32_e64 v159, 0, v55, s4                         // 0000000040dc: d501009f 00126e80
	v_cndmask_b32_e64 v158, 0, v54, s4                         // 0000000040e4: d501009e 00126c80
	v_add_co_u32 v170, s4, s6, v4                              // 0000000040ec: d70004aa 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000040f4: bf88f19f
	v_add_co_ci_u32_e64 v171, null, s7, v5, s4                 // 0000000040f8: d5207cab 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000004100: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[158:159]                    // 000000004104: 3e0d3c82
	v_cndmask_b32_e64 v159, 0, v53, s3                         // 000000004108: d501009f 000e6a80
	v_cndmask_b32_e64 v158, 0, v52, s3                         // 000000004110: d501009e 000e6880
	v_cmp_gt_i64_e64 s3, s[16:17], v[50:51]                    // 000000004118: d4540003 02026410
	v_add_co_u32 v172, s4, s6, v4                              // 000000004120: d70004ac 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000004128: bf88f19f
	v_add_co_ci_u32_e64 v173, null, s7, v5, s4                 // 00000000412c: d5207cad 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[158:159]                    // 000000004134: 3e093c82
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000004138: bf8701d4
	v_cndmask_b32_e64 v159, 0, v51, s3                         // 00000000413c: d501009f 000e6680
	v_cndmask_b32_e64 v158, 0, v50, s3                         // 000000004144: d501009e 000e6480
	v_add_co_u32 v174, s3, s6, v6                              // 00000000414c: d70003ae 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000004154: bf88f19f
	v_add_co_ci_u32_e64 v175, null, s7, v7, s3                 // 000000004158: d5207caf 000e0e07
	v_lshlrev_b64_e32 v[6:7], 2, v[158:159]                    // 000000004160: 3e0d3c82
	v_add_co_u32 v158, s3, s6, v4                              // 000000004164: d700039e 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000416c: bf88f19f
	v_add_co_ci_u32_e64 v159, null, s7, v5, s3                 // 000000004170: d5207c9f 000e0a07
	s_delay_alu instid0(valu_dep_3)                            // 000000004178: bf870003
	v_add_co_u32 v176, s3, s6, v6                              // 00000000417c: d70003b0 02020c06
	s_wait_alu depctr_va_sdst(0)                               // 000000004184: bf88f19f
	v_add_co_ci_u32_e64 v177, null, s7, v7, s3                 // 000000004188: d5207cb1 000e0e07
	s_clause 0x7                                               // 000000004190: bf850007
	global_load_b32 v4, v[0:1], off                            // 000000004194: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 0000000041a0: ee05007c 00000005 00000002
	global_load_b32 v6, v[156:157], off                        // 0000000041ac: ee05007c 00000006 0000009c
	global_load_b32 v7, v[170:171], off                        // 0000000041b8: ee05007c 00000007 000000aa
	global_load_b32 v0, v[172:173], off                        // 0000000041c4: ee05007c 00000000 000000ac
	global_load_b32 v1, v[174:175], off                        // 0000000041d0: ee05007c 00000001 000000ae
	global_load_b32 v2, v[158:159], off                        // 0000000041dc: ee05007c 00000002 0000009e
	global_load_b32 v3, v[176:177], off                        // 0000000041e8: ee05007c 00000003 000000b0
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041f4: bf88ff9e
	s_and_not1_saveexec_b32 s4, s5                             // 0000000041f8: be843005
	s_cbranch_execz 28                                         // 0000000041fc: bfa5001c <packed_folded_w4a8+0x2770>
	s_wait_loadcnt 0x3                                         // 000000004200: bfc00003
	v_add_co_u32 v0, s3, v74, s14                              // 000000004204: d7000300 02001d4a
	s_wait_loadcnt 0x2                                         // 00000000420c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000004210: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s3                   // 000000004214: d5207c01 000c1e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000421c: bf870122
	v_add_co_u32 v0, s3, v0, v75                               // 000000004220: d7000300 02029700
	s_wait_alu depctr_va_sdst(0)                               // 000000004228: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, v1, s3                    // 00000000422c: d5207c01 000e0280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004234: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000004238: 3e000082
	v_add_co_u32 v0, s3, s6, v0                                // 00000000423c: d7000300 02020006
	s_wait_alu depctr_va_sdst(0)                               // 000000004244: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000004248: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, s3                   // 00000000424c: d5207c01 000e0207
	global_load_b128 v[4:7], v[0:1], off offset:192            // 000000004254: ee05c07c 00000004 0000c000
	s_wait_loadcnt 0x1                                         // 000000004260: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:208            // 000000004264: ee05c07c 00000000 0000d000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004270: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004274: 8c7e047e
	global_load_u8 v53, v[154:155], off                        // 000000004278: ee04007c 00000035 0000009a
	s_wait_loadcnt 0x0                                         // 000000004284: bfc00000
	v_lshlrev_b32_e32 v55, 23, v53                             // 000000004288: 306e6a97
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000428c: bf870091
	v_mul_f32_e32 v53, v4, v55                                 // 000000004290: 106a6f04
	v_cmp_class_f32_e64 s3, v53, 0x198                         // 000000004294: d47e0003 0201ff35 00000198
	v_mul_f32_e32 v53, v42, v53                                // 0000000042a0: 106a6b2a
	s_xor_b32 s3, s3, -1                                       // 0000000042a4: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042a8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000042ac: be842003
	s_cbranch_execnz 2217                                      // 0000000042b0: bfa608a9 <packed_folded_w4a8+0x4a58>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000042b8: 8c7e047e
	v_mul_f32_e32 v4, v5, v55                                  // 0000000042bc: 10086f05
	s_delay_alu instid0(valu_dep_1)                            // 0000000042c0: bf870001
	v_cmp_class_f32_e64 s3, v4, 0x198                          // 0000000042c4: d47e0003 0201ff04 00000198
	v_mul_f32_e32 v4, v43, v4                                  // 0000000042d0: 1008092b
	s_xor_b32 s3, s3, -1                                       // 0000000042d4: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042d8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000042dc: be842003
	s_cbranch_execnz 2223                                      // 0000000042e0: bfa608af <packed_folded_w4a8+0x4aa0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000042e8: 8c7e047e
	v_mul_f32_e32 v5, v6, v55                                  // 0000000042ec: 100a6f06
	s_delay_alu instid0(valu_dep_1)                            // 0000000042f0: bf870001
	v_cmp_class_f32_e64 s3, v5, 0x198                          // 0000000042f4: d47e0003 0201ff05 00000198
	v_mul_f32_e32 v5, v44, v5                                  // 000000004300: 100a0b2c
	s_xor_b32 s3, s3, -1                                       // 000000004304: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004308: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000430c: be842003
	s_cbranch_execnz 2229                                      // 000000004310: bfa608b5 <packed_folded_w4a8+0x4ae8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004314: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004318: 8c7e047e
	v_mul_f32_e32 v6, v7, v55                                  // 00000000431c: 100c6f07
	s_delay_alu instid0(valu_dep_1)                            // 000000004320: bf870001
	v_cmp_class_f32_e64 s3, v6, 0x198                          // 000000004324: d47e0003 0201ff06 00000198
	v_mul_f32_e32 v6, v45, v6                                  // 000000004330: 100c0d2d
	s_xor_b32 s3, s3, -1                                       // 000000004334: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004338: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000433c: be842003
	s_cbranch_execnz 2235                                      // 000000004340: bfa608bb <packed_folded_w4a8+0x4b30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004344: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004348: 8c7e047e
	v_mul_f32_e32 v7, v0, v55                                  // 00000000434c: 100e6f00
	s_delay_alu instid0(valu_dep_1)                            // 000000004350: bf870001
	v_cmp_class_f32_e64 s3, v7, 0x198                          // 000000004354: d47e0003 0201ff07 00000198
	v_mul_f32_e32 v7, v46, v7                                  // 000000004360: 100e0f2e
	s_xor_b32 s3, s3, -1                                       // 000000004364: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004368: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000436c: be842003
	s_cbranch_execnz 2241                                      // 000000004370: bfa608c1 <packed_folded_w4a8+0x4b78>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004374: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004378: 8c7e047e
	v_mul_f32_e32 v0, v1, v55                                  // 00000000437c: 10006f01
	s_delay_alu instid0(valu_dep_1)                            // 000000004380: bf870001
	v_cmp_class_f32_e64 s3, v0, 0x198                          // 000000004384: d47e0003 0201ff00 00000198
	v_mul_f32_e32 v0, v47, v0                                  // 000000004390: 1000012f
	s_xor_b32 s3, s3, -1                                       // 000000004394: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004398: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000439c: be842003
	s_cbranch_execnz 2247                                      // 0000000043a0: bfa608c7 <packed_folded_w4a8+0x4bc0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000043a8: 8c7e047e
	v_mul_f32_e32 v1, v2, v55                                  // 0000000043ac: 10026f02
	s_delay_alu instid0(valu_dep_1)                            // 0000000043b0: bf870001
	v_cmp_class_f32_e64 s3, v1, 0x198                          // 0000000043b4: d47e0003 0201ff01 00000198
	v_mul_f32_e32 v57, v48, v1                                 // 0000000043c0: 10720330
	s_xor_b32 s3, s3, -1                                       // 0000000043c4: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043c8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000043cc: be842003
	s_cbranch_execnz 2253                                      // 0000000043d0: bfa608cd <packed_folded_w4a8+0x4c08>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000043d8: 8c7e047e
	v_mul_f32_e32 v1, v3, v55                                  // 0000000043dc: 10026f03
	s_delay_alu instid0(valu_dep_1)                            // 0000000043e0: bf870001
	v_cmp_class_f32_e64 s3, v1, 0x198                          // 0000000043e4: d47e0003 0201ff01 00000198
	v_mul_f32_e32 v1, v49, v1                                  // 0000000043f0: 10020331
	s_xor_b32 s3, s3, -1                                       // 0000000043f4: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043f8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000043fc: be842003
	s_cbranch_execnz 2259                                      // 000000004400: bfa608d3 <packed_folded_w4a8+0x4c50>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004404: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004408: 8c7e047e
	v_mul_lo_u32 v42, s19, v134                                // 00000000440c: d72c002a 02030c13
	v_mul_lo_u32 v43, s18, v135                                // 000000004414: d72c002b 02030e12
	v_mad_co_u64_u32 v[2:3], null, s18, v134, 0                // 00000000441c: d6fe7c02 02030c12
	v_bfe_u32 v44, v53, 16, 1                                  // 000000004424: d610002c 02052135
	v_cmp_u_f32_e64 s3, v53, v53                               // 00000000442c: d4180003 02026b35
	v_or_b32_e32 v45, 0x400000, v4                             // 000000004434: 385a08ff 00400000
	v_bfe_u32 v48, v5, 16, 1                                   // 00000000443c: d6100030 02052105
	v_or_b32_e32 v76, v76, v162                                // 000000004444: 3899454c
	v_add3_u32 v44, v44, v53, 0x7fff                           // 000000004448: d655002c 03fe6b2c 00007fff
	v_cmp_u_f32_e64 s4, v1, v1                                 // 000000004454: d4180004 02020301
	v_add3_u32 v3, v3, v43, v42                                // 00000000445c: d6550003 04aa5703
	v_or_b32_e32 v42, 0x400000, v53                            // 000000004464: 38546aff 00400000
	v_bfe_u32 v43, v4, 16, 1                                   // 00000000446c: d610002b 02052104
	v_or_b32_e32 v53, 0x400000, v0                             // 000000004474: 386a00ff 00400000
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_3)// 00000000447c: bf8701c4
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004480: 3e040481
	s_wait_alu depctr_va_sdst(0)                               // 000000004484: bf88f19f
	v_cndmask_b32_e64 v42, v44, v42, s3                        // 000000004488: d501002a 000e552c
	v_add3_u32 v43, v43, v4, 0x7fff                            // 000000004490: d655002b 03fe092b 00007fff
	v_add_co_u32 v2, s3, s8, v2                                // 00000000449c: d7000302 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000044a4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s9, v3, s3                   // 0000000044a8: d5207c03 000e0609
	v_cmp_u_f32_e64 s3, v4, v4                                 // 0000000044b0: d4180003 02020904
	s_wait_alu depctr_va_sdst(0)                               // 0000000044b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000044bc: bf870001
	v_cndmask_b32_e64 v4, v43, v45, s3                         // 0000000044c0: d5010004 000e5b2b
	v_add_co_u32 v43, s3, v2, s22                              // 0000000044c8: d700032b 02002d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000044d0: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s23, v3, s3                 // 0000000044d4: d5207c31 000e0617
	v_add_co_u32 v46, s3, v2, v160                             // 0000000044dc: d700032e 02034102
	s_wait_alu depctr_va_sdst(0)                               // 0000000044e4: bf88f19f
	v_add_co_ci_u32_e64 v47, null, v3, v161, s3                // 0000000044e8: d5207c2f 000f4303
	v_add_co_u32 v44, s3, v43, v160                            // 0000000044f0: d700032c 0203412b
	s_wait_alu depctr_va_sdst(0)                               // 0000000044f8: bf88f19f
	v_add_co_ci_u32_e64 v45, null, v49, v161, s3               // 0000000044fc: d5207c2d 000f4331
	v_add3_u32 v2, v48, v5, 0x7fff                             // 000000004504: d6550002 03fe0b30 00007fff
	v_or_b32_e32 v3, 0x400000, v5                              // 000000004510: 38060aff 00400000
	v_cmp_u_f32_e64 s3, v5, v5                                 // 000000004518: d4180003 02020b05
	s_clause 0x1                                               // 000000004520: bf850001
	global_store_d16_hi_b16 v[46:47], v42, off                 // 000000004524: ee09407c 15000000 0000002e
	global_store_d16_hi_b16 v[44:45], v4, off                  // 000000004530: ee09407c 02000000 0000002c
	v_or_b32_e32 v42, 0x400000, v6                             // 00000000453c: 38540cff 00400000
	v_or_b32_e32 v48, 0x400000, v7                             // 000000004544: 38600eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000454c: bf88f19f
	v_cndmask_b32_e64 v2, v2, v3, s3                           // 000000004550: d5010002 000e0702
	v_add_co_u32 v4, s3, v43, s22                              // 000000004558: d7000304 02002d2b
	v_bfe_u32 v3, v6, 16, 1                                    // 000000004560: d6100003 02052106
	s_wait_alu depctr_va_sdst(0)                               // 000000004568: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v49, s3                 // 00000000456c: d5207c05 000e6217
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004574: bf870193
	v_add_co_u32 v154, s3, v4, v160                            // 000000004578: d700039a 02034104
	v_add3_u32 v3, v3, v6, 0x7fff                              // 000000004580: d6550003 03fe0d03 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000458c: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000004590: bf870143
	v_add_co_ci_u32_e64 v155, null, v5, v161, s3               // 000000004594: d5207c9b 000f4305
	v_cmp_u_f32_e64 s3, v6, v6                                 // 00000000459c: d4180003 02020d06
	v_bfe_u32 v6, v7, 16, 1                                    // 0000000045a4: d6100006 02052107
	s_wait_alu depctr_va_sdst(0)                               // 0000000045ac: bf88f19f
	v_cndmask_b32_e64 v3, v3, v42, s3                          // 0000000045b0: d5010003 000e5503
	v_add_co_u32 v4, s3, v4, s22                               // 0000000045b8: d7000304 02002d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000045c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s3                  // 0000000045c4: d5207c05 000e0a17
	v_add3_u32 v6, v6, v7, 0x7fff                              // 0000000045cc: d6550006 03fe0f06 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045d8: bf8701a3
	v_add_co_u32 v42, s3, v4, v160                             // 0000000045dc: d700032a 02034104
	s_wait_alu depctr_va_sdst(0)                               // 0000000045e4: bf88f19f
	v_add_co_ci_u32_e64 v43, null, v5, v161, s3                // 0000000045e8: d5207c2b 000f4305
	v_cmp_u_f32_e64 s3, v7, v7                                 // 0000000045f0: d4180003 02020f07
	v_bfe_u32 v7, v0, 16, 1                                    // 0000000045f8: d6100007 02052100
	s_wait_alu depctr_va_sdst(0)                               // 000000004600: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_3)// 000000004604: bf8701d2
	v_cndmask_b32_e64 v6, v6, v48, s3                          // 000000004608: d5010006 000e6106
	v_add_co_u32 v4, s3, v4, s22                               // 000000004610: d7000304 02002d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004618: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s3                  // 00000000461c: d5207c05 000e0a17
	v_add3_u32 v7, v7, v0, 0x7fff                              // 000000004624: d6550007 03fe0107 00007fff
	v_add_co_u32 v48, s3, v4, v160                             // 000000004630: d7000330 02034104
	s_wait_alu depctr_va_sdst(0)                               // 000000004638: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 00000000463c: bf870003
	v_add_co_ci_u32_e64 v49, null, v5, v161, s3                // 000000004640: d5207c31 000f4305
	v_cmp_u_f32_e64 s3, v0, v0                                 // 000000004648: d4180003 02020100
	s_clause 0x2                                               // 000000004650: bf850002
	global_store_d16_hi_b16 v[154:155], v2, off                // 000000004654: ee09407c 01000000 0000009a
	global_store_d16_hi_b16 v[42:43], v3, off                  // 000000004660: ee09407c 01800000 0000002a
	global_store_d16_hi_b16 v[48:49], v6, off                  // 00000000466c: ee09407c 03000000 00000030
	v_bfe_u32 v2, v57, 16, 1                                   // 000000004678: d6100002 02052139
	v_or_b32_e32 v6, 0x400000, v1                              // 000000004680: 380c02ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004688: bf88f19f
	v_cndmask_b32_e64 v0, v7, v53, s3                          // 00000000468c: d5010000 000e6b07
	v_add_co_u32 v3, s3, v4, s22                               // 000000004694: d7000303 02002d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000469c: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s23, v5, s3                  // 0000000046a0: d5207c04 000e0a17
	v_add3_u32 v2, v2, v57, 0x7fff                             // 0000000046a8: d6550002 03fe7302 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000046b4: bf870003
	v_add_co_u32 v156, s3, v3, v160                            // 0000000046b8: d700039c 02034103
	v_or_b32_e32 v5, 0x400000, v57                             // 0000000046c0: 380a72ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000046c8: bf88f19f
	v_add_co_ci_u32_e64 v157, null, v4, v161, s3               // 0000000046cc: d5207c9d 000f4304
	v_cmp_u_f32_e64 s3, v57, v57                               // 0000000046d4: d4180003 02027339
	global_store_d16_hi_b16 v[156:157], v0, off                // 0000000046dc: ee09407c 00000000 0000009c
	s_wait_alu depctr_va_sdst(0)                               // 0000000046e8: bf88f19f
	v_cndmask_b32_e64 v2, v2, v5, s3                           // 0000000046ec: d5010002 000e0b02
	v_add_co_u32 v3, s3, v3, s22                               // 0000000046f4: d7000303 02002d03
	s_wait_alu depctr_va_sdst(0)                               // 0000000046fc: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s23, v4, s3                  // 000000004700: d5207c04 000e0817
	v_bfe_u32 v5, v1, 16, 1                                    // 000000004708: d6100005 02052101
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004710: bf8701a3
	v_add_co_u32 v158, s3, v3, v160                            // 000000004714: d700039e 02034103
	s_wait_alu depctr_va_sdst(0)                               // 00000000471c: bf88f19f
	v_add_co_ci_u32_e64 v159, null, v4, v161, s3               // 000000004720: d5207c9f 000f4304
	v_add_co_u32 v3, s3, v3, s22                               // 000000004728: d7000303 02002d03
	v_add3_u32 v5, v5, v1, 0x7fff                              // 000000004730: d6550005 03fe0305 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 00000000473c: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s23, v4, s3                  // 000000004740: d5207c04 000e0817
	v_cmp_gt_u64_e64 s3, s[18:19], v[76:77]                    // 000000004748: d45c0003 02029812
	global_store_d16_hi_b16 v[158:159], v2, off                // 000000004750: ee09407c 01000000 0000009e
	v_cndmask_b32_e64 v1, v5, v6, s4                           // 00000000475c: d5010001 00120d05
	v_add_co_u32 v160, s4, v3, v160                            // 000000004764: d70004a0 02034103
	s_wait_alu depctr_va_sdst(0)                               // 00000000476c: bf88f19f
	v_add_co_ci_u32_e64 v161, null, v4, v161, s4               // 000000004770: d5207ca1 00134304
	s_and_b32 s4, vcc_lo, s3                                   // 000000004778: 8b04036a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000477c: bf88ff9e
	s_and_b32 s4, s10, s4                                      // 000000004780: 8b04040a
	global_store_d16_hi_b16 v[160:161], v1, off                // 000000004784: ee09407c 00800000 000000a0
	s_wait_alu depctr_sa_sdst(0)                               // 000000004790: bf88ff9e
	s_xor_b32 s4, s4, -1                                       // 000000004794: 8d04c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004798: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 00000000479c: be852004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047a0: bf88ff9e
	s_xor_b32 s8, exec_lo, s5                                  // 0000000047a4: 8d08057e
	s_cbranch_execz 124                                        // 0000000047a8: bfa5007c <packed_folded_w4a8+0x2e9c>
	v_or_b32_e32 v0, v164, v78                                 // 0000000047ac: 38009da4
	v_mov_b32_e32 v1, v79                                      // 0000000047b0: 7e02034f
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[80:81]                // 0000000047b4: 7ca8a010
	v_or_b32_e32 v2, v163, v78                                 // 0000000047b8: 38049da3
	v_mov_b32_e32 v3, v79                                      // 0000000047bc: 7e06034f
	v_or_b32_e32 v4, v167, v78                                 // 0000000047c0: 38089da7
	v_cmp_gt_i64_e64 s4, s[16:17], v[0:1]                      // 0000000047c4: d4540004 02020010
	v_mov_b32_e32 v5, v79                                      // 0000000047cc: 7e0a034f
	s_wait_alu depctr_va_vcc(0)                                // 0000000047d0: bf88ff9d
	v_dual_cndmask_b32 v81, 0, v81 :: v_dual_cndmask_b32 v80, 0, v80// 0000000047d4: ca52a280 5150a080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 0000000047dc: 7ca80410
	v_or_b32_e32 v6, v165, v78                                 // 0000000047e0: 380c9da5
	s_wait_alu depctr_va_sdst(0)                               // 0000000047e4: bf88f19f
	v_cndmask_b32_e64 v1, 0, v1, s4                            // 0000000047e8: d5010001 00120280
	v_cndmask_b32_e64 v0, 0, v0, s4                            // 0000000047f0: d5010000 00120080
	v_lshlrev_b64_e32 v[80:81], 2, v[80:81]                    // 0000000047f8: 3ea0a082
	v_mov_b32_e32 v7, v79                                      // 0000000047fc: 7e0e034f
	s_wait_alu depctr_va_vcc(0)                                // 000000004800: bf88ff9d
	v_dual_cndmask_b32 v3, 0, v3 :: v_dual_cndmask_b32 v2, 0, v2// 000000004804: ca520680 03020480
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 00000000480c: 3e000082
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000004810: 7ca80810
	v_or_b32_e32 v117, v168, v78                               // 000000004814: 38ea9da8
	v_mov_b32_e32 v118, v79                                    // 000000004818: 7eec034f
	v_add_co_u32 v80, s4, s6, v80                              // 00000000481c: d7000450 0202a006
	s_wait_alu depctr_va_sdst(0)                               // 000000004824: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s7, v81, s4                 // 000000004828: d5207c51 0012a207
	v_add_co_u32 v0, s4, s6, v0                                // 000000004830: d7000400 02020006
	s_wait_alu depctr_va_vcc(0)                                // 000000004838: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_cndmask_b32 v4, 0, v4// 00000000483c: ca520a80 05040880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000004844: 7ca80c10
	v_or_b32_e32 v162, v166, v78                               // 000000004848: 39449da6
	v_mov_b32_e32 v163, v79                                    // 00000000484c: 7f46034f
	s_wait_alu depctr_va_sdst(0)                               // 000000004850: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s4                   // 000000004854: d5207c01 00120207
	v_cmp_gt_i64_e64 s4, s[16:17], v[117:118]                  // 00000000485c: d4540004 0202ea10
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000004864: 3e080882
	s_wait_alu depctr_va_vcc(0)                                // 000000004868: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v7 :: v_dual_cndmask_b32 v6, 0, v6// 00000000486c: ca520e80 07060c80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[162:163]              // 000000004874: 7ca94410
	v_or_b32_e32 v78, v169, v78                                // 000000004878: 389c9da9
	s_wait_alu depctr_va_sdst(0)                               // 00000000487c: bf88f19f
	v_cndmask_b32_e64 v118, 0, v118, s4                        // 000000004880: d5010076 0012ec80
	v_cndmask_b32_e64 v117, 0, v117, s4                        // 000000004888: d5010075 0012ea80
	v_add_co_u32 v164, s4, s6, v4                              // 000000004890: d70004a4 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000004898: bf88f19f
	v_add_co_ci_u32_e64 v165, null, s7, v5, s4                 // 00000000489c: d5207ca5 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000048a4: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[117:118]                    // 0000000048a8: 3e0cea82
	s_wait_alu depctr_va_vcc(0)                                // 0000000048ac: bf88ff9d
	v_dual_cndmask_b32 v118, 0, v163 :: v_dual_cndmask_b32 v117, 0, v162// 0000000048b0: ca534680 76754480
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[78:79]                // 0000000048b8: 7ca89c10
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 0000000048bc: 3e040482
	v_add_co_u32 v162, s4, s6, v4                              // 0000000048c0: d70004a2 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000048c8: bf88f19f
	v_add_co_ci_u32_e64 v163, null, s7, v5, s4                 // 0000000048cc: d5207ca3 00120a07
	s_wait_alu depctr_va_vcc(0)                                // 0000000048d4: bf88ff9d
	v_dual_cndmask_b32 v79, 0, v79 :: v_dual_cndmask_b32 v78, 0, v78// 0000000048d8: ca529e80 4f4e9c80
	v_lshlrev_b64_e32 v[4:5], 2, v[117:118]                    // 0000000048e0: 3e08ea82
	v_add_co_u32 v117, vcc_lo, s6, v6                          // 0000000048e4: d7006a75 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 0000000048ec: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, s7, v7, vcc_lo             // 0000000048f0: d5207c76 01aa0e07
	v_lshlrev_b64_e32 v[6:7], 2, v[78:79]                      // 0000000048f8: 3e0c9c82
	v_add_co_u32 v2, s5, s6, v2                                // 0000000048fc: d7000502 02020406
	v_add_co_u32 v78, vcc_lo, s6, v4                           // 000000004904: d7006a4e 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000490c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s5                   // 000000004910: d5207c03 00160607
	s_wait_alu depctr_va_vcc(0)                                // 000000004918: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, s7, v5, vcc_lo              // 00000000491c: d5207c4f 01aa0a07
	v_add_co_u32 v166, vcc_lo, s6, v6                          // 000000004924: d7006aa6 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 00000000492c: bf88ff9d
	v_add_co_ci_u32_e64 v167, null, s7, v7, vcc_lo             // 000000004930: d5207ca7 01aa0e07
	s_clause 0x7                                               // 000000004938: bf850007
	global_load_b32 v4, v[80:81], off                          // 00000000493c: ee05007c 00000004 00000050
	global_load_b32 v5, v[0:1], off                            // 000000004948: ee05007c 00000005 00000000
	global_load_b32 v6, v[2:3], off                            // 000000004954: ee05007c 00000006 00000002
	global_load_b32 v7, v[164:165], off                        // 000000004960: ee05007c 00000007 000000a4
	global_load_b32 v0, v[162:163], off                        // 00000000496c: ee05007c 00000000 000000a2
	global_load_b32 v1, v[117:118], off                        // 000000004978: ee05007c 00000001 00000075
	global_load_b32 v2, v[78:79], off                          // 000000004984: ee05007c 00000002 0000004e
	global_load_b32 v3, v[166:167], off                        // 000000004990: ee05007c 00000003 000000a6
	s_wait_alu depctr_sa_sdst(0)                               // 00000000499c: bf88ff9e
	s_and_not1_saveexec_b32 s4, s8                             // 0000000049a0: be843008
	s_cbranch_execz 14                                         // 0000000049a4: bfa5000e <packed_folded_w4a8+0x2ee0>
	s_wait_loadcnt 0x3                                         // 0000000049a8: bfc00003
	v_add_co_u32 v0, vcc_lo, s6, v118                          // 0000000049ac: d7006a00 0202ec06
	s_wait_loadcnt 0x2                                         // 0000000049b4: bfc00002
	s_wait_alu depctr_va_vcc(0)                                // 0000000049b8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s7, v119, vcc_lo             // 0000000049bc: d5207c01 01aaee07
	global_load_b128 v[4:7], v[0:1], off                       // 0000000049c4: ee05c07c 00000004 00000000
	s_wait_loadcnt 0x1                                         // 0000000049d0: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:16             // 0000000049d4: ee05c07c 00000000 00001000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000049e4: 8c7e047e
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[76:77]                // 0000000049e8: 7ca89812
	s_wait_alu depctr_va_vcc(0)                                // 0000000049ec: bf88ff9d
	v_cndmask_b32_e32 v55, 0, v76, vcc_lo                      // 0000000049f0: 026e9880
	v_cndmask_b32_e32 v53, 0, v77, vcc_lo                      // 0000000049f4: 026a9a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000049f8: bf870122
	v_add_co_u32 v77, vcc_lo, s20, v55                         // 0000000049fc: d7006a4d 02026e14
	s_wait_alu depctr_va_vcc(0)                                // 000000004a04: bf88ff9d
	v_add_co_ci_u32_e64 v78, null, s21, v53, vcc_lo            // 000000004a08: d5207c4e 01aa6a15
	global_load_u8 v53, v[77:78], off                          // 000000004a10: ee04007c 00000035 0000004d
	s_wait_loadcnt 0x0                                         // 000000004a1c: bfc00000
	v_lshlrev_b32_e32 v55, 23, v53                             // 000000004a20: 306e6a97
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004a24: bf870091
	v_mul_f32_e32 v53, v4, v55                                 // 000000004a28: 106a6f04
	v_cmp_class_f32_e64 s4, v53, 0x198                         // 000000004a2c: d47e0004 0201ff35 00000198
	v_mul_f32_e32 v53, v34, v53                                // 000000004a38: 106a6b22
	s_xor_b32 s5, s4, -1                                       // 000000004a3c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a40: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004a44: be842005
	s_cbranch_execnz 1875                                      // 000000004a48: bfa60753 <packed_folded_w4a8+0x4c98>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a50: 8c7e047e
	v_mul_f32_e32 v4, v5, v55                                  // 000000004a54: 10086f05
	s_delay_alu instid0(valu_dep_1)                            // 000000004a58: bf870001
	v_cmp_class_f32_e64 s4, v4, 0x198                          // 000000004a5c: d47e0004 0201ff04 00000198
	v_mul_f32_e32 v4, v35, v4                                  // 000000004a68: 10080923
	s_xor_b32 s5, s4, -1                                       // 000000004a6c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a70: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004a74: be842005
	s_cbranch_execnz 1880                                      // 000000004a78: bfa60758 <packed_folded_w4a8+0x4cdc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a80: 8c7e047e
	v_mul_f32_e32 v5, v6, v55                                  // 000000004a84: 100a6f06
	s_delay_alu instid0(valu_dep_1)                            // 000000004a88: bf870001
	v_cmp_class_f32_e64 s4, v5, 0x198                          // 000000004a8c: d47e0004 0201ff05 00000198
	v_mul_f32_e32 v5, v36, v5                                  // 000000004a98: 100a0b24
	s_xor_b32 s5, s4, -1                                       // 000000004a9c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004aa0: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004aa4: be842005
	s_cbranch_execnz 1885                                      // 000000004aa8: bfa6075d <packed_folded_w4a8+0x4d20>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004aac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004ab0: 8c7e047e
	v_mul_f32_e32 v6, v7, v55                                  // 000000004ab4: 100c6f07
	s_delay_alu instid0(valu_dep_1)                            // 000000004ab8: bf870001
	v_cmp_class_f32_e64 s4, v6, 0x198                          // 000000004abc: d47e0004 0201ff06 00000198
	v_mul_f32_e32 v6, v37, v6                                  // 000000004ac8: 100c0d25
	s_xor_b32 s5, s4, -1                                       // 000000004acc: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ad0: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004ad4: be842005
	s_cbranch_execnz 1890                                      // 000000004ad8: bfa60762 <packed_folded_w4a8+0x4d64>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004adc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004ae0: 8c7e047e
	v_mul_f32_e32 v7, v0, v55                                  // 000000004ae4: 100e6f00
	s_delay_alu instid0(valu_dep_1)                            // 000000004ae8: bf870001
	v_cmp_class_f32_e64 s4, v7, 0x198                          // 000000004aec: d47e0004 0201ff07 00000198
	v_mul_f32_e32 v7, v38, v7                                  // 000000004af8: 100e0f26
	s_xor_b32 s5, s4, -1                                       // 000000004afc: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b00: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004b04: be842005
	s_cbranch_execnz 1895                                      // 000000004b08: bfa60767 <packed_folded_w4a8+0x4da8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b10: 8c7e047e
	v_mul_f32_e32 v0, v1, v55                                  // 000000004b14: 10006f01
	s_delay_alu instid0(valu_dep_1)                            // 000000004b18: bf870001
	v_cmp_class_f32_e64 s4, v0, 0x198                          // 000000004b1c: d47e0004 0201ff00 00000198
	v_mul_f32_e32 v0, v39, v0                                  // 000000004b28: 10000127
	s_xor_b32 s5, s4, -1                                       // 000000004b2c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b30: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004b34: be842005
	s_cbranch_execnz 1900                                      // 000000004b38: bfa6076c <packed_folded_w4a8+0x4dec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b40: 8c7e047e
	v_mul_f32_e32 v1, v2, v55                                  // 000000004b44: 10026f02
	s_delay_alu instid0(valu_dep_1)                            // 000000004b48: bf870001
	v_cmp_class_f32_e64 s4, v1, 0x198                          // 000000004b4c: d47e0004 0201ff01 00000198
	v_mul_f32_e32 v1, v40, v1                                  // 000000004b58: 10020328
	s_xor_b32 s5, s4, -1                                       // 000000004b5c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b60: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004b64: be842005
	s_cbranch_execnz 1905                                      // 000000004b68: bfa60771 <packed_folded_w4a8+0x4e30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b70: 8c7e047e
	v_mul_f32_e32 v2, v3, v55                                  // 000000004b74: 10046f03
	s_delay_alu instid0(valu_dep_1)                            // 000000004b78: bf870001
	v_cmp_class_f32_e64 s4, v2, 0x198                          // 000000004b7c: d47e0004 0201ff02 00000198
	v_mul_f32_e32 v2, v41, v2                                  // 000000004b88: 10040529
	s_xor_b32 s5, s4, -1                                       // 000000004b8c: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b90: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004b94: be842005
	s_cbranch_execnz 1910                                      // 000000004b98: bfa60776 <packed_folded_w4a8+0x4e74>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004ba0: 8c7e047e
	v_bfe_u32 v3, v53, 16, 1                                   // 000000004ba4: d6100003 02052135
	v_bfe_u32 v34, v4, 16, 1                                   // 000000004bac: d6100022 02052104
	v_or_b32_e32 v35, 0x400000, v53                            // 000000004bb4: 38466aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 000000004bbc: 7c306b35
	v_or_b32_e32 v36, 0x400000, v4                             // 000000004bc0: 384808ff 00400000
	v_add3_u32 v3, v3, v53, 0x7fff                             // 000000004bc8: d6550003 03fe6b03 00007fff
	v_add3_u32 v34, v34, v4, 0x7fff                            // 000000004bd4: d6550022 03fe0922 00007fff
	v_bfe_u32 v37, v5, 16, 1                                   // 000000004be0: d6100025 02052105
	s_and_b32 s0, s0, s3                                       // 000000004be8: 8b000300
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bec: bf88ff9e
	s_and_b32 s0, s10, s0                                      // 000000004bf0: 8b00000a
	s_wait_alu depctr_va_vcc(0)                                // 000000004bf4: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v35, vcc_lo                      // 000000004bf8: 02064703
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 000000004bfc: 7c300904
	v_bfe_u32 v35, v7, 16, 1                                   // 000000004c00: d6100023 02052107
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c08: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 000000004c0c: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 000000004c10: bf88ff9d
	v_cndmask_b32_e32 v4, v34, v36, vcc_lo                     // 000000004c14: 02084922
	v_bfe_u32 v34, v6, 16, 1                                   // 000000004c18: d6100022 02052106
	v_add3_u32 v36, v37, v5, 0x7fff                            // 000000004c20: d6550024 03fe0b25 00007fff
	s_clause 0x1                                               // 000000004c2c: bf850001
	global_store_d16_hi_b16 v[72:73], v3, off offset:32        // 000000004c30: ee09407c 01800000 00002048
	global_store_d16_hi_b16 v[84:85], v4, off offset:32        // 000000004c3c: ee09407c 02000000 00002054
	v_or_b32_e32 v3, 0x400000, v5                              // 000000004c48: 38060aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000004c50: 7c300b05
	v_add3_u32 v4, v34, v6, 0x7fff                             // 000000004c54: d6550004 03fe0d22 00007fff
	v_or_b32_e32 v34, 0x400000, v6                             // 000000004c60: 38440cff 00400000
	v_add3_u32 v35, v35, v7, 0x7fff                            // 000000004c68: d6550023 03fe0f23 00007fff
	v_or_b32_e32 v37, 0x400000, v7                             // 000000004c74: 384a0eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c7c: bf88ff9d
	v_cndmask_b32_e32 v3, v36, v3, vcc_lo                      // 000000004c80: 02060724
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000004c84: 7c300d06
	v_or_b32_e32 v6, 0x400000, v0                              // 000000004c88: 380c00ff 00400000
	global_store_d16_hi_b16 v[90:91], v3, off offset:32        // 000000004c90: ee09407c 01800000 0000205a
	s_wait_alu depctr_va_vcc(0)                                // 000000004c9c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v34, vcc_lo                      // 000000004ca0: 02084504
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 000000004ca4: 7c300f07
	v_bfe_u32 v3, v0, 16, 1                                    // 000000004ca8: d6100003 02052100
	v_or_b32_e32 v7, 0x400000, v1                              // 000000004cb0: 380e02ff 00400000
	v_or_b32_e32 v34, 0x400000, v2                             // 000000004cb8: 384404ff 00400000
	global_store_d16_hi_b16 v[92:93], v4, off offset:32        // 000000004cc0: ee09407c 02000000 0000205c
	s_wait_alu depctr_va_vcc(0)                                // 000000004ccc: bf88ff9d
	v_cndmask_b32_e32 v5, v35, v37, vcc_lo                     // 000000004cd0: 020a4b23
	v_bfe_u32 v4, v1, 16, 1                                    // 000000004cd4: d6100004 02052101
	v_add3_u32 v3, v3, v0, 0x7fff                              // 000000004cdc: d6550003 03fe0103 00007fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 000000004ce8: 7c300100
	global_store_d16_hi_b16 v[98:99], v5, off offset:32        // 000000004cec: ee09407c 02800000 00002062
	v_bfe_u32 v5, v2, 16, 1                                    // 000000004cf8: d6100005 02052102
	v_add3_u32 v4, v4, v1, 0x7fff                              // 000000004d00: d6550004 03fe0304 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004d0c: bf88ff9d
	v_cndmask_b32_e32 v0, v3, v6, vcc_lo                       // 000000004d10: 02000d03
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000004d14: 7c300301
	v_add3_u32 v5, v5, v2, 0x7fff                              // 000000004d18: d6550005 03fe0505 00007fff
	global_store_d16_hi_b16 v[96:97], v0, off offset:32        // 000000004d24: ee09407c 00000000 00002060
	s_wait_alu depctr_va_vcc(0)                                // 000000004d30: bf88ff9d
	v_cndmask_b32_e32 v1, v4, v7, vcc_lo                       // 000000004d34: 02020f04
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000004d38: 7c300502
	global_store_d16_hi_b16 v[100:101], v1, off offset:32      // 000000004d3c: ee09407c 00800000 00002064
	s_wait_alu depctr_va_vcc(0)                                // 000000004d48: bf88ff9d
	v_cndmask_b32_e32 v2, v5, v34, vcc_lo                      // 000000004d4c: 02044505
	global_store_d16_hi_b16 v[102:103], v2, off offset:32      // 000000004d50: ee09407c 01000000 00002066
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d5c: bf88ff9e
	s_and_saveexec_b32 s4, s0                                  // 000000004d60: be842000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d64: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000004d68: 8d05047e
	s_cbranch_execz 120                                        // 000000004d6c: bfa50078 <packed_folded_w4a8+0x3450>
	v_mov_b32_e32 v95, v9                                      // 000000004d70: 7ebe0309
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[86:87]                // 000000004d74: 7ca8ac10
	v_mov_b32_e32 v89, v9                                      // 000000004d78: 7eb20309
	v_mov_b32_e32 v83, v9                                      // 000000004d7c: 7ea60309
	v_mov_b32_e32 v71, v9                                      // 000000004d80: 7e8e0309
	v_cmp_gt_i64_e64 s0, s[16:17], v[94:95]                    // 000000004d84: d4540000 0202bc10
	v_mov_b32_e32 v69, v9                                      // 000000004d8c: 7e8a0309
	s_wait_alu depctr_va_vcc(0)                                // 000000004d90: bf88ff9d
	v_dual_cndmask_b32 v1, 0, v87 :: v_dual_cndmask_b32 v0, 0, v86// 000000004d94: ca52ae80 0100ac80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[88:89]                // 000000004d9c: 7ca8b010
	v_mov_b32_e32 v67, v9                                      // 000000004da0: 7e860309
	s_wait_alu depctr_va_sdst(0)                               // 000000004da4: bf88f19f
	v_cndmask_b32_e64 v3, 0, v95, s0                           // 000000004da8: d5010003 0002be80
	v_cndmask_b32_e64 v2, 0, v94, s0                           // 000000004db0: d5010002 0002bc80
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000004db8: 3e000082
	s_wait_alu depctr_va_vcc(0)                                // 000000004dbc: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v89 :: v_dual_cndmask_b32 v4, 0, v88// 000000004dc0: ca52b280 0504b080
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004dc8: bf870223
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000004dcc: 3e040482
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[82:83]                // 000000004dd0: 7ca8a410
	v_add_co_u32 v0, s0, s6, v0                                // 000000004dd4: d7000000 02020006
	s_delay_alu instid0(valu_dep_4)                            // 000000004ddc: bf870004
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000004de0: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 000000004de4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s0                   // 000000004de8: d5207c01 00020207
	v_add_co_u32 v2, s0, s6, v2                                // 000000004df0: d7000002 02020406
	s_wait_alu depctr_va_vcc(0)                                // 000000004df8: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v83 :: v_dual_cndmask_b32 v6, 0, v82// 000000004dfc: ca52a680 0706a480
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[70:71]                // 000000004e04: 7ca88c10
	s_wait_alu depctr_va_sdst(0)                               // 000000004e08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s0                   // 000000004e0c: d5207c03 00020607
	v_cmp_gt_i64_e64 s0, s[16:17], v[68:69]                    // 000000004e14: d4540000 02028810
	v_add_co_u32 v34, s4, s6, v4                               // 000000004e1c: d7000422 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000004e24: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s7, v5, s4                  // 000000004e28: d5207c23 00120a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000004e30: 3e080c82
	s_wait_alu depctr_va_vcc(0)                                // 000000004e34: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v71 :: v_dual_cndmask_b32 v6, 0, v70// 000000004e38: ca528e80 07068c80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[66:67]                // 000000004e40: 7ca88410
	v_cndmask_b32_e64 v37, 0, v69, s0                          // 000000004e44: d5010025 00028a80
	v_cndmask_b32_e64 v36, 0, v68, s0                          // 000000004e4c: d5010024 00028880
	v_add_co_u32 v38, s0, s6, v4                               // 000000004e54: d7000026 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000004e5c: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s7, v5, s0                  // 000000004e60: d5207c27 00020a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000004e68: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[36:37]                      // 000000004e6c: 3e0c4882
	s_wait_alu depctr_va_vcc(0)                                // 000000004e70: bf88ff9d
	v_dual_cndmask_b32 v37, 0, v67 :: v_dual_cndmask_b32 v36, 0, v66// 000000004e74: ca528680 25248480
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000004e7c: 7ca81010
	s_delay_alu instid0(valu_dep_4)                            // 000000004e80: bf870004
	v_add_co_u32 v40, s0, s6, v4                               // 000000004e84: d7000028 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000004e8c: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s7, v5, s0                  // 000000004e90: d5207c29 00020a07
	s_wait_alu depctr_va_vcc(0)                                // 000000004e98: bf88ff9d
	v_dual_cndmask_b32 v9, 0, v9 :: v_dual_cndmask_b32 v8, 0, v8// 000000004e9c: ca521280 09081080
	v_lshlrev_b64_e32 v[4:5], 2, v[36:37]                      // 000000004ea4: 3e084882
	v_add_co_u32 v36, vcc_lo, s6, v6                           // 000000004ea8: d7006a24 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 000000004eb0: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s7, v7, vcc_lo              // 000000004eb4: d5207c25 01aa0e07
	v_lshlrev_b64_e32 v[6:7], 2, v[8:9]                        // 000000004ebc: 3e0c1082
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 000000004ec0: bf8701b4
	v_add_co_u32 v8, vcc_lo, s6, v4                            // 000000004ec4: d7006a08 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000004ecc: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s7, v5, vcc_lo               // 000000004ed0: d5207c09 01aa0a07
	v_add_co_u32 v65, vcc_lo, s6, v6                           // 000000004ed8: d7006a41 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 000000004ee0: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, s7, v7, vcc_lo              // 000000004ee4: d5207c42 01aa0e07
	s_clause 0x7                                               // 000000004eec: bf850007
	global_load_b32 v4, v[0:1], off                            // 000000004ef0: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 000000004efc: ee05007c 00000005 00000002
	global_load_b32 v6, v[34:35], off                          // 000000004f08: ee05007c 00000006 00000022
	global_load_b32 v7, v[38:39], off                          // 000000004f14: ee05007c 00000007 00000026
	global_load_b32 v0, v[40:41], off                          // 000000004f20: ee05007c 00000000 00000028
	global_load_b32 v1, v[36:37], off                          // 000000004f2c: ee05007c 00000001 00000024
	global_load_b32 v2, v[8:9], off                            // 000000004f38: ee05007c 00000002 00000008
	global_load_b32 v3, v[65:66], off                          // 000000004f44: ee05007c 00000003 00000041
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f50: bf88ff9e
	s_and_not1_saveexec_b32 s0, s5                             // 000000004f54: be803005
	s_cbranch_execz 28                                         // 000000004f58: bfa5001c <packed_folded_w4a8+0x34cc>
	s_wait_loadcnt 0x3                                         // 000000004f5c: bfc00003
	v_add_co_u32 v0, s4, v74, s14                              // 000000004f60: d7000400 02001d4a
	s_wait_loadcnt 0x2                                         // 000000004f68: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000004f6c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s4                   // 000000004f70: d5207c01 00101e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000004f78: bf870122
	v_add_co_u32 v0, vcc_lo, v0, v75                           // 000000004f7c: d7006a00 02029700
	s_wait_alu depctr_va_vcc(0)                                // 000000004f84: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo                // 000000004f88: d5207c01 01aa0280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004f90: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000004f94: 3e000082
	v_add_co_u32 v0, vcc_lo, s6, v0                            // 000000004f98: d7006a00 02020006
	s_wait_alu depctr_va_vcc(0)                                // 000000004fa0: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000004fa4: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, vcc_lo               // 000000004fa8: d5207c01 01aa0207
	global_load_b128 v[4:7], v[0:1], off offset:64             // 000000004fb0: ee05c07c 00000004 00004000
	s_wait_loadcnt 0x1                                         // 000000004fbc: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:80             // 000000004fc0: ee05c07c 00000000 00005000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fcc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004fd0: 8c7e007e
	global_load_u8 v8, v[77:78], off                           // 000000004fd4: ee04007c 00000008 0000004d
	s_wait_loadcnt 0x0                                         // 000000004fe0: bfc00000
	v_lshlrev_b32_e32 v9, 23, v8                               // 000000004fe4: 30121097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004fe8: bf870091
	v_mul_f32_e32 v8, v4, v9                                   // 000000004fec: 10101304
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000004ff0: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v26, v8                                  // 000000004ffc: 1010111a
	s_xor_b32 s4, s0, -1                                       // 000000005000: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005004: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005008: be802004
	s_cbranch_execnz 1642                                      // 00000000500c: bfa6066a <packed_folded_w4a8+0x4eb8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005010: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005014: 8c7e007e
	v_mul_f32_e32 v4, v5, v9                                   // 000000005018: 10081305
	s_delay_alu instid0(valu_dep_1)                            // 00000000501c: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 000000005020: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v27, v4                                  // 00000000502c: 1008091b
	s_xor_b32 s4, s0, -1                                       // 000000005030: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005034: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005038: be802004
	s_cbranch_execnz 1647                                      // 00000000503c: bfa6066f <packed_folded_w4a8+0x4efc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005040: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005044: 8c7e007e
	v_mul_f32_e32 v5, v6, v9                                   // 000000005048: 100a1306
	s_delay_alu instid0(valu_dep_1)                            // 00000000504c: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 000000005050: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v28, v5                                  // 00000000505c: 100a0b1c
	s_xor_b32 s4, s0, -1                                       // 000000005060: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005064: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005068: be802004
	s_cbranch_execnz 1652                                      // 00000000506c: bfa60674 <packed_folded_w4a8+0x4f40>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005070: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005074: 8c7e007e
	v_mul_f32_e32 v6, v7, v9                                   // 000000005078: 100c1307
	s_delay_alu instid0(valu_dep_1)                            // 00000000507c: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000005080: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v29, v6                                  // 00000000508c: 100c0d1d
	s_xor_b32 s4, s0, -1                                       // 000000005090: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005094: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005098: be802004
	s_cbranch_execnz 1657                                      // 00000000509c: bfa60679 <packed_folded_w4a8+0x4f84>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050a4: 8c7e007e
	v_mul_f32_e32 v7, v0, v9                                   // 0000000050a8: 100e1300
	s_delay_alu instid0(valu_dep_1)                            // 0000000050ac: bf870001
	v_cmp_class_f32_e64 s0, v7, 0x198                          // 0000000050b0: d47e0000 0201ff07 00000198
	v_mul_f32_e32 v7, v30, v7                                  // 0000000050bc: 100e0f1e
	s_xor_b32 s4, s0, -1                                       // 0000000050c0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050c4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 0000000050c8: be802004
	s_cbranch_execnz 1662                                      // 0000000050cc: bfa6067e <packed_folded_w4a8+0x4fc8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050d4: 8c7e007e
	v_mul_f32_e32 v0, v1, v9                                   // 0000000050d8: 10001301
	s_delay_alu instid0(valu_dep_1)                            // 0000000050dc: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 0000000050e0: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v31, v0                                  // 0000000050ec: 1000011f
	s_xor_b32 s4, s0, -1                                       // 0000000050f0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050f4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 0000000050f8: be802004
	s_cbranch_execnz 1667                                      // 0000000050fc: bfa60683 <packed_folded_w4a8+0x500c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005100: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005104: 8c7e007e
	v_mul_f32_e32 v1, v2, v9                                   // 000000005108: 10021302
	s_delay_alu instid0(valu_dep_1)                            // 00000000510c: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 000000005110: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v32, v1                                  // 00000000511c: 10020320
	s_xor_b32 s4, s0, -1                                       // 000000005120: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005124: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005128: be802004
	s_cbranch_execnz 1672                                      // 00000000512c: bfa60688 <packed_folded_w4a8+0x5050>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005130: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005134: 8c7e007e
	v_mul_f32_e32 v2, v3, v9                                   // 000000005138: 10041303
	s_delay_alu instid0(valu_dep_1)                            // 00000000513c: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000005140: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v33, v2                                  // 00000000514c: 10040521
	s_xor_b32 s4, s0, -1                                       // 000000005150: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005154: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005158: be802004
	s_cbranch_execnz 1677                                      // 00000000515c: bfa6068d <packed_folded_w4a8+0x5094>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005160: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005164: 8c7e007e
	v_bfe_u32 v3, v8, 16, 1                                    // 000000005168: d6100003 02052108
	v_bfe_u32 v9, v4, 16, 1                                    // 000000005170: d6100009 02052104
	v_or_b32_e32 v26, 0x400000, v8                             // 000000005178: 383410ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 000000005180: 7c301108
	v_or_b32_e32 v27, 0x400000, v4                             // 000000005184: 383608ff 00400000
	v_add3_u32 v3, v3, v8, 0x7fff                              // 00000000518c: d6550003 03fe1103 00007fff
	v_add3_u32 v9, v9, v4, 0x7fff                              // 000000005198: d6550009 03fe0909 00007fff
	v_bfe_u32 v28, v5, 16, 1                                   // 0000000051a4: d610001c 02052105
	v_bfe_u32 v8, v6, 16, 1                                    // 0000000051ac: d6100008 02052106
	s_and_b32 s0, s1, s3                                       // 0000000051b4: 8b000301
	s_wait_alu depctr_va_vcc(0)                                // 0000000051b8: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v26, vcc_lo                      // 0000000051bc: 02063503
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 0000000051c0: 7c300904
	v_add3_u32 v26, v28, v5, 0x7fff                            // 0000000051c4: d655001a 03fe0b1c 00007fff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051d0: bf88ff9e
	s_and_b32 s0, s10, s0                                      // 0000000051d4: 8b00000a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051d8: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 0000000051dc: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 0000000051e0: bf88ff9d
	v_cndmask_b32_e32 v4, v9, v27, vcc_lo                      // 0000000051e4: 02083709
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 0000000051e8: 7c300b05
	v_bfe_u32 v9, v7, 16, 1                                    // 0000000051ec: d6100009 02052107
	s_clause 0x1                                               // 0000000051f4: bf850001
	global_store_d16_hi_b16 v[106:107], v3, off offset:32      // 0000000051f8: ee09407c 01800000 0000206a
	global_store_d16_hi_b16 v[112:113], v4, off offset:32      // 000000005204: ee09407c 02000000 00002070
	v_or_b32_e32 v3, 0x400000, v5                              // 000000005210: 38060aff 00400000
	v_add3_u32 v4, v8, v6, 0x7fff                              // 000000005218: d6550004 03fe0d08 00007fff
	v_or_b32_e32 v8, 0x400000, v6                              // 000000005224: 38100cff 00400000
	v_add3_u32 v9, v9, v7, 0x7fff                              // 00000000522c: d6550009 03fe0f09 00007fff
	v_or_b32_e32 v27, 0x400000, v7                             // 000000005238: 38360eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005240: bf88ff9d
	v_cndmask_b32_e32 v3, v26, v3, vcc_lo                      // 000000005244: 0206071a
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000005248: 7c300d06
	v_or_b32_e32 v6, 0x400000, v0                              // 00000000524c: 380c00ff 00400000
	global_store_d16_hi_b16 v[114:115], v3, off offset:32      // 000000005254: ee09407c 01800000 00002072
	s_wait_alu depctr_va_vcc(0)                                // 000000005260: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v8, vcc_lo                       // 000000005264: 02081104
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 000000005268: 7c300f07
	v_bfe_u32 v3, v0, 16, 1                                    // 00000000526c: d6100003 02052100
	v_or_b32_e32 v7, 0x400000, v1                              // 000000005274: 380e02ff 00400000
	v_or_b32_e32 v8, 0x400000, v2                              // 00000000527c: 381004ff 00400000
	global_store_d16_hi_b16 v[120:121], v4, off offset:32      // 000000005284: ee09407c 02000000 00002078
	s_wait_alu depctr_va_vcc(0)                                // 000000005290: bf88ff9d
	v_cndmask_b32_e32 v5, v9, v27, vcc_lo                      // 000000005294: 020a3709
	v_bfe_u32 v4, v1, 16, 1                                    // 000000005298: d6100004 02052101
	v_add3_u32 v3, v3, v0, 0x7fff                              // 0000000052a0: d6550003 03fe0103 00007fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 0000000052ac: 7c300100
	global_store_d16_hi_b16 v[124:125], v5, off offset:32      // 0000000052b0: ee09407c 02800000 0000207c
	v_bfe_u32 v5, v2, 16, 1                                    // 0000000052bc: d6100005 02052102
	v_add3_u32 v4, v4, v1, 0x7fff                              // 0000000052c4: d6550004 03fe0304 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000052d0: bf88ff9d
	v_cndmask_b32_e32 v0, v3, v6, vcc_lo                       // 0000000052d4: 02000d03
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 0000000052d8: 7c300301
	v_add3_u32 v5, v5, v2, 0x7fff                              // 0000000052dc: d6550005 03fe0505 00007fff
	global_store_d16_hi_b16 v[122:123], v0, off offset:32      // 0000000052e8: ee09407c 00000000 0000207a
	s_wait_alu depctr_va_vcc(0)                                // 0000000052f4: bf88ff9d
	v_cndmask_b32_e32 v1, v4, v7, vcc_lo                       // 0000000052f8: 02020f04
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 0000000052fc: 7c300502
	global_store_d16_hi_b16 v[126:127], v1, off offset:32      // 000000005300: ee09407c 00800000 0000207e
	s_wait_alu depctr_va_vcc(0)                                // 00000000530c: bf88ff9d
	v_cndmask_b32_e32 v2, v5, v8, vcc_lo                       // 000000005310: 02041105
	global_store_d16_hi_b16 v[128:129], v2, off offset:32      // 000000005314: ee09407c 01000000 00002080
	s_wait_alu depctr_sa_sdst(0)                               // 000000005320: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005324: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005328: bf88ff9e
	s_xor_b32 s4, exec_lo, s1                                  // 00000000532c: 8d04017e
	s_cbranch_execz 119                                        // 000000005330: bfa50077 <packed_folded_w4a8+0x3a10>
	v_mov_b32_e32 v117, v59                                    // 000000005334: 7eea033b
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[108:109]              // 000000005338: 7ca8d810
	v_mov_b32_e32 v111, v59                                    // 00000000533c: 7ede033b
	v_mov_b32_e32 v105, v59                                    // 000000005340: 7ed2033b
	v_mov_b32_e32 v65, v59                                     // 000000005344: 7e82033b
	v_cmp_gt_i64_e64 s0, s[16:17], v[116:117]                  // 000000005348: d4540000 0202e810
	v_mov_b32_e32 v63, v59                                     // 000000005350: 7e7e033b
	s_wait_alu depctr_va_vcc(0)                                // 000000005354: bf88ff9d
	v_dual_cndmask_b32 v1, 0, v109 :: v_dual_cndmask_b32 v0, 0, v108// 000000005358: ca52da80 0100d880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[110:111]              // 000000005360: 7ca8dc10
	v_mov_b32_e32 v61, v59                                     // 000000005364: 7e7a033b
	s_wait_alu depctr_va_sdst(0)                               // 000000005368: bf88f19f
	v_cndmask_b32_e64 v3, 0, v117, s0                          // 00000000536c: d5010003 0002ea80
	v_cndmask_b32_e64 v2, 0, v116, s0                          // 000000005374: d5010002 0002e880
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 00000000537c: 3e000082
	s_wait_alu depctr_va_vcc(0)                                // 000000005380: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v111 :: v_dual_cndmask_b32 v4, 0, v110// 000000005384: ca52de80 0504dc80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[104:105]              // 00000000538c: 7ca8d010
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000005390: 3e040482
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000005394: bf870214
	v_add_co_u32 v0, s0, s6, v0                                // 000000005398: d7000000 02020006
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 0000000053a0: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 0000000053a4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s0                   // 0000000053a8: d5207c01 00020207
	s_wait_alu depctr_va_vcc(0)                                // 0000000053b0: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v105 :: v_dual_cndmask_b32 v6, 0, v104// 0000000053b4: ca52d280 0706d080
	v_add_co_u32 v2, s0, s6, v2                                // 0000000053bc: d7000002 02020406
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[64:65]                // 0000000053c4: 7ca88010
	v_add_co_u32 v8, s1, s6, v4                                // 0000000053c8: d7000108 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000053d0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s0                   // 0000000053d4: d5207c03 00020607
	v_cmp_gt_i64_e64 s0, s[16:17], v[62:63]                    // 0000000053dc: d4540000 02027c10
	v_add_co_ci_u32_e64 v9, null, s7, v5, s1                   // 0000000053e4: d5207c09 00060a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000053ec: 3e080c82
	s_wait_alu depctr_va_vcc(0)                                // 0000000053f0: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v65 :: v_dual_cndmask_b32 v6, 0, v64// 0000000053f4: ca528280 07068080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[60:61]                // 0000000053fc: 7ca87810
	s_wait_alu depctr_va_sdst(0)                               // 000000005400: bf88f19f
	v_cndmask_b32_e64 v27, 0, v63, s0                          // 000000005404: d501001b 00027e80
	v_cndmask_b32_e64 v26, 0, v62, s0                          // 00000000540c: d501001a 00027c80
	v_add_co_u32 v28, s0, s6, v4                               // 000000005414: d700001c 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000541c: bf88f19f
	v_add_co_ci_u32_e64 v29, null, s7, v5, s0                  // 000000005420: d5207c1d 00020a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000005428: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[26:27]                      // 00000000542c: 3e0c3482
	s_wait_alu depctr_va_vcc(0)                                // 000000005430: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v61 :: v_dual_cndmask_b32 v26, 0, v60// 000000005434: ca527a80 1b1a7880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[58:59]                // 00000000543c: 7ca87410
	s_delay_alu instid0(valu_dep_4)                            // 000000005440: bf870004
	v_add_co_u32 v30, s0, s6, v4                               // 000000005444: d700001e 02020806
	s_wait_alu depctr_va_sdst(0)                               // 00000000544c: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s7, v5, s0                  // 000000005450: d5207c1f 00020a07
	v_lshlrev_b64_e32 v[4:5], 2, v[26:27]                      // 000000005458: 3e083482
	s_wait_alu depctr_va_vcc(0)                                // 00000000545c: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v59 :: v_dual_cndmask_b32 v26, 0, v58// 000000005460: ca527680 1b1a7480
	v_add_co_u32 v32, vcc_lo, s6, v6                           // 000000005468: d7006a20 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 000000005470: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s7, v7, vcc_lo              // 000000005474: d5207c21 01aa0e07
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 00000000547c: bf8701c3
	v_lshlrev_b64_e32 v[6:7], 2, v[26:27]                      // 000000005480: 3e0c3482
	v_add_co_u32 v26, vcc_lo, s6, v4                           // 000000005484: d7006a1a 02020806
	s_wait_alu depctr_va_vcc(0)                                // 00000000548c: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, s7, v5, vcc_lo              // 000000005490: d5207c1b 01aa0a07
	v_add_co_u32 v34, vcc_lo, s6, v6                           // 000000005498: d7006a22 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 0000000054a0: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s7, v7, vcc_lo              // 0000000054a4: d5207c23 01aa0e07
	s_clause 0x7                                               // 0000000054ac: bf850007
	global_load_b32 v4, v[0:1], off                            // 0000000054b0: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 0000000054bc: ee05007c 00000005 00000002
	global_load_b32 v6, v[8:9], off                            // 0000000054c8: ee05007c 00000006 00000008
	global_load_b32 v7, v[28:29], off                          // 0000000054d4: ee05007c 00000007 0000001c
	global_load_b32 v0, v[30:31], off                          // 0000000054e0: ee05007c 00000000 0000001e
	global_load_b32 v1, v[32:33], off                          // 0000000054ec: ee05007c 00000001 00000020
	global_load_b32 v2, v[26:27], off                          // 0000000054f8: ee05007c 00000002 0000001a
	global_load_b32 v3, v[34:35], off                          // 000000005504: ee05007c 00000003 00000022
	s_wait_alu depctr_sa_sdst(0)                               // 000000005510: bf88ff9e
	s_and_not1_saveexec_b32 s0, s4                             // 000000005514: be803004
	s_cbranch_execz 28                                         // 000000005518: bfa5001c <packed_folded_w4a8+0x3a8c>
	s_wait_loadcnt 0x3                                         // 00000000551c: bfc00003
	v_add_co_u32 v0, s1, v74, s14                              // 000000005520: d7000100 02001d4a
	s_wait_loadcnt 0x2                                         // 000000005528: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 00000000552c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s1                   // 000000005530: d5207c01 00041e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000005538: bf870122
	v_add_co_u32 v0, vcc_lo, v0, v75                           // 00000000553c: d7006a00 02029700
	s_wait_alu depctr_va_vcc(0)                                // 000000005544: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo                // 000000005548: d5207c01 01aa0280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005550: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000005554: 3e000082
	v_add_co_u32 v0, vcc_lo, s6, v0                            // 000000005558: d7006a00 02020006
	s_wait_alu depctr_va_vcc(0)                                // 000000005560: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000005564: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, vcc_lo               // 000000005568: d5207c01 01aa0207
	global_load_b128 v[4:7], v[0:1], off offset:128            // 000000005570: ee05c07c 00000004 00008000
	s_wait_loadcnt 0x1                                         // 00000000557c: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:144            // 000000005580: ee05c07c 00000000 00009000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000558c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005590: 8c7e007e
	global_load_u8 v8, v[77:78], off                           // 000000005594: ee04007c 00000008 0000004d
	s_wait_loadcnt 0x0                                         // 0000000055a0: bfc00000
	v_lshlrev_b32_e32 v9, 23, v8                               // 0000000055a4: 30121097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000055a8: bf870091
	v_mul_f32_e32 v8, v4, v9                                   // 0000000055ac: 10101304
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 0000000055b0: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v18, v8                                  // 0000000055bc: 10101112
	s_xor_b32 s1, s0, -1                                       // 0000000055c0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055c4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000055c8: be802001
	s_cbranch_execnz 1410                                      // 0000000055cc: bfa60582 <packed_folded_w4a8+0x50d8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000055d4: 8c7e007e
	v_mul_f32_e32 v4, v5, v9                                   // 0000000055d8: 10081305
	s_delay_alu instid0(valu_dep_1)                            // 0000000055dc: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 0000000055e0: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v19, v4                                  // 0000000055ec: 10080913
	s_xor_b32 s1, s0, -1                                       // 0000000055f0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055f4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000055f8: be802001
	s_cbranch_execnz 1415                                      // 0000000055fc: bfa60587 <packed_folded_w4a8+0x511c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005600: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005604: 8c7e007e
	v_mul_f32_e32 v5, v6, v9                                   // 000000005608: 100a1306
	s_delay_alu instid0(valu_dep_1)                            // 00000000560c: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 000000005610: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v20, v5                                  // 00000000561c: 100a0b14
	s_xor_b32 s1, s0, -1                                       // 000000005620: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005624: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005628: be802001
	s_cbranch_execnz 1420                                      // 00000000562c: bfa6058c <packed_folded_w4a8+0x5160>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005630: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005634: 8c7e007e
	v_mul_f32_e32 v6, v7, v9                                   // 000000005638: 100c1307
	s_delay_alu instid0(valu_dep_1)                            // 00000000563c: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000005640: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v21, v6                                  // 00000000564c: 100c0d15
	s_xor_b32 s1, s0, -1                                       // 000000005650: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005654: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005658: be802001
	s_cbranch_execnz 1425                                      // 00000000565c: bfa60591 <packed_folded_w4a8+0x51a4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005660: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005664: 8c7e007e
	v_mul_f32_e32 v7, v0, v9                                   // 000000005668: 100e1300
	s_delay_alu instid0(valu_dep_1)                            // 00000000566c: bf870001
	v_cmp_class_f32_e64 s0, v7, 0x198                          // 000000005670: d47e0000 0201ff07 00000198
	v_mul_f32_e32 v7, v22, v7                                  // 00000000567c: 100e0f16
	s_xor_b32 s1, s0, -1                                       // 000000005680: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005684: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005688: be802001
	s_cbranch_execnz 1430                                      // 00000000568c: bfa60596 <packed_folded_w4a8+0x51e8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005690: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005694: 8c7e007e
	v_mul_f32_e32 v0, v1, v9                                   // 000000005698: 10001301
	s_delay_alu instid0(valu_dep_1)                            // 00000000569c: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 0000000056a0: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v23, v0                                  // 0000000056ac: 10000117
	s_xor_b32 s1, s0, -1                                       // 0000000056b0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056b4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000056b8: be802001
	s_cbranch_execnz 1435                                      // 0000000056bc: bfa6059b <packed_folded_w4a8+0x522c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000056c4: 8c7e007e
	v_mul_f32_e32 v1, v2, v9                                   // 0000000056c8: 10021302
	s_delay_alu instid0(valu_dep_1)                            // 0000000056cc: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 0000000056d0: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v24, v1                                  // 0000000056dc: 10020318
	s_xor_b32 s1, s0, -1                                       // 0000000056e0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056e4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000056e8: be802001
	s_cbranch_execnz 1440                                      // 0000000056ec: bfa605a0 <packed_folded_w4a8+0x5270>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000056f4: 8c7e007e
	v_mul_f32_e32 v2, v3, v9                                   // 0000000056f8: 10041303
	s_delay_alu instid0(valu_dep_1)                            // 0000000056fc: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000005700: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v25, v2                                  // 00000000570c: 10040519
	s_xor_b32 s1, s0, -1                                       // 000000005710: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005714: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005718: be802001
	s_cbranch_execnz 1445                                      // 00000000571c: bfa605a5 <packed_folded_w4a8+0x52b4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005720: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005724: 8c7e007e
	v_bfe_u32 v3, v8, 16, 1                                    // 000000005728: d6100003 02052108
	v_bfe_u32 v9, v4, 16, 1                                    // 000000005730: d6100009 02052104
	v_or_b32_e32 v18, 0x400000, v8                             // 000000005738: 382410ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 000000005740: 7c301108
	v_or_b32_e32 v19, 0x400000, v4                             // 000000005744: 382608ff 00400000
	v_add3_u32 v3, v3, v8, 0x7fff                              // 00000000574c: d6550003 03fe1103 00007fff
	v_add3_u32 v9, v9, v4, 0x7fff                              // 000000005758: d6550009 03fe0909 00007fff
	v_bfe_u32 v20, v5, 16, 1                                   // 000000005764: d6100014 02052105
	v_bfe_u32 v8, v6, 16, 1                                    // 00000000576c: d6100008 02052106
	s_and_b32 s0, s2, s3                                       // 000000005774: 8b000302
	s_wait_alu depctr_va_vcc(0)                                // 000000005778: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v18, vcc_lo                      // 00000000577c: 02062503
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 000000005780: 7c300904
	v_add3_u32 v18, v20, v5, 0x7fff                            // 000000005784: d6550012 03fe0b14 00007fff
	s_wait_alu depctr_sa_sdst(0)                               // 000000005790: bf88ff9e
	s_and_b32 s0, s10, s0                                      // 000000005794: 8b00000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005798: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 00000000579c: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 0000000057a0: bf88ff9d
	v_cndmask_b32_e32 v4, v9, v19, vcc_lo                      // 0000000057a4: 02082709
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 0000000057a8: 7c300b05
	v_bfe_u32 v9, v7, 16, 1                                    // 0000000057ac: d6100009 02052107
	s_clause 0x1                                               // 0000000057b4: bf850001
	global_store_d16_hi_b16 v[132:133], v3, off offset:32      // 0000000057b8: ee09407c 01800000 00002084
	global_store_d16_hi_b16 v[138:139], v4, off offset:32      // 0000000057c4: ee09407c 02000000 0000208a
	v_or_b32_e32 v3, 0x400000, v5                              // 0000000057d0: 38060aff 00400000
	v_add3_u32 v4, v8, v6, 0x7fff                              // 0000000057d8: d6550004 03fe0d08 00007fff
	v_or_b32_e32 v8, 0x400000, v6                              // 0000000057e4: 38100cff 00400000
	v_add3_u32 v9, v9, v7, 0x7fff                              // 0000000057ec: d6550009 03fe0f09 00007fff
	v_or_b32_e32 v19, 0x400000, v7                             // 0000000057f8: 38260eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005800: bf88ff9d
	v_cndmask_b32_e32 v3, v18, v3, vcc_lo                      // 000000005804: 02060712
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000005808: 7c300d06
	v_or_b32_e32 v6, 0x400000, v0                              // 00000000580c: 380c00ff 00400000
	global_store_d16_hi_b16 v[140:141], v3, off offset:32      // 000000005814: ee09407c 01800000 0000208c
	s_wait_alu depctr_va_vcc(0)                                // 000000005820: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v8, vcc_lo                       // 000000005824: 02081104
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 000000005828: 7c300f07
	v_bfe_u32 v3, v0, 16, 1                                    // 00000000582c: d6100003 02052100
	v_or_b32_e32 v7, 0x400000, v1                              // 000000005834: 380e02ff 00400000
	v_or_b32_e32 v8, 0x400000, v2                              // 00000000583c: 381004ff 00400000
	global_store_d16_hi_b16 v[144:145], v4, off offset:32      // 000000005844: ee09407c 02000000 00002090
	s_wait_alu depctr_va_vcc(0)                                // 000000005850: bf88ff9d
	v_cndmask_b32_e32 v5, v9, v19, vcc_lo                      // 000000005854: 020a2709
	v_bfe_u32 v4, v1, 16, 1                                    // 000000005858: d6100004 02052101
	v_add3_u32 v3, v3, v0, 0x7fff                              // 000000005860: d6550003 03fe0103 00007fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 00000000586c: 7c300100
	global_store_d16_hi_b16 v[148:149], v5, off offset:32      // 000000005870: ee09407c 02800000 00002094
	v_bfe_u32 v5, v2, 16, 1                                    // 00000000587c: d6100005 02052102
	v_add3_u32 v4, v4, v1, 0x7fff                              // 000000005884: d6550004 03fe0304 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005890: bf88ff9d
	v_cndmask_b32_e32 v0, v3, v6, vcc_lo                       // 000000005894: 02000d03
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000005898: 7c300301
	v_add3_u32 v5, v5, v2, 0x7fff                              // 00000000589c: d6550005 03fe0505 00007fff
	global_store_d16_hi_b16 v[146:147], v0, off offset:32      // 0000000058a8: ee09407c 00000000 00002092
	s_wait_alu depctr_va_vcc(0)                                // 0000000058b4: bf88ff9d
	v_cndmask_b32_e32 v1, v4, v7, vcc_lo                       // 0000000058b8: 02020f04
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 0000000058bc: 7c300502
	global_store_d16_hi_b16 v[150:151], v1, off offset:32      // 0000000058c0: ee09407c 00800000 00002096
	s_wait_alu depctr_va_vcc(0)                                // 0000000058cc: bf88ff9d
	v_cndmask_b32_e32 v2, v5, v8, vcc_lo                       // 0000000058d0: 02041105
	global_store_d16_hi_b16 v[152:153], v2, off offset:32      // 0000000058d4: ee09407c 01000000 00002098
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058e0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000058e4: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058e8: bf88ff9e
	s_xor_b32 s2, exec_lo, s1                                  // 0000000058ec: 8d02017e
	s_cbranch_execz 119                                        // 0000000058f0: bfa50077 <packed_folded_w4a8+0x3fd0>
	v_mov_b32_e32 v143, v51                                    // 0000000058f4: 7f1e0333
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[134:135]              // 0000000058f8: 7ca90c10
	v_mov_b32_e32 v137, v51                                    // 0000000058fc: 7f120333
	v_mov_b32_e32 v131, v51                                    // 000000005900: 7f060333
	v_mov_b32_e32 v57, v51                                     // 000000005904: 7e720333
	v_cmp_gt_i64_e64 s0, s[16:17], v[142:143]                  // 000000005908: d4540000 02031c10
	v_mov_b32_e32 v55, v51                                     // 000000005910: 7e6e0333
	s_wait_alu depctr_va_vcc(0)                                // 000000005914: bf88ff9d
	v_dual_cndmask_b32 v1, 0, v135 :: v_dual_cndmask_b32 v0, 0, v134// 000000005918: ca530e80 01010c80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[136:137]              // 000000005920: 7ca91010
	v_mov_b32_e32 v53, v51                                     // 000000005924: 7e6a0333
	s_wait_alu depctr_va_sdst(0)                               // 000000005928: bf88f19f
	v_cndmask_b32_e64 v3, 0, v143, s0                          // 00000000592c: d5010003 00031e80
	v_cndmask_b32_e64 v2, 0, v142, s0                          // 000000005934: d5010002 00031c80
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 00000000593c: 3e000082
	s_wait_alu depctr_va_vcc(0)                                // 000000005940: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v137 :: v_dual_cndmask_b32 v4, 0, v136// 000000005944: ca531280 05051080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[130:131]              // 00000000594c: 7ca90410
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000005950: 3e040482
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000005954: bf870214
	v_add_co_u32 v0, s0, s6, v0                                // 000000005958: d7000000 02020006
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000005960: 3e080882
	s_wait_alu depctr_va_sdst(0)                               // 000000005964: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s7, v1, s0                   // 000000005968: d5207c01 00020207
	s_wait_alu depctr_va_vcc(0)                                // 000000005970: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v131 :: v_dual_cndmask_b32 v6, 0, v130// 000000005974: ca530680 07070480
	v_add_co_u32 v2, s0, s6, v2                                // 00000000597c: d7000002 02020406
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[56:57]                // 000000005984: 7ca87010
	v_add_co_u32 v8, s1, s6, v4                                // 000000005988: d7000108 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000005990: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s7, v3, s0                   // 000000005994: d5207c03 00020607
	v_cmp_gt_i64_e64 s0, s[16:17], v[54:55]                    // 00000000599c: d4540000 02026c10
	v_add_co_ci_u32_e64 v9, null, s7, v5, s1                   // 0000000059a4: d5207c09 00060a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000059ac: 3e080c82
	s_wait_alu depctr_va_vcc(0)                                // 0000000059b0: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v57 :: v_dual_cndmask_b32 v6, 0, v56// 0000000059b4: ca527280 07067080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[52:53]                // 0000000059bc: 7ca86810
	s_wait_alu depctr_va_sdst(0)                               // 0000000059c0: bf88f19f
	v_cndmask_b32_e64 v19, 0, v55, s0                          // 0000000059c4: d5010013 00026e80
	v_cndmask_b32_e64 v18, 0, v54, s0                          // 0000000059cc: d5010012 00026c80
	v_add_co_u32 v20, s0, s6, v4                               // 0000000059d4: d7000014 02020806
	s_wait_alu depctr_va_sdst(0)                               // 0000000059dc: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s7, v5, s0                  // 0000000059e0: d5207c15 00020a07
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 0000000059e8: 3e080c82
	v_lshlrev_b64_e32 v[6:7], 2, v[18:19]                      // 0000000059ec: 3e0c2482
	s_wait_alu depctr_va_vcc(0)                                // 0000000059f0: bf88ff9d
	v_dual_cndmask_b32 v19, 0, v53 :: v_dual_cndmask_b32 v18, 0, v52// 0000000059f4: ca526a80 13126880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[50:51]                // 0000000059fc: 7ca86410
	s_delay_alu instid0(valu_dep_4)                            // 000000005a00: bf870004
	v_add_co_u32 v22, s0, s6, v4                               // 000000005a04: d7000016 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000005a0c: bf88f19f
	v_add_co_ci_u32_e64 v23, null, s7, v5, s0                  // 000000005a10: d5207c17 00020a07
	v_lshlrev_b64_e32 v[4:5], 2, v[18:19]                      // 000000005a18: 3e082482
	s_wait_alu depctr_va_vcc(0)                                // 000000005a1c: bf88ff9d
	v_dual_cndmask_b32 v19, 0, v51 :: v_dual_cndmask_b32 v18, 0, v50// 000000005a20: ca526680 13126480
	v_add_co_u32 v24, vcc_lo, s6, v6                           // 000000005a28: d7006a18 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 000000005a30: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s7, v7, vcc_lo              // 000000005a34: d5207c19 01aa0e07
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000005a3c: bf8701c3
	v_lshlrev_b64_e32 v[6:7], 2, v[18:19]                      // 000000005a40: 3e0c2482
	v_add_co_u32 v18, vcc_lo, s6, v4                           // 000000005a44: d7006a12 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000005a4c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s7, v5, vcc_lo              // 000000005a50: d5207c13 01aa0a07
	v_add_co_u32 v26, vcc_lo, s6, v6                           // 000000005a58: d7006a1a 02020c06
	s_wait_alu depctr_va_vcc(0)                                // 000000005a60: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, s7, v7, vcc_lo              // 000000005a64: d5207c1b 01aa0e07
	s_clause 0x7                                               // 000000005a6c: bf850007
	global_load_b32 v4, v[0:1], off                            // 000000005a70: ee05007c 00000004 00000000
	global_load_b32 v5, v[2:3], off                            // 000000005a7c: ee05007c 00000005 00000002
	global_load_b32 v6, v[8:9], off                            // 000000005a88: ee05007c 00000006 00000008
	global_load_b32 v7, v[20:21], off                          // 000000005a94: ee05007c 00000007 00000014
	global_load_b32 v0, v[22:23], off                          // 000000005aa0: ee05007c 00000000 00000016
	global_load_b32 v1, v[24:25], off                          // 000000005aac: ee05007c 00000001 00000018
	global_load_b32 v2, v[18:19], off                          // 000000005ab8: ee05007c 00000002 00000012
	global_load_b32 v3, v[26:27], off                          // 000000005ac4: ee05007c 00000003 0000001a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ad0: bf88ff9e
	s_and_not1_saveexec_b32 s0, s2                             // 000000005ad4: be803002
	s_cbranch_execz 28                                         // 000000005ad8: bfa5001c <packed_folded_w4a8+0x404c>
	s_wait_loadcnt 0x3                                         // 000000005adc: bfc00003
	v_add_co_u32 v0, s1, v74, s14                              // 000000005ae0: d7000100 02001d4a
	s_wait_loadcnt 0x2                                         // 000000005ae8: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000005aec: bf88f19f
	v_add_co_ci_u32_e64 v1, null, 0, s15, s1                   // 000000005af0: d5207c01 00041e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000005af8: bf870122
	v_add_co_u32 v0, vcc_lo, v0, v75                           // 000000005afc: d7006a00 02029700
	s_wait_alu depctr_va_vcc(0)                                // 000000005b04: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo                // 000000005b08: d5207c01 01aa0280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b10: bf870091
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000005b14: 3e000082
	v_add_co_u32 v0, vcc_lo, s6, v0                            // 000000005b18: d7006a00 02020006
	s_wait_alu depctr_va_vcc(0)                                // 000000005b20: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000005b24: bf870002
	v_add_co_ci_u32_e64 v1, null, s7, v1, vcc_lo               // 000000005b28: d5207c01 01aa0207
	global_load_b128 v[4:7], v[0:1], off offset:192            // 000000005b30: ee05c07c 00000004 0000c000
	s_wait_loadcnt 0x1                                         // 000000005b3c: bfc00001
	global_load_b128 v[0:3], v[0:1], off offset:208            // 000000005b40: ee05c07c 00000000 0000d000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b50: 8c7e007e
	global_load_u8 v8, v[77:78], off                           // 000000005b54: ee04007c 00000008 0000004d
	s_wait_loadcnt 0x0                                         // 000000005b60: bfc00000
	v_lshlrev_b32_e32 v9, 23, v8                               // 000000005b64: 30121097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b68: bf870091
	v_mul_f32_e32 v8, v4, v9                                   // 000000005b6c: 10101304
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000005b70: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v10, v8                                  // 000000005b7c: 1010110a
	s_xor_b32 s1, s0, -1                                       // 000000005b80: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b84: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005b88: be802001
	s_cbranch_execnz 1178                                      // 000000005b8c: bfa6049a <packed_folded_w4a8+0x52f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b90: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b94: 8c7e007e
	v_mul_f32_e32 v4, v5, v9                                   // 000000005b98: 10081305
	s_delay_alu instid0(valu_dep_1)                            // 000000005b9c: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 000000005ba0: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v11, v4                                  // 000000005bac: 1008090b
	s_xor_b32 s1, s0, -1                                       // 000000005bb0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bb4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005bb8: be802001
	s_cbranch_execnz 1183                                      // 000000005bbc: bfa6049f <packed_folded_w4a8+0x533c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bc0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005bc4: 8c7e007e
	v_mul_f32_e32 v5, v6, v9                                   // 000000005bc8: 100a1306
	s_delay_alu instid0(valu_dep_1)                            // 000000005bcc: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 000000005bd0: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v12, v5                                  // 000000005bdc: 100a0b0c
	s_xor_b32 s1, s0, -1                                       // 000000005be0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005be4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005be8: be802001
	s_cbranch_execnz 1188                                      // 000000005bec: bfa604a4 <packed_folded_w4a8+0x5380>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bf0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005bf4: 8c7e007e
	v_mul_f32_e32 v6, v7, v9                                   // 000000005bf8: 100c1307
	s_delay_alu instid0(valu_dep_1)                            // 000000005bfc: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000005c00: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v13, v6                                  // 000000005c0c: 100c0d0d
	s_xor_b32 s1, s0, -1                                       // 000000005c10: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c14: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005c18: be802001
	s_cbranch_execnz 1193                                      // 000000005c1c: bfa604a9 <packed_folded_w4a8+0x53c4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005c24: 8c7e007e
	v_mul_f32_e32 v7, v0, v9                                   // 000000005c28: 100e1300
	s_delay_alu instid0(valu_dep_1)                            // 000000005c2c: bf870001
	v_cmp_class_f32_e64 s0, v7, 0x198                          // 000000005c30: d47e0000 0201ff07 00000198
	v_mul_f32_e32 v7, v14, v7                                  // 000000005c3c: 100e0f0e
	s_xor_b32 s1, s0, -1                                       // 000000005c40: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c44: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005c48: be802001
	s_cbranch_execnz 1198                                      // 000000005c4c: bfa604ae <packed_folded_w4a8+0x5408>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005c54: 8c7e007e
	v_mul_f32_e32 v0, v1, v9                                   // 000000005c58: 10001301
	s_delay_alu instid0(valu_dep_1)                            // 000000005c5c: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 000000005c60: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v15, v0                                  // 000000005c6c: 1000010f
	s_xor_b32 s1, s0, -1                                       // 000000005c70: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c74: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005c78: be802001
	s_cbranch_execnz 1203                                      // 000000005c7c: bfa604b3 <packed_folded_w4a8+0x544c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005c84: 8c7e007e
	v_mul_f32_e32 v1, v2, v9                                   // 000000005c88: 10021302
	s_delay_alu instid0(valu_dep_1)                            // 000000005c8c: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 000000005c90: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v16, v1                                  // 000000005c9c: 10020310
	s_xor_b32 s1, s0, -1                                       // 000000005ca0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ca4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005ca8: be802001
	s_cbranch_execnz 1208                                      // 000000005cac: bfa604b8 <packed_folded_w4a8+0x5490>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cb0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005cb4: 8c7e007e
	v_mul_f32_e32 v2, v3, v9                                   // 000000005cb8: 10041303
	s_delay_alu instid0(valu_dep_1)                            // 000000005cbc: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000005cc0: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v17, v2                                  // 000000005ccc: 10040511
	s_xor_b32 s1, s0, -1                                       // 000000005cd0: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cd4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005cd8: be802001
	s_cbranch_execnz 1213                                      // 000000005cdc: bfa604bd <packed_folded_w4a8+0x54d4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ce0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005ce4: 8c7e007e
	v_bfe_u32 v3, v8, 16, 1                                    // 000000005ce8: d6100003 02052108
	v_or_b32_e32 v9, 0x400000, v8                              // 000000005cf0: 381210ff 00400000
	v_bfe_u32 v10, v4, 16, 1                                   // 000000005cf8: d610000a 02052104
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 000000005d00: 7c301108
	v_or_b32_e32 v11, 0x400000, v4                             // 000000005d04: 381608ff 00400000
	v_add3_u32 v3, v3, v8, 0x7fff                              // 000000005d0c: d6550003 03fe1103 00007fff
	v_bfe_u32 v12, v5, 16, 1                                   // 000000005d18: d610000c 02052105
	v_add3_u32 v10, v10, v4, 0x7fff                            // 000000005d20: d655000a 03fe090a 00007fff
	v_or_b32_e32 v13, 0x400000, v5                             // 000000005d2c: 381a0aff 00400000
	v_bfe_u32 v8, v6, 16, 1                                    // 000000005d34: d6100008 02052106
	s_wait_alu depctr_va_vcc(0)                                // 000000005d3c: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v9, vcc_lo                       // 000000005d40: 02061303
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 000000005d44: 7c300904
	v_add3_u32 v9, v12, v5, 0x7fff                             // 000000005d48: d6550009 03fe0b0c 00007fff
	global_store_d16_hi_b16 v[46:47], v3, off offset:32        // 000000005d54: ee09407c 01800000 0000202e
	s_wait_alu depctr_va_vcc(0)                                // 000000005d60: bf88ff9d
	v_cndmask_b32_e32 v4, v10, v11, vcc_lo                     // 000000005d64: 0208170a
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000005d68: 7c300b05
	v_bfe_u32 v5, v7, 16, 1                                    // 000000005d6c: d6100005 02052107
	v_or_b32_e32 v10, 0x400000, v2                             // 000000005d74: 381404ff 00400000
	global_store_d16_hi_b16 v[44:45], v4, off offset:32        // 000000005d7c: ee09407c 02000000 0000202c
	s_wait_alu depctr_va_vcc(0)                                // 000000005d88: bf88ff9d
	v_cndmask_b32_e32 v3, v9, v13, vcc_lo                      // 000000005d8c: 02061b09
	v_add3_u32 v4, v8, v6, 0x7fff                              // 000000005d90: d6550004 03fe0d08 00007fff
	v_or_b32_e32 v8, 0x400000, v6                              // 000000005d9c: 38100cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000005da4: 7c300d06
	v_bfe_u32 v6, v0, 16, 1                                    // 000000005da8: d6100006 02052100
	global_store_d16_hi_b16 v[154:155], v3, off offset:32      // 000000005db0: ee09407c 01800000 0000209a
	v_add3_u32 v3, v5, v7, 0x7fff                              // 000000005dbc: d6550003 03fe0f05 00007fff
	v_or_b32_e32 v5, 0x400000, v7                              // 000000005dc8: 380a0eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005dd0: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v8, vcc_lo                       // 000000005dd4: 02081104
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 000000005dd8: 7c300f07
	v_bfe_u32 v8, v1, 16, 1                                    // 000000005ddc: d6100008 02052101
	v_add3_u32 v6, v6, v0, 0x7fff                              // 000000005de4: d6550006 03fe0106 00007fff
	v_or_b32_e32 v7, 0x400000, v0                              // 000000005df0: 380e00ff 00400000
	v_or_b32_e32 v9, 0x400000, v1                              // 000000005df8: 381202ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005e00: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v5, vcc_lo                       // 000000005e04: 02060b03
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 000000005e08: 7c300100
	v_bfe_u32 v5, v2, 16, 1                                    // 000000005e0c: d6100005 02052102
	v_add3_u32 v8, v8, v1, 0x7fff                              // 000000005e14: d6550008 03fe0308 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005e20: bf88ff9d
	v_cndmask_b32_e32 v0, v6, v7, vcc_lo                       // 000000005e24: 02000f06
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000005e28: 7c300301
	v_add3_u32 v5, v5, v2, 0x7fff                              // 000000005e2c: d6550005 03fe0505 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005e38: bf88ff9d
	v_cndmask_b32_e32 v1, v8, v9, vcc_lo                       // 000000005e3c: 02021308
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000005e40: 7c300502
	s_wait_alu depctr_va_vcc(0)                                // 000000005e44: bf88ff9d
	v_cndmask_b32_e32 v2, v5, v10, vcc_lo                      // 000000005e48: 02041505
	s_clause 0x3                                               // 000000005e4c: bf850003
	global_store_d16_hi_b16 v[42:43], v4, off offset:32        // 000000005e50: ee09407c 02000000 0000202a
	global_store_d16_hi_b16 v[48:49], v3, off offset:32        // 000000005e5c: ee09407c 01800000 00002030
	global_store_d16_hi_b16 v[156:157], v0, off offset:32      // 000000005e68: ee09407c 00000000 0000209c
	global_store_d16_hi_b16 v[158:159], v1, off offset:32      // 000000005e74: ee09407c 00800000 0000209e
	global_store_d16_hi_b16 v[160:161], v2, off offset:32      // 000000005e80: ee09407c 01000000 000020a0
	s_nop 0                                                    // 000000005e8c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005e90: bfb60003
	s_endpgm                                                   // 000000005e94: bfb00000
	v_cvt_f64_f32_e32 v[85:86], v2                             // 000000005e98: 7eaa2102
	v_cvt_f64_f32_e32 v[87:88], v84                            // 000000005e9c: 7eae2154
	v_cvt_f64_f32_e32 v[89:90], v70                            // 000000005ea0: 7eb22146
	v_cmp_eq_f32_e64 s2, 0, v2                                 // 000000005ea4: d4120002 02020480
	v_cmp_class_f32_e64 s5, v70, 0x1f8                         // 000000005eac: d47e0005 0201ff46 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005eb8: 8b020502
	v_mul_f64_e32 v[85:86], v[85:86], v[87:88]                 // 000000005ebc: 0caaaf55
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ec0: bf870091
	v_mul_f64_e32 v[85:86], v[85:86], v[89:90]                 // 000000005ec4: 0caab355
	v_cvt_f32_f64_e32 v83, v[85:86]                            // 000000005ec8: 7ea61f55
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ecc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ed0: bf870001
	v_cndmask_b32_e64 v83, v83, 0, s2                          // 000000005ed4: d5010053 00090153
	s_branch 62190                                             // 000000005edc: bfa0f2ee <packed_folded_w4a8+0xf98>
	v_cvt_f64_f32_e32 v[85:86], v3                             // 000000005ee0: 7eaa2103
	v_cvt_f64_f32_e32 v[87:88], v84                            // 000000005ee4: 7eae2154
	v_cvt_f64_f32_e32 v[89:90], v71                            // 000000005ee8: 7eb22147
	v_cmp_eq_f32_e64 s2, 0, v3                                 // 000000005eec: d4120002 02020680
	v_cmp_class_f32_e64 s5, v71, 0x1f8                         // 000000005ef4: d47e0005 0201ff47 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f00: 8b020502
	v_mul_f64_e32 v[85:86], v[85:86], v[87:88]                 // 000000005f04: 0caaaf55
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f08: bf870091
	v_mul_f64_e32 v[85:86], v[85:86], v[89:90]                 // 000000005f0c: 0caab355
	v_cvt_f32_f64_e32 v2, v[85:86]                             // 000000005f10: 7e041f55
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f14: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f18: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s2                            // 000000005f1c: d5010002 00090102
	s_branch 62184                                             // 000000005f24: bfa0f2e8 <packed_folded_w4a8+0xfc8>
	v_cvt_f64_f32_e32 v[70:71], v4                             // 000000005f28: 7e8c2104
	v_cvt_f64_f32_e32 v[85:86], v84                            // 000000005f2c: 7eaa2154
	v_cvt_f64_f32_e32 v[87:88], v72                            // 000000005f30: 7eae2148
	v_cmp_eq_f32_e64 s2, 0, v4                                 // 000000005f34: d4120002 02020880
	v_cmp_class_f32_e64 s5, v72, 0x1f8                         // 000000005f3c: d47e0005 0201ff48 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f48: 8b020502
	v_mul_f64_e32 v[70:71], v[70:71], v[85:86]                 // 000000005f4c: 0c8cab46
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f50: bf870091
	v_mul_f64_e32 v[70:71], v[70:71], v[87:88]                 // 000000005f54: 0c8caf46
	v_cvt_f32_f64_e32 v3, v[70:71]                             // 000000005f58: 7e061f46
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f5c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f60: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s2                            // 000000005f64: d5010003 00090103
	s_branch 62178                                             // 000000005f6c: bfa0f2e2 <packed_folded_w4a8+0xff8>
	v_cvt_f64_f32_e32 v[70:71], v5                             // 000000005f70: 7e8c2105
	v_cvt_f64_f32_e32 v[85:86], v84                            // 000000005f74: 7eaa2154
	v_cvt_f64_f32_e32 v[87:88], v73                            // 000000005f78: 7eae2149
	v_cmp_eq_f32_e64 s2, 0, v5                                 // 000000005f7c: d4120002 02020a80
	v_cmp_class_f32_e64 s5, v73, 0x1f8                         // 000000005f84: d47e0005 0201ff49 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f90: 8b020502
	v_mul_f64_e32 v[70:71], v[70:71], v[85:86]                 // 000000005f94: 0c8cab46
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f98: bf870091
	v_mul_f64_e32 v[70:71], v[70:71], v[87:88]                 // 000000005f9c: 0c8caf46
	v_cvt_f32_f64_e32 v4, v[70:71]                             // 000000005fa0: 7e081f46
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fa4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005fa8: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s2                            // 000000005fac: d5010004 00090104
	s_branch 62172                                             // 000000005fb4: bfa0f2dc <packed_folded_w4a8+0x1028>
	v_cvt_f64_f32_e32 v[70:71], v6                             // 000000005fb8: 7e8c2106
	v_cvt_f64_f32_e32 v[72:73], v84                            // 000000005fbc: 7e902154
	v_cvt_f64_f32_e32 v[85:86], v66                            // 000000005fc0: 7eaa2142
	v_cmp_eq_f32_e64 s2, 0, v6                                 // 000000005fc4: d4120002 02020c80
	v_cmp_class_f32_e64 s5, v66, 0x1f8                         // 000000005fcc: d47e0005 0201ff42 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005fd8: 8b020502
	v_mul_f64_e32 v[70:71], v[70:71], v[72:73]                 // 000000005fdc: 0c8c9146
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005fe0: bf870091
	v_mul_f64_e32 v[70:71], v[70:71], v[85:86]                 // 000000005fe4: 0c8cab46
	v_cvt_f32_f64_e32 v5, v[70:71]                             // 000000005fe8: 7e0a1f46
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ff0: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s2                            // 000000005ff4: d5010005 00090105
	s_branch 62166                                             // 000000005ffc: bfa0f2d6 <packed_folded_w4a8+0x1058>
	v_cvt_f64_f32_e32 v[70:71], v7                             // 000000006000: 7e8c2107
	v_cvt_f64_f32_e32 v[72:73], v84                            // 000000006004: 7e902154
	v_cvt_f64_f32_e32 v[85:86], v67                            // 000000006008: 7eaa2143
	v_cmp_eq_f32_e64 s2, 0, v7                                 // 00000000600c: d4120002 02020e80
	v_cmp_class_f32_e64 s5, v67, 0x1f8                         // 000000006014: d47e0005 0201ff43 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006020: 8b020502
	v_mul_f64_e32 v[70:71], v[70:71], v[72:73]                 // 000000006024: 0c8c9146
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006028: bf870091
	v_mul_f64_e32 v[70:71], v[70:71], v[85:86]                 // 00000000602c: 0c8cab46
	v_cvt_f32_f64_e32 v6, v[70:71]                             // 000000006030: 7e0c1f46
	s_wait_alu depctr_sa_sdst(0)                               // 000000006034: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006038: bf870001
	v_cndmask_b32_e64 v66, v6, 0, s2                           // 00000000603c: d5010042 00090106
	s_branch 62160                                             // 000000006044: bfa0f2d0 <packed_folded_w4a8+0x1088>
	v_cvt_f64_f32_e32 v[6:7], v8                               // 000000006048: 7e0c2108
	v_cvt_f64_f32_e32 v[70:71], v84                            // 00000000604c: 7e8c2154
	v_cvt_f64_f32_e32 v[72:73], v68                            // 000000006050: 7e902144
	v_cmp_eq_f32_e64 s2, 0, v8                                 // 000000006054: d4120002 02021080
	v_cmp_class_f32_e64 s5, v68, 0x1f8                         // 00000000605c: d47e0005 0201ff44 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006068: 8b020502
	v_mul_f64_e32 v[6:7], v[6:7], v[70:71]                     // 00000000606c: 0c0c8d06
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006070: bf870091
	v_mul_f64_e32 v[6:7], v[6:7], v[72:73]                     // 000000006074: 0c0c9106
	v_cvt_f32_f64_e32 v6, v[6:7]                               // 000000006078: 7e0c1f06
	s_wait_alu depctr_sa_sdst(0)                               // 00000000607c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006080: bf870001
	v_cndmask_b32_e64 v7, v6, 0, s2                            // 000000006084: d5010007 00090106
	s_branch 62154                                             // 00000000608c: bfa0f2ca <packed_folded_w4a8+0x10b8>
	v_cvt_f64_f32_e32 v[67:68], v9                             // 000000006090: 7e862109
	v_cvt_f64_f32_e32 v[70:71], v84                            // 000000006094: 7e8c2154
	v_cvt_f64_f32_e32 v[72:73], v69                            // 000000006098: 7e902145
	v_cmp_eq_f32_e64 s2, 0, v9                                 // 00000000609c: d4120002 02021280
	v_cmp_class_f32_e64 s5, v69, 0x1f8                         // 0000000060a4: d47e0005 0201ff45 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000060b0: 8b020502
	v_mul_f64_e32 v[67:68], v[67:68], v[70:71]                 // 0000000060b4: 0c868d43
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000060b8: bf870091
	v_mul_f64_e32 v[67:68], v[67:68], v[72:73]                 // 0000000060bc: 0c869143
	v_cvt_f32_f64_e32 v6, v[67:68]                             // 0000000060c0: 7e0c1f43
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000060c8: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s2                            // 0000000060cc: d5010006 00090106
	s_branch 62148                                             // 0000000060d4: bfa0f2c4 <packed_folded_w4a8+0x10e8>
	v_cvt_f64_f32_e32 v[105:106], v58                          // 0000000060d8: 7ed2213a
	v_cvt_f64_f32_e32 v[107:108], v69                          // 0000000060dc: 7ed62145
	v_cvt_f64_f32_e32 v[109:110], v4                           // 0000000060e0: 7eda2104
	v_cmp_eq_f32_e64 s1, 0, v58                                // 0000000060e4: d4120001 02027480
	v_cmp_class_f32_e64 s4, v4, 0x1f8                          // 0000000060ec: d47e0004 0201ff04 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000060f8: 8b010401
	v_mul_f64_e32 v[105:106], v[105:106], v[107:108]           // 0000000060fc: 0cd2d769
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006100: bf870091
	v_mul_f64_e32 v[105:106], v[105:106], v[109:110]           // 000000006104: 0cd2db69
	v_cvt_f32_f64_e32 v67, v[105:106]                          // 000000006108: 7e861f69
	s_wait_alu depctr_sa_sdst(0)                               // 00000000610c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006110: bf870001
	v_cndmask_b32_e64 v67, v67, 0, s1                          // 000000006114: d5010043 00050143
	s_branch 62565                                             // 00000000611c: bfa0f465 <packed_folded_w4a8+0x17b4>
	v_cvt_f64_f32_e32 v[105:106], v59                          // 000000006120: 7ed2213b
	v_cvt_f64_f32_e32 v[107:108], v69                          // 000000006124: 7ed62145
	v_cvt_f64_f32_e32 v[109:110], v5                           // 000000006128: 7eda2105
	v_cmp_eq_f32_e64 s1, 0, v59                                // 00000000612c: d4120001 02027680
	v_cmp_class_f32_e64 s4, v5, 0x1f8                          // 000000006134: d47e0004 0201ff05 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006140: 8b010401
	v_mul_f64_e32 v[105:106], v[105:106], v[107:108]           // 000000006144: 0cd2d769
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006148: bf870091
	v_mul_f64_e32 v[105:106], v[105:106], v[109:110]           // 00000000614c: 0cd2db69
	v_cvt_f32_f64_e32 v4, v[105:106]                           // 000000006150: 7e081f69
	s_wait_alu depctr_sa_sdst(0)                               // 000000006154: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006158: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s1                            // 00000000615c: d5010004 00050104
	s_branch 62559                                             // 000000006164: bfa0f45f <packed_folded_w4a8+0x17e4>
	v_cvt_f64_f32_e32 v[58:59], v60                            // 000000006168: 7e74213c
	v_cvt_f64_f32_e32 v[105:106], v69                          // 00000000616c: 7ed22145
	v_cvt_f64_f32_e32 v[107:108], v6                           // 000000006170: 7ed62106
	v_cmp_eq_f32_e64 s1, 0, v60                                // 000000006174: d4120001 02027880
	v_cmp_class_f32_e64 s4, v6, 0x1f8                          // 00000000617c: d47e0004 0201ff06 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006188: 8b010401
	v_mul_f64_e32 v[58:59], v[58:59], v[105:106]               // 00000000618c: 0c74d33a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006190: bf870091
	v_mul_f64_e32 v[58:59], v[58:59], v[107:108]               // 000000006194: 0c74d73a
	v_cvt_f32_f64_e32 v5, v[58:59]                             // 000000006198: 7e0a1f3a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000619c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061a0: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s1                            // 0000000061a4: d5010005 00050105
	s_branch 62553                                             // 0000000061ac: bfa0f459 <packed_folded_w4a8+0x1814>
	v_cvt_f64_f32_e32 v[58:59], v61                            // 0000000061b0: 7e74213d
	v_cvt_f64_f32_e32 v[105:106], v69                          // 0000000061b4: 7ed22145
	v_cvt_f64_f32_e32 v[107:108], v7                           // 0000000061b8: 7ed62107
	v_cmp_eq_f32_e64 s1, 0, v61                                // 0000000061bc: d4120001 02027a80
	v_cmp_class_f32_e64 s4, v7, 0x1f8                          // 0000000061c4: d47e0004 0201ff07 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000061d0: 8b010401
	v_mul_f64_e32 v[58:59], v[58:59], v[105:106]               // 0000000061d4: 0c74d33a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000061d8: bf870091
	v_mul_f64_e32 v[58:59], v[58:59], v[107:108]               // 0000000061dc: 0c74d73a
	v_cvt_f32_f64_e32 v6, v[58:59]                             // 0000000061e0: 7e0c1f3a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061e8: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s1                            // 0000000061ec: d5010006 00050106
	s_branch 62547                                             // 0000000061f4: bfa0f453 <packed_folded_w4a8+0x1844>
	v_cvt_f64_f32_e32 v[58:59], v62                            // 0000000061f8: 7e74213e
	v_cvt_f64_f32_e32 v[60:61], v69                            // 0000000061fc: 7e782145
	v_cvt_f64_f32_e32 v[105:106], v0                           // 000000006200: 7ed22100
	v_cmp_eq_f32_e64 s1, 0, v62                                // 000000006204: d4120001 02027c80
	v_cmp_class_f32_e64 s4, v0, 0x1f8                          // 00000000620c: d47e0004 0201ff00 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006218: 8b010401
	v_mul_f64_e32 v[58:59], v[58:59], v[60:61]                 // 00000000621c: 0c74793a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006220: bf870091
	v_mul_f64_e32 v[58:59], v[58:59], v[105:106]               // 000000006224: 0c74d33a
	v_cvt_f32_f64_e32 v7, v[58:59]                             // 000000006228: 7e0e1f3a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000622c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006230: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s1                            // 000000006234: d5010007 00050107
	s_branch 62541                                             // 00000000623c: bfa0f44d <packed_folded_w4a8+0x1874>
	v_cvt_f64_f32_e32 v[58:59], v63                            // 000000006240: 7e74213f
	v_cvt_f64_f32_e32 v[60:61], v69                            // 000000006244: 7e782145
	v_cvt_f64_f32_e32 v[105:106], v1                           // 000000006248: 7ed22101
	v_cmp_eq_f32_e64 s1, 0, v63                                // 00000000624c: d4120001 02027e80
	v_cmp_class_f32_e64 s4, v1, 0x1f8                          // 000000006254: d47e0004 0201ff01 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006260: 8b010401
	v_mul_f64_e32 v[58:59], v[58:59], v[60:61]                 // 000000006264: 0c74793a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006268: bf870091
	v_mul_f64_e32 v[58:59], v[58:59], v[105:106]               // 00000000626c: 0c74d33a
	v_cvt_f32_f64_e32 v0, v[58:59]                             // 000000006270: 7e001f3a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006274: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006278: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s1                            // 00000000627c: d5010000 00050100
	s_branch 62535                                             // 000000006284: bfa0f447 <packed_folded_w4a8+0x18a4>
	v_cvt_f64_f32_e32 v[58:59], v64                            // 000000006288: 7e742140
	v_cvt_f64_f32_e32 v[60:61], v69                            // 00000000628c: 7e782145
	v_cvt_f64_f32_e32 v[62:63], v2                             // 000000006290: 7e7c2102
	v_cmp_eq_f32_e64 s1, 0, v64                                // 000000006294: d4120001 02028080
	v_cmp_class_f32_e64 s4, v2, 0x1f8                          // 00000000629c: d47e0004 0201ff02 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000062a8: 8b010401
	v_mul_f64_e32 v[58:59], v[58:59], v[60:61]                 // 0000000062ac: 0c74793a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062b0: bf870091
	v_mul_f64_e32 v[58:59], v[58:59], v[62:63]                 // 0000000062b4: 0c747d3a
	v_cvt_f32_f64_e32 v1, v[58:59]                             // 0000000062b8: 7e021f3a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000062c0: bf870001
	v_cndmask_b32_e64 v60, v1, 0, s1                           // 0000000062c4: d501003c 00050101
	s_branch 62529                                             // 0000000062cc: bfa0f441 <packed_folded_w4a8+0x18d4>
	v_cvt_f64_f32_e32 v[1:2], v65                              // 0000000062d0: 7e022141
	v_cvt_f64_f32_e32 v[58:59], v69                            // 0000000062d4: 7e742145
	v_cvt_f64_f32_e32 v[61:62], v3                             // 0000000062d8: 7e7a2103
	v_cmp_eq_f32_e64 s1, 0, v65                                // 0000000062dc: d4120001 02028280
	v_cmp_class_f32_e64 s4, v3, 0x1f8                          // 0000000062e4: d47e0004 0201ff03 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000062f0: 8b010401
	v_mul_f64_e32 v[1:2], v[1:2], v[58:59]                     // 0000000062f4: 0c027501
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062f8: bf870091
	v_mul_f64_e32 v[1:2], v[1:2], v[61:62]                     // 0000000062fc: 0c027b01
	v_cvt_f32_f64_e32 v1, v[1:2]                               // 000000006300: 7e021f01
	s_wait_alu depctr_sa_sdst(0)                               // 000000006304: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006308: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s1                            // 00000000630c: d5010001 00050101
	s_branch 62523                                             // 000000006314: bfa0f43b <packed_folded_w4a8+0x1904>
	v_cvt_f64_f32_e32 v[131:132], v50                          // 000000006318: 7f062132
	v_cvt_f64_f32_e32 v[133:134], v63                          // 00000000631c: 7f0a213f
	v_cvt_f64_f32_e32 v[135:136], v4                           // 000000006320: 7f0e2104
	v_cmp_eq_f32_e64 s2, 0, v50                                // 000000006324: d4120002 02026480
	v_cmp_class_f32_e64 s5, v4, 0x1f8                          // 00000000632c: d47e0005 0201ff04 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006338: 8b020502
	v_mul_f64_e32 v[131:132], v[131:132], v[133:134]           // 00000000633c: 0d070b83
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006340: bf870091
	v_mul_f64_e32 v[131:132], v[131:132], v[135:136]           // 000000006344: 0d070f83
	v_cvt_f32_f64_e32 v61, v[131:132]                          // 000000006348: 7e7a1f83
	s_wait_alu depctr_sa_sdst(0)                               // 00000000634c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006350: bf870001
	v_cndmask_b32_e64 v61, v61, 0, s2                          // 000000006354: d501003d 0009013d
	s_branch 62933                                             // 00000000635c: bfa0f5d5 <packed_folded_w4a8+0x1fb4>
	v_cvt_f64_f32_e32 v[131:132], v51                          // 000000006360: 7f062133
	v_cvt_f64_f32_e32 v[133:134], v63                          // 000000006364: 7f0a213f
	v_cvt_f64_f32_e32 v[135:136], v5                           // 000000006368: 7f0e2105
	v_cmp_eq_f32_e64 s2, 0, v51                                // 00000000636c: d4120002 02026680
	v_cmp_class_f32_e64 s5, v5, 0x1f8                          // 000000006374: d47e0005 0201ff05 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006380: 8b020502
	v_mul_f64_e32 v[131:132], v[131:132], v[133:134]           // 000000006384: 0d070b83
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006388: bf870091
	v_mul_f64_e32 v[131:132], v[131:132], v[135:136]           // 00000000638c: 0d070f83
	v_cvt_f32_f64_e32 v4, v[131:132]                           // 000000006390: 7e081f83
	s_wait_alu depctr_sa_sdst(0)                               // 000000006394: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006398: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s2                            // 00000000639c: d5010004 00090104
	s_branch 62927                                             // 0000000063a4: bfa0f5cf <packed_folded_w4a8+0x1fe4>
	v_cvt_f64_f32_e32 v[50:51], v52                            // 0000000063a8: 7e642134
	v_cvt_f64_f32_e32 v[131:132], v63                          // 0000000063ac: 7f06213f
	v_cvt_f64_f32_e32 v[133:134], v6                           // 0000000063b0: 7f0a2106
	v_cmp_eq_f32_e64 s2, 0, v52                                // 0000000063b4: d4120002 02026880
	v_cmp_class_f32_e64 s5, v6, 0x1f8                          // 0000000063bc: d47e0005 0201ff06 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000063c8: 8b020502
	v_mul_f64_e32 v[50:51], v[50:51], v[131:132]               // 0000000063cc: 0c650732
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063d0: bf870091
	v_mul_f64_e32 v[50:51], v[50:51], v[133:134]               // 0000000063d4: 0c650b32
	v_cvt_f32_f64_e32 v5, v[50:51]                             // 0000000063d8: 7e0a1f32
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000063e0: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s2                            // 0000000063e4: d5010005 00090105
	s_branch 62921                                             // 0000000063ec: bfa0f5c9 <packed_folded_w4a8+0x2014>
	v_cvt_f64_f32_e32 v[50:51], v53                            // 0000000063f0: 7e642135
	v_cvt_f64_f32_e32 v[131:132], v63                          // 0000000063f4: 7f06213f
	v_cvt_f64_f32_e32 v[133:134], v7                           // 0000000063f8: 7f0a2107
	v_cmp_eq_f32_e64 s2, 0, v53                                // 0000000063fc: d4120002 02026a80
	v_cmp_class_f32_e64 s5, v7, 0x1f8                          // 000000006404: d47e0005 0201ff07 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006410: 8b020502
	v_mul_f64_e32 v[50:51], v[50:51], v[131:132]               // 000000006414: 0c650732
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006418: bf870091
	v_mul_f64_e32 v[50:51], v[50:51], v[133:134]               // 00000000641c: 0c650b32
	v_cvt_f32_f64_e32 v6, v[50:51]                             // 000000006420: 7e0c1f32
	s_wait_alu depctr_sa_sdst(0)                               // 000000006424: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006428: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s2                            // 00000000642c: d5010006 00090106
	s_branch 62915                                             // 000000006434: bfa0f5c3 <packed_folded_w4a8+0x2044>
	v_cvt_f64_f32_e32 v[50:51], v54                            // 000000006438: 7e642136
	v_cvt_f64_f32_e32 v[52:53], v63                            // 00000000643c: 7e68213f
	v_cvt_f64_f32_e32 v[131:132], v0                           // 000000006440: 7f062100
	v_cmp_eq_f32_e64 s2, 0, v54                                // 000000006444: d4120002 02026c80
	v_cmp_class_f32_e64 s5, v0, 0x1f8                          // 00000000644c: d47e0005 0201ff00 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006458: 8b020502
	v_mul_f64_e32 v[50:51], v[50:51], v[52:53]                 // 00000000645c: 0c646932
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006460: bf870091
	v_mul_f64_e32 v[50:51], v[50:51], v[131:132]               // 000000006464: 0c650732
	v_cvt_f32_f64_e32 v7, v[50:51]                             // 000000006468: 7e0e1f32
	s_wait_alu depctr_sa_sdst(0)                               // 00000000646c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006470: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s2                            // 000000006474: d5010007 00090107
	s_branch 62909                                             // 00000000647c: bfa0f5bd <packed_folded_w4a8+0x2074>
	v_cvt_f64_f32_e32 v[50:51], v55                            // 000000006480: 7e642137
	v_cvt_f64_f32_e32 v[52:53], v63                            // 000000006484: 7e68213f
	v_cvt_f64_f32_e32 v[131:132], v1                           // 000000006488: 7f062101
	v_cmp_eq_f32_e64 s2, 0, v55                                // 00000000648c: d4120002 02026e80
	v_cmp_class_f32_e64 s5, v1, 0x1f8                          // 000000006494: d47e0005 0201ff01 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000064a0: 8b020502
	v_mul_f64_e32 v[50:51], v[50:51], v[52:53]                 // 0000000064a4: 0c646932
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000064a8: bf870091
	v_mul_f64_e32 v[50:51], v[50:51], v[131:132]               // 0000000064ac: 0c650732
	v_cvt_f32_f64_e32 v0, v[50:51]                             // 0000000064b0: 7e001f32
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000064b8: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s2                            // 0000000064bc: d5010000 00090100
	s_branch 62903                                             // 0000000064c4: bfa0f5b7 <packed_folded_w4a8+0x20a4>
	v_cvt_f64_f32_e32 v[50:51], v56                            // 0000000064c8: 7e642138
	v_cvt_f64_f32_e32 v[52:53], v63                            // 0000000064cc: 7e68213f
	v_cvt_f64_f32_e32 v[54:55], v2                             // 0000000064d0: 7e6c2102
	v_cmp_eq_f32_e64 s2, 0, v56                                // 0000000064d4: d4120002 02027080
	v_cmp_class_f32_e64 s5, v2, 0x1f8                          // 0000000064dc: d47e0005 0201ff02 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000064e8: 8b020502
	v_mul_f64_e32 v[50:51], v[50:51], v[52:53]                 // 0000000064ec: 0c646932
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000064f0: bf870091
	v_mul_f64_e32 v[50:51], v[50:51], v[54:55]                 // 0000000064f4: 0c646d32
	v_cvt_f32_f64_e32 v1, v[50:51]                             // 0000000064f8: 7e021f32
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064fc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006500: bf870001
	v_cndmask_b32_e64 v52, v1, 0, s2                           // 000000006504: d5010034 00090101
	s_branch 62897                                             // 00000000650c: bfa0f5b1 <packed_folded_w4a8+0x20d4>
	v_cvt_f64_f32_e32 v[1:2], v57                              // 000000006510: 7e022139
	v_cvt_f64_f32_e32 v[50:51], v63                            // 000000006514: 7e64213f
	v_cvt_f64_f32_e32 v[53:54], v3                             // 000000006518: 7e6a2103
	v_cmp_eq_f32_e64 s2, 0, v57                                // 00000000651c: d4120002 02027280
	v_cmp_class_f32_e64 s5, v3, 0x1f8                          // 000000006524: d47e0005 0201ff03 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006530: 8b020502
	v_mul_f64_e32 v[1:2], v[1:2], v[50:51]                     // 000000006534: 0c026501
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006538: bf870091
	v_mul_f64_e32 v[1:2], v[1:2], v[53:54]                     // 00000000653c: 0c026b01
	v_cvt_f32_f64_e32 v1, v[1:2]                               // 000000006540: 7e021f01
	s_wait_alu depctr_sa_sdst(0)                               // 000000006544: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006548: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s2                            // 00000000654c: d5010001 00090101
	s_branch 62891                                             // 000000006554: bfa0f5ab <packed_folded_w4a8+0x2104>
	v_cvt_f64_f32_e32 v[154:155], v42                          // 000000006558: 7f34212a
	v_cvt_f64_f32_e32 v[156:157], v55                          // 00000000655c: 7f382137
	v_cvt_f64_f32_e32 v[158:159], v4                           // 000000006560: 7f3c2104
	v_cmp_eq_f32_e64 s3, 0, v42                                // 000000006564: d4120003 02025480
	v_cmp_class_f32_e64 s5, v4, 0x1f8                          // 00000000656c: d47e0005 0201ff04 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006578: 8b030503
	v_mul_f64_e32 v[154:155], v[154:155], v[156:157]           // 00000000657c: 0d35399a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006580: bf870091
	v_mul_f64_e32 v[154:155], v[154:155], v[158:159]           // 000000006584: 0d353d9a
	v_cvt_f32_f64_e32 v53, v[154:155]                          // 000000006588: 7e6a1f9a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000658c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006590: bf870001
	v_cndmask_b32_e64 v53, v53, 0, s3                          // 000000006594: d5010035 000d0135
	s_branch 63301                                             // 00000000659c: bfa0f745 <packed_folded_w4a8+0x27b4>
	v_cvt_f64_f32_e32 v[154:155], v43                          // 0000000065a0: 7f34212b
	v_cvt_f64_f32_e32 v[156:157], v55                          // 0000000065a4: 7f382137
	v_cvt_f64_f32_e32 v[158:159], v5                           // 0000000065a8: 7f3c2105
	v_cmp_eq_f32_e64 s3, 0, v43                                // 0000000065ac: d4120003 02025680
	v_cmp_class_f32_e64 s5, v5, 0x1f8                          // 0000000065b4: d47e0005 0201ff05 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000065c0: 8b030503
	v_mul_f64_e32 v[154:155], v[154:155], v[156:157]           // 0000000065c4: 0d35399a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065c8: bf870091
	v_mul_f64_e32 v[154:155], v[154:155], v[158:159]           // 0000000065cc: 0d353d9a
	v_cvt_f32_f64_e32 v4, v[154:155]                           // 0000000065d0: 7e081f9a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000065d8: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s3                            // 0000000065dc: d5010004 000d0104
	s_branch 63295                                             // 0000000065e4: bfa0f73f <packed_folded_w4a8+0x27e4>
	v_cvt_f64_f32_e32 v[42:43], v44                            // 0000000065e8: 7e54212c
	v_cvt_f64_f32_e32 v[154:155], v55                          // 0000000065ec: 7f342137
	v_cvt_f64_f32_e32 v[156:157], v6                           // 0000000065f0: 7f382106
	v_cmp_eq_f32_e64 s3, 0, v44                                // 0000000065f4: d4120003 02025880
	v_cmp_class_f32_e64 s5, v6, 0x1f8                          // 0000000065fc: d47e0005 0201ff06 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006608: 8b030503
	v_mul_f64_e32 v[42:43], v[42:43], v[154:155]               // 00000000660c: 0c55352a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006610: bf870091
	v_mul_f64_e32 v[42:43], v[42:43], v[156:157]               // 000000006614: 0c55392a
	v_cvt_f32_f64_e32 v5, v[42:43]                             // 000000006618: 7e0a1f2a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000661c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006620: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s3                            // 000000006624: d5010005 000d0105
	s_branch 63289                                             // 00000000662c: bfa0f739 <packed_folded_w4a8+0x2814>
	v_cvt_f64_f32_e32 v[42:43], v45                            // 000000006630: 7e54212d
	v_cvt_f64_f32_e32 v[154:155], v55                          // 000000006634: 7f342137
	v_cvt_f64_f32_e32 v[156:157], v7                           // 000000006638: 7f382107
	v_cmp_eq_f32_e64 s3, 0, v45                                // 00000000663c: d4120003 02025a80
	v_cmp_class_f32_e64 s5, v7, 0x1f8                          // 000000006644: d47e0005 0201ff07 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006650: 8b030503
	v_mul_f64_e32 v[42:43], v[42:43], v[154:155]               // 000000006654: 0c55352a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006658: bf870091
	v_mul_f64_e32 v[42:43], v[42:43], v[156:157]               // 00000000665c: 0c55392a
	v_cvt_f32_f64_e32 v6, v[42:43]                             // 000000006660: 7e0c1f2a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006664: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006668: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s3                            // 00000000666c: d5010006 000d0106
	s_branch 63283                                             // 000000006674: bfa0f733 <packed_folded_w4a8+0x2844>
	v_cvt_f64_f32_e32 v[42:43], v46                            // 000000006678: 7e54212e
	v_cvt_f64_f32_e32 v[44:45], v55                            // 00000000667c: 7e582137
	v_cvt_f64_f32_e32 v[154:155], v0                           // 000000006680: 7f342100
	v_cmp_eq_f32_e64 s3, 0, v46                                // 000000006684: d4120003 02025c80
	v_cmp_class_f32_e64 s5, v0, 0x1f8                          // 00000000668c: d47e0005 0201ff00 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006698: 8b030503
	v_mul_f64_e32 v[42:43], v[42:43], v[44:45]                 // 00000000669c: 0c54592a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066a0: bf870091
	v_mul_f64_e32 v[42:43], v[42:43], v[154:155]               // 0000000066a4: 0c55352a
	v_cvt_f32_f64_e32 v7, v[42:43]                             // 0000000066a8: 7e0e1f2a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066ac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066b0: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s3                            // 0000000066b4: d5010007 000d0107
	s_branch 63277                                             // 0000000066bc: bfa0f72d <packed_folded_w4a8+0x2874>
	v_cvt_f64_f32_e32 v[42:43], v47                            // 0000000066c0: 7e54212f
	v_cvt_f64_f32_e32 v[44:45], v55                            // 0000000066c4: 7e582137
	v_cvt_f64_f32_e32 v[154:155], v1                           // 0000000066c8: 7f342101
	v_cmp_eq_f32_e64 s3, 0, v47                                // 0000000066cc: d4120003 02025e80
	v_cmp_class_f32_e64 s5, v1, 0x1f8                          // 0000000066d4: d47e0005 0201ff01 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000066e0: 8b030503
	v_mul_f64_e32 v[42:43], v[42:43], v[44:45]                 // 0000000066e4: 0c54592a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066e8: bf870091
	v_mul_f64_e32 v[42:43], v[42:43], v[154:155]               // 0000000066ec: 0c55352a
	v_cvt_f32_f64_e32 v0, v[42:43]                             // 0000000066f0: 7e001f2a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066f4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066f8: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s3                            // 0000000066fc: d5010000 000d0100
	s_branch 63271                                             // 000000006704: bfa0f727 <packed_folded_w4a8+0x28a4>
	v_cvt_f64_f32_e32 v[42:43], v48                            // 000000006708: 7e542130
	v_cvt_f64_f32_e32 v[44:45], v55                            // 00000000670c: 7e582137
	v_cvt_f64_f32_e32 v[46:47], v2                             // 000000006710: 7e5c2102
	v_cmp_eq_f32_e64 s3, 0, v48                                // 000000006714: d4120003 02026080
	v_cmp_class_f32_e64 s5, v2, 0x1f8                          // 00000000671c: d47e0005 0201ff02 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006728: 8b030503
	v_mul_f64_e32 v[42:43], v[42:43], v[44:45]                 // 00000000672c: 0c54592a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006730: bf870091
	v_mul_f64_e32 v[42:43], v[42:43], v[46:47]                 // 000000006734: 0c545d2a
	v_cvt_f32_f64_e32 v1, v[42:43]                             // 000000006738: 7e021f2a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000673c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006740: bf870001
	v_cndmask_b32_e64 v57, v1, 0, s3                           // 000000006744: d5010039 000d0101
	s_branch 63265                                             // 00000000674c: bfa0f721 <packed_folded_w4a8+0x28d4>
	v_cvt_f64_f32_e32 v[1:2], v49                              // 000000006750: 7e022131
	v_cvt_f64_f32_e32 v[42:43], v55                            // 000000006754: 7e542137
	v_cvt_f64_f32_e32 v[44:45], v3                             // 000000006758: 7e582103
	v_cmp_eq_f32_e64 s3, 0, v49                                // 00000000675c: d4120003 02026280
	v_cmp_class_f32_e64 s5, v3, 0x1f8                          // 000000006764: d47e0005 0201ff03 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006770: 8b030503
	v_mul_f64_e32 v[1:2], v[1:2], v[42:43]                     // 000000006774: 0c025501
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006778: bf870091
	v_mul_f64_e32 v[1:2], v[1:2], v[44:45]                     // 00000000677c: 0c025901
	v_cvt_f32_f64_e32 v1, v[1:2]                               // 000000006780: 7e021f01
	s_wait_alu depctr_sa_sdst(0)                               // 000000006784: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006788: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s3                            // 00000000678c: d5010001 000d0101
	s_branch 63259                                             // 000000006794: bfa0f71b <packed_folded_w4a8+0x2904>
	v_cvt_f64_f32_e32 v[79:80], v34                            // 000000006798: 7e9e2122
	v_cvt_f64_f32_e32 v[117:118], v55                          // 00000000679c: 7eea2137
	v_cvt_f64_f32_e32 v[162:163], v4                           // 0000000067a0: 7f442104
	v_cmp_eq_f32_e32 vcc_lo, 0, v34                            // 0000000067a4: 7c244480
	v_cmp_class_f32_e64 s5, v4, 0x1f8                          // 0000000067a8: d47e0005 0201ff04 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000067b4: 8b05056a
	v_mul_f64_e32 v[79:80], v[79:80], v[117:118]               // 0000000067b8: 0c9eeb4f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000067bc: bf870091
	v_mul_f64_e32 v[79:80], v[79:80], v[162:163]               // 0000000067c0: 0c9f454f
	v_cvt_f32_f64_e32 v53, v[79:80]                            // 0000000067c4: 7e6a1f4f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067c8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000067cc: bf870001
	v_cndmask_b32_e64 v53, v53, 0, s5                          // 0000000067d0: d5010035 00150135
	s_branch 63644                                             // 0000000067d8: bfa0f89c <packed_folded_w4a8+0x2f4c>
	v_cvt_f64_f32_e32 v[79:80], v35                            // 0000000067dc: 7e9e2123
	v_cvt_f64_f32_e32 v[117:118], v55                          // 0000000067e0: 7eea2137
	v_cvt_f64_f32_e32 v[162:163], v5                           // 0000000067e4: 7f442105
	v_cmp_eq_f32_e32 vcc_lo, 0, v35                            // 0000000067e8: 7c244680
	v_cmp_class_f32_e64 s5, v5, 0x1f8                          // 0000000067ec: d47e0005 0201ff05 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000067f8: 8b05056a
	v_mul_f64_e32 v[79:80], v[79:80], v[117:118]               // 0000000067fc: 0c9eeb4f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006800: bf870091
	v_mul_f64_e32 v[79:80], v[79:80], v[162:163]               // 000000006804: 0c9f454f
	v_cvt_f32_f64_e32 v4, v[79:80]                             // 000000006808: 7e081f4f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000680c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006810: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s5                            // 000000006814: d5010004 00150104
	s_branch 63639                                             // 00000000681c: bfa0f897 <packed_folded_w4a8+0x2f7c>
	v_cvt_f64_f32_e32 v[34:35], v36                            // 000000006820: 7e442124
	v_cvt_f64_f32_e32 v[79:80], v55                            // 000000006824: 7e9e2137
	v_cvt_f64_f32_e32 v[117:118], v6                           // 000000006828: 7eea2106
	v_cmp_eq_f32_e32 vcc_lo, 0, v36                            // 00000000682c: 7c244880
	v_cmp_class_f32_e64 s5, v6, 0x1f8                          // 000000006830: d47e0005 0201ff06 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 00000000683c: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[79:80]                 // 000000006840: 0c449f22
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006844: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[117:118]               // 000000006848: 0c44eb22
	v_cvt_f32_f64_e32 v5, v[34:35]                             // 00000000684c: 7e0a1f22
	s_wait_alu depctr_sa_sdst(0)                               // 000000006850: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006854: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s5                            // 000000006858: d5010005 00150105
	s_branch 63634                                             // 000000006860: bfa0f892 <packed_folded_w4a8+0x2fac>
	v_cvt_f64_f32_e32 v[34:35], v37                            // 000000006864: 7e442125
	v_cvt_f64_f32_e32 v[79:80], v55                            // 000000006868: 7e9e2137
	v_cvt_f64_f32_e32 v[117:118], v7                           // 00000000686c: 7eea2107
	v_cmp_eq_f32_e32 vcc_lo, 0, v37                            // 000000006870: 7c244a80
	v_cmp_class_f32_e64 s5, v7, 0x1f8                          // 000000006874: d47e0005 0201ff07 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006880: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[79:80]                 // 000000006884: 0c449f22
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006888: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[117:118]               // 00000000688c: 0c44eb22
	v_cvt_f32_f64_e32 v6, v[34:35]                             // 000000006890: 7e0c1f22
	s_wait_alu depctr_sa_sdst(0)                               // 000000006894: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006898: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s5                            // 00000000689c: d5010006 00150106
	s_branch 63629                                             // 0000000068a4: bfa0f88d <packed_folded_w4a8+0x2fdc>
	v_cvt_f64_f32_e32 v[34:35], v38                            // 0000000068a8: 7e442126
	v_cvt_f64_f32_e32 v[36:37], v55                            // 0000000068ac: 7e482137
	v_cvt_f64_f32_e32 v[79:80], v0                             // 0000000068b0: 7e9e2100
	v_cmp_eq_f32_e32 vcc_lo, 0, v38                            // 0000000068b4: 7c244c80
	v_cmp_class_f32_e64 s5, v0, 0x1f8                          // 0000000068b8: d47e0005 0201ff00 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000068c4: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 0000000068c8: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000068cc: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[79:80]                 // 0000000068d0: 0c449f22
	v_cvt_f32_f64_e32 v7, v[34:35]                             // 0000000068d4: 7e0e1f22
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068d8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000068dc: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s5                            // 0000000068e0: d5010007 00150107
	s_branch 63624                                             // 0000000068e8: bfa0f888 <packed_folded_w4a8+0x300c>
	v_cvt_f64_f32_e32 v[34:35], v39                            // 0000000068ec: 7e442127
	v_cvt_f64_f32_e32 v[36:37], v55                            // 0000000068f0: 7e482137
	v_cvt_f64_f32_e32 v[79:80], v1                             // 0000000068f4: 7e9e2101
	v_cmp_eq_f32_e32 vcc_lo, 0, v39                            // 0000000068f8: 7c244e80
	v_cmp_class_f32_e64 s5, v1, 0x1f8                          // 0000000068fc: d47e0005 0201ff01 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006908: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 00000000690c: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006910: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[79:80]                 // 000000006914: 0c449f22
	v_cvt_f32_f64_e32 v0, v[34:35]                             // 000000006918: 7e001f22
	s_wait_alu depctr_sa_sdst(0)                               // 00000000691c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006920: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s5                            // 000000006924: d5010000 00150100
	s_branch 63619                                             // 00000000692c: bfa0f883 <packed_folded_w4a8+0x303c>
	v_cvt_f64_f32_e32 v[34:35], v40                            // 000000006930: 7e442128
	v_cvt_f64_f32_e32 v[36:37], v55                            // 000000006934: 7e482137
	v_cvt_f64_f32_e32 v[38:39], v2                             // 000000006938: 7e4c2102
	v_cmp_eq_f32_e32 vcc_lo, 0, v40                            // 00000000693c: 7c245080
	v_cmp_class_f32_e64 s5, v2, 0x1f8                          // 000000006940: d47e0005 0201ff02 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 00000000694c: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 000000006950: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006954: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 000000006958: 0c444d22
	v_cvt_f32_f64_e32 v1, v[34:35]                             // 00000000695c: 7e021f22
	s_wait_alu depctr_sa_sdst(0)                               // 000000006960: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006964: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s5                            // 000000006968: d5010001 00150101
	s_branch 63614                                             // 000000006970: bfa0f87e <packed_folded_w4a8+0x306c>
	v_cvt_f64_f32_e32 v[34:35], v41                            // 000000006974: 7e442129
	v_cvt_f64_f32_e32 v[36:37], v55                            // 000000006978: 7e482137
	v_cvt_f64_f32_e32 v[38:39], v3                             // 00000000697c: 7e4c2103
	v_cmp_eq_f32_e32 vcc_lo, 0, v41                            // 000000006980: 7c245280
	v_cmp_class_f32_e64 s5, v3, 0x1f8                          // 000000006984: d47e0005 0201ff03 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006990: 8b05056a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 000000006994: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006998: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 00000000699c: 0c444d22
	v_cvt_f32_f64_e32 v2, v[34:35]                             // 0000000069a0: 7e041f22
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069a8: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s5                            // 0000000069ac: d5010002 00150102
	s_branch 63609                                             // 0000000069b4: bfa0f879 <packed_folded_w4a8+0x309c>
	v_cvt_f64_f32_e32 v[34:35], v26                            // 0000000069b8: 7e44211a
	v_cvt_f64_f32_e32 v[36:37], v9                             // 0000000069bc: 7e482109
	v_cvt_f64_f32_e32 v[38:39], v4                             // 0000000069c0: 7e4c2104
	v_cmp_eq_f32_e32 vcc_lo, 0, v26                            // 0000000069c4: 7c243480
	v_cmp_class_f32_e64 s4, v4, 0x1f8                          // 0000000069c8: d47e0004 0201ff04 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 0000000069d4: 8b04046a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 0000000069d8: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069dc: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 0000000069e0: 0c444d22
	v_cvt_f32_f64_e32 v8, v[34:35]                             // 0000000069e4: 7e101f22
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069e8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069ec: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s4                            // 0000000069f0: d5010008 00110108
	s_branch 63877                                             // 0000000069f8: bfa0f985 <packed_folded_w4a8+0x3510>
	v_cvt_f64_f32_e32 v[34:35], v27                            // 0000000069fc: 7e44211b
	v_cvt_f64_f32_e32 v[36:37], v9                             // 000000006a00: 7e482109
	v_cvt_f64_f32_e32 v[38:39], v5                             // 000000006a04: 7e4c2105
	v_cmp_eq_f32_e32 vcc_lo, 0, v27                            // 000000006a08: 7c243680
	v_cmp_class_f32_e64 s4, v5, 0x1f8                          // 000000006a0c: d47e0004 0201ff05 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006a18: 8b04046a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 000000006a1c: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a20: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 000000006a24: 0c444d22
	v_cvt_f32_f64_e32 v4, v[34:35]                             // 000000006a28: 7e081f22
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a2c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a30: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s4                            // 000000006a34: d5010004 00110104
	s_branch 63872                                             // 000000006a3c: bfa0f980 <packed_folded_w4a8+0x3540>
	v_cvt_f64_f32_e32 v[26:27], v28                            // 000000006a40: 7e34211c
	v_cvt_f64_f32_e32 v[34:35], v9                             // 000000006a44: 7e442109
	v_cvt_f64_f32_e32 v[36:37], v6                             // 000000006a48: 7e482106
	v_cmp_eq_f32_e32 vcc_lo, 0, v28                            // 000000006a4c: 7c243880
	v_cmp_class_f32_e64 s4, v6, 0x1f8                          // 000000006a50: d47e0004 0201ff06 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006a5c: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[34:35]                 // 000000006a60: 0c34451a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a64: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[36:37]                 // 000000006a68: 0c34491a
	v_cvt_f32_f64_e32 v5, v[26:27]                             // 000000006a6c: 7e0a1f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a70: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a74: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s4                            // 000000006a78: d5010005 00110105
	s_branch 63867                                             // 000000006a80: bfa0f97b <packed_folded_w4a8+0x3570>
	v_cvt_f64_f32_e32 v[26:27], v29                            // 000000006a84: 7e34211d
	v_cvt_f64_f32_e32 v[34:35], v9                             // 000000006a88: 7e442109
	v_cvt_f64_f32_e32 v[36:37], v7                             // 000000006a8c: 7e482107
	v_cmp_eq_f32_e32 vcc_lo, 0, v29                            // 000000006a90: 7c243a80
	v_cmp_class_f32_e64 s4, v7, 0x1f8                          // 000000006a94: d47e0004 0201ff07 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006aa0: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[34:35]                 // 000000006aa4: 0c34451a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006aa8: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[36:37]                 // 000000006aac: 0c34491a
	v_cvt_f32_f64_e32 v6, v[26:27]                             // 000000006ab0: 7e0c1f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ab4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ab8: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s4                            // 000000006abc: d5010006 00110106
	s_branch 63862                                             // 000000006ac4: bfa0f976 <packed_folded_w4a8+0x35a0>
	v_cvt_f64_f32_e32 v[26:27], v30                            // 000000006ac8: 7e34211e
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006acc: 7e382109
	v_cvt_f64_f32_e32 v[34:35], v0                             // 000000006ad0: 7e442100
	v_cmp_eq_f32_e32 vcc_lo, 0, v30                            // 000000006ad4: 7c243c80
	v_cmp_class_f32_e64 s4, v0, 0x1f8                          // 000000006ad8: d47e0004 0201ff00 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006ae4: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006ae8: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006aec: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[34:35]                 // 000000006af0: 0c34451a
	v_cvt_f32_f64_e32 v7, v[26:27]                             // 000000006af4: 7e0e1f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006af8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006afc: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s4                            // 000000006b00: d5010007 00110107
	s_branch 63857                                             // 000000006b08: bfa0f971 <packed_folded_w4a8+0x35d0>
	v_cvt_f64_f32_e32 v[26:27], v31                            // 000000006b0c: 7e34211f
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006b10: 7e382109
	v_cvt_f64_f32_e32 v[34:35], v1                             // 000000006b14: 7e442101
	v_cmp_eq_f32_e32 vcc_lo, 0, v31                            // 000000006b18: 7c243e80
	v_cmp_class_f32_e64 s4, v1, 0x1f8                          // 000000006b1c: d47e0004 0201ff01 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006b28: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006b2c: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b30: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[34:35]                 // 000000006b34: 0c34451a
	v_cvt_f32_f64_e32 v0, v[26:27]                             // 000000006b38: 7e001f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b3c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b40: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s4                            // 000000006b44: d5010000 00110100
	s_branch 63852                                             // 000000006b4c: bfa0f96c <packed_folded_w4a8+0x3600>
	v_cvt_f64_f32_e32 v[26:27], v32                            // 000000006b50: 7e342120
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006b54: 7e382109
	v_cvt_f64_f32_e32 v[30:31], v2                             // 000000006b58: 7e3c2102
	v_cmp_eq_f32_e32 vcc_lo, 0, v32                            // 000000006b5c: 7c244080
	v_cmp_class_f32_e64 s4, v2, 0x1f8                          // 000000006b60: d47e0004 0201ff02 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006b6c: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006b70: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b74: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006b78: 0c343d1a
	v_cvt_f32_f64_e32 v1, v[26:27]                             // 000000006b7c: 7e021f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b80: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b84: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s4                            // 000000006b88: d5010001 00110101
	s_branch 63847                                             // 000000006b90: bfa0f967 <packed_folded_w4a8+0x3630>
	v_cvt_f64_f32_e32 v[26:27], v33                            // 000000006b94: 7e342121
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006b98: 7e382109
	v_cvt_f64_f32_e32 v[30:31], v3                             // 000000006b9c: 7e3c2103
	v_cmp_eq_f32_e32 vcc_lo, 0, v33                            // 000000006ba0: 7c244280
	v_cmp_class_f32_e64 s4, v3, 0x1f8                          // 000000006ba4: d47e0004 0201ff03 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006bb0: 8b04046a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006bb4: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bb8: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006bbc: 0c343d1a
	v_cvt_f32_f64_e32 v2, v[26:27]                             // 000000006bc0: 7e041f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006bc4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006bc8: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s4                            // 000000006bcc: d5010002 00110102
	s_branch 63842                                             // 000000006bd4: bfa0f962 <packed_folded_w4a8+0x3660>
	v_cvt_f64_f32_e32 v[26:27], v18                            // 000000006bd8: 7e342112
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006bdc: 7e382109
	v_cvt_f64_f32_e32 v[30:31], v4                             // 000000006be0: 7e3c2104
	v_cmp_eq_f32_e32 vcc_lo, 0, v18                            // 000000006be4: 7c242480
	v_cmp_class_f32_e64 s1, v4, 0x1f8                          // 000000006be8: d47e0001 0201ff04 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006bf4: 8b01016a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006bf8: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bfc: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006c00: 0c343d1a
	v_cvt_f32_f64_e32 v8, v[26:27]                             // 000000006c04: 7e101f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c08: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c0c: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s1                            // 000000006c10: d5010008 00050108
	s_branch 64109                                             // 000000006c18: bfa0fa6d <packed_folded_w4a8+0x3ad0>
	v_cvt_f64_f32_e32 v[26:27], v19                            // 000000006c1c: 7e342113
	v_cvt_f64_f32_e32 v[28:29], v9                             // 000000006c20: 7e382109
	v_cvt_f64_f32_e32 v[30:31], v5                             // 000000006c24: 7e3c2105
	v_cmp_eq_f32_e32 vcc_lo, 0, v19                            // 000000006c28: 7c242680
	v_cmp_class_f32_e64 s1, v5, 0x1f8                          // 000000006c2c: d47e0001 0201ff05 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006c38: 8b01016a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006c3c: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c40: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006c44: 0c343d1a
	v_cvt_f32_f64_e32 v4, v[26:27]                             // 000000006c48: 7e081f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c4c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c50: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s1                            // 000000006c54: d5010004 00050104
	s_branch 64104                                             // 000000006c5c: bfa0fa68 <packed_folded_w4a8+0x3b00>
	v_cvt_f64_f32_e32 v[18:19], v20                            // 000000006c60: 7e242114
	v_cvt_f64_f32_e32 v[26:27], v9                             // 000000006c64: 7e342109
	v_cvt_f64_f32_e32 v[28:29], v6                             // 000000006c68: 7e382106
	v_cmp_eq_f32_e32 vcc_lo, 0, v20                            // 000000006c6c: 7c242880
	v_cmp_class_f32_e64 s1, v6, 0x1f8                          // 000000006c70: d47e0001 0201ff06 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006c7c: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[26:27]                 // 000000006c80: 0c243512
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c84: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[28:29]                 // 000000006c88: 0c243912
	v_cvt_f32_f64_e32 v5, v[18:19]                             // 000000006c8c: 7e0a1f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c90: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c94: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s1                            // 000000006c98: d5010005 00050105
	s_branch 64099                                             // 000000006ca0: bfa0fa63 <packed_folded_w4a8+0x3b30>
	v_cvt_f64_f32_e32 v[18:19], v21                            // 000000006ca4: 7e242115
	v_cvt_f64_f32_e32 v[26:27], v9                             // 000000006ca8: 7e342109
	v_cvt_f64_f32_e32 v[28:29], v7                             // 000000006cac: 7e382107
	v_cmp_eq_f32_e32 vcc_lo, 0, v21                            // 000000006cb0: 7c242a80
	v_cmp_class_f32_e64 s1, v7, 0x1f8                          // 000000006cb4: d47e0001 0201ff07 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006cc0: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[26:27]                 // 000000006cc4: 0c243512
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006cc8: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[28:29]                 // 000000006ccc: 0c243912
	v_cvt_f32_f64_e32 v6, v[18:19]                             // 000000006cd0: 7e0c1f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006cd4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006cd8: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s1                            // 000000006cdc: d5010006 00050106
	s_branch 64094                                             // 000000006ce4: bfa0fa5e <packed_folded_w4a8+0x3b60>
	v_cvt_f64_f32_e32 v[18:19], v22                            // 000000006ce8: 7e242116
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006cec: 7e282109
	v_cvt_f64_f32_e32 v[26:27], v0                             // 000000006cf0: 7e342100
	v_cmp_eq_f32_e32 vcc_lo, 0, v22                            // 000000006cf4: 7c242c80
	v_cmp_class_f32_e64 s1, v0, 0x1f8                          // 000000006cf8: d47e0001 0201ff00 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d04: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006d08: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d0c: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[26:27]                 // 000000006d10: 0c243512
	v_cvt_f32_f64_e32 v7, v[18:19]                             // 000000006d14: 7e0e1f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d18: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d1c: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s1                            // 000000006d20: d5010007 00050107
	s_branch 64089                                             // 000000006d28: bfa0fa59 <packed_folded_w4a8+0x3b90>
	v_cvt_f64_f32_e32 v[18:19], v23                            // 000000006d2c: 7e242117
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006d30: 7e282109
	v_cvt_f64_f32_e32 v[26:27], v1                             // 000000006d34: 7e342101
	v_cmp_eq_f32_e32 vcc_lo, 0, v23                            // 000000006d38: 7c242e80
	v_cmp_class_f32_e64 s1, v1, 0x1f8                          // 000000006d3c: d47e0001 0201ff01 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d48: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006d4c: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d50: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[26:27]                 // 000000006d54: 0c243512
	v_cvt_f32_f64_e32 v0, v[18:19]                             // 000000006d58: 7e001f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d5c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d60: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s1                            // 000000006d64: d5010000 00050100
	s_branch 64084                                             // 000000006d6c: bfa0fa54 <packed_folded_w4a8+0x3bc0>
	v_cvt_f64_f32_e32 v[18:19], v24                            // 000000006d70: 7e242118
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006d74: 7e282109
	v_cvt_f64_f32_e32 v[22:23], v2                             // 000000006d78: 7e2c2102
	v_cmp_eq_f32_e32 vcc_lo, 0, v24                            // 000000006d7c: 7c243080
	v_cmp_class_f32_e64 s1, v2, 0x1f8                          // 000000006d80: d47e0001 0201ff02 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d8c: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006d90: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d94: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006d98: 0c242d12
	v_cvt_f32_f64_e32 v1, v[18:19]                             // 000000006d9c: 7e021f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006da0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006da4: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s1                            // 000000006da8: d5010001 00050101
	s_branch 64079                                             // 000000006db0: bfa0fa4f <packed_folded_w4a8+0x3bf0>
	v_cvt_f64_f32_e32 v[18:19], v25                            // 000000006db4: 7e242119
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006db8: 7e282109
	v_cvt_f64_f32_e32 v[22:23], v3                             // 000000006dbc: 7e2c2103
	v_cmp_eq_f32_e32 vcc_lo, 0, v25                            // 000000006dc0: 7c243280
	v_cmp_class_f32_e64 s1, v3, 0x1f8                          // 000000006dc4: d47e0001 0201ff03 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006dd0: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006dd4: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006dd8: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006ddc: 0c242d12
	v_cvt_f32_f64_e32 v2, v[18:19]                             // 000000006de0: 7e041f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006de4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006de8: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s1                            // 000000006dec: d5010002 00050102
	s_branch 64074                                             // 000000006df4: bfa0fa4a <packed_folded_w4a8+0x3c20>
	v_cvt_f64_f32_e32 v[18:19], v10                            // 000000006df8: 7e24210a
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006dfc: 7e282109
	v_cvt_f64_f32_e32 v[22:23], v4                             // 000000006e00: 7e2c2104
	v_cmp_eq_f32_e32 vcc_lo, 0, v10                            // 000000006e04: 7c241480
	v_cmp_class_f32_e64 s1, v4, 0x1f8                          // 000000006e08: d47e0001 0201ff04 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e14: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006e18: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e1c: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006e20: 0c242d12
	v_cvt_f32_f64_e32 v8, v[18:19]                             // 000000006e24: 7e101f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e28: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e2c: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s1                            // 000000006e30: d5010008 00050108
	s_branch 64341                                             // 000000006e38: bfa0fb55 <packed_folded_w4a8+0x4090>
	v_cvt_f64_f32_e32 v[18:19], v11                            // 000000006e3c: 7e24210b
	v_cvt_f64_f32_e32 v[20:21], v9                             // 000000006e40: 7e282109
	v_cvt_f64_f32_e32 v[22:23], v5                             // 000000006e44: 7e2c2105
	v_cmp_eq_f32_e32 vcc_lo, 0, v11                            // 000000006e48: 7c241680
	v_cmp_class_f32_e64 s1, v5, 0x1f8                          // 000000006e4c: d47e0001 0201ff05 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e58: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006e5c: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e60: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006e64: 0c242d12
	v_cvt_f32_f64_e32 v4, v[18:19]                             // 000000006e68: 7e081f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e6c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e70: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s1                            // 000000006e74: d5010004 00050104
	s_branch 64336                                             // 000000006e7c: bfa0fb50 <packed_folded_w4a8+0x40c0>
	v_cvt_f64_f32_e32 v[10:11], v12                            // 000000006e80: 7e14210c
	v_cvt_f64_f32_e32 v[18:19], v9                             // 000000006e84: 7e242109
	v_cvt_f64_f32_e32 v[20:21], v6                             // 000000006e88: 7e282106
	v_cmp_eq_f32_e32 vcc_lo, 0, v12                            // 000000006e8c: 7c241880
	v_cmp_class_f32_e64 s1, v6, 0x1f8                          // 000000006e90: d47e0001 0201ff06 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e9c: 8b01016a
	v_mul_f64_e32 v[10:11], v[10:11], v[18:19]                 // 000000006ea0: 0c14250a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ea4: bf870091
	v_mul_f64_e32 v[10:11], v[10:11], v[20:21]                 // 000000006ea8: 0c14290a
	v_cvt_f32_f64_e32 v5, v[10:11]                             // 000000006eac: 7e0a1f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006eb0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006eb4: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s1                            // 000000006eb8: d5010005 00050105
	s_branch 64331                                             // 000000006ec0: bfa0fb4b <packed_folded_w4a8+0x40f0>
	v_cvt_f64_f32_e32 v[10:11], v13                            // 000000006ec4: 7e14210d
	v_cvt_f64_f32_e32 v[18:19], v9                             // 000000006ec8: 7e242109
	v_cvt_f64_f32_e32 v[20:21], v7                             // 000000006ecc: 7e282107
	v_cmp_eq_f32_e32 vcc_lo, 0, v13                            // 000000006ed0: 7c241a80
	v_cmp_class_f32_e64 s1, v7, 0x1f8                          // 000000006ed4: d47e0001 0201ff07 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006ee0: 8b01016a
	v_mul_f64_e32 v[10:11], v[10:11], v[18:19]                 // 000000006ee4: 0c14250a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ee8: bf870091
	v_mul_f64_e32 v[10:11], v[10:11], v[20:21]                 // 000000006eec: 0c14290a
	v_cvt_f32_f64_e32 v6, v[10:11]                             // 000000006ef0: 7e0c1f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ef4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ef8: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s1                            // 000000006efc: d5010006 00050106
	s_branch 64326                                             // 000000006f04: bfa0fb46 <packed_folded_w4a8+0x4120>
	v_cvt_f64_f32_e32 v[10:11], v14                            // 000000006f08: 7e14210e
	v_cvt_f64_f32_e32 v[12:13], v9                             // 000000006f0c: 7e182109
	v_cvt_f64_f32_e32 v[18:19], v0                             // 000000006f10: 7e242100
	v_cmp_eq_f32_e32 vcc_lo, 0, v14                            // 000000006f14: 7c241c80
	v_cmp_class_f32_e64 s1, v0, 0x1f8                          // 000000006f18: d47e0001 0201ff00 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006f24: 8b01016a
	v_mul_f64_e32 v[10:11], v[10:11], v[12:13]                 // 000000006f28: 0c14190a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f2c: bf870091
	v_mul_f64_e32 v[10:11], v[10:11], v[18:19]                 // 000000006f30: 0c14250a
	v_cvt_f32_f64_e32 v7, v[10:11]                             // 000000006f34: 7e0e1f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f38: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f3c: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s1                            // 000000006f40: d5010007 00050107
	s_branch 64321                                             // 000000006f48: bfa0fb41 <packed_folded_w4a8+0x4150>
	v_cvt_f64_f32_e32 v[10:11], v15                            // 000000006f4c: 7e14210f
	v_cvt_f64_f32_e32 v[12:13], v9                             // 000000006f50: 7e182109
	v_cvt_f64_f32_e32 v[18:19], v1                             // 000000006f54: 7e242101
	v_cmp_eq_f32_e32 vcc_lo, 0, v15                            // 000000006f58: 7c241e80
	v_cmp_class_f32_e64 s1, v1, 0x1f8                          // 000000006f5c: d47e0001 0201ff01 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006f68: 8b01016a
	v_mul_f64_e32 v[10:11], v[10:11], v[12:13]                 // 000000006f6c: 0c14190a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f70: bf870091
	v_mul_f64_e32 v[10:11], v[10:11], v[18:19]                 // 000000006f74: 0c14250a
	v_cvt_f32_f64_e32 v0, v[10:11]                             // 000000006f78: 7e001f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f7c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f80: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s1                            // 000000006f84: d5010000 00050100
	s_branch 64316                                             // 000000006f8c: bfa0fb3c <packed_folded_w4a8+0x4180>
	v_cvt_f64_f32_e32 v[10:11], v16                            // 000000006f90: 7e142110
	v_cvt_f64_f32_e32 v[12:13], v9                             // 000000006f94: 7e182109
	v_cvt_f64_f32_e32 v[14:15], v2                             // 000000006f98: 7e1c2102
	v_cmp_eq_f32_e32 vcc_lo, 0, v16                            // 000000006f9c: 7c242080
	v_cmp_class_f32_e64 s1, v2, 0x1f8                          // 000000006fa0: d47e0001 0201ff02 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006fac: 8b01016a
	v_mul_f64_e32 v[10:11], v[10:11], v[12:13]                 // 000000006fb0: 0c14190a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006fb4: bf870091
	v_mul_f64_e32 v[10:11], v[10:11], v[14:15]                 // 000000006fb8: 0c141d0a
	v_cvt_f32_f64_e32 v1, v[10:11]                             // 000000006fbc: 7e021f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006fc0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006fc4: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s1                            // 000000006fc8: d5010001 00050101
	s_branch 64311                                             // 000000006fd0: bfa0fb37 <packed_folded_w4a8+0x41b0>
	v_cvt_f64_f32_e32 v[10:11], v17                            // 000000006fd4: 7e142111
	v_cvt_f64_f32_e32 v[12:13], v9                             // 000000006fd8: 7e182109
	v_cvt_f64_f32_e32 v[14:15], v3                             // 000000006fdc: 7e1c2103
	v_cmp_eq_f32_e32 vcc_lo, 0, v17                            // 000000006fe0: 7c242280
	v_cmp_class_f32_e64 s1, v3, 0x1f8                          // 000000006fe4: d47e0001 0201ff03 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006ff0: 8b01016a
	v_mul_f64_e32 v[9:10], v[10:11], v[12:13]                  // 000000006ff4: 0c12190a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ff8: bf870091
	v_mul_f64_e32 v[9:10], v[9:10], v[14:15]                   // 000000006ffc: 0c121d09
	v_cvt_f32_f64_e32 v2, v[9:10]                              // 000000007000: 7e041f09
	s_wait_alu depctr_sa_sdst(0)                               // 000000007004: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007008: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s1                            // 00000000700c: d5010002 00050102
	s_branch 64306                                             // 000000007014: bfa0fb32 <packed_folded_w4a8+0x41e0>
	s_code_end                                                 // 000000007018: bf9f0000
	s_code_end                                                 // 00000000701c: bf9f0000
	s_code_end                                                 // 000000007020: bf9f0000
	s_code_end                                                 // 000000007024: bf9f0000
	s_code_end                                                 // 000000007028: bf9f0000
	s_code_end                                                 // 00000000702c: bf9f0000
	s_code_end                                                 // 000000007030: bf9f0000
	s_code_end                                                 // 000000007034: bf9f0000
	s_code_end                                                 // 000000007038: bf9f0000
	s_code_end                                                 // 00000000703c: bf9f0000
	s_code_end                                                 // 000000007040: bf9f0000
	s_code_end                                                 // 000000007044: bf9f0000
	s_code_end                                                 // 000000007048: bf9f0000
	s_code_end                                                 // 00000000704c: bf9f0000
	s_code_end                                                 // 000000007050: bf9f0000
	s_code_end                                                 // 000000007054: bf9f0000
	s_code_end                                                 // 000000007058: bf9f0000
	s_code_end                                                 // 00000000705c: bf9f0000
	s_code_end                                                 // 000000007060: bf9f0000
	s_code_end                                                 // 000000007064: bf9f0000
	s_code_end                                                 // 000000007068: bf9f0000
	s_code_end                                                 // 00000000706c: bf9f0000
	s_code_end                                                 // 000000007070: bf9f0000
	s_code_end                                                 // 000000007074: bf9f0000
	s_code_end                                                 // 000000007078: bf9f0000
	s_code_end                                                 // 00000000707c: bf9f0000
	s_code_end                                                 // 000000007080: bf9f0000
	s_code_end                                                 // 000000007084: bf9f0000
	s_code_end                                                 // 000000007088: bf9f0000
	s_code_end                                                 // 00000000708c: bf9f0000
	s_code_end                                                 // 000000007090: bf9f0000
	s_code_end                                                 // 000000007094: bf9f0000
	s_code_end                                                 // 000000007098: bf9f0000
	s_code_end                                                 // 00000000709c: bf9f0000
	s_code_end                                                 // 0000000070a0: bf9f0000
	s_code_end                                                 // 0000000070a4: bf9f0000
	s_code_end                                                 // 0000000070a8: bf9f0000
	s_code_end                                                 // 0000000070ac: bf9f0000
	s_code_end                                                 // 0000000070b0: bf9f0000
	s_code_end                                                 // 0000000070b4: bf9f0000
	s_code_end                                                 // 0000000070b8: bf9f0000
	s_code_end                                                 // 0000000070bc: bf9f0000
	s_code_end                                                 // 0000000070c0: bf9f0000
	s_code_end                                                 // 0000000070c4: bf9f0000
	s_code_end                                                 // 0000000070c8: bf9f0000
	s_code_end                                                 // 0000000070cc: bf9f0000
	s_code_end                                                 // 0000000070d0: bf9f0000
	s_code_end                                                 // 0000000070d4: bf9f0000
	s_code_end                                                 // 0000000070d8: bf9f0000
	s_code_end                                                 // 0000000070dc: bf9f0000
	s_code_end                                                 // 0000000070e0: bf9f0000
	s_code_end                                                 // 0000000070e4: bf9f0000
	s_code_end                                                 // 0000000070e8: bf9f0000
	s_code_end                                                 // 0000000070ec: bf9f0000
	s_code_end                                                 // 0000000070f0: bf9f0000
	s_code_end                                                 // 0000000070f4: bf9f0000
	s_code_end                                                 // 0000000070f8: bf9f0000
	s_code_end                                                 // 0000000070fc: bf9f0000
	s_code_end                                                 // 000000007100: bf9f0000
	s_code_end                                                 // 000000007104: bf9f0000
	s_code_end                                                 // 000000007108: bf9f0000
	s_code_end                                                 // 00000000710c: bf9f0000
	s_code_end                                                 // 000000007110: bf9f0000
	s_code_end                                                 // 000000007114: bf9f0000
	s_code_end                                                 // 000000007118: bf9f0000
	s_code_end                                                 // 00000000711c: bf9f0000
	s_code_end                                                 // 000000007120: bf9f0000
	s_code_end                                                 // 000000007124: bf9f0000
	s_code_end                                                 // 000000007128: bf9f0000
	s_code_end                                                 // 00000000712c: bf9f0000
	s_code_end                                                 // 000000007130: bf9f0000
	s_code_end                                                 // 000000007134: bf9f0000
	s_code_end                                                 // 000000007138: bf9f0000
	s_code_end                                                 // 00000000713c: bf9f0000
	s_code_end                                                 // 000000007140: bf9f0000
	s_code_end                                                 // 000000007144: bf9f0000
	s_code_end                                                 // 000000007148: bf9f0000
	s_code_end                                                 // 00000000714c: bf9f0000
	s_code_end                                                 // 000000007150: bf9f0000
	s_code_end                                                 // 000000007154: bf9f0000
	s_code_end                                                 // 000000007158: bf9f0000
	s_code_end                                                 // 00000000715c: bf9f0000
	s_code_end                                                 // 000000007160: bf9f0000
	s_code_end                                                 // 000000007164: bf9f0000
	s_code_end                                                 // 000000007168: bf9f0000
	s_code_end                                                 // 00000000716c: bf9f0000
	s_code_end                                                 // 000000007170: bf9f0000
	s_code_end                                                 // 000000007174: bf9f0000
	s_code_end                                                 // 000000007178: bf9f0000
	s_code_end                                                 // 00000000717c: bf9f0000
	s_code_end                                                 // 000000007180: bf9f0000
	s_code_end                                                 // 000000007184: bf9f0000
	s_code_end                                                 // 000000007188: bf9f0000
	s_code_end                                                 // 00000000718c: bf9f0000
	s_code_end                                                 // 000000007190: bf9f0000
	s_code_end                                                 // 000000007194: bf9f0000
	s_code_end                                                 // 000000007198: bf9f0000
	s_code_end                                                 // 00000000719c: bf9f0000
	s_code_end                                                 // 0000000071a0: bf9f0000
	s_code_end                                                 // 0000000071a4: bf9f0000
	s_code_end                                                 // 0000000071a8: bf9f0000
	s_code_end                                                 // 0000000071ac: bf9f0000
	s_code_end                                                 // 0000000071b0: bf9f0000
	s_code_end                                                 // 0000000071b4: bf9f0000
	s_code_end                                                 // 0000000071b8: bf9f0000
	s_code_end                                                 // 0000000071bc: bf9f0000
	s_code_end                                                 // 0000000071c0: bf9f0000
	s_code_end                                                 // 0000000071c4: bf9f0000
	s_code_end                                                 // 0000000071c8: bf9f0000
	s_code_end                                                 // 0000000071cc: bf9f0000
	s_code_end                                                 // 0000000071d0: bf9f0000
	s_code_end                                                 // 0000000071d4: bf9f0000
	s_code_end                                                 // 0000000071d8: bf9f0000
	s_code_end                                                 // 0000000071dc: bf9f0000
	s_code_end                                                 // 0000000071e0: bf9f0000
	s_code_end                                                 // 0000000071e4: bf9f0000
	s_code_end                                                 // 0000000071e8: bf9f0000
	s_code_end                                                 // 0000000071ec: bf9f0000
	s_code_end                                                 // 0000000071f0: bf9f0000
	s_code_end                                                 // 0000000071f4: bf9f0000
	s_code_end                                                 // 0000000071f8: bf9f0000
	s_code_end                                                 // 0000000071fc: bf9f0000
