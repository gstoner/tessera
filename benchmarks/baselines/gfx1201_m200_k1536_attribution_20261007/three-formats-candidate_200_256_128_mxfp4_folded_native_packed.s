
/tmp/tmpkneqhrpj.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <packed_folded_w4a8>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b64 s[6:7], s[0:1], 0xd8                            // 000000001b04: f4002180 f80000d8
	s_load_b128 s[36:39], s[0:1], 0xc8                         // 000000001b0c: f4004900 f80000c8
	v_lshrrev_b32_e32 v8, 2, v0                                // 000000001b14: 32100082
	s_mov_b32 s10, ttmp7                                       // 000000001b18: be8a0073
	s_ashr_i32 s11, ttmp7, 31                                  // 000000001b1c: 860b9f73
	s_clause 0x2                                               // 000000001b20: bf850002
	s_load_b64 s[12:13], s[0:1], 0x8                           // 000000001b24: f4002300 f8000008
	s_load_b64 s[4:5], s[0:1], 0x30                            // 000000001b2c: f4002100 f8000030
	s_load_b64 s[8:9], s[0:1], 0x80                            // 000000001b34: f4002200 f8000080
	s_lshl_b64 s[26:27], s[10:11], 8                           // 000000001b3c: 849a880a
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b40: bf870009
	v_dual_mov_b32 v9, s27 :: v_dual_and_b32 v70, 0xc0, v0     // 000000001b44: ca24001b 094600ff 000000c0
	v_dual_mov_b32 v2, s27 :: v_dual_mov_b32 v3, s27           // 000000001b50: ca10001b 0202001b
	v_or_b32_e32 v1, s26, v8                                   // 000000001b58: 3802101a
	v_mul_u32_u24_e32 v11, 0x50, v8                            // 000000001b5c: 161610ff 00000050
	v_dual_mov_b32 v56, 0 :: v_dual_and_b32 v71, 15, v0        // 000000001b64: ca240080 3846008f
	v_mov_b32_e32 v6, s27                                      // 000000001b6c: 7e0c021b
	s_delay_alu instid0(valu_dep_4)                            // 000000001b70: bf870004
	v_or_b32_e32 v5, 0xc0, v1                                  // 000000001b74: 380a02ff 000000c0
	s_mov_b32 s2, ttmp9                                        // 000000001b7c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b80: 86039f75
	v_or_b32_e32 v90, 16, v70                                  // 000000001b84: 38b48c90
	s_lshl_b64 s[16:17], s[2:3], 6                             // 000000001b88: 84908602
	s_wait_kmcnt 0x0                                           // 000000001b8c: bfc70000
	s_add_nc_u64 s[14:15], s[36:37], -1                        // 000000001b90: a98ec124
	s_lshr_b64 s[2:3], s[6:7], 5                               // 000000001b94: 85828506
	v_cmp_gt_u64_e32 vcc_lo, s[14:15], v[1:2]                  // 000000001b98: 7cb8020e
	v_or_b32_e32 v2, 64, v1                                    // 000000001b9c: 380402c0
	v_and_b32_e32 v124, 32, v0                                 // 000000001ba0: 36f800a0
	s_mul_u64 s[10:11], s[2:3], s[38:39]                       // 000000001ba4: aa8a2602
	v_cmp_gt_u64_e64 s2, s[14:15], v[5:6]                      // 000000001ba8: d45c0002 02020a0e
	v_or_b32_e32 v8, s16, v8                                   // 000000001bb0: 38101010
	v_cndmask_b32_e32 v7, s14, v1, vcc_lo                      // 000000001bb4: 020e020e
	v_cndmask_b32_e32 v10, s15, v9, vcc_lo                     // 000000001bb8: 0214120f
	v_cmp_gt_u64_e32 vcc_lo, s[14:15], v[2:3]                  // 000000001bbc: 7cb8040e
	v_lshlrev_b32_e32 v4, 4, v0                                // 000000001bc0: 30080084
	v_or_b32_e32 v3, 0x80, v1                                  // 000000001bc4: 380602ff 00000080
	v_cndmask_b32_e64 v5, s14, v5, s2                          // 000000001bcc: d5010005 000a0a0e
	v_dual_mov_b32 v57, v56 :: v_dual_mov_b32 v14, s17         // 000000001bd4: ca100138 390e0011
	s_delay_alu instid0(valu_dep_4)                            // 000000001bdc: bf870004
	v_and_b32_e32 v72, 48, v4                                  // 000000001be0: 369008b0
	v_mov_b32_e32 v4, s27                                      // 000000001be4: 7e08021b
	s_wait_alu depctr_va_vcc(0)                                // 000000001be8: bf88ff9d
	v_cndmask_b32_e32 v12, s15, v9, vcc_lo                     // 000000001bec: 0218120f
	s_add_nc_u64 s[40:41], s[8:9], s[10:11]                    // 000000001bf0: a9a80a08
	v_or_b32_e32 v127, 16, v124                                // 000000001bf4: 38fef890
	v_add_nc_u32_e32 v73, v11, v72                             // 000000001bf8: 4a92910b
	v_cndmask_b32_e32 v11, s14, v2, vcc_lo                     // 000000001bfc: 0216040e
	v_cmp_gt_u64_e32 vcc_lo, s[14:15], v[3:4]                  // 000000001c00: 7cb8060e
	v_mul_lo_u32 v4, v7, s7                                    // 000000001c04: d72c0004 02000f07
	v_mad_co_u64_u32 v[1:2], null, v7, s6, s[12:13]            // 000000001c0c: d6fe7c01 00300d07
	v_mul_lo_u32 v10, v10, s6                                  // 000000001c14: d72c000a 02000d0a
	v_mul_lo_u32 v13, v11, s7                                  // 000000001c1c: d72c000d 02000f0b
	v_mad_co_u64_u32 v[6:7], null, v11, s6, s[12:13]           // 000000001c24: d6fe7c06 00300d0b
	v_mul_lo_u32 v11, v12, s6                                  // 000000001c2c: d72c000b 02000d0c
	s_wait_alu depctr_va_vcc(0)                                // 000000001c34: bf88ff9d
	v_cndmask_b32_e32 v3, s14, v3, vcc_lo                      // 000000001c38: 0206060e
	v_dual_cndmask_b32 v12, s15, v9 :: v_dual_mov_b32 v59, v56 // 000000001c3c: ca50120f 0c3a0138
	v_cndmask_b32_e64 v9, s15, v9, s2                          // 000000001c44: d5010009 000a120f
	v_add3_u32 v10, v10, v2, v4                                // 000000001c4c: d655000a 0412050a
	v_add_co_u32 v74, vcc_lo, v1, v72                          // 000000001c54: d7006a4a 02029101
	v_add3_u32 v7, v11, v7, v13                                // 000000001c5c: d6550007 04360f0b
	v_mul_lo_u32 v11, v3, s7                                   // 000000001c64: d72c000b 02000f03
	v_mad_co_u64_u32 v[2:3], null, v3, s6, s[12:13]            // 000000001c6c: d6fe7c02 00300d03
	v_mul_lo_u32 v12, v12, s6                                  // 000000001c74: d72c000c 02000d0c
	v_mul_lo_u32 v13, v5, s7                                   // 000000001c7c: d72c000d 02000f05
	v_mad_co_u64_u32 v[4:5], null, v5, s6, s[12:13]            // 000000001c84: d6fe7c04 00300d05
	v_mul_lo_u32 v9, v9, s6                                    // 000000001c8c: d72c0009 02000d09
	s_wait_alu depctr_va_vcc(0)                                // 000000001c94: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v10, vcc_lo              // 000000001c98: d5207c4b 01aa1480
	v_add_co_u32 v76, vcc_lo, v6, v72                          // 000000001ca0: d7006a4c 02029106
	v_add3_u32 v1, v12, v3, v11                                // 000000001ca8: d6550001 042e070c
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb0: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, 0, v7, vcc_lo               // 000000001cb4: d5207c4d 01aa0e80
	v_add3_u32 v3, v9, v5, v13                                 // 000000001cbc: d6550003 04360b09
	v_add_co_u32 v78, vcc_lo, v2, v72                          // 000000001cc4: d7006a4e 02029102
	s_wait_alu depctr_va_vcc(0)                                // 000000001ccc: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, 0, v1, vcc_lo               // 000000001cd0: d5207c4f 01aa0280
	v_alignbit_b32 v1, v14, v8, 4                              // 000000001cd8: d6160001 0212110e
	v_add_co_u32 v80, vcc_lo, v4, v72                          // 000000001ce0: d7006a50 02029104
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce8: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v3, vcc_lo               // 000000001cec: d5207c51 01aa0680
	v_add_co_u32 v64, vcc_lo, s8, v8                           // 000000001cf4: d7006a40 02021008
	s_lshr_b32 s8, s7, 4                                       // 000000001cfc: 85088407
	s_lshr_b64 s[2:3], s[6:7], 4                               // 000000001d00: 85828406
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d04: bf88ff9e
	v_mul_lo_u32 v2, s8, v1                                    // 000000001d08: d72c0002 02020208
	v_mad_co_u64_u32 v[66:67], null, s2, v1, 0                 // 000000001d10: d6fe7c42 02020202
	v_lshrrev_b32_e32 v1, 1, v0                                // 000000001d18: 32020081
	v_and_b32_e32 v3, 0xcf, v0                                 // 000000001d1c: 360600ff 000000cf
	v_or_b32_e32 v102, 32, v70                                 // 000000001d24: 38cc8ca0
	v_or_b32_e32 v114, 48, v70                                 // 000000001d28: 38e48cb0
	v_dual_mov_b32 v61, v56 :: v_dual_and_b32 v4, 60, v0       // 000000001d2c: ca240138 3d0400bc
	v_and_b32_e32 v115, 8, v1                                  // 000000001d34: 36e60288
	v_mul_u32_u24_e32 v1, 0x50, v3                             // 000000001d38: 160206ff 00000050
	v_dual_mov_b32 v63, v56 :: v_dual_and_b32 v0, 47, v0       // 000000001d40: ca240138 3f0000af
	s_lshr_b32 s3, s17, 4                                      // 000000001d48: 85038411
	v_or_b32_e32 v3, v102, v71                                 // 000000001d4c: 38068f66
	s_delay_alu instid0(valu_dep_3)                            // 000000001d50: bf870003
	v_or_b32_e32 v84, v115, v1                                 // 000000001d54: 38a80373
	v_or_b32_e32 v1, v127, v71                                 // 000000001d58: 38028f7f
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d5c: bf88ff9e
	s_mul_i32 s2, s2, s3                                       // 000000001d60: 96020302
	v_mul_u32_u24_e32 v0, 0x50, v0                             // 000000001d64: 160000ff 00000050
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d6c: bf88ff9e
	v_add3_u32 v67, v67, s2, v2                                // 000000001d70: d6550043 04080543
	v_add_co_u32 v82, s2, s4, v4                               // 000000001d78: d7000252 02020804
	v_mul_u32_u24_e32 v1, 0x50, v1                             // 000000001d80: 160202ff 00000050
	v_or_b32_e32 v2, v90, v71                                  // 000000001d88: 38048f5a
	v_or_b32_e32 v4, v114, v71                                 // 000000001d8c: 38088f72
	v_mul_u32_u24_e32 v3, 0x50, v3                             // 000000001d90: 160606ff 00000050
	v_or_b32_e32 v0, v115, v0                                  // 000000001d98: 38000173
	v_or_b32_e32 v1, v1, v115                                  // 000000001d9c: 3802e701
	v_mul_u32_u24_e32 v2, 0x50, v2                             // 000000001da0: 160404ff 00000050
	v_mul_u32_u24_e32 v4, 0x50, v4                             // 000000001da8: 160808ff 00000050
	s_wait_alu depctr_va_sdst(0)                               // 000000001db0: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s5, 0, s2                   // 000000001db4: d5207c53 00090005
	v_add_nc_u32_e32 v89, 0x5000, v1                           // 000000001dbc: 4ab202ff 00005000
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc4: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s9, v14, vcc_lo             // 000000001dc8: d5207c41 01aa1c09
	v_add_co_u32 v68, vcc_lo, s40, v8                          // 000000001dd0: d7006a44 02021028
	s_wait_alu depctr_va_vcc(0)                                // 000000001dd8: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s41, v14, vcc_lo            // 000000001ddc: d5207c45 01aa1c29
	v_or_b32_e32 v85, v2, v115                                 // 000000001de4: 38aae702
	v_or_b32_e32 v86, v3, v115                                 // 000000001de8: 38ace703
	v_or_b32_e32 v87, v4, v115                                 // 000000001dec: 38aee704
	v_dual_mov_b32 v25, v56 :: v_dual_add_nc_u32 v88, 0x5000, v0// 000000001df0: ca200138 195800ff 00005000
	v_dual_mov_b32 v58, v56 :: v_dual_mov_b32 v27, v56         // 000000001dfc: ca100138 3a1a0138
	v_dual_mov_b32 v60, v56 :: v_dual_mov_b32 v29, v56         // 000000001e04: ca100138 3c1c0138
	v_dual_mov_b32 v62, v56 :: v_dual_mov_b32 v31, v56         // 000000001e0c: ca100138 3e1e0138
	v_dual_mov_b32 v24, v56 :: v_dual_mov_b32 v49, v56         // 000000001e14: ca100138 18300138
	v_dual_mov_b32 v26, v56 :: v_dual_mov_b32 v51, v56         // 000000001e1c: ca100138 1a320138
	v_dual_mov_b32 v28, v56 :: v_dual_mov_b32 v53, v56         // 000000001e24: ca100138 1c340138
	v_dual_mov_b32 v30, v56 :: v_dual_mov_b32 v55, v56         // 000000001e2c: ca100138 1e360138
	v_dual_mov_b32 v48, v56 :: v_dual_mov_b32 v17, v56         // 000000001e34: ca100138 30100138
	v_dual_mov_b32 v50, v56 :: v_dual_mov_b32 v19, v56         // 000000001e3c: ca100138 32120138
	v_dual_mov_b32 v52, v56 :: v_dual_mov_b32 v21, v56         // 000000001e44: ca100138 34140138
	v_dual_mov_b32 v54, v56 :: v_dual_mov_b32 v23, v56         // 000000001e4c: ca100138 36160138
	v_dual_mov_b32 v16, v56 :: v_dual_mov_b32 v41, v56         // 000000001e54: ca100138 10280138
	v_dual_mov_b32 v18, v56 :: v_dual_mov_b32 v43, v56         // 000000001e5c: ca100138 122a0138
	v_dual_mov_b32 v20, v56 :: v_dual_mov_b32 v45, v56         // 000000001e64: ca100138 142c0138
	v_dual_mov_b32 v22, v56 :: v_dual_mov_b32 v47, v56         // 000000001e6c: ca100138 162e0138
	v_dual_mov_b32 v40, v56 :: v_dual_mov_b32 v9, v56          // 000000001e74: ca100138 28080138
	v_dual_mov_b32 v42, v56 :: v_dual_mov_b32 v11, v56         // 000000001e7c: ca100138 2a0a0138
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v13, v56         // 000000001e84: ca100138 2c0c0138
	v_dual_mov_b32 v46, v56 :: v_dual_mov_b32 v15, v56         // 000000001e8c: ca100138 2e0e0138
	v_dual_mov_b32 v8, v56 :: v_dual_mov_b32 v33, v56          // 000000001e94: ca100138 08200138
	v_dual_mov_b32 v10, v56 :: v_dual_mov_b32 v35, v56         // 000000001e9c: ca100138 0a220138
	v_dual_mov_b32 v12, v56 :: v_dual_mov_b32 v37, v56         // 000000001ea4: ca100138 0c240138
	v_dual_mov_b32 v14, v56 :: v_dual_mov_b32 v39, v56         // 000000001eac: ca100138 0e260138
	v_dual_mov_b32 v32, v56 :: v_dual_mov_b32 v1, v56          // 000000001eb4: ca100138 20000138
	v_dual_mov_b32 v34, v56 :: v_dual_mov_b32 v3, v56          // 000000001ebc: ca100138 22020138
	v_dual_mov_b32 v36, v56 :: v_dual_mov_b32 v5, v56          // 000000001ec4: ca100138 24040138
	v_dual_mov_b32 v38, v56 :: v_dual_mov_b32 v7, v56          // 000000001ecc: ca100138 26060138
	v_mov_b32_e32 v0, v56                                      // 000000001ed4: 7e000338
	v_mov_b32_e32 v2, v56                                      // 000000001ed8: 7e040338
	v_mov_b32_e32 v4, v56                                      // 000000001edc: 7e080338
	v_mov_b32_e32 v6, v56                                      // 000000001ee0: 7e0c0338
	s_mov_b64 s[6:7], 0                                        // 000000001ee4: be860180
	s_mov_b32 s15, -1                                          // 000000001ee8: be8f00c1
	s_wait_alu depctr_sa_sdst(0)                               // 000000001eec: bf88ff9e
	v_or_b32_e32 v93, s6, v72                                  // 000000001ef0: 38ba9006
	v_mov_b32_e32 v94, s7                                      // 000000001ef4: 7ebc0207
	v_add_co_u32 v100, s3, v78, s6                             // 000000001ef8: d7000364 02000d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000001f00: bf88f19f
	v_add_co_ci_u32_e64 v101, null, s7, v79, s3                // 000000001f04: d5207c65 000e9e07
	v_alignbit_b32 v112, s7, v93, 5                            // 000000001f0c: d6160070 0216ba07
	v_lshrrev_b64 v[109:110], 4, v[93:94]                      // 000000001f14: d73d006d 0202ba84
	v_add_co_u32 v91, vcc_lo, v74, s6                          // 000000001f1c: d7006a5b 02000d4a
	global_load_b128 v[103:106], v[100:101], off               // 000000001f24: ee05c07c 00000067 00000064
	v_mul_lo_u32 v113, v112, s39                               // 000000001f30: d72c0071 02004f70
	v_mad_co_u64_u32 v[100:101], null, v112, s38, v[64:65]     // 000000001f38: d6fe7c64 05004d70
	s_wait_alu depctr_va_vcc(0)                                // 000000001f40: bf88ff9d
	v_add_co_ci_u32_e64 v92, null, s7, v75, vcc_lo             // 000000001f44: d5207c5c 01aa9607
	v_add_co_u32 v109, vcc_lo, v109, v66                       // 000000001f4c: d7006a6d 0202856d
	s_lshr_b32 s5, s7, 5                                       // 000000001f54: 85058507
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v110, null, v110, v67, vcc_lo          // 000000001f5c: d5207c6e 01aa876e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f64: bf88ff9e
	s_mul_i32 s5, s5, s38                                      // 000000001f68: 96052605
	v_add_co_u32 v95, s2, v76, s6                              // 000000001f6c: d700025f 02000d4c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f74: bf88ff9e
	v_add3_u32 v101, s5, v101, v113                            // 000000001f78: d6550065 05c6ca05
	v_lshlrev_b64_e32 v[109:110], 7, v[109:110]                // 000000001f80: 3edada87
	s_clause 0x1                                               // 000000001f84: bf850001
	global_load_d16_u8 v99, v[68:69], off                      // 000000001f88: ee07807c 00000063 00000044
	global_load_u8 v111, v[68:69], off                         // 000000001f94: ee04007c 0000006f 00000044
	s_wait_alu depctr_va_sdst(0)                               // 000000001fa0: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s7, v77, s2                 // 000000001fa4: d5207c60 000a9a07
	global_load_b128 v[91:94], v[91:92], off                   // 000000001fac: ee05c07c 0000005b 0000005b
	s_clause 0x1                                               // 000000001fb8: bf850001
	global_load_d16_hi_u8 v99, v[100:101], off                 // 000000001fbc: ee08407c 00000063 00000064
	global_load_u8 v112, v[100:101], off                       // 000000001fc8: ee04007c 00000070 00000064
	v_add_co_u32 v107, s4, v80, s6                             // 000000001fd4: d700046b 02000d50
	v_add_co_u32 v100, vcc_lo, v82, v109                       // 000000001fdc: d7006a64 0202db52
	s_wait_alu depctr_va_sdst(0)                               // 000000001fe4: bf88f19f
	v_add_co_ci_u32_e64 v108, null, s7, v81, s4                // 000000001fe8: d5207c6c 0012a207
	global_load_b128 v[95:98], v[95:96], off                   // 000000001ff0: ee05c07c 0000005f 0000005f
	s_wait_alu depctr_va_vcc(0)                                // 000000001ffc: bf88ff9d
	v_add_co_ci_u32_e64 v101, null, v83, v110, vcc_lo          // 000000002000: d5207c65 01aadd53
	global_load_b128 v[107:110], v[107:108], off               // 000000002008: ee05c07c 0000006b 0000006b
	s_and_b32 vcc_lo, exec_lo, s15                             // 000000002014: 8b6a0f7e
	s_mov_b32 s15, 0                                           // 000000002018: be8f0080
	s_clause 0x1                                               // 00000000201c: bf850001
	global_load_b32 v113, v[100:101], off                      // 000000002020: ee05007c 00000071 00000064
	global_load_b32 v100, v[100:101], off offset:64            // 00000000202c: ee05007c 00000064 00004064
	s_barrier_signal -1                                        // 000000002038: be804ec1
	s_barrier_wait 0xffff                                      // 00000000203c: bf94ffff
	s_wait_loadcnt 0x9                                         // 000000002040: bfc00009
	ds_store_b128 v73, v[103:106] offset:10240                 // 000000002044: db7c2800 00006749
	s_wait_loadcnt 0x6                                         // 00000000204c: bfc00006
	ds_store_b128 v73, v[91:94]                                // 000000002050: db7c0000 00005b49
	s_wait_loadcnt 0x5                                         // 000000002058: bfc00005
	v_cmp_eq_u16_e64 s2, v99.l, v99.h op_sel:[0,1,0]           // 00000000205c: d43a1002 0202c763
	s_wait_loadcnt 0x4                                         // 000000002064: bfc00004
	v_sub_nc_u32_e32 v91, v111, v112                           // 000000002068: 4cb6e16f
	v_cmp_eq_u16_e64 s3, 0, v99.h op_sel:[0,1,0]               // 00000000206c: d43a1003 0202c680
	s_delay_alu instid0(valu_dep_2)                            // 000000002074: bf870002
	v_cmp_ne_u32_e64 s4, 2, v91                                // 000000002078: d44d0004 0202b682
	v_cmp_ne_u32_e64 s5, 3, v91                                // 000000002080: d44d0005 0202b683
	s_wait_loadcnt 0x3                                         // 000000002088: bfc00003
	ds_store_b128 v73, v[95:98] offset:5120                    // 00000000208c: db7c1400 00005f49
	s_wait_alu depctr_va_sdst(0)                               // 000000002094: bf88f19f
	v_cndmask_b32_e64 v95, 0, 0x3c383000, s2                   // 000000002098: d501005f 0009fe80 3c383000
	v_cndmask_b32_e64 v96, 0, 0x4c484440, s2                   // 0000000020a4: d5010060 0009fe80 4c484440
	v_cmp_ne_u32_e64 s2, 1, v91                                // 0000000020b0: d44d0002 0202b681
	s_wait_loadcnt 0x2                                         // 0000000020b8: bfc00002
	ds_store_b128 v73, v[107:110] offset:15360                 // 0000000020bc: db7c3c00 00006b49
	v_cmp_ne_u32_e64 s6, 4, v91                                // 0000000020c4: d44d0006 0202b684
	v_cmp_ne_u32_e64 s7, 5, v91                                // 0000000020cc: d44d0007 0202b685
	v_cmp_ne_u32_e64 s8, 6, v91                                // 0000000020d4: d44d0008 0202b686
	v_cmp_ne_u32_e64 s9, 7, v91                                // 0000000020dc: d44d0009 0202b687
	v_cmp_ne_u32_e64 s10, 8, v91                               // 0000000020e4: d44d000a 0202b688
	v_cmp_ne_u32_e64 s11, 9, v91                               // 0000000020ec: d44d000b 0202b689
	v_cmp_ne_u32_e64 s12, 10, v91                              // 0000000020f4: d44d000c 0202b68a
	v_cmp_ne_u32_e64 s13, 11, v91                              // 0000000020fc: d44d000d 0202b68b
	v_cmp_ne_u32_e64 s14, 12, v91                              // 000000002104: d44d000e 0202b68c
	s_wait_loadcnt 0x1                                         // 00000000210c: bfc00001
	v_lshrrev_b32_e32 v97, 8, v113                             // 000000002110: 32c2e288
	v_lshrrev_b32_e32 v98, 24, v113                            // 000000002114: 32c4e298
	s_wait_loadcnt 0x0                                         // 000000002118: bfc00000
	v_lshrrev_b32_e32 v99, 8, v100                             // 00000000211c: 32c6c888
	v_lshrrev_b32_e32 v101, 24, v100                           // 000000002120: 32cac898
	v_lshlrev_b32_e32 v103, 3, v113                            // 000000002124: 30cee283
	v_lshlrev_b16 v91.l, 4, v113.l                             // 000000002128: d738005b 0202e284
	v_lshrrev_b32_e32 v104, 1, v113                            // 000000002130: 32d0e281
	v_and_b16 v91.h, 0x80, v113.l op_sel:[0,0,1]               // 000000002134: d762405b 0202e2ff 00000080
	v_lshrrev_b32_e32 v105, 5, v113                            // 000000002140: 32d2e285
	v_lshrrev_b32_e32 v106, 9, v113                            // 000000002144: 32d4e289
	v_lshrrev_b32_e32 v107, 13, v113                           // 000000002148: 32d6e28d
	v_lshlrev_b16 v92.l, 4, v113.h op_sel:[0,1,0]              // 00000000214c: d738105c 0202e284
	v_lshrrev_b32_e32 v108, 17, v113                           // 000000002154: 32d8e291
	v_and_b16 v92.h, 0x80, v113.h op_sel:[0,1,1]               // 000000002158: d762505c 0202e2ff 00000080
	v_lshrrev_b32_e32 v109, 21, v113                           // 000000002164: 32dae295
	v_lshrrev_b32_e32 v110, 25, v113                           // 000000002168: 32dce299
	v_lshlrev_b32_e32 v112, 3, v100                            // 00000000216c: 30e0c883
	v_lshlrev_b16 v93.l, 4, v100.l                             // 000000002170: d738005d 0202c884
	v_lshrrev_b32_e32 v111, 1, v100                            // 000000002178: 32dec881
	v_and_b16 v93.h, 0x80, v100.l op_sel:[0,0,1]               // 00000000217c: d762405d 0202c8ff 00000080
	v_lshrrev_b32_e32 v113, 5, v100                            // 000000002188: 32e2c885
	v_lshrrev_b32_e32 v116, 9, v100                            // 00000000218c: 32e8c889
	v_lshrrev_b32_e32 v117, 13, v100                           // 000000002190: 32eac88d
	v_lshlrev_b16 v94.l, 4, v100.h op_sel:[0,1,0]              // 000000002194: d738105e 0202c884
	v_lshrrev_b32_e32 v118, 17, v100                           // 00000000219c: 32ecc891
	v_and_b16 v94.h, 0x80, v100.h op_sel:[0,1,1]               // 0000000021a0: d762505e 0202c8ff 00000080
	v_lshrrev_b32_e32 v119, 21, v100                           // 0000000021ac: 32eec895
	v_lshrrev_b32_e32 v100, 25, v100                           // 0000000021b0: 32c8c899
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b4: bf88f19f
	v_cndmask_b32_e64 v120, 0x44403c38, v96, s2                // 0000000021b8: d5010078 000ac0ff 44403c38
	v_cndmask_b32_e64 v121, 0x34302800, v95, s2                // 0000000021c4: d5010079 000abeff 34302800
	v_lshlrev_b16 v95.l, 4, v97.l                              // 0000000021d0: d738005f 0202c284
	v_and_b16 v95.h, 0x80, v97.l op_sel:[0,0,1]                // 0000000021d8: d762405f 0202c2ff 00000080
	v_and_b32_e32 v137, 56, v100                               // 0000000021e4: 3712c8b8
	v_cndmask_b32_e64 v100, 0x3c383430, v120, s4               // 0000000021e8: d5010064 0012f0ff 3c383430
	v_lshlrev_b16 v97.l, 4, v99.l                              // 0000000021f4: d7380061 0202c684
	v_and_b16 v97.h, 0x80, v99.l op_sel:[0,0,1]                // 0000000021fc: d7624061 0202c6ff 00000080
	v_cndmask_b32_e64 v99, 0x2c282000, v121, s4                // 000000002208: d5010063 0012f2ff 2c282000
	s_and_b32 s2, s14, s13                                     // 000000002214: 8b020d0e
	v_cndmask_b32_e64 v100, 0x34302c28, v100, s5               // 000000002218: d5010064 0016c8ff 34302c28
	v_and_b32_e32 v122, 56, v104                               // 000000002224: 36f4d0b8
	v_and_b32_e32 v123, 56, v105                               // 000000002228: 36f6d2b8
	v_cndmask_b32_e64 v99, 0x24201800, v99, s5                 // 00000000222c: d5010063 0016c6ff 24201800
	v_and_b32_e32 v125, 56, v106                               // 000000002238: 36fad4b8
	v_cndmask_b32_e64 v100, 0x2c282420, v100, s6               // 00000000223c: d5010064 001ac8ff 2c282420
	v_and_b32_e32 v126, 56, v107                               // 000000002248: 36fcd6b8
	v_and_b32_e32 v128, 56, v108                               // 00000000224c: 3700d8b8
	v_cndmask_b32_e64 v99, 0x1c181000, v99, s6                 // 000000002250: d5010063 001ac6ff 1c181000
	v_and_b32_e32 v132, 56, v113                               // 00000000225c: 3708e2b8
	v_cndmask_b32_e64 v100, 0x24201c18, v100, s7               // 000000002260: d5010064 001ec8ff 24201c18
	v_and_b32_e32 v129, 56, v109                               // 00000000226c: 3702dab8
	v_and_b32_e32 v133, 56, v116                               // 000000002270: 370ae8b8
	v_cndmask_b32_e64 v99, 0x14100800, v99, s7                 // 000000002274: d5010063 001ec6ff 14100800
	v_and_b32_e32 v130, 56, v110                               // 000000002280: 3704dcb8
	v_cndmask_b32_e64 v100, 0x1c181410, v100, s8               // 000000002284: d5010064 0022c8ff 1c181410
	v_and_b32_e32 v134, 56, v117                               // 000000002290: 370ceab8
	v_and_b32_e32 v135, 56, v118                               // 000000002294: 370eecb8
	v_cndmask_b32_e64 v99, 0xc080400, v99, s8                  // 000000002298: d5010063 0022c6ff 0c080400
	v_and_b32_e32 v131, 56, v111                               // 0000000022a4: 3706deb8
	v_cndmask_b32_e64 v100, 0x14100c08, v100, s9               // 0000000022a8: d5010064 0026c8ff 14100c08
	v_and_b32_e32 v136, 56, v119                               // 0000000022b4: 3710eeb8
	v_lshlrev_b16 v96.l, 4, v98.l                              // 0000000022b8: d7380060 0202c484
	v_cndmask_b32_e64 v99, 0x6040200, v99, s9                  // 0000000022c0: d5010063 0026c6ff 06040200
	v_and_b16 v96.h, 0x80, v98.l op_sel:[0,0,1]                // 0000000022cc: d7624060 0202c4ff 00000080
	v_cndmask_b32_e64 v100, 0xc080604, v100, s10               // 0000000022d8: d5010064 002ac8ff 0c080604
	v_lshlrev_b16 v98.l, 4, v101.l                             // 0000000022e4: d7380062 0202ca84
	v_and_b16 v91.l, 0x80, v91.l                               // 0000000022ec: d762005b 0202b6ff 00000080
	v_cndmask_b32_e64 v99, 0x3020100, v99, s10                 // 0000000022f8: d5010063 002ac6ff 03020100
	v_and_b16 v92.l, 0x80, v92.l                               // 000000002304: d762005c 0202b8ff 00000080
	v_cndmask_b32_e64 v100, 0x6040302, v100, s11               // 000000002310: d5010064 002ec8ff 06040302
	v_and_b16 v93.l, 0x80, v93.l                               // 00000000231c: d762005d 0202baff 00000080
	v_and_b16 v94.l, 0x80, v94.l                               // 000000002328: d762005e 0202bcff 00000080
	v_cndmask_b32_e64 v99, 0x2010000, v99, s11                 // 000000002334: d5010063 002ec6ff 02010000
	v_and_b16 v98.h, 0x80, v101.l op_sel:[0,0,1]               // 000000002340: d7624062 0202caff 00000080
	v_cndmask_b32_e64 v100, 0x3020201, v100, s12               // 00000000234c: d5010064 0032c8ff 03020201
	v_and_b16 v95.l, 0x80, v95.l                               // 000000002358: d762005f 0202beff 00000080
	v_and_b16 v96.l, 0x80, v96.l                               // 000000002364: d7620060 0202c0ff 00000080
	v_cndmask_b32_e64 v99, 0x1000000, v99, s12                 // 000000002370: d5010063 0032c6ff 01000000
	v_and_b16 v97.l, 0x80, v97.l                               // 00000000237c: d7620061 0202c2ff 00000080
	v_cndmask_b32_e64 v100, 0x2010100, v100, s13               // 000000002388: d5010064 0036c8ff 02010100
	v_and_b16 v98.l, 0x80, v98.l                               // 000000002394: d7620062 0202c4ff 00000080
	s_mov_b64 s[6:7], 64                                       // 0000000023a0: be8601c0
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023a4: bf88ff9e
	v_cndmask_b32_e64 v99, 0, v99, s2                          // 0000000023a8: d5010063 000ac680
	v_cndmask_b32_e64 v100, 0x1000000, v100, s14               // 0000000023b0: d5010064 003ac8ff 01000000
	s_delay_alu instid0(valu_dep_1)                            // 0000000023bc: bf870001
	v_lshrrev_b64 v[103:104], v103, v[99:100]                  // 0000000023c0: d73d0067 0202c767
	v_lshrrev_b64 v[104:105], v122, v[99:100]                  // 0000000023c8: d73d0068 0202c77a
	v_lshrrev_b64 v[105:106], v123, v[99:100]                  // 0000000023d0: d73d0069 0202c77b
	v_lshrrev_b64 v[106:107], v125, v[99:100]                  // 0000000023d8: d73d006a 0202c77d
	v_lshrrev_b64 v[107:108], v126, v[99:100]                  // 0000000023e0: d73d006b 0202c77e
	v_lshrrev_b64 v[108:109], v128, v[99:100]                  // 0000000023e8: d73d006c 0202c780
	v_lshrrev_b64 v[116:117], v132, v[99:100]                  // 0000000023f0: d73d0074 0202c784
	v_lshrrev_b64 v[109:110], v129, v[99:100]                  // 0000000023f8: d73d006d 0202c781
	v_lshrrev_b64 v[117:118], v133, v[99:100]                  // 000000002400: d73d0075 0202c785
	v_lshrrev_b64 v[110:111], v130, v[99:100]                  // 000000002408: d73d006e 0202c782
	v_lshrrev_b64 v[118:119], v134, v[99:100]                  // 000000002410: d73d0076 0202c786
	v_lshrrev_b64 v[111:112], v112, v[99:100]                  // 000000002418: d73d006f 0202c770
	v_lshrrev_b64 v[119:120], v135, v[99:100]                  // 000000002420: d73d0077 0202c787
	v_lshrrev_b64 v[112:113], v131, v[99:100]                  // 000000002428: d73d0070 0202c783
	v_lshrrev_b64 v[120:121], v136, v[99:100]                  // 000000002430: d73d0078 0202c788
	v_lshrrev_b64 v[99:100], v137, v[99:100]                   // 000000002438: d73d0063 0202c789
	v_or_b16 v91.l, v91.l, v103.l                              // 000000002440: d763005b 0202cf5b
	v_or_b16 v91.h, v91.h, v104.l op_sel:[1,0,1]               // 000000002448: d763485b 0202d15b
	v_or_b16 v95.l, v95.l, v105.l                              // 000000002450: d763005f 0202d35f
	v_or_b16 v95.h, v95.h, v106.l op_sel:[1,0,1]               // 000000002458: d763485f 0202d55f
	v_or_b16 v92.l, v92.l, v107.l                              // 000000002460: d763005c 0202d75c
	v_or_b16 v92.h, v92.h, v108.l op_sel:[1,0,1]               // 000000002468: d763485c 0202d95c
	v_or_b16 v96.l, v96.l, v109.l                              // 000000002470: d7630060 0202db60
	v_or_b16 v96.h, v96.h, v110.l op_sel:[1,0,1]               // 000000002478: d7634860 0202dd60
	v_or_b16 v93.l, v93.l, v111.l                              // 000000002480: d763005d 0202df5d
	v_or_b16 v93.h, v93.h, v112.l op_sel:[1,0,1]               // 000000002488: d763485d 0202e15d
	v_or_b16 v97.l, v97.l, v116.l                              // 000000002490: d7630061 0202e961
	v_or_b16 v97.h, v97.h, v117.l op_sel:[1,0,1]               // 000000002498: d7634861 0202eb61
	v_or_b16 v94.l, v94.l, v118.l                              // 0000000024a0: d763005e 0202ed5e
	v_or_b16 v94.h, v94.h, v119.l op_sel:[1,0,1]               // 0000000024a8: d763485e 0202ef5e
	v_or_b16 v98.l, v98.l, v120.l                              // 0000000024b0: d7630062 0202f162
	v_or_b16 v98.h, v98.h, v99.l op_sel:[1,0,1]                // 0000000024b8: d7634862 0202c762
	v_cndmask_b16 v91.l, v91.l, 0, s3                          // 0000000024c0: d65d005b 000d015b
	v_cndmask_b16 v91.h, v91.h, 0, s3                          // 0000000024c8: d65d485b 000d015b
	v_cndmask_b16 v95.l, v95.l, 0, s3                          // 0000000024d0: d65d005f 000d015f
	v_cndmask_b16 v95.h, v95.h, 0, s3                          // 0000000024d8: d65d485f 000d015f
	v_cndmask_b16 v92.l, v92.l, 0, s3                          // 0000000024e0: d65d005c 000d015c
	v_cndmask_b16 v92.h, v92.h, 0, s3                          // 0000000024e8: d65d485c 000d015c
	v_cndmask_b16 v96.l, v96.l, 0, s3                          // 0000000024f0: d65d0060 000d0160
	v_cndmask_b16 v96.h, v96.h, 0, s3                          // 0000000024f8: d65d4860 000d0160
	v_cndmask_b16 v93.l, v93.l, 0, s3                          // 000000002500: d65d005d 000d015d
	v_cndmask_b16 v93.h, v93.h, 0, s3                          // 000000002508: d65d485d 000d015d
	v_cndmask_b16 v97.l, v97.l, 0, s3                          // 000000002510: d65d0061 000d0161
	v_cndmask_b16 v98.h, v98.h, 0, s3                          // 000000002518: d65d4862 000d0162
	v_cndmask_b16 v98.l, v98.l, 0, s3                          // 000000002520: d65d0062 000d0162
	v_cndmask_b16 v94.h, v94.h, 0, s3                          // 000000002528: d65d485e 000d015e
	v_cndmask_b16 v94.l, v94.l, 0, s3                          // 000000002530: d65d005e 000d015e
	v_cndmask_b16 v97.h, v97.h, 0, s3                          // 000000002538: d65d4861 000d0161
	v_lshlrev_b16 v98.h, 8, v98.h op_sel:[0,1,1]               // 000000002540: d7385062 0202c488
	v_and_b16 v98.l, 0xff, v98.l                               // 000000002548: d7620062 0202c4ff 000000ff
	v_lshlrev_b16 v99.l, 8, v94.h op_sel:[0,1,0]               // 000000002554: d7381063 0202bc88
	v_and_b16 v94.l, 0xff, v94.l                               // 00000000255c: d762005e 0202bcff 000000ff
	v_lshlrev_b16 v97.h, 8, v97.h op_sel:[0,1,1]               // 000000002568: d7385061 0202c288
	v_and_b16 v97.l, 0xff, v97.l                               // 000000002570: d7620061 0202c2ff 000000ff
	v_lshlrev_b16 v99.h, 8, v93.h op_sel:[0,1,1]               // 00000000257c: d7385063 0202ba88
	v_and_b16 v93.l, 0xff, v93.l                               // 000000002584: d762005d 0202baff 000000ff
	v_lshlrev_b16 v96.h, 8, v96.h op_sel:[0,1,1]               // 000000002590: d7385060 0202c088
	v_and_b16 v96.l, 0xff, v96.l                               // 000000002598: d7620060 0202c0ff 000000ff
	v_lshlrev_b16 v100.l, 8, v92.h op_sel:[0,1,0]              // 0000000025a4: d7381064 0202b888
	v_and_b16 v92.l, 0xff, v92.l                               // 0000000025ac: d762005c 0202b8ff 000000ff
	v_lshlrev_b16 v95.h, 8, v95.h op_sel:[0,1,1]               // 0000000025b8: d738505f 0202be88
	v_and_b16 v95.l, 0xff, v95.l                               // 0000000025c0: d762005f 0202beff 000000ff
	v_lshlrev_b16 v100.h, 8, v91.h op_sel:[0,1,1]              // 0000000025cc: d7385064 0202b688
	v_and_b16 v91.l, 0xff, v91.l                               // 0000000025d4: d762005b 0202b6ff 000000ff
	v_or_b16 v94.h, v98.l, v98.h op_sel:[0,1,1]                // 0000000025e0: d763505e 0202c562
	v_or_b16 v94.l, v94.l, v99.l                               // 0000000025e8: d763005e 0202c75e
	v_or_b16 v93.h, v97.l, v97.h op_sel:[0,1,1]                // 0000000025f0: d763505d 0202c361
	v_or_b16 v93.l, v93.l, v99.h op_sel:[0,1,0]                // 0000000025f8: d763105d 0202c75d
	v_or_b16 v92.h, v96.l, v96.h op_sel:[0,1,1]                // 000000002600: d763505c 0202c160
	v_or_b16 v92.l, v92.l, v100.l                              // 000000002608: d763005c 0202c95c
	v_or_b16 v91.h, v95.l, v95.h op_sel:[0,1,1]                // 000000002610: d763505b 0202bf5f
	v_or_b16 v91.l, v91.l, v100.h op_sel:[0,1,0]               // 000000002618: d763105b 0202c95b
	ds_store_b128 v73, v[91:94] offset:20480                   // 000000002620: db7c5000 00005b49
	s_wait_dscnt 0x0                                           // 000000002628: bfc60000
	s_barrier_signal -1                                        // 00000000262c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002630: bf94ffff
	ds_load_2addr_b64 v[91:94], v84 offset1:2                  // 000000002634: d9dc0200 5b000054
	ds_load_2addr_b64 v[95:98], v88 offset1:2                  // 00000000263c: d9dc0200 5f000058
	ds_load_2addr_b64 v[103:106], v89 offset1:2                // 000000002644: d9dc0200 67000059
	ds_load_2addr_b64 v[107:110], v85 offset1:2                // 00000000264c: d9dc0200 6b000055
	ds_load_2addr_b64 v[116:119], v86 offset1:2                // 000000002654: d9dc0200 74000056
	ds_load_2addr_b64 v[120:123], v87 offset1:2                // 00000000265c: d9dc0200 78000057
	ds_load_2addr_b64 v[128:131], v84 offset0:4 offset1:6      // 000000002664: d9dc0604 80000054
	ds_load_2addr_b64 v[132:135], v88 offset0:4 offset1:6      // 00000000266c: d9dc0604 84000058
	ds_load_2addr_b64 v[136:139], v89 offset0:4 offset1:6      // 000000002674: d9dc0604 88000059
	ds_load_2addr_b64 v[140:143], v85 offset0:4 offset1:6      // 00000000267c: d9dc0604 8c000055
	ds_load_2addr_b64 v[144:147], v86 offset0:4 offset1:6      // 000000002684: d9dc0604 90000056
	ds_load_2addr_b64 v[148:151], v87 offset0:4 offset1:6      // 00000000268c: d9dc0604 94000057
	s_wait_dscnt 0xa                                           // 000000002694: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[91:92], v[95:96], v[56:63]// 000000002698: cc464038 1ce2bf5b
	s_wait_dscnt 0x9                                           // 0000000026a0: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[91:92], v[103:104], v[24:31]// 0000000026a4: cc464018 1c62cf5b
	s_wait_dscnt 0x8                                           // 0000000026ac: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[107:108], v[95:96], v[48:55]// 0000000026b0: cc464030 1cc2bf6b
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[107:108], v[103:104], v[16:23]// 0000000026b8: cc464010 1c42cf6b
	s_wait_dscnt 0x7                                           // 0000000026c0: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[116:117], v[95:96], v[40:47]// 0000000026c4: cc464028 1ca2bf74
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[116:117], v[103:104], v[8:15]// 0000000026cc: cc464008 1c22cf74
	s_wait_dscnt 0x6                                           // 0000000026d4: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[120:121], v[95:96], v[32:39]// 0000000026d8: cc464020 1c82bf78
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[120:121], v[103:104], v[0:7]// 0000000026e0: cc464000 1c02cf78
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[93:94], v[97:98], v[56:63]// 0000000026e8: cc464038 1ce2c35d
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[93:94], v[105:106], v[24:31]// 0000000026f0: cc464018 1c62d35d
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[109:110], v[97:98], v[48:55]// 0000000026f8: cc464030 1cc2c36d
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[109:110], v[105:106], v[16:23]// 000000002700: cc464010 1c42d36d
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[118:119], v[97:98], v[40:47]// 000000002708: cc464028 1ca2c376
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[118:119], v[105:106], v[8:15]// 000000002710: cc464008 1c22d376
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[122:123], v[97:98], v[32:39]// 000000002718: cc464020 1c82c37a
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[122:123], v[105:106], v[0:7]// 000000002720: cc464000 1c02d37a
	s_wait_dscnt 0x4                                           // 000000002728: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[128:129], v[132:133], v[56:63]// 00000000272c: cc464038 1ce30980
	s_wait_dscnt 0x3                                           // 000000002734: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[136:137], v[24:31]// 000000002738: cc464018 1c631180
	s_wait_dscnt 0x2                                           // 000000002740: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[140:141], v[132:133], v[48:55]// 000000002744: cc464030 1cc3098c
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[140:141], v[136:137], v[16:23]// 00000000274c: cc464010 1c43118c
	s_wait_dscnt 0x1                                           // 000000002754: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[144:145], v[132:133], v[40:47]// 000000002758: cc464028 1ca30990
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[136:137], v[8:15]// 000000002760: cc464008 1c231190
	s_wait_dscnt 0x0                                           // 000000002768: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[132:133], v[32:39]// 00000000276c: cc464020 1c830994
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[148:149], v[136:137], v[0:7]// 000000002774: cc464000 1c031194
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[130:131], v[134:135], v[56:63]// 00000000277c: cc464038 1ce30d82
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[138:139], v[24:31]// 000000002784: cc464018 1c631582
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[142:143], v[134:135], v[48:55]// 00000000278c: cc464030 1cc30d8e
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[142:143], v[138:139], v[16:23]// 000000002794: cc464010 1c43158e
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[146:147], v[134:135], v[40:47]// 00000000279c: cc464028 1ca30d92
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[138:139], v[8:15]// 0000000027a4: cc464008 1c231592
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[134:135], v[32:39]// 0000000027ac: cc464020 1c830d96
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[138:139], v[0:7]// 0000000027b4: cc464000 1c031596
	s_cbranch_vccnz 64971                                      // 0000000027bc: bfa4fdcb <packed_folded_w4a8+0x3ec>
	v_or_b32_e32 v68, s26, v70                                 // 0000000027c0: 38888c1a
	v_or_b32_e32 v125, s16, v71                                // 0000000027c4: 38fa8e10
	v_mov_b32_e32 v67, s27                                     // 0000000027c8: 7e86021b
	s_load_b64 s[28:29], s[0:1], 0x58                          // 0000000027cc: f4002700 f8000058
	v_mov_b32_e32 v65, s17                                     // 0000000027d4: 7e820211
	v_or_b32_e32 v66, v115, v68                                // 0000000027d8: 38848973
	v_or_b32_e32 v64, v125, v124                               // 0000000027dc: 3880f97d
	v_mov_b32_e32 v126, s17                                    // 0000000027e0: 7efc0211
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000027e4: bf870193
	v_cmp_gt_i64_e64 s2, s[36:37], v[66:67]                    // 0000000027e8: d4540002 02028424
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[64:65]                // 0000000027f0: 7ca88026
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 0000000027f8: bf870142
	v_cndmask_b32_e64 v70, 0, v67, s2                          // 0000000027fc: d5010046 000a8680
	v_cndmask_b32_e64 v69, 0, v66, s2                          // 000000002804: d5010045 000a8480
	s_wait_alu depctr_va_vcc(0)                                // 00000000280c: bf88ff9d
	v_dual_cndmask_b32 v72, 0, v64 :: v_dual_cndmask_b32 v71, 0, v65// 000000002810: ca528080 48468280
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]                    // 000000002818: 3e8a8a82
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000281c: bf8701a2
	v_add_co_u32 v110, s2, s40, v72                            // 000000002820: d700026e 02029028
	s_wait_alu depctr_va_sdst(0)                               // 000000002828: bf88f19f
	v_add_co_ci_u32_e64 v111, null, s41, v71, s2               // 00000000282c: d5207c6f 000a8e29
	s_wait_kmcnt 0x0                                           // 000000002834: bfc70000
	s_delay_alu instid0(valu_dep_3)                            // 000000002838: bf870003
	v_add_co_u32 v72, s2, s28, v69                             // 00000000283c: d7000248 02028a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002844: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s29, v70, s2                // 000000002848: d5207c49 000a8c1d
	global_load_u8 v69, v[110:111], off                        // 000000002850: ee04007c 00000045 0000006e
	global_load_b32 v71, v[72:73], off                         // 00000000285c: ee05007c 00000047 00000048
	s_wait_loadcnt 0x1                                         // 000000002868: bfc00001
	v_dual_mov_b32 v69, s27 :: v_dual_lshlrev_b32 v70, 23, v69 // 00000000286c: ca22001b 45468a97
	s_wait_loadcnt 0x0                                         // 000000002874: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002878: bf870091
	v_mul_f32_e32 v74, v71, v70                                // 00000000287c: 10948d47
	v_cmp_class_f32_e64 s2, v74, 0x198                         // 000000002880: d47e0002 0201ff4a 00000198
	v_mul_f32_e32 v88, v56, v74                                // 00000000288c: 10b09538
	s_xor_b32 s2, s2, -1                                       // 000000002890: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002894: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002898: be832002
	s_cbranch_execnz 3966                                      // 00000000289c: bfa60f7e <packed_folded_w4a8+0x4b98>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000028a4: 8c7e037e
	v_or_b32_e32 v116, 1, v115                                 // 0000000028a8: 38e8e681
	v_mov_b32_e32 v75, v69                                     // 0000000028ac: 7e960345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 0000000028b0: bf870092
	v_or_b32_e32 v74, v116, v68                                // 0000000028b4: 38948974
	v_cmp_gt_i64_e64 s2, s[36:37], v[74:75]                    // 0000000028b8: d4540002 02029424
	s_wait_alu depctr_va_sdst(0)                               // 0000000028c0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000028c4: bf8700a1
	v_cndmask_b32_e64 v75, 0, v75, s2                          // 0000000028c8: d501004b 000a9680
	v_cndmask_b32_e64 v74, 0, v74, s2                          // 0000000028d0: d501004a 000a9480
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]                    // 0000000028d8: 3e949482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000028dc: bf870121
	v_add_co_u32 v74, s2, s28, v74                             // 0000000028e0: d700024a 0202941c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e8: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s29, v75, s2                // 0000000028ec: d5207c4b 000a961d
	global_load_b32 v56, v[74:75], off                         // 0000000028f4: ee05007c 00000038 0000004a
	s_wait_loadcnt 0x0                                         // 000000002900: bfc00000
	v_mul_f32_e32 v71, v56, v70                                // 000000002904: 108e8d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002908: bf870001
	v_cmp_class_f32_e64 s2, v71, 0x198                         // 00000000290c: d47e0002 0201ff47 00000198
	v_mul_f32_e32 v89, v57, v71                                // 000000002918: 10b28f39
	s_xor_b32 s2, s2, -1                                       // 00000000291c: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002920: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002924: be832002
	s_cbranch_execnz 3949                                      // 000000002928: bfa60f6d <packed_folded_w4a8+0x4be0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000292c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002930: 8c7e037e
	v_or_b32_e32 v117, 2, v115                                 // 000000002934: 38eae682
	v_mov_b32_e32 v57, v69                                     // 000000002938: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 00000000293c: bf870092
	v_or_b32_e32 v56, v117, v68                                // 000000002940: 38708975
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002944: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 00000000294c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002950: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002954: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 00000000295c: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002964: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002968: bf870121
	v_add_co_u32 v76, s2, s28, v56                             // 00000000296c: d700024c 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002974: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s29, v57, s2                // 000000002978: d5207c4d 000a721d
	global_load_b32 v56, v[76:77], off                         // 000000002980: ee05007c 00000038 0000004c
	s_wait_loadcnt 0x0                                         // 00000000298c: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002990: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002994: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002998: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v91, v58, v57                                // 0000000029a4: 10b6733a
	s_xor_b32 s2, s2, -1                                       // 0000000029a8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029ac: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000029b0: be832002
	s_cbranch_execnz 3932                                      // 0000000029b4: bfa60f5c <packed_folded_w4a8+0x4c28>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000029bc: 8c7e037e
	v_or_b32_e32 v128, 3, v115                                 // 0000000029c0: 3900e683
	v_mov_b32_e32 v57, v69                                     // 0000000029c4: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 0000000029c8: bf870092
	v_or_b32_e32 v56, v128, v68                                // 0000000029cc: 38708980
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 0000000029d0: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 0000000029d8: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000029dc: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 0000000029e0: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 0000000029e8: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 0000000029f0: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000029f4: bf870121
	v_add_co_u32 v78, s2, s28, v56                             // 0000000029f8: d700024e 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a00: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s29, v57, s2                // 000000002a04: d5207c4f 000a721d
	global_load_b32 v56, v[78:79], off                         // 000000002a0c: ee05007c 00000038 0000004e
	s_wait_loadcnt 0x0                                         // 000000002a18: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002a1c: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002a20: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002a24: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v92, v59, v57                                // 000000002a30: 10b8733b
	s_xor_b32 s2, s2, -1                                       // 000000002a34: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a38: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002a3c: be832002
	s_cbranch_execnz 3915                                      // 000000002a40: bfa60f4b <packed_folded_w4a8+0x4c70>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002a48: 8c7e037e
	v_or_b32_e32 v129, 4, v115                                 // 000000002a4c: 3902e684
	v_mov_b32_e32 v57, v69                                     // 000000002a50: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002a54: bf870092
	v_or_b32_e32 v56, v129, v68                                // 000000002a58: 38708981
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002a5c: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002a64: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002a68: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002a6c: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002a74: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002a7c: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a80: bf870121
	v_add_co_u32 v80, s2, s28, v56                             // 000000002a84: d7000250 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a8c: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s29, v57, s2                // 000000002a90: d5207c51 000a721d
	global_load_b32 v56, v[80:81], off                         // 000000002a98: ee05007c 00000038 00000050
	s_wait_loadcnt 0x0                                         // 000000002aa4: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002aa8: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002aac: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002ab0: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v93, v60, v57                                // 000000002abc: 10ba733c
	s_xor_b32 s2, s2, -1                                       // 000000002ac0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ac4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002ac8: be832002
	s_cbranch_execnz 3898                                      // 000000002acc: bfa60f3a <packed_folded_w4a8+0x4cb8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002ad4: 8c7e037e
	v_or_b32_e32 v130, 5, v115                                 // 000000002ad8: 3904e685
	v_mov_b32_e32 v57, v69                                     // 000000002adc: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002ae0: bf870092
	v_or_b32_e32 v56, v130, v68                                // 000000002ae4: 38708982
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002ae8: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002af0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002af4: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002af8: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002b00: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002b08: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b0c: bf870121
	v_add_co_u32 v82, s2, s28, v56                             // 000000002b10: d7000252 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002b18: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s29, v57, s2                // 000000002b1c: d5207c53 000a721d
	global_load_b32 v56, v[82:83], off                         // 000000002b24: ee05007c 00000038 00000052
	s_wait_loadcnt 0x0                                         // 000000002b30: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002b34: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002b38: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b3c: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v94, v61, v57                                // 000000002b48: 10bc733d
	s_xor_b32 s2, s2, -1                                       // 000000002b4c: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b50: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b54: be832002
	s_cbranch_execnz 3881                                      // 000000002b58: bfa60f29 <packed_folded_w4a8+0x4d00>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b60: 8c7e037e
	v_or_b32_e32 v131, 6, v115                                 // 000000002b64: 3906e686
	v_mov_b32_e32 v57, v69                                     // 000000002b68: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002b6c: bf870092
	v_or_b32_e32 v56, v131, v68                                // 000000002b70: 38708983
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002b74: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002b7c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002b80: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002b84: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002b8c: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002b94: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b98: bf870121
	v_add_co_u32 v84, s2, s28, v56                             // 000000002b9c: d7000254 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba4: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s29, v57, s2                // 000000002ba8: d5207c55 000a721d
	global_load_b32 v56, v[84:85], off                         // 000000002bb0: ee05007c 00000038 00000054
	s_wait_loadcnt 0x0                                         // 000000002bbc: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002bc0: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002bc4: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002bc8: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v95, v62, v57                                // 000000002bd4: 10be733e
	s_xor_b32 s2, s2, -1                                       // 000000002bd8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bdc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002be0: be832002
	s_cbranch_execnz 3864                                      // 000000002be4: bfa60f18 <packed_folded_w4a8+0x4d48>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002be8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002bec: 8c7e037e
	v_or_b32_e32 v132, 7, v115                                 // 000000002bf0: 3908e687
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002bf4: bf870091
	v_or_b32_e32 v68, v132, v68                                // 000000002bf8: 38888984
	v_cmp_gt_i64_e64 s2, s[36:37], v[68:69]                    // 000000002bfc: d4540002 02028824
	s_wait_alu depctr_va_sdst(0)                               // 000000002c04: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002c08: bf8700a1
	v_cndmask_b32_e64 v57, 0, v69, s2                          // 000000002c0c: d5010039 000a8a80
	v_cndmask_b32_e64 v56, 0, v68, s2                          // 000000002c14: d5010038 000a8880
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002c1c: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c20: bf870121
	v_add_co_u32 v86, s2, s28, v56                             // 000000002c24: d7000256 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c2c: bf88f19f
	v_add_co_ci_u32_e64 v87, null, s29, v57, s2                // 000000002c30: d5207c57 000a721d
	global_load_b32 v56, v[86:87], off                         // 000000002c38: ee05007c 00000038 00000056
	s_wait_loadcnt 0x0                                         // 000000002c44: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002c48: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002c4c: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002c50: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v96, v63, v57                                // 000000002c5c: 10c0733f
	s_xor_b32 s2, s2, -1                                       // 000000002c60: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c64: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002c68: be832002
	s_cbranch_execnz 3848                                      // 000000002c6c: bfa60f08 <packed_folded_w4a8+0x4d90>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002c74: 8c7e037e
	s_load_b64 s[34:35], s[0:1], 0xa8                          // 000000002c78: f4002880 f80000a8
	v_mul_lo_u32 v58, s39, v66                                 // 000000002c80: d72c003a 02028427
	v_mul_lo_u32 v59, s38, v67                                 // 000000002c88: d72c003b 02028626
	v_mad_co_u64_u32 v[56:57], null, s38, v66, 0               // 000000002c90: d6fe7c38 02028426
	v_sub_co_u32 v68, s0, s36, v66                             // 000000002c98: d7010044 02028424
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_4)// 000000002ca0: bf870221
	v_sub_co_ci_u32_e64 v69, null, s37, v67, s0                // 000000002ca4: d5217c45 00028625
	v_lshlrev_b64_e32 v[118:119], 1, v[64:65]                  // 000000002cac: 3eec8081
	v_add3_u32 v57, v57, v59, v58                              // 000000002cb0: d6550039 04ea7739
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002cb8: bf870113
	v_cmp_lt_i64_e64 s0, 0, v[68:69]                           // 000000002cbc: d4510000 02028880
	v_lshlrev_b64_e32 v[70:71], 1, v[56:57]                    // 000000002cc4: 3e8c7081
	s_and_b32 s1, s0, vcc_lo                                   // 000000002cc8: 8b016a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ccc: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000002cd0: be822001
	s_cbranch_execz 28                                         // 000000002cd4: bfa5001c <packed_folded_w4a8+0x1248>
	v_bfe_u32 v56, v88, 16, 1                                  // 000000002cd8: d6100038 02052158
	s_wait_kmcnt 0x0                                           // 000000002ce0: bfc70000
	v_add_co_u32 v57, s1, s34, v70                             // 000000002ce4: d7000139 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002cec: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s35, v71, s1                // 000000002cf0: d5207c3a 00068e23
	v_add3_u32 v59, v56, v88, 0x7fff                           // 000000002cf8: d655003b 03feb138 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d04: bf870003
	v_add_co_u32 v56, s1, v57, v118                            // 000000002d08: d7000138 0202ed39
	v_or_b32_e32 v60, 0x400000, v88                            // 000000002d10: 3878b0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d18: bf88f19f
	v_add_co_ci_u32_e64 v57, null, v58, v119, s1               // 000000002d1c: d5207c39 0006ef3a
	v_cmp_u_f32_e64 s1, v88, v88                               // 000000002d24: d4180001 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 000000002d2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d30: bf870001
	v_cndmask_b32_e64 v58, v59, v60, s1                        // 000000002d34: d501003a 0006793b
	global_store_d16_hi_b16 v[56:57], v58, off                 // 000000002d3c: ee09407c 1d000000 00000038
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002d4c: 8c7e027e
	v_add_co_u32 v58, s1, s38, v64                             // 000000002d50: d700013a 02028026
	s_wait_alu depctr_va_sdst(0)                               // 000000002d58: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s39, v65, s1                // 000000002d5c: d5207c3b 00068227
	v_cmp_lt_i64_e64 s1, 1, v[68:69]                           // 000000002d64: d4510001 02028881
	s_delay_alu instid0(valu_dep_2)                            // 000000002d6c: bf870002
	v_lshlrev_b64_e32 v[56:57], 1, v[58:59]                    // 000000002d70: 3e707481
	s_and_b32 s2, s1, vcc_lo                                   // 000000002d74: 8b026a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d78: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002d7c: be832002
	s_cbranch_execz 28                                         // 000000002d80: bfa5001c <packed_folded_w4a8+0x12f4>
	v_bfe_u32 v60, v89, 16, 1                                  // 000000002d84: d610003c 02052159
	s_wait_kmcnt 0x0                                           // 000000002d8c: bfc70000
	v_add_co_u32 v61, s2, s34, v70                             // 000000002d90: d700023d 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002d98: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s35, v71, s2                // 000000002d9c: d5207c3e 000a8e23
	v_add3_u32 v63, v60, v89, 0x7fff                           // 000000002da4: d655003f 03feb33c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002db0: bf870003
	v_add_co_u32 v60, s2, v61, v56                             // 000000002db4: d700023c 0202713d
	v_or_b32_e32 v64, 0x400000, v89                            // 000000002dbc: 3880b2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002dc4: bf88f19f
	v_add_co_ci_u32_e64 v61, null, v62, v57, s2                // 000000002dc8: d5207c3d 000a733e
	v_cmp_u_f32_e64 s2, v89, v89                               // 000000002dd0: d4180002 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 000000002dd8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ddc: bf870001
	v_cndmask_b32_e64 v62, v63, v64, s2                        // 000000002de0: d501003e 000a813f
	global_store_d16_hi_b16 v[60:61], v62, off                 // 000000002de8: ee09407c 1f000000 0000003c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002df4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002df8: 8c7e037e
	v_add_co_u32 v60, s2, v58, s38                             // 000000002dfc: d700023c 02004d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002e04: bf88f19f
	v_add_co_ci_u32_e64 v61, null, s39, v59, s2                // 000000002e08: d5207c3d 000a7627
	v_cmp_lt_i64_e64 s2, 2, v[68:69]                           // 000000002e10: d4510002 02028882
	s_delay_alu instid0(valu_dep_2)                            // 000000002e18: bf870002
	v_lshlrev_b64_e32 v[58:59], 1, v[60:61]                    // 000000002e1c: 3e747881
	s_and_b32 s3, s2, vcc_lo                                   // 000000002e20: 8b036a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e24: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000002e28: be842003
	s_cbranch_execz 28                                         // 000000002e2c: bfa5001c <packed_folded_w4a8+0x13a0>
	v_bfe_u32 v62, v91, 16, 1                                  // 000000002e30: d610003e 0205215b
	s_wait_kmcnt 0x0                                           // 000000002e38: bfc70000
	v_add_co_u32 v63, s3, s34, v70                             // 000000002e3c: d700033f 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002e44: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s35, v71, s3                // 000000002e48: d5207c40 000e8e23
	v_add3_u32 v65, v62, v91, 0x7fff                           // 000000002e50: d6550041 03feb73e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e5c: bf870003
	v_add_co_u32 v62, s3, v63, v58                             // 000000002e60: d700033e 0202753f
	v_or_b32_e32 v66, 0x400000, v91                            // 000000002e68: 3884b6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e70: bf88f19f
	v_add_co_ci_u32_e64 v63, null, v64, v59, s3                // 000000002e74: d5207c3f 000e7740
	v_cmp_u_f32_e64 s3, v91, v91                               // 000000002e7c: d4180003 0202b75b
	s_wait_alu depctr_va_sdst(0)                               // 000000002e84: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e88: bf870001
	v_cndmask_b32_e64 v64, v65, v66, s3                        // 000000002e8c: d5010040 000e8541
	global_store_d16_hi_b16 v[62:63], v64, off                 // 000000002e94: ee09407c 20000000 0000003e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ea0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002ea4: 8c7e047e
	v_add_co_u32 v62, s3, v60, s38                             // 000000002ea8: d700033e 02004d3c
	s_wait_alu depctr_va_sdst(0)                               // 000000002eb0: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s39, v61, s3                // 000000002eb4: d5207c3f 000e7a27
	v_cmp_lt_i64_e64 s3, 3, v[68:69]                           // 000000002ebc: d4510003 02028883
	s_delay_alu instid0(valu_dep_2)                            // 000000002ec4: bf870002
	v_lshlrev_b64_e32 v[60:61], 1, v[62:63]                    // 000000002ec8: 3e787c81
	s_and_b32 s4, s3, vcc_lo                                   // 000000002ecc: 8b046a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ed0: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000002ed4: be852004
	s_cbranch_execz 28                                         // 000000002ed8: bfa5001c <packed_folded_w4a8+0x144c>
	v_bfe_u32 v64, v92, 16, 1                                  // 000000002edc: d6100040 0205215c
	s_wait_kmcnt 0x0                                           // 000000002ee4: bfc70000
	v_add_co_u32 v65, s4, s34, v70                             // 000000002ee8: d7000441 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002ef0: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s35, v71, s4                // 000000002ef4: d5207c42 00128e23
	v_add3_u32 v67, v64, v92, 0x7fff                           // 000000002efc: d6550043 03feb940 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f08: bf870003
	v_add_co_u32 v64, s4, v65, v60                             // 000000002f0c: d7000440 02027941
	v_or_b32_e32 v88, 0x400000, v92                            // 000000002f14: 38b0b8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f1c: bf88f19f
	v_add_co_ci_u32_e64 v65, null, v66, v61, s4                // 000000002f20: d5207c41 00127b42
	v_cmp_u_f32_e64 s4, v92, v92                               // 000000002f28: d4180004 0202b95c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f30: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f34: bf870001
	v_cndmask_b32_e64 v66, v67, v88, s4                        // 000000002f38: d5010042 0012b143
	global_store_d16_hi_b16 v[64:65], v66, off                 // 000000002f40: ee09407c 21000000 00000040
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000002f50: 8c7e057e
	v_add_co_u32 v64, s4, v62, s38                             // 000000002f54: d7000440 02004d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000002f5c: bf88f19f
	v_add_co_ci_u32_e64 v65, null, s39, v63, s4                // 000000002f60: d5207c41 00127e27
	v_cmp_lt_i64_e64 s4, 4, v[68:69]                           // 000000002f68: d4510004 02028884
	s_delay_alu instid0(valu_dep_2)                            // 000000002f70: bf870002
	v_lshlrev_b64_e32 v[62:63], 1, v[64:65]                    // 000000002f74: 3e7c8081
	s_and_b32 s5, s4, vcc_lo                                   // 000000002f78: 8b056a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f7c: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000002f80: be862005
	s_cbranch_execz 28                                         // 000000002f84: bfa5001c <packed_folded_w4a8+0x14f8>
	v_bfe_u32 v66, v93, 16, 1                                  // 000000002f88: d6100042 0205215d
	s_wait_kmcnt 0x0                                           // 000000002f90: bfc70000
	v_add_co_u32 v67, s5, s34, v70                             // 000000002f94: d7000543 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002f9c: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s35, v71, s5                // 000000002fa0: d5207c58 00168e23
	v_add3_u32 v89, v66, v93, 0x7fff                           // 000000002fa8: d6550059 03febb42 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002fb4: bf870003
	v_add_co_u32 v66, s5, v67, v62                             // 000000002fb8: d7000542 02027d43
	v_or_b32_e32 v91, 0x400000, v93                            // 000000002fc0: 38b6baff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc8: bf88f19f
	v_add_co_ci_u32_e64 v67, null, v88, v63, s5                // 000000002fcc: d5207c43 00167f58
	v_cmp_u_f32_e64 s5, v93, v93                               // 000000002fd4: d4180005 0202bb5d
	s_wait_alu depctr_va_sdst(0)                               // 000000002fdc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fe0: bf870001
	v_cndmask_b32_e64 v88, v89, v91, s5                        // 000000002fe4: d5010058 0016b759
	global_store_d16_hi_b16 v[66:67], v88, off                 // 000000002fec: ee09407c 2c000000 00000042
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002ffc: 8c7e067e
	v_add_co_u32 v66, s5, v64, s38                             // 000000003000: d7000542 02004d40
	s_wait_alu depctr_va_sdst(0)                               // 000000003008: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s39, v65, s5                // 00000000300c: d5207c43 00168227
	v_cmp_lt_i64_e64 s5, 5, v[68:69]                           // 000000003014: d4510005 02028885
	s_delay_alu instid0(valu_dep_2)                            // 00000000301c: bf870002
	v_lshlrev_b64_e32 v[64:65], 1, v[66:67]                    // 000000003020: 3e808481
	s_and_b32 s6, s5, vcc_lo                                   // 000000003024: 8b066a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003028: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 00000000302c: be872006
	s_cbranch_execz 28                                         // 000000003030: bfa5001c <packed_folded_w4a8+0x15a4>
	v_bfe_u32 v88, v94, 16, 1                                  // 000000003034: d6100058 0205215e
	s_wait_kmcnt 0x0                                           // 00000000303c: bfc70000
	v_add_co_u32 v89, s6, s34, v70                             // 000000003040: d7000659 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000003048: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s35, v71, s6                // 00000000304c: d5207c5b 001a8e23
	v_add3_u32 v92, v88, v94, 0x7fff                           // 000000003054: d655005c 03febd58 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003060: bf870003
	v_add_co_u32 v88, s6, v89, v64                             // 000000003064: d7000658 02028159
	v_or_b32_e32 v93, 0x400000, v94                            // 00000000306c: 38babcff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003074: bf88f19f
	v_add_co_ci_u32_e64 v89, null, v91, v65, s6                // 000000003078: d5207c59 001a835b
	v_cmp_u_f32_e64 s6, v94, v94                               // 000000003080: d4180006 0202bd5e
	s_wait_alu depctr_va_sdst(0)                               // 000000003088: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000308c: bf870001
	v_cndmask_b32_e64 v91, v92, v93, s6                        // 000000003090: d501005b 001abb5c
	global_store_d16_hi_b16 v[88:89], v91, off                 // 000000003098: ee09407c 2d800000 00000058
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000030a8: 8c7e077e
	v_add_co_u32 v88, s6, v66, s38                             // 0000000030ac: d7000658 02004d42
	s_wait_alu depctr_va_sdst(0)                               // 0000000030b4: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s39, v67, s6                // 0000000030b8: d5207c59 001a8627
	v_cmp_lt_i64_e64 s6, 6, v[68:69]                           // 0000000030c0: d4510006 02028886
	s_delay_alu instid0(valu_dep_2)                            // 0000000030c8: bf870002
	v_lshlrev_b64_e32 v[66:67], 1, v[88:89]                    // 0000000030cc: 3e84b081
	s_and_b32 s7, s6, vcc_lo                                   // 0000000030d0: 8b076a06
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030d4: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 0000000030d8: be882007
	s_cbranch_execz 28                                         // 0000000030dc: bfa5001c <packed_folded_w4a8+0x1650>
	v_bfe_u32 v91, v95, 16, 1                                  // 0000000030e0: d610005b 0205215f
	s_wait_kmcnt 0x0                                           // 0000000030e8: bfc70000
	v_add_co_u32 v92, s7, s34, v70                             // 0000000030ec: d700075c 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f4: bf88f19f
	v_add_co_ci_u32_e64 v93, null, s35, v71, s7                // 0000000030f8: d5207c5d 001e8e23
	v_add3_u32 v94, v91, v95, 0x7fff                           // 000000003100: d655005e 03febf5b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000310c: bf870003
	v_add_co_u32 v91, s7, v92, v66                             // 000000003110: d700075b 0202855c
	v_or_b32_e32 v97, 0x400000, v95                            // 000000003118: 38c2beff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003120: bf88f19f
	v_add_co_ci_u32_e64 v92, null, v93, v67, s7                // 000000003124: d5207c5c 001e875d
	v_cmp_u_f32_e64 s7, v95, v95                               // 00000000312c: d4180007 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 000000003134: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003138: bf870001
	v_cndmask_b32_e64 v93, v94, v97, s7                        // 00000000313c: d501005d 001ec35e
	global_store_d16_hi_b16 v[91:92], v93, off                 // 000000003144: ee09407c 2e800000 0000005b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003150: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003154: 8c7e087e
	v_add_co_u32 v88, s7, v88, s38                             // 000000003158: d7000758 02004d58
	s_wait_alu depctr_va_sdst(0)                               // 000000003160: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s39, v89, s7                // 000000003164: d5207c59 001eb227
	v_cmp_lt_i64_e64 s7, 7, v[68:69]                           // 00000000316c: d4510007 02028887
	s_delay_alu instid0(valu_dep_2)                            // 000000003174: bf870002
	v_lshlrev_b64_e32 v[68:69], 1, v[88:89]                    // 000000003178: 3e88b081
	s_and_b32 s8, s7, vcc_lo                                   // 00000000317c: 8b086a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003180: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003184: be892008
	s_cbranch_execz 28                                         // 000000003188: bfa5001c <packed_folded_w4a8+0x16fc>
	v_bfe_u32 v88, v96, 16, 1                                  // 00000000318c: d6100058 02052160
	s_wait_kmcnt 0x0                                           // 000000003194: bfc70000
	v_add_co_u32 v89, s8, s34, v70                             // 000000003198: d7000859 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a0: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s35, v71, s8                // 0000000031a4: d5207c5b 00228e23
	v_add3_u32 v92, v88, v96, 0x7fff                           // 0000000031ac: d655005c 03fec158 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031b8: bf870003
	v_add_co_u32 v88, s8, v89, v68                             // 0000000031bc: d7000858 02028959
	v_or_b32_e32 v93, 0x400000, v96                            // 0000000031c4: 38bac0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031cc: bf88f19f
	v_add_co_ci_u32_e64 v89, null, v91, v69, s8                // 0000000031d0: d5207c59 00228b5b
	v_cmp_u_f32_e64 s8, v96, v96                               // 0000000031d8: d4180008 0202c160
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031e4: bf870001
	v_cndmask_b32_e64 v91, v92, v93, s8                        // 0000000031e8: d501005b 0022bb5c
	global_store_d16_hi_b16 v[88:89], v91, off                 // 0000000031f0: ee09407c 2d800000 00000058
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003200: 8c7e097e
	v_or_b32_e32 v98, s26, v90                                 // 000000003204: 38c4b41a
	v_mov_b32_e32 v101, s27                                    // 000000003208: 7eca021b
	v_mov_b32_e32 v99, s27                                     // 00000000320c: 7ec6021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003210: bf870093
	v_or_b32_e32 v100, v98, v115                               // 000000003214: 38c8e762
	v_cmp_gt_i64_e64 s8, s[36:37], v[100:101]                  // 000000003218: d4540008 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 000000003220: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003224: bf8700a1
	v_cndmask_b32_e64 v89, 0, v101, s8                         // 000000003228: d5010059 0022ca80
	v_cndmask_b32_e64 v88, 0, v100, s8                         // 000000003230: d5010058 0022c880
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]                    // 000000003238: 3eb0b082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000323c: bf870121
	v_add_co_u32 v88, s8, s28, v88                             // 000000003240: d7000858 0202b01c
	s_wait_alu depctr_va_sdst(0)                               // 000000003248: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s29, v89, s8                // 00000000324c: d5207c59 0022b21d
	global_load_u8 v91, v[110:111], off                        // 000000003254: ee04007c 0000005b 0000006e
	global_load_b32 v90, v[88:89], off                         // 000000003260: ee05007c 0000005a 00000058
	s_wait_loadcnt 0x1                                         // 00000000326c: bfc00001
	v_lshlrev_b32_e32 v106, 23, v91                            // 000000003270: 30d4b697
	s_wait_loadcnt 0x0                                         // 000000003274: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003278: bf870091
	v_mul_f32_e32 v91, v90, v106                               // 00000000327c: 10b6d55a
	v_cmp_class_f32_e64 s8, v91, 0x198                         // 000000003280: d47e0008 0201ff5b 00000198
	v_mul_f32_e32 v103, v48, v91                               // 00000000328c: 10ceb730
	s_xor_b32 s8, s8, -1                                       // 000000003290: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003294: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003298: be892008
	s_cbranch_execnz 3470                                      // 00000000329c: bfa60d8e <packed_folded_w4a8+0x4dd8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000032a4: 8c7e097e
	v_or_b32_e32 v90, v116, v98                                // 0000000032a8: 38b4c574
	v_mov_b32_e32 v91, v99                                     // 0000000032ac: 7eb60363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000032b0: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[90:91]                    // 0000000032b4: d4540008 0202b424
	s_wait_alu depctr_va_sdst(0)                               // 0000000032bc: bf88f19f
	v_cndmask_b32_e64 v91, 0, v91, s8                          // 0000000032c0: d501005b 0022b680
	v_cndmask_b32_e64 v90, 0, v90, s8                          // 0000000032c8: d501005a 0022b480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000032d0: bf870091
	v_lshlrev_b64_e32 v[90:91], 2, v[90:91]                    // 0000000032d4: 3eb4b482
	v_add_co_u32 v90, s8, s28, v90                             // 0000000032d8: d700085a 0202b41c
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000032e4: bf8700c2
	v_add_co_ci_u32_e64 v91, null, s29, v91, s8                // 0000000032e8: d5207c5b 0022b61d
	global_load_b32 v48, v[90:91], off                         // 0000000032f0: ee05007c 00000030 0000005a
	s_wait_loadcnt 0x0                                         // 0000000032fc: bfc00000
	v_mul_f32_e32 v92, v48, v106                               // 000000003300: 10b8d530
	v_cmp_class_f32_e64 s8, v92, 0x198                         // 000000003304: d47e0008 0201ff5c 00000198
	v_mul_f32_e32 v104, v49, v92                               // 000000003310: 10d0b931
	s_xor_b32 s8, s8, -1                                       // 000000003314: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003318: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000331c: be892008
	s_cbranch_execnz 3455                                      // 000000003320: bfa60d7f <packed_folded_w4a8+0x4e20>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003324: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003328: 8c7e097e
	v_or_b32_e32 v48, v117, v98                                // 00000000332c: 3860c575
	v_mov_b32_e32 v49, v99                                     // 000000003330: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003334: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003338: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003340: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 000000003344: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 00000000334c: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003354: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 000000003358: 3e606082
	v_add_co_u32 v92, s8, s28, v48                             // 00000000335c: d700085c 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 000000003364: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003368: bf8700c2
	v_add_co_ci_u32_e64 v93, null, s29, v49, s8                // 00000000336c: d5207c5d 0022621d
	global_load_b32 v48, v[92:93], off                         // 000000003374: ee05007c 00000030 0000005c
	s_wait_loadcnt 0x0                                         // 000000003380: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003384: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003388: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v105, v50, v49                               // 000000003394: 10d26332
	s_xor_b32 s8, s8, -1                                       // 000000003398: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 00000000339c: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000033a0: be892008
	s_cbranch_execnz 3440                                      // 0000000033a4: bfa60d70 <packed_folded_w4a8+0x4e68>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000033ac: 8c7e097e
	v_or_b32_e32 v48, v128, v98                                // 0000000033b0: 3860c580
	v_mov_b32_e32 v49, v99                                     // 0000000033b4: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000033b8: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 0000000033bc: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 0000000033c4: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 0000000033c8: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 0000000033d0: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033d8: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000033dc: 3e606082
	v_add_co_u32 v94, s8, s28, v48                             // 0000000033e0: d700085e 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e8: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000033ec: bf8700c2
	v_add_co_ci_u32_e64 v95, null, s29, v49, s8                // 0000000033f0: d5207c5f 0022621d
	global_load_b32 v48, v[94:95], off                         // 0000000033f8: ee05007c 00000030 0000005e
	s_wait_loadcnt 0x0                                         // 000000003404: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003408: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 00000000340c: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v107, v51, v49                               // 000000003418: 10d66333
	s_xor_b32 s8, s8, -1                                       // 00000000341c: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003420: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003424: be892008
	s_cbranch_execnz 3425                                      // 000000003428: bfa60d61 <packed_folded_w4a8+0x4eb0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000342c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003430: 8c7e097e
	v_or_b32_e32 v48, v129, v98                                // 000000003434: 3860c581
	v_mov_b32_e32 v49, v99                                     // 000000003438: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 00000000343c: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003440: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003448: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 00000000344c: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 000000003454: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000345c: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 000000003460: 3e606082
	v_add_co_u32 v50, s8, s28, v48                             // 000000003464: d7000832 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 00000000346c: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003470: bf8700c2
	v_add_co_ci_u32_e64 v51, null, s29, v49, s8                // 000000003474: d5207c33 0022621d
	global_load_b32 v48, v[50:51], off                         // 00000000347c: ee05007c 00000030 00000032
	s_wait_loadcnt 0x0                                         // 000000003488: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 00000000348c: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003490: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v108, v52, v49                               // 00000000349c: 10d86334
	s_xor_b32 s8, s8, -1                                       // 0000000034a0: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034a4: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000034a8: be892008
	s_cbranch_execnz 3410                                      // 0000000034ac: bfa60d52 <packed_folded_w4a8+0x4ef8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000034b4: 8c7e097e
	v_or_b32_e32 v48, v130, v98                                // 0000000034b8: 3860c582
	v_mov_b32_e32 v49, v99                                     // 0000000034bc: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000034c0: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 0000000034c4: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 0000000034cc: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 0000000034d0: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 0000000034d8: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000034e0: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000034e4: 3e606082
	v_add_co_u32 v96, s8, s28, v48                             // 0000000034e8: d7000860 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000034f4: bf8700c2
	v_add_co_ci_u32_e64 v97, null, s29, v49, s8                // 0000000034f8: d5207c61 0022621d
	global_load_b32 v48, v[96:97], off                         // 000000003500: ee05007c 00000030 00000060
	s_wait_loadcnt 0x0                                         // 00000000350c: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003510: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003514: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v109, v53, v49                               // 000000003520: 10da6335
	s_xor_b32 s8, s8, -1                                       // 000000003524: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003528: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000352c: be892008
	s_cbranch_execnz 3395                                      // 000000003530: bfa60d43 <packed_folded_w4a8+0x4f40>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003534: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003538: 8c7e097e
	v_or_b32_e32 v48, v131, v98                                // 00000000353c: 3860c583
	v_mov_b32_e32 v49, v99                                     // 000000003540: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003544: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003548: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003550: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 000000003554: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 00000000355c: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003564: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 000000003568: 3e606082
	v_add_co_u32 v52, s8, s28, v48                             // 00000000356c: d7000834 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 000000003574: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003578: bf8700c2
	v_add_co_ci_u32_e64 v53, null, s29, v49, s8                // 00000000357c: d5207c35 0022621d
	global_load_b32 v48, v[52:53], off                         // 000000003584: ee05007c 00000030 00000034
	s_wait_loadcnt 0x0                                         // 000000003590: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003594: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003598: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v112, v54, v49                               // 0000000035a4: 10e06336
	s_xor_b32 s8, s8, -1                                       // 0000000035a8: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035ac: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000035b0: be892008
	s_cbranch_execnz 3380                                      // 0000000035b4: bfa60d34 <packed_folded_w4a8+0x4f88>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000035bc: 8c7e097e
	v_or_b32_e32 v98, v132, v98                                // 0000000035c0: 38c4c584
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000035c4: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[98:99]                    // 0000000035c8: d4540008 0202c424
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d0: bf88f19f
	v_cndmask_b32_e64 v49, 0, v99, s8                          // 0000000035d4: d5010031 0022c680
	v_cndmask_b32_e64 v48, 0, v98, s8                          // 0000000035dc: d5010030 0022c480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035e4: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000035e8: 3e606082
	v_add_co_u32 v98, s8, s28, v48                             // 0000000035ec: d7000862 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000035f4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000035f8: bf8700c2
	v_add_co_ci_u32_e64 v99, null, s29, v49, s8                // 0000000035fc: d5207c63 0022621d
	global_load_b32 v48, v[98:99], off                         // 000000003604: ee05007c 00000030 00000062
	s_wait_loadcnt 0x0                                         // 000000003610: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003614: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003618: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v113, v55, v49                               // 000000003624: 10e26337
	s_xor_b32 s8, s8, -1                                       // 000000003628: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 00000000362c: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003630: be892008
	s_cbranch_execnz 3366                                      // 000000003634: bfa60d26 <packed_folded_w4a8+0x4fd0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003638: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000363c: 8c7e097e
	v_mul_lo_u32 v106, s39, v100                               // 000000003640: d72c006a 0202c827
	v_mul_lo_u32 v120, s38, v101                               // 000000003648: d72c0078 0202ca26
	v_mad_co_u64_u32 v[48:49], null, s38, v100, 0              // 000000003650: d6fe7c30 0202c826
	v_sub_co_u32 v54, s8, s36, v100                            // 000000003658: d7010836 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 000000003660: bf88f19f
	v_sub_co_ci_u32_e64 v55, null, s37, v101, s8               // 000000003664: d5217c37 0022ca25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 00000000366c: bf870211
	v_cmp_lt_i64_e64 s11, 0, v[54:55]                          // 000000003670: d451000b 02026c80
	v_add3_u32 v49, v49, v120, v106                            // 000000003678: d6550031 05aaf131
	s_delay_alu instid0(valu_dep_1)                            // 000000003680: bf870001
	v_lshlrev_b64_e32 v[48:49], 1, v[48:49]                    // 000000003684: 3e606081
	s_and_b32 s8, s11, vcc_lo                                  // 000000003688: 8b086a0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000368c: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003690: be892008
	s_cbranch_execz 28                                         // 000000003694: bfa5001c <packed_folded_w4a8+0x1c08>
	v_bfe_u32 v100, v103, 16, 1                                // 000000003698: d6100064 02052167
	s_wait_kmcnt 0x0                                           // 0000000036a0: bfc70000
	v_add_co_u32 v101, s8, s34, v48                            // 0000000036a4: d7000865 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ac: bf88f19f
	v_add_co_ci_u32_e64 v106, null, s35, v49, s8               // 0000000036b0: d5207c6a 00226223
	v_add3_u32 v120, v100, v103, 0x7fff                        // 0000000036b8: d6550078 03fecf64 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000036c4: bf870003
	v_add_co_u32 v100, s8, v101, v118                          // 0000000036c8: d7000864 0202ed65
	v_or_b32_e32 v121, 0x400000, v103                          // 0000000036d0: 38f2ceff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036d8: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v106, v119, s8             // 0000000036dc: d5207c65 0022ef6a
	v_cmp_u_f32_e64 s8, v103, v103                             // 0000000036e4: d4180008 0202cf67
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036f0: bf870001
	v_cndmask_b32_e64 v103, v120, v121, s8                     // 0000000036f4: d5010067 0022f378
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000036fc: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003708: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000370c: 8c7e097e
	v_cmp_lt_i64_e64 s8, 1, v[54:55]                           // 000000003710: d4510008 02026c81
	s_and_b32 s9, s8, vcc_lo                                   // 000000003718: 8b096a08
	s_wait_alu depctr_sa_sdst(0)                               // 00000000371c: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000003720: be8a2009
	s_cbranch_execz 28                                         // 000000003724: bfa5001c <packed_folded_w4a8+0x1c98>
	v_bfe_u32 v100, v104, 16, 1                                // 000000003728: d6100064 02052168
	s_wait_kmcnt 0x0                                           // 000000003730: bfc70000
	v_add_co_u32 v101, s9, s34, v48                            // 000000003734: d7000965 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000373c: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s9               // 000000003740: d5207c67 00266223
	v_add3_u32 v106, v100, v104, 0x7fff                        // 000000003748: d655006a 03fed164 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003754: bf870003
	v_add_co_u32 v100, s9, v101, v56                           // 000000003758: d7000964 02027165
	v_or_b32_e32 v120, 0x400000, v104                          // 000000003760: 38f0d0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003768: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v57, s9              // 00000000376c: d5207c65 00267367
	v_cmp_u_f32_e64 s9, v104, v104                             // 000000003774: d4180009 0202d168
	s_wait_alu depctr_va_sdst(0)                               // 00000000377c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003780: bf870001
	v_cndmask_b32_e64 v103, v106, v120, s9                     // 000000003784: d5010067 0026f16a
	global_store_d16_hi_b16 v[100:101], v103, off              // 00000000378c: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003798: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 00000000379c: 8c7e0a7e
	v_cmp_lt_i64_e64 s9, 2, v[54:55]                           // 0000000037a0: d4510009 02026c82
	s_and_b32 s10, s9, vcc_lo                                  // 0000000037a8: 8b0a6a09
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037ac: bf88ff9e
	s_and_saveexec_b32 s12, s10                                // 0000000037b0: be8c200a
	s_cbranch_execz 28                                         // 0000000037b4: bfa5001c <packed_folded_w4a8+0x1d28>
	v_bfe_u32 v100, v105, 16, 1                                // 0000000037b8: d6100064 02052169
	s_wait_kmcnt 0x0                                           // 0000000037c0: bfc70000
	v_add_co_u32 v101, s10, s34, v48                           // 0000000037c4: d7000a65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000037cc: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s10              // 0000000037d0: d5207c67 002a6223
	v_add3_u32 v104, v100, v105, 0x7fff                        // 0000000037d8: d6550068 03fed364 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000037e4: bf870003
	v_add_co_u32 v100, s10, v101, v58                          // 0000000037e8: d7000a64 02027565
	v_or_b32_e32 v106, 0x400000, v105                          // 0000000037f0: 38d4d2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f8: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v59, s10             // 0000000037fc: d5207c65 002a7767
	v_cmp_u_f32_e64 s10, v105, v105                            // 000000003804: d418000a 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 00000000380c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003810: bf870001
	v_cndmask_b32_e64 v103, v104, v106, s10                    // 000000003814: d5010067 002ad568
	global_store_d16_hi_b16 v[100:101], v103, off              // 00000000381c: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003828: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 00000000382c: 8c7e0c7e
	v_cmp_lt_i64_e64 s10, 3, v[54:55]                          // 000000003830: d451000a 02026c83
	s_and_b32 s12, s10, vcc_lo                                 // 000000003838: 8b0c6a0a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000383c: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000003840: be8d200c
	s_cbranch_execz 28                                         // 000000003844: bfa5001c <packed_folded_w4a8+0x1db8>
	v_bfe_u32 v100, v107, 16, 1                                // 000000003848: d6100064 0205216b
	s_wait_kmcnt 0x0                                           // 000000003850: bfc70000
	v_add_co_u32 v101, s12, s34, v48                           // 000000003854: d7000c65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000385c: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s12              // 000000003860: d5207c67 00326223
	v_add3_u32 v104, v100, v107, 0x7fff                        // 000000003868: d6550068 03fed764 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003874: bf870003
	v_add_co_u32 v100, s12, v101, v60                          // 000000003878: d7000c64 02027965
	v_or_b32_e32 v105, 0x400000, v107                          // 000000003880: 38d2d6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003888: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v61, s12             // 00000000388c: d5207c65 00327b67
	v_cmp_u_f32_e64 s12, v107, v107                            // 000000003894: d418000c 0202d76b
	s_wait_alu depctr_va_sdst(0)                               // 00000000389c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038a0: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s12                    // 0000000038a4: d5010067 0032d368
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000038ac: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 0000000038bc: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 4, v[54:55]                          // 0000000038c0: d451000c 02026c84
	s_and_b32 s13, s12, vcc_lo                                 // 0000000038c8: 8b0d6a0c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038cc: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 0000000038d0: be8e200d
	s_cbranch_execz 28                                         // 0000000038d4: bfa5001c <packed_folded_w4a8+0x1e48>
	v_bfe_u32 v100, v108, 16, 1                                // 0000000038d8: d6100064 0205216c
	s_wait_kmcnt 0x0                                           // 0000000038e0: bfc70000
	v_add_co_u32 v101, s13, s34, v48                           // 0000000038e4: d7000d65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000038ec: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s13              // 0000000038f0: d5207c67 00366223
	v_add3_u32 v104, v100, v108, 0x7fff                        // 0000000038f8: d6550068 03fed964 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003904: bf870003
	v_add_co_u32 v100, s13, v101, v62                          // 000000003908: d7000d64 02027d65
	v_or_b32_e32 v105, 0x400000, v108                          // 000000003910: 38d2d8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003918: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v63, s13             // 00000000391c: d5207c65 00367f67
	v_cmp_u_f32_e64 s13, v108, v108                            // 000000003924: d418000d 0202d96c
	s_wait_alu depctr_va_sdst(0)                               // 00000000392c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003930: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s13                    // 000000003934: d5010067 0036d368
	global_store_d16_hi_b16 v[100:101], v103, off              // 00000000393c: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003948: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 00000000394c: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 5, v[54:55]                          // 000000003950: d451000d 02026c85
	s_and_b32 s14, s13, vcc_lo                                 // 000000003958: 8b0e6a0d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000395c: bf88ff9e
	s_and_saveexec_b32 s15, s14                                // 000000003960: be8f200e
	s_cbranch_execz 28                                         // 000000003964: bfa5001c <packed_folded_w4a8+0x1ed8>
	v_bfe_u32 v100, v109, 16, 1                                // 000000003968: d6100064 0205216d
	s_wait_kmcnt 0x0                                           // 000000003970: bfc70000
	v_add_co_u32 v101, s14, s34, v48                           // 000000003974: d7000e65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s14              // 000000003980: d5207c67 003a6223
	v_add3_u32 v104, v100, v109, 0x7fff                        // 000000003988: d6550068 03fedb64 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003994: bf870003
	v_add_co_u32 v100, s14, v101, v64                          // 000000003998: d7000e64 02028165
	v_or_b32_e32 v105, 0x400000, v109                          // 0000000039a0: 38d2daff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a8: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v65, s14             // 0000000039ac: d5207c65 003a8367
	v_cmp_u_f32_e64 s14, v109, v109                            // 0000000039b4: d418000e 0202db6d
	s_wait_alu depctr_va_sdst(0)                               // 0000000039bc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039c0: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s14                    // 0000000039c4: d5010067 003ad368
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000039cc: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000039dc: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 6, v[54:55]                          // 0000000039e0: d451000e 02026c86
	s_and_b32 s15, s14, vcc_lo                                 // 0000000039e8: 8b0f6a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039ec: bf88ff9e
	s_and_saveexec_b32 s16, s15                                // 0000000039f0: be90200f
	s_cbranch_execz 28                                         // 0000000039f4: bfa5001c <packed_folded_w4a8+0x1f68>
	v_bfe_u32 v100, v112, 16, 1                                // 0000000039f8: d6100064 02052170
	s_wait_kmcnt 0x0                                           // 000000003a00: bfc70000
	v_add_co_u32 v101, s15, s34, v48                           // 000000003a04: d7000f65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003a0c: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s15              // 000000003a10: d5207c67 003e6223
	v_add3_u32 v104, v100, v112, 0x7fff                        // 000000003a18: d6550068 03fee164 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003a24: bf870003
	v_add_co_u32 v100, s15, v101, v66                          // 000000003a28: d7000f64 02028565
	v_or_b32_e32 v105, 0x400000, v112                          // 000000003a30: 38d2e0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a38: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v67, s15             // 000000003a3c: d5207c65 003e8767
	v_cmp_u_f32_e64 s15, v112, v112                            // 000000003a44: d418000f 0202e170
	s_wait_alu depctr_va_sdst(0)                               // 000000003a4c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a50: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s15                    // 000000003a54: d5010067 003ed368
	global_store_d16_hi_b16 v[100:101], v103, off              // 000000003a5c: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s16                             // 000000003a6c: 8c7e107e
	v_cmp_lt_i64_e64 s15, 7, v[54:55]                          // 000000003a70: d451000f 02026c87
	s_and_b32 s16, s15, vcc_lo                                 // 000000003a78: 8b106a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a7c: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003a80: be912010
	s_cbranch_execz 28                                         // 000000003a84: bfa5001c <packed_folded_w4a8+0x1ff8>
	v_bfe_u32 v54, v113, 16, 1                                 // 000000003a88: d6100036 02052171
	s_wait_kmcnt 0x0                                           // 000000003a90: bfc70000
	v_add_co_u32 v55, s16, s34, v48                            // 000000003a94: d7001037 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003a9c: bf88f19f
	v_add_co_ci_u32_e64 v100, null, s35, v49, s16              // 000000003aa0: d5207c64 00426223
	v_add3_u32 v101, v54, v113, 0x7fff                         // 000000003aa8: d6550065 03fee336 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003ab4: bf870003
	v_add_co_u32 v54, s16, v55, v68                            // 000000003ab8: d7001036 02028937
	v_or_b32_e32 v103, 0x400000, v113                          // 000000003ac0: 38cee2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac8: bf88f19f
	v_add_co_ci_u32_e64 v55, null, v100, v69, s16              // 000000003acc: d5207c37 00428b64
	v_cmp_u_f32_e64 s16, v113, v113                            // 000000003ad4: d4180010 0202e371
	s_wait_alu depctr_va_sdst(0)                               // 000000003adc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ae0: bf870001
	v_cndmask_b32_e64 v100, v101, v103, s16                    // 000000003ae4: d5010064 0042cf65
	global_store_d16_hi_b16 v[54:55], v100, off                // 000000003aec: ee09407c 32000000 00000036
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003afc: 8c7e117e
	v_or_b32_e32 v108, s26, v102                               // 000000003b00: 38d8cc1a
	v_mov_b32_e32 v113, s27                                    // 000000003b04: 7ee2021b
	v_mov_b32_e32 v109, s27                                    // 000000003b08: 7eda021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003b0c: bf870093
	v_or_b32_e32 v112, v108, v115                              // 000000003b10: 38e0e76c
	v_cmp_gt_i64_e64 s16, s[36:37], v[112:113]                 // 000000003b14: d4540010 0202e024
	s_wait_alu depctr_va_sdst(0)                               // 000000003b1c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003b20: bf8700a1
	v_cndmask_b32_e64 v55, 0, v113, s16                        // 000000003b24: d5010037 0042e280
	v_cndmask_b32_e64 v54, 0, v112, s16                        // 000000003b2c: d5010036 0042e080
	v_lshlrev_b64_e32 v[54:55], 2, v[54:55]                    // 000000003b34: 3e6c6c82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003b38: bf870121
	v_add_co_u32 v54, s16, s28, v54                            // 000000003b3c: d7001036 02026c1c
	s_wait_alu depctr_va_sdst(0)                               // 000000003b44: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s16               // 000000003b48: d5207c37 00426e1d
	global_load_u8 v101, v[110:111], off                       // 000000003b50: ee04007c 00000065 0000006e
	global_load_b32 v100, v[54:55], off                        // 000000003b5c: ee05007c 00000064 00000036
	s_wait_loadcnt 0x1                                         // 000000003b68: bfc00001
	v_lshlrev_b32_e32 v123, 23, v101                           // 000000003b6c: 30f6ca97
	s_wait_loadcnt 0x0                                         // 000000003b70: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003b74: bf870091
	v_mul_f32_e32 v101, v100, v123                             // 000000003b78: 10caf764
	v_cmp_class_f32_e64 s16, v101, 0x198                       // 000000003b7c: d47e0010 0201ff65 00000198
	v_mul_f32_e32 v120, v40, v101                              // 000000003b88: 10f0cb28
	s_xor_b32 s16, s16, -1                                     // 000000003b8c: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b90: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003b94: be912010
	s_cbranch_execnz 3039                                      // 000000003b98: bfa60bdf <packed_folded_w4a8+0x5018>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003ba0: 8c7e117e
	v_or_b32_e32 v100, v116, v108                              // 000000003ba4: 38c8d974
	v_mov_b32_e32 v101, v109                                   // 000000003ba8: 7eca036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003bac: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[100:101]                 // 000000003bb0: d4540010 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb8: bf88f19f
	v_cndmask_b32_e64 v101, 0, v101, s16                       // 000000003bbc: d5010065 0042ca80
	v_cndmask_b32_e64 v100, 0, v100, s16                       // 000000003bc4: d5010064 0042c880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003bcc: bf870091
	v_lshlrev_b64_e32 v[100:101], 2, v[100:101]                // 000000003bd0: 3ec8c882
	v_add_co_u32 v100, s16, s28, v100                          // 000000003bd4: d7001064 0202c81c
	s_wait_alu depctr_va_sdst(0)                               // 000000003bdc: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003be0: bf8700c2
	v_add_co_ci_u32_e64 v101, null, s29, v101, s16             // 000000003be4: d5207c65 0042ca1d
	global_load_b32 v40, v[100:101], off                       // 000000003bec: ee05007c 00000028 00000064
	s_wait_loadcnt 0x0                                         // 000000003bf8: bfc00000
	v_mul_f32_e32 v102, v40, v123                              // 000000003bfc: 10ccf728
	v_cmp_class_f32_e64 s16, v102, 0x198                       // 000000003c00: d47e0010 0201ff66 00000198
	v_mul_f32_e32 v121, v41, v102                              // 000000003c0c: 10f2cd29
	s_xor_b32 s16, s16, -1                                     // 000000003c10: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c14: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003c18: be912010
	s_cbranch_execnz 3024                                      // 000000003c1c: bfa60bd0 <packed_folded_w4a8+0x5060>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003c24: 8c7e117e
	v_or_b32_e32 v40, v117, v108                               // 000000003c28: 3850d975
	v_mov_b32_e32 v41, v109                                    // 000000003c2c: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003c30: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003c34: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003c3c: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003c40: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003c48: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c50: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003c54: 3e505082
	v_add_co_u32 v102, s16, s28, v40                           // 000000003c58: d7001066 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c60: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003c64: bf8700c2
	v_add_co_ci_u32_e64 v103, null, s29, v41, s16              // 000000003c68: d5207c67 0042521d
	global_load_b32 v40, v[102:103], off                       // 000000003c70: ee05007c 00000028 00000066
	s_wait_loadcnt 0x0                                         // 000000003c7c: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003c80: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003c84: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v122, v42, v41                               // 000000003c90: 10f4532a
	s_xor_b32 s16, s16, -1                                     // 000000003c94: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c98: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003c9c: be912010
	s_cbranch_execnz 3009                                      // 000000003ca0: bfa60bc1 <packed_folded_w4a8+0x50a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ca4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003ca8: 8c7e117e
	v_or_b32_e32 v40, v128, v108                               // 000000003cac: 3850d980
	v_mov_b32_e32 v41, v109                                    // 000000003cb0: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003cb4: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003cb8: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc0: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003cc4: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003ccc: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cd4: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003cd8: 3e505082
	v_add_co_u32 v104, s16, s28, v40                           // 000000003cdc: d7001068 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003ce8: bf8700c2
	v_add_co_ci_u32_e64 v105, null, s29, v41, s16              // 000000003cec: d5207c69 0042521d
	global_load_b32 v40, v[104:105], off                       // 000000003cf4: ee05007c 00000028 00000068
	s_wait_loadcnt 0x0                                         // 000000003d00: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003d04: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003d08: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v133, v43, v41                               // 000000003d14: 110a532b
	s_xor_b32 s16, s16, -1                                     // 000000003d18: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d1c: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003d20: be912010
	s_cbranch_execnz 2994                                      // 000000003d24: bfa60bb2 <packed_folded_w4a8+0x50f0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003d2c: 8c7e117e
	v_or_b32_e32 v40, v129, v108                               // 000000003d30: 3850d981
	v_mov_b32_e32 v41, v109                                    // 000000003d34: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d38: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003d3c: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003d44: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003d48: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003d50: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d58: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003d5c: 3e505082
	v_add_co_u32 v42, s16, s28, v40                            // 000000003d60: d700102a 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003d68: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003d6c: bf8700c2
	v_add_co_ci_u32_e64 v43, null, s29, v41, s16               // 000000003d70: d5207c2b 0042521d
	global_load_b32 v40, v[42:43], off                         // 000000003d78: ee05007c 00000028 0000002a
	s_wait_loadcnt 0x0                                         // 000000003d84: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003d88: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003d8c: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v134, v44, v41                               // 000000003d98: 110c532c
	s_xor_b32 s16, s16, -1                                     // 000000003d9c: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003da0: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003da4: be912010
	s_cbranch_execnz 2979                                      // 000000003da8: bfa60ba3 <packed_folded_w4a8+0x5138>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003db0: 8c7e117e
	v_or_b32_e32 v40, v130, v108                               // 000000003db4: 3850d982
	v_mov_b32_e32 v41, v109                                    // 000000003db8: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003dbc: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003dc0: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc8: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003dcc: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003dd4: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ddc: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003de0: 3e505082
	v_add_co_u32 v106, s16, s28, v40                           // 000000003de4: d700106a 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003dec: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003df0: bf8700c2
	v_add_co_ci_u32_e64 v107, null, s29, v41, s16              // 000000003df4: d5207c6b 0042521d
	global_load_b32 v40, v[106:107], off                       // 000000003dfc: ee05007c 00000028 0000006a
	s_wait_loadcnt 0x0                                         // 000000003e08: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003e0c: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003e10: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v135, v45, v41                               // 000000003e1c: 110e532d
	s_xor_b32 s16, s16, -1                                     // 000000003e20: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e24: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003e28: be912010
	s_cbranch_execnz 2964                                      // 000000003e2c: bfa60b94 <packed_folded_w4a8+0x5180>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003e34: 8c7e117e
	v_or_b32_e32 v40, v131, v108                               // 000000003e38: 3850d983
	v_mov_b32_e32 v41, v109                                    // 000000003e3c: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003e40: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003e44: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003e4c: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003e50: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003e58: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e60: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003e64: 3e505082
	v_add_co_u32 v44, s16, s28, v40                            // 000000003e68: d700102c 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003e70: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003e74: bf8700c2
	v_add_co_ci_u32_e64 v45, null, s29, v41, s16               // 000000003e78: d5207c2d 0042521d
	global_load_b32 v40, v[44:45], off                         // 000000003e80: ee05007c 00000028 0000002c
	s_wait_loadcnt 0x0                                         // 000000003e8c: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003e90: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003e94: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v136, v46, v41                               // 000000003ea0: 1110532e
	s_xor_b32 s16, s16, -1                                     // 000000003ea4: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ea8: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003eac: be912010
	s_cbranch_execnz 2949                                      // 000000003eb0: bfa60b85 <packed_folded_w4a8+0x51c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003eb8: 8c7e117e
	v_or_b32_e32 v108, v132, v108                              // 000000003ebc: 38d8d984
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003ec0: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[108:109]                 // 000000003ec4: d4540010 0202d824
	s_wait_alu depctr_va_sdst(0)                               // 000000003ecc: bf88f19f
	v_cndmask_b32_e64 v41, 0, v109, s16                        // 000000003ed0: d5010029 0042da80
	v_cndmask_b32_e64 v40, 0, v108, s16                        // 000000003ed8: d5010028 0042d880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ee0: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003ee4: 3e505082
	v_add_co_u32 v108, s16, s28, v40                           // 000000003ee8: d700106c 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003ef0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003ef4: bf8700c2
	v_add_co_ci_u32_e64 v109, null, s29, v41, s16              // 000000003ef8: d5207c6d 0042521d
	global_load_b32 v40, v[108:109], off                       // 000000003f00: ee05007c 00000028 0000006c
	s_wait_loadcnt 0x0                                         // 000000003f0c: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003f10: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003f14: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v137, v47, v41                               // 000000003f20: 1112532f
	s_xor_b32 s16, s16, -1                                     // 000000003f24: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f28: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003f2c: be912010
	s_cbranch_execnz 2935                                      // 000000003f30: bfa60b77 <packed_folded_w4a8+0x5210>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003f38: 8c7e117e
	v_mul_lo_u32 v123, s39, v112                               // 000000003f3c: d72c007b 0202e027
	v_mul_lo_u32 v138, s38, v113                               // 000000003f44: d72c008a 0202e226
	v_mad_co_u64_u32 v[40:41], null, s38, v112, 0              // 000000003f4c: d6fe7c28 0202e026
	v_sub_co_u32 v46, s16, s36, v112                           // 000000003f54: d701102e 0202e024
	s_wait_alu depctr_va_sdst(0)                               // 000000003f5c: bf88f19f
	v_sub_co_ci_u32_e64 v47, null, s37, v113, s16              // 000000003f60: d5217c2f 0042e225
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003f68: bf870211
	v_cmp_lt_i64_e64 s19, 0, v[46:47]                          // 000000003f6c: d4510013 02025c80
	v_add3_u32 v41, v41, v138, v123                            // 000000003f74: d6550029 05ef1529
	s_delay_alu instid0(valu_dep_1)                            // 000000003f7c: bf870001
	v_lshlrev_b64_e32 v[40:41], 1, v[40:41]                    // 000000003f80: 3e505081
	s_and_b32 s16, s19, vcc_lo                                 // 000000003f84: 8b106a13
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f88: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003f8c: be912010
	s_cbranch_execz 28                                         // 000000003f90: bfa5001c <packed_folded_w4a8+0x2504>
	v_bfe_u32 v112, v120, 16, 1                                // 000000003f94: d6100070 02052178
	s_wait_kmcnt 0x0                                           // 000000003f9c: bfc70000
	v_add_co_u32 v113, s16, s34, v40                           // 000000003fa0: d7001071 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa8: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s35, v41, s16              // 000000003fac: d5207c7b 00425223
	v_add3_u32 v138, v112, v120, 0x7fff                        // 000000003fb4: d655008a 03fef170 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003fc0: bf870003
	v_add_co_u32 v112, s16, v113, v118                         // 000000003fc4: d7001070 0202ed71
	v_or_b32_e32 v139, 0x400000, v120                          // 000000003fcc: 3916f0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003fd4: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v123, v119, s16            // 000000003fd8: d5207c71 0042ef7b
	v_cmp_u_f32_e64 s16, v120, v120                            // 000000003fe0: d4180010 0202f178
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003fec: bf870001
	v_cndmask_b32_e64 v120, v138, v139, s16                    // 000000003ff0: d5010078 0043178a
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000003ff8: ee09407c 3c000000 00000070
	s_wait_alu depctr_sa_sdst(0)                               // 000000004004: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000004008: 8c7e117e
	v_cmp_lt_i64_e64 s16, 1, v[46:47]                          // 00000000400c: d4510010 02025c81
	s_and_b32 s17, s16, vcc_lo                                 // 000000004014: 8b116a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000004018: bf88ff9e
	s_and_saveexec_b32 s18, s17                                // 00000000401c: be922011
	s_cbranch_execz 28                                         // 000000004020: bfa5001c <packed_folded_w4a8+0x2594>
	v_bfe_u32 v112, v121, 16, 1                                // 000000004024: d6100070 02052179
	s_wait_kmcnt 0x0                                           // 00000000402c: bfc70000
	v_add_co_u32 v113, s17, s34, v40                           // 000000004030: d7001171 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000004038: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s17              // 00000000403c: d5207c78 00465223
	v_add3_u32 v123, v112, v121, 0x7fff                        // 000000004044: d655007b 03fef370 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004050: bf870003
	v_add_co_u32 v112, s17, v113, v56                          // 000000004054: d7001170 02027171
	v_or_b32_e32 v138, 0x400000, v121                          // 00000000405c: 3914f2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004064: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v57, s17             // 000000004068: d5207c71 00467378
	v_cmp_u_f32_e64 s17, v121, v121                            // 000000004070: d4180011 0202f379
	s_wait_alu depctr_va_sdst(0)                               // 000000004078: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000407c: bf870001
	v_cndmask_b32_e64 v120, v123, v138, s17                    // 000000004080: d5010078 0047157b
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004088: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s18                             // 000000004094: 8c7e127e
	v_cmp_lt_i64_e64 s17, 2, v[46:47]                          // 000000004098: d4510011 02025c82
	s_and_b32 s18, s17, vcc_lo                                 // 0000000040a0: 8b126a11
	s_delay_alu instid0(salu_cycle_1)                          // 0000000040a4: bf870009
	s_and_saveexec_b32 s20, s18                                // 0000000040a8: be942012
	s_cbranch_execz 28                                         // 0000000040ac: bfa5001c <packed_folded_w4a8+0x2620>
	v_bfe_u32 v112, v122, 16, 1                                // 0000000040b0: d6100070 0205217a
	s_wait_kmcnt 0x0                                           // 0000000040b8: bfc70000
	v_add_co_u32 v113, s18, s34, v40                           // 0000000040bc: d7001271 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 0000000040c4: bf870191
	v_add_co_ci_u32_e64 v120, null, s35, v41, s18              // 0000000040c8: d5207c78 004a5223
	v_add3_u32 v121, v112, v122, 0x7fff                        // 0000000040d0: d6550079 03fef570 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000040dc: bf870003
	v_add_co_u32 v112, s18, v113, v58                          // 0000000040e0: d7001270 02027571
	v_or_b32_e32 v123, 0x400000, v122                          // 0000000040e8: 38f6f4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000040f0: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v59, s18             // 0000000040f4: d5207c71 004a7778
	v_cmp_u_f32_e64 s18, v122, v122                            // 0000000040fc: d4180012 0202f57a
	s_wait_alu depctr_va_sdst(0)                               // 000000004104: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004108: bf870001
	v_cndmask_b32_e64 v120, v121, v123, s18                    // 00000000410c: d5010078 004af779
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004114: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s20                             // 000000004120: 8c7e147e
	v_cmp_lt_i64_e64 s18, 3, v[46:47]                          // 000000004124: d4510012 02025c83
	s_and_b32 s20, s18, vcc_lo                                 // 00000000412c: 8b146a12
	s_delay_alu instid0(salu_cycle_1)                          // 000000004130: bf870009
	s_and_saveexec_b32 s21, s20                                // 000000004134: be952014
	s_cbranch_execz 28                                         // 000000004138: bfa5001c <packed_folded_w4a8+0x26ac>
	v_bfe_u32 v112, v133, 16, 1                                // 00000000413c: d6100070 02052185
	s_wait_kmcnt 0x0                                           // 000000004144: bfc70000
	v_add_co_u32 v113, s20, s34, v40                           // 000000004148: d7001471 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004150: bf870191
	v_add_co_ci_u32_e64 v120, null, s35, v41, s20              // 000000004154: d5207c78 00525223
	v_add3_u32 v121, v112, v133, 0x7fff                        // 00000000415c: d6550079 03ff0b70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004168: bf870003
	v_add_co_u32 v112, s20, v113, v60                          // 00000000416c: d7001470 02027971
	v_or_b32_e32 v122, 0x400000, v133                          // 000000004174: 38f50aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000417c: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v61, s20             // 000000004180: d5207c71 00527b78
	v_cmp_u_f32_e64 s20, v133, v133                            // 000000004188: d4180014 02030b85
	s_wait_alu depctr_va_sdst(0)                               // 000000004190: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004194: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s20                    // 000000004198: d5010078 0052f579
	global_store_d16_hi_b16 v[112:113], v120, off              // 0000000041a0: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s21                             // 0000000041ac: 8c7e157e
	v_cmp_lt_i64_e64 s20, 4, v[46:47]                          // 0000000041b0: d4510014 02025c84
	s_and_b32 s21, s20, vcc_lo                                 // 0000000041b8: 8b156a14
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041bc: bf88ff9e
	s_and_saveexec_b32 s22, s21                                // 0000000041c0: be962015
	s_cbranch_execz 28                                         // 0000000041c4: bfa5001c <packed_folded_w4a8+0x2738>
	v_bfe_u32 v112, v134, 16, 1                                // 0000000041c8: d6100070 02052186
	s_wait_kmcnt 0x0                                           // 0000000041d0: bfc70000
	v_add_co_u32 v113, s21, s34, v40                           // 0000000041d4: d7001571 02025022
	s_wait_alu depctr_va_sdst(0)                               // 0000000041dc: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s21              // 0000000041e0: d5207c78 00565223
	v_add3_u32 v121, v112, v134, 0x7fff                        // 0000000041e8: d6550079 03ff0d70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000041f4: bf870003
	v_add_co_u32 v112, s21, v113, v62                          // 0000000041f8: d7001570 02027d71
	v_or_b32_e32 v122, 0x400000, v134                          // 000000004200: 38f50cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004208: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v63, s21             // 00000000420c: d5207c71 00567f78
	v_cmp_u_f32_e64 s21, v134, v134                            // 000000004214: d4180015 02030d86
	s_wait_alu depctr_va_sdst(0)                               // 00000000421c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004220: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s21                    // 000000004224: d5010078 0056f579
	global_store_d16_hi_b16 v[112:113], v120, off              // 00000000422c: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s22                             // 000000004238: 8c7e167e
	v_cmp_lt_i64_e64 s21, 5, v[46:47]                          // 00000000423c: d4510015 02025c85
	s_and_b32 s22, s21, vcc_lo                                 // 000000004244: 8b166a15
	s_delay_alu instid0(salu_cycle_1)                          // 000000004248: bf870009
	s_and_saveexec_b32 s23, s22                                // 00000000424c: be972016
	s_cbranch_execz 28                                         // 000000004250: bfa5001c <packed_folded_w4a8+0x27c4>
	v_bfe_u32 v112, v135, 16, 1                                // 000000004254: d6100070 02052187
	s_wait_kmcnt 0x0                                           // 00000000425c: bfc70000
	v_add_co_u32 v113, s22, s34, v40                           // 000000004260: d7001671 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004268: bf870191
	v_add_co_ci_u32_e64 v120, null, s35, v41, s22              // 00000000426c: d5207c78 005a5223
	v_add3_u32 v121, v112, v135, 0x7fff                        // 000000004274: d6550079 03ff0f70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004280: bf870003
	v_add_co_u32 v112, s22, v113, v64                          // 000000004284: d7001670 02028171
	v_or_b32_e32 v122, 0x400000, v135                          // 00000000428c: 38f50eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004294: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v65, s22             // 000000004298: d5207c71 005a8378
	v_cmp_u_f32_e64 s22, v135, v135                            // 0000000042a0: d4180016 02030f87
	s_wait_alu depctr_va_sdst(0)                               // 0000000042a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042ac: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s22                    // 0000000042b0: d5010078 005af579
	global_store_d16_hi_b16 v[112:113], v120, off              // 0000000042b8: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s23                             // 0000000042c4: 8c7e177e
	v_cmp_lt_i64_e64 s22, 6, v[46:47]                          // 0000000042c8: d4510016 02025c86
	s_and_b32 s23, s22, vcc_lo                                 // 0000000042d0: 8b176a16
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042d4: bf88ff9e
	s_and_saveexec_b32 s24, s23                                // 0000000042d8: be982017
	s_cbranch_execz 28                                         // 0000000042dc: bfa5001c <packed_folded_w4a8+0x2850>
	v_bfe_u32 v112, v136, 16, 1                                // 0000000042e0: d6100070 02052188
	s_wait_kmcnt 0x0                                           // 0000000042e8: bfc70000
	v_add_co_u32 v113, s23, s34, v40                           // 0000000042ec: d7001771 02025022
	s_wait_alu depctr_va_sdst(0)                               // 0000000042f4: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s23              // 0000000042f8: d5207c78 005e5223
	v_add3_u32 v121, v112, v136, 0x7fff                        // 000000004300: d6550079 03ff1170 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000430c: bf870003
	v_add_co_u32 v112, s23, v113, v66                          // 000000004310: d7001770 02028571
	v_or_b32_e32 v122, 0x400000, v136                          // 000000004318: 38f510ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004320: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v67, s23             // 000000004324: d5207c71 005e8778
	v_cmp_u_f32_e64 s23, v136, v136                            // 00000000432c: d4180017 02031188
	s_wait_alu depctr_va_sdst(0)                               // 000000004334: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004338: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s23                    // 00000000433c: d5010078 005ef579
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004344: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s24                             // 000000004350: 8c7e187e
	v_cmp_lt_i64_e64 s23, 7, v[46:47]                          // 000000004354: d4510017 02025c87
	s_and_b32 s24, s23, vcc_lo                                 // 00000000435c: 8b186a17
	s_delay_alu instid0(salu_cycle_1)                          // 000000004360: bf870009
	s_and_saveexec_b32 s25, s24                                // 000000004364: be992018
	s_cbranch_execz 28                                         // 000000004368: bfa5001c <packed_folded_w4a8+0x28dc>
	v_bfe_u32 v46, v137, 16, 1                                 // 00000000436c: d610002e 02052189
	s_wait_kmcnt 0x0                                           // 000000004374: bfc70000
	v_add_co_u32 v47, s24, s34, v40                            // 000000004378: d700182f 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004380: bf870191
	v_add_co_ci_u32_e64 v112, null, s35, v41, s24              // 000000004384: d5207c70 00625223
	v_add3_u32 v113, v46, v137, 0x7fff                         // 00000000438c: d6550071 03ff132e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004398: bf870003
	v_add_co_u32 v46, s24, v47, v68                            // 00000000439c: d700182e 0202892f
	v_or_b32_e32 v120, 0x400000, v137                          // 0000000043a4: 38f112ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000043ac: bf88f19f
	v_add_co_ci_u32_e64 v47, null, v112, v69, s24              // 0000000043b0: d5207c2f 00628b70
	v_cmp_u_f32_e64 s24, v137, v137                            // 0000000043b8: d4180018 02031389
	s_wait_alu depctr_va_sdst(0)                               // 0000000043c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043c4: bf870001
	v_cndmask_b32_e64 v112, v113, v120, s24                    // 0000000043c8: d5010070 0062f171
	global_store_d16_hi_b16 v[46:47], v112, off                // 0000000043d0: ee09407c 38000000 0000002e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000043dc: 8c7e197e
	v_or_b32_e32 v120, s26, v114                               // 0000000043e0: 38f0e41a
	v_mov_b32_e32 v123, s27                                    // 0000000043e4: 7ef6021b
	v_mov_b32_e32 v121, s27                                    // 0000000043e8: 7ef2021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000043ec: bf870093
	v_or_b32_e32 v122, v120, v115                              // 0000000043f0: 38f4e778
	v_cmp_gt_i64_e64 s24, s[36:37], v[122:123]                 // 0000000043f4: d4540018 0202f424
	s_wait_alu depctr_va_sdst(0)                               // 0000000043fc: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004400: bf8700a1
	v_cndmask_b32_e64 v47, 0, v123, s24                        // 000000004404: d501002f 0062f680
	v_cndmask_b32_e64 v46, 0, v122, s24                        // 00000000440c: d501002e 0062f480
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]                    // 000000004414: 3e5c5c82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000004418: bf870121
	v_add_co_u32 v46, s24, s28, v46                            // 00000000441c: d700182e 02025c1c
	s_wait_alu depctr_va_sdst(0)                               // 000000004424: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s29, v47, s24               // 000000004428: d5207c2f 00625e1d
	global_load_u8 v111, v[110:111], off                       // 000000004430: ee04007c 0000006f 0000006e
	global_load_b32 v110, v[46:47], off                        // 00000000443c: ee05007c 0000006e 0000002e
	s_wait_loadcnt 0x1                                         // 000000004448: bfc00001
	v_lshlrev_b32_e32 v136, 23, v111                           // 00000000444c: 3110de97
	s_wait_loadcnt 0x0                                         // 000000004450: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004454: bf870091
	v_mul_f32_e32 v111, v110, v136                             // 000000004458: 10df116e
	v_cmp_class_f32_e64 s24, v111, 0x198                       // 00000000445c: d47e0018 0201ff6f 00000198
	v_mul_f32_e32 v133, v32, v111                              // 000000004468: 110adf20
	s_xor_b32 s24, s24, -1                                     // 00000000446c: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004470: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004474: be992018
	s_cbranch_execnz 2615                                      // 000000004478: bfa60a37 <packed_folded_w4a8+0x5258>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000447c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004480: 8c7e197e
	v_or_b32_e32 v110, v116, v120                              // 000000004484: 38dcf174
	v_mov_b32_e32 v111, v121                                   // 000000004488: 7ede0379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 00000000448c: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[110:111]                 // 000000004490: d4540018 0202dc24
	s_wait_alu depctr_va_sdst(0)                               // 000000004498: bf88f19f
	v_cndmask_b32_e64 v111, 0, v111, s24                       // 00000000449c: d501006f 0062de80
	v_cndmask_b32_e64 v110, 0, v110, s24                       // 0000000044a4: d501006e 0062dc80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000044ac: bf870091
	v_lshlrev_b64_e32 v[110:111], 2, v[110:111]                // 0000000044b0: 3edcdc82
	v_add_co_u32 v110, s24, s28, v110                          // 0000000044b4: d700186e 0202dc1c
	s_wait_alu depctr_va_sdst(0)                               // 0000000044bc: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000044c0: bf8700c2
	v_add_co_ci_u32_e64 v111, null, s29, v111, s24             // 0000000044c4: d5207c6f 0062de1d
	global_load_b32 v32, v[110:111], off                       // 0000000044cc: ee05007c 00000020 0000006e
	s_wait_loadcnt 0x0                                         // 0000000044d8: bfc00000
	v_mul_f32_e32 v112, v32, v136                              // 0000000044dc: 10e11120
	v_cmp_class_f32_e64 s24, v112, 0x198                       // 0000000044e0: d47e0018 0201ff70 00000198
	v_mul_f32_e32 v134, v33, v112                              // 0000000044ec: 110ce121
	s_xor_b32 s24, s24, -1                                     // 0000000044f0: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044f4: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 0000000044f8: be992018
	s_cbranch_execnz 2600                                      // 0000000044fc: bfa60a28 <packed_folded_w4a8+0x52a0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004500: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004504: 8c7e197e
	v_or_b32_e32 v32, v117, v120                               // 000000004508: 3840f175
	v_mov_b32_e32 v33, v121                                    // 00000000450c: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004510: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 000000004514: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 00000000451c: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004520: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004528: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004530: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004534: 3e404082
	v_add_co_u32 v112, s24, s28, v32                           // 000000004538: d7001870 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004540: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000004544: bf8700c2
	v_add_co_ci_u32_e64 v113, null, s29, v33, s24              // 000000004548: d5207c71 0062421d
	global_load_b32 v32, v[112:113], off                       // 000000004550: ee05007c 00000020 00000070
	s_wait_loadcnt 0x0                                         // 00000000455c: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004560: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 000000004564: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v135, v34, v33                               // 000000004570: 110e4322
	s_xor_b32 s24, s24, -1                                     // 000000004574: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004578: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000457c: be992018
	s_cbranch_execnz 2585                                      // 000000004580: bfa60a19 <packed_folded_w4a8+0x52e8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004588: 8c7e197e
	v_or_b32_e32 v32, v128, v120                               // 00000000458c: 3840f180
	v_mov_b32_e32 v33, v121                                    // 000000004590: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004594: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 000000004598: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 0000000045a0: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 0000000045a4: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 0000000045ac: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000045b4: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 0000000045b8: 3e404082
	v_add_co_u32 v114, s24, s28, v32                           // 0000000045bc: d7001872 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 0000000045c4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000045c8: bf8700c2
	v_add_co_ci_u32_e64 v115, null, s29, v33, s24              // 0000000045cc: d5207c73 0062421d
	global_load_b32 v32, v[114:115], off                       // 0000000045d4: ee05007c 00000020 00000072
	s_wait_loadcnt 0x0                                         // 0000000045e0: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000045e4: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000045e8: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v128, v35, v33                               // 0000000045f4: 11004323
	s_xor_b32 s24, s24, -1                                     // 0000000045f8: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045fc: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004600: be992018
	s_cbranch_execnz 2570                                      // 000000004604: bfa60a0a <packed_folded_w4a8+0x5330>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004608: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 00000000460c: 8c7e197e
	v_or_b32_e32 v32, v129, v120                               // 000000004610: 3840f181
	v_mov_b32_e32 v33, v121                                    // 000000004614: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004618: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 00000000461c: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 000000004624: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004628: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004630: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004638: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 00000000463c: 3e404082
	v_add_co_u32 v34, s24, s28, v32                            // 000000004640: d7001822 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000464c: bf8700c2
	v_add_co_ci_u32_e64 v35, null, s29, v33, s24               // 000000004650: d5207c23 0062421d
	global_load_b32 v32, v[34:35], off                         // 000000004658: ee05007c 00000020 00000022
	s_wait_loadcnt 0x0                                         // 000000004664: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004668: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 00000000466c: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v129, v36, v33                               // 000000004678: 11024324
	s_xor_b32 s24, s24, -1                                     // 00000000467c: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004680: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004684: be992018
	s_cbranch_execnz 2555                                      // 000000004688: bfa609fb <packed_folded_w4a8+0x5378>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000468c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004690: 8c7e197e
	v_or_b32_e32 v32, v130, v120                               // 000000004694: 3840f182
	v_mov_b32_e32 v33, v121                                    // 000000004698: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 00000000469c: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 0000000046a0: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a8: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 0000000046ac: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 0000000046b4: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000046bc: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 0000000046c0: 3e404082
	v_add_co_u32 v116, s24, s28, v32                           // 0000000046c4: d7001874 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 0000000046cc: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000046d0: bf8700c2
	v_add_co_ci_u32_e64 v117, null, s29, v33, s24              // 0000000046d4: d5207c75 0062421d
	global_load_b32 v32, v[116:117], off                       // 0000000046dc: ee05007c 00000020 00000074
	s_wait_loadcnt 0x0                                         // 0000000046e8: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000046ec: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000046f0: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v130, v37, v33                               // 0000000046fc: 11044325
	s_xor_b32 s24, s24, -1                                     // 000000004700: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004704: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004708: be992018
	s_cbranch_execnz 2540                                      // 00000000470c: bfa609ec <packed_folded_w4a8+0x53c0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004710: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004714: 8c7e197e
	v_or_b32_e32 v32, v131, v120                               // 000000004718: 3840f183
	v_mov_b32_e32 v33, v121                                    // 00000000471c: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004720: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 000000004724: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 00000000472c: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004730: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004738: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004740: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004744: 3e404082
	v_add_co_u32 v36, s24, s28, v32                            // 000000004748: d7001824 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004750: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000004754: bf8700c2
	v_add_co_ci_u32_e64 v37, null, s29, v33, s24               // 000000004758: d5207c25 0062421d
	global_load_b32 v32, v[36:37], off                         // 000000004760: ee05007c 00000020 00000024
	s_wait_loadcnt 0x0                                         // 00000000476c: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004770: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 000000004774: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v131, v38, v33                               // 000000004780: 11064326
	s_xor_b32 s24, s24, -1                                     // 000000004784: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004788: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000478c: be992018
	s_cbranch_execnz 2525                                      // 000000004790: bfa609dd <packed_folded_w4a8+0x5408>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004794: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004798: 8c7e197e
	v_or_b32_e32 v120, v132, v120                              // 00000000479c: 38f0f184
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000047a0: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[120:121]                 // 0000000047a4: d4540018 0202f024
	s_wait_alu depctr_va_sdst(0)                               // 0000000047ac: bf88f19f
	v_cndmask_b32_e64 v33, 0, v121, s24                        // 0000000047b0: d5010021 0062f280
	v_cndmask_b32_e64 v32, 0, v120, s24                        // 0000000047b8: d5010020 0062f080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000047c0: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 0000000047c4: 3e404082
	v_add_co_u32 v120, s24, s28, v32                           // 0000000047c8: d7001878 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000047d4: bf8700c2
	v_add_co_ci_u32_e64 v121, null, s29, v33, s24              // 0000000047d8: d5207c79 0062421d
	global_load_b32 v32, v[120:121], off                       // 0000000047e0: ee05007c 00000020 00000078
	s_wait_loadcnt 0x0                                         // 0000000047ec: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000047f0: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000047f4: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v132, v39, v33                               // 000000004800: 11084327
	s_xor_b32 s24, s24, -1                                     // 000000004804: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004808: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000480c: be992018
	s_cbranch_execnz 2511                                      // 000000004810: bfa609cf <packed_folded_w4a8+0x5450>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004814: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004818: 8c7e197e
	v_mul_lo_u32 v136, s39, v122                               // 00000000481c: d72c0088 0202f427
	v_mul_lo_u32 v137, s38, v123                               // 000000004824: d72c0089 0202f626
	v_mad_co_u64_u32 v[32:33], null, s38, v122, 0              // 00000000482c: d6fe7c20 0202f426
	v_sub_co_u32 v38, s24, s36, v122                           // 000000004834: d7011826 0202f424
	s_wait_alu depctr_va_sdst(0)                               // 00000000483c: bf88f19f
	v_sub_co_ci_u32_e64 v39, null, s37, v123, s24              // 000000004840: d5217c27 0062f625
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000004848: bf870211
	v_cmp_lt_i64_e64 s27, 0, v[38:39]                          // 00000000484c: d451001b 02024c80
	v_add3_u32 v33, v33, v137, v136                            // 000000004854: d6550021 06231321
	s_delay_alu instid0(valu_dep_1)                            // 00000000485c: bf870001
	v_lshlrev_b64_e32 v[32:33], 1, v[32:33]                    // 000000004860: 3e404081
	s_and_b32 s24, s27, vcc_lo                                 // 000000004864: 8b186a1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004868: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000486c: be992018
	s_cbranch_execz 28                                         // 000000004870: bfa5001c <packed_folded_w4a8+0x2de4>
	s_wait_kmcnt 0x0                                           // 000000004874: bfc70000
	v_add_co_u32 v123, s24, s34, v32                           // 000000004878: d700187b 02024022
	v_bfe_u32 v122, v133, 16, 1                                // 000000004880: d610007a 02052185
	s_wait_alu depctr_va_sdst(0)                               // 000000004888: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v33, s24              // 00000000488c: d5207c88 00624223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004894: bf870193
	v_add_co_u32 v118, s24, v123, v118                         // 000000004898: d7001876 0202ed7b
	v_add3_u32 v122, v122, v133, 0x7fff                        // 0000000048a0: d655007a 03ff0b7a 00007fff
	v_or_b32_e32 v137, 0x400000, v133                          // 0000000048ac: 39130aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000048b4: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v136, v119, s24            // 0000000048b8: d5207c77 0062ef88
	v_cmp_u_f32_e64 s24, v133, v133                            // 0000000048c0: d4180018 02030b85
	s_wait_alu depctr_va_sdst(0)                               // 0000000048c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048cc: bf870001
	v_cndmask_b32_e64 v122, v122, v137, s24                    // 0000000048d0: d501007a 0063137a
	global_store_d16_hi_b16 v[118:119], v122, off              // 0000000048d8: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000048e8: 8c7e197e
	v_cmp_lt_i64_e64 s24, 1, v[38:39]                          // 0000000048ec: d4510018 02024c81
	s_and_b32 s25, s24, vcc_lo                                 // 0000000048f4: 8b196a18
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048f8: bf88ff9e
	s_and_saveexec_b32 s26, s25                                // 0000000048fc: be9a2019
	s_cbranch_execz 28                                         // 000000004900: bfa5001c <packed_folded_w4a8+0x2e74>
	v_bfe_u32 v118, v134, 16, 1                                // 000000004904: d6100076 02052186
	s_wait_kmcnt 0x0                                           // 00000000490c: bfc70000
	v_add_co_u32 v119, s25, s34, v32                           // 000000004910: d7001977 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004918: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s25              // 00000000491c: d5207c7a 00664223
	v_add3_u32 v123, v118, v134, 0x7fff                        // 000000004924: d655007b 03ff0d76 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004930: bf870003
	v_add_co_u32 v118, s25, v119, v56                          // 000000004934: d7001976 02027177
	v_or_b32_e32 v133, 0x400000, v134                          // 00000000493c: 390b0cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004944: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v57, s25             // 000000004948: d5207c77 0066737a
	v_cmp_u_f32_e64 s25, v134, v134                            // 000000004950: d4180019 02030d86
	s_wait_alu depctr_va_sdst(0)                               // 000000004958: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000495c: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s25                    // 000000004960: d501007a 00670b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004968: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 000000004974: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s26                             // 000000004978: 8c7e1a7e
	v_cmp_lt_i64_e64 s25, 2, v[38:39]                          // 00000000497c: d4510019 02024c82
	s_and_b32 s26, s25, vcc_lo                                 // 000000004984: 8b1a6a19
	s_wait_alu depctr_sa_sdst(0)                               // 000000004988: bf88ff9e
	s_and_saveexec_b32 s28, s26                                // 00000000498c: be9c201a
	s_cbranch_execz 28                                         // 000000004990: bfa5001c <packed_folded_w4a8+0x2f04>
	v_bfe_u32 v118, v135, 16, 1                                // 000000004994: d6100076 02052187
	s_wait_kmcnt 0x0                                           // 00000000499c: bfc70000
	v_add_co_u32 v119, s26, s34, v32                           // 0000000049a0: d7001a77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000049a8: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s26              // 0000000049ac: d5207c7a 006a4223
	v_add3_u32 v123, v118, v135, 0x7fff                        // 0000000049b4: d655007b 03ff0f76 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000049c0: bf870003
	v_add_co_u32 v118, s26, v119, v58                          // 0000000049c4: d7001a76 02027577
	v_or_b32_e32 v133, 0x400000, v135                          // 0000000049cc: 390b0eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000049d4: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v59, s26             // 0000000049d8: d5207c77 006a777a
	v_cmp_u_f32_e64 s26, v135, v135                            // 0000000049e0: d418001a 02030f87
	s_wait_alu depctr_va_sdst(0)                               // 0000000049e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000049ec: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s26                    // 0000000049f0: d501007a 006b0b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 0000000049f8: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004a08: 8c7e1c7e
	v_cmp_lt_i64_e64 s26, 3, v[38:39]                          // 000000004a0c: d451001a 02024c83
	s_and_b32 s28, s26, vcc_lo                                 // 000000004a14: 8b1c6a1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a18: bf88ff9e
	s_and_saveexec_b32 s29, s28                                // 000000004a1c: be9d201c
	s_cbranch_execz 28                                         // 000000004a20: bfa5001c <packed_folded_w4a8+0x2f94>
	v_bfe_u32 v118, v128, 16, 1                                // 000000004a24: d6100076 02052180
	s_wait_kmcnt 0x0                                           // 000000004a2c: bfc70000
	v_add_co_u32 v119, s28, s34, v32                           // 000000004a30: d7001c77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004a38: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s28              // 000000004a3c: d5207c7a 00724223
	v_add3_u32 v123, v118, v128, 0x7fff                        // 000000004a44: d655007b 03ff0176 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004a50: bf870003
	v_add_co_u32 v118, s28, v119, v60                          // 000000004a54: d7001c76 02027977
	v_or_b32_e32 v133, 0x400000, v128                          // 000000004a5c: 390b00ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004a64: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v61, s28             // 000000004a68: d5207c77 00727b7a
	v_cmp_u_f32_e64 s28, v128, v128                            // 000000004a70: d418001c 02030180
	s_wait_alu depctr_va_sdst(0)                               // 000000004a78: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a7c: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s28                    // 000000004a80: d501007a 00730b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004a88: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s29                             // 000000004a98: 8c7e1d7e
	v_cmp_lt_i64_e64 s28, 4, v[38:39]                          // 000000004a9c: d451001c 02024c84
	s_and_b32 s29, s28, vcc_lo                                 // 000000004aa4: 8b1d6a1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004aa8: bf88ff9e
	s_and_saveexec_b32 s30, s29                                // 000000004aac: be9e201d
	s_cbranch_execz 28                                         // 000000004ab0: bfa5001c <packed_folded_w4a8+0x3024>
	v_bfe_u32 v118, v129, 16, 1                                // 000000004ab4: d6100076 02052181
	s_wait_kmcnt 0x0                                           // 000000004abc: bfc70000
	v_add_co_u32 v119, s29, s34, v32                           // 000000004ac0: d7001d77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004ac8: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s29              // 000000004acc: d5207c7a 00764223
	v_add3_u32 v123, v118, v129, 0x7fff                        // 000000004ad4: d655007b 03ff0376 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004ae0: bf870003
	v_add_co_u32 v118, s29, v119, v62                          // 000000004ae4: d7001d76 02027d77
	v_or_b32_e32 v128, 0x400000, v129                          // 000000004aec: 390102ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004af4: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v63, s29             // 000000004af8: d5207c77 00767f7a
	v_cmp_u_f32_e64 s29, v129, v129                            // 000000004b00: d418001d 02030381
	s_wait_alu depctr_va_sdst(0)                               // 000000004b08: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b0c: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s29                    // 000000004b10: d501007a 0077017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004b18: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s30                             // 000000004b24: 8c7e1e7e
	v_cmp_lt_i64_e64 s29, 5, v[38:39]                          // 000000004b28: d451001d 02024c85
	s_and_b32 s30, s29, vcc_lo                                 // 000000004b30: 8b1e6a1d
	s_delay_alu instid0(salu_cycle_1)                          // 000000004b34: bf870009
	s_and_saveexec_b32 s31, s30                                // 000000004b38: be9f201e
	s_cbranch_execz 28                                         // 000000004b3c: bfa5001c <packed_folded_w4a8+0x30b0>
	v_bfe_u32 v118, v130, 16, 1                                // 000000004b40: d6100076 02052182
	s_wait_kmcnt 0x0                                           // 000000004b48: bfc70000
	v_add_co_u32 v119, s30, s34, v32                           // 000000004b4c: d7001e77 02024022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004b54: bf870191
	v_add_co_ci_u32_e64 v122, null, s35, v33, s30              // 000000004b58: d5207c7a 007a4223
	v_add3_u32 v123, v118, v130, 0x7fff                        // 000000004b60: d655007b 03ff0576 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004b6c: bf870003
	v_add_co_u32 v118, s30, v119, v64                          // 000000004b70: d7001e76 02028177
	v_or_b32_e32 v128, 0x400000, v130                          // 000000004b78: 390104ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b80: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v65, s30             // 000000004b84: d5207c77 007a837a
	v_cmp_u_f32_e64 s30, v130, v130                            // 000000004b8c: d418001e 02030582
	s_wait_alu depctr_va_sdst(0)                               // 000000004b94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b98: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s30                    // 000000004b9c: d501007a 007b017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004ba4: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004bb0: 8c7e1f7e
	v_cmp_lt_i64_e64 s30, 6, v[38:39]                          // 000000004bb4: d451001e 02024c86
	s_and_b32 s31, s30, vcc_lo                                 // 000000004bbc: 8b1f6a1e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bc0: bf88ff9e
	s_and_saveexec_b32 s33, s31                                // 000000004bc4: bea1201f
	s_cbranch_execz 28                                         // 000000004bc8: bfa5001c <packed_folded_w4a8+0x313c>
	v_bfe_u32 v118, v131, 16, 1                                // 000000004bcc: d6100076 02052183
	s_wait_kmcnt 0x0                                           // 000000004bd4: bfc70000
	v_add_co_u32 v119, s31, s34, v32                           // 000000004bd8: d7001f77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004be0: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s31              // 000000004be4: d5207c7a 007e4223
	v_add3_u32 v123, v118, v131, 0x7fff                        // 000000004bec: d655007b 03ff0776 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004bf8: bf870003
	v_add_co_u32 v118, s31, v119, v66                          // 000000004bfc: d7001f76 02028577
	v_or_b32_e32 v128, 0x400000, v131                          // 000000004c04: 390106ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004c0c: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v67, s31             // 000000004c10: d5207c77 007e877a
	v_cmp_u_f32_e64 s31, v131, v131                            // 000000004c18: d418001f 02030783
	s_wait_alu depctr_va_sdst(0)                               // 000000004c20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c24: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s31                    // 000000004c28: d501007a 007f017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004c30: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004c3c: 8c7e217e
	v_cmp_lt_i64_e64 s31, 7, v[38:39]                          // 000000004c40: d451001f 02024c87
	s_and_b32 s36, s31, vcc_lo                                 // 000000004c48: 8b246a1f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c4c: bf88ff9e
	s_and_saveexec_b32 s33, s36                                // 000000004c50: bea12024
	s_cbranch_execz 25                                         // 000000004c54: bfa50019 <packed_folded_w4a8+0x31bc>
	v_bfe_u32 v38, v132, 16, 1                                 // 000000004c58: d6100026 02052184
	s_wait_kmcnt 0x0                                           // 000000004c60: bfc70000
	v_add_co_u32 v39, vcc_lo, s34, v32                         // 000000004c64: d7006a27 02024022
	s_wait_alu depctr_va_vcc(0)                                // 000000004c6c: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, s35, v33, vcc_lo           // 000000004c70: d5207c76 01aa4223
	v_add3_u32 v119, v38, v132, 0x7fff                         // 000000004c78: d6550077 03ff0926 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004c84: bf870003
	v_add_co_u32 v38, vcc_lo, v39, v68                         // 000000004c88: d7006a26 02028927
	v_or_b32_e32 v122, 0x400000, v132                          // 000000004c90: 38f508ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c98: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, v118, v69, vcc_lo           // 000000004c9c: d5207c27 01aa8b76
	v_cmp_u_f32_e32 vcc_lo, v132, v132                         // 000000004ca4: 7c310984
	s_wait_alu depctr_va_vcc(0)                                // 000000004ca8: bf88ff9d
	v_cndmask_b32_e32 v118, v119, v122, vcc_lo                 // 000000004cac: 02ecf577
	global_store_d16_hi_b16 v[38:39], v118, off                // 000000004cb0: ee09407c 3b000000 00000026
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004cbc: 8c7e217e
	v_or_b32_e32 v38, v125, v127                               // 000000004cc0: 384cff7d
	v_mov_b32_e32 v39, v126                                    // 000000004cc4: 7e4e037e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000004cc8: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[38:39]                // 000000004ccc: 7ca84c26
	s_wait_alu depctr_va_vcc(0)                                // 000000004cd0: bf88ff9d
	v_dual_cndmask_b32 v38, 0, v38 :: v_dual_cndmask_b32 v39, 0, v39// 000000004cd4: ca524c80 26264e80
	v_add_co_u32 v38, s33, s40, v38                            // 000000004cdc: d7002126 02024c28
	s_delay_alu instid0(valu_dep_1)                            // 000000004ce4: bf870001
	v_add_co_ci_u32_e64 v39, null, s41, v39, s33               // 000000004ce8: d5207c27 00864e29
	global_load_u8 v119, v[38:39], off                         // 000000004cf0: ee04007c 00000077 00000026
	global_load_b32 v118, v[72:73], off                        // 000000004cfc: ee05007c 00000076 00000048
	s_wait_loadcnt 0x1                                         // 000000004d08: bfc00001
	v_lshlrev_b32_e32 v73, 23, v119                            // 000000004d0c: 3092ee97
	s_wait_loadcnt 0x0                                         // 000000004d10: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004d14: bf870091
	v_mul_f32_e32 v72, v118, v73                               // 000000004d18: 10909376
	v_cmp_class_f32_e64 s33, v72, 0x198                        // 000000004d1c: d47e0021 0201ff48 00000198
	v_mul_f32_e32 v72, v24, v72                                // 000000004d28: 10909118
	s_xor_b32 s33, s33, -1                                     // 000000004d2c: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d30: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004d34: bea42021
	s_cbranch_execnz 2199                                      // 000000004d38: bfa60897 <packed_folded_w4a8+0x5498>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004d40: 8c7e247e
	global_load_b32 v74, v[74:75], off                         // 000000004d44: ee05007c 0000004a 0000004a
	s_wait_loadcnt 0x0                                         // 000000004d50: bfc00000
	v_mul_f32_e32 v24, v74, v73                                // 000000004d54: 1030934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004d58: bf870001
	v_cmp_class_f32_e64 s33, v24, 0x198                        // 000000004d5c: d47e0021 0201ff18 00000198
	v_mul_f32_e32 v24, v25, v24                                // 000000004d68: 10303119
	s_xor_b32 s33, s33, -1                                     // 000000004d6c: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d70: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004d74: bea42021
	s_cbranch_execnz 2201                                      // 000000004d78: bfa60899 <packed_folded_w4a8+0x54e0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004d80: 8c7e247e
	global_load_b32 v74, v[76:77], off                         // 000000004d84: ee05007c 0000004a 0000004c
	s_wait_loadcnt 0x0                                         // 000000004d90: bfc00000
	v_mul_f32_e32 v25, v74, v73                                // 000000004d94: 1032934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004d98: bf870001
	v_cmp_class_f32_e64 s33, v25, 0x198                        // 000000004d9c: d47e0021 0201ff19 00000198
	v_mul_f32_e32 v25, v26, v25                                // 000000004da8: 1032331a
	s_xor_b32 s33, s33, -1                                     // 000000004dac: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004db0: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004db4: bea42021
	s_cbranch_execnz 2203                                      // 000000004db8: bfa6089b <packed_folded_w4a8+0x5528>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004dbc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004dc0: 8c7e247e
	global_load_b32 v74, v[78:79], off                         // 000000004dc4: ee05007c 0000004a 0000004e
	s_wait_loadcnt 0x0                                         // 000000004dd0: bfc00000
	v_mul_f32_e32 v26, v74, v73                                // 000000004dd4: 1034934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004dd8: bf870001
	v_cmp_class_f32_e64 s33, v26, 0x198                        // 000000004ddc: d47e0021 0201ff1a 00000198
	v_mul_f32_e32 v26, v27, v26                                // 000000004de8: 1034351b
	s_xor_b32 s33, s33, -1                                     // 000000004dec: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004df0: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004df4: bea42021
	s_cbranch_execnz 2205                                      // 000000004df8: bfa6089d <packed_folded_w4a8+0x5570>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004dfc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e00: 8c7e247e
	global_load_b32 v74, v[80:81], off                         // 000000004e04: ee05007c 0000004a 00000050
	s_wait_loadcnt 0x0                                         // 000000004e10: bfc00000
	v_mul_f32_e32 v27, v74, v73                                // 000000004e14: 1036934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004e18: bf870001
	v_cmp_class_f32_e64 s33, v27, 0x198                        // 000000004e1c: d47e0021 0201ff1b 00000198
	v_mul_f32_e32 v27, v28, v27                                // 000000004e28: 1036371c
	s_xor_b32 s33, s33, -1                                     // 000000004e2c: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e30: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004e34: bea42021
	s_cbranch_execnz 2207                                      // 000000004e38: bfa6089f <packed_folded_w4a8+0x55b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e40: 8c7e247e
	global_load_b32 v74, v[82:83], off                         // 000000004e44: ee05007c 0000004a 00000052
	s_wait_loadcnt 0x0                                         // 000000004e50: bfc00000
	v_mul_f32_e32 v28, v74, v73                                // 000000004e54: 1038934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004e58: bf870001
	v_cmp_class_f32_e64 s33, v28, 0x198                        // 000000004e5c: d47e0021 0201ff1c 00000198
	v_mul_f32_e32 v28, v29, v28                                // 000000004e68: 1038391d
	s_xor_b32 s33, s33, -1                                     // 000000004e6c: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e70: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004e74: bea42021
	s_cbranch_execnz 2209                                      // 000000004e78: bfa608a1 <packed_folded_w4a8+0x5600>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e80: 8c7e247e
	global_load_b32 v74, v[84:85], off                         // 000000004e84: ee05007c 0000004a 00000054
	s_wait_loadcnt 0x0                                         // 000000004e90: bfc00000
	v_mul_f32_e32 v29, v74, v73                                // 000000004e94: 103a934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004e98: bf870001
	v_cmp_class_f32_e64 s33, v29, 0x198                        // 000000004e9c: d47e0021 0201ff1d 00000198
	v_mul_f32_e32 v29, v30, v29                                // 000000004ea8: 103a3b1e
	s_xor_b32 s33, s33, -1                                     // 000000004eac: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004eb0: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004eb4: bea42021
	s_cbranch_execnz 2211                                      // 000000004eb8: bfa608a3 <packed_folded_w4a8+0x5648>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ebc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004ec0: 8c7e247e
	global_load_b32 v74, v[86:87], off                         // 000000004ec4: ee05007c 0000004a 00000056
	s_wait_loadcnt 0x0                                         // 000000004ed0: bfc00000
	v_mul_f32_e32 v30, v74, v73                                // 000000004ed4: 103c934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004ed8: bf870001
	v_cmp_class_f32_e64 s33, v30, 0x198                        // 000000004edc: d47e0021 0201ff1e 00000198
	v_mul_f32_e32 v30, v31, v30                                // 000000004ee8: 103c3d1f
	s_xor_b32 s33, s33, -1                                     // 000000004eec: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ef0: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004ef4: bea42021
	s_cbranch_execnz 2213                                      // 000000004ef8: bfa608a5 <packed_folded_w4a8+0x5690>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004efc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004f00: 8c7e247e
	s_and_b32 s0, s0, vcc_lo                                   // 000000004f04: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f08: bf88ff9e
	s_and_saveexec_b32 s33, s0                                 // 000000004f0c: bea12000
	s_cbranch_execz 34                                         // 000000004f10: bfa50022 <packed_folded_w4a8+0x349c>
	v_add_co_u32 v73, s0, v125, v124                           // 000000004f14: d7000049 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000004f1c: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v126, s0                 // 000000004f20: d5207c4a 0002fc80
	s_wait_kmcnt 0x0                                           // 000000004f28: bfc70000
	v_add_co_u32 v75, s0, s34, v70                             // 000000004f2c: d700004b 02028c22
	v_bfe_u32 v31, v72, 16, 1                                  // 000000004f34: d610001f 02052148
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000004f3c: bf870253
	v_lshlrev_b64_e32 v[73:74], 1, v[73:74]                    // 000000004f40: 3e929281
	s_wait_alu depctr_va_sdst(0)                               // 000000004f44: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s35, v71, s0                // 000000004f48: d5207c4c 00028e23
	v_or_b32_e32 v77, 0x400000, v72                            // 000000004f50: 389a90ff 00400000
	v_add3_u32 v31, v31, v72, 0x7fff                           // 000000004f58: d655001f 03fe911f 00007fff
	v_add_co_u32 v73, s0, v75, v73                             // 000000004f64: d7000049 0202934b
	s_wait_alu depctr_va_sdst(0)                               // 000000004f6c: bf88f19f
	v_add_co_ci_u32_e64 v74, null, v76, v74, s0                // 000000004f70: d5207c4a 0002954c
	v_cmp_u_f32_e64 s0, v72, v72                               // 000000004f78: d4180000 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000004f80: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004f84: bf870001
	v_cndmask_b32_e64 v31, v31, v77, s0                        // 000000004f88: d501001f 00029b1f
	global_store_d16_hi_b16 v[73:74], v31, off offset:32       // 000000004f90: ee09407c 0f800000 00002049
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004fa0: 8c7e217e
	s_and_b32 s0, s1, vcc_lo                                   // 000000004fa4: 8b006a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fa8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004fac: be812000
	s_cbranch_execz 28                                         // 000000004fb0: bfa5001c <packed_folded_w4a8+0x3524>
	s_wait_kmcnt 0x0                                           // 000000004fb4: bfc70000
	v_add_co_u32 v72, s0, s34, v70                             // 000000004fb8: d7000048 02028c22
	v_bfe_u32 v31, v24, 16, 1                                  // 000000004fc0: d610001f 02052118
	s_wait_alu depctr_va_sdst(0)                               // 000000004fc8: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s35, v71, s0                // 000000004fcc: d5207c49 00028e23
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004fd4: bf870193
	v_add_co_u32 v72, s0, v72, v56                             // 000000004fd8: d7000048 02027148
	v_add3_u32 v31, v31, v24, 0x7fff                           // 000000004fe0: d655001f 03fe311f 00007fff
	v_or_b32_e32 v74, 0x400000, v24                            // 000000004fec: 389430ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004ff4: bf88f19f
	v_add_co_ci_u32_e64 v73, null, v73, v57, s0                // 000000004ff8: d5207c49 00027349
	v_cmp_u_f32_e64 s0, v24, v24                               // 000000005000: d4180000 02023118
	s_wait_alu depctr_va_sdst(0)                               // 000000005008: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000500c: bf870001
	v_cndmask_b32_e64 v24, v31, v74, s0                        // 000000005010: d5010018 0002951f
	global_store_d16_hi_b16 v[72:73], v24, off offset:32       // 000000005018: ee09407c 0c000000 00002048
	s_wait_alu depctr_sa_sdst(0)                               // 000000005024: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005028: 8c7e017e
	s_and_b32 s0, s2, vcc_lo                                   // 00000000502c: 8b006a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000005030: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005034: be812000
	s_cbranch_execz 28                                         // 000000005038: bfa5001c <packed_folded_w4a8+0x35ac>
	s_wait_kmcnt 0x0                                           // 00000000503c: bfc70000
	v_add_co_u32 v31, s0, s34, v70                             // 000000005040: d700001f 02028c22
	v_bfe_u32 v24, v25, 16, 1                                  // 000000005048: d6100018 02052119
	s_wait_alu depctr_va_sdst(0)                               // 000000005050: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s35, v71, s0                // 000000005054: d5207c49 00028e23
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000505c: bf870193
	v_add_co_u32 v72, s0, v31, v58                             // 000000005060: d7000048 0202751f
	v_add3_u32 v24, v24, v25, 0x7fff                           // 000000005068: d6550018 03fe3318 00007fff
	v_or_b32_e32 v74, 0x400000, v25                            // 000000005074: 389432ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000507c: bf88f19f
	v_add_co_ci_u32_e64 v73, null, v73, v59, s0                // 000000005080: d5207c49 00027749
	v_cmp_u_f32_e64 s0, v25, v25                               // 000000005088: d4180000 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000005090: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005094: bf870001
	v_cndmask_b32_e64 v24, v24, v74, s0                        // 000000005098: d5010018 00029518
	global_store_d16_hi_b16 v[72:73], v24, off offset:32       // 0000000050a0: ee09407c 0c000000 00002048
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000050b0: 8c7e017e
	s_and_b32 s0, s3, vcc_lo                                   // 0000000050b4: 8b006a03
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050b8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000050bc: be812000
	s_cbranch_execz 28                                         // 0000000050c0: bfa5001c <packed_folded_w4a8+0x3634>
	v_bfe_u32 v24, v26, 16, 1                                  // 0000000050c4: d6100018 0205211a
	s_wait_kmcnt 0x0                                           // 0000000050cc: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000050d0: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000050d8: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s35, v71, s0                // 0000000050dc: d5207c1f 00028e23
	v_add3_u32 v72, v24, v26, 0x7fff                           // 0000000050e4: d6550048 03fe3518 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000050f0: bf870003
	v_add_co_u32 v24, s0, v25, v60                             // 0000000050f4: d7000018 02027919
	v_or_b32_e32 v73, 0x400000, v26                            // 0000000050fc: 389234ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005104: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v31, v61, s0                // 000000005108: d5207c19 00027b1f
	v_cmp_u_f32_e64 s0, v26, v26                               // 000000005110: d4180000 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000005118: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000511c: bf870001
	v_cndmask_b32_e64 v26, v72, v73, s0                        // 000000005120: d501001a 00029348
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005128: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005134: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005138: 8c7e017e
	s_and_b32 s0, s4, vcc_lo                                   // 00000000513c: 8b006a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000005140: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005144: be812000
	s_cbranch_execz 28                                         // 000000005148: bfa5001c <packed_folded_w4a8+0x36bc>
	v_bfe_u32 v24, v27, 16, 1                                  // 00000000514c: d6100018 0205211b
	s_wait_kmcnt 0x0                                           // 000000005154: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 000000005158: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000005160: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 000000005164: d5207c1a 00028e23
	v_add3_u32 v31, v24, v27, 0x7fff                           // 00000000516c: d655001f 03fe3718 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005178: bf870003
	v_add_co_u32 v24, s0, v25, v62                             // 00000000517c: d7000018 02027d19
	v_or_b32_e32 v72, 0x400000, v27                            // 000000005184: 389036ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000518c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v63, s0                // 000000005190: d5207c19 00027f1a
	v_cmp_u_f32_e64 s0, v27, v27                               // 000000005198: d4180000 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 0000000051a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000051a4: bf870001
	v_cndmask_b32_e64 v26, v31, v72, s0                        // 0000000051a8: d501001a 0002911f
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 0000000051b0: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000051c0: 8c7e017e
	s_and_b32 s0, s5, vcc_lo                                   // 0000000051c4: 8b006a05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051c8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000051cc: be812000
	s_cbranch_execz 28                                         // 0000000051d0: bfa5001c <packed_folded_w4a8+0x3744>
	v_bfe_u32 v24, v28, 16, 1                                  // 0000000051d4: d6100018 0205211c
	s_wait_kmcnt 0x0                                           // 0000000051dc: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000051e0: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000051e8: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 0000000051ec: d5207c1a 00028e23
	v_add3_u32 v27, v24, v28, 0x7fff                           // 0000000051f4: d655001b 03fe3918 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005200: bf870003
	v_add_co_u32 v24, s0, v25, v64                             // 000000005204: d7000018 02028119
	v_or_b32_e32 v31, 0x400000, v28                            // 00000000520c: 383e38ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005214: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v65, s0                // 000000005218: d5207c19 0002831a
	v_cmp_u_f32_e64 s0, v28, v28                               // 000000005220: d4180000 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000005228: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000522c: bf870001
	v_cndmask_b32_e64 v26, v27, v31, s0                        // 000000005230: d501001a 00023f1b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005238: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005244: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005248: 8c7e017e
	s_and_b32 s0, s6, vcc_lo                                   // 00000000524c: 8b006a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005250: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005254: be812000
	s_cbranch_execz 28                                         // 000000005258: bfa5001c <packed_folded_w4a8+0x37cc>
	v_bfe_u32 v24, v29, 16, 1                                  // 00000000525c: d6100018 0205211d
	s_wait_kmcnt 0x0                                           // 000000005264: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 000000005268: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000005270: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 000000005274: d5207c1a 00028e23
	v_add3_u32 v27, v24, v29, 0x7fff                           // 00000000527c: d655001b 03fe3b18 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005288: bf870003
	v_add_co_u32 v24, s0, v25, v66                             // 00000000528c: d7000018 02028519
	v_or_b32_e32 v28, 0x400000, v29                            // 000000005294: 38383aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000529c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v67, s0                // 0000000052a0: d5207c19 0002871a
	v_cmp_u_f32_e64 s0, v29, v29                               // 0000000052a8: d4180000 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 0000000052b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000052b4: bf870001
	v_cndmask_b32_e64 v26, v27, v28, s0                        // 0000000052b8: d501001a 0002391b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 0000000052c0: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000052d0: 8c7e017e
	s_and_b32 s0, s7, vcc_lo                                   // 0000000052d4: 8b006a07
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052d8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000052dc: be812000
	s_cbranch_execz 28                                         // 0000000052e0: bfa5001c <packed_folded_w4a8+0x3854>
	v_bfe_u32 v24, v30, 16, 1                                  // 0000000052e4: d6100018 0205211e
	s_wait_kmcnt 0x0                                           // 0000000052ec: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000052f0: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000052f8: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 0000000052fc: d5207c1a 00028e23
	v_add3_u32 v27, v24, v30, 0x7fff                           // 000000005304: d655001b 03fe3d18 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005310: bf870003
	v_add_co_u32 v24, s0, v25, v68                             // 000000005314: d7000018 02028919
	v_or_b32_e32 v28, 0x400000, v30                            // 00000000531c: 38383cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005324: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v69, s0                // 000000005328: d5207c19 00028b1a
	v_cmp_u_f32_e64 s0, v30, v30                               // 000000005330: d4180000 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000005338: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000533c: bf870001
	v_cndmask_b32_e64 v26, v27, v28, s0                        // 000000005340: d501001a 0002391b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005348: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005354: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005358: 8c7e017e
	global_load_u8 v24, v[38:39], off                          // 00000000535c: ee04007c 00000018 00000026
	global_load_b32 v26, v[88:89], off                         // 000000005368: ee05007c 0000001a 00000058
	s_wait_loadcnt 0x1                                         // 000000005374: bfc00001
	v_lshlrev_b32_e32 v25, 23, v24                             // 000000005378: 30323097
	s_wait_loadcnt 0x0                                         // 00000000537c: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005380: bf870091
	v_mul_f32_e32 v24, v26, v25                                // 000000005384: 1030331a
	v_cmp_class_f32_e64 s0, v24, 0x198                         // 000000005388: d47e0000 0201ff18 00000198
	v_mul_f32_e32 v24, v16, v24                                // 000000005394: 10303110
	s_xor_b32 s0, s0, -1                                       // 000000005398: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000539c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000053a0: be812000
	s_cbranch_execnz 1932                                      // 0000000053a4: bfa6078c <packed_folded_w4a8+0x56d8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000053ac: 8c7e017e
	global_load_b32 v26, v[90:91], off                         // 0000000053b0: ee05007c 0000001a 0000005a
	s_wait_loadcnt 0x0                                         // 0000000053bc: bfc00000
	v_mul_f32_e32 v16, v26, v25                                // 0000000053c0: 1020331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000053c4: bf870001
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 0000000053c8: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v17, v16                                // 0000000053d4: 10202111
	s_xor_b32 s0, s0, -1                                       // 0000000053d8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053dc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000053e0: be812000
	s_cbranch_execnz 1934                                      // 0000000053e4: bfa6078e <packed_folded_w4a8+0x5720>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000053ec: 8c7e017e
	global_load_b32 v26, v[92:93], off                         // 0000000053f0: ee05007c 0000001a 0000005c
	s_wait_loadcnt 0x0                                         // 0000000053fc: bfc00000
	v_mul_f32_e32 v17, v26, v25                                // 000000005400: 1022331a
	s_delay_alu instid0(valu_dep_1)                            // 000000005404: bf870001
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 000000005408: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v18, v17                                // 000000005414: 10222312
	s_xor_b32 s0, s0, -1                                       // 000000005418: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000541c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005420: be812000
	s_cbranch_execnz 1936                                      // 000000005424: bfa60790 <packed_folded_w4a8+0x5768>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005428: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000542c: 8c7e017e
	global_load_b32 v26, v[94:95], off                         // 000000005430: ee05007c 0000001a 0000005e
	s_wait_loadcnt 0x0                                         // 00000000543c: bfc00000
	v_mul_f32_e32 v18, v26, v25                                // 000000005440: 1024331a
	s_delay_alu instid0(valu_dep_1)                            // 000000005444: bf870001
	v_cmp_class_f32_e64 s0, v18, 0x198                         // 000000005448: d47e0000 0201ff12 00000198
	v_mul_f32_e32 v18, v19, v18                                // 000000005454: 10242513
	s_xor_b32 s0, s0, -1                                       // 000000005458: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000545c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005460: be812000
	s_cbranch_execnz 1938                                      // 000000005464: bfa60792 <packed_folded_w4a8+0x57b0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005468: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000546c: 8c7e017e
	global_load_b32 v26, v[50:51], off                         // 000000005470: ee05007c 0000001a 00000032
	s_wait_loadcnt 0x0                                         // 00000000547c: bfc00000
	v_mul_f32_e32 v19, v26, v25                                // 000000005480: 1026331a
	s_delay_alu instid0(valu_dep_1)                            // 000000005484: bf870001
	v_cmp_class_f32_e64 s0, v19, 0x198                         // 000000005488: d47e0000 0201ff13 00000198
	v_mul_f32_e32 v19, v20, v19                                // 000000005494: 10262714
	s_xor_b32 s0, s0, -1                                       // 000000005498: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000549c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000054a0: be812000
	s_cbranch_execnz 1940                                      // 0000000054a4: bfa60794 <packed_folded_w4a8+0x57f8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000054ac: 8c7e017e
	global_load_b32 v26, v[96:97], off                         // 0000000054b0: ee05007c 0000001a 00000060
	s_wait_loadcnt 0x0                                         // 0000000054bc: bfc00000
	v_mul_f32_e32 v20, v26, v25                                // 0000000054c0: 1028331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000054c4: bf870001
	v_cmp_class_f32_e64 s0, v20, 0x198                         // 0000000054c8: d47e0000 0201ff14 00000198
	v_mul_f32_e32 v20, v21, v20                                // 0000000054d4: 10282915
	s_xor_b32 s0, s0, -1                                       // 0000000054d8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054dc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000054e0: be812000
	s_cbranch_execnz 1942                                      // 0000000054e4: bfa60796 <packed_folded_w4a8+0x5840>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000054ec: 8c7e017e
	global_load_b32 v26, v[52:53], off                         // 0000000054f0: ee05007c 0000001a 00000034
	s_wait_loadcnt 0x0                                         // 0000000054fc: bfc00000
	v_mul_f32_e32 v21, v26, v25                                // 000000005500: 102a331a
	s_delay_alu instid0(valu_dep_1)                            // 000000005504: bf870001
	v_cmp_class_f32_e64 s0, v21, 0x198                         // 000000005508: d47e0000 0201ff15 00000198
	v_mul_f32_e32 v21, v22, v21                                // 000000005514: 102a2b16
	s_xor_b32 s0, s0, -1                                       // 000000005518: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000551c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005520: be812000
	s_cbranch_execnz 1944                                      // 000000005524: bfa60798 <packed_folded_w4a8+0x5888>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005528: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000552c: 8c7e017e
	global_load_b32 v26, v[98:99], off                         // 000000005530: ee05007c 0000001a 00000062
	s_wait_loadcnt 0x0                                         // 00000000553c: bfc00000
	v_mul_f32_e32 v22, v26, v25                                // 000000005540: 102c331a
	s_delay_alu instid0(valu_dep_1)                            // 000000005544: bf870001
	v_cmp_class_f32_e64 s0, v22, 0x198                         // 000000005548: d47e0000 0201ff16 00000198
	v_mul_f32_e32 v22, v23, v22                                // 000000005554: 102c2d17
	s_xor_b32 s0, s0, -1                                       // 000000005558: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000555c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005560: be812000
	s_cbranch_execnz 1946                                      // 000000005564: bfa6079a <packed_folded_w4a8+0x58d0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005568: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000556c: 8c7e017e
	s_and_b32 s0, s11, vcc_lo                                  // 000000005570: 8b006a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005574: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005578: be812000
	s_cbranch_execz 34                                         // 00000000557c: bfa50022 <packed_folded_w4a8+0x3b08>
	v_add_co_u32 v25, s0, v125, v124                           // 000000005580: d7000019 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000005588: bf88f19f
	v_add_co_ci_u32_e64 v26, null, 0, v126, s0                 // 00000000558c: d5207c1a 0002fc80
	s_wait_kmcnt 0x0                                           // 000000005594: bfc70000
	v_add_co_u32 v27, s0, s34, v48                             // 000000005598: d700001b 02026022
	v_bfe_u32 v23, v24, 16, 1                                  // 0000000055a0: d6100017 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 0000000055a8: bf870253
	v_lshlrev_b64_e32 v[25:26], 1, v[25:26]                    // 0000000055ac: 3e323281
	s_wait_alu depctr_va_sdst(0)                               // 0000000055b0: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s35, v49, s0                // 0000000055b4: d5207c1c 00026223
	v_or_b32_e32 v29, 0x400000, v24                            // 0000000055bc: 383a30ff 00400000
	v_add3_u32 v23, v23, v24, 0x7fff                           // 0000000055c4: d6550017 03fe3117 00007fff
	v_add_co_u32 v25, s0, v27, v25                             // 0000000055d0: d7000019 0202331b
	s_wait_alu depctr_va_sdst(0)                               // 0000000055d8: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v28, v26, s0                // 0000000055dc: d5207c1a 0002351c
	v_cmp_u_f32_e64 s0, v24, v24                               // 0000000055e4: d4180000 02023118
	s_wait_alu depctr_va_sdst(0)                               // 0000000055ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000055f0: bf870001
	v_cndmask_b32_e64 v23, v23, v29, s0                        // 0000000055f4: d5010017 00023b17
	global_store_d16_hi_b16 v[25:26], v23, off offset:32       // 0000000055fc: ee09407c 0b800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 000000005608: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000560c: 8c7e017e
	s_and_b32 s0, s8, vcc_lo                                   // 000000005610: 8b006a08
	s_wait_alu depctr_sa_sdst(0)                               // 000000005614: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005618: be812000
	s_cbranch_execz 28                                         // 00000000561c: bfa5001c <packed_folded_w4a8+0x3b90>
	v_bfe_u32 v23, v16, 16, 1                                  // 000000005620: d6100017 02052110
	s_wait_kmcnt 0x0                                           // 000000005628: bfc70000
	v_add_co_u32 v24, s0, s34, v48                             // 00000000562c: d7000018 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000005634: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s35, v49, s0                // 000000005638: d5207c19 00026223
	v_add3_u32 v26, v23, v16, 0x7fff                           // 000000005640: d655001a 03fe2117 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000564c: bf870003
	v_add_co_u32 v23, s0, v24, v56                             // 000000005650: d7000017 02027118
	v_or_b32_e32 v27, 0x400000, v16                            // 000000005658: 383620ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005660: bf88f19f
	v_add_co_ci_u32_e64 v24, null, v25, v57, s0                // 000000005664: d5207c18 00027319
	v_cmp_u_f32_e64 s0, v16, v16                               // 00000000566c: d4180000 02022110
	s_wait_alu depctr_va_sdst(0)                               // 000000005674: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005678: bf870001
	v_cndmask_b32_e64 v16, v26, v27, s0                        // 00000000567c: d5010010 0002371a
	global_store_d16_hi_b16 v[23:24], v16, off offset:32       // 000000005684: ee09407c 08000000 00002017
	s_wait_alu depctr_sa_sdst(0)                               // 000000005690: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005694: 8c7e017e
	s_and_b32 s0, s9, vcc_lo                                   // 000000005698: 8b006a09
	s_wait_alu depctr_sa_sdst(0)                               // 00000000569c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000056a0: be812000
	s_cbranch_execz 28                                         // 0000000056a4: bfa5001c <packed_folded_w4a8+0x3c18>
	s_wait_kmcnt 0x0                                           // 0000000056a8: bfc70000
	v_add_co_u32 v23, s0, s34, v48                             // 0000000056ac: d7000017 02026022
	v_bfe_u32 v16, v17, 16, 1                                  // 0000000056b4: d6100010 02052111
	s_wait_alu depctr_va_sdst(0)                               // 0000000056bc: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s35, v49, s0                // 0000000056c0: d5207c18 00026223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000056c8: bf870193
	v_add_co_u32 v23, s0, v23, v58                             // 0000000056cc: d7000017 02027517
	v_add3_u32 v16, v16, v17, 0x7fff                           // 0000000056d4: d6550010 03fe2310 00007fff
	v_or_b32_e32 v25, 0x400000, v17                            // 0000000056e0: 383222ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000056e8: bf88f19f
	v_add_co_ci_u32_e64 v24, null, v24, v59, s0                // 0000000056ec: d5207c18 00027718
	v_cmp_u_f32_e64 s0, v17, v17                               // 0000000056f4: d4180000 02022311
	s_wait_alu depctr_va_sdst(0)                               // 0000000056fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005700: bf870001
	v_cndmask_b32_e64 v16, v16, v25, s0                        // 000000005704: d5010010 00023310
	global_store_d16_hi_b16 v[23:24], v16, off offset:32       // 00000000570c: ee09407c 08000000 00002017
	s_wait_alu depctr_sa_sdst(0)                               // 000000005718: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000571c: 8c7e017e
	s_and_b32 s0, s10, vcc_lo                                  // 000000005720: 8b006a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005724: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005728: be812000
	s_cbranch_execz 28                                         // 00000000572c: bfa5001c <packed_folded_w4a8+0x3ca0>
	v_bfe_u32 v16, v18, 16, 1                                  // 000000005730: d6100010 02052112
	s_wait_kmcnt 0x0                                           // 000000005738: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 00000000573c: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000005744: bf88f19f
	v_add_co_ci_u32_e64 v23, null, s35, v49, s0                // 000000005748: d5207c17 00026223
	v_add3_u32 v24, v16, v18, 0x7fff                           // 000000005750: d6550018 03fe2510 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000575c: bf870003
	v_add_co_u32 v16, s0, v17, v60                             // 000000005760: d7000010 02027911
	v_or_b32_e32 v25, 0x400000, v18                            // 000000005768: 383224ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005770: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v23, v61, s0                // 000000005774: d5207c11 00027b17
	v_cmp_u_f32_e64 s0, v18, v18                               // 00000000577c: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 000000005784: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005788: bf870001
	v_cndmask_b32_e64 v18, v24, v25, s0                        // 00000000578c: d5010012 00023318
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 000000005794: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000057a4: 8c7e017e
	s_and_b32 s0, s12, vcc_lo                                  // 0000000057a8: 8b006a0c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057ac: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000057b0: be812000
	s_cbranch_execz 28                                         // 0000000057b4: bfa5001c <packed_folded_w4a8+0x3d28>
	v_bfe_u32 v16, v19, 16, 1                                  // 0000000057b8: d6100010 02052113
	s_wait_kmcnt 0x0                                           // 0000000057c0: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 0000000057c4: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000057cc: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 0000000057d0: d5207c12 00026223
	v_add3_u32 v23, v16, v19, 0x7fff                           // 0000000057d8: d6550017 03fe2710 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000057e4: bf870003
	v_add_co_u32 v16, s0, v17, v62                             // 0000000057e8: d7000010 02027d11
	v_or_b32_e32 v24, 0x400000, v19                            // 0000000057f0: 383026ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000057f8: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v63, s0                // 0000000057fc: d5207c11 00027f12
	v_cmp_u_f32_e64 s0, v19, v19                               // 000000005804: d4180000 02022713
	s_wait_alu depctr_va_sdst(0)                               // 00000000580c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005810: bf870001
	v_cndmask_b32_e64 v18, v23, v24, s0                        // 000000005814: d5010012 00023117
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 00000000581c: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005828: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000582c: 8c7e017e
	s_and_b32 s0, s13, vcc_lo                                  // 000000005830: 8b006a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005834: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005838: be812000
	s_cbranch_execz 28                                         // 00000000583c: bfa5001c <packed_folded_w4a8+0x3db0>
	v_bfe_u32 v16, v20, 16, 1                                  // 000000005840: d6100010 02052114
	s_wait_kmcnt 0x0                                           // 000000005848: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 00000000584c: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000005854: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 000000005858: d5207c12 00026223
	v_add3_u32 v19, v16, v20, 0x7fff                           // 000000005860: d6550013 03fe2910 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000586c: bf870003
	v_add_co_u32 v16, s0, v17, v64                             // 000000005870: d7000010 02028111
	v_or_b32_e32 v23, 0x400000, v20                            // 000000005878: 382e28ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005880: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v65, s0                // 000000005884: d5207c11 00028312
	v_cmp_u_f32_e64 s0, v20, v20                               // 00000000588c: d4180000 02022914
	s_wait_alu depctr_va_sdst(0)                               // 000000005894: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005898: bf870001
	v_cndmask_b32_e64 v18, v19, v23, s0                        // 00000000589c: d5010012 00022f13
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 0000000058a4: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000058b4: 8c7e017e
	s_and_b32 s0, s14, vcc_lo                                  // 0000000058b8: 8b006a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000058c0: be812000
	s_cbranch_execz 28                                         // 0000000058c4: bfa5001c <packed_folded_w4a8+0x3e38>
	v_bfe_u32 v16, v21, 16, 1                                  // 0000000058c8: d6100010 02052115
	s_wait_kmcnt 0x0                                           // 0000000058d0: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 0000000058d4: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000058dc: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 0000000058e0: d5207c12 00026223
	v_add3_u32 v19, v16, v21, 0x7fff                           // 0000000058e8: d6550013 03fe2b10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000058f4: bf870003
	v_add_co_u32 v16, s0, v17, v66                             // 0000000058f8: d7000010 02028511
	v_or_b32_e32 v20, 0x400000, v21                            // 000000005900: 38282aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005908: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v67, s0                // 00000000590c: d5207c11 00028712
	v_cmp_u_f32_e64 s0, v21, v21                               // 000000005914: d4180000 02022b15
	s_wait_alu depctr_va_sdst(0)                               // 00000000591c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005920: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000005924: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 00000000592c: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005938: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000593c: 8c7e017e
	s_and_b32 s0, s15, vcc_lo                                  // 000000005940: 8b006a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005944: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005948: be812000
	s_cbranch_execz 28                                         // 00000000594c: bfa5001c <packed_folded_w4a8+0x3ec0>
	v_bfe_u32 v16, v22, 16, 1                                  // 000000005950: d6100010 02052116
	s_wait_kmcnt 0x0                                           // 000000005958: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 00000000595c: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000005964: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 000000005968: d5207c12 00026223
	v_add3_u32 v19, v16, v22, 0x7fff                           // 000000005970: d6550013 03fe2d10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000597c: bf870003
	v_add_co_u32 v16, s0, v17, v68                             // 000000005980: d7000010 02028911
	v_or_b32_e32 v20, 0x400000, v22                            // 000000005988: 38282cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005990: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v69, s0                // 000000005994: d5207c11 00028b12
	v_cmp_u_f32_e64 s0, v22, v22                               // 00000000599c: d4180000 02022d16
	s_wait_alu depctr_va_sdst(0)                               // 0000000059a4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000059a8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 0000000059ac: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 0000000059b4: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000059c4: 8c7e017e
	global_load_u8 v16, v[38:39], off                          // 0000000059c8: ee04007c 00000010 00000026
	global_load_b32 v18, v[54:55], off                         // 0000000059d4: ee05007c 00000012 00000036
	s_wait_loadcnt 0x1                                         // 0000000059e0: bfc00001
	v_lshlrev_b32_e32 v17, 23, v16                             // 0000000059e4: 30222097
	s_wait_loadcnt 0x0                                         // 0000000059e8: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000059ec: bf870091
	v_mul_f32_e32 v16, v18, v17                                // 0000000059f0: 10202312
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 0000000059f4: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v8, v16                                 // 000000005a00: 10202108
	s_xor_b32 s0, s0, -1                                       // 000000005a04: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a08: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a0c: be812000
	s_cbranch_execnz 1665                                      // 000000005a10: bfa60681 <packed_folded_w4a8+0x5918>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a18: 8c7e017e
	global_load_b32 v18, v[100:101], off                       // 000000005a1c: ee05007c 00000012 00000064
	s_wait_loadcnt 0x0                                         // 000000005a28: bfc00000
	v_mul_f32_e32 v8, v18, v17                                 // 000000005a2c: 10102312
	s_delay_alu instid0(valu_dep_1)                            // 000000005a30: bf870001
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000005a34: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v9, v8                                   // 000000005a40: 10101109
	s_xor_b32 s0, s0, -1                                       // 000000005a44: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a48: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a4c: be812000
	s_cbranch_execnz 1667                                      // 000000005a50: bfa60683 <packed_folded_w4a8+0x5960>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a58: 8c7e017e
	global_load_b32 v18, v[102:103], off                       // 000000005a5c: ee05007c 00000012 00000066
	s_wait_loadcnt 0x0                                         // 000000005a68: bfc00000
	v_mul_f32_e32 v9, v18, v17                                 // 000000005a6c: 10122312
	s_delay_alu instid0(valu_dep_1)                            // 000000005a70: bf870001
	v_cmp_class_f32_e64 s0, v9, 0x198                          // 000000005a74: d47e0000 0201ff09 00000198
	v_mul_f32_e32 v9, v10, v9                                  // 000000005a80: 1012130a
	s_xor_b32 s0, s0, -1                                       // 000000005a84: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a88: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a8c: be812000
	s_cbranch_execnz 1669                                      // 000000005a90: bfa60685 <packed_folded_w4a8+0x59a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a98: 8c7e017e
	global_load_b32 v18, v[104:105], off                       // 000000005a9c: ee05007c 00000012 00000068
	s_wait_loadcnt 0x0                                         // 000000005aa8: bfc00000
	v_mul_f32_e32 v10, v18, v17                                // 000000005aac: 10142312
	s_delay_alu instid0(valu_dep_1)                            // 000000005ab0: bf870001
	v_cmp_class_f32_e64 s0, v10, 0x198                         // 000000005ab4: d47e0000 0201ff0a 00000198
	v_mul_f32_e32 v10, v11, v10                                // 000000005ac0: 1014150b
	s_xor_b32 s0, s0, -1                                       // 000000005ac4: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ac8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005acc: be812000
	s_cbranch_execnz 1671                                      // 000000005ad0: bfa60687 <packed_folded_w4a8+0x59f0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ad4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005ad8: 8c7e017e
	global_load_b32 v18, v[42:43], off                         // 000000005adc: ee05007c 00000012 0000002a
	s_wait_loadcnt 0x0                                         // 000000005ae8: bfc00000
	v_mul_f32_e32 v11, v18, v17                                // 000000005aec: 10162312
	s_delay_alu instid0(valu_dep_1)                            // 000000005af0: bf870001
	v_cmp_class_f32_e64 s0, v11, 0x198                         // 000000005af4: d47e0000 0201ff0b 00000198
	v_mul_f32_e32 v11, v12, v11                                // 000000005b00: 1016170c
	s_xor_b32 s0, s0, -1                                       // 000000005b04: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b08: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b0c: be812000
	s_cbranch_execnz 1673                                      // 000000005b10: bfa60689 <packed_folded_w4a8+0x5a38>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b18: 8c7e017e
	global_load_b32 v18, v[106:107], off                       // 000000005b1c: ee05007c 00000012 0000006a
	s_wait_loadcnt 0x0                                         // 000000005b28: bfc00000
	v_mul_f32_e32 v12, v18, v17                                // 000000005b2c: 10182312
	s_delay_alu instid0(valu_dep_1)                            // 000000005b30: bf870001
	v_cmp_class_f32_e64 s0, v12, 0x198                         // 000000005b34: d47e0000 0201ff0c 00000198
	v_mul_f32_e32 v12, v13, v12                                // 000000005b40: 1018190d
	s_xor_b32 s0, s0, -1                                       // 000000005b44: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b48: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b4c: be812000
	s_cbranch_execnz 1675                                      // 000000005b50: bfa6068b <packed_folded_w4a8+0x5a80>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b58: 8c7e017e
	global_load_b32 v18, v[44:45], off                         // 000000005b5c: ee05007c 00000012 0000002c
	s_wait_loadcnt 0x0                                         // 000000005b68: bfc00000
	v_mul_f32_e32 v13, v18, v17                                // 000000005b6c: 101a2312
	s_delay_alu instid0(valu_dep_1)                            // 000000005b70: bf870001
	v_cmp_class_f32_e64 s0, v13, 0x198                         // 000000005b74: d47e0000 0201ff0d 00000198
	v_mul_f32_e32 v13, v14, v13                                // 000000005b80: 101a1b0e
	s_xor_b32 s0, s0, -1                                       // 000000005b84: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b88: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b8c: be812000
	s_cbranch_execnz 1677                                      // 000000005b90: bfa6068d <packed_folded_w4a8+0x5ac8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b98: 8c7e017e
	global_load_b32 v18, v[108:109], off                       // 000000005b9c: ee05007c 00000012 0000006c
	s_wait_loadcnt 0x0                                         // 000000005ba8: bfc00000
	v_mul_f32_e32 v14, v18, v17                                // 000000005bac: 101c2312
	s_delay_alu instid0(valu_dep_1)                            // 000000005bb0: bf870001
	v_cmp_class_f32_e64 s0, v14, 0x198                         // 000000005bb4: d47e0000 0201ff0e 00000198
	v_mul_f32_e32 v14, v15, v14                                // 000000005bc0: 101c1d0f
	s_xor_b32 s0, s0, -1                                       // 000000005bc4: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bc8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005bcc: be812000
	s_cbranch_execnz 1679                                      // 000000005bd0: bfa6068f <packed_folded_w4a8+0x5b10>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005bd8: 8c7e017e
	s_and_b32 s0, s19, vcc_lo                                  // 000000005bdc: 8b006a13
	s_wait_alu depctr_sa_sdst(0)                               // 000000005be0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005be4: be812000
	s_cbranch_execz 34                                         // 000000005be8: bfa50022 <packed_folded_w4a8+0x4174>
	v_add_co_u32 v17, s0, v125, v124                           // 000000005bec: d7000011 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000005bf4: bf88f19f
	v_add_co_ci_u32_e64 v18, null, 0, v126, s0                 // 000000005bf8: d5207c12 0002fc80
	s_wait_kmcnt 0x0                                           // 000000005c00: bfc70000
	v_add_co_u32 v19, s0, s34, v40                             // 000000005c04: d7000013 02025022
	v_bfe_u32 v15, v16, 16, 1                                  // 000000005c0c: d610000f 02052110
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000005c14: bf870253
	v_lshlrev_b64_e32 v[17:18], 1, v[17:18]                    // 000000005c18: 3e222281
	s_wait_alu depctr_va_sdst(0)                               // 000000005c1c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s35, v41, s0                // 000000005c20: d5207c14 00025223
	v_or_b32_e32 v21, 0x400000, v16                            // 000000005c28: 382a20ff 00400000
	v_add3_u32 v15, v15, v16, 0x7fff                           // 000000005c30: d655000f 03fe210f 00007fff
	v_add_co_u32 v17, s0, v19, v17                             // 000000005c3c: d7000011 02022313
	s_wait_alu depctr_va_sdst(0)                               // 000000005c44: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v20, v18, s0                // 000000005c48: d5207c12 00022514
	v_cmp_u_f32_e64 s0, v16, v16                               // 000000005c50: d4180000 02022110
	s_wait_alu depctr_va_sdst(0)                               // 000000005c58: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005c5c: bf870001
	v_cndmask_b32_e64 v15, v15, v21, s0                        // 000000005c60: d501000f 00022b0f
	global_store_d16_hi_b16 v[17:18], v15, off offset:32       // 000000005c68: ee09407c 07800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005c78: 8c7e017e
	s_and_b32 s0, s16, vcc_lo                                  // 000000005c7c: 8b006a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c80: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005c84: be812000
	s_cbranch_execz 28                                         // 000000005c88: bfa5001c <packed_folded_w4a8+0x41fc>
	v_bfe_u32 v15, v8, 16, 1                                   // 000000005c8c: d610000f 02052108
	s_wait_kmcnt 0x0                                           // 000000005c94: bfc70000
	v_add_co_u32 v16, s0, s34, v40                             // 000000005c98: d7000010 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005ca0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v41, s0                // 000000005ca4: d5207c11 00025223
	v_add3_u32 v18, v15, v8, 0x7fff                            // 000000005cac: d6550012 03fe110f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005cb8: bf870003
	v_add_co_u32 v15, s0, v16, v56                             // 000000005cbc: d700000f 02027110
	v_or_b32_e32 v19, 0x400000, v8                             // 000000005cc4: 382610ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ccc: bf88f19f
	v_add_co_ci_u32_e64 v16, null, v17, v57, s0                // 000000005cd0: d5207c10 00027311
	v_cmp_u_f32_e64 s0, v8, v8                                 // 000000005cd8: d4180000 02021108
	s_wait_alu depctr_va_sdst(0)                               // 000000005ce0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005ce4: bf870001
	v_cndmask_b32_e64 v8, v18, v19, s0                         // 000000005ce8: d5010008 00022712
	global_store_d16_hi_b16 v[15:16], v8, off offset:32        // 000000005cf0: ee09407c 04000000 0000200f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cfc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005d00: 8c7e017e
	s_and_b32 s0, s17, vcc_lo                                  // 000000005d04: 8b006a11
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d08: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005d0c: be812000
	s_cbranch_execz 28                                         // 000000005d10: bfa5001c <packed_folded_w4a8+0x4284>
	s_wait_kmcnt 0x0                                           // 000000005d14: bfc70000
	v_add_co_u32 v15, s0, s34, v40                             // 000000005d18: d700000f 02025022
	v_bfe_u32 v8, v9, 16, 1                                    // 000000005d20: d6100008 02052109
	s_wait_alu depctr_va_sdst(0)                               // 000000005d28: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s35, v41, s0                // 000000005d2c: d5207c10 00025223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005d34: bf870193
	v_add_co_u32 v15, s0, v15, v58                             // 000000005d38: d700000f 0202750f
	v_add3_u32 v8, v8, v9, 0x7fff                              // 000000005d40: d6550008 03fe1308 00007fff
	v_or_b32_e32 v17, 0x400000, v9                             // 000000005d4c: 382212ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005d54: bf88f19f
	v_add_co_ci_u32_e64 v16, null, v16, v59, s0                // 000000005d58: d5207c10 00027710
	v_cmp_u_f32_e64 s0, v9, v9                                 // 000000005d60: d4180000 02021309
	s_wait_alu depctr_va_sdst(0)                               // 000000005d68: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005d6c: bf870001
	v_cndmask_b32_e64 v8, v8, v17, s0                          // 000000005d70: d5010008 00022308
	global_store_d16_hi_b16 v[15:16], v8, off offset:32        // 000000005d78: ee09407c 04000000 0000200f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d84: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005d88: 8c7e017e
	s_and_b32 s0, s18, vcc_lo                                  // 000000005d8c: 8b006a12
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d90: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005d94: be812000
	s_cbranch_execz 28                                         // 000000005d98: bfa5001c <packed_folded_w4a8+0x430c>
	v_bfe_u32 v8, v10, 16, 1                                   // 000000005d9c: d6100008 0205210a
	s_wait_kmcnt 0x0                                           // 000000005da4: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005da8: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005db0: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s35, v41, s0                // 000000005db4: d5207c0f 00025223
	v_add3_u32 v16, v8, v10, 0x7fff                            // 000000005dbc: d6550010 03fe1508 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005dc8: bf870003
	v_add_co_u32 v8, s0, v9, v60                               // 000000005dcc: d7000008 02027909
	v_or_b32_e32 v17, 0x400000, v10                            // 000000005dd4: 382214ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ddc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v15, v61, s0                 // 000000005de0: d5207c09 00027b0f
	v_cmp_u_f32_e64 s0, v10, v10                               // 000000005de8: d4180000 0202150a
	s_wait_alu depctr_va_sdst(0)                               // 000000005df0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005df4: bf870001
	v_cndmask_b32_e64 v10, v16, v17, s0                        // 000000005df8: d501000a 00022310
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005e00: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005e10: 8c7e017e
	s_and_b32 s0, s20, vcc_lo                                  // 000000005e14: 8b006a14
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e18: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005e1c: be812000
	s_cbranch_execz 28                                         // 000000005e20: bfa5001c <packed_folded_w4a8+0x4394>
	v_bfe_u32 v8, v11, 16, 1                                   // 000000005e24: d6100008 0205210b
	s_wait_kmcnt 0x0                                           // 000000005e2c: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005e30: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005e38: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005e3c: d5207c0a 00025223
	v_add3_u32 v15, v8, v11, 0x7fff                            // 000000005e44: d655000f 03fe1708 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005e50: bf870003
	v_add_co_u32 v8, s0, v9, v62                               // 000000005e54: d7000008 02027d09
	v_or_b32_e32 v16, 0x400000, v11                            // 000000005e5c: 382016ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005e64: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v63, s0                 // 000000005e68: d5207c09 00027f0a
	v_cmp_u_f32_e64 s0, v11, v11                               // 000000005e70: d4180000 0202170b
	s_wait_alu depctr_va_sdst(0)                               // 000000005e78: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005e7c: bf870001
	v_cndmask_b32_e64 v10, v15, v16, s0                        // 000000005e80: d501000a 0002210f
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005e88: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005e98: 8c7e017e
	s_and_b32 s0, s21, vcc_lo                                  // 000000005e9c: 8b006a15
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ea0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ea4: be812000
	s_cbranch_execz 28                                         // 000000005ea8: bfa5001c <packed_folded_w4a8+0x441c>
	v_bfe_u32 v8, v12, 16, 1                                   // 000000005eac: d6100008 0205210c
	s_wait_kmcnt 0x0                                           // 000000005eb4: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005eb8: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005ec0: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005ec4: d5207c0a 00025223
	v_add3_u32 v11, v8, v12, 0x7fff                            // 000000005ecc: d655000b 03fe1908 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005ed8: bf870003
	v_add_co_u32 v8, s0, v9, v64                               // 000000005edc: d7000008 02028109
	v_or_b32_e32 v15, 0x400000, v12                            // 000000005ee4: 381e18ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005eec: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v65, s0                 // 000000005ef0: d5207c09 0002830a
	v_cmp_u_f32_e64 s0, v12, v12                               // 000000005ef8: d4180000 0202190c
	s_wait_alu depctr_va_sdst(0)                               // 000000005f00: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005f04: bf870001
	v_cndmask_b32_e64 v10, v11, v15, s0                        // 000000005f08: d501000a 00021f0b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005f10: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f1c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005f20: 8c7e017e
	s_and_b32 s0, s22, vcc_lo                                  // 000000005f24: 8b006a16
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f28: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005f2c: be812000
	s_cbranch_execz 28                                         // 000000005f30: bfa5001c <packed_folded_w4a8+0x44a4>
	v_bfe_u32 v8, v13, 16, 1                                   // 000000005f34: d6100008 0205210d
	s_wait_kmcnt 0x0                                           // 000000005f3c: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005f40: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005f48: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005f4c: d5207c0a 00025223
	v_add3_u32 v11, v8, v13, 0x7fff                            // 000000005f54: d655000b 03fe1b08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005f60: bf870003
	v_add_co_u32 v8, s0, v9, v66                               // 000000005f64: d7000008 02028509
	v_or_b32_e32 v12, 0x400000, v13                            // 000000005f6c: 38181aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005f74: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v67, s0                 // 000000005f78: d5207c09 0002870a
	v_cmp_u_f32_e64 s0, v13, v13                               // 000000005f80: d4180000 02021b0d
	s_wait_alu depctr_va_sdst(0)                               // 000000005f88: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005f8c: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000005f90: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005f98: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fa4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005fa8: 8c7e017e
	s_and_b32 s0, s23, vcc_lo                                  // 000000005fac: 8b006a17
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fb0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005fb4: be812000
	s_cbranch_execz 28                                         // 000000005fb8: bfa5001c <packed_folded_w4a8+0x452c>
	v_bfe_u32 v8, v14, 16, 1                                   // 000000005fbc: d6100008 0205210e
	s_wait_kmcnt 0x0                                           // 000000005fc4: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005fc8: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005fd0: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005fd4: d5207c0a 00025223
	v_add3_u32 v11, v8, v14, 0x7fff                            // 000000005fdc: d655000b 03fe1d08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005fe8: bf870003
	v_add_co_u32 v8, s0, v9, v68                               // 000000005fec: d7000008 02028909
	v_or_b32_e32 v12, 0x400000, v14                            // 000000005ff4: 38181cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ffc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v69, s0                 // 000000006000: d5207c09 00028b0a
	v_cmp_u_f32_e64 s0, v14, v14                               // 000000006008: d4180000 02021d0e
	s_wait_alu depctr_va_sdst(0)                               // 000000006010: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006014: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000006018: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000006020: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 00000000602c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006030: 8c7e017e
	global_load_u8 v8, v[38:39], off                           // 000000006034: ee04007c 00000008 00000026
	global_load_b32 v10, v[46:47], off                         // 000000006040: ee05007c 0000000a 0000002e
	s_wait_loadcnt 0x1                                         // 00000000604c: bfc00001
	v_lshlrev_b32_e32 v9, 23, v8                               // 000000006050: 30121097
	s_wait_loadcnt 0x0                                         // 000000006054: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006058: bf870091
	v_mul_f32_e32 v8, v10, v9                                  // 00000000605c: 1010130a
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000006060: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v0, v8                                   // 00000000606c: 10101100
	s_xor_b32 s0, s0, -1                                       // 000000006070: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006074: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006078: be812000
	s_cbranch_execnz 1398                                      // 00000000607c: bfa60576 <packed_folded_w4a8+0x5b58>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006080: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006084: 8c7e017e
	global_load_b32 v10, v[110:111], off                       // 000000006088: ee05007c 0000000a 0000006e
	s_wait_loadcnt 0x0                                         // 000000006094: bfc00000
	v_mul_f32_e32 v0, v10, v9                                  // 000000006098: 1000130a
	s_delay_alu instid0(valu_dep_1)                            // 00000000609c: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 0000000060a0: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v1, v0                                   // 0000000060ac: 10000101
	s_xor_b32 s0, s0, -1                                       // 0000000060b0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060b4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000060b8: be812000
	s_cbranch_execnz 1400                                      // 0000000060bc: bfa60578 <packed_folded_w4a8+0x5ba0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000060c4: 8c7e017e
	global_load_b32 v10, v[112:113], off                       // 0000000060c8: ee05007c 0000000a 00000070
	s_wait_loadcnt 0x0                                         // 0000000060d4: bfc00000
	v_mul_f32_e32 v1, v10, v9                                  // 0000000060d8: 1002130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000060dc: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 0000000060e0: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v2, v1                                   // 0000000060ec: 10020302
	s_xor_b32 s0, s0, -1                                       // 0000000060f0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000060f8: be812000
	s_cbranch_execnz 1402                                      // 0000000060fc: bfa6057a <packed_folded_w4a8+0x5be8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006100: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006104: 8c7e017e
	global_load_b32 v10, v[114:115], off                       // 000000006108: ee05007c 0000000a 00000072
	s_wait_loadcnt 0x0                                         // 000000006114: bfc00000
	v_mul_f32_e32 v2, v10, v9                                  // 000000006118: 1004130a
	s_delay_alu instid0(valu_dep_1)                            // 00000000611c: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000006120: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 00000000612c: 10040503
	s_xor_b32 s0, s0, -1                                       // 000000006130: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006134: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006138: be812000
	s_cbranch_execnz 1404                                      // 00000000613c: bfa6057c <packed_folded_w4a8+0x5c30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006140: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006144: 8c7e017e
	global_load_b32 v10, v[34:35], off                         // 000000006148: ee05007c 0000000a 00000022
	s_wait_loadcnt 0x0                                         // 000000006154: bfc00000
	v_mul_f32_e32 v3, v10, v9                                  // 000000006158: 1006130a
	s_delay_alu instid0(valu_dep_1)                            // 00000000615c: bf870001
	v_cmp_class_f32_e64 s0, v3, 0x198                          // 000000006160: d47e0000 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 00000000616c: 10060704
	s_xor_b32 s0, s0, -1                                       // 000000006170: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006174: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006178: be812000
	s_cbranch_execnz 1406                                      // 00000000617c: bfa6057e <packed_folded_w4a8+0x5c78>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006180: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006184: 8c7e017e
	global_load_b32 v10, v[116:117], off                       // 000000006188: ee05007c 0000000a 00000074
	s_wait_loadcnt 0x0                                         // 000000006194: bfc00000
	v_mul_f32_e32 v4, v10, v9                                  // 000000006198: 1008130a
	s_delay_alu instid0(valu_dep_1)                            // 00000000619c: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 0000000061a0: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 0000000061ac: 10080905
	s_xor_b32 s0, s0, -1                                       // 0000000061b0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061b4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000061b8: be812000
	s_cbranch_execnz 1408                                      // 0000000061bc: bfa60580 <packed_folded_w4a8+0x5cc0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000061c4: 8c7e017e
	global_load_b32 v10, v[36:37], off                         // 0000000061c8: ee05007c 0000000a 00000024
	s_wait_loadcnt 0x0                                         // 0000000061d4: bfc00000
	v_mul_f32_e32 v5, v10, v9                                  // 0000000061d8: 100a130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000061dc: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 0000000061e0: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 0000000061ec: 100a0b06
	s_xor_b32 s0, s0, -1                                       // 0000000061f0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000061f8: be812000
	s_cbranch_execnz 1410                                      // 0000000061fc: bfa60582 <packed_folded_w4a8+0x5d08>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006200: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006204: 8c7e017e
	global_load_b32 v10, v[120:121], off                       // 000000006208: ee05007c 0000000a 00000078
	s_wait_loadcnt 0x0                                         // 000000006214: bfc00000
	v_mul_f32_e32 v6, v10, v9                                  // 000000006218: 100c130a
	s_delay_alu instid0(valu_dep_1)                            // 00000000621c: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000006220: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v7, v6                                   // 00000000622c: 100c0d07
	s_xor_b32 s0, s0, -1                                       // 000000006230: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006234: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006238: be812000
	s_cbranch_execnz 1412                                      // 00000000623c: bfa60584 <packed_folded_w4a8+0x5d50>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006240: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006244: 8c7e017e
	s_and_b32 s0, s27, vcc_lo                                  // 000000006248: 8b006a1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000624c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006250: be812000
	s_cbranch_execz 34                                         // 000000006254: bfa50022 <packed_folded_w4a8+0x47e0>
	v_add_co_u32 v9, s0, v125, v124                            // 000000006258: d7000009 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000006260: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v126, s0                 // 000000006264: d5207c0a 0002fc80
	s_wait_kmcnt 0x0                                           // 00000000626c: bfc70000
	v_add_co_u32 v11, s0, s34, v32                             // 000000006270: d700000b 02024022
	v_bfe_u32 v7, v8, 16, 1                                    // 000000006278: d6100007 02052108
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000006280: bf870253
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000006284: 3e121281
	s_wait_alu depctr_va_sdst(0)                               // 000000006288: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s35, v33, s0                // 00000000628c: d5207c0c 00024223
	v_or_b32_e32 v13, 0x400000, v8                             // 000000006294: 381a10ff 00400000
	v_add3_u32 v7, v7, v8, 0x7fff                              // 00000000629c: d6550007 03fe1107 00007fff
	v_add_co_u32 v9, s0, v11, v9                               // 0000000062a8: d7000009 0202130b
	s_wait_alu depctr_va_sdst(0)                               // 0000000062b0: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v12, v10, s0                // 0000000062b4: d5207c0a 0002150c
	v_cmp_u_f32_e64 s0, v8, v8                                 // 0000000062bc: d4180000 02021108
	s_wait_alu depctr_va_sdst(0)                               // 0000000062c4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000062c8: bf870001
	v_cndmask_b32_e64 v7, v7, v13, s0                          // 0000000062cc: d5010007 00021b07
	global_store_d16_hi_b16 v[9:10], v7, off offset:32         // 0000000062d4: ee09407c 03800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000062e4: 8c7e017e
	s_and_b32 s0, s24, vcc_lo                                  // 0000000062e8: 8b006a18
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000062f0: be812000
	s_cbranch_execz 28                                         // 0000000062f4: bfa5001c <packed_folded_w4a8+0x4868>
	v_bfe_u32 v7, v0, 16, 1                                    // 0000000062f8: d6100007 02052100
	s_wait_kmcnt 0x0                                           // 000000006300: bfc70000
	v_add_co_u32 v8, s0, s34, v32                              // 000000006304: d7000008 02024022
	s_wait_alu depctr_va_sdst(0)                               // 00000000630c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s35, v33, s0                 // 000000006310: d5207c09 00024223
	v_add3_u32 v10, v7, v0, 0x7fff                             // 000000006318: d655000a 03fe0107 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006324: bf870003
	v_add_co_u32 v7, s0, v8, v56                               // 000000006328: d7000007 02027108
	v_or_b32_e32 v11, 0x400000, v0                             // 000000006330: 381600ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006338: bf88f19f
	v_add_co_ci_u32_e64 v8, null, v9, v57, s0                  // 00000000633c: d5207c08 00027309
	v_cmp_u_f32_e64 s0, v0, v0                                 // 000000006344: d4180000 02020100
	s_wait_alu depctr_va_sdst(0)                               // 00000000634c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006350: bf870001
	v_cndmask_b32_e64 v0, v10, v11, s0                         // 000000006354: d5010000 0002170a
	global_store_d16_hi_b16 v[7:8], v0, off offset:32          // 00000000635c: ee09407c 00000000 00002007
	s_wait_alu depctr_sa_sdst(0)                               // 000000006368: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000636c: 8c7e017e
	s_and_b32 s0, s25, vcc_lo                                  // 000000006370: 8b006a19
	s_wait_alu depctr_sa_sdst(0)                               // 000000006374: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006378: be812000
	s_cbranch_execz 28                                         // 00000000637c: bfa5001c <packed_folded_w4a8+0x48f0>
	s_wait_kmcnt 0x0                                           // 000000006380: bfc70000
	v_add_co_u32 v7, s0, s34, v32                              // 000000006384: d7000007 02024022
	v_bfe_u32 v0, v1, 16, 1                                    // 00000000638c: d6100000 02052101
	s_wait_alu depctr_va_sdst(0)                               // 000000006394: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s35, v33, s0                 // 000000006398: d5207c08 00024223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000063a0: bf870193
	v_add_co_u32 v7, s0, v7, v58                               // 0000000063a4: d7000007 02027507
	v_add3_u32 v0, v0, v1, 0x7fff                              // 0000000063ac: d6550000 03fe0300 00007fff
	v_or_b32_e32 v9, 0x400000, v1                              // 0000000063b8: 381202ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000063c0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, v8, v59, s0                  // 0000000063c4: d5207c08 00027708
	v_cmp_u_f32_e64 s0, v1, v1                                 // 0000000063cc: d4180000 02020301
	s_wait_alu depctr_va_sdst(0)                               // 0000000063d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000063d8: bf870001
	v_cndmask_b32_e64 v0, v0, v9, s0                           // 0000000063dc: d5010000 00021300
	global_store_d16_hi_b16 v[7:8], v0, off offset:32          // 0000000063e4: ee09407c 00000000 00002007
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000063f4: 8c7e017e
	s_and_b32 s0, s26, vcc_lo                                  // 0000000063f8: 8b006a1a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063fc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006400: be812000
	s_cbranch_execz 28                                         // 000000006404: bfa5001c <packed_folded_w4a8+0x4978>
	v_bfe_u32 v0, v2, 16, 1                                    // 000000006408: d6100000 02052102
	s_wait_kmcnt 0x0                                           // 000000006410: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 000000006414: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 00000000641c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v33, s0                 // 000000006420: d5207c07 00024223
	v_add3_u32 v8, v0, v2, 0x7fff                              // 000000006428: d6550008 03fe0500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006434: bf870003
	v_add_co_u32 v0, s0, v1, v60                               // 000000006438: d7000000 02027901
	v_or_b32_e32 v9, 0x400000, v2                              // 000000006440: 381204ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006448: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v7, v61, s0                  // 00000000644c: d5207c01 00027b07
	v_cmp_u_f32_e64 s0, v2, v2                                 // 000000006454: d4180000 02020502
	s_wait_alu depctr_va_sdst(0)                               // 00000000645c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006460: bf870001
	v_cndmask_b32_e64 v2, v8, v9, s0                           // 000000006464: d5010002 00021308
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000646c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006478: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000647c: 8c7e017e
	s_and_b32 s0, s28, vcc_lo                                  // 000000006480: 8b006a1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006484: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006488: be812000
	s_cbranch_execz 28                                         // 00000000648c: bfa5001c <packed_folded_w4a8+0x4a00>
	v_bfe_u32 v0, v3, 16, 1                                    // 000000006490: d6100000 02052103
	s_wait_kmcnt 0x0                                           // 000000006498: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 00000000649c: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000064a4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 0000000064a8: d5207c02 00024223
	v_add3_u32 v7, v0, v3, 0x7fff                              // 0000000064b0: d6550007 03fe0700 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000064bc: bf870003
	v_add_co_u32 v0, s0, v1, v62                               // 0000000064c0: d7000000 02027d01
	v_or_b32_e32 v8, 0x400000, v3                              // 0000000064c8: 381006ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000064d0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v63, s0                  // 0000000064d4: d5207c01 00027f02
	v_cmp_u_f32_e64 s0, v3, v3                                 // 0000000064dc: d4180000 02020703
	s_wait_alu depctr_va_sdst(0)                               // 0000000064e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000064e8: bf870001
	v_cndmask_b32_e64 v2, v7, v8, s0                           // 0000000064ec: d5010002 00021107
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000064f4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006500: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006504: 8c7e017e
	s_and_b32 s0, s29, vcc_lo                                  // 000000006508: 8b006a1d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000650c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006510: be812000
	s_cbranch_execz 28                                         // 000000006514: bfa5001c <packed_folded_w4a8+0x4a88>
	v_bfe_u32 v0, v4, 16, 1                                    // 000000006518: d6100000 02052104
	s_wait_kmcnt 0x0                                           // 000000006520: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 000000006524: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 00000000652c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 000000006530: d5207c02 00024223
	v_add3_u32 v3, v0, v4, 0x7fff                              // 000000006538: d6550003 03fe0900 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006544: bf870003
	v_add_co_u32 v0, s0, v1, v64                               // 000000006548: d7000000 02028101
	v_or_b32_e32 v7, 0x400000, v4                              // 000000006550: 380e08ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006558: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v65, s0                  // 00000000655c: d5207c01 00028302
	v_cmp_u_f32_e64 s0, v4, v4                                 // 000000006564: d4180000 02020904
	s_wait_alu depctr_va_sdst(0)                               // 00000000656c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006570: bf870001
	v_cndmask_b32_e64 v2, v3, v7, s0                           // 000000006574: d5010002 00020f03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000657c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006588: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000658c: 8c7e017e
	s_and_b32 s0, s30, vcc_lo                                  // 000000006590: 8b006a1e
	s_wait_alu depctr_sa_sdst(0)                               // 000000006594: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006598: be812000
	s_cbranch_execz 28                                         // 00000000659c: bfa5001c <packed_folded_w4a8+0x4b10>
	v_bfe_u32 v0, v5, 16, 1                                    // 0000000065a0: d6100000 02052105
	s_wait_kmcnt 0x0                                           // 0000000065a8: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 0000000065ac: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000065b4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 0000000065b8: d5207c02 00024223
	v_add3_u32 v3, v0, v5, 0x7fff                              // 0000000065c0: d6550003 03fe0b00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000065cc: bf870003
	v_add_co_u32 v0, s0, v1, v66                               // 0000000065d0: d7000000 02028501
	v_or_b32_e32 v4, 0x400000, v5                              // 0000000065d8: 38080aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000065e0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v67, s0                  // 0000000065e4: d5207c01 00028702
	v_cmp_u_f32_e64 s0, v5, v5                                 // 0000000065ec: d4180000 02020b05
	s_wait_alu depctr_va_sdst(0)                               // 0000000065f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000065f8: bf870001
	v_cndmask_b32_e64 v2, v3, v4, s0                           // 0000000065fc: d5010002 00020903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006604: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006610: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006614: 8c7e017e
	s_and_b32 s0, s31, vcc_lo                                  // 000000006618: 8b006a1f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000661c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006620: be812000
	s_cbranch_execz 25                                         // 000000006624: bfa50019 <packed_folded_w4a8+0x4b8c>
	v_bfe_u32 v0, v6, 16, 1                                    // 000000006628: d6100000 02052106
	s_wait_kmcnt 0x0                                           // 000000006630: bfc70000
	v_add_co_u32 v1, vcc_lo, s34, v32                          // 000000006634: d7006a01 02024022
	s_wait_alu depctr_va_vcc(0)                                // 00000000663c: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s35, v33, vcc_lo             // 000000006640: d5207c02 01aa4223
	v_add3_u32 v3, v0, v6, 0x7fff                              // 000000006648: d6550003 03fe0d00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006654: bf870003
	v_add_co_u32 v0, vcc_lo, v1, v68                           // 000000006658: d7006a00 02028901
	v_or_b32_e32 v4, 0x400000, v6                              // 000000006660: 38080cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006668: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v69, vcc_lo              // 00000000666c: d5207c01 01aa8b02
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000006674: 7c300d06
	s_wait_alu depctr_va_vcc(0)                                // 000000006678: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 00000000667c: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006680: ee09407c 01000000 00002000
	s_nop 0                                                    // 00000000668c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000006690: bfb60003
	s_endpgm                                                   // 000000006694: bfb00000
	v_cvt_f64_f32_e32 v[74:75], v56                            // 000000006698: 7e942138
	v_cvt_f64_f32_e32 v[76:77], v70                            // 00000000669c: 7e982146
	v_cvt_f64_f32_e32 v[78:79], v71                            // 0000000066a0: 7e9c2147
	v_cmp_eq_f32_e64 s2, 0, v56                                // 0000000066a4: d4120002 02027080
	v_cmp_class_f32_e64 s4, v71, 0x1f8                         // 0000000066ac: d47e0004 0201ff47 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000066b8: 8b020402
	v_mul_f64_e32 v[74:75], v[74:75], v[76:77]                 // 0000000066bc: 0c94994a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066c0: bf870091
	v_mul_f64_e32 v[74:75], v[74:75], v[78:79]                 // 0000000066c4: 0c949d4a
	v_cvt_f32_f64_e32 v74, v[74:75]                            // 0000000066c8: 7e941f4a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066cc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066d0: bf870001
	v_cndmask_b32_e64 v88, v74, 0, s2                          // 0000000066d4: d5010058 0009014a
	s_branch 61552                                             // 0000000066dc: bfa0f070 <packed_folded_w4a8+0xda0>
	v_cvt_f64_f32_e32 v[76:77], v57                            // 0000000066e0: 7e982139
	v_cvt_f64_f32_e32 v[78:79], v70                            // 0000000066e4: 7e9c2146
	v_cvt_f64_f32_e32 v[80:81], v56                            // 0000000066e8: 7ea02138
	v_cmp_eq_f32_e64 s2, 0, v57                                // 0000000066ec: d4120002 02027280
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000066f4: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006700: 8b020402
	v_mul_f64_e32 v[76:77], v[76:77], v[78:79]                 // 000000006704: 0c989d4c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006708: bf870091
	v_mul_f64_e32 v[76:77], v[76:77], v[80:81]                 // 00000000670c: 0c98a14c
	v_cvt_f32_f64_e32 v71, v[76:77]                            // 000000006710: 7e8e1f4c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006714: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006718: bf870001
	v_cndmask_b32_e64 v89, v71, 0, s2                          // 00000000671c: d5010059 00090147
	s_branch 61569                                             // 000000006724: bfa0f081 <packed_folded_w4a8+0xe2c>
	v_cvt_f64_f32_e32 v[78:79], v58                            // 000000006728: 7e9c213a
	v_cvt_f64_f32_e32 v[80:81], v70                            // 00000000672c: 7ea02146
	v_cvt_f64_f32_e32 v[82:83], v56                            // 000000006730: 7ea42138
	v_cmp_eq_f32_e64 s2, 0, v58                                // 000000006734: d4120002 02027480
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 00000000673c: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006748: 8b020402
	v_mul_f64_e32 v[78:79], v[78:79], v[80:81]                 // 00000000674c: 0c9ca14e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006750: bf870091
	v_mul_f64_e32 v[78:79], v[78:79], v[82:83]                 // 000000006754: 0c9ca54e
	v_cvt_f32_f64_e32 v57, v[78:79]                            // 000000006758: 7e721f4e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000675c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006760: bf870001
	v_cndmask_b32_e64 v91, v57, 0, s2                          // 000000006764: d501005b 00090139
	s_branch 61586                                             // 00000000676c: bfa0f092 <packed_folded_w4a8+0xeb8>
	v_cvt_f64_f32_e32 v[57:58], v59                            // 000000006770: 7e72213b
	v_cvt_f64_f32_e32 v[80:81], v70                            // 000000006774: 7ea02146
	v_cvt_f64_f32_e32 v[82:83], v56                            // 000000006778: 7ea42138
	v_cmp_eq_f32_e64 s2, 0, v59                                // 00000000677c: d4120002 02027680
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 000000006784: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006790: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[80:81]                 // 000000006794: 0c72a139
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006798: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[82:83]                 // 00000000679c: 0c72a539
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000067a0: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000067a8: bf870001
	v_cndmask_b32_e64 v92, v57, 0, s2                          // 0000000067ac: d501005c 00090139
	s_branch 61603                                             // 0000000067b4: bfa0f0a3 <packed_folded_w4a8+0xf44>
	v_cvt_f64_f32_e32 v[57:58], v60                            // 0000000067b8: 7e72213c
	v_cvt_f64_f32_e32 v[82:83], v70                            // 0000000067bc: 7ea42146
	v_cvt_f64_f32_e32 v[84:85], v56                            // 0000000067c0: 7ea82138
	v_cmp_eq_f32_e64 s2, 0, v60                                // 0000000067c4: d4120002 02027880
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000067cc: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000067d8: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[82:83]                 // 0000000067dc: 0c72a539
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000067e0: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[84:85]                 // 0000000067e4: 0c72a939
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000067e8: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067ec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000067f0: bf870001
	v_cndmask_b32_e64 v93, v57, 0, s2                          // 0000000067f4: d501005d 00090139
	s_branch 61620                                             // 0000000067fc: bfa0f0b4 <packed_folded_w4a8+0xfd0>
	v_cvt_f64_f32_e32 v[57:58], v61                            // 000000006800: 7e72213d
	v_cvt_f64_f32_e32 v[59:60], v70                            // 000000006804: 7e762146
	v_cvt_f64_f32_e32 v[84:85], v56                            // 000000006808: 7ea82138
	v_cmp_eq_f32_e64 s2, 0, v61                                // 00000000680c: d4120002 02027a80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 000000006814: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006820: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 000000006824: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006828: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[84:85]                 // 00000000682c: 0c72a939
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006830: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000006834: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006838: bf870001
	v_cndmask_b32_e64 v94, v57, 0, s2                          // 00000000683c: d501005e 00090139
	s_branch 61637                                             // 000000006844: bfa0f0c5 <packed_folded_w4a8+0x105c>
	v_cvt_f64_f32_e32 v[57:58], v62                            // 000000006848: 7e72213e
	v_cvt_f64_f32_e32 v[59:60], v70                            // 00000000684c: 7e762146
	v_cvt_f64_f32_e32 v[86:87], v56                            // 000000006850: 7eac2138
	v_cmp_eq_f32_e64 s2, 0, v62                                // 000000006854: d4120002 02027c80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 00000000685c: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006868: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 00000000686c: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006870: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[86:87]                 // 000000006874: 0c72ad39
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006878: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000687c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006880: bf870001
	v_cndmask_b32_e64 v95, v57, 0, s2                          // 000000006884: d501005f 00090139
	s_branch 61654                                             // 00000000688c: bfa0f0d6 <packed_folded_w4a8+0x10e8>
	v_cvt_f64_f32_e32 v[57:58], v63                            // 000000006890: 7e72213f
	v_cvt_f64_f32_e32 v[59:60], v70                            // 000000006894: 7e762146
	v_cvt_f64_f32_e32 v[61:62], v56                            // 000000006898: 7e7a2138
	v_cmp_eq_f32_e64 s2, 0, v63                                // 00000000689c: d4120002 02027e80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000068a4: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000068b0: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 0000000068b4: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000068b8: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[61:62]                 // 0000000068bc: 0c727b39
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000068c0: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000068c8: bf870001
	v_cndmask_b32_e64 v96, v57, 0, s2                          // 0000000068cc: d5010060 00090139
	s_branch 61670                                             // 0000000068d4: bfa0f0e6 <packed_folded_w4a8+0x1170>
	v_cvt_f64_f32_e32 v[91:92], v48                            // 0000000068d8: 7eb62130
	v_cvt_f64_f32_e32 v[93:94], v106                           // 0000000068dc: 7eba216a
	v_cvt_f64_f32_e32 v[95:96], v90                            // 0000000068e0: 7ebe215a
	v_cmp_eq_f32_e64 s8, 0, v48                                // 0000000068e4: d4120008 02026080
	v_cmp_class_f32_e64 s10, v90, 0x1f8                        // 0000000068ec: d47e000a 0201ff5a 000001f8
	s_and_b32 s8, s8, s10                                      // 0000000068f8: 8b080a08
	v_mul_f64_e32 v[91:92], v[91:92], v[93:94]                 // 0000000068fc: 0cb6bb5b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006900: bf870091
	v_mul_f64_e32 v[91:92], v[91:92], v[95:96]                 // 000000006904: 0cb6bf5b
	v_cvt_f32_f64_e32 v91, v[91:92]                            // 000000006908: 7eb61f5b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000690c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006910: bf870001
	v_cndmask_b32_e64 v103, v91, 0, s8                         // 000000006914: d5010067 0021015b
	s_branch 62048                                             // 00000000691c: bfa0f260 <packed_folded_w4a8+0x17a0>
	v_cvt_f64_f32_e32 v[92:93], v49                            // 000000006920: 7eb82131
	v_cvt_f64_f32_e32 v[94:95], v106                           // 000000006924: 7ebc216a
	v_cvt_f64_f32_e32 v[96:97], v48                            // 000000006928: 7ec02130
	v_cmp_eq_f32_e64 s8, 0, v49                                // 00000000692c: d4120008 02026280
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006934: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006940: 8b080a08
	v_mul_f64_e32 v[92:93], v[92:93], v[94:95]                 // 000000006944: 0cb8bd5c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006948: bf870091
	v_mul_f64_e32 v[92:93], v[92:93], v[96:97]                 // 00000000694c: 0cb8c15c
	v_cvt_f32_f64_e32 v92, v[92:93]                            // 000000006950: 7eb81f5c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006954: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006958: bf870001
	v_cndmask_b32_e64 v104, v92, 0, s8                         // 00000000695c: d5010068 0021015c
	s_branch 62063                                             // 000000006964: bfa0f26f <packed_folded_w4a8+0x1824>
	v_cvt_f64_f32_e32 v[94:95], v50                            // 000000006968: 7ebc2132
	v_cvt_f64_f32_e32 v[96:97], v106                           // 00000000696c: 7ec0216a
	v_cvt_f64_f32_e32 v[107:108], v48                          // 000000006970: 7ed62130
	v_cmp_eq_f32_e64 s8, 0, v50                                // 000000006974: d4120008 02026480
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 00000000697c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006988: 8b080a08
	v_mul_f64_e32 v[94:95], v[94:95], v[96:97]                 // 00000000698c: 0cbcc15e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006990: bf870091
	v_mul_f64_e32 v[94:95], v[94:95], v[107:108]               // 000000006994: 0cbcd75e
	v_cvt_f32_f64_e32 v49, v[94:95]                            // 000000006998: 7e621f5e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000699c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069a0: bf870001
	v_cndmask_b32_e64 v105, v49, 0, s8                         // 0000000069a4: d5010069 00210131
	s_branch 62078                                             // 0000000069ac: bfa0f27e <packed_folded_w4a8+0x18a8>
	v_cvt_f64_f32_e32 v[49:50], v51                            // 0000000069b0: 7e622133
	v_cvt_f64_f32_e32 v[96:97], v106                           // 0000000069b4: 7ec0216a
	v_cvt_f64_f32_e32 v[107:108], v48                          // 0000000069b8: 7ed62130
	v_cmp_eq_f32_e64 s8, 0, v51                                // 0000000069bc: d4120008 02026680
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 0000000069c4: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 0000000069d0: 8b080a08
	v_mul_f64_e32 v[49:50], v[49:50], v[96:97]                 // 0000000069d4: 0c62c131
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069d8: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[107:108]               // 0000000069dc: 0c62d731
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 0000000069e0: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069e8: bf870001
	v_cndmask_b32_e64 v107, v49, 0, s8                         // 0000000069ec: d501006b 00210131
	s_branch 62093                                             // 0000000069f4: bfa0f28d <packed_folded_w4a8+0x192c>
	v_cvt_f64_f32_e32 v[96:97], v52                            // 0000000069f8: 7ec02134
	v_cvt_f64_f32_e32 v[108:109], v106                         // 0000000069fc: 7ed8216a
	v_cvt_f64_f32_e32 v[112:113], v48                          // 000000006a00: 7ee02130
	v_cmp_eq_f32_e64 s8, 0, v52                                // 000000006a04: d4120008 02026880
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006a0c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006a18: 8b080a08
	v_mul_f64_e32 v[96:97], v[96:97], v[108:109]               // 000000006a1c: 0cc0d960
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a20: bf870091
	v_mul_f64_e32 v[96:97], v[96:97], v[112:113]               // 000000006a24: 0cc0e160
	v_cvt_f32_f64_e32 v49, v[96:97]                            // 000000006a28: 7e621f60
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a2c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a30: bf870001
	v_cndmask_b32_e64 v108, v49, 0, s8                         // 000000006a34: d501006c 00210131
	s_branch 62108                                             // 000000006a3c: bfa0f29c <packed_folded_w4a8+0x19b0>
	v_cvt_f64_f32_e32 v[112:113], v53                          // 000000006a40: 7ee02135
	v_cvt_f64_f32_e32 v[120:121], v106                         // 000000006a44: 7ef0216a
	v_cvt_f64_f32_e32 v[122:123], v48                          // 000000006a48: 7ef42130
	v_cmp_eq_f32_e64 s8, 0, v53                                // 000000006a4c: d4120008 02026a80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006a54: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006a60: 8b080a08
	v_mul_f64_e32 v[112:113], v[112:113], v[120:121]           // 000000006a64: 0ce0f170
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a68: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 000000006a6c: 0ce0f570
	v_cvt_f32_f64_e32 v49, v[112:113]                          // 000000006a70: 7e621f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a74: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a78: bf870001
	v_cndmask_b32_e64 v109, v49, 0, s8                         // 000000006a7c: d501006d 00210131
	s_branch 62123                                             // 000000006a84: bfa0f2ab <packed_folded_w4a8+0x1a34>
	v_cvt_f64_f32_e32 v[112:113], v54                          // 000000006a88: 7ee02136
	v_cvt_f64_f32_e32 v[120:121], v106                         // 000000006a8c: 7ef0216a
	v_cvt_f64_f32_e32 v[122:123], v48                          // 000000006a90: 7ef42130
	v_cmp_eq_f32_e64 s8, 0, v54                                // 000000006a94: d4120008 02026c80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006a9c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006aa8: 8b080a08
	v_mul_f64_e32 v[112:113], v[112:113], v[120:121]           // 000000006aac: 0ce0f170
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ab0: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 000000006ab4: 0ce0f570
	v_cvt_f32_f64_e32 v49, v[112:113]                          // 000000006ab8: 7e621f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006abc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ac0: bf870001
	v_cndmask_b32_e64 v112, v49, 0, s8                         // 000000006ac4: d5010070 00210131
	s_branch 62138                                             // 000000006acc: bfa0f2ba <packed_folded_w4a8+0x1ab8>
	v_cvt_f64_f32_e32 v[120:121], v55                          // 000000006ad0: 7ef02137
	v_cvt_f64_f32_e32 v[122:123], v106                         // 000000006ad4: 7ef4216a
	v_cvt_f64_f32_e32 v[133:134], v48                          // 000000006ad8: 7f0a2130
	v_cmp_eq_f32_e64 s8, 0, v55                                // 000000006adc: d4120008 02026e80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006ae4: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006af0: 8b080a08
	v_mul_f64_e32 v[120:121], v[120:121], v[122:123]           // 000000006af4: 0cf0f578
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006af8: bf870091
	v_mul_f64_e32 v[120:121], v[120:121], v[133:134]           // 000000006afc: 0cf10b78
	v_cvt_f32_f64_e32 v49, v[120:121]                          // 000000006b00: 7e621f78
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b04: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b08: bf870001
	v_cndmask_b32_e64 v113, v49, 0, s8                         // 000000006b0c: d5010071 00210131
	s_branch 62152                                             // 000000006b14: bfa0f2c8 <packed_folded_w4a8+0x1b38>
	v_cvt_f64_f32_e32 v[101:102], v40                          // 000000006b18: 7eca2128
	v_cvt_f64_f32_e32 v[103:104], v123                         // 000000006b1c: 7ece217b
	v_cvt_f64_f32_e32 v[105:106], v100                         // 000000006b20: 7ed22164
	v_cmp_eq_f32_e64 s16, 0, v40                               // 000000006b24: d4120010 02025080
	v_cmp_class_f32_e64 s18, v100, 0x1f8                       // 000000006b2c: d47e0012 0201ff64 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006b38: 8b101210
	v_mul_f64_e32 v[101:102], v[101:102], v[103:104]           // 000000006b3c: 0ccacf65
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b40: bf870091
	v_mul_f64_e32 v[101:102], v[101:102], v[105:106]           // 000000006b44: 0ccad365
	v_cvt_f32_f64_e32 v101, v[101:102]                         // 000000006b48: 7eca1f65
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b4c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b50: bf870001
	v_cndmask_b32_e64 v120, v101, 0, s16                       // 000000006b54: d5010078 00410165
	s_branch 62479                                             // 000000006b5c: bfa0f40f <packed_folded_w4a8+0x209c>
	v_cvt_f64_f32_e32 v[102:103], v41                          // 000000006b60: 7ecc2129
	v_cvt_f64_f32_e32 v[104:105], v123                         // 000000006b64: 7ed0217b
	v_cvt_f64_f32_e32 v[106:107], v40                          // 000000006b68: 7ed42128
	v_cmp_eq_f32_e64 s16, 0, v41                               // 000000006b6c: d4120010 02025280
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006b74: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006b80: 8b101210
	v_mul_f64_e32 v[102:103], v[102:103], v[104:105]           // 000000006b84: 0cccd166
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b88: bf870091
	v_mul_f64_e32 v[102:103], v[102:103], v[106:107]           // 000000006b8c: 0cccd566
	v_cvt_f32_f64_e32 v102, v[102:103]                         // 000000006b90: 7ecc1f66
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b94: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b98: bf870001
	v_cndmask_b32_e64 v121, v102, 0, s16                       // 000000006b9c: d5010079 00410166
	s_branch 62494                                             // 000000006ba4: bfa0f41e <packed_folded_w4a8+0x2120>
	v_cvt_f64_f32_e32 v[104:105], v42                          // 000000006ba8: 7ed0212a
	v_cvt_f64_f32_e32 v[106:107], v123                         // 000000006bac: 7ed4217b
	v_cvt_f64_f32_e32 v[133:134], v40                          // 000000006bb0: 7f0a2128
	v_cmp_eq_f32_e64 s16, 0, v42                               // 000000006bb4: d4120010 02025480
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006bbc: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006bc8: 8b101210
	v_mul_f64_e32 v[104:105], v[104:105], v[106:107]           // 000000006bcc: 0cd0d568
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bd0: bf870091
	v_mul_f64_e32 v[104:105], v[104:105], v[133:134]           // 000000006bd4: 0cd10b68
	v_cvt_f32_f64_e32 v41, v[104:105]                          // 000000006bd8: 7e521f68
	s_wait_alu depctr_sa_sdst(0)                               // 000000006bdc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006be0: bf870001
	v_cndmask_b32_e64 v122, v41, 0, s16                        // 000000006be4: d501007a 00410129
	s_branch 62509                                             // 000000006bec: bfa0f42d <packed_folded_w4a8+0x21a4>
	v_cvt_f64_f32_e32 v[41:42], v43                            // 000000006bf0: 7e52212b
	v_cvt_f64_f32_e32 v[106:107], v123                         // 000000006bf4: 7ed4217b
	v_cvt_f64_f32_e32 v[133:134], v40                          // 000000006bf8: 7f0a2128
	v_cmp_eq_f32_e64 s16, 0, v43                               // 000000006bfc: d4120010 02025680
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006c04: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006c10: 8b101210
	v_mul_f64_e32 v[41:42], v[41:42], v[106:107]               // 000000006c14: 0c52d529
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c18: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[133:134]               // 000000006c1c: 0c530b29
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000006c20: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c24: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c28: bf870001
	v_cndmask_b32_e64 v133, v41, 0, s16                        // 000000006c2c: d5010085 00410129
	s_branch 62524                                             // 000000006c34: bfa0f43c <packed_folded_w4a8+0x2228>
	v_cvt_f64_f32_e32 v[106:107], v44                          // 000000006c38: 7ed4212c
	v_cvt_f64_f32_e32 v[134:135], v123                         // 000000006c3c: 7f0c217b
	v_cvt_f64_f32_e32 v[136:137], v40                          // 000000006c40: 7f102128
	v_cmp_eq_f32_e64 s16, 0, v44                               // 000000006c44: d4120010 02025880
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006c4c: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006c58: 8b101210
	v_mul_f64_e32 v[106:107], v[106:107], v[134:135]           // 000000006c5c: 0cd50d6a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c60: bf870091
	v_mul_f64_e32 v[106:107], v[106:107], v[136:137]           // 000000006c64: 0cd5116a
	v_cvt_f32_f64_e32 v41, v[106:107]                          // 000000006c68: 7e521f6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c6c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c70: bf870001
	v_cndmask_b32_e64 v134, v41, 0, s16                        // 000000006c74: d5010086 00410129
	s_branch 62539                                             // 000000006c7c: bfa0f44b <packed_folded_w4a8+0x22ac>
	v_cvt_f64_f32_e32 v[135:136], v45                          // 000000006c80: 7f0e212d
	v_cvt_f64_f32_e32 v[137:138], v123                         // 000000006c84: 7f12217b
	v_cvt_f64_f32_e32 v[139:140], v40                          // 000000006c88: 7f162128
	v_cmp_eq_f32_e64 s16, 0, v45                               // 000000006c8c: d4120010 02025a80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006c94: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006ca0: 8b101210
	v_mul_f64_e32 v[135:136], v[135:136], v[137:138]           // 000000006ca4: 0d0f1387
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ca8: bf870091
	v_mul_f64_e32 v[135:136], v[135:136], v[139:140]           // 000000006cac: 0d0f1787
	v_cvt_f32_f64_e32 v41, v[135:136]                          // 000000006cb0: 7e521f87
	s_wait_alu depctr_sa_sdst(0)                               // 000000006cb4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006cb8: bf870001
	v_cndmask_b32_e64 v135, v41, 0, s16                        // 000000006cbc: d5010087 00410129
	s_branch 62554                                             // 000000006cc4: bfa0f45a <packed_folded_w4a8+0x2330>
	v_cvt_f64_f32_e32 v[136:137], v46                          // 000000006cc8: 7f10212e
	v_cvt_f64_f32_e32 v[138:139], v123                         // 000000006ccc: 7f14217b
	v_cvt_f64_f32_e32 v[140:141], v40                          // 000000006cd0: 7f182128
	v_cmp_eq_f32_e64 s16, 0, v46                               // 000000006cd4: d4120010 02025c80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006cdc: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006ce8: 8b101210
	v_mul_f64_e32 v[136:137], v[136:137], v[138:139]           // 000000006cec: 0d111588
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006cf0: bf870091
	v_mul_f64_e32 v[136:137], v[136:137], v[140:141]           // 000000006cf4: 0d111988
	v_cvt_f32_f64_e32 v41, v[136:137]                          // 000000006cf8: 7e521f88
	s_wait_alu depctr_sa_sdst(0)                               // 000000006cfc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d00: bf870001
	v_cndmask_b32_e64 v136, v41, 0, s16                        // 000000006d04: d5010088 00410129
	s_branch 62569                                             // 000000006d0c: bfa0f469 <packed_folded_w4a8+0x23b4>
	v_cvt_f64_f32_e32 v[137:138], v47                          // 000000006d10: 7f12212f
	v_cvt_f64_f32_e32 v[139:140], v123                         // 000000006d14: 7f16217b
	v_cvt_f64_f32_e32 v[141:142], v40                          // 000000006d18: 7f1a2128
	v_cmp_eq_f32_e64 s16, 0, v47                               // 000000006d1c: d4120010 02025e80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006d24: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006d30: 8b101210
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006d34: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d38: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006d3c: 0d131b89
	v_cvt_f32_f64_e32 v41, v[137:138]                          // 000000006d40: 7e521f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d44: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d48: bf870001
	v_cndmask_b32_e64 v137, v41, 0, s16                        // 000000006d4c: d5010089 00410129
	s_branch 62583                                             // 000000006d54: bfa0f477 <packed_folded_w4a8+0x2434>
	v_cvt_f64_f32_e32 v[111:112], v32                          // 000000006d58: 7ede2120
	v_cvt_f64_f32_e32 v[113:114], v136                         // 000000006d5c: 7ee22188
	v_cvt_f64_f32_e32 v[133:134], v110                         // 000000006d60: 7f0a216e
	v_cmp_eq_f32_e64 s24, 0, v32                               // 000000006d64: d4120018 02024080
	v_cmp_class_f32_e64 s26, v110, 0x1f8                       // 000000006d6c: d47e001a 0201ff6e 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006d78: 8b181a18
	v_mul_f64_e32 v[111:112], v[111:112], v[113:114]           // 000000006d7c: 0cdee36f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d80: bf870091
	v_mul_f64_e32 v[111:112], v[111:112], v[133:134]           // 000000006d84: 0cdf0b6f
	v_cvt_f32_f64_e32 v111, v[111:112]                         // 000000006d88: 7ede1f6f
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d8c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d90: bf870001
	v_cndmask_b32_e64 v133, v111, 0, s24                       // 000000006d94: d5010085 0061016f
	s_branch 62903                                             // 000000006d9c: bfa0f5b7 <packed_folded_w4a8+0x297c>
	v_cvt_f64_f32_e32 v[112:113], v33                          // 000000006da0: 7ee02121
	v_cvt_f64_f32_e32 v[114:115], v136                         // 000000006da4: 7ee42188
	v_cvt_f64_f32_e32 v[134:135], v32                          // 000000006da8: 7f0c2120
	v_cmp_eq_f32_e64 s24, 0, v33                               // 000000006dac: d4120018 02024280
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006db4: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006dc0: 8b181a18
	v_mul_f64_e32 v[112:113], v[112:113], v[114:115]           // 000000006dc4: 0ce0e570
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006dc8: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[134:135]           // 000000006dcc: 0ce10d70
	v_cvt_f32_f64_e32 v112, v[112:113]                         // 000000006dd0: 7ee01f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006dd4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006dd8: bf870001
	v_cndmask_b32_e64 v134, v112, 0, s24                       // 000000006ddc: d5010086 00610170
	s_branch 62918                                             // 000000006de4: bfa0f5c6 <packed_folded_w4a8+0x2a00>
	v_cvt_f64_f32_e32 v[114:115], v34                          // 000000006de8: 7ee42122
	v_cvt_f64_f32_e32 v[116:117], v136                         // 000000006dec: 7ee82188
	v_cvt_f64_f32_e32 v[137:138], v32                          // 000000006df0: 7f122120
	v_cmp_eq_f32_e64 s24, 0, v34                               // 000000006df4: d4120018 02024480
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006dfc: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006e08: 8b181a18
	v_mul_f64_e32 v[114:115], v[114:115], v[116:117]           // 000000006e0c: 0ce4e972
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e10: bf870091
	v_mul_f64_e32 v[114:115], v[114:115], v[137:138]           // 000000006e14: 0ce51372
	v_cvt_f32_f64_e32 v33, v[114:115]                          // 000000006e18: 7e421f72
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e1c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e20: bf870001
	v_cndmask_b32_e64 v135, v33, 0, s24                        // 000000006e24: d5010087 00610121
	s_branch 62933                                             // 000000006e2c: bfa0f5d5 <packed_folded_w4a8+0x2a84>
	v_cvt_f64_f32_e32 v[33:34], v35                            // 000000006e30: 7e422123
	v_cvt_f64_f32_e32 v[116:117], v136                         // 000000006e34: 7ee82188
	v_cvt_f64_f32_e32 v[137:138], v32                          // 000000006e38: 7f122120
	v_cmp_eq_f32_e64 s24, 0, v35                               // 000000006e3c: d4120018 02024680
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006e44: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006e50: 8b181a18
	v_mul_f64_e32 v[33:34], v[33:34], v[116:117]               // 000000006e54: 0c42e921
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e58: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[137:138]               // 000000006e5c: 0c431321
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000006e60: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e64: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e68: bf870001
	v_cndmask_b32_e64 v128, v33, 0, s24                        // 000000006e6c: d5010080 00610121
	s_branch 62948                                             // 000000006e74: bfa0f5e4 <packed_folded_w4a8+0x2b08>
	v_cvt_f64_f32_e32 v[116:117], v36                          // 000000006e78: 7ee82124
	v_cvt_f64_f32_e32 v[137:138], v136                         // 000000006e7c: 7f122188
	v_cvt_f64_f32_e32 v[139:140], v32                          // 000000006e80: 7f162120
	v_cmp_eq_f32_e64 s24, 0, v36                               // 000000006e84: d4120018 02024880
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006e8c: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006e98: 8b181a18
	v_mul_f64_e32 v[116:117], v[116:117], v[137:138]           // 000000006e9c: 0ce91374
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ea0: bf870091
	v_mul_f64_e32 v[116:117], v[116:117], v[139:140]           // 000000006ea4: 0ce91774
	v_cvt_f32_f64_e32 v33, v[116:117]                          // 000000006ea8: 7e421f74
	s_wait_alu depctr_sa_sdst(0)                               // 000000006eac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006eb0: bf870001
	v_cndmask_b32_e64 v129, v33, 0, s24                        // 000000006eb4: d5010081 00610121
	s_branch 62963                                             // 000000006ebc: bfa0f5f3 <packed_folded_w4a8+0x2b8c>
	v_cvt_f64_f32_e32 v[137:138], v37                          // 000000006ec0: 7f122125
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006ec4: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006ec8: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v37                               // 000000006ecc: d4120018 02024a80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006ed4: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006ee0: 8b181a18
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006ee4: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ee8: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006eec: 0d131b89
	v_cvt_f32_f64_e32 v33, v[137:138]                          // 000000006ef0: 7e421f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ef4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ef8: bf870001
	v_cndmask_b32_e64 v130, v33, 0, s24                        // 000000006efc: d5010082 00610121
	s_branch 62978                                             // 000000006f04: bfa0f602 <packed_folded_w4a8+0x2c10>
	v_cvt_f64_f32_e32 v[137:138], v38                          // 000000006f08: 7f122126
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006f0c: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006f10: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v38                               // 000000006f14: d4120018 02024c80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006f1c: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006f28: 8b181a18
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006f2c: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f30: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006f34: 0d131b89
	v_cvt_f32_f64_e32 v33, v[137:138]                          // 000000006f38: 7e421f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f3c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f40: bf870001
	v_cndmask_b32_e64 v131, v33, 0, s24                        // 000000006f44: d5010083 00610121
	s_branch 62993                                             // 000000006f4c: bfa0f611 <packed_folded_w4a8+0x2c94>
	v_cvt_f64_f32_e32 v[137:138], v39                          // 000000006f50: 7f122127
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006f54: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006f58: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v39                               // 000000006f5c: d4120018 02024e80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006f64: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006f70: 8b181a18
	v_mul_f64_e32 v[136:137], v[137:138], v[139:140]           // 000000006f74: 0d111789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f78: bf870091
	v_mul_f64_e32 v[136:137], v[136:137], v[141:142]           // 000000006f7c: 0d111b88
	v_cvt_f32_f64_e32 v33, v[136:137]                          // 000000006f80: 7e421f88
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f84: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f88: bf870001
	v_cndmask_b32_e64 v132, v33, 0, s24                        // 000000006f8c: d5010084 00610121
	s_branch 63007                                             // 000000006f94: bfa0f61f <packed_folded_w4a8+0x2d14>
	v_cvt_f64_f32_e32 v[122:123], v24                          // 000000006f98: 7ef42118
	v_cvt_f64_f32_e32 v[127:128], v73                          // 000000006f9c: 7efe2149
	v_cvt_f64_f32_e32 v[129:130], v118                         // 000000006fa0: 7f022176
	v_cmp_eq_f32_e64 s33, 0, v24                               // 000000006fa4: d4120021 02023080
	v_cmp_class_f32_e64 s37, v118, 0x1f8                       // 000000006fac: d47e0025 0201ff76 000001f8
	s_and_b32 s33, s33, s37                                    // 000000006fb8: 8b212521
	v_mul_f64_e32 v[122:123], v[122:123], v[127:128]           // 000000006fbc: 0cf4ff7a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006fc0: bf870091
	v_mul_f64_e32 v[122:123], v[122:123], v[129:130]           // 000000006fc4: 0cf5037a
	v_cvt_f32_f64_e32 v72, v[122:123]                          // 000000006fc8: 7e901f7a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006fcc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006fd0: bf870001
	v_cndmask_b32_e64 v72, v72, 0, s33                         // 000000006fd4: d5010048 00850148
	s_branch 63319                                             // 000000006fdc: bfa0f757 <packed_folded_w4a8+0x323c>
	v_cvt_f64_f32_e32 v[118:119], v25                          // 000000006fe0: 7eec2119
	v_cvt_f64_f32_e32 v[122:123], v73                          // 000000006fe4: 7ef42149
	v_cvt_f64_f32_e32 v[127:128], v74                          // 000000006fe8: 7efe214a
	v_cmp_eq_f32_e64 s33, 0, v25                               // 000000006fec: d4120021 02023280
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000006ff4: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007000: 8b212521
	v_mul_f64_e32 v[118:119], v[118:119], v[122:123]           // 000000007004: 0cecf576
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007008: bf870091
	v_mul_f64_e32 v[118:119], v[118:119], v[127:128]           // 00000000700c: 0cecff76
	v_cvt_f32_f64_e32 v24, v[118:119]                          // 000000007010: 7e301f76
	s_wait_alu depctr_sa_sdst(0)                               // 000000007014: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007018: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s33                         // 00000000701c: d5010018 00850118
	s_branch 63317                                             // 000000007024: bfa0f755 <packed_folded_w4a8+0x327c>
	v_cvt_f64_f32_e32 v[75:76], v26                            // 000000007028: 7e96211a
	v_cvt_f64_f32_e32 v[118:119], v73                          // 00000000702c: 7eec2149
	v_cvt_f64_f32_e32 v[122:123], v74                          // 000000007030: 7ef4214a
	v_cmp_eq_f32_e64 s33, 0, v26                               // 000000007034: d4120021 02023480
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 00000000703c: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007048: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[118:119]               // 00000000704c: 0c96ed4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007050: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[122:123]               // 000000007054: 0c96f54b
	v_cvt_f32_f64_e32 v25, v[75:76]                            // 000000007058: 7e321f4b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000705c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007060: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s33                         // 000000007064: d5010019 00850119
	s_branch 63315                                             // 00000000706c: bfa0f753 <packed_folded_w4a8+0x32bc>
	v_cvt_f64_f32_e32 v[75:76], v27                            // 000000007070: 7e96211b
	v_cvt_f64_f32_e32 v[77:78], v73                            // 000000007074: 7e9a2149
	v_cvt_f64_f32_e32 v[118:119], v74                          // 000000007078: 7eec214a
	v_cmp_eq_f32_e64 s33, 0, v27                               // 00000000707c: d4120021 02023680
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000007084: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007090: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 000000007094: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007098: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[118:119]               // 00000000709c: 0c96ed4b
	v_cvt_f32_f64_e32 v26, v[75:76]                            // 0000000070a0: 7e341f4b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000070a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000070a8: bf870001
	v_cndmask_b32_e64 v26, v26, 0, s33                         // 0000000070ac: d501001a 0085011a
	s_branch 63313                                             // 0000000070b4: bfa0f751 <packed_folded_w4a8+0x32fc>
	v_cvt_f64_f32_e32 v[75:76], v28                            // 0000000070b8: 7e96211c
	v_cvt_f64_f32_e32 v[77:78], v73                            // 0000000070bc: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 0000000070c0: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v28                               // 0000000070c4: d4120021 02023880
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 0000000070cc: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 0000000070d8: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 0000000070dc: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000070e0: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 0000000070e4: 0c969f4b
	v_cvt_f32_f64_e32 v27, v[75:76]                            // 0000000070e8: 7e361f4b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000070ec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000070f0: bf870001
	v_cndmask_b32_e64 v27, v27, 0, s33                         // 0000000070f4: d501001b 0085011b
	s_branch 63311                                             // 0000000070fc: bfa0f74f <packed_folded_w4a8+0x333c>
	v_cvt_f64_f32_e32 v[75:76], v29                            // 000000007100: 7e96211d
	v_cvt_f64_f32_e32 v[77:78], v73                            // 000000007104: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007108: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v29                               // 00000000710c: d4120021 02023a80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000007114: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007120: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 000000007124: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007128: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 00000000712c: 0c969f4b
	v_cvt_f32_f64_e32 v28, v[75:76]                            // 000000007130: 7e381f4b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007134: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007138: bf870001
	v_cndmask_b32_e64 v28, v28, 0, s33                         // 00000000713c: d501001c 0085011c
	s_branch 63309                                             // 000000007144: bfa0f74d <packed_folded_w4a8+0x337c>
	v_cvt_f64_f32_e32 v[75:76], v30                            // 000000007148: 7e96211e
	v_cvt_f64_f32_e32 v[77:78], v73                            // 00000000714c: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007150: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v30                               // 000000007154: d4120021 02023c80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 00000000715c: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007168: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 00000000716c: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007170: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 000000007174: 0c969f4b
	v_cvt_f32_f64_e32 v29, v[75:76]                            // 000000007178: 7e3a1f4b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000717c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007180: bf870001
	v_cndmask_b32_e64 v29, v29, 0, s33                         // 000000007184: d501001d 0085011d
	s_branch 63307                                             // 00000000718c: bfa0f74b <packed_folded_w4a8+0x33bc>
	v_cvt_f64_f32_e32 v[75:76], v31                            // 000000007190: 7e96211f
	v_cvt_f64_f32_e32 v[77:78], v73                            // 000000007194: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007198: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v31                               // 00000000719c: d4120021 02023e80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 0000000071a4: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 0000000071b0: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 0000000071b4: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000071b8: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 0000000071bc: 0c969f4b
	v_cvt_f32_f64_e32 v30, v[75:76]                            // 0000000071c0: 7e3c1f4b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000071c8: bf870001
	v_cndmask_b32_e64 v30, v30, 0, s33                         // 0000000071cc: d501001e 0085011e
	s_branch 63305                                             // 0000000071d4: bfa0f749 <packed_folded_w4a8+0x33fc>
	v_cvt_f64_f32_e32 v[27:28], v16                            // 0000000071d8: 7e362110
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000071dc: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 0000000071e0: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v16                                // 0000000071e4: d4120000 02022080
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000071ec: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000071f8: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000071fc: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007200: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 000000007204: 0c368d1b
	v_cvt_f32_f64_e32 v24, v[27:28]                            // 000000007208: 7e301f1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000720c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007210: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s0                          // 000000007214: d5010018 00010118
	s_branch 63586                                             // 00000000721c: bfa0f862 <packed_folded_w4a8+0x38a8>
	v_cvt_f64_f32_e32 v[27:28], v17                            // 000000007220: 7e362111
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000007224: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 000000007228: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v17                                // 00000000722c: d4120000 02022280
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 000000007234: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007240: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007244: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007248: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 00000000724c: 0c368d1b
	v_cvt_f32_f64_e32 v16, v[27:28]                            // 000000007250: 7e201f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007254: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007258: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s0                          // 00000000725c: d5010010 00010110
	s_branch 63584                                             // 000000007264: bfa0f860 <packed_folded_w4a8+0x38e8>
	v_cvt_f64_f32_e32 v[27:28], v18                            // 000000007268: 7e362112
	v_cvt_f64_f32_e32 v[29:30], v25                            // 00000000726c: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 000000007270: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v18                                // 000000007274: d4120000 02022480
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000727c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007288: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 00000000728c: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007290: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 000000007294: 0c368d1b
	v_cvt_f32_f64_e32 v17, v[27:28]                            // 000000007298: 7e221f1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000729c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000072a0: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s0                          // 0000000072a4: d5010011 00010111
	s_branch 63582                                             // 0000000072ac: bfa0f85e <packed_folded_w4a8+0x3928>
	v_cvt_f64_f32_e32 v[27:28], v19                            // 0000000072b0: 7e362113
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000072b4: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 0000000072b8: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v19                                // 0000000072bc: d4120000 02022680
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000072c4: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000072d0: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000072d4: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000072d8: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 0000000072dc: 0c368d1b
	v_cvt_f32_f64_e32 v18, v[27:28]                            // 0000000072e0: 7e241f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000072e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000072e8: bf870001
	v_cndmask_b32_e64 v18, v18, 0, s0                          // 0000000072ec: d5010012 00010112
	s_branch 63580                                             // 0000000072f4: bfa0f85c <packed_folded_w4a8+0x3968>
	v_cvt_f64_f32_e32 v[27:28], v20                            // 0000000072f8: 7e362114
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000072fc: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 000000007300: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v20                                // 000000007304: d4120000 02022880
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000730c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007318: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 00000000731c: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007320: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 000000007324: 0c36651b
	v_cvt_f32_f64_e32 v19, v[27:28]                            // 000000007328: 7e261f1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000732c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007330: bf870001
	v_cndmask_b32_e64 v19, v19, 0, s0                          // 000000007334: d5010013 00010113
	s_branch 63578                                             // 00000000733c: bfa0f85a <packed_folded_w4a8+0x39a8>
	v_cvt_f64_f32_e32 v[27:28], v21                            // 000000007340: 7e362115
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000007344: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 000000007348: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v21                                // 00000000734c: d4120000 02022a80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 000000007354: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007360: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007364: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007368: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 00000000736c: 0c36651b
	v_cvt_f32_f64_e32 v20, v[27:28]                            // 000000007370: 7e281f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007374: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007378: bf870001
	v_cndmask_b32_e64 v20, v20, 0, s0                          // 00000000737c: d5010014 00010114
	s_branch 63576                                             // 000000007384: bfa0f858 <packed_folded_w4a8+0x39e8>
	v_cvt_f64_f32_e32 v[27:28], v22                            // 000000007388: 7e362116
	v_cvt_f64_f32_e32 v[29:30], v25                            // 00000000738c: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 000000007390: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v22                                // 000000007394: d4120000 02022c80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000739c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000073a8: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000073ac: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000073b0: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 0000000073b4: 0c36651b
	v_cvt_f32_f64_e32 v21, v[27:28]                            // 0000000073b8: 7e2a1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000073c0: bf870001
	v_cndmask_b32_e64 v21, v21, 0, s0                          // 0000000073c4: d5010015 00010115
	s_branch 63574                                             // 0000000073cc: bfa0f856 <packed_folded_w4a8+0x3a28>
	v_cvt_f64_f32_e32 v[27:28], v23                            // 0000000073d0: 7e362117
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000073d4: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 0000000073d8: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v23                                // 0000000073dc: d4120000 02022e80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000073e4: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000073f0: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000073f4: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000073f8: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 0000000073fc: 0c36651b
	v_cvt_f32_f64_e32 v22, v[27:28]                            // 000000007400: 7e2c1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007404: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007408: bf870001
	v_cndmask_b32_e64 v22, v22, 0, s0                          // 00000000740c: d5010016 00010116
	s_branch 63572                                             // 000000007414: bfa0f854 <packed_folded_w4a8+0x3a68>
	v_cvt_f64_f32_e32 v[19:20], v8                             // 000000007418: 7e262108
	v_cvt_f64_f32_e32 v[21:22], v17                            // 00000000741c: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007420: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v8                                 // 000000007424: d4120000 02021080
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 00000000742c: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007438: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000743c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007440: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007444: 0c262f13
	v_cvt_f32_f64_e32 v16, v[19:20]                            // 000000007448: 7e201f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000744c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007450: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s0                          // 000000007454: d5010010 00010110
	s_branch 63853                                             // 00000000745c: bfa0f96d <packed_folded_w4a8+0x3f14>
	v_cvt_f64_f32_e32 v[19:20], v9                             // 000000007460: 7e262109
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000007464: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007468: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v9                                 // 00000000746c: d4120000 02021280
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007474: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007480: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007484: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007488: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000748c: 0c262f13
	v_cvt_f32_f64_e32 v8, v[19:20]                             // 000000007490: 7e101f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007494: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007498: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s0                            // 00000000749c: d5010008 00010108
	s_branch 63851                                             // 0000000074a4: bfa0f96b <packed_folded_w4a8+0x3f54>
	v_cvt_f64_f32_e32 v[19:20], v10                            // 0000000074a8: 7e26210a
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000074ac: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000074b0: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v10                                // 0000000074b4: d4120000 02021480
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 0000000074bc: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000074c8: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000074cc: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000074d0: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000074d4: 0c262f13
	v_cvt_f32_f64_e32 v9, v[19:20]                             // 0000000074d8: 7e121f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000074e0: bf870001
	v_cndmask_b32_e64 v9, v9, 0, s0                            // 0000000074e4: d5010009 00010109
	s_branch 63849                                             // 0000000074ec: bfa0f969 <packed_folded_w4a8+0x3f94>
	v_cvt_f64_f32_e32 v[19:20], v11                            // 0000000074f0: 7e26210b
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000074f4: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000074f8: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v11                                // 0000000074fc: d4120000 02021680
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007504: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007510: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007514: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007518: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000751c: 0c262f13
	v_cvt_f32_f64_e32 v10, v[19:20]                            // 000000007520: 7e141f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007524: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007528: bf870001
	v_cndmask_b32_e64 v10, v10, 0, s0                          // 00000000752c: d501000a 0001010a
	s_branch 63847                                             // 000000007534: bfa0f967 <packed_folded_w4a8+0x3fd4>
	v_cvt_f64_f32_e32 v[19:20], v12                            // 000000007538: 7e26210c
	v_cvt_f64_f32_e32 v[21:22], v17                            // 00000000753c: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007540: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v12                                // 000000007544: d4120000 02021880
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 00000000754c: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007558: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000755c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007560: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007564: 0c262f13
	v_cvt_f32_f64_e32 v11, v[19:20]                            // 000000007568: 7e161f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000756c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007570: bf870001
	v_cndmask_b32_e64 v11, v11, 0, s0                          // 000000007574: d501000b 0001010b
	s_branch 63845                                             // 00000000757c: bfa0f965 <packed_folded_w4a8+0x4014>
	v_cvt_f64_f32_e32 v[19:20], v13                            // 000000007580: 7e26210d
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000007584: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007588: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v13                                // 00000000758c: d4120000 02021a80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007594: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000075a0: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000075a4: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000075a8: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000075ac: 0c262f13
	v_cvt_f32_f64_e32 v12, v[19:20]                            // 0000000075b0: 7e181f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000075b8: bf870001
	v_cndmask_b32_e64 v12, v12, 0, s0                          // 0000000075bc: d501000c 0001010c
	s_branch 63843                                             // 0000000075c4: bfa0f963 <packed_folded_w4a8+0x4054>
	v_cvt_f64_f32_e32 v[19:20], v14                            // 0000000075c8: 7e26210e
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000075cc: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000075d0: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v14                                // 0000000075d4: d4120000 02021c80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 0000000075dc: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000075e8: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000075ec: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000075f0: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000075f4: 0c262f13
	v_cvt_f32_f64_e32 v13, v[19:20]                            // 0000000075f8: 7e1a1f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075fc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007600: bf870001
	v_cndmask_b32_e64 v13, v13, 0, s0                          // 000000007604: d501000d 0001010d
	s_branch 63841                                             // 00000000760c: bfa0f961 <packed_folded_w4a8+0x4094>
	v_cvt_f64_f32_e32 v[19:20], v15                            // 000000007610: 7e26210f
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000007614: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007618: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v15                                // 00000000761c: d4120000 02021e80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007624: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007630: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007634: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007638: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000763c: 0c262f13
	v_cvt_f32_f64_e32 v14, v[19:20]                            // 000000007640: 7e1c1f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007644: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007648: bf870001
	v_cndmask_b32_e64 v14, v14, 0, s0                          // 00000000764c: d501000e 0001010e
	s_branch 63839                                             // 000000007654: bfa0f95f <packed_folded_w4a8+0x40d4>
	v_cvt_f64_f32_e32 v[11:12], v0                             // 000000007658: 7e162100
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000765c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007660: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v0                                 // 000000007664: d4120000 02020080
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000766c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007678: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000767c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007680: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000007684: 0c161f0b
	v_cvt_f32_f64_e32 v8, v[11:12]                             // 000000007688: 7e101f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000768c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007690: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s0                            // 000000007694: d5010008 00010108
	s_branch 64120                                             // 00000000769c: bfa0fa78 <packed_folded_w4a8+0x4580>
	v_cvt_f64_f32_e32 v[11:12], v1                             // 0000000076a0: 7e162101
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000076a4: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000076a8: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v1                                 // 0000000076ac: d4120000 02020280
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000076b4: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000076c0: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000076c4: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000076c8: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000076cc: 0c161f0b
	v_cvt_f32_f64_e32 v0, v[11:12]                             // 0000000076d0: 7e001f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000076d8: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s0                            // 0000000076dc: d5010000 00010100
	s_branch 64118                                             // 0000000076e4: bfa0fa76 <packed_folded_w4a8+0x45c0>
	v_cvt_f64_f32_e32 v[11:12], v2                             // 0000000076e8: 7e162102
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000076ec: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000076f0: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v2                                 // 0000000076f4: d4120000 02020480
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000076fc: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007708: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000770c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007710: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000007714: 0c161f0b
	v_cvt_f32_f64_e32 v1, v[11:12]                             // 000000007718: 7e021f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000771c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007720: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s0                            // 000000007724: d5010001 00010101
	s_branch 64116                                             // 00000000772c: bfa0fa74 <packed_folded_w4a8+0x4600>
	v_cvt_f64_f32_e32 v[11:12], v3                             // 000000007730: 7e162103
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000007734: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007738: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v3                                 // 00000000773c: d4120000 02020680
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 000000007744: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007750: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000007754: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007758: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 00000000775c: 0c161f0b
	v_cvt_f32_f64_e32 v2, v[11:12]                             // 000000007760: 7e041f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007764: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007768: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s0                            // 00000000776c: d5010002 00010102
	s_branch 64114                                             // 000000007774: bfa0fa72 <packed_folded_w4a8+0x4640>
	v_cvt_f64_f32_e32 v[11:12], v4                             // 000000007778: 7e162104
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000777c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007780: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v4                                 // 000000007784: d4120000 02020880
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000778c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007798: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000779c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000077a0: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000077a4: 0c161f0b
	v_cvt_f32_f64_e32 v3, v[11:12]                             // 0000000077a8: 7e061f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000077ac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000077b0: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s0                            // 0000000077b4: d5010003 00010103
	s_branch 64112                                             // 0000000077bc: bfa0fa70 <packed_folded_w4a8+0x4680>
	v_cvt_f64_f32_e32 v[11:12], v5                             // 0000000077c0: 7e162105
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000077c4: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000077c8: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v5                                 // 0000000077cc: d4120000 02020a80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000077d4: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000077e0: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000077e4: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000077e8: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000077ec: 0c161f0b
	v_cvt_f32_f64_e32 v4, v[11:12]                             // 0000000077f0: 7e081f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000077f4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000077f8: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s0                            // 0000000077fc: d5010004 00010104
	s_branch 64110                                             // 000000007804: bfa0fa6e <packed_folded_w4a8+0x46c0>
	v_cvt_f64_f32_e32 v[11:12], v6                             // 000000007808: 7e162106
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000780c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007810: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v6                                 // 000000007814: d4120000 02020c80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000781c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007828: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000782c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007830: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000007834: 0c161f0b
	v_cvt_f32_f64_e32 v5, v[11:12]                             // 000000007838: 7e0a1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000783c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007840: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s0                            // 000000007844: d5010005 00010105
	s_branch 64108                                             // 00000000784c: bfa0fa6c <packed_folded_w4a8+0x4700>
	v_cvt_f64_f32_e32 v[11:12], v7                             // 000000007850: 7e162107
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000007854: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007858: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v7                                 // 00000000785c: d4120000 02020e80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 000000007864: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007870: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000007874: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007878: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 00000000787c: 0c161f0b
	v_cvt_f32_f64_e32 v6, v[11:12]                             // 000000007880: 7e0c1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007884: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007888: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s0                            // 00000000788c: d5010006 00010106
	s_branch 64106                                             // 000000007894: bfa0fa6a <packed_folded_w4a8+0x4740>
	s_code_end                                                 // 000000007898: bf9f0000
	s_code_end                                                 // 00000000789c: bf9f0000
	s_code_end                                                 // 0000000078a0: bf9f0000
	s_code_end                                                 // 0000000078a4: bf9f0000
	s_code_end                                                 // 0000000078a8: bf9f0000
	s_code_end                                                 // 0000000078ac: bf9f0000
	s_code_end                                                 // 0000000078b0: bf9f0000
	s_code_end                                                 // 0000000078b4: bf9f0000
	s_code_end                                                 // 0000000078b8: bf9f0000
	s_code_end                                                 // 0000000078bc: bf9f0000
	s_code_end                                                 // 0000000078c0: bf9f0000
	s_code_end                                                 // 0000000078c4: bf9f0000
	s_code_end                                                 // 0000000078c8: bf9f0000
	s_code_end                                                 // 0000000078cc: bf9f0000
	s_code_end                                                 // 0000000078d0: bf9f0000
	s_code_end                                                 // 0000000078d4: bf9f0000
	s_code_end                                                 // 0000000078d8: bf9f0000
	s_code_end                                                 // 0000000078dc: bf9f0000
	s_code_end                                                 // 0000000078e0: bf9f0000
	s_code_end                                                 // 0000000078e4: bf9f0000
	s_code_end                                                 // 0000000078e8: bf9f0000
	s_code_end                                                 // 0000000078ec: bf9f0000
	s_code_end                                                 // 0000000078f0: bf9f0000
	s_code_end                                                 // 0000000078f4: bf9f0000
	s_code_end                                                 // 0000000078f8: bf9f0000
	s_code_end                                                 // 0000000078fc: bf9f0000
	s_code_end                                                 // 000000007900: bf9f0000
	s_code_end                                                 // 000000007904: bf9f0000
	s_code_end                                                 // 000000007908: bf9f0000
	s_code_end                                                 // 00000000790c: bf9f0000
	s_code_end                                                 // 000000007910: bf9f0000
	s_code_end                                                 // 000000007914: bf9f0000
	s_code_end                                                 // 000000007918: bf9f0000
	s_code_end                                                 // 00000000791c: bf9f0000
	s_code_end                                                 // 000000007920: bf9f0000
	s_code_end                                                 // 000000007924: bf9f0000
	s_code_end                                                 // 000000007928: bf9f0000
	s_code_end                                                 // 00000000792c: bf9f0000
	s_code_end                                                 // 000000007930: bf9f0000
	s_code_end                                                 // 000000007934: bf9f0000
	s_code_end                                                 // 000000007938: bf9f0000
	s_code_end                                                 // 00000000793c: bf9f0000
	s_code_end                                                 // 000000007940: bf9f0000
	s_code_end                                                 // 000000007944: bf9f0000
	s_code_end                                                 // 000000007948: bf9f0000
	s_code_end                                                 // 00000000794c: bf9f0000
	s_code_end                                                 // 000000007950: bf9f0000
	s_code_end                                                 // 000000007954: bf9f0000
	s_code_end                                                 // 000000007958: bf9f0000
	s_code_end                                                 // 00000000795c: bf9f0000
	s_code_end                                                 // 000000007960: bf9f0000
	s_code_end                                                 // 000000007964: bf9f0000
	s_code_end                                                 // 000000007968: bf9f0000
	s_code_end                                                 // 00000000796c: bf9f0000
	s_code_end                                                 // 000000007970: bf9f0000
	s_code_end                                                 // 000000007974: bf9f0000
	s_code_end                                                 // 000000007978: bf9f0000
	s_code_end                                                 // 00000000797c: bf9f0000
	s_code_end                                                 // 000000007980: bf9f0000
	s_code_end                                                 // 000000007984: bf9f0000
	s_code_end                                                 // 000000007988: bf9f0000
	s_code_end                                                 // 00000000798c: bf9f0000
	s_code_end                                                 // 000000007990: bf9f0000
	s_code_end                                                 // 000000007994: bf9f0000
	s_code_end                                                 // 000000007998: bf9f0000
	s_code_end                                                 // 00000000799c: bf9f0000
	s_code_end                                                 // 0000000079a0: bf9f0000
	s_code_end                                                 // 0000000079a4: bf9f0000
	s_code_end                                                 // 0000000079a8: bf9f0000
	s_code_end                                                 // 0000000079ac: bf9f0000
	s_code_end                                                 // 0000000079b0: bf9f0000
	s_code_end                                                 // 0000000079b4: bf9f0000
	s_code_end                                                 // 0000000079b8: bf9f0000
	s_code_end                                                 // 0000000079bc: bf9f0000
	s_code_end                                                 // 0000000079c0: bf9f0000
	s_code_end                                                 // 0000000079c4: bf9f0000
	s_code_end                                                 // 0000000079c8: bf9f0000
	s_code_end                                                 // 0000000079cc: bf9f0000
	s_code_end                                                 // 0000000079d0: bf9f0000
	s_code_end                                                 // 0000000079d4: bf9f0000
	s_code_end                                                 // 0000000079d8: bf9f0000
	s_code_end                                                 // 0000000079dc: bf9f0000
	s_code_end                                                 // 0000000079e0: bf9f0000
	s_code_end                                                 // 0000000079e4: bf9f0000
	s_code_end                                                 // 0000000079e8: bf9f0000
	s_code_end                                                 // 0000000079ec: bf9f0000
	s_code_end                                                 // 0000000079f0: bf9f0000
	s_code_end                                                 // 0000000079f4: bf9f0000
	s_code_end                                                 // 0000000079f8: bf9f0000
	s_code_end                                                 // 0000000079fc: bf9f0000
	s_code_end                                                 // 000000007a00: bf9f0000
	s_code_end                                                 // 000000007a04: bf9f0000
	s_code_end                                                 // 000000007a08: bf9f0000
	s_code_end                                                 // 000000007a0c: bf9f0000
	s_code_end                                                 // 000000007a10: bf9f0000
	s_code_end                                                 // 000000007a14: bf9f0000
	s_code_end                                                 // 000000007a18: bf9f0000
	s_code_end                                                 // 000000007a1c: bf9f0000
	s_code_end                                                 // 000000007a20: bf9f0000
	s_code_end                                                 // 000000007a24: bf9f0000
	s_code_end                                                 // 000000007a28: bf9f0000
	s_code_end                                                 // 000000007a2c: bf9f0000
	s_code_end                                                 // 000000007a30: bf9f0000
	s_code_end                                                 // 000000007a34: bf9f0000
	s_code_end                                                 // 000000007a38: bf9f0000
	s_code_end                                                 // 000000007a3c: bf9f0000
	s_code_end                                                 // 000000007a40: bf9f0000
	s_code_end                                                 // 000000007a44: bf9f0000
	s_code_end                                                 // 000000007a48: bf9f0000
	s_code_end                                                 // 000000007a4c: bf9f0000
	s_code_end                                                 // 000000007a50: bf9f0000
	s_code_end                                                 // 000000007a54: bf9f0000
	s_code_end                                                 // 000000007a58: bf9f0000
	s_code_end                                                 // 000000007a5c: bf9f0000
	s_code_end                                                 // 000000007a60: bf9f0000
	s_code_end                                                 // 000000007a64: bf9f0000
	s_code_end                                                 // 000000007a68: bf9f0000
	s_code_end                                                 // 000000007a6c: bf9f0000
	s_code_end                                                 // 000000007a70: bf9f0000
	s_code_end                                                 // 000000007a74: bf9f0000
	s_code_end                                                 // 000000007a78: bf9f0000
	s_code_end                                                 // 000000007a7c: bf9f0000
