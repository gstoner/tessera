
/tmp/tmpq3dnzj3m.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <packed_folded_w4a8>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b04: f4002100 f80000d8
	s_load_b128 s[36:39], s[0:1], 0xc8                         // 000000001b0c: f4004900 f80000c8
	s_mov_b32 s12, ttmp7                                       // 000000001b14: be8c0073
	s_ashr_i32 s13, ttmp7, 31                                  // 000000001b18: 860d9f73
	v_lshrrev_b32_e32 v1, 2, v0                                // 000000001b1c: 32020082
	s_lshl_b64 s[26:27], s[12:13], 8                           // 000000001b20: 849a880c
	v_dual_mov_b32 v56, 0 :: v_dual_and_b32 v71, 0xc0, v0      // 000000001b24: ca240080 384600ff 000000c0
	s_clause 0x2                                               // 000000001b30: bf850002
	s_load_b64 s[8:9], s[0:1], 0x8                             // 000000001b34: f4002200 f8000008
	s_load_b64 s[6:7], s[0:1], 0x30                            // 000000001b3c: f4002180 f8000030
	s_load_b64 s[10:11], s[0:1], 0x80                          // 000000001b44: f4002280 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b4c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b50: 86039f75
	v_mul_u32_u24_e32 v5, 0x50, v1                             // 000000001b54: 160a02ff 00000050
	s_lshl_b64 s[14:15], s[2:3], 6                             // 000000001b5c: 848e8602
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b60: bf870009
	v_dual_mov_b32 v13, s15 :: v_dual_and_b32 v70, 15, v0      // 000000001b64: ca24000f 0d46008f
	v_or_b32_e32 v114, 48, v71                                 // 000000001b6c: 38e48eb0
	v_or_b32_e32 v90, 16, v71                                  // 000000001b70: 38b48e90
	v_and_b32_e32 v6, 0xcf, v0                                 // 000000001b74: 360c00ff 000000cf
	v_mov_b32_e32 v14, s27                                     // 000000001b7c: 7e1c021b
	v_or_b32_e32 v102, 32, v71                                 // 000000001b80: 38cc8ea0
	v_or_b32_e32 v8, v114, v70                                 // 000000001b84: 38108d72
	s_wait_kmcnt 0x0                                           // 000000001b88: bfc70000
	s_add_nc_u64 s[12:13], s[36:37], -1                        // 000000001b8c: a98cc124
	v_lshlrev_b32_e32 v2, 4, v0                                // 000000001b90: 30040084
	v_and_b32_e32 v10, 47, v0                                  // 000000001b94: 361400af
	v_or_b32_e32 v7, v102, v70                                 // 000000001b98: 380e8d66
	v_mul_u32_u24_e32 v8, 0x50, v8                             // 000000001b9c: 161010ff 00000050
	s_lshr_b64 s[2:3], s[4:5], 5                               // 000000001ba4: 85828504
	v_dual_mov_b32 v2, v56 :: v_dual_and_b32 v3, 48, v2        // 000000001ba8: ca240138 020204b0
	v_or_b32_e32 v12, s14, v1                                  // 000000001bb0: 3818020e
	s_mul_u64 s[2:3], s[2:3], s[38:39]                         // 000000001bb4: aa822602
	v_mul_u32_u24_e32 v7, 0x50, v7                             // 000000001bb8: 160e0eff 00000050
	s_delay_alu instid0(valu_dep_3)                            // 000000001bc0: bf870003
	v_add_nc_u32_e32 v72, v5, v3                               // 000000001bc4: 4a900705
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001bc8: 320a0081
	v_mov_b32_e32 v4, v56                                      // 000000001bcc: 7e080338
	s_add_nc_u64 s[40:41], s[10:11], s[2:3]                    // 000000001bd0: a9a8020a
	v_and_b32_e32 v124, 32, v0                                 // 000000001bd4: 36f800a0
	v_add_co_u32 v64, vcc_lo, s40, v12                         // 000000001bd8: d7006a40 02021828
	v_and_b32_e32 v115, 8, v5                                  // 000000001be0: 36e60a88
	v_mul_u32_u24_e32 v5, 0x50, v6                             // 000000001be4: 160a0cff 00000050
	v_or_b32_e32 v6, v90, v70                                  // 000000001bec: 380c8d5a
	v_add_co_ci_u32_e64 v65, null, s41, v13, vcc_lo            // 000000001bf0: d5207c41 01aa1a29
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001bf8: bf870214
	v_or_b32_e32 v76, v8, v115                                 // 000000001bfc: 3898e708
	v_or_b32_e32 v73, v115, v5                                 // 000000001c00: 38920b73
	s_delay_alu instid0(valu_dep_4)                            // 000000001c04: bf870004
	v_mul_u32_u24_e32 v9, 0x50, v6                             // 000000001c08: 16120cff 00000050
	v_or_b32_e32 v5, s26, v1                                   // 000000001c10: 380a021a
	v_mov_b32_e32 v6, s27                                      // 000000001c14: 7e0c021b
	v_mul_u32_u24_e32 v8, 0x50, v10                            // 000000001c18: 161014ff 00000050
	v_or_b32_e32 v75, v7, v115                                 // 000000001c20: 3896e707
	v_mov_b32_e32 v7, s27                                      // 000000001c24: 7e0e021b
	v_or_b32_e32 v127, 16, v124                                // 000000001c28: 38fef890
	v_cmp_gt_u64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001c2c: 7cb80a0c
	v_or_b32_e32 v6, 64, v5                                    // 000000001c30: 380c0ac0
	v_or_b32_e32 v15, v115, v8                                 // 000000001c34: 381e1173
	v_or_b32_e32 v74, v9, v115                                 // 000000001c38: 3894e709
	v_or_b32_e32 v9, v127, v70                                 // 000000001c3c: 38128d7f
	v_alignbit_b32 v12, v13, v12, 4                            // 000000001c40: d616000c 0212190d
	s_wait_alu depctr_va_vcc(0)                                // 000000001c48: bf88ff9d
	v_cndmask_b32_e32 v8, s12, v5, vcc_lo                      // 000000001c4c: 02100a0c
	v_add_nc_u32_e32 v85, 0x5000, v15                          // 000000001c50: 4aaa1eff 00005000
	v_cndmask_b32_e32 v10, s13, v14, vcc_lo                    // 000000001c58: 02141c0d
	v_cmp_gt_u64_e32 vcc_lo, s[12:13], v[6:7]                  // 000000001c5c: 7cb80c0c
	v_mul_u32_u24_e32 v16, 0x50, v9                            // 000000001c60: 162012ff 00000050
	v_mul_lo_u32 v17, s5, v8                                   // 000000001c68: d72c0011 02021005
	v_mov_b32_e32 v9, s27                                      // 000000001c70: 7e12021b
	v_mul_lo_u32 v18, s4, v10                                  // 000000001c74: d72c0012 02021404
	v_mov_b32_e32 v59, v56                                     // 000000001c7c: 7e760338
	s_wait_alu depctr_va_vcc(0)                                // 000000001c80: bf88ff9d
	v_cndmask_b32_e32 v11, s12, v6, vcc_lo                     // 000000001c84: 02160c0c
	v_mad_co_u64_u32 v[6:7], null, s4, v8, v[3:4]              // 000000001c88: d6fe7c06 040e1004
	v_cndmask_b32_e32 v19, s13, v14, vcc_lo                    // 000000001c90: 02261c0d
	v_or_b32_e32 v8, 0x80, v5                                  // 000000001c94: 38100aff 00000080
	v_or_b32_e32 v5, 0xc0, v5                                  // 000000001c9c: 380a0aff 000000c0
	v_mul_lo_u32 v20, s5, v11                                  // 000000001ca4: d72c0014 02021605
	v_mad_co_u64_u32 v[10:11], null, s4, v11, v[3:4]           // 000000001cac: d6fe7c0a 040e1604
	v_mul_lo_u32 v19, s4, v19                                  // 000000001cb4: d72c0013 02022604
	v_cmp_gt_u64_e32 vcc_lo, s[12:13], v[8:9]                  // 000000001cbc: 7cb8100c
	v_add3_u32 v7, v17, v7, v18                                // 000000001cc0: d6550007 044a0f11
	v_add_co_u32 v77, s2, s8, v6                               // 000000001cc8: d700024d 02020c08
	v_mov_b32_e32 v6, s27                                      // 000000001cd0: 7e0c021b
	s_lshr_b64 s[16:17], s[4:5], 4                             // 000000001cd4: 85908404
	s_delay_alu instid0(valu_dep_3)                            // 000000001cd8: bf870003
	v_add_co_ci_u32_e64 v78, null, s9, v7, s2                  // 000000001cdc: d5207c4e 000a0e09
	v_add3_u32 v7, v20, v11, v19                               // 000000001ce4: d6550007 044e1714
	s_wait_alu depctr_va_vcc(0)                                // 000000001cec: bf88ff9d
	v_dual_cndmask_b32 v8, s12, v8 :: v_dual_mov_b32 v57, v56  // 000000001cf0: ca50100c 08380138
	v_cndmask_b32_e32 v9, s13, v14, vcc_lo                     // 000000001cf8: 02121c0d
	v_add_co_u32 v79, vcc_lo, s8, v10                          // 000000001cfc: d7006a4f 02021408
	s_wait_alu depctr_va_vcc(0)                                // 000000001d04: bf88ff9d
	v_add_co_ci_u32_e64 v80, null, s9, v7, vcc_lo              // 000000001d08: d5207c50 01aa0e09
	v_cmp_gt_u64_e32 vcc_lo, s[12:13], v[5:6]                  // 000000001d10: 7cb80a0c
	v_dual_mov_b32 v6, v56 :: v_dual_mov_b32 v61, v56          // 000000001d14: ca100138 063c0138
	s_lshr_b32 s2, s5, 4                                       // 000000001d1c: 85028405
	v_mul_lo_u32 v10, s5, v8                                   // 000000001d20: d72c000a 02021005
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d28: bf88ff9e
	v_mul_lo_u32 v17, s2, v12                                  // 000000001d2c: d72c0011 02021802
	s_wait_alu depctr_va_vcc(0)                                // 000000001d34: bf88ff9d
	v_cndmask_b32_e32 v11, s12, v5, vcc_lo                     // 000000001d38: 02160a0c
	v_and_b32_e32 v5, 3, v0                                    // 000000001d3c: 360a0083
	v_cndmask_b32_e32 v13, s13, v14, vcc_lo                    // 000000001d40: 021a1c0d
	v_mad_co_u64_u32 v[7:8], null, s4, v8, v[3:4]              // 000000001d44: d6fe7c07 040e1004
	v_mul_lo_u32 v9, s4, v9                                    // 000000001d4c: d72c0009 02021204
	v_bfe_u32 v14, v0, 1, 1                                    // 000000001d54: d610000e 02050300
	v_mad_co_u64_u32 v[5:6], null, s16, v12, v[5:6]            // 000000001d5c: d6fe7c05 04161810
	s_lshr_b32 s2, s15, 4                                      // 000000001d64: 8502840f
	v_mul_lo_u32 v12, s5, v11                                  // 000000001d68: d72c000c 02021605
	v_mad_co_u64_u32 v[3:4], null, s4, v11, v[3:4]             // 000000001d70: d6fe7c03 040e1604
	v_mul_lo_u32 v11, s4, v13                                  // 000000001d78: d72c000b 02021a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d80: bf88ff9e
	s_mul_i32 s2, s16, s2                                      // 000000001d84: 96020210
	v_mad_co_u64_u32 v[1:2], null, s38, v14, v[1:2]            // 000000001d88: d6fe7c01 04061c26
	v_add3_u32 v8, v10, v8, v9                                 // 000000001d90: d6550008 0426110a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d98: bf88ff9e
	v_add3_u32 v6, v17, v6, s2                                 // 000000001d9c: d6550006 000a0d11
	v_add_co_u32 v81, vcc_lo, s8, v7                           // 000000001da4: d7006a51 02020e08
	s_add_nc_u64 s[2:3], s[10:11], s[14:15]                    // 000000001dac: a9820e0a
	v_add3_u32 v9, v12, v4, v11                                // 000000001db0: d6550009 042e090c
	v_lshlrev_b64_e32 v[4:5], 7, v[5:6]                        // 000000001db8: 3e080a87
	s_wait_alu depctr_va_vcc(0)                                // 000000001dbc: bf88ff9d
	v_add_co_ci_u32_e64 v82, null, s9, v8, vcc_lo              // 000000001dc0: d5207c52 01aa1009
	v_mad_co_u64_u32 v[7:8], null, s39, v14, v[2:3]            // 000000001dc8: d6fe7c07 040a1c27
	v_add_co_u32 v83, vcc_lo, s8, v3                           // 000000001dd0: d7006a53 02020608
	s_delay_alu instid0(valu_dep_4)                            // 000000001dd8: bf870004
	v_and_or_b32 v0, v0, 60, v4                                // 000000001ddc: d6570000 04117900
	v_or_b32_e32 v16, v16, v115                                // 000000001de4: 3820e710
	s_wait_alu depctr_va_vcc(0)                                // 000000001de8: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, s9, v9, vcc_lo              // 000000001dec: d5207c54 01aa1209
	s_wait_alu depctr_sa_sdst(0)                               // 000000001df4: bf88ff9e
	v_add_co_u32 v66, vcc_lo, s2, v1                           // 000000001df8: d7006a42 02020202
	s_wait_alu depctr_va_vcc(0)                                // 000000001e00: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s3, v7, vcc_lo              // 000000001e04: d5207c43 01aa0e03
	v_add_co_u32 v68, vcc_lo, s6, v0                           // 000000001e0c: d7006a44 02020006
	s_wait_alu depctr_va_vcc(0)                                // 000000001e14: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s7, v5, vcc_lo              // 000000001e18: d5207c45 01aa0a07
	v_dual_mov_b32 v63, v56 :: v_dual_add_nc_u32 v86, 0x5000, v16// 000000001e20: ca200138 3f5620ff 00005000
	v_dual_mov_b32 v58, v56 :: v_dual_mov_b32 v25, v56         // 000000001e2c: ca100138 3a180138
	v_dual_mov_b32 v60, v56 :: v_dual_mov_b32 v27, v56         // 000000001e34: ca100138 3c1a0138
	v_dual_mov_b32 v62, v56 :: v_dual_mov_b32 v29, v56         // 000000001e3c: ca100138 3e1c0138
	v_dual_mov_b32 v24, v56 :: v_dual_mov_b32 v31, v56         // 000000001e44: ca100138 181e0138
	v_dual_mov_b32 v26, v56 :: v_dual_mov_b32 v49, v56         // 000000001e4c: ca100138 1a300138
	v_dual_mov_b32 v28, v56 :: v_dual_mov_b32 v51, v56         // 000000001e54: ca100138 1c320138
	v_dual_mov_b32 v30, v56 :: v_dual_mov_b32 v53, v56         // 000000001e5c: ca100138 1e340138
	v_dual_mov_b32 v48, v56 :: v_dual_mov_b32 v55, v56         // 000000001e64: ca100138 30360138
	v_dual_mov_b32 v50, v56 :: v_dual_mov_b32 v17, v56         // 000000001e6c: ca100138 32100138
	v_dual_mov_b32 v52, v56 :: v_dual_mov_b32 v19, v56         // 000000001e74: ca100138 34120138
	v_dual_mov_b32 v54, v56 :: v_dual_mov_b32 v21, v56         // 000000001e7c: ca100138 36140138
	v_dual_mov_b32 v16, v56 :: v_dual_mov_b32 v23, v56         // 000000001e84: ca100138 10160138
	v_dual_mov_b32 v18, v56 :: v_dual_mov_b32 v41, v56         // 000000001e8c: ca100138 12280138
	v_dual_mov_b32 v20, v56 :: v_dual_mov_b32 v43, v56         // 000000001e94: ca100138 142a0138
	v_dual_mov_b32 v22, v56 :: v_dual_mov_b32 v45, v56         // 000000001e9c: ca100138 162c0138
	v_dual_mov_b32 v40, v56 :: v_dual_mov_b32 v47, v56         // 000000001ea4: ca100138 282e0138
	v_dual_mov_b32 v42, v56 :: v_dual_mov_b32 v9, v56          // 000000001eac: ca100138 2a080138
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v11, v56         // 000000001eb4: ca100138 2c0a0138
	v_dual_mov_b32 v46, v56 :: v_dual_mov_b32 v13, v56         // 000000001ebc: ca100138 2e0c0138
	v_dual_mov_b32 v8, v56 :: v_dual_mov_b32 v15, v56          // 000000001ec4: ca100138 080e0138
	v_dual_mov_b32 v10, v56 :: v_dual_mov_b32 v33, v56         // 000000001ecc: ca100138 0a200138
	v_dual_mov_b32 v12, v56 :: v_dual_mov_b32 v35, v56         // 000000001ed4: ca100138 0c220138
	v_dual_mov_b32 v14, v56 :: v_dual_mov_b32 v37, v56         // 000000001edc: ca100138 0e240138
	v_dual_mov_b32 v32, v56 :: v_dual_mov_b32 v39, v56         // 000000001ee4: ca100138 20260138
	v_dual_mov_b32 v34, v56 :: v_dual_mov_b32 v1, v56          // 000000001eec: ca100138 22000138
	v_dual_mov_b32 v36, v56 :: v_dual_mov_b32 v3, v56          // 000000001ef4: ca100138 24020138
	v_dual_mov_b32 v38, v56 :: v_dual_mov_b32 v5, v56          // 000000001efc: ca100138 26040138
	v_dual_mov_b32 v0, v56 :: v_dual_mov_b32 v7, v56           // 000000001f04: ca100138 00060138
	v_mov_b32_e32 v2, v56                                      // 000000001f0c: 7e040338
	v_mov_b32_e32 v4, v56                                      // 000000001f10: 7e080338
	v_mov_b32_e32 v6, v56                                      // 000000001f14: 7e0c0338
	s_lshl_b64 s[16:17], s[38:39], 1                           // 000000001f18: 84908126
	s_mov_b64 s[18:19], 0                                      // 000000001f1c: be920180
	global_load_u8 v101, v[66:67], off                         // 000000001f20: ee04007c 00000065 00000042
	global_load_u8 v111, v[64:65], off                         // 000000001f2c: ee04007c 0000006f 00000040
	global_load_d16_u8 v87, v[66:67], off                      // 000000001f38: ee07807c 00000057 00000042
	global_load_d16_hi_u8 v87, v[64:65], off                   // 000000001f44: ee08407c 00000057 00000040
	v_add_co_u32 v88, vcc_lo, v77, s18                         // 000000001f50: d7006a58 0200254d
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v89, null, s19, v78, vcc_lo            // 000000001f5c: d5207c59 01aa9c13
	v_add_co_u32 v95, s2, v79, s18                             // 000000001f64: d700025f 0200254f
	s_wait_alu depctr_va_sdst(0)                               // 000000001f6c: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s19, v80, s2                // 000000001f70: d5207c60 000aa013
	global_load_b128 v[91:94], v[88:89], off                   // 000000001f78: ee05c07c 0000005b 00000058
	s_clause 0x1                                               // 000000001f84: bf850001
	global_load_b32 v112, v[68:69], off                        // 000000001f88: ee05007c 00000070 00000044
	global_load_b32 v113, v[68:69], off offset:64              // 000000001f94: ee05007c 00000071 00004044
	global_load_b128 v[95:98], v[95:96], off                   // 000000001fa0: ee05c07c 0000005f 0000005f
	v_add_co_u32 v99, s3, v81, s18                             // 000000001fac: d7000363 02002551
	s_wait_alu depctr_va_sdst(0)                               // 000000001fb4: bf88f19f
	v_add_co_ci_u32_e64 v100, null, s19, v82, s3               // 000000001fb8: d5207c64 000ea413
	v_add_co_u32 v107, s4, v83, s18                            // 000000001fc0: d700046b 02002553
	s_wait_alu depctr_va_sdst(0)                               // 000000001fc8: bf88f19f
	v_add_co_ci_u32_e64 v108, null, s19, v84, s4               // 000000001fcc: d5207c6c 0012a813
	global_load_b128 v[103:106], v[99:100], off                // 000000001fd4: ee05c07c 00000067 00000063
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fe0: bf88ff9e
	v_add_co_u32 v66, vcc_lo, v66, s16                         // 000000001fe4: d7006a42 02002142
	global_load_b128 v[107:110], v[107:108], off               // 000000001fec: ee05c07c 0000006b 0000006b
	s_wait_alu depctr_va_vcc(0)                                // 000000001ff8: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s17, v67, vcc_lo            // 000000001ffc: d5207c43 01aa8611
	s_barrier_signal -1                                        // 000000002004: be804ec1
	s_barrier_wait 0xffff                                      // 000000002008: bf94ffff
	v_add_co_u32 v68, s2, 0x200, v68                           // 00000000200c: d7000244 020288ff 00000200
	s_wait_alu depctr_va_sdst(0)                               // 000000002018: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v69, s2                  // 00000000201c: d5207c45 000a8a80
	s_add_nc_u64 s[18:19], s[18:19], 64                        // 000000002024: a992c012
	s_wait_loadcnt 0x8                                         // 000000002028: bfc00008
	v_sub_nc_u32_e32 v100, v111, v101                          // 00000000202c: 4cc8cb6f
	s_wait_loadcnt 0x6                                         // 000000002030: bfc00006
	v_cmp_eq_u16_e64 s2, 0, v87.l                              // 000000002034: d43a0002 0202ae80
	v_cmp_eq_u16_e32 vcc_lo, v87.h, v87.l                      // 00000000203c: 7c74afd7
	s_delay_alu instid0(valu_dep_3)                            // 000000002040: bf870003
	v_cmp_ne_u32_e64 s3, 2, v100                               // 000000002044: d44d0003 0202c882
	v_cmp_ne_u32_e64 s4, 3, v100                               // 00000000204c: d44d0004 0202c883
	s_wait_alu depctr_va_vcc(0)                                // 000000002054: bf88ff9d
	v_cndmask_b32_e64 v134, 0, 0x3c383000, vcc_lo              // 000000002058: d5010086 01a9fe80 3c383000
	v_cndmask_b32_e64 v135, 0, 0x4c484440, vcc_lo              // 000000002064: d5010087 01a9fe80 4c484440
	v_cmp_ne_u32_e32 vcc_lo, 1, v100                           // 000000002070: 7c9ac881
	v_cmp_ne_u32_e64 s5, 4, v100                               // 000000002074: d44d0005 0202c884
	s_wait_loadcnt 0x5                                         // 00000000207c: bfc00005
	ds_store_b128 v72, v[91:94]                                // 000000002080: db7c0000 00005b48
	v_cmp_ne_u32_e64 s6, 5, v100                               // 000000002088: d44d0006 0202c885
	v_cmp_ne_u32_e64 s7, 6, v100                               // 000000002090: d44d0007 0202c886
	s_wait_alu depctr_va_vcc(0)                                // 000000002098: bf88ff9d
	v_cndmask_b32_e32 v93, 0x44403c38, v135, vcc_lo            // 00000000209c: 02bb0eff 44403c38
	v_cndmask_b32_e32 v94, 0x34302800, v134, vcc_lo            // 0000000020a4: 02bd0cff 34302800
	v_cmp_ne_u32_e64 s8, 7, v100                               // 0000000020ac: d44d0008 0202c887
	v_cmp_ne_u32_e64 s9, 8, v100                               // 0000000020b4: d44d0009 0202c888
	v_cmp_ne_u32_e64 s10, 9, v100                              // 0000000020bc: d44d000a 0202c889
	s_wait_alu depctr_va_sdst(0)                               // 0000000020c4: bf88f19f
	v_cndmask_b32_e64 v93, 0x3c383430, v93, s3                 // 0000000020c8: d501005d 000ebaff 3c383430
	v_cndmask_b32_e64 v94, 0x2c282000, v94, s3                 // 0000000020d4: d501005e 000ebcff 2c282000
	v_cmp_ne_u32_e64 s11, 10, v100                             // 0000000020e0: d44d000b 0202c88a
	v_cmp_ne_u32_e64 s12, 11, v100                             // 0000000020e8: d44d000c 0202c88b
	v_cmp_ne_u32_e64 s13, 12, v100                             // 0000000020f0: d44d000d 0202c88c
	v_cndmask_b32_e64 v93, 0x34302c28, v93, s4                 // 0000000020f8: d501005d 0012baff 34302c28
	v_cndmask_b32_e64 v94, 0x24201800, v94, s4                 // 000000002104: d501005e 0012bcff 24201800
	s_wait_loadcnt 0x4                                         // 000000002110: bfc00004
	v_lshrrev_b32_e32 v121, 9, v112                            // 000000002114: 32f2e089
	s_wait_loadcnt 0x2                                         // 000000002118: bfc00002
	ds_store_b128 v72, v[95:98] offset:5120                    // 00000000211c: db7c1400 00005f48
	v_lshrrev_b32_e32 v122, 13, v112                           // 000000002124: 32f4e08d
	v_cndmask_b32_e64 v93, 0x2c282420, v93, s5                 // 000000002128: d501005d 0016baff 2c282420
	v_cndmask_b32_e64 v94, 0x1c181000, v94, s5                 // 000000002134: d501005e 0016bcff 1c181000
	v_lshrrev_b32_e32 v129, 5, v113                            // 000000002140: 3302e285
	v_lshrrev_b32_e32 v123, 17, v112                           // 000000002144: 32f6e091
	v_lshrrev_b32_e32 v130, 9, v113                            // 000000002148: 3304e289
	v_cndmask_b32_e64 v93, 0x24201c18, v93, s6                 // 00000000214c: d501005d 001abaff 24201c18
	v_cndmask_b32_e64 v94, 0x14100800, v94, s6                 // 000000002158: d501005e 001abcff 14100800
	v_lshrrev_b32_e32 v119, 1, v112                            // 000000002164: 32eee081
	v_lshrrev_b32_e32 v125, 21, v112                           // 000000002168: 32fae095
	v_lshrrev_b32_e32 v131, 13, v113                           // 00000000216c: 3306e28d
	v_cndmask_b32_e64 v93, 0x1c181410, v93, s7                 // 000000002170: d501005d 001ebaff 1c181410
	v_cndmask_b32_e64 v94, 0xc080400, v94, s7                  // 00000000217c: d501005e 001ebcff 0c080400
	s_and_b32 vcc_lo, s13, s12                                 // 000000002188: 8b6a0c0d
	v_lshrrev_b32_e32 v101, 8, v112                            // 00000000218c: 32cae088
	v_lshrrev_b32_e32 v111, 24, v112                           // 000000002190: 32dee098
	v_cndmask_b32_e64 v93, 0x14100c08, v93, s8                 // 000000002194: d501005d 0022baff 14100c08
	v_cndmask_b32_e64 v94, 0x6040200, v94, s8                  // 0000000021a0: d501005e 0022bcff 06040200
	v_lshlrev_b32_e32 v118, 3, v112                            // 0000000021ac: 30ece083
	v_lshlrev_b16 v87.h, 4, v112.l op_sel:[0,0,1]              // 0000000021b0: d7384057 0202e084
	v_and_b16 v87.l, 0x80, v112.l                              // 0000000021b8: d7620057 0202e0ff 00000080
	v_cndmask_b32_e64 v93, 0xc080604, v93, s9                  // 0000000021c4: d501005d 0026baff 0c080604
	v_cndmask_b32_e64 v94, 0x3020100, v94, s9                  // 0000000021d0: d501005e 0026bcff 03020100
	v_lshrrev_b32_e32 v120, 5, v112                            // 0000000021dc: 32f0e085
	v_lshlrev_b16 v88.l, 4, v112.h op_sel:[0,1,0]              // 0000000021e0: d7381058 0202e084
	v_and_b16 v88.h, 0x80, v112.h op_sel:[0,1,1]               // 0000000021e8: d7625058 0202e0ff 00000080
	v_cndmask_b32_e64 v93, 0x6040302, v93, s10                 // 0000000021f4: d501005d 002abaff 06040302
	v_cndmask_b32_e64 v94, 0x2010000, v94, s10                 // 000000002200: d501005e 002abcff 02010000
	v_lshrrev_b32_e32 v112, 25, v112                           // 00000000220c: 32e0e099
	v_lshrrev_b32_e32 v132, 17, v113                           // 000000002210: 3308e291
	v_lshrrev_b32_e32 v133, 21, v113                           // 000000002214: 330ae295
	v_cndmask_b32_e64 v93, 0x3020201, v93, s11                 // 000000002218: d501005d 002ebaff 03020201
	v_cndmask_b32_e64 v94, 0x1000000, v94, s11                 // 000000002224: d501005e 002ebcff 01000000
	v_and_b32_e32 v121, 56, v121                               // 000000002230: 36f2f2b8
	v_lshrrev_b32_e32 v116, 8, v113                            // 000000002234: 32e8e288
	v_lshrrev_b32_e32 v117, 24, v113                           // 000000002238: 32eae298
	v_cndmask_b32_e64 v95, 0x2010100, v93, s12                 // 00000000223c: d501005f 0032baff 02010100
	s_wait_alu depctr_sa_sdst(0)                               // 000000002248: bf88ff9e
	v_dual_cndmask_b32 v93, 0, v94 :: v_dual_lshlrev_b32 v126, 3, v113// 00000000224c: ca62bc80 5d7ee283
	v_lshlrev_b16 v89.l, 4, v113.l                             // 000000002254: d7380059 0202e284
	v_lshrrev_b32_e32 v128, 1, v113                            // 00000000225c: 3300e281
	v_cndmask_b32_e64 v94, 0x1000000, v95, s13                 // 000000002260: d501005e 0036beff 01000000
	v_and_b16 v89.h, 0x80, v113.l op_sel:[0,0,1]               // 00000000226c: d7624059 0202e2ff 00000080
	v_lshlrev_b16 v99.l, 4, v113.h op_sel:[0,1,0]              // 000000002278: d7381063 0202e284
	v_and_b16 v99.h, 0x80, v113.h op_sel:[0,1,1]               // 000000002280: d7625063 0202e2ff 00000080
	v_lshrrev_b32_e32 v113, 25, v113                           // 00000000228c: 32e2e299
	v_and_b32_e32 v122, 56, v122                               // 000000002290: 36f4f4b8
	v_and_b32_e32 v129, 56, v129                               // 000000002294: 370302b8
	v_and_b32_e32 v123, 56, v123                               // 000000002298: 36f6f6b8
	v_and_b32_e32 v130, 56, v130                               // 00000000229c: 370504b8
	v_and_b32_e32 v119, 56, v119                               // 0000000022a0: 36eeeeb8
	v_and_b32_e32 v125, 56, v125                               // 0000000022a4: 36fafab8
	v_and_b32_e32 v131, 56, v131                               // 0000000022a8: 370706b8
	v_and_b32_e32 v120, 56, v120                               // 0000000022ac: 36f0f0b8
	v_and_b32_e32 v136, 56, v112                               // 0000000022b0: 3710e0b8
	v_and_b32_e32 v132, 56, v132                               // 0000000022b4: 370908b8
	v_and_b32_e32 v133, 56, v133                               // 0000000022b8: 370b0ab8
	s_wait_loadcnt 0x1                                         // 0000000022bc: bfc00001
	ds_store_b128 v72, v[103:106] offset:10240                 // 0000000022c0: db7c2800 00006748
	v_lshrrev_b64 v[103:104], v121, v[93:94]                   // 0000000022c8: d73d0067 0202bb79
	v_lshlrev_b16 v100.l, 4, v101.l                            // 0000000022d0: d7380064 0202ca84
	v_and_b16 v100.h, 0x80, v101.l op_sel:[0,0,1]              // 0000000022d8: d7624064 0202caff 00000080
	v_lshlrev_b16 v101.l, 4, v111.l                            // 0000000022e4: d7380065 0202de84
	v_and_b16 v101.h, 0x80, v111.l op_sel:[0,0,1]              // 0000000022ec: d7624065 0202deff 00000080
	v_and_b32_e32 v128, 56, v128                               // 0000000022f8: 370100b8
	v_lshlrev_b16 v111.l, 4, v116.l                            // 0000000022fc: d738006f 0202e884
	v_and_b16 v111.h, 0x80, v116.l op_sel:[0,0,1]              // 000000002304: d762406f 0202e8ff 00000080
	v_lshlrev_b16 v112.l, 4, v117.l                            // 000000002310: d7380070 0202ea84
	v_and_b32_e32 v113, 56, v113                               // 000000002318: 36e2e2b8
	v_and_b16 v112.h, 0x80, v117.l op_sel:[0,0,1]              // 00000000231c: d7624070 0202eaff 00000080
	v_lshrrev_b64 v[104:105], v122, v[93:94]                   // 000000002328: d73d0068 0202bb7a
	v_lshrrev_b64 v[116:117], v129, v[93:94]                   // 000000002330: d73d0074 0202bb81
	v_lshrrev_b64 v[95:96], v118, v[93:94]                     // 000000002338: d73d005f 0202bb76
	v_lshrrev_b64 v[105:106], v123, v[93:94]                   // 000000002340: d73d0069 0202bb7b
	v_lshrrev_b64 v[117:118], v130, v[93:94]                   // 000000002348: d73d0075 0202bb82
	s_wait_loadcnt 0x0                                         // 000000002350: bfc00000
	ds_store_b128 v72, v[107:110] offset:15360                 // 000000002354: db7c3c00 00006b48
	v_lshrrev_b64 v[96:97], v119, v[93:94]                     // 00000000235c: d73d0060 0202bb77
	v_lshrrev_b64 v[106:107], v125, v[93:94]                   // 000000002364: d73d006a 0202bb7d
	v_lshrrev_b64 v[118:119], v131, v[93:94]                   // 00000000236c: d73d0076 0202bb83
	v_lshrrev_b64 v[97:98], v120, v[93:94]                     // 000000002374: d73d0061 0202bb78
	v_lshrrev_b64 v[107:108], v136, v[93:94]                   // 00000000237c: d73d006b 0202bb88
	v_lshrrev_b64 v[119:120], v132, v[93:94]                   // 000000002384: d73d0077 0202bb84
	v_lshrrev_b64 v[108:109], v126, v[93:94]                   // 00000000238c: d73d006c 0202bb7e
	v_lshrrev_b64 v[120:121], v133, v[93:94]                   // 000000002394: d73d0078 0202bb85
	v_lshrrev_b64 v[109:110], v128, v[93:94]                   // 00000000239c: d73d006d 0202bb80
	v_lshrrev_b64 v[121:122], v113, v[93:94]                   // 0000000023a4: d73d0079 0202bb71
	v_and_b16 v87.h, 0x80, v87.h op_sel:[0,1,1]                // 0000000023ac: d7625057 0202aeff 00000080
	v_and_b16 v88.l, 0x80, v88.l                               // 0000000023b8: d7620058 0202b0ff 00000080
	v_and_b16 v89.l, 0x80, v89.l                               // 0000000023c4: d7620059 0202b2ff 00000080
	v_and_b16 v99.l, 0x80, v99.l                               // 0000000023d0: d7620063 0202c6ff 00000080
	v_and_b16 v91.l, 0x80, v100.l                              // 0000000023dc: d762005b 0202c8ff 00000080
	v_and_b16 v91.h, 0x80, v101.l op_sel:[0,0,1]               // 0000000023e8: d762405b 0202caff 00000080
	v_and_b16 v92.l, 0x80, v111.l                              // 0000000023f4: d762005c 0202deff 00000080
	v_and_b16 v92.h, 0x80, v112.l op_sel:[0,0,1]               // 000000002400: d762405c 0202e0ff 00000080
	v_or_b16 v87.h, v87.h, v95.l op_sel:[1,0,1]                // 00000000240c: d7634857 0202bf57
	v_or_b16 v87.l, v87.l, v96.l                               // 000000002414: d7630057 0202c157
	v_or_b16 v91.l, v91.l, v97.l                               // 00000000241c: d763005b 0202c35b
	v_or_b16 v93.l, v100.h, v103.l op_sel:[1,0,0]              // 000000002424: d763085d 0202cf64
	v_or_b16 v88.l, v88.l, v104.l                              // 00000000242c: d7630058 0202d158
	v_or_b16 v88.h, v88.h, v105.l op_sel:[1,0,1]               // 000000002434: d7634858 0202d358
	v_or_b16 v91.h, v91.h, v106.l op_sel:[1,0,1]               // 00000000243c: d763485b 0202d55b
	v_or_b16 v93.h, v101.h, v107.l op_sel:[1,0,1]              // 000000002444: d763485d 0202d765
	v_or_b16 v89.l, v89.l, v108.l                              // 00000000244c: d7630059 0202d959
	v_or_b16 v89.h, v89.h, v109.l op_sel:[1,0,1]               // 000000002454: d7634859 0202db59
	v_or_b16 v92.l, v92.l, v116.l                              // 00000000245c: d763005c 0202e95c
	v_or_b16 v94.l, v111.h, v117.l op_sel:[1,0,0]              // 000000002464: d763085e 0202eb6f
	v_or_b16 v94.h, v99.l, v118.l op_sel:[0,0,1]               // 00000000246c: d763405e 0202ed63
	v_or_b16 v95.l, v99.h, v119.l op_sel:[1,0,0]               // 000000002474: d763085f 0202ef63
	v_or_b16 v92.h, v92.h, v120.l op_sel:[1,0,1]               // 00000000247c: d763485c 0202f15c
	v_or_b16 v95.h, v112.h, v121.l op_sel:[1,0,1]              // 000000002484: d763485f 0202f370
	v_cndmask_b16 v87.h, v87.h, 0, s2                          // 00000000248c: d65d4857 00090157
	v_cndmask_b16 v87.l, v87.l, 0, s2                          // 000000002494: d65d0057 00090157
	v_cndmask_b16 v91.l, v91.l, 0, s2                          // 00000000249c: d65d005b 0009015b
	v_cndmask_b16 v93.l, v93.l, 0, s2                          // 0000000024a4: d65d005d 0009015d
	v_cndmask_b16 v88.l, v88.l, 0, s2                          // 0000000024ac: d65d0058 00090158
	v_cndmask_b16 v88.h, v88.h, 0, s2                          // 0000000024b4: d65d4858 00090158
	v_cndmask_b16 v91.h, v91.h, 0, s2                          // 0000000024bc: d65d485b 0009015b
	v_cndmask_b16 v93.h, v93.h, 0, s2                          // 0000000024c4: d65d485d 0009015d
	v_cndmask_b16 v89.l, v89.l, 0, s2                          // 0000000024cc: d65d0059 00090159
	v_cndmask_b16 v89.h, v89.h, 0, s2                          // 0000000024d4: d65d4859 00090159
	v_cndmask_b16 v92.l, v92.l, 0, s2                          // 0000000024dc: d65d005c 0009015c
	v_cndmask_b16 v95.h, v95.h, 0, s2                          // 0000000024e4: d65d485f 0009015f
	v_cndmask_b16 v92.h, v92.h, 0, s2                          // 0000000024ec: d65d485c 0009015c
	v_cndmask_b16 v95.l, v95.l, 0, s2                          // 0000000024f4: d65d005f 0009015f
	v_cndmask_b16 v94.h, v94.h, 0, s2                          // 0000000024fc: d65d485e 0009015e
	v_cndmask_b16 v94.l, v94.l, 0, s2                          // 000000002504: d65d005e 0009015e
	v_lshlrev_b16 v95.h, 8, v95.h op_sel:[0,1,1]               // 00000000250c: d738505f 0202be88
	v_and_b16 v92.h, 0xff, v92.h op_sel:[0,1,1]                // 000000002514: d762505c 0202b8ff 000000ff
	v_lshlrev_b16 v95.l, 8, v95.l                              // 000000002520: d738005f 0202be88
	v_and_b16 v96.l, 0xff, v94.h op_sel:[0,1,0]                // 000000002528: d7621060 0202bcff 000000ff
	v_lshlrev_b16 v96.h, 8, v94.l op_sel:[0,0,1]               // 000000002534: d7384060 0202bc88
	v_and_b16 v92.l, 0xff, v92.l                               // 00000000253c: d762005c 0202b8ff 000000ff
	v_lshlrev_b16 v89.h, 8, v89.h op_sel:[0,1,1]               // 000000002548: d7385059 0202b288
	v_and_b16 v89.l, 0xff, v89.l                               // 000000002550: d7620059 0202b2ff 000000ff
	v_lshlrev_b16 v97.l, 8, v93.h op_sel:[0,1,0]               // 00000000255c: d7381061 0202ba88
	v_and_b16 v91.h, 0xff, v91.h op_sel:[0,1,1]                // 000000002564: d762505b 0202b6ff 000000ff
	v_lshlrev_b16 v88.h, 8, v88.h op_sel:[0,1,1]               // 000000002570: d7385058 0202b088
	v_and_b16 v88.l, 0xff, v88.l                               // 000000002578: d7620058 0202b0ff 000000ff
	v_lshlrev_b16 v97.h, 8, v93.l op_sel:[0,0,1]               // 000000002584: d7384061 0202ba88
	v_and_b16 v91.l, 0xff, v91.l                               // 00000000258c: d762005b 0202b6ff 000000ff
	v_lshlrev_b16 v87.l, 8, v87.l                              // 000000002598: d7380057 0202ae88
	v_and_b16 v87.h, 0xff, v87.h op_sel:[0,1,1]                // 0000000025a0: d7625057 0202aeff 000000ff
	v_or_b16 v94.h, v92.h, v95.h op_sel:[1,1,1]                // 0000000025ac: d763585e 0202bf5c
	v_or_b16 v94.l, v96.l, v95.l                               // 0000000025b4: d763005e 0202bf60
	v_or_b16 v93.h, v92.l, v96.h op_sel:[0,1,1]                // 0000000025bc: d763505d 0202c15c
	v_or_b16 v93.l, v89.l, v89.h op_sel:[0,1,0]                // 0000000025c4: d763105d 0202b359
	v_or_b16 v92.h, v91.h, v97.l op_sel:[1,0,1]                // 0000000025cc: d763485c 0202c35b
	v_or_b16 v92.l, v88.l, v88.h op_sel:[0,1,0]                // 0000000025d4: d763105c 0202b158
	v_or_b16 v91.h, v91.l, v97.h op_sel:[0,1,1]                // 0000000025dc: d763505b 0202c35b
	v_or_b16 v91.l, v87.h, v87.l op_sel:[1,0,0]                // 0000000025e4: d763085b 0202af57
	s_cmp_lg_u64 s[18:19], 0x600                               // 0000000025ec: bf11ff12 00000600
	ds_store_b128 v72, v[91:94] offset:20480                   // 0000000025f4: db7c5000 00005b48
	s_wait_dscnt 0x0                                           // 0000000025fc: bfc60000
	s_barrier_signal -1                                        // 000000002600: be804ec1
	s_barrier_wait 0xffff                                      // 000000002604: bf94ffff
	ds_load_2addr_b64 v[91:94], v73 offset1:2                  // 000000002608: d9dc0200 5b000049
	ds_load_2addr_b64 v[95:98], v85 offset1:2                  // 000000002610: d9dc0200 5f000055
	ds_load_2addr_b64 v[103:106], v86 offset1:2                // 000000002618: d9dc0200 67000056
	ds_load_2addr_b64 v[107:110], v74 offset1:2                // 000000002620: d9dc0200 6b00004a
	ds_load_2addr_b64 v[116:119], v75 offset1:2                // 000000002628: d9dc0200 7400004b
	ds_load_2addr_b64 v[120:123], v76 offset1:2                // 000000002630: d9dc0200 7800004c
	ds_load_2addr_b64 v[128:131], v73 offset0:4 offset1:6      // 000000002638: d9dc0604 80000049
	ds_load_2addr_b64 v[132:135], v85 offset0:4 offset1:6      // 000000002640: d9dc0604 84000055
	ds_load_2addr_b64 v[136:139], v86 offset0:4 offset1:6      // 000000002648: d9dc0604 88000056
	ds_load_2addr_b64 v[140:143], v74 offset0:4 offset1:6      // 000000002650: d9dc0604 8c00004a
	ds_load_2addr_b64 v[144:147], v75 offset0:4 offset1:6      // 000000002658: d9dc0604 9000004b
	ds_load_2addr_b64 v[148:151], v76 offset0:4 offset1:6      // 000000002660: d9dc0604 9400004c
	s_wait_dscnt 0xa                                           // 000000002668: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[91:92], v[95:96], v[56:63]// 00000000266c: cc464038 1ce2bf5b
	s_wait_dscnt 0x9                                           // 000000002674: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[91:92], v[103:104], v[24:31]// 000000002678: cc464018 1c62cf5b
	s_wait_dscnt 0x8                                           // 000000002680: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[107:108], v[95:96], v[48:55]// 000000002684: cc464030 1cc2bf6b
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[107:108], v[103:104], v[16:23]// 00000000268c: cc464010 1c42cf6b
	s_wait_dscnt 0x7                                           // 000000002694: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[116:117], v[95:96], v[40:47]// 000000002698: cc464028 1ca2bf74
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[116:117], v[103:104], v[8:15]// 0000000026a0: cc464008 1c22cf74
	s_wait_dscnt 0x6                                           // 0000000026a8: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[120:121], v[95:96], v[32:39]// 0000000026ac: cc464020 1c82bf78
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[120:121], v[103:104], v[0:7]// 0000000026b4: cc464000 1c02cf78
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[93:94], v[97:98], v[56:63]// 0000000026bc: cc464038 1ce2c35d
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[93:94], v[105:106], v[24:31]// 0000000026c4: cc464018 1c62d35d
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[109:110], v[97:98], v[48:55]// 0000000026cc: cc464030 1cc2c36d
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[109:110], v[105:106], v[16:23]// 0000000026d4: cc464010 1c42d36d
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[118:119], v[97:98], v[40:47]// 0000000026dc: cc464028 1ca2c376
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[118:119], v[105:106], v[8:15]// 0000000026e4: cc464008 1c22d376
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[122:123], v[97:98], v[32:39]// 0000000026ec: cc464020 1c82c37a
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[122:123], v[105:106], v[0:7]// 0000000026f4: cc464000 1c02d37a
	s_wait_dscnt 0x4                                           // 0000000026fc: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[128:129], v[132:133], v[56:63]// 000000002700: cc464038 1ce30980
	s_wait_dscnt 0x3                                           // 000000002708: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[128:129], v[136:137], v[24:31]// 00000000270c: cc464018 1c631180
	s_wait_dscnt 0x2                                           // 000000002714: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[140:141], v[132:133], v[48:55]// 000000002718: cc464030 1cc3098c
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[140:141], v[136:137], v[16:23]// 000000002720: cc464010 1c43118c
	s_wait_dscnt 0x1                                           // 000000002728: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[144:145], v[132:133], v[40:47]// 00000000272c: cc464028 1ca30990
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[136:137], v[8:15]// 000000002734: cc464008 1c231190
	s_wait_dscnt 0x0                                           // 00000000273c: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[148:149], v[132:133], v[32:39]// 000000002740: cc464020 1c830994
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[148:149], v[136:137], v[0:7]// 000000002748: cc464000 1c031194
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[130:131], v[134:135], v[56:63]// 000000002750: cc464038 1ce30d82
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[130:131], v[138:139], v[24:31]// 000000002758: cc464018 1c631582
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[142:143], v[134:135], v[48:55]// 000000002760: cc464030 1cc30d8e
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[142:143], v[138:139], v[16:23]// 000000002768: cc464010 1c43158e
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[146:147], v[134:135], v[40:47]// 000000002770: cc464028 1ca30d92
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[146:147], v[138:139], v[8:15]// 000000002778: cc464008 1c231592
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[150:151], v[134:135], v[32:39]// 000000002780: cc464020 1c830d96
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[138:139], v[0:7]// 000000002788: cc464000 1c031596
	s_cbranch_scc1 64995                                       // 000000002790: bfa2fde3 <packed_folded_w4a8+0x420>
	v_or_b32_e32 v68, s26, v71                                 // 000000002794: 38888e1a
	v_or_b32_e32 v125, s14, v70                                // 000000002798: 38fa8c0e
	v_mov_b32_e32 v67, s27                                     // 00000000279c: 7e86021b
	s_load_b64 s[28:29], s[0:1], 0x58                          // 0000000027a0: f4002700 f8000058
	v_mov_b32_e32 v65, s15                                     // 0000000027a8: 7e82020f
	v_or_b32_e32 v66, v115, v68                                // 0000000027ac: 38848973
	v_or_b32_e32 v64, v125, v124                               // 0000000027b0: 3880f97d
	v_mov_b32_e32 v126, s15                                    // 0000000027b4: 7efc020f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000027b8: bf870193
	v_cmp_gt_i64_e64 s2, s[36:37], v[66:67]                    // 0000000027bc: d4540002 02028424
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[64:65]                // 0000000027c4: 7ca88026
	s_wait_alu depctr_va_sdst(0)                               // 0000000027c8: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_2)// 0000000027cc: bf870142
	v_cndmask_b32_e64 v70, 0, v67, s2                          // 0000000027d0: d5010046 000a8680
	v_cndmask_b32_e64 v69, 0, v66, s2                          // 0000000027d8: d5010045 000a8480
	s_wait_alu depctr_va_vcc(0)                                // 0000000027e0: bf88ff9d
	v_dual_cndmask_b32 v72, 0, v64 :: v_dual_cndmask_b32 v71, 0, v65// 0000000027e4: ca528080 48468280
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]                    // 0000000027ec: 3e8a8a82
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000027f0: bf8701a2
	v_add_co_u32 v110, s2, s40, v72                            // 0000000027f4: d700026e 02029028
	s_wait_alu depctr_va_sdst(0)                               // 0000000027fc: bf88f19f
	v_add_co_ci_u32_e64 v111, null, s41, v71, s2               // 000000002800: d5207c6f 000a8e29
	s_wait_kmcnt 0x0                                           // 000000002808: bfc70000
	s_delay_alu instid0(valu_dep_3)                            // 00000000280c: bf870003
	v_add_co_u32 v72, s2, s28, v69                             // 000000002810: d7000248 02028a1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002818: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s29, v70, s2                // 00000000281c: d5207c49 000a8c1d
	global_load_u8 v69, v[110:111], off                        // 000000002824: ee04007c 00000045 0000006e
	global_load_b32 v71, v[72:73], off                         // 000000002830: ee05007c 00000047 00000048
	s_wait_loadcnt 0x1                                         // 00000000283c: bfc00001
	v_dual_mov_b32 v69, s27 :: v_dual_lshlrev_b32 v70, 23, v69 // 000000002840: ca22001b 45468a97
	s_wait_loadcnt 0x0                                         // 000000002848: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000284c: bf870091
	v_mul_f32_e32 v74, v71, v70                                // 000000002850: 10948d47
	v_cmp_class_f32_e64 s2, v74, 0x198                         // 000000002854: d47e0002 0201ff4a 00000198
	v_mul_f32_e32 v88, v56, v74                                // 000000002860: 10b09538
	s_xor_b32 s2, s2, -1                                       // 000000002864: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002868: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000286c: be832002
	s_cbranch_execnz 3967                                      // 000000002870: bfa60f7f <packed_folded_w4a8+0x4b70>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002874: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002878: 8c7e037e
	v_or_b32_e32 v116, 1, v115                                 // 00000000287c: 38e8e681
	v_mov_b32_e32 v75, v69                                     // 000000002880: 7e960345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002884: bf870092
	v_or_b32_e32 v74, v116, v68                                // 000000002888: 38948974
	v_cmp_gt_i64_e64 s2, s[36:37], v[74:75]                    // 00000000288c: d4540002 02029424
	s_wait_alu depctr_va_sdst(0)                               // 000000002894: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002898: bf8700a1
	v_cndmask_b32_e64 v75, 0, v75, s2                          // 00000000289c: d501004b 000a9680
	v_cndmask_b32_e64 v74, 0, v74, s2                          // 0000000028a4: d501004a 000a9480
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]                    // 0000000028ac: 3e949482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000028b0: bf870121
	v_add_co_u32 v74, s2, s28, v74                             // 0000000028b4: d700024a 0202941c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028bc: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s29, v75, s2                // 0000000028c0: d5207c4b 000a961d
	global_load_b32 v56, v[74:75], off                         // 0000000028c8: ee05007c 00000038 0000004a
	s_wait_loadcnt 0x0                                         // 0000000028d4: bfc00000
	v_mul_f32_e32 v71, v56, v70                                // 0000000028d8: 108e8d38
	s_delay_alu instid0(valu_dep_1)                            // 0000000028dc: bf870001
	v_cmp_class_f32_e64 s2, v71, 0x198                         // 0000000028e0: d47e0002 0201ff47 00000198
	v_mul_f32_e32 v89, v57, v71                                // 0000000028ec: 10b28f39
	s_xor_b32 s2, s2, -1                                       // 0000000028f0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028f4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000028f8: be832002
	s_cbranch_execnz 3950                                      // 0000000028fc: bfa60f6e <packed_folded_w4a8+0x4bb8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002900: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002904: 8c7e037e
	v_or_b32_e32 v117, 2, v115                                 // 000000002908: 38eae682
	v_mov_b32_e32 v57, v69                                     // 00000000290c: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002910: bf870092
	v_or_b32_e32 v56, v117, v68                                // 000000002914: 38708975
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002918: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002920: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002924: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002928: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002930: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002938: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000293c: bf870121
	v_add_co_u32 v76, s2, s28, v56                             // 000000002940: d700024c 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002948: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s29, v57, s2                // 00000000294c: d5207c4d 000a721d
	global_load_b32 v56, v[76:77], off                         // 000000002954: ee05007c 00000038 0000004c
	s_wait_loadcnt 0x0                                         // 000000002960: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002964: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002968: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 00000000296c: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v91, v58, v57                                // 000000002978: 10b6733a
	s_xor_b32 s2, s2, -1                                       // 00000000297c: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002980: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002984: be832002
	s_cbranch_execnz 3933                                      // 000000002988: bfa60f5d <packed_folded_w4a8+0x4c00>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000298c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002990: 8c7e037e
	v_or_b32_e32 v128, 3, v115                                 // 000000002994: 3900e683
	v_mov_b32_e32 v57, v69                                     // 000000002998: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 00000000299c: bf870092
	v_or_b32_e32 v56, v128, v68                                // 0000000029a0: 38708980
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 0000000029a4: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 0000000029ac: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000029b0: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 0000000029b4: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 0000000029bc: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 0000000029c4: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000029c8: bf870121
	v_add_co_u32 v78, s2, s28, v56                             // 0000000029cc: d700024e 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 0000000029d4: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s29, v57, s2                // 0000000029d8: d5207c4f 000a721d
	global_load_b32 v56, v[78:79], off                         // 0000000029e0: ee05007c 00000038 0000004e
	s_wait_loadcnt 0x0                                         // 0000000029ec: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 0000000029f0: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 0000000029f4: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 0000000029f8: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v92, v59, v57                                // 000000002a04: 10b8733b
	s_xor_b32 s2, s2, -1                                       // 000000002a08: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a0c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002a10: be832002
	s_cbranch_execnz 3916                                      // 000000002a14: bfa60f4c <packed_folded_w4a8+0x4c48>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002a1c: 8c7e037e
	v_or_b32_e32 v129, 4, v115                                 // 000000002a20: 3902e684
	v_mov_b32_e32 v57, v69                                     // 000000002a24: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002a28: bf870092
	v_or_b32_e32 v56, v129, v68                                // 000000002a2c: 38708981
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002a30: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002a38: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002a3c: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002a40: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002a48: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002a50: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a54: bf870121
	v_add_co_u32 v80, s2, s28, v56                             // 000000002a58: d7000250 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a60: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s29, v57, s2                // 000000002a64: d5207c51 000a721d
	global_load_b32 v56, v[80:81], off                         // 000000002a6c: ee05007c 00000038 00000050
	s_wait_loadcnt 0x0                                         // 000000002a78: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002a7c: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002a80: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002a84: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v93, v60, v57                                // 000000002a90: 10ba733c
	s_xor_b32 s2, s2, -1                                       // 000000002a94: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a98: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002a9c: be832002
	s_cbranch_execnz 3899                                      // 000000002aa0: bfa60f3b <packed_folded_w4a8+0x4c90>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002aa4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002aa8: 8c7e037e
	v_or_b32_e32 v130, 5, v115                                 // 000000002aac: 3904e685
	v_mov_b32_e32 v57, v69                                     // 000000002ab0: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002ab4: bf870092
	v_or_b32_e32 v56, v130, v68                                // 000000002ab8: 38708982
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002abc: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002ac4: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002ac8: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002acc: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002ad4: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002adc: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002ae0: bf870121
	v_add_co_u32 v82, s2, s28, v56                             // 000000002ae4: d7000252 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002aec: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s29, v57, s2                // 000000002af0: d5207c53 000a721d
	global_load_b32 v56, v[82:83], off                         // 000000002af8: ee05007c 00000038 00000052
	s_wait_loadcnt 0x0                                         // 000000002b04: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002b08: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002b0c: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b10: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v94, v61, v57                                // 000000002b1c: 10bc733d
	s_xor_b32 s2, s2, -1                                       // 000000002b20: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b24: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b28: be832002
	s_cbranch_execnz 3882                                      // 000000002b2c: bfa60f2a <packed_folded_w4a8+0x4cd8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b34: 8c7e037e
	v_or_b32_e32 v131, 6, v115                                 // 000000002b38: 3906e686
	v_mov_b32_e32 v57, v69                                     // 000000002b3c: 7e720345
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002b40: bf870092
	v_or_b32_e32 v56, v131, v68                                // 000000002b44: 38708983
	v_cmp_gt_i64_e64 s2, s[36:37], v[56:57]                    // 000000002b48: d4540002 02027024
	s_wait_alu depctr_va_sdst(0)                               // 000000002b50: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002b54: bf8700a1
	v_cndmask_b32_e64 v57, 0, v57, s2                          // 000000002b58: d5010039 000a7280
	v_cndmask_b32_e64 v56, 0, v56, s2                          // 000000002b60: d5010038 000a7080
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002b68: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b6c: bf870121
	v_add_co_u32 v84, s2, s28, v56                             // 000000002b70: d7000254 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002b78: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s29, v57, s2                // 000000002b7c: d5207c55 000a721d
	global_load_b32 v56, v[84:85], off                         // 000000002b84: ee05007c 00000038 00000054
	s_wait_loadcnt 0x0                                         // 000000002b90: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002b94: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002b98: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b9c: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v95, v62, v57                                // 000000002ba8: 10be733e
	s_xor_b32 s2, s2, -1                                       // 000000002bac: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bb0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002bb4: be832002
	s_cbranch_execnz 3865                                      // 000000002bb8: bfa60f19 <packed_folded_w4a8+0x4d20>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bbc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002bc0: 8c7e037e
	v_or_b32_e32 v132, 7, v115                                 // 000000002bc4: 3908e687
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002bc8: bf870091
	v_or_b32_e32 v68, v132, v68                                // 000000002bcc: 38888984
	v_cmp_gt_i64_e64 s2, s[36:37], v[68:69]                    // 000000002bd0: d4540002 02028824
	s_wait_alu depctr_va_sdst(0)                               // 000000002bd8: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000002bdc: bf8700a1
	v_cndmask_b32_e64 v57, 0, v69, s2                          // 000000002be0: d5010039 000a8a80
	v_cndmask_b32_e64 v56, 0, v68, s2                          // 000000002be8: d5010038 000a8880
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002bf0: 3e707082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bf4: bf870121
	v_add_co_u32 v86, s2, s28, v56                             // 000000002bf8: d7000256 0202701c
	s_wait_alu depctr_va_sdst(0)                               // 000000002c00: bf88f19f
	v_add_co_ci_u32_e64 v87, null, s29, v57, s2                // 000000002c04: d5207c57 000a721d
	global_load_b32 v56, v[86:87], off                         // 000000002c0c: ee05007c 00000038 00000056
	s_wait_loadcnt 0x0                                         // 000000002c18: bfc00000
	v_mul_f32_e32 v57, v56, v70                                // 000000002c1c: 10728d38
	s_delay_alu instid0(valu_dep_1)                            // 000000002c20: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002c24: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v96, v63, v57                                // 000000002c30: 10c0733f
	s_xor_b32 s2, s2, -1                                       // 000000002c34: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c38: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002c3c: be832002
	s_cbranch_execnz 3849                                      // 000000002c40: bfa60f09 <packed_folded_w4a8+0x4d68>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002c48: 8c7e037e
	s_load_b64 s[34:35], s[0:1], 0xa8                          // 000000002c4c: f4002880 f80000a8
	v_mul_lo_u32 v58, s39, v66                                 // 000000002c54: d72c003a 02028427
	v_mul_lo_u32 v59, s38, v67                                 // 000000002c5c: d72c003b 02028626
	v_mad_co_u64_u32 v[56:57], null, s38, v66, 0               // 000000002c64: d6fe7c38 02028426
	v_sub_co_u32 v68, s0, s36, v66                             // 000000002c6c: d7010044 02028424
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_4)// 000000002c74: bf870221
	v_sub_co_ci_u32_e64 v69, null, s37, v67, s0                // 000000002c78: d5217c45 00028625
	v_lshlrev_b64_e32 v[118:119], 1, v[64:65]                  // 000000002c80: 3eec8081
	v_add3_u32 v57, v57, v59, v58                              // 000000002c84: d6550039 04ea7739
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002c8c: bf870113
	v_cmp_lt_i64_e64 s0, 0, v[68:69]                           // 000000002c90: d4510000 02028880
	v_lshlrev_b64_e32 v[70:71], 1, v[56:57]                    // 000000002c98: 3e8c7081
	s_and_b32 s1, s0, vcc_lo                                   // 000000002c9c: 8b016a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ca0: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000002ca4: be822001
	s_cbranch_execz 28                                         // 000000002ca8: bfa5001c <packed_folded_w4a8+0x121c>
	v_bfe_u32 v56, v88, 16, 1                                  // 000000002cac: d6100038 02052158
	s_wait_kmcnt 0x0                                           // 000000002cb4: bfc70000
	v_add_co_u32 v57, s1, s34, v70                             // 000000002cb8: d7000139 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc0: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s35, v71, s1                // 000000002cc4: d5207c3a 00068e23
	v_add3_u32 v59, v56, v88, 0x7fff                           // 000000002ccc: d655003b 03feb138 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002cd8: bf870003
	v_add_co_u32 v56, s1, v57, v118                            // 000000002cdc: d7000138 0202ed39
	v_or_b32_e32 v60, 0x400000, v88                            // 000000002ce4: 3878b0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002cec: bf88f19f
	v_add_co_ci_u32_e64 v57, null, v58, v119, s1               // 000000002cf0: d5207c39 0006ef3a
	v_cmp_u_f32_e64 s1, v88, v88                               // 000000002cf8: d4180001 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 000000002d00: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d04: bf870001
	v_cndmask_b32_e64 v58, v59, v60, s1                        // 000000002d08: d501003a 0006793b
	global_store_d16_hi_b16 v[56:57], v58, off                 // 000000002d10: ee09407c 1d000000 00000038
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d1c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002d20: 8c7e027e
	v_add_co_u32 v58, s1, s38, v64                             // 000000002d24: d700013a 02028026
	s_wait_alu depctr_va_sdst(0)                               // 000000002d2c: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s39, v65, s1                // 000000002d30: d5207c3b 00068227
	v_cmp_lt_i64_e64 s1, 1, v[68:69]                           // 000000002d38: d4510001 02028881
	s_delay_alu instid0(valu_dep_2)                            // 000000002d40: bf870002
	v_lshlrev_b64_e32 v[56:57], 1, v[58:59]                    // 000000002d44: 3e707481
	s_and_b32 s2, s1, vcc_lo                                   // 000000002d48: 8b026a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d4c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002d50: be832002
	s_cbranch_execz 28                                         // 000000002d54: bfa5001c <packed_folded_w4a8+0x12c8>
	v_bfe_u32 v60, v89, 16, 1                                  // 000000002d58: d610003c 02052159
	s_wait_kmcnt 0x0                                           // 000000002d60: bfc70000
	v_add_co_u32 v61, s2, s34, v70                             // 000000002d64: d700023d 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002d6c: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s35, v71, s2                // 000000002d70: d5207c3e 000a8e23
	v_add3_u32 v63, v60, v89, 0x7fff                           // 000000002d78: d655003f 03feb33c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d84: bf870003
	v_add_co_u32 v60, s2, v61, v56                             // 000000002d88: d700023c 0202713d
	v_or_b32_e32 v64, 0x400000, v89                            // 000000002d90: 3880b2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d98: bf88f19f
	v_add_co_ci_u32_e64 v61, null, v62, v57, s2                // 000000002d9c: d5207c3d 000a733e
	v_cmp_u_f32_e64 s2, v89, v89                               // 000000002da4: d4180002 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 000000002dac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002db0: bf870001
	v_cndmask_b32_e64 v62, v63, v64, s2                        // 000000002db4: d501003e 000a813f
	global_store_d16_hi_b16 v[60:61], v62, off                 // 000000002dbc: ee09407c 1f000000 0000003c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002dcc: 8c7e037e
	v_add_co_u32 v60, s2, v58, s38                             // 000000002dd0: d700023c 02004d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002dd8: bf88f19f
	v_add_co_ci_u32_e64 v61, null, s39, v59, s2                // 000000002ddc: d5207c3d 000a7627
	v_cmp_lt_i64_e64 s2, 2, v[68:69]                           // 000000002de4: d4510002 02028882
	s_delay_alu instid0(valu_dep_2)                            // 000000002dec: bf870002
	v_lshlrev_b64_e32 v[58:59], 1, v[60:61]                    // 000000002df0: 3e747881
	s_and_b32 s3, s2, vcc_lo                                   // 000000002df4: 8b036a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002df8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000002dfc: be842003
	s_cbranch_execz 28                                         // 000000002e00: bfa5001c <packed_folded_w4a8+0x1374>
	v_bfe_u32 v62, v91, 16, 1                                  // 000000002e04: d610003e 0205215b
	s_wait_kmcnt 0x0                                           // 000000002e0c: bfc70000
	v_add_co_u32 v63, s3, s34, v70                             // 000000002e10: d700033f 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002e18: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s35, v71, s3                // 000000002e1c: d5207c40 000e8e23
	v_add3_u32 v65, v62, v91, 0x7fff                           // 000000002e24: d6550041 03feb73e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e30: bf870003
	v_add_co_u32 v62, s3, v63, v58                             // 000000002e34: d700033e 0202753f
	v_or_b32_e32 v66, 0x400000, v91                            // 000000002e3c: 3884b6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e44: bf88f19f
	v_add_co_ci_u32_e64 v63, null, v64, v59, s3                // 000000002e48: d5207c3f 000e7740
	v_cmp_u_f32_e64 s3, v91, v91                               // 000000002e50: d4180003 0202b75b
	s_wait_alu depctr_va_sdst(0)                               // 000000002e58: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e5c: bf870001
	v_cndmask_b32_e64 v64, v65, v66, s3                        // 000000002e60: d5010040 000e8541
	global_store_d16_hi_b16 v[62:63], v64, off                 // 000000002e68: ee09407c 20000000 0000003e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002e78: 8c7e047e
	v_add_co_u32 v62, s3, v60, s38                             // 000000002e7c: d700033e 02004d3c
	s_wait_alu depctr_va_sdst(0)                               // 000000002e84: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s39, v61, s3                // 000000002e88: d5207c3f 000e7a27
	v_cmp_lt_i64_e64 s3, 3, v[68:69]                           // 000000002e90: d4510003 02028883
	s_delay_alu instid0(valu_dep_2)                            // 000000002e98: bf870002
	v_lshlrev_b64_e32 v[60:61], 1, v[62:63]                    // 000000002e9c: 3e787c81
	s_and_b32 s4, s3, vcc_lo                                   // 000000002ea0: 8b046a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ea4: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000002ea8: be852004
	s_cbranch_execz 28                                         // 000000002eac: bfa5001c <packed_folded_w4a8+0x1420>
	v_bfe_u32 v64, v92, 16, 1                                  // 000000002eb0: d6100040 0205215c
	s_wait_kmcnt 0x0                                           // 000000002eb8: bfc70000
	v_add_co_u32 v65, s4, s34, v70                             // 000000002ebc: d7000441 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002ec4: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s35, v71, s4                // 000000002ec8: d5207c42 00128e23
	v_add3_u32 v67, v64, v92, 0x7fff                           // 000000002ed0: d6550043 03feb940 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002edc: bf870003
	v_add_co_u32 v64, s4, v65, v60                             // 000000002ee0: d7000440 02027941
	v_or_b32_e32 v88, 0x400000, v92                            // 000000002ee8: 38b0b8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ef0: bf88f19f
	v_add_co_ci_u32_e64 v65, null, v66, v61, s4                // 000000002ef4: d5207c41 00127b42
	v_cmp_u_f32_e64 s4, v92, v92                               // 000000002efc: d4180004 0202b95c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f04: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f08: bf870001
	v_cndmask_b32_e64 v66, v67, v88, s4                        // 000000002f0c: d5010042 0012b143
	global_store_d16_hi_b16 v[64:65], v66, off                 // 000000002f14: ee09407c 21000000 00000040
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000002f24: 8c7e057e
	v_add_co_u32 v64, s4, v62, s38                             // 000000002f28: d7000440 02004d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000002f30: bf88f19f
	v_add_co_ci_u32_e64 v65, null, s39, v63, s4                // 000000002f34: d5207c41 00127e27
	v_cmp_lt_i64_e64 s4, 4, v[68:69]                           // 000000002f3c: d4510004 02028884
	s_delay_alu instid0(valu_dep_2)                            // 000000002f44: bf870002
	v_lshlrev_b64_e32 v[62:63], 1, v[64:65]                    // 000000002f48: 3e7c8081
	s_and_b32 s5, s4, vcc_lo                                   // 000000002f4c: 8b056a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f50: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000002f54: be862005
	s_cbranch_execz 28                                         // 000000002f58: bfa5001c <packed_folded_w4a8+0x14cc>
	v_bfe_u32 v66, v93, 16, 1                                  // 000000002f5c: d6100042 0205215d
	s_wait_kmcnt 0x0                                           // 000000002f64: bfc70000
	v_add_co_u32 v67, s5, s34, v70                             // 000000002f68: d7000543 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002f70: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s35, v71, s5                // 000000002f74: d5207c58 00168e23
	v_add3_u32 v89, v66, v93, 0x7fff                           // 000000002f7c: d6550059 03febb42 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f88: bf870003
	v_add_co_u32 v66, s5, v67, v62                             // 000000002f8c: d7000542 02027d43
	v_or_b32_e32 v91, 0x400000, v93                            // 000000002f94: 38b6baff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f9c: bf88f19f
	v_add_co_ci_u32_e64 v67, null, v88, v63, s5                // 000000002fa0: d5207c43 00167f58
	v_cmp_u_f32_e64 s5, v93, v93                               // 000000002fa8: d4180005 0202bb5d
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fb4: bf870001
	v_cndmask_b32_e64 v88, v89, v91, s5                        // 000000002fb8: d5010058 0016b759
	global_store_d16_hi_b16 v[66:67], v88, off                 // 000000002fc0: ee09407c 2c000000 00000042
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fcc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002fd0: 8c7e067e
	v_add_co_u32 v66, s5, v64, s38                             // 000000002fd4: d7000542 02004d40
	s_wait_alu depctr_va_sdst(0)                               // 000000002fdc: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s39, v65, s5                // 000000002fe0: d5207c43 00168227
	v_cmp_lt_i64_e64 s5, 5, v[68:69]                           // 000000002fe8: d4510005 02028885
	s_delay_alu instid0(valu_dep_2)                            // 000000002ff0: bf870002
	v_lshlrev_b64_e32 v[64:65], 1, v[66:67]                    // 000000002ff4: 3e808481
	s_and_b32 s6, s5, vcc_lo                                   // 000000002ff8: 8b066a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ffc: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000003000: be872006
	s_cbranch_execz 28                                         // 000000003004: bfa5001c <packed_folded_w4a8+0x1578>
	v_bfe_u32 v88, v94, 16, 1                                  // 000000003008: d6100058 0205215e
	s_wait_kmcnt 0x0                                           // 000000003010: bfc70000
	v_add_co_u32 v89, s6, s34, v70                             // 000000003014: d7000659 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 00000000301c: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s35, v71, s6                // 000000003020: d5207c5b 001a8e23
	v_add3_u32 v92, v88, v94, 0x7fff                           // 000000003028: d655005c 03febd58 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003034: bf870003
	v_add_co_u32 v88, s6, v89, v64                             // 000000003038: d7000658 02028159
	v_or_b32_e32 v93, 0x400000, v94                            // 000000003040: 38babcff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003048: bf88f19f
	v_add_co_ci_u32_e64 v89, null, v91, v65, s6                // 00000000304c: d5207c59 001a835b
	v_cmp_u_f32_e64 s6, v94, v94                               // 000000003054: d4180006 0202bd5e
	s_wait_alu depctr_va_sdst(0)                               // 00000000305c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003060: bf870001
	v_cndmask_b32_e64 v91, v92, v93, s6                        // 000000003064: d501005b 001abb5c
	global_store_d16_hi_b16 v[88:89], v91, off                 // 00000000306c: ee09407c 2d800000 00000058
	s_wait_alu depctr_sa_sdst(0)                               // 000000003078: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000307c: 8c7e077e
	v_add_co_u32 v88, s6, v66, s38                             // 000000003080: d7000658 02004d42
	s_wait_alu depctr_va_sdst(0)                               // 000000003088: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s39, v67, s6                // 00000000308c: d5207c59 001a8627
	v_cmp_lt_i64_e64 s6, 6, v[68:69]                           // 000000003094: d4510006 02028886
	s_delay_alu instid0(valu_dep_2)                            // 00000000309c: bf870002
	v_lshlrev_b64_e32 v[66:67], 1, v[88:89]                    // 0000000030a0: 3e84b081
	s_and_b32 s7, s6, vcc_lo                                   // 0000000030a4: 8b076a06
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a8: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 0000000030ac: be882007
	s_cbranch_execz 28                                         // 0000000030b0: bfa5001c <packed_folded_w4a8+0x1624>
	v_bfe_u32 v91, v95, 16, 1                                  // 0000000030b4: d610005b 0205215f
	s_wait_kmcnt 0x0                                           // 0000000030bc: bfc70000
	v_add_co_u32 v92, s7, s34, v70                             // 0000000030c0: d700075c 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c8: bf88f19f
	v_add_co_ci_u32_e64 v93, null, s35, v71, s7                // 0000000030cc: d5207c5d 001e8e23
	v_add3_u32 v94, v91, v95, 0x7fff                           // 0000000030d4: d655005e 03febf5b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030e0: bf870003
	v_add_co_u32 v91, s7, v92, v66                             // 0000000030e4: d700075b 0202855c
	v_or_b32_e32 v97, 0x400000, v95                            // 0000000030ec: 38c2beff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f4: bf88f19f
	v_add_co_ci_u32_e64 v92, null, v93, v67, s7                // 0000000030f8: d5207c5c 001e875d
	v_cmp_u_f32_e64 s7, v95, v95                               // 000000003100: d4180007 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 000000003108: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000310c: bf870001
	v_cndmask_b32_e64 v93, v94, v97, s7                        // 000000003110: d501005d 001ec35e
	global_store_d16_hi_b16 v[91:92], v93, off                 // 000000003118: ee09407c 2e800000 0000005b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003124: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003128: 8c7e087e
	v_add_co_u32 v88, s7, v88, s38                             // 00000000312c: d7000758 02004d58
	s_wait_alu depctr_va_sdst(0)                               // 000000003134: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s39, v89, s7                // 000000003138: d5207c59 001eb227
	v_cmp_lt_i64_e64 s7, 7, v[68:69]                           // 000000003140: d4510007 02028887
	s_delay_alu instid0(valu_dep_2)                            // 000000003148: bf870002
	v_lshlrev_b64_e32 v[68:69], 1, v[88:89]                    // 00000000314c: 3e88b081
	s_and_b32 s8, s7, vcc_lo                                   // 000000003150: 8b086a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003154: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003158: be892008
	s_cbranch_execz 28                                         // 00000000315c: bfa5001c <packed_folded_w4a8+0x16d0>
	v_bfe_u32 v88, v96, 16, 1                                  // 000000003160: d6100058 02052160
	s_wait_kmcnt 0x0                                           // 000000003168: bfc70000
	v_add_co_u32 v89, s8, s34, v70                             // 00000000316c: d7000859 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000003174: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s35, v71, s8                // 000000003178: d5207c5b 00228e23
	v_add3_u32 v92, v88, v96, 0x7fff                           // 000000003180: d655005c 03fec158 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000318c: bf870003
	v_add_co_u32 v88, s8, v89, v68                             // 000000003190: d7000858 02028959
	v_or_b32_e32 v93, 0x400000, v96                            // 000000003198: 38bac0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a0: bf88f19f
	v_add_co_ci_u32_e64 v89, null, v91, v69, s8                // 0000000031a4: d5207c59 00228b5b
	v_cmp_u_f32_e64 s8, v96, v96                               // 0000000031ac: d4180008 0202c160
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031b8: bf870001
	v_cndmask_b32_e64 v91, v92, v93, s8                        // 0000000031bc: d501005b 0022bb5c
	global_store_d16_hi_b16 v[88:89], v91, off                 // 0000000031c4: ee09407c 2d800000 00000058
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000031d4: 8c7e097e
	v_or_b32_e32 v98, s26, v90                                 // 0000000031d8: 38c4b41a
	v_mov_b32_e32 v101, s27                                    // 0000000031dc: 7eca021b
	v_mov_b32_e32 v99, s27                                     // 0000000031e0: 7ec6021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000031e4: bf870093
	v_or_b32_e32 v100, v98, v115                               // 0000000031e8: 38c8e762
	v_cmp_gt_i64_e64 s8, s[36:37], v[100:101]                  // 0000000031ec: d4540008 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f4: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000031f8: bf8700a1
	v_cndmask_b32_e64 v89, 0, v101, s8                         // 0000000031fc: d5010059 0022ca80
	v_cndmask_b32_e64 v88, 0, v100, s8                         // 000000003204: d5010058 0022c880
	v_lshlrev_b64_e32 v[88:89], 2, v[88:89]                    // 00000000320c: 3eb0b082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003210: bf870121
	v_add_co_u32 v88, s8, s28, v88                             // 000000003214: d7000858 0202b01c
	s_wait_alu depctr_va_sdst(0)                               // 00000000321c: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s29, v89, s8                // 000000003220: d5207c59 0022b21d
	global_load_u8 v91, v[110:111], off                        // 000000003228: ee04007c 0000005b 0000006e
	global_load_b32 v90, v[88:89], off                         // 000000003234: ee05007c 0000005a 00000058
	s_wait_loadcnt 0x1                                         // 000000003240: bfc00001
	v_lshlrev_b32_e32 v106, 23, v91                            // 000000003244: 30d4b697
	s_wait_loadcnt 0x0                                         // 000000003248: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000324c: bf870091
	v_mul_f32_e32 v91, v90, v106                               // 000000003250: 10b6d55a
	v_cmp_class_f32_e64 s8, v91, 0x198                         // 000000003254: d47e0008 0201ff5b 00000198
	v_mul_f32_e32 v103, v48, v91                               // 000000003260: 10ceb730
	s_xor_b32 s8, s8, -1                                       // 000000003264: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003268: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000326c: be892008
	s_cbranch_execnz 3471                                      // 000000003270: bfa60d8f <packed_folded_w4a8+0x4db0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003274: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003278: 8c7e097e
	v_or_b32_e32 v90, v116, v98                                // 00000000327c: 38b4c574
	v_mov_b32_e32 v91, v99                                     // 000000003280: 7eb60363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003284: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[90:91]                    // 000000003288: d4540008 0202b424
	s_wait_alu depctr_va_sdst(0)                               // 000000003290: bf88f19f
	v_cndmask_b32_e64 v91, 0, v91, s8                          // 000000003294: d501005b 0022b680
	v_cndmask_b32_e64 v90, 0, v90, s8                          // 00000000329c: d501005a 0022b480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000032a4: bf870091
	v_lshlrev_b64_e32 v[90:91], 2, v[90:91]                    // 0000000032a8: 3eb4b482
	v_add_co_u32 v90, s8, s28, v90                             // 0000000032ac: d700085a 0202b41c
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000032b8: bf8700c2
	v_add_co_ci_u32_e64 v91, null, s29, v91, s8                // 0000000032bc: d5207c5b 0022b61d
	global_load_b32 v48, v[90:91], off                         // 0000000032c4: ee05007c 00000030 0000005a
	s_wait_loadcnt 0x0                                         // 0000000032d0: bfc00000
	v_mul_f32_e32 v92, v48, v106                               // 0000000032d4: 10b8d530
	v_cmp_class_f32_e64 s8, v92, 0x198                         // 0000000032d8: d47e0008 0201ff5c 00000198
	v_mul_f32_e32 v104, v49, v92                               // 0000000032e4: 10d0b931
	s_xor_b32 s8, s8, -1                                       // 0000000032e8: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032ec: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000032f0: be892008
	s_cbranch_execnz 3456                                      // 0000000032f4: bfa60d80 <packed_folded_w4a8+0x4df8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000032fc: 8c7e097e
	v_or_b32_e32 v48, v117, v98                                // 000000003300: 3860c575
	v_mov_b32_e32 v49, v99                                     // 000000003304: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003308: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 00000000330c: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003314: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 000000003318: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 000000003320: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003328: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 00000000332c: 3e606082
	v_add_co_u32 v92, s8, s28, v48                             // 000000003330: d700085c 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 000000003338: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000333c: bf8700c2
	v_add_co_ci_u32_e64 v93, null, s29, v49, s8                // 000000003340: d5207c5d 0022621d
	global_load_b32 v48, v[92:93], off                         // 000000003348: ee05007c 00000030 0000005c
	s_wait_loadcnt 0x0                                         // 000000003354: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003358: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 00000000335c: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v105, v50, v49                               // 000000003368: 10d26332
	s_xor_b32 s8, s8, -1                                       // 00000000336c: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003370: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003374: be892008
	s_cbranch_execnz 3441                                      // 000000003378: bfa60d71 <packed_folded_w4a8+0x4e40>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000337c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003380: 8c7e097e
	v_or_b32_e32 v48, v128, v98                                // 000000003384: 3860c580
	v_mov_b32_e32 v49, v99                                     // 000000003388: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 00000000338c: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003390: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003398: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 00000000339c: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 0000000033a4: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000033ac: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000033b0: 3e606082
	v_add_co_u32 v94, s8, s28, v48                             // 0000000033b4: d700085e 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000033bc: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000033c0: bf8700c2
	v_add_co_ci_u32_e64 v95, null, s29, v49, s8                // 0000000033c4: d5207c5f 0022621d
	global_load_b32 v48, v[94:95], off                         // 0000000033cc: ee05007c 00000030 0000005e
	s_wait_loadcnt 0x0                                         // 0000000033d8: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 0000000033dc: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 0000000033e0: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v107, v51, v49                               // 0000000033ec: 10d66333
	s_xor_b32 s8, s8, -1                                       // 0000000033f0: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033f4: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000033f8: be892008
	s_cbranch_execnz 3426                                      // 0000000033fc: bfa60d62 <packed_folded_w4a8+0x4e88>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003400: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003404: 8c7e097e
	v_or_b32_e32 v48, v129, v98                                // 000000003408: 3860c581
	v_mov_b32_e32 v49, v99                                     // 00000000340c: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003410: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003414: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 00000000341c: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 000000003420: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 000000003428: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003430: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 000000003434: 3e606082
	v_add_co_u32 v50, s8, s28, v48                             // 000000003438: d7000832 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 000000003440: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003444: bf8700c2
	v_add_co_ci_u32_e64 v51, null, s29, v49, s8                // 000000003448: d5207c33 0022621d
	global_load_b32 v48, v[50:51], off                         // 000000003450: ee05007c 00000030 00000032
	s_wait_loadcnt 0x0                                         // 00000000345c: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003460: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 000000003464: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v108, v52, v49                               // 000000003470: 10d86334
	s_xor_b32 s8, s8, -1                                       // 000000003474: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003478: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000347c: be892008
	s_cbranch_execnz 3411                                      // 000000003480: bfa60d53 <packed_folded_w4a8+0x4ed0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003484: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003488: 8c7e097e
	v_or_b32_e32 v48, v130, v98                                // 00000000348c: 3860c582
	v_mov_b32_e32 v49, v99                                     // 000000003490: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003494: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 000000003498: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 0000000034a0: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 0000000034a4: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 0000000034ac: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000034b4: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000034b8: 3e606082
	v_add_co_u32 v96, s8, s28, v48                             // 0000000034bc: d7000860 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000034c8: bf8700c2
	v_add_co_ci_u32_e64 v97, null, s29, v49, s8                // 0000000034cc: d5207c61 0022621d
	global_load_b32 v48, v[96:97], off                         // 0000000034d4: ee05007c 00000030 00000060
	s_wait_loadcnt 0x0                                         // 0000000034e0: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 0000000034e4: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 0000000034e8: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v109, v53, v49                               // 0000000034f4: 10da6335
	s_xor_b32 s8, s8, -1                                       // 0000000034f8: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034fc: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003500: be892008
	s_cbranch_execnz 3396                                      // 000000003504: bfa60d44 <packed_folded_w4a8+0x4f18>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003508: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000350c: 8c7e097e
	v_or_b32_e32 v48, v131, v98                                // 000000003510: 3860c583
	v_mov_b32_e32 v49, v99                                     // 000000003514: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003518: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[48:49]                    // 00000000351c: d4540008 02026024
	s_wait_alu depctr_va_sdst(0)                               // 000000003524: bf88f19f
	v_cndmask_b32_e64 v49, 0, v49, s8                          // 000000003528: d5010031 00226280
	v_cndmask_b32_e64 v48, 0, v48, s8                          // 000000003530: d5010030 00226080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003538: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 00000000353c: 3e606082
	v_add_co_u32 v52, s8, s28, v48                             // 000000003540: d7000834 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 000000003548: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000354c: bf8700c2
	v_add_co_ci_u32_e64 v53, null, s29, v49, s8                // 000000003550: d5207c35 0022621d
	global_load_b32 v48, v[52:53], off                         // 000000003558: ee05007c 00000030 00000034
	s_wait_loadcnt 0x0                                         // 000000003564: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 000000003568: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 00000000356c: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v112, v54, v49                               // 000000003578: 10e06336
	s_xor_b32 s8, s8, -1                                       // 00000000357c: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003580: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003584: be892008
	s_cbranch_execnz 3381                                      // 000000003588: bfa60d35 <packed_folded_w4a8+0x4f60>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000358c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003590: 8c7e097e
	v_or_b32_e32 v98, v132, v98                                // 000000003594: 38c4c584
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003598: bf8700a1
	v_cmp_gt_i64_e64 s8, s[36:37], v[98:99]                    // 00000000359c: d4540008 0202c424
	s_wait_alu depctr_va_sdst(0)                               // 0000000035a4: bf88f19f
	v_cndmask_b32_e64 v49, 0, v99, s8                          // 0000000035a8: d5010031 0022c680
	v_cndmask_b32_e64 v48, 0, v98, s8                          // 0000000035b0: d5010030 0022c480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035b8: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000035bc: 3e606082
	v_add_co_u32 v98, s8, s28, v48                             // 0000000035c0: d7000862 0202601c
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000035cc: bf8700c2
	v_add_co_ci_u32_e64 v99, null, s29, v49, s8                // 0000000035d0: d5207c63 0022621d
	global_load_b32 v48, v[98:99], off                         // 0000000035d8: ee05007c 00000030 00000062
	s_wait_loadcnt 0x0                                         // 0000000035e4: bfc00000
	v_mul_f32_e32 v49, v48, v106                               // 0000000035e8: 1062d530
	v_cmp_class_f32_e64 s8, v49, 0x198                         // 0000000035ec: d47e0008 0201ff31 00000198
	v_mul_f32_e32 v113, v55, v49                               // 0000000035f8: 10e26337
	s_xor_b32 s8, s8, -1                                       // 0000000035fc: 8d08c108
	s_wait_alu depctr_sa_sdst(0)                               // 000000003600: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003604: be892008
	s_cbranch_execnz 3367                                      // 000000003608: bfa60d27 <packed_folded_w4a8+0x4fa8>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000360c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003610: 8c7e097e
	v_mul_lo_u32 v106, s39, v100                               // 000000003614: d72c006a 0202c827
	v_mul_lo_u32 v120, s38, v101                               // 00000000361c: d72c0078 0202ca26
	v_mad_co_u64_u32 v[48:49], null, s38, v100, 0              // 000000003624: d6fe7c30 0202c826
	v_sub_co_u32 v54, s8, s36, v100                            // 00000000362c: d7010836 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 000000003634: bf88f19f
	v_sub_co_ci_u32_e64 v55, null, s37, v101, s8               // 000000003638: d5217c37 0022ca25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003640: bf870211
	v_cmp_lt_i64_e64 s11, 0, v[54:55]                          // 000000003644: d451000b 02026c80
	v_add3_u32 v49, v49, v120, v106                            // 00000000364c: d6550031 05aaf131
	s_delay_alu instid0(valu_dep_1)                            // 000000003654: bf870001
	v_lshlrev_b64_e32 v[48:49], 1, v[48:49]                    // 000000003658: 3e606081
	s_and_b32 s8, s11, vcc_lo                                  // 00000000365c: 8b086a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003660: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 000000003664: be892008
	s_cbranch_execz 28                                         // 000000003668: bfa5001c <packed_folded_w4a8+0x1bdc>
	v_bfe_u32 v100, v103, 16, 1                                // 00000000366c: d6100064 02052167
	s_wait_kmcnt 0x0                                           // 000000003674: bfc70000
	v_add_co_u32 v101, s8, s34, v48                            // 000000003678: d7000865 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003680: bf88f19f
	v_add_co_ci_u32_e64 v106, null, s35, v49, s8               // 000000003684: d5207c6a 00226223
	v_add3_u32 v120, v100, v103, 0x7fff                        // 00000000368c: d6550078 03fecf64 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003698: bf870003
	v_add_co_u32 v100, s8, v101, v118                          // 00000000369c: d7000864 0202ed65
	v_or_b32_e32 v121, 0x400000, v103                          // 0000000036a4: 38f2ceff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ac: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v106, v119, s8             // 0000000036b0: d5207c65 0022ef6a
	v_cmp_u_f32_e64 s8, v103, v103                             // 0000000036b8: d4180008 0202cf67
	s_wait_alu depctr_va_sdst(0)                               // 0000000036c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036c4: bf870001
	v_cndmask_b32_e64 v103, v120, v121, s8                     // 0000000036c8: d5010067 0022f378
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000036d0: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000036e0: 8c7e097e
	v_cmp_lt_i64_e64 s8, 1, v[54:55]                           // 0000000036e4: d4510008 02026c81
	s_and_b32 s9, s8, vcc_lo                                   // 0000000036ec: 8b096a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036f0: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 0000000036f4: be8a2009
	s_cbranch_execz 28                                         // 0000000036f8: bfa5001c <packed_folded_w4a8+0x1c6c>
	v_bfe_u32 v100, v104, 16, 1                                // 0000000036fc: d6100064 02052168
	s_wait_kmcnt 0x0                                           // 000000003704: bfc70000
	v_add_co_u32 v101, s9, s34, v48                            // 000000003708: d7000965 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003710: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s9               // 000000003714: d5207c67 00266223
	v_add3_u32 v106, v100, v104, 0x7fff                        // 00000000371c: d655006a 03fed164 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003728: bf870003
	v_add_co_u32 v100, s9, v101, v56                           // 00000000372c: d7000964 02027165
	v_or_b32_e32 v120, 0x400000, v104                          // 000000003734: 38f0d0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000373c: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v57, s9              // 000000003740: d5207c65 00267367
	v_cmp_u_f32_e64 s9, v104, v104                             // 000000003748: d4180009 0202d168
	s_wait_alu depctr_va_sdst(0)                               // 000000003750: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003754: bf870001
	v_cndmask_b32_e64 v103, v106, v120, s9                     // 000000003758: d5010067 0026f16a
	global_store_d16_hi_b16 v[100:101], v103, off              // 000000003760: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 00000000376c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003770: 8c7e0a7e
	v_cmp_lt_i64_e64 s9, 2, v[54:55]                           // 000000003774: d4510009 02026c82
	s_and_b32 s10, s9, vcc_lo                                  // 00000000377c: 8b0a6a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000003780: bf88ff9e
	s_and_saveexec_b32 s12, s10                                // 000000003784: be8c200a
	s_cbranch_execz 28                                         // 000000003788: bfa5001c <packed_folded_w4a8+0x1cfc>
	v_bfe_u32 v100, v105, 16, 1                                // 00000000378c: d6100064 02052169
	s_wait_kmcnt 0x0                                           // 000000003794: bfc70000
	v_add_co_u32 v101, s10, s34, v48                           // 000000003798: d7000a65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000037a0: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s10              // 0000000037a4: d5207c67 002a6223
	v_add3_u32 v104, v100, v105, 0x7fff                        // 0000000037ac: d6550068 03fed364 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000037b8: bf870003
	v_add_co_u32 v100, s10, v101, v58                          // 0000000037bc: d7000a64 02027565
	v_or_b32_e32 v106, 0x400000, v105                          // 0000000037c4: 38d4d2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000037cc: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v59, s10             // 0000000037d0: d5207c65 002a7767
	v_cmp_u_f32_e64 s10, v105, v105                            // 0000000037d8: d418000a 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037e4: bf870001
	v_cndmask_b32_e64 v103, v104, v106, s10                    // 0000000037e8: d5010067 002ad568
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000037f0: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003800: 8c7e0c7e
	v_cmp_lt_i64_e64 s10, 3, v[54:55]                          // 000000003804: d451000a 02026c83
	s_and_b32 s12, s10, vcc_lo                                 // 00000000380c: 8b0c6a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003810: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000003814: be8d200c
	s_cbranch_execz 28                                         // 000000003818: bfa5001c <packed_folded_w4a8+0x1d8c>
	v_bfe_u32 v100, v107, 16, 1                                // 00000000381c: d6100064 0205216b
	s_wait_kmcnt 0x0                                           // 000000003824: bfc70000
	v_add_co_u32 v101, s12, s34, v48                           // 000000003828: d7000c65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003830: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s12              // 000000003834: d5207c67 00326223
	v_add3_u32 v104, v100, v107, 0x7fff                        // 00000000383c: d6550068 03fed764 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003848: bf870003
	v_add_co_u32 v100, s12, v101, v60                          // 00000000384c: d7000c64 02027965
	v_or_b32_e32 v105, 0x400000, v107                          // 000000003854: 38d2d6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000385c: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v61, s12             // 000000003860: d5207c65 00327b67
	v_cmp_u_f32_e64 s12, v107, v107                            // 000000003868: d418000c 0202d76b
	s_wait_alu depctr_va_sdst(0)                               // 000000003870: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003874: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s12                    // 000000003878: d5010067 0032d368
	global_store_d16_hi_b16 v[100:101], v103, off              // 000000003880: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 00000000388c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003890: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 4, v[54:55]                          // 000000003894: d451000c 02026c84
	s_and_b32 s13, s12, vcc_lo                                 // 00000000389c: 8b0d6a0c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038a0: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 0000000038a4: be8e200d
	s_cbranch_execz 28                                         // 0000000038a8: bfa5001c <packed_folded_w4a8+0x1e1c>
	v_bfe_u32 v100, v108, 16, 1                                // 0000000038ac: d6100064 0205216c
	s_wait_kmcnt 0x0                                           // 0000000038b4: bfc70000
	v_add_co_u32 v101, s13, s34, v48                           // 0000000038b8: d7000d65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000038c0: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s13              // 0000000038c4: d5207c67 00366223
	v_add3_u32 v104, v100, v108, 0x7fff                        // 0000000038cc: d6550068 03fed964 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000038d8: bf870003
	v_add_co_u32 v100, s13, v101, v62                          // 0000000038dc: d7000d64 02027d65
	v_or_b32_e32 v105, 0x400000, v108                          // 0000000038e4: 38d2d8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000038ec: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v63, s13             // 0000000038f0: d5207c65 00367f67
	v_cmp_u_f32_e64 s13, v108, v108                            // 0000000038f8: d418000d 0202d96c
	s_wait_alu depctr_va_sdst(0)                               // 000000003900: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003904: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s13                    // 000000003908: d5010067 0036d368
	global_store_d16_hi_b16 v[100:101], v103, off              // 000000003910: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 00000000391c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 000000003920: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 5, v[54:55]                          // 000000003924: d451000d 02026c85
	s_and_b32 s14, s13, vcc_lo                                 // 00000000392c: 8b0e6a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003930: bf88ff9e
	s_and_saveexec_b32 s15, s14                                // 000000003934: be8f200e
	s_cbranch_execz 28                                         // 000000003938: bfa5001c <packed_folded_w4a8+0x1eac>
	v_bfe_u32 v100, v109, 16, 1                                // 00000000393c: d6100064 0205216d
	s_wait_kmcnt 0x0                                           // 000000003944: bfc70000
	v_add_co_u32 v101, s14, s34, v48                           // 000000003948: d7000e65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003950: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s14              // 000000003954: d5207c67 003a6223
	v_add3_u32 v104, v100, v109, 0x7fff                        // 00000000395c: d6550068 03fedb64 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003968: bf870003
	v_add_co_u32 v100, s14, v101, v64                          // 00000000396c: d7000e64 02028165
	v_or_b32_e32 v105, 0x400000, v109                          // 000000003974: 38d2daff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v65, s14             // 000000003980: d5207c65 003a8367
	v_cmp_u_f32_e64 s14, v109, v109                            // 000000003988: d418000e 0202db6d
	s_wait_alu depctr_va_sdst(0)                               // 000000003990: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003994: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s14                    // 000000003998: d5010067 003ad368
	global_store_d16_hi_b16 v[100:101], v103, off              // 0000000039a0: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000039b0: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 6, v[54:55]                          // 0000000039b4: d451000e 02026c86
	s_and_b32 s15, s14, vcc_lo                                 // 0000000039bc: 8b0f6a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c0: bf88ff9e
	s_and_saveexec_b32 s16, s15                                // 0000000039c4: be90200f
	s_cbranch_execz 28                                         // 0000000039c8: bfa5001c <packed_folded_w4a8+0x1f3c>
	v_bfe_u32 v100, v112, 16, 1                                // 0000000039cc: d6100064 02052170
	s_wait_kmcnt 0x0                                           // 0000000039d4: bfc70000
	v_add_co_u32 v101, s15, s34, v48                           // 0000000039d8: d7000f65 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000039e0: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s35, v49, s15              // 0000000039e4: d5207c67 003e6223
	v_add3_u32 v104, v100, v112, 0x7fff                        // 0000000039ec: d6550068 03fee164 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000039f8: bf870003
	v_add_co_u32 v100, s15, v101, v66                          // 0000000039fc: d7000f64 02028565
	v_or_b32_e32 v105, 0x400000, v112                          // 000000003a04: 38d2e0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a0c: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v103, v67, s15             // 000000003a10: d5207c65 003e8767
	v_cmp_u_f32_e64 s15, v112, v112                            // 000000003a18: d418000f 0202e170
	s_wait_alu depctr_va_sdst(0)                               // 000000003a20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a24: bf870001
	v_cndmask_b32_e64 v103, v104, v105, s15                    // 000000003a28: d5010067 003ed368
	global_store_d16_hi_b16 v[100:101], v103, off              // 000000003a30: ee09407c 33800000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s16                             // 000000003a40: 8c7e107e
	v_cmp_lt_i64_e64 s15, 7, v[54:55]                          // 000000003a44: d451000f 02026c87
	s_and_b32 s16, s15, vcc_lo                                 // 000000003a4c: 8b106a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a50: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003a54: be912010
	s_cbranch_execz 28                                         // 000000003a58: bfa5001c <packed_folded_w4a8+0x1fcc>
	v_bfe_u32 v54, v113, 16, 1                                 // 000000003a5c: d6100036 02052171
	s_wait_kmcnt 0x0                                           // 000000003a64: bfc70000
	v_add_co_u32 v55, s16, s34, v48                            // 000000003a68: d7001037 02026022
	s_wait_alu depctr_va_sdst(0)                               // 000000003a70: bf88f19f
	v_add_co_ci_u32_e64 v100, null, s35, v49, s16              // 000000003a74: d5207c64 00426223
	v_add3_u32 v101, v54, v113, 0x7fff                         // 000000003a7c: d6550065 03fee336 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003a88: bf870003
	v_add_co_u32 v54, s16, v55, v68                            // 000000003a8c: d7001036 02028937
	v_or_b32_e32 v103, 0x400000, v113                          // 000000003a94: 38cee2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a9c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, v100, v69, s16              // 000000003aa0: d5207c37 00428b64
	v_cmp_u_f32_e64 s16, v113, v113                            // 000000003aa8: d4180010 0202e371
	s_wait_alu depctr_va_sdst(0)                               // 000000003ab0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ab4: bf870001
	v_cndmask_b32_e64 v100, v101, v103, s16                    // 000000003ab8: d5010064 0042cf65
	global_store_d16_hi_b16 v[54:55], v100, off                // 000000003ac0: ee09407c 32000000 00000036
	s_wait_alu depctr_sa_sdst(0)                               // 000000003acc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003ad0: 8c7e117e
	v_or_b32_e32 v108, s26, v102                               // 000000003ad4: 38d8cc1a
	v_mov_b32_e32 v113, s27                                    // 000000003ad8: 7ee2021b
	v_mov_b32_e32 v109, s27                                    // 000000003adc: 7eda021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003ae0: bf870093
	v_or_b32_e32 v112, v108, v115                              // 000000003ae4: 38e0e76c
	v_cmp_gt_i64_e64 s16, s[36:37], v[112:113]                 // 000000003ae8: d4540010 0202e024
	s_wait_alu depctr_va_sdst(0)                               // 000000003af0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003af4: bf8700a1
	v_cndmask_b32_e64 v55, 0, v113, s16                        // 000000003af8: d5010037 0042e280
	v_cndmask_b32_e64 v54, 0, v112, s16                        // 000000003b00: d5010036 0042e080
	v_lshlrev_b64_e32 v[54:55], 2, v[54:55]                    // 000000003b08: 3e6c6c82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003b0c: bf870121
	v_add_co_u32 v54, s16, s28, v54                            // 000000003b10: d7001036 02026c1c
	s_wait_alu depctr_va_sdst(0)                               // 000000003b18: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s29, v55, s16               // 000000003b1c: d5207c37 00426e1d
	global_load_u8 v101, v[110:111], off                       // 000000003b24: ee04007c 00000065 0000006e
	global_load_b32 v100, v[54:55], off                        // 000000003b30: ee05007c 00000064 00000036
	s_wait_loadcnt 0x1                                         // 000000003b3c: bfc00001
	v_lshlrev_b32_e32 v123, 23, v101                           // 000000003b40: 30f6ca97
	s_wait_loadcnt 0x0                                         // 000000003b44: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003b48: bf870091
	v_mul_f32_e32 v101, v100, v123                             // 000000003b4c: 10caf764
	v_cmp_class_f32_e64 s16, v101, 0x198                       // 000000003b50: d47e0010 0201ff65 00000198
	v_mul_f32_e32 v120, v40, v101                              // 000000003b5c: 10f0cb28
	s_xor_b32 s16, s16, -1                                     // 000000003b60: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b64: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003b68: be912010
	s_cbranch_execnz 3040                                      // 000000003b6c: bfa60be0 <packed_folded_w4a8+0x4ff0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003b74: 8c7e117e
	v_or_b32_e32 v100, v116, v108                              // 000000003b78: 38c8d974
	v_mov_b32_e32 v101, v109                                   // 000000003b7c: 7eca036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003b80: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[100:101]                 // 000000003b84: d4540010 0202c824
	s_wait_alu depctr_va_sdst(0)                               // 000000003b8c: bf88f19f
	v_cndmask_b32_e64 v101, 0, v101, s16                       // 000000003b90: d5010065 0042ca80
	v_cndmask_b32_e64 v100, 0, v100, s16                       // 000000003b98: d5010064 0042c880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ba0: bf870091
	v_lshlrev_b64_e32 v[100:101], 2, v[100:101]                // 000000003ba4: 3ec8c882
	v_add_co_u32 v100, s16, s28, v100                          // 000000003ba8: d7001064 0202c81c
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003bb4: bf8700c2
	v_add_co_ci_u32_e64 v101, null, s29, v101, s16             // 000000003bb8: d5207c65 0042ca1d
	global_load_b32 v40, v[100:101], off                       // 000000003bc0: ee05007c 00000028 00000064
	s_wait_loadcnt 0x0                                         // 000000003bcc: bfc00000
	v_mul_f32_e32 v102, v40, v123                              // 000000003bd0: 10ccf728
	v_cmp_class_f32_e64 s16, v102, 0x198                       // 000000003bd4: d47e0010 0201ff66 00000198
	v_mul_f32_e32 v121, v41, v102                              // 000000003be0: 10f2cd29
	s_xor_b32 s16, s16, -1                                     // 000000003be4: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003be8: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003bec: be912010
	s_cbranch_execnz 3025                                      // 000000003bf0: bfa60bd1 <packed_folded_w4a8+0x5038>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bf4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003bf8: 8c7e117e
	v_or_b32_e32 v40, v117, v108                               // 000000003bfc: 3850d975
	v_mov_b32_e32 v41, v109                                    // 000000003c00: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003c04: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003c08: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003c10: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003c14: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003c1c: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c24: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003c28: 3e505082
	v_add_co_u32 v102, s16, s28, v40                           // 000000003c2c: d7001066 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c34: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003c38: bf8700c2
	v_add_co_ci_u32_e64 v103, null, s29, v41, s16              // 000000003c3c: d5207c67 0042521d
	global_load_b32 v40, v[102:103], off                       // 000000003c44: ee05007c 00000028 00000066
	s_wait_loadcnt 0x0                                         // 000000003c50: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003c54: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003c58: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v122, v42, v41                               // 000000003c64: 10f4532a
	s_xor_b32 s16, s16, -1                                     // 000000003c68: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c6c: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003c70: be912010
	s_cbranch_execnz 3010                                      // 000000003c74: bfa60bc2 <packed_folded_w4a8+0x5080>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003c7c: 8c7e117e
	v_or_b32_e32 v40, v128, v108                               // 000000003c80: 3850d980
	v_mov_b32_e32 v41, v109                                    // 000000003c84: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003c88: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003c8c: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003c94: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003c98: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003ca0: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ca8: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003cac: 3e505082
	v_add_co_u32 v104, s16, s28, v40                           // 000000003cb0: d7001068 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb8: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003cbc: bf8700c2
	v_add_co_ci_u32_e64 v105, null, s29, v41, s16              // 000000003cc0: d5207c69 0042521d
	global_load_b32 v40, v[104:105], off                       // 000000003cc8: ee05007c 00000028 00000068
	s_wait_loadcnt 0x0                                         // 000000003cd4: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003cd8: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003cdc: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v133, v43, v41                               // 000000003ce8: 110a532b
	s_xor_b32 s16, s16, -1                                     // 000000003cec: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cf0: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003cf4: be912010
	s_cbranch_execnz 2995                                      // 000000003cf8: bfa60bb3 <packed_folded_w4a8+0x50c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cfc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003d00: 8c7e117e
	v_or_b32_e32 v40, v129, v108                               // 000000003d04: 3850d981
	v_mov_b32_e32 v41, v109                                    // 000000003d08: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d0c: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003d10: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003d18: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003d1c: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003d24: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d2c: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003d30: 3e505082
	v_add_co_u32 v42, s16, s28, v40                            // 000000003d34: d700102a 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003d3c: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003d40: bf8700c2
	v_add_co_ci_u32_e64 v43, null, s29, v41, s16               // 000000003d44: d5207c2b 0042521d
	global_load_b32 v40, v[42:43], off                         // 000000003d4c: ee05007c 00000028 0000002a
	s_wait_loadcnt 0x0                                         // 000000003d58: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003d5c: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003d60: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v134, v44, v41                               // 000000003d6c: 110c532c
	s_xor_b32 s16, s16, -1                                     // 000000003d70: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d74: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003d78: be912010
	s_cbranch_execnz 2980                                      // 000000003d7c: bfa60ba4 <packed_folded_w4a8+0x5110>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003d84: 8c7e117e
	v_or_b32_e32 v40, v130, v108                               // 000000003d88: 3850d982
	v_mov_b32_e32 v41, v109                                    // 000000003d8c: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d90: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003d94: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003d9c: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003da0: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003da8: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003db0: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003db4: 3e505082
	v_add_co_u32 v106, s16, s28, v40                           // 000000003db8: d700106a 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc0: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003dc4: bf8700c2
	v_add_co_ci_u32_e64 v107, null, s29, v41, s16              // 000000003dc8: d5207c6b 0042521d
	global_load_b32 v40, v[106:107], off                       // 000000003dd0: ee05007c 00000028 0000006a
	s_wait_loadcnt 0x0                                         // 000000003ddc: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003de0: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003de4: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v135, v45, v41                               // 000000003df0: 110e532d
	s_xor_b32 s16, s16, -1                                     // 000000003df4: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003df8: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003dfc: be912010
	s_cbranch_execnz 2965                                      // 000000003e00: bfa60b95 <packed_folded_w4a8+0x5158>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003e08: 8c7e117e
	v_or_b32_e32 v40, v131, v108                               // 000000003e0c: 3850d983
	v_mov_b32_e32 v41, v109                                    // 000000003e10: 7e52036d
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003e14: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[40:41]                   // 000000003e18: d4540010 02025024
	s_wait_alu depctr_va_sdst(0)                               // 000000003e20: bf88f19f
	v_cndmask_b32_e64 v41, 0, v41, s16                         // 000000003e24: d5010029 00425280
	v_cndmask_b32_e64 v40, 0, v40, s16                         // 000000003e2c: d5010028 00425080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e34: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003e38: 3e505082
	v_add_co_u32 v44, s16, s28, v40                            // 000000003e3c: d700102c 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003e44: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003e48: bf8700c2
	v_add_co_ci_u32_e64 v45, null, s29, v41, s16               // 000000003e4c: d5207c2d 0042521d
	global_load_b32 v40, v[44:45], off                         // 000000003e54: ee05007c 00000028 0000002c
	s_wait_loadcnt 0x0                                         // 000000003e60: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003e64: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003e68: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v136, v46, v41                               // 000000003e74: 1110532e
	s_xor_b32 s16, s16, -1                                     // 000000003e78: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e7c: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003e80: be912010
	s_cbranch_execnz 2950                                      // 000000003e84: bfa60b86 <packed_folded_w4a8+0x51a0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003e8c: 8c7e117e
	v_or_b32_e32 v108, v132, v108                              // 000000003e90: 38d8d984
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003e94: bf8700a1
	v_cmp_gt_i64_e64 s16, s[36:37], v[108:109]                 // 000000003e98: d4540010 0202d824
	s_wait_alu depctr_va_sdst(0)                               // 000000003ea0: bf88f19f
	v_cndmask_b32_e64 v41, 0, v109, s16                        // 000000003ea4: d5010029 0042da80
	v_cndmask_b32_e64 v40, 0, v108, s16                        // 000000003eac: d5010028 0042d880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003eb4: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003eb8: 3e505082
	v_add_co_u32 v108, s16, s28, v40                           // 000000003ebc: d700106c 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003ec4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000003ec8: bf8700c2
	v_add_co_ci_u32_e64 v109, null, s29, v41, s16              // 000000003ecc: d5207c6d 0042521d
	global_load_b32 v40, v[108:109], off                       // 000000003ed4: ee05007c 00000028 0000006c
	s_wait_loadcnt 0x0                                         // 000000003ee0: bfc00000
	v_mul_f32_e32 v41, v40, v123                               // 000000003ee4: 1052f728
	v_cmp_class_f32_e64 s16, v41, 0x198                        // 000000003ee8: d47e0010 0201ff29 00000198
	v_mul_f32_e32 v137, v47, v41                               // 000000003ef4: 1112532f
	s_xor_b32 s16, s16, -1                                     // 000000003ef8: 8d10c110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003efc: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003f00: be912010
	s_cbranch_execnz 2936                                      // 000000003f04: bfa60b78 <packed_folded_w4a8+0x51e8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003f0c: 8c7e117e
	v_mul_lo_u32 v123, s39, v112                               // 000000003f10: d72c007b 0202e027
	v_mul_lo_u32 v138, s38, v113                               // 000000003f18: d72c008a 0202e226
	v_mad_co_u64_u32 v[40:41], null, s38, v112, 0              // 000000003f20: d6fe7c28 0202e026
	v_sub_co_u32 v46, s16, s36, v112                           // 000000003f28: d701102e 0202e024
	s_wait_alu depctr_va_sdst(0)                               // 000000003f30: bf88f19f
	v_sub_co_ci_u32_e64 v47, null, s37, v113, s16              // 000000003f34: d5217c2f 0042e225
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003f3c: bf870211
	v_cmp_lt_i64_e64 s19, 0, v[46:47]                          // 000000003f40: d4510013 02025c80
	v_add3_u32 v41, v41, v138, v123                            // 000000003f48: d6550029 05ef1529
	s_delay_alu instid0(valu_dep_1)                            // 000000003f50: bf870001
	v_lshlrev_b64_e32 v[40:41], 1, v[40:41]                    // 000000003f54: 3e505081
	s_and_b32 s16, s19, vcc_lo                                 // 000000003f58: 8b106a13
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f5c: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000003f60: be912010
	s_cbranch_execz 28                                         // 000000003f64: bfa5001c <packed_folded_w4a8+0x24d8>
	v_bfe_u32 v112, v120, 16, 1                                // 000000003f68: d6100070 02052178
	s_wait_kmcnt 0x0                                           // 000000003f70: bfc70000
	v_add_co_u32 v113, s16, s34, v40                           // 000000003f74: d7001071 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000003f7c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s35, v41, s16              // 000000003f80: d5207c7b 00425223
	v_add3_u32 v138, v112, v120, 0x7fff                        // 000000003f88: d655008a 03fef170 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003f94: bf870003
	v_add_co_u32 v112, s16, v113, v118                         // 000000003f98: d7001070 0202ed71
	v_or_b32_e32 v139, 0x400000, v120                          // 000000003fa0: 3916f0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v123, v119, s16            // 000000003fac: d5207c71 0042ef7b
	v_cmp_u_f32_e64 s16, v120, v120                            // 000000003fb4: d4180010 0202f178
	s_wait_alu depctr_va_sdst(0)                               // 000000003fbc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003fc0: bf870001
	v_cndmask_b32_e64 v120, v138, v139, s16                    // 000000003fc4: d5010078 0043178a
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000003fcc: ee09407c 3c000000 00000070
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fd8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003fdc: 8c7e117e
	v_cmp_lt_i64_e64 s16, 1, v[46:47]                          // 000000003fe0: d4510010 02025c81
	s_and_b32 s17, s16, vcc_lo                                 // 000000003fe8: 8b116a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fec: bf88ff9e
	s_and_saveexec_b32 s18, s17                                // 000000003ff0: be922011
	s_cbranch_execz 28                                         // 000000003ff4: bfa5001c <packed_folded_w4a8+0x2568>
	v_bfe_u32 v112, v121, 16, 1                                // 000000003ff8: d6100070 02052179
	s_wait_kmcnt 0x0                                           // 000000004000: bfc70000
	v_add_co_u32 v113, s17, s34, v40                           // 000000004004: d7001171 02025022
	s_wait_alu depctr_va_sdst(0)                               // 00000000400c: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s17              // 000000004010: d5207c78 00465223
	v_add3_u32 v123, v112, v121, 0x7fff                        // 000000004018: d655007b 03fef370 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004024: bf870003
	v_add_co_u32 v112, s17, v113, v56                          // 000000004028: d7001170 02027171
	v_or_b32_e32 v138, 0x400000, v121                          // 000000004030: 3914f2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004038: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v57, s17             // 00000000403c: d5207c71 00467378
	v_cmp_u_f32_e64 s17, v121, v121                            // 000000004044: d4180011 0202f379
	s_wait_alu depctr_va_sdst(0)                               // 00000000404c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004050: bf870001
	v_cndmask_b32_e64 v120, v123, v138, s17                    // 000000004054: d5010078 0047157b
	global_store_d16_hi_b16 v[112:113], v120, off              // 00000000405c: ee09407c 3c000000 00000070
	s_wait_alu depctr_sa_sdst(0)                               // 000000004068: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s18                             // 00000000406c: 8c7e127e
	v_cmp_lt_i64_e64 s17, 2, v[46:47]                          // 000000004070: d4510011 02025c82
	s_and_b32 s18, s17, vcc_lo                                 // 000000004078: 8b126a11
	s_wait_alu depctr_sa_sdst(0)                               // 00000000407c: bf88ff9e
	s_and_saveexec_b32 s20, s18                                // 000000004080: be942012
	s_cbranch_execz 28                                         // 000000004084: bfa5001c <packed_folded_w4a8+0x25f8>
	v_bfe_u32 v112, v122, 16, 1                                // 000000004088: d6100070 0205217a
	s_wait_kmcnt 0x0                                           // 000000004090: bfc70000
	v_add_co_u32 v113, s18, s34, v40                           // 000000004094: d7001271 02025022
	s_wait_alu depctr_va_sdst(0)                               // 00000000409c: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s18              // 0000000040a0: d5207c78 004a5223
	v_add3_u32 v121, v112, v122, 0x7fff                        // 0000000040a8: d6550079 03fef570 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000040b4: bf870003
	v_add_co_u32 v112, s18, v113, v58                          // 0000000040b8: d7001270 02027571
	v_or_b32_e32 v123, 0x400000, v122                          // 0000000040c0: 38f6f4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000040c8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v59, s18             // 0000000040cc: d5207c71 004a7778
	v_cmp_u_f32_e64 s18, v122, v122                            // 0000000040d4: d4180012 0202f57a
	s_wait_alu depctr_va_sdst(0)                               // 0000000040dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040e0: bf870001
	v_cndmask_b32_e64 v120, v121, v123, s18                    // 0000000040e4: d5010078 004af779
	global_store_d16_hi_b16 v[112:113], v120, off              // 0000000040ec: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s20                             // 0000000040f8: 8c7e147e
	v_cmp_lt_i64_e64 s18, 3, v[46:47]                          // 0000000040fc: d4510012 02025c83
	s_and_b32 s20, s18, vcc_lo                                 // 000000004104: 8b146a12
	s_delay_alu instid0(salu_cycle_1)                          // 000000004108: bf870009
	s_and_saveexec_b32 s21, s20                                // 00000000410c: be952014
	s_cbranch_execz 28                                         // 000000004110: bfa5001c <packed_folded_w4a8+0x2684>
	v_bfe_u32 v112, v133, 16, 1                                // 000000004114: d6100070 02052185
	s_wait_kmcnt 0x0                                           // 00000000411c: bfc70000
	v_add_co_u32 v113, s20, s34, v40                           // 000000004120: d7001471 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004128: bf870191
	v_add_co_ci_u32_e64 v120, null, s35, v41, s20              // 00000000412c: d5207c78 00525223
	v_add3_u32 v121, v112, v133, 0x7fff                        // 000000004134: d6550079 03ff0b70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004140: bf870003
	v_add_co_u32 v112, s20, v113, v60                          // 000000004144: d7001470 02027971
	v_or_b32_e32 v122, 0x400000, v133                          // 00000000414c: 38f50aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004154: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v61, s20             // 000000004158: d5207c71 00527b78
	v_cmp_u_f32_e64 s20, v133, v133                            // 000000004160: d4180014 02030b85
	s_wait_alu depctr_va_sdst(0)                               // 000000004168: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000416c: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s20                    // 000000004170: d5010078 0052f579
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004178: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s21                             // 000000004184: 8c7e157e
	v_cmp_lt_i64_e64 s20, 4, v[46:47]                          // 000000004188: d4510014 02025c84
	s_and_b32 s21, s20, vcc_lo                                 // 000000004190: 8b156a14
	s_wait_alu depctr_sa_sdst(0)                               // 000000004194: bf88ff9e
	s_and_saveexec_b32 s22, s21                                // 000000004198: be962015
	s_cbranch_execz 28                                         // 00000000419c: bfa5001c <packed_folded_w4a8+0x2710>
	v_bfe_u32 v112, v134, 16, 1                                // 0000000041a0: d6100070 02052186
	s_wait_kmcnt 0x0                                           // 0000000041a8: bfc70000
	v_add_co_u32 v113, s21, s34, v40                           // 0000000041ac: d7001571 02025022
	s_wait_alu depctr_va_sdst(0)                               // 0000000041b4: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s21              // 0000000041b8: d5207c78 00565223
	v_add3_u32 v121, v112, v134, 0x7fff                        // 0000000041c0: d6550079 03ff0d70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000041cc: bf870003
	v_add_co_u32 v112, s21, v113, v62                          // 0000000041d0: d7001570 02027d71
	v_or_b32_e32 v122, 0x400000, v134                          // 0000000041d8: 38f50cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000041e0: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v63, s21             // 0000000041e4: d5207c71 00567f78
	v_cmp_u_f32_e64 s21, v134, v134                            // 0000000041ec: d4180015 02030d86
	s_wait_alu depctr_va_sdst(0)                               // 0000000041f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000041f8: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s21                    // 0000000041fc: d5010078 0056f579
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004204: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s22                             // 000000004210: 8c7e167e
	v_cmp_lt_i64_e64 s21, 5, v[46:47]                          // 000000004214: d4510015 02025c85
	s_and_b32 s22, s21, vcc_lo                                 // 00000000421c: 8b166a15
	s_delay_alu instid0(salu_cycle_1)                          // 000000004220: bf870009
	s_and_saveexec_b32 s23, s22                                // 000000004224: be972016
	s_cbranch_execz 28                                         // 000000004228: bfa5001c <packed_folded_w4a8+0x279c>
	v_bfe_u32 v112, v135, 16, 1                                // 00000000422c: d6100070 02052187
	s_wait_kmcnt 0x0                                           // 000000004234: bfc70000
	v_add_co_u32 v113, s22, s34, v40                           // 000000004238: d7001671 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004240: bf870191
	v_add_co_ci_u32_e64 v120, null, s35, v41, s22              // 000000004244: d5207c78 005a5223
	v_add3_u32 v121, v112, v135, 0x7fff                        // 00000000424c: d6550079 03ff0f70 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004258: bf870003
	v_add_co_u32 v112, s22, v113, v64                          // 00000000425c: d7001670 02028171
	v_or_b32_e32 v122, 0x400000, v135                          // 000000004264: 38f50eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000426c: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v65, s22             // 000000004270: d5207c71 005a8378
	v_cmp_u_f32_e64 s22, v135, v135                            // 000000004278: d4180016 02030f87
	s_wait_alu depctr_va_sdst(0)                               // 000000004280: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004284: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s22                    // 000000004288: d5010078 005af579
	global_store_d16_hi_b16 v[112:113], v120, off              // 000000004290: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s23                             // 00000000429c: 8c7e177e
	v_cmp_lt_i64_e64 s22, 6, v[46:47]                          // 0000000042a0: d4510016 02025c86
	s_and_b32 s23, s22, vcc_lo                                 // 0000000042a8: 8b176a16
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042ac: bf88ff9e
	s_and_saveexec_b32 s24, s23                                // 0000000042b0: be982017
	s_cbranch_execz 28                                         // 0000000042b4: bfa5001c <packed_folded_w4a8+0x2828>
	v_bfe_u32 v112, v136, 16, 1                                // 0000000042b8: d6100070 02052188
	s_wait_kmcnt 0x0                                           // 0000000042c0: bfc70000
	v_add_co_u32 v113, s23, s34, v40                           // 0000000042c4: d7001771 02025022
	s_wait_alu depctr_va_sdst(0)                               // 0000000042cc: bf88f19f
	v_add_co_ci_u32_e64 v120, null, s35, v41, s23              // 0000000042d0: d5207c78 005e5223
	v_add3_u32 v121, v112, v136, 0x7fff                        // 0000000042d8: d6550079 03ff1170 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000042e4: bf870003
	v_add_co_u32 v112, s23, v113, v66                          // 0000000042e8: d7001770 02028571
	v_or_b32_e32 v122, 0x400000, v136                          // 0000000042f0: 38f510ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000042f8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v120, v67, s23             // 0000000042fc: d5207c71 005e8778
	v_cmp_u_f32_e64 s23, v136, v136                            // 000000004304: d4180017 02031188
	s_wait_alu depctr_va_sdst(0)                               // 00000000430c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004310: bf870001
	v_cndmask_b32_e64 v120, v121, v122, s23                    // 000000004314: d5010078 005ef579
	global_store_d16_hi_b16 v[112:113], v120, off              // 00000000431c: ee09407c 3c000000 00000070
	s_or_b32 exec_lo, exec_lo, s24                             // 000000004328: 8c7e187e
	v_cmp_lt_i64_e64 s23, 7, v[46:47]                          // 00000000432c: d4510017 02025c87
	s_and_b32 s24, s23, vcc_lo                                 // 000000004334: 8b186a17
	s_delay_alu instid0(salu_cycle_1)                          // 000000004338: bf870009
	s_and_saveexec_b32 s25, s24                                // 00000000433c: be992018
	s_cbranch_execz 28                                         // 000000004340: bfa5001c <packed_folded_w4a8+0x28b4>
	v_bfe_u32 v46, v137, 16, 1                                 // 000000004344: d610002e 02052189
	s_wait_kmcnt 0x0                                           // 00000000434c: bfc70000
	v_add_co_u32 v47, s24, s34, v40                            // 000000004350: d700182f 02025022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004358: bf870191
	v_add_co_ci_u32_e64 v112, null, s35, v41, s24              // 00000000435c: d5207c70 00625223
	v_add3_u32 v113, v46, v137, 0x7fff                         // 000000004364: d6550071 03ff132e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004370: bf870003
	v_add_co_u32 v46, s24, v47, v68                            // 000000004374: d700182e 0202892f
	v_or_b32_e32 v120, 0x400000, v137                          // 00000000437c: 38f112ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004384: bf88f19f
	v_add_co_ci_u32_e64 v47, null, v112, v69, s24              // 000000004388: d5207c2f 00628b70
	v_cmp_u_f32_e64 s24, v137, v137                            // 000000004390: d4180018 02031389
	s_wait_alu depctr_va_sdst(0)                               // 000000004398: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000439c: bf870001
	v_cndmask_b32_e64 v112, v113, v120, s24                    // 0000000043a0: d5010070 0062f171
	global_store_d16_hi_b16 v[46:47], v112, off                // 0000000043a8: ee09407c 38000000 0000002e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000043b4: 8c7e197e
	v_or_b32_e32 v120, s26, v114                               // 0000000043b8: 38f0e41a
	v_mov_b32_e32 v123, s27                                    // 0000000043bc: 7ef6021b
	v_mov_b32_e32 v121, s27                                    // 0000000043c0: 7ef2021b
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000043c4: bf870093
	v_or_b32_e32 v122, v120, v115                              // 0000000043c8: 38f4e778
	v_cmp_gt_i64_e64 s24, s[36:37], v[122:123]                 // 0000000043cc: d4540018 0202f424
	s_wait_alu depctr_va_sdst(0)                               // 0000000043d4: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000043d8: bf8700a1
	v_cndmask_b32_e64 v47, 0, v123, s24                        // 0000000043dc: d501002f 0062f680
	v_cndmask_b32_e64 v46, 0, v122, s24                        // 0000000043e4: d501002e 0062f480
	v_lshlrev_b64_e32 v[46:47], 2, v[46:47]                    // 0000000043ec: 3e5c5c82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000043f0: bf870121
	v_add_co_u32 v46, s24, s28, v46                            // 0000000043f4: d700182e 02025c1c
	s_wait_alu depctr_va_sdst(0)                               // 0000000043fc: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s29, v47, s24               // 000000004400: d5207c2f 00625e1d
	global_load_u8 v111, v[110:111], off                       // 000000004408: ee04007c 0000006f 0000006e
	global_load_b32 v110, v[46:47], off                        // 000000004414: ee05007c 0000006e 0000002e
	s_wait_loadcnt 0x1                                         // 000000004420: bfc00001
	v_lshlrev_b32_e32 v136, 23, v111                           // 000000004424: 3110de97
	s_wait_loadcnt 0x0                                         // 000000004428: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000442c: bf870091
	v_mul_f32_e32 v111, v110, v136                             // 000000004430: 10df116e
	v_cmp_class_f32_e64 s24, v111, 0x198                       // 000000004434: d47e0018 0201ff6f 00000198
	v_mul_f32_e32 v133, v32, v111                              // 000000004440: 110adf20
	s_xor_b32 s24, s24, -1                                     // 000000004444: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004448: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000444c: be992018
	s_cbranch_execnz 2615                                      // 000000004450: bfa60a37 <packed_folded_w4a8+0x5230>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004454: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004458: 8c7e197e
	v_or_b32_e32 v110, v116, v120                              // 00000000445c: 38dcf174
	v_mov_b32_e32 v111, v121                                   // 000000004460: 7ede0379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004464: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[110:111]                 // 000000004468: d4540018 0202dc24
	s_wait_alu depctr_va_sdst(0)                               // 000000004470: bf88f19f
	v_cndmask_b32_e64 v111, 0, v111, s24                       // 000000004474: d501006f 0062de80
	v_cndmask_b32_e64 v110, 0, v110, s24                       // 00000000447c: d501006e 0062dc80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004484: bf870091
	v_lshlrev_b64_e32 v[110:111], 2, v[110:111]                // 000000004488: 3edcdc82
	v_add_co_u32 v110, s24, s28, v110                          // 00000000448c: d700186e 0202dc1c
	s_wait_alu depctr_va_sdst(0)                               // 000000004494: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000004498: bf8700c2
	v_add_co_ci_u32_e64 v111, null, s29, v111, s24             // 00000000449c: d5207c6f 0062de1d
	global_load_b32 v32, v[110:111], off                       // 0000000044a4: ee05007c 00000020 0000006e
	s_wait_loadcnt 0x0                                         // 0000000044b0: bfc00000
	v_mul_f32_e32 v112, v32, v136                              // 0000000044b4: 10e11120
	v_cmp_class_f32_e64 s24, v112, 0x198                       // 0000000044b8: d47e0018 0201ff70 00000198
	v_mul_f32_e32 v134, v33, v112                              // 0000000044c4: 110ce121
	s_xor_b32 s24, s24, -1                                     // 0000000044c8: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044cc: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 0000000044d0: be992018
	s_cbranch_execnz 2600                                      // 0000000044d4: bfa60a28 <packed_folded_w4a8+0x5278>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000044dc: 8c7e197e
	v_or_b32_e32 v32, v117, v120                               // 0000000044e0: 3840f175
	v_mov_b32_e32 v33, v121                                    // 0000000044e4: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000044e8: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 0000000044ec: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 0000000044f4: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 0000000044f8: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004500: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004508: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 00000000450c: 3e404082
	v_add_co_u32 v112, s24, s28, v32                           // 000000004510: d7001870 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004518: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000451c: bf8700c2
	v_add_co_ci_u32_e64 v113, null, s29, v33, s24              // 000000004520: d5207c71 0062421d
	global_load_b32 v32, v[112:113], off                       // 000000004528: ee05007c 00000020 00000070
	s_wait_loadcnt 0x0                                         // 000000004534: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004538: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 00000000453c: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v135, v34, v33                               // 000000004548: 110e4322
	s_xor_b32 s24, s24, -1                                     // 00000000454c: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004550: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004554: be992018
	s_cbranch_execnz 2585                                      // 000000004558: bfa60a19 <packed_folded_w4a8+0x52c0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000455c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004560: 8c7e197e
	v_or_b32_e32 v32, v128, v120                               // 000000004564: 3840f180
	v_mov_b32_e32 v33, v121                                    // 000000004568: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 00000000456c: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 000000004570: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 000000004578: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 00000000457c: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004584: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000458c: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004590: 3e404082
	v_add_co_u32 v114, s24, s28, v32                           // 000000004594: d7001872 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 00000000459c: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000045a0: bf8700c2
	v_add_co_ci_u32_e64 v115, null, s29, v33, s24              // 0000000045a4: d5207c73 0062421d
	global_load_b32 v32, v[114:115], off                       // 0000000045ac: ee05007c 00000020 00000072
	s_wait_loadcnt 0x0                                         // 0000000045b8: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000045bc: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000045c0: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v128, v35, v33                               // 0000000045cc: 11004323
	s_xor_b32 s24, s24, -1                                     // 0000000045d0: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045d4: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 0000000045d8: be992018
	s_cbranch_execnz 2570                                      // 0000000045dc: bfa60a0a <packed_folded_w4a8+0x5308>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000045e4: 8c7e197e
	v_or_b32_e32 v32, v129, v120                               // 0000000045e8: 3840f181
	v_mov_b32_e32 v33, v121                                    // 0000000045ec: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000045f0: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 0000000045f4: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 0000000045fc: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004600: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004608: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004610: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004614: 3e404082
	v_add_co_u32 v34, s24, s28, v32                            // 000000004618: d7001822 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004620: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000004624: bf8700c2
	v_add_co_ci_u32_e64 v35, null, s29, v33, s24               // 000000004628: d5207c23 0062421d
	global_load_b32 v32, v[34:35], off                         // 000000004630: ee05007c 00000020 00000022
	s_wait_loadcnt 0x0                                         // 00000000463c: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004640: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 000000004644: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v129, v36, v33                               // 000000004650: 11024324
	s_xor_b32 s24, s24, -1                                     // 000000004654: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004658: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 00000000465c: be992018
	s_cbranch_execnz 2555                                      // 000000004660: bfa609fb <packed_folded_w4a8+0x5350>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004664: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004668: 8c7e197e
	v_or_b32_e32 v32, v130, v120                               // 00000000466c: 3840f182
	v_mov_b32_e32 v33, v121                                    // 000000004670: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004674: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 000000004678: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 000000004680: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004684: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 00000000468c: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004694: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004698: 3e404082
	v_add_co_u32 v116, s24, s28, v32                           // 00000000469c: d7001874 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000046a8: bf8700c2
	v_add_co_ci_u32_e64 v117, null, s29, v33, s24              // 0000000046ac: d5207c75 0062421d
	global_load_b32 v32, v[116:117], off                       // 0000000046b4: ee05007c 00000020 00000074
	s_wait_loadcnt 0x0                                         // 0000000046c0: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000046c4: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000046c8: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v130, v37, v33                               // 0000000046d4: 11044325
	s_xor_b32 s24, s24, -1                                     // 0000000046d8: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046dc: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 0000000046e0: be992018
	s_cbranch_execnz 2540                                      // 0000000046e4: bfa609ec <packed_folded_w4a8+0x5398>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000046ec: 8c7e197e
	v_or_b32_e32 v32, v131, v120                               // 0000000046f0: 3840f183
	v_mov_b32_e32 v33, v121                                    // 0000000046f4: 7e420379
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 0000000046f8: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[32:33]                   // 0000000046fc: d4540018 02024024
	s_wait_alu depctr_va_sdst(0)                               // 000000004704: bf88f19f
	v_cndmask_b32_e64 v33, 0, v33, s24                         // 000000004708: d5010021 00624280
	v_cndmask_b32_e64 v32, 0, v32, s24                         // 000000004710: d5010020 00624080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004718: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 00000000471c: 3e404082
	v_add_co_u32 v36, s24, s28, v32                            // 000000004720: d7001824 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000004728: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000472c: bf8700c2
	v_add_co_ci_u32_e64 v37, null, s29, v33, s24               // 000000004730: d5207c25 0062421d
	global_load_b32 v32, v[36:37], off                         // 000000004738: ee05007c 00000020 00000024
	s_wait_loadcnt 0x0                                         // 000000004744: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 000000004748: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 00000000474c: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v131, v38, v33                               // 000000004758: 11064326
	s_xor_b32 s24, s24, -1                                     // 00000000475c: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 000000004760: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004764: be992018
	s_cbranch_execnz 2525                                      // 000000004768: bfa609dd <packed_folded_w4a8+0x53e0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000476c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 000000004770: 8c7e197e
	v_or_b32_e32 v120, v132, v120                              // 000000004774: 38f0f184
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004778: bf8700a1
	v_cmp_gt_i64_e64 s24, s[36:37], v[120:121]                 // 00000000477c: d4540018 0202f024
	s_wait_alu depctr_va_sdst(0)                               // 000000004784: bf88f19f
	v_cndmask_b32_e64 v33, 0, v121, s24                        // 000000004788: d5010021 0062f280
	v_cndmask_b32_e64 v32, 0, v120, s24                        // 000000004790: d5010020 0062f080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004798: bf870091
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 00000000479c: 3e404082
	v_add_co_u32 v120, s24, s28, v32                           // 0000000047a0: d7001878 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 0000000047a8: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000047ac: bf8700c2
	v_add_co_ci_u32_e64 v121, null, s29, v33, s24              // 0000000047b0: d5207c79 0062421d
	global_load_b32 v32, v[120:121], off                       // 0000000047b8: ee05007c 00000020 00000078
	s_wait_loadcnt 0x0                                         // 0000000047c4: bfc00000
	v_mul_f32_e32 v33, v32, v136                               // 0000000047c8: 10431120
	v_cmp_class_f32_e64 s24, v33, 0x198                        // 0000000047cc: d47e0018 0201ff21 00000198
	v_mul_f32_e32 v132, v39, v33                               // 0000000047d8: 11084327
	s_xor_b32 s24, s24, -1                                     // 0000000047dc: 8d18c118
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047e0: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 0000000047e4: be992018
	s_cbranch_execnz 2511                                      // 0000000047e8: bfa609cf <packed_folded_w4a8+0x5428>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000047f0: 8c7e197e
	v_mul_lo_u32 v136, s39, v122                               // 0000000047f4: d72c0088 0202f427
	v_mul_lo_u32 v137, s38, v123                               // 0000000047fc: d72c0089 0202f626
	v_mad_co_u64_u32 v[32:33], null, s38, v122, 0              // 000000004804: d6fe7c20 0202f426
	v_sub_co_u32 v38, s24, s36, v122                           // 00000000480c: d7011826 0202f424
	s_wait_alu depctr_va_sdst(0)                               // 000000004814: bf88f19f
	v_sub_co_ci_u32_e64 v39, null, s37, v123, s24              // 000000004818: d5217c27 0062f625
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000004820: bf870211
	v_cmp_lt_i64_e64 s27, 0, v[38:39]                          // 000000004824: d451001b 02024c80
	v_add3_u32 v33, v33, v137, v136                            // 00000000482c: d6550021 06231321
	s_delay_alu instid0(valu_dep_1)                            // 000000004834: bf870001
	v_lshlrev_b64_e32 v[32:33], 1, v[32:33]                    // 000000004838: 3e404081
	s_and_b32 s24, s27, vcc_lo                                 // 00000000483c: 8b186a1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004840: bf88ff9e
	s_and_saveexec_b32 s25, s24                                // 000000004844: be992018
	s_cbranch_execz 28                                         // 000000004848: bfa5001c <packed_folded_w4a8+0x2dbc>
	s_wait_kmcnt 0x0                                           // 00000000484c: bfc70000
	v_add_co_u32 v123, s24, s34, v32                           // 000000004850: d700187b 02024022
	v_bfe_u32 v122, v133, 16, 1                                // 000000004858: d610007a 02052185
	s_wait_alu depctr_va_sdst(0)                               // 000000004860: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v33, s24              // 000000004864: d5207c88 00624223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000486c: bf870193
	v_add_co_u32 v118, s24, v123, v118                         // 000000004870: d7001876 0202ed7b
	v_add3_u32 v122, v122, v133, 0x7fff                        // 000000004878: d655007a 03ff0b7a 00007fff
	v_or_b32_e32 v137, 0x400000, v133                          // 000000004884: 39130aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000488c: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v136, v119, s24            // 000000004890: d5207c77 0062ef88
	v_cmp_u_f32_e64 s24, v133, v133                            // 000000004898: d4180018 02030b85
	s_wait_alu depctr_va_sdst(0)                               // 0000000048a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048a4: bf870001
	v_cndmask_b32_e64 v122, v122, v137, s24                    // 0000000048a8: d501007a 0063137a
	global_store_d16_hi_b16 v[118:119], v122, off              // 0000000048b0: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000048c0: 8c7e197e
	v_cmp_lt_i64_e64 s24, 1, v[38:39]                          // 0000000048c4: d4510018 02024c81
	s_and_b32 s25, s24, vcc_lo                                 // 0000000048cc: 8b196a18
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048d0: bf88ff9e
	s_and_saveexec_b32 s26, s25                                // 0000000048d4: be9a2019
	s_cbranch_execz 28                                         // 0000000048d8: bfa5001c <packed_folded_w4a8+0x2e4c>
	v_bfe_u32 v118, v134, 16, 1                                // 0000000048dc: d6100076 02052186
	s_wait_kmcnt 0x0                                           // 0000000048e4: bfc70000
	v_add_co_u32 v119, s25, s34, v32                           // 0000000048e8: d7001977 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000048f0: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s25              // 0000000048f4: d5207c7a 00664223
	v_add3_u32 v123, v118, v134, 0x7fff                        // 0000000048fc: d655007b 03ff0d76 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004908: bf870003
	v_add_co_u32 v118, s25, v119, v56                          // 00000000490c: d7001976 02027177
	v_or_b32_e32 v133, 0x400000, v134                          // 000000004914: 390b0cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000491c: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v57, s25             // 000000004920: d5207c77 0066737a
	v_cmp_u_f32_e64 s25, v134, v134                            // 000000004928: d4180019 02030d86
	s_wait_alu depctr_va_sdst(0)                               // 000000004930: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004934: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s25                    // 000000004938: d501007a 00670b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004940: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 00000000494c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s26                             // 000000004950: 8c7e1a7e
	v_cmp_lt_i64_e64 s25, 2, v[38:39]                          // 000000004954: d4510019 02024c82
	s_and_b32 s26, s25, vcc_lo                                 // 00000000495c: 8b1a6a19
	s_wait_alu depctr_sa_sdst(0)                               // 000000004960: bf88ff9e
	s_and_saveexec_b32 s28, s26                                // 000000004964: be9c201a
	s_cbranch_execz 28                                         // 000000004968: bfa5001c <packed_folded_w4a8+0x2edc>
	v_bfe_u32 v118, v135, 16, 1                                // 00000000496c: d6100076 02052187
	s_wait_kmcnt 0x0                                           // 000000004974: bfc70000
	v_add_co_u32 v119, s26, s34, v32                           // 000000004978: d7001a77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004980: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s26              // 000000004984: d5207c7a 006a4223
	v_add3_u32 v123, v118, v135, 0x7fff                        // 00000000498c: d655007b 03ff0f76 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004998: bf870003
	v_add_co_u32 v118, s26, v119, v58                          // 00000000499c: d7001a76 02027577
	v_or_b32_e32 v133, 0x400000, v135                          // 0000000049a4: 390b0eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000049ac: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v59, s26             // 0000000049b0: d5207c77 006a777a
	v_cmp_u_f32_e64 s26, v135, v135                            // 0000000049b8: d418001a 02030f87
	s_wait_alu depctr_va_sdst(0)                               // 0000000049c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000049c4: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s26                    // 0000000049c8: d501007a 006b0b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 0000000049d0: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 0000000049e0: 8c7e1c7e
	v_cmp_lt_i64_e64 s26, 3, v[38:39]                          // 0000000049e4: d451001a 02024c83
	s_and_b32 s28, s26, vcc_lo                                 // 0000000049ec: 8b1c6a1a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049f0: bf88ff9e
	s_and_saveexec_b32 s29, s28                                // 0000000049f4: be9d201c
	s_cbranch_execz 28                                         // 0000000049f8: bfa5001c <packed_folded_w4a8+0x2f6c>
	v_bfe_u32 v118, v128, 16, 1                                // 0000000049fc: d6100076 02052180
	s_wait_kmcnt 0x0                                           // 000000004a04: bfc70000
	v_add_co_u32 v119, s28, s34, v32                           // 000000004a08: d7001c77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004a10: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s28              // 000000004a14: d5207c7a 00724223
	v_add3_u32 v123, v118, v128, 0x7fff                        // 000000004a1c: d655007b 03ff0176 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004a28: bf870003
	v_add_co_u32 v118, s28, v119, v60                          // 000000004a2c: d7001c76 02027977
	v_or_b32_e32 v133, 0x400000, v128                          // 000000004a34: 390b00ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004a3c: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v61, s28             // 000000004a40: d5207c77 00727b7a
	v_cmp_u_f32_e64 s28, v128, v128                            // 000000004a48: d418001c 02030180
	s_wait_alu depctr_va_sdst(0)                               // 000000004a50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a54: bf870001
	v_cndmask_b32_e64 v122, v123, v133, s28                    // 000000004a58: d501007a 00730b7b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004a60: ee09407c 3d000000 00000076
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s29                             // 000000004a70: 8c7e1d7e
	v_cmp_lt_i64_e64 s28, 4, v[38:39]                          // 000000004a74: d451001c 02024c84
	s_and_b32 s29, s28, vcc_lo                                 // 000000004a7c: 8b1d6a1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a80: bf88ff9e
	s_and_saveexec_b32 s30, s29                                // 000000004a84: be9e201d
	s_cbranch_execz 28                                         // 000000004a88: bfa5001c <packed_folded_w4a8+0x2ffc>
	v_bfe_u32 v118, v129, 16, 1                                // 000000004a8c: d6100076 02052181
	s_wait_kmcnt 0x0                                           // 000000004a94: bfc70000
	v_add_co_u32 v119, s29, s34, v32                           // 000000004a98: d7001d77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004aa0: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s29              // 000000004aa4: d5207c7a 00764223
	v_add3_u32 v123, v118, v129, 0x7fff                        // 000000004aac: d655007b 03ff0376 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004ab8: bf870003
	v_add_co_u32 v118, s29, v119, v62                          // 000000004abc: d7001d76 02027d77
	v_or_b32_e32 v128, 0x400000, v129                          // 000000004ac4: 390102ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004acc: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v63, s29             // 000000004ad0: d5207c77 00767f7a
	v_cmp_u_f32_e64 s29, v129, v129                            // 000000004ad8: d418001d 02030381
	s_wait_alu depctr_va_sdst(0)                               // 000000004ae0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ae4: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s29                    // 000000004ae8: d501007a 0077017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004af0: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s30                             // 000000004afc: 8c7e1e7e
	v_cmp_lt_i64_e64 s29, 5, v[38:39]                          // 000000004b00: d451001d 02024c85
	s_and_b32 s30, s29, vcc_lo                                 // 000000004b08: 8b1e6a1d
	s_delay_alu instid0(salu_cycle_1)                          // 000000004b0c: bf870009
	s_and_saveexec_b32 s31, s30                                // 000000004b10: be9f201e
	s_cbranch_execz 28                                         // 000000004b14: bfa5001c <packed_folded_w4a8+0x3088>
	v_bfe_u32 v118, v130, 16, 1                                // 000000004b18: d6100076 02052182
	s_wait_kmcnt 0x0                                           // 000000004b20: bfc70000
	v_add_co_u32 v119, s30, s34, v32                           // 000000004b24: d7001e77 02024022
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004b2c: bf870191
	v_add_co_ci_u32_e64 v122, null, s35, v33, s30              // 000000004b30: d5207c7a 007a4223
	v_add3_u32 v123, v118, v130, 0x7fff                        // 000000004b38: d655007b 03ff0576 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004b44: bf870003
	v_add_co_u32 v118, s30, v119, v64                          // 000000004b48: d7001e76 02028177
	v_or_b32_e32 v128, 0x400000, v130                          // 000000004b50: 390104ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b58: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v65, s30             // 000000004b5c: d5207c77 007a837a
	v_cmp_u_f32_e64 s30, v130, v130                            // 000000004b64: d418001e 02030582
	s_wait_alu depctr_va_sdst(0)                               // 000000004b6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b70: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s30                    // 000000004b74: d501007a 007b017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004b7c: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004b88: 8c7e1f7e
	v_cmp_lt_i64_e64 s30, 6, v[38:39]                          // 000000004b8c: d451001e 02024c86
	s_and_b32 s31, s30, vcc_lo                                 // 000000004b94: 8b1f6a1e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b98: bf88ff9e
	s_and_saveexec_b32 s33, s31                                // 000000004b9c: bea1201f
	s_cbranch_execz 28                                         // 000000004ba0: bfa5001c <packed_folded_w4a8+0x3114>
	v_bfe_u32 v118, v131, 16, 1                                // 000000004ba4: d6100076 02052183
	s_wait_kmcnt 0x0                                           // 000000004bac: bfc70000
	v_add_co_u32 v119, s31, s34, v32                           // 000000004bb0: d7001f77 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000004bb8: bf88f19f
	v_add_co_ci_u32_e64 v122, null, s35, v33, s31              // 000000004bbc: d5207c7a 007e4223
	v_add3_u32 v123, v118, v131, 0x7fff                        // 000000004bc4: d655007b 03ff0776 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004bd0: bf870003
	v_add_co_u32 v118, s31, v119, v66                          // 000000004bd4: d7001f76 02028577
	v_or_b32_e32 v128, 0x400000, v131                          // 000000004bdc: 390106ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004be4: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v122, v67, s31             // 000000004be8: d5207c77 007e877a
	v_cmp_u_f32_e64 s31, v131, v131                            // 000000004bf0: d418001f 02030783
	s_wait_alu depctr_va_sdst(0)                               // 000000004bf8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004bfc: bf870001
	v_cndmask_b32_e64 v122, v123, v128, s31                    // 000000004c00: d501007a 007f017b
	global_store_d16_hi_b16 v[118:119], v122, off              // 000000004c08: ee09407c 3d000000 00000076
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004c14: 8c7e217e
	v_cmp_lt_i64_e64 s31, 7, v[38:39]                          // 000000004c18: d451001f 02024c87
	s_and_b32 s36, s31, vcc_lo                                 // 000000004c20: 8b246a1f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c24: bf88ff9e
	s_and_saveexec_b32 s33, s36                                // 000000004c28: bea12024
	s_cbranch_execz 25                                         // 000000004c2c: bfa50019 <packed_folded_w4a8+0x3194>
	v_bfe_u32 v38, v132, 16, 1                                 // 000000004c30: d6100026 02052184
	s_wait_kmcnt 0x0                                           // 000000004c38: bfc70000
	v_add_co_u32 v39, vcc_lo, s34, v32                         // 000000004c3c: d7006a27 02024022
	s_wait_alu depctr_va_vcc(0)                                // 000000004c44: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, s35, v33, vcc_lo           // 000000004c48: d5207c76 01aa4223
	v_add3_u32 v119, v38, v132, 0x7fff                         // 000000004c50: d6550077 03ff0926 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004c5c: bf870003
	v_add_co_u32 v38, vcc_lo, v39, v68                         // 000000004c60: d7006a26 02028927
	v_or_b32_e32 v122, 0x400000, v132                          // 000000004c68: 38f508ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c70: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, v118, v69, vcc_lo           // 000000004c74: d5207c27 01aa8b76
	v_cmp_u_f32_e32 vcc_lo, v132, v132                         // 000000004c7c: 7c310984
	s_wait_alu depctr_va_vcc(0)                                // 000000004c80: bf88ff9d
	v_cndmask_b32_e32 v118, v119, v122, vcc_lo                 // 000000004c84: 02ecf577
	global_store_d16_hi_b16 v[38:39], v118, off                // 000000004c88: ee09407c 3b000000 00000026
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004c94: 8c7e217e
	v_or_b32_e32 v38, v125, v127                               // 000000004c98: 384cff7d
	v_mov_b32_e32 v39, v126                                    // 000000004c9c: 7e4e037e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000004ca0: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[38:39]                // 000000004ca4: 7ca84c26
	s_wait_alu depctr_va_vcc(0)                                // 000000004ca8: bf88ff9d
	v_dual_cndmask_b32 v38, 0, v38 :: v_dual_cndmask_b32 v39, 0, v39// 000000004cac: ca524c80 26264e80
	v_add_co_u32 v38, s33, s40, v38                            // 000000004cb4: d7002126 02024c28
	s_delay_alu instid0(valu_dep_1)                            // 000000004cbc: bf870001
	v_add_co_ci_u32_e64 v39, null, s41, v39, s33               // 000000004cc0: d5207c27 00864e29
	global_load_u8 v119, v[38:39], off                         // 000000004cc8: ee04007c 00000077 00000026
	global_load_b32 v118, v[72:73], off                        // 000000004cd4: ee05007c 00000076 00000048
	s_wait_loadcnt 0x1                                         // 000000004ce0: bfc00001
	v_lshlrev_b32_e32 v73, 23, v119                            // 000000004ce4: 3092ee97
	s_wait_loadcnt 0x0                                         // 000000004ce8: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004cec: bf870091
	v_mul_f32_e32 v72, v118, v73                               // 000000004cf0: 10909376
	v_cmp_class_f32_e64 s33, v72, 0x198                        // 000000004cf4: d47e0021 0201ff48 00000198
	v_mul_f32_e32 v72, v24, v72                                // 000000004d00: 10909118
	s_xor_b32 s33, s33, -1                                     // 000000004d04: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d08: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004d0c: bea42021
	s_cbranch_execnz 2199                                      // 000000004d10: bfa60897 <packed_folded_w4a8+0x5470>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004d18: 8c7e247e
	global_load_b32 v74, v[74:75], off                         // 000000004d1c: ee05007c 0000004a 0000004a
	s_wait_loadcnt 0x0                                         // 000000004d28: bfc00000
	v_mul_f32_e32 v24, v74, v73                                // 000000004d2c: 1030934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004d30: bf870001
	v_cmp_class_f32_e64 s33, v24, 0x198                        // 000000004d34: d47e0021 0201ff18 00000198
	v_mul_f32_e32 v24, v25, v24                                // 000000004d40: 10303119
	s_xor_b32 s33, s33, -1                                     // 000000004d44: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d48: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004d4c: bea42021
	s_cbranch_execnz 2201                                      // 000000004d50: bfa60899 <packed_folded_w4a8+0x54b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004d58: 8c7e247e
	global_load_b32 v74, v[76:77], off                         // 000000004d5c: ee05007c 0000004a 0000004c
	s_wait_loadcnt 0x0                                         // 000000004d68: bfc00000
	v_mul_f32_e32 v25, v74, v73                                // 000000004d6c: 1032934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004d70: bf870001
	v_cmp_class_f32_e64 s33, v25, 0x198                        // 000000004d74: d47e0021 0201ff19 00000198
	v_mul_f32_e32 v25, v26, v25                                // 000000004d80: 1032331a
	s_xor_b32 s33, s33, -1                                     // 000000004d84: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d88: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004d8c: bea42021
	s_cbranch_execnz 2203                                      // 000000004d90: bfa6089b <packed_folded_w4a8+0x5500>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004d98: 8c7e247e
	global_load_b32 v74, v[78:79], off                         // 000000004d9c: ee05007c 0000004a 0000004e
	s_wait_loadcnt 0x0                                         // 000000004da8: bfc00000
	v_mul_f32_e32 v26, v74, v73                                // 000000004dac: 1034934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004db0: bf870001
	v_cmp_class_f32_e64 s33, v26, 0x198                        // 000000004db4: d47e0021 0201ff1a 00000198
	v_mul_f32_e32 v26, v27, v26                                // 000000004dc0: 1034351b
	s_xor_b32 s33, s33, -1                                     // 000000004dc4: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004dc8: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004dcc: bea42021
	s_cbranch_execnz 2205                                      // 000000004dd0: bfa6089d <packed_folded_w4a8+0x5548>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004dd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004dd8: 8c7e247e
	global_load_b32 v74, v[80:81], off                         // 000000004ddc: ee05007c 0000004a 00000050
	s_wait_loadcnt 0x0                                         // 000000004de8: bfc00000
	v_mul_f32_e32 v27, v74, v73                                // 000000004dec: 1036934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004df0: bf870001
	v_cmp_class_f32_e64 s33, v27, 0x198                        // 000000004df4: d47e0021 0201ff1b 00000198
	v_mul_f32_e32 v27, v28, v27                                // 000000004e00: 1036371c
	s_xor_b32 s33, s33, -1                                     // 000000004e04: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e08: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004e0c: bea42021
	s_cbranch_execnz 2207                                      // 000000004e10: bfa6089f <packed_folded_w4a8+0x5590>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e18: 8c7e247e
	global_load_b32 v74, v[82:83], off                         // 000000004e1c: ee05007c 0000004a 00000052
	s_wait_loadcnt 0x0                                         // 000000004e28: bfc00000
	v_mul_f32_e32 v28, v74, v73                                // 000000004e2c: 1038934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004e30: bf870001
	v_cmp_class_f32_e64 s33, v28, 0x198                        // 000000004e34: d47e0021 0201ff1c 00000198
	v_mul_f32_e32 v28, v29, v28                                // 000000004e40: 1038391d
	s_xor_b32 s33, s33, -1                                     // 000000004e44: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e48: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004e4c: bea42021
	s_cbranch_execnz 2209                                      // 000000004e50: bfa608a1 <packed_folded_w4a8+0x55d8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e58: 8c7e247e
	global_load_b32 v74, v[84:85], off                         // 000000004e5c: ee05007c 0000004a 00000054
	s_wait_loadcnt 0x0                                         // 000000004e68: bfc00000
	v_mul_f32_e32 v29, v74, v73                                // 000000004e6c: 103a934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004e70: bf870001
	v_cmp_class_f32_e64 s33, v29, 0x198                        // 000000004e74: d47e0021 0201ff1d 00000198
	v_mul_f32_e32 v29, v30, v29                                // 000000004e80: 103a3b1e
	s_xor_b32 s33, s33, -1                                     // 000000004e84: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e88: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004e8c: bea42021
	s_cbranch_execnz 2211                                      // 000000004e90: bfa608a3 <packed_folded_w4a8+0x5620>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004e98: 8c7e247e
	global_load_b32 v74, v[86:87], off                         // 000000004e9c: ee05007c 0000004a 00000056
	s_wait_loadcnt 0x0                                         // 000000004ea8: bfc00000
	v_mul_f32_e32 v30, v74, v73                                // 000000004eac: 103c934a
	s_delay_alu instid0(valu_dep_1)                            // 000000004eb0: bf870001
	v_cmp_class_f32_e64 s33, v30, 0x198                        // 000000004eb4: d47e0021 0201ff1e 00000198
	v_mul_f32_e32 v30, v31, v30                                // 000000004ec0: 103c3d1f
	s_xor_b32 s33, s33, -1                                     // 000000004ec4: 8d21c121
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ec8: bf88ff9e
	s_and_saveexec_b32 s36, s33                                // 000000004ecc: bea42021
	s_cbranch_execnz 2213                                      // 000000004ed0: bfa608a5 <packed_folded_w4a8+0x5668>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ed4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s36                             // 000000004ed8: 8c7e247e
	s_and_b32 s0, s0, vcc_lo                                   // 000000004edc: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ee0: bf88ff9e
	s_and_saveexec_b32 s33, s0                                 // 000000004ee4: bea12000
	s_cbranch_execz 34                                         // 000000004ee8: bfa50022 <packed_folded_w4a8+0x3474>
	v_add_co_u32 v73, s0, v125, v124                           // 000000004eec: d7000049 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000004ef4: bf88f19f
	v_add_co_ci_u32_e64 v74, null, 0, v126, s0                 // 000000004ef8: d5207c4a 0002fc80
	s_wait_kmcnt 0x0                                           // 000000004f00: bfc70000
	v_add_co_u32 v75, s0, s34, v70                             // 000000004f04: d700004b 02028c22
	v_bfe_u32 v31, v72, 16, 1                                  // 000000004f0c: d610001f 02052148
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000004f14: bf870253
	v_lshlrev_b64_e32 v[73:74], 1, v[73:74]                    // 000000004f18: 3e929281
	s_wait_alu depctr_va_sdst(0)                               // 000000004f1c: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s35, v71, s0                // 000000004f20: d5207c4c 00028e23
	v_or_b32_e32 v77, 0x400000, v72                            // 000000004f28: 389a90ff 00400000
	v_add3_u32 v31, v31, v72, 0x7fff                           // 000000004f30: d655001f 03fe911f 00007fff
	v_add_co_u32 v73, s0, v75, v73                             // 000000004f3c: d7000049 0202934b
	s_wait_alu depctr_va_sdst(0)                               // 000000004f44: bf88f19f
	v_add_co_ci_u32_e64 v74, null, v76, v74, s0                // 000000004f48: d5207c4a 0002954c
	v_cmp_u_f32_e64 s0, v72, v72                               // 000000004f50: d4180000 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000004f58: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004f5c: bf870001
	v_cndmask_b32_e64 v31, v31, v77, s0                        // 000000004f60: d501001f 00029b1f
	global_store_d16_hi_b16 v[73:74], v31, off offset:32       // 000000004f68: ee09407c 0f800000 00002049
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004f78: 8c7e217e
	s_and_b32 s0, s1, vcc_lo                                   // 000000004f7c: 8b006a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f80: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004f84: be812000
	s_cbranch_execz 28                                         // 000000004f88: bfa5001c <packed_folded_w4a8+0x34fc>
	s_wait_kmcnt 0x0                                           // 000000004f8c: bfc70000
	v_add_co_u32 v72, s0, s34, v70                             // 000000004f90: d7000048 02028c22
	v_bfe_u32 v31, v24, 16, 1                                  // 000000004f98: d610001f 02052118
	s_wait_alu depctr_va_sdst(0)                               // 000000004fa0: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s35, v71, s0                // 000000004fa4: d5207c49 00028e23
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004fac: bf870193
	v_add_co_u32 v72, s0, v72, v56                             // 000000004fb0: d7000048 02027148
	v_add3_u32 v31, v31, v24, 0x7fff                           // 000000004fb8: d655001f 03fe311f 00007fff
	v_or_b32_e32 v74, 0x400000, v24                            // 000000004fc4: 389430ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004fcc: bf88f19f
	v_add_co_ci_u32_e64 v73, null, v73, v57, s0                // 000000004fd0: d5207c49 00027349
	v_cmp_u_f32_e64 s0, v24, v24                               // 000000004fd8: d4180000 02023118
	s_wait_alu depctr_va_sdst(0)                               // 000000004fe0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004fe4: bf870001
	v_cndmask_b32_e64 v24, v31, v74, s0                        // 000000004fe8: d5010018 0002951f
	global_store_d16_hi_b16 v[72:73], v24, off offset:32       // 000000004ff0: ee09407c 0c000000 00002048
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ffc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005000: 8c7e017e
	s_and_b32 s0, s2, vcc_lo                                   // 000000005004: 8b006a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000005008: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000500c: be812000
	s_cbranch_execz 28                                         // 000000005010: bfa5001c <packed_folded_w4a8+0x3584>
	s_wait_kmcnt 0x0                                           // 000000005014: bfc70000
	v_add_co_u32 v31, s0, s34, v70                             // 000000005018: d700001f 02028c22
	v_bfe_u32 v24, v25, 16, 1                                  // 000000005020: d6100018 02052119
	s_wait_alu depctr_va_sdst(0)                               // 000000005028: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s35, v71, s0                // 00000000502c: d5207c49 00028e23
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005034: bf870193
	v_add_co_u32 v72, s0, v31, v58                             // 000000005038: d7000048 0202751f
	v_add3_u32 v24, v24, v25, 0x7fff                           // 000000005040: d6550018 03fe3318 00007fff
	v_or_b32_e32 v74, 0x400000, v25                            // 00000000504c: 389432ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005054: bf88f19f
	v_add_co_ci_u32_e64 v73, null, v73, v59, s0                // 000000005058: d5207c49 00027749
	v_cmp_u_f32_e64 s0, v25, v25                               // 000000005060: d4180000 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000005068: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000506c: bf870001
	v_cndmask_b32_e64 v24, v24, v74, s0                        // 000000005070: d5010018 00029518
	global_store_d16_hi_b16 v[72:73], v24, off offset:32       // 000000005078: ee09407c 0c000000 00002048
	s_wait_alu depctr_sa_sdst(0)                               // 000000005084: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005088: 8c7e017e
	s_and_b32 s0, s3, vcc_lo                                   // 00000000508c: 8b006a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000005090: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005094: be812000
	s_cbranch_execz 28                                         // 000000005098: bfa5001c <packed_folded_w4a8+0x360c>
	v_bfe_u32 v24, v26, 16, 1                                  // 00000000509c: d6100018 0205211a
	s_wait_kmcnt 0x0                                           // 0000000050a4: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000050a8: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000050b0: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s35, v71, s0                // 0000000050b4: d5207c1f 00028e23
	v_add3_u32 v72, v24, v26, 0x7fff                           // 0000000050bc: d6550048 03fe3518 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000050c8: bf870003
	v_add_co_u32 v24, s0, v25, v60                             // 0000000050cc: d7000018 02027919
	v_or_b32_e32 v73, 0x400000, v26                            // 0000000050d4: 389234ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000050dc: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v31, v61, s0                // 0000000050e0: d5207c19 00027b1f
	v_cmp_u_f32_e64 s0, v26, v26                               // 0000000050e8: d4180000 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 0000000050f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000050f4: bf870001
	v_cndmask_b32_e64 v26, v72, v73, s0                        // 0000000050f8: d501001a 00029348
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005100: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 00000000510c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005110: 8c7e017e
	s_and_b32 s0, s4, vcc_lo                                   // 000000005114: 8b006a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000005118: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000511c: be812000
	s_cbranch_execz 28                                         // 000000005120: bfa5001c <packed_folded_w4a8+0x3694>
	v_bfe_u32 v24, v27, 16, 1                                  // 000000005124: d6100018 0205211b
	s_wait_kmcnt 0x0                                           // 00000000512c: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 000000005130: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000005138: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 00000000513c: d5207c1a 00028e23
	v_add3_u32 v31, v24, v27, 0x7fff                           // 000000005144: d655001f 03fe3718 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005150: bf870003
	v_add_co_u32 v24, s0, v25, v62                             // 000000005154: d7000018 02027d19
	v_or_b32_e32 v72, 0x400000, v27                            // 00000000515c: 389036ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005164: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v63, s0                // 000000005168: d5207c19 00027f1a
	v_cmp_u_f32_e64 s0, v27, v27                               // 000000005170: d4180000 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000005178: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000517c: bf870001
	v_cndmask_b32_e64 v26, v31, v72, s0                        // 000000005180: d501001a 0002911f
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005188: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005194: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005198: 8c7e017e
	s_and_b32 s0, s5, vcc_lo                                   // 00000000519c: 8b006a05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000051a4: be812000
	s_cbranch_execz 28                                         // 0000000051a8: bfa5001c <packed_folded_w4a8+0x371c>
	v_bfe_u32 v24, v28, 16, 1                                  // 0000000051ac: d6100018 0205211c
	s_wait_kmcnt 0x0                                           // 0000000051b4: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000051b8: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000051c0: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 0000000051c4: d5207c1a 00028e23
	v_add3_u32 v27, v24, v28, 0x7fff                           // 0000000051cc: d655001b 03fe3918 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000051d8: bf870003
	v_add_co_u32 v24, s0, v25, v64                             // 0000000051dc: d7000018 02028119
	v_or_b32_e32 v31, 0x400000, v28                            // 0000000051e4: 383e38ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000051ec: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v65, s0                // 0000000051f0: d5207c19 0002831a
	v_cmp_u_f32_e64 s0, v28, v28                               // 0000000051f8: d4180000 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000005200: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005204: bf870001
	v_cndmask_b32_e64 v26, v27, v31, s0                        // 000000005208: d501001a 00023f1b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005210: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 00000000521c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005220: 8c7e017e
	s_and_b32 s0, s6, vcc_lo                                   // 000000005224: 8b006a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005228: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000522c: be812000
	s_cbranch_execz 28                                         // 000000005230: bfa5001c <packed_folded_w4a8+0x37a4>
	v_bfe_u32 v24, v29, 16, 1                                  // 000000005234: d6100018 0205211d
	s_wait_kmcnt 0x0                                           // 00000000523c: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 000000005240: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 000000005248: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 00000000524c: d5207c1a 00028e23
	v_add3_u32 v27, v24, v29, 0x7fff                           // 000000005254: d655001b 03fe3b18 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005260: bf870003
	v_add_co_u32 v24, s0, v25, v66                             // 000000005264: d7000018 02028519
	v_or_b32_e32 v28, 0x400000, v29                            // 00000000526c: 38383aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005274: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v67, s0                // 000000005278: d5207c19 0002871a
	v_cmp_u_f32_e64 s0, v29, v29                               // 000000005280: d4180000 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000005288: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000528c: bf870001
	v_cndmask_b32_e64 v26, v27, v28, s0                        // 000000005290: d501001a 0002391b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005298: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000052a8: 8c7e017e
	s_and_b32 s0, s7, vcc_lo                                   // 0000000052ac: 8b006a07
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052b0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000052b4: be812000
	s_cbranch_execz 28                                         // 0000000052b8: bfa5001c <packed_folded_w4a8+0x382c>
	v_bfe_u32 v24, v30, 16, 1                                  // 0000000052bc: d6100018 0205211e
	s_wait_kmcnt 0x0                                           // 0000000052c4: bfc70000
	v_add_co_u32 v25, s0, s34, v70                             // 0000000052c8: d7000019 02028c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000052d0: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s35, v71, s0                // 0000000052d4: d5207c1a 00028e23
	v_add3_u32 v27, v24, v30, 0x7fff                           // 0000000052dc: d655001b 03fe3d18 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000052e8: bf870003
	v_add_co_u32 v24, s0, v25, v68                             // 0000000052ec: d7000018 02028919
	v_or_b32_e32 v28, 0x400000, v30                            // 0000000052f4: 38383cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000052fc: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v69, s0                // 000000005300: d5207c19 00028b1a
	v_cmp_u_f32_e64 s0, v30, v30                               // 000000005308: d4180000 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000005310: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005314: bf870001
	v_cndmask_b32_e64 v26, v27, v28, s0                        // 000000005318: d501001a 0002391b
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005320: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 00000000532c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005330: 8c7e017e
	global_load_u8 v24, v[38:39], off                          // 000000005334: ee04007c 00000018 00000026
	global_load_b32 v26, v[88:89], off                         // 000000005340: ee05007c 0000001a 00000058
	s_wait_loadcnt 0x1                                         // 00000000534c: bfc00001
	v_lshlrev_b32_e32 v25, 23, v24                             // 000000005350: 30323097
	s_wait_loadcnt 0x0                                         // 000000005354: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005358: bf870091
	v_mul_f32_e32 v24, v26, v25                                // 00000000535c: 1030331a
	v_cmp_class_f32_e64 s0, v24, 0x198                         // 000000005360: d47e0000 0201ff18 00000198
	v_mul_f32_e32 v24, v16, v24                                // 00000000536c: 10303110
	s_xor_b32 s0, s0, -1                                       // 000000005370: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005374: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005378: be812000
	s_cbranch_execnz 1932                                      // 00000000537c: bfa6078c <packed_folded_w4a8+0x56b0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005380: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005384: 8c7e017e
	global_load_b32 v26, v[90:91], off                         // 000000005388: ee05007c 0000001a 0000005a
	s_wait_loadcnt 0x0                                         // 000000005394: bfc00000
	v_mul_f32_e32 v16, v26, v25                                // 000000005398: 1020331a
	s_delay_alu instid0(valu_dep_1)                            // 00000000539c: bf870001
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 0000000053a0: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v17, v16                                // 0000000053ac: 10202111
	s_xor_b32 s0, s0, -1                                       // 0000000053b0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053b4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000053b8: be812000
	s_cbranch_execnz 1934                                      // 0000000053bc: bfa6078e <packed_folded_w4a8+0x56f8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000053c4: 8c7e017e
	global_load_b32 v26, v[92:93], off                         // 0000000053c8: ee05007c 0000001a 0000005c
	s_wait_loadcnt 0x0                                         // 0000000053d4: bfc00000
	v_mul_f32_e32 v17, v26, v25                                // 0000000053d8: 1022331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000053dc: bf870001
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 0000000053e0: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v18, v17                                // 0000000053ec: 10222312
	s_xor_b32 s0, s0, -1                                       // 0000000053f0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000053f8: be812000
	s_cbranch_execnz 1936                                      // 0000000053fc: bfa60790 <packed_folded_w4a8+0x5740>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005400: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005404: 8c7e017e
	global_load_b32 v26, v[94:95], off                         // 000000005408: ee05007c 0000001a 0000005e
	s_wait_loadcnt 0x0                                         // 000000005414: bfc00000
	v_mul_f32_e32 v18, v26, v25                                // 000000005418: 1024331a
	s_delay_alu instid0(valu_dep_1)                            // 00000000541c: bf870001
	v_cmp_class_f32_e64 s0, v18, 0x198                         // 000000005420: d47e0000 0201ff12 00000198
	v_mul_f32_e32 v18, v19, v18                                // 00000000542c: 10242513
	s_xor_b32 s0, s0, -1                                       // 000000005430: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005434: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005438: be812000
	s_cbranch_execnz 1938                                      // 00000000543c: bfa60792 <packed_folded_w4a8+0x5788>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005440: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005444: 8c7e017e
	global_load_b32 v26, v[50:51], off                         // 000000005448: ee05007c 0000001a 00000032
	s_wait_loadcnt 0x0                                         // 000000005454: bfc00000
	v_mul_f32_e32 v19, v26, v25                                // 000000005458: 1026331a
	s_delay_alu instid0(valu_dep_1)                            // 00000000545c: bf870001
	v_cmp_class_f32_e64 s0, v19, 0x198                         // 000000005460: d47e0000 0201ff13 00000198
	v_mul_f32_e32 v19, v20, v19                                // 00000000546c: 10262714
	s_xor_b32 s0, s0, -1                                       // 000000005470: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005474: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005478: be812000
	s_cbranch_execnz 1940                                      // 00000000547c: bfa60794 <packed_folded_w4a8+0x57d0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005480: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005484: 8c7e017e
	global_load_b32 v26, v[96:97], off                         // 000000005488: ee05007c 0000001a 00000060
	s_wait_loadcnt 0x0                                         // 000000005494: bfc00000
	v_mul_f32_e32 v20, v26, v25                                // 000000005498: 1028331a
	s_delay_alu instid0(valu_dep_1)                            // 00000000549c: bf870001
	v_cmp_class_f32_e64 s0, v20, 0x198                         // 0000000054a0: d47e0000 0201ff14 00000198
	v_mul_f32_e32 v20, v21, v20                                // 0000000054ac: 10282915
	s_xor_b32 s0, s0, -1                                       // 0000000054b0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054b4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000054b8: be812000
	s_cbranch_execnz 1942                                      // 0000000054bc: bfa60796 <packed_folded_w4a8+0x5818>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000054c4: 8c7e017e
	global_load_b32 v26, v[52:53], off                         // 0000000054c8: ee05007c 0000001a 00000034
	s_wait_loadcnt 0x0                                         // 0000000054d4: bfc00000
	v_mul_f32_e32 v21, v26, v25                                // 0000000054d8: 102a331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000054dc: bf870001
	v_cmp_class_f32_e64 s0, v21, 0x198                         // 0000000054e0: d47e0000 0201ff15 00000198
	v_mul_f32_e32 v21, v22, v21                                // 0000000054ec: 102a2b16
	s_xor_b32 s0, s0, -1                                       // 0000000054f0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000054f8: be812000
	s_cbranch_execnz 1944                                      // 0000000054fc: bfa60798 <packed_folded_w4a8+0x5860>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005500: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005504: 8c7e017e
	global_load_b32 v26, v[98:99], off                         // 000000005508: ee05007c 0000001a 00000062
	s_wait_loadcnt 0x0                                         // 000000005514: bfc00000
	v_mul_f32_e32 v22, v26, v25                                // 000000005518: 102c331a
	s_delay_alu instid0(valu_dep_1)                            // 00000000551c: bf870001
	v_cmp_class_f32_e64 s0, v22, 0x198                         // 000000005520: d47e0000 0201ff16 00000198
	v_mul_f32_e32 v22, v23, v22                                // 00000000552c: 102c2d17
	s_xor_b32 s0, s0, -1                                       // 000000005530: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005534: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005538: be812000
	s_cbranch_execnz 1946                                      // 00000000553c: bfa6079a <packed_folded_w4a8+0x58a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005540: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005544: 8c7e017e
	s_and_b32 s0, s11, vcc_lo                                  // 000000005548: 8b006a0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000554c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005550: be812000
	s_cbranch_execz 34                                         // 000000005554: bfa50022 <packed_folded_w4a8+0x3ae0>
	v_add_co_u32 v25, s0, v125, v124                           // 000000005558: d7000019 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000005560: bf88f19f
	v_add_co_ci_u32_e64 v26, null, 0, v126, s0                 // 000000005564: d5207c1a 0002fc80
	s_wait_kmcnt 0x0                                           // 00000000556c: bfc70000
	v_add_co_u32 v27, s0, s34, v48                             // 000000005570: d700001b 02026022
	v_bfe_u32 v23, v24, 16, 1                                  // 000000005578: d6100017 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000005580: bf870253
	v_lshlrev_b64_e32 v[25:26], 1, v[25:26]                    // 000000005584: 3e323281
	s_wait_alu depctr_va_sdst(0)                               // 000000005588: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s35, v49, s0                // 00000000558c: d5207c1c 00026223
	v_or_b32_e32 v29, 0x400000, v24                            // 000000005594: 383a30ff 00400000
	v_add3_u32 v23, v23, v24, 0x7fff                           // 00000000559c: d6550017 03fe3117 00007fff
	v_add_co_u32 v25, s0, v27, v25                             // 0000000055a8: d7000019 0202331b
	s_wait_alu depctr_va_sdst(0)                               // 0000000055b0: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v28, v26, s0                // 0000000055b4: d5207c1a 0002351c
	v_cmp_u_f32_e64 s0, v24, v24                               // 0000000055bc: d4180000 02023118
	s_wait_alu depctr_va_sdst(0)                               // 0000000055c4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000055c8: bf870001
	v_cndmask_b32_e64 v23, v23, v29, s0                        // 0000000055cc: d5010017 00023b17
	global_store_d16_hi_b16 v[25:26], v23, off offset:32       // 0000000055d4: ee09407c 0b800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000055e4: 8c7e017e
	s_and_b32 s0, s8, vcc_lo                                   // 0000000055e8: 8b006a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000055f0: be812000
	s_cbranch_execz 28                                         // 0000000055f4: bfa5001c <packed_folded_w4a8+0x3b68>
	v_bfe_u32 v23, v16, 16, 1                                  // 0000000055f8: d6100017 02052110
	s_wait_kmcnt 0x0                                           // 000000005600: bfc70000
	v_add_co_u32 v24, s0, s34, v48                             // 000000005604: d7000018 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000560c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s35, v49, s0                // 000000005610: d5207c19 00026223
	v_add3_u32 v26, v23, v16, 0x7fff                           // 000000005618: d655001a 03fe2117 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005624: bf870003
	v_add_co_u32 v23, s0, v24, v56                             // 000000005628: d7000017 02027118
	v_or_b32_e32 v27, 0x400000, v16                            // 000000005630: 383620ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005638: bf88f19f
	v_add_co_ci_u32_e64 v24, null, v25, v57, s0                // 00000000563c: d5207c18 00027319
	v_cmp_u_f32_e64 s0, v16, v16                               // 000000005644: d4180000 02022110
	s_wait_alu depctr_va_sdst(0)                               // 00000000564c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005650: bf870001
	v_cndmask_b32_e64 v16, v26, v27, s0                        // 000000005654: d5010010 0002371a
	global_store_d16_hi_b16 v[23:24], v16, off offset:32       // 00000000565c: ee09407c 08000000 00002017
	s_wait_alu depctr_sa_sdst(0)                               // 000000005668: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000566c: 8c7e017e
	s_and_b32 s0, s9, vcc_lo                                   // 000000005670: 8b006a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000005674: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005678: be812000
	s_cbranch_execz 28                                         // 00000000567c: bfa5001c <packed_folded_w4a8+0x3bf0>
	s_wait_kmcnt 0x0                                           // 000000005680: bfc70000
	v_add_co_u32 v23, s0, s34, v48                             // 000000005684: d7000017 02026022
	v_bfe_u32 v16, v17, 16, 1                                  // 00000000568c: d6100010 02052111
	s_wait_alu depctr_va_sdst(0)                               // 000000005694: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s35, v49, s0                // 000000005698: d5207c18 00026223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000056a0: bf870193
	v_add_co_u32 v23, s0, v23, v58                             // 0000000056a4: d7000017 02027517
	v_add3_u32 v16, v16, v17, 0x7fff                           // 0000000056ac: d6550010 03fe2310 00007fff
	v_or_b32_e32 v25, 0x400000, v17                            // 0000000056b8: 383222ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000056c0: bf88f19f
	v_add_co_ci_u32_e64 v24, null, v24, v59, s0                // 0000000056c4: d5207c18 00027718
	v_cmp_u_f32_e64 s0, v17, v17                               // 0000000056cc: d4180000 02022311
	s_wait_alu depctr_va_sdst(0)                               // 0000000056d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000056d8: bf870001
	v_cndmask_b32_e64 v16, v16, v25, s0                        // 0000000056dc: d5010010 00023310
	global_store_d16_hi_b16 v[23:24], v16, off offset:32       // 0000000056e4: ee09407c 08000000 00002017
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000056f4: 8c7e017e
	s_and_b32 s0, s10, vcc_lo                                  // 0000000056f8: 8b006a0a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056fc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005700: be812000
	s_cbranch_execz 28                                         // 000000005704: bfa5001c <packed_folded_w4a8+0x3c78>
	v_bfe_u32 v16, v18, 16, 1                                  // 000000005708: d6100010 02052112
	s_wait_kmcnt 0x0                                           // 000000005710: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 000000005714: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000571c: bf88f19f
	v_add_co_ci_u32_e64 v23, null, s35, v49, s0                // 000000005720: d5207c17 00026223
	v_add3_u32 v24, v16, v18, 0x7fff                           // 000000005728: d6550018 03fe2510 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005734: bf870003
	v_add_co_u32 v16, s0, v17, v60                             // 000000005738: d7000010 02027911
	v_or_b32_e32 v25, 0x400000, v18                            // 000000005740: 383224ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005748: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v23, v61, s0                // 00000000574c: d5207c11 00027b17
	v_cmp_u_f32_e64 s0, v18, v18                               // 000000005754: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 00000000575c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005760: bf870001
	v_cndmask_b32_e64 v18, v24, v25, s0                        // 000000005764: d5010012 00023318
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 00000000576c: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005778: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000577c: 8c7e017e
	s_and_b32 s0, s12, vcc_lo                                  // 000000005780: 8b006a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000005784: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005788: be812000
	s_cbranch_execz 28                                         // 00000000578c: bfa5001c <packed_folded_w4a8+0x3d00>
	v_bfe_u32 v16, v19, 16, 1                                  // 000000005790: d6100010 02052113
	s_wait_kmcnt 0x0                                           // 000000005798: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 00000000579c: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000057a4: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 0000000057a8: d5207c12 00026223
	v_add3_u32 v23, v16, v19, 0x7fff                           // 0000000057b0: d6550017 03fe2710 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000057bc: bf870003
	v_add_co_u32 v16, s0, v17, v62                             // 0000000057c0: d7000010 02027d11
	v_or_b32_e32 v24, 0x400000, v19                            // 0000000057c8: 383026ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000057d0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v63, s0                // 0000000057d4: d5207c11 00027f12
	v_cmp_u_f32_e64 s0, v19, v19                               // 0000000057dc: d4180000 02022713
	s_wait_alu depctr_va_sdst(0)                               // 0000000057e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000057e8: bf870001
	v_cndmask_b32_e64 v18, v23, v24, s0                        // 0000000057ec: d5010012 00023117
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 0000000057f4: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005800: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005804: 8c7e017e
	s_and_b32 s0, s13, vcc_lo                                  // 000000005808: 8b006a0d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000580c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005810: be812000
	s_cbranch_execz 28                                         // 000000005814: bfa5001c <packed_folded_w4a8+0x3d88>
	v_bfe_u32 v16, v20, 16, 1                                  // 000000005818: d6100010 02052114
	s_wait_kmcnt 0x0                                           // 000000005820: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 000000005824: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000582c: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 000000005830: d5207c12 00026223
	v_add3_u32 v19, v16, v20, 0x7fff                           // 000000005838: d6550013 03fe2910 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005844: bf870003
	v_add_co_u32 v16, s0, v17, v64                             // 000000005848: d7000010 02028111
	v_or_b32_e32 v23, 0x400000, v20                            // 000000005850: 382e28ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005858: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v65, s0                // 00000000585c: d5207c11 00028312
	v_cmp_u_f32_e64 s0, v20, v20                               // 000000005864: d4180000 02022914
	s_wait_alu depctr_va_sdst(0)                               // 00000000586c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005870: bf870001
	v_cndmask_b32_e64 v18, v19, v23, s0                        // 000000005874: d5010012 00022f13
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 00000000587c: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005888: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000588c: 8c7e017e
	s_and_b32 s0, s14, vcc_lo                                  // 000000005890: 8b006a0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000005894: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005898: be812000
	s_cbranch_execz 28                                         // 00000000589c: bfa5001c <packed_folded_w4a8+0x3e10>
	v_bfe_u32 v16, v21, 16, 1                                  // 0000000058a0: d6100010 02052115
	s_wait_kmcnt 0x0                                           // 0000000058a8: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 0000000058ac: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000058b4: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 0000000058b8: d5207c12 00026223
	v_add3_u32 v19, v16, v21, 0x7fff                           // 0000000058c0: d6550013 03fe2b10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000058cc: bf870003
	v_add_co_u32 v16, s0, v17, v66                             // 0000000058d0: d7000010 02028511
	v_or_b32_e32 v20, 0x400000, v21                            // 0000000058d8: 38282aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000058e0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v67, s0                // 0000000058e4: d5207c11 00028712
	v_cmp_u_f32_e64 s0, v21, v21                               // 0000000058ec: d4180000 02022b15
	s_wait_alu depctr_va_sdst(0)                               // 0000000058f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000058f8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 0000000058fc: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 000000005904: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005910: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005914: 8c7e017e
	s_and_b32 s0, s15, vcc_lo                                  // 000000005918: 8b006a0f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000591c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005920: be812000
	s_cbranch_execz 28                                         // 000000005924: bfa5001c <packed_folded_w4a8+0x3e98>
	v_bfe_u32 v16, v22, 16, 1                                  // 000000005928: d6100010 02052116
	s_wait_kmcnt 0x0                                           // 000000005930: bfc70000
	v_add_co_u32 v17, s0, s34, v48                             // 000000005934: d7000011 02026022
	s_wait_alu depctr_va_sdst(0)                               // 00000000593c: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s35, v49, s0                // 000000005940: d5207c12 00026223
	v_add3_u32 v19, v16, v22, 0x7fff                           // 000000005948: d6550013 03fe2d10 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005954: bf870003
	v_add_co_u32 v16, s0, v17, v68                             // 000000005958: d7000010 02028911
	v_or_b32_e32 v20, 0x400000, v22                            // 000000005960: 38282cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005968: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v69, s0                // 00000000596c: d5207c11 00028b12
	v_cmp_u_f32_e64 s0, v22, v22                               // 000000005974: d4180000 02022d16
	s_wait_alu depctr_va_sdst(0)                               // 00000000597c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005980: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000005984: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 00000000598c: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005998: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000599c: 8c7e017e
	global_load_u8 v16, v[38:39], off                          // 0000000059a0: ee04007c 00000010 00000026
	global_load_b32 v18, v[54:55], off                         // 0000000059ac: ee05007c 00000012 00000036
	s_wait_loadcnt 0x1                                         // 0000000059b8: bfc00001
	v_lshlrev_b32_e32 v17, 23, v16                             // 0000000059bc: 30222097
	s_wait_loadcnt 0x0                                         // 0000000059c0: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000059c4: bf870091
	v_mul_f32_e32 v16, v18, v17                                // 0000000059c8: 10202312
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 0000000059cc: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v8, v16                                 // 0000000059d8: 10202108
	s_xor_b32 s0, s0, -1                                       // 0000000059dc: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059e0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000059e4: be812000
	s_cbranch_execnz 1665                                      // 0000000059e8: bfa60681 <packed_folded_w4a8+0x58f0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000059f0: 8c7e017e
	global_load_b32 v18, v[100:101], off                       // 0000000059f4: ee05007c 00000012 00000064
	s_wait_loadcnt 0x0                                         // 000000005a00: bfc00000
	v_mul_f32_e32 v8, v18, v17                                 // 000000005a04: 10102312
	s_delay_alu instid0(valu_dep_1)                            // 000000005a08: bf870001
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000005a0c: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v9, v8                                   // 000000005a18: 10101109
	s_xor_b32 s0, s0, -1                                       // 000000005a1c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a20: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a24: be812000
	s_cbranch_execnz 1667                                      // 000000005a28: bfa60683 <packed_folded_w4a8+0x5938>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a30: 8c7e017e
	global_load_b32 v18, v[102:103], off                       // 000000005a34: ee05007c 00000012 00000066
	s_wait_loadcnt 0x0                                         // 000000005a40: bfc00000
	v_mul_f32_e32 v9, v18, v17                                 // 000000005a44: 10122312
	s_delay_alu instid0(valu_dep_1)                            // 000000005a48: bf870001
	v_cmp_class_f32_e64 s0, v9, 0x198                          // 000000005a4c: d47e0000 0201ff09 00000198
	v_mul_f32_e32 v9, v10, v9                                  // 000000005a58: 1012130a
	s_xor_b32 s0, s0, -1                                       // 000000005a5c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a60: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a64: be812000
	s_cbranch_execnz 1669                                      // 000000005a68: bfa60685 <packed_folded_w4a8+0x5980>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a70: 8c7e017e
	global_load_b32 v18, v[104:105], off                       // 000000005a74: ee05007c 00000012 00000068
	s_wait_loadcnt 0x0                                         // 000000005a80: bfc00000
	v_mul_f32_e32 v10, v18, v17                                // 000000005a84: 10142312
	s_delay_alu instid0(valu_dep_1)                            // 000000005a88: bf870001
	v_cmp_class_f32_e64 s0, v10, 0x198                         // 000000005a8c: d47e0000 0201ff0a 00000198
	v_mul_f32_e32 v10, v11, v10                                // 000000005a98: 1014150b
	s_xor_b32 s0, s0, -1                                       // 000000005a9c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005aa0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005aa4: be812000
	s_cbranch_execnz 1671                                      // 000000005aa8: bfa60687 <packed_folded_w4a8+0x59c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005aac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005ab0: 8c7e017e
	global_load_b32 v18, v[42:43], off                         // 000000005ab4: ee05007c 00000012 0000002a
	s_wait_loadcnt 0x0                                         // 000000005ac0: bfc00000
	v_mul_f32_e32 v11, v18, v17                                // 000000005ac4: 10162312
	s_delay_alu instid0(valu_dep_1)                            // 000000005ac8: bf870001
	v_cmp_class_f32_e64 s0, v11, 0x198                         // 000000005acc: d47e0000 0201ff0b 00000198
	v_mul_f32_e32 v11, v12, v11                                // 000000005ad8: 1016170c
	s_xor_b32 s0, s0, -1                                       // 000000005adc: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ae0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ae4: be812000
	s_cbranch_execnz 1673                                      // 000000005ae8: bfa60689 <packed_folded_w4a8+0x5a10>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005aec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005af0: 8c7e017e
	global_load_b32 v18, v[106:107], off                       // 000000005af4: ee05007c 00000012 0000006a
	s_wait_loadcnt 0x0                                         // 000000005b00: bfc00000
	v_mul_f32_e32 v12, v18, v17                                // 000000005b04: 10182312
	s_delay_alu instid0(valu_dep_1)                            // 000000005b08: bf870001
	v_cmp_class_f32_e64 s0, v12, 0x198                         // 000000005b0c: d47e0000 0201ff0c 00000198
	v_mul_f32_e32 v12, v13, v12                                // 000000005b18: 1018190d
	s_xor_b32 s0, s0, -1                                       // 000000005b1c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b20: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b24: be812000
	s_cbranch_execnz 1675                                      // 000000005b28: bfa6068b <packed_folded_w4a8+0x5a58>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b30: 8c7e017e
	global_load_b32 v18, v[44:45], off                         // 000000005b34: ee05007c 00000012 0000002c
	s_wait_loadcnt 0x0                                         // 000000005b40: bfc00000
	v_mul_f32_e32 v13, v18, v17                                // 000000005b44: 101a2312
	s_delay_alu instid0(valu_dep_1)                            // 000000005b48: bf870001
	v_cmp_class_f32_e64 s0, v13, 0x198                         // 000000005b4c: d47e0000 0201ff0d 00000198
	v_mul_f32_e32 v13, v14, v13                                // 000000005b58: 101a1b0e
	s_xor_b32 s0, s0, -1                                       // 000000005b5c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b60: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b64: be812000
	s_cbranch_execnz 1677                                      // 000000005b68: bfa6068d <packed_folded_w4a8+0x5aa0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b70: 8c7e017e
	global_load_b32 v18, v[108:109], off                       // 000000005b74: ee05007c 00000012 0000006c
	s_wait_loadcnt 0x0                                         // 000000005b80: bfc00000
	v_mul_f32_e32 v14, v18, v17                                // 000000005b84: 101c2312
	s_delay_alu instid0(valu_dep_1)                            // 000000005b88: bf870001
	v_cmp_class_f32_e64 s0, v14, 0x198                         // 000000005b8c: d47e0000 0201ff0e 00000198
	v_mul_f32_e32 v14, v15, v14                                // 000000005b98: 101c1d0f
	s_xor_b32 s0, s0, -1                                       // 000000005b9c: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ba0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ba4: be812000
	s_cbranch_execnz 1679                                      // 000000005ba8: bfa6068f <packed_folded_w4a8+0x5ae8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005bb0: 8c7e017e
	s_and_b32 s0, s19, vcc_lo                                  // 000000005bb4: 8b006a13
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bb8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005bbc: be812000
	s_cbranch_execz 34                                         // 000000005bc0: bfa50022 <packed_folded_w4a8+0x414c>
	v_add_co_u32 v17, s0, v125, v124                           // 000000005bc4: d7000011 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000005bcc: bf88f19f
	v_add_co_ci_u32_e64 v18, null, 0, v126, s0                 // 000000005bd0: d5207c12 0002fc80
	s_wait_kmcnt 0x0                                           // 000000005bd8: bfc70000
	v_add_co_u32 v19, s0, s34, v40                             // 000000005bdc: d7000013 02025022
	v_bfe_u32 v15, v16, 16, 1                                  // 000000005be4: d610000f 02052110
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000005bec: bf870253
	v_lshlrev_b64_e32 v[17:18], 1, v[17:18]                    // 000000005bf0: 3e222281
	s_wait_alu depctr_va_sdst(0)                               // 000000005bf4: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s35, v41, s0                // 000000005bf8: d5207c14 00025223
	v_or_b32_e32 v21, 0x400000, v16                            // 000000005c00: 382a20ff 00400000
	v_add3_u32 v15, v15, v16, 0x7fff                           // 000000005c08: d655000f 03fe210f 00007fff
	v_add_co_u32 v17, s0, v19, v17                             // 000000005c14: d7000011 02022313
	s_wait_alu depctr_va_sdst(0)                               // 000000005c1c: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v20, v18, s0                // 000000005c20: d5207c12 00022514
	v_cmp_u_f32_e64 s0, v16, v16                               // 000000005c28: d4180000 02022110
	s_wait_alu depctr_va_sdst(0)                               // 000000005c30: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005c34: bf870001
	v_cndmask_b32_e64 v15, v15, v21, s0                        // 000000005c38: d501000f 00022b0f
	global_store_d16_hi_b16 v[17:18], v15, off offset:32       // 000000005c40: ee09407c 07800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005c50: 8c7e017e
	s_and_b32 s0, s16, vcc_lo                                  // 000000005c54: 8b006a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c58: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005c5c: be812000
	s_cbranch_execz 28                                         // 000000005c60: bfa5001c <packed_folded_w4a8+0x41d4>
	v_bfe_u32 v15, v8, 16, 1                                   // 000000005c64: d610000f 02052108
	s_wait_kmcnt 0x0                                           // 000000005c6c: bfc70000
	v_add_co_u32 v16, s0, s34, v40                             // 000000005c70: d7000010 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005c78: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v41, s0                // 000000005c7c: d5207c11 00025223
	v_add3_u32 v18, v15, v8, 0x7fff                            // 000000005c84: d6550012 03fe110f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005c90: bf870003
	v_add_co_u32 v15, s0, v16, v56                             // 000000005c94: d700000f 02027110
	v_or_b32_e32 v19, 0x400000, v8                             // 000000005c9c: 382610ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ca4: bf88f19f
	v_add_co_ci_u32_e64 v16, null, v17, v57, s0                // 000000005ca8: d5207c10 00027311
	v_cmp_u_f32_e64 s0, v8, v8                                 // 000000005cb0: d4180000 02021108
	s_wait_alu depctr_va_sdst(0)                               // 000000005cb8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005cbc: bf870001
	v_cndmask_b32_e64 v8, v18, v19, s0                         // 000000005cc0: d5010008 00022712
	global_store_d16_hi_b16 v[15:16], v8, off offset:32        // 000000005cc8: ee09407c 04000000 0000200f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005cd8: 8c7e017e
	s_and_b32 s0, s17, vcc_lo                                  // 000000005cdc: 8b006a11
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ce0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ce4: be812000
	s_cbranch_execz 28                                         // 000000005ce8: bfa5001c <packed_folded_w4a8+0x425c>
	s_wait_kmcnt 0x0                                           // 000000005cec: bfc70000
	v_add_co_u32 v15, s0, s34, v40                             // 000000005cf0: d700000f 02025022
	v_bfe_u32 v8, v9, 16, 1                                    // 000000005cf8: d6100008 02052109
	s_wait_alu depctr_va_sdst(0)                               // 000000005d00: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s35, v41, s0                // 000000005d04: d5207c10 00025223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005d0c: bf870193
	v_add_co_u32 v15, s0, v15, v58                             // 000000005d10: d700000f 0202750f
	v_add3_u32 v8, v8, v9, 0x7fff                              // 000000005d18: d6550008 03fe1308 00007fff
	v_or_b32_e32 v17, 0x400000, v9                             // 000000005d24: 382212ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005d2c: bf88f19f
	v_add_co_ci_u32_e64 v16, null, v16, v59, s0                // 000000005d30: d5207c10 00027710
	v_cmp_u_f32_e64 s0, v9, v9                                 // 000000005d38: d4180000 02021309
	s_wait_alu depctr_va_sdst(0)                               // 000000005d40: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005d44: bf870001
	v_cndmask_b32_e64 v8, v8, v17, s0                          // 000000005d48: d5010008 00022308
	global_store_d16_hi_b16 v[15:16], v8, off offset:32        // 000000005d50: ee09407c 04000000 0000200f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005d60: 8c7e017e
	s_and_b32 s0, s18, vcc_lo                                  // 000000005d64: 8b006a12
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d68: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005d6c: be812000
	s_cbranch_execz 28                                         // 000000005d70: bfa5001c <packed_folded_w4a8+0x42e4>
	v_bfe_u32 v8, v10, 16, 1                                   // 000000005d74: d6100008 0205210a
	s_wait_kmcnt 0x0                                           // 000000005d7c: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005d80: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005d88: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s35, v41, s0                // 000000005d8c: d5207c0f 00025223
	v_add3_u32 v16, v8, v10, 0x7fff                            // 000000005d94: d6550010 03fe1508 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005da0: bf870003
	v_add_co_u32 v8, s0, v9, v60                               // 000000005da4: d7000008 02027909
	v_or_b32_e32 v17, 0x400000, v10                            // 000000005dac: 382214ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005db4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v15, v61, s0                 // 000000005db8: d5207c09 00027b0f
	v_cmp_u_f32_e64 s0, v10, v10                               // 000000005dc0: d4180000 0202150a
	s_wait_alu depctr_va_sdst(0)                               // 000000005dc8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005dcc: bf870001
	v_cndmask_b32_e64 v10, v16, v17, s0                        // 000000005dd0: d501000a 00022310
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005dd8: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005de4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005de8: 8c7e017e
	s_and_b32 s0, s20, vcc_lo                                  // 000000005dec: 8b006a14
	s_wait_alu depctr_sa_sdst(0)                               // 000000005df0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005df4: be812000
	s_cbranch_execz 28                                         // 000000005df8: bfa5001c <packed_folded_w4a8+0x436c>
	v_bfe_u32 v8, v11, 16, 1                                   // 000000005dfc: d6100008 0205210b
	s_wait_kmcnt 0x0                                           // 000000005e04: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005e08: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005e10: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005e14: d5207c0a 00025223
	v_add3_u32 v15, v8, v11, 0x7fff                            // 000000005e1c: d655000f 03fe1708 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005e28: bf870003
	v_add_co_u32 v8, s0, v9, v62                               // 000000005e2c: d7000008 02027d09
	v_or_b32_e32 v16, 0x400000, v11                            // 000000005e34: 382016ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005e3c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v63, s0                 // 000000005e40: d5207c09 00027f0a
	v_cmp_u_f32_e64 s0, v11, v11                               // 000000005e48: d4180000 0202170b
	s_wait_alu depctr_va_sdst(0)                               // 000000005e50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005e54: bf870001
	v_cndmask_b32_e64 v10, v15, v16, s0                        // 000000005e58: d501000a 0002210f
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005e60: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005e70: 8c7e017e
	s_and_b32 s0, s21, vcc_lo                                  // 000000005e74: 8b006a15
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e78: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005e7c: be812000
	s_cbranch_execz 28                                         // 000000005e80: bfa5001c <packed_folded_w4a8+0x43f4>
	v_bfe_u32 v8, v12, 16, 1                                   // 000000005e84: d6100008 0205210c
	s_wait_kmcnt 0x0                                           // 000000005e8c: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005e90: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005e98: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005e9c: d5207c0a 00025223
	v_add3_u32 v11, v8, v12, 0x7fff                            // 000000005ea4: d655000b 03fe1908 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005eb0: bf870003
	v_add_co_u32 v8, s0, v9, v64                               // 000000005eb4: d7000008 02028109
	v_or_b32_e32 v15, 0x400000, v12                            // 000000005ebc: 381e18ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ec4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v65, s0                 // 000000005ec8: d5207c09 0002830a
	v_cmp_u_f32_e64 s0, v12, v12                               // 000000005ed0: d4180000 0202190c
	s_wait_alu depctr_va_sdst(0)                               // 000000005ed8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005edc: bf870001
	v_cndmask_b32_e64 v10, v11, v15, s0                        // 000000005ee0: d501000a 00021f0b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005ee8: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ef4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005ef8: 8c7e017e
	s_and_b32 s0, s22, vcc_lo                                  // 000000005efc: 8b006a16
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f00: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005f04: be812000
	s_cbranch_execz 28                                         // 000000005f08: bfa5001c <packed_folded_w4a8+0x447c>
	v_bfe_u32 v8, v13, 16, 1                                   // 000000005f0c: d6100008 0205210d
	s_wait_kmcnt 0x0                                           // 000000005f14: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005f18: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005f20: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005f24: d5207c0a 00025223
	v_add3_u32 v11, v8, v13, 0x7fff                            // 000000005f2c: d655000b 03fe1b08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005f38: bf870003
	v_add_co_u32 v8, s0, v9, v66                               // 000000005f3c: d7000008 02028509
	v_or_b32_e32 v12, 0x400000, v13                            // 000000005f44: 38181aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005f4c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v67, s0                 // 000000005f50: d5207c09 0002870a
	v_cmp_u_f32_e64 s0, v13, v13                               // 000000005f58: d4180000 02021b0d
	s_wait_alu depctr_va_sdst(0)                               // 000000005f60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005f64: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000005f68: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005f70: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005f80: 8c7e017e
	s_and_b32 s0, s23, vcc_lo                                  // 000000005f84: 8b006a17
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f88: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005f8c: be812000
	s_cbranch_execz 28                                         // 000000005f90: bfa5001c <packed_folded_w4a8+0x4504>
	v_bfe_u32 v8, v14, 16, 1                                   // 000000005f94: d6100008 0205210e
	s_wait_kmcnt 0x0                                           // 000000005f9c: bfc70000
	v_add_co_u32 v9, s0, s34, v40                              // 000000005fa0: d7000009 02025022
	s_wait_alu depctr_va_sdst(0)                               // 000000005fa8: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s35, v41, s0                // 000000005fac: d5207c0a 00025223
	v_add3_u32 v11, v8, v14, 0x7fff                            // 000000005fb4: d655000b 03fe1d08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005fc0: bf870003
	v_add_co_u32 v8, s0, v9, v68                               // 000000005fc4: d7000008 02028909
	v_or_b32_e32 v12, 0x400000, v14                            // 000000005fcc: 38181cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005fd4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v69, s0                 // 000000005fd8: d5207c09 00028b0a
	v_cmp_u_f32_e64 s0, v14, v14                               // 000000005fe0: d4180000 02021d0e
	s_wait_alu depctr_va_sdst(0)                               // 000000005fe8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005fec: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000005ff0: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off offset:32         // 000000005ff8: ee09407c 05000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000006004: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006008: 8c7e017e
	global_load_u8 v8, v[38:39], off                           // 00000000600c: ee04007c 00000008 00000026
	global_load_b32 v10, v[46:47], off                         // 000000006018: ee05007c 0000000a 0000002e
	s_wait_loadcnt 0x1                                         // 000000006024: bfc00001
	v_lshlrev_b32_e32 v9, 23, v8                               // 000000006028: 30121097
	s_wait_loadcnt 0x0                                         // 00000000602c: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006030: bf870091
	v_mul_f32_e32 v8, v10, v9                                  // 000000006034: 1010130a
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000006038: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v0, v8                                   // 000000006044: 10101100
	s_xor_b32 s0, s0, -1                                       // 000000006048: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000604c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006050: be812000
	s_cbranch_execnz 1398                                      // 000000006054: bfa60576 <packed_folded_w4a8+0x5b30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006058: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000605c: 8c7e017e
	global_load_b32 v10, v[110:111], off                       // 000000006060: ee05007c 0000000a 0000006e
	s_wait_loadcnt 0x0                                         // 00000000606c: bfc00000
	v_mul_f32_e32 v0, v10, v9                                  // 000000006070: 1000130a
	s_delay_alu instid0(valu_dep_1)                            // 000000006074: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 000000006078: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v1, v0                                   // 000000006084: 10000101
	s_xor_b32 s0, s0, -1                                       // 000000006088: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000608c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006090: be812000
	s_cbranch_execnz 1400                                      // 000000006094: bfa60578 <packed_folded_w4a8+0x5b78>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006098: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000609c: 8c7e017e
	global_load_b32 v10, v[112:113], off                       // 0000000060a0: ee05007c 0000000a 00000070
	s_wait_loadcnt 0x0                                         // 0000000060ac: bfc00000
	v_mul_f32_e32 v1, v10, v9                                  // 0000000060b0: 1002130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000060b4: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 0000000060b8: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v2, v1                                   // 0000000060c4: 10020302
	s_xor_b32 s0, s0, -1                                       // 0000000060c8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060cc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000060d0: be812000
	s_cbranch_execnz 1402                                      // 0000000060d4: bfa6057a <packed_folded_w4a8+0x5bc0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000060dc: 8c7e017e
	global_load_b32 v10, v[114:115], off                       // 0000000060e0: ee05007c 0000000a 00000072
	s_wait_loadcnt 0x0                                         // 0000000060ec: bfc00000
	v_mul_f32_e32 v2, v10, v9                                  // 0000000060f0: 1004130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000060f4: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 0000000060f8: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 000000006104: 10040503
	s_xor_b32 s0, s0, -1                                       // 000000006108: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000610c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006110: be812000
	s_cbranch_execnz 1404                                      // 000000006114: bfa6057c <packed_folded_w4a8+0x5c08>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006118: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000611c: 8c7e017e
	global_load_b32 v10, v[34:35], off                         // 000000006120: ee05007c 0000000a 00000022
	s_wait_loadcnt 0x0                                         // 00000000612c: bfc00000
	v_mul_f32_e32 v3, v10, v9                                  // 000000006130: 1006130a
	s_delay_alu instid0(valu_dep_1)                            // 000000006134: bf870001
	v_cmp_class_f32_e64 s0, v3, 0x198                          // 000000006138: d47e0000 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 000000006144: 10060704
	s_xor_b32 s0, s0, -1                                       // 000000006148: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000614c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006150: be812000
	s_cbranch_execnz 1406                                      // 000000006154: bfa6057e <packed_folded_w4a8+0x5c50>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006158: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000615c: 8c7e017e
	global_load_b32 v10, v[116:117], off                       // 000000006160: ee05007c 0000000a 00000074
	s_wait_loadcnt 0x0                                         // 00000000616c: bfc00000
	v_mul_f32_e32 v4, v10, v9                                  // 000000006170: 1008130a
	s_delay_alu instid0(valu_dep_1)                            // 000000006174: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 000000006178: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 000000006184: 10080905
	s_xor_b32 s0, s0, -1                                       // 000000006188: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000618c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006190: be812000
	s_cbranch_execnz 1408                                      // 000000006194: bfa60580 <packed_folded_w4a8+0x5c98>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006198: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000619c: 8c7e017e
	global_load_b32 v10, v[36:37], off                         // 0000000061a0: ee05007c 0000000a 00000024
	s_wait_loadcnt 0x0                                         // 0000000061ac: bfc00000
	v_mul_f32_e32 v5, v10, v9                                  // 0000000061b0: 100a130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000061b4: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 0000000061b8: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 0000000061c4: 100a0b06
	s_xor_b32 s0, s0, -1                                       // 0000000061c8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061cc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000061d0: be812000
	s_cbranch_execnz 1410                                      // 0000000061d4: bfa60582 <packed_folded_w4a8+0x5ce0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000061dc: 8c7e017e
	global_load_b32 v10, v[120:121], off                       // 0000000061e0: ee05007c 0000000a 00000078
	s_wait_loadcnt 0x0                                         // 0000000061ec: bfc00000
	v_mul_f32_e32 v6, v10, v9                                  // 0000000061f0: 100c130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000061f4: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 0000000061f8: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v7, v6                                   // 000000006204: 100c0d07
	s_xor_b32 s0, s0, -1                                       // 000000006208: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000620c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006210: be812000
	s_cbranch_execnz 1412                                      // 000000006214: bfa60584 <packed_folded_w4a8+0x5d28>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006218: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000621c: 8c7e017e
	s_and_b32 s0, s27, vcc_lo                                  // 000000006220: 8b006a1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006224: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006228: be812000
	s_cbranch_execz 34                                         // 00000000622c: bfa50022 <packed_folded_w4a8+0x47b8>
	v_add_co_u32 v9, s0, v125, v124                            // 000000006230: d7000009 0202f97d
	s_wait_alu depctr_va_sdst(0)                               // 000000006238: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v126, s0                 // 00000000623c: d5207c0a 0002fc80
	s_wait_kmcnt 0x0                                           // 000000006244: bfc70000
	v_add_co_u32 v11, s0, s34, v32                             // 000000006248: d700000b 02024022
	v_bfe_u32 v7, v8, 16, 1                                    // 000000006250: d6100007 02052108
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000006258: bf870253
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 00000000625c: 3e121281
	s_wait_alu depctr_va_sdst(0)                               // 000000006260: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s35, v33, s0                // 000000006264: d5207c0c 00024223
	v_or_b32_e32 v13, 0x400000, v8                             // 00000000626c: 381a10ff 00400000
	v_add3_u32 v7, v7, v8, 0x7fff                              // 000000006274: d6550007 03fe1107 00007fff
	v_add_co_u32 v9, s0, v11, v9                               // 000000006280: d7000009 0202130b
	s_wait_alu depctr_va_sdst(0)                               // 000000006288: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v12, v10, s0                // 00000000628c: d5207c0a 0002150c
	v_cmp_u_f32_e64 s0, v8, v8                                 // 000000006294: d4180000 02021108
	s_wait_alu depctr_va_sdst(0)                               // 00000000629c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000062a0: bf870001
	v_cndmask_b32_e64 v7, v7, v13, s0                          // 0000000062a4: d5010007 00021b07
	global_store_d16_hi_b16 v[9:10], v7, off offset:32         // 0000000062ac: ee09407c 03800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000062bc: 8c7e017e
	s_and_b32 s0, s24, vcc_lo                                  // 0000000062c0: 8b006a18
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062c4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000062c8: be812000
	s_cbranch_execz 28                                         // 0000000062cc: bfa5001c <packed_folded_w4a8+0x4840>
	v_bfe_u32 v7, v0, 16, 1                                    // 0000000062d0: d6100007 02052100
	s_wait_kmcnt 0x0                                           // 0000000062d8: bfc70000
	v_add_co_u32 v8, s0, s34, v32                              // 0000000062dc: d7000008 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000062e4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s35, v33, s0                 // 0000000062e8: d5207c09 00024223
	v_add3_u32 v10, v7, v0, 0x7fff                             // 0000000062f0: d655000a 03fe0107 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000062fc: bf870003
	v_add_co_u32 v7, s0, v8, v56                               // 000000006300: d7000007 02027108
	v_or_b32_e32 v11, 0x400000, v0                             // 000000006308: 381600ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006310: bf88f19f
	v_add_co_ci_u32_e64 v8, null, v9, v57, s0                  // 000000006314: d5207c08 00027309
	v_cmp_u_f32_e64 s0, v0, v0                                 // 00000000631c: d4180000 02020100
	s_wait_alu depctr_va_sdst(0)                               // 000000006324: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006328: bf870001
	v_cndmask_b32_e64 v0, v10, v11, s0                         // 00000000632c: d5010000 0002170a
	global_store_d16_hi_b16 v[7:8], v0, off offset:32          // 000000006334: ee09407c 00000000 00002007
	s_wait_alu depctr_sa_sdst(0)                               // 000000006340: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006344: 8c7e017e
	s_and_b32 s0, s25, vcc_lo                                  // 000000006348: 8b006a19
	s_wait_alu depctr_sa_sdst(0)                               // 00000000634c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006350: be812000
	s_cbranch_execz 28                                         // 000000006354: bfa5001c <packed_folded_w4a8+0x48c8>
	s_wait_kmcnt 0x0                                           // 000000006358: bfc70000
	v_add_co_u32 v7, s0, s34, v32                              // 00000000635c: d7000007 02024022
	v_bfe_u32 v0, v1, 16, 1                                    // 000000006364: d6100000 02052101
	s_wait_alu depctr_va_sdst(0)                               // 00000000636c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s35, v33, s0                 // 000000006370: d5207c08 00024223
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006378: bf870193
	v_add_co_u32 v7, s0, v7, v58                               // 00000000637c: d7000007 02027507
	v_add3_u32 v0, v0, v1, 0x7fff                              // 000000006384: d6550000 03fe0300 00007fff
	v_or_b32_e32 v9, 0x400000, v1                              // 000000006390: 381202ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006398: bf88f19f
	v_add_co_ci_u32_e64 v8, null, v8, v59, s0                  // 00000000639c: d5207c08 00027708
	v_cmp_u_f32_e64 s0, v1, v1                                 // 0000000063a4: d4180000 02020301
	s_wait_alu depctr_va_sdst(0)                               // 0000000063ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000063b0: bf870001
	v_cndmask_b32_e64 v0, v0, v9, s0                           // 0000000063b4: d5010000 00021300
	global_store_d16_hi_b16 v[7:8], v0, off offset:32          // 0000000063bc: ee09407c 00000000 00002007
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000063cc: 8c7e017e
	s_and_b32 s0, s26, vcc_lo                                  // 0000000063d0: 8b006a1a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063d4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000063d8: be812000
	s_cbranch_execz 28                                         // 0000000063dc: bfa5001c <packed_folded_w4a8+0x4950>
	v_bfe_u32 v0, v2, 16, 1                                    // 0000000063e0: d6100000 02052102
	s_wait_kmcnt 0x0                                           // 0000000063e8: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 0000000063ec: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000063f4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v33, s0                 // 0000000063f8: d5207c07 00024223
	v_add3_u32 v8, v0, v2, 0x7fff                              // 000000006400: d6550008 03fe0500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000640c: bf870003
	v_add_co_u32 v0, s0, v1, v60                               // 000000006410: d7000000 02027901
	v_or_b32_e32 v9, 0x400000, v2                              // 000000006418: 381204ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006420: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v7, v61, s0                  // 000000006424: d5207c01 00027b07
	v_cmp_u_f32_e64 s0, v2, v2                                 // 00000000642c: d4180000 02020502
	s_wait_alu depctr_va_sdst(0)                               // 000000006434: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006438: bf870001
	v_cndmask_b32_e64 v2, v8, v9, s0                           // 00000000643c: d5010002 00021308
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006444: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006450: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006454: 8c7e017e
	s_and_b32 s0, s28, vcc_lo                                  // 000000006458: 8b006a1c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000645c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006460: be812000
	s_cbranch_execz 28                                         // 000000006464: bfa5001c <packed_folded_w4a8+0x49d8>
	v_bfe_u32 v0, v3, 16, 1                                    // 000000006468: d6100000 02052103
	s_wait_kmcnt 0x0                                           // 000000006470: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 000000006474: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 00000000647c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 000000006480: d5207c02 00024223
	v_add3_u32 v7, v0, v3, 0x7fff                              // 000000006488: d6550007 03fe0700 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006494: bf870003
	v_add_co_u32 v0, s0, v1, v62                               // 000000006498: d7000000 02027d01
	v_or_b32_e32 v8, 0x400000, v3                              // 0000000064a0: 381006ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000064a8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v63, s0                  // 0000000064ac: d5207c01 00027f02
	v_cmp_u_f32_e64 s0, v3, v3                                 // 0000000064b4: d4180000 02020703
	s_wait_alu depctr_va_sdst(0)                               // 0000000064bc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000064c0: bf870001
	v_cndmask_b32_e64 v2, v7, v8, s0                           // 0000000064c4: d5010002 00021107
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000064cc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000064dc: 8c7e017e
	s_and_b32 s0, s29, vcc_lo                                  // 0000000064e0: 8b006a1d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064e4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000064e8: be812000
	s_cbranch_execz 28                                         // 0000000064ec: bfa5001c <packed_folded_w4a8+0x4a60>
	v_bfe_u32 v0, v4, 16, 1                                    // 0000000064f0: d6100000 02052104
	s_wait_kmcnt 0x0                                           // 0000000064f8: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 0000000064fc: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 000000006504: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 000000006508: d5207c02 00024223
	v_add3_u32 v3, v0, v4, 0x7fff                              // 000000006510: d6550003 03fe0900 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000651c: bf870003
	v_add_co_u32 v0, s0, v1, v64                               // 000000006520: d7000000 02028101
	v_or_b32_e32 v7, 0x400000, v4                              // 000000006528: 380e08ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006530: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v65, s0                  // 000000006534: d5207c01 00028302
	v_cmp_u_f32_e64 s0, v4, v4                                 // 00000000653c: d4180000 02020904
	s_wait_alu depctr_va_sdst(0)                               // 000000006544: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006548: bf870001
	v_cndmask_b32_e64 v2, v3, v7, s0                           // 00000000654c: d5010002 00020f03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006554: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006560: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006564: 8c7e017e
	s_and_b32 s0, s30, vcc_lo                                  // 000000006568: 8b006a1e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000656c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006570: be812000
	s_cbranch_execz 28                                         // 000000006574: bfa5001c <packed_folded_w4a8+0x4ae8>
	v_bfe_u32 v0, v5, 16, 1                                    // 000000006578: d6100000 02052105
	s_wait_kmcnt 0x0                                           // 000000006580: bfc70000
	v_add_co_u32 v1, s0, s34, v32                              // 000000006584: d7000001 02024022
	s_wait_alu depctr_va_sdst(0)                               // 00000000658c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s35, v33, s0                 // 000000006590: d5207c02 00024223
	v_add3_u32 v3, v0, v5, 0x7fff                              // 000000006598: d6550003 03fe0b00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000065a4: bf870003
	v_add_co_u32 v0, s0, v1, v66                               // 0000000065a8: d7000000 02028501
	v_or_b32_e32 v4, 0x400000, v5                              // 0000000065b0: 38080aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000065b8: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v67, s0                  // 0000000065bc: d5207c01 00028702
	v_cmp_u_f32_e64 s0, v5, v5                                 // 0000000065c4: d4180000 02020b05
	s_wait_alu depctr_va_sdst(0)                               // 0000000065cc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000065d0: bf870001
	v_cndmask_b32_e64 v2, v3, v4, s0                           // 0000000065d4: d5010002 00020903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000065dc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000065ec: 8c7e017e
	s_and_b32 s0, s31, vcc_lo                                  // 0000000065f0: 8b006a1f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000065f8: be812000
	s_cbranch_execz 25                                         // 0000000065fc: bfa50019 <packed_folded_w4a8+0x4b64>
	v_bfe_u32 v0, v6, 16, 1                                    // 000000006600: d6100000 02052106
	s_wait_kmcnt 0x0                                           // 000000006608: bfc70000
	v_add_co_u32 v1, vcc_lo, s34, v32                          // 00000000660c: d7006a01 02024022
	s_wait_alu depctr_va_vcc(0)                                // 000000006614: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s35, v33, vcc_lo             // 000000006618: d5207c02 01aa4223
	v_add3_u32 v3, v0, v6, 0x7fff                              // 000000006620: d6550003 03fe0d00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000662c: bf870003
	v_add_co_u32 v0, vcc_lo, v1, v68                           // 000000006630: d7006a00 02028901
	v_or_b32_e32 v4, 0x400000, v6                              // 000000006638: 38080cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006640: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v69, vcc_lo              // 000000006644: d5207c01 01aa8b02
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 00000000664c: 7c300d06
	s_wait_alu depctr_va_vcc(0)                                // 000000006650: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000006654: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006658: ee09407c 01000000 00002000
	s_nop 0                                                    // 000000006664: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000006668: bfb60003
	s_endpgm                                                   // 00000000666c: bfb00000
	v_cvt_f64_f32_e32 v[74:75], v56                            // 000000006670: 7e942138
	v_cvt_f64_f32_e32 v[76:77], v70                            // 000000006674: 7e982146
	v_cvt_f64_f32_e32 v[78:79], v71                            // 000000006678: 7e9c2147
	v_cmp_eq_f32_e64 s2, 0, v56                                // 00000000667c: d4120002 02027080
	v_cmp_class_f32_e64 s4, v71, 0x1f8                         // 000000006684: d47e0004 0201ff47 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006690: 8b020402
	v_mul_f64_e32 v[74:75], v[74:75], v[76:77]                 // 000000006694: 0c94994a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006698: bf870091
	v_mul_f64_e32 v[74:75], v[74:75], v[78:79]                 // 00000000669c: 0c949d4a
	v_cvt_f32_f64_e32 v74, v[74:75]                            // 0000000066a0: 7e941f4a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066a8: bf870001
	v_cndmask_b32_e64 v88, v74, 0, s2                          // 0000000066ac: d5010058 0009014a
	s_branch 61551                                             // 0000000066b4: bfa0f06f <packed_folded_w4a8+0xd74>
	v_cvt_f64_f32_e32 v[76:77], v57                            // 0000000066b8: 7e982139
	v_cvt_f64_f32_e32 v[78:79], v70                            // 0000000066bc: 7e9c2146
	v_cvt_f64_f32_e32 v[80:81], v56                            // 0000000066c0: 7ea02138
	v_cmp_eq_f32_e64 s2, 0, v57                                // 0000000066c4: d4120002 02027280
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000066cc: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000066d8: 8b020402
	v_mul_f64_e32 v[76:77], v[76:77], v[78:79]                 // 0000000066dc: 0c989d4c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066e0: bf870091
	v_mul_f64_e32 v[76:77], v[76:77], v[80:81]                 // 0000000066e4: 0c98a14c
	v_cvt_f32_f64_e32 v71, v[76:77]                            // 0000000066e8: 7e8e1f4c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066ec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066f0: bf870001
	v_cndmask_b32_e64 v89, v71, 0, s2                          // 0000000066f4: d5010059 00090147
	s_branch 61568                                             // 0000000066fc: bfa0f080 <packed_folded_w4a8+0xe00>
	v_cvt_f64_f32_e32 v[78:79], v58                            // 000000006700: 7e9c213a
	v_cvt_f64_f32_e32 v[80:81], v70                            // 000000006704: 7ea02146
	v_cvt_f64_f32_e32 v[82:83], v56                            // 000000006708: 7ea42138
	v_cmp_eq_f32_e64 s2, 0, v58                                // 00000000670c: d4120002 02027480
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 000000006714: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006720: 8b020402
	v_mul_f64_e32 v[78:79], v[78:79], v[80:81]                 // 000000006724: 0c9ca14e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006728: bf870091
	v_mul_f64_e32 v[78:79], v[78:79], v[82:83]                 // 00000000672c: 0c9ca54e
	v_cvt_f32_f64_e32 v57, v[78:79]                            // 000000006730: 7e721f4e
	s_wait_alu depctr_sa_sdst(0)                               // 000000006734: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006738: bf870001
	v_cndmask_b32_e64 v91, v57, 0, s2                          // 00000000673c: d501005b 00090139
	s_branch 61585                                             // 000000006744: bfa0f091 <packed_folded_w4a8+0xe8c>
	v_cvt_f64_f32_e32 v[57:58], v59                            // 000000006748: 7e72213b
	v_cvt_f64_f32_e32 v[80:81], v70                            // 00000000674c: 7ea02146
	v_cvt_f64_f32_e32 v[82:83], v56                            // 000000006750: 7ea42138
	v_cmp_eq_f32_e64 s2, 0, v59                                // 000000006754: d4120002 02027680
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 00000000675c: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006768: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[80:81]                 // 00000000676c: 0c72a139
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006770: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[82:83]                 // 000000006774: 0c72a539
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006778: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000677c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006780: bf870001
	v_cndmask_b32_e64 v92, v57, 0, s2                          // 000000006784: d501005c 00090139
	s_branch 61602                                             // 00000000678c: bfa0f0a2 <packed_folded_w4a8+0xf18>
	v_cvt_f64_f32_e32 v[57:58], v60                            // 000000006790: 7e72213c
	v_cvt_f64_f32_e32 v[82:83], v70                            // 000000006794: 7ea42146
	v_cvt_f64_f32_e32 v[84:85], v56                            // 000000006798: 7ea82138
	v_cmp_eq_f32_e64 s2, 0, v60                                // 00000000679c: d4120002 02027880
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000067a4: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000067b0: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[82:83]                 // 0000000067b4: 0c72a539
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000067b8: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[84:85]                 // 0000000067bc: 0c72a939
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000067c0: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000067c8: bf870001
	v_cndmask_b32_e64 v93, v57, 0, s2                          // 0000000067cc: d501005d 00090139
	s_branch 61619                                             // 0000000067d4: bfa0f0b3 <packed_folded_w4a8+0xfa4>
	v_cvt_f64_f32_e32 v[57:58], v61                            // 0000000067d8: 7e72213d
	v_cvt_f64_f32_e32 v[59:60], v70                            // 0000000067dc: 7e762146
	v_cvt_f64_f32_e32 v[84:85], v56                            // 0000000067e0: 7ea82138
	v_cmp_eq_f32_e64 s2, 0, v61                                // 0000000067e4: d4120002 02027a80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 0000000067ec: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000067f8: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 0000000067fc: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006800: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[84:85]                 // 000000006804: 0c72a939
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006808: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000680c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006810: bf870001
	v_cndmask_b32_e64 v94, v57, 0, s2                          // 000000006814: d501005e 00090139
	s_branch 61636                                             // 00000000681c: bfa0f0c4 <packed_folded_w4a8+0x1030>
	v_cvt_f64_f32_e32 v[57:58], v62                            // 000000006820: 7e72213e
	v_cvt_f64_f32_e32 v[59:60], v70                            // 000000006824: 7e762146
	v_cvt_f64_f32_e32 v[86:87], v56                            // 000000006828: 7eac2138
	v_cmp_eq_f32_e64 s2, 0, v62                                // 00000000682c: d4120002 02027c80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 000000006834: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006840: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 000000006844: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006848: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[86:87]                 // 00000000684c: 0c72ad39
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006850: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000006854: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006858: bf870001
	v_cndmask_b32_e64 v95, v57, 0, s2                          // 00000000685c: d501005f 00090139
	s_branch 61653                                             // 000000006864: bfa0f0d5 <packed_folded_w4a8+0x10bc>
	v_cvt_f64_f32_e32 v[57:58], v63                            // 000000006868: 7e72213f
	v_cvt_f64_f32_e32 v[59:60], v70                            // 00000000686c: 7e762146
	v_cvt_f64_f32_e32 v[61:62], v56                            // 000000006870: 7e7a2138
	v_cmp_eq_f32_e64 s2, 0, v63                                // 000000006874: d4120002 02027e80
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 00000000687c: d47e0004 0201ff38 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006888: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 00000000688c: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006890: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[61:62]                 // 000000006894: 0c727b39
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000006898: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000689c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000068a0: bf870001
	v_cndmask_b32_e64 v96, v57, 0, s2                          // 0000000068a4: d5010060 00090139
	s_branch 61669                                             // 0000000068ac: bfa0f0e5 <packed_folded_w4a8+0x1144>
	v_cvt_f64_f32_e32 v[91:92], v48                            // 0000000068b0: 7eb62130
	v_cvt_f64_f32_e32 v[93:94], v106                           // 0000000068b4: 7eba216a
	v_cvt_f64_f32_e32 v[95:96], v90                            // 0000000068b8: 7ebe215a
	v_cmp_eq_f32_e64 s8, 0, v48                                // 0000000068bc: d4120008 02026080
	v_cmp_class_f32_e64 s10, v90, 0x1f8                        // 0000000068c4: d47e000a 0201ff5a 000001f8
	s_and_b32 s8, s8, s10                                      // 0000000068d0: 8b080a08
	v_mul_f64_e32 v[91:92], v[91:92], v[93:94]                 // 0000000068d4: 0cb6bb5b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000068d8: bf870091
	v_mul_f64_e32 v[91:92], v[91:92], v[95:96]                 // 0000000068dc: 0cb6bf5b
	v_cvt_f32_f64_e32 v91, v[91:92]                            // 0000000068e0: 7eb61f5b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000068e8: bf870001
	v_cndmask_b32_e64 v103, v91, 0, s8                         // 0000000068ec: d5010067 0021015b
	s_branch 62047                                             // 0000000068f4: bfa0f25f <packed_folded_w4a8+0x1774>
	v_cvt_f64_f32_e32 v[92:93], v49                            // 0000000068f8: 7eb82131
	v_cvt_f64_f32_e32 v[94:95], v106                           // 0000000068fc: 7ebc216a
	v_cvt_f64_f32_e32 v[96:97], v48                            // 000000006900: 7ec02130
	v_cmp_eq_f32_e64 s8, 0, v49                                // 000000006904: d4120008 02026280
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 00000000690c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006918: 8b080a08
	v_mul_f64_e32 v[92:93], v[92:93], v[94:95]                 // 00000000691c: 0cb8bd5c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006920: bf870091
	v_mul_f64_e32 v[92:93], v[92:93], v[96:97]                 // 000000006924: 0cb8c15c
	v_cvt_f32_f64_e32 v92, v[92:93]                            // 000000006928: 7eb81f5c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000692c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006930: bf870001
	v_cndmask_b32_e64 v104, v92, 0, s8                         // 000000006934: d5010068 0021015c
	s_branch 62062                                             // 00000000693c: bfa0f26e <packed_folded_w4a8+0x17f8>
	v_cvt_f64_f32_e32 v[94:95], v50                            // 000000006940: 7ebc2132
	v_cvt_f64_f32_e32 v[96:97], v106                           // 000000006944: 7ec0216a
	v_cvt_f64_f32_e32 v[107:108], v48                          // 000000006948: 7ed62130
	v_cmp_eq_f32_e64 s8, 0, v50                                // 00000000694c: d4120008 02026480
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006954: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006960: 8b080a08
	v_mul_f64_e32 v[94:95], v[94:95], v[96:97]                 // 000000006964: 0cbcc15e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006968: bf870091
	v_mul_f64_e32 v[94:95], v[94:95], v[107:108]               // 00000000696c: 0cbcd75e
	v_cvt_f32_f64_e32 v49, v[94:95]                            // 000000006970: 7e621f5e
	s_wait_alu depctr_sa_sdst(0)                               // 000000006974: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006978: bf870001
	v_cndmask_b32_e64 v105, v49, 0, s8                         // 00000000697c: d5010069 00210131
	s_branch 62077                                             // 000000006984: bfa0f27d <packed_folded_w4a8+0x187c>
	v_cvt_f64_f32_e32 v[49:50], v51                            // 000000006988: 7e622133
	v_cvt_f64_f32_e32 v[96:97], v106                           // 00000000698c: 7ec0216a
	v_cvt_f64_f32_e32 v[107:108], v48                          // 000000006990: 7ed62130
	v_cmp_eq_f32_e64 s8, 0, v51                                // 000000006994: d4120008 02026680
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 00000000699c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 0000000069a8: 8b080a08
	v_mul_f64_e32 v[49:50], v[49:50], v[96:97]                 // 0000000069ac: 0c62c131
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069b0: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[107:108]               // 0000000069b4: 0c62d731
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 0000000069b8: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069c0: bf870001
	v_cndmask_b32_e64 v107, v49, 0, s8                         // 0000000069c4: d501006b 00210131
	s_branch 62092                                             // 0000000069cc: bfa0f28c <packed_folded_w4a8+0x1900>
	v_cvt_f64_f32_e32 v[96:97], v52                            // 0000000069d0: 7ec02134
	v_cvt_f64_f32_e32 v[108:109], v106                         // 0000000069d4: 7ed8216a
	v_cvt_f64_f32_e32 v[112:113], v48                          // 0000000069d8: 7ee02130
	v_cmp_eq_f32_e64 s8, 0, v52                                // 0000000069dc: d4120008 02026880
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 0000000069e4: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 0000000069f0: 8b080a08
	v_mul_f64_e32 v[96:97], v[96:97], v[108:109]               // 0000000069f4: 0cc0d960
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069f8: bf870091
	v_mul_f64_e32 v[96:97], v[96:97], v[112:113]               // 0000000069fc: 0cc0e160
	v_cvt_f32_f64_e32 v49, v[96:97]                            // 000000006a00: 7e621f60
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a04: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a08: bf870001
	v_cndmask_b32_e64 v108, v49, 0, s8                         // 000000006a0c: d501006c 00210131
	s_branch 62107                                             // 000000006a14: bfa0f29b <packed_folded_w4a8+0x1984>
	v_cvt_f64_f32_e32 v[112:113], v53                          // 000000006a18: 7ee02135
	v_cvt_f64_f32_e32 v[120:121], v106                         // 000000006a1c: 7ef0216a
	v_cvt_f64_f32_e32 v[122:123], v48                          // 000000006a20: 7ef42130
	v_cmp_eq_f32_e64 s8, 0, v53                                // 000000006a24: d4120008 02026a80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006a2c: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006a38: 8b080a08
	v_mul_f64_e32 v[112:113], v[112:113], v[120:121]           // 000000006a3c: 0ce0f170
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a40: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 000000006a44: 0ce0f570
	v_cvt_f32_f64_e32 v49, v[112:113]                          // 000000006a48: 7e621f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a4c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a50: bf870001
	v_cndmask_b32_e64 v109, v49, 0, s8                         // 000000006a54: d501006d 00210131
	s_branch 62122                                             // 000000006a5c: bfa0f2aa <packed_folded_w4a8+0x1a08>
	v_cvt_f64_f32_e32 v[112:113], v54                          // 000000006a60: 7ee02136
	v_cvt_f64_f32_e32 v[120:121], v106                         // 000000006a64: 7ef0216a
	v_cvt_f64_f32_e32 v[122:123], v48                          // 000000006a68: 7ef42130
	v_cmp_eq_f32_e64 s8, 0, v54                                // 000000006a6c: d4120008 02026c80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006a74: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006a80: 8b080a08
	v_mul_f64_e32 v[112:113], v[112:113], v[120:121]           // 000000006a84: 0ce0f170
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a88: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 000000006a8c: 0ce0f570
	v_cvt_f32_f64_e32 v49, v[112:113]                          // 000000006a90: 7e621f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a94: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a98: bf870001
	v_cndmask_b32_e64 v112, v49, 0, s8                         // 000000006a9c: d5010070 00210131
	s_branch 62137                                             // 000000006aa4: bfa0f2b9 <packed_folded_w4a8+0x1a8c>
	v_cvt_f64_f32_e32 v[120:121], v55                          // 000000006aa8: 7ef02137
	v_cvt_f64_f32_e32 v[122:123], v106                         // 000000006aac: 7ef4216a
	v_cvt_f64_f32_e32 v[133:134], v48                          // 000000006ab0: 7f0a2130
	v_cmp_eq_f32_e64 s8, 0, v55                                // 000000006ab4: d4120008 02026e80
	v_cmp_class_f32_e64 s10, v48, 0x1f8                        // 000000006abc: d47e000a 0201ff30 000001f8
	s_and_b32 s8, s8, s10                                      // 000000006ac8: 8b080a08
	v_mul_f64_e32 v[120:121], v[120:121], v[122:123]           // 000000006acc: 0cf0f578
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ad0: bf870091
	v_mul_f64_e32 v[120:121], v[120:121], v[133:134]           // 000000006ad4: 0cf10b78
	v_cvt_f32_f64_e32 v49, v[120:121]                          // 000000006ad8: 7e621f78
	s_wait_alu depctr_sa_sdst(0)                               // 000000006adc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ae0: bf870001
	v_cndmask_b32_e64 v113, v49, 0, s8                         // 000000006ae4: d5010071 00210131
	s_branch 62151                                             // 000000006aec: bfa0f2c7 <packed_folded_w4a8+0x1b0c>
	v_cvt_f64_f32_e32 v[101:102], v40                          // 000000006af0: 7eca2128
	v_cvt_f64_f32_e32 v[103:104], v123                         // 000000006af4: 7ece217b
	v_cvt_f64_f32_e32 v[105:106], v100                         // 000000006af8: 7ed22164
	v_cmp_eq_f32_e64 s16, 0, v40                               // 000000006afc: d4120010 02025080
	v_cmp_class_f32_e64 s18, v100, 0x1f8                       // 000000006b04: d47e0012 0201ff64 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006b10: 8b101210
	v_mul_f64_e32 v[101:102], v[101:102], v[103:104]           // 000000006b14: 0ccacf65
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b18: bf870091
	v_mul_f64_e32 v[101:102], v[101:102], v[105:106]           // 000000006b1c: 0ccad365
	v_cvt_f32_f64_e32 v101, v[101:102]                         // 000000006b20: 7eca1f65
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b24: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b28: bf870001
	v_cndmask_b32_e64 v120, v101, 0, s16                       // 000000006b2c: d5010078 00410165
	s_branch 62478                                             // 000000006b34: bfa0f40e <packed_folded_w4a8+0x2070>
	v_cvt_f64_f32_e32 v[102:103], v41                          // 000000006b38: 7ecc2129
	v_cvt_f64_f32_e32 v[104:105], v123                         // 000000006b3c: 7ed0217b
	v_cvt_f64_f32_e32 v[106:107], v40                          // 000000006b40: 7ed42128
	v_cmp_eq_f32_e64 s16, 0, v41                               // 000000006b44: d4120010 02025280
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006b4c: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006b58: 8b101210
	v_mul_f64_e32 v[102:103], v[102:103], v[104:105]           // 000000006b5c: 0cccd166
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b60: bf870091
	v_mul_f64_e32 v[102:103], v[102:103], v[106:107]           // 000000006b64: 0cccd566
	v_cvt_f32_f64_e32 v102, v[102:103]                         // 000000006b68: 7ecc1f66
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b6c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b70: bf870001
	v_cndmask_b32_e64 v121, v102, 0, s16                       // 000000006b74: d5010079 00410166
	s_branch 62493                                             // 000000006b7c: bfa0f41d <packed_folded_w4a8+0x20f4>
	v_cvt_f64_f32_e32 v[104:105], v42                          // 000000006b80: 7ed0212a
	v_cvt_f64_f32_e32 v[106:107], v123                         // 000000006b84: 7ed4217b
	v_cvt_f64_f32_e32 v[133:134], v40                          // 000000006b88: 7f0a2128
	v_cmp_eq_f32_e64 s16, 0, v42                               // 000000006b8c: d4120010 02025480
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006b94: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006ba0: 8b101210
	v_mul_f64_e32 v[104:105], v[104:105], v[106:107]           // 000000006ba4: 0cd0d568
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ba8: bf870091
	v_mul_f64_e32 v[104:105], v[104:105], v[133:134]           // 000000006bac: 0cd10b68
	v_cvt_f32_f64_e32 v41, v[104:105]                          // 000000006bb0: 7e521f68
	s_wait_alu depctr_sa_sdst(0)                               // 000000006bb4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006bb8: bf870001
	v_cndmask_b32_e64 v122, v41, 0, s16                        // 000000006bbc: d501007a 00410129
	s_branch 62508                                             // 000000006bc4: bfa0f42c <packed_folded_w4a8+0x2178>
	v_cvt_f64_f32_e32 v[41:42], v43                            // 000000006bc8: 7e52212b
	v_cvt_f64_f32_e32 v[106:107], v123                         // 000000006bcc: 7ed4217b
	v_cvt_f64_f32_e32 v[133:134], v40                          // 000000006bd0: 7f0a2128
	v_cmp_eq_f32_e64 s16, 0, v43                               // 000000006bd4: d4120010 02025680
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006bdc: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006be8: 8b101210
	v_mul_f64_e32 v[41:42], v[41:42], v[106:107]               // 000000006bec: 0c52d529
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bf0: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[133:134]               // 000000006bf4: 0c530b29
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000006bf8: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 000000006bfc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c00: bf870001
	v_cndmask_b32_e64 v133, v41, 0, s16                        // 000000006c04: d5010085 00410129
	s_branch 62523                                             // 000000006c0c: bfa0f43b <packed_folded_w4a8+0x21fc>
	v_cvt_f64_f32_e32 v[106:107], v44                          // 000000006c10: 7ed4212c
	v_cvt_f64_f32_e32 v[134:135], v123                         // 000000006c14: 7f0c217b
	v_cvt_f64_f32_e32 v[136:137], v40                          // 000000006c18: 7f102128
	v_cmp_eq_f32_e64 s16, 0, v44                               // 000000006c1c: d4120010 02025880
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006c24: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006c30: 8b101210
	v_mul_f64_e32 v[106:107], v[106:107], v[134:135]           // 000000006c34: 0cd50d6a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c38: bf870091
	v_mul_f64_e32 v[106:107], v[106:107], v[136:137]           // 000000006c3c: 0cd5116a
	v_cvt_f32_f64_e32 v41, v[106:107]                          // 000000006c40: 7e521f6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c44: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c48: bf870001
	v_cndmask_b32_e64 v134, v41, 0, s16                        // 000000006c4c: d5010086 00410129
	s_branch 62538                                             // 000000006c54: bfa0f44a <packed_folded_w4a8+0x2280>
	v_cvt_f64_f32_e32 v[135:136], v45                          // 000000006c58: 7f0e212d
	v_cvt_f64_f32_e32 v[137:138], v123                         // 000000006c5c: 7f12217b
	v_cvt_f64_f32_e32 v[139:140], v40                          // 000000006c60: 7f162128
	v_cmp_eq_f32_e64 s16, 0, v45                               // 000000006c64: d4120010 02025a80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006c6c: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006c78: 8b101210
	v_mul_f64_e32 v[135:136], v[135:136], v[137:138]           // 000000006c7c: 0d0f1387
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c80: bf870091
	v_mul_f64_e32 v[135:136], v[135:136], v[139:140]           // 000000006c84: 0d0f1787
	v_cvt_f32_f64_e32 v41, v[135:136]                          // 000000006c88: 7e521f87
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c8c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c90: bf870001
	v_cndmask_b32_e64 v135, v41, 0, s16                        // 000000006c94: d5010087 00410129
	s_branch 62553                                             // 000000006c9c: bfa0f459 <packed_folded_w4a8+0x2304>
	v_cvt_f64_f32_e32 v[136:137], v46                          // 000000006ca0: 7f10212e
	v_cvt_f64_f32_e32 v[138:139], v123                         // 000000006ca4: 7f14217b
	v_cvt_f64_f32_e32 v[140:141], v40                          // 000000006ca8: 7f182128
	v_cmp_eq_f32_e64 s16, 0, v46                               // 000000006cac: d4120010 02025c80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006cb4: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006cc0: 8b101210
	v_mul_f64_e32 v[136:137], v[136:137], v[138:139]           // 000000006cc4: 0d111588
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006cc8: bf870091
	v_mul_f64_e32 v[136:137], v[136:137], v[140:141]           // 000000006ccc: 0d111988
	v_cvt_f32_f64_e32 v41, v[136:137]                          // 000000006cd0: 7e521f88
	s_wait_alu depctr_sa_sdst(0)                               // 000000006cd4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006cd8: bf870001
	v_cndmask_b32_e64 v136, v41, 0, s16                        // 000000006cdc: d5010088 00410129
	s_branch 62568                                             // 000000006ce4: bfa0f468 <packed_folded_w4a8+0x2388>
	v_cvt_f64_f32_e32 v[137:138], v47                          // 000000006ce8: 7f12212f
	v_cvt_f64_f32_e32 v[139:140], v123                         // 000000006cec: 7f16217b
	v_cvt_f64_f32_e32 v[141:142], v40                          // 000000006cf0: 7f1a2128
	v_cmp_eq_f32_e64 s16, 0, v47                               // 000000006cf4: d4120010 02025e80
	v_cmp_class_f32_e64 s18, v40, 0x1f8                        // 000000006cfc: d47e0012 0201ff28 000001f8
	s_and_b32 s16, s16, s18                                    // 000000006d08: 8b101210
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006d0c: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d10: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006d14: 0d131b89
	v_cvt_f32_f64_e32 v41, v[137:138]                          // 000000006d18: 7e521f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d1c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d20: bf870001
	v_cndmask_b32_e64 v137, v41, 0, s16                        // 000000006d24: d5010089 00410129
	s_branch 62582                                             // 000000006d2c: bfa0f476 <packed_folded_w4a8+0x2408>
	v_cvt_f64_f32_e32 v[111:112], v32                          // 000000006d30: 7ede2120
	v_cvt_f64_f32_e32 v[113:114], v136                         // 000000006d34: 7ee22188
	v_cvt_f64_f32_e32 v[133:134], v110                         // 000000006d38: 7f0a216e
	v_cmp_eq_f32_e64 s24, 0, v32                               // 000000006d3c: d4120018 02024080
	v_cmp_class_f32_e64 s26, v110, 0x1f8                       // 000000006d44: d47e001a 0201ff6e 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006d50: 8b181a18
	v_mul_f64_e32 v[111:112], v[111:112], v[113:114]           // 000000006d54: 0cdee36f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d58: bf870091
	v_mul_f64_e32 v[111:112], v[111:112], v[133:134]           // 000000006d5c: 0cdf0b6f
	v_cvt_f32_f64_e32 v111, v[111:112]                         // 000000006d60: 7ede1f6f
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d64: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d68: bf870001
	v_cndmask_b32_e64 v133, v111, 0, s24                       // 000000006d6c: d5010085 0061016f
	s_branch 62903                                             // 000000006d74: bfa0f5b7 <packed_folded_w4a8+0x2954>
	v_cvt_f64_f32_e32 v[112:113], v33                          // 000000006d78: 7ee02121
	v_cvt_f64_f32_e32 v[114:115], v136                         // 000000006d7c: 7ee42188
	v_cvt_f64_f32_e32 v[134:135], v32                          // 000000006d80: 7f0c2120
	v_cmp_eq_f32_e64 s24, 0, v33                               // 000000006d84: d4120018 02024280
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006d8c: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006d98: 8b181a18
	v_mul_f64_e32 v[112:113], v[112:113], v[114:115]           // 000000006d9c: 0ce0e570
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006da0: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[134:135]           // 000000006da4: 0ce10d70
	v_cvt_f32_f64_e32 v112, v[112:113]                         // 000000006da8: 7ee01f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000006dac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006db0: bf870001
	v_cndmask_b32_e64 v134, v112, 0, s24                       // 000000006db4: d5010086 00610170
	s_branch 62918                                             // 000000006dbc: bfa0f5c6 <packed_folded_w4a8+0x29d8>
	v_cvt_f64_f32_e32 v[114:115], v34                          // 000000006dc0: 7ee42122
	v_cvt_f64_f32_e32 v[116:117], v136                         // 000000006dc4: 7ee82188
	v_cvt_f64_f32_e32 v[137:138], v32                          // 000000006dc8: 7f122120
	v_cmp_eq_f32_e64 s24, 0, v34                               // 000000006dcc: d4120018 02024480
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006dd4: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006de0: 8b181a18
	v_mul_f64_e32 v[114:115], v[114:115], v[116:117]           // 000000006de4: 0ce4e972
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006de8: bf870091
	v_mul_f64_e32 v[114:115], v[114:115], v[137:138]           // 000000006dec: 0ce51372
	v_cvt_f32_f64_e32 v33, v[114:115]                          // 000000006df0: 7e421f72
	s_wait_alu depctr_sa_sdst(0)                               // 000000006df4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006df8: bf870001
	v_cndmask_b32_e64 v135, v33, 0, s24                        // 000000006dfc: d5010087 00610121
	s_branch 62933                                             // 000000006e04: bfa0f5d5 <packed_folded_w4a8+0x2a5c>
	v_cvt_f64_f32_e32 v[33:34], v35                            // 000000006e08: 7e422123
	v_cvt_f64_f32_e32 v[116:117], v136                         // 000000006e0c: 7ee82188
	v_cvt_f64_f32_e32 v[137:138], v32                          // 000000006e10: 7f122120
	v_cmp_eq_f32_e64 s24, 0, v35                               // 000000006e14: d4120018 02024680
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006e1c: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006e28: 8b181a18
	v_mul_f64_e32 v[33:34], v[33:34], v[116:117]               // 000000006e2c: 0c42e921
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e30: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[137:138]               // 000000006e34: 0c431321
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000006e38: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e3c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e40: bf870001
	v_cndmask_b32_e64 v128, v33, 0, s24                        // 000000006e44: d5010080 00610121
	s_branch 62948                                             // 000000006e4c: bfa0f5e4 <packed_folded_w4a8+0x2ae0>
	v_cvt_f64_f32_e32 v[116:117], v36                          // 000000006e50: 7ee82124
	v_cvt_f64_f32_e32 v[137:138], v136                         // 000000006e54: 7f122188
	v_cvt_f64_f32_e32 v[139:140], v32                          // 000000006e58: 7f162120
	v_cmp_eq_f32_e64 s24, 0, v36                               // 000000006e5c: d4120018 02024880
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006e64: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006e70: 8b181a18
	v_mul_f64_e32 v[116:117], v[116:117], v[137:138]           // 000000006e74: 0ce91374
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e78: bf870091
	v_mul_f64_e32 v[116:117], v[116:117], v[139:140]           // 000000006e7c: 0ce91774
	v_cvt_f32_f64_e32 v33, v[116:117]                          // 000000006e80: 7e421f74
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e84: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e88: bf870001
	v_cndmask_b32_e64 v129, v33, 0, s24                        // 000000006e8c: d5010081 00610121
	s_branch 62963                                             // 000000006e94: bfa0f5f3 <packed_folded_w4a8+0x2b64>
	v_cvt_f64_f32_e32 v[137:138], v37                          // 000000006e98: 7f122125
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006e9c: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006ea0: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v37                               // 000000006ea4: d4120018 02024a80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006eac: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006eb8: 8b181a18
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006ebc: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ec0: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006ec4: 0d131b89
	v_cvt_f32_f64_e32 v33, v[137:138]                          // 000000006ec8: 7e421f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ecc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ed0: bf870001
	v_cndmask_b32_e64 v130, v33, 0, s24                        // 000000006ed4: d5010082 00610121
	s_branch 62978                                             // 000000006edc: bfa0f602 <packed_folded_w4a8+0x2be8>
	v_cvt_f64_f32_e32 v[137:138], v38                          // 000000006ee0: 7f122126
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006ee4: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006ee8: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v38                               // 000000006eec: d4120018 02024c80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006ef4: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006f00: 8b181a18
	v_mul_f64_e32 v[137:138], v[137:138], v[139:140]           // 000000006f04: 0d131789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f08: bf870091
	v_mul_f64_e32 v[137:138], v[137:138], v[141:142]           // 000000006f0c: 0d131b89
	v_cvt_f32_f64_e32 v33, v[137:138]                          // 000000006f10: 7e421f89
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f14: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f18: bf870001
	v_cndmask_b32_e64 v131, v33, 0, s24                        // 000000006f1c: d5010083 00610121
	s_branch 62993                                             // 000000006f24: bfa0f611 <packed_folded_w4a8+0x2c6c>
	v_cvt_f64_f32_e32 v[137:138], v39                          // 000000006f28: 7f122127
	v_cvt_f64_f32_e32 v[139:140], v136                         // 000000006f2c: 7f162188
	v_cvt_f64_f32_e32 v[141:142], v32                          // 000000006f30: 7f1a2120
	v_cmp_eq_f32_e64 s24, 0, v39                               // 000000006f34: d4120018 02024e80
	v_cmp_class_f32_e64 s26, v32, 0x1f8                        // 000000006f3c: d47e001a 0201ff20 000001f8
	s_and_b32 s24, s24, s26                                    // 000000006f48: 8b181a18
	v_mul_f64_e32 v[136:137], v[137:138], v[139:140]           // 000000006f4c: 0d111789
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f50: bf870091
	v_mul_f64_e32 v[136:137], v[136:137], v[141:142]           // 000000006f54: 0d111b88
	v_cvt_f32_f64_e32 v33, v[136:137]                          // 000000006f58: 7e421f88
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f5c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f60: bf870001
	v_cndmask_b32_e64 v132, v33, 0, s24                        // 000000006f64: d5010084 00610121
	s_branch 63007                                             // 000000006f6c: bfa0f61f <packed_folded_w4a8+0x2cec>
	v_cvt_f64_f32_e32 v[122:123], v24                          // 000000006f70: 7ef42118
	v_cvt_f64_f32_e32 v[127:128], v73                          // 000000006f74: 7efe2149
	v_cvt_f64_f32_e32 v[129:130], v118                         // 000000006f78: 7f022176
	v_cmp_eq_f32_e64 s33, 0, v24                               // 000000006f7c: d4120021 02023080
	v_cmp_class_f32_e64 s37, v118, 0x1f8                       // 000000006f84: d47e0025 0201ff76 000001f8
	s_and_b32 s33, s33, s37                                    // 000000006f90: 8b212521
	v_mul_f64_e32 v[122:123], v[122:123], v[127:128]           // 000000006f94: 0cf4ff7a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f98: bf870091
	v_mul_f64_e32 v[122:123], v[122:123], v[129:130]           // 000000006f9c: 0cf5037a
	v_cvt_f32_f64_e32 v72, v[122:123]                          // 000000006fa0: 7e901f7a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006fa4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006fa8: bf870001
	v_cndmask_b32_e64 v72, v72, 0, s33                         // 000000006fac: d5010048 00850148
	s_branch 63319                                             // 000000006fb4: bfa0f757 <packed_folded_w4a8+0x3214>
	v_cvt_f64_f32_e32 v[118:119], v25                          // 000000006fb8: 7eec2119
	v_cvt_f64_f32_e32 v[122:123], v73                          // 000000006fbc: 7ef42149
	v_cvt_f64_f32_e32 v[127:128], v74                          // 000000006fc0: 7efe214a
	v_cmp_eq_f32_e64 s33, 0, v25                               // 000000006fc4: d4120021 02023280
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000006fcc: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000006fd8: 8b212521
	v_mul_f64_e32 v[118:119], v[118:119], v[122:123]           // 000000006fdc: 0cecf576
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006fe0: bf870091
	v_mul_f64_e32 v[118:119], v[118:119], v[127:128]           // 000000006fe4: 0cecff76
	v_cvt_f32_f64_e32 v24, v[118:119]                          // 000000006fe8: 7e301f76
	s_wait_alu depctr_sa_sdst(0)                               // 000000006fec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ff0: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s33                         // 000000006ff4: d5010018 00850118
	s_branch 63317                                             // 000000006ffc: bfa0f755 <packed_folded_w4a8+0x3254>
	v_cvt_f64_f32_e32 v[75:76], v26                            // 000000007000: 7e96211a
	v_cvt_f64_f32_e32 v[118:119], v73                          // 000000007004: 7eec2149
	v_cvt_f64_f32_e32 v[122:123], v74                          // 000000007008: 7ef4214a
	v_cmp_eq_f32_e64 s33, 0, v26                               // 00000000700c: d4120021 02023480
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000007014: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007020: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[118:119]               // 000000007024: 0c96ed4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007028: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[122:123]               // 00000000702c: 0c96f54b
	v_cvt_f32_f64_e32 v25, v[75:76]                            // 000000007030: 7e321f4b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007034: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007038: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s33                         // 00000000703c: d5010019 00850119
	s_branch 63315                                             // 000000007044: bfa0f753 <packed_folded_w4a8+0x3294>
	v_cvt_f64_f32_e32 v[75:76], v27                            // 000000007048: 7e96211b
	v_cvt_f64_f32_e32 v[77:78], v73                            // 00000000704c: 7e9a2149
	v_cvt_f64_f32_e32 v[118:119], v74                          // 000000007050: 7eec214a
	v_cmp_eq_f32_e64 s33, 0, v27                               // 000000007054: d4120021 02023680
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 00000000705c: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007068: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 00000000706c: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007070: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[118:119]               // 000000007074: 0c96ed4b
	v_cvt_f32_f64_e32 v26, v[75:76]                            // 000000007078: 7e341f4b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000707c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007080: bf870001
	v_cndmask_b32_e64 v26, v26, 0, s33                         // 000000007084: d501001a 0085011a
	s_branch 63313                                             // 00000000708c: bfa0f751 <packed_folded_w4a8+0x32d4>
	v_cvt_f64_f32_e32 v[75:76], v28                            // 000000007090: 7e96211c
	v_cvt_f64_f32_e32 v[77:78], v73                            // 000000007094: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007098: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v28                               // 00000000709c: d4120021 02023880
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 0000000070a4: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 0000000070b0: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 0000000070b4: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000070b8: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 0000000070bc: 0c969f4b
	v_cvt_f32_f64_e32 v27, v[75:76]                            // 0000000070c0: 7e361f4b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000070c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000070c8: bf870001
	v_cndmask_b32_e64 v27, v27, 0, s33                         // 0000000070cc: d501001b 0085011b
	s_branch 63311                                             // 0000000070d4: bfa0f74f <packed_folded_w4a8+0x3314>
	v_cvt_f64_f32_e32 v[75:76], v29                            // 0000000070d8: 7e96211d
	v_cvt_f64_f32_e32 v[77:78], v73                            // 0000000070dc: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 0000000070e0: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v29                               // 0000000070e4: d4120021 02023a80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 0000000070ec: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 0000000070f8: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 0000000070fc: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007100: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 000000007104: 0c969f4b
	v_cvt_f32_f64_e32 v28, v[75:76]                            // 000000007108: 7e381f4b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000710c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007110: bf870001
	v_cndmask_b32_e64 v28, v28, 0, s33                         // 000000007114: d501001c 0085011c
	s_branch 63309                                             // 00000000711c: bfa0f74d <packed_folded_w4a8+0x3354>
	v_cvt_f64_f32_e32 v[75:76], v30                            // 000000007120: 7e96211e
	v_cvt_f64_f32_e32 v[77:78], v73                            // 000000007124: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007128: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v30                               // 00000000712c: d4120021 02023c80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 000000007134: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007140: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 000000007144: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007148: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 00000000714c: 0c969f4b
	v_cvt_f32_f64_e32 v29, v[75:76]                            // 000000007150: 7e3a1f4b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007154: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007158: bf870001
	v_cndmask_b32_e64 v29, v29, 0, s33                         // 00000000715c: d501001d 0085011d
	s_branch 63307                                             // 000000007164: bfa0f74b <packed_folded_w4a8+0x3394>
	v_cvt_f64_f32_e32 v[75:76], v31                            // 000000007168: 7e96211f
	v_cvt_f64_f32_e32 v[77:78], v73                            // 00000000716c: 7e9a2149
	v_cvt_f64_f32_e32 v[79:80], v74                            // 000000007170: 7e9e214a
	v_cmp_eq_f32_e64 s33, 0, v31                               // 000000007174: d4120021 02023e80
	v_cmp_class_f32_e64 s37, v74, 0x1f8                        // 00000000717c: d47e0025 0201ff4a 000001f8
	s_and_b32 s33, s33, s37                                    // 000000007188: 8b212521
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 00000000718c: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007190: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[79:80]                 // 000000007194: 0c969f4b
	v_cvt_f32_f64_e32 v30, v[75:76]                            // 000000007198: 7e3c1f4b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000719c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000071a0: bf870001
	v_cndmask_b32_e64 v30, v30, 0, s33                         // 0000000071a4: d501001e 0085011e
	s_branch 63305                                             // 0000000071ac: bfa0f749 <packed_folded_w4a8+0x33d4>
	v_cvt_f64_f32_e32 v[27:28], v16                            // 0000000071b0: 7e362110
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000071b4: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 0000000071b8: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v16                                // 0000000071bc: d4120000 02022080
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000071c4: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000071d0: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000071d4: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000071d8: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 0000000071dc: 0c368d1b
	v_cvt_f32_f64_e32 v24, v[27:28]                            // 0000000071e0: 7e301f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000071e8: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s0                          // 0000000071ec: d5010018 00010118
	s_branch 63586                                             // 0000000071f4: bfa0f862 <packed_folded_w4a8+0x3880>
	v_cvt_f64_f32_e32 v[27:28], v17                            // 0000000071f8: 7e362111
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000071fc: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 000000007200: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v17                                // 000000007204: d4120000 02022280
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000720c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007218: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 00000000721c: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007220: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 000000007224: 0c368d1b
	v_cvt_f32_f64_e32 v16, v[27:28]                            // 000000007228: 7e201f1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000722c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007230: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s0                          // 000000007234: d5010010 00010110
	s_branch 63584                                             // 00000000723c: bfa0f860 <packed_folded_w4a8+0x38c0>
	v_cvt_f64_f32_e32 v[27:28], v18                            // 000000007240: 7e362112
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000007244: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 000000007248: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v18                                // 00000000724c: d4120000 02022480
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 000000007254: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007260: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007264: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007268: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 00000000726c: 0c368d1b
	v_cvt_f32_f64_e32 v17, v[27:28]                            // 000000007270: 7e221f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007274: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007278: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s0                          // 00000000727c: d5010011 00010111
	s_branch 63582                                             // 000000007284: bfa0f85e <packed_folded_w4a8+0x3900>
	v_cvt_f64_f32_e32 v[27:28], v19                            // 000000007288: 7e362113
	v_cvt_f64_f32_e32 v[29:30], v25                            // 00000000728c: 7e3a2119
	v_cvt_f64_f32_e32 v[70:71], v26                            // 000000007290: 7e8c211a
	v_cmp_eq_f32_e64 s0, 0, v19                                // 000000007294: d4120000 02022680
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000729c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000072a8: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000072ac: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000072b0: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[70:71]                 // 0000000072b4: 0c368d1b
	v_cvt_f32_f64_e32 v18, v[27:28]                            // 0000000072b8: 7e241f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000072bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000072c0: bf870001
	v_cndmask_b32_e64 v18, v18, 0, s0                          // 0000000072c4: d5010012 00010112
	s_branch 63580                                             // 0000000072cc: bfa0f85c <packed_folded_w4a8+0x3940>
	v_cvt_f64_f32_e32 v[27:28], v20                            // 0000000072d0: 7e362114
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000072d4: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 0000000072d8: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v20                                // 0000000072dc: d4120000 02022880
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000072e4: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000072f0: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000072f4: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000072f8: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 0000000072fc: 0c36651b
	v_cvt_f32_f64_e32 v19, v[27:28]                            // 000000007300: 7e261f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007304: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007308: bf870001
	v_cndmask_b32_e64 v19, v19, 0, s0                          // 00000000730c: d5010013 00010113
	s_branch 63578                                             // 000000007314: bfa0f85a <packed_folded_w4a8+0x3980>
	v_cvt_f64_f32_e32 v[27:28], v21                            // 000000007318: 7e362115
	v_cvt_f64_f32_e32 v[29:30], v25                            // 00000000731c: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 000000007320: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v21                                // 000000007324: d4120000 02022a80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 00000000732c: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007338: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 00000000733c: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007340: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 000000007344: 0c36651b
	v_cvt_f32_f64_e32 v20, v[27:28]                            // 000000007348: 7e281f1b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000734c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007350: bf870001
	v_cndmask_b32_e64 v20, v20, 0, s0                          // 000000007354: d5010014 00010114
	s_branch 63576                                             // 00000000735c: bfa0f858 <packed_folded_w4a8+0x39c0>
	v_cvt_f64_f32_e32 v[27:28], v22                            // 000000007360: 7e362116
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000007364: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 000000007368: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v22                                // 00000000736c: d4120000 02022c80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 000000007374: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007380: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007384: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007388: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 00000000738c: 0c36651b
	v_cvt_f32_f64_e32 v21, v[27:28]                            // 000000007390: 7e2a1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007394: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007398: bf870001
	v_cndmask_b32_e64 v21, v21, 0, s0                          // 00000000739c: d5010015 00010115
	s_branch 63574                                             // 0000000073a4: bfa0f856 <packed_folded_w4a8+0x3a00>
	v_cvt_f64_f32_e32 v[27:28], v23                            // 0000000073a8: 7e362117
	v_cvt_f64_f32_e32 v[29:30], v25                            // 0000000073ac: 7e3a2119
	v_cvt_f64_f32_e32 v[50:51], v26                            // 0000000073b0: 7e64211a
	v_cmp_eq_f32_e64 s0, 0, v23                                // 0000000073b4: d4120000 02022e80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 0000000073bc: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000073c8: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 0000000073cc: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000073d0: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[50:51]                 // 0000000073d4: 0c36651b
	v_cvt_f32_f64_e32 v22, v[27:28]                            // 0000000073d8: 7e2c1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000073e0: bf870001
	v_cndmask_b32_e64 v22, v22, 0, s0                          // 0000000073e4: d5010016 00010116
	s_branch 63572                                             // 0000000073ec: bfa0f854 <packed_folded_w4a8+0x3a40>
	v_cvt_f64_f32_e32 v[19:20], v8                             // 0000000073f0: 7e262108
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000073f4: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000073f8: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v8                                 // 0000000073fc: d4120000 02021080
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007404: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007410: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007414: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007418: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000741c: 0c262f13
	v_cvt_f32_f64_e32 v16, v[19:20]                            // 000000007420: 7e201f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007424: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007428: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s0                          // 00000000742c: d5010010 00010110
	s_branch 63853                                             // 000000007434: bfa0f96d <packed_folded_w4a8+0x3eec>
	v_cvt_f64_f32_e32 v[19:20], v9                             // 000000007438: 7e262109
	v_cvt_f64_f32_e32 v[21:22], v17                            // 00000000743c: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007440: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v9                                 // 000000007444: d4120000 02021280
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 00000000744c: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007458: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000745c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007460: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007464: 0c262f13
	v_cvt_f32_f64_e32 v8, v[19:20]                             // 000000007468: 7e101f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000746c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007470: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s0                            // 000000007474: d5010008 00010108
	s_branch 63851                                             // 00000000747c: bfa0f96b <packed_folded_w4a8+0x3f2c>
	v_cvt_f64_f32_e32 v[19:20], v10                            // 000000007480: 7e26210a
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000007484: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007488: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v10                                // 00000000748c: d4120000 02021480
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007494: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000074a0: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000074a4: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000074a8: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000074ac: 0c262f13
	v_cvt_f32_f64_e32 v9, v[19:20]                             // 0000000074b0: 7e121f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000074b8: bf870001
	v_cndmask_b32_e64 v9, v9, 0, s0                            // 0000000074bc: d5010009 00010109
	s_branch 63849                                             // 0000000074c4: bfa0f969 <packed_folded_w4a8+0x3f6c>
	v_cvt_f64_f32_e32 v[19:20], v11                            // 0000000074c8: 7e26210b
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000074cc: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000074d0: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v11                                // 0000000074d4: d4120000 02021680
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 0000000074dc: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000074e8: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000074ec: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000074f0: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000074f4: 0c262f13
	v_cvt_f32_f64_e32 v10, v[19:20]                            // 0000000074f8: 7e141f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074fc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007500: bf870001
	v_cndmask_b32_e64 v10, v10, 0, s0                          // 000000007504: d501000a 0001010a
	s_branch 63847                                             // 00000000750c: bfa0f967 <packed_folded_w4a8+0x3fac>
	v_cvt_f64_f32_e32 v[19:20], v12                            // 000000007510: 7e26210c
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000007514: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007518: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v12                                // 00000000751c: d4120000 02021880
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007524: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007530: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007534: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007538: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000753c: 0c262f13
	v_cvt_f32_f64_e32 v11, v[19:20]                            // 000000007540: 7e161f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007544: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007548: bf870001
	v_cndmask_b32_e64 v11, v11, 0, s0                          // 00000000754c: d501000b 0001010b
	s_branch 63845                                             // 000000007554: bfa0f965 <packed_folded_w4a8+0x3fec>
	v_cvt_f64_f32_e32 v[19:20], v13                            // 000000007558: 7e26210d
	v_cvt_f64_f32_e32 v[21:22], v17                            // 00000000755c: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000007560: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v13                                // 000000007564: d4120000 02021a80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 00000000756c: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007578: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000757c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007580: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007584: 0c262f13
	v_cvt_f32_f64_e32 v12, v[19:20]                            // 000000007588: 7e181f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000758c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007590: bf870001
	v_cndmask_b32_e64 v12, v12, 0, s0                          // 000000007594: d501000c 0001010c
	s_branch 63843                                             // 00000000759c: bfa0f963 <packed_folded_w4a8+0x402c>
	v_cvt_f64_f32_e32 v[19:20], v14                            // 0000000075a0: 7e26210e
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000075a4: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000075a8: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v14                                // 0000000075ac: d4120000 02021c80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 0000000075b4: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000075c0: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000075c4: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000075c8: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000075cc: 0c262f13
	v_cvt_f32_f64_e32 v13, v[19:20]                            // 0000000075d0: 7e1a1f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000075d8: bf870001
	v_cndmask_b32_e64 v13, v13, 0, s0                          // 0000000075dc: d501000d 0001010d
	s_branch 63841                                             // 0000000075e4: bfa0f961 <packed_folded_w4a8+0x406c>
	v_cvt_f64_f32_e32 v[19:20], v15                            // 0000000075e8: 7e26210f
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000075ec: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000075f0: 7e2e2112
	v_cmp_eq_f32_e64 s0, 0, v15                                // 0000000075f4: d4120000 02021e80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 0000000075fc: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007608: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000760c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007610: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007614: 0c262f13
	v_cvt_f32_f64_e32 v14, v[19:20]                            // 000000007618: 7e1c1f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000761c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007620: bf870001
	v_cndmask_b32_e64 v14, v14, 0, s0                          // 000000007624: d501000e 0001010e
	s_branch 63839                                             // 00000000762c: bfa0f95f <packed_folded_w4a8+0x40ac>
	v_cvt_f64_f32_e32 v[11:12], v0                             // 000000007630: 7e162100
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000007634: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007638: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v0                                 // 00000000763c: d4120000 02020080
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 000000007644: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007650: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000007654: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007658: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 00000000765c: 0c161f0b
	v_cvt_f32_f64_e32 v8, v[11:12]                             // 000000007660: 7e101f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007664: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007668: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s0                            // 00000000766c: d5010008 00010108
	s_branch 64120                                             // 000000007674: bfa0fa78 <packed_folded_w4a8+0x4558>
	v_cvt_f64_f32_e32 v[11:12], v1                             // 000000007678: 7e162101
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000767c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007680: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v1                                 // 000000007684: d4120000 02020280
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000768c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007698: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000769c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000076a0: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000076a4: 0c161f0b
	v_cvt_f32_f64_e32 v0, v[11:12]                             // 0000000076a8: 7e001f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076ac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000076b0: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s0                            // 0000000076b4: d5010000 00010100
	s_branch 64118                                             // 0000000076bc: bfa0fa76 <packed_folded_w4a8+0x4598>
	v_cvt_f64_f32_e32 v[11:12], v2                             // 0000000076c0: 7e162102
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000076c4: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000076c8: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v2                                 // 0000000076cc: d4120000 02020480
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000076d4: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000076e0: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000076e4: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000076e8: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000076ec: 0c161f0b
	v_cvt_f32_f64_e32 v1, v[11:12]                             // 0000000076f0: 7e021f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076f4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000076f8: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s0                            // 0000000076fc: d5010001 00010101
	s_branch 64116                                             // 000000007704: bfa0fa74 <packed_folded_w4a8+0x45d8>
	v_cvt_f64_f32_e32 v[11:12], v3                             // 000000007708: 7e162103
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000770c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007710: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v3                                 // 000000007714: d4120000 02020680
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000771c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007728: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000772c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007730: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000007734: 0c161f0b
	v_cvt_f32_f64_e32 v2, v[11:12]                             // 000000007738: 7e041f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000773c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007740: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s0                            // 000000007744: d5010002 00010102
	s_branch 64114                                             // 00000000774c: bfa0fa72 <packed_folded_w4a8+0x4618>
	v_cvt_f64_f32_e32 v[11:12], v4                             // 000000007750: 7e162104
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000007754: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007758: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v4                                 // 00000000775c: d4120000 02020880
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 000000007764: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007770: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000007774: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007778: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 00000000777c: 0c161f0b
	v_cvt_f32_f64_e32 v3, v[11:12]                             // 000000007780: 7e061f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007784: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007788: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s0                            // 00000000778c: d5010003 00010103
	s_branch 64112                                             // 000000007794: bfa0fa70 <packed_folded_w4a8+0x4658>
	v_cvt_f64_f32_e32 v[11:12], v5                             // 000000007798: 7e162105
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000779c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000077a0: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v5                                 // 0000000077a4: d4120000 02020a80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000077ac: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000077b8: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000077bc: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000077c0: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000077c4: 0c161f0b
	v_cvt_f32_f64_e32 v4, v[11:12]                             // 0000000077c8: 7e081f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000077cc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000077d0: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s0                            // 0000000077d4: d5010004 00010104
	s_branch 64110                                             // 0000000077dc: bfa0fa6e <packed_folded_w4a8+0x4698>
	v_cvt_f64_f32_e32 v[11:12], v6                             // 0000000077e0: 7e162106
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000077e4: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000077e8: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v6                                 // 0000000077ec: d4120000 02020c80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 0000000077f4: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007800: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000007804: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007808: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 00000000780c: 0c161f0b
	v_cvt_f32_f64_e32 v5, v[11:12]                             // 000000007810: 7e0a1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007814: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007818: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s0                            // 00000000781c: d5010005 00010105
	s_branch 64108                                             // 000000007824: bfa0fa6c <packed_folded_w4a8+0x46d8>
	v_cvt_f64_f32_e32 v[11:12], v7                             // 000000007828: 7e162107
	v_cvt_f64_f32_e32 v[13:14], v9                             // 00000000782c: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000007830: 7e1e210a
	v_cmp_eq_f32_e64 s0, 0, v7                                 // 000000007834: d4120000 02020e80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 00000000783c: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007848: 8b000200
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000784c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007850: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000007854: 0c161f0b
	v_cvt_f32_f64_e32 v6, v[11:12]                             // 000000007858: 7e0c1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000785c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007860: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s0                            // 000000007864: d5010006 00010106
	s_branch 64106                                             // 00000000786c: bfa0fa6a <packed_folded_w4a8+0x4718>
	s_code_end                                                 // 000000007870: bf9f0000
	s_code_end                                                 // 000000007874: bf9f0000
	s_code_end                                                 // 000000007878: bf9f0000
	s_code_end                                                 // 00000000787c: bf9f0000
	s_code_end                                                 // 000000007880: bf9f0000
	s_code_end                                                 // 000000007884: bf9f0000
	s_code_end                                                 // 000000007888: bf9f0000
	s_code_end                                                 // 00000000788c: bf9f0000
	s_code_end                                                 // 000000007890: bf9f0000
	s_code_end                                                 // 000000007894: bf9f0000
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
