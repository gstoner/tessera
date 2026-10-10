
/tmp/tmph_qap5d_.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164>:
	v_lshrrev_b32_e32 v1, 1, v0                                // 000000001b00: 32020081
	s_mov_b32 s4, ttmp7                                        // 000000001b04: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b08: 86059f73
	s_clause 0x1                                               // 000000001b0c: bf850001
	s_load_b64 s[14:15], s[0:1], 0xd8                          // 000000001b10: f4002380 f80000d8
	s_load_b128 s[44:47], s[0:1], 0xc8                         // 000000001b18: f4004b00 f80000c8
	s_lshl_b64 s[16:17], s[4:5], 7                             // 000000001b20: 84908704
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000001b24: bf8701b9
	v_dual_mov_b32 v45, s17 :: v_dual_and_b32 v2, 0x60, v1     // 000000001b28: ca240011 2d0202ff 00000060
	v_dual_mov_b32 v47, 0 :: v_dual_and_b32 v54, 15, v0        // 000000001b34: ca240080 2f36008f
	v_and_b32_e32 v55, 32, v0                                  // 000000001b3c: 366e00a0
	v_or_b32_e32 v3, 16, v2                                    // 000000001b40: 38060490
	v_or_b32_e32 v6, s16, v2                                   // 000000001b44: 380c0410
	s_delay_alu instid0(valu_dep_4)                            // 000000001b48: bf870004
	v_or_b32_e32 v2, v2, v54                                   // 000000001b4c: 38046d02
	v_and_b32_e32 v10, 8, v1                                   // 000000001b50: 36140288
	v_or_b32_e32 v5, v54, v55                                  // 000000001b54: 380a6f36
	v_or_b32_e32 v9, s16, v3                                   // 000000001b58: 38120610
	v_or_b32_e32 v3, v3, v54                                   // 000000001b5c: 38066d03
	s_mov_b32 s2, ttmp9                                        // 000000001b60: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b64: 86039f75
	v_mul_u32_u24_e32 v1, 0x50, v2                             // 000000001b68: 160204ff 00000050
	s_lshl_b64 s[48:49], s[2:3], 6                             // 000000001b70: 84b08602
	v_mul_u32_u24_e32 v2, 0x50, v3                             // 000000001b74: 160406ff 00000050
	v_mov_b32_e32 v3, s17                                      // 000000001b7c: 7e060211
	v_or_b32_e32 v44, v6, v10                                  // 000000001b80: 38581506
	v_dual_mov_b32 v41, s49 :: v_dual_lshlrev_b32 v4, 4, v0    // 000000001b84: ca220031 29040084
	v_lshrrev_b32_e32 v8, 2, v0                                // 000000001b8c: 32100082
	v_and_b32_e32 v0, 47, v0                                   // 000000001b90: 360000af
	v_or_b32_e32 v11, 16, v5                                   // 000000001b94: 38160a90
	s_wait_kmcnt 0x0                                           // 000000001b98: bfc70000
	v_cmp_gt_i64_e32 vcc_lo, s[44:45], v[44:45]                // 000000001b9c: 7ca8582c
	v_and_b32_e32 v46, 48, v4                                  // 000000001ba0: 365c08b0
	v_mul_u32_u24_e32 v4, 0x50, v8                             // 000000001ba4: 160810ff 00000050
	v_or_b32_e32 v74, v1, v10                                  // 000000001bac: 38941501
	v_mul_u32_u24_e32 v1, 0x50, v11                            // 000000001bb0: 160216ff 00000050
	v_or_b32_e32 v40, s48, v5                                  // 000000001bb8: 38500a30
	v_or_b32_e32 v12, 1, v10                                   // 000000001bbc: 38181481
	v_mov_b32_e32 v5, s17                                      // 000000001bc0: 7e0a0211
	v_mul_u32_u24_e32 v0, 0x50, v0                             // 000000001bc4: 160000ff 00000050
	v_or_b32_e32 v77, v1, v10                                  // 000000001bcc: 389a1501
	v_dual_cndmask_b32 v15, 0, v45 :: v_dual_add_nc_u32 v72, v4, v46// 000000001bd0: ca605a80 0f485d04
	v_mov_b32_e32 v1, s17                                      // 000000001bd8: 7e020211
	s_delay_alu instid0(valu_dep_4)                            // 000000001bdc: bf870004
	v_or_b32_e32 v76, v10, v0                                  // 000000001be0: 3898010a
	v_or_b32_e32 v0, v12, v6                                   // 000000001be4: 38000d0c
	v_or_b32_e32 v13, 2, v10                                   // 000000001be8: 381a1482
	v_or_b32_e32 v16, 3, v10                                   // 000000001bec: 38201483
	v_cndmask_b32_e32 v14, 0, v44, vcc_lo                      // 000000001bf0: 021c5880
	v_or_b32_e32 v75, v2, v10                                  // 000000001bf4: 38961502
	v_cmp_gt_i64_e32 vcc_lo, s[44:45], v[0:1]                  // 000000001bf8: 7ca8002c
	v_or_b32_e32 v2, v13, v6                                   // 000000001bfc: 38040d0d
	v_or_b32_e32 v4, v16, v6                                   // 000000001c00: 38080d10
	v_or_b32_e32 v18, 4, v10                                   // 000000001c04: 38241484
	v_or_b32_e32 v23, 5, v10                                   // 000000001c08: 382e1485
	v_or_b32_e32 v25, 6, v10                                   // 000000001c0c: 38321486
	v_cmp_gt_i64_e64 s3, s[44:45], v[2:3]                      // 000000001c10: d4540003 0202042c
	s_wait_alu depctr_va_vcc(0)                                // 000000001c18: bf88ff9d
	v_cndmask_b32_e32 v17, 0, v0, vcc_lo                       // 000000001c1c: 02220080
	v_cndmask_b32_e32 v19, 0, v1, vcc_lo                       // 000000001c20: 02260280
	v_cmp_gt_i64_e32 vcc_lo, s[44:45], v[4:5]                  // 000000001c24: 7ca8082c
	v_or_b32_e32 v0, v18, v6                                   // 000000001c28: 38000d12
	v_or_b32_e32 v27, 7, v10                                   // 000000001c2c: 38361487
	v_cndmask_b32_e64 v20, 0, v2, s3                           // 000000001c30: d5010014 000e0480
	v_or_b32_e32 v2, v23, v6                                   // 000000001c38: 38040d17
	v_cndmask_b32_e64 v21, 0, v3, s3                           // 000000001c3c: d5010015 000e0680
	s_wait_alu depctr_va_vcc(0)                                // 000000001c44: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v4, vcc_lo                       // 000000001c48: 022c0880
	v_cndmask_b32_e32 v24, 0, v5, vcc_lo                       // 000000001c4c: 02300a80
	v_cmp_gt_i64_e32 vcc_lo, s[44:45], v[0:1]                  // 000000001c50: 7ca8002c
	v_or_b32_e32 v4, v25, v6                                   // 000000001c54: 38080d19
	v_dual_mov_b32 v43, s17 :: v_dual_mov_b32 v90, v47         // 000000001c58: ca100011 2b5a012f
	v_or_b32_e32 v42, v9, v10                                  // 000000001c60: 38541509
	s_wait_alu depctr_va_vcc(0)                                // 000000001c64: bf88ff9d
	v_dual_mov_b32 v7, s49 :: v_dual_cndmask_b32 v28, 0, v1    // 000000001c68: ca120031 071c0280
	v_cndmask_b32_e32 v26, 0, v0, vcc_lo                       // 000000001c70: 02340080
	v_or_b32_e32 v0, v27, v6                                   // 000000001c74: 38000d1b
	v_cmp_gt_i64_e64 s3, s[44:45], v[4:5]                      // 000000001c78: d4540003 0202082c
	v_cmp_gt_i64_e32 vcc_lo, s[44:45], v[2:3]                  // 000000001c80: 7ca8042c
	v_or_b32_e32 v6, s48, v11                                  // 000000001c84: 380c1630
	s_add_nc_u64 s[18:19], s[44:45], -1                        // 000000001c88: a992c12c
	v_cmp_gt_i64_e64 s4, s[44:45], v[0:1]                      // 000000001c8c: d4540004 0202002c
	v_mov_b32_e32 v70, v47                                     // 000000001c94: 7e8c032f
	v_cndmask_b32_e64 v30, 0, v4, s3                           // 000000001c98: d501001e 000e0880
	v_cndmask_b32_e64 v31, 0, v5, s3                           // 000000001ca0: d501001f 000e0a80
	v_cmp_gt_i64_e64 s3, s[44:45], v[42:43]                    // 000000001ca8: d4540003 0202542c
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb0: bf88ff9d
	v_cndmask_b32_e32 v29, 0, v2, vcc_lo                       // 000000001cb4: 023a0480
	v_cndmask_b32_e64 v10, 0, v0, s4                           // 000000001cb8: d501000a 00120080
	v_or_b32_e32 v0, v9, v12                                   // 000000001cc0: 38001909
	v_or_b32_e32 v2, v9, v13                                   // 000000001cc4: 38041b09
	v_or_b32_e32 v4, v9, v16                                   // 000000001cc8: 38082109
	s_wait_alu depctr_va_sdst(0)                               // 000000001ccc: bf88f19f
	v_cndmask_b32_e64 v12, 0, v42, s3                          // 000000001cd0: d501000c 000e5480
	v_cndmask_b32_e64 v13, 0, v43, s3                          // 000000001cd8: d501000d 000e5680
	v_cmp_gt_i64_e64 s3, s[44:45], v[0:1]                      // 000000001ce0: d4540003 0202002c
	v_dual_cndmask_b32 v11, 0, v3 :: v_dual_mov_b32 v118, v47  // 000000001ce8: ca500680 0b76012f
	v_cmp_gt_i64_e32 vcc_lo, s[46:47], v[6:7]                  // 000000001cf0: 7ca80c2e
	v_cndmask_b32_e64 v32, 0, v1, s4                           // 000000001cf4: d5010020 00120280
	v_cmp_gt_i64_e64 s4, s[44:45], v[2:3]                      // 000000001cfc: d4540004 0202042c
	s_wait_alu depctr_va_sdst(0)                               // 000000001d04: bf88f19f
	v_cndmask_b32_e64 v33, 0, v0, s3                           // 000000001d08: d5010021 000e0080
	v_cndmask_b32_e64 v16, 0, v1, s3                           // 000000001d10: d5010010 000e0280
	v_cmp_gt_i64_e64 s3, s[44:45], v[4:5]                      // 000000001d18: d4540003 0202082c
	v_or_b32_e32 v0, v9, v23                                   // 000000001d20: 38002f09
	s_wait_alu depctr_va_vcc(0)                                // 000000001d24: bf88ff9d
	v_dual_cndmask_b32 v85, 0, v6 :: v_dual_mov_b32 v88, v47   // 000000001d28: ca500c80 5558012f
	v_mov_b32_e32 v7, s17                                      // 000000001d30: 7e0e0211
	v_or_b32_e32 v6, v9, v18                                   // 000000001d34: 380c2509
	s_wait_alu depctr_va_sdst(0)                               // 000000001d38: bf88f19f
	v_cndmask_b32_e64 v23, 0, v4, s3                           // 000000001d3c: d5010017 000e0880
	v_cndmask_b32_e64 v35, 0, v5, s3                           // 000000001d44: d5010023 000e0a80
	v_cmp_gt_i64_e64 s3, s[44:45], v[0:1]                      // 000000001d4c: d4540003 0202002c
	v_cndmask_b32_e64 v18, 0, v2, s4                           // 000000001d54: d5010012 00120480
	v_or_b32_e32 v2, s16, v8                                   // 000000001d5c: 38041010
	v_cndmask_b32_e64 v34, 0, v3, s4                           // 000000001d60: d5010022 00120680
	v_cmp_gt_i64_e64 s4, s[44:45], v[6:7]                      // 000000001d68: d4540004 02020c2c
	v_or_b32_e32 v4, v9, v25                                   // 000000001d70: 38083309
	s_wait_alu depctr_va_sdst(0)                               // 000000001d74: bf88f19f
	v_cndmask_b32_e64 v25, 0, v0, s3                           // 000000001d78: d5010019 000e0080
	v_cndmask_b32_e64 v38, 0, v1, s3                           // 000000001d80: d5010026 000e0280
	v_cmp_gt_u64_e64 s3, s[18:19], v[2:3]                      // 000000001d88: d45c0003 02020412
	v_dual_mov_b32 v39, s17 :: v_dual_mov_b32 v68, v47         // 000000001d90: ca100011 2744012f
	v_cndmask_b32_e64 v36, 0, v6, s4                           // 000000001d98: d5010024 00120c80
	v_or_b32_e32 v0, v9, v27                                   // 000000001da0: 38003709
	s_clause 0x3                                               // 000000001da4: bf850003
	s_load_b64 s[12:13], s[0:1], 0x8                           // 000000001da8: f4002300 f8000008
	s_load_b64 s[10:11], s[0:1], 0x30                          // 000000001db0: f4002280 f8000030
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000001db8: f4002180 f8000058
	s_load_b64 s[52:53], s[0:1], 0x80                          // 000000001dc0: f4002d00 f8000080
	v_cndmask_b32_e64 v6, s18, v2, s3                          // 000000001dc8: d5010006 000e0412
	v_or_b32_e32 v2, 64, v2                                    // 000000001dd0: 380404c0
	v_cndmask_b32_e64 v9, s19, v39, s3                         // 000000001dd4: d5010009 000e4e13
	v_cmp_gt_i64_e64 s3, s[44:45], v[0:1]                      // 000000001ddc: d4540003 0202002c
	v_cndmask_b32_e64 v37, 0, v7, s4                           // 000000001de4: d5010025 00120e80
	v_mul_lo_u32 v27, s15, v6                                  // 000000001dec: d72c001b 02020c0f
	v_cmp_gt_u64_e64 s5, s[18:19], v[2:3]                      // 000000001df4: d45c0005 02020412
	v_mad_co_u64_u32 v[6:7], null, s14, v6, v[46:47]           // 000000001dfc: d6fe7c06 04ba0c0e
	v_mul_lo_u32 v9, s14, v9                                   // 000000001e04: d72c0009 0202120e
	v_cmp_gt_i64_e64 s4, s[44:45], v[4:5]                      // 000000001e0c: d4540004 0202082c
	s_wait_alu depctr_va_sdst(0)                               // 000000001e14: bf88f19f
	v_cndmask_b32_e64 v58, 0, v0, s3                           // 000000001e18: d501003a 000e0080
	v_or_b32_e32 v3, s48, v8                                   // 000000001e20: 38061030
	v_cndmask_b32_e64 v0, s18, v2, s5                          // 000000001e24: d5010000 00160412
	v_cndmask_b32_e64 v2, s19, v39, s5                         // 000000001e2c: d5010002 00164e13
	v_cndmask_b32_e64 v59, 0, v1, s3                           // 000000001e34: d501003b 000e0280
	v_cndmask_b32_e64 v56, 0, v4, s4                           // 000000001e3c: d5010038 00120880
	v_cndmask_b32_e64 v57, 0, v5, s4                           // 000000001e44: d5010039 00120a80
	v_add3_u32 v4, v27, v7, v9                                 // 000000001e4c: d6550004 04260f1b
	v_mul_lo_u32 v5, s15, v0                                   // 000000001e54: d72c0005 0202000f
	v_mad_co_u64_u32 v[0:1], null, s14, v0, v[46:47]           // 000000001e5c: d6fe7c00 04ba000e
	v_mul_lo_u32 v7, s14, v2                                   // 000000001e64: d72c0007 0202040e
	v_mul_lo_u32 v8, s15, v3                                   // 000000001e6c: d72c0008 0202060f
	v_mad_co_u64_u32 v[2:3], null, s14, v3, v[46:47]           // 000000001e74: d6fe7c02 04ba060e
	s_wait_kmcnt 0x0                                           // 000000001e7c: bfc70000
	v_add_co_u32 v48, s3, s12, v6                              // 000000001e80: d7000330 02020c0c
	s_lshr_b64 s[8:9], s[14:15], 5                             // 000000001e88: 8588850e
	v_add_co_ci_u32_e64 v49, null, s13, v4, s3                 // 000000001e8c: d5207c31 000e080d
	s_mul_i32 s3, s14, s49                                     // 000000001e94: 9603310e
	s_lshr_b32 s4, s15, 5                                      // 000000001e98: 8504850f
	v_add3_u32 v1, v5, v1, v7                                  // 000000001e9c: d6550001 041e0305
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ea4: bf88ff9e
	v_add3_u32 v7, v8, v3, s3                                  // 000000001ea8: d6550007 000e0708
	v_mul_lo_u32 v8, s8, v15                                   // 000000001eb0: d72c0008 02021e08
	v_mul_lo_u32 v9, s4, v14                                   // 000000001eb8: d72c0009 02021c04
	v_mad_co_u64_u32 v[3:4], null, s8, v14, s[6:7]             // 000000001ec0: d6fe7c03 001a1c08
	v_mul_lo_u32 v14, s8, v19                                  // 000000001ec8: d72c000e 02022608
	v_mul_lo_u32 v15, s4, v17                                  // 000000001ed0: d72c000f 02022204
	v_mad_co_u64_u32 v[5:6], null, s8, v17, s[6:7]             // 000000001ed8: d6fe7c05 001a2208
	v_add_co_u32 v50, s3, s12, v0                              // 000000001ee0: d7000332 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001ee8: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s13, v1, s3                 // 000000001eec: d5207c33 000e020d
	v_add_co_u32 v52, s3, s10, v2                              // 000000001ef4: d7000334 0202040a
	s_wait_alu depctr_va_sdst(0)                               // 000000001efc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s11, v7, s3                 // 000000001f00: d5207c35 000e0e0b
	v_add3_u32 v2, v9, v4, v8                                  // 000000001f08: d6550002 04220909
	v_add3_u32 v4, v15, v6, v14                                // 000000001f10: d6550004 043a0d0f
	v_mul_lo_u32 v6, s8, v21                                   // 000000001f18: d72c0006 02022a08
	v_mul_lo_u32 v7, s4, v20                                   // 000000001f20: d72c0007 02022804
	v_mad_co_u64_u32 v[0:1], null, s8, v20, s[6:7]             // 000000001f28: d6fe7c00 001a2808
	v_cmp_gt_i64_e64 s2, s[46:47], v[40:41]                    // 000000001f30: d4540002 0202502e
	v_add_co_u32 v92, s3, v3, 1                                // 000000001f38: d700035c 02010303
	s_wait_alu depctr_va_sdst(0)                               // 000000001f40: bf88f19f
	v_add_co_ci_u32_e64 v93, null, 0, v2, s3                   // 000000001f44: d5207c5d 000e0480
	v_add_co_u32 v94, s3, v5, 1                                // 000000001f4c: d700035e 02010305
	s_delay_alu instid0(valu_dep_4)                            // 000000001f54: bf870004
	v_cndmask_b32_e64 v80, 0, v40, s2                          // 000000001f58: d5010050 000a5080
	s_wait_alu depctr_va_sdst(0)                               // 000000001f60: bf88f19f
	v_add_co_ci_u32_e64 v95, null, 0, v4, s3                   // 000000001f64: d5207c5f 000e0880
	v_add3_u32 v3, v7, v1, v6                                  // 000000001f6c: d6550003 041a0307
	v_mul_lo_u32 v4, s8, v24                                   // 000000001f74: d72c0004 02023008
	v_mul_lo_u32 v5, s4, v22                                   // 000000001f7c: d72c0005 02022c04
	v_mad_co_u64_u32 v[1:2], null, s8, v22, s[6:7]             // 000000001f84: d6fe7c01 001a2c08
	v_cndmask_b32_e64 v79, 0, s49, s2                          // 000000001f8c: d501004f 00086280
	v_add_co_u32 v96, s3, v80, s46                             // 000000001f94: d7000360 02005d50
	v_mul_lo_u32 v6, s8, v28                                   // 000000001f9c: d72c0006 02023808
	v_mul_lo_u32 v7, s4, v26                                   // 000000001fa4: d72c0007 02023404
	s_wait_alu depctr_va_sdst(0)                               // 000000001fac: bf88f19f
	v_add_co_ci_u32_e64 v97, null, s47, v79, s3                // 000000001fb0: d5207c61 000e9e2f
	v_add_co_u32 v98, s3, v0, 1                                // 000000001fb8: d7000362 02010300
	s_wait_alu depctr_va_sdst(0)                               // 000000001fc0: bf88f19f
	v_add_co_ci_u32_e64 v99, null, 0, v3, s3                   // 000000001fc4: d5207c63 000e0680
	v_add3_u32 v0, v5, v2, v4                                  // 000000001fcc: d6550000 04120505
	v_mad_co_u64_u32 v[2:3], null, s8, v26, s[6:7]             // 000000001fd4: d6fe7c02 001a3408
	v_mul_lo_u32 v8, s8, v59                                   // 000000001fdc: d72c0008 02027608
	v_mul_lo_u32 v9, s4, v58                                   // 000000001fe4: d72c0009 02027404
	v_mad_co_u64_u32 v[4:5], null, s8, v58, s[6:7]             // 000000001fec: d6fe7c04 001a7408
	v_add_co_u32 v100, s3, v1, 1                               // 000000001ff4: d7000364 02010301
	v_cndmask_b32_e64 v84, 0, s49, vcc_lo                      // 000000001ffc: d5010054 01a86280
	s_wait_alu depctr_va_sdst(0)                               // 000000002004: bf88f19f
	v_add_co_ci_u32_e64 v102, null, 0, v0, s3                  // 000000002008: d5207c66 000e0080
	v_add3_u32 v3, v7, v3, v6                                  // 000000002010: d6550003 041a0707
	v_mad_co_u64_u32 v[0:1], null, s8, v29, s[6:7]             // 000000002018: d6fe7c00 001a3a08
	v_add3_u32 v7, v9, v5, v8                                  // 000000002020: d6550007 04220b09
	v_mul_lo_u32 v8, s8, v11                                   // 000000002028: d72c0008 02021608
	v_mul_lo_u32 v9, s4, v29                                   // 000000002030: d72c0009 02023a04
	v_mul_lo_u32 v11, s8, v57                                  // 000000002038: d72c000b 02027208
	v_mul_lo_u32 v14, s4, v56                                  // 000000002040: d72c000e 02027004
	v_mad_co_u64_u32 v[5:6], null, s8, v56, s[6:7]             // 000000002048: d6fe7c05 001a7008
	v_add_co_u32 v103, s3, v85, s46                            // 000000002050: d7000367 02005d55
	s_wait_alu depctr_va_sdst(0)                               // 000000002058: bf88f19f
	v_add_co_ci_u32_e64 v104, null, s47, v84, s3               // 00000000205c: d5207c68 000ea82f
	v_add_co_u32 v105, s3, v2, 1                               // 000000002064: d7000369 02010302
	s_wait_alu depctr_va_sdst(0)                               // 00000000206c: bf88f19f
	v_add_co_ci_u32_e64 v106, null, 0, v3, s3                  // 000000002070: d5207c6a 000e0680
	v_add_co_u32 v107, s3, v4, 1                               // 000000002078: d700036b 02010304
	s_wait_alu depctr_va_sdst(0)                               // 000000002080: bf88f19f
	v_add_co_ci_u32_e64 v108, null, 0, v7, s3                  // 000000002084: d5207c6c 000e0e80
	v_add3_u32 v7, v9, v1, v8                                  // 00000000208c: d6550007 04220309
	v_add3_u32 v6, v14, v6, v11                                // 000000002094: d6550006 042e0d0e
	v_mul_lo_u32 v8, s8, v31                                   // 00000000209c: d72c0008 02023e08
	v_mul_lo_u32 v9, s4, v30                                   // 0000000020a4: d72c0009 02023c04
	v_mad_co_u64_u32 v[1:2], null, s8, v30, s[6:7]             // 0000000020ac: d6fe7c01 001a3c08
	v_mul_lo_u32 v11, s8, v38                                  // 0000000020b4: d72c000b 02024c08
	v_mul_lo_u32 v14, s4, v25                                  // 0000000020bc: d72c000e 02023204
	v_mad_co_u64_u32 v[3:4], null, s8, v25, s[6:7]             // 0000000020c4: d6fe7c03 001a3208
	v_add_co_u32 v110, s3, v0, 1                               // 0000000020cc: d700036e 02010300
	s_wait_alu depctr_va_sdst(0)                               // 0000000020d4: bf88f19f
	v_add_co_ci_u32_e64 v111, null, 0, v7, s3                  // 0000000020d8: d5207c6f 000e0e80
	v_add_co_u32 v112, s3, v5, 1                               // 0000000020e0: d7000370 02010305
	s_wait_alu depctr_va_sdst(0)                               // 0000000020e8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, 0, v6, s3                  // 0000000020ec: d5207c71 000e0c80
	v_add3_u32 v0, v9, v2, v8                                  // 0000000020f4: d6550000 04220509
	v_add3_u32 v2, v14, v4, v11                                // 0000000020fc: d6550002 042e090e
	v_mul_lo_u32 v8, s8, v32                                   // 000000002104: d72c0008 02024008
	v_mul_lo_u32 v9, s4, v10                                   // 00000000210c: d72c0009 02021404
	v_mad_co_u64_u32 v[4:5], null, s8, v10, s[6:7]             // 000000002114: d6fe7c04 001a1408
	v_mul_lo_u32 v10, s8, v37                                  // 00000000211c: d72c000a 02024a08
	v_mul_lo_u32 v11, s4, v36                                  // 000000002124: d72c000b 02024804
	v_mad_co_u64_u32 v[6:7], null, s8, v36, s[6:7]             // 00000000212c: d6fe7c06 001a4808
	v_add_co_u32 v114, s3, v1, 1                               // 000000002134: d7000372 02010301
	s_wait_alu depctr_va_sdst(0)                               // 00000000213c: bf88f19f
	v_add_co_ci_u32_e64 v115, null, 0, v0, s3                  // 000000002140: d5207c73 000e0080
	v_add_co_u32 v116, s3, v3, 1                               // 000000002148: d7000374 02010303
	s_wait_alu depctr_va_sdst(0)                               // 000000002150: bf88f19f
	v_add_co_ci_u32_e64 v117, null, 0, v2, s3                  // 000000002154: d5207c75 000e0480
	v_add3_u32 v5, v9, v5, v8                                  // 00000000215c: d6550005 04220b09
	v_add3_u32 v7, v11, v7, v10                                // 000000002164: d6550007 042a0f0b
	v_mul_lo_u32 v8, s8, v13                                   // 00000000216c: d72c0008 02021a08
	v_mul_lo_u32 v9, s4, v12                                   // 000000002174: d72c0009 02021804
	v_mad_co_u64_u32 v[0:1], null, s8, v12, s[6:7]             // 00000000217c: d6fe7c00 001a1808
	v_mul_lo_u32 v10, s8, v35                                  // 000000002184: d72c000a 02024608
	v_mul_lo_u32 v11, s4, v23                                  // 00000000218c: d72c000b 02022e04
	v_mad_co_u64_u32 v[2:3], null, s8, v23, s[6:7]             // 000000002194: d6fe7c02 001a2e08
	v_add_co_u32 v119, s3, v4, 1                               // 00000000219c: d7000377 02010304
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a4: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v5, s3                  // 0000000021a8: d5207c78 000e0a80
	v_add_co_u32 v121, s3, v6, 1                               // 0000000021b0: d7000379 02010306
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b8: bf88f19f
	v_add_co_ci_u32_e64 v122, null, 0, v7, s3                  // 0000000021bc: d5207c7a 000e0e80
	v_add3_u32 v1, v9, v1, v8                                  // 0000000021c4: d6550001 04220309
	v_add3_u32 v7, v11, v3, v10                                // 0000000021cc: d6550007 042a070b
	v_mul_lo_u32 v8, s8, v16                                   // 0000000021d4: d72c0008 02022008
	v_mul_lo_u32 v9, s4, v33                                   // 0000000021dc: d72c0009 02024204
	v_mad_co_u64_u32 v[3:4], null, s8, v33, s[6:7]             // 0000000021e4: d6fe7c03 001a4208
	v_mul_lo_u32 v10, s8, v34                                  // 0000000021ec: d72c000a 02024408
	v_mul_lo_u32 v11, s4, v18                                  // 0000000021f4: d72c000b 02022404
	v_mad_co_u64_u32 v[5:6], null, s8, v18, s[6:7]             // 0000000021fc: d6fe7c05 001a2408
	v_add_co_u32 v123, s3, v0, 1                               // 000000002204: d700037b 02010300
	s_wait_alu depctr_va_sdst(0)                               // 00000000220c: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v1, s3                  // 000000002210: d5207c7c 000e0280
	v_add3_u32 v0, v9, v4, v8                                  // 000000002218: d6550000 04220909
	v_add_co_u32 v125, s3, v2, 1                               // 000000002220: d700037d 02010302
	v_add3_u32 v1, v11, v6, v10                                // 000000002228: d6550001 042a0d0b
	s_wait_alu depctr_va_sdst(0)                               // 000000002230: bf88f19f
	v_add_co_ci_u32_e64 v126, null, 0, v7, s3                  // 000000002234: d5207c7e 000e0e80
	v_add_co_u32 v127, s3, v3, 1                               // 00000000223c: d700037f 02010303
	s_wait_alu depctr_va_sdst(0)                               // 000000002244: bf88f19f
	v_add_co_ci_u32_e64 v128, null, 0, v0, s3                  // 000000002248: d5207c80 000e0080
	v_add_co_u32 v129, s3, v5, 1                               // 000000002250: d7000381 02010305
	s_wait_alu depctr_va_sdst(0)                               // 000000002258: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v1, s3                  // 00000000225c: d5207c82 000e0280
	v_dual_mov_b32 v109, v47 :: v_dual_mov_b32 v66, v47        // 000000002264: ca10012f 6d42012f
	v_dual_mov_b32 v101, v47 :: v_dual_mov_b32 v64, v47        // 00000000226c: ca10012f 6540012f
	v_dual_mov_b32 v91, v47 :: v_dual_mov_b32 v86, v47         // 000000002274: ca10012f 5b56012f
	v_dual_mov_b32 v89, v47 :: v_dual_mov_b32 v82, v47         // 00000000227c: ca10012f 5952012f
	v_dual_mov_b32 v69, v47 :: v_dual_mov_b32 v78, v47         // 000000002284: ca10012f 454e012f
	v_dual_mov_b32 v67, v47 :: v_dual_mov_b32 v62, v47         // 00000000228c: ca10012f 433e012f
	v_dual_mov_b32 v65, v47 :: v_dual_mov_b32 v60, v47         // 000000002294: ca10012f 413c012f
	v_dual_mov_b32 v63, v47 :: v_dual_mov_b32 v58, v47         // 00000000229c: ca10012f 3f3a012f
	v_dual_mov_b32 v87, v47 :: v_dual_mov_b32 v56, v47         // 0000000022a4: ca10012f 5738012f
	v_dual_mov_b32 v83, v47 :: v_dual_mov_b32 v46, v47         // 0000000022ac: ca10012f 532e012f
	v_mov_b32_e32 v81, v47                                     // 0000000022b4: 7ea2032f
	v_mov_b32_e32 v73, v47                                     // 0000000022b8: 7e92032f
	v_mov_b32_e32 v71, v47                                     // 0000000022bc: 7e8e032f
	v_mov_b32_e32 v61, v47                                     // 0000000022c0: 7e7a032f
	v_mov_b32_e32 v59, v47                                     // 0000000022c4: 7e76032f
	v_mov_b32_e32 v57, v47                                     // 0000000022c8: 7e72032f
	s_and_b32 s54, s8, -2                                      // 0000000022cc: 8b36c208
	s_mov_b32 s55, s9                                          // 0000000022d0: beb70009
	s_lshl_b64 s[50:51], s[46:47], 1                           // 0000000022d4: 84b2812e
	s_mov_b64 s[56:57], 0                                      // 0000000022d8: beb80180
	global_load_b128 v[0:3], v[48:49], off                     // 0000000022dc: ee05c07c 00000000 00000030
	global_load_b128 v[4:7], v[50:51], off                     // 0000000022e8: ee05c07c 00000004 00000032
	global_load_b128 v[8:11], v[52:53], off                    // 0000000022f4: ee05c07c 00000008 00000034
	v_add_co_u32 v14, s3, s52, v80                             // 000000002300: d700030e 0202a034
	v_add_co_u32 v32, s12, s52, v85                            // 000000002308: d7000c20 0202aa34
	s_wait_alu depctr_va_sdst(0)                               // 000000002310: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s53, v79, s3                // 000000002314: d5207c0f 000e9e35
	v_add_co_ci_u32_e64 v33, null, s53, v84, s12               // 00000000231c: d5207c21 0032a835
	v_add_co_u32 v16, s4, v92, s56                             // 000000002324: d7000410 0200715c
	v_add_co_u32 v18, s5, v94, s56                             // 00000000232c: d7000512 0200715e
	v_add_co_u32 v20, s6, v98, s56                             // 000000002334: d7000614 02007162
	v_add_co_u32 v22, s7, v100, s56                            // 00000000233c: d7000716 02007164
	v_add_co_u32 v24, s8, v105, s56                            // 000000002344: d7000818 02007169
	v_add_co_u32 v26, s9, v110, s56                            // 00000000234c: d700091a 0200716e
	v_add_co_u32 v28, s10, v114, s56                           // 000000002354: d7000a1c 02007172
	v_add_co_u32 v30, s11, v119, s56                           // 00000000235c: d7000b1e 02007177
	v_add_co_u32 v34, s13, v123, s56                           // 000000002364: d7000d22 0200717b
	v_add_co_u32 v36, s14, v127, s56                           // 00000000236c: d7000e24 0200717f
	v_add_co_u32 v38, s15, v129, s56                           // 000000002374: d7000f26 02007181
	v_add_co_u32 v131, s16, v125, s56                          // 00000000237c: d7001083 0200717d
	v_add_co_u32 v133, s17, v121, s56                          // 000000002384: d7001185 02007179
	v_add_co_u32 v135, s18, v116, s56                          // 00000000238c: d7001287 02007174
	v_add_co_u32 v137, s19, v112, s56                          // 000000002394: d7001389 02007170
	v_add_co_u32 v139, s20, v107, s56                          // 00000000239c: d700148b 0200716b
	v_add_co_u32 v141, s21, s52, v96                           // 0000000023a4: d700158d 0202c034
	v_add_co_u32 v143, s22, s52, v103                          // 0000000023ac: d700168f 0202ce34
	s_barrier_signal -1                                        // 0000000023b4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000023b8: bf94ffff
	s_wait_alu depctr_va_sdst(0)                               // 0000000023bc: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s57, v93, s4                // 0000000023c0: d5207c11 0012ba39
	v_add_co_ci_u32_e64 v19, null, s57, v95, s5                // 0000000023c8: d5207c13 0016be39
	v_add_co_ci_u32_e64 v21, null, s57, v99, s6                // 0000000023d0: d5207c15 001ac639
	v_add_co_ci_u32_e64 v23, null, s57, v102, s7               // 0000000023d8: d5207c17 001ecc39
	v_add_co_ci_u32_e64 v25, null, s57, v106, s8               // 0000000023e0: d5207c19 0022d439
	v_add_co_ci_u32_e64 v27, null, s57, v111, s9               // 0000000023e8: d5207c1b 0026de39
	v_add_co_ci_u32_e64 v29, null, s57, v115, s10              // 0000000023f0: d5207c1d 002ae639
	v_add_co_ci_u32_e64 v31, null, s57, v120, s11              // 0000000023f8: d5207c1f 002ef039
	v_add_co_ci_u32_e64 v35, null, s57, v124, s13              // 000000002400: d5207c23 0036f839
	v_add_co_ci_u32_e64 v37, null, s57, v128, s14              // 000000002408: d5207c25 003b0039
	v_add_co_ci_u32_e64 v39, null, s57, v130, s15              // 000000002410: d5207c27 003f0439
	v_add_co_ci_u32_e64 v132, null, s57, v126, s16             // 000000002418: d5207c84 0042fc39
	v_add_co_ci_u32_e64 v134, null, s57, v122, s17             // 000000002420: d5207c86 0046f439
	v_add_co_ci_u32_e64 v136, null, s57, v117, s18             // 000000002428: d5207c88 004aea39
	v_add_co_ci_u32_e64 v138, null, s57, v113, s19             // 000000002430: d5207c8a 004ee239
	v_add_co_ci_u32_e64 v140, null, s57, v108, s20             // 000000002438: d5207c8c 0052d839
	v_add_co_ci_u32_e64 v142, null, s53, v97, s21              // 000000002440: d5207c8e 0056c235
	v_add_co_ci_u32_e64 v144, null, s53, v104, s22             // 000000002448: d5207c90 005ad035
	v_add_nc_u32_e32 v12, 0x2800, v76                          // 000000002450: 4a1898ff 00002800
	v_add_nc_u32_e32 v13, 0x2800, v77                          // 000000002458: 4a1a9aff 00002800
	v_add_co_u32 v48, s40, v48, 64                             // 000000002460: d7002830 02018130
	v_add_co_u32 v50, s41, v50, 64                             // 000000002468: d7002932 02018132
	v_add_co_u32 v52, s42, v52, 64                             // 000000002470: d7002a34 02018134
	s_wait_alu depctr_va_sdst(0)                               // 000000002478: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v49, s40                 // 00000000247c: d5207c31 00a26280
	v_add_co_ci_u32_e64 v51, null, 0, v51, s41                 // 000000002484: d5207c33 00a66680
	v_add_co_ci_u32_e64 v53, null, 0, v53, s42                 // 00000000248c: d5207c35 00aa6a80
	s_add_nc_u64 s[56:57], s[56:57], 2                         // 000000002494: a9b88238
	s_add_nc_u64 s[52:53], s[52:53], s[50:51]                  // 000000002498: a9b43234
	s_wait_loadcnt 0x2                                         // 00000000249c: bfc00002
	ds_store_b128 v72, v[0:3]                                  // 0000000024a0: db7c0000 00000048
	s_wait_loadcnt 0x1                                         // 0000000024a8: bfc00001
	ds_store_b128 v72, v[4:7] offset:5120                      // 0000000024ac: db7c1400 00000448
	s_wait_loadcnt 0x0                                         // 0000000024b4: bfc00000
	ds_store_b128 v72, v[8:11] offset:10240                    // 0000000024b8: db7c2800 00000848
	s_wait_dscnt 0x0                                           // 0000000024c0: bfc60000
	s_barrier_signal -1                                        // 0000000024c4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000024c8: bf94ffff
	s_clause 0x1                                               // 0000000024cc: bf850001
	global_load_u8 v171, v[14:15], off                         // 0000000024d0: ee04007c 000000ab 0000000e
	global_load_u8 v172, v[32:33], off                         // 0000000024dc: ee04007c 000000ac 00000020
	s_clause 0x3                                               // 0000000024e8: bf850003
	global_load_u8 v173, v[16:17], off offset:-1               // 0000000024ec: ee04007c 000000ad ffffff10
	global_load_u8 v174, v[18:19], off offset:-1               // 0000000024f8: ee04007c 000000ae ffffff12
	global_load_u8 v175, v[20:21], off offset:-1               // 000000002504: ee04007c 000000af ffffff14
	global_load_u8 v176, v[22:23], off offset:-1               // 000000002510: ee04007c 000000b0 ffffff16
	global_load_u8 v177, v[141:142], off                       // 00000000251c: ee04007c 000000b1 0000008d
	global_load_u8 v178, v[24:25], off offset:-1               // 000000002528: ee04007c 000000b2 ffffff18
	global_load_u8 v179, v[143:144], off                       // 000000002534: ee04007c 000000b3 0000008f
	s_clause 0x1a                                              // 000000002540: bf85001a
	global_load_u8 v180, v[26:27], off offset:-1               // 000000002544: ee04007c 000000b4 ffffff1a
	global_load_u8 v181, v[28:29], off offset:-1               // 000000002550: ee04007c 000000b5 ffffff1c
	global_load_u8 v182, v[30:31], off offset:-1               // 00000000255c: ee04007c 000000b6 ffffff1e
	global_load_u8 v183, v[34:35], off offset:-1               // 000000002568: ee04007c 000000b7 ffffff22
	global_load_u8 v184, v[36:37], off offset:-1               // 000000002574: ee04007c 000000b8 ffffff24
	global_load_u8 v185, v[38:39], off offset:-1               // 000000002580: ee04007c 000000b9 ffffff26
	global_load_u8 v186, v[131:132], off offset:-1             // 00000000258c: ee04007c 000000ba ffffff83
	global_load_u8 v187, v[133:134], off offset:-1             // 000000002598: ee04007c 000000bb ffffff85
	global_load_u8 v188, v[135:136], off offset:-1             // 0000000025a4: ee04007c 000000bc ffffff87
	global_load_u8 v189, v[137:138], off offset:-1             // 0000000025b0: ee04007c 000000bd ffffff89
	global_load_u8 v190, v[139:140], off offset:-1             // 0000000025bc: ee04007c 000000be ffffff8b
	global_load_u8 v191, v[16:17], off                         // 0000000025c8: ee04007c 000000bf 00000010
	global_load_u8 v192, v[18:19], off                         // 0000000025d4: ee04007c 000000c0 00000012
	global_load_u8 v193, v[20:21], off                         // 0000000025e0: ee04007c 000000c1 00000014
	global_load_u8 v194, v[22:23], off                         // 0000000025ec: ee04007c 000000c2 00000016
	global_load_u8 v195, v[24:25], off                         // 0000000025f8: ee04007c 000000c3 00000018
	global_load_u8 v196, v[26:27], off                         // 000000002604: ee04007c 000000c4 0000001a
	global_load_u8 v197, v[28:29], off                         // 000000002610: ee04007c 000000c5 0000001c
	global_load_u8 v198, v[30:31], off                         // 00000000261c: ee04007c 000000c6 0000001e
	global_load_u8 v199, v[34:35], off                         // 000000002628: ee04007c 000000c7 00000022
	global_load_u8 v200, v[36:37], off                         // 000000002634: ee04007c 000000c8 00000024
	global_load_u8 v201, v[38:39], off                         // 000000002640: ee04007c 000000c9 00000026
	global_load_u8 v202, v[131:132], off                       // 00000000264c: ee04007c 000000ca 00000083
	global_load_u8 v203, v[133:134], off                       // 000000002658: ee04007c 000000cb 00000085
	global_load_u8 v204, v[135:136], off                       // 000000002664: ee04007c 000000cc 00000087
	global_load_u8 v205, v[137:138], off                       // 000000002670: ee04007c 000000cd 00000089
	global_load_u8 v206, v[139:140], off                       // 00000000267c: ee04007c 000000ce 0000008b
	ds_load_2addr_b64 v[0:3], v74 offset1:2                    // 000000002688: d9dc0200 0000004a
	ds_load_2addr_b64 v[4:7], v12 offset1:2                    // 000000002690: d9dc0200 0400000c
	ds_load_2addr_b64 v[8:11], v13 offset1:2                   // 000000002698: d9dc0200 0800000d
	ds_load_2addr_b64 v[14:17], v75 offset1:2                  // 0000000026a0: d9dc0200 0e00004b
	ds_load_2addr_b64 v[155:158], v12 offset0:4 offset1:6      // 0000000026a8: d9dc0604 9b00000c
	ds_load_2addr_b64 v[159:162], v74 offset0:4 offset1:6      // 0000000026b0: d9dc0604 9f00004a
	ds_load_2addr_b64 v[163:166], v13 offset0:4 offset1:6      // 0000000026b8: d9dc0604 a300000d
	ds_load_2addr_b64 v[167:170], v75 offset0:4 offset1:6      // 0000000026c0: d9dc0604 a700004b
	s_wait_dscnt 0x6                                           // 0000000026c8: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[0:1], v[4:5], 0  // 0000000026cc: cc464083 1a020900
	s_wait_dscnt 0x5                                           // 0000000026d4: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[139:146], v[0:1], v[8:9], 0  // 0000000026d8: cc46408b 1a021100
	s_wait_dscnt 0x4                                           // 0000000026e0: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[147:154], v[14:15], v[4:5], 0// 0000000026e4: cc464093 1a02090e
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[14:15], v[8:9], 0  // 0000000026ec: cc464020 1a02110e
	v_wmma_f32_16x16x16_fp8_fp8 v[131:138], v[2:3], v[6:7], v[131:138]// 0000000026f4: cc464083 1e0e0d02
	v_wmma_f32_16x16x16_fp8_fp8 v[139:146], v[2:3], v[10:11], v[139:146]// 0000000026fc: cc46408b 1e2e1502
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002704: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[147:154], v[16:17], v[6:7], v[147:154]// 000000002708: cc464093 1e4e0d10
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[10:11], v[32:39]// 000000002710: cc464020 1c821510
	s_wait_dscnt 0x2                                           // 000000002718: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[159:160], v[155:156], 0// 00000000271c: cc464018 1a03379f
	s_wait_dscnt 0x1                                           // 000000002724: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[159:160], v[163:164], 0// 000000002728: cc464010 1a03479f
	s_wait_dscnt 0x0                                           // 000000002730: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[167:168], v[155:156], 0// 000000002734: cc464008 1a0337a7
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[167:168], v[163:164], 0// 00000000273c: cc464000 1a0347a7
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[161:162], v[157:158], v[24:31]// 000000002744: cc464018 1c633ba1
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[161:162], v[165:166], v[16:23]// 00000000274c: cc464010 1c434ba1
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002754: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[169:170], v[157:158], v[8:15]// 000000002758: cc464008 1c233ba9
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[169:170], v[165:166], v[0:7]// 000000002760: cc464000 1c034ba9
	s_wait_loadcnt 0x23                                        // 000000002768: bfc00023
	v_add_nc_u32_e32 v155, 0xffffff02, v171                    // 00000000276c: 4b3756ff ffffff02
	s_wait_loadcnt 0x22                                        // 000000002774: bfc00022
	v_add_nc_u32_e32 v156, 0xffffff02, v172                    // 000000002778: 4b3958ff ffffff02
	s_wait_loadcnt 0x21                                        // 000000002780: bfc00021
	v_cmp_eq_u32_e64 s4, 0xff, v173                            // 000000002784: d44a0004 02035aff 000000ff
	v_cmp_eq_u32_e64 s3, 0xff, v171                            // 000000002790: d44a0003 020356ff 000000ff
	s_wait_loadcnt 0x20                                        // 00000000279c: bfc00020
	v_cmp_eq_u32_e64 s5, 0xff, v174                            // 0000000027a0: d44a0005 02035cff 000000ff
	s_wait_loadcnt 0x1f                                        // 0000000027ac: bfc0001f
	v_cmp_eq_u32_e64 s6, 0xff, v175                            // 0000000027b0: d44a0006 02035eff 000000ff
	s_wait_loadcnt 0x1e                                        // 0000000027bc: bfc0001e
	v_cmp_eq_u32_e64 s7, 0xff, v176                            // 0000000027c0: d44a0007 020360ff 000000ff
	s_wait_loadcnt 0x1d                                        // 0000000027cc: bfc0001d
	v_add_nc_u32_e32 v157, 0xffffff02, v177                    // 0000000027d0: 4b3b62ff ffffff02
	s_wait_loadcnt 0x1c                                        // 0000000027d8: bfc0001c
	v_cmp_eq_u32_e64 s8, 0xff, v178                            // 0000000027dc: d44a0008 020364ff 000000ff
	s_wait_loadcnt 0x1b                                        // 0000000027e8: bfc0001b
	v_add_nc_u32_e32 v158, 0xffffff02, v179                    // 0000000027ec: 4b3d66ff ffffff02
	s_wait_loadcnt 0x1a                                        // 0000000027f4: bfc0001a
	v_cmp_eq_u32_e64 s9, 0xff, v180                            // 0000000027f8: d44a0009 020368ff 000000ff
	s_wait_loadcnt 0x19                                        // 000000002804: bfc00019
	v_cmp_eq_u32_e64 s10, 0xff, v181                           // 000000002808: d44a000a 02036aff 000000ff
	s_wait_loadcnt 0x18                                        // 000000002814: bfc00018
	v_cmp_eq_u32_e64 s11, 0xff, v182                           // 000000002818: d44a000b 02036cff 000000ff
	v_cmp_eq_u32_e64 s12, 0xff, v172                           // 000000002824: d44a000c 020358ff 000000ff
	s_wait_loadcnt 0x17                                        // 000000002830: bfc00017
	v_cmp_eq_u32_e64 s13, 0xff, v183                           // 000000002834: d44a000d 02036eff 000000ff
	s_wait_loadcnt 0x16                                        // 000000002840: bfc00016
	v_cmp_eq_u32_e64 s14, 0xff, v184                           // 000000002844: d44a000e 020370ff 000000ff
	s_wait_loadcnt 0x15                                        // 000000002850: bfc00015
	v_cmp_eq_u32_e64 s15, 0xff, v185                           // 000000002854: d44a000f 020372ff 000000ff
	s_wait_loadcnt 0x14                                        // 000000002860: bfc00014
	v_cmp_eq_u32_e64 s16, 0xff, v186                           // 000000002864: d44a0010 020374ff 000000ff
	s_wait_loadcnt 0x13                                        // 000000002870: bfc00013
	v_cmp_eq_u32_e64 s17, 0xff, v187                           // 000000002874: d44a0011 020376ff 000000ff
	s_wait_loadcnt 0x12                                        // 000000002880: bfc00012
	v_cmp_eq_u32_e64 s18, 0xff, v188                           // 000000002884: d44a0012 020378ff 000000ff
	v_cmp_eq_u32_e64 s22, 0xff, v177                           // 000000002890: d44a0016 020362ff 000000ff
	v_cmp_eq_u32_e64 s30, 0xff, v179                           // 00000000289c: d44a001e 020366ff 000000ff
	v_add_nc_u32_e32 v159, v155, v173                          // 0000000028a8: 4b3f5b9b
	v_add_nc_u32_e32 v160, v155, v174                          // 0000000028ac: 4b415d9b
	v_add_nc_u32_e32 v161, v155, v175                          // 0000000028b0: 4b435f9b
	v_add_nc_u32_e32 v162, v155, v176                          // 0000000028b4: 4b45619b
	v_add_nc_u32_e32 v163, v155, v178                          // 0000000028b8: 4b47659b
	v_add_nc_u32_e32 v164, v155, v180                          // 0000000028bc: 4b49699b
	v_add_nc_u32_e32 v165, v155, v181                          // 0000000028c0: 4b4b6b9b
	v_add_nc_u32_e32 v166, v155, v182                          // 0000000028c4: 4b4d6d9b
	v_add_nc_u32_e32 v167, v156, v173                          // 0000000028c8: 4b4f5b9c
	v_add_nc_u32_e32 v168, v156, v174                          // 0000000028cc: 4b515d9c
	v_add_nc_u32_e32 v169, v156, v175                          // 0000000028d0: 4b535f9c
	v_add_nc_u32_e32 v170, v156, v176                          // 0000000028d4: 4b55619c
	v_add_nc_u32_e32 v171, v156, v178                          // 0000000028d8: 4b57659c
	v_add_nc_u32_e32 v172, v156, v180                          // 0000000028dc: 4b59699c
	v_add_nc_u32_e32 v173, v156, v181                          // 0000000028e0: 4b5b6b9c
	v_add_nc_u32_e32 v174, v156, v182                          // 0000000028e4: 4b5d6d9c
	v_add_nc_u32_e32 v175, v155, v183                          // 0000000028e8: 4b5f6f9b
	v_add_nc_u32_e32 v176, v155, v184                          // 0000000028ec: 4b61719b
	v_add_nc_u32_e32 v177, v155, v185                          // 0000000028f0: 4b63739b
	v_add_nc_u32_e32 v178, v155, v186                          // 0000000028f4: 4b65759b
	v_add_nc_u32_e32 v179, v155, v187                          // 0000000028f8: 4b67779b
	v_add_nc_u32_e32 v180, v155, v188                          // 0000000028fc: 4b69799b
	s_wait_loadcnt 0x11                                        // 000000002900: bfc00011
	v_add_nc_u32_e32 v181, v155, v189                          // 000000002904: 4b6b7b9b
	s_wait_loadcnt 0x10                                        // 000000002908: bfc00010
	v_add_nc_u32_e32 v155, v155, v190                          // 00000000290c: 4b377d9b
	v_add_nc_u32_e32 v182, v156, v183                          // 000000002910: 4b6d6f9c
	v_add_nc_u32_e32 v183, v156, v184                          // 000000002914: 4b6f719c
	v_add_nc_u32_e32 v184, v156, v185                          // 000000002918: 4b71739c
	v_add_nc_u32_e32 v185, v156, v186                          // 00000000291c: 4b73759c
	v_add_nc_u32_e32 v186, v156, v187                          // 000000002920: 4b75779c
	v_add_nc_u32_e32 v187, v156, v188                          // 000000002924: 4b77799c
	v_add_nc_u32_e32 v188, v156, v189                          // 000000002928: 4b797b9c
	v_add_nc_u32_e32 v156, v156, v190                          // 00000000292c: 4b397d9c
	v_cmp_eq_u32_e64 s19, 0xff, v189                           // 000000002930: d44a0013 02037aff 000000ff
	v_cmp_eq_u32_e64 s20, 0xff, v190                           // 00000000293c: d44a0014 02037cff 000000ff
	s_wait_loadcnt 0xf                                         // 000000002948: bfc0000f
	v_cmp_eq_u32_e64 s21, 0xff, v191                           // 00000000294c: d44a0015 02037eff 000000ff
	s_wait_loadcnt 0xe                                         // 000000002958: bfc0000e
	v_cmp_eq_u32_e64 s23, 0xff, v192                           // 00000000295c: d44a0017 020380ff 000000ff
	s_wait_loadcnt 0xd                                         // 000000002968: bfc0000d
	v_cmp_eq_u32_e64 s24, 0xff, v193                           // 00000000296c: d44a0018 020382ff 000000ff
	s_wait_loadcnt 0xc                                         // 000000002978: bfc0000c
	v_cmp_eq_u32_e64 s25, 0xff, v194                           // 00000000297c: d44a0019 020384ff 000000ff
	s_wait_loadcnt 0xb                                         // 000000002988: bfc0000b
	v_cmp_eq_u32_e64 s26, 0xff, v195                           // 00000000298c: d44a001a 020386ff 000000ff
	s_wait_loadcnt 0xa                                         // 000000002998: bfc0000a
	v_cmp_eq_u32_e64 s27, 0xff, v196                           // 00000000299c: d44a001b 020388ff 000000ff
	s_wait_loadcnt 0x9                                         // 0000000029a8: bfc00009
	v_cmp_eq_u32_e64 s28, 0xff, v197                           // 0000000029ac: d44a001c 02038aff 000000ff
	s_wait_loadcnt 0x8                                         // 0000000029b8: bfc00008
	v_cmp_eq_u32_e64 s29, 0xff, v198                           // 0000000029bc: d44a001d 02038cff 000000ff
	s_wait_loadcnt 0x7                                         // 0000000029c8: bfc00007
	v_cmp_eq_u32_e64 s31, 0xff, v199                           // 0000000029cc: d44a001f 02038eff 000000ff
	s_wait_loadcnt 0x6                                         // 0000000029d8: bfc00006
	v_cmp_eq_u32_e64 s33, 0xff, v200                           // 0000000029dc: d44a0021 020390ff 000000ff
	s_wait_loadcnt 0x5                                         // 0000000029e8: bfc00005
	v_cmp_eq_u32_e64 s34, 0xff, v201                           // 0000000029ec: d44a0022 020392ff 000000ff
	s_wait_loadcnt 0x4                                         // 0000000029f8: bfc00004
	v_cmp_eq_u32_e64 s35, 0xff, v202                           // 0000000029fc: d44a0023 020394ff 000000ff
	s_wait_loadcnt 0x3                                         // 000000002a08: bfc00003
	v_cmp_eq_u32_e64 s36, 0xff, v203                           // 000000002a0c: d44a0024 020396ff 000000ff
	s_wait_loadcnt 0x2                                         // 000000002a18: bfc00002
	v_cmp_eq_u32_e64 s37, 0xff, v204                           // 000000002a1c: d44a0025 020398ff 000000ff
	s_wait_loadcnt 0x1                                         // 000000002a28: bfc00001
	v_cmp_eq_u32_e64 s38, 0xff, v205                           // 000000002a2c: d44a0026 02039aff 000000ff
	v_add_nc_u32_e32 v189, v157, v191                          // 000000002a38: 4b7b7f9d
	v_add_nc_u32_e32 v190, v157, v192                          // 000000002a3c: 4b7d819d
	v_add_nc_u32_e32 v207, v157, v193                          // 000000002a40: 4b9f839d
	v_add_nc_u32_e32 v208, v157, v194                          // 000000002a44: 4ba1859d
	v_add_nc_u32_e32 v209, v157, v195                          // 000000002a48: 4ba3879d
	v_add_nc_u32_e32 v210, v157, v196                          // 000000002a4c: 4ba5899d
	v_add_nc_u32_e32 v211, v157, v197                          // 000000002a50: 4ba78b9d
	v_add_nc_u32_e32 v212, v157, v198                          // 000000002a54: 4ba98d9d
	v_add_nc_u32_e32 v191, v158, v191                          // 000000002a58: 4b7f7f9e
	v_add_nc_u32_e32 v192, v158, v192                          // 000000002a5c: 4b81819e
	v_add_nc_u32_e32 v193, v158, v193                          // 000000002a60: 4b83839e
	v_add_nc_u32_e32 v194, v158, v194                          // 000000002a64: 4b85859e
	v_add_nc_u32_e32 v195, v158, v195                          // 000000002a68: 4b87879e
	v_add_nc_u32_e32 v196, v158, v196                          // 000000002a6c: 4b89899e
	v_add_nc_u32_e32 v197, v158, v197                          // 000000002a70: 4b8b8b9e
	v_add_nc_u32_e32 v198, v158, v198                          // 000000002a74: 4b8d8d9e
	v_add_nc_u32_e32 v213, v157, v199                          // 000000002a78: 4bab8f9d
	v_add_nc_u32_e32 v214, v157, v200                          // 000000002a7c: 4bad919d
	v_add_nc_u32_e32 v215, v157, v201                          // 000000002a80: 4baf939d
	v_add_nc_u32_e32 v216, v157, v202                          // 000000002a84: 4bb1959d
	v_add_nc_u32_e32 v217, v157, v203                          // 000000002a88: 4bb3979d
	v_add_nc_u32_e32 v218, v157, v204                          // 000000002a8c: 4bb5999d
	v_add_nc_u32_e32 v219, v157, v205                          // 000000002a90: 4bb79b9d
	s_wait_loadcnt 0x0                                         // 000000002a94: bfc00000
	v_add_nc_u32_e32 v157, v157, v206                          // 000000002a98: 4b3b9d9d
	v_add_nc_u32_e32 v199, v158, v199                          // 000000002a9c: 4b8f8f9e
	v_add_nc_u32_e32 v200, v158, v200                          // 000000002aa0: 4b91919e
	v_add_nc_u32_e32 v201, v158, v201                          // 000000002aa4: 4b93939e
	v_add_nc_u32_e32 v202, v158, v202                          // 000000002aa8: 4b95959e
	v_add_nc_u32_e32 v203, v158, v203                          // 000000002aac: 4b97979e
	v_add_nc_u32_e32 v204, v158, v204                          // 000000002ab0: 4b99999e
	v_add_nc_u32_e32 v205, v158, v205                          // 000000002ab4: 4b9b9b9e
	v_add_nc_u32_e32 v158, v158, v206                          // 000000002ab8: 4b3d9d9e
	v_ldexp_f32 v131, v131, v159                               // 000000002abc: d71c0083 02033f83
	v_ldexp_f32 v132, v132, v160                               // 000000002ac4: d71c0084 02034184
	v_ldexp_f32 v133, v133, v161                               // 000000002acc: d71c0085 02034385
	v_ldexp_f32 v134, v134, v162                               // 000000002ad4: d71c0086 02034586
	v_ldexp_f32 v135, v135, v163                               // 000000002adc: d71c0087 02034787
	v_ldexp_f32 v136, v136, v164                               // 000000002ae4: d71c0088 02034988
	v_ldexp_f32 v137, v137, v165                               // 000000002aec: d71c0089 02034b89
	v_ldexp_f32 v138, v138, v166                               // 000000002af4: d71c008a 02034d8a
	v_ldexp_f32 v139, v139, v167                               // 000000002afc: d71c008b 02034f8b
	v_ldexp_f32 v140, v140, v168                               // 000000002b04: d71c008c 0203518c
	v_ldexp_f32 v141, v141, v169                               // 000000002b0c: d71c008d 0203538d
	v_ldexp_f32 v142, v142, v170                               // 000000002b14: d71c008e 0203558e
	v_ldexp_f32 v143, v143, v171                               // 000000002b1c: d71c008f 0203578f
	v_ldexp_f32 v144, v144, v172                               // 000000002b24: d71c0090 02035990
	v_ldexp_f32 v145, v145, v173                               // 000000002b2c: d71c0091 02035b91
	v_ldexp_f32 v146, v146, v174                               // 000000002b34: d71c0092 02035d92
	v_ldexp_f32 v147, v147, v175                               // 000000002b3c: d71c0093 02035f93
	v_ldexp_f32 v148, v148, v176                               // 000000002b44: d71c0094 02036194
	v_ldexp_f32 v149, v149, v177                               // 000000002b4c: d71c0095 02036395
	v_ldexp_f32 v150, v150, v178                               // 000000002b54: d71c0096 02036596
	v_ldexp_f32 v151, v151, v179                               // 000000002b5c: d71c0097 02036797
	v_ldexp_f32 v152, v152, v180                               // 000000002b64: d71c0098 02036998
	v_ldexp_f32 v153, v153, v181                               // 000000002b6c: d71c0099 02036b99
	v_ldexp_f32 v154, v154, v155                               // 000000002b74: d71c009a 0203379a
	v_ldexp_f32 v32, v32, v182                                 // 000000002b7c: d71c0020 02036d20
	v_ldexp_f32 v33, v33, v183                                 // 000000002b84: d71c0021 02036f21
	v_ldexp_f32 v34, v34, v184                                 // 000000002b8c: d71c0022 02037122
	v_ldexp_f32 v35, v35, v185                                 // 000000002b94: d71c0023 02037323
	v_ldexp_f32 v36, v36, v186                                 // 000000002b9c: d71c0024 02037524
	v_ldexp_f32 v37, v37, v187                                 // 000000002ba4: d71c0025 02037725
	v_ldexp_f32 v38, v38, v188                                 // 000000002bac: d71c0026 02037926
	v_ldexp_f32 v39, v39, v156                                 // 000000002bb4: d71c0027 02033927
	v_cmp_eq_u32_e64 s39, 0xff, v206                           // 000000002bbc: d44a0027 02039cff 000000ff
	s_or_b32 s40, s4, s3                                       // 000000002bc8: 8c280304
	s_or_b32 s41, s3, s5                                       // 000000002bcc: 8c290503
	s_or_b32 s42, s3, s6                                       // 000000002bd0: 8c2a0603
	s_or_b32 s43, s3, s7                                       // 000000002bd4: 8c2b0703
	s_or_b32 s58, s3, s8                                       // 000000002bd8: 8c3a0803
	s_or_b32 s59, s3, s9                                       // 000000002bdc: 8c3b0903
	s_or_b32 s60, s3, s10                                      // 000000002be0: 8c3c0a03
	s_or_b32 s61, s3, s11                                      // 000000002be4: 8c3d0b03
	s_or_b32 s4, s4, s12                                       // 000000002be8: 8c040c04
	s_or_b32 s5, s5, s12                                       // 000000002bec: 8c050c05
	s_or_b32 s6, s6, s12                                       // 000000002bf0: 8c060c06
	s_or_b32 s7, s7, s12                                       // 000000002bf4: 8c070c07
	s_or_b32 s8, s8, s12                                       // 000000002bf8: 8c080c08
	s_or_b32 s9, s9, s12                                       // 000000002bfc: 8c090c09
	s_or_b32 s10, s10, s12                                     // 000000002c00: 8c0a0c0a
	s_or_b32 s11, s11, s12                                     // 000000002c04: 8c0b0c0b
	s_or_b32 s62, s3, s13                                      // 000000002c08: 8c3e0d03
	s_or_b32 s63, s3, s14                                      // 000000002c0c: 8c3f0e03
	s_or_b32 s64, s3, s15                                      // 000000002c10: 8c400f03
	s_or_b32 s65, s3, s16                                      // 000000002c14: 8c411003
	s_or_b32 s66, s3, s17                                      // 000000002c18: 8c421103
	s_or_b32 s67, s3, s18                                      // 000000002c1c: 8c431203
	s_or_b32 s68, s3, s19                                      // 000000002c20: 8c441303
	s_or_b32 s3, s3, s20                                       // 000000002c24: 8c031403
	s_or_b32 s13, s12, s13                                     // 000000002c28: 8c0d0d0c
	s_or_b32 s14, s12, s14                                     // 000000002c2c: 8c0e0e0c
	s_or_b32 s15, s12, s15                                     // 000000002c30: 8c0f0f0c
	s_or_b32 s16, s12, s16                                     // 000000002c34: 8c10100c
	s_or_b32 s17, s12, s17                                     // 000000002c38: 8c11110c
	s_or_b32 s18, s12, s18                                     // 000000002c3c: 8c12120c
	s_or_b32 s19, s12, s19                                     // 000000002c40: 8c13130c
	s_or_b32 s12, s12, s20                                     // 000000002c44: 8c0c140c
	v_ldexp_f32 v24, v24, v189                                 // 000000002c48: d71c0018 02037b18
	v_ldexp_f32 v25, v25, v190                                 // 000000002c50: d71c0019 02037d19
	v_ldexp_f32 v26, v26, v207                                 // 000000002c58: d71c001a 02039f1a
	v_ldexp_f32 v27, v27, v208                                 // 000000002c60: d71c001b 0203a11b
	v_ldexp_f32 v28, v28, v209                                 // 000000002c68: d71c001c 0203a31c
	v_ldexp_f32 v29, v29, v210                                 // 000000002c70: d71c001d 0203a51d
	v_ldexp_f32 v30, v30, v211                                 // 000000002c78: d71c001e 0203a71e
	v_ldexp_f32 v31, v31, v212                                 // 000000002c80: d71c001f 0203a91f
	v_ldexp_f32 v16, v16, v191                                 // 000000002c88: d71c0010 02037f10
	v_ldexp_f32 v17, v17, v192                                 // 000000002c90: d71c0011 02038111
	v_ldexp_f32 v18, v18, v193                                 // 000000002c98: d71c0012 02038312
	v_ldexp_f32 v19, v19, v194                                 // 000000002ca0: d71c0013 02038513
	v_ldexp_f32 v20, v20, v195                                 // 000000002ca8: d71c0014 02038714
	v_ldexp_f32 v21, v21, v196                                 // 000000002cb0: d71c0015 02038915
	v_ldexp_f32 v22, v22, v197                                 // 000000002cb8: d71c0016 02038b16
	v_ldexp_f32 v23, v23, v198                                 // 000000002cc0: d71c0017 02038d17
	v_ldexp_f32 v8, v8, v213                                   // 000000002cc8: d71c0008 0203ab08
	v_ldexp_f32 v9, v9, v214                                   // 000000002cd0: d71c0009 0203ad09
	v_ldexp_f32 v10, v10, v215                                 // 000000002cd8: d71c000a 0203af0a
	v_ldexp_f32 v11, v11, v216                                 // 000000002ce0: d71c000b 0203b10b
	v_ldexp_f32 v12, v12, v217                                 // 000000002ce8: d71c000c 0203b30c
	v_ldexp_f32 v13, v13, v218                                 // 000000002cf0: d71c000d 0203b50d
	v_ldexp_f32 v14, v14, v219                                 // 000000002cf8: d71c000e 0203b70e
	v_ldexp_f32 v15, v15, v157                                 // 000000002d00: d71c000f 02033b0f
	v_ldexp_f32 v0, v0, v199                                   // 000000002d08: d71c0000 02038f00
	v_ldexp_f32 v1, v1, v200                                   // 000000002d10: d71c0001 02039101
	v_ldexp_f32 v2, v2, v201                                   // 000000002d18: d71c0002 02039302
	v_ldexp_f32 v3, v3, v202                                   // 000000002d20: d71c0003 02039503
	v_ldexp_f32 v4, v4, v203                                   // 000000002d28: d71c0004 02039704
	v_ldexp_f32 v5, v5, v204                                   // 000000002d30: d71c0005 02039905
	v_ldexp_f32 v6, v6, v205                                   // 000000002d38: d71c0006 02039b06
	v_ldexp_f32 v7, v7, v158                                   // 000000002d40: d71c0007 02033d07
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d48: bf88ff9e
	v_cndmask_b32_e64 v131, v131, 0x7fc00000, s40              // 000000002d4c: d5010083 00a1ff83 7fc00000
	v_cndmask_b32_e64 v132, v132, 0x7fc00000, s41              // 000000002d58: d5010084 00a5ff84 7fc00000
	v_cndmask_b32_e64 v133, v133, 0x7fc00000, s42              // 000000002d64: d5010085 00a9ff85 7fc00000
	v_cndmask_b32_e64 v134, v134, 0x7fc00000, s43              // 000000002d70: d5010086 00adff86 7fc00000
	v_cndmask_b32_e64 v135, v135, 0x7fc00000, s58              // 000000002d7c: d5010087 00e9ff87 7fc00000
	v_cndmask_b32_e64 v136, v136, 0x7fc00000, s59              // 000000002d88: d5010088 00edff88 7fc00000
	v_cndmask_b32_e64 v137, v137, 0x7fc00000, s60              // 000000002d94: d5010089 00f1ff89 7fc00000
	v_cndmask_b32_e64 v138, v138, 0x7fc00000, s61              // 000000002da0: d501008a 00f5ff8a 7fc00000
	v_cndmask_b32_e64 v139, v139, 0x7fc00000, s4               // 000000002dac: d501008b 0011ff8b 7fc00000
	v_cndmask_b32_e64 v140, v140, 0x7fc00000, s5               // 000000002db8: d501008c 0015ff8c 7fc00000
	v_cndmask_b32_e64 v141, v141, 0x7fc00000, s6               // 000000002dc4: d501008d 0019ff8d 7fc00000
	v_cndmask_b32_e64 v142, v142, 0x7fc00000, s7               // 000000002dd0: d501008e 001dff8e 7fc00000
	v_cndmask_b32_e64 v143, v143, 0x7fc00000, s8               // 000000002ddc: d501008f 0021ff8f 7fc00000
	v_cndmask_b32_e64 v144, v144, 0x7fc00000, s9               // 000000002de8: d5010090 0025ff90 7fc00000
	v_cndmask_b32_e64 v145, v145, 0x7fc00000, s10              // 000000002df4: d5010091 0029ff91 7fc00000
	v_cndmask_b32_e64 v146, v146, 0x7fc00000, s11              // 000000002e00: d5010092 002dff92 7fc00000
	v_cndmask_b32_e64 v147, v147, 0x7fc00000, s62              // 000000002e0c: d5010093 00f9ff93 7fc00000
	v_cndmask_b32_e64 v148, v148, 0x7fc00000, s63              // 000000002e18: d5010094 00fdff94 7fc00000
	v_cndmask_b32_e64 v149, v149, 0x7fc00000, s64              // 000000002e24: d5010095 0101ff95 7fc00000
	v_cndmask_b32_e64 v150, v150, 0x7fc00000, s65              // 000000002e30: d5010096 0105ff96 7fc00000
	v_cndmask_b32_e64 v151, v151, 0x7fc00000, s66              // 000000002e3c: d5010097 0109ff97 7fc00000
	v_cndmask_b32_e64 v152, v152, 0x7fc00000, s67              // 000000002e48: d5010098 010dff98 7fc00000
	v_cndmask_b32_e64 v153, v153, 0x7fc00000, s68              // 000000002e54: d5010099 0111ff99 7fc00000
	v_cndmask_b32_e64 v154, v154, 0x7fc00000, s3               // 000000002e60: d501009a 000dff9a 7fc00000
	v_cndmask_b32_e64 v32, v32, 0x7fc00000, s13                // 000000002e6c: d5010020 0035ff20 7fc00000
	v_cndmask_b32_e64 v33, v33, 0x7fc00000, s14                // 000000002e78: d5010021 0039ff21 7fc00000
	v_cndmask_b32_e64 v34, v34, 0x7fc00000, s15                // 000000002e84: d5010022 003dff22 7fc00000
	v_cndmask_b32_e64 v35, v35, 0x7fc00000, s16                // 000000002e90: d5010023 0041ff23 7fc00000
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s17                // 000000002e9c: d5010024 0045ff24 7fc00000
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s18                // 000000002ea8: d5010025 0049ff25 7fc00000
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s19                // 000000002eb4: d5010026 004dff26 7fc00000
	v_cndmask_b32_e64 v39, v39, 0x7fc00000, s12                // 000000002ec0: d5010027 0031ff27 7fc00000
	s_or_b32 s20, s22, s23                                     // 000000002ecc: 8c141716
	s_or_b32 s69, s22, s24                                     // 000000002ed0: 8c451816
	s_or_b32 s70, s22, s25                                     // 000000002ed4: 8c461916
	s_or_b32 s71, s22, s26                                     // 000000002ed8: 8c471a16
	s_or_b32 s72, s22, s27                                     // 000000002edc: 8c481b16
	s_or_b32 s73, s22, s28                                     // 000000002ee0: 8c491c16
	s_or_b32 s74, s22, s29                                     // 000000002ee4: 8c4a1d16
	s_or_b32 s75, s21, s30                                     // 000000002ee8: 8c4b1e15
	s_or_b32 s23, s23, s30                                     // 000000002eec: 8c171e17
	s_or_b32 s24, s24, s30                                     // 000000002ef0: 8c181e18
	s_or_b32 s25, s25, s30                                     // 000000002ef4: 8c191e19
	s_or_b32 s26, s26, s30                                     // 000000002ef8: 8c1a1e1a
	s_or_b32 s27, s27, s30                                     // 000000002efc: 8c1b1e1b
	s_or_b32 s28, s28, s30                                     // 000000002f00: 8c1c1e1c
	s_or_b32 s29, s29, s30                                     // 000000002f04: 8c1d1e1d
	s_or_b32 s76, s22, s31                                     // 000000002f08: 8c4c1f16
	s_or_b32 s77, s22, s33                                     // 000000002f0c: 8c4d2116
	s_or_b32 s78, s22, s34                                     // 000000002f10: 8c4e2216
	s_or_b32 s79, s22, s35                                     // 000000002f14: 8c4f2316
	s_or_b32 s80, s22, s36                                     // 000000002f18: 8c502416
	s_or_b32 s81, s22, s37                                     // 000000002f1c: 8c512516
	s_or_b32 s82, s22, s38                                     // 000000002f20: 8c522616
	s_or_b32 s83, s22, s39                                     // 000000002f24: 8c532716
	s_or_b32 s21, s21, s22                                     // 000000002f28: 8c151615
	s_or_b32 s22, s30, s31                                     // 000000002f2c: 8c161f1e
	s_or_b32 s31, s30, s33                                     // 000000002f30: 8c1f211e
	s_or_b32 s33, s30, s34                                     // 000000002f34: 8c21221e
	s_or_b32 s34, s30, s35                                     // 000000002f38: 8c22231e
	s_or_b32 s35, s30, s36                                     // 000000002f3c: 8c23241e
	s_or_b32 s36, s30, s37                                     // 000000002f40: 8c24251e
	s_or_b32 s37, s30, s38                                     // 000000002f44: 8c25261e
	s_or_b32 s30, s30, s39                                     // 000000002f48: 8c1e271e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f4c: bf88ff9e
	v_cndmask_b32_e64 v25, v25, 0x7fc00000, s20                // 000000002f50: d5010019 0051ff19 7fc00000
	v_cndmask_b32_e64 v26, v26, 0x7fc00000, s69                // 000000002f5c: d501001a 0115ff1a 7fc00000
	v_cndmask_b32_e64 v27, v27, 0x7fc00000, s70                // 000000002f68: d501001b 0119ff1b 7fc00000
	v_cndmask_b32_e64 v28, v28, 0x7fc00000, s71                // 000000002f74: d501001c 011dff1c 7fc00000
	v_cndmask_b32_e64 v29, v29, 0x7fc00000, s72                // 000000002f80: d501001d 0121ff1d 7fc00000
	v_cndmask_b32_e64 v30, v30, 0x7fc00000, s73                // 000000002f8c: d501001e 0125ff1e 7fc00000
	v_cndmask_b32_e64 v31, v31, 0x7fc00000, s74                // 000000002f98: d501001f 0129ff1f 7fc00000
	v_cndmask_b32_e64 v16, v16, 0x7fc00000, s75                // 000000002fa4: d5010010 012dff10 7fc00000
	v_cndmask_b32_e64 v17, v17, 0x7fc00000, s23                // 000000002fb0: d5010011 005dff11 7fc00000
	v_cndmask_b32_e64 v18, v18, 0x7fc00000, s24                // 000000002fbc: d5010012 0061ff12 7fc00000
	v_cndmask_b32_e64 v19, v19, 0x7fc00000, s25                // 000000002fc8: d5010013 0065ff13 7fc00000
	v_cndmask_b32_e64 v20, v20, 0x7fc00000, s26                // 000000002fd4: d5010014 0069ff14 7fc00000
	v_cndmask_b32_e64 v21, v21, 0x7fc00000, s27                // 000000002fe0: d5010015 006dff15 7fc00000
	v_cndmask_b32_e64 v22, v22, 0x7fc00000, s28                // 000000002fec: d5010016 0071ff16 7fc00000
	v_cndmask_b32_e64 v23, v23, 0x7fc00000, s29                // 000000002ff8: d5010017 0075ff17 7fc00000
	v_cndmask_b32_e64 v8, v8, 0x7fc00000, s76                  // 000000003004: d5010008 0131ff08 7fc00000
	v_cndmask_b32_e64 v9, v9, 0x7fc00000, s77                  // 000000003010: d5010009 0135ff09 7fc00000
	v_cndmask_b32_e64 v10, v10, 0x7fc00000, s78                // 00000000301c: d501000a 0139ff0a 7fc00000
	v_cndmask_b32_e64 v11, v11, 0x7fc00000, s79                // 000000003028: d501000b 013dff0b 7fc00000
	v_cndmask_b32_e64 v12, v12, 0x7fc00000, s80                // 000000003034: d501000c 0141ff0c 7fc00000
	v_cndmask_b32_e64 v13, v13, 0x7fc00000, s81                // 000000003040: d501000d 0145ff0d 7fc00000
	v_cndmask_b32_e64 v14, v14, 0x7fc00000, s82                // 00000000304c: d501000e 0149ff0e 7fc00000
	v_cndmask_b32_e64 v15, v15, 0x7fc00000, s83                // 000000003058: d501000f 014dff0f 7fc00000
	v_cndmask_b32_e64 v24, v24, 0x7fc00000, s21                // 000000003064: d5010018 0055ff18 7fc00000
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s22                  // 000000003070: d5010000 0059ff00 7fc00000
	v_cndmask_b32_e64 v1, v1, 0x7fc00000, s31                  // 00000000307c: d5010001 007dff01 7fc00000
	v_cndmask_b32_e64 v2, v2, 0x7fc00000, s33                  // 000000003088: d5010002 0085ff02 7fc00000
	v_cndmask_b32_e64 v3, v3, 0x7fc00000, s34                  // 000000003094: d5010003 0089ff03 7fc00000
	v_cndmask_b32_e64 v4, v4, 0x7fc00000, s35                  // 0000000030a0: d5010004 008dff04 7fc00000
	v_cndmask_b32_e64 v5, v5, 0x7fc00000, s36                  // 0000000030ac: d5010005 0091ff05 7fc00000
	v_cndmask_b32_e64 v6, v6, 0x7fc00000, s37                  // 0000000030b8: d5010006 0095ff06 7fc00000
	v_cndmask_b32_e64 v7, v7, 0x7fc00000, s30                  // 0000000030c4: d5010007 0079ff07 7fc00000
	v_dual_add_f32 v47, v47, v131 :: v_dual_add_f32 v118, v118, v132// 0000000030d0: c909072f 2f770976
	v_add_f32_e32 v109, v109, v133                             // 0000000030d8: 06db0b6d
	v_add_f32_e32 v101, v101, v134                             // 0000000030dc: 06cb0d65
	v_dual_add_f32 v91, v91, v135 :: v_dual_add_f32 v90, v90, v136// 0000000030e0: c9090f5b 5b5b115a
	v_dual_add_f32 v89, v89, v137 :: v_dual_add_f32 v88, v88, v138// 0000000030e8: c9091359 59591558
	v_dual_add_f32 v70, v70, v139 :: v_dual_add_f32 v69, v69, v140// 0000000030f0: c9091746 46451945
	v_dual_add_f32 v68, v68, v141 :: v_dual_add_f32 v67, v67, v142// 0000000030f8: c9091b44 44431d43
	v_dual_add_f32 v66, v66, v143 :: v_dual_add_f32 v65, v65, v144// 000000003100: c9091f42 42412141
	v_dual_add_f32 v64, v64, v145 :: v_dual_add_f32 v63, v63, v146// 000000003108: c9092340 403f253f
	v_dual_add_f32 v87, v87, v147 :: v_dual_add_f32 v86, v86, v148// 000000003110: c9092757 57572956
	v_dual_add_f32 v83, v83, v149 :: v_dual_add_f32 v82, v82, v150// 000000003118: c9092b53 53532d52
	v_dual_add_f32 v81, v81, v151 :: v_dual_add_f32 v78, v78, v152// 000000003120: c9092f51 514f314e
	v_add_f32_e32 v73, v73, v153                               // 000000003128: 06933349
	v_dual_add_f32 v71, v71, v154 :: v_dual_add_f32 v32, v62, v32// 00000000312c: c9093547 4720413e
	v_dual_add_f32 v33, v61, v33 :: v_dual_add_f32 v34, v60, v34// 000000003134: c908433d 2122453c
	v_dual_add_f32 v35, v59, v35 :: v_dual_add_f32 v36, v58, v36// 00000000313c: c908473b 2324493a
	v_dual_add_f32 v37, v57, v37 :: v_dual_add_f32 v38, v56, v38// 000000003144: c9084b39 25264d38
	v_add_f32_e32 v39, v46, v39                                // 00000000314c: 064e4f2e
	v_dual_add_f32 v118, v118, v25 :: v_dual_add_f32 v109, v109, v26// 000000003150: c9083376 766c356d
	v_add_f32_e32 v101, v101, v27                              // 000000003158: 06ca3765
	v_dual_add_f32 v91, v91, v28 :: v_dual_add_f32 v90, v90, v29// 00000000315c: c908395b 5b5a3b5a
	v_dual_add_f32 v89, v89, v30 :: v_dual_add_f32 v88, v88, v31// 000000003164: c9083d59 59583f58
	v_dual_add_f32 v70, v70, v16 :: v_dual_add_f32 v69, v69, v17// 00000000316c: c9082146 46442345
	v_dual_add_f32 v68, v68, v18 :: v_dual_add_f32 v67, v67, v19// 000000003174: c9082544 44422743
	v_dual_add_f32 v66, v66, v20 :: v_dual_add_f32 v65, v65, v21// 00000000317c: c9082942 42402b41
	v_dual_add_f32 v64, v64, v22 :: v_dual_add_f32 v63, v63, v23// 000000003184: c9082d40 403e2f3f
	v_dual_add_f32 v87, v87, v8 :: v_dual_add_f32 v86, v86, v9 // 00000000318c: c9081157 57561356
	v_dual_add_f32 v83, v83, v10 :: v_dual_add_f32 v82, v82, v11// 000000003194: c9081553 53521752
	v_dual_add_f32 v81, v81, v12 :: v_dual_add_f32 v78, v78, v13// 00000000319c: c9081951 514e1b4e
	v_add_f32_e32 v73, v73, v14                                // 0000000031a4: 06921d49
	v_add_f32_e32 v71, v71, v15                                // 0000000031a8: 068e1f47
	v_add_f32_e32 v47, v47, v24                                // 0000000031ac: 065e312f
	v_dual_add_f32 v62, v32, v0 :: v_dual_add_f32 v61, v33, v1 // 0000000031b0: c9080120 3e3c0321
	v_dual_add_f32 v60, v34, v2 :: v_dual_add_f32 v59, v35, v3 // 0000000031b8: c9080522 3c3a0723
	v_dual_add_f32 v58, v36, v4 :: v_dual_add_f32 v57, v37, v5 // 0000000031c0: c9080924 3a380b25
	v_add_f32_e32 v56, v38, v6                                 // 0000000031c8: 06700d26
	v_add_f32_e32 v46, v39, v7                                 // 0000000031cc: 065c0f27
	s_cmp_lg_u64 s[54:55], s[56:57]                            // 0000000031d0: bf113836
	s_cbranch_scc1 64577                                       // 0000000031d4: bfa2fc41 <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x7dc>
	s_load_b64 s[18:19], s[0:1], 0xa8                          // 0000000031d8: f4002480 f80000a8
	v_mul_lo_u32 v4, s47, v44                                  // 0000000031e0: d72c0004 0202582f
	v_mul_lo_u32 v5, s46, v45                                  // 0000000031e8: d72c0005 02025a2e
	v_mad_co_u64_u32 v[2:3], null, s46, v44, 0                 // 0000000031f0: d6fe7c02 0202582e
	v_sub_co_u32 v0, s0, s44, v44                              // 0000000031f8: d7010000 0202582c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003200: bf870191
	v_sub_co_ci_u32_e64 v1, null, s45, v45, s0                 // 000000003204: d5217c01 00025a2d
	v_add3_u32 v3, v3, v5, v4                                  // 00000000320c: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003214: bf8701a2
	v_cmp_lt_i64_e64 s15, 0, v[0:1]                            // 000000003218: d451000f 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[40:41]                      // 000000003220: 3e085081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003224: 3e040481
	s_and_b32 s0, s15, s2                                      // 000000003228: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000322c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003230: be812000
	s_cbranch_execz 28                                         // 000000003234: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x17a8>
	v_bfe_u32 v6, v47, 16, 1                                   // 000000003238: d6100006 0205212f
	s_wait_kmcnt 0x0                                           // 000000003240: bfc70000
	v_add_co_u32 v7, s0, s18, v2                               // 000000003244: d7000007 02020412
	s_wait_alu depctr_va_sdst(0)                               // 00000000324c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s0                  // 000000003250: d5207c08 00020613
	v_add3_u32 v9, v6, v47, 0x7fff                             // 000000003258: d6550009 03fe5f06 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003264: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 000000003268: d7000006 02020907
	v_or_b32_e32 v10, 0x400000, v47                            // 000000003270: 38145eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003278: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v5, s0                   // 00000000327c: d5207c07 00020b08
	v_cmp_u_f32_e64 s0, v47, v47                               // 000000003284: d4180000 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 00000000328c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003290: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003294: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 00000000329c: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032ac: 8c7e017e
	v_add_co_u32 v6, s0, s46, v40                              // 0000000032b0: d7000006 0202502e
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s47, v41, s0                 // 0000000032bc: d5207c07 0002522f
	v_cmp_lt_i64_e64 s16, 1, v[0:1]                            // 0000000032c4: d4510010 02020081
	s_delay_alu instid0(valu_dep_2)                            // 0000000032cc: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 0000000032d0: 3e0c0c81
	s_and_b32 s0, s16, s2                                      // 0000000032d4: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032d8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032dc: be812000
	s_cbranch_execz 28                                         // 0000000032e0: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1854>
	v_bfe_u32 v8, v118, 16, 1                                  // 0000000032e4: d6100008 02052176
	s_wait_kmcnt 0x0                                           // 0000000032ec: bfc70000
	v_add_co_u32 v9, s0, s18, v2                               // 0000000032f0: d7000009 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000032f8: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s19, v3, s0                 // 0000000032fc: d5207c0a 00020613
	v_add3_u32 v11, v8, v118, 0x7fff                           // 000000003304: d655000b 03feed08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003310: bf870003
	v_add_co_u32 v8, s0, v9, v6                                // 000000003314: d7000008 02020d09
	v_or_b32_e32 v12, 0x400000, v118                           // 00000000331c: 3818ecff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003324: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v7, s0                  // 000000003328: d5207c09 00020f0a
	v_cmp_u_f32_e64 s0, v118, v118                             // 000000003330: d4180000 0202ed76
	s_wait_alu depctr_va_sdst(0)                               // 000000003338: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000333c: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000003340: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 000000003348: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000003354: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003358: 8c7e017e
	v_add_co_u32 v8, s0, s50, v40                              // 00000000335c: d7000008 02025032
	s_wait_alu depctr_va_sdst(0)                               // 000000003364: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s51, v41, s0                 // 000000003368: d5207c09 00025233
	v_cmp_lt_i64_e64 s14, 2, v[0:1]                            // 000000003370: d451000e 02020082
	s_delay_alu instid0(valu_dep_2)                            // 000000003378: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 00000000337c: 3e101081
	s_and_b32 s0, s14, s2                                      // 000000003380: 8b00020e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003384: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003388: be812000
	s_cbranch_execz 28                                         // 00000000338c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1900>
	v_bfe_u32 v10, v109, 16, 1                                 // 000000003390: d610000a 0205216d
	s_wait_kmcnt 0x0                                           // 000000003398: bfc70000
	v_add_co_u32 v11, s0, s18, v2                              // 00000000339c: d700000b 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a4: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s19, v3, s0                 // 0000000033a8: d5207c0c 00020613
	v_add3_u32 v13, v10, v109, 0x7fff                          // 0000000033b0: d655000d 03fedb0a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000033bc: bf870003
	v_add_co_u32 v10, s0, v11, v8                              // 0000000033c0: d700000a 0202110b
	v_or_b32_e32 v14, 0x400000, v109                           // 0000000033c8: 381cdaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d0: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s0                 // 0000000033d4: d5207c0b 0002130c
	v_cmp_u_f32_e64 s0, v109, v109                             // 0000000033dc: d4180000 0202db6d
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033e8: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 0000000033ec: d501000c 00021d0d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 0000000033f4: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003400: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003404: 8c7e017e
	s_mul_u64 s[28:29], s[46:47], 3                            // 000000003408: aa9c832e
	v_cmp_lt_i64_e64 s13, 3, v[0:1]                            // 00000000340c: d451000d 02020083
	s_wait_alu depctr_sa_sdst(0)                               // 000000003414: bf88ff9e
	v_add_co_u32 v10, s0, s28, v40                             // 000000003418: d700000a 0202501c
	s_wait_alu depctr_va_sdst(0)                               // 000000003420: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s29, v41, s0                // 000000003424: d5207c0b 0002521d
	s_and_b32 s0, s13, s2                                      // 00000000342c: 8b00020d
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000003430: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003434: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003438: be812000
	s_cbranch_execz 28                                         // 00000000343c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x19b0>
	v_bfe_u32 v12, v101, 16, 1                                 // 000000003440: d610000c 02052165
	s_wait_kmcnt 0x0                                           // 000000003448: bfc70000
	v_add_co_u32 v13, s0, s18, v2                              // 00000000344c: d700000d 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003454: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s19, v3, s0                 // 000000003458: d5207c0e 00020613
	v_add3_u32 v15, v12, v101, 0x7fff                          // 000000003460: d655000f 03fecb0c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000346c: bf870003
	v_add_co_u32 v12, s0, v13, v10                             // 000000003470: d700000c 0202150d
	v_or_b32_e32 v16, 0x400000, v101                           // 000000003478: 3820caff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003480: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v11, s0                // 000000003484: d5207c0d 0002170e
	v_cmp_u_f32_e64 s0, v101, v101                             // 00000000348c: d4180000 0202cb65
	s_wait_alu depctr_va_sdst(0)                               // 000000003494: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003498: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 00000000349c: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 0000000034a4: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000034b4: 8c7e017e
	s_lshl_b64 s[26:27], s[46:47], 2                           // 0000000034b8: 849a822e
	v_cmp_lt_i64_e64 s12, 4, v[0:1]                            // 0000000034bc: d451000c 02020084
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034c4: bf88ff9e
	v_add_co_u32 v12, s0, s26, v40                             // 0000000034c8: d700000c 0202501a
	s_wait_alu depctr_va_sdst(0)                               // 0000000034d0: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s27, v41, s0                // 0000000034d4: d5207c0d 0002521b
	s_and_b32 s0, s12, s2                                      // 0000000034dc: 8b00020c
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 0000000034e0: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034e4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000034e8: be812000
	s_cbranch_execz 28                                         // 0000000034ec: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1a60>
	v_bfe_u32 v14, v91, 16, 1                                  // 0000000034f0: d610000e 0205215b
	s_wait_kmcnt 0x0                                           // 0000000034f8: bfc70000
	v_add_co_u32 v15, s0, s18, v2                              // 0000000034fc: d700000f 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003504: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s19, v3, s0                 // 000000003508: d5207c10 00020613
	v_add3_u32 v17, v14, v91, 0x7fff                           // 000000003510: d6550011 03feb70e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000351c: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000003520: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v91                            // 000000003528: 3824b6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003530: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000003534: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v91, v91                               // 00000000353c: d4180000 0202b75b
	s_wait_alu depctr_va_sdst(0)                               // 000000003544: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003548: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 00000000354c: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000003554: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003560: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003564: 8c7e017e
	s_mul_u64 s[24:25], s[46:47], 5                            // 000000003568: aa98852e
	v_cmp_lt_i64_e64 s11, 5, v[0:1]                            // 00000000356c: d451000b 02020085
	s_wait_alu depctr_sa_sdst(0)                               // 000000003574: bf88ff9e
	v_add_co_u32 v14, s0, s24, v40                             // 000000003578: d700000e 02025018
	s_wait_alu depctr_va_sdst(0)                               // 000000003580: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s25, v41, s0                // 000000003584: d5207c0f 00025219
	s_and_b32 s0, s11, s2                                      // 00000000358c: 8b00020b
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000003590: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000003594: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003598: be812000
	s_cbranch_execz 28                                         // 00000000359c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1b10>
	v_bfe_u32 v16, v90, 16, 1                                  // 0000000035a0: d6100010 0205215a
	s_wait_kmcnt 0x0                                           // 0000000035a8: bfc70000
	v_add_co_u32 v17, s0, s18, v2                              // 0000000035ac: d7000011 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b4: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v3, s0                 // 0000000035b8: d5207c12 00020613
	v_add3_u32 v19, v16, v90, 0x7fff                           // 0000000035c0: d6550013 03feb510 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000035cc: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 0000000035d0: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v90                            // 0000000035d8: 3828b4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000035e0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 0000000035e4: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v90, v90                               // 0000000035ec: d4180000 0202b55a
	s_wait_alu depctr_va_sdst(0)                               // 0000000035f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035f8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 0000000035fc: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000003604: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000003610: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003614: 8c7e017e
	s_mul_u64 s[22:23], s[46:47], 6                            // 000000003618: aa96862e
	v_cmp_lt_i64_e64 s9, 6, v[0:1]                             // 00000000361c: d4510009 02020086
	s_wait_alu depctr_sa_sdst(0)                               // 000000003624: bf88ff9e
	v_add_co_u32 v16, s0, s22, v40                             // 000000003628: d7000010 02025016
	s_wait_alu depctr_va_sdst(0)                               // 000000003630: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s23, v41, s0                // 000000003634: d5207c11 00025217
	s_and_b32 s0, s9, s2                                       // 00000000363c: 8b000209
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000003640: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003644: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003648: be812000
	s_cbranch_execz 28                                         // 00000000364c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1bc0>
	v_bfe_u32 v18, v89, 16, 1                                  // 000000003650: d6100012 02052159
	s_wait_kmcnt 0x0                                           // 000000003658: bfc70000
	v_add_co_u32 v19, s0, s18, v2                              // 00000000365c: d7000013 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003664: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v3, s0                 // 000000003668: d5207c14 00020613
	v_add3_u32 v21, v18, v89, 0x7fff                           // 000000003670: d6550015 03feb312 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000367c: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000003680: d7000012 02022113
	v_or_b32_e32 v22, 0x400000, v89                            // 000000003688: 382cb2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003690: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 000000003694: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v89, v89                               // 00000000369c: d4180000 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 0000000036a4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036a8: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 0000000036ac: d5010014 00022d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 0000000036b4: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000036c4: 8c7e017e
	s_mul_u64 s[20:21], s[46:47], 7                            // 0000000036c8: aa94872e
	v_cmp_lt_i64_e64 s8, 7, v[0:1]                             // 0000000036cc: d4510008 02020087
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036d4: bf88ff9e
	v_add_co_u32 v18, s0, s20, v40                             // 0000000036d8: d7000012 02025014
	s_wait_alu depctr_va_sdst(0)                               // 0000000036e0: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s21, v41, s0                // 0000000036e4: d5207c13 00025215
	s_and_b32 s0, s8, s2                                       // 0000000036ec: 8b000208
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 0000000036f0: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036f4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000036f8: be812000
	s_cbranch_execz 28                                         // 0000000036fc: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1c70>
	v_bfe_u32 v0, v88, 16, 1                                   // 000000003700: d6100000 02052158
	s_wait_kmcnt 0x0                                           // 000000003708: bfc70000
	v_add_co_u32 v1, s0, s18, v2                               // 00000000370c: d7000001 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003714: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v3, s0                 // 000000003718: d5207c14 00020613
	v_add3_u32 v21, v0, v88, 0x7fff                            // 000000003720: d6550015 03feb100 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000372c: bf870003
	v_add_co_u32 v0, s0, v1, v18                               // 000000003730: d7000000 02022501
	v_or_b32_e32 v22, 0x400000, v88                            // 000000003738: 382cb0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003740: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v20, v19, s0                 // 000000003744: d5207c01 00022714
	v_cmp_u_f32_e64 s0, v88, v88                               // 00000000374c: d4180000 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 000000003754: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003758: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 00000000375c: d5010014 00022d15
	global_store_d16_hi_b16 v[0:1], v20, off                   // 000000003764: ee09407c 0a000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003770: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003774: 8c7e017e
	v_mul_lo_u32 v22, s47, v42                                 // 000000003778: d72c0016 0202542f
	v_mul_lo_u32 v23, s46, v43                                 // 000000003780: d72c0017 0202562e
	v_mad_co_u64_u32 v[0:1], null, s46, v42, 0                 // 000000003788: d6fe7c00 0202542e
	v_sub_co_u32 v20, s0, s44, v42                             // 000000003790: d7010014 0202542c
	s_wait_alu depctr_va_sdst(0)                               // 000000003798: bf88f19f
	v_sub_co_ci_u32_e64 v21, null, s45, v43, s0                // 00000000379c: d5217c15 0002562d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000037a4: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[20:21]                          // 0000000037a8: d451000a 02022880
	v_add3_u32 v1, v1, v23, v22                                // 0000000037b0: d6550001 045a2f01
	s_delay_alu instid0(valu_dep_1)                            // 0000000037b8: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000037bc: 3e000081
	s_and_b32 s0, s10, s2                                      // 0000000037c0: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037c4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000037c8: be812000
	s_cbranch_execz 28                                         // 0000000037cc: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1d40>
	s_wait_kmcnt 0x0                                           // 0000000037d0: bfc70000
	v_add_co_u32 v23, s0, s18, v0                              // 0000000037d4: d7000017 02020012
	v_bfe_u32 v22, v87, 16, 1                                  // 0000000037dc: d6100016 02052157
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e4: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v1, s0                 // 0000000037e8: d5207c18 00020213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037f0: bf870193
	v_add_co_u32 v4, s0, v23, v4                               // 0000000037f4: d7000004 02020917
	v_add3_u32 v22, v22, v87, 0x7fff                           // 0000000037fc: d6550016 03feaf16 00007fff
	v_or_b32_e32 v25, 0x400000, v87                            // 000000003808: 3832aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003810: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v24, v5, s0                  // 000000003814: d5207c05 00020b18
	v_cmp_u_f32_e64 s0, v87, v87                               // 00000000381c: d4180000 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 000000003824: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003828: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s0                        // 00000000382c: d5010016 00023316
	global_store_d16_hi_b16 v[4:5], v22, off                   // 000000003834: ee09407c 0b000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003840: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003844: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[20:21]                           // 000000003848: d4510007 02022881
	s_and_b32 s0, s7, s2                                       // 000000003850: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 000000003854: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003858: be812000
	s_cbranch_execz 28                                         // 00000000385c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1dd0>
	v_bfe_u32 v4, v86, 16, 1                                   // 000000003860: d6100004 02052156
	s_wait_kmcnt 0x0                                           // 000000003868: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 00000000386c: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003874: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v1, s0                 // 000000003878: d5207c16 00020213
	v_add3_u32 v23, v4, v86, 0x7fff                            // 000000003880: d6550017 03fead04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000388c: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 000000003890: d7000004 02020d05
	v_or_b32_e32 v24, 0x400000, v86                            // 000000003898: 3830acff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v22, v7, s0                  // 0000000038a4: d5207c05 00020f16
	v_cmp_u_f32_e64 s0, v86, v86                               // 0000000038ac: d4180000 0202ad56
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038b8: bf870001
	v_cndmask_b32_e64 v6, v23, v24, s0                         // 0000000038bc: d5010006 00023117
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000038c4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038d0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000038d4: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[20:21]                           // 0000000038d8: d4510006 02022882
	s_and_b32 s0, s6, s2                                       // 0000000038e0: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038e4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000038e8: be812000
	s_cbranch_execz 28                                         // 0000000038ec: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1e60>
	v_bfe_u32 v4, v83, 16, 1                                   // 0000000038f0: d6100004 02052153
	s_wait_kmcnt 0x0                                           // 0000000038f8: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 0000000038fc: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003904: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003908: d5207c06 00020213
	v_add3_u32 v7, v4, v83, 0x7fff                             // 000000003910: d6550007 03fea704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000391c: bf870003
	v_add_co_u32 v4, s0, v5, v8                                // 000000003920: d7000004 02021105
	v_or_b32_e32 v22, 0x400000, v83                            // 000000003928: 382ca6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003930: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s0                   // 000000003934: d5207c05 00021306
	v_cmp_u_f32_e64 s0, v83, v83                               // 00000000393c: d4180000 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000003944: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003948: bf870001
	v_cndmask_b32_e64 v6, v7, v22, s0                          // 00000000394c: d5010006 00022d07
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003954: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003960: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003964: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[20:21]                           // 000000003968: d4510005 02022883
	s_and_b32 s0, s5, s2                                       // 000000003970: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 000000003974: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003978: be812000
	s_cbranch_execz 28                                         // 00000000397c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1ef0>
	v_bfe_u32 v4, v82, 16, 1                                   // 000000003980: d6100004 02052152
	s_wait_kmcnt 0x0                                           // 000000003988: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 00000000398c: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003994: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003998: d5207c06 00020213
	v_add3_u32 v7, v4, v82, 0x7fff                             // 0000000039a0: d6550007 03fea504 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000039ac: bf870003
	v_add_co_u32 v4, s0, v5, v10                               // 0000000039b0: d7000004 02021505
	v_or_b32_e32 v8, 0x400000, v82                             // 0000000039b8: 3810a4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v11, s0                  // 0000000039c4: d5207c05 00021706
	v_cmp_u_f32_e64 s0, v82, v82                               // 0000000039cc: d4180000 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039d8: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 0000000039dc: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000039e4: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000039f4: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[20:21]                           // 0000000039f8: d4510004 02022884
	s_and_b32 s0, s4, s2                                       // 000000003a00: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a04: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003a08: be812000
	s_cbranch_execz 28                                         // 000000003a0c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x1f80>
	v_bfe_u32 v4, v81, 16, 1                                   // 000000003a10: d6100004 02052151
	s_wait_kmcnt 0x0                                           // 000000003a18: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003a1c: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003a24: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003a28: d5207c06 00020213
	v_add3_u32 v7, v4, v81, 0x7fff                             // 000000003a30: d6550007 03fea304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003a3c: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000003a40: d7000004 02021905
	v_or_b32_e32 v8, 0x400000, v81                             // 000000003a48: 3810a2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 000000003a54: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v81, v81                               // 000000003a5c: d4180000 0202a351
	s_wait_alu depctr_va_sdst(0)                               // 000000003a64: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a68: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003a6c: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003a74: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003a84: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[20:21]                           // 000000003a88: d4510003 02022885
	s_and_b32 s0, s3, s2                                       // 000000003a90: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a94: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003a98: be812000
	s_cbranch_execz 28                                         // 000000003a9c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2010>
	v_bfe_u32 v4, v78, 16, 1                                   // 000000003aa0: d6100004 0205214e
	s_wait_kmcnt 0x0                                           // 000000003aa8: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003aac: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003ab4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003ab8: d5207c06 00020213
	v_add3_u32 v7, v4, v78, 0x7fff                             // 000000003ac0: d6550007 03fe9d04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003acc: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 000000003ad0: d7000004 02021d05
	v_or_b32_e32 v8, 0x400000, v78                             // 000000003ad8: 38109cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ae0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 000000003ae4: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v78, v78                               // 000000003aec: d4180000 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000003af4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003af8: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003afc: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003b04: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003b14: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[20:21]                           // 000000003b18: d4510001 02022886
	s_and_b32 s0, s1, s2                                       // 000000003b20: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b24: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 000000003b28: be912000
	s_cbranch_execz 28                                         // 000000003b2c: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x20a0>
	v_bfe_u32 v4, v73, 16, 1                                   // 000000003b30: d6100004 02052149
	s_wait_kmcnt 0x0                                           // 000000003b38: bfc70000
	v_add_co_u32 v5, s0, s18, v0                               // 000000003b3c: d7000005 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003b44: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s0                  // 000000003b48: d5207c06 00020213
	v_add3_u32 v7, v4, v73, 0x7fff                             // 000000003b50: d6550007 03fe9304 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003b5c: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003b60: d7000004 02022105
	v_or_b32_e32 v8, 0x400000, v73                             // 000000003b68: 381092ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003b70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 000000003b74: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v73, v73                               // 000000003b7c: d4180000 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000003b84: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b88: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003b8c: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003b94: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ba0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003ba4: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[20:21]                           // 000000003ba8: d4510000 02022887
	s_and_b32 s2, s0, s2                                       // 000000003bb0: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb4: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 000000003bb8: be912002
	s_cbranch_execz 28                                         // 000000003bbc: bfa5001c <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2130>
	v_bfe_u32 v4, v71, 16, 1                                   // 000000003bc0: d6100004 02052147
	s_wait_kmcnt 0x0                                           // 000000003bc8: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000003bcc: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 000000003bd8: d5207c06 000a0213
	v_add3_u32 v7, v4, v71, 0x7fff                             // 000000003be0: d6550007 03fe8f04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003bec: bf870003
	v_add_co_u32 v4, s2, v5, v18                               // 000000003bf0: d7000204 02022505
	v_or_b32_e32 v8, 0x400000, v71                             // 000000003bf8: 38108eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003c00: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s2                  // 000000003c04: d5207c05 000a2706
	v_cmp_u_f32_e64 s2, v71, v71                               // 000000003c0c: d4180002 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000003c14: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c18: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s2                           // 000000003c1c: d5010006 000a1107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003c24: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003c34: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 000000003c38: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c3c: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003c40: be8f2002
	s_cbranch_execz 40                                         // 000000003c44: bfa50028 <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x21e8>
	v_add_co_u32 v4, s2, v55, s48                              // 000000003c48: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000003c50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 000000003c54: d5207c05 00086280
	v_bfe_u32 v6, v70, 16, 1                                   // 000000003c5c: d6100006 02052146
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c64: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000003c68: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003c70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003c74: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 000000003c7c: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003c80: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003c88: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003c8c: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003c94: 3e080881
	v_add3_u32 v6, v6, v70, 0x7fff                             // 000000003c98: d6550006 03fe8d06 00007fff
	v_or_b32_e32 v9, 0x400000, v70                             // 000000003ca4: 38128cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003cac: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 000000003cb0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003cbc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v70, v70                               // 000000003cc4: d4180002 02028d46
	s_wait_alu depctr_va_sdst(0)                               // 000000003ccc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003cd0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003cd4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003cdc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ce8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000003cec: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 000000003cf0: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cf4: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003cf8: be8f2002
	s_cbranch_execz 46                                         // 000000003cfc: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x22b8>
	v_add_co_u32 v4, s2, v55, s48                              // 000000003d00: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000003d08: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 000000003d0c: d5207c05 00086280
	v_bfe_u32 v6, v69, 16, 1                                   // 000000003d14: d6100006 02052145
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d1c: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000003d20: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003d28: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003d2c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v69                             // 000000003d34: 38128aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d3c: bf8701a3
	v_add_co_u32 v4, s2, s46, v4                               // 000000003d40: d7000204 0202082e
	s_wait_alu depctr_va_sdst(0)                               // 000000003d48: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s47, v5, s2                  // 000000003d4c: d5207c05 000a0a2f
	s_wait_kmcnt 0x0                                           // 000000003d54: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003d58: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003d60: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003d64: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003d6c: 3e080881
	v_add3_u32 v6, v6, v69, 0x7fff                             // 000000003d70: d6550006 03fe8b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d7c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003d80: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003d88: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003d8c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v69, v69                               // 000000003d94: d4180002 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000003d9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003da0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003da4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003dac: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003db8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000003dbc: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 000000003dc0: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dc4: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 000000003dc8: be8e2002
	s_cbranch_execz 46                                         // 000000003dcc: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2388>
	v_add_co_u32 v4, s2, v55, s48                              // 000000003dd0: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 000000003ddc: d5207c05 00086280
	v_bfe_u32 v6, v68, 16, 1                                   // 000000003de4: d6100006 02052144
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dec: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000003df0: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003df8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003dfc: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v68                             // 000000003e04: 381288ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e0c: bf8701a3
	v_add_co_u32 v4, s2, s50, v4                               // 000000003e10: d7000204 02020832
	s_wait_alu depctr_va_sdst(0)                               // 000000003e18: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s51, v5, s2                  // 000000003e1c: d5207c05 000a0a33
	s_wait_kmcnt 0x0                                           // 000000003e24: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003e28: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003e30: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003e34: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003e3c: 3e080881
	v_add3_u32 v6, v6, v68, 0x7fff                             // 000000003e40: d6550006 03fe8906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e4c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003e50: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003e58: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003e5c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v68, v68                               // 000000003e64: d4180002 02028944
	s_wait_alu depctr_va_sdst(0)                               // 000000003e6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e70: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003e74: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003e7c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 000000003e8c: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 000000003e90: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e94: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 000000003e98: be8d2002
	s_cbranch_execz 46                                         // 000000003e9c: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2458>
	v_add_co_u32 v4, s2, v55, s48                              // 000000003ea0: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000003ea8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 000000003eac: d5207c05 00086280
	v_bfe_u32 v6, v67, 16, 1                                   // 000000003eb4: d6100006 02052143
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ebc: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000003ec0: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003ec8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003ecc: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v67                             // 000000003ed4: 381286ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003edc: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003ee0: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003ee8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000003eec: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000003ef4: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003ef8: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003f00: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003f04: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003f0c: 3e080881
	v_add3_u32 v6, v6, v67, 0x7fff                             // 000000003f10: d6550006 03fe8706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f1c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003f20: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003f28: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003f2c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v67, v67                               // 000000003f34: d4180002 02028743
	s_wait_alu depctr_va_sdst(0)                               // 000000003f3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003f40: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003f44: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003f4c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003f5c: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003f60: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f64: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 000000003f68: be8c2002
	s_cbranch_execz 46                                         // 000000003f6c: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2528>
	v_add_co_u32 v4, s2, v55, s48                              // 000000003f70: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000003f78: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 000000003f7c: d5207c05 00086280
	v_bfe_u32 v6, v66, 16, 1                                   // 000000003f84: d6100006 02052142
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f8c: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000003f90: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003f98: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003f9c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v66                             // 000000003fa4: 381284ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fac: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003fb0: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003fb8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000003fbc: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000003fc4: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000003fc8: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000003fd0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000003fd4: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003fdc: 3e080881
	v_add3_u32 v6, v6, v66, 0x7fff                             // 000000003fe0: d6550006 03fe8506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fec: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003ff0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ff8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003ffc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v66, v66                               // 000000004004: d4180002 02028542
	s_wait_alu depctr_va_sdst(0)                               // 00000000400c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004010: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000004014: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000401c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004028: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 00000000402c: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 000000004030: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004034: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 000000004038: be8b2002
	s_cbranch_execz 46                                         // 00000000403c: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x25f8>
	v_add_co_u32 v4, s2, v55, s48                              // 000000004040: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004048: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 00000000404c: d5207c05 00086280
	v_bfe_u32 v6, v65, 16, 1                                   // 000000004054: d6100006 02052141
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000405c: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000004060: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004068: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000406c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v65                             // 000000004074: 381282ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000407c: bf8701a3
	v_add_co_u32 v4, s2, s24, v4                               // 000000004080: d7000204 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000004088: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s2                  // 00000000408c: d5207c05 000a0a19
	s_wait_kmcnt 0x0                                           // 000000004094: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000004098: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 0000000040a0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 0000000040a4: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000040ac: 3e080881
	v_add3_u32 v6, v6, v65, 0x7fff                             // 0000000040b0: d6550006 03fe8306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040bc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000040c0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000040c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000040cc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v65, v65                               // 0000000040d4: d4180002 02028341
	s_wait_alu depctr_va_sdst(0)                               // 0000000040dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040e0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000040e4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000040ec: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 0000000040fc: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 000000004100: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000004104: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 000000004108: be892002
	s_cbranch_execz 46                                         // 00000000410c: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x26c8>
	v_add_co_u32 v4, s2, v55, s48                              // 000000004110: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004118: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 00000000411c: d5207c05 00086280
	v_bfe_u32 v6, v64, 16, 1                                   // 000000004124: d6100006 02052140
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000412c: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000004130: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004138: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000413c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v64                             // 000000004144: 381280ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000414c: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 000000004150: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000004158: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 00000000415c: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 000000004164: bfc70000
	v_add_co_u32 v7, s2, s18, v2                               // 000000004168: d7000207 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000004170: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v3, s2                  // 000000004174: d5207c08 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000417c: 3e080881
	v_add3_u32 v6, v6, v64, 0x7fff                             // 000000004180: d6550006 03fe8106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000418c: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000004190: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004198: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 00000000419c: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v64, v64                               // 0000000041a4: d4180002 02028140
	s_wait_alu depctr_va_sdst(0)                               // 0000000041ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000041b0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000041b4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000041bc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000041cc: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 0000000041d0: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041d4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000041d8: be882002
	s_cbranch_execz 46                                         // 0000000041dc: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2798>
	v_add_co_u32 v4, s2, v55, s48                              // 0000000041e0: d7000204 02006137
	s_wait_alu depctr_va_sdst(0)                               // 0000000041e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s49, s2                   // 0000000041ec: d5207c05 00086280
	v_bfe_u32 v6, v63, 16, 1                                   // 0000000041f4: d6100006 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041fc: bf8701a3
	v_add_co_u32 v4, s2, v4, v54                               // 000000004200: d7000204 02026d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004208: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000420c: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v63                             // 000000004214: 380e7eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000421c: bf8701a3
	v_add_co_u32 v4, s2, s20, v4                               // 000000004220: d7000204 02020814
	s_wait_alu depctr_va_sdst(0)                               // 000000004228: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s21, v5, s2                  // 00000000422c: d5207c05 000a0a15
	s_wait_kmcnt 0x0                                           // 000000004234: bfc70000
	v_add_co_u32 v2, s2, s18, v2                               // 000000004238: d7000202 02020412
	s_wait_alu depctr_va_sdst(0)                               // 000000004240: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s19, v3, s2                  // 000000004244: d5207c03 000a0613
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000424c: 3e080881
	v_add3_u32 v6, v6, v63, 0x7fff                             // 000000004250: d6550006 03fe7f06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000425c: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000004260: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000004268: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 00000000426c: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v63, v63                               // 000000004274: d4180002 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 00000000427c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004280: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000004284: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 00000000428c: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004298: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000429c: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 0000000042a0: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042a4: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 0000000042a8: be882002
	s_cbranch_execz 40                                         // 0000000042ac: bfa50028 <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2850>
	v_add_co_u32 v2, s2, v55, s48                              // 0000000042b0: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 0000000042b8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 0000000042bc: d5207c03 00086280
	v_bfe_u32 v4, v62, 16, 1                                   // 0000000042c4: d6100004 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042cc: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 0000000042d0: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000042d8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000042dc: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 0000000042e4: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 0000000042e8: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000042f0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 0000000042f4: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000042fc: 3e040481
	v_add3_u32 v4, v4, v62, 0x7fff                             // 000000004300: d6550004 03fe7d04 00007fff
	v_or_b32_e32 v7, 0x400000, v62                             // 00000000430c: 380e7cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004314: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000004318: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004320: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000004324: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v62, v62                               // 00000000432c: d4180002 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000004334: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004338: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000433c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004344: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004350: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000004354: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000004358: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 00000000435c: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000004360: be872002
	s_cbranch_execz 46                                         // 000000004364: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2920>
	v_add_co_u32 v2, s2, v55, s48                              // 000000004368: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004370: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 000000004374: d5207c03 00086280
	v_bfe_u32 v4, v61, 16, 1                                   // 00000000437c: d6100004 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004384: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 000000004388: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 000000004390: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000004394: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v61                             // 00000000439c: 380e7aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000043a4: bf8701a3
	v_add_co_u32 v2, s2, s46, v2                               // 0000000043a8: d7000202 0202042e
	s_wait_alu depctr_va_sdst(0)                               // 0000000043b0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s47, v3, s2                  // 0000000043b4: d5207c03 000a062f
	s_wait_kmcnt 0x0                                           // 0000000043bc: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 0000000043c0: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000043c8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 0000000043cc: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000043d4: 3e040481
	v_add3_u32 v4, v4, v61, 0x7fff                             // 0000000043d8: d6550004 03fe7b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000043e4: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000043e8: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000043f0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000043f4: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v61, v61                               // 0000000043fc: d4180002 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000004404: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004408: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000440c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004414: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004420: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000004424: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000004428: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 00000000442c: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000004430: be862002
	s_cbranch_execz 46                                         // 000000004434: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x29f0>
	v_add_co_u32 v2, s2, v55, s48                              // 000000004438: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004440: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 000000004444: d5207c03 00086280
	v_bfe_u32 v4, v60, 16, 1                                   // 00000000444c: d6100004 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004454: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 000000004458: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 000000004460: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000004464: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v60                             // 00000000446c: 380e78ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004474: bf8701a3
	v_add_co_u32 v2, s2, s50, v2                               // 000000004478: d7000202 02020432
	s_wait_alu depctr_va_sdst(0)                               // 000000004480: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s51, v3, s2                  // 000000004484: d5207c03 000a0633
	s_wait_kmcnt 0x0                                           // 00000000448c: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000004490: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000004498: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 00000000449c: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000044a4: 3e040481
	v_add3_u32 v4, v4, v60, 0x7fff                             // 0000000044a8: d6550004 03fe7904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000044b4: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 0000000044b8: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000044c0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000044c4: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v60, v60                               // 0000000044cc: d4180002 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 0000000044d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000044d8: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 0000000044dc: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 0000000044e4: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000044f4: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 0000000044f8: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044fc: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000004500: be852002
	s_cbranch_execz 46                                         // 000000004504: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2ac0>
	v_add_co_u32 v2, s2, v55, s48                              // 000000004508: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004510: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 000000004514: d5207c03 00086280
	v_bfe_u32 v4, v59, 16, 1                                   // 00000000451c: d6100004 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004524: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 000000004528: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 000000004530: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000004534: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v59                             // 00000000453c: 380e76ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004544: bf8701a3
	v_add_co_u32 v2, s2, s28, v2                               // 000000004548: d7000202 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 000000004550: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s2                  // 000000004554: d5207c03 000a061d
	s_wait_kmcnt 0x0                                           // 00000000455c: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000004560: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000004568: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 00000000456c: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004574: 3e040481
	v_add3_u32 v4, v4, v59, 0x7fff                             // 000000004578: d6550004 03fe7704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004584: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000004588: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004590: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000004594: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v59, v59                               // 00000000459c: d4180002 0202773b
	s_wait_alu depctr_va_sdst(0)                               // 0000000045a4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000045a8: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 0000000045ac: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 0000000045b4: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000045c4: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 0000000045c8: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045cc: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 0000000045d0: be842002
	s_cbranch_execz 46                                         // 0000000045d4: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2b90>
	v_add_co_u32 v2, s2, v55, s48                              // 0000000045d8: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 0000000045e0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 0000000045e4: d5207c03 00086280
	v_bfe_u32 v4, v58, 16, 1                                   // 0000000045ec: d6100004 0205213a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045f4: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 0000000045f8: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 000000004600: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000004604: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v58                             // 00000000460c: 380e74ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004614: bf8701a3
	v_add_co_u32 v2, s2, s26, v2                               // 000000004618: d7000202 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000004620: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s2                  // 000000004624: d5207c03 000a061b
	s_wait_kmcnt 0x0                                           // 00000000462c: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000004630: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000004638: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 00000000463c: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004644: 3e040481
	v_add3_u32 v4, v4, v58, 0x7fff                             // 000000004648: d6550004 03fe7504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004654: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000004658: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004660: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000004664: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v58, v58                               // 00000000466c: d4180002 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 000000004674: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004678: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000467c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004684: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004690: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004694: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000004698: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 00000000469c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000046a0: be832002
	s_cbranch_execz 46                                         // 0000000046a4: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2c60>
	v_add_co_u32 v2, s2, v55, s48                              // 0000000046a8: d7000202 02006137
	s_wait_alu depctr_va_sdst(0)                               // 0000000046b0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s2                   // 0000000046b4: d5207c03 00086280
	v_bfe_u32 v4, v57, 16, 1                                   // 0000000046bc: d6100004 02052139
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000046c4: bf8701a3
	v_add_co_u32 v2, s2, v2, v54                               // 0000000046c8: d7000202 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000046d0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000046d4: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v57                             // 0000000046dc: 380e72ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000046e4: bf8701a3
	v_add_co_u32 v2, s2, s24, v2                               // 0000000046e8: d7000202 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000046f0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s2                  // 0000000046f4: d5207c03 000a0619
	s_wait_kmcnt 0x0                                           // 0000000046fc: bfc70000
	v_add_co_u32 v5, s2, s18, v0                               // 000000004700: d7000205 02020012
	s_wait_alu depctr_va_sdst(0)                               // 000000004708: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s2                  // 00000000470c: d5207c06 000a0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004714: 3e040481
	v_add3_u32 v4, v4, v57, 0x7fff                             // 000000004718: d6550004 03fe7304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004724: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000004728: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004730: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000004734: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v57, v57                               // 00000000473c: d4180002 02027339
	s_wait_alu depctr_va_sdst(0)                               // 000000004744: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004748: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 00000000474c: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004754: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004760: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004764: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000004768: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 00000000476c: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000004770: be822001
	s_cbranch_execz 46                                         // 000000004774: bfa5002e <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2d30>
	v_add_co_u32 v2, s1, v55, s48                              // 000000004778: d7000102 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004780: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s1                   // 000000004784: d5207c03 00046280
	v_bfe_u32 v4, v56, 16, 1                                   // 00000000478c: d6100004 02052138
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004794: bf8701a3
	v_add_co_u32 v2, s1, v2, v54                               // 000000004798: d7000102 02026d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000047a0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 0000000047a4: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v56                             // 0000000047ac: 380e70ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000047b4: bf8701a3
	v_add_co_u32 v2, s1, s22, v2                               // 0000000047b8: d7000102 02020416
	s_wait_alu depctr_va_sdst(0)                               // 0000000047c0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s1                  // 0000000047c4: d5207c03 00060617
	s_wait_kmcnt 0x0                                           // 0000000047cc: bfc70000
	v_add_co_u32 v5, s1, s18, v0                               // 0000000047d0: d7000105 02020012
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s19, v1, s1                  // 0000000047dc: d5207c06 00060213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000047e4: 3e040481
	v_add3_u32 v4, v4, v56, 0x7fff                             // 0000000047e8: d6550004 03fe7104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000047f4: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 0000000047f8: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000004800: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000004804: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v56, v56                               // 00000000480c: d4180001 02027138
	s_wait_alu depctr_va_sdst(0)                               // 000000004814: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004818: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 00000000481c: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004824: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004830: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004834: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000004838: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000483c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004840: be812000
	s_cbranch_execz 43                                         // 000000004844: bfa5002b <tessera_rocm_scaled_matmul_lds_ca3ee2619c809164+0x2df4>
	v_add_co_u32 v2, s0, v55, s48                              // 000000004848: d7000002 02006137
	s_wait_alu depctr_va_sdst(0)                               // 000000004850: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s49, s0                   // 000000004854: d5207c03 00006280
	v_bfe_u32 v4, v46, 16, 1                                   // 00000000485c: d6100004 0205212e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004864: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v54                           // 000000004868: d7006a02 02026d02
	s_wait_alu depctr_va_vcc(0)                                // 000000004870: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000004874: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v46                             // 00000000487c: 380a5cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004884: bf8701a3
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 000000004888: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000004890: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 000000004894: d5207c03 01aa0615
	s_wait_kmcnt 0x0                                           // 00000000489c: bfc70000
	v_add_co_u32 v0, vcc_lo, s18, v0                           // 0000000048a0: d7006a00 02020012
	s_wait_alu depctr_va_vcc(0)                                // 0000000048a8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s19, v1, vcc_lo              // 0000000048ac: d5207c01 01aa0213
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000048b4: 3e040481
	v_add3_u32 v4, v4, v46, 0x7fff                             // 0000000048b8: d6550004 03fe5d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048c4: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000048c8: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000048d0: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000048d4: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 0000000048dc: 7c305d2e
	s_wait_alu depctr_va_vcc(0)                                // 0000000048e0: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 0000000048e4: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000048e8: ee09407c 01000000 00002000
	s_nop 0                                                    // 0000000048f4: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000048f8: bfb60003
	s_endpgm                                                   // 0000000048fc: bfb00000
	s_code_end                                                 // 000000004900: bf9f0000
	s_code_end                                                 // 000000004904: bf9f0000
	s_code_end                                                 // 000000004908: bf9f0000
	s_code_end                                                 // 00000000490c: bf9f0000
	s_code_end                                                 // 000000004910: bf9f0000
	s_code_end                                                 // 000000004914: bf9f0000
	s_code_end                                                 // 000000004918: bf9f0000
	s_code_end                                                 // 00000000491c: bf9f0000
	s_code_end                                                 // 000000004920: bf9f0000
	s_code_end                                                 // 000000004924: bf9f0000
	s_code_end                                                 // 000000004928: bf9f0000
	s_code_end                                                 // 00000000492c: bf9f0000
	s_code_end                                                 // 000000004930: bf9f0000
	s_code_end                                                 // 000000004934: bf9f0000
	s_code_end                                                 // 000000004938: bf9f0000
	s_code_end                                                 // 00000000493c: bf9f0000
	s_code_end                                                 // 000000004940: bf9f0000
	s_code_end                                                 // 000000004944: bf9f0000
	s_code_end                                                 // 000000004948: bf9f0000
	s_code_end                                                 // 00000000494c: bf9f0000
	s_code_end                                                 // 000000004950: bf9f0000
	s_code_end                                                 // 000000004954: bf9f0000
	s_code_end                                                 // 000000004958: bf9f0000
	s_code_end                                                 // 00000000495c: bf9f0000
	s_code_end                                                 // 000000004960: bf9f0000
	s_code_end                                                 // 000000004964: bf9f0000
	s_code_end                                                 // 000000004968: bf9f0000
	s_code_end                                                 // 00000000496c: bf9f0000
	s_code_end                                                 // 000000004970: bf9f0000
	s_code_end                                                 // 000000004974: bf9f0000
	s_code_end                                                 // 000000004978: bf9f0000
	s_code_end                                                 // 00000000497c: bf9f0000
	s_code_end                                                 // 000000004980: bf9f0000
	s_code_end                                                 // 000000004984: bf9f0000
	s_code_end                                                 // 000000004988: bf9f0000
	s_code_end                                                 // 00000000498c: bf9f0000
	s_code_end                                                 // 000000004990: bf9f0000
	s_code_end                                                 // 000000004994: bf9f0000
	s_code_end                                                 // 000000004998: bf9f0000
	s_code_end                                                 // 00000000499c: bf9f0000
	s_code_end                                                 // 0000000049a0: bf9f0000
	s_code_end                                                 // 0000000049a4: bf9f0000
	s_code_end                                                 // 0000000049a8: bf9f0000
	s_code_end                                                 // 0000000049ac: bf9f0000
	s_code_end                                                 // 0000000049b0: bf9f0000
	s_code_end                                                 // 0000000049b4: bf9f0000
	s_code_end                                                 // 0000000049b8: bf9f0000
	s_code_end                                                 // 0000000049bc: bf9f0000
	s_code_end                                                 // 0000000049c0: bf9f0000
	s_code_end                                                 // 0000000049c4: bf9f0000
	s_code_end                                                 // 0000000049c8: bf9f0000
	s_code_end                                                 // 0000000049cc: bf9f0000
	s_code_end                                                 // 0000000049d0: bf9f0000
	s_code_end                                                 // 0000000049d4: bf9f0000
	s_code_end                                                 // 0000000049d8: bf9f0000
	s_code_end                                                 // 0000000049dc: bf9f0000
	s_code_end                                                 // 0000000049e0: bf9f0000
	s_code_end                                                 // 0000000049e4: bf9f0000
	s_code_end                                                 // 0000000049e8: bf9f0000
	s_code_end                                                 // 0000000049ec: bf9f0000
	s_code_end                                                 // 0000000049f0: bf9f0000
	s_code_end                                                 // 0000000049f4: bf9f0000
	s_code_end                                                 // 0000000049f8: bf9f0000
	s_code_end                                                 // 0000000049fc: bf9f0000
	s_code_end                                                 // 000000004a00: bf9f0000
	s_code_end                                                 // 000000004a04: bf9f0000
	s_code_end                                                 // 000000004a08: bf9f0000
	s_code_end                                                 // 000000004a0c: bf9f0000
	s_code_end                                                 // 000000004a10: bf9f0000
	s_code_end                                                 // 000000004a14: bf9f0000
	s_code_end                                                 // 000000004a18: bf9f0000
	s_code_end                                                 // 000000004a1c: bf9f0000
	s_code_end                                                 // 000000004a20: bf9f0000
	s_code_end                                                 // 000000004a24: bf9f0000
	s_code_end                                                 // 000000004a28: bf9f0000
	s_code_end                                                 // 000000004a2c: bf9f0000
	s_code_end                                                 // 000000004a30: bf9f0000
	s_code_end                                                 // 000000004a34: bf9f0000
	s_code_end                                                 // 000000004a38: bf9f0000
	s_code_end                                                 // 000000004a3c: bf9f0000
	s_code_end                                                 // 000000004a40: bf9f0000
	s_code_end                                                 // 000000004a44: bf9f0000
	s_code_end                                                 // 000000004a48: bf9f0000
	s_code_end                                                 // 000000004a4c: bf9f0000
	s_code_end                                                 // 000000004a50: bf9f0000
	s_code_end                                                 // 000000004a54: bf9f0000
	s_code_end                                                 // 000000004a58: bf9f0000
	s_code_end                                                 // 000000004a5c: bf9f0000
	s_code_end                                                 // 000000004a60: bf9f0000
	s_code_end                                                 // 000000004a64: bf9f0000
	s_code_end                                                 // 000000004a68: bf9f0000
	s_code_end                                                 // 000000004a6c: bf9f0000
	s_code_end                                                 // 000000004a70: bf9f0000
	s_code_end                                                 // 000000004a74: bf9f0000
	s_code_end                                                 // 000000004a78: bf9f0000
	s_code_end                                                 // 000000004a7c: bf9f0000
