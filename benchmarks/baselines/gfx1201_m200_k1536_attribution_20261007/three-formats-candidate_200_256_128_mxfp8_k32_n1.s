
/tmp/tmp19s1u19i.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_b867c3d0aa38586b>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[36:39], s[0:1], 0xc8                         // 000000001b04: f4004900 f80000c8
	s_load_b64 s[40:41], s[0:1], 0xa8                          // 000000001b0c: f4002a00 f80000a8
	s_mov_b32 s2, ttmp9                                        // 000000001b14: be820075
	s_mov_b32 s4, ttmp7                                        // 000000001b18: be840073
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b1c: 86039f75
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b20: 86059f73
	s_clause 0x4                                               // 000000001b24: bf850004
	s_load_b64 s[42:43], s[0:1], 0xd8                          // 000000001b28: f4002a80 f80000d8
	s_load_b64 s[46:47], s[0:1], 0x8                           // 000000001b30: f4002b80 f8000008
	s_load_b64 s[48:49], s[0:1], 0x30                          // 000000001b38: f4002c00 f8000030
	s_load_b64 s[44:45], s[0:1], 0x58                          // 000000001b40: f4002b00 f8000058
	s_load_b64 s[56:57], s[0:1], 0x80                          // 000000001b48: f4002e00 f8000080
	s_lshl_b64 s[50:51], s[4:5], 4                             // 000000001b50: 84b28404
	s_lshl_b64 s[52:53], s[2:3], 4                             // 000000001b54: 84b48402
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_2)// 000000001b58: bf870149
	v_dual_mov_b32 v4, s53 :: v_dual_and_b32 v25, 15, v0       // 000000001b5c: ca240035 0418008f
	s_add_nc_u64 s[0:1], s[50:51], 16                          // 000000001b64: a9809032
	s_add_nc_u64 s[2:3], s[52:53], 16                          // 000000001b68: a9829034
	v_bfe_u32 v0, v0, 4, 1                                     // 000000001b6c: d6100000 02050900
	v_or_b32_e32 v3, s52, v25                                  // 000000001b74: 38063234
	s_delay_alu instid0(valu_dep_2)                            // 000000001b78: bf870002
	v_dual_mov_b32 v6, 0 :: v_dual_lshlrev_b32 v5, 3, v0       // 000000001b7c: ca220080 06040083
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[36:37]                      // 000000001b88: d4540000 02004800
	v_cmp_gt_i64_e64 s1, s[2:3], s[38:39]                      // 000000001b90: d4540001 02004c02
	s_lshr_b64 s[54:55], s[42:43], 5                           // 000000001b98: 85b6852a
	s_or_b32 s1, s0, s1                                        // 000000001b9c: 8c010100
	v_cmp_gt_i64_e64 s0, s[38:39], v[3:4]                      // 000000001ba0: d4540000 02020626
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ba8: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s1                              // 000000001bac: 8b6a017e
	s_cbranch_vccz 1406                                        // 000000001bb0: bfa3057e <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x16ac>
	v_or_b32_e32 v7, s50, v25                                  // 000000001bb4: 380e3232
	v_dual_mov_b32 v8, s51 :: v_dual_mov_b32 v1, s51           // 000000001bb8: ca100033 08000033
	v_or_b32_e32 v0, s50, v5                                   // 000000001bc0: 38000a32
	v_mul_lo_u32 v2, s43, v3                                   // 000000001bc4: d72c0002 0202062b
	s_delay_alu instid0(valu_dep_4)                            // 000000001bcc: bf870004
	v_mul_lo_u32 v14, s43, v7                                  // 000000001bd0: d72c000e 02020e2b
	v_mad_co_u64_u32 v[9:10], null, s42, v7, 0                 // 000000001bd8: d6fe7c09 02020e2a
	v_mul_lo_u32 v13, s42, v4                                  // 000000001be0: d72c000d 0202082a
	v_mad_co_u64_u32 v[11:12], null, s42, v3, 0                // 000000001be8: d6fe7c0b 0202062a
	s_mul_i32 s2, s42, s51                                     // 000000001bf0: 9602332a
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[0:1]                  // 000000001bf4: 7ca80024
	v_cmp_gt_i64_e64 s1, s[36:37], v[7:8]                      // 000000001bf8: d4540001 02020e24
	v_or_b32_e32 v15, 4, v0                                    // 000000001c00: 381e0084
	v_or_b32_e32 v17, 5, v0                                    // 000000001c04: 38220085
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c08: bf88ff9e
	v_add3_u32 v28, v10, s2, v14                               // 000000001c0c: d655001c 0438050a
	v_or_b32_e32 v29, v9, v5                                   // 000000001c14: 383a0b09
	v_or_b32_e32 v9, 1, v0                                     // 000000001c18: 38120081
	v_mov_b32_e32 v10, s51                                     // 000000001c1c: 7e140233
	v_add3_u32 v30, v12, v13, v2                               // 000000001c20: d655001e 040a1b0c
	v_cndmask_b32_e32 v2, 0, v0, vcc_lo                        // 000000001c28: 02040080
	v_cndmask_b32_e64 v12, 0, s51, vcc_lo                      // 000000001c2c: d501000c 01a86680
	v_or_b32_e32 v31, v11, v5                                  // 000000001c34: 383e0b0b
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[9:10]                 // 000000001c38: 7ca81224
	v_or_b32_e32 v11, 2, v0                                    // 000000001c3c: 38160082
	v_mov_b32_e32 v18, s51                                     // 000000001c40: 7e240233
	v_or_b32_e32 v20, 7, v0                                    // 000000001c44: 38280087
	v_mov_b32_e32 v21, s51                                     // 000000001c48: 7e2a0233
	v_cndmask_b32_e64 v23, 0, v3, s0                           // 000000001c4c: d5010017 00020680
	s_wait_alu depctr_va_vcc(0)                                // 000000001c54: bf88ff9d
	v_cndmask_b32_e32 v9, 0, v9, vcc_lo                        // 000000001c58: 02121280
	v_mul_lo_u32 v13, s55, v2                                  // 000000001c5c: d72c000d 02020437
	v_mad_co_u64_u32 v[7:8], null, s54, v2, s[44:45]           // 000000001c64: d6fe7c07 00b20436
	v_mul_lo_u32 v2, s54, v12                                  // 000000001c6c: d72c0002 02021836
	v_mov_b32_e32 v12, s51                                     // 000000001c74: 7e180233
	v_cndmask_b32_e64 v14, 0, s51, vcc_lo                      // 000000001c78: d501000e 01a86680
	v_cndmask_b32_e64 v24, 0, v4, s0                           // 000000001c80: d5010018 00020880
	s_add_nc_u64 s[58:59], s[42:43], -1                        // 000000001c88: a9bac12a
	s_mov_b64 s[60:61], 0                                      // 000000001c8c: bebc0180
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[11:12]                // 000000001c90: 7ca81624
	v_or_b32_e32 v12, 3, v0                                    // 000000001c94: 38180083
	v_add3_u32 v8, v13, v8, v2                                 // 000000001c98: d6550008 040a110d
	v_mov_b32_e32 v13, s51                                     // 000000001ca0: 7e1a0233
	v_mul_lo_u32 v26, s54, v14                                 // 000000001ca4: d72c001a 02021c36
	v_mul_lo_u32 v2, s55, v9                                   // 000000001cac: d72c0002 02021237
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb4: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v11, vcc_lo                      // 000000001cb8: 02161680
	v_cndmask_b32_e64 v14, 0, s51, vcc_lo                      // 000000001cbc: d501000e 01a86680
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[12:13]                // 000000001cc4: 7ca81824
	v_mov_b32_e32 v16, s51                                     // 000000001cc8: 7e200233
	v_mad_co_u64_u32 v[9:10], null, s54, v9, s[44:45]          // 000000001ccc: d6fe7c09 00b21236
	v_mul_lo_u32 v27, s55, v11                                 // 000000001cd4: d72c001b 02021637
	v_mul_lo_u32 v32, s54, v14                                 // 000000001cdc: d72c0020 02021c36
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce4: bf88ff9d
	v_cndmask_b32_e32 v13, 0, v12, vcc_lo                      // 000000001ce8: 021a1880
	v_cmp_gt_i64_e64 s2, s[36:37], v[15:16]                    // 000000001cec: d4540002 02021e24
	v_mad_co_u64_u32 v[11:12], null, s54, v11, s[44:45]        // 000000001cf4: d6fe7c0b 00b21636
	v_cndmask_b32_e64 v16, 0, s51, vcc_lo                      // 000000001cfc: d5010010 01a86680
	v_add3_u32 v10, v2, v10, v26                               // 000000001d04: d655000a 046a1502
	v_mul_lo_u32 v33, s55, v13                                 // 000000001d0c: d72c0021 02021a37
	v_mad_co_u64_u32 v[13:14], null, s54, v13, s[44:45]        // 000000001d14: d6fe7c0d 00b21a36
	s_wait_alu depctr_va_sdst(0)                               // 000000001d1c: bf88f19f
	v_cndmask_b32_e64 v19, 0, s51, s2                          // 000000001d20: d5010013 00086680
	v_cndmask_b32_e64 v15, 0, v15, s2                          // 000000001d28: d501000f 000a1e80
	v_mul_lo_u32 v34, s54, v16                                 // 000000001d30: d72c0022 02022036
	v_add3_u32 v12, v27, v12, v32                              // 000000001d38: d655000c 0482191b
	v_mov_b32_e32 v32, 0                                       // 000000001d40: 7e400280
	v_mul_lo_u32 v36, s54, v19                                 // 000000001d44: d72c0024 02022636
	v_mov_b32_e32 v19, s51                                     // 000000001d4c: 7e260233
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[17:18]                // 000000001d50: 7ca82224
	v_or_b32_e32 v18, 6, v0                                    // 000000001d54: 38240086
	v_cmp_gt_i64_e64 s2, s[36:37], v[20:21]                    // 000000001d58: d4540002 02022824
	v_mul_lo_u32 v35, s55, v15                                 // 000000001d60: d72c0023 02021e37
	v_mad_co_u64_u32 v[15:16], null, s54, v15, s[44:45]        // 000000001d68: d6fe7c0f 00b21e36
	v_add3_u32 v14, v33, v14, v34                              // 000000001d70: d655000e 048a1d21
	s_wait_alu depctr_va_vcc(0)                                // 000000001d78: bf88ff9d
	v_dual_cndmask_b32 v17, 0, v17 :: v_dual_mov_b32 v34, 0    // 000000001d7c: ca502280 11220080
	v_cndmask_b32_e64 v22, 0, s51, vcc_lo                      // 000000001d84: d5010016 01a86680
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[18:19]                // 000000001d8c: 7ca82424
	s_wait_alu depctr_va_sdst(0)                               // 000000001d90: bf88f19f
	v_cndmask_b32_e64 v40, 0, s51, s2                          // 000000001d94: d5010028 00086680
	v_mul_lo_u32 v37, s55, v17                                 // 000000001d9c: d72c0025 02022237
	v_mov_b32_e32 v26, 0                                       // 000000001da4: 7e340280
	v_mul_lo_u32 v38, s54, v22                                 // 000000001da8: d72c0026 02022c36
	v_cndmask_b32_e64 v22, 0, v20, s2                          // 000000001db0: d5010016 000a2880
	s_wait_alu depctr_va_vcc(0)                                // 000000001db8: bf88ff9d
	v_cndmask_b32_e32 v19, 0, v18, vcc_lo                      // 000000001dbc: 02262480
	v_cndmask_b32_e64 v21, 0, s51, vcc_lo                      // 000000001dc0: d5010015 01a86680
	v_mad_co_u64_u32 v[17:18], null, s54, v17, s[44:45]        // 000000001dc8: d6fe7c11 00b22236
	v_mul_lo_u32 v40, s54, v40                                 // 000000001dd0: d72c0028 02025036
	v_mul_lo_u32 v42, s55, v22                                 // 000000001dd8: d72c002a 02022c37
	v_mul_lo_u32 v39, s55, v19                                 // 000000001de0: d72c0027 02022637
	v_mad_co_u64_u32 v[19:20], null, s54, v19, s[44:45]        // 000000001de8: d6fe7c13 00b22636
	v_mul_lo_u32 v41, s54, v21                                 // 000000001df0: d72c0029 02022a36
	v_mad_co_u64_u32 v[21:22], null, s54, v22, s[44:45]        // 000000001df8: d6fe7c15 00b22c36
	v_add_co_u32 v23, vcc_lo, s56, v23                         // 000000001e00: d7006a17 02022e38
	s_wait_alu depctr_va_vcc(0)                                // 000000001e08: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s57, v24, vcc_lo            // 000000001e0c: d5207c18 01aa3039
	v_add3_u32 v16, v35, v16, v36                              // 000000001e14: d6550010 04922123
	v_add3_u32 v18, v37, v18, v38                              // 000000001e1c: d6550012 049a2525
	v_add3_u32 v20, v39, v20, v41                              // 000000001e24: d6550014 04a62927
	v_add3_u32 v22, v42, v22, v40                              // 000000001e2c: d6550016 04a22d2a
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v2, 0              // 000000001e34: ca100080 23020080
	v_mov_b32_e32 v33, 0                                       // 000000001e3c: 7e420280
	v_mov_b32_e32 v27, 0                                       // 000000001e40: 7e360280
	v_or_b32_e32 v36, s60, v5                                  // 000000001e44: 38480a3c
	v_add_co_u32 v50, vcc_lo, s60, v29                         // 000000001e48: d7006a32 02023a3c
	v_mov_b32_e32 v37, s61                                     // 000000001e50: 7e4a023d
	s_wait_alu depctr_va_vcc(0)                                // 000000001e54: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s61, v28, vcc_lo            // 000000001e58: d5207c33 01aa383d
	s_delay_alu instid0(valu_dep_3)                            // 000000001e60: bf870003
	v_or_b32_e32 v38, 1, v50                                   // 000000001e64: 384c6481
	v_mov_b32_e32 v41, s61                                     // 000000001e68: 7e52023d
	v_cmp_gt_u64_e64 s9, s[42:43], v[36:37]                    // 000000001e6c: d45c0009 0202482a
	v_or_b32_e32 v46, 2, v50                                   // 000000001e74: 385c6482
	v_or_b32_e32 v44, 4, v36                                   // 000000001e78: 38584884
	v_mov_b32_e32 v45, s61                                     // 000000001e7c: 7e5a023d
	v_or_b32_e32 v48, 3, v50                                   // 000000001e80: 38606483
	v_or_b32_e32 v52, 5, v50                                   // 000000001e84: 38686485
	s_and_b32 vcc_lo, s1, s9                                   // 000000001e88: 8b6a0901
	v_or_b32_e32 v54, 6, v50                                   // 000000001e8c: 386c6486
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e90: bf88ff9e
	v_cndmask_b32_e32 v40, 0, v50, vcc_lo                      // 000000001e94: 02506480
	v_cmp_gt_u64_e64 s10, s[58:59], v[36:37]                   // 000000001e98: d45c000a 0202483a
	v_cndmask_b32_e32 v39, 0, v51, vcc_lo                      // 000000001ea0: 024e6680
	v_cmp_gt_u64_e64 s14, s[42:43], v[44:45]                   // 000000001ea4: d45c000e 0202582a
	v_or_b32_e32 v44, 4, v50                                   // 000000001eac: 38586484
	s_and_b32 s9, s0, s9                                       // 000000001eb0: 8b090900
	s_or_b32 s35, s60, 16                                      // 000000001eb4: 8c23903c
	s_and_b32 s2, s1, s10                                      // 000000001eb8: 8b020a01
	s_and_b32 s10, s0, s10                                     // 000000001ebc: 8b0a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ec0: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v38, s2                          // 000000001ec4: d501002a 000a4c80
	v_add_co_u32 v38, s3, s46, v40                             // 000000001ecc: d7000326 0202502e
	v_or_b32_e32 v40, 2, v36                                   // 000000001ed4: 38504882
	v_cndmask_b32_e64 v43, 0, v51, s2                          // 000000001ed8: d501002b 000a6680
	s_wait_alu depctr_va_sdst(0)                               // 000000001ee0: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s47, v39, s3                // 000000001ee4: d5207c27 000e4e2f
	v_add_co_u32 v42, s3, s46, v42                             // 000000001eec: d700032a 0202542e
	v_cmp_gt_u64_e64 s11, s[42:43], v[40:41]                   // 000000001ef4: d45c000b 0202502a
	v_or_b32_e32 v40, 3, v36                                   // 000000001efc: 38504883
	s_wait_alu depctr_va_sdst(0)                               // 000000001f00: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s47, v43, s3                // 000000001f04: d5207c2b 000e562f
	s_and_b32 s5, s1, s14                                      // 000000001f0c: 8b050e01
	v_mov_b32_e32 v57, s61                                     // 000000001f10: 7e72023d
	s_and_b32 s3, s1, s11                                      // 000000001f14: 8b030b01
	v_cmp_gt_u64_e64 s13, s[42:43], v[40:41]                   // 000000001f18: d45c000d 0202502a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f20: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v46, s3                          // 000000001f24: d501002e 000e5c80
	v_cndmask_b32_e64 v47, 0, v51, s3                          // 000000001f2c: d501002f 000e6680
	v_cndmask_b32_e64 v49, 0, v44, s5                          // 000000001f34: d5010031 00165880
	v_or_b32_e32 v44, 5, v36                                   // 000000001f3c: 38584885
	s_lshr_b64 s[62:63], s[60:61], 5                           // 000000001f40: 85be853c
	v_add_co_u32 v40, s4, s46, v46                             // 000000001f44: d7000428 02025c2e
	s_wait_alu depctr_va_sdst(0)                               // 000000001f4c: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s47, v47, s4                // 000000001f50: d5207c29 00125e2f
	s_and_b32 s4, s1, s13                                      // 000000001f58: 8b040d01
	v_cmp_gt_u64_e64 s15, s[42:43], v[44:45]                   // 000000001f5c: d45c000f 0202582a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f64: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v48, s4                          // 000000001f68: d501002e 00126080
	v_cndmask_b32_e64 v47, 0, v51, s4                          // 000000001f70: d501002f 00126680
	v_cndmask_b32_e64 v48, 0, v51, s5                          // 000000001f78: d5010030 00166680
	s_mul_i32 s64, s62, s39                                    // 000000001f80: 9640273e
	s_delay_alu instid0(valu_dep_3)                            // 000000001f84: bf870003
	v_add_co_u32 v46, s6, s46, v46                             // 000000001f88: d700062e 02025c2e
	s_wait_alu depctr_va_sdst(0)                               // 000000001f90: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s47, v47, s6                // 000000001f94: d5207c2f 001a5e2f
	v_add_co_u32 v44, s6, s46, v49                             // 000000001f9c: d700062c 0202622e
	s_wait_alu depctr_va_sdst(0)                               // 000000001fa4: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s47, v48, s6                // 000000001fa8: d5207c2d 001a602f
	v_or_b32_e32 v48, 6, v36                                   // 000000001fb0: 38604886
	v_mov_b32_e32 v49, s61                                     // 000000001fb4: 7e62023d
	v_or_b32_e32 v36, 7, v36                                   // 000000001fb8: 38484887
	s_and_b32 s6, s1, s15                                      // 000000001fbc: 8b060f01
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fc0: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v52, s6                          // 000000001fc4: d5010034 001a6880
	v_cmp_gt_u64_e64 s16, s[42:43], v[48:49]                   // 000000001fcc: d45c0010 0202602a
	v_cndmask_b32_e64 v53, 0, v51, s6                          // 000000001fd4: d5010035 001a6680
	v_cmp_gt_u64_e64 s17, s[42:43], v[36:37]                   // 000000001fdc: d45c0011 0202482a
	v_or_b32_e32 v36, 7, v50                                   // 000000001fe4: 38486487
	v_add_co_u32 v48, s7, s46, v52                             // 000000001fe8: d7000730 0202682e
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff0: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s47, v53, s7                // 000000001ff4: d5207c31 001e6a2f
	s_and_b32 s7, s1, s16                                      // 000000001ffc: 8b071001
	s_and_b32 s8, s1, s17                                      // 000000002000: 8b081101
	s_wait_alu depctr_sa_sdst(0)                               // 000000002004: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v54, s7                          // 000000002008: d5010032 001e6c80
	v_cndmask_b32_e64 v37, 0, v51, s7                          // 000000002010: d5010025 001e6680
	v_cndmask_b32_e64 v36, 0, v36, s8                          // 000000002018: d5010024 00224880
	v_cndmask_b32_e64 v53, 0, v51, s8                          // 000000002020: d5010035 00226680
	s_delay_alu instid0(valu_dep_4)                            // 000000002028: bf870004
	v_add_co_u32 v50, s12, s46, v50                            // 00000000202c: d7000c32 0202642e
	s_wait_alu depctr_va_sdst(0)                               // 000000002034: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s47, v37, s12               // 000000002038: d5207c33 00324a2f
	v_add_co_u32 v52, s12, s46, v36                            // 000000002040: d7000c34 0202482e
	s_wait_alu depctr_va_sdst(0)                               // 000000002048: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s47, v53, s12               // 00000000204c: d5207c35 00326a2f
	v_add_co_u32 v54, s12, s60, v31                            // 000000002054: d7000c36 02023e3c
	s_wait_alu depctr_va_sdst(0)                               // 00000000205c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s61, v30, s12               // 000000002060: d5207c37 00323c3d
	s_clause 0x7                                               // 000000002068: bf850007
	global_load_d16_u8 v36, v[38:39], off                      // 00000000206c: ee07807c 00000024 00000026
	global_load_d16_hi_u8 v36, v[42:43], off                   // 000000002078: ee08407c 00000024 0000002a
	global_load_d16_u8 v37, v[40:41], off                      // 000000002084: ee07807c 00000025 00000028
	global_load_d16_hi_u8 v37, v[46:47], off                   // 000000002090: ee08407c 00000025 0000002e
	global_load_d16_u8 v38, v[44:45], off                      // 00000000209c: ee07807c 00000026 0000002c
	global_load_d16_hi_u8 v38, v[48:49], off                   // 0000000020a8: ee08407c 00000026 00000030
	global_load_d16_u8 v39, v[50:51], off                      // 0000000020b4: ee07807c 00000027 00000032
	global_load_d16_hi_u8 v39, v[52:53], off                   // 0000000020c0: ee08407c 00000027 00000034
	v_or_b32_e32 v40, 1, v54                                   // 0000000020cc: 38506c81
	v_cndmask_b32_e64 v42, 0, v54, s9                          // 0000000020d0: d501002a 00266c80
	v_cndmask_b32_e64 v41, 0, v55, s9                          // 0000000020d8: d5010029 00266e80
	v_cndmask_b32_e64 v43, 0, v55, s10                         // 0000000020e0: d501002b 002a6e80
	v_or_b32_e32 v45, 2, v54                                   // 0000000020e8: 385a6c82
	v_cndmask_b32_e64 v44, 0, v40, s10                         // 0000000020ec: d501002c 002a5080
	v_add_co_u32 v40, s12, s48, v42                            // 0000000020f4: d7000c28 02025430
	s_wait_alu depctr_va_sdst(0)                               // 0000000020fc: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s49, v41, s12               // 000000002100: d5207c29 00325231
	s_delay_alu instid0(valu_dep_3)                            // 000000002108: bf870003
	v_add_co_u32 v42, s12, s48, v44                            // 00000000210c: d7000c2a 02025830
	v_or_b32_e32 v44, 3, v54                                   // 000000002114: 38586c83
	s_wait_alu depctr_va_sdst(0)                               // 000000002118: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s49, v43, s12               // 00000000211c: d5207c2b 00325631
	s_and_b32 s12, s0, s11                                     // 000000002124: 8b0c0b00
	s_and_b32 s11, s0, s13                                     // 000000002128: 8b0b0d00
	s_wait_alu depctr_sa_sdst(0)                               // 00000000212c: bf88ff9e
	v_cndmask_b32_e64 v45, 0, v45, s12                         // 000000002130: d501002d 00325a80
	v_cndmask_b32_e64 v46, 0, v55, s12                         // 000000002138: d501002e 00326e80
	v_cndmask_b32_e64 v48, 0, v44, s11                         // 000000002140: d5010030 002e5880
	v_cndmask_b32_e64 v47, 0, v55, s11                         // 000000002148: d501002f 002e6e80
	v_or_b32_e32 v49, 4, v54                                   // 000000002150: 38626c84
	v_add_co_u32 v44, s13, s48, v45                            // 000000002154: d7000d2c 02025a30
	s_wait_alu depctr_va_sdst(0)                               // 00000000215c: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s49, v46, s13               // 000000002160: d5207c2d 00365c31
	v_add_co_u32 v46, s13, s48, v48                            // 000000002168: d7000d2e 02026030
	v_or_b32_e32 v48, 5, v54                                   // 000000002170: 38606c85
	s_wait_alu depctr_va_sdst(0)                               // 000000002174: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s49, v47, s13               // 000000002178: d5207c2f 00365e31
	s_and_b32 s13, s0, s14                                     // 000000002180: 8b0d0e00
	s_and_b32 s14, s0, s15                                     // 000000002184: 8b0e0f00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002188: bf88ff9e
	v_cndmask_b32_e64 v49, 0, v49, s13                         // 00000000218c: d5010031 00366280
	v_cndmask_b32_e64 v50, 0, v55, s13                         // 000000002194: d5010032 00366e80
	v_cndmask_b32_e64 v52, 0, v48, s14                         // 00000000219c: d5010034 003a6080
	v_cndmask_b32_e64 v51, 0, v55, s14                         // 0000000021a4: d5010033 003a6e80
	v_or_b32_e32 v53, 6, v54                                   // 0000000021ac: 386a6c86
	v_add_co_u32 v48, s15, s48, v49                            // 0000000021b0: d7000f30 02026230
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b8: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s49, v50, s15               // 0000000021bc: d5207c31 003e6431
	v_add_co_u32 v50, s15, s48, v52                            // 0000000021c4: d7000f32 02026830
	v_or_b32_e32 v52, 7, v54                                   // 0000000021cc: 38686c87
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d0: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s49, v51, s15               // 0000000021d4: d5207c33 003e6631
	s_and_b32 s15, s0, s16                                     // 0000000021dc: 8b0f1000
	s_and_b32 s16, s0, s17                                     // 0000000021e0: 8b101100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021e4: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s15                         // 0000000021e8: d5010035 003e6a80
	v_cndmask_b32_e64 v54, 0, v55, s15                         // 0000000021f0: d5010036 003e6e80
	v_cndmask_b32_e64 v56, 0, v52, s16                         // 0000000021f8: d5010038 00426880
	v_cndmask_b32_e64 v55, 0, v55, s16                         // 000000002200: d5010037 00426e80
	s_delay_alu instid0(valu_dep_4)                            // 000000002208: bf870004
	v_add_co_u32 v52, s17, s48, v53                            // 00000000220c: d7001134 02026a30
	s_wait_alu depctr_va_sdst(0)                               // 000000002214: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s49, v54, s17               // 000000002218: d5207c35 00466c31
	v_add_co_u32 v54, s17, s48, v56                            // 000000002220: d7001136 02027030
	v_or_b32_e32 v56, s35, v5                                  // 000000002228: 38700a23
	s_wait_alu depctr_va_sdst(0)                               // 00000000222c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s49, v55, s17               // 000000002230: d5207c37 00466e31
	v_add_co_u32 v58, s17, s35, v29                            // 000000002238: d700113a 02023a23
	s_delay_alu instid0(valu_dep_3)                            // 000000002240: bf870003
	v_cmp_gt_u64_e64 s25, s[42:43], v[56:57]                   // 000000002244: d45c0019 0202702a
	v_cmp_gt_u64_e64 s26, s[58:59], v[56:57]                   // 00000000224c: d45c001a 0202703a
	s_wait_alu depctr_va_sdst(0)                               // 000000002254: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s61, v28, s17               // 000000002258: d5207c3b 0046383d
	s_clause 0x7                                               // 000000002260: bf850007
	global_load_d16_u8 v40, v[40:41], off                      // 000000002264: ee07807c 00000028 00000028
	global_load_d16_hi_u8 v40, v[42:43], off                   // 000000002270: ee08407c 00000028 0000002a
	global_load_d16_u8 v41, v[44:45], off                      // 00000000227c: ee07807c 00000029 0000002c
	global_load_d16_hi_u8 v41, v[46:47], off                   // 000000002288: ee08407c 00000029 0000002e
	global_load_d16_u8 v42, v[48:49], off                      // 000000002294: ee07807c 0000002a 00000030
	global_load_d16_hi_u8 v42, v[50:51], off                   // 0000000022a0: ee08407c 0000002a 00000032
	global_load_d16_u8 v43, v[52:53], off                      // 0000000022ac: ee07807c 0000002b 00000034
	global_load_d16_hi_u8 v43, v[54:55], off                   // 0000000022b8: ee08407c 0000002b 00000036
	v_or_b32_e32 v44, 1, v58                                   // 0000000022c4: 38587481
	s_and_b32 s17, s1, s25                                     // 0000000022c8: 8b111901
	s_and_b32 s18, s1, s26                                     // 0000000022cc: 8b121a01
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022d0: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v58, s17                         // 0000000022d4: d501002e 00467480
	v_mov_b32_e32 v47, s61                                     // 0000000022dc: 7e5e023d
	v_cndmask_b32_e64 v48, 0, v44, s18                         // 0000000022e0: d5010030 004a5880
	v_cndmask_b32_e64 v45, 0, v59, s17                         // 0000000022e8: d501002d 00467680
	v_cndmask_b32_e64 v49, 0, v59, s18                         // 0000000022f0: d5010031 004a7680
	v_add_co_u32 v44, s19, s46, v46                            // 0000000022f8: d700132c 02025c2e
	v_or_b32_e32 v46, 2, v56                                   // 000000002300: 385c7082
	v_or_b32_e32 v52, 2, v58                                   // 000000002304: 38687482
	s_wait_alu depctr_va_sdst(0)                               // 000000002308: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s47, v45, s19               // 00000000230c: d5207c2d 004e5a2f
	v_add_co_u32 v48, s19, s46, v48                            // 000000002314: d7001330 0202602e
	v_cmp_gt_u64_e64 s27, s[42:43], v[46:47]                   // 00000000231c: d45c001b 02025c2a
	v_or_b32_e32 v46, 3, v56                                   // 000000002324: 385c7083
	v_or_b32_e32 v50, 4, v56                                   // 000000002328: 38647084
	v_mov_b32_e32 v51, s61                                     // 00000000232c: 7e66023d
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s47, v49, s19               // 000000002334: d5207c31 004e622f
	s_and_b32 s19, s1, s27                                     // 00000000233c: 8b131b01
	v_cmp_gt_u64_e64 s29, s[42:43], v[46:47]                   // 000000002340: d45c001d 02025c2a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002348: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v52, s19                         // 00000000234c: d5010034 004e6880
	v_cmp_gt_u64_e64 s30, s[42:43], v[50:51]                   // 000000002354: d45c001e 0202642a
	v_cndmask_b32_e64 v53, 0, v59, s19                         // 00000000235c: d5010035 004e7680
	v_or_b32_e32 v54, 3, v58                                   // 000000002364: 386c7483
	v_or_b32_e32 v50, 4, v58                                   // 000000002368: 38647484
	v_add_co_u32 v46, s20, s46, v52                            // 00000000236c: d700142e 0202682e
	s_wait_alu depctr_va_sdst(0)                               // 000000002374: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s47, v53, s20               // 000000002378: d5207c2f 00526a2f
	s_and_b32 s20, s1, s29                                     // 000000002380: 8b141d01
	s_and_b32 s21, s1, s30                                     // 000000002384: 8b151e01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002388: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v54, s20                         // 00000000238c: d5010034 00526c80
	v_cndmask_b32_e64 v55, 0, v50, s21                         // 000000002394: d5010037 00566480
	v_or_b32_e32 v50, 5, v56                                   // 00000000239c: 38647085
	v_cndmask_b32_e64 v53, 0, v59, s20                         // 0000000023a0: d5010035 00527680
	v_cndmask_b32_e64 v54, 0, v59, s21                         // 0000000023a8: d5010036 00567680
	v_add_co_u32 v52, s22, s46, v52                            // 0000000023b0: d7001634 0202682e
	s_delay_alu instid0(valu_dep_4)                            // 0000000023b8: bf870004
	v_cmp_gt_u64_e64 s31, s[42:43], v[50:51]                   // 0000000023bc: d45c001f 0202642a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c4: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s47, v53, s22               // 0000000023c8: d5207c35 005a6a2f
	v_add_co_u32 v50, s22, s46, v55                            // 0000000023d0: d7001632 02026e2e
	v_or_b32_e32 v60, 5, v58                                   // 0000000023d8: 38787485
	s_wait_alu depctr_va_sdst(0)                               // 0000000023dc: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s47, v54, s22               // 0000000023e0: d5207c33 005a6c2f
	v_or_b32_e32 v54, 6, v56                                   // 0000000023e8: 386c7086
	v_mov_b32_e32 v55, s61                                     // 0000000023ec: 7e6e023d
	v_or_b32_e32 v56, 7, v56                                   // 0000000023f0: 38707087
	s_and_b32 s22, s1, s31                                     // 0000000023f4: 8b161f01
	v_or_b32_e32 v62, 6, v58                                   // 0000000023f8: 387c7486
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023fc: bf88ff9e
	v_cndmask_b32_e64 v60, 0, v60, s22                         // 000000002400: d501003c 005a7880
	v_cmp_gt_u64_e64 s33, s[42:43], v[54:55]                   // 000000002408: d45c0021 02026c2a
	v_cndmask_b32_e64 v61, 0, v59, s22                         // 000000002410: d501003d 005a7680
	v_cmp_gt_u64_e64 s34, s[42:43], v[56:57]                   // 000000002418: d45c0022 0202702a
	v_or_b32_e32 v56, 7, v58                                   // 000000002420: 38707487
	v_add_co_u32 v54, s23, s46, v60                            // 000000002424: d7001736 0202782e
	s_wait_alu depctr_va_sdst(0)                               // 00000000242c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s47, v61, s23               // 000000002430: d5207c37 005e7a2f
	s_and_b32 s23, s1, s33                                     // 000000002438: 8b172101
	s_and_b32 s24, s1, s34                                     // 00000000243c: 8b182201
	s_wait_alu depctr_sa_sdst(0)                               // 000000002440: bf88ff9e
	v_cndmask_b32_e64 v58, 0, v62, s23                         // 000000002444: d501003a 005e7c80
	v_cndmask_b32_e64 v57, 0, v59, s23                         // 00000000244c: d5010039 005e7680
	v_cndmask_b32_e64 v60, 0, v56, s24                         // 000000002454: d501003c 00627080
	v_cndmask_b32_e64 v59, 0, v59, s24                         // 00000000245c: d501003b 00627680
	s_and_b32 s25, s0, s25                                     // 000000002464: 8b191900
	v_add_co_u32 v56, s28, s46, v58                            // 000000002468: d7001c38 0202742e
	s_wait_alu depctr_va_sdst(0)                               // 000000002470: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s47, v57, s28               // 000000002474: d5207c39 0072722f
	v_add_co_u32 v58, s28, s46, v60                            // 00000000247c: d7001c3a 0202782e
	s_wait_alu depctr_va_sdst(0)                               // 000000002484: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s47, v59, s28               // 000000002488: d5207c3b 0072762f
	v_add_co_u32 v60, s28, s35, v31                            // 000000002490: d7001c3c 02023e23
	s_wait_alu depctr_va_sdst(0)                               // 000000002498: bf88f19f
	v_add_co_ci_u32_e64 v61, null, s61, v30, s28               // 00000000249c: d5207c3d 00723c3d
	s_clause 0x7                                               // 0000000024a4: bf850007
	global_load_d16_u8 v44, v[44:45], off                      // 0000000024a8: ee07807c 0000002c 0000002c
	global_load_d16_hi_u8 v44, v[48:49], off                   // 0000000024b4: ee08407c 0000002c 00000030
	global_load_d16_u8 v45, v[46:47], off                      // 0000000024c0: ee07807c 0000002d 0000002e
	global_load_d16_hi_u8 v45, v[52:53], off                   // 0000000024cc: ee08407c 0000002d 00000034
	global_load_d16_u8 v46, v[50:51], off                      // 0000000024d8: ee07807c 0000002e 00000032
	global_load_d16_hi_u8 v46, v[54:55], off                   // 0000000024e4: ee08407c 0000002e 00000036
	global_load_d16_u8 v47, v[56:57], off                      // 0000000024f0: ee07807c 0000002f 00000038
	global_load_d16_hi_u8 v47, v[58:59], off                   // 0000000024fc: ee08407c 0000002f 0000003a
	v_or_b32_e32 v48, 1, v60                                   // 000000002508: 38607881
	s_wait_alu depctr_sa_sdst(0)                               // 00000000250c: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v60, s25                         // 000000002510: d5010032 00667880
	s_and_b32 s26, s0, s26                                     // 000000002518: 8b1a1a00
	v_cndmask_b32_e64 v49, 0, v61, s25                         // 00000000251c: d5010031 00667a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002524: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v61, s26                         // 000000002528: d5010033 006a7a80
	v_cndmask_b32_e64 v52, 0, v48, s26                         // 000000002530: d5010034 006a6080
	v_add_co_u32 v48, s28, s48, v50                            // 000000002538: d7001c30 02026430
	v_or_b32_e32 v53, 2, v60                                   // 000000002540: 386a7882
	s_wait_alu depctr_va_sdst(0)                               // 000000002544: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s49, v49, s28               // 000000002548: d5207c31 00726231
	v_add_co_u32 v50, s28, s48, v52                            // 000000002550: d7001c32 02026830
	v_or_b32_e32 v52, 3, v60                                   // 000000002558: 38687883
	s_wait_alu depctr_va_sdst(0)                               // 00000000255c: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s49, v51, s28               // 000000002560: d5207c33 00726631
	s_and_b32 s28, s0, s27                                     // 000000002568: 8b1c1b00
	s_and_b32 s27, s0, s29                                     // 00000000256c: 8b1b1d00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002570: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s28                         // 000000002574: d5010035 00726a80
	v_cndmask_b32_e64 v54, 0, v61, s28                         // 00000000257c: d5010036 00727a80
	v_cndmask_b32_e64 v56, 0, v52, s27                         // 000000002584: d5010038 006e6880
	v_cndmask_b32_e64 v55, 0, v61, s27                         // 00000000258c: d5010037 006e7a80
	v_or_b32_e32 v57, 4, v60                                   // 000000002594: 38727884
	v_add_co_u32 v52, s29, s48, v53                            // 000000002598: d7001d34 02026a30
	s_wait_alu depctr_va_sdst(0)                               // 0000000025a0: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s49, v54, s29               // 0000000025a4: d5207c35 00766c31
	v_add_co_u32 v54, s29, s48, v56                            // 0000000025ac: d7001d36 02027030
	v_or_b32_e32 v56, 5, v60                                   // 0000000025b4: 38707885
	s_wait_alu depctr_va_sdst(0)                               // 0000000025b8: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s49, v55, s29               // 0000000025bc: d5207c37 00766e31
	s_and_b32 s29, s0, s30                                     // 0000000025c4: 8b1d1e00
	s_and_b32 s30, s0, s31                                     // 0000000025c8: 8b1e1f00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025cc: bf88ff9e
	v_cndmask_b32_e64 v57, 0, v57, s29                         // 0000000025d0: d5010039 00767280
	v_cndmask_b32_e64 v58, 0, v61, s29                         // 0000000025d8: d501003a 00767a80
	v_cndmask_b32_e64 v62, 0, v56, s30                         // 0000000025e0: d501003e 007a7080
	v_cndmask_b32_e64 v59, 0, v61, s30                         // 0000000025e8: d501003b 007a7a80
	v_or_b32_e32 v63, 6, v60                                   // 0000000025f0: 387e7886
	v_add_co_u32 v56, s31, s48, v57                            // 0000000025f4: d7001f38 02027230
	s_wait_alu depctr_va_sdst(0)                               // 0000000025fc: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s49, v58, s31               // 000000002600: d5207c39 007e7431
	v_add_co_u32 v58, s31, s48, v62                            // 000000002608: d7001f3a 02027c30
	v_or_b32_e32 v60, 7, v60                                   // 000000002610: 38787887
	s_wait_alu depctr_va_sdst(0)                               // 000000002614: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s49, v59, s31               // 000000002618: d5207c3b 007e7631
	s_and_b32 s31, s0, s33                                     // 000000002620: 8b1f2100
	s_and_b32 s33, s0, s34                                     // 000000002624: 8b212200
	s_wait_alu depctr_sa_sdst(0)                               // 000000002628: bf88ff9e
	v_cndmask_b32_e64 v63, 0, v63, s31                         // 00000000262c: d501003f 007e7e80
	v_cndmask_b32_e64 v62, 0, v61, s31                         // 000000002634: d501003e 007e7a80
	v_cndmask_b32_e64 v65, 0, v60, s33                         // 00000000263c: d5010041 00867880
	v_cndmask_b32_e64 v64, 0, v61, s33                         // 000000002644: d5010040 00867a80
	s_delay_alu instid0(valu_dep_4)                            // 00000000264c: bf870004
	v_add_co_u32 v60, s34, s48, v63                            // 000000002650: d700223c 02027e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002658: bf88f19f
	v_add_co_ci_u32_e64 v61, null, s49, v62, s34               // 00000000265c: d5207c3d 008a7c31
	v_add_co_u32 v62, s34, s48, v65                            // 000000002664: d700223e 02028230
	s_wait_alu depctr_va_sdst(0)                               // 00000000266c: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s49, v64, s34               // 000000002670: d5207c3f 008a8031
	v_mad_co_u64_u32 v[64:65], null, s62, s38, v[23:24]        // 000000002678: d6fe7c40 045c4c3e
	s_clause 0x7                                               // 000000002680: bf850007
	global_load_d16_u8 v48, v[48:49], off                      // 000000002684: ee07807c 00000030 00000030
	global_load_d16_hi_u8 v48, v[50:51], off                   // 000000002690: ee08407c 00000030 00000032
	global_load_d16_u8 v49, v[52:53], off                      // 00000000269c: ee07807c 00000031 00000034
	global_load_d16_hi_u8 v49, v[58:59], off                   // 0000000026a8: ee08407c 00000031 0000003a
	global_load_d16_u8 v50, v[62:63], off                      // 0000000026b4: ee07807c 00000032 0000003e
	global_load_d16_hi_u8 v50, v[60:61], off                   // 0000000026c0: ee08407c 00000032 0000003c
	global_load_d16_u8 v51, v[54:55], off                      // 0000000026cc: ee07807c 00000033 00000036
	global_load_d16_hi_u8 v51, v[56:57], off                   // 0000000026d8: ee08407c 00000033 00000038
	s_lshr_b32 s34, s61, 5                                     // 0000000026e4: 8522853d
	s_add_nc_u64 s[60:61], s[60:61], 32                        // 0000000026e8: a9bca03c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026ec: bf88ff9e
	s_mul_i32 s35, s34, s38                                    // 0000000026f0: 96232622
	v_add_co_u32 v52, s34, v7, s62                             // 0000000026f4: d7002234 02007d07
	s_wait_alu depctr_va_sdst(0)                               // 0000000026fc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s63, v8, s34                // 000000002700: d5207c35 008a103f
	v_add_co_u32 v54, s34, v9, s62                             // 000000002708: d7002236 02007d09
	s_wait_alu depctr_sa_sdst(0)                               // 000000002710: bf88ff9e
	v_add3_u32 v65, s64, s35, v65                              // 000000002714: d6550041 05044640
	s_wait_alu depctr_va_sdst(0)                               // 00000000271c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s63, v10, s34               // 000000002720: d5207c37 008a143f
	global_load_u8 v56, v[52:53], off                          // 000000002728: ee04007c 00000038 00000034
	global_load_u8 v57, v[64:65], off                          // 000000002734: ee04007c 00000039 00000040
	global_load_u8 v58, v[54:55], off                          // 000000002740: ee04007c 0000003a 00000036
	v_add_co_u32 v52, s34, v11, s62                            // 00000000274c: d7002234 02007d0b
	s_wait_alu depctr_va_sdst(0)                               // 000000002754: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s63, v12, s34               // 000000002758: d5207c35 008a183f
	v_add_co_u32 v54, s34, v13, s62                            // 000000002760: d7002236 02007d0d
	s_wait_alu depctr_va_sdst(0)                               // 000000002768: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s63, v14, s34               // 00000000276c: d5207c37 008a1c3f
	s_clause 0x1                                               // 000000002774: bf850001
	global_load_u8 v59, v[52:53], off                          // 000000002778: ee04007c 0000003b 00000034
	global_load_u8 v60, v[54:55], off                          // 000000002784: ee04007c 0000003c 00000036
	v_add_co_u32 v52, s34, v15, s62                            // 000000002790: d7002234 02007d0f
	s_wait_alu depctr_va_sdst(0)                               // 000000002798: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s63, v16, s34               // 00000000279c: d5207c35 008a203f
	v_add_co_u32 v54, s34, v17, s62                            // 0000000027a4: d7002236 02007d11
	s_wait_alu depctr_va_sdst(0)                               // 0000000027ac: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s63, v18, s34               // 0000000027b0: d5207c37 008a243f
	s_clause 0x1                                               // 0000000027b8: bf850001
	global_load_u8 v61, v[52:53], off                          // 0000000027bc: ee04007c 0000003d 00000034
	global_load_u8 v62, v[54:55], off                          // 0000000027c8: ee04007c 0000003e 00000036
	v_add_co_u32 v52, s34, v19, s62                            // 0000000027d4: d7002234 02007d13
	s_wait_alu depctr_va_sdst(0)                               // 0000000027dc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s63, v20, s34               // 0000000027e0: d5207c35 008a283f
	v_add_co_u32 v54, s34, v21, s62                            // 0000000027e8: d7002236 02007d15
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f0: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s63, v22, s34               // 0000000027f4: d5207c37 008a2c3f
	s_clause 0x1                                               // 0000000027fc: bf850001
	global_load_u8 v63, v[52:53], off                          // 000000002800: ee04007c 0000003f 00000034
	global_load_u8 v64, v[54:55], off                          // 00000000280c: ee04007c 00000040 00000036
	s_wait_loadcnt 0x27                                        // 000000002818: bfc00027
	v_cndmask_b16 v36.l, 0, v36.l, vcc_lo                      // 00000000281c: d65d0024 01aa4880
	v_cndmask_b16 v36.h, 0, v36.h, s2                          // 000000002824: d65d5024 000a4880
	s_wait_loadcnt 0x25                                        // 00000000282c: bfc00025
	v_cndmask_b16 v37.l, 0, v37.l, s3                          // 000000002830: d65d0025 000e4a80
	v_cndmask_b16 v37.h, 0, v37.h, s4                          // 000000002838: d65d5025 00124a80
	s_wait_loadcnt 0x23                                        // 000000002840: bfc00023
	v_cndmask_b16 v38.l, 0, v38.l, s5                          // 000000002844: d65d0026 00164c80
	v_cndmask_b16 v38.h, 0, v38.h, s6                          // 00000000284c: d65d5026 001a4c80
	s_wait_loadcnt 0x21                                        // 000000002854: bfc00021
	v_cndmask_b16 v39.l, 0, v39.l, s7                          // 000000002858: d65d0027 001e4e80
	v_cndmask_b16 v39.h, 0, v39.h, s8                          // 000000002860: d65d5027 00224e80
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002868: d7385025 02024a88
	v_and_b16 v38.l, 0xff, v38.l                               // 000000002870: d7620026 02024cff 000000ff
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 00000000287c: d7385026 02024c88
	v_and_b16 v39.l, 0xff, v39.l                               // 000000002884: d7620027 02024eff 000000ff
	v_lshlrev_b16 v39.h, 8, v39.h op_sel:[0,1,1]               // 000000002890: d7385027 02024e88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002898: d7620025 02024aff 000000ff
	v_lshlrev_b16 v36.h, 8, v36.h op_sel:[0,1,1]               // 0000000028a4: d7385024 02024888
	v_or_b16 v53.l, v38.l, v38.h op_sel:[0,1,0]                // 0000000028ac: d7631035 02024d26
	v_and_b16 v36.l, 0xff, v36.l                               // 0000000028b4: d7620024 020248ff 000000ff
	v_or_b16 v53.h, v39.l, v39.h op_sel:[0,1,1]                // 0000000028c0: d7635035 02024f27
	v_or_b16 v52.h, v37.l, v37.h op_sel:[0,1,1]                // 0000000028c8: d7635034 02024b25
	s_delay_alu instid0(valu_dep_3)                            // 0000000028d0: bf870003
	v_or_b16 v52.l, v36.l, v36.h op_sel:[0,1,0]                // 0000000028d4: d7631034 02024924
	s_wait_loadcnt 0x1f                                        // 0000000028dc: bfc0001f
	v_cndmask_b16 v37.l, 0, v40.l, s9                          // 0000000028e0: d65d0025 00265080
	v_cndmask_b16 v37.h, 0, v40.h, s10                         // 0000000028e8: d65d5025 002a5080
	s_wait_loadcnt 0x1d                                        // 0000000028f0: bfc0001d
	v_cndmask_b16 v38.l, 0, v41.l, s12                         // 0000000028f4: d65d0026 00325280
	v_cndmask_b16 v40.h, 0, v41.h, s11                         // 0000000028fc: d65d5028 002e5280
	s_wait_loadcnt 0x1b                                        // 000000002904: bfc0001b
	v_cndmask_b16 v40.l, 0, v42.l, s13                         // 000000002908: d65d0028 00365480
	v_cndmask_b16 v39.h, 0, v42.h, s14                         // 000000002910: d65d5027 003a5480
	s_wait_loadcnt 0x19                                        // 000000002918: bfc00019
	v_cndmask_b16 v39.l, 0, v43.l, s15                         // 00000000291c: d65d0027 003e5680
	v_cndmask_b16 v38.h, 0, v43.h, s16                         // 000000002924: d65d5026 00425680
	v_lshlrev_b16 v40.h, 8, v40.h op_sel:[0,1,1]               // 00000000292c: d7385028 02025088
	v_and_b16 v40.l, 0xff, v40.l                               // 000000002934: d7620028 020250ff 000000ff
	v_lshlrev_b16 v39.h, 8, v39.h op_sel:[0,1,1]               // 000000002940: d7385027 02024e88
	v_and_b16 v39.l, 0xff, v39.l                               // 000000002948: d7620027 02024eff 000000ff
	v_lshlrev_b16 v38.h, 8, v38.h op_sel:[0,1,1]               // 000000002954: d7385026 02024c88
	v_and_b16 v38.l, 0xff, v38.l                               // 00000000295c: d7620026 02024cff 000000ff
	v_lshlrev_b16 v37.h, 8, v37.h op_sel:[0,1,1]               // 000000002968: d7385025 02024a88
	v_and_b16 v37.l, 0xff, v37.l                               // 000000002970: d7620025 02024aff 000000ff
	v_or_b16 v55.l, v40.l, v39.h op_sel:[0,1,0]                // 00000000297c: d7631037 02024f28
	v_or_b16 v55.h, v39.l, v38.h op_sel:[0,1,1]                // 000000002984: d7635037 02024d27
	v_or_b16 v54.h, v38.l, v40.h op_sel:[0,1,1]                // 00000000298c: d7635036 02025126
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000002994: bf870094
	v_or_b16 v54.l, v37.l, v37.h op_sel:[0,1,0]                // 000000002998: d7631036 02024b25
	v_wmma_f32_16x16x16_fp8_fp8 v[36:43], v[52:53], v[54:55], 0// 0000000029a0: cc464024 1a026d34
	s_wait_loadcnt 0x17                                        // 0000000029a8: bfc00017
	v_cndmask_b16 v44.l, 0, v44.l, s17                         // 0000000029ac: d65d002c 00465880
	v_cndmask_b16 v44.h, 0, v44.h, s18                         // 0000000029b4: d65d502c 004a5880
	s_wait_loadcnt 0x15                                        // 0000000029bc: bfc00015
	v_cndmask_b16 v45.l, 0, v45.l, s19                         // 0000000029c0: d65d002d 004e5a80
	v_cndmask_b16 v45.h, 0, v45.h, s20                         // 0000000029c8: d65d502d 00525a80
	s_wait_loadcnt 0x13                                        // 0000000029d0: bfc00013
	v_cndmask_b16 v46.l, 0, v46.l, s21                         // 0000000029d4: d65d002e 00565c80
	v_cndmask_b16 v46.h, 0, v46.h, s22                         // 0000000029dc: d65d502e 005a5c80
	s_wait_loadcnt 0x11                                        // 0000000029e4: bfc00011
	v_cndmask_b16 v47.l, 0, v47.l, s23                         // 0000000029e8: d65d002f 005e5e80
	v_cndmask_b16 v47.h, 0, v47.h, s24                         // 0000000029f0: d65d502f 00625e80
	v_lshlrev_b16 v45.h, 8, v45.h op_sel:[0,1,1]               // 0000000029f8: d738502d 02025a88
	v_and_b16 v46.l, 0xff, v46.l                               // 000000002a00: d762002e 02025cff 000000ff
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000002a0c: d738502e 02025c88
	v_and_b16 v47.l, 0xff, v47.l                               // 000000002a14: d762002f 02025eff 000000ff
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000002a20: d738502f 02025e88
	v_and_b16 v45.l, 0xff, v45.l                               // 000000002a28: d762002d 02025aff 000000ff
	v_lshlrev_b16 v44.h, 8, v44.h op_sel:[0,1,1]               // 000000002a34: d738502c 02025888
	v_or_b16 v53.l, v46.l, v46.h op_sel:[0,1,0]                // 000000002a3c: d7631035 02025d2e
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002a44: d762002c 020258ff 000000ff
	v_or_b16 v53.h, v47.l, v47.h op_sel:[0,1,1]                // 000000002a50: d7635035 02025f2f
	v_or_b16 v52.h, v45.l, v45.h op_sel:[0,1,1]                // 000000002a58: d7635034 02025b2d
	s_delay_alu instid0(valu_dep_3)                            // 000000002a60: bf870003
	v_or_b16 v52.l, v44.l, v44.h op_sel:[0,1,0]                // 000000002a64: d7631034 0202592c
	s_wait_loadcnt 0xf                                         // 000000002a6c: bfc0000f
	v_cndmask_b16 v45.l, 0, v48.l, s25                         // 000000002a70: d65d002d 00666080
	v_cndmask_b16 v45.h, 0, v48.h, s26                         // 000000002a78: d65d502d 006a6080
	s_wait_loadcnt 0xd                                         // 000000002a80: bfc0000d
	v_cndmask_b16 v46.l, 0, v49.l, s28                         // 000000002a84: d65d002e 00726280
	v_cndmask_b16 v47.h, 0, v49.h, s30                         // 000000002a8c: d65d502f 007a6280
	s_wait_loadcnt 0xb                                         // 000000002a94: bfc0000b
	v_cndmask_b16 v46.h, 0, v50.l, s33                         // 000000002a98: d65d402e 00866480
	v_cndmask_b16 v47.l, 0, v50.h, s31                         // 000000002aa0: d65d102f 007e6480
	s_wait_loadcnt 0x9                                         // 000000002aa8: bfc00009
	v_cndmask_b16 v48.h, 0, v51.l, s27                         // 000000002aac: d65d4030 006e6680
	v_cndmask_b16 v48.l, 0, v51.h, s29                         // 000000002ab4: d65d1030 00766680
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000002abc: d738502f 02025e88
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000002ac4: d738502e 02025c88
	v_and_b16 v47.l, 0xff, v47.l                               // 000000002acc: d762002f 02025eff 000000ff
	v_lshlrev_b16 v48.h, 8, v48.h op_sel:[0,1,1]               // 000000002ad8: d7385030 02026088
	v_and_b16 v48.l, 0xff, v48.l                               // 000000002ae0: d7620030 020260ff 000000ff
	v_and_b16 v46.l, 0xff, v46.l                               // 000000002aec: d762002e 02025cff 000000ff
	v_lshlrev_b16 v45.h, 8, v45.h op_sel:[0,1,1]               // 000000002af8: d738502d 02025a88
	v_and_b16 v45.l, 0xff, v45.l                               // 000000002b00: d762002d 02025aff 000000ff
	v_or_b16 v49.h, v47.l, v46.h op_sel:[0,1,1]                // 000000002b0c: d7635031 02025d2f
	v_or_b16 v49.l, v48.l, v47.h op_sel:[0,1,0]                // 000000002b14: d7631031 02025f30
	v_or_b16 v48.h, v46.l, v48.h op_sel:[0,1,1]                // 000000002b1c: d7635030 0202612e
	s_wait_loadcnt 0x8                                         // 000000002b24: bfc00008
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v56                         // 000000002b28: 7c9470ff 000000ff
	v_or_b16 v48.l, v45.l, v45.h op_sel:[0,1,0]                // 000000002b30: d7631030 02025b2d
	s_wait_loadcnt 0x7                                         // 000000002b38: bfc00007
	v_add_nc_u32_e32 v44, 0xffffff02, v57                      // 000000002b3c: 4a5872ff ffffff02
	v_cmp_eq_u32_e64 s2, 0xff, v57                             // 000000002b44: d44a0002 020272ff 000000ff
	s_wait_loadcnt 0x6                                         // 000000002b50: bfc00006
	v_cmp_eq_u32_e64 s3, 0xff, v58                             // 000000002b54: d44a0003 020274ff 000000ff
	v_wmma_f32_16x16x16_fp8_fp8 v[36:43], v[52:53], v[48:49], v[36:43]// 000000002b60: cc464024 1c926134
	v_add_nc_u32_e32 v45, v44, v56                             // 000000002b68: 4a5a712c
	s_or_b32 s4, vcc_lo, s2                                    // 000000002b6c: 8c04026a
	s_or_b32 s3, s2, s3                                        // 000000002b70: 8c030302
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 000000002b74: bf870141
	v_ldexp_f32 v36, v36, v45                                  // 000000002b78: d71c0024 02025b24
	s_wait_loadcnt 0x5                                         // 000000002b80: bfc00005
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v59                         // 000000002b84: 7c9476ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b8c: bf88ff9e
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s4                 // 000000002b90: d5010024 0011ff24 7fc00000
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b9c: bf870121
	v_dual_add_f32 v35, v35, v36 :: v_dual_add_nc_u32 v46, v44, v58// 000000002ba0: c9204923 232e752c
	v_add_nc_u32_e32 v45, v44, v59                             // 000000002ba8: 4a5a772c
	v_ldexp_f32 v37, v37, v46                                  // 000000002bac: d71c0025 02025d25
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002bb4: bf870112
	v_ldexp_f32 v38, v38, v45                                  // 000000002bb8: d71c0026 02025b26
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s3                 // 000000002bc0: d5010025 000dff25 7fc00000
	s_or_b32 s3, s2, vcc_lo                                    // 000000002bcc: 8c036a02
	s_wait_loadcnt 0x4                                         // 000000002bd0: bfc00004
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v60                         // 000000002bd4: 7c9478ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bdc: bf88ff9e
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s3                 // 000000002be0: d5010026 000dff26 7fc00000
	v_add_nc_u32_e32 v36, v44, v60                             // 000000002bec: 4a48792c
	v_add_f32_e32 v34, v34, v37                                // 000000002bf0: 06444b22
	s_wait_loadcnt 0x3                                         // 000000002bf4: bfc00003
	v_add_nc_u32_e32 v37, v44, v61                             // 000000002bf8: 4a4a7b2c
	v_cmp_eq_u32_e64 s3, 0xff, v61                             // 000000002bfc: d44a0003 02027aff 000000ff
	v_add_f32_e32 v33, v33, v38                                // 000000002c08: 06424d21
	v_ldexp_f32 v36, v39, v36                                  // 000000002c0c: d71c0024 02024927
	s_or_b32 s4, s2, vcc_lo                                    // 000000002c14: 8c046a02
	v_ldexp_f32 v37, v40, v37                                  // 000000002c18: d71c0025 02024b28
	s_wait_loadcnt 0x2                                         // 000000002c20: bfc00002
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v62                         // 000000002c24: 7c947cff 000000ff
	s_wait_loadcnt 0x0                                         // 000000002c2c: bfc00000
	v_add_nc_u32_e32 v40, v44, v64                             // 000000002c30: 4a50812c
	s_or_b32 s3, s2, s3                                        // 000000002c34: 8c030302
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c38: bf88ff9e
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s4                 // 000000002c3c: d5010024 0011ff24 7fc00000
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s3                 // 000000002c48: d5010025 000dff25 7fc00000
	v_cmp_eq_u32_e64 s3, 0xff, v63                             // 000000002c54: d44a0003 02027eff 000000ff
	s_or_b32 s4, s2, vcc_lo                                    // 000000002c60: 8c046a02
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v64                         // 000000002c64: 7c9480ff 000000ff
	v_ldexp_f32 v40, v43, v40                                  // 000000002c6c: d71c0028 0202512b
	v_add_nc_u32_e32 v38, v44, v62                             // 000000002c74: 4a4c7d2c
	s_or_b32 s3, s2, s3                                        // 000000002c78: 8c030302
	v_add_f32_e32 v32, v32, v36                                // 000000002c7c: 06404920
	s_or_b32 s2, s2, vcc_lo                                    // 000000002c80: 8c026a02
	v_add_f32_e32 v27, v27, v37                                // 000000002c84: 06364b1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c88: bf88ff9e
	v_cndmask_b32_e64 v40, v40, 0x7fc00000, s2                 // 000000002c8c: d5010028 0009ff28 7fc00000
	v_add_nc_u32_e32 v39, v44, v63                             // 000000002c98: 4a4e7f2c
	v_ldexp_f32 v38, v41, v38                                  // 000000002c9c: d71c0026 02024d29
	v_cmp_lt_u64_e64 s2, s[60:61], s[42:43]                    // 000000002ca4: d4590002 0200543c
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002cac: bf870214
	v_add_f32_e32 v6, v6, v40                                  // 000000002cb0: 060c5106
	v_ldexp_f32 v39, v42, v39                                  // 000000002cb4: d71c0027 02024f2a
	s_delay_alu instid0(valu_dep_4) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cbc: bf870124
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s4                 // 000000002cc0: d5010026 0011ff26 7fc00000
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000002ccc: 8b6a027e
	v_cndmask_b32_e64 v39, v39, 0x7fc00000, s3                 // 000000002cd0: d5010027 000dff27 7fc00000
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002cdc: bf870112
	v_add_f32_e32 v26, v26, v38                                // 000000002ce0: 06344d1a
	v_add_f32_e32 v2, v2, v39                                  // 000000002ce4: 06044f02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ce8: bf88ff9e
	s_cbranch_vccnz 64597                                      // 000000002cec: bfa4fc55 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x344>
	v_mul_lo_u32 v9, s39, v0                                   // 000000002cf0: d72c0009 02020027
	v_mul_lo_u32 v10, s38, v1                                  // 000000002cf8: d72c000a 02020226
	v_mad_co_u64_u32 v[7:8], null, s38, v0, 0                  // 000000002d00: d6fe7c07 02020026
	v_sub_co_u32 v0, vcc_lo, s36, v0                           // 000000002d08: d7016a00 02020024
	s_wait_alu depctr_va_vcc(0)                                // 000000002d10: bf88ff9d
	v_sub_co_ci_u32_e64 v1, null, s37, v1, vcc_lo              // 000000002d14: d5217c01 01aa0225
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000002d1c: bf870211
	v_cmp_lt_i64_e32 vcc_lo, 0, v[0:1]                         // 000000002d20: 7ca20080
	v_add3_u32 v8, v8, v10, v9                                 // 000000002d24: d6550008 04261508
	s_delay_alu instid0(valu_dep_1)                            // 000000002d2c: bf870001
	v_lshlrev_b64_e32 v[7:8], 1, v[7:8]                        // 000000002d30: 3e0e0e81
	s_and_b32 s2, vcc_lo, s0                                   // 000000002d34: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d38: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000002d3c: be812002
	s_cbranch_execz 25                                         // 000000002d40: bfa50019 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x12a8>
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002d44: 3e120681
	v_add_co_u32 v12, vcc_lo, s40, v7                          // 000000002d48: d7006a0c 02020e28
	v_bfe_u32 v11, v35, 16, 1                                  // 000000002d50: d610000b 02052123
	s_wait_alu depctr_va_vcc(0)                                // 000000002d58: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s41, v8, vcc_lo             // 000000002d5c: d5207c0d 01aa1029
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002d64: bf870193
	v_add_co_u32 v9, vcc_lo, v12, v9                           // 000000002d68: d7006a09 0202130c
	v_add3_u32 v11, v11, v35, 0x7fff                           // 000000002d70: d655000b 03fe470b 00007fff
	v_or_b32_e32 v14, 0x400000, v35                            // 000000002d7c: 381c46ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002d84: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v13, v10, vcc_lo            // 000000002d88: d5207c0a 01aa150d
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000002d90: 7c304723
	s_wait_alu depctr_va_vcc(0)                                // 000000002d94: bf88ff9d
	v_cndmask_b32_e32 v11, v11, v14, vcc_lo                    // 000000002d98: 02161d0b
	global_store_d16_hi_b16 v[9:10], v11, off                  // 000000002d9c: ee09407c 05800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002da8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002dac: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 1, v[0:1]                         // 000000002db0: 7ca20081
	s_and_b32 s2, vcc_lo, s0                                   // 000000002db4: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002db8: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000002dbc: be812002
	s_cbranch_execz 32                                         // 000000002dc0: bfa50020 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1344>
	v_add_co_u32 v11, vcc_lo, s40, v7                          // 000000002dc4: d7006a0b 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 000000002dcc: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s41, v8, vcc_lo             // 000000002dd0: d5207c0c 01aa1029
	s_lshl_b64 s[2:3], s[38:39], 1                             // 000000002dd8: 84828126
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002ddc: 3e120681
	s_wait_alu depctr_sa_sdst(0)                               // 000000002de0: bf88ff9e
	v_add_co_u32 v11, vcc_lo, v11, s2                          // 000000002de4: d7006a0b 0200050b
	v_bfe_u32 v13, v34, 16, 1                                  // 000000002dec: d610000d 02052122
	s_wait_alu depctr_va_vcc(0)                                // 000000002df4: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s3, v12, vcc_lo             // 000000002df8: d5207c0c 01aa1803
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002e00: bf870193
	v_add_co_u32 v9, vcc_lo, v11, v9                           // 000000002e04: d7006a09 0202130b
	v_add3_u32 v13, v13, v34, 0x7fff                           // 000000002e0c: d655000d 03fe450d 00007fff
	v_or_b32_e32 v14, 0x400000, v34                            // 000000002e18: 381c44ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002e20: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v12, v10, vcc_lo            // 000000002e24: d5207c0a 01aa150c
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 000000002e2c: 7c304522
	s_wait_alu depctr_va_vcc(0)                                // 000000002e30: bf88ff9d
	v_cndmask_b32_e32 v11, v13, v14, vcc_lo                    // 000000002e34: 02161d0d
	global_store_d16_hi_b16 v[9:10], v11, off                  // 000000002e38: ee09407c 05800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e48: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 2, v[0:1]                         // 000000002e4c: 7ca20082
	s_and_b32 s2, vcc_lo, s0                                   // 000000002e50: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e54: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000002e58: be812002
	s_cbranch_execz 31                                         // 000000002e5c: bfa5001f <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x13dc>
	v_add_co_u32 v13, vcc_lo, s40, v7                          // 000000002e60: d7006a0d 02020e28
	v_bfe_u32 v9, v33, 16, 1                                   // 000000002e68: d6100009 02052121
	s_wait_alu depctr_va_vcc(0)                                // 000000002e70: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s41, v8, vcc_lo             // 000000002e74: d5207c0e 01aa1029
	s_lshl_b64 s[2:3], s[38:39], 2                             // 000000002e7c: 84828226
	v_or_b32_e32 v11, 0x400000, v33                            // 000000002e80: 381642ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e88: bf88ff9e
	v_add_co_u32 v13, vcc_lo, v13, s2                          // 000000002e8c: d7006a0d 0200050d
	v_add3_u32 v12, v9, v33, 0x7fff                            // 000000002e94: d655000c 03fe4309 00007fff
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002ea0: 3e120681
	s_wait_alu depctr_va_vcc(0)                                // 000000002ea4: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s3, v14, vcc_lo             // 000000002ea8: d5207c0e 01aa1c03
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000002eb0: 7c304321
	s_wait_alu depctr_va_vcc(0)                                // 000000002eb4: bf88ff9d
	v_cndmask_b32_e32 v11, v12, v11, vcc_lo                    // 000000002eb8: 0216170c
	v_add_co_u32 v9, vcc_lo, v13, v9                           // 000000002ebc: d7006a09 0202130d
	s_wait_alu depctr_va_vcc(0)                                // 000000002ec4: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v14, v10, vcc_lo            // 000000002ec8: d5207c0a 01aa150e
	global_store_d16_hi_b16 v[9:10], v11, off                  // 000000002ed0: ee09407c 05800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002edc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ee0: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 3, v[0:1]                         // 000000002ee4: 7ca20083
	s_and_b32 s2, vcc_lo, s0                                   // 000000002ee8: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002eec: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000002ef0: be812002
	s_cbranch_execz 31                                         // 000000002ef4: bfa5001f <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x1474>
	v_add_co_u32 v9, vcc_lo, s40, v7                           // 000000002ef8: d7006a09 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 000000002f00: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s41, v8, vcc_lo             // 000000002f04: d5207c0a 01aa1029
	v_bfe_u32 v11, v32, 16, 1                                  // 000000002f0c: d610000b 02052120
	v_or_b32_e32 v13, 0x400000, v32                            // 000000002f14: 381a40ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000002f1c: 7c304120
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002f20: bf870214
	v_mad_co_u64_u32 v[9:10], null, s38, 6, v[9:10]            // 000000002f24: d6fe7c09 04250c26
	v_add3_u32 v14, v11, v32, 0x7fff                           // 000000002f2c: d655000e 03fe410b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002f38: bf88ff9d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002f3c: bf870191
	v_cndmask_b32_e32 v13, v14, v13, vcc_lo                    // 000000002f40: 021a1b0e
	v_mad_co_u64_u32 v[10:11], null, s39, 6, v[10:11]          // 000000002f44: d6fe7c0a 04290c27
	v_lshlrev_b64_e32 v[11:12], 1, v[3:4]                      // 000000002f4c: 3e160681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f50: bf870121
	v_add_co_u32 v9, vcc_lo, v9, v11                           // 000000002f54: d7006a09 02021709
	s_wait_alu depctr_va_vcc(0)                                // 000000002f5c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo            // 000000002f60: d5207c0a 01aa190a
	global_store_d16_hi_b16 v[9:10], v13, off                  // 000000002f68: ee09407c 06800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f78: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 4, v[0:1]                         // 000000002f7c: 7ca20084
	s_and_b32 s2, vcc_lo, s0                                   // 000000002f80: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f84: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000002f88: be812002
	s_cbranch_execz 31                                         // 000000002f8c: bfa5001f <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x150c>
	v_add_co_u32 v13, vcc_lo, s40, v7                          // 000000002f90: d7006a0d 02020e28
	v_bfe_u32 v9, v27, 16, 1                                   // 000000002f98: d6100009 0205211b
	s_wait_alu depctr_va_vcc(0)                                // 000000002fa0: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s41, v8, vcc_lo             // 000000002fa4: d5207c0e 01aa1029
	s_lshl_b64 s[2:3], s[38:39], 3                             // 000000002fac: 84828326
	v_or_b32_e32 v11, 0x400000, v27                            // 000000002fb0: 381636ff 00400000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb8: bf88ff9e
	v_add_co_u32 v13, vcc_lo, v13, s2                          // 000000002fbc: d7006a0d 0200050d
	v_add3_u32 v12, v9, v27, 0x7fff                            // 000000002fc4: d655000c 03fe3709 00007fff
	v_lshlrev_b64_e32 v[9:10], 1, v[3:4]                       // 000000002fd0: 3e120681
	s_wait_alu depctr_va_vcc(0)                                // 000000002fd4: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s3, v14, vcc_lo             // 000000002fd8: d5207c0e 01aa1c03
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000002fe0: 7c30371b
	s_wait_alu depctr_va_vcc(0)                                // 000000002fe4: bf88ff9d
	v_cndmask_b32_e32 v11, v12, v11, vcc_lo                    // 000000002fe8: 0216170c
	v_add_co_u32 v9, vcc_lo, v13, v9                           // 000000002fec: d7006a09 0202130d
	s_wait_alu depctr_va_vcc(0)                                // 000000002ff4: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v14, v10, vcc_lo            // 000000002ff8: d5207c0a 01aa150e
	global_store_d16_hi_b16 v[9:10], v11, off                  // 000000003000: ee09407c 05800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 00000000300c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003010: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 5, v[0:1]                         // 000000003014: 7ca20085
	s_and_b32 s2, vcc_lo, s0                                   // 000000003018: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000003020: be812002
	s_cbranch_execz 31                                         // 000000003024: bfa5001f <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x15a4>
	v_add_co_u32 v9, vcc_lo, s40, v7                           // 000000003028: d7006a09 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 000000003030: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s41, v8, vcc_lo             // 000000003034: d5207c0a 01aa1029
	v_bfe_u32 v11, v26, 16, 1                                  // 00000000303c: d610000b 0205211a
	v_or_b32_e32 v13, 0x400000, v26                            // 000000003044: 381a34ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 00000000304c: 7c30351a
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003050: bf870214
	v_mad_co_u64_u32 v[9:10], null, s38, 10, v[9:10]           // 000000003054: d6fe7c09 04251426
	v_add3_u32 v14, v11, v26, 0x7fff                           // 00000000305c: d655000e 03fe350b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003068: bf88ff9d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 00000000306c: bf870191
	v_cndmask_b32_e32 v13, v14, v13, vcc_lo                    // 000000003070: 021a1b0e
	v_mad_co_u64_u32 v[10:11], null, s39, 10, v[10:11]         // 000000003074: d6fe7c0a 04291427
	v_lshlrev_b64_e32 v[11:12], 1, v[3:4]                      // 00000000307c: 3e160681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003080: bf870121
	v_add_co_u32 v9, vcc_lo, v9, v11                           // 000000003084: d7006a09 02021709
	s_wait_alu depctr_va_vcc(0)                                // 00000000308c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo            // 000000003090: d5207c0a 01aa190a
	global_store_d16_hi_b16 v[9:10], v13, off                  // 000000003098: ee09407c 06800000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030a8: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 6, v[0:1]                         // 0000000030ac: 7ca20086
	s_and_b32 s2, vcc_lo, s0                                   // 0000000030b0: 8b02006a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030b4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 0000000030b8: be812002
	s_cbranch_execz 31                                         // 0000000030bc: bfa5001f <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x163c>
	v_add_co_u32 v9, vcc_lo, s40, v7                           // 0000000030c0: d7006a09 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 0000000030c8: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s41, v8, vcc_lo             // 0000000030cc: d5207c0a 01aa1029
	v_bfe_u32 v11, v2, 16, 1                                   // 0000000030d4: d610000b 02052102
	v_or_b32_e32 v13, 0x400000, v2                             // 0000000030dc: 381a04ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 0000000030e4: 7c300502
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000030e8: bf870214
	v_mad_co_u64_u32 v[9:10], null, s38, 12, v[9:10]           // 0000000030ec: d6fe7c09 04251826
	v_add3_u32 v14, v11, v2, 0x7fff                            // 0000000030f4: d655000e 03fe050b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003100: bf88ff9d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003104: bf870191
	v_cndmask_b32_e32 v2, v14, v13, vcc_lo                     // 000000003108: 02041b0e
	v_mad_co_u64_u32 v[10:11], null, s39, 12, v[10:11]         // 00000000310c: d6fe7c0a 04291827
	v_lshlrev_b64_e32 v[11:12], 1, v[3:4]                      // 000000003114: 3e160681
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003118: bf870121
	v_add_co_u32 v9, vcc_lo, v9, v11                           // 00000000311c: d7006a09 02021709
	s_wait_alu depctr_va_vcc(0)                                // 000000003124: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo            // 000000003128: d5207c0a 01aa190a
	global_store_d16_hi_b16 v[9:10], v2, off                   // 000000003130: ee09407c 01000000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 00000000313c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003140: 8c7e017e
	v_cmp_lt_i64_e32 vcc_lo, 7, v[0:1]                         // 000000003144: 7ca20087
	s_mov_b32 s1, 0                                            // 000000003148: be810080
	s_and_b32 s2, vcc_lo, s0                                   // 00000000314c: 8b02006a
	s_mov_b32 s0, 0                                            // 000000003150: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003154: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003158: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000315c: bf88ff9e
	s_xor_b32 s2, exec_lo, s3                                  // 000000003160: 8d02037e
	v_add_co_u32 v0, vcc_lo, s40, v7                           // 000000003164: d7006a00 02020e28
	s_wait_alu depctr_va_vcc(0)                                // 00000000316c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s41, v8, vcc_lo              // 000000003170: d5207c01 01aa1029
	s_mov_b32 s0, exec_lo                                      // 000000003178: be80007e
	v_mad_co_u64_u32 v[0:1], null, s38, 14, v[0:1]             // 00000000317c: d6fe7c00 04011c26
	s_delay_alu instid0(valu_dep_1)                            // 000000003184: bf870001
	v_mad_co_u64_u32 v[1:2], null, s39, 14, v[1:2]             // 000000003188: d6fe7c01 04051c27
	s_wait_alu depctr_sa_sdst(0)                               // 000000003190: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003194: 8c7e027e
	s_delay_alu instid0(salu_cycle_1)                          // 000000003198: bf870009
	s_and_b32 vcc_lo, exec_lo, s1                              // 00000000319c: 8b6a017e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031a0: bf88ff9e
	s_cbranch_vccnz 3                                          // 0000000031a4: bfa40003 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x16b4>
	s_branch 601                                               // 0000000031a8: bfa00259 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2010>
	s_mov_b32 s0, 0                                            // 0000000031ac: be800080
	s_cbranch_execz 599                                        // 0000000031b0: bfa50257 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2010>
	v_or_b32_e32 v0, s50, v5                                   // 0000000031b4: 38000a32
	v_cmp_gt_i64_e32 vcc_lo, s[38:39], v[3:4]                  // 0000000031b8: 7ca80626
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v1, s51             // 0000000031bc: ca100080 06000033
	v_mov_b32_e32 v8, s51                                      // 0000000031c4: 7e100233
	s_delay_alu instid0(valu_dep_4)                            // 0000000031c8: bf870004
	v_or_b32_e32 v7, 1, v0                                     // 0000000031cc: 380e0081
	s_wait_alu depctr_va_vcc(0)                                // 0000000031d0: bf88ff9d
	v_dual_mov_b32 v14, s51 :: v_dual_cndmask_b32 v11, 0, v3   // 0000000031d4: ca120033 0e0a0680
	v_cmp_gt_i64_e64 s0, s[36:37], v[0:1]                      // 0000000031dc: d4540000 02020024
	v_or_b32_e32 v13, 2, v0                                    // 0000000031e4: 381a0082
	v_cmp_gt_i64_e64 s1, s[36:37], v[7:8]                      // 0000000031e8: d4540001 02020e24
	v_mov_b32_e32 v20, s51                                     // 0000000031f0: 7e280233
	v_or_b32_e32 v22, 7, v0                                    // 0000000031f4: 382c0087
	s_mov_b64 s[10:11], 0                                      // 0000000031f8: be8a0180
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)             // 0000000031fc: bf88f19e
	v_cndmask_b32_e64 v9, 0, v0, s0                            // 000000003200: d5010009 00020080
	v_cndmask_b32_e64 v12, 0, s51, s0                          // 000000003208: d501000c 00006680
	v_cndmask_b32_e64 v10, 0, v7, s1                           // 000000003210: d501000a 00060e80
	v_cndmask_b32_e64 v15, 0, s51, s1                          // 000000003218: d501000f 00046680
	v_add_co_u32 v11, s0, s56, v11                             // 000000003220: d700000b 02021638
	v_mul_lo_u32 v16, s55, v9                                  // 000000003228: d72c0010 02021237
	v_mad_co_u64_u32 v[7:8], null, s54, v9, 0                  // 000000003230: d6fe7c07 02021236
	v_mul_lo_u32 v17, s55, v10                                 // 000000003238: d72c0011 02021437
	v_mad_co_u64_u32 v[9:10], null, s54, v10, 0                // 000000003240: d6fe7c09 02021436
	v_mul_lo_u32 v18, s54, v12                                 // 000000003248: d72c0012 02021836
	v_mul_lo_u32 v19, s54, v15                                 // 000000003250: d72c0013 02021e36
	v_or_b32_e32 v15, 3, v0                                    // 000000003258: 381e0083
	v_dual_cndmask_b32 v2, 0, v4 :: v_dual_mov_b32 v23, s51    // 00000000325c: ca500880 02160033
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003264: bf870214
	v_add3_u32 v8, v8, v18, v16                                // 000000003268: d6550008 04422508
	v_add3_u32 v10, v10, v19, v17                              // 000000003270: d655000a 0446270a
	v_or_b32_e32 v17, 4, v0                                    // 000000003278: 38220084
	v_mov_b32_e32 v18, s51                                     // 00000000327c: 7e240233
	s_wait_alu depctr_va_sdst(0)                               // 000000003280: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s57, v2, s0                 // 000000003284: d5207c0c 00020439
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_1)// 00000000328c: bf8700a2
	v_cmp_gt_i64_e64 s0, s[36:37], v[17:18]                    // 000000003290: d4540000 02022224
	s_wait_alu depctr_va_sdst(0)                               // 000000003298: bf88f19f
	v_cndmask_b32_e64 v21, 0, s51, s0                          // 00000000329c: d5010015 00006680
	v_cndmask_b32_e64 v17, 0, v17, s0                          // 0000000032a4: d5010011 00022280
	v_cmp_gt_i64_e64 s0, s[36:37], v[22:23]                    // 0000000032ac: d4540000 02022c24
	s_delay_alu instid0(valu_dep_3)                            // 0000000032b4: bf870003
	v_mul_lo_u32 v33, s54, v21                                 // 0000000032b8: d72c0021 02022a36
	v_mov_b32_e32 v21, s51                                     // 0000000032c0: 7e2a0233
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[13:14]                // 0000000032c4: 7ca81a24
	v_mul_lo_u32 v32, s55, v17                                 // 0000000032c8: d72c0020 02022237
	s_wait_alu depctr_va_sdst(0)                               // 0000000032d0: bf88f19f
	v_cndmask_b32_e64 v26, 0, s51, s0                          // 0000000032d4: d501001a 00006680
	s_wait_alu depctr_va_vcc(0)                                // 0000000032dc: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v13, vcc_lo                       // 0000000032e0: 02041a80
	v_cndmask_b32_e64 v19, 0, s51, vcc_lo                      // 0000000032e4: d5010013 01a86680
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000032ec: bf870193
	v_mul_lo_u32 v39, s54, v26                                 // 0000000032f0: d72c0027 02023436
	v_mul_lo_u32 v29, s55, v2                                  // 0000000032f8: d72c001d 02020437
	v_mad_co_u64_u32 v[13:14], null, s54, v2, 0                // 000000003300: d6fe7c0d 02020436
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_1)// 000000003308: bf8700b4
	v_mul_lo_u32 v2, s54, v19                                  // 00000000330c: d72c0002 02022636
	v_or_b32_e32 v19, 5, v0                                    // 000000003314: 38260085
	v_mov_b32_e32 v16, s51                                     // 000000003318: 7e200233
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[15:16]                // 00000000331c: 7ca81e24
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_2)// 000000003320: bf870154
	v_add3_u32 v14, v14, v2, v29                               // 000000003324: d655000e 0476050e
	v_mov_b32_e32 v29, v6                                      // 00000000332c: 7e3a0306
	s_wait_alu depctr_va_vcc(0)                                // 000000003330: bf88ff9d
	v_cndmask_b32_e64 v18, 0, s51, vcc_lo                      // 000000003334: d5010012 01a86680
	v_cndmask_b32_e32 v15, 0, v15, vcc_lo                      // 00000000333c: 021e1e80
	v_mul_lo_u32 v31, s54, v18                                 // 000000003340: d72c001f 02022436
	v_mad_co_u64_u32 v[17:18], null, s54, v17, 0               // 000000003348: d6fe7c11 02022236
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003350: bf8701a3
	v_mul_lo_u32 v30, s55, v15                                 // 000000003354: d72c001e 02021e37
	v_mad_co_u64_u32 v[15:16], null, s54, v15, 0               // 00000000335c: d6fe7c0f 02021e36
	v_add3_u32 v18, v18, v33, v32                              // 000000003364: d6550012 04824312
	v_mov_b32_e32 v33, v6                                      // 00000000336c: 7e420306
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[19:20]                // 000000003370: 7ca82624
	v_or_b32_e32 v20, 6, v0                                    // 000000003374: 38280086
	v_add3_u32 v16, v16, v31, v30                              // 000000003378: d6550010 047a3f10
	v_dual_mov_b32 v32, v6 :: v_dual_mov_b32 v31, v6           // 000000003380: ca100106 201e0106
	v_mov_b32_e32 v30, v6                                      // 000000003388: 7e3c0306
	s_wait_alu depctr_va_vcc(0)                                // 00000000338c: bf88ff9d
	v_cndmask_b32_e64 v24, 0, s51, vcc_lo                      // 000000003390: d5010018 01a86680
	v_cndmask_b32_e32 v19, 0, v19, vcc_lo                      // 000000003398: 02262680
	v_cmp_gt_i64_e32 vcc_lo, s[36:37], v[20:21]                // 00000000339c: 7ca82824
	s_delay_alu instid0(valu_dep_3)                            // 0000000033a0: bf870003
	v_mul_lo_u32 v35, s54, v24                                 // 0000000033a4: d72c0023 02023036
	v_cndmask_b32_e64 v24, 0, v22, s0                          // 0000000033ac: d5010018 00022c80
	v_add_co_u32 v27, s0, s52, v25                             // 0000000033b4: d700001b 02023234
	s_wait_alu depctr_va_sdst(0)                               // 0000000033bc: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s53, 0, s0                  // 0000000033c0: d5207c1c 00010035
	v_add_co_u32 v40, s0, s50, v25                             // 0000000033c8: d7000028 02023232
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d0: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s51, 0, s0                  // 0000000033d4: d5207c29 00010033
	v_mad_co_u64_u32 v[25:26], null, s42, v27, v[5:6]          // 0000000033dc: d6fe7c19 0416362a
	v_mul_lo_u32 v42, s42, v28                                 // 0000000033e4: d72c002a 0202382a
	v_mul_lo_u32 v43, s43, v27                                 // 0000000033ec: d72c002b 0202362b
	s_wait_alu depctr_va_vcc(0)                                // 0000000033f4: bf88ff9d
	v_cndmask_b32_e32 v21, 0, v20, vcc_lo                      // 0000000033f8: 022a2880
	v_cndmask_b32_e64 v23, 0, s51, vcc_lo                      // 0000000033fc: d5010017 01a86680
	v_mad_co_u64_u32 v[27:28], null, s42, v40, v[5:6]          // 000000003404: d6fe7c1b 0416502a
	v_mul_lo_u32 v5, s42, v41                                  // 00000000340c: d72c0005 0202522a
	v_mul_lo_u32 v40, s43, v40                                 // 000000003414: d72c0028 0202502b
	v_mul_lo_u32 v34, s55, v19                                 // 00000000341c: d72c0022 02022637
	v_mad_co_u64_u32 v[19:20], null, s54, v19, 0               // 000000003424: d6fe7c13 02022636
	v_mul_lo_u32 v36, s55, v21                                 // 00000000342c: d72c0024 02022a37
	v_mad_co_u64_u32 v[21:22], null, s54, v21, 0               // 000000003434: d6fe7c15 02022a36
	v_mul_lo_u32 v37, s54, v23                                 // 00000000343c: d72c0025 02022e36
	v_mul_lo_u32 v38, s55, v24                                 // 000000003444: d72c0026 02023037
	v_mad_co_u64_u32 v[23:24], null, s54, v24, 0               // 00000000344c: d6fe7c17 02023036
	v_add3_u32 v2, v43, v26, v42                               // 000000003454: d6550002 04aa352b
	v_add3_u32 v5, v40, v28, v5                                // 00000000345c: d6550005 04163928
	v_add_co_u32 v25, vcc_lo, s48, v25                         // 000000003464: d7006a19 02023230
	v_add3_u32 v20, v20, v35, v34                              // 00000000346c: d6550014 048a4714
	s_wait_alu depctr_va_vcc(0)                                // 000000003474: bf88ff9d
	v_add_co_ci_u32_e64 v26, null, s49, v2, vcc_lo             // 000000003478: d5207c1a 01aa0431
	v_add_co_u32 v27, vcc_lo, s46, v27                         // 000000003480: d7006a1b 0202362e
	v_add3_u32 v22, v22, v37, v36                              // 000000003488: d6550016 04924b16
	v_add3_u32 v24, v24, v39, v38                              // 000000003490: d6550018 049a4f18
	s_wait_alu depctr_va_vcc(0)                                // 000000003498: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s47, v5, vcc_lo             // 00000000349c: d5207c1c 01aa0a2f
	v_dual_mov_b32 v5, v6 :: v_dual_mov_b32 v2, v6             // 0000000034a4: ca100106 05020106
	v_add_co_u32 v34, vcc_lo, s44, v7                          // 0000000034ac: d7006a22 02020e2c
	s_wait_alu depctr_va_vcc(0)                                // 0000000034b4: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s45, v8, vcc_lo             // 0000000034b8: d5207c23 01aa102d
	v_add_co_u32 v36, vcc_lo, s44, v9                          // 0000000034c0: d7006a24 0202122c
	s_wait_alu depctr_va_vcc(0)                                // 0000000034c8: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s45, v10, vcc_lo            // 0000000034cc: d5207c25 01aa142d
	v_add_co_u32 v38, vcc_lo, s44, v13                         // 0000000034d4: d7006a26 02021a2c
	global_load_u8 v58, v[11:12], off                          // 0000000034dc: ee04007c 0000003a 0000000b
	s_wait_alu depctr_va_vcc(0)                                // 0000000034e8: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s45, v14, vcc_lo            // 0000000034ec: d5207c27 01aa1c2d
	v_add_co_u32 v40, vcc_lo, s44, v15                         // 0000000034f4: d7006a28 02021e2c
	s_clause 0x1                                               // 0000000034fc: bf850001
	global_load_b64 v[42:43], v[27:28], off                    // 000000003500: ee05407c 0000002a 0000001b
	global_load_b64 v[44:45], v[27:28], off offset:16          // 00000000350c: ee05407c 0000002c 0000101b
	s_clause 0x1                                               // 000000003518: bf850001
	global_load_b64 v[46:47], v[25:26], off                    // 00000000351c: ee05407c 0000002e 00000019
	global_load_b64 v[48:49], v[25:26], off offset:16          // 000000003528: ee05407c 00000030 00001019
	s_wait_alu depctr_va_vcc(0)                                // 000000003534: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s45, v16, vcc_lo            // 000000003538: d5207c29 01aa202d
	v_add_co_u32 v50, vcc_lo, s44, v17                         // 000000003540: d7006a32 0202222c
	s_wait_alu depctr_va_vcc(0)                                // 000000003548: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s45, v18, vcc_lo            // 00000000354c: d5207c33 01aa242d
	v_add_co_u32 v52, vcc_lo, s44, v19                         // 000000003554: d7006a34 0202262c
	s_wait_alu depctr_va_vcc(0)                                // 00000000355c: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s45, v20, vcc_lo            // 000000003560: d5207c35 01aa282d
	v_add_co_u32 v54, vcc_lo, s44, v21                         // 000000003568: d7006a36 02022a2c
	s_wait_alu depctr_va_vcc(0)                                // 000000003570: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s45, v22, vcc_lo            // 000000003574: d5207c37 01aa2c2d
	v_add_co_u32 v56, vcc_lo, s44, v23                         // 00000000357c: d7006a38 02022e2c
	s_wait_alu depctr_va_vcc(0)                                // 000000003584: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s45, v24, vcc_lo            // 000000003588: d5207c39 01aa302d
	s_clause 0x7                                               // 000000003590: bf850007
	global_load_u8 v59, v[34:35], off                          // 000000003594: ee04007c 0000003b 00000022
	global_load_u8 v60, v[36:37], off                          // 0000000035a0: ee04007c 0000003c 00000024
	global_load_u8 v61, v[38:39], off                          // 0000000035ac: ee04007c 0000003d 00000026
	global_load_u8 v62, v[40:41], off                          // 0000000035b8: ee04007c 0000003e 00000028
	global_load_u8 v50, v[50:51], off                          // 0000000035c4: ee04007c 00000032 00000032
	global_load_u8 v51, v[52:53], off                          // 0000000035d0: ee04007c 00000033 00000034
	global_load_u8 v52, v[54:55], off                          // 0000000035dc: ee04007c 00000034 00000036
	global_load_u8 v53, v[56:57], off                          // 0000000035e8: ee04007c 00000035 00000038
	s_add_nc_u64 s[10:11], s[10:11], 32                        // 0000000035f4: a98aa00a
	v_add_co_u32 v11, vcc_lo, v11, s38                         // 0000000035f8: d7006a0b 02004d0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003600: bf88ff9e
	v_cmp_lt_u64_e64 s0, s[10:11], s[42:43]                    // 000000003604: d4590000 0200540a
	s_wait_alu depctr_va_vcc(0)                                // 00000000360c: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s39, v12, vcc_lo            // 000000003610: d5207c0c 01aa1827
	v_add_co_u32 v25, vcc_lo, v25, 32                          // 000000003618: d7006a19 02014119
	s_wait_alu depctr_va_vcc(0)                                // 000000003620: bf88ff9d
	v_add_co_ci_u32_e64 v26, null, 0, v26, vcc_lo              // 000000003624: d5207c1a 01aa3480
	v_add_co_u32 v27, vcc_lo, v27, 32                          // 00000000362c: d7006a1b 0201411b
	s_wait_alu depctr_va_vcc(0)                                // 000000003634: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, 0, v28, vcc_lo              // 000000003638: d5207c1c 01aa3880
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000003640: 8b6a007e
	s_add_nc_u64 s[44:45], s[44:45], 1                         // 000000003644: a9ac812c
	s_wait_loadcnt 0xc                                         // 000000003648: bfc0000c
	v_cmp_eq_u32_e64 s0, 0xff, v58                             // 00000000364c: d44a0000 020274ff 000000ff
	s_wait_loadcnt 0x9                                         // 000000003658: bfc00009
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[42:43], v[46:47], 0// 00000000365c: cc464022 1a025d2a
	v_add_nc_u32_e32 v42, 0xffffff02, v58                      // 000000003664: 4a5474ff ffffff02
	s_wait_loadcnt 0x8                                         // 00000000366c: bfc00008
	s_delay_alu instid0(valu_dep_2)                            // 000000003670: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[34:41], v[44:45], v[48:49], v[34:41]// 000000003674: cc464022 1c8a612c
	s_wait_loadcnt 0x7                                         // 00000000367c: bfc00007
	v_cmp_eq_u32_e64 s1, 0xff, v59                             // 000000003680: d44a0001 020276ff 000000ff
	s_wait_loadcnt 0x6                                         // 00000000368c: bfc00006
	v_cmp_eq_u32_e64 s2, 0xff, v60                             // 000000003690: d44a0002 020278ff 000000ff
	s_wait_loadcnt 0x5                                         // 00000000369c: bfc00005
	v_add_nc_u32_e32 v45, v42, v61                             // 0000000036a0: 4a5a7b2a
	v_cmp_eq_u32_e64 s3, 0xff, v61                             // 0000000036a4: d44a0003 02027aff 000000ff
	s_wait_loadcnt 0x3                                         // 0000000036b0: bfc00003
	v_cmp_eq_u32_e64 s5, 0xff, v50                             // 0000000036b4: d44a0005 020264ff 000000ff
	v_cmp_eq_u32_e64 s4, 0xff, v62                             // 0000000036c0: d44a0004 02027cff 000000ff
	s_wait_loadcnt 0x2                                         // 0000000036cc: bfc00002
	v_cmp_eq_u32_e64 s6, 0xff, v51                             // 0000000036d0: d44a0006 020266ff 000000ff
	v_ldexp_f32 v36, v36, v45                                  // 0000000036dc: d71c0024 02025b24
	s_or_b32 s3, s0, s3                                        // 0000000036e4: 8c030300
	s_or_b32 s5, s0, s5                                        // 0000000036e8: 8c050500
	s_wait_loadcnt 0x1                                         // 0000000036ec: bfc00001
	v_cmp_eq_u32_e64 s7, 0xff, v52                             // 0000000036f0: d44a0007 020268ff 000000ff
	s_wait_loadcnt 0x0                                         // 0000000036fc: bfc00000
	v_cmp_eq_u32_e64 s8, 0xff, v53                             // 000000003700: d44a0008 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 00000000370c: bf88ff9e
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s3                 // 000000003710: d5010024 000dff24 7fc00000
	s_or_b32 s2, s0, s2                                        // 00000000371c: 8c020200
	s_or_b32 s4, s0, s4                                        // 000000003720: 8c040400
	s_or_b32 s6, s0, s6                                        // 000000003724: 8c060600
	s_or_b32 s7, s0, s7                                        // 000000003728: 8c070700
	v_add_f32_e32 v31, v31, v36                                // 00000000372c: 063e491f
	v_add_nc_u32_e32 v47, v42, v50                             // 000000003730: 4a5e652a
	v_add_nc_u32_e32 v43, v42, v59                             // 000000003734: 4a56772a
	s_or_b32 s8, s0, s8                                        // 000000003738: 8c080800
	s_or_b32 s0, s1, s0                                        // 00000000373c: 8c000001
	v_add_nc_u32_e32 v49, v42, v52                             // 000000003740: 4a62692a
	v_ldexp_f32 v38, v38, v47                                  // 000000003744: d71c0026 02025f26
	v_ldexp_f32 v34, v34, v43                                  // 00000000374c: d71c0022 02025722
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003754: bf870193
	v_ldexp_f32 v40, v40, v49                                  // 000000003758: d71c0028 02026328
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s5                 // 000000003760: d5010026 0015ff26 7fc00000
	v_add_nc_u32_e32 v48, v42, v51                             // 00000000376c: 4a60672a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003770: bf88ff9e
	v_cndmask_b32_e64 v34, v34, 0x7fc00000, s0                 // 000000003774: d5010022 0001ff22 7fc00000
	v_add_nc_u32_e32 v44, v42, v60                             // 000000003780: 4a58792a
	v_cndmask_b32_e64 v40, v40, 0x7fc00000, s7                 // 000000003784: d5010028 001dff28 7fc00000
	v_add_f32_e32 v29, v29, v38                                // 000000003790: 063a4d1d
	v_ldexp_f32 v39, v39, v48                                  // 000000003794: d71c0027 02026127
	v_add_f32_e32 v33, v33, v34                                // 00000000379c: 06424521
	v_ldexp_f32 v35, v35, v44                                  // 0000000037a0: d71c0023 02025923
	v_add_f32_e32 v2, v2, v40                                  // 0000000037a8: 06045102
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 0000000037ac: bf870244
	v_cndmask_b32_e64 v39, v39, 0x7fc00000, s6                 // 0000000037b0: d5010027 0019ff27 7fc00000
	v_add_nc_u32_e32 v46, v42, v62                             // 0000000037bc: 4a5c7d2a
	v_add_nc_u32_e32 v42, v42, v53                             // 0000000037c0: 4a546b2a
	v_cndmask_b32_e64 v35, v35, 0x7fc00000, s2                 // 0000000037c4: d5010023 0009ff23 7fc00000
	v_add_f32_e32 v5, v5, v39                                  // 0000000037d0: 060a4f05
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000037d4: bf870214
	v_ldexp_f32 v37, v37, v46                                  // 0000000037d8: d71c0025 02025d25
	v_ldexp_f32 v41, v41, v42                                  // 0000000037e0: d71c0029 02025529
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 0000000037e8: bf870194
	v_add_f32_e32 v32, v32, v35                                // 0000000037ec: 06404720
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s4                 // 0000000037f0: d5010025 0011ff25 7fc00000
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000037fc: bf870113
	v_cndmask_b32_e64 v41, v41, 0x7fc00000, s8                 // 000000003800: d5010029 0021ff29 7fc00000
	v_add_f32_e32 v30, v30, v37                                // 00000000380c: 063c4b1e
	s_delay_alu instid0(valu_dep_2)                            // 000000003810: bf870002
	v_add_f32_e32 v6, v6, v41                                  // 000000003814: 060c5306
	s_cbranch_vccnz 65316                                      // 000000003818: bfa4ff24 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x19ac>
	v_mul_lo_u32 v7, s39, v0                                   // 00000000381c: d72c0007 02020027
	v_mul_lo_u32 v8, s38, v1                                   // 000000003824: d72c0008 02020226
	v_mad_co_u64_u32 v[0:1], null, s38, v0, 0                  // 00000000382c: d6fe7c00 02020026
	v_bfe_u32 v9, v33, 16, 1                                   // 000000003834: d6100009 02052121
	v_or_b32_e32 v10, 0x400000, v33                            // 00000000383c: 381442ff 00400000
	v_bfe_u32 v11, v32, 16, 1                                  // 000000003844: d610000b 02052120
	v_or_b32_e32 v16, 0x400000, v30                            // 00000000384c: 38203cff 00400000
	v_or_b32_e32 v12, 0x400000, v32                            // 000000003854: 381840ff 00400000
	v_add3_u32 v9, v9, v33, 0x7fff                             // 00000000385c: d6550009 03fe4309 00007fff
	s_lshl_b64 s[0:1], s[38:39], 1                             // 000000003868: 84808126
	v_add3_u32 v1, v1, v8, v7                                  // 00000000386c: d6550001 041e1101
	v_lshlrev_b64_e32 v[7:8], 1, v[3:4]                        // 000000003874: 3e0e0681
	v_add3_u32 v11, v11, v32, 0x7fff                           // 000000003878: d655000b 03fe410b 00007fff
	v_bfe_u32 v18, v29, 16, 1                                  // 000000003884: d6100012 0205211d
	v_or_b32_e32 v19, 0x400000, v29                            // 00000000388c: 38263aff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003894: 3e000081
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003898: bf870113
	v_add3_u32 v18, v18, v29, 0x7fff                           // 00000000389c: d6550012 03fe3b12 00007fff
	v_add_co_u32 v13, vcc_lo, s40, v0                          // 0000000038a8: d7006a0d 02020028
	s_wait_alu depctr_va_vcc(0)                                // 0000000038b0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000038b4: bf870003
	v_add_co_ci_u32_e64 v14, null, s41, v1, vcc_lo             // 0000000038b8: d5207c0e 01aa0229
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 0000000038c0: 7c304321
	s_wait_alu depctr_va_vcc(0)                                // 0000000038c4: bf88ff9d
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo                      // 0000000038c8: 02121509
	v_add_co_u32 v0, vcc_lo, v13, v7                           // 0000000038cc: d7006a00 02020f0d
	s_wait_alu depctr_va_vcc(0)                                // 0000000038d4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v14, v8, vcc_lo              // 0000000038d8: d5207c01 01aa110e
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 0000000038e0: 7c304120
	v_bfe_u32 v10, v31, 16, 1                                  // 0000000038e4: d610000a 0205211f
	global_store_d16_hi_b16 v[0:1], v9, off                    // 0000000038ec: ee09407c 04800000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 0000000038f8: bf88ff9d
	v_cndmask_b32_e32 v15, v11, v12, vcc_lo                    // 0000000038fc: 021e190b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003900: bf88ff9e
	v_add_co_u32 v9, vcc_lo, v13, s0                           // 000000003904: d7006a09 0200010d
	s_wait_alu depctr_va_vcc(0)                                // 00000000390c: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s1, v14, vcc_lo             // 000000003910: d5207c0b 01aa1c01
	v_add3_u32 v10, v10, v31, 0x7fff                           // 000000003918: d655000a 03fe3f0a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003924: bf870003
	v_add_co_u32 v0, vcc_lo, v9, v7                            // 000000003928: d7006a00 02020f09
	v_or_b32_e32 v12, 0x400000, v31                            // 000000003930: 38183eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003938: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v11, v8, vcc_lo              // 00000000393c: d5207c01 01aa110b
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 000000003944: 7c303f1f
	s_wait_alu depctr_va_vcc(0)                                // 000000003948: bf88ff9d
	v_cndmask_b32_e32 v13, v10, v12, vcc_lo                    // 00000000394c: 021a190a
	v_add_co_u32 v12, vcc_lo, v9, s0                           // 000000003950: d7006a0c 02000109
	v_bfe_u32 v10, v30, 16, 1                                  // 000000003958: d610000a 0205211e
	s_wait_alu depctr_va_vcc(0)                                // 000000003960: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s1, v11, vcc_lo             // 000000003964: d5207c0b 01aa1601
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000396c: bf870193
	v_add_co_u32 v9, vcc_lo, v12, v7                           // 000000003970: d7006a09 02020f0c
	v_add3_u32 v14, v10, v30, 0x7fff                           // 000000003978: d655000e 03fe3d0a 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003984: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003988: bf870003
	v_add_co_ci_u32_e64 v10, null, v11, v8, vcc_lo             // 00000000398c: d5207c0a 01aa110b
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000003994: 7c303d1e
	s_wait_alu depctr_va_vcc(0)                                // 000000003998: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo                    // 00000000399c: 021c210e
	v_add_co_u32 v16, vcc_lo, v12, s0                          // 0000000039a0: d7006a10 0200010c
	s_wait_alu depctr_va_vcc(0)                                // 0000000039a8: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v11, vcc_lo             // 0000000039ac: d5207c11 01aa1601
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000039b4: bf870122
	v_add_co_u32 v11, vcc_lo, v16, v7                          // 0000000039b8: d7006a0b 02020f10
	s_wait_alu depctr_va_vcc(0)                                // 0000000039c0: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, v17, v8, vcc_lo             // 0000000039c4: d5207c0c 01aa1111
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 0000000039cc: 7c303b1d
	s_clause 0x2                                               // 0000000039d0: bf850002
	global_store_d16_hi_b16 v[0:1], v15, off                   // 0000000039d4: ee09407c 07800000 00000000
	global_store_d16_hi_b16 v[9:10], v13, off                  // 0000000039e0: ee09407c 06800000 00000009
	global_store_d16_hi_b16 v[11:12], v14, off                 // 0000000039ec: ee09407c 07000000 0000000b
	v_bfe_u32 v0, v5, 16, 1                                    // 0000000039f8: d6100000 02052105
	v_or_b32_e32 v12, 0x400000, v5                             // 000000003a00: 38180aff 00400000
	v_bfe_u32 v14, v2, 16, 1                                   // 000000003a08: d610000e 02052102
	s_wait_alu depctr_va_vcc(0)                                // 000000003a10: bf88ff9d
	v_cndmask_b32_e32 v13, v18, v19, vcc_lo                    // 000000003a14: 021a2712
	v_add_co_u32 v1, vcc_lo, v16, s0                           // 000000003a18: d7006a01 02000110
	s_wait_alu depctr_va_vcc(0)                                // 000000003a20: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s1, v17, vcc_lo             // 000000003a24: d5207c0b 01aa2201
	v_add3_u32 v0, v0, v5, 0x7fff                              // 000000003a2c: d6550000 03fe0b00 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a38: bf8701a3
	v_add_co_u32 v9, vcc_lo, v1, v7                            // 000000003a3c: d7006a09 02020f01
	s_wait_alu depctr_va_vcc(0)                                // 000000003a44: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, v11, v8, vcc_lo             // 000000003a48: d5207c0a 01aa110b
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000003a50: 7c300b05
	v_add3_u32 v14, v14, v2, 0x7fff                            // 000000003a54: d655000e 03fe050e 00007fff
	v_or_b32_e32 v15, 0x400000, v2                             // 000000003a60: 381e04ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a68: bf88ff9d
	v_cndmask_b32_e32 v5, v0, v12, vcc_lo                      // 000000003a6c: 020a1900
	v_add_co_u32 v0, vcc_lo, v1, s0                            // 000000003a70: d7006a00 02000101
	s_wait_alu depctr_va_vcc(0)                                // 000000003a78: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s1, v11, vcc_lo              // 000000003a7c: d5207c01 01aa1601
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003a84: bf870122
	v_add_co_u32 v11, vcc_lo, v0, v7                           // 000000003a88: d7006a0b 02020f00
	s_wait_alu depctr_va_vcc(0)                                // 000000003a90: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, v1, v8, vcc_lo              // 000000003a94: d5207c0c 01aa1101
	v_add_co_u32 v0, vcc_lo, v0, s0                            // 000000003a9c: d7006a00 02000100
	s_wait_alu depctr_va_vcc(0)                                // 000000003aa4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo               // 000000003aa8: d5207c01 01aa0201
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000003ab0: 7c300502
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab4: bf88ff9d
	v_cndmask_b32_e32 v2, v14, v15, vcc_lo                     // 000000003ab8: 02041f0e
	v_add_co_u32 v7, vcc_lo, v0, v7                            // 000000003abc: d7006a07 02020f00
	s_wait_alu depctr_va_vcc(0)                                // 000000003ac4: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, v1, v8, vcc_lo               // 000000003ac8: d5207c08 01aa1101
	v_add_co_u32 v0, vcc_lo, v0, s0                            // 000000003ad0: d7006a00 02000100
	s_wait_alu depctr_va_vcc(0)                                // 000000003ad8: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo               // 000000003adc: d5207c01 01aa0201
	s_mov_b32 s0, -1                                           // 000000003ae4: be8000c1
	s_clause 0x2                                               // 000000003ae8: bf850002
	global_store_d16_hi_b16 v[9:10], v13, off                  // 000000003aec: ee09407c 06800000 00000009
	global_store_d16_hi_b16 v[11:12], v5, off                  // 000000003af8: ee09407c 02800000 0000000b
	global_store_d16_hi_b16 v[7:8], v2, off                    // 000000003b04: ee09407c 01000000 00000007
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b10: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003b14: be812000
	s_cbranch_execz 21                                         // 000000003b18: bfa50015 <tessera_rocm_scaled_matmul_b867c3d0aa38586b+0x2070>
	v_bfe_u32 v2, v6, 16, 1                                    // 000000003b1c: d6100002 02052106
	v_or_b32_e32 v5, 0x400000, v6                              // 000000003b24: 380a0cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000003b2c: 7c300d06
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_2)// 000000003b30: bf870133
	v_add3_u32 v7, v2, v6, 0x7fff                              // 000000003b34: d6550007 03fe0d02 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[3:4]                        // 000000003b40: 3e040681
	s_wait_alu depctr_va_vcc(0)                                // 000000003b44: bf88ff9d
	v_cndmask_b32_e32 v4, v7, v5, vcc_lo                       // 000000003b48: 02080b07
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b4c: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000003b50: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000003b58: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000003b5c: d5207c01 01aa0701
	global_store_d16_hi_b16 v[0:1], v4, off                    // 000000003b64: ee09407c 02000000 00000000
	s_endpgm                                                   // 000000003b70: bfb00000
	s_code_end                                                 // 000000003b74: bf9f0000
	s_code_end                                                 // 000000003b78: bf9f0000
	s_code_end                                                 // 000000003b7c: bf9f0000
	s_code_end                                                 // 000000003b80: bf9f0000
	s_code_end                                                 // 000000003b84: bf9f0000
	s_code_end                                                 // 000000003b88: bf9f0000
	s_code_end                                                 // 000000003b8c: bf9f0000
	s_code_end                                                 // 000000003b90: bf9f0000
	s_code_end                                                 // 000000003b94: bf9f0000
	s_code_end                                                 // 000000003b98: bf9f0000
	s_code_end                                                 // 000000003b9c: bf9f0000
	s_code_end                                                 // 000000003ba0: bf9f0000
	s_code_end                                                 // 000000003ba4: bf9f0000
	s_code_end                                                 // 000000003ba8: bf9f0000
	s_code_end                                                 // 000000003bac: bf9f0000
	s_code_end                                                 // 000000003bb0: bf9f0000
	s_code_end                                                 // 000000003bb4: bf9f0000
	s_code_end                                                 // 000000003bb8: bf9f0000
	s_code_end                                                 // 000000003bbc: bf9f0000
	s_code_end                                                 // 000000003bc0: bf9f0000
	s_code_end                                                 // 000000003bc4: bf9f0000
	s_code_end                                                 // 000000003bc8: bf9f0000
	s_code_end                                                 // 000000003bcc: bf9f0000
	s_code_end                                                 // 000000003bd0: bf9f0000
	s_code_end                                                 // 000000003bd4: bf9f0000
	s_code_end                                                 // 000000003bd8: bf9f0000
	s_code_end                                                 // 000000003bdc: bf9f0000
	s_code_end                                                 // 000000003be0: bf9f0000
	s_code_end                                                 // 000000003be4: bf9f0000
	s_code_end                                                 // 000000003be8: bf9f0000
	s_code_end                                                 // 000000003bec: bf9f0000
	s_code_end                                                 // 000000003bf0: bf9f0000
	s_code_end                                                 // 000000003bf4: bf9f0000
	s_code_end                                                 // 000000003bf8: bf9f0000
	s_code_end                                                 // 000000003bfc: bf9f0000
	s_code_end                                                 // 000000003c00: bf9f0000
	s_code_end                                                 // 000000003c04: bf9f0000
	s_code_end                                                 // 000000003c08: bf9f0000
	s_code_end                                                 // 000000003c0c: bf9f0000
	s_code_end                                                 // 000000003c10: bf9f0000
	s_code_end                                                 // 000000003c14: bf9f0000
	s_code_end                                                 // 000000003c18: bf9f0000
	s_code_end                                                 // 000000003c1c: bf9f0000
	s_code_end                                                 // 000000003c20: bf9f0000
	s_code_end                                                 // 000000003c24: bf9f0000
	s_code_end                                                 // 000000003c28: bf9f0000
	s_code_end                                                 // 000000003c2c: bf9f0000
	s_code_end                                                 // 000000003c30: bf9f0000
	s_code_end                                                 // 000000003c34: bf9f0000
	s_code_end                                                 // 000000003c38: bf9f0000
	s_code_end                                                 // 000000003c3c: bf9f0000
	s_code_end                                                 // 000000003c40: bf9f0000
	s_code_end                                                 // 000000003c44: bf9f0000
	s_code_end                                                 // 000000003c48: bf9f0000
	s_code_end                                                 // 000000003c4c: bf9f0000
	s_code_end                                                 // 000000003c50: bf9f0000
	s_code_end                                                 // 000000003c54: bf9f0000
	s_code_end                                                 // 000000003c58: bf9f0000
	s_code_end                                                 // 000000003c5c: bf9f0000
	s_code_end                                                 // 000000003c60: bf9f0000
	s_code_end                                                 // 000000003c64: bf9f0000
	s_code_end                                                 // 000000003c68: bf9f0000
	s_code_end                                                 // 000000003c6c: bf9f0000
	s_code_end                                                 // 000000003c70: bf9f0000
	s_code_end                                                 // 000000003c74: bf9f0000
	s_code_end                                                 // 000000003c78: bf9f0000
	s_code_end                                                 // 000000003c7c: bf9f0000
	s_code_end                                                 // 000000003c80: bf9f0000
	s_code_end                                                 // 000000003c84: bf9f0000
	s_code_end                                                 // 000000003c88: bf9f0000
	s_code_end                                                 // 000000003c8c: bf9f0000
	s_code_end                                                 // 000000003c90: bf9f0000
	s_code_end                                                 // 000000003c94: bf9f0000
	s_code_end                                                 // 000000003c98: bf9f0000
	s_code_end                                                 // 000000003c9c: bf9f0000
	s_code_end                                                 // 000000003ca0: bf9f0000
	s_code_end                                                 // 000000003ca4: bf9f0000
	s_code_end                                                 // 000000003ca8: bf9f0000
	s_code_end                                                 // 000000003cac: bf9f0000
	s_code_end                                                 // 000000003cb0: bf9f0000
	s_code_end                                                 // 000000003cb4: bf9f0000
	s_code_end                                                 // 000000003cb8: bf9f0000
	s_code_end                                                 // 000000003cbc: bf9f0000
	s_code_end                                                 // 000000003cc0: bf9f0000
	s_code_end                                                 // 000000003cc4: bf9f0000
	s_code_end                                                 // 000000003cc8: bf9f0000
	s_code_end                                                 // 000000003ccc: bf9f0000
	s_code_end                                                 // 000000003cd0: bf9f0000
	s_code_end                                                 // 000000003cd4: bf9f0000
	s_code_end                                                 // 000000003cd8: bf9f0000
	s_code_end                                                 // 000000003cdc: bf9f0000
	s_code_end                                                 // 000000003ce0: bf9f0000
	s_code_end                                                 // 000000003ce4: bf9f0000
	s_code_end                                                 // 000000003ce8: bf9f0000
	s_code_end                                                 // 000000003cec: bf9f0000
	s_code_end                                                 // 000000003cf0: bf9f0000
	s_code_end                                                 // 000000003cf4: bf9f0000
	s_code_end                                                 // 000000003cf8: bf9f0000
	s_code_end                                                 // 000000003cfc: bf9f0000
