
/tmp/tmpajfnseg7.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_305100ba3501aae0>:
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b00: f4002080 f80000d8
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b08: 320a0081
	s_clause 0x4                                               // 000000001b0c: bf850004
	s_load_b64 s[18:19], s[0:1], 0x8                           // 000000001b10: f4002480 f8000008
	s_load_b64 s[20:21], s[0:1], 0x30                          // 000000001b18: f4002500 f8000030
	s_load_b64 s[12:13], s[0:1], 0x58                          // 000000001b20: f4002300 f8000058
	s_load_b64 s[8:9], s[0:1], 0x80                            // 000000001b28: f4002200 f8000080
	s_load_b128 s[4:7], s[0:1], 0xc8                           // 000000001b30: f4004100 f80000c8
	s_mov_b32 s14, ttmp7                                       // 000000001b38: be8e0073
	s_ashr_i32 s15, ttmp7, 31                                  // 000000001b3c: 860f9f73
	s_mov_b32 s10, ttmp9                                       // 000000001b40: be8a0075
	s_lshl_b64 s[14:15], s[14:15], 7                           // 000000001b44: 848e870e
	v_and_b32_e32 v6, 0x60, v5                                 // 000000001b48: 360c0aff 00000060
	s_ashr_i32 s11, ttmp9, 31                                  // 000000001b50: 860b9f75
	v_or_b32_e32 v1, s14, v5                                   // 000000001b54: 38020a0e
	s_lshl_b64 s[16:17], s[10:11], 6                           // 000000001b58: 8490860a
	v_dual_mov_b32 v18, 0 :: v_dual_lshlrev_b32 v3, 4, v0      // 000000001b5c: ca220080 12020084
	v_or_b32_e32 v7, 16, v6                                    // 000000001b64: 380e0c90
	v_or_b32_e32 v14, s14, v6                                  // 000000001b68: 381c0c0e
	v_mul_u32_u24_e32 v11, 48, v5                              // 000000001b6c: 16160ab0
	v_cmp_gt_u32_e32 vcc_lo, 0x80, v0                          // 000000001b70: 7c9800ff 00000080
	v_and_b32_e32 v12, 16, v3                                  // 000000001b78: 36180690
	v_or_b32_e32 v28, s14, v7                                  // 000000001b7c: 38380e0e
	v_add_co_u32 v4, s14, s16, v5                              // 000000001b80: d7000e04 02020a10
	s_wait_alu depctr_va_sdst(0)                               // 000000001b88: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, 0, s14                  // 000000001b8c: d5207c08 00390011
	s_wait_kmcnt 0x0                                           // 000000001b94: bfc70000
	v_mul_lo_u32 v9, s3, v1                                    // 000000001b98: d72c0009 02020203
	v_mad_co_u64_u32 v[1:2], null, s2, v1, s[18:19]            // 000000001ba0: d6fe7c01 004a0202
	v_mul_lo_u32 v13, s3, v4                                   // 000000001ba8: d72c000d 02020803
	v_mul_lo_u32 v8, s2, v8                                    // 000000001bb0: d72c0008 02021002
	v_mad_co_u64_u32 v[3:4], null, s2, v4, s[20:21]            // 000000001bb8: d6fe7c03 00520802
	s_mul_i32 s14, s2, s15                                     // 000000001bc0: 960e0f02
	v_and_b32_e32 v10, 47, v0                                  // 000000001bc4: 361400af
	v_and_b32_e32 v0, 15, v0                                   // 000000001bc8: 3600008f
	s_lshr_b64 s[10:11], s[2:3], 5                             // 000000001bcc: 858a8502
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bd0: bf88ff9e
	v_add3_u32 v2, v9, v2, s14                                 // 000000001bd4: d6550002 003a0509
	v_dual_mov_b32 v78, 0 :: v_dual_add_nc_u32 v19, v11, v12   // 000000001bdc: ca200080 4e12190b
	v_mov_b32_e32 v11, s17                                     // 000000001be4: 7e160211
	v_add_co_u32 v20, s2, v1, v12                              // 000000001be8: d7000214 02021901
	s_wait_alu depctr_va_sdst(0)                               // 000000001bf0: bf88f19f
	v_add_co_ci_u32_e64 v21, null, 0, v2, s2                   // 000000001bf4: d5207c15 000a0480
	v_add3_u32 v1, v13, v4, v8                                 // 000000001bfc: d6550001 0422090d
	v_mov_b32_e32 v9, s15                                      // 000000001c04: 7e12020f
	v_mov_b32_e32 v13, s15                                     // 000000001c08: 7e1a020f
	v_or_b32_e32 v2, v6, v0                                    // 000000001c0c: 38040106
	v_add_co_u32 v22, s2, v3, v12                              // 000000001c10: d7000216 02021903
	s_wait_alu depctr_va_sdst(0)                               // 000000001c18: bf88f19f
	v_add_co_ci_u32_e64 v23, null, 0, v1, s2                   // 000000001c1c: d5207c17 000a0280
	v_and_b32_e32 v16, 8, v5                                   // 000000001c24: 36200a88
	v_mul_u32_u24_e32 v1, 48, v2                               // 000000001c28: 160204b0
	v_or_b32_e32 v0, v7, v0                                    // 000000001c2c: 38000107
	s_lshr_b32 s14, s3, 5                                      // 000000001c30: 850e8503
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v52, 0             // 000000001c34: ca100080 4d340080
	s_delay_alu instid0(valu_dep_3)                            // 000000001c3c: bf870003
	v_or_b32_e32 v24, v1, v16                                  // 000000001c40: 38302101
	v_mul_u32_u24_e32 v1, 48, v10                              // 000000001c44: 160214b0
	v_or_b32_e32 v8, v14, v16                                  // 000000001c48: 3810210e
	v_mul_u32_u24_e32 v0, 48, v0                               // 000000001c4c: 160000b0
	v_or_b32_e32 v26, 1, v16                                   // 000000001c50: 38342081
	v_or_b32_e32 v27, 2, v16                                   // 000000001c54: 38362082
	v_or_b32_e32 v29, v16, v1                                  // 000000001c58: 383a0310
	v_mov_b32_e32 v1, s15                                      // 000000001c5c: 7e02020f
	v_or_b32_e32 v17, 16, v10                                  // 000000001c60: 38221490
	v_cmp_gt_i64_e64 s2, s[4:5], v[8:9]                        // 000000001c64: d4540002 02021004
	v_or_b32_e32 v25, v0, v16                                  // 000000001c6c: 38322100
	v_or_b32_e32 v0, v26, v14                                  // 000000001c70: 38001d1a
	v_or_b32_e32 v10, s16, v10                                 // 000000001c74: 38141410
	v_mul_u32_u24_e32 v2, 48, v17                              // 000000001c78: 160422b0
	v_or_b32_e32 v32, 3, v16                                   // 000000001c7c: 38402083
	s_wait_alu depctr_va_sdst(0)                               // 000000001c80: bf88f19f
	v_cndmask_b32_e64 v3, 0, v8, s2                            // 000000001c84: d5010003 000a1080
	v_dual_mov_b32 v62, 0 :: v_dual_add_nc_u32 v89, 0x1800, v29// 000000001c8c: ca200080 3e583aff 00001800
	v_or_b32_e32 v30, v2, v16                                  // 000000001c98: 383c2102
	v_cndmask_b32_e64 v2, 0, v9, s2                            // 000000001c9c: d5010002 000a1280
	v_cmp_gt_i64_e64 s2, s[4:5], v[0:1]                        // 000000001ca4: d4540002 02020004
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cac: bf88ff9e
	v_mul_lo_u32 v4, s14, v3                                   // 000000001cb0: d72c0004 0202060e
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v48, 0             // 000000001cb8: ca100080 49300080
	v_mul_lo_u32 v12, s10, v2                                  // 000000001cc0: d72c000c 0202040a
	v_mad_co_u64_u32 v[2:3], null, s10, v3, 0                  // 000000001cc8: d6fe7c02 0202060a
	s_wait_alu depctr_va_sdst(0)                               // 000000001cd0: bf88f19f
	v_cndmask_b32_e64 v7, 0, v0, s2                            // 000000001cd4: d5010007 000a0080
	v_or_b32_e32 v0, v27, v14                                  // 000000001cdc: 38001d1b
	v_cndmask_b32_e64 v6, 0, v1, s2                            // 000000001ce0: d5010006 000a0280
	v_cmp_gt_i64_e64 s2, s[6:7], v[10:11]                      // 000000001ce8: d4540002 02021406
	v_mov_b32_e32 v71, 0                                       // 000000001cf0: 7e8e0280
	v_mul_lo_u32 v15, s14, v7                                  // 000000001cf4: d72c000f 02020e0e
	v_cmp_gt_i64_e64 s3, s[4:5], v[0:1]                        // 000000001cfc: d4540003 02020004
	v_mul_lo_u32 v31, s10, v6                                  // 000000001d04: d72c001f 02020c0a
	v_mad_co_u64_u32 v[6:7], null, s10, v7, 0                  // 000000001d0c: d6fe7c06 02020e0a
	v_add3_u32 v3, v3, v12, v4                                 // 000000001d14: d6550003 04121903
	v_or_b32_e32 v12, v32, v14                                 // 000000001d1c: 38181d20
	s_wait_alu depctr_va_sdst(0)                               // 000000001d20: bf88f19f
	v_cndmask_b32_e64 v5, 0, v11, s2                           // 000000001d24: d5010005 000a1680
	v_cndmask_b32_e64 v33, 0, v1, s3                           // 000000001d2c: d5010021 000e0280
	v_cndmask_b32_e64 v34, 0, v0, s3                           // 000000001d34: d5010022 000e0080
	v_cndmask_b32_e64 v4, 0, v10, s2                           // 000000001d3c: d5010004 000a1480
	v_cmp_gt_i64_e64 s2, s[4:5], v[12:13]                      // 000000001d44: d4540002 02021804
	v_add3_u32 v7, v7, v31, v15                                // 000000001d4c: d6550007 043e3f07
	v_mul_lo_u32 v31, s10, v33                                 // 000000001d54: d72c001f 0202420a
	v_or_b32_e32 v33, 4, v16                                   // 000000001d5c: 38422084
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001d60: 3e000482
	v_mul_lo_u32 v15, s14, v34                                 // 000000001d64: d72c000f 0202440e
	v_mad_co_u64_u32 v[2:3], null, s10, v34, 0                 // 000000001d6c: d6fe7c02 0202440a
	s_wait_alu depctr_va_sdst(0)                               // 000000001d74: bf88f19f
	v_cndmask_b32_e64 v35, 0, v12, s2                          // 000000001d78: d5010023 000a1880
	v_or_b32_e32 v12, v33, v14                                 // 000000001d80: 38181d21
	v_cndmask_b32_e64 v34, 0, v13, s2                          // 000000001d84: d5010022 000a1a80
	v_add_co_u32 v39, s2, s12, v0                              // 000000001d8c: d7000227 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001d94: bf88f19f
	v_add_co_ci_u32_e64 v40, null, s13, v1, s2                 // 000000001d98: d5207c28 000a020d
	v_cmp_gt_i64_e64 s2, s[4:5], v[12:13]                      // 000000001da0: d4540002 02021804
	v_add3_u32 v3, v3, v31, v15                                // 000000001da8: d6550003 043e3f03
	v_mul_lo_u32 v31, s10, v34                                 // 000000001db0: d72c001f 0202440a
	v_or_b32_e32 v34, 5, v16                                   // 000000001db8: 38442085
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001dbc: 3e000c82
	v_mul_lo_u32 v15, s14, v35                                 // 000000001dc0: d72c000f 0202460e
	v_mad_co_u64_u32 v[6:7], null, s10, v35, 0                 // 000000001dc8: d6fe7c06 0202460a
	s_wait_alu depctr_va_sdst(0)                               // 000000001dd0: bf88f19f
	v_cndmask_b32_e64 v36, 0, v12, s2                          // 000000001dd4: d5010024 000a1880
	v_or_b32_e32 v12, v34, v14                                 // 000000001ddc: 38181d22
	v_cndmask_b32_e64 v35, 0, v13, s2                          // 000000001de0: d5010023 000a1a80
	v_add_co_u32 v44, s2, s12, v0                              // 000000001de8: d700022c 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001df0: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s13, v1, s2                 // 000000001df4: d5207c2d 000a020d
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001dfc: 3e000482
	v_cmp_gt_i64_e64 s2, s[4:5], v[12:13]                      // 000000001e00: d4540002 02021804
	v_add3_u32 v7, v7, v31, v15                                // 000000001e08: d6550007 043e3f07
	v_mul_lo_u32 v15, s14, v36                                 // 000000001e10: d72c000f 0202480e
	v_mad_co_u64_u32 v[2:3], null, s10, v36, 0                 // 000000001e18: d6fe7c02 0202480a
	v_mul_lo_u32 v31, s10, v35                                 // 000000001e20: d72c001f 0202460a
	v_mov_b32_e32 v69, 0                                       // 000000001e28: 7e8a0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001e2c: bf88f19f
	v_cndmask_b32_e64 v36, 0, v13, s2                          // 000000001e30: d5010024 000a1a80
	v_cndmask_b32_e64 v37, 0, v12, s2                          // 000000001e38: d5010025 000a1880
	v_add_co_u32 v46, s2, s12, v0                              // 000000001e40: d700022e 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001e48: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s13, v1, s2                 // 000000001e4c: d5207c2f 000a020d
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e54: 3e000c82
	v_mov_b32_e32 v7, s15                                      // 000000001e58: 7e0e020f
	v_or_b32_e32 v35, 6, v16                                   // 000000001e5c: 38462086
	v_or_b32_e32 v38, 7, v16                                   // 000000001e60: 384c2087
	v_add3_u32 v3, v3, v31, v15                                // 000000001e64: d6550003 043e3f03
	v_mul_lo_u32 v31, s14, v37                                 // 000000001e6c: d72c001f 02024a0e
	v_mul_lo_u32 v36, s10, v36                                 // 000000001e74: d72c0024 0202480a
	v_or_b32_e32 v12, v35, v14                                 // 000000001e7c: 38181d23
	v_or_b32_e32 v6, v38, v14                                  // 000000001e80: 380c1d26
	v_add_co_u32 v50, s3, s12, v0                              // 000000001e84: d7000332 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001e8c: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s13, v1, s3                 // 000000001e90: d5207c33 000e020d
	v_cmp_gt_i64_e64 s2, s[4:5], v[12:13]                      // 000000001e98: d4540002 02021804
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001ea0: 3e000482
	v_add_nc_u32_e32 v90, 0x1800, v30                          // 000000001ea4: 4ab43cff 00001800
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v43, 0             // 000000001eac: ca100080 4a2a0080
	v_mov_b32_e32 v30, 0                                       // 000000001eb4: 7e3c0280
	s_wait_alu depctr_va_sdst(0)                               // 000000001eb8: bf88f19f
	v_cndmask_b32_e64 v15, 0, v13, s2                          // 000000001ebc: d501000f 000a1a80
	v_cndmask_b32_e64 v41, 0, v12, s2                          // 000000001ec4: d5010029 000a1880
	v_mad_co_u64_u32 v[12:13], null, s10, v37, 0               // 000000001ecc: d6fe7c0c 02024a0a
	v_cmp_gt_i64_e64 s2, s[4:5], v[6:7]                        // 000000001ed4: d4540002 02020c04
	v_mov_b32_e32 v66, 0                                       // 000000001edc: 7e840280
	v_mul_lo_u32 v42, s10, v15                                 // 000000001ee0: d72c002a 02021e0a
	v_mul_lo_u32 v37, s14, v41                                 // 000000001ee8: d72c0025 0202520e
	v_mad_co_u64_u32 v[14:15], null, s10, v41, 0               // 000000001ef0: d6fe7c0e 0202520a
	v_mov_b32_e32 v41, 0                                       // 000000001ef8: 7e520280
	s_wait_alu depctr_va_sdst(0)                               // 000000001efc: bf88f19f
	v_cndmask_b32_e64 v7, 0, v7, s2                            // 000000001f00: d5010007 000a0e80
	v_add3_u32 v13, v13, v36, v31                              // 000000001f08: d655000d 047e490d
	v_cndmask_b32_e64 v6, 0, v6, s2                            // 000000001f10: d5010006 000a0c80
	v_add_co_u32 v53, s2, s12, v0                              // 000000001f18: d7000235 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001f20: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s13, v1, s2                 // 000000001f24: d5207c36 000a020d
	v_lshlrev_b64_e32 v[0:1], 2, v[12:13]                      // 000000001f2c: 3e001882
	v_add3_u32 v15, v15, v42, v37                              // 000000001f30: d655000f 0496550f
	v_mov_b32_e32 v13, s15                                     // 000000001f38: 7e1a020f
	v_or_b32_e32 v12, v28, v16                                 // 000000001f3c: 3818211c
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v37, 0             // 000000001f40: ca100080 2a240080
	s_delay_alu instid0(valu_dep_4)                            // 000000001f48: bf870004
	v_lshlrev_b64_e32 v[2:3], 2, v[14:15]                      // 000000001f4c: 3e041c82
	v_mul_lo_u32 v14, s14, v6                                  // 000000001f50: d72c000e 02020c0e
	v_mul_lo_u32 v15, s10, v7                                  // 000000001f58: d72c000f 02020e0a
	v_mad_co_u64_u32 v[6:7], null, s10, v6, 0                  // 000000001f60: d6fe7c06 02020c0a
	v_add_co_u32 v56, s2, s12, v0                              // 000000001f68: d7000238 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000001f70: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s13, v1, s2                 // 000000001f74: d5207c39 000a020d
	v_cmp_gt_i64_e64 s2, s[4:5], v[12:13]                      // 000000001f7c: d4540002 02021804
	v_mov_b32_e32 v1, s15                                      // 000000001f84: 7e02020f
	v_or_b32_e32 v0, v28, v26                                  // 000000001f88: 3800351c
	v_add3_u32 v7, v7, v15, v14                                // 000000001f8c: d6550007 043a1f07
	v_add_co_u32 v58, s3, s12, v2                              // 000000001f94: d700033a 0202040c
	s_wait_alu depctr_va_sdst(0)                               // 000000001f9c: bf88f19f
	v_cndmask_b32_e64 v14, 0, v13, s2                          // 000000001fa0: d501000e 000a1a80
	v_cndmask_b32_e64 v15, 0, v12, s2                          // 000000001fa8: d501000f 000a1880
	v_cmp_gt_i64_e64 s2, s[4:5], v[0:1]                        // 000000001fb0: d4540002 02020004
	v_add_co_ci_u32_e64 v60, null, s13, v3, s3                 // 000000001fb8: d5207c3c 000e060d
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001fc0: 3e040c82
	v_or_b32_e32 v6, s16, v17                                  // 000000001fc4: 380c2210
	v_mul_lo_u32 v16, s14, v15                                 // 000000001fc8: d72c0010 02021e0e
	v_mul_lo_u32 v17, s10, v14                                 // 000000001fd0: d72c0011 02021c0a
	v_mad_co_u64_u32 v[14:15], null, s10, v15, 0               // 000000001fd8: d6fe7c0e 02021e0a
	s_wait_alu depctr_va_sdst(0)                               // 000000001fe0: bf88f19f
	v_cndmask_b32_e64 v31, 0, v0, s2                           // 000000001fe4: d501001f 000a0080
	v_or_b32_e32 v0, v28, v27                                  // 000000001fec: 3800371c
	v_cndmask_b32_e64 v26, 0, v1, s2                           // 000000001ff0: d501001a 000a0280
	v_add_co_u32 v63, s2, s12, v2                              // 000000001ff8: d700023f 0202040c
	v_mov_b32_e32 v7, s17                                      // 000000002000: 7e0e0211
	s_delay_alu instid0(valu_dep_4)                            // 000000002004: bf870004
	v_cmp_gt_i64_e64 s3, s[4:5], v[0:1]                        // 000000002008: d4540003 02020004
	s_wait_alu depctr_va_sdst(0)                               // 000000002010: bf88f19f
	v_add_co_ci_u32_e64 v64, null, s13, v3, s2                 // 000000002014: d5207c40 000a060d
	v_add3_u32 v15, v15, v17, v16                              // 00000000201c: d655000f 0442230f
	v_mul_lo_u32 v16, s14, v31                                 // 000000002024: d72c0010 02023e0e
	v_mul_lo_u32 v17, s10, v26                                 // 00000000202c: d72c0011 0202340a
	v_mad_co_u64_u32 v[2:3], null, s10, v31, 0                 // 000000002034: d6fe7c02 02023e0a
	v_cmp_gt_i64_e64 s2, s[6:7], v[6:7]                        // 00000000203c: d4540002 02020c06
	v_cndmask_b32_e64 v26, 0, v1, s3                           // 000000002044: d501001a 000e0280
	v_cndmask_b32_e64 v27, 0, v0, s3                           // 00000000204c: d501001b 000e0080
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 000000002054: 3e001c82
	v_mov_b32_e32 v15, s15                                     // 000000002058: 7e1e020f
	v_or_b32_e32 v14, v28, v32                                 // 00000000205c: 381c411c
	s_wait_alu depctr_va_sdst(0)                               // 000000002060: bf88f19f
	v_cndmask_b32_e64 v7, 0, v7, s2                            // 000000002064: d5010007 000a0e80
	v_add3_u32 v3, v3, v17, v16                                // 00000000206c: d6550003 04422303
	v_cndmask_b32_e64 v6, 0, v6, s2                            // 000000002074: d5010006 000a0c80
	v_add_co_u32 v67, s3, s12, v0                              // 00000000207c: d7000343 0202000c
	v_cmp_gt_i64_e64 s2, s[4:5], v[14:15]                      // 000000002084: d4540002 02021c04
	s_wait_alu depctr_va_sdst(0)                               // 00000000208c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s13, v1, s3                 // 000000002090: d5207c44 000e020d
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000002098: 3e000482
	v_mov_b32_e32 v3, s15                                      // 00000000209c: 7e06020f
	v_or_b32_e32 v2, v28, v33                                  // 0000000020a0: 3804431c
	v_mul_lo_u32 v31, s14, v27                                 // 0000000020a4: d72c001f 0202360e
	v_mul_lo_u32 v26, s10, v26                                 // 0000000020ac: d72c001a 0202340a
	v_mad_co_u64_u32 v[16:17], null, s10, v27, 0               // 0000000020b4: d6fe7c10 0202360a
	v_cndmask_b32_e64 v15, 0, v15, s2                          // 0000000020bc: d501000f 000a1e80
	v_cndmask_b32_e64 v14, 0, v14, s2                          // 0000000020c4: d501000e 000a1c80
	v_cmp_gt_i64_e64 s2, s[4:5], v[2:3]                        // 0000000020cc: d4540002 02020404
	v_add_co_u32 v70, s3, s12, v0                              // 0000000020d4: d7000346 0202000c
	s_delay_alu instid0(valu_dep_4)                            // 0000000020dc: bf870004
	v_mul_lo_u32 v27, s10, v15                                 // 0000000020e0: d72c001b 02021e0a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020e8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s13, v1, s3                 // 0000000020ec: d5207c48 000e020d
	v_add3_u32 v17, v17, v26, v31                              // 0000000020f4: d6550011 047e3511
	v_mul_lo_u32 v26, s14, v14                                 // 0000000020fc: d72c001a 02021c0e
	v_mad_co_u64_u32 v[14:15], null, s10, v14, 0               // 000000002104: d6fe7c0e 02021c0a
	v_cndmask_b32_e64 v32, 0, v2, s2                           // 00000000210c: d5010020 000a0480
	v_or_b32_e32 v2, v28, v34                                  // 000000002114: 3804451c
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 000000002118: 3e002082
	v_cndmask_b32_e64 v31, 0, v3, s2                           // 00000000211c: d501001f 000a0680
	v_mov_b32_e32 v36, 0                                       // 000000002124: 7e480280
	v_mul_lo_u32 v33, s14, v32                                 // 000000002128: d72c0021 0202400e
	v_cmp_gt_i64_e64 s2, s[4:5], v[2:3]                        // 000000002130: d4540002 02020404
	v_add3_u32 v15, v15, v27, v26                              // 000000002138: d655000f 046a370f
	v_mov_b32_e32 v27, s15                                     // 000000002140: 7e36020f
	v_or_b32_e32 v26, v28, v35                                 // 000000002144: 3834471c
	v_add_co_u32 v75, s3, s12, v0                              // 000000002148: d700034b 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000002150: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s13, v1, s3                 // 000000002154: d5207c4c 000e020d
	v_lshlrev_b64_e32 v[0:1], 2, v[14:15]                      // 00000000215c: 3e001c82
	v_cndmask_b32_e64 v15, 0, v2, s2                           // 000000002160: d501000f 000a0480
	v_or_b32_e32 v2, v28, v38                                  // 000000002168: 38044d1c
	v_mul_lo_u32 v31, s10, v31                                 // 00000000216c: d72c001f 02023e0a
	v_mad_co_u64_u32 v[16:17], null, s10, v32, 0               // 000000002174: d6fe7c10 0202400a
	v_cmp_gt_i64_e64 s3, s[4:5], v[26:27]                      // 00000000217c: d4540003 02023404
	v_mov_b32_e32 v38, 0                                       // 000000002184: 7e4c0280
	v_cndmask_b32_e64 v14, 0, v3, s2                           // 000000002188: d501000e 000a0680
	v_cmp_gt_i64_e64 s2, s[4:5], v[2:3]                        // 000000002190: d4540002 02020404
	v_mul_lo_u32 v28, s14, v15                                 // 000000002198: d72c001c 02021e0e
	v_mov_b32_e32 v35, 0                                       // 0000000021a0: 7e460280
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a4: bf88f19f
	v_cndmask_b32_e64 v27, 0, v27, s3                          // 0000000021a8: d501001b 000e3680
	v_cndmask_b32_e64 v26, 0, v26, s3                          // 0000000021b0: d501001a 000e3480
	v_add3_u32 v17, v17, v31, v33                              // 0000000021b8: d6550011 04863f11
	v_cndmask_b32_e64 v3, 0, v3, s2                            // 0000000021c0: d5010003 000a0680
	v_cndmask_b32_e64 v2, 0, v2, s2                            // 0000000021c8: d5010002 000a0480
	v_mul_lo_u32 v31, s10, v14                                 // 0000000021d0: d72c001f 02021c0a
	v_mad_co_u64_u32 v[14:15], null, s10, v15, 0               // 0000000021d8: d6fe7c0e 02021e0a
	v_mul_lo_u32 v32, s14, v26                                 // 0000000021e0: d72c0020 0202340e
	v_mul_lo_u32 v33, s10, v27                                 // 0000000021e8: d72c0021 0202360a
	v_mad_co_u64_u32 v[26:27], null, s10, v26, 0               // 0000000021f0: d6fe7c1a 0202340a
	v_add_co_u32 v79, s2, s12, v0                              // 0000000021f8: d700024f 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000002200: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s13, v1, s2                 // 000000002204: d5207c50 000a020d
	v_lshlrev_b64_e32 v[0:1], 2, v[16:17]                      // 00000000220c: 3e002082
	v_mul_lo_u32 v16, s14, v2                                  // 000000002210: d72c0010 0202040e
	v_mul_lo_u32 v17, s10, v3                                  // 000000002218: d72c0011 0202060a
	v_mad_co_u64_u32 v[2:3], null, s10, v2, 0                  // 000000002220: d6fe7c02 0202040a
	v_add3_u32 v15, v15, v31, v28                              // 000000002228: d655000f 04723f0f
	v_add3_u32 v27, v27, v33, v32                              // 000000002230: d655001b 0482431b
	v_add_co_u32 v81, s2, s12, v0                              // 000000002238: d7000251 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000002240: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s13, v1, s2                 // 000000002244: d5207c52 000a020d
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]                    // 00000000224c: 3e1c1c82
	v_add3_u32 v3, v3, v17, v16                                // 000000002250: d6550003 04422303
	v_lshlrev_b64_e32 v[0:1], 2, v[26:27]                      // 000000002258: 3e003482
	v_lshlrev_b64_e32 v[16:17], 2, v[6:7]                      // 00000000225c: 3e200c82
	v_mov_b32_e32 v33, 0                                       // 000000002260: 7e420280
	v_mov_b32_e32 v65, 0                                       // 000000002264: 7e820280
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000002268: 3e040482
	v_add_co_u32 v83, s2, s12, v14                             // 00000000226c: d7000253 02021c0c
	s_wait_alu depctr_va_sdst(0)                               // 000000002274: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s13, v15, s2                // 000000002278: d5207c54 000a1e0d
	v_add_co_u32 v85, s2, s12, v0                              // 000000002280: d7000255 0202000c
	s_wait_alu depctr_va_sdst(0)                               // 000000002288: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s13, v1, s2                 // 00000000228c: d5207c56 000a020d
	v_add_co_u32 v87, s2, s12, v2                              // 000000002294: d7000257 0202040c
	v_lshlrev_b64_e32 v[14:15], 2, v[4:5]                      // 00000000229c: 3e1c0882
	s_wait_alu depctr_va_sdst(0)                               // 0000000022a0: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s13, v3, s2                 // 0000000022a4: d5207c58 000a060d
	v_mov_b32_e32 v61, 0                                       // 0000000022ac: 7e7a0280
	v_mov_b32_e32 v59, 0                                       // 0000000022b0: 7e760280
	v_mov_b32_e32 v55, 0                                       // 0000000022b4: 7e6e0280
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v34, 0             // 0000000022b8: ca100080 31220080
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0             // 0000000022c0: ca100080 201e0080
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0             // 0000000022c8: ca100080 1d1c0080
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0             // 0000000022d0: ca100080 1b1a0080
	s_mov_b64 s[4:5], 0                                        // 0000000022d8: be840180
	s_branch 301                                               // 0000000022dc: bfa0012d <tessera_rocm_scaled_matmul_lds_305100ba3501aae0+0xc94>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000022e4: 8c7e027e
	s_mul_u64 s[14:15], s[4:5], s[6:7]                         // 0000000022e8: aa8e0604
	s_lshl_b64 s[12:13], s[4:5], 2                             // 0000000022ec: 848c8204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022f0: bf88ff9e
	s_lshl_b64 s[14:15], s[14:15], 2                           // 0000000022f4: 848e820e
	v_add_co_u32 v0, s2, v39, s12                              // 0000000022f8: d7000200 02001927
	s_wait_alu depctr_sa_sdst(0)                               // 000000002300: bf88ff9e
	s_add_nc_u64 s[14:15], s[8:9], s[14:15]                    // 000000002304: a98e0e08
	v_add_co_ci_u32_e64 v1, null, s13, v40, s2                 // 000000002308: d5207c01 000a500d
	s_wait_alu depctr_sa_sdst(0)                               // 000000002310: bf88ff9e
	v_add_co_u32 v2, s2, s14, v14                              // 000000002314: d7000202 02021c0e
	s_wait_alu depctr_va_sdst(0)                               // 00000000231c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v15, s2                 // 000000002320: d5207c03 000a1e0f
	v_add_co_u32 v4, s2, v44, s12                              // 000000002328: d7000204 0200192c
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v45, s2                 // 000000002334: d5207c05 000a5a0d
	s_wait_dscnt 0x0                                           // 00000000233c: bfc60000
	s_barrier_signal -1                                        // 000000002340: be804ec1
	s_barrier_wait 0xffff                                      // 000000002344: bf94ffff
	global_load_b32 v129, v[0:1], off                          // 000000002348: ee05007c 00000081 00000000
	v_add_co_u32 v0, s2, v46, s12                              // 000000002354: d7000200 0200192e
	global_load_b32 v130, v[2:3], off                          // 00000000235c: ee05007c 00000082 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002368: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v47, s2                 // 00000000236c: d5207c01 000a5e0d
	v_add_co_u32 v2, s2, v50, s12                              // 000000002374: d7000202 02001932
	global_load_b32 v131, v[4:5], off                          // 00000000237c: ee05007c 00000083 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002388: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v51, s2                 // 00000000238c: d5207c03 000a660d
	v_add_co_u32 v4, s2, v53, s12                              // 000000002394: d7000204 02001935
	s_wait_alu depctr_va_sdst(0)                               // 00000000239c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v54, s2                 // 0000000023a0: d5207c05 000a6c0d
	v_add_co_u32 v6, s2, v56, s12                              // 0000000023a8: d7000206 02001938
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v57, s2                 // 0000000023b4: d5207c07 000a720d
	v_add_co_u32 v91, s2, v58, s12                             // 0000000023bc: d700025b 0200193a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c4: bf88f19f
	v_add_co_ci_u32_e64 v92, null, s13, v60, s2                // 0000000023c8: d5207c5c 000a780d
	global_load_b32 v132, v[0:1], off                          // 0000000023d0: ee05007c 00000084 00000000
	v_add_co_u32 v0, s2, v63, s12                              // 0000000023dc: d7000200 0200193f
	global_load_b32 v133, v[2:3], off                          // 0000000023e4: ee05007c 00000085 00000002
	s_wait_alu depctr_va_sdst(0)                               // 0000000023f0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v64, s2                 // 0000000023f4: d5207c01 000a800d
	v_add_co_u32 v2, s2, s14, v16                              // 0000000023fc: d7000202 0202200e
	global_load_b32 v134, v[4:5], off                          // 000000002404: ee05007c 00000086 00000004
	s_wait_alu depctr_va_sdst(0)                               // 000000002410: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s15, v17, s2                 // 000000002414: d5207c03 000a220f
	v_add_co_u32 v4, s2, v67, s12                              // 00000000241c: d7000204 02001943
	global_load_b32 v135, v[6:7], off                          // 000000002424: ee05007c 00000087 00000006
	s_wait_alu depctr_va_sdst(0)                               // 000000002430: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v68, s2                 // 000000002434: d5207c05 000a880d
	v_add_co_u32 v6, s2, v70, s12                              // 00000000243c: d7000206 02001946
	global_load_b32 v136, v[91:92], off                        // 000000002444: ee05007c 00000088 0000005b
	s_wait_alu depctr_va_sdst(0)                               // 000000002450: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v72, s2                 // 000000002454: d5207c07 000a900d
	v_add_co_u32 v91, s2, v75, s12                             // 00000000245c: d700025b 0200194b
	s_wait_alu depctr_va_sdst(0)                               // 000000002464: bf88f19f
	v_add_co_ci_u32_e64 v92, null, s13, v76, s2                // 000000002468: d5207c5c 000a980d
	global_load_b32 v137, v[0:1], off                          // 000000002470: ee05007c 00000089 00000000
	v_add_co_u32 v0, s2, v79, s12                              // 00000000247c: d7000200 0200194f
	global_load_b32 v138, v[2:3], off                          // 000000002484: ee05007c 0000008a 00000002
	s_wait_alu depctr_va_sdst(0)                               // 000000002490: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v80, s2                 // 000000002494: d5207c01 000aa00d
	v_add_co_u32 v2, s2, v81, s12                              // 00000000249c: d7000202 02001951
	global_load_b32 v139, v[4:5], off                          // 0000000024a4: ee05007c 0000008b 00000004
	s_wait_alu depctr_va_sdst(0)                               // 0000000024b0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s13, v82, s2                 // 0000000024b4: d5207c03 000aa40d
	v_add_co_u32 v4, s2, v83, s12                              // 0000000024bc: d7000204 02001953
	global_load_b32 v140, v[6:7], off                          // 0000000024c4: ee05007c 0000008c 00000006
	s_wait_alu depctr_va_sdst(0)                               // 0000000024d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s13, v84, s2                 // 0000000024d4: d5207c05 000aa80d
	v_add_co_u32 v6, s2, v85, s12                              // 0000000024dc: d7000206 02001955
	global_load_b32 v141, v[91:92], off                        // 0000000024e4: ee05007c 0000008d 0000005b
	s_wait_alu depctr_va_sdst(0)                               // 0000000024f0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v86, s2                 // 0000000024f4: d5207c07 000aac0d
	v_add_co_u32 v91, s2, v87, s12                             // 0000000024fc: d700025b 02001957
	s_wait_alu depctr_va_sdst(0)                               // 000000002504: bf88f19f
	v_add_co_ci_u32_e64 v92, null, s13, v88, s2                // 000000002508: d5207c5c 000ab00d
	s_clause 0x4                                               // 000000002510: bf850004
	global_load_b32 v142, v[0:1], off                          // 000000002514: ee05007c 0000008e 00000000
	global_load_b32 v143, v[2:3], off                          // 000000002520: ee05007c 0000008f 00000002
	global_load_b32 v144, v[4:5], off                          // 00000000252c: ee05007c 00000090 00000004
	global_load_b32 v145, v[6:7], off                          // 000000002538: ee05007c 00000091 00000006
	global_load_b32 v146, v[91:92], off                        // 000000002544: ee05007c 00000092 0000005b
	ds_load_2addr_b64 v[113:116], v24 offset1:2                // 000000002550: d9dc0200 71000018
	ds_load_2addr_b64 v[117:120], v89 offset1:2                // 000000002558: d9dc0200 75000059
	ds_load_2addr_b64 v[121:124], v90 offset1:2                // 000000002560: d9dc0200 7900005a
	ds_load_2addr_b64 v[125:128], v25 offset1:2                // 000000002568: d9dc0200 7d000019
	s_add_nc_u64 s[4:5], s[4:5], 1                             // 000000002570: a9848104
	s_wait_alu depctr_sa_sdst(0)                               // 000000002574: bf88ff9e
	s_cmp_lg_u64 s[4:5], s[10:11]                              // 000000002578: bf110a04
	s_wait_dscnt 0x2                                           // 00000000257c: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[113:114], v[117:118], 0// 000000002580: cc464000 1a02eb71
	s_wait_dscnt 0x1                                           // 000000002588: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[91:98], v[113:114], v[121:122], 0// 00000000258c: cc46405b 1a02f371
	s_wait_dscnt 0x0                                           // 000000002594: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[99:106], v[125:126], v[117:118], 0// 000000002598: cc464063 1a02eb7d
	v_wmma_f32_16x16x16_fp8_fp8 v[107:114], v[125:126], v[121:122], 0// 0000000025a0: cc46406b 1a02f37d
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[119:120], v[0:7]// 0000000025a8: cc464000 1c02ef73
	v_wmma_f32_16x16x16_fp8_fp8 v[91:98], v[115:116], v[123:124], v[91:98]// 0000000025b0: cc46405b 1d6ef773
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000025b8: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[99:106], v[127:128], v[119:120], v[99:106]// 0000000025bc: cc464063 1d8eef7f
	v_wmma_f32_16x16x16_fp8_fp8 v[107:114], v[127:128], v[123:124], v[107:114]// 0000000025c4: cc46406b 1daef77f
	s_wait_loadcnt 0xf                                         // 0000000025cc: bfc0000f
	v_dual_mul_f32 v115, v129, v130 :: v_dual_mul_f32 v116, v130, v131// 0000000025d0: c8c70581 73750782
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000025d8: bf870091
	v_dual_mul_f32 v0, v0, v115 :: v_dual_mul_f32 v1, v1, v116 // 0000000025dc: c8c6e700 0000e901
	v_add_f32_e32 v18, v18, v0                                 // 0000000025e4: 06240112
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 0000000025e8: bf8700b2
	v_add_f32_e32 v78, v78, v1                                 // 0000000025ec: 069c034e
	s_wait_loadcnt 0xd                                         // 0000000025f0: bfc0000d
	v_dual_mul_f32 v117, v130, v132 :: v_dual_mul_f32 v118, v130, v133// 0000000025f4: c8c70982 75770b82
	v_mul_f32_e32 v2, v2, v117                                 // 0000000025fc: 1004eb02
	s_wait_loadcnt 0xc                                         // 000000002600: bfc0000c
	v_mul_f32_e32 v119, v130, v134                             // 000000002604: 10ef0d82
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000002608: bf8701b3
	v_mul_f32_e32 v3, v3, v118                                 // 00000000260c: 1006ed03
	s_wait_loadcnt 0xb                                         // 000000002610: bfc0000b
	v_dual_add_f32 v77, v77, v2 :: v_dual_mul_f32 v120, v130, v135// 000000002614: c906054d 4d790f82
	v_mul_f32_e32 v4, v4, v119                                 // 00000000261c: 1008ef04
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000002620: bf870143
	v_add_f32_e32 v74, v74, v3                                 // 000000002624: 0694074a
	s_wait_loadcnt 0xa                                         // 000000002628: bfc0000a
	v_mul_f32_e32 v121, v130, v136                             // 00000000262c: 10f31182
	v_mul_f32_e32 v5, v5, v120                                 // 000000002630: 100af105
	v_dual_add_f32 v73, v73, v4 :: v_dual_mul_f32 v6, v6, v121 // 000000002634: c9060949 4906f306
	s_delay_alu instid0(valu_dep_2)                            // 00000000263c: bf870002
	v_add_f32_e32 v71, v71, v5                                 // 000000002640: 068e0b47
	s_wait_loadcnt 0x8                                         // 000000002644: bfc00008
	v_dual_mul_f32 v122, v130, v137 :: v_dual_mul_f32 v123, v129, v138// 000000002648: c8c71382 7a7b1581
	v_dual_mul_f32 v124, v131, v138 :: v_dual_mul_f32 v125, v132, v138// 000000002650: c8c71583 7c7d1584
	v_dual_mul_f32 v126, v133, v138 :: v_dual_mul_f32 v127, v134, v138// 000000002658: c8c71585 7e7f1586
	v_dual_mul_f32 v128, v135, v138 :: v_dual_mul_f32 v129, v136, v138// 000000002660: c8c71587 80811588
	s_wait_loadcnt 0x7                                         // 000000002668: bfc00007
	v_dual_mul_f32 v131, v137, v138 :: v_dual_mul_f32 v132, v130, v139// 00000000266c: c8c71589 83851782
	v_mul_f32_e32 v139, v138, v139                             // 000000002674: 1117178a
	s_wait_loadcnt 0x6                                         // 000000002678: bfc00006
	v_mul_f32_e32 v133, v130, v140                             // 00000000267c: 110b1982
	v_dual_mul_f32 v140, v138, v140 :: v_dual_mul_f32 v7, v7, v122// 000000002680: c8c7198a 8c06f507
	v_dual_mul_f32 v91, v91, v123 :: v_dual_mul_f32 v92, v92, v124// 000000002688: c8c6f75b 5b5cf95c
	s_wait_loadcnt 0x5                                         // 000000002690: bfc00005
	v_mul_f32_e32 v134, v130, v141                             // 000000002694: 110d1b82
	v_mul_f32_e32 v141, v138, v141                             // 000000002698: 111b1b8a
	v_dual_mul_f32 v93, v93, v125 :: v_dual_mul_f32 v94, v94, v126// 00000000269c: c8c6fb5d 5d5efd5e
	v_dual_mul_f32 v95, v95, v127 :: v_dual_mul_f32 v96, v96, v128// 0000000026a4: c8c6ff5f 5f610160
	v_mul_f32_e32 v97, v97, v129                               // 0000000026ac: 10c30361
	s_wait_loadcnt 0x3                                         // 0000000026b0: bfc00003
	v_dual_mul_f32 v135, v130, v142 :: v_dual_mul_f32 v136, v130, v143// 0000000026b4: c8c71d82 87891f82
	s_wait_loadcnt 0x2                                         // 0000000026bc: bfc00002
	v_mul_f32_e32 v137, v130, v144                             // 0000000026c0: 11132182
	s_wait_loadcnt 0x0                                         // 0000000026c4: bfc00000
	v_dual_mul_f32 v147, v130, v145 :: v_dual_mul_f32 v130, v130, v146// 0000000026c8: c8c72382 93832582
	v_dual_mul_f32 v142, v138, v142 :: v_dual_mul_f32 v143, v138, v143// 0000000026d0: c8c71d8a 8e8f1f8a
	v_dual_mul_f32 v144, v138, v144 :: v_dual_mul_f32 v145, v138, v145// 0000000026d8: c8c7218a 9091238a
	v_mul_f32_e32 v138, v138, v146                             // 0000000026e0: 1115258a
	v_dual_mul_f32 v98, v98, v131 :: v_dual_mul_f32 v99, v99, v132// 0000000026e4: c8c70762 62630963
	v_dual_mul_f32 v100, v100, v133 :: v_dual_mul_f32 v101, v101, v134// 0000000026ec: c8c70b64 64650d65
	v_dual_mul_f32 v102, v102, v135 :: v_dual_mul_f32 v103, v103, v136// 0000000026f4: c8c70f66 66671167
	v_dual_mul_f32 v104, v104, v137 :: v_dual_mul_f32 v105, v105, v147// 0000000026fc: c8c71368 68692769
	v_dual_mul_f32 v106, v106, v130 :: v_dual_mul_f32 v107, v107, v139// 000000002704: c8c7056a 6a6b176b
	v_dual_mul_f32 v108, v108, v140 :: v_dual_mul_f32 v109, v109, v141// 00000000270c: c8c7196c 6c6d1b6d
	v_dual_mul_f32 v110, v110, v142 :: v_dual_mul_f32 v111, v111, v143// 000000002714: c8c71d6e 6e6f1f6f
	v_dual_mul_f32 v112, v112, v144 :: v_dual_mul_f32 v113, v113, v145// 00000000271c: c8c72170 70712371
	v_mul_f32_e32 v114, v114, v138                             // 000000002724: 10e51572
	v_dual_add_f32 v69, v69, v6 :: v_dual_add_f32 v66, v66, v7 // 000000002728: c9080d45 45420f42
	v_dual_add_f32 v43, v43, v91 :: v_dual_add_f32 v42, v42, v92// 000000002730: c908b72b 2b2ab92a
	v_dual_add_f32 v41, v41, v93 :: v_dual_add_f32 v38, v38, v94// 000000002738: c908bb29 2926bd26
	v_dual_add_f32 v37, v37, v95 :: v_dual_add_f32 v36, v36, v96// 000000002740: c908bf25 2524c124
	v_add_f32_e32 v35, v35, v97                                // 000000002748: 0646c323
	v_add_f32_e32 v33, v33, v98                                // 00000000274c: 0642c521
	v_dual_add_f32 v65, v65, v99 :: v_dual_add_f32 v62, v62, v100// 000000002750: c908c741 413ec93e
	v_add_f32_e32 v61, v61, v101                               // 000000002758: 067acb3d
	v_add_f32_e32 v59, v59, v102                               // 00000000275c: 0676cd3b
	v_dual_add_f32 v55, v55, v103 :: v_dual_add_f32 v52, v52, v104// 000000002760: c908cf37 3734d134
	v_dual_add_f32 v49, v49, v105 :: v_dual_add_f32 v48, v48, v106// 000000002768: c908d331 3130d530
	v_add_f32_e32 v34, v34, v107                               // 000000002770: 0644d722
	v_dual_add_f32 v32, v32, v108 :: v_dual_add_f32 v31, v31, v109// 000000002774: c908d920 201edb1f
	v_dual_add_f32 v30, v30, v110 :: v_dual_add_f32 v29, v29, v111// 00000000277c: c908dd1e 1e1cdf1d
	v_dual_add_f32 v28, v28, v112 :: v_dual_add_f32 v27, v27, v113// 000000002784: c908e11c 1c1ae31b
	v_add_f32_e32 v26, v26, v114                               // 00000000278c: 0634e51a
	s_cbranch_scc0 37                                          // 000000002790: bfa10025 <tessera_rocm_scaled_matmul_lds_305100ba3501aae0+0xd28>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002794: bf88ff9e
	s_lshl_b64 s[12:13], s[4:5], 5                             // 000000002798: 848c8504
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, 0               // 00000000279c: ca100080 02020080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027a4: bf88ff9e
	v_add_co_u32 v0, s2, v20, s12                              // 0000000027a8: d7000200 02001914
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v21, s2                 // 0000000027b4: d5207c01 000a2a0d
	global_load_b128 v[4:7], v[0:1], off                       // 0000000027bc: ee05c07c 00000004 00000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 0000000027c8: ca100080 00000080
	s_and_saveexec_b32 s3, vcc_lo                              // 0000000027d0: be83206a
	s_cbranch_execz 8                                          // 0000000027d4: bfa50008 <tessera_rocm_scaled_matmul_lds_305100ba3501aae0+0xcf8>
	v_add_co_u32 v0, s2, v22, s12                              // 0000000027d8: d7000200 02001916
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s13, v23, s2                 // 0000000027e4: d5207c01 000a2e0d
	global_load_b128 v[0:3], v[0:1], off                       // 0000000027ec: ee05c07c 00000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000027fc: 8c7e037e
	s_barrier_signal -1                                        // 000000002800: be804ec1
	s_barrier_wait 0xffff                                      // 000000002804: bf94ffff
	s_wait_loadcnt 0x0                                         // 000000002808: bfc00000
	ds_store_b128 v19, v[4:7]                                  // 00000000280c: db7c0000 00000413
	s_and_saveexec_b32 s2, vcc_lo                              // 000000002814: be82206a
	s_cbranch_execz 65201                                      // 000000002818: bfa5feb1 <tessera_rocm_scaled_matmul_lds_305100ba3501aae0+0x7e0>
	ds_store_b128 v19, v[0:3] offset:6144                      // 00000000281c: db7c1800 00000013
	s_branch 65198                                             // 000000002824: bfa0feae <tessera_rocm_scaled_matmul_lds_305100ba3501aae0+0x7e0>
	s_load_b64 s[2:3], s[0:1], 0xa8                            // 000000002828: f4002080 f80000a8
	v_mul_lo_u32 v2, s7, v8                                    // 000000002830: d72c0002 02021007
	v_mul_lo_u32 v3, s6, v9                                    // 000000002838: d72c0003 02021206
	v_mad_co_u64_u32 v[0:1], null, s6, v8, 0                   // 000000002840: d6fe7c00 02021006
	v_bfe_u32 v4, v18, 16, 1                                   // 000000002848: d6100004 02052112
	v_or_b32_e32 v5, 0x400000, v18                             // 000000002850: 380a24ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v18, v18                           // 000000002858: 7c302512
	v_bfe_u32 v6, v78, 16, 1                                   // 00000000285c: d6100006 0205214e
	v_or_b32_e32 v7, 0x400000, v78                             // 000000002864: 380e9cff 00400000
	v_add3_u32 v4, v4, v18, 0x7fff                             // 00000000286c: d6550004 03fe2504 00007fff
	s_lshl_b64 s[0:1], s[6:7], 1                               // 000000002878: 84808106
	v_add3_u32 v1, v1, v3, v2                                  // 00000000287c: d6550001 040a0701
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000002884: 3e041481
	v_add3_u32 v6, v6, v78, 0x7fff                             // 000000002888: d6550006 03fe9d06 00007fff
	v_cndmask_b32_e32 v8, v4, v5, vcc_lo                       // 000000002894: 02100b04
	v_or_b32_e32 v11, 0x400000, v77                            // 000000002898: 38169aff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000028a0: 3e000081
	v_bfe_u32 v15, v74, 16, 1                                  // 0000000028a4: d610000f 0205214a
	v_or_b32_e32 v16, 0x400000, v74                            // 0000000028ac: 382094ff 00400000
	v_or_b32_e32 v19, 0x400000, v71                            // 0000000028b4: 38268eff 00400000
	v_bfe_u32 v21, v69, 16, 1                                  // 0000000028bc: d6100015 02052145
	v_or_b32_e32 v22, 0x400000, v69                            // 0000000028c4: 382c8aff 00400000
	s_wait_kmcnt 0x0                                           // 0000000028cc: bfc70000
	v_add_co_u32 v4, vcc_lo, s2, v0                            // 0000000028d0: d7006a04 02020002
	s_wait_alu depctr_va_vcc(0)                                // 0000000028d8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s3, v1, vcc_lo               // 0000000028dc: d5207c05 01aa0203
	v_cmp_u_f32_e32 vcc_lo, v78, v78                           // 0000000028e4: 7c309d4e
	v_add3_u32 v15, v15, v74, 0x7fff                           // 0000000028e8: d655000f 03fe950f 00007fff
	v_add3_u32 v21, v21, v69, 0x7fff                           // 0000000028f4: d6550015 03fe8b15 00007fff
	v_mul_lo_u32 v23, s6, v13                                  // 000000002900: d72c0017 02021a06
	v_or_b32_e32 v24, 0x400000, v66                            // 000000002908: 383084ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002910: bf88ff9d
	v_cndmask_b32_e32 v9, v6, v7, vcc_lo                       // 000000002914: 02120f06
	v_add_co_u32 v0, vcc_lo, v4, v2                            // 000000002918: d7006a00 02020504
	s_wait_alu depctr_va_vcc(0)                                // 000000002920: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v5, v3, vcc_lo               // 000000002924: d5207c01 01aa0705
	v_add_co_u32 v7, vcc_lo, v4, s0                            // 00000000292c: d7006a07 02000104
	v_bfe_u32 v6, v77, 16, 1                                   // 000000002934: d6100006 0205214d
	s_wait_alu depctr_va_vcc(0)                                // 00000000293c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v5, vcc_lo              // 000000002940: d5207c0a 01aa0a01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002948: bf870193
	v_add_co_u32 v4, vcc_lo, v7, v2                            // 00000000294c: d7006a04 02020507
	v_add3_u32 v6, v6, v77, 0x7fff                             // 000000002954: d6550006 03fe9b06 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002960: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002964: bf870003
	v_add_co_ci_u32_e64 v5, null, v10, v3, vcc_lo              // 000000002968: d5207c05 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000002970: 7c309b4d
	v_bfe_u32 v25, v62, 16, 1                                  // 000000002974: d6100019 0205213e
	v_or_b32_e32 v39, 0x400000, v62                            // 00000000297c: 384e7cff 00400000
	v_or_b32_e32 v45, 0x400000, v59                            // 000000002984: 385a76ff 00400000
	v_bfe_u32 v47, v55, 16, 1                                  // 00000000298c: d610002f 02052137
	s_wait_alu depctr_va_vcc(0)                                // 000000002994: bf88ff9d
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo                     // 000000002998: 02161706
	v_add_co_u32 v14, vcc_lo, v7, s0                           // 00000000299c: d7006a0e 02000107
	s_wait_alu depctr_va_vcc(0)                                // 0000000029a4: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 0000000029a8: d5207c0a 01aa1401
	v_add3_u32 v25, v25, v62, 0x7fff                           // 0000000029b0: d6550019 03fe7d19 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000029bc: bf8701a3
	v_add_co_u32 v6, vcc_lo, v14, v2                           // 0000000029c0: d7006a06 0202050e
	s_wait_alu depctr_va_vcc(0)                                // 0000000029c8: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v10, v3, vcc_lo              // 0000000029cc: d5207c07 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v74, v74                           // 0000000029d4: 7c30954a
	s_clause 0x2                                               // 0000000029d8: bf850002
	global_store_d16_hi_b16 v[0:1], v8, off                    // 0000000029dc: ee09407c 04000000 00000000
	global_store_d16_hi_b16 v[4:5], v9, off                    // 0000000029e8: ee09407c 04800000 00000004
	global_store_d16_hi_b16 v[6:7], v11, off                   // 0000000029f4: ee09407c 05800000 00000006
	v_bfe_u32 v8, v73, 16, 1                                   // 000000002a00: d6100008 02052149
	v_add3_u32 v47, v47, v55, 0x7fff                           // 000000002a08: d655002f 03fe6f2f 00007fff
	v_or_b32_e32 v50, 0x400000, v55                            // 000000002a14: 38646eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002a1c: bf88ff9d
	v_cndmask_b32_e32 v16, v15, v16, vcc_lo                    // 000000002a20: 0220210f
	v_add_co_u32 v11, vcc_lo, v14, s0                          // 000000002a24: d7006a0b 0200010e
	s_wait_alu depctr_va_vcc(0)                                // 000000002a2c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 000000002a30: d5207c0a 01aa1401
	v_add3_u32 v14, v8, v73, 0x7fff                            // 000000002a38: d655000e 03fe9308 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002a44: bf870003
	v_add_co_u32 v8, vcc_lo, v11, v2                           // 000000002a48: d7006a08 0202050b
	v_or_b32_e32 v15, 0x400000, v73                            // 000000002a50: 381e92ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002a58: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v10, v3, vcc_lo              // 000000002a5c: d5207c09 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 000000002a64: 7c309349
	v_or_b32_e32 v53, 0x400000, v49                            // 000000002a68: 386a62ff 00400000
	v_bfe_u32 v54, v48, 16, 1                                  // 000000002a70: d6100036 02052130
	s_wait_alu depctr_va_vcc(0)                                // 000000002a78: bf88ff9d
	v_cndmask_b32_e32 v17, v14, v15, vcc_lo                    // 000000002a7c: 02221f0e
	v_add_co_u32 v15, vcc_lo, v11, s0                          // 000000002a80: d7006a0f 0200010b
	v_bfe_u32 v14, v71, 16, 1                                  // 000000002a88: d610000e 02052147
	s_wait_alu depctr_va_vcc(0)                                // 000000002a90: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v10, vcc_lo             // 000000002a94: d5207c12 01aa1401
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002a9c: bf870193
	v_add_co_u32 v10, vcc_lo, v15, v2                          // 000000002aa0: d7006a0a 0202050f
	v_add3_u32 v14, v14, v71, 0x7fff                           // 000000002aa8: d655000e 03fe8f0e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002ab4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ab8: bf870003
	v_add_co_ci_u32_e64 v11, null, v18, v3, vcc_lo             // 000000002abc: d5207c0b 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000002ac4: 7c308f47
	v_add3_u32 v54, v54, v48, 0x7fff                           // 000000002ac8: d6550036 03fe6136 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002ad4: bf88ff9d
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo                    // 000000002ad8: 0226270e
	v_add_co_u32 v20, vcc_lo, v15, s0                          // 000000002adc: d7006a14 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 000000002ae4: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000002ae8: d5207c12 01aa2401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002af0: bf870122
	v_add_co_u32 v14, vcc_lo, v20, v2                          // 000000002af4: d7006a0e 02020514
	s_wait_alu depctr_va_vcc(0)                                // 000000002afc: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v18, v3, vcc_lo             // 000000002b00: d5207c0f 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000002b08: 7c308b45
	s_clause 0x2                                               // 000000002b0c: bf850002
	global_store_d16_hi_b16 v[8:9], v16, off                   // 000000002b10: ee09407c 08000000 00000008
	global_store_d16_hi_b16 v[10:11], v17, off                 // 000000002b1c: ee09407c 08800000 0000000a
	global_store_d16_hi_b16 v[14:15], v19, off                 // 000000002b28: ee09407c 09800000 0000000e
	v_bfe_u32 v16, v66, 16, 1                                  // 000000002b34: d6100010 02052142
	s_wait_alu depctr_va_vcc(0)                                // 000000002b3c: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v22, vcc_lo                    // 000000002b40: 022a2d15
	v_add_co_u32 v19, vcc_lo, v20, s0                          // 000000002b44: d7006a13 02000114
	s_wait_alu depctr_va_vcc(0)                                // 000000002b4c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000002b50: d5207c12 01aa2401
	v_add3_u32 v20, v16, v66, 0x7fff                           // 000000002b58: d6550014 03fe8510 00007fff
	v_mul_lo_u32 v22, s7, v12                                  // 000000002b64: d72c0016 02021807
	v_mad_co_u64_u32 v[12:13], null, s6, v12, 0                // 000000002b6c: d6fe7c0c 02021806
	v_add_co_u32 v16, vcc_lo, v19, v2                          // 000000002b74: d7006a10 02020513
	s_wait_alu depctr_va_vcc(0)                                // 000000002b7c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v18, v3, vcc_lo             // 000000002b80: d5207c11 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 000000002b88: 7c308542
	s_delay_alu instid0(valu_dep_4)                            // 000000002b8c: bf870004
	v_add3_u32 v13, v13, v23, v22                              // 000000002b90: d655000d 045a2f0d
	v_bfe_u32 v22, v65, 16, 1                                  // 000000002b98: d6100016 02052141
	s_wait_alu depctr_va_vcc(0)                                // 000000002ba0: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v24, vcc_lo                    // 000000002ba4: 02283114
	v_add_co_u32 v19, vcc_lo, v19, s0                          // 000000002ba8: d7006a13 02000113
	s_wait_alu depctr_va_vcc(0)                                // 000000002bb0: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s1, v18, vcc_lo             // 000000002bb4: d5207c17 01aa2401
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002bbc: 3e181881
	s_delay_alu instid0(valu_dep_3)                            // 000000002bc0: bf870003
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000002bc4: d7006a12 02020513
	v_add3_u32 v22, v22, v65, 0x7fff                           // 000000002bcc: d6550016 03fe8316 00007fff
	v_or_b32_e32 v24, 0x400000, v65                            // 000000002bd8: 383082ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002be0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v23, v3, vcc_lo             // 000000002be4: d5207c13 01aa0717
	v_cmp_u_f32_e32 vcc_lo, v65, v65                           // 000000002bec: 7c308341
	s_wait_alu depctr_va_vcc(0)                                // 000000002bf0: bf88ff9d
	v_cndmask_b32_e32 v22, v22, v24, vcc_lo                    // 000000002bf4: 022c3116
	v_add_co_u32 v23, vcc_lo, s2, v12                          // 000000002bf8: d7006a17 02021802
	s_wait_alu depctr_va_vcc(0)                                // 000000002c00: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s3, v13, vcc_lo             // 000000002c04: d5207c18 01aa1a03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c0c: bf870122
	v_add_co_u32 v12, vcc_lo, v23, v2                          // 000000002c10: d7006a0c 02020517
	s_wait_alu depctr_va_vcc(0)                                // 000000002c18: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v24, v3, vcc_lo             // 000000002c1c: d5207c0d 01aa0718
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000002c24: 7c307d3e
	s_clause 0x2                                               // 000000002c28: bf850002
	global_store_d16_hi_b16 v[16:17], v21, off                 // 000000002c2c: ee09407c 0a800000 00000010
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002c38: ee09407c 0a000000 00000012
	global_store_d16_hi_b16 v[12:13], v22, off                 // 000000002c44: ee09407c 0b000000 0000000c
	v_bfe_u32 v20, v61, 16, 1                                  // 000000002c50: d6100014 0205213d
	s_wait_alu depctr_va_vcc(0)                                // 000000002c58: bf88ff9d
	v_cndmask_b32_e32 v39, v25, v39, vcc_lo                    // 000000002c5c: 024e4f19
	v_add_co_u32 v22, vcc_lo, v23, s0                          // 000000002c60: d7006a16 02000117
	s_wait_alu depctr_va_vcc(0)                                // 000000002c68: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s1, v24, vcc_lo             // 000000002c6c: d5207c17 01aa3001
	v_add3_u32 v24, v20, v61, 0x7fff                           // 000000002c74: d6550018 03fe7b14 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c80: bf870003
	v_add_co_u32 v20, vcc_lo, v22, v2                          // 000000002c84: d7006a14 02020516
	v_or_b32_e32 v25, 0x400000, v61                            // 000000002c8c: 38327aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c94: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v23, v3, vcc_lo             // 000000002c98: d5207c15 01aa0717
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000002ca0: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 000000002ca4: bf88ff9d
	v_cndmask_b32_e32 v40, v24, v25, vcc_lo                    // 000000002ca8: 02503318
	v_add_co_u32 v25, vcc_lo, v22, s0                          // 000000002cac: d7006a19 02000116
	v_bfe_u32 v24, v59, 16, 1                                  // 000000002cb4: d6100018 0205213b
	s_wait_alu depctr_va_vcc(0)                                // 000000002cbc: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s1, v23, vcc_lo             // 000000002cc0: d5207c2c 01aa2e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002cc8: bf870193
	v_add_co_u32 v22, vcc_lo, v25, v2                          // 000000002ccc: d7006a16 02020519
	v_add3_u32 v24, v24, v59, 0x7fff                           // 000000002cd4: d6550018 03fe7718 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002ce0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ce4: bf870003
	v_add_co_ci_u32_e64 v23, null, v44, v3, vcc_lo             // 000000002ce8: d5207c17 01aa072c
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000002cf0: 7c30773b
	s_wait_alu depctr_va_vcc(0)                                // 000000002cf4: bf88ff9d
	v_cndmask_b32_e32 v45, v24, v45, vcc_lo                    // 000000002cf8: 025a5b18
	v_add_co_u32 v46, vcc_lo, v25, s0                          // 000000002cfc: d7006a2e 02000119
	s_wait_alu depctr_va_vcc(0)                                // 000000002d04: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s1, v44, vcc_lo             // 000000002d08: d5207c2c 01aa5801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002d10: bf870122
	v_add_co_u32 v24, vcc_lo, v46, v2                          // 000000002d14: d7006a18 0202052e
	s_wait_alu depctr_va_vcc(0)                                // 000000002d1c: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v44, v3, vcc_lo             // 000000002d20: d5207c19 01aa072c
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000002d28: 7c306f37
	s_clause 0x2                                               // 000000002d2c: bf850002
	global_store_d16_hi_b16 v[20:21], v39, off                 // 000000002d30: ee09407c 13800000 00000014
	global_store_d16_hi_b16 v[22:23], v40, off                 // 000000002d3c: ee09407c 14000000 00000016
	global_store_d16_hi_b16 v[24:25], v45, off                 // 000000002d48: ee09407c 16800000 00000018
	v_bfe_u32 v39, v52, 16, 1                                  // 000000002d54: d6100027 02052134
	v_or_b32_e32 v55, 0x400000, v48                            // 000000002d5c: 386e60ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002d64: bf88ff9d
	v_cndmask_b32_e32 v50, v47, v50, vcc_lo                    // 000000002d68: 0264652f
	v_add_co_u32 v45, vcc_lo, v46, s0                          // 000000002d6c: d7006a2d 0200012e
	s_wait_alu depctr_va_vcc(0)                                // 000000002d74: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, s1, v44, vcc_lo             // 000000002d78: d5207c2c 01aa5801
	v_add3_u32 v46, v39, v52, 0x7fff                           // 000000002d80: d655002e 03fe6927 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d8c: bf870003
	v_add_co_u32 v39, vcc_lo, v45, v2                          // 000000002d90: d7006a27 0202052d
	v_or_b32_e32 v47, 0x400000, v52                            // 000000002d98: 385e68ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002da0: bf88ff9d
	v_add_co_ci_u32_e64 v40, null, v44, v3, vcc_lo             // 000000002da4: d5207c28 01aa072c
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000002dac: 7c306934
	s_wait_alu depctr_va_vcc(0)                                // 000000002db0: bf88ff9d
	v_cndmask_b32_e32 v51, v46, v47, vcc_lo                    // 000000002db4: 02665f2e
	v_add_co_u32 v47, vcc_lo, v45, s0                          // 000000002db8: d7006a2f 0200012d
	v_bfe_u32 v46, v49, 16, 1                                  // 000000002dc0: d610002e 02052131
	s_wait_alu depctr_va_vcc(0)                                // 000000002dc8: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, s1, v44, vcc_lo             // 000000002dcc: d5207c34 01aa5801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002dd4: bf870193
	v_add_co_u32 v44, vcc_lo, v47, v2                          // 000000002dd8: d7006a2c 0202052f
	v_add3_u32 v46, v46, v49, 0x7fff                           // 000000002de0: d655002e 03fe632e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002dec: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002df0: bf870003
	v_add_co_ci_u32_e64 v45, null, v52, v3, vcc_lo             // 000000002df4: d5207c2d 01aa0734
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 000000002dfc: 7c306331
	s_wait_alu depctr_va_vcc(0)                                // 000000002e00: bf88ff9d
	v_cndmask_b32_e32 v49, v46, v53, vcc_lo                    // 000000002e04: 02626b2e
	v_add_co_u32 v53, vcc_lo, v47, s0                          // 000000002e08: d7006a35 0200012f
	s_wait_alu depctr_va_vcc(0)                                // 000000002e10: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, s1, v52, vcc_lo             // 000000002e14: d5207c34 01aa6801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002e1c: bf870122
	v_add_co_u32 v46, vcc_lo, v53, v2                          // 000000002e20: d7006a2e 02020535
	s_wait_alu depctr_va_vcc(0)                                // 000000002e28: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, v52, v3, vcc_lo             // 000000002e2c: d5207c2f 01aa0734
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000002e34: 7c306130
	s_clause 0x2                                               // 000000002e38: bf850002
	global_store_d16_hi_b16 v[39:40], v50, off                 // 000000002e3c: ee09407c 19000000 00000027
	global_store_d16_hi_b16 v[44:45], v51, off                 // 000000002e48: ee09407c 19800000 0000002c
	global_store_d16_hi_b16 v[46:47], v49, off                 // 000000002e54: ee09407c 18800000 0000002e
	v_bfe_u32 v49, v43, 16, 1                                  // 000000002e60: d6100031 0205212b
	s_wait_alu depctr_va_vcc(0)                                // 000000002e68: bf88ff9d
	v_cndmask_b32_e32 v48, v54, v55, vcc_lo                    // 000000002e6c: 02606f36
	v_add_co_u32 v50, vcc_lo, v53, s0                          // 000000002e70: d7006a32 02000135
	s_wait_alu depctr_va_vcc(0)                                // 000000002e78: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s1, v52, vcc_lo             // 000000002e7c: d5207c33 01aa6801
	v_add3_u32 v49, v49, v43, 0x7fff                           // 000000002e84: d6550031 03fe5731 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e90: bf870003
	v_add_co_u32 v2, vcc_lo, v50, v2                           // 000000002e94: d7006a02 02020532
	v_or_b32_e32 v52, 0x400000, v43                            // 000000002e9c: 386856ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002ea4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v51, v3, vcc_lo              // 000000002ea8: d5207c03 01aa0733
	v_bfe_u32 v50, v42, 16, 1                                  // 000000002eb0: d6100032 0205212a
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 000000002eb8: 7c30572b
	global_store_d16_hi_b16 v[2:3], v48, off                   // 000000002ebc: ee09407c 18000000 00000002
	v_add3_u32 v48, v50, v42, 0x7fff                           // 000000002ec8: d6550030 03fe5532 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002ed4: bf88ff9d
	v_cndmask_b32_e32 v43, v49, v52, vcc_lo                    // 000000002ed8: 02566931
	v_bfe_u32 v49, v41, 16, 1                                  // 000000002edc: d6100031 02052129
	v_or_b32_e32 v50, 0x400000, v42                            // 000000002ee4: 386454ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000002eec: 7c30552a
	global_store_d16_hi_b16 v[0:1], v43, off offset:32         // 000000002ef0: ee09407c 15800000 00002000
	v_add3_u32 v0, v49, v41, 0x7fff                            // 000000002efc: d6550000 03fe5331 00007fff
	v_or_b32_e32 v1, 0x400000, v41                             // 000000002f08: 380252ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002f10: bf88ff9d
	v_cndmask_b32_e32 v42, v48, v50, vcc_lo                    // 000000002f14: 02546530
	v_bfe_u32 v43, v38, 16, 1                                  // 000000002f18: d610002b 02052126
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000002f20: 7c305329
	global_store_d16_hi_b16 v[4:5], v42, off offset:32         // 000000002f24: ee09407c 15000000 00002004
	v_add3_u32 v4, v43, v38, 0x7fff                            // 000000002f30: d6550004 03fe4d2b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002f3c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000002f40: 02000300
	v_bfe_u32 v1, v37, 16, 1                                   // 000000002f44: d6100001 02052125
	v_or_b32_e32 v5, 0x400000, v38                             // 000000002f4c: 380a4cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 000000002f54: 7c304d26
	global_store_d16_hi_b16 v[6:7], v0, off offset:32          // 000000002f58: ee09407c 00000000 00002006
	v_add3_u32 v0, v1, v37, 0x7fff                             // 000000002f64: d6550000 03fe4b01 00007fff
	v_or_b32_e32 v1, 0x400000, v37                             // 000000002f70: 38024aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002f78: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000002f7c: 02080b04
	v_bfe_u32 v5, v36, 16, 1                                   // 000000002f80: d6100005 02052124
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000002f88: 7c304b25
	v_bfe_u32 v6, v27, 16, 1                                   // 000000002f8c: d6100006 0205211b
	v_or_b32_e32 v7, 0x400000, v28                             // 000000002f94: 380e38ff 00400000
	global_store_d16_hi_b16 v[8:9], v4, off offset:32          // 000000002f9c: ee09407c 02000000 00002008
	v_add3_u32 v4, v5, v36, 0x7fff                             // 000000002fa8: d6550004 03fe4905 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002fb4: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000002fb8: 02000300
	v_bfe_u32 v1, v35, 16, 1                                   // 000000002fbc: d6100001 02052123
	v_or_b32_e32 v5, 0x400000, v36                             // 000000002fc4: 380a48ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000002fcc: 7c304924
	v_add3_u32 v6, v6, v27, 0x7fff                             // 000000002fd0: d6550006 03fe3706 00007fff
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 000000002fdc: ee09407c 00000000 0000200a
	v_add3_u32 v0, v1, v35, 0x7fff                             // 000000002fe8: d6550000 03fe4701 00007fff
	v_or_b32_e32 v1, 0x400000, v35                             // 000000002ff4: 380246ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002ffc: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003000: 02080b04
	v_bfe_u32 v5, v33, 16, 1                                   // 000000003004: d6100005 02052121
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 00000000300c: 7c304723
	v_or_b32_e32 v8, 0x400000, v27                             // 000000003010: 381036ff 00400000
	v_or_b32_e32 v9, 0x400000, v26                             // 000000003018: 381234ff 00400000
	global_store_d16_hi_b16 v[14:15], v4, off offset:32        // 000000003020: ee09407c 02000000 0000200e
	v_add3_u32 v4, v5, v33, 0x7fff                             // 00000000302c: d6550004 03fe4305 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003038: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000303c: 02000300
	v_bfe_u32 v1, v34, 16, 1                                   // 000000003040: d6100001 02052122
	v_or_b32_e32 v5, 0x400000, v33                             // 000000003048: 380a42ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000003050: 7c304321
	global_store_d16_hi_b16 v[16:17], v0, off offset:32        // 000000003054: ee09407c 00000000 00002010
	v_add3_u32 v0, v1, v34, 0x7fff                             // 000000003060: d6550000 03fe4501 00007fff
	v_or_b32_e32 v1, 0x400000, v34                             // 00000000306c: 380244ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003074: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003078: 02080b04
	v_bfe_u32 v5, v32, 16, 1                                   // 00000000307c: d6100005 02052120
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 000000003084: 7c304522
	global_store_d16_hi_b16 v[18:19], v4, off offset:32        // 000000003088: ee09407c 02000000 00002012
	v_add3_u32 v4, v5, v32, 0x7fff                             // 000000003094: d6550004 03fe4105 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000030a0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000030a4: 02000300
	v_bfe_u32 v1, v31, 16, 1                                   // 0000000030a8: d6100001 0205211f
	v_or_b32_e32 v5, 0x400000, v32                             // 0000000030b0: 380a40ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 0000000030b8: 7c304120
	global_store_d16_hi_b16 v[12:13], v0, off offset:32        // 0000000030bc: ee09407c 00000000 0000200c
	v_add3_u32 v0, v1, v31, 0x7fff                             // 0000000030c8: d6550000 03fe3f01 00007fff
	v_or_b32_e32 v1, 0x400000, v31                             // 0000000030d4: 38023eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000030dc: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000030e0: 02080b04
	v_bfe_u32 v5, v30, 16, 1                                   // 0000000030e4: d6100005 0205211e
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 0000000030ec: 7c303f1f
	global_store_d16_hi_b16 v[20:21], v4, off offset:32        // 0000000030f0: ee09407c 02000000 00002014
	v_add3_u32 v4, v5, v30, 0x7fff                             // 0000000030fc: d6550004 03fe3d05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003108: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000310c: 02000300
	v_bfe_u32 v1, v29, 16, 1                                   // 000000003110: d6100001 0205211d
	v_or_b32_e32 v5, 0x400000, v30                             // 000000003118: 380a3cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000003120: 7c303d1e
	global_store_d16_hi_b16 v[22:23], v0, off offset:32        // 000000003124: ee09407c 00000000 00002016
	v_add3_u32 v0, v1, v29, 0x7fff                             // 000000003130: d6550000 03fe3b01 00007fff
	v_or_b32_e32 v1, 0x400000, v29                             // 00000000313c: 38023aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003144: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003148: 02080b04
	v_bfe_u32 v5, v28, 16, 1                                   // 00000000314c: d6100005 0205211c
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000003154: 7c303b1d
	s_delay_alu instid0(valu_dep_2)                            // 000000003158: bf870002
	v_add3_u32 v5, v5, v28, 0x7fff                             // 00000000315c: d6550005 03fe3905 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003168: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000316c: 02000300
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 000000003170: 7c30391c
	v_bfe_u32 v1, v26, 16, 1                                   // 000000003174: d6100001 0205211a
	s_wait_alu depctr_va_vcc(0)                                // 00000000317c: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000003180: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000003184: 7c30371b
	s_delay_alu instid0(valu_dep_3)                            // 000000003188: bf870003
	v_add3_u32 v1, v1, v26, 0x7fff                             // 00000000318c: d6550001 03fe3501 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003198: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo                       // 00000000319c: 020c1106
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 0000000031a0: 7c30351a
	s_wait_alu depctr_va_vcc(0)                                // 0000000031a4: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo                       // 0000000031a8: 02021301
	s_clause 0x3                                               // 0000000031ac: bf850003
	global_store_d16_hi_b16 v[24:25], v4, off offset:32        // 0000000031b0: ee09407c 02000000 00002018
	global_store_d16_hi_b16 v[39:40], v0, off offset:32        // 0000000031bc: ee09407c 00000000 00002027
	global_store_d16_hi_b16 v[44:45], v5, off offset:32        // 0000000031c8: ee09407c 02800000 0000202c
	global_store_d16_hi_b16 v[46:47], v6, off offset:32        // 0000000031d4: ee09407c 03000000 0000202e
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 0000000031e0: ee09407c 00800000 00002002
	s_nop 0                                                    // 0000000031ec: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000031f0: bfb60003
	s_endpgm                                                   // 0000000031f4: bfb00000
	s_code_end                                                 // 0000000031f8: bf9f0000
	s_code_end                                                 // 0000000031fc: bf9f0000
	s_code_end                                                 // 000000003200: bf9f0000
	s_code_end                                                 // 000000003204: bf9f0000
	s_code_end                                                 // 000000003208: bf9f0000
	s_code_end                                                 // 00000000320c: bf9f0000
	s_code_end                                                 // 000000003210: bf9f0000
	s_code_end                                                 // 000000003214: bf9f0000
	s_code_end                                                 // 000000003218: bf9f0000
	s_code_end                                                 // 00000000321c: bf9f0000
	s_code_end                                                 // 000000003220: bf9f0000
	s_code_end                                                 // 000000003224: bf9f0000
	s_code_end                                                 // 000000003228: bf9f0000
	s_code_end                                                 // 00000000322c: bf9f0000
	s_code_end                                                 // 000000003230: bf9f0000
	s_code_end                                                 // 000000003234: bf9f0000
	s_code_end                                                 // 000000003238: bf9f0000
	s_code_end                                                 // 00000000323c: bf9f0000
	s_code_end                                                 // 000000003240: bf9f0000
	s_code_end                                                 // 000000003244: bf9f0000
	s_code_end                                                 // 000000003248: bf9f0000
	s_code_end                                                 // 00000000324c: bf9f0000
	s_code_end                                                 // 000000003250: bf9f0000
	s_code_end                                                 // 000000003254: bf9f0000
	s_code_end                                                 // 000000003258: bf9f0000
	s_code_end                                                 // 00000000325c: bf9f0000
	s_code_end                                                 // 000000003260: bf9f0000
	s_code_end                                                 // 000000003264: bf9f0000
	s_code_end                                                 // 000000003268: bf9f0000
	s_code_end                                                 // 00000000326c: bf9f0000
	s_code_end                                                 // 000000003270: bf9f0000
	s_code_end                                                 // 000000003274: bf9f0000
	s_code_end                                                 // 000000003278: bf9f0000
	s_code_end                                                 // 00000000327c: bf9f0000
	s_code_end                                                 // 000000003280: bf9f0000
	s_code_end                                                 // 000000003284: bf9f0000
	s_code_end                                                 // 000000003288: bf9f0000
	s_code_end                                                 // 00000000328c: bf9f0000
	s_code_end                                                 // 000000003290: bf9f0000
	s_code_end                                                 // 000000003294: bf9f0000
	s_code_end                                                 // 000000003298: bf9f0000
	s_code_end                                                 // 00000000329c: bf9f0000
	s_code_end                                                 // 0000000032a0: bf9f0000
	s_code_end                                                 // 0000000032a4: bf9f0000
	s_code_end                                                 // 0000000032a8: bf9f0000
	s_code_end                                                 // 0000000032ac: bf9f0000
	s_code_end                                                 // 0000000032b0: bf9f0000
	s_code_end                                                 // 0000000032b4: bf9f0000
	s_code_end                                                 // 0000000032b8: bf9f0000
	s_code_end                                                 // 0000000032bc: bf9f0000
	s_code_end                                                 // 0000000032c0: bf9f0000
	s_code_end                                                 // 0000000032c4: bf9f0000
	s_code_end                                                 // 0000000032c8: bf9f0000
	s_code_end                                                 // 0000000032cc: bf9f0000
	s_code_end                                                 // 0000000032d0: bf9f0000
	s_code_end                                                 // 0000000032d4: bf9f0000
	s_code_end                                                 // 0000000032d8: bf9f0000
	s_code_end                                                 // 0000000032dc: bf9f0000
	s_code_end                                                 // 0000000032e0: bf9f0000
	s_code_end                                                 // 0000000032e4: bf9f0000
	s_code_end                                                 // 0000000032e8: bf9f0000
	s_code_end                                                 // 0000000032ec: bf9f0000
	s_code_end                                                 // 0000000032f0: bf9f0000
	s_code_end                                                 // 0000000032f4: bf9f0000
	s_code_end                                                 // 0000000032f8: bf9f0000
	s_code_end                                                 // 0000000032fc: bf9f0000
	s_code_end                                                 // 000000003300: bf9f0000
	s_code_end                                                 // 000000003304: bf9f0000
	s_code_end                                                 // 000000003308: bf9f0000
	s_code_end                                                 // 00000000330c: bf9f0000
	s_code_end                                                 // 000000003310: bf9f0000
	s_code_end                                                 // 000000003314: bf9f0000
	s_code_end                                                 // 000000003318: bf9f0000
	s_code_end                                                 // 00000000331c: bf9f0000
	s_code_end                                                 // 000000003320: bf9f0000
	s_code_end                                                 // 000000003324: bf9f0000
	s_code_end                                                 // 000000003328: bf9f0000
	s_code_end                                                 // 00000000332c: bf9f0000
	s_code_end                                                 // 000000003330: bf9f0000
	s_code_end                                                 // 000000003334: bf9f0000
	s_code_end                                                 // 000000003338: bf9f0000
	s_code_end                                                 // 00000000333c: bf9f0000
	s_code_end                                                 // 000000003340: bf9f0000
	s_code_end                                                 // 000000003344: bf9f0000
	s_code_end                                                 // 000000003348: bf9f0000
	s_code_end                                                 // 00000000334c: bf9f0000
	s_code_end                                                 // 000000003350: bf9f0000
	s_code_end                                                 // 000000003354: bf9f0000
	s_code_end                                                 // 000000003358: bf9f0000
	s_code_end                                                 // 00000000335c: bf9f0000
	s_code_end                                                 // 000000003360: bf9f0000
	s_code_end                                                 // 000000003364: bf9f0000
	s_code_end                                                 // 000000003368: bf9f0000
	s_code_end                                                 // 00000000336c: bf9f0000
	s_code_end                                                 // 000000003370: bf9f0000
	s_code_end                                                 // 000000003374: bf9f0000
	s_code_end                                                 // 000000003378: bf9f0000
	s_code_end                                                 // 00000000337c: bf9f0000
