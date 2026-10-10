
/tmp/tmpf6eiy67e.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492>:
	s_clause 0x4                                               // 000000001b00: bf850004
	s_load_b64 s[16:17], s[0:1], 0x8                           // 000000001b04: f4002400 f8000008
	s_load_b64 s[8:9], s[0:1], 0x30                            // 000000001b0c: f4002200 f8000030
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000001b14: f4002180 f8000058
	s_load_b64 s[12:13], s[0:1], 0x80                          // 000000001b1c: f4002300 f8000080
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b24: f4004500 f80000c8
	v_lshrrev_b32_e32 v9, 3, v0                                // 000000001b2c: 32120083
	s_mov_b32 s4, ttmp7                                        // 000000001b30: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b34: 86059f73
	s_mov_b32 s2, ttmp9                                        // 000000001b38: be820075
	s_lshl_b64 s[4:5], s[4:5], 7                               // 000000001b3c: 84848704
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000001b40: bf870159
	v_dual_mov_b32 v5, s5 :: v_dual_lshlrev_b32 v2, 4, v0      // 000000001b44: ca220005 05020084
	v_or_b32_e32 v1, s4, v9                                    // 000000001b4c: 38021204
	s_load_b64 s[10:11], s[0:1], 0xd8                          // 000000001b50: f4002280 f80000d8
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b58: 86039f75
	v_dual_mov_b32 v12, s5 :: v_dual_and_b32 v23, 32, v0       // 000000001b5c: ca240005 0c1600a0
	v_or_b32_e32 v4, 0x60, v1                                  // 000000001b64: 380802ff 00000060
	s_lshl_b64 s[18:19], s[2:3], 6                             // 000000001b6c: 84928602
	v_or_b32_e32 v10, 32, v9                                   // 000000001b70: 381412a0
	v_or_b32_e32 v16, s18, v9                                  // 000000001b74: 38201212
	v_or_b32_e32 v6, 64, v1                                    // 000000001b78: 380c02c0
	v_dual_mov_b32 v7, s5 :: v_dual_and_b32 v26, 0x70, v2      // 000000001b7c: ca240005 071a04ff 00000070
	s_wait_kmcnt 0x0                                           // 000000001b88: bfc70000
	s_add_nc_u64 s[24:25], s[20:21], -1                        // 000000001b8c: a998c114
	v_mul_u32_u24_e32 v9, 0x90, v9                             // 000000001b90: 161212ff 00000090
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001b98: 7cb80818
	v_mov_b32_e32 v2, s5                                       // 000000001b9c: 7e040205
	v_cmp_gt_u64_e64 s2, s[24:25], v[6:7]                      // 000000001ba0: d45c0002 02020c18
	v_mul_u32_u24_e32 v7, 0x90, v10                            // 000000001ba8: 160e14ff 00000090
	v_or_b32_e32 v15, s18, v10                                 // 000000001bb0: 381e1412
	v_lshrrev_b32_e32 v21, 1, v0                               // 000000001bb4: 322a0081
	v_cndmask_b32_e32 v11, s24, v4, vcc_lo                     // 000000001bb8: 02160818
	v_cndmask_b32_e32 v18, s25, v12, vcc_lo                    // 000000001bbc: 02241819
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[1:2]                  // 000000001bc0: 7cb80218
	v_or_b32_e32 v4, s4, v10                                   // 000000001bc4: 38081404
	v_cndmask_b32_e64 v10, s24, v6, s2                         // 000000001bc8: d501000a 000a0c18
	v_add_nc_u32_e32 v6, v7, v26                               // 000000001bd0: 4a0c3507
	v_dual_mov_b32 v3, s5 :: v_dual_and_b32 v22, 15, v0        // 000000001bd4: ca240005 0316008f
	s_wait_alu depctr_va_vcc(0)                                // 000000001bdc: bf88ff9d
	v_dual_cndmask_b32 v1, s24, v1 :: v_dual_and_b32 v0, 47, v0// 000000001be0: ca640218 010000af
	v_cndmask_b32_e32 v7, s25, v12, vcc_lo                     // 000000001be8: 020e1819
	v_cmp_gt_u64_e64 s3, s[24:25], v[4:5]                      // 000000001bec: d45c0003 02020818
	v_cndmask_b32_e64 v19, s25, v12, s2                        // 000000001bf4: d5010013 000a1819
	s_delay_alu instid0(valu_dep_4)                            // 000000001bfc: bf870004
	v_mul_lo_u32 v12, v1, s11                                  // 000000001c00: d72c000c 02001701
	v_mad_co_u64_u32 v[1:2], null, v1, s10, s[16:17]           // 000000001c08: d6fe7c01 00401501
	v_mul_lo_u32 v7, v7, s10                                   // 000000001c10: d72c0007 02001507
	v_mul_lo_u32 v29, v10, s11                                 // 000000001c18: d72c001d 0200170a
	s_wait_alu depctr_va_sdst(0)                               // 000000001c20: bf88f19f
	v_cndmask_b32_e64 v5, s25, v5, s3                          // 000000001c24: d5010005 000e0a19
	v_cndmask_b32_e64 v4, s24, v4, s3                          // 000000001c2c: d5010004 000e0818
	v_mad_co_u64_u32 v[13:14], null, v10, s10, s[16:17]        // 000000001c34: d6fe7c0d 0040150a
	v_mul_lo_u32 v19, v19, s10                                 // 000000001c3c: d72c0013 02001513
	v_mul_lo_u32 v18, v18, s10                                 // 000000001c44: d72c0012 02001512
	v_mul_lo_u32 v28, v5, s10                                  // 000000001c4c: d72c001c 02001505
	v_mul_lo_u32 v20, v4, s11                                  // 000000001c54: d72c0014 02001704
	v_mad_co_u64_u32 v[4:5], null, v4, s10, s[16:17]           // 000000001c5c: d6fe7c04 00401504
	v_add3_u32 v2, v7, v2, v12                                 // 000000001c64: d6550002 04320507
	v_add_nc_u32_e32 v7, v9, v26                               // 000000001c6c: 4a0e3509
	v_add_co_u32 v9, vcc_lo, v1, v26                           // 000000001c70: d7006a09 02023501
	v_add3_u32 v14, v19, v14, v29                              // 000000001c78: d655000e 04761d13
	s_wait_alu depctr_va_vcc(0)                                // 000000001c80: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, 0, v2, vcc_lo               // 000000001c84: d5207c0a 01aa0480
	v_mul_lo_u32 v19, v11, s11                                 // 000000001c8c: d72c0013 0200170b
	v_mad_co_u64_u32 v[1:2], null, v11, s10, s[16:17]          // 000000001c94: d6fe7c01 0040150b
	v_add3_u32 v5, v28, v5, v20                                // 000000001c9c: d6550005 04520b1c
	v_add_co_u32 v11, vcc_lo, v4, v26                          // 000000001ca4: d7006a0b 02023504
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v17, 0x60, v21      // 000000001cac: ca240080 08102aff 00000060
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb8: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000001cbc: bf870003
	v_add_co_ci_u32_e64 v12, null, 0, v5, vcc_lo               // 000000001cc0: d5207c0c 01aa0a80
	v_add3_u32 v2, v18, v2, v19                                // 000000001cc8: d6550002 044e0512
	v_mul_lo_u32 v18, s11, v16                                 // 000000001cd0: d72c0012 0202200b
	v_mad_co_u64_u32 v[4:5], null, s10, v16, s[8:9]            // 000000001cd8: d6fe7c04 0022200a
	v_add_co_u32 v13, vcc_lo, v13, v26                         // 000000001ce0: d7006a0d 0202350d
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce8: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, 0, v14, vcc_lo              // 000000001cec: d5207c0e 01aa1c80
	v_mul_lo_u32 v28, s11, v15                                 // 000000001cf4: d72c001c 02021e0b
	v_mad_co_u64_u32 v[19:20], null, s10, v15, s[8:9]          // 000000001cfc: d6fe7c13 00221e0a
	v_add_co_u32 v15, vcc_lo, v1, v26                          // 000000001d04: d7006a0f 02023501
	s_mul_i32 s2, s10, s19                                     // 000000001d0c: 9602130a
	s_wait_alu depctr_va_vcc(0)                                // 000000001d10: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, 0, v2, vcc_lo               // 000000001d14: d5207c10 01aa0480
	v_or_b32_e32 v2, v17, v22                                  // 000000001d1c: 38042d11
	v_or_b32_e32 v25, 16, v17                                  // 000000001d20: 38322290
	v_or_b32_e32 v27, v22, v23                                 // 000000001d24: 38362f16
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d28: bf88ff9e
	v_add3_u32 v1, v18, v5, s2                                 // 000000001d2c: d6550001 000a0b12
	v_or_b32_e32 v31, s4, v17                                  // 000000001d34: 383e2204
	v_add_co_u32 v17, vcc_lo, v4, v26                          // 000000001d38: d7006a11 02023504
	v_dual_mov_b32 v5, s5 :: v_dual_and_b32 v32, 8, v21        // 000000001d40: ca240005 05202a88
	v_mul_u32_u24_e32 v2, 0x90, v2                             // 000000001d48: 160404ff 00000090
	s_wait_alu depctr_va_vcc(0)                                // 000000001d50: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, 0, v1, vcc_lo               // 000000001d54: d5207c12 01aa0280
	v_add3_u32 v1, v28, v20, s2                                // 000000001d5c: d6550001 000a291c
	v_or_b32_e32 v4, v25, v22                                  // 000000001d64: 38082d19
	v_or_b32_e32 v33, 16, v27                                  // 000000001d68: 38423690
	v_add_co_u32 v19, vcc_lo, v19, v26                         // 000000001d6c: d7006a13 02023513
	v_or_b32_e32 v21, v2, v32                                  // 000000001d74: 382a4102
	v_or_b32_e32 v2, v31, v32                                  // 000000001d78: 3804411f
	s_wait_alu depctr_va_vcc(0)                                // 000000001d7c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, 0, v1, vcc_lo               // 000000001d80: d5207c14 01aa0280
	v_mul_u32_u24_e32 v1, 0x90, v4                             // 000000001d88: 160208ff 00000090
	v_mul_u32_u24_e32 v4, 0x90, v33                            // 000000001d90: 160842ff 00000090
	v_or_b32_e32 v34, 1, v32                                   // 000000001d98: 38444081
	s_add_nc_u64 s[2:3], s[22:23], 0x7f                        // 000000001d9c: a982ff16 0000007f
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000001da4: 7ca80414
	s_wait_alu depctr_sa_sdst(0)                               // 000000001da8: bf88ff9e
	s_lshr_b64 s[16:17], s[2:3], 7                             // 000000001dac: 85908702
	v_mul_u32_u24_e32 v0, 0x90, v0                             // 000000001db0: 160000ff 00000090
	s_wait_alu depctr_sa_sdst(0)                               // 000000001db8: bf88ff9e
	s_add_nc_u64 s[2:3], s[16:17], -1                          // 000000001dbc: a982c110
	s_lshr_b64 s[8:9], s[18:19], 7                             // 000000001dc0: 85888712
	v_or_b32_e32 v46, v4, v32                                  // 000000001dc4: 385c4104
	v_or_b32_e32 v4, v34, v31                                  // 000000001dc8: 38083f22
	v_or_b32_e32 v24, s4, v25                                  // 000000001dcc: 38303204
	s_wait_alu depctr_sa_sdst(0)                               // 000000001dd0: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[8:9], s[2:3]                        // 000000001dd4: d4590004 02000408
	v_or_b32_e32 v44, v32, v0                                  // 000000001ddc: 38580120
	s_wait_alu depctr_va_vcc(0)                                // 000000001de0: bf88ff9d
	v_dual_cndmask_b32 v0, 0, v3 :: v_dual_cndmask_b32 v25, 0, v2// 000000001de4: ca520680 00180480
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000001dec: 7ca80814
	s_lshr_b64 s[14:15], s[10:11], 7                           // 000000001df0: 858e870a
	s_and_b32 s4, s4, exec_lo                                  // 000000001df4: 8b047e04
	s_cselect_b32 s9, s9, s3                                   // 000000001df8: 98090309
	s_cselect_b32 s8, s8, s2                                   // 000000001dfc: 98080208
	s_lshr_b32 s10, s11, 7                                     // 000000001e00: 850a870b
	v_or_b32_e32 v43, v1, v32                                  // 000000001e04: 38564101
	s_wait_alu depctr_va_vcc(0)                                // 000000001e08: bf88ff9d
	v_dual_mov_b32 v1, s19 :: v_dual_cndmask_b32 v28, 0, v5    // 000000001e0c: ca120013 011c0a80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e14: bf88ff9e
	v_mul_lo_u32 v29, s10, v25                                 // 000000001e18: d72c001d 0202320a
	v_mul_lo_u32 v30, s14, v0                                  // 000000001e20: d72c001e 0202000e
	v_cndmask_b32_e32 v36, 0, v4, vcc_lo                       // 000000001e28: 02480880
	v_mad_co_u64_u32 v[4:5], null, s14, v25, 0                 // 000000001e2c: d6fe7c04 0202320e
	v_or_b32_e32 v35, 2, v32                                   // 000000001e34: 38464082
	v_or_b32_e32 v39, 3, v32                                   // 000000001e38: 384e4083
	v_or_b32_e32 v0, s18, v27                                  // 000000001e3c: 38003612
	v_mul_lo_u32 v38, s14, v28                                 // 000000001e40: d72c0026 0202380e
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v55, 0             // 000000001e48: ca100080 4a360080
	v_or_b32_e32 v25, v35, v31                                 // 000000001e50: 38323f23
	v_add3_u32 v5, v5, v30, v29                                // 000000001e54: d6550005 04763d05
	v_or_b32_e32 v29, v39, v31                                 // 000000001e5c: 383a3f27
	v_mov_b32_e32 v26, s5                                      // 000000001e60: 7e340205
	v_mov_b32_e32 v30, s5                                      // 000000001e64: 7e3c0205
	v_cmp_gt_i64_e64 s2, s[22:23], v[0:1]                      // 000000001e68: d4540002 02020016
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000001e70: 3e080882
	v_mov_b32_e32 v68, 0                                       // 000000001e74: 7e880280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[25:26]                // 000000001e78: 7ca83214
	v_mov_b32_e32 v64, 0                                       // 000000001e7c: 7e800280
	v_mov_b32_e32 v60, 0                                       // 000000001e80: 7e780280
	v_mov_b32_e32 v52, 0                                       // 000000001e84: 7e680280
	v_mov_b32_e32 v48, 0                                       // 000000001e88: 7e600280
	s_mov_b64 s[24:25], 0                                      // 000000001e8c: be980180
	s_wait_alu depctr_va_vcc(0)                                // 000000001e90: bf88ff9d
	v_cndmask_b32_e32 v25, 0, v25, vcc_lo                      // 000000001e94: 02323280
	v_mul_lo_u32 v37, s10, v36                                 // 000000001e98: d72c0025 0202480a
	v_mad_co_u64_u32 v[27:28], null, s14, v36, 0               // 000000001ea0: d6fe7c1b 0202480e
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo                      // 000000001ea8: 02343480
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[29:30]                // 000000001eac: 7ca83a14
	v_mul_lo_u32 v36, s10, v25                                 // 000000001eb0: d72c0024 0202320a
	s_lshl_b64 s[26:27], s[8:9], 2                             // 000000001eb8: 849a8208
	v_mov_b32_e32 v77, 0                                       // 000000001ebc: 7e9a0280
	v_mov_b32_e32 v71, 0                                       // 000000001ec0: 7e8e0280
	v_mov_b32_e32 v49, 0                                       // 000000001ec4: 7e620280
	v_add3_u32 v28, v28, v38, v37                              // 000000001ec8: d655001c 04964d1c
	v_or_b32_e32 v37, 4, v32                                   // 000000001ed0: 384a4084
	v_mul_lo_u32 v38, s14, v26                                 // 000000001ed4: d72c0026 0202340e
	v_mad_co_u64_u32 v[25:26], null, s14, v25, 0               // 000000001edc: d6fe7c19 0202320e
	s_wait_alu depctr_va_vcc(0)                                // 000000001ee4: bf88ff9d
	v_dual_cndmask_b32 v41, 0, v29 :: v_dual_cndmask_b32 v40, 0, v30// 000000001ee8: ca523a80 29283c80
	v_or_b32_e32 v29, v37, v31                                 // 000000001ef0: 383a3f25
	v_add_co_u32 v50, vcc_lo, s6, v4                           // 000000001ef4: d7006a32 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000001efc: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s7, v5, vcc_lo              // 000000001f00: d5207c33 01aa0a07
	s_delay_alu instid0(valu_dep_3)                            // 000000001f08: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[29:30]                // 000000001f0c: 7ca83a14
	v_add3_u32 v26, v26, v38, v36                              // 000000001f10: d655001a 04924d1a
	v_or_b32_e32 v38, 5, v32                                   // 000000001f18: 384c4085
	v_lshlrev_b64_e32 v[4:5], 2, v[27:28]                      // 000000001f1c: 3e083682
	v_mul_lo_u32 v36, s10, v41                                 // 000000001f20: d72c0024 0202520a
	v_mad_co_u64_u32 v[27:28], null, s14, v41, 0               // 000000001f28: d6fe7c1b 0202520e
	s_wait_alu depctr_va_vcc(0)                                // 000000001f30: bf88ff9d
	v_cndmask_b32_e32 v41, 0, v30, vcc_lo                      // 000000001f34: 02523c80
	v_mul_lo_u32 v40, s14, v40                                 // 000000001f38: d72c0028 0202500e
	v_cndmask_b32_e32 v42, 0, v29, vcc_lo                      // 000000001f40: 02543a80
	v_or_b32_e32 v29, v38, v31                                 // 000000001f44: 383a3f26
	v_add_co_u32 v53, vcc_lo, s6, v4                           // 000000001f48: d7006a35 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000001f50: bf88ff9d
	v_add_co_ci_u32_e64 v54, null, s7, v5, vcc_lo              // 000000001f54: d5207c36 01aa0a07
	s_delay_alu instid0(valu_dep_3)                            // 000000001f5c: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[29:30]                // 000000001f60: 7ca83a14
	v_add3_u32 v28, v28, v40, v36                              // 000000001f64: d655001c 0492511c
	v_or_b32_e32 v40, 6, v32                                   // 000000001f6c: 38504086
	v_lshlrev_b64_e32 v[4:5], 2, v[25:26]                      // 000000001f70: 3e083282
	v_mul_lo_u32 v36, s10, v42                                 // 000000001f74: d72c0024 0202540a
	v_mul_lo_u32 v41, s14, v41                                 // 000000001f7c: d72c0029 0202520e
	v_mad_co_u64_u32 v[25:26], null, s14, v42, 0               // 000000001f84: d6fe7c19 0202540e
	s_wait_alu depctr_va_vcc(0)                                // 000000001f8c: bf88ff9d
	v_cndmask_b32_e32 v45, 0, v29, vcc_lo                      // 000000001f90: 025a3a80
	v_or_b32_e32 v29, v40, v31                                 // 000000001f94: 383a3f28
	v_cndmask_b32_e32 v42, 0, v30, vcc_lo                      // 000000001f98: 02543c80
	v_add_co_u32 v56, vcc_lo, s6, v4                           // 000000001f9c: d7006a38 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa4: bf88ff9d
	v_add_co_ci_u32_e64 v57, null, s7, v5, vcc_lo              // 000000001fa8: d5207c39 01aa0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[27:28]                      // 000000001fb0: 3e083682
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[29:30]                // 000000001fb4: 7ca83a14
	v_add3_u32 v26, v26, v41, v36                              // 000000001fb8: d655001a 0492531a
	v_mul_lo_u32 v36, s10, v45                                 // 000000001fc0: d72c0024 02025a0a
	v_mad_co_u64_u32 v[27:28], null, s14, v45, 0               // 000000001fc8: d6fe7c1b 02025a0e
	v_mov_b32_e32 v47, 0                                       // 000000001fd0: 7e5e0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001fd4: bf88ff9d
	v_dual_cndmask_b32 v29, 0, v29 :: v_dual_cndmask_b32 v30, 0, v30// 000000001fd8: ca523a80 1d1e3c80
	v_add_co_u32 v58, vcc_lo, s6, v4                           // 000000001fe0: d7006a3a 02020806
	s_wait_alu depctr_va_vcc(0)                                // 000000001fe8: bf88ff9d
	v_add_co_ci_u32_e64 v59, null, s7, v5, vcc_lo              // 000000001fec: d5207c3b 01aa0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[25:26]                      // 000000001ff4: 3e083282
	v_mov_b32_e32 v26, s5                                      // 000000001ff8: 7e340205
	v_mul_lo_u32 v45, s14, v30                                 // 000000001ffc: d72c002d 02023c0e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000002004: bf870223
	v_add_co_u32 v61, s3, s6, v4                               // 000000002008: d700033d 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000002010: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s7, v5, s3                  // 000000002014: d5207c3e 000e0a07
	v_mov_b32_e32 v5, s5                                       // 00000000201c: 7e0a0205
	v_mul_lo_u32 v41, s14, v42                                 // 000000002020: d72c0029 0202540e
	v_or_b32_e32 v42, 7, v32                                   // 000000002028: 38544087
	v_or_b32_e32 v4, v24, v32                                  // 00000000202c: 38084118
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_4)// 000000002030: bf870242
	v_or_b32_e32 v25, v42, v31                                 // 000000002034: 38323f2a
	v_mul_lo_u32 v31, s10, v29                                 // 000000002038: d72c001f 02023a0a
	v_mad_co_u64_u32 v[29:30], null, s14, v29, 0               // 000000002040: d6fe7c1d 02023a0e
	v_add3_u32 v28, v28, v41, v36                              // 000000002048: d655001c 0492531c
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[25:26]                // 000000002050: 7ca83214
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_4)// 000000002054: bf870212
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]                    // 000000002058: 3e363682
	v_add3_u32 v30, v30, v45, v31                              // 00000000205c: d655001e 047e5b1e
	s_wait_alu depctr_va_vcc(0)                                // 000000002064: bf88ff9d
	v_dual_cndmask_b32 v31, 0, v26 :: v_dual_cndmask_b32 v36, 0, v25// 000000002068: ca523480 1f243280
	v_mov_b32_e32 v45, 0                                       // 000000002070: 7e5a0280
	s_delay_alu instid0(valu_dep_4)                            // 000000002074: bf870004
	v_add_co_u32 v63, vcc_lo, s6, v27                          // 000000002078: d7006a3f 02023606
	s_wait_alu depctr_va_vcc(0)                                // 000000002080: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s7, v28, vcc_lo             // 000000002084: d5207c41 01aa3807
	v_mul_lo_u32 v32, s10, v36                                 // 00000000208c: d72c0020 0202480a
	v_mul_lo_u32 v31, s14, v31                                 // 000000002094: d72c001f 02023e0e
	v_mad_co_u64_u32 v[27:28], null, s14, v36, 0               // 00000000209c: d6fe7c1b 0202480e
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 0000000020a4: 7ca80814
	v_lshlrev_b64_e32 v[25:26], 2, v[29:30]                    // 0000000020a8: 3e323a82
	v_or_b32_e32 v29, v24, v34                                 // 0000000020ac: 383a4518
	v_mov_b32_e32 v30, s5                                      // 0000000020b0: 7e3c0205
	s_wait_alu depctr_va_vcc(0)                                // 0000000020b4: bf88ff9d
	v_dual_cndmask_b32 v36, 0, v5 :: v_dual_cndmask_b32 v41, 0, v4// 0000000020b8: ca520a80 24280880
	s_delay_alu instid0(valu_dep_4)                            // 0000000020c0: bf870004
	v_add_co_u32 v66, vcc_lo, s6, v25                          // 0000000020c4: d7006a42 02023206
	v_add3_u32 v28, v28, v31, v32                              // 0000000020cc: d655001c 04823f1c
	s_wait_alu depctr_va_vcc(0)                                // 0000000020d4: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s7, v26, vcc_lo             // 0000000020d8: d5207c43 01aa3407
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[29:30]                // 0000000020e0: 7ca83a14
	v_mul_lo_u32 v34, s10, v41                                 // 0000000020e4: d72c0022 0202520a
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]                    // 0000000020ec: 3e363682
	v_mad_co_u64_u32 v[25:26], null, s14, v41, 0               // 0000000020f0: d6fe7c19 0202520e
	v_or_b32_e32 v32, v24, v35                                 // 0000000020f8: 38404718
	v_mov_b32_e32 v31, s19                                     // 0000000020fc: 7e3e0213
	s_wait_alu depctr_va_vcc(0)                                // 000000002100: bf88ff9d
	v_cndmask_b32_e32 v29, 0, v29, vcc_lo                      // 000000002104: 023a3a80
	v_cndmask_b32_e32 v41, 0, v30, vcc_lo                      // 000000002108: 02523c80
	v_or_b32_e32 v30, s18, v33                                 // 00000000210c: 383c4212
	v_mov_b32_e32 v33, s5                                      // 000000002110: 7e420205
	v_mul_lo_u32 v36, s14, v36                                 // 000000002114: d72c0024 0202480e
	v_add_co_u32 v69, vcc_lo, s6, v27                          // 00000000211c: d7006a45 02023606
	s_wait_alu depctr_va_vcc(0)                                // 000000002124: bf88ff9d
	v_add_co_ci_u32_e64 v70, null, s7, v28, vcc_lo             // 000000002128: d5207c46 01aa3807
	v_cmp_gt_i64_e64 s3, s[20:21], v[32:33]                    // 000000002130: d4540003 02024014
	v_mov_b32_e32 v28, s5                                      // 000000002138: 7e380205
	v_or_b32_e32 v27, v24, v39                                 // 00000000213c: 38364f18
	v_add3_u32 v26, v26, v36, v34                              // 000000002140: d655001a 048a491a
	v_mul_lo_u32 v36, s10, v29                                 // 000000002148: d72c0024 02023a0a
	v_mul_lo_u32 v41, s14, v41                                 // 000000002150: d72c0029 0202520e
	v_mad_co_u64_u32 v[34:35], null, s14, v29, 0               // 000000002158: d6fe7c22 02023a0e
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[30:31]                // 000000002160: 7ca83c16
	s_wait_alu depctr_va_sdst(0)                               // 000000002164: bf88f19f
	v_cndmask_b32_e64 v29, 0, v33, s3                          // 000000002168: d501001d 000e4280
	v_cndmask_b32_e64 v30, 0, v32, s3                          // 000000002170: d501001e 000e4080
	v_cmp_gt_i64_e64 s3, s[20:21], v[27:28]                    // 000000002178: d4540003 02023614
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]                    // 000000002180: 3e323282
	v_mov_b32_e32 v39, 0                                       // 000000002184: 7e4e0280
	v_mul_lo_u32 v32, s14, v29                                 // 000000002188: d72c0020 02023a0e
	v_add3_u32 v35, v35, v41, v36                              // 000000002190: d6550023 04925323
	v_mul_lo_u32 v31, s10, v30                                 // 000000002198: d72c001f 02023c0a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a0: bf88f19f
	v_cndmask_b32_e64 v36, 0, v27, s3                          // 0000000021a4: d5010024 000e3680
	v_or_b32_e32 v27, v24, v37                                 // 0000000021ac: 38364b18
	v_mad_co_u64_u32 v[29:30], null, s14, v30, 0               // 0000000021b0: d6fe7c1d 02023c0e
	v_add_co_u32 v72, s4, s6, v25                              // 0000000021b8: d7000448 02023206
	v_cndmask_b32_e64 v33, 0, v28, s3                          // 0000000021c0: d5010021 000e3880
	s_delay_alu instid0(valu_dep_4)                            // 0000000021c8: bf870004
	v_cmp_gt_i64_e64 s3, s[20:21], v[27:28]                    // 0000000021cc: d4540003 02023614
	s_wait_alu depctr_va_sdst(0)                               // 0000000021d4: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s7, v26, s4                 // 0000000021d8: d5207c49 00123407
	v_lshlrev_b64_e32 v[25:26], 2, v[34:35]                    // 0000000021e0: 3e324482
	v_add3_u32 v30, v30, v32, v31                              // 0000000021e4: d655001e 047e411e
	v_mul_lo_u32 v34, s10, v36                                 // 0000000021ec: d72c0022 0202480a
	v_mul_lo_u32 v33, s14, v33                                 // 0000000021f4: d72c0021 0202420e
	v_mad_co_u64_u32 v[31:32], null, s14, v36, 0               // 0000000021fc: d6fe7c1f 0202480e
	v_cndmask_b32_e64 v36, 0, v27, s3                          // 000000002204: d5010024 000e3680
	v_or_b32_e32 v27, v24, v38                                 // 00000000220c: 38364d18
	v_add_co_u32 v75, s4, s6, v25                              // 000000002210: d700044b 02023206
	s_wait_alu depctr_va_sdst(0)                               // 000000002218: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s7, v26, s4                 // 00000000221c: d5207c4c 00123407
	v_lshlrev_b64_e32 v[25:26], 2, v[29:30]                    // 000000002224: 3e323a82
	v_cndmask_b32_e64 v35, 0, v28, s3                          // 000000002228: d5010023 000e3880
	v_cmp_gt_i64_e64 s3, s[20:21], v[27:28]                    // 000000002230: d4540003 02023614
	v_add3_u32 v32, v32, v33, v34                              // 000000002238: d6550020 048a4320
	v_mov_b32_e32 v34, s5                                      // 000000002240: 7e440205
	v_or_b32_e32 v33, v24, v40                                 // 000000002244: 38425118
	v_add_co_u32 v78, s4, s6, v25                              // 000000002248: d700044e 02023206
	v_mul_lo_u32 v37, s10, v36                                 // 000000002250: d72c0025 0202480a
	v_mul_lo_u32 v35, s14, v35                                 // 000000002258: d72c0023 0202460e
	v_mad_co_u64_u32 v[29:30], null, s14, v36, 0               // 000000002260: d6fe7c1d 0202480e
	s_wait_alu depctr_va_sdst(0)                               // 000000002268: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s7, v26, s4                 // 00000000226c: d5207c4f 00123407
	v_lshlrev_b64_e32 v[25:26], 2, v[31:32]                    // 000000002274: 3e323e82
	v_cndmask_b32_e64 v32, 0, v27, s3                          // 000000002278: d5010020 000e3680
	v_or_b32_e32 v27, v24, v42                                 // 000000002280: 38365518
	v_cmp_gt_i64_e64 s4, s[20:21], v[33:34]                    // 000000002284: d4540004 02024214
	v_cndmask_b32_e64 v31, 0, v28, s3                          // 00000000228c: d501001f 000e3880
	v_add3_u32 v30, v30, v35, v37                              // 000000002294: d655001e 0496471e
	v_mul_lo_u32 v35, s10, v32                                 // 00000000229c: d72c0023 0202400a
	v_cmp_gt_i64_e64 s3, s[20:21], v[27:28]                    // 0000000022a4: d4540003 02023614
	v_mov_b32_e32 v42, 0                                       // 0000000022ac: 7e540280
	v_mul_lo_u32 v36, s14, v31                                 // 0000000022b0: d72c0024 02023e0e
	v_mad_co_u64_u32 v[31:32], null, s14, v32, 0               // 0000000022b8: d6fe7c1f 0202400e
	s_wait_alu depctr_va_sdst(0)                               // 0000000022c0: bf88f19f
	v_cndmask_b32_e64 v24, 0, v34, s4                          // 0000000022c4: d5010018 00124480
	v_cndmask_b32_e64 v33, 0, v33, s4                          // 0000000022cc: d5010021 00124280
	v_cndmask_b32_e64 v28, 0, v28, s3                          // 0000000022d4: d501001c 000e3880
	v_cndmask_b32_e64 v27, 0, v27, s3                          // 0000000022dc: d501001b 000e3680
	v_add_co_u32 v80, s3, s6, v25                              // 0000000022e4: d7000350 02023206
	s_delay_alu instid0(valu_dep_4)                            // 0000000022ec: bf870004
	v_mul_lo_u32 v37, s10, v33                                 // 0000000022f0: d72c0025 0202420a
	v_mul_lo_u32 v38, s14, v24                                 // 0000000022f8: d72c0026 0202300e
	v_mad_co_u64_u32 v[33:34], null, s14, v33, 0               // 000000002300: d6fe7c21 0202420e
	s_wait_alu depctr_va_sdst(0)                               // 000000002308: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s7, v26, s3                 // 00000000230c: d5207c51 000e3407
	v_add3_u32 v32, v32, v36, v35                              // 000000002314: d6550020 048e4920
	v_lshlrev_b64_e32 v[24:25], 2, v[29:30]                    // 00000000231c: 3e303a82
	v_mul_lo_u32 v30, s10, v27                                 // 000000002320: d72c001e 0202360a
	v_mul_lo_u32 v35, s14, v28                                 // 000000002328: d72c0023 0202380e
	v_mad_co_u64_u32 v[26:27], null, s14, v27, 0               // 000000002330: d6fe7c1a 0202360e
	v_add3_u32 v34, v34, v38, v37                              // 000000002338: d6550022 04964d22
	v_lshlrev_b64_e32 v[28:29], 2, v[31:32]                    // 000000002340: 3e383e82
	v_add_co_u32 v82, s3, s6, v24                              // 000000002344: d7000352 02023006
	s_wait_alu depctr_va_sdst(0)                               // 00000000234c: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s7, v25, s3                 // 000000002350: d5207c53 000e3207
	v_lshlrev_b64_e32 v[24:25], 2, v[33:34]                    // 000000002358: 3e304282
	v_add3_u32 v27, v27, v35, v30                              // 00000000235c: d655001b 047a471b
	v_add_co_u32 v84, s3, s6, v28                              // 000000002364: d7000354 02023806
	s_wait_alu depctr_va_sdst(0)                               // 00000000236c: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s7, v29, s3                 // 000000002370: d5207c55 000e3a07
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000002378: bf870253
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]                    // 00000000237c: 3e343482
	v_add_co_u32 v86, s3, s6, v24                              // 000000002380: d7000356 02023006
	s_wait_alu depctr_va_sdst(0)                               // 000000002388: bf88f19f
	v_add_co_ci_u32_e64 v87, null, s7, v25, s3                 // 00000000238c: d5207c57 000e3207
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v33, 0             // 000000002394: ca100080 26200080
	v_add_co_u32 v88, s3, s6, v26                              // 00000000239c: d7000358 02023406
	s_wait_alu depctr_va_sdst(0)                               // 0000000023a4: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s7, v27, s3                 // 0000000023a8: d5207c59 000e3607
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0             // 0000000023b0: ca100080 25240080
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0             // 0000000023b8: ca100080 23220080
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v41, 0             // 0000000023c0: ca100080 20280080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v27, 0             // 0000000023c8: ca100080 281a0080
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0             // 0000000023d0: ca100080 1f1e0080
	v_mov_b32_e32 v25, 0                                       // 0000000023d8: 7e320280
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0             // 0000000023dc: ca100080 1d1c0080
	v_mov_b32_e32 v26, 0                                       // 0000000023e4: 7e340280
	v_mov_b32_e32 v24, 0                                       // 0000000023e8: 7e300280
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023ec: bf88ff9e
	s_lshl_b64 s[8:9], s[24:25], 7                             // 0000000023f0: 84888718
	v_add_nc_u32_e32 v120, 0x4800, v44                         // 0000000023f4: 4af058ff 00004800
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023fc: bf88ff9e
	v_add_co_u32 v90, s3, v9, s8                               // 000000002400: d700035a 02001109
	v_add_co_u32 v94, s4, v11, s8                              // 000000002408: d700045e 0200110b
	v_add_co_u32 v98, s5, v13, s8                              // 000000002410: d7000562 0200110d
	v_add_co_u32 v102, s6, v15, s8                             // 000000002418: d7000666 0200110f
	v_add_co_u32 v106, s7, v17, s8                             // 000000002420: d700076a 02001111
	v_add_co_u32 v110, s8, v19, s8                             // 000000002428: d700086e 02001113
	s_wait_alu depctr_va_sdst(0)                               // 000000002430: bf88f19f
	v_add_co_ci_u32_e64 v91, null, s9, v10, s3                 // 000000002434: d5207c5b 000e1409
	v_add_co_ci_u32_e64 v95, null, s9, v12, s4                 // 00000000243c: d5207c5f 00121809
	v_add_co_ci_u32_e64 v99, null, s9, v14, s5                 // 000000002444: d5207c63 00161c09
	v_add_co_ci_u32_e64 v103, null, s9, v16, s6                // 00000000244c: d5207c67 001a2009
	v_add_co_ci_u32_e64 v107, null, s9, v18, s7                // 000000002454: d5207c6b 001e2409
	v_add_co_ci_u32_e64 v111, null, s9, v20, s8                // 00000000245c: d5207c6f 00222809
	s_clause 0x3                                               // 000000002464: bf850003
	global_load_b128 v[90:93], v[90:91], off                   // 000000002468: ee05c07c 0000005a 0000005a
	global_load_b128 v[94:97], v[94:95], off                   // 000000002474: ee05c07c 0000005e 0000005e
	global_load_b128 v[98:101], v[98:99], off                  // 000000002480: ee05c07c 00000062 00000062
	global_load_b128 v[102:105], v[102:103], off               // 00000000248c: ee05c07c 00000066 00000066
	s_clause 0x1                                               // 000000002498: bf850001
	global_load_b128 v[106:109], v[106:107], off               // 00000000249c: ee05c07c 0000006a 0000006a
	global_load_b128 v[110:113], v[110:111], off               // 0000000024a8: ee05c07c 0000006e 0000006e
	v_add_nc_u32_e32 v121, 0x4800, v46                         // 0000000024b4: 4af25cff 00004800
	s_barrier_signal -1                                        // 0000000024bc: be804ec1
	s_barrier_wait 0xffff                                      // 0000000024c0: bf94ffff
	s_lshl_b64 s[28:29], s[24:25], 2                           // 0000000024c4: 849c8218
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024c8: bf88ff9e
	v_add_co_u32 v178, s3, v50, s28                            // 0000000024cc: d70003b2 02003932
	v_add_co_u32 v180, s4, v53, s28                            // 0000000024d4: d70004b4 02003935
	v_add_co_u32 v184, s6, v58, s28                            // 0000000024dc: d70006b8 0200393a
	v_add_co_u32 v186, s7, v61, s28                            // 0000000024e4: d70007ba 0200393d
	s_wait_alu depctr_va_sdst(0)                               // 0000000024ec: bf88f19f
	v_add_co_ci_u32_e64 v179, null, s29, v51, s3               // 0000000024f0: d5207cb3 000e661d
	v_add_co_ci_u32_e64 v181, null, s29, v54, s4               // 0000000024f8: d5207cb5 00126c1d
	v_add_co_ci_u32_e64 v185, null, s29, v59, s6               // 000000002500: d5207cb9 001a761d
	v_add_co_ci_u32_e64 v187, null, s29, v62, s7               // 000000002508: d5207cbb 001e7c1d
	v_add_co_u32 v182, s5, v56, s28                            // 000000002510: d70005b6 02003938
	s_wait_alu depctr_va_sdst(0)                               // 000000002518: bf88f19f
	v_add_co_ci_u32_e64 v183, null, s29, v57, s5               // 00000000251c: d5207cb7 0016721d
	s_wait_loadcnt 0x5                                         // 000000002524: bfc00005
	ds_store_b128 v7, v[90:93]                                 // 000000002528: db7c0000 00005a07
	s_wait_loadcnt 0x4                                         // 000000002530: bfc00004
	ds_store_b128 v7, v[94:97] offset:4608                     // 000000002534: db7c1200 00005e07
	s_wait_loadcnt 0x3                                         // 00000000253c: bfc00003
	ds_store_b128 v7, v[98:101] offset:9216                    // 000000002540: db7c2400 00006207
	s_wait_loadcnt 0x2                                         // 000000002548: bfc00002
	ds_store_b128 v7, v[102:105] offset:13824                  // 00000000254c: db7c3600 00006607
	s_wait_loadcnt 0x1                                         // 000000002554: bfc00001
	ds_store_b128 v7, v[106:109] offset:18432                  // 000000002558: db7c4800 00006a07
	s_wait_loadcnt 0x0                                         // 000000002560: bfc00000
	ds_store_b128 v6, v[110:113] offset:18432                  // 000000002564: db7c4800 00006e06
	s_wait_dscnt 0x0                                           // 00000000256c: bfc60000
	s_barrier_signal -1                                        // 000000002570: be804ec1
	s_barrier_wait 0xffff                                      // 000000002574: bf94ffff
	ds_load_2addr_b64 v[112:115], v21 offset1:2                // 000000002578: d9dc0200 70000015
	ds_load_2addr_b64 v[116:119], v120 offset1:2               // 000000002580: d9dc0200 74000078
	ds_load_2addr_b64 v[122:125], v121 offset1:2               // 000000002588: d9dc0200 7a000079
	ds_load_2addr_b64 v[126:129], v43 offset1:2                // 000000002590: d9dc0200 7e00002b
	ds_load_2addr_b64 v[130:133], v21 offset0:4 offset1:6      // 000000002598: d9dc0604 82000015
	ds_load_2addr_b64 v[134:137], v21 offset0:8 offset1:10     // 0000000025a0: d9dc0a08 86000015
	ds_load_2addr_b64 v[138:141], v120 offset0:4 offset1:6     // 0000000025a8: d9dc0604 8a000078
	ds_load_2addr_b64 v[142:145], v120 offset0:8 offset1:10    // 0000000025b0: d9dc0a08 8e000078
	ds_load_2addr_b64 v[146:149], v121 offset0:4 offset1:6     // 0000000025b8: d9dc0604 92000079
	ds_load_2addr_b64 v[150:153], v121 offset0:8 offset1:10    // 0000000025c0: d9dc0a08 96000079
	ds_load_2addr_b64 v[154:157], v43 offset0:4 offset1:6      // 0000000025c8: d9dc0604 9a00002b
	ds_load_2addr_b64 v[158:161], v43 offset0:8 offset1:10     // 0000000025d0: d9dc0a08 9e00002b
	ds_load_2addr_b64 v[162:165], v21 offset0:12 offset1:14    // 0000000025d8: d9dc0e0c a2000015
	ds_load_2addr_b64 v[166:169], v43 offset0:12 offset1:14    // 0000000025e0: d9dc0e0c a600002b
	ds_load_2addr_b64 v[170:173], v120 offset0:12 offset1:14   // 0000000025e8: d9dc0e0c aa000078
	ds_load_2addr_b64 v[174:177], v121 offset0:12 offset1:14   // 0000000025f0: d9dc0e0c ae000079
	global_load_b32 v188, v[178:179], off                      // 0000000025f8: ee05007c 000000bc 000000b2
	v_add_co_u32 v178, s3, v75, s28                            // 000000002604: d70003b2 0200394b
	global_load_b32 v189, v[180:181], off                      // 00000000260c: ee05007c 000000bd 000000b4
	v_add_co_u32 v180, s4, v78, s28                            // 000000002618: d70004b4 0200394e
	s_wait_dscnt 0xe                                           // 000000002620: bfc6000e
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[112:113], v[116:117], 0// 000000002624: cc46405a 1a02e970
	s_wait_dscnt 0xd                                           // 00000000262c: bfc6000d
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[112:113], v[122:123], 0// 000000002630: cc464062 1a02f570
	s_wait_dscnt 0xc                                           // 000000002638: bfc6000c
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[126:127], v[116:117], 0// 00000000263c: cc46406a 1a02e97e
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[114:115], v[118:119], v[90:97]// 000000002644: cc46405a 1d6aed72
	s_delay_alu instid0(valu_dep_3)                            // 00000000264c: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[114:115], v[124:125], v[98:105]// 000000002650: cc464062 1d8af972
	global_load_b32 v184, v[184:185], off                      // 000000002658: ee05007c 000000b8 000000b8
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[128:129], v[118:119], v[106:113]// 000000002664: cc46406a 1daaed80
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[126:127], v[122:123], 0// 00000000266c: cc464072 1a02f57e
	v_add_co_u32 v122, s8, v63, s28                            // 000000002674: d700087a 0200393f
	s_wait_alu depctr_va_sdst(0)                               // 00000000267c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s29, v65, s8               // 000000002680: d5207c7b 0022821d
	s_delay_alu instid0(valu_dep_3)                            // 000000002688: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[128:129], v[124:125], v[114:121]// 00000000268c: cc464072 1dcaf980
	v_add_co_u32 v124, s9, v66, s28                            // 000000002694: d700097c 02003942
	v_add_co_u32 v126, s10, v69, s28                           // 00000000269c: d7000a7e 02003945
	v_add_co_u32 v128, s11, v72, s28                           // 0000000026a4: d7000b80 02003948
	s_clause 0x1                                               // 0000000026ac: bf850001
	global_load_b32 v185, v[186:187], off                      // 0000000026b0: ee05007c 000000b9 000000ba
	global_load_b32 v186, v[122:123], off                      // 0000000026bc: ee05007c 000000ba 0000007a
	v_add_co_u32 v122, s6, v82, s28                            // 0000000026c8: d700067a 02003952
	s_wait_alu depctr_va_sdst(0)                               // 0000000026d0: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s29, v67, s9               // 0000000026d4: d5207c7d 0026861d
	v_add_co_ci_u32_e64 v127, null, s29, v70, s10              // 0000000026dc: d5207c7f 002a8c1d
	v_add_co_ci_u32_e64 v129, null, s29, v73, s11              // 0000000026e4: d5207c81 002e921d
	v_add_co_ci_u32_e64 v179, null, s29, v76, s3               // 0000000026ec: d5207cb3 000e981d
	v_add_co_ci_u32_e64 v181, null, s29, v79, s4               // 0000000026f4: d5207cb5 00129e1d
	v_add_co_ci_u32_e64 v123, null, s29, v83, s6               // 0000000026fc: d5207c7b 001aa61d
	global_load_b32 v190, v[182:183], off                      // 000000002704: ee05007c 000000be 000000b6
	v_add_co_u32 v182, s5, v80, s28                            // 000000002710: d70005b6 02003950
	s_clause 0x2                                               // 000000002718: bf850002
	global_load_b32 v187, v[124:125], off                      // 00000000271c: ee05007c 000000bb 0000007c
	global_load_b32 v191, v[126:127], off                      // 000000002728: ee05007c 000000bf 0000007e
	global_load_b32 v128, v[128:129], off                      // 000000002734: ee05007c 00000080 00000080
	v_add_co_u32 v124, s7, v84, s28                            // 000000002740: d700077c 02003954
	s_clause 0x1                                               // 000000002748: bf850001
	global_load_b32 v129, v[178:179], off                      // 00000000274c: ee05007c 00000081 000000b2
	global_load_b32 v178, v[180:181], off                      // 000000002758: ee05007c 000000b2 000000b4
	v_add_co_u32 v126, s3, v86, s28                            // 000000002764: d700037e 02003956
	global_load_b32 v180, v[122:123], off                      // 00000000276c: ee05007c 000000b4 0000007a
	v_add_co_u32 v122, s4, v88, s28                            // 000000002778: d700047a 02003958
	s_wait_alu depctr_va_sdst(0)                               // 000000002780: bf88f19f
	v_add_co_ci_u32_e64 v183, null, s29, v81, s5               // 000000002784: d5207cb7 0016a21d
	v_add_co_ci_u32_e64 v125, null, s29, v85, s7               // 00000000278c: d5207c7d 001eaa1d
	v_add_co_ci_u32_e64 v127, null, s29, v87, s3               // 000000002794: d5207c7f 000eae1d
	v_add_co_ci_u32_e64 v123, null, s29, v89, s4               // 00000000279c: d5207c7b 0012b21d
	s_clause 0x3                                               // 0000000027a4: bf850003
	global_load_b32 v179, v[182:183], off                      // 0000000027a8: ee05007c 000000b3 000000b6
	global_load_b32 v124, v[124:125], off                      // 0000000027b4: ee05007c 0000007c 0000007c
	global_load_b32 v125, v[126:127], off                      // 0000000027c0: ee05007c 0000007d 0000007e
	global_load_b32 v122, v[122:123], off                      // 0000000027cc: ee05007c 0000007a 0000007a
	s_mul_u64 s[4:5], s[24:25], s[16:17]                       // 0000000027d8: aa841018
	s_wait_dscnt 0x9                                           // 0000000027dc: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[130:131], v[138:139], v[90:97]// 0000000027e0: cc46405a 1d6b1582
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e8: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 0000000027ec: 84848204
	s_wait_dscnt 0x7                                           // 0000000027f0: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[130:131], v[146:147], v[98:105]// 0000000027f4: cc464062 1d8b2582
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027fc: bf88ff9e
	s_add_nc_u64 s[4:5], s[12:13], s[4:5]                      // 000000002800: a984040c
	s_wait_dscnt 0x5                                           // 000000002804: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[154:155], v[138:139], v[106:113]// 000000002808: cc46406a 1dab159a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002810: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[26:27]                      // 000000002814: a9861a04
	s_clause 0x1                                               // 000000002818: bf850001
	s_load_b32 s4, s[4:5], 0x0                                 // 00000000281c: f4000102 f8000000
	s_load_b32 s3, s[6:7], 0x0                                 // 000000002824: f40000c3 f8000000
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[154:155], v[146:147], v[114:121]// 00000000282c: cc464072 1dcb259a
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[132:133], v[140:141], v[90:97]// 000000002834: cc46405a 1d6b1984
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[132:133], v[148:149], v[98:105]// 00000000283c: cc464062 1d8b2984
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[156:157], v[140:141], v[106:113]// 000000002844: cc46406a 1dab199c
	s_add_nc_u64 s[24:25], s[24:25], 1                         // 00000000284c: a9988118
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[156:157], v[148:149], v[114:121]// 000000002850: cc464072 1dcb299c
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[134:135], v[142:143], v[90:97]// 000000002858: cc46405a 1d6b1d86
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[134:135], v[150:151], v[98:105]// 000000002860: cc464062 1d8b2d86
	s_wait_dscnt 0x4                                           // 000000002868: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[158:159], v[142:143], v[106:113]// 00000000286c: cc46406a 1dab1d9e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002874: bf88ff9e
	s_cmp_lg_u64 s[24:25], s[14:15]                            // 000000002878: bf110e18
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[158:159], v[150:151], v[114:121]// 00000000287c: cc464072 1dcb2d9e
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[136:137], v[144:145], v[90:97]// 000000002884: cc46405a 1d6b2188
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[136:137], v[152:153], v[98:105]// 00000000288c: cc464062 1d8b3188
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[160:161], v[144:145], v[106:113]// 000000002894: cc46406a 1dab21a0
	s_delay_alu instid0(valu_dep_4)                            // 00000000289c: bf870004
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[160:161], v[152:153], v[114:121]// 0000000028a0: cc464072 1dcb31a0
	s_wait_dscnt 0x1                                           // 0000000028a8: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[162:163], v[170:171], v[90:97]// 0000000028ac: cc46405a 1d6b55a2
	s_wait_dscnt 0x0                                           // 0000000028b4: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[162:163], v[174:175], v[98:105]// 0000000028b8: cc464062 1d8b5da2
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[166:167], v[170:171], v[106:113]// 0000000028c0: cc46406a 1dab55a6
	s_wait_kmcnt 0x0                                           // 0000000028c8: bfc70000
	v_mov_b32_e32 v123, s3                                     // 0000000028cc: 7ef60203
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[166:167], v[174:175], v[114:121]// 0000000028d0: cc464072 1dcb5da6
	v_wmma_f32_16x16x16_fp8_fp8 v[90:97], v[164:165], v[172:173], v[90:97]// 0000000028d8: cc46405a 1d6b59a4
	v_wmma_f32_16x16x16_fp8_fp8 v[98:105], v[164:165], v[176:177], v[98:105]// 0000000028e0: cc464062 1d8b61a4
	v_wmma_f32_16x16x16_fp8_fp8 v[106:113], v[168:169], v[172:173], v[106:113]// 0000000028e8: cc46406a 1dab59a8
	v_cndmask_b32_e64 v126, s4, v123, s2                       // 0000000028f0: d501007e 000af604
	v_cndmask_b32_e32 v123, s4, v123, vcc_lo                   // 0000000028f8: 02f6f604
	v_wmma_f32_16x16x16_fp8_fp8 v[114:121], v[168:169], v[176:177], v[114:121]// 0000000028fc: cc464072 1dcb61a8
	s_wait_loadcnt 0xf                                         // 000000002904: bfc0000f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000002908: bf8701c3
	v_mul_f32_e32 v127, v188, v126                             // 00000000290c: 10fefdbc
	s_wait_loadcnt 0xe                                         // 000000002910: bfc0000e
	v_dual_mul_f32 v137, v188, v123 :: v_dual_mul_f32 v130, v126, v189// 000000002914: c8c6f7bc 89837b7e
	v_mul_f32_e32 v138, v123, v189                             // 00000000291c: 11157b7b
	v_mul_f32_e32 v90, v90, v127                               // 000000002920: 10b4ff5a
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002924: bf870193
	v_dual_mul_f32 v98, v98, v137 :: v_dual_mul_f32 v91, v91, v130// 000000002928: c8c71362 625b055b
	v_mul_f32_e32 v99, v99, v138                               // 000000002930: 10c71563
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002934: bf870193
	v_add_f32_e32 v8, v8, v90                                  // 000000002938: 0610b508
	v_add_f32_e32 v39, v39, v98                                // 00000000293c: 064ec527
	s_wait_loadcnt 0xd                                         // 000000002940: bfc0000d
	v_dual_add_f32 v77, v77, v91 :: v_dual_mul_f32 v132, v126, v184// 000000002944: c906b74d 4d85717e
	v_mul_f32_e32 v140, v123, v184                             // 00000000294c: 1119717b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002950: bf870112
	v_dual_add_f32 v38, v38, v99 :: v_dual_mul_f32 v93, v93, v132// 000000002954: c906c726 265d095d
	v_mul_f32_e32 v101, v101, v140                             // 00000000295c: 10cb1965
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002960: bf870112
	v_add_f32_e32 v71, v71, v93                                // 000000002964: 068ebb47
	v_add_f32_e32 v36, v36, v101                               // 000000002968: 0648cb24
	s_wait_loadcnt 0xb                                         // 00000000296c: bfc0000b
	v_dual_mul_f32 v133, v126, v185 :: v_dual_mul_f32 v134, v126, v186// 000000002970: c8c7737e 8587757e
	v_dual_mul_f32 v141, v123, v185 :: v_dual_mul_f32 v142, v123, v186// 000000002978: c8c7737b 8d8f757b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002980: bf870112
	v_dual_mul_f32 v94, v94, v133 :: v_dual_mul_f32 v95, v95, v134// 000000002984: c8c70b5e 5e5f0d5f
	v_dual_mul_f32 v102, v102, v141 :: v_dual_mul_f32 v103, v103, v142// 00000000298c: c8c71b66 66671d67
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002994: bf870112
	v_add_f32_e32 v68, v68, v94                                // 000000002998: 0688bd44
	v_dual_add_f32 v64, v64, v95 :: v_dual_add_f32 v35, v35, v102// 00000000299c: c908bf40 4022cd23
	s_delay_alu instid0(valu_dep_3)                            // 0000000029a4: bf870003
	v_add_f32_e32 v34, v34, v103                               // 0000000029a8: 0644cf22
	s_wait_loadcnt 0xa                                         // 0000000029ac: bfc0000a
	v_mul_f32_e32 v131, v126, v190                             // 0000000029b0: 11077d7e
	v_mul_f32_e32 v139, v123, v190                             // 0000000029b4: 11177d7b
	s_wait_loadcnt 0x9                                         // 0000000029b8: bfc00009
	v_mul_f32_e32 v135, v126, v187                             // 0000000029bc: 110f777e
	s_wait_loadcnt 0x8                                         // 0000000029c0: bfc00008
	v_mul_f32_e32 v136, v126, v191                             // 0000000029c4: 11117f7e
	v_mul_f32_e32 v143, v123, v187                             // 0000000029c8: 111f777b
	s_wait_loadcnt 0x7                                         // 0000000029cc: bfc00007
	v_dual_mul_f32 v144, v123, v191 :: v_dual_mul_f32 v145, v126, v128// 0000000029d0: c8c77f7b 9091017e
	s_wait_loadcnt 0x5                                         // 0000000029d8: bfc00005
	v_dual_mul_f32 v146, v126, v129 :: v_dual_mul_f32 v147, v126, v178// 0000000029dc: c8c7037e 9293657e
	v_dual_mul_f32 v128, v123, v128 :: v_dual_mul_f32 v129, v123, v129// 0000000029e4: c8c7017b 8081037b
	s_wait_loadcnt 0x4                                         // 0000000029ec: bfc00004
	v_dual_mul_f32 v149, v126, v180 :: v_dual_mul_f32 v152, v123, v178// 0000000029f0: c8c7697e 9599657b
	v_mul_f32_e32 v154, v123, v180                             // 0000000029f8: 1135697b
	v_mul_f32_e32 v92, v92, v131                               // 0000000029fc: 10b9075c
	v_dual_mul_f32 v96, v96, v135 :: v_dual_mul_f32 v97, v97, v136// 000000002a00: c8c70f60 60611161
	v_mul_f32_e32 v100, v100, v139                             // 000000002a08: 10c91764
	v_dual_mul_f32 v104, v104, v143 :: v_dual_mul_f32 v105, v105, v144// 000000002a0c: c8c71f68 68692169
	v_dual_mul_f32 v106, v106, v145 :: v_dual_mul_f32 v107, v107, v146// 000000002a14: c8c7236a 6a6b256b
	s_wait_loadcnt 0x3                                         // 000000002a1c: bfc00003
	v_mul_f32_e32 v148, v126, v179                             // 000000002a20: 1129677e
	s_wait_loadcnt 0x1                                         // 000000002a24: bfc00001
	v_dual_mul_f32 v150, v126, v124 :: v_dual_mul_f32 v151, v126, v125// 000000002a28: c8c6f97e 9696fb7e
	s_wait_loadcnt 0x0                                         // 000000002a30: bfc00000
	v_dual_mul_f32 v126, v126, v122 :: v_dual_mul_f32 v153, v123, v179// 000000002a34: c8c6f57e 7e99677b
	v_dual_mul_f32 v124, v123, v124 :: v_dual_mul_f32 v125, v123, v125// 000000002a3c: c8c6f97b 7c7cfb7b
	v_mul_f32_e32 v122, v123, v122                             // 000000002a44: 10f4f57b
	v_dual_mul_f32 v108, v108, v147 :: v_dual_mul_f32 v109, v109, v148// 000000002a48: c8c7276c 6c6d296d
	v_dual_mul_f32 v110, v110, v149 :: v_dual_mul_f32 v111, v111, v150// 000000002a50: c8c72b6e 6e6f2d6f
	v_dual_mul_f32 v112, v112, v151 :: v_dual_mul_f32 v113, v113, v126// 000000002a58: c8c72f70 7070fd71
	v_dual_mul_f32 v114, v114, v128 :: v_dual_mul_f32 v115, v115, v129// 000000002a60: c8c70172 72730373
	v_dual_mul_f32 v116, v116, v152 :: v_dual_mul_f32 v117, v117, v153// 000000002a68: c8c73174 74753375
	v_dual_mul_f32 v118, v118, v154 :: v_dual_mul_f32 v119, v119, v124// 000000002a70: c8c73576 7676f977
	v_dual_mul_f32 v120, v120, v125 :: v_dual_mul_f32 v121, v121, v122// 000000002a78: c8c6fb78 7878f579
	v_add_f32_e32 v74, v74, v92                                // 000000002a80: 0694b94a
	v_dual_add_f32 v60, v60, v96 :: v_dual_add_f32 v55, v55, v97// 000000002a84: c908c13c 3c36c337
	v_add_f32_e32 v37, v37, v100                               // 000000002a8c: 064ac925
	v_dual_add_f32 v33, v33, v104 :: v_dual_add_f32 v32, v32, v105// 000000002a90: c908d121 2120d320
	v_dual_add_f32 v52, v52, v106 :: v_dual_add_f32 v49, v49, v107// 000000002a98: c908d534 3430d731
	v_dual_add_f32 v48, v48, v108 :: v_dual_add_f32 v47, v47, v109// 000000002aa0: c908d930 302edb2f
	v_dual_add_f32 v45, v45, v110 :: v_dual_add_f32 v42, v42, v111// 000000002aa8: c908dd2d 2d2adf2a
	v_dual_add_f32 v41, v41, v112 :: v_dual_add_f32 v40, v40, v113// 000000002ab0: c908e129 2928e328
	v_dual_add_f32 v31, v31, v114 :: v_dual_add_f32 v30, v30, v115// 000000002ab8: c908e51f 1f1ee71e
	v_dual_add_f32 v29, v29, v116 :: v_dual_add_f32 v28, v28, v117// 000000002ac0: c908e91d 1d1ceb1c
	v_dual_add_f32 v27, v27, v118 :: v_dual_add_f32 v26, v26, v119// 000000002ac8: c908ed1b 1b1aef1a
	v_dual_add_f32 v25, v25, v120 :: v_dual_add_f32 v24, v24, v121// 000000002ad0: c908f119 1918f318
	s_cbranch_scc1 65092                                       // 000000002ad8: bfa2fe44 <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x8ec>
	s_load_b64 s[24:25], s[0:1], 0xa8                          // 000000002adc: f4002600 f80000a8
	v_mul_lo_u32 v9, s23, v2                                   // 000000002ae4: d72c0009 02020417
	v_mul_lo_u32 v10, s22, v3                                  // 000000002aec: d72c000a 02020616
	v_mad_co_u64_u32 v[6:7], null, s22, v2, 0                  // 000000002af4: d6fe7c06 02020416
	v_sub_co_u32 v20, s0, s20, v2                              // 000000002afc: d7010014 02020414
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002b04: bf870191
	v_sub_co_ci_u32_e64 v21, null, s21, v3, s0                 // 000000002b08: d5217c15 00020615
	v_add3_u32 v7, v7, v10, v9                                 // 000000002b10: d6550007 04261507
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002b18: bf870112
	v_cmp_lt_i64_e64 s15, 0, v[20:21]                          // 000000002b1c: d451000f 02022880
	v_lshlrev_b64_e32 v[2:3], 1, v[6:7]                        // 000000002b24: 3e040c81
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000002b28: 3e0c0081
	s_and_b32 s0, s15, s2                                      // 000000002b2c: 8b00020f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b30: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002b34: be812000
	s_cbranch_execz 28                                         // 000000002b38: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x10ac>
	v_bfe_u32 v9, v8, 16, 1                                    // 000000002b3c: d6100009 02052108
	s_wait_kmcnt 0x0                                           // 000000002b44: bfc70000
	v_add_co_u32 v10, s0, s24, v2                              // 000000002b48: d700000a 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002b50: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s25, v3, s0                 // 000000002b54: d5207c0b 00020619
	v_add3_u32 v12, v9, v8, 0x7fff                             // 000000002b5c: d655000c 03fe1109 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002b68: bf870003
	v_add_co_u32 v9, s0, v10, v6                               // 000000002b6c: d7000009 02020d0a
	v_or_b32_e32 v13, 0x400000, v8                             // 000000002b74: 381a10ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002b7c: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v11, v7, s0                 // 000000002b80: d5207c0a 00020f0b
	v_cmp_u_f32_e64 s0, v8, v8                                 // 000000002b88: d4180000 02021108
	s_wait_alu depctr_va_sdst(0)                               // 000000002b90: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002b94: bf870001
	v_cndmask_b32_e64 v8, v12, v13, s0                         // 000000002b98: d5010008 00021b0c
	global_store_d16_hi_b16 v[9:10], v8, off                   // 000000002ba0: ee09407c 04000000 00000009
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002bb0: 8c7e017e
	v_add_co_u32 v8, s0, s22, v0                               // 000000002bb4: d7000008 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000002bbc: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s23, v1, s0                  // 000000002bc0: d5207c09 00020217
	v_cmp_lt_i64_e64 s16, 1, v[20:21]                          // 000000002bc8: d4510010 02022881
	s_delay_alu instid0(valu_dep_2)                            // 000000002bd0: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000002bd4: 3e101081
	s_and_b32 s0, s16, s2                                      // 000000002bd8: 8b000210
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bdc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002be0: be812000
	s_cbranch_execz 28                                         // 000000002be4: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1158>
	v_bfe_u32 v10, v77, 16, 1                                  // 000000002be8: d610000a 0205214d
	s_wait_kmcnt 0x0                                           // 000000002bf0: bfc70000
	v_add_co_u32 v11, s0, s24, v2                              // 000000002bf4: d700000b 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002bfc: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s25, v3, s0                 // 000000002c00: d5207c0c 00020619
	v_add3_u32 v13, v10, v77, 0x7fff                           // 000000002c08: d655000d 03fe9b0a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c14: bf870003
	v_add_co_u32 v10, s0, v11, v8                              // 000000002c18: d700000a 0202110b
	v_or_b32_e32 v14, 0x400000, v77                            // 000000002c20: 381c9aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002c28: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s0                 // 000000002c2c: d5207c0b 0002130c
	v_cmp_u_f32_e64 s0, v77, v77                               // 000000002c34: d4180000 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000002c3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002c40: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 000000002c44: d501000c 00021d0d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 000000002c4c: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002c5c: 8c7e017e
	s_lshl_b64 s[38:39], s[22:23], 1                           // 000000002c60: 84a68116
	v_cmp_lt_i64_e64 s14, 2, v[20:21]                          // 000000002c64: d451000e 02022882
	v_add_co_u32 v10, s0, s38, v0                              // 000000002c6c: d700000a 02020026
	s_wait_alu depctr_va_sdst(0)                               // 000000002c74: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s39, v1, s0                 // 000000002c78: d5207c0b 00020227
	s_and_b32 s0, s14, s2                                      // 000000002c80: 8b00020e
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000002c84: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c88: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002c8c: be812000
	s_cbranch_execz 28                                         // 000000002c90: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1204>
	v_bfe_u32 v12, v74, 16, 1                                  // 000000002c94: d610000c 0205214a
	s_wait_kmcnt 0x0                                           // 000000002c9c: bfc70000
	v_add_co_u32 v13, s0, s24, v2                              // 000000002ca0: d700000d 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca8: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s25, v3, s0                 // 000000002cac: d5207c0e 00020619
	v_add3_u32 v15, v12, v74, 0x7fff                           // 000000002cb4: d655000f 03fe950c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002cc0: bf870003
	v_add_co_u32 v12, s0, v13, v10                             // 000000002cc4: d700000c 0202150d
	v_or_b32_e32 v16, 0x400000, v74                            // 000000002ccc: 382094ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002cd4: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v11, s0                // 000000002cd8: d5207c0d 0002170e
	v_cmp_u_f32_e64 s0, v74, v74                               // 000000002ce0: d4180000 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ce8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002cec: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 000000002cf0: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 000000002cf8: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002d08: 8c7e017e
	s_mul_u64 s[36:37], s[22:23], 3                            // 000000002d0c: aaa48316
	v_cmp_lt_i64_e64 s13, 3, v[20:21]                          // 000000002d10: d451000d 02022883
	v_add_co_u32 v12, s0, s36, v0                              // 000000002d18: d700000c 02020024
	s_wait_alu depctr_va_sdst(0)                               // 000000002d20: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s37, v1, s0                 // 000000002d24: d5207c0d 00020225
	s_and_b32 s0, s13, s2                                      // 000000002d2c: 8b00020d
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000002d30: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d34: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002d38: be812000
	s_cbranch_execz 28                                         // 000000002d3c: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x12b0>
	v_bfe_u32 v14, v71, 16, 1                                  // 000000002d40: d610000e 02052147
	s_wait_kmcnt 0x0                                           // 000000002d48: bfc70000
	v_add_co_u32 v15, s0, s24, v2                              // 000000002d4c: d700000f 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002d54: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s25, v3, s0                 // 000000002d58: d5207c10 00020619
	v_add3_u32 v17, v14, v71, 0x7fff                           // 000000002d60: d6550011 03fe8f0e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d6c: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000002d70: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v71                            // 000000002d78: 38248eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d80: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000002d84: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v71, v71                               // 000000002d8c: d4180000 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000002d94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d98: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 000000002d9c: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000002da4: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002db0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002db4: 8c7e017e
	s_lshl_b64 s[34:35], s[22:23], 2                           // 000000002db8: 84a28216
	v_cmp_lt_i64_e64 s12, 4, v[20:21]                          // 000000002dbc: d451000c 02022884
	v_add_co_u32 v14, s0, s34, v0                              // 000000002dc4: d700000e 02020022
	s_wait_alu depctr_va_sdst(0)                               // 000000002dcc: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s35, v1, s0                 // 000000002dd0: d5207c0f 00020223
	s_and_b32 s0, s12, s2                                      // 000000002dd8: 8b00020c
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000002ddc: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000002de0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002de4: be812000
	s_cbranch_execz 28                                         // 000000002de8: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x135c>
	v_bfe_u32 v16, v68, 16, 1                                  // 000000002dec: d6100010 02052144
	s_wait_kmcnt 0x0                                           // 000000002df4: bfc70000
	v_add_co_u32 v17, s0, s24, v2                              // 000000002df8: d7000011 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002e00: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s25, v3, s0                 // 000000002e04: d5207c12 00020619
	v_add3_u32 v19, v16, v68, 0x7fff                           // 000000002e0c: d6550013 03fe8910 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e18: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000002e1c: d7000010 02021d11
	v_or_b32_e32 v43, 0x400000, v68                            // 000000002e24: 385688ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e2c: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000002e30: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v68, v68                               // 000000002e38: d4180000 02028944
	s_wait_alu depctr_va_sdst(0)                               // 000000002e40: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e44: bf870001
	v_cndmask_b32_e64 v18, v19, v43, s0                        // 000000002e48: d5010012 00025713
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000002e50: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e5c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002e60: 8c7e017e
	s_mul_u64 s[30:31], s[22:23], 5                            // 000000002e64: aa9e8516
	v_cmp_lt_i64_e64 s11, 5, v[20:21]                          // 000000002e68: d451000b 02022885
	v_add_co_u32 v16, s0, s30, v0                              // 000000002e70: d7000010 0202001e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e78: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s31, v1, s0                 // 000000002e7c: d5207c11 0002021f
	s_and_b32 s0, s11, s2                                      // 000000002e84: 8b00020b
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000002e88: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e8c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002e90: be812000
	s_cbranch_execz 28                                         // 000000002e94: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1408>
	v_bfe_u32 v18, v64, 16, 1                                  // 000000002e98: d6100012 02052140
	s_wait_kmcnt 0x0                                           // 000000002ea0: bfc70000
	v_add_co_u32 v19, s0, s24, v2                              // 000000002ea4: d7000013 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002eac: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s25, v3, s0                 // 000000002eb0: d5207c2b 00020619
	v_add3_u32 v44, v18, v64, 0x7fff                           // 000000002eb8: d655002c 03fe8112 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002ec4: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 000000002ec8: d7000012 02022113
	v_or_b32_e32 v46, 0x400000, v64                            // 000000002ed0: 385c80ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed8: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v43, v17, s0                // 000000002edc: d5207c13 0002232b
	v_cmp_u_f32_e64 s0, v64, v64                               // 000000002ee4: d4180000 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000002eec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ef0: bf870001
	v_cndmask_b32_e64 v43, v44, v46, s0                        // 000000002ef4: d501002b 00025d2c
	global_store_d16_hi_b16 v[18:19], v43, off                 // 000000002efc: ee09407c 15800000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002f0c: 8c7e017e
	s_mul_u64 s[28:29], s[22:23], 6                            // 000000002f10: aa9c8616
	v_cmp_lt_i64_e64 s9, 6, v[20:21]                           // 000000002f14: d4510009 02022886
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f1c: bf88ff9e
	v_add_co_u32 v18, s0, s28, v0                              // 000000002f20: d7000012 0202001c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f28: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s29, v1, s0                 // 000000002f2c: d5207c13 0002021d
	s_and_b32 s0, s9, s2                                       // 000000002f34: 8b000209
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000002f38: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f3c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f40: be812000
	s_cbranch_execz 28                                         // 000000002f44: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x14b8>
	v_bfe_u32 v43, v60, 16, 1                                  // 000000002f48: d610002b 0205213c
	s_wait_kmcnt 0x0                                           // 000000002f50: bfc70000
	v_add_co_u32 v44, s0, s24, v2                              // 000000002f54: d700002c 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000002f5c: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s25, v3, s0                 // 000000002f60: d5207c2e 00020619
	v_add3_u32 v50, v43, v60, 0x7fff                           // 000000002f68: d6550032 03fe792b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f74: bf870003
	v_add_co_u32 v43, s0, v44, v18                             // 000000002f78: d700002b 0202252c
	v_or_b32_e32 v51, 0x400000, v60                            // 000000002f80: 386678ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f88: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v46, v19, s0                // 000000002f8c: d5207c2c 0002272e
	v_cmp_u_f32_e64 s0, v60, v60                               // 000000002f94: d4180000 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fa0: bf870001
	v_cndmask_b32_e64 v46, v50, v51, s0                        // 000000002fa4: d501002e 00026732
	global_store_d16_hi_b16 v[43:44], v46, off                 // 000000002fac: ee09407c 17000000 0000002b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002fbc: 8c7e017e
	s_mul_u64 s[26:27], s[22:23], 7                            // 000000002fc0: aa9a8716
	v_cmp_lt_i64_e64 s8, 7, v[20:21]                           // 000000002fc4: d4510008 02022887
	v_add_co_u32 v0, s0, s26, v0                               // 000000002fcc: d7000000 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd4: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s27, v1, s0                  // 000000002fd8: d5207c01 0002021b
	s_and_b32 s0, s8, s2                                       // 000000002fe0: 8b000208
	v_lshlrev_b64_e32 v[20:21], 1, v[0:1]                      // 000000002fe4: 3e280081
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002fec: be812000
	s_cbranch_execz 28                                         // 000000002ff0: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1564>
	v_bfe_u32 v0, v55, 16, 1                                   // 000000002ff4: d6100000 02052137
	s_wait_kmcnt 0x0                                           // 000000002ffc: bfc70000
	v_add_co_u32 v1, s0, s24, v2                               // 000000003000: d7000001 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003008: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s25, v3, s0                 // 00000000300c: d5207c2b 00020619
	v_add3_u32 v44, v0, v55, 0x7fff                            // 000000003014: d655002c 03fe6f00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003020: bf870003
	v_add_co_u32 v0, s0, v1, v20                               // 000000003024: d7000000 02022901
	v_or_b32_e32 v46, 0x400000, v55                            // 00000000302c: 385c6eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v43, v21, s0                 // 000000003038: d5207c01 00022b2b
	v_cmp_u_f32_e64 s0, v55, v55                               // 000000003040: d4180000 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000003048: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000304c: bf870001
	v_cndmask_b32_e64 v43, v44, v46, s0                        // 000000003050: d501002b 00025d2c
	global_store_d16_hi_b16 v[0:1], v43, off                   // 000000003058: ee09407c 15800000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003064: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003068: 8c7e017e
	v_mul_lo_u32 v43, s23, v4                                  // 00000000306c: d72c002b 02020817
	v_mul_lo_u32 v44, s22, v5                                  // 000000003074: d72c002c 02020a16
	v_mad_co_u64_u32 v[0:1], null, s22, v4, 0                  // 00000000307c: d6fe7c00 02020816
	v_sub_co_u32 v4, s0, s20, v4                               // 000000003084: d7010004 02020814
	s_wait_alu depctr_va_sdst(0)                               // 00000000308c: bf88f19f
	v_sub_co_ci_u32_e64 v5, null, s21, v5, s0                  // 000000003090: d5217c05 00020a15
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003098: bf870211
	v_cmp_lt_i64_e64 s10, 0, v[4:5]                            // 00000000309c: d451000a 02020880
	v_add3_u32 v1, v1, v44, v43                                // 0000000030a4: d6550001 04ae5901
	s_delay_alu instid0(valu_dep_1)                            // 0000000030ac: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000030b0: 3e000081
	s_and_b32 s0, s10, s2                                      // 0000000030b4: 8b00020a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030b8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030bc: be812000
	s_cbranch_execz 28                                         // 0000000030c0: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1634>
	s_wait_kmcnt 0x0                                           // 0000000030c4: bfc70000
	v_add_co_u32 v44, s0, s24, v0                              // 0000000030c8: d700002c 02020018
	v_bfe_u32 v43, v52, 16, 1                                  // 0000000030d0: d610002b 02052134
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d8: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s25, v1, s0                 // 0000000030dc: d5207c2e 00020219
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000030e4: bf870193
	v_add_co_u32 v6, s0, v44, v6                               // 0000000030e8: d7000006 02020d2c
	v_add3_u32 v43, v43, v52, 0x7fff                           // 0000000030f0: d655002b 03fe692b 00007fff
	v_or_b32_e32 v50, 0x400000, v52                            // 0000000030fc: 386468ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003104: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v46, v7, s0                  // 000000003108: d5207c07 00020f2e
	v_cmp_u_f32_e64 s0, v52, v52                               // 000000003110: d4180000 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000003118: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000311c: bf870001
	v_cndmask_b32_e64 v43, v43, v50, s0                        // 000000003120: d501002b 0002652b
	global_store_d16_hi_b16 v[6:7], v43, off                   // 000000003128: ee09407c 15800000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003134: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003138: 8c7e017e
	v_cmp_lt_i64_e64 s7, 1, v[4:5]                             // 00000000313c: d4510007 02020881
	s_and_b32 s0, s7, s2                                       // 000000003144: 8b000207
	s_wait_alu depctr_sa_sdst(0)                               // 000000003148: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000314c: be812000
	s_cbranch_execz 28                                         // 000000003150: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x16c4>
	v_bfe_u32 v6, v49, 16, 1                                   // 000000003154: d6100006 02052131
	s_wait_kmcnt 0x0                                           // 00000000315c: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 000000003160: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003168: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s25, v1, s0                 // 00000000316c: d5207c2b 00020219
	v_add3_u32 v44, v6, v49, 0x7fff                            // 000000003174: d655002c 03fe6306 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003180: bf870003
	v_add_co_u32 v6, s0, v7, v8                                // 000000003184: d7000006 02021107
	v_or_b32_e32 v46, 0x400000, v49                            // 00000000318c: 385c62ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003194: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v43, v9, s0                  // 000000003198: d5207c07 0002132b
	v_cmp_u_f32_e64 s0, v49, v49                               // 0000000031a0: d4180000 02026331
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031ac: bf870001
	v_cndmask_b32_e64 v8, v44, v46, s0                         // 0000000031b0: d5010008 00025d2c
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000031b8: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000031c8: 8c7e017e
	v_cmp_lt_i64_e64 s6, 2, v[4:5]                             // 0000000031cc: d4510006 02020882
	s_and_b32 s0, s6, s2                                       // 0000000031d4: 8b000206
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031d8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000031dc: be812000
	s_cbranch_execz 28                                         // 0000000031e0: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1754>
	v_bfe_u32 v6, v48, 16, 1                                   // 0000000031e4: d6100006 02052130
	s_wait_kmcnt 0x0                                           // 0000000031ec: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 0000000031f0: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v1, s0                  // 0000000031fc: d5207c08 00020219
	v_add3_u32 v9, v6, v48, 0x7fff                             // 000000003204: d6550009 03fe6106 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003210: bf870003
	v_add_co_u32 v6, s0, v7, v10                               // 000000003214: d7000006 02021507
	v_or_b32_e32 v43, 0x400000, v48                            // 00000000321c: 385660ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003224: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v11, s0                  // 000000003228: d5207c07 00021708
	v_cmp_u_f32_e64 s0, v48, v48                               // 000000003230: d4180000 02026130
	s_wait_alu depctr_va_sdst(0)                               // 000000003238: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000323c: bf870001
	v_cndmask_b32_e64 v8, v9, v43, s0                          // 000000003240: d5010008 00025709
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003248: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003254: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003258: 8c7e017e
	v_cmp_lt_i64_e64 s5, 3, v[4:5]                             // 00000000325c: d4510005 02020883
	s_and_b32 s0, s5, s2                                       // 000000003264: 8b000205
	s_wait_alu depctr_sa_sdst(0)                               // 000000003268: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000326c: be812000
	s_cbranch_execz 28                                         // 000000003270: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x17e4>
	v_bfe_u32 v6, v47, 16, 1                                   // 000000003274: d6100006 0205212f
	s_wait_kmcnt 0x0                                           // 00000000327c: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 000000003280: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003288: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v1, s0                  // 00000000328c: d5207c08 00020219
	v_add3_u32 v9, v6, v47, 0x7fff                             // 000000003294: d6550009 03fe5f06 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000032a0: bf870003
	v_add_co_u32 v6, s0, v7, v12                               // 0000000032a4: d7000006 02021907
	v_or_b32_e32 v10, 0x400000, v47                            // 0000000032ac: 38145eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v13, s0                  // 0000000032b8: d5207c07 00021b08
	v_cmp_u_f32_e64 s0, v47, v47                               // 0000000032c0: d4180000 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000032c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000032cc: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 0000000032d0: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000032d8: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032e8: 8c7e017e
	v_cmp_lt_i64_e64 s4, 4, v[4:5]                             // 0000000032ec: d4510004 02020884
	s_and_b32 s0, s4, s2                                       // 0000000032f4: 8b000204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032fc: be812000
	s_cbranch_execz 28                                         // 000000003300: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1874>
	v_bfe_u32 v6, v45, 16, 1                                   // 000000003304: d6100006 0205212d
	s_wait_kmcnt 0x0                                           // 00000000330c: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 000000003310: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v1, s0                  // 00000000331c: d5207c08 00020219
	v_add3_u32 v9, v6, v45, 0x7fff                             // 000000003324: d6550009 03fe5b06 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003330: bf870003
	v_add_co_u32 v6, s0, v7, v14                               // 000000003334: d7000006 02021d07
	v_or_b32_e32 v10, 0x400000, v45                            // 00000000333c: 38145aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003344: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v15, s0                  // 000000003348: d5207c07 00021f08
	v_cmp_u_f32_e64 s0, v45, v45                               // 000000003350: d4180000 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 000000003358: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000335c: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003360: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003368: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003374: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003378: 8c7e017e
	v_cmp_lt_i64_e64 s3, 5, v[4:5]                             // 00000000337c: d4510003 02020885
	s_and_b32 s0, s3, s2                                       // 000000003384: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 000000003388: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000338c: be812000
	s_cbranch_execz 28                                         // 000000003390: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1904>
	v_bfe_u32 v6, v42, 16, 1                                   // 000000003394: d6100006 0205212a
	s_wait_kmcnt 0x0                                           // 00000000339c: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 0000000033a0: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v1, s0                  // 0000000033ac: d5207c08 00020219
	v_add3_u32 v9, v6, v42, 0x7fff                             // 0000000033b4: d6550009 03fe5506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000033c0: bf870003
	v_add_co_u32 v6, s0, v7, v16                               // 0000000033c4: d7000006 02022107
	v_or_b32_e32 v10, 0x400000, v42                            // 0000000033cc: 381454ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v17, s0                  // 0000000033d8: d5207c07 00022308
	v_cmp_u_f32_e64 s0, v42, v42                               // 0000000033e0: d4180000 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033ec: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 0000000033f0: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000033f8: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003404: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003408: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[4:5]                             // 00000000340c: d4510001 02020886
	s_and_b32 s0, s1, s2                                       // 000000003414: 8b000201
	s_wait_alu depctr_sa_sdst(0)                               // 000000003418: bf88ff9e
	s_and_saveexec_b32 s17, s0                                 // 00000000341c: be912000
	s_cbranch_execz 28                                         // 000000003420: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1994>
	v_bfe_u32 v6, v41, 16, 1                                   // 000000003424: d6100006 02052129
	s_wait_kmcnt 0x0                                           // 00000000342c: bfc70000
	v_add_co_u32 v7, s0, s24, v0                               // 000000003430: d7000007 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003438: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v1, s0                  // 00000000343c: d5207c08 00020219
	v_add3_u32 v9, v6, v41, 0x7fff                             // 000000003444: d6550009 03fe5306 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003450: bf870003
	v_add_co_u32 v6, s0, v7, v18                               // 000000003454: d7000006 02022507
	v_or_b32_e32 v10, 0x400000, v41                            // 00000000345c: 381452ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003464: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v19, s0                  // 000000003468: d5207c07 00022708
	v_cmp_u_f32_e64 s0, v41, v41                               // 000000003470: d4180000 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000003478: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000347c: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003480: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003488: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000003494: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003498: 8c7e117e
	v_cmp_lt_i64_e64 s0, 7, v[4:5]                             // 00000000349c: d4510000 02020887
	s_and_b32 s2, s0, s2                                       // 0000000034a4: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034a8: bf88ff9e
	s_and_saveexec_b32 s17, s2                                 // 0000000034ac: be912002
	s_cbranch_execz 28                                         // 0000000034b0: bfa5001c <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1a24>
	v_bfe_u32 v4, v40, 16, 1                                   // 0000000034b4: d6100004 02052128
	s_wait_kmcnt 0x0                                           // 0000000034bc: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 0000000034c0: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 0000000034cc: d5207c06 000a0219
	v_add3_u32 v7, v4, v40, 0x7fff                             // 0000000034d4: d6550007 03fe5104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000034e0: bf870003
	v_add_co_u32 v4, s2, v5, v20                               // 0000000034e4: d7000204 02022905
	v_or_b32_e32 v8, 0x400000, v40                             // 0000000034ec: 381050ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000034f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s2                  // 0000000034f8: d5207c05 000a2b06
	v_cmp_u_f32_e64 s2, v40, v40                               // 000000003500: d4180002 02025128
	s_wait_alu depctr_va_sdst(0)                               // 000000003508: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000350c: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s2                           // 000000003510: d5010006 000a1107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003518: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003524: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003528: 8c7e117e
	s_and_b32 s2, s15, vcc_lo                                  // 00000000352c: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003530: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 000000003534: be8f2002
	s_cbranch_execz 40                                         // 000000003538: bfa50028 <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1adc>
	v_add_co_u32 v4, s2, v23, s18                              // 00000000353c: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003544: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003548: d5207c05 00082680
	v_bfe_u32 v6, v39, 16, 1                                   // 000000003550: d6100006 02052127
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003558: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000355c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003564: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003568: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 000000003570: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003574: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 00000000357c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003580: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003588: 3e080881
	v_add3_u32 v6, v6, v39, 0x7fff                             // 00000000358c: d6550006 03fe4f06 00007fff
	v_or_b32_e32 v9, 0x400000, v39                             // 000000003598: 38124eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000035a0: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 0000000035a4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000035ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000035b0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v39, v39                               // 0000000035b8: d4180002 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035c4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000035c8: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000035d0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000035e0: 8c7e0f7e
	s_and_b32 s2, s16, vcc_lo                                  // 0000000035e4: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e8: bf88ff9e
	s_and_saveexec_b32 s15, s2                                 // 0000000035ec: be8f2002
	s_cbranch_execz 46                                         // 0000000035f0: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1bac>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000035f4: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000035fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003600: d5207c05 00082680
	v_bfe_u32 v6, v38, 16, 1                                   // 000000003608: d6100006 02052126
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003610: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003614: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 00000000361c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003620: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v38                             // 000000003628: 38124cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003630: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 000000003634: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 00000000363c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 000000003640: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 000000003648: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 00000000364c: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003654: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003658: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003660: 3e080881
	v_add3_u32 v6, v6, v38, 0x7fff                             // 000000003664: d6550006 03fe4d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003670: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003674: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000367c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003680: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v38, v38                               // 000000003688: d4180002 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 000000003690: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003694: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003698: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000036a0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000036b0: 8c7e0f7e
	s_and_b32 s2, s14, vcc_lo                                  // 0000000036b4: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036b8: bf88ff9e
	s_and_saveexec_b32 s14, s2                                 // 0000000036bc: be8e2002
	s_cbranch_execz 46                                         // 0000000036c0: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1c7c>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000036c4: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 0000000036cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000036d0: d5207c05 00082680
	v_bfe_u32 v6, v37, 16, 1                                   // 0000000036d8: d6100006 02052125
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036e0: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000036e4: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000036ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000036f0: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v37                             // 0000000036f8: 38124aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003700: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 000000003704: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 00000000370c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 000000003710: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 000000003718: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 00000000371c: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003724: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003728: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003730: 3e080881
	v_add3_u32 v6, v6, v37, 0x7fff                             // 000000003734: d6550006 03fe4b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003740: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003744: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000374c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003750: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v37, v37                               // 000000003758: d4180002 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 000000003760: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003764: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003768: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003770: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000377c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 000000003780: 8c7e0e7e
	s_and_b32 s2, s13, vcc_lo                                  // 000000003784: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003788: bf88ff9e
	s_and_saveexec_b32 s13, s2                                 // 00000000378c: be8d2002
	s_cbranch_execz 46                                         // 000000003790: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1d4c>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003794: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 00000000379c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 0000000037a0: d5207c05 00082680
	v_bfe_u32 v6, v36, 16, 1                                   // 0000000037a8: d6100006 02052124
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037b0: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 0000000037b4: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000037bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000037c0: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v36                             // 0000000037c8: 381248ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000037d0: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 0000000037d4: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000037dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 0000000037e0: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 0000000037e8: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000037ec: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000037f8: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003800: 3e080881
	v_add3_u32 v6, v6, v36, 0x7fff                             // 000000003804: d6550006 03fe4906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003810: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003814: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000381c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003820: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v36, v36                               // 000000003828: d4180002 02024924
	s_wait_alu depctr_va_sdst(0)                               // 000000003830: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003834: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003838: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003840: ee09407c 03000000 00002004
	s_or_b32 exec_lo, exec_lo, s13                             // 00000000384c: 8c7e0d7e
	s_and_b32 s2, s12, vcc_lo                                  // 000000003850: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003854: bf88ff9e
	s_and_saveexec_b32 s12, s2                                 // 000000003858: be8c2002
	s_cbranch_execz 46                                         // 00000000385c: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1e18>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003860: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 00000000386c: d5207c05 00082680
	v_bfe_u32 v6, v35, 16, 1                                   // 000000003874: d6100006 02052123
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000387c: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003880: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003888: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000388c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v35                             // 000000003894: 381246ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000389c: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 0000000038a0: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 0000000038ac: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 0000000038b4: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 0000000038b8: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 0000000038c0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 0000000038c4: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000038cc: 3e080881
	v_add3_u32 v6, v6, v35, 0x7fff                             // 0000000038d0: d6550006 03fe4706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000038dc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000038e0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000038ec: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v35, v35                               // 0000000038f4: d4180002 02024723
	s_wait_alu depctr_va_sdst(0)                               // 0000000038fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003900: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003904: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000390c: ee09407c 03000000 00002004
	s_or_b32 exec_lo, exec_lo, s12                             // 000000003918: 8c7e0c7e
	s_and_b32 s2, s11, vcc_lo                                  // 00000000391c: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003920: bf88ff9e
	s_and_saveexec_b32 s11, s2                                 // 000000003924: be8b2002
	s_cbranch_execz 46                                         // 000000003928: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1ee4>
	v_add_co_u32 v4, s2, v23, s18                              // 00000000392c: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003934: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003938: d5207c05 00082680
	v_bfe_u32 v6, v34, 16, 1                                   // 000000003940: d6100006 02052122
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003948: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 00000000394c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003954: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003958: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v34                             // 000000003960: 381244ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003968: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 00000000396c: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000003974: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 000000003978: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 000000003980: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003984: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 00000000398c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003990: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003998: 3e080881
	v_add3_u32 v6, v6, v34, 0x7fff                             // 00000000399c: d6550006 03fe4506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039a8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000039ac: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000039b8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v34, v34                               // 0000000039c0: d4180002 02024522
	s_wait_alu depctr_va_sdst(0)                               // 0000000039c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039cc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000039d0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000039d8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 0000000039e8: 8c7e0b7e
	s_and_b32 s2, s9, vcc_lo                                   // 0000000039ec: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039f0: bf88ff9e
	s_and_saveexec_b32 s9, s2                                  // 0000000039f4: be892002
	s_cbranch_execz 46                                         // 0000000039f8: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x1fb4>
	v_add_co_u32 v4, s2, v23, s18                              // 0000000039fc: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003a04: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003a08: d5207c05 00082680
	v_bfe_u32 v6, v33, 16, 1                                   // 000000003a10: d6100006 02052121
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a18: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003a1c: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003a24: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003a28: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v33                             // 000000003a30: 381242ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a38: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000003a3c: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003a44: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 000000003a48: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000003a50: bfc70000
	v_add_co_u32 v7, s2, s24, v2                               // 000000003a54: d7000207 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003a5c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s25, v3, s2                  // 000000003a60: d5207c08 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003a68: 3e080881
	v_add3_u32 v6, v6, v33, 0x7fff                             // 000000003a6c: d6550006 03fe4306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a78: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000003a7c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003a84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000003a88: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v33, v33                               // 000000003a90: d4180002 02024321
	s_wait_alu depctr_va_sdst(0)                               // 000000003a98: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a9c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000003aa0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003aa8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ab4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000003ab8: 8c7e097e
	s_and_b32 s2, s8, vcc_lo                                   // 000000003abc: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ac0: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 000000003ac4: be882002
	s_cbranch_execz 46                                         // 000000003ac8: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x2084>
	v_add_co_u32 v4, s2, v23, s18                              // 000000003acc: d7000204 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s19, s2                   // 000000003ad8: d5207c05 00082680
	v_bfe_u32 v6, v32, 16, 1                                   // 000000003ae0: d6100006 02052120
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ae8: bf8701a3
	v_add_co_u32 v4, s2, v4, v22                               // 000000003aec: d7000204 02022d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003af4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000003af8: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003b00: 380e40ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b08: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000003b0c: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003b14: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000003b18: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 000000003b20: bfc70000
	v_add_co_u32 v2, s2, s24, v2                               // 000000003b24: d7000202 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000003b2c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s2                  // 000000003b30: d5207c03 000a0619
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003b38: 3e080881
	v_add3_u32 v6, v6, v32, 0x7fff                             // 000000003b3c: d6550006 03fe4106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b48: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000003b4c: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000003b54: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 000000003b58: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v32, v32                               // 000000003b60: d4180002 02024120
	s_wait_alu depctr_va_sdst(0)                               // 000000003b68: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b6c: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000003b70: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003b78: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b84: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003b88: 8c7e087e
	s_and_b32 s2, s10, vcc_lo                                  // 000000003b8c: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b90: bf88ff9e
	s_and_saveexec_b32 s8, s2                                  // 000000003b94: be882002
	s_cbranch_execz 40                                         // 000000003b98: bfa50028 <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x213c>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003b9c: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003ba4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003ba8: d5207c03 00082680
	v_bfe_u32 v4, v31, 16, 1                                   // 000000003bb0: d6100004 0205211f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bb8: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003bbc: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003bc4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003bc8: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 000000003bd0: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003bd4: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003bdc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003be0: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003be8: 3e040481
	v_add3_u32 v4, v4, v31, 0x7fff                             // 000000003bec: d6550004 03fe3f04 00007fff
	v_or_b32_e32 v7, 0x400000, v31                             // 000000003bf8: 380e3eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000003c00: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000003c04: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003c0c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003c10: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v31, v31                               // 000000003c18: d4180002 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000003c20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c24: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003c28: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003c30: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000003c40: 8c7e087e
	s_and_b32 s2, s7, vcc_lo                                   // 000000003c44: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c48: bf88ff9e
	s_and_saveexec_b32 s7, s2                                  // 000000003c4c: be872002
	s_cbranch_execz 46                                         // 000000003c50: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x220c>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003c54: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003c5c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003c60: d5207c03 00082680
	v_bfe_u32 v4, v30, 16, 1                                   // 000000003c68: d6100004 0205211e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c70: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003c74: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003c7c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003c80: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v30                             // 000000003c88: 380e3cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c90: bf8701a3
	v_add_co_u32 v2, s2, s22, v2                               // 000000003c94: d7000202 02020416
	s_wait_alu depctr_va_sdst(0)                               // 000000003c9c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s23, v3, s2                  // 000000003ca0: d5207c03 000a0617
	s_wait_kmcnt 0x0                                           // 000000003ca8: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003cac: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003cb8: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003cc0: 3e040481
	v_add3_u32 v4, v4, v30, 0x7fff                             // 000000003cc4: d6550004 03fe3d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cd0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003cd4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003cdc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003ce0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v30, v30                               // 000000003ce8: d4180002 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003cf4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003cf8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003d00: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003d10: 8c7e077e
	s_and_b32 s2, s6, vcc_lo                                   // 000000003d14: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d18: bf88ff9e
	s_and_saveexec_b32 s6, s2                                  // 000000003d1c: be862002
	s_cbranch_execz 46                                         // 000000003d20: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x22dc>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003d24: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003d2c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003d30: d5207c03 00082680
	v_bfe_u32 v4, v29, 16, 1                                   // 000000003d38: d6100004 0205211d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d40: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003d44: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003d4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003d50: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v29                             // 000000003d58: 380e3aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d60: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000003d64: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000003d6c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000003d70: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000003d78: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003d7c: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003d84: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003d88: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003d90: 3e040481
	v_add3_u32 v4, v4, v29, 0x7fff                             // 000000003d94: d6550004 03fe3b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003da0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003da4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003dac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003db0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v29, v29                               // 000000003db8: d4180002 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003dc4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003dc8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003dd0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ddc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000003de0: 8c7e067e
	s_and_b32 s2, s5, vcc_lo                                   // 000000003de4: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de8: bf88ff9e
	s_and_saveexec_b32 s5, s2                                  // 000000003dec: be852002
	s_cbranch_execz 46                                         // 000000003df0: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x23ac>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003df4: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003dfc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003e00: d5207c03 00082680
	v_bfe_u32 v4, v28, 16, 1                                   // 000000003e08: d6100004 0205211c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e10: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003e14: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003e1c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003e20: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v28                             // 000000003e28: 380e38ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e30: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000003e34: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000003e3c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000003e40: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000003e48: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003e4c: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003e54: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003e58: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003e60: 3e040481
	v_add3_u32 v4, v4, v28, 0x7fff                             // 000000003e64: d6550004 03fe3904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e70: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003e74: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003e7c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003e80: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v28, v28                               // 000000003e88: d4180002 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000003e90: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e94: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003e98: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003ea0: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003eb0: 8c7e057e
	s_and_b32 s2, s4, vcc_lo                                   // 000000003eb4: 8b026a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003eb8: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003ebc: be842002
	s_cbranch_execz 46                                         // 000000003ec0: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x247c>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003ec4: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003ecc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003ed0: d5207c03 00082680
	v_bfe_u32 v4, v27, 16, 1                                   // 000000003ed8: d6100004 0205211b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ee0: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003ee4: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003eec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003ef0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v27                             // 000000003ef8: 380e36ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f00: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000003f04: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000003f0c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000003f10: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000003f18: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003f1c: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003f24: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003f28: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000003f30: 3e040481
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000003f34: d6550004 03fe3704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f40: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000003f44: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000003f4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000003f50: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v27, v27                               // 000000003f58: d4180002 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000003f60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003f64: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000003f68: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000003f70: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003f80: 8c7e047e
	s_and_b32 s2, s3, vcc_lo                                   // 000000003f84: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f88: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003f8c: be832002
	s_cbranch_execz 46                                         // 000000003f90: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x254c>
	v_add_co_u32 v2, s2, v23, s18                              // 000000003f94: d7000202 02002517
	s_wait_alu depctr_va_sdst(0)                               // 000000003f9c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s2                   // 000000003fa0: d5207c03 00082680
	v_bfe_u32 v4, v26, 16, 1                                   // 000000003fa8: d6100004 0205211a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fb0: bf8701a3
	v_add_co_u32 v2, s2, v2, v22                               // 000000003fb4: d7000202 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 000000003fbc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000003fc0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v26                             // 000000003fc8: 380e34ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fd0: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000003fd4: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000003fdc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000003fe0: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000003fe8: bfc70000
	v_add_co_u32 v5, s2, s24, v0                               // 000000003fec: d7000205 02020018
	s_wait_alu depctr_va_sdst(0)                               // 000000003ff4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s2                  // 000000003ff8: d5207c06 000a0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004000: 3e040481
	v_add3_u32 v4, v4, v26, 0x7fff                             // 000000004004: d6550004 03fe3504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004010: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000004014: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 00000000401c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000004020: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v26, v26                               // 000000004028: d4180002 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000004030: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004034: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000004038: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004040: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000404c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004050: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000004054: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000004058: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000405c: be822001
	s_cbranch_execz 46                                         // 000000004060: bfa5002e <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x261c>
	v_add_co_u32 v2, s1, v23, s18                              // 000000004064: d7000102 02002517
	s_wait_alu depctr_va_sdst(0)                               // 00000000406c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s1                   // 000000004070: d5207c03 00042680
	v_bfe_u32 v4, v25, 16, 1                                   // 000000004078: d6100004 02052119
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004080: bf8701a3
	v_add_co_u32 v2, s1, v2, v22                               // 000000004084: d7000102 02022d02
	s_wait_alu depctr_va_sdst(0)                               // 00000000408c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000004090: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v25                             // 000000004098: 380e32ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040a0: bf8701a3
	v_add_co_u32 v2, s1, s28, v2                               // 0000000040a4: d7000102 0202041c
	s_wait_alu depctr_va_sdst(0)                               // 0000000040ac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s29, v3, s1                  // 0000000040b0: d5207c03 0006061d
	s_wait_kmcnt 0x0                                           // 0000000040b8: bfc70000
	v_add_co_u32 v5, s1, s24, v0                               // 0000000040bc: d7000105 02020018
	s_wait_alu depctr_va_sdst(0)                               // 0000000040c4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s25, v1, s1                  // 0000000040c8: d5207c06 00060219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000040d0: 3e040481
	v_add3_u32 v4, v4, v25, 0x7fff                             // 0000000040d4: d6550004 03fe3304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040e0: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 0000000040e4: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000040ec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 0000000040f0: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v25, v25                               // 0000000040f8: d4180001 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000004100: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004104: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000004108: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:32          // 000000004110: ee09407c 02000000 00002002
	s_wait_alu depctr_sa_sdst(0)                               // 00000000411c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004120: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000004124: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000004128: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000412c: be812000
	s_cbranch_execz 43                                         // 000000004130: bfa5002b <tessera_rocm_scaled_matmul_lds_0ac40669ce7dd492+0x26e0>
	v_add_co_u32 v2, s0, v23, s18                              // 000000004134: d7000002 02002517
	s_wait_alu depctr_va_sdst(0)                               // 00000000413c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s19, s0                   // 000000004140: d5207c03 00002680
	v_bfe_u32 v4, v24, 16, 1                                   // 000000004148: d6100004 02052118
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004150: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v22                           // 000000004154: d7006a02 02022d02
	s_wait_alu depctr_va_vcc(0)                                // 00000000415c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000004160: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v24                             // 000000004168: 380a30ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004170: bf8701a3
	v_add_co_u32 v2, vcc_lo, s26, v2                           // 000000004174: d7006a02 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 00000000417c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s27, v3, vcc_lo              // 000000004180: d5207c03 01aa061b
	s_wait_kmcnt 0x0                                           // 000000004188: bfc70000
	v_add_co_u32 v0, vcc_lo, s24, v0                           // 00000000418c: d7006a00 02020018
	s_wait_alu depctr_va_vcc(0)                                // 000000004194: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s25, v1, vcc_lo              // 000000004198: d5207c01 01aa0219
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000041a0: 3e040481
	v_add3_u32 v4, v4, v24, 0x7fff                             // 0000000041a4: d6550004 03fe3104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041b0: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 0000000041b4: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 0000000041bc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 0000000041c0: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 0000000041c8: 7c303118
	s_wait_alu depctr_va_vcc(0)                                // 0000000041cc: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 0000000041d0: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000041d4: ee09407c 01000000 00002000
	s_nop 0                                                    // 0000000041e0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000041e4: bfb60003
	s_endpgm                                                   // 0000000041e8: bfb00000
	s_code_end                                                 // 0000000041ec: bf9f0000
	s_code_end                                                 // 0000000041f0: bf9f0000
	s_code_end                                                 // 0000000041f4: bf9f0000
	s_code_end                                                 // 0000000041f8: bf9f0000
	s_code_end                                                 // 0000000041fc: bf9f0000
	s_code_end                                                 // 000000004200: bf9f0000
	s_code_end                                                 // 000000004204: bf9f0000
	s_code_end                                                 // 000000004208: bf9f0000
	s_code_end                                                 // 00000000420c: bf9f0000
	s_code_end                                                 // 000000004210: bf9f0000
	s_code_end                                                 // 000000004214: bf9f0000
	s_code_end                                                 // 000000004218: bf9f0000
	s_code_end                                                 // 00000000421c: bf9f0000
	s_code_end                                                 // 000000004220: bf9f0000
	s_code_end                                                 // 000000004224: bf9f0000
	s_code_end                                                 // 000000004228: bf9f0000
	s_code_end                                                 // 00000000422c: bf9f0000
	s_code_end                                                 // 000000004230: bf9f0000
	s_code_end                                                 // 000000004234: bf9f0000
	s_code_end                                                 // 000000004238: bf9f0000
	s_code_end                                                 // 00000000423c: bf9f0000
	s_code_end                                                 // 000000004240: bf9f0000
	s_code_end                                                 // 000000004244: bf9f0000
	s_code_end                                                 // 000000004248: bf9f0000
	s_code_end                                                 // 00000000424c: bf9f0000
	s_code_end                                                 // 000000004250: bf9f0000
	s_code_end                                                 // 000000004254: bf9f0000
	s_code_end                                                 // 000000004258: bf9f0000
	s_code_end                                                 // 00000000425c: bf9f0000
	s_code_end                                                 // 000000004260: bf9f0000
	s_code_end                                                 // 000000004264: bf9f0000
	s_code_end                                                 // 000000004268: bf9f0000
	s_code_end                                                 // 00000000426c: bf9f0000
	s_code_end                                                 // 000000004270: bf9f0000
	s_code_end                                                 // 000000004274: bf9f0000
	s_code_end                                                 // 000000004278: bf9f0000
	s_code_end                                                 // 00000000427c: bf9f0000
	s_code_end                                                 // 000000004280: bf9f0000
	s_code_end                                                 // 000000004284: bf9f0000
	s_code_end                                                 // 000000004288: bf9f0000
	s_code_end                                                 // 00000000428c: bf9f0000
	s_code_end                                                 // 000000004290: bf9f0000
	s_code_end                                                 // 000000004294: bf9f0000
	s_code_end                                                 // 000000004298: bf9f0000
	s_code_end                                                 // 00000000429c: bf9f0000
	s_code_end                                                 // 0000000042a0: bf9f0000
	s_code_end                                                 // 0000000042a4: bf9f0000
	s_code_end                                                 // 0000000042a8: bf9f0000
	s_code_end                                                 // 0000000042ac: bf9f0000
	s_code_end                                                 // 0000000042b0: bf9f0000
	s_code_end                                                 // 0000000042b4: bf9f0000
	s_code_end                                                 // 0000000042b8: bf9f0000
	s_code_end                                                 // 0000000042bc: bf9f0000
	s_code_end                                                 // 0000000042c0: bf9f0000
	s_code_end                                                 // 0000000042c4: bf9f0000
	s_code_end                                                 // 0000000042c8: bf9f0000
	s_code_end                                                 // 0000000042cc: bf9f0000
	s_code_end                                                 // 0000000042d0: bf9f0000
	s_code_end                                                 // 0000000042d4: bf9f0000
	s_code_end                                                 // 0000000042d8: bf9f0000
	s_code_end                                                 // 0000000042dc: bf9f0000
	s_code_end                                                 // 0000000042e0: bf9f0000
	s_code_end                                                 // 0000000042e4: bf9f0000
	s_code_end                                                 // 0000000042e8: bf9f0000
	s_code_end                                                 // 0000000042ec: bf9f0000
	s_code_end                                                 // 0000000042f0: bf9f0000
	s_code_end                                                 // 0000000042f4: bf9f0000
	s_code_end                                                 // 0000000042f8: bf9f0000
	s_code_end                                                 // 0000000042fc: bf9f0000
	s_code_end                                                 // 000000004300: bf9f0000
	s_code_end                                                 // 000000004304: bf9f0000
	s_code_end                                                 // 000000004308: bf9f0000
	s_code_end                                                 // 00000000430c: bf9f0000
	s_code_end                                                 // 000000004310: bf9f0000
	s_code_end                                                 // 000000004314: bf9f0000
	s_code_end                                                 // 000000004318: bf9f0000
	s_code_end                                                 // 00000000431c: bf9f0000
	s_code_end                                                 // 000000004320: bf9f0000
	s_code_end                                                 // 000000004324: bf9f0000
	s_code_end                                                 // 000000004328: bf9f0000
	s_code_end                                                 // 00000000432c: bf9f0000
	s_code_end                                                 // 000000004330: bf9f0000
	s_code_end                                                 // 000000004334: bf9f0000
	s_code_end                                                 // 000000004338: bf9f0000
	s_code_end                                                 // 00000000433c: bf9f0000
	s_code_end                                                 // 000000004340: bf9f0000
	s_code_end                                                 // 000000004344: bf9f0000
	s_code_end                                                 // 000000004348: bf9f0000
	s_code_end                                                 // 00000000434c: bf9f0000
	s_code_end                                                 // 000000004350: bf9f0000
	s_code_end                                                 // 000000004354: bf9f0000
	s_code_end                                                 // 000000004358: bf9f0000
	s_code_end                                                 // 00000000435c: bf9f0000
	s_code_end                                                 // 000000004360: bf9f0000
	s_code_end                                                 // 000000004364: bf9f0000
	s_code_end                                                 // 000000004368: bf9f0000
	s_code_end                                                 // 00000000436c: bf9f0000
	s_code_end                                                 // 000000004370: bf9f0000
	s_code_end                                                 // 000000004374: bf9f0000
	s_code_end                                                 // 000000004378: bf9f0000
	s_code_end                                                 // 00000000437c: bf9f0000
