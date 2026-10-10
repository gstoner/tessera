
/tmp/tmpkptovwuk.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_6bb337b5a3dd5b79>:
	s_clause 0x5                                               // 000000001b00: bf850005
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b04: f4002080 f80000d8
	s_load_b64 s[12:13], s[0:1], 0x8                           // 000000001b0c: f4002300 f8000008
	s_load_b64 s[14:15], s[0:1], 0x30                          // 000000001b14: f4002380 f8000030
	s_load_b64 s[6:7], s[0:1], 0x58                            // 000000001b1c: f4002180 f8000058
	s_load_b64 s[24:25], s[0:1], 0x80                          // 000000001b24: f4002600 f8000080
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b2c: f4004500 f80000c8
	v_lshrrev_b32_e32 v4, 3, v0                                // 000000001b34: 32080083
	s_mov_b32 s4, ttmp9                                        // 000000001b38: be840075
	s_mov_b32 s8, ttmp7                                        // 000000001b3c: be880073
	v_lshrrev_b32_e32 v21, 1, v0                               // 000000001b40: 322a0081
	s_ashr_i32 s5, ttmp9, 31                                   // 000000001b44: 86059f75
	s_ashr_i32 s9, ttmp7, 31                                   // 000000001b48: 86099f73
	s_lshl_b64 s[10:11], s[4:5], 6                             // 000000001b4c: 848a8604
	s_lshl_b64 s[4:5], s[8:9], 7                               // 000000001b50: 84848708
	v_or_b32_e32 v2, 32, v4                                    // 000000001b54: 380408a0
	v_lshlrev_b32_e32 v3, 4, v0                                // 000000001b58: 30060084
	v_or_b32_e32 v5, s4, v4                                    // 000000001b5c: 380a0804
	v_dual_mov_b32 v6, 0 :: v_dual_and_b32 v17, 0x60, v21      // 000000001b60: ca240080 06102aff 00000060
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001b6c: bf870214
	v_or_b32_e32 v15, s10, v2                                  // 000000001b70: 381e040a
	v_and_b32_e32 v23, 0x70, v3                                // 000000001b74: 362e06ff 00000070
	v_mul_u32_u24_e32 v7, 0x90, v2                             // 000000001b7c: 160e04ff 00000090
	v_or_b32_e32 v9, s4, v2                                    // 000000001b84: 38120404
	v_or_b32_e32 v22, 16, v17                                  // 000000001b88: 382c2290
	s_wait_kmcnt 0x0                                           // 000000001b8c: bfc70000
	v_mul_lo_u32 v10, s3, v5                                   // 000000001b90: d72c000a 02020a03
	v_mad_co_u64_u32 v[2:3], null, s2, v5, s[12:13]            // 000000001b98: d6fe7c02 00320a02
	v_or_b32_e32 v16, s10, v4                                  // 000000001ba0: 3820080a
	v_or_b32_e32 v11, 0x60, v5                                 // 000000001ba4: 38160aff 00000060
	v_or_b32_e32 v8, 64, v5                                    // 000000001bac: 38100ac0
	v_mul_u32_u24_e32 v12, 0x90, v4                            // 000000001bb0: 161808ff 00000090
	v_mul_lo_u32 v18, s3, v9                                   // 000000001bb8: d72c0012 02021203
	v_mad_co_u64_u32 v[4:5], null, s2, v9, s[12:13]            // 000000001bc0: d6fe7c04 00321202
	v_or_b32_e32 v27, s4, v17                                  // 000000001bc8: 38362204
	v_or_b32_e32 v38, s4, v22                                  // 000000001bcc: 384c2c04
	s_mul_i32 s4, s2, s5                                       // 000000001bd0: 96040502
	v_add_co_u32 v9, vcc_lo, v2, v23                           // 000000001bd4: d7006a09 02022f02
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bdc: bf88ff9e
	v_add3_u32 v3, v10, v3, s4                                 // 000000001be0: d6550003 0012070a
	v_mul_lo_u32 v19, s3, v8                                   // 000000001be8: d72c0013 02021003
	v_add3_u32 v5, v18, v5, s4                                 // 000000001bf0: d6550005 00120b12
	v_mul_lo_u32 v18, s3, v11                                  // 000000001bf8: d72c0012 02021603
	v_mad_co_u64_u32 v[13:14], null, s2, v8, s[12:13]          // 000000001c00: d6fe7c0d 00321002
	v_add_co_ci_u32_e64 v10, null, 0, v3, vcc_lo               // 000000001c08: d5207c0a 01aa0680
	v_mad_co_u64_u32 v[2:3], null, s2, v11, s[12:13]           // 000000001c10: d6fe7c02 00321602
	v_add_co_u32 v11, vcc_lo, v4, v23                          // 000000001c18: d7006a0b 02022f04
	v_add_nc_u32_e32 v8, v12, v23                              // 000000001c20: 4a102f0c
	s_wait_alu depctr_va_vcc(0)                                // 000000001c24: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, 0, v5, vcc_lo               // 000000001c28: d5207c0c 01aa0a80
	v_add3_u32 v14, v19, v14, s4                               // 000000001c30: d655000e 00121d13
	v_mul_lo_u32 v24, s3, v15                                  // 000000001c38: d72c0018 02021e03
	v_add3_u32 v5, v18, v3, s4                                 // 000000001c40: d6550005 00120712
	v_mul_lo_u32 v18, s3, v16                                  // 000000001c48: d72c0012 02022003
	v_mad_co_u64_u32 v[3:4], null, s2, v16, s[14:15]           // 000000001c50: d6fe7c03 003a2002
	v_mad_co_u64_u32 v[19:20], null, s2, v15, s[14:15]         // 000000001c58: d6fe7c13 003a1e02
	v_mov_b32_e32 v1, s5                                       // 000000001c60: 7e020205
	v_and_b32_e32 v25, 47, v0                                  // 000000001c64: 363200af
	v_and_b32_e32 v0, 15, v0                                   // 000000001c68: 3600008f
	v_add_co_u32 v13, vcc_lo, v13, v23                         // 000000001c6c: d7006a0d 02022f0d
	s_mul_i32 s4, s2, s11                                      // 000000001c74: 96040b02
	s_wait_alu depctr_va_vcc(0)                                // 000000001c78: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, 0, v14, vcc_lo              // 000000001c7c: d5207c0e 01aa1c80
	v_add_co_u32 v15, vcc_lo, v2, v23                          // 000000001c84: d7006a0f 02022f02
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c8c: bf88ff9e
	v_add3_u32 v2, v18, v4, s4                                 // 000000001c90: d6550002 00120912
	s_wait_alu depctr_va_vcc(0)                                // 000000001c98: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, 0, v5, vcc_lo               // 000000001c9c: d5207c10 01aa0a80
	v_or_b32_e32 v4, v17, v0                                   // 000000001ca4: 38080111
	v_add_co_u32 v17, vcc_lo, v3, v23                          // 000000001ca8: d7006a11 02022f03
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, 0, v2, vcc_lo               // 000000001cb4: d5207c12 01aa0480
	v_add3_u32 v2, v24, v20, s4                                // 000000001cbc: d6550002 00122918
	v_and_b32_e32 v33, 8, v21                                  // 000000001cc4: 36422a88
	v_or_b32_e32 v0, v22, v0                                   // 000000001cc8: 38000116
	v_add_co_u32 v19, vcc_lo, v19, v23                         // 000000001ccc: d7006a13 02022f13
	s_wait_alu depctr_va_vcc(0)                                // 000000001cd4: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, 0, v2, vcc_lo               // 000000001cd8: d5207c14 01aa0480
	s_delay_alu instid0(valu_dep_3)                            // 000000001ce0: bf870003
	v_mul_u32_u24_e32 v2, 0x90, v0                             // 000000001ce4: 160400ff 00000090
	v_or_b32_e32 v0, v27, v33                                  // 000000001cec: 3800431b
	v_mul_u32_u24_e32 v3, 0x90, v4                             // 000000001cf0: 160608ff 00000090
	v_or_b32_e32 v35, 1, v33                                   // 000000001cf8: 38464281
	s_add_nc_u64 s[8:9], s[22:23], 0x7f                        // 000000001cfc: a988ff16 0000007f
	v_or_b32_e32 v24, v2, v33                                  // 000000001d04: 38304302
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001d08: 7ca80014
	s_lshr_b64 s[28:29], s[8:9], 7                             // 000000001d0c: 859c8708
	s_lshr_b64 s[12:13], s[10:11], 7                           // 000000001d10: 858c870a
	s_add_nc_u64 s[8:9], s[28:29], -1                          // 000000001d14: a988c11c
	v_mov_b32_e32 v5, s5                                       // 000000001d18: 7e0a0205
	s_lshr_b64 s[26:27], s[2:3], 7                             // 000000001d1c: 859a8702
	s_wait_alu depctr_va_vcc(0)                                // 000000001d20: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v1, vcc_lo                        // 000000001d24: 02040280
	v_or_b32_e32 v34, 16, v25                                  // 000000001d28: 38443290
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d2c: bf88ff9e
	v_cmp_lt_u64_e64 s2, s[12:13], s[8:9]                      // 000000001d30: d4590002 0200100c
	v_cndmask_b32_e32 v22, 0, v0, vcc_lo                       // 000000001d38: 022c0080
	v_or_b32_e32 v36, 2, v33                                   // 000000001d3c: 38484282
	v_mul_lo_u32 v32, s26, v2                                  // 000000001d40: d72c0020 0202041a
	v_mul_u32_u24_e32 v4, 0x90, v34                            // 000000001d48: 160844ff 00000090
	v_or_b32_e32 v2, s10, v25                                  // 000000001d50: 3804320a
	s_and_b32 s2, s2, exec_lo                                  // 000000001d54: 8b027e02
	s_cselect_b32 s9, s13, s9                                  // 000000001d58: 9809090d
	s_cselect_b32 s8, s12, s8                                  // 000000001d5c: 9808080c
	v_or_b32_e32 v28, v4, v33                                  // 000000001d60: 38384304
	v_or_b32_e32 v4, v35, v27                                  // 000000001d64: 38083723
	s_lshr_b32 s12, s3, 7                                      // 000000001d68: 850c8703
	v_mov_b32_e32 v76, 0                                       // 000000001d6c: 7e980280
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d70: bf88ff9e
	v_mul_lo_u32 v31, s12, v22                                 // 000000001d74: d72c001f 02022c0c
	v_mov_b32_e32 v77, 0                                       // 000000001d7c: 7e9a0280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000001d80: 7ca80814
	v_mov_b32_e32 v73, 0                                       // 000000001d84: 7e920280
	v_mov_b32_e32 v69, 0                                       // 000000001d88: 7e8a0280
	v_mov_b32_e32 v61, 0                                       // 000000001d8c: 7e7a0280
	v_mov_b32_e32 v57, 0                                       // 000000001d90: 7e720280
	s_lshl_b64 s[30:31], s[8:9], 2                             // 000000001d94: 849e8208
	s_wait_alu depctr_va_vcc(0)                                // 000000001d98: bf88ff9d
	v_dual_cndmask_b32 v29, 0, v5 :: v_dual_cndmask_b32 v30, 0, v4// 000000001d9c: ca520a80 1d1e0880
	v_add_nc_u32_e32 v7, v7, v23                               // 000000001da4: 4a0e2f07
	v_mad_co_u64_u32 v[4:5], null, s26, v22, 0                 // 000000001da8: d6fe7c04 02022c1a
	v_or_b32_e32 v22, v36, v27                                 // 000000001db0: 382c3724
	v_mov_b32_e32 v23, s5                                      // 000000001db4: 7e2e0205
	v_mul_lo_u32 v37, s26, v29                                 // 000000001db8: d72c0025 02023a1a
	v_mov_b32_e32 v72, 0                                       // 000000001dc0: 7e900280
	v_mov_b32_e32 v68, 0                                       // 000000001dc4: 7e880280
	v_mov_b32_e32 v64, 0                                       // 000000001dc8: 7e800280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[22:23]                // 000000001dcc: 7ca82c14
	v_add3_u32 v5, v5, v32, v31                                // 000000001dd0: d6550005 047e4105
	v_mov_b32_e32 v32, s5                                      // 000000001dd8: 7e400205
	v_mov_b32_e32 v58, 0                                       // 000000001ddc: 7e740280
	v_mov_b32_e32 v62, 0                                       // 000000001de0: 7e7c0280
	s_wait_alu depctr_va_vcc(0)                                // 000000001de4: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v22, vcc_lo                      // 000000001de8: 022c2c80
	v_or_b32_e32 v39, 3, v33                                   // 000000001dec: 384e4283
	v_or_b32_e32 v21, v3, v33                                  // 000000001df0: 382a4303
	v_mul_u32_u24_e32 v3, 0x90, v25                            // 000000001df4: 160632ff 00000090
	v_mul_lo_u32 v25, s12, v30                                 // 000000001dfc: d72c0019 02023c0c
	v_mad_co_u64_u32 v[29:30], null, s26, v30, 0               // 000000001e04: d6fe7c1d 02023c1a
	v_or_b32_e32 v31, v39, v27                                 // 000000001e0c: 383e3727
	v_or_b32_e32 v42, 4, v33                                   // 000000001e10: 38544284
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000001e14: 3e080882
	v_cndmask_b32_e32 v23, 0, v23, vcc_lo                      // 000000001e18: 022e2e80
	v_or_b32_e32 v45, 5, v33                                   // 000000001e1c: 385a4285
	v_cmp_gt_i64_e64 s2, s[20:21], v[31:32]                    // 000000001e20: d4540002 02023e14
	v_or_b32_e32 v48, 6, v33                                   // 000000001e28: 38604286
	v_add3_u32 v30, v30, v37, v25                              // 000000001e2c: d655001e 04664b1e
	v_mul_lo_u32 v25, s12, v22                                 // 000000001e34: d72c0019 02022c0c
	v_mul_lo_u32 v37, s26, v23                                 // 000000001e3c: d72c0025 02022e1a
	v_mad_co_u64_u32 v[22:23], null, s26, v22, 0               // 000000001e44: d6fe7c16 02022c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000001e4c: bf88f19f
	v_cndmask_b32_e64 v44, 0, v31, s2                          // 000000001e50: d501002c 000a3e80
	v_or_b32_e32 v31, v42, v27                                 // 000000001e58: 383e372a
	v_cndmask_b32_e64 v43, 0, v32, s2                          // 000000001e5c: d501002b 000a4080
	v_add_co_u32 v40, s2, s6, v4                               // 000000001e64: d7000228 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000001e6c: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s7, v5, s2                  // 000000001e70: d5207c29 000a0a07
	v_cmp_gt_i64_e64 s2, s[20:21], v[31:32]                    // 000000001e78: d4540002 02023e14
	v_lshlrev_b64_e32 v[4:5], 2, v[29:30]                      // 000000001e80: 3e083a82
	v_add3_u32 v23, v23, v37, v25                              // 000000001e84: d6550017 04664b17
	v_mul_lo_u32 v25, s12, v44                                 // 000000001e8c: d72c0019 0202580c
	v_mul_lo_u32 v37, s26, v43                                 // 000000001e94: d72c0025 0202561a
	v_mad_co_u64_u32 v[29:30], null, s26, v44, 0               // 000000001e9c: d6fe7c1d 0202581a
	s_wait_alu depctr_va_sdst(0)                               // 000000001ea4: bf88f19f
	v_cndmask_b32_e64 v47, 0, v31, s2                          // 000000001ea8: d501002f 000a3e80
	v_or_b32_e32 v31, v45, v27                                 // 000000001eb0: 383e372d
	v_cndmask_b32_e64 v46, 0, v32, s2                          // 000000001eb4: d501002e 000a4080
	v_add_co_u32 v43, s2, s6, v4                               // 000000001ebc: d700022b 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000001ec4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s7, v5, s2                  // 000000001ec8: d5207c2c 000a0a07
	v_cmp_gt_i64_e64 s2, s[20:21], v[31:32]                    // 000000001ed0: d4540002 02023e14
	v_lshlrev_b64_e32 v[4:5], 2, v[22:23]                      // 000000001ed8: 3e082c82
	v_add3_u32 v30, v30, v37, v25                              // 000000001edc: d655001e 04664b1e
	v_mul_lo_u32 v25, s12, v47                                 // 000000001ee4: d72c0019 02025e0c
	v_mul_lo_u32 v37, s26, v46                                 // 000000001eec: d72c0025 02025c1a
	v_mad_co_u64_u32 v[22:23], null, s26, v47, 0               // 000000001ef4: d6fe7c16 02025e1a
	s_wait_alu depctr_va_sdst(0)                               // 000000001efc: bf88f19f
	v_cndmask_b32_e64 v50, 0, v31, s2                          // 000000001f00: d5010032 000a3e80
	v_or_b32_e32 v31, v48, v27                                 // 000000001f08: 383e3730
	v_cndmask_b32_e64 v49, 0, v32, s2                          // 000000001f0c: d5010031 000a4080
	v_add_co_u32 v46, s2, s6, v4                               // 000000001f14: d700022e 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000001f1c: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s7, v5, s2                  // 000000001f20: d5207c2f 000a0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[29:30]                      // 000000001f28: 3e083a82
	v_cmp_gt_i64_e64 s2, s[20:21], v[31:32]                    // 000000001f2c: d4540002 02023e14
	v_add3_u32 v23, v23, v37, v25                              // 000000001f34: d6550017 04664b17
	v_mul_lo_u32 v25, s12, v50                                 // 000000001f3c: d72c0019 0202640c
	v_mul_lo_u32 v37, s26, v49                                 // 000000001f44: d72c0025 0202621a
	v_mad_co_u64_u32 v[29:30], null, s26, v50, 0               // 000000001f4c: d6fe7c1d 0202641a
	v_or_b32_e32 v51, 7, v33                                   // 000000001f54: 38664287
	s_wait_alu depctr_va_sdst(0)                               // 000000001f58: bf88f19f
	v_cndmask_b32_e64 v32, 0, v32, s2                          // 000000001f5c: d5010020 000a4080
	v_cndmask_b32_e64 v31, 0, v31, s2                          // 000000001f64: d501001f 000a3e80
	v_add_co_u32 v49, s2, s6, v4                               // 000000001f6c: d7000231 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000001f74: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s7, v5, s2                  // 000000001f78: d5207c32 000a0a07
	v_lshlrev_b64_e32 v[4:5], 2, v[22:23]                      // 000000001f80: 3e082c82
	v_or_b32_e32 v22, v51, v27                                 // 000000001f84: 382c3733
	v_mov_b32_e32 v23, s5                                      // 000000001f88: 7e2e0205
	v_mul_lo_u32 v27, s12, v31                                 // 000000001f8c: d72c001b 02023e0c
	v_mul_lo_u32 v54, s26, v32                                 // 000000001f94: d72c0036 0202401a
	v_mad_co_u64_u32 v[31:32], null, s26, v31, 0               // 000000001f9c: d6fe7c1f 02023e1a
	v_add3_u32 v30, v30, v37, v25                              // 000000001fa4: d655001e 04664b1e
	v_cmp_gt_i64_e64 s2, s[20:21], v[22:23]                    // 000000001fac: d4540002 02022c14
	v_add_co_u32 v52, s3, s6, v4                               // 000000001fb4: d7000334 02020806
	s_wait_alu depctr_va_sdst(0)                               // 000000001fbc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s7, v5, s3                  // 000000001fc0: d5207c35 000e0a07
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]                    // 000000001fc8: 3e3a3a82
	v_add3_u32 v32, v32, v54, v27                              // 000000001fcc: d6550020 046e6d20
	v_cndmask_b32_e64 v25, 0, v23, s2                          // 000000001fd4: d5010019 000a2e80
	v_cndmask_b32_e64 v27, 0, v22, s2                          // 000000001fdc: d501001b 000a2c80
	v_mov_b32_e32 v5, s5                                       // 000000001fe4: 7e0a0205
	v_or_b32_e32 v4, v38, v33                                  // 000000001fe8: 38084326
	v_add_co_u32 v55, s2, s6, v29                              // 000000001fec: d7000237 02023a06
	v_or_b32_e32 v26, v33, v3                                  // 000000001ff4: 38340721
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff8: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s7, v30, s2                 // 000000001ffc: d5207c38 000a3c07
	v_mul_lo_u32 v33, s12, v27                                 // 000000002004: d72c0021 0202360c
	v_mul_lo_u32 v25, s26, v25                                 // 00000000200c: d72c0019 0202321a
	v_mad_co_u64_u32 v[29:30], null, s26, v27, 0               // 000000002014: d6fe7c1d 0202361a
	v_cmp_gt_i64_e64 s2, s[20:21], v[4:5]                      // 00000000201c: d4540002 02020814
	v_lshlrev_b64_e32 v[22:23], 2, v[31:32]                    // 000000002024: 3e2c3e82
	v_dual_mov_b32 v3, s11 :: v_dual_mov_b32 v32, s5           // 000000002028: ca10000b 03200005
	v_or_b32_e32 v31, v38, v35                                 // 000000002030: 383e4726
	v_mov_b32_e32 v35, s5                                      // 000000002034: 7e460205
	s_wait_alu depctr_va_sdst(0)                               // 000000002038: bf88f19f
	v_cndmask_b32_e64 v27, 0, v5, s2                           // 00000000203c: d501001b 000a0a80
	v_cndmask_b32_e64 v37, 0, v4, s2                           // 000000002044: d5010025 000a0880
	v_add_co_u32 v59, s2, s6, v22                              // 00000000204c: d700023b 02022c06
	v_add3_u32 v30, v30, v25, v33                              // 000000002054: d655001e 0486331e
	s_wait_alu depctr_va_sdst(0)                               // 00000000205c: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s7, v23, s2                 // 000000002060: d5207c3c 000a2e07
	v_cmp_gt_i64_e64 s2, s[20:21], v[31:32]                    // 000000002068: d4540002 02023e14
	v_mul_lo_u32 v25, s12, v37                                 // 000000002070: d72c0019 02024a0c
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]                    // 000000002078: 3e3a3a82
	v_mad_co_u64_u32 v[22:23], null, s26, v37, 0               // 00000000207c: d6fe7c16 02024a1a
	v_mul_lo_u32 v27, s26, v27                                 // 000000002084: d72c001b 0202361a
	v_mov_b32_e32 v33, s11                                     // 00000000208c: 7e42020b
	s_wait_alu depctr_va_sdst(0)                               // 000000002090: bf88f19f
	v_cndmask_b32_e64 v37, 0, v32, s2                          // 000000002094: d5010025 000a4080
	v_or_b32_e32 v32, s10, v34                                 // 00000000209c: 3840440a
	v_or_b32_e32 v34, v38, v36                                 // 0000000020a0: 38444926
	v_cndmask_b32_e64 v31, 0, v31, s2                          // 0000000020a4: d501001f 000a3e80
	v_add_co_u32 v63, s2, s6, v29                              // 0000000020ac: d700023f 02023a06
	s_wait_alu depctr_va_sdst(0)                               // 0000000020b4: bf88f19f
	v_add_co_ci_u32_e64 v65, null, s7, v30, s2                 // 0000000020b8: d5207c41 000a3c07
	v_cmp_gt_i64_e64 s3, s[20:21], v[34:35]                    // 0000000020c0: d4540003 02024414
	v_mov_b32_e32 v30, s5                                      // 0000000020c8: 7e3c0205
	v_or_b32_e32 v29, v38, v39                                 // 0000000020cc: 383a4f26
	v_add3_u32 v23, v23, v27, v25                              // 0000000020d0: d6550017 04663717
	v_mul_lo_u32 v25, s12, v31                                 // 0000000020d8: d72c0019 02023e0c
	v_mul_lo_u32 v27, s26, v37                                 // 0000000020e0: d72c001b 02024a1a
	v_mad_co_u64_u32 v[36:37], null, s26, v31, 0               // 0000000020e8: d6fe7c24 02023e1a
	v_cmp_gt_i64_e64 s2, s[22:23], v[32:33]                    // 0000000020f0: d4540002 02024016
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f8: bf88f19f
	v_cndmask_b32_e64 v31, 0, v35, s3                          // 0000000020fc: d501001f 000e4680
	v_cndmask_b32_e64 v32, 0, v34, s3                          // 000000002104: d5010020 000e4480
	v_cmp_gt_i64_e64 s3, s[20:21], v[29:30]                    // 00000000210c: d4540003 02023a14
	v_lshlrev_b64_e32 v[22:23], 2, v[22:23]                    // 000000002114: 3e2c2c82
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v54, 0             // 000000002118: ca100080 27360080
	v_add3_u32 v37, v37, v27, v25                              // 000000002120: d6550025 04663725
	v_mul_lo_u32 v25, s12, v32                                 // 000000002128: d72c0019 0202400c
	v_mul_lo_u32 v27, s26, v31                                 // 000000002130: d72c001b 02023e1a
	v_mad_co_u64_u32 v[31:32], null, s26, v32, 0               // 000000002138: d6fe7c1f 0202401a
	s_wait_alu depctr_va_sdst(0)                               // 000000002140: bf88f19f
	v_cndmask_b32_e64 v34, 0, v29, s3                          // 000000002144: d5010022 000e3a80
	v_or_b32_e32 v29, v38, v42                                 // 00000000214c: 383a5526
	v_cndmask_b32_e64 v33, 0, v30, s3                          // 000000002150: d5010021 000e3c80
	v_add_co_u32 v66, s4, s6, v22                              // 000000002158: d7000442 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 000000002160: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s7, v23, s4                 // 000000002164: d5207c43 00122e07
	v_cmp_gt_i64_e64 s3, s[20:21], v[29:30]                    // 00000000216c: d4540003 02023a14
	v_lshlrev_b64_e32 v[22:23], 2, v[36:37]                    // 000000002174: 3e2c4882
	v_add3_u32 v32, v32, v27, v25                              // 000000002178: d6550020 04663720
	v_mul_lo_u32 v25, s12, v34                                 // 000000002180: d72c0019 0202440c
	v_mul_lo_u32 v27, s26, v33                                 // 000000002188: d72c001b 0202421a
	v_mad_co_u64_u32 v[33:34], null, s26, v34, 0               // 000000002190: d6fe7c21 0202441a
	s_wait_alu depctr_va_sdst(0)                               // 000000002198: bf88f19f
	v_cndmask_b32_e64 v35, 0, v30, s3                          // 00000000219c: d5010023 000e3c80
	v_cndmask_b32_e64 v36, 0, v29, s3                          // 0000000021a4: d5010024 000e3a80
	v_or_b32_e32 v29, v38, v45                                 // 0000000021ac: 383a5b26
	v_add_co_u32 v70, s4, s6, v22                              // 0000000021b0: d7000446 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b8: bf88f19f
	v_add_co_ci_u32_e64 v71, null, s7, v23, s4                 // 0000000021bc: d5207c47 00122e07
	v_lshlrev_b64_e32 v[22:23], 2, v[31:32]                    // 0000000021c4: 3e2c3e82
	v_add3_u32 v34, v34, v27, v25                              // 0000000021c8: d6550022 04663722
	v_mul_lo_u32 v25, s12, v36                                 // 0000000021d0: d72c0019 0202480c
	v_mul_lo_u32 v27, s26, v35                                 // 0000000021d8: d72c001b 0202461a
	v_mad_co_u64_u32 v[31:32], null, s26, v36, 0               // 0000000021e0: d6fe7c1f 0202481a
	v_cmp_gt_i64_e64 s3, s[20:21], v[29:30]                    // 0000000021e8: d4540003 02023a14
	v_mov_b32_e32 v36, s5                                      // 0000000021f0: 7e480205
	v_or_b32_e32 v35, v38, v48                                 // 0000000021f4: 38466126
	v_add_co_u32 v74, s4, s6, v22                              // 0000000021f8: d700044a 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 000000002200: bf88f19f
	v_add_co_ci_u32_e64 v75, null, s7, v23, s4                 // 000000002204: d5207c4b 00122e07
	v_add3_u32 v32, v32, v27, v25                              // 00000000220c: d6550020 04663720
	v_cndmask_b32_e64 v27, 0, v29, s3                          // 000000002214: d501001b 000e3a80
	v_or_b32_e32 v29, v38, v51                                 // 00000000221c: 383a6726
	v_cmp_gt_i64_e64 s4, s[20:21], v[35:36]                    // 000000002220: d4540004 02024614
	v_cndmask_b32_e64 v25, 0, v30, s3                          // 000000002228: d5010019 000e3c80
	v_lshlrev_b64_e32 v[22:23], 2, v[33:34]                    // 000000002230: 3e2c4282
	v_mul_lo_u32 v37, s12, v27                                 // 000000002234: d72c0025 0202360c
	v_cmp_gt_i64_e64 s3, s[20:21], v[29:30]                    // 00000000223c: d4540003 02023a14
	v_mad_co_u64_u32 v[33:34], null, s26, v27, 0               // 000000002244: d6fe7c21 0202361a
	v_mul_lo_u32 v25, s26, v25                                 // 00000000224c: d72c0019 0202321a
	s_wait_alu depctr_va_sdst(0)                               // 000000002254: bf88f19f
	v_cndmask_b32_e64 v36, 0, v36, s4                          // 000000002258: d5010024 00124880
	v_cndmask_b32_e64 v35, 0, v35, s4                          // 000000002260: d5010023 00124680
	v_mov_b32_e32 v42, 0                                       // 000000002268: 7e540280
	v_cndmask_b32_e64 v30, 0, v30, s3                          // 00000000226c: d501001e 000e3c80
	v_cndmask_b32_e64 v29, 0, v29, s3                          // 000000002274: d501001d 000e3a80
	v_mul_lo_u32 v38, s26, v36                                 // 00000000227c: d72c0026 0202481a
	v_mul_lo_u32 v27, s12, v35                                 // 000000002284: d72c001b 0202460c
	v_mad_co_u64_u32 v[35:36], null, s26, v35, 0               // 00000000228c: d6fe7c23 0202461a
	v_add3_u32 v34, v34, v25, v37                              // 000000002294: d6550022 04963322
	v_mul_lo_u32 v25, s12, v29                                 // 00000000229c: d72c0019 02023a0c
	v_mul_lo_u32 v37, s26, v30                                 // 0000000022a4: d72c0025 02023c1a
	v_mad_co_u64_u32 v[29:30], null, s26, v29, 0               // 0000000022ac: d6fe7c1d 02023a1a
	v_add_co_u32 v78, s3, s6, v22                              // 0000000022b4: d700034e 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 0000000022bc: bf88f19f
	v_add_co_ci_u32_e64 v79, null, s7, v23, s3                 // 0000000022c0: d5207c4f 000e2e07
	v_lshlrev_b64_e32 v[22:23], 2, v[31:32]                    // 0000000022c8: 3e2c3e82
	v_add3_u32 v36, v36, v38, v27                              // 0000000022cc: d6550024 046e4d24
	v_lshlrev_b64_e32 v[31:32], 2, v[33:34]                    // 0000000022d4: 3e3e4282
	v_add3_u32 v30, v30, v37, v25                              // 0000000022d8: d655001e 04664b1e
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0             // 0000000022e0: ca100080 26240080
	v_add_co_u32 v80, s3, s6, v22                              // 0000000022e8: d7000350 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f0: bf88f19f
	v_add_co_ci_u32_e64 v81, null, s7, v23, s3                 // 0000000022f4: d5207c51 000e2e07
	v_lshlrev_b64_e32 v[22:23], 2, v[35:36]                    // 0000000022fc: 3e2c4682
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]                    // 000000002300: 3e3a3a82
	v_add_co_u32 v82, s3, s6, v31                              // 000000002304: d7000352 02023e06
	s_wait_alu depctr_va_sdst(0)                               // 00000000230c: bf88f19f
	v_add_co_ci_u32_e64 v83, null, s7, v32, s3                 // 000000002310: d5207c53 000e4007
	s_delay_alu instid0(valu_dep_4)                            // 000000002318: bf870004
	v_add_co_u32 v84, s3, s6, v22                              // 00000000231c: d7000354 02022c06
	s_wait_alu depctr_va_sdst(0)                               // 000000002324: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s7, v23, s3                 // 000000002328: d5207c55 000e2e07
	v_add_co_u32 v86, s3, s6, v29                              // 000000002330: d7000356 02023a06
	s_wait_alu depctr_va_sdst(0)                               // 000000002338: bf88f19f
	v_add_co_ci_u32_e64 v87, null, s7, v30, s3                 // 00000000233c: d5207c57 000e3c07
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0             // 000000002344: ca100080 24220080
	v_mov_b32_e32 v34, 0                                       // 00000000234c: 7e440280
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v51, 0             // 000000002350: ca100080 20320080
	v_mov_b32_e32 v30, 0                                       // 000000002358: 7e3c0280
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v45, 0             // 00000000235c: ca100080 302c0080
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v33, 0             // 000000002364: ca100080 16200080
	v_mov_b32_e32 v31, 0                                       // 00000000236c: 7e3e0280
	v_mov_b32_e32 v29, 0                                       // 000000002370: 7e3a0280
	v_mov_b32_e32 v27, 0                                       // 000000002374: 7e360280
	v_mov_b32_e32 v25, 0                                       // 000000002378: 7e320280
	v_mov_b32_e32 v23, 0                                       // 00000000237c: 7e2e0280
	s_mov_b64 s[20:21], 0                                      // 000000002380: be940180
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[2:3]                  // 000000002384: 7ca80416
	s_wait_alu depctr_sa_sdst(0)                               // 000000002388: bf88ff9e
	s_lshl_b64 s[8:9], s[20:21], 7                             // 00000000238c: 84888714
	s_lshl_b64 s[18:19], s[20:21], 2                           // 000000002390: 84928214
	s_wait_alu depctr_sa_sdst(0)                               // 000000002394: bf88ff9e
	v_add_co_u32 v88, s3, v9, s8                               // 000000002398: d7000358 02001109
	v_add_co_u32 v92, s4, v11, s8                              // 0000000023a0: d700045c 0200110b
	v_add_co_u32 v96, s5, v13, s8                              // 0000000023a8: d7000560 0200110d
	v_add_co_u32 v100, s6, v15, s8                             // 0000000023b0: d7000664 0200110f
	v_add_co_u32 v104, s7, v17, s8                             // 0000000023b8: d7000768 02001111
	v_add_co_u32 v108, s8, v19, s8                             // 0000000023c0: d700086c 02001113
	s_wait_alu depctr_va_sdst(0)                               // 0000000023c8: bf88f19f
	v_add_co_ci_u32_e64 v89, null, s9, v10, s3                 // 0000000023cc: d5207c59 000e1409
	v_add_co_ci_u32_e64 v93, null, s9, v12, s4                 // 0000000023d4: d5207c5d 00121809
	v_add_co_ci_u32_e64 v97, null, s9, v14, s5                 // 0000000023dc: d5207c61 00161c09
	v_add_co_ci_u32_e64 v101, null, s9, v16, s6                // 0000000023e4: d5207c65 001a2009
	v_add_co_ci_u32_e64 v105, null, s9, v18, s7                // 0000000023ec: d5207c69 001e2409
	v_add_co_ci_u32_e64 v109, null, s9, v20, s8                // 0000000023f4: d5207c6d 00222809
	s_clause 0x3                                               // 0000000023fc: bf850003
	global_load_b128 v[88:91], v[88:89], off                   // 000000002400: ee05c07c 00000058 00000058
	global_load_b128 v[92:95], v[92:93], off                   // 00000000240c: ee05c07c 0000005c 0000005c
	global_load_b128 v[96:99], v[96:97], off                   // 000000002418: ee05c07c 00000060 00000060
	global_load_b128 v[100:103], v[100:101], off               // 000000002424: ee05c07c 00000064 00000064
	s_clause 0x1                                               // 000000002430: bf850001
	global_load_b128 v[104:107], v[104:105], off               // 000000002434: ee05c07c 00000068 00000068
	global_load_b128 v[108:111], v[108:109], off               // 000000002440: ee05c07c 0000006c 0000006c
	v_add_co_u32 v112, s3, v40, s18                            // 00000000244c: d7000370 02002528
	v_add_co_u32 v114, s4, v43, s18                            // 000000002454: d7000472 0200252b
	v_add_co_u32 v116, s5, v46, s18                            // 00000000245c: d7000574 0200252e
	v_add_co_u32 v118, s6, v49, s18                            // 000000002464: d7000676 02002531
	v_add_co_u32 v120, s7, v52, s18                            // 00000000246c: d7000778 02002534
	v_add_co_u32 v122, s8, v55, s18                            // 000000002474: d700087a 02002537
	v_add_co_u32 v124, s9, v59, s18                            // 00000000247c: d700097c 0200253b
	v_add_co_u32 v126, s10, v63, s18                           // 000000002484: d7000a7e 0200253f
	v_add_co_u32 v128, s11, v66, s18                           // 00000000248c: d7000b80 02002542
	v_add_co_u32 v130, s12, v70, s18                           // 000000002494: d7000c82 02002546
	v_add_co_u32 v132, s13, v74, s18                           // 00000000249c: d7000d84 0200254a
	v_add_co_u32 v134, s14, v78, s18                           // 0000000024a4: d7000e86 0200254e
	v_add_co_u32 v136, s15, v80, s18                           // 0000000024ac: d7000f88 02002550
	v_add_co_u32 v138, s16, v82, s18                           // 0000000024b4: d700108a 02002552
	v_add_co_u32 v140, s17, v84, s18                           // 0000000024bc: d700118c 02002554
	v_add_co_u32 v142, s18, v86, s18                           // 0000000024c4: d700128e 02002556
	s_wait_alu depctr_va_sdst(0)                               // 0000000024cc: bf88f19f
	v_add_co_ci_u32_e64 v113, null, s19, v41, s3               // 0000000024d0: d5207c71 000e5213
	v_add_co_ci_u32_e64 v115, null, s19, v44, s4               // 0000000024d8: d5207c73 00125813
	v_add_co_ci_u32_e64 v117, null, s19, v47, s5               // 0000000024e0: d5207c75 00165e13
	v_add_co_ci_u32_e64 v119, null, s19, v50, s6               // 0000000024e8: d5207c77 001a6413
	v_add_co_ci_u32_e64 v121, null, s19, v53, s7               // 0000000024f0: d5207c79 001e6a13
	v_add_co_ci_u32_e64 v123, null, s19, v56, s8               // 0000000024f8: d5207c7b 00227013
	v_add_co_ci_u32_e64 v125, null, s19, v60, s9               // 000000002500: d5207c7d 00267813
	v_add_co_ci_u32_e64 v127, null, s19, v65, s10              // 000000002508: d5207c7f 002a8213
	v_add_co_ci_u32_e64 v129, null, s19, v67, s11              // 000000002510: d5207c81 002e8613
	v_add_co_ci_u32_e64 v131, null, s19, v71, s12              // 000000002518: d5207c83 00328e13
	v_add_co_ci_u32_e64 v133, null, s19, v75, s13              // 000000002520: d5207c85 00369613
	v_add_co_ci_u32_e64 v135, null, s19, v79, s14              // 000000002528: d5207c87 003a9e13
	v_add_co_ci_u32_e64 v137, null, s19, v81, s15              // 000000002530: d5207c89 003ea213
	v_add_co_ci_u32_e64 v139, null, s19, v83, s16              // 000000002538: d5207c8b 0042a613
	v_add_co_ci_u32_e64 v141, null, s19, v85, s17              // 000000002540: d5207c8d 0046aa13
	v_add_co_ci_u32_e64 v143, null, s19, v87, s18              // 000000002548: d5207c8f 004aae13
	s_barrier_signal -1                                        // 000000002550: be804ec1
	s_barrier_wait 0xffff                                      // 000000002554: bf94ffff
	s_mul_u64 s[4:5], s[20:21], s[28:29]                       // 000000002558: aa841c14
	s_add_nc_u64 s[20:21], s[20:21], 1                         // 00000000255c: a9948114
	s_wait_alu depctr_sa_sdst(0)                               // 000000002560: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000002564: 84848204
	s_cmp_lg_u64 s[20:21], s[26:27]                            // 000000002568: bf111a14
	s_wait_alu depctr_sa_sdst(0)                               // 00000000256c: bf88ff9e
	s_add_nc_u64 s[4:5], s[24:25], s[4:5]                      // 000000002570: a9840418
	s_wait_alu depctr_sa_sdst(0)                               // 000000002574: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[30:31]                      // 000000002578: a9861e04
	s_wait_loadcnt 0x5                                         // 00000000257c: bfc00005
	ds_store_b128 v8, v[88:91]                                 // 000000002580: db7c0000 00005808
	s_wait_loadcnt 0x4                                         // 000000002588: bfc00004
	ds_store_b128 v8, v[92:95] offset:4608                     // 00000000258c: db7c1200 00005c08
	s_wait_loadcnt 0x3                                         // 000000002594: bfc00003
	ds_store_b128 v8, v[96:99] offset:9216                     // 000000002598: db7c2400 00006008
	s_wait_loadcnt 0x2                                         // 0000000025a0: bfc00002
	ds_store_b128 v8, v[100:103] offset:13824                  // 0000000025a4: db7c3600 00006408
	s_wait_loadcnt 0x1                                         // 0000000025ac: bfc00001
	ds_store_b128 v8, v[104:107] offset:18432                  // 0000000025b0: db7c4800 00006808
	s_wait_loadcnt 0x0                                         // 0000000025b8: bfc00000
	ds_store_b128 v7, v[108:111] offset:18432                  // 0000000025bc: db7c4800 00006c07
	s_wait_dscnt 0x0                                           // 0000000025c4: bfc60000
	s_barrier_signal -1                                        // 0000000025c8: be804ec1
	s_barrier_wait 0xffff                                      // 0000000025cc: bf94ffff
	s_clause 0xf                                               // 0000000025d0: bf85000f
	global_load_b32 v176, v[112:113], off                      // 0000000025d4: ee05007c 000000b0 00000070
	global_load_b32 v177, v[114:115], off                      // 0000000025e0: ee05007c 000000b1 00000072
	global_load_b32 v178, v[116:117], off                      // 0000000025ec: ee05007c 000000b2 00000074
	global_load_b32 v179, v[118:119], off                      // 0000000025f8: ee05007c 000000b3 00000076
	global_load_b32 v180, v[120:121], off                      // 000000002604: ee05007c 000000b4 00000078
	global_load_b32 v181, v[122:123], off                      // 000000002610: ee05007c 000000b5 0000007a
	global_load_b32 v182, v[124:125], off                      // 00000000261c: ee05007c 000000b6 0000007c
	global_load_b32 v183, v[126:127], off                      // 000000002628: ee05007c 000000b7 0000007e
	global_load_b32 v184, v[128:129], off                      // 000000002634: ee05007c 000000b8 00000080
	global_load_b32 v185, v[130:131], off                      // 000000002640: ee05007c 000000b9 00000082
	global_load_b32 v186, v[132:133], off                      // 00000000264c: ee05007c 000000ba 00000084
	global_load_b32 v187, v[134:135], off                      // 000000002658: ee05007c 000000bb 00000086
	global_load_b32 v188, v[136:137], off                      // 000000002664: ee05007c 000000bc 00000088
	global_load_b32 v189, v[138:139], off                      // 000000002670: ee05007c 000000bd 0000008a
	global_load_b32 v190, v[140:141], off                      // 00000000267c: ee05007c 000000be 0000008c
	global_load_b32 v191, v[142:143], off                      // 000000002688: ee05007c 000000bf 0000008e
	v_add_nc_u32_e32 v88, 0x4800, v26                          // 000000002694: 4ab034ff 00004800
	v_add_nc_u32_e32 v89, 0x4800, v28                          // 00000000269c: 4ab238ff 00004800
	ds_load_2addr_b64 v[110:113], v21 offset1:2                // 0000000026a4: d9dc0200 6e000015
	ds_load_2addr_b64 v[124:127], v24 offset1:2                // 0000000026ac: d9dc0200 7c000018
	ds_load_2addr_b64 v[128:131], v21 offset0:4 offset1:6      // 0000000026b4: d9dc0604 80000015
	ds_load_2addr_b64 v[114:117], v88 offset1:2                // 0000000026bc: d9dc0200 72000058
	ds_load_2addr_b64 v[120:123], v89 offset1:2                // 0000000026c4: d9dc0200 78000059
	ds_load_2addr_b64 v[136:139], v88 offset0:4 offset1:6      // 0000000026cc: d9dc0604 88000058
	ds_load_2addr_b64 v[144:147], v89 offset0:4 offset1:6      // 0000000026d4: d9dc0604 90000059
	ds_load_2addr_b64 v[152:155], v24 offset0:4 offset1:6      // 0000000026dc: d9dc0604 98000018
	ds_load_2addr_b64 v[132:135], v21 offset0:8 offset1:10     // 0000000026e4: d9dc0a08 84000015
	ds_load_2addr_b64 v[140:143], v88 offset0:8 offset1:10     // 0000000026ec: d9dc0a08 8c000058
	ds_load_2addr_b64 v[148:151], v89 offset0:8 offset1:10     // 0000000026f4: d9dc0a08 94000059
	ds_load_2addr_b64 v[156:159], v24 offset0:8 offset1:10     // 0000000026fc: d9dc0a08 9c000018
	ds_load_2addr_b64 v[160:163], v21 offset0:12 offset1:14    // 000000002704: d9dc0e0c a0000015
	ds_load_2addr_b64 v[164:167], v24 offset0:12 offset1:14    // 00000000270c: d9dc0e0c a4000018
	ds_load_2addr_b64 v[168:171], v88 offset0:12 offset1:14    // 000000002714: d9dc0e0c a8000058
	ds_load_2addr_b64 v[172:175], v89 offset0:12 offset1:14    // 00000000271c: d9dc0e0c ac000059
	s_clause 0x1                                               // 000000002724: bf850001
	s_load_b32 s3, s[6:7], 0x0                                 // 000000002728: f40000c3 f8000000
	s_load_b32 s4, s[4:5], 0x0                                 // 000000002730: f4000102 f8000000
	s_wait_dscnt 0xc                                           // 000000002738: bfc6000c
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[110:111], v[114:115], 0// 00000000273c: cc464058 1a02e56e
	s_wait_dscnt 0xb                                           // 000000002744: bfc6000b
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[110:111], v[120:121], 0// 000000002748: cc464060 1a02f16e
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[124:125], v[114:115], 0// 000000002750: cc464068 1a02e57c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002758: bf870193
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[112:113], v[116:117], v[88:95]// 00000000275c: cc464058 1d62e970
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[112:113], v[122:123], v[96:103]// 000000002764: cc464060 1d82f570
	s_delay_alu instid0(valu_dep_3)                            // 00000000276c: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[126:127], v[116:117], v[104:111]// 000000002770: cc464068 1da2e97e
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[124:125], v[120:121], 0// 000000002778: cc464070 1a02f17c
	s_wait_dscnt 0xa                                           // 000000002780: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[128:129], v[136:137], v[88:95]// 000000002784: cc464058 1d631180
	s_wait_dscnt 0x9                                           // 00000000278c: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[128:129], v[144:145], v[96:103]// 000000002790: cc464060 1d832180
	s_wait_dscnt 0x8                                           // 000000002798: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[152:153], v[136:137], v[104:111]// 00000000279c: cc464068 1da31198
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[126:127], v[122:123], v[112:119]// 0000000027a4: cc464070 1dc2f57e
	s_wait_kmcnt 0x0                                           // 0000000027ac: bfc70000
	v_mov_b32_e32 v120, s3                                     // 0000000027b0: 7ef00203
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[130:131], v[138:139], v[88:95]// 0000000027b4: cc464058 1d631582
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[130:131], v[146:147], v[96:103]// 0000000027bc: cc464060 1d832582
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[154:155], v[138:139], v[104:111]// 0000000027c4: cc464068 1da3159a
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[152:153], v[144:145], v[112:119]// 0000000027cc: cc464070 1dc32198
	v_cndmask_b32_e32 v121, s4, v120, vcc_lo                   // 0000000027d4: 02f2f004
	s_wait_dscnt 0x6                                           // 0000000027d8: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[132:133], v[140:141], v[88:95]// 0000000027dc: cc464058 1d631984
	s_wait_dscnt 0x5                                           // 0000000027e4: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[132:133], v[148:149], v[96:103]// 0000000027e8: cc464060 1d832984
	s_wait_dscnt 0x4                                           // 0000000027f0: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[156:157], v[140:141], v[104:111]// 0000000027f4: cc464068 1da3199c
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[154:155], v[146:147], v[112:119]// 0000000027fc: cc464070 1dc3259a
	v_cndmask_b32_e64 v120, s4, v120, s2                       // 000000002804: d5010078 000af004
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[134:135], v[142:143], v[88:95]// 00000000280c: cc464058 1d631d86
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[134:135], v[150:151], v[96:103]// 000000002814: cc464060 1d832d86
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[158:159], v[142:143], v[104:111]// 00000000281c: cc464068 1da31d9e
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[156:157], v[148:149], v[112:119]// 000000002824: cc464070 1dc3299c
	s_wait_dscnt 0x1                                           // 00000000282c: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[160:161], v[168:169], v[88:95]// 000000002830: cc464058 1d6351a0
	s_wait_dscnt 0x0                                           // 000000002838: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[160:161], v[172:173], v[96:103]// 00000000283c: cc464060 1d8359a0
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[164:165], v[168:169], v[104:111]// 000000002844: cc464068 1da351a4
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[158:159], v[150:151], v[112:119]// 00000000284c: cc464070 1dc32d9e
	v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[162:163], v[170:171], v[88:95]// 000000002854: cc464058 1d6355a2
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000285c: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[162:163], v[174:175], v[96:103]// 000000002860: cc464060 1d835da2
	v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[166:167], v[170:171], v[104:111]// 000000002868: cc464068 1da355a6
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 000000002870: bf870094
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[164:165], v[172:173], v[112:119]// 000000002874: cc464070 1dc359a4
	v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[166:167], v[174:175], v[112:119]// 00000000287c: cc464070 1dc35da6
	s_wait_loadcnt 0xf                                         // 000000002884: bfc0000f
	v_mul_f32_e32 v122, v176, v121                             // 000000002888: 10f4f3b0
	s_wait_loadcnt 0xd                                         // 00000000288c: bfc0000d
	v_dual_mul_f32 v123, v121, v177 :: v_dual_mul_f32 v124, v121, v178// 000000002890: c8c76379 7b7d6579
	s_wait_loadcnt 0xb                                         // 000000002898: bfc0000b
	v_dual_mul_f32 v125, v121, v179 :: v_dual_mul_f32 v126, v121, v180// 00000000289c: c8c76779 7d7f6979
	s_wait_loadcnt 0x9                                         // 0000000028a4: bfc00009
	v_dual_mul_f32 v127, v121, v181 :: v_dual_mul_f32 v128, v121, v182// 0000000028a8: c8c76b79 7f816d79
	s_wait_loadcnt 0x8                                         // 0000000028b0: bfc00008
	v_dual_mul_f32 v129, v121, v183 :: v_dual_mul_f32 v130, v176, v120// 0000000028b4: c8c76f79 8182f1b0
	v_dual_mul_f32 v131, v120, v177 :: v_dual_mul_f32 v132, v120, v178// 0000000028bc: c8c76378 83856578
	v_dual_mul_f32 v133, v120, v179 :: v_dual_mul_f32 v134, v120, v180// 0000000028c4: c8c76778 85876978
	v_dual_mul_f32 v135, v120, v181 :: v_dual_mul_f32 v136, v120, v182// 0000000028cc: c8c76b78 87896d78
	s_wait_loadcnt 0x7                                         // 0000000028d4: bfc00007
	v_dual_mul_f32 v137, v120, v183 :: v_dual_mul_f32 v138, v121, v184// 0000000028d8: c8c76f78 898b7179
	s_wait_loadcnt 0x5                                         // 0000000028e0: bfc00005
	v_dual_mul_f32 v139, v121, v185 :: v_dual_mul_f32 v140, v121, v186// 0000000028e4: c8c77379 8b8d7579
	s_wait_loadcnt 0x3                                         // 0000000028ec: bfc00003
	v_dual_mul_f32 v141, v121, v187 :: v_dual_mul_f32 v142, v121, v188// 0000000028f0: c8c77779 8d8f7979
	s_wait_loadcnt 0x1                                         // 0000000028f8: bfc00001
	v_dual_mul_f32 v143, v121, v189 :: v_dual_mul_f32 v144, v121, v190// 0000000028fc: c8c77b79 8f917d79
	s_wait_loadcnt 0x0                                         // 000000002904: bfc00000
	v_mul_f32_e32 v121, v121, v191                             // 000000002908: 10f37f79
	v_dual_mul_f32 v145, v120, v184 :: v_dual_mul_f32 v146, v120, v185// 00000000290c: c8c77178 91937378
	v_dual_mul_f32 v147, v120, v186 :: v_dual_mul_f32 v148, v120, v187// 000000002914: c8c77578 93957778
	v_dual_mul_f32 v149, v120, v188 :: v_dual_mul_f32 v150, v120, v189// 00000000291c: c8c77978 95977b78
	v_dual_mul_f32 v151, v120, v190 :: v_dual_mul_f32 v120, v120, v191// 000000002924: c8c77d78 97797f78
	v_dual_mul_f32 v88, v88, v122 :: v_dual_mul_f32 v89, v89, v123// 00000000292c: c8c6f558 5858f759
	v_dual_mul_f32 v90, v90, v124 :: v_dual_mul_f32 v91, v91, v125// 000000002934: c8c6f95a 5a5afb5b
	v_dual_mul_f32 v92, v92, v126 :: v_dual_mul_f32 v93, v93, v127// 00000000293c: c8c6fd5c 5c5cff5d
	v_dual_mul_f32 v94, v94, v128 :: v_dual_mul_f32 v95, v95, v129// 000000002944: c8c7015e 5e5f035f
	v_dual_mul_f32 v96, v96, v130 :: v_dual_mul_f32 v97, v97, v131// 00000000294c: c8c70560 60610761
	v_dual_mul_f32 v98, v98, v132 :: v_dual_mul_f32 v99, v99, v133// 000000002954: c8c70962 62630b63
	v_dual_mul_f32 v100, v100, v134 :: v_dual_mul_f32 v101, v101, v135// 00000000295c: c8c70d64 64650f65
	v_dual_mul_f32 v102, v102, v136 :: v_dual_mul_f32 v103, v103, v137// 000000002964: c8c71166 66671367
	v_dual_mul_f32 v104, v104, v138 :: v_dual_mul_f32 v105, v105, v139// 00000000296c: c8c71568 68691769
	v_dual_mul_f32 v106, v106, v140 :: v_dual_mul_f32 v107, v107, v141// 000000002974: c8c7196a 6a6b1b6b
	v_dual_mul_f32 v108, v108, v142 :: v_dual_mul_f32 v109, v109, v143// 00000000297c: c8c71d6c 6c6d1f6d
	v_dual_mul_f32 v110, v110, v144 :: v_dual_mul_f32 v111, v111, v121// 000000002984: c8c7216e 6e6ef36f
	v_dual_mul_f32 v112, v112, v145 :: v_dual_mul_f32 v113, v113, v146// 00000000298c: c8c72370 70712571
	v_dual_mul_f32 v114, v114, v147 :: v_dual_mul_f32 v115, v115, v148// 000000002994: c8c72772 72732973
	v_dual_mul_f32 v116, v116, v149 :: v_dual_mul_f32 v117, v117, v150// 00000000299c: c8c72b74 74752d75
	v_dual_mul_f32 v118, v118, v151 :: v_dual_mul_f32 v119, v119, v120// 0000000029a4: c8c72f76 7676f177
	v_dual_add_f32 v6, v6, v88 :: v_dual_add_f32 v77, v77, v89 // 0000000029ac: c908b106 064cb34d
	v_dual_add_f32 v76, v76, v90 :: v_dual_add_f32 v73, v73, v91// 0000000029b4: c908b54c 4c48b749
	v_dual_add_f32 v72, v72, v92 :: v_dual_add_f32 v69, v69, v93// 0000000029bc: c908b948 4844bb45
	v_add_f32_e32 v68, v68, v94                                // 0000000029c4: 0688bd44
	v_add_f32_e32 v64, v64, v95                                // 0000000029c8: 0680bf40
	v_dual_add_f32 v42, v42, v96 :: v_dual_add_f32 v39, v39, v97// 0000000029cc: c908c12a 2a26c327
	v_dual_add_f32 v38, v38, v98 :: v_dual_add_f32 v37, v37, v99// 0000000029d4: c908c526 2624c725
	v_dual_add_f32 v36, v36, v100 :: v_dual_add_f32 v35, v35, v101// 0000000029dc: c908c924 2422cb23
	v_add_f32_e32 v34, v34, v102                               // 0000000029e4: 0644cd22
	v_add_f32_e32 v32, v32, v103                               // 0000000029e8: 0640cf20
	v_dual_add_f32 v62, v62, v104 :: v_dual_add_f32 v61, v61, v105// 0000000029ec: c908d13e 3e3cd33d
	v_dual_add_f32 v58, v58, v106 :: v_dual_add_f32 v57, v57, v107// 0000000029f4: c908d53a 3a38d739
	v_dual_add_f32 v54, v54, v108 :: v_dual_add_f32 v51, v51, v109// 0000000029fc: c908d936 3632db33
	v_dual_add_f32 v48, v48, v110 :: v_dual_add_f32 v45, v45, v111// 000000002a04: c908dd30 302cdf2d
	v_add_f32_e32 v33, v33, v112                               // 000000002a0c: 0642e121
	v_dual_add_f32 v31, v31, v113 :: v_dual_add_f32 v30, v30, v114// 000000002a10: c908e31f 1f1ee51e
	v_add_f32_e32 v29, v29, v115                               // 000000002a18: 063ae71d
	v_add_f32_e32 v27, v27, v116                               // 000000002a1c: 0636e91b
	v_add_f32_e32 v25, v25, v117                               // 000000002a20: 0632eb19
	v_dual_add_f32 v23, v23, v118 :: v_dual_add_f32 v22, v22, v119// 000000002a24: c908ed17 1716ef16
	s_cbranch_scc1 65110                                       // 000000002a2c: bfa2fe56 <tessera_rocm_scaled_matmul_lds_6bb337b5a3dd5b79+0x888>
	s_load_b64 s[2:3], s[0:1], 0xa8                            // 000000002a30: f4002080 f80000a8
	v_mul_lo_u32 v7, s23, v0                                   // 000000002a38: d72c0007 02020017
	v_mul_lo_u32 v8, s22, v1                                   // 000000002a40: d72c0008 02020216
	v_mad_co_u64_u32 v[0:1], null, s22, v0, 0                  // 000000002a48: d6fe7c00 02020016
	v_bfe_u32 v9, v6, 16, 1                                    // 000000002a50: d6100009 02052106
	v_or_b32_e32 v10, 0x400000, v6                             // 000000002a58: 38140cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000002a60: 7c300d06
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002a64: 3e040481
	s_lshl_b64 s[0:1], s[22:23], 1                             // 000000002a68: 84808116
	v_add3_u32 v9, v9, v6, 0x7fff                              // 000000002a6c: d6550009 03fe0d09 00007fff
	v_or_b32_e32 v13, 0x400000, v76                            // 000000002a78: 381a98ff 00400000
	v_add3_u32 v1, v1, v8, v7                                  // 000000002a80: d6550001 041e1101
	v_bfe_u32 v7, v77, 16, 1                                   // 000000002a88: d6100007 0205214d
	v_or_b32_e32 v8, 0x400000, v77                             // 000000002a90: 38109aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002a98: bf88ff9d
	v_cndmask_b32_e32 v10, v9, v10, vcc_lo                     // 000000002a9c: 02141509
	v_bfe_u32 v15, v73, 16, 1                                  // 000000002aa0: d610000f 02052149
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002aa8: 3e000081
	v_add3_u32 v7, v7, v77, 0x7fff                             // 000000002aac: d6550007 03fe9b07 00007fff
	v_or_b32_e32 v16, 0x400000, v73                            // 000000002ab8: 382092ff 00400000
	v_or_b32_e32 v19, 0x400000, v69                            // 000000002ac0: 38268aff 00400000
	v_add3_u32 v15, v15, v73, 0x7fff                           // 000000002ac8: d655000f 03fe930f 00007fff
	v_bfe_u32 v21, v68, 16, 1                                  // 000000002ad4: d6100015 02052144
	s_wait_kmcnt 0x0                                           // 000000002adc: bfc70000
	v_add_co_u32 v6, vcc_lo, s2, v0                            // 000000002ae0: d7006a06 02020002
	s_wait_alu depctr_va_vcc(0)                                // 000000002ae8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s3, v1, vcc_lo               // 000000002aec: d5207c09 01aa0203
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000002af4: 7c309b4d
	v_add3_u32 v21, v21, v68, 0x7fff                           // 000000002af8: d6550015 03fe8915 00007fff
	v_or_b32_e32 v24, 0x400000, v68                            // 000000002b04: 383088ff 00400000
	v_mul_lo_u32 v26, s22, v5                                  // 000000002b0c: d72c001a 02020a16
	v_or_b32_e32 v28, 0x400000, v64                            // 000000002b14: 383880ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002b1c: bf88ff9d
	v_cndmask_b32_e32 v11, v7, v8, vcc_lo                      // 000000002b20: 02161107
	v_add_co_u32 v0, vcc_lo, v6, v2                            // 000000002b24: d7006a00 02020506
	s_wait_alu depctr_va_vcc(0)                                // 000000002b2c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v9, v3, vcc_lo               // 000000002b30: d5207c01 01aa0709
	v_add_co_u32 v8, vcc_lo, v6, s0                            // 000000002b38: d7006a08 02000106
	v_bfe_u32 v7, v76, 16, 1                                   // 000000002b40: d6100007 0205214c
	s_wait_alu depctr_va_vcc(0)                                // 000000002b48: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s1, v9, vcc_lo               // 000000002b4c: d5207c09 01aa1201
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002b54: bf870193
	v_add_co_u32 v6, vcc_lo, v8, v2                            // 000000002b58: d7006a06 02020508
	v_add3_u32 v12, v7, v76, 0x7fff                            // 000000002b60: d655000c 03fe9907 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002b6c: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002b70: bf870003
	v_add_co_ci_u32_e64 v7, null, v9, v3, vcc_lo               // 000000002b74: d5207c07 01aa0709
	v_cmp_u_f32_e32 vcc_lo, v76, v76                           // 000000002b7c: 7c30994c
	v_bfe_u32 v40, v61, 16, 1                                  // 000000002b80: d6100028 0205213d
	v_or_b32_e32 v41, 0x400000, v61                            // 000000002b88: 38527aff 00400000
	v_or_b32_e32 v44, 0x400000, v57                            // 000000002b90: 385872ff 00400000
	v_bfe_u32 v49, v54, 16, 1                                  // 000000002b98: d6100031 02052136
	s_wait_alu depctr_va_vcc(0)                                // 000000002ba0: bf88ff9d
	v_cndmask_b32_e32 v12, v12, v13, vcc_lo                    // 000000002ba4: 02181b0c
	v_add_co_u32 v13, vcc_lo, v8, s0                           // 000000002ba8: d7006a0d 02000108
	s_wait_alu depctr_va_vcc(0)                                // 000000002bb0: bf88ff9d
	v_add_co_ci_u32_e64 v14, null, s1, v9, vcc_lo              // 000000002bb4: d5207c0e 01aa1201
	v_add3_u32 v40, v40, v61, 0x7fff                           // 000000002bbc: d6550028 03fe7b28 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002bc8: bf8701a3
	v_add_co_u32 v8, vcc_lo, v13, v2                           // 000000002bcc: d7006a08 0202050d
	s_wait_alu depctr_va_vcc(0)                                // 000000002bd4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v14, v3, vcc_lo              // 000000002bd8: d5207c09 01aa070e
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 000000002be0: 7c309349
	s_clause 0x2                                               // 000000002be4: bf850002
	global_store_d16_hi_b16 v[0:1], v10, off                   // 000000002be8: ee09407c 05000000 00000000
	global_store_d16_hi_b16 v[6:7], v11, off                   // 000000002bf4: ee09407c 05800000 00000006
	global_store_d16_hi_b16 v[8:9], v12, off                   // 000000002c00: ee09407c 06000000 00000008
	v_bfe_u32 v10, v72, 16, 1                                  // 000000002c0c: d610000a 02052148
	v_add3_u32 v49, v49, v54, 0x7fff                           // 000000002c14: d6550031 03fe6d31 00007fff
	v_or_b32_e32 v50, 0x400000, v54                            // 000000002c20: 38646cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c28: bf88ff9d
	v_cndmask_b32_e32 v16, v15, v16, vcc_lo                    // 000000002c2c: 0220210f
	v_add_co_u32 v12, vcc_lo, v13, s0                          // 000000002c30: d7006a0c 0200010d
	s_wait_alu depctr_va_vcc(0)                                // 000000002c38: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s1, v14, vcc_lo             // 000000002c3c: d5207c0d 01aa1c01
	v_add3_u32 v14, v10, v72, 0x7fff                           // 000000002c44: d655000e 03fe910a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c50: bf870003
	v_add_co_u32 v10, vcc_lo, v12, v2                          // 000000002c54: d7006a0a 0202050c
	v_or_b32_e32 v15, 0x400000, v72                            // 000000002c5c: 381e90ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c64: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v13, v3, vcc_lo             // 000000002c68: d5207c0b 01aa070d
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 000000002c70: 7c309148
	v_or_b32_e32 v52, 0x400000, v48                            // 000000002c74: 386860ff 00400000
	v_or_b32_e32 v55, 0x400000, v45                            // 000000002c7c: 386e5aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c84: bf88ff9d
	v_cndmask_b32_e32 v17, v14, v15, vcc_lo                    // 000000002c88: 02221f0e
	v_add_co_u32 v15, vcc_lo, v12, s0                          // 000000002c8c: d7006a0f 0200010c
	v_bfe_u32 v14, v69, 16, 1                                  // 000000002c94: d610000e 02052145
	s_wait_alu depctr_va_vcc(0)                                // 000000002c9c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v13, vcc_lo             // 000000002ca0: d5207c12 01aa1a01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ca8: bf870193
	v_add_co_u32 v12, vcc_lo, v15, v2                          // 000000002cac: d7006a0c 0202050f
	v_add3_u32 v14, v14, v69, 0x7fff                           // 000000002cb4: d655000e 03fe8b0e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002cc0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002cc4: bf870003
	v_add_co_ci_u32_e64 v13, null, v18, v3, vcc_lo             // 000000002cc8: d5207c0d 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000002cd0: 7c308b45
	s_wait_alu depctr_va_vcc(0)                                // 000000002cd4: bf88ff9d
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo                    // 000000002cd8: 0226270e
	v_add_co_u32 v20, vcc_lo, v15, s0                          // 000000002cdc: d7006a14 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 000000002ce4: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000002ce8: d5207c12 01aa2401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cf0: bf870122
	v_add_co_u32 v14, vcc_lo, v20, v2                          // 000000002cf4: d7006a0e 02020514
	s_wait_alu depctr_va_vcc(0)                                // 000000002cfc: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v18, v3, vcc_lo             // 000000002d00: d5207c0f 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 000000002d08: 7c308944
	s_clause 0x2                                               // 000000002d0c: bf850002
	global_store_d16_hi_b16 v[10:11], v16, off                 // 000000002d10: ee09407c 08000000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 000000002d1c: ee09407c 08800000 0000000c
	global_store_d16_hi_b16 v[14:15], v19, off                 // 000000002d28: ee09407c 09800000 0000000e
	v_bfe_u32 v16, v64, 16, 1                                  // 000000002d34: d6100010 02052140
	s_wait_alu depctr_va_vcc(0)                                // 000000002d3c: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v24, vcc_lo                    // 000000002d40: 022a3115
	v_add_co_u32 v19, vcc_lo, v20, s0                          // 000000002d44: d7006a13 02000114
	s_wait_alu depctr_va_vcc(0)                                // 000000002d4c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000002d50: d5207c12 01aa2401
	v_add3_u32 v20, v16, v64, 0x7fff                           // 000000002d58: d6550014 03fe8110 00007fff
	v_mul_lo_u32 v24, s23, v4                                  // 000000002d64: d72c0018 02020817
	v_mad_co_u64_u32 v[4:5], null, s22, v4, 0                  // 000000002d6c: d6fe7c04 02020816
	v_add_co_u32 v16, vcc_lo, v19, v2                          // 000000002d74: d7006a10 02020513
	s_wait_alu depctr_va_vcc(0)                                // 000000002d7c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v18, v3, vcc_lo             // 000000002d80: d5207c11 01aa0712
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 000000002d88: 7c308140
	s_delay_alu instid0(valu_dep_4)                            // 000000002d8c: bf870004
	v_add3_u32 v5, v5, v26, v24                                // 000000002d90: d6550005 04623505
	v_bfe_u32 v24, v62, 16, 1                                  // 000000002d98: d6100018 0205213e
	s_wait_alu depctr_va_vcc(0)                                // 000000002da0: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v28, vcc_lo                    // 000000002da4: 02283914
	v_add_co_u32 v19, vcc_lo, v19, s0                          // 000000002da8: d7006a13 02000113
	s_wait_alu depctr_va_vcc(0)                                // 000000002db0: bf88ff9d
	v_add_co_ci_u32_e64 v26, null, s1, v18, vcc_lo             // 000000002db4: d5207c1a 01aa2401
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000002dbc: 3e080881
	s_delay_alu instid0(valu_dep_3)                            // 000000002dc0: bf870003
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000002dc4: d7006a12 02020513
	v_add3_u32 v24, v24, v62, 0x7fff                           // 000000002dcc: d6550018 03fe7d18 00007fff
	v_or_b32_e32 v28, 0x400000, v62                            // 000000002dd8: 38387cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002de0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v26, v3, vcc_lo             // 000000002de4: d5207c13 01aa071a
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000002dec: 7c307d3e
	s_wait_alu depctr_va_vcc(0)                                // 000000002df0: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v28, vcc_lo                    // 000000002df4: 02303918
	v_add_co_u32 v26, vcc_lo, s2, v4                           // 000000002df8: d7006a1a 02020802
	s_wait_alu depctr_va_vcc(0)                                // 000000002e00: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s3, v5, vcc_lo              // 000000002e04: d5207c1c 01aa0a03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002e0c: bf870122
	v_add_co_u32 v4, vcc_lo, v26, v2                           // 000000002e10: d7006a04 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 000000002e18: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v28, v3, vcc_lo              // 000000002e1c: d5207c05 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000002e24: 7c307b3d
	s_clause 0x2                                               // 000000002e28: bf850002
	global_store_d16_hi_b16 v[16:17], v21, off                 // 000000002e2c: ee09407c 0a800000 00000010
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000002e38: ee09407c 0a000000 00000012
	global_store_d16_hi_b16 v[4:5], v24, off                   // 000000002e44: ee09407c 0c000000 00000004
	v_bfe_u32 v20, v58, 16, 1                                  // 000000002e50: d6100014 0205213a
	s_wait_alu depctr_va_vcc(0)                                // 000000002e58: bf88ff9d
	v_cndmask_b32_e32 v24, v40, v41, vcc_lo                    // 000000002e5c: 02305328
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000002e60: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000002e68: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000002e6c: d5207c1c 01aa3801
	v_add3_u32 v40, v20, v58, 0x7fff                           // 000000002e74: d6550028 03fe7514 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e80: bf870003
	v_add_co_u32 v20, vcc_lo, v26, v2                          // 000000002e84: d7006a14 0202051a
	v_or_b32_e32 v41, 0x400000, v58                            // 000000002e8c: 385274ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002e94: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v28, v3, vcc_lo             // 000000002e98: d5207c15 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000002ea0: 7c30753a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ea4: bf88ff9d
	v_cndmask_b32_e32 v46, v40, v41, vcc_lo                    // 000000002ea8: 025c5328
	v_bfe_u32 v40, v57, 16, 1                                  // 000000002eac: d6100028 02052139
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000002eb4: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ebc: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000002ec0: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ec8: bf870193
	v_add3_u32 v43, v40, v57, 0x7fff                           // 000000002ecc: d655002b 03fe7328 00007fff
	v_add_co_u32 v40, vcc_lo, v26, v2                          // 000000002ed8: d7006a28 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ee0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ee4: bf870003
	v_add_co_ci_u32_e64 v41, null, v28, v3, vcc_lo             // 000000002ee8: d5207c29 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000002ef0: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 000000002ef4: bf88ff9d
	v_cndmask_b32_e32 v47, v43, v44, vcc_lo                    // 000000002ef8: 025e592b
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000002efc: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000002f04: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000002f08: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f10: bf870122
	v_add_co_u32 v43, vcc_lo, v26, v2                          // 000000002f14: d7006a2b 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 000000002f1c: bf88ff9d
	v_add_co_ci_u32_e64 v44, null, v28, v3, vcc_lo             // 000000002f20: d5207c2c 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 000000002f28: 7c306d36
	s_clause 0x2                                               // 000000002f2c: bf850002
	global_store_d16_hi_b16 v[20:21], v24, off                 // 000000002f30: ee09407c 0c000000 00000014
	global_store_d16_hi_b16 v[40:41], v46, off                 // 000000002f3c: ee09407c 17000000 00000028
	global_store_d16_hi_b16 v[43:44], v47, off                 // 000000002f48: ee09407c 17800000 0000002b
	v_bfe_u32 v46, v51, 16, 1                                  // 000000002f54: d610002e 02052133
	v_bfe_u32 v54, v45, 16, 1                                  // 000000002f5c: d6100036 0205212d
	s_wait_alu depctr_va_vcc(0)                                // 000000002f64: bf88ff9d
	v_cndmask_b32_e32 v24, v49, v50, vcc_lo                    // 000000002f68: 02306531
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000002f6c: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000002f74: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000002f78: d5207c1c 01aa3801
	v_add3_u32 v49, v46, v51, 0x7fff                           // 000000002f80: d6550031 03fe672e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f8c: bf870003
	v_add_co_u32 v46, vcc_lo, v26, v2                          // 000000002f90: d7006a2e 0202051a
	v_or_b32_e32 v50, 0x400000, v51                            // 000000002f98: 386466ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002fa0: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, v28, v3, vcc_lo             // 000000002fa4: d5207c2f 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000002fac: 7c306733
	v_add3_u32 v54, v54, v45, 0x7fff                           // 000000002fb0: d6550036 03fe5b36 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002fbc: bf88ff9d
	v_cndmask_b32_e32 v53, v49, v50, vcc_lo                    // 000000002fc0: 026a6531
	v_bfe_u32 v49, v48, 16, 1                                  // 000000002fc4: d6100031 02052130
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000002fcc: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 000000002fd4: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000002fd8: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002fe0: bf870193
	v_add3_u32 v51, v49, v48, 0x7fff                           // 000000002fe4: d6550033 03fe6131 00007fff
	v_add_co_u32 v49, vcc_lo, v26, v2                          // 000000002ff0: d7006a31 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ff8: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ffc: bf870003
	v_add_co_ci_u32_e64 v50, null, v28, v3, vcc_lo             // 000000003000: d5207c32 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000003008: 7c306130
	s_wait_alu depctr_va_vcc(0)                                // 00000000300c: bf88ff9d
	v_cndmask_b32_e32 v48, v51, v52, vcc_lo                    // 000000003010: 02606933
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000003014: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 00000000301c: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000003020: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003028: bf870122
	v_add_co_u32 v51, vcc_lo, v26, v2                          // 00000000302c: d7006a33 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 000000003034: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, v28, v3, vcc_lo             // 000000003038: d5207c34 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000003040: 7c305b2d
	s_clause 0x2                                               // 000000003044: bf850002
	global_store_d16_hi_b16 v[46:47], v24, off                 // 000000003048: ee09407c 0c000000 0000002e
	global_store_d16_hi_b16 v[49:50], v53, off                 // 000000003054: ee09407c 1a800000 00000031
	global_store_d16_hi_b16 v[51:52], v48, off                 // 000000003060: ee09407c 18000000 00000033
	v_bfe_u32 v45, v42, 16, 1                                  // 00000000306c: d610002d 0205212a
	v_or_b32_e32 v48, 0x400000, v42                            // 000000003074: 386054ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000307c: bf88ff9d
	v_cndmask_b32_e32 v24, v54, v55, vcc_lo                    // 000000003080: 02306f36
	v_add_co_u32 v26, vcc_lo, v26, s0                          // 000000003084: d7006a1a 0200011a
	s_wait_alu depctr_va_vcc(0)                                // 00000000308c: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 000000003090: d5207c1c 01aa3801
	v_add3_u32 v45, v45, v42, 0x7fff                           // 000000003098: d655002d 03fe552d 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000030a4: bf8701a3
	v_add_co_u32 v2, vcc_lo, v26, v2                           // 0000000030a8: d7006a02 0202051a
	s_wait_alu depctr_va_vcc(0)                                // 0000000030b0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v28, v3, vcc_lo              // 0000000030b4: d5207c03 01aa071c
	v_bfe_u32 v26, v39, 16, 1                                  // 0000000030bc: d610001a 02052127
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 0000000030c4: 7c30552a
	v_bfe_u32 v42, v38, 16, 1                                  // 0000000030c8: d610002a 02052126
	global_store_d16_hi_b16 v[2:3], v24, off                   // 0000000030d0: ee09407c 0c000000 00000002
	v_add3_u32 v24, v26, v39, 0x7fff                           // 0000000030dc: d6550018 03fe4f1a 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000030e8: bf88ff9d
	v_cndmask_b32_e32 v28, v45, v48, vcc_lo                    // 0000000030ec: 0238612d
	v_or_b32_e32 v26, 0x400000, v39                            // 0000000030f0: 38344eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 0000000030f8: 7c304f27
	global_store_d16_hi_b16 v[0:1], v28, off offset:32         // 0000000030fc: ee09407c 0e000000 00002000
	v_add3_u32 v0, v42, v38, 0x7fff                            // 000000003108: d6550000 03fe4d2a 00007fff
	v_or_b32_e32 v1, 0x400000, v38                             // 000000003114: 38024cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000311c: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v26, vcc_lo                    // 000000003120: 02303518
	v_bfe_u32 v26, v37, 16, 1                                  // 000000003124: d610001a 02052125
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 00000000312c: 7c304d26
	global_store_d16_hi_b16 v[6:7], v24, off offset:32         // 000000003130: ee09407c 0c000000 00002006
	v_add3_u32 v6, v26, v37, 0x7fff                            // 00000000313c: d6550006 03fe4b1a 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003148: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000314c: 02000300
	v_bfe_u32 v1, v36, 16, 1                                   // 000000003150: d6100001 02052124
	v_or_b32_e32 v7, 0x400000, v37                             // 000000003158: 380e4aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000003160: 7c304b25
	global_store_d16_hi_b16 v[8:9], v0, off offset:32          // 000000003164: ee09407c 00000000 00002008
	v_add3_u32 v0, v1, v36, 0x7fff                             // 000000003170: d6550000 03fe4901 00007fff
	v_or_b32_e32 v1, 0x400000, v36                             // 00000000317c: 380248ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003184: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000003188: 020c0f06
	v_bfe_u32 v7, v35, 16, 1                                   // 00000000318c: d6100007 02052123
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 000000003194: 7c304924
	v_or_b32_e32 v8, 0x400000, v23                             // 000000003198: 38102eff 00400000
	v_or_b32_e32 v9, 0x400000, v22                             // 0000000031a0: 38122cff 00400000
	global_store_d16_hi_b16 v[10:11], v6, off offset:32        // 0000000031a8: ee09407c 03000000 0000200a
	v_add3_u32 v6, v7, v35, 0x7fff                             // 0000000031b4: d6550006 03fe4707 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000031c0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000031c4: 02000300
	v_bfe_u32 v1, v34, 16, 1                                   // 0000000031c8: d6100001 02052122
	v_or_b32_e32 v7, 0x400000, v35                             // 0000000031d0: 380e46ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 0000000031d8: 7c304723
	global_store_d16_hi_b16 v[12:13], v0, off offset:32        // 0000000031dc: ee09407c 00000000 0000200c
	v_add3_u32 v0, v1, v34, 0x7fff                             // 0000000031e8: d6550000 03fe4501 00007fff
	v_or_b32_e32 v1, 0x400000, v34                             // 0000000031f4: 380244ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000031fc: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000003200: 020c0f06
	v_bfe_u32 v7, v32, 16, 1                                   // 000000003204: d6100007 02052120
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 00000000320c: 7c304522
	global_store_d16_hi_b16 v[14:15], v6, off offset:32        // 000000003210: ee09407c 03000000 0000200e
	v_add3_u32 v6, v7, v32, 0x7fff                             // 00000000321c: d6550006 03fe4107 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003228: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000322c: 02000300
	v_bfe_u32 v1, v33, 16, 1                                   // 000000003230: d6100001 02052121
	v_or_b32_e32 v7, 0x400000, v32                             // 000000003238: 380e40ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000003240: 7c304120
	global_store_d16_hi_b16 v[16:17], v0, off offset:32        // 000000003244: ee09407c 00000000 00002010
	v_add3_u32 v0, v1, v33, 0x7fff                             // 000000003250: d6550000 03fe4301 00007fff
	v_or_b32_e32 v1, 0x400000, v33                             // 00000000325c: 380242ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003264: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000003268: 020c0f06
	v_bfe_u32 v7, v31, 16, 1                                   // 00000000326c: d6100007 0205211f
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 000000003274: 7c304321
	global_store_d16_hi_b16 v[18:19], v6, off offset:32        // 000000003278: ee09407c 03000000 00002012
	v_add3_u32 v6, v7, v31, 0x7fff                             // 000000003284: d6550006 03fe3f07 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003290: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003294: 02000300
	v_bfe_u32 v1, v30, 16, 1                                   // 000000003298: d6100001 0205211e
	v_or_b32_e32 v7, 0x400000, v31                             // 0000000032a0: 380e3eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 0000000032a8: 7c303f1f
	global_store_d16_hi_b16 v[4:5], v0, off offset:32          // 0000000032ac: ee09407c 00000000 00002004
	v_add3_u32 v0, v1, v30, 0x7fff                             // 0000000032b8: d6550000 03fe3d01 00007fff
	v_or_b32_e32 v1, 0x400000, v30                             // 0000000032c4: 38023cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000032cc: bf88ff9d
	v_cndmask_b32_e32 v4, v6, v7, vcc_lo                       // 0000000032d0: 02080f06
	v_bfe_u32 v5, v29, 16, 1                                   // 0000000032d4: d6100005 0205211d
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 0000000032dc: 7c303d1e
	v_bfe_u32 v6, v23, 16, 1                                   // 0000000032e0: d6100006 02052117
	v_or_b32_e32 v7, 0x400000, v25                             // 0000000032e8: 380e32ff 00400000
	global_store_d16_hi_b16 v[20:21], v4, off offset:32        // 0000000032f0: ee09407c 02000000 00002014
	v_add3_u32 v4, v5, v29, 0x7fff                             // 0000000032fc: d6550004 03fe3b05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003308: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000330c: 02000300
	v_bfe_u32 v1, v27, 16, 1                                   // 000000003310: d6100001 0205211b
	v_or_b32_e32 v5, 0x400000, v29                             // 000000003318: 380a3aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000003320: 7c303b1d
	v_add3_u32 v6, v6, v23, 0x7fff                             // 000000003324: d6550006 03fe2f06 00007fff
	global_store_d16_hi_b16 v[40:41], v0, off offset:32        // 000000003330: ee09407c 00000000 00002028
	v_add3_u32 v0, v1, v27, 0x7fff                             // 00000000333c: d6550000 03fe3701 00007fff
	v_or_b32_e32 v1, 0x400000, v27                             // 000000003348: 380236ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003350: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003354: 02080b04
	v_bfe_u32 v5, v25, 16, 1                                   // 000000003358: d6100005 02052119
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000003360: 7c30371b
	s_delay_alu instid0(valu_dep_2)                            // 000000003364: bf870002
	v_add3_u32 v5, v5, v25, 0x7fff                             // 000000003368: d6550005 03fe3305 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003374: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003378: 02000300
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 00000000337c: 7c303319
	v_bfe_u32 v1, v22, 16, 1                                   // 000000003380: d6100001 02052116
	s_wait_alu depctr_va_vcc(0)                                // 000000003388: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 00000000338c: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v23, v23                           // 000000003390: 7c302f17
	s_delay_alu instid0(valu_dep_3)                            // 000000003394: bf870003
	v_add3_u32 v1, v1, v22, 0x7fff                             // 000000003398: d6550001 03fe2d01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000033a4: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo                       // 0000000033a8: 020c1106
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 0000000033ac: 7c302d16
	s_wait_alu depctr_va_vcc(0)                                // 0000000033b0: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo                       // 0000000033b4: 02021301
	s_clause 0x3                                               // 0000000033b8: bf850003
	global_store_d16_hi_b16 v[43:44], v4, off offset:32        // 0000000033bc: ee09407c 02000000 0000202b
	global_store_d16_hi_b16 v[46:47], v0, off offset:32        // 0000000033c8: ee09407c 00000000 0000202e
	global_store_d16_hi_b16 v[49:50], v5, off offset:32        // 0000000033d4: ee09407c 02800000 00002031
	global_store_d16_hi_b16 v[51:52], v6, off offset:32        // 0000000033e0: ee09407c 03000000 00002033
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 0000000033ec: ee09407c 00800000 00002002
	s_nop 0                                                    // 0000000033f8: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000033fc: bfb60003
	s_endpgm                                                   // 000000003400: bfb00000
	s_code_end                                                 // 000000003404: bf9f0000
	s_code_end                                                 // 000000003408: bf9f0000
	s_code_end                                                 // 00000000340c: bf9f0000
	s_code_end                                                 // 000000003410: bf9f0000
	s_code_end                                                 // 000000003414: bf9f0000
	s_code_end                                                 // 000000003418: bf9f0000
	s_code_end                                                 // 00000000341c: bf9f0000
	s_code_end                                                 // 000000003420: bf9f0000
	s_code_end                                                 // 000000003424: bf9f0000
	s_code_end                                                 // 000000003428: bf9f0000
	s_code_end                                                 // 00000000342c: bf9f0000
	s_code_end                                                 // 000000003430: bf9f0000
	s_code_end                                                 // 000000003434: bf9f0000
	s_code_end                                                 // 000000003438: bf9f0000
	s_code_end                                                 // 00000000343c: bf9f0000
	s_code_end                                                 // 000000003440: bf9f0000
	s_code_end                                                 // 000000003444: bf9f0000
	s_code_end                                                 // 000000003448: bf9f0000
	s_code_end                                                 // 00000000344c: bf9f0000
	s_code_end                                                 // 000000003450: bf9f0000
	s_code_end                                                 // 000000003454: bf9f0000
	s_code_end                                                 // 000000003458: bf9f0000
	s_code_end                                                 // 00000000345c: bf9f0000
	s_code_end                                                 // 000000003460: bf9f0000
	s_code_end                                                 // 000000003464: bf9f0000
	s_code_end                                                 // 000000003468: bf9f0000
	s_code_end                                                 // 00000000346c: bf9f0000
	s_code_end                                                 // 000000003470: bf9f0000
	s_code_end                                                 // 000000003474: bf9f0000
	s_code_end                                                 // 000000003478: bf9f0000
	s_code_end                                                 // 00000000347c: bf9f0000
	s_code_end                                                 // 000000003480: bf9f0000
	s_code_end                                                 // 000000003484: bf9f0000
	s_code_end                                                 // 000000003488: bf9f0000
	s_code_end                                                 // 00000000348c: bf9f0000
	s_code_end                                                 // 000000003490: bf9f0000
	s_code_end                                                 // 000000003494: bf9f0000
	s_code_end                                                 // 000000003498: bf9f0000
	s_code_end                                                 // 00000000349c: bf9f0000
	s_code_end                                                 // 0000000034a0: bf9f0000
	s_code_end                                                 // 0000000034a4: bf9f0000
	s_code_end                                                 // 0000000034a8: bf9f0000
	s_code_end                                                 // 0000000034ac: bf9f0000
	s_code_end                                                 // 0000000034b0: bf9f0000
	s_code_end                                                 // 0000000034b4: bf9f0000
	s_code_end                                                 // 0000000034b8: bf9f0000
	s_code_end                                                 // 0000000034bc: bf9f0000
	s_code_end                                                 // 0000000034c0: bf9f0000
	s_code_end                                                 // 0000000034c4: bf9f0000
	s_code_end                                                 // 0000000034c8: bf9f0000
	s_code_end                                                 // 0000000034cc: bf9f0000
	s_code_end                                                 // 0000000034d0: bf9f0000
	s_code_end                                                 // 0000000034d4: bf9f0000
	s_code_end                                                 // 0000000034d8: bf9f0000
	s_code_end                                                 // 0000000034dc: bf9f0000
	s_code_end                                                 // 0000000034e0: bf9f0000
	s_code_end                                                 // 0000000034e4: bf9f0000
	s_code_end                                                 // 0000000034e8: bf9f0000
	s_code_end                                                 // 0000000034ec: bf9f0000
	s_code_end                                                 // 0000000034f0: bf9f0000
	s_code_end                                                 // 0000000034f4: bf9f0000
	s_code_end                                                 // 0000000034f8: bf9f0000
	s_code_end                                                 // 0000000034fc: bf9f0000
	s_code_end                                                 // 000000003500: bf9f0000
	s_code_end                                                 // 000000003504: bf9f0000
	s_code_end                                                 // 000000003508: bf9f0000
	s_code_end                                                 // 00000000350c: bf9f0000
	s_code_end                                                 // 000000003510: bf9f0000
	s_code_end                                                 // 000000003514: bf9f0000
	s_code_end                                                 // 000000003518: bf9f0000
	s_code_end                                                 // 00000000351c: bf9f0000
	s_code_end                                                 // 000000003520: bf9f0000
	s_code_end                                                 // 000000003524: bf9f0000
	s_code_end                                                 // 000000003528: bf9f0000
	s_code_end                                                 // 00000000352c: bf9f0000
	s_code_end                                                 // 000000003530: bf9f0000
	s_code_end                                                 // 000000003534: bf9f0000
	s_code_end                                                 // 000000003538: bf9f0000
	s_code_end                                                 // 00000000353c: bf9f0000
	s_code_end                                                 // 000000003540: bf9f0000
	s_code_end                                                 // 000000003544: bf9f0000
	s_code_end                                                 // 000000003548: bf9f0000
	s_code_end                                                 // 00000000354c: bf9f0000
	s_code_end                                                 // 000000003550: bf9f0000
	s_code_end                                                 // 000000003554: bf9f0000
	s_code_end                                                 // 000000003558: bf9f0000
	s_code_end                                                 // 00000000355c: bf9f0000
	s_code_end                                                 // 000000003560: bf9f0000
	s_code_end                                                 // 000000003564: bf9f0000
	s_code_end                                                 // 000000003568: bf9f0000
	s_code_end                                                 // 00000000356c: bf9f0000
	s_code_end                                                 // 000000003570: bf9f0000
	s_code_end                                                 // 000000003574: bf9f0000
	s_code_end                                                 // 000000003578: bf9f0000
	s_code_end                                                 // 00000000357c: bf9f0000
	s_code_end                                                 // 000000003580: bf9f0000
	s_code_end                                                 // 000000003584: bf9f0000
	s_code_end                                                 // 000000003588: bf9f0000
	s_code_end                                                 // 00000000358c: bf9f0000
	s_code_end                                                 // 000000003590: bf9f0000
	s_code_end                                                 // 000000003594: bf9f0000
	s_code_end                                                 // 000000003598: bf9f0000
	s_code_end                                                 // 00000000359c: bf9f0000
	s_code_end                                                 // 0000000035a0: bf9f0000
	s_code_end                                                 // 0000000035a4: bf9f0000
	s_code_end                                                 // 0000000035a8: bf9f0000
	s_code_end                                                 // 0000000035ac: bf9f0000
	s_code_end                                                 // 0000000035b0: bf9f0000
	s_code_end                                                 // 0000000035b4: bf9f0000
	s_code_end                                                 // 0000000035b8: bf9f0000
	s_code_end                                                 // 0000000035bc: bf9f0000
	s_code_end                                                 // 0000000035c0: bf9f0000
	s_code_end                                                 // 0000000035c4: bf9f0000
	s_code_end                                                 // 0000000035c8: bf9f0000
	s_code_end                                                 // 0000000035cc: bf9f0000
	s_code_end                                                 // 0000000035d0: bf9f0000
	s_code_end                                                 // 0000000035d4: bf9f0000
	s_code_end                                                 // 0000000035d8: bf9f0000
	s_code_end                                                 // 0000000035dc: bf9f0000
	s_code_end                                                 // 0000000035e0: bf9f0000
	s_code_end                                                 // 0000000035e4: bf9f0000
	s_code_end                                                 // 0000000035e8: bf9f0000
	s_code_end                                                 // 0000000035ec: bf9f0000
	s_code_end                                                 // 0000000035f0: bf9f0000
	s_code_end                                                 // 0000000035f4: bf9f0000
	s_code_end                                                 // 0000000035f8: bf9f0000
	s_code_end                                                 // 0000000035fc: bf9f0000
