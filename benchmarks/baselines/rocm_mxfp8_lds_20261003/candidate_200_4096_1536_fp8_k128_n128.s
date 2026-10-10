
/tmp/tmpqkqoibxb.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f>:
	s_clause 0x4                                               // 000000001b00: bf850004
	s_load_b64 s[16:17], s[0:1], 0x8                           // 000000001b04: f4002400 f8000008
	s_load_b64 s[14:15], s[0:1], 0x30                          // 000000001b0c: f4002380 f8000030
	s_load_b64 s[8:9], s[0:1], 0x58                            // 000000001b14: f4002200 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b1c: f4002700 f8000080
	s_load_b128 s[24:27], s[0:1], 0xc8                         // 000000001b24: f4004600 f80000c8
	v_lshrrev_b32_e32 v12, 3, v0                               // 000000001b2c: 32180083
	s_mov_b32 s12, ttmp9                                       // 000000001b30: be8c0075
	s_mov_b32 s2, ttmp7                                        // 000000001b34: be820073
	s_ashr_i32 s13, ttmp9, 31                                  // 000000001b38: 860d9f75
	s_ashr_i32 s3, ttmp7, 31                                   // 000000001b3c: 86039f73
	v_or_b32_e32 v2, 0x60, v12                                 // 000000001b40: 380418ff 00000060
	s_lshl_b64 s[6:7], s[2:3], 7                               // 000000001b48: 84868702
	s_lshl_b64 s[22:23], s[12:13], 7                           // 000000001b4c: 8496870c
	v_dual_mov_b32 v3, s7 :: v_dual_mov_b32 v6, s7             // 000000001b50: ca100007 03060007
	s_delay_alu instid0(valu_dep_2)                            // 000000001b58: bf870002
	v_or_b32_e32 v15, s22, v2                                  // 000000001b5c: 381e0416
	v_mul_u32_u24_e32 v4, 0x90, v2                             // 000000001b60: 160804ff 00000090
	v_or_b32_e32 v2, s6, v2                                    // 000000001b68: 38040406
	v_lshrrev_b32_e32 v9, 1, v0                                // 000000001b6c: 32120081
	v_dual_mov_b32 v83, 0 :: v_dual_and_b32 v38, 15, v0        // 000000001b70: ca240080 5326008f
	v_or_b32_e32 v17, 32, v12                                  // 000000001b78: 382218a0
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b7c: f4002100 f80000d8
	v_or_b32_e32 v5, s6, v12                                   // 000000001b84: 380a1806
	s_wait_kmcnt 0x0                                           // 000000001b88: bfc70000
	s_add_nc_u64 s[18:19], s[24:25], -1                        // 000000001b8c: a992c118
	v_or_b32_e32 v13, 64, v12                                  // 000000001b90: 381a18c0
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[2:3]                  // 000000001b94: 7cb80412
	v_lshlrev_b32_e32 v1, 1, v0                                // 000000001b98: 30020081
	v_lshlrev_b32_e32 v0, 4, v0                                // 000000001b9c: 30000084
	v_cmp_gt_u64_e64 s2, s[18:19], v[5:6]                      // 000000001ba0: d45c0002 02020a12
	v_mul_u32_u24_e32 v6, 0x90, v13                            // 000000001ba8: 160c1aff 00000090
	v_or_b32_e32 v7, s6, v13                                   // 000000001bb0: 380e1a06
	v_cndmask_b32_e32 v21, s18, v2, vcc_lo                     // 000000001bb4: 022a0412
	v_and_b32_e32 v0, 0x70, v0                                 // 000000001bb8: 360000ff 00000070
	v_dual_cndmask_b32 v20, s19, v3 :: v_dual_and_b32 v39, 64, v1// 000000001bc0: ca640613 142602c0
	v_or_b32_e32 v3, s6, v17                                   // 000000001bc8: 38062206
	s_delay_alu instid0(valu_dep_3)                            // 000000001bcc: bf870003
	v_dual_mov_b32 v2, s7 :: v_dual_add_nc_u32 v89, v6, v0     // 000000001bd0: ca200007 02580106
	v_dual_mov_b32 v8, s7 :: v_dual_add_nc_u32 v87, v4, v0     // 000000001bd8: ca200007 08560104
	v_mov_b32_e32 v4, s7                                       // 000000001be0: 7e080207
	v_or_b32_e32 v16, s22, v13                                 // 000000001be4: 38201a16
	v_or_b32_e32 v19, s22, v12                                 // 000000001be8: 38261816
	v_cndmask_b32_e64 v5, s18, v5, s2                          // 000000001bec: d5010005 000a0a12
	v_cndmask_b32_e64 v13, s19, v2, s2                         // 000000001bf4: d501000d 000a0413
	v_cmp_gt_u64_e32 vcc_lo, s[18:19], v[3:4]                  // 000000001bfc: 7cb80612
	v_mul_u32_u24_e32 v12, 0x90, v12                           // 000000001c00: 161818ff 00000090
	v_cmp_gt_u64_e64 s3, s[18:19], v[7:8]                      // 000000001c08: d45c0003 02020e12
	v_mul_lo_u32 v23, v5, s5                                   // 000000001c10: d72c0017 02000b05
	v_mul_lo_u32 v13, v13, s4                                  // 000000001c18: d72c000d 0200090d
	v_or_b32_e32 v18, s22, v17                                 // 000000001c20: 38242216
	s_wait_alu depctr_va_vcc(0)                                // 000000001c24: bf88ff9d
	v_cndmask_b32_e32 v4, s19, v4, vcc_lo                      // 000000001c28: 02080813
	v_dual_cndmask_b32 v22, s18, v3 :: v_dual_add_nc_u32 v91, v12, v0// 000000001c2c: ca600612 165a010c
	v_mad_co_u64_u32 v[2:3], null, v5, s4, s[16:17]            // 000000001c34: d6fe7c02 00400905
	s_wait_alu depctr_va_sdst(0)                               // 000000001c3c: bf88f19f
	v_cndmask_b32_e64 v6, s19, v8, s3                          // 000000001c40: d5010006 000e1013
	v_mul_lo_u32 v25, v4, s4                                   // 000000001c48: d72c0019 02000904
	v_mul_lo_u32 v24, v22, s5                                  // 000000001c50: d72c0018 02000b16
	v_mad_co_u64_u32 v[4:5], null, v22, s4, s[16:17]           // 000000001c58: d6fe7c04 00400916
	v_cndmask_b32_e64 v7, s18, v7, s3                          // 000000001c60: d5010007 000e0e12
	v_mul_u32_u24_e32 v17, 0x90, v17                           // 000000001c68: 162222ff 00000090
	s_mul_i32 s2, s4, s23                                      // 000000001c70: 96021704
	v_add3_u32 v3, v13, v3, v23                                // 000000001c74: d6550003 045e070d
	v_mul_lo_u32 v13, v6, s4                                   // 000000001c7c: d72c000d 02000906
	v_mul_lo_u32 v12, v7, s5                                   // 000000001c84: d72c000c 02000b07
	v_add_co_u32 v92, vcc_lo, v2, v0                           // 000000001c8c: d7006a5c 02020102
	v_add3_u32 v8, v25, v5, v24                                // 000000001c94: d6550008 04620b19
	v_mad_co_u64_u32 v[5:6], null, v7, s4, s[16:17]            // 000000001c9c: d6fe7c05 00400907
	s_wait_alu depctr_va_vcc(0)                                // 000000001ca4: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, 0, v3, vcc_lo               // 000000001ca8: d5207c5d 01aa0680
	v_add_co_u32 v94, vcc_lo, v4, v0                           // 000000001cb0: d7006a5e 02020104
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb8: bf88ff9d
	v_add_co_ci_u32_e64 v95, null, 0, v8, vcc_lo               // 000000001cbc: d5207c5f 01aa1080
	v_mul_lo_u32 v8, v21, s5                                   // 000000001cc4: d72c0008 02000b15
	v_add3_u32 v4, v13, v6, v12                                // 000000001ccc: d6550004 04320d0d
	v_mul_lo_u32 v12, v20, s4                                  // 000000001cd4: d72c000c 02000914
	v_mad_co_u64_u32 v[2:3], null, v21, s4, s[16:17]           // 000000001cdc: d6fe7c02 00400915
	v_add_co_u32 v97, vcc_lo, v5, v0                           // 000000001ce4: d7006a61 02020105
	s_wait_alu depctr_va_vcc(0)                                // 000000001cec: bf88ff9d
	v_add_co_ci_u32_e64 v98, null, 0, v4, vcc_lo               // 000000001cf0: d5207c62 01aa0880
	v_mul_lo_u32 v13, s5, v19                                  // 000000001cf8: d72c000d 02022605
	v_mad_co_u64_u32 v[4:5], null, s4, v19, s[14:15]           // 000000001d00: d6fe7c04 003a2604
	v_dual_mov_b32 v33, s23 :: v_dual_add_nc_u32 v90, v17, v0  // 000000001d08: ca200017 215a0111
	v_add3_u32 v3, v12, v3, v8                                 // 000000001d10: d6550003 0422070c
	v_mul_lo_u32 v17, s5, v18                                  // 000000001d18: d72c0011 02022405
	v_mad_co_u64_u32 v[6:7], null, s4, v18, s[14:15]           // 000000001d20: d6fe7c06 003a2404
	v_add_co_u32 v99, vcc_lo, v2, v0                           // 000000001d28: d7006a63 02020102
	s_wait_alu depctr_va_vcc(0)                                // 000000001d30: bf88ff9d
	v_add_co_ci_u32_e64 v100, null, 0, v3, vcc_lo              // 000000001d34: d5207c64 01aa0680
	v_mul_lo_u32 v8, s5, v16                                   // 000000001d3c: d72c0008 02022005
	v_mad_co_u64_u32 v[2:3], null, s4, v16, s[14:15]           // 000000001d44: d6fe7c02 003a2004
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d4c: bf88ff9e
	v_add3_u32 v5, v13, v5, s2                                 // 000000001d50: d6550005 000a0b0d
	v_dual_mov_b32 v35, s7 :: v_dual_and_b32 v10, 0x60, v9     // 000000001d58: ca240007 230a12ff 00000060
	v_add3_u32 v7, v17, v7, s2                                 // 000000001d64: d6550007 000a0f11
	v_add_co_u32 v101, vcc_lo, v4, v0                          // 000000001d6c: d7006a65 02020104
	s_wait_alu depctr_va_vcc(0)                                // 000000001d74: bf88ff9d
	v_add_co_ci_u32_e64 v102, null, 0, v5, vcc_lo              // 000000001d78: d5207c66 01aa0a80
	v_add3_u32 v5, v8, v3, s2                                  // 000000001d80: d6550005 000a0708
	v_mad_co_u64_u32 v[3:4], null, s4, v15, s[14:15]           // 000000001d88: d6fe7c03 003a1e04
	v_add_co_u32 v103, vcc_lo, v6, v0                          // 000000001d90: d7006a67 02020106
	v_or_b32_e32 v11, 16, v10                                  // 000000001d98: 38161490
	s_wait_alu depctr_va_vcc(0)                                // 000000001d9c: bf88ff9d
	v_add_co_ci_u32_e64 v104, null, 0, v7, vcc_lo              // 000000001da0: d5207c68 01aa0e80
	v_add_co_u32 v105, vcc_lo, v2, v0                          // 000000001da8: d7006a69 02020102
	v_or_b32_e32 v2, v10, v38                                  // 000000001db0: 38044d0a
	v_or_b32_e32 v7, v39, v38                                  // 000000001db4: 380e4d27
	v_mul_lo_u32 v6, s5, v15                                   // 000000001db8: d72c0006 02021e05
	v_or_b32_e32 v14, s6, v10                                  // 000000001dc0: 381c1406
	v_and_b32_e32 v10, 8, v9                                   // 000000001dc4: 36141288
	v_or_b32_e32 v1, s6, v11                                   // 000000001dc8: 38021606
	s_wait_alu depctr_va_vcc(0)                                // 000000001dcc: bf88ff9d
	v_add_co_ci_u32_e64 v106, null, 0, v5, vcc_lo              // 000000001dd0: d5207c6a 01aa0a80
	v_or_b32_e32 v5, v11, v38                                  // 000000001dd8: 380a4d0b
	v_mul_u32_u24_e32 v2, 0x90, v2                             // 000000001ddc: 160404ff 00000090
	v_add_co_u32 v107, vcc_lo, v3, v0                          // 000000001de4: d7006a6b 02020103
	v_or_b32_e32 v0, 16, v7                                    // 000000001dec: 38000e90
	v_or_b32_e32 v11, 32, v7                                   // 000000001df0: 38160ea0
	v_add3_u32 v4, v6, v4, s2                                  // 000000001df4: d6550004 000a0906
	v_or_b32_e32 v34, v14, v10                                 // 000000001dfc: 3844150e
	v_or_b32_e32 v109, v2, v10                                 // 000000001e00: 38da1502
	v_mul_u32_u24_e32 v2, 0x90, v0                             // 000000001e04: 160400ff 00000090
	v_or_b32_e32 v12, 48, v7                                   // 000000001e0c: 38180eb0
	v_mul_u32_u24_e32 v3, 0x90, v11                            // 000000001e10: 160616ff 00000090
	v_or_b32_e32 v13, 1, v10                                   // 000000001e18: 381a1481
	s_add_nc_u64 s[2:3], s[26:27], 0x7f                        // 000000001e1c: a982ff1a 0000007f
	s_wait_alu depctr_va_vcc(0)                                // 000000001e24: bf88ff9d
	v_add_co_ci_u32_e64 v108, null, 0, v4, vcc_lo              // 000000001e28: d5207c6c 01aa0880
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e30: bf88ff9e
	s_lshr_b64 s[34:35], s[2:3], 7                             // 000000001e34: 85a28702
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[34:35]                // 000000001e38: 7ca84418
	s_mov_b32 s10, ttmp9                                       // 000000001e3c: be8a0075
	v_mul_u32_u24_e32 v5, 0x90, v5                             // 000000001e40: 160a0aff 00000090
	s_add_nc_u64 s[2:3], s[34:35], -1                          // 000000001e48: a982c122
	s_and_b32 s11, s13, 0x1ffffff                              // 000000001e4c: 8b0bff0d 01ffffff
	v_mul_u32_u24_e32 v4, 0x90, v12                            // 000000001e54: 160818ff 00000090
	v_or_b32_e32 v113, v2, v10                                 // 000000001e5c: 38e21502
	v_or_b32_e32 v114, v3, v10                                 // 000000001e60: 38e41503
	v_or_b32_e32 v2, v13, v14                                  // 000000001e64: 38041d0d
	v_mov_b32_e32 v3, s7                                       // 000000001e68: 7e060207
	s_lshr_b64 s[30:31], s[4:5], 7                             // 000000001e6c: 859e8704
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e70: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[10:11], s[2:3]                      // 000000001e74: d4590004 0200040a
	v_or_b32_e32 v110, v5, v10                                 // 000000001e7c: 38dc1505
	v_or_b32_e32 v116, v4, v10                                 // 000000001e80: 38e81504
	s_wait_alu depctr_va_vcc(0)                                // 000000001e84: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v35 :: v_dual_cndmask_b32 v5, 0, v34// 000000001e88: ca524680 04044480
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[2:3]                  // 000000001e90: 7ca80418
	v_mul_u32_u24_e32 v6, 0x90, v7                             // 000000001e94: 160c0eff 00000090
	s_and_b32 s4, s4, exec_lo                                  // 000000001e9c: 8b047e04
	s_cselect_b32 s11, s11, s3                                 // 000000001ea0: 980b030b
	s_cselect_b32 s10, ttmp9, s2                               // 000000001ea4: 980a0275
	s_lshr_b32 s12, s5, 7                                      // 000000001ea8: 850c8705
	v_or_b32_e32 v111, v6, v10                                 // 000000001eac: 38de1506
	v_mul_lo_u32 v8, s12, v5                                   // 000000001eb0: d72c0008 02020a0c
	s_wait_alu depctr_va_vcc(0)                                // 000000001eb8: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v3, vcc_lo                        // 000000001ebc: 020c0680
	v_cndmask_b32_e32 v16, 0, v2, vcc_lo                       // 000000001ec0: 02200480
	v_mad_co_u64_u32 v[2:3], null, s30, v5, 0                  // 000000001ec4: d6fe7c02 02020a1e
	v_mov_b32_e32 v5, s7                                       // 000000001ecc: 7e0a0207
	v_or_b32_e32 v15, 2, v10                                   // 000000001ed0: 381e1482
	v_mul_lo_u32 v9, s30, v4                                   // 000000001ed4: d72c0009 0202081e
	v_or_b32_e32 v19, 3, v10                                   // 000000001edc: 38261483
	v_or_b32_e32 v32, s22, v7                                  // 000000001ee0: 38400e16
	v_or_b32_e32 v36, v1, v10                                  // 000000001ee4: 38481501
	v_or_b32_e32 v4, v15, v14                                  // 000000001ee8: 38081d0f
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v86, 0            // 000000001eec: ca100080 95560080
	s_delay_alu instid0(valu_dep_4) | instskip(skip_1) | instid1(valu_dep_4)// 000000001ef4: bf870224
	v_cmp_gt_i64_e64 s4, s[26:27], v[32:33]                    // 000000001ef8: d4540004 0202401a
	v_add3_u32 v3, v3, v9, v8                                  // 000000001f00: d6550003 04221303
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000001f08: 7ca80818
	v_or_b32_e32 v8, v19, v14                                  // 000000001f0c: 38101d13
	v_mov_b32_e32 v9, s7                                       // 000000001f10: 7e120207
	v_mul_lo_u32 v18, s30, v6                                  // 000000001f14: d72c0012 02020c1e
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000001f1c: 3e040482
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v84, 0            // 000000001f20: ca100080 8f540080
	s_wait_alu depctr_va_vcc(0)                                // 000000001f28: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_cndmask_b32 v4, 0, v4// 000000001f2c: ca520a80 05040880
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[8:9]                  // 000000001f34: 7ca81018
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v82, 0            // 000000001f38: ca100080 89520080
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v80, 0            // 000000001f40: ca100080 81500080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v70, 0             // 000000001f48: ca100080 55460080
	s_wait_alu depctr_va_vcc(0)                                // 000000001f50: bf88ff9d
	v_cndmask_b32_e32 v21, 0, v8, vcc_lo                       // 000000001f54: 022a1080
	v_mul_lo_u32 v17, s12, v16                                 // 000000001f58: d72c0011 0202200c
	v_mad_co_u64_u32 v[6:7], null, s30, v16, 0                 // 000000001f60: d6fe7c06 0202201e
	v_mul_lo_u32 v16, s12, v4                                  // 000000001f68: d72c0010 0202080c
	v_cndmask_b32_e32 v20, 0, v9, vcc_lo                       // 000000001f70: 02281280
	v_add_co_u32 v121, vcc_lo, s8, v2                          // 000000001f74: d7006a79 02020408
	s_wait_alu depctr_va_vcc(0)                                // 000000001f7c: bf88ff9d
	v_add_co_ci_u32_e64 v122, null, s9, v3, vcc_lo             // 000000001f80: d5207c7a 01aa0609
	v_mov_b32_e32 v37, s7                                      // 000000001f88: 7e4a0207
	v_add3_u32 v7, v7, v18, v17                                // 000000001f8c: d6550007 04462507
	v_or_b32_e32 v17, 4, v10                                   // 000000001f94: 38221484
	v_mul_lo_u32 v18, s30, v5                                  // 000000001f98: d72c0012 02020a1e
	v_mad_co_u64_u32 v[4:5], null, s30, v4, 0                  // 000000001fa0: d6fe7c04 0202081e
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v68, 0             // 000000001fa8: ca100080 51440080
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 000000001fb0: bf870244
	v_or_b32_e32 v8, v17, v14                                  // 000000001fb4: 38101d11
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001fb8: 3e040c82
	v_mad_co_u64_u32 v[6:7], null, s30, v21, 0                 // 000000001fbc: d6fe7c06 02022a1e
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v66, 0             // 000000001fc4: ca100080 47420080
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[8:9]                  // 000000001fcc: 7ca81018
	v_add3_u32 v5, v5, v18, v16                                // 000000001fd0: d6550005 04422505
	v_or_b32_e32 v18, 5, v10                                   // 000000001fd8: 38241485
	v_mul_lo_u32 v16, s12, v21                                 // 000000001fdc: d72c0010 02022a0c
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v64, 0             // 000000001fe4: ca100080 45400080
	s_wait_alu depctr_va_vcc(0)                                // 000000001fec: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v8, vcc_lo                       // 000000001ff0: 022c1080
	v_or_b32_e32 v8, v18, v14                                  // 000000001ff4: 38101d12
	v_cndmask_b32_e32 v21, 0, v9, vcc_lo                       // 000000001ff8: 022a1280
	v_add_co_u32 v124, vcc_lo, s8, v2                          // 000000001ffc: d7006a7c 02020408
	s_wait_alu depctr_va_vcc(0)                                // 000000002004: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s9, v3, vcc_lo             // 000000002008: d5207c7d 01aa0609
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[8:9]                  // 000000002010: 7ca81018
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 000000002014: 3e040882
	v_mul_lo_u32 v21, s30, v21                                 // 000000002018: d72c0015 02022a1e
	v_mad_co_u64_u32 v[4:5], null, s30, v22, 0                 // 000000002020: d6fe7c04 02022c1e
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v54, 0             // 000000002028: ca100080 43360080
	s_wait_alu depctr_va_vcc(0)                                // 000000002030: bf88ff9d
	v_cndmask_b32_e32 v23, 0, v8, vcc_lo                       // 000000002034: 022e1080
	v_mul_lo_u32 v20, s30, v20                                 // 000000002038: d72c0014 0202281e
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v65, 0            // 000000002040: ca100080 86400080
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v55, 0             // 000000002048: ca100080 34360080
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v53, 0             // 000000002050: ca100080 32340080
	v_mov_b32_e32 v48, 0                                       // 000000002058: 7e600280
	v_add3_u32 v7, v7, v20, v16                                // 00000000205c: d6550007 04422907
	v_or_b32_e32 v20, 6, v10                                   // 000000002064: 38281486
	v_mul_lo_u32 v16, s12, v22                                 // 000000002068: d72c0010 02022c0c
	v_cndmask_b32_e32 v22, 0, v9, vcc_lo                       // 000000002070: 022c1280
	v_add_co_u32 v127, vcc_lo, s8, v2                          // 000000002074: d7006a7f 02020408
	s_delay_alu instid0(valu_dep_4)                            // 00000000207c: bf870004
	v_or_b32_e32 v8, v20, v14                                  // 000000002080: 38101d14
	s_wait_alu depctr_va_vcc(0)                                // 000000002084: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, s9, v3, vcc_lo             // 000000002088: d5207c80 01aa0609
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000002090: 3e040c82
	v_add3_u32 v5, v5, v21, v16                                // 000000002094: d6550005 04422b05
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[8:9]                  // 00000000209c: 7ca81018
	v_or_b32_e32 v21, 7, v10                                   // 0000000020a0: 382a1487
	v_mul_lo_u32 v16, s12, v23                                 // 0000000020a4: d72c0010 02022e0c
	v_mad_co_u64_u32 v[6:7], null, s30, v23, 0                 // 0000000020ac: d6fe7c06 02022e1e
	v_mov_b32_e32 v146, 0                                      // 0000000020b4: 7f240280
	s_wait_alu depctr_va_vcc(0)                                // 0000000020b8: bf88ff9d
	v_dual_mov_b32 v142, 0 :: v_dual_cndmask_b32 v23, 0, v9    // 0000000020bc: ca120080 8e161280
	v_cndmask_b32_e32 v24, 0, v8, vcc_lo                       // 0000000020c4: 02301080
	v_or_b32_e32 v8, v21, v14                                  // 0000000020c8: 38101d15
	v_add_co_u32 v130, vcc_lo, s8, v2                          // 0000000020cc: d7006a82 02020408
	s_wait_alu depctr_va_vcc(0)                                // 0000000020d4: bf88ff9d
	v_add_co_ci_u32_e64 v131, null, s9, v3, vcc_lo             // 0000000020d8: d5207c83 01aa0609
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 0000000020e0: 3e040882
	v_mul_lo_u32 v14, s12, v24                                 // 0000000020e4: d72c000e 0202300c
	v_mul_lo_u32 v23, s30, v23                                 // 0000000020ec: d72c0017 02022e1e
	v_mad_co_u64_u32 v[4:5], null, s30, v24, 0                 // 0000000020f4: d6fe7c04 0202301e
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[8:9]                  // 0000000020fc: 7ca81018
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v51, 0             // 000000002100: ca100080 60320080
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v49, 0            // 000000002108: ca100080 7e300080
	s_wait_alu depctr_va_vcc(0)                                // 000000002110: bf88ff9d
	v_dual_mov_b32 v120, 0 :: v_dual_cndmask_b32 v9, 0, v9     // 000000002114: ca120080 78081280
	v_mul_lo_u32 v22, s30, v22                                 // 00000000211c: d72c0016 02022c1e
	v_add3_u32 v5, v5, v23, v14                                // 000000002124: d6550005 043a2f05
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo                        // 00000000212c: 02101080
	v_add_co_u32 v132, vcc_lo, s8, v2                          // 000000002130: d7006a84 02020408
	s_wait_alu depctr_va_vcc(0)                                // 000000002138: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, s9, v3, vcc_lo             // 00000000213c: d5207c85 01aa0609
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000002144: 3e080882
	v_add3_u32 v7, v7, v22, v16                                // 000000002148: d6550007 04422d07
	v_mul_lo_u32 v9, s30, v9                                   // 000000002150: d72c0009 0202121e
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v123, 0            // 000000002158: ca100080 587a0080
	v_mov_b32_e32 v118, 0                                      // 000000002160: 7eec0280
	v_add_co_u32 v138, s2, s8, v4                              // 000000002164: d700028a 02020808
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 00000000216c: 3e040c82
	s_wait_alu depctr_va_sdst(0)                               // 000000002170: bf88f19f
	v_add_co_ci_u32_e64 v139, null, s9, v5, s2                 // 000000002174: d5207c8b 000a0a09
	v_mov_b32_e32 v5, s23                                      // 00000000217c: 7e0a0217
	v_mul_lo_u32 v14, s12, v8                                  // 000000002180: d72c000e 0202100c
	v_mad_co_u64_u32 v[6:7], null, s30, v8, 0                  // 000000002188: d6fe7c06 0202101e
	v_add_co_u32 v135, vcc_lo, s8, v2                          // 000000002190: d7006a87 02020408
	s_wait_alu depctr_va_vcc(0)                                // 000000002198: bf88ff9d
	v_add_co_ci_u32_e64 v136, null, s9, v3, vcc_lo             // 00000000219c: d5207c88 01aa0609
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[36:37]                // 0000000021a4: 7ca84818
	v_or_b32_e32 v4, s22, v0                                   // 0000000021a8: 38080016
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v112, 0           // 0000000021ac: ca100080 77700080
	v_add3_u32 v7, v7, v9, v14                                 // 0000000021b4: d6550007 043a1307
	s_wait_alu depctr_va_vcc(0)                                // 0000000021bc: bf88ff9d
	v_dual_mov_b32 v9, s23 :: v_dual_cndmask_b32 v0, 0, v37    // 0000000021c0: ca120017 09004a80
	v_cndmask_b32_e32 v8, 0, v36, vcc_lo                       // 0000000021c8: 02104880
	v_cmp_gt_i64_e64 s3, s[26:27], v[4:5]                      // 0000000021cc: d4540003 0202081a
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 0000000021d4: 3e040c82
	v_mov_b32_e32 v7, s7                                       // 0000000021d8: 7e0e0207
	v_or_b32_e32 v6, v1, v13                                   // 0000000021dc: 380c1b01
	v_mul_lo_u32 v0, s30, v0                                   // 0000000021e0: d72c0000 0202001e
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v78, 0            // 0000000021e8: ca100080 754e0080
	v_add_co_u32 v140, vcc_lo, s8, v2                          // 0000000021f0: d7006a8c 02020408
	s_wait_alu depctr_va_vcc(0)                                // 0000000021f8: bf88ff9d
	v_add_co_ci_u32_e64 v141, null, s9, v3, vcc_lo             // 0000000021fc: d5207c8d 01aa0609
	v_mov_b32_e32 v3, s23                                      // 000000002204: 7e060217
	v_mul_lo_u32 v10, s12, v8                                  // 000000002208: d72c000a 0202100c
	v_mad_co_u64_u32 v[4:5], null, s30, v8, 0                  // 000000002210: d6fe7c04 0202101e
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[6:7]                  // 000000002218: 7ca80c18
	v_or_b32_e32 v2, s22, v11                                  // 00000000221c: 38041616
	v_or_b32_e32 v8, s22, v12                                  // 000000002220: 38101816
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v76, 0            // 000000002224: ca100080 734c0080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v74, 0             // 00000000222c: ca100080 4f4a0080
	s_wait_alu depctr_va_vcc(0)                                // 000000002234: bf88ff9d
	v_cndmask_b32_e32 v13, 0, v6, vcc_lo                       // 000000002238: 021a0c80
	v_or_b32_e32 v6, v1, v15                                   // 00000000223c: 380c1f01
	v_add3_u32 v5, v5, v0, v10                                 // 000000002240: d6550005 042a0105
	v_cndmask_b32_e32 v11, 0, v7, vcc_lo                       // 000000002248: 02160e80
	v_cmp_gt_i64_e64 s2, s[26:27], v[2:3]                      // 00000000224c: d4540002 0202041a
	v_mul_lo_u32 v0, s12, v13                                  // 000000002254: d72c0000 02021a0c
	v_cmp_gt_i64_e64 s5, s[24:25], v[6:7]                      // 00000000225c: d4540005 02020c18
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 000000002264: 3e040882
	v_mov_b32_e32 v5, s7                                       // 000000002268: 7e0a0207
	v_or_b32_e32 v4, v1, v19                                   // 00000000226c: 38082701
	v_mul_lo_u32 v12, s30, v11                                 // 000000002270: d72c000c 0202161e
	v_mad_co_u64_u32 v[10:11], null, s30, v13, 0               // 000000002278: d6fe7c0a 02021a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002280: bf88f19f
	v_cndmask_b32_e64 v7, 0, v7, s5                            // 000000002284: d5010007 00160e80
	v_cndmask_b32_e64 v6, 0, v6, s5                            // 00000000228c: d5010006 00160c80
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 000000002294: d4540005 02020818
	v_cmp_gt_i64_e32 vcc_lo, s[26:27], v[8:9]                  // 00000000229c: 7ca8101a
	v_add_co_u32 v144, s6, s8, v2                              // 0000000022a0: d7000690 02020408
	v_mul_lo_u32 v8, s30, v7                                   // 0000000022a8: d72c0008 02020e1e
	v_add3_u32 v11, v11, v12, v0                               // 0000000022b0: d655000b 0402190b
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b8: bf88f19f
	v_cndmask_b32_e64 v12, 0, v4, s5                           // 0000000022bc: d501000c 00160880
	v_or_b32_e32 v4, v1, v17                                   // 0000000022c4: 38082301
	v_mul_lo_u32 v0, s12, v6                                   // 0000000022c8: d72c0000 02020c0c
	v_mad_co_u64_u32 v[6:7], null, s30, v6, 0                  // 0000000022d0: d6fe7c06 02020c1e
	v_cndmask_b32_e64 v9, 0, v5, s5                            // 0000000022d8: d5010009 00160a80
	v_add_co_ci_u32_e64 v145, null, s9, v3, s6                 // 0000000022e0: d5207c91 001a0609
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 0000000022e8: d4540005 02020818
	v_lshlrev_b64_e32 v[2:3], 2, v[10:11]                      // 0000000022f0: 3e041482
	s_delay_alu instid0(valu_dep_4)                            // 0000000022f4: bf870004
	v_mul_lo_u32 v10, s30, v9                                  // 0000000022f8: d72c000a 0202121e
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v72, 0             // 000000002300: ca100080 4d480080
	v_add3_u32 v7, v7, v8, v0                                  // 000000002308: d6550007 04021107
	v_mul_lo_u32 v0, s12, v12                                  // 000000002310: d72c0000 0202180c
	v_mad_co_u64_u32 v[8:9], null, s30, v12, 0                 // 000000002318: d6fe7c08 0202181e
	s_wait_alu depctr_va_sdst(0)                               // 000000002320: bf88f19f
	v_cndmask_b32_e64 v12, 0, v4, s5                           // 000000002324: d501000c 00160880
	v_or_b32_e32 v4, v1, v18                                   // 00000000232c: 38082501
	v_add_co_u32 v147, s6, s8, v2                              // 000000002330: d7000693 02020408
	v_cndmask_b32_e64 v11, 0, v5, s5                           // 000000002338: d501000b 00160a80
	s_wait_alu depctr_va_sdst(0)                               // 000000002340: bf88f19f
	v_add_co_ci_u32_e64 v148, null, s9, v3, s6                 // 000000002344: d5207c94 001a0609
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 00000000234c: 3e040c82
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 000000002350: d4540005 02020818
	v_add3_u32 v9, v9, v10, v0                                 // 000000002358: d6550009 04021509
	v_mul_lo_u32 v0, s12, v12                                  // 000000002360: d72c0000 0202180c
	v_mul_lo_u32 v13, s30, v11                                 // 000000002368: d72c000d 0202161e
	v_mad_co_u64_u32 v[6:7], null, s30, v12, 0                 // 000000002370: d6fe7c06 0202181e
	v_mov_b32_e32 v11, s7                                      // 000000002378: 7e160207
	v_or_b32_e32 v10, v1, v20                                  // 00000000237c: 38142901
	v_add_co_u32 v150, s6, s8, v2                              // 000000002380: d7000696 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000002388: bf88f19f
	v_add_co_ci_u32_e64 v151, null, s9, v3, s6                 // 00000000238c: d5207c97 001a0609
	v_lshlrev_b64_e32 v[2:3], 2, v[8:9]                        // 000000002394: 3e041082
	v_cndmask_b32_e64 v8, 0, v4, s5                            // 000000002398: d5010008 00160880
	v_or_b32_e32 v4, v1, v21                                   // 0000000023a0: 38082b01
	v_cmp_gt_i64_e64 s6, s[24:25], v[10:11]                    // 0000000023a4: d4540006 02021418
	v_add3_u32 v7, v7, v13, v0                                 // 0000000023ac: d6550007 04021b07
	v_cndmask_b32_e64 v0, 0, v5, s5                            // 0000000023b4: d5010000 00160a80
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v62, 0             // 0000000023bc: ca100080 4b3e0080
	v_cmp_gt_i64_e64 s5, s[24:25], v[4:5]                      // 0000000023c4: d4540005 02020818
	s_wait_alu depctr_va_sdst(0)                               // 0000000023cc: bf88f19f
	v_cndmask_b32_e64 v9, 0, v11, s6                           // 0000000023d0: d5010009 001a1680
	v_mul_lo_u32 v11, s12, v8                                  // 0000000023d8: d72c000b 0202100c
	v_mul_lo_u32 v12, s30, v0                                  // 0000000023e0: d72c000c 0202001e
	v_mad_co_u64_u32 v[0:1], null, s30, v8, 0                  // 0000000023e8: d6fe7c00 0202101e
	v_cndmask_b32_e64 v10, 0, v10, s6                          // 0000000023f0: d501000a 001a1480
	v_cndmask_b32_e64 v5, 0, v5, s5                            // 0000000023f8: d5010005 00160a80
	v_cndmask_b32_e64 v4, 0, v4, s5                            // 000000002400: d5010004 00160880
	v_mul_lo_u32 v14, s30, v9                                  // 000000002408: d72c000e 0202121e
	v_add_co_u32 v152, s5, s8, v2                              // 000000002410: d7000598 02020408
	v_mul_lo_u32 v13, s12, v10                                 // 000000002418: d72c000d 0202140c
	v_mad_co_u64_u32 v[8:9], null, s30, v10, 0                 // 000000002420: d6fe7c08 0202141e
	s_wait_alu depctr_va_sdst(0)                               // 000000002428: bf88f19f
	v_add_co_ci_u32_e64 v153, null, s9, v3, s5                 // 00000000242c: d5207c99 00160609
	v_add3_u32 v1, v1, v12, v11                                // 000000002434: d6550001 042e1901
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 00000000243c: 3e040c82
	v_mul_lo_u32 v6, s12, v4                                   // 000000002440: d72c0006 0202080c
	v_mul_lo_u32 v7, s30, v5                                   // 000000002448: d72c0007 02020a1e
	v_mad_co_u64_u32 v[4:5], null, s30, v4, 0                  // 000000002450: d6fe7c04 0202081e
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000002458: 3e000082
	v_add3_u32 v9, v9, v14, v13                                // 00000000245c: d6550009 04361d09
	v_add_co_u32 v154, s5, s8, v2                              // 000000002464: d700059a 02020408
	s_wait_alu depctr_va_sdst(0)                               // 00000000246c: bf88f19f
	v_add_co_ci_u32_e64 v155, null, s9, v3, s5                 // 000000002470: d5207c9b 00160609
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000002478: bf8701d3
	v_lshlrev_b64_e32 v[2:3], 2, v[8:9]                        // 00000000247c: 3e041082
	v_add3_u32 v5, v5, v7, v6                                  // 000000002480: d6550005 041a0f05
	v_add_co_u32 v156, s5, s8, v0                              // 000000002488: d700059c 02020008
	s_wait_alu depctr_va_sdst(0)                               // 000000002490: bf88f19f
	v_add_co_ci_u32_e64 v157, null, s9, v1, s5                 // 000000002494: d5207c9d 00160209
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 00000000249c: 3e000882
	v_add_co_u32 v158, s5, s8, v2                              // 0000000024a0: d700059e 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000024a8: bf88f19f
	v_add_co_ci_u32_e64 v159, null, s9, v3, s5                 // 0000000024ac: d5207c9f 00160609
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v60, 0             // 0000000024b4: ca100080 493c0080
	s_delay_alu instid0(valu_dep_4)                            // 0000000024bc: bf870004
	v_add_co_u32 v160, s5, s8, v0                              // 0000000024c0: d70005a0 02020008
	s_wait_alu depctr_va_sdst(0)                               // 0000000024c8: bf88f19f
	v_add_co_ci_u32_e64 v161, null, s9, v1, s5                 // 0000000024cc: d5207ca1 00160209
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v58, 0             // 0000000024d4: ca100080 3f3a0080
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v56, 0             // 0000000024dc: ca100080 3d380080
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v46, 0             // 0000000024e4: ca100080 3b2e0080
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v44, 0             // 0000000024ec: ca100080 392c0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v42, 0             // 0000000024f4: ca100080 2f2a0080
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v40, 0             // 0000000024fc: ca100080 2d280080
	v_mov_b32_e32 v43, 0                                       // 000000002504: 7e560280
	v_mov_b32_e32 v41, 0                                       // 000000002508: 7e520280
	s_mov_b64 s[36:37], 0                                      // 00000000250c: bea40180
	s_wait_alu depctr_sa_sdst(0)                               // 000000002510: bf88ff9e
	s_lshl_b64 s[38:39], s[10:11], 2                           // 000000002514: 84a6820a
	s_lshl_b64 s[12:13], s[36:37], 7                           // 000000002518: 848c8724
	s_lshl_b64 s[20:21], s[36:37], 2                           // 00000000251c: 84948224
	s_wait_alu depctr_sa_sdst(0)                               // 000000002520: bf88ff9e
	v_add_co_u32 v6, s8, v99, s12                              // 000000002524: d7000806 02001963
	v_add_co_u32 v8, s9, v101, s12                             // 00000000252c: d7000908 02001965
	v_add_co_u32 v4, s7, v97, s12                              // 000000002534: d7000704 02001961
	s_wait_alu depctr_va_sdst(0)                               // 00000000253c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v100, s8                // 000000002540: d5207c07 0022c80d
	v_add_co_u32 v20, s10, v103, s12                           // 000000002548: d7000a14 02001967
	v_add_co_ci_u32_e64 v9, null, s13, v102, s9                // 000000002550: d5207c09 0026cc0d
	v_add_co_u32 v0, s5, v92, s12                              // 000000002558: d7000500 0200195c
	v_add_co_u32 v2, s6, v94, s12                              // 000000002560: d7000602 0200195e
	v_add_co_u32 v24, s11, v105, s12                           // 000000002568: d7000b18 02001969
	v_add_co_u32 v28, s12, v107, s12                           // 000000002570: d7000c1c 0200196b
	v_add_co_ci_u32_e64 v5, null, s13, v98, s7                 // 000000002578: d5207c05 001ec40d
	s_wait_alu depctr_va_sdst(0)                               // 000000002580: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s13, v104, s10              // 000000002584: d5207c15 002ad00d
	v_add_co_ci_u32_e64 v1, null, s13, v93, s5                 // 00000000258c: d5207c01 0016ba0d
	v_add_co_ci_u32_e64 v3, null, s13, v95, s6                 // 000000002594: d5207c03 001abe0d
	v_add_co_ci_u32_e64 v25, null, s13, v106, s11              // 00000000259c: d5207c19 002ed40d
	v_add_co_ci_u32_e64 v29, null, s13, v108, s12              // 0000000025a4: d5207c1d 0032d80d
	global_load_b128 v[12:15], v[6:7], off                     // 0000000025ac: ee05c07c 0000000c 00000006
	global_load_b128 v[16:19], v[8:9], off                     // 0000000025b8: ee05c07c 00000010 00000008
	global_load_b128 v[8:11], v[4:5], off                      // 0000000025c4: ee05c07c 00000008 00000004
	global_load_b128 v[20:23], v[20:21], off                   // 0000000025d0: ee05c07c 00000014 00000014
	global_load_b128 v[4:7], v[2:3], off                       // 0000000025dc: ee05c07c 00000004 00000002
	global_load_b128 v[24:27], v[24:25], off                   // 0000000025e8: ee05c07c 00000018 00000018
	global_load_b128 v[0:3], v[0:1], off                       // 0000000025f4: ee05c07c 00000000 00000000
	global_load_b128 v[28:31], v[28:29], off                   // 000000002600: ee05c07c 0000001c 0000001c
	s_mul_u64 s[6:7], s[36:37], s[34:35]                       // 00000000260c: aa862224
	v_add_co_u32 v166, s5, v121, s20                           // 000000002610: d70005a6 02002979
	s_wait_alu depctr_sa_sdst(0)                               // 000000002618: bf88ff9e
	s_lshl_b64 s[40:41], s[6:7], 2                             // 00000000261c: 84a88206
	v_add_co_u32 v168, s6, v124, s20                           // 000000002620: d70006a8 0200297c
	v_add_co_u32 v170, s7, v127, s20                           // 000000002628: d70007aa 0200297f
	v_add_co_ci_u32_e64 v167, null, s21, v122, s5              // 000000002630: d5207ca7 0016f415
	v_add_co_u32 v172, s8, v130, s20                           // 000000002638: d70008ac 02002982
	s_wait_alu depctr_va_sdst(0)                               // 000000002640: bf88f19f
	v_add_co_ci_u32_e64 v169, null, s21, v125, s6              // 000000002644: d5207ca9 001afa15
	v_add_co_u32 v174, s9, v132, s20                           // 00000000264c: d70009ae 02002984
	v_add_co_ci_u32_e64 v171, null, s21, v128, s7              // 000000002654: d5207cab 001f0015
	v_add_co_u32 v176, s10, v135, s20                          // 00000000265c: d7000ab0 02002987
	v_add_co_u32 v178, s11, v138, s20                          // 000000002664: d7000bb2 0200298a
	v_add_co_u32 v180, s12, v140, s20                          // 00000000266c: d7000cb4 0200298c
	v_add_co_u32 v182, s13, v144, s20                          // 000000002674: d7000db6 02002990
	v_add_co_u32 v184, s14, v147, s20                          // 00000000267c: d7000eb8 02002993
	v_add_co_u32 v186, s15, v150, s20                          // 000000002684: d7000fba 02002996
	v_add_co_u32 v188, s16, v152, s20                          // 00000000268c: d70010bc 02002998
	v_add_co_u32 v190, s17, v154, s20                          // 000000002694: d70011be 0200299a
	v_add_co_u32 v192, s18, v156, s20                          // 00000000269c: d70012c0 0200299c
	v_add_co_u32 v194, s19, v158, s20                          // 0000000026a4: d70013c2 0200299e
	v_add_co_u32 v196, s20, v160, s20                          // 0000000026ac: d70014c4 020029a0
	v_add_co_ci_u32_e64 v173, null, s21, v131, s8              // 0000000026b4: d5207cad 00230615
	s_wait_alu depctr_va_sdst(0)                               // 0000000026bc: bf88f19f
	v_add_co_ci_u32_e64 v175, null, s21, v133, s9              // 0000000026c0: d5207caf 00270a15
	v_add_co_ci_u32_e64 v177, null, s21, v136, s10             // 0000000026c8: d5207cb1 002b1015
	v_add_co_ci_u32_e64 v179, null, s21, v139, s11             // 0000000026d0: d5207cb3 002f1615
	v_add_co_ci_u32_e64 v181, null, s21, v141, s12             // 0000000026d8: d5207cb5 00331a15
	v_add_co_ci_u32_e64 v183, null, s21, v145, s13             // 0000000026e0: d5207cb7 00372215
	v_add_co_ci_u32_e64 v185, null, s21, v148, s14             // 0000000026e8: d5207cb9 003b2815
	v_add_co_ci_u32_e64 v187, null, s21, v151, s15             // 0000000026f0: d5207cbb 003f2e15
	v_add_co_ci_u32_e64 v189, null, s21, v153, s16             // 0000000026f8: d5207cbd 00433215
	v_add_co_ci_u32_e64 v191, null, s21, v155, s17             // 000000002700: d5207cbf 00473615
	v_add_co_ci_u32_e64 v193, null, s21, v157, s18             // 000000002708: d5207cc1 004b3a15
	v_add_co_ci_u32_e64 v195, null, s21, v159, s19             // 000000002710: d5207cc3 004f3e15
	v_add_co_ci_u32_e64 v197, null, s21, v161, s20             // 000000002718: d5207cc5 00534215
	v_add_nc_u32_e32 v165, 0x4800, v111                        // 000000002720: 4b4adeff 00004800
	v_add_nc_u32_e32 v164, 0x4800, v113                        // 000000002728: 4b48e2ff 00004800
	v_add_nc_u32_e32 v163, 0x4800, v114                        // 000000002730: 4b46e4ff 00004800
	v_add_nc_u32_e32 v162, 0x4800, v116                        // 000000002738: 4b44e8ff 00004800
	s_add_nc_u64 s[6:7], s[28:29], s[40:41]                    // 000000002740: a986281c
	s_add_nc_u64 s[36:37], s[36:37], 1                         // 000000002744: a9a48124
	s_wait_alu depctr_sa_sdst(0)                               // 000000002748: bf88ff9e
	s_add_nc_u64 s[8:9], s[6:7], s[38:39]                      // 00000000274c: a9882606
	s_cmp_lg_u64 s[36:37], s[30:31]                            // 000000002750: bf111e24
	s_barrier_signal -1                                        // 000000002754: be804ec1
	s_barrier_wait 0xffff                                      // 000000002758: bf94ffff
	s_wait_loadcnt 0x7                                         // 00000000275c: bfc00007
	ds_store_b128 v91, v[12:15] offset:13824                   // 000000002760: db7c3600 00000c5b
	s_wait_loadcnt 0x6                                         // 000000002768: bfc00006
	ds_store_b128 v91, v[16:19] offset:18432                   // 00000000276c: db7c4800 0000105b
	s_wait_loadcnt 0x5                                         // 000000002774: bfc00005
	ds_store_b128 v91, v[8:11] offset:9216                     // 000000002778: db7c2400 0000085b
	s_wait_loadcnt 0x4                                         // 000000002780: bfc00004
	ds_store_b128 v90, v[20:23] offset:18432                   // 000000002784: db7c4800 0000145a
	s_wait_loadcnt 0x3                                         // 00000000278c: bfc00003
	ds_store_b128 v91, v[4:7] offset:4608                      // 000000002790: db7c1200 0000045b
	s_wait_loadcnt 0x2                                         // 000000002798: bfc00002
	ds_store_b128 v89, v[24:27] offset:18432                   // 00000000279c: db7c4800 00001859
	s_wait_loadcnt 0x1                                         // 0000000027a4: bfc00001
	ds_store_b128 v91, v[0:3]                                  // 0000000027a8: db7c0000 0000005b
	s_wait_loadcnt 0x0                                         // 0000000027b0: bfc00000
	ds_store_b128 v87, v[28:31] offset:18432                   // 0000000027b4: db7c4800 00001c57
	s_wait_dscnt 0x0                                           // 0000000027bc: bfc60000
	s_barrier_signal -1                                        // 0000000027c0: be804ec1
	s_barrier_wait 0xffff                                      // 0000000027c4: bf94ffff
	s_clause 0xf                                               // 0000000027c8: bf85000f
	global_load_b32 v222, v[166:167], off                      // 0000000027cc: ee05007c 000000de 000000a6
	global_load_b32 v223, v[168:169], off                      // 0000000027d8: ee05007c 000000df 000000a8
	global_load_b32 v224, v[170:171], off                      // 0000000027e4: ee05007c 000000e0 000000aa
	global_load_b32 v225, v[172:173], off                      // 0000000027f0: ee05007c 000000e1 000000ac
	global_load_b32 v226, v[174:175], off                      // 0000000027fc: ee05007c 000000e2 000000ae
	global_load_b32 v227, v[176:177], off                      // 000000002808: ee05007c 000000e3 000000b0
	global_load_b32 v228, v[178:179], off                      // 000000002814: ee05007c 000000e4 000000b2
	global_load_b32 v229, v[180:181], off                      // 000000002820: ee05007c 000000e5 000000b4
	global_load_b32 v230, v[182:183], off                      // 00000000282c: ee05007c 000000e6 000000b6
	global_load_b32 v231, v[184:185], off                      // 000000002838: ee05007c 000000e7 000000b8
	global_load_b32 v232, v[186:187], off                      // 000000002844: ee05007c 000000e8 000000ba
	global_load_b32 v233, v[188:189], off                      // 000000002850: ee05007c 000000e9 000000bc
	global_load_b32 v234, v[190:191], off                      // 00000000285c: ee05007c 000000ea 000000be
	global_load_b32 v235, v[192:193], off                      // 000000002868: ee05007c 000000eb 000000c0
	global_load_b32 v236, v[194:195], off                      // 000000002874: ee05007c 000000ec 000000c2
	global_load_b32 v237, v[196:197], off                      // 000000002880: ee05007c 000000ed 000000c4
	ds_load_2addr_b64 v[180:183], v109 offset1:2               // 00000000288c: d9dc0200 b400006d
	ds_load_2addr_b64 v[184:187], v165 offset1:2               // 000000002894: d9dc0200 b80000a5
	ds_load_2addr_b64 v[188:191], v164 offset1:2               // 00000000289c: d9dc0200 bc0000a4
	ds_load_2addr_b64 v[192:195], v163 offset1:2               // 0000000028a4: d9dc0200 c00000a3
	ds_load_2addr_b64 v[198:201], v162 offset1:2               // 0000000028ac: d9dc0200 c60000a2
	ds_load_2addr_b64 v[202:205], v110 offset1:2               // 0000000028b4: d9dc0200 ca00006e
	ds_load_2addr_b64 v[206:209], v109 offset0:4 offset1:6     // 0000000028bc: d9dc0604 ce00006d
	ds_load_2addr_b64 v[210:213], v165 offset0:4 offset1:6     // 0000000028c4: d9dc0604 d20000a5
	ds_load_2addr_b64 v[214:217], v164 offset0:4 offset1:6     // 0000000028cc: d9dc0604 d60000a4
	s_clause 0x1                                               // 0000000028d4: bf850001
	s_load_b32 s5, s[8:9], 0x0                                 // 0000000028d8: f4000144 f8000000
	s_load_b32 s6, s[6:7], 0x0                                 // 0000000028e0: f4000183 f8000000
	s_wait_dscnt 0x7                                           // 0000000028e8: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[180:181], v[184:185], 0// 0000000028ec: cc464000 1a0371b4
	s_wait_dscnt 0x6                                           // 0000000028f4: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[180:181], v[188:189], 0// 0000000028f8: cc464008 1a0379b4
	s_wait_dscnt 0x5                                           // 000000002900: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[180:181], v[192:193], 0// 000000002904: cc464010 1a0381b4
	s_wait_dscnt 0x4                                           // 00000000290c: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[180:181], v[198:199], 0// 000000002910: cc464018 1a038db4
	s_wait_dscnt 0x3                                           // 000000002918: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[202:203], v[184:185], 0// 00000000291c: cc4640a6 1a0371ca
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[202:203], v[188:189], 0// 000000002924: cc4640ae 1a0379ca
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[182:183], v[186:187], v[0:7]// 00000000292c: cc464000 1c0375b6
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[182:183], v[190:191], v[8:15]// 000000002934: cc464008 1c237db6
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[182:183], v[194:195], v[16:23]// 00000000293c: cc464010 1c4385b6
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[182:183], v[200:201], v[24:31]// 000000002944: cc464018 1c6391b6
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[204:205], v[186:187], v[166:173]// 00000000294c: cc4640a6 1e9b75cc
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[202:203], v[192:193], 0// 000000002954: cc4640b6 1a0381ca
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[204:205], v[190:191], v[174:181]// 00000000295c: cc4640ae 1ebb7dcc
	s_wait_dscnt 0x1                                           // 000000002964: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[206:207], v[210:211], v[0:7]// 000000002968: cc464000 1c03a5ce
	s_wait_dscnt 0x0                                           // 000000002970: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[206:207], v[214:215], v[8:15]// 000000002974: cc464008 1c23adce
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[204:205], v[194:195], v[182:189]// 00000000297c: cc4640b6 1edb85cc
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[202:203], v[198:199], 0// 000000002984: cc4640be 1a038dca
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[208:209], v[212:213], v[0:7]// 00000000298c: cc464000 1c03a9d0
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002994: bf870194
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[208:209], v[216:217], v[8:15]// 000000002998: cc464008 1c23b1d0
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[204:205], v[200:201], v[190:197]// 0000000029a0: cc4640be 1efb91cc
	ds_load_2addr_b64 v[198:201], v163 offset0:4 offset1:6     // 0000000029a8: d9dc0604 c60000a3
	ds_load_2addr_b64 v[202:205], v162 offset0:4 offset1:6     // 0000000029b0: d9dc0604 ca0000a2
	s_wait_dscnt 0x1                                           // 0000000029b8: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[206:207], v[198:199], v[16:23]// 0000000029bc: cc464010 1c438dce
	s_wait_dscnt 0x0                                           // 0000000029c4: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[206:207], v[202:203], v[24:31]// 0000000029c8: cc464018 1c6395ce
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000029d0: bf870112
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[208:209], v[200:201], v[16:23]// 0000000029d4: cc464010 1c4391d0
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[208:209], v[204:205], v[24:31]// 0000000029dc: cc464018 1c6399d0
	ds_load_2addr_b64 v[206:209], v110 offset0:4 offset1:6     // 0000000029e4: d9dc0604 ce00006e
	s_wait_dscnt 0x0                                           // 0000000029ec: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[206:207], v[210:211], v[166:173]// 0000000029f0: cc4640a6 1e9ba5ce
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[206:207], v[214:215], v[174:181]// 0000000029f8: cc4640ae 1ebbadce
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[206:207], v[198:199], v[182:189]// 000000002a00: cc4640b6 1edb8dce
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[206:207], v[202:203], v[190:197]// 000000002a08: cc4640be 1efb95ce
	s_delay_alu instid0(valu_dep_4)                            // 000000002a10: bf870004
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[208:209], v[212:213], v[166:173]// 000000002a14: cc4640a6 1e9ba9d0
	ds_load_2addr_b64 v[210:213], v109 offset0:8 offset1:10    // 000000002a1c: d9dc0a08 d200006d
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[208:209], v[216:217], v[174:181]// 000000002a24: cc4640ae 1ebbb1d0
	ds_load_2addr_b64 v[214:217], v165 offset0:8 offset1:10    // 000000002a2c: d9dc0a08 d60000a5
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[208:209], v[200:201], v[182:189]// 000000002a34: cc4640b6 1edb91d0
	ds_load_2addr_b64 v[198:201], v164 offset0:8 offset1:10    // 000000002a3c: d9dc0a08 c60000a4
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[208:209], v[204:205], v[190:197]// 000000002a44: cc4640be 1efb99d0
	ds_load_2addr_b64 v[202:205], v163 offset0:8 offset1:10    // 000000002a4c: d9dc0a08 ca0000a3
	ds_load_2addr_b64 v[206:209], v162 offset0:8 offset1:10    // 000000002a54: d9dc0a08 ce0000a2
	s_wait_dscnt 0x3                                           // 000000002a5c: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[210:211], v[214:215], v[0:7]// 000000002a60: cc464000 1c03add2
	s_wait_dscnt 0x2                                           // 000000002a68: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[210:211], v[198:199], v[8:15]// 000000002a6c: cc464008 1c238dd2
	s_wait_dscnt 0x1                                           // 000000002a74: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[210:211], v[202:203], v[16:23]// 000000002a78: cc464010 1c4395d2
	s_wait_dscnt 0x0                                           // 000000002a80: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[210:211], v[206:207], v[24:31]// 000000002a84: cc464018 1c639dd2
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[212:213], v[216:217], v[0:7]// 000000002a8c: cc464000 1c03b1d4
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[212:213], v[200:201], v[8:15]// 000000002a94: cc464008 1c2391d4
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[212:213], v[204:205], v[16:23]// 000000002a9c: cc464010 1c4399d4
	s_delay_alu instid0(valu_dep_4)                            // 000000002aa4: bf870004
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[212:213], v[208:209], v[24:31]// 000000002aa8: cc464018 1c63a1d4
	ds_load_2addr_b64 v[210:213], v110 offset0:8 offset1:10    // 000000002ab0: d9dc0a08 d200006e
	s_wait_dscnt 0x0                                           // 000000002ab8: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[210:211], v[214:215], v[166:173]// 000000002abc: cc4640a6 1e9badd2
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[210:211], v[198:199], v[174:181]// 000000002ac4: cc4640ae 1ebb8dd2
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[210:211], v[202:203], v[182:189]// 000000002acc: cc4640b6 1edb95d2
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[210:211], v[206:207], v[190:197]// 000000002ad4: cc4640be 1efb9dd2
	s_wait_kmcnt 0x0                                           // 000000002adc: bfc70000
	v_mov_b32_e32 v210, s5                                     // 000000002ae0: 7fa40205
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[212:213], v[216:217], v[166:173]// 000000002ae4: cc4640a6 1e9bb1d4
	ds_load_2addr_b64 v[214:217], v109 offset0:12 offset1:14   // 000000002aec: d9dc0e0c d600006d
	ds_load_2addr_b64 v[218:221], v165 offset0:12 offset1:14   // 000000002af4: d9dc0e0c da0000a5
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[212:213], v[200:201], v[174:181]// 000000002afc: cc4640ae 1ebb91d4
	ds_load_2addr_b64 v[198:201], v164 offset0:12 offset1:14   // 000000002b04: d9dc0e0c c60000a4
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[212:213], v[204:205], v[182:189]// 000000002b0c: cc4640b6 1edb99d4
	ds_load_2addr_b64 v[202:205], v163 offset0:12 offset1:14   // 000000002b14: d9dc0e0c ca0000a3
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[212:213], v[208:209], v[190:197]// 000000002b1c: cc4640be 1efba1d4
	ds_load_2addr_b64 v[162:165], v162 offset0:12 offset1:14   // 000000002b24: d9dc0e0c a20000a2
	ds_load_2addr_b64 v[206:209], v110 offset0:12 offset1:14   // 000000002b2c: d9dc0e0c ce00006e
	v_cndmask_b32_e64 v211, s6, v210, s4                       // 000000002b34: d50100d3 0013a406
	s_wait_dscnt 0x4                                           // 000000002b3c: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[214:215], v[218:219], v[0:7]// 000000002b40: cc464000 1c03b5d6
	s_wait_dscnt 0x3                                           // 000000002b48: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[214:215], v[198:199], v[8:15]// 000000002b4c: cc464008 1c238dd6
	s_wait_dscnt 0x2                                           // 000000002b54: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[214:215], v[202:203], v[16:23]// 000000002b58: cc464010 1c4395d6
	s_wait_dscnt 0x1                                           // 000000002b60: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[214:215], v[162:163], v[24:31]// 000000002b64: cc464018 1c6345d6
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[216:217], v[220:221], v[0:7]// 000000002b6c: cc464000 1c03b9d8
	s_wait_dscnt 0x0                                           // 000000002b74: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[206:207], v[218:219], v[166:173]// 000000002b78: cc4640a6 1e9bb5ce
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[206:207], v[198:199], v[174:181]// 000000002b80: cc4640ae 1ebb8dce
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[206:207], v[202:203], v[182:189]// 000000002b88: cc4640b6 1edb95ce
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[206:207], v[162:163], v[190:197]// 000000002b90: cc4640be 1efb45ce
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[216:217], v[200:201], v[8:15]// 000000002b98: cc464008 1c2391d8
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[216:217], v[204:205], v[16:23]// 000000002ba0: cc464010 1c4399d8
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[216:217], v[164:165], v[24:31]// 000000002ba8: cc464018 1c6349d8
	v_wmma_f32_16x16x16_fp8_fp8 v[166:173], v[208:209], v[220:221], v[166:173]// 000000002bb0: cc4640a6 1e9bb9d0
	v_wmma_f32_16x16x16_fp8_fp8 v[174:181], v[208:209], v[200:201], v[174:181]// 000000002bb8: cc4640ae 1ebb91d0
	v_wmma_f32_16x16x16_fp8_fp8 v[182:189], v[208:209], v[204:205], v[182:189]// 000000002bc0: cc4640b6 1edb99d0
	v_wmma_f32_16x16x16_fp8_fp8 v[190:197], v[208:209], v[164:165], v[190:197]// 000000002bc8: cc4640be 1efb49d0
	s_wait_loadcnt 0xf                                         // 000000002bd0: bfc0000f
	v_mul_f32_e32 v162, v222, v211                             // 000000002bd4: 1145a7de
	s_wait_loadcnt 0xe                                         // 000000002bd8: bfc0000e
	v_mul_f32_e32 v163, v211, v223                             // 000000002bdc: 1147bfd3
	v_cndmask_b32_e64 v212, s6, v210, s3                       // 000000002be0: d50100d4 000fa406
	v_cndmask_b32_e64 v213, s6, v210, s2                       // 000000002be8: d50100d5 000ba406
	s_wait_loadcnt 0xc                                         // 000000002bf0: bfc0000c
	v_dual_cndmask_b32 v210, s6, v210 :: v_dual_mul_f32 v165, v211, v225// 000000002bf4: ca47a406 d2a5c3d3
	s_wait_loadcnt 0xa                                         // 000000002bfc: bfc0000a
	v_dual_mul_f32 v164, v211, v224 :: v_dual_mul_f32 v199, v211, v227// 000000002c00: c8c7c1d3 a4c7c7d3
	v_dual_mul_f32 v198, v211, v226 :: v_dual_mul_f32 v203, v212, v223// 000000002c08: c8c7c5d3 c6cbbfd4
	s_wait_loadcnt 0x9                                         // 000000002c10: bfc00009
	v_dual_mul_f32 v200, v211, v228 :: v_dual_mul_f32 v205, v212, v225// 000000002c14: c8c7c9d3 c8cdc3d4
	s_wait_loadcnt 0x8                                         // 000000002c1c: bfc00008
	v_dual_mul_f32 v201, v211, v229 :: v_dual_mul_f32 v202, v222, v212// 000000002c20: c8c7cbd3 c9cba9de
	v_dual_mul_f32 v207, v212, v227 :: v_dual_mul_f32 v204, v212, v224// 000000002c28: c8c7c7d4 cfcdc1d4
	v_dual_mul_f32 v209, v212, v229 :: v_dual_mul_f32 v206, v212, v226// 000000002c30: c8c7cbd4 d1cfc5d4
	v_dual_mul_f32 v215, v213, v223 :: v_dual_mul_f32 v208, v212, v228// 000000002c38: c8c7bfd5 d7d1c9d4
	v_mul_f32_e32 v217, v213, v225                             // 000000002c40: 11b3c3d5
	v_dual_mul_f32 v214, v222, v213 :: v_dual_mul_f32 v219, v213, v227// 000000002c44: c8c7abde d6dbc7d5
	v_dual_mul_f32 v216, v213, v224 :: v_dual_mul_f32 v3, v3, v165// 000000002c4c: c8c7c1d5 d8034b03
	v_dual_mul_f32 v218, v213, v226 :: v_dual_mul_f32 v7, v7, v201// 000000002c54: c8c7c5d5 da079307
	v_mul_f32_e32 v220, v213, v228                             // 000000002c5c: 11b9c9d5
	v_dual_mul_f32 v0, v0, v162 :: v_dual_mul_f32 v1, v1, v163 // 000000002c60: c8c74500 00014701
	v_mul_f32_e32 v162, v213, v229                             // 000000002c68: 1145cbd5
	v_dual_mul_f32 v2, v2, v164 :: v_dual_mul_f32 v5, v5, v199 // 000000002c6c: c8c74902 02058f05
	v_dual_mul_f32 v4, v4, v198 :: v_dual_mul_f32 v165, v210, v224// 000000002c74: c8c78d04 04a5c1d2
	v_mul_f32_e32 v6, v6, v200                                 // 000000002c7c: 100d9106
	v_mul_f32_e32 v163, v222, v210                             // 000000002c80: 1147a5de
	v_dual_mul_f32 v164, v210, v223 :: v_dual_mul_f32 v201, v210, v228// 000000002c84: c8c7bfd2 a4c9c9d2
	v_dual_mul_f32 v198, v210, v225 :: v_dual_mul_f32 v199, v210, v226// 000000002c8c: c8c7c3d2 c6c7c5d2
	v_dual_mul_f32 v200, v210, v227 :: v_dual_mul_f32 v221, v210, v229// 000000002c94: c8c7c7d2 c8ddcbd2
	s_wait_loadcnt 0x4                                         // 000000002c9c: bfc00004
	v_dual_mul_f32 v222, v211, v230 :: v_dual_mul_f32 v225, v211, v233// 000000002ca0: c8c7cdd3 dee1d3d3
	v_dual_mul_f32 v223, v211, v231 :: v_dual_mul_f32 v224, v211, v232// 000000002ca8: c8c7cfd3 dfe1d1d3
	s_wait_loadcnt 0x3                                         // 000000002cb0: bfc00003
	v_dual_mul_f32 v226, v211, v234 :: v_dual_mul_f32 v9, v9, v203// 000000002cb4: c8c7d5d3 e2099709
	s_wait_loadcnt 0x1                                         // 000000002cbc: bfc00001
	v_dual_mul_f32 v227, v211, v235 :: v_dual_mul_f32 v228, v211, v236// 000000002cc0: c8c7d7d3 e3e5d9d3
	v_mul_f32_e32 v13, v13, v207                               // 000000002cc8: 101b9f0d
	s_wait_loadcnt 0x0                                         // 000000002ccc: bfc00000
	v_dual_mul_f32 v211, v211, v237 :: v_dual_mul_f32 v8, v8, v202// 000000002cd0: c8c7dbd3 d3099508
	v_dual_mul_f32 v11, v11, v205 :: v_dual_mul_f32 v10, v10, v204// 000000002cd8: c8c79b0b 0b0b990a
	v_dual_mul_f32 v15, v15, v209 :: v_dual_mul_f32 v12, v12, v206// 000000002ce0: c8c7a30f 0f0d9d0c
	v_dual_mul_f32 v17, v17, v215 :: v_dual_mul_f32 v14, v14, v208// 000000002ce8: c8c7af11 110fa10e
	v_dual_mul_f32 v203, v212, v231 :: v_dual_mul_f32 v202, v212, v230// 000000002cf0: c8c7cfd4 cbcbcdd4
	v_dual_mul_f32 v205, v212, v233 :: v_dual_mul_f32 v204, v212, v232// 000000002cf8: c8c7d3d4 cdcdd1d4
	v_dual_mul_f32 v207, v212, v235 :: v_dual_mul_f32 v206, v212, v234// 000000002d00: c8c7d7d4 cfcfd5d4
	v_dual_mul_f32 v209, v212, v237 :: v_dual_mul_f32 v208, v212, v236// 000000002d08: c8c7dbd4 d1d1d9d4
	v_dual_mul_f32 v19, v19, v217 :: v_dual_mul_f32 v212, v213, v230// 000000002d10: c8c7b313 13d5cdd5
	v_dual_mul_f32 v215, v213, v233 :: v_dual_mul_f32 v16, v16, v214// 000000002d18: c8c7d3d5 d711ad10
	v_dual_mul_f32 v21, v21, v219 :: v_dual_mul_f32 v18, v18, v216// 000000002d20: c8c7b715 1513b112
	v_mul_f32_e32 v23, v23, v162                               // 000000002d28: 102f4517
	v_dual_mul_f32 v20, v20, v218 :: v_dual_mul_f32 v217, v213, v235// 000000002d2c: c8c7b514 14d9d7d5
	v_mul_f32_e32 v22, v22, v220                               // 000000002d34: 102db916
	v_dual_mul_f32 v162, v213, v231 :: v_dual_mul_f32 v219, v210, v230// 000000002d38: c8c7cfd5 a2dbcdd2
	v_dual_mul_f32 v214, v213, v232 :: v_dual_mul_f32 v229, v210, v232// 000000002d40: c8c7d1d5 d6e5d1d2
	v_mul_f32_e32 v216, v213, v234                             // 000000002d48: 11b1d5d5
	v_dual_mul_f32 v218, v213, v236 :: v_dual_mul_f32 v213, v213, v237// 000000002d4c: c8c7d9d5 dad5dbd5
	v_dual_mul_f32 v220, v210, v231 :: v_dual_mul_f32 v25, v25, v164// 000000002d54: c8c7cfd2 dc194919
	v_dual_mul_f32 v230, v210, v233 :: v_dual_mul_f32 v27, v27, v198// 000000002d5c: c8c7d3d2 e61b8d1b
	v_dual_mul_f32 v231, v210, v234 :: v_dual_mul_f32 v232, v210, v235// 000000002d64: c8c7d5d2 e7e9d7d2
	v_mul_f32_e32 v29, v29, v200                               // 000000002d6c: 103b911d
	v_dual_mul_f32 v233, v210, v236 :: v_dual_mul_f32 v210, v210, v237// 000000002d70: c8c7d9d2 e9d3dbd2
	v_dual_mul_f32 v24, v24, v163 :: v_dual_mul_f32 v31, v31, v221// 000000002d78: c8c74718 181fbb1f
	v_mul_f32_e32 v26, v26, v165                               // 000000002d80: 10354b1a
	v_dual_mul_f32 v28, v28, v199 :: v_dual_mul_f32 v163, v166, v222// 000000002d84: c8c78f1c 1ca3bda6
	v_mul_f32_e32 v30, v30, v201                               // 000000002d8c: 103d931e
	v_dual_mul_f32 v164, v167, v223 :: v_dual_mul_f32 v167, v170, v226// 000000002d90: c8c7bfa7 a4a7c5aa
	v_dual_mul_f32 v165, v168, v224 :: v_dual_mul_f32 v166, v169, v225// 000000002d98: c8c7c1a8 a5a7c3a9
	v_dual_mul_f32 v168, v171, v227 :: v_dual_mul_f32 v169, v172, v228// 000000002da0: c8c7c7ab a8a9c9ac
	v_dual_mul_f32 v170, v173, v211 :: v_dual_mul_f32 v171, v174, v202// 000000002da8: c8c7a7ad aaab95ae
	v_dual_mul_f32 v172, v175, v203 :: v_dual_mul_f32 v173, v176, v204// 000000002db0: c8c797af acad99b0
	v_dual_mul_f32 v174, v177, v205 :: v_dual_mul_f32 v175, v178, v206// 000000002db8: c8c79bb1 aeaf9db2
	v_dual_mul_f32 v176, v179, v207 :: v_dual_mul_f32 v177, v180, v208// 000000002dc0: c8c79fb3 b0b1a1b4
	v_dual_mul_f32 v178, v181, v209 :: v_dual_mul_f32 v179, v182, v212// 000000002dc8: c8c7a3b5 b2b3a9b6
	v_dual_mul_f32 v162, v183, v162 :: v_dual_mul_f32 v181, v185, v215// 000000002dd0: c8c745b7 a2b5afb9
	v_dual_mul_f32 v180, v184, v214 :: v_dual_mul_f32 v183, v187, v217// 000000002dd8: c8c7adb8 b4b7b3bb
	v_mul_f32_e32 v182, v186, v216                             // 000000002de0: 116db1ba
	v_dual_mul_f32 v184, v188, v218 :: v_dual_add_f32 v83, v83, v0// 000000002de4: c8c9b5bc b8520153
	v_dual_mul_f32 v185, v189, v213 :: v_dual_mul_f32 v186, v190, v219// 000000002dec: c8c7abbd b9bbb7be
	v_dual_mul_f32 v187, v191, v220 :: v_dual_mul_f32 v188, v192, v229// 000000002df4: c8c7b9bf bbbdcbc0
	v_add_f32_e32 v143, v143, v3                               // 000000002dfc: 071e078f
	v_dual_mul_f32 v189, v193, v230 :: v_dual_mul_f32 v190, v194, v231// 000000002e00: c8c7cdc1 bdbfcfc2
	v_add_f32_e32 v149, v149, v1                               // 000000002e08: 072a0395
	v_dual_mul_f32 v191, v195, v232 :: v_dual_mul_f32 v192, v196, v233// 000000002e0c: c8c7d1c3 bfc1d3c4
	v_add_f32_e32 v129, v129, v7                               // 000000002e14: 07020f81
	v_mul_f32_e32 v193, v197, v210                             // 000000002e18: 1183a5c5
	v_dual_add_f32 v146, v146, v2 :: v_dual_add_f32 v137, v137, v5// 000000002e1c: c9080592 92880b89
	v_dual_add_f32 v142, v142, v4 :: v_dual_add_f32 v85, v85, v11// 000000002e24: c908098e 8e541755
	v_dual_add_f32 v134, v134, v6 :: v_dual_add_f32 v71, v71, v16// 000000002e2c: c9080d86 86462147
	v_dual_add_f32 v96, v96, v8 :: v_dual_add_f32 v81, v81, v14// 000000002e34: c9081160 60501d51
	v_dual_add_f32 v88, v88, v9 :: v_dual_add_f32 v69, v69, v18// 000000002e3c: c9081358 58442545
	v_dual_add_f32 v86, v86, v10 :: v_dual_add_f32 v67, v67, v20// 000000002e44: c9081556 56422943
	v_dual_add_f32 v84, v84, v12 :: v_dual_add_f32 v65, v65, v22// 000000002e4c: c9081954 54402d41
	v_dual_add_f32 v82, v82, v13 :: v_dual_add_f32 v55, v55, v24// 000000002e54: c9081b52 52363137
	v_dual_add_f32 v80, v80, v15 :: v_dual_add_f32 v53, v53, v26// 000000002e5c: c9081f50 50343535
	v_dual_add_f32 v70, v70, v17 :: v_dual_add_f32 v51, v51, v28// 000000002e64: c9082346 46323933
	v_dual_add_f32 v68, v68, v19 :: v_dual_add_f32 v49, v49, v30// 000000002e6c: c9082744 44303d31
	v_dual_add_f32 v66, v66, v21 :: v_dual_add_f32 v123, v123, v164// 000000002e74: c9082b42 427b497b
	v_dual_add_f32 v64, v64, v23 :: v_dual_add_f32 v119, v119, v166// 000000002e7c: c9082f40 40774d77
	v_dual_add_f32 v54, v54, v25 :: v_dual_add_f32 v117, v117, v168// 000000002e84: c9083336 36755175
	v_dual_add_f32 v52, v52, v27 :: v_dual_add_f32 v115, v115, v169// 000000002e8c: c9083734 34735373
	v_dual_add_f32 v50, v50, v29 :: v_dual_add_f32 v79, v79, v171// 000000002e94: c9083b32 324f574f
	v_dual_add_f32 v48, v48, v31 :: v_dual_add_f32 v77, v77, v173// 000000002e9c: c9083f30 304d5b4d
	v_dual_add_f32 v126, v126, v163 :: v_dual_add_f32 v73, v73, v177// 000000002ea4: c909477e 7e496349
	v_dual_add_f32 v120, v120, v165 :: v_dual_add_f32 v75, v75, v175// 000000002eac: c9094b78 784b5f4b
	v_dual_add_f32 v118, v118, v167 :: v_dual_add_f32 v61, v61, v180// 000000002eb4: c9094f76 763d693d
	v_dual_add_f32 v112, v112, v170 :: v_dual_add_f32 v63, v63, v179// 000000002ebc: c9095570 703f673f
	v_dual_add_f32 v78, v78, v172 :: v_dual_add_f32 v59, v59, v182// 000000002ec4: c909594e 4e3b6d3b
	v_dual_add_f32 v76, v76, v174 :: v_dual_add_f32 v57, v57, v184// 000000002ecc: c9095d4c 4c397139
	v_dual_add_f32 v74, v74, v176 :: v_dual_add_f32 v47, v47, v186// 000000002ed4: c909614a 4a2f752f
	v_dual_add_f32 v72, v72, v178 :: v_dual_add_f32 v45, v45, v188// 000000002edc: c9096548 482d792d
	v_dual_add_f32 v62, v62, v162 :: v_dual_add_f32 v41, v41, v192// 000000002ee4: c909453e 3e298129
	v_dual_add_f32 v60, v60, v181 :: v_dual_add_f32 v43, v43, v190// 000000002eec: c9096b3c 3c2b7d2b
	v_add_f32_e32 v58, v58, v183                               // 000000002ef4: 06756f3a
	v_add_f32_e32 v56, v56, v185                               // 000000002ef8: 06717338
	v_add_f32_e32 v46, v46, v187                               // 000000002efc: 065d772e
	v_add_f32_e32 v44, v44, v189                               // 000000002f00: 06597b2c
	v_add_f32_e32 v42, v42, v191                               // 000000002f04: 06557f2a
	v_add_f32_e32 v40, v40, v193                               // 000000002f08: 06518328
	s_cbranch_scc1 64898                                       // 000000002f0c: bfa2fd82 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0xa18>
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000002f10: f4002500 f80000a8
	v_mul_lo_u32 v4, s27, v34                                  // 000000002f18: d72c0004 0202441b
	v_mul_lo_u32 v5, s26, v35                                  // 000000002f20: d72c0005 0202461a
	v_mad_co_u64_u32 v[2:3], null, s26, v34, 0                 // 000000002f28: d6fe7c02 0202441a
	v_sub_co_u32 v0, s0, s24, v34                              // 000000002f30: d7010000 02024418
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002f38: bf870191
	v_sub_co_ci_u32_e64 v1, null, s25, v35, s0                 // 000000002f3c: d5217c01 00024619
	v_add3_u32 v3, v3, v5, v4                                  // 000000002f44: d6550003 04120b03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002f4c: bf8701a2
	v_cmp_lt_i64_e64 s17, 0, v[0:1]                            // 000000002f50: d4510011 02020080
	v_lshlrev_b64_e32 v[4:5], 1, v[32:33]                      // 000000002f58: 3e084081
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000002f5c: 3e040481
	s_and_b32 s0, s17, s4                                      // 000000002f60: 8b000411
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f64: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f68: be812000
	s_cbranch_execz 28                                         // 000000002f6c: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x14e0>
	v_bfe_u32 v6, v83, 16, 1                                   // 000000002f70: d6100006 02052153
	s_wait_kmcnt 0x0                                           // 000000002f78: bfc70000
	v_add_co_u32 v7, s0, s20, v2                               // 000000002f7c: d7000007 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002f84: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s0                  // 000000002f88: d5207c08 00020615
	v_add3_u32 v9, v6, v83, 0x7fff                             // 000000002f90: d6550009 03fea706 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f9c: bf870003
	v_add_co_u32 v6, s0, v7, v4                                // 000000002fa0: d7000006 02020907
	v_or_b32_e32 v10, 0x400000, v83                            // 000000002fa8: 3814a6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fb0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v5, s0                   // 000000002fb4: d5207c07 00020b08
	v_cmp_u_f32_e64 s0, v83, v83                               // 000000002fbc: d4180000 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fc8: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000002fcc: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000002fd4: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002fe4: 8c7e017e
	v_add_co_u32 v6, s0, s26, v32                              // 000000002fe8: d7000006 0202401a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ff0: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s27, v33, s0                 // 000000002ff4: d5207c07 0002421b
	v_cmp_lt_i64_e64 s18, 1, v[0:1]                            // 000000002ffc: d4510012 02020081
	s_delay_alu instid0(valu_dep_2)                            // 000000003004: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]                        // 000000003008: 3e0c0c81
	s_and_b32 s0, s18, s4                                      // 00000000300c: 8b000412
	s_wait_alu depctr_sa_sdst(0)                               // 000000003010: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003014: be812000
	s_cbranch_execz 28                                         // 000000003018: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x158c>
	v_bfe_u32 v8, v149, 16, 1                                  // 00000000301c: d6100008 02052195
	s_wait_kmcnt 0x0                                           // 000000003024: bfc70000
	v_add_co_u32 v9, s0, s20, v2                               // 000000003028: d7000009 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003030: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s21, v3, s0                 // 000000003034: d5207c0a 00020615
	v_add3_u32 v11, v8, v149, 0x7fff                           // 00000000303c: d655000b 03ff2b08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003048: bf870003
	v_add_co_u32 v8, s0, v9, v6                                // 00000000304c: d7000008 02020d09
	v_or_b32_e32 v12, 0x400000, v149                           // 000000003054: 38192aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000305c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v7, s0                  // 000000003060: d5207c09 00020f0a
	v_cmp_u_f32_e64 s0, v149, v149                             // 000000003068: d4180000 02032b95
	s_wait_alu depctr_va_sdst(0)                               // 000000003070: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003074: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000003078: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 000000003080: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 00000000308c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003090: 8c7e017e
	s_lshl_b64 s[40:41], s[26:27], 1                           // 000000003094: 84a8811a
	v_cmp_lt_i64_e64 s16, 2, v[0:1]                            // 000000003098: d4510010 02020082
	v_add_co_u32 v8, s0, s40, v32                              // 0000000030a0: d7000008 02024028
	s_wait_alu depctr_va_sdst(0)                               // 0000000030a8: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s41, v33, s0                 // 0000000030ac: d5207c09 00024229
	s_and_b32 s0, s16, s4                                      // 0000000030b4: 8b000410
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 0000000030b8: 3e101081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030c0: be812000
	s_cbranch_execz 28                                         // 0000000030c4: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1638>
	v_bfe_u32 v10, v146, 16, 1                                 // 0000000030c8: d610000a 02052192
	s_wait_kmcnt 0x0                                           // 0000000030d0: bfc70000
	v_add_co_u32 v11, s0, s20, v2                              // 0000000030d4: d700000b 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000030dc: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s21, v3, s0                 // 0000000030e0: d5207c0c 00020615
	v_add3_u32 v13, v10, v146, 0x7fff                          // 0000000030e8: d655000d 03ff250a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030f4: bf870003
	v_add_co_u32 v10, s0, v11, v8                              // 0000000030f8: d700000a 0202110b
	v_or_b32_e32 v14, 0x400000, v146                           // 000000003100: 381d24ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003108: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s0                 // 00000000310c: d5207c0b 0002130c
	v_cmp_u_f32_e64 s0, v146, v146                             // 000000003114: d4180000 02032592
	s_wait_alu depctr_va_sdst(0)                               // 00000000311c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003120: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 000000003124: d501000c 00021d0d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 00000000312c: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003138: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000313c: 8c7e017e
	s_mul_u64 s[38:39], s[26:27], 3                            // 000000003140: aaa6831a
	v_cmp_lt_i64_e64 s15, 3, v[0:1]                            // 000000003144: d451000f 02020083
	v_add_co_u32 v10, s0, s38, v32                             // 00000000314c: d700000a 02024026
	s_wait_alu depctr_va_sdst(0)                               // 000000003154: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s39, v33, s0                // 000000003158: d5207c0b 00024227
	s_and_b32 s0, s15, s4                                      // 000000003160: 8b00040f
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000003164: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003168: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000316c: be812000
	s_cbranch_execz 28                                         // 000000003170: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x16e4>
	v_bfe_u32 v12, v143, 16, 1                                 // 000000003174: d610000c 0205218f
	s_wait_kmcnt 0x0                                           // 00000000317c: bfc70000
	v_add_co_u32 v13, s0, s20, v2                              // 000000003180: d700000d 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003188: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s21, v3, s0                 // 00000000318c: d5207c0e 00020615
	v_add3_u32 v15, v12, v143, 0x7fff                          // 000000003194: d655000f 03ff1f0c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031a0: bf870003
	v_add_co_u32 v12, s0, v13, v10                             // 0000000031a4: d700000c 0202150d
	v_or_b32_e32 v16, 0x400000, v143                           // 0000000031ac: 38211eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b4: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v11, s0                // 0000000031b8: d5207c0d 0002170e
	v_cmp_u_f32_e64 s0, v143, v143                             // 0000000031c0: d4180000 02031f8f
	s_wait_alu depctr_va_sdst(0)                               // 0000000031c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031cc: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 0000000031d0: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 0000000031d8: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000031e8: 8c7e017e
	s_lshl_b64 s[36:37], s[26:27], 2                           // 0000000031ec: 84a4821a
	v_cmp_lt_i64_e64 s14, 4, v[0:1]                            // 0000000031f0: d451000e 02020084
	v_add_co_u32 v12, s0, s36, v32                             // 0000000031f8: d700000c 02024024
	s_wait_alu depctr_va_sdst(0)                               // 000000003200: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s37, v33, s0                // 000000003204: d5207c0d 00024225
	s_and_b32 s0, s14, s4                                      // 00000000320c: 8b00040e
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000003210: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003218: be812000
	s_cbranch_execz 28                                         // 00000000321c: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1790>
	v_bfe_u32 v14, v142, 16, 1                                 // 000000003220: d610000e 0205218e
	s_wait_kmcnt 0x0                                           // 000000003228: bfc70000
	v_add_co_u32 v15, s0, s20, v2                              // 00000000322c: d700000f 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003234: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s21, v3, s0                 // 000000003238: d5207c10 00020615
	v_add3_u32 v17, v14, v142, 0x7fff                          // 000000003240: d6550011 03ff1d0e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000324c: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 000000003250: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v142                           // 000000003258: 38251cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003260: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 000000003264: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v142, v142                             // 00000000326c: d4180000 02031d8e
	s_wait_alu depctr_va_sdst(0)                               // 000000003274: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003278: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 00000000327c: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 000000003284: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003290: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003294: 8c7e017e
	s_mul_u64 s[34:35], s[26:27], 5                            // 000000003298: aaa2851a
	v_cmp_lt_i64_e64 s13, 5, v[0:1]                            // 00000000329c: d451000d 02020085
	v_add_co_u32 v14, s0, s34, v32                             // 0000000032a4: d700000e 02024022
	s_wait_alu depctr_va_sdst(0)                               // 0000000032ac: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s35, v33, s0                // 0000000032b0: d5207c0f 00024223
	s_and_b32 s0, s13, s4                                      // 0000000032b8: 8b00040d
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 0000000032bc: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032c0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032c4: be812000
	s_cbranch_execz 28                                         // 0000000032c8: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x183c>
	v_bfe_u32 v16, v137, 16, 1                                 // 0000000032cc: d6100010 02052189
	s_wait_kmcnt 0x0                                           // 0000000032d4: bfc70000
	v_add_co_u32 v17, s0, s20, v2                              // 0000000032d8: d7000011 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e0: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v3, s0                 // 0000000032e4: d5207c12 00020615
	v_add3_u32 v19, v16, v137, 0x7fff                          // 0000000032ec: d6550013 03ff1310 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000032f8: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 0000000032fc: d7000010 02021d11
	v_or_b32_e32 v20, 0x400000, v137                           // 000000003304: 382912ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000330c: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000003310: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v137, v137                             // 000000003318: d4180000 02031389
	s_wait_alu depctr_va_sdst(0)                               // 000000003320: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003324: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s0                        // 000000003328: d5010012 00022913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000003330: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 00000000333c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003340: 8c7e017e
	s_mul_u64 s[30:31], s[26:27], 6                            // 000000003344: aa9e861a
	v_cmp_lt_i64_e64 s11, 6, v[0:1]                            // 000000003348: d451000b 02020086
	s_wait_alu depctr_sa_sdst(0)                               // 000000003350: bf88ff9e
	v_add_co_u32 v16, s0, s30, v32                             // 000000003354: d7000010 0202401e
	s_wait_alu depctr_va_sdst(0)                               // 00000000335c: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s31, v33, s0                // 000000003360: d5207c11 0002421f
	s_and_b32 s0, s11, s4                                      // 000000003368: 8b00040b
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 00000000336c: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003370: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003374: be812000
	s_cbranch_execz 28                                         // 000000003378: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x18ec>
	v_bfe_u32 v18, v134, 16, 1                                 // 00000000337c: d6100012 02052186
	s_wait_kmcnt 0x0                                           // 000000003384: bfc70000
	v_add_co_u32 v19, s0, s20, v2                              // 000000003388: d7000013 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003390: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s21, v3, s0                 // 000000003394: d5207c14 00020615
	v_add3_u32 v21, v18, v134, 0x7fff                          // 00000000339c: d6550015 03ff0d12 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000033a8: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 0000000033ac: d7000012 02022113
	v_or_b32_e32 v22, 0x400000, v134                           // 0000000033b4: 382d0cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000033bc: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v17, s0                // 0000000033c0: d5207c13 00022314
	v_cmp_u_f32_e64 s0, v134, v134                             // 0000000033c8: d4180000 02030d86
	s_wait_alu depctr_va_sdst(0)                               // 0000000033d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033d4: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 0000000033d8: d5010014 00022d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 0000000033e0: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000033f0: 8c7e017e
	s_mul_u64 s[28:29], s[26:27], 7                            // 0000000033f4: aa9c871a
	v_cmp_lt_i64_e64 s10, 7, v[0:1]                            // 0000000033f8: d451000a 02020087
	v_add_co_u32 v18, s0, s28, v32                             // 000000003400: d7000012 0202401c
	s_wait_alu depctr_va_sdst(0)                               // 000000003408: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s29, v33, s0                // 00000000340c: d5207c13 0002421d
	s_and_b32 s0, s10, s4                                      // 000000003414: 8b00040a
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 000000003418: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 00000000341c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003420: be812000
	s_cbranch_execz 28                                         // 000000003424: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1998>
	v_bfe_u32 v0, v129, 16, 1                                  // 000000003428: d6100000 02052181
	s_wait_kmcnt 0x0                                           // 000000003430: bfc70000
	v_add_co_u32 v1, s0, s20, v2                               // 000000003434: d7000001 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000343c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s21, v3, s0                 // 000000003440: d5207c14 00020615
	v_add3_u32 v21, v0, v129, 0x7fff                           // 000000003448: d6550015 03ff0300 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003454: bf870003
	v_add_co_u32 v0, s0, v1, v18                               // 000000003458: d7000000 02022501
	v_or_b32_e32 v22, 0x400000, v129                           // 000000003460: 382d02ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003468: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v20, v19, s0                 // 00000000346c: d5207c01 00022714
	v_cmp_u_f32_e64 s0, v129, v129                             // 000000003474: d4180000 02030381
	s_wait_alu depctr_va_sdst(0)                               // 00000000347c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003480: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s0                        // 000000003484: d5010014 00022d15
	global_store_d16_hi_b16 v[0:1], v20, off                   // 00000000348c: ee09407c 0a000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000003498: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000349c: 8c7e017e
	v_mul_lo_u32 v22, s27, v36                                 // 0000000034a0: d72c0016 0202481b
	v_mul_lo_u32 v23, s26, v37                                 // 0000000034a8: d72c0017 02024a1a
	v_mad_co_u64_u32 v[0:1], null, s26, v36, 0                 // 0000000034b0: d6fe7c00 0202481a
	v_sub_co_u32 v20, s0, s24, v36                             // 0000000034b8: d7010014 02024818
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c0: bf88f19f
	v_sub_co_ci_u32_e64 v21, null, s25, v37, s0                // 0000000034c4: d5217c15 00024a19
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000034cc: bf870211
	v_cmp_lt_i64_e64 s12, 0, v[20:21]                          // 0000000034d0: d451000c 02022880
	v_add3_u32 v1, v1, v23, v22                                // 0000000034d8: d6550001 045a2f01
	s_delay_alu instid0(valu_dep_1)                            // 0000000034e0: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000034e4: 3e000081
	s_and_b32 s0, s12, s4                                      // 0000000034e8: 8b00040c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000034f0: be812000
	s_cbranch_execz 28                                         // 0000000034f4: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1a68>
	s_wait_kmcnt 0x0                                           // 0000000034f8: bfc70000
	v_add_co_u32 v23, s0, s20, v0                              // 0000000034fc: d7000017 02020014
	v_bfe_u32 v22, v126, 16, 1                                 // 000000003504: d6100016 0205217e
	s_wait_alu depctr_va_sdst(0)                               // 00000000350c: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s21, v1, s0                 // 000000003510: d5207c18 00020215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003518: bf870193
	v_add_co_u32 v4, s0, v23, v4                               // 00000000351c: d7000004 02020917
	v_add3_u32 v22, v22, v126, 0x7fff                          // 000000003524: d6550016 03fefd16 00007fff
	v_or_b32_e32 v25, 0x400000, v126                           // 000000003530: 3832fcff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003538: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v24, v5, s0                  // 00000000353c: d5207c05 00020b18
	v_cmp_u_f32_e64 s0, v126, v126                             // 000000003544: d4180000 0202fd7e
	s_wait_alu depctr_va_sdst(0)                               // 00000000354c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003550: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s0                        // 000000003554: d5010016 00023316
	global_store_d16_hi_b16 v[4:5], v22, off                   // 00000000355c: ee09407c 0b000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003568: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000356c: 8c7e017e
	v_cmp_lt_i64_e64 s9, 1, v[20:21]                           // 000000003570: d4510009 02022881
	s_and_b32 s0, s9, s4                                       // 000000003578: 8b000409
	s_wait_alu depctr_sa_sdst(0)                               // 00000000357c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003580: be812000
	s_cbranch_execz 28                                         // 000000003584: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1af8>
	v_bfe_u32 v4, v123, 16, 1                                  // 000000003588: d6100004 0205217b
	s_wait_kmcnt 0x0                                           // 000000003590: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 000000003594: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000359c: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v1, s0                 // 0000000035a0: d5207c16 00020215
	v_add3_u32 v23, v4, v123, 0x7fff                           // 0000000035a8: d6550017 03fef704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000035b4: bf870003
	v_add_co_u32 v4, s0, v5, v6                                // 0000000035b8: d7000004 02020d05
	v_or_b32_e32 v24, 0x400000, v123                           // 0000000035c0: 3830f6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v22, v7, s0                  // 0000000035cc: d5207c05 00020f16
	v_cmp_u_f32_e64 s0, v123, v123                             // 0000000035d4: d4180000 0202f77b
	s_wait_alu depctr_va_sdst(0)                               // 0000000035dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000035e0: bf870001
	v_cndmask_b32_e64 v6, v23, v24, s0                         // 0000000035e4: d5010006 00023117
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000035ec: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000035fc: 8c7e017e
	v_cmp_lt_i64_e64 s8, 2, v[20:21]                           // 000000003600: d4510008 02022882
	s_and_b32 s0, s8, s4                                       // 000000003608: 8b000408
	s_wait_alu depctr_sa_sdst(0)                               // 00000000360c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003610: be812000
	s_cbranch_execz 28                                         // 000000003614: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1b88>
	v_bfe_u32 v4, v120, 16, 1                                  // 000000003618: d6100004 02052178
	s_wait_kmcnt 0x0                                           // 000000003620: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 000000003624: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000362c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s0                  // 000000003630: d5207c06 00020215
	v_add3_u32 v7, v4, v120, 0x7fff                            // 000000003638: d6550007 03fef104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003644: bf870003
	v_add_co_u32 v4, s0, v5, v8                                // 000000003648: d7000004 02021105
	v_or_b32_e32 v22, 0x400000, v120                           // 000000003650: 382cf0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003658: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v9, s0                   // 00000000365c: d5207c05 00021306
	v_cmp_u_f32_e64 s0, v120, v120                             // 000000003664: d4180000 0202f178
	s_wait_alu depctr_va_sdst(0)                               // 00000000366c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003670: bf870001
	v_cndmask_b32_e64 v6, v7, v22, s0                          // 000000003674: d5010006 00022d07
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000367c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003688: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000368c: 8c7e017e
	v_cmp_lt_i64_e64 s7, 3, v[20:21]                           // 000000003690: d4510007 02022883
	s_and_b32 s0, s7, s4                                       // 000000003698: 8b000407
	s_wait_alu depctr_sa_sdst(0)                               // 00000000369c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000036a0: be812000
	s_cbranch_execz 28                                         // 0000000036a4: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1c18>
	v_bfe_u32 v4, v119, 16, 1                                  // 0000000036a8: d6100004 02052177
	s_wait_kmcnt 0x0                                           // 0000000036b0: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 0000000036b4: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000036bc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s0                  // 0000000036c0: d5207c06 00020215
	v_add3_u32 v7, v4, v119, 0x7fff                            // 0000000036c8: d6550007 03feef04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000036d4: bf870003
	v_add_co_u32 v4, s0, v5, v10                               // 0000000036d8: d7000004 02021505
	v_or_b32_e32 v8, 0x400000, v119                            // 0000000036e0: 3810eeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v11, s0                  // 0000000036ec: d5207c05 00021706
	v_cmp_u_f32_e64 s0, v119, v119                             // 0000000036f4: d4180000 0202ef77
	s_wait_alu depctr_va_sdst(0)                               // 0000000036fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003700: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003704: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000370c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003718: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000371c: 8c7e017e
	v_cmp_lt_i64_e64 s6, 4, v[20:21]                           // 000000003720: d4510006 02022884
	s_and_b32 s0, s6, s4                                       // 000000003728: 8b000406
	s_wait_alu depctr_sa_sdst(0)                               // 00000000372c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003730: be812000
	s_cbranch_execz 28                                         // 000000003734: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1ca8>
	v_bfe_u32 v4, v118, 16, 1                                  // 000000003738: d6100004 02052176
	s_wait_kmcnt 0x0                                           // 000000003740: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 000000003744: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000374c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s0                  // 000000003750: d5207c06 00020215
	v_add3_u32 v7, v4, v118, 0x7fff                            // 000000003758: d6550007 03feed04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003764: bf870003
	v_add_co_u32 v4, s0, v5, v12                               // 000000003768: d7000004 02021905
	v_or_b32_e32 v8, 0x400000, v118                            // 000000003770: 3810ecff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003778: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v13, s0                  // 00000000377c: d5207c05 00021b06
	v_cmp_u_f32_e64 s0, v118, v118                             // 000000003784: d4180000 0202ed76
	s_wait_alu depctr_va_sdst(0)                               // 00000000378c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003790: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003794: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000379c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000037ac: 8c7e017e
	v_cmp_lt_i64_e64 s5, 5, v[20:21]                           // 0000000037b0: d4510005 02022885
	s_and_b32 s0, s5, s4                                       // 0000000037b8: 8b000405
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000037c0: be812000
	s_cbranch_execz 28                                         // 0000000037c4: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1d38>
	v_bfe_u32 v4, v117, 16, 1                                  // 0000000037c8: d6100004 02052175
	s_wait_kmcnt 0x0                                           // 0000000037d0: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 0000000037d4: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000037dc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s0                  // 0000000037e0: d5207c06 00020215
	v_add3_u32 v7, v4, v117, 0x7fff                            // 0000000037e8: d6550007 03feeb04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000037f4: bf870003
	v_add_co_u32 v4, s0, v5, v14                               // 0000000037f8: d7000004 02021d05
	v_or_b32_e32 v8, 0x400000, v117                            // 000000003800: 3810eaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003808: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v15, s0                  // 00000000380c: d5207c05 00021f06
	v_cmp_u_f32_e64 s0, v117, v117                             // 000000003814: d4180000 0202eb75
	s_wait_alu depctr_va_sdst(0)                               // 00000000381c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003820: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 000000003824: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000382c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003838: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000383c: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[20:21]                           // 000000003840: d4510001 02022886
	s_and_b32 s0, s1, s4                                       // 000000003848: 8b000401
	s_wait_alu depctr_sa_sdst(0)                               // 00000000384c: bf88ff9e
	s_and_saveexec_b32 s19, s0                                 // 000000003850: be932000
	s_cbranch_execz 28                                         // 000000003854: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1dc8>
	v_bfe_u32 v4, v115, 16, 1                                  // 000000003858: d6100004 02052173
	s_wait_kmcnt 0x0                                           // 000000003860: bfc70000
	v_add_co_u32 v5, s0, s20, v0                               // 000000003864: d7000005 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000386c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s0                  // 000000003870: d5207c06 00020215
	v_add3_u32 v7, v4, v115, 0x7fff                            // 000000003878: d6550007 03fee704 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003884: bf870003
	v_add_co_u32 v4, s0, v5, v16                               // 000000003888: d7000004 02022105
	v_or_b32_e32 v8, 0x400000, v115                            // 000000003890: 3810e6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003898: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v17, s0                  // 00000000389c: d5207c05 00022306
	v_cmp_u_f32_e64 s0, v115, v115                             // 0000000038a4: d4180000 0202e773
	s_wait_alu depctr_va_sdst(0)                               // 0000000038ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038b0: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s0                           // 0000000038b4: d5010006 00021107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 0000000038bc: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000038cc: 8c7e137e
	v_cmp_lt_i64_e64 s0, 7, v[20:21]                           // 0000000038d0: d4510000 02022887
	s_and_b32 s4, s0, s4                                       // 0000000038d8: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038dc: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000038e0: be932004
	s_cbranch_execz 28                                         // 0000000038e4: bfa5001c <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1e58>
	v_bfe_u32 v4, v112, 16, 1                                  // 0000000038e8: d6100004 02052170
	s_wait_kmcnt 0x0                                           // 0000000038f0: bfc70000
	v_add_co_u32 v5, s4, s20, v0                               // 0000000038f4: d7000405 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000038fc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s4                  // 000000003900: d5207c06 00120215
	v_add3_u32 v7, v4, v112, 0x7fff                            // 000000003908: d6550007 03fee104 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003914: bf870003
	v_add_co_u32 v4, s4, v5, v18                               // 000000003918: d7000404 02022505
	v_or_b32_e32 v8, 0x400000, v112                            // 000000003920: 3810e0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003928: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v19, s4                  // 00000000392c: d5207c05 00122706
	v_cmp_u_f32_e64 s4, v112, v112                             // 000000003934: d4180004 0202e170
	s_wait_alu depctr_va_sdst(0)                               // 00000000393c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003940: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s4                           // 000000003944: d5010006 00121107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 00000000394c: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003958: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000395c: 8c7e137e
	s_and_b32 s4, s17, s3                                      // 000000003960: 8b040311
	s_wait_alu depctr_sa_sdst(0)                               // 000000003964: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003968: be932004
	s_cbranch_execz 40                                         // 00000000396c: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1f10>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003970: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003978: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 00000000397c: d5207c05 00102e80
	v_bfe_u32 v6, v96, 16, 1                                   // 000000003984: d6100006 02052160
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000398c: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003990: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003998: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 00000000399c: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 0000000039a4: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 0000000039a8: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 0000000039b4: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000039bc: 3e080881
	v_add3_u32 v6, v6, v96, 0x7fff                             // 0000000039c0: d6550006 03fec106 00007fff
	v_or_b32_e32 v9, 0x400000, v96                             // 0000000039cc: 3812c0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000039d4: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 0000000039d8: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000039e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000039e4: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v96, v96                               // 0000000039ec: d4180004 0202c160
	s_wait_alu depctr_va_sdst(0)                               // 0000000039f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039f8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000039fc: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003a04: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003a14: 8c7e137e
	s_and_b32 s4, s18, s3                                      // 000000003a18: 8b040312
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a1c: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003a20: be932004
	s_cbranch_execz 46                                         // 000000003a24: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x1fe0>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003a28: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003a30: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003a34: d5207c05 00102e80
	v_bfe_u32 v6, v88, 16, 1                                   // 000000003a3c: d6100006 02052158
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a44: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003a48: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003a50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003a54: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v88                             // 000000003a5c: 3812b0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a64: bf8701a3
	v_add_co_u32 v4, s4, s26, v4                               // 000000003a68: d7000404 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000003a70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s4                  // 000000003a74: d5207c05 00120a1b
	s_wait_kmcnt 0x0                                           // 000000003a7c: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003a80: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003a88: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003a8c: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003a94: 3e080881
	v_add3_u32 v6, v6, v88, 0x7fff                             // 000000003a98: d6550006 03feb106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003aa4: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003aa8: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ab0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003ab4: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v88, v88                               // 000000003abc: d4180004 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ac8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003acc: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003ad4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ae0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003ae4: 8c7e137e
	s_and_b32 s4, s16, s3                                      // 000000003ae8: 8b040310
	s_wait_alu depctr_sa_sdst(0)                               // 000000003aec: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003af0: be932004
	s_cbranch_execz 46                                         // 000000003af4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x20b0>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003af8: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003b00: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003b04: d5207c05 00102e80
	v_bfe_u32 v6, v86, 16, 1                                   // 000000003b0c: d6100006 02052156
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b14: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003b18: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003b20: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003b24: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v86                             // 000000003b2c: 3812acff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b34: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 000000003b38: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000003b40: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 000000003b44: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 000000003b4c: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003b50: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003b58: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003b5c: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003b64: 3e080881
	v_add3_u32 v6, v6, v86, 0x7fff                             // 000000003b68: d6550006 03fead06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b74: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003b78: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003b80: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003b84: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v86, v86                               // 000000003b8c: d4180004 0202ad56
	s_wait_alu depctr_va_sdst(0)                               // 000000003b94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003b98: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003b9c: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003ba4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003bb4: 8c7e137e
	s_and_b32 s4, s15, s3                                      // 000000003bb8: 8b04030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bbc: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003bc0: be932004
	s_cbranch_execz 46                                         // 000000003bc4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2180>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003bc8: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003bd4: d5207c05 00102e80
	v_bfe_u32 v6, v85, 16, 1                                   // 000000003bdc: d6100006 02052155
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003be4: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003be8: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003bf0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003bf4: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v85                             // 000000003bfc: 3812aaff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c04: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 000000003c08: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003c10: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 000000003c14: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 000000003c1c: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003c20: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003c28: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003c2c: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003c34: 3e080881
	v_add3_u32 v6, v6, v85, 0x7fff                             // 000000003c38: d6550006 03feab06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c44: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003c48: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003c50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003c54: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v85, v85                               // 000000003c5c: d4180004 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 000000003c64: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c68: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003c6c: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003c74: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003c84: 8c7e137e
	s_and_b32 s4, s14, s3                                      // 000000003c88: 8b04030e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c8c: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003c90: be932004
	s_cbranch_execz 46                                         // 000000003c94: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2250>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003c98: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003ca4: d5207c05 00102e80
	v_bfe_u32 v6, v84, 16, 1                                   // 000000003cac: d6100006 02052154
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cb4: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003cb8: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003cc4: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v84                             // 000000003ccc: 3812a8ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cd4: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 000000003cd8: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 000000003ce4: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000003cec: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003cf0: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003cfc: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003d04: 3e080881
	v_add3_u32 v6, v6, v84, 0x7fff                             // 000000003d08: d6550006 03fea906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d14: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003d18: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003d20: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003d24: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v84, v84                               // 000000003d2c: d4180004 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 000000003d34: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d38: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003d3c: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003d44: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003d54: 8c7e137e
	s_and_b32 s4, s13, s3                                      // 000000003d58: 8b04030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d5c: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003d60: be932004
	s_cbranch_execz 46                                         // 000000003d64: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2320>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003d68: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003d70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003d74: d5207c05 00102e80
	v_bfe_u32 v6, v82, 16, 1                                   // 000000003d7c: d6100006 02052152
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d84: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003d88: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003d90: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003d94: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v82                             // 000000003d9c: 3812a4ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003da4: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 000000003da8: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000003db0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000003db4: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000003dbc: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003dc0: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003dc8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003dcc: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003dd4: 3e080881
	v_add3_u32 v6, v6, v82, 0x7fff                             // 000000003dd8: d6550006 03fea506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003de4: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003de8: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003df0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003df4: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v82, v82                               // 000000003dfc: d4180004 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000003e04: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e08: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003e0c: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003e14: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e24: 8c7e137e
	s_and_b32 s4, s11, s3                                      // 000000003e28: 8b04030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e2c: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003e30: be932004
	s_cbranch_execz 46                                         // 000000003e34: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x23f0>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003e38: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003e40: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003e44: d5207c05 00102e80
	v_bfe_u32 v6, v81, 16, 1                                   // 000000003e4c: d6100006 02052151
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e54: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003e58: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003e60: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003e64: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v81                             // 000000003e6c: 3812a2ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e74: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 000000003e78: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000003e80: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000003e84: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 000000003e8c: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003e90: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003e98: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003e9c: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003ea4: 3e080881
	v_add3_u32 v6, v6, v81, 0x7fff                             // 000000003ea8: d6550006 03fea306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003eb4: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003eb8: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ec0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003ec4: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v81, v81                               // 000000003ecc: d4180004 0202a351
	s_wait_alu depctr_va_sdst(0)                               // 000000003ed4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ed8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003edc: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003ee4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ef0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003ef4: 8c7e137e
	s_and_b32 s4, s10, s3                                      // 000000003ef8: 8b04030a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003efc: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003f00: be932004
	s_cbranch_execz 46                                         // 000000003f04: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x24c0>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003f08: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003f10: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003f14: d5207c05 00102e80
	v_bfe_u32 v6, v80, 16, 1                                   // 000000003f1c: d6100006 02052150
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f24: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003f28: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000003f30: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003f34: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v80                             // 000000003f3c: 3812a0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f44: bf8701a3
	v_add_co_u32 v4, s4, s28, v4                               // 000000003f48: d7000404 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000003f50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s4                  // 000000003f54: d5207c05 00120a1d
	s_wait_kmcnt 0x0                                           // 000000003f5c: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003f60: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003f68: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003f6c: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003f74: 3e080881
	v_add3_u32 v6, v6, v80, 0x7fff                             // 000000003f78: d6550006 03fea106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f84: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003f88: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003f90: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003f94: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v80, v80                               // 000000003f9c: d4180004 0202a150
	s_wait_alu depctr_va_sdst(0)                               // 000000003fa4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003fa8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003fac: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003fb4: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fc0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003fc4: 8c7e137e
	s_and_b32 s4, s12, s3                                      // 000000003fc8: 8b04030c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fcc: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003fd0: be932004
	s_cbranch_execz 40                                         // 000000003fd4: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2578>
	v_add_co_u32 v4, s4, v39, s22                              // 000000003fd8: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000003fe0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 000000003fe4: d5207c05 00102e80
	v_bfe_u32 v6, v79, 16, 1                                   // 000000003fec: d6100006 0205214f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ff4: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000003ff8: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004000: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004004: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 00000000400c: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000004010: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004018: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 00000000401c: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004024: 3e080881
	v_add3_u32 v6, v6, v79, 0x7fff                             // 000000004028: d6550006 03fe9f06 00007fff
	v_or_b32_e32 v9, 0x400000, v79                             // 000000004034: 38129eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 00000000403c: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 000000004040: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004048: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 00000000404c: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v79, v79                               // 000000004054: d4180004 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 00000000405c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004060: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004064: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000406c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004078: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000407c: 8c7e137e
	s_and_b32 s4, s9, s3                                       // 000000004080: 8b040309
	s_wait_alu depctr_sa_sdst(0)                               // 000000004084: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000004088: be932004
	s_cbranch_execz 46                                         // 00000000408c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2648>
	v_add_co_u32 v4, s4, v39, s22                              // 000000004090: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004098: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 00000000409c: d5207c05 00102e80
	v_bfe_u32 v6, v78, 16, 1                                   // 0000000040a4: d6100006 0205214e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040ac: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 0000000040b0: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000040b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000040bc: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v78                             // 0000000040c4: 38129cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040cc: bf8701a3
	v_add_co_u32 v4, s4, s26, v4                               // 0000000040d0: d7000404 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 0000000040d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s4                  // 0000000040dc: d5207c05 00120a1b
	s_wait_kmcnt 0x0                                           // 0000000040e4: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 0000000040e8: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000040f0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 0000000040f4: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000040fc: 3e080881
	v_add3_u32 v6, v6, v78, 0x7fff                             // 000000004100: d6550006 03fe9d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000410c: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004110: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004118: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 00000000411c: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v78, v78                               // 000000004124: d4180004 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 00000000412c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004130: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004134: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000413c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004148: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000414c: 8c7e137e
	s_and_b32 s4, s8, s3                                       // 000000004150: 8b040308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004154: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000004158: be932004
	s_cbranch_execz 46                                         // 00000000415c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2718>
	v_add_co_u32 v4, s4, v39, s22                              // 000000004160: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004168: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 00000000416c: d5207c05 00102e80
	v_bfe_u32 v6, v77, 16, 1                                   // 000000004174: d6100006 0205214d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000417c: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000004180: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004188: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 00000000418c: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v77                             // 000000004194: 38129aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000419c: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 0000000041a0: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 0000000041a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 0000000041ac: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 0000000041b4: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 0000000041b8: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000041c0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 0000000041c4: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000041cc: 3e080881
	v_add3_u32 v6, v6, v77, 0x7fff                             // 0000000041d0: d6550006 03fe9b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041dc: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000041e0: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000041e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000041ec: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v77, v77                               // 0000000041f4: d4180004 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 0000000041fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004200: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004204: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000420c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004218: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000421c: 8c7e137e
	s_and_b32 s4, s7, s3                                       // 000000004220: 8b040307
	s_wait_alu depctr_sa_sdst(0)                               // 000000004224: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000004228: be932004
	s_cbranch_execz 46                                         // 00000000422c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x27e8>
	v_add_co_u32 v4, s4, v39, s22                              // 000000004230: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004238: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 00000000423c: d5207c05 00102e80
	v_bfe_u32 v6, v76, 16, 1                                   // 000000004244: d6100006 0205214c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000424c: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000004250: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004258: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 00000000425c: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v76                             // 000000004264: 381298ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000426c: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 000000004270: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004278: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 00000000427c: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 000000004284: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000004288: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004290: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004294: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000429c: 3e080881
	v_add3_u32 v6, v6, v76, 0x7fff                             // 0000000042a0: d6550006 03fe9906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042ac: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000042b0: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000042b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000042bc: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v76, v76                               // 0000000042c4: d4180004 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 0000000042cc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042d0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000042d4: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000042dc: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000042ec: 8c7e137e
	s_and_b32 s4, s6, s3                                       // 0000000042f0: 8b040306
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042f4: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000042f8: be932004
	s_cbranch_execz 46                                         // 0000000042fc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x28b8>
	v_add_co_u32 v4, s4, v39, s22                              // 000000004300: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004308: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 00000000430c: d5207c05 00102e80
	v_bfe_u32 v6, v75, 16, 1                                   // 000000004314: d6100006 0205214b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000431c: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 000000004320: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004328: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 00000000432c: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v75                             // 000000004334: 381296ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000433c: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 000000004340: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000004348: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 00000000434c: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000004354: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000004358: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004360: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004364: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000436c: 3e080881
	v_add3_u32 v6, v6, v75, 0x7fff                             // 000000004370: d6550006 03fe9706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000437c: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004380: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004388: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 00000000438c: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v75, v75                               // 000000004394: d4180004 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 00000000439c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043a0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000043a4: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000043ac: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000043bc: 8c7e137e
	s_and_b32 s4, s5, s3                                       // 0000000043c0: 8b040305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043c4: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000043c8: be932004
	s_cbranch_execz 46                                         // 0000000043cc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2988>
	v_add_co_u32 v4, s4, v39, s22                              // 0000000043d0: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000043d8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000043dc: d5207c05 00102e80
	v_bfe_u32 v6, v74, 16, 1                                   // 0000000043e4: d6100006 0205214a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000043ec: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 0000000043f0: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000043f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000043fc: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v74                             // 000000004404: 381294ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000440c: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 000000004410: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000004418: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 00000000441c: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000004424: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000004428: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004430: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004434: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000443c: 3e080881
	v_add3_u32 v6, v6, v74, 0x7fff                             // 000000004440: d6550006 03fe9506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000444c: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004450: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004458: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 00000000445c: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v74, v74                               // 000000004464: d4180004 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 00000000446c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004470: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004474: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000447c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004488: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000448c: 8c7e137e
	s_and_b32 s4, s1, s3                                       // 000000004490: 8b040301
	s_wait_alu depctr_sa_sdst(0)                               // 000000004494: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000004498: be932004
	s_cbranch_execz 46                                         // 00000000449c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2a58>
	v_add_co_u32 v4, s4, v39, s22                              // 0000000044a0: d7000404 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000044a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s4                   // 0000000044ac: d5207c05 00102e80
	v_bfe_u32 v6, v73, 16, 1                                   // 0000000044b4: d6100006 02052149
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000044bc: bf8701a3
	v_add_co_u32 v4, s4, v4, v38                               // 0000000044c0: d7000404 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000044c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000044cc: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v73                             // 0000000044d4: 381292ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000044dc: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 0000000044e0: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000044e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 0000000044ec: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 0000000044f4: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 0000000044f8: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004500: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004504: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000450c: 3e080881
	v_add3_u32 v6, v6, v73, 0x7fff                             // 000000004510: d6550006 03fe9306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000451c: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004520: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004528: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 00000000452c: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v73, v73                               // 000000004534: d4180004 02029349
	s_wait_alu depctr_va_sdst(0)                               // 00000000453c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004540: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004544: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000454c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004558: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000455c: 8c7e137e
	s_and_b32 s3, s0, s3                                       // 000000004560: 8b030300
	s_wait_alu depctr_sa_sdst(0)                               // 000000004564: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004568: be842003
	s_cbranch_execz 46                                         // 00000000456c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2b28>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004570: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004578: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 00000000457c: d5207c05 000c2e80
	v_bfe_u32 v6, v72, 16, 1                                   // 000000004584: d6100006 02052148
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000458c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004590: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004598: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 00000000459c: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v72                             // 0000000045a4: 381290ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045ac: bf8701a3
	v_add_co_u32 v4, s3, s28, v4                               // 0000000045b0: d7000304 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s3                  // 0000000045bc: d5207c05 000e0a1d
	s_wait_kmcnt 0x0                                           // 0000000045c4: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000045c8: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000045d0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 0000000045d4: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000045dc: 3e080881
	v_add3_u32 v6, v6, v72, 0x7fff                             // 0000000045e0: d6550006 03fe9106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045ec: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000045f0: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000045fc: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v72, v72                               // 000000004604: d4180003 02029148
	s_wait_alu depctr_va_sdst(0)                               // 00000000460c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004610: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004614: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 00000000461c: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004628: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000462c: 8c7e047e
	s_and_b32 s3, s17, s2                                      // 000000004630: 8b030211
	s_wait_alu depctr_sa_sdst(0)                               // 000000004634: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004638: be842003
	s_cbranch_execz 40                                         // 00000000463c: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2be0>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004640: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 00000000464c: d5207c05 000c2e80
	v_bfe_u32 v6, v71, 16, 1                                   // 000000004654: d6100006 02052147
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000465c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004660: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004668: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 00000000466c: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 000000004674: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004678: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004680: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004684: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000468c: 3e080881
	v_add3_u32 v6, v6, v71, 0x7fff                             // 000000004690: d6550006 03fe8f06 00007fff
	v_or_b32_e32 v9, 0x400000, v71                             // 00000000469c: 38128eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000046a4: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 0000000046a8: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000046b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000046b4: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v71, v71                               // 0000000046bc: d4180003 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 0000000046c4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000046c8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000046cc: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000046d4: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000046e4: 8c7e047e
	s_and_b32 s3, s18, s2                                      // 0000000046e8: 8b030212
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046ec: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000046f0: be842003
	s_cbranch_execz 46                                         // 0000000046f4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2cb0>
	v_add_co_u32 v4, s3, v39, s22                              // 0000000046f8: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004700: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004704: d5207c05 000c2e80
	v_bfe_u32 v6, v70, 16, 1                                   // 00000000470c: d6100006 02052146
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004714: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004718: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004720: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004724: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v70                             // 00000000472c: 38128cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004734: bf8701a3
	v_add_co_u32 v4, s3, s26, v4                               // 000000004738: d7000304 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000004740: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s3                  // 000000004744: d5207c05 000e0a1b
	s_wait_kmcnt 0x0                                           // 00000000474c: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004750: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004758: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 00000000475c: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004764: 3e080881
	v_add3_u32 v6, v6, v70, 0x7fff                             // 000000004768: d6550006 03fe8d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004774: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004778: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004780: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004784: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v70, v70                               // 00000000478c: d4180003 02028d46
	s_wait_alu depctr_va_sdst(0)                               // 000000004794: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004798: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 00000000479c: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000047a4: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000047b4: 8c7e047e
	s_and_b32 s3, s16, s2                                      // 0000000047b8: 8b030210
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047bc: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000047c0: be842003
	s_cbranch_execz 46                                         // 0000000047c4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2d80>
	v_add_co_u32 v4, s3, v39, s22                              // 0000000047c8: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000047d4: d5207c05 000c2e80
	v_bfe_u32 v6, v69, 16, 1                                   // 0000000047dc: d6100006 02052145
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000047e4: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 0000000047e8: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000047f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000047f4: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v69                             // 0000000047fc: 38128aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004804: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 000000004808: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000004810: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 000000004814: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 00000000481c: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004820: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004828: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 00000000482c: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004834: 3e080881
	v_add3_u32 v6, v6, v69, 0x7fff                             // 000000004838: d6550006 03fe8b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004844: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004848: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004850: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004854: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v69, v69                               // 00000000485c: d4180003 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000004864: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004868: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 00000000486c: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004874: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004880: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004884: 8c7e047e
	s_and_b32 s3, s15, s2                                      // 000000004888: 8b03020f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000488c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004890: be842003
	s_cbranch_execz 46                                         // 000000004894: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2e50>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004898: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000048a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000048a4: d5207c05 000c2e80
	v_bfe_u32 v6, v68, 16, 1                                   // 0000000048ac: d6100006 02052144
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048b4: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 0000000048b8: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000048c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000048c4: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v68                             // 0000000048cc: 381288ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048d4: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 0000000048d8: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 0000000048e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 0000000048e4: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 0000000048ec: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000048f0: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000048f8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000048fc: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004904: 3e080881
	v_add3_u32 v6, v6, v68, 0x7fff                             // 000000004908: d6550006 03fe8906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004914: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004918: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004920: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004924: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v68, v68                               // 00000000492c: d4180003 02028944
	s_wait_alu depctr_va_sdst(0)                               // 000000004934: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004938: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 00000000493c: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004944: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004950: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004954: 8c7e047e
	s_and_b32 s3, s14, s2                                      // 000000004958: 8b03020e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000495c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004960: be842003
	s_cbranch_execz 46                                         // 000000004964: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2f20>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004968: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004970: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004974: d5207c05 000c2e80
	v_bfe_u32 v6, v67, 16, 1                                   // 00000000497c: d6100006 02052143
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004984: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004988: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004990: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004994: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v67                             // 00000000499c: 381286ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049a4: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 0000000049a8: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000049b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 0000000049b4: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 0000000049bc: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000049c0: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000049c8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000049cc: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000049d4: 3e080881
	v_add3_u32 v6, v6, v67, 0x7fff                             // 0000000049d8: d6550006 03fe8706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049e4: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000049e8: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000049f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000049f4: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v67, v67                               // 0000000049fc: d4180003 02028743
	s_wait_alu depctr_va_sdst(0)                               // 000000004a04: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a08: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004a0c: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004a14: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a24: 8c7e047e
	s_and_b32 s3, s13, s2                                      // 000000004a28: 8b03020d
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a2c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004a30: be842003
	s_cbranch_execz 46                                         // 000000004a34: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x2ff0>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004a38: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004a40: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004a44: d5207c05 000c2e80
	v_bfe_u32 v6, v66, 16, 1                                   // 000000004a4c: d6100006 02052142
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a54: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004a58: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004a60: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004a64: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v66                             // 000000004a6c: 381284ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a74: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 000000004a78: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000004a80: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 000000004a84: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 000000004a8c: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004a90: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004a98: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004a9c: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004aa4: 3e080881
	v_add3_u32 v6, v6, v66, 0x7fff                             // 000000004aa8: d6550006 03fe8506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ab4: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004ab8: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004ac0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004ac4: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v66, v66                               // 000000004acc: d4180003 02028542
	s_wait_alu depctr_va_sdst(0)                               // 000000004ad4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ad8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004adc: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004ae4: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004af0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004af4: 8c7e047e
	s_and_b32 s3, s11, s2                                      // 000000004af8: 8b03020b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004afc: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004b00: be842003
	s_cbranch_execz 46                                         // 000000004b04: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x30c0>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004b08: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004b10: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004b14: d5207c05 000c2e80
	v_bfe_u32 v6, v65, 16, 1                                   // 000000004b1c: d6100006 02052141
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b24: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004b28: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004b30: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004b34: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v65                             // 000000004b3c: 381282ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b44: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 000000004b48: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000004b50: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 000000004b54: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 000000004b5c: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004b60: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004b68: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004b6c: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004b74: 3e080881
	v_add3_u32 v6, v6, v65, 0x7fff                             // 000000004b78: d6550006 03fe8306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b84: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004b88: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004b90: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004b94: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v65, v65                               // 000000004b9c: d4180003 02028341
	s_wait_alu depctr_va_sdst(0)                               // 000000004ba4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ba8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004bac: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004bb4: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bc0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004bc4: 8c7e047e
	s_and_b32 s3, s10, s2                                      // 000000004bc8: 8b03020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bcc: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004bd0: be842003
	s_cbranch_execz 46                                         // 000000004bd4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3190>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004bd8: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004be0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004be4: d5207c05 000c2e80
	v_bfe_u32 v6, v64, 16, 1                                   // 000000004bec: d6100006 02052140
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004bf4: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004bf8: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004c00: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004c04: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v64                             // 000000004c0c: 381280ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c14: bf8701a3
	v_add_co_u32 v4, s3, s28, v4                               // 000000004c18: d7000304 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000004c20: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s3                  // 000000004c24: d5207c05 000e0a1d
	s_wait_kmcnt 0x0                                           // 000000004c2c: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004c30: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004c38: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004c3c: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004c44: 3e080881
	v_add3_u32 v6, v6, v64, 0x7fff                             // 000000004c48: d6550006 03fe8106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c54: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004c58: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004c60: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004c64: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v64, v64                               // 000000004c6c: d4180003 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000004c74: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c78: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004c7c: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004c84: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c90: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004c94: 8c7e047e
	s_and_b32 s3, s12, s2                                      // 000000004c98: 8b03020c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c9c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004ca0: be842003
	s_cbranch_execz 40                                         // 000000004ca4: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3248>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004ca8: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004cb0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004cb4: d5207c05 000c2e80
	v_bfe_u32 v6, v63, 16, 1                                   // 000000004cbc: d6100006 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004cc4: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004cc8: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004cd0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004cd4: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 000000004cdc: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004ce0: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004ce8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004cec: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004cf4: 3e080881
	v_add3_u32 v6, v6, v63, 0x7fff                             // 000000004cf8: d6550006 03fe7f06 00007fff
	v_or_b32_e32 v9, 0x400000, v63                             // 000000004d04: 38127eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004d0c: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 000000004d10: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004d18: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004d1c: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v63, v63                               // 000000004d24: d4180003 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 000000004d2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d30: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004d34: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004d3c: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004d4c: 8c7e047e
	s_and_b32 s3, s9, s2                                       // 000000004d50: 8b030209
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d54: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004d58: be842003
	s_cbranch_execz 46                                         // 000000004d5c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3318>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004d60: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004d68: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004d6c: d5207c05 000c2e80
	v_bfe_u32 v6, v62, 16, 1                                   // 000000004d74: d6100006 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004d7c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004d80: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004d88: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004d8c: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v62                             // 000000004d94: 38127cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004d9c: bf8701a3
	v_add_co_u32 v4, s3, s26, v4                               // 000000004da0: d7000304 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000004da8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s3                  // 000000004dac: d5207c05 000e0a1b
	s_wait_kmcnt 0x0                                           // 000000004db4: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004db8: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004dc0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004dc4: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004dcc: 3e080881
	v_add3_u32 v6, v6, v62, 0x7fff                             // 000000004dd0: d6550006 03fe7d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ddc: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004de0: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004de8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004dec: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v62, v62                               // 000000004df4: d4180003 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000004dfc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004e00: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004e04: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004e0c: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004e1c: 8c7e047e
	s_and_b32 s3, s8, s2                                       // 000000004e20: 8b030208
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e24: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004e28: be842003
	s_cbranch_execz 46                                         // 000000004e2c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x33e8>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004e30: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004e38: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004e3c: d5207c05 000c2e80
	v_bfe_u32 v6, v61, 16, 1                                   // 000000004e44: d6100006 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e4c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004e50: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004e58: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004e5c: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v61                             // 000000004e64: 38127aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e6c: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 000000004e70: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000004e78: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 000000004e7c: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 000000004e84: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004e88: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004e90: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004e94: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004e9c: 3e080881
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000004ea0: d6550006 03fe7b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004eac: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004eb0: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004eb8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004ebc: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v61, v61                               // 000000004ec4: d4180003 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000004ecc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ed0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004ed4: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004edc: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ee8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004eec: 8c7e047e
	s_and_b32 s3, s7, s2                                       // 000000004ef0: 8b030207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ef4: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004ef8: be842003
	s_cbranch_execz 46                                         // 000000004efc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x34b8>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004f00: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004f08: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004f0c: d5207c05 000c2e80
	v_bfe_u32 v6, v60, 16, 1                                   // 000000004f14: d6100006 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f1c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004f20: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004f28: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004f2c: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v60                             // 000000004f34: 381278ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f3c: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 000000004f40: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004f48: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 000000004f4c: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 000000004f54: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004f58: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004f60: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004f64: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004f6c: 3e080881
	v_add3_u32 v6, v6, v60, 0x7fff                             // 000000004f70: d6550006 03fe7906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f7c: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004f80: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004f88: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004f8c: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v60, v60                               // 000000004f94: d4180003 0202793c
	s_wait_alu depctr_va_sdst(0)                               // 000000004f9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004fa0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004fa4: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004fac: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004fbc: 8c7e047e
	s_and_b32 s3, s6, s2                                       // 000000004fc0: 8b030206
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fc4: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004fc8: be842003
	s_cbranch_execz 46                                         // 000000004fcc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3588>
	v_add_co_u32 v4, s3, v39, s22                              // 000000004fd0: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000004fd8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 000000004fdc: d5207c05 000c2e80
	v_bfe_u32 v6, v59, 16, 1                                   // 000000004fe4: d6100006 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004fec: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000004ff0: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000004ff8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004ffc: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v59                             // 000000005004: 381276ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000500c: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 000000005010: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000005018: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 00000000501c: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 000000005024: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000005028: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005030: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000005034: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000503c: 3e080881
	v_add3_u32 v6, v6, v59, 0x7fff                             // 000000005040: d6550006 03fe7706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000504c: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000005050: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005058: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 00000000505c: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v59, v59                               // 000000005064: d4180003 0202773b
	s_wait_alu depctr_va_sdst(0)                               // 00000000506c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005070: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005074: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 00000000507c: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005088: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000508c: 8c7e047e
	s_and_b32 s3, s5, s2                                       // 000000005090: 8b030205
	s_wait_alu depctr_sa_sdst(0)                               // 000000005094: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000005098: be842003
	s_cbranch_execz 46                                         // 00000000509c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3658>
	v_add_co_u32 v4, s3, v39, s22                              // 0000000050a0: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000050a8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 0000000050ac: d5207c05 000c2e80
	v_bfe_u32 v6, v58, 16, 1                                   // 0000000050b4: d6100006 0205213a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000050bc: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 0000000050c0: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000050c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000050cc: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v58                             // 0000000050d4: 381274ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000050dc: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 0000000050e0: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 0000000050e8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 0000000050ec: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 0000000050f4: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000050f8: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005100: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000005104: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000510c: 3e080881
	v_add3_u32 v6, v6, v58, 0x7fff                             // 000000005110: d6550006 03fe7506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000511c: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000005120: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005128: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 00000000512c: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v58, v58                               // 000000005134: d4180003 0202753a
	s_wait_alu depctr_va_sdst(0)                               // 00000000513c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005140: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005144: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 00000000514c: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005158: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000515c: 8c7e047e
	s_and_b32 s3, s1, s2                                       // 000000005160: 8b030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000005164: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000005168: be842003
	s_cbranch_execz 46                                         // 00000000516c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3728>
	v_add_co_u32 v4, s3, v39, s22                              // 000000005170: d7000304 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005178: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s3                   // 00000000517c: d5207c05 000c2e80
	v_bfe_u32 v6, v57, 16, 1                                   // 000000005184: d6100006 02052139
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000518c: bf8701a3
	v_add_co_u32 v4, s3, v4, v38                               // 000000005190: d7000304 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005198: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 00000000519c: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v57                             // 0000000051a4: 381272ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051ac: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 0000000051b0: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000051b8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 0000000051bc: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 0000000051c4: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000051c8: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000051d0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 0000000051d4: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000051dc: 3e080881
	v_add3_u32 v6, v6, v57, 0x7fff                             // 0000000051e0: d6550006 03fe7306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051ec: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 0000000051f0: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000051f8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000051fc: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v57, v57                               // 000000005204: d4180003 02027339
	s_wait_alu depctr_va_sdst(0)                               // 00000000520c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005210: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005214: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 00000000521c: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005228: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000522c: 8c7e047e
	s_and_b32 s2, s0, s2                                       // 000000005230: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 000000005234: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005238: be832002
	s_cbranch_execz 46                                         // 00000000523c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x37f8>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005240: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005248: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 00000000524c: d5207c05 00082e80
	v_bfe_u32 v6, v56, 16, 1                                   // 000000005254: d6100006 02052138
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000525c: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 000000005260: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005268: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000526c: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v56                             // 000000005274: 381270ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000527c: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 000000005280: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 000000005288: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 00000000528c: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 000000005294: bfc70000
	v_add_co_u32 v7, s2, s20, v0                               // 000000005298: d7000207 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000052a0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s2                  // 0000000052a4: d5207c08 000a0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000052ac: 3e080881
	v_add3_u32 v6, v6, v56, 0x7fff                             // 0000000052b0: d6550006 03fe7106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000052bc: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000052c0: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000052c8: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000052cc: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v56, v56                               // 0000000052d4: d4180002 02027138
	s_wait_alu depctr_va_sdst(0)                               // 0000000052dc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000052e0: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000052e4: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000052ec: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000052fc: 8c7e037e
	s_and_b32 s2, s17, vcc_lo                                  // 000000005300: 8b026a11
	s_wait_alu depctr_sa_sdst(0)                               // 000000005304: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005308: be832002
	s_cbranch_execz 40                                         // 00000000530c: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x38b0>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005310: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005318: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 00000000531c: d5207c05 00082e80
	v_bfe_u32 v6, v55, 16, 1                                   // 000000005324: d6100006 02052137
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000532c: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 000000005330: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005338: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 00000000533c: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 000000005344: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005348: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005350: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005354: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 00000000535c: 3e080881
	v_add3_u32 v6, v6, v55, 0x7fff                             // 000000005360: d6550006 03fe6f06 00007fff
	v_or_b32_e32 v9, 0x400000, v55                             // 00000000536c: 38126eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000005374: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 000000005378: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005380: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005384: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v55, v55                               // 00000000538c: d4180002 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000005394: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005398: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000539c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000053a4: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000053b4: 8c7e037e
	s_and_b32 s2, s18, vcc_lo                                  // 0000000053b8: 8b026a12
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053bc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000053c0: be832002
	s_cbranch_execz 46                                         // 0000000053c4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3980>
	v_add_co_u32 v4, s2, v39, s22                              // 0000000053c8: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000053d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000053d4: d5207c05 00082e80
	v_bfe_u32 v6, v54, 16, 1                                   // 0000000053dc: d6100006 02052136
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000053e4: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 0000000053e8: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000053f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000053f4: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v54                             // 0000000053fc: 38126cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005404: bf8701a3
	v_add_co_u32 v4, s2, s26, v4                               // 000000005408: d7000204 0202081a
	s_wait_alu depctr_va_sdst(0)                               // 000000005410: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s27, v5, s2                  // 000000005414: d5207c05 000a0a1b
	s_wait_kmcnt 0x0                                           // 00000000541c: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005420: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005428: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 00000000542c: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005434: 3e080881
	v_add3_u32 v6, v6, v54, 0x7fff                             // 000000005438: d6550006 03fe6d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005444: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000005448: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005450: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005454: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v54, v54                               // 00000000545c: d4180002 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000005464: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005468: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000546c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005474: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005480: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005484: 8c7e037e
	s_and_b32 s2, s16, vcc_lo                                  // 000000005488: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 00000000548c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005490: be832002
	s_cbranch_execz 46                                         // 000000005494: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3a50>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005498: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000054a0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000054a4: d5207c05 00082e80
	v_bfe_u32 v6, v53, 16, 1                                   // 0000000054ac: d6100006 02052135
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000054b4: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 0000000054b8: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000054c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000054c4: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v53                             // 0000000054cc: 38126aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000054d4: bf8701a3
	v_add_co_u32 v4, s2, s40, v4                               // 0000000054d8: d7000204 02020828
	s_wait_alu depctr_va_sdst(0)                               // 0000000054e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s2                  // 0000000054e4: d5207c05 000a0a29
	s_wait_kmcnt 0x0                                           // 0000000054ec: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000054f0: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000054f8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000054fc: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005504: 3e080881
	v_add3_u32 v6, v6, v53, 0x7fff                             // 000000005508: d6550006 03fe6b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005514: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000005518: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005520: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005524: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v53, v53                               // 00000000552c: d4180002 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 000000005534: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005538: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000553c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005544: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005550: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005554: 8c7e037e
	s_and_b32 s2, s15, vcc_lo                                  // 000000005558: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000555c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005560: be832002
	s_cbranch_execz 46                                         // 000000005564: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3b20>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005568: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005570: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005574: d5207c05 00082e80
	v_bfe_u32 v6, v52, 16, 1                                   // 00000000557c: d6100006 02052134
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005584: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 000000005588: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005590: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005594: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v52                             // 00000000559c: 381268ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055a4: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 0000000055a8: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 0000000055b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 0000000055b4: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 0000000055bc: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000055c0: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000055c8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000055cc: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000055d4: 3e080881
	v_add3_u32 v6, v6, v52, 0x7fff                             // 0000000055d8: d6550006 03fe6906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055e4: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000055e8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000055f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000055f4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v52, v52                               // 0000000055fc: d4180002 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000005604: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005608: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000560c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005614: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005620: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005624: 8c7e037e
	s_and_b32 s2, s14, vcc_lo                                  // 000000005628: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000562c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005630: be832002
	s_cbranch_execz 46                                         // 000000005634: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3bf0>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005638: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005640: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005644: d5207c05 00082e80
	v_bfe_u32 v6, v51, 16, 1                                   // 00000000564c: d6100006 02052133
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005654: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 000000005658: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005660: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005664: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v51                             // 00000000566c: 381266ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005674: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 000000005678: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000005680: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 000000005684: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 00000000568c: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005690: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005698: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 00000000569c: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000056a4: 3e080881
	v_add3_u32 v6, v6, v51, 0x7fff                             // 0000000056a8: d6550006 03fe6706 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000056b4: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000056b8: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000056c0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000056c4: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v51, v51                               // 0000000056cc: d4180002 02026733
	s_wait_alu depctr_va_sdst(0)                               // 0000000056d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000056d8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000056dc: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000056e4: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000056f4: 8c7e037e
	s_and_b32 s2, s13, vcc_lo                                  // 0000000056f8: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056fc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005700: be832002
	s_cbranch_execz 46                                         // 000000005704: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3cc0>
	v_add_co_u32 v4, s2, v39, s22                              // 000000005708: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005710: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 000000005714: d5207c05 00082e80
	v_bfe_u32 v6, v50, 16, 1                                   // 00000000571c: d6100006 02052132
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005724: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 000000005728: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005730: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005734: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v50                             // 00000000573c: 381264ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005744: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 000000005748: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000005750: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 000000005754: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 00000000575c: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005760: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005768: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 00000000576c: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005774: 3e080881
	v_add3_u32 v6, v6, v50, 0x7fff                             // 000000005778: d6550006 03fe6506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005784: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000005788: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005790: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005794: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v50, v50                               // 00000000579c: d4180002 02026532
	s_wait_alu depctr_va_sdst(0)                               // 0000000057a4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000057a8: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000057ac: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000057b4: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057c0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000057c4: 8c7e037e
	s_and_b32 s2, s11, vcc_lo                                  // 0000000057c8: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057cc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000057d0: be832002
	s_cbranch_execz 46                                         // 0000000057d4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3d90>
	v_add_co_u32 v4, s2, v39, s22                              // 0000000057d8: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000057e0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000057e4: d5207c05 00082e80
	v_bfe_u32 v6, v49, 16, 1                                   // 0000000057ec: d6100006 02052131
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000057f4: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 0000000057f8: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 000000005800: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005804: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v49                             // 00000000580c: 381262ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005814: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 000000005818: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000005820: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 000000005824: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 00000000582c: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005830: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005838: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 00000000583c: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005844: 3e080881
	v_add3_u32 v6, v6, v49, 0x7fff                             // 000000005848: d6550006 03fe6306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005854: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 000000005858: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005860: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005864: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v49, v49                               // 00000000586c: d4180002 02026331
	s_wait_alu depctr_va_sdst(0)                               // 000000005874: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005878: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 00000000587c: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005884: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005890: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005894: 8c7e037e
	s_and_b32 s2, s10, vcc_lo                                  // 000000005898: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000589c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000058a0: be832002
	s_cbranch_execz 46                                         // 0000000058a4: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3e60>
	v_add_co_u32 v4, s2, v39, s22                              // 0000000058a8: d7000204 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 0000000058b0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s23, s2                   // 0000000058b4: d5207c05 00082e80
	v_bfe_u32 v6, v48, 16, 1                                   // 0000000058bc: d6100006 02052130
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000058c4: bf8701a3
	v_add_co_u32 v4, s2, v4, v38                               // 0000000058c8: d7000204 02024d04
	s_wait_alu depctr_va_sdst(0)                               // 0000000058d0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000058d4: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v48                             // 0000000058dc: 380e60ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000058e4: bf8701a3
	v_add_co_u32 v4, s2, s28, v4                               // 0000000058e8: d7000204 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 0000000058f0: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s29, v5, s2                  // 0000000058f4: d5207c05 000a0a1d
	s_wait_kmcnt 0x0                                           // 0000000058fc: bfc70000
	v_add_co_u32 v2, s2, s20, v2                               // 000000005900: d7000202 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005908: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s21, v3, s2                  // 00000000590c: d5207c03 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005914: 3e080881
	v_add3_u32 v6, v6, v48, 0x7fff                             // 000000005918: d6550006 03fe6106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005924: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 000000005928: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000005930: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 000000005934: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v48, v48                               // 00000000593c: d4180002 02026130
	s_wait_alu depctr_va_sdst(0)                               // 000000005944: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005948: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 00000000594c: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005954: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005960: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005964: 8c7e037e
	s_and_b32 s2, s12, vcc_lo                                  // 000000005968: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000596c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005970: be832002
	s_cbranch_execz 40                                         // 000000005974: bfa50028 <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3f18>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005978: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005980: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005984: d5207c03 00082e80
	v_bfe_u32 v4, v47, 16, 1                                   // 00000000598c: d6100004 0205212f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005994: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005998: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 0000000059a0: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000059a4: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 0000000059ac: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 0000000059b0: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000059b8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 0000000059bc: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000059c4: 3e040481
	v_add3_u32 v4, v4, v47, 0x7fff                             // 0000000059c8: d6550004 03fe5f04 00007fff
	v_or_b32_e32 v7, 0x400000, v47                             // 0000000059d4: 380e5eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000059dc: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 0000000059e0: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 0000000059e8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 0000000059ec: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v47, v47                               // 0000000059f4: d4180002 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000059fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005a00: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005a04: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005a0c: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005a1c: 8c7e037e
	s_and_b32 s2, s9, vcc_lo                                   // 000000005a20: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a24: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005a28: be832002
	s_cbranch_execz 46                                         // 000000005a2c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x3fe8>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005a30: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005a38: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005a3c: d5207c03 00082e80
	v_bfe_u32 v4, v46, 16, 1                                   // 000000005a44: d6100004 0205212e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005a4c: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005a50: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005a58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005a5c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v46                             // 000000005a64: 380e5cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005a6c: bf8701a3
	v_add_co_u32 v2, s2, s26, v2                               // 000000005a70: d7000202 0202041a
	s_wait_alu depctr_va_sdst(0)                               // 000000005a78: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s27, v3, s2                  // 000000005a7c: d5207c03 000a061b
	s_wait_kmcnt 0x0                                           // 000000005a84: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005a88: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005a90: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005a94: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005a9c: 3e040481
	v_add3_u32 v4, v4, v46, 0x7fff                             // 000000005aa0: d6550004 03fe5d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005aac: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005ab0: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005ab8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005abc: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v46, v46                               // 000000005ac4: d4180002 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005acc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005ad0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005ad4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005adc: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ae8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005aec: 8c7e037e
	s_and_b32 s2, s8, vcc_lo                                   // 000000005af0: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 000000005af4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005af8: be832002
	s_cbranch_execz 46                                         // 000000005afc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x40b8>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005b00: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005b08: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005b0c: d5207c03 00082e80
	v_bfe_u32 v4, v45, 16, 1                                   // 000000005b14: d6100004 0205212d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b1c: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005b20: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005b28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005b2c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v45                             // 000000005b34: 380e5aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b3c: bf8701a3
	v_add_co_u32 v2, s2, s40, v2                               // 000000005b40: d7000202 02020428
	s_wait_alu depctr_va_sdst(0)                               // 000000005b48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s41, v3, s2                  // 000000005b4c: d5207c03 000a0629
	s_wait_kmcnt 0x0                                           // 000000005b54: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005b58: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005b60: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005b64: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005b6c: 3e040481
	v_add3_u32 v4, v4, v45, 0x7fff                             // 000000005b70: d6550004 03fe5b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b7c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005b80: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005b88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005b8c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v45, v45                               // 000000005b94: d4180002 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 000000005b9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005ba0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005ba4: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005bac: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005bbc: 8c7e037e
	s_and_b32 s2, s7, vcc_lo                                   // 000000005bc0: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bc4: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005bc8: be832002
	s_cbranch_execz 46                                         // 000000005bcc: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x4188>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005bd0: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005bd8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005bdc: d5207c03 00082e80
	v_bfe_u32 v4, v44, 16, 1                                   // 000000005be4: d6100004 0205212c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005bec: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005bf0: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005bf8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005bfc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v44                             // 000000005c04: 380e58ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005c0c: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000005c10: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000005c18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000005c1c: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000005c24: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005c28: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005c30: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005c34: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005c3c: 3e040481
	v_add3_u32 v4, v4, v44, 0x7fff                             // 000000005c40: d6550004 03fe5904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005c4c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005c50: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005c58: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005c5c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v44, v44                               // 000000005c64: d4180002 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000005c6c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005c70: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005c74: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005c7c: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005c8c: 8c7e037e
	s_and_b32 s2, s6, vcc_lo                                   // 000000005c90: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c94: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005c98: be832002
	s_cbranch_execz 46                                         // 000000005c9c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x4258>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005ca0: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005ca8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005cac: d5207c03 00082e80
	v_bfe_u32 v4, v43, 16, 1                                   // 000000005cb4: d6100004 0205212b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005cbc: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005cc0: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005cc8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005ccc: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v43                             // 000000005cd4: 380e56ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005cdc: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000005ce0: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000005ce8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000005cec: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000005cf4: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005cf8: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005d00: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005d04: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005d0c: 3e040481
	v_add3_u32 v4, v4, v43, 0x7fff                             // 000000005d10: d6550004 03fe5704 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005d1c: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005d20: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005d28: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005d2c: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v43, v43                               // 000000005d34: d4180002 0202572b
	s_wait_alu depctr_va_sdst(0)                               // 000000005d3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005d40: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005d44: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005d4c: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005d5c: 8c7e037e
	s_and_b32 s2, s5, vcc_lo                                   // 000000005d60: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d64: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005d68: be832002
	s_cbranch_execz 46                                         // 000000005d6c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x4328>
	v_add_co_u32 v2, s2, v39, s22                              // 000000005d70: d7000202 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005d78: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s2                   // 000000005d7c: d5207c03 00082e80
	v_bfe_u32 v4, v42, 16, 1                                   // 000000005d84: d6100004 0205212a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005d8c: bf8701a3
	v_add_co_u32 v2, s2, v2, v38                               // 000000005d90: d7000202 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005d98: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005d9c: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v42                             // 000000005da4: 380e54ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005dac: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000005db0: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000005db8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000005dbc: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000005dc4: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005dc8: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005dd0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005dd4: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005ddc: 3e040481
	v_add3_u32 v4, v4, v42, 0x7fff                             // 000000005de0: d6550004 03fe5504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005dec: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005df0: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005df8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005dfc: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v42, v42                               // 000000005e04: d4180002 0202552a
	s_wait_alu depctr_va_sdst(0)                               // 000000005e0c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005e10: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005e14: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005e1c: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005e2c: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000005e30: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e34: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000005e38: be822001
	s_cbranch_execz 46                                         // 000000005e3c: bfa5002e <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x43f8>
	v_add_co_u32 v2, s1, v39, s22                              // 000000005e40: d7000102 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005e48: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s1                   // 000000005e4c: d5207c03 00042e80
	v_bfe_u32 v4, v41, 16, 1                                   // 000000005e54: d6100004 02052129
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005e5c: bf8701a3
	v_add_co_u32 v2, s1, v2, v38                               // 000000005e60: d7000102 02024d02
	s_wait_alu depctr_va_sdst(0)                               // 000000005e68: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000005e6c: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v41                             // 000000005e74: 380e52ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005e7c: bf8701a3
	v_add_co_u32 v2, s1, s30, v2                               // 000000005e80: d7000102 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000005e88: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s1                  // 000000005e8c: d5207c03 0006061f
	s_wait_kmcnt 0x0                                           // 000000005e94: bfc70000
	v_add_co_u32 v5, s1, s20, v0                               // 000000005e98: d7000105 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005ea0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s1                  // 000000005ea4: d5207c06 00060215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005eac: 3e040481
	v_add3_u32 v4, v4, v41, 0x7fff                             // 000000005eb0: d6550004 03fe5304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ebc: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000005ec0: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005ec8: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000005ecc: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v41, v41                               // 000000005ed4: d4180001 02025329
	s_wait_alu depctr_va_sdst(0)                               // 000000005edc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005ee0: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000005ee4: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005eec: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ef8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005efc: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000005f00: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f04: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005f08: be812000
	s_cbranch_execz 43                                         // 000000005f0c: bfa5002b <tessera_rocm_scaled_matmul_lds_901985de99e2d03f+0x44bc>
	v_add_co_u32 v2, s0, v39, s22                              // 000000005f10: d7000002 02002d27
	s_wait_alu depctr_va_sdst(0)                               // 000000005f18: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s23, s0                   // 000000005f1c: d5207c03 00002e80
	v_bfe_u32 v4, v40, 16, 1                                   // 000000005f24: d6100004 02052128
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005f2c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v38                           // 000000005f30: d7006a02 02024d02
	s_wait_alu depctr_va_vcc(0)                                // 000000005f38: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000005f3c: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v40                             // 000000005f44: 380a50ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005f4c: bf8701a3
	v_add_co_u32 v2, vcc_lo, s28, v2                           // 000000005f50: d7006a02 0202041c
	s_wait_alu depctr_va_vcc(0)                                // 000000005f58: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s29, v3, vcc_lo              // 000000005f5c: d5207c03 01aa061d
	s_wait_kmcnt 0x0                                           // 000000005f64: bfc70000
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 000000005f68: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000005f70: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 000000005f74: d5207c01 01aa0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005f7c: 3e040481
	v_add3_u32 v4, v4, v40, 0x7fff                             // 000000005f80: d6550004 03fe5104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005f8c: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000005f90: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000005f98: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000005f9c: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000005fa4: 7c305128
	s_wait_alu depctr_va_vcc(0)                                // 000000005fa8: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000005fac: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:96          // 000000005fb0: ee09407c 01000000 00006000
	s_nop 0                                                    // 000000005fbc: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005fc0: bfb60003
	s_endpgm                                                   // 000000005fc4: bfb00000
	s_code_end                                                 // 000000005fc8: bf9f0000
	s_code_end                                                 // 000000005fcc: bf9f0000
	s_code_end                                                 // 000000005fd0: bf9f0000
	s_code_end                                                 // 000000005fd4: bf9f0000
	s_code_end                                                 // 000000005fd8: bf9f0000
	s_code_end                                                 // 000000005fdc: bf9f0000
	s_code_end                                                 // 000000005fe0: bf9f0000
	s_code_end                                                 // 000000005fe4: bf9f0000
	s_code_end                                                 // 000000005fe8: bf9f0000
	s_code_end                                                 // 000000005fec: bf9f0000
	s_code_end                                                 // 000000005ff0: bf9f0000
	s_code_end                                                 // 000000005ff4: bf9f0000
	s_code_end                                                 // 000000005ff8: bf9f0000
	s_code_end                                                 // 000000005ffc: bf9f0000
	s_code_end                                                 // 000000006000: bf9f0000
	s_code_end                                                 // 000000006004: bf9f0000
	s_code_end                                                 // 000000006008: bf9f0000
	s_code_end                                                 // 00000000600c: bf9f0000
	s_code_end                                                 // 000000006010: bf9f0000
	s_code_end                                                 // 000000006014: bf9f0000
	s_code_end                                                 // 000000006018: bf9f0000
	s_code_end                                                 // 00000000601c: bf9f0000
	s_code_end                                                 // 000000006020: bf9f0000
	s_code_end                                                 // 000000006024: bf9f0000
	s_code_end                                                 // 000000006028: bf9f0000
	s_code_end                                                 // 00000000602c: bf9f0000
	s_code_end                                                 // 000000006030: bf9f0000
	s_code_end                                                 // 000000006034: bf9f0000
	s_code_end                                                 // 000000006038: bf9f0000
	s_code_end                                                 // 00000000603c: bf9f0000
	s_code_end                                                 // 000000006040: bf9f0000
	s_code_end                                                 // 000000006044: bf9f0000
	s_code_end                                                 // 000000006048: bf9f0000
	s_code_end                                                 // 00000000604c: bf9f0000
	s_code_end                                                 // 000000006050: bf9f0000
	s_code_end                                                 // 000000006054: bf9f0000
	s_code_end                                                 // 000000006058: bf9f0000
	s_code_end                                                 // 00000000605c: bf9f0000
	s_code_end                                                 // 000000006060: bf9f0000
	s_code_end                                                 // 000000006064: bf9f0000
	s_code_end                                                 // 000000006068: bf9f0000
	s_code_end                                                 // 00000000606c: bf9f0000
	s_code_end                                                 // 000000006070: bf9f0000
	s_code_end                                                 // 000000006074: bf9f0000
	s_code_end                                                 // 000000006078: bf9f0000
	s_code_end                                                 // 00000000607c: bf9f0000
	s_code_end                                                 // 000000006080: bf9f0000
	s_code_end                                                 // 000000006084: bf9f0000
	s_code_end                                                 // 000000006088: bf9f0000
	s_code_end                                                 // 00000000608c: bf9f0000
	s_code_end                                                 // 000000006090: bf9f0000
	s_code_end                                                 // 000000006094: bf9f0000
	s_code_end                                                 // 000000006098: bf9f0000
	s_code_end                                                 // 00000000609c: bf9f0000
	s_code_end                                                 // 0000000060a0: bf9f0000
	s_code_end                                                 // 0000000060a4: bf9f0000
	s_code_end                                                 // 0000000060a8: bf9f0000
	s_code_end                                                 // 0000000060ac: bf9f0000
	s_code_end                                                 // 0000000060b0: bf9f0000
	s_code_end                                                 // 0000000060b4: bf9f0000
	s_code_end                                                 // 0000000060b8: bf9f0000
	s_code_end                                                 // 0000000060bc: bf9f0000
	s_code_end                                                 // 0000000060c0: bf9f0000
	s_code_end                                                 // 0000000060c4: bf9f0000
	s_code_end                                                 // 0000000060c8: bf9f0000
	s_code_end                                                 // 0000000060cc: bf9f0000
	s_code_end                                                 // 0000000060d0: bf9f0000
	s_code_end                                                 // 0000000060d4: bf9f0000
	s_code_end                                                 // 0000000060d8: bf9f0000
	s_code_end                                                 // 0000000060dc: bf9f0000
	s_code_end                                                 // 0000000060e0: bf9f0000
	s_code_end                                                 // 0000000060e4: bf9f0000
	s_code_end                                                 // 0000000060e8: bf9f0000
	s_code_end                                                 // 0000000060ec: bf9f0000
	s_code_end                                                 // 0000000060f0: bf9f0000
	s_code_end                                                 // 0000000060f4: bf9f0000
	s_code_end                                                 // 0000000060f8: bf9f0000
	s_code_end                                                 // 0000000060fc: bf9f0000
	s_code_end                                                 // 000000006100: bf9f0000
	s_code_end                                                 // 000000006104: bf9f0000
	s_code_end                                                 // 000000006108: bf9f0000
	s_code_end                                                 // 00000000610c: bf9f0000
	s_code_end                                                 // 000000006110: bf9f0000
	s_code_end                                                 // 000000006114: bf9f0000
	s_code_end                                                 // 000000006118: bf9f0000
	s_code_end                                                 // 00000000611c: bf9f0000
	s_code_end                                                 // 000000006120: bf9f0000
	s_code_end                                                 // 000000006124: bf9f0000
	s_code_end                                                 // 000000006128: bf9f0000
	s_code_end                                                 // 00000000612c: bf9f0000
	s_code_end                                                 // 000000006130: bf9f0000
	s_code_end                                                 // 000000006134: bf9f0000
	s_code_end                                                 // 000000006138: bf9f0000
	s_code_end                                                 // 00000000613c: bf9f0000
	s_code_end                                                 // 000000006140: bf9f0000
	s_code_end                                                 // 000000006144: bf9f0000
	s_code_end                                                 // 000000006148: bf9f0000
	s_code_end                                                 // 00000000614c: bf9f0000
	s_code_end                                                 // 000000006150: bf9f0000
	s_code_end                                                 // 000000006154: bf9f0000
	s_code_end                                                 // 000000006158: bf9f0000
	s_code_end                                                 // 00000000615c: bf9f0000
	s_code_end                                                 // 000000006160: bf9f0000
	s_code_end                                                 // 000000006164: bf9f0000
	s_code_end                                                 // 000000006168: bf9f0000
	s_code_end                                                 // 00000000616c: bf9f0000
	s_code_end                                                 // 000000006170: bf9f0000
	s_code_end                                                 // 000000006174: bf9f0000
	s_code_end                                                 // 000000006178: bf9f0000
	s_code_end                                                 // 00000000617c: bf9f0000
