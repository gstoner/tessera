
/tmp/tmp8kpbfa7u.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_ed0a211cd1c74a4f>:
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b00: f4002080 f80000d8
	v_lshrrev_b32_e32 v6, 3, v0                                // 000000001b08: 320c0083
	s_clause 0x4                                               // 000000001b0c: bf850004
	s_load_b64 s[16:17], s[0:1], 0x8                           // 000000001b10: f4002400 f8000008
	s_load_b64 s[14:15], s[0:1], 0x30                          // 000000001b18: f4002380 f8000030
	s_load_b64 s[8:9], s[0:1], 0x58                            // 000000001b20: f4002200 f8000058
	s_load_b64 s[24:25], s[0:1], 0x80                          // 000000001b28: f4002600 f8000080
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b30: f4004500 f80000c8
	v_lshlrev_b32_e32 v3, 4, v0                                // 000000001b38: 30060084
	s_mov_b32 s6, ttmp7                                        // 000000001b3c: be860073
	v_lshrrev_b32_e32 v8, 1, v0                                // 000000001b40: 32100081
	v_or_b32_e32 v2, 0x60, v6                                  // 000000001b44: 38040cff 00000060
	s_ashr_i32 s7, ttmp7, 31                                   // 000000001b4c: 86079f73
	v_or_b32_e32 v4, 32, v6                                    // 000000001b50: 38080ca0
	v_and_b32_e32 v13, 0x70, v3                                // 000000001b54: 361a06ff 00000070
	s_lshl_b64 s[6:7], s[6:7], 7                               // 000000001b5c: 84868706
	v_mul_u32_u24_e32 v5, 0x90, v2                             // 000000001b60: 160a04ff 00000090
	v_or_b32_e32 v3, s6, v6                                    // 000000001b68: 38060c06
	s_mov_b32 s12, ttmp9                                       // 000000001b6c: be8c0075
	s_ashr_i32 s13, ttmp9, 31                                  // 000000001b70: 860d9f75
	v_dual_mov_b32 v50, 0 :: v_dual_and_b32 v9, 0x60, v8       // 000000001b74: ca240080 320810ff 00000060
	v_add_nc_u32_e32 v56, v5, v13                              // 000000001b80: 4a701b05
	v_or_b32_e32 v5, s6, v4                                    // 000000001b84: 380a0806
	s_lshl_b64 s[4:5], s[12:13], 7                             // 000000001b88: 8484870c
	v_or_b32_e32 v7, 64, v6                                    // 000000001b8c: 380e0cc0
	v_or_b32_e32 v14, s4, v2                                   // 000000001b90: 381c0404
	v_or_b32_e32 v18, s6, v2                                   // 000000001b94: 38240406
	s_wait_kmcnt 0x0                                           // 000000001b98: bfc70000
	v_mul_lo_u32 v21, s3, v3                                   // 000000001b9c: d72c0015 02020603
	v_mad_co_u64_u32 v[2:3], null, s2, v3, s[16:17]            // 000000001ba4: d6fe7c02 00420602
	v_or_b32_e32 v10, 16, v9                                   // 000000001bac: 38141290
	v_or_b32_e32 v16, s4, v4                                   // 000000001bb0: 38200804
	v_mul_u32_u24_e32 v20, 0x90, v4                            // 000000001bb4: 162808ff 00000090
	v_mul_lo_u32 v22, s3, v5                                   // 000000001bbc: d72c0016 02020a03
	v_mad_co_u64_u32 v[4:5], null, s2, v5, s[16:17]            // 000000001bc4: d6fe7c04 00420a02
	v_or_b32_e32 v15, s4, v7                                   // 000000001bcc: 381e0e04
	v_or_b32_e32 v17, s4, v6                                   // 000000001bd0: 38220c04
	v_mul_u32_u24_e32 v19, 0x90, v7                            // 000000001bd4: 16260eff 00000090
	v_mul_u32_u24_e32 v6, 0x90, v6                             // 000000001bdc: 160c0cff 00000090
	v_or_b32_e32 v7, s6, v7                                    // 000000001be4: 380e0e06
	v_or_b32_e32 v1, s6, v10                                   // 000000001be8: 38021406
	v_or_b32_e32 v12, s6, v9                                   // 000000001bec: 38181206
	s_mul_i32 s6, s2, s7                                       // 000000001bf0: 96060702
	v_add_nc_u32_e32 v59, v19, v13                             // 000000001bf4: 4a761b13
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bf8: bf88ff9e
	v_add3_u32 v3, v21, v3, s6                                 // 000000001bfc: d6550003 001a0715
	v_dual_mov_b32 v148, 0 :: v_dual_add_nc_u32 v61, v20, v13  // 000000001c04: ca200080 943c1b14
	v_add_nc_u32_e32 v62, v6, v13                              // 000000001c0c: 4a7c1b06
	v_add3_u32 v19, v22, v5, s6                                // 000000001c10: d6550013 001a0b16
	v_mul_lo_u32 v20, s3, v7                                   // 000000001c18: d72c0014 02020e03
	v_mad_co_u64_u32 v[5:6], null, s2, v7, s[16:17]            // 000000001c20: d6fe7c05 00420e02
	v_add_co_u32 v64, vcc_lo, v2, v13                          // 000000001c28: d7006a40 02021b02
	s_delay_alu instid0(valu_dep_1)                            // 000000001c30: bf870001
	v_add_co_ci_u32_e64 v65, null, 0, v3, vcc_lo               // 000000001c34: d5207c41 01aa0680
	v_add_co_u32 v66, vcc_lo, v4, v13                          // 000000001c3c: d7006a42 02021b04
	s_wait_alu depctr_va_vcc(0)                                // 000000001c44: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, 0, v19, vcc_lo              // 000000001c48: d5207c43 01aa2680
	v_mul_lo_u32 v19, s3, v18                                  // 000000001c50: d72c0013 02022403
	v_mad_co_u64_u32 v[2:3], null, s2, v18, s[16:17]           // 000000001c58: d6fe7c02 00422402
	v_add3_u32 v4, v20, v6, s6                                 // 000000001c60: d6550004 001a0d14
	v_add_co_u32 v70, vcc_lo, v5, v13                          // 000000001c68: d7006a46 02021b05
	v_mul_lo_u32 v18, s3, v17                                  // 000000001c70: d72c0012 02022203
	v_mad_co_u64_u32 v[6:7], null, s2, v16, s[14:15]           // 000000001c78: d6fe7c06 003a2002
	s_wait_alu depctr_va_vcc(0)                                // 000000001c80: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, 0, v4, vcc_lo               // 000000001c84: d5207c47 01aa0880
	v_mad_co_u64_u32 v[4:5], null, s2, v17, s[14:15]           // 000000001c8c: d6fe7c04 003a2202
	v_add3_u32 v3, v19, v3, s6                                 // 000000001c94: d6550003 001a0713
	v_mul_lo_u32 v17, s3, v16                                  // 000000001c9c: d72c0011 02022003
	v_add_co_u32 v74, vcc_lo, v2, v13                          // 000000001ca4: d7006a4a 02021b02
	v_mul_lo_u32 v16, s3, v15                                  // 000000001cac: d72c0010 02021e03
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb4: bf88ff9d
	v_add_co_ci_u32_e64 v76, null, 0, v3, vcc_lo               // 000000001cb8: d5207c4c 01aa0680
	v_mad_co_u64_u32 v[2:3], null, s2, v15, s[14:15]           // 000000001cc0: d6fe7c02 003a1e02
	s_mul_i32 s6, s2, s5                                       // 000000001cc8: 96060502
	v_add_co_u32 v78, vcc_lo, v4, v13                          // 000000001ccc: d7006a4e 02021b04
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cd4: bf88ff9e
	v_add3_u32 v5, v18, v5, s6                                 // 000000001cd8: d6550005 001a0b12
	v_add3_u32 v7, v17, v7, s6                                 // 000000001ce0: d6550007 001a0f11
	v_mov_b32_e32 v33, s7                                      // 000000001ce8: 7e420207
	v_lshlrev_b32_e32 v11, 1, v0                               // 000000001cec: 30160081
	v_and_b32_e32 v0, 15, v0                                   // 000000001cf0: 3600008f
	s_wait_alu depctr_va_vcc(0)                                // 000000001cf4: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, 0, v5, vcc_lo               // 000000001cf8: d5207c4f 01aa0a80
	v_add3_u32 v5, v16, v3, s6                                 // 000000001d00: d6550005 001a0710
	v_add_co_u32 v80, vcc_lo, v6, v13                          // 000000001d08: d7006a50 02021b06
	v_mad_co_u64_u32 v[3:4], null, s2, v14, s[14:15]           // 000000001d10: d6fe7c03 003a1c02
	s_wait_alu depctr_va_vcc(0)                                // 000000001d18: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v7, vcc_lo               // 000000001d1c: d5207c51 01aa0e80
	v_add_co_u32 v83, vcc_lo, v2, v13                          // 000000001d24: d7006a53 02021b02
	s_wait_alu depctr_va_vcc(0)                                // 000000001d2c: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, 0, v5, vcc_lo               // 000000001d30: d5207c54 01aa0a80
	v_or_b32_e32 v2, v9, v0                                    // 000000001d38: 38040109
	v_or_b32_e32 v5, v10, v0                                   // 000000001d3c: 380a010a
	v_and_or_b32 v0, v11, 64, v0                               // 000000001d40: d6570000 0401810b
	v_mul_lo_u32 v6, s3, v14                                   // 000000001d48: d72c0006 02021c03
	v_and_b32_e32 v10, 8, v8                                   // 000000001d50: 36141088
	v_mul_u32_u24_e32 v2, 0x90, v2                             // 000000001d54: 160404ff 00000090
	v_add_co_u32 v87, vcc_lo, v3, v13                          // 000000001d5c: d7006a57 02021b03
	v_or_b32_e32 v11, 16, v0                                   // 000000001d64: 38160090
	v_or_b32_e32 v13, 32, v0                                   // 000000001d68: 381a00a0
	v_or_b32_e32 v32, v12, v10                                 // 000000001d6c: 3840150c
	v_add3_u32 v4, v6, v4, s6                                  // 000000001d70: d6550004 001a0906
	v_or_b32_e32 v90, v2, v10                                  // 000000001d78: 38b41502
	v_mul_u32_u24_e32 v2, 0x90, v11                            // 000000001d7c: 160416ff 00000090
	v_or_b32_e32 v14, 48, v0                                   // 000000001d84: 381c00b0
	v_mul_u32_u24_e32 v3, 0x90, v13                            // 000000001d88: 16061aff 00000090
	v_or_b32_e32 v15, 1, v10                                   // 000000001d90: 381e1481
	s_wait_alu depctr_va_vcc(0)                                // 000000001d94: bf88ff9d
	v_add_co_ci_u32_e64 v88, null, 0, v4, vcc_lo               // 000000001d98: d5207c58 01aa0880
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[32:33]                // 000000001da0: 7ca84014
	v_mul_u32_u24_e32 v5, 0x90, v5                             // 000000001da4: 160a0aff 00000090
	s_add_nc_u64 s[14:15], s[22:23], 0x7f                      // 000000001dac: a98eff16 0000007f
	v_mul_u32_u24_e32 v4, 0x90, v14                            // 000000001db4: 16081cff 00000090
	v_or_b32_e32 v94, v2, v10                                  // 000000001dbc: 38bc1502
	v_or_b32_e32 v95, v3, v10                                  // 000000001dc0: 38be1503
	v_or_b32_e32 v2, v15, v12                                  // 000000001dc4: 3804190f
	v_mov_b32_e32 v3, s7                                       // 000000001dc8: 7e060207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001dcc: bf88ff9e
	s_lshr_b64 s[28:29], s[14:15], 7                           // 000000001dd0: 859c870e
	s_mov_b32 s10, ttmp9                                       // 000000001dd4: be8a0075
	s_add_nc_u64 s[14:15], s[28:29], -1                        // 000000001dd8: a98ec11c
	s_and_b32 s11, s13, 0x1ffffff                              // 000000001ddc: 8b0bff0d 01ffffff
	v_or_b32_e32 v91, v5, v10                                  // 000000001de4: 38b61505
	v_or_b32_e32 v97, v4, v10                                  // 000000001de8: 38c21504
	s_wait_alu depctr_va_vcc(0)                                // 000000001dec: bf88ff9d
	v_dual_cndmask_b32 v4, 0, v33 :: v_dual_cndmask_b32 v5, 0, v32// 000000001df0: ca524280 04044080
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000001df8: 7ca80414
	s_lshr_b64 s[26:27], s[2:3], 7                             // 000000001dfc: 859a8702
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e00: bf88ff9e
	v_cmp_lt_u64_e64 s2, s[10:11], s[14:15]                    // 000000001e04: d4590002 02001c0a
	v_mul_u32_u24_e32 v6, 0x90, v0                             // 000000001e0c: 160c00ff 00000090
	v_or_b32_e32 v34, s4, v0                                   // 000000001e14: 38440004
	v_mul_lo_u32 v8, s26, v4                                   // 000000001e18: d72c0008 0202081a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e20: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v2, vcc_lo                        // 000000001e24: 020e0480
	v_or_b32_e32 v16, 2, v10                                   // 000000001e28: 38201482
	s_and_b32 s2, s2, exec_lo                                  // 000000001e2c: 8b027e02
	s_cselect_b32 s11, s11, s15                                // 000000001e30: 980b0f0b
	s_cselect_b32 s10, ttmp9, s14                              // 000000001e34: 980a0e75
	s_lshr_b32 s12, s3, 7                                      // 000000001e38: 850c8703
	v_or_b32_e32 v92, v6, v10                                  // 000000001e3c: 38b81506
	v_mul_lo_u32 v0, s12, v5                                   // 000000001e40: d72c0000 02020a0c
	v_cndmask_b32_e32 v6, 0, v3, vcc_lo                        // 000000001e48: 020c0680
	v_mad_co_u64_u32 v[2:3], null, s26, v5, 0                  // 000000001e4c: d6fe7c02 02020a1a
	v_or_b32_e32 v4, v16, v12                                  // 000000001e54: 38081910
	v_dual_mov_b32 v5, s7 :: v_dual_mov_b32 v144, 0            // 000000001e58: ca100007 05900080
	v_or_b32_e32 v19, 3, v10                                   // 000000001e60: 38261483
	v_mul_lo_u32 v17, s12, v7                                  // 000000001e64: d72c0011 02020e0c
	v_mul_lo_u32 v18, s26, v6                                  // 000000001e6c: d72c0012 02020c1a
	s_delay_alu instid0(valu_dep_4)                            // 000000001e74: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000001e78: 7ca80814
	v_mad_co_u64_u32 v[6:7], null, s26, v7, 0                  // 000000001e7c: d6fe7c06 02020e1a
	v_add3_u32 v3, v3, v8, v0                                  // 000000001e84: d6550003 04021103
	v_or_b32_e32 v8, v19, v12                                  // 000000001e8c: 38101913
	v_dual_mov_b32 v9, s7 :: v_dual_mov_b32 v136, 0            // 000000001e90: ca100007 09880080
	s_wait_alu depctr_va_vcc(0)                                // 000000001e98: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v5, vcc_lo                        // 000000001e9c: 02000a80
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000001ea0: 02080880
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]                        // 000000001ea4: 3e040482
	v_cmp_gt_i64_e64 s2, s[20:21], v[8:9]                      // 000000001ea8: d4540002 02021014
	v_add3_u32 v7, v7, v18, v17                                // 000000001eb0: d6550007 04462507
	v_or_b32_e32 v18, 4, v10                                   // 000000001eb8: 38241484
	v_mul_lo_u32 v17, s12, v4                                  // 000000001ebc: d72c0011 0202080c
	v_mul_lo_u32 v0, s26, v0                                   // 000000001ec4: d72c0000 0202001a
	v_mad_co_u64_u32 v[4:5], null, s26, v4, 0                  // 000000001ecc: d6fe7c04 0202081a
	s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)             // 000000001ed4: bf88f19e
	v_cndmask_b32_e64 v21, 0, v8, s2                           // 000000001ed8: d5010015 000a1080
	v_or_b32_e32 v8, v18, v12                                  // 000000001ee0: 38101912
	v_cndmask_b32_e64 v20, 0, v9, s2                           // 000000001ee4: d5010014 000a1280
	v_add_co_u32 v110, s2, s8, v2                              // 000000001eec: d700026e 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000001ef4: bf88f19f
	v_add_co_ci_u32_e64 v111, null, s9, v3, s2                 // 000000001ef8: d5207c6f 000a0609
	v_cmp_gt_i64_e64 s2, s[20:21], v[8:9]                      // 000000001f00: d4540002 02021014
	v_add3_u32 v5, v5, v0, v17                                 // 000000001f08: d6550005 04460105
	v_or_b32_e32 v17, 5, v10                                   // 000000001f10: 38221485
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001f14: 3e040c82
	v_mul_lo_u32 v0, s12, v21                                  // 000000001f18: d72c0000 02022a0c
	v_mul_lo_u32 v20, s26, v20                                 // 000000001f20: d72c0014 0202281a
	v_mad_co_u64_u32 v[6:7], null, s26, v21, 0                 // 000000001f28: d6fe7c06 02022a1a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f30: bf88f19f
	v_cndmask_b32_e64 v22, 0, v8, s2                           // 000000001f34: d5010016 000a1080
	v_or_b32_e32 v8, v17, v12                                  // 000000001f3c: 38101911
	v_cndmask_b32_e64 v21, 0, v9, s2                           // 000000001f40: d5010015 000a1280
	v_add_co_u32 v114, s2, s8, v2                              // 000000001f48: d7000272 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000001f50: bf88f19f
	v_add_co_ci_u32_e64 v116, null, s9, v3, s2                 // 000000001f54: d5207c74 000a0609
	v_cmp_gt_i64_e64 s2, s[20:21], v[8:9]                      // 000000001f5c: d4540002 02021014
	v_add3_u32 v7, v7, v20, v0                                 // 000000001f64: d6550007 04022907
	v_or_b32_e32 v20, 6, v10                                   // 000000001f6c: 38281486
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 000000001f70: 3e040882
	v_mul_lo_u32 v0, s12, v22                                  // 000000001f74: d72c0000 02022c0c
	v_mul_lo_u32 v21, s26, v21                                 // 000000001f7c: d72c0015 02022a1a
	v_mad_co_u64_u32 v[4:5], null, s26, v22, 0                 // 000000001f84: d6fe7c04 02022c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f8c: bf88f19f
	v_cndmask_b32_e64 v23, 0, v8, s2                           // 000000001f90: d5010017 000a1080
	v_or_b32_e32 v8, v20, v12                                  // 000000001f98: 38101914
	v_cndmask_b32_e64 v22, 0, v9, s2                           // 000000001f9c: d5010016 000a1280
	v_add_co_u32 v117, s2, s8, v2                              // 000000001fa4: d7000275 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000001fac: bf88f19f
	v_add_co_ci_u32_e64 v118, null, s9, v3, s2                 // 000000001fb0: d5207c76 000a0609
	v_cmp_gt_i64_e64 s2, s[20:21], v[8:9]                      // 000000001fb8: d4540002 02021014
	v_add3_u32 v5, v5, v21, v0                                 // 000000001fc0: d6550005 04022b05
	v_or_b32_e32 v21, 7, v10                                   // 000000001fc8: 382a1487
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000001fcc: 3e040c82
	v_mul_lo_u32 v0, s12, v23                                  // 000000001fd0: d72c0000 02022e0c
	v_mul_lo_u32 v22, s26, v22                                 // 000000001fd8: d72c0016 02022c1a
	s_wait_alu depctr_va_sdst(0)                               // 000000001fe0: bf88f19f
	v_cndmask_b32_e64 v24, 0, v8, s2                           // 000000001fe4: d5010018 000a1080
	v_or_b32_e32 v8, v21, v12                                  // 000000001fec: 38101915
	v_mad_co_u64_u32 v[6:7], null, s26, v23, 0                 // 000000001ff0: d6fe7c06 02022e1a
	v_cndmask_b32_e64 v23, 0, v9, s2                           // 000000001ff8: d5010017 000a1280
	v_add_co_u32 v120, s2, s8, v2                              // 000000002000: d7000278 02020408
	s_wait_alu depctr_va_sdst(0)                               // 000000002008: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s9, v3, s2                 // 00000000200c: d5207c79 000a0609
	v_cmp_gt_i64_e64 s2, s[20:21], v[8:9]                      // 000000002014: d4540002 02021014
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 00000000201c: 3e040882
	v_mul_lo_u32 v12, s12, v24                                 // 000000002020: d72c000c 0202300c
	v_mul_lo_u32 v23, s26, v23                                 // 000000002028: d72c0017 02022e1a
	v_mad_co_u64_u32 v[4:5], null, s26, v24, 0                 // 000000002030: d6fe7c04 0202301a
	v_add3_u32 v7, v7, v22, v0                                 // 000000002038: d6550007 04022d07
	s_wait_alu depctr_va_sdst(0)                               // 000000002040: bf88f19f
	v_cndmask_b32_e64 v0, 0, v9, s2                            // 000000002044: d5010000 000a1280
	v_cndmask_b32_e64 v8, 0, v8, s2                            // 00000000204c: d5010008 000a1080
	v_add_co_u32 v122, s2, s8, v2                              // 000000002054: d700027a 02020408
	s_wait_alu depctr_va_sdst(0)                               // 00000000205c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s9, v3, s2                 // 000000002060: d5207c7b 000a0609
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000002068: 3e040c82
	v_mul_lo_u32 v9, s12, v8                                   // 00000000206c: d72c0009 0202100c
	v_mul_lo_u32 v0, s26, v0                                   // 000000002074: d72c0000 0202001a
	v_mad_co_u64_u32 v[6:7], null, s26, v8, 0                  // 00000000207c: d6fe7c06 0202101a
	v_add3_u32 v5, v5, v23, v12                                // 000000002084: d6550005 04322f05
	v_dual_mov_b32 v37, s7 :: v_dual_mov_b32 v112, 0           // 00000000208c: ca100007 25700080
	v_or_b32_e32 v36, v1, v10                                  // 000000002094: 38481501
	v_add_co_u32 v126, s2, s8, v2                              // 000000002098: d700027e 02020408
	s_delay_alu instid0(valu_dep_4)                            // 0000000020a0: bf870004
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 0000000020a4: 3e080882
	v_add3_u32 v7, v7, v0, v9                                  // 0000000020a8: d6550007 04260107
	s_wait_alu depctr_va_sdst(0)                               // 0000000020b0: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s9, v3, s2                 // 0000000020b4: d5207c7f 000a0609
	v_cmp_gt_i64_e64 s2, s[20:21], v[36:37]                    // 0000000020bc: d4540002 02024814
	v_mov_b32_e32 v108, 0                                      // 0000000020c4: 7ed80280
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 0000000020c8: 3e040c82
	v_add_co_u32 v129, s3, s8, v4                              // 0000000020cc: d7000381 02020808
	s_wait_alu depctr_va_sdst(0)                               // 0000000020d4: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s9, v5, s3                 // 0000000020d8: d5207c82 000e0a09
	v_mov_b32_e32 v5, s5                                       // 0000000020e0: 7e0a0205
	v_or_b32_e32 v4, s4, v11                                   // 0000000020e4: 38081604
	v_cndmask_b32_e64 v0, 0, v37, s2                           // 0000000020e8: d5010000 000a4a80
	v_cndmask_b32_e64 v8, 0, v36, s2                           // 0000000020f0: d5010008 000a4880
	v_dual_mov_b32 v7, s7 :: v_dual_mov_b32 v106, 0            // 0000000020f8: ca100007 076a0080
	v_or_b32_e32 v6, v1, v15                                   // 000000002100: 380c1f01
	v_add_co_u32 v132, s2, s8, v2                              // 000000002104: d7000284 02020408
	s_wait_alu depctr_va_sdst(0)                               // 00000000210c: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s9, v3, s2                 // 000000002110: d5207c85 000a0609
	v_cmp_gt_i64_e64 s2, s[22:23], v[4:5]                      // 000000002118: d4540002 02020816
	v_mul_lo_u32 v10, s12, v8                                  // 000000002120: d72c000a 0202100c
	v_mul_lo_u32 v0, s26, v0                                   // 000000002128: d72c0000 0202001a
	v_mad_co_u64_u32 v[4:5], null, s26, v8, 0                  // 000000002130: d6fe7c04 0202101a
	v_cmp_gt_i64_e64 s3, s[20:21], v[6:7]                      // 000000002138: d4540003 02020c14
	v_dual_mov_b32 v3, s5 :: v_dual_mov_b32 v102, 0            // 000000002140: ca100005 03660080
	v_or_b32_e32 v2, s4, v13                                   // 000000002148: 38041a04
	v_dual_mov_b32 v35, s5 :: v_dual_mov_b32 v140, 0           // 00000000214c: ca100005 238c0080
	s_wait_alu depctr_va_sdst(0)                               // 000000002154: bf88f19f
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_4)// 000000002158: bf870254
	v_cndmask_b32_e64 v12, 0, v6, s3                           // 00000000215c: d501000c 000e0c80
	v_or_b32_e32 v6, v1, v16                                   // 000000002164: 380c2101
	v_add3_u32 v5, v5, v0, v10                                 // 000000002168: d6550005 042a0105
	v_cndmask_b32_e64 v11, 0, v7, s3                           // 000000002170: d501000b 000e0e80
	v_dual_mov_b32 v9, s5 :: v_dual_mov_b32 v86, 0             // 000000002178: ca100005 09560080
	v_cmp_gt_i64_e64 s5, s[20:21], v[6:7]                      // 000000002180: d4540005 02020c14
	v_cmp_gt_i64_e64 s3, s[22:23], v[2:3]                      // 000000002188: d4540003 02020416
	v_lshlrev_b64_e32 v[2:3], 2, v[4:5]                        // 000000002190: 3e040882
	v_dual_mov_b32 v5, s7 :: v_dual_mov_b32 v82, 0             // 000000002194: ca100007 05520080
	v_or_b32_e32 v4, v1, v19                                   // 00000000219c: 38082701
	v_mul_lo_u32 v0, s12, v12                                  // 0000000021a0: d72c0000 0202180c
	v_mul_lo_u32 v13, s26, v11                                 // 0000000021a8: d72c000d 0202161a
	v_mad_co_u64_u32 v[10:11], null, s26, v12, 0               // 0000000021b0: d6fe7c0a 0202181a
	s_wait_alu depctr_va_sdst(0)                               // 0000000021b8: bf88f19f
	v_cndmask_b32_e64 v7, 0, v7, s5                            // 0000000021bc: d5010007 00160e80
	v_cndmask_b32_e64 v6, 0, v6, s5                            // 0000000021c4: d5010006 00160c80
	v_cmp_gt_i64_e64 s5, s[20:21], v[4:5]                      // 0000000021cc: d4540005 02020814
	v_or_b32_e32 v8, s4, v14                                   // 0000000021d4: 38101c04
	v_add_co_u32 v138, s6, s8, v2                              // 0000000021d8: d700068a 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000021e0: bf88f19f
	v_add_co_ci_u32_e64 v139, null, s9, v3, s6                 // 0000000021e4: d5207c8b 001a0609
	s_delay_alu instid0(valu_dep_4)                            // 0000000021ec: bf870004
	v_cndmask_b32_e64 v12, 0, v4, s5                           // 0000000021f0: d501000c 00160880
	v_or_b32_e32 v4, v1, v18                                   // 0000000021f8: 38082501
	v_cmp_gt_i64_e64 s4, s[22:23], v[8:9]                      // 0000000021fc: d4540004 02021016
	v_add3_u32 v11, v11, v13, v0                               // 000000002204: d655000b 04021b0b
	v_mul_lo_u32 v0, s12, v6                                   // 00000000220c: d72c0000 02020c0c
	v_mul_lo_u32 v8, s26, v7                                   // 000000002214: d72c0008 02020e1a
	v_mad_co_u64_u32 v[6:7], null, s26, v6, 0                  // 00000000221c: d6fe7c06 02020c1a
	v_cndmask_b32_e64 v9, 0, v5, s5                            // 000000002224: d5010009 00160a80
	v_cmp_gt_i64_e64 s5, s[20:21], v[4:5]                      // 00000000222c: d4540005 02020814
	v_lshlrev_b64_e32 v[2:3], 2, v[10:11]                      // 000000002234: 3e041482
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v149, 0            // 000000002238: ca100080 44940080
	v_mov_b32_e32 v54, 0                                       // 000000002240: 7e6c0280
	v_mul_lo_u32 v10, s26, v9                                  // 000000002244: d72c000a 0202121a
	v_add3_u32 v7, v7, v8, v0                                  // 00000000224c: d6550007 04021107
	v_mul_lo_u32 v0, s12, v12                                  // 000000002254: d72c0000 0202180c
	v_mad_co_u64_u32 v[8:9], null, s26, v12, 0                 // 00000000225c: d6fe7c08 0202181a
	s_wait_alu depctr_va_sdst(0)                               // 000000002264: bf88f19f
	v_cndmask_b32_e64 v12, 0, v4, s5                           // 000000002268: d501000c 00160880
	v_or_b32_e32 v4, v1, v17                                   // 000000002270: 38082301
	v_add_co_u32 v142, s6, s8, v2                              // 000000002274: d700068e 02020408
	v_cndmask_b32_e64 v11, 0, v5, s5                           // 00000000227c: d501000b 00160a80
	s_wait_alu depctr_va_sdst(0)                               // 000000002284: bf88f19f
	v_add_co_ci_u32_e64 v143, null, s9, v3, s6                 // 000000002288: d5207c8f 001a0609
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000002290: 3e040c82
	v_cmp_gt_i64_e64 s5, s[20:21], v[4:5]                      // 000000002294: d4540005 02020814
	v_add3_u32 v9, v9, v10, v0                                 // 00000000229c: d6550009 04021509
	v_mul_lo_u32 v0, s12, v12                                  // 0000000022a4: d72c0000 0202180c
	v_mul_lo_u32 v13, s26, v11                                 // 0000000022ac: d72c000d 0202161a
	v_mad_co_u64_u32 v[6:7], null, s26, v12, 0                 // 0000000022b4: d6fe7c06 0202181a
	v_mov_b32_e32 v11, s7                                      // 0000000022bc: 7e160207
	v_or_b32_e32 v10, v1, v20                                  // 0000000022c0: 38142901
	v_add_co_u32 v146, s6, s8, v2                              // 0000000022c4: d7000692 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000022cc: bf88f19f
	v_add_co_ci_u32_e64 v147, null, s9, v3, s6                 // 0000000022d0: d5207c93 001a0609
	v_lshlrev_b64_e32 v[2:3], 2, v[8:9]                        // 0000000022d8: 3e041082
	v_cndmask_b32_e64 v8, 0, v4, s5                            // 0000000022dc: d5010008 00160880
	v_or_b32_e32 v4, v1, v21                                   // 0000000022e4: 38082b01
	v_cmp_gt_i64_e64 s6, s[20:21], v[10:11]                    // 0000000022e8: d4540006 02021414
	v_add3_u32 v7, v7, v13, v0                                 // 0000000022f0: d6550007 04021b07
	v_cndmask_b32_e64 v0, 0, v5, s5                            // 0000000022f8: d5010000 00160a80
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v52, 0            // 000000002300: ca100080 91340080
	v_cmp_gt_i64_e64 s5, s[20:21], v[4:5]                      // 000000002308: d4540005 02020814
	s_wait_alu depctr_va_sdst(0)                               // 000000002310: bf88f19f
	v_cndmask_b32_e64 v9, 0, v11, s6                           // 000000002314: d5010009 001a1680
	v_mul_lo_u32 v11, s12, v8                                  // 00000000231c: d72c000b 0202100c
	v_mul_lo_u32 v12, s26, v0                                  // 000000002324: d72c000c 0202001a
	v_mad_co_u64_u32 v[0:1], null, s26, v8, 0                  // 00000000232c: d6fe7c00 0202101a
	v_cndmask_b32_e64 v10, 0, v10, s6                          // 000000002334: d501000a 001a1480
	v_cndmask_b32_e64 v5, 0, v5, s5                            // 00000000233c: d5010005 00160a80
	v_cndmask_b32_e64 v4, 0, v4, s5                            // 000000002344: d5010004 00160880
	v_mul_lo_u32 v14, s26, v9                                  // 00000000234c: d72c000e 0202121a
	v_add_co_u32 v150, s5, s8, v2                              // 000000002354: d7000596 02020408
	v_mul_lo_u32 v13, s12, v10                                 // 00000000235c: d72c000d 0202140c
	v_mad_co_u64_u32 v[8:9], null, s26, v10, 0                 // 000000002364: d6fe7c08 0202141a
	s_wait_alu depctr_va_sdst(0)                               // 00000000236c: bf88f19f
	v_add_co_ci_u32_e64 v151, null, s9, v3, s5                 // 000000002370: d5207c97 00160609
	v_add3_u32 v1, v1, v12, v11                                // 000000002378: d6550001 042e1901
	v_lshlrev_b64_e32 v[2:3], 2, v[6:7]                        // 000000002380: 3e040c82
	v_mul_lo_u32 v6, s12, v4                                   // 000000002384: d72c0006 0202080c
	v_mul_lo_u32 v7, s26, v5                                   // 00000000238c: d72c0007 02020a1a
	v_mad_co_u64_u32 v[4:5], null, s26, v4, 0                  // 000000002394: d6fe7c04 0202081a
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 00000000239c: 3e000082
	v_add3_u32 v9, v9, v14, v13                                // 0000000023a0: d6550009 04361d09
	v_add_co_u32 v152, s5, s8, v2                              // 0000000023a8: d7000598 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000023b0: bf88f19f
	v_add_co_ci_u32_e64 v153, null, s9, v3, s5                 // 0000000023b4: d5207c99 00160609
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 0000000023bc: bf8701d3
	v_lshlrev_b64_e32 v[2:3], 2, v[8:9]                        // 0000000023c0: 3e041082
	v_add3_u32 v5, v5, v7, v6                                  // 0000000023c4: d6550005 041a0f05
	v_add_co_u32 v154, s5, s8, v0                              // 0000000023cc: d700059a 02020008
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d4: bf88f19f
	v_add_co_ci_u32_e64 v155, null, s9, v1, s5                 // 0000000023d8: d5207c9b 00160209
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 0000000023e0: 3e000882
	v_add_co_u32 v156, s5, s8, v2                              // 0000000023e4: d700059c 02020408
	s_wait_alu depctr_va_sdst(0)                               // 0000000023ec: bf88f19f
	v_add_co_ci_u32_e64 v157, null, s9, v3, s5                 // 0000000023f0: d5207c9d 00160609
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v46, 0            // 0000000023f8: ca100080 8d2e0080
	s_delay_alu instid0(valu_dep_4)                            // 000000002400: bf870004
	v_add_co_u32 v158, s5, s8, v0                              // 000000002404: d700059e 02020008
	s_wait_alu depctr_va_sdst(0)                               // 00000000240c: bf88f19f
	v_add_co_ci_u32_e64 v159, null, s9, v1, s5                 // 000000002410: d5207c9f 00160209
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v134, 0           // 000000002418: ca100080 73860080
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v128, 0           // 000000002420: ca100080 71800080
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v124, 0           // 000000002428: ca100080 6d7c0080
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v104, 0           // 000000002430: ca100080 69680080
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v100, 0            // 000000002438: ca100080 5d640080
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v98, 0             // 000000002440: ca100080 59620080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v96, 0             // 000000002448: ca100080 55600080
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v72, 0             // 000000002450: ca100080 4b480080
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v60, 0             // 000000002458: ca100080 493c0080
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v58, 0             // 000000002460: ca100080 353a0080
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v48, 0             // 000000002468: ca100080 33300080
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v44, 0             // 000000002470: ca100080 312c0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v42, 0             // 000000002478: ca100080 2f2a0080
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v40, 0             // 000000002480: ca100080 2b280080
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v38, 0            // 000000002488: ca100080 89260080
	v_mov_b32_e32 v135, 0                                      // 000000002490: 7f0e0280
	v_mov_b32_e32 v131, 0                                      // 000000002494: 7f060280
	v_mov_b32_e32 v125, 0                                      // 000000002498: 7efa0280
	v_mov_b32_e32 v119, 0                                      // 00000000249c: 7eee0280
	v_mov_b32_e32 v107, 0                                      // 0000000024a0: 7ed60280
	v_mov_b32_e32 v103, 0                                      // 0000000024a4: 7ece0280
	v_mov_b32_e32 v101, 0                                      // 0000000024a8: 7eca0280
	v_mov_b32_e32 v99, 0                                       // 0000000024ac: 7ec60280
	v_mov_b32_e32 v77, 0                                       // 0000000024b0: 7e9a0280
	v_mov_b32_e32 v69, 0                                       // 0000000024b4: 7e8a0280
	v_mov_b32_e32 v63, 0                                       // 0000000024b8: 7e7e0280
	v_mov_b32_e32 v57, 0                                       // 0000000024bc: 7e720280
	v_mov_b32_e32 v55, 0                                       // 0000000024c0: 7e6e0280
	v_mov_b32_e32 v45, 0                                       // 0000000024c4: 7e5a0280
	v_mov_b32_e32 v41, 0                                       // 0000000024c8: 7e520280
	v_mov_b32_e32 v39, 0                                       // 0000000024cc: 7e4e0280
	s_mov_b64 s[30:31], 0                                      // 0000000024d0: be9e0180
	s_lshl_b64 s[34:35], s[10:11], 2                           // 0000000024d4: 84a2820a
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[34:35]                // 0000000024d8: 7ca84416
	s_lshl_b64 s[12:13], s[30:31], 7                           // 0000000024dc: 848c871e
	s_lshl_b64 s[20:21], s[30:31], 2                           // 0000000024e0: 8494821e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024e4: bf88ff9e
	v_add_co_u32 v6, s8, v74, s12                              // 0000000024e8: d7000806 0200194a
	v_add_co_u32 v8, s9, v78, s12                              // 0000000024f0: d7000908 0200194e
	v_add_co_u32 v4, s7, v70, s12                              // 0000000024f8: d7000704 02001946
	s_wait_alu depctr_va_sdst(0)                               // 000000002500: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s13, v76, s8                 // 000000002504: d5207c07 0022980d
	v_add_co_u32 v20, s10, v80, s12                            // 00000000250c: d7000a14 02001950
	v_add_co_ci_u32_e64 v9, null, s13, v79, s9                 // 000000002514: d5207c09 00269e0d
	v_add_co_u32 v0, s5, v64, s12                              // 00000000251c: d7000500 02001940
	v_add_co_u32 v2, s6, v66, s12                              // 000000002524: d7000602 02001942
	v_add_co_u32 v24, s11, v83, s12                            // 00000000252c: d7000b18 02001953
	v_add_co_u32 v28, s12, v87, s12                            // 000000002534: d7000c1c 02001957
	v_add_co_ci_u32_e64 v5, null, s13, v71, s7                 // 00000000253c: d5207c05 001e8e0d
	s_wait_alu depctr_va_sdst(0)                               // 000000002544: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s13, v81, s10               // 000000002548: d5207c15 002aa20d
	v_add_co_ci_u32_e64 v1, null, s13, v65, s5                 // 000000002550: d5207c01 0016820d
	v_add_co_ci_u32_e64 v3, null, s13, v67, s6                 // 000000002558: d5207c03 001a860d
	v_add_co_ci_u32_e64 v25, null, s13, v84, s11               // 000000002560: d5207c19 002ea80d
	v_add_co_ci_u32_e64 v29, null, s13, v88, s12               // 000000002568: d5207c1d 0032b00d
	global_load_b128 v[12:15], v[6:7], off                     // 000000002570: ee05c07c 0000000c 00000006
	global_load_b128 v[16:19], v[8:9], off                     // 00000000257c: ee05c07c 00000010 00000008
	global_load_b128 v[8:11], v[4:5], off                      // 000000002588: ee05c07c 00000008 00000004
	global_load_b128 v[20:23], v[20:21], off                   // 000000002594: ee05c07c 00000014 00000014
	global_load_b128 v[4:7], v[2:3], off                       // 0000000025a0: ee05c07c 00000004 00000002
	global_load_b128 v[24:27], v[24:25], off                   // 0000000025ac: ee05c07c 00000018 00000018
	global_load_b128 v[0:3], v[0:1], off                       // 0000000025b8: ee05c07c 00000000 00000000
	global_load_b128 v[28:31], v[28:29], off                   // 0000000025c4: ee05c07c 0000001c 0000001c
	s_mul_u64 s[6:7], s[30:31], s[28:29]                       // 0000000025d0: aa861c1e
	v_add_co_u32 v164, s5, v110, s20                           // 0000000025d4: d70005a4 0200296e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025dc: bf88ff9e
	s_lshl_b64 s[36:37], s[6:7], 2                             // 0000000025e0: 84a48206
	v_add_co_u32 v166, s6, v114, s20                           // 0000000025e4: d70006a6 02002972
	v_add_co_u32 v168, s7, v117, s20                           // 0000000025ec: d70007a8 02002975
	v_add_co_ci_u32_e64 v165, null, s21, v111, s5              // 0000000025f4: d5207ca5 0016de15
	v_add_co_u32 v170, s8, v120, s20                           // 0000000025fc: d70008aa 02002978
	s_wait_alu depctr_va_sdst(0)                               // 000000002604: bf88f19f
	v_add_co_ci_u32_e64 v167, null, s21, v116, s6              // 000000002608: d5207ca7 001ae815
	v_add_co_u32 v172, s9, v122, s20                           // 000000002610: d70009ac 0200297a
	v_add_co_ci_u32_e64 v169, null, s21, v118, s7              // 000000002618: d5207ca9 001eec15
	v_add_co_u32 v174, s10, v126, s20                          // 000000002620: d7000aae 0200297e
	v_add_co_u32 v176, s11, v129, s20                          // 000000002628: d7000bb0 02002981
	v_add_co_u32 v178, s12, v132, s20                          // 000000002630: d7000cb2 02002984
	v_add_co_u32 v180, s13, v138, s20                          // 000000002638: d7000db4 0200298a
	v_add_co_u32 v182, s14, v142, s20                          // 000000002640: d7000eb6 0200298e
	v_add_co_u32 v184, s15, v146, s20                          // 000000002648: d7000fb8 02002992
	v_add_co_u32 v186, s16, v150, s20                          // 000000002650: d70010ba 02002996
	v_add_co_u32 v188, s17, v152, s20                          // 000000002658: d70011bc 02002998
	v_add_co_u32 v190, s18, v154, s20                          // 000000002660: d70012be 0200299a
	v_add_co_u32 v192, s19, v156, s20                          // 000000002668: d70013c0 0200299c
	v_add_co_u32 v194, s20, v158, s20                          // 000000002670: d70014c2 0200299e
	v_add_co_ci_u32_e64 v171, null, s21, v121, s8              // 000000002678: d5207cab 0022f215
	s_wait_alu depctr_va_sdst(0)                               // 000000002680: bf88f19f
	v_add_co_ci_u32_e64 v173, null, s21, v123, s9              // 000000002684: d5207cad 0026f615
	v_add_co_ci_u32_e64 v175, null, s21, v127, s10             // 00000000268c: d5207caf 002afe15
	v_add_co_ci_u32_e64 v177, null, s21, v130, s11             // 000000002694: d5207cb1 002f0415
	v_add_co_ci_u32_e64 v179, null, s21, v133, s12             // 00000000269c: d5207cb3 00330a15
	v_add_co_ci_u32_e64 v181, null, s21, v139, s13             // 0000000026a4: d5207cb5 00371615
	v_add_co_ci_u32_e64 v183, null, s21, v143, s14             // 0000000026ac: d5207cb7 003b1e15
	v_add_co_ci_u32_e64 v185, null, s21, v147, s15             // 0000000026b4: d5207cb9 003f2615
	v_add_co_ci_u32_e64 v187, null, s21, v151, s16             // 0000000026bc: d5207cbb 00432e15
	v_add_co_ci_u32_e64 v189, null, s21, v153, s17             // 0000000026c4: d5207cbd 00473215
	v_add_co_ci_u32_e64 v191, null, s21, v155, s18             // 0000000026cc: d5207cbf 004b3615
	v_add_co_ci_u32_e64 v193, null, s21, v157, s19             // 0000000026d4: d5207cc1 004f3a15
	v_add_co_ci_u32_e64 v195, null, s21, v159, s20             // 0000000026dc: d5207cc3 00533e15
	v_add_nc_u32_e32 v163, 0x4800, v92                         // 0000000026e4: 4b46b8ff 00004800
	v_add_nc_u32_e32 v162, 0x4800, v94                         // 0000000026ec: 4b44bcff 00004800
	v_add_nc_u32_e32 v161, 0x4800, v95                         // 0000000026f4: 4b42beff 00004800
	v_add_nc_u32_e32 v160, 0x4800, v97                         // 0000000026fc: 4b40c2ff 00004800
	s_add_nc_u64 s[6:7], s[24:25], s[36:37]                    // 000000002704: a9862418
	s_add_nc_u64 s[30:31], s[30:31], 1                         // 000000002708: a99e811e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000270c: bf88ff9e
	s_add_nc_u64 s[8:9], s[6:7], s[34:35]                      // 000000002710: a9882206
	s_cmp_lg_u64 s[30:31], s[26:27]                            // 000000002714: bf111a1e
	s_barrier_signal -1                                        // 000000002718: be804ec1
	s_barrier_wait 0xffff                                      // 00000000271c: bf94ffff
	s_wait_loadcnt 0x7                                         // 000000002720: bfc00007
	ds_store_b128 v62, v[12:15] offset:13824                   // 000000002724: db7c3600 00000c3e
	s_wait_loadcnt 0x6                                         // 00000000272c: bfc00006
	ds_store_b128 v62, v[16:19] offset:18432                   // 000000002730: db7c4800 0000103e
	s_wait_loadcnt 0x5                                         // 000000002738: bfc00005
	ds_store_b128 v62, v[8:11] offset:9216                     // 00000000273c: db7c2400 0000083e
	s_wait_loadcnt 0x4                                         // 000000002744: bfc00004
	ds_store_b128 v61, v[20:23] offset:18432                   // 000000002748: db7c4800 0000143d
	s_wait_loadcnt 0x3                                         // 000000002750: bfc00003
	ds_store_b128 v62, v[4:7] offset:4608                      // 000000002754: db7c1200 0000043e
	s_wait_loadcnt 0x2                                         // 00000000275c: bfc00002
	ds_store_b128 v59, v[24:27] offset:18432                   // 000000002760: db7c4800 0000183b
	s_wait_loadcnt 0x1                                         // 000000002768: bfc00001
	ds_store_b128 v62, v[0:3]                                  // 00000000276c: db7c0000 0000003e
	s_wait_loadcnt 0x0                                         // 000000002774: bfc00000
	ds_store_b128 v56, v[28:31] offset:18432                   // 000000002778: db7c4800 00001c38
	s_wait_dscnt 0x0                                           // 000000002780: bfc60000
	s_barrier_signal -1                                        // 000000002784: be804ec1
	s_barrier_wait 0xffff                                      // 000000002788: bf94ffff
	s_clause 0xf                                               // 00000000278c: bf85000f
	global_load_b32 v220, v[164:165], off                      // 000000002790: ee05007c 000000dc 000000a4
	global_load_b32 v221, v[166:167], off                      // 00000000279c: ee05007c 000000dd 000000a6
	global_load_b32 v222, v[168:169], off                      // 0000000027a8: ee05007c 000000de 000000a8
	global_load_b32 v223, v[170:171], off                      // 0000000027b4: ee05007c 000000df 000000aa
	global_load_b32 v224, v[172:173], off                      // 0000000027c0: ee05007c 000000e0 000000ac
	global_load_b32 v225, v[174:175], off                      // 0000000027cc: ee05007c 000000e1 000000ae
	global_load_b32 v226, v[176:177], off                      // 0000000027d8: ee05007c 000000e2 000000b0
	global_load_b32 v227, v[178:179], off                      // 0000000027e4: ee05007c 000000e3 000000b2
	global_load_b32 v228, v[180:181], off                      // 0000000027f0: ee05007c 000000e4 000000b4
	global_load_b32 v229, v[182:183], off                      // 0000000027fc: ee05007c 000000e5 000000b6
	global_load_b32 v230, v[184:185], off                      // 000000002808: ee05007c 000000e6 000000b8
	global_load_b32 v231, v[186:187], off                      // 000000002814: ee05007c 000000e7 000000ba
	global_load_b32 v232, v[188:189], off                      // 000000002820: ee05007c 000000e8 000000bc
	global_load_b32 v233, v[190:191], off                      // 00000000282c: ee05007c 000000e9 000000be
	global_load_b32 v234, v[192:193], off                      // 000000002838: ee05007c 000000ea 000000c0
	global_load_b32 v235, v[194:195], off                      // 000000002844: ee05007c 000000eb 000000c2
	ds_load_2addr_b64 v[178:181], v90 offset1:2                // 000000002850: d9dc0200 b200005a
	ds_load_2addr_b64 v[182:185], v163 offset1:2               // 000000002858: d9dc0200 b60000a3
	ds_load_2addr_b64 v[186:189], v162 offset1:2               // 000000002860: d9dc0200 ba0000a2
	ds_load_2addr_b64 v[190:193], v161 offset1:2               // 000000002868: d9dc0200 be0000a1
	ds_load_2addr_b64 v[196:199], v160 offset1:2               // 000000002870: d9dc0200 c40000a0
	ds_load_2addr_b64 v[200:203], v91 offset1:2                // 000000002878: d9dc0200 c800005b
	ds_load_2addr_b64 v[204:207], v90 offset0:4 offset1:6      // 000000002880: d9dc0604 cc00005a
	ds_load_2addr_b64 v[208:211], v163 offset0:4 offset1:6     // 000000002888: d9dc0604 d00000a3
	ds_load_2addr_b64 v[212:215], v162 offset0:4 offset1:6     // 000000002890: d9dc0604 d40000a2
	s_clause 0x1                                               // 000000002898: bf850001
	s_load_b32 s5, s[8:9], 0x0                                 // 00000000289c: f4000144 f8000000
	s_load_b32 s6, s[6:7], 0x0                                 // 0000000028a4: f4000183 f8000000
	s_wait_dscnt 0x7                                           // 0000000028ac: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[178:179], v[182:183], 0// 0000000028b0: cc464000 1a036db2
	s_wait_dscnt 0x6                                           // 0000000028b8: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[178:179], v[186:187], 0// 0000000028bc: cc464008 1a0375b2
	s_wait_dscnt 0x5                                           // 0000000028c4: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[178:179], v[190:191], 0// 0000000028c8: cc464010 1a037db2
	s_wait_dscnt 0x4                                           // 0000000028d0: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[178:179], v[196:197], 0// 0000000028d4: cc464018 1a0389b2
	s_wait_dscnt 0x3                                           // 0000000028dc: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[200:201], v[182:183], 0// 0000000028e0: cc4640a4 1a036dc8
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[200:201], v[186:187], 0// 0000000028e8: cc4640ac 1a0375c8
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[180:181], v[184:185], v[0:7]// 0000000028f0: cc464000 1c0371b4
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[180:181], v[188:189], v[8:15]// 0000000028f8: cc464008 1c2379b4
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[180:181], v[192:193], v[16:23]// 000000002900: cc464010 1c4381b4
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[180:181], v[198:199], v[24:31]// 000000002908: cc464018 1c638db4
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[202:203], v[184:185], v[164:171]// 000000002910: cc4640a4 1e9371ca
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[200:201], v[190:191], 0// 000000002918: cc4640b4 1a037dc8
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[202:203], v[188:189], v[172:179]// 000000002920: cc4640ac 1eb379ca
	s_wait_dscnt 0x1                                           // 000000002928: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[204:205], v[208:209], v[0:7]// 00000000292c: cc464000 1c03a1cc
	s_wait_dscnt 0x0                                           // 000000002934: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[204:205], v[212:213], v[8:15]// 000000002938: cc464008 1c23a9cc
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[202:203], v[192:193], v[180:187]// 000000002940: cc4640b4 1ed381ca
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[200:201], v[196:197], 0// 000000002948: cc4640bc 1a0389c8
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[206:207], v[210:211], v[0:7]// 000000002950: cc464000 1c03a5ce
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000002958: bf870194
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[206:207], v[214:215], v[8:15]// 00000000295c: cc464008 1c23adce
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[202:203], v[198:199], v[188:195]// 000000002964: cc4640bc 1ef38dca
	ds_load_2addr_b64 v[196:199], v161 offset0:4 offset1:6     // 00000000296c: d9dc0604 c40000a1
	ds_load_2addr_b64 v[200:203], v160 offset0:4 offset1:6     // 000000002974: d9dc0604 c80000a0
	s_wait_dscnt 0x1                                           // 00000000297c: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[204:205], v[196:197], v[16:23]// 000000002980: cc464010 1c4389cc
	s_wait_dscnt 0x0                                           // 000000002988: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[204:205], v[200:201], v[24:31]// 00000000298c: cc464018 1c6391cc
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002994: bf870112
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[206:207], v[198:199], v[16:23]// 000000002998: cc464010 1c438dce
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[206:207], v[202:203], v[24:31]// 0000000029a0: cc464018 1c6395ce
	ds_load_2addr_b64 v[204:207], v91 offset0:4 offset1:6      // 0000000029a8: d9dc0604 cc00005b
	s_wait_dscnt 0x0                                           // 0000000029b0: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[204:205], v[208:209], v[164:171]// 0000000029b4: cc4640a4 1e93a1cc
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[204:205], v[212:213], v[172:179]// 0000000029bc: cc4640ac 1eb3a9cc
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[204:205], v[196:197], v[180:187]// 0000000029c4: cc4640b4 1ed389cc
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[204:205], v[200:201], v[188:195]// 0000000029cc: cc4640bc 1ef391cc
	s_delay_alu instid0(valu_dep_4)                            // 0000000029d4: bf870004
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[206:207], v[210:211], v[164:171]// 0000000029d8: cc4640a4 1e93a5ce
	ds_load_2addr_b64 v[208:211], v90 offset0:8 offset1:10     // 0000000029e0: d9dc0a08 d000005a
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[206:207], v[214:215], v[172:179]// 0000000029e8: cc4640ac 1eb3adce
	ds_load_2addr_b64 v[212:215], v163 offset0:8 offset1:10    // 0000000029f0: d9dc0a08 d40000a3
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[206:207], v[198:199], v[180:187]// 0000000029f8: cc4640b4 1ed38dce
	ds_load_2addr_b64 v[196:199], v162 offset0:8 offset1:10    // 000000002a00: d9dc0a08 c40000a2
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[206:207], v[202:203], v[188:195]// 000000002a08: cc4640bc 1ef395ce
	ds_load_2addr_b64 v[200:203], v161 offset0:8 offset1:10    // 000000002a10: d9dc0a08 c80000a1
	ds_load_2addr_b64 v[204:207], v160 offset0:8 offset1:10    // 000000002a18: d9dc0a08 cc0000a0
	s_wait_dscnt 0x3                                           // 000000002a20: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[208:209], v[212:213], v[0:7]// 000000002a24: cc464000 1c03a9d0
	s_wait_dscnt 0x2                                           // 000000002a2c: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[208:209], v[196:197], v[8:15]// 000000002a30: cc464008 1c2389d0
	s_wait_dscnt 0x1                                           // 000000002a38: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[208:209], v[200:201], v[16:23]// 000000002a3c: cc464010 1c4391d0
	s_wait_dscnt 0x0                                           // 000000002a44: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[208:209], v[204:205], v[24:31]// 000000002a48: cc464018 1c6399d0
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[210:211], v[214:215], v[0:7]// 000000002a50: cc464000 1c03add2
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[210:211], v[198:199], v[8:15]// 000000002a58: cc464008 1c238dd2
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[210:211], v[202:203], v[16:23]// 000000002a60: cc464010 1c4395d2
	s_delay_alu instid0(valu_dep_4)                            // 000000002a68: bf870004
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[210:211], v[206:207], v[24:31]// 000000002a6c: cc464018 1c639dd2
	ds_load_2addr_b64 v[208:211], v91 offset0:8 offset1:10     // 000000002a74: d9dc0a08 d000005b
	s_wait_dscnt 0x0                                           // 000000002a7c: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[208:209], v[212:213], v[164:171]// 000000002a80: cc4640a4 1e93a9d0
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[208:209], v[196:197], v[172:179]// 000000002a88: cc4640ac 1eb389d0
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[208:209], v[200:201], v[180:187]// 000000002a90: cc4640b4 1ed391d0
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[208:209], v[204:205], v[188:195]// 000000002a98: cc4640bc 1ef399d0
	s_wait_kmcnt 0x0                                           // 000000002aa0: bfc70000
	v_mov_b32_e32 v208, s5                                     // 000000002aa4: 7fa00205
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[210:211], v[214:215], v[164:171]// 000000002aa8: cc4640a4 1e93add2
	ds_load_2addr_b64 v[212:215], v90 offset0:12 offset1:14    // 000000002ab0: d9dc0e0c d400005a
	ds_load_2addr_b64 v[216:219], v163 offset0:12 offset1:14   // 000000002ab8: d9dc0e0c d80000a3
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[210:211], v[198:199], v[172:179]// 000000002ac0: cc4640ac 1eb38dd2
	ds_load_2addr_b64 v[196:199], v162 offset0:12 offset1:14   // 000000002ac8: d9dc0e0c c40000a2
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[210:211], v[202:203], v[180:187]// 000000002ad0: cc4640b4 1ed395d2
	ds_load_2addr_b64 v[200:203], v161 offset0:12 offset1:14   // 000000002ad8: d9dc0e0c c80000a1
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[210:211], v[206:207], v[188:195]// 000000002ae0: cc4640bc 1ef39dd2
	ds_load_2addr_b64 v[160:163], v160 offset0:12 offset1:14   // 000000002ae8: d9dc0e0c a00000a0
	ds_load_2addr_b64 v[204:207], v91 offset0:12 offset1:14    // 000000002af0: d9dc0e0c cc00005b
	v_cndmask_b32_e32 v209, s6, v208, vcc_lo                   // 000000002af8: 03a3a006
	s_wait_dscnt 0x4                                           // 000000002afc: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[212:213], v[216:217], v[0:7]// 000000002b00: cc464000 1c03b1d4
	s_wait_dscnt 0x3                                           // 000000002b08: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[212:213], v[196:197], v[8:15]// 000000002b0c: cc464008 1c2389d4
	s_wait_dscnt 0x2                                           // 000000002b14: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[212:213], v[200:201], v[16:23]// 000000002b18: cc464010 1c4391d4
	s_wait_dscnt 0x1                                           // 000000002b20: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[212:213], v[160:161], v[24:31]// 000000002b24: cc464018 1c6341d4
	s_wait_dscnt 0x0                                           // 000000002b2c: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[204:205], v[160:161], v[188:195]// 000000002b30: cc4640bc 1ef341cc
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[214:215], v[218:219], v[0:7]// 000000002b38: cc464000 1c03b5d6
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[204:205], v[216:217], v[164:171]// 000000002b40: cc4640a4 1e93b1cc
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[204:205], v[196:197], v[172:179]// 000000002b48: cc4640ac 1eb389cc
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[204:205], v[200:201], v[180:187]// 000000002b50: cc4640b4 1ed391cc
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[214:215], v[198:199], v[8:15]// 000000002b58: cc464008 1c238dd6
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[214:215], v[202:203], v[16:23]// 000000002b60: cc464010 1c4395d6
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[214:215], v[162:163], v[24:31]// 000000002b68: cc464018 1c6345d6
	v_wmma_f32_16x16x16_fp8_fp8 v[164:171], v[206:207], v[218:219], v[164:171]// 000000002b70: cc4640a4 1e93b5ce
	v_wmma_f32_16x16x16_fp8_fp8 v[172:179], v[206:207], v[198:199], v[172:179]// 000000002b78: cc4640ac 1eb38dce
	v_wmma_f32_16x16x16_fp8_fp8 v[180:187], v[206:207], v[202:203], v[180:187]// 000000002b80: cc4640b4 1ed395ce
	v_wmma_f32_16x16x16_fp8_fp8 v[188:195], v[206:207], v[162:163], v[188:195]// 000000002b88: cc4640bc 1ef345ce
	s_wait_loadcnt 0xe                                         // 000000002b90: bfc0000e
	v_mul_f32_e32 v161, v209, v221                             // 000000002b94: 1143bbd1
	v_cndmask_b32_e64 v210, s6, v208, s2                       // 000000002b98: d50100d2 000ba006
	v_cndmask_b32_e64 v211, s6, v208, s3                       // 000000002ba0: d50100d3 000fa006
	v_cndmask_b32_e64 v208, s6, v208, s4                       // 000000002ba8: d50100d0 0013a006
	s_delay_alu instid0(valu_dep_3)                            // 000000002bb0: bf870003
	v_dual_mul_f32 v1, v1, v161 :: v_dual_mul_f32 v200, v220, v210// 000000002bb4: c8c74301 01c9a5dc
	s_wait_loadcnt 0xc                                         // 000000002bbc: bfc0000c
	v_dual_mul_f32 v160, v220, v209 :: v_dual_mul_f32 v163, v209, v223// 000000002bc0: c8c7a3dc a0a3bfd1
	s_wait_loadcnt 0xa                                         // 000000002bc8: bfc0000a
	v_dual_mul_f32 v162, v209, v222 :: v_dual_mul_f32 v197, v209, v225// 000000002bcc: c8c7bdd1 a2c5c3d1
	s_wait_loadcnt 0x8                                         // 000000002bd4: bfc00008
	v_dual_mul_f32 v196, v209, v224 :: v_dual_mul_f32 v199, v209, v227// 000000002bd8: c8c7c1d1 c4c7c7d1
	v_dual_mul_f32 v198, v209, v226 :: v_dual_mul_f32 v201, v210, v221// 000000002be0: c8c7c5d1 c6c9bbd2
	v_dual_mul_f32 v202, v210, v222 :: v_dual_mul_f32 v203, v210, v223// 000000002be8: c8c7bdd2 cacbbfd2
	v_dual_mul_f32 v204, v210, v224 :: v_dual_mul_f32 v205, v210, v225// 000000002bf0: c8c7c1d2 cccdc3d2
	v_dual_mul_f32 v206, v210, v226 :: v_dual_mul_f32 v207, v210, v227// 000000002bf8: c8c7c5d2 cecfc7d2
	v_dual_mul_f32 v212, v220, v211 :: v_dual_mul_f32 v213, v211, v221// 000000002c00: c8c7a7dc d4d5bbd3
	v_dual_mul_f32 v214, v211, v222 :: v_dual_mul_f32 v215, v211, v223// 000000002c08: c8c7bdd3 d6d7bfd3
	v_dual_mul_f32 v216, v211, v224 :: v_dual_mul_f32 v217, v211, v225// 000000002c10: c8c7c1d3 d8d9c3d3
	v_mul_f32_e32 v218, v211, v226                             // 000000002c18: 11b5c5d3
	v_dual_mul_f32 v0, v0, v160 :: v_dual_mul_f32 v3, v3, v163 // 000000002c1c: c8c74100 00034703
	v_dual_mul_f32 v160, v211, v227 :: v_dual_mul_f32 v163, v208, v222// 000000002c24: c8c7c7d3 a0a3bdd0
	v_dual_mul_f32 v2, v2, v162 :: v_dual_mul_f32 v5, v5, v197 // 000000002c2c: c8c74502 02058b05
	v_dual_mul_f32 v4, v4, v196 :: v_dual_mul_f32 v7, v7, v199 // 000000002c34: c8c78904 04078f07
	v_dual_mul_f32 v6, v6, v198 :: v_dual_mul_f32 v161, v220, v208// 000000002c3c: c8c78d06 06a1a1dc
	v_dual_mul_f32 v162, v208, v221 :: v_dual_mul_f32 v197, v208, v224// 000000002c44: c8c7bbd0 a2c5c1d0
	v_dual_mul_f32 v196, v208, v223 :: v_dual_mul_f32 v199, v208, v226// 000000002c4c: c8c7bfd0 c4c7c5d0
	v_dual_mul_f32 v198, v208, v225 :: v_dual_mul_f32 v219, v208, v227// 000000002c54: c8c7c3d0 c6dbc7d0
	s_wait_loadcnt 0x6                                         // 000000002c5c: bfc00006
	v_dual_mul_f32 v220, v209, v228 :: v_dual_mul_f32 v221, v209, v229// 000000002c60: c8c7c9d1 dcddcbd1
	s_wait_loadcnt 0x4                                         // 000000002c68: bfc00004
	v_dual_mul_f32 v222, v209, v230 :: v_dual_mul_f32 v223, v209, v231// 000000002c6c: c8c7cdd1 dedfcfd1
	s_wait_loadcnt 0x2                                         // 000000002c74: bfc00002
	v_dual_mul_f32 v224, v209, v232 :: v_dual_mul_f32 v225, v209, v233// 000000002c78: c8c7d1d1 e0e1d3d1
	s_wait_loadcnt 0x0                                         // 000000002c80: bfc00000
	v_dual_mul_f32 v226, v209, v234 :: v_dual_mul_f32 v209, v209, v235// 000000002c84: c8c7d5d1 e2d1d7d1
	v_dual_mul_f32 v8, v8, v200 :: v_dual_mul_f32 v9, v9, v201 // 000000002c8c: c8c79108 08099309
	v_dual_mul_f32 v10, v10, v202 :: v_dual_mul_f32 v11, v11, v203// 000000002c94: c8c7950a 0a0b970b
	v_dual_mul_f32 v12, v12, v204 :: v_dual_mul_f32 v13, v13, v205// 000000002c9c: c8c7990c 0c0d9b0d
	v_dual_mul_f32 v14, v14, v206 :: v_dual_mul_f32 v15, v15, v207// 000000002ca4: c8c79d0e 0e0f9f0f
	v_dual_mul_f32 v200, v210, v228 :: v_dual_mul_f32 v201, v210, v229// 000000002cac: c8c7c9d2 c8c9cbd2
	v_dual_mul_f32 v202, v210, v230 :: v_dual_mul_f32 v203, v210, v231// 000000002cb4: c8c7cdd2 cacbcfd2
	v_dual_mul_f32 v204, v210, v232 :: v_dual_mul_f32 v205, v210, v233// 000000002cbc: c8c7d1d2 cccdd3d2
	v_dual_mul_f32 v206, v210, v234 :: v_dual_mul_f32 v207, v210, v235// 000000002cc4: c8c7d5d2 cecfd7d2
	v_dual_mul_f32 v210, v211, v228 :: v_dual_mul_f32 v17, v17, v213// 000000002ccc: c8c7c9d3 d211ab11
	v_dual_mul_f32 v16, v16, v212 :: v_dual_mul_f32 v19, v19, v215// 000000002cd4: c8c7a910 1013af13
	v_dual_mul_f32 v18, v18, v214 :: v_dual_mul_f32 v21, v21, v217// 000000002cdc: c8c7ad12 1215b315
	v_dual_mul_f32 v20, v20, v216 :: v_dual_mul_f32 v213, v211, v231// 000000002ce4: c8c7b114 14d5cfd3
	v_dual_mul_f32 v22, v22, v218 :: v_dual_mul_f32 v23, v23, v160// 000000002cec: c8c7b516 16174117
	v_mul_f32_e32 v160, v211, v229                             // 000000002cf4: 1141cbd3
	v_dual_mul_f32 v212, v211, v230 :: v_dual_mul_f32 v215, v211, v233// 000000002cf8: c8c7cdd3 d4d7d3d3
	v_dual_mul_f32 v214, v211, v232 :: v_dual_mul_f32 v227, v208, v230// 000000002d00: c8c7d1d3 d6e3cdd0
	v_dual_mul_f32 v216, v211, v234 :: v_dual_mul_f32 v217, v208, v228// 000000002d08: c8c7d5d3 d8d9c9d0
	v_dual_mul_f32 v211, v211, v235 :: v_dual_mul_f32 v218, v208, v229// 000000002d10: c8c7d7d3 d3dbcbd0
	v_dual_mul_f32 v229, v208, v232 :: v_dual_mul_f32 v228, v208, v231// 000000002d18: c8c7d1d0 e5e5cfd0
	v_dual_mul_f32 v231, v208, v234 :: v_dual_mul_f32 v230, v208, v233// 000000002d20: c8c7d5d0 e7e7d3d0
	v_dual_mul_f32 v25, v25, v162 :: v_dual_mul_f32 v208, v208, v235// 000000002d28: c8c74519 19d1d7d0
	v_dual_mul_f32 v27, v27, v196 :: v_dual_mul_f32 v24, v24, v161// 000000002d30: c8c7891b 1b194318
	v_dual_mul_f32 v29, v29, v198 :: v_dual_mul_f32 v26, v26, v163// 000000002d38: c8c78d1d 1d1b471a
	v_mul_f32_e32 v161, v164, v220                             // 000000002d40: 1143b9a4
	v_dual_mul_f32 v28, v28, v197 :: v_dual_mul_f32 v31, v31, v219// 000000002d44: c8c78b1c 1c1fb71f
	v_mul_f32_e32 v30, v30, v199                               // 000000002d4c: 103d8f1e
	v_dual_mul_f32 v162, v165, v221 :: v_dual_mul_f32 v163, v166, v222// 000000002d50: c8c7bba5 a2a3bda6
	v_dual_mul_f32 v164, v167, v223 :: v_dual_mul_f32 v167, v170, v226// 000000002d58: c8c7bfa7 a4a7c5aa
	v_dual_mul_f32 v165, v168, v224 :: v_dual_mul_f32 v166, v169, v225// 000000002d60: c8c7c1a8 a5a7c3a9
	v_dual_mul_f32 v169, v172, v200 :: v_dual_mul_f32 v168, v171, v209// 000000002d68: c8c791ac a9a9a3ab
	v_dual_mul_f32 v171, v174, v202 :: v_dual_mul_f32 v170, v173, v201// 000000002d70: c8c795ae abab93ad
	v_dual_mul_f32 v173, v176, v204 :: v_dual_mul_f32 v172, v175, v203// 000000002d78: c8c799b0 adad97af
	v_dual_mul_f32 v175, v178, v206 :: v_dual_mul_f32 v174, v177, v205// 000000002d80: c8c79db2 afaf9bb1
	v_dual_mul_f32 v177, v180, v210 :: v_dual_mul_f32 v176, v179, v207// 000000002d88: c8c7a5b4 b1b19fb3
	v_dual_mul_f32 v160, v181, v160 :: v_dual_mul_f32 v179, v183, v213// 000000002d90: c8c741b5 a0b3abb7
	v_dual_mul_f32 v178, v182, v212 :: v_dual_mul_f32 v181, v185, v215// 000000002d98: c8c7a9b6 b2b5afb9
	v_dual_mul_f32 v180, v184, v214 :: v_dual_mul_f32 v183, v187, v211// 000000002da0: c8c7adb8 b4b7a7bb
	v_mul_f32_e32 v182, v186, v216                             // 000000002da8: 116db1ba
	v_dual_mul_f32 v184, v188, v217 :: v_dual_mul_f32 v187, v191, v228// 000000002dac: c8c7b3bc b8bbc9bf
	v_dual_mul_f32 v185, v189, v218 :: v_dual_mul_f32 v186, v190, v227// 000000002db4: c8c7b5bd b9bbc7be
	v_dual_mul_f32 v191, v195, v208 :: v_dual_mul_f32 v188, v192, v229// 000000002dbc: c8c7a1c3 bfbdcbc0
	v_add_f32_e32 v145, v145, v3                               // 000000002dc4: 07220791
	v_dual_mul_f32 v189, v193, v230 :: v_dual_mul_f32 v190, v194, v231// 000000002dc8: c8c7cdc1 bdbfcfc2
	v_dual_add_f32 v149, v149, v1 :: v_dual_add_f32 v50, v50, v0// 000000002dd0: c9080395 95320132
	v_dual_add_f32 v141, v141, v5 :: v_dual_add_f32 v148, v148, v2// 000000002dd8: c9080b8d 8d940594
	v_add_f32_e32 v115, v115, v8                               // 000000002de0: 06e61173
	v_dual_add_f32 v144, v144, v4 :: v_dual_add_f32 v113, v113, v9// 000000002de4: c9080990 90701371
	v_dual_add_f32 v140, v140, v6 :: v_dual_add_f32 v109, v109, v11// 000000002dec: c9080d8c 8c6c176d
	v_dual_add_f32 v136, v136, v7 :: v_dual_add_f32 v105, v105, v14// 000000002df4: c9080f88 88681d69
	v_dual_add_f32 v112, v112, v10 :: v_dual_add_f32 v93, v93, v16// 000000002dfc: c9081570 705c215d
	v_dual_add_f32 v108, v108, v12 :: v_dual_add_f32 v89, v89, v17// 000000002e04: c908196c 6c582359
	v_dual_add_f32 v106, v106, v13 :: v_dual_add_f32 v85, v85, v19// 000000002e0c: c9081b6a 6a542755
	v_dual_add_f32 v102, v102, v15 :: v_dual_add_f32 v75, v75, v21// 000000002e14: c9081f66 664a2b4b
	v_dual_add_f32 v86, v86, v18 :: v_dual_add_f32 v53, v53, v25// 000000002e1c: c9082556 56343335
	v_dual_add_f32 v82, v82, v20 :: v_dual_add_f32 v73, v73, v22// 000000002e24: c9082952 52482d49
	v_dual_add_f32 v68, v68, v23 :: v_dual_add_f32 v49, v49, v28// 000000002e2c: c9082f44 44303931
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v51, v51, v27// 000000002e34: c9083136 36323733
	v_dual_add_f32 v52, v52, v26 :: v_dual_add_f32 v47, v47, v29// 000000002e3c: c9083534 342e3b2f
	v_dual_add_f32 v46, v46, v30 :: v_dual_add_f32 v43, v43, v31// 000000002e44: c9083d2e 2e2a3f2b
	v_dual_add_f32 v137, v137, v161 :: v_dual_add_f32 v134, v134, v163// 000000002e4c: c9094389 89874786
	v_dual_add_f32 v135, v135, v162 :: v_dual_add_f32 v128, v128, v165// 000000002e54: c9094587 87814b80
	v_dual_add_f32 v131, v131, v164 :: v_dual_add_f32 v124, v124, v167// 000000002e5c: c9094983 837d4f7c
	v_dual_add_f32 v125, v125, v166 :: v_dual_add_f32 v100, v100, v173// 000000002e64: c9094d7d 7d655b64
	v_dual_add_f32 v119, v119, v168 :: v_dual_add_f32 v104, v104, v170// 000000002e6c: c9095177 77695568
	v_dual_add_f32 v107, v107, v169 :: v_dual_add_f32 v98, v98, v175// 000000002e74: c909536b 6b635f62
	v_dual_add_f32 v103, v103, v171 :: v_dual_add_f32 v96, v96, v176// 000000002e7c: c9095767 67616160
	v_dual_add_f32 v101, v101, v172 :: v_dual_add_f32 v58, v58, v181// 000000002e84: c9095965 653b6b3a
	v_dual_add_f32 v99, v99, v174 :: v_dual_add_f32 v72, v72, v160// 000000002e8c: c9095d63 63494148
	v_dual_add_f32 v77, v77, v177 :: v_dual_add_f32 v60, v60, v180// 000000002e94: c909634d 4d3d693c
	v_dual_add_f32 v69, v69, v178 :: v_dual_add_f32 v48, v48, v184// 000000002e9c: c9096545 45317130
	v_dual_add_f32 v63, v63, v179 :: v_dual_add_f32 v44, v44, v186// 000000002ea4: c909673f 3f2d752c
	v_dual_add_f32 v57, v57, v182 :: v_dual_add_f32 v42, v42, v187// 000000002eac: c9096d39 392b772a
	v_dual_add_f32 v55, v55, v183 :: v_dual_add_f32 v40, v40, v189// 000000002eb4: c9096f37 37297b28
	v_dual_add_f32 v45, v45, v185 :: v_dual_add_f32 v38, v38, v191// 000000002ebc: c909732d 2d277f26
	v_add_f32_e32 v41, v41, v188                               // 000000002ec4: 06537929
	v_add_f32_e32 v39, v39, v190                               // 000000002ec8: 064f7d27
	s_cbranch_scc1 64899                                       // 000000002ecc: bfa2fd83 <tessera_rocm_scaled_matmul_lds_ed0a211cd1c74a4f+0x9dc>
	s_load_b64 s[2:3], s[0:1], 0xa8                            // 000000002ed0: f4002080 f80000a8
	v_mul_lo_u32 v2, s23, v32                                  // 000000002ed8: d72c0002 02024017
	v_mul_lo_u32 v3, s22, v33                                  // 000000002ee0: d72c0003 02024216
	v_mad_co_u64_u32 v[0:1], null, s22, v32, 0                 // 000000002ee8: d6fe7c00 02024016
	v_bfe_u32 v4, v50, 16, 1                                   // 000000002ef0: d6100004 02052132
	v_or_b32_e32 v5, 0x400000, v50                             // 000000002ef8: 380a64ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000002f00: 7c306532
	v_lshlrev_b64_e32 v[26:27], 1, v[34:35]                    // 000000002f04: 3e344481
	s_lshl_b64 s[0:1], s[22:23], 1                             // 000000002f08: 84808116
	v_add3_u32 v4, v4, v50, 0x7fff                             // 000000002f0c: d6550004 03fe6504 00007fff
	v_or_b32_e32 v9, 0x400000, v148                            // 000000002f18: 381328ff 00400000
	v_add3_u32 v1, v1, v3, v2                                  // 000000002f20: d6550001 040a0701
	v_bfe_u32 v2, v149, 16, 1                                  // 000000002f28: d6100002 02052195
	v_or_b32_e32 v3, 0x400000, v149                            // 000000002f30: 38072aff 00400000
	v_bfe_u32 v11, v145, 16, 1                                 // 000000002f38: d610000b 02052191
	v_or_b32_e32 v12, 0x400000, v145                           // 000000002f40: 381922ff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002f48: 3e000081
	v_add3_u32 v2, v2, v149, 0x7fff                            // 000000002f4c: d6550002 03ff2b02 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002f58: bf88ff9d
	v_cndmask_b32_e32 v6, v4, v5, vcc_lo                       // 000000002f5c: 020c0b04
	v_bfe_u32 v4, v148, 16, 1                                  // 000000002f60: d6100004 02052194
	v_add3_u32 v11, v11, v145, 0x7fff                          // 000000002f68: d655000b 03ff230b 00007fff
	v_bfe_u32 v17, v140, 16, 1                                 // 000000002f74: d6100011 0205218c
	s_wait_kmcnt 0x0                                           // 000000002f7c: bfc70000
	v_add_co_u32 v0, vcc_lo, s2, v0                            // 000000002f80: d7006a00 02020002
	s_wait_alu depctr_va_vcc(0)                                // 000000002f88: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo               // 000000002f8c: d5207c01 01aa0203
	v_cmp_u_f32_e32 vcc_lo, v149, v149                         // 000000002f94: 7c312b95
	v_add3_u32 v4, v4, v148, 0x7fff                            // 000000002f98: d6550004 03ff2904 00007fff
	v_add3_u32 v17, v17, v140, 0x7fff                          // 000000002fa4: d6550011 03ff1911 00007fff
	v_or_b32_e32 v18, 0x400000, v140                           // 000000002fb0: 382518ff 00400000
	v_or_b32_e32 v15, 0x400000, v141                           // 000000002fb8: 381f1aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002fc0: bf88ff9d
	v_cndmask_b32_e32 v7, v2, v3, vcc_lo                       // 000000002fc4: 020e0702
	v_add_co_u32 v2, vcc_lo, v0, v26                           // 000000002fc8: d7006a02 02023500
	s_wait_alu depctr_va_vcc(0)                                // 000000002fd0: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v1, v27, vcc_lo              // 000000002fd4: d5207c03 01aa3701
	v_add_co_u32 v5, vcc_lo, v0, s0                            // 000000002fdc: d7006a05 02000100
	s_wait_alu depctr_va_vcc(0)                                // 000000002fe4: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s1, v1, vcc_lo               // 000000002fe8: d5207c08 01aa0201
	v_mul_lo_u32 v19, s23, v36                                 // 000000002ff0: d72c0013 02024817
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002ff8: bf8701a3
	v_add_co_u32 v0, vcc_lo, v5, v26                           // 000000002ffc: d7006a00 02023505
	s_wait_alu depctr_va_vcc(0)                                // 000000003004: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v8, v27, vcc_lo              // 000000003008: d5207c01 01aa3708
	v_cmp_u_f32_e32 vcc_lo, v148, v148                         // 000000003010: 7c312994
	v_mul_lo_u32 v21, s22, v37                                 // 000000003014: d72c0015 02024a16
	v_or_b32_e32 v22, 0x400000, v136                           // 00000000301c: 382d10ff 00400000
	v_or_b32_e32 v23, 0x400000, v137                           // 000000003024: 382f12ff 00400000
	v_or_b32_e32 v24, 0x400000, v135                           // 00000000302c: 38310eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003034: bf88ff9d
	v_cndmask_b32_e32 v9, v4, v9, vcc_lo                       // 000000003038: 02121304
	v_add_co_u32 v10, vcc_lo, v5, s0                           // 00000000303c: d7006a0a 02000105
	s_wait_alu depctr_va_vcc(0)                                // 000000003044: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s1, v8, vcc_lo               // 000000003048: d5207c08 01aa1001
	v_or_b32_e32 v29, 0x400000, v131                           // 000000003050: 383b06ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003058: bf8701a3
	v_add_co_u32 v4, vcc_lo, v10, v26                          // 00000000305c: d7006a04 0202350a
	s_wait_alu depctr_va_vcc(0)                                // 000000003064: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v8, v27, vcc_lo              // 000000003068: d5207c05 01aa3708
	v_cmp_u_f32_e32 vcc_lo, v145, v145                         // 000000003070: 7c312391
	v_or_b32_e32 v35, 0x400000, v124                           // 000000003074: 3846f8ff 00400000
	v_bfe_u32 v31, v128, 16, 1                                 // 00000000307c: d610001f 02052180
	v_or_b32_e32 v32, 0x400000, v128                           // 000000003084: 384100ff 00400000
	v_bfe_u32 v37, v119, 16, 1                                 // 00000000308c: d6100025 02052177
	s_wait_alu depctr_va_vcc(0)                                // 000000003094: bf88ff9d
	v_cndmask_b32_e32 v12, v11, v12, vcc_lo                    // 000000003098: 0218190b
	s_clause 0x2                                               // 00000000309c: bf850002
	global_store_d16_hi_b16 v[2:3], v6, off                    // 0000000030a0: ee09407c 03000000 00000002
	global_store_d16_hi_b16 v[0:1], v7, off                    // 0000000030ac: ee09407c 03800000 00000000
	global_store_d16_hi_b16 v[4:5], v9, off                    // 0000000030b8: ee09407c 04800000 00000004
	v_bfe_u32 v6, v144, 16, 1                                  // 0000000030c4: d6100006 02052190
	v_add_co_u32 v9, vcc_lo, v10, s0                           // 0000000030cc: d7006a09 0200010a
	s_wait_alu depctr_va_vcc(0)                                // 0000000030d4: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s1, v8, vcc_lo               // 0000000030d8: d5207c08 01aa1001
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000030e0: bf870193
	v_add3_u32 v10, v6, v144, 0x7fff                           // 0000000030e4: d655000a 03ff2106 00007fff
	v_add_co_u32 v6, vcc_lo, v9, v26                           // 0000000030f0: d7006a06 02023509
	v_or_b32_e32 v11, 0x400000, v144                           // 0000000030f8: 381720ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003100: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v8, v27, vcc_lo              // 000000003104: d5207c07 01aa3708
	v_cmp_u_f32_e32 vcc_lo, v144, v144                         // 00000000310c: 7c312190
	v_add3_u32 v31, v31, v128, 0x7fff                          // 000000003110: d655001f 03ff011f 00007fff
	v_add3_u32 v37, v37, v119, 0x7fff                          // 00000000311c: d6550025 03feef25 00007fff
	v_or_b32_e32 v50, 0x400000, v119                           // 000000003128: 3864eeff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003130: bf88ff9d
	v_cndmask_b32_e32 v13, v10, v11, vcc_lo                    // 000000003134: 021a170a
	v_bfe_u32 v10, v141, 16, 1                                 // 000000003138: d610000a 0205218d
	v_add_co_u32 v9, vcc_lo, v9, s0                            // 000000003140: d7006a09 02000109
	s_wait_alu depctr_va_vcc(0)                                // 000000003148: bf88ff9d
	v_add_co_ci_u32_e64 v8, null, s1, v8, vcc_lo               // 00000000314c: d5207c08 01aa1001
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003154: bf870193
	v_add3_u32 v14, v10, v141, 0x7fff                          // 000000003158: d655000e 03ff1b0a 00007fff
	v_add_co_u32 v10, vcc_lo, v9, v26                          // 000000003164: d7006a0a 02023509
	s_wait_alu depctr_va_vcc(0)                                // 00000000316c: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003170: bf870003
	v_add_co_ci_u32_e64 v11, null, v8, v27, vcc_lo             // 000000003174: d5207c0b 01aa3708
	v_cmp_u_f32_e32 vcc_lo, v141, v141                         // 00000000317c: 7c311b8d
	s_wait_alu depctr_va_vcc(0)                                // 000000003180: bf88ff9d
	v_cndmask_b32_e32 v14, v14, v15, vcc_lo                    // 000000003184: 021c1f0e
	v_add_co_u32 v15, vcc_lo, v9, s0                           // 000000003188: d7006a0f 02000109
	s_wait_alu depctr_va_vcc(0)                                // 000000003190: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s1, v8, vcc_lo              // 000000003194: d5207c10 01aa1001
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000319c: bf870122
	v_add_co_u32 v8, vcc_lo, v15, v26                          // 0000000031a0: d7006a08 0202350f
	s_wait_alu depctr_va_vcc(0)                                // 0000000031a8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v16, v27, vcc_lo             // 0000000031ac: d5207c09 01aa3710
	v_cmp_u_f32_e32 vcc_lo, v140, v140                         // 0000000031b4: 7c31198c
	s_wait_alu depctr_va_vcc(0)                                // 0000000031b8: bf88ff9d
	v_cndmask_b32_e32 v20, v17, v18, vcc_lo                    // 0000000031bc: 02282511
	s_clause 0x2                                               // 0000000031c0: bf850002
	global_store_d16_hi_b16 v[6:7], v12, off                   // 0000000031c4: ee09407c 06000000 00000006
	global_store_d16_hi_b16 v[10:11], v13, off                 // 0000000031d0: ee09407c 06800000 0000000a
	global_store_d16_hi_b16 v[8:9], v14, off                   // 0000000031dc: ee09407c 07000000 00000008
	v_bfe_u32 v12, v136, 16, 1                                 // 0000000031e8: d610000c 02052188
	v_add_co_u32 v17, vcc_lo, v15, s0                          // 0000000031f0: d7006a11 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 0000000031f8: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo             // 0000000031fc: d5207c10 01aa2001
	s_delay_alu instid0(valu_dep_3)                            // 000000003204: bf870003
	v_add3_u32 v18, v12, v136, 0x7fff                          // 000000003208: d6550012 03ff110c 00007fff
	v_mad_co_u64_u32 v[12:13], null, s22, v36, 0               // 000000003214: d6fe7c0c 02024816
	v_add_co_u32 v14, vcc_lo, v17, v26                         // 00000000321c: d7006a0e 02023511
	s_wait_alu depctr_va_vcc(0)                                // 000000003224: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v16, v27, vcc_lo            // 000000003228: d5207c0f 01aa3710
	v_cmp_u_f32_e32 vcc_lo, v136, v136                         // 000000003230: 7c311188
	s_delay_alu instid0(valu_dep_4)                            // 000000003234: bf870004
	v_add3_u32 v13, v13, v21, v19                              // 000000003238: d655000d 044e2b0d
	s_wait_alu depctr_va_vcc(0)                                // 000000003240: bf88ff9d
	v_cndmask_b32_e32 v22, v18, v22, vcc_lo                    // 000000003244: 022c2d12
	v_add_co_u32 v19, vcc_lo, v17, s0                          // 000000003248: d7006a13 02000111
	v_bfe_u32 v18, v137, 16, 1                                 // 000000003250: d6100012 02052189
	s_wait_alu depctr_va_vcc(0)                                // 000000003258: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, s1, v16, vcc_lo             // 00000000325c: d5207c15 01aa2001
	v_lshlrev_b64_e32 v[16:17], 1, v[12:13]                    // 000000003264: 3e201881
	v_add_co_u32 v12, vcc_lo, v19, v26                         // 000000003268: d7006a0c 02023513
	v_add3_u32 v18, v18, v137, 0x7fff                          // 000000003270: d6550012 03ff1312 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000327c: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v21, v27, vcc_lo            // 000000003280: d5207c0d 01aa3715
	v_cmp_u_f32_e32 vcc_lo, v137, v137                         // 000000003288: 7c311389
	s_wait_alu depctr_va_vcc(0)                                // 00000000328c: bf88ff9d
	v_cndmask_b32_e32 v21, v18, v23, vcc_lo                    // 000000003290: 022a2f12
	v_add_co_u32 v16, vcc_lo, s2, v16                          // 000000003294: d7006a10 02022002
	s_wait_alu depctr_va_vcc(0)                                // 00000000329c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s3, v17, vcc_lo             // 0000000032a0: d5207c11 01aa2203
	v_bfe_u32 v23, v135, 16, 1                                 // 0000000032a8: d6100017 02052187
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000032b0: bf8701a3
	v_add_co_u32 v18, vcc_lo, v16, v26                         // 0000000032b4: d7006a12 02023510
	s_wait_alu depctr_va_vcc(0)                                // 0000000032bc: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v17, v27, vcc_lo            // 0000000032c0: d5207c13 01aa3711
	s_delay_alu instid0(valu_dep_3)                            // 0000000032c8: bf870003
	v_add3_u32 v23, v23, v135, 0x7fff                          // 0000000032cc: d6550017 03ff0f17 00007fff
	v_cmp_u_f32_e32 vcc_lo, v135, v135                         // 0000000032d8: 7c310f87
	s_clause 0x2                                               // 0000000032dc: bf850002
	global_store_d16_hi_b16 v[14:15], v20, off                 // 0000000032e0: ee09407c 0a000000 0000000e
	global_store_d16_hi_b16 v[12:13], v22, off                 // 0000000032ec: ee09407c 0b000000 0000000c
	global_store_d16_hi_b16 v[18:19], v21, off                 // 0000000032f8: ee09407c 0a800000 00000012
	v_bfe_u32 v20, v134, 16, 1                                 // 000000003304: d6100014 02052186
	s_wait_alu depctr_va_vcc(0)                                // 00000000330c: bf88ff9d
	v_cndmask_b32_e32 v22, v23, v24, vcc_lo                    // 000000003310: 022c3117
	v_add_co_u32 v16, vcc_lo, v16, s0                          // 000000003314: d7006a10 02000110
	s_wait_alu depctr_va_vcc(0)                                // 00000000331c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v17, vcc_lo             // 000000003320: d5207c11 01aa2201
	v_add3_u32 v23, v20, v134, 0x7fff                          // 000000003328: d6550017 03ff0d14 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003334: bf870003
	v_add_co_u32 v20, vcc_lo, v16, v26                         // 000000003338: d7006a14 02023510
	v_or_b32_e32 v24, 0x400000, v134                           // 000000003340: 38310cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003348: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v17, v27, vcc_lo            // 00000000334c: d5207c15 01aa3711
	v_cmp_u_f32_e32 vcc_lo, v134, v134                         // 000000003354: 7c310d86
	s_wait_alu depctr_va_vcc(0)                                // 000000003358: bf88ff9d
	v_cndmask_b32_e32 v23, v23, v24, vcc_lo                    // 00000000335c: 022e3117
	v_bfe_u32 v24, v131, 16, 1                                 // 000000003360: d6100018 02052183
	v_add_co_u32 v16, vcc_lo, v16, s0                          // 000000003368: d7006a10 02000110
	s_wait_alu depctr_va_vcc(0)                                // 000000003370: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v17, vcc_lo             // 000000003374: d5207c11 01aa2201
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000337c: bf870193
	v_add3_u32 v28, v24, v131, 0x7fff                          // 000000003380: d655001c 03ff0718 00007fff
	v_add_co_u32 v24, vcc_lo, v16, v26                         // 00000000338c: d7006a18 02023510
	s_wait_alu depctr_va_vcc(0)                                // 000000003394: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003398: bf870003
	v_add_co_ci_u32_e64 v25, null, v17, v27, vcc_lo            // 00000000339c: d5207c19 01aa3711
	v_cmp_u_f32_e32 vcc_lo, v131, v131                         // 0000000033a4: 7c310783
	s_wait_alu depctr_va_vcc(0)                                // 0000000033a8: bf88ff9d
	v_cndmask_b32_e32 v28, v28, v29, vcc_lo                    // 0000000033ac: 02383b1c
	v_add_co_u32 v29, vcc_lo, v16, s0                          // 0000000033b0: d7006a1d 02000110
	s_wait_alu depctr_va_vcc(0)                                // 0000000033b8: bf88ff9d
	v_add_co_ci_u32_e64 v30, null, s1, v17, vcc_lo             // 0000000033bc: d5207c1e 01aa2201
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000033c4: bf870122
	v_add_co_u32 v16, vcc_lo, v29, v26                         // 0000000033c8: d7006a10 0202351d
	s_wait_alu depctr_va_vcc(0)                                // 0000000033d0: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v30, v27, vcc_lo            // 0000000033d4: d5207c11 01aa371e
	v_cmp_u_f32_e32 vcc_lo, v128, v128                         // 0000000033dc: 7c310180
	s_clause 0x2                                               // 0000000033e0: bf850002
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000033e4: ee09407c 0b000000 00000014
	global_store_d16_hi_b16 v[24:25], v23, off                 // 0000000033f0: ee09407c 0b800000 00000018
	global_store_d16_hi_b16 v[16:17], v28, off                 // 0000000033fc: ee09407c 0e000000 00000010
	v_bfe_u32 v22, v125, 16, 1                                 // 000000003408: d6100016 0205217d
	s_wait_alu depctr_va_vcc(0)                                // 000000003410: bf88ff9d
	v_cndmask_b32_e32 v32, v31, v32, vcc_lo                    // 000000003414: 0240411f
	v_add_co_u32 v28, vcc_lo, v29, s0                          // 000000003418: d7006a1c 0200011d
	s_wait_alu depctr_va_vcc(0)                                // 000000003420: bf88ff9d
	v_add_co_ci_u32_e64 v29, null, s1, v30, vcc_lo             // 000000003424: d5207c1d 01aa3c01
	v_add3_u32 v30, v22, v125, 0x7fff                          // 00000000342c: d655001e 03fefb16 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003438: bf870003
	v_add_co_u32 v22, vcc_lo, v28, v26                         // 00000000343c: d7006a16 0202351c
	v_or_b32_e32 v31, 0x400000, v125                           // 000000003444: 383efaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000344c: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, v29, v27, vcc_lo            // 000000003450: d5207c17 01aa371d
	v_cmp_u_f32_e32 vcc_lo, v125, v125                         // 000000003458: 7c30fb7d
	s_wait_alu depctr_va_vcc(0)                                // 00000000345c: bf88ff9d
	v_cndmask_b32_e32 v33, v30, v31, vcc_lo                    // 000000003460: 02423f1e
	v_add_co_u32 v31, vcc_lo, v28, s0                          // 000000003464: d7006a1f 0200011c
	v_bfe_u32 v30, v124, 16, 1                                 // 00000000346c: d610001e 0205217c
	s_wait_alu depctr_va_vcc(0)                                // 000000003474: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v29, vcc_lo             // 000000003478: d5207c22 01aa3a01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003480: bf870193
	v_add_co_u32 v28, vcc_lo, v31, v26                         // 000000003484: d7006a1c 0202351f
	v_add3_u32 v30, v30, v124, 0x7fff                          // 00000000348c: d655001e 03fef91e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003498: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000349c: bf870003
	v_add_co_ci_u32_e64 v29, null, v34, v27, vcc_lo            // 0000000034a0: d5207c1d 01aa3722
	v_cmp_u_f32_e32 vcc_lo, v124, v124                         // 0000000034a8: 7c30f97c
	s_wait_alu depctr_va_vcc(0)                                // 0000000034ac: bf88ff9d
	v_cndmask_b32_e32 v35, v30, v35, vcc_lo                    // 0000000034b0: 0246471e
	v_add_co_u32 v36, vcc_lo, v31, s0                          // 0000000034b4: d7006a24 0200011f
	s_wait_alu depctr_va_vcc(0)                                // 0000000034bc: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v34, vcc_lo             // 0000000034c0: d5207c22 01aa4401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000034c8: bf870122
	v_add_co_u32 v30, vcc_lo, v36, v26                         // 0000000034cc: d7006a1e 02023524
	s_wait_alu depctr_va_vcc(0)                                // 0000000034d4: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, v34, v27, vcc_lo            // 0000000034d8: d5207c1f 01aa3722
	v_cmp_u_f32_e32 vcc_lo, v119, v119                         // 0000000034e0: 7c30ef77
	s_clause 0x2                                               // 0000000034e4: bf850002
	global_store_d16_hi_b16 v[22:23], v32, off                 // 0000000034e8: ee09407c 10000000 00000016
	global_store_d16_hi_b16 v[28:29], v33, off                 // 0000000034f4: ee09407c 10800000 0000001c
	global_store_d16_hi_b16 v[30:31], v35, off                 // 000000003500: ee09407c 11800000 0000001e
	v_bfe_u32 v33, v115, 16, 1                                 // 00000000350c: d6100021 02052173
	s_wait_alu depctr_va_vcc(0)                                // 000000003514: bf88ff9d
	v_cndmask_b32_e32 v32, v37, v50, vcc_lo                    // 000000003518: 02406525
	v_add_co_u32 v35, vcc_lo, v36, s0                          // 00000000351c: d7006a23 02000124
	s_wait_alu depctr_va_vcc(0)                                // 000000003524: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v34, vcc_lo             // 000000003528: d5207c22 01aa4401
	v_add3_u32 v33, v33, v115, 0x7fff                          // 000000003530: d6550021 03fee721 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000353c: bf870003
	v_add_co_u32 v26, vcc_lo, v35, v26                         // 000000003540: d7006a1a 02023523
	v_or_b32_e32 v36, 0x400000, v115                           // 000000003548: 3848e6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003550: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v34, v27, vcc_lo            // 000000003554: d5207c1b 01aa3722
	v_bfe_u32 v34, v113, 16, 1                                 // 00000000355c: d6100022 02052171
	v_cmp_u_f32_e32 vcc_lo, v115, v115                         // 000000003564: 7c30e773
	v_bfe_u32 v35, v112, 16, 1                                 // 000000003568: d6100023 02052170
	global_store_d16_hi_b16 v[26:27], v32, off                 // 000000003570: ee09407c 10000000 0000001a
	v_add3_u32 v32, v34, v113, 0x7fff                          // 00000000357c: d6550020 03fee322 00007fff
	v_or_b32_e32 v34, 0x400000, v113                           // 000000003588: 3844e2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003590: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v36, vcc_lo                    // 000000003594: 02424921
	v_cmp_u_f32_e32 vcc_lo, v113, v113                         // 000000003598: 7c30e371
	s_wait_alu depctr_va_vcc(0)                                // 00000000359c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 0000000035a0: 02404520
	global_store_d16_hi_b16 v[2:3], v33, off offset:32         // 0000000035a4: ee09407c 10800000 00002002
	v_add3_u32 v33, v35, v112, 0x7fff                          // 0000000035b0: d6550021 03fee123 00007fff
	v_or_b32_e32 v35, 0x400000, v112                           // 0000000035bc: 3846e0ff 00400000
	v_bfe_u32 v34, v109, 16, 1                                 // 0000000035c4: d6100022 0205216d
	v_cmp_u_f32_e32 vcc_lo, v112, v112                         // 0000000035cc: 7c30e170
	global_store_d16_hi_b16 v[0:1], v32, off offset:32         // 0000000035d0: ee09407c 10000000 00002000
	v_add3_u32 v32, v34, v109, 0x7fff                          // 0000000035dc: d6550020 03fedb22 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000035e8: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 0000000035ec: 02424721
	v_bfe_u32 v35, v108, 16, 1                                 // 0000000035f0: d6100023 0205216c
	v_or_b32_e32 v34, 0x400000, v109                           // 0000000035f8: 3844daff 00400000
	v_cmp_u_f32_e32 vcc_lo, v109, v109                         // 000000003600: 7c30db6d
	global_store_d16_hi_b16 v[4:5], v33, off offset:32         // 000000003604: ee09407c 10800000 00002004
	v_add3_u32 v33, v35, v108, 0x7fff                          // 000000003610: d6550021 03fed923 00007fff
	v_or_b32_e32 v35, 0x400000, v108                           // 00000000361c: 3846d8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003624: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003628: 02404520
	v_bfe_u32 v34, v106, 16, 1                                 // 00000000362c: d6100022 0205216a
	v_cmp_u_f32_e32 vcc_lo, v108, v108                         // 000000003634: 7c30d96c
	s_wait_alu depctr_va_vcc(0)                                // 000000003638: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 00000000363c: 02424721
	v_bfe_u32 v35, v105, 16, 1                                 // 000000003640: d6100023 02052169
	global_store_d16_hi_b16 v[6:7], v32, off offset:32         // 000000003648: ee09407c 10000000 00002006
	v_add3_u32 v32, v34, v106, 0x7fff                          // 000000003654: d6550020 03fed522 00007fff
	v_or_b32_e32 v34, 0x400000, v106                           // 000000003660: 3844d4ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v106, v106                         // 000000003668: 7c30d56a
	global_store_d16_hi_b16 v[10:11], v33, off offset:32       // 00000000366c: ee09407c 10800000 0000200a
	v_add3_u32 v33, v35, v105, 0x7fff                          // 000000003678: d6550021 03fed323 00007fff
	v_or_b32_e32 v35, 0x400000, v105                           // 000000003684: 3846d2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000368c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003690: 02404520
	v_bfe_u32 v34, v102, 16, 1                                 // 000000003694: d6100022 02052166
	v_cmp_u_f32_e32 vcc_lo, v105, v105                         // 00000000369c: 7c30d369
	s_wait_alu depctr_va_vcc(0)                                // 0000000036a0: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 0000000036a4: 02424721
	v_bfe_u32 v35, v107, 16, 1                                 // 0000000036a8: d6100023 0205216b
	global_store_d16_hi_b16 v[8:9], v32, off offset:32         // 0000000036b0: ee09407c 10000000 00002008
	v_add3_u32 v32, v34, v102, 0x7fff                          // 0000000036bc: d6550020 03fecd22 00007fff
	v_or_b32_e32 v34, 0x400000, v102                           // 0000000036c8: 3844ccff 00400000
	v_cmp_u_f32_e32 vcc_lo, v102, v102                         // 0000000036d0: 7c30cd66
	global_store_d16_hi_b16 v[14:15], v33, off offset:32       // 0000000036d4: ee09407c 10800000 0000200e
	v_add3_u32 v33, v35, v107, 0x7fff                          // 0000000036e0: d6550021 03fed723 00007fff
	v_or_b32_e32 v35, 0x400000, v107                           // 0000000036ec: 3846d6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000036f4: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 0000000036f8: 02404520
	v_bfe_u32 v34, v104, 16, 1                                 // 0000000036fc: d6100022 02052168
	v_cmp_u_f32_e32 vcc_lo, v107, v107                         // 000000003704: 7c30d76b
	s_wait_alu depctr_va_vcc(0)                                // 000000003708: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 00000000370c: 02424721
	v_bfe_u32 v35, v103, 16, 1                                 // 000000003710: d6100023 02052167
	global_store_d16_hi_b16 v[12:13], v32, off offset:32       // 000000003718: ee09407c 10000000 0000200c
	v_add3_u32 v32, v34, v104, 0x7fff                          // 000000003724: d6550020 03fed122 00007fff
	v_or_b32_e32 v34, 0x400000, v104                           // 000000003730: 3844d0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v104, v104                         // 000000003738: 7c30d168
	global_store_d16_hi_b16 v[18:19], v33, off offset:32       // 00000000373c: ee09407c 10800000 00002012
	v_add3_u32 v33, v35, v103, 0x7fff                          // 000000003748: d6550021 03fecf23 00007fff
	v_or_b32_e32 v35, 0x400000, v103                           // 000000003754: 3846ceff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000375c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003760: 02404520
	v_bfe_u32 v34, v101, 16, 1                                 // 000000003764: d6100022 02052165
	v_cmp_u_f32_e32 vcc_lo, v103, v103                         // 00000000376c: 7c30cf67
	s_wait_alu depctr_va_vcc(0)                                // 000000003770: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003774: 02424721
	v_bfe_u32 v35, v100, 16, 1                                 // 000000003778: d6100023 02052164
	global_store_d16_hi_b16 v[20:21], v32, off offset:32       // 000000003780: ee09407c 10000000 00002014
	v_add3_u32 v32, v34, v101, 0x7fff                          // 00000000378c: d6550020 03fecb22 00007fff
	v_or_b32_e32 v34, 0x400000, v101                           // 000000003798: 3844caff 00400000
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 0000000037a0: 7c30cb65
	global_store_d16_hi_b16 v[24:25], v33, off offset:32       // 0000000037a4: ee09407c 10800000 00002018
	v_add3_u32 v33, v35, v100, 0x7fff                          // 0000000037b0: d6550021 03fec923 00007fff
	v_or_b32_e32 v35, 0x400000, v100                           // 0000000037bc: 3846c8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000037c4: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 0000000037c8: 02404520
	v_bfe_u32 v34, v99, 16, 1                                  // 0000000037cc: d6100022 02052163
	v_cmp_u_f32_e32 vcc_lo, v100, v100                         // 0000000037d4: 7c30c964
	s_wait_alu depctr_va_vcc(0)                                // 0000000037d8: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 0000000037dc: 02424721
	v_bfe_u32 v35, v98, 16, 1                                  // 0000000037e0: d6100023 02052162
	global_store_d16_hi_b16 v[16:17], v32, off offset:32       // 0000000037e8: ee09407c 10000000 00002010
	v_add3_u32 v32, v34, v99, 0x7fff                           // 0000000037f4: d6550020 03fec722 00007fff
	v_or_b32_e32 v34, 0x400000, v99                            // 000000003800: 3844c6ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v99, v99                           // 000000003808: 7c30c763
	global_store_d16_hi_b16 v[22:23], v33, off offset:32       // 00000000380c: ee09407c 10800000 00002016
	v_add3_u32 v33, v35, v98, 0x7fff                           // 000000003818: d6550021 03fec523 00007fff
	v_or_b32_e32 v35, 0x400000, v98                            // 000000003824: 3846c4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000382c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003830: 02404520
	v_bfe_u32 v34, v96, 16, 1                                  // 000000003834: d6100022 02052160
	v_cmp_u_f32_e32 vcc_lo, v98, v98                           // 00000000383c: 7c30c562
	s_wait_alu depctr_va_vcc(0)                                // 000000003840: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003844: 02424721
	v_bfe_u32 v35, v93, 16, 1                                  // 000000003848: d6100023 0205215d
	global_store_d16_hi_b16 v[28:29], v32, off offset:32       // 000000003850: ee09407c 10000000 0000201c
	v_add3_u32 v32, v34, v96, 0x7fff                           // 00000000385c: d6550020 03fec122 00007fff
	v_or_b32_e32 v34, 0x400000, v96                            // 000000003868: 3844c0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v96, v96                           // 000000003870: 7c30c160
	global_store_d16_hi_b16 v[30:31], v33, off offset:32       // 000000003874: ee09407c 10800000 0000201e
	v_add3_u32 v33, v35, v93, 0x7fff                           // 000000003880: d6550021 03febb23 00007fff
	v_or_b32_e32 v35, 0x400000, v93                            // 00000000388c: 3846baff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003894: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003898: 02404520
	v_bfe_u32 v34, v89, 16, 1                                  // 00000000389c: d6100022 02052159
	v_cmp_u_f32_e32 vcc_lo, v93, v93                           // 0000000038a4: 7c30bb5d
	s_wait_alu depctr_va_vcc(0)                                // 0000000038a8: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 0000000038ac: 02424721
	v_bfe_u32 v35, v86, 16, 1                                  // 0000000038b0: d6100023 02052156
	global_store_d16_hi_b16 v[26:27], v32, off offset:32       // 0000000038b8: ee09407c 10000000 0000201a
	v_add3_u32 v32, v34, v89, 0x7fff                           // 0000000038c4: d6550020 03feb322 00007fff
	v_or_b32_e32 v34, 0x400000, v89                            // 0000000038d0: 3844b2ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 0000000038d8: 7c30b359
	global_store_d16_hi_b16 v[2:3], v33, off offset:64         // 0000000038dc: ee09407c 10800000 00004002
	v_add3_u32 v33, v35, v86, 0x7fff                           // 0000000038e8: d6550021 03fead23 00007fff
	v_or_b32_e32 v35, 0x400000, v86                            // 0000000038f4: 3846acff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000038fc: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003900: 02404520
	v_bfe_u32 v34, v85, 16, 1                                  // 000000003904: d6100022 02052155
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 00000000390c: 7c30ad56
	s_wait_alu depctr_va_vcc(0)                                // 000000003910: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003914: 02424721
	v_bfe_u32 v35, v82, 16, 1                                  // 000000003918: d6100023 02052152
	global_store_d16_hi_b16 v[0:1], v32, off offset:64         // 000000003920: ee09407c 10000000 00004000
	v_add3_u32 v32, v34, v85, 0x7fff                           // 00000000392c: d6550020 03feab22 00007fff
	v_or_b32_e32 v34, 0x400000, v85                            // 000000003938: 3844aaff 00400000
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 000000003940: 7c30ab55
	global_store_d16_hi_b16 v[4:5], v33, off offset:64         // 000000003944: ee09407c 10800000 00004004
	v_add3_u32 v33, v35, v82, 0x7fff                           // 000000003950: d6550021 03fea523 00007fff
	v_or_b32_e32 v35, 0x400000, v82                            // 00000000395c: 3846a4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003964: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003968: 02404520
	v_bfe_u32 v34, v75, 16, 1                                  // 00000000396c: d6100022 0205214b
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000003974: 7c30a552
	s_wait_alu depctr_va_vcc(0)                                // 000000003978: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 00000000397c: 02424721
	v_bfe_u32 v35, v73, 16, 1                                  // 000000003980: d6100023 02052149
	global_store_d16_hi_b16 v[6:7], v32, off offset:64         // 000000003988: ee09407c 10000000 00004006
	v_add3_u32 v32, v34, v75, 0x7fff                           // 000000003994: d6550020 03fe9722 00007fff
	v_or_b32_e32 v34, 0x400000, v75                            // 0000000039a0: 384496ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 0000000039a8: 7c30974b
	global_store_d16_hi_b16 v[10:11], v33, off offset:64       // 0000000039ac: ee09407c 10800000 0000400a
	v_add3_u32 v33, v35, v73, 0x7fff                           // 0000000039b8: d6550021 03fe9323 00007fff
	v_or_b32_e32 v35, 0x400000, v73                            // 0000000039c4: 384692ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000039cc: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 0000000039d0: 02404520
	v_bfe_u32 v34, v68, 16, 1                                  // 0000000039d4: d6100022 02052144
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 0000000039dc: 7c309349
	s_wait_alu depctr_va_vcc(0)                                // 0000000039e0: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 0000000039e4: 02424721
	v_bfe_u32 v35, v77, 16, 1                                  // 0000000039e8: d6100023 0205214d
	global_store_d16_hi_b16 v[8:9], v32, off offset:64         // 0000000039f0: ee09407c 10000000 00004008
	v_add3_u32 v32, v34, v68, 0x7fff                           // 0000000039fc: d6550020 03fe8922 00007fff
	v_or_b32_e32 v34, 0x400000, v68                            // 000000003a08: 384488ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 000000003a10: 7c308944
	global_store_d16_hi_b16 v[14:15], v33, off offset:64       // 000000003a14: ee09407c 10800000 0000400e
	v_add3_u32 v33, v35, v77, 0x7fff                           // 000000003a20: d6550021 03fe9b23 00007fff
	v_or_b32_e32 v35, 0x400000, v77                            // 000000003a2c: 38469aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a34: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003a38: 02404520
	v_bfe_u32 v34, v72, 16, 1                                  // 000000003a3c: d6100022 02052148
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000003a44: 7c309b4d
	s_wait_alu depctr_va_vcc(0)                                // 000000003a48: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003a4c: 02424721
	v_bfe_u32 v35, v69, 16, 1                                  // 000000003a50: d6100023 02052145
	global_store_d16_hi_b16 v[12:13], v32, off offset:64       // 000000003a58: ee09407c 10000000 0000400c
	v_add3_u32 v32, v34, v72, 0x7fff                           // 000000003a64: d6550020 03fe9122 00007fff
	v_or_b32_e32 v34, 0x400000, v72                            // 000000003a70: 384490ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 000000003a78: 7c309148
	global_store_d16_hi_b16 v[18:19], v33, off offset:64       // 000000003a7c: ee09407c 10800000 00004012
	v_add3_u32 v33, v35, v69, 0x7fff                           // 000000003a88: d6550021 03fe8b23 00007fff
	v_or_b32_e32 v35, 0x400000, v69                            // 000000003a94: 38468aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a9c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003aa0: 02404520
	v_bfe_u32 v34, v63, 16, 1                                  // 000000003aa4: d6100022 0205213f
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000003aac: 7c308b45
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab0: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003ab4: 02424721
	v_bfe_u32 v35, v60, 16, 1                                  // 000000003ab8: d6100023 0205213c
	global_store_d16_hi_b16 v[20:21], v32, off offset:64       // 000000003ac0: ee09407c 10000000 00004014
	v_add3_u32 v32, v34, v63, 0x7fff                           // 000000003acc: d6550020 03fe7f22 00007fff
	v_or_b32_e32 v34, 0x400000, v63                            // 000000003ad8: 38447eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000003ae0: 7c307f3f
	global_store_d16_hi_b16 v[24:25], v33, off offset:64       // 000000003ae4: ee09407c 10800000 00004018
	v_add3_u32 v33, v35, v60, 0x7fff                           // 000000003af0: d6550021 03fe7923 00007fff
	v_or_b32_e32 v35, 0x400000, v60                            // 000000003afc: 384678ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003b04: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003b08: 02404520
	v_bfe_u32 v34, v58, 16, 1                                  // 000000003b0c: d6100022 0205213a
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000003b14: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 000000003b18: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003b1c: 02424721
	v_bfe_u32 v35, v57, 16, 1                                  // 000000003b20: d6100023 02052139
	global_store_d16_hi_b16 v[16:17], v32, off offset:64       // 000000003b28: ee09407c 10000000 00004010
	v_add3_u32 v32, v34, v58, 0x7fff                           // 000000003b34: d6550020 03fe7522 00007fff
	v_or_b32_e32 v34, 0x400000, v58                            // 000000003b40: 384474ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000003b48: 7c30753a
	global_store_d16_hi_b16 v[22:23], v33, off offset:64       // 000000003b4c: ee09407c 10800000 00004016
	v_add3_u32 v33, v35, v57, 0x7fff                           // 000000003b58: d6550021 03fe7323 00007fff
	v_or_b32_e32 v35, 0x400000, v57                            // 000000003b64: 384672ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003b6c: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003b70: 02404520
	v_bfe_u32 v34, v55, 16, 1                                  // 000000003b74: d6100022 02052137
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000003b7c: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 000000003b80: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003b84: 02424721
	v_bfe_u32 v35, v54, 16, 1                                  // 000000003b88: d6100023 02052136
	global_store_d16_hi_b16 v[28:29], v32, off offset:64       // 000000003b90: ee09407c 10000000 0000401c
	v_add3_u32 v32, v34, v55, 0x7fff                           // 000000003b9c: d6550020 03fe6f22 00007fff
	v_or_b32_e32 v34, 0x400000, v55                            // 000000003ba8: 38446eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000003bb0: 7c306f37
	global_store_d16_hi_b16 v[30:31], v33, off offset:64       // 000000003bb4: ee09407c 10800000 0000401e
	v_add3_u32 v33, v35, v54, 0x7fff                           // 000000003bc0: d6550021 03fe6d23 00007fff
	v_or_b32_e32 v35, 0x400000, v54                            // 000000003bcc: 38466cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003bd4: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003bd8: 02404520
	v_bfe_u32 v34, v53, 16, 1                                  // 000000003bdc: d6100022 02052135
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 000000003be4: 7c306d36
	s_wait_alu depctr_va_vcc(0)                                // 000000003be8: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo                    // 000000003bec: 02424721
	v_bfe_u32 v35, v52, 16, 1                                  // 000000003bf0: d6100023 02052134
	global_store_d16_hi_b16 v[26:27], v32, off offset:64       // 000000003bf8: ee09407c 10000000 0000401a
	v_add3_u32 v32, v34, v53, 0x7fff                           // 000000003c04: d6550020 03fe6b22 00007fff
	v_or_b32_e32 v34, 0x400000, v53                            // 000000003c10: 38446aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 000000003c18: 7c306b35
	global_store_d16_hi_b16 v[2:3], v33, off offset:96         // 000000003c1c: ee09407c 10800000 00006002
	v_add3_u32 v2, v35, v52, 0x7fff                            // 000000003c28: d6550002 03fe6923 00007fff
	v_or_b32_e32 v3, 0x400000, v52                             // 000000003c34: 380668ff 00400000
	v_bfe_u32 v33, v51, 16, 1                                  // 000000003c3c: d6100021 02052133
	s_wait_alu depctr_va_vcc(0)                                // 000000003c44: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003c48: 02404520
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000003c4c: 7c306934
	global_store_d16_hi_b16 v[0:1], v32, off offset:96         // 000000003c50: ee09407c 10000000 00006000
	s_wait_alu depctr_va_vcc(0)                                // 000000003c5c: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003c60: 02040702
	v_bfe_u32 v3, v49, 16, 1                                   // 000000003c64: d6100003 02052131
	v_add3_u32 v0, v33, v51, 0x7fff                            // 000000003c6c: d6550000 03fe6721 00007fff
	v_or_b32_e32 v1, 0x400000, v51                             // 000000003c78: 380266ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000003c80: 7c306733
	global_store_d16_hi_b16 v[4:5], v2, off offset:96          // 000000003c84: ee09407c 01000000 00006004
	v_add3_u32 v2, v3, v49, 0x7fff                             // 000000003c90: d6550002 03fe6303 00007fff
	v_or_b32_e32 v3, 0x400000, v49                             // 000000003c9c: 380662ff 00400000
	v_bfe_u32 v4, v39, 16, 1                                   // 000000003ca4: d6100004 02052127
	s_wait_alu depctr_va_vcc(0)                                // 000000003cac: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003cb0: 02000300
	v_bfe_u32 v1, v47, 16, 1                                   // 000000003cb4: d6100001 0205212f
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 000000003cbc: 7c306331
	v_or_b32_e32 v5, 0x400000, v40                             // 000000003cc0: 380a50ff 00400000
	v_add3_u32 v4, v4, v39, 0x7fff                             // 000000003cc8: d6550004 03fe4f04 00007fff
	global_store_d16_hi_b16 v[6:7], v0, off offset:96          // 000000003cd4: ee09407c 00000000 00006006
	v_add3_u32 v0, v1, v47, 0x7fff                             // 000000003ce0: d6550000 03fe5f01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003cec: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003cf0: 02040702
	v_bfe_u32 v3, v46, 16, 1                                   // 000000003cf4: d6100003 0205212e
	v_or_b32_e32 v1, 0x400000, v47                             // 000000003cfc: 38025eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000003d04: 7c305f2f
	v_or_b32_e32 v6, 0x400000, v39                             // 000000003d08: 380c4eff 00400000
	global_store_d16_hi_b16 v[10:11], v2, off offset:96        // 000000003d10: ee09407c 01000000 0000600a
	v_add3_u32 v2, v3, v46, 0x7fff                             // 000000003d1c: d6550002 03fe5d03 00007fff
	v_or_b32_e32 v3, 0x400000, v46                             // 000000003d28: 38065cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003d30: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003d34: 02000300
	v_bfe_u32 v1, v43, 16, 1                                   // 000000003d38: d6100001 0205212b
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000003d40: 7c305d2e
	v_or_b32_e32 v7, 0x400000, v38                             // 000000003d44: 380e4cff 00400000
	global_store_d16_hi_b16 v[8:9], v0, off offset:96          // 000000003d4c: ee09407c 00000000 00006008
	v_add3_u32 v0, v1, v43, 0x7fff                             // 000000003d58: d6550000 03fe5701 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003d64: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003d68: 02040702
	v_bfe_u32 v3, v48, 16, 1                                   // 000000003d6c: d6100003 02052130
	v_or_b32_e32 v1, 0x400000, v43                             // 000000003d74: 380256ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 000000003d7c: 7c30572b
	global_store_d16_hi_b16 v[14:15], v2, off offset:96        // 000000003d80: ee09407c 01000000 0000600e
	v_add3_u32 v2, v3, v48, 0x7fff                             // 000000003d8c: d6550002 03fe6103 00007fff
	v_or_b32_e32 v3, 0x400000, v48                             // 000000003d98: 380660ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003da0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003da4: 02000300
	v_bfe_u32 v1, v45, 16, 1                                   // 000000003da8: d6100001 0205212d
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000003db0: 7c306130
	global_store_d16_hi_b16 v[12:13], v0, off offset:96        // 000000003db4: ee09407c 00000000 0000600c
	v_add3_u32 v0, v1, v45, 0x7fff                             // 000000003dc0: d6550000 03fe5b01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003dcc: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003dd0: 02040702
	v_bfe_u32 v3, v44, 16, 1                                   // 000000003dd4: d6100003 0205212c
	v_or_b32_e32 v1, 0x400000, v45                             // 000000003ddc: 38025aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000003de4: 7c305b2d
	global_store_d16_hi_b16 v[18:19], v2, off offset:96        // 000000003de8: ee09407c 01000000 00006012
	v_add3_u32 v2, v3, v44, 0x7fff                             // 000000003df4: d6550002 03fe5903 00007fff
	v_or_b32_e32 v3, 0x400000, v44                             // 000000003e00: 380658ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e08: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003e0c: 02000300
	v_bfe_u32 v1, v42, 16, 1                                   // 000000003e10: d6100001 0205212a
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000003e18: 7c30592c
	global_store_d16_hi_b16 v[20:21], v0, off offset:96        // 000000003e1c: ee09407c 00000000 00006014
	v_add3_u32 v0, v1, v42, 0x7fff                             // 000000003e28: d6550000 03fe5501 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003e34: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003e38: 02040702
	v_bfe_u32 v3, v41, 16, 1                                   // 000000003e3c: d6100003 02052129
	v_or_b32_e32 v1, 0x400000, v42                             // 000000003e44: 380254ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000003e4c: 7c30552a
	global_store_d16_hi_b16 v[24:25], v2, off offset:96        // 000000003e50: ee09407c 01000000 00006018
	v_add3_u32 v2, v3, v41, 0x7fff                             // 000000003e5c: d6550002 03fe5303 00007fff
	v_or_b32_e32 v3, 0x400000, v41                             // 000000003e68: 380652ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003e70: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003e74: 02000300
	v_bfe_u32 v1, v40, 16, 1                                   // 000000003e78: d6100001 02052128
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000003e80: 7c305329
	s_delay_alu instid0(valu_dep_2)                            // 000000003e84: bf870002
	v_add3_u32 v1, v1, v40, 0x7fff                             // 000000003e88: d6550001 03fe5101 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003e98: 02040702
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 000000003e9c: 7c305128
	v_bfe_u32 v3, v38, 16, 1                                   // 000000003ea0: d6100003 02052126
	s_wait_alu depctr_va_vcc(0)                                // 000000003ea8: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo                       // 000000003eac: 02020b01
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 000000003eb0: 7c304f27
	s_delay_alu instid0(valu_dep_3)                            // 000000003eb4: bf870003
	v_add3_u32 v3, v3, v38, 0x7fff                             // 000000003eb8: d6550003 03fe4d03 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003ec4: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       // 000000003ec8: 02080d04
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 000000003ecc: 7c304d26
	s_wait_alu depctr_va_vcc(0)                                // 000000003ed0: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo                       // 000000003ed4: 02060f03
	s_clause 0x3                                               // 000000003ed8: bf850003
	global_store_d16_hi_b16 v[16:17], v0, off offset:96        // 000000003edc: ee09407c 00000000 00006010
	global_store_d16_hi_b16 v[22:23], v2, off offset:96        // 000000003ee8: ee09407c 01000000 00006016
	global_store_d16_hi_b16 v[28:29], v1, off offset:96        // 000000003ef4: ee09407c 00800000 0000601c
	global_store_d16_hi_b16 v[30:31], v4, off offset:96        // 000000003f00: ee09407c 02000000 0000601e
	global_store_d16_hi_b16 v[26:27], v3, off offset:96        // 000000003f0c: ee09407c 01800000 0000601a
	s_nop 0                                                    // 000000003f18: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000003f1c: bfb60003
	s_endpgm                                                   // 000000003f20: bfb00000
	s_code_end                                                 // 000000003f24: bf9f0000
	s_code_end                                                 // 000000003f28: bf9f0000
	s_code_end                                                 // 000000003f2c: bf9f0000
	s_code_end                                                 // 000000003f30: bf9f0000
	s_code_end                                                 // 000000003f34: bf9f0000
	s_code_end                                                 // 000000003f38: bf9f0000
	s_code_end                                                 // 000000003f3c: bf9f0000
	s_code_end                                                 // 000000003f40: bf9f0000
	s_code_end                                                 // 000000003f44: bf9f0000
	s_code_end                                                 // 000000003f48: bf9f0000
	s_code_end                                                 // 000000003f4c: bf9f0000
	s_code_end                                                 // 000000003f50: bf9f0000
	s_code_end                                                 // 000000003f54: bf9f0000
	s_code_end                                                 // 000000003f58: bf9f0000
	s_code_end                                                 // 000000003f5c: bf9f0000
	s_code_end                                                 // 000000003f60: bf9f0000
	s_code_end                                                 // 000000003f64: bf9f0000
	s_code_end                                                 // 000000003f68: bf9f0000
	s_code_end                                                 // 000000003f6c: bf9f0000
	s_code_end                                                 // 000000003f70: bf9f0000
	s_code_end                                                 // 000000003f74: bf9f0000
	s_code_end                                                 // 000000003f78: bf9f0000
	s_code_end                                                 // 000000003f7c: bf9f0000
	s_code_end                                                 // 000000003f80: bf9f0000
	s_code_end                                                 // 000000003f84: bf9f0000
	s_code_end                                                 // 000000003f88: bf9f0000
	s_code_end                                                 // 000000003f8c: bf9f0000
	s_code_end                                                 // 000000003f90: bf9f0000
	s_code_end                                                 // 000000003f94: bf9f0000
	s_code_end                                                 // 000000003f98: bf9f0000
	s_code_end                                                 // 000000003f9c: bf9f0000
	s_code_end                                                 // 000000003fa0: bf9f0000
	s_code_end                                                 // 000000003fa4: bf9f0000
	s_code_end                                                 // 000000003fa8: bf9f0000
	s_code_end                                                 // 000000003fac: bf9f0000
	s_code_end                                                 // 000000003fb0: bf9f0000
	s_code_end                                                 // 000000003fb4: bf9f0000
	s_code_end                                                 // 000000003fb8: bf9f0000
	s_code_end                                                 // 000000003fbc: bf9f0000
	s_code_end                                                 // 000000003fc0: bf9f0000
	s_code_end                                                 // 000000003fc4: bf9f0000
	s_code_end                                                 // 000000003fc8: bf9f0000
	s_code_end                                                 // 000000003fcc: bf9f0000
	s_code_end                                                 // 000000003fd0: bf9f0000
	s_code_end                                                 // 000000003fd4: bf9f0000
	s_code_end                                                 // 000000003fd8: bf9f0000
	s_code_end                                                 // 000000003fdc: bf9f0000
	s_code_end                                                 // 000000003fe0: bf9f0000
	s_code_end                                                 // 000000003fe4: bf9f0000
	s_code_end                                                 // 000000003fe8: bf9f0000
	s_code_end                                                 // 000000003fec: bf9f0000
	s_code_end                                                 // 000000003ff0: bf9f0000
	s_code_end                                                 // 000000003ff4: bf9f0000
	s_code_end                                                 // 000000003ff8: bf9f0000
	s_code_end                                                 // 000000003ffc: bf9f0000
	s_code_end                                                 // 000000004000: bf9f0000
	s_code_end                                                 // 000000004004: bf9f0000
	s_code_end                                                 // 000000004008: bf9f0000
	s_code_end                                                 // 00000000400c: bf9f0000
	s_code_end                                                 // 000000004010: bf9f0000
	s_code_end                                                 // 000000004014: bf9f0000
	s_code_end                                                 // 000000004018: bf9f0000
	s_code_end                                                 // 00000000401c: bf9f0000
	s_code_end                                                 // 000000004020: bf9f0000
	s_code_end                                                 // 000000004024: bf9f0000
	s_code_end                                                 // 000000004028: bf9f0000
	s_code_end                                                 // 00000000402c: bf9f0000
	s_code_end                                                 // 000000004030: bf9f0000
	s_code_end                                                 // 000000004034: bf9f0000
	s_code_end                                                 // 000000004038: bf9f0000
	s_code_end                                                 // 00000000403c: bf9f0000
	s_code_end                                                 // 000000004040: bf9f0000
	s_code_end                                                 // 000000004044: bf9f0000
	s_code_end                                                 // 000000004048: bf9f0000
	s_code_end                                                 // 00000000404c: bf9f0000
	s_code_end                                                 // 000000004050: bf9f0000
	s_code_end                                                 // 000000004054: bf9f0000
	s_code_end                                                 // 000000004058: bf9f0000
	s_code_end                                                 // 00000000405c: bf9f0000
	s_code_end                                                 // 000000004060: bf9f0000
	s_code_end                                                 // 000000004064: bf9f0000
	s_code_end                                                 // 000000004068: bf9f0000
	s_code_end                                                 // 00000000406c: bf9f0000
	s_code_end                                                 // 000000004070: bf9f0000
	s_code_end                                                 // 000000004074: bf9f0000
	s_code_end                                                 // 000000004078: bf9f0000
	s_code_end                                                 // 00000000407c: bf9f0000
	s_code_end                                                 // 000000004080: bf9f0000
	s_code_end                                                 // 000000004084: bf9f0000
	s_code_end                                                 // 000000004088: bf9f0000
	s_code_end                                                 // 00000000408c: bf9f0000
	s_code_end                                                 // 000000004090: bf9f0000
	s_code_end                                                 // 000000004094: bf9f0000
	s_code_end                                                 // 000000004098: bf9f0000
	s_code_end                                                 // 00000000409c: bf9f0000
	s_code_end                                                 // 0000000040a0: bf9f0000
	s_code_end                                                 // 0000000040a4: bf9f0000
	s_code_end                                                 // 0000000040a8: bf9f0000
	s_code_end                                                 // 0000000040ac: bf9f0000
	s_code_end                                                 // 0000000040b0: bf9f0000
	s_code_end                                                 // 0000000040b4: bf9f0000
	s_code_end                                                 // 0000000040b8: bf9f0000
	s_code_end                                                 // 0000000040bc: bf9f0000
	s_code_end                                                 // 0000000040c0: bf9f0000
	s_code_end                                                 // 0000000040c4: bf9f0000
	s_code_end                                                 // 0000000040c8: bf9f0000
	s_code_end                                                 // 0000000040cc: bf9f0000
	s_code_end                                                 // 0000000040d0: bf9f0000
	s_code_end                                                 // 0000000040d4: bf9f0000
	s_code_end                                                 // 0000000040d8: bf9f0000
	s_code_end                                                 // 0000000040dc: bf9f0000
	s_code_end                                                 // 0000000040e0: bf9f0000
	s_code_end                                                 // 0000000040e4: bf9f0000
	s_code_end                                                 // 0000000040e8: bf9f0000
	s_code_end                                                 // 0000000040ec: bf9f0000
	s_code_end                                                 // 0000000040f0: bf9f0000
	s_code_end                                                 // 0000000040f4: bf9f0000
	s_code_end                                                 // 0000000040f8: bf9f0000
	s_code_end                                                 // 0000000040fc: bf9f0000
