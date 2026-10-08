
/tmp/tmplsmdcxho.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2>:
	s_load_b128 s[28:31], s[0:1], 0xc8                         // 000000001b00: f4004700 f80000c8
	v_lshrrev_b32_e32 v6, 1, v0                                // 000000001b08: 320c0081
	s_mov_b32 s6, ttmp7                                        // 000000001b0c: be860073
	s_ashr_i32 s7, ttmp7, 31                                   // 000000001b10: 86079f73
	s_clause 0x4                                               // 000000001b14: bf850004
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b18: f4002080 f80000d8
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001b20: f4002280 f8000008
	s_load_b64 s[12:13], s[0:1], 0x30                          // 000000001b28: f4002300 f8000030
	s_load_b64 s[8:9], s[0:1], 0x58                            // 000000001b30: f4002200 f8000058
	s_load_b64 s[34:35], s[0:1], 0x80                          // 000000001b38: f4002880 f8000080
	s_lshl_b64 s[6:7], s[6:7], 7                               // 000000001b40: 84868706
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b44: bf870009
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v2, s7            // 000000001b48: ca100080 6b020007
	v_or_b32_e32 v1, s6, v6                                    // 000000001b50: 38020c06
	s_mov_b32 s4, ttmp9                                        // 000000001b54: be840075
	s_ashr_i32 s5, ttmp9, 31                                   // 000000001b58: 86059f75
	v_dual_mov_b32 v3, s7 :: v_dual_mov_b32 v4, s7             // 000000001b5c: ca100007 03040007
	s_lshl_b64 s[26:27], s[4:5], 7                             // 000000001b64: 849a8704
	v_lshlrev_b32_e32 v5, 1, v0                                // 000000001b68: 300a0081
	v_mul_u32_u24_e32 v7, 48, v6                               // 000000001b6c: 160e0cb0
	v_and_b32_e32 v25, 8, v6                                   // 000000001b70: 36320c88
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v104, 0           // 000000001b74: ca100080 89680080
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v102, 0           // 000000001b7c: ca100080 87660080
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	s_add_nc_u64 s[4:5], s[28:29], -1                          // 000000001b88: a984c11c
	s_lshr_b64 s[36:37], s[2:3], 5                             // 000000001b8c: 85a48502
	v_cmp_gt_u64_e32 vcc_lo, s[4:5], v[1:2]                    // 000000001b90: 7cb80204
	v_and_b32_e32 v58, 15, v0                                  // 000000001b94: 3674008f
	v_lshlrev_b32_e32 v0, 4, v0                                // 000000001b98: 30000084
	v_and_b32_e32 v59, 64, v5                                  // 000000001b9c: 36760ac0
	v_or_b32_e32 v5, s26, v6                                   // 000000001ba0: 380a0c1a
	v_and_b32_e32 v2, 0x60, v6                                 // 000000001ba4: 36040cff 00000060
	v_cndmask_b32_e32 v1, s4, v1, vcc_lo                       // 000000001bac: 02020204
	v_cndmask_b32_e32 v4, s5, v4, vcc_lo                       // 000000001bb0: 02080805
	v_and_b32_e32 v8, 16, v0                                   // 000000001bb4: 36100090
	v_or_b32_e32 v27, 1, v25                                   // 000000001bb8: 38363281
	v_or_b32_e32 v9, 16, v2                                    // 000000001bbc: 38120490
	v_mul_lo_u32 v10, v1, s3                                   // 000000001bc0: d72c000a 02000701
	v_mad_co_u64_u32 v[0:1], null, v1, s2, s[10:11]            // 000000001bc8: d6fe7c00 00280501
	v_mul_lo_u32 v11, v4, s2                                   // 000000001bd0: d72c000b 02000504
	v_add_nc_u32_e32 v110, v7, v8                              // 000000001bd8: 4adc1107
	v_mul_lo_u32 v7, s3, v5                                    // 000000001bdc: d72c0007 02020a03
	v_mad_co_u64_u32 v[4:5], null, s2, v5, s[12:13]            // 000000001be4: d6fe7c04 00320a02
	s_mul_i32 s2, s2, s27                                      // 000000001bec: 96021b02
	v_or_b32_e32 v34, s6, v9                                   // 000000001bf0: 38441206
	s_lshr_b32 s10, s3, 5                                      // 000000001bf4: 850a8503
	v_add_co_u32 v112, vcc_lo, v0, v8                          // 000000001bf8: d7006a70 02021100
	v_add3_u32 v1, v11, v1, v10                                // 000000001c00: d6550001 042a030b
	v_or_b32_e32 v0, v59, v58                                  // 000000001c08: 3800753b
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v100, 0           // 000000001c0c: ca100080 7f640080
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v90, 0            // 000000001c14: ca100080 695a0080
	s_wait_alu depctr_va_vcc(0)                                // 000000001c1c: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, 0, v1, vcc_lo              // 000000001c20: d5207c71 01aa0280
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c28: bf88ff9e
	v_add3_u32 v1, v7, v5, s2                                  // 000000001c2c: d6550001 000a0b07
	v_or_b32_e32 v5, v9, v58                                   // 000000001c34: 380a7509
	v_or_b32_e32 v26, 48, v0                                   // 000000001c38: 383400b0
	v_add_co_u32 v115, vcc_lo, v4, v8                          // 000000001c3c: d7006a73 02021104
	s_wait_alu depctr_va_vcc(0)                                // 000000001c44: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, 0, v1, vcc_lo              // 000000001c48: d5207c74 01aa0280
	v_mul_u32_u24_e32 v1, 48, v5                               // 000000001c50: 16020ab0
	v_mul_u32_u24_e32 v5, 48, v26                              // 000000001c54: 160a34b0
	v_or_b32_e32 v20, 16, v0                                   // 000000001c58: 38280090
	v_or_b32_e32 v22, 32, v0                                   // 000000001c5c: 382c00a0
	v_mov_b32_e32 v9, s7                                       // 000000001c60: 7e120207
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v88, 0            // 000000001c64: ca100080 67580080
	v_or_b32_e32 v41, v5, v25                                  // 000000001c6c: 38523305
	v_mov_b32_e32 v5, s7                                       // 000000001c70: 7e0a0207
	v_or_b32_e32 v14, s6, v2                                   // 000000001c74: 381c0406
	v_or_b32_e32 v2, v2, v58                                   // 000000001c78: 38047502
	v_mul_u32_u24_e32 v4, 48, v20                              // 000000001c7c: 160828b0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v101, 0           // 000000001c80: ca100080 6a640080
	v_mov_b32_e32 v86, 0                                       // 000000001c88: 7eac0280
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001c8c: bf870214
	v_mul_u32_u24_e32 v2, 48, v2                               // 000000001c90: 160404b0
	v_or_b32_e32 v39, v4, v25                                  // 000000001c94: 384e3304
	v_mul_u32_u24_e32 v4, 48, v22                              // 000000001c98: 16082cb0
	v_or_b32_e32 v22, s26, v22                                 // 000000001c9c: 382c2c1a
	v_mov_b32_e32 v108, 0                                      // 000000001ca0: 7ed80280
	v_or_b32_e32 v117, v2, v25                                 // 000000001ca4: 38ea3302
	v_mul_u32_u24_e32 v2, 48, v0                               // 000000001ca8: 160400b0
	v_or_b32_e32 v40, v4, v25                                  // 000000001cac: 38503304
	v_or_b32_e32 v4, v27, v14                                  // 000000001cb0: 38081d1b
	v_or_b32_e32 v0, s26, v0                                   // 000000001cb4: 3800001a
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v84, 0             // 000000001cb8: ca100080 5b540080
	v_or_b32_e32 v38, v2, v25                                  // 000000001cc0: 384c3302
	v_or_b32_e32 v2, v14, v25                                  // 000000001cc4: 3804330e
	v_add_nc_u32_e32 v141, 0x1800, v40                         // 000000001cc8: 4b1a50ff 00001800
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v74, 0             // 000000001cd0: ca100080 594a0080
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001cd8: bf870214
	v_add_nc_u32_e32 v139, 0x1800, v38                         // 000000001cdc: 4b164cff 00001800
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[2:3]                  // 000000001ce4: 7ca8041c
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v72, 0             // 000000001ce8: ca100080 57480080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v70, 0             // 000000001cf0: ca100080 55460080
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v68, 0             // 000000001cf8: ca100080 4b440080
	s_wait_alu depctr_va_vcc(0)                                // 000000001d00: bf88ff9d
	v_dual_cndmask_b32 v6, 0, v2 :: v_dual_cndmask_b32 v7, 0, v3// 000000001d04: ca520480 06060680
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[4:5]                  // 000000001d0c: 7ca8081c
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v124, 0            // 000000001d10: ca100080 497c0080
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_4)// 000000001d18: bf870213
	v_mul_lo_u32 v16, s10, v6                                  // 000000001d1c: d72c0010 02020c0a
	v_mul_lo_u32 v15, s36, v7                                  // 000000001d24: d72c000f 02020e24
	v_mad_co_u64_u32 v[6:7], null, s36, v6, s[8:9]             // 000000001d2c: d6fe7c06 00220c24
	s_wait_alu depctr_va_vcc(0)                                // 000000001d34: bf88ff9d
	v_cndmask_b32_e32 v10, 0, v4, vcc_lo                       // 000000001d38: 02140880
	v_or_b32_e32 v29, 2, v25                                   // 000000001d3c: 383a3282
	v_or_b32_e32 v30, 3, v25                                   // 000000001d40: 383c3283
	v_or_b32_e32 v32, 4, v25                                   // 000000001d44: 38403284
	v_mov_b32_e32 v13, s7                                      // 000000001d48: 7e1a0207
	v_mul_lo_u32 v18, s10, v10                                 // 000000001d4c: d72c0012 0202140a
	v_or_b32_e32 v8, v29, v14                                  // 000000001d54: 38101d1d
	v_cndmask_b32_e32 v4, 0, v5, vcc_lo                        // 000000001d58: 02080a80
	v_add3_u32 v7, v16, v7, v15                                // 000000001d5c: d6550007 043e0f10
	v_or_b32_e32 v36, 7, v25                                   // 000000001d64: 38483287
	v_or_b32_e32 v33, 5, v25                                   // 000000001d68: 38423285
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[8:9]                  // 000000001d6c: 7ca8101c
	v_or_b32_e32 v35, 6, v25                                   // 000000001d70: 38463286
	v_or_b32_e32 v118, v1, v25                                 // 000000001d74: 38ec3301
	v_mov_b32_e32 v1, s27                                      // 000000001d78: 7e02021b
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v120, 0            // 000000001d7c: ca100080 47780080
	s_wait_alu depctr_va_vcc(0)                                // 000000001d84: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v8, vcc_lo                       // 000000001d88: 02161080
	v_mul_lo_u32 v17, s36, v4                                  // 000000001d8c: d72c0011 02020824
	v_or_b32_e32 v4, v30, v14                                  // 000000001d94: 38081d1e
	v_cndmask_b32_e32 v12, 0, v9, vcc_lo                       // 000000001d98: 02181280
	v_mad_co_u64_u32 v[8:9], null, s36, v10, s[8:9]            // 000000001d9c: d6fe7c08 00221424
	v_mul_lo_u32 v21, s10, v11                                 // 000000001da4: d72c0015 0202160a
	v_mad_co_u64_u32 v[10:11], null, s36, v11, s[8:9]          // 000000001dac: d6fe7c0a 00221624
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[4:5]                  // 000000001db4: 7ca8081c
	v_mul_lo_u32 v19, s36, v12                                 // 000000001db8: d72c0013 02021824
	v_or_b32_e32 v12, v32, v14                                 // 000000001dc0: 38181d20
	v_add_nc_u32_e32 v140, 0x1800, v39                         // 000000001dc4: 4b184eff 00001800
	v_mov_b32_e32 v132, 0                                      // 000000001dcc: 7f080280
	v_add3_u32 v9, v18, v9, v17                                // 000000001dd0: d6550009 04461312
	s_wait_alu depctr_va_vcc(0)                                // 000000001dd8: bf88ff9d
	v_dual_cndmask_b32 v15, 0, v4 :: v_dual_cndmask_b32 v4, 0, v5// 000000001ddc: ca520880 0f040a80
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[12:13]                // 000000001de4: 7ca8181c
	v_add3_u32 v11, v21, v11, v19                              // 000000001de8: d655000b 044e1715
	v_or_b32_e32 v18, v36, v14                                 // 000000001df0: 38241d24
	s_delay_alu instid0(valu_dep_4)                            // 000000001df4: bf870004
	v_mul_lo_u32 v24, s10, v15                                 // 000000001df8: d72c0018 02021e0a
	v_mul_lo_u32 v23, s36, v4                                  // 000000001e00: d72c0017 02020824
	v_mov_b32_e32 v19, s7                                      // 000000001e08: 7e260207
	s_wait_alu depctr_va_vcc(0)                                // 000000001e0c: bf88ff9d
	v_dual_cndmask_b32 v21, 0, v12 :: v_dual_cndmask_b32 v16, 0, v13// 000000001e10: ca521880 15101a80
	v_mad_co_u64_u32 v[12:13], null, s36, v15, s[8:9]          // 000000001e18: d6fe7c0c 00221e24
	v_or_b32_e32 v4, v33, v14                                  // 000000001e20: 38081d21
	v_cmp_gt_i64_e64 s4, s[30:31], v[0:1]                      // 000000001e24: d4540004 0202001e
	s_delay_alu instid0(valu_dep_4)                            // 000000001e2c: bf870004
	v_mul_lo_u32 v31, s10, v21                                 // 000000001e30: d72c001f 02022a0a
	v_mul_lo_u32 v28, s36, v16                                 // 000000001e38: d72c001c 02022024
	v_or_b32_e32 v16, v35, v14                                 // 000000001e40: 38201d23
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[4:5]                  // 000000001e44: 7ca8081c
	v_mov_b32_e32 v17, s7                                      // 000000001e48: 7e220207
	v_add3_u32 v13, v24, v13, v23                              // 000000001e4c: d655000d 045e1b18
	v_mov_b32_e32 v24, s27                                     // 000000001e54: 7e30021b
	v_cmp_gt_i64_e64 s2, s[28:29], v[18:19]                    // 000000001e58: d4540002 0202241c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e60: bf88ff9d
	v_dual_mov_b32 v23, s27 :: v_dual_cndmask_b32 v4, 0, v4    // 000000001e64: ca12001b 17040880
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000001e6c: 020a0a80
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[16:17]                // 000000001e70: 7ca8201c
	v_mad_co_u64_u32 v[14:15], null, s36, v21, s[8:9]          // 000000001e74: d6fe7c0e 00222a24
	s_wait_alu depctr_va_sdst(0)                               // 000000001e7c: bf88f19f
	v_cndmask_b32_e64 v43, 0, v18, s2                          // 000000001e80: d501002b 000a2480
	v_cndmask_b32_e64 v44, 0, v19, s2                          // 000000001e88: d501002c 000a2680
	v_cmp_gt_i64_e64 s2, s[30:31], v[22:23]                    // 000000001e90: d4540002 02022c1e
	v_mul_lo_u32 v42, s10, v4                                  // 000000001e98: d72c002a 0202080a
	s_wait_alu depctr_va_vcc(0)                                // 000000001ea0: bf88ff9d
	v_cndmask_b32_e32 v21, 0, v17, vcc_lo                      // 000000001ea4: 022a2280
	v_add_nc_u32_e32 v142, 0x1800, v41                         // 000000001ea8: 4b1c52ff 00001800
	v_mul_lo_u32 v44, s36, v44                                 // 000000001eb0: d72c002c 02025824
	v_add3_u32 v15, v31, v15, v28                              // 000000001eb8: d655000f 04721f1f
	s_wait_alu depctr_va_sdst(0)                               // 000000001ec0: bf88f19f
	v_cndmask_b32_e64 v131, 0, v23, s2                         // 000000001ec4: d5010083 000a2e80
	v_or_b32_e32 v23, s26, v26                                 // 000000001ecc: 382e341a
	v_mov_b32_e32 v26, s7                                      // 000000001ed0: 7e340207
	v_mul_lo_u32 v37, s36, v5                                  // 000000001ed4: d72c0025 02020a24
	v_dual_cndmask_b32 v5, 0, v16 :: v_dual_mov_b32 v28, s7    // 000000001edc: ca502080 051c0007
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000001ee4: bf870234
	v_cmp_gt_i64_e32 vcc_lo, s[30:31], v[23:24]                // 000000001ee8: 7ca82e1e
	v_mad_co_u64_u32 v[16:17], null, s36, v4, s[8:9]           // 000000001eec: d6fe7c10 00220824
	v_or_b32_e32 v4, s26, v20                                  // 000000001ef4: 3808281a
	v_mul_lo_u32 v46, s10, v5                                  // 000000001ef8: d72c002e 02020a0a
	v_mad_co_u64_u32 v[18:19], null, s36, v5, s[8:9]           // 000000001f00: d6fe7c12 00220a24
	s_wait_alu depctr_va_vcc(0)                                // 000000001f08: bf88ff9d
	v_dual_mov_b32 v5, s27 :: v_dual_cndmask_b32 v136, 0, v23  // 000000001f0c: ca12001b 05882e80
	v_mul_lo_u32 v47, s10, v43                                 // 000000001f14: d72c002f 0202560a
	v_cndmask_b32_e32 v134, 0, v24, vcc_lo                     // 000000001f1c: 030c3080
	v_mul_lo_u32 v45, s36, v21                                 // 000000001f20: d72c002d 02022a24
	s_delay_alu instid0(valu_dep_4)                            // 000000001f28: bf870004
	v_cmp_gt_i64_e64 s3, s[30:31], v[4:5]                      // 000000001f2c: d4540003 0202081e
	v_mad_co_u64_u32 v[20:21], null, s36, v43, s[8:9]          // 000000001f34: d6fe7c14 00225624
	v_cndmask_b32_e64 v133, 0, v22, s2                         // 000000001f3c: d5010085 000a2c80
	v_dual_mov_b32 v31, s7 :: v_dual_mov_b32 v138, 0           // 000000001f44: ca100007 1f8a0080
	v_add3_u32 v17, v42, v17, v37                              // 000000001f4c: d6550011 0496232a
	s_wait_alu depctr_va_sdst(0)                               // 000000001f54: bf88f19f
	v_cndmask_b32_e64 v129, 0, v5, s3                          // 000000001f58: d5010081 000e0a80
	v_cndmask_b32_e64 v130, 0, v4, s3                          // 000000001f60: d5010082 000e0880
	v_mov_b32_e32 v5, s7                                       // 000000001f68: 7e0a0207
	v_or_b32_e32 v4, v34, v25                                  // 000000001f6c: 38083322
	v_or_b32_e32 v25, v34, v27                                 // 000000001f70: 38323722
	v_or_b32_e32 v27, v34, v29                                 // 000000001f74: 38363b22
	v_add3_u32 v21, v47, v21, v44                              // 000000001f78: d6550015 04b22b2f
	v_mov_b32_e32 v29, s7                                      // 000000001f80: 7e3a0207
	v_cmp_gt_i64_e64 s5, s[28:29], v[4:5]                      // 000000001f84: d4540005 0202081c
	v_cmp_gt_i64_e64 s6, s[28:29], v[25:26]                    // 000000001f8c: d4540006 0202321c
	v_mov_b32_e32 v128, 0                                      // 000000001f94: 7f000280
	v_mov_b32_e32 v126, 0                                      // 000000001f98: 7efc0280
	v_add3_u32 v19, v46, v19, v45                              // 000000001f9c: d6550013 04b6272e
	v_cndmask_b32_e64 v122, 0, v1, s4                          // 000000001fa4: d501007a 00120280
	s_wait_alu depctr_va_sdst(0)                               // 000000001fac: bf88f19f
	v_cndmask_b32_e64 v22, 0, v4, s5                           // 000000001fb0: d5010016 00160880
	v_cndmask_b32_e64 v24, 0, v5, s5                           // 000000001fb8: d5010018 00160a80
	v_cmp_gt_i64_e64 s5, s[28:29], v[27:28]                    // 000000001fc0: d4540005 0202361c
	v_cndmask_b32_e64 v26, 0, v26, s6                          // 000000001fc8: d501001a 001a3480
	v_cndmask_b32_e64 v25, 0, v25, s6                          // 000000001fd0: d5010019 001a3280
	v_mul_lo_u32 v43, s10, v22                                 // 000000001fd8: d72c002b 02022c0a
	v_mul_lo_u32 v42, s36, v24                                 // 000000001fe0: d72c002a 02023024
	v_mad_co_u64_u32 v[22:23], null, s36, v22, s[8:9]          // 000000001fe8: d6fe7c16 00222c24
	v_mul_lo_u32 v44, s36, v26                                 // 000000001ff0: d72c002c 02023424
	s_wait_alu depctr_va_sdst(0)                               // 000000001ff8: bf88f19f
	v_cndmask_b32_e64 v26, 0, v27, s5                          // 000000001ffc: d501001a 00163680
	v_cndmask_b32_e64 v27, 0, v28, s5                          // 000000002004: d501001b 00163880
	v_or_b32_e32 v28, v34, v30                                 // 00000000200c: 38383d22
	v_or_b32_e32 v30, v34, v32                                 // 000000002010: 383c4122
	v_mul_lo_u32 v45, s10, v25                                 // 000000002014: d72c002d 0202320a
	v_mad_co_u64_u32 v[24:25], null, s36, v25, s[8:9]          // 00000000201c: d6fe7c18 00223224
	v_mul_lo_u32 v46, s36, v27                                 // 000000002024: d72c002e 02023624
	v_cmp_gt_i64_e64 s5, s[28:29], v[28:29]                    // 00000000202c: d4540005 0202381c
	v_cmp_gt_i64_e64 s6, s[28:29], v[30:31]                    // 000000002034: d4540006 02023c1c
	v_mul_lo_u32 v47, s10, v26                                 // 00000000203c: d72c002f 0202340a
	v_mad_co_u64_u32 v[26:27], null, s36, v26, s[8:9]          // 000000002044: d6fe7c1a 00223424
	v_cndmask_b32_e64 v123, 0, v0, s4                          // 00000000204c: d501007b 00120080
	v_add3_u32 v23, v43, v23, v42                              // 000000002054: d6550017 04aa2f2b
	s_wait_alu depctr_va_sdst(0)                               // 00000000205c: bf88f19f
	v_cndmask_b32_e64 v28, 0, v28, s5                          // 000000002060: d501001c 00163880
	v_cndmask_b32_e64 v37, 0, v30, s6                          // 000000002068: d5010025 001a3c80
	v_or_b32_e32 v30, v34, v33                                 // 000000002070: 383c4322
	v_cndmask_b32_e64 v32, 0, v31, s6                          // 000000002074: d5010020 001a3e80
	v_cndmask_b32_e64 v29, 0, v29, s5                          // 00000000207c: d501001d 00163a80
	v_mov_b32_e32 v33, s7                                      // 000000002084: 7e420207
	v_mul_lo_u32 v52, s10, v37                                 // 000000002088: d72c0034 02024a0a
	v_cmp_gt_i64_e64 s5, s[28:29], v[30:31]                    // 000000002090: d4540005 02023c1c
	v_mul_lo_u32 v50, s36, v32                                 // 000000002098: d72c0032 02024024
	v_or_b32_e32 v32, v34, v35                                 // 0000000020a0: 38404722
	v_mov_b32_e32 v35, s7                                      // 0000000020a4: 7e460207
	v_or_b32_e32 v34, v34, v36                                 // 0000000020a8: 38444922
	v_mul_lo_u32 v48, s36, v29                                 // 0000000020ac: d72c0030 02023a24
	s_wait_alu depctr_va_sdst(0)                               // 0000000020b4: bf88f19f
	v_cndmask_b32_e64 v51, 0, v30, s5                          // 0000000020b8: d5010033 00163c80
	v_cndmask_b32_e64 v36, 0, v31, s5                          // 0000000020c0: d5010024 00163e80
	v_cmp_gt_i64_e64 s5, s[28:29], v[32:33]                    // 0000000020c8: d4540005 0202401c
	v_cmp_gt_i64_e64 s6, s[28:29], v[34:35]                    // 0000000020d0: d4540006 0202441c
	v_mad_co_u64_u32 v[30:31], null, s36, v37, s[8:9]          // 0000000020d8: d6fe7c1e 00224a24
	v_mul_lo_u32 v49, s10, v28                                 // 0000000020e0: d72c0031 0202380a
	v_mul_lo_u32 v53, s36, v36                                 // 0000000020e8: d72c0035 02024824
	v_mad_co_u64_u32 v[28:29], null, s36, v28, s[8:9]          // 0000000020f0: d6fe7c1c 00223824
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f8: bf88f19f
	v_cndmask_b32_e64 v36, 0, v32, s5                          // 0000000020fc: d5010024 00164080
	v_cndmask_b32_e64 v37, 0, v33, s5                          // 000000002104: d5010025 00164280
	v_cndmask_b32_e64 v55, 0, v34, s6                          // 00000000210c: d5010037 001a4480
	v_cndmask_b32_e64 v56, 0, v35, s6                          // 000000002114: d5010038 001a4680
	v_mul_lo_u32 v54, s10, v51                                 // 00000000211c: d72c0036 0202660a
	v_mad_co_u64_u32 v[32:33], null, s36, v51, s[8:9]          // 000000002124: d6fe7c20 00226624
	v_mul_lo_u32 v51, s36, v37                                 // 00000000212c: d72c0033 02024a24
	v_mul_lo_u32 v57, s10, v36                                 // 000000002134: d72c0039 0202480a
	v_mad_co_u64_u32 v[34:35], null, s36, v36, s[8:9]          // 00000000213c: d6fe7c22 00224824
	v_mul_lo_u32 v56, s36, v56                                 // 000000002144: d72c0038 02027024
	v_mul_lo_u32 v60, s10, v55                                 // 00000000214c: d72c003c 02026e0a
	v_mad_co_u64_u32 v[36:37], null, s36, v55, s[8:9]          // 000000002154: d6fe7c24 00226e24
	v_add3_u32 v25, v45, v25, v44                              // 00000000215c: d6550019 04b2332d
	v_add3_u32 v27, v47, v27, v46                              // 000000002164: d655001b 04ba372f
	v_add3_u32 v29, v49, v29, v48                              // 00000000216c: d655001d 04c23b31
	v_add3_u32 v31, v52, v31, v50                              // 000000002174: d655001f 04ca3f34
	v_add3_u32 v33, v54, v33, v53                              // 00000000217c: d6550021 04d64336
	v_add3_u32 v35, v57, v35, v51                              // 000000002184: d6550023 04ce4739
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v114, 0            // 00000000218c: ca100080 45720080
	v_add3_u32 v37, v60, v37, v56                              // 000000002194: d6550025 04e24b3c
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v98, 0            // 00000000219c: ca100080 7d620080
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v96, 0            // 0000000021a4: ca100080 79600080
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v94, 0            // 0000000021ac: ca100080 775e0080
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v92, 0            // 0000000021b4: ca100080 6f5c0080
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v82, 0            // 0000000021bc: ca100080 6d520080
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v80, 0             // 0000000021c4: ca100080 63500080
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v78, 0             // 0000000021cc: ca100080 614e0080
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v76, 0             // 0000000021d4: ca100080 5f4c0080
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v66, 0             // 0000000021dc: ca100080 5d420080
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v64, 0             // 0000000021e4: ca100080 53400080
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v62, 0             // 0000000021ec: ca100080 513e0080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v60, 0             // 0000000021f4: ca100080 4f3c0080
	v_mov_b32_e32 v77, 0                                       // 0000000021fc: 7e9a0280
	v_mov_b32_e32 v67, 0                                       // 000000002200: 7e860280
	v_mov_b32_e32 v65, 0                                       // 000000002204: 7e820280
	v_mov_b32_e32 v63, 0                                       // 000000002208: 7e7e0280
	v_mov_b32_e32 v61, 0                                       // 00000000220c: 7e7a0280
	s_mov_b64 s[38:39], 0                                      // 000000002210: bea60180
	s_delay_alu instid0(salu_cycle_1)                          // 000000002214: bf870009
	s_lshl_b64 s[22:23], s[38:39], 5                           // 000000002218: 84968526
	s_mul_u64 s[24:25], s[38:39], s[30:31]                     // 00000000221c: aa981e26
	s_wait_alu depctr_sa_sdst(0)                               // 000000002220: bf88ff9e
	v_add_co_u32 v143, s21, v112, s22                          // 000000002224: d700158f 02002d70
	v_add_co_u32 v147, s22, v115, s22                          // 00000000222c: d7001693 02002d73
	s_wait_alu depctr_va_sdst(0)                               // 000000002234: bf88f19f
	v_add_co_ci_u32_e64 v144, null, s23, v113, s21             // 000000002238: d5207c90 0056e217
	v_add_co_ci_u32_e64 v148, null, s23, v116, s22             // 000000002240: d5207c94 005ae817
	v_add_co_u32 v38, s5, v6, s38                              // 000000002248: d7000526 02004d06
	global_load_b128 v[143:146], v[143:144], off               // 000000002250: ee05c07c 0000008f 0000008f
	global_load_b128 v[147:150], v[147:148], off               // 00000000225c: ee05c07c 00000093 00000093
	s_add_nc_u64 s[24:25], s[34:35], s[24:25]                  // 000000002268: a9981822
	v_add_co_ci_u32_e64 v39, null, s39, v7, s5                 // 00000000226c: d5207c27 00160e27
	s_wait_alu depctr_sa_sdst(0)                               // 000000002274: bf88ff9e
	v_add_co_u32 v163, s5, s24, v123                           // 000000002278: d70005a3 0202f618
	s_wait_alu depctr_va_sdst(0)                               // 000000002280: bf88f19f
	v_add_co_ci_u32_e64 v164, null, s25, v122, s5              // 000000002284: d5207ca4 0016f419
	v_add_co_u32 v40, s6, v8, s38                              // 00000000228c: d7000628 02004d08
	v_add_co_u32 v42, s7, v10, s38                             // 000000002294: d700072a 02004d0a
	s_wait_alu depctr_va_sdst(0)                               // 00000000229c: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s39, v9, s6                 // 0000000022a0: d5207c29 001a1227
	v_add_co_u32 v165, s6, s24, v130                           // 0000000022a8: d70006a5 02030418
	v_add_co_ci_u32_e64 v43, null, s39, v11, s7                // 0000000022b0: d5207c2b 001e1627
	s_wait_alu depctr_va_sdst(0)                               // 0000000022b8: bf88f19f
	v_add_co_ci_u32_e64 v166, null, s25, v129, s6              // 0000000022bc: d5207ca6 001b0219
	s_barrier_signal -1                                        // 0000000022c4: be804ec1
	s_barrier_wait 0xffff                                      // 0000000022c8: bf94ffff
	v_add_co_u32 v44, s8, v12, s38                             // 0000000022cc: d700082c 02004d0c
	v_add_co_u32 v46, s9, v14, s38                             // 0000000022d4: d700092e 02004d0e
	v_add_co_u32 v48, s10, v16, s38                            // 0000000022dc: d7000a30 02004d10
	v_add_co_u32 v50, s11, v18, s38                            // 0000000022e4: d7000b32 02004d12
	v_add_co_u32 v52, s12, v20, s38                            // 0000000022ec: d7000c34 02004d14
	s_wait_alu depctr_va_sdst(0)                               // 0000000022f4: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s39, v13, s8                // 0000000022f8: d5207c2d 00221a27
	v_add_co_u32 v167, s7, s24, v133                           // 000000002300: d70007a7 02030a18
	v_add_co_u32 v169, s8, s24, v136                           // 000000002308: d70008a9 02031018
	v_add_co_ci_u32_e64 v47, null, s39, v15, s9                // 000000002310: d5207c2f 00261e27
	v_add_co_ci_u32_e64 v49, null, s39, v17, s10               // 000000002318: d5207c31 002a2227
	v_add_co_ci_u32_e64 v51, null, s39, v19, s11               // 000000002320: d5207c33 002e2627
	v_add_co_ci_u32_e64 v53, null, s39, v21, s12               // 000000002328: d5207c35 00322a27
	s_wait_alu depctr_va_sdst(0)                               // 000000002330: bf88f19f
	v_add_co_ci_u32_e64 v168, null, s25, v131, s7              // 000000002334: d5207ca8 001f0619
	v_add_co_ci_u32_e64 v170, null, s25, v134, s8              // 00000000233c: d5207caa 00230c19
	v_add_co_u32 v54, s13, v22, s38                            // 000000002344: d7000d36 02004d16
	v_add_co_u32 v56, s14, v24, s38                            // 00000000234c: d7000e38 02004d18
	v_add_co_u32 v151, s15, v26, s38                           // 000000002354: d7000f97 02004d1a
	v_add_co_u32 v153, s16, v28, s38                           // 00000000235c: d7001099 02004d1c
	v_add_co_u32 v155, s17, v30, s38                           // 000000002364: d700119b 02004d1e
	v_add_co_u32 v157, s18, v32, s38                           // 00000000236c: d700129d 02004d20
	v_add_co_u32 v159, s19, v34, s38                           // 000000002374: d700139f 02004d22
	v_add_co_u32 v161, s20, v36, s38                           // 00000000237c: d70014a1 02004d24
	s_wait_alu depctr_va_sdst(0)                               // 000000002384: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s39, v23, s13               // 000000002388: d5207c37 00362e27
	v_add_co_ci_u32_e64 v57, null, s39, v25, s14               // 000000002390: d5207c39 003a3227
	v_add_co_ci_u32_e64 v152, null, s39, v27, s15              // 000000002398: d5207c98 003e3627
	v_add_co_ci_u32_e64 v154, null, s39, v29, s16              // 0000000023a0: d5207c9a 00423a27
	v_add_co_ci_u32_e64 v156, null, s39, v31, s17              // 0000000023a8: d5207c9c 00463e27
	v_add_co_ci_u32_e64 v158, null, s39, v33, s18              // 0000000023b0: d5207c9e 004a4227
	v_add_co_ci_u32_e64 v160, null, s39, v35, s19              // 0000000023b8: d5207ca0 004e4627
	v_add_co_ci_u32_e64 v162, null, s39, v37, s20              // 0000000023c0: d5207ca2 00524a27
	s_add_nc_u64 s[38:39], s[38:39], 1                         // 0000000023c8: a9a68126
	s_wait_loadcnt 0x1                                         // 0000000023cc: bfc00001
	ds_store_b128 v110, v[143:146]                             // 0000000023d0: db7c0000 00008f6e
	s_wait_loadcnt 0x0                                         // 0000000023d8: bfc00000
	ds_store_b128 v110, v[147:150] offset:6144                 // 0000000023dc: db7c1800 0000936e
	s_wait_dscnt 0x0                                           // 0000000023e4: bfc60000
	s_barrier_signal -1                                        // 0000000023e8: be804ec1
	s_barrier_wait 0xffff                                      // 0000000023ec: bf94ffff
	s_clause 0x1                                               // 0000000023f0: bf850001
	global_load_u8 v201, v[163:164], off                       // 0000000023f4: ee04007c 000000c9 000000a3
	global_load_u8 v202, v[165:166], off                       // 000000002400: ee04007c 000000ca 000000a5
	s_clause 0x2                                               // 00000000240c: bf850002
	global_load_u8 v205, v[38:39], off                         // 000000002410: ee04007c 000000cd 00000026
	global_load_u8 v206, v[40:41], off                         // 00000000241c: ee04007c 000000ce 00000028
	global_load_u8 v207, v[42:43], off                         // 000000002428: ee04007c 000000cf 0000002a
	s_clause 0x1                                               // 000000002434: bf850001
	global_load_u8 v203, v[167:168], off                       // 000000002438: ee04007c 000000cb 000000a7
	global_load_u8 v204, v[169:170], off                       // 000000002444: ee04007c 000000cc 000000a9
	s_clause 0xc                                               // 000000002450: bf85000c
	global_load_u8 v208, v[44:45], off                         // 000000002454: ee04007c 000000d0 0000002c
	global_load_u8 v209, v[46:47], off                         // 000000002460: ee04007c 000000d1 0000002e
	global_load_u8 v210, v[48:49], off                         // 00000000246c: ee04007c 000000d2 00000030
	global_load_u8 v211, v[50:51], off                         // 000000002478: ee04007c 000000d3 00000032
	global_load_u8 v212, v[52:53], off                         // 000000002484: ee04007c 000000d4 00000034
	global_load_u8 v213, v[54:55], off                         // 000000002490: ee04007c 000000d5 00000036
	global_load_u8 v214, v[56:57], off                         // 00000000249c: ee04007c 000000d6 00000038
	global_load_u8 v215, v[151:152], off                       // 0000000024a8: ee04007c 000000d7 00000097
	global_load_u8 v216, v[153:154], off                       // 0000000024b4: ee04007c 000000d8 00000099
	global_load_u8 v217, v[155:156], off                       // 0000000024c0: ee04007c 000000d9 0000009b
	global_load_u8 v218, v[157:158], off                       // 0000000024cc: ee04007c 000000da 0000009d
	global_load_u8 v219, v[159:160], off                       // 0000000024d8: ee04007c 000000db 0000009f
	global_load_u8 v220, v[161:162], off                       // 0000000024e4: ee04007c 000000dc 000000a1
	ds_load_2addr_b64 v[54:57], v117 offset1:2                 // 0000000024f0: d9dc0200 36000075
	ds_load_2addr_b64 v[181:184], v139 offset1:2               // 0000000024f8: d9dc0200 b500008b
	ds_load_2addr_b64 v[185:188], v140 offset1:2               // 000000002500: d9dc0200 b900008c
	ds_load_2addr_b64 v[189:192], v141 offset1:2               // 000000002508: d9dc0200 bd00008d
	ds_load_2addr_b64 v[193:196], v142 offset1:2               // 000000002510: d9dc0200 c100008e
	ds_load_2addr_b64 v[197:200], v118 offset1:2               // 000000002518: d9dc0200 c5000076
	s_wait_dscnt 0x4                                           // 000000002520: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[54:55], v[181:182], 0// 000000002524: cc464026 1a036b36
	s_wait_dscnt 0x3                                           // 00000000252c: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[54:55], v[185:186], 0// 000000002530: cc46402e 1a037336
	s_wait_dscnt 0x2                                           // 000000002538: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[143:150], v[54:55], v[189:190], 0// 00000000253c: cc46408f 1a037b36
	s_wait_dscnt 0x1                                           // 000000002544: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[151:158], v[54:55], v[193:194], 0// 000000002548: cc464097 1a038336
	s_wait_dscnt 0x0                                           // 000000002550: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[159:166], v[197:198], v[181:182], 0// 000000002554: cc46409f 1a036bc5
	v_wmma_f32_16x16x16_fp8_fp8 v[167:174], v[197:198], v[185:186], 0// 00000000255c: cc4640a7 1a0373c5
	v_wmma_f32_16x16x16_fp8_fp8 v[175:182], v[197:198], v[189:190], 0// 000000002564: cc4640af 1a037bc5
	v_wmma_f32_16x16x16_fp8_fp8 v[38:45], v[56:57], v[183:184], v[38:45]// 00000000256c: cc464026 1c9b6f38
	v_wmma_f32_16x16x16_fp8_fp8 v[46:53], v[56:57], v[187:188], v[46:53]// 000000002574: cc46402e 1cbb7738
	v_wmma_f32_16x16x16_fp8_fp8 v[143:150], v[56:57], v[191:192], v[143:150]// 00000000257c: cc46408f 1e3f7f38
	v_wmma_f32_16x16x16_fp8_fp8 v[159:166], v[199:200], v[183:184], v[159:166]// 000000002584: cc46409f 1e7f6fc7
	v_wmma_f32_16x16x16_fp8_fp8 v[167:174], v[199:200], v[187:188], v[167:174]// 00000000258c: cc4640a7 1e9f77c7
	v_wmma_f32_16x16x16_fp8_fp8 v[183:190], v[197:198], v[193:194], 0// 000000002594: cc4640b7 1a0383c5
	v_wmma_f32_16x16x16_fp8_fp8 v[175:182], v[199:200], v[191:192], v[175:182]// 00000000259c: cc4640af 1ebf7fc7
	v_wmma_f32_16x16x16_fp8_fp8 v[151:158], v[56:57], v[195:196], v[151:158]// 0000000025a4: cc464097 1e5f8738
	s_delay_alu instid0(valu_dep_3)                            // 0000000025ac: bf870003
	v_wmma_f32_16x16x16_fp8_fp8 v[183:190], v[199:200], v[195:196], v[183:190]// 0000000025b0: cc4640b7 1edf87c7
	s_wait_loadcnt 0x13                                        // 0000000025b8: bfc00013
	v_add_nc_u32_e32 v54, 0xffffff02, v201                     // 0000000025bc: 4a6d92ff ffffff02
	s_wait_loadcnt 0x12                                        // 0000000025c4: bfc00012
	v_add_nc_u32_e32 v55, 0xffffff02, v202                     // 0000000025c8: 4a6f94ff ffffff02
	v_cmp_eq_u32_e64 s6, 0xff, v201                            // 0000000025d0: d44a0006 020392ff 000000ff
	v_cmp_eq_u32_e64 s14, 0xff, v202                           // 0000000025dc: d44a000e 020394ff 000000ff
	s_wait_loadcnt 0x11                                        // 0000000025e8: bfc00011
	v_cmp_eq_u32_e64 s5, 0xff, v205                            // 0000000025ec: d44a0005 02039aff 000000ff
	v_add_nc_u32_e32 v191, v54, v205                           // 0000000025f8: 4b7f9b36
	s_wait_loadcnt 0x10                                        // 0000000025fc: bfc00010
	v_add_nc_u32_e32 v192, v54, v206                           // 000000002600: 4b819d36
	s_wait_loadcnt 0xf                                         // 000000002604: bfc0000f
	v_add_nc_u32_e32 v193, v54, v207                           // 000000002608: 4b839f36
	s_wait_loadcnt 0xe                                         // 00000000260c: bfc0000e
	v_add_nc_u32_e32 v56, 0xffffff02, v203                     // 000000002610: 4a7196ff ffffff02
	s_wait_loadcnt 0xd                                         // 000000002618: bfc0000d
	v_add_nc_u32_e32 v57, 0xffffff02, v204                     // 00000000261c: 4a7398ff ffffff02
	v_cmp_eq_u32_e64 s15, 0xff, v203                           // 000000002624: d44a000f 020396ff 000000ff
	s_wait_loadcnt 0xc                                         // 000000002630: bfc0000c
	v_add_nc_u32_e32 v194, v54, v208                           // 000000002634: 4b85a136
	s_wait_loadcnt 0xb                                         // 000000002638: bfc0000b
	v_add_nc_u32_e32 v195, v54, v209                           // 00000000263c: 4b87a336
	s_wait_loadcnt 0xa                                         // 000000002640: bfc0000a
	v_add_nc_u32_e32 v196, v54, v210                           // 000000002644: 4b89a536
	s_wait_loadcnt 0x9                                         // 000000002648: bfc00009
	v_add_nc_u32_e32 v197, v54, v211                           // 00000000264c: 4b8ba736
	s_wait_loadcnt 0x8                                         // 000000002650: bfc00008
	v_add_nc_u32_e32 v198, v54, v212                           // 000000002654: 4b8da936
	v_add_nc_u32_e32 v199, v55, v205                           // 000000002658: 4b8f9b37
	v_add_nc_u32_e32 v200, v55, v206                           // 00000000265c: 4b919d37
	v_add_nc_u32_e32 v201, v55, v207                           // 000000002660: 4b939f37
	v_add_nc_u32_e32 v202, v55, v208                           // 000000002664: 4b95a137
	v_add_nc_u32_e32 v203, v55, v209                           // 000000002668: 4b97a337
	v_ldexp_f32 v38, v38, v191                                 // 00000000266c: d71c0026 02037f26
	v_ldexp_f32 v39, v39, v192                                 // 000000002674: d71c0027 02038127
	v_ldexp_f32 v40, v40, v193                                 // 00000000267c: d71c0028 02038328
	v_add_nc_u32_e32 v191, v55, v210                           // 000000002684: 4b7fa537
	v_add_nc_u32_e32 v192, v55, v211                           // 000000002688: 4b81a737
	v_add_nc_u32_e32 v193, v55, v212                           // 00000000268c: 4b83a937
	v_cmp_eq_u32_e64 s7, 0xff, v206                            // 000000002690: d44a0007 02039cff 000000ff
	v_cmp_eq_u32_e64 s16, 0xff, v204                           // 00000000269c: d44a0010 020398ff 000000ff
	v_ldexp_f32 v41, v41, v194                                 // 0000000026a8: d71c0029 02038529
	v_ldexp_f32 v42, v42, v195                                 // 0000000026b0: d71c002a 0203872a
	v_ldexp_f32 v43, v43, v196                                 // 0000000026b8: d71c002b 0203892b
	v_ldexp_f32 v44, v44, v197                                 // 0000000026c0: d71c002c 02038b2c
	v_ldexp_f32 v45, v45, v198                                 // 0000000026c8: d71c002d 02038d2d
	v_add_nc_u32_e32 v194, v56, v205                           // 0000000026d0: 4b859b38
	v_add_nc_u32_e32 v195, v56, v206                           // 0000000026d4: 4b879d38
	v_add_nc_u32_e32 v196, v56, v207                           // 0000000026d8: 4b899f38
	v_add_nc_u32_e32 v197, v56, v208                           // 0000000026dc: 4b8ba138
	v_add_nc_u32_e32 v198, v56, v209                           // 0000000026e0: 4b8da338
	v_ldexp_f32 v46, v46, v199                                 // 0000000026e4: d71c002e 02038f2e
	v_ldexp_f32 v47, v47, v200                                 // 0000000026ec: d71c002f 0203912f
	v_ldexp_f32 v48, v48, v201                                 // 0000000026f4: d71c0030 02039330
	v_ldexp_f32 v49, v49, v202                                 // 0000000026fc: d71c0031 02039531
	v_ldexp_f32 v50, v50, v203                                 // 000000002704: d71c0032 02039732
	v_ldexp_f32 v51, v51, v191                                 // 00000000270c: d71c0033 02037f33
	v_ldexp_f32 v52, v52, v192                                 // 000000002714: d71c0034 02038134
	v_ldexp_f32 v53, v53, v193                                 // 00000000271c: d71c0035 02038335
	v_add_nc_u32_e32 v191, v56, v210                           // 000000002724: 4b7fa538
	v_add_nc_u32_e32 v192, v56, v211                           // 000000002728: 4b81a738
	v_add_nc_u32_e32 v193, v56, v212                           // 00000000272c: 4b83a938
	v_add_nc_u32_e32 v199, v57, v205                           // 000000002730: 4b8f9b39
	v_add_nc_u32_e32 v200, v57, v206                           // 000000002734: 4b919d39
	v_add_nc_u32_e32 v201, v57, v207                           // 000000002738: 4b939f39
	v_add_nc_u32_e32 v202, v57, v208                           // 00000000273c: 4b95a139
	v_add_nc_u32_e32 v203, v57, v209                           // 000000002740: 4b97a339
	v_add_nc_u32_e32 v204, v57, v210                           // 000000002744: 4b99a539
	v_add_nc_u32_e32 v205, v57, v211                           // 000000002748: 4b9ba739
	v_add_nc_u32_e32 v206, v57, v212                           // 00000000274c: 4b9da939
	v_cmp_eq_u32_e64 s8, 0xff, v207                            // 000000002750: d44a0008 02039eff 000000ff
	v_cmp_eq_u32_e64 s9, 0xff, v208                            // 00000000275c: d44a0009 0203a0ff 000000ff
	v_cmp_eq_u32_e64 s10, 0xff, v209                           // 000000002768: d44a000a 0203a2ff 000000ff
	v_cmp_eq_u32_e64 s11, 0xff, v210                           // 000000002774: d44a000b 0203a4ff 000000ff
	v_cmp_eq_u32_e64 s12, 0xff, v211                           // 000000002780: d44a000c 0203a6ff 000000ff
	v_cmp_eq_u32_e64 s13, 0xff, v212                           // 00000000278c: d44a000d 0203a8ff 000000ff
	s_wait_loadcnt 0x7                                         // 000000002798: bfc00007
	v_cmp_eq_u32_e64 s17, 0xff, v213                           // 00000000279c: d44a0011 0203aaff 000000ff
	s_wait_loadcnt 0x6                                         // 0000000027a8: bfc00006
	v_cmp_eq_u32_e64 s18, 0xff, v214                           // 0000000027ac: d44a0012 0203acff 000000ff
	s_wait_loadcnt 0x5                                         // 0000000027b8: bfc00005
	v_cmp_eq_u32_e64 s19, 0xff, v215                           // 0000000027bc: d44a0013 0203aeff 000000ff
	s_wait_loadcnt 0x4                                         // 0000000027c8: bfc00004
	v_cmp_eq_u32_e64 s20, 0xff, v216                           // 0000000027cc: d44a0014 0203b0ff 000000ff
	s_wait_loadcnt 0x3                                         // 0000000027d8: bfc00003
	v_cmp_eq_u32_e64 s21, 0xff, v217                           // 0000000027dc: d44a0015 0203b2ff 000000ff
	s_wait_loadcnt 0x2                                         // 0000000027e8: bfc00002
	v_cmp_eq_u32_e64 s22, 0xff, v218                           // 0000000027ec: d44a0016 0203b4ff 000000ff
	v_add_nc_u32_e32 v207, v54, v213                           // 0000000027f8: 4b9fab36
	v_add_nc_u32_e32 v208, v54, v214                           // 0000000027fc: 4ba1ad36
	v_add_nc_u32_e32 v209, v54, v215                           // 000000002800: 4ba3af36
	v_add_nc_u32_e32 v210, v54, v216                           // 000000002804: 4ba5b136
	v_add_nc_u32_e32 v211, v54, v217                           // 000000002808: 4ba7b336
	v_ldexp_f32 v143, v143, v194                               // 00000000280c: d71c008f 0203858f
	v_ldexp_f32 v144, v144, v195                               // 000000002814: d71c0090 02038790
	v_ldexp_f32 v145, v145, v196                               // 00000000281c: d71c0091 02038991
	v_ldexp_f32 v146, v146, v197                               // 000000002824: d71c0092 02038b92
	v_ldexp_f32 v147, v147, v198                               // 00000000282c: d71c0093 02038d93
	v_ldexp_f32 v148, v148, v191                               // 000000002834: d71c0094 02037f94
	v_ldexp_f32 v149, v149, v192                               // 00000000283c: d71c0095 02038195
	v_ldexp_f32 v150, v150, v193                               // 000000002844: d71c0096 02038396
	v_add_nc_u32_e32 v191, v54, v218                           // 00000000284c: 4b7fb536
	s_wait_loadcnt 0x1                                         // 000000002850: bfc00001
	v_add_nc_u32_e32 v192, v54, v219                           // 000000002854: 4b81b736
	s_wait_loadcnt 0x0                                         // 000000002858: bfc00000
	v_add_nc_u32_e32 v54, v54, v220                            // 00000000285c: 4a6db936
	v_add_nc_u32_e32 v193, v55, v213                           // 000000002860: 4b83ab37
	v_add_nc_u32_e32 v194, v55, v214                           // 000000002864: 4b85ad37
	v_add_nc_u32_e32 v195, v55, v215                           // 000000002868: 4b87af37
	v_add_nc_u32_e32 v196, v55, v216                           // 00000000286c: 4b89b137
	v_add_nc_u32_e32 v197, v55, v217                           // 000000002870: 4b8bb337
	v_add_nc_u32_e32 v198, v55, v218                           // 000000002874: 4b8db537
	v_ldexp_f32 v151, v151, v199                               // 000000002878: d71c0097 02038f97
	v_ldexp_f32 v152, v152, v200                               // 000000002880: d71c0098 02039198
	v_ldexp_f32 v153, v153, v201                               // 000000002888: d71c0099 02039399
	v_ldexp_f32 v154, v154, v202                               // 000000002890: d71c009a 0203959a
	v_ldexp_f32 v155, v155, v203                               // 000000002898: d71c009b 0203979b
	v_ldexp_f32 v156, v156, v204                               // 0000000028a0: d71c009c 0203999c
	v_ldexp_f32 v157, v157, v205                               // 0000000028a8: d71c009d 02039b9d
	v_ldexp_f32 v158, v158, v206                               // 0000000028b0: d71c009e 02039d9e
	v_add_nc_u32_e32 v199, v55, v219                           // 0000000028b8: 4b8fb737
	v_add_nc_u32_e32 v55, v55, v220                            // 0000000028bc: 4a6fb937
	v_add_nc_u32_e32 v200, v56, v213                           // 0000000028c0: 4b91ab38
	v_add_nc_u32_e32 v201, v56, v214                           // 0000000028c4: 4b93ad38
	v_add_nc_u32_e32 v202, v56, v215                           // 0000000028c8: 4b95af38
	v_add_nc_u32_e32 v203, v56, v216                           // 0000000028cc: 4b97b138
	v_add_nc_u32_e32 v204, v56, v217                           // 0000000028d0: 4b99b338
	v_add_nc_u32_e32 v205, v56, v218                           // 0000000028d4: 4b9bb538
	v_add_nc_u32_e32 v206, v56, v219                           // 0000000028d8: 4b9db738
	v_add_nc_u32_e32 v56, v56, v220                            // 0000000028dc: 4a71b938
	v_add_nc_u32_e32 v212, v57, v213                           // 0000000028e0: 4ba9ab39
	v_add_nc_u32_e32 v213, v57, v214                           // 0000000028e4: 4babad39
	v_add_nc_u32_e32 v214, v57, v215                           // 0000000028e8: 4badaf39
	v_add_nc_u32_e32 v215, v57, v216                           // 0000000028ec: 4bafb139
	v_add_nc_u32_e32 v216, v57, v217                           // 0000000028f0: 4bb1b339
	v_add_nc_u32_e32 v217, v57, v218                           // 0000000028f4: 4bb3b539
	v_add_nc_u32_e32 v218, v57, v219                           // 0000000028f8: 4bb5b739
	v_add_nc_u32_e32 v57, v57, v220                            // 0000000028fc: 4a73b939
	v_cmp_eq_u32_e64 s23, 0xff, v219                           // 000000002900: d44a0017 0203b6ff 000000ff
	v_cmp_eq_u32_e64 s24, 0xff, v220                           // 00000000290c: d44a0018 0203b8ff 000000ff
	v_ldexp_f32 v159, v159, v207                               // 000000002918: d71c009f 02039f9f
	v_ldexp_f32 v160, v160, v208                               // 000000002920: d71c00a0 0203a1a0
	v_ldexp_f32 v161, v161, v209                               // 000000002928: d71c00a1 0203a3a1
	v_ldexp_f32 v162, v162, v210                               // 000000002930: d71c00a2 0203a5a2
	v_ldexp_f32 v163, v163, v211                               // 000000002938: d71c00a3 0203a7a3
	v_ldexp_f32 v164, v164, v191                               // 000000002940: d71c00a4 02037fa4
	v_ldexp_f32 v165, v165, v192                               // 000000002948: d71c00a5 020381a5
	v_ldexp_f32 v54, v166, v54                                 // 000000002950: d71c0036 02026da6
	v_ldexp_f32 v166, v167, v193                               // 000000002958: d71c00a6 020383a7
	v_ldexp_f32 v167, v168, v194                               // 000000002960: d71c00a7 020385a8
	v_ldexp_f32 v168, v169, v195                               // 000000002968: d71c00a8 020387a9
	v_ldexp_f32 v169, v170, v196                               // 000000002970: d71c00a9 020389aa
	v_ldexp_f32 v170, v171, v197                               // 000000002978: d71c00aa 02038bab
	v_ldexp_f32 v171, v172, v198                               // 000000002980: d71c00ab 02038dac
	v_ldexp_f32 v172, v173, v199                               // 000000002988: d71c00ac 02038fad
	v_ldexp_f32 v55, v174, v55                                 // 000000002990: d71c0037 02026fae
	v_ldexp_f32 v173, v175, v200                               // 000000002998: d71c00ad 020391af
	v_ldexp_f32 v174, v176, v201                               // 0000000029a0: d71c00ae 020393b0
	v_ldexp_f32 v175, v177, v202                               // 0000000029a8: d71c00af 020395b1
	v_ldexp_f32 v176, v178, v203                               // 0000000029b0: d71c00b0 020397b2
	v_ldexp_f32 v177, v179, v204                               // 0000000029b8: d71c00b1 020399b3
	v_ldexp_f32 v178, v180, v205                               // 0000000029c0: d71c00b2 02039bb4
	v_ldexp_f32 v179, v181, v206                               // 0000000029c8: d71c00b3 02039db5
	v_ldexp_f32 v56, v182, v56                                 // 0000000029d0: d71c0038 020271b6
	v_ldexp_f32 v180, v183, v212                               // 0000000029d8: d71c00b4 0203a9b7
	v_ldexp_f32 v181, v184, v213                               // 0000000029e0: d71c00b5 0203abb8
	v_ldexp_f32 v182, v185, v214                               // 0000000029e8: d71c00b6 0203adb9
	v_ldexp_f32 v183, v186, v215                               // 0000000029f0: d71c00b7 0203afba
	v_ldexp_f32 v184, v187, v216                               // 0000000029f8: d71c00b8 0203b1bb
	v_ldexp_f32 v185, v188, v217                               // 000000002a00: d71c00b9 0203b3bc
	v_ldexp_f32 v186, v189, v218                               // 000000002a08: d71c00ba 0203b5bd
	v_ldexp_f32 v57, v190, v57                                 // 000000002a10: d71c0039 020273be
	s_or_b32 s25, s5, s6                                       // 000000002a18: 8c190605
	s_or_b32 s33, s6, s7                                       // 000000002a1c: 8c210706
	s_or_b32 s40, s6, s8                                       // 000000002a20: 8c280806
	s_or_b32 s41, s6, s9                                       // 000000002a24: 8c290906
	s_or_b32 s42, s6, s10                                      // 000000002a28: 8c2a0a06
	s_or_b32 s43, s6, s11                                      // 000000002a2c: 8c2b0b06
	s_or_b32 s44, s6, s12                                      // 000000002a30: 8c2c0c06
	s_or_b32 s45, s6, s13                                      // 000000002a34: 8c2d0d06
	s_or_b32 s46, s5, s14                                      // 000000002a38: 8c2e0e05
	s_or_b32 s47, s7, s14                                      // 000000002a3c: 8c2f0e07
	s_or_b32 s48, s8, s14                                      // 000000002a40: 8c300e08
	s_or_b32 s49, s9, s14                                      // 000000002a44: 8c310e09
	s_or_b32 s50, s10, s14                                     // 000000002a48: 8c320e0a
	s_or_b32 s51, s11, s14                                     // 000000002a4c: 8c330e0b
	s_or_b32 s52, s12, s14                                     // 000000002a50: 8c340e0c
	s_or_b32 s53, s13, s14                                     // 000000002a54: 8c350e0d
	s_or_b32 s54, s5, s15                                      // 000000002a58: 8c360f05
	s_or_b32 s55, s7, s15                                      // 000000002a5c: 8c370f07
	s_or_b32 s56, s8, s15                                      // 000000002a60: 8c380f08
	s_or_b32 s57, s9, s15                                      // 000000002a64: 8c390f09
	s_or_b32 s58, s10, s15                                     // 000000002a68: 8c3a0f0a
	s_or_b32 s59, s11, s15                                     // 000000002a6c: 8c3b0f0b
	s_or_b32 s60, s12, s15                                     // 000000002a70: 8c3c0f0c
	s_or_b32 s61, s13, s15                                     // 000000002a74: 8c3d0f0d
	s_or_b32 s5, s5, s16                                       // 000000002a78: 8c051005
	s_or_b32 s7, s7, s16                                       // 000000002a7c: 8c071007
	s_or_b32 s8, s8, s16                                       // 000000002a80: 8c081008
	s_or_b32 s9, s9, s16                                       // 000000002a84: 8c091009
	s_or_b32 s10, s10, s16                                     // 000000002a88: 8c0a100a
	s_or_b32 s11, s11, s16                                     // 000000002a8c: 8c0b100b
	s_or_b32 s12, s12, s16                                     // 000000002a90: 8c0c100c
	s_or_b32 s13, s13, s16                                     // 000000002a94: 8c0d100d
	s_or_b32 s62, s6, s17                                      // 000000002a98: 8c3e1106
	s_or_b32 s63, s6, s18                                      // 000000002a9c: 8c3f1206
	s_or_b32 s64, s6, s19                                      // 000000002aa0: 8c401306
	s_or_b32 s65, s6, s20                                      // 000000002aa4: 8c411406
	s_or_b32 s66, s6, s21                                      // 000000002aa8: 8c421506
	s_or_b32 s67, s6, s22                                      // 000000002aac: 8c431606
	s_or_b32 s68, s6, s23                                      // 000000002ab0: 8c441706
	s_or_b32 s6, s6, s24                                       // 000000002ab4: 8c061806
	s_or_b32 s69, s14, s17                                     // 000000002ab8: 8c45110e
	s_or_b32 s70, s14, s18                                     // 000000002abc: 8c46120e
	s_or_b32 s71, s14, s19                                     // 000000002ac0: 8c47130e
	s_or_b32 s72, s14, s20                                     // 000000002ac4: 8c48140e
	s_or_b32 s73, s14, s21                                     // 000000002ac8: 8c49150e
	s_or_b32 s74, s14, s22                                     // 000000002acc: 8c4a160e
	s_or_b32 s75, s14, s23                                     // 000000002ad0: 8c4b170e
	s_or_b32 s14, s14, s24                                     // 000000002ad4: 8c0e180e
	s_or_b32 s76, s15, s17                                     // 000000002ad8: 8c4c110f
	s_or_b32 s77, s15, s18                                     // 000000002adc: 8c4d120f
	s_or_b32 s78, s15, s19                                     // 000000002ae0: 8c4e130f
	s_or_b32 s79, s15, s20                                     // 000000002ae4: 8c4f140f
	s_or_b32 s80, s15, s21                                     // 000000002ae8: 8c50150f
	s_or_b32 s81, s15, s22                                     // 000000002aec: 8c51160f
	s_or_b32 s82, s15, s23                                     // 000000002af0: 8c52170f
	s_or_b32 s15, s15, s24                                     // 000000002af4: 8c0f180f
	s_or_b32 s17, s16, s17                                     // 000000002af8: 8c111110
	s_or_b32 s18, s16, s18                                     // 000000002afc: 8c121210
	s_or_b32 s19, s16, s19                                     // 000000002b00: 8c131310
	s_or_b32 s20, s16, s20                                     // 000000002b04: 8c141410
	s_or_b32 s21, s16, s21                                     // 000000002b08: 8c151510
	s_or_b32 s22, s16, s22                                     // 000000002b0c: 8c161610
	s_or_b32 s23, s16, s23                                     // 000000002b10: 8c171710
	s_or_b32 s16, s16, s24                                     // 000000002b14: 8c101810
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b18: bf88ff9e
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s25                // 000000002b1c: d5010026 0065ff26 7fc00000
	v_cndmask_b32_e64 v39, v39, 0x7fc00000, s33                // 000000002b28: d5010027 0085ff27 7fc00000
	v_cndmask_b32_e64 v40, v40, 0x7fc00000, s40                // 000000002b34: d5010028 00a1ff28 7fc00000
	v_cndmask_b32_e64 v41, v41, 0x7fc00000, s41                // 000000002b40: d5010029 00a5ff29 7fc00000
	v_cndmask_b32_e64 v42, v42, 0x7fc00000, s42                // 000000002b4c: d501002a 00a9ff2a 7fc00000
	v_cndmask_b32_e64 v43, v43, 0x7fc00000, s43                // 000000002b58: d501002b 00adff2b 7fc00000
	v_cndmask_b32_e64 v44, v44, 0x7fc00000, s44                // 000000002b64: d501002c 00b1ff2c 7fc00000
	v_cndmask_b32_e64 v45, v45, 0x7fc00000, s45                // 000000002b70: d501002d 00b5ff2d 7fc00000
	v_cndmask_b32_e64 v46, v46, 0x7fc00000, s46                // 000000002b7c: d501002e 00b9ff2e 7fc00000
	v_cndmask_b32_e64 v47, v47, 0x7fc00000, s47                // 000000002b88: d501002f 00bdff2f 7fc00000
	v_cndmask_b32_e64 v48, v48, 0x7fc00000, s48                // 000000002b94: d5010030 00c1ff30 7fc00000
	v_cndmask_b32_e64 v49, v49, 0x7fc00000, s49                // 000000002ba0: d5010031 00c5ff31 7fc00000
	v_cndmask_b32_e64 v50, v50, 0x7fc00000, s50                // 000000002bac: d5010032 00c9ff32 7fc00000
	v_cndmask_b32_e64 v51, v51, 0x7fc00000, s51                // 000000002bb8: d5010033 00cdff33 7fc00000
	v_cndmask_b32_e64 v52, v52, 0x7fc00000, s52                // 000000002bc4: d5010034 00d1ff34 7fc00000
	v_cndmask_b32_e64 v53, v53, 0x7fc00000, s53                // 000000002bd0: d5010035 00d5ff35 7fc00000
	v_cndmask_b32_e64 v143, v143, 0x7fc00000, s54              // 000000002bdc: d501008f 00d9ff8f 7fc00000
	v_cndmask_b32_e64 v144, v144, 0x7fc00000, s55              // 000000002be8: d5010090 00ddff90 7fc00000
	v_cndmask_b32_e64 v145, v145, 0x7fc00000, s56              // 000000002bf4: d5010091 00e1ff91 7fc00000
	v_cndmask_b32_e64 v146, v146, 0x7fc00000, s57              // 000000002c00: d5010092 00e5ff92 7fc00000
	v_cndmask_b32_e64 v147, v147, 0x7fc00000, s58              // 000000002c0c: d5010093 00e9ff93 7fc00000
	v_cndmask_b32_e64 v148, v148, 0x7fc00000, s59              // 000000002c18: d5010094 00edff94 7fc00000
	v_cndmask_b32_e64 v149, v149, 0x7fc00000, s60              // 000000002c24: d5010095 00f1ff95 7fc00000
	v_cndmask_b32_e64 v150, v150, 0x7fc00000, s61              // 000000002c30: d5010096 00f5ff96 7fc00000
	v_cndmask_b32_e64 v151, v151, 0x7fc00000, s5               // 000000002c3c: d5010097 0015ff97 7fc00000
	v_cndmask_b32_e64 v152, v152, 0x7fc00000, s7               // 000000002c48: d5010098 001dff98 7fc00000
	v_cndmask_b32_e64 v153, v153, 0x7fc00000, s8               // 000000002c54: d5010099 0021ff99 7fc00000
	v_cndmask_b32_e64 v154, v154, 0x7fc00000, s9               // 000000002c60: d501009a 0025ff9a 7fc00000
	v_cndmask_b32_e64 v155, v155, 0x7fc00000, s10              // 000000002c6c: d501009b 0029ff9b 7fc00000
	v_cndmask_b32_e64 v156, v156, 0x7fc00000, s11              // 000000002c78: d501009c 002dff9c 7fc00000
	v_cndmask_b32_e64 v157, v157, 0x7fc00000, s12              // 000000002c84: d501009d 0031ff9d 7fc00000
	v_cndmask_b32_e64 v158, v158, 0x7fc00000, s13              // 000000002c90: d501009e 0035ff9e 7fc00000
	v_cndmask_b32_e64 v159, v159, 0x7fc00000, s62              // 000000002c9c: d501009f 00f9ff9f 7fc00000
	v_cndmask_b32_e64 v160, v160, 0x7fc00000, s63              // 000000002ca8: d50100a0 00fdffa0 7fc00000
	v_cndmask_b32_e64 v161, v161, 0x7fc00000, s64              // 000000002cb4: d50100a1 0101ffa1 7fc00000
	v_cndmask_b32_e64 v162, v162, 0x7fc00000, s65              // 000000002cc0: d50100a2 0105ffa2 7fc00000
	v_cndmask_b32_e64 v163, v163, 0x7fc00000, s66              // 000000002ccc: d50100a3 0109ffa3 7fc00000
	v_cndmask_b32_e64 v164, v164, 0x7fc00000, s67              // 000000002cd8: d50100a4 010dffa4 7fc00000
	v_cndmask_b32_e64 v165, v165, 0x7fc00000, s68              // 000000002ce4: d50100a5 0111ffa5 7fc00000
	v_cndmask_b32_e64 v54, v54, 0x7fc00000, s6                 // 000000002cf0: d5010036 0019ff36 7fc00000
	v_cndmask_b32_e64 v166, v166, 0x7fc00000, s69              // 000000002cfc: d50100a6 0115ffa6 7fc00000
	v_cndmask_b32_e64 v167, v167, 0x7fc00000, s70              // 000000002d08: d50100a7 0119ffa7 7fc00000
	v_cndmask_b32_e64 v168, v168, 0x7fc00000, s71              // 000000002d14: d50100a8 011dffa8 7fc00000
	v_cndmask_b32_e64 v169, v169, 0x7fc00000, s72              // 000000002d20: d50100a9 0121ffa9 7fc00000
	v_cndmask_b32_e64 v170, v170, 0x7fc00000, s73              // 000000002d2c: d50100aa 0125ffaa 7fc00000
	v_cndmask_b32_e64 v171, v171, 0x7fc00000, s74              // 000000002d38: d50100ab 0129ffab 7fc00000
	v_cndmask_b32_e64 v172, v172, 0x7fc00000, s75              // 000000002d44: d50100ac 012dffac 7fc00000
	v_cndmask_b32_e64 v55, v55, 0x7fc00000, s14                // 000000002d50: d5010037 0039ff37 7fc00000
	v_cndmask_b32_e64 v173, v173, 0x7fc00000, s76              // 000000002d5c: d50100ad 0131ffad 7fc00000
	v_cndmask_b32_e64 v174, v174, 0x7fc00000, s77              // 000000002d68: d50100ae 0135ffae 7fc00000
	v_cndmask_b32_e64 v175, v175, 0x7fc00000, s78              // 000000002d74: d50100af 0139ffaf 7fc00000
	v_cndmask_b32_e64 v176, v176, 0x7fc00000, s79              // 000000002d80: d50100b0 013dffb0 7fc00000
	v_cndmask_b32_e64 v177, v177, 0x7fc00000, s80              // 000000002d8c: d50100b1 0141ffb1 7fc00000
	v_cndmask_b32_e64 v178, v178, 0x7fc00000, s81              // 000000002d98: d50100b2 0145ffb2 7fc00000
	v_cndmask_b32_e64 v179, v179, 0x7fc00000, s82              // 000000002da4: d50100b3 0149ffb3 7fc00000
	v_cndmask_b32_e64 v56, v56, 0x7fc00000, s15                // 000000002db0: d5010038 003dff38 7fc00000
	v_cndmask_b32_e64 v180, v180, 0x7fc00000, s17              // 000000002dbc: d50100b4 0045ffb4 7fc00000
	v_cndmask_b32_e64 v181, v181, 0x7fc00000, s18              // 000000002dc8: d50100b5 0049ffb5 7fc00000
	v_cndmask_b32_e64 v182, v182, 0x7fc00000, s19              // 000000002dd4: d50100b6 004dffb6 7fc00000
	v_cndmask_b32_e64 v183, v183, 0x7fc00000, s20              // 000000002de0: d50100b7 0051ffb7 7fc00000
	v_cndmask_b32_e64 v184, v184, 0x7fc00000, s21              // 000000002dec: d50100b8 0055ffb8 7fc00000
	v_cndmask_b32_e64 v185, v185, 0x7fc00000, s22              // 000000002df8: d50100b9 0059ffb9 7fc00000
	v_cndmask_b32_e64 v186, v186, 0x7fc00000, s23              // 000000002e04: d50100ba 005dffba 7fc00000
	v_cndmask_b32_e64 v57, v57, 0x7fc00000, s16                // 000000002e10: d5010039 0041ff39 7fc00000
	v_dual_add_f32 v107, v107, v38 :: v_dual_add_f32 v138, v138, v39// 000000002e1c: c9084d6b 6b8a4f8a
	v_add_f32_e32 v137, v137, v40                              // 000000002e24: 07125189
	v_dual_add_f32 v135, v135, v41 :: v_dual_add_f32 v132, v132, v42// 000000002e28: c9085387 87845584
	v_dual_add_f32 v128, v128, v43 :: v_dual_add_f32 v127, v127, v44// 000000002e30: c9085780 807e597f
	v_add_f32_e32 v126, v126, v45                              // 000000002e38: 06fc5b7e
	v_add_f32_e32 v108, v108, v46                              // 000000002e3c: 06d85d6c
	v_dual_add_f32 v106, v106, v47 :: v_dual_add_f32 v105, v105, v48// 000000002e40: c9085f6a 6a686169
	v_dual_add_f32 v104, v104, v49 :: v_dual_add_f32 v103, v103, v50// 000000002e48: c9086368 68666567
	v_dual_add_f32 v102, v102, v51 :: v_dual_add_f32 v101, v101, v52// 000000002e50: c9086766 66646965
	v_dual_add_f32 v100, v100, v53 :: v_dual_add_f32 v91, v91, v143// 000000002e58: c9086b64 645b1f5b
	v_dual_add_f32 v90, v90, v144 :: v_dual_add_f32 v89, v89, v145// 000000002e60: c909215a 5a592359
	v_dual_add_f32 v88, v88, v146 :: v_dual_add_f32 v87, v87, v147// 000000002e68: c9092558 58572757
	v_dual_add_f32 v86, v86, v148 :: v_dual_add_f32 v85, v85, v149// 000000002e70: c9092956 56552b55
	v_dual_add_f32 v84, v84, v150 :: v_dual_add_f32 v75, v75, v151// 000000002e78: c9092d54 544b2f4b
	v_dual_add_f32 v74, v74, v152 :: v_dual_add_f32 v73, v73, v153// 000000002e80: c909314a 4a493349
	v_dual_add_f32 v72, v72, v154 :: v_dual_add_f32 v71, v71, v155// 000000002e88: c9093548 48473747
	v_dual_add_f32 v70, v70, v156 :: v_dual_add_f32 v69, v69, v157// 000000002e90: c9093946 46453b45
	v_dual_add_f32 v68, v68, v158 :: v_dual_add_f32 v125, v125, v159// 000000002e98: c9093d44 447d3f7d
	v_dual_add_f32 v124, v124, v160 :: v_dual_add_f32 v121, v121, v161// 000000002ea0: c909417c 7c794379
	v_dual_add_f32 v120, v120, v162 :: v_dual_add_f32 v119, v119, v163// 000000002ea8: c9094578 78774777
	v_dual_add_f32 v114, v114, v164 :: v_dual_add_f32 v111, v111, v165// 000000002eb0: c9094972 726f4b6f
	v_add_f32_e32 v109, v109, v54                              // 000000002eb8: 06da6d6d
	v_dual_add_f32 v99, v99, v166 :: v_dual_add_f32 v98, v98, v167// 000000002ebc: c9094d63 63634f62
	v_dual_add_f32 v97, v97, v168 :: v_dual_add_f32 v96, v96, v169// 000000002ec4: c9095161 61615360
	v_dual_add_f32 v95, v95, v170 :: v_dual_add_f32 v94, v94, v171// 000000002ecc: c909555f 5f5f575e
	v_dual_add_f32 v93, v93, v172 :: v_dual_add_f32 v92, v92, v55// 000000002ed4: c909595d 5d5c6f5c
	v_dual_add_f32 v83, v83, v173 :: v_dual_add_f32 v82, v82, v174// 000000002edc: c9095b53 53535d52
	v_dual_add_f32 v81, v81, v175 :: v_dual_add_f32 v80, v80, v176// 000000002ee4: c9095f51 51516150
	v_dual_add_f32 v79, v79, v177 :: v_dual_add_f32 v78, v78, v178// 000000002eec: c909634f 4f4f654e
	v_dual_add_f32 v77, v77, v179 :: v_dual_add_f32 v76, v76, v56// 000000002ef4: c909674d 4d4c714c
	v_dual_add_f32 v67, v67, v180 :: v_dual_add_f32 v66, v66, v181// 000000002efc: c9096943 43436b42
	v_dual_add_f32 v65, v65, v182 :: v_dual_add_f32 v64, v64, v183// 000000002f04: c9096d41 41416f40
	v_dual_add_f32 v63, v63, v184 :: v_dual_add_f32 v62, v62, v185// 000000002f0c: c909713f 3f3f733e
	v_dual_add_f32 v61, v61, v186 :: v_dual_add_f32 v60, v60, v57// 000000002f14: c909753d 3d3c733c
	s_cmp_lg_u64 s[38:39], s[36:37]                            // 000000002f1c: bf112426
	s_cbranch_scc1 64700                                       // 000000002f20: bfa2fcbc <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x714>
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000002f24: f4002500 f80000a8
	v_mul_lo_u32 v8, s31, v2                                   // 000000002f2c: d72c0008 0202041f
	v_mul_lo_u32 v9, s30, v3                                   // 000000002f34: d72c0009 0202061e
	v_mad_co_u64_u32 v[6:7], null, s30, v2, 0                  // 000000002f3c: d6fe7c06 0202041e
	v_sub_co_u32 v20, s0, s28, v2                              // 000000002f44: d7010014 0202041c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000002f4c: bf870191
	v_sub_co_ci_u32_e64 v21, null, s29, v3, s0                 // 000000002f50: d5217c15 0002061d
	v_add3_u32 v7, v7, v9, v8                                  // 000000002f58: d6550007 04221307
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002f60: bf870112
	v_cmp_lt_i64_e64 s17, 0, v[20:21]                          // 000000002f64: d4510011 02022880
	v_lshlrev_b64_e32 v[2:3], 1, v[6:7]                        // 000000002f6c: 3e040c81
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000002f70: 3e0c0081
	s_and_b32 s0, s17, s4                                      // 000000002f74: 8b000411
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f78: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000002f7c: be812000
	s_cbranch_execz 28                                         // 000000002f80: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x14f4>
	v_bfe_u32 v8, v107, 16, 1                                  // 000000002f84: d6100008 0205216b
	s_wait_kmcnt 0x0                                           // 000000002f8c: bfc70000
	v_add_co_u32 v9, s0, s20, v2                               // 000000002f90: d7000009 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000002f98: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s21, v3, s0                 // 000000002f9c: d5207c0a 00020615
	v_add3_u32 v11, v8, v107, 0x7fff                           // 000000002fa4: d655000b 03fed708 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002fb0: bf870003
	v_add_co_u32 v8, s0, v9, v6                                // 000000002fb4: d7000008 02020d09
	v_or_b32_e32 v12, 0x400000, v107                           // 000000002fbc: 3818d6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v7, s0                  // 000000002fc8: d5207c09 00020f0a
	v_cmp_u_f32_e64 s0, v107, v107                             // 000000002fd0: d4180000 0202d76b
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002fdc: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s0                        // 000000002fe0: d501000a 0002190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 000000002fe8: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000002ff8: 8c7e017e
	v_add_co_u32 v8, s0, s30, v0                               // 000000002ffc: d7000008 0202001e
	s_wait_alu depctr_va_sdst(0)                               // 000000003004: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s31, v1, s0                  // 000000003008: d5207c09 0002021f
	v_cmp_lt_i64_e64 s18, 1, v[20:21]                          // 000000003010: d4510012 02022881
	s_delay_alu instid0(valu_dep_2)                            // 000000003018: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 00000000301c: 3e101081
	s_and_b32 s0, s18, s4                                      // 000000003020: 8b000412
	s_wait_alu depctr_sa_sdst(0)                               // 000000003024: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003028: be812000
	s_cbranch_execz 28                                         // 00000000302c: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x15a0>
	v_bfe_u32 v10, v138, 16, 1                                 // 000000003030: d610000a 0205218a
	s_wait_kmcnt 0x0                                           // 000000003038: bfc70000
	v_add_co_u32 v11, s0, s20, v2                              // 00000000303c: d700000b 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003044: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s21, v3, s0                 // 000000003048: d5207c0c 00020615
	v_add3_u32 v13, v10, v138, 0x7fff                          // 000000003050: d655000d 03ff150a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000305c: bf870003
	v_add_co_u32 v10, s0, v11, v8                              // 000000003060: d700000a 0202110b
	v_or_b32_e32 v14, 0x400000, v138                           // 000000003068: 381d14ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003070: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s0                 // 000000003074: d5207c0b 0002130c
	v_cmp_u_f32_e64 s0, v138, v138                             // 00000000307c: d4180000 0203158a
	s_wait_alu depctr_va_sdst(0)                               // 000000003084: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003088: bf870001
	v_cndmask_b32_e64 v12, v13, v14, s0                        // 00000000308c: d501000c 00021d0d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 000000003094: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000030a4: 8c7e017e
	s_lshl_b64 s[40:41], s[30:31], 1                           // 0000000030a8: 84a8811e
	v_cmp_lt_i64_e64 s16, 2, v[20:21]                          // 0000000030ac: d4510010 02022882
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030b4: bf88ff9e
	v_add_co_u32 v10, s0, s40, v0                              // 0000000030b8: d700000a 02020028
	s_wait_alu depctr_va_sdst(0)                               // 0000000030c0: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s41, v1, s0                 // 0000000030c4: d5207c0b 00020229
	s_and_b32 s0, s16, s4                                      // 0000000030cc: 8b000410
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 0000000030d0: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030d4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000030d8: be812000
	s_cbranch_execz 28                                         // 0000000030dc: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1650>
	v_bfe_u32 v12, v137, 16, 1                                 // 0000000030e0: d610000c 02052189
	s_wait_kmcnt 0x0                                           // 0000000030e8: bfc70000
	v_add_co_u32 v13, s0, s20, v2                              // 0000000030ec: d700000d 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000030f4: bf88f19f
	v_add_co_ci_u32_e64 v14, null, s21, v3, s0                 // 0000000030f8: d5207c0e 00020615
	v_add3_u32 v15, v12, v137, 0x7fff                          // 000000003100: d655000f 03ff130c 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000310c: bf870003
	v_add_co_u32 v12, s0, v13, v10                             // 000000003110: d700000c 0202150d
	v_or_b32_e32 v16, 0x400000, v137                           // 000000003118: 382112ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003120: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v14, v11, s0                // 000000003124: d5207c0d 0002170e
	v_cmp_u_f32_e64 s0, v137, v137                             // 00000000312c: d4180000 02031389
	s_wait_alu depctr_va_sdst(0)                               // 000000003134: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003138: bf870001
	v_cndmask_b32_e64 v14, v15, v16, s0                        // 00000000313c: d501000e 0002210f
	global_store_d16_hi_b16 v[12:13], v14, off                 // 000000003144: ee09407c 07000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003150: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003154: 8c7e017e
	s_mul_u64 s[38:39], s[30:31], 3                            // 000000003158: aaa6831e
	v_cmp_lt_i64_e64 s15, 3, v[20:21]                          // 00000000315c: d451000f 02022883
	s_wait_alu depctr_sa_sdst(0)                               // 000000003164: bf88ff9e
	v_add_co_u32 v12, s0, s38, v0                              // 000000003168: d700000c 02020026
	s_wait_alu depctr_va_sdst(0)                               // 000000003170: bf88f19f
	v_add_co_ci_u32_e64 v13, null, s39, v1, s0                 // 000000003174: d5207c0d 00020227
	s_and_b32 s0, s15, s4                                      // 00000000317c: 8b00040f
	v_lshlrev_b64_e32 v[12:13], 1, v[12:13]                    // 000000003180: 3e181881
	s_wait_alu depctr_sa_sdst(0)                               // 000000003184: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003188: be812000
	s_cbranch_execz 28                                         // 00000000318c: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1700>
	v_bfe_u32 v14, v135, 16, 1                                 // 000000003190: d610000e 02052187
	s_wait_kmcnt 0x0                                           // 000000003198: bfc70000
	v_add_co_u32 v15, s0, s20, v2                              // 00000000319c: d700000f 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a4: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s21, v3, s0                 // 0000000031a8: d5207c10 00020615
	v_add3_u32 v17, v14, v135, 0x7fff                          // 0000000031b0: d6550011 03ff0f0e 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000031bc: bf870003
	v_add_co_u32 v14, s0, v15, v12                             // 0000000031c0: d700000e 0202190f
	v_or_b32_e32 v18, 0x400000, v135                           // 0000000031c8: 38250eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000031d0: bf88f19f
	v_add_co_ci_u32_e64 v15, null, v16, v13, s0                // 0000000031d4: d5207c0f 00021b10
	v_cmp_u_f32_e64 s0, v135, v135                             // 0000000031dc: d4180000 02030f87
	s_wait_alu depctr_va_sdst(0)                               // 0000000031e4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000031e8: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s0                        // 0000000031ec: d5010010 00022511
	global_store_d16_hi_b16 v[14:15], v16, off                 // 0000000031f4: ee09407c 08000000 0000000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003200: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003204: 8c7e017e
	s_lshl_b64 s[36:37], s[30:31], 2                           // 000000003208: 84a4821e
	v_cmp_lt_i64_e64 s14, 4, v[20:21]                          // 00000000320c: d451000e 02022884
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	v_add_co_u32 v14, s0, s36, v0                              // 000000003218: d700000e 02020024
	s_wait_alu depctr_va_sdst(0)                               // 000000003220: bf88f19f
	v_add_co_ci_u32_e64 v15, null, s37, v1, s0                 // 000000003224: d5207c0f 00020225
	s_and_b32 s0, s14, s4                                      // 00000000322c: 8b00040e
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000003230: 3e1c1c81
	s_wait_alu depctr_sa_sdst(0)                               // 000000003234: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003238: be812000
	s_cbranch_execz 28                                         // 00000000323c: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x17b0>
	v_bfe_u32 v16, v132, 16, 1                                 // 000000003240: d6100010 02052184
	s_wait_kmcnt 0x0                                           // 000000003248: bfc70000
	v_add_co_u32 v17, s0, s20, v2                              // 00000000324c: d7000011 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003254: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v3, s0                 // 000000003258: d5207c12 00020615
	v_add3_u32 v19, v16, v132, 0x7fff                          // 000000003260: d6550013 03ff0910 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000326c: bf870003
	v_add_co_u32 v16, s0, v17, v14                             // 000000003270: d7000010 02021d11
	v_or_b32_e32 v22, 0x400000, v132                           // 000000003278: 382d08ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003280: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s0                // 000000003284: d5207c11 00021f12
	v_cmp_u_f32_e64 s0, v132, v132                             // 00000000328c: d4180000 02030984
	s_wait_alu depctr_va_sdst(0)                               // 000000003294: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003298: bf870001
	v_cndmask_b32_e64 v18, v19, v22, s0                        // 00000000329c: d5010012 00022d13
	global_store_d16_hi_b16 v[16:17], v18, off                 // 0000000032a4: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000032b4: 8c7e017e
	s_mul_u64 s[34:35], s[30:31], 5                            // 0000000032b8: aaa2851e
	v_cmp_lt_i64_e64 s13, 5, v[20:21]                          // 0000000032bc: d451000d 02022885
	v_add_co_u32 v16, s0, s34, v0                              // 0000000032c4: d7000010 02020022
	s_wait_alu depctr_va_sdst(0)                               // 0000000032cc: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s35, v1, s0                 // 0000000032d0: d5207c11 00020223
	s_and_b32 s0, s13, s4                                      // 0000000032d8: 8b00040d
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 0000000032dc: 3e202081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032e0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000032e4: be812000
	s_cbranch_execz 28                                         // 0000000032e8: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x185c>
	v_bfe_u32 v18, v128, 16, 1                                 // 0000000032ec: d6100012 02052180
	s_wait_kmcnt 0x0                                           // 0000000032f4: bfc70000
	v_add_co_u32 v19, s0, s20, v2                              // 0000000032f8: d7000013 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003300: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v3, s0                 // 000000003304: d5207c16 00020615
	v_add3_u32 v23, v18, v128, 0x7fff                          // 00000000330c: d6550017 03ff0112 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003318: bf870003
	v_add_co_u32 v18, s0, v19, v16                             // 00000000331c: d7000012 02022113
	v_or_b32_e32 v24, 0x400000, v128                           // 000000003324: 383100ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000332c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v22, v17, s0                // 000000003330: d5207c13 00022316
	v_cmp_u_f32_e64 s0, v128, v128                             // 000000003338: d4180000 02030180
	s_wait_alu depctr_va_sdst(0)                               // 000000003340: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003344: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 000000003348: d5010016 00023117
	global_store_d16_hi_b16 v[18:19], v22, off                 // 000000003350: ee09407c 0b000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 00000000335c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003360: 8c7e017e
	s_mul_u64 s[24:25], s[30:31], 6                            // 000000003364: aa98861e
	v_cmp_lt_i64_e64 s11, 6, v[20:21]                          // 000000003368: d451000b 02022886
	s_wait_alu depctr_sa_sdst(0)                               // 000000003370: bf88ff9e
	v_add_co_u32 v18, s0, s24, v0                              // 000000003374: d7000012 02020018
	s_wait_alu depctr_va_sdst(0)                               // 00000000337c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s25, v1, s0                 // 000000003380: d5207c13 00020219
	s_and_b32 s0, s11, s4                                      // 000000003388: 8b00040b
	v_lshlrev_b64_e32 v[18:19], 1, v[18:19]                    // 00000000338c: 3e242481
	s_wait_alu depctr_sa_sdst(0)                               // 000000003390: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003394: be812000
	s_cbranch_execz 28                                         // 000000003398: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x190c>
	v_bfe_u32 v22, v127, 16, 1                                 // 00000000339c: d6100016 0205217f
	s_wait_kmcnt 0x0                                           // 0000000033a4: bfc70000
	v_add_co_u32 v23, s0, s20, v2                              // 0000000033a8: d7000017 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000033b0: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s21, v3, s0                 // 0000000033b4: d5207c18 00020615
	v_add3_u32 v25, v22, v127, 0x7fff                          // 0000000033bc: d6550019 03feff16 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000033c8: bf870003
	v_add_co_u32 v22, s0, v23, v18                             // 0000000033cc: d7000016 02022517
	v_or_b32_e32 v26, 0x400000, v127                           // 0000000033d4: 3834feff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000033dc: bf88f19f
	v_add_co_ci_u32_e64 v23, null, v24, v19, s0                // 0000000033e0: d5207c17 00022718
	v_cmp_u_f32_e64 s0, v127, v127                             // 0000000033e8: d4180000 0202ff7f
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000033f4: bf870001
	v_cndmask_b32_e64 v24, v25, v26, s0                        // 0000000033f8: d5010018 00023519
	global_store_d16_hi_b16 v[22:23], v24, off                 // 000000003400: ee09407c 0c000000 00000016
	s_wait_alu depctr_sa_sdst(0)                               // 00000000340c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003410: 8c7e017e
	s_mul_u64 s[22:23], s[30:31], 7                            // 000000003414: aa96871e
	v_cmp_lt_i64_e64 s10, 7, v[20:21]                          // 000000003418: d451000a 02022887
	s_wait_alu depctr_sa_sdst(0)                               // 000000003420: bf88ff9e
	v_add_co_u32 v0, s0, s22, v0                               // 000000003424: d7000000 02020016
	s_wait_alu depctr_va_sdst(0)                               // 00000000342c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s23, v1, s0                  // 000000003430: d5207c01 00020217
	s_and_b32 s0, s10, s4                                      // 000000003438: 8b00040a
	v_lshlrev_b64_e32 v[20:21], 1, v[0:1]                      // 00000000343c: 3e280081
	s_wait_alu depctr_sa_sdst(0)                               // 000000003440: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003444: be812000
	s_cbranch_execz 28                                         // 000000003448: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x19bc>
	v_bfe_u32 v0, v126, 16, 1                                  // 00000000344c: d6100000 0205217e
	s_wait_kmcnt 0x0                                           // 000000003454: bfc70000
	v_add_co_u32 v1, s0, s20, v2                               // 000000003458: d7000001 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003460: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v3, s0                 // 000000003464: d5207c16 00020615
	v_add3_u32 v23, v0, v126, 0x7fff                           // 00000000346c: d6550017 03fefd00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003478: bf870003
	v_add_co_u32 v0, s0, v1, v20                               // 00000000347c: d7000000 02022901
	v_or_b32_e32 v24, 0x400000, v126                           // 000000003484: 3830fcff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000348c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v22, v21, s0                 // 000000003490: d5207c01 00022b16
	v_cmp_u_f32_e64 s0, v126, v126                             // 000000003498: d4180000 0202fd7e
	s_wait_alu depctr_va_sdst(0)                               // 0000000034a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000034a4: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s0                        // 0000000034a8: d5010016 00023117
	global_store_d16_hi_b16 v[0:1], v22, off                   // 0000000034b0: ee09407c 0b000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000034c0: 8c7e017e
	v_mul_lo_u32 v22, s31, v4                                  // 0000000034c4: d72c0016 0202081f
	v_mul_lo_u32 v23, s30, v5                                  // 0000000034cc: d72c0017 02020a1e
	v_mad_co_u64_u32 v[0:1], null, s30, v4, 0                  // 0000000034d4: d6fe7c00 0202081e
	v_sub_co_u32 v4, s0, s28, v4                               // 0000000034dc: d7010004 0202081c
	s_wait_alu depctr_va_sdst(0)                               // 0000000034e4: bf88f19f
	v_sub_co_ci_u32_e64 v5, null, s29, v5, s0                  // 0000000034e8: d5217c05 00020a1d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000034f0: bf870211
	v_cmp_lt_i64_e64 s12, 0, v[4:5]                            // 0000000034f4: d451000c 02020880
	v_add3_u32 v1, v1, v23, v22                                // 0000000034fc: d6550001 045a2f01
	s_delay_alu instid0(valu_dep_1)                            // 000000003504: bf870001
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003508: 3e000081
	s_and_b32 s0, s12, s4                                      // 00000000350c: 8b00040c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003510: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003514: be812000
	s_cbranch_execz 28                                         // 000000003518: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1a8c>
	s_wait_kmcnt 0x0                                           // 00000000351c: bfc70000
	v_add_co_u32 v23, s0, s20, v0                              // 000000003520: d7000017 02020014
	v_bfe_u32 v22, v125, 16, 1                                 // 000000003528: d6100016 0205217d
	s_wait_alu depctr_va_sdst(0)                               // 000000003530: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s21, v1, s0                 // 000000003534: d5207c18 00020215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000353c: bf870193
	v_add_co_u32 v6, s0, v23, v6                               // 000000003540: d7000006 02020d17
	v_add3_u32 v22, v22, v125, 0x7fff                          // 000000003548: d6550016 03fefb16 00007fff
	v_or_b32_e32 v25, 0x400000, v125                           // 000000003554: 3832faff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000355c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v24, v7, s0                  // 000000003560: d5207c07 00020f18
	v_cmp_u_f32_e64 s0, v125, v125                             // 000000003568: d4180000 0202fb7d
	s_wait_alu depctr_va_sdst(0)                               // 000000003570: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003574: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s0                        // 000000003578: d5010016 00023316
	global_store_d16_hi_b16 v[6:7], v22, off                   // 000000003580: ee09407c 0b000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000358c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003590: 8c7e017e
	v_cmp_lt_i64_e64 s9, 1, v[4:5]                             // 000000003594: d4510009 02020881
	s_and_b32 s0, s9, s4                                       // 00000000359c: 8b000409
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000035a4: be812000
	s_cbranch_execz 28                                         // 0000000035a8: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1b1c>
	v_bfe_u32 v6, v124, 16, 1                                  // 0000000035ac: d6100006 0205217c
	s_wait_kmcnt 0x0                                           // 0000000035b4: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 0000000035b8: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c0: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v1, s0                 // 0000000035c4: d5207c16 00020215
	v_add3_u32 v23, v6, v124, 0x7fff                           // 0000000035cc: d6550017 03fef906 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000035d8: bf870003
	v_add_co_u32 v6, s0, v7, v8                                // 0000000035dc: d7000006 02021107
	v_or_b32_e32 v24, 0x400000, v124                           // 0000000035e4: 3830f8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000035ec: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v22, v9, s0                  // 0000000035f0: d5207c07 00021316
	v_cmp_u_f32_e64 s0, v124, v124                             // 0000000035f8: d4180000 0202f97c
	s_wait_alu depctr_va_sdst(0)                               // 000000003600: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003604: bf870001
	v_cndmask_b32_e64 v8, v23, v24, s0                         // 000000003608: d5010008 00023117
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003610: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000361c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003620: 8c7e017e
	v_cmp_lt_i64_e64 s8, 2, v[4:5]                             // 000000003624: d4510008 02020882
	s_and_b32 s0, s8, s4                                       // 00000000362c: 8b000408
	s_wait_alu depctr_sa_sdst(0)                               // 000000003630: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003634: be812000
	s_cbranch_execz 28                                         // 000000003638: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1bac>
	v_bfe_u32 v6, v121, 16, 1                                  // 00000000363c: d6100006 02052179
	s_wait_kmcnt 0x0                                           // 000000003644: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003648: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003650: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003654: d5207c08 00020215
	v_add3_u32 v9, v6, v121, 0x7fff                            // 00000000365c: d6550009 03fef306 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003668: bf870003
	v_add_co_u32 v6, s0, v7, v10                               // 00000000366c: d7000006 02021507
	v_or_b32_e32 v22, 0x400000, v121                           // 000000003674: 382cf2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000367c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v11, s0                  // 000000003680: d5207c07 00021708
	v_cmp_u_f32_e64 s0, v121, v121                             // 000000003688: d4180000 0202f379
	s_wait_alu depctr_va_sdst(0)                               // 000000003690: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003694: bf870001
	v_cndmask_b32_e64 v8, v9, v22, s0                          // 000000003698: d5010008 00022d09
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000036a0: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000036b0: 8c7e017e
	v_cmp_lt_i64_e64 s7, 3, v[4:5]                             // 0000000036b4: d4510007 02020883
	s_and_b32 s0, s7, s4                                       // 0000000036bc: 8b000407
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036c0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000036c4: be812000
	s_cbranch_execz 28                                         // 0000000036c8: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1c3c>
	v_bfe_u32 v6, v120, 16, 1                                  // 0000000036cc: d6100006 02052178
	s_wait_kmcnt 0x0                                           // 0000000036d4: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 0000000036d8: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000036e0: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 0000000036e4: d5207c08 00020215
	v_add3_u32 v9, v6, v120, 0x7fff                            // 0000000036ec: d6550009 03fef106 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000036f8: bf870003
	v_add_co_u32 v6, s0, v7, v12                               // 0000000036fc: d7000006 02021907
	v_or_b32_e32 v10, 0x400000, v120                           // 000000003704: 3814f0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000370c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v13, s0                  // 000000003710: d5207c07 00021b08
	v_cmp_u_f32_e64 s0, v120, v120                             // 000000003718: d4180000 0202f178
	s_wait_alu depctr_va_sdst(0)                               // 000000003720: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003724: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003728: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003730: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000373c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003740: 8c7e017e
	v_cmp_lt_i64_e64 s6, 4, v[4:5]                             // 000000003744: d4510006 02020884
	s_and_b32 s0, s6, s4                                       // 00000000374c: 8b000406
	s_wait_alu depctr_sa_sdst(0)                               // 000000003750: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000003754: be812000
	s_cbranch_execz 28                                         // 000000003758: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1ccc>
	v_bfe_u32 v6, v119, 16, 1                                  // 00000000375c: d6100006 02052177
	s_wait_kmcnt 0x0                                           // 000000003764: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003768: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003770: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003774: d5207c08 00020215
	v_add3_u32 v9, v6, v119, 0x7fff                            // 00000000377c: d6550009 03feef06 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003788: bf870003
	v_add_co_u32 v6, s0, v7, v14                               // 00000000378c: d7000006 02021d07
	v_or_b32_e32 v10, 0x400000, v119                           // 000000003794: 3814eeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000379c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v15, s0                  // 0000000037a0: d5207c07 00021f08
	v_cmp_u_f32_e64 s0, v119, v119                             // 0000000037a8: d4180000 0202ef77
	s_wait_alu depctr_va_sdst(0)                               // 0000000037b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000037b4: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 0000000037b8: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000037c0: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000037d0: 8c7e017e
	v_cmp_lt_i64_e64 s5, 5, v[4:5]                             // 0000000037d4: d4510005 02020885
	s_and_b32 s0, s5, s4                                       // 0000000037dc: 8b000405
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037e0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000037e4: be812000
	s_cbranch_execz 28                                         // 0000000037e8: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1d5c>
	v_bfe_u32 v6, v114, 16, 1                                  // 0000000037ec: d6100006 02052172
	s_wait_kmcnt 0x0                                           // 0000000037f4: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 0000000037f8: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003800: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003804: d5207c08 00020215
	v_add3_u32 v9, v6, v114, 0x7fff                            // 00000000380c: d6550009 03fee506 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003818: bf870003
	v_add_co_u32 v6, s0, v7, v16                               // 00000000381c: d7000006 02022107
	v_or_b32_e32 v10, 0x400000, v114                           // 000000003824: 3814e4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000382c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v17, s0                  // 000000003830: d5207c07 00022308
	v_cmp_u_f32_e64 s0, v114, v114                             // 000000003838: d4180000 0202e572
	s_wait_alu depctr_va_sdst(0)                               // 000000003840: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003844: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 000000003848: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003850: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 00000000385c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000003860: 8c7e017e
	v_cmp_lt_i64_e64 s1, 6, v[4:5]                             // 000000003864: d4510001 02020886
	s_and_b32 s0, s1, s4                                       // 00000000386c: 8b000401
	s_wait_alu depctr_sa_sdst(0)                               // 000000003870: bf88ff9e
	s_and_saveexec_b32 s19, s0                                 // 000000003874: be932000
	s_cbranch_execz 28                                         // 000000003878: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1dec>
	v_bfe_u32 v6, v111, 16, 1                                  // 00000000387c: d6100006 0205216f
	s_wait_kmcnt 0x0                                           // 000000003884: bfc70000
	v_add_co_u32 v7, s0, s20, v0                               // 000000003888: d7000007 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003890: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s0                  // 000000003894: d5207c08 00020215
	v_add3_u32 v9, v6, v111, 0x7fff                            // 00000000389c: d6550009 03fedf06 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000038a8: bf870003
	v_add_co_u32 v6, s0, v7, v18                               // 0000000038ac: d7000006 02022507
	v_or_b32_e32 v10, 0x400000, v111                           // 0000000038b4: 3814deff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000038bc: bf88f19f
	v_add_co_ci_u32_e64 v7, null, v8, v19, s0                  // 0000000038c0: d5207c07 00022708
	v_cmp_u_f32_e64 s0, v111, v111                             // 0000000038c8: d4180000 0202df6f
	s_wait_alu depctr_va_sdst(0)                               // 0000000038d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038d4: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s0                          // 0000000038d8: d5010008 00021509
	global_store_d16_hi_b16 v[6:7], v8, off                    // 0000000038e0: ee09407c 04000000 00000006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000038f0: 8c7e137e
	v_cmp_lt_i64_e64 s0, 7, v[4:5]                             // 0000000038f4: d4510000 02020887
	s_and_b32 s4, s0, s4                                       // 0000000038fc: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 000000003900: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003904: be932004
	s_cbranch_execz 28                                         // 000000003908: bfa5001c <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1e7c>
	v_bfe_u32 v4, v109, 16, 1                                  // 00000000390c: d6100004 0205216d
	s_wait_kmcnt 0x0                                           // 000000003914: bfc70000
	v_add_co_u32 v5, s4, s20, v0                               // 000000003918: d7000405 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000003920: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s4                  // 000000003924: d5207c06 00120215
	v_add3_u32 v7, v4, v109, 0x7fff                            // 00000000392c: d6550007 03fedb04 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003938: bf870003
	v_add_co_u32 v4, s4, v5, v20                               // 00000000393c: d7000404 02022905
	v_or_b32_e32 v8, 0x400000, v109                            // 000000003944: 3810daff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000394c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v6, v21, s4                  // 000000003950: d5207c05 00122b06
	v_cmp_u_f32_e64 s4, v109, v109                             // 000000003958: d4180004 0202db6d
	s_wait_alu depctr_va_sdst(0)                               // 000000003960: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003964: bf870001
	v_cndmask_b32_e64 v6, v7, v8, s4                           // 000000003968: d5010006 00121107
	global_store_d16_hi_b16 v[4:5], v6, off                    // 000000003970: ee09407c 03000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000397c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003980: 8c7e137e
	s_and_b32 s4, s17, s3                                      // 000000003984: 8b040311
	s_wait_alu depctr_sa_sdst(0)                               // 000000003988: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 00000000398c: be932004
	s_cbranch_execz 40                                         // 000000003990: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x1f34>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003994: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000399c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 0000000039a0: d5207c05 00103680
	v_bfe_u32 v6, v108, 16, 1                                  // 0000000039a8: d6100006 0205216c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039b0: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 0000000039b4: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000039bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000039c0: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 0000000039c8: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 0000000039cc: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 0000000039d8: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000039e0: 3e080881
	v_add3_u32 v6, v6, v108, 0x7fff                            // 0000000039e4: d6550006 03fed906 00007fff
	v_or_b32_e32 v9, 0x400000, v108                            // 0000000039f0: 3812d8ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000039f8: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 0000000039fc: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003a04: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003a08: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v108, v108                             // 000000003a10: d4180004 0202d96c
	s_wait_alu depctr_va_sdst(0)                               // 000000003a18: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a1c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003a20: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003a28: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003a38: 8c7e137e
	s_and_b32 s4, s18, s3                                      // 000000003a3c: 8b040312
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a40: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003a44: be932004
	s_cbranch_execz 46                                         // 000000003a48: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2004>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003a4c: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003a54: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003a58: d5207c05 00103680
	v_bfe_u32 v6, v106, 16, 1                                  // 000000003a60: d6100006 0205216a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a68: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003a6c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003a74: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003a78: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v106                            // 000000003a80: 3812d4ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003a88: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 000000003a8c: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a94: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000003a98: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 000000003aa0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003aa4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003aac: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003ab0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003ab8: 3e080881
	v_add3_u32 v6, v6, v106, 0x7fff                            // 000000003abc: d6550006 03fed506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ac8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003acc: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003ad8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v106, v106                             // 000000003ae0: d4180004 0202d56a
	s_wait_alu depctr_va_sdst(0)                               // 000000003ae8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003aec: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003af0: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003af8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003b08: 8c7e137e
	s_and_b32 s4, s16, s3                                      // 000000003b0c: 8b040310
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b10: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003b14: be932004
	s_cbranch_execz 46                                         // 000000003b18: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x20d4>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003b1c: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003b24: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003b28: d5207c05 00103680
	v_bfe_u32 v6, v105, 16, 1                                  // 000000003b30: d6100006 02052169
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b38: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003b3c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003b44: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003b48: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v105                            // 000000003b50: 3812d2ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b58: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 000000003b5c: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000003b64: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 000000003b68: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 000000003b70: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003b74: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003b7c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003b80: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003b88: 3e080881
	v_add3_u32 v6, v6, v105, 0x7fff                            // 000000003b8c: d6550006 03fed306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003b98: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003b9c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ba4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003ba8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v105, v105                             // 000000003bb0: d4180004 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 000000003bb8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003bbc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003bc0: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003bc8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003bd8: 8c7e137e
	s_and_b32 s4, s15, s3                                      // 000000003bdc: 8b04030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003be0: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003be4: be932004
	s_cbranch_execz 46                                         // 000000003be8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x21a4>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003bec: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003bf4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003bf8: d5207c05 00103680
	v_bfe_u32 v6, v104, 16, 1                                  // 000000003c00: d6100006 02052168
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c08: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003c0c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003c14: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003c18: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v104                            // 000000003c20: 3812d0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c28: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 000000003c2c: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000003c34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 000000003c38: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 000000003c40: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003c44: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003c4c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003c50: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003c58: 3e080881
	v_add3_u32 v6, v6, v104, 0x7fff                            // 000000003c5c: d6550006 03fed106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003c68: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003c6c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003c74: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003c78: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v104, v104                             // 000000003c80: d4180004 0202d168
	s_wait_alu depctr_va_sdst(0)                               // 000000003c88: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003c8c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003c90: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003c98: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ca4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003ca8: 8c7e137e
	s_and_b32 s4, s14, s3                                      // 000000003cac: 8b04030e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003cb0: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003cb4: be932004
	s_cbranch_execz 46                                         // 000000003cb8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2274>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003cbc: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003cc8: d5207c05 00103680
	v_bfe_u32 v6, v103, 16, 1                                  // 000000003cd0: d6100006 02052167
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cd8: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003cdc: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003ce8: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v103                            // 000000003cf0: 3812ceff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cf8: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 000000003cfc: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 000000003d04: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 000000003d08: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000003d10: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003d14: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003d1c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003d20: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003d28: 3e080881
	v_add3_u32 v6, v6, v103, 0x7fff                            // 000000003d2c: d6550006 03fecf06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003d38: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003d3c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003d44: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003d48: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v103, v103                             // 000000003d50: d4180004 0202cf67
	s_wait_alu depctr_va_sdst(0)                               // 000000003d58: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003d5c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003d60: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003d68: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003d78: 8c7e137e
	s_and_b32 s4, s13, s3                                      // 000000003d7c: 8b04030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d80: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003d84: be932004
	s_cbranch_execz 46                                         // 000000003d88: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2344>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003d8c: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003d94: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003d98: d5207c05 00103680
	v_bfe_u32 v6, v102, 16, 1                                  // 000000003da0: d6100006 02052166
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003da8: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003dac: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003db4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003db8: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v102                            // 000000003dc0: 3812ccff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003dc8: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 000000003dcc: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000003dd8: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000003de0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003de4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003dec: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003df0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003df8: 3e080881
	v_add3_u32 v6, v6, v102, 0x7fff                            // 000000003dfc: d6550006 03fecd06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e08: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003e0c: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003e14: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003e18: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v102, v102                             // 000000003e20: d4180004 0202cd66
	s_wait_alu depctr_va_sdst(0)                               // 000000003e28: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003e2c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003e30: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003e38: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e48: 8c7e137e
	s_and_b32 s4, s11, s3                                      // 000000003e4c: 8b04030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e50: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003e54: be932004
	s_cbranch_execz 46                                         // 000000003e58: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2414>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003e5c: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003e64: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003e68: d5207c05 00103680
	v_bfe_u32 v6, v101, 16, 1                                  // 000000003e70: d6100006 02052165
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e78: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003e7c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003e84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003e88: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v101                            // 000000003e90: 3812caff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003e98: bf8701a3
	v_add_co_u32 v4, s4, s24, v4                               // 000000003e9c: d7000404 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000003ea4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s4                  // 000000003ea8: d5207c05 00120a19
	s_wait_kmcnt 0x0                                           // 000000003eb0: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003eb4: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003ebc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003ec0: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003ec8: 3e080881
	v_add3_u32 v6, v6, v101, 0x7fff                            // 000000003ecc: d6550006 03fecb06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ed8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003edc: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003ee4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003ee8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v101, v101                             // 000000003ef0: d4180004 0202cb65
	s_wait_alu depctr_va_sdst(0)                               // 000000003ef8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003efc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003f00: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003f08: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003f18: 8c7e137e
	s_and_b32 s4, s10, s3                                      // 000000003f1c: 8b04030a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f20: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003f24: be932004
	s_cbranch_execz 46                                         // 000000003f28: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x24e4>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003f2c: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000003f34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000003f38: d5207c05 00103680
	v_bfe_u32 v6, v100, 16, 1                                  // 000000003f40: d6100006 02052164
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f48: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000003f4c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000003f54: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000003f58: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v100                            // 000000003f60: 3812c8ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003f68: bf8701a3
	v_add_co_u32 v4, s4, s22, v4                               // 000000003f6c: d7000404 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000003f74: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s4                  // 000000003f78: d5207c05 00120a17
	s_wait_kmcnt 0x0                                           // 000000003f80: bfc70000
	v_add_co_u32 v7, s4, s20, v2                               // 000000003f84: d7000407 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000003f8c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s4                  // 000000003f90: d5207c08 00120615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000003f98: 3e080881
	v_add3_u32 v6, v6, v100, 0x7fff                            // 000000003f9c: d6550006 03fec906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000003fa8: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000003fac: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000003fb4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000003fb8: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v100, v100                             // 000000003fc0: d4180004 0202c964
	s_wait_alu depctr_va_sdst(0)                               // 000000003fc8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003fcc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000003fd0: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000003fd8: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fe4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003fe8: 8c7e137e
	s_and_b32 s4, s12, s3                                      // 000000003fec: 8b04030c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ff0: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 000000003ff4: be932004
	s_cbranch_execz 40                                         // 000000003ff8: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x259c>
	v_add_co_u32 v4, s4, v59, s26                              // 000000003ffc: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004004: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000004008: d5207c05 00103680
	v_bfe_u32 v6, v99, 16, 1                                   // 000000004010: d6100006 02052163
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004018: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 00000000401c: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004024: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004028: d5207c05 00120a80
	s_wait_kmcnt 0x0                                           // 000000004030: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 000000004034: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 00000000403c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004040: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004048: 3e080881
	v_add3_u32 v6, v6, v99, 0x7fff                             // 00000000404c: d6550006 03fec706 00007fff
	v_or_b32_e32 v9, 0x400000, v99                             // 000000004058: 3812c6ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004060: bf870223
	v_add_co_u32 v4, s4, v7, v4                                // 000000004064: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000406c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000004070: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v99, v99                               // 000000004078: d4180004 0202c763
	s_wait_alu depctr_va_sdst(0)                               // 000000004080: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004084: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004088: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004090: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000409c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000040a0: 8c7e137e
	s_and_b32 s4, s9, s3                                       // 0000000040a4: 8b040309
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040a8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000040ac: be932004
	s_cbranch_execz 46                                         // 0000000040b0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x266c>
	v_add_co_u32 v4, s4, v59, s26                              // 0000000040b4: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000040bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 0000000040c0: d5207c05 00103680
	v_bfe_u32 v6, v98, 16, 1                                   // 0000000040c8: d6100006 02052162
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040d0: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 0000000040d4: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000040dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000040e0: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v98                             // 0000000040e8: 3812c4ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000040f0: bf8701a3
	v_add_co_u32 v4, s4, s30, v4                               // 0000000040f4: d7000404 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000040fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s4                  // 000000004100: d5207c05 00120a1f
	s_wait_kmcnt 0x0                                           // 000000004108: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 00000000410c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004114: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004118: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004120: 3e080881
	v_add3_u32 v6, v6, v98, 0x7fff                             // 000000004124: d6550006 03fec506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004130: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004134: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000413c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000004140: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v98, v98                               // 000000004148: d4180004 0202c562
	s_wait_alu depctr_va_sdst(0)                               // 000000004150: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004154: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004158: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004160: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000416c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000004170: 8c7e137e
	s_and_b32 s4, s8, s3                                       // 000000004174: 8b040308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004178: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 00000000417c: be932004
	s_cbranch_execz 46                                         // 000000004180: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x273c>
	v_add_co_u32 v4, s4, v59, s26                              // 000000004184: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000418c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000004190: d5207c05 00103680
	v_bfe_u32 v6, v97, 16, 1                                   // 000000004198: d6100006 02052161
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041a0: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 0000000041a4: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000041ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000041b0: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v97                             // 0000000041b8: 3812c2ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000041c0: bf8701a3
	v_add_co_u32 v4, s4, s40, v4                               // 0000000041c4: d7000404 02020828
	s_wait_alu depctr_va_sdst(0)                               // 0000000041cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s4                  // 0000000041d0: d5207c05 00120a29
	s_wait_kmcnt 0x0                                           // 0000000041d8: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 0000000041dc: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000041e4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 0000000041e8: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000041f0: 3e080881
	v_add3_u32 v6, v6, v97, 0x7fff                             // 0000000041f4: d6550006 03fec306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004200: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004204: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000420c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000004210: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v97, v97                               // 000000004218: d4180004 0202c361
	s_wait_alu depctr_va_sdst(0)                               // 000000004220: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004224: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004228: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004230: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000423c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000004240: 8c7e137e
	s_and_b32 s4, s7, s3                                       // 000000004244: 8b040307
	s_wait_alu depctr_sa_sdst(0)                               // 000000004248: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 00000000424c: be932004
	s_cbranch_execz 46                                         // 000000004250: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x280c>
	v_add_co_u32 v4, s4, v59, s26                              // 000000004254: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000425c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000004260: d5207c05 00103680
	v_bfe_u32 v6, v96, 16, 1                                   // 000000004268: d6100006 02052160
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004270: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000004274: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000427c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004280: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v96                             // 000000004288: 3812c0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004290: bf8701a3
	v_add_co_u32 v4, s4, s38, v4                               // 000000004294: d7000404 02020826
	s_wait_alu depctr_va_sdst(0)                               // 00000000429c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s4                  // 0000000042a0: d5207c05 00120a27
	s_wait_kmcnt 0x0                                           // 0000000042a8: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 0000000042ac: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000042b4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 0000000042b8: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000042c0: 3e080881
	v_add3_u32 v6, v6, v96, 0x7fff                             // 0000000042c4: d6550006 03fec106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000042d0: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000042d4: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000042dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000042e0: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v96, v96                               // 0000000042e8: d4180004 0202c160
	s_wait_alu depctr_va_sdst(0)                               // 0000000042f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042f4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000042f8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004300: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000430c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000004310: 8c7e137e
	s_and_b32 s4, s6, s3                                       // 000000004314: 8b040306
	s_wait_alu depctr_sa_sdst(0)                               // 000000004318: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 00000000431c: be932004
	s_cbranch_execz 46                                         // 000000004320: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x28dc>
	v_add_co_u32 v4, s4, v59, s26                              // 000000004324: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000432c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000004330: d5207c05 00103680
	v_bfe_u32 v6, v95, 16, 1                                   // 000000004338: d6100006 0205215f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004340: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000004344: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000434c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004350: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v95                             // 000000004358: 3812beff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004360: bf8701a3
	v_add_co_u32 v4, s4, s36, v4                               // 000000004364: d7000404 02020824
	s_wait_alu depctr_va_sdst(0)                               // 00000000436c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s4                  // 000000004370: d5207c05 00120a25
	s_wait_kmcnt 0x0                                           // 000000004378: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 00000000437c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004384: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004388: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004390: 3e080881
	v_add3_u32 v6, v6, v95, 0x7fff                             // 000000004394: d6550006 03febf06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000043a0: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 0000000043a4: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000043ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 0000000043b0: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v95, v95                               // 0000000043b8: d4180004 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 0000000043c0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043c4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 0000000043c8: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000043d0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000043e0: 8c7e137e
	s_and_b32 s4, s5, s3                                       // 0000000043e4: 8b040305
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043e8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000043ec: be932004
	s_cbranch_execz 46                                         // 0000000043f0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x29ac>
	v_add_co_u32 v4, s4, v59, s26                              // 0000000043f4: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000043fc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 000000004400: d5207c05 00103680
	v_bfe_u32 v6, v94, 16, 1                                   // 000000004408: d6100006 0205215e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004410: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 000000004414: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000441c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 000000004420: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v94                             // 000000004428: 3812bcff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004430: bf8701a3
	v_add_co_u32 v4, s4, s34, v4                               // 000000004434: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 00000000443c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000004440: d5207c05 00120a23
	s_wait_kmcnt 0x0                                           // 000000004448: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 00000000444c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004454: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004458: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004460: 3e080881
	v_add3_u32 v6, v6, v94, 0x7fff                             // 000000004464: d6550006 03febd06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004470: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004474: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000447c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000004480: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v94, v94                               // 000000004488: d4180004 0202bd5e
	s_wait_alu depctr_va_sdst(0)                               // 000000004490: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004494: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004498: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 0000000044a0: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000044b0: 8c7e137e
	s_and_b32 s4, s1, s3                                       // 0000000044b4: 8b040301
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044b8: bf88ff9e
	s_and_saveexec_b32 s19, s4                                 // 0000000044bc: be932004
	s_cbranch_execz 46                                         // 0000000044c0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2a7c>
	v_add_co_u32 v4, s4, v59, s26                              // 0000000044c4: d7000404 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000044cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s4                   // 0000000044d0: d5207c05 00103680
	v_bfe_u32 v6, v93, 16, 1                                   // 0000000044d8: d6100006 0205215d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000044e0: bf8701a3
	v_add_co_u32 v4, s4, v4, v58                               // 0000000044e4: d7000404 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000044ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s4                    // 0000000044f0: d5207c05 00120a80
	v_or_b32_e32 v9, 0x400000, v93                             // 0000000044f8: 3812baff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004500: bf8701a3
	v_add_co_u32 v4, s4, s24, v4                               // 000000004504: d7000404 02020818
	s_wait_alu depctr_va_sdst(0)                               // 00000000450c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s4                  // 000000004510: d5207c05 00120a19
	s_wait_kmcnt 0x0                                           // 000000004518: bfc70000
	v_add_co_u32 v7, s4, s20, v0                               // 00000000451c: d7000407 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004524: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s4                  // 000000004528: d5207c08 00120215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004530: 3e080881
	v_add3_u32 v6, v6, v93, 0x7fff                             // 000000004534: d6550006 03febb06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004540: bf8701a2
	v_add_co_u32 v4, s4, v7, v4                                // 000000004544: d7000404 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000454c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s4                   // 000000004550: d5207c05 00120b08
	v_cmp_u_f32_e64 s4, v93, v93                               // 000000004558: d4180004 0202bb5d
	s_wait_alu depctr_va_sdst(0)                               // 000000004560: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004564: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s4                           // 000000004568: d5010006 00121306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004570: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000457c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000004580: 8c7e137e
	s_and_b32 s3, s0, s3                                       // 000000004584: 8b030300
	s_wait_alu depctr_sa_sdst(0)                               // 000000004588: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000458c: be842003
	s_cbranch_execz 46                                         // 000000004590: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2b4c>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004594: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000459c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 0000000045a0: d5207c05 000c3680
	v_bfe_u32 v6, v92, 16, 1                                   // 0000000045a8: d6100006 0205215c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045b0: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 0000000045b4: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000045bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000045c0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v92                             // 0000000045c8: 3812b8ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045d0: bf8701a3
	v_add_co_u32 v4, s3, s22, v4                               // 0000000045d4: d7000304 02020816
	s_wait_alu depctr_va_sdst(0)                               // 0000000045dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s3                  // 0000000045e0: d5207c05 000e0a17
	s_wait_kmcnt 0x0                                           // 0000000045e8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000045ec: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 0000000045f8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004600: 3e080881
	v_add3_u32 v6, v6, v92, 0x7fff                             // 000000004604: d6550006 03feb906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004610: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004614: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000461c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004620: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v92, v92                               // 000000004628: d4180003 0202b95c
	s_wait_alu depctr_va_sdst(0)                               // 000000004630: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004634: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004638: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:32          // 000000004640: ee09407c 03000000 00002004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000464c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004650: 8c7e047e
	s_and_b32 s3, s17, s2                                      // 000000004654: 8b030211
	s_wait_alu depctr_sa_sdst(0)                               // 000000004658: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000465c: be842003
	s_cbranch_execz 40                                         // 000000004660: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2c04>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004664: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000466c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004670: d5207c05 000c3680
	v_bfe_u32 v6, v91, 16, 1                                   // 000000004678: d6100006 0205215b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004680: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004684: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000468c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004690: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 000000004698: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 00000000469c: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000046a8: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000046b0: 3e080881
	v_add3_u32 v6, v6, v91, 0x7fff                             // 0000000046b4: d6550006 03feb706 00007fff
	v_or_b32_e32 v9, 0x400000, v91                             // 0000000046c0: 3812b6ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 0000000046c8: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 0000000046cc: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000046d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000046d8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v91, v91                               // 0000000046e0: d4180003 0202b75b
	s_wait_alu depctr_va_sdst(0)                               // 0000000046e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000046ec: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000046f0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000046f8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004704: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004708: 8c7e047e
	s_and_b32 s3, s18, s2                                      // 00000000470c: 8b030212
	s_wait_alu depctr_sa_sdst(0)                               // 000000004710: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004714: be842003
	s_cbranch_execz 46                                         // 000000004718: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2cd4>
	v_add_co_u32 v4, s3, v59, s26                              // 00000000471c: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004724: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004728: d5207c05 000c3680
	v_bfe_u32 v6, v90, 16, 1                                   // 000000004730: d6100006 0205215a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004738: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 00000000473c: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004744: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004748: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v90                             // 000000004750: 3812b4ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004758: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 00000000475c: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000004764: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 000000004768: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 000000004770: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004774: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000477c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004780: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004788: 3e080881
	v_add3_u32 v6, v6, v90, 0x7fff                             // 00000000478c: d6550006 03feb506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004798: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000479c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000047a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 0000000047a8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v90, v90                               // 0000000047b0: d4180003 0202b55a
	s_wait_alu depctr_va_sdst(0)                               // 0000000047b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000047bc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 0000000047c0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000047c8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000047d8: 8c7e047e
	s_and_b32 s3, s16, s2                                      // 0000000047dc: 8b030210
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047e0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000047e4: be842003
	s_cbranch_execz 46                                         // 0000000047e8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2da4>
	v_add_co_u32 v4, s3, v59, s26                              // 0000000047ec: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000047f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 0000000047f8: d5207c05 000c3680
	v_bfe_u32 v6, v89, 16, 1                                   // 000000004800: d6100006 02052159
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004808: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 00000000480c: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004814: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004818: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v89                             // 000000004820: 3812b2ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004828: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 00000000482c: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000004834: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 000000004838: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 000000004840: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004844: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000484c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004850: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004858: 3e080881
	v_add3_u32 v6, v6, v89, 0x7fff                             // 00000000485c: d6550006 03feb306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004868: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000486c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004874: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004878: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v89, v89                               // 000000004880: d4180003 0202b359
	s_wait_alu depctr_va_sdst(0)                               // 000000004888: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000488c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004890: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004898: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000048a8: 8c7e047e
	s_and_b32 s3, s15, s2                                      // 0000000048ac: 8b03020f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048b0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000048b4: be842003
	s_cbranch_execz 46                                         // 0000000048b8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2e74>
	v_add_co_u32 v4, s3, v59, s26                              // 0000000048bc: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000048c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 0000000048c8: d5207c05 000c3680
	v_bfe_u32 v6, v88, 16, 1                                   // 0000000048d0: d6100006 02052158
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048d8: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 0000000048dc: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000048e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000048e8: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v88                             // 0000000048f0: 3812b0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000048f8: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 0000000048fc: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004904: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 000000004908: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 000000004910: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004914: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000491c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004920: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004928: 3e080881
	v_add3_u32 v6, v6, v88, 0x7fff                             // 00000000492c: d6550006 03feb106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004938: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 00000000493c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004944: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004948: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v88, v88                               // 000000004950: d4180003 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 000000004958: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000495c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004960: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004968: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004974: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004978: 8c7e047e
	s_and_b32 s3, s14, s2                                      // 00000000497c: 8b03020e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004980: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004984: be842003
	s_cbranch_execz 46                                         // 000000004988: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x2f44>
	v_add_co_u32 v4, s3, v59, s26                              // 00000000498c: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004994: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004998: d5207c05 000c3680
	v_bfe_u32 v6, v87, 16, 1                                   // 0000000049a0: d6100006 02052157
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049a8: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 0000000049ac: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000049b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000049b8: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v87                             // 0000000049c0: 3812aeff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000049c8: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 0000000049cc: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000049d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 0000000049d8: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 0000000049e0: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 0000000049e4: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000049ec: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 0000000049f0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000049f8: 3e080881
	v_add3_u32 v6, v6, v87, 0x7fff                             // 0000000049fc: d6550006 03feaf06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a08: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004a0c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004a14: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004a18: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v87, v87                               // 000000004a20: d4180003 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 000000004a28: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a2c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004a30: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004a38: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a44: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a48: 8c7e047e
	s_and_b32 s3, s13, s2                                      // 000000004a4c: 8b03020d
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a50: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004a54: be842003
	s_cbranch_execz 46                                         // 000000004a58: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3014>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004a5c: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004a64: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004a68: d5207c05 000c3680
	v_bfe_u32 v6, v86, 16, 1                                   // 000000004a70: d6100006 02052156
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a78: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004a7c: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004a84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004a88: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v86                             // 000000004a90: 3812acff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004a98: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 000000004a9c: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000004aa4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 000000004aa8: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 000000004ab0: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004ab4: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004abc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004ac0: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004ac8: 3e080881
	v_add3_u32 v6, v6, v86, 0x7fff                             // 000000004acc: d6550006 03fead06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ad8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004adc: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004ae4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004ae8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v86, v86                               // 000000004af0: d4180003 0202ad56
	s_wait_alu depctr_va_sdst(0)                               // 000000004af8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004afc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004b00: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004b08: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b18: 8c7e047e
	s_and_b32 s3, s11, s2                                      // 000000004b1c: 8b03020b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b20: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004b24: be842003
	s_cbranch_execz 46                                         // 000000004b28: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x30e4>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004b2c: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004b34: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004b38: d5207c05 000c3680
	v_bfe_u32 v6, v85, 16, 1                                   // 000000004b40: d6100006 02052155
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b48: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004b4c: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004b54: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004b58: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v85                             // 000000004b60: 3812aaff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004b68: bf8701a3
	v_add_co_u32 v4, s3, s24, v4                               // 000000004b6c: d7000304 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000004b74: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s3                  // 000000004b78: d5207c05 000e0a19
	s_wait_kmcnt 0x0                                           // 000000004b80: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004b84: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004b8c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004b90: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004b98: 3e080881
	v_add3_u32 v6, v6, v85, 0x7fff                             // 000000004b9c: d6550006 03feab06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ba8: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004bac: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004bb4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004bb8: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v85, v85                               // 000000004bc0: d4180003 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 000000004bc8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004bcc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004bd0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004bd8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004be4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004be8: 8c7e047e
	s_and_b32 s3, s10, s2                                      // 000000004bec: 8b03020a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bf0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004bf4: be842003
	s_cbranch_execz 46                                         // 000000004bf8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x31b4>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004bfc: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004c04: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004c08: d5207c05 000c3680
	v_bfe_u32 v6, v84, 16, 1                                   // 000000004c10: d6100006 02052154
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c18: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004c1c: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004c24: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004c28: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v84                             // 000000004c30: 3812a8ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c38: bf8701a3
	v_add_co_u32 v4, s3, s22, v4                               // 000000004c3c: d7000304 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000004c44: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s3                  // 000000004c48: d5207c05 000e0a17
	s_wait_kmcnt 0x0                                           // 000000004c50: bfc70000
	v_add_co_u32 v7, s3, s20, v2                               // 000000004c54: d7000307 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000004c5c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s3                  // 000000004c60: d5207c08 000e0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004c68: 3e080881
	v_add3_u32 v6, v6, v84, 0x7fff                             // 000000004c6c: d6550006 03fea906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004c78: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004c7c: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004c84: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004c88: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v84, v84                               // 000000004c90: d4180003 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 000000004c98: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c9c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004ca0: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004ca8: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004cb8: 8c7e047e
	s_and_b32 s3, s12, s2                                      // 000000004cbc: 8b03020c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cc0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004cc4: be842003
	s_cbranch_execz 40                                         // 000000004cc8: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x326c>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004ccc: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004cd4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004cd8: d5207c05 000c3680
	v_bfe_u32 v6, v83, 16, 1                                   // 000000004ce0: d6100006 02052153
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ce8: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004cec: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004cf4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004cf8: d5207c05 000e0a80
	s_wait_kmcnt 0x0                                           // 000000004d00: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004d04: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004d0c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004d10: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004d18: 3e080881
	v_add3_u32 v6, v6, v83, 0x7fff                             // 000000004d1c: d6550006 03fea706 00007fff
	v_or_b32_e32 v9, 0x400000, v83                             // 000000004d28: 3812a6ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000004d30: bf870223
	v_add_co_u32 v4, s3, v7, v4                                // 000000004d34: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004d3c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004d40: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v83, v83                               // 000000004d48: d4180003 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000004d50: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d54: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004d58: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004d60: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004d70: 8c7e047e
	s_and_b32 s3, s9, s2                                       // 000000004d74: 8b030209
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d78: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004d7c: be842003
	s_cbranch_execz 46                                         // 000000004d80: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x333c>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004d84: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004d8c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004d90: d5207c05 000c3680
	v_bfe_u32 v6, v82, 16, 1                                   // 000000004d98: d6100006 02052152
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004da0: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004da4: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004dac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004db0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v82                             // 000000004db8: 3812a4ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004dc0: bf8701a3
	v_add_co_u32 v4, s3, s30, v4                               // 000000004dc4: d7000304 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000004dcc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s3                  // 000000004dd0: d5207c05 000e0a1f
	s_wait_kmcnt 0x0                                           // 000000004dd8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004ddc: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004de4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004de8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004df0: 3e080881
	v_add3_u32 v6, v6, v82, 0x7fff                             // 000000004df4: d6550006 03fea506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e00: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004e04: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004e0c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004e10: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v82, v82                               // 000000004e18: d4180003 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000004e20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004e24: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004e28: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004e30: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004e40: 8c7e047e
	s_and_b32 s3, s8, s2                                       // 000000004e44: 8b030208
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e48: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004e4c: be842003
	s_cbranch_execz 46                                         // 000000004e50: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x340c>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004e54: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004e5c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004e60: d5207c05 000c3680
	v_bfe_u32 v6, v81, 16, 1                                   // 000000004e68: d6100006 02052151
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e70: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004e74: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004e7c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004e80: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v81                             // 000000004e88: 3812a2ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004e90: bf8701a3
	v_add_co_u32 v4, s3, s40, v4                               // 000000004e94: d7000304 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000004e9c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s3                  // 000000004ea0: d5207c05 000e0a29
	s_wait_kmcnt 0x0                                           // 000000004ea8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004eac: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004eb4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004eb8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004ec0: 3e080881
	v_add3_u32 v6, v6, v81, 0x7fff                             // 000000004ec4: d6550006 03fea306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004ed0: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004ed4: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004edc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004ee0: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v81, v81                               // 000000004ee8: d4180003 0202a351
	s_wait_alu depctr_va_sdst(0)                               // 000000004ef0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ef4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004ef8: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004f00: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004f10: 8c7e047e
	s_and_b32 s3, s7, s2                                       // 000000004f14: 8b030207
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f18: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004f1c: be842003
	s_cbranch_execz 46                                         // 000000004f20: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x34dc>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004f24: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004f2c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000004f30: d5207c05 000c3680
	v_bfe_u32 v6, v80, 16, 1                                   // 000000004f38: d6100006 02052150
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f40: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000004f44: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000004f4c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000004f50: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v80                             // 000000004f58: 3812a0ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004f60: bf8701a3
	v_add_co_u32 v4, s3, s38, v4                               // 000000004f64: d7000304 02020826
	s_wait_alu depctr_va_sdst(0)                               // 000000004f6c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s3                  // 000000004f70: d5207c05 000e0a27
	s_wait_kmcnt 0x0                                           // 000000004f78: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 000000004f7c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000004f84: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000004f88: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000004f90: 3e080881
	v_add3_u32 v6, v6, v80, 0x7fff                             // 000000004f94: d6550006 03fea106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000004fa0: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000004fa4: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000004fac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000004fb0: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v80, v80                               // 000000004fb8: d4180003 0202a150
	s_wait_alu depctr_va_sdst(0)                               // 000000004fc0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004fc4: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000004fc8: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000004fd0: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fdc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004fe0: 8c7e047e
	s_and_b32 s3, s6, s2                                       // 000000004fe4: 8b030206
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fe8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004fec: be842003
	s_cbranch_execz 46                                         // 000000004ff0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x35ac>
	v_add_co_u32 v4, s3, v59, s26                              // 000000004ff4: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000004ffc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 000000005000: d5207c05 000c3680
	v_bfe_u32 v6, v79, 16, 1                                   // 000000005008: d6100006 0205214f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005010: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 000000005014: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000501c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 000000005020: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v79                             // 000000005028: 38129eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005030: bf8701a3
	v_add_co_u32 v4, s3, s36, v4                               // 000000005034: d7000304 02020824
	s_wait_alu depctr_va_sdst(0)                               // 00000000503c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s3                  // 000000005040: d5207c05 000e0a25
	s_wait_kmcnt 0x0                                           // 000000005048: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 00000000504c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005054: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000005058: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005060: 3e080881
	v_add3_u32 v6, v6, v79, 0x7fff                             // 000000005064: d6550006 03fe9f06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005070: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000005074: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000507c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000005080: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v79, v79                               // 000000005088: d4180003 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 000000005090: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005094: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005098: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 0000000050a0: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000050b0: 8c7e047e
	s_and_b32 s3, s5, s2                                       // 0000000050b4: 8b030205
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050b8: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000050bc: be842003
	s_cbranch_execz 46                                         // 0000000050c0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x367c>
	v_add_co_u32 v4, s3, v59, s26                              // 0000000050c4: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000050cc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 0000000050d0: d5207c05 000c3680
	v_bfe_u32 v6, v78, 16, 1                                   // 0000000050d8: d6100006 0205214e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000050e0: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 0000000050e4: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000050ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000050f0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v78                             // 0000000050f8: 38129cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005100: bf8701a3
	v_add_co_u32 v4, s3, s34, v4                               // 000000005104: d7000304 02020822
	s_wait_alu depctr_va_sdst(0)                               // 00000000510c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s3                  // 000000005110: d5207c05 000e0a23
	s_wait_kmcnt 0x0                                           // 000000005118: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 00000000511c: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005124: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 000000005128: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005130: 3e080881
	v_add3_u32 v6, v6, v78, 0x7fff                             // 000000005134: d6550006 03fe9d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005140: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000005144: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000514c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000005150: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v78, v78                               // 000000005158: d4180003 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000005160: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005164: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005168: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000005170: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000517c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000005180: 8c7e047e
	s_and_b32 s3, s1, s2                                       // 000000005184: 8b030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000005188: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 00000000518c: be842003
	s_cbranch_execz 46                                         // 000000005190: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x374c>
	v_add_co_u32 v4, s3, v59, s26                              // 000000005194: d7000304 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000519c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s3                   // 0000000051a0: d5207c05 000c3680
	v_bfe_u32 v6, v77, 16, 1                                   // 0000000051a8: d6100006 0205214d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051b0: bf8701a3
	v_add_co_u32 v4, s3, v4, v58                               // 0000000051b4: d7000304 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000051bc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s3                    // 0000000051c0: d5207c05 000e0a80
	v_or_b32_e32 v9, 0x400000, v77                             // 0000000051c8: 38129aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000051d0: bf8701a3
	v_add_co_u32 v4, s3, s24, v4                               // 0000000051d4: d7000304 02020818
	s_wait_alu depctr_va_sdst(0)                               // 0000000051dc: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s3                  // 0000000051e0: d5207c05 000e0a19
	s_wait_kmcnt 0x0                                           // 0000000051e8: bfc70000
	v_add_co_u32 v7, s3, s20, v0                               // 0000000051ec: d7000307 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000051f4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s3                  // 0000000051f8: d5207c08 000e0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005200: 3e080881
	v_add3_u32 v6, v6, v77, 0x7fff                             // 000000005204: d6550006 03fe9b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005210: bf8701a2
	v_add_co_u32 v4, s3, v7, v4                                // 000000005214: d7000304 02020907
	s_wait_alu depctr_va_sdst(0)                               // 00000000521c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s3                   // 000000005220: d5207c05 000e0b08
	v_cmp_u_f32_e64 s3, v77, v77                               // 000000005228: d4180003 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000005230: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005234: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s3                           // 000000005238: d5010006 000e1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000005240: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000524c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000005250: 8c7e047e
	s_and_b32 s2, s0, s2                                       // 000000005254: 8b020200
	s_wait_alu depctr_sa_sdst(0)                               // 000000005258: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000525c: be832002
	s_cbranch_execz 46                                         // 000000005260: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x381c>
	v_add_co_u32 v4, s2, v59, s26                              // 000000005264: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000526c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005270: d5207c05 00083680
	v_bfe_u32 v6, v76, 16, 1                                   // 000000005278: d6100006 0205214c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005280: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 000000005284: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000528c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005290: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v76                             // 000000005298: 381298ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000052a0: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 0000000052a4: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 0000000052ac: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 0000000052b0: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 0000000052b8: bfc70000
	v_add_co_u32 v7, s2, s20, v0                               // 0000000052bc: d7000207 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000052c4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v1, s2                  // 0000000052c8: d5207c08 000a0215
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000052d0: 3e080881
	v_add3_u32 v6, v6, v76, 0x7fff                             // 0000000052d4: d6550006 03fe9906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000052e0: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000052e4: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000052ec: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000052f0: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v76, v76                               // 0000000052f8: d4180002 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 000000005300: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005304: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005308: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:64          // 000000005310: ee09407c 03000000 00004004
	s_wait_alu depctr_sa_sdst(0)                               // 00000000531c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005320: 8c7e037e
	s_and_b32 s2, s17, vcc_lo                                  // 000000005324: 8b026a11
	s_wait_alu depctr_sa_sdst(0)                               // 000000005328: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 00000000532c: be832002
	s_cbranch_execz 40                                         // 000000005330: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x38d4>
	v_add_co_u32 v4, s2, v59, s26                              // 000000005334: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 00000000533c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005340: d5207c05 00083680
	v_bfe_u32 v6, v75, 16, 1                                   // 000000005348: d6100006 0205214b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005350: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 000000005354: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 00000000535c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005360: d5207c05 000a0a80
	s_wait_kmcnt 0x0                                           // 000000005368: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 00000000536c: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 000000005374: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005378: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005380: 3e080881
	v_add3_u32 v6, v6, v75, 0x7fff                             // 000000005384: d6550006 03fe9706 00007fff
	v_or_b32_e32 v9, 0x400000, v75                             // 000000005390: 381296ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000005398: bf870223
	v_add_co_u32 v4, s2, v7, v4                                // 00000000539c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000053a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000053a8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v75, v75                               // 0000000053b0: d4180002 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 0000000053b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000053bc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000053c0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000053c8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000053d8: 8c7e037e
	s_and_b32 s2, s18, vcc_lo                                  // 0000000053dc: 8b026a12
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053e0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000053e4: be832002
	s_cbranch_execz 46                                         // 0000000053e8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x39a4>
	v_add_co_u32 v4, s2, v59, s26                              // 0000000053ec: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000053f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 0000000053f8: d5207c05 00083680
	v_bfe_u32 v6, v74, 16, 1                                   // 000000005400: d6100006 0205214a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005408: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 00000000540c: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000005414: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005418: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v74                             // 000000005420: 381294ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005428: bf8701a3
	v_add_co_u32 v4, s2, s30, v4                               // 00000000542c: d7000204 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000005434: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s2                  // 000000005438: d5207c05 000a0a1f
	s_wait_kmcnt 0x0                                           // 000000005440: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005444: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000544c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005450: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005458: 3e080881
	v_add3_u32 v6, v6, v74, 0x7fff                             // 00000000545c: d6550006 03fe9506 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005468: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000546c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005474: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005478: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v74, v74                               // 000000005480: d4180002 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 000000005488: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000548c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005490: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005498: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000054a8: 8c7e037e
	s_and_b32 s2, s16, vcc_lo                                  // 0000000054ac: 8b026a10
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054b0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000054b4: be832002
	s_cbranch_execz 46                                         // 0000000054b8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3a74>
	v_add_co_u32 v4, s2, v59, s26                              // 0000000054bc: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000054c4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 0000000054c8: d5207c05 00083680
	v_bfe_u32 v6, v73, 16, 1                                   // 0000000054d0: d6100006 02052149
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000054d8: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 0000000054dc: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000054e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000054e8: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v73                             // 0000000054f0: 381292ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000054f8: bf8701a3
	v_add_co_u32 v4, s2, s40, v4                               // 0000000054fc: d7000204 02020828
	s_wait_alu depctr_va_sdst(0)                               // 000000005504: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s41, v5, s2                  // 000000005508: d5207c05 000a0a29
	s_wait_kmcnt 0x0                                           // 000000005510: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005514: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000551c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005520: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005528: 3e080881
	v_add3_u32 v6, v6, v73, 0x7fff                             // 00000000552c: d6550006 03fe9306 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005538: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000553c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005544: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005548: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v73, v73                               // 000000005550: d4180002 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000005558: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000555c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005560: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005568: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005574: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005578: 8c7e037e
	s_and_b32 s2, s15, vcc_lo                                  // 00000000557c: 8b026a0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005580: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005584: be832002
	s_cbranch_execz 46                                         // 000000005588: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3b44>
	v_add_co_u32 v4, s2, v59, s26                              // 00000000558c: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005594: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005598: d5207c05 00083680
	v_bfe_u32 v6, v72, 16, 1                                   // 0000000055a0: d6100006 02052148
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055a8: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 0000000055ac: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000055b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000055b8: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v72                             // 0000000055c0: 381290ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000055c8: bf8701a3
	v_add_co_u32 v4, s2, s38, v4                               // 0000000055cc: d7000204 02020826
	s_wait_alu depctr_va_sdst(0)                               // 0000000055d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s39, v5, s2                  // 0000000055d8: d5207c05 000a0a27
	s_wait_kmcnt 0x0                                           // 0000000055e0: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000055e4: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000055ec: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000055f0: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000055f8: 3e080881
	v_add3_u32 v6, v6, v72, 0x7fff                             // 0000000055fc: d6550006 03fe9106 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005608: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000560c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005614: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005618: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v72, v72                               // 000000005620: d4180002 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000005628: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000562c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005630: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005638: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005644: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005648: 8c7e037e
	s_and_b32 s2, s14, vcc_lo                                  // 00000000564c: 8b026a0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000005650: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005654: be832002
	s_cbranch_execz 46                                         // 000000005658: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3c14>
	v_add_co_u32 v4, s2, v59, s26                              // 00000000565c: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005664: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005668: d5207c05 00083680
	v_bfe_u32 v6, v71, 16, 1                                   // 000000005670: d6100006 02052147
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005678: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 00000000567c: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000005684: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005688: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v71                             // 000000005690: 38128eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005698: bf8701a3
	v_add_co_u32 v4, s2, s36, v4                               // 00000000569c: d7000204 02020824
	s_wait_alu depctr_va_sdst(0)                               // 0000000056a4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s37, v5, s2                  // 0000000056a8: d5207c05 000a0a25
	s_wait_kmcnt 0x0                                           // 0000000056b0: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 0000000056b4: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 0000000056bc: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 0000000056c0: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 0000000056c8: 3e080881
	v_add3_u32 v6, v6, v71, 0x7fff                             // 0000000056cc: d6550006 03fe8f06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000056d8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000056dc: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000056e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000056e8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v71, v71                               // 0000000056f0: d4180002 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 0000000056f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000056fc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 000000005700: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 000000005708: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 000000005714: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005718: 8c7e037e
	s_and_b32 s2, s13, vcc_lo                                  // 00000000571c: 8b026a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005720: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005724: be832002
	s_cbranch_execz 46                                         // 000000005728: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3ce4>
	v_add_co_u32 v4, s2, v59, s26                              // 00000000572c: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005734: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005738: d5207c05 00083680
	v_bfe_u32 v6, v70, 16, 1                                   // 000000005740: d6100006 02052146
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005748: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 00000000574c: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000005754: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005758: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v70                             // 000000005760: 38128cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005768: bf8701a3
	v_add_co_u32 v4, s2, s34, v4                               // 00000000576c: d7000204 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000005774: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s2                  // 000000005778: d5207c05 000a0a23
	s_wait_kmcnt 0x0                                           // 000000005780: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005784: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000578c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005790: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005798: 3e080881
	v_add3_u32 v6, v6, v70, 0x7fff                             // 00000000579c: d6550006 03fe8d06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 0000000057a8: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 0000000057ac: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 0000000057b4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 0000000057b8: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v70, v70                               // 0000000057c0: d4180002 02028d46
	s_wait_alu depctr_va_sdst(0)                               // 0000000057c8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000057cc: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000057d0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000057d8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000057e8: 8c7e037e
	s_and_b32 s2, s11, vcc_lo                                  // 0000000057ec: 8b026a0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057f0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000057f4: be832002
	s_cbranch_execz 46                                         // 0000000057f8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3db4>
	v_add_co_u32 v4, s2, v59, s26                              // 0000000057fc: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005804: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 000000005808: d5207c05 00083680
	v_bfe_u32 v6, v69, 16, 1                                   // 000000005810: d6100006 02052145
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005818: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 00000000581c: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 000000005824: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 000000005828: d5207c05 000a0a80
	v_or_b32_e32 v9, 0x400000, v69                             // 000000005830: 38128aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005838: bf8701a3
	v_add_co_u32 v4, s2, s24, v4                               // 00000000583c: d7000204 02020818
	s_wait_alu depctr_va_sdst(0)                               // 000000005844: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s25, v5, s2                  // 000000005848: d5207c05 000a0a19
	s_wait_kmcnt 0x0                                           // 000000005850: bfc70000
	v_add_co_u32 v7, s2, s20, v2                               // 000000005854: d7000207 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000585c: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v3, s2                  // 000000005860: d5207c08 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005868: 3e080881
	v_add3_u32 v6, v6, v69, 0x7fff                             // 00000000586c: d6550006 03fe8b06 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005878: bf8701a2
	v_add_co_u32 v4, s2, v7, v4                                // 00000000587c: d7000204 02020907
	s_wait_alu depctr_va_sdst(0)                               // 000000005884: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v5, s2                   // 000000005888: d5207c05 000a0b08
	v_cmp_u_f32_e64 s2, v69, v69                               // 000000005890: d4180002 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000005898: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000589c: bf870001
	v_cndmask_b32_e64 v6, v6, v9, s2                           // 0000000058a0: d5010006 000a1306
	global_store_d16_hi_b16 v[4:5], v6, off offset:96          // 0000000058a8: ee09407c 03000000 00006004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000058b8: 8c7e037e
	s_and_b32 s2, s10, vcc_lo                                  // 0000000058bc: 8b026a0a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058c0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000058c4: be832002
	s_cbranch_execz 46                                         // 0000000058c8: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3e84>
	v_add_co_u32 v4, s2, v59, s26                              // 0000000058cc: d7000204 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000058d4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, s27, s2                   // 0000000058d8: d5207c05 00083680
	v_bfe_u32 v6, v68, 16, 1                                   // 0000000058e0: d6100006 02052144
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000058e8: bf8701a3
	v_add_co_u32 v4, s2, v4, v58                               // 0000000058ec: d7000204 02027504
	s_wait_alu depctr_va_sdst(0)                               // 0000000058f4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, 0, v5, s2                    // 0000000058f8: d5207c05 000a0a80
	v_or_b32_e32 v7, 0x400000, v68                             // 000000005900: 380e88ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005908: bf8701a3
	v_add_co_u32 v4, s2, s22, v4                               // 00000000590c: d7000204 02020816
	s_wait_alu depctr_va_sdst(0)                               // 000000005914: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s23, v5, s2                  // 000000005918: d5207c05 000a0a17
	s_wait_kmcnt 0x0                                           // 000000005920: bfc70000
	v_add_co_u32 v2, s2, s20, v2                               // 000000005924: d7000202 02020414
	s_wait_alu depctr_va_sdst(0)                               // 00000000592c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s21, v3, s2                  // 000000005930: d5207c03 000a0615
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]                        // 000000005938: 3e080881
	v_add3_u32 v6, v6, v68, 0x7fff                             // 00000000593c: d6550006 03fe8906 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005948: bf8701a2
	v_add_co_u32 v2, s2, v2, v4                                // 00000000594c: d7000202 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000005954: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v3, v5, s2                   // 000000005958: d5207c03 000a0b03
	v_cmp_u_f32_e64 s2, v68, v68                               // 000000005960: d4180002 02028944
	s_wait_alu depctr_va_sdst(0)                               // 000000005968: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000596c: bf870001
	v_cndmask_b32_e64 v4, v6, v7, s2                           // 000000005970: d5010004 000a0f06
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005978: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005984: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005988: 8c7e037e
	s_and_b32 s2, s12, vcc_lo                                  // 00000000598c: 8b026a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000005990: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005994: be832002
	s_cbranch_execz 40                                         // 000000005998: bfa50028 <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x3f3c>
	v_add_co_u32 v2, s2, v59, s26                              // 00000000599c: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 0000000059a4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 0000000059a8: d5207c03 00083680
	v_bfe_u32 v4, v67, 16, 1                                   // 0000000059b0: d6100004 02052143
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000059b8: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 0000000059bc: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 0000000059c4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 0000000059c8: d5207c03 000a0680
	s_wait_kmcnt 0x0                                           // 0000000059d0: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 0000000059d4: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 0000000059dc: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 0000000059e0: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000059e8: 3e040481
	v_add3_u32 v4, v4, v67, 0x7fff                             // 0000000059ec: d6550004 03fe8704 00007fff
	v_or_b32_e32 v7, 0x400000, v67                             // 0000000059f8: 380e86ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000005a00: bf870223
	v_add_co_u32 v2, s2, v5, v2                                // 000000005a04: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005a0c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005a10: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v67, v67                               // 000000005a18: d4180002 02028743
	s_wait_alu depctr_va_sdst(0)                               // 000000005a20: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005a24: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005a28: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005a30: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005a40: 8c7e037e
	s_and_b32 s2, s9, vcc_lo                                   // 000000005a44: 8b026a09
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a48: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005a4c: be832002
	s_cbranch_execz 46                                         // 000000005a50: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x400c>
	v_add_co_u32 v2, s2, v59, s26                              // 000000005a54: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005a5c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 000000005a60: d5207c03 00083680
	v_bfe_u32 v4, v66, 16, 1                                   // 000000005a68: d6100004 02052142
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005a70: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 000000005a74: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005a7c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005a80: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v66                             // 000000005a88: 380e84ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005a90: bf8701a3
	v_add_co_u32 v2, s2, s30, v2                               // 000000005a94: d7000202 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000005a9c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s2                  // 000000005aa0: d5207c03 000a061f
	s_wait_kmcnt 0x0                                           // 000000005aa8: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005aac: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005ab4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005ab8: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005ac0: 3e040481
	v_add3_u32 v4, v4, v66, 0x7fff                             // 000000005ac4: d6550004 03fe8504 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ad0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005ad4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005adc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005ae0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v66, v66                               // 000000005ae8: d4180002 02028542
	s_wait_alu depctr_va_sdst(0)                               // 000000005af0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005af4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005af8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005b00: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b0c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005b10: 8c7e037e
	s_and_b32 s2, s8, vcc_lo                                   // 000000005b14: 8b026a08
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b18: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005b1c: be832002
	s_cbranch_execz 46                                         // 000000005b20: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x40dc>
	v_add_co_u32 v2, s2, v59, s26                              // 000000005b24: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005b2c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 000000005b30: d5207c03 00083680
	v_bfe_u32 v4, v65, 16, 1                                   // 000000005b38: d6100004 02052141
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b40: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 000000005b44: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005b4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005b50: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v65                             // 000000005b58: 380e82ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005b60: bf8701a3
	v_add_co_u32 v2, s2, s40, v2                               // 000000005b64: d7000202 02020428
	s_wait_alu depctr_va_sdst(0)                               // 000000005b6c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s41, v3, s2                  // 000000005b70: d5207c03 000a0629
	s_wait_kmcnt 0x0                                           // 000000005b78: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005b7c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005b84: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005b88: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005b90: 3e040481
	v_add3_u32 v4, v4, v65, 0x7fff                             // 000000005b94: d6550004 03fe8304 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ba0: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005ba4: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005bac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005bb0: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v65, v65                               // 000000005bb8: d4180002 02028341
	s_wait_alu depctr_va_sdst(0)                               // 000000005bc0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005bc4: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005bc8: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005bd0: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bdc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005be0: 8c7e037e
	s_and_b32 s2, s7, vcc_lo                                   // 000000005be4: 8b026a07
	s_wait_alu depctr_sa_sdst(0)                               // 000000005be8: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005bec: be832002
	s_cbranch_execz 46                                         // 000000005bf0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x41ac>
	v_add_co_u32 v2, s2, v59, s26                              // 000000005bf4: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005bfc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 000000005c00: d5207c03 00083680
	v_bfe_u32 v4, v64, 16, 1                                   // 000000005c08: d6100004 02052140
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005c10: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 000000005c14: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005c1c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005c20: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v64                             // 000000005c28: 380e80ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005c30: bf8701a3
	v_add_co_u32 v2, s2, s38, v2                               // 000000005c34: d7000202 02020426
	s_wait_alu depctr_va_sdst(0)                               // 000000005c3c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s39, v3, s2                  // 000000005c40: d5207c03 000a0627
	s_wait_kmcnt 0x0                                           // 000000005c48: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005c4c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005c54: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005c58: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005c60: 3e040481
	v_add3_u32 v4, v4, v64, 0x7fff                             // 000000005c64: d6550004 03fe8104 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005c70: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005c74: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005c7c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005c80: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v64, v64                               // 000000005c88: d4180002 02028140
	s_wait_alu depctr_va_sdst(0)                               // 000000005c90: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005c94: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005c98: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005ca0: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005cb0: 8c7e037e
	s_and_b32 s2, s6, vcc_lo                                   // 000000005cb4: 8b026a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cb8: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005cbc: be832002
	s_cbranch_execz 46                                         // 000000005cc0: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x427c>
	v_add_co_u32 v2, s2, v59, s26                              // 000000005cc4: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005ccc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 000000005cd0: d5207c03 00083680
	v_bfe_u32 v4, v63, 16, 1                                   // 000000005cd8: d6100004 0205213f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ce0: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 000000005ce4: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005cec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005cf0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v63                             // 000000005cf8: 380e7eff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005d00: bf8701a3
	v_add_co_u32 v2, s2, s36, v2                               // 000000005d04: d7000202 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000005d0c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, v3, s2                  // 000000005d10: d5207c03 000a0625
	s_wait_kmcnt 0x0                                           // 000000005d18: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005d1c: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005d24: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005d28: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005d30: 3e040481
	v_add3_u32 v4, v4, v63, 0x7fff                             // 000000005d34: d6550004 03fe7f04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005d40: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005d44: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005d4c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005d50: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v63, v63                               // 000000005d58: d4180002 02027f3f
	s_wait_alu depctr_va_sdst(0)                               // 000000005d60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005d64: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005d68: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005d70: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005d80: 8c7e037e
	s_and_b32 s2, s5, vcc_lo                                   // 000000005d84: 8b026a05
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d88: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000005d8c: be832002
	s_cbranch_execz 46                                         // 000000005d90: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x434c>
	v_add_co_u32 v2, s2, v59, s26                              // 000000005d94: d7000202 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005d9c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s2                   // 000000005da0: d5207c03 00083680
	v_bfe_u32 v4, v62, 16, 1                                   // 000000005da8: d6100004 0205213e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005db0: bf8701a3
	v_add_co_u32 v2, s2, v2, v58                               // 000000005db4: d7000202 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005dbc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000005dc0: d5207c03 000a0680
	v_or_b32_e32 v7, 0x400000, v62                             // 000000005dc8: 380e7cff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005dd0: bf8701a3
	v_add_co_u32 v2, s2, s34, v2                               // 000000005dd4: d7000202 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000005ddc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s2                  // 000000005de0: d5207c03 000a0623
	s_wait_kmcnt 0x0                                           // 000000005de8: bfc70000
	v_add_co_u32 v5, s2, s20, v0                               // 000000005dec: d7000205 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005df4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s2                  // 000000005df8: d5207c06 000a0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005e00: 3e040481
	v_add3_u32 v4, v4, v62, 0x7fff                             // 000000005e04: d6550004 03fe7d04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005e10: bf8701a2
	v_add_co_u32 v2, s2, v5, v2                                // 000000005e14: d7000202 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005e1c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s2                   // 000000005e20: d5207c03 000a0706
	v_cmp_u_f32_e64 s2, v62, v62                               // 000000005e28: d4180002 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000005e30: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005e34: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s2                           // 000000005e38: d5010004 000a0f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005e40: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000005e50: 8c7e037e
	s_and_b32 s1, s1, vcc_lo                                   // 000000005e54: 8b016a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e58: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000005e5c: be822001
	s_cbranch_execz 46                                         // 000000005e60: bfa5002e <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x441c>
	v_add_co_u32 v2, s1, v59, s26                              // 000000005e64: d7000102 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005e6c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s1                   // 000000005e70: d5207c03 00043680
	v_bfe_u32 v4, v61, 16, 1                                   // 000000005e78: d6100004 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005e80: bf8701a3
	v_add_co_u32 v2, s1, v2, v58                               // 000000005e84: d7000102 02027502
	s_wait_alu depctr_va_sdst(0)                               // 000000005e8c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, v3, s1                    // 000000005e90: d5207c03 00060680
	v_or_b32_e32 v7, 0x400000, v61                             // 000000005e98: 380e7aff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ea0: bf8701a3
	v_add_co_u32 v2, s1, s24, v2                               // 000000005ea4: d7000102 02020418
	s_wait_alu depctr_va_sdst(0)                               // 000000005eac: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s25, v3, s1                  // 000000005eb0: d5207c03 00060619
	s_wait_kmcnt 0x0                                           // 000000005eb8: bfc70000
	v_add_co_u32 v5, s1, s20, v0                               // 000000005ebc: d7000105 02020014
	s_wait_alu depctr_va_sdst(0)                               // 000000005ec4: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s21, v1, s1                  // 000000005ec8: d5207c06 00060215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005ed0: 3e040481
	v_add3_u32 v4, v4, v61, 0x7fff                             // 000000005ed4: d6550004 03fe7b04 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005ee0: bf8701a2
	v_add_co_u32 v2, s1, v5, v2                                // 000000005ee4: d7000102 02020505
	s_wait_alu depctr_va_sdst(0)                               // 000000005eec: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v6, v3, s1                   // 000000005ef0: d5207c03 00060706
	v_cmp_u_f32_e64 s1, v61, v61                               // 000000005ef8: d4180001 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000005f00: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005f04: bf870001
	v_cndmask_b32_e64 v4, v4, v7, s1                           // 000000005f08: d5010004 00060f04
	global_store_d16_hi_b16 v[2:3], v4, off offset:96          // 000000005f10: ee09407c 02000000 00006002
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f1c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005f20: 8c7e027e
	s_and_b32 s0, s0, vcc_lo                                   // 000000005f24: 8b006a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f28: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005f2c: be812000
	s_cbranch_execz 43                                         // 000000005f30: bfa5002b <tessera_rocm_scaled_matmul_lds_f0c97b8aa06b45a2+0x44e0>
	v_add_co_u32 v2, s0, v59, s26                              // 000000005f34: d7000002 0200353b
	s_wait_alu depctr_va_sdst(0)                               // 000000005f3c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, 0, s27, s0                   // 000000005f40: d5207c03 00003680
	v_bfe_u32 v4, v60, 16, 1                                   // 000000005f48: d6100004 0205213c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005f50: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v58                           // 000000005f54: d7006a02 02027502
	s_wait_alu depctr_va_vcc(0)                                // 000000005f5c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000005f60: d5207c03 01aa0680
	v_or_b32_e32 v5, 0x400000, v60                             // 000000005f68: 380a78ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005f70: bf8701a3
	v_add_co_u32 v2, vcc_lo, s22, v2                           // 000000005f74: d7006a02 02020416
	s_wait_alu depctr_va_vcc(0)                                // 000000005f7c: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo              // 000000005f80: d5207c03 01aa0617
	s_wait_kmcnt 0x0                                           // 000000005f88: bfc70000
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 000000005f8c: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000005f94: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 000000005f98: d5207c01 01aa0215
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005fa0: 3e040481
	v_add3_u32 v4, v4, v60, 0x7fff                             // 000000005fa4: d6550004 03fe7904 00007fff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000005fb0: bf8701a2
	v_add_co_u32 v0, vcc_lo, v0, v2                            // 000000005fb4: d7006a00 02020500
	s_wait_alu depctr_va_vcc(0)                                // 000000005fbc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo               // 000000005fc0: d5207c01 01aa0701
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000005fc8: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 000000005fcc: bf88ff9d
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 000000005fd0: 02040b04
	global_store_d16_hi_b16 v[0:1], v2, off offset:96          // 000000005fd4: ee09407c 01000000 00006000
	s_nop 0                                                    // 000000005fe0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005fe4: bfb60003
	s_endpgm                                                   // 000000005fe8: bfb00000
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
