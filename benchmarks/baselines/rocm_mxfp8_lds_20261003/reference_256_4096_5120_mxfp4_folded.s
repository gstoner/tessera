
/tmp/tmp7ylpvyut.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_folded_matmul_708d500594ff51c6>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[8:11], s[0:1], 0xc8                          // 000000001b04: f4004200 f80000c8
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b0c: f4002080 f80000d8
	s_mov_b32 s6, ttmp9                                        // 000000001b14: be860075
	s_ashr_i32 s7, ttmp9, 31                                   // 000000001b18: 86079f75
	s_wait_kmcnt 0x0                                           // 000000001b1c: bfc70000
	s_add_nc_u64 s[4:5], s[10:11], 63                          // 000000001b20: a984bf0a
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001b24: bf870499
	s_lshr_b64 s[4:5], s[4:5], 4                               // 000000001b28: 85848404
	s_and_b32 s13, s5, 0xfffffff                               // 000000001b2c: 8b0dff05 0fffffff
	s_and_b32 s12, s4, -4                                      // 000000001b34: 8b0cc404
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001b38: bf870499
	s_or_b64 s[4:5], s[6:7], s[12:13]                          // 000000001b3c: 8c840c06
	s_cmp_lg_u32 s5, 0                                         // 000000001b40: bf078005
	s_mov_b32 s5, 0                                            // 000000001b44: be850080
	s_cbranch_scc0 111                                         // 000000001b48: bfa1006f <tessera_rocm_folded_matmul_708d500594ff51c6+0x208>
	s_cvt_f32_u32 s4, s12                                      // 000000001b4c: be84650c
	s_cvt_f32_u32 s14, s13                                     // 000000001b50: be8e650d
	s_sub_nc_u64 s[16:17], 0, s[12:13]                         // 000000001b54: aa100c80
	s_delay_alu instid0(salu_cycle_2) | instskip(next) | instid1(salu_cycle_3)// 000000001b58: bf87059a
	s_fmac_f32 s4, s14, 0x4f800000                             // 000000001b5c: a384ff0e 4f800000
	v_s_rcp_f32 s4, s4                                         // 000000001b64: d6840004 02010004
	s_delay_alu instid0(trans32_dep_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001b6c: bf870525
	s_mul_f32 s4, s4, 0x5f7ffffc                               // 000000001b70: a204ff04 5f7ffffc
	s_wait_alu depctr_sa_sdst(0)                               // 000000001b78: bf88ff9e
	s_mul_f32 s14, s4, 0x2f800000                              // 000000001b7c: a20eff04 2f800000
	s_delay_alu instid0(salu_cycle_3) | instskip(next) | instid1(salu_cycle_3)// 000000001b84: bf87059b
	s_trunc_f32 s14, s14                                       // 000000001b88: be8e620e
	s_fmac_f32 s4, s14, 0xcf800000                             // 000000001b8c: a384ff0e cf800000
	s_cvt_u32_f32 s15, s14                                     // 000000001b94: be8f670e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001b98: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_3)// 000000001b9c: bf870599
	s_cvt_u32_f32 s14, s4                                      // 000000001ba0: be8e6704
	s_mul_u64 s[18:19], s[16:17], s[14:15]                     // 000000001ba4: aa920e10
	s_delay_alu instid0(salu_cycle_1)                          // 000000001ba8: bf870009
	s_mul_hi_u32 s21, s14, s19                                 // 000000001bac: 9695130e
	s_mul_i32 s20, s14, s19                                    // 000000001bb0: 9614130e
	s_mul_hi_u32 s4, s14, s18                                  // 000000001bb4: 9684120e
	s_mul_i32 s23, s15, s18                                    // 000000001bb8: 9617120f
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bbc: bf88ff9e
	s_add_nc_u64 s[20:21], s[4:5], s[20:21]                    // 000000001bc0: a9941404
	s_mul_hi_u32 s22, s15, s18                                 // 000000001bc4: 9696120f
	s_mul_hi_u32 s24, s15, s19                                 // 000000001bc8: 9698130f
	s_add_co_u32 s4, s20, s23                                  // 000000001bcc: 80041714
	s_add_co_ci_u32 s4, s21, s22                               // 000000001bd0: 82041615
	s_mul_i32 s18, s15, s19                                    // 000000001bd4: 9612130f
	s_add_co_ci_u32 s19, s24, 0                                // 000000001bd8: 82138018
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bdc: bf88ff9e
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]                    // 000000001be0: a9921204
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_1) | instid1(salu_cycle_1)// 000000001be4: bf8704a9
	s_add_co_u32 s14, s14, s18                                 // 000000001be8: 800e120e
	s_add_co_ci_u32 s15, s15, s19                              // 000000001bec: 820f130f
	s_mul_u64 s[16:17], s[16:17], s[14:15]                     // 000000001bf0: aa900e10
	s_delay_alu instid0(salu_cycle_1)                          // 000000001bf4: bf870009
	s_mul_hi_u32 s19, s14, s17                                 // 000000001bf8: 9693110e
	s_mul_i32 s18, s14, s17                                    // 000000001bfc: 9612110e
	s_mul_hi_u32 s4, s14, s16                                  // 000000001c00: 9684100e
	s_mul_i32 s21, s15, s16                                    // 000000001c04: 9615100f
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c08: bf88ff9e
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]                    // 000000001c0c: a9921204
	s_mul_hi_u32 s20, s15, s16                                 // 000000001c10: 9694100f
	s_mul_hi_u32 s22, s15, s17                                 // 000000001c14: 9696110f
	s_add_co_u32 s4, s18, s21                                  // 000000001c18: 80041512
	s_add_co_ci_u32 s4, s19, s20                               // 000000001c1c: 82041413
	s_mul_i32 s16, s15, s17                                    // 000000001c20: 9610110f
	s_add_co_ci_u32 s17, s22, 0                                // 000000001c24: 82118016
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c28: bf88ff9e
	s_add_nc_u64 s[16:17], s[4:5], s[16:17]                    // 000000001c2c: a9901004
	s_delay_alu instid0(salu_cycle_1)                          // 000000001c30: bf870009
	s_add_co_u32 s14, s14, s16                                 // 000000001c34: 800e100e
	s_add_co_ci_u32 s16, s15, s17                              // 000000001c38: 8210110f
	s_mul_hi_u32 s4, s6, s14                                   // 000000001c3c: 96840e06
	s_mul_hi_u32 s17, s7, s14                                  // 000000001c40: 96910e07
	s_mul_i32 s18, s7, s14                                     // 000000001c44: 96120e07
	s_mul_hi_u32 s15, s6, s16                                  // 000000001c48: 968f1006
	s_mul_i32 s14, s6, s16                                     // 000000001c4c: 960e1006
	s_mul_hi_u32 s19, s7, s16                                  // 000000001c50: 96931007
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c54: bf88ff9e
	s_add_nc_u64 s[14:15], s[4:5], s[14:15]                    // 000000001c58: a98e0e04
	s_mul_i32 s16, s7, s16                                     // 000000001c5c: 96101007
	s_add_co_u32 s4, s14, s18                                  // 000000001c60: 8004120e
	s_add_co_ci_u32 s4, s15, s17                               // 000000001c64: 8204110f
	s_add_co_ci_u32 s17, s19, 0                                // 000000001c68: 82118013
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c6c: bf88ff9e
	s_add_nc_u64 s[14:15], s[4:5], s[16:17]                    // 000000001c70: a98e1004
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001c74: bf870499
	s_mul_u64 s[16:17], s[12:13], s[14:15]                     // 000000001c78: aa900e0c
	s_sub_co_u32 s4, s6, s16                                   // 000000001c7c: 80841006
	s_cselect_b32 s16, -1, 0                                   // 000000001c80: 981080c1
	s_sub_co_i32 s18, s7, s17                                  // 000000001c84: 81921107
	s_cmp_lg_u32 s16, 0                                        // 000000001c88: bf078010
	s_sub_co_ci_u32 s18, s18, s13                              // 000000001c8c: 82920d12
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c90: bf88ff9e
	s_sub_co_u32 s19, s4, s12                                  // 000000001c94: 80930c04
	s_sub_co_ci_u32 s18, s18, 0                                // 000000001c98: 82928012
	s_delay_alu instid0(salu_cycle_1)                          // 000000001c9c: bf870009
	s_cmp_ge_u32 s18, s13                                      // 000000001ca0: bf090d12
	s_cselect_b32 s20, -1, 0                                   // 000000001ca4: 981480c1
	s_cmp_ge_u32 s19, s12                                      // 000000001ca8: bf090c13
	s_cselect_b32 s21, -1, 0                                   // 000000001cac: 981580c1
	s_cmp_eq_u32 s18, s13                                      // 000000001cb0: bf060d12
	s_add_nc_u64 s[18:19], s[14:15], 1                         // 000000001cb4: a992810e
	s_cselect_b32 s22, s21, s20                                // 000000001cb8: 98161415
	s_add_nc_u64 s[20:21], s[14:15], 2                         // 000000001cbc: a994820e
	s_cmp_lg_u32 s22, 0                                        // 000000001cc0: bf078016
	s_cselect_b32 s18, s20, s18                                // 000000001cc4: 98121214
	s_cselect_b32 s19, s21, s19                                // 000000001cc8: 98131315
	s_cmp_lg_u32 s16, 0                                        // 000000001ccc: bf078010
	s_sub_co_ci_u32 s16, s7, s17                               // 000000001cd0: 82901107
	s_delay_alu instid0(salu_cycle_1)                          // 000000001cd4: bf870009
	s_cmp_ge_u32 s16, s13                                      // 000000001cd8: bf090d10
	s_cselect_b32 s17, -1, 0                                   // 000000001cdc: 981180c1
	s_cmp_ge_u32 s4, s12                                       // 000000001ce0: bf090c04
	s_cselect_b32 s4, -1, 0                                    // 000000001ce4: 980480c1
	s_cmp_eq_u32 s16, s13                                      // 000000001ce8: bf060d10
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cec: bf88ff9e
	s_cselect_b32 s4, s4, s17                                  // 000000001cf0: 98041104
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cf4: bf88ff9e
	s_cmp_lg_u32 s4, 0                                         // 000000001cf8: bf078004
	s_cselect_b32 s15, s19, s15                                // 000000001cfc: 980f0f13
	s_cselect_b32 s14, s18, s14                                // 000000001d00: 980e0e12
	s_branch 1                                                 // 000000001d04: bfa00001 <tessera_rocm_folded_matmul_708d500594ff51c6+0x20c>
	s_mov_b32 s5, -1                                           // 000000001d08: be8500c1
	s_delay_alu instid0(salu_cycle_1)                          // 000000001d0c: bf870009
	s_and_b32 s4, s5, exec_lo                                  // 000000001d10: 8b047e05
	s_cselect_b32 s4, 1, 0                                     // 000000001d14: 98048081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d18: bf88ff9e
	s_cmp_lg_u32 s4, 1                                         // 000000001d1c: bf078104
	s_cbranch_scc1 32                                          // 000000001d20: bfa20020 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2a4>
	v_cvt_f32_u32_e32 v1, s12                                  // 000000001d24: 7e020c0c
	s_sub_co_i32 s5, 0, s12                                    // 000000001d28: 81850c80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(trans32_dep_1)// 000000001d2c: bf870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001d30: 7e025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001d34: 100202ff 4f7ffffe
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000001d3c: bf870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001d40: 7e020f01
	v_readfirstlane_b32 s4, v1                                 // 000000001d44: 7e080501
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d48: bf88ff9e
	s_mul_i32 s5, s5, s4                                       // 000000001d4c: 96050405
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d50: bf88ff9e
	s_mul_hi_u32 s5, s4, s5                                    // 000000001d54: 96850504
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d58: bf88ff9e
	s_add_co_i32 s4, s4, s5                                    // 000000001d5c: 81040504
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d60: bf88ff9e
	s_mul_hi_u32 s4, s6, s4                                    // 000000001d64: 96840406
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d68: bf88ff9e
	s_mul_i32 s5, s4, s12                                      // 000000001d6c: 96050c04
	s_add_co_i32 s14, s4, 1                                    // 000000001d70: 810e8104
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d74: bf88ff9e
	s_sub_co_i32 s5, s6, s5                                    // 000000001d78: 81850506
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d7c: bf88ff9e
	s_sub_co_i32 s15, s5, s12                                  // 000000001d80: 818f0c05
	s_cmp_ge_u32 s5, s12                                       // 000000001d84: bf090c05
	s_cselect_b32 s4, s14, s4                                  // 000000001d88: 9804040e
	s_cselect_b32 s5, s15, s5                                  // 000000001d8c: 9805050f
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d90: bf88ff9e
	s_add_co_i32 s14, s4, 1                                    // 000000001d94: 810e8104
	s_cmp_ge_u32 s5, s12                                       // 000000001d98: bf090c05
	s_mov_b32 s15, 0                                           // 000000001d9c: be8f0080
	s_cselect_b32 s14, s14, s4                                 // 000000001da0: 980e040e
	s_add_nc_u64 s[16:17], s[8:9], 0xff                        // 000000001da4: a990ff08 000000ff
	s_lshl_b64 s[4:5], s[14:15], 2                             // 000000001dac: 8484820e
	s_lshr_b64 s[16:17], s[16:17], 8                           // 000000001db0: 85908810
	s_mul_u64 s[12:13], s[14:15], s[12:13]                     // 000000001db4: aa8c0c0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001db8: bf88ff9e
	s_sub_nc_u64 s[16:17], s[16:17], s[4:5]                    // 000000001dbc: aa100410
	s_sub_nc_u64 s[6:7], s[6:7], s[12:13]                      // 000000001dc0: aa060c06
	v_cmp_lt_u64_e64 s18, s[16:17], 4                          // 000000001dc4: d4590012 02010810
	s_mov_b32 s15, 0                                           // 000000001dcc: be8f0080
	s_and_b32 s12, s18, exec_lo                                // 000000001dd0: 8b0c7e12
	s_cselect_b32 s13, s17, 0                                  // 000000001dd4: 980d8011
	s_cselect_b32 s12, s16, 4                                  // 000000001dd8: 980c8410
	s_cmp_lg_u32 s7, 0                                         // 000000001ddc: bf078007
	s_cbranch_scc0 119                                         // 000000001de0: bfa10077 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4c0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001de4: bf88ff9e
	s_cvt_f32_u32 s14, s12                                     // 000000001de8: be8e650c
	s_cvt_f32_u32 s16, s13                                     // 000000001dec: be90650d
	s_sub_nc_u64 s[18:19], 0, s[12:13]                         // 000000001df0: aa120c80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001df4: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_3)// 000000001df8: bf870599
	s_fmac_f32 s14, s16, 0x4f800000                            // 000000001dfc: a38eff10 4f800000
	v_s_rcp_f32 s14, s14                                       // 000000001e04: d684000e 0201000e
	s_delay_alu instid0(trans32_dep_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e0c: bf870525
	s_mul_f32 s14, s14, 0x5f7ffffc                             // 000000001e10: a20eff0e 5f7ffffc
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e18: bf88ff9e
	s_mul_f32 s16, s14, 0x2f800000                             // 000000001e1c: a210ff0e 2f800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e24: bf88ff9e
	s_delay_alu instid0(salu_cycle_2) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e28: bf87052a
	s_trunc_f32 s16, s16                                       // 000000001e2c: be906210
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e30: bf88ff9e
	s_fmac_f32 s14, s16, 0xcf800000                            // 000000001e34: a38eff10 cf800000
	s_cvt_u32_f32 s17, s16                                     // 000000001e3c: be916710
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e40: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e44: bf870529
	s_cvt_u32_f32 s16, s14                                     // 000000001e48: be90670e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e4c: bf88ff9e
	s_mul_u64 s[20:21], s[18:19], s[16:17]                     // 000000001e50: aa941012
	s_delay_alu instid0(salu_cycle_1)                          // 000000001e54: bf870009
	s_mul_hi_u32 s23, s16, s21                                 // 000000001e58: 96971510
	s_mul_i32 s22, s16, s21                                    // 000000001e5c: 96161510
	s_mul_hi_u32 s14, s16, s20                                 // 000000001e60: 968e1410
	s_mul_i32 s25, s17, s20                                    // 000000001e64: 96191411
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e68: bf88ff9e
	s_add_nc_u64 s[22:23], s[14:15], s[22:23]                  // 000000001e6c: a996160e
	s_mul_hi_u32 s24, s17, s20                                 // 000000001e70: 96981411
	s_mul_hi_u32 s26, s17, s21                                 // 000000001e74: 969a1511
	s_add_co_u32 s14, s22, s25                                 // 000000001e78: 800e1916
	s_add_co_ci_u32 s14, s23, s24                              // 000000001e7c: 820e1817
	s_mul_i32 s20, s17, s21                                    // 000000001e80: 96141511
	s_add_co_ci_u32 s21, s26, 0                                // 000000001e84: 8215801a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e88: bf88ff9e
	s_add_nc_u64 s[20:21], s[14:15], s[20:21]                  // 000000001e8c: a994140e
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(salu_cycle_1)// 000000001e90: bf8704c9
	s_add_co_u32 s16, s16, s20                                 // 000000001e94: 80101410
	s_add_co_ci_u32 s17, s17, s21                              // 000000001e98: 82111511
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e9c: bf88ff9e
	s_mul_u64 s[18:19], s[18:19], s[16:17]                     // 000000001ea0: aa921012
	s_mul_hi_u32 s21, s16, s19                                 // 000000001ea4: 96951310
	s_mul_i32 s20, s16, s19                                    // 000000001ea8: 96141310
	s_mul_hi_u32 s14, s16, s18                                 // 000000001eac: 968e1210
	s_mul_i32 s23, s17, s18                                    // 000000001eb0: 96171211
	s_wait_alu depctr_sa_sdst(0)                               // 000000001eb4: bf88ff9e
	s_add_nc_u64 s[20:21], s[14:15], s[20:21]                  // 000000001eb8: a994140e
	s_mul_hi_u32 s22, s17, s18                                 // 000000001ebc: 96961211
	s_mul_hi_u32 s24, s17, s19                                 // 000000001ec0: 96981311
	s_add_co_u32 s14, s20, s23                                 // 000000001ec4: 800e1714
	s_add_co_ci_u32 s14, s21, s22                              // 000000001ec8: 820e1615
	s_mul_i32 s18, s17, s19                                    // 000000001ecc: 96121311
	s_add_co_ci_u32 s19, s24, 0                                // 000000001ed0: 82138018
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ed4: bf88ff9e
	s_add_nc_u64 s[18:19], s[14:15], s[18:19]                  // 000000001ed8: a992120e
	s_delay_alu instid0(salu_cycle_1)                          // 000000001edc: bf870009
	s_add_co_u32 s16, s16, s18                                 // 000000001ee0: 80101210
	s_add_co_ci_u32 s18, s17, s19                              // 000000001ee4: 82121311
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ee8: bf88ff9e
	s_mul_hi_u32 s14, s6, s16                                  // 000000001eec: 968e1006
	s_mul_hi_u32 s19, s7, s16                                  // 000000001ef0: 96931007
	s_mul_i32 s20, s7, s16                                     // 000000001ef4: 96141007
	s_mul_hi_u32 s17, s6, s18                                  // 000000001ef8: 96911206
	s_mul_i32 s16, s6, s18                                     // 000000001efc: 96101206
	s_mul_hi_u32 s21, s7, s18                                  // 000000001f00: 96951207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f04: bf88ff9e
	s_add_nc_u64 s[16:17], s[14:15], s[16:17]                  // 000000001f08: a990100e
	s_mul_i32 s18, s7, s18                                     // 000000001f0c: 96121207
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f10: bf88ff9e
	s_add_co_u32 s14, s16, s20                                 // 000000001f14: 800e1410
	s_add_co_ci_u32 s14, s17, s19                              // 000000001f18: 820e1311
	s_add_co_ci_u32 s19, s21, 0                                // 000000001f1c: 82138015
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f20: bf88ff9e
	s_add_nc_u64 s[16:17], s[14:15], s[18:19]                  // 000000001f24: a990120e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f28: bf88ff9e
	s_mul_u64 s[18:19], s[12:13], s[16:17]                     // 000000001f2c: aa92100c
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f30: bf870009
	s_sub_co_u32 s14, s6, s18                                  // 000000001f34: 808e1206
	s_cselect_b32 s18, -1, 0                                   // 000000001f38: 981280c1
	s_sub_co_i32 s20, s7, s19                                  // 000000001f3c: 81941307
	s_cmp_lg_u32 s18, 0                                        // 000000001f40: bf078012
	s_sub_co_ci_u32 s20, s20, s13                              // 000000001f44: 82940d14
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f48: bf88ff9e
	s_sub_co_u32 s21, s14, s12                                 // 000000001f4c: 80950c0e
	s_sub_co_ci_u32 s20, s20, 0                                // 000000001f50: 82948014
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f54: bf870009
	s_cmp_ge_u32 s20, s13                                      // 000000001f58: bf090d14
	s_cselect_b32 s22, -1, 0                                   // 000000001f5c: 981680c1
	s_cmp_ge_u32 s21, s12                                      // 000000001f60: bf090c15
	s_cselect_b32 s23, -1, 0                                   // 000000001f64: 981780c1
	s_cmp_eq_u32 s20, s13                                      // 000000001f68: bf060d14
	s_add_nc_u64 s[20:21], s[16:17], 1                         // 000000001f6c: a9948110
	s_cselect_b32 s24, s23, s22                                // 000000001f70: 98181617
	s_add_nc_u64 s[22:23], s[16:17], 2                         // 000000001f74: a9968210
	s_cmp_lg_u32 s24, 0                                        // 000000001f78: bf078018
	s_cselect_b32 s20, s22, s20                                // 000000001f7c: 98141416
	s_cselect_b32 s21, s23, s21                                // 000000001f80: 98151517
	s_cmp_lg_u32 s18, 0                                        // 000000001f84: bf078012
	s_sub_co_ci_u32 s18, s7, s19                               // 000000001f88: 82921307
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f8c: bf870009
	s_cmp_ge_u32 s18, s13                                      // 000000001f90: bf090d12
	s_cselect_b32 s19, -1, 0                                   // 000000001f94: 981380c1
	s_cmp_ge_u32 s14, s12                                      // 000000001f98: bf090c0e
	s_cselect_b32 s14, -1, 0                                   // 000000001f9c: 980e80c1
	s_cmp_eq_u32 s18, s13                                      // 000000001fa0: bf060d12
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa4: bf88ff9e
	s_cselect_b32 s14, s14, s19                                // 000000001fa8: 980e130e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fac: bf88ff9e
	s_cmp_lg_u32 s14, 0                                        // 000000001fb0: bf07800e
	s_cselect_b32 s17, s21, s17                                // 000000001fb4: 98111115
	s_cselect_b32 s16, s20, s16                                // 000000001fb8: 98101014
	s_branch 1                                                 // 000000001fbc: bfa00001 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4c4>
	s_mov_b32 s15, -1                                          // 000000001fc0: be8f00c1
	s_delay_alu instid0(salu_cycle_1)                          // 000000001fc4: bf870009
	s_and_b32 s14, s15, exec_lo                                // 000000001fc8: 8b0e7e0f
	s_cselect_b32 s14, 1, 0                                    // 000000001fcc: 980e8081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fd0: bf88ff9e
	s_cmp_lg_u32 s14, 1                                        // 000000001fd4: bf07810e
	s_cbranch_scc1 33                                          // 000000001fd8: bfa20021 <tessera_rocm_folded_matmul_708d500594ff51c6+0x560>
	v_cvt_f32_u32_e32 v1, s12                                  // 000000001fdc: 7e020c0c
	s_sub_co_i32 s15, 0, s12                                   // 000000001fe0: 818f0c80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(trans32_dep_1)// 000000001fe4: bf870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001fe8: 7e025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001fec: 100202ff 4f7ffffe
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000001ff4: bf870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001ff8: 7e020f01
	v_readfirstlane_b32 s14, v1                                // 000000001ffc: 7e1c0501
	s_wait_alu depctr_sa_sdst(0)                               // 000000002000: bf88ff9e
	s_mul_i32 s15, s15, s14                                    // 000000002004: 960f0e0f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002008: bf88ff9e
	s_mul_hi_u32 s15, s14, s15                                 // 00000000200c: 968f0f0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002010: bf88ff9e
	s_add_co_i32 s14, s14, s15                                 // 000000002014: 810e0f0e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002018: bf88ff9e
	s_mul_hi_u32 s14, s6, s14                                  // 00000000201c: 968e0e06
	s_wait_alu depctr_sa_sdst(0)                               // 000000002020: bf88ff9e
	s_mul_i32 s15, s14, s12                                    // 000000002024: 960f0c0e
	s_add_co_i32 s16, s14, 1                                   // 000000002028: 8110810e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000202c: bf88ff9e
	s_sub_co_i32 s15, s6, s15                                  // 000000002030: 818f0f06
	s_wait_alu depctr_sa_sdst(0)                               // 000000002034: bf88ff9e
	s_sub_co_i32 s17, s15, s12                                 // 000000002038: 81910c0f
	s_cmp_ge_u32 s15, s12                                      // 00000000203c: bf090c0f
	s_cselect_b32 s14, s16, s14                                // 000000002040: 980e0e10
	s_wait_alu depctr_sa_sdst(0)                               // 000000002044: bf88ff9e
	s_cselect_b32 s15, s17, s15                                // 000000002048: 980f0f11
	s_add_co_i32 s16, s14, 1                                   // 00000000204c: 8110810e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002050: bf88ff9e
	s_cmp_ge_u32 s15, s12                                      // 000000002054: bf090c0f
	s_mov_b32 s17, 0                                           // 000000002058: be910080
	s_cselect_b32 s16, s16, s14                                // 00000000205c: 98100e10
	s_load_b64 s[18:19], s[0:1], 0x8                           // 000000002060: f4002480 f8000008
	s_mul_u64 s[12:13], s[16:17], s[12:13]                     // 000000002068: aa8c0c10
	v_lshrrev_b32_e32 v11, 2, v0                               // 00000000206c: 32160082
	s_wait_alu depctr_sa_sdst(0)                               // 000000002070: bf88ff9e
	s_sub_nc_u64 s[6:7], s[6:7], s[12:13]                      // 000000002074: aa060c06
	s_load_b64 s[12:13], s[0:1], 0x30                          // 000000002078: f4002300 f8000030
	s_add_nc_u64 s[4:5], s[6:7], s[4:5]                        // 000000002080: a9840406
	v_dual_mov_b32 v56, 0 :: v_dual_lshlrev_b32 v1, 4, v0      // 000000002084: ca220080 38000084
	s_wait_alu depctr_sa_sdst(0)                               // 00000000208c: bf88ff9e
	s_lshl_b64 s[6:7], s[4:5], 8                               // 000000002090: 84868804
	s_lshl_b64 s[14:15], s[16:17], 6                           // 000000002094: 848e8610
	v_or_b32_e32 v2, s6, v11                                   // 000000002098: 38041606
	v_dual_mov_b32 v57, v56 :: v_dual_and_b32 v76, 48, v1      // 00000000209c: ca240138 394c02b0
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020a4: bf88ff9e
	v_or_b32_e32 v9, s14, v11                                  // 0000000020a8: 3812160e
	s_mul_i32 s4, s7, s2                                       // 0000000020ac: 96040207
	v_or_b32_e32 v3, 64, v2                                    // 0000000020b0: 380604c0
	v_or_b32_e32 v5, 0x80, v2                                  // 0000000020b4: 380a04ff 00000080
	v_or_b32_e32 v7, 0xc0, v2                                  // 0000000020bc: 380e04ff 000000c0
	v_mul_lo_u32 v12, v2, s3                                   // 0000000020c4: d72c000c 02000702
	v_mul_lo_u32 v15, v9, s3                                   // 0000000020cc: d72c000f 02000709
	s_wait_kmcnt 0x0                                           // 0000000020d4: bfc70000
	v_mad_co_u64_u32 v[1:2], null, v2, s2, s[18:19]            // 0000000020d8: d6fe7c01 00480502
	v_mul_lo_u32 v13, v3, s3                                   // 0000000020e0: d72c000d 02000703
	v_mad_co_u64_u32 v[3:4], null, v3, s2, s[18:19]            // 0000000020e8: d6fe7c03 00480503
	v_mul_lo_u32 v14, v5, s3                                   // 0000000020f0: d72c000e 02000705
	v_mad_co_u64_u32 v[5:6], null, v5, s2, s[18:19]            // 0000000020f8: d6fe7c05 00480505
	v_mul_lo_u32 v16, v7, s3                                   // 000000002100: d72c0010 02000707
	v_mad_co_u64_u32 v[7:8], null, v7, s2, s[18:19]            // 000000002108: d6fe7c07 00480507
	v_mad_co_u64_u32 v[9:10], null, v9, s2, s[12:13]           // 000000002110: d6fe7c09 00300509
	s_wait_alu depctr_sa_sdst(0)                               // 000000002118: bf88ff9e
	v_add3_u32 v2, s4, v2, v12                                 // 00000000211c: d6550002 04320404
	v_add_co_u32 v64, vcc_lo, v1, v76                          // 000000002124: d7006a40 02029901
	v_add3_u32 v4, s4, v4, v13                                 // 00000000212c: d6550004 04360804
	v_add3_u32 v6, s4, v6, v14                                 // 000000002134: d6550006 043a0c04
	s_delay_alu instid0(valu_dep_4)                            // 00000000213c: bf870004
	v_add_co_ci_u32_e64 v65, null, 0, v2, vcc_lo               // 000000002140: d5207c41 01aa0480
	v_add_co_u32 v66, vcc_lo, v3, v76                          // 000000002148: d7006a42 02029903
	s_mul_i32 s5, s15, s2                                      // 000000002150: 9605020f
	v_add3_u32 v1, s4, v8, v16                                 // 000000002154: d6550001 04421004
	s_wait_alu depctr_va_vcc(0)                                // 00000000215c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, 0, v4, vcc_lo               // 000000002160: d5207c43 01aa0880
	v_add_co_u32 v68, vcc_lo, v5, v76                          // 000000002168: d7006a44 02029905
	s_wait_alu depctr_sa_sdst(0)                               // 000000002170: bf88ff9e
	v_add3_u32 v2, s5, v10, v15                                // 000000002174: d6550002 043e1405
	s_wait_alu depctr_va_vcc(0)                                // 00000000217c: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v6, vcc_lo               // 000000002180: d5207c45 01aa0c80
	v_add_co_u32 v70, vcc_lo, v7, v76                          // 000000002188: d7006a46 02029907
	s_wait_alu depctr_va_vcc(0)                                // 000000002190: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, 0, v1, vcc_lo               // 000000002194: d5207c47 01aa0280
	v_add_co_u32 v74, vcc_lo, v9, v76                          // 00000000219c: d7006a4a 02029909
	s_wait_alu depctr_va_vcc(0)                                // 0000000021a4: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, 0, v2, vcc_lo               // 0000000021a8: d5207c4b 01aa0480
	s_clause 0x3                                               // 0000000021b0: bf850003
	global_load_b128 v[82:85], v[64:65], off                   // 0000000021b4: ee05c07c 00000052 00000040
	global_load_b128 v[86:89], v[66:67], off                   // 0000000021c0: ee05c07c 00000056 00000042
	global_load_b128 v[90:93], v[68:69], off                   // 0000000021cc: ee05c07c 0000005a 00000044
	global_load_b128 v[94:97], v[70:71], off                   // 0000000021d8: ee05c07c 0000005e 00000046
	global_load_b128 v[98:101], v[74:75], off                  // 0000000021e4: ee05c07c 00000062 0000004a
	v_mul_u32_u24_e32 v6, 0x50, v11                            // 0000000021f0: 160c16ff 00000050
	v_dual_mov_b32 v59, v56 :: v_dual_and_b32 v78, 15, v0      // 0000000021f8: ca240138 3b4e008f
	v_dual_mov_b32 v61, v56 :: v_dual_and_b32 v72, 0xc0, v0    // 000000002200: ca240138 3d4800ff 000000c0
	v_dual_mov_b32 v58, v56 :: v_dual_and_b32 v81, 32, v0      // 00000000220c: ca240138 3a5000a0
	s_delay_alu instid0(valu_dep_4)                            // 000000002214: bf870004
	v_add_nc_u32_e32 v76, v6, v76                              // 000000002218: 4a989906
	v_lshrrev_b32_e32 v1, 1, v0                                // 00000000221c: 32020081
	v_dual_mov_b32 v63, v56 :: v_dual_and_b32 v2, 0xcf, v0     // 000000002220: ca240138 3f0200ff 000000cf
	v_dual_mov_b32 v25, v56 :: v_dual_and_b32 v0, 47, v0       // 00000000222c: ca240138 190000af
	v_or_b32_e32 v80, 16, v72                                  // 000000002234: 38a09090
	v_or_b32_e32 v102, 32, v72                                 // 000000002238: 38cc90a0
	v_or_b32_e32 v128, 48, v72                                 // 00000000223c: 390090b0
	v_or3_b32 v160, v78, v81, 16                               // 000000002240: d65800a0 0242a34e
	v_dual_mov_b32 v60, v56 :: v_dual_and_b32 v73, 8, v1       // 000000002248: ca240138 3c480288
	v_mul_u32_u24_e32 v79, 0x50, v0                            // 000000002250: 169e00ff 00000050
	v_or_b32_e32 v103, v80, v78                                // 000000002258: 38ce9d50
	v_mul_u32_u24_e32 v77, 0x50, v2                            // 00000000225c: 169a04ff 00000050
	v_dual_mov_b32 v62, v56 :: v_dual_mov_b32 v27, v56         // 000000002264: ca100138 3e1a0138
	s_delay_alu instid0(valu_dep_4)                            // 00000000226c: bf870004
	v_or_b32_e32 v79, v73, v79                                 // 000000002270: 389e9f49
	v_dual_mov_b32 v24, v56 :: v_dual_mov_b32 v29, v56         // 000000002274: ca100138 181c0138
	v_dual_mov_b32 v26, v56 :: v_dual_mov_b32 v31, v56         // 00000000227c: ca100138 1a1e0138
	v_dual_mov_b32 v28, v56 :: v_dual_mov_b32 v49, v56         // 000000002284: ca100138 1c300138
	v_dual_mov_b32 v30, v56 :: v_dual_mov_b32 v51, v56         // 00000000228c: ca100138 1e320138
	v_dual_mov_b32 v48, v56 :: v_dual_mov_b32 v53, v56         // 000000002294: ca100138 30340138
	v_dual_mov_b32 v50, v56 :: v_dual_mov_b32 v55, v56         // 00000000229c: ca100138 32360138
	v_dual_mov_b32 v52, v56 :: v_dual_mov_b32 v17, v56         // 0000000022a4: ca100138 34100138
	v_dual_mov_b32 v54, v56 :: v_dual_mov_b32 v19, v56         // 0000000022ac: ca100138 36120138
	v_dual_mov_b32 v16, v56 :: v_dual_mov_b32 v21, v56         // 0000000022b4: ca100138 10140138
	v_dual_mov_b32 v18, v56 :: v_dual_mov_b32 v23, v56         // 0000000022bc: ca100138 12160138
	v_dual_mov_b32 v20, v56 :: v_dual_mov_b32 v41, v56         // 0000000022c4: ca100138 14280138
	v_dual_mov_b32 v22, v56 :: v_dual_mov_b32 v43, v56         // 0000000022cc: ca100138 162a0138
	v_dual_mov_b32 v40, v56 :: v_dual_mov_b32 v45, v56         // 0000000022d4: ca100138 282c0138
	v_dual_mov_b32 v42, v56 :: v_dual_mov_b32 v47, v56         // 0000000022dc: ca100138 2a2e0138
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v9, v56          // 0000000022e4: ca100138 2c080138
	v_dual_mov_b32 v46, v56 :: v_dual_mov_b32 v11, v56         // 0000000022ec: ca100138 2e0a0138
	v_dual_mov_b32 v8, v56 :: v_dual_mov_b32 v13, v56          // 0000000022f4: ca100138 080c0138
	v_dual_mov_b32 v10, v56 :: v_dual_mov_b32 v15, v56         // 0000000022fc: ca100138 0a0e0138
	v_dual_mov_b32 v12, v56 :: v_dual_mov_b32 v33, v56         // 000000002304: ca100138 0c200138
	v_dual_mov_b32 v14, v56 :: v_dual_mov_b32 v35, v56         // 00000000230c: ca100138 0e220138
	v_dual_mov_b32 v32, v56 :: v_dual_mov_b32 v37, v56         // 000000002314: ca100138 20240138
	v_dual_mov_b32 v34, v56 :: v_dual_mov_b32 v39, v56         // 00000000231c: ca100138 22260138
	v_dual_mov_b32 v36, v56 :: v_dual_mov_b32 v1, v56          // 000000002324: ca100138 24000138
	v_dual_mov_b32 v38, v56 :: v_dual_mov_b32 v3, v56          // 00000000232c: ca100138 26020138
	v_dual_mov_b32 v0, v56 :: v_dual_mov_b32 v5, v56           // 000000002334: ca100138 00040138
	v_dual_mov_b32 v2, v56 :: v_dual_mov_b32 v7, v56           // 00000000233c: ca100138 02060138
	v_dual_mov_b32 v4, v56 :: v_dual_add_nc_u32 v79, 0x5000, v79// 000000002344: ca200138 044e9eff 00005000
	v_or_b32_e32 v77, v73, v77                                 // 000000002350: 389a9b49
	s_movk_i32 s16, 0xffc0                                     // 000000002354: b010ffc0
	s_mov_b32 s17, -1                                          // 000000002358: be9100c1
	s_lshr_b64 s[4:5], s[2:3], 6                               // 00000000235c: 85848602
	s_mov_b64 s[12:13], 64                                     // 000000002360: be8c01c0
	s_wait_alu depctr_sa_sdst(0)                               // 000000002364: bf88ff9e
	s_add_nc_u64 s[2:3], s[2:3], s[16:17]                      // 000000002368: a9821002
	s_wait_loadcnt 0x4                                         // 00000000236c: bfc00004
	ds_store_b128 v76, v[82:85]                                // 000000002370: db7c0000 0000524c
	s_wait_loadcnt 0x3                                         // 000000002378: bfc00003
	ds_store_b128 v76, v[86:89] offset:5120                    // 00000000237c: db7c1400 0000564c
	s_wait_loadcnt 0x2                                         // 000000002384: bfc00002
	ds_store_b128 v76, v[90:93] offset:10240                   // 000000002388: db7c2800 00005a4c
	s_wait_loadcnt 0x1                                         // 000000002390: bfc00001
	ds_store_b128 v76, v[94:97] offset:15360                   // 000000002394: db7c3c00 00005e4c
	s_wait_loadcnt 0x0                                         // 00000000239c: bfc00000
	ds_store_b128 v76, v[98:101] offset:20480                  // 0000000023a0: db7c5000 0000624c
	v_or_b32_e32 v82, v102, v78                                // 0000000023a8: 38a49d66
	v_or_b32_e32 v83, v128, v78                                // 0000000023ac: 38a69d80
	v_mul_u32_u24_e32 v84, 0x50, v160                          // 0000000023b0: 16a940ff 00000050
	s_wait_dscnt 0x0                                           // 0000000023b8: bfc60000
	s_barrier_signal -1                                        // 0000000023bc: be804ec1
	v_mul_u32_u24_e32 v85, 0x50, v103                          // 0000000023c0: 16aaceff 00000050
	v_mul_u32_u24_e32 v86, 0x50, v82                           // 0000000023c8: 16aca4ff 00000050
	v_mul_u32_u24_e32 v87, 0x50, v83                           // 0000000023d0: 16aea6ff 00000050
	v_or_b32_e32 v88, v84, v73                                 // 0000000023d8: 38b09354
	v_mov_b32_e32 v6, v56                                      // 0000000023dc: 7e0c0338
	v_or_b32_e32 v82, v85, v73                                 // 0000000023e0: 38a49355
	v_or_b32_e32 v83, v86, v73                                 // 0000000023e4: 38a69356
	v_or_b32_e32 v84, v87, v73                                 // 0000000023e8: 38a89357
	v_add_nc_u32_e32 v85, 0x5000, v88                          // 0000000023ec: 4aaab0ff 00005000
	s_barrier_wait 0xffff                                      // 0000000023f4: bf94ffff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023f8: bf88ff9e
	v_cmp_lt_u64_e64 s16, s[12:13], s[2:3]                     // 0000000023fc: d4590010 0200040c
	ds_load_2addr_b64 v[107:110], v77 offset1:2                // 000000002404: d9dc0200 6b00004d
	ds_load_2addr_b64 v[111:114], v79 offset1:2                // 00000000240c: d9dc0200 6f00004f
	ds_load_2addr_b64 v[115:118], v85 offset1:2                // 000000002414: d9dc0200 73000055
	ds_load_2addr_b64 v[119:122], v82 offset1:2                // 00000000241c: d9dc0200 77000052
	ds_load_2addr_b64 v[123:126], v83 offset1:2                // 000000002424: d9dc0200 7b000053
	ds_load_2addr_b64 v[129:132], v84 offset1:2                // 00000000242c: d9dc0200 81000054
	ds_load_2addr_b64 v[133:136], v77 offset0:4 offset1:6      // 000000002434: d9dc0604 8500004d
	ds_load_2addr_b64 v[137:140], v79 offset0:4 offset1:6      // 00000000243c: d9dc0604 8900004f
	ds_load_2addr_b64 v[141:144], v85 offset0:4 offset1:6      // 000000002444: d9dc0604 8d000055
	ds_load_2addr_b64 v[145:148], v82 offset0:4 offset1:6      // 00000000244c: d9dc0604 91000052
	ds_load_2addr_b64 v[149:152], v83 offset0:4 offset1:6      // 000000002454: d9dc0604 95000053
	ds_load_2addr_b64 v[153:156], v84 offset0:4 offset1:6      // 00000000245c: d9dc0604 99000054
	s_add_nc_u64 s[4:5], s[4:5], -1                            // 000000002464: a984c104
	s_and_b32 s16, s16, exec_lo                                // 000000002468: 8b107e10
	s_cselect_b32 s17, s12, s2                                 // 00000000246c: 9811020c
	s_cselect_b32 s16, s13, s3                                 // 000000002470: 9810030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000002474: bf88ff9e
	v_add_co_u32 v86, vcc_lo, v64, s17                         // 000000002478: d7006a56 02002340
	s_wait_alu depctr_va_vcc(0)                                // 000000002480: bf88ff9d
	v_add_co_ci_u32_e64 v87, null, s16, v65, vcc_lo            // 000000002484: d5207c57 01aa8210
	v_add_co_u32 v90, vcc_lo, v66, s17                         // 00000000248c: d7006a5a 02002342
	s_wait_alu depctr_va_vcc(0)                                // 000000002494: bf88ff9d
	v_add_co_ci_u32_e64 v91, null, s16, v67, vcc_lo            // 000000002498: d5207c5b 01aa8610
	v_add_co_u32 v94, vcc_lo, v68, s17                         // 0000000024a0: d7006a5e 02002344
	s_wait_alu depctr_va_vcc(0)                                // 0000000024a8: bf88ff9d
	v_add_co_ci_u32_e64 v95, null, s16, v69, vcc_lo            // 0000000024ac: d5207c5f 01aa8a10
	v_add_co_u32 v98, vcc_lo, v70, s17                         // 0000000024b4: d7006a62 02002346
	s_wait_alu depctr_va_vcc(0)                                // 0000000024bc: bf88ff9d
	v_add_co_ci_u32_e64 v99, null, s16, v71, vcc_lo            // 0000000024c0: d5207c63 01aa8e10
	v_add_co_u32 v103, vcc_lo, v74, s17                        // 0000000024c8: d7006a67 0200234a
	s_wait_alu depctr_va_vcc(0)                                // 0000000024d0: bf88ff9d
	v_add_co_ci_u32_e64 v104, null, s16, v75, vcc_lo           // 0000000024d4: d5207c68 01aa9610
	s_clause 0x3                                               // 0000000024dc: bf850003
	global_load_b128 v[86:89], v[86:87], off                   // 0000000024e0: ee05c07c 00000056 00000056
	global_load_b128 v[90:93], v[90:91], off                   // 0000000024ec: ee05c07c 0000005a 0000005a
	global_load_b128 v[94:97], v[94:95], off                   // 0000000024f8: ee05c07c 0000005e 0000005e
	global_load_b128 v[98:101], v[98:99], off                  // 000000002504: ee05c07c 00000062 00000062
	global_load_b128 v[103:106], v[103:104], off               // 000000002510: ee05c07c 00000067 00000067
	s_wait_dscnt 0xa                                           // 00000000251c: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[107:108], v[111:112], v[56:63]// 000000002520: cc464038 1ce2df6b
	s_wait_dscnt 0x9                                           // 000000002528: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[107:108], v[115:116], v[24:31]// 00000000252c: cc464018 1c62e76b
	s_wait_dscnt 0x8                                           // 000000002534: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[119:120], v[111:112], v[48:55]// 000000002538: cc464030 1cc2df77
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[119:120], v[115:116], v[16:23]// 000000002540: cc464010 1c42e777
	s_wait_dscnt 0x7                                           // 000000002548: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[123:124], v[111:112], v[40:47]// 00000000254c: cc464028 1ca2df7b
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[123:124], v[115:116], v[8:15]// 000000002554: cc464008 1c22e77b
	s_wait_dscnt 0x6                                           // 00000000255c: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[129:130], v[111:112], v[32:39]// 000000002560: cc464020 1c82df81
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[129:130], v[115:116], v[0:7]// 000000002568: cc464000 1c02e781
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[109:110], v[113:114], v[56:63]// 000000002570: cc464038 1ce2e36d
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[109:110], v[117:118], v[24:31]// 000000002578: cc464018 1c62eb6d
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[121:122], v[113:114], v[48:55]// 000000002580: cc464030 1cc2e379
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[121:122], v[117:118], v[16:23]// 000000002588: cc464010 1c42eb79
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[125:126], v[113:114], v[40:47]// 000000002590: cc464028 1ca2e37d
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[125:126], v[117:118], v[8:15]// 000000002598: cc464008 1c22eb7d
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[131:132], v[113:114], v[32:39]// 0000000025a0: cc464020 1c82e383
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[131:132], v[117:118], v[0:7]// 0000000025a8: cc464000 1c02eb83
	s_wait_dscnt 0x4                                           // 0000000025b0: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[133:134], v[137:138], v[56:63]// 0000000025b4: cc464038 1ce31385
	s_wait_dscnt 0x3                                           // 0000000025bc: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[133:134], v[141:142], v[24:31]// 0000000025c0: cc464018 1c631b85
	s_wait_dscnt 0x2                                           // 0000000025c8: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[145:146], v[137:138], v[48:55]// 0000000025cc: cc464030 1cc31391
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[145:146], v[141:142], v[16:23]// 0000000025d4: cc464010 1c431b91
	s_wait_dscnt 0x1                                           // 0000000025dc: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[149:150], v[137:138], v[40:47]// 0000000025e0: cc464028 1ca31395
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[149:150], v[141:142], v[8:15]// 0000000025e8: cc464008 1c231b95
	s_wait_dscnt 0x0                                           // 0000000025f0: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[153:154], v[137:138], v[32:39]// 0000000025f4: cc464020 1c831399
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[153:154], v[141:142], v[0:7]// 0000000025fc: cc464000 1c031b99
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[135:136], v[139:140], v[56:63]// 000000002604: cc464038 1ce31787
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[135:136], v[143:144], v[24:31]// 00000000260c: cc464018 1c631f87
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[147:148], v[139:140], v[48:55]// 000000002614: cc464030 1cc31793
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[147:148], v[143:144], v[16:23]// 00000000261c: cc464010 1c431f93
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[151:152], v[139:140], v[40:47]// 000000002624: cc464028 1ca31797
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[151:152], v[143:144], v[8:15]// 00000000262c: cc464008 1c231f97
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[155:156], v[139:140], v[32:39]// 000000002634: cc464020 1c83179b
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[155:156], v[143:144], v[0:7]// 00000000263c: cc464000 1c031f9b
	s_add_nc_u64 s[12:13], s[12:13], 64                        // 000000002644: a98cc00c
	s_cmp_lg_u64 s[4:5], 0                                     // 000000002648: bf118004
	s_barrier_signal -1                                        // 00000000264c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002650: bf94ffff
	s_wait_loadcnt 0x4                                         // 000000002654: bfc00004
	ds_store_b128 v76, v[86:89]                                // 000000002658: db7c0000 0000564c
	s_wait_loadcnt 0x3                                         // 000000002660: bfc00003
	ds_store_b128 v76, v[90:93] offset:5120                    // 000000002664: db7c1400 00005a4c
	s_wait_loadcnt 0x2                                         // 00000000266c: bfc00002
	ds_store_b128 v76, v[94:97] offset:10240                   // 000000002670: db7c2800 00005e4c
	s_wait_loadcnt 0x1                                         // 000000002678: bfc00001
	ds_store_b128 v76, v[98:101] offset:15360                  // 00000000267c: db7c3c00 0000624c
	s_wait_loadcnt 0x0                                         // 000000002684: bfc00000
	ds_store_b128 v76, v[103:106] offset:20480                 // 000000002688: db7c5000 0000674c
	s_wait_dscnt 0x0                                           // 000000002690: bfc60000
	s_barrier_signal -1                                        // 000000002694: be804ec1
	s_barrier_wait 0xffff                                      // 000000002698: bf94ffff
	s_cbranch_scc1 65366                                       // 00000000269c: bfa2ff56 <tessera_rocm_folded_matmul_708d500594ff51c6+0x8f8>
	s_load_b64 s[12:13], s[0:1], 0x58                          // 0000000026a0: f4002300 f8000058
	v_or_b32_e32 v74, s6, v72                                  // 0000000026a8: 38949006
	v_or3_b32 v79, s15, 0, 0                                   // 0000000026ac: d658004f 0201000f
	v_or3_b32 v78, s14, v78, v81                               // 0000000026b4: d658004e 05469c0e
	v_mov_b32_e32 v65, s7                                      // 0000000026bc: 7e820207
	s_mov_b32 s5, 0                                            // 0000000026c0: be850080
	v_or_b32_e32 v76, v74, v73                                 // 0000000026c4: 3898934a
	v_mov_b32_e32 v77, s7                                      // 0000000026c8: 7e9a0207
	v_cmp_gt_u64_e64 s3, s[10:11], v[78:79]                    // 0000000026cc: d45c0003 02029c0a
	v_mov_b32_e32 v75, s7                                      // 0000000026d4: 7e960207
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_1)// 0000000026d8: bf870094
	v_or_b32_e32 v64, 7, v76                                   // 0000000026dc: 38809887
	v_cmp_gt_u64_e32 vcc_lo, s[8:9], v[64:65]                  // 0000000026e0: 7cb88008
	s_wait_kmcnt 0x0                                           // 0000000026e4: bfc70000
	s_and_b32 s4, s12, 15                                      // 0000000026e8: 8b048f0c
	s_and_b32 s2, s3, vcc_lo                                   // 0000000026ec: 8b026a03
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026f0: bf88ff9e
	s_cmp_eq_u64 s[4:5], 0                                     // 0000000026f4: bf108004
	s_cselect_b32 s22, -1, 0                                   // 0000000026f8: 981680c1
	s_delay_alu instid0(salu_cycle_1)                          // 0000000026fc: bf870009
	s_and_b32 s2, s22, s2                                      // 000000002700: 8b020216
	s_wait_alu depctr_sa_sdst(0)                               // 000000002704: bf88ff9e
	s_xor_b32 s2, s2, -1                                       // 000000002708: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 00000000270c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002710: be842002
	s_wait_alu depctr_sa_sdst(0)                               // 000000002714: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000002718: 8d05047e
	s_cbranch_execz 138                                        // 00000000271c: bfa5008a <tessera_rocm_folded_matmul_708d500594ff51c6+0xe48>
	v_cmp_gt_i64_e64 s2, s[8:9], v[76:77]                      // 000000002720: d4540002 02029808
	v_or_b32_e32 v66, 1, v76                                   // 000000002728: 38849881
	v_dual_mov_b32 v67, v77 :: v_dual_mov_b32 v70, v77         // 00000000272c: ca10014d 4346014d
	v_or_b32_e32 v69, 2, v76                                   // 000000002734: 388a9882
	v_or_b32_e32 v83, 3, v76                                   // 000000002738: 38a69883
	s_wait_alu depctr_va_sdst(0)                               // 00000000273c: bf88f19f
	v_cndmask_b32_e64 v68, 0, v77, s2                          // 000000002740: d5010044 000a9a80
	v_cmp_gt_i64_e64 s4, s[8:9], v[66:67]                      // 000000002748: d4540004 02028408
	v_cndmask_b32_e64 v67, 0, v76, s2                          // 000000002750: d5010043 000a9880
	v_cmp_gt_i64_e64 s2, s[8:9], v[69:70]                      // 000000002758: d4540002 02028a08
	v_mov_b32_e32 v84, v77                                     // 000000002760: 7ea8034d
	v_or_b32_e32 v85, 5, v76                                   // 000000002764: 38aa9885
	v_mov_b32_e32 v86, v77                                     // 000000002768: 7eac034d
	s_wait_alu depctr_va_sdst(0)                               // 00000000276c: bf88f19f
	v_cndmask_b32_e64 v81, 0, v66, s4                          // 000000002770: d5010051 00128480
	v_lshlrev_b64_e32 v[66:67], 2, v[67:68]                    // 000000002778: 3e848682
	v_cndmask_b32_e64 v68, 0, v69, s2                          // 00000000277c: d5010044 000a8a80
	v_cndmask_b32_e64 v69, 0, v77, s2                          // 000000002784: d5010045 000a9a80
	v_cndmask_b32_e64 v82, 0, v77, s4                          // 00000000278c: d5010052 00129a80
	v_cmp_gt_i64_e64 s2, s[8:9], v[83:84]                      // 000000002794: d4540002 0202a608
	v_add_co_u32 v66, s4, s12, v66                             // 00000000279c: d7000442 0202840c
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000027a4: bf870214
	v_lshlrev_b64_e32 v[68:69], 2, v[68:69]                    // 0000000027a8: 3e888882
	v_lshlrev_b64_e32 v[70:71], 2, v[81:82]                    // 0000000027ac: 3e8ca282
	s_wait_alu depctr_va_sdst(0)                               // 0000000027b0: bf88f19f
	s_delay_alu instid0(valu_dep_4)                            // 0000000027b4: bf870004
	v_cndmask_b32_e64 v81, 0, v83, s2                          // 0000000027b8: d5010051 000aa680
	v_or_b32_e32 v83, 4, v76                                   // 0000000027c0: 38a69884
	v_cndmask_b32_e64 v82, 0, v77, s2                          // 0000000027c4: d5010052 000a9a80
	v_add_co_ci_u32_e64 v67, null, s13, v67, s4                // 0000000027cc: d5207c43 0012860d
	v_add_co_u32 v87, s2, s12, v68                             // 0000000027d4: d7000257 0202880c
	v_add_co_u32 v70, s4, s12, v70                             // 0000000027dc: d7000446 02028c0c
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e4: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s13, v69, s2                // 0000000027e8: d5207c58 000a8a0d
	v_cmp_gt_i64_e64 s2, s[8:9], v[83:84]                      // 0000000027f0: d4540002 0202a608
	v_add_co_ci_u32_e64 v71, null, s13, v71, s4                // 0000000027f8: d5207c47 00128e0d
	v_lshlrev_b64_e32 v[68:69], 2, v[81:82]                    // 000000002800: 3e88a282
	v_cmp_gt_i64_e64 s4, s[8:9], v[85:86]                      // 000000002804: d4540004 0202aa08
	v_or_b32_e32 v81, 6, v76                                   // 00000000280c: 38a29886
	v_mov_b32_e32 v82, v77                                     // 000000002810: 7ea4034d
	s_wait_alu depctr_va_sdst(0)                               // 000000002814: bf88f19f
	v_cndmask_b32_e64 v83, 0, v83, s2                          // 000000002818: d5010053 000aa680
	v_cndmask_b32_e64 v84, 0, v77, s2                          // 000000002820: d5010054 000a9a80
	v_cndmask_b32_e64 v85, 0, v85, s4                          // 000000002828: d5010055 0012aa80
	v_cmp_gt_i64_e64 s2, s[8:9], v[81:82]                      // 000000002830: d4540002 0202a208
	v_cndmask_b32_e64 v86, 0, v77, s4                          // 000000002838: d5010056 00129a80
	v_add_co_u32 v89, s4, s12, v68                             // 000000002840: d7000459 0202880c
	s_wait_alu depctr_va_sdst(0)                               // 000000002848: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s13, v69, s4                // 00000000284c: d5207c5a 00128a0d
	v_lshlrev_b64_e32 v[68:69], 2, v[83:84]                    // 000000002854: 3e88a682
	v_lshlrev_b64_e32 v[82:83], 2, v[85:86]                    // 000000002858: 3ea4aa82
	v_cndmask_b32_e64 v84, 0, v81, s2                          // 00000000285c: d5010054 000aa280
	v_cndmask_b32_e64 v85, 0, v77, s2                          // 000000002864: d5010055 000a9a80
	v_cmp_gt_i64_e64 s2, s[8:9], v[64:65]                      // 00000000286c: d4540002 02028008
	v_add_co_u32 v91, s4, s12, v68                             // 000000002874: d700045b 0202880c
	s_wait_alu depctr_va_sdst(0)                               // 00000000287c: bf88f19f
	v_add_co_ci_u32_e64 v92, null, s13, v69, s4                // 000000002880: d5207c5c 00128a0d
	s_delay_alu instid0(valu_dep_3)                            // 000000002888: bf870003
	v_cndmask_b32_e64 v65, 0, v65, s2                          // 00000000288c: d5010041 000a8280
	v_cndmask_b32_e64 v64, 0, v64, s2                          // 000000002894: d5010040 000a8080
	v_lshlrev_b64_e32 v[68:69], 2, v[84:85]                    // 00000000289c: 3e88a882
	v_add_co_u32 v81, s2, s12, v82                             // 0000000028a0: d7000251 0202a40c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a8: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s13, v83, s2                // 0000000028ac: d5207c52 000aa60d
	v_lshlrev_b64_e32 v[64:65], 2, v[64:65]                    // 0000000028b4: 3e808082
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_3)// 0000000028b8: bf8701b4
	v_add_co_u32 v83, s2, s12, v68                             // 0000000028bc: d7000253 0202880c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028c4: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s13, v69, s2                // 0000000028c8: d5207c54 000a8a0d
	v_add_co_u32 v85, s2, s12, v64                             // 0000000028d0: d7000255 0202800c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028d8: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s13, v65, s2                // 0000000028dc: d5207c56 000a820d
	s_clause 0x7                                               // 0000000028e4: bf850007
	global_load_b32 v68, v[66:67], off                         // 0000000028e8: ee05007c 00000044 00000042
	global_load_b32 v69, v[70:71], off                         // 0000000028f4: ee05007c 00000045 00000046
	global_load_b32 v70, v[87:88], off                         // 000000002900: ee05007c 00000046 00000057
	global_load_b32 v71, v[89:90], off                         // 00000000290c: ee05007c 00000047 00000059
	global_load_b32 v64, v[91:92], off                         // 000000002918: ee05007c 00000040 0000005b
	global_load_b32 v65, v[81:82], off                         // 000000002924: ee05007c 00000041 00000051
	global_load_b32 v66, v[83:84], off                         // 000000002930: ee05007c 00000042 00000053
	global_load_b32 v67, v[85:86], off                         // 00000000293c: ee05007c 00000043 00000055
	s_wait_alu depctr_sa_sdst(0)                               // 000000002948: bf88ff9e
	s_or_saveexec_b32 s4, s5                                   // 00000000294c: be842205
	v_lshlrev_b64_e32 v[114:115], 2, v[76:77]                  // 000000002950: 3ee49882
	s_wait_alu depctr_sa_sdst(0)                               // 000000002954: bf88ff9e
	s_xor_b32 exec_lo, exec_lo, s4                             // 000000002958: 8d7e047e
	s_cbranch_execz 15                                         // 00000000295c: bfa5000f <tessera_rocm_folded_matmul_708d500594ff51c6+0xe9c>
	s_wait_loadcnt 0x3                                         // 000000002960: bfc00003
	s_delay_alu instid0(valu_dep_1)                            // 000000002964: bf870001
	v_add_co_u32 v64, s2, s12, v114                            // 000000002968: d7000240 0202e40c
	s_wait_loadcnt 0x2                                         // 000000002970: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000002974: bf88f19f
	v_add_co_ci_u32_e64 v65, null, s13, v115, s2               // 000000002978: d5207c41 000ae60d
	global_load_b128 v[68:71], v[64:65], off                   // 000000002980: ee05c07c 00000044 00000040
	s_wait_loadcnt 0x1                                         // 00000000298c: bfc00001
	global_load_b128 v[64:67], v[64:65], off offset:16         // 000000002990: ee05c07c 00000040 00001040
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000299c: 8c7e047e
	s_load_b64 s[16:17], s[0:1], 0x80                          // 0000000029a0: f4002400 f8000080
	v_cmp_gt_i64_e64 s2, s[10:11], v[78:79]                    // 0000000029a8: d4540002 02029c0a
	s_wait_alu depctr_va_sdst(0)                               // 0000000029b0: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 0000000029b4: bf870131
	v_cndmask_b32_e64 v82, 0, v78, s2                          // 0000000029b8: d5010052 000a9c80
	v_cndmask_b32_e64 v81, 0, v79, s2                          // 0000000029c0: d5010051 000a9e80
	s_wait_kmcnt 0x0                                           // 0000000029c8: bfc70000
	v_add_co_u32 v152, s2, s16, v82                            // 0000000029cc: d7000298 0202a410
	s_wait_alu depctr_va_sdst(0)                               // 0000000029d4: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000029d8: bf8700c2
	v_add_co_ci_u32_e64 v153, null, s17, v81, s2               // 0000000029dc: d5207c99 000aa211
	global_load_u8 v81, v[152:153], off                        // 0000000029e4: ee04007c 00000051 00000098
	s_wait_loadcnt 0x0                                         // 0000000029f0: bfc00000
	v_lshlrev_b32_e32 v82, 23, v81                             // 0000000029f4: 30a4a297
	v_mul_f32_e32 v81, v68, v82                                // 0000000029f8: 10a2a544
	s_delay_alu instid0(valu_dep_1)                            // 0000000029fc: bf870001
	v_cmp_class_f32_e64 s2, v81, 0x198                         // 000000002a00: d47e0002 0201ff51 00000198
	v_mul_f32_e32 v81, v56, v81                                // 000000002a0c: 10a2a338
	s_xor_b32 s2, s2, -1                                       // 000000002a10: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a14: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002a18: be842002
	s_cbranch_execnz 3324                                      // 000000002a1c: bfa60cfc <tessera_rocm_folded_matmul_708d500594ff51c6+0x4310>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002a24: 8c7e047e
	v_mul_f32_e32 v56, v69, v82                                // 000000002a28: 1070a545
	s_delay_alu instid0(valu_dep_1)                            // 000000002a2c: bf870001
	v_cmp_class_f32_e64 s2, v56, 0x198                         // 000000002a30: d47e0002 0201ff38 00000198
	v_mul_f32_e32 v56, v57, v56                                // 000000002a3c: 10707139
	s_xor_b32 s2, s2, -1                                       // 000000002a40: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a44: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002a48: be842002
	s_cbranch_execnz 3330                                      // 000000002a4c: bfa60d02 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4358>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002a54: 8c7e047e
	v_mul_f32_e32 v57, v70, v82                                // 000000002a58: 1072a546
	s_delay_alu instid0(valu_dep_1)                            // 000000002a5c: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002a60: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v57, v58, v57                                // 000000002a6c: 1072733a
	s_xor_b32 s2, s2, -1                                       // 000000002a70: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a74: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002a78: be842002
	s_cbranch_execnz 3336                                      // 000000002a7c: bfa60d08 <tessera_rocm_folded_matmul_708d500594ff51c6+0x43a0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002a84: 8c7e047e
	v_mul_f32_e32 v58, v71, v82                                // 000000002a88: 1074a547
	s_delay_alu instid0(valu_dep_1)                            // 000000002a8c: bf870001
	v_cmp_class_f32_e64 s2, v58, 0x198                         // 000000002a90: d47e0002 0201ff3a 00000198
	v_mul_f32_e32 v58, v59, v58                                // 000000002a9c: 1074753b
	s_xor_b32 s2, s2, -1                                       // 000000002aa0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002aa4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002aa8: be842002
	s_cbranch_execnz 3342                                      // 000000002aac: bfa60d0e <tessera_rocm_folded_matmul_708d500594ff51c6+0x43e8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ab0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002ab4: 8c7e047e
	v_mul_f32_e32 v59, v64, v82                                // 000000002ab8: 1076a540
	s_delay_alu instid0(valu_dep_1)                            // 000000002abc: bf870001
	v_cmp_class_f32_e64 s2, v59, 0x198                         // 000000002ac0: d47e0002 0201ff3b 00000198
	v_mul_f32_e32 v59, v60, v59                                // 000000002acc: 1076773c
	s_xor_b32 s2, s2, -1                                       // 000000002ad0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002ad8: be842002
	s_cbranch_execnz 3348                                      // 000000002adc: bfa60d14 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4430>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ae0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002ae4: 8c7e047e
	v_mul_f32_e32 v60, v65, v82                                // 000000002ae8: 1078a541
	s_delay_alu instid0(valu_dep_1)                            // 000000002aec: bf870001
	v_cmp_class_f32_e64 s2, v60, 0x198                         // 000000002af0: d47e0002 0201ff3c 00000198
	v_mul_f32_e32 v68, v61, v60                                // 000000002afc: 1088793d
	s_xor_b32 s2, s2, -1                                       // 000000002b00: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b04: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b08: be842002
	s_cbranch_execnz 3354                                      // 000000002b0c: bfa60d1a <tessera_rocm_folded_matmul_708d500594ff51c6+0x4478>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b14: 8c7e047e
	v_mul_f32_e32 v60, v66, v82                                // 000000002b18: 1078a542
	s_delay_alu instid0(valu_dep_1)                            // 000000002b1c: bf870001
	v_cmp_class_f32_e64 s2, v60, 0x198                         // 000000002b20: d47e0002 0201ff3c 00000198
	v_mul_f32_e32 v61, v62, v60                                // 000000002b2c: 107a793e
	s_xor_b32 s2, s2, -1                                       // 000000002b30: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b34: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b38: be842002
	s_cbranch_execnz 3360                                      // 000000002b3c: bfa60d20 <tessera_rocm_folded_matmul_708d500594ff51c6+0x44c0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b44: 8c7e047e
	v_mul_f32_e32 v60, v67, v82                                // 000000002b48: 1078a543
	s_delay_alu instid0(valu_dep_1)                            // 000000002b4c: bf870001
	v_cmp_class_f32_e64 s2, v60, 0x198                         // 000000002b50: d47e0002 0201ff3c 00000198
	v_mul_f32_e32 v60, v63, v60                                // 000000002b5c: 1078793f
	s_xor_b32 s2, s2, -1                                       // 000000002b60: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b64: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000002b68: be842002
	s_cbranch_execnz 3366                                      // 000000002b6c: bfa60d26 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4508>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002b74: 8c7e047e
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000002b78: f4002500 f80000a8
	v_mul_lo_u32 v64, v77, s10                                 // 000000002b80: d72c0040 0200154d
	v_mul_lo_u32 v66, v76, s11                                 // 000000002b88: d72c0042 0200174c
	v_mad_co_u64_u32 v[62:63], null, v76, s10, 0               // 000000002b90: d6fe7c3e 0200154c
	v_lshlrev_b64_e32 v[158:159], 1, v[78:79]                  // 000000002b98: 3f3c9c81
	v_or_b32_e32 v67, 0x400000, v81                            // 000000002b9c: 3886a2ff 00400000
	v_bfe_u32 v69, v56, 16, 1                                  // 000000002ba4: d6100045 02052138
	v_or_b32_e32 v70, 0x400000, v56                            // 000000002bac: 388c70ff 00400000
	s_lshl_b64 s[18:19], s[10:11], 1                           // 000000002bb4: 8492810a
	v_cmp_u_f32_e64 s1, v60, v60                               // 000000002bb8: d4180001 0202793c
	v_or_b32_e32 v163, 1, v73                                  // 000000002bc0: 39469281
	v_add3_u32 v63, v63, v66, v64                              // 000000002bc4: d655003f 0502853f
	v_bfe_u32 v66, v81, 16, 1                                  // 000000002bcc: d6100042 02052151
	v_add3_u32 v69, v69, v56, 0x7fff                           // 000000002bd4: d6550045 03fe7145 00007fff
	v_or_b32_e32 v64, s6, v80                                  // 000000002be0: 3880a006
	v_or_b32_e32 v162, 2, v73                                  // 000000002be4: 39449282
	v_lshlrev_b64_e32 v[62:63], 1, v[62:63]                    // 000000002be8: 3e7c7c81
	v_add3_u32 v66, v66, v81, 0x7fff                           // 000000002bec: d6550042 03fea342 00007fff
	v_or_b32_e32 v165, 3, v73                                  // 000000002bf8: 394a9283
	v_or_b32_e32 v86, v64, v73                                 // 000000002bfc: 38ac9340
	v_or_b32_e32 v164, 4, v73                                  // 000000002c00: 39489284
	v_or_b32_e32 v167, 5, v73                                  // 000000002c04: 394e9285
	s_wait_kmcnt 0x0                                           // 000000002c08: bfc70000
	v_add_co_u32 v62, s0, s20, v62                             // 000000002c0c: d700003e 02027c14
	s_delay_alu instid0(valu_dep_1)                            // 000000002c14: bf870001
	v_add_co_ci_u32_e64 v63, null, s21, v63, s0                // 000000002c18: d5207c3f 00027e15
	v_cmp_u_f32_e64 s0, v81, v81                               // 000000002c20: d4180000 0202a351
	v_or_b32_e32 v166, 6, v73                                  // 000000002c28: 394c9286
	v_or_b32_e32 v168, 7, v73                                  // 000000002c2c: 39509287
	v_mov_b32_e32 v65, s7                                      // 000000002c30: 7e820207
	v_mov_b32_e32 v87, s7                                      // 000000002c34: 7eae0207
	s_wait_alu depctr_va_sdst(0)                               // 000000002c38: bf88f19f
	v_cndmask_b32_e64 v66, v66, v67, s0                        // 000000002c3c: d5010042 00028742
	v_add_co_u32 v78, s0, v62, v158                            // 000000002c44: d700004e 02033d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c4c: bf88f19f
	v_add_co_ci_u32_e64 v79, null, v63, v159, s0               // 000000002c50: d5207c4f 00033f3f
	v_cmp_u_f32_e64 s0, v56, v56                               // 000000002c58: d4180000 02027138
	v_bfe_u32 v67, v57, 16, 1                                  // 000000002c60: d6100043 02052139
	v_or_b32_e32 v92, v64, v163                                // 000000002c68: 38b94740
	global_store_d16_hi_b16 v[78:79], v66, off                 // 000000002c6c: ee09407c 21000000 0000004e
	v_or_b32_e32 v88, v64, v162                                // 000000002c78: 38b14540
	s_wait_alu depctr_va_sdst(0)                               // 000000002c7c: bf88f19f
	v_cndmask_b32_e64 v56, v69, v70, s0                        // 000000002c80: d5010038 00028d45
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c88: bf88ff9e
	v_add_co_u32 v62, s0, v62, s18                             // 000000002c8c: d700003e 0200253e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c94: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s19, v63, s0                // 000000002c98: d5207c3f 00027e13
	v_add3_u32 v66, v67, v57, 0x7fff                           // 000000002ca0: d6550042 03fe7343 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002cac: bf8701a3
	v_add_co_u32 v82, s0, v62, v158                            // 000000002cb0: d7000052 02033d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000002cb8: bf88f19f
	v_add_co_ci_u32_e64 v83, null, v63, v159, s0               // 000000002cbc: d5207c53 00033f3f
	v_or_b32_e32 v67, 0x400000, v57                            // 000000002cc4: 388672ff 00400000
	v_cmp_u_f32_e64 s0, v57, v57                               // 000000002ccc: d4180000 02027339
	v_bfe_u32 v57, v58, 16, 1                                  // 000000002cd4: d6100039 0205213a
	global_store_d16_hi_b16 v[82:83], v56, off                 // 000000002cdc: ee09407c 1c000000 00000052
	v_or_b32_e32 v80, v64, v165                                // 000000002ce8: 38a14b40
	v_or_b32_e32 v70, v64, v164                                // 000000002cec: 388d4940
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf0: bf88f19f
	v_cndmask_b32_e64 v56, v66, v67, s0                        // 000000002cf4: d5010038 00028742
	v_add_co_u32 v62, s0, v62, s18                             // 000000002cfc: d700003e 0200253e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d04: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s19, v63, s0                // 000000002d08: d5207c3f 00027e13
	v_add3_u32 v57, v57, v58, 0x7fff                           // 000000002d10: d6550039 03fe7539 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d1c: bf870003
	v_add_co_u32 v84, s0, v62, v158                            // 000000002d20: d7000054 02033d3e
	v_or_b32_e32 v66, 0x400000, v58                            // 000000002d28: 388474ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d30: bf88f19f
	v_add_co_ci_u32_e64 v85, null, v63, v159, s0               // 000000002d34: d5207c55 00033f3f
	v_cmp_u_f32_e64 s0, v58, v58                               // 000000002d3c: d4180000 0202753a
	global_store_d16_hi_b16 v[84:85], v56, off                 // 000000002d44: ee09407c 1c000000 00000054
	s_wait_alu depctr_va_sdst(0)                               // 000000002d50: bf88f19f
	v_cndmask_b32_e64 v57, v57, v66, s0                        // 000000002d54: d5010039 00028539
	v_add_co_u32 v58, s0, v62, s18                             // 000000002d5c: d700003a 0200253e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d64: bf88f19f
	v_add_co_ci_u32_e64 v62, null, s19, v63, s0                // 000000002d68: d5207c3e 00027e13
	v_bfe_u32 v63, v59, 16, 1                                  // 000000002d70: d610003f 0205213b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d78: bf8701a3
	v_add_co_u32 v90, s0, v58, v158                            // 000000002d7c: d700005a 02033d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002d84: bf88f19f
	v_add_co_ci_u32_e64 v91, null, v62, v159, s0               // 000000002d88: d5207c5b 00033f3e
	s_delay_alu instid0(valu_dep_3)                            // 000000002d90: bf870003
	v_add3_u32 v56, v63, v59, 0x7fff                           // 000000002d94: d6550038 03fe773f 00007fff
	v_or_b32_e32 v63, 0x400000, v59                            // 000000002da0: 387e76ff 00400000
	v_cmp_u_f32_e64 s0, v59, v59                               // 000000002da8: d4180000 0202773b
	global_store_d16_hi_b16 v[90:91], v57, off                 // 000000002db0: ee09407c 1c800000 0000005a
	v_bfe_u32 v57, v68, 16, 1                                  // 000000002dbc: d6100039 02052144
	v_or_b32_e32 v66, v64, v166                                // 000000002dc4: 38854d40
	s_wait_alu depctr_va_sdst(0)                               // 000000002dc8: bf88f19f
	v_cndmask_b32_e64 v56, v56, v63, s0                        // 000000002dcc: d5010038 00027f38
	v_add_co_u32 v58, s0, v58, s18                             // 000000002dd4: d700003a 0200253a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ddc: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s19, v62, s0                // 000000002de0: d5207c3b 00027c13
	v_add3_u32 v57, v57, v68, 0x7fff                           // 000000002de8: d6550039 03fe8939 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002df4: bf870003
	v_add_co_u32 v96, s0, v58, v158                            // 000000002df8: d7000060 02033d3a
	v_or_b32_e32 v62, 0x400000, v68                            // 000000002e00: 387c88ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e08: bf88f19f
	v_add_co_ci_u32_e64 v97, null, v59, v159, s0               // 000000002e0c: d5207c61 00033f3b
	v_cmp_u_f32_e64 s0, v68, v68                               // 000000002e14: d4180000 02028944
	v_or_b32_e32 v63, 0x400000, v60                            // 000000002e1c: 387e78ff 00400000
	v_or_b32_e32 v68, v64, v167                                // 000000002e24: 38894f40
	global_store_d16_hi_b16 v[96:97], v56, off                 // 000000002e28: ee09407c 1c000000 00000060
	v_or_b32_e32 v64, v64, v168                                // 000000002e34: 38815140
	s_wait_alu depctr_va_sdst(0)                               // 000000002e38: bf88f19f
	v_cndmask_b32_e64 v57, v57, v62, s0                        // 000000002e3c: d5010039 00027d39
	v_add_co_u32 v58, s0, v58, s18                             // 000000002e44: d700003a 0200253a
	s_wait_alu depctr_va_sdst(0)                               // 000000002e4c: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s19, v59, s0                // 000000002e50: d5207c3b 00027613
	v_bfe_u32 v62, v61, 16, 1                                  // 000000002e58: d610003e 0205213d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e60: bf8701a3
	v_add_co_u32 v94, s0, v58, v158                            // 000000002e64: d700005e 02033d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002e6c: bf88f19f
	v_add_co_ci_u32_e64 v95, null, v59, v159, s0               // 000000002e70: d5207c5f 00033f3b
	s_delay_alu instid0(valu_dep_3)                            // 000000002e78: bf870003
	v_add3_u32 v56, v62, v61, 0x7fff                           // 000000002e7c: d6550038 03fe7b3e 00007fff
	v_or_b32_e32 v62, 0x400000, v61                            // 000000002e88: 387c7aff 00400000
	v_cmp_u_f32_e64 s0, v61, v61                               // 000000002e90: d4180000 02027b3d
	global_store_d16_hi_b16 v[94:95], v57, off                 // 000000002e98: ee09407c 1c800000 0000005e
	v_mov_b32_e32 v57, s7                                      // 000000002ea4: 7e720207
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea8: bf88f19f
	v_cndmask_b32_e64 v61, v56, v62, s0                        // 000000002eac: d501003d 00027d38
	v_bfe_u32 v56, v60, 16, 1                                  // 000000002eb4: d6100038 0205213c
	v_add_co_u32 v58, s0, v58, s18                             // 000000002ebc: d700003a 0200253a
	s_wait_alu depctr_va_sdst(0)                               // 000000002ec4: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s19, v59, s0                // 000000002ec8: d5207c3b 00027613
	s_delay_alu instid0(valu_dep_3)                            // 000000002ed0: bf870003
	v_add3_u32 v62, v56, v60, 0x7fff                           // 000000002ed4: d655003e 03fe7938 00007fff
	v_or_b32_e32 v56, 7, v86                                   // 000000002ee0: 3870ac87
	v_add_co_u32 v98, s0, v58, v158                            // 000000002ee4: d7000062 02033d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002eec: bf88f19f
	v_add_co_ci_u32_e64 v99, null, v59, v159, s0               // 000000002ef0: d5207c63 00033f3b
	v_add_co_u32 v58, s0, v58, s18                             // 000000002ef8: d700003a 0200253a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f00: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s19, v59, s0                // 000000002f04: d5207c3b 00027613
	v_cmp_gt_u64_e64 s0, s[8:9], v[56:57]                      // 000000002f0c: d45c0000 02027008
	v_cndmask_b32_e64 v56, v62, v63, s1                        // 000000002f14: d5010038 00067f3e
	v_add_co_u32 v100, s1, v58, v158                           // 000000002f1c: d7000164 02033d3a
	s_wait_alu depctr_va_sdst(0)                               // 000000002f24: bf88f19f
	v_add_co_ci_u32_e64 v101, null, v59, v159, s1              // 000000002f28: d5207c65 00073f3b
	s_and_b32 s1, s3, s0                                       // 000000002f30: 8b010003
	global_store_d16_hi_b16 v[98:99], v61, off                 // 000000002f34: ee09407c 1e800000 00000062
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f40: bf88ff9e
	s_and_b32 s1, s22, s1                                      // 000000002f44: 8b010116
	global_store_d16_hi_b16 v[100:101], v56, off               // 000000002f48: ee09407c 1c000000 00000064
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f54: bf88ff9e
	s_xor_b32 s1, s1, -1                                       // 000000002f58: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f5c: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000002f60: be822001
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f64: bf88ff9e
	s_xor_b32 s4, exec_lo, s2                                  // 000000002f68: 8d04027e
	s_cbranch_execz 133                                        // 000000002f6c: bfa50085 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1684>
	v_mov_b32_e32 v93, v65                                     // 000000002f70: 7eba0341
	v_cmp_gt_i64_e64 s1, s[8:9], v[86:87]                      // 000000002f74: d4540001 0202ac08
	v_mov_b32_e32 v89, v65                                     // 000000002f7c: 7eb20341
	v_mov_b32_e32 v81, v65                                     // 000000002f80: 7ea20341
	v_mov_b32_e32 v71, v65                                     // 000000002f84: 7e8e0341
	v_cmp_gt_i64_e64 s2, s[8:9], v[92:93]                      // 000000002f88: d4540002 0202b808
	v_mov_b32_e32 v69, v65                                     // 000000002f90: 7e8a0341
	s_wait_alu depctr_va_sdst(0)                               // 000000002f94: bf88f19f
	v_cndmask_b32_e64 v57, 0, v87, s1                          // 000000002f98: d5010039 0006ae80
	v_cndmask_b32_e64 v56, 0, v86, s1                          // 000000002fa0: d5010038 0006ac80
	v_cmp_gt_i64_e64 s1, s[8:9], v[88:89]                      // 000000002fa8: d4540001 0202b008
	v_mov_b32_e32 v67, v65                                     // 000000002fb0: 7e860341
	v_cndmask_b32_e64 v59, 0, v65, s2                          // 000000002fb4: d501003b 000a8280
	v_cndmask_b32_e64 v58, 0, v92, s2                          // 000000002fbc: d501003a 000ab880
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 000000002fc4: 3e707082
	v_cmp_gt_i64_e64 s2, s[8:9], v[80:81]                      // 000000002fc8: d4540002 0202a008
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd0: bf88f19f
	v_cndmask_b32_e64 v61, 0, v65, s1                          // 000000002fd4: d501003d 00068280
	v_cndmask_b32_e64 v60, 0, v88, s1                          // 000000002fdc: d501003c 0006b080
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]                    // 000000002fe4: 3e747482
	v_add_co_u32 v56, s1, s12, v56                             // 000000002fe8: d7000138 0202700c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000002ff0: bf870233
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]                    // 000000002ff4: 3e787882
	s_wait_alu depctr_va_sdst(0)                               // 000000002ff8: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s13, v57, s1                // 000000002ffc: d5207c39 0006720d
	v_add_co_u32 v58, s1, s12, v58                             // 000000003004: d700013a 0202740c
	v_cndmask_b32_e64 v63, 0, v65, s2                          // 00000000300c: d501003f 000a8280
	v_cndmask_b32_e64 v62, 0, v80, s2                          // 000000003014: d501003e 000aa080
	s_wait_alu depctr_va_sdst(0)                               // 00000000301c: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s13, v59, s1                // 000000003020: d5207c3b 0006760d
	v_cmp_gt_i64_e64 s1, s[8:9], v[70:71]                      // 000000003028: d4540001 02028c08
	v_add_co_u32 v103, s2, s12, v60                            // 000000003030: d7000267 0202780c
	s_wait_alu depctr_va_sdst(0)                               // 000000003038: bf88f19f
	v_add_co_ci_u32_e64 v104, null, s13, v61, s2               // 00000000303c: d5207c68 000a7a0d
	v_cmp_gt_i64_e64 s2, s[8:9], v[68:69]                      // 000000003044: d4540002 02028808
	v_lshlrev_b64_e32 v[60:61], 2, v[62:63]                    // 00000000304c: 3e787c82
	v_cndmask_b32_e64 v63, 0, v65, s1                          // 000000003050: d501003f 00068280
	v_cndmask_b32_e64 v62, 0, v70, s1                          // 000000003058: d501003e 00068c80
	v_cmp_gt_i64_e64 s1, s[8:9], v[66:67]                      // 000000003060: d4540001 02028408
	s_wait_alu depctr_va_sdst(0)                               // 000000003068: bf88f19f
	v_cndmask_b32_e64 v106, 0, v65, s2                         // 00000000306c: d501006a 000a8280
	v_cndmask_b32_e64 v105, 0, v68, s2                         // 000000003074: d5010069 000a8880
	v_add_co_u32 v107, s2, s12, v60                            // 00000000307c: d700026b 0202780c
	s_wait_alu depctr_va_sdst(0)                               // 000000003084: bf88f19f
	v_add_co_ci_u32_e64 v108, null, s13, v61, s2               // 000000003088: d5207c6c 000a7a0d
	v_lshlrev_b64_e32 v[60:61], 2, v[62:63]                    // 000000003090: 3e787c82
	v_lshlrev_b64_e32 v[62:63], 2, v[105:106]                  // 000000003094: 3e7cd282
	v_cndmask_b32_e64 v106, 0, v65, s1                         // 000000003098: d501006a 00068280
	v_cndmask_b32_e64 v105, 0, v66, s1                         // 0000000030a0: d5010069 00068480
	v_cmp_gt_i64_e64 s1, s[8:9], v[64:65]                      // 0000000030a8: d4540001 02028008
	v_add_co_u32 v109, s2, s12, v60                            // 0000000030b0: d700026d 0202780c
	s_wait_alu depctr_va_sdst(0)                               // 0000000030b8: bf88f19f
	v_add_co_ci_u32_e64 v110, null, s13, v61, s2               // 0000000030bc: d5207c6e 000a7a0d
	v_lshlrev_b64_e32 v[60:61], 2, v[105:106]                  // 0000000030c4: 3e78d282
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 0000000030c8: bf8701d4
	v_cndmask_b32_e64 v106, 0, v65, s1                         // 0000000030cc: d501006a 00068280
	v_cndmask_b32_e64 v105, 0, v64, s1                         // 0000000030d4: d5010069 00068080
	v_add_co_u32 v111, s1, s12, v62                            // 0000000030dc: d700016f 02027c0c
	s_wait_alu depctr_va_sdst(0)                               // 0000000030e4: bf88f19f
	v_add_co_ci_u32_e64 v112, null, s13, v63, s1               // 0000000030e8: d5207c70 00067e0d
	v_lshlrev_b64_e32 v[62:63], 2, v[105:106]                  // 0000000030f0: 3e7cd282
	v_add_co_u32 v105, s1, s12, v60                            // 0000000030f4: d7000169 0202780c
	s_wait_alu depctr_va_sdst(0)                               // 0000000030fc: bf88f19f
	v_add_co_ci_u32_e64 v106, null, s13, v61, s1               // 000000003100: d5207c6a 00067a0d
	s_delay_alu instid0(valu_dep_3)                            // 000000003108: bf870003
	v_add_co_u32 v116, s1, s12, v62                            // 00000000310c: d7000174 02027c0c
	s_wait_alu depctr_va_sdst(0)                               // 000000003114: bf88f19f
	v_add_co_ci_u32_e64 v117, null, s13, v63, s1               // 000000003118: d5207c75 00067e0d
	s_clause 0x7                                               // 000000003120: bf850007
	global_load_b32 v60, v[56:57], off                         // 000000003124: ee05007c 0000003c 00000038
	global_load_b32 v61, v[58:59], off                         // 000000003130: ee05007c 0000003d 0000003a
	global_load_b32 v62, v[103:104], off                       // 00000000313c: ee05007c 0000003e 00000067
	global_load_b32 v63, v[107:108], off                       // 000000003148: ee05007c 0000003f 0000006b
	global_load_b32 v56, v[109:110], off                       // 000000003154: ee05007c 00000038 0000006d
	global_load_b32 v57, v[111:112], off                       // 000000003160: ee05007c 00000039 0000006f
	global_load_b32 v58, v[105:106], off                       // 00000000316c: ee05007c 0000003a 00000069
	global_load_b32 v59, v[116:117], off                       // 000000003178: ee05007c 0000003b 00000074
	s_wait_alu depctr_sa_sdst(0)                               // 000000003184: bf88ff9e
	s_and_not1_saveexec_b32 s2, s4                             // 000000003188: be823004
	s_cbranch_execz 28                                         // 00000000318c: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x1700>
	s_wait_loadcnt 0x3                                         // 000000003190: bfc00003
	v_add_co_u32 v56, s1, s6, v72                              // 000000003194: d7000138 02029006
	s_wait_loadcnt 0x2                                         // 00000000319c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 0000000031a0: bf88f19f
	v_add_co_ci_u32_e64 v57, null, s7, 0, s1                   // 0000000031a4: d5207c39 00050007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031ac: bf870122
	v_add_co_u32 v56, s1, v56, v73                             // 0000000031b0: d7000138 02029338
	s_wait_alu depctr_va_sdst(0)                               // 0000000031b8: bf88f19f
	v_add_co_ci_u32_e64 v57, null, 0, v57, s1                  // 0000000031bc: d5207c39 00067280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000031c4: bf870091
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 0000000031c8: 3e707082
	v_add_co_u32 v56, s1, s12, v56                             // 0000000031cc: d7000138 0202700c
	s_wait_alu depctr_va_sdst(0)                               // 0000000031d4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000031d8: bf870002
	v_add_co_ci_u32_e64 v57, null, s13, v57, s1                // 0000000031dc: d5207c39 0006720d
	global_load_b128 v[60:63], v[56:57], off offset:64         // 0000000031e4: ee05c07c 0000003c 00004038
	s_wait_loadcnt 0x1                                         // 0000000031f0: bfc00001
	global_load_b128 v[56:59], v[56:57], off offset:80         // 0000000031f4: ee05c07c 00000038 00005038
	s_wait_alu depctr_sa_sdst(0)                               // 000000003200: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003204: 8c7e027e
	global_load_u8 v67, v[152:153], off                        // 000000003208: ee04007c 00000043 00000098
	s_wait_loadcnt 0x0                                         // 000000003214: bfc00000
	v_lshlrev_b32_e32 v69, 23, v67                             // 000000003218: 308a8697
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000321c: bf870091
	v_mul_f32_e32 v67, v60, v69                                // 000000003220: 10868b3c
	v_cmp_class_f32_e64 s1, v67, 0x198                         // 000000003224: d47e0001 0201ff43 00000198
	v_mul_f32_e32 v67, v48, v67                                // 000000003230: 10868730
	s_xor_b32 s1, s1, -1                                       // 000000003234: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003238: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000323c: be822001
	s_cbranch_execnz 2947                                      // 000000003240: bfa60b83 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4550>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003244: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003248: 8c7e027e
	v_mul_f32_e32 v48, v61, v69                                // 00000000324c: 10608b3d
	s_delay_alu instid0(valu_dep_1)                            // 000000003250: bf870001
	v_cmp_class_f32_e64 s1, v48, 0x198                         // 000000003254: d47e0001 0201ff30 00000198
	v_mul_f32_e32 v48, v49, v48                                // 000000003260: 10606131
	s_xor_b32 s1, s1, -1                                       // 000000003264: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003268: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000326c: be822001
	s_cbranch_execnz 2953                                      // 000000003270: bfa60b89 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4598>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003274: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003278: 8c7e027e
	v_mul_f32_e32 v49, v62, v69                                // 00000000327c: 10628b3e
	s_delay_alu instid0(valu_dep_1)                            // 000000003280: bf870001
	v_cmp_class_f32_e64 s1, v49, 0x198                         // 000000003284: d47e0001 0201ff31 00000198
	v_mul_f32_e32 v49, v50, v49                                // 000000003290: 10626332
	s_xor_b32 s1, s1, -1                                       // 000000003294: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003298: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000329c: be822001
	s_cbranch_execnz 2959                                      // 0000000032a0: bfa60b8f <tessera_rocm_folded_matmul_708d500594ff51c6+0x45e0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000032a8: 8c7e027e
	v_mul_f32_e32 v50, v63, v69                                // 0000000032ac: 10648b3f
	s_delay_alu instid0(valu_dep_1)                            // 0000000032b0: bf870001
	v_cmp_class_f32_e64 s1, v50, 0x198                         // 0000000032b4: d47e0001 0201ff32 00000198
	v_mul_f32_e32 v50, v51, v50                                // 0000000032c0: 10646533
	s_xor_b32 s1, s1, -1                                       // 0000000032c4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032c8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000032cc: be822001
	s_cbranch_execnz 2965                                      // 0000000032d0: bfa60b95 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4628>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000032d8: 8c7e027e
	v_mul_f32_e32 v51, v56, v69                                // 0000000032dc: 10668b38
	s_delay_alu instid0(valu_dep_1)                            // 0000000032e0: bf870001
	v_cmp_class_f32_e64 s1, v51, 0x198                         // 0000000032e4: d47e0001 0201ff33 00000198
	v_mul_f32_e32 v51, v52, v51                                // 0000000032f0: 10666734
	s_xor_b32 s1, s1, -1                                       // 0000000032f4: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000032fc: be822001
	s_cbranch_execnz 2971                                      // 000000003300: bfa60b9b <tessera_rocm_folded_matmul_708d500594ff51c6+0x4670>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003304: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003308: 8c7e027e
	v_mul_f32_e32 v52, v57, v69                                // 00000000330c: 10688b39
	s_delay_alu instid0(valu_dep_1)                            // 000000003310: bf870001
	v_cmp_class_f32_e64 s1, v52, 0x198                         // 000000003314: d47e0001 0201ff34 00000198
	v_mul_f32_e32 v52, v53, v52                                // 000000003320: 10686935
	s_xor_b32 s1, s1, -1                                       // 000000003324: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003328: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000332c: be822001
	s_cbranch_execnz 2977                                      // 000000003330: bfa60ba1 <tessera_rocm_folded_matmul_708d500594ff51c6+0x46b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003334: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003338: 8c7e027e
	v_mul_f32_e32 v53, v58, v69                                // 00000000333c: 106a8b3a
	s_delay_alu instid0(valu_dep_1)                            // 000000003340: bf870001
	v_cmp_class_f32_e64 s1, v53, 0x198                         // 000000003344: d47e0001 0201ff35 00000198
	v_mul_f32_e32 v60, v54, v53                                // 000000003350: 10786b36
	s_xor_b32 s1, s1, -1                                       // 000000003354: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003358: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000335c: be822001
	s_cbranch_execnz 2983                                      // 000000003360: bfa60ba7 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4700>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003364: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003368: 8c7e027e
	v_mul_f32_e32 v53, v59, v69                                // 00000000336c: 106a8b3b
	s_delay_alu instid0(valu_dep_1)                            // 000000003370: bf870001
	v_cmp_class_f32_e64 s1, v53, 0x198                         // 000000003374: d47e0001 0201ff35 00000198
	v_mul_f32_e32 v53, v55, v53                                // 000000003380: 106a6b37
	s_xor_b32 s1, s1, -1                                       // 000000003384: 8d01c101
	s_wait_alu depctr_sa_sdst(0)                               // 000000003388: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 00000000338c: be822001
	s_cbranch_execnz 2989                                      // 000000003390: bfa60bad <tessera_rocm_folded_matmul_708d500594ff51c6+0x4748>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003394: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000003398: 8c7e027e
	v_mul_lo_u32 v56, v87, s10                                 // 00000000339c: d72c0038 02001557
	v_mul_lo_u32 v58, v86, s11                                 // 0000000033a4: d72c003a 02001756
	v_mad_co_u64_u32 v[54:55], null, v86, s10, 0               // 0000000033ac: d6fe7c36 02001556
	v_bfe_u32 v59, v67, 16, 1                                  // 0000000033b4: d610003b 02052143
	v_or_b32_e32 v61, 0x400000, v67                            // 0000000033bc: 387a86ff 00400000
	v_cmp_u_f32_e64 s1, v67, v67                               // 0000000033c4: d4180001 02028743
	v_or_b32_e32 v62, 0x400000, v48                            // 0000000033cc: 387c60ff 00400000
	v_cmp_u_f32_e64 s2, v53, v53                               // 0000000033d4: d4180002 02026b35
	v_add3_u32 v59, v59, v67, 0x7fff                           // 0000000033dc: d655003b 03fe873b 00007fff
	v_mov_b32_e32 v57, s7                                      // 0000000033e8: 7e720207
	v_add3_u32 v55, v55, v58, v56                              // 0000000033ec: d6550037 04e27537
	v_bfe_u32 v58, v48, 16, 1                                  // 0000000033f4: d610003a 02052130
	v_or_b32_e32 v56, s6, v102                                 // 0000000033fc: 3870cc06
	s_wait_alu depctr_va_sdst(0)                               // 000000003400: bf88f19f
	v_cndmask_b32_e64 v59, v59, v61, s1                        // 000000003404: d501003b 00067b3b
	v_or_b32_e32 v61, 0x400000, v49                            // 00000000340c: 387a62ff 00400000
	v_lshlrev_b64_e32 v[54:55], 1, v[54:55]                    // 000000003414: 3e6c6c81
	v_add3_u32 v58, v58, v48, 0x7fff                           // 000000003418: d655003a 03fe613a 00007fff
	v_or_b32_e32 v106, v56, v73                                // 000000003424: 38d49338
	v_mov_b32_e32 v107, s7                                     // 000000003428: 7ed60207
	v_or_b32_e32 v116, v56, v163                               // 00000000342c: 38e94738
	v_or_b32_e32 v108, v56, v162                               // 000000003430: 38d94538
	v_add_co_u32 v54, s1, s20, v54                             // 000000003434: d7000136 02026c14
	s_wait_alu depctr_va_sdst(0)                               // 00000000343c: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s21, v55, s1                // 000000003440: d5207c37 00066e15
	v_cmp_u_f32_e64 s1, v48, v48                               // 000000003448: d4180001 02026130
	v_or_b32_e32 v102, v56, v165                               // 000000003450: 38cd4b38
	s_wait_alu depctr_va_sdst(0)                               // 000000003454: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003458: bf870002
	v_cndmask_b32_e64 v48, v58, v62, s1                        // 00000000345c: d5010030 00067d3a
	v_add_co_u32 v104, s1, v54, v158                           // 000000003464: d7000168 02033d36
	s_wait_alu depctr_va_sdst(0)                               // 00000000346c: bf88f19f
	v_add_co_ci_u32_e64 v105, null, v55, v159, s1              // 000000003470: d5207c69 00073f37
	v_add_co_u32 v54, s1, v54, s18                             // 000000003478: d7000136 02002536
	v_bfe_u32 v58, v49, 16, 1                                  // 000000003480: d610003a 02052131
	s_wait_alu depctr_va_sdst(0)                               // 000000003488: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s19, v55, s1                // 00000000348c: d5207c37 00066e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003494: bf870193
	v_add_co_u32 v110, s1, v54, v158                           // 000000003498: d700016e 02033d36
	v_add3_u32 v58, v58, v49, 0x7fff                           // 0000000034a0: d655003a 03fe633a 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 0000000034ac: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 0000000034b0: bf870143
	v_add_co_ci_u32_e64 v111, null, v55, v159, s1              // 0000000034b4: d5207c6f 00073f37
	v_cmp_u_f32_e64 s1, v49, v49                               // 0000000034bc: d4180001 02026331
	v_or_b32_e32 v62, v56, v164                                // 0000000034c4: 387d4938
	s_wait_alu depctr_va_sdst(0)                               // 0000000034c8: bf88f19f
	v_cndmask_b32_e64 v49, v58, v61, s1                        // 0000000034cc: d5010031 00067b3a
	v_add_co_u32 v54, s1, v54, s18                             // 0000000034d4: d7000136 02002536
	s_wait_alu depctr_va_sdst(0)                               // 0000000034dc: bf88f19f
	v_add_co_ci_u32_e64 v55, null, s19, v55, s1                // 0000000034e0: d5207c37 00066e13
	v_bfe_u32 v58, v50, 16, 1                                  // 0000000034e8: d610003a 02052132
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034f0: bf8701a3
	v_add_co_u32 v112, s1, v54, v158                           // 0000000034f4: d7000170 02033d36
	s_wait_alu depctr_va_sdst(0)                               // 0000000034fc: bf88f19f
	v_add_co_ci_u32_e64 v113, null, v55, v159, s1              // 000000003500: d5207c71 00073f37
	s_delay_alu instid0(valu_dep_3)                            // 000000003508: bf870003
	v_add3_u32 v58, v58, v50, 0x7fff                           // 00000000350c: d655003a 03fe653a 00007fff
	v_or_b32_e32 v61, 0x400000, v50                            // 000000003518: 387a64ff 00400000
	v_cmp_u_f32_e64 s1, v50, v50                               // 000000003520: d4180001 02026532
	s_clause 0x2                                               // 000000003528: bf850002
	global_store_d16_hi_b16 v[104:105], v59, off               // 00000000352c: ee09407c 1d800000 00000068
	global_store_d16_hi_b16 v[110:111], v48, off               // 000000003538: ee09407c 18000000 0000006e
	global_store_d16_hi_b16 v[112:113], v49, off               // 000000003544: ee09407c 18800000 00000070
	v_bfe_u32 v49, v51, 16, 1                                  // 000000003550: d6100031 02052133
	s_wait_alu depctr_va_sdst(0)                               // 000000003558: bf88f19f
	v_cndmask_b32_e64 v48, v58, v61, s1                        // 00000000355c: d5010030 00067b3a
	v_add_co_u32 v50, s1, v54, s18                             // 000000003564: d7000132 02002536
	s_wait_alu depctr_va_sdst(0)                               // 00000000356c: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s19, v55, s1                // 000000003570: d5207c36 00066e13
	v_add3_u32 v49, v49, v51, 0x7fff                           // 000000003578: d6550031 03fe6731 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003584: bf870003
	v_add_co_u32 v118, s1, v50, v158                           // 000000003588: d7000176 02033d32
	v_or_b32_e32 v55, 0x400000, v51                            // 000000003590: 386e66ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003598: bf88f19f
	v_add_co_ci_u32_e64 v119, null, v54, v159, s1              // 00000000359c: d5207c77 00073f36
	v_cmp_u_f32_e64 s1, v51, v51                               // 0000000035a4: d4180001 02026733
	v_bfe_u32 v51, v52, 16, 1                                  // 0000000035ac: d6100033 02052134
	v_or_b32_e32 v58, v56, v166                                // 0000000035b4: 38754d38
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b8: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 0000000035bc: bf8701d3
	v_cndmask_b32_e64 v49, v49, v55, s1                        // 0000000035c0: d5010031 00066f31
	v_add_co_u32 v50, s1, v50, s18                             // 0000000035c8: d7000132 02002532
	s_wait_alu depctr_va_sdst(0)                               // 0000000035d0: bf88f19f
	v_add_co_ci_u32_e64 v54, null, s19, v54, s1                // 0000000035d4: d5207c36 00066c13
	v_add3_u32 v51, v51, v52, 0x7fff                           // 0000000035dc: d6550033 03fe6933 00007fff
	v_add_co_u32 v122, s1, v50, v158                           // 0000000035e8: d700017a 02033d32
	v_or_b32_e32 v55, 0x400000, v52                            // 0000000035f0: 386e68ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000035f8: bf88f19f
	v_add_co_ci_u32_e64 v123, null, v54, v159, s1              // 0000000035fc: d5207c7b 00073f36
	v_cmp_u_f32_e64 s1, v52, v52                               // 000000003604: d4180001 02026934
	s_wait_alu depctr_va_sdst(0)                               // 00000000360c: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003610: bf8701d1
	v_cndmask_b32_e64 v51, v51, v55, s1                        // 000000003614: d5010033 00066f33
	v_add_co_u32 v50, s1, v50, s18                             // 00000000361c: d7000132 02002532
	s_wait_alu depctr_va_sdst(0)                               // 000000003624: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s19, v54, s1                // 000000003628: d5207c34 00066c13
	v_bfe_u32 v54, v60, 16, 1                                  // 000000003630: d6100036 0205213c
	v_add_co_u32 v120, s1, v50, v158                           // 000000003638: d7000178 02033d32
	s_wait_alu depctr_va_sdst(0)                               // 000000003640: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003644: bf870193
	v_add_co_ci_u32_e64 v121, null, v52, v159, s1              // 000000003648: d5207c79 00073f34
	v_add3_u32 v54, v54, v60, 0x7fff                           // 000000003650: d6550036 03fe7936 00007fff
	v_or_b32_e32 v55, 0x400000, v60                            // 00000000365c: 386e78ff 00400000
	v_cmp_u_f32_e64 s1, v60, v60                               // 000000003664: d4180001 0202793c
	s_clause 0x2                                               // 00000000366c: bf850002
	global_store_d16_hi_b16 v[118:119], v48, off               // 000000003670: ee09407c 18000000 00000076
	global_store_d16_hi_b16 v[122:123], v49, off               // 00000000367c: ee09407c 18800000 0000007a
	global_store_d16_hi_b16 v[120:121], v51, off               // 000000003688: ee09407c 19800000 00000078
	v_bfe_u32 v48, v53, 16, 1                                  // 000000003694: d6100030 02052135
	v_mov_b32_e32 v49, s7                                      // 00000000369c: 7e620207
	v_or_b32_e32 v60, v56, v167                                // 0000000036a0: 38794f38
	s_wait_alu depctr_va_sdst(0)                               // 0000000036a4: bf88f19f
	v_cndmask_b32_e64 v51, v54, v55, s1                        // 0000000036a8: d5010033 00066f36
	v_add_co_u32 v50, s1, v50, s18                             // 0000000036b0: d7000132 02002532
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s19, v52, s1                // 0000000036bc: d5207c34 00066813
	v_add3_u32 v54, v48, v53, 0x7fff                           // 0000000036c4: d6550036 03fe6b30 00007fff
	v_or_b32_e32 v48, 7, v106                                  // 0000000036d0: 3860d487
	v_add_co_u32 v124, s1, v50, v158                           // 0000000036d4: d700017c 02033d32
	s_wait_alu depctr_va_sdst(0)                               // 0000000036dc: bf88f19f
	v_add_co_ci_u32_e64 v125, null, v52, v159, s1              // 0000000036e0: d5207c7d 00073f34
	v_add_co_u32 v50, s1, v50, s18                             // 0000000036e8: d7000132 02002532
	v_or_b32_e32 v55, 0x400000, v53                            // 0000000036f0: 386e6aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s19, v52, s1                // 0000000036fc: d5207c34 00066813
	v_cmp_gt_u64_e64 s1, s[8:9], v[48:49]                      // 000000003704: d45c0001 02026008
	v_or_b32_e32 v56, v56, v168                                // 00000000370c: 38715138
	v_cndmask_b32_e64 v48, v54, v55, s2                        // 000000003710: d5010030 000a6f36
	v_add_co_u32 v126, s2, v50, v158                           // 000000003718: d700027e 02033d32
	s_wait_alu depctr_va_sdst(0)                               // 000000003720: bf88f19f
	v_add_co_ci_u32_e64 v127, null, v52, v159, s2              // 000000003724: d5207c7f 000b3f34
	s_and_b32 s2, s3, s1                                       // 00000000372c: 8b020103
	global_store_d16_hi_b16 v[124:125], v51, off               // 000000003730: ee09407c 19800000 0000007c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000373c: bf88ff9e
	s_and_b32 s2, s22, s2                                      // 000000003740: 8b020216
	global_store_d16_hi_b16 v[126:127], v48, off               // 000000003744: ee09407c 18000000 0000007e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003750: bf88ff9e
	s_xor_b32 s2, s2, -1                                       // 000000003754: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003758: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 00000000375c: be842002
	s_wait_alu depctr_sa_sdst(0)                               // 000000003760: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000003764: 8d05047e
	s_cbranch_execz 133                                        // 000000003768: bfa50085 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1e80>
	v_mov_b32_e32 v117, v57                                    // 00000000376c: 7eea0339
	v_cmp_gt_i64_e64 s2, s[8:9], v[106:107]                    // 000000003770: d4540002 0202d408
	v_mov_b32_e32 v109, v57                                    // 000000003778: 7eda0339
	v_mov_b32_e32 v103, v57                                    // 00000000377c: 7ece0339
	v_mov_b32_e32 v63, v57                                     // 000000003780: 7e7e0339
	v_cmp_gt_i64_e64 s4, s[8:9], v[116:117]                    // 000000003784: d4540004 0202e808
	v_mov_b32_e32 v61, v57                                     // 00000000378c: 7e7a0339
	s_wait_alu depctr_va_sdst(0)                               // 000000003790: bf88f19f
	v_cndmask_b32_e64 v49, 0, v107, s2                         // 000000003794: d5010031 000ad680
	v_cndmask_b32_e64 v48, 0, v106, s2                         // 00000000379c: d5010030 000ad480
	v_cmp_gt_i64_e64 s2, s[8:9], v[108:109]                    // 0000000037a4: d4540002 0202d808
	v_mov_b32_e32 v59, v57                                     // 0000000037ac: 7e760339
	v_cndmask_b32_e64 v51, 0, v57, s4                          // 0000000037b0: d5010033 00127280
	v_cndmask_b32_e64 v50, 0, v116, s4                         // 0000000037b8: d5010032 0012e880
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000037c0: 3e606082
	v_cmp_gt_i64_e64 s4, s[8:9], v[102:103]                    // 0000000037c4: d4540004 0202cc08
	s_wait_alu depctr_va_sdst(0)                               // 0000000037cc: bf88f19f
	v_cndmask_b32_e64 v53, 0, v57, s2                          // 0000000037d0: d5010035 000a7280
	v_cndmask_b32_e64 v52, 0, v108, s2                         // 0000000037d8: d5010034 000ad880
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]                    // 0000000037e0: 3e646482
	v_add_co_u32 v48, s2, s12, v48                             // 0000000037e4: d7000230 0202600c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 0000000037ec: bf870233
	v_lshlrev_b64_e32 v[52:53], 2, v[52:53]                    // 0000000037f0: 3e686882
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f4: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s13, v49, s2                // 0000000037f8: d5207c31 000a620d
	v_add_co_u32 v50, s2, s12, v50                             // 000000003800: d7000232 0202640c
	v_cndmask_b32_e64 v55, 0, v57, s4                          // 000000003808: d5010037 00127280
	v_cndmask_b32_e64 v54, 0, v102, s4                         // 000000003810: d5010036 0012cc80
	s_wait_alu depctr_va_sdst(0)                               // 000000003818: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s13, v51, s2                // 00000000381c: d5207c33 000a660d
	v_cmp_gt_i64_e64 s2, s[8:9], v[62:63]                      // 000000003824: d4540002 02027c08
	v_add_co_u32 v129, s4, s12, v52                            // 00000000382c: d7000481 0202680c
	s_wait_alu depctr_va_sdst(0)                               // 000000003834: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s13, v53, s4               // 000000003838: d5207c82 00126a0d
	v_cmp_gt_i64_e64 s4, s[8:9], v[60:61]                      // 000000003840: d4540004 02027808
	v_lshlrev_b64_e32 v[52:53], 2, v[54:55]                    // 000000003848: 3e686c82
	v_cndmask_b32_e64 v55, 0, v57, s2                          // 00000000384c: d5010037 000a7280
	v_cndmask_b32_e64 v54, 0, v62, s2                          // 000000003854: d5010036 000a7c80
	v_cmp_gt_i64_e64 s2, s[8:9], v[58:59]                      // 00000000385c: d4540002 02027408
	s_wait_alu depctr_va_sdst(0)                               // 000000003864: bf88f19f
	v_cndmask_b32_e64 v132, 0, v57, s4                         // 000000003868: d5010084 00127280
	v_cndmask_b32_e64 v131, 0, v60, s4                         // 000000003870: d5010083 00127880
	v_add_co_u32 v133, s4, s12, v52                            // 000000003878: d7000485 0202680c
	s_wait_alu depctr_va_sdst(0)                               // 000000003880: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s13, v53, s4               // 000000003884: d5207c86 00126a0d
	v_lshlrev_b64_e32 v[52:53], 2, v[54:55]                    // 00000000388c: 3e686c82
	v_lshlrev_b64_e32 v[54:55], 2, v[131:132]                  // 000000003890: 3e6d0682
	v_cndmask_b32_e64 v132, 0, v57, s2                         // 000000003894: d5010084 000a7280
	v_cndmask_b32_e64 v131, 0, v58, s2                         // 00000000389c: d5010083 000a7480
	v_cmp_gt_i64_e64 s2, s[8:9], v[56:57]                      // 0000000038a4: d4540002 02027008
	v_add_co_u32 v135, s4, s12, v52                            // 0000000038ac: d7000487 0202680c
	s_wait_alu depctr_va_sdst(0)                               // 0000000038b4: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s13, v53, s4               // 0000000038b8: d5207c88 00126a0d
	v_lshlrev_b64_e32 v[52:53], 2, v[131:132]                  // 0000000038c0: 3e690682
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 0000000038c4: bf8701d4
	v_cndmask_b32_e64 v132, 0, v57, s2                         // 0000000038c8: d5010084 000a7280
	v_cndmask_b32_e64 v131, 0, v56, s2                         // 0000000038d0: d5010083 000a7080
	v_add_co_u32 v137, s2, s12, v54                            // 0000000038d8: d7000289 02026c0c
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e0: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s13, v55, s2               // 0000000038e4: d5207c8a 000a6e0d
	v_lshlrev_b64_e32 v[54:55], 2, v[131:132]                  // 0000000038ec: 3e6d0682
	v_add_co_u32 v131, s2, s12, v52                            // 0000000038f0: d7000283 0202680c
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f8: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s13, v53, s2               // 0000000038fc: d5207c84 000a6a0d
	s_delay_alu instid0(valu_dep_3)                            // 000000003904: bf870003
	v_add_co_u32 v139, s2, s12, v54                            // 000000003908: d700028b 02026c0c
	s_wait_alu depctr_va_sdst(0)                               // 000000003910: bf88f19f
	v_add_co_ci_u32_e64 v140, null, s13, v55, s2               // 000000003914: d5207c8c 000a6e0d
	s_clause 0x7                                               // 00000000391c: bf850007
	global_load_b32 v52, v[48:49], off                         // 000000003920: ee05007c 00000034 00000030
	global_load_b32 v53, v[50:51], off                         // 00000000392c: ee05007c 00000035 00000032
	global_load_b32 v54, v[129:130], off                       // 000000003938: ee05007c 00000036 00000081
	global_load_b32 v55, v[133:134], off                       // 000000003944: ee05007c 00000037 00000085
	global_load_b32 v48, v[135:136], off                       // 000000003950: ee05007c 00000030 00000087
	global_load_b32 v49, v[137:138], off                       // 00000000395c: ee05007c 00000031 00000089
	global_load_b32 v50, v[131:132], off                       // 000000003968: ee05007c 00000032 00000083
	global_load_b32 v51, v[139:140], off                       // 000000003974: ee05007c 00000033 0000008b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003980: bf88ff9e
	s_and_not1_saveexec_b32 s4, s5                             // 000000003984: be843005
	s_cbranch_execz 28                                         // 000000003988: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x1efc>
	s_wait_loadcnt 0x3                                         // 00000000398c: bfc00003
	v_add_co_u32 v48, s2, s6, v72                              // 000000003990: d7000230 02029006
	s_wait_loadcnt 0x2                                         // 000000003998: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 00000000399c: bf88f19f
	v_add_co_ci_u32_e64 v49, null, s7, 0, s2                   // 0000000039a0: d5207c31 00090007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000039a8: bf870122
	v_add_co_u32 v48, s2, v48, v73                             // 0000000039ac: d7000230 02029330
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b4: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v49, s2                  // 0000000039b8: d5207c31 000a6280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000039c0: bf870091
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 0000000039c4: 3e606082
	v_add_co_u32 v48, s2, s12, v48                             // 0000000039c8: d7000230 0202600c
	s_wait_alu depctr_va_sdst(0)                               // 0000000039d0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000039d4: bf870002
	v_add_co_ci_u32_e64 v49, null, s13, v49, s2                // 0000000039d8: d5207c31 000a620d
	global_load_b128 v[52:55], v[48:49], off offset:128        // 0000000039e0: ee05c07c 00000034 00008030
	s_wait_loadcnt 0x1                                         // 0000000039ec: bfc00001
	global_load_b128 v[48:51], v[48:49], off offset:144        // 0000000039f0: ee05c07c 00000030 00009030
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003a00: 8c7e047e
	global_load_u8 v59, v[152:153], off                        // 000000003a04: ee04007c 0000003b 00000098
	s_wait_loadcnt 0x0                                         // 000000003a10: bfc00000
	v_lshlrev_b32_e32 v61, 23, v59                             // 000000003a14: 307a7697
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003a18: bf870091
	v_mul_f32_e32 v59, v52, v61                                // 000000003a1c: 10767b34
	v_cmp_class_f32_e64 s2, v59, 0x198                         // 000000003a20: d47e0002 0201ff3b 00000198
	v_mul_f32_e32 v59, v40, v59                                // 000000003a2c: 10767728
	s_xor_b32 s2, s2, -1                                       // 000000003a30: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a34: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003a38: be842002
	s_cbranch_execnz 2580                                      // 000000003a3c: bfa60a14 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4790>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003a44: 8c7e047e
	v_mul_f32_e32 v40, v53, v61                                // 000000003a48: 10507b35
	s_delay_alu instid0(valu_dep_1)                            // 000000003a4c: bf870001
	v_cmp_class_f32_e64 s2, v40, 0x198                         // 000000003a50: d47e0002 0201ff28 00000198
	v_mul_f32_e32 v40, v41, v40                                // 000000003a5c: 10505129
	s_xor_b32 s2, s2, -1                                       // 000000003a60: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a64: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003a68: be842002
	s_cbranch_execnz 2586                                      // 000000003a6c: bfa60a1a <tessera_rocm_folded_matmul_708d500594ff51c6+0x47d8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003a74: 8c7e047e
	v_mul_f32_e32 v41, v54, v61                                // 000000003a78: 10527b36
	s_delay_alu instid0(valu_dep_1)                            // 000000003a7c: bf870001
	v_cmp_class_f32_e64 s2, v41, 0x198                         // 000000003a80: d47e0002 0201ff29 00000198
	v_mul_f32_e32 v41, v42, v41                                // 000000003a8c: 1052532a
	s_xor_b32 s2, s2, -1                                       // 000000003a90: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a94: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003a98: be842002
	s_cbranch_execnz 2592                                      // 000000003a9c: bfa60a20 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4820>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003aa0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003aa4: 8c7e047e
	v_mul_f32_e32 v42, v55, v61                                // 000000003aa8: 10547b37
	s_delay_alu instid0(valu_dep_1)                            // 000000003aac: bf870001
	v_cmp_class_f32_e64 s2, v42, 0x198                         // 000000003ab0: d47e0002 0201ff2a 00000198
	v_mul_f32_e32 v42, v43, v42                                // 000000003abc: 1054552b
	s_xor_b32 s2, s2, -1                                       // 000000003ac0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ac4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003ac8: be842002
	s_cbranch_execnz 2598                                      // 000000003acc: bfa60a26 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4868>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ad0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003ad4: 8c7e047e
	v_mul_f32_e32 v43, v48, v61                                // 000000003ad8: 10567b30
	s_delay_alu instid0(valu_dep_1)                            // 000000003adc: bf870001
	v_cmp_class_f32_e64 s2, v43, 0x198                         // 000000003ae0: d47e0002 0201ff2b 00000198
	v_mul_f32_e32 v43, v44, v43                                // 000000003aec: 1056572c
	s_xor_b32 s2, s2, -1                                       // 000000003af0: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af4: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003af8: be842002
	s_cbranch_execnz 2604                                      // 000000003afc: bfa60a2c <tessera_rocm_folded_matmul_708d500594ff51c6+0x48b0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b00: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b04: 8c7e047e
	v_mul_f32_e32 v44, v49, v61                                // 000000003b08: 10587b31
	s_delay_alu instid0(valu_dep_1)                            // 000000003b0c: bf870001
	v_cmp_class_f32_e64 s2, v44, 0x198                         // 000000003b10: d47e0002 0201ff2c 00000198
	v_mul_f32_e32 v44, v45, v44                                // 000000003b1c: 1058592d
	s_xor_b32 s2, s2, -1                                       // 000000003b20: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b24: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b28: be842002
	s_cbranch_execnz 2610                                      // 000000003b2c: bfa60a32 <tessera_rocm_folded_matmul_708d500594ff51c6+0x48f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b34: 8c7e047e
	v_mul_f32_e32 v45, v50, v61                                // 000000003b38: 105a7b32
	s_delay_alu instid0(valu_dep_1)                            // 000000003b3c: bf870001
	v_cmp_class_f32_e64 s2, v45, 0x198                         // 000000003b40: d47e0002 0201ff2d 00000198
	v_mul_f32_e32 v52, v46, v45                                // 000000003b4c: 10685b2e
	s_xor_b32 s2, s2, -1                                       // 000000003b50: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b54: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b58: be842002
	s_cbranch_execnz 2616                                      // 000000003b5c: bfa60a38 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4940>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b64: 8c7e047e
	v_mul_f32_e32 v45, v51, v61                                // 000000003b68: 105a7b33
	s_delay_alu instid0(valu_dep_1)                            // 000000003b6c: bf870001
	v_cmp_class_f32_e64 s2, v45, 0x198                         // 000000003b70: d47e0002 0201ff2d 00000198
	v_mul_f32_e32 v45, v47, v45                                // 000000003b7c: 105a5b2f
	s_xor_b32 s2, s2, -1                                       // 000000003b80: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b84: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 000000003b88: be842002
	s_cbranch_execnz 2622                                      // 000000003b8c: bfa60a3e <tessera_rocm_folded_matmul_708d500594ff51c6+0x4988>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b90: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003b94: 8c7e047e
	v_mul_lo_u32 v48, v107, s10                                // 000000003b98: d72c0030 0200156b
	v_mul_lo_u32 v50, v106, s11                                // 000000003ba0: d72c0032 0200176a
	v_mad_co_u64_u32 v[46:47], null, v106, s10, 0              // 000000003ba8: d6fe7c2e 0200156a
	v_bfe_u32 v51, v59, 16, 1                                  // 000000003bb0: d6100033 0205213b
	v_or_b32_e32 v53, 0x400000, v59                            // 000000003bb8: 386a76ff 00400000
	v_cmp_u_f32_e64 s2, v59, v59                               // 000000003bc0: d4180002 0202773b
	v_or_b32_e32 v54, 0x400000, v40                            // 000000003bc8: 386c50ff 00400000
	v_cmp_u_f32_e64 s4, v45, v45                               // 000000003bd0: d4180004 02025b2d
	v_add3_u32 v51, v51, v59, 0x7fff                           // 000000003bd8: d6550033 03fe7733 00007fff
	v_mov_b32_e32 v49, s7                                      // 000000003be4: 7e620207
	v_add3_u32 v47, v47, v50, v48                              // 000000003be8: d655002f 04c2652f
	v_bfe_u32 v50, v40, 16, 1                                  // 000000003bf0: d6100032 02052128
	v_or_b32_e32 v48, s6, v128                                 // 000000003bf8: 38610006
	s_wait_alu depctr_va_sdst(0)                               // 000000003bfc: bf88f19f
	v_cndmask_b32_e64 v51, v51, v53, s2                        // 000000003c00: d5010033 000a6b33
	v_or_b32_e32 v53, 0x400000, v41                            // 000000003c08: 386a52ff 00400000
	v_lshlrev_b64_e32 v[46:47], 1, v[46:47]                    // 000000003c10: 3e5c5c81
	v_add3_u32 v50, v50, v40, 0x7fff                           // 000000003c14: d6550032 03fe5132 00007fff
	v_or_b32_e32 v132, v48, v73                                // 000000003c20: 39089330
	v_mov_b32_e32 v133, s7                                     // 000000003c24: 7f0a0207
	v_or_b32_e32 v140, v48, v163                               // 000000003c28: 39194730
	v_or_b32_e32 v134, v48, v162                               // 000000003c2c: 390d4530
	v_add_co_u32 v46, s2, s20, v46                             // 000000003c30: d700022e 02025c14
	s_wait_alu depctr_va_sdst(0)                               // 000000003c38: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s21, v47, s2                // 000000003c3c: d5207c2f 000a5e15
	v_cmp_u_f32_e64 s2, v40, v40                               // 000000003c44: d4180002 02025128
	v_or_b32_e32 v128, v48, v165                               // 000000003c4c: 39014b30
	s_wait_alu depctr_va_sdst(0)                               // 000000003c50: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003c54: bf870002
	v_cndmask_b32_e64 v40, v50, v54, s2                        // 000000003c58: d5010028 000a6d32
	v_add_co_u32 v130, s2, v46, v158                           // 000000003c60: d7000282 02033d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c68: bf88f19f
	v_add_co_ci_u32_e64 v131, null, v47, v159, s2              // 000000003c6c: d5207c83 000b3f2f
	v_add_co_u32 v46, s2, v46, s18                             // 000000003c74: d700022e 0200252e
	v_bfe_u32 v50, v41, 16, 1                                  // 000000003c7c: d6100032 02052129
	s_wait_alu depctr_va_sdst(0)                               // 000000003c84: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s19, v47, s2                // 000000003c88: d5207c2f 000a5e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003c90: bf870193
	v_add_co_u32 v136, s2, v46, v158                           // 000000003c94: d7000288 02033d2e
	v_add3_u32 v50, v50, v41, 0x7fff                           // 000000003c9c: d6550032 03fe5332 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 000000003ca8: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000003cac: bf870143
	v_add_co_ci_u32_e64 v137, null, v47, v159, s2              // 000000003cb0: d5207c89 000b3f2f
	v_cmp_u_f32_e64 s2, v41, v41                               // 000000003cb8: d4180002 02025329
	v_or_b32_e32 v54, v48, v164                                // 000000003cc0: 386d4930
	s_wait_alu depctr_va_sdst(0)                               // 000000003cc4: bf88f19f
	v_cndmask_b32_e64 v41, v50, v53, s2                        // 000000003cc8: d5010029 000a6b32
	v_add_co_u32 v46, s2, v46, s18                             // 000000003cd0: d700022e 0200252e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cd8: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s19, v47, s2                // 000000003cdc: d5207c2f 000a5e13
	v_bfe_u32 v50, v42, 16, 1                                  // 000000003ce4: d6100032 0205212a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003cec: bf8701a3
	v_add_co_u32 v138, s2, v46, v158                           // 000000003cf0: d700028a 02033d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cf8: bf88f19f
	v_add_co_ci_u32_e64 v139, null, v47, v159, s2              // 000000003cfc: d5207c8b 000b3f2f
	s_delay_alu instid0(valu_dep_3)                            // 000000003d04: bf870003
	v_add3_u32 v50, v50, v42, 0x7fff                           // 000000003d08: d6550032 03fe5532 00007fff
	v_or_b32_e32 v53, 0x400000, v42                            // 000000003d14: 386a54ff 00400000
	v_cmp_u_f32_e64 s2, v42, v42                               // 000000003d1c: d4180002 0202552a
	s_clause 0x2                                               // 000000003d24: bf850002
	global_store_d16_hi_b16 v[130:131], v51, off               // 000000003d28: ee09407c 19800000 00000082
	global_store_d16_hi_b16 v[136:137], v40, off               // 000000003d34: ee09407c 14000000 00000088
	global_store_d16_hi_b16 v[138:139], v41, off               // 000000003d40: ee09407c 14800000 0000008a
	v_bfe_u32 v41, v43, 16, 1                                  // 000000003d4c: d6100029 0205212b
	s_wait_alu depctr_va_sdst(0)                               // 000000003d54: bf88f19f
	v_cndmask_b32_e64 v40, v50, v53, s2                        // 000000003d58: d5010028 000a6b32
	v_add_co_u32 v42, s2, v46, s18                             // 000000003d60: d700022a 0200252e
	s_wait_alu depctr_va_sdst(0)                               // 000000003d68: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s19, v47, s2                // 000000003d6c: d5207c2e 000a5e13
	v_add3_u32 v41, v41, v43, 0x7fff                           // 000000003d74: d6550029 03fe5729 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003d80: bf870003
	v_add_co_u32 v142, s2, v42, v158                           // 000000003d84: d700028e 02033d2a
	v_or_b32_e32 v47, 0x400000, v43                            // 000000003d8c: 385e56ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003d94: bf88f19f
	v_add_co_ci_u32_e64 v143, null, v46, v159, s2              // 000000003d98: d5207c8f 000b3f2e
	v_cmp_u_f32_e64 s2, v43, v43                               // 000000003da0: d4180002 0202572b
	v_bfe_u32 v43, v44, 16, 1                                  // 000000003da8: d610002b 0205212c
	v_or_b32_e32 v50, v48, v166                                // 000000003db0: 38654d30
	s_wait_alu depctr_va_sdst(0)                               // 000000003db4: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000003db8: bf8701d3
	v_cndmask_b32_e64 v41, v41, v47, s2                        // 000000003dbc: d5010029 000a5f29
	v_add_co_u32 v42, s2, v42, s18                             // 000000003dc4: d700022a 0200252a
	s_wait_alu depctr_va_sdst(0)                               // 000000003dcc: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s19, v46, s2                // 000000003dd0: d5207c2e 000a5c13
	v_add3_u32 v43, v43, v44, 0x7fff                           // 000000003dd8: d655002b 03fe592b 00007fff
	v_add_co_u32 v146, s2, v42, v158                           // 000000003de4: d7000292 02033d2a
	v_or_b32_e32 v47, 0x400000, v44                            // 000000003dec: 385e58ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003df4: bf88f19f
	v_add_co_ci_u32_e64 v147, null, v46, v159, s2              // 000000003df8: d5207c93 000b3f2e
	v_cmp_u_f32_e64 s2, v44, v44                               // 000000003e00: d4180002 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000003e08: bf88f19f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003e0c: bf8701d1
	v_cndmask_b32_e64 v43, v43, v47, s2                        // 000000003e10: d501002b 000a5f2b
	v_add_co_u32 v42, s2, v42, s18                             // 000000003e18: d700022a 0200252a
	s_wait_alu depctr_va_sdst(0)                               // 000000003e20: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s19, v46, s2                // 000000003e24: d5207c2c 000a5c13
	v_bfe_u32 v46, v52, 16, 1                                  // 000000003e2c: d610002e 02052134
	v_add_co_u32 v144, s2, v42, v158                           // 000000003e34: d7000290 02033d2a
	s_wait_alu depctr_va_sdst(0)                               // 000000003e3c: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003e40: bf870193
	v_add_co_ci_u32_e64 v145, null, v44, v159, s2              // 000000003e44: d5207c91 000b3f2c
	v_add3_u32 v46, v46, v52, 0x7fff                           // 000000003e4c: d655002e 03fe692e 00007fff
	v_or_b32_e32 v47, 0x400000, v52                            // 000000003e58: 385e68ff 00400000
	v_cmp_u_f32_e64 s2, v52, v52                               // 000000003e60: d4180002 02026934
	s_clause 0x2                                               // 000000003e68: bf850002
	global_store_d16_hi_b16 v[142:143], v40, off               // 000000003e6c: ee09407c 14000000 0000008e
	global_store_d16_hi_b16 v[146:147], v41, off               // 000000003e78: ee09407c 14800000 00000092
	global_store_d16_hi_b16 v[144:145], v43, off               // 000000003e84: ee09407c 15800000 00000090
	v_bfe_u32 v40, v45, 16, 1                                  // 000000003e90: d6100028 0205212d
	v_mov_b32_e32 v41, s7                                      // 000000003e98: 7e520207
	v_or_b32_e32 v52, v48, v167                                // 000000003e9c: 38694f30
	s_wait_alu depctr_va_sdst(0)                               // 000000003ea0: bf88f19f
	v_cndmask_b32_e64 v43, v46, v47, s2                        // 000000003ea4: d501002b 000a5f2e
	v_add_co_u32 v42, s2, v42, s18                             // 000000003eac: d700022a 0200252a
	s_wait_alu depctr_va_sdst(0)                               // 000000003eb4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s19, v44, s2                // 000000003eb8: d5207c2c 000a5813
	v_add3_u32 v46, v40, v45, 0x7fff                           // 000000003ec0: d655002e 03fe5b28 00007fff
	v_or_b32_e32 v40, 7, v132                                  // 000000003ecc: 38510887
	v_add_co_u32 v148, s2, v42, v158                           // 000000003ed0: d7000294 02033d2a
	s_wait_alu depctr_va_sdst(0)                               // 000000003ed8: bf88f19f
	v_add_co_ci_u32_e64 v149, null, v44, v159, s2              // 000000003edc: d5207c95 000b3f2c
	v_add_co_u32 v42, s2, v42, s18                             // 000000003ee4: d700022a 0200252a
	v_or_b32_e32 v47, 0x400000, v45                            // 000000003eec: 385e5aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ef4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s19, v44, s2                // 000000003ef8: d5207c2c 000a5813
	v_cmp_gt_u64_e64 s2, s[8:9], v[40:41]                      // 000000003f00: d45c0002 02025008
	v_or_b32_e32 v48, v48, v168                                // 000000003f08: 38615130
	v_cndmask_b32_e64 v40, v46, v47, s4                        // 000000003f0c: d5010028 00125f2e
	v_add_co_u32 v150, s4, v42, v158                           // 000000003f14: d7000496 02033d2a
	s_wait_alu depctr_va_sdst(0)                               // 000000003f1c: bf88f19f
	v_add_co_ci_u32_e64 v151, null, v44, v159, s4              // 000000003f20: d5207c97 00133f2c
	s_and_b32 s3, s3, s2                                       // 000000003f28: 8b030203
	global_store_d16_hi_b16 v[148:149], v43, off               // 000000003f2c: ee09407c 15800000 00000094
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f38: bf88ff9e
	s_and_b32 s3, s22, s3                                      // 000000003f3c: 8b030316
	global_store_d16_hi_b16 v[150:151], v40, off               // 000000003f40: ee09407c 14000000 00000096
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f4c: bf88ff9e
	s_xor_b32 s3, s3, -1                                       // 000000003f50: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f54: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000003f58: be842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f5c: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000003f60: 8d05047e
	s_cbranch_execz 133                                        // 000000003f64: bfa50085 <tessera_rocm_folded_matmul_708d500594ff51c6+0x267c>
	v_mov_b32_e32 v141, v49                                    // 000000003f68: 7f1a0331
	v_cmp_gt_i64_e64 s3, s[8:9], v[132:133]                    // 000000003f6c: d4540003 02030808
	v_mov_b32_e32 v135, v49                                    // 000000003f74: 7f0e0331
	v_mov_b32_e32 v129, v49                                    // 000000003f78: 7f020331
	v_mov_b32_e32 v55, v49                                     // 000000003f7c: 7e6e0331
	v_cmp_gt_i64_e64 s4, s[8:9], v[140:141]                    // 000000003f80: d4540004 02031808
	v_mov_b32_e32 v53, v49                                     // 000000003f88: 7e6a0331
	s_wait_alu depctr_va_sdst(0)                               // 000000003f8c: bf88f19f
	v_cndmask_b32_e64 v41, 0, v133, s3                         // 000000003f90: d5010029 000f0a80
	v_cndmask_b32_e64 v40, 0, v132, s3                         // 000000003f98: d5010028 000f0880
	v_cmp_gt_i64_e64 s3, s[8:9], v[134:135]                    // 000000003fa0: d4540003 02030c08
	v_mov_b32_e32 v51, v49                                     // 000000003fa8: 7e660331
	v_cndmask_b32_e64 v43, 0, v49, s4                          // 000000003fac: d501002b 00126280
	v_cndmask_b32_e64 v42, 0, v140, s4                         // 000000003fb4: d501002a 00131880
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 000000003fbc: 3e505082
	v_cmp_gt_i64_e64 s4, s[8:9], v[128:129]                    // 000000003fc0: d4540004 02030008
	s_wait_alu depctr_va_sdst(0)                               // 000000003fc8: bf88f19f
	v_cndmask_b32_e64 v45, 0, v49, s3                          // 000000003fcc: d501002d 000e6280
	v_cndmask_b32_e64 v44, 0, v134, s3                         // 000000003fd4: d501002c 000f0c80
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]                    // 000000003fdc: 3e545482
	v_add_co_u32 v40, s3, s12, v40                             // 000000003fe0: d7000328 0202500c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000003fe8: bf870233
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]                    // 000000003fec: 3e585882
	s_wait_alu depctr_va_sdst(0)                               // 000000003ff0: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s13, v41, s3                // 000000003ff4: d5207c29 000e520d
	v_add_co_u32 v42, s3, s12, v42                             // 000000003ffc: d700032a 0202540c
	v_cndmask_b32_e64 v47, 0, v49, s4                          // 000000004004: d501002f 00126280
	v_cndmask_b32_e64 v46, 0, v128, s4                         // 00000000400c: d501002e 00130080
	s_wait_alu depctr_va_sdst(0)                               // 000000004014: bf88f19f
	v_add_co_ci_u32_e64 v43, null, s13, v43, s3                // 000000004018: d5207c2b 000e560d
	v_cmp_gt_i64_e64 s3, s[8:9], v[54:55]                      // 000000004020: d4540003 02026c08
	v_add_co_u32 v154, s4, s12, v44                            // 000000004028: d700049a 0202580c
	s_wait_alu depctr_va_sdst(0)                               // 000000004030: bf88f19f
	v_add_co_ci_u32_e64 v155, null, s13, v45, s4               // 000000004034: d5207c9b 00125a0d
	v_cmp_gt_i64_e64 s4, s[8:9], v[52:53]                      // 00000000403c: d4540004 02026808
	v_lshlrev_b64_e32 v[44:45], 2, v[46:47]                    // 000000004044: 3e585c82
	v_cndmask_b32_e64 v47, 0, v49, s3                          // 000000004048: d501002f 000e6280
	v_cndmask_b32_e64 v46, 0, v54, s3                          // 000000004050: d501002e 000e6c80
	v_cmp_gt_i64_e64 s3, s[8:9], v[50:51]                      // 000000004058: d4540003 02026408
	s_wait_alu depctr_va_sdst(0)                               // 000000004060: bf88f19f
	v_cndmask_b32_e64 v157, 0, v49, s4                         // 000000004064: d501009d 00126280
	v_cndmask_b32_e64 v156, 0, v52, s4                         // 00000000406c: d501009c 00126880
	v_add_co_u32 v169, s4, s12, v44                            // 000000004074: d70004a9 0202580c
	s_wait_alu depctr_va_sdst(0)                               // 00000000407c: bf88f19f
	v_add_co_ci_u32_e64 v170, null, s13, v45, s4               // 000000004080: d5207caa 00125a0d
	v_lshlrev_b64_e32 v[44:45], 2, v[46:47]                    // 000000004088: 3e585c82
	v_lshlrev_b64_e32 v[46:47], 2, v[156:157]                  // 00000000408c: 3e5d3882
	v_cndmask_b32_e64 v157, 0, v49, s3                         // 000000004090: d501009d 000e6280
	v_cndmask_b32_e64 v156, 0, v50, s3                         // 000000004098: d501009c 000e6480
	v_cmp_gt_i64_e64 s3, s[8:9], v[48:49]                      // 0000000040a0: d4540003 02026008
	v_add_co_u32 v171, s4, s12, v44                            // 0000000040a8: d70004ab 0202580c
	s_wait_alu depctr_va_sdst(0)                               // 0000000040b0: bf88f19f
	v_add_co_ci_u32_e64 v172, null, s13, v45, s4               // 0000000040b4: d5207cac 00125a0d
	v_lshlrev_b64_e32 v[44:45], 2, v[156:157]                  // 0000000040bc: 3e593882
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 0000000040c0: bf8701d4
	v_cndmask_b32_e64 v157, 0, v49, s3                         // 0000000040c4: d501009d 000e6280
	v_cndmask_b32_e64 v156, 0, v48, s3                         // 0000000040cc: d501009c 000e6080
	v_add_co_u32 v173, s3, s12, v46                            // 0000000040d4: d70003ad 02025c0c
	s_wait_alu depctr_va_sdst(0)                               // 0000000040dc: bf88f19f
	v_add_co_ci_u32_e64 v174, null, s13, v47, s3               // 0000000040e0: d5207cae 000e5e0d
	v_lshlrev_b64_e32 v[46:47], 2, v[156:157]                  // 0000000040e8: 3e5d3882
	v_add_co_u32 v156, s3, s12, v44                            // 0000000040ec: d700039c 0202580c
	s_wait_alu depctr_va_sdst(0)                               // 0000000040f4: bf88f19f
	v_add_co_ci_u32_e64 v157, null, s13, v45, s3               // 0000000040f8: d5207c9d 000e5a0d
	s_delay_alu instid0(valu_dep_3)                            // 000000004100: bf870003
	v_add_co_u32 v175, s3, s12, v46                            // 000000004104: d70003af 02025c0c
	s_wait_alu depctr_va_sdst(0)                               // 00000000410c: bf88f19f
	v_add_co_ci_u32_e64 v176, null, s13, v47, s3               // 000000004110: d5207cb0 000e5e0d
	s_clause 0x7                                               // 000000004118: bf850007
	global_load_b32 v44, v[40:41], off                         // 00000000411c: ee05007c 0000002c 00000028
	global_load_b32 v45, v[42:43], off                         // 000000004128: ee05007c 0000002d 0000002a
	global_load_b32 v46, v[154:155], off                       // 000000004134: ee05007c 0000002e 0000009a
	global_load_b32 v47, v[169:170], off                       // 000000004140: ee05007c 0000002f 000000a9
	global_load_b32 v40, v[171:172], off                       // 00000000414c: ee05007c 00000028 000000ab
	global_load_b32 v41, v[173:174], off                       // 000000004158: ee05007c 00000029 000000ad
	global_load_b32 v42, v[156:157], off                       // 000000004164: ee05007c 0000002a 0000009c
	global_load_b32 v43, v[175:176], off                       // 000000004170: ee05007c 0000002b 000000af
	s_wait_alu depctr_sa_sdst(0)                               // 00000000417c: bf88ff9e
	s_and_not1_saveexec_b32 s4, s5                             // 000000004180: be843005
	s_cbranch_execz 28                                         // 000000004184: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x26f8>
	s_wait_loadcnt 0x3                                         // 000000004188: bfc00003
	v_add_co_u32 v40, s3, s6, v72                              // 00000000418c: d7000328 02029006
	s_wait_loadcnt 0x2                                         // 000000004194: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000004198: bf88f19f
	v_add_co_ci_u32_e64 v41, null, s7, 0, s3                   // 00000000419c: d5207c29 000d0007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000041a4: bf870122
	v_add_co_u32 v40, s3, v40, v73                             // 0000000041a8: d7000328 02029328
	s_wait_alu depctr_va_sdst(0)                               // 0000000041b0: bf88f19f
	v_add_co_ci_u32_e64 v41, null, 0, v41, s3                  // 0000000041b4: d5207c29 000e5280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000041bc: bf870091
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 0000000041c0: 3e505082
	v_add_co_u32 v40, s3, s12, v40                             // 0000000041c4: d7000328 0202500c
	s_wait_alu depctr_va_sdst(0)                               // 0000000041cc: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000041d0: bf870002
	v_add_co_ci_u32_e64 v41, null, s13, v41, s3                // 0000000041d4: d5207c29 000e520d
	global_load_b128 v[44:47], v[40:41], off offset:192        // 0000000041dc: ee05c07c 0000002c 0000c028
	s_wait_loadcnt 0x1                                         // 0000000041e8: bfc00001
	global_load_b128 v[40:43], v[40:41], off offset:208        // 0000000041ec: ee05c07c 00000028 0000d028
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000041fc: 8c7e047e
	global_load_u8 v51, v[152:153], off                        // 000000004200: ee04007c 00000033 00000098
	s_wait_loadcnt 0x0                                         // 00000000420c: bfc00000
	v_lshlrev_b32_e32 v53, 23, v51                             // 000000004210: 306a6697
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004214: bf870091
	v_mul_f32_e32 v51, v44, v53                                // 000000004218: 10666b2c
	v_cmp_class_f32_e64 s3, v51, 0x198                         // 00000000421c: d47e0003 0201ff33 00000198
	v_mul_f32_e32 v51, v32, v51                                // 000000004228: 10666720
	s_xor_b32 s3, s3, -1                                       // 00000000422c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004230: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004234: be842003
	s_cbranch_execnz 2213                                      // 000000004238: bfa608a5 <tessera_rocm_folded_matmul_708d500594ff51c6+0x49d0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000423c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004240: 8c7e047e
	v_mul_f32_e32 v32, v45, v53                                // 000000004244: 10406b2d
	s_delay_alu instid0(valu_dep_1)                            // 000000004248: bf870001
	v_cmp_class_f32_e64 s3, v32, 0x198                         // 00000000424c: d47e0003 0201ff20 00000198
	v_mul_f32_e32 v32, v33, v32                                // 000000004258: 10404121
	s_xor_b32 s3, s3, -1                                       // 00000000425c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004260: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004264: be842003
	s_cbranch_execnz 2219                                      // 000000004268: bfa608ab <tessera_rocm_folded_matmul_708d500594ff51c6+0x4a18>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000426c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004270: 8c7e047e
	v_mul_f32_e32 v33, v46, v53                                // 000000004274: 10426b2e
	s_delay_alu instid0(valu_dep_1)                            // 000000004278: bf870001
	v_cmp_class_f32_e64 s3, v33, 0x198                         // 00000000427c: d47e0003 0201ff21 00000198
	v_mul_f32_e32 v33, v34, v33                                // 000000004288: 10424322
	s_xor_b32 s3, s3, -1                                       // 00000000428c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004290: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004294: be842003
	s_cbranch_execnz 2225                                      // 000000004298: bfa608b1 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4a60>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000429c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000042a0: 8c7e047e
	v_mul_f32_e32 v34, v47, v53                                // 0000000042a4: 10446b2f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042a8: bf870001
	v_cmp_class_f32_e64 s3, v34, 0x198                         // 0000000042ac: d47e0003 0201ff22 00000198
	v_mul_f32_e32 v34, v35, v34                                // 0000000042b8: 10444523
	s_xor_b32 s3, s3, -1                                       // 0000000042bc: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042c0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000042c4: be842003
	s_cbranch_execnz 2231                                      // 0000000042c8: bfa608b7 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4aa8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000042d0: 8c7e047e
	v_mul_f32_e32 v35, v40, v53                                // 0000000042d4: 10466b28
	s_delay_alu instid0(valu_dep_1)                            // 0000000042d8: bf870001
	v_cmp_class_f32_e64 s3, v35, 0x198                         // 0000000042dc: d47e0003 0201ff23 00000198
	v_mul_f32_e32 v35, v36, v35                                // 0000000042e8: 10464724
	s_xor_b32 s3, s3, -1                                       // 0000000042ec: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042f0: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000042f4: be842003
	s_cbranch_execnz 2237                                      // 0000000042f8: bfa608bd <tessera_rocm_folded_matmul_708d500594ff51c6+0x4af0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004300: 8c7e047e
	v_mul_f32_e32 v36, v41, v53                                // 000000004304: 10486b29
	s_delay_alu instid0(valu_dep_1)                            // 000000004308: bf870001
	v_cmp_class_f32_e64 s3, v36, 0x198                         // 00000000430c: d47e0003 0201ff24 00000198
	v_mul_f32_e32 v36, v37, v36                                // 000000004318: 10484925
	s_xor_b32 s3, s3, -1                                       // 00000000431c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004320: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004324: be842003
	s_cbranch_execnz 2243                                      // 000000004328: bfa608c3 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4b38>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000432c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004330: 8c7e047e
	v_mul_f32_e32 v37, v42, v53                                // 000000004334: 104a6b2a
	s_delay_alu instid0(valu_dep_1)                            // 000000004338: bf870001
	v_cmp_class_f32_e64 s3, v37, 0x198                         // 00000000433c: d47e0003 0201ff25 00000198
	v_mul_f32_e32 v55, v38, v37                                // 000000004348: 106e4b26
	s_xor_b32 s3, s3, -1                                       // 00000000434c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004350: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004354: be842003
	s_cbranch_execnz 2249                                      // 000000004358: bfa608c9 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4b80>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000435c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004360: 8c7e047e
	v_mul_f32_e32 v37, v43, v53                                // 000000004364: 104a6b2b
	s_delay_alu instid0(valu_dep_1)                            // 000000004368: bf870001
	v_cmp_class_f32_e64 s3, v37, 0x198                         // 00000000436c: d47e0003 0201ff25 00000198
	v_mul_f32_e32 v37, v39, v37                                // 000000004378: 104a4b27
	s_xor_b32 s3, s3, -1                                       // 00000000437c: 8d03c103
	s_wait_alu depctr_sa_sdst(0)                               // 000000004380: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004384: be842003
	s_cbranch_execnz 2255                                      // 000000004388: bfa608cf <tessera_rocm_folded_matmul_708d500594ff51c6+0x4bc8>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000438c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004390: 8c7e047e
	v_mul_lo_u32 v40, v133, s10                                // 000000004394: d72c0028 02001585
	v_mul_lo_u32 v41, v132, s11                                // 00000000439c: d72c0029 02001784
	v_mad_co_u64_u32 v[38:39], null, v132, s10, 0              // 0000000043a4: d6fe7c26 02001584
	v_bfe_u32 v42, v51, 16, 1                                  // 0000000043ac: d610002a 02052133
	v_cmp_u_f32_e64 s3, v51, v51                               // 0000000043b4: d4180003 02026733
	v_or_b32_e32 v43, 0x400000, v32                            // 0000000043bc: 385640ff 00400000
	v_bfe_u32 v46, v33, 16, 1                                  // 0000000043c4: d610002e 02052121
	v_mov_b32_e32 v161, s15                                    // 0000000043cc: 7f42020f
	v_add3_u32 v42, v42, v51, 0x7fff                           // 0000000043d0: d655002a 03fe672a 00007fff
	v_or_b32_e32 v160, s14, v160                               // 0000000043dc: 3941400e
	v_add3_u32 v39, v39, v41, v40                              // 0000000043e0: d6550027 04a25327
	v_or_b32_e32 v40, 0x400000, v51                            // 0000000043e8: 385066ff 00400000
	v_bfe_u32 v41, v32, 16, 1                                  // 0000000043f0: d6100029 02052120
	v_or_b32_e32 v51, 0x400000, v36                            // 0000000043f8: 386648ff 00400000
	v_cmp_u_f32_e64 s4, v37, v37                               // 000000004400: d4180004 02024b25
	v_lshlrev_b64_e32 v[38:39], 1, v[38:39]                    // 000000004408: 3e4c4c81
	s_wait_alu depctr_va_sdst(0)                               // 00000000440c: bf88f19f
	v_cndmask_b32_e64 v40, v42, v40, s3                        // 000000004410: d5010028 000e512a
	v_add3_u32 v41, v41, v32, 0x7fff                           // 000000004418: d6550029 03fe4129 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_1)// 000000004424: bf8700d3
	v_add_co_u32 v38, s3, s20, v38                             // 000000004428: d7000326 02024c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004430: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s21, v39, s3                // 000000004434: d5207c27 000e4e15
	v_cmp_u_f32_e64 s3, v32, v32                               // 00000000443c: d4180003 02024120
	s_wait_alu depctr_va_sdst(0)                               // 000000004444: bf88f19f
	v_cndmask_b32_e64 v32, v41, v43, s3                        // 000000004448: d5010020 000e5729
	v_add_co_u32 v41, s3, v38, s18                             // 000000004450: d7000329 02002526
	s_wait_alu depctr_va_sdst(0)                               // 000000004458: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s19, v39, s3                // 00000000445c: d5207c2f 000e4e13
	v_add_co_u32 v44, s3, v38, v158                            // 000000004464: d700032c 02033d26
	s_wait_alu depctr_va_sdst(0)                               // 00000000446c: bf88f19f
	v_add_co_ci_u32_e64 v45, null, v39, v159, s3               // 000000004470: d5207c2d 000f3f27
	v_add_co_u32 v42, s3, v41, v158                            // 000000004478: d700032a 02033d29
	s_wait_alu depctr_va_sdst(0)                               // 000000004480: bf88f19f
	v_add_co_ci_u32_e64 v43, null, v47, v159, s3               // 000000004484: d5207c2b 000f3f2f
	v_add3_u32 v38, v46, v33, 0x7fff                           // 00000000448c: d6550026 03fe432e 00007fff
	v_or_b32_e32 v39, 0x400000, v33                            // 000000004498: 384e42ff 00400000
	v_cmp_u_f32_e64 s3, v33, v33                               // 0000000044a0: d4180003 02024321
	s_clause 0x1                                               // 0000000044a8: bf850001
	global_store_d16_hi_b16 v[44:45], v40, off                 // 0000000044ac: ee09407c 14000000 0000002c
	global_store_d16_hi_b16 v[42:43], v32, off                 // 0000000044b8: ee09407c 10000000 0000002a
	v_bfe_u32 v33, v34, 16, 1                                  // 0000000044c4: d6100021 02052122
	v_or_b32_e32 v40, 0x400000, v34                            // 0000000044cc: 385044ff 00400000
	v_or_b32_e32 v46, 0x400000, v35                            // 0000000044d4: 385c46ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000044dc: bf88f19f
	v_cndmask_b32_e64 v32, v38, v39, s3                        // 0000000044e0: d5010020 000e4f26
	v_add_co_u32 v38, s3, v41, s18                             // 0000000044e8: d7000326 02002529
	s_wait_alu depctr_va_sdst(0)                               // 0000000044f0: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s19, v47, s3                // 0000000044f4: d5207c27 000e5e13
	v_add3_u32 v33, v33, v34, 0x7fff                           // 0000000044fc: d6550021 03fe4521 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004508: bf8701a3
	v_add_co_u32 v152, s3, v38, v158                           // 00000000450c: d7000398 02033d26
	s_wait_alu depctr_va_sdst(0)                               // 000000004514: bf88f19f
	v_add_co_ci_u32_e64 v153, null, v39, v159, s3              // 000000004518: d5207c99 000f3f27
	v_cmp_u_f32_e64 s3, v34, v34                               // 000000004520: d4180003 02024522
	v_bfe_u32 v34, v35, 16, 1                                  // 000000004528: d6100022 02052123
	s_wait_alu depctr_va_sdst(0)                               // 000000004530: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_3)// 000000004534: bf8701d2
	v_cndmask_b32_e64 v33, v33, v40, s3                        // 000000004538: d5010021 000e5121
	v_add_co_u32 v38, s3, v38, s18                             // 000000004540: d7000326 02002526
	s_wait_alu depctr_va_sdst(0)                               // 000000004548: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s19, v39, s3                // 00000000454c: d5207c27 000e4e13
	v_add3_u32 v34, v34, v35, 0x7fff                           // 000000004554: d6550022 03fe4722 00007fff
	v_add_co_u32 v40, s3, v38, v158                            // 000000004560: d7000328 02033d26
	s_wait_alu depctr_va_sdst(0)                               // 000000004568: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_1)// 00000000456c: bf8700b3
	v_add_co_ci_u32_e64 v41, null, v39, v159, s3               // 000000004570: d5207c29 000f3f27
	v_cmp_u_f32_e64 s3, v35, v35                               // 000000004578: d4180003 02024723
	s_wait_alu depctr_va_sdst(0)                               // 000000004580: bf88f19f
	v_cndmask_b32_e64 v34, v34, v46, s3                        // 000000004584: d5010022 000e5d22
	v_add_co_u32 v35, s3, v38, s18                             // 00000000458c: d7000323 02002526
	s_wait_alu depctr_va_sdst(0)                               // 000000004594: bf88f19f
	v_add_co_ci_u32_e64 v38, null, s19, v39, s3                // 000000004598: d5207c26 000e4e13
	v_bfe_u32 v39, v36, 16, 1                                  // 0000000045a0: d6100027 02052124
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000045a8: bf8701a3
	v_add_co_u32 v46, s3, v35, v158                            // 0000000045ac: d700032e 02033d23
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b4: bf88f19f
	v_add_co_ci_u32_e64 v47, null, v38, v159, s3               // 0000000045b8: d5207c2f 000f3f26
	s_delay_alu instid0(valu_dep_3)                            // 0000000045c0: bf870003
	v_add3_u32 v39, v39, v36, 0x7fff                           // 0000000045c4: d6550027 03fe4927 00007fff
	v_cmp_u_f32_e64 s3, v36, v36                               // 0000000045d0: d4180003 02024924
	s_clause 0x2                                               // 0000000045d8: bf850002
	global_store_d16_hi_b16 v[152:153], v32, off               // 0000000045dc: ee09407c 10000000 00000098
	global_store_d16_hi_b16 v[40:41], v33, off                 // 0000000045e8: ee09407c 10800000 00000028
	global_store_d16_hi_b16 v[46:47], v34, off                 // 0000000045f4: ee09407c 11000000 0000002e
	v_bfe_u32 v33, v55, 16, 1                                  // 000000004600: d6100021 02052137
	v_or_b32_e32 v36, 0x400000, v55                            // 000000004608: 38486eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004610: bf88f19f
	v_cndmask_b32_e64 v32, v39, v51, s3                        // 000000004614: d5010020 000e6727
	v_add_co_u32 v34, s3, v35, s18                             // 00000000461c: d7000322 02002523
	s_wait_alu depctr_va_sdst(0)                               // 000000004624: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s19, v38, s3                // 000000004628: d5207c23 000e4c13
	v_add3_u32 v33, v33, v55, 0x7fff                           // 000000004630: d6550021 03fe6f21 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000463c: bf8701a3
	v_add_co_u32 v154, s3, v34, v158                           // 000000004640: d700039a 02033d22
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: bf88f19f
	v_add_co_ci_u32_e64 v155, null, v35, v159, s3              // 00000000464c: d5207c9b 000f3f23
	v_cmp_u_f32_e64 s3, v55, v55                               // 000000004654: d4180003 02026f37
	v_or_b32_e32 v38, 0x400000, v37                            // 00000000465c: 384c4aff 00400000
	global_store_d16_hi_b16 v[154:155], v32, off               // 000000004664: ee09407c 10000000 0000009a
	s_wait_alu depctr_va_sdst(0)                               // 000000004670: bf88f19f
	v_cndmask_b32_e64 v33, v33, v36, s3                        // 000000004674: d5010021 000e4921
	v_add_co_u32 v34, s3, v34, s18                             // 00000000467c: d7000322 02002522
	s_wait_alu depctr_va_sdst(0)                               // 000000004684: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s19, v35, s3                // 000000004688: d5207c23 000e4613
	v_bfe_u32 v36, v37, 16, 1                                  // 000000004690: d6100024 02052125
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000004698: bf8701a3
	v_add_co_u32 v156, s3, v34, v158                           // 00000000469c: d700039c 02033d22
	s_wait_alu depctr_va_sdst(0)                               // 0000000046a4: bf88f19f
	v_add_co_ci_u32_e64 v157, null, v35, v159, s3              // 0000000046a8: d5207c9d 000f3f23
	v_add_co_u32 v34, s3, v34, s18                             // 0000000046b0: d7000322 02002522
	v_add3_u32 v36, v36, v37, 0x7fff                           // 0000000046b8: d6550024 03fe4b24 00007fff
	s_wait_alu depctr_va_sdst(0)                               // 0000000046c4: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s19, v35, s3                // 0000000046c8: d5207c23 000e4613
	v_cmp_gt_u64_e64 s3, s[10:11], v[160:161]                  // 0000000046d0: d45c0003 0203400a
	global_store_d16_hi_b16 v[156:157], v33, off               // 0000000046d8: ee09407c 10800000 0000009c
	v_cndmask_b32_e64 v36, v36, v38, s4                        // 0000000046e4: d5010024 00124d24
	v_add_co_u32 v158, s4, v34, v158                           // 0000000046ec: d700049e 02033d22
	s_wait_alu depctr_va_sdst(0)                               // 0000000046f4: bf88f19f
	v_add_co_ci_u32_e64 v159, null, v35, v159, s4              // 0000000046f8: d5207c9f 00133f23
	s_and_b32 s4, s3, vcc_lo                                   // 000000004700: 8b046a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000004704: bf88ff9e
	s_and_b32 s4, s22, s4                                      // 000000004708: 8b040416
	global_store_d16_hi_b16 v[158:159], v36, off               // 00000000470c: ee09407c 12000000 0000009e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004718: bf88ff9e
	s_xor_b32 s4, s4, -1                                       // 00000000471c: 8d04c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004720: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000004724: be852004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004728: bf88ff9e
	s_xor_b32 s14, exec_lo, s5                                 // 00000000472c: 8d0e057e
	s_cbranch_execz 124                                        // 000000004730: bfa5007c <tessera_rocm_folded_matmul_708d500594ff51c6+0x2e24>
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[76:77]                  // 000000004734: 7ca89808
	v_or_b32_e32 v32, v74, v163                                // 000000004738: 3841474a
	v_mov_b32_e32 v33, v75                                     // 00000000473c: 7e42034b
	v_or_b32_e32 v34, v74, v162                                // 000000004740: 3845454a
	v_mov_b32_e32 v35, v75                                     // 000000004744: 7e46034b
	v_or_b32_e32 v162, v74, v166                               // 000000004748: 39454d4a
	s_wait_alu depctr_va_vcc(0)                                // 00000000474c: bf88ff9d
	v_cndmask_b32_e32 v76, 0, v76, vcc_lo                      // 000000004750: 02989880
	v_cmp_gt_i64_e64 s4, s[8:9], v[32:33]                      // 000000004754: d4540004 02024008
	v_cndmask_b32_e32 v77, 0, v77, vcc_lo                      // 00000000475c: 029a9a80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[34:35]                  // 000000004760: 7ca84408
	v_or_b32_e32 v36, v74, v165                                // 000000004764: 38494b4a
	v_mov_b32_e32 v37, v75                                     // 000000004768: 7e4a034b
	v_or_b32_e32 v38, v74, v164                                // 00000000476c: 384d494a
	s_wait_alu depctr_va_sdst(0)                               // 000000004770: bf88f19f
	v_cndmask_b32_e64 v33, 0, v75, s4                          // 000000004774: d5010021 00129680
	v_cndmask_b32_e64 v32, 0, v32, s4                          // 00000000477c: d5010020 00124080
	v_lshlrev_b64_e32 v[76:77], 2, v[76:77]                    // 000000004784: 3e989882
	s_wait_alu depctr_va_vcc(0)                                // 000000004788: bf88ff9d
	v_dual_cndmask_b32 v35, 0, v75 :: v_dual_cndmask_b32 v34, 0, v34// 00000000478c: ca529680 23224480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[36:37]                  // 000000004794: 7ca84808
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000004798: 3e404082
	v_mov_b32_e32 v39, v75                                     // 00000000479c: 7e4e034b
	v_or_b32_e32 v114, v74, v167                               // 0000000047a0: 38e54f4a
	v_or_b32_e32 v74, v74, v168                                // 0000000047a4: 3895514a
	v_mov_b32_e32 v115, v75                                    // 0000000047a8: 7ee6034b
	v_add_co_u32 v76, s4, s12, v76                             // 0000000047ac: d700044c 0202980c
	s_wait_alu depctr_va_sdst(0)                               // 0000000047b4: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s13, v77, s4                // 0000000047b8: d5207c4d 00129a0d
	v_add_co_u32 v32, s4, s12, v32                             // 0000000047c0: d7000420 0202400c
	s_wait_alu depctr_va_vcc(0)                                // 0000000047c8: bf88ff9d
	v_dual_cndmask_b32 v37, 0, v75 :: v_dual_cndmask_b32 v36, 0, v36// 0000000047cc: ca529680 25244880
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[38:39]                  // 0000000047d4: 7ca84c08
	v_mov_b32_e32 v163, v75                                    // 0000000047d8: 7f46034b
	s_wait_alu depctr_va_sdst(0)                               // 0000000047dc: bf88f19f
	v_add_co_ci_u32_e64 v33, null, s13, v33, s4                // 0000000047e0: d5207c21 0012420d
	v_cmp_gt_i64_e64 s4, s[8:9], v[114:115]                    // 0000000047e8: d4540004 0202e408
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]                    // 0000000047f0: 3e484882
	s_wait_alu depctr_va_vcc(0)                                // 0000000047f4: bf88ff9d
	v_dual_cndmask_b32 v39, 0, v75 :: v_dual_cndmask_b32 v38, 0, v38// 0000000047f8: ca529680 27264c80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[162:163]                // 000000004800: 7ca94408
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]                    // 000000004804: 3e444482
	s_wait_alu depctr_va_sdst(0)                               // 000000004808: bf88f19f
	v_cndmask_b32_e64 v115, 0, v75, s4                         // 00000000480c: d5010073 00129680
	v_cndmask_b32_e64 v114, 0, v114, s4                        // 000000004814: d5010072 0012e480
	v_add_co_u32 v163, s4, s12, v36                            // 00000000481c: d70004a3 0202480c
	s_wait_alu depctr_va_sdst(0)                               // 000000004824: bf88f19f
	v_add_co_ci_u32_e64 v164, null, s13, v37, s4               // 000000004828: d5207ca4 00124a0d
	v_lshlrev_b64_e32 v[36:37], 2, v[38:39]                    // 000000004830: 3e484c82
	v_lshlrev_b64_e32 v[38:39], 2, v[114:115]                  // 000000004834: 3e4ce482
	s_wait_alu depctr_va_vcc(0)                                // 000000004838: bf88ff9d
	v_dual_cndmask_b32 v115, 0, v75 :: v_dual_cndmask_b32 v114, 0, v162// 00000000483c: ca529680 73734480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[74:75]                  // 000000004844: 7ca89408
	v_add_co_u32 v34, s5, s12, v34                             // 000000004848: d7000522 0202440c
	v_add_co_u32 v165, s4, s12, v36                            // 000000004850: d70004a5 0202480c
	s_wait_alu depctr_va_sdst(0)                               // 000000004858: bf88f19f
	v_add_co_ci_u32_e64 v166, null, s13, v37, s4               // 00000000485c: d5207ca6 00124a0d
	s_wait_alu depctr_va_vcc(0)                                // 000000004864: bf88ff9d
	v_dual_cndmask_b32 v75, 0, v75 :: v_dual_cndmask_b32 v74, 0, v74// 000000004868: ca529680 4b4a9480
	v_lshlrev_b64_e32 v[36:37], 2, v[114:115]                  // 000000004870: 3e48e482
	v_add_co_u32 v114, vcc_lo, s12, v38                        // 000000004874: d7006a72 02024c0c
	s_wait_alu depctr_va_vcc(0)                                // 00000000487c: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, s13, v39, vcc_lo           // 000000004880: d5207c73 01aa4e0d
	v_lshlrev_b64_e32 v[38:39], 2, v[74:75]                    // 000000004888: 3e4c9482
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 00000000488c: bf870244
	v_add_co_u32 v74, vcc_lo, s12, v36                         // 000000004890: d7006a4a 0202480c
	v_add_co_ci_u32_e64 v35, null, s13, v35, s5                // 000000004898: d5207c23 0016460d
	s_wait_alu depctr_va_vcc(0)                                // 0000000048a0: bf88ff9d
	v_add_co_ci_u32_e64 v75, null, s13, v37, vcc_lo            // 0000000048a4: d5207c4b 01aa4a0d
	v_add_co_u32 v167, vcc_lo, s12, v38                        // 0000000048ac: d7006aa7 02024c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000048b4: bf88ff9d
	v_add_co_ci_u32_e64 v168, null, s13, v39, vcc_lo           // 0000000048b8: d5207ca8 01aa4e0d
	s_clause 0x7                                               // 0000000048c0: bf850007
	global_load_b32 v36, v[76:77], off                         // 0000000048c4: ee05007c 00000024 0000004c
	global_load_b32 v37, v[32:33], off                         // 0000000048d0: ee05007c 00000025 00000020
	global_load_b32 v38, v[34:35], off                         // 0000000048dc: ee05007c 00000026 00000022
	global_load_b32 v39, v[163:164], off                       // 0000000048e8: ee05007c 00000027 000000a3
	global_load_b32 v32, v[165:166], off                       // 0000000048f4: ee05007c 00000020 000000a5
	global_load_b32 v33, v[114:115], off                       // 000000004900: ee05007c 00000021 00000072
	global_load_b32 v34, v[74:75], off                         // 00000000490c: ee05007c 00000022 0000004a
	global_load_b32 v35, v[167:168], off                       // 000000004918: ee05007c 00000023 000000a7
	s_wait_alu depctr_sa_sdst(0)                               // 000000004924: bf88ff9e
	s_and_not1_saveexec_b32 s4, s14                            // 000000004928: be84300e
	s_cbranch_execz 14                                         // 00000000492c: bfa5000e <tessera_rocm_folded_matmul_708d500594ff51c6+0x2e68>
	s_wait_loadcnt 0x3                                         // 000000004930: bfc00003
	v_add_co_u32 v32, vcc_lo, s12, v114                        // 000000004934: d7006a20 0202e40c
	s_wait_loadcnt 0x2                                         // 00000000493c: bfc00002
	s_wait_alu depctr_va_vcc(0)                                // 000000004940: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s13, v115, vcc_lo           // 000000004944: d5207c21 01aae60d
	global_load_b128 v[36:39], v[32:33], off                   // 00000000494c: ee05c07c 00000024 00000020
	s_wait_loadcnt 0x1                                         // 000000004958: bfc00001
	global_load_b128 v[32:35], v[32:33], off offset:16         // 00000000495c: ee05c07c 00000020 00001020
	s_wait_alu depctr_sa_sdst(0)                               // 000000004968: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000496c: 8c7e047e
	v_cmp_gt_i64_e32 vcc_lo, s[10:11], v[160:161]              // 000000004970: 7ca9400a
	s_wait_alu depctr_va_vcc(0)                                // 000000004974: bf88ff9d
	v_cndmask_b32_e32 v53, 0, v160, vcc_lo                     // 000000004978: 026b4080
	v_cndmask_b32_e32 v51, 0, v161, vcc_lo                     // 00000000497c: 02674280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000004980: bf870122
	v_add_co_u32 v75, vcc_lo, s16, v53                         // 000000004984: d7006a4b 02026a10
	s_wait_alu depctr_va_vcc(0)                                // 00000000498c: bf88ff9d
	v_add_co_ci_u32_e64 v76, null, s17, v51, vcc_lo            // 000000004990: d5207c4c 01aa6611
	global_load_u8 v51, v[75:76], off                          // 000000004998: ee04007c 00000033 0000004b
	s_wait_loadcnt 0x0                                         // 0000000049a4: bfc00000
	v_lshlrev_b32_e32 v53, 23, v51                             // 0000000049a8: 306a6697
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000049ac: bf870091
	v_mul_f32_e32 v51, v36, v53                                // 0000000049b0: 10666b24
	v_cmp_class_f32_e64 s4, v51, 0x198                         // 0000000049b4: d47e0004 0201ff33 00000198
	v_mul_f32_e32 v51, v24, v51                                // 0000000049c0: 10666718
	s_xor_b32 s5, s4, -1                                       // 0000000049c4: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049c8: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 0000000049cc: be842005
	s_cbranch_execnz 1871                                      // 0000000049d0: bfa6074f <tessera_rocm_folded_matmul_708d500594ff51c6+0x4c10>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000049d8: 8c7e047e
	v_mul_f32_e32 v24, v37, v53                                // 0000000049dc: 10306b25
	s_delay_alu instid0(valu_dep_1)                            // 0000000049e0: bf870001
	v_cmp_class_f32_e64 s4, v24, 0x198                         // 0000000049e4: d47e0004 0201ff18 00000198
	v_mul_f32_e32 v24, v25, v24                                // 0000000049f0: 10303119
	s_xor_b32 s5, s4, -1                                       // 0000000049f4: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049f8: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 0000000049fc: be842005
	s_cbranch_execnz 1876                                      // 000000004a00: bfa60754 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4c54>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a08: 8c7e047e
	v_mul_f32_e32 v25, v38, v53                                // 000000004a0c: 10326b26
	s_delay_alu instid0(valu_dep_1)                            // 000000004a10: bf870001
	v_cmp_class_f32_e64 s4, v25, 0x198                         // 000000004a14: d47e0004 0201ff19 00000198
	v_mul_f32_e32 v25, v26, v25                                // 000000004a20: 1032331a
	s_xor_b32 s5, s4, -1                                       // 000000004a24: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a28: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004a2c: be842005
	s_cbranch_execnz 1881                                      // 000000004a30: bfa60759 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4c98>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a38: 8c7e047e
	v_mul_f32_e32 v26, v39, v53                                // 000000004a3c: 10346b27
	s_delay_alu instid0(valu_dep_1)                            // 000000004a40: bf870001
	v_cmp_class_f32_e64 s4, v26, 0x198                         // 000000004a44: d47e0004 0201ff1a 00000198
	v_mul_f32_e32 v26, v27, v26                                // 000000004a50: 1034351b
	s_xor_b32 s5, s4, -1                                       // 000000004a54: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a58: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004a5c: be842005
	s_cbranch_execnz 1886                                      // 000000004a60: bfa6075e <tessera_rocm_folded_matmul_708d500594ff51c6+0x4cdc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a64: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a68: 8c7e047e
	v_mul_f32_e32 v27, v32, v53                                // 000000004a6c: 10366b20
	s_delay_alu instid0(valu_dep_1)                            // 000000004a70: bf870001
	v_cmp_class_f32_e64 s4, v27, 0x198                         // 000000004a74: d47e0004 0201ff1b 00000198
	v_mul_f32_e32 v27, v28, v27                                // 000000004a80: 1036371c
	s_xor_b32 s5, s4, -1                                       // 000000004a84: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a88: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004a8c: be842005
	s_cbranch_execnz 1891                                      // 000000004a90: bfa60763 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4d20>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a94: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004a98: 8c7e047e
	v_mul_f32_e32 v28, v33, v53                                // 000000004a9c: 10386b21
	s_delay_alu instid0(valu_dep_1)                            // 000000004aa0: bf870001
	v_cmp_class_f32_e64 s4, v28, 0x198                         // 000000004aa4: d47e0004 0201ff1c 00000198
	v_mul_f32_e32 v28, v29, v28                                // 000000004ab0: 1038391d
	s_xor_b32 s5, s4, -1                                       // 000000004ab4: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ab8: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004abc: be842005
	s_cbranch_execnz 1896                                      // 000000004ac0: bfa60768 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4d64>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ac4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004ac8: 8c7e047e
	v_mul_f32_e32 v29, v34, v53                                // 000000004acc: 103a6b22
	s_delay_alu instid0(valu_dep_1)                            // 000000004ad0: bf870001
	v_cmp_class_f32_e64 s4, v29, 0x198                         // 000000004ad4: d47e0004 0201ff1d 00000198
	v_mul_f32_e32 v29, v30, v29                                // 000000004ae0: 103a3b1e
	s_xor_b32 s5, s4, -1                                       // 000000004ae4: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ae8: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004aec: be842005
	s_cbranch_execnz 1901                                      // 000000004af0: bfa6076d <tessera_rocm_folded_matmul_708d500594ff51c6+0x4da8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004af4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004af8: 8c7e047e
	v_mul_f32_e32 v30, v35, v53                                // 000000004afc: 103c6b23
	s_delay_alu instid0(valu_dep_1)                            // 000000004b00: bf870001
	v_cmp_class_f32_e64 s4, v30, 0x198                         // 000000004b04: d47e0004 0201ff1e 00000198
	v_mul_f32_e32 v30, v31, v30                                // 000000004b10: 103c3d1f
	s_xor_b32 s5, s4, -1                                       // 000000004b14: 8d05c104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b18: bf88ff9e
	s_and_saveexec_b32 s4, s5                                  // 000000004b1c: be842005
	s_cbranch_execnz 1906                                      // 000000004b20: bfa60772 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4dec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004b28: 8c7e047e
	v_bfe_u32 v31, v51, 16, 1                                  // 000000004b2c: d610001f 02052133
	v_bfe_u32 v32, v24, 16, 1                                  // 000000004b34: d6100020 02052118
	v_or_b32_e32 v33, 0x400000, v51                            // 000000004b3c: 384266ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000004b44: 7c306733
	v_or_b32_e32 v34, 0x400000, v24                            // 000000004b48: 384430ff 00400000
	v_add3_u32 v31, v31, v51, 0x7fff                           // 000000004b50: d655001f 03fe671f 00007fff
	v_add3_u32 v32, v32, v24, 0x7fff                           // 000000004b5c: d6550020 03fe3120 00007fff
	v_bfe_u32 v35, v25, 16, 1                                  // 000000004b68: d6100023 02052119
	s_and_b32 s0, s3, s0                                       // 000000004b70: 8b000003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b74: bf88ff9e
	s_and_b32 s0, s22, s0                                      // 000000004b78: 8b000016
	s_wait_alu depctr_va_vcc(0)                                // 000000004b7c: bf88ff9d
	v_cndmask_b32_e32 v31, v31, v33, vcc_lo                    // 000000004b80: 023e431f
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000004b84: 7c303118
	v_bfe_u32 v33, v27, 16, 1                                  // 000000004b88: d6100021 0205211b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b90: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 000000004b94: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 000000004b98: bf88ff9d
	v_cndmask_b32_e32 v24, v32, v34, vcc_lo                    // 000000004b9c: 02304520
	v_bfe_u32 v32, v26, 16, 1                                  // 000000004ba0: d6100020 0205211a
	v_add3_u32 v34, v35, v25, 0x7fff                           // 000000004ba8: d6550022 03fe3323 00007fff
	s_clause 0x1                                               // 000000004bb4: bf850001
	global_store_d16_hi_b16 v[78:79], v31, off offset:32       // 000000004bb8: ee09407c 0f800000 0000204e
	global_store_d16_hi_b16 v[82:83], v24, off offset:32       // 000000004bc4: ee09407c 0c000000 00002052
	v_or_b32_e32 v24, 0x400000, v25                            // 000000004bd0: 383032ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 000000004bd8: 7c303319
	v_add3_u32 v31, v32, v26, 0x7fff                           // 000000004bdc: d655001f 03fe3520 00007fff
	v_or_b32_e32 v32, 0x400000, v26                            // 000000004be8: 384034ff 00400000
	v_add3_u32 v33, v33, v27, 0x7fff                           // 000000004bf0: d6550021 03fe3721 00007fff
	v_or_b32_e32 v35, 0x400000, v27                            // 000000004bfc: 384636ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c04: bf88ff9d
	v_cndmask_b32_e32 v24, v34, v24, vcc_lo                    // 000000004c08: 02303122
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 000000004c0c: 7c30351a
	global_store_d16_hi_b16 v[84:85], v24, off offset:32       // 000000004c10: ee09407c 0c000000 00002054
	s_wait_alu depctr_va_vcc(0)                                // 000000004c1c: bf88ff9d
	v_cndmask_b32_e32 v25, v31, v32, vcc_lo                    // 000000004c20: 0232411f
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000004c24: 7c30371b
	v_bfe_u32 v24, v28, 16, 1                                  // 000000004c28: d6100018 0205211c
	v_or_b32_e32 v27, 0x400000, v28                            // 000000004c30: 383638ff 00400000
	v_or_b32_e32 v31, 0x400000, v29                            // 000000004c38: 383e3aff 00400000
	v_or_b32_e32 v32, 0x400000, v30                            // 000000004c40: 38403cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c48: bf88ff9d
	v_cndmask_b32_e32 v26, v33, v35, vcc_lo                    // 000000004c4c: 02344721
	global_store_d16_hi_b16 v[90:91], v25, off offset:32       // 000000004c50: ee09407c 0c800000 0000205a
	v_bfe_u32 v25, v29, 16, 1                                  // 000000004c5c: d6100019 0205211d
	v_add3_u32 v24, v24, v28, 0x7fff                           // 000000004c64: d6550018 03fe3918 00007fff
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 000000004c70: 7c30391c
	global_store_d16_hi_b16 v[96:97], v26, off offset:32       // 000000004c74: ee09407c 0d000000 00002060
	v_bfe_u32 v26, v30, 16, 1                                  // 000000004c80: d610001a 0205211e
	v_add3_u32 v25, v25, v29, 0x7fff                           // 000000004c88: d6550019 03fe3b19 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004c94: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v27, vcc_lo                    // 000000004c98: 02303718
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000004c9c: 7c303b1d
	v_add3_u32 v26, v26, v30, 0x7fff                           // 000000004ca0: d655001a 03fe3d1a 00007fff
	global_store_d16_hi_b16 v[94:95], v24, off offset:32       // 000000004cac: ee09407c 0c000000 0000205e
	s_wait_alu depctr_va_vcc(0)                                // 000000004cb8: bf88ff9d
	v_cndmask_b32_e32 v25, v25, v31, vcc_lo                    // 000000004cbc: 02323f19
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000004cc0: 7c303d1e
	global_store_d16_hi_b16 v[98:99], v25, off offset:32       // 000000004cc4: ee09407c 0c800000 00002062
	s_wait_alu depctr_va_vcc(0)                                // 000000004cd0: bf88ff9d
	v_cndmask_b32_e32 v26, v26, v32, vcc_lo                    // 000000004cd4: 0234411a
	global_store_d16_hi_b16 v[100:101], v26, off offset:32     // 000000004cd8: ee09407c 0d000000 00002064
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ce4: bf88ff9e
	s_and_saveexec_b32 s4, s0                                  // 000000004ce8: be842000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cec: bf88ff9e
	s_xor_b32 s5, exec_lo, s4                                  // 000000004cf0: 8d05047e
	s_cbranch_execz 118                                        // 000000004cf4: bfa50076 <tessera_rocm_folded_matmul_708d500594ff51c6+0x33d0>
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[86:87]                  // 000000004cf8: 7ca8ac08
	v_mov_b32_e32 v89, v65                                     // 000000004cfc: 7eb20341
	v_mov_b32_e32 v93, v65                                     // 000000004d00: 7eba0341
	v_mov_b32_e32 v81, v65                                     // 000000004d04: 7ea20341
	v_mov_b32_e32 v71, v65                                     // 000000004d08: 7e8e0341
	v_mov_b32_e32 v69, v65                                     // 000000004d0c: 7e8a0341
	s_wait_alu depctr_va_vcc(0)                                // 000000004d10: bf88ff9d
	v_dual_cndmask_b32 v25, 0, v87 :: v_dual_cndmask_b32 v24, 0, v86// 000000004d14: ca52ae80 1918ac80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[88:89]                  // 000000004d1c: 7ca8b008
	v_cmp_gt_i64_e64 s0, s[8:9], v[92:93]                      // 000000004d20: d4540000 0202b808
	v_mov_b32_e32 v67, v65                                     // 000000004d28: 7e860341
	s_wait_alu depctr_va_vcc(0)                                // 000000004d2c: bf88ff9d
	v_dual_cndmask_b32 v29, 0, v65 :: v_dual_cndmask_b32 v28, 0, v88// 000000004d30: ca528280 1d1cb080
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[80:81]                  // 000000004d38: 7ca8a008
	s_wait_alu depctr_va_sdst(0)                               // 000000004d3c: bf88f19f
	v_cndmask_b32_e64 v27, 0, v65, s0                          // 000000004d40: d501001b 00028280
	v_cndmask_b32_e64 v26, 0, v92, s0                          // 000000004d48: d501001a 0002b880
	v_lshlrev_b64_e32 v[28:29], 2, v[28:29]                    // 000000004d50: 3e383882
	s_wait_alu depctr_va_vcc(0)                                // 000000004d54: bf88ff9d
	v_cndmask_b32_e32 v30, 0, v80, vcc_lo                      // 000000004d58: 023ca080
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]                    // 000000004d5c: 3e303082
	v_lshlrev_b64_e32 v[26:27], 2, v[26:27]                    // 000000004d60: 3e343482
	v_cndmask_b32_e32 v31, 0, v65, vcc_lo                      // 000000004d64: 023e8280
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[70:71]                  // 000000004d68: 7ca88c08
	v_add_co_u32 v32, s4, s12, v28                             // 000000004d6c: d7000420 0202380c
	v_add_co_u32 v24, s0, s12, v24                             // 000000004d74: d7000018 0202300c
	s_wait_alu depctr_va_sdst(0)                               // 000000004d7c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s13, v25, s0                // 000000004d80: d5207c19 0002320d
	v_add_co_u32 v26, s0, s12, v26                             // 000000004d88: d700001a 0202340c
	s_wait_alu depctr_va_sdst(0)                               // 000000004d90: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s13, v27, s0                // 000000004d94: d5207c1b 0002360d
	v_cmp_gt_i64_e64 s0, s[8:9], v[68:69]                      // 000000004d9c: d4540000 02028808
	v_add_co_ci_u32_e64 v33, null, s13, v29, s4                // 000000004da4: d5207c21 00123a0d
	v_lshlrev_b64_e32 v[28:29], 2, v[30:31]                    // 000000004dac: 3e383c82
	s_wait_alu depctr_va_vcc(0)                                // 000000004db0: bf88ff9d
	v_dual_cndmask_b32 v31, 0, v65 :: v_dual_cndmask_b32 v30, 0, v70// 000000004db4: ca528280 1f1e8c80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[66:67]                  // 000000004dbc: 7ca88408
	s_wait_alu depctr_va_sdst(0)                               // 000000004dc0: bf88f19f
	v_cndmask_b32_e64 v35, 0, v65, s0                          // 000000004dc4: d5010023 00028280
	v_cndmask_b32_e64 v34, 0, v68, s0                          // 000000004dcc: d5010022 00028880
	v_add_co_u32 v36, s0, s12, v28                             // 000000004dd4: d7000024 0202380c
	s_wait_alu depctr_va_sdst(0)                               // 000000004ddc: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s13, v29, s0                // 000000004de0: d5207c25 00023a0d
	v_lshlrev_b64_e32 v[28:29], 2, v[30:31]                    // 000000004de8: 3e383c82
	v_lshlrev_b64_e32 v[30:31], 2, v[34:35]                    // 000000004dec: 3e3c4482
	s_wait_alu depctr_va_vcc(0)                                // 000000004df0: bf88ff9d
	v_dual_cndmask_b32 v35, 0, v65 :: v_dual_cndmask_b32 v34, 0, v66// 000000004df4: ca528280 23228480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[64:65]                  // 000000004dfc: 7ca88008
	s_delay_alu instid0(valu_dep_4)                            // 000000004e00: bf870004
	v_add_co_u32 v38, s0, s12, v28                             // 000000004e04: d7000026 0202380c
	s_wait_alu depctr_va_sdst(0)                               // 000000004e0c: bf88f19f
	v_add_co_ci_u32_e64 v39, null, s13, v29, s0                // 000000004e10: d5207c27 00023a0d
	v_lshlrev_b64_e32 v[28:29], 2, v[34:35]                    // 000000004e18: 3e384482
	s_wait_alu depctr_va_vcc(0)                                // 000000004e1c: bf88ff9d
	v_dual_cndmask_b32 v35, 0, v65 :: v_dual_cndmask_b32 v34, 0, v64// 000000004e20: ca528280 23228080
	v_add_co_u32 v63, vcc_lo, s12, v30                         // 000000004e28: d7006a3f 02023c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000004e30: bf88ff9d
	v_add_co_ci_u32_e64 v64, null, s13, v31, vcc_lo            // 000000004e34: d5207c40 01aa3e0d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000004e3c: bf8701c3
	v_lshlrev_b64_e32 v[30:31], 2, v[34:35]                    // 000000004e40: 3e3c4482
	v_add_co_u32 v34, vcc_lo, s12, v28                         // 000000004e44: d7006a22 0202380c
	s_wait_alu depctr_va_vcc(0)                                // 000000004e4c: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s13, v29, vcc_lo            // 000000004e50: d5207c23 01aa3a0d
	v_add_co_u32 v65, vcc_lo, s12, v30                         // 000000004e58: d7006a41 02023c0c
	s_wait_alu depctr_va_vcc(0)                                // 000000004e60: bf88ff9d
	v_add_co_ci_u32_e64 v66, null, s13, v31, vcc_lo            // 000000004e64: d5207c42 01aa3e0d
	s_clause 0x7                                               // 000000004e6c: bf850007
	global_load_b32 v28, v[24:25], off                         // 000000004e70: ee05007c 0000001c 00000018
	global_load_b32 v29, v[26:27], off                         // 000000004e7c: ee05007c 0000001d 0000001a
	global_load_b32 v30, v[32:33], off                         // 000000004e88: ee05007c 0000001e 00000020
	global_load_b32 v31, v[36:37], off                         // 000000004e94: ee05007c 0000001f 00000024
	global_load_b32 v24, v[38:39], off                         // 000000004ea0: ee05007c 00000018 00000026
	global_load_b32 v25, v[63:64], off                         // 000000004eac: ee05007c 00000019 0000003f
	global_load_b32 v26, v[34:35], off                         // 000000004eb8: ee05007c 0000001a 00000022
	global_load_b32 v27, v[65:66], off                         // 000000004ec4: ee05007c 0000001b 00000041
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ed0: bf88ff9e
	s_and_not1_saveexec_b32 s0, s5                             // 000000004ed4: be803005
	s_cbranch_execz 28                                         // 000000004ed8: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x344c>
	s_wait_loadcnt 0x3                                         // 000000004edc: bfc00003
	v_add_co_u32 v24, s4, s6, v72                              // 000000004ee0: d7000418 02029006
	s_wait_loadcnt 0x2                                         // 000000004ee8: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000004eec: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s7, 0, s4                   // 000000004ef0: d5207c19 00110007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000004ef8: bf870122
	v_add_co_u32 v24, vcc_lo, v24, v73                         // 000000004efc: d7006a18 02029318
	s_wait_alu depctr_va_vcc(0)                                // 000000004f04: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, 0, v25, vcc_lo              // 000000004f08: d5207c19 01aa3280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004f10: bf870091
	v_lshlrev_b64_e32 v[24:25], 2, v[24:25]                    // 000000004f14: 3e303082
	v_add_co_u32 v24, vcc_lo, s12, v24                         // 000000004f18: d7006a18 0202300c
	s_wait_alu depctr_va_vcc(0)                                // 000000004f20: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000004f24: bf870002
	v_add_co_ci_u32_e64 v25, null, s13, v25, vcc_lo            // 000000004f28: d5207c19 01aa320d
	global_load_b128 v[28:31], v[24:25], off offset:64         // 000000004f30: ee05c07c 0000001c 00004018
	s_wait_loadcnt 0x1                                         // 000000004f3c: bfc00001
	global_load_b128 v[24:27], v[24:25], off offset:80         // 000000004f40: ee05c07c 00000018 00005018
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004f50: 8c7e007e
	global_load_u8 v32, v[75:76], off                          // 000000004f54: ee04007c 00000020 0000004b
	s_wait_loadcnt 0x0                                         // 000000004f60: bfc00000
	v_lshlrev_b32_e32 v33, 23, v32                             // 000000004f64: 30424097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004f68: bf870091
	v_mul_f32_e32 v32, v28, v33                                // 000000004f6c: 1040431c
	v_cmp_class_f32_e64 s0, v32, 0x198                         // 000000004f70: d47e0000 0201ff20 00000198
	v_mul_f32_e32 v32, v16, v32                                // 000000004f7c: 10404110
	s_xor_b32 s4, s0, -1                                       // 000000004f80: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f84: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000004f88: be802004
	s_cbranch_execnz 1640                                      // 000000004f8c: bfa60668 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4e30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f90: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004f94: 8c7e007e
	v_mul_f32_e32 v16, v29, v33                                // 000000004f98: 1020431d
	s_delay_alu instid0(valu_dep_1)                            // 000000004f9c: bf870001
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 000000004fa0: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v17, v16                                // 000000004fac: 10202111
	s_xor_b32 s4, s0, -1                                       // 000000004fb0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fb4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000004fb8: be802004
	s_cbranch_execnz 1645                                      // 000000004fbc: bfa6066d <tessera_rocm_folded_matmul_708d500594ff51c6+0x4e74>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fc0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004fc4: 8c7e007e
	v_mul_f32_e32 v17, v30, v33                                // 000000004fc8: 1022431e
	s_delay_alu instid0(valu_dep_1)                            // 000000004fcc: bf870001
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 000000004fd0: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v18, v17                                // 000000004fdc: 10222312
	s_xor_b32 s4, s0, -1                                       // 000000004fe0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fe4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000004fe8: be802004
	s_cbranch_execnz 1650                                      // 000000004fec: bfa60672 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4eb8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ff0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004ff4: 8c7e007e
	v_mul_f32_e32 v18, v31, v33                                // 000000004ff8: 1024431f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ffc: bf870001
	v_cmp_class_f32_e64 s0, v18, 0x198                         // 000000005000: d47e0000 0201ff12 00000198
	v_mul_f32_e32 v18, v19, v18                                // 00000000500c: 10242513
	s_xor_b32 s4, s0, -1                                       // 000000005010: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005014: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005018: be802004
	s_cbranch_execnz 1655                                      // 00000000501c: bfa60677 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4efc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005020: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005024: 8c7e007e
	v_mul_f32_e32 v19, v24, v33                                // 000000005028: 10264318
	s_delay_alu instid0(valu_dep_1)                            // 00000000502c: bf870001
	v_cmp_class_f32_e64 s0, v19, 0x198                         // 000000005030: d47e0000 0201ff13 00000198
	v_mul_f32_e32 v19, v20, v19                                // 00000000503c: 10262714
	s_xor_b32 s4, s0, -1                                       // 000000005040: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005044: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005048: be802004
	s_cbranch_execnz 1660                                      // 00000000504c: bfa6067c <tessera_rocm_folded_matmul_708d500594ff51c6+0x4f40>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005050: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005054: 8c7e007e
	v_mul_f32_e32 v20, v25, v33                                // 000000005058: 10284319
	s_delay_alu instid0(valu_dep_1)                            // 00000000505c: bf870001
	v_cmp_class_f32_e64 s0, v20, 0x198                         // 000000005060: d47e0000 0201ff14 00000198
	v_mul_f32_e32 v20, v21, v20                                // 00000000506c: 10282915
	s_xor_b32 s4, s0, -1                                       // 000000005070: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005074: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 000000005078: be802004
	s_cbranch_execnz 1665                                      // 00000000507c: bfa60681 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4f84>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005080: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005084: 8c7e007e
	v_mul_f32_e32 v21, v26, v33                                // 000000005088: 102a431a
	s_delay_alu instid0(valu_dep_1)                            // 00000000508c: bf870001
	v_cmp_class_f32_e64 s0, v21, 0x198                         // 000000005090: d47e0000 0201ff15 00000198
	v_mul_f32_e32 v21, v22, v21                                // 00000000509c: 102a2b16
	s_xor_b32 s4, s0, -1                                       // 0000000050a0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050a4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 0000000050a8: be802004
	s_cbranch_execnz 1670                                      // 0000000050ac: bfa60686 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4fc8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050b4: 8c7e007e
	v_mul_f32_e32 v22, v27, v33                                // 0000000050b8: 102c431b
	s_delay_alu instid0(valu_dep_1)                            // 0000000050bc: bf870001
	v_cmp_class_f32_e64 s0, v22, 0x198                         // 0000000050c0: d47e0000 0201ff16 00000198
	v_mul_f32_e32 v22, v23, v22                                // 0000000050cc: 102c2d17
	s_xor_b32 s4, s0, -1                                       // 0000000050d0: 8d04c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050d4: bf88ff9e
	s_and_saveexec_b32 s0, s4                                  // 0000000050d8: be802004
	s_cbranch_execnz 1675                                      // 0000000050dc: bfa6068b <tessera_rocm_folded_matmul_708d500594ff51c6+0x500c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050e4: 8c7e007e
	v_bfe_u32 v23, v32, 16, 1                                  // 0000000050e8: d6100017 02052120
	v_bfe_u32 v24, v16, 16, 1                                  // 0000000050f0: d6100018 02052110
	v_or_b32_e32 v25, 0x400000, v32                            // 0000000050f8: 383240ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000005100: 7c304120
	v_or_b32_e32 v26, 0x400000, v16                            // 000000005104: 383420ff 00400000
	v_add3_u32 v23, v23, v32, 0x7fff                           // 00000000510c: d6550017 03fe4117 00007fff
	v_add3_u32 v24, v24, v16, 0x7fff                           // 000000005118: d6550018 03fe2118 00007fff
	v_bfe_u32 v27, v17, 16, 1                                  // 000000005124: d610001b 02052111
	s_and_b32 s0, s3, s1                                       // 00000000512c: 8b000103
	s_wait_alu depctr_sa_sdst(0)                               // 000000005130: bf88ff9e
	s_and_b32 s0, s22, s0                                      // 000000005134: 8b000016
	s_wait_alu depctr_va_vcc(0)                                // 000000005138: bf88ff9d
	v_cndmask_b32_e32 v23, v23, v25, vcc_lo                    // 00000000513c: 022e3317
	v_cmp_u_f32_e32 vcc_lo, v16, v16                           // 000000005140: 7c302110
	v_bfe_u32 v25, v19, 16, 1                                  // 000000005144: d6100019 02052113
	s_wait_alu depctr_sa_sdst(0)                               // 00000000514c: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 000000005150: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 000000005154: bf88ff9d
	v_cndmask_b32_e32 v16, v24, v26, vcc_lo                    // 000000005158: 02203518
	v_bfe_u32 v24, v18, 16, 1                                  // 00000000515c: d6100018 02052112
	v_add3_u32 v26, v27, v17, 0x7fff                           // 000000005164: d655001a 03fe231b 00007fff
	s_clause 0x1                                               // 000000005170: bf850001
	global_store_d16_hi_b16 v[104:105], v23, off offset:32     // 000000005174: ee09407c 0b800000 00002068
	global_store_d16_hi_b16 v[110:111], v16, off offset:32     // 000000005180: ee09407c 08000000 0000206e
	v_or_b32_e32 v16, 0x400000, v17                            // 00000000518c: 382022ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v17, v17                           // 000000005194: 7c302311
	v_add3_u32 v23, v24, v18, 0x7fff                           // 000000005198: d6550017 03fe2518 00007fff
	v_or_b32_e32 v24, 0x400000, v18                            // 0000000051a4: 383024ff 00400000
	v_add3_u32 v25, v25, v19, 0x7fff                           // 0000000051ac: d6550019 03fe2719 00007fff
	v_or_b32_e32 v27, 0x400000, v19                            // 0000000051b8: 383626ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000051c0: bf88ff9d
	v_cndmask_b32_e32 v16, v26, v16, vcc_lo                    // 0000000051c4: 0220211a
	v_cmp_u_f32_e32 vcc_lo, v18, v18                           // 0000000051c8: 7c302512
	global_store_d16_hi_b16 v[112:113], v16, off offset:32     // 0000000051cc: ee09407c 08000000 00002070
	s_wait_alu depctr_va_vcc(0)                                // 0000000051d8: bf88ff9d
	v_cndmask_b32_e32 v17, v23, v24, vcc_lo                    // 0000000051dc: 02223117
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 0000000051e0: 7c302713
	v_bfe_u32 v16, v20, 16, 1                                  // 0000000051e4: d6100010 02052114
	v_or_b32_e32 v19, 0x400000, v20                            // 0000000051ec: 382628ff 00400000
	v_or_b32_e32 v23, 0x400000, v21                            // 0000000051f4: 382e2aff 00400000
	v_or_b32_e32 v24, 0x400000, v22                            // 0000000051fc: 38302cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005204: bf88ff9d
	v_cndmask_b32_e32 v18, v25, v27, vcc_lo                    // 000000005208: 02243719
	global_store_d16_hi_b16 v[118:119], v17, off offset:32     // 00000000520c: ee09407c 08800000 00002076
	v_bfe_u32 v17, v21, 16, 1                                  // 000000005218: d6100011 02052115
	v_add3_u32 v16, v16, v20, 0x7fff                           // 000000005220: d6550010 03fe2910 00007fff
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 00000000522c: 7c302914
	global_store_d16_hi_b16 v[122:123], v18, off offset:32     // 000000005230: ee09407c 09000000 0000207a
	v_bfe_u32 v18, v22, 16, 1                                  // 00000000523c: d6100012 02052116
	v_add3_u32 v17, v17, v21, 0x7fff                           // 000000005244: d6550011 03fe2b11 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005250: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v19, vcc_lo                    // 000000005254: 02202710
	v_cmp_u_f32_e32 vcc_lo, v21, v21                           // 000000005258: 7c302b15
	v_add3_u32 v18, v18, v22, 0x7fff                           // 00000000525c: d6550012 03fe2d12 00007fff
	global_store_d16_hi_b16 v[120:121], v16, off offset:32     // 000000005268: ee09407c 08000000 00002078
	s_wait_alu depctr_va_vcc(0)                                // 000000005274: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v23, vcc_lo                    // 000000005278: 02222f11
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 00000000527c: 7c302d16
	global_store_d16_hi_b16 v[124:125], v17, off offset:32     // 000000005280: ee09407c 08800000 0000207c
	s_wait_alu depctr_va_vcc(0)                                // 00000000528c: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo                    // 000000005290: 02243112
	global_store_d16_hi_b16 v[126:127], v18, off offset:32     // 000000005294: ee09407c 09000000 0000207e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000052a4: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052a8: bf88ff9e
	s_xor_b32 s4, exec_lo, s1                                  // 0000000052ac: 8d04017e
	s_cbranch_execz 118                                        // 0000000052b0: bfa50076 <tessera_rocm_folded_matmul_708d500594ff51c6+0x398c>
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[106:107]                // 0000000052b4: 7ca8d408
	v_mov_b32_e32 v109, v57                                    // 0000000052b8: 7eda0339
	v_mov_b32_e32 v117, v57                                    // 0000000052bc: 7eea0339
	v_mov_b32_e32 v103, v57                                    // 0000000052c0: 7ece0339
	v_mov_b32_e32 v63, v57                                     // 0000000052c4: 7e7e0339
	v_mov_b32_e32 v61, v57                                     // 0000000052c8: 7e7a0339
	s_wait_alu depctr_va_vcc(0)                                // 0000000052cc: bf88ff9d
	v_dual_cndmask_b32 v17, 0, v107 :: v_dual_cndmask_b32 v16, 0, v106// 0000000052d0: ca52d680 1110d480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[108:109]                // 0000000052d8: 7ca8d808
	v_cmp_gt_i64_e64 s0, s[8:9], v[116:117]                    // 0000000052dc: d4540000 0202e808
	v_mov_b32_e32 v59, v57                                     // 0000000052e4: 7e760339
	s_wait_alu depctr_va_vcc(0)                                // 0000000052e8: bf88ff9d
	v_dual_cndmask_b32 v21, 0, v57 :: v_dual_cndmask_b32 v20, 0, v108// 0000000052ec: ca527280 1514d880
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[102:103]                // 0000000052f4: 7ca8cc08
	s_wait_alu depctr_va_sdst(0)                               // 0000000052f8: bf88f19f
	v_cndmask_b32_e64 v19, 0, v57, s0                          // 0000000052fc: d5010013 00027280
	v_cndmask_b32_e64 v18, 0, v116, s0                         // 000000005304: d5010012 0002e880
	v_lshlrev_b64_e32 v[20:21], 2, v[20:21]                    // 00000000530c: 3e282882
	s_wait_alu depctr_va_vcc(0)                                // 000000005310: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v102, vcc_lo                     // 000000005314: 022ccc80
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]                    // 000000005318: 3e202082
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]                    // 00000000531c: 3e242482
	v_cndmask_b32_e32 v23, 0, v57, vcc_lo                      // 000000005320: 022e7280
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[62:63]                  // 000000005324: 7ca87c08
	v_add_co_u32 v24, s1, s12, v20                             // 000000005328: d7000118 0202280c
	v_add_co_u32 v16, s0, s12, v16                             // 000000005330: d7000010 0202200c
	s_wait_alu depctr_va_sdst(0)                               // 000000005338: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s13, v17, s0                // 00000000533c: d5207c11 0002220d
	v_add_co_u32 v18, s0, s12, v18                             // 000000005344: d7000012 0202240c
	s_wait_alu depctr_va_sdst(0)                               // 00000000534c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s13, v19, s0                // 000000005350: d5207c13 0002260d
	v_cmp_gt_i64_e64 s0, s[8:9], v[60:61]                      // 000000005358: d4540000 02027808
	v_add_co_ci_u32_e64 v25, null, s13, v21, s1                // 000000005360: d5207c19 00062a0d
	v_lshlrev_b64_e32 v[20:21], 2, v[22:23]                    // 000000005368: 3e282c82
	s_wait_alu depctr_va_vcc(0)                                // 00000000536c: bf88ff9d
	v_dual_cndmask_b32 v23, 0, v57 :: v_dual_cndmask_b32 v22, 0, v62// 000000005370: ca527280 17167c80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[58:59]                  // 000000005378: 7ca87408
	s_wait_alu depctr_va_sdst(0)                               // 00000000537c: bf88f19f
	v_cndmask_b32_e64 v27, 0, v57, s0                          // 000000005380: d501001b 00027280
	v_cndmask_b32_e64 v26, 0, v60, s0                          // 000000005388: d501001a 00027880
	v_add_co_u32 v28, s0, s12, v20                             // 000000005390: d700001c 0202280c
	s_wait_alu depctr_va_sdst(0)                               // 000000005398: bf88f19f
	v_add_co_ci_u32_e64 v29, null, s13, v21, s0                // 00000000539c: d5207c1d 00022a0d
	v_lshlrev_b64_e32 v[20:21], 2, v[22:23]                    // 0000000053a4: 3e282c82
	v_lshlrev_b64_e32 v[22:23], 2, v[26:27]                    // 0000000053a8: 3e2c3482
	s_wait_alu depctr_va_vcc(0)                                // 0000000053ac: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v57 :: v_dual_cndmask_b32 v26, 0, v58// 0000000053b0: ca527280 1b1a7480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[56:57]                  // 0000000053b8: 7ca87008
	s_delay_alu instid0(valu_dep_4)                            // 0000000053bc: bf870004
	v_add_co_u32 v30, s0, s12, v20                             // 0000000053c0: d700001e 0202280c
	s_wait_alu depctr_va_sdst(0)                               // 0000000053c8: bf88f19f
	v_add_co_ci_u32_e64 v31, null, s13, v21, s0                // 0000000053cc: d5207c1f 00022a0d
	v_lshlrev_b64_e32 v[20:21], 2, v[26:27]                    // 0000000053d4: 3e283482
	s_wait_alu depctr_va_vcc(0)                                // 0000000053d8: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v57 :: v_dual_cndmask_b32 v26, 0, v56// 0000000053dc: ca527280 1b1a7080
	v_add_co_u32 v32, vcc_lo, s12, v22                         // 0000000053e4: d7006a20 02022c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000053ec: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s13, v23, vcc_lo            // 0000000053f0: d5207c21 01aa2e0d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 0000000053f8: bf8701c3
	v_lshlrev_b64_e32 v[22:23], 2, v[26:27]                    // 0000000053fc: 3e2c3482
	v_add_co_u32 v26, vcc_lo, s12, v20                         // 000000005400: d7006a1a 0202280c
	s_wait_alu depctr_va_vcc(0)                                // 000000005408: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, s13, v21, vcc_lo            // 00000000540c: d5207c1b 01aa2a0d
	v_add_co_u32 v34, vcc_lo, s12, v22                         // 000000005414: d7006a22 02022c0c
	s_wait_alu depctr_va_vcc(0)                                // 00000000541c: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s13, v23, vcc_lo            // 000000005420: d5207c23 01aa2e0d
	s_clause 0x7                                               // 000000005428: bf850007
	global_load_b32 v20, v[16:17], off                         // 00000000542c: ee05007c 00000014 00000010
	global_load_b32 v21, v[18:19], off                         // 000000005438: ee05007c 00000015 00000012
	global_load_b32 v22, v[24:25], off                         // 000000005444: ee05007c 00000016 00000018
	global_load_b32 v23, v[28:29], off                         // 000000005450: ee05007c 00000017 0000001c
	global_load_b32 v16, v[30:31], off                         // 00000000545c: ee05007c 00000010 0000001e
	global_load_b32 v17, v[32:33], off                         // 000000005468: ee05007c 00000011 00000020
	global_load_b32 v18, v[26:27], off                         // 000000005474: ee05007c 00000012 0000001a
	global_load_b32 v19, v[34:35], off                         // 000000005480: ee05007c 00000013 00000022
	s_wait_alu depctr_sa_sdst(0)                               // 00000000548c: bf88ff9e
	s_and_not1_saveexec_b32 s0, s4                             // 000000005490: be803004
	s_cbranch_execz 28                                         // 000000005494: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x3a08>
	s_wait_loadcnt 0x3                                         // 000000005498: bfc00003
	v_add_co_u32 v16, s1, s6, v72                              // 00000000549c: d7000110 02029006
	s_wait_loadcnt 0x2                                         // 0000000054a4: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 0000000054a8: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s7, 0, s1                   // 0000000054ac: d5207c11 00050007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000054b4: bf870122
	v_add_co_u32 v16, vcc_lo, v16, v73                         // 0000000054b8: d7006a10 02029310
	s_wait_alu depctr_va_vcc(0)                                // 0000000054c0: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, 0, v17, vcc_lo              // 0000000054c4: d5207c11 01aa2280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000054cc: bf870091
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]                    // 0000000054d0: 3e202082
	v_add_co_u32 v16, vcc_lo, s12, v16                         // 0000000054d4: d7006a10 0202200c
	s_wait_alu depctr_va_vcc(0)                                // 0000000054dc: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 0000000054e0: bf870002
	v_add_co_ci_u32_e64 v17, null, s13, v17, vcc_lo            // 0000000054e4: d5207c11 01aa220d
	global_load_b128 v[20:23], v[16:17], off offset:128        // 0000000054ec: ee05c07c 00000014 00008010
	s_wait_loadcnt 0x1                                         // 0000000054f8: bfc00001
	global_load_b128 v[16:19], v[16:17], off offset:144        // 0000000054fc: ee05c07c 00000010 00009010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005508: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000550c: 8c7e007e
	global_load_u8 v24, v[75:76], off                          // 000000005510: ee04007c 00000018 0000004b
	s_wait_loadcnt 0x0                                         // 00000000551c: bfc00000
	v_lshlrev_b32_e32 v25, 23, v24                             // 000000005520: 30323097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005524: bf870091
	v_mul_f32_e32 v24, v20, v25                                // 000000005528: 10303314
	v_cmp_class_f32_e64 s0, v24, 0x198                         // 00000000552c: d47e0000 0201ff18 00000198
	v_mul_f32_e32 v24, v8, v24                                 // 000000005538: 10303108
	s_xor_b32 s1, s0, -1                                       // 00000000553c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005540: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005544: be802001
	s_cbranch_execnz 1409                                      // 000000005548: bfa60581 <tessera_rocm_folded_matmul_708d500594ff51c6+0x5050>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000554c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005550: 8c7e007e
	v_mul_f32_e32 v8, v21, v25                                 // 000000005554: 10103315
	s_delay_alu instid0(valu_dep_1)                            // 000000005558: bf870001
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 00000000555c: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v9, v8                                   // 000000005568: 10101109
	s_xor_b32 s1, s0, -1                                       // 00000000556c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005570: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005574: be802001
	s_cbranch_execnz 1414                                      // 000000005578: bfa60586 <tessera_rocm_folded_matmul_708d500594ff51c6+0x5094>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000557c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005580: 8c7e007e
	v_mul_f32_e32 v9, v22, v25                                 // 000000005584: 10123316
	s_delay_alu instid0(valu_dep_1)                            // 000000005588: bf870001
	v_cmp_class_f32_e64 s0, v9, 0x198                          // 00000000558c: d47e0000 0201ff09 00000198
	v_mul_f32_e32 v9, v10, v9                                  // 000000005598: 1012130a
	s_xor_b32 s1, s0, -1                                       // 00000000559c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055a0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000055a4: be802001
	s_cbranch_execnz 1419                                      // 0000000055a8: bfa6058b <tessera_rocm_folded_matmul_708d500594ff51c6+0x50d8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000055b0: 8c7e007e
	v_mul_f32_e32 v10, v23, v25                                // 0000000055b4: 10143317
	s_delay_alu instid0(valu_dep_1)                            // 0000000055b8: bf870001
	v_cmp_class_f32_e64 s0, v10, 0x198                         // 0000000055bc: d47e0000 0201ff0a 00000198
	v_mul_f32_e32 v10, v11, v10                                // 0000000055c8: 1014150b
	s_xor_b32 s1, s0, -1                                       // 0000000055cc: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055d0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000055d4: be802001
	s_cbranch_execnz 1424                                      // 0000000055d8: bfa60590 <tessera_rocm_folded_matmul_708d500594ff51c6+0x511c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000055e0: 8c7e007e
	v_mul_f32_e32 v11, v16, v25                                // 0000000055e4: 10163310
	s_delay_alu instid0(valu_dep_1)                            // 0000000055e8: bf870001
	v_cmp_class_f32_e64 s0, v11, 0x198                         // 0000000055ec: d47e0000 0201ff0b 00000198
	v_mul_f32_e32 v11, v12, v11                                // 0000000055f8: 1016170c
	s_xor_b32 s1, s0, -1                                       // 0000000055fc: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005600: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005604: be802001
	s_cbranch_execnz 1429                                      // 000000005608: bfa60595 <tessera_rocm_folded_matmul_708d500594ff51c6+0x5160>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000560c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005610: 8c7e007e
	v_mul_f32_e32 v12, v17, v25                                // 000000005614: 10183311
	s_delay_alu instid0(valu_dep_1)                            // 000000005618: bf870001
	v_cmp_class_f32_e64 s0, v12, 0x198                         // 00000000561c: d47e0000 0201ff0c 00000198
	v_mul_f32_e32 v12, v13, v12                                // 000000005628: 1018190d
	s_xor_b32 s1, s0, -1                                       // 00000000562c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005630: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005634: be802001
	s_cbranch_execnz 1434                                      // 000000005638: bfa6059a <tessera_rocm_folded_matmul_708d500594ff51c6+0x51a4>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000563c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005640: 8c7e007e
	v_mul_f32_e32 v13, v18, v25                                // 000000005644: 101a3312
	s_delay_alu instid0(valu_dep_1)                            // 000000005648: bf870001
	v_cmp_class_f32_e64 s0, v13, 0x198                         // 00000000564c: d47e0000 0201ff0d 00000198
	v_mul_f32_e32 v13, v14, v13                                // 000000005658: 101a1b0e
	s_xor_b32 s1, s0, -1                                       // 00000000565c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005660: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005664: be802001
	s_cbranch_execnz 1439                                      // 000000005668: bfa6059f <tessera_rocm_folded_matmul_708d500594ff51c6+0x51e8>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000566c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005670: 8c7e007e
	v_mul_f32_e32 v14, v19, v25                                // 000000005674: 101c3313
	s_delay_alu instid0(valu_dep_1)                            // 000000005678: bf870001
	v_cmp_class_f32_e64 s0, v14, 0x198                         // 00000000567c: d47e0000 0201ff0e 00000198
	v_mul_f32_e32 v14, v15, v14                                // 000000005688: 101c1d0f
	s_xor_b32 s1, s0, -1                                       // 00000000568c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005690: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005694: be802001
	s_cbranch_execnz 1444                                      // 000000005698: bfa605a4 <tessera_rocm_folded_matmul_708d500594ff51c6+0x522c>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000569c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000056a0: 8c7e007e
	v_bfe_u32 v15, v24, 16, 1                                  // 0000000056a4: d610000f 02052118
	v_bfe_u32 v16, v8, 16, 1                                   // 0000000056ac: d6100010 02052108
	v_or_b32_e32 v17, 0x400000, v24                            // 0000000056b4: 382230ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 0000000056bc: 7c303118
	v_or_b32_e32 v18, 0x400000, v8                             // 0000000056c0: 382410ff 00400000
	v_add3_u32 v15, v15, v24, 0x7fff                           // 0000000056c8: d655000f 03fe310f 00007fff
	v_add3_u32 v16, v16, v8, 0x7fff                            // 0000000056d4: d6550010 03fe1110 00007fff
	v_bfe_u32 v19, v9, 16, 1                                   // 0000000056e0: d6100013 02052109
	s_and_b32 s0, s3, s2                                       // 0000000056e8: 8b000203
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056ec: bf88ff9e
	s_and_b32 s0, s22, s0                                      // 0000000056f0: 8b000016
	s_wait_alu depctr_va_vcc(0)                                // 0000000056f4: bf88ff9d
	v_cndmask_b32_e32 v15, v15, v17, vcc_lo                    // 0000000056f8: 021e230f
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 0000000056fc: 7c301108
	v_bfe_u32 v17, v11, 16, 1                                  // 000000005700: d6100011 0205210b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005708: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 00000000570c: 8d00c100
	s_wait_alu depctr_va_vcc(0)                                // 000000005710: bf88ff9d
	v_cndmask_b32_e32 v8, v16, v18, vcc_lo                     // 000000005714: 02102510
	v_bfe_u32 v16, v10, 16, 1                                  // 000000005718: d6100010 0205210a
	v_add3_u32 v18, v19, v9, 0x7fff                            // 000000005720: d6550012 03fe1313 00007fff
	s_clause 0x1                                               // 00000000572c: bf850001
	global_store_d16_hi_b16 v[130:131], v15, off offset:32     // 000000005730: ee09407c 07800000 00002082
	global_store_d16_hi_b16 v[136:137], v8, off offset:32      // 00000000573c: ee09407c 04000000 00002088
	v_or_b32_e32 v8, 0x400000, v9                              // 000000005748: 381012ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v9, v9                             // 000000005750: 7c301309
	v_add3_u32 v15, v16, v10, 0x7fff                           // 000000005754: d655000f 03fe1510 00007fff
	v_or_b32_e32 v16, 0x400000, v10                            // 000000005760: 382014ff 00400000
	v_add3_u32 v17, v17, v11, 0x7fff                           // 000000005768: d6550011 03fe1711 00007fff
	v_or_b32_e32 v19, 0x400000, v11                            // 000000005774: 382616ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000577c: bf88ff9d
	v_cndmask_b32_e32 v8, v18, v8, vcc_lo                      // 000000005780: 02101112
	v_cmp_u_f32_e32 vcc_lo, v10, v10                           // 000000005784: 7c30150a
	global_store_d16_hi_b16 v[138:139], v8, off offset:32      // 000000005788: ee09407c 04000000 0000208a
	s_wait_alu depctr_va_vcc(0)                                // 000000005794: bf88ff9d
	v_cndmask_b32_e32 v9, v15, v16, vcc_lo                     // 000000005798: 0212210f
	v_cmp_u_f32_e32 vcc_lo, v11, v11                           // 00000000579c: 7c30170b
	v_bfe_u32 v8, v12, 16, 1                                   // 0000000057a0: d6100008 0205210c
	v_or_b32_e32 v11, 0x400000, v12                            // 0000000057a8: 381618ff 00400000
	v_or_b32_e32 v15, 0x400000, v13                            // 0000000057b0: 381e1aff 00400000
	v_or_b32_e32 v16, 0x400000, v14                            // 0000000057b8: 38201cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000057c0: bf88ff9d
	v_cndmask_b32_e32 v10, v17, v19, vcc_lo                    // 0000000057c4: 02142711
	global_store_d16_hi_b16 v[142:143], v9, off offset:32      // 0000000057c8: ee09407c 04800000 0000208e
	v_bfe_u32 v9, v13, 16, 1                                   // 0000000057d4: d6100009 0205210d
	v_add3_u32 v8, v8, v12, 0x7fff                             // 0000000057dc: d6550008 03fe1908 00007fff
	v_cmp_u_f32_e32 vcc_lo, v12, v12                           // 0000000057e8: 7c30190c
	global_store_d16_hi_b16 v[146:147], v10, off offset:32     // 0000000057ec: ee09407c 05000000 00002092
	v_bfe_u32 v10, v14, 16, 1                                  // 0000000057f8: d610000a 0205210e
	v_add3_u32 v9, v9, v13, 0x7fff                             // 000000005800: d6550009 03fe1b09 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000580c: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v11, vcc_lo                      // 000000005810: 02101708
	v_cmp_u_f32_e32 vcc_lo, v13, v13                           // 000000005814: 7c301b0d
	v_add3_u32 v10, v10, v14, 0x7fff                           // 000000005818: d655000a 03fe1d0a 00007fff
	global_store_d16_hi_b16 v[144:145], v8, off offset:32      // 000000005824: ee09407c 04000000 00002090
	s_wait_alu depctr_va_vcc(0)                                // 000000005830: bf88ff9d
	v_cndmask_b32_e32 v9, v9, v15, vcc_lo                      // 000000005834: 02121f09
	v_cmp_u_f32_e32 vcc_lo, v14, v14                           // 000000005838: 7c301d0e
	global_store_d16_hi_b16 v[148:149], v9, off offset:32      // 00000000583c: ee09407c 04800000 00002094
	s_wait_alu depctr_va_vcc(0)                                // 000000005848: bf88ff9d
	v_cndmask_b32_e32 v10, v10, v16, vcc_lo                    // 00000000584c: 0214210a
	global_store_d16_hi_b16 v[150:151], v10, off offset:32     // 000000005850: ee09407c 05000000 00002096
	s_wait_alu depctr_sa_sdst(0)                               // 00000000585c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005860: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005864: bf88ff9e
	s_xor_b32 s2, exec_lo, s1                                  // 000000005868: 8d02017e
	s_cbranch_execz 118                                        // 00000000586c: bfa50076 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3f48>
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[132:133]                // 000000005870: 7ca90808
	v_mov_b32_e32 v135, v49                                    // 000000005874: 7f0e0331
	v_mov_b32_e32 v141, v49                                    // 000000005878: 7f1a0331
	v_mov_b32_e32 v129, v49                                    // 00000000587c: 7f020331
	v_mov_b32_e32 v55, v49                                     // 000000005880: 7e6e0331
	v_mov_b32_e32 v53, v49                                     // 000000005884: 7e6a0331
	s_wait_alu depctr_va_vcc(0)                                // 000000005888: bf88ff9d
	v_dual_cndmask_b32 v9, 0, v133 :: v_dual_cndmask_b32 v8, 0, v132// 00000000588c: ca530a80 09090880
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[134:135]                // 000000005894: 7ca90c08
	v_cmp_gt_i64_e64 s0, s[8:9], v[140:141]                    // 000000005898: d4540000 02031808
	v_mov_b32_e32 v51, v49                                     // 0000000058a0: 7e660331
	s_wait_alu depctr_va_vcc(0)                                // 0000000058a4: bf88ff9d
	v_dual_cndmask_b32 v13, 0, v49 :: v_dual_cndmask_b32 v12, 0, v134// 0000000058a8: ca526280 0d0d0c80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[128:129]                // 0000000058b0: 7ca90008
	s_wait_alu depctr_va_sdst(0)                               // 0000000058b4: bf88f19f
	v_cndmask_b32_e64 v11, 0, v49, s0                          // 0000000058b8: d501000b 00026280
	v_cndmask_b32_e64 v10, 0, v140, s0                         // 0000000058c0: d501000a 00031880
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]                    // 0000000058c8: 3e181882
	s_wait_alu depctr_va_vcc(0)                                // 0000000058cc: bf88ff9d
	v_cndmask_b32_e32 v14, 0, v128, vcc_lo                     // 0000000058d0: 021d0080
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]                        // 0000000058d4: 3e101082
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 0000000058d8: 3e141482
	v_cndmask_b32_e32 v15, 0, v49, vcc_lo                      // 0000000058dc: 021e6280
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[54:55]                  // 0000000058e0: 7ca86c08
	v_add_co_u32 v16, s1, s12, v12                             // 0000000058e4: d7000110 0202180c
	v_add_co_u32 v8, s0, s12, v8                               // 0000000058ec: d7000008 0202100c
	s_wait_alu depctr_va_sdst(0)                               // 0000000058f4: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s13, v9, s0                  // 0000000058f8: d5207c09 0002120d
	v_add_co_u32 v10, s0, s12, v10                             // 000000005900: d700000a 0202140c
	s_wait_alu depctr_va_sdst(0)                               // 000000005908: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s13, v11, s0                // 00000000590c: d5207c0b 0002160d
	v_cmp_gt_i64_e64 s0, s[8:9], v[52:53]                      // 000000005914: d4540000 02026808
	v_add_co_ci_u32_e64 v17, null, s13, v13, s1                // 00000000591c: d5207c11 00061a0d
	v_lshlrev_b64_e32 v[12:13], 2, v[14:15]                    // 000000005924: 3e181c82
	s_wait_alu depctr_va_vcc(0)                                // 000000005928: bf88ff9d
	v_dual_cndmask_b32 v15, 0, v49 :: v_dual_cndmask_b32 v14, 0, v54// 00000000592c: ca526280 0f0e6c80
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[50:51]                  // 000000005934: 7ca86408
	s_wait_alu depctr_va_sdst(0)                               // 000000005938: bf88f19f
	v_cndmask_b32_e64 v19, 0, v49, s0                          // 00000000593c: d5010013 00026280
	v_cndmask_b32_e64 v18, 0, v52, s0                          // 000000005944: d5010012 00026880
	v_add_co_u32 v20, s0, s12, v12                             // 00000000594c: d7000014 0202180c
	s_wait_alu depctr_va_sdst(0)                               // 000000005954: bf88f19f
	v_add_co_ci_u32_e64 v21, null, s13, v13, s0                // 000000005958: d5207c15 00021a0d
	v_lshlrev_b64_e32 v[12:13], 2, v[14:15]                    // 000000005960: 3e181c82
	v_lshlrev_b64_e32 v[14:15], 2, v[18:19]                    // 000000005964: 3e1c2482
	s_wait_alu depctr_va_vcc(0)                                // 000000005968: bf88ff9d
	v_dual_cndmask_b32 v19, 0, v49 :: v_dual_cndmask_b32 v18, 0, v50// 00000000596c: ca526280 13126480
	v_cmp_gt_i64_e32 vcc_lo, s[8:9], v[48:49]                  // 000000005974: 7ca86008
	s_delay_alu instid0(valu_dep_4)                            // 000000005978: bf870004
	v_add_co_u32 v22, s0, s12, v12                             // 00000000597c: d7000016 0202180c
	s_wait_alu depctr_va_sdst(0)                               // 000000005984: bf88f19f
	v_add_co_ci_u32_e64 v23, null, s13, v13, s0                // 000000005988: d5207c17 00021a0d
	v_lshlrev_b64_e32 v[12:13], 2, v[18:19]                    // 000000005990: 3e182482
	s_wait_alu depctr_va_vcc(0)                                // 000000005994: bf88ff9d
	v_dual_cndmask_b32 v19, 0, v49 :: v_dual_cndmask_b32 v18, 0, v48// 000000005998: ca526280 13126080
	v_add_co_u32 v24, vcc_lo, s12, v14                         // 0000000059a0: d7006a18 02021c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000059a8: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s13, v15, vcc_lo            // 0000000059ac: d5207c19 01aa1e0d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 0000000059b4: bf8701c3
	v_lshlrev_b64_e32 v[14:15], 2, v[18:19]                    // 0000000059b8: 3e1c2482
	v_add_co_u32 v18, vcc_lo, s12, v12                         // 0000000059bc: d7006a12 0202180c
	s_wait_alu depctr_va_vcc(0)                                // 0000000059c4: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s13, v13, vcc_lo            // 0000000059c8: d5207c13 01aa1a0d
	v_add_co_u32 v26, vcc_lo, s12, v14                         // 0000000059d0: d7006a1a 02021c0c
	s_wait_alu depctr_va_vcc(0)                                // 0000000059d8: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, s13, v15, vcc_lo            // 0000000059dc: d5207c1b 01aa1e0d
	s_clause 0x7                                               // 0000000059e4: bf850007
	global_load_b32 v12, v[8:9], off                           // 0000000059e8: ee05007c 0000000c 00000008
	global_load_b32 v13, v[10:11], off                         // 0000000059f4: ee05007c 0000000d 0000000a
	global_load_b32 v14, v[16:17], off                         // 000000005a00: ee05007c 0000000e 00000010
	global_load_b32 v15, v[20:21], off                         // 000000005a0c: ee05007c 0000000f 00000014
	global_load_b32 v8, v[22:23], off                          // 000000005a18: ee05007c 00000008 00000016
	global_load_b32 v9, v[24:25], off                          // 000000005a24: ee05007c 00000009 00000018
	global_load_b32 v10, v[18:19], off                         // 000000005a30: ee05007c 0000000a 00000012
	global_load_b32 v11, v[26:27], off                         // 000000005a3c: ee05007c 0000000b 0000001a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a48: bf88ff9e
	s_and_not1_saveexec_b32 s0, s2                             // 000000005a4c: be803002
	s_cbranch_execz 28                                         // 000000005a50: bfa5001c <tessera_rocm_folded_matmul_708d500594ff51c6+0x3fc4>
	s_wait_loadcnt 0x3                                         // 000000005a54: bfc00003
	v_add_co_u32 v8, s1, s6, v72                               // 000000005a58: d7000108 02029006
	s_wait_loadcnt 0x2                                         // 000000005a60: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000005a64: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s7, 0, s1                    // 000000005a68: d5207c09 00050007
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000005a70: bf870122
	v_add_co_u32 v8, vcc_lo, v8, v73                           // 000000005a74: d7006a08 02029308
	s_wait_alu depctr_va_vcc(0)                                // 000000005a7c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v9, vcc_lo                // 000000005a80: d5207c09 01aa1280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005a88: bf870091
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]                        // 000000005a8c: 3e101082
	v_add_co_u32 v8, vcc_lo, s12, v8                           // 000000005a90: d7006a08 0202100c
	s_wait_alu depctr_va_vcc(0)                                // 000000005a98: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000005a9c: bf870002
	v_add_co_ci_u32_e64 v9, null, s13, v9, vcc_lo              // 000000005aa0: d5207c09 01aa120d
	global_load_b128 v[12:15], v[8:9], off offset:192          // 000000005aa8: ee05c07c 0000000c 0000c008
	s_wait_loadcnt 0x1                                         // 000000005ab4: bfc00001
	global_load_b128 v[8:11], v[8:9], off offset:208           // 000000005ab8: ee05c07c 00000008 0000d008
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ac4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005ac8: 8c7e007e
	global_load_u8 v16, v[75:76], off                          // 000000005acc: ee04007c 00000010 0000004b
	s_wait_loadcnt 0x0                                         // 000000005ad8: bfc00000
	v_lshlrev_b32_e32 v17, 23, v16                             // 000000005adc: 30222097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ae0: bf870091
	v_mul_f32_e32 v16, v12, v17                                // 000000005ae4: 1020230c
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 000000005ae8: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v0, v16                                 // 000000005af4: 10202100
	s_xor_b32 s1, s0, -1                                       // 000000005af8: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005afc: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005b00: be802001
	s_cbranch_execnz 1178                                      // 000000005b04: bfa6049a <tessera_rocm_folded_matmul_708d500594ff51c6+0x5270>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b0c: 8c7e007e
	v_mul_f32_e32 v0, v13, v17                                 // 000000005b10: 1000230d
	s_delay_alu instid0(valu_dep_1)                            // 000000005b14: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 000000005b18: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v1, v0                                   // 000000005b24: 10000101
	s_xor_b32 s1, s0, -1                                       // 000000005b28: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b2c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005b30: be802001
	s_cbranch_execnz 1183                                      // 000000005b34: bfa6049f <tessera_rocm_folded_matmul_708d500594ff51c6+0x52b4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b38: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b3c: 8c7e007e
	v_mul_f32_e32 v1, v14, v17                                 // 000000005b40: 1002230e
	s_delay_alu instid0(valu_dep_1)                            // 000000005b44: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 000000005b48: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v2, v1                                   // 000000005b54: 10020302
	s_xor_b32 s1, s0, -1                                       // 000000005b58: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b5c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005b60: be802001
	s_cbranch_execnz 1188                                      // 000000005b64: bfa604a4 <tessera_rocm_folded_matmul_708d500594ff51c6+0x52f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b6c: 8c7e007e
	v_mul_f32_e32 v2, v15, v17                                 // 000000005b70: 1004230f
	s_delay_alu instid0(valu_dep_1)                            // 000000005b74: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000005b78: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 000000005b84: 10040503
	s_xor_b32 s1, s0, -1                                       // 000000005b88: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b8c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005b90: be802001
	s_cbranch_execnz 1193                                      // 000000005b94: bfa604a9 <tessera_rocm_folded_matmul_708d500594ff51c6+0x533c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005b9c: 8c7e007e
	v_mul_f32_e32 v3, v8, v17                                  // 000000005ba0: 10062308
	s_delay_alu instid0(valu_dep_1)                            // 000000005ba4: bf870001
	v_cmp_class_f32_e64 s0, v3, 0x198                          // 000000005ba8: d47e0000 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 000000005bb4: 10060704
	s_xor_b32 s1, s0, -1                                       // 000000005bb8: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bbc: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005bc0: be802001
	s_cbranch_execnz 1198                                      // 000000005bc4: bfa604ae <tessera_rocm_folded_matmul_708d500594ff51c6+0x5380>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005bcc: 8c7e007e
	v_mul_f32_e32 v4, v9, v17                                  // 000000005bd0: 10082309
	s_delay_alu instid0(valu_dep_1)                            // 000000005bd4: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 000000005bd8: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 000000005be4: 10080905
	s_xor_b32 s1, s0, -1                                       // 000000005be8: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bec: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005bf0: be802001
	s_cbranch_execnz 1203                                      // 000000005bf4: bfa604b3 <tessera_rocm_folded_matmul_708d500594ff51c6+0x53c4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bf8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005bfc: 8c7e007e
	v_mul_f32_e32 v5, v10, v17                                 // 000000005c00: 100a230a
	s_delay_alu instid0(valu_dep_1)                            // 000000005c04: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 000000005c08: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 000000005c14: 100a0b06
	s_xor_b32 s1, s0, -1                                       // 000000005c18: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c1c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005c20: be802001
	s_cbranch_execnz 1208                                      // 000000005c24: bfa604b8 <tessera_rocm_folded_matmul_708d500594ff51c6+0x5408>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005c2c: 8c7e007e
	v_mul_f32_e32 v6, v11, v17                                 // 000000005c30: 100c230b
	s_delay_alu instid0(valu_dep_1)                            // 000000005c34: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000005c38: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v7, v6                                   // 000000005c44: 100c0d07
	s_xor_b32 s1, s0, -1                                       // 000000005c48: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c4c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005c50: be802001
	s_cbranch_execnz 1213                                      // 000000005c54: bfa604bd <tessera_rocm_folded_matmul_708d500594ff51c6+0x544c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005c5c: 8c7e007e
	v_bfe_u32 v7, v16, 16, 1                                   // 000000005c60: d6100007 02052110
	v_or_b32_e32 v8, 0x400000, v16                             // 000000005c68: 381020ff 00400000
	v_bfe_u32 v9, v0, 16, 1                                    // 000000005c70: d6100009 02052100
	v_cmp_u_f32_e32 vcc_lo, v16, v16                           // 000000005c78: 7c302110
	v_or_b32_e32 v10, 0x400000, v0                             // 000000005c7c: 381400ff 00400000
	v_add3_u32 v7, v7, v16, 0x7fff                             // 000000005c84: d6550007 03fe2107 00007fff
	v_bfe_u32 v11, v1, 16, 1                                   // 000000005c90: d610000b 02052101
	v_add3_u32 v9, v9, v0, 0x7fff                              // 000000005c98: d6550009 03fe0109 00007fff
	v_or_b32_e32 v12, 0x400000, v1                             // 000000005ca4: 381802ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005cac: bf88ff9d
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo                       // 000000005cb0: 020e1107
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 000000005cb4: 7c300100
	v_bfe_u32 v8, v2, 16, 1                                    // 000000005cb8: d6100008 02052102
	v_add3_u32 v11, v11, v1, 0x7fff                            // 000000005cc0: d655000b 03fe030b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005ccc: bf88ff9d
	v_cndmask_b32_e32 v0, v9, v10, vcc_lo                      // 000000005cd0: 02001509
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000005cd4: 7c300301
	global_store_d16_hi_b16 v[44:45], v7, off offset:32        // 000000005cd8: ee09407c 03800000 0000202c
	v_bfe_u32 v7, v3, 16, 1                                    // 000000005ce4: d6100007 02052103
	v_or_b32_e32 v9, 0x400000, v5                              // 000000005cec: 38120aff 00400000
	global_store_d16_hi_b16 v[42:43], v0, off offset:32        // 000000005cf4: ee09407c 00000000 0000202a
	s_wait_alu depctr_va_vcc(0)                                // 000000005d00: bf88ff9d
	v_cndmask_b32_e32 v1, v11, v12, vcc_lo                     // 000000005d04: 0202190b
	v_add3_u32 v0, v8, v2, 0x7fff                              // 000000005d08: d6550000 03fe0508 00007fff
	v_or_b32_e32 v8, 0x400000, v2                              // 000000005d14: 381004ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 000000005d1c: 7c300502
	v_bfe_u32 v2, v4, 16, 1                                    // 000000005d20: d6100002 02052104
	global_store_d16_hi_b16 v[152:153], v1, off offset:32      // 000000005d28: ee09407c 00800000 00002098
	v_add3_u32 v1, v7, v3, 0x7fff                              // 000000005d34: d6550001 03fe0707 00007fff
	v_or_b32_e32 v7, 0x400000, v3                              // 000000005d40: 380e06ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005d48: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v8, vcc_lo                       // 000000005d4c: 02001100
	v_cmp_u_f32_e32 vcc_lo, v3, v3                             // 000000005d50: 7c300703
	v_bfe_u32 v8, v5, 16, 1                                    // 000000005d54: d6100008 02052105
	v_add3_u32 v2, v2, v4, 0x7fff                              // 000000005d5c: d6550002 03fe0902 00007fff
	v_bfe_u32 v3, v6, 16, 1                                    // 000000005d68: d6100003 02052106
	v_or_b32_e32 v10, 0x400000, v6                             // 000000005d70: 38140cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005d78: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 000000005d7c: 02020f01
	v_or_b32_e32 v7, 0x400000, v4                              // 000000005d80: 380e08ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 000000005d88: 7c300904
	v_add3_u32 v8, v8, v5, 0x7fff                              // 000000005d8c: d6550008 03fe0b08 00007fff
	v_add3_u32 v3, v3, v6, 0x7fff                              // 000000005d98: d6550003 03fe0d03 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005da4: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v7, vcc_lo                       // 000000005da8: 02040f02
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000005dac: 7c300b05
	s_wait_alu depctr_va_vcc(0)                                // 000000005db0: bf88ff9d
	v_cndmask_b32_e32 v4, v8, v9, vcc_lo                       // 000000005db4: 02081308
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 000000005db8: 7c300d06
	s_wait_alu depctr_va_vcc(0)                                // 000000005dbc: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v10, vcc_lo                      // 000000005dc0: 02061503
	s_clause 0x3                                               // 000000005dc4: bf850003
	global_store_d16_hi_b16 v[40:41], v0, off offset:32        // 000000005dc8: ee09407c 00000000 00002028
	global_store_d16_hi_b16 v[46:47], v1, off offset:32        // 000000005dd4: ee09407c 00800000 0000202e
	global_store_d16_hi_b16 v[154:155], v2, off offset:32      // 000000005de0: ee09407c 01000000 0000209a
	global_store_d16_hi_b16 v[156:157], v4, off offset:32      // 000000005dec: ee09407c 02000000 0000209c
	global_store_d16_hi_b16 v[158:159], v3, off offset:32      // 000000005df8: ee09407c 01800000 0000209e
	s_nop 0                                                    // 000000005e04: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005e08: bfb60003
	s_endpgm                                                   // 000000005e0c: bfb00000
	v_cvt_f64_f32_e32 v[83:84], v56                            // 000000005e10: 7ea62138
	v_cvt_f64_f32_e32 v[85:86], v82                            // 000000005e14: 7eaa2152
	v_cvt_f64_f32_e32 v[87:88], v68                            // 000000005e18: 7eae2144
	v_cmp_eq_f32_e64 s2, 0, v56                                // 000000005e1c: d4120002 02027080
	v_cmp_class_f32_e64 s5, v68, 0x1f8                         // 000000005e24: d47e0005 0201ff44 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005e30: 8b020502
	v_mul_f64_e32 v[83:84], v[83:84], v[85:86]                 // 000000005e34: 0ca6ab53
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e38: bf870091
	v_mul_f64_e32 v[83:84], v[83:84], v[87:88]                 // 000000005e3c: 0ca6af53
	v_cvt_f32_f64_e32 v81, v[83:84]                            // 000000005e40: 7ea21f53
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e44: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005e48: bf870001
	v_cndmask_b32_e64 v81, v81, 0, s2                          // 000000005e4c: d5010051 00090151
	s_branch 62194                                             // 000000005e54: bfa0f2f2 <tessera_rocm_folded_matmul_708d500594ff51c6+0xf20>
	v_cvt_f64_f32_e32 v[83:84], v57                            // 000000005e58: 7ea62139
	v_cvt_f64_f32_e32 v[85:86], v82                            // 000000005e5c: 7eaa2152
	v_cvt_f64_f32_e32 v[87:88], v69                            // 000000005e60: 7eae2145
	v_cmp_eq_f32_e64 s2, 0, v57                                // 000000005e64: d4120002 02027280
	v_cmp_class_f32_e64 s5, v69, 0x1f8                         // 000000005e6c: d47e0005 0201ff45 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005e78: 8b020502
	v_mul_f64_e32 v[83:84], v[83:84], v[85:86]                 // 000000005e7c: 0ca6ab53
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e80: bf870091
	v_mul_f64_e32 v[83:84], v[83:84], v[87:88]                 // 000000005e84: 0ca6af53
	v_cvt_f32_f64_e32 v56, v[83:84]                            // 000000005e88: 7e701f53
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e8c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005e90: bf870001
	v_cndmask_b32_e64 v56, v56, 0, s2                          // 000000005e94: d5010038 00090138
	s_branch 62188                                             // 000000005e9c: bfa0f2ec <tessera_rocm_folded_matmul_708d500594ff51c6+0xf50>
	v_cvt_f64_f32_e32 v[68:69], v58                            // 000000005ea0: 7e88213a
	v_cvt_f64_f32_e32 v[83:84], v82                            // 000000005ea4: 7ea62152
	v_cvt_f64_f32_e32 v[85:86], v70                            // 000000005ea8: 7eaa2146
	v_cmp_eq_f32_e64 s2, 0, v58                                // 000000005eac: d4120002 02027480
	v_cmp_class_f32_e64 s5, v70, 0x1f8                         // 000000005eb4: d47e0005 0201ff46 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005ec0: 8b020502
	v_mul_f64_e32 v[68:69], v[68:69], v[83:84]                 // 000000005ec4: 0c88a744
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ec8: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[85:86]                 // 000000005ecc: 0c88ab44
	v_cvt_f32_f64_e32 v57, v[68:69]                            // 000000005ed0: 7e721f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ed4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ed8: bf870001
	v_cndmask_b32_e64 v57, v57, 0, s2                          // 000000005edc: d5010039 00090139
	s_branch 62182                                             // 000000005ee4: bfa0f2e6 <tessera_rocm_folded_matmul_708d500594ff51c6+0xf80>
	v_cvt_f64_f32_e32 v[68:69], v59                            // 000000005ee8: 7e88213b
	v_cvt_f64_f32_e32 v[83:84], v82                            // 000000005eec: 7ea62152
	v_cvt_f64_f32_e32 v[85:86], v71                            // 000000005ef0: 7eaa2147
	v_cmp_eq_f32_e64 s2, 0, v59                                // 000000005ef4: d4120002 02027680
	v_cmp_class_f32_e64 s5, v71, 0x1f8                         // 000000005efc: d47e0005 0201ff47 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f08: 8b020502
	v_mul_f64_e32 v[68:69], v[68:69], v[83:84]                 // 000000005f0c: 0c88a744
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f10: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[85:86]                 // 000000005f14: 0c88ab44
	v_cvt_f32_f64_e32 v58, v[68:69]                            // 000000005f18: 7e741f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f1c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f20: bf870001
	v_cndmask_b32_e64 v58, v58, 0, s2                          // 000000005f24: d501003a 0009013a
	s_branch 62176                                             // 000000005f2c: bfa0f2e0 <tessera_rocm_folded_matmul_708d500594ff51c6+0xfb0>
	v_cvt_f64_f32_e32 v[68:69], v60                            // 000000005f30: 7e88213c
	v_cvt_f64_f32_e32 v[70:71], v82                            // 000000005f34: 7e8c2152
	v_cvt_f64_f32_e32 v[83:84], v64                            // 000000005f38: 7ea62140
	v_cmp_eq_f32_e64 s2, 0, v60                                // 000000005f3c: d4120002 02027880
	v_cmp_class_f32_e64 s5, v64, 0x1f8                         // 000000005f44: d47e0005 0201ff40 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f50: 8b020502
	v_mul_f64_e32 v[68:69], v[68:69], v[70:71]                 // 000000005f54: 0c888d44
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f58: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[83:84]                 // 000000005f5c: 0c88a744
	v_cvt_f32_f64_e32 v59, v[68:69]                            // 000000005f60: 7e761f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f64: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f68: bf870001
	v_cndmask_b32_e64 v59, v59, 0, s2                          // 000000005f6c: d501003b 0009013b
	s_branch 62170                                             // 000000005f74: bfa0f2da <tessera_rocm_folded_matmul_708d500594ff51c6+0xfe0>
	v_cvt_f64_f32_e32 v[68:69], v61                            // 000000005f78: 7e88213d
	v_cvt_f64_f32_e32 v[70:71], v82                            // 000000005f7c: 7e8c2152
	v_cvt_f64_f32_e32 v[83:84], v65                            // 000000005f80: 7ea62141
	v_cmp_eq_f32_e64 s2, 0, v61                                // 000000005f84: d4120002 02027a80
	v_cmp_class_f32_e64 s5, v65, 0x1f8                         // 000000005f8c: d47e0005 0201ff41 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005f98: 8b020502
	v_mul_f64_e32 v[68:69], v[68:69], v[70:71]                 // 000000005f9c: 0c888d44
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005fa0: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[83:84]                 // 000000005fa4: 0c88a744
	v_cvt_f32_f64_e32 v60, v[68:69]                            // 000000005fa8: 7e781f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005fb0: bf870001
	v_cndmask_b32_e64 v68, v60, 0, s2                          // 000000005fb4: d5010044 0009013c
	s_branch 62164                                             // 000000005fbc: bfa0f2d4 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1010>
	v_cvt_f64_f32_e32 v[60:61], v62                            // 000000005fc0: 7e78213e
	v_cvt_f64_f32_e32 v[64:65], v82                            // 000000005fc4: 7e802152
	v_cvt_f64_f32_e32 v[69:70], v66                            // 000000005fc8: 7e8a2142
	v_cmp_eq_f32_e64 s2, 0, v62                                // 000000005fcc: d4120002 02027c80
	v_cmp_class_f32_e64 s5, v66, 0x1f8                         // 000000005fd4: d47e0005 0201ff42 000001f8
	s_and_b32 s2, s2, s5                                       // 000000005fe0: 8b020502
	v_mul_f64_e32 v[60:61], v[60:61], v[64:65]                 // 000000005fe4: 0c78813c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005fe8: bf870091
	v_mul_f64_e32 v[60:61], v[60:61], v[69:70]                 // 000000005fec: 0c788b3c
	v_cvt_f32_f64_e32 v60, v[60:61]                            // 000000005ff0: 7e781f3c
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ff4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ff8: bf870001
	v_cndmask_b32_e64 v61, v60, 0, s2                          // 000000005ffc: d501003d 0009013c
	s_branch 62158                                             // 000000006004: bfa0f2ce <tessera_rocm_folded_matmul_708d500594ff51c6+0x1040>
	v_cvt_f64_f32_e32 v[64:65], v63                            // 000000006008: 7e80213f
	v_cvt_f64_f32_e32 v[69:70], v82                            // 00000000600c: 7e8a2152
	v_cvt_f64_f32_e32 v[82:83], v67                            // 000000006010: 7ea42143
	v_cmp_eq_f32_e64 s2, 0, v63                                // 000000006014: d4120002 02027e80
	v_cmp_class_f32_e64 s5, v67, 0x1f8                         // 00000000601c: d47e0005 0201ff43 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006028: 8b020502
	v_mul_f64_e32 v[64:65], v[64:65], v[69:70]                 // 00000000602c: 0c808b40
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006030: bf870091
	v_mul_f64_e32 v[64:65], v[64:65], v[82:83]                 // 000000006034: 0c80a540
	v_cvt_f32_f64_e32 v60, v[64:65]                            // 000000006038: 7e781f40
	s_wait_alu depctr_sa_sdst(0)                               // 00000000603c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006040: bf870001
	v_cndmask_b32_e64 v60, v60, 0, s2                          // 000000006044: d501003c 0009013c
	s_branch 62152                                             // 00000000604c: bfa0f2c8 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1070>
	v_cvt_f64_f32_e32 v[103:104], v48                          // 000000006050: 7ece2130
	v_cvt_f64_f32_e32 v[105:106], v69                          // 000000006054: 7ed22145
	v_cvt_f64_f32_e32 v[107:108], v60                          // 000000006058: 7ed6213c
	v_cmp_eq_f32_e64 s1, 0, v48                                // 00000000605c: d4120001 02026080
	v_cmp_class_f32_e64 s4, v60, 0x1f8                         // 000000006064: d47e0004 0201ff3c 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006070: 8b010401
	v_mul_f64_e32 v[103:104], v[103:104], v[105:106]           // 000000006074: 0cced367
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006078: bf870091
	v_mul_f64_e32 v[103:104], v[103:104], v[107:108]           // 00000000607c: 0cced767
	v_cvt_f32_f64_e32 v67, v[103:104]                          // 000000006080: 7e861f67
	s_wait_alu depctr_sa_sdst(0)                               // 000000006084: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006088: bf870001
	v_cndmask_b32_e64 v67, v67, 0, s1                          // 00000000608c: d5010043 00050143
	s_branch 62571                                             // 000000006094: bfa0f46b <tessera_rocm_folded_matmul_708d500594ff51c6+0x1744>
	v_cvt_f64_f32_e32 v[103:104], v49                          // 000000006098: 7ece2131
	v_cvt_f64_f32_e32 v[105:106], v69                          // 00000000609c: 7ed22145
	v_cvt_f64_f32_e32 v[107:108], v61                          // 0000000060a0: 7ed6213d
	v_cmp_eq_f32_e64 s1, 0, v49                                // 0000000060a4: d4120001 02026280
	v_cmp_class_f32_e64 s4, v61, 0x1f8                         // 0000000060ac: d47e0004 0201ff3d 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000060b8: 8b010401
	v_mul_f64_e32 v[103:104], v[103:104], v[105:106]           // 0000000060bc: 0cced367
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000060c0: bf870091
	v_mul_f64_e32 v[103:104], v[103:104], v[107:108]           // 0000000060c4: 0cced767
	v_cvt_f32_f64_e32 v48, v[103:104]                          // 0000000060c8: 7e601f67
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060cc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000060d0: bf870001
	v_cndmask_b32_e64 v48, v48, 0, s1                          // 0000000060d4: d5010030 00050130
	s_branch 62565                                             // 0000000060dc: bfa0f465 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1774>
	v_cvt_f64_f32_e32 v[60:61], v50                            // 0000000060e0: 7e782132
	v_cvt_f64_f32_e32 v[103:104], v69                          // 0000000060e4: 7ece2145
	v_cvt_f64_f32_e32 v[105:106], v62                          // 0000000060e8: 7ed2213e
	v_cmp_eq_f32_e64 s1, 0, v50                                // 0000000060ec: d4120001 02026480
	v_cmp_class_f32_e64 s4, v62, 0x1f8                         // 0000000060f4: d47e0004 0201ff3e 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006100: 8b010401
	v_mul_f64_e32 v[60:61], v[60:61], v[103:104]               // 000000006104: 0c78cf3c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006108: bf870091
	v_mul_f64_e32 v[60:61], v[60:61], v[105:106]               // 00000000610c: 0c78d33c
	v_cvt_f32_f64_e32 v49, v[60:61]                            // 000000006110: 7e621f3c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006114: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006118: bf870001
	v_cndmask_b32_e64 v49, v49, 0, s1                          // 00000000611c: d5010031 00050131
	s_branch 62559                                             // 000000006124: bfa0f45f <tessera_rocm_folded_matmul_708d500594ff51c6+0x17a4>
	v_cvt_f64_f32_e32 v[60:61], v51                            // 000000006128: 7e782133
	v_cvt_f64_f32_e32 v[103:104], v69                          // 00000000612c: 7ece2145
	v_cvt_f64_f32_e32 v[105:106], v63                          // 000000006130: 7ed2213f
	v_cmp_eq_f32_e64 s1, 0, v51                                // 000000006134: d4120001 02026680
	v_cmp_class_f32_e64 s4, v63, 0x1f8                         // 00000000613c: d47e0004 0201ff3f 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006148: 8b010401
	v_mul_f64_e32 v[60:61], v[60:61], v[103:104]               // 00000000614c: 0c78cf3c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006150: bf870091
	v_mul_f64_e32 v[60:61], v[60:61], v[105:106]               // 000000006154: 0c78d33c
	v_cvt_f32_f64_e32 v50, v[60:61]                            // 000000006158: 7e641f3c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000615c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006160: bf870001
	v_cndmask_b32_e64 v50, v50, 0, s1                          // 000000006164: d5010032 00050132
	s_branch 62553                                             // 00000000616c: bfa0f459 <tessera_rocm_folded_matmul_708d500594ff51c6+0x17d4>
	v_cvt_f64_f32_e32 v[60:61], v52                            // 000000006170: 7e782134
	v_cvt_f64_f32_e32 v[62:63], v69                            // 000000006174: 7e7c2145
	v_cvt_f64_f32_e32 v[103:104], v56                          // 000000006178: 7ece2138
	v_cmp_eq_f32_e64 s1, 0, v52                                // 00000000617c: d4120001 02026880
	v_cmp_class_f32_e64 s4, v56, 0x1f8                         // 000000006184: d47e0004 0201ff38 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006190: 8b010401
	v_mul_f64_e32 v[60:61], v[60:61], v[62:63]                 // 000000006194: 0c787d3c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006198: bf870091
	v_mul_f64_e32 v[60:61], v[60:61], v[103:104]               // 00000000619c: 0c78cf3c
	v_cvt_f32_f64_e32 v51, v[60:61]                            // 0000000061a0: 7e661f3c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061a8: bf870001
	v_cndmask_b32_e64 v51, v51, 0, s1                          // 0000000061ac: d5010033 00050133
	s_branch 62547                                             // 0000000061b4: bfa0f453 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1804>
	v_cvt_f64_f32_e32 v[60:61], v53                            // 0000000061b8: 7e782135
	v_cvt_f64_f32_e32 v[62:63], v69                            // 0000000061bc: 7e7c2145
	v_cvt_f64_f32_e32 v[103:104], v57                          // 0000000061c0: 7ece2139
	v_cmp_eq_f32_e64 s1, 0, v53                                // 0000000061c4: d4120001 02026a80
	v_cmp_class_f32_e64 s4, v57, 0x1f8                         // 0000000061cc: d47e0004 0201ff39 000001f8
	s_and_b32 s1, s1, s4                                       // 0000000061d8: 8b010401
	v_mul_f64_e32 v[60:61], v[60:61], v[62:63]                 // 0000000061dc: 0c787d3c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000061e0: bf870091
	v_mul_f64_e32 v[60:61], v[60:61], v[103:104]               // 0000000061e4: 0c78cf3c
	v_cvt_f32_f64_e32 v52, v[60:61]                            // 0000000061e8: 7e681f3c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061ec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061f0: bf870001
	v_cndmask_b32_e64 v52, v52, 0, s1                          // 0000000061f4: d5010034 00050134
	s_branch 62541                                             // 0000000061fc: bfa0f44d <tessera_rocm_folded_matmul_708d500594ff51c6+0x1834>
	v_cvt_f64_f32_e32 v[56:57], v54                            // 000000006200: 7e702136
	v_cvt_f64_f32_e32 v[60:61], v69                            // 000000006204: 7e782145
	v_cvt_f64_f32_e32 v[62:63], v58                            // 000000006208: 7e7c213a
	v_cmp_eq_f32_e64 s1, 0, v54                                // 00000000620c: d4120001 02026c80
	v_cmp_class_f32_e64 s4, v58, 0x1f8                         // 000000006214: d47e0004 0201ff3a 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006220: 8b010401
	v_mul_f64_e32 v[56:57], v[56:57], v[60:61]                 // 000000006224: 0c707938
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006228: bf870091
	v_mul_f64_e32 v[56:57], v[56:57], v[62:63]                 // 00000000622c: 0c707d38
	v_cvt_f32_f64_e32 v53, v[56:57]                            // 000000006230: 7e6a1f38
	s_wait_alu depctr_sa_sdst(0)                               // 000000006234: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006238: bf870001
	v_cndmask_b32_e64 v60, v53, 0, s1                          // 00000000623c: d501003c 00050135
	s_branch 62535                                             // 000000006244: bfa0f447 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1864>
	v_cvt_f64_f32_e32 v[53:54], v55                            // 000000006248: 7e6a2137
	v_cvt_f64_f32_e32 v[56:57], v69                            // 00000000624c: 7e702145
	v_cvt_f64_f32_e32 v[61:62], v59                            // 000000006250: 7e7a213b
	v_cmp_eq_f32_e64 s1, 0, v55                                // 000000006254: d4120001 02026e80
	v_cmp_class_f32_e64 s4, v59, 0x1f8                         // 00000000625c: d47e0004 0201ff3b 000001f8
	s_and_b32 s1, s1, s4                                       // 000000006268: 8b010401
	v_mul_f64_e32 v[53:54], v[53:54], v[56:57]                 // 00000000626c: 0c6a7135
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006270: bf870091
	v_mul_f64_e32 v[53:54], v[53:54], v[61:62]                 // 000000006274: 0c6a7b35
	v_cvt_f32_f64_e32 v53, v[53:54]                            // 000000006278: 7e6a1f35
	s_wait_alu depctr_sa_sdst(0)                               // 00000000627c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006280: bf870001
	v_cndmask_b32_e64 v53, v53, 0, s1                          // 000000006284: d5010035 00050135
	s_branch 62529                                             // 00000000628c: bfa0f441 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1894>
	v_cvt_f64_f32_e32 v[129:130], v40                          // 000000006290: 7f022128
	v_cvt_f64_f32_e32 v[131:132], v61                          // 000000006294: 7f06213d
	v_cvt_f64_f32_e32 v[133:134], v52                          // 000000006298: 7f0a2134
	v_cmp_eq_f32_e64 s2, 0, v40                                // 00000000629c: d4120002 02025080
	v_cmp_class_f32_e64 s5, v52, 0x1f8                         // 0000000062a4: d47e0005 0201ff34 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000062b0: 8b020502
	v_mul_f64_e32 v[129:130], v[129:130], v[131:132]           // 0000000062b4: 0d030781
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062b8: bf870091
	v_mul_f64_e32 v[129:130], v[129:130], v[133:134]           // 0000000062bc: 0d030b81
	v_cvt_f32_f64_e32 v59, v[129:130]                          // 0000000062c0: 7e761f81
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000062c8: bf870001
	v_cndmask_b32_e64 v59, v59, 0, s2                          // 0000000062cc: d501003b 0009013b
	s_branch 62938                                             // 0000000062d4: bfa0f5da <tessera_rocm_folded_matmul_708d500594ff51c6+0x1f40>
	v_cvt_f64_f32_e32 v[129:130], v41                          // 0000000062d8: 7f022129
	v_cvt_f64_f32_e32 v[131:132], v61                          // 0000000062dc: 7f06213d
	v_cvt_f64_f32_e32 v[133:134], v53                          // 0000000062e0: 7f0a2135
	v_cmp_eq_f32_e64 s2, 0, v41                                // 0000000062e4: d4120002 02025280
	v_cmp_class_f32_e64 s5, v53, 0x1f8                         // 0000000062ec: d47e0005 0201ff35 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000062f8: 8b020502
	v_mul_f64_e32 v[129:130], v[129:130], v[131:132]           // 0000000062fc: 0d030781
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006300: bf870091
	v_mul_f64_e32 v[129:130], v[129:130], v[133:134]           // 000000006304: 0d030b81
	v_cvt_f32_f64_e32 v40, v[129:130]                          // 000000006308: 7e501f81
	s_wait_alu depctr_sa_sdst(0)                               // 00000000630c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006310: bf870001
	v_cndmask_b32_e64 v40, v40, 0, s2                          // 000000006314: d5010028 00090128
	s_branch 62932                                             // 00000000631c: bfa0f5d4 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1f70>
	v_cvt_f64_f32_e32 v[52:53], v42                            // 000000006320: 7e68212a
	v_cvt_f64_f32_e32 v[129:130], v61                          // 000000006324: 7f02213d
	v_cvt_f64_f32_e32 v[131:132], v54                          // 000000006328: 7f062136
	v_cmp_eq_f32_e64 s2, 0, v42                                // 00000000632c: d4120002 02025480
	v_cmp_class_f32_e64 s5, v54, 0x1f8                         // 000000006334: d47e0005 0201ff36 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006340: 8b020502
	v_mul_f64_e32 v[52:53], v[52:53], v[129:130]               // 000000006344: 0c690334
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006348: bf870091
	v_mul_f64_e32 v[52:53], v[52:53], v[131:132]               // 00000000634c: 0c690734
	v_cvt_f32_f64_e32 v41, v[52:53]                            // 000000006350: 7e521f34
	s_wait_alu depctr_sa_sdst(0)                               // 000000006354: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006358: bf870001
	v_cndmask_b32_e64 v41, v41, 0, s2                          // 00000000635c: d5010029 00090129
	s_branch 62926                                             // 000000006364: bfa0f5ce <tessera_rocm_folded_matmul_708d500594ff51c6+0x1fa0>
	v_cvt_f64_f32_e32 v[52:53], v43                            // 000000006368: 7e68212b
	v_cvt_f64_f32_e32 v[129:130], v61                          // 00000000636c: 7f02213d
	v_cvt_f64_f32_e32 v[131:132], v55                          // 000000006370: 7f062137
	v_cmp_eq_f32_e64 s2, 0, v43                                // 000000006374: d4120002 02025680
	v_cmp_class_f32_e64 s5, v55, 0x1f8                         // 00000000637c: d47e0005 0201ff37 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006388: 8b020502
	v_mul_f64_e32 v[52:53], v[52:53], v[129:130]               // 00000000638c: 0c690334
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006390: bf870091
	v_mul_f64_e32 v[52:53], v[52:53], v[131:132]               // 000000006394: 0c690734
	v_cvt_f32_f64_e32 v42, v[52:53]                            // 000000006398: 7e541f34
	s_wait_alu depctr_sa_sdst(0)                               // 00000000639c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000063a0: bf870001
	v_cndmask_b32_e64 v42, v42, 0, s2                          // 0000000063a4: d501002a 0009012a
	s_branch 62920                                             // 0000000063ac: bfa0f5c8 <tessera_rocm_folded_matmul_708d500594ff51c6+0x1fd0>
	v_cvt_f64_f32_e32 v[52:53], v44                            // 0000000063b0: 7e68212c
	v_cvt_f64_f32_e32 v[54:55], v61                            // 0000000063b4: 7e6c213d
	v_cvt_f64_f32_e32 v[129:130], v48                          // 0000000063b8: 7f022130
	v_cmp_eq_f32_e64 s2, 0, v44                                // 0000000063bc: d4120002 02025880
	v_cmp_class_f32_e64 s5, v48, 0x1f8                         // 0000000063c4: d47e0005 0201ff30 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000063d0: 8b020502
	v_mul_f64_e32 v[52:53], v[52:53], v[54:55]                 // 0000000063d4: 0c686d34
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063d8: bf870091
	v_mul_f64_e32 v[52:53], v[52:53], v[129:130]               // 0000000063dc: 0c690334
	v_cvt_f32_f64_e32 v43, v[52:53]                            // 0000000063e0: 7e561f34
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000063e8: bf870001
	v_cndmask_b32_e64 v43, v43, 0, s2                          // 0000000063ec: d501002b 0009012b
	s_branch 62914                                             // 0000000063f4: bfa0f5c2 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2000>
	v_cvt_f64_f32_e32 v[52:53], v45                            // 0000000063f8: 7e68212d
	v_cvt_f64_f32_e32 v[54:55], v61                            // 0000000063fc: 7e6c213d
	v_cvt_f64_f32_e32 v[129:130], v49                          // 000000006400: 7f022131
	v_cmp_eq_f32_e64 s2, 0, v45                                // 000000006404: d4120002 02025a80
	v_cmp_class_f32_e64 s5, v49, 0x1f8                         // 00000000640c: d47e0005 0201ff31 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006418: 8b020502
	v_mul_f64_e32 v[52:53], v[52:53], v[54:55]                 // 00000000641c: 0c686d34
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006420: bf870091
	v_mul_f64_e32 v[52:53], v[52:53], v[129:130]               // 000000006424: 0c690334
	v_cvt_f32_f64_e32 v44, v[52:53]                            // 000000006428: 7e581f34
	s_wait_alu depctr_sa_sdst(0)                               // 00000000642c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006430: bf870001
	v_cndmask_b32_e64 v44, v44, 0, s2                          // 000000006434: d501002c 0009012c
	s_branch 62908                                             // 00000000643c: bfa0f5bc <tessera_rocm_folded_matmul_708d500594ff51c6+0x2030>
	v_cvt_f64_f32_e32 v[48:49], v46                            // 000000006440: 7e60212e
	v_cvt_f64_f32_e32 v[52:53], v61                            // 000000006444: 7e68213d
	v_cvt_f64_f32_e32 v[54:55], v50                            // 000000006448: 7e6c2132
	v_cmp_eq_f32_e64 s2, 0, v46                                // 00000000644c: d4120002 02025c80
	v_cmp_class_f32_e64 s5, v50, 0x1f8                         // 000000006454: d47e0005 0201ff32 000001f8
	s_and_b32 s2, s2, s5                                       // 000000006460: 8b020502
	v_mul_f64_e32 v[48:49], v[48:49], v[52:53]                 // 000000006464: 0c606930
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006468: bf870091
	v_mul_f64_e32 v[48:49], v[48:49], v[54:55]                 // 00000000646c: 0c606d30
	v_cvt_f32_f64_e32 v45, v[48:49]                            // 000000006470: 7e5a1f30
	s_wait_alu depctr_sa_sdst(0)                               // 000000006474: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006478: bf870001
	v_cndmask_b32_e64 v52, v45, 0, s2                          // 00000000647c: d5010034 0009012d
	s_branch 62902                                             // 000000006484: bfa0f5b6 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2060>
	v_cvt_f64_f32_e32 v[45:46], v47                            // 000000006488: 7e5a212f
	v_cvt_f64_f32_e32 v[48:49], v61                            // 00000000648c: 7e60213d
	v_cvt_f64_f32_e32 v[53:54], v51                            // 000000006490: 7e6a2133
	v_cmp_eq_f32_e64 s2, 0, v47                                // 000000006494: d4120002 02025e80
	v_cmp_class_f32_e64 s5, v51, 0x1f8                         // 00000000649c: d47e0005 0201ff33 000001f8
	s_and_b32 s2, s2, s5                                       // 0000000064a8: 8b020502
	v_mul_f64_e32 v[45:46], v[45:46], v[48:49]                 // 0000000064ac: 0c5a612d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000064b0: bf870091
	v_mul_f64_e32 v[45:46], v[45:46], v[53:54]                 // 0000000064b4: 0c5a6b2d
	v_cvt_f32_f64_e32 v45, v[45:46]                            // 0000000064b8: 7e5a1f2d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000064c0: bf870001
	v_cndmask_b32_e64 v45, v45, 0, s2                          // 0000000064c4: d501002d 0009012d
	s_branch 62896                                             // 0000000064cc: bfa0f5b0 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2090>
	v_cvt_f64_f32_e32 v[152:153], v32                          // 0000000064d0: 7f302120
	v_cvt_f64_f32_e32 v[154:155], v53                          // 0000000064d4: 7f342135
	v_cvt_f64_f32_e32 v[156:157], v44                          // 0000000064d8: 7f38212c
	v_cmp_eq_f32_e64 s3, 0, v32                                // 0000000064dc: d4120003 02024080
	v_cmp_class_f32_e64 s5, v44, 0x1f8                         // 0000000064e4: d47e0005 0201ff2c 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000064f0: 8b030503
	v_mul_f64_e32 v[152:153], v[152:153], v[154:155]           // 0000000064f4: 0d313598
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000064f8: bf870091
	v_mul_f64_e32 v[152:153], v[152:153], v[156:157]           // 0000000064fc: 0d313998
	v_cvt_f32_f64_e32 v51, v[152:153]                          // 000000006500: 7e661f98
	s_wait_alu depctr_sa_sdst(0)                               // 000000006504: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006508: bf870001
	v_cndmask_b32_e64 v51, v51, 0, s3                          // 00000000650c: d5010033 000d0133
	s_branch 63305                                             // 000000006514: bfa0f749 <tessera_rocm_folded_matmul_708d500594ff51c6+0x273c>
	v_cvt_f64_f32_e32 v[152:153], v33                          // 000000006518: 7f302121
	v_cvt_f64_f32_e32 v[154:155], v53                          // 00000000651c: 7f342135
	v_cvt_f64_f32_e32 v[156:157], v45                          // 000000006520: 7f38212d
	v_cmp_eq_f32_e64 s3, 0, v33                                // 000000006524: d4120003 02024280
	v_cmp_class_f32_e64 s5, v45, 0x1f8                         // 00000000652c: d47e0005 0201ff2d 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006538: 8b030503
	v_mul_f64_e32 v[152:153], v[152:153], v[154:155]           // 00000000653c: 0d313598
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006540: bf870091
	v_mul_f64_e32 v[152:153], v[152:153], v[156:157]           // 000000006544: 0d313998
	v_cvt_f32_f64_e32 v32, v[152:153]                          // 000000006548: 7e401f98
	s_wait_alu depctr_sa_sdst(0)                               // 00000000654c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006550: bf870001
	v_cndmask_b32_e64 v32, v32, 0, s3                          // 000000006554: d5010020 000d0120
	s_branch 63299                                             // 00000000655c: bfa0f743 <tessera_rocm_folded_matmul_708d500594ff51c6+0x276c>
	v_cvt_f64_f32_e32 v[44:45], v34                            // 000000006560: 7e582122
	v_cvt_f64_f32_e32 v[152:153], v53                          // 000000006564: 7f302135
	v_cvt_f64_f32_e32 v[154:155], v46                          // 000000006568: 7f34212e
	v_cmp_eq_f32_e64 s3, 0, v34                                // 00000000656c: d4120003 02024480
	v_cmp_class_f32_e64 s5, v46, 0x1f8                         // 000000006574: d47e0005 0201ff2e 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006580: 8b030503
	v_mul_f64_e32 v[44:45], v[44:45], v[152:153]               // 000000006584: 0c59312c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006588: bf870091
	v_mul_f64_e32 v[44:45], v[44:45], v[154:155]               // 00000000658c: 0c59352c
	v_cvt_f32_f64_e32 v33, v[44:45]                            // 000000006590: 7e421f2c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006594: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006598: bf870001
	v_cndmask_b32_e64 v33, v33, 0, s3                          // 00000000659c: d5010021 000d0121
	s_branch 63293                                             // 0000000065a4: bfa0f73d <tessera_rocm_folded_matmul_708d500594ff51c6+0x279c>
	v_cvt_f64_f32_e32 v[44:45], v35                            // 0000000065a8: 7e582123
	v_cvt_f64_f32_e32 v[152:153], v53                          // 0000000065ac: 7f302135
	v_cvt_f64_f32_e32 v[154:155], v47                          // 0000000065b0: 7f34212f
	v_cmp_eq_f32_e64 s3, 0, v35                                // 0000000065b4: d4120003 02024680
	v_cmp_class_f32_e64 s5, v47, 0x1f8                         // 0000000065bc: d47e0005 0201ff2f 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000065c8: 8b030503
	v_mul_f64_e32 v[44:45], v[44:45], v[152:153]               // 0000000065cc: 0c59312c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065d0: bf870091
	v_mul_f64_e32 v[44:45], v[44:45], v[154:155]               // 0000000065d4: 0c59352c
	v_cvt_f32_f64_e32 v34, v[44:45]                            // 0000000065d8: 7e441f2c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000065dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000065e0: bf870001
	v_cndmask_b32_e64 v34, v34, 0, s3                          // 0000000065e4: d5010022 000d0122
	s_branch 63287                                             // 0000000065ec: bfa0f737 <tessera_rocm_folded_matmul_708d500594ff51c6+0x27cc>
	v_cvt_f64_f32_e32 v[44:45], v36                            // 0000000065f0: 7e582124
	v_cvt_f64_f32_e32 v[46:47], v53                            // 0000000065f4: 7e5c2135
	v_cvt_f64_f32_e32 v[152:153], v40                          // 0000000065f8: 7f302128
	v_cmp_eq_f32_e64 s3, 0, v36                                // 0000000065fc: d4120003 02024880
	v_cmp_class_f32_e64 s5, v40, 0x1f8                         // 000000006604: d47e0005 0201ff28 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006610: 8b030503
	v_mul_f64_e32 v[44:45], v[44:45], v[46:47]                 // 000000006614: 0c585d2c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006618: bf870091
	v_mul_f64_e32 v[44:45], v[44:45], v[152:153]               // 00000000661c: 0c59312c
	v_cvt_f32_f64_e32 v35, v[44:45]                            // 000000006620: 7e461f2c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006624: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006628: bf870001
	v_cndmask_b32_e64 v35, v35, 0, s3                          // 00000000662c: d5010023 000d0123
	s_branch 63281                                             // 000000006634: bfa0f731 <tessera_rocm_folded_matmul_708d500594ff51c6+0x27fc>
	v_cvt_f64_f32_e32 v[44:45], v37                            // 000000006638: 7e582125
	v_cvt_f64_f32_e32 v[46:47], v53                            // 00000000663c: 7e5c2135
	v_cvt_f64_f32_e32 v[152:153], v41                          // 000000006640: 7f302129
	v_cmp_eq_f32_e64 s3, 0, v37                                // 000000006644: d4120003 02024a80
	v_cmp_class_f32_e64 s5, v41, 0x1f8                         // 00000000664c: d47e0005 0201ff29 000001f8
	s_and_b32 s3, s3, s5                                       // 000000006658: 8b030503
	v_mul_f64_e32 v[44:45], v[44:45], v[46:47]                 // 00000000665c: 0c585d2c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006660: bf870091
	v_mul_f64_e32 v[44:45], v[44:45], v[152:153]               // 000000006664: 0c59312c
	v_cvt_f32_f64_e32 v36, v[44:45]                            // 000000006668: 7e481f2c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000666c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006670: bf870001
	v_cndmask_b32_e64 v36, v36, 0, s3                          // 000000006674: d5010024 000d0124
	s_branch 63275                                             // 00000000667c: bfa0f72b <tessera_rocm_folded_matmul_708d500594ff51c6+0x282c>
	v_cvt_f64_f32_e32 v[40:41], v38                            // 000000006680: 7e502126
	v_cvt_f64_f32_e32 v[44:45], v53                            // 000000006684: 7e582135
	v_cvt_f64_f32_e32 v[46:47], v42                            // 000000006688: 7e5c212a
	v_cmp_eq_f32_e64 s3, 0, v38                                // 00000000668c: d4120003 02024c80
	v_cmp_class_f32_e64 s5, v42, 0x1f8                         // 000000006694: d47e0005 0201ff2a 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000066a0: 8b030503
	v_mul_f64_e32 v[40:41], v[40:41], v[44:45]                 // 0000000066a4: 0c505928
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066a8: bf870091
	v_mul_f64_e32 v[40:41], v[40:41], v[46:47]                 // 0000000066ac: 0c505d28
	v_cvt_f32_f64_e32 v37, v[40:41]                            // 0000000066b0: 7e4a1f28
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000066b8: bf870001
	v_cndmask_b32_e64 v55, v37, 0, s3                          // 0000000066bc: d5010037 000d0125
	s_branch 63269                                             // 0000000066c4: bfa0f725 <tessera_rocm_folded_matmul_708d500594ff51c6+0x285c>
	v_cvt_f64_f32_e32 v[37:38], v39                            // 0000000066c8: 7e4a2127
	v_cvt_f64_f32_e32 v[40:41], v53                            // 0000000066cc: 7e502135
	v_cvt_f64_f32_e32 v[44:45], v43                            // 0000000066d0: 7e58212b
	v_cmp_eq_f32_e64 s3, 0, v39                                // 0000000066d4: d4120003 02024e80
	v_cmp_class_f32_e64 s5, v43, 0x1f8                         // 0000000066dc: d47e0005 0201ff2b 000001f8
	s_and_b32 s3, s3, s5                                       // 0000000066e8: 8b030503
	v_mul_f64_e32 v[37:38], v[37:38], v[40:41]                 // 0000000066ec: 0c4a5125
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000066f0: bf870091
	v_mul_f64_e32 v[37:38], v[37:38], v[44:45]                 // 0000000066f4: 0c4a5925
	v_cvt_f32_f64_e32 v37, v[37:38]                            // 0000000066f8: 7e4a1f25
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066fc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006700: bf870001
	v_cndmask_b32_e64 v37, v37, 0, s3                          // 000000006704: d5010025 000d0125
	s_branch 63263                                             // 00000000670c: bfa0f71f <tessera_rocm_folded_matmul_708d500594ff51c6+0x288c>
	v_cvt_f64_f32_e32 v[114:115], v24                          // 000000006710: 7ee42118
	v_cvt_f64_f32_e32 v[160:161], v53                          // 000000006714: 7f402135
	v_cvt_f64_f32_e32 v[162:163], v36                          // 000000006718: 7f442124
	v_cmp_eq_f32_e32 vcc_lo, 0, v24                            // 00000000671c: 7c243080
	v_cmp_class_f32_e64 s5, v36, 0x1f8                         // 000000006720: d47e0005 0201ff24 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 00000000672c: 8b05056a
	v_mul_f64_e32 v[114:115], v[114:115], v[160:161]           // 000000006730: 0ce54172
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006734: bf870091
	v_mul_f64_e32 v[114:115], v[114:115], v[162:163]           // 000000006738: 0ce54572
	v_cvt_f32_f64_e32 v51, v[114:115]                          // 00000000673c: 7e661f72
	s_wait_alu depctr_sa_sdst(0)                               // 000000006740: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006744: bf870001
	v_cndmask_b32_e64 v51, v51, 0, s5                          // 000000006748: d5010033 00150133
	s_branch 63648                                             // 000000006750: bfa0f8a0 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2ed4>
	v_cvt_f64_f32_e32 v[114:115], v25                          // 000000006754: 7ee42119
	v_cvt_f64_f32_e32 v[160:161], v53                          // 000000006758: 7f402135
	v_cvt_f64_f32_e32 v[162:163], v37                          // 00000000675c: 7f442125
	v_cmp_eq_f32_e32 vcc_lo, 0, v25                            // 000000006760: 7c243280
	v_cmp_class_f32_e64 s5, v37, 0x1f8                         // 000000006764: d47e0005 0201ff25 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006770: 8b05056a
	v_mul_f64_e32 v[114:115], v[114:115], v[160:161]           // 000000006774: 0ce54172
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006778: bf870091
	v_mul_f64_e32 v[114:115], v[114:115], v[162:163]           // 00000000677c: 0ce54572
	v_cvt_f32_f64_e32 v24, v[114:115]                          // 000000006780: 7e301f72
	s_wait_alu depctr_sa_sdst(0)                               // 000000006784: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006788: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s5                          // 00000000678c: d5010018 00150118
	s_branch 63643                                             // 000000006794: bfa0f89b <tessera_rocm_folded_matmul_708d500594ff51c6+0x2f04>
	v_cvt_f64_f32_e32 v[36:37], v26                            // 000000006798: 7e48211a
	v_cvt_f64_f32_e32 v[114:115], v53                          // 00000000679c: 7ee42135
	v_cvt_f64_f32_e32 v[160:161], v38                          // 0000000067a0: 7f402126
	v_cmp_eq_f32_e32 vcc_lo, 0, v26                            // 0000000067a4: 7c243480
	v_cmp_class_f32_e64 s5, v38, 0x1f8                         // 0000000067a8: d47e0005 0201ff26 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000067b4: 8b05056a
	v_mul_f64_e32 v[36:37], v[36:37], v[114:115]               // 0000000067b8: 0c48e524
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000067bc: bf870091
	v_mul_f64_e32 v[36:37], v[36:37], v[160:161]               // 0000000067c0: 0c494124
	v_cvt_f32_f64_e32 v25, v[36:37]                            // 0000000067c4: 7e321f24
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067c8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000067cc: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s5                          // 0000000067d0: d5010019 00150119
	s_branch 63638                                             // 0000000067d8: bfa0f896 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2f34>
	v_cvt_f64_f32_e32 v[36:37], v27                            // 0000000067dc: 7e48211b
	v_cvt_f64_f32_e32 v[114:115], v53                          // 0000000067e0: 7ee42135
	v_cvt_f64_f32_e32 v[160:161], v39                          // 0000000067e4: 7f402127
	v_cmp_eq_f32_e32 vcc_lo, 0, v27                            // 0000000067e8: 7c243680
	v_cmp_class_f32_e64 s5, v39, 0x1f8                         // 0000000067ec: d47e0005 0201ff27 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000067f8: 8b05056a
	v_mul_f64_e32 v[36:37], v[36:37], v[114:115]               // 0000000067fc: 0c48e524
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006800: bf870091
	v_mul_f64_e32 v[36:37], v[36:37], v[160:161]               // 000000006804: 0c494124
	v_cvt_f32_f64_e32 v26, v[36:37]                            // 000000006808: 7e341f24
	s_wait_alu depctr_sa_sdst(0)                               // 00000000680c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006810: bf870001
	v_cndmask_b32_e64 v26, v26, 0, s5                          // 000000006814: d501001a 0015011a
	s_branch 63633                                             // 00000000681c: bfa0f891 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2f64>
	v_cvt_f64_f32_e32 v[36:37], v28                            // 000000006820: 7e48211c
	v_cvt_f64_f32_e32 v[38:39], v53                            // 000000006824: 7e4c2135
	v_cvt_f64_f32_e32 v[114:115], v32                          // 000000006828: 7ee42120
	v_cmp_eq_f32_e32 vcc_lo, 0, v28                            // 00000000682c: 7c243880
	v_cmp_class_f32_e64 s5, v32, 0x1f8                         // 000000006830: d47e0005 0201ff20 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 00000000683c: 8b05056a
	v_mul_f64_e32 v[36:37], v[36:37], v[38:39]                 // 000000006840: 0c484d24
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006844: bf870091
	v_mul_f64_e32 v[36:37], v[36:37], v[114:115]               // 000000006848: 0c48e524
	v_cvt_f32_f64_e32 v27, v[36:37]                            // 00000000684c: 7e361f24
	s_wait_alu depctr_sa_sdst(0)                               // 000000006850: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006854: bf870001
	v_cndmask_b32_e64 v27, v27, 0, s5                          // 000000006858: d501001b 0015011b
	s_branch 63628                                             // 000000006860: bfa0f88c <tessera_rocm_folded_matmul_708d500594ff51c6+0x2f94>
	v_cvt_f64_f32_e32 v[36:37], v29                            // 000000006864: 7e48211d
	v_cvt_f64_f32_e32 v[38:39], v53                            // 000000006868: 7e4c2135
	v_cvt_f64_f32_e32 v[114:115], v33                          // 00000000686c: 7ee42121
	v_cmp_eq_f32_e32 vcc_lo, 0, v29                            // 000000006870: 7c243a80
	v_cmp_class_f32_e64 s5, v33, 0x1f8                         // 000000006874: d47e0005 0201ff21 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006880: 8b05056a
	v_mul_f64_e32 v[36:37], v[36:37], v[38:39]                 // 000000006884: 0c484d24
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006888: bf870091
	v_mul_f64_e32 v[36:37], v[36:37], v[114:115]               // 00000000688c: 0c48e524
	v_cvt_f32_f64_e32 v28, v[36:37]                            // 000000006890: 7e381f24
	s_wait_alu depctr_sa_sdst(0)                               // 000000006894: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006898: bf870001
	v_cndmask_b32_e64 v28, v28, 0, s5                          // 00000000689c: d501001c 0015011c
	s_branch 63623                                             // 0000000068a4: bfa0f887 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2fc4>
	v_cvt_f64_f32_e32 v[32:33], v30                            // 0000000068a8: 7e40211e
	v_cvt_f64_f32_e32 v[36:37], v53                            // 0000000068ac: 7e482135
	v_cvt_f64_f32_e32 v[38:39], v34                            // 0000000068b0: 7e4c2122
	v_cmp_eq_f32_e32 vcc_lo, 0, v30                            // 0000000068b4: 7c243c80
	v_cmp_class_f32_e64 s5, v34, 0x1f8                         // 0000000068b8: d47e0005 0201ff22 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 0000000068c4: 8b05056a
	v_mul_f64_e32 v[32:33], v[32:33], v[36:37]                 // 0000000068c8: 0c404920
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000068cc: bf870091
	v_mul_f64_e32 v[32:33], v[32:33], v[38:39]                 // 0000000068d0: 0c404d20
	v_cvt_f32_f64_e32 v29, v[32:33]                            // 0000000068d4: 7e3a1f20
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068d8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000068dc: bf870001
	v_cndmask_b32_e64 v29, v29, 0, s5                          // 0000000068e0: d501001d 0015011d
	s_branch 63618                                             // 0000000068e8: bfa0f882 <tessera_rocm_folded_matmul_708d500594ff51c6+0x2ff4>
	v_cvt_f64_f32_e32 v[32:33], v31                            // 0000000068ec: 7e40211f
	v_cvt_f64_f32_e32 v[36:37], v53                            // 0000000068f0: 7e482135
	v_cvt_f64_f32_e32 v[38:39], v35                            // 0000000068f4: 7e4c2123
	v_cmp_eq_f32_e32 vcc_lo, 0, v31                            // 0000000068f8: 7c243e80
	v_cmp_class_f32_e64 s5, v35, 0x1f8                         // 0000000068fc: d47e0005 0201ff23 000001f8
	s_and_b32 s5, vcc_lo, s5                                   // 000000006908: 8b05056a
	v_mul_f64_e32 v[32:33], v[32:33], v[36:37]                 // 00000000690c: 0c404920
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006910: bf870091
	v_mul_f64_e32 v[32:33], v[32:33], v[38:39]                 // 000000006914: 0c404d20
	v_cvt_f32_f64_e32 v30, v[32:33]                            // 000000006918: 7e3c1f20
	s_wait_alu depctr_sa_sdst(0)                               // 00000000691c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006920: bf870001
	v_cndmask_b32_e64 v30, v30, 0, s5                          // 000000006924: d501001e 0015011e
	s_branch 63613                                             // 00000000692c: bfa0f87d <tessera_rocm_folded_matmul_708d500594ff51c6+0x3024>
	v_cvt_f64_f32_e32 v[34:35], v16                            // 000000006930: 7e442110
	v_cvt_f64_f32_e32 v[36:37], v33                            // 000000006934: 7e482121
	v_cvt_f64_f32_e32 v[38:39], v28                            // 000000006938: 7e4c211c
	v_cmp_eq_f32_e32 vcc_lo, 0, v16                            // 00000000693c: 7c242080
	v_cmp_class_f32_e64 s4, v28, 0x1f8                         // 000000006940: d47e0004 0201ff1c 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 00000000694c: 8b04046a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 000000006950: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006954: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 000000006958: 0c444d22
	v_cvt_f32_f64_e32 v32, v[34:35]                            // 00000000695c: 7e401f22
	s_wait_alu depctr_sa_sdst(0)                               // 000000006960: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006964: bf870001
	v_cndmask_b32_e64 v32, v32, 0, s4                          // 000000006968: d5010020 00110120
	s_branch 63879                                             // 000000006970: bfa0f987 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3490>
	v_cvt_f64_f32_e32 v[34:35], v17                            // 000000006974: 7e442111
	v_cvt_f64_f32_e32 v[36:37], v33                            // 000000006978: 7e482121
	v_cvt_f64_f32_e32 v[38:39], v29                            // 00000000697c: 7e4c211d
	v_cmp_eq_f32_e32 vcc_lo, 0, v17                            // 000000006980: 7c242280
	v_cmp_class_f32_e64 s4, v29, 0x1f8                         // 000000006984: d47e0004 0201ff1d 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006990: 8b04046a
	v_mul_f64_e32 v[34:35], v[34:35], v[36:37]                 // 000000006994: 0c444922
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006998: bf870091
	v_mul_f64_e32 v[34:35], v[34:35], v[38:39]                 // 00000000699c: 0c444d22
	v_cvt_f32_f64_e32 v16, v[34:35]                            // 0000000069a0: 7e201f22
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069a8: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s4                          // 0000000069ac: d5010010 00110110
	s_branch 63874                                             // 0000000069b4: bfa0f982 <tessera_rocm_folded_matmul_708d500594ff51c6+0x34c0>
	v_cvt_f64_f32_e32 v[28:29], v18                            // 0000000069b8: 7e382112
	v_cvt_f64_f32_e32 v[34:35], v33                            // 0000000069bc: 7e442121
	v_cvt_f64_f32_e32 v[36:37], v30                            // 0000000069c0: 7e48211e
	v_cmp_eq_f32_e32 vcc_lo, 0, v18                            // 0000000069c4: 7c242480
	v_cmp_class_f32_e64 s4, v30, 0x1f8                         // 0000000069c8: d47e0004 0201ff1e 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 0000000069d4: 8b04046a
	v_mul_f64_e32 v[28:29], v[28:29], v[34:35]                 // 0000000069d8: 0c38451c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069dc: bf870091
	v_mul_f64_e32 v[28:29], v[28:29], v[36:37]                 // 0000000069e0: 0c38491c
	v_cvt_f32_f64_e32 v17, v[28:29]                            // 0000000069e4: 7e221f1c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069e8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000069ec: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s4                          // 0000000069f0: d5010011 00110111
	s_branch 63869                                             // 0000000069f8: bfa0f97d <tessera_rocm_folded_matmul_708d500594ff51c6+0x34f0>
	v_cvt_f64_f32_e32 v[28:29], v19                            // 0000000069fc: 7e382113
	v_cvt_f64_f32_e32 v[34:35], v33                            // 000000006a00: 7e442121
	v_cvt_f64_f32_e32 v[36:37], v31                            // 000000006a04: 7e48211f
	v_cmp_eq_f32_e32 vcc_lo, 0, v19                            // 000000006a08: 7c242680
	v_cmp_class_f32_e64 s4, v31, 0x1f8                         // 000000006a0c: d47e0004 0201ff1f 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006a18: 8b04046a
	v_mul_f64_e32 v[28:29], v[28:29], v[34:35]                 // 000000006a1c: 0c38451c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a20: bf870091
	v_mul_f64_e32 v[28:29], v[28:29], v[36:37]                 // 000000006a24: 0c38491c
	v_cvt_f32_f64_e32 v18, v[28:29]                            // 000000006a28: 7e241f1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a2c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a30: bf870001
	v_cndmask_b32_e64 v18, v18, 0, s4                          // 000000006a34: d5010012 00110112
	s_branch 63864                                             // 000000006a3c: bfa0f978 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3520>
	v_cvt_f64_f32_e32 v[28:29], v20                            // 000000006a40: 7e382114
	v_cvt_f64_f32_e32 v[30:31], v33                            // 000000006a44: 7e3c2121
	v_cvt_f64_f32_e32 v[34:35], v24                            // 000000006a48: 7e442118
	v_cmp_eq_f32_e32 vcc_lo, 0, v20                            // 000000006a4c: 7c242880
	v_cmp_class_f32_e64 s4, v24, 0x1f8                         // 000000006a50: d47e0004 0201ff18 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006a5c: 8b04046a
	v_mul_f64_e32 v[28:29], v[28:29], v[30:31]                 // 000000006a60: 0c383d1c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006a64: bf870091
	v_mul_f64_e32 v[28:29], v[28:29], v[34:35]                 // 000000006a68: 0c38451c
	v_cvt_f32_f64_e32 v19, v[28:29]                            // 000000006a6c: 7e261f1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a70: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a74: bf870001
	v_cndmask_b32_e64 v19, v19, 0, s4                          // 000000006a78: d5010013 00110113
	s_branch 63859                                             // 000000006a80: bfa0f973 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3550>
	v_cvt_f64_f32_e32 v[28:29], v21                            // 000000006a84: 7e382115
	v_cvt_f64_f32_e32 v[30:31], v33                            // 000000006a88: 7e3c2121
	v_cvt_f64_f32_e32 v[34:35], v25                            // 000000006a8c: 7e442119
	v_cmp_eq_f32_e32 vcc_lo, 0, v21                            // 000000006a90: 7c242a80
	v_cmp_class_f32_e64 s4, v25, 0x1f8                         // 000000006a94: d47e0004 0201ff19 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006aa0: 8b04046a
	v_mul_f64_e32 v[28:29], v[28:29], v[30:31]                 // 000000006aa4: 0c383d1c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006aa8: bf870091
	v_mul_f64_e32 v[28:29], v[28:29], v[34:35]                 // 000000006aac: 0c38451c
	v_cvt_f32_f64_e32 v20, v[28:29]                            // 000000006ab0: 7e281f1c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ab4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ab8: bf870001
	v_cndmask_b32_e64 v20, v20, 0, s4                          // 000000006abc: d5010014 00110114
	s_branch 63854                                             // 000000006ac4: bfa0f96e <tessera_rocm_folded_matmul_708d500594ff51c6+0x3580>
	v_cvt_f64_f32_e32 v[24:25], v22                            // 000000006ac8: 7e302116
	v_cvt_f64_f32_e32 v[28:29], v33                            // 000000006acc: 7e382121
	v_cvt_f64_f32_e32 v[30:31], v26                            // 000000006ad0: 7e3c211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v22                            // 000000006ad4: 7c242c80
	v_cmp_class_f32_e64 s4, v26, 0x1f8                         // 000000006ad8: d47e0004 0201ff1a 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006ae4: 8b04046a
	v_mul_f64_e32 v[24:25], v[24:25], v[28:29]                 // 000000006ae8: 0c303918
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006aec: bf870091
	v_mul_f64_e32 v[24:25], v[24:25], v[30:31]                 // 000000006af0: 0c303d18
	v_cvt_f32_f64_e32 v21, v[24:25]                            // 000000006af4: 7e2a1f18
	s_wait_alu depctr_sa_sdst(0)                               // 000000006af8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006afc: bf870001
	v_cndmask_b32_e64 v21, v21, 0, s4                          // 000000006b00: d5010015 00110115
	s_branch 63849                                             // 000000006b08: bfa0f969 <tessera_rocm_folded_matmul_708d500594ff51c6+0x35b0>
	v_cvt_f64_f32_e32 v[24:25], v23                            // 000000006b0c: 7e302117
	v_cvt_f64_f32_e32 v[28:29], v33                            // 000000006b10: 7e382121
	v_cvt_f64_f32_e32 v[30:31], v27                            // 000000006b14: 7e3c211b
	v_cmp_eq_f32_e32 vcc_lo, 0, v23                            // 000000006b18: 7c242e80
	v_cmp_class_f32_e64 s4, v27, 0x1f8                         // 000000006b1c: d47e0004 0201ff1b 000001f8
	s_and_b32 s4, vcc_lo, s4                                   // 000000006b28: 8b04046a
	v_mul_f64_e32 v[24:25], v[24:25], v[28:29]                 // 000000006b2c: 0c303918
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b30: bf870091
	v_mul_f64_e32 v[24:25], v[24:25], v[30:31]                 // 000000006b34: 0c303d18
	v_cvt_f32_f64_e32 v22, v[24:25]                            // 000000006b38: 7e2c1f18
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b3c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b40: bf870001
	v_cndmask_b32_e64 v22, v22, 0, s4                          // 000000006b44: d5010016 00110116
	s_branch 63844                                             // 000000006b4c: bfa0f964 <tessera_rocm_folded_matmul_708d500594ff51c6+0x35e0>
	v_cvt_f64_f32_e32 v[26:27], v8                             // 000000006b50: 7e342108
	v_cvt_f64_f32_e32 v[28:29], v25                            // 000000006b54: 7e382119
	v_cvt_f64_f32_e32 v[30:31], v20                            // 000000006b58: 7e3c2114
	v_cmp_eq_f32_e32 vcc_lo, 0, v8                             // 000000006b5c: 7c241080
	v_cmp_class_f32_e64 s1, v20, 0x1f8                         // 000000006b60: d47e0001 0201ff14 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006b6c: 8b01016a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006b70: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006b74: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006b78: 0c343d1a
	v_cvt_f32_f64_e32 v24, v[26:27]                            // 000000006b7c: 7e301f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b80: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006b84: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s1                          // 000000006b88: d5010018 00050118
	s_branch 64110                                             // 000000006b90: bfa0fa6e <tessera_rocm_folded_matmul_708d500594ff51c6+0x3a4c>
	v_cvt_f64_f32_e32 v[26:27], v9                             // 000000006b94: 7e342109
	v_cvt_f64_f32_e32 v[28:29], v25                            // 000000006b98: 7e382119
	v_cvt_f64_f32_e32 v[30:31], v21                            // 000000006b9c: 7e3c2115
	v_cmp_eq_f32_e32 vcc_lo, 0, v9                             // 000000006ba0: 7c241280
	v_cmp_class_f32_e64 s1, v21, 0x1f8                         // 000000006ba4: d47e0001 0201ff15 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006bb0: 8b01016a
	v_mul_f64_e32 v[26:27], v[26:27], v[28:29]                 // 000000006bb4: 0c34391a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bb8: bf870091
	v_mul_f64_e32 v[26:27], v[26:27], v[30:31]                 // 000000006bbc: 0c343d1a
	v_cvt_f32_f64_e32 v8, v[26:27]                             // 000000006bc0: 7e101f1a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006bc4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006bc8: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s1                            // 000000006bcc: d5010008 00050108
	s_branch 64105                                             // 000000006bd4: bfa0fa69 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3a7c>
	v_cvt_f64_f32_e32 v[20:21], v10                            // 000000006bd8: 7e28210a
	v_cvt_f64_f32_e32 v[26:27], v25                            // 000000006bdc: 7e342119
	v_cvt_f64_f32_e32 v[28:29], v22                            // 000000006be0: 7e382116
	v_cmp_eq_f32_e32 vcc_lo, 0, v10                            // 000000006be4: 7c241480
	v_cmp_class_f32_e64 s1, v22, 0x1f8                         // 000000006be8: d47e0001 0201ff16 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006bf4: 8b01016a
	v_mul_f64_e32 v[20:21], v[20:21], v[26:27]                 // 000000006bf8: 0c283514
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006bfc: bf870091
	v_mul_f64_e32 v[20:21], v[20:21], v[28:29]                 // 000000006c00: 0c283914
	v_cvt_f32_f64_e32 v9, v[20:21]                             // 000000006c04: 7e121f14
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c08: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c0c: bf870001
	v_cndmask_b32_e64 v9, v9, 0, s1                            // 000000006c10: d5010009 00050109
	s_branch 64100                                             // 000000006c18: bfa0fa64 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3aac>
	v_cvt_f64_f32_e32 v[20:21], v11                            // 000000006c1c: 7e28210b
	v_cvt_f64_f32_e32 v[26:27], v25                            // 000000006c20: 7e342119
	v_cvt_f64_f32_e32 v[28:29], v23                            // 000000006c24: 7e382117
	v_cmp_eq_f32_e32 vcc_lo, 0, v11                            // 000000006c28: 7c241680
	v_cmp_class_f32_e64 s1, v23, 0x1f8                         // 000000006c2c: d47e0001 0201ff17 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006c38: 8b01016a
	v_mul_f64_e32 v[20:21], v[20:21], v[26:27]                 // 000000006c3c: 0c283514
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c40: bf870091
	v_mul_f64_e32 v[20:21], v[20:21], v[28:29]                 // 000000006c44: 0c283914
	v_cvt_f32_f64_e32 v10, v[20:21]                            // 000000006c48: 7e141f14
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c4c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c50: bf870001
	v_cndmask_b32_e64 v10, v10, 0, s1                          // 000000006c54: d501000a 0005010a
	s_branch 64095                                             // 000000006c5c: bfa0fa5f <tessera_rocm_folded_matmul_708d500594ff51c6+0x3adc>
	v_cvt_f64_f32_e32 v[20:21], v12                            // 000000006c60: 7e28210c
	v_cvt_f64_f32_e32 v[22:23], v25                            // 000000006c64: 7e2c2119
	v_cvt_f64_f32_e32 v[26:27], v16                            // 000000006c68: 7e342110
	v_cmp_eq_f32_e32 vcc_lo, 0, v12                            // 000000006c6c: 7c241880
	v_cmp_class_f32_e64 s1, v16, 0x1f8                         // 000000006c70: d47e0001 0201ff10 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006c7c: 8b01016a
	v_mul_f64_e32 v[20:21], v[20:21], v[22:23]                 // 000000006c80: 0c282d14
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006c84: bf870091
	v_mul_f64_e32 v[20:21], v[20:21], v[26:27]                 // 000000006c88: 0c283514
	v_cvt_f32_f64_e32 v11, v[20:21]                            // 000000006c8c: 7e161f14
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c90: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006c94: bf870001
	v_cndmask_b32_e64 v11, v11, 0, s1                          // 000000006c98: d501000b 0005010b
	s_branch 64090                                             // 000000006ca0: bfa0fa5a <tessera_rocm_folded_matmul_708d500594ff51c6+0x3b0c>
	v_cvt_f64_f32_e32 v[20:21], v13                            // 000000006ca4: 7e28210d
	v_cvt_f64_f32_e32 v[22:23], v25                            // 000000006ca8: 7e2c2119
	v_cvt_f64_f32_e32 v[26:27], v17                            // 000000006cac: 7e342111
	v_cmp_eq_f32_e32 vcc_lo, 0, v13                            // 000000006cb0: 7c241a80
	v_cmp_class_f32_e64 s1, v17, 0x1f8                         // 000000006cb4: d47e0001 0201ff11 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006cc0: 8b01016a
	v_mul_f64_e32 v[20:21], v[20:21], v[22:23]                 // 000000006cc4: 0c282d14
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006cc8: bf870091
	v_mul_f64_e32 v[20:21], v[20:21], v[26:27]                 // 000000006ccc: 0c283514
	v_cvt_f32_f64_e32 v12, v[20:21]                            // 000000006cd0: 7e181f14
	s_wait_alu depctr_sa_sdst(0)                               // 000000006cd4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006cd8: bf870001
	v_cndmask_b32_e64 v12, v12, 0, s1                          // 000000006cdc: d501000c 0005010c
	s_branch 64085                                             // 000000006ce4: bfa0fa55 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3b3c>
	v_cvt_f64_f32_e32 v[16:17], v14                            // 000000006ce8: 7e20210e
	v_cvt_f64_f32_e32 v[20:21], v25                            // 000000006cec: 7e282119
	v_cvt_f64_f32_e32 v[22:23], v18                            // 000000006cf0: 7e2c2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v14                            // 000000006cf4: 7c241c80
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000006cf8: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d04: 8b01016a
	v_mul_f64_e32 v[16:17], v[16:17], v[20:21]                 // 000000006d08: 0c202910
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d0c: bf870091
	v_mul_f64_e32 v[16:17], v[16:17], v[22:23]                 // 000000006d10: 0c202d10
	v_cvt_f32_f64_e32 v13, v[16:17]                            // 000000006d14: 7e1a1f10
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d18: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d1c: bf870001
	v_cndmask_b32_e64 v13, v13, 0, s1                          // 000000006d20: d501000d 0005010d
	s_branch 64080                                             // 000000006d28: bfa0fa50 <tessera_rocm_folded_matmul_708d500594ff51c6+0x3b6c>
	v_cvt_f64_f32_e32 v[16:17], v15                            // 000000006d2c: 7e20210f
	v_cvt_f64_f32_e32 v[20:21], v25                            // 000000006d30: 7e282119
	v_cvt_f64_f32_e32 v[22:23], v19                            // 000000006d34: 7e2c2113
	v_cmp_eq_f32_e32 vcc_lo, 0, v15                            // 000000006d38: 7c241e80
	v_cmp_class_f32_e64 s1, v19, 0x1f8                         // 000000006d3c: d47e0001 0201ff13 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d48: 8b01016a
	v_mul_f64_e32 v[16:17], v[16:17], v[20:21]                 // 000000006d4c: 0c202910
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d50: bf870091
	v_mul_f64_e32 v[16:17], v[16:17], v[22:23]                 // 000000006d54: 0c202d10
	v_cvt_f32_f64_e32 v14, v[16:17]                            // 000000006d58: 7e1c1f10
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d5c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006d60: bf870001
	v_cndmask_b32_e64 v14, v14, 0, s1                          // 000000006d64: d501000e 0005010e
	s_branch 64075                                             // 000000006d6c: bfa0fa4b <tessera_rocm_folded_matmul_708d500594ff51c6+0x3b9c>
	v_cvt_f64_f32_e32 v[18:19], v0                             // 000000006d70: 7e242100
	v_cvt_f64_f32_e32 v[20:21], v17                            // 000000006d74: 7e282111
	v_cvt_f64_f32_e32 v[22:23], v12                            // 000000006d78: 7e2c210c
	v_cmp_eq_f32_e32 vcc_lo, 0, v0                             // 000000006d7c: 7c240080
	v_cmp_class_f32_e64 s1, v12, 0x1f8                         // 000000006d80: d47e0001 0201ff0c 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006d8c: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006d90: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006d94: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006d98: 0c242d12
	v_cvt_f32_f64_e32 v16, v[18:19]                            // 000000006d9c: 7e201f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006da0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006da4: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s1                          // 000000006da8: d5010010 00050110
	s_branch 64341                                             // 000000006db0: bfa0fb55 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4008>
	v_cvt_f64_f32_e32 v[18:19], v1                             // 000000006db4: 7e242101
	v_cvt_f64_f32_e32 v[20:21], v17                            // 000000006db8: 7e282111
	v_cvt_f64_f32_e32 v[22:23], v13                            // 000000006dbc: 7e2c210d
	v_cmp_eq_f32_e32 vcc_lo, 0, v1                             // 000000006dc0: 7c240280
	v_cmp_class_f32_e64 s1, v13, 0x1f8                         // 000000006dc4: d47e0001 0201ff0d 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006dd0: 8b01016a
	v_mul_f64_e32 v[18:19], v[18:19], v[20:21]                 // 000000006dd4: 0c242912
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006dd8: bf870091
	v_mul_f64_e32 v[18:19], v[18:19], v[22:23]                 // 000000006ddc: 0c242d12
	v_cvt_f32_f64_e32 v0, v[18:19]                             // 000000006de0: 7e001f12
	s_wait_alu depctr_sa_sdst(0)                               // 000000006de4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006de8: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s1                            // 000000006dec: d5010000 00050100
	s_branch 64336                                             // 000000006df4: bfa0fb50 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4038>
	v_cvt_f64_f32_e32 v[12:13], v2                             // 000000006df8: 7e182102
	v_cvt_f64_f32_e32 v[18:19], v17                            // 000000006dfc: 7e242111
	v_cvt_f64_f32_e32 v[20:21], v14                            // 000000006e00: 7e28210e
	v_cmp_eq_f32_e32 vcc_lo, 0, v2                             // 000000006e04: 7c240480
	v_cmp_class_f32_e64 s1, v14, 0x1f8                         // 000000006e08: d47e0001 0201ff0e 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e14: 8b01016a
	v_mul_f64_e32 v[12:13], v[12:13], v[18:19]                 // 000000006e18: 0c18250c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e1c: bf870091
	v_mul_f64_e32 v[12:13], v[12:13], v[20:21]                 // 000000006e20: 0c18290c
	v_cvt_f32_f64_e32 v1, v[12:13]                             // 000000006e24: 7e021f0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e28: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e2c: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s1                            // 000000006e30: d5010001 00050101
	s_branch 64331                                             // 000000006e38: bfa0fb4b <tessera_rocm_folded_matmul_708d500594ff51c6+0x4068>
	v_cvt_f64_f32_e32 v[12:13], v3                             // 000000006e3c: 7e182103
	v_cvt_f64_f32_e32 v[18:19], v17                            // 000000006e40: 7e242111
	v_cvt_f64_f32_e32 v[20:21], v15                            // 000000006e44: 7e28210f
	v_cmp_eq_f32_e32 vcc_lo, 0, v3                             // 000000006e48: 7c240680
	v_cmp_class_f32_e64 s1, v15, 0x1f8                         // 000000006e4c: d47e0001 0201ff0f 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e58: 8b01016a
	v_mul_f64_e32 v[12:13], v[12:13], v[18:19]                 // 000000006e5c: 0c18250c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006e60: bf870091
	v_mul_f64_e32 v[12:13], v[12:13], v[20:21]                 // 000000006e64: 0c18290c
	v_cvt_f32_f64_e32 v2, v[12:13]                             // 000000006e68: 7e041f0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e6c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006e70: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s1                            // 000000006e74: d5010002 00050102
	s_branch 64326                                             // 000000006e7c: bfa0fb46 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4098>
	v_cvt_f64_f32_e32 v[12:13], v4                             // 000000006e80: 7e182104
	v_cvt_f64_f32_e32 v[14:15], v17                            // 000000006e84: 7e1c2111
	v_cvt_f64_f32_e32 v[18:19], v8                             // 000000006e88: 7e242108
	v_cmp_eq_f32_e32 vcc_lo, 0, v4                             // 000000006e8c: 7c240880
	v_cmp_class_f32_e64 s1, v8, 0x1f8                          // 000000006e90: d47e0001 0201ff08 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006e9c: 8b01016a
	v_mul_f64_e32 v[12:13], v[12:13], v[14:15]                 // 000000006ea0: 0c181d0c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ea4: bf870091
	v_mul_f64_e32 v[12:13], v[12:13], v[18:19]                 // 000000006ea8: 0c18250c
	v_cvt_f32_f64_e32 v3, v[12:13]                             // 000000006eac: 7e061f0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006eb0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006eb4: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s1                            // 000000006eb8: d5010003 00050103
	s_branch 64321                                             // 000000006ec0: bfa0fb41 <tessera_rocm_folded_matmul_708d500594ff51c6+0x40c8>
	v_cvt_f64_f32_e32 v[12:13], v5                             // 000000006ec4: 7e182105
	v_cvt_f64_f32_e32 v[14:15], v17                            // 000000006ec8: 7e1c2111
	v_cvt_f64_f32_e32 v[18:19], v9                             // 000000006ecc: 7e242109
	v_cmp_eq_f32_e32 vcc_lo, 0, v5                             // 000000006ed0: 7c240a80
	v_cmp_class_f32_e64 s1, v9, 0x1f8                          // 000000006ed4: d47e0001 0201ff09 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006ee0: 8b01016a
	v_mul_f64_e32 v[12:13], v[12:13], v[14:15]                 // 000000006ee4: 0c181d0c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ee8: bf870091
	v_mul_f64_e32 v[12:13], v[12:13], v[18:19]                 // 000000006eec: 0c18250c
	v_cvt_f32_f64_e32 v4, v[12:13]                             // 000000006ef0: 7e081f0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ef4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006ef8: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s1                            // 000000006efc: d5010004 00050104
	s_branch 64316                                             // 000000006f04: bfa0fb3c <tessera_rocm_folded_matmul_708d500594ff51c6+0x40f8>
	v_cvt_f64_f32_e32 v[8:9], v6                               // 000000006f08: 7e102106
	v_cvt_f64_f32_e32 v[12:13], v17                            // 000000006f0c: 7e182111
	v_cvt_f64_f32_e32 v[14:15], v10                            // 000000006f10: 7e1c210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v6                             // 000000006f14: 7c240c80
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 000000006f18: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006f24: 8b01016a
	v_mul_f64_e32 v[8:9], v[8:9], v[12:13]                     // 000000006f28: 0c101908
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f2c: bf870091
	v_mul_f64_e32 v[8:9], v[8:9], v[14:15]                     // 000000006f30: 0c101d08
	v_cvt_f32_f64_e32 v5, v[8:9]                               // 000000006f34: 7e0a1f08
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f38: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f3c: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s1                            // 000000006f40: d5010005 00050105
	s_branch 64311                                             // 000000006f48: bfa0fb37 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4128>
	v_cvt_f64_f32_e32 v[8:9], v7                               // 000000006f4c: 7e102107
	v_cvt_f64_f32_e32 v[12:13], v17                            // 000000006f50: 7e182111
	v_cvt_f64_f32_e32 v[14:15], v11                            // 000000006f54: 7e1c210b
	v_cmp_eq_f32_e32 vcc_lo, 0, v7                             // 000000006f58: 7c240e80
	v_cmp_class_f32_e64 s1, v11, 0x1f8                         // 000000006f5c: d47e0001 0201ff0b 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006f68: 8b01016a
	v_mul_f64_e32 v[8:9], v[8:9], v[12:13]                     // 000000006f6c: 0c101908
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006f70: bf870091
	v_mul_f64_e32 v[8:9], v[8:9], v[14:15]                     // 000000006f74: 0c101d08
	v_cvt_f32_f64_e32 v6, v[8:9]                               // 000000006f78: 7e0c1f08
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f7c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006f80: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s1                            // 000000006f84: d5010006 00050106
	s_branch 64306                                             // 000000006f8c: bfa0fb32 <tessera_rocm_folded_matmul_708d500594ff51c6+0x4158>
	s_code_end                                                 // 000000006f90: bf9f0000
	s_code_end                                                 // 000000006f94: bf9f0000
	s_code_end                                                 // 000000006f98: bf9f0000
	s_code_end                                                 // 000000006f9c: bf9f0000
	s_code_end                                                 // 000000006fa0: bf9f0000
	s_code_end                                                 // 000000006fa4: bf9f0000
	s_code_end                                                 // 000000006fa8: bf9f0000
	s_code_end                                                 // 000000006fac: bf9f0000
	s_code_end                                                 // 000000006fb0: bf9f0000
	s_code_end                                                 // 000000006fb4: bf9f0000
	s_code_end                                                 // 000000006fb8: bf9f0000
	s_code_end                                                 // 000000006fbc: bf9f0000
	s_code_end                                                 // 000000006fc0: bf9f0000
	s_code_end                                                 // 000000006fc4: bf9f0000
	s_code_end                                                 // 000000006fc8: bf9f0000
	s_code_end                                                 // 000000006fcc: bf9f0000
	s_code_end                                                 // 000000006fd0: bf9f0000
	s_code_end                                                 // 000000006fd4: bf9f0000
	s_code_end                                                 // 000000006fd8: bf9f0000
	s_code_end                                                 // 000000006fdc: bf9f0000
	s_code_end                                                 // 000000006fe0: bf9f0000
	s_code_end                                                 // 000000006fe4: bf9f0000
	s_code_end                                                 // 000000006fe8: bf9f0000
	s_code_end                                                 // 000000006fec: bf9f0000
	s_code_end                                                 // 000000006ff0: bf9f0000
	s_code_end                                                 // 000000006ff4: bf9f0000
	s_code_end                                                 // 000000006ff8: bf9f0000
	s_code_end                                                 // 000000006ffc: bf9f0000
	s_code_end                                                 // 000000007000: bf9f0000
	s_code_end                                                 // 000000007004: bf9f0000
	s_code_end                                                 // 000000007008: bf9f0000
	s_code_end                                                 // 00000000700c: bf9f0000
	s_code_end                                                 // 000000007010: bf9f0000
	s_code_end                                                 // 000000007014: bf9f0000
	s_code_end                                                 // 000000007018: bf9f0000
	s_code_end                                                 // 00000000701c: bf9f0000
	s_code_end                                                 // 000000007020: bf9f0000
	s_code_end                                                 // 000000007024: bf9f0000
	s_code_end                                                 // 000000007028: bf9f0000
	s_code_end                                                 // 00000000702c: bf9f0000
	s_code_end                                                 // 000000007030: bf9f0000
	s_code_end                                                 // 000000007034: bf9f0000
	s_code_end                                                 // 000000007038: bf9f0000
	s_code_end                                                 // 00000000703c: bf9f0000
	s_code_end                                                 // 000000007040: bf9f0000
	s_code_end                                                 // 000000007044: bf9f0000
	s_code_end                                                 // 000000007048: bf9f0000
	s_code_end                                                 // 00000000704c: bf9f0000
	s_code_end                                                 // 000000007050: bf9f0000
	s_code_end                                                 // 000000007054: bf9f0000
	s_code_end                                                 // 000000007058: bf9f0000
	s_code_end                                                 // 00000000705c: bf9f0000
	s_code_end                                                 // 000000007060: bf9f0000
	s_code_end                                                 // 000000007064: bf9f0000
	s_code_end                                                 // 000000007068: bf9f0000
	s_code_end                                                 // 00000000706c: bf9f0000
	s_code_end                                                 // 000000007070: bf9f0000
	s_code_end                                                 // 000000007074: bf9f0000
	s_code_end                                                 // 000000007078: bf9f0000
	s_code_end                                                 // 00000000707c: bf9f0000
	s_code_end                                                 // 000000007080: bf9f0000
	s_code_end                                                 // 000000007084: bf9f0000
	s_code_end                                                 // 000000007088: bf9f0000
	s_code_end                                                 // 00000000708c: bf9f0000
	s_code_end                                                 // 000000007090: bf9f0000
	s_code_end                                                 // 000000007094: bf9f0000
	s_code_end                                                 // 000000007098: bf9f0000
	s_code_end                                                 // 00000000709c: bf9f0000
	s_code_end                                                 // 0000000070a0: bf9f0000
	s_code_end                                                 // 0000000070a4: bf9f0000
	s_code_end                                                 // 0000000070a8: bf9f0000
	s_code_end                                                 // 0000000070ac: bf9f0000
	s_code_end                                                 // 0000000070b0: bf9f0000
	s_code_end                                                 // 0000000070b4: bf9f0000
	s_code_end                                                 // 0000000070b8: bf9f0000
	s_code_end                                                 // 0000000070bc: bf9f0000
	s_code_end                                                 // 0000000070c0: bf9f0000
	s_code_end                                                 // 0000000070c4: bf9f0000
	s_code_end                                                 // 0000000070c8: bf9f0000
	s_code_end                                                 // 0000000070cc: bf9f0000
	s_code_end                                                 // 0000000070d0: bf9f0000
	s_code_end                                                 // 0000000070d4: bf9f0000
	s_code_end                                                 // 0000000070d8: bf9f0000
	s_code_end                                                 // 0000000070dc: bf9f0000
	s_code_end                                                 // 0000000070e0: bf9f0000
	s_code_end                                                 // 0000000070e4: bf9f0000
	s_code_end                                                 // 0000000070e8: bf9f0000
	s_code_end                                                 // 0000000070ec: bf9f0000
	s_code_end                                                 // 0000000070f0: bf9f0000
	s_code_end                                                 // 0000000070f4: bf9f0000
	s_code_end                                                 // 0000000070f8: bf9f0000
	s_code_end                                                 // 0000000070fc: bf9f0000
	s_code_end                                                 // 000000007100: bf9f0000
	s_code_end                                                 // 000000007104: bf9f0000
	s_code_end                                                 // 000000007108: bf9f0000
	s_code_end                                                 // 00000000710c: bf9f0000
	s_code_end                                                 // 000000007110: bf9f0000
	s_code_end                                                 // 000000007114: bf9f0000
	s_code_end                                                 // 000000007118: bf9f0000
	s_code_end                                                 // 00000000711c: bf9f0000
	s_code_end                                                 // 000000007120: bf9f0000
	s_code_end                                                 // 000000007124: bf9f0000
	s_code_end                                                 // 000000007128: bf9f0000
	s_code_end                                                 // 00000000712c: bf9f0000
	s_code_end                                                 // 000000007130: bf9f0000
	s_code_end                                                 // 000000007134: bf9f0000
	s_code_end                                                 // 000000007138: bf9f0000
	s_code_end                                                 // 00000000713c: bf9f0000
	s_code_end                                                 // 000000007140: bf9f0000
	s_code_end                                                 // 000000007144: bf9f0000
	s_code_end                                                 // 000000007148: bf9f0000
	s_code_end                                                 // 00000000714c: bf9f0000
	s_code_end                                                 // 000000007150: bf9f0000
	s_code_end                                                 // 000000007154: bf9f0000
	s_code_end                                                 // 000000007158: bf9f0000
	s_code_end                                                 // 00000000715c: bf9f0000
	s_code_end                                                 // 000000007160: bf9f0000
	s_code_end                                                 // 000000007164: bf9f0000
	s_code_end                                                 // 000000007168: bf9f0000
	s_code_end                                                 // 00000000716c: bf9f0000
	s_code_end                                                 // 000000007170: bf9f0000
	s_code_end                                                 // 000000007174: bf9f0000
	s_code_end                                                 // 000000007178: bf9f0000
	s_code_end                                                 // 00000000717c: bf9f0000
