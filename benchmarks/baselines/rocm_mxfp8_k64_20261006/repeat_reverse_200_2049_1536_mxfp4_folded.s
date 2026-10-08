
/tmp/tmpd927xebn.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_folded_matmul_a629485a5b78541c>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[40:43], s[0:1], 0xc8                         // 000000001b04: f4004a00 f80000c8
	s_load_b64 s[4:5], s[0:1], 0xd8                            // 000000001b0c: f4002100 f80000d8
	s_mov_b32 s6, ttmp9                                        // 000000001b14: be860075
	s_ashr_i32 s7, ttmp9, 31                                   // 000000001b18: 86079f75
	s_wait_kmcnt 0x0                                           // 000000001b1c: bfc70000
	s_add_nc_u64 s[2:3], s[42:43], 63                          // 000000001b20: a982bf2a
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001b24: bf870499
	s_lshr_b64 s[2:3], s[2:3], 4                               // 000000001b28: 85828402
	s_and_b32 s9, s3, 0xfffffff                                // 000000001b2c: 8b09ff03 0fffffff
	s_and_b32 s8, s2, -4                                       // 000000001b34: 8b08c402
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001b38: bf870499
	s_or_b64 s[2:3], s[6:7], s[8:9]                            // 000000001b3c: 8c820806
	s_cmp_lg_u32 s3, 0                                         // 000000001b40: bf078003
	s_mov_b32 s3, 0                                            // 000000001b44: be830080
	s_cbranch_scc0 111                                         // 000000001b48: bfa1006f <tessera_rocm_folded_matmul_a629485a5b78541c+0x208>
	s_cvt_f32_u32 s2, s8                                       // 000000001b4c: be826508
	s_cvt_f32_u32 s10, s9                                      // 000000001b50: be8a6509
	s_sub_nc_u64 s[12:13], 0, s[8:9]                           // 000000001b54: aa0c0880
	s_delay_alu instid0(salu_cycle_2) | instskip(next) | instid1(salu_cycle_3)// 000000001b58: bf87059a
	s_fmac_f32 s2, s10, 0x4f800000                             // 000000001b5c: a382ff0a 4f800000
	v_s_rcp_f32 s2, s2                                         // 000000001b64: d6840002 02010002
	s_delay_alu instid0(trans32_dep_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001b6c: bf870525
	s_mul_f32 s2, s2, 0x5f7ffffc                               // 000000001b70: a202ff02 5f7ffffc
	s_wait_alu depctr_sa_sdst(0)                               // 000000001b78: bf88ff9e
	s_mul_f32 s10, s2, 0x2f800000                              // 000000001b7c: a20aff02 2f800000
	s_delay_alu instid0(salu_cycle_3) | instskip(next) | instid1(salu_cycle_3)// 000000001b84: bf87059b
	s_trunc_f32 s10, s10                                       // 000000001b88: be8a620a
	s_fmac_f32 s2, s10, 0xcf800000                             // 000000001b8c: a382ff0a cf800000
	s_cvt_u32_f32 s11, s10                                     // 000000001b94: be8b670a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001b98: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_3)// 000000001b9c: bf870599
	s_cvt_u32_f32 s10, s2                                      // 000000001ba0: be8a6702
	s_mul_u64 s[14:15], s[12:13], s[10:11]                     // 000000001ba4: aa8e0a0c
	s_delay_alu instid0(salu_cycle_1)                          // 000000001ba8: bf870009
	s_mul_hi_u32 s17, s10, s15                                 // 000000001bac: 96910f0a
	s_mul_i32 s16, s10, s15                                    // 000000001bb0: 96100f0a
	s_mul_hi_u32 s2, s10, s14                                  // 000000001bb4: 96820e0a
	s_mul_i32 s19, s11, s14                                    // 000000001bb8: 96130e0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bbc: bf88ff9e
	s_add_nc_u64 s[16:17], s[2:3], s[16:17]                    // 000000001bc0: a9901002
	s_mul_hi_u32 s18, s11, s14                                 // 000000001bc4: 96920e0b
	s_mul_hi_u32 s20, s11, s15                                 // 000000001bc8: 96940f0b
	s_add_co_u32 s2, s16, s19                                  // 000000001bcc: 80021310
	s_add_co_ci_u32 s2, s17, s18                               // 000000001bd0: 82021211
	s_mul_i32 s14, s11, s15                                    // 000000001bd4: 960e0f0b
	s_add_co_ci_u32 s15, s20, 0                                // 000000001bd8: 820f8014
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bdc: bf88ff9e
	s_add_nc_u64 s[14:15], s[2:3], s[14:15]                    // 000000001be0: a98e0e02
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_1) | instid1(salu_cycle_1)// 000000001be4: bf8704a9
	s_add_co_u32 s10, s10, s14                                 // 000000001be8: 800a0e0a
	s_add_co_ci_u32 s11, s11, s15                              // 000000001bec: 820b0f0b
	s_mul_u64 s[12:13], s[12:13], s[10:11]                     // 000000001bf0: aa8c0a0c
	s_delay_alu instid0(salu_cycle_1)                          // 000000001bf4: bf870009
	s_mul_hi_u32 s15, s10, s13                                 // 000000001bf8: 968f0d0a
	s_mul_i32 s14, s10, s13                                    // 000000001bfc: 960e0d0a
	s_mul_hi_u32 s2, s10, s12                                  // 000000001c00: 96820c0a
	s_mul_i32 s17, s11, s12                                    // 000000001c04: 96110c0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c08: bf88ff9e
	s_add_nc_u64 s[14:15], s[2:3], s[14:15]                    // 000000001c0c: a98e0e02
	s_mul_hi_u32 s16, s11, s12                                 // 000000001c10: 96900c0b
	s_mul_hi_u32 s18, s11, s13                                 // 000000001c14: 96920d0b
	s_add_co_u32 s2, s14, s17                                  // 000000001c18: 8002110e
	s_add_co_ci_u32 s2, s15, s16                               // 000000001c1c: 8202100f
	s_mul_i32 s12, s11, s13                                    // 000000001c20: 960c0d0b
	s_add_co_ci_u32 s13, s18, 0                                // 000000001c24: 820d8012
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c28: bf88ff9e
	s_add_nc_u64 s[12:13], s[2:3], s[12:13]                    // 000000001c2c: a98c0c02
	s_delay_alu instid0(salu_cycle_1)                          // 000000001c30: bf870009
	s_add_co_u32 s10, s10, s12                                 // 000000001c34: 800a0c0a
	s_add_co_ci_u32 s12, s11, s13                              // 000000001c38: 820c0d0b
	s_mul_hi_u32 s2, s6, s10                                   // 000000001c3c: 96820a06
	s_mul_hi_u32 s13, s7, s10                                  // 000000001c40: 968d0a07
	s_mul_i32 s14, s7, s10                                     // 000000001c44: 960e0a07
	s_mul_hi_u32 s11, s6, s12                                  // 000000001c48: 968b0c06
	s_mul_i32 s10, s6, s12                                     // 000000001c4c: 960a0c06
	s_mul_hi_u32 s15, s7, s12                                  // 000000001c50: 968f0c07
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c54: bf88ff9e
	s_add_nc_u64 s[10:11], s[2:3], s[10:11]                    // 000000001c58: a98a0a02
	s_mul_i32 s12, s7, s12                                     // 000000001c5c: 960c0c07
	s_add_co_u32 s2, s10, s14                                  // 000000001c60: 80020e0a
	s_add_co_ci_u32 s2, s11, s13                               // 000000001c64: 82020d0b
	s_add_co_ci_u32 s13, s15, 0                                // 000000001c68: 820d800f
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c6c: bf88ff9e
	s_add_nc_u64 s[10:11], s[2:3], s[12:13]                    // 000000001c70: a98a0c02
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000001c74: bf870499
	s_mul_u64 s[12:13], s[8:9], s[10:11]                       // 000000001c78: aa8c0a08
	s_sub_co_u32 s2, s6, s12                                   // 000000001c7c: 80820c06
	s_cselect_b32 s12, -1, 0                                   // 000000001c80: 980c80c1
	s_sub_co_i32 s14, s7, s13                                  // 000000001c84: 818e0d07
	s_cmp_lg_u32 s12, 0                                        // 000000001c88: bf07800c
	s_sub_co_ci_u32 s14, s14, s9                               // 000000001c8c: 828e090e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c90: bf88ff9e
	s_sub_co_u32 s15, s2, s8                                   // 000000001c94: 808f0802
	s_sub_co_ci_u32 s14, s14, 0                                // 000000001c98: 828e800e
	s_delay_alu instid0(salu_cycle_1)                          // 000000001c9c: bf870009
	s_cmp_ge_u32 s14, s9                                       // 000000001ca0: bf09090e
	s_cselect_b32 s16, -1, 0                                   // 000000001ca4: 981080c1
	s_cmp_ge_u32 s15, s8                                       // 000000001ca8: bf09080f
	s_cselect_b32 s17, -1, 0                                   // 000000001cac: 981180c1
	s_cmp_eq_u32 s14, s9                                       // 000000001cb0: bf06090e
	s_add_nc_u64 s[14:15], s[10:11], 1                         // 000000001cb4: a98e810a
	s_cselect_b32 s18, s17, s16                                // 000000001cb8: 98121011
	s_add_nc_u64 s[16:17], s[10:11], 2                         // 000000001cbc: a990820a
	s_cmp_lg_u32 s18, 0                                        // 000000001cc0: bf078012
	s_cselect_b32 s14, s16, s14                                // 000000001cc4: 980e0e10
	s_cselect_b32 s15, s17, s15                                // 000000001cc8: 980f0f11
	s_cmp_lg_u32 s12, 0                                        // 000000001ccc: bf07800c
	s_sub_co_ci_u32 s12, s7, s13                               // 000000001cd0: 828c0d07
	s_delay_alu instid0(salu_cycle_1)                          // 000000001cd4: bf870009
	s_cmp_ge_u32 s12, s9                                       // 000000001cd8: bf09090c
	s_cselect_b32 s13, -1, 0                                   // 000000001cdc: 980d80c1
	s_cmp_ge_u32 s2, s8                                        // 000000001ce0: bf090802
	s_cselect_b32 s2, -1, 0                                    // 000000001ce4: 980280c1
	s_cmp_eq_u32 s12, s9                                       // 000000001ce8: bf06090c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cec: bf88ff9e
	s_cselect_b32 s2, s2, s13                                  // 000000001cf0: 98020d02
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cf4: bf88ff9e
	s_cmp_lg_u32 s2, 0                                         // 000000001cf8: bf078002
	s_cselect_b32 s11, s15, s11                                // 000000001cfc: 980b0b0f
	s_cselect_b32 s10, s14, s10                                // 000000001d00: 980a0a0e
	s_branch 1                                                 // 000000001d04: bfa00001 <tessera_rocm_folded_matmul_a629485a5b78541c+0x20c>
	s_mov_b32 s3, -1                                           // 000000001d08: be8300c1
	s_delay_alu instid0(salu_cycle_1)                          // 000000001d0c: bf870009
	s_and_b32 s2, s3, exec_lo                                  // 000000001d10: 8b027e03
	s_cselect_b32 s2, 1, 0                                     // 000000001d14: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d18: bf88ff9e
	s_cmp_lg_u32 s2, 1                                         // 000000001d1c: bf078102
	s_cbranch_scc1 32                                          // 000000001d20: bfa20020 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2a4>
	v_cvt_f32_u32_e32 v1, s8                                   // 000000001d24: 7e020c08
	s_sub_co_i32 s3, 0, s8                                     // 000000001d28: 81830880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(trans32_dep_1)// 000000001d2c: bf870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001d30: 7e025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001d34: 100202ff 4f7ffffe
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000001d3c: bf870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001d40: 7e020f01
	v_readfirstlane_b32 s2, v1                                 // 000000001d44: 7e040501
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d48: bf88ff9e
	s_mul_i32 s3, s3, s2                                       // 000000001d4c: 96030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d50: bf88ff9e
	s_mul_hi_u32 s3, s2, s3                                    // 000000001d54: 96830302
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d58: bf88ff9e
	s_add_co_i32 s2, s2, s3                                    // 000000001d5c: 81020302
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d60: bf88ff9e
	s_mul_hi_u32 s2, s6, s2                                    // 000000001d64: 96820206
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d68: bf88ff9e
	s_mul_i32 s3, s2, s8                                       // 000000001d6c: 96030802
	s_add_co_i32 s10, s2, 1                                    // 000000001d70: 810a8102
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d74: bf88ff9e
	s_sub_co_i32 s3, s6, s3                                    // 000000001d78: 81830306
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d7c: bf88ff9e
	s_sub_co_i32 s11, s3, s8                                   // 000000001d80: 818b0803
	s_cmp_ge_u32 s3, s8                                        // 000000001d84: bf090803
	s_cselect_b32 s2, s10, s2                                  // 000000001d88: 9802020a
	s_cselect_b32 s3, s11, s3                                  // 000000001d8c: 9803030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d90: bf88ff9e
	s_add_co_i32 s10, s2, 1                                    // 000000001d94: 810a8102
	s_cmp_ge_u32 s3, s8                                        // 000000001d98: bf090803
	s_mov_b32 s11, 0                                           // 000000001d9c: be8b0080
	s_cselect_b32 s10, s10, s2                                 // 000000001da0: 980a020a
	s_add_nc_u64 s[12:13], s[40:41], 0xff                      // 000000001da4: a98cff28 000000ff
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001dac: 8482820a
	s_lshr_b64 s[12:13], s[12:13], 8                           // 000000001db0: 858c880c
	s_mul_u64 s[8:9], s[10:11], s[8:9]                         // 000000001db4: aa88080a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001db8: bf88ff9e
	s_sub_nc_u64 s[12:13], s[12:13], s[2:3]                    // 000000001dbc: aa0c020c
	s_sub_nc_u64 s[6:7], s[6:7], s[8:9]                        // 000000001dc0: aa060806
	v_cmp_lt_u64_e64 s14, s[12:13], 4                          // 000000001dc4: d459000e 0201080c
	s_mov_b32 s11, 0                                           // 000000001dcc: be8b0080
	s_and_b32 s8, s14, exec_lo                                 // 000000001dd0: 8b087e0e
	s_cselect_b32 s9, s13, 0                                   // 000000001dd4: 9809800d
	s_cselect_b32 s8, s12, 4                                   // 000000001dd8: 9808840c
	s_cmp_lg_u32 s7, 0                                         // 000000001ddc: bf078007
	s_cbranch_scc0 119                                         // 000000001de0: bfa10077 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4c0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001de4: bf88ff9e
	s_cvt_f32_u32 s10, s8                                      // 000000001de8: be8a6508
	s_cvt_f32_u32 s12, s9                                      // 000000001dec: be8c6509
	s_sub_nc_u64 s[14:15], 0, s[8:9]                           // 000000001df0: aa0e0880
	s_wait_alu depctr_sa_sdst(0)                               // 000000001df4: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_3)// 000000001df8: bf870599
	s_fmac_f32 s10, s12, 0x4f800000                            // 000000001dfc: a38aff0c 4f800000
	v_s_rcp_f32 s10, s10                                       // 000000001e04: d684000a 0201000a
	s_delay_alu instid0(trans32_dep_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e0c: bf870525
	s_mul_f32 s10, s10, 0x5f7ffffc                             // 000000001e10: a20aff0a 5f7ffffc
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e18: bf88ff9e
	s_mul_f32 s12, s10, 0x2f800000                             // 000000001e1c: a20cff0a 2f800000
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e24: bf88ff9e
	s_delay_alu instid0(salu_cycle_2) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e28: bf87052a
	s_trunc_f32 s12, s12                                       // 000000001e2c: be8c620c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e30: bf88ff9e
	s_fmac_f32 s10, s12, 0xcf800000                            // 000000001e34: a38aff0c cf800000
	s_cvt_u32_f32 s13, s12                                     // 000000001e3c: be8d670c
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e40: bf88ff9e
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_1) | instid1(salu_cycle_2)// 000000001e44: bf870529
	s_cvt_u32_f32 s12, s10                                     // 000000001e48: be8c670a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e4c: bf88ff9e
	s_mul_u64 s[16:17], s[14:15], s[12:13]                     // 000000001e50: aa900c0e
	s_delay_alu instid0(salu_cycle_1)                          // 000000001e54: bf870009
	s_mul_hi_u32 s19, s12, s17                                 // 000000001e58: 9693110c
	s_mul_i32 s18, s12, s17                                    // 000000001e5c: 9612110c
	s_mul_hi_u32 s10, s12, s16                                 // 000000001e60: 968a100c
	s_mul_i32 s21, s13, s16                                    // 000000001e64: 9615100d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e68: bf88ff9e
	s_add_nc_u64 s[18:19], s[10:11], s[18:19]                  // 000000001e6c: a992120a
	s_mul_hi_u32 s20, s13, s16                                 // 000000001e70: 9694100d
	s_mul_hi_u32 s22, s13, s17                                 // 000000001e74: 9696110d
	s_add_co_u32 s10, s18, s21                                 // 000000001e78: 800a1512
	s_add_co_ci_u32 s10, s19, s20                              // 000000001e7c: 820a1413
	s_mul_i32 s16, s13, s17                                    // 000000001e80: 9610110d
	s_add_co_ci_u32 s17, s22, 0                                // 000000001e84: 82118016
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e88: bf88ff9e
	s_add_nc_u64 s[16:17], s[10:11], s[16:17]                  // 000000001e8c: a990100a
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(salu_cycle_1)// 000000001e90: bf8704c9
	s_add_co_u32 s12, s12, s16                                 // 000000001e94: 800c100c
	s_add_co_ci_u32 s13, s13, s17                              // 000000001e98: 820d110d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e9c: bf88ff9e
	s_mul_u64 s[14:15], s[14:15], s[12:13]                     // 000000001ea0: aa8e0c0e
	s_mul_hi_u32 s17, s12, s15                                 // 000000001ea4: 96910f0c
	s_mul_i32 s16, s12, s15                                    // 000000001ea8: 96100f0c
	s_mul_hi_u32 s10, s12, s14                                 // 000000001eac: 968a0e0c
	s_mul_i32 s19, s13, s14                                    // 000000001eb0: 96130e0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001eb4: bf88ff9e
	s_add_nc_u64 s[16:17], s[10:11], s[16:17]                  // 000000001eb8: a990100a
	s_mul_hi_u32 s18, s13, s14                                 // 000000001ebc: 96920e0d
	s_mul_hi_u32 s20, s13, s15                                 // 000000001ec0: 96940f0d
	s_add_co_u32 s10, s16, s19                                 // 000000001ec4: 800a1310
	s_add_co_ci_u32 s10, s17, s18                              // 000000001ec8: 820a1211
	s_mul_i32 s14, s13, s15                                    // 000000001ecc: 960e0f0d
	s_add_co_ci_u32 s15, s20, 0                                // 000000001ed0: 820f8014
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ed4: bf88ff9e
	s_add_nc_u64 s[14:15], s[10:11], s[14:15]                  // 000000001ed8: a98e0e0a
	s_delay_alu instid0(salu_cycle_1)                          // 000000001edc: bf870009
	s_add_co_u32 s12, s12, s14                                 // 000000001ee0: 800c0e0c
	s_add_co_ci_u32 s14, s13, s15                              // 000000001ee4: 820e0f0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ee8: bf88ff9e
	s_mul_hi_u32 s10, s6, s12                                  // 000000001eec: 968a0c06
	s_mul_hi_u32 s15, s7, s12                                  // 000000001ef0: 968f0c07
	s_mul_i32 s16, s7, s12                                     // 000000001ef4: 96100c07
	s_mul_hi_u32 s13, s6, s14                                  // 000000001ef8: 968d0e06
	s_mul_i32 s12, s6, s14                                     // 000000001efc: 960c0e06
	s_mul_hi_u32 s17, s7, s14                                  // 000000001f00: 96910e07
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f04: bf88ff9e
	s_add_nc_u64 s[12:13], s[10:11], s[12:13]                  // 000000001f08: a98c0c0a
	s_mul_i32 s14, s7, s14                                     // 000000001f0c: 960e0e07
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f10: bf88ff9e
	s_add_co_u32 s10, s12, s16                                 // 000000001f14: 800a100c
	s_add_co_ci_u32 s10, s13, s15                              // 000000001f18: 820a0f0d
	s_add_co_ci_u32 s15, s17, 0                                // 000000001f1c: 820f8011
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f20: bf88ff9e
	s_add_nc_u64 s[12:13], s[10:11], s[14:15]                  // 000000001f24: a98c0e0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f28: bf88ff9e
	s_mul_u64 s[14:15], s[8:9], s[12:13]                       // 000000001f2c: aa8e0c08
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f30: bf870009
	s_sub_co_u32 s10, s6, s14                                  // 000000001f34: 808a0e06
	s_cselect_b32 s14, -1, 0                                   // 000000001f38: 980e80c1
	s_sub_co_i32 s16, s7, s15                                  // 000000001f3c: 81900f07
	s_cmp_lg_u32 s14, 0                                        // 000000001f40: bf07800e
	s_sub_co_ci_u32 s16, s16, s9                               // 000000001f44: 82900910
	s_wait_alu depctr_sa_sdst(0)                               // 000000001f48: bf88ff9e
	s_sub_co_u32 s17, s10, s8                                  // 000000001f4c: 8091080a
	s_sub_co_ci_u32 s16, s16, 0                                // 000000001f50: 82908010
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f54: bf870009
	s_cmp_ge_u32 s16, s9                                       // 000000001f58: bf090910
	s_cselect_b32 s18, -1, 0                                   // 000000001f5c: 981280c1
	s_cmp_ge_u32 s17, s8                                       // 000000001f60: bf090811
	s_cselect_b32 s19, -1, 0                                   // 000000001f64: 981380c1
	s_cmp_eq_u32 s16, s9                                       // 000000001f68: bf060910
	s_add_nc_u64 s[16:17], s[12:13], 1                         // 000000001f6c: a990810c
	s_cselect_b32 s20, s19, s18                                // 000000001f70: 98141213
	s_add_nc_u64 s[18:19], s[12:13], 2                         // 000000001f74: a992820c
	s_cmp_lg_u32 s20, 0                                        // 000000001f78: bf078014
	s_cselect_b32 s16, s18, s16                                // 000000001f7c: 98101012
	s_cselect_b32 s17, s19, s17                                // 000000001f80: 98111113
	s_cmp_lg_u32 s14, 0                                        // 000000001f84: bf07800e
	s_sub_co_ci_u32 s14, s7, s15                               // 000000001f88: 828e0f07
	s_delay_alu instid0(salu_cycle_1)                          // 000000001f8c: bf870009
	s_cmp_ge_u32 s14, s9                                       // 000000001f90: bf09090e
	s_cselect_b32 s15, -1, 0                                   // 000000001f94: 980f80c1
	s_cmp_ge_u32 s10, s8                                       // 000000001f98: bf09080a
	s_cselect_b32 s10, -1, 0                                   // 000000001f9c: 980a80c1
	s_cmp_eq_u32 s14, s9                                       // 000000001fa0: bf06090e
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fa4: bf88ff9e
	s_cselect_b32 s10, s10, s15                                // 000000001fa8: 980a0f0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fac: bf88ff9e
	s_cmp_lg_u32 s10, 0                                        // 000000001fb0: bf07800a
	s_cselect_b32 s13, s17, s13                                // 000000001fb4: 980d0d11
	s_cselect_b32 s12, s16, s12                                // 000000001fb8: 980c0c10
	s_branch 1                                                 // 000000001fbc: bfa00001 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4c4>
	s_mov_b32 s11, -1                                          // 000000001fc0: be8b00c1
	s_delay_alu instid0(salu_cycle_1)                          // 000000001fc4: bf870009
	s_and_b32 s10, s11, exec_lo                                // 000000001fc8: 8b0a7e0b
	s_cselect_b32 s10, 1, 0                                    // 000000001fcc: 980a8081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001fd0: bf88ff9e
	s_cmp_lg_u32 s10, 1                                        // 000000001fd4: bf07810a
	s_cbranch_scc1 33                                          // 000000001fd8: bfa20021 <tessera_rocm_folded_matmul_a629485a5b78541c+0x560>
	v_cvt_f32_u32_e32 v1, s8                                   // 000000001fdc: 7e020c08
	s_sub_co_i32 s11, 0, s8                                    // 000000001fe0: 818b0880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(trans32_dep_1)// 000000001fe4: bf870291
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001fe8: 7e025701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001fec: 100202ff 4f7ffffe
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000001ff4: bf870091
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001ff8: 7e020f01
	v_readfirstlane_b32 s10, v1                                // 000000001ffc: 7e140501
	s_wait_alu depctr_sa_sdst(0)                               // 000000002000: bf88ff9e
	s_mul_i32 s11, s11, s10                                    // 000000002004: 960b0a0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002008: bf88ff9e
	s_mul_hi_u32 s11, s10, s11                                 // 00000000200c: 968b0b0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002010: bf88ff9e
	s_add_co_i32 s10, s10, s11                                 // 000000002014: 810a0b0a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002018: bf88ff9e
	s_mul_hi_u32 s10, s6, s10                                  // 00000000201c: 968a0a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000002020: bf88ff9e
	s_mul_i32 s11, s10, s8                                     // 000000002024: 960b080a
	s_add_co_i32 s12, s10, 1                                   // 000000002028: 810c810a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000202c: bf88ff9e
	s_sub_co_i32 s11, s6, s11                                  // 000000002030: 818b0b06
	s_wait_alu depctr_sa_sdst(0)                               // 000000002034: bf88ff9e
	s_sub_co_i32 s13, s11, s8                                  // 000000002038: 818d080b
	s_cmp_ge_u32 s11, s8                                       // 00000000203c: bf09080b
	s_cselect_b32 s10, s12, s10                                // 000000002040: 980a0a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002044: bf88ff9e
	s_cselect_b32 s11, s13, s11                                // 000000002048: 980b0b0d
	s_add_co_i32 s12, s10, 1                                   // 00000000204c: 810c810a
	s_wait_alu depctr_sa_sdst(0)                               // 000000002050: bf88ff9e
	s_cmp_ge_u32 s11, s8                                       // 000000002054: bf09080b
	s_mov_b32 s13, 0                                           // 000000002058: be8d0080
	s_cselect_b32 s12, s12, s10                                // 00000000205c: 980c0a0c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002060: bf88ff9e
	s_mul_u64 s[8:9], s[12:13], s[8:9]                         // 000000002064: aa88080c
	v_lshrrev_b32_e32 v8, 2, v0                                // 000000002068: 32100082
	s_wait_alu depctr_sa_sdst(0)                               // 00000000206c: bf88ff9e
	s_sub_nc_u64 s[6:7], s[6:7], s[8:9]                        // 000000002070: aa060806
	s_add_nc_u64 s[10:11], s[40:41], -1                        // 000000002074: a98ac128
	s_add_nc_u64 s[2:3], s[6:7], s[2:3]                        // 000000002078: a9820206
	s_clause 0x1                                               // 00000000207c: bf850001
	s_load_b64 s[8:9], s[0:1], 0x8                             // 000000002080: f4002200 f8000008
	s_load_b64 s[6:7], s[0:1], 0x30                            // 000000002088: f4002180 f8000030
	s_lshl_b64 s[44:45], s[2:3], 8                             // 000000002090: 84ac8802
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000002094: bf8701b9
	v_dual_mov_b32 v2, s45 :: v_dual_lshlrev_b32 v5, 4, v0     // 000000002098: ca22002d 02040084
	v_or_b32_e32 v1, s44, v8                                   // 0000000020a0: 3802102c
	v_mov_b32_e32 v4, s45                                      // 0000000020a4: 7e08022d
	v_dual_mov_b32 v6, s45 :: v_dual_and_b32 v9, 48, v5        // 0000000020a8: ca24002d 06080ab0
	v_mov_b32_e32 v5, s45                                      // 0000000020b0: 7e0a022d
	s_delay_alu instid0(valu_dep_4)                            // 0000000020b4: bf870004
	v_or_b32_e32 v3, 64, v1                                    // 0000000020b8: 380602c0
	v_cmp_gt_u64_e32 vcc_lo, s[10:11], v[1:2]                  // 0000000020bc: 7cb8020a
	s_lshl_b64 s[38:39], s[12:13], 6                           // 0000000020c0: 84a6860c
	s_add_nc_u64 s[12:13], s[42:43], -1                        // 0000000020c4: a98cc12a
	s_mov_b32 s3, -1                                           // 0000000020c8: be8300c1
	v_cmp_gt_u64_e64 s2, s[10:11], v[3:4]                      // 0000000020cc: d45c0002 0202060a
	v_or_b32_e32 v4, 0x80, v1                                  // 0000000020d4: 380802ff 00000080
	v_cndmask_b32_e32 v7, s10, v1, vcc_lo                      // 0000000020dc: 020e020a
	v_or_b32_e32 v1, 0xc0, v1                                  // 0000000020e0: 380202ff 000000c0
	v_dual_cndmask_b32 v10, s11, v6 :: v_dual_and_b32 v141, 15, v0// 0000000020e8: ca640c0b 0a8c008f
	s_delay_alu instid0(valu_dep_4)                            // 0000000020f0: bf870004
	v_cmp_gt_u64_e32 vcc_lo, s[10:11], v[4:5]                  // 0000000020f4: 7cb8080a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f8: bf88f19f
	v_cndmask_b32_e64 v11, s10, v3, s2                         // 0000000020fc: d501000b 000a060a
	v_cndmask_b32_e64 v5, s11, v6, s2                          // 000000002104: d5010005 000a0c0b
	v_cmp_gt_u64_e64 s2, s[10:11], v[1:2]                      // 00000000210c: d45c0002 0202020a
	v_or_b32_e32 v2, s38, v8                                   // 000000002114: 38041026
	v_mov_b32_e32 v3, s39                                      // 000000002118: 7e060227
	s_wait_alu depctr_va_vcc(0)                                // 00000000211c: bf88ff9d
	v_cndmask_b32_e32 v12, s10, v4, vcc_lo                     // 000000002120: 0218080a
	v_cndmask_b32_e32 v13, s11, v6, vcc_lo                     // 000000002124: 021a0c0b
	v_mul_lo_u32 v17, v7, s5                                   // 000000002128: d72c0011 02000b07
	s_wait_alu depctr_va_sdst(0)                               // 000000002130: bf88f19f
	v_cndmask_b32_e64 v14, s10, v1, s2                         // 000000002134: d501000e 000a020a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000213c: bf88ff9e
	v_cmp_gt_u64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000002140: 7cb8040c
	v_cndmask_b32_e64 v15, s11, v6, s2                         // 000000002144: d501000f 000a0c0b
	v_mul_lo_u32 v18, v11, s5                                  // 00000000214c: d72c0012 02000b0b
	s_wait_kmcnt 0x0                                           // 000000002154: bfc70000
	v_mad_co_u64_u32 v[3:4], null, v11, s4, s[8:9]             // 000000002158: d6fe7c03 0020090b
	v_mul_lo_u32 v11, v5, s4                                   // 000000002160: d72c000b 02000905
	v_mul_lo_u32 v19, v12, s5                                  // 000000002168: d72c0013 02000b0c
	s_wait_alu depctr_va_vcc(0)                                // 000000002170: bf88ff9d
	v_cndmask_b32_e32 v16, s12, v2, vcc_lo                     // 000000002174: 0220040c
	v_mad_co_u64_u32 v[1:2], null, v7, s4, s[8:9]              // 000000002178: d6fe7c01 00200907
	v_mul_lo_u32 v7, v10, s4                                   // 000000002180: d72c0007 0200090a
	v_mov_b32_e32 v10, s39                                     // 000000002188: 7e140227
	v_mad_co_u64_u32 v[5:6], null, v12, s4, s[8:9]             // 00000000218c: d6fe7c05 0020090c
	v_mul_lo_u32 v12, v13, s4                                  // 000000002194: d72c000c 0200090d
	v_add3_u32 v4, v11, v4, v18                                // 00000000219c: d6550004 044a090b
	v_mul_lo_u32 v13, v15, s4                                  // 0000000021a4: d72c000d 0200090f
	v_cndmask_b32_e32 v10, s13, v10, vcc_lo                    // 0000000021ac: 0214140d
	v_add_co_u32 v89, vcc_lo, v1, v9                           // 0000000021b0: d7006a59 02021301
	v_add3_u32 v2, v7, v2, v17                                 // 0000000021b8: d6550002 04460507
	s_movk_i32 s2, 0xffc0                                      // 0000000021c0: b002ffc0
	v_add3_u32 v11, v12, v6, v19                               // 0000000021c4: d655000b 044e0d0c
	v_mul_lo_u32 v12, v14, s5                                  // 0000000021cc: d72c000c 02000b0e
	v_mad_co_u64_u32 v[6:7], null, v16, s4, s[6:7]             // 0000000021d4: d6fe7c06 00180910
	s_wait_alu depctr_va_vcc(0)                                // 0000000021dc: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, 0, v2, vcc_lo               // 0000000021e0: d5207c5a 01aa0480
	v_mad_co_u64_u32 v[1:2], null, v14, s4, s[8:9]             // 0000000021e8: d6fe7c01 0020090e
	v_mul_lo_u32 v14, v16, s5                                  // 0000000021f0: d72c000e 02000b10
	v_mul_lo_u32 v10, v10, s4                                  // 0000000021f8: d72c000a 0200090a
	v_add_co_u32 v91, vcc_lo, v3, v9                           // 000000002200: d7006a5b 02021303
	s_wait_alu depctr_va_vcc(0)                                // 000000002208: bf88ff9d
	v_add_co_ci_u32_e64 v92, null, 0, v4, vcc_lo               // 00000000220c: d5207c5c 01aa0880
	v_add_co_u32 v93, vcc_lo, v5, v9                           // 000000002214: d7006a5d 02021305
	s_wait_alu depctr_va_vcc(0)                                // 00000000221c: bf88ff9d
	v_add_co_ci_u32_e64 v94, null, 0, v11, vcc_lo              // 000000002220: d5207c5e 01aa1680
	v_add3_u32 v3, v10, v7, v14                                // 000000002228: d6550003 043a0f0a
	v_add_co_u32 v95, vcc_lo, v6, v9                           // 000000002230: d7006a5f 02021306
	v_add3_u32 v2, v13, v2, v12                                // 000000002238: d6550002 0432050d
	v_mul_u32_u24_e32 v10, 0x50, v8                            // 000000002240: 161410ff 00000050
	s_wait_alu depctr_va_vcc(0)                                // 000000002248: bf88ff9d
	v_add_co_ci_u32_e64 v96, null, 0, v3, vcc_lo               // 00000000224c: d5207c60 01aa0680
	v_add_co_u32 v97, vcc_lo, v1, v9                           // 000000002254: d7006a61 02021301
	s_wait_alu depctr_va_vcc(0)                                // 00000000225c: bf88ff9d
	v_add_co_ci_u32_e64 v98, null, 0, v2, vcc_lo               // 000000002260: d5207c62 01aa0480
	global_load_b128 v[17:20], v[95:96], off                   // 000000002268: ee05c07c 00000011 0000005f
	s_clause 0x3                                               // 000000002274: bf850003
	global_load_b128 v[21:24], v[89:90], off                   // 000000002278: ee05c07c 00000015 00000059
	global_load_b128 v[25:28], v[91:92], off                   // 000000002284: ee05c07c 00000019 0000005b
	global_load_b128 v[29:32], v[93:94], off                   // 000000002290: ee05c07c 0000001d 0000005d
	global_load_b128 v[41:44], v[97:98], off                   // 00000000229c: ee05c07c 00000029 00000061
	v_mov_b32_e32 v1, 0                                        // 0000000022a8: 7e020280
	v_lshrrev_b32_e32 v2, 1, v0                                // 0000000022ac: 32040081
	v_and_b32_e32 v3, 0xcf, v0                                 // 0000000022b0: 360600ff 000000cf
	v_and_b32_e32 v85, 0xc0, v0                                // 0000000022b8: 36aa00ff 000000c0
	v_and_b32_e32 v142, 32, v0                                 // 0000000022c0: 371c00a0
	v_dual_mov_b32 v5, v1 :: v_dual_and_b32 v0, 47, v0         // 0000000022c4: ca240101 050000af
	v_dual_mov_b32 v7, v1 :: v_dual_and_b32 v86, 8, v2         // 0000000022cc: ca240101 07560488
	v_mov_b32_e32 v2, v1                                       // 0000000022d4: 7e040301
	v_mov_b32_e32 v4, v1                                       // 0000000022d8: 7e080301
	v_mul_u32_u24_e32 v11, 0x50, v3                            // 0000000022dc: 161606ff 00000050
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v6, v1             // 0000000022e4: ca100101 03060101
	v_mov_b32_e32 v8, v1                                       // 0000000022ec: 7e100301
	v_dual_mov_b32 v37, v5 :: v_dual_add_nc_u32 v100, v10, v9  // 0000000022f0: ca200105 2564130a
	v_or_b32_e32 v99, 16, v85                                  // 0000000022f8: 38c6aa90
	v_or_b32_e32 v111, 32, v85                                 // 0000000022fc: 38deaaa0
	v_or_b32_e32 v121, 48, v85                                 // 000000002300: 38f2aab0
	v_mul_u32_u24_e32 v12, 0x50, v0                            // 000000002304: 161800ff 00000050
	v_or3_b32 v0, v141, v142, 16                               // 00000000230c: d6580000 02431d8d
	v_or_b32_e32 v65, v99, v141                                // 000000002314: 38831b63
	v_or_b32_e32 v66, v111, v141                               // 000000002318: 38851b6f
	v_or_b32_e32 v67, v121, v141                               // 00000000231c: 38871b79
	v_dual_mov_b32 v88, s45 :: v_dual_mov_b32 v39, v7          // 000000002320: ca10002d 58260107
	v_mul_u32_u24_e32 v68, 0x50, v0                            // 000000002328: 168800ff 00000050
	v_mul_u32_u24_e32 v65, 0x50, v65                           // 000000002330: 168282ff 00000050
	v_mul_u32_u24_e32 v66, 0x50, v66                           // 000000002338: 168484ff 00000050
	v_mul_u32_u24_e32 v67, 0x50, v67                           // 000000002340: 168686ff 00000050
	v_or_b32_e32 v87, s44, v85                                 // 000000002348: 38aeaa2c
	v_or_b32_e32 v101, v86, v11                                // 00000000234c: 38ca1756
	v_or_b32_e32 v102, v86, v12                                // 000000002350: 38cc1956
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v35, v3           // 000000002354: ca100108 28220103
	v_dual_mov_b32 v38, v6 :: v_dual_mov_b32 v33, v1           // 00000000235c: ca100106 26200101
	v_dual_mov_b32 v36, v4 :: v_dual_mov_b32 v15, v7           // 000000002364: ca100104 240e0107
	v_dual_mov_b32 v34, v2 :: v_dual_mov_b32 v13, v5           // 00000000236c: ca100102 220c0105
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v11, v3           // 000000002374: ca100108 100a0103
	v_dual_mov_b32 v14, v6 :: v_dual_mov_b32 v9, v1            // 00000000237c: ca100106 0e080101
	v_dual_mov_b32 v12, v4 :: v_dual_mov_b32 v47, v7           // 000000002384: ca100104 0c2e0107
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v45, v5           // 00000000238c: ca100102 0a2c0105
	v_mov_b32_e32 v48, v8                                      // 000000002394: 7e600308
	v_mov_b32_e32 v46, v6                                      // 000000002398: 7e5c0306
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v51, v3           // 00000000239c: ca100108 38320103
	v_dual_mov_b32 v54, v6 :: v_dual_mov_b32 v49, v1           // 0000000023a4: ca100106 36300101
	v_mov_b32_e32 v52, v4                                      // 0000000023ac: 7e680304
	v_mov_b32_e32 v50, v2                                      // 0000000023b0: 7e640302
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v59, v3           // 0000000023b4: ca100108 403a0103
	v_dual_mov_b32 v62, v6 :: v_dual_mov_b32 v57, v1           // 0000000023bc: ca100106 3e380101
	v_mov_b32_e32 v60, v4                                      // 0000000023c4: 7e780304
	v_mov_b32_e32 v58, v2                                      // 0000000023c8: 7e740302
	v_or_b32_e32 v103, v68, v86                                // 0000000023cc: 38cead44
	v_or_b32_e32 v104, v65, v86                                // 0000000023d0: 38d0ad41
	v_or_b32_e32 v105, v66, v86                                // 0000000023d4: 38d2ad42
	v_or_b32_e32 v106, v67, v86                                // 0000000023d8: 38d4ad43
	s_lshr_b64 s[6:7], s[4:5], 6                               // 0000000023dc: 85868604
	s_mov_b64 s[8:9], 64                                       // 0000000023e0: be8801c0
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023e4: bf88ff9e
	s_add_nc_u64 s[4:5], s[4:5], s[2:3]                        // 0000000023e8: a9840204
	v_cmp_gt_u64_e32 vcc_lo, s[40:41], v[87:88]                // 0000000023ec: 7cb8ae28
	s_wait_loadcnt 0x4                                         // 0000000023f0: bfc00004
	ds_store_b128 v100, v[17:20] offset:20480                  // 0000000023f4: db7c5000 00001164
	s_wait_loadcnt 0x3                                         // 0000000023fc: bfc00003
	ds_store_b128 v100, v[21:24]                               // 000000002400: db7c0000 00001564
	s_wait_loadcnt 0x2                                         // 000000002408: bfc00002
	ds_store_b128 v100, v[25:28] offset:5120                   // 00000000240c: db7c1400 00001964
	s_wait_loadcnt 0x1                                         // 000000002414: bfc00001
	ds_store_b128 v100, v[29:32] offset:10240                  // 000000002418: db7c2800 00001d64
	s_wait_loadcnt 0x0                                         // 000000002420: bfc00000
	ds_store_b128 v100, v[41:44] offset:15360                  // 000000002424: db7c3c00 00002964
	v_dual_mov_b32 v44, v4 :: v_dual_mov_b32 v23, v7           // 00000000242c: ca100104 2c160107
	v_mov_b32_e32 v43, v3                                      // 000000002434: 7e560303
	s_wait_dscnt 0x0                                           // 000000002438: bfc60000
	s_barrier_signal -1                                        // 00000000243c: be804ec1
	v_dual_mov_b32 v42, v2 :: v_dual_mov_b32 v21, v5           // 000000002440: ca100102 2a140105
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v24, v8           // 000000002448: ca100101 29180108
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v22, v6           // 000000002450: ca100103 13160106
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v20, v4           // 000000002458: ca100101 11140104
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v18, v2           // 000000002460: ca100107 37120102
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v32, v8           // 000000002468: ca100105 35200108
	v_mov_b32_e32 v27, v3                                      // 000000002470: 7e360303
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6           // 000000002474: ca100107 1f1e0106
	v_mov_b32_e32 v25, v1                                      // 00000000247c: 7e320301
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v28, v4           // 000000002480: ca100105 1d1c0104
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v26, v2           // 000000002488: ca100107 3f1a0102
	v_mov_b32_e32 v61, v5                                      // 000000002490: 7e7a0305
	s_barrier_wait 0xffff                                      // 000000002494: bf94ffff
	s_branch 27                                                // 000000002498: bfa0001b <tessera_rocm_folded_matmul_a629485a5b78541c+0xa08>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000249c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000024a0: 8c7e027e
	s_barrier_signal -1                                        // 0000000024a4: be804ec1
	s_add_nc_u64 s[6:7], s[6:7], -1                            // 0000000024a8: a986c106
	s_add_nc_u64 s[8:9], s[8:9], 64                            // 0000000024ac: a988c008
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024b0: bf88ff9e
	s_cmp_lg_u64 s[6:7], 0                                     // 0000000024b4: bf118006
	s_barrier_wait 0xffff                                      // 0000000024b8: bf94ffff
	s_wait_loadcnt 0x4                                         // 0000000024bc: bfc00004
	ds_store_b128 v100, v[65:68]                               // 0000000024c0: db7c0000 00004164
	s_wait_loadcnt 0x3                                         // 0000000024c8: bfc00003
	ds_store_b128 v100, v[69:72] offset:5120                   // 0000000024cc: db7c1400 00004564
	s_wait_loadcnt 0x2                                         // 0000000024d4: bfc00002
	ds_store_b128 v100, v[73:76] offset:10240                  // 0000000024d8: db7c2800 00004964
	s_wait_loadcnt 0x1                                         // 0000000024e0: bfc00001
	ds_store_b128 v100, v[77:80] offset:15360                  // 0000000024e4: db7c3c00 00004d64
	s_wait_loadcnt 0x0                                         // 0000000024ec: bfc00000
	ds_store_b128 v100, v[81:84] offset:20480                  // 0000000024f0: db7c5000 00005164
	s_wait_dscnt 0x0                                           // 0000000024f8: bfc60000
	s_barrier_signal -1                                        // 0000000024fc: be804ec1
	s_barrier_wait 0xffff                                      // 000000002500: bf94ffff
	s_cbranch_scc0 152                                         // 000000002504: bfa10098 <tessera_rocm_folded_matmul_a629485a5b78541c+0xc68>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002508: bf88ff9e
	v_cmp_lt_u64_e64 s2, s[8:9], s[4:5]                        // 00000000250c: d4590002 02000808
	s_and_b32 s2, s2, exec_lo                                  // 000000002514: 8b027e02
	s_cselect_b32 s10, s8, s4                                  // 000000002518: 980a0408
	s_cselect_b32 s3, s9, s5                                   // 00000000251c: 98030509
	s_wait_alu depctr_sa_sdst(0)                               // 000000002520: bf88ff9e
	v_add_co_u32 v65, s2, v89, s10                             // 000000002524: d7000241 02001559
	s_wait_alu depctr_va_sdst(0)                               // 00000000252c: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s3, v90, s2                 // 000000002530: d5207c42 000ab403
	v_add_co_u32 v69, s2, v91, s10                             // 000000002538: d7000245 0200155b
	s_wait_alu depctr_va_sdst(0)                               // 000000002540: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s3, v92, s2                 // 000000002544: d5207c46 000ab803
	v_add_co_u32 v73, s2, v93, s10                             // 00000000254c: d7000249 0200155d
	s_wait_alu depctr_va_sdst(0)                               // 000000002554: bf88f19f
	v_add_co_ci_u32_e64 v74, null, s3, v94, s2                 // 000000002558: d5207c4a 000abc03
	v_add_co_u32 v77, s2, v97, s10                             // 000000002560: d700024d 02001561
	s_wait_alu depctr_va_sdst(0)                               // 000000002568: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s3, v98, s2                 // 00000000256c: d5207c4e 000ac403
	v_add_co_u32 v81, s2, v95, s10                             // 000000002574: d7000251 0200155f
	s_wait_alu depctr_va_sdst(0)                               // 00000000257c: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s3, v96, s2                 // 000000002580: d5207c52 000ac003
	s_clause 0x3                                               // 000000002588: bf850003
	global_load_b128 v[65:68], v[65:66], off                   // 00000000258c: ee05c07c 00000041 00000041
	global_load_b128 v[69:72], v[69:70], off                   // 000000002598: ee05c07c 00000045 00000045
	global_load_b128 v[73:76], v[73:74], off                   // 0000000025a4: ee05c07c 00000049 00000049
	global_load_b128 v[77:80], v[77:78], off                   // 0000000025b0: ee05c07c 0000004d 0000004d
	global_load_b128 v[81:84], v[81:82], off                   // 0000000025bc: ee05c07c 00000051 00000051
	s_and_saveexec_b32 s2, vcc_lo                              // 0000000025c8: be82206a
	s_cbranch_execz 65459                                      // 0000000025cc: bfa5ffb3 <tessera_rocm_folded_matmul_a629485a5b78541c+0x99c>
	v_add_nc_u32_e32 v120, 0x5000, v102                        // 0000000025d0: 4af0ccff 00005000
	v_add_nc_u32_e32 v138, 0x5000, v103                        // 0000000025d8: 4b14ceff 00005000
	ds_load_2addr_b64 v[107:110], v101 offset1:2               // 0000000025e0: d9dc0200 6b000065
	ds_load_2addr_b64 v[112:115], v120 offset1:2               // 0000000025e8: d9dc0200 70000078
	ds_load_2addr_b64 v[116:119], v138 offset1:2               // 0000000025f0: d9dc0200 7400008a
	ds_load_2addr_b64 v[122:125], v104 offset1:2               // 0000000025f8: d9dc0200 7a000068
	ds_load_2addr_b64 v[126:129], v105 offset1:2               // 000000002600: d9dc0200 7e000069
	ds_load_2addr_b64 v[130:133], v106 offset1:2               // 000000002608: d9dc0200 8200006a
	ds_load_2addr_b64 v[134:137], v101 offset0:4 offset1:6     // 000000002610: d9dc0604 86000065
	ds_load_2addr_b64 v[143:146], v104 offset0:4 offset1:6     // 000000002618: d9dc0604 8f000068
	ds_load_2addr_b64 v[147:150], v120 offset0:4 offset1:6     // 000000002620: d9dc0604 93000078
	ds_load_2addr_b64 v[151:154], v138 offset0:4 offset1:6     // 000000002628: d9dc0604 9700008a
	ds_load_2addr_b64 v[155:158], v105 offset0:4 offset1:6     // 000000002630: d9dc0604 9b000069
	ds_load_2addr_b64 v[159:162], v106 offset0:4 offset1:6     // 000000002638: d9dc0604 9f00006a
	s_wait_dscnt 0xa                                           // 000000002640: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[107:108], v[112:113], v[57:64]// 000000002644: cc464039 1ce6e16b
	s_wait_dscnt 0x9                                           // 00000000264c: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[107:108], v[116:117], v[25:32]// 000000002650: cc464019 1c66e96b
	s_wait_dscnt 0x8                                           // 000000002658: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[122:123], v[112:113], v[49:56]// 00000000265c: cc464031 1cc6e17a
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[122:123], v[116:117], v[17:24]// 000000002664: cc464011 1c46e97a
	s_wait_dscnt 0x7                                           // 00000000266c: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[126:127], v[112:113], v[41:48]// 000000002670: cc464029 1ca6e17e
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[126:127], v[116:117], v[9:16]// 000000002678: cc464009 1c26e97e
	s_wait_dscnt 0x6                                           // 000000002680: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[130:131], v[112:113], v[33:40]// 000000002684: cc464021 1c86e182
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[130:131], v[116:117], v[1:8]// 00000000268c: cc464001 1c06e982
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[109:110], v[114:115], v[57:64]// 000000002694: cc464039 1ce6e56d
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[109:110], v[118:119], v[25:32]// 00000000269c: cc464019 1c66ed6d
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[124:125], v[114:115], v[49:56]// 0000000026a4: cc464031 1cc6e57c
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[124:125], v[118:119], v[17:24]// 0000000026ac: cc464011 1c46ed7c
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[128:129], v[114:115], v[41:48]// 0000000026b4: cc464029 1ca6e580
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[128:129], v[118:119], v[9:16]// 0000000026bc: cc464009 1c26ed80
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[132:133], v[114:115], v[33:40]// 0000000026c4: cc464021 1c86e584
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[132:133], v[118:119], v[1:8]// 0000000026cc: cc464001 1c06ed84
	s_wait_dscnt 0x3                                           // 0000000026d4: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[134:135], v[147:148], v[57:64]// 0000000026d8: cc464039 1ce72786
	s_wait_dscnt 0x2                                           // 0000000026e0: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[134:135], v[151:152], v[25:32]// 0000000026e4: cc464019 1c672f86
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[143:144], v[147:148], v[49:56]// 0000000026ec: cc464031 1cc7278f
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[143:144], v[151:152], v[17:24]// 0000000026f4: cc464011 1c472f8f
	s_wait_dscnt 0x1                                           // 0000000026fc: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[155:156], v[147:148], v[41:48]// 000000002700: cc464029 1ca7279b
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[155:156], v[151:152], v[9:16]// 000000002708: cc464009 1c272f9b
	s_wait_dscnt 0x0                                           // 000000002710: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[159:160], v[147:148], v[33:40]// 000000002714: cc464021 1c87279f
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[159:160], v[151:152], v[1:8]// 00000000271c: cc464001 1c072f9f
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[136:137], v[149:150], v[57:64]// 000000002724: cc464039 1ce72b88
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[136:137], v[153:154], v[25:32]// 00000000272c: cc464019 1c673388
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[145:146], v[149:150], v[49:56]// 000000002734: cc464031 1cc72b91
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[145:146], v[153:154], v[17:24]// 00000000273c: cc464011 1c473391
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[157:158], v[149:150], v[41:48]// 000000002744: cc464029 1ca72b9d
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[157:158], v[153:154], v[9:16]// 00000000274c: cc464009 1c27339d
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[161:162], v[149:150], v[33:40]// 000000002754: cc464021 1c872ba1
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[161:162], v[153:154], v[1:8]// 00000000275c: cc464001 1c0733a1
	s_branch 65357                                             // 000000002764: bfa0ff4d <tessera_rocm_folded_matmul_a629485a5b78541c+0x99c>
	s_load_b64 s[46:47], s[0:1], 0x58                          // 000000002768: f4002b80 f8000058
	v_or_b32_e32 v87, v87, v86                                 // 000000002770: 38aead57
	v_or3_b32 v74, s39, 0, 0                                   // 000000002774: d658004a 02010027
	v_or3_b32 v73, s38, v141, v142                             // 00000000277c: d6580049 063b1a26
	v_mov_b32_e32 v82, v88                                     // 000000002784: 7ea40358
	s_mov_b32 s3, 0                                            // 000000002788: be830080
	v_or_b32_e32 v81, 7, v87                                   // 00000000278c: 38a2ae87
	v_or_b32_e32 v97, 1, v87                                   // 000000002790: 38c2ae81
	v_cmp_gt_u64_e64 s27, s[42:43], v[73:74]                   // 000000002794: d45c001b 0202922a
	v_or_b32_e32 v95, 2, v87                                   // 00000000279c: 38beae82
	v_or_b32_e32 v93, 3, v87                                   // 0000000027a0: 38baae83
	v_cmp_gt_u64_e32 vcc_lo, s[40:41], v[81:82]                // 0000000027a4: 7cb8a228
	v_or_b32_e32 v89, 4, v87                                   // 0000000027a8: 38b2ae84
	v_or_b32_e32 v91, 5, v87                                   // 0000000027ac: 38b6ae85
	v_or_b32_e32 v83, 6, v87                                   // 0000000027b0: 38a6ae86
	s_and_b32 s4, s27, vcc_lo                                  // 0000000027b4: 8b046a1b
	s_wait_kmcnt 0x0                                           // 0000000027b8: bfc70000
	s_and_b32 s2, s46, 15                                      // 0000000027bc: 8b028f2e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027c0: bf88ff9e
	s_cmp_eq_u64 s[2:3], 0                                     // 0000000027c4: bf108002
	s_cselect_b32 s52, -1, 0                                   // 0000000027c8: 983480c1
	s_delay_alu instid0(salu_cycle_1)                          // 0000000027cc: bf870009
	s_and_b32 s2, s52, s4                                      // 0000000027d0: 8b020434
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027d4: bf88ff9e
	s_xor_b32 s2, s2, -1                                       // 0000000027d8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027dc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 0000000027e0: be832002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027e4: bf88ff9e
	s_xor_b32 s4, exec_lo, s3                                  // 0000000027e8: 8d04037e
	s_cbranch_execz 133                                        // 0000000027ec: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0xf04>
	v_mov_b32_e32 v98, v88                                     // 0000000027f0: 7ec40358
	v_cmp_gt_i64_e64 s2, s[40:41], v[87:88]                    // 0000000027f4: d4540002 0202ae28
	v_mov_b32_e32 v96, v88                                     // 0000000027fc: 7ec00358
	v_mov_b32_e32 v94, v88                                     // 000000002800: 7ebc0358
	v_mov_b32_e32 v90, v88                                     // 000000002804: 7eb40358
	v_cmp_gt_i64_e64 s3, s[40:41], v[97:98]                    // 000000002808: d4540003 0202c228
	v_mov_b32_e32 v92, v88                                     // 000000002810: 7eb80358
	s_wait_alu depctr_va_sdst(0)                               // 000000002814: bf88f19f
	v_cndmask_b32_e64 v66, 0, v88, s2                          // 000000002818: d5010042 000ab080
	v_cndmask_b32_e64 v65, 0, v87, s2                          // 000000002820: d5010041 000aae80
	v_cmp_gt_i64_e64 s2, s[40:41], v[95:96]                    // 000000002828: d4540002 0202be28
	v_mov_b32_e32 v84, v88                                     // 000000002830: 7ea80358
	v_cndmask_b32_e64 v67, 0, v97, s3                          // 000000002834: d5010043 000ec280
	v_cndmask_b32_e64 v68, 0, v88, s3                          // 00000000283c: d5010044 000eb080
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]                    // 000000002844: 3e828282
	v_cmp_gt_i64_e64 s3, s[40:41], v[93:94]                    // 000000002848: d4540003 0202ba28
	s_wait_alu depctr_va_sdst(0)                               // 000000002850: bf88f19f
	v_cndmask_b32_e64 v69, 0, v95, s2                          // 000000002854: d5010045 000abe80
	v_cndmask_b32_e64 v70, 0, v88, s2                          // 00000000285c: d5010046 000ab080
	v_lshlrev_b64_e32 v[67:68], 2, v[67:68]                    // 000000002864: 3e868682
	v_add_co_u32 v65, s2, s46, v65                             // 000000002868: d7000241 0202822e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000002870: bf870233
	v_lshlrev_b64_e32 v[69:70], 2, v[69:70]                    // 000000002874: 3e8a8a82
	s_wait_alu depctr_va_sdst(0)                               // 000000002878: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s47, v66, s2                // 00000000287c: d5207c42 000a842f
	v_add_co_u32 v67, s2, s46, v67                             // 000000002884: d7000243 0202862e
	v_cndmask_b32_e64 v71, 0, v93, s3                          // 00000000288c: d5010047 000eba80
	v_cndmask_b32_e64 v72, 0, v88, s3                          // 000000002894: d5010048 000eb080
	s_wait_alu depctr_va_sdst(0)                               // 00000000289c: bf88f19f
	v_add_co_ci_u32_e64 v68, null, s47, v68, s2                // 0000000028a0: d5207c44 000a882f
	v_cmp_gt_i64_e64 s2, s[40:41], v[89:90]                    // 0000000028a8: d4540002 0202b228
	v_add_co_u32 v75, s3, s46, v69                             // 0000000028b0: d700034b 02028a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000028b8: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s47, v70, s3                // 0000000028bc: d5207c4c 000e8c2f
	v_cmp_gt_i64_e64 s3, s[40:41], v[91:92]                    // 0000000028c4: d4540003 0202b628
	v_lshlrev_b64_e32 v[69:70], 2, v[71:72]                    // 0000000028cc: 3e8a8e82
	v_cndmask_b32_e64 v71, 0, v89, s2                          // 0000000028d0: d5010047 000ab280
	v_cndmask_b32_e64 v72, 0, v88, s2                          // 0000000028d8: d5010048 000ab080
	v_cmp_gt_i64_e64 s2, s[40:41], v[83:84]                    // 0000000028e0: d4540002 0202a628
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e8: bf88f19f
	v_cndmask_b32_e64 v77, 0, v91, s3                          // 0000000028ec: d501004d 000eb680
	v_cndmask_b32_e64 v78, 0, v88, s3                          // 0000000028f4: d501004e 000eb080
	v_add_co_u32 v79, s3, s46, v69                             // 0000000028fc: d700034f 02028a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002904: bf88f19f
	v_add_co_ci_u32_e64 v80, null, s47, v70, s3                // 000000002908: d5207c50 000e8c2f
	v_lshlrev_b64_e32 v[69:70], 2, v[71:72]                    // 000000002910: 3e8a8e82
	v_lshlrev_b64_e32 v[71:72], 2, v[77:78]                    // 000000002914: 3e8e9a82
	v_cndmask_b32_e64 v77, 0, v83, s2                          // 000000002918: d501004d 000aa680
	v_cndmask_b32_e64 v78, 0, v88, s2                          // 000000002920: d501004e 000ab080
	v_cmp_gt_i64_e64 s2, s[40:41], v[81:82]                    // 000000002928: d4540002 0202a228
	v_add_co_u32 v100, s3, s46, v69                            // 000000002930: d7000364 02028a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002938: bf88f19f
	v_add_co_ci_u32_e64 v101, null, s47, v70, s3               // 00000000293c: d5207c65 000e8c2f
	v_lshlrev_b64_e32 v[69:70], 2, v[77:78]                    // 000000002944: 3e8a9a82
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000002948: bf8701d4
	v_cndmask_b32_e64 v78, 0, v82, s2                          // 00000000294c: d501004e 000aa480
	v_cndmask_b32_e64 v77, 0, v81, s2                          // 000000002954: d501004d 000aa280
	v_add_co_u32 v102, s2, s46, v71                            // 00000000295c: d7000266 02028e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002964: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s47, v72, s2               // 000000002968: d5207c67 000a902f
	v_lshlrev_b64_e32 v[71:72], 2, v[77:78]                    // 000000002970: 3e8e9a82
	v_add_co_u32 v77, s2, s46, v69                             // 000000002974: d700024d 02028a2e
	s_wait_alu depctr_va_sdst(0)                               // 00000000297c: bf88f19f
	v_add_co_ci_u32_e64 v78, null, s47, v70, s2                // 000000002980: d5207c4e 000a8c2f
	s_delay_alu instid0(valu_dep_3)                            // 000000002988: bf870003
	v_add_co_u32 v104, s2, s46, v71                            // 00000000298c: d7000268 02028e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000002994: bf88f19f
	v_add_co_ci_u32_e64 v105, null, s47, v72, s2               // 000000002998: d5207c69 000a902f
	s_clause 0x7                                               // 0000000029a0: bf850007
	global_load_b32 v69, v[65:66], off                         // 0000000029a4: ee05007c 00000045 00000041
	global_load_b32 v70, v[67:68], off                         // 0000000029b0: ee05007c 00000046 00000043
	global_load_b32 v71, v[75:76], off                         // 0000000029bc: ee05007c 00000047 0000004b
	global_load_b32 v72, v[79:80], off                         // 0000000029c8: ee05007c 00000048 0000004f
	global_load_b32 v65, v[100:101], off                       // 0000000029d4: ee05007c 00000041 00000064
	global_load_b32 v66, v[102:103], off                       // 0000000029e0: ee05007c 00000042 00000066
	global_load_b32 v67, v[77:78], off                         // 0000000029ec: ee05007c 00000043 0000004d
	global_load_b32 v68, v[104:105], off                       // 0000000029f8: ee05007c 00000044 00000068
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a04: bf88ff9e
	s_or_saveexec_b32 s3, s4                                   // 000000002a08: be832204
	v_lshlrev_b64_e32 v[135:136], 2, v[87:88]                  // 000000002a0c: 3f0eae82
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a10: bf88ff9e
	s_xor_b32 exec_lo, exec_lo, s3                             // 000000002a14: 8d7e037e
	s_cbranch_execz 15                                         // 000000002a18: bfa5000f <tessera_rocm_folded_matmul_a629485a5b78541c+0xf58>
	s_wait_loadcnt 0x3                                         // 000000002a1c: bfc00003
	s_delay_alu instid0(valu_dep_1)                            // 000000002a20: bf870001
	v_add_co_u32 v65, s2, s46, v135                            // 000000002a24: d7000241 02030e2e
	s_wait_loadcnt 0x2                                         // 000000002a2c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000002a30: bf88f19f
	v_add_co_ci_u32_e64 v66, null, s47, v136, s2               // 000000002a34: d5207c42 000b102f
	global_load_b128 v[69:72], v[65:66], off                   // 000000002a3c: ee05c07c 00000045 00000041
	s_wait_loadcnt 0x1                                         // 000000002a48: bfc00001
	global_load_b128 v[65:68], v[65:66], off offset:16         // 000000002a4c: ee05c07c 00000041 00001041
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002a58: 8c7e037e
	s_load_b64 s[50:51], s[0:1], 0x80                          // 000000002a5c: f4002c80 f8000080
	v_cmp_gt_i64_e64 s36, s[42:43], v[73:74]                   // 000000002a64: d4540024 0202922a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_2)// 000000002a6c: bf870131
	v_cndmask_b32_e64 v76, 0, v73, s36                         // 000000002a70: d501004c 00929280
	v_cndmask_b32_e64 v75, 0, v74, s36                         // 000000002a78: d501004b 00929480
	s_wait_kmcnt 0x0                                           // 000000002a80: bfc70000
	v_add_co_u32 v139, s2, s50, v76                            // 000000002a84: d700028b 02029832
	s_wait_alu depctr_va_sdst(0)                               // 000000002a8c: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002a90: bf8700c2
	v_add_co_ci_u32_e64 v140, null, s51, v75, s2               // 000000002a94: d5207c8c 000a9633
	global_load_u8 v75, v[139:140], off                        // 000000002a9c: ee04007c 0000004b 0000008b
	s_wait_loadcnt 0x0                                         // 000000002aa8: bfc00000
	v_lshlrev_b32_e32 v79, 23, v75                             // 000000002aac: 309e9697
	v_mul_f32_e32 v75, v69, v79                                // 000000002ab0: 10969f45
	s_delay_alu instid0(valu_dep_1)                            // 000000002ab4: bf870001
	v_cmp_class_f32_e64 s2, v75, 0x198                         // 000000002ab8: d47e0002 0201ff4b 00000198
	v_mul_f32_e32 v78, v57, v75                                // 000000002ac4: 109c9739
	s_xor_b32 s2, s2, -1                                       // 000000002ac8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002acc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002ad0: be832002
	s_cbranch_execnz 4412                                      // 000000002ad4: bfa6113c <tessera_rocm_folded_matmul_a629485a5b78541c+0x54c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ad8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002adc: 8c7e037e
	v_mul_f32_e32 v57, v70, v79                                // 000000002ae0: 10729f46
	s_delay_alu instid0(valu_dep_1)                            // 000000002ae4: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002ae8: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v77, v58, v57                                // 000000002af4: 109a733a
	s_xor_b32 s2, s2, -1                                       // 000000002af8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002afc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b00: be832002
	s_cbranch_execnz 4418                                      // 000000002b04: bfa61142 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5510>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b0c: 8c7e037e
	v_mul_f32_e32 v57, v71, v79                                // 000000002b10: 10729f47
	s_delay_alu instid0(valu_dep_1)                            // 000000002b14: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b18: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v69, v59, v57                                // 000000002b24: 108a733b
	s_xor_b32 s2, s2, -1                                       // 000000002b28: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b2c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b30: be832002
	s_cbranch_execnz 4424                                      // 000000002b34: bfa61148 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5558>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b38: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b3c: 8c7e037e
	v_mul_f32_e32 v57, v72, v79                                // 000000002b40: 10729f48
	s_delay_alu instid0(valu_dep_1)                            // 000000002b44: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b48: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v71, v60, v57                                // 000000002b54: 108e733c
	s_xor_b32 s2, s2, -1                                       // 000000002b58: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b5c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b60: be832002
	s_cbranch_execnz 4430                                      // 000000002b64: bfa6114e <tessera_rocm_folded_matmul_a629485a5b78541c+0x55a0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b6c: 8c7e037e
	v_mul_f32_e32 v57, v65, v79                                // 000000002b70: 10729f41
	s_delay_alu instid0(valu_dep_1)                            // 000000002b74: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002b78: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v76, v61, v57                                // 000000002b84: 1098733d
	s_xor_b32 s2, s2, -1                                       // 000000002b88: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b8c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002b90: be832002
	s_cbranch_execnz 4436                                      // 000000002b94: bfa61154 <tessera_rocm_folded_matmul_a629485a5b78541c+0x55e8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002b9c: 8c7e037e
	v_mul_f32_e32 v57, v66, v79                                // 000000002ba0: 10729f42
	s_delay_alu instid0(valu_dep_1)                            // 000000002ba4: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002ba8: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v75, v62, v57                                // 000000002bb4: 1096733e
	s_xor_b32 s2, s2, -1                                       // 000000002bb8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bbc: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002bc0: be832002
	s_cbranch_execnz 4442                                      // 000000002bc4: bfa6115a <tessera_rocm_folded_matmul_a629485a5b78541c+0x5630>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002bcc: 8c7e037e
	v_mul_f32_e32 v57, v67, v79                                // 000000002bd0: 10729f43
	s_delay_alu instid0(valu_dep_1)                            // 000000002bd4: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002bd8: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v62, v63, v57                                // 000000002be4: 107c733f
	s_xor_b32 s2, s2, -1                                       // 000000002be8: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bec: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002bf0: be832002
	s_cbranch_execnz 4448                                      // 000000002bf4: bfa61160 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5678>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002bf8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002bfc: 8c7e037e
	v_mul_f32_e32 v57, v68, v79                                // 000000002c00: 10729f44
	s_delay_alu instid0(valu_dep_1)                            // 000000002c04: bf870001
	v_cmp_class_f32_e64 s2, v57, 0x198                         // 000000002c08: d47e0002 0201ff39 00000198
	v_mul_f32_e32 v61, v64, v57                                // 000000002c14: 107a7340
	s_xor_b32 s2, s2, -1                                       // 000000002c18: 8d02c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c1c: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002c20: be832002
	s_cbranch_execnz 4454                                      // 000000002c24: bfa61166 <tessera_rocm_folded_matmul_a629485a5b78541c+0x56c0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c28: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002c2c: 8c7e037e
	s_load_b64 s[48:49], s[0:1], 0xa8                          // 000000002c30: f4002c00 f80000a8
	v_mul_lo_u32 v63, v88, s42                                 // 000000002c38: d72c003f 02005558
	v_mul_lo_u32 v64, v87, s43                                 // 000000002c40: d72c0040 02005757
	v_mad_co_u64_u32 v[59:60], null, v87, s42, 0               // 000000002c48: d6fe7c3b 02005557
	v_sub_co_u32 v57, s0, s40, v87                             // 000000002c50: d7010039 0202ae28
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_4)// 000000002c58: bf870221
	v_sub_co_ci_u32_e64 v58, null, s41, v88, s0                // 000000002c5c: d5217c3a 0002b029
	v_lshlrev_b64_e32 v[137:138], 1, v[73:74]                  // 000000002c64: 3f129281
	v_add3_u32 v60, v60, v64, v63                              // 000000002c68: d655003c 04fe813c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002c70: bf870113
	v_cmp_lt_i64_e64 s0, 0, v[57:58]                           // 000000002c74: d4510000 02027280
	v_lshlrev_b64_e32 v[79:80], 1, v[59:60]                    // 000000002c7c: 3e9e7681
	s_and_b32 s1, s36, s0                                      // 000000002c80: 8b010024
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c84: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 000000002c88: be822001
	s_cbranch_execz 28                                         // 000000002c8c: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1200>
	v_bfe_u32 v59, v78, 16, 1                                  // 000000002c90: d610003b 0205214e
	s_wait_kmcnt 0x0                                           // 000000002c98: bfc70000
	v_add_co_u32 v60, s1, s48, v79                             // 000000002c9c: d700013c 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002ca4: bf88f19f
	v_add_co_ci_u32_e64 v63, null, s49, v80, s1                // 000000002ca8: d5207c3f 0006a031
	v_add3_u32 v64, v59, v78, 0x7fff                           // 000000002cb0: d6550040 03fe9d3b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002cbc: bf870003
	v_add_co_u32 v59, s1, v60, v137                            // 000000002cc0: d700013b 0203133c
	v_or_b32_e32 v65, 0x400000, v78                            // 000000002cc8: 38829cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002cd0: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v63, v138, s1               // 000000002cd4: d5207c3c 0007153f
	v_cmp_u_f32_e64 s1, v78, v78                               // 000000002cdc: d4180001 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ce4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002ce8: bf870001
	v_cndmask_b32_e64 v63, v64, v65, s1                        // 000000002cec: d501003f 00068340
	global_store_d16_hi_b16 v[59:60], v63, off                 // 000000002cf4: ee09407c 1f800000 0000003b
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d00: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002d04: 8c7e027e
	v_add_co_u32 v59, s1, v73, s42                             // 000000002d08: d700013b 02005549
	s_wait_alu depctr_va_sdst(0)                               // 000000002d10: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v74, s1                // 000000002d14: d5207c3c 0006942b
	v_cmp_lt_i64_e64 s1, 1, v[57:58]                           // 000000002d1c: d4510001 02027281
	s_delay_alu instid0(valu_dep_2)                            // 000000002d24: bf870002
	v_lshlrev_b64_e32 v[65:66], 1, v[59:60]                    // 000000002d28: 3e827681
	s_and_b32 s2, s36, s1                                      // 000000002d2c: 8b020124
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d30: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000002d34: be832002
	s_cbranch_execz 28                                         // 000000002d38: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x12ac>
	v_bfe_u32 v63, v77, 16, 1                                  // 000000002d3c: d610003f 0205214d
	s_wait_kmcnt 0x0                                           // 000000002d44: bfc70000
	v_add_co_u32 v64, s2, s48, v79                             // 000000002d48: d7000240 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002d50: bf88f19f
	v_add_co_ci_u32_e64 v67, null, s49, v80, s2                // 000000002d54: d5207c43 000aa031
	v_add3_u32 v68, v63, v77, 0x7fff                           // 000000002d5c: d6550044 03fe9b3f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d68: bf870003
	v_add_co_u32 v63, s2, v64, v65                             // 000000002d6c: d700023f 02028340
	v_or_b32_e32 v70, 0x400000, v77                            // 000000002d74: 388c9aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002d7c: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v67, v66, s2                // 000000002d80: d5207c40 000a8543
	v_cmp_u_f32_e64 s2, v77, v77                               // 000000002d88: d4180002 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000002d90: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002d94: bf870001
	v_cndmask_b32_e64 v67, v68, v70, s2                        // 000000002d98: d5010043 000a8d44
	global_store_d16_hi_b16 v[63:64], v67, off                 // 000000002da0: ee09407c 21800000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000002db0: 8c7e037e
	v_add_co_u32 v59, s2, v59, s42                             // 000000002db4: d700023b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 000000002dbc: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s2                // 000000002dc0: d5207c3c 000a782b
	v_cmp_lt_i64_e64 s2, 2, v[57:58]                           // 000000002dc8: d4510002 02027282
	s_delay_alu instid0(valu_dep_2)                            // 000000002dd0: bf870002
	v_lshlrev_b64_e32 v[67:68], 1, v[59:60]                    // 000000002dd4: 3e867681
	s_and_b32 s3, s36, s2                                      // 000000002dd8: 8b030224
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ddc: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000002de0: be842003
	s_cbranch_execz 28                                         // 000000002de4: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1358>
	v_bfe_u32 v63, v69, 16, 1                                  // 000000002de8: d610003f 02052145
	s_wait_kmcnt 0x0                                           // 000000002df0: bfc70000
	v_add_co_u32 v64, s3, s48, v79                             // 000000002df4: d7000340 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002dfc: bf88f19f
	v_add_co_ci_u32_e64 v70, null, s49, v80, s3                // 000000002e00: d5207c46 000ea031
	v_add3_u32 v72, v63, v69, 0x7fff                           // 000000002e08: d6550048 03fe8b3f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e14: bf870003
	v_add_co_u32 v63, s3, v64, v67                             // 000000002e18: d700033f 02028740
	v_or_b32_e32 v73, 0x400000, v69                            // 000000002e20: 38928aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002e28: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v70, v68, s3                // 000000002e2c: d5207c40 000e8946
	v_cmp_u_f32_e64 s3, v69, v69                               // 000000002e34: d4180003 02028b45
	s_wait_alu depctr_va_sdst(0)                               // 000000002e3c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002e40: bf870001
	v_cndmask_b32_e64 v69, v72, v73, s3                        // 000000002e44: d5010045 000e9348
	global_store_d16_hi_b16 v[63:64], v69, off                 // 000000002e4c: ee09407c 22800000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e58: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000002e5c: 8c7e047e
	v_add_co_u32 v59, s3, v59, s42                             // 000000002e60: d700033b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 000000002e68: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s3                // 000000002e6c: d5207c3c 000e782b
	v_cmp_lt_i64_e64 s3, 3, v[57:58]                           // 000000002e74: d4510003 02027283
	s_delay_alu instid0(valu_dep_2)                            // 000000002e7c: bf870002
	v_lshlrev_b64_e32 v[69:70], 1, v[59:60]                    // 000000002e80: 3e8a7681
	s_and_b32 s4, s36, s3                                      // 000000002e84: 8b040324
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e88: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000002e8c: be852004
	s_cbranch_execz 28                                         // 000000002e90: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1404>
	v_bfe_u32 v63, v71, 16, 1                                  // 000000002e94: d610003f 02052147
	s_wait_kmcnt 0x0                                           // 000000002e9c: bfc70000
	v_add_co_u32 v64, s4, s48, v79                             // 000000002ea0: d7000440 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002ea8: bf88f19f
	v_add_co_ci_u32_e64 v72, null, s49, v80, s4                // 000000002eac: d5207c48 0012a031
	v_add3_u32 v73, v63, v71, 0x7fff                           // 000000002eb4: d6550049 03fe8f3f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002ec0: bf870003
	v_add_co_u32 v63, s4, v64, v69                             // 000000002ec4: d700043f 02028b40
	v_or_b32_e32 v74, 0x400000, v71                            // 000000002ecc: 38948eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed4: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v72, v70, s4                // 000000002ed8: d5207c40 00128d48
	v_cmp_u_f32_e64 s4, v71, v71                               // 000000002ee0: d4180004 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000002ee8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002eec: bf870001
	v_cndmask_b32_e64 v71, v73, v74, s4                        // 000000002ef0: d5010047 00129549
	global_store_d16_hi_b16 v[63:64], v71, off                 // 000000002ef8: ee09407c 23800000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000002f08: 8c7e057e
	v_add_co_u32 v59, s4, v59, s42                             // 000000002f0c: d700043b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 000000002f14: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s4                // 000000002f18: d5207c3c 0012782b
	v_cmp_lt_i64_e64 s4, 4, v[57:58]                           // 000000002f20: d4510004 02027284
	s_delay_alu instid0(valu_dep_2)                            // 000000002f28: bf870002
	v_lshlrev_b64_e32 v[71:72], 1, v[59:60]                    // 000000002f2c: 3e8e7681
	s_and_b32 s5, s36, s4                                      // 000000002f30: 8b050424
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f34: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 000000002f38: be862005
	s_cbranch_execz 28                                         // 000000002f3c: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x14b0>
	v_bfe_u32 v63, v76, 16, 1                                  // 000000002f40: d610003f 0205214c
	s_wait_kmcnt 0x0                                           // 000000002f48: bfc70000
	v_add_co_u32 v64, s5, s48, v79                             // 000000002f4c: d7000540 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000002f54: bf88f19f
	v_add_co_ci_u32_e64 v73, null, s49, v80, s5                // 000000002f58: d5207c49 0016a031
	v_add3_u32 v74, v63, v76, 0x7fff                           // 000000002f60: d655004a 03fe993f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f6c: bf870003
	v_add_co_u32 v63, s5, v64, v71                             // 000000002f70: d700053f 02028f40
	v_or_b32_e32 v77, 0x400000, v76                            // 000000002f78: 389a98ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000002f80: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v73, v72, s5                // 000000002f84: d5207c40 00169149
	v_cmp_u_f32_e64 s5, v76, v76                               // 000000002f8c: d4180005 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f94: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000002f98: bf870001
	v_cndmask_b32_e64 v73, v74, v77, s5                        // 000000002f9c: d5010049 00169b4a
	global_store_d16_hi_b16 v[63:64], v73, off                 // 000000002fa4: ee09407c 24800000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002fb4: 8c7e067e
	v_add_co_u32 v59, s5, v59, s42                             // 000000002fb8: d700053b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 000000002fc0: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s5                // 000000002fc4: d5207c3c 0016782b
	v_cmp_lt_i64_e64 s5, 5, v[57:58]                           // 000000002fcc: d4510005 02027285
	s_delay_alu instid0(valu_dep_2)                            // 000000002fd4: bf870002
	v_lshlrev_b64_e32 v[73:74], 1, v[59:60]                    // 000000002fd8: 3e927681
	s_and_b32 s6, s36, s5                                      // 000000002fdc: 8b060524
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe0: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000002fe4: be872006
	s_cbranch_execz 28                                         // 000000002fe8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x155c>
	v_bfe_u32 v63, v75, 16, 1                                  // 000000002fec: d610003f 0205214b
	s_wait_kmcnt 0x0                                           // 000000002ff4: bfc70000
	v_add_co_u32 v64, s6, s48, v79                             // 000000002ff8: d7000640 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000003000: bf88f19f
	v_add_co_ci_u32_e64 v76, null, s49, v80, s6                // 000000003004: d5207c4c 001aa031
	v_add3_u32 v77, v63, v75, 0x7fff                           // 00000000300c: d655004d 03fe973f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003018: bf870003
	v_add_co_u32 v63, s6, v64, v73                             // 00000000301c: d700063f 02029340
	v_or_b32_e32 v78, 0x400000, v75                            // 000000003024: 389c96ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000302c: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v76, v74, s6                // 000000003030: d5207c40 001a954c
	v_cmp_u_f32_e64 s6, v75, v75                               // 000000003038: d4180006 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 000000003040: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003044: bf870001
	v_cndmask_b32_e64 v75, v77, v78, s6                        // 000000003048: d501004b 001a9d4d
	global_store_d16_hi_b16 v[63:64], v75, off                 // 000000003050: ee09407c 25800000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000305c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000003060: 8c7e077e
	v_add_co_u32 v59, s6, v59, s42                             // 000000003064: d700063b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 00000000306c: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s6                // 000000003070: d5207c3c 001a782b
	v_cmp_lt_i64_e64 s6, 6, v[57:58]                           // 000000003078: d4510006 02027286
	s_delay_alu instid0(valu_dep_2)                            // 000000003080: bf870002
	v_lshlrev_b64_e32 v[75:76], 1, v[59:60]                    // 000000003084: 3e967681
	s_and_b32 s7, s36, s6                                      // 000000003088: 8b070624
	s_wait_alu depctr_sa_sdst(0)                               // 00000000308c: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 000000003090: be882007
	s_cbranch_execz 28                                         // 000000003094: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1608>
	v_bfe_u32 v63, v62, 16, 1                                  // 000000003098: d610003f 0205213e
	s_wait_kmcnt 0x0                                           // 0000000030a0: bfc70000
	v_add_co_u32 v64, s7, s48, v79                             // 0000000030a4: d7000740 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 0000000030ac: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s49, v80, s7                // 0000000030b0: d5207c4d 001ea031
	v_add3_u32 v78, v63, v62, 0x7fff                           // 0000000030b8: d655004e 03fe7d3f 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000030c4: bf870003
	v_add_co_u32 v63, s7, v64, v75                             // 0000000030c8: d700073f 02029740
	v_or_b32_e32 v84, 0x400000, v62                            // 0000000030d0: 38a87cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d8: bf88f19f
	v_add_co_ci_u32_e64 v64, null, v77, v76, s7                // 0000000030dc: d5207c40 001e994d
	v_cmp_u_f32_e64 s7, v62, v62                               // 0000000030e4: d4180007 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000030f0: bf870001
	v_cndmask_b32_e64 v62, v78, v84, s7                        // 0000000030f4: d501003e 001ea94e
	global_store_d16_hi_b16 v[63:64], v62, off                 // 0000000030fc: ee09407c 1f000000 0000003f
	s_wait_alu depctr_sa_sdst(0)                               // 000000003108: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 00000000310c: 8c7e087e
	v_add_co_u32 v59, s7, v59, s42                             // 000000003110: d700073b 0200553b
	s_wait_alu depctr_va_sdst(0)                               // 000000003118: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s43, v60, s7                // 00000000311c: d5207c3c 001e782b
	v_cmp_lt_i64_e64 s7, 7, v[57:58]                           // 000000003124: d4510007 02027287
	s_delay_alu instid0(valu_dep_2)                            // 00000000312c: bf870002
	v_lshlrev_b64_e32 v[77:78], 1, v[59:60]                    // 000000003130: 3e9a7681
	s_and_b32 s8, s36, s7                                      // 000000003134: 8b080724
	s_wait_alu depctr_sa_sdst(0)                               // 000000003138: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000313c: be892008
	s_cbranch_execz 28                                         // 000000003140: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x16b4>
	v_bfe_u32 v57, v61, 16, 1                                  // 000000003144: d6100039 0205213d
	s_wait_kmcnt 0x0                                           // 00000000314c: bfc70000
	v_add_co_u32 v58, s8, s48, v79                             // 000000003150: d700083a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000003158: bf88f19f
	v_add_co_ci_u32_e64 v59, null, s49, v80, s8                // 00000000315c: d5207c3b 0022a031
	v_add3_u32 v60, v57, v61, 0x7fff                           // 000000003164: d655003c 03fe7b39 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003170: bf870003
	v_add_co_u32 v57, s8, v58, v77                             // 000000003174: d7000839 02029b3a
	v_or_b32_e32 v62, 0x400000, v61                            // 00000000317c: 387c7aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003184: bf88f19f
	v_add_co_ci_u32_e64 v58, null, v59, v78, s8                // 000000003188: d5207c3a 00229d3b
	v_cmp_u_f32_e64 s8, v61, v61                               // 000000003190: d4180008 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000003198: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000319c: bf870001
	v_cndmask_b32_e64 v59, v60, v62, s8                        // 0000000031a0: d501003b 00227d3c
	global_store_d16_hi_b16 v[57:58], v59, off                 // 0000000031a8: ee09407c 1d800000 00000039
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 0000000031b8: 8c7e097e
	v_or3_b32 v101, s44, v99, v86                              // 0000000031bc: d6580065 055ac62c
	v_or3_b32 v102, s45, 0, 0                                  // 0000000031c4: d6580066 0201002d
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000031cc: bf870112
	v_or_b32_e32 v99, 7, v101                                  // 0000000031d0: 38c6ca87
	v_mov_b32_e32 v100, v102                                   // 0000000031d4: 7ec80366
	v_or_b32_e32 v117, 1, v101                                 // 0000000031d8: 38eaca81
	v_or_b32_e32 v115, 2, v101                                 // 0000000031dc: 38e6ca82
	v_or_b32_e32 v109, 3, v101                                 // 0000000031e0: 38daca83
	v_or_b32_e32 v105, 4, v101                                 // 0000000031e4: 38d2ca84
	v_cmp_gt_u64_e64 s8, s[40:41], v[99:100]                   // 0000000031e8: d45c0008 0202c628
	v_or_b32_e32 v107, 5, v101                                 // 0000000031f0: 38d6ca85
	v_or_b32_e32 v103, 6, v101                                 // 0000000031f4: 38ceca86
	s_and_b32 s9, s27, s8                                      // 0000000031f8: 8b09081b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031fc: bf88ff9e
	s_and_b32 s9, s52, s9                                      // 000000003200: 8b090934
	s_wait_alu depctr_sa_sdst(0)                               // 000000003204: bf88ff9e
	s_xor_b32 s9, s9, -1                                       // 000000003208: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 00000000320c: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000003210: be8a2009
	s_wait_alu depctr_sa_sdst(0)                               // 000000003214: bf88ff9e
	s_xor_b32 s11, exec_lo, s10                                // 000000003218: 8d0b0a7e
	s_cbranch_execz 133                                        // 00000000321c: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1934>
	v_mov_b32_e32 v118, v102                                   // 000000003220: 7eec0366
	v_cmp_gt_i64_e64 s9, s[40:41], v[101:102]                  // 000000003224: d4540009 0202ca28
	v_mov_b32_e32 v116, v102                                   // 00000000322c: 7ee80366
	v_mov_b32_e32 v110, v102                                   // 000000003230: 7edc0366
	v_mov_b32_e32 v106, v102                                   // 000000003234: 7ed40366
	v_cmp_gt_i64_e64 s10, s[40:41], v[117:118]                 // 000000003238: d454000a 0202ea28
	v_mov_b32_e32 v108, v102                                   // 000000003240: 7ed80366
	s_wait_alu depctr_va_sdst(0)                               // 000000003244: bf88f19f
	v_cndmask_b32_e64 v58, 0, v102, s9                         // 000000003248: d501003a 0026cc80
	v_cndmask_b32_e64 v57, 0, v101, s9                         // 000000003250: d5010039 0026ca80
	v_cmp_gt_i64_e64 s9, s[40:41], v[115:116]                  // 000000003258: d4540009 0202e628
	v_mov_b32_e32 v104, v102                                   // 000000003260: 7ed00366
	v_cndmask_b32_e64 v59, 0, v117, s10                        // 000000003264: d501003b 002aea80
	v_cndmask_b32_e64 v60, 0, v102, s10                        // 00000000326c: d501003c 002acc80
	v_lshlrev_b64_e32 v[57:58], 2, v[57:58]                    // 000000003274: 3e727282
	v_cmp_gt_i64_e64 s10, s[40:41], v[109:110]                 // 000000003278: d454000a 0202da28
	s_wait_alu depctr_va_sdst(0)                               // 000000003280: bf88f19f
	v_cndmask_b32_e64 v61, 0, v115, s9                         // 000000003284: d501003d 0026e680
	v_cndmask_b32_e64 v62, 0, v102, s9                         // 00000000328c: d501003e 0026cc80
	v_lshlrev_b64_e32 v[59:60], 2, v[59:60]                    // 000000003294: 3e767682
	v_add_co_u32 v57, s9, s46, v57                             // 000000003298: d7000939 0202722e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 0000000032a0: bf870233
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]                    // 0000000032a4: 3e7a7a82
	s_wait_alu depctr_va_sdst(0)                               // 0000000032a8: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s47, v58, s9                // 0000000032ac: d5207c3a 0026742f
	v_add_co_u32 v59, s9, s46, v59                             // 0000000032b4: d700093b 0202762e
	v_cndmask_b32_e64 v63, 0, v109, s10                        // 0000000032bc: d501003f 002ada80
	v_cndmask_b32_e64 v64, 0, v102, s10                        // 0000000032c4: d5010040 002acc80
	s_wait_alu depctr_va_sdst(0)                               // 0000000032cc: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s47, v60, s9                // 0000000032d0: d5207c3c 0026782f
	v_cmp_gt_i64_e64 s9, s[40:41], v[105:106]                  // 0000000032d8: d4540009 0202d228
	v_add_co_u32 v112, s10, s46, v61                           // 0000000032e0: d7000a70 02027a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, s47, v62, s10              // 0000000032ec: d5207c71 002a7c2f
	v_cmp_gt_i64_e64 s10, s[40:41], v[107:108]                 // 0000000032f4: d454000a 0202d628
	v_lshlrev_b64_e32 v[61:62], 2, v[63:64]                    // 0000000032fc: 3e7a7e82
	v_cndmask_b32_e64 v63, 0, v105, s9                         // 000000003300: d501003f 0026d280
	v_cndmask_b32_e64 v64, 0, v102, s9                         // 000000003308: d5010040 0026cc80
	v_cmp_gt_i64_e64 s9, s[40:41], v[103:104]                  // 000000003310: d4540009 0202ce28
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: bf88f19f
	v_cndmask_b32_e64 v118, 0, v107, s10                       // 00000000331c: d5010076 002ad680
	v_cndmask_b32_e64 v119, 0, v102, s10                       // 000000003324: d5010077 002acc80
	v_add_co_u32 v122, s10, s46, v61                           // 00000000332c: d7000a7a 02027a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003334: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s47, v62, s10              // 000000003338: d5207c7b 002a7c2f
	v_lshlrev_b64_e32 v[61:62], 2, v[63:64]                    // 000000003340: 3e7a7e82
	v_lshlrev_b64_e32 v[63:64], 2, v[118:119]                  // 000000003344: 3e7eec82
	v_cndmask_b32_e64 v118, 0, v103, s9                        // 000000003348: d5010076 0026ce80
	v_cndmask_b32_e64 v119, 0, v102, s9                        // 000000003350: d5010077 0026cc80
	v_cmp_gt_i64_e64 s9, s[40:41], v[99:100]                   // 000000003358: d4540009 0202c628
	v_add_co_u32 v124, s10, s46, v61                           // 000000003360: d7000a7c 02027a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003368: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s47, v62, s10              // 00000000336c: d5207c7d 002a7c2f
	v_lshlrev_b64_e32 v[61:62], 2, v[118:119]                  // 000000003374: 3e7aec82
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000003378: bf8701d4
	v_cndmask_b32_e64 v119, 0, v100, s9                        // 00000000337c: d5010077 0026c880
	v_cndmask_b32_e64 v118, 0, v99, s9                         // 000000003384: d5010076 0026c680
	v_add_co_u32 v126, s9, s46, v63                            // 00000000338c: d700097e 02027e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003394: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s47, v64, s9               // 000000003398: d5207c7f 0026802f
	v_lshlrev_b64_e32 v[63:64], 2, v[118:119]                  // 0000000033a0: 3e7eec82
	v_add_co_u32 v118, s9, s46, v61                            // 0000000033a4: d7000976 02027a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000033ac: bf88f19f
	v_add_co_ci_u32_e64 v119, null, s47, v62, s9               // 0000000033b0: d5207c77 00267c2f
	s_delay_alu instid0(valu_dep_3)                            // 0000000033b8: bf870003
	v_add_co_u32 v128, s9, s46, v63                            // 0000000033bc: d7000980 02027e2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000033c4: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s47, v64, s9               // 0000000033c8: d5207c81 0026802f
	s_clause 0x7                                               // 0000000033d0: bf850007
	global_load_b32 v61, v[57:58], off                         // 0000000033d4: ee05007c 0000003d 00000039
	global_load_b32 v62, v[59:60], off                         // 0000000033e0: ee05007c 0000003e 0000003b
	global_load_b32 v63, v[112:113], off                       // 0000000033ec: ee05007c 0000003f 00000070
	global_load_b32 v64, v[122:123], off                       // 0000000033f8: ee05007c 00000040 0000007a
	global_load_b32 v57, v[124:125], off                       // 000000003404: ee05007c 00000039 0000007c
	global_load_b32 v58, v[126:127], off                       // 000000003410: ee05007c 0000003a 0000007e
	global_load_b32 v59, v[118:119], off                       // 00000000341c: ee05007c 0000003b 00000076
	global_load_b32 v60, v[128:129], off                       // 000000003428: ee05007c 0000003c 00000080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003434: bf88ff9e
	s_and_not1_saveexec_b32 s10, s11                           // 000000003438: be8a300b
	s_cbranch_execz 28                                         // 00000000343c: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x19b0>
	s_wait_loadcnt 0x3                                         // 000000003440: bfc00003
	v_add_co_u32 v57, s9, s44, v85                             // 000000003444: d7000939 0202aa2c
	s_wait_loadcnt 0x2                                         // 00000000344c: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000003450: bf88f19f
	v_add_co_ci_u32_e64 v58, null, s45, 0, s9                  // 000000003454: d5207c3a 0025002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000345c: bf870122
	v_add_co_u32 v57, s9, v57, v86                             // 000000003460: d7000939 0202ad39
	s_wait_alu depctr_va_sdst(0)                               // 000000003468: bf88f19f
	v_add_co_ci_u32_e64 v58, null, 0, v58, s9                  // 00000000346c: d5207c3a 00267480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003474: bf870091
	v_lshlrev_b64_e32 v[57:58], 2, v[57:58]                    // 000000003478: 3e727282
	v_add_co_u32 v57, s9, s46, v57                             // 00000000347c: d7000939 0202722e
	s_wait_alu depctr_va_sdst(0)                               // 000000003484: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003488: bf870002
	v_add_co_ci_u32_e64 v58, null, s47, v58, s9                // 00000000348c: d5207c3a 0026742f
	global_load_b128 v[61:64], v[57:58], off offset:64         // 000000003494: ee05c07c 0000003d 00004039
	s_wait_loadcnt 0x1                                         // 0000000034a0: bfc00001
	global_load_b128 v[57:60], v[57:58], off offset:80         // 0000000034a4: ee05c07c 00000039 00005039
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 0000000034b4: 8c7e0a7e
	global_load_u8 v84, v[139:140], off                        // 0000000034b8: ee04007c 00000054 0000008b
	s_wait_loadcnt 0x0                                         // 0000000034c4: bfc00000
	v_lshlrev_b32_e32 v90, 23, v84                             // 0000000034c8: 30b4a897
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000034cc: bf870091
	v_mul_f32_e32 v84, v61, v90                                // 0000000034d0: 10a8b53d
	v_cmp_class_f32_e64 s9, v84, 0x198                         // 0000000034d4: d47e0009 0201ff54 00000198
	v_mul_f32_e32 v84, v49, v84                                // 0000000034e0: 10a8a931
	s_xor_b32 s9, s9, -1                                       // 0000000034e4: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034e8: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 0000000034ec: be8a2009
	s_cbranch_execnz 3909                                      // 0000000034f0: bfa60f45 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5708>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 0000000034f8: 8c7e0a7e
	v_mul_f32_e32 v49, v62, v90                                // 0000000034fc: 1062b53e
	s_delay_alu instid0(valu_dep_1)                            // 000000003500: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 000000003504: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v61, v50, v49                                // 000000003510: 107a6332
	s_xor_b32 s9, s9, -1                                       // 000000003514: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 000000003518: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000351c: be8a2009
	s_cbranch_execnz 3915                                      // 000000003520: bfa60f4b <tessera_rocm_folded_matmul_a629485a5b78541c+0x5750>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003524: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003528: 8c7e0a7e
	v_mul_f32_e32 v49, v63, v90                                // 00000000352c: 1062b53f
	s_delay_alu instid0(valu_dep_1)                            // 000000003530: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 000000003534: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v62, v51, v49                                // 000000003540: 107c6333
	s_xor_b32 s9, s9, -1                                       // 000000003544: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 000000003548: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000354c: be8a2009
	s_cbranch_execnz 3921                                      // 000000003550: bfa60f51 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5798>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003554: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003558: 8c7e0a7e
	v_mul_f32_e32 v49, v64, v90                                // 00000000355c: 1062b540
	s_delay_alu instid0(valu_dep_1)                            // 000000003560: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 000000003564: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v51, v52, v49                                // 000000003570: 10666334
	s_xor_b32 s9, s9, -1                                       // 000000003574: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 000000003578: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000357c: be8a2009
	s_cbranch_execnz 3927                                      // 000000003580: bfa60f57 <tessera_rocm_folded_matmul_a629485a5b78541c+0x57e0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003588: 8c7e0a7e
	v_mul_f32_e32 v49, v57, v90                                // 00000000358c: 1062b539
	s_delay_alu instid0(valu_dep_1)                            // 000000003590: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 000000003594: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v52, v53, v49                                // 0000000035a0: 10686335
	s_xor_b32 s9, s9, -1                                       // 0000000035a4: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035a8: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 0000000035ac: be8a2009
	s_cbranch_execnz 3933                                      // 0000000035b0: bfa60f5d <tessera_rocm_folded_matmul_a629485a5b78541c+0x5828>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 0000000035b8: 8c7e0a7e
	v_mul_f32_e32 v49, v58, v90                                // 0000000035bc: 1062b53a
	s_delay_alu instid0(valu_dep_1)                            // 0000000035c0: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 0000000035c4: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v53, v54, v49                                // 0000000035d0: 106a6336
	s_xor_b32 s9, s9, -1                                       // 0000000035d4: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035d8: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 0000000035dc: be8a2009
	s_cbranch_execnz 3939                                      // 0000000035e0: bfa60f63 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5870>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 0000000035e8: 8c7e0a7e
	v_mul_f32_e32 v49, v59, v90                                // 0000000035ec: 1062b53b
	s_delay_alu instid0(valu_dep_1)                            // 0000000035f0: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 0000000035f4: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v54, v55, v49                                // 000000003600: 106c6337
	s_xor_b32 s9, s9, -1                                       // 000000003604: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 000000003608: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000360c: be8a2009
	s_cbranch_execnz 3945                                      // 000000003610: bfa60f69 <tessera_rocm_folded_matmul_a629485a5b78541c+0x58b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003614: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003618: 8c7e0a7e
	v_mul_f32_e32 v49, v60, v90                                // 00000000361c: 1062b53c
	s_delay_alu instid0(valu_dep_1)                            // 000000003620: bf870001
	v_cmp_class_f32_e64 s9, v49, 0x198                         // 000000003624: d47e0009 0201ff31 00000198
	v_mul_f32_e32 v55, v56, v49                                // 000000003630: 106e6338
	s_xor_b32 s9, s9, -1                                       // 000000003634: 8d09c109
	s_wait_alu depctr_sa_sdst(0)                               // 000000003638: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000363c: be8a2009
	s_cbranch_execnz 3951                                      // 000000003640: bfa60f6f <tessera_rocm_folded_matmul_a629485a5b78541c+0x5900>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003644: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003648: 8c7e0a7e
	v_mul_lo_u32 v58, v102, s42                                // 00000000364c: d72c003a 02005566
	v_mul_lo_u32 v59, v101, s43                                // 000000003654: d72c003b 02005765
	v_mad_co_u64_u32 v[56:57], null, v101, s42, 0              // 00000000365c: d6fe7c38 02005565
	v_sub_co_u32 v49, s9, s40, v101                            // 000000003664: d7010931 0202ca28
	s_wait_alu depctr_va_sdst(0)                               // 00000000366c: bf88f19f
	v_sub_co_ci_u32_e64 v50, null, s41, v102, s9               // 000000003670: d5217c32 0026cc29
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003678: bf870211
	v_cmp_lt_i64_e64 s12, 0, v[49:50]                          // 00000000367c: d451000c 02026280
	v_add3_u32 v57, v57, v59, v58                              // 000000003684: d6550039 04ea7739
	s_delay_alu instid0(valu_dep_1)                            // 00000000368c: bf870001
	v_lshlrev_b64_e32 v[57:58], 1, v[56:57]                    // 000000003690: 3e727081
	s_and_b32 s9, s36, s12                                     // 000000003694: 8b090c24
	s_wait_alu depctr_sa_sdst(0)                               // 000000003698: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 00000000369c: be8a2009
	s_cbranch_execz 28                                         // 0000000036a0: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1c14>
	s_wait_kmcnt 0x0                                           // 0000000036a4: bfc70000
	v_add_co_u32 v59, s9, s48, v57                             // 0000000036a8: d700093b 02027230
	v_bfe_u32 v56, v84, 16, 1                                  // 0000000036b0: d6100038 02052154
	s_wait_alu depctr_va_sdst(0)                               // 0000000036b8: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s49, v58, s9                // 0000000036bc: d5207c3c 00267431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000036c4: bf870193
	v_add_co_u32 v59, s9, v59, v137                            // 0000000036c8: d700093b 0203133b
	v_add3_u32 v56, v56, v84, 0x7fff                           // 0000000036d0: d6550038 03fea938 00007fff
	v_or_b32_e32 v63, 0x400000, v84                            // 0000000036dc: 387ea8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000036e4: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v60, v138, s9               // 0000000036e8: d5207c3c 0027153c
	v_cmp_u_f32_e64 s9, v84, v84                               // 0000000036f0: d4180009 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000036fc: bf870001
	v_cndmask_b32_e64 v56, v56, v63, s9                        // 000000003700: d5010038 00267f38
	global_store_d16_hi_b16 v[59:60], v56, off                 // 000000003708: ee09407c 1c000000 0000003b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003714: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000003718: 8c7e0a7e
	v_cmp_lt_i64_e64 s9, 1, v[49:50]                           // 00000000371c: d4510009 02026281
	s_and_b32 s10, s36, s9                                     // 000000003724: 8b0a0924
	s_wait_alu depctr_sa_sdst(0)                               // 000000003728: bf88ff9e
	s_and_saveexec_b32 s11, s10                                // 00000000372c: be8b200a
	s_cbranch_execz 28                                         // 000000003730: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1ca4>
	s_wait_kmcnt 0x0                                           // 000000003734: bfc70000
	v_add_co_u32 v59, s10, s48, v57                            // 000000003738: d7000a3b 02027230
	v_bfe_u32 v56, v61, 16, 1                                  // 000000003740: d6100038 0205213d
	s_wait_alu depctr_va_sdst(0)                               // 000000003748: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s49, v58, s10               // 00000000374c: d5207c3c 002a7431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003754: bf870193
	v_add_co_u32 v59, s10, v59, v65                            // 000000003758: d7000a3b 0202833b
	v_add3_u32 v56, v56, v61, 0x7fff                           // 000000003760: d6550038 03fe7b38 00007fff
	v_or_b32_e32 v63, 0x400000, v61                            // 00000000376c: 387e7aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003774: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v60, v66, s10               // 000000003778: d5207c3c 002a853c
	v_cmp_u_f32_e64 s10, v61, v61                              // 000000003780: d418000a 02027b3d
	s_wait_alu depctr_va_sdst(0)                               // 000000003788: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000378c: bf870001
	v_cndmask_b32_e64 v56, v56, v63, s10                       // 000000003790: d5010038 002a7f38
	global_store_d16_hi_b16 v[59:60], v56, off                 // 000000003798: ee09407c 1c000000 0000003b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 0000000037a8: 8c7e0b7e
	v_cmp_lt_i64_e64 s10, 2, v[49:50]                          // 0000000037ac: d451000a 02026282
	s_and_b32 s11, s36, s10                                    // 0000000037b4: 8b0b0a24
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037b8: bf88ff9e
	s_and_saveexec_b32 s13, s11                                // 0000000037bc: be8d200b
	s_cbranch_execz 28                                         // 0000000037c0: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1d34>
	s_wait_kmcnt 0x0                                           // 0000000037c4: bfc70000
	v_add_co_u32 v59, s11, s48, v57                            // 0000000037c8: d7000b3b 02027230
	v_bfe_u32 v56, v62, 16, 1                                  // 0000000037d0: d6100038 0205213e
	s_wait_alu depctr_va_sdst(0)                               // 0000000037d8: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s49, v58, s11               // 0000000037dc: d5207c3c 002e7431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037e4: bf870193
	v_add_co_u32 v59, s11, v59, v67                            // 0000000037e8: d7000b3b 0202873b
	v_add3_u32 v56, v56, v62, 0x7fff                           // 0000000037f0: d6550038 03fe7d38 00007fff
	v_or_b32_e32 v61, 0x400000, v62                            // 0000000037fc: 387a7cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003804: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v60, v68, s11               // 000000003808: d5207c3c 002e893c
	v_cmp_u_f32_e64 s11, v62, v62                              // 000000003810: d418000b 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000003818: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000381c: bf870001
	v_cndmask_b32_e64 v56, v56, v61, s11                       // 000000003820: d5010038 002e7b38
	global_store_d16_hi_b16 v[59:60], v56, off                 // 000000003828: ee09407c 1c000000 0000003b
	s_wait_alu depctr_sa_sdst(0)                               // 000000003834: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000003838: 8c7e0d7e
	v_cmp_lt_i64_e64 s11, 3, v[49:50]                          // 00000000383c: d451000b 02026283
	s_and_b32 s13, s36, s11                                    // 000000003844: 8b0d0b24
	s_wait_alu depctr_sa_sdst(0)                               // 000000003848: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 00000000384c: be8e200d
	s_cbranch_execz 28                                         // 000000003850: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1dc4>
	s_wait_kmcnt 0x0                                           // 000000003854: bfc70000
	v_add_co_u32 v59, s13, s48, v57                            // 000000003858: d7000d3b 02027230
	v_bfe_u32 v56, v51, 16, 1                                  // 000000003860: d6100038 02052133
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v60, null, s49, v58, s13               // 00000000386c: d5207c3c 00367431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003874: bf870193
	v_add_co_u32 v59, s13, v59, v69                            // 000000003878: d7000d3b 02028b3b
	v_add3_u32 v56, v56, v51, 0x7fff                           // 000000003880: d6550038 03fe6738 00007fff
	v_or_b32_e32 v61, 0x400000, v51                            // 00000000388c: 387a66ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003894: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v60, v70, s13               // 000000003898: d5207c3c 00368d3c
	v_cmp_u_f32_e64 s13, v51, v51                              // 0000000038a0: d418000d 02026733
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000038ac: bf870001
	v_cndmask_b32_e64 v51, v56, v61, s13                       // 0000000038b0: d5010033 00367b38
	global_store_d16_hi_b16 v[59:60], v51, off                 // 0000000038b8: ee09407c 19800000 0000003b
	s_or_b32 exec_lo, exec_lo, s14                             // 0000000038c4: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 4, v[49:50]                          // 0000000038c8: d451000d 02026284
	s_and_b32 s14, s36, s13                                    // 0000000038d0: 8b0e0d24
	s_delay_alu instid0(salu_cycle_1)                          // 0000000038d4: bf870009
	s_and_saveexec_b32 s15, s14                                // 0000000038d8: be8f200e
	s_cbranch_execz 27                                         // 0000000038dc: bfa5001b <tessera_rocm_folded_matmul_a629485a5b78541c+0x1e4c>
	s_wait_kmcnt 0x0                                           // 0000000038e0: bfc70000
	v_add_co_u32 v56, s14, s48, v57                            // 0000000038e4: d7000e38 02027230
	v_bfe_u32 v51, v52, 16, 1                                  // 0000000038ec: d6100033 02052134
	v_add_co_ci_u32_e64 v60, null, s49, v58, s14               // 0000000038f4: d5207c3c 003a7431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000038fc: bf870193
	v_add_co_u32 v59, s14, v56, v71                            // 000000003900: d7000e3b 02028f38
	v_add3_u32 v51, v51, v52, 0x7fff                           // 000000003908: d6550033 03fe6933 00007fff
	v_or_b32_e32 v61, 0x400000, v52                            // 000000003914: 387a68ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000391c: bf88f19f
	v_add_co_ci_u32_e64 v60, null, v60, v72, s14               // 000000003920: d5207c3c 003a913c
	v_cmp_u_f32_e64 s14, v52, v52                              // 000000003928: d418000e 02026934
	s_wait_alu depctr_va_sdst(0)                               // 000000003930: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003934: bf870001
	v_cndmask_b32_e64 v51, v51, v61, s14                       // 000000003938: d5010033 003a7b33
	global_store_d16_hi_b16 v[59:60], v51, off                 // 000000003940: ee09407c 19800000 0000003b
	s_or_b32 exec_lo, exec_lo, s15                             // 00000000394c: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 5, v[49:50]                          // 000000003950: d451000e 02026285
	s_and_b32 s15, s36, s14                                    // 000000003958: 8b0f0e24
	s_wait_alu depctr_sa_sdst(0)                               // 00000000395c: bf88ff9e
	s_and_saveexec_b32 s16, s15                                // 000000003960: be90200f
	s_cbranch_execz 28                                         // 000000003964: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1ed8>
	v_bfe_u32 v51, v53, 16, 1                                  // 000000003968: d6100033 02052135
	s_wait_kmcnt 0x0                                           // 000000003970: bfc70000
	v_add_co_u32 v52, s15, s48, v57                            // 000000003974: d7000f34 02027230
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s49, v58, s15               // 000000003980: d5207c38 003e7431
	v_add3_u32 v59, v51, v53, 0x7fff                           // 000000003988: d655003b 03fe6b33 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003994: bf870003
	v_add_co_u32 v51, s15, v52, v73                            // 000000003998: d7000f33 02029334
	v_or_b32_e32 v60, 0x400000, v53                            // 0000000039a0: 38786aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000039a8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v56, v74, s15               // 0000000039ac: d5207c34 003e9538
	v_cmp_u_f32_e64 s15, v53, v53                              // 0000000039b4: d418000f 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 0000000039bc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000039c0: bf870001
	v_cndmask_b32_e64 v53, v59, v60, s15                       // 0000000039c4: d5010035 003e793b
	global_store_d16_hi_b16 v[51:52], v53, off                 // 0000000039cc: ee09407c 1a800000 00000033
	s_or_b32 exec_lo, exec_lo, s16                             // 0000000039d8: 8c7e107e
	v_cmp_lt_i64_e64 s15, 6, v[49:50]                          // 0000000039dc: d451000f 02026286
	s_and_b32 s16, s36, s15                                    // 0000000039e4: 8b100f24
	s_delay_alu instid0(salu_cycle_1)                          // 0000000039e8: bf870009
	s_and_saveexec_b32 s17, s16                                // 0000000039ec: be912010
	s_cbranch_execz 28                                         // 0000000039f0: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1f64>
	v_bfe_u32 v51, v54, 16, 1                                  // 0000000039f4: d6100033 02052136
	s_wait_kmcnt 0x0                                           // 0000000039fc: bfc70000
	v_add_co_u32 v52, s16, s48, v57                            // 000000003a00: d7001034 02027230
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000003a08: bf870191
	v_add_co_ci_u32_e64 v53, null, s49, v58, s16               // 000000003a0c: d5207c35 00427431
	v_add3_u32 v56, v51, v54, 0x7fff                           // 000000003a14: d6550038 03fe6d33 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003a20: bf870003
	v_add_co_u32 v51, s16, v52, v75                            // 000000003a24: d7001033 02029734
	v_or_b32_e32 v59, 0x400000, v54                            // 000000003a2c: 38766cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003a34: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v53, v76, s16               // 000000003a38: d5207c34 00429935
	v_cmp_u_f32_e64 s16, v54, v54                              // 000000003a40: d4180010 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000003a48: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003a4c: bf870001
	v_cndmask_b32_e64 v53, v56, v59, s16                       // 000000003a50: d5010035 00427738
	global_store_d16_hi_b16 v[51:52], v53, off                 // 000000003a58: ee09407c 1a800000 00000033
	s_or_b32 exec_lo, exec_lo, s17                             // 000000003a64: 8c7e117e
	v_cmp_lt_i64_e64 s16, 7, v[49:50]                          // 000000003a68: d4510010 02026287
	s_and_b32 s17, s36, s16                                    // 000000003a70: 8b111024
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a74: bf88ff9e
	s_and_saveexec_b32 s18, s17                                // 000000003a78: be922011
	s_cbranch_execz 28                                         // 000000003a7c: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x1ff0>
	v_bfe_u32 v49, v55, 16, 1                                  // 000000003a80: d6100031 02052137
	s_wait_kmcnt 0x0                                           // 000000003a88: bfc70000
	v_add_co_u32 v50, s17, s48, v57                            // 000000003a8c: d7001132 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000003a94: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s49, v58, s17               // 000000003a98: d5207c33 00467431
	v_add3_u32 v52, v49, v55, 0x7fff                           // 000000003aa0: d6550034 03fe6f31 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003aac: bf870003
	v_add_co_u32 v49, s17, v50, v77                            // 000000003ab0: d7001131 02029b32
	v_or_b32_e32 v53, 0x400000, v55                            // 000000003ab8: 386a6eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac0: bf88f19f
	v_add_co_ci_u32_e64 v50, null, v51, v78, s17               // 000000003ac4: d5207c32 00469d33
	v_cmp_u_f32_e64 s17, v55, v55                              // 000000003acc: d4180011 02026f37
	s_wait_alu depctr_va_sdst(0)                               // 000000003ad4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000003ad8: bf870001
	v_cndmask_b32_e64 v51, v52, v53, s17                       // 000000003adc: d5010033 00466b34
	global_store_d16_hi_b16 v[49:50], v51, off                 // 000000003ae4: ee09407c 19800000 00000031
	s_or_b32 exec_lo, exec_lo, s18                             // 000000003af0: 8c7e127e
	v_or3_b32 v61, s44, v111, v86                              // 000000003af4: d658003d 055ade2c
	v_mov_b32_e32 v60, v102                                    // 000000003afc: 7e780366
	v_mov_b32_e32 v62, v102                                    // 000000003b00: 7e7c0366
	s_delay_alu instid0(valu_dep_3)                            // 000000003b04: bf870003
	v_or_b32_e32 v59, 7, v61                                   // 000000003b08: 38767a87
	v_or_b32_e32 v127, 1, v61                                  // 000000003b0c: 38fe7a81
	v_or_b32_e32 v125, 2, v61                                  // 000000003b10: 38fa7a82
	v_or_b32_e32 v119, 3, v61                                  // 000000003b14: 38ee7a83
	v_or_b32_e32 v111, 4, v61                                  // 000000003b18: 38de7a84
	v_cmp_gt_u64_e64 s17, s[40:41], v[59:60]                   // 000000003b1c: d45c0011 02027628
	v_or_b32_e32 v113, 5, v61                                  // 000000003b24: 38e27a85
	v_or_b32_e32 v63, 6, v61                                   // 000000003b28: 387e7a86
	s_and_b32 s18, s27, s17                                    // 000000003b2c: 8b12111b
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000003b30: bf870499
	s_and_b32 s18, s52, s18                                    // 000000003b34: 8b121234
	s_xor_b32 s18, s18, -1                                     // 000000003b38: 8d12c112
	s_delay_alu instid0(salu_cycle_1) | instskip(next) | instid1(salu_cycle_1)// 000000003b3c: bf870499
	s_and_saveexec_b32 s19, s18                                // 000000003b40: be932012
	s_xor_b32 s20, exec_lo, s19                                // 000000003b44: 8d14137e
	s_cbranch_execz 132                                        // 000000003b48: bfa50084 <tessera_rocm_folded_matmul_a629485a5b78541c+0x225c>
	v_mov_b32_e32 v128, v62                                    // 000000003b4c: 7f00033e
	v_cmp_gt_i64_e64 s18, s[40:41], v[61:62]                   // 000000003b50: d4540012 02027a28
	v_mov_b32_e32 v126, v62                                    // 000000003b58: 7efc033e
	v_mov_b32_e32 v120, v62                                    // 000000003b5c: 7ef0033e
	v_mov_b32_e32 v112, v62                                    // 000000003b60: 7ee0033e
	v_cmp_gt_i64_e64 s19, s[40:41], v[127:128]                 // 000000003b64: d4540013 0202fe28
	v_mov_b32_e32 v114, v62                                    // 000000003b6c: 7ee4033e
	v_cndmask_b32_e64 v50, 0, v62, s18                         // 000000003b70: d5010032 004a7c80
	v_cndmask_b32_e64 v49, 0, v61, s18                         // 000000003b78: d5010031 004a7a80
	v_cmp_gt_i64_e64 s18, s[40:41], v[125:126]                 // 000000003b80: d4540012 0202fa28
	v_mov_b32_e32 v64, v62                                     // 000000003b88: 7e80033e
	v_cndmask_b32_e64 v51, 0, v127, s19                        // 000000003b8c: d5010033 004efe80
	v_cndmask_b32_e64 v52, 0, v62, s19                         // 000000003b94: d5010034 004e7c80
	v_lshlrev_b64_e32 v[49:50], 2, v[49:50]                    // 000000003b9c: 3e626282
	v_cmp_gt_i64_e64 s19, s[40:41], v[119:120]                 // 000000003ba0: d4540013 0202ee28
	s_wait_alu depctr_va_sdst(0)                               // 000000003ba8: bf88f19f
	v_cndmask_b32_e64 v53, 0, v125, s18                        // 000000003bac: d5010035 004afa80
	v_cndmask_b32_e64 v54, 0, v62, s18                         // 000000003bb4: d5010036 004a7c80
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]                    // 000000003bbc: 3e666682
	v_add_co_u32 v49, s18, s46, v49                            // 000000003bc0: d7001231 0202622e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000003bc8: bf870233
	v_lshlrev_b64_e32 v[53:54], 2, v[53:54]                    // 000000003bcc: 3e6a6a82
	s_wait_alu depctr_va_sdst(0)                               // 000000003bd0: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s47, v50, s18               // 000000003bd4: d5207c32 004a642f
	v_add_co_u32 v51, s18, s46, v51                            // 000000003bdc: d7001233 0202662e
	v_cndmask_b32_e64 v55, 0, v119, s19                        // 000000003be4: d5010037 004eee80
	v_cndmask_b32_e64 v56, 0, v62, s19                         // 000000003bec: d5010038 004e7c80
	s_wait_alu depctr_va_sdst(0)                               // 000000003bf4: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s47, v52, s18               // 000000003bf8: d5207c34 004a682f
	v_cmp_gt_i64_e64 s18, s[40:41], v[111:112]                 // 000000003c00: d4540012 0202de28
	v_add_co_u32 v122, s19, s46, v53                           // 000000003c08: d700137a 02026a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c10: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s47, v54, s19              // 000000003c14: d5207c7b 004e6c2f
	v_cmp_gt_i64_e64 s19, s[40:41], v[113:114]                 // 000000003c1c: d4540013 0202e228
	v_lshlrev_b64_e32 v[53:54], 2, v[55:56]                    // 000000003c24: 3e6a6e82
	v_cndmask_b32_e64 v55, 0, v111, s18                        // 000000003c28: d5010037 004ade80
	v_cndmask_b32_e64 v56, 0, v62, s18                         // 000000003c30: d5010038 004a7c80
	v_cmp_gt_i64_e64 s18, s[40:41], v[63:64]                   // 000000003c38: d4540012 02027e28
	s_wait_alu depctr_va_sdst(0)                               // 000000003c40: bf88f19f
	v_cndmask_b32_e64 v128, 0, v113, s19                       // 000000003c44: d5010080 004ee280
	v_cndmask_b32_e64 v129, 0, v62, s19                        // 000000003c4c: d5010081 004e7c80
	v_add_co_u32 v130, s19, s46, v53                           // 000000003c54: d7001382 02026a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c5c: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s47, v54, s19              // 000000003c60: d5207c83 004e6c2f
	v_lshlrev_b64_e32 v[53:54], 2, v[55:56]                    // 000000003c68: 3e6a6e82
	v_lshlrev_b64_e32 v[55:56], 2, v[128:129]                  // 000000003c6c: 3e6f0082
	v_cndmask_b32_e64 v128, 0, v63, s18                        // 000000003c70: d5010080 004a7e80
	v_cndmask_b32_e64 v129, 0, v62, s18                        // 000000003c78: d5010081 004a7c80
	v_cmp_gt_i64_e64 s18, s[40:41], v[59:60]                   // 000000003c80: d4540012 02027628
	v_add_co_u32 v132, s19, s46, v53                           // 000000003c88: d7001384 02026a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c90: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s47, v54, s19              // 000000003c94: d5207c85 004e6c2f
	v_lshlrev_b64_e32 v[53:54], 2, v[128:129]                  // 000000003c9c: 3e6b0082
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000003ca0: bf8701d4
	v_cndmask_b32_e64 v129, 0, v60, s18                        // 000000003ca4: d5010081 004a7880
	v_cndmask_b32_e64 v128, 0, v59, s18                        // 000000003cac: d5010080 004a7680
	v_add_co_u32 v143, s18, s46, v55                           // 000000003cb4: d700128f 02026e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cbc: bf88f19f
	v_add_co_ci_u32_e64 v144, null, s47, v56, s18              // 000000003cc0: d5207c90 004a702f
	v_lshlrev_b64_e32 v[55:56], 2, v[128:129]                  // 000000003cc8: 3e6f0082
	v_add_co_u32 v128, s18, s46, v53                           // 000000003ccc: d7001280 02026a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cd4: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s47, v54, s18              // 000000003cd8: d5207c81 004a6c2f
	s_delay_alu instid0(valu_dep_3)                            // 000000003ce0: bf870003
	v_add_co_u32 v145, s18, s46, v55                           // 000000003ce4: d7001291 02026e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cec: bf88f19f
	v_add_co_ci_u32_e64 v146, null, s47, v56, s18              // 000000003cf0: d5207c92 004a702f
	s_clause 0x7                                               // 000000003cf8: bf850007
	global_load_b32 v53, v[49:50], off                         // 000000003cfc: ee05007c 00000035 00000031
	global_load_b32 v54, v[51:52], off                         // 000000003d08: ee05007c 00000036 00000033
	global_load_b32 v55, v[122:123], off                       // 000000003d14: ee05007c 00000037 0000007a
	global_load_b32 v56, v[130:131], off                       // 000000003d20: ee05007c 00000038 00000082
	global_load_b32 v49, v[132:133], off                       // 000000003d2c: ee05007c 00000031 00000084
	global_load_b32 v50, v[143:144], off                       // 000000003d38: ee05007c 00000032 0000008f
	global_load_b32 v51, v[128:129], off                       // 000000003d44: ee05007c 00000033 00000080
	global_load_b32 v52, v[145:146], off                       // 000000003d50: ee05007c 00000034 00000091
	s_and_not1_saveexec_b32 s19, s20                           // 000000003d5c: be933014
	s_cbranch_execz 28                                         // 000000003d60: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x22d4>
	s_wait_loadcnt 0x3                                         // 000000003d64: bfc00003
	v_add_co_u32 v49, s18, s44, v85                            // 000000003d68: d7001231 0202aa2c
	s_wait_loadcnt 0x2                                         // 000000003d70: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000003d74: bf88f19f
	v_add_co_ci_u32_e64 v50, null, s45, 0, s18                 // 000000003d78: d5207c32 0049002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003d80: bf870122
	v_add_co_u32 v49, s18, v49, v86                            // 000000003d84: d7001231 0202ad31
	s_wait_alu depctr_va_sdst(0)                               // 000000003d8c: bf88f19f
	v_add_co_ci_u32_e64 v50, null, 0, v50, s18                 // 000000003d90: d5207c32 004a6480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d98: bf870091
	v_lshlrev_b64_e32 v[49:50], 2, v[49:50]                    // 000000003d9c: 3e626282
	v_add_co_u32 v49, s18, s46, v49                            // 000000003da0: d7001231 0202622e
	s_wait_alu depctr_va_sdst(0)                               // 000000003da8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003dac: bf870002
	v_add_co_ci_u32_e64 v50, null, s47, v50, s18               // 000000003db0: d5207c32 004a642f
	global_load_b128 v[53:56], v[49:50], off offset:128        // 000000003db8: ee05c07c 00000035 00008031
	s_wait_loadcnt 0x1                                         // 000000003dc4: bfc00001
	global_load_b128 v[49:52], v[49:50], off offset:144        // 000000003dc8: ee05c07c 00000031 00009031
	s_wait_alu depctr_sa_sdst(0)                               // 000000003dd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003dd8: 8c7e137e
	global_load_u8 v64, v[139:140], off                        // 000000003ddc: ee04007c 00000040 0000008b
	s_wait_loadcnt 0x0                                         // 000000003de8: bfc00000
	v_lshlrev_b32_e32 v84, 23, v64                             // 000000003dec: 30a88097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003df0: bf870091
	v_mul_f32_e32 v64, v53, v84                                // 000000003df4: 1080a935
	v_cmp_class_f32_e64 s18, v64, 0x198                        // 000000003df8: d47e0012 0201ff40 00000198
	v_mul_f32_e32 v64, v41, v64                                // 000000003e04: 10808129
	s_xor_b32 s18, s18, -1                                     // 000000003e08: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e0c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003e10: be932012
	s_cbranch_execnz 3468                                      // 000000003e14: bfa60d8c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5948>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e1c: 8c7e137e
	v_mul_f32_e32 v41, v54, v84                                // 000000003e20: 1052a936
	s_delay_alu instid0(valu_dep_1)                            // 000000003e24: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003e28: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v53, v42, v41                                // 000000003e34: 106a532a
	s_xor_b32 s18, s18, -1                                     // 000000003e38: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e3c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003e40: be932012
	s_cbranch_execnz 3474                                      // 000000003e44: bfa60d92 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5990>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e4c: 8c7e137e
	v_mul_f32_e32 v41, v55, v84                                // 000000003e50: 1052a937
	s_delay_alu instid0(valu_dep_1)                            // 000000003e54: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003e58: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v54, v43, v41                                // 000000003e64: 106c532b
	s_xor_b32 s18, s18, -1                                     // 000000003e68: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e6c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003e70: be932012
	s_cbranch_execnz 3480                                      // 000000003e74: bfa60d98 <tessera_rocm_folded_matmul_a629485a5b78541c+0x59d8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003e7c: 8c7e137e
	v_mul_f32_e32 v41, v56, v84                                // 000000003e80: 1052a938
	s_delay_alu instid0(valu_dep_1)                            // 000000003e84: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003e88: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v43, v44, v41                                // 000000003e94: 1056532c
	s_xor_b32 s18, s18, -1                                     // 000000003e98: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e9c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003ea0: be932012
	s_cbranch_execnz 3486                                      // 000000003ea4: bfa60d9e <tessera_rocm_folded_matmul_a629485a5b78541c+0x5a20>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ea8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003eac: 8c7e137e
	v_mul_f32_e32 v41, v49, v84                                // 000000003eb0: 1052a931
	s_delay_alu instid0(valu_dep_1)                            // 000000003eb4: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003eb8: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v44, v45, v41                                // 000000003ec4: 1058532d
	s_xor_b32 s18, s18, -1                                     // 000000003ec8: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ecc: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003ed0: be932012
	s_cbranch_execnz 3492                                      // 000000003ed4: bfa60da4 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5a68>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ed8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003edc: 8c7e137e
	v_mul_f32_e32 v41, v50, v84                                // 000000003ee0: 1052a932
	s_delay_alu instid0(valu_dep_1)                            // 000000003ee4: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003ee8: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v45, v46, v41                                // 000000003ef4: 105a532e
	s_xor_b32 s18, s18, -1                                     // 000000003ef8: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003efc: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003f00: be932012
	s_cbranch_execnz 3498                                      // 000000003f04: bfa60daa <tessera_rocm_folded_matmul_a629485a5b78541c+0x5ab0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f08: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003f0c: 8c7e137e
	v_mul_f32_e32 v41, v51, v84                                // 000000003f10: 1052a933
	s_delay_alu instid0(valu_dep_1)                            // 000000003f14: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003f18: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v46, v47, v41                                // 000000003f24: 105c532f
	s_xor_b32 s18, s18, -1                                     // 000000003f28: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f2c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003f30: be932012
	s_cbranch_execnz 3504                                      // 000000003f34: bfa60db0 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5af8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f38: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003f3c: 8c7e137e
	v_mul_f32_e32 v41, v52, v84                                // 000000003f40: 1052a934
	s_delay_alu instid0(valu_dep_1)                            // 000000003f44: bf870001
	v_cmp_class_f32_e64 s18, v41, 0x198                        // 000000003f48: d47e0012 0201ff29 00000198
	v_mul_f32_e32 v47, v48, v41                                // 000000003f54: 105e5330
	s_xor_b32 s18, s18, -1                                     // 000000003f58: 8d12c112
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f5c: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003f60: be932012
	s_cbranch_execnz 3510                                      // 000000003f64: bfa60db6 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5b40>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f68: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 000000003f6c: 8c7e137e
	v_mul_lo_u32 v50, v62, s42                                 // 000000003f70: d72c0032 0200553e
	v_mul_lo_u32 v51, v61, s43                                 // 000000003f78: d72c0033 0200573d
	v_mad_co_u64_u32 v[48:49], null, v61, s42, 0               // 000000003f80: d6fe7c30 0200553d
	v_sub_co_u32 v41, s18, s40, v61                            // 000000003f88: d7011229 02027a28
	s_wait_alu depctr_va_sdst(0)                               // 000000003f90: bf88f19f
	v_sub_co_ci_u32_e64 v42, null, s41, v62, s18               // 000000003f94: d5217c2a 004a7c29
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003f9c: bf870211
	v_cmp_lt_i64_e64 s21, 0, v[41:42]                          // 000000003fa0: d4510015 02025280
	v_add3_u32 v49, v49, v51, v50                              // 000000003fa8: d6550031 04ca6731
	s_delay_alu instid0(valu_dep_1)                            // 000000003fb0: bf870001
	v_lshlrev_b64_e32 v[49:50], 1, v[48:49]                    // 000000003fb4: 3e626081
	s_and_b32 s18, s36, s21                                    // 000000003fb8: 8b121524
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fbc: bf88ff9e
	s_and_saveexec_b32 s19, s18                                // 000000003fc0: be932012
	s_cbranch_execz 28                                         // 000000003fc4: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2538>
	s_wait_kmcnt 0x0                                           // 000000003fc8: bfc70000
	v_add_co_u32 v51, s18, s48, v49                            // 000000003fcc: d7001233 02026230
	v_bfe_u32 v48, v64, 16, 1                                  // 000000003fd4: d6100030 02052140
	s_wait_alu depctr_va_sdst(0)                               // 000000003fdc: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s49, v50, s18               // 000000003fe0: d5207c34 004a6431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003fe8: bf870193
	v_add_co_u32 v51, s18, v51, v137                           // 000000003fec: d7001233 02031333
	v_add3_u32 v48, v48, v64, 0x7fff                           // 000000003ff4: d6550030 03fe8130 00007fff
	v_or_b32_e32 v55, 0x400000, v64                            // 000000004000: 386e80ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004008: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v52, v138, s18              // 00000000400c: d5207c34 004b1534
	v_cmp_u_f32_e64 s18, v64, v64                              // 000000004014: d4180012 02028140
	s_wait_alu depctr_va_sdst(0)                               // 00000000401c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004020: bf870001
	v_cndmask_b32_e64 v48, v48, v55, s18                       // 000000004024: d5010030 004a6f30
	global_store_d16_hi_b16 v[51:52], v48, off                 // 00000000402c: ee09407c 18000000 00000033
	s_wait_alu depctr_sa_sdst(0)                               // 000000004038: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s19                             // 00000000403c: 8c7e137e
	v_cmp_lt_i64_e64 s18, 1, v[41:42]                          // 000000004040: d4510012 02025281
	s_and_b32 s19, s36, s18                                    // 000000004048: 8b131224
	s_wait_alu depctr_sa_sdst(0)                               // 00000000404c: bf88ff9e
	s_and_saveexec_b32 s20, s19                                // 000000004050: be942013
	s_cbranch_execz 28                                         // 000000004054: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x25c8>
	s_wait_kmcnt 0x0                                           // 000000004058: bfc70000
	v_add_co_u32 v51, s19, s48, v49                            // 00000000405c: d7001333 02026230
	v_bfe_u32 v48, v53, 16, 1                                  // 000000004064: d6100030 02052135
	s_wait_alu depctr_va_sdst(0)                               // 00000000406c: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s49, v50, s19               // 000000004070: d5207c34 004e6431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004078: bf870193
	v_add_co_u32 v51, s19, v51, v65                            // 00000000407c: d7001333 02028333
	v_add3_u32 v48, v48, v53, 0x7fff                           // 000000004084: d6550030 03fe6b30 00007fff
	v_or_b32_e32 v55, 0x400000, v53                            // 000000004090: 386e6aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004098: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v52, v66, s19               // 00000000409c: d5207c34 004e8534
	v_cmp_u_f32_e64 s19, v53, v53                              // 0000000040a4: d4180013 02026b35
	s_wait_alu depctr_va_sdst(0)                               // 0000000040ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040b0: bf870001
	v_cndmask_b32_e64 v48, v48, v55, s19                       // 0000000040b4: d5010030 004e6f30
	global_store_d16_hi_b16 v[51:52], v48, off                 // 0000000040bc: ee09407c 18000000 00000033
	s_or_b32 exec_lo, exec_lo, s20                             // 0000000040c8: 8c7e147e
	v_cmp_lt_i64_e64 s19, 2, v[41:42]                          // 0000000040cc: d4510013 02025282
	s_and_b32 s20, s36, s19                                    // 0000000040d4: 8b141324
	s_delay_alu instid0(salu_cycle_1)                          // 0000000040d8: bf870009
	s_and_saveexec_b32 s22, s20                                // 0000000040dc: be962014
	s_cbranch_execz 27                                         // 0000000040e0: bfa5001b <tessera_rocm_folded_matmul_a629485a5b78541c+0x2650>
	s_wait_kmcnt 0x0                                           // 0000000040e4: bfc70000
	v_add_co_u32 v51, s20, s48, v49                            // 0000000040e8: d7001433 02026230
	v_bfe_u32 v48, v54, 16, 1                                  // 0000000040f0: d6100030 02052136
	v_add_co_ci_u32_e64 v52, null, s49, v50, s20               // 0000000040f8: d5207c34 00526431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004100: bf870193
	v_add_co_u32 v51, s20, v51, v67                            // 000000004104: d7001433 02028733
	v_add3_u32 v48, v48, v54, 0x7fff                           // 00000000410c: d6550030 03fe6d30 00007fff
	v_or_b32_e32 v53, 0x400000, v54                            // 000000004118: 386a6cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004120: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v52, v68, s20               // 000000004124: d5207c34 00528934
	v_cmp_u_f32_e64 s20, v54, v54                              // 00000000412c: d4180014 02026d36
	s_wait_alu depctr_va_sdst(0)                               // 000000004134: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004138: bf870001
	v_cndmask_b32_e64 v48, v48, v53, s20                       // 00000000413c: d5010030 00526b30
	global_store_d16_hi_b16 v[51:52], v48, off                 // 000000004144: ee09407c 18000000 00000033
	s_or_b32 exec_lo, exec_lo, s22                             // 000000004150: 8c7e167e
	v_cmp_lt_i64_e64 s20, 3, v[41:42]                          // 000000004154: d4510014 02025283
	s_and_b32 s22, s36, s20                                    // 00000000415c: 8b161424
	s_delay_alu instid0(salu_cycle_1)                          // 000000004160: bf870009
	s_and_saveexec_b32 s23, s22                                // 000000004164: be972016
	s_cbranch_execz 27                                         // 000000004168: bfa5001b <tessera_rocm_folded_matmul_a629485a5b78541c+0x26d8>
	s_wait_kmcnt 0x0                                           // 00000000416c: bfc70000
	v_add_co_u32 v51, s22, s48, v49                            // 000000004170: d7001633 02026230
	v_bfe_u32 v48, v43, 16, 1                                  // 000000004178: d6100030 0205212b
	v_add_co_ci_u32_e64 v52, null, s49, v50, s22               // 000000004180: d5207c34 005a6431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004188: bf870193
	v_add_co_u32 v51, s22, v51, v69                            // 00000000418c: d7001633 02028b33
	v_add3_u32 v48, v48, v43, 0x7fff                           // 000000004194: d6550030 03fe5730 00007fff
	v_or_b32_e32 v53, 0x400000, v43                            // 0000000041a0: 386a56ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000041a8: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v52, v70, s22               // 0000000041ac: d5207c34 005a8d34
	v_cmp_u_f32_e64 s22, v43, v43                              // 0000000041b4: d4180016 0202572b
	s_wait_alu depctr_va_sdst(0)                               // 0000000041bc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000041c0: bf870001
	v_cndmask_b32_e64 v43, v48, v53, s22                       // 0000000041c4: d501002b 005a6b30
	global_store_d16_hi_b16 v[51:52], v43, off                 // 0000000041cc: ee09407c 15800000 00000033
	s_or_b32 exec_lo, exec_lo, s23                             // 0000000041d8: 8c7e177e
	v_cmp_lt_i64_e64 s22, 4, v[41:42]                          // 0000000041dc: d4510016 02025284
	s_and_b32 s23, s36, s22                                    // 0000000041e4: 8b171624
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041e8: bf88ff9e
	s_and_saveexec_b32 s24, s23                                // 0000000041ec: be982017
	s_cbranch_execz 28                                         // 0000000041f0: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2764>
	s_wait_kmcnt 0x0                                           // 0000000041f4: bfc70000
	v_add_co_u32 v48, s23, s48, v49                            // 0000000041f8: d7001730 02026230
	v_bfe_u32 v43, v44, 16, 1                                  // 000000004200: d610002b 0205212c
	s_wait_alu depctr_va_sdst(0)                               // 000000004208: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s49, v50, s23               // 00000000420c: d5207c34 005e6431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004214: bf870193
	v_add_co_u32 v51, s23, v48, v71                            // 000000004218: d7001733 02028f30
	v_add3_u32 v43, v43, v44, 0x7fff                           // 000000004220: d655002b 03fe592b 00007fff
	v_or_b32_e32 v53, 0x400000, v44                            // 00000000422c: 386a58ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004234: bf88f19f
	v_add_co_ci_u32_e64 v52, null, v52, v72, s23               // 000000004238: d5207c34 005e9134
	v_cmp_u_f32_e64 s23, v44, v44                              // 000000004240: d4180017 0202592c
	s_wait_alu depctr_va_sdst(0)                               // 000000004248: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000424c: bf870001
	v_cndmask_b32_e64 v43, v43, v53, s23                       // 000000004250: d501002b 005e6b2b
	global_store_d16_hi_b16 v[51:52], v43, off                 // 000000004258: ee09407c 15800000 00000033
	s_or_b32 exec_lo, exec_lo, s24                             // 000000004264: 8c7e187e
	v_cmp_lt_i64_e64 s23, 5, v[41:42]                          // 000000004268: d4510017 02025285
	s_and_b32 s24, s36, s23                                    // 000000004270: 8b181724
	s_delay_alu instid0(salu_cycle_1)                          // 000000004274: bf870009
	s_and_saveexec_b32 s25, s24                                // 000000004278: be992018
	s_cbranch_execz 28                                         // 00000000427c: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x27f0>
	v_bfe_u32 v43, v45, 16, 1                                  // 000000004280: d610002b 0205212d
	s_wait_kmcnt 0x0                                           // 000000004288: bfc70000
	v_add_co_u32 v44, s24, s48, v49                            // 00000000428c: d700182c 02026230
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004294: bf870191
	v_add_co_ci_u32_e64 v48, null, s49, v50, s24               // 000000004298: d5207c30 00626431
	v_add3_u32 v51, v43, v45, 0x7fff                           // 0000000042a0: d6550033 03fe5b2b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000042ac: bf870003
	v_add_co_u32 v43, s24, v44, v73                            // 0000000042b0: d700182b 0202932c
	v_or_b32_e32 v52, 0x400000, v45                            // 0000000042b8: 38685aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000042c0: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v48, v74, s24               // 0000000042c4: d5207c2c 00629530
	v_cmp_u_f32_e64 s24, v45, v45                              // 0000000042cc: d4180018 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 0000000042d4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042d8: bf870001
	v_cndmask_b32_e64 v45, v51, v52, s24                       // 0000000042dc: d501002d 00626933
	global_store_d16_hi_b16 v[43:44], v45, off                 // 0000000042e4: ee09407c 16800000 0000002b
	s_or_b32 exec_lo, exec_lo, s25                             // 0000000042f0: 8c7e197e
	v_cmp_lt_i64_e64 s24, 6, v[41:42]                          // 0000000042f4: d4510018 02025286
	s_and_b32 s25, s36, s24                                    // 0000000042fc: 8b191824
	s_wait_alu depctr_sa_sdst(0)                               // 000000004300: bf88ff9e
	s_and_saveexec_b32 s26, s25                                // 000000004304: be9a2019
	s_cbranch_execz 28                                         // 000000004308: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x287c>
	v_bfe_u32 v43, v46, 16, 1                                  // 00000000430c: d610002b 0205212e
	s_wait_kmcnt 0x0                                           // 000000004314: bfc70000
	v_add_co_u32 v44, s25, s48, v49                            // 000000004318: d700192c 02026230
	s_wait_alu depctr_va_sdst(0)                               // 000000004320: bf88f19f
	v_add_co_ci_u32_e64 v45, null, s49, v50, s25               // 000000004324: d5207c2d 00666431
	v_add3_u32 v48, v43, v46, 0x7fff                           // 00000000432c: d6550030 03fe5d2b 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004338: bf870003
	v_add_co_u32 v43, s25, v44, v75                            // 00000000433c: d700192b 0202972c
	v_or_b32_e32 v51, 0x400000, v46                            // 000000004344: 38665cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000434c: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v45, v76, s25               // 000000004350: d5207c2c 0066992d
	v_cmp_u_f32_e64 s25, v46, v46                              // 000000004358: d4180019 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004360: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004364: bf870001
	v_cndmask_b32_e64 v45, v48, v51, s25                       // 000000004368: d501002d 00666730
	global_store_d16_hi_b16 v[43:44], v45, off                 // 000000004370: ee09407c 16800000 0000002b
	s_or_b32 exec_lo, exec_lo, s26                             // 00000000437c: 8c7e1a7e
	v_cmp_lt_i64_e64 s25, 7, v[41:42]                          // 000000004380: d4510019 02025287
	s_and_b32 s26, s36, s25                                    // 000000004388: 8b1a1924
	s_delay_alu instid0(salu_cycle_1)                          // 00000000438c: bf870009
	s_and_saveexec_b32 s28, s26                                // 000000004390: be9c201a
	s_cbranch_execz 28                                         // 000000004394: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2908>
	v_bfe_u32 v41, v47, 16, 1                                  // 000000004398: d6100029 0205212f
	s_wait_kmcnt 0x0                                           // 0000000043a0: bfc70000
	v_add_co_u32 v42, s26, s48, v49                            // 0000000043a4: d7001a2a 02026230
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 0000000043ac: bf870191
	v_add_co_ci_u32_e64 v43, null, s49, v50, s26               // 0000000043b0: d5207c2b 006a6431
	v_add3_u32 v44, v41, v47, 0x7fff                           // 0000000043b8: d655002c 03fe5f29 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000043c4: bf870003
	v_add_co_u32 v41, s26, v42, v77                            // 0000000043c8: d7001a29 02029b2a
	v_or_b32_e32 v45, 0x400000, v47                            // 0000000043d0: 385a5eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000043d8: bf88f19f
	v_add_co_ci_u32_e64 v42, null, v43, v78, s26               // 0000000043dc: d5207c2a 006a9d2b
	v_cmp_u_f32_e64 s26, v47, v47                              // 0000000043e4: d418001a 02025f2f
	s_wait_alu depctr_va_sdst(0)                               // 0000000043ec: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043f0: bf870001
	v_cndmask_b32_e64 v43, v44, v45, s26                       // 0000000043f4: d501002b 006a5b2c
	global_store_d16_hi_b16 v[41:42], v43, off                 // 0000000043fc: ee09407c 15800000 00000029
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004408: 8c7e1c7e
	v_or3_b32 v53, s44, v121, v86                              // 00000000440c: d6580035 055af22c
	v_mov_b32_e32 v52, v102                                    // 000000004414: 7e680366
	v_mov_b32_e32 v54, v102                                    // 000000004418: 7e6c0366
	s_delay_alu instid0(valu_dep_3)                            // 00000000441c: bf870003
	v_or_b32_e32 v51, 7, v53                                   // 000000004420: 38666a87
	v_or_b32_e32 v133, 1, v53                                  // 000000004424: 390a6a81
	v_or_b32_e32 v131, 2, v53                                  // 000000004428: 39066a82
	v_or_b32_e32 v129, 3, v53                                  // 00000000442c: 39026a83
	v_or_b32_e32 v121, 4, v53                                  // 000000004430: 38f26a84
	v_cmp_gt_u64_e64 s26, s[40:41], v[51:52]                   // 000000004434: d45c001a 02026628
	v_or_b32_e32 v123, 5, v53                                  // 00000000443c: 38f66a85
	v_or_b32_e32 v55, 6, v53                                   // 000000004440: 386e6a86
	s_and_b32 s27, s27, s26                                    // 000000004444: 8b1b1a1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004448: bf88ff9e
	s_and_b32 s27, s52, s27                                    // 00000000444c: 8b1b1b34
	s_wait_alu depctr_sa_sdst(0)                               // 000000004450: bf88ff9e
	s_xor_b32 s27, s27, -1                                     // 000000004454: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004458: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 00000000445c: be9c201b
	s_delay_alu instid0(salu_cycle_1)                          // 000000004460: bf870009
	s_xor_b32 s29, exec_lo, s28                                // 000000004464: 8d1d1c7e
	s_cbranch_execz 133                                        // 000000004468: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2b80>
	v_mov_b32_e32 v134, v54                                    // 00000000446c: 7f0c0336
	v_cmp_gt_i64_e64 s27, s[40:41], v[53:54]                   // 000000004470: d454001b 02026a28
	v_mov_b32_e32 v132, v54                                    // 000000004478: 7f080336
	v_mov_b32_e32 v130, v54                                    // 00000000447c: 7f040336
	v_mov_b32_e32 v122, v54                                    // 000000004480: 7ef40336
	v_cmp_gt_i64_e64 s28, s[40:41], v[133:134]                 // 000000004484: d454001c 02030a28
	v_mov_b32_e32 v124, v54                                    // 00000000448c: 7ef80336
	s_wait_alu depctr_va_sdst(0)                               // 000000004490: bf88f19f
	v_cndmask_b32_e64 v42, 0, v54, s27                         // 000000004494: d501002a 006e6c80
	v_cndmask_b32_e64 v41, 0, v53, s27                         // 00000000449c: d5010029 006e6a80
	v_cmp_gt_i64_e64 s27, s[40:41], v[131:132]                 // 0000000044a4: d454001b 02030628
	v_mov_b32_e32 v56, v54                                     // 0000000044ac: 7e700336
	v_cndmask_b32_e64 v43, 0, v133, s28                        // 0000000044b0: d501002b 00730a80
	v_cndmask_b32_e64 v44, 0, v54, s28                         // 0000000044b8: d501002c 00726c80
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]                    // 0000000044c0: 3e525282
	v_cmp_gt_i64_e64 s28, s[40:41], v[129:130]                 // 0000000044c4: d454001c 02030228
	s_wait_alu depctr_va_sdst(0)                               // 0000000044cc: bf88f19f
	v_cndmask_b32_e64 v45, 0, v131, s27                        // 0000000044d0: d501002d 006f0680
	v_cndmask_b32_e64 v46, 0, v54, s27                         // 0000000044d8: d501002e 006e6c80
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]                    // 0000000044e0: 3e565682
	v_add_co_u32 v41, s27, s46, v41                            // 0000000044e4: d7001b29 0202522e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 0000000044ec: bf870233
	v_lshlrev_b64_e32 v[45:46], 2, v[45:46]                    // 0000000044f0: 3e5a5a82
	s_wait_alu depctr_va_sdst(0)                               // 0000000044f4: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s47, v42, s27               // 0000000044f8: d5207c2a 006e542f
	v_add_co_u32 v43, s27, s46, v43                            // 000000004500: d7001b2b 0202562e
	v_cndmask_b32_e64 v47, 0, v129, s28                        // 000000004508: d501002f 00730280
	v_cndmask_b32_e64 v48, 0, v54, s28                         // 000000004510: d5010030 00726c80
	s_wait_alu depctr_va_sdst(0)                               // 000000004518: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s47, v44, s27               // 00000000451c: d5207c2c 006e582f
	v_cmp_gt_i64_e64 s27, s[40:41], v[121:122]                 // 000000004524: d454001b 0202f228
	v_add_co_u32 v143, s28, s46, v45                           // 00000000452c: d7001c8f 02025a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004534: bf88f19f
	v_add_co_ci_u32_e64 v144, null, s47, v46, s28              // 000000004538: d5207c90 00725c2f
	v_cmp_gt_i64_e64 s28, s[40:41], v[123:124]                 // 000000004540: d454001c 0202f628
	v_lshlrev_b64_e32 v[45:46], 2, v[47:48]                    // 000000004548: 3e5a5e82
	v_cndmask_b32_e64 v47, 0, v121, s27                        // 00000000454c: d501002f 006ef280
	v_cndmask_b32_e64 v48, 0, v54, s27                         // 000000004554: d5010030 006e6c80
	v_cmp_gt_i64_e64 s27, s[40:41], v[55:56]                   // 00000000455c: d454001b 02026e28
	s_wait_alu depctr_va_sdst(0)                               // 000000004564: bf88f19f
	v_cndmask_b32_e64 v145, 0, v123, s28                       // 000000004568: d5010091 0072f680
	v_cndmask_b32_e64 v146, 0, v54, s28                        // 000000004570: d5010092 00726c80
	v_add_co_u32 v147, s28, s46, v45                           // 000000004578: d7001c93 02025a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004580: bf88f19f
	v_add_co_ci_u32_e64 v148, null, s47, v46, s28              // 000000004584: d5207c94 00725c2f
	v_lshlrev_b64_e32 v[45:46], 2, v[47:48]                    // 00000000458c: 3e5a5e82
	v_lshlrev_b64_e32 v[47:48], 2, v[145:146]                  // 000000004590: 3e5f2282
	v_cndmask_b32_e64 v145, 0, v55, s27                        // 000000004594: d5010091 006e6e80
	v_cndmask_b32_e64 v146, 0, v54, s27                        // 00000000459c: d5010092 006e6c80
	v_cmp_gt_i64_e64 s27, s[40:41], v[51:52]                   // 0000000045a4: d454001b 02026628
	v_add_co_u32 v149, s28, s46, v45                           // 0000000045ac: d7001c95 02025a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b4: bf88f19f
	v_add_co_ci_u32_e64 v150, null, s47, v46, s28              // 0000000045b8: d5207c96 00725c2f
	v_lshlrev_b64_e32 v[45:46], 2, v[145:146]                  // 0000000045c0: 3e5b2282
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 0000000045c4: bf8701d4
	v_cndmask_b32_e64 v146, 0, v52, s27                        // 0000000045c8: d5010092 006e6880
	v_cndmask_b32_e64 v145, 0, v51, s27                        // 0000000045d0: d5010091 006e6680
	v_add_co_u32 v151, s27, s46, v47                           // 0000000045d8: d7001b97 02025e2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000045e0: bf88f19f
	v_add_co_ci_u32_e64 v152, null, s47, v48, s27              // 0000000045e4: d5207c98 006e602f
	v_lshlrev_b64_e32 v[47:48], 2, v[145:146]                  // 0000000045ec: 3e5f2282
	v_add_co_u32 v145, s27, s46, v45                           // 0000000045f0: d7001b91 02025a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f8: bf88f19f
	v_add_co_ci_u32_e64 v146, null, s47, v46, s27              // 0000000045fc: d5207c92 006e5c2f
	s_delay_alu instid0(valu_dep_3)                            // 000000004604: bf870003
	v_add_co_u32 v153, s27, s46, v47                           // 000000004608: d7001b99 02025e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004610: bf88f19f
	v_add_co_ci_u32_e64 v154, null, s47, v48, s27              // 000000004614: d5207c9a 006e602f
	s_clause 0x7                                               // 00000000461c: bf850007
	global_load_b32 v45, v[41:42], off                         // 000000004620: ee05007c 0000002d 00000029
	global_load_b32 v46, v[43:44], off                         // 00000000462c: ee05007c 0000002e 0000002b
	global_load_b32 v47, v[143:144], off                       // 000000004638: ee05007c 0000002f 0000008f
	global_load_b32 v48, v[147:148], off                       // 000000004644: ee05007c 00000030 00000093
	global_load_b32 v41, v[149:150], off                       // 000000004650: ee05007c 00000029 00000095
	global_load_b32 v42, v[151:152], off                       // 00000000465c: ee05007c 0000002a 00000097
	global_load_b32 v43, v[145:146], off                       // 000000004668: ee05007c 0000002b 00000091
	global_load_b32 v44, v[153:154], off                       // 000000004674: ee05007c 0000002c 00000099
	s_and_not1_saveexec_b32 s28, s29                           // 000000004680: be9c301d
	s_cbranch_execz 28                                         // 000000004684: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2bf8>
	s_wait_loadcnt 0x3                                         // 000000004688: bfc00003
	v_add_co_u32 v41, s27, s44, v85                            // 00000000468c: d7001b29 0202aa2c
	s_wait_loadcnt 0x2                                         // 000000004694: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000004698: bf88f19f
	v_add_co_ci_u32_e64 v42, null, s45, 0, s27                 // 00000000469c: d5207c2a 006d002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000046a4: bf870122
	v_add_co_u32 v41, s27, v41, v86                            // 0000000046a8: d7001b29 0202ad29
	s_wait_alu depctr_va_sdst(0)                               // 0000000046b0: bf88f19f
	v_add_co_ci_u32_e64 v42, null, 0, v42, s27                 // 0000000046b4: d5207c2a 006e5480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000046bc: bf870091
	v_lshlrev_b64_e32 v[41:42], 2, v[41:42]                    // 0000000046c0: 3e525282
	v_add_co_u32 v41, s27, s46, v41                            // 0000000046c4: d7001b29 0202522e
	s_wait_alu depctr_va_sdst(0)                               // 0000000046cc: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000046d0: bf870002
	v_add_co_ci_u32_e64 v42, null, s47, v42, s27               // 0000000046d4: d5207c2a 006e542f
	global_load_b128 v[45:48], v[41:42], off offset:192        // 0000000046dc: ee05c07c 0000002d 0000c029
	s_wait_loadcnt 0x1                                         // 0000000046e8: bfc00001
	global_load_b128 v[41:44], v[41:42], off offset:208        // 0000000046ec: ee05c07c 00000029 0000d029
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 0000000046fc: 8c7e1c7e
	global_load_u8 v56, v[139:140], off                        // 000000004700: ee04007c 00000038 0000008b
	s_wait_loadcnt 0x0                                         // 00000000470c: bfc00000
	v_lshlrev_b32_e32 v64, 23, v56                             // 000000004710: 30807097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004714: bf870091
	v_mul_f32_e32 v56, v45, v64                                // 000000004718: 1070812d
	v_cmp_class_f32_e64 s27, v56, 0x198                        // 00000000471c: d47e001b 0201ff38 00000198
	v_mul_f32_e32 v56, v33, v56                                // 000000004728: 10707121
	s_xor_b32 s27, s27, -1                                     // 00000000472c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004730: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004734: be9c201b
	s_cbranch_execnz 3027                                      // 000000004738: bfa60bd3 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5b88>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000473c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004740: 8c7e1c7e
	v_mul_f32_e32 v33, v46, v64                                // 000000004744: 1042812e
	s_delay_alu instid0(valu_dep_1)                            // 000000004748: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 00000000474c: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v45, v34, v33                                // 000000004758: 105a4322
	s_xor_b32 s27, s27, -1                                     // 00000000475c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004760: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004764: be9c201b
	s_cbranch_execnz 3033                                      // 000000004768: bfa60bd9 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5bd0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000476c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004770: 8c7e1c7e
	v_mul_f32_e32 v33, v47, v64                                // 000000004774: 1042812f
	s_delay_alu instid0(valu_dep_1)                            // 000000004778: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 00000000477c: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v46, v35, v33                                // 000000004788: 105c4323
	s_xor_b32 s27, s27, -1                                     // 00000000478c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004790: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004794: be9c201b
	s_cbranch_execnz 3039                                      // 000000004798: bfa60bdf <tessera_rocm_folded_matmul_a629485a5b78541c+0x5c18>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000479c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 0000000047a0: 8c7e1c7e
	v_mul_f32_e32 v33, v48, v64                                // 0000000047a4: 10428130
	s_delay_alu instid0(valu_dep_1)                            // 0000000047a8: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 0000000047ac: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v35, v36, v33                                // 0000000047b8: 10464324
	s_xor_b32 s27, s27, -1                                     // 0000000047bc: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047c0: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 0000000047c4: be9c201b
	s_cbranch_execnz 3045                                      // 0000000047c8: bfa60be5 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5c60>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 0000000047d0: 8c7e1c7e
	v_mul_f32_e32 v33, v41, v64                                // 0000000047d4: 10428129
	s_delay_alu instid0(valu_dep_1)                            // 0000000047d8: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 0000000047dc: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v36, v37, v33                                // 0000000047e8: 10484325
	s_xor_b32 s27, s27, -1                                     // 0000000047ec: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047f0: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 0000000047f4: be9c201b
	s_cbranch_execnz 3051                                      // 0000000047f8: bfa60beb <tessera_rocm_folded_matmul_a629485a5b78541c+0x5ca8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004800: 8c7e1c7e
	v_mul_f32_e32 v33, v42, v64                                // 000000004804: 1042812a
	s_delay_alu instid0(valu_dep_1)                            // 000000004808: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 00000000480c: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v37, v38, v33                                // 000000004818: 104a4326
	s_xor_b32 s27, s27, -1                                     // 00000000481c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004820: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004824: be9c201b
	s_cbranch_execnz 3057                                      // 000000004828: bfa60bf1 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5cf0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000482c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004830: 8c7e1c7e
	v_mul_f32_e32 v33, v43, v64                                // 000000004834: 1042812b
	s_delay_alu instid0(valu_dep_1)                            // 000000004838: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 00000000483c: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v38, v39, v33                                // 000000004848: 104c4327
	s_xor_b32 s27, s27, -1                                     // 00000000484c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004850: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004854: be9c201b
	s_cbranch_execnz 3063                                      // 000000004858: bfa60bf7 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5d38>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000485c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004860: 8c7e1c7e
	v_mul_f32_e32 v33, v44, v64                                // 000000004864: 1042812c
	s_delay_alu instid0(valu_dep_1)                            // 000000004868: bf870001
	v_cmp_class_f32_e64 s27, v33, 0x198                        // 00000000486c: d47e001b 0201ff21 00000198
	v_mul_f32_e32 v39, v40, v33                                // 000000004878: 104e4328
	s_xor_b32 s27, s27, -1                                     // 00000000487c: 8d1bc11b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004880: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 000000004884: be9c201b
	s_cbranch_execnz 3069                                      // 000000004888: bfa60bfd <tessera_rocm_folded_matmul_a629485a5b78541c+0x5d80>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000488c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004890: 8c7e1c7e
	v_mul_lo_u32 v42, v54, s42                                 // 000000004894: d72c002a 02005536
	v_mul_lo_u32 v43, v53, s43                                 // 00000000489c: d72c002b 02005735
	v_mad_co_u64_u32 v[40:41], null, v53, s42, 0               // 0000000048a4: d6fe7c28 02005535
	v_sub_co_u32 v33, s27, s40, v53                            // 0000000048ac: d7011b21 02026a28
	s_wait_alu depctr_va_sdst(0)                               // 0000000048b4: bf88f19f
	v_sub_co_ci_u32_e64 v34, null, s41, v54, s27               // 0000000048b8: d5217c22 006e6c29
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000048c0: bf870211
	v_cmp_lt_i64_e64 s30, 0, v[33:34]                          // 0000000048c4: d451001e 02024280
	v_add3_u32 v41, v41, v43, v42                              // 0000000048cc: d6550029 04aa5729
	s_delay_alu instid0(valu_dep_1)                            // 0000000048d4: bf870001
	v_lshlrev_b64_e32 v[41:42], 1, v[40:41]                    // 0000000048d8: 3e525081
	s_and_b32 s27, s36, s30                                    // 0000000048dc: 8b1b1e24
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048e0: bf88ff9e
	s_and_saveexec_b32 s28, s27                                // 0000000048e4: be9c201b
	s_cbranch_execz 28                                         // 0000000048e8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2e5c>
	s_wait_kmcnt 0x0                                           // 0000000048ec: bfc70000
	v_add_co_u32 v43, s27, s48, v41                            // 0000000048f0: d7001b2b 02025230
	v_bfe_u32 v40, v56, 16, 1                                  // 0000000048f8: d6100028 02052138
	s_wait_alu depctr_va_sdst(0)                               // 000000004900: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s49, v42, s27               // 000000004904: d5207c2c 006e5431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000490c: bf870193
	v_add_co_u32 v43, s27, v43, v137                           // 000000004910: d7001b2b 0203132b
	v_add3_u32 v40, v40, v56, 0x7fff                           // 000000004918: d6550028 03fe7128 00007fff
	v_or_b32_e32 v47, 0x400000, v56                            // 000000004924: 385e70ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000492c: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v44, v138, s27              // 000000004930: d5207c2c 006f152c
	v_cmp_u_f32_e64 s27, v56, v56                              // 000000004938: d418001b 02027138
	s_wait_alu depctr_va_sdst(0)                               // 000000004940: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004944: bf870001
	v_cndmask_b32_e64 v40, v40, v47, s27                       // 000000004948: d5010028 006e5f28
	global_store_d16_hi_b16 v[43:44], v40, off                 // 000000004950: ee09407c 14000000 0000002b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000495c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s28                             // 000000004960: 8c7e1c7e
	v_cmp_lt_i64_e64 s27, 1, v[33:34]                          // 000000004964: d451001b 02024281
	s_and_b32 s28, s36, s27                                    // 00000000496c: 8b1c1b24
	s_wait_alu depctr_sa_sdst(0)                               // 000000004970: bf88ff9e
	s_and_saveexec_b32 s29, s28                                // 000000004974: be9d201c
	s_cbranch_execz 28                                         // 000000004978: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2eec>
	s_wait_kmcnt 0x0                                           // 00000000497c: bfc70000
	v_add_co_u32 v43, s28, s48, v41                            // 000000004980: d7001c2b 02025230
	v_bfe_u32 v40, v45, 16, 1                                  // 000000004988: d6100028 0205212d
	s_wait_alu depctr_va_sdst(0)                               // 000000004990: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s49, v42, s28               // 000000004994: d5207c2c 00725431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000499c: bf870193
	v_add_co_u32 v43, s28, v43, v65                            // 0000000049a0: d7001c2b 0202832b
	v_add3_u32 v40, v40, v45, 0x7fff                           // 0000000049a8: d6550028 03fe5b28 00007fff
	v_or_b32_e32 v47, 0x400000, v45                            // 0000000049b4: 385e5aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000049bc: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v44, v66, s28               // 0000000049c0: d5207c2c 0072852c
	v_cmp_u_f32_e64 s28, v45, v45                              // 0000000049c8: d418001c 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 0000000049d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000049d4: bf870001
	v_cndmask_b32_e64 v40, v40, v47, s28                       // 0000000049d8: d5010028 00725f28
	global_store_d16_hi_b16 v[43:44], v40, off                 // 0000000049e0: ee09407c 14000000 0000002b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s29                             // 0000000049f0: 8c7e1d7e
	v_cmp_lt_i64_e64 s28, 2, v[33:34]                          // 0000000049f4: d451001c 02024282
	s_and_b32 s29, s36, s28                                    // 0000000049fc: 8b1d1c24
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a00: bf88ff9e
	s_and_saveexec_b32 s31, s29                                // 000000004a04: be9f201d
	s_cbranch_execz 28                                         // 000000004a08: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2f7c>
	s_wait_kmcnt 0x0                                           // 000000004a0c: bfc70000
	v_add_co_u32 v43, s29, s48, v41                            // 000000004a10: d7001d2b 02025230
	v_bfe_u32 v40, v46, 16, 1                                  // 000000004a18: d6100028 0205212e
	s_wait_alu depctr_va_sdst(0)                               // 000000004a20: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s49, v42, s29               // 000000004a24: d5207c2c 00765431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a2c: bf870193
	v_add_co_u32 v43, s29, v43, v67                            // 000000004a30: d7001d2b 0202872b
	v_add3_u32 v40, v40, v46, 0x7fff                           // 000000004a38: d6550028 03fe5d28 00007fff
	v_or_b32_e32 v45, 0x400000, v46                            // 000000004a44: 385a5cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004a4c: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v44, v68, s29               // 000000004a50: d5207c2c 0076892c
	v_cmp_u_f32_e64 s29, v46, v46                              // 000000004a58: d418001d 02025d2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004a60: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a64: bf870001
	v_cndmask_b32_e64 v40, v40, v45, s29                       // 000000004a68: d5010028 00765b28
	global_store_d16_hi_b16 v[43:44], v40, off                 // 000000004a70: ee09407c 14000000 0000002b
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004a7c: 8c7e1f7e
	v_cmp_lt_i64_e64 s29, 3, v[33:34]                          // 000000004a80: d451001d 02024283
	s_and_b32 s31, s36, s29                                    // 000000004a88: 8b1f1d24
	s_delay_alu instid0(salu_cycle_1)                          // 000000004a8c: bf870009
	s_and_saveexec_b32 s33, s31                                // 000000004a90: bea1201f
	s_cbranch_execz 27                                         // 000000004a94: bfa5001b <tessera_rocm_folded_matmul_a629485a5b78541c+0x3004>
	s_wait_kmcnt 0x0                                           // 000000004a98: bfc70000
	v_add_co_u32 v43, s31, s48, v41                            // 000000004a9c: d7001f2b 02025230
	v_bfe_u32 v40, v35, 16, 1                                  // 000000004aa4: d6100028 02052123
	v_add_co_ci_u32_e64 v44, null, s49, v42, s31               // 000000004aac: d5207c2c 007e5431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ab4: bf870193
	v_add_co_u32 v43, s31, v43, v69                            // 000000004ab8: d7001f2b 02028b2b
	v_add3_u32 v40, v40, v35, 0x7fff                           // 000000004ac0: d6550028 03fe4728 00007fff
	v_or_b32_e32 v45, 0x400000, v35                            // 000000004acc: 385a46ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004ad4: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v44, v70, s31               // 000000004ad8: d5207c2c 007e8d2c
	v_cmp_u_f32_e64 s31, v35, v35                              // 000000004ae0: d418001f 02024723
	s_wait_alu depctr_va_sdst(0)                               // 000000004ae8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004aec: bf870001
	v_cndmask_b32_e64 v35, v40, v45, s31                       // 000000004af0: d5010023 007e5b28
	global_store_d16_hi_b16 v[43:44], v35, off                 // 000000004af8: ee09407c 11800000 0000002b
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004b04: 8c7e217e
	v_cmp_lt_i64_e64 s31, 4, v[33:34]                          // 000000004b08: d451001f 02024284
	s_and_b32 s33, s36, s31                                    // 000000004b10: 8b211f24
	s_delay_alu instid0(salu_cycle_1)                          // 000000004b14: bf870009
	s_and_saveexec_b32 s34, s33                                // 000000004b18: bea22021
	s_cbranch_execz 27                                         // 000000004b1c: bfa5001b <tessera_rocm_folded_matmul_a629485a5b78541c+0x308c>
	s_wait_kmcnt 0x0                                           // 000000004b20: bfc70000
	v_add_co_u32 v40, s33, s48, v41                            // 000000004b24: d7002128 02025230
	v_bfe_u32 v35, v36, 16, 1                                  // 000000004b2c: d6100023 02052124
	v_add_co_ci_u32_e64 v44, null, s49, v42, s33               // 000000004b34: d5207c2c 00865431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b3c: bf870193
	v_add_co_u32 v43, s33, v40, v71                            // 000000004b40: d700212b 02028f28
	v_add3_u32 v35, v35, v36, 0x7fff                           // 000000004b48: d6550023 03fe4923 00007fff
	v_or_b32_e32 v45, 0x400000, v36                            // 000000004b54: 385a48ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b5c: bf88f19f
	v_add_co_ci_u32_e64 v44, null, v44, v72, s33               // 000000004b60: d5207c2c 0086912c
	v_cmp_u_f32_e64 s33, v36, v36                              // 000000004b68: d4180021 02024924
	s_wait_alu depctr_va_sdst(0)                               // 000000004b70: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b74: bf870001
	v_cndmask_b32_e64 v35, v35, v45, s33                       // 000000004b78: d5010023 00865b23
	global_store_d16_hi_b16 v[43:44], v35, off                 // 000000004b80: ee09407c 11800000 0000002b
	s_or_b32 exec_lo, exec_lo, s34                             // 000000004b8c: 8c7e227e
	v_cmp_lt_i64_e64 s33, 5, v[33:34]                          // 000000004b90: d4510021 02024285
	s_and_b32 s34, s36, s33                                    // 000000004b98: 8b222124
	s_delay_alu instid0(salu_cycle_1)                          // 000000004b9c: bf870009
	s_and_saveexec_b32 s35, s34                                // 000000004ba0: bea32022
	s_cbranch_execz 28                                         // 000000004ba4: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3118>
	v_bfe_u32 v35, v37, 16, 1                                  // 000000004ba8: d6100023 02052125
	s_wait_kmcnt 0x0                                           // 000000004bb0: bfc70000
	v_add_co_u32 v36, s34, s48, v41                            // 000000004bb4: d7002224 02025230
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004bbc: bf870191
	v_add_co_ci_u32_e64 v40, null, s49, v42, s34               // 000000004bc0: d5207c28 008a5431
	v_add3_u32 v43, v35, v37, 0x7fff                           // 000000004bc8: d655002b 03fe4b23 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004bd4: bf870003
	v_add_co_u32 v35, s34, v36, v73                            // 000000004bd8: d7002223 02029324
	v_or_b32_e32 v44, 0x400000, v37                            // 000000004be0: 38584aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004be8: bf88f19f
	v_add_co_ci_u32_e64 v36, null, v40, v74, s34               // 000000004bec: d5207c24 008a9528
	v_cmp_u_f32_e64 s34, v37, v37                              // 000000004bf4: d4180022 02024b25
	s_wait_alu depctr_va_sdst(0)                               // 000000004bfc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c00: bf870001
	v_cndmask_b32_e64 v37, v43, v44, s34                       // 000000004c04: d5010025 008a592b
	global_store_d16_hi_b16 v[35:36], v37, off                 // 000000004c0c: ee09407c 12800000 00000023
	s_or_b32 exec_lo, exec_lo, s35                             // 000000004c18: 8c7e237e
	v_cmp_lt_i64_e64 s34, 6, v[33:34]                          // 000000004c1c: d4510022 02024286
	s_and_b32 s35, s36, s34                                    // 000000004c24: 8b232224
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c28: bf88ff9e
	s_and_saveexec_b32 s37, s35                                // 000000004c2c: bea52023
	s_cbranch_execz 28                                         // 000000004c30: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x31a4>
	v_bfe_u32 v35, v38, 16, 1                                  // 000000004c34: d6100023 02052126
	s_wait_kmcnt 0x0                                           // 000000004c3c: bfc70000
	v_add_co_u32 v36, s35, s48, v41                            // 000000004c40: d7002324 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000004c48: bf88f19f
	v_add_co_ci_u32_e64 v37, null, s49, v42, s35               // 000000004c4c: d5207c25 008e5431
	v_add3_u32 v40, v35, v38, 0x7fff                           // 000000004c54: d6550028 03fe4d23 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004c60: bf870003
	v_add_co_u32 v35, s35, v36, v75                            // 000000004c64: d7002323 02029724
	v_or_b32_e32 v43, 0x400000, v38                            // 000000004c6c: 38564cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004c74: bf88f19f
	v_add_co_ci_u32_e64 v36, null, v37, v76, s35               // 000000004c78: d5207c24 008e9925
	v_cmp_u_f32_e64 s35, v38, v38                              // 000000004c80: d4180023 02024d26
	s_wait_alu depctr_va_sdst(0)                               // 000000004c88: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c8c: bf870001
	v_cndmask_b32_e64 v37, v40, v43, s35                       // 000000004c90: d5010025 008e5728
	global_store_d16_hi_b16 v[35:36], v37, off                 // 000000004c98: ee09407c 12800000 00000023
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ca4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s37                             // 000000004ca8: 8c7e257e
	v_cmp_lt_i64_e64 s35, 7, v[33:34]                          // 000000004cac: d4510023 02024287
	s_and_b32 s36, s36, s35                                    // 000000004cb4: 8b242324
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cb8: bf88ff9e
	s_and_saveexec_b32 s37, s36                                // 000000004cbc: bea52024
	s_cbranch_execz 28                                         // 000000004cc0: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3234>
	v_bfe_u32 v33, v39, 16, 1                                  // 000000004cc4: d6100021 02052127
	s_wait_kmcnt 0x0                                           // 000000004ccc: bfc70000
	v_add_co_u32 v34, s36, s48, v41                            // 000000004cd0: d7002422 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000004cd8: bf88f19f
	v_add_co_ci_u32_e64 v35, null, s49, v42, s36               // 000000004cdc: d5207c23 00925431
	v_add3_u32 v36, v33, v39, 0x7fff                           // 000000004ce4: d6550024 03fe4f21 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004cf0: bf870003
	v_add_co_u32 v33, s36, v34, v77                            // 000000004cf4: d7002421 02029b22
	v_or_b32_e32 v37, 0x400000, v39                            // 000000004cfc: 384a4eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004d04: bf88f19f
	v_add_co_ci_u32_e64 v34, null, v35, v78, s36               // 000000004d08: d5207c22 00929d23
	v_cmp_u_f32_e64 s36, v39, v39                              // 000000004d10: d4180024 02024f27
	s_wait_alu depctr_va_sdst(0)                               // 000000004d18: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d1c: bf870001
	v_cndmask_b32_e64 v35, v36, v37, s36                       // 000000004d20: d5010023 00924b24
	global_store_d16_hi_b16 v[33:34], v35, off                 // 000000004d28: ee09407c 11800000 00000021
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s37                             // 000000004d38: 8c7e257e
	v_mov_b32_e32 v44, s39                                     // 000000004d3c: 7e580227
	v_or_b32_e32 v43, s38, v0                                  // 000000004d40: 38560026
	s_delay_alu instid0(valu_dep_1)                            // 000000004d44: bf870001
	v_cmp_gt_u64_e64 s36, s[42:43], v[43:44]                   // 000000004d48: d45c0024 0202562a
	s_and_b32 s37, s36, vcc_lo                                 // 000000004d50: 8b256a24
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d54: bf88ff9e
	s_and_b32 s37, s52, s37                                    // 000000004d58: 8b252534
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d5c: bf88ff9e
	s_xor_b32 s37, s37, -1                                     // 000000004d60: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d64: bf88ff9e
	s_and_saveexec_b32 s53, s37                                // 000000004d68: beb52025
	s_delay_alu instid0(salu_cycle_1)                          // 000000004d6c: bf870009
	s_xor_b32 s53, exec_lo, s53                                // 000000004d70: 8d35357e
	s_cbranch_execz 125                                        // 000000004d74: bfa5007d <tessera_rocm_folded_matmul_a629485a5b78541c+0x346c>
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[87:88]                // 000000004d78: 7ca8ae28
	v_mov_b32_e32 v98, v88                                     // 000000004d7c: 7ec40358
	v_mov_b32_e32 v96, v88                                     // 000000004d80: 7ec00358
	v_mov_b32_e32 v94, v88                                     // 000000004d84: 7ebc0358
	v_mov_b32_e32 v90, v88                                     // 000000004d88: 7eb40358
	s_wait_alu depctr_va_vcc(0)                                // 000000004d8c: bf88ff9d
	v_dual_mov_b32 v92, v88 :: v_dual_cndmask_b32 v33, 0, v87  // 000000004d90: ca120158 5c20ae80
	v_cmp_gt_i64_e64 s37, s[40:41], v[97:98]                   // 000000004d98: d4540025 0202c228
	v_cndmask_b32_e32 v34, 0, v88, vcc_lo                      // 000000004da0: 0244b080
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[95:96]                // 000000004da4: 7ca8be28
	v_mov_b32_e32 v84, v88                                     // 000000004da8: 7ea80358
	s_wait_alu depctr_va_sdst(0)                               // 000000004dac: bf88f19f
	s_delay_alu instid0(valu_dep_4)                            // 000000004db0: bf870004
	v_cndmask_b32_e64 v35, 0, v97, s37                         // 000000004db4: d5010023 0096c280
	v_cndmask_b32_e64 v36, 0, v88, s37                         // 000000004dbc: d5010024 0096b080
	s_wait_alu depctr_va_vcc(0)                                // 000000004dc4: bf88ff9d
	v_dual_cndmask_b32 v37, 0, v95 :: v_dual_cndmask_b32 v38, 0, v88// 000000004dc8: ca52be80 2526b080
	v_lshlrev_b64_e32 v[33:34], 2, v[33:34]                    // 000000004dd0: 3e424282
	v_cmp_gt_i64_e64 s37, s[40:41], v[93:94]                   // 000000004dd4: d4540025 0202ba28
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]                    // 000000004ddc: 3e464682
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000004de0: bf870214
	v_lshlrev_b64_e32 v[37:38], 2, v[37:38]                    // 000000004de4: 3e4a4a82
	v_add_co_u32 v33, vcc_lo, s46, v33                         // 000000004de8: d7006a21 0202422e
	s_wait_alu depctr_va_vcc(0)                                // 000000004df0: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s47, v34, vcc_lo            // 000000004df4: d5207c22 01aa442f
	s_delay_alu instid0(valu_dep_4)                            // 000000004dfc: bf870004
	v_add_co_u32 v35, vcc_lo, s46, v35                         // 000000004e00: d7006a23 0202462e
	s_wait_alu depctr_va_sdst(0)                               // 000000004e08: bf88f19f
	v_cndmask_b32_e64 v39, 0, v93, s37                         // 000000004e0c: d5010027 0096ba80
	v_cndmask_b32_e64 v40, 0, v88, s37                         // 000000004e14: d5010028 0096b080
	s_wait_alu depctr_va_vcc(0)                                // 000000004e1c: bf88ff9d
	v_add_co_ci_u32_e64 v36, null, s47, v36, vcc_lo            // 000000004e20: d5207c24 01aa482f
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[89:90]                // 000000004e28: 7ca8b228
	v_add_co_u32 v45, s37, s46, v37                            // 000000004e2c: d700252d 02024a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004e34: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s47, v38, s37               // 000000004e38: d5207c2e 00964c2f
	v_cmp_gt_i64_e64 s37, s[40:41], v[91:92]                   // 000000004e40: d4540025 0202b628
	v_lshlrev_b64_e32 v[37:38], 2, v[39:40]                    // 000000004e48: 3e4a4e82
	s_wait_alu depctr_va_vcc(0)                                // 000000004e4c: bf88ff9d
	v_dual_cndmask_b32 v39, 0, v89 :: v_dual_cndmask_b32 v40, 0, v88// 000000004e50: ca52b280 2728b080
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[83:84]                // 000000004e58: 7ca8a628
	s_wait_alu depctr_va_sdst(0)                               // 000000004e5c: bf88f19f
	v_cndmask_b32_e64 v47, 0, v91, s37                         // 000000004e60: d501002f 0096b680
	v_cndmask_b32_e64 v48, 0, v88, s37                         // 000000004e68: d5010030 0096b080
	v_add_co_u32 v89, s37, s46, v37                            // 000000004e70: d7002559 02024a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004e78: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s47, v38, s37               // 000000004e7c: d5207c5a 00964c2f
	v_lshlrev_b64_e32 v[37:38], 2, v[39:40]                    // 000000004e84: 3e4a4e82
	v_lshlrev_b64_e32 v[39:40], 2, v[47:48]                    // 000000004e88: 3e4e5e82
	s_wait_alu depctr_va_vcc(0)                                // 000000004e8c: bf88ff9d
	v_dual_cndmask_b32 v47, 0, v83 :: v_dual_cndmask_b32 v48, 0, v88// 000000004e90: ca52a680 2f30b080
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[81:82]                // 000000004e98: 7ca8a228
	s_delay_alu instid0(valu_dep_4)                            // 000000004e9c: bf870004
	v_add_co_u32 v83, s37, s46, v37                            // 000000004ea0: d7002553 02024a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000004ea8: bf88f19f
	v_add_co_ci_u32_e64 v84, null, s47, v38, s37               // 000000004eac: d5207c54 00964c2f
	v_lshlrev_b64_e32 v[37:38], 2, v[47:48]                    // 000000004eb4: 3e4a5e82
	s_wait_alu depctr_va_vcc(0)                                // 000000004eb8: bf88ff9d
	v_dual_cndmask_b32 v48, 0, v82 :: v_dual_cndmask_b32 v47, 0, v81// 000000004ebc: ca52a480 302ea280
	v_add_co_u32 v81, vcc_lo, s46, v39                         // 000000004ec4: d7006a51 02024e2e
	s_wait_alu depctr_va_vcc(0)                                // 000000004ecc: bf88ff9d
	v_add_co_ci_u32_e64 v82, null, s47, v40, vcc_lo            // 000000004ed0: d5207c52 01aa502f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_3)// 000000004ed8: bf8701c3
	v_lshlrev_b64_e32 v[39:40], 2, v[47:48]                    // 000000004edc: 3e4e5e82
	v_add_co_u32 v47, vcc_lo, s46, v37                         // 000000004ee0: d7006a2f 02024a2e
	s_wait_alu depctr_va_vcc(0)                                // 000000004ee8: bf88ff9d
	v_add_co_ci_u32_e64 v48, null, s47, v38, vcc_lo            // 000000004eec: d5207c30 01aa4c2f
	v_add_co_u32 v87, vcc_lo, s46, v39                         // 000000004ef4: d7006a57 02024e2e
	s_wait_alu depctr_va_vcc(0)                                // 000000004efc: bf88ff9d
	v_add_co_ci_u32_e64 v88, null, s47, v40, vcc_lo            // 000000004f00: d5207c58 01aa502f
	s_clause 0x7                                               // 000000004f08: bf850007
	global_load_b32 v37, v[33:34], off                         // 000000004f0c: ee05007c 00000025 00000021
	global_load_b32 v38, v[35:36], off                         // 000000004f18: ee05007c 00000026 00000023
	global_load_b32 v39, v[45:46], off                         // 000000004f24: ee05007c 00000027 0000002d
	global_load_b32 v40, v[89:90], off                         // 000000004f30: ee05007c 00000028 00000059
	global_load_b32 v33, v[83:84], off                         // 000000004f3c: ee05007c 00000021 00000053
	global_load_b32 v34, v[81:82], off                         // 000000004f48: ee05007c 00000022 00000051
	global_load_b32 v35, v[47:48], off                         // 000000004f54: ee05007c 00000023 0000002f
	global_load_b32 v36, v[87:88], off                         // 000000004f60: ee05007c 00000024 00000057
	s_and_not1_saveexec_b32 s37, s53                           // 000000004f6c: bea53035
	s_cbranch_execz 14                                         // 000000004f70: bfa5000e <tessera_rocm_folded_matmul_a629485a5b78541c+0x34ac>
	s_wait_loadcnt 0x3                                         // 000000004f74: bfc00003
	v_add_co_u32 v33, vcc_lo, s46, v135                        // 000000004f78: d7006a21 02030e2e
	s_wait_loadcnt 0x2                                         // 000000004f80: bfc00002
	s_wait_alu depctr_va_vcc(0)                                // 000000004f84: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s47, v136, vcc_lo           // 000000004f88: d5207c22 01ab102f
	global_load_b128 v[37:40], v[33:34], off                   // 000000004f90: ee05c07c 00000025 00000021
	s_wait_loadcnt 0x1                                         // 000000004f9c: bfc00001
	global_load_b128 v[33:36], v[33:34], off offset:16         // 000000004fa0: ee05c07c 00000021 00001021
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s37                             // 000000004fb0: 8c7e257e
	v_cmp_gt_i64_e32 vcc_lo, s[42:43], v[43:44]                // 000000004fb4: 7ca8562a
	s_wait_alu depctr_va_vcc(0)                                // 000000004fb8: bf88ff9d
	v_dual_cndmask_b32 v0, 0, v44 :: v_dual_cndmask_b32 v43, 0, v43// 000000004fbc: ca525880 002a5680
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000004fc4: bf870121
	v_add_co_u32 v43, s37, s50, v43                            // 000000004fc8: d700252b 02025632
	s_wait_alu depctr_va_sdst(0)                               // 000000004fd0: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s51, v0, s37                // 000000004fd4: d5207c2c 00960033
	global_load_u8 v0, v[43:44], off                           // 000000004fdc: ee04007c 00000000 0000002b
	s_wait_loadcnt 0x0                                         // 000000004fe8: bfc00000
	v_lshlrev_b32_e32 v0, 23, v0                               // 000000004fec: 30000097
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004ff0: bf870091
	v_mul_f32_e32 v45, v37, v0                                 // 000000004ff4: 105a0125
	v_cmp_class_f32_e64 s37, v45, 0x198                        // 000000004ff8: d47e0025 0201ff2d 00000198
	v_mul_f32_e32 v45, v25, v45                                // 000000005004: 105a5b19
	s_xor_b32 s37, s37, -1                                     // 000000005008: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000500c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005010: beaa2025
	s_cbranch_execnz 2604                                      // 000000005014: bfa60a2c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5dc8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005018: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000501c: 8c7e2a7e
	v_mul_f32_e32 v25, v38, v0                                 // 000000005020: 10320126
	s_delay_alu instid0(valu_dep_1)                            // 000000005024: bf870001
	v_cmp_class_f32_e64 s37, v25, 0x198                        // 000000005028: d47e0025 0201ff19 00000198
	v_mul_f32_e32 v25, v26, v25                                // 000000005034: 1032331a
	s_xor_b32 s37, s37, -1                                     // 000000005038: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000503c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005040: beaa2025
	s_cbranch_execnz 2610                                      // 000000005044: bfa60a32 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5e10>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005048: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000504c: 8c7e2a7e
	v_mul_f32_e32 v26, v39, v0                                 // 000000005050: 10340127
	s_delay_alu instid0(valu_dep_1)                            // 000000005054: bf870001
	v_cmp_class_f32_e64 s37, v26, 0x198                        // 000000005058: d47e0025 0201ff1a 00000198
	v_mul_f32_e32 v26, v27, v26                                // 000000005064: 1034351b
	s_xor_b32 s37, s37, -1                                     // 000000005068: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000506c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005070: beaa2025
	s_cbranch_execnz 2616                                      // 000000005074: bfa60a38 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5e58>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005078: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000507c: 8c7e2a7e
	v_mul_f32_e32 v27, v40, v0                                 // 000000005080: 10360128
	s_delay_alu instid0(valu_dep_1)                            // 000000005084: bf870001
	v_cmp_class_f32_e64 s37, v27, 0x198                        // 000000005088: d47e0025 0201ff1b 00000198
	v_mul_f32_e32 v27, v28, v27                                // 000000005094: 1036371c
	s_xor_b32 s37, s37, -1                                     // 000000005098: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000509c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 0000000050a0: beaa2025
	s_cbranch_execnz 2622                                      // 0000000050a4: bfa60a3e <tessera_rocm_folded_matmul_a629485a5b78541c+0x5ea0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 0000000050ac: 8c7e2a7e
	v_mul_f32_e32 v28, v33, v0                                 // 0000000050b0: 10380121
	s_delay_alu instid0(valu_dep_1)                            // 0000000050b4: bf870001
	v_cmp_class_f32_e64 s37, v28, 0x198                        // 0000000050b8: d47e0025 0201ff1c 00000198
	v_mul_f32_e32 v28, v29, v28                                // 0000000050c4: 1038391d
	s_xor_b32 s37, s37, -1                                     // 0000000050c8: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050cc: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 0000000050d0: beaa2025
	s_cbranch_execnz 2628                                      // 0000000050d4: bfa60a44 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5ee8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050d8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 0000000050dc: 8c7e2a7e
	v_mul_f32_e32 v29, v34, v0                                 // 0000000050e0: 103a0122
	s_delay_alu instid0(valu_dep_1)                            // 0000000050e4: bf870001
	v_cmp_class_f32_e64 s37, v29, 0x198                        // 0000000050e8: d47e0025 0201ff1d 00000198
	v_mul_f32_e32 v29, v30, v29                                // 0000000050f4: 103a3b1e
	s_xor_b32 s37, s37, -1                                     // 0000000050f8: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050fc: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005100: beaa2025
	s_cbranch_execnz 2634                                      // 000000005104: bfa60a4a <tessera_rocm_folded_matmul_a629485a5b78541c+0x5f30>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005108: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000510c: 8c7e2a7e
	v_mul_f32_e32 v30, v35, v0                                 // 000000005110: 103c0123
	s_delay_alu instid0(valu_dep_1)                            // 000000005114: bf870001
	v_cmp_class_f32_e64 s37, v30, 0x198                        // 000000005118: d47e0025 0201ff1e 00000198
	v_mul_f32_e32 v30, v31, v30                                // 000000005124: 103c3d1f
	s_xor_b32 s37, s37, -1                                     // 000000005128: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000512c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005130: beaa2025
	s_cbranch_execnz 2640                                      // 000000005134: bfa60a50 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5f78>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005138: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000513c: 8c7e2a7e
	v_mul_f32_e32 v31, v36, v0                                 // 000000005140: 103e0124
	s_delay_alu instid0(valu_dep_1)                            // 000000005144: bf870001
	v_cmp_class_f32_e64 s37, v31, 0x198                        // 000000005148: d47e0025 0201ff1f 00000198
	v_mul_f32_e32 v31, v32, v31                                // 000000005154: 103e3f20
	s_xor_b32 s37, s37, -1                                     // 000000005158: 8d25c125
	s_wait_alu depctr_sa_sdst(0)                               // 00000000515c: bf88ff9e
	s_and_saveexec_b32 s42, s37                                // 000000005160: beaa2025
	s_cbranch_execnz 2646                                      // 000000005164: bfa60a56 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5fc0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005168: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s42                             // 00000000516c: 8c7e2a7e
	v_add_nc_u32_e32 v0, v142, v141                            // 000000005170: 4a011b8e
	s_and_b32 s0, vcc_lo, s0                                   // 000000005174: 8b00006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005178: bf88ff9e
	s_and_saveexec_b32 s37, s0                                 // 00000000517c: bea52000
	s_cbranch_execz 34                                         // 000000005180: bfa50022 <tessera_rocm_folded_matmul_a629485a5b78541c+0x370c>
	v_add_co_u32 v32, s0, v0, s38                              // 000000005184: d7000020 02004d00
	s_wait_alu depctr_va_sdst(0)                               // 00000000518c: bf88f19f
	v_add_co_ci_u32_e64 v33, null, 0, s39, s0                  // 000000005190: d5207c21 00004e80
	s_wait_kmcnt 0x0                                           // 000000005198: bfc70000
	v_add_co_u32 v35, s0, s48, v79                             // 00000000519c: d7000023 02029e30
	v_bfe_u32 v34, v45, 16, 1                                  // 0000000051a4: d6100022 0205212d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 0000000051ac: bf870253
	v_lshlrev_b64_e32 v[32:33], 1, v[32:33]                    // 0000000051b0: 3e404081
	s_wait_alu depctr_va_sdst(0)                               // 0000000051b4: bf88f19f
	v_add_co_ci_u32_e64 v36, null, s49, v80, s0                // 0000000051b8: d5207c24 0002a031
	v_or_b32_e32 v37, 0x400000, v45                            // 0000000051c0: 384a5aff 00400000
	v_add3_u32 v34, v34, v45, 0x7fff                           // 0000000051c8: d6550022 03fe5b22 00007fff
	v_add_co_u32 v32, s0, v35, v32                             // 0000000051d4: d7000020 02024123
	s_wait_alu depctr_va_sdst(0)                               // 0000000051dc: bf88f19f
	v_add_co_ci_u32_e64 v33, null, v36, v33, s0                // 0000000051e0: d5207c21 00024324
	v_cmp_u_f32_e64 s0, v45, v45                               // 0000000051e8: d4180000 02025b2d
	s_wait_alu depctr_va_sdst(0)                               // 0000000051f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000051f4: bf870001
	v_cndmask_b32_e64 v34, v34, v37, s0                        // 0000000051f8: d5010022 00024b22
	global_store_d16_hi_b16 v[32:33], v34, off offset:32       // 000000005200: ee09407c 11000000 00002020
	s_wait_alu depctr_sa_sdst(0)                               // 00000000520c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s37                             // 000000005210: 8c7e257e
	s_and_b32 s0, vcc_lo, s1                                   // 000000005214: 8b00016a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005218: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000521c: be812000
	s_cbranch_execz 28                                         // 000000005220: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3794>
	v_bfe_u32 v32, v25, 16, 1                                  // 000000005224: d6100020 02052119
	s_wait_kmcnt 0x0                                           // 00000000522c: bfc70000
	v_add_co_u32 v33, s0, s48, v79                             // 000000005230: d7000021 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000005238: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s49, v80, s0                // 00000000523c: d5207c22 0002a031
	v_add3_u32 v35, v32, v25, 0x7fff                           // 000000005244: d6550023 03fe3320 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005250: bf870003
	v_add_co_u32 v32, s0, v33, v65                             // 000000005254: d7000020 02028321
	v_or_b32_e32 v36, 0x400000, v25                            // 00000000525c: 384832ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005264: bf88f19f
	v_add_co_ci_u32_e64 v33, null, v34, v66, s0                // 000000005268: d5207c21 00028522
	v_cmp_u_f32_e64 s0, v25, v25                               // 000000005270: d4180000 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000005278: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000527c: bf870001
	v_cndmask_b32_e64 v25, v35, v36, s0                        // 000000005280: d5010019 00024923
	global_store_d16_hi_b16 v[32:33], v25, off offset:32       // 000000005288: ee09407c 0c800000 00002020
	s_wait_alu depctr_sa_sdst(0)                               // 000000005294: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005298: 8c7e017e
	s_and_b32 s0, vcc_lo, s2                                   // 00000000529c: 8b00026a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000052a4: be812000
	s_cbranch_execz 28                                         // 0000000052a8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x381c>
	s_wait_kmcnt 0x0                                           // 0000000052ac: bfc70000
	v_add_co_u32 v32, s0, s48, v79                             // 0000000052b0: d7000020 02029e30
	v_bfe_u32 v25, v26, 16, 1                                  // 0000000052b8: d6100019 0205211a
	s_wait_alu depctr_va_sdst(0)                               // 0000000052c0: bf88f19f
	v_add_co_ci_u32_e64 v33, null, s49, v80, s0                // 0000000052c4: d5207c21 0002a031
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000052cc: bf870193
	v_add_co_u32 v32, s0, v32, v67                             // 0000000052d0: d7000020 02028720
	v_add3_u32 v25, v25, v26, 0x7fff                           // 0000000052d8: d6550019 03fe3519 00007fff
	v_or_b32_e32 v34, 0x400000, v26                            // 0000000052e4: 384434ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000052ec: bf88f19f
	v_add_co_ci_u32_e64 v33, null, v33, v68, s0                // 0000000052f0: d5207c21 00028921
	v_cmp_u_f32_e64 s0, v26, v26                               // 0000000052f8: d4180000 0202351a
	s_wait_alu depctr_va_sdst(0)                               // 000000005300: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005304: bf870001
	v_cndmask_b32_e64 v25, v25, v34, s0                        // 000000005308: d5010019 00024519
	global_store_d16_hi_b16 v[32:33], v25, off offset:32       // 000000005310: ee09407c 0c800000 00002020
	s_wait_alu depctr_sa_sdst(0)                               // 00000000531c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005320: 8c7e017e
	s_and_b32 s0, vcc_lo, s3                                   // 000000005324: 8b00036a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005328: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000532c: be812000
	s_cbranch_execz 28                                         // 000000005330: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x38a4>
	v_bfe_u32 v25, v27, 16, 1                                  // 000000005334: d6100019 0205211b
	s_wait_kmcnt 0x0                                           // 00000000533c: bfc70000
	v_add_co_u32 v26, s0, s48, v79                             // 000000005340: d700001a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000005348: bf88f19f
	v_add_co_ci_u32_e64 v32, null, s49, v80, s0                // 00000000534c: d5207c20 0002a031
	v_add3_u32 v33, v25, v27, 0x7fff                           // 000000005354: d6550021 03fe3719 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005360: bf870003
	v_add_co_u32 v25, s0, v26, v69                             // 000000005364: d7000019 02028b1a
	v_or_b32_e32 v34, 0x400000, v27                            // 00000000536c: 384436ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005374: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v32, v70, s0                // 000000005378: d5207c1a 00028d20
	v_cmp_u_f32_e64 s0, v27, v27                               // 000000005380: d4180000 0202371b
	s_wait_alu depctr_va_sdst(0)                               // 000000005388: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000538c: bf870001
	v_cndmask_b32_e64 v27, v33, v34, s0                        // 000000005390: d501001b 00024521
	global_store_d16_hi_b16 v[25:26], v27, off offset:32       // 000000005398: ee09407c 0d800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000053a8: 8c7e017e
	s_and_b32 s0, vcc_lo, s4                                   // 0000000053ac: 8b00046a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053b0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000053b4: be812000
	s_cbranch_execz 28                                         // 0000000053b8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x392c>
	v_bfe_u32 v25, v28, 16, 1                                  // 0000000053bc: d6100019 0205211c
	s_wait_kmcnt 0x0                                           // 0000000053c4: bfc70000
	v_add_co_u32 v26, s0, s48, v79                             // 0000000053c8: d700001a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 0000000053d0: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s49, v80, s0                // 0000000053d4: d5207c1b 0002a031
	v_add3_u32 v32, v25, v28, 0x7fff                           // 0000000053dc: d6550020 03fe3919 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000053e8: bf870003
	v_add_co_u32 v25, s0, v26, v71                             // 0000000053ec: d7000019 02028f1a
	v_or_b32_e32 v33, 0x400000, v28                            // 0000000053f4: 384238ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000053fc: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v27, v72, s0                // 000000005400: d5207c1a 0002911b
	v_cmp_u_f32_e64 s0, v28, v28                               // 000000005408: d4180000 0202391c
	s_wait_alu depctr_va_sdst(0)                               // 000000005410: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005414: bf870001
	v_cndmask_b32_e64 v27, v32, v33, s0                        // 000000005418: d501001b 00024320
	global_store_d16_hi_b16 v[25:26], v27, off offset:32       // 000000005420: ee09407c 0d800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 00000000542c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005430: 8c7e017e
	s_and_b32 s0, vcc_lo, s5                                   // 000000005434: 8b00056a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005438: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000543c: be812000
	s_cbranch_execz 28                                         // 000000005440: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x39b4>
	v_bfe_u32 v25, v29, 16, 1                                  // 000000005444: d6100019 0205211d
	s_wait_kmcnt 0x0                                           // 00000000544c: bfc70000
	v_add_co_u32 v26, s0, s48, v79                             // 000000005450: d700001a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000005458: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s49, v80, s0                // 00000000545c: d5207c1b 0002a031
	v_add3_u32 v28, v25, v29, 0x7fff                           // 000000005464: d655001c 03fe3b19 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005470: bf870003
	v_add_co_u32 v25, s0, v26, v73                             // 000000005474: d7000019 0202931a
	v_or_b32_e32 v32, 0x400000, v29                            // 00000000547c: 38403aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005484: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v27, v74, s0                // 000000005488: d5207c1a 0002951b
	v_cmp_u_f32_e64 s0, v29, v29                               // 000000005490: d4180000 02023b1d
	s_wait_alu depctr_va_sdst(0)                               // 000000005498: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000549c: bf870001
	v_cndmask_b32_e64 v27, v28, v32, s0                        // 0000000054a0: d501001b 0002411c
	global_store_d16_hi_b16 v[25:26], v27, off offset:32       // 0000000054a8: ee09407c 0d800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000054b8: 8c7e017e
	s_and_b32 s0, vcc_lo, s6                                   // 0000000054bc: 8b00066a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054c0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000054c4: be812000
	s_cbranch_execz 28                                         // 0000000054c8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3a3c>
	v_bfe_u32 v25, v30, 16, 1                                  // 0000000054cc: d6100019 0205211e
	s_wait_kmcnt 0x0                                           // 0000000054d4: bfc70000
	v_add_co_u32 v26, s0, s48, v79                             // 0000000054d8: d700001a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 0000000054e0: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s49, v80, s0                // 0000000054e4: d5207c1b 0002a031
	v_add3_u32 v28, v25, v30, 0x7fff                           // 0000000054ec: d655001c 03fe3d19 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000054f8: bf870003
	v_add_co_u32 v25, s0, v26, v75                             // 0000000054fc: d7000019 0202971a
	v_or_b32_e32 v29, 0x400000, v30                            // 000000005504: 383a3cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000550c: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v27, v76, s0                // 000000005510: d5207c1a 0002991b
	v_cmp_u_f32_e64 s0, v30, v30                               // 000000005518: d4180000 02023d1e
	s_wait_alu depctr_va_sdst(0)                               // 000000005520: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005524: bf870001
	v_cndmask_b32_e64 v27, v28, v29, s0                        // 000000005528: d501001b 00023b1c
	global_store_d16_hi_b16 v[25:26], v27, off offset:32       // 000000005530: ee09407c 0d800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 00000000553c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005540: 8c7e017e
	s_and_b32 s0, vcc_lo, s7                                   // 000000005544: 8b00076a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005548: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000554c: be812000
	s_cbranch_execz 28                                         // 000000005550: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3ac4>
	v_bfe_u32 v25, v31, 16, 1                                  // 000000005554: d6100019 0205211f
	s_wait_kmcnt 0x0                                           // 00000000555c: bfc70000
	v_add_co_u32 v26, s0, s48, v79                             // 000000005560: d700001a 02029e30
	s_wait_alu depctr_va_sdst(0)                               // 000000005568: bf88f19f
	v_add_co_ci_u32_e64 v27, null, s49, v80, s0                // 00000000556c: d5207c1b 0002a031
	v_add3_u32 v28, v25, v31, 0x7fff                           // 000000005574: d655001c 03fe3f19 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005580: bf870003
	v_add_co_u32 v25, s0, v26, v77                             // 000000005584: d7000019 02029b1a
	v_or_b32_e32 v29, 0x400000, v31                            // 00000000558c: 383a3eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005594: bf88f19f
	v_add_co_ci_u32_e64 v26, null, v27, v78, s0                // 000000005598: d5207c1a 00029d1b
	v_cmp_u_f32_e64 s0, v31, v31                               // 0000000055a0: d4180000 02023f1f
	s_wait_alu depctr_va_sdst(0)                               // 0000000055a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000055ac: bf870001
	v_cndmask_b32_e64 v27, v28, v29, s0                        // 0000000055b0: d501001b 00023b1c
	global_store_d16_hi_b16 v[25:26], v27, off offset:32       // 0000000055b8: ee09407c 0d800000 00002019
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000055c8: 8c7e017e
	s_and_b32 s0, s36, s8                                      // 0000000055cc: 8b000824
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055d0: bf88ff9e
	s_and_b32 s0, s52, s0                                      // 0000000055d4: 8b000034
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055d8: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 0000000055dc: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055e0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000055e4: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055e8: bf88ff9e
	s_xor_b32 s2, exec_lo, s1                                  // 0000000055ec: 8d02017e
	s_cbranch_execz 133                                        // 0000000055f0: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3d08>
	v_mov_b32_e32 v118, v102                                   // 0000000055f4: 7eec0366
	v_cmp_gt_i64_e64 s0, s[40:41], v[101:102]                  // 0000000055f8: d4540000 0202ca28
	v_mov_b32_e32 v116, v102                                   // 000000005600: 7ee80366
	v_mov_b32_e32 v110, v102                                   // 000000005604: 7edc0366
	v_mov_b32_e32 v106, v102                                   // 000000005608: 7ed40366
	v_cmp_gt_i64_e64 s1, s[40:41], v[117:118]                  // 00000000560c: d4540001 0202ea28
	v_mov_b32_e32 v108, v102                                   // 000000005614: 7ed80366
	s_wait_alu depctr_va_sdst(0)                               // 000000005618: bf88f19f
	v_cndmask_b32_e64 v26, 0, v102, s0                         // 00000000561c: d501001a 0002cc80
	v_cndmask_b32_e64 v25, 0, v101, s0                         // 000000005624: d5010019 0002ca80
	v_cmp_gt_i64_e64 s0, s[40:41], v[115:116]                  // 00000000562c: d4540000 0202e628
	v_mov_b32_e32 v104, v102                                   // 000000005634: 7ed00366
	v_cndmask_b32_e64 v27, 0, v117, s1                         // 000000005638: d501001b 0006ea80
	v_cndmask_b32_e64 v28, 0, v102, s1                         // 000000005640: d501001c 0006cc80
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]                    // 000000005648: 3e323282
	v_cmp_gt_i64_e64 s1, s[40:41], v[109:110]                  // 00000000564c: d4540001 0202da28
	s_wait_alu depctr_va_sdst(0)                               // 000000005654: bf88f19f
	v_cndmask_b32_e64 v29, 0, v115, s0                         // 000000005658: d501001d 0002e680
	v_cndmask_b32_e64 v30, 0, v102, s0                         // 000000005660: d501001e 0002cc80
	v_lshlrev_b64_e32 v[27:28], 2, v[27:28]                    // 000000005668: 3e363682
	v_add_co_u32 v25, s0, s46, v25                             // 00000000566c: d7000019 0202322e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000005674: bf870233
	v_lshlrev_b64_e32 v[29:30], 2, v[29:30]                    // 000000005678: 3e3a3a82
	s_wait_alu depctr_va_sdst(0)                               // 00000000567c: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s47, v26, s0                // 000000005680: d5207c1a 0002342f
	v_add_co_u32 v27, s0, s46, v27                             // 000000005688: d700001b 0202362e
	v_cndmask_b32_e64 v31, 0, v109, s1                         // 000000005690: d501001f 0006da80
	v_cndmask_b32_e64 v32, 0, v102, s1                         // 000000005698: d5010020 0006cc80
	s_wait_alu depctr_va_sdst(0)                               // 0000000056a0: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s47, v28, s0                // 0000000056a4: d5207c1c 0002382f
	v_cmp_gt_i64_e64 s0, s[40:41], v[105:106]                  // 0000000056ac: d4540000 0202d228
	v_add_co_u32 v33, s1, s46, v29                             // 0000000056b4: d7000121 02023a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000056bc: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s47, v30, s1                // 0000000056c0: d5207c22 00063c2f
	v_cmp_gt_i64_e64 s1, s[40:41], v[107:108]                  // 0000000056c8: d4540001 0202d628
	v_lshlrev_b64_e32 v[29:30], 2, v[31:32]                    // 0000000056d0: 3e3a3e82
	v_cndmask_b32_e64 v31, 0, v105, s0                         // 0000000056d4: d501001f 0002d280
	v_cndmask_b32_e64 v32, 0, v102, s0                         // 0000000056dc: d5010020 0002cc80
	v_cmp_gt_i64_e64 s0, s[40:41], v[103:104]                  // 0000000056e4: d4540000 0202ce28
	s_wait_alu depctr_va_sdst(0)                               // 0000000056ec: bf88f19f
	v_cndmask_b32_e64 v35, 0, v107, s1                         // 0000000056f0: d5010023 0006d680
	v_cndmask_b32_e64 v36, 0, v102, s1                         // 0000000056f8: d5010024 0006cc80
	v_add_co_u32 v37, s1, s46, v29                             // 000000005700: d7000125 02023a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005708: bf88f19f
	v_add_co_ci_u32_e64 v38, null, s47, v30, s1                // 00000000570c: d5207c26 00063c2f
	v_lshlrev_b64_e32 v[29:30], 2, v[31:32]                    // 000000005714: 3e3a3e82
	v_lshlrev_b64_e32 v[31:32], 2, v[35:36]                    // 000000005718: 3e3e4682
	v_cndmask_b32_e64 v35, 0, v103, s0                         // 00000000571c: d5010023 0002ce80
	v_cndmask_b32_e64 v36, 0, v102, s0                         // 000000005724: d5010024 0002cc80
	v_cmp_gt_i64_e64 s0, s[40:41], v[99:100]                   // 00000000572c: d4540000 0202c628
	v_add_co_u32 v39, s1, s46, v29                             // 000000005734: d7000127 02023a2e
	s_wait_alu depctr_va_sdst(0)                               // 00000000573c: bf88f19f
	v_add_co_ci_u32_e64 v40, null, s47, v30, s1                // 000000005740: d5207c28 00063c2f
	v_lshlrev_b64_e32 v[29:30], 2, v[35:36]                    // 000000005748: 3e3a4682
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 00000000574c: bf8701d4
	v_cndmask_b32_e64 v36, 0, v100, s0                         // 000000005750: d5010024 0002c880
	v_cndmask_b32_e64 v35, 0, v99, s0                          // 000000005758: d5010023 0002c680
	v_add_co_u32 v45, s0, s46, v31                             // 000000005760: d700002d 02023e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005768: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s47, v32, s0                // 00000000576c: d5207c2e 0002402f
	v_lshlrev_b64_e32 v[31:32], 2, v[35:36]                    // 000000005774: 3e3e4682
	v_add_co_u32 v35, s0, s46, v29                             // 000000005778: d7000023 02023a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005780: bf88f19f
	v_add_co_ci_u32_e64 v36, null, s47, v30, s0                // 000000005784: d5207c24 00023c2f
	s_delay_alu instid0(valu_dep_3)                            // 00000000578c: bf870003
	v_add_co_u32 v47, s0, s46, v31                             // 000000005790: d700002f 02023e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005798: bf88f19f
	v_add_co_ci_u32_e64 v48, null, s47, v32, s0                // 00000000579c: d5207c30 0002402f
	s_clause 0x7                                               // 0000000057a4: bf850007
	global_load_b32 v29, v[25:26], off                         // 0000000057a8: ee05007c 0000001d 00000019
	global_load_b32 v30, v[27:28], off                         // 0000000057b4: ee05007c 0000001e 0000001b
	global_load_b32 v31, v[33:34], off                         // 0000000057c0: ee05007c 0000001f 00000021
	global_load_b32 v32, v[37:38], off                         // 0000000057cc: ee05007c 00000020 00000025
	global_load_b32 v25, v[39:40], off                         // 0000000057d8: ee05007c 00000019 00000027
	global_load_b32 v26, v[45:46], off                         // 0000000057e4: ee05007c 0000001a 0000002d
	global_load_b32 v27, v[35:36], off                         // 0000000057f0: ee05007c 0000001b 00000023
	global_load_b32 v28, v[47:48], off                         // 0000000057fc: ee05007c 0000001c 0000002f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005808: bf88ff9e
	s_and_not1_saveexec_b32 s1, s2                             // 00000000580c: be813002
	s_cbranch_execz 28                                         // 000000005810: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3d84>
	s_wait_loadcnt 0x3                                         // 000000005814: bfc00003
	v_add_co_u32 v25, s0, s44, v85                             // 000000005818: d7000019 0202aa2c
	s_wait_loadcnt 0x2                                         // 000000005820: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 000000005824: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s45, 0, s0                  // 000000005828: d5207c1a 0001002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000005830: bf870122
	v_add_co_u32 v25, s0, v25, v86                             // 000000005834: d7000019 0202ad19
	s_wait_alu depctr_va_sdst(0)                               // 00000000583c: bf88f19f
	v_add_co_ci_u32_e64 v26, null, 0, v26, s0                  // 000000005840: d5207c1a 00023480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005848: bf870091
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]                    // 00000000584c: 3e323282
	v_add_co_u32 v25, s0, s46, v25                             // 000000005850: d7000019 0202322e
	s_wait_alu depctr_va_sdst(0)                               // 000000005858: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000585c: bf870002
	v_add_co_ci_u32_e64 v26, null, s47, v26, s0                // 000000005860: d5207c1a 0002342f
	global_load_b128 v[29:32], v[25:26], off offset:64         // 000000005868: ee05c07c 0000001d 00004019
	s_wait_loadcnt 0x1                                         // 000000005874: bfc00001
	global_load_b128 v[25:28], v[25:26], off offset:80         // 000000005878: ee05c07c 00000019 00005019
	s_wait_alu depctr_sa_sdst(0)                               // 000000005884: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005888: 8c7e017e
	global_load_u8 v33, v[43:44], off                          // 00000000588c: ee04007c 00000021 0000002b
	s_wait_loadcnt 0x0                                         // 000000005898: bfc00000
	v_lshlrev_b32_e32 v34, 23, v33                             // 00000000589c: 30444297
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000058a0: bf870091
	v_mul_f32_e32 v33, v29, v34                                // 0000000058a4: 1042451d
	v_cmp_class_f32_e64 s0, v33, 0x198                         // 0000000058a8: d47e0000 0201ff21 00000198
	v_mul_f32_e32 v33, v17, v33                                // 0000000058b4: 10424311
	s_xor_b32 s0, s0, -1                                       // 0000000058b8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058bc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000058c0: be812000
	s_cbranch_execnz 2192                                      // 0000000058c4: bfa60890 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6008>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000058cc: 8c7e017e
	v_mul_f32_e32 v17, v30, v34                                // 0000000058d0: 1022451e
	s_delay_alu instid0(valu_dep_1)                            // 0000000058d4: bf870001
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 0000000058d8: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v18, v17                                // 0000000058e4: 10222312
	s_xor_b32 s0, s0, -1                                       // 0000000058e8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058ec: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000058f0: be812000
	s_cbranch_execnz 2198                                      // 0000000058f4: bfa60896 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6050>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000058fc: 8c7e017e
	v_mul_f32_e32 v18, v31, v34                                // 000000005900: 1024451f
	s_delay_alu instid0(valu_dep_1)                            // 000000005904: bf870001
	v_cmp_class_f32_e64 s0, v18, 0x198                         // 000000005908: d47e0000 0201ff12 00000198
	v_mul_f32_e32 v18, v19, v18                                // 000000005914: 10242513
	s_xor_b32 s0, s0, -1                                       // 000000005918: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000591c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005920: be812000
	s_cbranch_execnz 2204                                      // 000000005924: bfa6089c <tessera_rocm_folded_matmul_a629485a5b78541c+0x6098>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005928: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000592c: 8c7e017e
	v_mul_f32_e32 v19, v32, v34                                // 000000005930: 10264520
	s_delay_alu instid0(valu_dep_1)                            // 000000005934: bf870001
	v_cmp_class_f32_e64 s0, v19, 0x198                         // 000000005938: d47e0000 0201ff13 00000198
	v_mul_f32_e32 v19, v20, v19                                // 000000005944: 10262714
	s_xor_b32 s0, s0, -1                                       // 000000005948: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000594c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005950: be812000
	s_cbranch_execnz 2210                                      // 000000005954: bfa608a2 <tessera_rocm_folded_matmul_a629485a5b78541c+0x60e0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005958: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000595c: 8c7e017e
	v_mul_f32_e32 v20, v25, v34                                // 000000005960: 10284519
	s_delay_alu instid0(valu_dep_1)                            // 000000005964: bf870001
	v_cmp_class_f32_e64 s0, v20, 0x198                         // 000000005968: d47e0000 0201ff14 00000198
	v_mul_f32_e32 v20, v21, v20                                // 000000005974: 10282915
	s_xor_b32 s0, s0, -1                                       // 000000005978: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000597c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005980: be812000
	s_cbranch_execnz 2216                                      // 000000005984: bfa608a8 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6128>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005988: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000598c: 8c7e017e
	v_mul_f32_e32 v21, v26, v34                                // 000000005990: 102a451a
	s_delay_alu instid0(valu_dep_1)                            // 000000005994: bf870001
	v_cmp_class_f32_e64 s0, v21, 0x198                         // 000000005998: d47e0000 0201ff15 00000198
	v_mul_f32_e32 v21, v22, v21                                // 0000000059a4: 102a2b16
	s_xor_b32 s0, s0, -1                                       // 0000000059a8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059ac: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000059b0: be812000
	s_cbranch_execnz 2222                                      // 0000000059b4: bfa608ae <tessera_rocm_folded_matmul_a629485a5b78541c+0x6170>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000059bc: 8c7e017e
	v_mul_f32_e32 v22, v27, v34                                // 0000000059c0: 102c451b
	s_delay_alu instid0(valu_dep_1)                            // 0000000059c4: bf870001
	v_cmp_class_f32_e64 s0, v22, 0x198                         // 0000000059c8: d47e0000 0201ff16 00000198
	v_mul_f32_e32 v22, v23, v22                                // 0000000059d4: 102c2d17
	s_xor_b32 s0, s0, -1                                       // 0000000059d8: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059dc: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000059e0: be812000
	s_cbranch_execnz 2228                                      // 0000000059e4: bfa608b4 <tessera_rocm_folded_matmul_a629485a5b78541c+0x61b8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059e8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000059ec: 8c7e017e
	v_mul_f32_e32 v23, v28, v34                                // 0000000059f0: 102e451c
	s_delay_alu instid0(valu_dep_1)                            // 0000000059f4: bf870001
	v_cmp_class_f32_e64 s0, v23, 0x198                         // 0000000059f8: d47e0000 0201ff17 00000198
	v_mul_f32_e32 v23, v24, v23                                // 000000005a04: 102e2f18
	s_xor_b32 s0, s0, -1                                       // 000000005a08: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a0c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a10: be812000
	s_cbranch_execnz 2234                                      // 000000005a14: bfa608ba <tessera_rocm_folded_matmul_a629485a5b78541c+0x6200>
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a18: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005a1c: 8c7e017e
	s_and_b32 s0, vcc_lo, s12                                  // 000000005a20: 8b000c6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a24: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005a28: be812000
	s_cbranch_execz 34                                         // 000000005a2c: bfa50022 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3fb8>
	v_add_co_u32 v24, s0, v0, s38                              // 000000005a30: d7000018 02004d00
	s_wait_alu depctr_va_sdst(0)                               // 000000005a38: bf88f19f
	v_add_co_ci_u32_e64 v25, null, 0, s39, s0                  // 000000005a3c: d5207c19 00004e80
	s_wait_kmcnt 0x0                                           // 000000005a44: bfc70000
	v_add_co_u32 v27, s0, s48, v57                             // 000000005a48: d700001b 02027230
	v_bfe_u32 v26, v33, 16, 1                                  // 000000005a50: d610001a 02052121
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000005a58: bf870253
	v_lshlrev_b64_e32 v[24:25], 1, v[24:25]                    // 000000005a5c: 3e303081
	s_wait_alu depctr_va_sdst(0)                               // 000000005a60: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s49, v58, s0                // 000000005a64: d5207c1c 00027431
	v_or_b32_e32 v29, 0x400000, v33                            // 000000005a6c: 383a42ff 00400000
	v_add3_u32 v26, v26, v33, 0x7fff                           // 000000005a74: d655001a 03fe431a 00007fff
	v_add_co_u32 v24, s0, v27, v24                             // 000000005a80: d7000018 0202311b
	s_wait_alu depctr_va_sdst(0)                               // 000000005a88: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v28, v25, s0                // 000000005a8c: d5207c19 0002331c
	v_cmp_u_f32_e64 s0, v33, v33                               // 000000005a94: d4180000 02024321
	s_wait_alu depctr_va_sdst(0)                               // 000000005a9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005aa0: bf870001
	v_cndmask_b32_e64 v26, v26, v29, s0                        // 000000005aa4: d501001a 00023b1a
	global_store_d16_hi_b16 v[24:25], v26, off offset:32       // 000000005aac: ee09407c 0d000000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ab8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005abc: 8c7e017e
	s_and_b32 s0, vcc_lo, s9                                   // 000000005ac0: 8b00096a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ac4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ac8: be812000
	s_cbranch_execz 28                                         // 000000005acc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4040>
	v_bfe_u32 v24, v17, 16, 1                                  // 000000005ad0: d6100018 02052111
	s_wait_kmcnt 0x0                                           // 000000005ad8: bfc70000
	v_add_co_u32 v25, s0, s48, v57                             // 000000005adc: d7000019 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005ae4: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s49, v58, s0                // 000000005ae8: d5207c1a 00027431
	v_add3_u32 v27, v24, v17, 0x7fff                           // 000000005af0: d655001b 03fe2318 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005afc: bf870003
	v_add_co_u32 v24, s0, v25, v65                             // 000000005b00: d7000018 02028319
	v_or_b32_e32 v28, 0x400000, v17                            // 000000005b08: 383822ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005b10: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v26, v66, s0                // 000000005b14: d5207c19 0002851a
	v_cmp_u_f32_e64 s0, v17, v17                               // 000000005b1c: d4180000 02022311
	s_wait_alu depctr_va_sdst(0)                               // 000000005b24: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005b28: bf870001
	v_cndmask_b32_e64 v17, v27, v28, s0                        // 000000005b2c: d5010011 0002391b
	global_store_d16_hi_b16 v[24:25], v17, off offset:32       // 000000005b34: ee09407c 08800000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005b44: 8c7e017e
	s_and_b32 s0, vcc_lo, s10                                  // 000000005b48: 8b000a6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b4c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005b50: be812000
	s_cbranch_execz 28                                         // 000000005b54: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x40c8>
	s_wait_kmcnt 0x0                                           // 000000005b58: bfc70000
	v_add_co_u32 v24, s0, s48, v57                             // 000000005b5c: d7000018 02027230
	v_bfe_u32 v17, v18, 16, 1                                  // 000000005b64: d6100011 02052112
	s_wait_alu depctr_va_sdst(0)                               // 000000005b6c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s49, v58, s0                // 000000005b70: d5207c19 00027431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005b78: bf870193
	v_add_co_u32 v24, s0, v24, v67                             // 000000005b7c: d7000018 02028718
	v_add3_u32 v17, v17, v18, 0x7fff                           // 000000005b84: d6550011 03fe2511 00007fff
	v_or_b32_e32 v26, 0x400000, v18                            // 000000005b90: 383424ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005b98: bf88f19f
	v_add_co_ci_u32_e64 v25, null, v25, v68, s0                // 000000005b9c: d5207c19 00028919
	v_cmp_u_f32_e64 s0, v18, v18                               // 000000005ba4: d4180000 02022512
	s_wait_alu depctr_va_sdst(0)                               // 000000005bac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005bb0: bf870001
	v_cndmask_b32_e64 v17, v17, v26, s0                        // 000000005bb4: d5010011 00023511
	global_store_d16_hi_b16 v[24:25], v17, off offset:32       // 000000005bbc: ee09407c 08800000 00002018
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005bcc: 8c7e017e
	s_and_b32 s0, vcc_lo, s11                                  // 000000005bd0: 8b000b6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bd4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005bd8: be812000
	s_cbranch_execz 28                                         // 000000005bdc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4150>
	v_bfe_u32 v17, v19, 16, 1                                  // 000000005be0: d6100011 02052113
	s_wait_kmcnt 0x0                                           // 000000005be8: bfc70000
	v_add_co_u32 v18, s0, s48, v57                             // 000000005bec: d7000012 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005bf4: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s49, v58, s0                // 000000005bf8: d5207c18 00027431
	v_add3_u32 v25, v17, v19, 0x7fff                           // 000000005c00: d6550019 03fe2711 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005c0c: bf870003
	v_add_co_u32 v17, s0, v18, v69                             // 000000005c10: d7000011 02028b12
	v_or_b32_e32 v26, 0x400000, v19                            // 000000005c18: 383426ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005c20: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v24, v70, s0                // 000000005c24: d5207c12 00028d18
	v_cmp_u_f32_e64 s0, v19, v19                               // 000000005c2c: d4180000 02022713
	s_wait_alu depctr_va_sdst(0)                               // 000000005c34: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005c38: bf870001
	v_cndmask_b32_e64 v19, v25, v26, s0                        // 000000005c3c: d5010013 00023519
	global_store_d16_hi_b16 v[17:18], v19, off offset:32       // 000000005c44: ee09407c 09800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005c54: 8c7e017e
	s_and_b32 s0, vcc_lo, s13                                  // 000000005c58: 8b000d6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c5c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005c60: be812000
	s_cbranch_execz 28                                         // 000000005c64: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x41d8>
	v_bfe_u32 v17, v20, 16, 1                                  // 000000005c68: d6100011 02052114
	s_wait_kmcnt 0x0                                           // 000000005c70: bfc70000
	v_add_co_u32 v18, s0, s48, v57                             // 000000005c74: d7000012 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005c7c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s49, v58, s0                // 000000005c80: d5207c13 00027431
	v_add3_u32 v24, v17, v20, 0x7fff                           // 000000005c88: d6550018 03fe2911 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005c94: bf870003
	v_add_co_u32 v17, s0, v18, v71                             // 000000005c98: d7000011 02028f12
	v_or_b32_e32 v25, 0x400000, v20                            // 000000005ca0: 383228ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005ca8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v19, v72, s0                // 000000005cac: d5207c12 00029113
	v_cmp_u_f32_e64 s0, v20, v20                               // 000000005cb4: d4180000 02022914
	s_wait_alu depctr_va_sdst(0)                               // 000000005cbc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005cc0: bf870001
	v_cndmask_b32_e64 v19, v24, v25, s0                        // 000000005cc4: d5010013 00023318
	global_store_d16_hi_b16 v[17:18], v19, off offset:32       // 000000005ccc: ee09407c 09800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cd8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005cdc: 8c7e017e
	s_and_b32 s0, vcc_lo, s14                                  // 000000005ce0: 8b000e6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ce4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005ce8: be812000
	s_cbranch_execz 28                                         // 000000005cec: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4260>
	v_bfe_u32 v17, v21, 16, 1                                  // 000000005cf0: d6100011 02052115
	s_wait_kmcnt 0x0                                           // 000000005cf8: bfc70000
	v_add_co_u32 v18, s0, s48, v57                             // 000000005cfc: d7000012 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005d04: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s49, v58, s0                // 000000005d08: d5207c13 00027431
	v_add3_u32 v20, v17, v21, 0x7fff                           // 000000005d10: d6550014 03fe2b11 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005d1c: bf870003
	v_add_co_u32 v17, s0, v18, v73                             // 000000005d20: d7000011 02029312
	v_or_b32_e32 v24, 0x400000, v21                            // 000000005d28: 38302aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005d30: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v19, v74, s0                // 000000005d34: d5207c12 00029513
	v_cmp_u_f32_e64 s0, v21, v21                               // 000000005d3c: d4180000 02022b15
	s_wait_alu depctr_va_sdst(0)                               // 000000005d44: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005d48: bf870001
	v_cndmask_b32_e64 v19, v20, v24, s0                        // 000000005d4c: d5010013 00023114
	global_store_d16_hi_b16 v[17:18], v19, off offset:32       // 000000005d54: ee09407c 09800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005d64: 8c7e017e
	s_and_b32 s0, vcc_lo, s15                                  // 000000005d68: 8b000f6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d6c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005d70: be812000
	s_cbranch_execz 28                                         // 000000005d74: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x42e8>
	v_bfe_u32 v17, v22, 16, 1                                  // 000000005d78: d6100011 02052116
	s_wait_kmcnt 0x0                                           // 000000005d80: bfc70000
	v_add_co_u32 v18, s0, s48, v57                             // 000000005d84: d7000012 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005d8c: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s49, v58, s0                // 000000005d90: d5207c13 00027431
	v_add3_u32 v20, v17, v22, 0x7fff                           // 000000005d98: d6550014 03fe2d11 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005da4: bf870003
	v_add_co_u32 v17, s0, v18, v75                             // 000000005da8: d7000011 02029712
	v_or_b32_e32 v21, 0x400000, v22                            // 000000005db0: 382a2cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005db8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v19, v76, s0                // 000000005dbc: d5207c12 00029913
	v_cmp_u_f32_e64 s0, v22, v22                               // 000000005dc4: d4180000 02022d16
	s_wait_alu depctr_va_sdst(0)                               // 000000005dcc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005dd0: bf870001
	v_cndmask_b32_e64 v19, v20, v21, s0                        // 000000005dd4: d5010013 00022b14
	global_store_d16_hi_b16 v[17:18], v19, off offset:32       // 000000005ddc: ee09407c 09800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005de8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005dec: 8c7e017e
	s_and_b32 s0, vcc_lo, s16                                  // 000000005df0: 8b00106a
	s_wait_alu depctr_sa_sdst(0)                               // 000000005df4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005df8: be812000
	s_cbranch_execz 28                                         // 000000005dfc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4370>
	v_bfe_u32 v17, v23, 16, 1                                  // 000000005e00: d6100011 02052117
	s_wait_kmcnt 0x0                                           // 000000005e08: bfc70000
	v_add_co_u32 v18, s0, s48, v57                             // 000000005e0c: d7000012 02027230
	s_wait_alu depctr_va_sdst(0)                               // 000000005e14: bf88f19f
	v_add_co_ci_u32_e64 v19, null, s49, v58, s0                // 000000005e18: d5207c13 00027431
	v_add3_u32 v20, v17, v23, 0x7fff                           // 000000005e20: d6550014 03fe2f11 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000005e2c: bf870003
	v_add_co_u32 v17, s0, v18, v77                             // 000000005e30: d7000011 02029b12
	v_or_b32_e32 v21, 0x400000, v23                            // 000000005e38: 382a2eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000005e40: bf88f19f
	v_add_co_ci_u32_e64 v18, null, v19, v78, s0                // 000000005e44: d5207c12 00029d13
	v_cmp_u_f32_e64 s0, v23, v23                               // 000000005e4c: d4180000 02022f17
	s_wait_alu depctr_va_sdst(0)                               // 000000005e54: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000005e58: bf870001
	v_cndmask_b32_e64 v19, v20, v21, s0                        // 000000005e5c: d5010013 00022b14
	global_store_d16_hi_b16 v[17:18], v19, off offset:32       // 000000005e64: ee09407c 09800000 00002011
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005e74: 8c7e017e
	s_and_b32 s0, s36, s17                                     // 000000005e78: 8b001124
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e7c: bf88ff9e
	s_and_b32 s0, s52, s0                                      // 000000005e80: 8b000034
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e84: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 000000005e88: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e8c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000005e90: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e94: bf88ff9e
	s_xor_b32 s2, exec_lo, s1                                  // 000000005e98: 8d02017e
	s_cbranch_execz 133                                        // 000000005e9c: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x45b4>
	v_mov_b32_e32 v128, v62                                    // 000000005ea0: 7f00033e
	v_cmp_gt_i64_e64 s0, s[40:41], v[61:62]                    // 000000005ea4: d4540000 02027a28
	v_mov_b32_e32 v126, v62                                    // 000000005eac: 7efc033e
	v_mov_b32_e32 v120, v62                                    // 000000005eb0: 7ef0033e
	v_mov_b32_e32 v112, v62                                    // 000000005eb4: 7ee0033e
	v_cmp_gt_i64_e64 s1, s[40:41], v[127:128]                  // 000000005eb8: d4540001 0202fe28
	v_mov_b32_e32 v114, v62                                    // 000000005ec0: 7ee4033e
	s_wait_alu depctr_va_sdst(0)                               // 000000005ec4: bf88f19f
	v_cndmask_b32_e64 v18, 0, v62, s0                          // 000000005ec8: d5010012 00027c80
	v_cndmask_b32_e64 v17, 0, v61, s0                          // 000000005ed0: d5010011 00027a80
	v_cmp_gt_i64_e64 s0, s[40:41], v[125:126]                  // 000000005ed8: d4540000 0202fa28
	v_mov_b32_e32 v64, v62                                     // 000000005ee0: 7e80033e
	v_cndmask_b32_e64 v19, 0, v127, s1                         // 000000005ee4: d5010013 0006fe80
	v_cndmask_b32_e64 v20, 0, v62, s1                          // 000000005eec: d5010014 00067c80
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]                    // 000000005ef4: 3e222282
	v_cmp_gt_i64_e64 s1, s[40:41], v[119:120]                  // 000000005ef8: d4540001 0202ee28
	s_wait_alu depctr_va_sdst(0)                               // 000000005f00: bf88f19f
	v_cndmask_b32_e64 v21, 0, v125, s0                         // 000000005f04: d5010015 0002fa80
	v_cndmask_b32_e64 v22, 0, v62, s0                          // 000000005f0c: d5010016 00027c80
	v_lshlrev_b64_e32 v[19:20], 2, v[19:20]                    // 000000005f14: 3e262682
	v_add_co_u32 v17, s0, s46, v17                             // 000000005f18: d7000011 0202222e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 000000005f20: bf870233
	v_lshlrev_b64_e32 v[21:22], 2, v[21:22]                    // 000000005f24: 3e2a2a82
	s_wait_alu depctr_va_sdst(0)                               // 000000005f28: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s47, v18, s0                // 000000005f2c: d5207c12 0002242f
	v_add_co_u32 v19, s0, s46, v19                             // 000000005f34: d7000013 0202262e
	v_cndmask_b32_e64 v23, 0, v119, s1                         // 000000005f3c: d5010017 0006ee80
	v_cndmask_b32_e64 v24, 0, v62, s1                          // 000000005f44: d5010018 00067c80
	s_wait_alu depctr_va_sdst(0)                               // 000000005f4c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s47, v20, s0                // 000000005f50: d5207c14 0002282f
	v_cmp_gt_i64_e64 s0, s[40:41], v[111:112]                  // 000000005f58: d4540000 0202de28
	v_add_co_u32 v25, s1, s46, v21                             // 000000005f60: d7000119 02022a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005f68: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s47, v22, s1                // 000000005f6c: d5207c1a 00062c2f
	v_cmp_gt_i64_e64 s1, s[40:41], v[113:114]                  // 000000005f74: d4540001 0202e228
	v_lshlrev_b64_e32 v[21:22], 2, v[23:24]                    // 000000005f7c: 3e2a2e82
	v_cndmask_b32_e64 v23, 0, v111, s0                         // 000000005f80: d5010017 0002de80
	v_cndmask_b32_e64 v24, 0, v62, s0                          // 000000005f88: d5010018 00027c80
	v_cmp_gt_i64_e64 s0, s[40:41], v[63:64]                    // 000000005f90: d4540000 02027e28
	s_wait_alu depctr_va_sdst(0)                               // 000000005f98: bf88f19f
	v_cndmask_b32_e64 v27, 0, v113, s1                         // 000000005f9c: d501001b 0006e280
	v_cndmask_b32_e64 v28, 0, v62, s1                          // 000000005fa4: d501001c 00067c80
	v_add_co_u32 v29, s1, s46, v21                             // 000000005fac: d700011d 02022a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005fb4: bf88f19f
	v_add_co_ci_u32_e64 v30, null, s47, v22, s1                // 000000005fb8: d5207c1e 00062c2f
	v_lshlrev_b64_e32 v[21:22], 2, v[23:24]                    // 000000005fc0: 3e2a2e82
	v_lshlrev_b64_e32 v[23:24], 2, v[27:28]                    // 000000005fc4: 3e2e3682
	v_cndmask_b32_e64 v27, 0, v63, s0                          // 000000005fc8: d501001b 00027e80
	v_cndmask_b32_e64 v28, 0, v62, s0                          // 000000005fd0: d501001c 00027c80
	v_cmp_gt_i64_e64 s0, s[40:41], v[59:60]                    // 000000005fd8: d4540000 02027628
	v_add_co_u32 v31, s1, s46, v21                             // 000000005fe0: d700011f 02022a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000005fe8: bf88f19f
	v_add_co_ci_u32_e64 v32, null, s47, v22, s1                // 000000005fec: d5207c20 00062c2f
	v_lshlrev_b64_e32 v[21:22], 2, v[27:28]                    // 000000005ff4: 3e2a3682
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 000000005ff8: bf8701d4
	v_cndmask_b32_e64 v28, 0, v60, s0                          // 000000005ffc: d501001c 00027880
	v_cndmask_b32_e64 v27, 0, v59, s0                          // 000000006004: d501001b 00027680
	v_add_co_u32 v33, s0, s46, v23                             // 00000000600c: d7000021 02022e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000006014: bf88f19f
	v_add_co_ci_u32_e64 v34, null, s47, v24, s0                // 000000006018: d5207c22 0002302f
	v_lshlrev_b64_e32 v[23:24], 2, v[27:28]                    // 000000006020: 3e2e3682
	v_add_co_u32 v27, s0, s46, v21                             // 000000006024: d700001b 02022a2e
	s_wait_alu depctr_va_sdst(0)                               // 00000000602c: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s47, v22, s0                // 000000006030: d5207c1c 00022c2f
	s_delay_alu instid0(valu_dep_3)                            // 000000006038: bf870003
	v_add_co_u32 v35, s0, s46, v23                             // 00000000603c: d7000023 02022e2e
	s_wait_alu depctr_va_sdst(0)                               // 000000006044: bf88f19f
	v_add_co_ci_u32_e64 v36, null, s47, v24, s0                // 000000006048: d5207c24 0002302f
	s_clause 0x7                                               // 000000006050: bf850007
	global_load_b32 v21, v[17:18], off                         // 000000006054: ee05007c 00000015 00000011
	global_load_b32 v22, v[19:20], off                         // 000000006060: ee05007c 00000016 00000013
	global_load_b32 v23, v[25:26], off                         // 00000000606c: ee05007c 00000017 00000019
	global_load_b32 v24, v[29:30], off                         // 000000006078: ee05007c 00000018 0000001d
	global_load_b32 v17, v[31:32], off                         // 000000006084: ee05007c 00000011 0000001f
	global_load_b32 v18, v[33:34], off                         // 000000006090: ee05007c 00000012 00000021
	global_load_b32 v19, v[27:28], off                         // 00000000609c: ee05007c 00000013 0000001b
	global_load_b32 v20, v[35:36], off                         // 0000000060a8: ee05007c 00000014 00000023
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060b4: bf88ff9e
	s_and_not1_saveexec_b32 s1, s2                             // 0000000060b8: be813002
	s_cbranch_execz 28                                         // 0000000060bc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4630>
	s_wait_loadcnt 0x3                                         // 0000000060c0: bfc00003
	v_add_co_u32 v17, s0, s44, v85                             // 0000000060c4: d7000011 0202aa2c
	s_wait_loadcnt 0x2                                         // 0000000060cc: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 0000000060d0: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s45, 0, s0                  // 0000000060d4: d5207c12 0001002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000060dc: bf870122
	v_add_co_u32 v17, s0, v17, v86                             // 0000000060e0: d7000011 0202ad11
	s_wait_alu depctr_va_sdst(0)                               // 0000000060e8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, 0, v18, s0                  // 0000000060ec: d5207c12 00022480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000060f4: bf870091
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]                    // 0000000060f8: 3e222282
	v_add_co_u32 v17, s0, s46, v17                             // 0000000060fc: d7000011 0202222e
	s_wait_alu depctr_va_sdst(0)                               // 000000006104: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000006108: bf870002
	v_add_co_ci_u32_e64 v18, null, s47, v18, s0                // 00000000610c: d5207c12 0002242f
	global_load_b128 v[21:24], v[17:18], off offset:128        // 000000006114: ee05c07c 00000015 00008011
	s_wait_loadcnt 0x1                                         // 000000006120: bfc00001
	global_load_b128 v[17:20], v[17:18], off offset:144        // 000000006124: ee05c07c 00000011 00009011
	s_wait_alu depctr_sa_sdst(0)                               // 000000006130: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006134: 8c7e017e
	global_load_u8 v25, v[43:44], off                          // 000000006138: ee04007c 00000019 0000002b
	s_wait_loadcnt 0x0                                         // 000000006144: bfc00000
	v_lshlrev_b32_e32 v26, 23, v25                             // 000000006148: 30343297
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000614c: bf870091
	v_mul_f32_e32 v25, v21, v26                                // 000000006150: 10323515
	v_cmp_class_f32_e64 s0, v25, 0x198                         // 000000006154: d47e0000 0201ff19 00000198
	v_mul_f32_e32 v25, v9, v25                                 // 000000006160: 10323309
	s_xor_b32 s0, s0, -1                                       // 000000006164: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006168: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000616c: be812000
	s_cbranch_execnz 1781                                      // 000000006170: bfa606f5 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6248>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006174: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006178: 8c7e017e
	v_mul_f32_e32 v9, v22, v26                                 // 00000000617c: 10123516
	s_delay_alu instid0(valu_dep_1)                            // 000000006180: bf870001
	v_cmp_class_f32_e64 s0, v9, 0x198                          // 000000006184: d47e0000 0201ff09 00000198
	v_mul_f32_e32 v9, v10, v9                                  // 000000006190: 1012130a
	s_xor_b32 s0, s0, -1                                       // 000000006194: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006198: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000619c: be812000
	s_cbranch_execnz 1787                                      // 0000000061a0: bfa606fb <tessera_rocm_folded_matmul_a629485a5b78541c+0x6290>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061a4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000061a8: 8c7e017e
	v_mul_f32_e32 v10, v23, v26                                // 0000000061ac: 10143517
	s_delay_alu instid0(valu_dep_1)                            // 0000000061b0: bf870001
	v_cmp_class_f32_e64 s0, v10, 0x198                         // 0000000061b4: d47e0000 0201ff0a 00000198
	v_mul_f32_e32 v10, v11, v10                                // 0000000061c0: 1014150b
	s_xor_b32 s0, s0, -1                                       // 0000000061c4: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061c8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000061cc: be812000
	s_cbranch_execnz 1793                                      // 0000000061d0: bfa60701 <tessera_rocm_folded_matmul_a629485a5b78541c+0x62d8>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000061d8: 8c7e017e
	v_mul_f32_e32 v11, v24, v26                                // 0000000061dc: 10163518
	s_delay_alu instid0(valu_dep_1)                            // 0000000061e0: bf870001
	v_cmp_class_f32_e64 s0, v11, 0x198                         // 0000000061e4: d47e0000 0201ff0b 00000198
	v_mul_f32_e32 v11, v12, v11                                // 0000000061f0: 1016170c
	s_xor_b32 s0, s0, -1                                       // 0000000061f4: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061f8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000061fc: be812000
	s_cbranch_execnz 1799                                      // 000000006200: bfa60707 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6320>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006204: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006208: 8c7e017e
	v_mul_f32_e32 v12, v17, v26                                // 00000000620c: 10183511
	s_delay_alu instid0(valu_dep_1)                            // 000000006210: bf870001
	v_cmp_class_f32_e64 s0, v12, 0x198                         // 000000006214: d47e0000 0201ff0c 00000198
	v_mul_f32_e32 v12, v13, v12                                // 000000006220: 1018190d
	s_xor_b32 s0, s0, -1                                       // 000000006224: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006228: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000622c: be812000
	s_cbranch_execnz 1805                                      // 000000006230: bfa6070d <tessera_rocm_folded_matmul_a629485a5b78541c+0x6368>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006234: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006238: 8c7e017e
	v_mul_f32_e32 v13, v18, v26                                // 00000000623c: 101a3512
	s_delay_alu instid0(valu_dep_1)                            // 000000006240: bf870001
	v_cmp_class_f32_e64 s0, v13, 0x198                         // 000000006244: d47e0000 0201ff0d 00000198
	v_mul_f32_e32 v13, v14, v13                                // 000000006250: 101a1b0e
	s_xor_b32 s0, s0, -1                                       // 000000006254: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006258: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000625c: be812000
	s_cbranch_execnz 1811                                      // 000000006260: bfa60713 <tessera_rocm_folded_matmul_a629485a5b78541c+0x63b0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006264: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006268: 8c7e017e
	v_mul_f32_e32 v14, v19, v26                                // 00000000626c: 101c3513
	s_delay_alu instid0(valu_dep_1)                            // 000000006270: bf870001
	v_cmp_class_f32_e64 s0, v14, 0x198                         // 000000006274: d47e0000 0201ff0e 00000198
	v_mul_f32_e32 v14, v15, v14                                // 000000006280: 101c1d0f
	s_xor_b32 s0, s0, -1                                       // 000000006284: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006288: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000628c: be812000
	s_cbranch_execnz 1817                                      // 000000006290: bfa60719 <tessera_rocm_folded_matmul_a629485a5b78541c+0x63f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006294: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006298: 8c7e017e
	v_mul_f32_e32 v15, v20, v26                                // 00000000629c: 101e3514
	s_delay_alu instid0(valu_dep_1)                            // 0000000062a0: bf870001
	v_cmp_class_f32_e64 s0, v15, 0x198                         // 0000000062a4: d47e0000 0201ff0f 00000198
	v_mul_f32_e32 v15, v16, v15                                // 0000000062b0: 101e1f10
	s_xor_b32 s0, s0, -1                                       // 0000000062b4: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062b8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000062bc: be812000
	s_cbranch_execnz 1823                                      // 0000000062c0: bfa6071f <tessera_rocm_folded_matmul_a629485a5b78541c+0x6440>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000062c8: 8c7e017e
	s_and_b32 s0, vcc_lo, s21                                  // 0000000062cc: 8b00156a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062d0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000062d4: be812000
	s_cbranch_execz 34                                         // 0000000062d8: bfa50022 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4864>
	v_add_co_u32 v16, s0, v0, s38                              // 0000000062dc: d7000010 02004d00
	s_wait_alu depctr_va_sdst(0)                               // 0000000062e4: bf88f19f
	v_add_co_ci_u32_e64 v17, null, 0, s39, s0                  // 0000000062e8: d5207c11 00004e80
	s_wait_kmcnt 0x0                                           // 0000000062f0: bfc70000
	v_add_co_u32 v19, s0, s48, v49                             // 0000000062f4: d7000013 02026230
	v_bfe_u32 v18, v25, 16, 1                                  // 0000000062fc: d6100012 02052119
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000006304: bf870253
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000006308: 3e202081
	s_wait_alu depctr_va_sdst(0)                               // 00000000630c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s49, v50, s0                // 000000006310: d5207c14 00026431
	v_or_b32_e32 v21, 0x400000, v25                            // 000000006318: 382a32ff 00400000
	v_add3_u32 v18, v18, v25, 0x7fff                           // 000000006320: d6550012 03fe3312 00007fff
	v_add_co_u32 v16, s0, v19, v16                             // 00000000632c: d7000010 02022113
	s_wait_alu depctr_va_sdst(0)                               // 000000006334: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v20, v17, s0                // 000000006338: d5207c11 00022314
	v_cmp_u_f32_e64 s0, v25, v25                               // 000000006340: d4180000 02023319
	s_wait_alu depctr_va_sdst(0)                               // 000000006348: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000634c: bf870001
	v_cndmask_b32_e64 v18, v18, v21, s0                        // 000000006350: d5010012 00022b12
	global_store_d16_hi_b16 v[16:17], v18, off offset:32       // 000000006358: ee09407c 09000000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000006364: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006368: 8c7e017e
	s_and_b32 s0, vcc_lo, s18                                  // 00000000636c: 8b00126a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006370: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006374: be812000
	s_cbranch_execz 28                                         // 000000006378: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x48ec>
	v_bfe_u32 v16, v9, 16, 1                                   // 00000000637c: d6100010 02052109
	s_wait_kmcnt 0x0                                           // 000000006384: bfc70000
	v_add_co_u32 v17, s0, s48, v49                             // 000000006388: d7000011 02026230
	s_wait_alu depctr_va_sdst(0)                               // 000000006390: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s49, v50, s0                // 000000006394: d5207c12 00026431
	v_add3_u32 v19, v16, v9, 0x7fff                            // 00000000639c: d6550013 03fe1310 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000063a8: bf870003
	v_add_co_u32 v16, s0, v17, v65                             // 0000000063ac: d7000010 02028311
	v_or_b32_e32 v20, 0x400000, v9                             // 0000000063b4: 382812ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000063bc: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v66, s0                // 0000000063c0: d5207c11 00028512
	v_cmp_u_f32_e64 s0, v9, v9                                 // 0000000063c8: d4180000 02021309
	s_wait_alu depctr_va_sdst(0)                               // 0000000063d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000063d4: bf870001
	v_cndmask_b32_e64 v9, v19, v20, s0                         // 0000000063d8: d5010009 00022913
	global_store_d16_hi_b16 v[16:17], v9, off offset:32        // 0000000063e0: ee09407c 04800000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000063f0: 8c7e017e
	s_and_b32 s0, vcc_lo, s19                                  // 0000000063f4: 8b00136a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000063f8: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000063fc: be812000
	s_cbranch_execz 28                                         // 000000006400: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4974>
	s_wait_kmcnt 0x0                                           // 000000006404: bfc70000
	v_add_co_u32 v16, s0, s48, v49                             // 000000006408: d7000010 02026230
	v_bfe_u32 v9, v10, 16, 1                                   // 000000006410: d6100009 0205210a
	s_wait_alu depctr_va_sdst(0)                               // 000000006418: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s49, v50, s0                // 00000000641c: d5207c11 00026431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006424: bf870193
	v_add_co_u32 v16, s0, v16, v67                             // 000000006428: d7000010 02028710
	v_add3_u32 v9, v9, v10, 0x7fff                             // 000000006430: d6550009 03fe1509 00007fff
	v_or_b32_e32 v18, 0x400000, v10                            // 00000000643c: 382414ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006444: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v17, v68, s0                // 000000006448: d5207c11 00028911
	v_cmp_u_f32_e64 s0, v10, v10                               // 000000006450: d4180000 0202150a
	s_wait_alu depctr_va_sdst(0)                               // 000000006458: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000645c: bf870001
	v_cndmask_b32_e64 v9, v9, v18, s0                          // 000000006460: d5010009 00022509
	global_store_d16_hi_b16 v[16:17], v9, off offset:32        // 000000006468: ee09407c 04800000 00002010
	s_wait_alu depctr_sa_sdst(0)                               // 000000006474: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006478: 8c7e017e
	s_and_b32 s0, vcc_lo, s20                                  // 00000000647c: 8b00146a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006480: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006484: be812000
	s_cbranch_execz 28                                         // 000000006488: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x49fc>
	v_bfe_u32 v9, v11, 16, 1                                   // 00000000648c: d6100009 0205210b
	s_wait_kmcnt 0x0                                           // 000000006494: bfc70000
	v_add_co_u32 v10, s0, s48, v49                             // 000000006498: d700000a 02026230
	s_wait_alu depctr_va_sdst(0)                               // 0000000064a0: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s49, v50, s0                // 0000000064a4: d5207c10 00026431
	v_add3_u32 v17, v9, v11, 0x7fff                            // 0000000064ac: d6550011 03fe1709 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000064b8: bf870003
	v_add_co_u32 v9, s0, v10, v69                              // 0000000064bc: d7000009 02028b0a
	v_or_b32_e32 v18, 0x400000, v11                            // 0000000064c4: 382416ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000064cc: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v16, v70, s0                // 0000000064d0: d5207c0a 00028d10
	v_cmp_u_f32_e64 s0, v11, v11                               // 0000000064d8: d4180000 0202170b
	s_wait_alu depctr_va_sdst(0)                               // 0000000064e0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000064e4: bf870001
	v_cndmask_b32_e64 v11, v17, v18, s0                        // 0000000064e8: d501000b 00022511
	global_store_d16_hi_b16 v[9:10], v11, off offset:32        // 0000000064f0: ee09407c 05800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006500: 8c7e017e
	s_and_b32 s0, vcc_lo, s22                                  // 000000006504: 8b00166a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006508: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000650c: be812000
	s_cbranch_execz 28                                         // 000000006510: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4a84>
	v_bfe_u32 v9, v12, 16, 1                                   // 000000006514: d6100009 0205210c
	s_wait_kmcnt 0x0                                           // 00000000651c: bfc70000
	v_add_co_u32 v10, s0, s48, v49                             // 000000006520: d700000a 02026230
	s_wait_alu depctr_va_sdst(0)                               // 000000006528: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s49, v50, s0                // 00000000652c: d5207c0b 00026431
	v_add3_u32 v16, v9, v12, 0x7fff                            // 000000006534: d6550010 03fe1909 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006540: bf870003
	v_add_co_u32 v9, s0, v10, v71                              // 000000006544: d7000009 02028f0a
	v_or_b32_e32 v17, 0x400000, v12                            // 00000000654c: 382218ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006554: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v11, v72, s0                // 000000006558: d5207c0a 0002910b
	v_cmp_u_f32_e64 s0, v12, v12                               // 000000006560: d4180000 0202190c
	s_wait_alu depctr_va_sdst(0)                               // 000000006568: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000656c: bf870001
	v_cndmask_b32_e64 v11, v16, v17, s0                        // 000000006570: d501000b 00022310
	global_store_d16_hi_b16 v[9:10], v11, off offset:32        // 000000006578: ee09407c 05800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 000000006584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006588: 8c7e017e
	s_and_b32 s0, vcc_lo, s23                                  // 00000000658c: 8b00176a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006590: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006594: be812000
	s_cbranch_execz 28                                         // 000000006598: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4b0c>
	v_bfe_u32 v9, v13, 16, 1                                   // 00000000659c: d6100009 0205210d
	s_wait_kmcnt 0x0                                           // 0000000065a4: bfc70000
	v_add_co_u32 v10, s0, s48, v49                             // 0000000065a8: d700000a 02026230
	s_wait_alu depctr_va_sdst(0)                               // 0000000065b0: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s49, v50, s0                // 0000000065b4: d5207c0b 00026431
	v_add3_u32 v12, v9, v13, 0x7fff                            // 0000000065bc: d655000c 03fe1b09 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000065c8: bf870003
	v_add_co_u32 v9, s0, v10, v73                              // 0000000065cc: d7000009 0202930a
	v_or_b32_e32 v16, 0x400000, v13                            // 0000000065d4: 38201aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000065dc: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v11, v74, s0                // 0000000065e0: d5207c0a 0002950b
	v_cmp_u_f32_e64 s0, v13, v13                               // 0000000065e8: d4180000 02021b0d
	s_wait_alu depctr_va_sdst(0)                               // 0000000065f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000065f4: bf870001
	v_cndmask_b32_e64 v11, v12, v16, s0                        // 0000000065f8: d501000b 0002210c
	global_store_d16_hi_b16 v[9:10], v11, off offset:32        // 000000006600: ee09407c 05800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 00000000660c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006610: 8c7e017e
	s_and_b32 s0, vcc_lo, s24                                  // 000000006614: 8b00186a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006618: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000661c: be812000
	s_cbranch_execz 28                                         // 000000006620: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4b94>
	v_bfe_u32 v9, v14, 16, 1                                   // 000000006624: d6100009 0205210e
	s_wait_kmcnt 0x0                                           // 00000000662c: bfc70000
	v_add_co_u32 v10, s0, s48, v49                             // 000000006630: d700000a 02026230
	s_wait_alu depctr_va_sdst(0)                               // 000000006638: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s49, v50, s0                // 00000000663c: d5207c0b 00026431
	v_add3_u32 v12, v9, v14, 0x7fff                            // 000000006644: d655000c 03fe1d09 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006650: bf870003
	v_add_co_u32 v9, s0, v10, v75                              // 000000006654: d7000009 0202970a
	v_or_b32_e32 v13, 0x400000, v14                            // 00000000665c: 381a1cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006664: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v11, v76, s0                // 000000006668: d5207c0a 0002990b
	v_cmp_u_f32_e64 s0, v14, v14                               // 000000006670: d4180000 02021d0e
	s_wait_alu depctr_va_sdst(0)                               // 000000006678: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000667c: bf870001
	v_cndmask_b32_e64 v11, v12, v13, s0                        // 000000006680: d501000b 00021b0c
	global_store_d16_hi_b16 v[9:10], v11, off offset:32        // 000000006688: ee09407c 05800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 000000006694: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006698: 8c7e017e
	s_and_b32 s0, vcc_lo, s25                                  // 00000000669c: 8b00196a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066a0: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 0000000066a4: be812000
	s_cbranch_execz 28                                         // 0000000066a8: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4c1c>
	v_bfe_u32 v9, v15, 16, 1                                   // 0000000066ac: d6100009 0205210f
	s_wait_kmcnt 0x0                                           // 0000000066b4: bfc70000
	v_add_co_u32 v10, s0, s48, v49                             // 0000000066b8: d700000a 02026230
	s_wait_alu depctr_va_sdst(0)                               // 0000000066c0: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s49, v50, s0                // 0000000066c4: d5207c0b 00026431
	v_add3_u32 v12, v9, v15, 0x7fff                            // 0000000066cc: d655000c 03fe1f09 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000066d8: bf870003
	v_add_co_u32 v9, s0, v10, v77                              // 0000000066dc: d7000009 02029b0a
	v_or_b32_e32 v13, 0x400000, v15                            // 0000000066e4: 381a1eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000066ec: bf88f19f
	v_add_co_ci_u32_e64 v10, null, v11, v78, s0                // 0000000066f0: d5207c0a 00029d0b
	v_cmp_u_f32_e64 s0, v15, v15                               // 0000000066f8: d4180000 02021f0f
	s_wait_alu depctr_va_sdst(0)                               // 000000006700: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006704: bf870001
	v_cndmask_b32_e64 v11, v12, v13, s0                        // 000000006708: d501000b 00021b0c
	global_store_d16_hi_b16 v[9:10], v11, off offset:32        // 000000006710: ee09407c 05800000 00002009
	s_wait_alu depctr_sa_sdst(0)                               // 00000000671c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006720: 8c7e017e
	s_and_b32 s0, s36, s26                                     // 000000006724: 8b001a24
	s_wait_alu depctr_sa_sdst(0)                               // 000000006728: bf88ff9e
	s_and_b32 s0, s52, s0                                      // 00000000672c: 8b000034
	s_wait_alu depctr_sa_sdst(0)                               // 000000006730: bf88ff9e
	s_xor_b32 s0, s0, -1                                       // 000000006734: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006738: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 00000000673c: be812000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006740: bf88ff9e
	s_xor_b32 s2, exec_lo, s1                                  // 000000006744: 8d02017e
	s_cbranch_execz 133                                        // 000000006748: bfa50085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4e60>
	v_mov_b32_e32 v134, v54                                    // 00000000674c: 7f0c0336
	v_cmp_gt_i64_e64 s0, s[40:41], v[53:54]                    // 000000006750: d4540000 02026a28
	v_mov_b32_e32 v132, v54                                    // 000000006758: 7f080336
	v_mov_b32_e32 v130, v54                                    // 00000000675c: 7f040336
	v_mov_b32_e32 v122, v54                                    // 000000006760: 7ef40336
	v_cmp_gt_i64_e64 s1, s[40:41], v[133:134]                  // 000000006764: d4540001 02030a28
	v_mov_b32_e32 v124, v54                                    // 00000000676c: 7ef80336
	s_wait_alu depctr_va_sdst(0)                               // 000000006770: bf88f19f
	v_cndmask_b32_e64 v10, 0, v54, s0                          // 000000006774: d501000a 00026c80
	v_cndmask_b32_e64 v9, 0, v53, s0                           // 00000000677c: d5010009 00026a80
	v_cmp_gt_i64_e64 s0, s[40:41], v[131:132]                  // 000000006784: d4540000 02030628
	v_mov_b32_e32 v56, v54                                     // 00000000678c: 7e700336
	v_cndmask_b32_e64 v11, 0, v133, s1                         // 000000006790: d501000b 00070a80
	v_cndmask_b32_e64 v12, 0, v54, s1                          // 000000006798: d501000c 00066c80
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]                      // 0000000067a0: 3e121282
	v_cmp_gt_i64_e64 s1, s[40:41], v[129:130]                  // 0000000067a4: d4540001 02030228
	s_wait_alu depctr_va_sdst(0)                               // 0000000067ac: bf88f19f
	v_cndmask_b32_e64 v13, 0, v131, s0                         // 0000000067b0: d501000d 00030680
	v_cndmask_b32_e64 v14, 0, v54, s0                          // 0000000067b8: d501000e 00026c80
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]                    // 0000000067c0: 3e161682
	v_add_co_u32 v9, s0, s46, v9                               // 0000000067c4: d7000009 0202122e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_4)// 0000000067cc: bf870233
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]                    // 0000000067d0: 3e1a1a82
	s_wait_alu depctr_va_sdst(0)                               // 0000000067d4: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s47, v10, s0                // 0000000067d8: d5207c0a 0002142f
	v_add_co_u32 v11, s0, s46, v11                             // 0000000067e0: d700000b 0202162e
	v_cndmask_b32_e64 v15, 0, v129, s1                         // 0000000067e8: d501000f 00070280
	v_cndmask_b32_e64 v16, 0, v54, s1                          // 0000000067f0: d5010010 00066c80
	s_wait_alu depctr_va_sdst(0)                               // 0000000067f8: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s47, v12, s0                // 0000000067fc: d5207c0c 0002182f
	v_cmp_gt_i64_e64 s0, s[40:41], v[121:122]                  // 000000006804: d4540000 0202f228
	v_add_co_u32 v17, s1, s46, v13                             // 00000000680c: d7000111 02021a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000006814: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s47, v14, s1                // 000000006818: d5207c12 00061c2f
	v_cmp_gt_i64_e64 s1, s[40:41], v[123:124]                  // 000000006820: d4540001 0202f628
	v_lshlrev_b64_e32 v[13:14], 2, v[15:16]                    // 000000006828: 3e1a1e82
	v_cndmask_b32_e64 v15, 0, v121, s0                         // 00000000682c: d501000f 0002f280
	v_cndmask_b32_e64 v16, 0, v54, s0                          // 000000006834: d5010010 00026c80
	v_cmp_gt_i64_e64 s0, s[40:41], v[55:56]                    // 00000000683c: d4540000 02026e28
	s_wait_alu depctr_va_sdst(0)                               // 000000006844: bf88f19f
	v_cndmask_b32_e64 v19, 0, v123, s1                         // 000000006848: d5010013 0006f680
	v_cndmask_b32_e64 v20, 0, v54, s1                          // 000000006850: d5010014 00066c80
	v_add_co_u32 v21, s1, s46, v13                             // 000000006858: d7000115 02021a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000006860: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s47, v14, s1                // 000000006864: d5207c16 00061c2f
	v_lshlrev_b64_e32 v[13:14], 2, v[15:16]                    // 00000000686c: 3e1a1e82
	v_lshlrev_b64_e32 v[15:16], 2, v[19:20]                    // 000000006870: 3e1e2682
	v_cndmask_b32_e64 v19, 0, v55, s0                          // 000000006874: d5010013 00026e80
	v_cndmask_b32_e64 v20, 0, v54, s0                          // 00000000687c: d5010014 00026c80
	v_cmp_gt_i64_e64 s0, s[40:41], v[51:52]                    // 000000006884: d4540000 02026628
	v_add_co_u32 v23, s1, s46, v13                             // 00000000688c: d7000117 02021a2e
	s_wait_alu depctr_va_sdst(0)                               // 000000006894: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s47, v14, s1                // 000000006898: d5207c18 00061c2f
	v_lshlrev_b64_e32 v[13:14], 2, v[19:20]                    // 0000000068a0: 3e1a2682
	s_delay_alu instid0(valu_dep_4) | instskip(skip_4) | instid1(valu_dep_3)// 0000000068a4: bf8701d4
	v_cndmask_b32_e64 v20, 0, v52, s0                          // 0000000068a8: d5010014 00026880
	v_cndmask_b32_e64 v19, 0, v51, s0                          // 0000000068b0: d5010013 00026680
	v_add_co_u32 v25, s0, s46, v15                             // 0000000068b8: d7000019 02021e2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000068c0: bf88f19f
	v_add_co_ci_u32_e64 v26, null, s47, v16, s0                // 0000000068c4: d5207c1a 0002202f
	v_lshlrev_b64_e32 v[15:16], 2, v[19:20]                    // 0000000068cc: 3e1e2682
	v_add_co_u32 v19, s0, s46, v13                             // 0000000068d0: d7000013 02021a2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000068d8: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s47, v14, s0                // 0000000068dc: d5207c14 00021c2f
	s_delay_alu instid0(valu_dep_3)                            // 0000000068e4: bf870003
	v_add_co_u32 v27, s0, s46, v15                             // 0000000068e8: d700001b 02021e2e
	s_wait_alu depctr_va_sdst(0)                               // 0000000068f0: bf88f19f
	v_add_co_ci_u32_e64 v28, null, s47, v16, s0                // 0000000068f4: d5207c1c 0002202f
	s_clause 0x7                                               // 0000000068fc: bf850007
	global_load_b32 v13, v[9:10], off                          // 000000006900: ee05007c 0000000d 00000009
	global_load_b32 v14, v[11:12], off                         // 00000000690c: ee05007c 0000000e 0000000b
	global_load_b32 v15, v[17:18], off                         // 000000006918: ee05007c 0000000f 00000011
	global_load_b32 v16, v[21:22], off                         // 000000006924: ee05007c 00000010 00000015
	global_load_b32 v9, v[23:24], off                          // 000000006930: ee05007c 00000009 00000017
	global_load_b32 v10, v[25:26], off                         // 00000000693c: ee05007c 0000000a 00000019
	global_load_b32 v11, v[19:20], off                         // 000000006948: ee05007c 0000000b 00000013
	global_load_b32 v12, v[27:28], off                         // 000000006954: ee05007c 0000000c 0000001b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006960: bf88ff9e
	s_and_not1_saveexec_b32 s1, s2                             // 000000006964: be813002
	s_cbranch_execz 28                                         // 000000006968: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4edc>
	s_wait_loadcnt 0x3                                         // 00000000696c: bfc00003
	v_add_co_u32 v9, s0, s44, v85                              // 000000006970: d7000009 0202aa2c
	s_wait_loadcnt 0x2                                         // 000000006978: bfc00002
	s_wait_alu depctr_va_sdst(0)                               // 00000000697c: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s45, 0, s0                  // 000000006980: d5207c0a 0001002d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000006988: bf870122
	v_add_co_u32 v9, s0, v9, v86                               // 00000000698c: d7000009 0202ad09
	s_wait_alu depctr_va_sdst(0)                               // 000000006994: bf88f19f
	v_add_co_ci_u32_e64 v10, null, 0, v10, s0                  // 000000006998: d5207c0a 00021480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069a0: bf870091
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]                      // 0000000069a4: 3e121282
	v_add_co_u32 v9, s0, s46, v9                               // 0000000069a8: d7000009 0202122e
	s_wait_alu depctr_va_sdst(0)                               // 0000000069b0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000069b4: bf870002
	v_add_co_ci_u32_e64 v10, null, s47, v10, s0                // 0000000069b8: d5207c0a 0002142f
	global_load_b128 v[13:16], v[9:10], off offset:192         // 0000000069c0: ee05c07c 0000000d 0000c009
	s_wait_loadcnt 0x1                                         // 0000000069cc: bfc00001
	global_load_b128 v[9:12], v[9:10], off offset:208          // 0000000069d0: ee05c07c 00000009 0000d009
	s_wait_alu depctr_sa_sdst(0)                               // 0000000069dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000069e0: 8c7e017e
	global_load_u8 v17, v[43:44], off                          // 0000000069e4: ee04007c 00000011 0000002b
	s_wait_loadcnt 0x0                                         // 0000000069f0: bfc00000
	v_lshlrev_b32_e32 v18, 23, v17                             // 0000000069f4: 30242297
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000069f8: bf870091
	v_mul_f32_e32 v17, v13, v18                                // 0000000069fc: 1022250d
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 000000006a00: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v1, v17                                 // 000000006a0c: 10222301
	s_xor_b32 s0, s0, -1                                       // 000000006a10: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a14: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006a18: be812000
	s_cbranch_execnz 1370                                      // 000000006a1c: bfa6055a <tessera_rocm_folded_matmul_a629485a5b78541c+0x6488>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006a24: 8c7e017e
	v_mul_f32_e32 v1, v14, v18                                 // 000000006a28: 1002250e
	s_delay_alu instid0(valu_dep_1)                            // 000000006a2c: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 000000006a30: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v2, v1                                   // 000000006a3c: 10020302
	s_xor_b32 s0, s0, -1                                       // 000000006a40: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a44: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006a48: be812000
	s_cbranch_execnz 1376                                      // 000000006a4c: bfa60560 <tessera_rocm_folded_matmul_a629485a5b78541c+0x64d0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006a54: 8c7e017e
	v_mul_f32_e32 v2, v15, v18                                 // 000000006a58: 1004250f
	s_delay_alu instid0(valu_dep_1)                            // 000000006a5c: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000006a60: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 000000006a6c: 10040503
	s_xor_b32 s0, s0, -1                                       // 000000006a70: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a74: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006a78: be812000
	s_cbranch_execnz 1382                                      // 000000006a7c: bfa60566 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6518>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006a80: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006a84: 8c7e017e
	v_mul_f32_e32 v3, v16, v18                                 // 000000006a88: 10062510
	s_delay_alu instid0(valu_dep_1)                            // 000000006a8c: bf870001
	v_cmp_class_f32_e64 s0, v3, 0x198                          // 000000006a90: d47e0000 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 000000006a9c: 10060704
	s_xor_b32 s0, s0, -1                                       // 000000006aa0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006aa4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006aa8: be812000
	s_cbranch_execnz 1388                                      // 000000006aac: bfa6056c <tessera_rocm_folded_matmul_a629485a5b78541c+0x6560>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ab0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006ab4: 8c7e017e
	v_mul_f32_e32 v4, v9, v18                                  // 000000006ab8: 10082509
	s_delay_alu instid0(valu_dep_1)                            // 000000006abc: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 000000006ac0: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 000000006acc: 10080905
	s_xor_b32 s0, s0, -1                                       // 000000006ad0: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ad4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006ad8: be812000
	s_cbranch_execnz 1394                                      // 000000006adc: bfa60572 <tessera_rocm_folded_matmul_a629485a5b78541c+0x65a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ae0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006ae4: 8c7e017e
	v_mul_f32_e32 v5, v10, v18                                 // 000000006ae8: 100a250a
	s_delay_alu instid0(valu_dep_1)                            // 000000006aec: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 000000006af0: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 000000006afc: 100a0b06
	s_xor_b32 s0, s0, -1                                       // 000000006b00: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b04: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006b08: be812000
	s_cbranch_execnz 1400                                      // 000000006b0c: bfa60578 <tessera_rocm_folded_matmul_a629485a5b78541c+0x65f0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006b14: 8c7e017e
	v_mul_f32_e32 v6, v11, v18                                 // 000000006b18: 100c250b
	s_delay_alu instid0(valu_dep_1)                            // 000000006b1c: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 000000006b20: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v7, v6                                   // 000000006b2c: 100c0d07
	s_xor_b32 s0, s0, -1                                       // 000000006b30: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b34: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006b38: be812000
	s_cbranch_execnz 1406                                      // 000000006b3c: bfa6057e <tessera_rocm_folded_matmul_a629485a5b78541c+0x6638>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006b44: 8c7e017e
	v_mul_f32_e32 v7, v12, v18                                 // 000000006b48: 100e250c
	s_delay_alu instid0(valu_dep_1)                            // 000000006b4c: bf870001
	v_cmp_class_f32_e64 s0, v7, 0x198                          // 000000006b50: d47e0000 0201ff07 00000198
	v_mul_f32_e32 v7, v8, v7                                   // 000000006b5c: 100e0f08
	s_xor_b32 s0, s0, -1                                       // 000000006b60: 8d00c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b64: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006b68: be812000
	s_cbranch_execnz 1412                                      // 000000006b6c: bfa60584 <tessera_rocm_folded_matmul_a629485a5b78541c+0x6680>
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b70: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006b74: 8c7e017e
	s_and_b32 s0, vcc_lo, s30                                  // 000000006b78: 8b001e6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006b7c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006b80: be812000
	s_cbranch_execz 34                                         // 000000006b84: bfa50022 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5110>
	v_add_co_u32 v8, s0, v0, s38                               // 000000006b88: d7000008 02004d00
	s_wait_alu depctr_va_sdst(0)                               // 000000006b90: bf88f19f
	v_add_co_ci_u32_e64 v9, null, 0, s39, s0                   // 000000006b94: d5207c09 00004e80
	s_wait_kmcnt 0x0                                           // 000000006b9c: bfc70000
	v_add_co_u32 v10, s0, s48, v41                             // 000000006ba0: d700000a 02025230
	v_bfe_u32 v0, v17, 16, 1                                   // 000000006ba8: d6100000 02052111
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_4)// 000000006bb0: bf870253
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000006bb4: 3e101081
	s_wait_alu depctr_va_sdst(0)                               // 000000006bb8: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s49, v42, s0                // 000000006bbc: d5207c0b 00025431
	v_or_b32_e32 v12, 0x400000, v17                            // 000000006bc4: 381822ff 00400000
	v_add3_u32 v0, v0, v17, 0x7fff                             // 000000006bcc: d6550000 03fe2300 00007fff
	v_add_co_u32 v8, s0, v10, v8                               // 000000006bd8: d7000008 0202110a
	s_wait_alu depctr_va_sdst(0)                               // 000000006be0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v11, v9, s0                  // 000000006be4: d5207c09 0002130b
	v_cmp_u_f32_e64 s0, v17, v17                               // 000000006bec: d4180000 02022311
	s_wait_alu depctr_va_sdst(0)                               // 000000006bf4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006bf8: bf870001
	v_cndmask_b32_e64 v0, v0, v12, s0                          // 000000006bfc: d5010000 00021900
	global_store_d16_hi_b16 v[8:9], v0, off offset:32          // 000000006c04: ee09407c 00000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006c14: 8c7e017e
	s_and_b32 s0, vcc_lo, s27                                  // 000000006c18: 8b001b6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c1c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006c20: be812000
	s_cbranch_execz 28                                         // 000000006c24: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5198>
	s_wait_kmcnt 0x0                                           // 000000006c28: bfc70000
	v_add_co_u32 v8, s0, s48, v41                              // 000000006c2c: d7000008 02025230
	v_bfe_u32 v0, v1, 16, 1                                    // 000000006c34: d6100000 02052101
	s_wait_alu depctr_va_sdst(0)                               // 000000006c3c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s49, v42, s0                 // 000000006c40: d5207c09 00025431
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006c48: bf870193
	v_add_co_u32 v8, s0, v8, v65                               // 000000006c4c: d7000008 02028308
	v_add3_u32 v0, v0, v1, 0x7fff                              // 000000006c54: d6550000 03fe0300 00007fff
	v_or_b32_e32 v10, 0x400000, v1                             // 000000006c60: 381402ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006c68: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v9, v66, s0                  // 000000006c6c: d5207c09 00028509
	v_cmp_u_f32_e64 s0, v1, v1                                 // 000000006c74: d4180000 02020301
	s_wait_alu depctr_va_sdst(0)                               // 000000006c7c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006c80: bf870001
	v_cndmask_b32_e64 v0, v0, v10, s0                          // 000000006c84: d5010000 00021500
	global_store_d16_hi_b16 v[8:9], v0, off offset:32          // 000000006c8c: ee09407c 00000000 00002008
	s_wait_alu depctr_sa_sdst(0)                               // 000000006c98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006c9c: 8c7e017e
	s_and_b32 s0, vcc_lo, s28                                  // 000000006ca0: 8b001c6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ca4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006ca8: be812000
	s_cbranch_execz 28                                         // 000000006cac: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5220>
	v_bfe_u32 v0, v2, 16, 1                                    // 000000006cb0: d6100000 02052102
	s_wait_kmcnt 0x0                                           // 000000006cb8: bfc70000
	v_add_co_u32 v1, s0, s48, v41                              // 000000006cbc: d7000001 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000006cc4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s49, v42, s0                 // 000000006cc8: d5207c08 00025431
	v_add3_u32 v9, v0, v2, 0x7fff                              // 000000006cd0: d6550009 03fe0500 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006cdc: bf870003
	v_add_co_u32 v0, s0, v1, v67                               // 000000006ce0: d7000000 02028701
	v_or_b32_e32 v10, 0x400000, v2                             // 000000006ce8: 381404ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006cf0: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v8, v68, s0                  // 000000006cf4: d5207c01 00028908
	v_cmp_u_f32_e64 s0, v2, v2                                 // 000000006cfc: d4180000 02020502
	s_wait_alu depctr_va_sdst(0)                               // 000000006d04: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006d08: bf870001
	v_cndmask_b32_e64 v2, v9, v10, s0                          // 000000006d0c: d5010002 00021509
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006d14: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d20: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006d24: 8c7e017e
	s_and_b32 s0, vcc_lo, s29                                  // 000000006d28: 8b001d6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006d2c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006d30: be812000
	s_cbranch_execz 28                                         // 000000006d34: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x52a8>
	v_bfe_u32 v0, v3, 16, 1                                    // 000000006d38: d6100000 02052103
	s_wait_kmcnt 0x0                                           // 000000006d40: bfc70000
	v_add_co_u32 v1, s0, s48, v41                              // 000000006d44: d7000001 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000006d4c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s49, v42, s0                 // 000000006d50: d5207c02 00025431
	v_add3_u32 v8, v0, v3, 0x7fff                              // 000000006d58: d6550008 03fe0700 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006d64: bf870003
	v_add_co_u32 v0, s0, v1, v69                               // 000000006d68: d7000000 02028b01
	v_or_b32_e32 v9, 0x400000, v3                              // 000000006d70: 381206ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006d78: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v70, s0                  // 000000006d7c: d5207c01 00028d02
	v_cmp_u_f32_e64 s0, v3, v3                                 // 000000006d84: d4180000 02020703
	s_wait_alu depctr_va_sdst(0)                               // 000000006d8c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006d90: bf870001
	v_cndmask_b32_e64 v2, v8, v9, s0                           // 000000006d94: d5010002 00021308
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006d9c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006da8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006dac: 8c7e017e
	s_and_b32 s0, vcc_lo, s31                                  // 000000006db0: 8b001f6a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006db4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006db8: be812000
	s_cbranch_execz 28                                         // 000000006dbc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5330>
	v_bfe_u32 v0, v4, 16, 1                                    // 000000006dc0: d6100000 02052104
	s_wait_kmcnt 0x0                                           // 000000006dc8: bfc70000
	v_add_co_u32 v1, s0, s48, v41                              // 000000006dcc: d7000001 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000006dd4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s49, v42, s0                 // 000000006dd8: d5207c02 00025431
	v_add3_u32 v3, v0, v4, 0x7fff                              // 000000006de0: d6550003 03fe0900 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006dec: bf870003
	v_add_co_u32 v0, s0, v1, v71                               // 000000006df0: d7000000 02028f01
	v_or_b32_e32 v8, 0x400000, v4                              // 000000006df8: 381008ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006e00: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v72, s0                  // 000000006e04: d5207c01 00029102
	v_cmp_u_f32_e64 s0, v4, v4                                 // 000000006e0c: d4180000 02020904
	s_wait_alu depctr_va_sdst(0)                               // 000000006e14: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006e18: bf870001
	v_cndmask_b32_e64 v2, v3, v8, s0                           // 000000006e1c: d5010002 00021103
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006e24: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006e34: 8c7e017e
	s_and_b32 s0, vcc_lo, s33                                  // 000000006e38: 8b00216a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006e3c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006e40: be812000
	s_cbranch_execz 28                                         // 000000006e44: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x53b8>
	v_bfe_u32 v0, v5, 16, 1                                    // 000000006e48: d6100000 02052105
	s_wait_kmcnt 0x0                                           // 000000006e50: bfc70000
	v_add_co_u32 v1, s0, s48, v41                              // 000000006e54: d7000001 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000006e5c: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s49, v42, s0                 // 000000006e60: d5207c02 00025431
	v_add3_u32 v3, v0, v5, 0x7fff                              // 000000006e68: d6550003 03fe0b00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006e74: bf870003
	v_add_co_u32 v0, s0, v1, v73                               // 000000006e78: d7000000 02029301
	v_or_b32_e32 v4, 0x400000, v5                              // 000000006e80: 38080aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006e88: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v74, s0                  // 000000006e8c: d5207c01 00029502
	v_cmp_u_f32_e64 s0, v5, v5                                 // 000000006e94: d4180000 02020b05
	s_wait_alu depctr_va_sdst(0)                               // 000000006e9c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006ea0: bf870001
	v_cndmask_b32_e64 v2, v3, v4, s0                           // 000000006ea4: d5010002 00020903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006eac: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006eb8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006ebc: 8c7e017e
	s_and_b32 s0, vcc_lo, s34                                  // 000000006ec0: 8b00226a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ec4: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006ec8: be812000
	s_cbranch_execz 28                                         // 000000006ecc: bfa5001c <tessera_rocm_folded_matmul_a629485a5b78541c+0x5440>
	v_bfe_u32 v0, v6, 16, 1                                    // 000000006ed0: d6100000 02052106
	s_wait_kmcnt 0x0                                           // 000000006ed8: bfc70000
	v_add_co_u32 v1, s0, s48, v41                              // 000000006edc: d7000001 02025230
	s_wait_alu depctr_va_sdst(0)                               // 000000006ee4: bf88f19f
	v_add_co_ci_u32_e64 v2, null, s49, v42, s0                 // 000000006ee8: d5207c02 00025431
	v_add3_u32 v3, v0, v6, 0x7fff                              // 000000006ef0: d6550003 03fe0d00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006efc: bf870003
	v_add_co_u32 v0, s0, v1, v75                               // 000000006f00: d7000000 02029701
	v_or_b32_e32 v4, 0x400000, v6                              // 000000006f08: 38080cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000006f10: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v2, v76, s0                  // 000000006f14: d5207c01 00029902
	v_cmp_u_f32_e64 s0, v6, v6                                 // 000000006f1c: d4180000 02020d06
	s_wait_alu depctr_va_sdst(0)                               // 000000006f24: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000006f28: bf870001
	v_cndmask_b32_e64 v2, v3, v4, s0                           // 000000006f2c: d5010002 00020903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006f34: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f40: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006f44: 8c7e017e
	s_and_b32 s0, vcc_lo, s35                                  // 000000006f48: 8b00236a
	s_wait_alu depctr_sa_sdst(0)                               // 000000006f4c: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000006f50: be812000
	s_cbranch_execz 25                                         // 000000006f54: bfa50019 <tessera_rocm_folded_matmul_a629485a5b78541c+0x54bc>
	v_bfe_u32 v0, v7, 16, 1                                    // 000000006f58: d6100000 02052107
	s_wait_kmcnt 0x0                                           // 000000006f60: bfc70000
	v_add_co_u32 v1, vcc_lo, s48, v41                          // 000000006f64: d7006a01 02025230
	s_wait_alu depctr_va_vcc(0)                                // 000000006f6c: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s49, v42, vcc_lo             // 000000006f70: d5207c02 01aa5431
	v_add3_u32 v3, v0, v7, 0x7fff                              // 000000006f78: d6550003 03fe0f00 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006f84: bf870003
	v_add_co_u32 v0, vcc_lo, v1, v77                           // 000000006f88: d7006a00 02029b01
	v_or_b32_e32 v4, 0x400000, v7                              // 000000006f90: 38080eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006f98: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v78, vcc_lo              // 000000006f9c: d5207c01 01aa9d02
	v_cmp_u_f32_e32 vcc_lo, v7, v7                             // 000000006fa4: 7c300f07
	s_wait_alu depctr_va_vcc(0)                                // 000000006fa8: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000006fac: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000006fb0: ee09407c 01000000 00002000
	s_nop 0                                                    // 000000006fbc: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000006fc0: bfb60003
	s_endpgm                                                   // 000000006fc4: bfb00000
	v_cvt_f64_f32_e32 v[75:76], v57                            // 000000006fc8: 7e962139
	v_cvt_f64_f32_e32 v[77:78], v79                            // 000000006fcc: 7e9a214f
	v_cvt_f64_f32_e32 v[100:101], v69                          // 000000006fd0: 7ec82145
	v_cmp_eq_f32_e64 s2, 0, v57                                // 000000006fd4: d4120002 02027280
	v_cmp_class_f32_e64 s4, v69, 0x1f8                         // 000000006fdc: d47e0004 0201ff45 000001f8
	s_and_b32 s2, s2, s4                                       // 000000006fe8: 8b020402
	v_mul_f64_e32 v[75:76], v[75:76], v[77:78]                 // 000000006fec: 0c969b4b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006ff0: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[100:101]               // 000000006ff4: 0c96c94b
	v_cvt_f32_f64_e32 v75, v[75:76]                            // 000000006ff8: 7e961f4b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006ffc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007000: bf870001
	v_cndmask_b32_e64 v78, v75, 0, s2                          // 000000007004: d501004e 0009014b
	s_branch 61106                                             // 00000000700c: bfa0eeb2 <tessera_rocm_folded_matmul_a629485a5b78541c+0xfd8>
	v_cvt_f64_f32_e32 v[75:76], v58                            // 000000007010: 7e96213a
	v_cvt_f64_f32_e32 v[100:101], v79                          // 000000007014: 7ec8214f
	v_cvt_f64_f32_e32 v[102:103], v70                          // 000000007018: 7ecc2146
	v_cmp_eq_f32_e64 s2, 0, v58                                // 00000000701c: d4120002 02027480
	v_cmp_class_f32_e64 s4, v70, 0x1f8                         // 000000007024: d47e0004 0201ff46 000001f8
	s_and_b32 s2, s2, s4                                       // 000000007030: 8b020402
	v_mul_f64_e32 v[75:76], v[75:76], v[100:101]               // 000000007034: 0c96c94b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007038: bf870091
	v_mul_f64_e32 v[75:76], v[75:76], v[102:103]               // 00000000703c: 0c96cd4b
	v_cvt_f32_f64_e32 v57, v[75:76]                            // 000000007040: 7e721f4b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007044: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007048: bf870001
	v_cndmask_b32_e64 v77, v57, 0, s2                          // 00000000704c: d501004d 00090139
	s_branch 61100                                             // 000000007054: bfa0eeac <tessera_rocm_folded_matmul_a629485a5b78541c+0x1008>
	v_cvt_f64_f32_e32 v[57:58], v59                            // 000000007058: 7e72213b
	v_cvt_f64_f32_e32 v[69:70], v79                            // 00000000705c: 7e8a214f
	v_cvt_f64_f32_e32 v[75:76], v71                            // 000000007060: 7e962147
	v_cmp_eq_f32_e64 s2, 0, v59                                // 000000007064: d4120002 02027680
	v_cmp_class_f32_e64 s4, v71, 0x1f8                         // 00000000706c: d47e0004 0201ff47 000001f8
	s_and_b32 s2, s2, s4                                       // 000000007078: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[69:70]                 // 00000000707c: 0c728b39
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007080: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[75:76]                 // 000000007084: 0c729739
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000007088: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000708c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007090: bf870001
	v_cndmask_b32_e64 v69, v57, 0, s2                          // 000000007094: d5010045 00090139
	s_branch 61094                                             // 00000000709c: bfa0eea6 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1038>
	v_cvt_f64_f32_e32 v[57:58], v60                            // 0000000070a0: 7e72213c
	v_cvt_f64_f32_e32 v[70:71], v79                            // 0000000070a4: 7e8c214f
	v_cvt_f64_f32_e32 v[75:76], v72                            // 0000000070a8: 7e962148
	v_cmp_eq_f32_e64 s2, 0, v60                                // 0000000070ac: d4120002 02027880
	v_cmp_class_f32_e64 s4, v72, 0x1f8                         // 0000000070b4: d47e0004 0201ff48 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000070c0: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[70:71]                 // 0000000070c4: 0c728d39
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000070c8: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[75:76]                 // 0000000070cc: 0c729739
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000070d0: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000070d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000070d8: bf870001
	v_cndmask_b32_e64 v71, v57, 0, s2                          // 0000000070dc: d5010047 00090139
	s_branch 61088                                             // 0000000070e4: bfa0eea0 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1068>
	v_cvt_f64_f32_e32 v[57:58], v61                            // 0000000070e8: 7e72213d
	v_cvt_f64_f32_e32 v[59:60], v79                            // 0000000070ec: 7e76214f
	v_cvt_f64_f32_e32 v[75:76], v65                            // 0000000070f0: 7e962141
	v_cmp_eq_f32_e64 s2, 0, v61                                // 0000000070f4: d4120002 02027a80
	v_cmp_class_f32_e64 s4, v65, 0x1f8                         // 0000000070fc: d47e0004 0201ff41 000001f8
	s_and_b32 s2, s2, s4                                       // 000000007108: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 00000000710c: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007110: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[75:76]                 // 000000007114: 0c729739
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000007118: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 00000000711c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007120: bf870001
	v_cndmask_b32_e64 v76, v57, 0, s2                          // 000000007124: d501004c 00090139
	s_branch 61082                                             // 00000000712c: bfa0ee9a <tessera_rocm_folded_matmul_a629485a5b78541c+0x1098>
	v_cvt_f64_f32_e32 v[57:58], v62                            // 000000007130: 7e72213e
	v_cvt_f64_f32_e32 v[59:60], v79                            // 000000007134: 7e76214f
	v_cvt_f64_f32_e32 v[100:101], v66                          // 000000007138: 7ec82142
	v_cmp_eq_f32_e64 s2, 0, v62                                // 00000000713c: d4120002 02027c80
	v_cmp_class_f32_e64 s4, v66, 0x1f8                         // 000000007144: d47e0004 0201ff42 000001f8
	s_and_b32 s2, s2, s4                                       // 000000007150: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 000000007154: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007158: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[100:101]               // 00000000715c: 0c72c939
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 000000007160: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000007164: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007168: bf870001
	v_cndmask_b32_e64 v75, v57, 0, s2                          // 00000000716c: d501004b 00090139
	s_branch 61076                                             // 000000007174: bfa0ee94 <tessera_rocm_folded_matmul_a629485a5b78541c+0x10c8>
	v_cvt_f64_f32_e32 v[57:58], v63                            // 000000007178: 7e72213f
	v_cvt_f64_f32_e32 v[59:60], v79                            // 00000000717c: 7e76214f
	v_cvt_f64_f32_e32 v[61:62], v67                            // 000000007180: 7e7a2143
	v_cmp_eq_f32_e64 s2, 0, v63                                // 000000007184: d4120002 02027e80
	v_cmp_class_f32_e64 s4, v67, 0x1f8                         // 00000000718c: d47e0004 0201ff43 000001f8
	s_and_b32 s2, s2, s4                                       // 000000007198: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 00000000719c: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000071a0: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[61:62]                 // 0000000071a4: 0c727b39
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000071a8: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071ac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000071b0: bf870001
	v_cndmask_b32_e64 v62, v57, 0, s2                          // 0000000071b4: d501003e 00090139
	s_branch 61070                                             // 0000000071bc: bfa0ee8e <tessera_rocm_folded_matmul_a629485a5b78541c+0x10f8>
	v_cvt_f64_f32_e32 v[57:58], v64                            // 0000000071c0: 7e722140
	v_cvt_f64_f32_e32 v[59:60], v79                            // 0000000071c4: 7e76214f
	v_cvt_f64_f32_e32 v[65:66], v68                            // 0000000071c8: 7e822144
	v_cmp_eq_f32_e64 s2, 0, v64                                // 0000000071cc: d4120002 02028080
	v_cmp_class_f32_e64 s4, v68, 0x1f8                         // 0000000071d4: d47e0004 0201ff44 000001f8
	s_and_b32 s2, s2, s4                                       // 0000000071e0: 8b020402
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 0000000071e4: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000071e8: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[65:66]                 // 0000000071ec: 0c728339
	v_cvt_f32_f64_e32 v57, v[57:58]                            // 0000000071f0: 7e721f39
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071f4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000071f8: bf870001
	v_cndmask_b32_e64 v61, v57, 0, s2                          // 0000000071fc: d501003d 00090139
	s_branch 61064                                             // 000000007204: bfa0ee88 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1128>
	v_cvt_f64_f32_e32 v[112:113], v49                          // 000000007208: 7ee02131
	v_cvt_f64_f32_e32 v[118:119], v90                          // 00000000720c: 7eec215a
	v_cvt_f64_f32_e32 v[122:123], v61                          // 000000007210: 7ef4213d
	v_cmp_eq_f32_e64 s9, 0, v49                                // 000000007214: d4120009 02026280
	v_cmp_class_f32_e64 s11, v61, 0x1f8                        // 00000000721c: d47e000b 0201ff3d 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007228: 8b090b09
	v_mul_f64_e32 v[112:113], v[112:113], v[118:119]           // 00000000722c: 0ce0ed70
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007230: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 000000007234: 0ce0f570
	v_cvt_f32_f64_e32 v84, v[112:113]                          // 000000007238: 7ea81f70
	s_wait_alu depctr_sa_sdst(0)                               // 00000000723c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007240: bf870001
	v_cndmask_b32_e64 v84, v84, 0, s9                          // 000000007244: d5010054 00250154
	s_branch 61609                                             // 00000000724c: bfa0f0a9 <tessera_rocm_folded_matmul_a629485a5b78541c+0x19f4>
	v_cvt_f64_f32_e32 v[112:113], v50                          // 000000007250: 7ee02132
	v_cvt_f64_f32_e32 v[118:119], v90                          // 000000007254: 7eec215a
	v_cvt_f64_f32_e32 v[122:123], v62                          // 000000007258: 7ef4213e
	v_cmp_eq_f32_e64 s9, 0, v50                                // 00000000725c: d4120009 02026480
	v_cmp_class_f32_e64 s11, v62, 0x1f8                        // 000000007264: d47e000b 0201ff3e 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007270: 8b090b09
	v_mul_f64_e32 v[112:113], v[112:113], v[118:119]           // 000000007274: 0ce0ed70
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007278: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[122:123]           // 00000000727c: 0ce0f570
	v_cvt_f32_f64_e32 v49, v[112:113]                          // 000000007280: 7e621f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000007284: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007288: bf870001
	v_cndmask_b32_e64 v61, v49, 0, s9                          // 00000000728c: d501003d 00250131
	s_branch 61603                                             // 000000007294: bfa0f0a3 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1a24>
	v_cvt_f64_f32_e32 v[49:50], v51                            // 000000007298: 7e622133
	v_cvt_f64_f32_e32 v[112:113], v90                          // 00000000729c: 7ee0215a
	v_cvt_f64_f32_e32 v[118:119], v63                          // 0000000072a0: 7eec213f
	v_cmp_eq_f32_e64 s9, 0, v51                                // 0000000072a4: d4120009 02026680
	v_cmp_class_f32_e64 s11, v63, 0x1f8                        // 0000000072ac: d47e000b 0201ff3f 000001f8
	s_and_b32 s9, s9, s11                                      // 0000000072b8: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[112:113]               // 0000000072bc: 0c62e131
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000072c0: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[118:119]               // 0000000072c4: 0c62ed31
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 0000000072c8: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 0000000072cc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000072d0: bf870001
	v_cndmask_b32_e64 v62, v49, 0, s9                          // 0000000072d4: d501003e 00250131
	s_branch 61597                                             // 0000000072dc: bfa0f09d <tessera_rocm_folded_matmul_a629485a5b78541c+0x1a54>
	v_cvt_f64_f32_e32 v[49:50], v52                            // 0000000072e0: 7e622134
	v_cvt_f64_f32_e32 v[112:113], v90                          // 0000000072e4: 7ee0215a
	v_cvt_f64_f32_e32 v[118:119], v64                          // 0000000072e8: 7eec2140
	v_cmp_eq_f32_e64 s9, 0, v52                                // 0000000072ec: d4120009 02026880
	v_cmp_class_f32_e64 s11, v64, 0x1f8                        // 0000000072f4: d47e000b 0201ff40 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007300: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[112:113]               // 000000007304: 0c62e131
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007308: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[118:119]               // 00000000730c: 0c62ed31
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 000000007310: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 000000007314: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007318: bf870001
	v_cndmask_b32_e64 v51, v49, 0, s9                          // 00000000731c: d5010033 00250131
	s_branch 61591                                             // 000000007324: bfa0f097 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1a84>
	v_cvt_f64_f32_e32 v[49:50], v53                            // 000000007328: 7e622135
	v_cvt_f64_f32_e32 v[63:64], v90                            // 00000000732c: 7e7e215a
	v_cvt_f64_f32_e32 v[112:113], v57                          // 000000007330: 7ee02139
	v_cmp_eq_f32_e64 s9, 0, v53                                // 000000007334: d4120009 02026a80
	v_cmp_class_f32_e64 s11, v57, 0x1f8                        // 00000000733c: d47e000b 0201ff39 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007348: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[63:64]                 // 00000000734c: 0c627f31
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007350: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[112:113]               // 000000007354: 0c62e131
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 000000007358: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 00000000735c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007360: bf870001
	v_cndmask_b32_e64 v52, v49, 0, s9                          // 000000007364: d5010034 00250131
	s_branch 61585                                             // 00000000736c: bfa0f091 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1ab4>
	v_cvt_f64_f32_e32 v[49:50], v54                            // 000000007370: 7e622136
	v_cvt_f64_f32_e32 v[63:64], v90                            // 000000007374: 7e7e215a
	v_cvt_f64_f32_e32 v[112:113], v58                          // 000000007378: 7ee0213a
	v_cmp_eq_f32_e64 s9, 0, v54                                // 00000000737c: d4120009 02026c80
	v_cmp_class_f32_e64 s11, v58, 0x1f8                        // 000000007384: d47e000b 0201ff3a 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007390: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[63:64]                 // 000000007394: 0c627f31
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007398: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[112:113]               // 00000000739c: 0c62e131
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 0000000073a0: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000073a8: bf870001
	v_cndmask_b32_e64 v53, v49, 0, s9                          // 0000000073ac: d5010035 00250131
	s_branch 61579                                             // 0000000073b4: bfa0f08b <tessera_rocm_folded_matmul_a629485a5b78541c+0x1ae4>
	v_cvt_f64_f32_e32 v[49:50], v55                            // 0000000073b8: 7e622137
	v_cvt_f64_f32_e32 v[57:58], v90                            // 0000000073bc: 7e72215a
	v_cvt_f64_f32_e32 v[63:64], v59                            // 0000000073c0: 7e7e213b
	v_cmp_eq_f32_e64 s9, 0, v55                                // 0000000073c4: d4120009 02026e80
	v_cmp_class_f32_e64 s11, v59, 0x1f8                        // 0000000073cc: d47e000b 0201ff3b 000001f8
	s_and_b32 s9, s9, s11                                      // 0000000073d8: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[57:58]                 // 0000000073dc: 0c627331
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000073e0: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[63:64]                 // 0000000073e4: 0c627f31
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 0000000073e8: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073ec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000073f0: bf870001
	v_cndmask_b32_e64 v54, v49, 0, s9                          // 0000000073f4: d5010036 00250131
	s_branch 61573                                             // 0000000073fc: bfa0f085 <tessera_rocm_folded_matmul_a629485a5b78541c+0x1b14>
	v_cvt_f64_f32_e32 v[49:50], v56                            // 000000007400: 7e622138
	v_cvt_f64_f32_e32 v[57:58], v90                            // 000000007404: 7e72215a
	v_cvt_f64_f32_e32 v[63:64], v60                            // 000000007408: 7e7e213c
	v_cmp_eq_f32_e64 s9, 0, v56                                // 00000000740c: d4120009 02027080
	v_cmp_class_f32_e64 s11, v60, 0x1f8                        // 000000007414: d47e000b 0201ff3c 000001f8
	s_and_b32 s9, s9, s11                                      // 000000007420: 8b090b09
	v_mul_f64_e32 v[49:50], v[49:50], v[57:58]                 // 000000007424: 0c627331
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007428: bf870091
	v_mul_f64_e32 v[49:50], v[49:50], v[63:64]                 // 00000000742c: 0c627f31
	v_cvt_f32_f64_e32 v49, v[49:50]                            // 000000007430: 7e621f31
	s_wait_alu depctr_sa_sdst(0)                               // 000000007434: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007438: bf870001
	v_cndmask_b32_e64 v55, v49, 0, s9                          // 00000000743c: d5010037 00250131
	s_branch 61567                                             // 000000007444: bfa0f07f <tessera_rocm_folded_matmul_a629485a5b78541c+0x1b44>
	v_cvt_f64_f32_e32 v[122:123], v41                          // 000000007448: 7ef42129
	v_cvt_f64_f32_e32 v[128:129], v84                          // 00000000744c: 7f002154
	v_cvt_f64_f32_e32 v[130:131], v53                          // 000000007450: 7f042135
	v_cmp_eq_f32_e64 s18, 0, v41                               // 000000007454: d4120012 02025280
	v_cmp_class_f32_e64 s20, v53, 0x1f8                        // 00000000745c: d47e0014 0201ff35 000001f8
	s_and_b32 s18, s18, s20                                    // 000000007468: 8b121412
	v_mul_f64_e32 v[122:123], v[122:123], v[128:129]           // 00000000746c: 0cf5017a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007470: bf870091
	v_mul_f64_e32 v[122:123], v[122:123], v[130:131]           // 000000007474: 0cf5057a
	v_cvt_f32_f64_e32 v64, v[122:123]                          // 000000007478: 7e801f7a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000747c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007480: bf870001
	v_cndmask_b32_e64 v64, v64, 0, s18                         // 000000007484: d5010040 00490140
	s_branch 62050                                             // 00000000748c: bfa0f262 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2318>
	v_cvt_f64_f32_e32 v[122:123], v42                          // 000000007490: 7ef4212a
	v_cvt_f64_f32_e32 v[128:129], v84                          // 000000007494: 7f002154
	v_cvt_f64_f32_e32 v[130:131], v54                          // 000000007498: 7f042136
	v_cmp_eq_f32_e64 s18, 0, v42                               // 00000000749c: d4120012 02025480
	v_cmp_class_f32_e64 s20, v54, 0x1f8                        // 0000000074a4: d47e0014 0201ff36 000001f8
	s_and_b32 s18, s18, s20                                    // 0000000074b0: 8b121412
	v_mul_f64_e32 v[122:123], v[122:123], v[128:129]           // 0000000074b4: 0cf5017a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000074b8: bf870091
	v_mul_f64_e32 v[122:123], v[122:123], v[130:131]           // 0000000074bc: 0cf5057a
	v_cvt_f32_f64_e32 v41, v[122:123]                          // 0000000074c0: 7e521f7a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000074c8: bf870001
	v_cndmask_b32_e64 v53, v41, 0, s18                         // 0000000074cc: d5010035 00490129
	s_branch 62044                                             // 0000000074d4: bfa0f25c <tessera_rocm_folded_matmul_a629485a5b78541c+0x2348>
	v_cvt_f64_f32_e32 v[41:42], v43                            // 0000000074d8: 7e52212b
	v_cvt_f64_f32_e32 v[122:123], v84                          // 0000000074dc: 7ef42154
	v_cvt_f64_f32_e32 v[128:129], v55                          // 0000000074e0: 7f002137
	v_cmp_eq_f32_e64 s18, 0, v43                               // 0000000074e4: d4120012 02025680
	v_cmp_class_f32_e64 s20, v55, 0x1f8                        // 0000000074ec: d47e0014 0201ff37 000001f8
	s_and_b32 s18, s18, s20                                    // 0000000074f8: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[122:123]               // 0000000074fc: 0c52f529
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007500: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[128:129]               // 000000007504: 0c530129
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000007508: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 00000000750c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007510: bf870001
	v_cndmask_b32_e64 v54, v41, 0, s18                         // 000000007514: d5010036 00490129
	s_branch 62038                                             // 00000000751c: bfa0f256 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2378>
	v_cvt_f64_f32_e32 v[41:42], v44                            // 000000007520: 7e52212c
	v_cvt_f64_f32_e32 v[122:123], v84                          // 000000007524: 7ef42154
	v_cvt_f64_f32_e32 v[128:129], v56                          // 000000007528: 7f002138
	v_cmp_eq_f32_e64 s18, 0, v44                               // 00000000752c: d4120012 02025880
	v_cmp_class_f32_e64 s20, v56, 0x1f8                        // 000000007534: d47e0014 0201ff38 000001f8
	s_and_b32 s18, s18, s20                                    // 000000007540: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[122:123]               // 000000007544: 0c52f529
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007548: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[128:129]               // 00000000754c: 0c530129
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000007550: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 000000007554: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007558: bf870001
	v_cndmask_b32_e64 v43, v41, 0, s18                         // 00000000755c: d501002b 00490129
	s_branch 62032                                             // 000000007564: bfa0f250 <tessera_rocm_folded_matmul_a629485a5b78541c+0x23a8>
	v_cvt_f64_f32_e32 v[41:42], v45                            // 000000007568: 7e52212d
	v_cvt_f64_f32_e32 v[55:56], v84                            // 00000000756c: 7e6e2154
	v_cvt_f64_f32_e32 v[122:123], v49                          // 000000007570: 7ef42131
	v_cmp_eq_f32_e64 s18, 0, v45                               // 000000007574: d4120012 02025a80
	v_cmp_class_f32_e64 s20, v49, 0x1f8                        // 00000000757c: d47e0014 0201ff31 000001f8
	s_and_b32 s18, s18, s20                                    // 000000007588: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[55:56]                 // 00000000758c: 0c526f29
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007590: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[122:123]               // 000000007594: 0c52f529
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000007598: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 00000000759c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000075a0: bf870001
	v_cndmask_b32_e64 v44, v41, 0, s18                         // 0000000075a4: d501002c 00490129
	s_branch 62026                                             // 0000000075ac: bfa0f24a <tessera_rocm_folded_matmul_a629485a5b78541c+0x23d8>
	v_cvt_f64_f32_e32 v[41:42], v46                            // 0000000075b0: 7e52212e
	v_cvt_f64_f32_e32 v[55:56], v84                            // 0000000075b4: 7e6e2154
	v_cvt_f64_f32_e32 v[122:123], v50                          // 0000000075b8: 7ef42132
	v_cmp_eq_f32_e64 s18, 0, v46                               // 0000000075bc: d4120012 02025c80
	v_cmp_class_f32_e64 s20, v50, 0x1f8                        // 0000000075c4: d47e0014 0201ff32 000001f8
	s_and_b32 s18, s18, s20                                    // 0000000075d0: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[55:56]                 // 0000000075d4: 0c526f29
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000075d8: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[122:123]               // 0000000075dc: 0c52f529
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 0000000075e0: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000075e8: bf870001
	v_cndmask_b32_e64 v45, v41, 0, s18                         // 0000000075ec: d501002d 00490129
	s_branch 62020                                             // 0000000075f4: bfa0f244 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2408>
	v_cvt_f64_f32_e32 v[41:42], v47                            // 0000000075f8: 7e52212f
	v_cvt_f64_f32_e32 v[49:50], v84                            // 0000000075fc: 7e622154
	v_cvt_f64_f32_e32 v[55:56], v51                            // 000000007600: 7e6e2133
	v_cmp_eq_f32_e64 s18, 0, v47                               // 000000007604: d4120012 02025e80
	v_cmp_class_f32_e64 s20, v51, 0x1f8                        // 00000000760c: d47e0014 0201ff33 000001f8
	s_and_b32 s18, s18, s20                                    // 000000007618: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[49:50]                 // 00000000761c: 0c526329
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007620: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[55:56]                 // 000000007624: 0c526f29
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000007628: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 00000000762c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007630: bf870001
	v_cndmask_b32_e64 v46, v41, 0, s18                         // 000000007634: d501002e 00490129
	s_branch 62014                                             // 00000000763c: bfa0f23e <tessera_rocm_folded_matmul_a629485a5b78541c+0x2438>
	v_cvt_f64_f32_e32 v[41:42], v48                            // 000000007640: 7e522130
	v_cvt_f64_f32_e32 v[49:50], v84                            // 000000007644: 7e622154
	v_cvt_f64_f32_e32 v[55:56], v52                            // 000000007648: 7e6e2134
	v_cmp_eq_f32_e64 s18, 0, v48                               // 00000000764c: d4120012 02026080
	v_cmp_class_f32_e64 s20, v52, 0x1f8                        // 000000007654: d47e0014 0201ff34 000001f8
	s_and_b32 s18, s18, s20                                    // 000000007660: 8b121412
	v_mul_f64_e32 v[41:42], v[41:42], v[49:50]                 // 000000007664: 0c526329
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007668: bf870091
	v_mul_f64_e32 v[41:42], v[41:42], v[55:56]                 // 00000000766c: 0c526f29
	v_cvt_f32_f64_e32 v41, v[41:42]                            // 000000007670: 7e521f29
	s_wait_alu depctr_sa_sdst(0)                               // 000000007674: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007678: bf870001
	v_cndmask_b32_e64 v47, v41, 0, s18                         // 00000000767c: d501002f 00490129
	s_branch 62008                                             // 000000007684: bfa0f238 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2468>
	v_cvt_f64_f32_e32 v[139:140], v33                          // 000000007688: 7f162121
	v_cvt_f64_f32_e32 v[143:144], v64                          // 00000000768c: 7f1e2140
	v_cvt_f64_f32_e32 v[145:146], v45                          // 000000007690: 7f22212d
	v_cmp_eq_f32_e64 s27, 0, v33                               // 000000007694: d412001b 02024280
	v_cmp_class_f32_e64 s29, v45, 0x1f8                        // 00000000769c: d47e001d 0201ff2d 000001f8
	s_and_b32 s27, s27, s29                                    // 0000000076a8: 8b1b1d1b
	v_mul_f64_e32 v[139:140], v[139:140], v[143:144]           // 0000000076ac: 0d171f8b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000076b0: bf870091
	v_mul_f64_e32 v[139:140], v[139:140], v[145:146]           // 0000000076b4: 0d17238b
	v_cvt_f32_f64_e32 v56, v[139:140]                          // 0000000076b8: 7e701f8b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000076c0: bf870001
	v_cndmask_b32_e64 v56, v56, 0, s27                         // 0000000076c4: d5010038 006d0138
	s_branch 62491                                             // 0000000076cc: bfa0f41b <tessera_rocm_folded_matmul_a629485a5b78541c+0x2c3c>
	v_cvt_f64_f32_e32 v[139:140], v34                          // 0000000076d0: 7f162122
	v_cvt_f64_f32_e32 v[143:144], v64                          // 0000000076d4: 7f1e2140
	v_cvt_f64_f32_e32 v[145:146], v46                          // 0000000076d8: 7f22212e
	v_cmp_eq_f32_e64 s27, 0, v34                               // 0000000076dc: d412001b 02024480
	v_cmp_class_f32_e64 s29, v46, 0x1f8                        // 0000000076e4: d47e001d 0201ff2e 000001f8
	s_and_b32 s27, s27, s29                                    // 0000000076f0: 8b1b1d1b
	v_mul_f64_e32 v[139:140], v[139:140], v[143:144]           // 0000000076f4: 0d171f8b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000076f8: bf870091
	v_mul_f64_e32 v[139:140], v[139:140], v[145:146]           // 0000000076fc: 0d17238b
	v_cvt_f32_f64_e32 v33, v[139:140]                          // 000000007700: 7e421f8b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007704: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007708: bf870001
	v_cndmask_b32_e64 v45, v33, 0, s27                         // 00000000770c: d501002d 006d0121
	s_branch 62485                                             // 000000007714: bfa0f415 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2c6c>
	v_cvt_f64_f32_e32 v[33:34], v35                            // 000000007718: 7e422123
	v_cvt_f64_f32_e32 v[139:140], v64                          // 00000000771c: 7f162140
	v_cvt_f64_f32_e32 v[143:144], v47                          // 000000007720: 7f1e212f
	v_cmp_eq_f32_e64 s27, 0, v35                               // 000000007724: d412001b 02024680
	v_cmp_class_f32_e64 s29, v47, 0x1f8                        // 00000000772c: d47e001d 0201ff2f 000001f8
	s_and_b32 s27, s27, s29                                    // 000000007738: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[139:140]               // 00000000773c: 0c431721
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007740: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[143:144]               // 000000007744: 0c431f21
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000007748: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 00000000774c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007750: bf870001
	v_cndmask_b32_e64 v46, v33, 0, s27                         // 000000007754: d501002e 006d0121
	s_branch 62479                                             // 00000000775c: bfa0f40f <tessera_rocm_folded_matmul_a629485a5b78541c+0x2c9c>
	v_cvt_f64_f32_e32 v[33:34], v36                            // 000000007760: 7e422124
	v_cvt_f64_f32_e32 v[139:140], v64                          // 000000007764: 7f162140
	v_cvt_f64_f32_e32 v[143:144], v48                          // 000000007768: 7f1e2130
	v_cmp_eq_f32_e64 s27, 0, v36                               // 00000000776c: d412001b 02024880
	v_cmp_class_f32_e64 s29, v48, 0x1f8                        // 000000007774: d47e001d 0201ff30 000001f8
	s_and_b32 s27, s27, s29                                    // 000000007780: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[139:140]               // 000000007784: 0c431721
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007788: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[143:144]               // 00000000778c: 0c431f21
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000007790: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000007794: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007798: bf870001
	v_cndmask_b32_e64 v35, v33, 0, s27                         // 00000000779c: d5010023 006d0121
	s_branch 62473                                             // 0000000077a4: bfa0f409 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2ccc>
	v_cvt_f64_f32_e32 v[33:34], v37                            // 0000000077a8: 7e422125
	v_cvt_f64_f32_e32 v[47:48], v64                            // 0000000077ac: 7e5e2140
	v_cvt_f64_f32_e32 v[139:140], v41                          // 0000000077b0: 7f162129
	v_cmp_eq_f32_e64 s27, 0, v37                               // 0000000077b4: d412001b 02024a80
	v_cmp_class_f32_e64 s29, v41, 0x1f8                        // 0000000077bc: d47e001d 0201ff29 000001f8
	s_and_b32 s27, s27, s29                                    // 0000000077c8: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[47:48]                 // 0000000077cc: 0c425f21
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000077d0: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[139:140]               // 0000000077d4: 0c431721
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 0000000077d8: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 0000000077dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000077e0: bf870001
	v_cndmask_b32_e64 v36, v33, 0, s27                         // 0000000077e4: d5010024 006d0121
	s_branch 62467                                             // 0000000077ec: bfa0f403 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2cfc>
	v_cvt_f64_f32_e32 v[33:34], v38                            // 0000000077f0: 7e422126
	v_cvt_f64_f32_e32 v[47:48], v64                            // 0000000077f4: 7e5e2140
	v_cvt_f64_f32_e32 v[139:140], v42                          // 0000000077f8: 7f16212a
	v_cmp_eq_f32_e64 s27, 0, v38                               // 0000000077fc: d412001b 02024c80
	v_cmp_class_f32_e64 s29, v42, 0x1f8                        // 000000007804: d47e001d 0201ff2a 000001f8
	s_and_b32 s27, s27, s29                                    // 000000007810: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[47:48]                 // 000000007814: 0c425f21
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007818: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[139:140]               // 00000000781c: 0c431721
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000007820: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000007824: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007828: bf870001
	v_cndmask_b32_e64 v37, v33, 0, s27                         // 00000000782c: d5010025 006d0121
	s_branch 62461                                             // 000000007834: bfa0f3fd <tessera_rocm_folded_matmul_a629485a5b78541c+0x2d2c>
	v_cvt_f64_f32_e32 v[33:34], v39                            // 000000007838: 7e422127
	v_cvt_f64_f32_e32 v[41:42], v64                            // 00000000783c: 7e522140
	v_cvt_f64_f32_e32 v[47:48], v43                            // 000000007840: 7e5e212b
	v_cmp_eq_f32_e64 s27, 0, v39                               // 000000007844: d412001b 02024e80
	v_cmp_class_f32_e64 s29, v43, 0x1f8                        // 00000000784c: d47e001d 0201ff2b 000001f8
	s_and_b32 s27, s27, s29                                    // 000000007858: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[41:42]                 // 00000000785c: 0c425321
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007860: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[47:48]                 // 000000007864: 0c425f21
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 000000007868: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 00000000786c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007870: bf870001
	v_cndmask_b32_e64 v38, v33, 0, s27                         // 000000007874: d5010026 006d0121
	s_branch 62455                                             // 00000000787c: bfa0f3f7 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2d5c>
	v_cvt_f64_f32_e32 v[33:34], v40                            // 000000007880: 7e422128
	v_cvt_f64_f32_e32 v[41:42], v64                            // 000000007884: 7e522140
	v_cvt_f64_f32_e32 v[47:48], v44                            // 000000007888: 7e5e212c
	v_cmp_eq_f32_e64 s27, 0, v40                               // 00000000788c: d412001b 02025080
	v_cmp_class_f32_e64 s29, v44, 0x1f8                        // 000000007894: d47e001d 0201ff2c 000001f8
	s_and_b32 s27, s27, s29                                    // 0000000078a0: 8b1b1d1b
	v_mul_f64_e32 v[33:34], v[33:34], v[41:42]                 // 0000000078a4: 0c425321
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000078a8: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[47:48]                 // 0000000078ac: 0c425f21
	v_cvt_f32_f64_e32 v33, v[33:34]                            // 0000000078b0: 7e421f21
	s_wait_alu depctr_sa_sdst(0)                               // 0000000078b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000078b8: bf870001
	v_cndmask_b32_e64 v39, v33, 0, s27                         // 0000000078bc: d5010027 006d0121
	s_branch 62449                                             // 0000000078c4: bfa0f3f1 <tessera_rocm_folded_matmul_a629485a5b78541c+0x2d8c>
	v_cvt_f64_f32_e32 v[45:46], v25                            // 0000000078c8: 7e5a2119
	v_cvt_f64_f32_e32 v[47:48], v0                             // 0000000078cc: 7e5e2100
	v_cvt_f64_f32_e32 v[81:82], v37                            // 0000000078d0: 7ea22125
	v_cmp_eq_f32_e64 s37, 0, v25                               // 0000000078d4: d4120025 02023280
	v_cmp_class_f32_e64 s43, v37, 0x1f8                        // 0000000078dc: d47e002b 0201ff25 000001f8
	s_and_b32 s37, s37, s43                                    // 0000000078e8: 8b252b25
	v_mul_f64_e32 v[45:46], v[45:46], v[47:48]                 // 0000000078ec: 0c5a5f2d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000078f0: bf870091
	v_mul_f64_e32 v[45:46], v[45:46], v[81:82]                 // 0000000078f4: 0c5aa32d
	v_cvt_f32_f64_e32 v45, v[45:46]                            // 0000000078f8: 7e5a1f2d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000078fc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007900: bf870001
	v_cndmask_b32_e64 v45, v45, 0, s37                         // 000000007904: d501002d 0095012d
	s_branch 62914                                             // 00000000790c: bfa0f5c2 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3518>
	v_cvt_f64_f32_e32 v[46:47], v26                            // 000000007910: 7e5c211a
	v_cvt_f64_f32_e32 v[81:82], v0                             // 000000007914: 7ea22100
	v_cvt_f64_f32_e32 v[83:84], v38                            // 000000007918: 7ea62126
	v_cmp_eq_f32_e64 s37, 0, v26                               // 00000000791c: d4120025 02023480
	v_cmp_class_f32_e64 s43, v38, 0x1f8                        // 000000007924: d47e002b 0201ff26 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007930: 8b252b25
	v_mul_f64_e32 v[46:47], v[46:47], v[81:82]                 // 000000007934: 0c5ca32e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007938: bf870091
	v_mul_f64_e32 v[46:47], v[46:47], v[83:84]                 // 00000000793c: 0c5ca72e
	v_cvt_f32_f64_e32 v25, v[46:47]                            // 000000007940: 7e321f2e
	s_wait_alu depctr_sa_sdst(0)                               // 000000007944: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007948: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s37                         // 00000000794c: d5010019 00950119
	s_branch 62908                                             // 000000007954: bfa0f5bc <tessera_rocm_folded_matmul_a629485a5b78541c+0x3548>
	v_cvt_f64_f32_e32 v[37:38], v27                            // 000000007958: 7e4a211b
	v_cvt_f64_f32_e32 v[46:47], v0                             // 00000000795c: 7e5c2100
	v_cvt_f64_f32_e32 v[81:82], v39                            // 000000007960: 7ea22127
	v_cmp_eq_f32_e64 s37, 0, v27                               // 000000007964: d4120025 02023680
	v_cmp_class_f32_e64 s43, v39, 0x1f8                        // 00000000796c: d47e002b 0201ff27 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007978: 8b252b25
	v_mul_f64_e32 v[37:38], v[37:38], v[46:47]                 // 00000000797c: 0c4a5d25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007980: bf870091
	v_mul_f64_e32 v[37:38], v[37:38], v[81:82]                 // 000000007984: 0c4aa325
	v_cvt_f32_f64_e32 v26, v[37:38]                            // 000000007988: 7e341f25
	s_wait_alu depctr_sa_sdst(0)                               // 00000000798c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007990: bf870001
	v_cndmask_b32_e64 v26, v26, 0, s37                         // 000000007994: d501001a 0095011a
	s_branch 62902                                             // 00000000799c: bfa0f5b6 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3578>
	v_cvt_f64_f32_e32 v[37:38], v28                            // 0000000079a0: 7e4a211c
	v_cvt_f64_f32_e32 v[46:47], v0                             // 0000000079a4: 7e5c2100
	v_cvt_f64_f32_e32 v[81:82], v40                            // 0000000079a8: 7ea22128
	v_cmp_eq_f32_e64 s37, 0, v28                               // 0000000079ac: d4120025 02023880
	v_cmp_class_f32_e64 s43, v40, 0x1f8                        // 0000000079b4: d47e002b 0201ff28 000001f8
	s_and_b32 s37, s37, s43                                    // 0000000079c0: 8b252b25
	v_mul_f64_e32 v[37:38], v[37:38], v[46:47]                 // 0000000079c4: 0c4a5d25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000079c8: bf870091
	v_mul_f64_e32 v[37:38], v[37:38], v[81:82]                 // 0000000079cc: 0c4aa325
	v_cvt_f32_f64_e32 v27, v[37:38]                            // 0000000079d0: 7e361f25
	s_wait_alu depctr_sa_sdst(0)                               // 0000000079d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000079d8: bf870001
	v_cndmask_b32_e64 v27, v27, 0, s37                         // 0000000079dc: d501001b 0095011b
	s_branch 62896                                             // 0000000079e4: bfa0f5b0 <tessera_rocm_folded_matmul_a629485a5b78541c+0x35a8>
	v_cvt_f64_f32_e32 v[37:38], v29                            // 0000000079e8: 7e4a211d
	v_cvt_f64_f32_e32 v[39:40], v0                             // 0000000079ec: 7e4e2100
	v_cvt_f64_f32_e32 v[46:47], v33                            // 0000000079f0: 7e5c2121
	v_cmp_eq_f32_e64 s37, 0, v29                               // 0000000079f4: d4120025 02023a80
	v_cmp_class_f32_e64 s43, v33, 0x1f8                        // 0000000079fc: d47e002b 0201ff21 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007a08: 8b252b25
	v_mul_f64_e32 v[37:38], v[37:38], v[39:40]                 // 000000007a0c: 0c4a4f25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007a10: bf870091
	v_mul_f64_e32 v[37:38], v[37:38], v[46:47]                 // 000000007a14: 0c4a5d25
	v_cvt_f32_f64_e32 v28, v[37:38]                            // 000000007a18: 7e381f25
	s_wait_alu depctr_sa_sdst(0)                               // 000000007a1c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007a20: bf870001
	v_cndmask_b32_e64 v28, v28, 0, s37                         // 000000007a24: d501001c 0095011c
	s_branch 62890                                             // 000000007a2c: bfa0f5aa <tessera_rocm_folded_matmul_a629485a5b78541c+0x35d8>
	v_cvt_f64_f32_e32 v[37:38], v30                            // 000000007a30: 7e4a211e
	v_cvt_f64_f32_e32 v[39:40], v0                             // 000000007a34: 7e4e2100
	v_cvt_f64_f32_e32 v[46:47], v34                            // 000000007a38: 7e5c2122
	v_cmp_eq_f32_e64 s37, 0, v30                               // 000000007a3c: d4120025 02023c80
	v_cmp_class_f32_e64 s43, v34, 0x1f8                        // 000000007a44: d47e002b 0201ff22 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007a50: 8b252b25
	v_mul_f64_e32 v[37:38], v[37:38], v[39:40]                 // 000000007a54: 0c4a4f25
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007a58: bf870091
	v_mul_f64_e32 v[37:38], v[37:38], v[46:47]                 // 000000007a5c: 0c4a5d25
	v_cvt_f32_f64_e32 v29, v[37:38]                            // 000000007a60: 7e3a1f25
	s_wait_alu depctr_sa_sdst(0)                               // 000000007a64: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007a68: bf870001
	v_cndmask_b32_e64 v29, v29, 0, s37                         // 000000007a6c: d501001d 0095011d
	s_branch 62884                                             // 000000007a74: bfa0f5a4 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3608>
	v_cvt_f64_f32_e32 v[33:34], v31                            // 000000007a78: 7e42211f
	v_cvt_f64_f32_e32 v[37:38], v0                             // 000000007a7c: 7e4a2100
	v_cvt_f64_f32_e32 v[39:40], v35                            // 000000007a80: 7e4e2123
	v_cmp_eq_f32_e64 s37, 0, v31                               // 000000007a84: d4120025 02023e80
	v_cmp_class_f32_e64 s43, v35, 0x1f8                        // 000000007a8c: d47e002b 0201ff23 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007a98: 8b252b25
	v_mul_f64_e32 v[33:34], v[33:34], v[37:38]                 // 000000007a9c: 0c424b21
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007aa0: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[39:40]                 // 000000007aa4: 0c424f21
	v_cvt_f32_f64_e32 v30, v[33:34]                            // 000000007aa8: 7e3c1f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000007aac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007ab0: bf870001
	v_cndmask_b32_e64 v30, v30, 0, s37                         // 000000007ab4: d501001e 0095011e
	s_branch 62878                                             // 000000007abc: bfa0f59e <tessera_rocm_folded_matmul_a629485a5b78541c+0x3638>
	v_cvt_f64_f32_e32 v[33:34], v32                            // 000000007ac0: 7e422120
	v_cvt_f64_f32_e32 v[37:38], v0                             // 000000007ac4: 7e4a2100
	v_cvt_f64_f32_e32 v[39:40], v36                            // 000000007ac8: 7e4e2124
	v_cmp_eq_f32_e64 s37, 0, v32                               // 000000007acc: d4120025 02024080
	v_cmp_class_f32_e64 s43, v36, 0x1f8                        // 000000007ad4: d47e002b 0201ff24 000001f8
	s_and_b32 s37, s37, s43                                    // 000000007ae0: 8b252b25
	v_mul_f64_e32 v[33:34], v[33:34], v[37:38]                 // 000000007ae4: 0c424b21
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007ae8: bf870091
	v_mul_f64_e32 v[33:34], v[33:34], v[39:40]                 // 000000007aec: 0c424f21
	v_cvt_f32_f64_e32 v0, v[33:34]                             // 000000007af0: 7e001f21
	s_wait_alu depctr_sa_sdst(0)                               // 000000007af4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007af8: bf870001
	v_cndmask_b32_e64 v31, v0, 0, s37                          // 000000007afc: d501001f 00950100
	s_branch 62872                                             // 000000007b04: bfa0f598 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3668>
	v_cvt_f64_f32_e32 v[35:36], v17                            // 000000007b08: 7e462111
	v_cvt_f64_f32_e32 v[37:38], v34                            // 000000007b0c: 7e4a2122
	v_cvt_f64_f32_e32 v[39:40], v29                            // 000000007b10: 7e4e211d
	v_cmp_eq_f32_e64 s0, 0, v17                                // 000000007b14: d4120000 02022280
	v_cmp_class_f32_e64 s2, v29, 0x1f8                         // 000000007b1c: d47e0002 0201ff1d 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007b28: 8b000200
	v_mul_f64_e32 v[35:36], v[35:36], v[37:38]                 // 000000007b2c: 0c464b23
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007b30: bf870091
	v_mul_f64_e32 v[35:36], v[35:36], v[39:40]                 // 000000007b34: 0c464f23
	v_cvt_f32_f64_e32 v33, v[35:36]                            // 000000007b38: 7e421f23
	s_wait_alu depctr_sa_sdst(0)                               // 000000007b3c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007b40: bf870001
	v_cndmask_b32_e64 v33, v33, 0, s0                          // 000000007b44: d5010021 00010121
	s_branch 63326                                             // 000000007b4c: bfa0f75e <tessera_rocm_folded_matmul_a629485a5b78541c+0x3dc8>
	v_cvt_f64_f32_e32 v[35:36], v18                            // 000000007b50: 7e462112
	v_cvt_f64_f32_e32 v[37:38], v34                            // 000000007b54: 7e4a2122
	v_cvt_f64_f32_e32 v[39:40], v30                            // 000000007b58: 7e4e211e
	v_cmp_eq_f32_e64 s0, 0, v18                                // 000000007b5c: d4120000 02022480
	v_cmp_class_f32_e64 s2, v30, 0x1f8                         // 000000007b64: d47e0002 0201ff1e 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007b70: 8b000200
	v_mul_f64_e32 v[35:36], v[35:36], v[37:38]                 // 000000007b74: 0c464b23
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007b78: bf870091
	v_mul_f64_e32 v[35:36], v[35:36], v[39:40]                 // 000000007b7c: 0c464f23
	v_cvt_f32_f64_e32 v17, v[35:36]                            // 000000007b80: 7e221f23
	s_wait_alu depctr_sa_sdst(0)                               // 000000007b84: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007b88: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s0                          // 000000007b8c: d5010011 00010111
	s_branch 63320                                             // 000000007b94: bfa0f758 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3df8>
	v_cvt_f64_f32_e32 v[29:30], v19                            // 000000007b98: 7e3a2113
	v_cvt_f64_f32_e32 v[35:36], v34                            // 000000007b9c: 7e462122
	v_cvt_f64_f32_e32 v[37:38], v31                            // 000000007ba0: 7e4a211f
	v_cmp_eq_f32_e64 s0, 0, v19                                // 000000007ba4: d4120000 02022680
	v_cmp_class_f32_e64 s2, v31, 0x1f8                         // 000000007bac: d47e0002 0201ff1f 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007bb8: 8b000200
	v_mul_f64_e32 v[29:30], v[29:30], v[35:36]                 // 000000007bbc: 0c3a471d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007bc0: bf870091
	v_mul_f64_e32 v[29:30], v[29:30], v[37:38]                 // 000000007bc4: 0c3a4b1d
	v_cvt_f32_f64_e32 v18, v[29:30]                            // 000000007bc8: 7e241f1d
	s_wait_alu depctr_sa_sdst(0)                               // 000000007bcc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007bd0: bf870001
	v_cndmask_b32_e64 v18, v18, 0, s0                          // 000000007bd4: d5010012 00010112
	s_branch 63314                                             // 000000007bdc: bfa0f752 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3e28>
	v_cvt_f64_f32_e32 v[29:30], v20                            // 000000007be0: 7e3a2114
	v_cvt_f64_f32_e32 v[35:36], v34                            // 000000007be4: 7e462122
	v_cvt_f64_f32_e32 v[37:38], v32                            // 000000007be8: 7e4a2120
	v_cmp_eq_f32_e64 s0, 0, v20                                // 000000007bec: d4120000 02022880
	v_cmp_class_f32_e64 s2, v32, 0x1f8                         // 000000007bf4: d47e0002 0201ff20 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007c00: 8b000200
	v_mul_f64_e32 v[29:30], v[29:30], v[35:36]                 // 000000007c04: 0c3a471d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007c08: bf870091
	v_mul_f64_e32 v[29:30], v[29:30], v[37:38]                 // 000000007c0c: 0c3a4b1d
	v_cvt_f32_f64_e32 v19, v[29:30]                            // 000000007c10: 7e261f1d
	s_wait_alu depctr_sa_sdst(0)                               // 000000007c14: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007c18: bf870001
	v_cndmask_b32_e64 v19, v19, 0, s0                          // 000000007c1c: d5010013 00010113
	s_branch 63308                                             // 000000007c24: bfa0f74c <tessera_rocm_folded_matmul_a629485a5b78541c+0x3e58>
	v_cvt_f64_f32_e32 v[29:30], v21                            // 000000007c28: 7e3a2115
	v_cvt_f64_f32_e32 v[31:32], v34                            // 000000007c2c: 7e3e2122
	v_cvt_f64_f32_e32 v[35:36], v25                            // 000000007c30: 7e462119
	v_cmp_eq_f32_e64 s0, 0, v21                                // 000000007c34: d4120000 02022a80
	v_cmp_class_f32_e64 s2, v25, 0x1f8                         // 000000007c3c: d47e0002 0201ff19 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007c48: 8b000200
	v_mul_f64_e32 v[29:30], v[29:30], v[31:32]                 // 000000007c4c: 0c3a3f1d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007c50: bf870091
	v_mul_f64_e32 v[29:30], v[29:30], v[35:36]                 // 000000007c54: 0c3a471d
	v_cvt_f32_f64_e32 v20, v[29:30]                            // 000000007c58: 7e281f1d
	s_wait_alu depctr_sa_sdst(0)                               // 000000007c5c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007c60: bf870001
	v_cndmask_b32_e64 v20, v20, 0, s0                          // 000000007c64: d5010014 00010114
	s_branch 63302                                             // 000000007c6c: bfa0f746 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3e88>
	v_cvt_f64_f32_e32 v[29:30], v22                            // 000000007c70: 7e3a2116
	v_cvt_f64_f32_e32 v[31:32], v34                            // 000000007c74: 7e3e2122
	v_cvt_f64_f32_e32 v[35:36], v26                            // 000000007c78: 7e46211a
	v_cmp_eq_f32_e64 s0, 0, v22                                // 000000007c7c: d4120000 02022c80
	v_cmp_class_f32_e64 s2, v26, 0x1f8                         // 000000007c84: d47e0002 0201ff1a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007c90: 8b000200
	v_mul_f64_e32 v[29:30], v[29:30], v[31:32]                 // 000000007c94: 0c3a3f1d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007c98: bf870091
	v_mul_f64_e32 v[29:30], v[29:30], v[35:36]                 // 000000007c9c: 0c3a471d
	v_cvt_f32_f64_e32 v21, v[29:30]                            // 000000007ca0: 7e2a1f1d
	s_wait_alu depctr_sa_sdst(0)                               // 000000007ca4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007ca8: bf870001
	v_cndmask_b32_e64 v21, v21, 0, s0                          // 000000007cac: d5010015 00010115
	s_branch 63296                                             // 000000007cb4: bfa0f740 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3eb8>
	v_cvt_f64_f32_e32 v[25:26], v23                            // 000000007cb8: 7e322117
	v_cvt_f64_f32_e32 v[29:30], v34                            // 000000007cbc: 7e3a2122
	v_cvt_f64_f32_e32 v[31:32], v27                            // 000000007cc0: 7e3e211b
	v_cmp_eq_f32_e64 s0, 0, v23                                // 000000007cc4: d4120000 02022e80
	v_cmp_class_f32_e64 s2, v27, 0x1f8                         // 000000007ccc: d47e0002 0201ff1b 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007cd8: 8b000200
	v_mul_f64_e32 v[25:26], v[25:26], v[29:30]                 // 000000007cdc: 0c323b19
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007ce0: bf870091
	v_mul_f64_e32 v[25:26], v[25:26], v[31:32]                 // 000000007ce4: 0c323f19
	v_cvt_f32_f64_e32 v22, v[25:26]                            // 000000007ce8: 7e2c1f19
	s_wait_alu depctr_sa_sdst(0)                               // 000000007cec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007cf0: bf870001
	v_cndmask_b32_e64 v22, v22, 0, s0                          // 000000007cf4: d5010016 00010116
	s_branch 63290                                             // 000000007cfc: bfa0f73a <tessera_rocm_folded_matmul_a629485a5b78541c+0x3ee8>
	v_cvt_f64_f32_e32 v[25:26], v24                            // 000000007d00: 7e322118
	v_cvt_f64_f32_e32 v[29:30], v34                            // 000000007d04: 7e3a2122
	v_cvt_f64_f32_e32 v[31:32], v28                            // 000000007d08: 7e3e211c
	v_cmp_eq_f32_e64 s0, 0, v24                                // 000000007d0c: d4120000 02023080
	v_cmp_class_f32_e64 s2, v28, 0x1f8                         // 000000007d14: d47e0002 0201ff1c 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007d20: 8b000200
	v_mul_f64_e32 v[25:26], v[25:26], v[29:30]                 // 000000007d24: 0c323b19
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007d28: bf870091
	v_mul_f64_e32 v[25:26], v[25:26], v[31:32]                 // 000000007d2c: 0c323f19
	v_cvt_f32_f64_e32 v23, v[25:26]                            // 000000007d30: 7e2e1f19
	s_wait_alu depctr_sa_sdst(0)                               // 000000007d34: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007d38: bf870001
	v_cndmask_b32_e64 v23, v23, 0, s0                          // 000000007d3c: d5010017 00010117
	s_branch 63284                                             // 000000007d44: bfa0f734 <tessera_rocm_folded_matmul_a629485a5b78541c+0x3f18>
	v_cvt_f64_f32_e32 v[27:28], v9                             // 000000007d48: 7e362109
	v_cvt_f64_f32_e32 v[29:30], v26                            // 000000007d4c: 7e3a211a
	v_cvt_f64_f32_e32 v[31:32], v21                            // 000000007d50: 7e3e2115
	v_cmp_eq_f32_e64 s0, 0, v9                                 // 000000007d54: d4120000 02021280
	v_cmp_class_f32_e64 s2, v21, 0x1f8                         // 000000007d5c: d47e0002 0201ff15 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007d68: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007d6c: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007d70: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[31:32]                 // 000000007d74: 0c363f1b
	v_cvt_f32_f64_e32 v25, v[27:28]                            // 000000007d78: 7e321f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007d7c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007d80: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s0                          // 000000007d84: d5010019 00010119
	s_branch 63737                                             // 000000007d8c: bfa0f8f9 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4674>
	v_cvt_f64_f32_e32 v[27:28], v10                            // 000000007d90: 7e36210a
	v_cvt_f64_f32_e32 v[29:30], v26                            // 000000007d94: 7e3a211a
	v_cvt_f64_f32_e32 v[31:32], v22                            // 000000007d98: 7e3e2116
	v_cmp_eq_f32_e64 s0, 0, v10                                // 000000007d9c: d4120000 02021480
	v_cmp_class_f32_e64 s2, v22, 0x1f8                         // 000000007da4: d47e0002 0201ff16 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007db0: 8b000200
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000007db4: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007db8: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[31:32]                 // 000000007dbc: 0c363f1b
	v_cvt_f32_f64_e32 v9, v[27:28]                             // 000000007dc0: 7e121f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000007dc4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007dc8: bf870001
	v_cndmask_b32_e64 v9, v9, 0, s0                            // 000000007dcc: d5010009 00010109
	s_branch 63731                                             // 000000007dd4: bfa0f8f3 <tessera_rocm_folded_matmul_a629485a5b78541c+0x46a4>
	v_cvt_f64_f32_e32 v[21:22], v11                            // 000000007dd8: 7e2a210b
	v_cvt_f64_f32_e32 v[27:28], v26                            // 000000007ddc: 7e36211a
	v_cvt_f64_f32_e32 v[29:30], v23                            // 000000007de0: 7e3a2117
	v_cmp_eq_f32_e64 s0, 0, v11                                // 000000007de4: d4120000 02021680
	v_cmp_class_f32_e64 s2, v23, 0x1f8                         // 000000007dec: d47e0002 0201ff17 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007df8: 8b000200
	v_mul_f64_e32 v[21:22], v[21:22], v[27:28]                 // 000000007dfc: 0c2a3715
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007e00: bf870091
	v_mul_f64_e32 v[21:22], v[21:22], v[29:30]                 // 000000007e04: 0c2a3b15
	v_cvt_f32_f64_e32 v10, v[21:22]                            // 000000007e08: 7e141f15
	s_wait_alu depctr_sa_sdst(0)                               // 000000007e0c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007e10: bf870001
	v_cndmask_b32_e64 v10, v10, 0, s0                          // 000000007e14: d501000a 0001010a
	s_branch 63725                                             // 000000007e1c: bfa0f8ed <tessera_rocm_folded_matmul_a629485a5b78541c+0x46d4>
	v_cvt_f64_f32_e32 v[21:22], v12                            // 000000007e20: 7e2a210c
	v_cvt_f64_f32_e32 v[27:28], v26                            // 000000007e24: 7e36211a
	v_cvt_f64_f32_e32 v[29:30], v24                            // 000000007e28: 7e3a2118
	v_cmp_eq_f32_e64 s0, 0, v12                                // 000000007e2c: d4120000 02021880
	v_cmp_class_f32_e64 s2, v24, 0x1f8                         // 000000007e34: d47e0002 0201ff18 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007e40: 8b000200
	v_mul_f64_e32 v[21:22], v[21:22], v[27:28]                 // 000000007e44: 0c2a3715
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007e48: bf870091
	v_mul_f64_e32 v[21:22], v[21:22], v[29:30]                 // 000000007e4c: 0c2a3b15
	v_cvt_f32_f64_e32 v11, v[21:22]                            // 000000007e50: 7e161f15
	s_wait_alu depctr_sa_sdst(0)                               // 000000007e54: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007e58: bf870001
	v_cndmask_b32_e64 v11, v11, 0, s0                          // 000000007e5c: d501000b 0001010b
	s_branch 63719                                             // 000000007e64: bfa0f8e7 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4704>
	v_cvt_f64_f32_e32 v[21:22], v13                            // 000000007e68: 7e2a210d
	v_cvt_f64_f32_e32 v[23:24], v26                            // 000000007e6c: 7e2e211a
	v_cvt_f64_f32_e32 v[27:28], v17                            // 000000007e70: 7e362111
	v_cmp_eq_f32_e64 s0, 0, v13                                // 000000007e74: d4120000 02021a80
	v_cmp_class_f32_e64 s2, v17, 0x1f8                         // 000000007e7c: d47e0002 0201ff11 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007e88: 8b000200
	v_mul_f64_e32 v[21:22], v[21:22], v[23:24]                 // 000000007e8c: 0c2a2f15
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007e90: bf870091
	v_mul_f64_e32 v[21:22], v[21:22], v[27:28]                 // 000000007e94: 0c2a3715
	v_cvt_f32_f64_e32 v12, v[21:22]                            // 000000007e98: 7e181f15
	s_wait_alu depctr_sa_sdst(0)                               // 000000007e9c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007ea0: bf870001
	v_cndmask_b32_e64 v12, v12, 0, s0                          // 000000007ea4: d501000c 0001010c
	s_branch 63713                                             // 000000007eac: bfa0f8e1 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4734>
	v_cvt_f64_f32_e32 v[21:22], v14                            // 000000007eb0: 7e2a210e
	v_cvt_f64_f32_e32 v[23:24], v26                            // 000000007eb4: 7e2e211a
	v_cvt_f64_f32_e32 v[27:28], v18                            // 000000007eb8: 7e362112
	v_cmp_eq_f32_e64 s0, 0, v14                                // 000000007ebc: d4120000 02021c80
	v_cmp_class_f32_e64 s2, v18, 0x1f8                         // 000000007ec4: d47e0002 0201ff12 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007ed0: 8b000200
	v_mul_f64_e32 v[21:22], v[21:22], v[23:24]                 // 000000007ed4: 0c2a2f15
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007ed8: bf870091
	v_mul_f64_e32 v[21:22], v[21:22], v[27:28]                 // 000000007edc: 0c2a3715
	v_cvt_f32_f64_e32 v13, v[21:22]                            // 000000007ee0: 7e1a1f15
	s_wait_alu depctr_sa_sdst(0)                               // 000000007ee4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007ee8: bf870001
	v_cndmask_b32_e64 v13, v13, 0, s0                          // 000000007eec: d501000d 0001010d
	s_branch 63707                                             // 000000007ef4: bfa0f8db <tessera_rocm_folded_matmul_a629485a5b78541c+0x4764>
	v_cvt_f64_f32_e32 v[17:18], v15                            // 000000007ef8: 7e22210f
	v_cvt_f64_f32_e32 v[21:22], v26                            // 000000007efc: 7e2a211a
	v_cvt_f64_f32_e32 v[23:24], v19                            // 000000007f00: 7e2e2113
	v_cmp_eq_f32_e64 s0, 0, v15                                // 000000007f04: d4120000 02021e80
	v_cmp_class_f32_e64 s2, v19, 0x1f8                         // 000000007f0c: d47e0002 0201ff13 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007f18: 8b000200
	v_mul_f64_e32 v[17:18], v[17:18], v[21:22]                 // 000000007f1c: 0c222b11
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007f20: bf870091
	v_mul_f64_e32 v[17:18], v[17:18], v[23:24]                 // 000000007f24: 0c222f11
	v_cvt_f32_f64_e32 v14, v[17:18]                            // 000000007f28: 7e1c1f11
	s_wait_alu depctr_sa_sdst(0)                               // 000000007f2c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007f30: bf870001
	v_cndmask_b32_e64 v14, v14, 0, s0                          // 000000007f34: d501000e 0001010e
	s_branch 63701                                             // 000000007f3c: bfa0f8d5 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4794>
	v_cvt_f64_f32_e32 v[17:18], v16                            // 000000007f40: 7e222110
	v_cvt_f64_f32_e32 v[21:22], v26                            // 000000007f44: 7e2a211a
	v_cvt_f64_f32_e32 v[23:24], v20                            // 000000007f48: 7e2e2114
	v_cmp_eq_f32_e64 s0, 0, v16                                // 000000007f4c: d4120000 02022080
	v_cmp_class_f32_e64 s2, v20, 0x1f8                         // 000000007f54: d47e0002 0201ff14 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007f60: 8b000200
	v_mul_f64_e32 v[17:18], v[17:18], v[21:22]                 // 000000007f64: 0c222b11
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007f68: bf870091
	v_mul_f64_e32 v[17:18], v[17:18], v[23:24]                 // 000000007f6c: 0c222f11
	v_cvt_f32_f64_e32 v15, v[17:18]                            // 000000007f70: 7e1e1f11
	s_wait_alu depctr_sa_sdst(0)                               // 000000007f74: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007f78: bf870001
	v_cndmask_b32_e64 v15, v15, 0, s0                          // 000000007f7c: d501000f 0001010f
	s_branch 63695                                             // 000000007f84: bfa0f8cf <tessera_rocm_folded_matmul_a629485a5b78541c+0x47c4>
	v_cvt_f64_f32_e32 v[19:20], v1                             // 000000007f88: 7e262101
	v_cvt_f64_f32_e32 v[21:22], v18                            // 000000007f8c: 7e2a2112
	v_cvt_f64_f32_e32 v[23:24], v13                            // 000000007f90: 7e2e210d
	v_cmp_eq_f32_e64 s0, 0, v1                                 // 000000007f94: d4120000 02020280
	v_cmp_class_f32_e64 s2, v13, 0x1f8                         // 000000007f9c: d47e0002 0201ff0d 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007fa8: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007fac: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007fb0: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007fb4: 0c262f13
	v_cvt_f32_f64_e32 v17, v[19:20]                            // 000000007fb8: 7e221f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000007fbc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000007fc0: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s0                          // 000000007fc4: d5010011 00010111
	s_branch 64148                                             // 000000007fcc: bfa0fa94 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4f20>
	v_cvt_f64_f32_e32 v[19:20], v2                             // 000000007fd0: 7e262102
	v_cvt_f64_f32_e32 v[21:22], v18                            // 000000007fd4: 7e2a2112
	v_cvt_f64_f32_e32 v[23:24], v14                            // 000000007fd8: 7e2e210e
	v_cmp_eq_f32_e64 s0, 0, v2                                 // 000000007fdc: d4120000 02020480
	v_cmp_class_f32_e64 s2, v14, 0x1f8                         // 000000007fe4: d47e0002 0201ff0e 000001f8
	s_and_b32 s0, s0, s2                                       // 000000007ff0: 8b000200
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000007ff4: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000007ff8: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000007ffc: 0c262f13
	v_cvt_f32_f64_e32 v1, v[19:20]                             // 000000008000: 7e021f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000008004: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000008008: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s0                            // 00000000800c: d5010001 00010101
	s_branch 64142                                             // 000000008014: bfa0fa8e <tessera_rocm_folded_matmul_a629485a5b78541c+0x4f50>
	v_cvt_f64_f32_e32 v[13:14], v3                             // 000000008018: 7e1a2103
	v_cvt_f64_f32_e32 v[19:20], v18                            // 00000000801c: 7e262112
	v_cvt_f64_f32_e32 v[21:22], v15                            // 000000008020: 7e2a210f
	v_cmp_eq_f32_e64 s0, 0, v3                                 // 000000008024: d4120000 02020680
	v_cmp_class_f32_e64 s2, v15, 0x1f8                         // 00000000802c: d47e0002 0201ff0f 000001f8
	s_and_b32 s0, s0, s2                                       // 000000008038: 8b000200
	v_mul_f64_e32 v[13:14], v[13:14], v[19:20]                 // 00000000803c: 0c1a270d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000008040: bf870091
	v_mul_f64_e32 v[13:14], v[13:14], v[21:22]                 // 000000008044: 0c1a2b0d
	v_cvt_f32_f64_e32 v2, v[13:14]                             // 000000008048: 7e041f0d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000804c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000008050: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s0                            // 000000008054: d5010002 00010102
	s_branch 64136                                             // 00000000805c: bfa0fa88 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4f80>
	v_cvt_f64_f32_e32 v[13:14], v4                             // 000000008060: 7e1a2104
	v_cvt_f64_f32_e32 v[19:20], v18                            // 000000008064: 7e262112
	v_cvt_f64_f32_e32 v[21:22], v16                            // 000000008068: 7e2a2110
	v_cmp_eq_f32_e64 s0, 0, v4                                 // 00000000806c: d4120000 02020880
	v_cmp_class_f32_e64 s2, v16, 0x1f8                         // 000000008074: d47e0002 0201ff10 000001f8
	s_and_b32 s0, s0, s2                                       // 000000008080: 8b000200
	v_mul_f64_e32 v[13:14], v[13:14], v[19:20]                 // 000000008084: 0c1a270d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000008088: bf870091
	v_mul_f64_e32 v[13:14], v[13:14], v[21:22]                 // 00000000808c: 0c1a2b0d
	v_cvt_f32_f64_e32 v3, v[13:14]                             // 000000008090: 7e061f0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000008094: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000008098: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s0                            // 00000000809c: d5010003 00010103
	s_branch 64130                                             // 0000000080a4: bfa0fa82 <tessera_rocm_folded_matmul_a629485a5b78541c+0x4fb0>
	v_cvt_f64_f32_e32 v[13:14], v5                             // 0000000080a8: 7e1a2105
	v_cvt_f64_f32_e32 v[15:16], v18                            // 0000000080ac: 7e1e2112
	v_cvt_f64_f32_e32 v[19:20], v9                             // 0000000080b0: 7e262109
	v_cmp_eq_f32_e64 s0, 0, v5                                 // 0000000080b4: d4120000 02020a80
	v_cmp_class_f32_e64 s2, v9, 0x1f8                          // 0000000080bc: d47e0002 0201ff09 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000080c8: 8b000200
	v_mul_f64_e32 v[13:14], v[13:14], v[15:16]                 // 0000000080cc: 0c1a1f0d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000080d0: bf870091
	v_mul_f64_e32 v[13:14], v[13:14], v[19:20]                 // 0000000080d4: 0c1a270d
	v_cvt_f32_f64_e32 v4, v[13:14]                             // 0000000080d8: 7e081f0d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000080dc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000080e0: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s0                            // 0000000080e4: d5010004 00010104
	s_branch 64124                                             // 0000000080ec: bfa0fa7c <tessera_rocm_folded_matmul_a629485a5b78541c+0x4fe0>
	v_cvt_f64_f32_e32 v[13:14], v6                             // 0000000080f0: 7e1a2106
	v_cvt_f64_f32_e32 v[15:16], v18                            // 0000000080f4: 7e1e2112
	v_cvt_f64_f32_e32 v[19:20], v10                            // 0000000080f8: 7e26210a
	v_cmp_eq_f32_e64 s0, 0, v6                                 // 0000000080fc: d4120000 02020c80
	v_cmp_class_f32_e64 s2, v10, 0x1f8                         // 000000008104: d47e0002 0201ff0a 000001f8
	s_and_b32 s0, s0, s2                                       // 000000008110: 8b000200
	v_mul_f64_e32 v[13:14], v[13:14], v[15:16]                 // 000000008114: 0c1a1f0d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000008118: bf870091
	v_mul_f64_e32 v[13:14], v[13:14], v[19:20]                 // 00000000811c: 0c1a270d
	v_cvt_f32_f64_e32 v5, v[13:14]                             // 000000008120: 7e0a1f0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000008124: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000008128: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s0                            // 00000000812c: d5010005 00010105
	s_branch 64118                                             // 000000008134: bfa0fa76 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5010>
	v_cvt_f64_f32_e32 v[9:10], v7                              // 000000008138: 7e122107
	v_cvt_f64_f32_e32 v[13:14], v18                            // 00000000813c: 7e1a2112
	v_cvt_f64_f32_e32 v[15:16], v11                            // 000000008140: 7e1e210b
	v_cmp_eq_f32_e64 s0, 0, v7                                 // 000000008144: d4120000 02020e80
	v_cmp_class_f32_e64 s2, v11, 0x1f8                         // 00000000814c: d47e0002 0201ff0b 000001f8
	s_and_b32 s0, s0, s2                                       // 000000008158: 8b000200
	v_mul_f64_e32 v[9:10], v[9:10], v[13:14]                   // 00000000815c: 0c121b09
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000008160: bf870091
	v_mul_f64_e32 v[9:10], v[9:10], v[15:16]                   // 000000008164: 0c121f09
	v_cvt_f32_f64_e32 v6, v[9:10]                              // 000000008168: 7e0c1f09
	s_wait_alu depctr_sa_sdst(0)                               // 00000000816c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000008170: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s0                            // 000000008174: d5010006 00010106
	s_branch 64112                                             // 00000000817c: bfa0fa70 <tessera_rocm_folded_matmul_a629485a5b78541c+0x5040>
	v_cvt_f64_f32_e32 v[9:10], v8                              // 000000008180: 7e122108
	v_cvt_f64_f32_e32 v[13:14], v18                            // 000000008184: 7e1a2112
	v_cvt_f64_f32_e32 v[15:16], v12                            // 000000008188: 7e1e210c
	v_cmp_eq_f32_e64 s0, 0, v8                                 // 00000000818c: d4120000 02021080
	v_cmp_class_f32_e64 s2, v12, 0x1f8                         // 000000008194: d47e0002 0201ff0c 000001f8
	s_and_b32 s0, s0, s2                                       // 0000000081a0: 8b000200
	v_mul_f64_e32 v[9:10], v[9:10], v[13:14]                   // 0000000081a4: 0c121b09
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000081a8: bf870091
	v_mul_f64_e32 v[9:10], v[9:10], v[15:16]                   // 0000000081ac: 0c121f09
	v_cvt_f32_f64_e32 v7, v[9:10]                              // 0000000081b0: 7e0e1f09
	s_wait_alu depctr_sa_sdst(0)                               // 0000000081b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000081b8: bf870001
	v_cndmask_b32_e64 v7, v7, 0, s0                            // 0000000081bc: d5010007 00010107
	s_branch 64106                                             // 0000000081c4: bfa0fa6a <tessera_rocm_folded_matmul_a629485a5b78541c+0x5070>
	s_code_end                                                 // 0000000081c8: bf9f0000
	s_code_end                                                 // 0000000081cc: bf9f0000
	s_code_end                                                 // 0000000081d0: bf9f0000
	s_code_end                                                 // 0000000081d4: bf9f0000
	s_code_end                                                 // 0000000081d8: bf9f0000
	s_code_end                                                 // 0000000081dc: bf9f0000
	s_code_end                                                 // 0000000081e0: bf9f0000
	s_code_end                                                 // 0000000081e4: bf9f0000
	s_code_end                                                 // 0000000081e8: bf9f0000
	s_code_end                                                 // 0000000081ec: bf9f0000
	s_code_end                                                 // 0000000081f0: bf9f0000
	s_code_end                                                 // 0000000081f4: bf9f0000
	s_code_end                                                 // 0000000081f8: bf9f0000
	s_code_end                                                 // 0000000081fc: bf9f0000
	s_code_end                                                 // 000000008200: bf9f0000
	s_code_end                                                 // 000000008204: bf9f0000
	s_code_end                                                 // 000000008208: bf9f0000
	s_code_end                                                 // 00000000820c: bf9f0000
	s_code_end                                                 // 000000008210: bf9f0000
	s_code_end                                                 // 000000008214: bf9f0000
	s_code_end                                                 // 000000008218: bf9f0000
	s_code_end                                                 // 00000000821c: bf9f0000
	s_code_end                                                 // 000000008220: bf9f0000
	s_code_end                                                 // 000000008224: bf9f0000
	s_code_end                                                 // 000000008228: bf9f0000
	s_code_end                                                 // 00000000822c: bf9f0000
	s_code_end                                                 // 000000008230: bf9f0000
	s_code_end                                                 // 000000008234: bf9f0000
	s_code_end                                                 // 000000008238: bf9f0000
	s_code_end                                                 // 00000000823c: bf9f0000
	s_code_end                                                 // 000000008240: bf9f0000
	s_code_end                                                 // 000000008244: bf9f0000
	s_code_end                                                 // 000000008248: bf9f0000
	s_code_end                                                 // 00000000824c: bf9f0000
	s_code_end                                                 // 000000008250: bf9f0000
	s_code_end                                                 // 000000008254: bf9f0000
	s_code_end                                                 // 000000008258: bf9f0000
	s_code_end                                                 // 00000000825c: bf9f0000
	s_code_end                                                 // 000000008260: bf9f0000
	s_code_end                                                 // 000000008264: bf9f0000
	s_code_end                                                 // 000000008268: bf9f0000
	s_code_end                                                 // 00000000826c: bf9f0000
	s_code_end                                                 // 000000008270: bf9f0000
	s_code_end                                                 // 000000008274: bf9f0000
	s_code_end                                                 // 000000008278: bf9f0000
	s_code_end                                                 // 00000000827c: bf9f0000
	s_code_end                                                 // 000000008280: bf9f0000
	s_code_end                                                 // 000000008284: bf9f0000
	s_code_end                                                 // 000000008288: bf9f0000
	s_code_end                                                 // 00000000828c: bf9f0000
	s_code_end                                                 // 000000008290: bf9f0000
	s_code_end                                                 // 000000008294: bf9f0000
	s_code_end                                                 // 000000008298: bf9f0000
	s_code_end                                                 // 00000000829c: bf9f0000
	s_code_end                                                 // 0000000082a0: bf9f0000
	s_code_end                                                 // 0000000082a4: bf9f0000
	s_code_end                                                 // 0000000082a8: bf9f0000
	s_code_end                                                 // 0000000082ac: bf9f0000
	s_code_end                                                 // 0000000082b0: bf9f0000
	s_code_end                                                 // 0000000082b4: bf9f0000
	s_code_end                                                 // 0000000082b8: bf9f0000
	s_code_end                                                 // 0000000082bc: bf9f0000
	s_code_end                                                 // 0000000082c0: bf9f0000
	s_code_end                                                 // 0000000082c4: bf9f0000
	s_code_end                                                 // 0000000082c8: bf9f0000
	s_code_end                                                 // 0000000082cc: bf9f0000
	s_code_end                                                 // 0000000082d0: bf9f0000
	s_code_end                                                 // 0000000082d4: bf9f0000
	s_code_end                                                 // 0000000082d8: bf9f0000
	s_code_end                                                 // 0000000082dc: bf9f0000
	s_code_end                                                 // 0000000082e0: bf9f0000
	s_code_end                                                 // 0000000082e4: bf9f0000
	s_code_end                                                 // 0000000082e8: bf9f0000
	s_code_end                                                 // 0000000082ec: bf9f0000
	s_code_end                                                 // 0000000082f0: bf9f0000
	s_code_end                                                 // 0000000082f4: bf9f0000
	s_code_end                                                 // 0000000082f8: bf9f0000
	s_code_end                                                 // 0000000082fc: bf9f0000
	s_code_end                                                 // 000000008300: bf9f0000
	s_code_end                                                 // 000000008304: bf9f0000
	s_code_end                                                 // 000000008308: bf9f0000
	s_code_end                                                 // 00000000830c: bf9f0000
	s_code_end                                                 // 000000008310: bf9f0000
	s_code_end                                                 // 000000008314: bf9f0000
	s_code_end                                                 // 000000008318: bf9f0000
	s_code_end                                                 // 00000000831c: bf9f0000
	s_code_end                                                 // 000000008320: bf9f0000
	s_code_end                                                 // 000000008324: bf9f0000
	s_code_end                                                 // 000000008328: bf9f0000
	s_code_end                                                 // 00000000832c: bf9f0000
	s_code_end                                                 // 000000008330: bf9f0000
	s_code_end                                                 // 000000008334: bf9f0000
	s_code_end                                                 // 000000008338: bf9f0000
	s_code_end                                                 // 00000000833c: bf9f0000
	s_code_end                                                 // 000000008340: bf9f0000
	s_code_end                                                 // 000000008344: bf9f0000
	s_code_end                                                 // 000000008348: bf9f0000
	s_code_end                                                 // 00000000834c: bf9f0000
	s_code_end                                                 // 000000008350: bf9f0000
	s_code_end                                                 // 000000008354: bf9f0000
	s_code_end                                                 // 000000008358: bf9f0000
	s_code_end                                                 // 00000000835c: bf9f0000
	s_code_end                                                 // 000000008360: bf9f0000
	s_code_end                                                 // 000000008364: bf9f0000
	s_code_end                                                 // 000000008368: bf9f0000
	s_code_end                                                 // 00000000836c: bf9f0000
	s_code_end                                                 // 000000008370: bf9f0000
	s_code_end                                                 // 000000008374: bf9f0000
	s_code_end                                                 // 000000008378: bf9f0000
	s_code_end                                                 // 00000000837c: bf9f0000
