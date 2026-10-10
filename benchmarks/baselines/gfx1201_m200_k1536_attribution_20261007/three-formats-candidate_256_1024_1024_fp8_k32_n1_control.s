
/tmp/tmpsds_gghb.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b04: f4004500 f80000c8
	s_load_b64 s[18:19], s[0:1], 0xa8                          // 000000001b0c: f4002480 f80000a8
	s_mov_b32 s2, ttmp9                                        // 000000001b14: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b18: 86039f75
	s_clause 0x4                                               // 000000001b1c: bf850004
	s_load_b64 s[24:25], s[0:1], 0xd8                          // 000000001b20: f4002600 f80000d8
	s_load_b64 s[30:31], s[0:1], 0x8                           // 000000001b28: f4002780 f8000008
	s_load_b64 s[34:35], s[0:1], 0x30                          // 000000001b30: f4002880 f8000030
	s_load_b64 s[26:27], s[0:1], 0x58                          // 000000001b38: f4002680 f8000058
	s_load_b64 s[36:37], s[0:1], 0x80                          // 000000001b40: f4002900 f8000080
	s_lshl_b64 s[40:41], s[2:3], 5                             // 000000001b48: 84a88502
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b4c: bf870009
	v_dual_mov_b32 v37, s41 :: v_dual_and_b32 v48, 15, v0      // 000000001b50: ca240029 2530008f
	s_mov_b32 s4, ttmp7                                        // 000000001b58: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b5c: 86059f73
	s_add_nc_u64 s[2:3], s[40:41], 32                          // 000000001b60: a982a028
	s_lshl_b64 s[38:39], s[4:5], 5                             // 000000001b64: 84a68504
	v_or_b32_e32 v32, s40, v48                                 // 000000001b68: 38406028
	s_add_nc_u64 s[0:1], s[38:39], 32                          // 000000001b6c: a980a026
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v33, s41           // 000000001b70: ca100080 3e200029
	v_bfe_u32 v0, v0, 4, 1                                     // 000000001b78: d6100000 02050900
	s_delay_alu instid0(valu_dep_3)                            // 000000001b80: bf870003
	v_or_b32_e32 v36, 16, v32                                  // 000000001b84: 38484090
	s_or_b32 s33, s38, 16                                      // 000000001b88: 8c219026
	s_wait_kmcnt 0x0                                           // 000000001b8c: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[20:21]                      // 000000001b90: d4540000 02002800
	v_cmp_gt_i64_e64 s1, s[2:3], s[22:23]                      // 000000001b98: d4540001 02002c02
	v_lshlrev_b32_e32 v34, 3, v0                               // 000000001ba0: 30440083
	s_mov_b32 s2, -1                                           // 000000001ba4: be8200c1
	s_lshr_b64 s[28:29], s[24:25], 5                           // 000000001ba8: 859c8518
	s_or_b32 s0, s0, s1                                        // 000000001bac: 8c000100
	v_cmp_gt_i64_e64 s1, s[22:23], v[32:33]                    // 000000001bb0: d4540001 02024016
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb8: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bbc: 8b6a007e
	v_cmp_gt_i64_e64 s0, s[22:23], v[36:37]                    // 000000001bc0: d4540000 02024816
	s_cbranch_vccnz 5                                          // 000000001bc8: bfa40005 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0xe0>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001bcc: 8b6a027e
	s_cbranch_vccnz 3398                                       // 000000001bd0: bfa40d46 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x35ec>
	s_nop 0                                                    // 000000001bd4: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000001bd8: bfb60003
	s_endpgm                                                   // 000000001bdc: bfb00000
	v_or_b32_e32 v0, s38, v48                                  // 000000001be0: 38006026
	v_or_b32_e32 v2, s33, v48                                  // 000000001be4: 38046021
	v_mov_b32_e32 v1, s39                                      // 000000001be8: 7e020227
	v_mov_b32_e32 v3, s39                                      // 000000001bec: 7e060227
	v_or_b32_e32 v40, s38, v34                                 // 000000001bf0: 38504426
	v_mul_lo_u32 v8, s25, v0                                   // 000000001bf4: d72c0008 02020019
	v_mad_co_u64_u32 v[4:5], null, s24, v0, 0                  // 000000001bfc: d6fe7c04 02020018
	v_mul_lo_u32 v9, s25, v2                                   // 000000001c04: d72c0009 02020419
	v_mad_co_u64_u32 v[6:7], null, s24, v2, 0                  // 000000001c0c: d6fe7c06 02020418
	v_mov_b32_e32 v41, s39                                     // 000000001c14: 7e520227
	s_mul_i32 s3, s24, s39                                     // 000000001c18: 96032718
	v_or_b32_e32 v11, 1, v34                                   // 000000001c1c: 38164481
	v_or_b32_e32 v12, 2, v34                                   // 000000001c20: 38184482
	v_or_b32_e32 v14, 3, v34                                   // 000000001c24: 381c4483
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c28: bf88ff9e
	v_add3_u32 v67, v5, s3, v8                                 // 000000001c2c: d6550043 04200705
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[40:41]                // 000000001c34: 7ca85014
	v_add3_u32 v68, v7, s3, v9                                 // 000000001c38: d6550044 04240707
	v_cmp_gt_i64_e64 s3, s[20:21], v[2:3]                      // 000000001c40: d4540003 02020414
	v_mov_b32_e32 v2, s39                                      // 000000001c48: 7e040227
	v_cmp_gt_i64_e64 s2, s[20:21], v[0:1]                      // 000000001c4c: d4540002 02020014
	v_mad_co_u64_u32 v[0:1], null, s24, v32, 0                 // 000000001c54: d6fe7c00 02024018
	v_or_b32_e32 v70, v6, v34                                  // 000000001c5c: 388c4506
	v_mul_lo_u32 v6, s25, v32                                  // 000000001c60: d72c0006 02024019
	v_mul_lo_u32 v7, s24, v33                                  // 000000001c68: d72c0007 02024218
	v_or_b32_e32 v69, v4, v34                                  // 000000001c70: 388a4504
	v_mul_lo_u32 v8, s25, v36                                  // 000000001c74: d72c0008 02024819
	v_mul_lo_u32 v9, s24, v37                                  // 000000001c7c: d72c0009 02024a18
	v_mad_co_u64_u32 v[4:5], null, s24, v36, 0                 // 000000001c84: d6fe7c04 02024818
	v_cndmask_b32_e32 v3, 0, v40, vcc_lo                       // 000000001c8c: 02065080
	v_or_b32_e32 v74, v0, v34                                  // 000000001c90: 38944500
	v_cndmask_b32_e64 v0, 0, s39, vcc_lo                       // 000000001c94: d5010000 01a84e80
	v_add3_u32 v73, v1, v7, v6                                 // 000000001c9c: d6550049 041a0f01
	v_or_b32_e32 v1, s38, v11                                  // 000000001ca4: 38021626
	v_mul_lo_u32 v7, s29, v3                                   // 000000001ca8: d72c0007 0202061d
	v_or_b32_e32 v16, 4, v34                                   // 000000001cb0: 38204484
	v_add3_u32 v75, v5, v9, v8                                 // 000000001cb4: d655004b 04221305
	v_mad_co_u64_u32 v[5:6], null, s28, v3, 0                  // 000000001cbc: d6fe7c05 0202061c
	v_mul_lo_u32 v8, s28, v0                                   // 000000001cc4: d72c0008 0202001c
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[1:2]                  // 000000001ccc: 7ca80214
	v_or_b32_e32 v0, s38, v12                                  // 000000001cd0: 38001826
	v_or_b32_e32 v77, v4, v34                                  // 000000001cd4: 389a4504
	v_or_b32_e32 v38, s33, v34                                 // 000000001cd8: 384c4421
	v_mov_b32_e32 v39, s39                                     // 000000001cdc: 7e4e0227
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v56, 0            // 000000001ce0: ca100080 6b380080
	v_add3_u32 v6, v6, v8, v7                                  // 000000001ce8: d6550006 041e1106
	s_wait_alu depctr_va_vcc(0)                                // 000000001cf0: bf88ff9d
	v_dual_mov_b32 v8, s39 :: v_dual_cndmask_b32 v9, 0, v1     // 000000001cf4: ca120027 08080280
	v_mov_b32_e32 v1, s39                                      // 000000001cfc: 7e020227
	v_cndmask_b32_e32 v3, 0, v2, vcc_lo                        // 000000001d00: 02060480
	v_or_b32_e32 v7, s38, v14                                  // 000000001d04: 380e1c26
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000001d08: 3e0a0a82
	v_mul_lo_u32 v10, s29, v9                                  // 000000001d0c: d72c000a 0202121d
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000001d14: 7ca80014
	v_mul_lo_u32 v13, s28, v3                                  // 000000001d18: d72c000d 0202061c
	v_mad_co_u64_u32 v[3:4], null, s28, v9, 0                  // 000000001d20: d6fe7c03 0202121c
	v_cndmask_b32_e64 v2, 0, v33, s1                           // 000000001d28: d5010002 00064280
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v82, 0            // 000000001d30: ca100080 65520080
	s_wait_alu depctr_va_vcc(0)                                // 000000001d38: bf88ff9d
	v_dual_cndmask_b32 v9, 0, v1 :: v_dual_cndmask_b32 v0, 0, v0// 000000001d3c: ca520280 09000080
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001d44: 7ca80e14
	v_cndmask_b32_e64 v1, 0, v32, s1                           // 000000001d48: d5010001 00064080
	v_add3_u32 v4, v4, v13, v10                                // 000000001d50: d6550004 042a1b04
	s_delay_alu instid0(valu_dep_4)                            // 000000001d58: bf870004
	v_mul_lo_u32 v15, s28, v9                                  // 000000001d5c: d72c000f 0202121c
	v_mul_lo_u32 v13, s29, v0                                  // 000000001d64: d72c000d 0202001d
	v_mad_co_u64_u32 v[9:10], null, s28, v0, 0                 // 000000001d6c: d6fe7c09 0202001c
	s_wait_alu depctr_va_vcc(0)                                // 000000001d74: bf88ff9d
	v_cndmask_b32_e32 v17, 0, v7, vcc_lo                       // 000000001d78: 02220e80
	v_or_b32_e32 v7, s38, v16                                  // 000000001d7c: 380e2026
	v_cndmask_b32_e32 v0, 0, v8, vcc_lo                        // 000000001d80: 02001080
	v_add_co_u32 v80, vcc_lo, s26, v5                          // 000000001d84: d7006a50 02020a1a
	s_wait_alu depctr_va_vcc(0)                                // 000000001d8c: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, s27, v6, vcc_lo             // 000000001d90: d5207c51 01aa0c1b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001d98: 7ca80e14
	v_add3_u32 v10, v10, v15, v13                              // 000000001d9c: d655000a 04361f0a
	v_or_b32_e32 v15, 5, v34                                   // 000000001da4: 381e4485
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 000000001da8: 3e060682
	v_mul_lo_u32 v13, s29, v17                                 // 000000001dac: d72c000d 0202221d
	v_mul_lo_u32 v0, s28, v0                                   // 000000001db4: d72c0000 0202001c
	v_mad_co_u64_u32 v[5:6], null, s28, v17, 0                 // 000000001dbc: d6fe7c05 0202221c
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc4: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v7, vcc_lo                       // 000000001dc8: 02240e80
	v_or_b32_e32 v7, s38, v15                                  // 000000001dcc: 380e1e26
	v_cndmask_b32_e32 v17, 0, v8, vcc_lo                       // 000000001dd0: 02221080
	v_add_co_u32 v83, vcc_lo, s26, v3                          // 000000001dd4: d7006a53 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001ddc: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, s27, v4, vcc_lo             // 000000001de0: d5207c54 01aa081b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001de8: 7ca80e14
	v_add3_u32 v6, v6, v0, v13                                 // 000000001dec: d6550006 04360106
	v_mul_lo_u32 v13, s28, v17                                 // 000000001df4: d72c000d 0202221c
	v_or_b32_e32 v17, 6, v34                                   // 000000001dfc: 38224486
	v_lshlrev_b64_e32 v[3:4], 2, v[9:10]                       // 000000001e00: 3e061282
	v_mul_lo_u32 v0, s29, v18                                  // 000000001e04: d72c0000 0202241d
	v_mad_co_u64_u32 v[9:10], null, s28, v18, 0                // 000000001e0c: d6fe7c09 0202241c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e14: bf88ff9d
	v_cndmask_b32_e32 v19, 0, v7, vcc_lo                       // 000000001e18: 02260e80
	v_or_b32_e32 v7, s38, v17                                  // 000000001e1c: 380e2226
	v_cndmask_b32_e32 v18, 0, v8, vcc_lo                       // 000000001e20: 02241080
	v_add_co_u32 v86, vcc_lo, s26, v3                          // 000000001e24: d7006a56 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e2c: bf88ff9d
	v_add_co_ci_u32_e64 v87, null, s27, v4, vcc_lo             // 000000001e30: d5207c57 01aa081b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001e38: 7ca80e14
	v_add3_u32 v10, v10, v13, v0                               // 000000001e3c: d655000a 04021b0a
	v_mul_lo_u32 v13, s28, v18                                 // 000000001e44: d72c000d 0202241c
	v_or_b32_e32 v18, 7, v34                                   // 000000001e4c: 38244487
	v_lshlrev_b64_e32 v[3:4], 2, v[5:6]                        // 000000001e50: 3e060a82
	v_mul_lo_u32 v0, s29, v19                                  // 000000001e54: d72c0000 0202261d
	v_mad_co_u64_u32 v[5:6], null, s28, v19, 0                 // 000000001e5c: d6fe7c05 0202261c
	s_wait_alu depctr_va_vcc(0)                                // 000000001e64: bf88ff9d
	v_cndmask_b32_e32 v20, 0, v7, vcc_lo                       // 000000001e68: 02280e80
	v_or_b32_e32 v7, s38, v18                                  // 000000001e6c: 380e2426
	v_cndmask_b32_e32 v19, 0, v8, vcc_lo                       // 000000001e70: 02261080
	v_add_co_u32 v89, vcc_lo, s26, v3                          // 000000001e74: d7006a59 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e7c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s27, v4, vcc_lo             // 000000001e80: d5207c5a 01aa081b
	v_lshlrev_b64_e32 v[3:4], 2, v[9:10]                       // 000000001e88: 3e061282
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001e8c: 7ca80e14
	v_add3_u32 v6, v6, v13, v0                                 // 000000001e90: d6550006 04021b06
	v_mul_lo_u32 v0, s29, v20                                  // 000000001e98: d72c0000 0202281d
	v_mul_lo_u32 v13, s28, v19                                 // 000000001ea0: d72c000d 0202261c
	v_mad_co_u64_u32 v[9:10], null, s28, v20, 0                // 000000001ea8: d6fe7c09 0202281c
	v_mov_b32_e32 v104, 0                                      // 000000001eb0: 7ed00280
	s_wait_alu depctr_va_vcc(0)                                // 000000001eb4: bf88ff9d
	v_dual_cndmask_b32 v8, 0, v8 :: v_dual_cndmask_b32 v7, 0, v7// 000000001eb8: ca521080 08060e80
	v_add_co_u32 v91, vcc_lo, s26, v3                          // 000000001ec0: d7006a5b 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001ec8: bf88ff9d
	v_add_co_ci_u32_e64 v92, null, s27, v4, vcc_lo             // 000000001ecc: d5207c5c 01aa081b
	v_lshlrev_b64_e32 v[3:4], 2, v[5:6]                        // 000000001ed4: 3e060a82
	v_add3_u32 v10, v10, v13, v0                               // 000000001ed8: d655000a 04021b0a
	v_mul_lo_u32 v0, s29, v7                                   // 000000001ee0: d72c0000 02020e1d
	v_mul_lo_u32 v8, s28, v8                                   // 000000001ee8: d72c0008 0202101c
	v_mad_co_u64_u32 v[5:6], null, s28, v7, 0                  // 000000001ef0: d6fe7c05 02020e1c
	v_or_b32_e32 v7, s33, v11                                  // 000000001ef8: 380e1621
	v_add_co_u32 v94, vcc_lo, s26, v3                          // 000000001efc: d7006a5e 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f04: bf88ff9d
	v_add_co_ci_u32_e64 v95, null, s27, v4, vcc_lo             // 000000001f08: d5207c5f 01aa081b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[38:39]                // 000000001f10: 7ca84c14
	v_lshlrev_b64_e32 v[3:4], 2, v[9:10]                       // 000000001f14: 3e061282
	v_add3_u32 v6, v6, v8, v0                                  // 000000001f18: d6550006 04021106
	v_mov_b32_e32 v8, s39                                      // 000000001f20: 7e100227
	v_mov_b32_e32 v88, 0                                       // 000000001f24: 7eb00280
	v_mov_b32_e32 v98, 0                                       // 000000001f28: 7ec40280
	s_wait_alu depctr_va_vcc(0)                                // 000000001f2c: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v38, vcc_lo                       // 000000001f30: 02004c80
	v_cndmask_b32_e64 v9, 0, s39, vcc_lo                       // 000000001f34: d5010009 01a84e80
	v_add_co_u32 v96, s4, s26, v3                              // 000000001f3c: d7000460 0202061a
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001f44: 7ca80e14
	v_add_co_ci_u32_e64 v97, null, s27, v4, s4                 // 000000001f48: d5207c61 0012081b
	v_lshlrev_b64_e32 v[3:4], 2, v[5:6]                        // 000000001f50: 3e060a82
	v_mul_lo_u32 v10, s29, v0                                  // 000000001f54: d72c000a 0202001d
	v_mad_co_u64_u32 v[5:6], null, s28, v0, 0                  // 000000001f5c: d6fe7c05 0202001c
	v_mul_lo_u32 v0, s28, v9                                   // 000000001f64: d72c0000 0202121c
	s_wait_alu depctr_va_vcc(0)                                // 000000001f6c: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v7, vcc_lo                       // 000000001f70: 02160e80
	v_or_b32_e32 v7, s33, v12                                  // 000000001f74: 380e1821
	v_cndmask_b32_e64 v9, 0, s39, vcc_lo                       // 000000001f78: d5010009 01a84e80
	v_add_co_u32 v99, vcc_lo, s26, v3                          // 000000001f80: d7006a63 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000001f88: bf88ff9d
	v_add_co_ci_u32_e64 v100, null, s27, v4, vcc_lo            // 000000001f8c: d5207c64 01aa081b
	v_add3_u32 v6, v6, v0, v10                                 // 000000001f94: d6550006 042a0106
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001f9c: 7ca80e14
	v_mul_lo_u32 v0, s29, v11                                  // 000000001fa0: d72c0000 0202161d
	v_mul_lo_u32 v12, s28, v9                                  // 000000001fa8: d72c000c 0202121c
	v_mad_co_u64_u32 v[3:4], null, s28, v11, 0                 // 000000001fb0: d6fe7c03 0202161c
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000001fb8: 3e0a0a82
	v_mov_b32_e32 v11, s39                                     // 000000001fbc: 7e160227
	v_or_b32_e32 v10, s33, v14                                 // 000000001fc0: 38141c21
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc4: bf88ff9d
	v_cndmask_b32_e64 v13, 0, s39, vcc_lo                      // 000000001fc8: d501000d 01a84e80
	v_dual_cndmask_b32 v7, 0, v7 :: v_dual_mov_b32 v64, 0      // 000000001fd0: ca500e80 07400080
	v_add_co_u32 v102, s4, s26, v5                             // 000000001fd8: d7000466 02020a1a
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 000000001fe0: 7ca81414
	v_add3_u32 v4, v4, v12, v0                                 // 000000001fe4: d6550004 04021904
	s_delay_alu instid0(valu_dep_4)                            // 000000001fec: bf870004
	v_mul_lo_u32 v0, s29, v7                                   // 000000001ff0: d72c0000 02020e1d
	v_mul_lo_u32 v14, s28, v13                                 // 000000001ff8: d72c000e 02021a1c
	v_mad_co_u64_u32 v[12:13], null, s28, v7, 0                // 000000002000: d6fe7c0c 02020e1c
	s_wait_alu depctr_va_sdst(0)                               // 000000002008: bf88f19f
	v_add_co_ci_u32_e64 v103, null, s27, v6, s4                // 00000000200c: d5207c67 00120c1b
	v_mov_b32_e32 v6, s39                                      // 000000002014: 7e0c0227
	v_or_b32_e32 v5, s33, v16                                  // 000000002018: 380a2021
	s_wait_alu depctr_va_vcc(0)                                // 00000000201c: bf88ff9d
	v_cndmask_b32_e64 v7, 0, s39, vcc_lo                       // 000000002020: d5010007 01a84e80
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo                      // 000000002028: 02141480
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]                        // 00000000202c: 3e060682
	v_add3_u32 v13, v13, v14, v0                               // 000000002030: d655000d 04021d0d
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[5:6]                  // 000000002038: 7ca80a14
	v_mul_lo_u32 v7, s28, v7                                   // 00000000203c: d72c0007 02020e1c
	v_mul_lo_u32 v0, s29, v10                                  // 000000002044: d72c0000 0202141d
	v_mad_co_u64_u32 v[10:11], null, s28, v10, 0               // 00000000204c: d6fe7c0a 0202141c
	v_add_co_u32 v105, s4, s26, v3                             // 000000002054: d7000469 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 00000000205c: bf88ff9d
	v_cndmask_b32_e64 v14, 0, s39, vcc_lo                      // 000000002060: d501000e 01a84e80
	v_cndmask_b32_e32 v16, 0, v5, vcc_lo                       // 000000002068: 02200a80
	s_wait_alu depctr_va_sdst(0)                               // 00000000206c: bf88f19f
	v_add_co_ci_u32_e64 v106, null, s27, v4, s4                // 000000002070: d5207c6a 0012081b
	v_lshlrev_b64_e32 v[3:4], 2, v[12:13]                      // 000000002078: 3e061882
	v_or_b32_e32 v5, s33, v15                                  // 00000000207c: 380a1e21
	v_add3_u32 v11, v11, v7, v0                                // 000000002080: d655000b 04020f0b
	v_mul_lo_u32 v0, s29, v16                                  // 000000002088: d72c0000 0202201d
	v_mul_lo_u32 v14, s28, v14                                 // 000000002090: d72c000e 02021c1c
	v_mad_co_u64_u32 v[12:13], null, s28, v16, 0               // 000000002098: d6fe7c0c 0202201c
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[5:6]                  // 0000000020a0: 7ca80a14
	v_mov_b32_e32 v7, s39                                      // 0000000020a4: 7e0e0227
	v_or_b32_e32 v6, s33, v17                                  // 0000000020a8: 380c2221
	v_add_co_u32 v108, s4, s26, v3                             // 0000000020ac: d700046c 0202061a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020b4: bf88f19f
	v_add_co_ci_u32_e64 v109, null, s27, v4, s4                // 0000000020b8: d5207c6d 0012081b
	v_lshlrev_b64_e32 v[3:4], 2, v[10:11]                      // 0000000020c0: 3e061482
	v_dual_mov_b32 v11, s39 :: v_dual_mov_b32 v58, 0           // 0000000020c4: ca100027 0b3a0080
	v_or_b32_e32 v10, s33, v18                                 // 0000000020cc: 38142421
	v_cmp_gt_i64_e64 s4, s[20:21], v[6:7]                      // 0000000020d0: d4540004 02020c14
	v_add3_u32 v13, v13, v14, v0                               // 0000000020d8: d655000d 04021d0d
	s_wait_alu depctr_va_vcc(0)                                // 0000000020e0: bf88ff9d
	v_cndmask_b32_e64 v0, 0, s39, vcc_lo                       // 0000000020e4: d5010000 01a84e80
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_mov_b32 v60, 0      // 0000000020ec: ca500a80 053c0080
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 0000000020f4: 7ca81414
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f8: bf88f19f
	v_cndmask_b32_e64 v14, 0, v6, s4                           // 0000000020fc: d501000e 00120c80
	v_mul_lo_u32 v0, s28, v0                                   // 000000002104: d72c0000 0202001c
	v_mul_lo_u32 v16, s29, v5                                  // 00000000210c: d72c0010 02020a1d
	v_mad_co_u64_u32 v[5:6], null, s28, v5, 0                  // 000000002114: d6fe7c05 02020a1c
	v_cndmask_b32_e64 v7, 0, s39, s4                           // 00000000211c: d5010007 00104e80
	s_wait_alu depctr_va_vcc(0)                                // 000000002124: bf88ff9d
	v_cndmask_b32_e64 v11, 0, s39, vcc_lo                      // 000000002128: d501000b 01a84e80
	v_cndmask_b32_e32 v10, 0, v10, vcc_lo                      // 000000002130: 02141480
	v_mul_lo_u32 v17, s29, v14                                 // 000000002134: d72c0011 02021c1d
	v_mad_co_u64_u32 v[14:15], null, s28, v14, 0               // 00000000213c: d6fe7c0e 02021c1c
	v_mul_lo_u32 v7, s28, v7                                   // 000000002144: d72c0007 02020e1c
	v_add_co_u32 v110, vcc_lo, s26, v3                         // 00000000214c: d7006a6e 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000002154: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, s27, v4, vcc_lo            // 000000002158: d5207c6f 01aa081b
	v_add3_u32 v6, v6, v0, v16                                 // 000000002160: d6550006 04420106
	v_lshlrev_b64_e32 v[3:4], 2, v[12:13]                      // 000000002168: 3e061882
	v_mul_lo_u32 v0, s29, v10                                  // 00000000216c: d72c0000 0202141d
	v_mul_lo_u32 v12, s28, v11                                 // 000000002174: d72c000c 0202161c
	v_mad_co_u64_u32 v[10:11], null, s28, v10, 0               // 00000000217c: d6fe7c0a 0202141c
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000002184: 3e0a0a82
	v_add3_u32 v15, v15, v7, v17                               // 000000002188: d655000f 04460f0f
	v_add_co_u32 v112, vcc_lo, s26, v3                         // 000000002190: d7006a70 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 000000002198: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, s27, v4, vcc_lo            // 00000000219c: d5207c71 01aa081b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 0000000021a4: bf8701d3
	v_lshlrev_b64_e32 v[3:4], 2, v[14:15]                      // 0000000021a8: 3e061c82
	v_add3_u32 v11, v11, v12, v0                               // 0000000021ac: d655000b 0402190b
	v_add_co_u32 v114, vcc_lo, s26, v5                         // 0000000021b4: d7006a72 02020a1a
	s_wait_alu depctr_va_vcc(0)                                // 0000000021bc: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, s27, v6, vcc_lo            // 0000000021c0: d5207c73 01aa0c1b
	v_lshlrev_b64_e32 v[5:6], 2, v[10:11]                      // 0000000021c8: 3e0a1482
	v_cndmask_b32_e64 v9, 0, v37, s0                           // 0000000021cc: d5010009 00024a80
	v_cndmask_b32_e64 v8, 0, v36, s0                           // 0000000021d4: d5010008 00024880
	v_add_co_u32 v116, vcc_lo, s26, v3                         // 0000000021dc: d7006a74 0202061a
	s_wait_alu depctr_va_vcc(0)                                // 0000000021e4: bf88ff9d
	v_add_co_ci_u32_e64 v117, null, s27, v4, vcc_lo            // 0000000021e8: d5207c75 01aa081b
	v_add_co_u32 v118, vcc_lo, s26, v5                         // 0000000021f0: d7006a76 02020a1a
	v_lshlrev_b64_e32 v[42:43], 2, v[1:2]                      // 0000000021f8: 3e540282
	v_lshlrev_b64_e32 v[44:45], 2, v[8:9]                      // 0000000021fc: 3e581082
	s_wait_alu depctr_va_vcc(0)                                // 000000002200: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s27, v6, vcc_lo            // 000000002204: d5207c77 01aa0c1b
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v78, 0             // 00000000220c: ca100080 5d4e0080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v76, 0             // 000000002214: ca100080 554c0080
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v72, 0             // 00000000221c: ca100080 3f480080
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v66, 0             // 000000002224: ca100080 3d420080
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v54, 0             // 00000000222c: ca100080 3b360080
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v52, 0             // 000000002234: ca100080 39340080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v50, 0             // 00000000223c: ca100080 4f320080
	v_mov_b32_e32 v71, 0                                       // 000000002244: 7e8e0280
	v_mov_b32_e32 v65, 0                                       // 000000002248: 7e820280
	v_mov_b32_e32 v55, 0                                       // 00000000224c: 7e6e0280
	v_mov_b32_e32 v53, 0                                       // 000000002250: 7e6a0280
	v_mov_b32_e32 v51, 0                                       // 000000002254: 7e660280
	v_mov_b32_e32 v49, 0                                       // 000000002258: 7e620280
	v_mov_b32_e32 v35, 0                                       // 00000000225c: 7e460280
	s_add_nc_u64 s[14:15], s[24:25], -1                        // 000000002260: a98ec118
	s_mov_b64 s[16:17], 0                                      // 000000002264: be900180
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000002268: bf8701d9
	v_mov_b32_e32 v5, s17                                      // 00000000226c: 7e0a0211
	v_or_b32_e32 v4, s16, v34                                  // 000000002270: 38084410
	v_add_co_u32 v8, vcc_lo, s16, v69                          // 000000002274: d7006a08 02028a10
	s_wait_alu depctr_va_vcc(0)                                // 00000000227c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s17, v67, vcc_lo             // 000000002280: d5207c09 01aa8611
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[4:5]                  // 000000002288: 7cb80818
	v_or_b32_e32 v6, 2, v4                                     // 00000000228c: 380c0882
	v_mov_b32_e32 v7, s17                                      // 000000002290: 7e0e0211
	v_or_b32_e32 v10, 6, v8                                    // 000000002294: 38141086
	s_or_b32 s13, s16, 16                                      // 000000002298: 8c0d9010
	v_mov_b32_e32 v123, s17                                    // 00000000229c: 7ef60211
	s_and_b32 s4, s2, vcc_lo                                   // 0000000022a0: 8b046a02
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022a4: bf88ff9e
	v_or_b32_e32 v122, s13, v34                                // 0000000022a8: 38f4440d
	v_cndmask_b32_e64 v0, 0, v8, s4                            // 0000000022ac: d5010000 00121080
	v_cndmask_b32_e64 v1, 0, v9, s4                            // 0000000022b4: d5010001 00121280
	v_mov_b32_e32 v125, s17                                    // 0000000022bc: 7efa0211
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000022c0: bf8701a3
	v_add_co_u32 v0, s5, s30, v0                               // 0000000022c4: d7000500 0202001e
	s_wait_alu depctr_va_sdst(0)                               // 0000000022cc: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s31, v1, s5                  // 0000000022d0: d5207c01 0016021f
	global_load_d16_u8 v0, v[0:1], off                         // 0000000022d8: ee07807c 00000000 00000000
	v_or_b32_e32 v1, 1, v8                                     // 0000000022e4: 38021081
	s_wait_loadcnt 0x0                                         // 0000000022e8: bfc00000
	v_cndmask_b16 v0.l, 0, v0.l, s4                            // 0000000022ec: d65d0000 00120080
	v_cmp_gt_u64_e64 s4, s[14:15], v[4:5]                      // 0000000022f4: d45c0004 0202080e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000022fc: bf870152
	v_and_b16 v0.l, 0xff, v0.l                                 // 000000002300: d7620000 020200ff 000000ff
	s_and_b32 s5, s2, s4                                       // 00000000230c: 8b050402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002310: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s5                            // 000000002314: d5010001 00160280
	v_cndmask_b32_e64 v2, 0, v9, s5                            // 00000000231c: d5010002 00161280
	v_add_co_u32 v1, s6, s30, v1                               // 000000002324: d7000601 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 00000000232c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002330: bf870002
	v_add_co_ci_u32_e64 v2, null, s31, v2, s6                  // 000000002334: d5207c02 001a041f
	global_load_d16_hi_u8 v0, v[1:2], off                      // 00000000233c: ee08407c 00000000 00000001
	v_or_b32_e32 v1, 2, v8                                     // 000000002348: 38021082
	s_wait_loadcnt 0x0                                         // 00000000234c: bfc00000
	v_cndmask_b16 v2.l, 0, v0.h, s5                            // 000000002350: d65d1002 00160080
	v_cmp_gt_u64_e64 s5, s[24:25], v[6:7]                      // 000000002358: d45c0005 02020c18
	s_delay_alu instid0(valu_dep_2)                            // 000000002360: bf870002
	v_lshlrev_b16 v2.l, 8, v2.l                                // 000000002364: d7380002 02020488
	s_and_b32 s6, s2, s5                                       // 00000000236c: 8b060502
	s_wait_alu depctr_sa_sdst(0)                               // 000000002370: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s6                            // 000000002374: d5010001 001a0280
	v_cndmask_b32_e64 v3, 0, v9, s6                            // 00000000237c: d5010003 001a1280
	v_or_b16 v0.l, v0.l, v2.l                                  // 000000002384: d7630000 02020500
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000238c: bf8701a3
	v_add_co_u32 v6, s7, s30, v1                               // 000000002390: d7000706 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002398: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v3, s7                  // 00000000239c: d5207c07 001e061f
	v_or_b32_e32 v1, 3, v8                                     // 0000000023a4: 38021083
	global_load_d16_hi_u8 v0, v[6:7], off                      // 0000000023a8: ee08407c 00000000 00000006
	v_or_b32_e32 v6, 3, v4                                     // 0000000023b4: 380c0883
	v_mov_b32_e32 v7, s17                                      // 0000000023b8: 7e0e0211
	s_wait_loadcnt 0x0                                         // 0000000023bc: bfc00000
	v_cndmask_b16 v0.h, 0, v0.h, s6                            // 0000000023c0: d65d5000 001a0080
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000023c8: bf870112
	v_cmp_gt_u64_e64 s6, s[24:25], v[6:7]                      // 0000000023cc: d45c0006 02020c18
	v_and_b16 v0.h, 0xff, v0.h op_sel:[0,1,1]                  // 0000000023d4: d7625000 020200ff 000000ff
	s_and_b32 s7, s2, s6                                       // 0000000023e0: 8b070602
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023e4: bf88ff9e
	v_cndmask_b32_e64 v1, 0, v1, s7                            // 0000000023e8: d5010001 001e0280
	v_cndmask_b32_e64 v3, 0, v9, s7                            // 0000000023f0: d5010003 001e1280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000023f8: bf870122
	v_add_co_u32 v6, s8, s30, v1                               // 0000000023fc: d7000806 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002404: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v3, s8                  // 000000002408: d5207c07 0022061f
	global_load_d16_u8 v1, v[6:7], off                         // 000000002410: ee07807c 00000001 00000006
	v_or_b32_e32 v6, 4, v4                                     // 00000000241c: 380c0884
	v_mov_b32_e32 v7, s17                                      // 000000002420: 7e0e0211
	s_wait_loadcnt 0x0                                         // 000000002424: bfc00000
	v_cndmask_b16 v2.h, 0, v1.l, s7                            // 000000002428: d65d4002 001e0280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002430: bf8701a2
	v_cmp_gt_u64_e64 s7, s[24:25], v[6:7]                      // 000000002434: d45c0007 02020c18
	v_or_b32_e32 v1, 4, v8                                     // 00000000243c: 38021084
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002440: d7385002 02020488
	s_and_b32 s8, s2, s7                                       // 000000002448: 8b080702
	s_wait_alu depctr_sa_sdst(0)                               // 00000000244c: bf88ff9e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002450: bf8701b2
	v_cndmask_b32_e64 v1, 0, v1, s8                            // 000000002454: d5010001 00220280
	v_cndmask_b32_e64 v3, 0, v9, s8                            // 00000000245c: d5010003 00221280
	v_or_b16 v0.h, v0.h, v2.h op_sel:[1,1,1]                   // 000000002464: d7635800 02020500
	v_add_co_u32 v6, s9, s30, v1                               // 00000000246c: d7000906 0202021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002474: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002478: bf870003
	v_add_co_ci_u32_e64 v7, null, s31, v3, s9                  // 00000000247c: d5207c07 0026061f
	v_or_b32_e32 v3, 5, v8                                     // 000000002484: 38061085
	global_load_d16_u8 v1, v[6:7], off                         // 000000002488: ee07807c 00000001 00000006
	v_or_b32_e32 v6, 5, v4                                     // 000000002494: 380c0885
	v_mov_b32_e32 v7, s17                                      // 000000002498: 7e0e0211
	s_wait_loadcnt 0x0                                         // 00000000249c: bfc00000
	v_cndmask_b16 v1.l, 0, v1.l, s8                            // 0000000024a0: d65d0001 00220280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000024a8: bf870112
	v_cmp_gt_u64_e64 s8, s[24:25], v[6:7]                      // 0000000024ac: d45c0008 02020c18
	v_and_b16 v1.l, 0xff, v1.l                                 // 0000000024b4: d7620001 020202ff 000000ff
	s_and_b32 s9, s2, s8                                       // 0000000024c0: 8b090802
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024c4: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s9                            // 0000000024c8: d5010003 00260680
	v_cndmask_b32_e64 v7, 0, v9, s9                            // 0000000024d0: d5010007 00261280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000024d8: bf870122
	v_add_co_u32 v6, s10, s30, v3                              // 0000000024dc: d7000a06 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 0000000024e4: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v7, s10                 // 0000000024e8: d5207c07 002a0e1f
	global_load_d16_hi_u8 v1, v[6:7], off                      // 0000000024f0: ee08407c 00000001 00000006
	v_or_b32_e32 v6, 6, v4                                     // 0000000024fc: 380c0886
	v_mov_b32_e32 v7, s17                                      // 000000002500: 7e0e0211
	v_or_b32_e32 v4, 7, v4                                     // 000000002504: 38080887
	s_wait_loadcnt 0x0                                         // 000000002508: bfc00000
	v_cndmask_b16 v3.l, 0, v1.h, s9                            // 00000000250c: d65d1003 00260280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002514: bf870113
	v_cmp_gt_u64_e64 s9, s[24:25], v[6:7]                      // 000000002518: d45c0009 02020c18
	v_lshlrev_b16 v3.l, 8, v3.l                                // 000000002520: d7380003 02020688
	s_and_b32 s10, s2, s9                                      // 000000002528: 8b0a0902
	s_wait_alu depctr_sa_sdst(0)                               // 00000000252c: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v10, s10                          // 000000002530: d5010006 002a1480
	v_cndmask_b32_e64 v7, 0, v9, s10                           // 000000002538: d5010007 002a1280
	v_or_b16 v1.l, v1.l, v3.l                                  // 000000002540: d7630001 02020701
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002548: bf8701a3
	v_add_co_u32 v6, s11, s30, v6                              // 00000000254c: d7000b06 02020c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002554: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v7, s11                 // 000000002558: d5207c07 002e0e1f
	global_load_d16_hi_u8 v1, v[6:7], off                      // 000000002560: ee08407c 00000001 00000006
	v_or_b32_e32 v6, 7, v8                                     // 00000000256c: 380c1087
	s_wait_loadcnt 0x0                                         // 000000002570: bfc00000
	v_cndmask_b16 v1.h, 0, v1.h, s10                           // 000000002574: d65d5001 002a0280
	v_cmp_gt_u64_e64 s10, s[24:25], v[4:5]                     // 00000000257c: d45c000a 02020818
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002584: bf870152
	v_and_b16 v1.h, 0xff, v1.h op_sel:[0,1,1]                  // 000000002588: d7625001 020202ff 000000ff
	s_and_b32 s11, s2, s10                                     // 000000002594: 8b0b0a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002598: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v6, s11                           // 00000000259c: d5010004 002e0c80
	v_cndmask_b32_e64 v5, 0, v9, s11                           // 0000000025a4: d5010005 002e1280
	v_add_co_u32 v4, s12, s30, v4                              // 0000000025ac: d7000c04 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 0000000025b4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000025b8: bf870002
	v_add_co_ci_u32_e64 v5, null, s31, v5, s12                 // 0000000025bc: d5207c05 00320a1f
	global_load_d16_hi_u8 v3, v[4:5], off                      // 0000000025c4: ee08407c 00000003 00000004
	s_wait_loadcnt 0x0                                         // 0000000025d0: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 0000000025d4: d65d5003 002e0680
	v_add_co_u32 v7, s11, s16, v70                             // 0000000025dc: d7000b07 02028c10
	s_wait_alu depctr_va_sdst(0)                               // 0000000025e4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v68, s11                // 0000000025e8: d5207c08 002e8811
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 0000000025f0: bf870143
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 0000000025f4: d7385003 02020688
	s_and_b32 s11, s3, vcc_lo                                  // 0000000025fc: 8b0b6a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000002600: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v7, s11                           // 000000002604: d5010002 002e0e80
	v_or_b16 v1.h, v1.h, v3.h op_sel:[1,1,1]                   // 00000000260c: d7635801 02020701
	v_cndmask_b32_e64 v3, 0, v8, s11                           // 000000002614: d5010003 002e1080
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 00000000261c: bf870123
	v_add_co_u32 v2, s12, s30, v2                              // 000000002620: d7000c02 0202041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002628: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s31, v3, s12                 // 00000000262c: d5207c03 0032061f
	global_load_d16_u8 v2, v[2:3], off                         // 000000002634: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 1, v7                                     // 000000002640: 38060e81
	s_wait_loadcnt 0x0                                         // 000000002644: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s11                           // 000000002648: d65d0002 002e0480
	s_and_b32 s11, s3, s4                                      // 000000002650: 8b0b0403
	s_wait_alu depctr_sa_sdst(0)                               // 000000002654: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 000000002658: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v8, s11                           // 000000002660: d5010004 002e1080
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002668: d7620002 020204ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002674: bf8701a3
	v_add_co_u32 v3, s12, s30, v3                              // 000000002678: d7000c03 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 000000002680: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s31, v4, s12                 // 000000002684: d5207c04 0032081f
	global_load_d16_hi_u8 v2, v[3:4], off                      // 00000000268c: ee08407c 00000002 00000003
	v_or_b32_e32 v3, 2, v7                                     // 000000002698: 38060e82
	s_wait_loadcnt 0x0                                         // 00000000269c: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s11                           // 0000000026a0: d65d5002 002e0480
	s_and_b32 s11, s3, s5                                      // 0000000026a8: 8b0b0503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026ac: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 0000000026b0: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v8, s11                           // 0000000026b8: d5010004 002e1080
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000026c0: d7385002 02020488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000026c8: bf8701a3
	v_add_co_u32 v3, s12, s30, v3                              // 0000000026cc: d7000c03 0202061e
	s_wait_alu depctr_va_sdst(0)                               // 0000000026d4: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s31, v4, s12                 // 0000000026d8: d5207c04 0032081f
	s_delay_alu instid0(valu_dep_3)                            // 0000000026e0: bf870003
	v_or_b16 v46.l, v2.l, v2.h op_sel:[0,1,0]                  // 0000000026e4: d763102e 02020502
	global_load_d16_u8 v3, v[3:4], off                         // 0000000026ec: ee07807c 00000003 00000003
	v_or_b32_e32 v4, 3, v7                                     // 0000000026f8: 38080e83
	s_wait_loadcnt 0x0                                         // 0000000026fc: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s11                           // 000000002700: d65d0003 002e0680
	s_and_b32 s11, s3, s6                                      // 000000002708: 8b0b0603
	s_wait_alu depctr_sa_sdst(0)                               // 00000000270c: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002710: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v8, s11                           // 000000002718: d5010005 002e1080
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002720: d7620003 020206ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000272c: bf8701a3
	v_add_co_u32 v4, s12, s30, v4                              // 000000002730: d7000c04 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002738: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s12                 // 00000000273c: d5207c05 00320a1f
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002744: ee08407c 00000003 00000004
	v_or_b32_e32 v4, 4, v7                                     // 000000002750: 38080e84
	s_wait_loadcnt 0x0                                         // 000000002754: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 000000002758: d65d5003 002e0680
	s_and_b32 s11, s3, s7                                      // 000000002760: 8b0b0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000002764: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002768: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v8, s11                           // 000000002770: d5010005 002e1080
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002778: d7385003 02020688
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002780: bf8701a3
	v_add_co_u32 v4, s12, s30, v4                              // 000000002784: d7000c04 0202081e
	s_wait_alu depctr_va_sdst(0)                               // 00000000278c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s31, v5, s12                 // 000000002790: d5207c05 00320a1f
	s_delay_alu instid0(valu_dep_3)                            // 000000002798: bf870003
	v_or_b16 v46.h, v3.l, v3.h op_sel:[0,1,1]                  // 00000000279c: d763502e 02020703
	global_load_d16_u8 v4, v[4:5], off                         // 0000000027a4: ee07807c 00000004 00000004
	v_or_b32_e32 v5, 5, v7                                     // 0000000027b0: 380a0e85
	s_wait_loadcnt 0x0                                         // 0000000027b4: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, s11                           // 0000000027b8: d65d0004 002e0880
	s_and_b32 s11, s3, s8                                      // 0000000027c0: 8b0b0803
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027c4: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 0000000027c8: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v8, s11                           // 0000000027d0: d5010006 002e1080
	v_and_b16 v4.l, 0xff, v4.l                                 // 0000000027d8: d7620004 020208ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000027e4: bf8701a3
	v_add_co_u32 v5, s12, s30, v5                              // 0000000027e8: d7000c05 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000027f0: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s12                 // 0000000027f4: d5207c06 00320c1f
	global_load_d16_hi_u8 v4, v[5:6], off                      // 0000000027fc: ee08407c 00000004 00000005
	v_or_b32_e32 v5, 6, v7                                     // 000000002808: 380a0e86
	s_wait_loadcnt 0x0                                         // 00000000280c: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, s11                           // 000000002810: d65d5004 002e0880
	s_and_b32 s11, s3, s9                                      // 000000002818: 8b0b0903
	s_wait_alu depctr_sa_sdst(0)                               // 00000000281c: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002820: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v8, s11                           // 000000002828: d5010006 002e1080
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000002830: d7385004 02020888
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002838: bf8701a3
	v_add_co_u32 v5, s12, s30, v5                              // 00000000283c: d7000c05 02020a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002844: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s31, v6, s12                 // 000000002848: d5207c06 00320c1f
	s_delay_alu instid0(valu_dep_3)                            // 000000002850: bf870003
	v_or_b16 v47.l, v4.l, v4.h op_sel:[0,1,0]                  // 000000002854: d763102f 02020904
	global_load_d16_u8 v5, v[5:6], off                         // 00000000285c: ee07807c 00000005 00000005
	v_or_b32_e32 v6, 7, v7                                     // 000000002868: 380c0e87
	s_wait_loadcnt 0x0                                         // 00000000286c: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, s11                           // 000000002870: d65d0005 002e0a80
	s_and_b32 s11, s3, s10                                     // 000000002878: 8b0b0a03
	s_wait_alu depctr_sa_sdst(0)                               // 00000000287c: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002880: d5010006 002e0c80
	v_cndmask_b32_e64 v7, 0, v8, s11                           // 000000002888: d5010007 002e1080
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000002890: d7620005 02020aff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000289c: bf8701a3
	v_add_co_u32 v6, s12, s30, v6                              // 0000000028a0: d7000c06 02020c1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000028a8: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s31, v7, s12                 // 0000000028ac: d5207c07 00320e1f
	global_load_d16_hi_u8 v5, v[6:7], off                      // 0000000028b4: ee08407c 00000005 00000006
	s_wait_loadcnt 0x0                                         // 0000000028c0: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, s11                           // 0000000028c4: d65d5005 002e0a80
	v_add_co_u32 v7, s11, s16, v74                             // 0000000028cc: d7000b07 02029410
	s_wait_alu depctr_va_sdst(0)                               // 0000000028d4: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s17, v73, s11                // 0000000028d8: d5207c08 002e9211
	s_and_b32 s11, s1, vcc_lo                                  // 0000000028e0: 8b0b6a01
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 0000000028e4: d7385005 02020a88
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028ec: bf88ff9e
	v_cndmask_b32_e64 v2, 0, v7, s11                           // 0000000028f0: d5010002 002e0e80
	v_cndmask_b32_e64 v3, 0, v8, s11                           // 0000000028f8: d5010003 002e1080
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000002900: 8b6a6a00
	v_or_b16 v47.h, v5.l, v5.h op_sel:[0,1,1]                  // 000000002904: d763502f 02020b05
	s_delay_alu instid0(valu_dep_3)                            // 00000000290c: bf870003
	v_add_co_u32 v2, s12, s34, v2                              // 000000002910: d7000c02 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000002918: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s35, v3, s12                 // 00000000291c: d5207c03 00320623
	global_load_d16_u8 v2, v[2:3], off                         // 000000002924: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 1, v7                                     // 000000002930: 38060e81
	s_wait_loadcnt 0x0                                         // 000000002934: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, s11                           // 000000002938: d65d0002 002e0480
	s_and_b32 s11, s1, s4                                      // 000000002940: 8b0b0401
	s_wait_alu depctr_sa_sdst(0)                               // 000000002944: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 000000002948: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v8, s11                           // 000000002950: d5010004 002e1080
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002958: d7620002 020204ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002964: bf8701a3
	v_add_co_u32 v3, s12, s34, v3                              // 000000002968: d7000c03 02020622
	s_wait_alu depctr_va_sdst(0)                               // 000000002970: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s35, v4, s12                 // 000000002974: d5207c04 00320823
	global_load_d16_hi_u8 v2, v[3:4], off                      // 00000000297c: ee08407c 00000002 00000003
	v_or_b32_e32 v3, 2, v7                                     // 000000002988: 38060e82
	s_wait_loadcnt 0x0                                         // 00000000298c: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, s11                           // 000000002990: d65d5002 002e0480
	s_and_b32 s11, s1, s5                                      // 000000002998: 8b0b0501
	s_wait_alu depctr_sa_sdst(0)                               // 00000000299c: bf88ff9e
	v_cndmask_b32_e64 v3, 0, v3, s11                           // 0000000029a0: d5010003 002e0680
	v_cndmask_b32_e64 v4, 0, v8, s11                           // 0000000029a8: d5010004 002e1080
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 0000000029b0: d7385002 02020488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000029b8: bf8701a3
	v_add_co_u32 v3, s12, s34, v3                              // 0000000029bc: d7000c03 02020622
	s_wait_alu depctr_va_sdst(0)                               // 0000000029c4: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s35, v4, s12                 // 0000000029c8: d5207c04 00320823
	global_load_d16_u8 v3, v[3:4], off                         // 0000000029d0: ee07807c 00000003 00000003
	v_or_b32_e32 v4, 3, v7                                     // 0000000029dc: 38080e83
	s_wait_loadcnt 0x0                                         // 0000000029e0: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, s11                           // 0000000029e4: d65d0003 002e0680
	s_and_b32 s11, s1, s6                                      // 0000000029ec: 8b0b0601
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029f0: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 0000000029f4: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v8, s11                           // 0000000029fc: d5010005 002e1080
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002a04: d7620003 020206ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a10: bf8701a3
	v_add_co_u32 v4, s12, s34, v4                              // 000000002a14: d7000c04 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000002a1c: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s12                 // 000000002a20: d5207c05 00320a23
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002a28: ee08407c 00000003 00000004
	v_or_b32_e32 v4, 4, v7                                     // 000000002a34: 38080e84
	s_wait_loadcnt 0x0                                         // 000000002a38: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 000000002a3c: d65d5003 002e0680
	s_and_b32 s11, s1, s7                                      // 000000002a44: 8b0b0701
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a48: bf88ff9e
	v_cndmask_b32_e64 v4, 0, v4, s11                           // 000000002a4c: d5010004 002e0880
	v_cndmask_b32_e64 v5, 0, v8, s11                           // 000000002a54: d5010005 002e1080
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002a5c: d7385003 02020688
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a64: bf8701a3
	v_add_co_u32 v4, s12, s34, v4                              // 000000002a68: d7000c04 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000002a70: bf88f19f
	v_add_co_ci_u32_e64 v5, null, s35, v5, s12                 // 000000002a74: d5207c05 00320a23
	global_load_d16_u8 v4, v[4:5], off                         // 000000002a7c: ee07807c 00000004 00000004
	v_or_b32_e32 v5, 5, v7                                     // 000000002a88: 380a0e85
	s_wait_loadcnt 0x0                                         // 000000002a8c: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, s11                           // 000000002a90: d65d0004 002e0880
	s_and_b32 s11, s1, s8                                      // 000000002a98: 8b0b0801
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a9c: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002aa0: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v8, s11                           // 000000002aa8: d5010006 002e1080
	v_and_b16 v4.l, 0xff, v4.l                                 // 000000002ab0: d7620004 020208ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002abc: bf8701a3
	v_add_co_u32 v5, s12, s34, v5                              // 000000002ac0: d7000c05 02020a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002ac8: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s35, v6, s12                 // 000000002acc: d5207c06 00320c23
	global_load_d16_hi_u8 v4, v[5:6], off                      // 000000002ad4: ee08407c 00000004 00000005
	v_or_b32_e32 v5, 6, v7                                     // 000000002ae0: 380a0e86
	s_wait_loadcnt 0x0                                         // 000000002ae4: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, s11                           // 000000002ae8: d65d5004 002e0880
	s_and_b32 s11, s1, s9                                      // 000000002af0: 8b0b0901
	s_wait_alu depctr_sa_sdst(0)                               // 000000002af4: bf88ff9e
	v_cndmask_b32_e64 v5, 0, v5, s11                           // 000000002af8: d5010005 002e0a80
	v_cndmask_b32_e64 v6, 0, v8, s11                           // 000000002b00: d5010006 002e1080
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000002b08: d7385004 02020888
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002b10: bf8701a3
	v_add_co_u32 v5, s12, s34, v5                              // 000000002b14: d7000c05 02020a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002b1c: bf88f19f
	v_add_co_ci_u32_e64 v6, null, s35, v6, s12                 // 000000002b20: d5207c06 00320c23
	global_load_d16_u8 v5, v[5:6], off                         // 000000002b28: ee07807c 00000005 00000005
	v_or_b32_e32 v6, 7, v7                                     // 000000002b34: 380c0e87
	s_wait_loadcnt 0x0                                         // 000000002b38: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, s11                           // 000000002b3c: d65d0005 002e0a80
	s_and_b32 s11, s1, s10                                     // 000000002b44: 8b0b0a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b48: bf88ff9e
	v_cndmask_b32_e64 v6, 0, v6, s11                           // 000000002b4c: d5010006 002e0c80
	v_cndmask_b32_e64 v7, 0, v8, s11                           // 000000002b54: d5010007 002e1080
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000002b5c: d7620005 02020aff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002b68: bf8701a3
	v_add_co_u32 v6, s12, s34, v6                              // 000000002b6c: d7000c06 02020c22
	s_wait_alu depctr_va_sdst(0)                               // 000000002b74: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s35, v7, s12                 // 000000002b78: d5207c07 00320e23
	global_load_d16_hi_u8 v5, v[6:7], off                      // 000000002b80: ee08407c 00000005 00000006
	v_or_b16 v6.l, v2.l, v2.h op_sel:[0,1,0]                   // 000000002b8c: d7631006 02020502
	v_or_b16 v6.h, v3.l, v3.h op_sel:[0,1,1]                   // 000000002b94: d7635006 02020703
	v_or_b16 v7.l, v4.l, v4.h op_sel:[0,1,0]                   // 000000002b9c: d7631007 02020904
	s_wait_loadcnt 0x0                                         // 000000002ba4: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, s11                           // 000000002ba8: d65d5005 002e0a80
	v_add_co_u32 v10, s11, s16, v77                            // 000000002bb0: d7000b0a 02029a10
	s_wait_alu depctr_va_sdst(0)                               // 000000002bb8: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s17, v75, s11               // 000000002bbc: d5207c0b 002e9611
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002bc4: bf870113
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 000000002bc8: d7385005 02020a88
	v_dual_cndmask_b32 v2, 0, v10 :: v_dual_cndmask_b32 v3, 0, v11// 000000002bd0: ca521480 02021680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002bd8: bf870112
	v_or_b16 v7.h, v5.l, v5.h op_sel:[0,1,1]                   // 000000002bdc: d7635007 02020b05
	v_add_co_u32 v2, s11, s34, v2                              // 000000002be4: d7000b02 02020422
	s_wait_alu depctr_va_sdst(0)                               // 000000002bec: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002bf0: bf870193
	v_add_co_ci_u32_e64 v3, null, s35, v3, s11                 // 000000002bf4: d5207c03 002e0623
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[0:1], v[6:7], 0    // 000000002bfc: cc464018 1a020d00
	global_load_d16_u8 v2, v[2:3], off                         // 000000002c04: ee07807c 00000002 00000002
	v_or_b32_e32 v3, 1, v10                                    // 000000002c10: 38061481
	s_wait_loadcnt 0x0                                         // 000000002c14: bfc00000
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 000000002c18: d65d0002 01aa0480
	s_and_b32 vcc_lo, s0, s4                                   // 000000002c20: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c24: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000002c28: 02060680
	v_cndmask_b32_e32 v4, 0, v11, vcc_lo                       // 000000002c2c: 02081680
	v_and_b16 v2.l, 0xff, v2.l                                 // 000000002c30: d7620002 020204ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c3c: bf8701a3
	v_add_co_u32 v3, s4, s34, v3                               // 000000002c40: d7000403 02020622
	s_wait_alu depctr_va_sdst(0)                               // 000000002c48: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s35, v4, s4                  // 000000002c4c: d5207c04 00120823
	global_load_d16_hi_u8 v2, v[3:4], off                      // 000000002c54: ee08407c 00000002 00000003
	v_or_b32_e32 v3, 2, v10                                    // 000000002c60: 38061482
	s_wait_loadcnt 0x0                                         // 000000002c64: bfc00000
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 000000002c68: d65d5002 01aa0480
	s_and_b32 vcc_lo, s0, s5                                   // 000000002c70: 8b6a0500
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c74: bf88ff9e
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo                        // 000000002c78: 02060680
	v_cndmask_b32_e32 v4, 0, v11, vcc_lo                       // 000000002c7c: 02081680
	v_lshlrev_b16 v2.h, 8, v2.h op_sel:[0,1,1]                 // 000000002c80: d7385002 02020488
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c88: bf8701a3
	v_add_co_u32 v3, s4, s34, v3                               // 000000002c8c: d7000403 02020622
	s_wait_alu depctr_va_sdst(0)                               // 000000002c94: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s35, v4, s4                  // 000000002c98: d5207c04 00120823
	s_delay_alu instid0(valu_dep_3)                            // 000000002ca0: bf870003
	v_or_b16 v120.l, v2.l, v2.h op_sel:[0,1,0]                 // 000000002ca4: d7631078 02020502
	global_load_d16_u8 v3, v[3:4], off                         // 000000002cac: ee07807c 00000003 00000003
	v_or_b32_e32 v4, 3, v10                                    // 000000002cb8: 38081483
	s_wait_loadcnt 0x0                                         // 000000002cbc: bfc00000
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 000000002cc0: d65d0003 01aa0680
	s_and_b32 vcc_lo, s0, s6                                   // 000000002cc8: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ccc: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v11// 000000002cd0: ca520880 04041680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002cd8: bf870112
	v_and_b16 v3.l, 0xff, v3.l                                 // 000000002cdc: d7620003 020206ff 000000ff
	v_add_co_u32 v4, s4, s34, v4                               // 000000002ce8: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000002cf0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002cf4: bf870003
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000002cf8: d5207c05 00120a23
	global_load_d16_hi_u8 v3, v[4:5], off                      // 000000002d00: ee08407c 00000003 00000004
	v_or_b32_e32 v4, 4, v10                                    // 000000002d0c: 38081484
	s_wait_loadcnt 0x0                                         // 000000002d10: bfc00000
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 000000002d14: d65d5003 01aa0680
	s_and_b32 vcc_lo, s0, s7                                   // 000000002d1c: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d20: bf88ff9e
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_cndmask_b32 v5, 0, v11// 000000002d24: ca520880 04041680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002d2c: bf870112
	v_lshlrev_b16 v3.h, 8, v3.h op_sel:[0,1,1]                 // 000000002d30: d7385003 02020688
	v_add_co_u32 v4, s4, s34, v4                               // 000000002d38: d7000404 02020822
	s_wait_alu depctr_va_sdst(0)                               // 000000002d40: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002d44: bf870193
	v_add_co_ci_u32_e64 v5, null, s35, v5, s4                  // 000000002d48: d5207c05 00120a23
	v_or_b16 v120.h, v3.l, v3.h op_sel:[0,1,1]                 // 000000002d50: d7635078 02020703
	global_load_d16_u8 v4, v[4:5], off                         // 000000002d58: ee07807c 00000004 00000004
	v_or_b32_e32 v5, 5, v10                                    // 000000002d64: 380a1485
	s_wait_loadcnt 0x0                                         // 000000002d68: bfc00000
	v_cndmask_b16 v4.l, 0, v4.l, vcc_lo                        // 000000002d6c: d65d0004 01aa0880
	s_and_b32 vcc_lo, s0, s8                                   // 000000002d74: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d78: bf88ff9e
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000002d7c: 020a0a80
	v_cndmask_b32_e32 v9, 0, v11, vcc_lo                       // 000000002d80: 02121680
	v_and_b16 v4.l, 0xff, v4.l                                 // 000000002d84: d7620004 020208ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002d90: bf8701a3
	v_add_co_u32 v8, s4, s34, v5                               // 000000002d94: d7000408 02020a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002d9c: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s35, v9, s4                  // 000000002da0: d5207c09 00121223
	v_or_b32_e32 v5, 6, v10                                    // 000000002da8: 380a1486
	global_load_d16_hi_u8 v4, v[8:9], off                      // 000000002dac: ee08407c 00000004 00000008
	s_wait_loadcnt 0x0                                         // 000000002db8: bfc00000
	v_cndmask_b16 v4.h, 0, v4.h, vcc_lo                        // 000000002dbc: d65d5004 01aa0880
	s_and_b32 vcc_lo, s0, s9                                   // 000000002dc4: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dc8: bf88ff9e
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000002dcc: 020a0a80
	v_cndmask_b32_e32 v9, 0, v11, vcc_lo                       // 000000002dd0: 02121680
	v_lshlrev_b16 v4.h, 8, v4.h op_sel:[0,1,1]                 // 000000002dd4: d7385004 02020888
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002ddc: bf8701a3
	v_add_co_u32 v8, s4, s34, v5                               // 000000002de0: d7000408 02020a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002de8: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s35, v9, s4                  // 000000002dec: d5207c09 00121223
	s_delay_alu instid0(valu_dep_3)                            // 000000002df4: bf870003
	v_or_b16 v121.l, v4.l, v4.h op_sel:[0,1,0]                 // 000000002df8: d7631079 02020904
	global_load_d16_u8 v5, v[8:9], off                         // 000000002e00: ee07807c 00000005 00000008
	v_or_b32_e32 v8, 7, v10                                    // 000000002e0c: 38101487
	s_wait_loadcnt 0x0                                         // 000000002e10: bfc00000
	v_cndmask_b16 v5.l, 0, v5.l, vcc_lo                        // 000000002e14: d65d0005 01aa0a80
	s_and_b32 vcc_lo, s0, s10                                  // 000000002e1c: 8b6a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e20: bf88ff9e
	v_dual_cndmask_b32 v8, 0, v8 :: v_dual_cndmask_b32 v9, 0, v11// 000000002e24: ca521080 08081680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002e2c: bf870112
	v_and_b16 v5.l, 0xff, v5.l                                 // 000000002e30: d7620005 02020aff 000000ff
	v_add_co_u32 v8, s4, s34, v8                               // 000000002e3c: d7000408 02021022
	s_wait_alu depctr_va_sdst(0)                               // 000000002e44: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002e48: bf870003
	v_add_co_ci_u32_e64 v9, null, s35, v9, s4                  // 000000002e4c: d5207c09 00121223
	global_load_d16_hi_u8 v5, v[8:9], off                      // 000000002e54: ee08407c 00000005 00000008
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[46:47], v[6:7], 0   // 000000002e60: cc464008 1a020d2e
	s_wait_loadcnt 0x0                                         // 000000002e68: bfc00000
	v_cndmask_b16 v5.h, 0, v5.h, vcc_lo                        // 000000002e6c: d65d5005 01aa0a80
	v_add_co_u32 v126, vcc_lo, s13, v69                        // 000000002e74: d7006a7e 02028a0d
	s_wait_alu depctr_va_vcc(0)                                // 000000002e7c: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s17, v67, vcc_lo           // 000000002e80: d5207c7f 01aa8611
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000002e88: bf8701b3
	v_lshlrev_b16 v5.h, 8, v5.h op_sel:[0,1,1]                 // 000000002e8c: d7385005 02020a88
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[122:123]              // 000000002e94: 7cb8f418
	v_or_b32_e32 v124, 3, v126                                 // 000000002e98: 38f8fc83
	v_or_b16 v121.h, v5.l, v5.h op_sel:[0,1,1]                 // 000000002e9c: d7635079 02020b05
	s_and_b32 s4, s2, vcc_lo                                   // 000000002ea4: 8b046a02
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000002ea8: bf870151
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[0:1], v[120:121], 0// 000000002eac: cc464010 1a02f100
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[46:47], v[120:121], 0// 000000002eb4: cc464000 1a02f12e
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ebc: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v126, s4                         // 000000002ec0: d501002e 0012fc80
	v_cndmask_b32_e64 v47, 0, v127, s4                         // 000000002ec8: d501002f 0012fe80
	v_add_co_u32 v46, s5, s30, v46                             // 000000002ed0: d700052e 02025c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ed8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002edc: bf870002
	v_add_co_ci_u32_e64 v47, null, s31, v47, s5                // 000000002ee0: d5207c2f 00165e1f
	global_load_d16_u8 v46, v[46:47], off                      // 000000002ee8: ee07807c 0000002e 0000002e
	v_or_b32_e32 v47, 1, v126                                  // 000000002ef4: 385efc81
	s_wait_loadcnt 0x0                                         // 000000002ef8: bfc00000
	v_cndmask_b16 v46.l, 0, v46.l, s4                          // 000000002efc: d65d002e 00125c80
	v_cmp_gt_u64_e64 s4, s[14:15], v[122:123]                  // 000000002f04: d45c0004 0202f40e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 000000002f0c: bf870152
	v_and_b16 v46.l, 0xff, v46.l                               // 000000002f10: d762002e 02025cff 000000ff
	s_and_b32 s5, s2, s4                                       // 000000002f1c: 8b050402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f20: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s5                          // 000000002f24: d501002f 00165e80
	v_cndmask_b32_e64 v121, 0, v127, s5                        // 000000002f2c: d5010079 0016fe80
	v_add_co_u32 v120, s6, s30, v47                            // 000000002f34: d7000678 02025e1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002f3c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002f40: bf870002
	v_add_co_ci_u32_e64 v121, null, s31, v121, s6              // 000000002f44: d5207c79 001af21f
	v_or_b32_e32 v47, 2, v126                                  // 000000002f4c: 385efc82
	global_load_d16_hi_u8 v46, v[120:121], off                 // 000000002f50: ee08407c 0000002e 00000078
	v_or_b32_e32 v120, 2, v122                                 // 000000002f5c: 38f0f482
	v_mov_b32_e32 v121, s17                                    // 000000002f60: 7ef20211
	s_wait_loadcnt 0x0                                         // 000000002f64: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, s5                          // 000000002f68: d65d502e 00165c80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002f70: bf870112
	v_cmp_gt_u64_e64 s5, s[24:25], v[120:121]                  // 000000002f74: d45c0005 0202f018
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000002f7c: d738502e 02025c88
	s_and_b32 s6, s2, s5                                       // 000000002f84: 8b060502
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f88: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s6                          // 000000002f8c: d501002f 001a5e80
	v_cndmask_b32_e64 v121, 0, v127, s6                        // 000000002f94: d5010079 001afe80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f9c: bf870122
	v_add_co_u32 v120, s7, s30, v47                            // 000000002fa0: d7000778 02025e1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002fa8: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s7              // 000000002fac: d5207c79 001ef21f
	global_load_d16_u8 v47, v[120:121], off                    // 000000002fb4: ee07807c 0000002f 00000078
	v_or_b32_e32 v120, 3, v122                                 // 000000002fc0: 38f0f483
	v_mov_b32_e32 v121, s17                                    // 000000002fc4: 7ef20211
	s_wait_loadcnt 0x0                                         // 000000002fc8: bfc00000
	v_cndmask_b16 v47.l, 0, v47.l, s6                          // 000000002fcc: d65d002f 001a5e80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002fd4: bf870112
	v_cmp_gt_u64_e64 s6, s[24:25], v[120:121]                  // 000000002fd8: d45c0006 0202f018
	v_and_b16 v47.l, 0xff, v47.l                               // 000000002fe0: d762002f 02025eff 000000ff
	s_and_b32 s7, s2, s6                                       // 000000002fec: 8b070602
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ff0: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v124, s7                        // 000000002ff4: d5010078 001ef880
	v_cndmask_b32_e64 v121, 0, v127, s7                        // 000000002ffc: d5010079 001efe80
	v_or_b32_e32 v124, 4, v126                                 // 000000003004: 38f8fc84
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003008: bf8701a3
	v_add_co_u32 v120, s8, s30, v120                           // 00000000300c: d7000878 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000003014: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s8              // 000000003018: d5207c79 0022f21f
	global_load_d16_hi_u8 v47, v[120:121], off                 // 000000003020: ee08407c 0000002f 00000078
	v_or_b32_e32 v120, 4, v122                                 // 00000000302c: 38f0f484
	v_mov_b32_e32 v121, s17                                    // 000000003030: 7ef20211
	s_wait_loadcnt 0x0                                         // 000000003034: bfc00000
	v_cndmask_b16 v47.h, 0, v47.h, s7                          // 000000003038: d65d502f 001e5e80
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003040: bf870112
	v_cmp_gt_u64_e64 s7, s[24:25], v[120:121]                  // 000000003044: d45c0007 0202f018
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 00000000304c: d738502f 02025e88
	s_and_b32 s8, s2, s7                                       // 000000003054: 8b080702
	s_wait_alu depctr_sa_sdst(0)                               // 000000003058: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v124, s8                        // 00000000305c: d5010078 0022f880
	v_cndmask_b32_e64 v121, 0, v127, s8                        // 000000003064: d5010079 0022fe80
	v_or_b32_e32 v124, 5, v122                                 // 00000000306c: 38f8f485
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003070: bf8701a3
	v_add_co_u32 v120, s9, s30, v120                           // 000000003074: d7000978 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 00000000307c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s9              // 000000003080: d5207c79 0026f21f
	global_load_d16_u8 v120, v[120:121], off                   // 000000003088: ee07807c 00000078 00000078
	v_or_b32_e32 v121, 5, v126                                 // 000000003094: 38f2fc85
	s_wait_loadcnt 0x0                                         // 000000003098: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s8                        // 00000000309c: d65d0078 0022f080
	v_cmp_gt_u64_e64 s8, s[24:25], v[124:125]                  // 0000000030a4: d45c0008 0202f818
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000030ac: bf870152
	v_and_b16 v120.l, 0xff, v120.l                             // 0000000030b0: d7620078 0202f0ff 000000ff
	s_and_b32 s9, s2, s8                                       // 0000000030bc: 8b090802
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030c0: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s9                        // 0000000030c4: d5010079 0026f280
	v_cndmask_b32_e64 v125, 0, v127, s9                        // 0000000030cc: d501007d 0026fe80
	v_add_co_u32 v124, s10, s30, v121                          // 0000000030d4: d7000a7c 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030dc: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000030e0: bf870002
	v_add_co_ci_u32_e64 v125, null, s31, v125, s10             // 0000000030e4: d5207c7d 002afa1f
	v_or_b32_e32 v121, 6, v126                                 // 0000000030ec: 38f2fc86
	global_load_d16_hi_u8 v120, v[124:125], off                // 0000000030f0: ee08407c 00000078 0000007c
	v_or_b32_e32 v124, 6, v122                                 // 0000000030fc: 38f8f486
	v_mov_b32_e32 v125, s17                                    // 000000003100: 7efa0211
	v_or_b32_e32 v122, 7, v122                                 // 000000003104: 38f4f487
	s_wait_loadcnt 0x0                                         // 000000003108: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s9                        // 00000000310c: d65d5078 0026f080
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003114: bf870113
	v_cmp_gt_u64_e64 s9, s[24:25], v[124:125]                  // 000000003118: d45c0009 0202f818
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000003120: d7385078 0202f088
	s_and_b32 s10, s2, s9                                      // 000000003128: 8b0a0902
	s_wait_alu depctr_sa_sdst(0)                               // 00000000312c: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s10                       // 000000003130: d5010079 002af280
	v_cndmask_b32_e64 v125, 0, v127, s10                       // 000000003138: d501007d 002afe80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003140: bf870122
	v_add_co_u32 v124, s11, s30, v121                          // 000000003144: d7000b7c 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 00000000314c: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s11             // 000000003150: d5207c7d 002efa1f
	global_load_d16_u8 v121, v[124:125], off                   // 000000003158: ee07807c 00000079 0000007c
	v_or_b32_e32 v124, 7, v126                                 // 000000003164: 38f8fc87
	s_wait_loadcnt 0x0                                         // 000000003168: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s10                       // 00000000316c: d65d0079 002af280
	v_cmp_gt_u64_e64 s10, s[24:25], v[122:123]                 // 000000003174: d45c000a 0202f418
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 00000000317c: bf870152
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003180: d7620079 0202f2ff 000000ff
	s_and_b32 s11, s2, s10                                     // 00000000318c: 8b0b0a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000003190: bf88ff9e
	v_cndmask_b32_e64 v122, 0, v124, s11                       // 000000003194: d501007a 002ef880
	v_cndmask_b32_e64 v123, 0, v127, s11                       // 00000000319c: d501007b 002efe80
	v_add_co_u32 v122, s12, s30, v122                          // 0000000031a4: d7000c7a 0202f41e
	s_wait_alu depctr_va_sdst(0)                               // 0000000031ac: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000031b0: bf870002
	v_add_co_ci_u32_e64 v123, null, s31, v123, s12             // 0000000031b4: d5207c7b 0032f61f
	global_load_d16_hi_u8 v121, v[122:123], off                // 0000000031bc: ee08407c 00000079 0000007a
	v_or_b16 v122.l, v46.l, v46.h op_sel:[0,1,0]               // 0000000031c8: d763107a 02025d2e
	v_or_b16 v122.h, v47.l, v47.h op_sel:[0,1,1]               // 0000000031d0: d763507a 02025f2f
	v_or_b16 v123.l, v120.l, v120.h op_sel:[0,1,0]             // 0000000031d8: d763107b 0202f178
	s_wait_loadcnt 0x0                                         // 0000000031e0: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 0000000031e4: d65d5079 002ef280
	v_add_co_u32 v126, s11, s13, v70                           // 0000000031ec: d7000b7e 02028c0d
	s_wait_alu depctr_va_sdst(0)                               // 0000000031f4: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s17, v68, s11              // 0000000031f8: d5207c7f 002e8811
	s_and_b32 s11, s3, vcc_lo                                  // 000000003200: 8b0b6a03
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000003204: d7385079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 00000000320c: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v126, s11                        // 000000003210: d501002e 002efc80
	v_cndmask_b32_e64 v47, 0, v127, s11                        // 000000003218: d501002f 002efe80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003220: bf870193
	v_or_b16 v123.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003224: d763507b 0202f379
	v_add_co_u32 v46, s12, s30, v46                            // 00000000322c: d7000c2e 02025c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003234: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003238: bf870003
	v_add_co_ci_u32_e64 v47, null, s31, v47, s12               // 00000000323c: d5207c2f 00325e1f
	global_load_d16_u8 v46, v[46:47], off                      // 000000003244: ee07807c 0000002e 0000002e
	v_or_b32_e32 v47, 1, v126                                  // 000000003250: 385efc81
	s_wait_loadcnt 0x0                                         // 000000003254: bfc00000
	v_cndmask_b16 v46.l, 0, v46.l, s11                         // 000000003258: d65d002e 002e5c80
	s_and_b32 s11, s3, s4                                      // 000000003260: 8b0b0403
	s_wait_alu depctr_sa_sdst(0)                               // 000000003264: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 000000003268: d501002f 002e5e80
	v_cndmask_b32_e64 v121, 0, v127, s11                       // 000000003270: d5010079 002efe80
	v_and_b16 v46.l, 0xff, v46.l                               // 000000003278: d762002e 02025cff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003284: bf8701a3
	v_add_co_u32 v120, s12, s30, v47                           // 000000003288: d7000c78 02025e1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003290: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000003294: d5207c79 0032f21f
	v_or_b32_e32 v47, 2, v126                                  // 00000000329c: 385efc82
	global_load_d16_hi_u8 v46, v[120:121], off                 // 0000000032a0: ee08407c 0000002e 00000078
	s_wait_loadcnt 0x0                                         // 0000000032ac: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, s11                         // 0000000032b0: d65d502e 002e5c80
	s_and_b32 s11, s3, s5                                      // 0000000032b8: 8b0b0503
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032bc: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 0000000032c0: d501002f 002e5e80
	v_cndmask_b32_e64 v121, 0, v127, s11                       // 0000000032c8: d5010079 002efe80
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 0000000032d0: d738502e 02025c88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000032d8: bf8701a3
	v_add_co_u32 v120, s12, s30, v47                           // 0000000032dc: d7000c78 02025e1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e4: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 0000000032e8: d5207c79 0032f21f
	global_load_d16_u8 v47, v[120:121], off                    // 0000000032f0: ee07807c 0000002f 00000078
	v_or_b32_e32 v120, 3, v126                                 // 0000000032fc: 38f0fc83
	s_wait_loadcnt 0x0                                         // 000000003300: bfc00000
	v_cndmask_b16 v47.l, 0, v47.l, s11                         // 000000003304: d65d002f 002e5e80
	s_and_b32 s11, s3, s6                                      // 00000000330c: 8b0b0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000003310: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 000000003314: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v127, s11                       // 00000000331c: d5010079 002efe80
	v_and_b16 v47.l, 0xff, v47.l                               // 000000003324: d762002f 02025eff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003330: bf8701a3
	v_add_co_u32 v120, s12, s30, v120                          // 000000003334: d7000c78 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 00000000333c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000003340: d5207c79 0032f21f
	global_load_d16_hi_u8 v47, v[120:121], off                 // 000000003348: ee08407c 0000002f 00000078
	v_or_b32_e32 v120, 4, v126                                 // 000000003354: 38f0fc84
	s_wait_loadcnt 0x0                                         // 000000003358: bfc00000
	v_cndmask_b16 v47.h, 0, v47.h, s11                         // 00000000335c: d65d502f 002e5e80
	s_and_b32 s11, s3, s7                                      // 000000003364: 8b0b0703
	s_wait_alu depctr_sa_sdst(0)                               // 000000003368: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 00000000336c: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v127, s11                       // 000000003374: d5010079 002efe80
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 00000000337c: d738502f 02025e88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003384: bf8701a3
	v_add_co_u32 v120, s12, s30, v120                          // 000000003388: d7000c78 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000003390: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000003394: d5207c79 0032f21f
	global_load_d16_u8 v120, v[120:121], off                   // 00000000339c: ee07807c 00000078 00000078
	v_or_b32_e32 v121, 5, v126                                 // 0000000033a8: 38f2fc85
	s_wait_loadcnt 0x0                                         // 0000000033ac: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s11                       // 0000000033b0: d65d0078 002ef080
	s_and_b32 s11, s3, s8                                      // 0000000033b8: 8b0b0803
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033bc: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000033c0: d5010079 002ef280
	v_cndmask_b32_e64 v125, 0, v127, s11                       // 0000000033c8: d501007d 002efe80
	v_and_b16 v120.l, 0xff, v120.l                             // 0000000033d0: d7620078 0202f0ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000033dc: bf8701a3
	v_add_co_u32 v124, s12, s30, v121                          // 0000000033e0: d7000c7c 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 0000000033e8: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s12             // 0000000033ec: d5207c7d 0032fa1f
	v_or_b32_e32 v121, 6, v126                                 // 0000000033f4: 38f2fc86
	global_load_d16_hi_u8 v120, v[124:125], off                // 0000000033f8: ee08407c 00000078 0000007c
	s_wait_loadcnt 0x0                                         // 000000003404: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s11                       // 000000003408: d65d5078 002ef080
	s_and_b32 s11, s3, s9                                      // 000000003410: 8b0b0903
	s_wait_alu depctr_sa_sdst(0)                               // 000000003414: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000003418: d5010079 002ef280
	v_cndmask_b32_e64 v125, 0, v127, s11                       // 000000003420: d501007d 002efe80
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000003428: d7385078 0202f088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003430: bf8701a3
	v_add_co_u32 v124, s12, s30, v121                          // 000000003434: d7000c7c 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 00000000343c: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s12             // 000000003440: d5207c7d 0032fa1f
	global_load_d16_u8 v121, v[124:125], off                   // 000000003448: ee07807c 00000079 0000007c
	v_or_b32_e32 v124, 7, v126                                 // 000000003454: 38f8fc87
	s_wait_loadcnt 0x0                                         // 000000003458: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s11                       // 00000000345c: d65d0079 002ef280
	s_and_b32 s11, s3, s10                                     // 000000003464: 8b0b0a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003468: bf88ff9e
	v_cndmask_b32_e64 v124, 0, v124, s11                       // 00000000346c: d501007c 002ef880
	v_cndmask_b32_e64 v125, 0, v127, s11                       // 000000003474: d501007d 002efe80
	v_and_b16 v121.l, 0xff, v121.l                             // 00000000347c: d7620079 0202f2ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003488: bf8701a3
	v_add_co_u32 v124, s12, s30, v124                          // 00000000348c: d7000c7c 0202f81e
	s_wait_alu depctr_va_sdst(0)                               // 000000003494: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s12             // 000000003498: d5207c7d 0032fa1f
	global_load_d16_hi_u8 v121, v[124:125], off                // 0000000034a0: ee08407c 00000079 0000007c
	v_or_b16 v124.l, v46.l, v46.h op_sel:[0,1,0]               // 0000000034ac: d763107c 02025d2e
	v_or_b16 v124.h, v47.l, v47.h op_sel:[0,1,1]               // 0000000034b4: d763507c 02025f2f
	v_or_b16 v125.l, v120.l, v120.h op_sel:[0,1,0]             // 0000000034bc: d763107d 0202f178
	s_wait_loadcnt 0x0                                         // 0000000034c4: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 0000000034c8: d65d5079 002ef280
	v_add_co_u32 v128, s11, s13, v74                           // 0000000034d0: d7000b80 0202940d
	s_wait_alu depctr_va_sdst(0)                               // 0000000034d8: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s17, v73, s11              // 0000000034dc: d5207c81 002e9211
	s_and_b32 s11, s1, vcc_lo                                  // 0000000034e4: 8b0b6a01
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 0000000034e8: d7385079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034f0: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v128, s11                        // 0000000034f4: d501002e 002f0080
	v_cndmask_b32_e64 v47, 0, v129, s11                        // 0000000034fc: d501002f 002f0280
	s_and_b32 vcc_lo, s0, vcc_lo                               // 000000003504: 8b6a6a00
	v_or_b16 v125.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003508: d763507d 0202f379
	s_delay_alu instid0(valu_dep_3)                            // 000000003510: bf870003
	v_add_co_u32 v46, s12, s34, v46                            // 000000003514: d7000c2e 02025c22
	s_wait_alu depctr_va_sdst(0)                               // 00000000351c: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s35, v47, s12               // 000000003520: d5207c2f 00325e23
	global_load_d16_u8 v46, v[46:47], off                      // 000000003528: ee07807c 0000002e 0000002e
	v_or_b32_e32 v47, 1, v128                                  // 000000003534: 385f0081
	s_wait_loadcnt 0x0                                         // 000000003538: bfc00000
	v_cndmask_b16 v46.l, 0, v46.l, s11                         // 00000000353c: d65d002e 002e5c80
	s_and_b32 s11, s1, s4                                      // 000000003544: 8b0b0401
	s_wait_alu depctr_sa_sdst(0)                               // 000000003548: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 00000000354c: d501002f 002e5e80
	v_cndmask_b32_e64 v121, 0, v129, s11                       // 000000003554: d5010079 002f0280
	v_and_b16 v46.l, 0xff, v46.l                               // 00000000355c: d762002e 02025cff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003568: bf8701a3
	v_add_co_u32 v120, s12, s34, v47                           // 00000000356c: d7000c78 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 000000003574: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 000000003578: d5207c79 0032f223
	v_or_b32_e32 v47, 2, v128                                  // 000000003580: 385f0082
	global_load_d16_hi_u8 v46, v[120:121], off                 // 000000003584: ee08407c 0000002e 00000078
	s_wait_loadcnt 0x0                                         // 000000003590: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, s11                         // 000000003594: d65d502e 002e5c80
	s_and_b32 s11, s1, s5                                      // 00000000359c: 8b0b0501
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035a0: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 0000000035a4: d501002f 002e5e80
	v_cndmask_b32_e64 v121, 0, v129, s11                       // 0000000035ac: d5010079 002f0280
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 0000000035b4: d738502e 02025c88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000035bc: bf8701a3
	v_add_co_u32 v120, s12, s34, v47                           // 0000000035c0: d7000c78 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c8: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 0000000035cc: d5207c79 0032f223
	global_load_d16_u8 v47, v[120:121], off                    // 0000000035d4: ee07807c 0000002f 00000078
	v_or_b32_e32 v120, 3, v128                                 // 0000000035e0: 38f10083
	s_wait_loadcnt 0x0                                         // 0000000035e4: bfc00000
	v_cndmask_b16 v47.l, 0, v47.l, s11                         // 0000000035e8: d65d002f 002e5e80
	s_and_b32 s11, s1, s6                                      // 0000000035f0: 8b0b0601
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035f4: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 0000000035f8: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v129, s11                       // 000000003600: d5010079 002f0280
	v_and_b16 v47.l, 0xff, v47.l                               // 000000003608: d762002f 02025eff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003614: bf8701a3
	v_add_co_u32 v120, s12, s34, v120                          // 000000003618: d7000c78 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 000000003620: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 000000003624: d5207c79 0032f223
	global_load_d16_hi_u8 v47, v[120:121], off                 // 00000000362c: ee08407c 0000002f 00000078
	v_or_b32_e32 v120, 4, v128                                 // 000000003638: 38f10084
	s_wait_loadcnt 0x0                                         // 00000000363c: bfc00000
	v_cndmask_b16 v47.h, 0, v47.h, s11                         // 000000003640: d65d502f 002e5e80
	s_and_b32 s11, s1, s7                                      // 000000003648: 8b0b0701
	s_wait_alu depctr_sa_sdst(0)                               // 00000000364c: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v120, s11                       // 000000003650: d5010078 002ef080
	v_cndmask_b32_e64 v121, 0, v129, s11                       // 000000003658: d5010079 002f0280
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000003660: d738502f 02025e88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003668: bf8701a3
	v_add_co_u32 v120, s12, s34, v120                          // 00000000366c: d7000c78 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 000000003674: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 000000003678: d5207c79 0032f223
	global_load_d16_u8 v120, v[120:121], off                   // 000000003680: ee07807c 00000078 00000078
	v_or_b32_e32 v121, 5, v128                                 // 00000000368c: 38f30085
	s_wait_loadcnt 0x0                                         // 000000003690: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s11                       // 000000003694: d65d0078 002ef080
	s_and_b32 s11, s1, s8                                      // 00000000369c: 8b0b0801
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036a0: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000036a4: d5010079 002ef280
	v_cndmask_b32_e64 v127, 0, v129, s11                       // 0000000036ac: d501007f 002f0280
	v_and_b16 v120.l, 0xff, v120.l                             // 0000000036b4: d7620078 0202f0ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000036c0: bf8701a3
	v_add_co_u32 v126, s12, s34, v121                          // 0000000036c4: d7000c7e 0202f222
	s_wait_alu depctr_va_sdst(0)                               // 0000000036cc: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 0000000036d0: d5207c7f 0032fe23
	v_or_b32_e32 v121, 6, v128                                 // 0000000036d8: 38f30086
	global_load_d16_hi_u8 v120, v[126:127], off                // 0000000036dc: ee08407c 00000078 0000007e
	s_wait_loadcnt 0x0                                         // 0000000036e8: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s11                       // 0000000036ec: d65d5078 002ef080
	s_and_b32 s11, s1, s9                                      // 0000000036f4: 8b0b0901
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036f8: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000036fc: d5010079 002ef280
	v_cndmask_b32_e64 v127, 0, v129, s11                       // 000000003704: d501007f 002f0280
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 00000000370c: d7385078 0202f088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003714: bf8701a3
	v_add_co_u32 v126, s12, s34, v121                          // 000000003718: d7000c7e 0202f222
	s_wait_alu depctr_va_sdst(0)                               // 000000003720: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 000000003724: d5207c7f 0032fe23
	global_load_d16_u8 v121, v[126:127], off                   // 00000000372c: ee07807c 00000079 0000007e
	v_or_b32_e32 v126, 7, v128                                 // 000000003738: 38fd0087
	s_wait_loadcnt 0x0                                         // 00000000373c: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s11                       // 000000003740: d65d0079 002ef280
	s_and_b32 s11, s1, s10                                     // 000000003748: 8b0b0a01
	s_wait_alu depctr_sa_sdst(0)                               // 00000000374c: bf88ff9e
	v_cndmask_b32_e64 v126, 0, v126, s11                       // 000000003750: d501007e 002efc80
	v_cndmask_b32_e64 v127, 0, v129, s11                       // 000000003758: d501007f 002f0280
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003760: d7620079 0202f2ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000376c: bf8701a3
	v_add_co_u32 v126, s12, s34, v126                          // 000000003770: d7000c7e 0202fc22
	s_wait_alu depctr_va_sdst(0)                               // 000000003778: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 00000000377c: d5207c7f 0032fe23
	global_load_d16_hi_u8 v121, v[126:127], off                // 000000003784: ee08407c 00000079 0000007e
	v_or_b16 v126.l, v46.l, v46.h op_sel:[0,1,0]               // 000000003790: d763107e 02025d2e
	v_or_b16 v126.h, v47.l, v47.h op_sel:[0,1,1]               // 000000003798: d763507e 02025f2f
	v_or_b16 v127.l, v120.l, v120.h op_sel:[0,1,0]             // 0000000037a0: d763107f 0202f178
	s_wait_loadcnt 0x0                                         // 0000000037a8: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 0000000037ac: d65d5079 002ef280
	v_add_co_u32 v130, s11, s13, v77                           // 0000000037b4: d7000b82 02029a0d
	s_wait_alu depctr_va_sdst(0)                               // 0000000037bc: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s17, v75, s11              // 0000000037c0: d5207c83 002e9611
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000037c8: bf870113
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 0000000037cc: d7385079 0202f288
	v_dual_cndmask_b32 v46, 0, v130 :: v_dual_cndmask_b32 v47, 0, v131// 0000000037d4: ca530480 2e2f0680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000037dc: bf870112
	v_or_b16 v127.h, v121.l, v121.h op_sel:[0,1,1]             // 0000000037e0: d763507f 0202f379
	v_add_co_u32 v46, s11, s34, v46                            // 0000000037e8: d7000b2e 02025c22
	s_wait_alu depctr_va_sdst(0)                               // 0000000037f0: bf88f19f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037f4: bf870193
	v_add_co_ci_u32_e64 v47, null, s35, v47, s11               // 0000000037f8: d5207c2f 002e5e23
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[122:123], v[126:127], v[24:31]// 000000003800: cc464018 1c62fd7a
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[124:125], v[126:127], v[8:15]// 000000003808: cc464008 1c22fd7c
	global_load_d16_u8 v46, v[46:47], off                      // 000000003810: ee07807c 0000002e 0000002e
	v_or_b32_e32 v47, 1, v130                                  // 00000000381c: 385f0481
	s_wait_loadcnt 0x0                                         // 000000003820: bfc00000
	v_cndmask_b16 v46.l, 0, v46.l, vcc_lo                      // 000000003824: d65d002e 01aa5c80
	s_and_b32 vcc_lo, s0, s4                                   // 00000000382c: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 000000003830: bf88ff9e
	v_cndmask_b32_e32 v47, 0, v47, vcc_lo                      // 000000003834: 025e5e80
	v_cndmask_b32_e32 v121, 0, v131, vcc_lo                    // 000000003838: 02f30680
	v_and_b16 v46.l, 0xff, v46.l                               // 00000000383c: d762002e 02025cff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003848: bf8701a3
	v_add_co_u32 v120, s4, s34, v47                            // 00000000384c: d7000478 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 000000003854: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s4              // 000000003858: d5207c79 0012f223
	v_or_b32_e32 v47, 2, v130                                  // 000000003860: 385f0482
	global_load_d16_hi_u8 v46, v[120:121], off                 // 000000003864: ee08407c 0000002e 00000078
	s_wait_loadcnt 0x0                                         // 000000003870: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, vcc_lo                      // 000000003874: d65d502e 01aa5c80
	s_and_b32 vcc_lo, s0, s5                                   // 00000000387c: 8b6a0500
	s_wait_alu depctr_sa_sdst(0)                               // 000000003880: bf88ff9e
	v_cndmask_b32_e32 v47, 0, v47, vcc_lo                      // 000000003884: 025e5e80
	v_cndmask_b32_e32 v121, 0, v131, vcc_lo                    // 000000003888: 02f30680
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 00000000388c: d738502e 02025c88
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003894: bf8701a3
	v_add_co_u32 v120, s4, s34, v47                            // 000000003898: d7000478 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a0: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s4              // 0000000038a4: d5207c79 0012f223
	global_load_d16_u8 v47, v[120:121], off                    // 0000000038ac: ee07807c 0000002f 00000078
	v_or_b32_e32 v120, 3, v130                                 // 0000000038b8: 38f10483
	s_wait_loadcnt 0x0                                         // 0000000038bc: bfc00000
	v_cndmask_b16 v47.l, 0, v47.l, vcc_lo                      // 0000000038c0: d65d002f 01aa5e80
	s_and_b32 vcc_lo, s0, s6                                   // 0000000038c8: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038cc: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v120 :: v_dual_cndmask_b32 v121, 0, v131// 0000000038d0: ca52f080 78790680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 0000000038d8: bf870112
	v_and_b16 v47.l, 0xff, v47.l                               // 0000000038dc: d762002f 02025eff 000000ff
	v_add_co_u32 v120, s4, s34, v120                           // 0000000038e8: d7000478 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 0000000038f0: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000038f4: bf870003
	v_add_co_ci_u32_e64 v121, null, s35, v121, s4              // 0000000038f8: d5207c79 0012f223
	global_load_d16_hi_u8 v47, v[120:121], off                 // 000000003900: ee08407c 0000002f 00000078
	v_or_b32_e32 v120, 4, v130                                 // 00000000390c: 38f10484
	s_wait_loadcnt 0x0                                         // 000000003910: bfc00000
	v_cndmask_b16 v47.h, 0, v47.h, vcc_lo                      // 000000003914: d65d502f 01aa5e80
	s_and_b32 vcc_lo, s0, s7                                   // 00000000391c: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003920: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v120 :: v_dual_cndmask_b32 v121, 0, v131// 000000003924: ca52f080 78790680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000392c: bf870112
	v_lshlrev_b16 v47.h, 8, v47.h op_sel:[0,1,1]               // 000000003930: d738502f 02025e88
	v_add_co_u32 v120, s4, s34, v120                           // 000000003938: d7000478 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003944: bf870003
	v_add_co_ci_u32_e64 v121, null, s35, v121, s4              // 000000003948: d5207c79 0012f223
	global_load_d16_u8 v120, v[120:121], off                   // 000000003950: ee07807c 00000078 00000078
	v_or_b32_e32 v121, 5, v130                                 // 00000000395c: 38f30485
	s_wait_loadcnt 0x0                                         // 000000003960: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, vcc_lo                    // 000000003964: d65d0078 01aaf080
	s_and_b32 vcc_lo, s0, s8                                   // 00000000396c: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 000000003970: bf88ff9e
	v_cndmask_b32_e32 v121, 0, v121, vcc_lo                    // 000000003974: 02f2f280
	v_cndmask_b32_e32 v129, 0, v131, vcc_lo                    // 000000003978: 03030680
	v_and_b16 v120.l, 0xff, v120.l                             // 00000000397c: d7620078 0202f0ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003988: bf8701a3
	v_add_co_u32 v128, s4, s34, v121                           // 00000000398c: d7000480 0202f222
	s_wait_alu depctr_va_sdst(0)                               // 000000003994: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s35, v129, s4              // 000000003998: d5207c81 00130223
	v_or_b32_e32 v121, 6, v130                                 // 0000000039a0: 38f30486
	global_load_d16_hi_u8 v120, v[128:129], off                // 0000000039a4: ee08407c 00000078 00000080
	s_wait_loadcnt 0x0                                         // 0000000039b0: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, vcc_lo                    // 0000000039b4: d65d5078 01aaf080
	s_and_b32 vcc_lo, s0, s9                                   // 0000000039bc: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039c0: bf88ff9e
	v_cndmask_b32_e32 v121, 0, v121, vcc_lo                    // 0000000039c4: 02f2f280
	v_cndmask_b32_e32 v129, 0, v131, vcc_lo                    // 0000000039c8: 03030680
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 0000000039cc: d7385078 0202f088
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000039d4: bf8701a3
	v_add_co_u32 v128, s4, s34, v121                           // 0000000039d8: d7000480 0202f222
	s_wait_alu depctr_va_sdst(0)                               // 0000000039e0: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s35, v129, s4              // 0000000039e4: d5207c81 00130223
	global_load_d16_u8 v121, v[128:129], off                   // 0000000039ec: ee07807c 00000079 00000080
	v_or_b32_e32 v128, 7, v130                                 // 0000000039f8: 39010487
	s_wait_loadcnt 0x0                                         // 0000000039fc: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, vcc_lo                    // 000000003a00: d65d0079 01aaf280
	s_and_b32 vcc_lo, s0, s10                                  // 000000003a08: 8b6a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a0c: bf88ff9e
	v_dual_cndmask_b32 v128, 0, v128 :: v_dual_cndmask_b32 v129, 0, v131// 000000003a10: ca530080 80810680
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003a18: bf870112
	v_and_b16 v121.l, 0xff, v121.l                             // 000000003a1c: d7620079 0202f2ff 000000ff
	v_add_co_u32 v128, s4, s34, v128                           // 000000003a28: d7000480 02030022
	s_wait_alu depctr_va_sdst(0)                               // 000000003a30: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003a34: bf870003
	v_add_co_ci_u32_e64 v129, null, s35, v129, s4              // 000000003a38: d5207c81 00130223
	s_lshr_b64 s[4:5], s[16:17], 5                             // 000000003a40: 85848510
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a44: bf88ff9e
	s_mul_u64 s[6:7], s[4:5], s[22:23]                         // 000000003a48: aa861604
	global_load_d16_hi_u8 v121, v[128:129], off                // 000000003a4c: ee08407c 00000079 00000080
	s_lshr_b64 s[4:5], s[16:17], 3                             // 000000003a58: 85848310
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a5c: bf88ff9e
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000003a60: 84868206
	s_add_nc_u64 s[16:17], s[16:17], 32                        // 000000003a64: a990a010
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a68: bf88ff9e
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]                      // 000000003a6c: a9860624
	s_wait_loadcnt 0x0                                         // 000000003a70: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, vcc_lo                    // 000000003a74: d65d5079 01aaf280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003a7c: bf870091
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000003a80: d7385079 0202f288
	v_or_b16 v121.h, v121.l, v121.h op_sel:[0,1,1]             // 000000003a88: d7635079 0202f379
	v_or_b16 v121.l, v120.l, v120.h op_sel:[0,1,0]             // 000000003a90: d7631079 0202f178
	v_or_b16 v120.l, v46.l, v46.h op_sel:[0,1,0]               // 000000003a98: d7631078 02025d2e
	v_add_co_u32 v46, vcc_lo, v80, s4                          // 000000003aa0: d7006a2e 02000950
	v_or_b16 v120.h, v47.l, v47.h op_sel:[0,1,1]               // 000000003aa8: d7635078 02025f2f
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab0: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s5, v81, vcc_lo             // 000000003ab4: d5207c2f 01aaa205
	s_delay_alu instid0(valu_dep_2)                            // 000000003abc: bf870002
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[122:123], v[120:121], v[16:23]// 000000003ac0: cc464010 1c42f17a
	global_load_b32 v122, v[46:47], off                        // 000000003ac8: ee05007c 0000007a 0000002e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ad4: bf88ff9e
	v_add_co_u32 v46, vcc_lo, s6, v42                          // 000000003ad8: d7006a2e 02025406
	s_wait_alu depctr_va_vcc(0)                                // 000000003ae0: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s7, v43, vcc_lo             // 000000003ae4: d5207c2f 01aa5607
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[124:125], v[120:121], v[0:7]// 000000003aec: cc464000 1c02f17c
	v_add_co_u32 v120, vcc_lo, v83, s4                         // 000000003af4: d7006a78 02000953
	global_load_b32 v46, v[46:47], off                         // 000000003afc: ee05007c 0000002e 0000002e
	s_wait_alu depctr_va_vcc(0)                                // 000000003b08: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s5, v84, vcc_lo            // 000000003b0c: d5207c79 01aaa805
	s_wait_loadcnt 0x0                                         // 000000003b14: bfc00000
	v_mul_f32_e32 v47, v122, v46                               // 000000003b18: 105e5d7a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_1)// 000000003b1c: bf8700d1
	v_mul_f32_e32 v24, v24, v47                                // 000000003b20: 10305f18
	global_load_b32 v47, v[120:121], off                       // 000000003b24: ee05007c 0000002f 00000078
	v_add_f32_e32 v62, v62, v24                                // 000000003b30: 067c313e
	s_wait_loadcnt 0x0                                         // 000000003b34: bfc00000
	v_mul_f32_e32 v24, v46, v47                                // 000000003b38: 10305f2e
	v_mul_f32_e32 v24, v25, v24                                // 000000003b3c: 10303119
	s_delay_alu instid0(valu_dep_1)                            // 000000003b40: bf870001
	v_add_f32_e32 v107, v107, v24                              // 000000003b44: 06d6316b
	v_add_co_u32 v24, vcc_lo, v86, s4                          // 000000003b48: d7006a18 02000956
	s_wait_alu depctr_va_vcc(0)                                // 000000003b50: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v87, vcc_lo             // 000000003b54: d5207c19 01aaae05
	global_load_b32 v120, v[24:25], off                        // 000000003b5c: ee05007c 00000078 00000018
	s_wait_loadcnt 0x0                                         // 000000003b68: bfc00000
	v_mul_f32_e32 v24, v46, v120                               // 000000003b6c: 1030f12e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003b70: bf870091
	v_mul_f32_e32 v24, v26, v24                                // 000000003b74: 1030311a
	v_add_f32_e32 v104, v104, v24                              // 000000003b78: 06d03168
	v_add_co_u32 v24, vcc_lo, v89, s4                          // 000000003b7c: d7006a18 02000959
	s_wait_alu depctr_va_vcc(0)                                // 000000003b84: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v90, vcc_lo             // 000000003b88: d5207c19 01aab405
	global_load_b32 v26, v[24:25], off                         // 000000003b90: ee05007c 0000001a 00000018
	s_wait_loadcnt 0x0                                         // 000000003b9c: bfc00000
	v_mul_f32_e32 v24, v46, v26                                // 000000003ba0: 1030352e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ba4: bf870091
	v_mul_f32_e32 v24, v27, v24                                // 000000003ba8: 1030311b
	v_add_f32_e32 v101, v101, v24                              // 000000003bac: 06ca3165
	v_add_co_u32 v24, vcc_lo, v91, s4                          // 000000003bb0: d7006a18 0200095b
	s_wait_alu depctr_va_vcc(0)                                // 000000003bb8: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v92, vcc_lo             // 000000003bbc: d5207c19 01aab805
	global_load_b32 v27, v[24:25], off                         // 000000003bc4: ee05007c 0000001b 00000018
	s_wait_loadcnt 0x0                                         // 000000003bd0: bfc00000
	v_mul_f32_e32 v24, v46, v27                                // 000000003bd4: 1030372e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003bd8: bf870091
	v_mul_f32_e32 v24, v28, v24                                // 000000003bdc: 1030311c
	v_add_f32_e32 v98, v98, v24                                // 000000003be0: 06c43162
	v_add_co_u32 v24, vcc_lo, v94, s4                          // 000000003be4: d7006a18 0200095e
	s_wait_alu depctr_va_vcc(0)                                // 000000003bec: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v95, vcc_lo             // 000000003bf0: d5207c19 01aabe05
	global_load_b32 v28, v[24:25], off                         // 000000003bf8: ee05007c 0000001c 00000018
	s_wait_loadcnt 0x0                                         // 000000003c04: bfc00000
	v_mul_f32_e32 v24, v46, v28                                // 000000003c08: 1030392e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c0c: bf870091
	v_mul_f32_e32 v24, v29, v24                                // 000000003c10: 1030311d
	v_add_f32_e32 v93, v93, v24                                // 000000003c14: 06ba315d
	v_add_co_u32 v24, vcc_lo, v96, s4                          // 000000003c18: d7006a18 02000960
	s_wait_alu depctr_va_vcc(0)                                // 000000003c20: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v97, vcc_lo             // 000000003c24: d5207c19 01aac205
	global_load_b32 v29, v[24:25], off                         // 000000003c2c: ee05007c 0000001d 00000018
	s_wait_loadcnt 0x0                                         // 000000003c38: bfc00000
	v_mul_f32_e32 v24, v46, v29                                // 000000003c3c: 10303b2e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c40: bf870091
	v_mul_f32_e32 v24, v30, v24                                // 000000003c44: 1030311e
	v_add_f32_e32 v88, v88, v24                                // 000000003c48: 06b03158
	v_add_co_u32 v24, vcc_lo, v99, s4                          // 000000003c4c: d7006a18 02000963
	s_wait_alu depctr_va_vcc(0)                                // 000000003c54: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v100, vcc_lo            // 000000003c58: d5207c19 01aac805
	global_load_b32 v30, v[24:25], off                         // 000000003c60: ee05007c 0000001e 00000018
	s_wait_loadcnt 0x0                                         // 000000003c6c: bfc00000
	v_mul_f32_e32 v24, v46, v30                                // 000000003c70: 10303d2e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c74: bf870091
	v_mul_f32_e32 v24, v31, v24                                // 000000003c78: 1030311f
	v_add_f32_e32 v85, v85, v24                                // 000000003c7c: 06aa3155
	v_add_co_u32 v24, vcc_lo, s6, v44                          // 000000003c80: d7006a18 02025806
	s_wait_alu depctr_va_vcc(0)                                // 000000003c88: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s7, v45, vcc_lo             // 000000003c8c: d5207c19 01aa5a07
	global_load_b32 v24, v[24:25], off                         // 000000003c94: ee05007c 00000018 00000018
	s_wait_loadcnt 0x0                                         // 000000003ca0: bfc00000
	v_mul_f32_e32 v25, v122, v24                               // 000000003ca4: 1032317a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ca8: bf870091
	v_mul_f32_e32 v16, v16, v25                                // 000000003cac: 10203310
	v_add_f32_e32 v64, v64, v16                                // 000000003cb0: 06802140
	v_mul_f32_e32 v16, v47, v24                                // 000000003cb4: 1020312f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cb8: bf870091
	v_mul_f32_e32 v16, v17, v16                                // 000000003cbc: 10202111
	v_add_f32_e32 v63, v63, v16                                // 000000003cc0: 067e213f
	v_mul_f32_e32 v16, v120, v24                               // 000000003cc4: 10203178
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cc8: bf870091
	v_mul_f32_e32 v16, v18, v16                                // 000000003ccc: 10202112
	v_add_f32_e32 v61, v61, v16                                // 000000003cd0: 067a213d
	v_mul_f32_e32 v16, v26, v24                                // 000000003cd4: 1020311a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cd8: bf870091
	v_mul_f32_e32 v16, v19, v16                                // 000000003cdc: 10202113
	v_add_f32_e32 v60, v60, v16                                // 000000003ce0: 0678213c
	v_mul_f32_e32 v16, v27, v24                                // 000000003ce4: 1020311b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ce8: bf870091
	v_mul_f32_e32 v16, v20, v16                                // 000000003cec: 10202114
	v_add_f32_e32 v59, v59, v16                                // 000000003cf0: 0676213b
	v_mul_f32_e32 v16, v28, v24                                // 000000003cf4: 1020311c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cf8: bf870091
	v_mul_f32_e32 v16, v21, v16                                // 000000003cfc: 10202115
	v_add_f32_e32 v58, v58, v16                                // 000000003d00: 0674213a
	v_mul_f32_e32 v16, v29, v24                                // 000000003d04: 1020311d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d08: bf870091
	v_mul_f32_e32 v16, v22, v16                                // 000000003d0c: 10202116
	v_add_f32_e32 v57, v57, v16                                // 000000003d10: 06722139
	v_mul_f32_e32 v16, v30, v24                                // 000000003d14: 1020311e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003d18: bf870091
	v_mul_f32_e32 v16, v23, v16                                // 000000003d1c: 10202117
	v_add_f32_e32 v56, v56, v16                                // 000000003d20: 06702138
	v_add_co_u32 v16, vcc_lo, v102, s4                         // 000000003d24: d7006a10 02000966
	s_wait_alu depctr_va_vcc(0)                                // 000000003d2c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s5, v103, vcc_lo            // 000000003d30: d5207c11 01aace05
	global_load_b32 v16, v[16:17], off                         // 000000003d38: ee05007c 00000010 00000010
	s_wait_loadcnt 0x0                                         // 000000003d44: bfc00000
	v_mul_f32_e32 v17, v46, v16                                // 000000003d48: 1022212e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003d4c: bf8701c1
	v_mul_f32_e32 v8, v8, v17                                  // 000000003d50: 10102308
	v_add_co_u32 v17, vcc_lo, v105, s4                         // 000000003d54: d7006a11 02000969
	s_wait_alu depctr_va_vcc(0)                                // 000000003d5c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v106, vcc_lo            // 000000003d60: d5207c12 01aad405
	v_add_f32_e32 v82, v82, v8                                 // 000000003d68: 06a41152
	global_load_b32 v8, v[17:18], off                          // 000000003d6c: ee05007c 00000008 00000011
	s_wait_loadcnt 0x0                                         // 000000003d78: bfc00000
	v_mul_f32_e32 v17, v46, v8                                 // 000000003d7c: 1022112e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003d80: bf8701c1
	v_mul_f32_e32 v9, v9, v17                                  // 000000003d84: 10122309
	v_add_co_u32 v17, vcc_lo, v108, s4                         // 000000003d88: d7006a11 0200096c
	s_wait_alu depctr_va_vcc(0)                                // 000000003d90: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v109, vcc_lo            // 000000003d94: d5207c12 01aada05
	v_add_f32_e32 v79, v79, v9                                 // 000000003d9c: 069e134f
	global_load_b32 v9, v[17:18], off                          // 000000003da0: ee05007c 00000009 00000011
	s_wait_loadcnt 0x0                                         // 000000003dac: bfc00000
	v_mul_f32_e32 v17, v46, v9                                 // 000000003db0: 1022132e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003db4: bf8701c1
	v_mul_f32_e32 v10, v10, v17                                // 000000003db8: 1014230a
	v_add_co_u32 v17, vcc_lo, v110, s4                         // 000000003dbc: d7006a11 0200096e
	s_wait_alu depctr_va_vcc(0)                                // 000000003dc4: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v111, vcc_lo            // 000000003dc8: d5207c12 01aade05
	v_add_f32_e32 v78, v78, v10                                // 000000003dd0: 069c154e
	global_load_b32 v10, v[17:18], off                         // 000000003dd4: ee05007c 0000000a 00000011
	s_wait_loadcnt 0x0                                         // 000000003de0: bfc00000
	v_mul_f32_e32 v17, v46, v10                                // 000000003de4: 1022152e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003de8: bf8701c1
	v_mul_f32_e32 v11, v11, v17                                // 000000003dec: 1016230b
	v_add_co_u32 v17, vcc_lo, v112, s4                         // 000000003df0: d7006a11 02000970
	s_wait_alu depctr_va_vcc(0)                                // 000000003df8: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v113, vcc_lo            // 000000003dfc: d5207c12 01aae205
	v_add_f32_e32 v76, v76, v11                                // 000000003e04: 0698174c
	global_load_b32 v11, v[17:18], off                         // 000000003e08: ee05007c 0000000b 00000011
	s_wait_loadcnt 0x0                                         // 000000003e14: bfc00000
	v_mul_f32_e32 v17, v46, v11                                // 000000003e18: 1022172e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003e1c: bf8701c1
	v_mul_f32_e32 v12, v12, v17                                // 000000003e20: 1018230c
	v_add_co_u32 v17, vcc_lo, v114, s4                         // 000000003e24: d7006a11 02000972
	s_wait_alu depctr_va_vcc(0)                                // 000000003e2c: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v115, vcc_lo            // 000000003e30: d5207c12 01aae605
	v_add_f32_e32 v72, v72, v12                                // 000000003e38: 06901948
	global_load_b32 v12, v[17:18], off                         // 000000003e3c: ee05007c 0000000c 00000011
	s_wait_loadcnt 0x0                                         // 000000003e48: bfc00000
	v_mul_f32_e32 v17, v46, v12                                // 000000003e4c: 1022192e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003e50: bf8701c1
	v_mul_f32_e32 v13, v13, v17                                // 000000003e54: 101a230d
	v_add_co_u32 v17, vcc_lo, v116, s4                         // 000000003e58: d7006a11 02000974
	s_wait_alu depctr_va_vcc(0)                                // 000000003e60: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v117, vcc_lo            // 000000003e64: d5207c12 01aaea05
	v_add_f32_e32 v71, v71, v13                                // 000000003e6c: 068e1b47
	global_load_b32 v13, v[17:18], off                         // 000000003e70: ee05007c 0000000d 00000011
	s_wait_loadcnt 0x0                                         // 000000003e7c: bfc00000
	v_mul_f32_e32 v17, v46, v13                                // 000000003e80: 10221b2e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003e84: bf8701c1
	v_mul_f32_e32 v14, v14, v17                                // 000000003e88: 101c230e
	v_add_co_u32 v17, vcc_lo, v118, s4                         // 000000003e8c: d7006a11 02000976
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s5, v119, vcc_lo            // 000000003e98: d5207c12 01aaee05
	v_add_f32_e32 v66, v66, v14                                // 000000003ea0: 06841d42
	v_cmp_lt_u64_e64 s4, s[16:17], s[24:25]                    // 000000003ea4: d4590004 02003010
	global_load_b32 v14, v[17:18], off                         // 000000003eac: ee05007c 0000000e 00000011
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000003eb8: 8b6a047e
	s_wait_loadcnt 0x0                                         // 000000003ebc: bfc00000
	v_mul_f32_e32 v17, v46, v14                                // 000000003ec0: 10221d2e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ec4: bf870091
	v_mul_f32_e32 v15, v15, v17                                // 000000003ec8: 101e230f
	v_add_f32_e32 v65, v65, v15                                // 000000003ecc: 06821f41
	v_mul_f32_e32 v15, v24, v16                                // 000000003ed0: 101e2118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ed4: bf870091
	v_mul_f32_e32 v0, v0, v15                                  // 000000003ed8: 10001f00
	v_add_f32_e32 v55, v55, v0                                 // 000000003edc: 066e0137
	v_mul_f32_e32 v0, v24, v8                                  // 000000003ee0: 10001118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ee4: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000003ee8: 10000101
	v_add_f32_e32 v54, v54, v0                                 // 000000003eec: 066c0136
	v_mul_f32_e32 v0, v24, v9                                  // 000000003ef0: 10001318
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ef4: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 000000003ef8: 10000102
	v_dual_add_f32 v53, v53, v0 :: v_dual_mul_f32 v0, v24, v10 // 000000003efc: c9060135 35001518
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f04: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 000000003f08: 10000103
	v_add_f32_e32 v52, v52, v0                                 // 000000003f0c: 06680134
	v_mul_f32_e32 v0, v24, v11                                 // 000000003f10: 10001718
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f14: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 000000003f18: 10000104
	v_add_f32_e32 v51, v51, v0                                 // 000000003f1c: 06660133
	v_mul_f32_e32 v0, v24, v12                                 // 000000003f20: 10001918
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f24: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 000000003f28: 10000105
	v_add_f32_e32 v50, v50, v0                                 // 000000003f2c: 06640132
	v_mul_f32_e32 v0, v24, v13                                 // 000000003f30: 10001b18
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f34: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 000000003f38: 10000106
	v_dual_add_f32 v49, v49, v0 :: v_dual_mul_f32 v0, v24, v14 // 000000003f3c: c9060131 31001d18
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f44: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000003f48: 10000107
	v_add_f32_e32 v35, v35, v0                                 // 000000003f4c: 06460123
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f50: bf88ff9e
	s_cbranch_vccnz 63684                                      // 000000003f54: bfa4f8c4 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x768>
	v_mul_lo_u32 v2, s23, v40                                  // 000000003f58: d72c0002 02025017
	v_mul_lo_u32 v3, s22, v41                                  // 000000003f60: d72c0003 02025216
	v_mad_co_u64_u32 v[0:1], null, s22, v40, 0                 // 000000003f68: d6fe7c00 02025016
	v_sub_co_u32 v14, vcc_lo, s20, v40                         // 000000003f70: d7016a0e 02025014
	s_wait_alu depctr_va_vcc(0)                                // 000000003f78: bf88ff9d
	v_sub_co_ci_u32_e64 v15, null, s21, v41, vcc_lo            // 000000003f7c: d5217c0f 01aa5215
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003f84: bf870211
	v_cmp_lt_i64_e32 vcc_lo, 0, v[14:15]                       // 000000003f88: 7ca21c80
	v_add3_u32 v1, v1, v3, v2                                  // 000000003f8c: d6550001 040a0701
	s_delay_alu instid0(valu_dep_1)                            // 000000003f94: bf870001
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000003f98: 3e0c0081
	s_and_b32 s2, vcc_lo, s1                                   // 000000003f9c: 8b02016a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fa0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003fa4: be832002
	s_cbranch_execz 28                                         // 000000003fa8: bfa5001c <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x251c>
	v_lshlrev_b64_e32 v[0:1], 1, v[32:33]                      // 000000003fac: 3e004081
	v_add_co_u32 v3, s2, s18, v6                               // 000000003fb0: d7000203 02020c12
	v_bfe_u32 v2, v62, 16, 1                                   // 000000003fb8: d6100002 0205213e
	s_wait_alu depctr_va_sdst(0)                               // 000000003fc0: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s19, v7, s2                  // 000000003fc4: d5207c04 000a0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003fcc: bf870193
	v_add_co_u32 v0, s2, v3, v0                                // 000000003fd0: d7000200 02020103
	v_add3_u32 v2, v2, v62, 0x7fff                             // 000000003fd8: d6550002 03fe7d02 00007fff
	v_or_b32_e32 v5, 0x400000, v62                             // 000000003fe4: 380a7cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000003fec: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v4, v1, s2                   // 000000003ff0: d5207c01 000a0304
	v_cmp_u_f32_e64 s2, v62, v62                               // 000000003ff8: d4180002 02027d3e
	s_wait_alu depctr_va_sdst(0)                               // 000000004000: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004004: bf870001
	v_cndmask_b32_e64 v2, v2, v5, s2                           // 000000004008: d5010002 000a0b02
	global_store_d16_hi_b16 v[0:1], v2, off                    // 000000004010: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000401c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004020: 8c7e037e
	v_add_co_u32 v0, s2, s22, v32                              // 000000004024: d7000200 02024016
	s_wait_alu depctr_va_sdst(0)                               // 00000000402c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s23, v33, s2                 // 000000004030: d5207c01 000a4217
	v_cmp_lt_i64_e64 s2, 1, v[14:15]                           // 000000004038: d4510002 02021c81
	s_delay_alu instid0(valu_dep_2)                            // 000000004040: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004044: 3e000081
	s_and_b32 s3, s2, s1                                       // 000000004048: 8b030102
	s_wait_alu depctr_sa_sdst(0)                               // 00000000404c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 000000004050: be842003
	s_cbranch_execz 27                                         // 000000004054: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x25c4>
	v_bfe_u32 v2, v107, 16, 1                                  // 000000004058: d6100002 0205216b
	v_add_co_u32 v3, s3, s18, v6                               // 000000004060: d7000303 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004068: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s19, v7, s3                  // 00000000406c: d5207c04 000e0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004074: bf870193
	v_add3_u32 v5, v2, v107, 0x7fff                            // 000000004078: d6550005 03fed702 00007fff
	v_add_co_u32 v2, s3, v3, v0                                // 000000004084: d7000302 02020103
	v_or_b32_e32 v8, 0x400000, v107                            // 00000000408c: 3810d6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004094: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v4, v1, s3                   // 000000004098: d5207c03 000e0304
	v_cmp_u_f32_e64 s3, v107, v107                             // 0000000040a0: d4180003 0202d76b
	s_wait_alu depctr_va_sdst(0)                               // 0000000040a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040ac: bf870001
	v_cndmask_b32_e64 v4, v5, v8, s3                           // 0000000040b0: d5010004 000e1105
	global_store_d16_hi_b16 v[2:3], v4, off                    // 0000000040b8: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000040c8: 8c7e047e
	s_lshl_b64 s[4:5], s[22:23], 1                             // 0000000040cc: 84848116
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040d0: bf88ff9e
	v_add_co_u32 v2, s3, s4, v32                               // 0000000040d4: d7000302 02024004
	s_wait_alu depctr_va_sdst(0)                               // 0000000040dc: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s5, v33, s3                  // 0000000040e0: d5207c03 000e4205
	v_cmp_lt_i64_e64 s3, 2, v[14:15]                           // 0000000040e8: d4510003 02021c82
	s_delay_alu instid0(valu_dep_2)                            // 0000000040f0: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000040f4: 3e040481
	s_and_b32 s4, s3, s1                                       // 0000000040f8: 8b040103
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040fc: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000004100: be852004
	s_cbranch_execz 27                                         // 000000004104: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2674>
	v_bfe_u32 v4, v104, 16, 1                                  // 000000004108: d6100004 02052168
	v_add_co_u32 v5, s4, s18, v6                               // 000000004110: d7000405 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004118: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v7, s4                  // 00000000411c: d5207c08 00120e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004124: bf870193
	v_add3_u32 v9, v4, v104, 0x7fff                            // 000000004128: d6550009 03fed104 00007fff
	v_add_co_u32 v4, s4, v5, v2                                // 000000004134: d7000404 02020505
	v_or_b32_e32 v10, 0x400000, v104                           // 00000000413c: 3814d0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004144: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v3, s4                   // 000000004148: d5207c05 00120708
	v_cmp_u_f32_e64 s4, v104, v104                             // 000000004150: d4180004 0202d168
	s_wait_alu depctr_va_sdst(0)                               // 000000004158: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000415c: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s4                          // 000000004160: d5010008 00121509
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000004168: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004174: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 000000004178: 8c7e057e
	v_mad_co_u64_u32 v[8:9], null, s22, 3, v[32:33]            // 00000000417c: d6fe7c08 04810616
	v_cmp_lt_i64_e64 s4, 3, v[14:15]                           // 000000004184: d4510004 02021c83
	s_and_b32 s5, s4, s1                                       // 00000000418c: 8b050104
	v_mad_co_u64_u32 v[9:10], null, s23, 3, v[9:10]            // 000000004190: d6fe7c09 04250617
	s_delay_alu instid0(valu_dep_1)                            // 000000004198: bf870001
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 00000000419c: 3e081081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041a0: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 0000000041a4: be862005
	s_cbranch_execz 27                                         // 0000000041a8: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2718>
	v_bfe_u32 v8, v101, 16, 1                                  // 0000000041ac: d6100008 02052165
	v_add_co_u32 v9, s5, s18, v6                               // 0000000041b4: d7000509 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 0000000041bc: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s19, v7, s5                 // 0000000041c0: d5207c0a 00160e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000041c8: bf870193
	v_add3_u32 v11, v8, v101, 0x7fff                           // 0000000041cc: d655000b 03fecb08 00007fff
	v_add_co_u32 v8, s5, v9, v4                                // 0000000041d8: d7000508 02020909
	v_or_b32_e32 v12, 0x400000, v101                           // 0000000041e0: 3818caff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000041e8: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v5, s5                  // 0000000041ec: d5207c09 00160b0a
	v_cmp_u_f32_e64 s5, v101, v101                             // 0000000041f4: d4180005 0202cb65
	s_wait_alu depctr_va_sdst(0)                               // 0000000041fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004200: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s5                        // 000000004204: d501000a 0016190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 00000000420c: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000004218: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 00000000421c: 8c7e067e
	s_lshl_b64 s[6:7], s[22:23], 2                             // 000000004220: 84868216
	s_wait_alu depctr_sa_sdst(0)                               // 000000004224: bf88ff9e
	v_add_co_u32 v8, s5, s6, v32                               // 000000004228: d7000508 02024006
	s_wait_alu depctr_va_sdst(0)                               // 000000004230: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s7, v33, s5                  // 000000004234: d5207c09 00164207
	v_cmp_lt_i64_e64 s5, 4, v[14:15]                           // 00000000423c: d4510005 02021c84
	s_delay_alu instid0(valu_dep_2)                            // 000000004244: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000004248: 3e101081
	s_and_b32 s6, s5, s1                                       // 00000000424c: 8b060105
	s_wait_alu depctr_sa_sdst(0)                               // 000000004250: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000004254: be872006
	s_cbranch_execz 27                                         // 000000004258: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x27c8>
	v_bfe_u32 v10, v98, 16, 1                                  // 00000000425c: d610000a 02052162
	v_add_co_u32 v11, s6, s18, v6                              // 000000004264: d700060b 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 00000000426c: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s19, v7, s6                 // 000000004270: d5207c0c 001a0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004278: bf870193
	v_add3_u32 v13, v10, v98, 0x7fff                           // 00000000427c: d655000d 03fec50a 00007fff
	v_add_co_u32 v10, s6, v11, v8                              // 000000004288: d700060a 0202110b
	v_or_b32_e32 v16, 0x400000, v98                            // 000000004290: 3820c4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004298: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s6                 // 00000000429c: d5207c0b 001a130c
	v_cmp_u_f32_e64 s6, v98, v98                               // 0000000042a4: d4180006 0202c562
	s_wait_alu depctr_va_sdst(0)                               // 0000000042ac: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000042b0: bf870001
	v_cndmask_b32_e64 v12, v13, v16, s6                        // 0000000042b4: d501000c 001a210d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 0000000042bc: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042c8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 0000000042cc: 8c7e077e
	v_mad_co_u64_u32 v[10:11], null, s22, 5, v[32:33]          // 0000000042d0: d6fe7c0a 04810a16
	v_cmp_lt_i64_e64 s6, 5, v[14:15]                           // 0000000042d8: d4510006 02021c85
	s_and_b32 s7, s6, s1                                       // 0000000042e0: 8b070106
	v_mad_co_u64_u32 v[11:12], null, s23, 5, v[11:12]          // 0000000042e4: d6fe7c0b 042d0a17
	s_delay_alu instid0(valu_dep_1)                            // 0000000042ec: bf870001
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 0000000042f0: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042f4: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 0000000042f8: be882007
	s_cbranch_execz 27                                         // 0000000042fc: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x286c>
	v_bfe_u32 v12, v93, 16, 1                                  // 000000004300: d610000c 0205215d
	v_add_co_u32 v13, s7, s18, v6                              // 000000004308: d700070d 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004310: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s19, v7, s7                 // 000000004314: d5207c10 001e0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000431c: bf870193
	v_add3_u32 v17, v12, v93, 0x7fff                           // 000000004320: d6550011 03febb0c 00007fff
	v_add_co_u32 v12, s7, v13, v10                             // 00000000432c: d700070c 0202150d
	v_or_b32_e32 v18, 0x400000, v93                            // 000000004334: 3824baff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000433c: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v16, v11, s7                // 000000004340: d5207c0d 001e1710
	v_cmp_u_f32_e64 s7, v93, v93                               // 000000004348: d4180007 0202bb5d
	s_wait_alu depctr_va_sdst(0)                               // 000000004350: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004354: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s7                        // 000000004358: d5010010 001e2511
	global_store_d16_hi_b16 v[12:13], v16, off                 // 000000004360: ee09407c 08000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000436c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 000000004370: 8c7e087e
	v_mad_co_u64_u32 v[16:17], null, s22, 6, v[32:33]          // 000000004374: d6fe7c10 04810c16
	v_cmp_lt_i64_e64 s7, 6, v[14:15]                           // 00000000437c: d4510007 02021c86
	s_and_b32 s8, s7, s1                                       // 000000004384: 8b080107
	v_mad_co_u64_u32 v[17:18], null, s23, 6, v[17:18]          // 000000004388: d6fe7c11 04450c17
	s_delay_alu instid0(valu_dep_1)                            // 000000004390: bf870001
	v_lshlrev_b64_e32 v[12:13], 1, v[16:17]                    // 000000004394: 3e182081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004398: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 00000000439c: be892008
	s_cbranch_execz 27                                         // 0000000043a0: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2910>
	v_bfe_u32 v16, v88, 16, 1                                  // 0000000043a4: d6100010 02052158
	v_add_co_u32 v17, s8, s18, v6                              // 0000000043ac: d7000811 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 0000000043b4: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v7, s8                 // 0000000043b8: d5207c12 00220e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000043c0: bf870193
	v_add3_u32 v19, v16, v88, 0x7fff                           // 0000000043c4: d6550013 03feb110 00007fff
	v_add_co_u32 v16, s8, v17, v12                             // 0000000043d0: d7000810 02021911
	v_or_b32_e32 v20, 0x400000, v88                            // 0000000043d8: 3828b0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000043e0: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v13, s8                // 0000000043e4: d5207c11 00221b12
	v_cmp_u_f32_e64 s8, v88, v88                               // 0000000043ec: d4180008 0202b158
	s_wait_alu depctr_va_sdst(0)                               // 0000000043f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043f8: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s8                        // 0000000043fc: d5010012 00222913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000004404: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000004410: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000004414: 8c7e097e
	v_mad_co_u64_u32 v[16:17], null, s22, 7, v[32:33]          // 000000004418: d6fe7c10 04810e16
	v_cmp_lt_i64_e64 s8, 7, v[14:15]                           // 000000004420: d4510008 02021c87
	s_and_b32 s9, s8, s1                                       // 000000004428: 8b090108
	v_mad_co_u64_u32 v[17:18], null, s23, 7, v[17:18]          // 00000000442c: d6fe7c11 04450e17
	s_delay_alu instid0(valu_dep_1)                            // 000000004434: bf870001
	v_lshlrev_b64_e32 v[14:15], 1, v[16:17]                    // 000000004438: 3e1c2081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000443c: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000004440: be8a2009
	s_cbranch_execz 27                                         // 000000004444: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x29b4>
	v_bfe_u32 v16, v85, 16, 1                                  // 000000004448: d6100010 02052155
	v_add_co_u32 v17, s9, s18, v6                              // 000000004450: d7000911 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004458: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v7, s9                 // 00000000445c: d5207c12 00260e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004464: bf870193
	v_add3_u32 v19, v16, v85, 0x7fff                           // 000000004468: d6550013 03feab10 00007fff
	v_add_co_u32 v16, s9, v17, v14                             // 000000004474: d7000910 02021d11
	v_or_b32_e32 v20, 0x400000, v85                            // 00000000447c: 3828aaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004484: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s9                // 000000004488: d5207c11 00261f12
	v_cmp_u_f32_e64 s9, v85, v85                               // 000000004490: d4180009 0202ab55
	s_wait_alu depctr_va_sdst(0)                               // 000000004498: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000449c: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s9                        // 0000000044a0: d5010012 00262913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 0000000044a8: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 0000000044b8: 8c7e0a7e
	v_mul_lo_u32 v20, s23, v38                                 // 0000000044bc: d72c0014 02024c17
	v_mul_lo_u32 v21, s22, v39                                 // 0000000044c4: d72c0015 02024e16
	v_mad_co_u64_u32 v[16:17], null, s22, v38, 0               // 0000000044cc: d6fe7c10 02024c16
	v_sub_co_u32 v18, s9, s20, v38                             // 0000000044d4: d7010912 02024c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000044dc: bf88f19f
	v_sub_co_ci_u32_e64 v19, null, s21, v39, s9                // 0000000044e0: d5217c13 00264e15
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 0000000044e8: bf870211
	v_cmp_lt_i64_e64 s9, 0, v[18:19]                           // 0000000044ec: d4510009 02022480
	v_add3_u32 v17, v17, v21, v20                              // 0000000044f4: d6550011 04522b11
	s_delay_alu instid0(valu_dep_1)                            // 0000000044fc: bf870001
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000004500: 3e202081
	s_and_b32 s10, s9, s1                                      // 000000004504: 8b0a0109
	s_wait_alu depctr_sa_sdst(0)                               // 000000004508: bf88ff9e
	s_and_saveexec_b32 s11, s10                                // 00000000450c: be8b200a
	s_cbranch_execz 28                                         // 000000004510: bfa5001c <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2a84>
	v_lshlrev_b64_e32 v[20:21], 1, v[32:33]                    // 000000004514: 3e284081
	v_add_co_u32 v23, s10, s18, v16                            // 000000004518: d7000a17 02022012
	v_bfe_u32 v22, v82, 16, 1                                  // 000000004520: d6100016 02052152
	s_wait_alu depctr_va_sdst(0)                               // 000000004528: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v17, s10               // 00000000452c: d5207c18 002a2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004534: bf870193
	v_add_co_u32 v20, s10, v23, v20                            // 000000004538: d7000a14 02022917
	v_add3_u32 v22, v22, v82, 0x7fff                           // 000000004540: d6550016 03fea516 00007fff
	v_or_b32_e32 v25, 0x400000, v82                            // 00000000454c: 3832a4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004554: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v24, v21, s10               // 000000004558: d5207c15 002a2b18
	v_cmp_u_f32_e64 s10, v82, v82                              // 000000004560: d418000a 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 000000004568: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000456c: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s10                       // 000000004570: d5010016 002a3316
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004578: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 000000004588: 8c7e0b7e
	v_cmp_lt_i64_e64 s10, 1, v[18:19]                          // 00000000458c: d451000a 02022481
	s_and_b32 s11, s10, s1                                     // 000000004594: 8b0b010a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004598: bf88ff9e
	s_and_saveexec_b32 s12, s11                                // 00000000459c: be8c200b
	s_cbranch_execz 27                                         // 0000000045a0: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2b10>
	v_bfe_u32 v20, v79, 16, 1                                  // 0000000045a4: d6100014 0205214f
	v_add_co_u32 v21, s11, s18, v16                            // 0000000045ac: d7000b15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b4: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s11               // 0000000045b8: d5207c16 002e2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000045c0: bf870193
	v_add3_u32 v23, v20, v79, 0x7fff                           // 0000000045c4: d6550017 03fe9f14 00007fff
	v_add_co_u32 v20, s11, v21, v0                             // 0000000045d0: d7000b14 02020115
	v_or_b32_e32 v24, 0x400000, v79                            // 0000000045d8: 38309eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000045e0: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v1, s11                // 0000000045e4: d5207c15 002e0316
	v_cmp_u_f32_e64 s11, v79, v79                              // 0000000045ec: d418000b 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 0000000045f4: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000045f8: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s11                       // 0000000045fc: d5010016 002e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004604: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004610: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004614: 8c7e0c7e
	v_cmp_lt_i64_e64 s11, 2, v[18:19]                          // 000000004618: d451000b 02022482
	s_and_b32 s12, s11, s1                                     // 000000004620: 8b0c010b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004624: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000004628: be8d200c
	s_cbranch_execz 27                                         // 00000000462c: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2b9c>
	v_bfe_u32 v20, v78, 16, 1                                  // 000000004630: d6100014 0205214e
	v_add_co_u32 v21, s12, s18, v16                            // 000000004638: d7000c15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004640: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s12               // 000000004644: d5207c16 00322213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000464c: bf870193
	v_add3_u32 v23, v20, v78, 0x7fff                           // 000000004650: d6550017 03fe9d14 00007fff
	v_add_co_u32 v20, s12, v21, v2                             // 00000000465c: d7000c14 02020515
	v_or_b32_e32 v24, 0x400000, v78                            // 000000004664: 38309cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000466c: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v3, s12                // 000000004670: d5207c15 00320716
	v_cmp_u_f32_e64 s12, v78, v78                              // 000000004678: d418000c 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000004680: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004684: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s12                       // 000000004688: d5010016 00323117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004690: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 00000000469c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 0000000046a0: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 3, v[18:19]                          // 0000000046a4: d451000c 02022483
	s_and_b32 s13, s12, s1                                     // 0000000046ac: 8b0d010c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046b0: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 0000000046b4: be8e200d
	s_cbranch_execz 27                                         // 0000000046b8: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2c28>
	v_bfe_u32 v20, v76, 16, 1                                  // 0000000046bc: d6100014 0205214c
	v_add_co_u32 v21, s13, s18, v16                            // 0000000046c4: d7000d15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000046cc: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s13               // 0000000046d0: d5207c16 00362213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000046d8: bf870193
	v_add3_u32 v23, v20, v76, 0x7fff                           // 0000000046dc: d6550017 03fe9914 00007fff
	v_add_co_u32 v20, s13, v21, v4                             // 0000000046e8: d7000d14 02020915
	v_or_b32_e32 v24, 0x400000, v76                            // 0000000046f0: 383098ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000046f8: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v5, s13                // 0000000046fc: d5207c15 00360b16
	v_cmp_u_f32_e64 s13, v76, v76                              // 000000004704: d418000d 0202994c
	s_wait_alu depctr_va_sdst(0)                               // 00000000470c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004710: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s13                       // 000000004714: d5010016 00363117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 00000000471c: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004728: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 00000000472c: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 4, v[18:19]                          // 000000004730: d451000d 02022484
	s_and_b32 s14, s13, s1                                     // 000000004738: 8b0e010d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000473c: bf88ff9e
	s_and_saveexec_b32 s15, s14                                // 000000004740: be8f200e
	s_cbranch_execz 27                                         // 000000004744: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2cb4>
	v_bfe_u32 v20, v72, 16, 1                                  // 000000004748: d6100014 02052148
	v_add_co_u32 v21, s14, s18, v16                            // 000000004750: d7000e15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004758: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s14               // 00000000475c: d5207c16 003a2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004764: bf870193
	v_add3_u32 v23, v20, v72, 0x7fff                           // 000000004768: d6550017 03fe9114 00007fff
	v_add_co_u32 v20, s14, v21, v8                             // 000000004774: d7000e14 02021115
	v_or_b32_e32 v24, 0x400000, v72                            // 00000000477c: 383090ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004784: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v9, s14                // 000000004788: d5207c15 003a1316
	v_cmp_u_f32_e64 s14, v72, v72                              // 000000004790: d418000e 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000004798: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000479c: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s14                       // 0000000047a0: d5010016 003a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000047a8: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047b4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000047b8: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 5, v[18:19]                          // 0000000047bc: d451000e 02022485
	s_and_b32 s15, s14, s1                                     // 0000000047c4: 8b0f010e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047c8: bf88ff9e
	s_and_saveexec_b32 s16, s15                                // 0000000047cc: be90200f
	s_cbranch_execz 27                                         // 0000000047d0: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2d40>
	v_bfe_u32 v20, v71, 16, 1                                  // 0000000047d4: d6100014 02052147
	v_add_co_u32 v21, s15, s18, v16                            // 0000000047dc: d7000f15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000047e4: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s15               // 0000000047e8: d5207c16 003e2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000047f0: bf870193
	v_add3_u32 v23, v20, v71, 0x7fff                           // 0000000047f4: d6550017 03fe8f14 00007fff
	v_add_co_u32 v20, s15, v21, v10                            // 000000004800: d7000f14 02021515
	v_or_b32_e32 v24, 0x400000, v71                            // 000000004808: 38308eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004810: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v11, s15               // 000000004814: d5207c15 003e1716
	v_cmp_u_f32_e64 s15, v71, v71                              // 00000000481c: d418000f 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000004824: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004828: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s15                       // 00000000482c: d5010016 003e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004834: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004840: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s16                             // 000000004844: 8c7e107e
	v_cmp_lt_i64_e64 s15, 6, v[18:19]                          // 000000004848: d451000f 02022486
	s_and_b32 s16, s15, s1                                     // 000000004850: 8b10010f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004854: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 000000004858: be912010
	s_cbranch_execz 27                                         // 00000000485c: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2dcc>
	v_bfe_u32 v20, v66, 16, 1                                  // 000000004860: d6100014 02052142
	v_add_co_u32 v21, s16, s18, v16                            // 000000004868: d7001015 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004870: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s16               // 000000004874: d5207c16 00422213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000487c: bf870193
	v_add3_u32 v23, v20, v66, 0x7fff                           // 000000004880: d6550017 03fe8514 00007fff
	v_add_co_u32 v20, s16, v21, v12                            // 00000000488c: d7001014 02021915
	v_or_b32_e32 v24, 0x400000, v66                            // 000000004894: 383084ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000489c: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v13, s16               // 0000000048a0: d5207c15 00421b16
	v_cmp_u_f32_e64 s16, v66, v66                              // 0000000048a8: d4180010 02028542
	s_wait_alu depctr_va_sdst(0)                               // 0000000048b0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048b4: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s16                       // 0000000048b8: d5010016 00423117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000048c0: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000048d0: 8c7e117e
	v_cmp_lt_i64_e64 s16, 7, v[18:19]                          // 0000000048d4: d4510010 02022487
	s_and_b32 s1, s16, s1                                      // 0000000048dc: 8b010110
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048e0: bf88ff9e
	s_and_saveexec_b32 s17, s1                                 // 0000000048e4: be912001
	s_cbranch_execz 27                                         // 0000000048e8: bfa5001b <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2e58>
	v_bfe_u32 v18, v65, 16, 1                                  // 0000000048ec: d6100012 02052141
	v_add_co_u32 v19, s1, s18, v16                             // 0000000048f4: d7000113 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000048fc: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v17, s1                // 000000004900: d5207c14 00062213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004908: bf870193
	v_add3_u32 v21, v18, v65, 0x7fff                           // 00000000490c: d6550015 03fe8312 00007fff
	v_add_co_u32 v18, s1, v19, v14                             // 000000004918: d7000112 02021d13
	v_or_b32_e32 v22, 0x400000, v65                            // 000000004920: 382c82ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004928: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v15, s1                // 00000000492c: d5207c13 00061f14
	v_cmp_u_f32_e64 s1, v65, v65                               // 000000004934: d4180001 02028341
	s_wait_alu depctr_va_sdst(0)                               // 00000000493c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004940: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s1                        // 000000004944: d5010014 00062d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 00000000494c: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004958: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 00000000495c: 8c7e117e
	s_and_b32 s17, vcc_lo, s0                                  // 000000004960: 8b11006a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004964: bf88ff9e
	s_and_saveexec_b32 s1, s17                                 // 000000004968: be812011
	s_cbranch_execz 25                                         // 00000000496c: bfa50019 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2ed4>
	v_lshlrev_b64_e32 v[18:19], 1, v[32:33]                    // 000000004970: 3e244081
	v_add_co_u32 v21, vcc_lo, s18, v6                          // 000000004974: d7006a15 02020c12
	v_bfe_u32 v20, v64, 16, 1                                  // 00000000497c: d6100014 02052140
	s_wait_alu depctr_va_vcc(0)                                // 000000004984: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s19, v7, vcc_lo             // 000000004988: d5207c16 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004990: bf870193
	v_add_co_u32 v18, vcc_lo, v21, v18                         // 000000004994: d7006a12 02022515
	v_add3_u32 v20, v20, v64, 0x7fff                           // 00000000499c: d6550014 03fe8114 00007fff
	v_or_b32_e32 v23, 0x400000, v64                            // 0000000049a8: 382e80ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000049b0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v22, v19, vcc_lo            // 0000000049b4: d5207c13 01aa2716
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 0000000049bc: 7c308140
	s_wait_alu depctr_va_vcc(0)                                // 0000000049c0: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v23, vcc_lo                    // 0000000049c4: 02282f14
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 0000000049c8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000049d8: 8c7e017e
	s_and_b32 s2, s2, s0                                       // 0000000049dc: 8b020002
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049e0: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 0000000049e4: be812002
	s_cbranch_execz 24                                         // 0000000049e8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2f4c>
	v_bfe_u32 v18, v63, 16, 1                                  // 0000000049ec: d6100012 0205213f
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 0000000049f4: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 0000000049fc: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004a00: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a08: bf870193
	v_add3_u32 v21, v18, v63, 0x7fff                           // 000000004a0c: d6550015 03fe7f12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v0                          // 000000004a18: d7006a12 02020113
	v_or_b32_e32 v22, 0x400000, v63                            // 000000004a20: 382c7eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a28: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v1, vcc_lo             // 000000004a2c: d5207c13 01aa0314
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000004a34: 7c307f3f
	s_wait_alu depctr_va_vcc(0)                                // 000000004a38: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004a3c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004a40: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a4c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004a50: 8c7e017e
	s_and_b32 s2, s3, s0                                       // 000000004a54: 8b020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a58: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004a5c: be812002
	s_cbranch_execz 24                                         // 000000004a60: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x2fc4>
	v_bfe_u32 v18, v61, 16, 1                                  // 000000004a64: d6100012 0205213d
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004a6c: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004a74: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004a78: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a80: bf870193
	v_add3_u32 v21, v18, v61, 0x7fff                           // 000000004a84: d6550015 03fe7b12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000004a90: d7006a12 02020513
	v_or_b32_e32 v22, 0x400000, v61                            // 000000004a98: 382c7aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004aa0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v3, vcc_lo             // 000000004aa4: d5207c13 01aa0714
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000004aac: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 000000004ab0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004ab4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004ab8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ac4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004ac8: 8c7e017e
	s_and_b32 s2, s4, s0                                       // 000000004acc: 8b020004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ad0: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004ad4: be812002
	s_cbranch_execz 24                                         // 000000004ad8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x303c>
	v_bfe_u32 v18, v60, 16, 1                                  // 000000004adc: d6100012 0205213c
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004ae4: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004aec: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004af0: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004af8: bf870193
	v_add3_u32 v21, v18, v60, 0x7fff                           // 000000004afc: d6550015 03fe7912 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v4                          // 000000004b08: d7006a12 02020913
	v_or_b32_e32 v22, 0x400000, v60                            // 000000004b10: 382c78ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004b18: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v5, vcc_lo             // 000000004b1c: d5207c13 01aa0b14
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000004b24: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 000000004b28: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004b2c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004b30: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b3c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004b40: 8c7e017e
	s_and_b32 s2, s5, s0                                       // 000000004b44: 8b020005
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b48: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004b4c: be812002
	s_cbranch_execz 24                                         // 000000004b50: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x30b4>
	v_bfe_u32 v18, v59, 16, 1                                  // 000000004b54: d6100012 0205213b
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004b5c: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004b64: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004b68: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b70: bf870193
	v_add3_u32 v21, v18, v59, 0x7fff                           // 000000004b74: d6550015 03fe7712 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v8                          // 000000004b80: d7006a12 02021113
	v_or_b32_e32 v22, 0x400000, v59                            // 000000004b88: 382c76ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004b90: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v9, vcc_lo             // 000000004b94: d5207c13 01aa1314
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000004b9c: 7c30773b
	s_wait_alu depctr_va_vcc(0)                                // 000000004ba0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004ba4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004ba8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004bb8: 8c7e017e
	s_and_b32 s2, s6, s0                                       // 000000004bbc: 8b020006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bc0: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004bc4: be812002
	s_cbranch_execz 24                                         // 000000004bc8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x312c>
	v_bfe_u32 v18, v58, 16, 1                                  // 000000004bcc: d6100012 0205213a
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004bd4: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004bdc: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004be0: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004be8: bf870193
	v_add3_u32 v21, v18, v58, 0x7fff                           // 000000004bec: d6550015 03fe7512 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v10                         // 000000004bf8: d7006a12 02021513
	v_or_b32_e32 v22, 0x400000, v58                            // 000000004c00: 382c74ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c08: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v11, vcc_lo            // 000000004c0c: d5207c13 01aa1714
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000004c14: 7c30753a
	s_wait_alu depctr_va_vcc(0)                                // 000000004c18: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004c1c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004c20: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004c30: 8c7e017e
	s_and_b32 s2, s7, s0                                       // 000000004c34: 8b020007
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c38: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004c3c: be812002
	s_cbranch_execz 24                                         // 000000004c40: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x31a4>
	v_bfe_u32 v18, v57, 16, 1                                  // 000000004c44: d6100012 02052139
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004c4c: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004c54: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004c58: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004c60: bf870193
	v_add3_u32 v21, v18, v57, 0x7fff                           // 000000004c64: d6550015 03fe7312 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v12                         // 000000004c70: d7006a12 02021913
	v_or_b32_e32 v22, 0x400000, v57                            // 000000004c78: 382c72ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c80: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v13, vcc_lo            // 000000004c84: d5207c13 01aa1b14
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000004c8c: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 000000004c90: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004c94: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004c98: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ca4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004ca8: 8c7e017e
	s_and_b32 s2, s8, s0                                       // 000000004cac: 8b020008
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cb0: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004cb4: be812002
	s_cbranch_execz 24                                         // 000000004cb8: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x321c>
	v_add_co_u32 v6, vcc_lo, s18, v6                           // 000000004cbc: d7006a06 02020c12
	v_bfe_u32 v18, v56, 16, 1                                  // 000000004cc4: d6100012 02052138
	s_wait_alu depctr_va_vcc(0)                                // 000000004ccc: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s19, v7, vcc_lo              // 000000004cd0: d5207c07 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004cd8: bf870193
	v_add_co_u32 v6, vcc_lo, v6, v14                           // 000000004cdc: d7006a06 02021d06
	v_add3_u32 v18, v18, v56, 0x7fff                           // 000000004ce4: d6550012 03fe7112 00007fff
	v_or_b32_e32 v19, 0x400000, v56                            // 000000004cf0: 382670ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004cf8: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo              // 000000004cfc: d5207c07 01aa1f07
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 000000004d04: 7c307138
	s_wait_alu depctr_va_vcc(0)                                // 000000004d08: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 000000004d0c: 02242712
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 000000004d10: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d1c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004d20: 8c7e017e
	s_and_b32 s2, s9, s0                                       // 000000004d24: 8b020009
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d28: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004d2c: be812002
	s_cbranch_execz 25                                         // 000000004d30: bfa50019 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3298>
	v_lshlrev_b64_e32 v[6:7], 1, v[32:33]                      // 000000004d34: 3e0c4081
	v_add_co_u32 v19, vcc_lo, s18, v16                         // 000000004d38: d7006a13 02022012
	v_bfe_u32 v18, v55, 16, 1                                  // 000000004d40: d6100012 02052137
	s_wait_alu depctr_va_vcc(0)                                // 000000004d48: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v17, vcc_lo            // 000000004d4c: d5207c14 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004d54: bf870193
	v_add_co_u32 v6, vcc_lo, v19, v6                           // 000000004d58: d7006a06 02020d13
	v_add3_u32 v18, v18, v55, 0x7fff                           // 000000004d60: d6550012 03fe6f12 00007fff
	v_or_b32_e32 v21, 0x400000, v55                            // 000000004d6c: 382a6eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004d74: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v20, v7, vcc_lo              // 000000004d78: d5207c07 01aa0f14
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000004d80: 7c306f37
	s_wait_alu depctr_va_vcc(0)                                // 000000004d84: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo                    // 000000004d88: 02242b12
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 000000004d8c: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d98: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004d9c: 8c7e017e
	s_and_b32 s2, s10, s0                                      // 000000004da0: 8b02000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004da4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004da8: be812002
	s_cbranch_execz 24                                         // 000000004dac: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3310>
	v_add_co_u32 v7, vcc_lo, s18, v16                          // 000000004db0: d7006a07 02022012
	v_bfe_u32 v6, v54, 16, 1                                   // 000000004db8: d6100006 02052136
	s_wait_alu depctr_va_vcc(0)                                // 000000004dc0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s19, v17, vcc_lo            // 000000004dc4: d5207c12 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004dcc: bf870193
	v_add_co_u32 v0, vcc_lo, v7, v0                            // 000000004dd0: d7006a00 02020107
	v_add3_u32 v6, v6, v54, 0x7fff                             // 000000004dd8: d6550006 03fe6d06 00007fff
	v_or_b32_e32 v19, 0x400000, v54                            // 000000004de4: 38266cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004dec: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v18, v1, vcc_lo              // 000000004df0: d5207c01 01aa0312
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 000000004df8: 7c306d36
	s_wait_alu depctr_va_vcc(0)                                // 000000004dfc: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v19, vcc_lo                      // 000000004e00: 020c2706
	global_store_d16_hi_b16 v[0:1], v6, off offset:32          // 000000004e04: ee09407c 03000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e10: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004e14: 8c7e017e
	s_and_b32 s2, s11, s0                                      // 000000004e18: 8b02000b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e1c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004e20: be812002
	s_cbranch_execz 24                                         // 000000004e24: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3388>
	v_bfe_u32 v0, v53, 16, 1                                   // 000000004e28: d6100000 02052135
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004e30: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004e38: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s19, v17, vcc_lo             // 000000004e3c: d5207c06 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e44: bf870193
	v_add3_u32 v7, v0, v53, 0x7fff                             // 000000004e48: d6550007 03fe6b00 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v2                            // 000000004e54: d7006a00 02020501
	v_or_b32_e32 v18, 0x400000, v53                            // 000000004e5c: 38246aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e64: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v6, v3, vcc_lo               // 000000004e68: d5207c01 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 000000004e70: 7c306b35
	s_wait_alu depctr_va_vcc(0)                                // 000000004e74: bf88ff9d
	v_cndmask_b32_e32 v2, v7, v18, vcc_lo                      // 000000004e78: 02042507
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004e7c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e88: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004e8c: 8c7e017e
	s_and_b32 s2, s12, s0                                      // 000000004e90: 8b02000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e94: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004e98: be812002
	s_cbranch_execz 24                                         // 000000004e9c: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3400>
	v_bfe_u32 v0, v52, 16, 1                                   // 000000004ea0: d6100000 02052134
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004ea8: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004eb0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004eb4: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ebc: bf870193
	v_add3_u32 v3, v0, v52, 0x7fff                             // 000000004ec0: d6550003 03fe6900 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v4                            // 000000004ecc: d7006a00 02020901
	v_or_b32_e32 v6, 0x400000, v52                             // 000000004ed4: 380c68ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004edc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v5, vcc_lo               // 000000004ee0: d5207c01 01aa0b02
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000004ee8: 7c306934
	s_wait_alu depctr_va_vcc(0)                                // 000000004eec: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v6, vcc_lo                       // 000000004ef0: 02040d03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004ef4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f00: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004f04: 8c7e017e
	s_and_b32 s2, s13, s0                                      // 000000004f08: 8b02000d
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f0c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004f10: be812002
	s_cbranch_execz 24                                         // 000000004f14: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3478>
	v_bfe_u32 v0, v51, 16, 1                                   // 000000004f18: d6100000 02052133
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004f20: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004f28: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004f2c: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f34: bf870193
	v_add3_u32 v3, v0, v51, 0x7fff                             // 000000004f38: d6550003 03fe6700 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v8                            // 000000004f44: d7006a00 02021101
	v_or_b32_e32 v4, 0x400000, v51                             // 000000004f4c: 380866ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f54: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v9, vcc_lo               // 000000004f58: d5207c01 01aa1302
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000004f60: 7c306733
	s_wait_alu depctr_va_vcc(0)                                // 000000004f64: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000004f68: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004f6c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f78: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004f7c: 8c7e017e
	s_and_b32 s2, s14, s0                                      // 000000004f80: 8b02000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f84: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004f88: be812002
	s_cbranch_execz 24                                         // 000000004f8c: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x34f0>
	v_bfe_u32 v0, v50, 16, 1                                   // 000000004f90: d6100000 02052132
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004f98: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004fa0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004fa4: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004fac: bf870193
	v_add3_u32 v3, v0, v50, 0x7fff                             // 000000004fb0: d6550003 03fe6500 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v10                           // 000000004fbc: d7006a00 02021501
	v_or_b32_e32 v4, 0x400000, v50                             // 000000004fc4: 380864ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004fcc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v11, vcc_lo              // 000000004fd0: d5207c01 01aa1702
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000004fd8: 7c306532
	s_wait_alu depctr_va_vcc(0)                                // 000000004fdc: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000004fe0: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004fe4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ff0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004ff4: 8c7e017e
	s_and_b32 s2, s15, s0                                      // 000000004ff8: 8b02000f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ffc: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000005000: be812002
	s_cbranch_execz 24                                         // 000000005004: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3568>
	v_bfe_u32 v0, v49, 16, 1                                   // 000000005008: d6100000 02052131
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000005010: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000005018: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 00000000501c: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005024: bf870193
	v_add3_u32 v3, v0, v49, 0x7fff                             // 000000005028: d6550003 03fe6300 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v12                           // 000000005034: d7006a00 02021901
	v_or_b32_e32 v4, 0x400000, v49                             // 00000000503c: 380862ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005044: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v13, vcc_lo              // 000000005048: d5207c01 01aa1b02
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 000000005050: 7c306331
	s_wait_alu depctr_va_vcc(0)                                // 000000005054: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005058: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000505c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005068: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000506c: 8c7e017e
	s_and_b32 s1, s16, s0                                      // 000000005070: 8b010010
	s_wait_alu depctr_sa_sdst(0)                               // 000000005074: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005078: be802001
	s_cbranch_execz 24                                         // 00000000507c: bfa50018 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x35e0>
	v_bfe_u32 v0, v35, 16, 1                                   // 000000005080: d6100000 02052123
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000005088: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000005090: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000005094: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000509c: bf870193
	v_add3_u32 v3, v0, v35, 0x7fff                             // 0000000050a0: d6550003 03fe4700 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v14                           // 0000000050ac: d7006a00 02021d01
	v_or_b32_e32 v4, 0x400000, v35                             // 0000000050b4: 380846ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000050bc: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v15, vcc_lo              // 0000000050c0: d5207c01 01aa1f02
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 0000000050c8: 7c304723
	s_wait_alu depctr_va_vcc(0)                                // 0000000050cc: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 0000000050d0: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000050d4: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050e0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050e4: 8c7e007e
	s_branch 62138                                             // 0000000050e8: bfa0f2ba <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0xd4>
	v_cmp_gt_i64_e32 vcc_lo, s[22:23], v[32:33]                // 0000000050ec: 7ca84016
	v_or_b32_e32 v2, s38, v34                                  // 0000000050f0: 38044426
	v_mov_b32_e32 v3, s39                                      // 0000000050f4: 7e060227
	v_or_b32_e32 v12, 1, v34                                   // 0000000050f8: 38184481
	v_or_b32_e32 v14, 3, v34                                   // 0000000050fc: 381c4483
	v_dual_mov_b32 v11, s39 :: v_dual_cndmask_b32 v8, 0, v32   // 000000005100: ca120027 0b084080
	s_delay_alu instid0(valu_dep_4)                            // 000000005108: bf870004
	v_cmp_gt_i64_e64 s0, s[20:21], v[2:3]                      // 00000000510c: d4540000 02020414
	v_mov_b32_e32 v1, s39                                      // 000000005114: 7e020227
	v_or_b32_e32 v0, s38, v12                                  // 000000005118: 38001826
	v_mov_b32_e32 v7, s39                                      // 00000000511c: 7e0e0227
	v_or_b32_e32 v6, s38, v14                                  // 000000005120: 380c1c26
	v_or_b32_e32 v16, 4, v34                                   // 000000005124: 38204484
	s_wait_alu depctr_va_sdst(0)                               // 000000005128: bf88f19f
	v_cndmask_b32_e64 v15, 0, v2, s0                           // 00000000512c: d501000f 00020480
	v_cndmask_b32_e64 v17, 0, s39, s0                          // 000000005134: d5010011 00004e80
	v_cmp_gt_i64_e64 s0, s[20:21], v[0:1]                      // 00000000513c: d4540000 02020014
	v_cndmask_b32_e32 v9, 0, v33, vcc_lo                       // 000000005144: 02124280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000005148: 7ca80c14
	v_or_b32_e32 v13, 2, v34                                   // 00000000514c: 381a4482
	v_mov_b32_e32 v35, 0                                       // 000000005150: 7e460280
	v_or_b32_e32 v26, 7, v34                                   // 000000005154: 38344487
	s_wait_alu depctr_va_sdst(0)                               // 000000005158: bf88f19f
	v_cndmask_b32_e64 v18, 0, v0, s0                           // 00000000515c: d5010012 00020080
	v_or_b32_e32 v0, s38, v16                                  // 000000005164: 38002026
	s_wait_alu depctr_va_vcc(0)                                // 000000005168: bf88ff9d
	v_dual_cndmask_b32 v22, 0, v6 :: v_dual_cndmask_b32 v23, 0, v7// 00000000516c: ca520c80 16160e80
	v_or_b32_e32 v4, s38, v13                                  // 000000005174: 38081a26
	v_or_b32_e32 v10, s38, v26                                 // 000000005178: 38143426
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 00000000517c: 7ca80014
	v_mov_b32_e32 v5, s39                                      // 000000005180: 7e0a0227
	v_or_b32_e32 v24, 5, v34                                   // 000000005184: 38304485
	v_or_b32_e32 v25, 6, v34                                   // 000000005188: 38324486
	v_mul_lo_u32 v17, v17, s28                                 // 00000000518c: d72c0011 02003911
	s_wait_alu depctr_va_vcc(0)                                // 000000005194: bf88ff9d
	v_dual_mov_b32 v80, v35 :: v_dual_cndmask_b32 v27, 0, v0   // 000000005198: ca120123 501a0080
	v_cndmask_b32_e32 v28, 0, v1, vcc_lo                       // 0000000051a0: 02380280
	v_cmp_gt_i64_e64 s1, s[20:21], v[4:5]                      // 0000000051a4: d4540001 02020814
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[10:11]                // 0000000051ac: 7ca81414
	v_or_b32_e32 v6, s38, v25                                  // 0000000051b0: 380c3226
	v_or_b32_e32 v0, s33, v34                                  // 0000000051b4: 38004421
	v_mov_b32_e32 v74, v35                                     // 0000000051b8: 7e940323
	v_mul_lo_u32 v28, v28, s28                                 // 0000000051bc: d72c001c 0200391c
	s_wait_alu depctr_va_sdst(0)                               // 0000000051c4: bf88f19f
	v_cndmask_b32_e64 v20, 0, v4, s1                           // 0000000051c8: d5010014 00060880
	v_or_b32_e32 v4, s38, v24                                  // 0000000051d0: 38083026
	s_wait_alu depctr_va_vcc(0)                                // 0000000051d4: bf88ff9d
	v_cndmask_b32_e32 v40, 0, v11, vcc_lo                      // 0000000051d8: 02501680
	v_cndmask_b32_e64 v19, 0, v1, s0                           // 0000000051dc: d5010013 00020280
	v_cndmask_b32_e64 v21, 0, v5, s1                           // 0000000051e4: d5010015 00060a80
	v_cmp_gt_i64_e64 s1, s[20:21], v[6:7]                      // 0000000051ec: d4540001 02020c14
	v_cmp_gt_i64_e64 s0, s[20:21], v[4:5]                      // 0000000051f4: d4540000 02020814
	v_dual_cndmask_b32 v39, 0, v10 :: v_dual_mov_b32 v78, v35  // 0000000051fc: ca501480 274e0123
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000005204: 7ca80014
	v_or_b32_e32 v10, s33, v13                                 // 000000005208: 38141a21
	s_wait_alu depctr_va_sdst(0)                               // 00000000520c: bf88f19f
	v_cndmask_b32_e64 v31, 0, v6, s1                           // 000000005210: d501001f 00060c80
	v_cndmask_b32_e64 v29, 0, v4, s0                           // 000000005218: d501001d 00020880
	v_cndmask_b32_e64 v30, 0, s39, s0                          // 000000005220: d501001e 00004e80
	v_cmp_gt_i64_e64 s0, s[22:23], v[36:37]                    // 000000005228: d4540000 02024816
	v_or_b32_e32 v4, s33, v12                                  // 000000005230: 38081821
	s_wait_alu depctr_va_vcc(0)                                // 000000005234: bf88ff9d
	v_cndmask_b32_e64 v41, 0, s39, vcc_lo                      // 000000005238: d5010029 01a84e80
	v_dual_mov_b32 v12, s39 :: v_dual_mov_b32 v13, s39         // 000000005240: ca100027 0c0c0027
	v_mul_lo_u32 v55, v27, s29                                 // 000000005248: d72c0037 02003b1b
	s_wait_alu depctr_va_sdst(0)                               // 000000005250: bf88f19f
	v_cndmask_b32_e64 v6, 0, v36, s0                           // 000000005254: d5010006 00024880
	v_cndmask_b32_e32 v36, 0, v0, vcc_lo                       // 00000000525c: 02480080
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000005260: 7ca80814
	v_cndmask_b32_e64 v38, 0, v7, s1                           // 000000005264: d5010026 00060e80
	v_cndmask_b32_e64 v7, 0, v37, s0                           // 00000000526c: d5010007 00024a80
	v_mul_lo_u32 v54, v22, s29                                 // 000000005274: d72c0036 02003b16
	v_mul_lo_u32 v41, v41, s28                                 // 00000000527c: d72c0029 02003929
	s_wait_alu depctr_va_vcc(0)                                // 000000005284: bf88ff9d
	v_dual_mov_b32 v81, v35 :: v_dual_cndmask_b32 v42, 0, v4   // 000000005288: ca120123 512a0880
	v_or_b32_e32 v4, s33, v14                                  // 000000005290: 38081c21
	v_cndmask_b32_e64 v43, 0, s39, vcc_lo                      // 000000005294: d501002b 01a84e80
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 00000000529c: 3e0c0c82
	v_mov_b32_e32 v79, v35                                     // 0000000052a0: 7e9e0323
	v_mov_b32_e32 v75, v35                                     // 0000000052a4: 7e960323
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 0000000052a8: 7ca80814
	v_dual_mov_b32 v71, v35 :: v_dual_mov_b32 v70, v35         // 0000000052ac: ca100123 47460123
	v_dual_mov_b32 v57, v35 :: v_dual_mov_b32 v66, v35         // 0000000052b4: ca100123 39420123
	s_wait_alu depctr_va_vcc(0)                                // 0000000052bc: bf88ff9d
	v_dual_mov_b32 v67, v35 :: v_dual_cndmask_b32 v46, 0, v4   // 0000000052c0: ca120123 432e0880
	v_cmp_gt_i64_e64 s0, s[20:21], v[10:11]                    // 0000000052c8: d4540000 02021414
	v_or_b32_e32 v11, s33, v16                                 // 0000000052d0: 38162021
	v_or_b32_e32 v4, s33, v25                                  // 0000000052d4: 38083221
	v_cndmask_b32_e64 v47, 0, s39, vcc_lo                      // 0000000052d8: d501002f 01a84e80
	v_dual_mov_b32 v58, v35 :: v_dual_mov_b32 v63, v35         // 0000000052e0: ca100123 3a3e0123
	s_wait_alu depctr_va_sdst(0)                               // 0000000052e8: bf88f19f
	v_cndmask_b32_e64 v44, 0, v10, s0                          // 0000000052ec: d501002c 00021480
	v_cndmask_b32_e64 v45, 0, s39, s0                          // 0000000052f4: d501002d 00004e80
	v_cmp_gt_i64_e64 s0, s[20:21], v[11:12]                    // 0000000052fc: d4540000 02021614
	v_or_b32_e32 v12, s33, v24                                 // 000000005304: 38183021
	v_or_b32_e32 v10, s33, v26                                 // 000000005308: 38143421
	v_mov_b32_e32 v61, v35                                     // 00000000530c: 7e7a0323
	v_mov_b32_e32 v59, v35                                     // 000000005310: 7e760323
	s_lshl_b64 s[10:11], s[22:23], 2                           // 000000005314: 848a8216
	v_cndmask_b32_e64 v49, 0, v11, s0                          // 000000005318: d5010031 00021680
	v_cndmask_b32_e64 v50, 0, s39, s0                          // 000000005320: d5010032 00004e80
	v_add_co_u32 v14, s0, s40, v48                             // 000000005328: d700000e 02026028
	s_wait_alu depctr_va_sdst(0)                               // 000000005330: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s41, 0, s0                  // 000000005334: d5207c10 00010029
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[12:13]                // 00000000533c: 7ca81814
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005340: bf8701a3
	v_add_co_u32 v24, s0, v14, 16                              // 000000005344: d7000018 0201210e
	s_wait_alu depctr_va_sdst(0)                               // 00000000534c: bf88f19f
	v_add_co_ci_u32_e64 v25, null, 0, v16, s0                  // 000000005350: d5207c19 00022080
	v_mov_b32_e32 v11, s39                                     // 000000005358: 7e160227
	v_cmp_gt_i64_e64 s0, s[20:21], v[4:5]                      // 00000000535c: d4540000 02020814
	s_wait_alu depctr_va_vcc(0)                                // 000000005364: bf88ff9d
	v_cndmask_b32_e32 v51, 0, v12, vcc_lo                      // 000000005368: 02661880
	v_mad_co_u64_u32 v[12:13], null, s24, v24, v[34:35]        // 00000000536c: d6fe7c0c 048a3018
	v_mul_lo_u32 v25, s24, v25                                 // 000000005374: d72c0019 02023218
	v_mul_lo_u32 v24, s25, v24                                 // 00000000537c: d72c0018 02023019
	v_cmp_gt_i64_e64 s1, s[20:21], v[10:11]                    // 000000005384: d4540001 02021414
	s_wait_alu depctr_va_sdst(0)                               // 00000000538c: bf88f19f
	v_cndmask_b32_e64 v26, 0, v4, s0                           // 000000005390: d501001a 00020880
	v_lshlrev_b64_e32 v[4:5], 2, v[8:9]                        // 000000005398: 3e081082
	v_cndmask_b32_e64 v52, 0, s39, vcc_lo                      // 00000000539c: d5010034 01a84e80
	v_cndmask_b32_e64 v37, 0, s39, s0                          // 0000000053a4: d5010025 00004e80
	v_mul_lo_u32 v50, v50, s28                                 // 0000000053ac: d72c0032 02003932
	v_cndmask_b32_e64 v10, 0, v10, s1                          // 0000000053b4: d501000a 00061480
	v_cndmask_b32_e64 v11, 0, s39, s1                          // 0000000053bc: d501000b 00044e80
	v_add3_u32 v8, v24, v13, v25                               // 0000000053c4: d6550008 04661b18
	v_add_co_u32 v4, vcc_lo, s36, v4                           // 0000000053cc: d7006a04 02020824
	v_add_co_u32 v24, s0, s38, v48                             // 0000000053d4: d7000018 02026026
	s_wait_alu depctr_va_vcc(0)                                // 0000000053dc: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s37, v5, vcc_lo              // 0000000053e0: d5207c05 01aa0a25
	v_add_co_u32 v64, vcc_lo, s34, v12                         // 0000000053e8: d7006a40 02021822
	s_wait_alu depctr_va_sdst(0)                               // 0000000053f0: bf88f19f
	v_add_co_ci_u32_e64 v25, null, s39, 0, s0                  // 0000000053f4: d5207c19 00010027
	s_wait_alu depctr_va_vcc(0)                                // 0000000053fc: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, s35, v8, vcc_lo             // 000000005400: d5207c41 01aa1023
	v_mad_co_u64_u32 v[8:9], null, s24, v14, v[34:35]          // 000000005408: d6fe7c08 048a1c18
	v_mul_lo_u32 v12, s24, v16                                 // 000000005410: d72c000c 02022018
	v_mul_lo_u32 v13, s25, v14                                 // 000000005418: d72c000d 02021c19
	v_mul_lo_u32 v14, v11, s28                                 // 000000005420: d72c000e 0200390b
	v_mul_lo_u32 v16, v10, s29                                 // 000000005428: d72c0010 02003b0a
	v_mad_co_u64_u32 v[10:11], null, v10, s28, 0               // 000000005430: d6fe7c0a 0200390a
	v_add_co_u32 v48, vcc_lo, v24, 16                          // 000000005438: d7006a30 02012118
	s_wait_alu depctr_va_vcc(0)                                // 000000005440: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v25, vcc_lo              // 000000005444: d5207c35 01aa3280
	v_add3_u32 v9, v13, v9, v12                                // 00000000544c: d6550009 0432130d
	s_delay_alu instid0(valu_dep_3)                            // 000000005454: bf870003
	v_mad_co_u64_u32 v[12:13], null, s24, v48, v[34:35]        // 000000005458: d6fe7c0c 048a6018
	v_add_co_u32 v6, vcc_lo, s36, v6                           // 000000005460: d7006a06 02020c24
	v_add3_u32 v11, v11, v16, v14                              // 000000005468: d655000b 043a210b
	v_mul_lo_u32 v14, s24, v53                                 // 000000005470: d72c000e 02026a18
	v_mul_lo_u32 v16, s25, v48                                 // 000000005478: d72c0010 02026019
	s_wait_alu depctr_va_vcc(0)                                // 000000005480: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s37, v7, vcc_lo              // 000000005484: d5207c07 01aa0e25
	v_add_co_u32 v68, vcc_lo, s34, v8                          // 00000000548c: d7006a44 02021022
	s_wait_alu depctr_va_vcc(0)                                // 000000005494: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, s35, v9, vcc_lo             // 000000005498: d5207c45 01aa1223
	v_lshlrev_b64_e32 v[8:9], 2, v[10:11]                      // 0000000054a0: 3e101482
	v_mul_lo_u32 v37, v37, s28                                 // 0000000054a4: d72c0025 02003925
	v_mul_lo_u32 v48, v26, s29                                 // 0000000054ac: d72c0030 02003b1a
	v_mad_co_u64_u32 v[10:11], null, v26, s28, 0               // 0000000054b4: d6fe7c0a 0200391a
	v_add3_u32 v26, v16, v13, v14                              // 0000000054bc: d655001a 043a1b10
	v_mad_co_u64_u32 v[13:14], null, s24, v24, v[34:35]        // 0000000054c4: d6fe7c0d 048a3018
	v_mul_lo_u32 v25, s24, v25                                 // 0000000054cc: d72c0019 02023218
	v_mul_lo_u32 v24, s25, v24                                 // 0000000054d4: d72c0018 02023019
	v_mul_lo_u32 v34, v15, s29                                 // 0000000054dc: d72c0022 02003b0f
	v_mad_co_u64_u32 v[15:16], null, v15, s28, 0               // 0000000054e4: d6fe7c0f 0200390f
	v_add_co_u32 v72, vcc_lo, s30, v12                         // 0000000054ec: d7006a48 0202181e
	v_add3_u32 v11, v11, v48, v37                              // 0000000054f4: d655000b 0496610b
	s_wait_alu depctr_va_vcc(0)                                // 0000000054fc: bf88ff9d
	v_add_co_ci_u32_e64 v73, null, s31, v26, vcc_lo            // 000000005500: d5207c49 01aa341f
	v_add3_u32 v12, v24, v14, v25                              // 000000005508: d655000c 04661d18
	v_mul_lo_u32 v14, v19, s28                                 // 000000005510: d72c000e 02003913
	v_add3_u32 v16, v16, v34, v17                              // 000000005518: d6550010 04464510
	v_mul_lo_u32 v34, v18, s29                                 // 000000005520: d72c0022 02003b12
	v_mad_co_u64_u32 v[17:18], null, v18, s28, 0               // 000000005528: d6fe7c11 02003912
	v_mul_lo_u32 v37, v21, s28                                 // 000000005530: d72c0025 02003915
	v_mul_lo_u32 v48, v20, s29                                 // 000000005538: d72c0030 02003b14
	v_mad_co_u64_u32 v[19:20], null, v20, s28, 0               // 000000005540: d6fe7c13 02003914
	v_mul_lo_u32 v53, v23, s28                                 // 000000005548: d72c0035 02003917
	v_mad_co_u64_u32 v[23:24], null, v27, s28, 0               // 000000005550: d6fe7c17 0200391b
	v_mul_lo_u32 v27, v30, s28                                 // 000000005558: d72c001b 0200391e
	v_mul_lo_u32 v30, v29, s29                                 // 000000005560: d72c001e 02003b1d
	v_mad_co_u64_u32 v[25:26], null, v29, s28, 0               // 000000005568: d6fe7c19 0200391d
	v_mad_co_u64_u32 v[21:22], null, v22, s28, 0               // 000000005570: d6fe7c15 02003916
	v_add3_u32 v18, v18, v34, v14                              // 000000005578: d6550012 043a4512
	v_add3_u32 v20, v20, v48, v37                              // 000000005580: d6550014 04966114
	v_mul_lo_u32 v34, v38, s28                                 // 000000005588: d72c0022 02003926
	v_add3_u32 v24, v24, v55, v28                              // 000000005590: d6550018 04726f18
	v_mul_lo_u32 v38, v31, s29                                 // 000000005598: d72c0026 02003b1f
	v_mul_lo_u32 v48, v45, s28                                 // 0000000055a0: d72c0030 0200392d
	v_add3_u32 v26, v26, v30, v27                              // 0000000055a8: d655001a 046e3d1a
	v_mad_co_u64_u32 v[27:28], null, v31, s28, 0               // 0000000055b0: d6fe7c1b 0200391f
	v_mul_lo_u32 v31, v40, s28                                 // 0000000055b8: d72c001f 02003928
	v_mul_lo_u32 v40, v39, s29                                 // 0000000055c0: d72c0028 02003b27
	v_mad_co_u64_u32 v[29:30], null, v39, s28, 0               // 0000000055c8: d6fe7c1d 02003927
	v_mul_lo_u32 v39, v36, s29                                 // 0000000055d0: d72c0027 02003b24
	v_mad_co_u64_u32 v[36:37], null, v36, s28, 0               // 0000000055d8: d6fe7c24 02003924
	v_add3_u32 v22, v22, v54, v53                              // 0000000055e0: d6550016 04d66d16
	v_mul_lo_u32 v53, v44, s29                                 // 0000000055e8: d72c0035 02003b2c
	v_add3_u32 v28, v28, v38, v34                              // 0000000055f0: d655001c 048a4d1c
	v_mul_lo_u32 v34, v42, s29                                 // 0000000055f8: d72c0022 02003b2a
	v_mul_lo_u32 v54, v47, s28                                 // 000000005600: d72c0036 0200392f
	v_add3_u32 v30, v30, v40, v31                              // 000000005608: d655001e 047e511e
	v_mul_lo_u32 v31, v43, s28                                 // 000000005610: d72c001f 0200392b
	v_add3_u32 v37, v37, v39, v41                              // 000000005618: d6550025 04a64f25
	v_mad_co_u64_u32 v[38:39], null, v42, s28, 0               // 000000005620: d6fe7c26 0200392a
	v_mad_co_u64_u32 v[40:41], null, v44, s28, 0               // 000000005628: d6fe7c28 0200392c
	v_mul_lo_u32 v55, v46, s29                                 // 000000005630: d72c0037 02003b2e
	v_mad_co_u64_u32 v[42:43], null, v46, s28, 0               // 000000005638: d6fe7c2a 0200392e
	v_mul_lo_u32 v56, v49, s29                                 // 000000005640: d72c0038 02003b31
	v_mad_co_u64_u32 v[44:45], null, v49, s28, 0               // 000000005648: d6fe7c2c 02003931
	v_mul_lo_u32 v49, v52, s28                                 // 000000005650: d72c0031 02003934
	v_mul_lo_u32 v52, v51, s29                                 // 000000005658: d72c0034 02003b33
	v_mad_co_u64_u32 v[46:47], null, v51, s28, 0               // 000000005660: d6fe7c2e 02003933
	v_add3_u32 v39, v39, v34, v31                              // 000000005668: d6550027 047e4527
	v_add3_u32 v41, v41, v53, v48                              // 000000005670: d6550029 04c26b29
	v_add3_u32 v43, v43, v55, v54                              // 000000005678: d655002b 04da6f2b
	v_add_co_u32 v76, vcc_lo, s30, v13                         // 000000005680: d7006a4c 02021a1e
	v_add3_u32 v45, v45, v56, v50                              // 000000005688: d655002d 04ca712d
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 000000005690: 3e141482
	v_add3_u32 v47, v47, v52, v49                              // 000000005694: d655002f 04c6692f
	s_wait_alu depctr_va_vcc(0)                                // 00000000569c: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s31, v12, vcc_lo            // 0000000056a0: d5207c4d 01aa181f
	v_lshlrev_b64_e32 v[12:13], 2, v[15:16]                    // 0000000056a8: 3e181e82
	v_lshlrev_b64_e32 v[14:15], 2, v[17:18]                    // 0000000056ac: 3e1c2282
	v_lshlrev_b64_e32 v[16:17], 2, v[19:20]                    // 0000000056b0: 3e202682
	v_lshlrev_b64_e32 v[18:19], 2, v[21:22]                    // 0000000056b4: 3e242a82
	v_lshlrev_b64_e32 v[20:21], 2, v[23:24]                    // 0000000056b8: 3e282e82
	v_lshlrev_b64_e32 v[22:23], 2, v[25:26]                    // 0000000056bc: 3e2c3282
	v_lshlrev_b64_e32 v[24:25], 2, v[27:28]                    // 0000000056c0: 3e303682
	v_lshlrev_b64_e32 v[26:27], 2, v[29:30]                    // 0000000056c4: 3e343a82
	v_lshlrev_b64_e32 v[28:29], 2, v[36:37]                    // 0000000056c8: 3e384882
	v_lshlrev_b64_e32 v[30:31], 2, v[38:39]                    // 0000000056cc: 3e3c4c82
	v_lshlrev_b64_e32 v[36:37], 2, v[40:41]                    // 0000000056d0: 3e485082
	v_lshlrev_b64_e32 v[38:39], 2, v[42:43]                    // 0000000056d4: 3e4c5482
	v_lshlrev_b64_e32 v[40:41], 2, v[44:45]                    // 0000000056d8: 3e505882
	v_lshlrev_b64_e32 v[42:43], 2, v[46:47]                    // 0000000056dc: 3e545c82
	v_dual_mov_b32 v56, v35 :: v_dual_mov_b32 v55, v35         // 0000000056e0: ca100123 38360123
	v_mov_b32_e32 v62, v35                                     // 0000000056e8: 7e7c0323
	v_dual_mov_b32 v54, v35 :: v_dual_mov_b32 v53, v35         // 0000000056ec: ca100123 36340123
	v_mov_b32_e32 v60, v35                                     // 0000000056f4: 7e780323
	v_mov_b32_e32 v52, v35                                     // 0000000056f8: 7e680323
	v_dual_mov_b32 v50, v35 :: v_dual_mov_b32 v51, v35         // 0000000056fc: ca100123 32320123
	v_dual_mov_b32 v49, v35 :: v_dual_mov_b32 v48, v35         // 000000005704: ca100123 31300123
	v_dual_mov_b32 v47, v35 :: v_dual_mov_b32 v46, v35         // 00000000570c: ca100123 2f2e0123
	v_dual_mov_b32 v45, v35 :: v_dual_mov_b32 v44, v35         // 000000005714: ca100123 2d2c0123
	v_mov_b32_e32 v34, v35                                     // 00000000571c: 7e440323
	s_mov_b64 s[12:13], 0                                      // 000000005720: be8c0180
	v_add_co_u32 v84, s0, s26, v12                             // 000000005724: d7000054 0202181a
	v_add_co_u32 v82, vcc_lo, v76, s12                         // 00000000572c: d7006a52 0200194c
	v_add_co_u32 v100, s8, v68, s12                            // 000000005734: d7000864 02001944
	s_wait_alu depctr_va_sdst(0)                               // 00000000573c: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s27, v13, s0                // 000000005740: d5207c55 00021a1b
	v_add_co_u32 v86, s1, s26, v14                             // 000000005748: d7000156 02021c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005750: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s13, v77, vcc_lo            // 000000005754: d5207c53 01aa9a0d
	v_add_co_u32 v88, s2, s26, v16                             // 00000000575c: d7000258 0202201a
	v_add_co_u32 v90, s3, s26, v18                             // 000000005764: d700035a 0202241a
	v_add_co_u32 v92, s4, s26, v20                             // 00000000576c: d700045c 0202281a
	v_add_co_ci_u32_e64 v101, null, s13, v69, s8               // 000000005774: d5207c65 00228a0d
	v_add_co_u32 v94, s5, s26, v22                             // 00000000577c: d700055e 02022c1a
	v_add_co_u32 v96, s6, s26, v24                             // 000000005784: d7000660 0202301a
	v_add_co_u32 v98, s7, s26, v26                             // 00000000578c: d7000762 0202341a
	s_wait_alu depctr_va_sdst(0)                               // 000000005794: bf88f19f
	v_add_co_ci_u32_e64 v87, null, s27, v15, s1                // 000000005798: d5207c57 00061e1b
	global_load_b32 v106, v[4:5], off                          // 0000000057a0: ee05007c 0000006a 00000004
	v_add_co_ci_u32_e64 v89, null, s27, v17, s2                // 0000000057ac: d5207c59 000a221b
	v_add_co_ci_u32_e64 v91, null, s27, v19, s3                // 0000000057b4: d5207c5b 000e261b
	v_add_co_ci_u32_e64 v93, null, s27, v21, s4                // 0000000057bc: d5207c5d 00122a1b
	v_add_co_ci_u32_e64 v95, null, s27, v23, s5                // 0000000057c4: d5207c5f 00162e1b
	v_add_co_ci_u32_e64 v97, null, s27, v25, s6                // 0000000057cc: d5207c61 001a321b
	v_add_co_ci_u32_e64 v99, null, s27, v27, s7                // 0000000057d4: d5207c63 001e361b
	global_load_b32 v107, v[84:85], off                        // 0000000057dc: ee05007c 0000006b 00000054
	global_load_b64 v[102:103], v[82:83], off                  // 0000000057e8: ee05407c 00000066 00000052
	global_load_b64 v[104:105], v[100:101], off                // 0000000057f4: ee05407c 00000068 00000064
	s_clause 0x5                                               // 000000005800: bf850005
	global_load_b32 v108, v[86:87], off                        // 000000005804: ee05007c 0000006c 00000056
	global_load_b32 v109, v[88:89], off                        // 000000005810: ee05007c 0000006d 00000058
	global_load_b32 v110, v[90:91], off                        // 00000000581c: ee05007c 0000006e 0000005a
	global_load_b32 v111, v[92:93], off                        // 000000005828: ee05007c 0000006f 0000005c
	global_load_b32 v112, v[94:95], off                        // 000000005834: ee05007c 00000070 0000005e
	global_load_b32 v113, v[96:97], off                        // 000000005840: ee05007c 00000071 00000060
	global_load_b64 v[90:91], v[82:83], off offset:16          // 00000000584c: ee05407c 0000005a 00001052
	global_load_b64 v[92:93], v[100:101], off offset:16        // 000000005858: ee05407c 0000005c 00001064
	global_load_b32 v98, v[98:99], off                         // 000000005864: ee05007c 00000062 00000062
	s_wait_loadcnt 0xb                                         // 000000005870: bfc0000b
	v_mul_f32_e32 v94, v107, v106                              // 000000005874: 10bcd56b
	s_wait_loadcnt 0x9                                         // 000000005878: bfc00009
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[102:103], v[104:105], 0// 00000000587c: cc464052 1a02d166
	s_wait_loadcnt 0x7                                         // 000000005884: bfc00007
	v_dual_mul_f32 v95, v106, v108 :: v_dual_mul_f32 v96, v106, v109// 000000005888: c8c6d96a 5f60db6a
	s_wait_loadcnt 0x6                                         // 000000005890: bfc00006
	v_mul_f32_e32 v97, v106, v110                              // 000000005894: 10c2dd6a
	s_wait_loadcnt 0x4                                         // 000000005898: bfc00004
	v_dual_mul_f32 v99, v106, v111 :: v_dual_mul_f32 v100, v106, v112// 00000000589c: c8c6df6a 6364e16a
	s_wait_loadcnt 0x1                                         // 0000000058a4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[90:91], v[92:93], v[82:89]// 0000000058a8: cc464052 1d4ab95a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_2)// 0000000058b0: bf870111
	v_dual_mul_f32 v101, v106, v113 :: v_dual_mul_f32 v116, v84, v96// 0000000058b4: c8c6e36a 6574c154
	v_dual_mul_f32 v114, v82, v94 :: v_dual_mul_f32 v115, v83, v95// 0000000058bc: c8c6bd52 7272bf53
	s_wait_loadcnt 0x0                                         // 0000000058c4: bfc00000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 0000000058c8: bf8701b3
	v_dual_mul_f32 v82, v106, v98 :: v_dual_mul_f32 v117, v85, v97// 0000000058cc: c8c6c56a 5274c355
	v_mul_f32_e32 v118, v86, v99                               // 0000000058d4: 10ecc756
	v_dual_mul_f32 v100, v87, v100 :: v_dual_mul_f32 v101, v88, v101// 0000000058d8: c8c6c957 6464cb58
	v_mul_f32_e32 v119, v89, v82                               // 0000000058e0: 10eea559
	v_add_co_u32 v82, vcc_lo, v64, s12                         // 0000000058e4: d7006a52 02001940
	s_wait_alu depctr_va_vcc(0)                                // 0000000058ec: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s13, v65, vcc_lo            // 0000000058f0: d5207c53 01aa820d
	s_clause 0x1                                               // 0000000058f8: bf850001
	global_load_b64 v[94:95], v[82:83], off                    // 0000000058fc: ee05407c 0000005e 00000052
	global_load_b64 v[96:97], v[82:83], off offset:16          // 000000005908: ee05407c 00000060 00001052
	v_dual_add_f32 v35, v35, v114 :: v_dual_add_f32 v74, v74, v101// 000000005914: c908e523 234acb4a
	v_add_f32_e32 v80, v80, v116                               // 00000000591c: 06a0e950
	v_add_f32_e32 v78, v78, v118                               // 000000005920: 069ced4e
	s_wait_loadcnt 0x1                                         // 000000005924: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[102:103], v[94:95], 0// 000000005928: cc464052 1a02bd66
	global_load_b32 v102, v[6:7], off                          // 000000005930: ee05007c 00000066 00000006
	v_add_co_u32 v6, s0, v6, s10                               // 00000000593c: d7000006 02001506
	s_wait_loadcnt 0x1                                         // 000000005944: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[90:91], v[96:97], v[82:89]// 000000005948: cc464052 1d4ac15a
	s_wait_alu depctr_va_sdst(0)                               // 000000005950: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s11, v7, s0                  // 000000005954: d5207c07 00020e0b
	s_wait_loadcnt 0x0                                         // 00000000595c: bfc00000
	v_dual_mul_f32 v90, v107, v102 :: v_dual_mul_f32 v91, v108, v102// 000000005960: c8c6cd6b 5a5acd6c
	v_dual_mul_f32 v99, v109, v102 :: v_dual_mul_f32 v108, v112, v102// 000000005968: c8c6cd6d 636ccd70
	v_mul_f32_e32 v103, v110, v102                             // 000000005970: 10cecd6e
	v_dual_mul_f32 v107, v111, v102 :: v_dual_mul_f32 v98, v98, v102// 000000005974: c8c6cd6f 6b62cd62
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 00000000597c: bf870214
	v_mul_f32_e32 v111, v83, v91                               // 000000005980: 10deb753
	v_mul_f32_e32 v108, v87, v108                              // 000000005984: 10d8d957
	s_delay_alu instid0(valu_dep_4)                            // 000000005988: bf870004
	v_dual_mul_f32 v110, v82, v90 :: v_dual_mul_f32 v103, v85, v103// 00000000598c: c8c6b552 6e66cf55
	v_add_co_u32 v82, vcc_lo, v72, s12                         // 000000005994: d7006a52 02001948
	s_wait_alu depctr_va_vcc(0)                                // 00000000599c: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s13, v73, vcc_lo            // 0000000059a0: d5207c53 01aa920d
	v_dual_mul_f32 v109, v113, v102 :: v_dual_mul_f32 v112, v84, v99// 0000000059a8: c8c6cd71 6d70c754
	v_mul_f32_e32 v113, v89, v98                               // 0000000059b0: 10e2c559
	s_clause 0x1                                               // 0000000059b4: bf850001
	global_load_b64 v[90:91], v[82:83], off                    // 0000000059b8: ee05407c 0000005a 00000052
	global_load_b64 v[98:99], v[82:83], off offset:16          // 0000000059c4: ee05407c 00000062 00001052
	v_mul_f32_e32 v107, v86, v107                              // 0000000059d0: 10d6d756
	v_mul_f32_e32 v109, v88, v109                              // 0000000059d4: 10dadb58
	s_add_nc_u64 s[12:13], s[12:13], 32                        // 0000000059d8: a98ca00c
	v_dual_add_f32 v81, v81, v115 :: v_dual_add_f32 v58, v58, v110// 0000000059dc: c908e751 513add3a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059e4: bf88ff9e
	v_cmp_lt_u64_e64 s0, s[12:13], s[24:25]                    // 0000000059e8: d4590000 0200300c
	v_dual_add_f32 v79, v79, v117 :: v_dual_add_f32 v56, v56, v112// 0000000059f0: c908eb4f 4f38e138
	s_wait_loadcnt 0x1                                         // 0000000059f8: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[90:91], v[104:105], 0// 0000000059fc: cc464052 1a02d15a
	s_wait_loadcnt 0x0                                         // 000000005a04: bfc00000
	s_delay_alu instid0(valu_dep_1)                            // 000000005a08: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[98:99], v[92:93], v[82:89]// 000000005a0c: cc464052 1d4ab962
	v_add_co_u32 v92, vcc_lo, s26, v28                         // 000000005a14: d7006a5c 0202381a
	s_wait_alu depctr_va_vcc(0)                                // 000000005a1c: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v29, vcc_lo            // 000000005a20: d5207c5d 01aa3a1b
	global_load_b32 v104, v[92:93], off                        // 000000005a28: ee05007c 00000068 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v30                         // 000000005a34: d7006a5c 02023c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005a3c: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v31, vcc_lo            // 000000005a40: d5207c5d 01aa3e1b
	global_load_b32 v105, v[92:93], off                        // 000000005a48: ee05007c 00000069 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v36                         // 000000005a54: d7006a5c 0202481a
	s_wait_alu depctr_va_vcc(0)                                // 000000005a5c: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v37, vcc_lo            // 000000005a60: d5207c5d 01aa4a1b
	global_load_b32 v120, v[92:93], off                        // 000000005a68: ee05007c 00000078 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v38                         // 000000005a74: d7006a5c 02024c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005a7c: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v39, vcc_lo            // 000000005a80: d5207c5d 01aa4e1b
	global_load_b32 v121, v[92:93], off                        // 000000005a88: ee05007c 00000079 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v40                         // 000000005a94: d7006a5c 0202501a
	s_wait_alu depctr_va_vcc(0)                                // 000000005a9c: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v41, vcc_lo            // 000000005aa0: d5207c5d 01aa521b
	global_load_b32 v122, v[92:93], off                        // 000000005aa8: ee05007c 0000007a 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v42                         // 000000005ab4: d7006a5c 0202541a
	s_wait_alu depctr_va_vcc(0)                                // 000000005abc: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v43, vcc_lo            // 000000005ac0: d5207c5d 01aa561b
	global_load_b32 v123, v[92:93], off                        // 000000005ac8: ee05007c 0000007b 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v10                         // 000000005ad4: d7006a5c 0202141a
	s_wait_alu depctr_va_vcc(0)                                // 000000005adc: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v11, vcc_lo            // 000000005ae0: d5207c5d 01aa161b
	global_load_b32 v124, v[92:93], off                        // 000000005ae8: ee05007c 0000007c 0000005c
	v_add_co_u32 v92, vcc_lo, s26, v8                          // 000000005af4: d7006a5c 0202101a
	s_wait_alu depctr_va_vcc(0)                                // 000000005afc: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s27, v9, vcc_lo             // 000000005b00: d5207c5d 01aa121b
	v_add_co_u32 v4, vcc_lo, v4, s10                           // 000000005b08: d7006a04 02001504
	s_wait_alu depctr_va_vcc(0)                                // 000000005b10: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s11, v5, vcc_lo              // 000000005b14: d5207c05 01aa0a0b
	global_load_b32 v92, v[92:93], off                         // 000000005b1c: ee05007c 0000005c 0000005c
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000005b28: 8b6a007e
	s_add_nc_u64 s[26:27], s[26:27], 4                         // 000000005b2c: a99a841a
	s_wait_loadcnt 0x5                                         // 000000005b30: bfc00005
	v_dual_mul_f32 v125, v106, v105 :: v_dual_mul_f32 v126, v106, v120// 000000005b34: c8c6d36a 7d7ef16a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000005b3c: bf8701c1
	v_dual_mul_f32 v126, v84, v126 :: v_dual_add_f32 v75, v75, v100// 000000005b40: c8c8fd54 7e4ac94b
	v_add_f32_e32 v54, v54, v107                               // 000000005b48: 066cd736
	s_wait_loadcnt 0x4                                         // 000000005b4c: bfc00004
	v_mul_f32_e32 v127, v106, v121                             // 000000005b50: 10fef36a
	v_dual_add_f32 v53, v53, v108 :: v_dual_add_f32 v66, v66, v126// 000000005b54: c908d935 3542fd42
	s_wait_loadcnt 0x3                                         // 000000005b5c: bfc00003
	v_dual_mul_f32 v128, v106, v122 :: v_dual_add_f32 v71, v71, v119// 000000005b60: c8c8f56a 8046ef47
	v_add_f32_e32 v52, v52, v109                               // 000000005b68: 0668db34
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_1)// 000000005b6c: bf8700b2
	v_mul_f32_e32 v128, v86, v128                              // 000000005b70: 11010156
	s_wait_loadcnt 0x2                                         // 000000005b74: bfc00002
	v_mul_f32_e32 v129, v106, v123                             // 000000005b78: 1102f76a
	v_dual_add_f32 v62, v62, v128 :: v_dual_mul_f32 v129, v87, v129// 000000005b7c: c907013e 3e810357
	s_wait_loadcnt 0x1                                         // 000000005b84: bfc00001
	v_mul_f32_e32 v130, v106, v124                             // 000000005b88: 1104f96a
	v_mul_f32_e32 v93, v106, v104                              // 000000005b8c: 10bad16a
	v_dual_add_f32 v57, v57, v111 :: v_dual_add_f32 v50, v50, v113// 000000005b90: c908df39 3932e332
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_3)// 000000005b98: bf8701c2
	v_dual_mul_f32 v130, v88, v130 :: v_dual_mul_f32 v93, v82, v93// 000000005b9c: c8c70558 825cbb52
	s_wait_loadcnt 0x0                                         // 000000005ba4: bfc00000
	v_dual_mul_f32 v106, v106, v92 :: v_dual_mul_f32 v125, v83, v125// 000000005ba8: c8c6b96a 6a7cfb53
	v_mul_f32_e32 v92, v102, v92                               // 000000005bb0: 10b8b966
	v_add_f32_e32 v60, v60, v130                               // 000000005bb4: 0679053c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000005bb8: bf8701d3
	v_mul_f32_e32 v106, v89, v106                              // 000000005bbc: 10d4d559
	v_mul_f32_e32 v127, v85, v127                              // 000000005bc0: 10feff55
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[90:91], v[94:95], 0// 000000005bc4: cc464052 1a02bd5a
	v_dual_mul_f32 v90, v102, v104 :: v_dual_mul_f32 v91, v102, v105// 000000005bcc: c8c6d166 5a5ad366
	v_mul_f32_e32 v94, v102, v120                              // 000000005bd4: 10bcf166
	v_wmma_f32_16x16x16_fp8_fp8 v[82:89], v[98:99], v[96:97], v[82:89]// 000000005bd8: cc464052 1d4ac162
	v_dual_mul_f32 v95, v102, v121 :: v_dual_mul_f32 v98, v102, v124// 000000005be0: c8c6f366 5f62f966
	v_dual_mul_f32 v96, v102, v122 :: v_dual_mul_f32 v97, v102, v123// 000000005be8: c8c6f566 6060f766
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005bf0: bf870193
	v_dual_mul_f32 v82, v82, v90 :: v_dual_mul_f32 v83, v83, v91// 000000005bf4: c8c6b552 5252b753
	v_dual_mul_f32 v84, v84, v94 :: v_dual_mul_f32 v85, v85, v95// 000000005bfc: c8c6bd54 5454bf55
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000005c04: bf870214
	v_mul_f32_e32 v88, v88, v98                                // 000000005c08: 10b0c558
	v_dual_mul_f32 v86, v86, v96 :: v_dual_mul_f32 v87, v87, v97// 000000005c0c: c8c6c156 5656c357
	v_mul_f32_e32 v89, v89, v92                                // 000000005c14: 10b2b959
	v_dual_add_f32 v55, v55, v103 :: v_dual_add_f32 v70, v70, v93// 000000005c18: c908cf37 3746bb46
	v_add_f32_e32 v67, v67, v125                               // 000000005c20: 0686fb43
	v_add_f32_e32 v63, v63, v127                               // 000000005c24: 067eff3f
	v_dual_add_f32 v61, v61, v129 :: v_dual_add_f32 v48, v48, v84// 000000005c28: c909033d 3d30a930
	v_dual_add_f32 v59, v59, v106 :: v_dual_add_f32 v44, v44, v88// 000000005c30: c908d53b 3b2cb12c
	v_dual_add_f32 v51, v51, v82 :: v_dual_add_f32 v34, v34, v89// 000000005c38: c908a533 3322b322
	v_dual_add_f32 v49, v49, v83 :: v_dual_add_f32 v46, v46, v86// 000000005c40: c908a731 312ead2e
	v_add_f32_e32 v47, v47, v85                                // 000000005c48: 065eab2f
	v_add_f32_e32 v45, v45, v87                                // 000000005c4c: 065aaf2d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c50: bf88ff9e
	s_cbranch_vccnz 65203                                      // 000000005c54: bfa4feb3 <tessera_rocm_scaled_matmul_d3fc025a18c9d32c+0x3c24>
	v_mul_lo_u32 v4, s23, v2                                   // 000000005c58: d72c0004 02020417
	v_mul_lo_u32 v5, s22, v3                                   // 000000005c60: d72c0005 02020616
	v_mad_co_u64_u32 v[2:3], null, s22, v2, 0                  // 000000005c68: d6fe7c02 02020416
	v_bfe_u32 v6, v35, 16, 1                                   // 000000005c70: d6100006 02052123
	v_or_b32_e32 v7, 0x400000, v35                             // 000000005c78: 380e46ff 00400000
	v_bfe_u32 v8, v81, 16, 1                                   // 000000005c80: d6100008 02052151
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 000000005c88: 7c304723
	v_or_b32_e32 v9, 0x400000, v81                             // 000000005c8c: 3812a2ff 00400000
	v_add3_u32 v6, v6, v35, 0x7fff                             // 000000005c94: d6550006 03fe4706 00007fff
	s_lshl_b64 s[0:1], s[22:23], 1                             // 000000005ca0: 84808116
	v_add3_u32 v3, v3, v5, v4                                  // 000000005ca4: d6550003 04120b03
	v_add3_u32 v8, v8, v81, 0x7fff                             // 000000005cac: d6550008 03fea308 00007fff
	v_lshlrev_b64_e32 v[4:5], 1, v[32:33]                      // 000000005cb8: 3e084081
	s_wait_alu depctr_va_vcc(0)                                // 000000005cbc: bf88ff9d
	v_cndmask_b32_e32 v10, v6, v7, vcc_lo                      // 000000005cc0: 02140f06
	v_or_b32_e32 v13, 0x400000, v80                            // 000000005cc4: 381aa0ff 00400000
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000005ccc: 3e040481
	v_bfe_u32 v15, v79, 16, 1                                  // 000000005cd0: d610000f 0205214f
	v_or_b32_e32 v16, 0x400000, v79                            // 000000005cd8: 38209eff 00400000
	v_or_b32_e32 v24, 0x400000, v71                            // 000000005ce0: 38308eff 00400000
	v_or_b32_e32 v19, 0x400000, v75                            // 000000005ce8: 382696ff 00400000
	v_bfe_u32 v25, v67, 16, 1                                  // 000000005cf0: d6100019 02052143
	v_add_co_u32 v6, vcc_lo, s18, v2                           // 000000005cf8: d7006a06 02020412
	s_wait_alu depctr_va_vcc(0)                                // 000000005d00: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s19, v3, vcc_lo              // 000000005d04: d5207c07 01aa0613
	v_cmp_u_f32_e32 vcc_lo, v81, v81                           // 000000005d0c: 7c30a351
	v_add3_u32 v15, v15, v79, 0x7fff                           // 000000005d10: d655000f 03fe9f0f 00007fff
	v_add3_u32 v25, v25, v67, 0x7fff                           // 000000005d1c: d6550019 03fe8719 00007fff
	v_or_b32_e32 v26, 0x400000, v67                            // 000000005d28: 383486ff 00400000
	v_bfe_u32 v21, v74, 16, 1                                  // 000000005d30: d6100015 0205214a
	s_wait_alu depctr_va_vcc(0)                                // 000000005d38: bf88ff9d
	v_cndmask_b32_e32 v11, v8, v9, vcc_lo                      // 000000005d3c: 02161308
	v_add_co_u32 v2, vcc_lo, v6, v4                            // 000000005d40: d7006a02 02020906
	s_wait_alu depctr_va_vcc(0)                                // 000000005d48: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v7, v5, vcc_lo               // 000000005d4c: d5207c03 01aa0b07
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d54: bf88ff9e
	v_add_co_u32 v9, vcc_lo, v6, s0                            // 000000005d58: d7006a09 02000106
	v_bfe_u32 v8, v80, 16, 1                                   // 000000005d60: d6100008 02052150
	s_wait_alu depctr_va_vcc(0)                                // 000000005d68: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v7, vcc_lo              // 000000005d6c: d5207c0c 01aa0e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005d74: bf870193
	v_add_co_u32 v6, vcc_lo, v9, v4                            // 000000005d78: d7006a06 02020909
	v_add3_u32 v8, v8, v80, 0x7fff                             // 000000005d80: d6550008 03fea108 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005d8c: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000005d90: bf870003
	v_add_co_ci_u32_e64 v7, null, v12, v5, vcc_lo              // 000000005d94: d5207c07 01aa0b0c
	v_cmp_u_f32_e32 vcc_lo, v80, v80                           // 000000005d9c: 7c30a150
	v_add3_u32 v21, v21, v74, 0x7fff                           // 000000005da0: d6550015 03fe9515 00007fff
	v_or_b32_e32 v22, 0x400000, v74                            // 000000005dac: 382c94ff 00400000
	v_mul_lo_u32 v23, s22, v1                                  // 000000005db4: d72c0017 02020216
	v_bfe_u32 v31, v62, 16, 1                                  // 000000005dbc: d610001f 0205213e
	s_wait_alu depctr_va_vcc(0)                                // 000000005dc4: bf88ff9d
	v_cndmask_b32_e32 v13, v8, v13, vcc_lo                     // 000000005dc8: 021a1b08
	v_add_co_u32 v14, vcc_lo, v9, s0                           // 000000005dcc: d7006a0e 02000109
	s_wait_alu depctr_va_vcc(0)                                // 000000005dd4: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 000000005dd8: d5207c0c 01aa1801
	v_add3_u32 v31, v31, v62, 0x7fff                           // 000000005de0: d655001f 03fe7d1f 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000005dec: bf8701a3
	v_add_co_u32 v8, vcc_lo, v14, v4                           // 000000005df0: d7006a08 0202090e
	s_wait_alu depctr_va_vcc(0)                                // 000000005df8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v12, v5, vcc_lo              // 000000005dfc: d5207c09 01aa0b0c
	v_cmp_u_f32_e32 vcc_lo, v79, v79                           // 000000005e04: 7c309f4f
	v_or_b32_e32 v32, 0x400000, v62                            // 000000005e08: 38407cff 00400000
	v_or_b32_e32 v29, 0x400000, v63                            // 000000005e10: 383a7eff 00400000
	v_or_b32_e32 v36, 0x400000, v60                            // 000000005e18: 384878ff 00400000
	v_bfe_u32 v38, v59, 16, 1                                  // 000000005e20: d6100026 0205213b
	s_wait_alu depctr_va_vcc(0)                                // 000000005e28: bf88ff9d
	v_cndmask_b32_e32 v16, v15, v16, vcc_lo                    // 000000005e2c: 0220210f
	s_clause 0x2                                               // 000000005e30: bf850002
	global_store_d16_hi_b16 v[2:3], v10, off                   // 000000005e34: ee09407c 05000000 00000002
	global_store_d16_hi_b16 v[6:7], v11, off                   // 000000005e40: ee09407c 05800000 00000006
	global_store_d16_hi_b16 v[8:9], v13, off                   // 000000005e4c: ee09407c 06800000 00000008
	v_bfe_u32 v10, v78, 16, 1                                  // 000000005e58: d610000a 0205214e
	v_add_co_u32 v13, vcc_lo, v14, s0                          // 000000005e60: d7006a0d 0200010e
	s_wait_alu depctr_va_vcc(0)                                // 000000005e68: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s1, v12, vcc_lo             // 000000005e6c: d5207c0c 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005e74: bf870193
	v_add3_u32 v14, v10, v78, 0x7fff                           // 000000005e78: d655000e 03fe9d0a 00007fff
	v_add_co_u32 v10, vcc_lo, v13, v4                          // 000000005e84: d7006a0a 0202090d
	v_or_b32_e32 v15, 0x400000, v78                            // 000000005e8c: 381e9cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005e94: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v12, v5, vcc_lo             // 000000005e98: d5207c0b 01aa0b0c
	v_cmp_u_f32_e32 vcc_lo, v78, v78                           // 000000005ea0: 7c309d4e
	v_add3_u32 v38, v38, v59, 0x7fff                           // 000000005ea4: d6550026 03fe7726 00007fff
	v_or_b32_e32 v39, 0x400000, v59                            // 000000005eb0: 384e76ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005eb8: bf88ff9d
	v_cndmask_b32_e32 v17, v14, v15, vcc_lo                    // 000000005ebc: 02221f0e
	v_add_co_u32 v15, vcc_lo, v13, s0                          // 000000005ec0: d7006a0f 0200010d
	v_bfe_u32 v14, v75, 16, 1                                  // 000000005ec8: d610000e 0205214b
	s_wait_alu depctr_va_vcc(0)                                // 000000005ed0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v12, vcc_lo             // 000000005ed4: d5207c12 01aa1801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005edc: bf870193
	v_add_co_u32 v12, vcc_lo, v15, v4                          // 000000005ee0: d7006a0c 0202090f
	v_add3_u32 v14, v14, v75, 0x7fff                           // 000000005ee8: d655000e 03fe970e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005ef4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000005ef8: bf870003
	v_add_co_ci_u32_e64 v13, null, v18, v5, vcc_lo             // 000000005efc: d5207c0d 01aa0b12
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 000000005f04: 7c30974b
	s_wait_alu depctr_va_vcc(0)                                // 000000005f08: bf88ff9d
	v_cndmask_b32_e32 v19, v14, v19, vcc_lo                    // 000000005f0c: 0226270e
	v_add_co_u32 v20, vcc_lo, v15, s0                          // 000000005f10: d7006a14 0200010f
	s_wait_alu depctr_va_vcc(0)                                // 000000005f18: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000005f1c: d5207c12 01aa2401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000005f24: bf870122
	v_add_co_u32 v14, vcc_lo, v20, v4                          // 000000005f28: d7006a0e 02020914
	s_wait_alu depctr_va_vcc(0)                                // 000000005f30: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v18, v5, vcc_lo             // 000000005f34: d5207c0f 01aa0b12
	v_cmp_u_f32_e32 vcc_lo, v74, v74                           // 000000005f3c: 7c30954a
	s_clause 0x2                                               // 000000005f40: bf850002
	global_store_d16_hi_b16 v[10:11], v16, off                 // 000000005f44: ee09407c 08000000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 000000005f50: ee09407c 08800000 0000000c
	global_store_d16_hi_b16 v[14:15], v19, off                 // 000000005f5c: ee09407c 09800000 0000000e
	v_bfe_u32 v16, v71, 16, 1                                  // 000000005f68: d6100010 02052147
	s_wait_alu depctr_va_vcc(0)                                // 000000005f70: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v22, vcc_lo                    // 000000005f74: 022a2d15
	v_add_co_u32 v19, vcc_lo, v20, s0                          // 000000005f78: d7006a13 02000114
	s_wait_alu depctr_va_vcc(0)                                // 000000005f80: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v18, vcc_lo             // 000000005f84: d5207c12 01aa2401
	v_add3_u32 v20, v16, v71, 0x7fff                           // 000000005f8c: d6550014 03fe8f10 00007fff
	v_mul_lo_u32 v22, s23, v0                                  // 000000005f98: d72c0016 02020017
	v_mad_co_u64_u32 v[0:1], null, s22, v0, 0                  // 000000005fa0: d6fe7c00 02020016
	v_add_co_u32 v16, vcc_lo, v19, v4                          // 000000005fa8: d7006a10 02020913
	s_wait_alu depctr_va_vcc(0)                                // 000000005fb0: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v18, v5, vcc_lo             // 000000005fb4: d5207c11 01aa0b12
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000005fbc: 7c308f47
	s_delay_alu instid0(valu_dep_4)                            // 000000005fc0: bf870004
	v_add3_u32 v1, v1, v23, v22                                // 000000005fc4: d6550001 045a2f01
	v_bfe_u32 v22, v70, 16, 1                                  // 000000005fcc: d6100016 02052146
	s_wait_alu depctr_va_vcc(0)                                // 000000005fd4: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v24, vcc_lo                    // 000000005fd8: 02283114
	v_add_co_u32 v19, vcc_lo, v19, s0                          // 000000005fdc: d7006a13 02000113
	s_wait_alu depctr_va_vcc(0)                                // 000000005fe4: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s1, v18, vcc_lo             // 000000005fe8: d5207c17 01aa2401
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000005ff0: 3e000081
	s_delay_alu instid0(valu_dep_3)                            // 000000005ff4: bf870003
	v_add_co_u32 v18, vcc_lo, v19, v4                          // 000000005ff8: d7006a12 02020913
	v_add3_u32 v22, v22, v70, 0x7fff                           // 000000006000: d6550016 03fe8d16 00007fff
	v_or_b32_e32 v24, 0x400000, v70                            // 00000000600c: 38308cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006014: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v23, v5, vcc_lo             // 000000006018: d5207c13 01aa0b17
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000006020: 7c308d46
	s_wait_alu depctr_va_vcc(0)                                // 000000006024: bf88ff9d
	v_cndmask_b32_e32 v22, v22, v24, vcc_lo                    // 000000006028: 022c3116
	v_add_co_u32 v23, vcc_lo, s18, v0                          // 00000000602c: d7006a17 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000006034: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s19, v1, vcc_lo             // 000000006038: d5207c18 01aa0213
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000006040: bf870122
	v_add_co_u32 v0, vcc_lo, v23, v4                           // 000000006044: d7006a00 02020917
	s_wait_alu depctr_va_vcc(0)                                // 00000000604c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v24, v5, vcc_lo              // 000000006050: d5207c01 01aa0b18
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 000000006058: 7c308743
	s_clause 0x2                                               // 00000000605c: bf850002
	global_store_d16_hi_b16 v[16:17], v21, off                 // 000000006060: ee09407c 0a800000 00000010
	global_store_d16_hi_b16 v[18:19], v20, off                 // 00000000606c: ee09407c 0a000000 00000012
	global_store_d16_hi_b16 v[0:1], v22, off                   // 000000006078: ee09407c 0b000000 00000000
	v_bfe_u32 v20, v66, 16, 1                                  // 000000006084: d6100014 02052142
	s_wait_alu depctr_va_vcc(0)                                // 00000000608c: bf88ff9d
	v_cndmask_b32_e32 v26, v25, v26, vcc_lo                    // 000000006090: 02343519
	v_add_co_u32 v22, vcc_lo, v23, s0                          // 000000006094: d7006a16 02000117
	s_wait_alu depctr_va_vcc(0)                                // 00000000609c: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s1, v24, vcc_lo             // 0000000060a0: d5207c17 01aa3001
	v_add3_u32 v24, v20, v66, 0x7fff                           // 0000000060a8: d6550018 03fe8514 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000060b4: bf870003
	v_add_co_u32 v20, vcc_lo, v22, v4                          // 0000000060b8: d7006a14 02020916
	v_or_b32_e32 v25, 0x400000, v66                            // 0000000060c0: 383284ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000060c8: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v23, v5, vcc_lo             // 0000000060cc: d5207c15 01aa0b17
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 0000000060d4: 7c308542
	s_wait_alu depctr_va_vcc(0)                                // 0000000060d8: bf88ff9d
	v_cndmask_b32_e32 v27, v24, v25, vcc_lo                    // 0000000060dc: 02363318
	v_add_co_u32 v25, vcc_lo, v22, s0                          // 0000000060e0: d7006a19 02000116
	v_bfe_u32 v24, v63, 16, 1                                  // 0000000060e8: d6100018 0205213f
	s_wait_alu depctr_va_vcc(0)                                // 0000000060f0: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v23, vcc_lo             // 0000000060f4: d5207c1c 01aa2e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000060fc: bf870193
	v_add_co_u32 v22, vcc_lo, v25, v4                          // 000000006100: d7006a16 02020919
	v_add3_u32 v24, v24, v63, 0x7fff                           // 000000006108: d6550018 03fe7f18 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006114: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000006118: bf870003
	v_add_co_ci_u32_e64 v23, null, v28, v5, vcc_lo             // 00000000611c: d5207c17 01aa0b1c
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000006124: 7c307f3f
	s_wait_alu depctr_va_vcc(0)                                // 000000006128: bf88ff9d
	v_cndmask_b32_e32 v29, v24, v29, vcc_lo                    // 00000000612c: 023a3b18
	v_add_co_u32 v30, vcc_lo, v25, s0                          // 000000006130: d7006a1e 02000119
	s_wait_alu depctr_va_vcc(0)                                // 000000006138: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 00000000613c: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000006144: bf870122
	v_add_co_u32 v24, vcc_lo, v30, v4                          // 000000006148: d7006a18 0202091e
	s_wait_alu depctr_va_vcc(0)                                // 000000006150: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v28, v5, vcc_lo             // 000000006154: d5207c19 01aa0b1c
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 00000000615c: 7c307d3e
	s_wait_alu depctr_va_vcc(0)                                // 000000006160: bf88ff9d
	v_cndmask_b32_e32 v32, v31, v32, vcc_lo                    // 000000006164: 0240411f
	s_clause 0x2                                               // 000000006168: bf850002
	global_store_d16_hi_b16 v[20:21], v26, off                 // 00000000616c: ee09407c 0d000000 00000014
	global_store_d16_hi_b16 v[22:23], v27, off                 // 000000006178: ee09407c 0d800000 00000016
	global_store_d16_hi_b16 v[24:25], v29, off                 // 000000006184: ee09407c 0e800000 00000018
	v_bfe_u32 v26, v61, 16, 1                                  // 000000006190: d610001a 0205213d
	v_add_co_u32 v29, vcc_lo, v30, s0                          // 000000006198: d7006a1d 0200011e
	s_wait_alu depctr_va_vcc(0)                                // 0000000061a0: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 0000000061a4: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000061ac: bf870193
	v_add3_u32 v30, v26, v61, 0x7fff                           // 0000000061b0: d655001e 03fe7b1a 00007fff
	v_add_co_u32 v26, vcc_lo, v29, v4                          // 0000000061bc: d7006a1a 0202091d
	v_or_b32_e32 v31, 0x400000, v61                            // 0000000061c4: 383e7aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000061cc: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v28, v5, vcc_lo             // 0000000061d0: d5207c1b 01aa0b1c
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 0000000061d8: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 0000000061dc: bf88ff9d
	v_cndmask_b32_e32 v33, v30, v31, vcc_lo                    // 0000000061e0: 02423f1e
	v_add_co_u32 v31, vcc_lo, v29, s0                          // 0000000061e4: d7006a1f 0200011d
	v_bfe_u32 v30, v60, 16, 1                                  // 0000000061ec: d610001e 0205213c
	s_wait_alu depctr_va_vcc(0)                                // 0000000061f4: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s1, v28, vcc_lo             // 0000000061f8: d5207c23 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006200: bf870193
	v_add_co_u32 v28, vcc_lo, v31, v4                          // 000000006204: d7006a1c 0202091f
	v_add3_u32 v30, v30, v60, 0x7fff                           // 00000000620c: d655001e 03fe791e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006218: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000621c: bf870003
	v_add_co_ci_u32_e64 v29, null, v35, v5, vcc_lo             // 000000006220: d5207c1d 01aa0b23
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000006228: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 00000000622c: bf88ff9d
	v_cndmask_b32_e32 v36, v30, v36, vcc_lo                    // 000000006230: 0248491e
	v_add_co_u32 v37, vcc_lo, v31, s0                          // 000000006234: d7006a25 0200011f
	s_wait_alu depctr_va_vcc(0)                                // 00000000623c: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s1, v35, vcc_lo             // 000000006240: d5207c23 01aa4601
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000006248: bf870122
	v_add_co_u32 v30, vcc_lo, v37, v4                          // 00000000624c: d7006a1e 02020925
	s_wait_alu depctr_va_vcc(0)                                // 000000006254: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, v35, v5, vcc_lo             // 000000006258: d5207c1f 01aa0b23
	s_clause 0x2                                               // 000000006260: bf850002
	global_store_d16_hi_b16 v[26:27], v32, off                 // 000000006264: ee09407c 10000000 0000001a
	global_store_d16_hi_b16 v[28:29], v33, off                 // 000000006270: ee09407c 10800000 0000001c
	global_store_d16_hi_b16 v[30:31], v36, off                 // 00000000627c: ee09407c 12000000 0000001e
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000006288: 7c30773b
	v_bfe_u32 v33, v58, 16, 1                                  // 00000000628c: d6100021 0205213a
	s_delay_alu instid0(valu_dep_1)                            // 000000006294: bf870001
	v_add3_u32 v33, v33, v58, 0x7fff                           // 000000006298: d6550021 03fe7521 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000062a4: bf88ff9d
	v_cndmask_b32_e32 v32, v38, v39, vcc_lo                    // 0000000062a8: 02404f26
	v_add_co_u32 v36, vcc_lo, v37, s0                          // 0000000062ac: d7006a24 02000125
	s_wait_alu depctr_va_vcc(0)                                // 0000000062b4: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s1, v35, vcc_lo             // 0000000062b8: d5207c23 01aa4601
	v_or_b32_e32 v37, 0x400000, v58                            // 0000000062c0: 384a74ff 00400000
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000062c8: bf8701a3
	v_add_co_u32 v4, vcc_lo, v36, v4                           // 0000000062cc: d7006a04 02020924
	s_wait_alu depctr_va_vcc(0)                                // 0000000062d4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v35, v5, vcc_lo              // 0000000062d8: d5207c05 01aa0b23
	v_bfe_u32 v35, v57, 16, 1                                  // 0000000062e0: d6100023 02052139
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 0000000062e8: 7c30753a
	v_bfe_u32 v36, v56, 16, 1                                  // 0000000062ec: d6100024 02052138
	s_wait_alu depctr_va_vcc(0)                                // 0000000062f4: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v37, vcc_lo                    // 0000000062f8: 02424b21
	global_store_d16_hi_b16 v[4:5], v32, off                   // 0000000062fc: ee09407c 10000000 00000004
	v_add3_u32 v32, v35, v57, 0x7fff                           // 000000006308: d6550020 03fe7323 00007fff
	v_or_b32_e32 v35, 0x400000, v57                            // 000000006314: 384672ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 00000000631c: 7c307339
	global_store_d16_hi_b16 v[2:3], v33, off offset:32         // 000000006320: ee09407c 10800000 00002002
	v_add3_u32 v2, v36, v56, 0x7fff                            // 00000000632c: d6550002 03fe7124 00007fff
	v_or_b32_e32 v3, 0x400000, v56                             // 000000006338: 380670ff 00400000
	v_bfe_u32 v33, v55, 16, 1                                  // 000000006340: d6100021 02052137
	s_wait_alu depctr_va_vcc(0)                                // 000000006348: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v35, vcc_lo                    // 00000000634c: 02404720
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 000000006350: 7c307138
	global_store_d16_hi_b16 v[6:7], v32, off offset:32         // 000000006354: ee09407c 10000000 00002006
	s_wait_alu depctr_va_vcc(0)                                // 000000006360: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000006364: 02040702
	v_bfe_u32 v3, v54, 16, 1                                   // 000000006368: d6100003 02052136
	v_add3_u32 v6, v33, v55, 0x7fff                            // 000000006370: d6550006 03fe6f21 00007fff
	v_or_b32_e32 v7, 0x400000, v55                             // 00000000637c: 380e6eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000006384: 7c306f37
	global_store_d16_hi_b16 v[8:9], v2, off offset:32          // 000000006388: ee09407c 01000000 00002008
	v_add3_u32 v2, v3, v54, 0x7fff                             // 000000006394: d6550002 03fe6d03 00007fff
	v_or_b32_e32 v3, 0x400000, v54                             // 0000000063a0: 38066cff 00400000
	v_or_b32_e32 v8, 0x400000, v44                             // 0000000063a8: 381058ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000063b0: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 0000000063b4: 020c0f06
	v_bfe_u32 v7, v53, 16, 1                                   // 0000000063b8: d6100007 02052135
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 0000000063c0: 7c306d36
	v_or_b32_e32 v9, 0x400000, v34                             // 0000000063c4: 381244ff 00400000
	global_store_d16_hi_b16 v[10:11], v6, off offset:32        // 0000000063cc: ee09407c 03000000 0000200a
	v_add3_u32 v6, v7, v53, 0x7fff                             // 0000000063d8: d6550006 03fe6b07 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000063e4: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 0000000063e8: 02040702
	v_bfe_u32 v3, v52, 16, 1                                   // 0000000063ec: d6100003 02052134
	v_or_b32_e32 v7, 0x400000, v53                             // 0000000063f4: 380e6aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 0000000063fc: 7c306b35
	global_store_d16_hi_b16 v[12:13], v2, off offset:32        // 000000006400: ee09407c 01000000 0000200c
	v_add3_u32 v2, v3, v52, 0x7fff                             // 00000000640c: d6550002 03fe6903 00007fff
	v_or_b32_e32 v3, 0x400000, v52                             // 000000006418: 380668ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006420: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000006424: 020c0f06
	v_bfe_u32 v7, v50, 16, 1                                   // 000000006428: d6100007 02052132
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000006430: 7c306934
	global_store_d16_hi_b16 v[14:15], v6, off offset:32        // 000000006434: ee09407c 03000000 0000200e
	v_add3_u32 v6, v7, v50, 0x7fff                             // 000000006440: d6550006 03fe6507 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000644c: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000006450: 02040702
	v_bfe_u32 v3, v51, 16, 1                                   // 000000006454: d6100003 02052133
	v_or_b32_e32 v7, 0x400000, v50                             // 00000000645c: 380e64ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000006464: 7c306532
	global_store_d16_hi_b16 v[16:17], v2, off offset:32        // 000000006468: ee09407c 01000000 00002010
	v_add3_u32 v2, v3, v51, 0x7fff                             // 000000006474: d6550002 03fe6703 00007fff
	v_or_b32_e32 v3, 0x400000, v51                             // 000000006480: 380666ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006488: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 00000000648c: 020c0f06
	v_bfe_u32 v7, v49, 16, 1                                   // 000000006490: d6100007 02052131
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 000000006498: 7c306733
	global_store_d16_hi_b16 v[18:19], v6, off offset:32        // 00000000649c: ee09407c 03000000 00002012
	v_add3_u32 v6, v7, v49, 0x7fff                             // 0000000064a8: d6550006 03fe6307 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000064b4: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 0000000064b8: 02040702
	v_bfe_u32 v3, v48, 16, 1                                   // 0000000064bc: d6100003 02052130
	v_or_b32_e32 v7, 0x400000, v49                             // 0000000064c4: 380e62ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 0000000064cc: 7c306331
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000064d0: ee09407c 01000000 00002000
	v_add3_u32 v0, v3, v48, 0x7fff                             // 0000000064dc: d6550000 03fe6103 00007fff
	v_or_b32_e32 v1, 0x400000, v48                             // 0000000064e8: 380260ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000064f0: bf88ff9d
	v_cndmask_b32_e32 v2, v6, v7, vcc_lo                       // 0000000064f4: 02040f06
	v_bfe_u32 v3, v47, 16, 1                                   // 0000000064f8: d6100003 0205212f
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000006500: 7c306130
	v_bfe_u32 v6, v44, 16, 1                                   // 000000006504: d6100006 0205212c
	v_or_b32_e32 v7, 0x400000, v45                             // 00000000650c: 380e5aff 00400000
	global_store_d16_hi_b16 v[20:21], v2, off offset:32        // 000000006514: ee09407c 01000000 00002014
	v_add3_u32 v2, v3, v47, 0x7fff                             // 000000006520: d6550002 03fe5f03 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000652c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006530: 02000300
	v_bfe_u32 v1, v46, 16, 1                                   // 000000006534: d6100001 0205212e
	v_or_b32_e32 v3, 0x400000, v47                             // 00000000653c: 38065eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000006544: 7c305f2f
	v_add3_u32 v6, v6, v44, 0x7fff                             // 000000006548: d6550006 03fe5906 00007fff
	global_store_d16_hi_b16 v[22:23], v0, off offset:32        // 000000006554: ee09407c 00000000 00002016
	v_add3_u32 v0, v1, v46, 0x7fff                             // 000000006560: d6550000 03fe5d01 00007fff
	v_or_b32_e32 v1, 0x400000, v46                             // 00000000656c: 38025cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006574: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000006578: 02040702
	v_bfe_u32 v3, v45, 16, 1                                   // 00000000657c: d6100003 0205212d
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000006584: 7c305d2e
	s_delay_alu instid0(valu_dep_2)                            // 000000006588: bf870002
	v_add3_u32 v3, v3, v45, 0x7fff                             // 00000000658c: d6550003 03fe5b03 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006598: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000659c: 02000300
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 0000000065a0: 7c305b2d
	v_bfe_u32 v1, v34, 16, 1                                   // 0000000065a4: d6100001 02052122
	s_wait_alu depctr_va_vcc(0)                                // 0000000065ac: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo                       // 0000000065b0: 02060f03
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 0000000065b4: 7c30592c
	s_delay_alu instid0(valu_dep_3)                            // 0000000065b8: bf870003
	v_add3_u32 v1, v1, v34, 0x7fff                             // 0000000065bc: d6550001 03fe4501 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000065c8: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo                       // 0000000065cc: 020c1106
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 0000000065d0: 7c304522
	s_wait_alu depctr_va_vcc(0)                                // 0000000065d4: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo                       // 0000000065d8: 02021301
	s_clause 0x3                                               // 0000000065dc: bf850003
	global_store_d16_hi_b16 v[24:25], v2, off offset:32        // 0000000065e0: ee09407c 01000000 00002018
	global_store_d16_hi_b16 v[26:27], v0, off offset:32        // 0000000065ec: ee09407c 00000000 0000201a
	global_store_d16_hi_b16 v[28:29], v3, off offset:32        // 0000000065f8: ee09407c 01800000 0000201c
	global_store_d16_hi_b16 v[30:31], v6, off offset:32        // 000000006604: ee09407c 03000000 0000201e
	global_store_d16_hi_b16 v[4:5], v1, off offset:32          // 000000006610: ee09407c 00800000 00002004
	s_nop 0                                                    // 00000000661c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000006620: bfb60003
	s_endpgm                                                   // 000000006624: bfb00000
	s_code_end                                                 // 000000006628: bf9f0000
	s_code_end                                                 // 00000000662c: bf9f0000
	s_code_end                                                 // 000000006630: bf9f0000
	s_code_end                                                 // 000000006634: bf9f0000
	s_code_end                                                 // 000000006638: bf9f0000
	s_code_end                                                 // 00000000663c: bf9f0000
	s_code_end                                                 // 000000006640: bf9f0000
	s_code_end                                                 // 000000006644: bf9f0000
	s_code_end                                                 // 000000006648: bf9f0000
	s_code_end                                                 // 00000000664c: bf9f0000
	s_code_end                                                 // 000000006650: bf9f0000
	s_code_end                                                 // 000000006654: bf9f0000
	s_code_end                                                 // 000000006658: bf9f0000
	s_code_end                                                 // 00000000665c: bf9f0000
	s_code_end                                                 // 000000006660: bf9f0000
	s_code_end                                                 // 000000006664: bf9f0000
	s_code_end                                                 // 000000006668: bf9f0000
	s_code_end                                                 // 00000000666c: bf9f0000
	s_code_end                                                 // 000000006670: bf9f0000
	s_code_end                                                 // 000000006674: bf9f0000
	s_code_end                                                 // 000000006678: bf9f0000
	s_code_end                                                 // 00000000667c: bf9f0000
	s_code_end                                                 // 000000006680: bf9f0000
	s_code_end                                                 // 000000006684: bf9f0000
	s_code_end                                                 // 000000006688: bf9f0000
	s_code_end                                                 // 00000000668c: bf9f0000
	s_code_end                                                 // 000000006690: bf9f0000
	s_code_end                                                 // 000000006694: bf9f0000
	s_code_end                                                 // 000000006698: bf9f0000
	s_code_end                                                 // 00000000669c: bf9f0000
	s_code_end                                                 // 0000000066a0: bf9f0000
	s_code_end                                                 // 0000000066a4: bf9f0000
	s_code_end                                                 // 0000000066a8: bf9f0000
	s_code_end                                                 // 0000000066ac: bf9f0000
	s_code_end                                                 // 0000000066b0: bf9f0000
	s_code_end                                                 // 0000000066b4: bf9f0000
	s_code_end                                                 // 0000000066b8: bf9f0000
	s_code_end                                                 // 0000000066bc: bf9f0000
	s_code_end                                                 // 0000000066c0: bf9f0000
	s_code_end                                                 // 0000000066c4: bf9f0000
	s_code_end                                                 // 0000000066c8: bf9f0000
	s_code_end                                                 // 0000000066cc: bf9f0000
	s_code_end                                                 // 0000000066d0: bf9f0000
	s_code_end                                                 // 0000000066d4: bf9f0000
	s_code_end                                                 // 0000000066d8: bf9f0000
	s_code_end                                                 // 0000000066dc: bf9f0000
	s_code_end                                                 // 0000000066e0: bf9f0000
	s_code_end                                                 // 0000000066e4: bf9f0000
	s_code_end                                                 // 0000000066e8: bf9f0000
	s_code_end                                                 // 0000000066ec: bf9f0000
	s_code_end                                                 // 0000000066f0: bf9f0000
	s_code_end                                                 // 0000000066f4: bf9f0000
	s_code_end                                                 // 0000000066f8: bf9f0000
	s_code_end                                                 // 0000000066fc: bf9f0000
	s_code_end                                                 // 000000006700: bf9f0000
	s_code_end                                                 // 000000006704: bf9f0000
	s_code_end                                                 // 000000006708: bf9f0000
	s_code_end                                                 // 00000000670c: bf9f0000
	s_code_end                                                 // 000000006710: bf9f0000
	s_code_end                                                 // 000000006714: bf9f0000
	s_code_end                                                 // 000000006718: bf9f0000
	s_code_end                                                 // 00000000671c: bf9f0000
	s_code_end                                                 // 000000006720: bf9f0000
	s_code_end                                                 // 000000006724: bf9f0000
	s_code_end                                                 // 000000006728: bf9f0000
	s_code_end                                                 // 00000000672c: bf9f0000
	s_code_end                                                 // 000000006730: bf9f0000
	s_code_end                                                 // 000000006734: bf9f0000
	s_code_end                                                 // 000000006738: bf9f0000
	s_code_end                                                 // 00000000673c: bf9f0000
	s_code_end                                                 // 000000006740: bf9f0000
	s_code_end                                                 // 000000006744: bf9f0000
	s_code_end                                                 // 000000006748: bf9f0000
	s_code_end                                                 // 00000000674c: bf9f0000
	s_code_end                                                 // 000000006750: bf9f0000
	s_code_end                                                 // 000000006754: bf9f0000
	s_code_end                                                 // 000000006758: bf9f0000
	s_code_end                                                 // 00000000675c: bf9f0000
	s_code_end                                                 // 000000006760: bf9f0000
	s_code_end                                                 // 000000006764: bf9f0000
	s_code_end                                                 // 000000006768: bf9f0000
	s_code_end                                                 // 00000000676c: bf9f0000
	s_code_end                                                 // 000000006770: bf9f0000
	s_code_end                                                 // 000000006774: bf9f0000
	s_code_end                                                 // 000000006778: bf9f0000
	s_code_end                                                 // 00000000677c: bf9f0000
	s_code_end                                                 // 000000006780: bf9f0000
	s_code_end                                                 // 000000006784: bf9f0000
	s_code_end                                                 // 000000006788: bf9f0000
	s_code_end                                                 // 00000000678c: bf9f0000
	s_code_end                                                 // 000000006790: bf9f0000
	s_code_end                                                 // 000000006794: bf9f0000
	s_code_end                                                 // 000000006798: bf9f0000
	s_code_end                                                 // 00000000679c: bf9f0000
	s_code_end                                                 // 0000000067a0: bf9f0000
	s_code_end                                                 // 0000000067a4: bf9f0000
	s_code_end                                                 // 0000000067a8: bf9f0000
	s_code_end                                                 // 0000000067ac: bf9f0000
	s_code_end                                                 // 0000000067b0: bf9f0000
	s_code_end                                                 // 0000000067b4: bf9f0000
	s_code_end                                                 // 0000000067b8: bf9f0000
	s_code_end                                                 // 0000000067bc: bf9f0000
	s_code_end                                                 // 0000000067c0: bf9f0000
	s_code_end                                                 // 0000000067c4: bf9f0000
	s_code_end                                                 // 0000000067c8: bf9f0000
	s_code_end                                                 // 0000000067cc: bf9f0000
	s_code_end                                                 // 0000000067d0: bf9f0000
	s_code_end                                                 // 0000000067d4: bf9f0000
	s_code_end                                                 // 0000000067d8: bf9f0000
	s_code_end                                                 // 0000000067dc: bf9f0000
	s_code_end                                                 // 0000000067e0: bf9f0000
	s_code_end                                                 // 0000000067e4: bf9f0000
	s_code_end                                                 // 0000000067e8: bf9f0000
	s_code_end                                                 // 0000000067ec: bf9f0000
	s_code_end                                                 // 0000000067f0: bf9f0000
	s_code_end                                                 // 0000000067f4: bf9f0000
	s_code_end                                                 // 0000000067f8: bf9f0000
	s_code_end                                                 // 0000000067fc: bf9f0000
