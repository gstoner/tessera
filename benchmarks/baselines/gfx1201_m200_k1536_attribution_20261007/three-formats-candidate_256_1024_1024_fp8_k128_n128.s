
/tmp/tmplqj9n_id.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_28d379a9237322d1>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[20:23], s[0:1], 0xc8                         // 000000001b04: f4004500 f80000c8
	s_load_b64 s[18:19], s[0:1], 0xa8                          // 000000001b0c: f4002480 f80000a8
	s_load_b64 s[24:25], s[0:1], 0xd8                          // 000000001b14: f4002600 f80000d8
	s_load_b64 s[30:31], s[0:1], 0x8                           // 000000001b1c: f4002780 f8000008
	s_load_b64 s[34:35], s[0:1], 0x30                          // 000000001b24: f4002880 f8000030
	s_load_b64 s[26:27], s[0:1], 0x58                          // 000000001b2c: f4002680 f8000058
	s_load_b64 s[28:29], s[0:1], 0x80                          // 000000001b34: f4002700 f8000080
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v64, 15, v0        // 000000001b3c: ca240080 2740008f
	s_mov_b32 s2, ttmp9                                        // 000000001b44: be820075
	s_mov_b32 s4, ttmp7                                        // 000000001b48: be840073
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b4c: 86039f75
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b50: 86059f73
	s_lshl_b64 s[40:41], s[2:3], 5                             // 000000001b54: 84a88502
	s_lshl_b64 s[36:37], s[4:5], 5                             // 000000001b58: 84a48504
	s_add_nc_u64 s[2:3], s[40:41], 32                          // 000000001b5c: a982a028
	s_add_nc_u64 s[0:1], s[36:37], 32                          // 000000001b60: a980a024
	v_or_b32_e32 v34, s40, v64                                 // 000000001b64: 38448028
	v_lshrrev_b32_e32 v0, 1, v0                                // 000000001b68: 32000081
	v_mov_b32_e32 v35, s41                                     // 000000001b6c: 7e460229
	v_mov_b32_e32 v41, s41                                     // 000000001b70: 7e520229
	s_or_b32 s33, s36, 16                                      // 000000001b74: 8c219024
	v_or_b32_e32 v40, 16, v34                                  // 000000001b78: 38504490
	s_wait_kmcnt 0x0                                           // 000000001b7c: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[20:21]                      // 000000001b80: d4540000 02002800
	v_cmp_gt_i64_e64 s1, s[2:3], s[22:23]                      // 000000001b88: d4540001 02002c02
	v_and_b32_e32 v38, 8, v0                                   // 000000001b90: 364c0088
	s_lshr_b64 s[38:39], s[24:25], 7                           // 000000001b94: 85a68718
	s_mov_b32 s2, -1                                           // 000000001b98: be8200c1
	s_add_nc_u64 s[42:43], s[22:23], 0x7f                      // 000000001b9c: a9aaff16 0000007f
	s_or_b32 s0, s0, s1                                        // 000000001ba4: 8c000100
	v_cmp_gt_i64_e64 s1, s[22:23], v[34:35]                    // 000000001ba8: d4540001 02024416
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb0: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb4: 8b6a007e
	v_cmp_gt_i64_e64 s0, s[22:23], v[40:41]                    // 000000001bb8: d4540000 02025016
	v_or_b32_e32 v36, s36, v38                                 // 000000001bc0: 38484c24
	v_or_b32_e32 v54, 1, v38                                   // 000000001bc4: 386c4c81
	v_or_b32_e32 v53, 2, v38                                   // 000000001bc8: 386a4c82
	v_or_b32_e32 v52, 3, v38                                   // 000000001bcc: 38684c83
	v_or_b32_e32 v49, 4, v38                                   // 000000001bd0: 38624c84
	v_or_b32_e32 v48, 5, v38                                   // 000000001bd4: 38604c85
	v_or_b32_e32 v50, 6, v38                                   // 000000001bd8: 38644c86
	v_or_b32_e32 v51, 7, v38                                   // 000000001bdc: 38664c87
	v_or_b32_e32 v32, s33, v38                                 // 000000001be0: 38404c21
	s_cbranch_vccnz 5                                          // 000000001be4: bfa40005 <tessera_rocm_scaled_matmul_28d379a9237322d1+0xfc>
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000001be8: 8b6a027e
	s_cbranch_vccnz 3411                                       // 000000001bec: bfa40d53 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x363c>
	s_nop 0                                                    // 000000001bf0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000001bf4: bfb60003
	s_endpgm                                                   // 000000001bf8: bfb00000
	v_or_b32_e32 v0, s36, v64                                  // 000000001bfc: 38008024
	v_mov_b32_e32 v37, s37                                     // 000000001c00: 7e4a0225
	v_or_b32_e32 v2, s33, v64                                  // 000000001c04: 38048021
	s_mul_i32 s3, s24, s37                                     // 000000001c08: 96032518
	v_mov_b32_e32 v3, s37                                      // 000000001c0c: 7e060225
	v_mul_lo_u32 v8, s25, v0                                   // 000000001c10: d72c0008 02020019
	v_mad_co_u64_u32 v[4:5], null, s24, v0, 0                  // 000000001c18: d6fe7c04 02020018
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[36:37]                // 000000001c20: 7ca84814
	v_mov_b32_e32 v1, s37                                      // 000000001c24: 7e020225
	v_mul_lo_u32 v9, s25, v2                                   // 000000001c28: d72c0009 02020419
	v_mad_co_u64_u32 v[6:7], null, s24, v2, 0                  // 000000001c30: d6fe7c06 02020418
	s_lshr_b64 s[58:59], s[42:43], 7                           // 000000001c38: 85ba872a
	s_lshr_b64 s[8:9], s[40:41], 7                             // 000000001c3c: 85888728
	v_cmp_gt_i64_e64 s2, s[20:21], v[0:1]                      // 000000001c40: d4540002 02020014
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c48: bf88ff9e
	v_add3_u32 v74, v5, s3, v8                                 // 000000001c4c: d655004a 04200705
	v_or_b32_e32 v76, v4, v38                                  // 000000001c54: 38984d04
	v_mul_lo_u32 v4, s25, v34                                  // 000000001c58: d72c0004 02024419
	v_mul_lo_u32 v5, s24, v35                                  // 000000001c60: d72c0005 02024618
	v_mad_co_u64_u32 v[0:1], null, s24, v34, 0                 // 000000001c68: d6fe7c00 02024418
	v_add3_u32 v75, v7, s3, v9                                 // 000000001c70: d655004b 04240707
	v_or_b32_e32 v78, v6, v38                                  // 000000001c78: 389c4d06
	v_mul_lo_u32 v6, s24, v41                                  // 000000001c7c: d72c0006 02025218
	v_cndmask_b32_e32 v7, 0, v36, vcc_lo                       // 000000001c84: 020e4880
	s_add_nc_u64 s[6:7], s[58:59], -1                          // 000000001c88: a986c13a
	v_dual_mov_b32 v33, s37 :: v_dual_mov_b32 v108, v39        // 000000001c8c: ca100025 216c0127
	v_add3_u32 v80, v1, v5, v4                                 // 000000001c94: d6550050 04120b01
	v_mul_lo_u32 v5, s25, v40                                  // 000000001c9c: d72c0005 02025019
	v_mov_b32_e32 v4, s37                                      // 000000001ca4: 7e080225
	v_cmp_gt_i64_e64 s3, s[20:21], v[2:3]                      // 000000001ca8: d4540003 02020414
	v_mad_co_u64_u32 v[1:2], null, s24, v40, 0                 // 000000001cb0: d6fe7c01 02025018
	v_or_b32_e32 v81, v0, v38                                  // 000000001cb8: 38a24d00
	v_cndmask_b32_e32 v0, 0, v37, vcc_lo                       // 000000001cbc: 02004a80
	v_or_b32_e32 v3, s36, v54                                  // 000000001cc0: 38066c24
	v_mul_lo_u32 v8, s39, v7                                   // 000000001cc4: d72c0008 02020e27
	v_cmp_lt_u64_e64 s4, s[8:9], s[6:7]                        // 000000001ccc: d4590004 02000c08
	v_dual_mov_b32 v117, v39 :: v_dual_mov_b32 v72, v39        // 000000001cd4: ca100127 75480127
	v_add3_u32 v83, v2, v6, v5                                 // 000000001cdc: d6550053 04160d02
	v_mul_lo_u32 v0, s38, v0                                   // 000000001ce4: d72c0000 02020026
	v_mad_co_u64_u32 v[5:6], null, s38, v7, 0                  // 000000001cec: d6fe7c05 02020e26
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[3:4]                  // 000000001cf4: 7ca80614
	v_or_b32_e32 v2, s36, v53                                  // 000000001cf8: 38046a24
	v_or_b32_e32 v85, v1, v38                                  // 000000001cfc: 38aa4d01
	s_and_b32 s4, s4, exec_lo                                  // 000000001d00: 8b047e04
	v_dual_mov_b32 v111, v39 :: v_dual_mov_b32 v62, v39        // 000000001d04: ca100127 6f3e0127
	s_wait_alu depctr_va_vcc(0)                                // 000000001d0c: bf88ff9d
	v_cndmask_b32_e32 v7, 0, v3, vcc_lo                        // 000000001d10: 020e0680
	v_add3_u32 v6, v6, v0, v8                                  // 000000001d14: d6550006 04220106
	v_dual_mov_b32 v8, s37 :: v_dual_mov_b32 v3, s37           // 000000001d1c: ca100025 08020025
	v_dual_cndmask_b32 v4, 0, v4 :: v_dual_mov_b32 v105, v39   // 000000001d24: ca500880 04680127
	v_dual_mov_b32 v60, v39 :: v_dual_mov_b32 v95, v39         // 000000001d2c: ca100127 3c5e0127
	v_mov_b32_e32 v58, v39                                     // 000000001d34: 7e740327
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001d38: bf870214
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000001d3c: 7ca80414
	v_mul_lo_u32 v4, s38, v4                                   // 000000001d40: d72c0004 02020826
	v_dual_mov_b32 v71, v39 :: v_dual_mov_b32 v56, v39         // 000000001d48: ca100127 47380127
	v_mov_b32_e32 v69, v39                                     // 000000001d50: 7e8a0327
	s_wait_alu depctr_va_vcc(0)                                // 000000001d54: bf88ff9d
	v_dual_mov_b32 v67, v39 :: v_dual_cndmask_b32 v10, 0, v3   // 000000001d58: ca120127 430a0680
	v_mul_lo_u32 v9, s39, v7                                   // 000000001d60: d72c0009 02020e27
	v_mad_co_u64_u32 v[0:1], null, s38, v7, 0                  // 000000001d68: d6fe7c00 02020e26
	v_or_b32_e32 v7, s36, v52                                  // 000000001d70: 380e6824
	v_cndmask_b32_e32 v11, 0, v2, vcc_lo                       // 000000001d74: 02160480
	v_lshlrev_b64_e32 v[2:3], 2, v[5:6]                        // 000000001d78: 3e040a82
	v_or_b32_e32 v6, s36, v49                                  // 000000001d7c: 380c6224
	v_mul_lo_u32 v10, s38, v10                                 // 000000001d80: d72c000a 02021426
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[7:8]                  // 000000001d88: 7ca80e14
	v_mov_b32_e32 v65, v39                                     // 000000001d8c: 7e820327
	v_add3_u32 v1, v1, v4, v9                                  // 000000001d90: d6550001 04260901
	v_mul_lo_u32 v9, s39, v11                                  // 000000001d98: d72c0009 02021627
	v_mad_co_u64_u32 v[4:5], null, s38, v11, 0                 // 000000001da0: d6fe7c04 02021626
	v_mov_b32_e32 v63, v39                                     // 000000001da8: 7e7e0327
	s_wait_alu depctr_va_vcc(0)                                // 000000001dac: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v7, vcc_lo                       // 000000001db0: 02160e80
	v_dual_mov_b32 v7, s37 :: v_dual_cndmask_b32 v8, 0, v8     // 000000001db4: ca120025 07081080
	v_add_co_u32 v88, vcc_lo, s26, v2                          // 000000001dbc: d7006a58 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc4: bf88ff9d
	v_add_co_ci_u32_e64 v89, null, s27, v3, vcc_lo             // 000000001dc8: d5207c59 01aa061b
	s_delay_alu instid0(valu_dep_3)                            // 000000001dd0: bf870003
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001dd4: 7ca80c14
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]                        // 000000001dd8: 3e000082
	v_add3_u32 v5, v5, v10, v9                                 // 000000001ddc: d6550005 04261505
	v_mul_lo_u32 v9, s39, v11                                  // 000000001de4: d72c0009 02021627
	v_mad_co_u64_u32 v[2:3], null, s38, v11, 0                 // 000000001dec: d6fe7c02 02021626
	v_mul_lo_u32 v8, s38, v8                                   // 000000001df4: d72c0008 02021026
	s_wait_alu depctr_va_vcc(0)                                // 000000001dfc: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v6, vcc_lo                       // 000000001e00: 02160c80
	v_or_b32_e32 v6, s36, v48                                  // 000000001e04: 380c6024
	v_cndmask_b32_e32 v10, 0, v7, vcc_lo                       // 000000001e08: 02140e80
	v_add_co_u32 v90, vcc_lo, s26, v0                          // 000000001e0c: d7006a5a 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e14: bf88ff9d
	v_add_co_ci_u32_e64 v91, null, s27, v1, vcc_lo             // 000000001e18: d5207c5b 01aa021b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001e20: 7ca80c14
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001e24: 3e000882
	v_add3_u32 v3, v3, v8, v9                                  // 000000001e28: d6550003 04261103
	v_mul_lo_u32 v8, s39, v11                                  // 000000001e30: d72c0008 02021627
	v_mad_co_u64_u32 v[4:5], null, s38, v11, 0                 // 000000001e38: d6fe7c04 02021626
	v_mul_lo_u32 v9, s38, v10                                  // 000000001e40: d72c0009 02021426
	s_wait_alu depctr_va_vcc(0)                                // 000000001e48: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v6, vcc_lo                       // 000000001e4c: 02160c80
	v_or_b32_e32 v6, s36, v50                                  // 000000001e50: 380c6424
	v_cndmask_b32_e32 v10, 0, v7, vcc_lo                       // 000000001e54: 02140e80
	v_add_co_u32 v93, vcc_lo, s26, v0                          // 000000001e58: d7006a5d 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e60: bf88ff9d
	v_add_co_ci_u32_e64 v94, null, s27, v1, vcc_lo             // 000000001e64: d5207c5e 01aa021b
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001e6c: 7ca80c14
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001e70: 3e000482
	v_add3_u32 v5, v5, v9, v8                                  // 000000001e74: d6550005 04221305
	v_mul_lo_u32 v8, s39, v11                                  // 000000001e7c: d72c0008 02021627
	v_mul_lo_u32 v9, s38, v10                                  // 000000001e84: d72c0009 02021426
	v_mad_co_u64_u32 v[2:3], null, s38, v11, 0                 // 000000001e8c: d6fe7c02 02021626
	s_wait_alu depctr_va_vcc(0)                                // 000000001e94: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v6 :: v_dual_mov_b32 v114, v39  // 000000001e98: ca500c80 0b720127
	v_or_b32_e32 v6, s36, v51                                  // 000000001ea0: 380c6624
	v_cndmask_b32_e32 v10, 0, v7, vcc_lo                       // 000000001ea4: 02140e80
	v_add_co_u32 v96, vcc_lo, s26, v0                          // 000000001ea8: d7006a60 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 000000001eb0: bf88ff9d
	v_add_co_ci_u32_e64 v97, null, s27, v1, vcc_lo             // 000000001eb4: d5207c61 01aa021b
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001ebc: 3e000882
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[6:7]                  // 000000001ec0: 7ca80c14
	v_add3_u32 v3, v3, v9, v8                                  // 000000001ec4: d6550003 04221303
	v_mul_lo_u32 v8, s39, v11                                  // 000000001ecc: d72c0008 02021627
	v_mul_lo_u32 v9, s38, v10                                  // 000000001ed4: d72c0009 02021426
	v_mad_co_u64_u32 v[4:5], null, s38, v11, 0                 // 000000001edc: d6fe7c04 02021626
	v_mov_b32_e32 v92, v39                                     // 000000001ee4: 7eb80327
	s_wait_alu depctr_va_vcc(0)                                // 000000001ee8: bf88ff9d
	v_dual_cndmask_b32 v6, 0, v6 :: v_dual_cndmask_b32 v7, 0, v7// 000000001eec: ca520c80 06060e80
	v_add_co_u32 v98, vcc_lo, s26, v0                          // 000000001ef4: d7006a62 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 000000001efc: bf88ff9d
	v_add_co_ci_u32_e64 v99, null, s27, v1, vcc_lo             // 000000001f00: d5207c63 01aa021b
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001f08: 3e000482
	v_add3_u32 v5, v5, v9, v8                                  // 000000001f0c: d6550005 04221305
	v_mul_lo_u32 v8, s39, v6                                   // 000000001f14: d72c0008 02020c27
	v_mul_lo_u32 v7, s38, v7                                   // 000000001f1c: d72c0007 02020e26
	v_mad_co_u64_u32 v[2:3], null, s38, v6, 0                  // 000000001f24: d6fe7c02 02020c26
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[32:33]                // 000000001f2c: 7ca84014
	v_add_co_u32 v101, s4, s26, v0                             // 000000001f30: d7000465 0202001a
	s_delay_alu instid0(valu_dep_1)                            // 000000001f38: bf870001
	v_add_co_ci_u32_e64 v102, null, s27, v1, s4                // 000000001f3c: d5207c66 0012021b
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001f44: 3e000882
	v_mov_b32_e32 v5, s37                                      // 000000001f48: 7e0a0225
	v_or_b32_e32 v4, s33, v54                                  // 000000001f4c: 38086c21
	v_add3_u32 v3, v3, v7, v8                                  // 000000001f50: d6550003 04220f03
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_cndmask_b32_e64 v6, 0, s37, vcc_lo                       // 000000001f5c: d5010006 01a84a80
	v_dual_cndmask_b32 v9, 0, v32 :: v_dual_mov_b32 v100, v39  // 000000001f64: ca504080 09640127
	v_add_co_u32 v103, s4, s26, v0                             // 000000001f6c: d7000467 0202001a
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000001f74: 7ca80814
	s_wait_alu depctr_va_sdst(0)                               // 000000001f78: bf88f19f
	v_add_co_ci_u32_e64 v104, null, s27, v1, s4                // 000000001f7c: d5207c68 0012021b
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001f84: 3e000482
	v_dual_mov_b32 v3, s37 :: v_dual_mov_b32 v70, v39          // 000000001f88: ca100025 03460127
	v_or_b32_e32 v2, s33, v53                                  // 000000001f90: 38046a21
	s_wait_alu depctr_va_vcc(0)                                // 000000001f94: bf88ff9d
	v_cndmask_b32_e64 v5, 0, s37, vcc_lo                       // 000000001f98: d5010005 01a84a80
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo                        // 000000001fa0: 02080880
	v_mul_lo_u32 v8, s39, v9                                   // 000000001fa4: d72c0008 02021227
	v_mul_lo_u32 v10, s38, v6                                  // 000000001fac: d72c000a 02020c26
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000001fb4: 7ca80414
	v_mad_co_u64_u32 v[6:7], null, s38, v9, 0                  // 000000001fb8: d6fe7c06 02021226
	v_mul_lo_u32 v9, s38, v5                                   // 000000001fc0: d72c0009 02020a26
	v_add_co_u32 v106, s4, s26, v0                             // 000000001fc8: d700046a 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 000000001fd0: bf88f19f
	v_add_co_ci_u32_e64 v107, null, s27, v1, s4                // 000000001fd4: d5207c6b 0012021b
	s_wait_alu depctr_va_vcc(0)                                // 000000001fdc: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v2 :: v_dual_mov_b32 v68, v39   // 000000001fe0: ca500480 0b440127
	v_or_b32_e32 v2, s33, v52                                  // 000000001fe8: 38046821
	v_add3_u32 v7, v7, v10, v8                                 // 000000001fec: d6550007 04221507
	v_mul_lo_u32 v8, s39, v4                                   // 000000001ff4: d72c0008 02020827
	v_mad_co_u64_u32 v[4:5], null, s38, v4, 0                  // 000000001ffc: d6fe7c04 02020826
	v_cndmask_b32_e64 v10, 0, s37, vcc_lo                      // 000000002004: d501000a 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 00000000200c: 7ca80414
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000002010: 3e000c82
	v_mad_co_u64_u32 v[6:7], null, s38, v11, 0                 // 000000002014: d6fe7c06 02021626
	v_mov_b32_e32 v66, v39                                     // 00000000201c: 7e840327
	v_mov_b32_e32 v86, v39                                     // 000000002020: 7eac0327
	v_mov_b32_e32 v84, v39                                     // 000000002024: 7ea80327
	v_add3_u32 v5, v5, v9, v8                                  // 000000002028: d6550005 04221305
	v_mul_lo_u32 v8, s39, v11                                  // 000000002030: d72c0008 02021627
	s_wait_alu depctr_va_vcc(0)                                // 000000002038: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v2, vcc_lo                       // 00000000203c: 02160480
	v_or_b32_e32 v2, s33, v49                                  // 000000002040: 38046221
	v_mul_lo_u32 v9, s38, v10                                  // 000000002044: d72c0009 02021426
	v_add_co_u32 v109, s4, s26, v0                             // 00000000204c: d700046d 0202001a
	v_cndmask_b32_e64 v10, 0, s37, vcc_lo                      // 000000002054: d501000a 01a84a80
	s_delay_alu instid0(valu_dep_4)                            // 00000000205c: bf870004
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000002060: 7ca80414
	s_wait_alu depctr_va_sdst(0)                               // 000000002064: bf88f19f
	v_add_co_ci_u32_e64 v110, null, s27, v1, s4                // 000000002068: d5207c6e 0012021b
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000002070: 3e000882
	v_add3_u32 v7, v7, v9, v8                                  // 000000002074: d6550007 04221307
	v_mul_lo_u32 v8, s39, v11                                  // 00000000207c: d72c0008 02021627
	v_mul_lo_u32 v9, s38, v10                                  // 000000002084: d72c0009 02021426
	v_mad_co_u64_u32 v[4:5], null, s38, v11, 0                 // 00000000208c: d6fe7c04 02021626
	s_wait_alu depctr_va_vcc(0)                                // 000000002094: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v2, vcc_lo                       // 000000002098: 02160480
	v_or_b32_e32 v2, s33, v48                                  // 00000000209c: 38046021
	v_add_co_u32 v112, s4, s26, v0                             // 0000000020a0: d7000470 0202001a
	s_wait_alu depctr_va_sdst(0)                               // 0000000020a8: bf88f19f
	v_add_co_ci_u32_e64 v113, null, s27, v1, s4                // 0000000020ac: d5207c71 0012021b
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 0000000020b4: 3e000c82
	v_cndmask_b32_e64 v10, 0, s37, vcc_lo                      // 0000000020b8: d501000a 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 0000000020c0: 7ca80414
	v_add3_u32 v5, v5, v9, v8                                  // 0000000020c4: d6550005 04221305
	v_mov_b32_e32 v9, s37                                      // 0000000020cc: 7e120225
	v_or_b32_e32 v8, s33, v50                                  // 0000000020d0: 38106421
	v_add_co_u32 v115, s4, s26, v0                             // 0000000020d4: d7000473 0202001a
	v_mul_lo_u32 v12, s39, v11                                 // 0000000020dc: d72c000c 02021627
	v_mul_lo_u32 v10, s38, v10                                 // 0000000020e4: d72c000a 02021426
	v_mad_co_u64_u32 v[6:7], null, s38, v11, 0                 // 0000000020ec: d6fe7c06 02021626
	s_wait_alu depctr_va_sdst(0)                               // 0000000020f4: bf88f19f
	v_add_co_ci_u32_e64 v116, null, s27, v1, s4                // 0000000020f8: d5207c74 0012021b
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000002100: 3e000882
	s_wait_alu depctr_va_vcc(0)                                // 000000002104: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v2 :: v_dual_mov_b32 v82, v39    // 000000002108: ca500480 05520127
	v_or_b32_e32 v2, s33, v51                                  // 000000002110: 38046621
	v_cmp_gt_i64_e64 s4, s[20:21], v[8:9]                      // 000000002114: d4540004 02021014
	v_cndmask_b32_e64 v4, 0, s37, vcc_lo                       // 00000000211c: d5010004 01a84a80
	v_add3_u32 v7, v7, v10, v12                                // 000000002124: d6550007 04321507
	v_mul_lo_u32 v10, s39, v5                                  // 00000000212c: d72c000a 02020a27
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000002134: 7ca80414
	v_mov_b32_e32 v87, v39                                     // 000000002138: 7eae0327
	v_mul_lo_u32 v11, s38, v4                                  // 00000000213c: d72c000b 02020826
	v_mad_co_u64_u32 v[3:4], null, s38, v5, 0                  // 000000002144: d6fe7c03 02020a26
	s_wait_alu depctr_va_sdst(0)                               // 00000000214c: bf88f19f
	v_cndmask_b32_e64 v9, 0, s37, s4                           // 000000002150: d5010009 00104a80
	v_cndmask_b32_e64 v8, 0, v8, s4                            // 000000002158: d5010008 00121080
	s_wait_alu depctr_va_vcc(0)                                // 000000002160: bf88ff9d
	v_cndmask_b32_e64 v5, 0, s37, vcc_lo                       // 000000002164: d5010005 01a84a80
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 00000000216c: 02040480
	v_add_co_u32 v118, vcc_lo, s26, v0                         // 000000002170: d7006a76 0202001a
	v_mul_lo_u32 v12, s39, v8                                  // 000000002178: d72c000c 02021027
	v_mul_lo_u32 v13, s38, v9                                  // 000000002180: d72c000d 02021226
	v_mad_co_u64_u32 v[8:9], null, s38, v8, 0                  // 000000002188: d6fe7c08 02021026
	s_wait_alu depctr_va_vcc(0)                                // 000000002190: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s27, v1, vcc_lo            // 000000002194: d5207c77 01aa021b
	v_add3_u32 v4, v4, v11, v10                                // 00000000219c: d6550004 042a1704
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 0000000021a4: 3e000c82
	v_mul_lo_u32 v7, s39, v2                                   // 0000000021a8: d72c0007 02020427
	v_mul_lo_u32 v10, s38, v5                                  // 0000000021b0: d72c000a 02020a26
	v_mad_co_u64_u32 v[5:6], null, s38, v2, 0                  // 0000000021b8: d6fe7c05 02020426
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]                        // 0000000021c0: 3e040682
	v_add3_u32 v9, v9, v13, v12                                // 0000000021c4: d6550009 04321b09
	v_add_co_u32 v120, vcc_lo, s26, v0                         // 0000000021cc: d7006a78 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 0000000021d4: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s27, v1, vcc_lo            // 0000000021d8: d5207c79 01aa021b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 0000000021e0: bf8701d3
	v_lshlrev_b64_e32 v[0:1], 2, v[8:9]                        // 0000000021e4: 3e001082
	v_add3_u32 v6, v6, v10, v7                                 // 0000000021e8: d6550006 041e1506
	v_add_co_u32 v122, vcc_lo, s26, v2                         // 0000000021f0: d7006a7a 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 0000000021f8: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s27, v3, vcc_lo            // 0000000021fc: d5207c7b 01aa061b
	v_lshlrev_b64_e32 v[2:3], 2, v[5:6]                        // 000000002204: 3e040a82
	v_add_co_u32 v124, vcc_lo, s26, v0                         // 000000002208: d7006a7c 0202001a
	s_wait_alu depctr_va_vcc(0)                                // 000000002210: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s27, v1, vcc_lo            // 000000002214: d5207c7d 01aa021b
	v_mov_b32_e32 v79, v39                                     // 00000000221c: 7e9e0327
	s_delay_alu instid0(valu_dep_4)                            // 000000002220: bf870004
	v_add_co_u32 v126, vcc_lo, s26, v2                         // 000000002224: d7006a7e 0202041a
	s_wait_alu depctr_va_vcc(0)                                // 00000000222c: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s27, v3, vcc_lo            // 000000002230: d5207c7f 01aa061b
	v_mov_b32_e32 v77, v39                                     // 000000002238: 7e9a0327
	v_mov_b32_e32 v73, v39                                     // 00000000223c: 7e920327
	v_mov_b32_e32 v61, v39                                     // 000000002240: 7e7a0327
	v_mov_b32_e32 v59, v39                                     // 000000002244: 7e760327
	v_mov_b32_e32 v57, v39                                     // 000000002248: 7e720327
	v_mov_b32_e32 v55, v39                                     // 00000000224c: 7e6e0327
	s_cselect_b32 s5, s9, s7                                   // 000000002250: 98050709
	s_cselect_b32 s4, s8, s6                                   // 000000002254: 98040608
	s_add_nc_u64 s[44:45], s[24:25], -1                        // 000000002258: a9acc118
	s_add_nc_u64 s[46:47], s[24:25], -2                        // 00000000225c: a9aec218
	s_add_nc_u64 s[48:49], s[24:25], -3                        // 000000002260: a9b0c318
	s_add_nc_u64 s[50:51], s[24:25], -4                        // 000000002264: a9b2c418
	s_add_nc_u64 s[52:53], s[24:25], -5                        // 000000002268: a9b4c518
	s_add_nc_u64 s[54:55], s[24:25], -6                        // 00000000226c: a9b6c618
	s_add_nc_u64 s[56:57], s[24:25], -7                        // 000000002270: a9b8c718
	s_mov_b64 s[64:65], 0                                      // 000000002274: bec00180
	s_wait_alu depctr_sa_sdst(0)                               // 000000002278: bf88ff9e
	s_lshl_b64 s[60:61], s[4:5], 2                             // 00000000227c: 84bc8204
	v_mov_b32_e32 v24, 0                                       // 000000002280: 7e300280
	s_add_nc_u64 s[62:63], s[64:65], 0x80                      // 000000002284: a9beff40 00000080
	s_mov_b64 s[66:67], s[64:65]                               // 00000000228c: bec20140
	s_delay_alu instid0(valu_dep_1)                            // 000000002290: bf870001
	v_dual_mov_b32 v25, v24 :: v_dual_mov_b32 v26, v24         // 000000002294: ca100118 191a0118
	v_dual_mov_b32 v27, v24 :: v_dual_mov_b32 v28, v24         // 00000000229c: ca100118 1b1c0118
	v_dual_mov_b32 v29, v24 :: v_dual_mov_b32 v30, v24         // 0000000022a4: ca100118 1d1e0118
	v_dual_mov_b32 v31, v24 :: v_dual_mov_b32 v16, v24         // 0000000022ac: ca100118 1f100118
	v_dual_mov_b32 v17, v24 :: v_dual_mov_b32 v18, v24         // 0000000022b4: ca100118 11120118
	v_dual_mov_b32 v19, v24 :: v_dual_mov_b32 v20, v24         // 0000000022bc: ca100118 13140118
	v_dual_mov_b32 v21, v24 :: v_dual_mov_b32 v22, v24         // 0000000022c4: ca100118 15160118
	v_dual_mov_b32 v23, v24 :: v_dual_mov_b32 v8, v24          // 0000000022cc: ca100118 17080118
	v_dual_mov_b32 v9, v24 :: v_dual_mov_b32 v10, v24          // 0000000022d4: ca100118 090a0118
	v_dual_mov_b32 v11, v24 :: v_dual_mov_b32 v12, v24         // 0000000022dc: ca100118 0b0c0118
	v_dual_mov_b32 v13, v24 :: v_dual_mov_b32 v14, v24         // 0000000022e4: ca100118 0d0e0118
	v_dual_mov_b32 v15, v24 :: v_dual_mov_b32 v0, v24          // 0000000022ec: ca100118 0f000118
	v_dual_mov_b32 v1, v24 :: v_dual_mov_b32 v2, v24           // 0000000022f4: ca100118 01020118
	v_dual_mov_b32 v3, v24 :: v_dual_mov_b32 v4, v24           // 0000000022fc: ca100118 03040118
	v_dual_mov_b32 v5, v24 :: v_dual_mov_b32 v6, v24           // 000000002304: ca100118 05060118
	v_mov_b32_e32 v7, v24                                      // 00000000230c: 7e0e0318
	s_wait_alu depctr_sa_sdst(0)                               // 000000002310: bf88ff9e
	v_mov_b32_e32 v43, s67                                     // 000000002314: 7e560243
	v_or_b32_e32 v42, s66, v38                                 // 000000002318: 38544c42
	v_add_co_u32 v47, s11, s66, v76                            // 00000000231c: d7000b2f 02029842
	s_wait_alu depctr_va_sdst(0)                               // 000000002324: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s67, v74, s11              // 000000002328: d5207c86 002e9443
	s_delay_alu instid0(valu_dep_3)                            // 000000002330: bf870003
	v_cmp_gt_i64_e64 s8, s[52:53], v[42:43]                    // 000000002334: d4540008 02025434
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[42:43]                // 00000000233c: 7cb85418
	v_cmp_gt_i64_e64 s4, s[44:45], v[42:43]                    // 000000002340: d4540004 0202542c
	v_cmp_gt_i64_e64 s5, s[46:47], v[42:43]                    // 000000002348: d4540005 0202542e
	v_cmp_gt_i64_e64 s6, s[48:49], v[42:43]                    // 000000002350: d4540006 02025430
	v_cmp_gt_i64_e64 s7, s[50:51], v[42:43]                    // 000000002358: d4540007 02025432
	v_cmp_gt_i64_e64 s9, s[54:55], v[42:43]                    // 000000002360: d4540009 02025436
	v_cmp_gt_i64_e64 s10, s[56:57], v[42:43]                   // 000000002368: d454000a 02025438
	v_or_b32_e32 v42, 5, v47                                   // 000000002370: 38545e85
	s_and_b32 s11, s2, s8                                      // 000000002374: 8b0b0802
	s_or_b32 s68, s66, 16                                      // 000000002378: 8c449042
	s_wait_alu depctr_sa_sdst(0)                               // 00000000237c: bf88ff9e
	v_cndmask_b32_e64 v44, 0, v134, s11                        // 000000002380: d501002c 002f0c80
	v_cndmask_b32_e64 v42, 0, v42, s11                         // 000000002388: d501002a 002e5480
	s_delay_alu instid0(valu_dep_1)                            // 000000002390: bf870001
	v_add_co_u32 v43, s12, s30, v42                            // 000000002394: d7000c2b 0202541e
	v_or_b32_e32 v42, 6, v47                                   // 00000000239c: 38545e86
	s_wait_alu depctr_va_sdst(0)                               // 0000000023a0: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s31, v44, s12               // 0000000023a4: d5207c2c 0032581f
	s_and_b32 s12, s2, s9                                      // 0000000023ac: 8b0c0902
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023b0: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s12                         // 0000000023b4: d501002a 00325480
	v_cndmask_b32_e64 v46, 0, v134, s12                        // 0000000023bc: d501002e 00330c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 0000000023c4: bf8701b2
	v_add_co_u32 v45, s13, s30, v42                            // 0000000023c8: d7000d2d 0202541e
	v_or_b32_e32 v42, 7, v47                                   // 0000000023d0: 38545e87
	s_wait_alu depctr_va_sdst(0)                               // 0000000023d4: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s13               // 0000000023d8: d5207c2e 00365c1f
	s_and_b32 s13, s2, s10                                     // 0000000023e0: 8b0d0a02
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023e4: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s13                         // 0000000023e8: d501002a 00365480
	v_cndmask_b32_e64 v129, 0, v134, s13                       // 0000000023f0: d5010081 00370c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 0000000023f8: bf8701b2
	v_add_co_u32 v128, s14, s30, v42                           // 0000000023fc: d7000e80 0202541e
	v_or_b32_e32 v42, 4, v47                                   // 000000002404: 38545e84
	s_wait_alu depctr_va_sdst(0)                               // 000000002408: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s14             // 00000000240c: d5207c81 003b021f
	s_and_b32 s14, s2, s7                                      // 000000002414: 8b0e0702
	s_wait_alu depctr_sa_sdst(0)                               // 000000002418: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s14                         // 00000000241c: d501002a 003a5480
	v_cndmask_b32_e64 v131, 0, v134, s14                       // 000000002424: d5010083 003b0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 00000000242c: bf8701b2
	v_add_co_u32 v130, s15, s30, v42                           // 000000002430: d7000f82 0202541e
	v_or_b32_e32 v42, 3, v47                                   // 000000002438: 38545e83
	s_wait_alu depctr_va_sdst(0)                               // 00000000243c: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s31, v131, s15             // 000000002440: d5207c83 003f061f
	s_and_b32 s15, s2, s6                                      // 000000002448: 8b0f0602
	s_wait_alu depctr_sa_sdst(0)                               // 00000000244c: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s15                         // 000000002450: d501002a 003e5480
	v_cndmask_b32_e64 v133, 0, v134, s15                       // 000000002458: d5010085 003f0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002460: bf870122
	v_add_co_u32 v132, s16, s30, v42                           // 000000002464: d7001084 0202541e
	s_wait_alu depctr_va_sdst(0)                               // 00000000246c: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s31, v133, s16             // 000000002470: d5207c85 00430a1f
	s_clause 0x4                                               // 000000002478: bf850004
	global_load_d16_u8 v42, v[128:129], off                    // 00000000247c: ee07807c 0000002a 00000080
	global_load_d16_hi_u8 v42, v[45:46], off                   // 000000002488: ee08407c 0000002a 0000002d
	global_load_d16_u8 v43, v[43:44], off                      // 000000002494: ee07807c 0000002b 0000002b
	global_load_d16_hi_u8 v43, v[130:131], off                 // 0000000024a0: ee08407c 0000002b 00000082
	global_load_d16_u8 v44, v[132:133], off                    // 0000000024ac: ee07807c 0000002c 00000084
	s_wait_loadcnt 0x3                                         // 0000000024b8: bfc00003
	v_cndmask_b16 v42.l, 0, v42.l, s13                         // 0000000024bc: d65d002a 00365480
	v_cndmask_b16 v42.h, 0, v42.h, s12                         // 0000000024c4: d65d502a 00325480
	s_wait_loadcnt 0x1                                         // 0000000024cc: bfc00001
	v_cndmask_b16 v43.l, 0, v43.l, s11                         // 0000000024d0: d65d002b 002e5680
	v_cndmask_b16 v43.h, 0, v43.h, s14                         // 0000000024d8: d65d502b 003a5680
	s_and_b32 s11, s2, s5                                      // 0000000024e0: 8b0b0502
	v_lshlrev_b16 v42.l, 8, v42.l                              // 0000000024e4: d738002a 02025488
	v_and_b16 v42.h, 0xff, v42.h op_sel:[0,1,1]                // 0000000024ec: d762502a 020254ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024f8: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v134, s11                        // 0000000024fc: d501002e 002f0c80
	s_wait_loadcnt 0x0                                         // 000000002504: bfc00000
	v_and_b16 v44.h, 0xff, v43.h op_sel:[0,1,1]                // 000000002508: d762502c 020256ff 000000ff
	v_lshlrev_b16 v43.l, 8, v43.l                              // 000000002514: d738002b 02025688
	s_and_b32 s14, s3, s10                                     // 00000000251c: 8b0e0a03
	v_or_b16 v43.h, v42.h, v42.l op_sel:[1,0,1]                // 000000002520: d763482b 0202552a
	v_or_b32_e32 v42, 2, v47                                   // 000000002528: 38545e82
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 00000000252c: bf870113
	v_or_b16 v43.l, v44.h, v43.l op_sel:[1,0,0]                // 000000002530: d763082b 0202572c
	v_cndmask_b32_e64 v42, 0, v42, s11                         // 000000002538: d501002a 002e5480
	s_delay_alu instid0(valu_dep_1)                            // 000000002540: bf870001
	v_add_co_u32 v45, s12, s30, v42                            // 000000002544: d7000c2d 0202541e
	v_or_b32_e32 v42, 1, v47                                   // 00000000254c: 38545e81
	s_wait_alu depctr_va_sdst(0)                               // 000000002550: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s12               // 000000002554: d5207c2e 00325c1f
	s_and_b32 s12, s2, s4                                      // 00000000255c: 8b0c0402
	s_wait_alu depctr_sa_sdst(0)                               // 000000002560: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s12                         // 000000002564: d501002a 00325480
	v_cndmask_b32_e64 v129, 0, v134, s12                       // 00000000256c: d5010081 00330c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002574: bf870122
	v_add_co_u32 v128, s13, s30, v42                           // 000000002578: d7000d80 0202541e
	s_wait_alu depctr_va_sdst(0)                               // 000000002580: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s13             // 000000002584: d5207c81 0037021f
	s_clause 0x1                                               // 00000000258c: bf850001
	global_load_d16_u8 v42, v[45:46], off                      // 000000002590: ee07807c 0000002a 0000002d
	global_load_d16_hi_u8 v44, v[128:129], off                 // 00000000259c: ee08407c 0000002c 00000080
	s_wait_loadcnt 0x0                                         // 0000000025a8: bfc00000
	v_cndmask_b16 v42.h, 0, v44.l, s15                         // 0000000025ac: d65d402a 003e5880
	s_and_b32 s15, s3, s7                                      // 0000000025b4: 8b0f0703
	s_delay_alu instid0(valu_dep_1)                            // 0000000025b8: bf870001
	v_lshlrev_b16 v42.h, 8, v42.h op_sel:[0,1,1]               // 0000000025bc: d738502a 02025488
	v_cndmask_b16 v42.l, 0, v42.l, s11                         // 0000000025c4: d65d002a 002e5480
	s_and_b32 s11, s2, vcc_lo                                  // 0000000025cc: 8b0b6a02
	v_cndmask_b16 v44.l, 0, v44.h, s12                         // 0000000025d0: d65d102c 00325880
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025d8: bf88ff9e
	v_cndmask_b32_e64 v45, 0, v47, s11                         // 0000000025dc: d501002d 002e5e80
	v_cndmask_b32_e64 v46, 0, v134, s11                        // 0000000025e4: d501002e 002f0c80
	v_and_b16 v42.l, 0xff, v42.l                               // 0000000025ec: d762002a 020254ff 000000ff
	s_and_b32 s12, s3, s8                                      // 0000000025f8: 8b0c0803
	v_lshlrev_b16 v44.l, 8, v44.l                              // 0000000025fc: d738002c 02025888
	v_add_co_u32 v45, s13, s30, v45                            // 000000002604: d7000d2d 02025a1e
	s_wait_alu depctr_va_sdst(0)                               // 00000000260c: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s13               // 000000002610: d5207c2e 00365c1f
	v_or_b16 v42.h, v42.l, v42.h op_sel:[0,1,1]                // 000000002618: d763502a 0202552a
	s_and_b32 s13, s3, s9                                      // 000000002620: 8b0d0903
	global_load_d16_u8 v42, v[45:46], off                      // 000000002624: ee07807c 0000002a 0000002d
	s_wait_loadcnt 0x0                                         // 000000002630: bfc00000
	v_cndmask_b16 v42.l, 0, v42.l, s11                         // 000000002634: d65d002a 002e5480
	v_add_co_u32 v46, s11, s66, v78                            // 00000000263c: d7000b2e 02029c42
	s_wait_alu depctr_va_sdst(0)                               // 000000002644: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s67, v75, s11               // 000000002648: d5207c2f 002e9643
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002650: bf870193
	v_and_b16 v42.l, 0xff, v42.l                               // 000000002654: d762002a 020254ff 000000ff
	v_or_b32_e32 v132, 4, v46                                  // 000000002660: 39085c84
	v_or_b32_e32 v134, 3, v46                                  // 000000002664: 390c5c83
	s_wait_alu depctr_sa_sdst(0)                               // 000000002668: bf88ff9e
	v_cndmask_b32_e64 v45, 0, v47, s12                         // 00000000266c: d501002d 00325e80
	v_cndmask_b32_e64 v133, 0, v47, s15                        // 000000002674: d5010085 003e5e80
	v_or_b16 v42.l, v42.l, v44.l                               // 00000000267c: d763002a 0202592a
	v_or_b32_e32 v44, 5, v46                                   // 000000002684: 38585c85
	v_cndmask_b32_e64 v132, 0, v132, s15                       // 000000002688: d5010084 003f0880
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002690: bf870092
	v_cndmask_b32_e64 v44, 0, v44, s12                         // 000000002694: d501002c 00325880
	v_add_co_u32 v128, s11, s30, v44                           // 00000000269c: d7000b80 0202581e
	v_or_b32_e32 v44, 6, v46                                   // 0000000026a4: 38585c86
	s_wait_alu depctr_va_sdst(0)                               // 0000000026a8: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v45, s11              // 0000000026ac: d5207c81 002e5a1f
	v_cndmask_b32_e64 v45, 0, v47, s13                         // 0000000026b4: d501002d 00365e80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000026bc: bf870093
	v_cndmask_b32_e64 v44, 0, v44, s13                         // 0000000026c0: d501002c 00365880
	v_add_co_u32 v130, s11, s30, v44                           // 0000000026c8: d7000b82 0202581e
	v_or_b32_e32 v44, 7, v46                                   // 0000000026d0: 38585c87
	s_wait_alu depctr_va_sdst(0)                               // 0000000026d4: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s31, v45, s11              // 0000000026d8: d5207c83 002e5a1f
	v_cndmask_b32_e64 v45, 0, v47, s14                         // 0000000026e0: d501002d 003a5e80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000026e8: bf870093
	v_cndmask_b32_e64 v44, 0, v44, s14                         // 0000000026ec: d501002c 003a5880
	v_add_co_u32 v44, s11, s30, v44                            // 0000000026f4: d7000b2c 0202581e
	s_wait_alu depctr_va_sdst(0)                               // 0000000026fc: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002700: bf870003
	v_add_co_ci_u32_e64 v45, null, s31, v45, s11               // 000000002704: d5207c2d 002e5a1f
	v_add_co_u32 v132, s11, s30, v132                          // 00000000270c: d7000b84 0203081e
	s_wait_alu depctr_va_sdst(0)                               // 000000002714: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s31, v133, s11             // 000000002718: d5207c85 002f0a1f
	s_and_b32 s11, s3, s6                                      // 000000002720: 8b0b0603
	s_wait_alu depctr_sa_sdst(0)                               // 000000002724: bf88ff9e
	v_cndmask_b32_e64 v134, 0, v134, s11                       // 000000002728: d5010086 002f0c80
	v_cndmask_b32_e64 v135, 0, v47, s11                        // 000000002730: d5010087 002e5e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002738: bf870122
	v_add_co_u32 v134, s16, s30, v134                          // 00000000273c: d7001086 02030c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000002744: bf88f19f
	v_add_co_ci_u32_e64 v135, null, s31, v135, s16             // 000000002748: d5207c87 00430e1f
	s_clause 0x4                                               // 000000002750: bf850004
	global_load_d16_u8 v44, v[44:45], off                      // 000000002754: ee07807c 0000002c 0000002c
	global_load_d16_hi_u8 v44, v[130:131], off                 // 000000002760: ee08407c 0000002c 00000082
	global_load_d16_u8 v45, v[128:129], off                    // 00000000276c: ee07807c 0000002d 00000080
	global_load_d16_hi_u8 v45, v[132:133], off                 // 000000002778: ee08407c 0000002d 00000084
	global_load_d16_u8 v128, v[134:135], off                   // 000000002784: ee07807c 00000080 00000086
	s_wait_loadcnt 0x3                                         // 000000002790: bfc00003
	v_cndmask_b16 v44.l, 0, v44.l, s14                         // 000000002794: d65d002c 003a5880
	v_cndmask_b16 v44.h, 0, v44.h, s13                         // 00000000279c: d65d502c 00365880
	s_wait_loadcnt 0x1                                         // 0000000027a4: bfc00001
	v_cndmask_b16 v45.l, 0, v45.l, s12                         // 0000000027a8: d65d002d 00325a80
	v_cndmask_b16 v45.h, 0, v45.h, s15                         // 0000000027b0: d65d502d 003e5a80
	s_and_b32 s12, s3, s5                                      // 0000000027b8: 8b0c0503
	v_lshlrev_b16 v44.l, 8, v44.l                              // 0000000027bc: d738002c 02025888
	v_and_b16 v44.h, 0xff, v44.h op_sel:[0,1,1]                // 0000000027c4: d762502c 020258ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027d0: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v47, s12                        // 0000000027d4: d5010082 00325e80
	s_wait_loadcnt 0x0                                         // 0000000027dc: bfc00000
	v_and_b16 v128.h, 0xff, v45.h op_sel:[0,1,1]               // 0000000027e0: d7625080 02025aff 000000ff
	v_lshlrev_b16 v45.l, 8, v45.l                              // 0000000027ec: d738002d 02025a88
	v_or_b16 v45.h, v44.h, v44.l op_sel:[1,0,1]                // 0000000027f4: d763482d 0202592c
	v_or_b32_e32 v44, 2, v46                                   // 0000000027fc: 38585c82
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002800: bf870113
	v_or_b16 v45.l, v128.h, v45.l op_sel:[1,0,0]               // 000000002804: d763082d 02025b80
	v_cndmask_b32_e64 v44, 0, v44, s12                         // 00000000280c: d501002c 00325880
	s_delay_alu instid0(valu_dep_1)                            // 000000002814: bf870001
	v_add_co_u32 v129, s13, s30, v44                           // 000000002818: d7000d81 0202581e
	v_or_b32_e32 v44, 1, v46                                   // 000000002820: 38585c81
	s_wait_alu depctr_va_sdst(0)                               // 000000002824: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s31, v130, s13             // 000000002828: d5207c82 0037041f
	s_and_b32 s13, s3, s4                                      // 000000002830: 8b0d0403
	s_wait_alu depctr_sa_sdst(0)                               // 000000002834: bf88ff9e
	v_cndmask_b32_e64 v44, 0, v44, s13                         // 000000002838: d501002c 00365880
	v_cndmask_b32_e64 v132, 0, v47, s13                        // 000000002840: d5010084 00365e80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002848: bf870122
	v_add_co_u32 v131, s14, s30, v44                           // 00000000284c: d7000e83 0202581e
	s_wait_alu depctr_va_sdst(0)                               // 000000002854: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s31, v132, s14             // 000000002858: d5207c84 003b081f
	s_clause 0x1                                               // 000000002860: bf850001
	global_load_d16_u8 v44, v[129:130], off                    // 000000002864: ee07807c 0000002c 00000081
	global_load_d16_hi_u8 v128, v[131:132], off                // 000000002870: ee08407c 00000080 00000083
	s_wait_loadcnt 0x0                                         // 00000000287c: bfc00000
	v_cndmask_b16 v44.h, 0, v128.l, s11                        // 000000002880: d65d402c 002f0080
	v_add_co_u32 v139, s11, s66, v81                           // 000000002888: d7000b8b 0202a242
	s_wait_alu depctr_va_sdst(0)                               // 000000002890: bf88f19f
	v_add_co_ci_u32_e64 v140, null, s67, v80, s11              // 000000002894: d5207c8c 002ea043
	s_and_b32 s11, s1, s8                                      // 00000000289c: 8b0b0801
	v_or_b32_e32 v129, 5, v139                                 // 0000000028a0: 39031685
	v_or_b32_e32 v131, 6, v139                                 // 0000000028a4: 39071686
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028a8: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v140, s11                       // 0000000028ac: d5010082 002f1880
	v_or_b32_e32 v133, 7, v139                                 // 0000000028b4: 390b1687
	v_or_b32_e32 v135, 4, v139                                 // 0000000028b8: 390f1684
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 0000000028bc: d5010081 002f0280
	v_or_b32_e32 v137, 3, v139                                 // 0000000028c4: 39131683
	v_lshlrev_b16 v44.h, 8, v44.h op_sel:[0,1,1]               // 0000000028c8: d738502c 02025888
	s_and_b32 s8, s0, s8                                       // 0000000028d0: 8b080800
	v_cndmask_b16 v44.l, 0, v44.l, s12                         // 0000000028d4: d65d002c 00325880
	v_add_co_u32 v129, s12, s34, v129                          // 0000000028dc: d7000c81 02030222
	s_wait_alu depctr_va_sdst(0)                               // 0000000028e4: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s35, v130, s12             // 0000000028e8: d5207c82 00330423
	s_and_b32 s12, s1, s9                                      // 0000000028f0: 8b0c0901
	v_and_b16 v44.l, 0xff, v44.l                               // 0000000028f4: d762002c 020258ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002900: bf88ff9e
	v_cndmask_b32_e64 v131, 0, v131, s12                       // 000000002904: d5010083 00330680
	v_cndmask_b32_e64 v132, 0, v140, s12                       // 00000000290c: d5010084 00331880
	s_and_b32 s9, s0, s9                                       // 000000002914: 8b090900
	v_or_b16 v44.h, v44.l, v44.h op_sel:[0,1,1]                // 000000002918: d763502c 0202592c
	s_delay_alu instid0(valu_dep_3)                            // 000000002920: bf870003
	v_add_co_u32 v131, s14, s34, v131                          // 000000002924: d7000e83 02030622
	s_wait_alu depctr_va_sdst(0)                               // 00000000292c: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s35, v132, s14             // 000000002930: d5207c84 003b0823
	s_and_b32 s14, s1, s10                                     // 000000002938: 8b0e0a01
	s_and_b32 s10, s0, s10                                     // 00000000293c: 8b0a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002940: bf88ff9e
	v_cndmask_b32_e64 v133, 0, v133, s14                       // 000000002944: d5010085 003b0a80
	v_cndmask_b32_e64 v134, 0, v140, s14                       // 00000000294c: d5010086 003b1880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002954: bf870122
	v_add_co_u32 v133, s15, s34, v133                          // 000000002958: d7000f85 02030a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002960: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v134, s15             // 000000002964: d5207c86 003f0c23
	s_and_b32 s15, s1, s7                                      // 00000000296c: 8b0f0701
	s_and_b32 s7, s0, s7                                       // 000000002970: 8b070700
	s_wait_alu depctr_sa_sdst(0)                               // 000000002974: bf88ff9e
	v_cndmask_b32_e64 v135, 0, v135, s15                       // 000000002978: d5010087 003f0e80
	v_cndmask_b32_e64 v136, 0, v140, s15                       // 000000002980: d5010088 003f1880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002988: bf870122
	v_add_co_u32 v135, s16, s34, v135                          // 00000000298c: d7001087 02030e22
	s_wait_alu depctr_va_sdst(0)                               // 000000002994: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v136, s16             // 000000002998: d5207c88 00431023
	s_and_b32 s16, s1, s6                                      // 0000000029a0: 8b100601
	s_and_b32 s6, s0, s6                                       // 0000000029a4: 8b060600
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029a8: bf88ff9e
	v_cndmask_b32_e64 v137, 0, v137, s16                       // 0000000029ac: d5010089 00431280
	v_cndmask_b32_e64 v138, 0, v140, s16                       // 0000000029b4: d501008a 00431880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000029bc: bf870122
	v_add_co_u32 v137, s17, s34, v137                          // 0000000029c0: d7001189 02031222
	s_wait_alu depctr_va_sdst(0)                               // 0000000029c8: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s35, v138, s17             // 0000000029cc: d5207c8a 00471423
	s_clause 0x4                                               // 0000000029d4: bf850004
	global_load_d16_u8 v44, v[133:134], off                    // 0000000029d8: ee07807c 0000002c 00000085
	global_load_d16_u8 v128, v[131:132], off                   // 0000000029e4: ee07807c 00000080 00000083
	global_load_d16_u8 v129, v[129:130], off                   // 0000000029f0: ee07807c 00000081 00000081
	global_load_d16_hi_u8 v129, v[135:136], off                // 0000000029fc: ee08407c 00000081 00000087
	global_load_d16_u8 v130, v[137:138], off                   // 000000002a08: ee07807c 00000082 00000089
	s_wait_loadcnt 0x4                                         // 000000002a14: bfc00004
	v_cndmask_b16 v44.l, 0, v44.l, s14                         // 000000002a18: d65d002c 003a5880
	s_wait_loadcnt 0x3                                         // 000000002a20: bfc00003
	v_cndmask_b16 v128.l, 0, v128.l, s12                       // 000000002a24: d65d0080 00330080
	s_wait_loadcnt 0x1                                         // 000000002a2c: bfc00001
	v_cndmask_b16 v129.l, 0, v129.l, s11                       // 000000002a30: d65d0081 002f0280
	v_cndmask_b16 v129.h, 0, v129.h, s15                       // 000000002a38: d65d5081 003f0280
	s_and_b32 s11, s1, s5                                      // 000000002a40: 8b0b0501
	v_lshlrev_b16 v44.l, 8, v44.l                              // 000000002a44: d738002c 02025888
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a4c: bf88ff9e
	v_cndmask_b32_e64 v131, 0, v140, s11                       // 000000002a50: d5010083 002f1880
	v_lshlrev_b16 v129.l, 8, v129.l                            // 000000002a58: d7380081 02030288
	v_and_b16 v129.h, 0xff, v129.h op_sel:[0,1,1]              // 000000002a60: d7625081 020302ff 000000ff
	v_and_b16 v128.l, 0xff, v128.l                             // 000000002a6c: d7620080 020300ff 000000ff
	s_and_b32 s5, s0, s5                                       // 000000002a78: 8b050500
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002a7c: bf8701a2
	v_or_b16 v132.l, v129.h, v129.l op_sel:[1,0,0]             // 000000002a80: d7630884 02030381
	v_or_b32_e32 v129, 2, v139                                 // 000000002a88: 39031682
	v_or_b16 v132.h, v128.l, v44.l op_sel:[0,0,1]              // 000000002a8c: d7634084 02025980
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002a94: bf870092
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 000000002a98: d5010081 002f0280
	v_add_co_u32 v133, s12, s34, v129                          // 000000002aa0: d7000c85 02030222
	v_or_b32_e32 v129, 1, v139                                 // 000000002aa8: 39031681
	s_wait_alu depctr_va_sdst(0)                               // 000000002aac: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v131, s12             // 000000002ab0: d5207c86 00330623
	s_and_b32 s12, s1, s4                                      // 000000002ab8: 8b0c0401
	s_and_b32 s4, s0, s4                                       // 000000002abc: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ac0: bf88ff9e
	v_cndmask_b32_e64 v129, 0, v129, s12                       // 000000002ac4: d5010081 00330280
	v_cndmask_b32_e64 v131, 0, v140, s12                       // 000000002acc: d5010083 00331880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002ad4: bf870122
	v_add_co_u32 v135, s14, s34, v129                          // 000000002ad8: d7000e87 02030222
	s_wait_alu depctr_va_sdst(0)                               // 000000002ae0: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v131, s14             // 000000002ae4: d5207c88 003b0623
	s_clause 0x1                                               // 000000002aec: bf850001
	global_load_d16_u8 v44, v[133:134], off                    // 000000002af0: ee07807c 0000002c 00000085
	global_load_d16_u8 v128, v[135:136], off                   // 000000002afc: ee07807c 00000080 00000087
	s_wait_loadcnt 0x2                                         // 000000002b08: bfc00002
	v_cndmask_b16 v129.l, 0, v130.l, s16                       // 000000002b0c: d65d0081 00430480
	s_delay_alu instid0(valu_dep_1)                            // 000000002b14: bf870001
	v_lshlrev_b16 v129.l, 8, v129.l                            // 000000002b18: d7380081 02030288
	s_wait_loadcnt 0x1                                         // 000000002b20: bfc00001
	v_cndmask_b16 v44.l, 0, v44.l, s11                         // 000000002b24: d65d002c 002e5880
	s_and_b32 s11, s1, vcc_lo                                  // 000000002b2c: 8b0b6a01
	s_wait_loadcnt 0x0                                         // 000000002b30: bfc00000
	v_cndmask_b16 v128.l, 0, v128.l, s12                       // 000000002b34: d65d0080 00330080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b3c: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v140, s11                       // 000000002b40: d5010082 002f1880
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002b48: d762002c 020258ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002b54: bf870113
	v_lshlrev_b16 v128.l, 8, v128.l                            // 000000002b58: d7380080 02030088
	v_or_b16 v131.h, v44.l, v129.l op_sel:[0,0,1]              // 000000002b60: d7634083 0203032c
	v_cndmask_b32_e64 v129, 0, v139, s11                       // 000000002b68: d5010081 002f1680
	s_delay_alu instid0(valu_dep_1)                            // 000000002b70: bf870001
	v_add_co_u32 v129, s14, s34, v129                          // 000000002b74: d7000e81 02030222
	s_wait_alu depctr_va_sdst(0)                               // 000000002b7c: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s35, v130, s14             // 000000002b80: d5207c82 003b0423
	global_load_d16_u8 v44, v[129:130], off                    // 000000002b88: ee07807c 0000002c 00000081
	s_wait_loadcnt 0x0                                         // 000000002b94: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, s11                         // 000000002b98: d65d002c 002e5880
	v_add_co_u32 v141, s11, s66, v85                           // 000000002ba0: d7000b8d 0202aa42
	s_wait_alu depctr_va_sdst(0)                               // 000000002ba8: bf88f19f
	v_add_co_ci_u32_e64 v142, null, s67, v83, s11              // 000000002bac: d5207c8e 002ea643
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002bb4: bf870193
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002bb8: d762002c 020258ff 000000ff
	v_or_b32_e32 v129, 5, v141                                 // 000000002bc4: 39031a85
	v_or_b32_e32 v133, 6, v141                                 // 000000002bc8: 390b1a86
	v_or_b32_e32 v135, 7, v141                                 // 000000002bcc: 390f1a87
	v_or_b32_e32 v137, 4, v141                                 // 000000002bd0: 39131a84
	v_cndmask_b32_e64 v130, 0, v142, s8                        // 000000002bd4: d5010082 00231c80
	v_cndmask_b32_e64 v129, 0, v129, s8                        // 000000002bdc: d5010081 00230280
	v_cndmask_b32_e64 v133, 0, v133, s9                        // 000000002be4: d5010085 00270a80
	v_or_b32_e32 v139, 3, v141                                 // 000000002bec: 39171a83
	v_cndmask_b32_e64 v134, 0, v142, s9                        // 000000002bf0: d5010086 00271c80
	v_cndmask_b32_e64 v135, 0, v135, s10                       // 000000002bf8: d5010087 002b0e80
	v_add_co_u32 v129, s11, s34, v129                          // 000000002c00: d7000b81 02030222
	v_cndmask_b32_e64 v136, 0, v142, s10                       // 000000002c08: d5010088 002b1c80
	v_cndmask_b32_e64 v137, 0, v137, s7                        // 000000002c10: d5010089 001f1280
	s_wait_alu depctr_va_sdst(0)                               // 000000002c18: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s35, v130, s11             // 000000002c1c: d5207c82 002f0423
	v_add_co_u32 v133, s11, s34, v133                          // 000000002c24: d7000b85 02030a22
	v_cndmask_b32_e64 v138, 0, v142, s7                        // 000000002c2c: d501008a 001f1c80
	v_cndmask_b32_e64 v139, 0, v139, s6                        // 000000002c34: d501008b 001b1680
	s_wait_alu depctr_va_sdst(0)                               // 000000002c3c: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v134, s11             // 000000002c40: d5207c86 002f0c23
	v_add_co_u32 v135, s11, s34, v135                          // 000000002c48: d7000b87 02030e22
	v_cndmask_b32_e64 v140, 0, v142, s6                        // 000000002c50: d501008c 001b1c80
	s_wait_alu depctr_va_sdst(0)                               // 000000002c58: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v136, s11             // 000000002c5c: d5207c88 002f1023
	v_add_co_u32 v137, s11, s34, v137                          // 000000002c64: d7000b89 02031222
	s_wait_alu depctr_va_sdst(0)                               // 000000002c6c: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s35, v138, s11             // 000000002c70: d5207c8a 002f1423
	v_add_co_u32 v139, s11, s34, v139                          // 000000002c78: d7000b8b 02031622
	s_wait_alu depctr_va_sdst(0)                               // 000000002c80: bf88f19f
	v_add_co_ci_u32_e64 v140, null, s35, v140, s11             // 000000002c84: d5207c8c 002f1823
	v_or_b16 v131.l, v44.l, v128.l                             // 000000002c8c: d7630083 0203012c
	s_clause 0x4                                               // 000000002c94: bf850004
	global_load_d16_u8 v44, v[135:136], off                    // 000000002c98: ee07807c 0000002c 00000087
	global_load_d16_u8 v128, v[133:134], off                   // 000000002ca4: ee07807c 00000080 00000085
	global_load_d16_u8 v129, v[129:130], off                   // 000000002cb0: ee07807c 00000081 00000081
	global_load_d16_hi_u8 v129, v[137:138], off                // 000000002cbc: ee08407c 00000081 00000089
	global_load_d16_u8 v130, v[139:140], off                   // 000000002cc8: ee07807c 00000082 0000008b
	v_cndmask_b32_e64 v133, 0, v142, s5                        // 000000002cd4: d5010085 00171c80
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[42:43], v[131:132], v[24:31]// 000000002cdc: cc464018 1c63072a
	s_wait_loadcnt 0x4                                         // 000000002ce4: bfc00004
	v_cndmask_b16 v44.l, 0, v44.l, s10                         // 000000002ce8: d65d002c 002a5880
	s_wait_loadcnt 0x3                                         // 000000002cf0: bfc00003
	v_cndmask_b16 v128.l, 0, v128.l, s9                        // 000000002cf4: d65d0080 00270080
	s_wait_loadcnt 0x1                                         // 000000002cfc: bfc00001
	v_cndmask_b16 v129.l, 0, v129.l, s8                        // 000000002d00: d65d0081 00230280
	v_cndmask_b16 v129.h, 0, v129.h, s7                        // 000000002d08: d65d5081 001f0280
	v_lshlrev_b16 v44.l, 8, v44.l                              // 000000002d10: d738002c 02025888
	v_and_b16 v128.l, 0xff, v128.l                             // 000000002d18: d7620080 020300ff 000000ff
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000002d24: bf870214
	v_lshlrev_b16 v129.l, 8, v129.l                            // 000000002d28: d7380081 02030288
	v_and_b16 v129.h, 0xff, v129.h op_sel:[0,1,1]              // 000000002d30: d7625081 020302ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002d3c: bf870113
	v_or_b16 v134.h, v128.l, v44.l op_sel:[0,0,1]              // 000000002d40: d7634086 02025980
	v_or_b16 v134.l, v129.h, v129.l op_sel:[1,0,0]             // 000000002d48: d7630886 02030381
	v_or_b32_e32 v129, 2, v141                                 // 000000002d50: 39031a82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002d54: bf870091
	v_cndmask_b32_e64 v129, 0, v129, s5                        // 000000002d58: d5010081 00170280
	v_add_co_u32 v135, s7, s34, v129                           // 000000002d60: d7000787 02030222
	v_or_b32_e32 v129, 1, v141                                 // 000000002d68: 39031a81
	s_wait_alu depctr_va_sdst(0)                               // 000000002d6c: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v133, s7              // 000000002d70: d5207c88 001f0a23
	v_cndmask_b32_e64 v133, 0, v142, s4                        // 000000002d78: d5010085 00131c80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000002d80: bf870093
	v_cndmask_b32_e64 v129, 0, v129, s4                        // 000000002d84: d5010081 00130280
	v_add_co_u32 v137, s7, s34, v129                           // 000000002d8c: d7000789 02030222
	s_wait_alu depctr_va_sdst(0)                               // 000000002d94: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002d98: bf870003
	v_add_co_ci_u32_e64 v138, null, s35, v133, s7              // 000000002d9c: d5207c8a 001f0a23
	s_clause 0x1                                               // 000000002da4: bf850001
	global_load_d16_u8 v44, v[135:136], off                    // 000000002da8: ee07807c 0000002c 00000087
	global_load_d16_u8 v128, v[137:138], off                   // 000000002db4: ee07807c 00000080 00000089
	s_wait_loadcnt 0x2                                         // 000000002dc0: bfc00002
	v_cndmask_b16 v129.l, 0, v130.l, s6                        // 000000002dc4: d65d0081 001b0480
	s_delay_alu instid0(valu_dep_1)                            // 000000002dcc: bf870001
	v_lshlrev_b16 v129.l, 8, v129.l                            // 000000002dd0: d7380081 02030288
	s_wait_loadcnt 0x1                                         // 000000002dd8: bfc00001
	v_cndmask_b16 v44.l, 0, v44.l, s5                          // 000000002ddc: d65d002c 00165880
	s_and_b32 s5, s0, vcc_lo                                   // 000000002de4: 8b056a00
	s_and_b32 vcc_lo, s3, vcc_lo                               // 000000002de8: 8b6a6a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dec: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v142, s5                        // 000000002df0: d5010082 00171c80
	v_cndmask_b32_e32 v46, 0, v46, vcc_lo                      // 000000002df8: 025c5c80
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002dfc: d762002c 020258ff 000000ff
	s_wait_loadcnt 0x0                                         // 000000002e08: bfc00000
	v_cndmask_b16 v128.l, 0, v128.l, s4                        // 000000002e0c: d65d0080 00130080
	v_cndmask_b32_e32 v47, 0, v47, vcc_lo                      // 000000002e14: 025e5e80
	v_add_co_u32 v46, s4, s30, v46                             // 000000002e18: d700042e 02025c1e
	v_or_b16 v133.h, v44.l, v129.l op_sel:[0,0,1]              // 000000002e20: d7634085 0203032c
	v_cndmask_b32_e64 v129, 0, v141, s5                        // 000000002e28: d5010081 00171a80
	v_lshlrev_b16 v128.l, 8, v128.l                            // 000000002e30: d7380080 02030088
	s_wait_alu depctr_va_sdst(0)                               // 000000002e38: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s31, v47, s4                // 000000002e3c: d5207c2f 00125e1f
	s_delay_alu instid0(valu_dep_3)                            // 000000002e44: bf870003
	v_add_co_u32 v129, s6, s34, v129                           // 000000002e48: d7000681 02030222
	s_wait_alu depctr_va_sdst(0)                               // 000000002e50: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s35, v130, s6              // 000000002e54: d5207c82 001b0423
	global_load_d16_u8 v44, v[129:130], off                    // 000000002e5c: ee07807c 0000002c 00000081
	s_wait_loadcnt 0x0                                         // 000000002e68: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, s5                          // 000000002e6c: d65d002c 00165880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002e74: bf870091
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002e78: d762002c 020258ff 000000ff
	v_or_b16 v133.l, v44.l, v128.l                             // 000000002e84: d7630085 0203012c
	global_load_d16_u8 v44, v[46:47], off                      // 000000002e8c: ee07807c 0000002c 0000002e
	v_cndmask_b16 v46.l, 0, v128.h, s13                        // 000000002e98: d65d102e 00370080
	v_add_co_u32 v47, s11, s68, v76                            // 000000002ea0: d7000b2f 02029844
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[42:43], v[133:134], v[16:23]// 000000002ea8: cc464010 1c430b2a
	v_mov_b32_e32 v43, s67                                     // 000000002eb0: 7e560243
	v_or_b32_e32 v42, s68, v38                                 // 000000002eb4: 38544c44
	v_lshlrev_b16 v46.l, 8, v46.l                              // 000000002eb8: d738002e 02025c88
	s_delay_alu instid0(valu_dep_2)                            // 000000002ec0: bf870002
	v_cmp_gt_i64_e64 s8, s[52:53], v[42:43]                    // 000000002ec4: d4540008 02025434
	v_cmp_gt_i64_e64 s4, s[44:45], v[42:43]                    // 000000002ecc: d4540004 0202542c
	v_cmp_gt_i64_e64 s5, s[46:47], v[42:43]                    // 000000002ed4: d4540005 0202542e
	v_cmp_gt_i64_e64 s6, s[48:49], v[42:43]                    // 000000002edc: d4540006 02025430
	v_cmp_gt_i64_e64 s7, s[50:51], v[42:43]                    // 000000002ee4: d4540007 02025432
	v_cmp_gt_i64_e64 s9, s[54:55], v[42:43]                    // 000000002eec: d4540009 02025436
	v_cmp_gt_i64_e64 s10, s[56:57], v[42:43]                   // 000000002ef4: d454000a 02025438
	s_wait_loadcnt 0x0                                         // 000000002efc: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, vcc_lo                      // 000000002f00: d65d002c 01aa5880
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[42:43]                // 000000002f08: 7cb85418
	v_or_b32_e32 v42, 5, v47                                   // 000000002f0c: 38545e85
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000002f10: bf870093
	v_and_b16 v44.l, 0xff, v44.l                               // 000000002f14: d762002c 020258ff 000000ff
	v_or_b16 v44.l, v44.l, v46.l                               // 000000002f20: d763002c 02025d2c
	s_delay_alu instid0(valu_dep_1)                            // 000000002f28: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[44:45], v[133:134], v[0:7]// 000000002f2c: cc464000 1c030b2c
	s_wait_alu depctr_va_sdst(0)                               // 000000002f34: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s67, v74, s11              // 000000002f38: d5207c86 002e9443
	s_and_b32 s11, s2, s8                                      // 000000002f40: 8b0b0802
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[44:45], v[131:132], v[8:15]// 000000002f44: cc464008 1c23072c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f4c: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s11                         // 000000002f50: d501002a 002e5480
	v_cndmask_b32_e64 v44, 0, v134, s11                        // 000000002f58: d501002c 002f0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002f60: bf8701b2
	v_add_co_u32 v43, s12, s30, v42                            // 000000002f64: d7000c2b 0202541e
	v_or_b32_e32 v42, 6, v47                                   // 000000002f6c: 38545e86
	s_wait_alu depctr_va_sdst(0)                               // 000000002f70: bf88f19f
	v_add_co_ci_u32_e64 v44, null, s31, v44, s12               // 000000002f74: d5207c2c 0032581f
	s_and_b32 s12, s2, s9                                      // 000000002f7c: 8b0c0902
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f80: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s12                         // 000000002f84: d501002a 00325480
	v_cndmask_b32_e64 v46, 0, v134, s12                        // 000000002f8c: d501002e 00330c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002f94: bf8701b2
	v_add_co_u32 v45, s13, s30, v42                            // 000000002f98: d7000d2d 0202541e
	v_or_b32_e32 v42, 7, v47                                   // 000000002fa0: 38545e87
	s_wait_alu depctr_va_sdst(0)                               // 000000002fa4: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s13               // 000000002fa8: d5207c2e 00365c1f
	s_and_b32 s13, s2, s10                                     // 000000002fb0: 8b0d0a02
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fb4: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s13                         // 000000002fb8: d501002a 00365480
	v_cndmask_b32_e64 v129, 0, v134, s13                       // 000000002fc0: d5010081 00370c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002fc8: bf8701b2
	v_add_co_u32 v128, s14, s30, v42                           // 000000002fcc: d7000e80 0202541e
	v_or_b32_e32 v42, 4, v47                                   // 000000002fd4: 38545e84
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd8: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s14             // 000000002fdc: d5207c81 003b021f
	s_and_b32 s14, s2, s7                                      // 000000002fe4: 8b0e0702
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fe8: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s14                         // 000000002fec: d501002a 003a5480
	v_cndmask_b32_e64 v131, 0, v134, s14                       // 000000002ff4: d5010083 003b0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000002ffc: bf8701b2
	v_add_co_u32 v130, s15, s30, v42                           // 000000003000: d7000f82 0202541e
	v_or_b32_e32 v42, 3, v47                                   // 000000003008: 38545e83
	s_wait_alu depctr_va_sdst(0)                               // 00000000300c: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s31, v131, s15             // 000000003010: d5207c83 003f061f
	s_and_b32 s15, s2, s6                                      // 000000003018: 8b0f0602
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301c: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s15                         // 000000003020: d501002a 003e5480
	v_cndmask_b32_e64 v133, 0, v134, s15                       // 000000003028: d5010085 003f0c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003030: bf870122
	v_add_co_u32 v132, s16, s30, v42                           // 000000003034: d7001084 0202541e
	s_wait_alu depctr_va_sdst(0)                               // 00000000303c: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s31, v133, s16             // 000000003040: d5207c85 00430a1f
	s_clause 0x4                                               // 000000003048: bf850004
	global_load_d16_u8 v42, v[128:129], off                    // 00000000304c: ee07807c 0000002a 00000080
	global_load_d16_hi_u8 v42, v[45:46], off                   // 000000003058: ee08407c 0000002a 0000002d
	global_load_d16_u8 v43, v[43:44], off                      // 000000003064: ee07807c 0000002b 0000002b
	global_load_d16_hi_u8 v43, v[130:131], off                 // 000000003070: ee08407c 0000002b 00000082
	global_load_d16_u8 v44, v[132:133], off                    // 00000000307c: ee07807c 0000002c 00000084
	s_wait_loadcnt 0x3                                         // 000000003088: bfc00003
	v_cndmask_b16 v42.l, 0, v42.l, s13                         // 00000000308c: d65d002a 00365480
	v_cndmask_b16 v42.h, 0, v42.h, s12                         // 000000003094: d65d502a 00325480
	s_wait_loadcnt 0x1                                         // 00000000309c: bfc00001
	v_cndmask_b16 v43.l, 0, v43.l, s11                         // 0000000030a0: d65d002b 002e5680
	v_cndmask_b16 v43.h, 0, v43.h, s14                         // 0000000030a8: d65d502b 003a5680
	s_and_b32 s11, s2, s5                                      // 0000000030b0: 8b0b0502
	v_lshlrev_b16 v42.l, 8, v42.l                              // 0000000030b4: d738002a 02025488
	v_and_b16 v42.h, 0xff, v42.h op_sel:[0,1,1]                // 0000000030bc: d762502a 020254ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000030c8: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v134, s11                        // 0000000030cc: d501002e 002f0c80
	s_wait_loadcnt 0x0                                         // 0000000030d4: bfc00000
	v_and_b16 v44.h, 0xff, v43.h op_sel:[0,1,1]                // 0000000030d8: d762502c 020256ff 000000ff
	v_lshlrev_b16 v43.l, 8, v43.l                              // 0000000030e4: d738002b 02025688
	s_and_b32 s14, s3, s10                                     // 0000000030ec: 8b0e0a03
	v_or_b16 v43.h, v42.h, v42.l op_sel:[1,0,1]                // 0000000030f0: d763482b 0202552a
	v_or_b32_e32 v42, 2, v47                                   // 0000000030f8: 38545e82
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000030fc: bf870113
	v_or_b16 v43.l, v44.h, v43.l op_sel:[1,0,0]                // 000000003100: d763082b 0202572c
	v_cndmask_b32_e64 v42, 0, v42, s11                         // 000000003108: d501002a 002e5480
	s_delay_alu instid0(valu_dep_1)                            // 000000003110: bf870001
	v_add_co_u32 v45, s12, s30, v42                            // 000000003114: d7000c2d 0202541e
	v_or_b32_e32 v42, 1, v47                                   // 00000000311c: 38545e81
	s_wait_alu depctr_va_sdst(0)                               // 000000003120: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s12               // 000000003124: d5207c2e 00325c1f
	s_and_b32 s12, s2, s4                                      // 00000000312c: 8b0c0402
	s_wait_alu depctr_sa_sdst(0)                               // 000000003130: bf88ff9e
	v_cndmask_b32_e64 v42, 0, v42, s12                         // 000000003134: d501002a 00325480
	v_cndmask_b32_e64 v129, 0, v134, s12                       // 00000000313c: d5010081 00330c80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003144: bf870122
	v_add_co_u32 v128, s13, s30, v42                           // 000000003148: d7000d80 0202541e
	s_wait_alu depctr_va_sdst(0)                               // 000000003150: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s13             // 000000003154: d5207c81 0037021f
	s_clause 0x1                                               // 00000000315c: bf850001
	global_load_d16_u8 v42, v[45:46], off                      // 000000003160: ee07807c 0000002a 0000002d
	global_load_d16_hi_u8 v44, v[128:129], off                 // 00000000316c: ee08407c 0000002c 00000080
	s_wait_loadcnt 0x0                                         // 000000003178: bfc00000
	v_cndmask_b16 v42.h, 0, v44.l, s15                         // 00000000317c: d65d402a 003e5880
	s_and_b32 s15, s3, s7                                      // 000000003184: 8b0f0703
	s_delay_alu instid0(valu_dep_1)                            // 000000003188: bf870001
	v_lshlrev_b16 v42.h, 8, v42.h op_sel:[0,1,1]               // 00000000318c: d738502a 02025488
	v_cndmask_b16 v42.l, 0, v42.l, s11                         // 000000003194: d65d002a 002e5480
	s_and_b32 s11, s2, vcc_lo                                  // 00000000319c: 8b0b6a02
	v_cndmask_b16 v44.l, 0, v44.h, s12                         // 0000000031a0: d65d102c 00325880
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031a8: bf88ff9e
	v_cndmask_b32_e64 v45, 0, v47, s11                         // 0000000031ac: d501002d 002e5e80
	v_cndmask_b32_e64 v46, 0, v134, s11                        // 0000000031b4: d501002e 002f0c80
	v_and_b16 v42.l, 0xff, v42.l                               // 0000000031bc: d762002a 020254ff 000000ff
	s_and_b32 s12, s3, s8                                      // 0000000031c8: 8b0c0803
	v_lshlrev_b16 v44.l, 8, v44.l                              // 0000000031cc: d738002c 02025888
	v_add_co_u32 v45, s13, s30, v45                            // 0000000031d4: d7000d2d 02025a1e
	s_wait_alu depctr_va_sdst(0)                               // 0000000031dc: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s13               // 0000000031e0: d5207c2e 00365c1f
	v_or_b16 v42.h, v42.l, v42.h op_sel:[0,1,1]                // 0000000031e8: d763502a 0202552a
	s_and_b32 s13, s3, s9                                      // 0000000031f0: 8b0d0903
	global_load_d16_u8 v42, v[45:46], off                      // 0000000031f4: ee07807c 0000002a 0000002d
	s_wait_loadcnt 0x0                                         // 000000003200: bfc00000
	v_cndmask_b16 v42.l, 0, v42.l, s11                         // 000000003204: d65d002a 002e5480
	v_add_co_u32 v128, s11, s68, v78                           // 00000000320c: d7000b80 02029c44
	s_wait_alu depctr_va_sdst(0)                               // 000000003214: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s67, v75, s11              // 000000003218: d5207c81 002e9643
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000003220: bf870123
	v_and_b16 v42.l, 0xff, v42.l                               // 000000003224: d762002a 020254ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003230: bf88ff9e
	v_cndmask_b32_e64 v46, 0, v129, s12                        // 000000003234: d501002e 00330280
	v_cndmask_b32_e64 v47, 0, v129, s13                        // 00000000323c: d501002f 00370280
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_1)// 000000003244: bf8700a3
	v_or_b16 v42.l, v42.l, v44.l                               // 000000003248: d763002a 0202592a
	v_or_b32_e32 v44, 5, v128                                  // 000000003250: 38590085
	v_cndmask_b32_e64 v44, 0, v44, s12                         // 000000003254: d501002c 00325880
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_2)// 00000000325c: bf870141
	v_add_co_u32 v45, s11, s30, v44                            // 000000003260: d7000b2d 0202581e
	v_or_b32_e32 v44, 6, v128                                  // 000000003268: 38590086
	s_wait_alu depctr_va_sdst(0)                               // 00000000326c: bf88f19f
	v_add_co_ci_u32_e64 v46, null, s31, v46, s11               // 000000003270: d5207c2e 002e5c1f
	v_cndmask_b32_e64 v44, 0, v44, s13                         // 000000003278: d501002c 00365880
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003280: bf8701d1
	v_add_co_u32 v130, s11, s30, v44                           // 000000003284: d7000b82 0202581e
	v_or_b32_e32 v44, 7, v128                                  // 00000000328c: 38590087
	s_wait_alu depctr_va_sdst(0)                               // 000000003290: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s31, v47, s11              // 000000003294: d5207c83 002e5e1f
	v_cndmask_b32_e64 v47, 0, v129, s14                        // 00000000329c: d501002f 003b0280
	v_cndmask_b32_e64 v44, 0, v44, s14                         // 0000000032a4: d501002c 003a5880
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 0000000032ac: bf8701d1
	v_add_co_u32 v132, s11, s30, v44                           // 0000000032b0: d7000b84 0202581e
	v_or_b32_e32 v44, 4, v128                                  // 0000000032b8: 38590084
	s_wait_alu depctr_va_sdst(0)                               // 0000000032bc: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s31, v47, s11              // 0000000032c0: d5207c85 002e5e1f
	v_cndmask_b32_e64 v47, 0, v129, s15                        // 0000000032c8: d501002f 003f0280
	v_cndmask_b32_e64 v44, 0, v44, s15                         // 0000000032d0: d501002c 003e5880
	s_delay_alu instid0(valu_dep_1)                            // 0000000032d8: bf870001
	v_add_co_u32 v134, s11, s30, v44                           // 0000000032dc: d7000b86 0202581e
	v_or_b32_e32 v44, 3, v128                                  // 0000000032e4: 38590083
	s_wait_alu depctr_va_sdst(0)                               // 0000000032e8: bf88f19f
	v_add_co_ci_u32_e64 v135, null, s31, v47, s11              // 0000000032ec: d5207c87 002e5e1f
	s_and_b32 s11, s3, s6                                      // 0000000032f4: 8b0b0603
	s_wait_alu depctr_sa_sdst(0)                               // 0000000032f8: bf88ff9e
	v_cndmask_b32_e64 v44, 0, v44, s11                         // 0000000032fc: d501002c 002e5880
	v_cndmask_b32_e64 v47, 0, v129, s11                        // 000000003304: d501002f 002f0280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000330c: bf870122
	v_add_co_u32 v136, s16, s30, v44                           // 000000003310: d7001088 0202581e
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: bf88f19f
	v_add_co_ci_u32_e64 v137, null, s31, v47, s16              // 00000000331c: d5207c89 00425e1f
	s_clause 0x4                                               // 000000003324: bf850004
	global_load_d16_u8 v44, v[132:133], off                    // 000000003328: ee07807c 0000002c 00000084
	global_load_d16_hi_u8 v44, v[130:131], off                 // 000000003334: ee08407c 0000002c 00000082
	global_load_d16_u8 v45, v[45:46], off                      // 000000003340: ee07807c 0000002d 0000002d
	global_load_d16_hi_u8 v45, v[134:135], off                 // 00000000334c: ee08407c 0000002d 00000086
	global_load_d16_u8 v130, v[136:137], off                   // 000000003358: ee07807c 00000082 00000088
	s_wait_loadcnt 0x3                                         // 000000003364: bfc00003
	v_cndmask_b16 v44.l, 0, v44.l, s14                         // 000000003368: d65d002c 003a5880
	v_cndmask_b16 v44.h, 0, v44.h, s13                         // 000000003370: d65d502c 00365880
	s_wait_loadcnt 0x1                                         // 000000003378: bfc00001
	v_cndmask_b16 v45.l, 0, v45.l, s12                         // 00000000337c: d65d002d 00325a80
	v_cndmask_b16 v45.h, 0, v45.h, s15                         // 000000003384: d65d502d 003e5a80
	s_and_b32 s12, s3, s5                                      // 00000000338c: 8b0c0503
	v_lshlrev_b16 v44.l, 8, v44.l                              // 000000003390: d738002c 02025888
	v_and_b16 v44.h, 0xff, v44.h op_sel:[0,1,1]                // 000000003398: d762502c 020258ff 000000ff
	v_lshlrev_b16 v45.l, 8, v45.l                              // 0000000033a4: d738002d 02025a88
	v_and_b16 v46.l, 0xff, v45.h op_sel:[0,1,0]                // 0000000033ac: d762102e 02025aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000033b8: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v129, s12                        // 0000000033bc: d501002f 00330280
	v_or_b16 v45.h, v44.h, v44.l op_sel:[1,0,1]                // 0000000033c4: d763482d 0202592c
	v_or_b32_e32 v44, 2, v128                                  // 0000000033cc: 38590082
	v_or_b16 v45.l, v46.l, v45.l                               // 0000000033d0: d763002d 02025b2e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 0000000033d8: bf870092
	v_cndmask_b32_e64 v44, 0, v44, s12                         // 0000000033dc: d501002c 00325880
	v_add_co_u32 v46, s13, s30, v44                            // 0000000033e4: d7000d2e 0202581e
	v_or_b32_e32 v44, 1, v128                                  // 0000000033ec: 38590081
	s_wait_alu depctr_va_sdst(0)                               // 0000000033f0: bf88f19f
	v_add_co_ci_u32_e64 v47, null, s31, v47, s13               // 0000000033f4: d5207c2f 00365e1f
	s_and_b32 s13, s3, s4                                      // 0000000033fc: 8b0d0403
	s_wait_alu depctr_sa_sdst(0)                               // 000000003400: bf88ff9e
	v_cndmask_b32_e64 v44, 0, v44, s13                         // 000000003404: d501002c 00365880
	v_cndmask_b32_e64 v132, 0, v129, s13                       // 00000000340c: d5010084 00370280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003414: bf870122
	v_add_co_u32 v131, s14, s30, v44                           // 000000003418: d7000e83 0202581e
	s_wait_alu depctr_va_sdst(0)                               // 000000003420: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s31, v132, s14             // 000000003424: d5207c84 003b081f
	s_clause 0x1                                               // 00000000342c: bf850001
	global_load_d16_u8 v44, v[46:47], off                      // 000000003430: ee07807c 0000002c 0000002e
	global_load_d16_u8 v46, v[131:132], off                    // 00000000343c: ee07807c 0000002e 00000083
	s_wait_loadcnt 0x1                                         // 000000003448: bfc00001
	v_cndmask_b16 v44.h, 0, v130.l, s11                        // 00000000344c: d65d402c 002f0480
	v_add_co_u32 v140, s11, s68, v81                           // 000000003454: d7000b8c 0202a244
	s_wait_alu depctr_va_sdst(0)                               // 00000000345c: bf88f19f
	v_add_co_ci_u32_e64 v141, null, s67, v80, s11              // 000000003460: d5207c8d 002ea043
	s_and_b32 s11, s1, s8                                      // 000000003468: 8b0b0801
	v_or_b32_e32 v47, 5, v140                                  // 00000000346c: 385f1885
	v_lshlrev_b16 v44.h, 8, v44.h op_sel:[0,1,1]               // 000000003470: d738502c 02025888
	s_wait_alu depctr_sa_sdst(0)                               // 000000003478: bf88ff9e
	v_cndmask_b32_e64 v131, 0, v141, s11                       // 00000000347c: d5010083 002f1a80
	s_and_b32 s8, s0, s8                                       // 000000003484: 8b080800
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 000000003488: d501002f 002e5e80
	v_cndmask_b16 v44.l, 0, v44.l, s12                         // 000000003490: d65d002c 00325880
	s_delay_alu instid0(valu_dep_2)                            // 000000003498: bf870002
	v_add_co_u32 v130, s12, s34, v47                           // 00000000349c: d7000c82 02025e22
	v_or_b32_e32 v47, 6, v140                                  // 0000000034a4: 385f1886
	s_wait_alu depctr_va_sdst(0)                               // 0000000034a8: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s35, v131, s12             // 0000000034ac: d5207c83 00330623
	s_and_b32 s12, s1, s9                                      // 0000000034b4: 8b0c0901
	v_and_b16 v44.l, 0xff, v44.l                               // 0000000034b8: d762002c 020258ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034c4: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s12                         // 0000000034c8: d501002f 00325e80
	v_cndmask_b32_e64 v133, 0, v141, s12                       // 0000000034d0: d5010085 00331a80
	s_and_b32 s9, s0, s9                                       // 0000000034d8: 8b090900
	s_wait_loadcnt 0x0                                         // 0000000034dc: bfc00000
	v_cndmask_b16 v46.l, 0, v46.l, s13                         // 0000000034e0: d65d002e 00365c80
	v_or_b16 v44.h, v44.l, v44.h op_sel:[0,1,1]                // 0000000034e8: d763502c 0202592c
	v_add_co_u32 v132, s14, s34, v47                           // 0000000034f0: d7000e84 02025e22
	v_or_b32_e32 v47, 7, v140                                  // 0000000034f8: 385f1887
	s_wait_alu depctr_va_sdst(0)                               // 0000000034fc: bf88f19f
	v_add_co_ci_u32_e64 v133, null, s35, v133, s14             // 000000003500: d5207c85 003b0a23
	s_and_b32 s14, s1, s10                                     // 000000003508: 8b0e0a01
	s_and_b32 s10, s0, s10                                     // 00000000350c: 8b0a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003510: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s14                         // 000000003514: d501002f 003a5e80
	v_cndmask_b32_e64 v135, 0, v141, s14                       // 00000000351c: d5010087 003b1a80
	v_lshlrev_b16 v46.l, 8, v46.l                              // 000000003524: d738002e 02025c88
	s_delay_alu instid0(valu_dep_3)                            // 00000000352c: bf870003
	v_add_co_u32 v134, s15, s34, v47                           // 000000003530: d7000f86 02025e22
	v_or_b32_e32 v47, 4, v140                                  // 000000003538: 385f1884
	s_wait_alu depctr_va_sdst(0)                               // 00000000353c: bf88f19f
	v_add_co_ci_u32_e64 v135, null, s35, v135, s15             // 000000003540: d5207c87 003f0e23
	s_and_b32 s15, s1, s7                                      // 000000003548: 8b0f0701
	s_and_b32 s7, s0, s7                                       // 00000000354c: 8b070700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003550: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s15                         // 000000003554: d501002f 003e5e80
	v_cndmask_b32_e64 v137, 0, v141, s15                       // 00000000355c: d5010089 003f1a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000003564: bf8701b2
	v_add_co_u32 v136, s16, s34, v47                           // 000000003568: d7001088 02025e22
	v_or_b32_e32 v47, 3, v140                                  // 000000003570: 385f1883
	s_wait_alu depctr_va_sdst(0)                               // 000000003574: bf88f19f
	v_add_co_ci_u32_e64 v137, null, s35, v137, s16             // 000000003578: d5207c89 00431223
	s_and_b32 s16, s1, s6                                      // 000000003580: 8b100601
	s_and_b32 s6, s0, s6                                       // 000000003584: 8b060600
	s_wait_alu depctr_sa_sdst(0)                               // 000000003588: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s16                         // 00000000358c: d501002f 00425e80
	v_cndmask_b32_e64 v139, 0, v141, s16                       // 000000003594: d501008b 00431a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000359c: bf870122
	v_add_co_u32 v138, s17, s34, v47                           // 0000000035a0: d700118a 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 0000000035a8: bf88f19f
	v_add_co_ci_u32_e64 v139, null, s35, v139, s17             // 0000000035ac: d5207c8b 00471623
	s_clause 0x4                                               // 0000000035b4: bf850004
	global_load_d16_u8 v44, v[134:135], off                    // 0000000035b8: ee07807c 0000002c 00000086
	global_load_d16_hi_u8 v46, v[132:133], off                 // 0000000035c4: ee08407c 0000002e 00000084
	global_load_d16_u8 v47, v[130:131], off                    // 0000000035d0: ee07807c 0000002f 00000082
	global_load_d16_hi_u8 v47, v[136:137], off                 // 0000000035dc: ee08407c 0000002f 00000088
	global_load_d16_u8 v130, v[138:139], off                   // 0000000035e8: ee07807c 00000082 0000008a
	s_wait_loadcnt 0x4                                         // 0000000035f4: bfc00004
	v_cndmask_b16 v44.l, 0, v44.l, s14                         // 0000000035f8: d65d002c 003a5880
	s_wait_loadcnt 0x3                                         // 000000003600: bfc00003
	v_cndmask_b16 v46.h, 0, v46.h, s12                         // 000000003604: d65d502e 00325c80
	s_wait_loadcnt 0x1                                         // 00000000360c: bfc00001
	v_cndmask_b16 v47.l, 0, v47.l, s11                         // 000000003610: d65d002f 002e5e80
	v_cndmask_b16 v47.h, 0, v47.h, s15                         // 000000003618: d65d502f 003e5e80
	s_and_b32 s11, s1, s5                                      // 000000003620: 8b0b0501
	v_lshlrev_b16 v44.l, 8, v44.l                              // 000000003624: d738002c 02025888
	s_wait_alu depctr_sa_sdst(0)                               // 00000000362c: bf88ff9e
	v_cndmask_b32_e64 v131, 0, v141, s11                       // 000000003630: d5010083 002f1a80
	v_lshlrev_b16 v47.l, 8, v47.l                              // 000000003638: d738002f 02025e88
	v_and_b16 v47.h, 0xff, v47.h op_sel:[0,1,1]                // 000000003640: d762502f 02025eff 000000ff
	v_and_b16 v46.h, 0xff, v46.h op_sel:[0,1,1]                // 00000000364c: d762502e 02025cff 000000ff
	s_and_b32 s5, s0, s5                                       // 000000003658: 8b050500
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 00000000365c: bf8701a2
	v_or_b16 v132.l, v47.h, v47.l op_sel:[1,0,0]               // 000000003660: d7630884 02025f2f
	v_or_b32_e32 v47, 2, v140                                  // 000000003668: 385f1882
	v_or_b16 v132.h, v46.h, v44.l op_sel:[1,0,1]               // 00000000366c: d7634884 0202592e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000003674: bf870092
	v_cndmask_b32_e64 v47, 0, v47, s11                         // 000000003678: d501002f 002e5e80
	v_add_co_u32 v133, s12, s34, v47                           // 000000003680: d7000c85 02025e22
	v_or_b32_e32 v47, 1, v140                                  // 000000003688: 385f1881
	s_wait_alu depctr_va_sdst(0)                               // 00000000368c: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v131, s12             // 000000003690: d5207c86 00330623
	s_and_b32 s12, s1, s4                                      // 000000003698: 8b0c0401
	s_and_b32 s4, s0, s4                                       // 00000000369c: 8b040400
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036a0: bf88ff9e
	v_cndmask_b32_e64 v47, 0, v47, s12                         // 0000000036a4: d501002f 00325e80
	v_cndmask_b32_e64 v131, 0, v141, s12                       // 0000000036ac: d5010083 00331a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000036b4: bf870122
	v_add_co_u32 v135, s14, s34, v47                           // 0000000036b8: d7000e87 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 0000000036c0: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v131, s14             // 0000000036c4: d5207c88 003b0623
	s_clause 0x1                                               // 0000000036cc: bf850001
	global_load_d16_u8 v44, v[133:134], off                    // 0000000036d0: ee07807c 0000002c 00000085
	global_load_d16_hi_u8 v46, v[135:136], off                 // 0000000036dc: ee08407c 0000002e 00000087
	s_wait_loadcnt 0x2                                         // 0000000036e8: bfc00002
	v_cndmask_b16 v47.l, 0, v130.l, s16                        // 0000000036ec: d65d002f 00430480
	s_delay_alu instid0(valu_dep_1)                            // 0000000036f4: bf870001
	v_lshlrev_b16 v47.l, 8, v47.l                              // 0000000036f8: d738002f 02025e88
	s_wait_loadcnt 0x1                                         // 000000003700: bfc00001
	v_cndmask_b16 v44.l, 0, v44.l, s11                         // 000000003704: d65d002c 002e5880
	s_and_b32 s11, s1, vcc_lo                                  // 00000000370c: 8b0b6a01
	s_wait_loadcnt 0x0                                         // 000000003710: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, s12                         // 000000003714: d65d502e 00325c80
	s_wait_alu depctr_sa_sdst(0)                               // 00000000371c: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v141, s11                       // 000000003720: d5010082 002f1a80
	v_and_b16 v44.l, 0xff, v44.l                               // 000000003728: d762002c 020258ff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003734: bf870113
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000003738: d738502e 02025c88
	v_or_b16 v131.h, v44.l, v47.l op_sel:[0,0,1]               // 000000003740: d7634083 02025f2c
	v_cndmask_b32_e64 v47, 0, v140, s11                        // 000000003748: d501002f 002f1880
	s_delay_alu instid0(valu_dep_1)                            // 000000003750: bf870001
	v_add_co_u32 v133, s14, s34, v47                           // 000000003754: d7000e85 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 00000000375c: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v130, s14             // 000000003760: d5207c86 003b0423
	global_load_d16_u8 v44, v[133:134], off                    // 000000003768: ee07807c 0000002c 00000085
	s_wait_loadcnt 0x0                                         // 000000003774: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, s11                         // 000000003778: d65d002c 002e5880
	v_add_co_u32 v143, s11, s68, v85                           // 000000003780: d7000b8f 0202aa44
	s_wait_alu depctr_va_sdst(0)                               // 000000003788: bf88f19f
	v_add_co_ci_u32_e64 v144, null, s67, v83, s11              // 00000000378c: d5207c90 002ea643
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003794: bf870193
	v_and_b16 v44.l, 0xff, v44.l                               // 000000003798: d762002c 020258ff 000000ff
	v_or_b32_e32 v47, 5, v143                                  // 0000000037a4: 385f1e85
	s_add_nc_u64 s[66:67], s[66:67], 32                        // 0000000037a8: a9c2a042
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000037ac: bf870193
	v_cndmask_b32_e64 v130, 0, v144, s8                        // 0000000037b0: d5010082 00232080
	v_or_b16 v131.l, v44.l, v46.h op_sel:[0,1,0]               // 0000000037b8: d7631083 02025d2c
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000037c0: bf870113
	v_cndmask_b32_e64 v47, 0, v47, s8                          // 0000000037c4: d501002f 00225e80
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[42:43], v[131:132], v[24:31]// 0000000037cc: cc464018 1c63072a
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_3)// 0000000037d4: bf8701d2
	v_add_co_u32 v133, s11, s34, v47                           // 0000000037d8: d7000b85 02025e22
	v_or_b32_e32 v47, 6, v143                                  // 0000000037e0: 385f1e86
	s_wait_alu depctr_va_sdst(0)                               // 0000000037e4: bf88f19f
	v_add_co_ci_u32_e64 v134, null, s35, v130, s11             // 0000000037e8: d5207c86 002f0423
	v_cndmask_b32_e64 v130, 0, v144, s9                        // 0000000037f0: d5010082 00272080
	v_cndmask_b32_e64 v47, 0, v47, s9                          // 0000000037f8: d501002f 00265e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003800: bf8701d1
	v_add_co_u32 v135, s11, s34, v47                           // 000000003804: d7000b87 02025e22
	v_or_b32_e32 v47, 7, v143                                  // 00000000380c: 385f1e87
	s_wait_alu depctr_va_sdst(0)                               // 000000003810: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v130, s11             // 000000003814: d5207c88 002f0423
	v_cndmask_b32_e64 v130, 0, v144, s10                       // 00000000381c: d5010082 002b2080
	v_cndmask_b32_e64 v47, 0, v47, s10                         // 000000003824: d501002f 002a5e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 00000000382c: bf8701d1
	v_add_co_u32 v137, s11, s34, v47                           // 000000003830: d7000b89 02025e22
	v_or_b32_e32 v47, 4, v143                                  // 000000003838: 385f1e84
	s_wait_alu depctr_va_sdst(0)                               // 00000000383c: bf88f19f
	v_add_co_ci_u32_e64 v138, null, s35, v130, s11             // 000000003840: d5207c8a 002f0423
	v_cndmask_b32_e64 v130, 0, v144, s7                        // 000000003848: d5010082 001f2080
	v_cndmask_b32_e64 v47, 0, v47, s7                          // 000000003850: d501002f 001e5e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_3)// 000000003858: bf8701d1
	v_add_co_u32 v139, s11, s34, v47                           // 00000000385c: d7000b8b 02025e22
	v_or_b32_e32 v47, 3, v143                                  // 000000003864: 385f1e83
	s_wait_alu depctr_va_sdst(0)                               // 000000003868: bf88f19f
	v_add_co_ci_u32_e64 v140, null, s35, v130, s11             // 00000000386c: d5207c8c 002f0423
	v_cndmask_b32_e64 v130, 0, v144, s6                        // 000000003874: d5010082 001b2080
	v_cndmask_b32_e64 v47, 0, v47, s6                          // 00000000387c: d501002f 001a5e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_3)// 000000003884: bf8701a1
	v_add_co_u32 v141, s11, s34, v47                           // 000000003888: d7000b8d 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 000000003890: bf88f19f
	v_add_co_ci_u32_e64 v142, null, s35, v130, s11             // 000000003894: d5207c8e 002f0423
	s_clause 0x4                                               // 00000000389c: bf850004
	global_load_d16_u8 v44, v[137:138], off                    // 0000000038a0: ee07807c 0000002c 00000089
	global_load_d16_hi_u8 v46, v[135:136], off                 // 0000000038ac: ee08407c 0000002e 00000087
	global_load_d16_u8 v47, v[133:134], off                    // 0000000038b8: ee07807c 0000002f 00000085
	global_load_d16_hi_u8 v47, v[139:140], off                 // 0000000038c4: ee08407c 0000002f 0000008b
	global_load_d16_u8 v130, v[141:142], off                   // 0000000038d0: ee07807c 00000082 0000008d
	v_cndmask_b32_e64 v133, 0, v144, s5                        // 0000000038dc: d5010085 00172080
	s_wait_loadcnt 0x4                                         // 0000000038e4: bfc00004
	v_cndmask_b16 v44.l, 0, v44.l, s10                         // 0000000038e8: d65d002c 002a5880
	s_wait_loadcnt 0x3                                         // 0000000038f0: bfc00003
	v_cndmask_b16 v46.h, 0, v46.h, s9                          // 0000000038f4: d65d502e 00265c80
	s_wait_loadcnt 0x1                                         // 0000000038fc: bfc00001
	v_cndmask_b16 v47.l, 0, v47.l, s8                          // 000000003900: d65d002f 00225e80
	v_cndmask_b16 v47.h, 0, v47.h, s7                          // 000000003908: d65d502f 001e5e80
	v_lshlrev_b16 v44.l, 8, v44.l                              // 000000003910: d738002c 02025888
	v_and_b16 v46.h, 0xff, v46.h op_sel:[0,1,1]                // 000000003918: d762502e 02025cff 000000ff
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000003924: bf870214
	v_lshlrev_b16 v47.l, 8, v47.l                              // 000000003928: d738002f 02025e88
	v_and_b16 v47.h, 0xff, v47.h op_sel:[0,1,1]                // 000000003930: d762502f 02025eff 000000ff
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 00000000393c: bf870113
	v_or_b16 v134.h, v46.h, v44.l op_sel:[1,0,1]               // 000000003940: d7634886 0202592e
	v_or_b16 v134.l, v47.h, v47.l op_sel:[1,0,0]               // 000000003948: d7630886 02025f2f
	v_or_b32_e32 v47, 2, v143                                  // 000000003950: 385f1e82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003954: bf870091
	v_cndmask_b32_e64 v47, 0, v47, s5                          // 000000003958: d501002f 00165e80
	v_add_co_u32 v135, s7, s34, v47                            // 000000003960: d7000787 02025e22
	v_or_b32_e32 v47, 1, v143                                  // 000000003968: 385f1e81
	s_wait_alu depctr_va_sdst(0)                               // 00000000396c: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v133, s7              // 000000003970: d5207c88 001f0a23
	v_cndmask_b32_e64 v133, 0, v144, s4                        // 000000003978: d5010085 00132080
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 000000003980: bf870093
	v_cndmask_b32_e64 v47, 0, v47, s4                          // 000000003984: d501002f 00125e80
	v_add_co_u32 v137, s7, s34, v47                            // 00000000398c: d7000789 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 000000003994: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003998: bf870003
	v_add_co_ci_u32_e64 v138, null, s35, v133, s7              // 00000000399c: d5207c8a 001f0a23
	s_clause 0x1                                               // 0000000039a4: bf850001
	global_load_d16_u8 v44, v[135:136], off                    // 0000000039a8: ee07807c 0000002c 00000087
	global_load_d16_hi_u8 v46, v[137:138], off                 // 0000000039b4: ee08407c 0000002e 00000089
	s_wait_loadcnt 0x2                                         // 0000000039c0: bfc00002
	v_cndmask_b16 v47.l, 0, v130.l, s6                         // 0000000039c4: d65d002f 001b0480
	s_delay_alu instid0(valu_dep_1)                            // 0000000039cc: bf870001
	v_lshlrev_b16 v47.l, 8, v47.l                              // 0000000039d0: d738002f 02025e88
	s_wait_loadcnt 0x1                                         // 0000000039d8: bfc00001
	v_cndmask_b16 v44.l, 0, v44.l, s5                          // 0000000039dc: d65d002c 00165880
	s_and_b32 s5, s0, vcc_lo                                   // 0000000039e4: 8b056a00
	s_and_b32 vcc_lo, s3, vcc_lo                               // 0000000039e8: 8b6a6a03
	s_wait_alu depctr_sa_sdst(0)                               // 0000000039ec: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v144, s5                        // 0000000039f0: d5010082 00172080
	s_wait_loadcnt 0x0                                         // 0000000039f8: bfc00000
	v_cndmask_b16 v46.h, 0, v46.h, s4                          // 0000000039fc: d65d502e 00125c80
	v_and_b16 v44.l, 0xff, v44.l                               // 000000003a04: d762002c 020258ff 000000ff
	v_cndmask_b32_e32 v129, 0, v129, vcc_lo                    // 000000003a10: 03030280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003a14: bf870193
	v_lshlrev_b16 v46.h, 8, v46.h op_sel:[0,1,1]               // 000000003a18: d738502e 02025c88
	v_or_b16 v133.h, v44.l, v47.l op_sel:[0,0,1]               // 000000003a20: d7634085 02025f2c
	v_cndmask_b32_e64 v47, 0, v143, s5                         // 000000003a28: d501002f 00171e80
	s_delay_alu instid0(valu_dep_1)                            // 000000003a30: bf870001
	v_add_co_u32 v135, s6, s34, v47                            // 000000003a34: d7000687 02025e22
	s_wait_alu depctr_va_sdst(0)                               // 000000003a3c: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s35, v130, s6              // 000000003a40: d5207c88 001b0423
	v_cndmask_b32_e32 v47, 0, v128, vcc_lo                     // 000000003a48: 025f0080
	global_load_d16_u8 v44, v[135:136], off                    // 000000003a4c: ee07807c 0000002c 00000087
	v_add_co_u32 v128, s4, s30, v47                            // 000000003a58: d7000480 02025e1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a60: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s4              // 000000003a64: d5207c81 0013021f
	v_cmp_lt_u64_e64 s4, s[66:67], s[62:63]                    // 000000003a6c: d4590004 02007c42
	s_wait_loadcnt 0x0                                         // 000000003a74: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, s5                          // 000000003a78: d65d002c 00165880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003a80: bf870091
	v_and_b16 v44.l, 0xff, v44.l                               // 000000003a84: d762002c 020258ff 000000ff
	v_or_b16 v133.l, v44.l, v46.h op_sel:[0,1,0]               // 000000003a90: d7631085 02025d2c
	global_load_d16_u8 v44, v[128:129], off                    // 000000003a98: ee07807c 0000002c 00000080
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[42:43], v[133:134], v[16:23]// 000000003aa4: cc464010 1c430b2a
	s_wait_loadcnt 0x0                                         // 000000003aac: bfc00000
	v_cndmask_b16 v44.l, 0, v44.l, vcc_lo                      // 000000003ab0: d65d002c 01aa5880
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000003ab8: 8b6a047e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003abc: bf870091
	v_and_b16 v44.l, 0xff, v44.l                               // 000000003ac0: d762002c 020258ff 000000ff
	v_or_b16 v44.l, v44.l, v46.l                               // 000000003acc: d763002c 02025d2c
	s_delay_alu instid0(valu_dep_1)                            // 000000003ad4: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[44:45], v[131:132], v[8:15]// 000000003ad8: cc464008 1c23072c
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[44:45], v[133:134], v[0:7]// 000000003ae0: cc464000 1c030b2c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ae8: bf88ff9e
	s_cbranch_vccnz 64008                                      // 000000003aec: bfa4fa08 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x810>
	s_lshr_b64 s[4:5], s[64:65], 7                             // 000000003af0: 85848740
	s_wait_alu depctr_sa_sdst(0)                               // 000000003af4: bf88ff9e
	s_mul_u64 s[4:5], s[4:5], s[58:59]                         // 000000003af8: aa843a04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003afc: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000003b00: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b04: bf88ff9e
	s_add_nc_u64 s[4:5], s[28:29], s[4:5]                      // 000000003b08: a984041c
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b0c: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[60:61]                      // 000000003b10: a9863c04
	s_clause 0x1                                               // 000000003b14: bf850001
	s_load_b32 s6, s[6:7], 0x0                                 // 000000003b18: f4000183 f8000000
	s_load_b32 s7, s[4:5], 0x0                                 // 000000003b20: f40001c2 f8000000
	s_lshr_b64 s[4:5], s[64:65], 5                             // 000000003b28: 85848540
	s_mov_b64 s[64:65], s[62:63]                               // 000000003b2c: bec0013e
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b30: bf88ff9e
	v_add_co_u32 v42, vcc_lo, v88, s4                          // 000000003b34: d7006a2a 02000958
	s_wait_alu depctr_va_vcc(0)                                // 000000003b3c: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s5, v89, vcc_lo             // 000000003b40: d5207c2b 01aab205
	global_load_b32 v44, v[42:43], off                         // 000000003b48: ee05007c 0000002c 0000002a
	s_wait_kmcnt 0x0                                           // 000000003b54: bfc70000
	v_mov_b32_e32 v45, s6                                      // 000000003b58: 7e5a0206
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003b5c: bf8700a1
	v_cndmask_b32_e64 v46, s7, v45, s1                         // 000000003b60: d501002e 00065a07
	s_wait_loadcnt 0x0                                         // 000000003b68: bfc00000
	v_mul_f32_e32 v42, v44, v46                                // 000000003b6c: 10545d2c
	s_delay_alu instid0(valu_dep_1)                            // 000000003b70: bf870001
	v_mul_f32_e32 v24, v24, v42                                // 000000003b74: 10305518
	v_add_co_u32 v42, vcc_lo, v90, s4                          // 000000003b78: d7006a2a 0200095a
	s_wait_alu depctr_va_vcc(0)                                // 000000003b80: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s5, v91, vcc_lo             // 000000003b84: d5207c2b 01aab605
	global_load_b32 v42, v[42:43], off                         // 000000003b8c: ee05007c 0000002a 0000002a
	s_wait_loadcnt 0x0                                         // 000000003b98: bfc00000
	v_dual_add_f32 v117, v117, v24 :: v_dual_mul_f32 v24, v46, v42// 000000003b9c: c9063175 7518552e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ba4: bf870091
	v_mul_f32_e32 v24, v25, v24                                // 000000003ba8: 10303119
	v_add_f32_e32 v114, v114, v24                              // 000000003bac: 06e43172
	v_add_co_u32 v24, vcc_lo, v93, s4                          // 000000003bb0: d7006a18 0200095d
	s_wait_alu depctr_va_vcc(0)                                // 000000003bb8: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v94, vcc_lo             // 000000003bbc: d5207c19 01aabc05
	global_load_b32 v43, v[24:25], off                         // 000000003bc4: ee05007c 0000002b 00000018
	s_wait_loadcnt 0x0                                         // 000000003bd0: bfc00000
	v_mul_f32_e32 v24, v46, v43                                // 000000003bd4: 1030572e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003bd8: bf870091
	v_mul_f32_e32 v24, v26, v24                                // 000000003bdc: 1030311a
	v_add_f32_e32 v111, v111, v24                              // 000000003be0: 06de316f
	v_add_co_u32 v24, vcc_lo, v96, s4                          // 000000003be4: d7006a18 02000960
	s_wait_alu depctr_va_vcc(0)                                // 000000003bec: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v97, vcc_lo             // 000000003bf0: d5207c19 01aac205
	global_load_b32 v26, v[24:25], off                         // 000000003bf8: ee05007c 0000001a 00000018
	s_wait_loadcnt 0x0                                         // 000000003c04: bfc00000
	v_mul_f32_e32 v24, v46, v26                                // 000000003c08: 1030352e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c0c: bf870091
	v_mul_f32_e32 v24, v27, v24                                // 000000003c10: 1030311b
	v_add_f32_e32 v108, v108, v24                              // 000000003c14: 06d8316c
	v_add_co_u32 v24, vcc_lo, v98, s4                          // 000000003c18: d7006a18 02000962
	s_wait_alu depctr_va_vcc(0)                                // 000000003c20: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v99, vcc_lo             // 000000003c24: d5207c19 01aac605
	global_load_b32 v27, v[24:25], off                         // 000000003c2c: ee05007c 0000001b 00000018
	s_wait_loadcnt 0x0                                         // 000000003c38: bfc00000
	v_mul_f32_e32 v24, v46, v27                                // 000000003c3c: 1030372e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c40: bf870091
	v_mul_f32_e32 v24, v28, v24                                // 000000003c44: 1030311c
	v_add_f32_e32 v105, v105, v24                              // 000000003c48: 06d23169
	v_add_co_u32 v24, vcc_lo, v101, s4                         // 000000003c4c: d7006a18 02000965
	s_wait_alu depctr_va_vcc(0)                                // 000000003c54: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v102, vcc_lo            // 000000003c58: d5207c19 01aacc05
	global_load_b32 v28, v[24:25], off                         // 000000003c60: ee05007c 0000001c 00000018
	s_wait_loadcnt 0x0                                         // 000000003c6c: bfc00000
	v_mul_f32_e32 v24, v46, v28                                // 000000003c70: 1030392e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003c74: bf870091
	v_mul_f32_e32 v24, v29, v24                                // 000000003c78: 1030311d
	v_add_f32_e32 v100, v100, v24                              // 000000003c7c: 06c83164
	v_add_co_u32 v24, vcc_lo, v103, s4                         // 000000003c80: d7006a18 02000967
	s_wait_alu depctr_va_vcc(0)                                // 000000003c88: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v104, vcc_lo            // 000000003c8c: d5207c19 01aad005
	global_load_b32 v29, v[24:25], off                         // 000000003c94: ee05007c 0000001d 00000018
	s_wait_loadcnt 0x0                                         // 000000003ca0: bfc00000
	v_mul_f32_e32 v24, v46, v29                                // 000000003ca4: 10303b2e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ca8: bf870091
	v_mul_f32_e32 v24, v30, v24                                // 000000003cac: 1030311e
	v_add_f32_e32 v95, v95, v24                                // 000000003cb0: 06be315f
	v_add_co_u32 v24, vcc_lo, v106, s4                         // 000000003cb4: d7006a18 0200096a
	s_wait_alu depctr_va_vcc(0)                                // 000000003cbc: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v107, vcc_lo            // 000000003cc0: d5207c19 01aad605
	global_load_b32 v24, v[24:25], off                         // 000000003cc8: ee05007c 00000018 00000018
	s_wait_loadcnt 0x0                                         // 000000003cd4: bfc00000
	v_mul_f32_e32 v25, v46, v24                                // 000000003cd8: 1032312e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cdc: bf870091
	v_mul_f32_e32 v25, v31, v25                                // 000000003ce0: 1032331f
	v_add_f32_e32 v92, v92, v25                                // 000000003ce4: 06b8335c
	v_cndmask_b32_e64 v25, s7, v45, s0                         // 000000003ce8: d5010019 00025a07
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cf0: bf870091
	v_mul_f32_e32 v30, v44, v25                                // 000000003cf4: 103c332c
	v_mul_f32_e32 v16, v16, v30                                // 000000003cf8: 10203d10
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cfc: bf870091
	v_dual_add_f32 v71, v71, v16 :: v_dual_mul_f32 v16, v25, v42// 000000003d00: c9062147 47105519
	v_mul_f32_e32 v16, v17, v16                                // 000000003d08: 10202111
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d0c: bf8700a1
	v_add_f32_e32 v70, v70, v16                                // 000000003d10: 068c2146
	v_mul_f32_e32 v16, v25, v43                                // 000000003d14: 10205719
	v_mul_f32_e32 v16, v18, v16                                // 000000003d18: 10202112
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d1c: bf8700a1
	v_add_f32_e32 v69, v69, v16                                // 000000003d20: 068a2145
	v_mul_f32_e32 v16, v25, v26                                // 000000003d24: 10203519
	v_mul_f32_e32 v16, v19, v16                                // 000000003d28: 10202113
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d2c: bf8700a1
	v_add_f32_e32 v68, v68, v16                                // 000000003d30: 06882144
	v_mul_f32_e32 v16, v25, v27                                // 000000003d34: 10203719
	v_mul_f32_e32 v16, v20, v16                                // 000000003d38: 10202114
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d3c: bf8700a1
	v_add_f32_e32 v67, v67, v16                                // 000000003d40: 06862143
	v_mul_f32_e32 v16, v25, v28                                // 000000003d44: 10203919
	v_mul_f32_e32 v16, v21, v16                                // 000000003d48: 10202115
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d4c: bf8700a1
	v_add_f32_e32 v66, v66, v16                                // 000000003d50: 06842142
	v_mul_f32_e32 v16, v25, v29                                // 000000003d54: 10203b19
	v_mul_f32_e32 v16, v22, v16                                // 000000003d58: 10202116
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003d5c: bf8700a1
	v_add_f32_e32 v65, v65, v16                                // 000000003d60: 06822141
	v_mul_f32_e32 v16, v25, v24                                // 000000003d64: 10203119
	v_mul_f32_e32 v16, v23, v16                                // 000000003d68: 10202117
	s_delay_alu instid0(valu_dep_1)                            // 000000003d6c: bf870001
	v_add_f32_e32 v63, v63, v16                                // 000000003d70: 067e213f
	v_add_co_u32 v16, vcc_lo, v109, s4                         // 000000003d74: d7006a10 0200096d
	s_wait_alu depctr_va_vcc(0)                                // 000000003d7c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s5, v110, vcc_lo            // 000000003d80: d5207c11 01aadc05
	global_load_b32 v18, v[16:17], off                         // 000000003d88: ee05007c 00000012 00000010
	s_wait_loadcnt 0x0                                         // 000000003d94: bfc00000
	v_mul_f32_e32 v16, v46, v18                                // 000000003d98: 1020252e
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000003d9c: bf8701c1
	v_mul_f32_e32 v8, v8, v16                                  // 000000003da0: 10102108
	v_add_co_u32 v16, vcc_lo, v112, s4                         // 000000003da4: d7006a10 02000970
	s_wait_alu depctr_va_vcc(0)                                // 000000003dac: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s5, v113, vcc_lo            // 000000003db0: d5207c11 01aae205
	v_add_f32_e32 v87, v87, v8                                 // 000000003db8: 06ae1157
	global_load_b32 v16, v[16:17], off                         // 000000003dbc: ee05007c 00000010 00000010
	s_wait_loadcnt 0x0                                         // 000000003dc8: bfc00000
	v_mul_f32_e32 v8, v46, v16                                 // 000000003dcc: 1010212e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003dd0: bf870091
	v_mul_f32_e32 v8, v9, v8                                   // 000000003dd4: 10101109
	v_add_f32_e32 v86, v86, v8                                 // 000000003dd8: 06ac1156
	v_add_co_u32 v8, vcc_lo, v115, s4                          // 000000003ddc: d7006a08 02000973
	s_wait_alu depctr_va_vcc(0)                                // 000000003de4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v116, vcc_lo             // 000000003de8: d5207c09 01aae805
	global_load_b32 v17, v[8:9], off                           // 000000003df0: ee05007c 00000011 00000008
	s_wait_loadcnt 0x0                                         // 000000003dfc: bfc00000
	v_mul_f32_e32 v8, v46, v17                                 // 000000003e00: 1010232e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e04: bf870091
	v_mul_f32_e32 v8, v10, v8                                  // 000000003e08: 1010110a
	v_add_f32_e32 v84, v84, v8                                 // 000000003e0c: 06a81154
	v_add_co_u32 v8, vcc_lo, v118, s4                          // 000000003e10: d7006a08 02000976
	s_wait_alu depctr_va_vcc(0)                                // 000000003e18: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v119, vcc_lo             // 000000003e1c: d5207c09 01aaee05
	global_load_b32 v10, v[8:9], off                           // 000000003e24: ee05007c 0000000a 00000008
	s_wait_loadcnt 0x0                                         // 000000003e30: bfc00000
	v_mul_f32_e32 v8, v46, v10                                 // 000000003e34: 1010152e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e38: bf870091
	v_mul_f32_e32 v8, v11, v8                                  // 000000003e3c: 1010110b
	v_add_f32_e32 v82, v82, v8                                 // 000000003e40: 06a41152
	v_add_co_u32 v8, vcc_lo, v120, s4                          // 000000003e44: d7006a08 02000978
	s_wait_alu depctr_va_vcc(0)                                // 000000003e4c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v121, vcc_lo             // 000000003e50: d5207c09 01aaf205
	global_load_b32 v11, v[8:9], off                           // 000000003e58: ee05007c 0000000b 00000008
	s_wait_loadcnt 0x0                                         // 000000003e64: bfc00000
	v_mul_f32_e32 v8, v46, v11                                 // 000000003e68: 1010172e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003e6c: bf870091
	v_mul_f32_e32 v8, v12, v8                                  // 000000003e70: 1010110c
	v_add_f32_e32 v79, v79, v8                                 // 000000003e74: 069e114f
	v_add_co_u32 v8, vcc_lo, v122, s4                          // 000000003e78: d7006a08 0200097a
	s_wait_alu depctr_va_vcc(0)                                // 000000003e80: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v123, vcc_lo             // 000000003e84: d5207c09 01aaf605
	global_load_b32 v12, v[8:9], off                           // 000000003e8c: ee05007c 0000000c 00000008
	s_wait_loadcnt 0x0                                         // 000000003e98: bfc00000
	v_mul_f32_e32 v8, v46, v12                                 // 000000003e9c: 1010192e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ea0: bf870091
	v_mul_f32_e32 v8, v13, v8                                  // 000000003ea4: 1010110d
	v_add_f32_e32 v77, v77, v8                                 // 000000003ea8: 069a114d
	v_add_co_u32 v8, vcc_lo, v124, s4                          // 000000003eac: d7006a08 0200097c
	s_wait_alu depctr_va_vcc(0)                                // 000000003eb4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v125, vcc_lo             // 000000003eb8: d5207c09 01aafa05
	global_load_b32 v13, v[8:9], off                           // 000000003ec0: ee05007c 0000000d 00000008
	s_wait_loadcnt 0x0                                         // 000000003ecc: bfc00000
	v_mul_f32_e32 v8, v46, v13                                 // 000000003ed0: 10101b2e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003ed4: bf870091
	v_mul_f32_e32 v8, v14, v8                                  // 000000003ed8: 1010110e
	v_add_f32_e32 v73, v73, v8                                 // 000000003edc: 06921149
	v_add_co_u32 v8, vcc_lo, v126, s4                          // 000000003ee0: d7006a08 0200097e
	s_wait_alu depctr_va_vcc(0)                                // 000000003ee8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s5, v127, vcc_lo             // 000000003eec: d5207c09 01aafe05
	v_cmp_lt_u64_e64 s4, s[62:63], s[24:25]                    // 000000003ef4: d4590004 0200303e
	global_load_b32 v8, v[8:9], off                            // 000000003efc: ee05007c 00000008 00000008
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000003f08: 8b6a047e
	s_wait_loadcnt 0x0                                         // 000000003f0c: bfc00000
	v_mul_f32_e32 v9, v46, v8                                  // 000000003f10: 1012112e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f14: bf870091
	v_mul_f32_e32 v9, v15, v9                                  // 000000003f18: 1012130f
	v_dual_add_f32 v72, v72, v9 :: v_dual_mul_f32 v9, v25, v18 // 000000003f1c: c9061348 48082519
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f24: bf870091
	v_mul_f32_e32 v0, v0, v9                                   // 000000003f28: 10001300
	v_add_f32_e32 v62, v62, v0                                 // 000000003f2c: 067c013e
	v_mul_f32_e32 v0, v25, v16                                 // 000000003f30: 10002119
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f34: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000003f38: 10000101
	v_add_f32_e32 v61, v61, v0                                 // 000000003f3c: 067a013d
	v_mul_f32_e32 v0, v25, v17                                 // 000000003f40: 10002319
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f44: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 000000003f48: 10000102
	v_add_f32_e32 v60, v60, v0                                 // 000000003f4c: 0678013c
	v_mul_f32_e32 v0, v25, v10                                 // 000000003f50: 10001519
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f54: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 000000003f58: 10000103
	v_dual_add_f32 v59, v59, v0 :: v_dual_mul_f32 v0, v25, v11 // 000000003f5c: c906013b 3b001719
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f64: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 000000003f68: 10000104
	v_add_f32_e32 v58, v58, v0                                 // 000000003f6c: 0674013a
	v_mul_f32_e32 v0, v25, v12                                 // 000000003f70: 10001919
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f74: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 000000003f78: 10000105
	v_add_f32_e32 v57, v57, v0                                 // 000000003f7c: 06720139
	v_mul_f32_e32 v0, v25, v13                                 // 000000003f80: 10001b19
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f84: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 000000003f88: 10000106
	v_add_f32_e32 v56, v56, v0                                 // 000000003f8c: 06700138
	v_mul_f32_e32 v0, v25, v8                                  // 000000003f90: 10001119
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f94: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000003f98: 10000107
	v_add_f32_e32 v55, v55, v0                                 // 000000003f9c: 066e0137
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fa0: bf88ff9e
	s_cbranch_vccnz 63670                                      // 000000003fa4: bfa4f8b6 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x780>
	v_mul_lo_u32 v2, s23, v36                                  // 000000003fa8: d72c0002 02024817
	v_mul_lo_u32 v3, s22, v37                                  // 000000003fb0: d72c0003 02024a16
	v_mad_co_u64_u32 v[0:1], null, s22, v36, 0                 // 000000003fb8: d6fe7c00 02024816
	v_sub_co_u32 v14, vcc_lo, s20, v36                         // 000000003fc0: d7016a0e 02024814
	s_wait_alu depctr_va_vcc(0)                                // 000000003fc8: bf88ff9d
	v_sub_co_ci_u32_e64 v15, null, s21, v37, vcc_lo            // 000000003fcc: d5217c0f 01aa4a15
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000003fd4: bf870211
	v_cmp_lt_i64_e32 vcc_lo, 0, v[14:15]                       // 000000003fd8: 7ca21c80
	v_add3_u32 v1, v1, v3, v2                                  // 000000003fdc: d6550001 040a0701
	s_delay_alu instid0(valu_dep_1)                            // 000000003fe4: bf870001
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000003fe8: 3e0c0081
	s_and_b32 s2, vcc_lo, s1                                   // 000000003fec: 8b02016a
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ff0: bf88ff9e
	s_and_saveexec_b32 s3, s2                                  // 000000003ff4: be832002
	s_cbranch_execz 28                                         // 000000003ff8: bfa5001c <tessera_rocm_scaled_matmul_28d379a9237322d1+0x256c>
	v_lshlrev_b64_e32 v[0:1], 1, v[34:35]                      // 000000003ffc: 3e004481
	v_add_co_u32 v3, s2, s18, v6                               // 000000004000: d7000203 02020c12
	v_bfe_u32 v2, v117, 16, 1                                  // 000000004008: d6100002 02052175
	s_wait_alu depctr_va_sdst(0)                               // 000000004010: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s19, v7, s2                  // 000000004014: d5207c04 000a0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000401c: bf870193
	v_add_co_u32 v0, s2, v3, v0                                // 000000004020: d7000200 02020103
	v_add3_u32 v2, v2, v117, 0x7fff                            // 000000004028: d6550002 03feeb02 00007fff
	v_or_b32_e32 v5, 0x400000, v117                            // 000000004034: 380aeaff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000403c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v4, v1, s2                   // 000000004040: d5207c01 000a0304
	v_cmp_u_f32_e64 s2, v117, v117                             // 000000004048: d4180002 0202eb75
	s_wait_alu depctr_va_sdst(0)                               // 000000004050: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004054: bf870001
	v_cndmask_b32_e64 v2, v2, v5, s2                           // 000000004058: d5010002 000a0b02
	global_store_d16_hi_b16 v[0:1], v2, off                    // 000000004060: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 00000000406c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004070: 8c7e037e
	v_add_co_u32 v0, s2, s22, v34                              // 000000004074: d7000200 02024416
	s_wait_alu depctr_va_sdst(0)                               // 00000000407c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s23, v35, s2                 // 000000004080: d5207c01 000a4617
	v_cmp_lt_i64_e64 s2, 1, v[14:15]                           // 000000004088: d4510002 02021c81
	s_delay_alu instid0(valu_dep_2)                            // 000000004090: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000004094: 3e000081
	s_and_b32 s3, s2, s1                                       // 000000004098: 8b030102
	s_wait_alu depctr_sa_sdst(0)                               // 00000000409c: bf88ff9e
	s_and_saveexec_b32 s4, s3                                  // 0000000040a0: be842003
	s_cbranch_execz 27                                         // 0000000040a4: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2614>
	v_bfe_u32 v2, v114, 16, 1                                  // 0000000040a8: d6100002 02052172
	v_add_co_u32 v3, s3, s18, v6                               // 0000000040b0: d7000303 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 0000000040b8: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s19, v7, s3                  // 0000000040bc: d5207c04 000e0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000040c4: bf870193
	v_add3_u32 v5, v2, v114, 0x7fff                            // 0000000040c8: d6550005 03fee502 00007fff
	v_add_co_u32 v2, s3, v3, v0                                // 0000000040d4: d7000302 02020103
	v_or_b32_e32 v8, 0x400000, v114                            // 0000000040dc: 3810e4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000040e4: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v4, v1, s3                   // 0000000040e8: d5207c03 000e0304
	v_cmp_u_f32_e64 s3, v114, v114                             // 0000000040f0: d4180003 0202e572
	s_wait_alu depctr_va_sdst(0)                               // 0000000040f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000040fc: bf870001
	v_cndmask_b32_e64 v4, v5, v8, s3                           // 000000004100: d5010004 000e1105
	global_store_d16_hi_b16 v[2:3], v4, off                    // 000000004108: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004114: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004118: 8c7e047e
	s_lshl_b64 s[4:5], s[22:23], 1                             // 00000000411c: 84848116
	s_wait_alu depctr_sa_sdst(0)                               // 000000004120: bf88ff9e
	v_add_co_u32 v2, s3, s4, v34                               // 000000004124: d7000302 02024404
	s_wait_alu depctr_va_sdst(0)                               // 00000000412c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s5, v35, s3                  // 000000004130: d5207c03 000e4605
	v_cmp_lt_i64_e64 s3, 2, v[14:15]                           // 000000004138: d4510003 02021c82
	s_delay_alu instid0(valu_dep_2)                            // 000000004140: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004144: 3e040481
	s_and_b32 s4, s3, s1                                       // 000000004148: 8b040103
	s_wait_alu depctr_sa_sdst(0)                               // 00000000414c: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000004150: be852004
	s_cbranch_execz 27                                         // 000000004154: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x26c4>
	v_bfe_u32 v4, v111, 16, 1                                  // 000000004158: d6100004 0205216f
	v_add_co_u32 v5, s4, s18, v6                               // 000000004160: d7000405 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004168: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s19, v7, s4                  // 00000000416c: d5207c08 00120e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004174: bf870193
	v_add3_u32 v9, v4, v111, 0x7fff                            // 000000004178: d6550009 03fedf04 00007fff
	v_add_co_u32 v4, s4, v5, v2                                // 000000004184: d7000404 02020505
	v_or_b32_e32 v10, 0x400000, v111                           // 00000000418c: 3814deff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004194: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v3, s4                   // 000000004198: d5207c05 00120708
	v_cmp_u_f32_e64 s4, v111, v111                             // 0000000041a0: d4180004 0202df6f
	s_wait_alu depctr_va_sdst(0)                               // 0000000041a8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000041ac: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s4                          // 0000000041b0: d5010008 00121509
	global_store_d16_hi_b16 v[4:5], v8, off                    // 0000000041b8: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000041c8: 8c7e057e
	v_mad_co_u64_u32 v[8:9], null, s22, 3, v[34:35]            // 0000000041cc: d6fe7c08 04890616
	v_cmp_lt_i64_e64 s4, 3, v[14:15]                           // 0000000041d4: d4510004 02021c83
	s_and_b32 s5, s4, s1                                       // 0000000041dc: 8b050104
	v_mad_co_u64_u32 v[9:10], null, s23, 3, v[9:10]            // 0000000041e0: d6fe7c09 04250617
	s_delay_alu instid0(valu_dep_1)                            // 0000000041e8: bf870001
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 0000000041ec: 3e081081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000041f0: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 0000000041f4: be862005
	s_cbranch_execz 27                                         // 0000000041f8: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2768>
	v_bfe_u32 v8, v108, 16, 1                                  // 0000000041fc: d6100008 0205216c
	v_add_co_u32 v9, s5, s18, v6                               // 000000004204: d7000509 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 00000000420c: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s19, v7, s5                 // 000000004210: d5207c0a 00160e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004218: bf870193
	v_add3_u32 v11, v8, v108, 0x7fff                           // 00000000421c: d655000b 03fed908 00007fff
	v_add_co_u32 v8, s5, v9, v4                                // 000000004228: d7000508 02020909
	v_or_b32_e32 v12, 0x400000, v108                           // 000000004230: 3818d8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004238: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v5, s5                  // 00000000423c: d5207c09 00160b0a
	v_cmp_u_f32_e64 s5, v108, v108                             // 000000004244: d4180005 0202d96c
	s_wait_alu depctr_va_sdst(0)                               // 00000000424c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004250: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s5                        // 000000004254: d501000a 0016190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 00000000425c: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 000000004268: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 00000000426c: 8c7e067e
	s_lshl_b64 s[6:7], s[22:23], 2                             // 000000004270: 84868216
	s_wait_alu depctr_sa_sdst(0)                               // 000000004274: bf88ff9e
	v_add_co_u32 v8, s5, s6, v34                               // 000000004278: d7000508 02024406
	s_wait_alu depctr_va_sdst(0)                               // 000000004280: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s7, v35, s5                  // 000000004284: d5207c09 00164607
	v_cmp_lt_i64_e64 s5, 4, v[14:15]                           // 00000000428c: d4510005 02021c84
	s_delay_alu instid0(valu_dep_2)                            // 000000004294: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 000000004298: 3e101081
	s_and_b32 s6, s5, s1                                       // 00000000429c: 8b060105
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042a0: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 0000000042a4: be872006
	s_cbranch_execz 27                                         // 0000000042a8: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2818>
	v_bfe_u32 v10, v105, 16, 1                                 // 0000000042ac: d610000a 02052169
	v_add_co_u32 v11, s6, s18, v6                              // 0000000042b4: d700060b 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 0000000042bc: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s19, v7, s6                 // 0000000042c0: d5207c0c 001a0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000042c8: bf870193
	v_add3_u32 v13, v10, v105, 0x7fff                          // 0000000042cc: d655000d 03fed30a 00007fff
	v_add_co_u32 v10, s6, v11, v8                              // 0000000042d8: d700060a 0202110b
	v_or_b32_e32 v16, 0x400000, v105                           // 0000000042e0: 3820d2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000042e8: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s6                 // 0000000042ec: d5207c0b 001a130c
	v_cmp_u_f32_e64 s6, v105, v105                             // 0000000042f4: d4180006 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 0000000042fc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004300: bf870001
	v_cndmask_b32_e64 v12, v13, v16, s6                        // 000000004304: d501000c 001a210d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 00000000430c: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004318: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000431c: 8c7e077e
	v_mad_co_u64_u32 v[10:11], null, s22, 5, v[34:35]          // 000000004320: d6fe7c0a 04890a16
	v_cmp_lt_i64_e64 s6, 5, v[14:15]                           // 000000004328: d4510006 02021c85
	s_and_b32 s7, s6, s1                                       // 000000004330: 8b070106
	v_mad_co_u64_u32 v[11:12], null, s23, 5, v[11:12]          // 000000004334: d6fe7c0b 042d0a17
	s_delay_alu instid0(valu_dep_1)                            // 00000000433c: bf870001
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000004340: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000004344: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 000000004348: be882007
	s_cbranch_execz 27                                         // 00000000434c: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x28bc>
	v_bfe_u32 v12, v100, 16, 1                                 // 000000004350: d610000c 02052164
	v_add_co_u32 v13, s7, s18, v6                              // 000000004358: d700070d 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004360: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s19, v7, s7                 // 000000004364: d5207c10 001e0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000436c: bf870193
	v_add3_u32 v17, v12, v100, 0x7fff                          // 000000004370: d6550011 03fec90c 00007fff
	v_add_co_u32 v12, s7, v13, v10                             // 00000000437c: d700070c 0202150d
	v_or_b32_e32 v18, 0x400000, v100                           // 000000004384: 3824c8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000438c: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v16, v11, s7                // 000000004390: d5207c0d 001e1710
	v_cmp_u_f32_e64 s7, v100, v100                             // 000000004398: d4180007 0202c964
	s_wait_alu depctr_va_sdst(0)                               // 0000000043a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000043a4: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s7                        // 0000000043a8: d5010010 001e2511
	global_store_d16_hi_b16 v[12:13], v16, off                 // 0000000043b0: ee09407c 08000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000043c0: 8c7e087e
	v_mad_co_u64_u32 v[16:17], null, s22, 6, v[34:35]          // 0000000043c4: d6fe7c10 04890c16
	v_cmp_lt_i64_e64 s7, 6, v[14:15]                           // 0000000043cc: d4510007 02021c86
	s_and_b32 s8, s7, s1                                       // 0000000043d4: 8b080107
	v_mad_co_u64_u32 v[17:18], null, s23, 6, v[17:18]          // 0000000043d8: d6fe7c11 04450c17
	s_delay_alu instid0(valu_dep_1)                            // 0000000043e0: bf870001
	v_lshlrev_b64_e32 v[12:13], 1, v[16:17]                    // 0000000043e4: 3e182081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043e8: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000043ec: be892008
	s_cbranch_execz 27                                         // 0000000043f0: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2960>
	v_bfe_u32 v16, v95, 16, 1                                  // 0000000043f4: d6100010 0205215f
	v_add_co_u32 v17, s8, s18, v6                              // 0000000043fc: d7000811 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 000000004404: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v7, s8                 // 000000004408: d5207c12 00220e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004410: bf870193
	v_add3_u32 v19, v16, v95, 0x7fff                           // 000000004414: d6550013 03febf10 00007fff
	v_add_co_u32 v16, s8, v17, v12                             // 000000004420: d7000810 02021911
	v_or_b32_e32 v20, 0x400000, v95                            // 000000004428: 3828beff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004430: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v13, s8                // 000000004434: d5207c11 00221b12
	v_cmp_u_f32_e64 s8, v95, v95                               // 00000000443c: d4180008 0202bf5f
	s_wait_alu depctr_va_sdst(0)                               // 000000004444: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004448: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s8                        // 00000000444c: d5010012 00222913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000004454: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000004460: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000004464: 8c7e097e
	v_mad_co_u64_u32 v[16:17], null, s22, 7, v[34:35]          // 000000004468: d6fe7c10 04890e16
	v_cmp_lt_i64_e64 s8, 7, v[14:15]                           // 000000004470: d4510008 02021c87
	s_and_b32 s9, s8, s1                                       // 000000004478: 8b090108
	v_mad_co_u64_u32 v[17:18], null, s23, 7, v[17:18]          // 00000000447c: d6fe7c11 04450e17
	s_delay_alu instid0(valu_dep_1)                            // 000000004484: bf870001
	v_lshlrev_b64_e32 v[14:15], 1, v[16:17]                    // 000000004488: 3e1c2081
	s_wait_alu depctr_sa_sdst(0)                               // 00000000448c: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000004490: be8a2009
	s_cbranch_execz 27                                         // 000000004494: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2a04>
	v_bfe_u32 v16, v92, 16, 1                                  // 000000004498: d6100010 0205215c
	v_add_co_u32 v17, s9, s18, v6                              // 0000000044a0: d7000911 02020c12
	s_wait_alu depctr_va_sdst(0)                               // 0000000044a8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s19, v7, s9                 // 0000000044ac: d5207c12 00260e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000044b4: bf870193
	v_add3_u32 v19, v16, v92, 0x7fff                           // 0000000044b8: d6550013 03feb910 00007fff
	v_add_co_u32 v16, s9, v17, v14                             // 0000000044c4: d7000910 02021d11
	v_or_b32_e32 v20, 0x400000, v92                            // 0000000044cc: 3828b8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000044d4: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s9                // 0000000044d8: d5207c11 00261f12
	v_cmp_u_f32_e64 s9, v92, v92                               // 0000000044e0: d4180009 0202b95c
	s_wait_alu depctr_va_sdst(0)                               // 0000000044e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000044ec: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s9                        // 0000000044f0: d5010012 00262913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 0000000044f8: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000004504: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000004508: 8c7e0a7e
	v_mul_lo_u32 v20, s23, v32                                 // 00000000450c: d72c0014 02024017
	v_mul_lo_u32 v21, s22, v33                                 // 000000004514: d72c0015 02024216
	v_mad_co_u64_u32 v[16:17], null, s22, v32, 0               // 00000000451c: d6fe7c10 02024016
	v_sub_co_u32 v18, s9, s20, v32                             // 000000004524: d7010912 02024014
	s_wait_alu depctr_va_sdst(0)                               // 00000000452c: bf88f19f
	v_sub_co_ci_u32_e64 v19, null, s21, v33, s9                // 000000004530: d5217c13 00264215
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000004538: bf870211
	v_cmp_lt_i64_e64 s9, 0, v[18:19]                           // 00000000453c: d4510009 02022480
	v_add3_u32 v17, v17, v21, v20                              // 000000004544: d6550011 04522b11
	s_delay_alu instid0(valu_dep_1)                            // 00000000454c: bf870001
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000004550: 3e202081
	s_and_b32 s10, s9, s1                                      // 000000004554: 8b0a0109
	s_wait_alu depctr_sa_sdst(0)                               // 000000004558: bf88ff9e
	s_and_saveexec_b32 s11, s10                                // 00000000455c: be8b200a
	s_cbranch_execz 28                                         // 000000004560: bfa5001c <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2ad4>
	v_lshlrev_b64_e32 v[20:21], 1, v[34:35]                    // 000000004564: 3e284481
	v_add_co_u32 v23, s10, s18, v16                            // 000000004568: d7000a17 02022012
	v_bfe_u32 v22, v87, 16, 1                                  // 000000004570: d6100016 02052157
	s_wait_alu depctr_va_sdst(0)                               // 000000004578: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s19, v17, s10               // 00000000457c: d5207c18 002a2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004584: bf870193
	v_add_co_u32 v20, s10, v23, v20                            // 000000004588: d7000a14 02022917
	v_add3_u32 v22, v22, v87, 0x7fff                           // 000000004590: d6550016 03feaf16 00007fff
	v_or_b32_e32 v25, 0x400000, v87                            // 00000000459c: 3832aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000045a4: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v24, v21, s10               // 0000000045a8: d5207c15 002a2b18
	v_cmp_u_f32_e64 s10, v87, v87                              // 0000000045b0: d418000a 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 0000000045b8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000045bc: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s10                       // 0000000045c0: d5010016 002a3316
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000045c8: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045d4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 0000000045d8: 8c7e0b7e
	v_cmp_lt_i64_e64 s10, 1, v[18:19]                          // 0000000045dc: d451000a 02022481
	s_and_b32 s11, s10, s1                                     // 0000000045e4: 8b0b010a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045e8: bf88ff9e
	s_and_saveexec_b32 s12, s11                                // 0000000045ec: be8c200b
	s_cbranch_execz 27                                         // 0000000045f0: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2b60>
	v_bfe_u32 v20, v86, 16, 1                                  // 0000000045f4: d6100014 02052156
	v_add_co_u32 v21, s11, s18, v16                            // 0000000045fc: d7000b15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004604: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s11               // 000000004608: d5207c16 002e2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004610: bf870193
	v_add3_u32 v23, v20, v86, 0x7fff                           // 000000004614: d6550017 03fead14 00007fff
	v_add_co_u32 v20, s11, v21, v0                             // 000000004620: d7000b14 02020115
	v_or_b32_e32 v24, 0x400000, v86                            // 000000004628: 3830acff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004630: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v1, s11                // 000000004634: d5207c15 002e0316
	v_cmp_u_f32_e64 s11, v86, v86                              // 00000000463c: d418000b 0202ad56
	s_wait_alu depctr_va_sdst(0)                               // 000000004644: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004648: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s11                       // 00000000464c: d5010016 002e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004654: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004660: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004664: 8c7e0c7e
	v_cmp_lt_i64_e64 s11, 2, v[18:19]                          // 000000004668: d451000b 02022482
	s_and_b32 s12, s11, s1                                     // 000000004670: 8b0c010b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004674: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000004678: be8d200c
	s_cbranch_execz 27                                         // 00000000467c: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2bec>
	v_bfe_u32 v20, v84, 16, 1                                  // 000000004680: d6100014 02052154
	v_add_co_u32 v21, s12, s18, v16                            // 000000004688: d7000c15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004690: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s12               // 000000004694: d5207c16 00322213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000469c: bf870193
	v_add3_u32 v23, v20, v84, 0x7fff                           // 0000000046a0: d6550017 03fea914 00007fff
	v_add_co_u32 v20, s12, v21, v2                             // 0000000046ac: d7000c14 02020515
	v_or_b32_e32 v24, 0x400000, v84                            // 0000000046b4: 3830a8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000046bc: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v3, s12                // 0000000046c0: d5207c15 00320716
	v_cmp_u_f32_e64 s12, v84, v84                              // 0000000046c8: d418000c 0202a954
	s_wait_alu depctr_va_sdst(0)                               // 0000000046d0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000046d4: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s12                       // 0000000046d8: d5010016 00323117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000046e0: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 0000000046f0: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 3, v[18:19]                          // 0000000046f4: d451000c 02022483
	s_and_b32 s13, s12, s1                                     // 0000000046fc: 8b0d010c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004700: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 000000004704: be8e200d
	s_cbranch_execz 27                                         // 000000004708: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2c78>
	v_bfe_u32 v20, v82, 16, 1                                  // 00000000470c: d6100014 02052152
	v_add_co_u32 v21, s13, s18, v16                            // 000000004714: d7000d15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 00000000471c: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s13               // 000000004720: d5207c16 00362213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004728: bf870193
	v_add3_u32 v23, v20, v82, 0x7fff                           // 00000000472c: d6550017 03fea514 00007fff
	v_add_co_u32 v20, s13, v21, v4                             // 000000004738: d7000d14 02020915
	v_or_b32_e32 v24, 0x400000, v82                            // 000000004740: 3830a4ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004748: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v5, s13                // 00000000474c: d5207c15 00360b16
	v_cmp_u_f32_e64 s13, v82, v82                              // 000000004754: d418000d 0202a552
	s_wait_alu depctr_va_sdst(0)                               // 00000000475c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004760: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s13                       // 000000004764: d5010016 00363117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 00000000476c: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004778: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s14                             // 00000000477c: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 4, v[18:19]                          // 000000004780: d451000d 02022484
	s_and_b32 s14, s13, s1                                     // 000000004788: 8b0e010d
	s_wait_alu depctr_sa_sdst(0)                               // 00000000478c: bf88ff9e
	s_and_saveexec_b32 s15, s14                                // 000000004790: be8f200e
	s_cbranch_execz 27                                         // 000000004794: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2d04>
	v_bfe_u32 v20, v79, 16, 1                                  // 000000004798: d6100014 0205214f
	v_add_co_u32 v21, s14, s18, v16                            // 0000000047a0: d7000e15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000047a8: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s14               // 0000000047ac: d5207c16 003a2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000047b4: bf870193
	v_add3_u32 v23, v20, v79, 0x7fff                           // 0000000047b8: d6550017 03fe9f14 00007fff
	v_add_co_u32 v20, s14, v21, v8                             // 0000000047c4: d7000e14 02021115
	v_or_b32_e32 v24, 0x400000, v79                            // 0000000047cc: 38309eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d4: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v9, s14                // 0000000047d8: d5207c15 003a1316
	v_cmp_u_f32_e64 s14, v79, v79                              // 0000000047e0: d418000e 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 0000000047e8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000047ec: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s14                       // 0000000047f0: d5010016 003a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 0000000047f8: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004804: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s15                             // 000000004808: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 5, v[18:19]                          // 00000000480c: d451000e 02022485
	s_and_b32 s15, s14, s1                                     // 000000004814: 8b0f010e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004818: bf88ff9e
	s_and_saveexec_b32 s16, s15                                // 00000000481c: be90200f
	s_cbranch_execz 27                                         // 000000004820: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2d90>
	v_bfe_u32 v20, v77, 16, 1                                  // 000000004824: d6100014 0205214d
	v_add_co_u32 v21, s15, s18, v16                            // 00000000482c: d7000f15 02022012
	s_wait_alu depctr_va_sdst(0)                               // 000000004834: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s15               // 000000004838: d5207c16 003e2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004840: bf870193
	v_add3_u32 v23, v20, v77, 0x7fff                           // 000000004844: d6550017 03fe9b14 00007fff
	v_add_co_u32 v20, s15, v21, v10                            // 000000004850: d7000f14 02021515
	v_or_b32_e32 v24, 0x400000, v77                            // 000000004858: 38309aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004860: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v11, s15               // 000000004864: d5207c15 003e1716
	v_cmp_u_f32_e64 s15, v77, v77                              // 00000000486c: d418000f 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000004874: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004878: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s15                       // 00000000487c: d5010016 003e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004884: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004890: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s16                             // 000000004894: 8c7e107e
	v_cmp_lt_i64_e64 s15, 6, v[18:19]                          // 000000004898: d451000f 02022486
	s_and_b32 s16, s15, s1                                     // 0000000048a0: 8b10010f
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048a4: bf88ff9e
	s_and_saveexec_b32 s17, s16                                // 0000000048a8: be912010
	s_cbranch_execz 27                                         // 0000000048ac: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2e1c>
	v_bfe_u32 v20, v73, 16, 1                                  // 0000000048b0: d6100014 02052149
	v_add_co_u32 v21, s16, s18, v16                            // 0000000048b8: d7001015 02022012
	s_wait_alu depctr_va_sdst(0)                               // 0000000048c0: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s19, v17, s16               // 0000000048c4: d5207c16 00422213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000048cc: bf870193
	v_add3_u32 v23, v20, v73, 0x7fff                           // 0000000048d0: d6550017 03fe9314 00007fff
	v_add_co_u32 v20, s16, v21, v12                            // 0000000048dc: d7001014 02021915
	v_or_b32_e32 v24, 0x400000, v73                            // 0000000048e4: 383092ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000048ec: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v13, s16               // 0000000048f0: d5207c15 00421b16
	v_cmp_u_f32_e64 s16, v73, v73                              // 0000000048f8: d4180010 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000004900: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004904: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s16                       // 000000004908: d5010016 00423117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004910: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 00000000491c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 000000004920: 8c7e117e
	v_cmp_lt_i64_e64 s16, 7, v[18:19]                          // 000000004924: d4510010 02022487
	s_and_b32 s1, s16, s1                                      // 00000000492c: 8b010110
	s_wait_alu depctr_sa_sdst(0)                               // 000000004930: bf88ff9e
	s_and_saveexec_b32 s17, s1                                 // 000000004934: be912001
	s_cbranch_execz 27                                         // 000000004938: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2ea8>
	v_bfe_u32 v18, v72, 16, 1                                  // 00000000493c: d6100012 02052148
	v_add_co_u32 v19, s1, s18, v16                             // 000000004944: d7000113 02022012
	s_wait_alu depctr_va_sdst(0)                               // 00000000494c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s19, v17, s1                // 000000004950: d5207c14 00062213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004958: bf870193
	v_add3_u32 v21, v18, v72, 0x7fff                           // 00000000495c: d6550015 03fe9112 00007fff
	v_add_co_u32 v18, s1, v19, v14                             // 000000004968: d7000112 02021d13
	v_or_b32_e32 v22, 0x400000, v72                            // 000000004970: 382c90ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004978: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v15, s1                // 00000000497c: d5207c13 00061f14
	v_cmp_u_f32_e64 s1, v72, v72                               // 000000004984: d4180001 02029148
	s_wait_alu depctr_va_sdst(0)                               // 00000000498c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004990: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s1                        // 000000004994: d5010014 00062d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 00000000499c: ee09407c 0a000000 00000012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049a8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s17                             // 0000000049ac: 8c7e117e
	s_and_b32 s17, vcc_lo, s0                                  // 0000000049b0: 8b11006a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049b4: bf88ff9e
	s_and_saveexec_b32 s1, s17                                 // 0000000049b8: be812011
	s_cbranch_execz 25                                         // 0000000049bc: bfa50019 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2f24>
	v_lshlrev_b64_e32 v[18:19], 1, v[34:35]                    // 0000000049c0: 3e244481
	v_add_co_u32 v21, vcc_lo, s18, v6                          // 0000000049c4: d7006a15 02020c12
	v_bfe_u32 v20, v71, 16, 1                                  // 0000000049cc: d6100014 02052147
	s_wait_alu depctr_va_vcc(0)                                // 0000000049d4: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s19, v7, vcc_lo             // 0000000049d8: d5207c16 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000049e0: bf870193
	v_add_co_u32 v18, vcc_lo, v21, v18                         // 0000000049e4: d7006a12 02022515
	v_add3_u32 v20, v20, v71, 0x7fff                           // 0000000049ec: d6550014 03fe8f14 00007fff
	v_or_b32_e32 v23, 0x400000, v71                            // 0000000049f8: 382e8eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a00: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v22, v19, vcc_lo            // 000000004a04: d5207c13 01aa2716
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000004a0c: 7c308f47
	s_wait_alu depctr_va_vcc(0)                                // 000000004a10: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v23, vcc_lo                    // 000000004a14: 02282f14
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004a18: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a24: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004a28: 8c7e017e
	s_and_b32 s2, s2, s0                                       // 000000004a2c: 8b020002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a30: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004a34: be812002
	s_cbranch_execz 24                                         // 000000004a38: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2f9c>
	v_bfe_u32 v18, v70, 16, 1                                  // 000000004a3c: d6100012 02052146
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004a44: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004a4c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004a50: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a58: bf870193
	v_add3_u32 v21, v18, v70, 0x7fff                           // 000000004a5c: d6550015 03fe8d12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v0                          // 000000004a68: d7006a12 02020113
	v_or_b32_e32 v22, 0x400000, v70                            // 000000004a70: 382c8cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a78: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v1, vcc_lo             // 000000004a7c: d5207c13 01aa0314
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000004a84: 7c308d46
	s_wait_alu depctr_va_vcc(0)                                // 000000004a88: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004a8c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004a90: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a9c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004aa0: 8c7e017e
	s_and_b32 s2, s3, s0                                       // 000000004aa4: 8b020003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004aa8: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004aac: be812002
	s_cbranch_execz 24                                         // 000000004ab0: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3014>
	v_bfe_u32 v18, v69, 16, 1                                  // 000000004ab4: d6100012 02052145
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004abc: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004ac4: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004ac8: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ad0: bf870193
	v_add3_u32 v21, v18, v69, 0x7fff                           // 000000004ad4: d6550015 03fe8b12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000004ae0: d7006a12 02020513
	v_or_b32_e32 v22, 0x400000, v69                            // 000000004ae8: 382c8aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004af0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v3, vcc_lo             // 000000004af4: d5207c13 01aa0714
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000004afc: 7c308b45
	s_wait_alu depctr_va_vcc(0)                                // 000000004b00: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004b04: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004b08: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004b18: 8c7e017e
	s_and_b32 s2, s4, s0                                       // 000000004b1c: 8b020004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b20: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004b24: be812002
	s_cbranch_execz 24                                         // 000000004b28: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x308c>
	v_bfe_u32 v18, v68, 16, 1                                  // 000000004b2c: d6100012 02052144
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004b34: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004b3c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004b40: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b48: bf870193
	v_add3_u32 v21, v18, v68, 0x7fff                           // 000000004b4c: d6550015 03fe8912 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v4                          // 000000004b58: d7006a12 02020913
	v_or_b32_e32 v22, 0x400000, v68                            // 000000004b60: 382c88ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004b68: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v5, vcc_lo             // 000000004b6c: d5207c13 01aa0b14
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 000000004b74: 7c308944
	s_wait_alu depctr_va_vcc(0)                                // 000000004b78: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004b7c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004b80: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b8c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004b90: 8c7e017e
	s_and_b32 s2, s5, s0                                       // 000000004b94: 8b020005
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b98: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004b9c: be812002
	s_cbranch_execz 24                                         // 000000004ba0: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3104>
	v_bfe_u32 v18, v67, 16, 1                                  // 000000004ba4: d6100012 02052143
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004bac: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004bb4: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004bb8: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004bc0: bf870193
	v_add3_u32 v21, v18, v67, 0x7fff                           // 000000004bc4: d6550015 03fe8712 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v8                          // 000000004bd0: d7006a12 02021113
	v_or_b32_e32 v22, 0x400000, v67                            // 000000004bd8: 382c86ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004be0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v9, vcc_lo             // 000000004be4: d5207c13 01aa1314
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 000000004bec: 7c308743
	s_wait_alu depctr_va_vcc(0)                                // 000000004bf0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004bf4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004bf8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004c08: 8c7e017e
	s_and_b32 s2, s6, s0                                       // 000000004c0c: 8b020006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c10: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004c14: be812002
	s_cbranch_execz 24                                         // 000000004c18: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x317c>
	v_bfe_u32 v18, v66, 16, 1                                  // 000000004c1c: d6100012 02052142
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004c24: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004c2c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004c30: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004c38: bf870193
	v_add3_u32 v21, v18, v66, 0x7fff                           // 000000004c3c: d6550015 03fe8512 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v10                         // 000000004c48: d7006a12 02021513
	v_or_b32_e32 v22, 0x400000, v66                            // 000000004c50: 382c84ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004c58: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v11, vcc_lo            // 000000004c5c: d5207c13 01aa1714
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 000000004c64: 7c308542
	s_wait_alu depctr_va_vcc(0)                                // 000000004c68: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004c6c: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004c70: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004c80: 8c7e017e
	s_and_b32 s2, s7, s0                                       // 000000004c84: 8b020007
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c88: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004c8c: be812002
	s_cbranch_execz 24                                         // 000000004c90: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x31f4>
	v_bfe_u32 v18, v65, 16, 1                                  // 000000004c94: d6100012 02052141
	v_add_co_u32 v19, vcc_lo, s18, v6                          // 000000004c9c: d7006a13 02020c12
	s_wait_alu depctr_va_vcc(0)                                // 000000004ca4: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v7, vcc_lo             // 000000004ca8: d5207c14 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004cb0: bf870193
	v_add3_u32 v21, v18, v65, 0x7fff                           // 000000004cb4: d6550015 03fe8312 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v12                         // 000000004cc0: d7006a12 02021913
	v_or_b32_e32 v22, 0x400000, v65                            // 000000004cc8: 382c82ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004cd0: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v13, vcc_lo            // 000000004cd4: d5207c13 01aa1b14
	v_cmp_u_f32_e32 vcc_lo, v65, v65                           // 000000004cdc: 7c308341
	s_wait_alu depctr_va_vcc(0)                                // 000000004ce0: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004ce4: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004ce8: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cf4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004cf8: 8c7e017e
	s_and_b32 s2, s8, s0                                       // 000000004cfc: 8b020008
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d00: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004d04: be812002
	s_cbranch_execz 24                                         // 000000004d08: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x326c>
	v_add_co_u32 v6, vcc_lo, s18, v6                           // 000000004d0c: d7006a06 02020c12
	v_bfe_u32 v18, v63, 16, 1                                  // 000000004d14: d6100012 0205213f
	s_wait_alu depctr_va_vcc(0)                                // 000000004d1c: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s19, v7, vcc_lo              // 000000004d20: d5207c07 01aa0e13
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004d28: bf870193
	v_add_co_u32 v6, vcc_lo, v6, v14                           // 000000004d2c: d7006a06 02021d06
	v_add3_u32 v18, v18, v63, 0x7fff                           // 000000004d34: d6550012 03fe7f12 00007fff
	v_or_b32_e32 v19, 0x400000, v63                            // 000000004d40: 38267eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004d48: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo              // 000000004d4c: d5207c07 01aa1f07
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000004d54: 7c307f3f
	s_wait_alu depctr_va_vcc(0)                                // 000000004d58: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 000000004d5c: 02242712
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 000000004d60: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004d70: 8c7e017e
	s_and_b32 s2, s9, s0                                       // 000000004d74: 8b020009
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d78: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004d7c: be812002
	s_cbranch_execz 25                                         // 000000004d80: bfa50019 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x32e8>
	v_lshlrev_b64_e32 v[6:7], 1, v[34:35]                      // 000000004d84: 3e0c4481
	v_add_co_u32 v19, vcc_lo, s18, v16                         // 000000004d88: d7006a13 02022012
	v_bfe_u32 v18, v62, 16, 1                                  // 000000004d90: d6100012 0205213e
	s_wait_alu depctr_va_vcc(0)                                // 000000004d98: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s19, v17, vcc_lo            // 000000004d9c: d5207c14 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004da4: bf870193
	v_add_co_u32 v6, vcc_lo, v19, v6                           // 000000004da8: d7006a06 02020d13
	v_add3_u32 v18, v18, v62, 0x7fff                           // 000000004db0: d6550012 03fe7d12 00007fff
	v_or_b32_e32 v21, 0x400000, v62                            // 000000004dbc: 382a7cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004dc4: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v20, v7, vcc_lo              // 000000004dc8: d5207c07 01aa0f14
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000004dd0: 7c307d3e
	s_wait_alu depctr_va_vcc(0)                                // 000000004dd4: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo                    // 000000004dd8: 02242b12
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 000000004ddc: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004de8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004dec: 8c7e017e
	s_and_b32 s2, s10, s0                                      // 000000004df0: 8b02000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004df4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004df8: be812002
	s_cbranch_execz 24                                         // 000000004dfc: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3360>
	v_add_co_u32 v7, vcc_lo, s18, v16                          // 000000004e00: d7006a07 02022012
	v_bfe_u32 v6, v61, 16, 1                                   // 000000004e08: d6100006 0205213d
	s_wait_alu depctr_va_vcc(0)                                // 000000004e10: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s19, v17, vcc_lo            // 000000004e14: d5207c12 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e1c: bf870193
	v_add_co_u32 v0, vcc_lo, v7, v0                            // 000000004e20: d7006a00 02020107
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000004e28: d6550006 03fe7b06 00007fff
	v_or_b32_e32 v19, 0x400000, v61                            // 000000004e34: 38267aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e3c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v18, v1, vcc_lo              // 000000004e40: d5207c01 01aa0312
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000004e48: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 000000004e4c: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v19, vcc_lo                      // 000000004e50: 020c2706
	global_store_d16_hi_b16 v[0:1], v6, off offset:32          // 000000004e54: ee09407c 03000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004e64: 8c7e017e
	s_and_b32 s2, s11, s0                                      // 000000004e68: 8b02000b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e6c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004e70: be812002
	s_cbranch_execz 24                                         // 000000004e74: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x33d8>
	v_bfe_u32 v0, v60, 16, 1                                   // 000000004e78: d6100000 0205213c
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004e80: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004e88: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s19, v17, vcc_lo             // 000000004e8c: d5207c06 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e94: bf870193
	v_add3_u32 v7, v0, v60, 0x7fff                             // 000000004e98: d6550007 03fe7900 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v2                            // 000000004ea4: d7006a00 02020501
	v_or_b32_e32 v18, 0x400000, v60                            // 000000004eac: 382478ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004eb4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v6, v3, vcc_lo               // 000000004eb8: d5207c01 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000004ec0: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 000000004ec4: bf88ff9d
	v_cndmask_b32_e32 v2, v7, v18, vcc_lo                      // 000000004ec8: 02042507
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004ecc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ed8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004edc: 8c7e017e
	s_and_b32 s2, s12, s0                                      // 000000004ee0: 8b02000c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ee4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004ee8: be812002
	s_cbranch_execz 24                                         // 000000004eec: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3450>
	v_bfe_u32 v0, v59, 16, 1                                   // 000000004ef0: d6100000 0205213b
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004ef8: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004f00: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004f04: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f0c: bf870193
	v_add3_u32 v3, v0, v59, 0x7fff                             // 000000004f10: d6550003 03fe7700 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v4                            // 000000004f1c: d7006a00 02020901
	v_or_b32_e32 v6, 0x400000, v59                             // 000000004f24: 380c76ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f2c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v5, vcc_lo               // 000000004f30: d5207c01 01aa0b02
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000004f38: 7c30773b
	s_wait_alu depctr_va_vcc(0)                                // 000000004f3c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v6, vcc_lo                       // 000000004f40: 02040d03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004f44: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f50: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004f54: 8c7e017e
	s_and_b32 s2, s13, s0                                      // 000000004f58: 8b02000d
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f5c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004f60: be812002
	s_cbranch_execz 24                                         // 000000004f64: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x34c8>
	v_bfe_u32 v0, v58, 16, 1                                   // 000000004f68: d6100000 0205213a
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004f70: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004f78: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004f7c: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f84: bf870193
	v_add3_u32 v3, v0, v58, 0x7fff                             // 000000004f88: d6550003 03fe7500 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v8                            // 000000004f94: d7006a00 02021101
	v_or_b32_e32 v4, 0x400000, v58                             // 000000004f9c: 380874ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004fa4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v9, vcc_lo               // 000000004fa8: d5207c01 01aa1302
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000004fb0: 7c30753a
	s_wait_alu depctr_va_vcc(0)                                // 000000004fb4: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000004fb8: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000004fbc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fc8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004fcc: 8c7e017e
	s_and_b32 s2, s14, s0                                      // 000000004fd0: 8b02000e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fd4: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000004fd8: be812002
	s_cbranch_execz 24                                         // 000000004fdc: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3540>
	v_bfe_u32 v0, v57, 16, 1                                   // 000000004fe0: d6100000 02052139
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000004fe8: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000004ff0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 000000004ff4: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ffc: bf870193
	v_add3_u32 v3, v0, v57, 0x7fff                             // 000000005000: d6550003 03fe7300 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v10                           // 00000000500c: d7006a00 02021501
	v_or_b32_e32 v4, 0x400000, v57                             // 000000005014: 380872ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000501c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v11, vcc_lo              // 000000005020: d5207c01 01aa1702
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000005028: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 00000000502c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005030: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005034: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005040: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005044: 8c7e017e
	s_and_b32 s2, s15, s0                                      // 000000005048: 8b02000f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000504c: bf88ff9e
	s_and_saveexec_b32 s1, s2                                  // 000000005050: be812002
	s_cbranch_execz 24                                         // 000000005054: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x35b8>
	v_bfe_u32 v0, v56, 16, 1                                   // 000000005058: d6100000 02052138
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 000000005060: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 000000005068: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 00000000506c: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005074: bf870193
	v_add3_u32 v3, v0, v56, 0x7fff                             // 000000005078: d6550003 03fe7100 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v12                           // 000000005084: d7006a00 02021901
	v_or_b32_e32 v4, 0x400000, v56                             // 00000000508c: 380870ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005094: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v13, vcc_lo              // 000000005098: d5207c01 01aa1b02
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 0000000050a0: 7c307138
	s_wait_alu depctr_va_vcc(0)                                // 0000000050a4: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 0000000050a8: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000050ac: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000050bc: 8c7e017e
	s_and_b32 s1, s16, s0                                      // 0000000050c0: 8b010010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050c4: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000050c8: be802001
	s_cbranch_execz 24                                         // 0000000050cc: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3630>
	v_bfe_u32 v0, v55, 16, 1                                   // 0000000050d0: d6100000 02052137
	v_add_co_u32 v1, vcc_lo, s18, v16                          // 0000000050d8: d7006a01 02022012
	s_wait_alu depctr_va_vcc(0)                                // 0000000050e0: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s19, v17, vcc_lo             // 0000000050e4: d5207c02 01aa2213
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000050ec: bf870193
	v_add3_u32 v3, v0, v55, 0x7fff                             // 0000000050f0: d6550003 03fe6f00 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v14                           // 0000000050fc: d7006a00 02021d01
	v_or_b32_e32 v4, 0x400000, v55                             // 000000005104: 38086eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000510c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v15, vcc_lo              // 000000005110: d5207c01 01aa1f02
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000005118: 7c306f37
	s_wait_alu depctr_va_vcc(0)                                // 00000000511c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005120: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005124: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005130: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005134: 8c7e007e
	s_branch 62125                                             // 000000005138: bfa0f2ad <tessera_rocm_scaled_matmul_28d379a9237322d1+0xf0>
	v_mov_b32_e32 v37, s37                                     // 00000000513c: 7e4a0225
	s_lshr_b64 s[4:5], s[42:43], 7                             // 000000005140: 8584872a
	v_or_b32_e32 v2, s36, v53                                  // 000000005144: 38046a24
	v_or_b32_e32 v4, s36, v52                                  // 000000005148: 38086824
	v_mov_b32_e32 v3, s37                                      // 00000000514c: 7e060225
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[36:37]                // 000000005150: 7ca84814
	s_lshr_b64 s[0:1], s[40:41], 7                             // 000000005154: 85808728
	s_add_nc_u64 s[2:3], s[4:5], -1                            // 000000005158: a982c104
	v_or_b32_e32 v0, s36, v54                                  // 00000000515c: 38006c24
	s_wait_alu depctr_sa_sdst(0)                               // 000000005160: bf88ff9e
	v_cmp_lt_u64_e64 s6, s[0:1], s[2:3]                        // 000000005164: d4590006 02000400
	v_dual_mov_b32 v5, s37 :: v_dual_cndmask_b32 v16, 0, v36   // 00000000516c: ca120025 05104880
	v_cndmask_b32_e32 v17, 0, v37, vcc_lo                      // 000000005174: 02224a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[2:3]                  // 000000005178: 7ca80414
	v_mov_b32_e32 v1, s37                                      // 00000000517c: 7e020225
	s_and_b32 s6, s6, exec_lo                                  // 000000005180: 8b067e06
	s_cselect_b32 s7, s1, s3                                   // 000000005184: 98070301
	v_mov_b32_e32 v7, s37                                      // 000000005188: 7e0e0225
	v_or_b32_e32 v6, s36, v50                                  // 00000000518c: 380c6424
	s_wait_alu depctr_va_vcc(0)                                // 000000005190: bf88ff9d
	v_cndmask_b32_e32 v20, 0, v2, vcc_lo                       // 000000005194: 02280480
	v_cmp_gt_i64_e64 s1, s[20:21], v[0:1]                      // 000000005198: d4540001 02020014
	v_cndmask_b32_e32 v21, 0, v3, vcc_lo                       // 0000000051a0: 022a0680
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 0000000051a4: 7ca80814
	v_or_b32_e32 v2, s36, v48                                  // 0000000051a8: 38046024
	s_cselect_b32 s6, s0, s2                                   // 0000000051ac: 98060200
	v_cmp_gt_i64_e64 s0, s[22:23], v[34:35]                    // 0000000051b0: d4540000 02024416
	v_cndmask_b32_e64 v18, 0, v0, s1                           // 0000000051b8: d5010012 00060080
	v_or_b32_e32 v0, s36, v49                                  // 0000000051c0: 38006224
	v_cndmask_b32_e64 v19, 0, v1, s1                           // 0000000051c4: d5010013 00060280
	s_wait_alu depctr_va_vcc(0)                                // 0000000051cc: bf88ff9d
	v_cndmask_b32_e32 v22, 0, v4, vcc_lo                       // 0000000051d0: 022c0880
	v_cmp_gt_i64_e64 s1, s[20:21], v[2:3]                      // 0000000051d4: d4540001 02020414
	v_cndmask_b32_e32 v23, 0, v5, vcc_lo                       // 0000000051dc: 022e0a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 0000000051e0: 7ca80014
	v_or_b32_e32 v4, s36, v51                                  // 0000000051e4: 38086624
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v94, 0            // 0000000051e8: ca100080 675e0080
	s_wait_alu depctr_va_sdst(0)                               // 0000000051f0: bf88f19f
	v_cndmask_b32_e64 v26, 0, v2, s1                           // 0000000051f4: d501001a 00060480
	v_mov_b32_e32 v2, s37                                      // 0000000051fc: 7e040225
	v_cmp_gt_i64_e64 s2, s[20:21], v[6:7]                      // 000000005200: d4540002 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000005208: bf88ff9d
	v_dual_cndmask_b32 v24, 0, v0 :: v_dual_cndmask_b32 v25, 0, v1// 00000000520c: ca520080 18180280
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000005214: 7ca80814
	v_or_b32_e32 v0, s33, v54                                  // 000000005218: 38006c21
	v_cndmask_b32_e64 v27, 0, v3, s1                           // 00000000521c: d501001b 00060680
	s_wait_alu depctr_va_sdst(0)                               // 000000005224: bf88f19f
	v_cndmask_b32_e64 v28, 0, v6, s2                           // 000000005228: d501001c 000a0c80
	v_cndmask_b32_e64 v29, 0, v7, s2                           // 000000005230: d501001d 000a0e80
	v_or_b32_e32 v3, s33, v52                                  // 000000005238: 38066821
	v_cmp_gt_i64_e64 s2, s[20:21], v[0:1]                      // 00000000523c: d4540002 02020014
	s_wait_alu depctr_va_vcc(0)                                // 000000005244: bf88ff9d
	v_cndmask_b32_e32 v30, 0, v4, vcc_lo                       // 000000005248: 023c0880
	v_dual_mov_b32 v4, s37 :: v_dual_cndmask_b32 v31, 0, v5    // 00000000524c: ca120025 041e0a80
	v_mov_b32_e32 v110, 0                                      // 000000005254: 7edc0280
	v_or_b32_e32 v1, s33, v53                                  // 000000005258: 38026a21
	s_wait_alu depctr_va_sdst(0)                               // 00000000525c: bf88f19f
	v_cndmask_b32_e64 v14, 0, v0, s2                           // 000000005260: d501000e 000a0080
	v_cndmask_b32_e64 v15, 0, s37, s2                          // 000000005268: d501000f 00084a80
	v_cmp_gt_i64_e64 s2, s[20:21], v[3:4]                      // 000000005270: d4540002 02020614
	v_or_b32_e32 v4, s33, v51                                  // 000000005278: 38086621
	v_mov_b32_e32 v33, s37                                     // 00000000527c: 7e420225
	v_or_b32_e32 v0, s33, v49                                  // 000000005280: 38006221
	v_mov_b32_e32 v100, 0                                      // 000000005284: 7ec80280
	v_cmp_gt_i64_e64 s1, s[22:23], v[40:41]                    // 000000005288: d4540001 02025016
	s_wait_alu depctr_va_sdst(0)                               // 000000005290: bf88f19f
	v_cndmask_b32_e64 v8, 0, v3, s2                            // 000000005294: d5010008 000a0680
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[32:33]                // 00000000529c: 7ca84014
	v_mov_b32_e32 v3, s37                                      // 0000000052a0: 7e060225
	v_cndmask_b32_e64 v9, 0, s37, s2                           // 0000000052a4: d5010009 00084a80
	v_mov_b32_e32 v102, 0                                      // 0000000052ac: 7ecc0280
	v_mul_lo_u32 v45, v8, s39                                  // 0000000052b0: d72c002d 02004f08
	v_mul_lo_u32 v58, v15, s38                                 // 0000000052b8: d72c003a 02004d0f
	s_wait_alu depctr_va_vcc(0)                                // 0000000052c0: bf88ff9d
	v_cndmask_b32_e32 v54, 0, v32, vcc_lo                      // 0000000052c4: 026c4080
	v_cndmask_b32_e64 v52, 0, s37, vcc_lo                      // 0000000052c8: d5010034 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[1:2]                  // 0000000052d0: 7ca80214
	v_or_b32_e32 v2, s33, v50                                  // 0000000052d4: 38046421
	v_mul_lo_u32 v44, v9, s38                                  // 0000000052d8: d72c002c 02004d09
	v_mad_co_u64_u32 v[8:9], null, v8, s38, 0                  // 0000000052e0: d6fe7c08 02004d08
	v_mov_b32_e32 v96, 0                                       // 0000000052e8: 7ec00280
	v_mul_lo_u32 v59, v14, s39                                 // 0000000052ec: d72c003b 02004f0e
	s_wait_alu depctr_va_vcc(0)                                // 0000000052f4: bf88ff9d
	v_cndmask_b32_e32 v12, 0, v1, vcc_lo                       // 0000000052f8: 02180280
	v_cndmask_b32_e64 v13, 0, s37, vcc_lo                      // 0000000052fc: d501000d 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[4:5]                  // 000000005304: 7ca80814
	v_mov_b32_e32 v1, s37                                      // 000000005308: 7e020225
	v_cmp_gt_i64_e64 s2, s[20:21], v[2:3]                      // 00000000530c: d4540002 02020414
	v_mad_co_u64_u32 v[14:15], null, v14, s38, 0               // 000000005314: d6fe7c0e 02004d0e
	v_add3_u32 v9, v9, v45, v44                                // 00000000531c: d6550009 04b25b09
	v_mul_lo_u32 v53, v13, s38                                 // 000000005324: d72c0035 02004d0d
	s_wait_alu depctr_va_vcc(0)                                // 00000000532c: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v4, vcc_lo                        // 000000005330: 020c0880
	v_or_b32_e32 v4, s33, v48                                  // 000000005334: 38086021
	v_cndmask_b32_e64 v7, 0, s37, vcc_lo                       // 000000005338: d5010007 01a84a80
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[0:1]                  // 000000005340: 7ca80014
	v_mul_lo_u32 v57, v12, s39                                 // 000000005344: d72c0039 02004f0c
	v_mul_lo_u32 v3, v6, s39                                   // 00000000534c: d72c0003 02004f06
	v_cmp_gt_i64_e64 s3, s[20:21], v[4:5]                      // 000000005354: d4540003 02020814
	v_mul_lo_u32 v1, v7, s38                                   // 00000000535c: d72c0001 02004d07
	v_mad_co_u64_u32 v[6:7], null, v6, s38, 0                  // 000000005364: d6fe7c06 02004d06
	s_wait_alu depctr_va_vcc(0)                                // 00000000536c: bf88ff9d
	v_cndmask_b32_e32 v5, 0, v0, vcc_lo                        // 000000005370: 020a0080
	s_wait_alu depctr_va_sdst(0)                               // 000000005374: bf88f19f
	v_cndmask_b32_e64 v0, 0, v2, s2                            // 000000005378: d5010000 000a0480
	v_cndmask_b32_e64 v2, 0, s37, s2                           // 000000005380: d5010002 00084a80
	v_cndmask_b32_e64 v10, 0, s37, vcc_lo                      // 000000005388: d501000a 01a84a80
	v_cndmask_b32_e64 v4, 0, v4, s3                            // 000000005390: d5010004 000e0880
	v_cndmask_b32_e64 v11, 0, s37, s3                          // 000000005398: d501000b 000c4a80
	v_add_co_u32 v55, s2, s40, v64                             // 0000000053a0: d7000237 02028028
	s_wait_alu depctr_va_sdst(0)                               // 0000000053a8: bf88f19f
	v_add_co_ci_u32_e64 v56, null, s41, 0, s2                  // 0000000053ac: d5207c38 00090029
	v_add3_u32 v7, v7, v3, v1                                  // 0000000053b4: d6550007 04060707
	v_mul_lo_u32 v40, v2, s38                                  // 0000000053bc: d72c0028 02004d02
	v_mul_lo_u32 v41, v0, s39                                  // 0000000053c4: d72c0029 02004f00
	v_mad_co_u64_u32 v[0:1], null, v0, s38, 0                  // 0000000053cc: d6fe7c00 02004d00
	v_mul_lo_u32 v11, v11, s38                                 // 0000000053d4: d72c000b 02004d0b
	v_mul_lo_u32 v42, v4, s39                                  // 0000000053dc: d72c002a 02004f04
	v_mad_co_u64_u32 v[2:3], null, v4, s38, 0                  // 0000000053e4: d6fe7c02 02004d04
	v_mul_lo_u32 v10, v10, s38                                 // 0000000053ec: d72c000a 02004d0a
	v_mul_lo_u32 v43, v5, s39                                  // 0000000053f4: d72c002b 02004f05
	v_mad_co_u64_u32 v[4:5], null, v5, s38, 0                  // 0000000053fc: d6fe7c04 02004d05
	v_add_co_u32 v46, vcc_lo, v55, 16                          // 000000005404: d7006a2e 02012137
	s_wait_alu depctr_va_vcc(0)                                // 00000000540c: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, 0, v56, vcc_lo              // 000000005410: d5207c2f 01aa7080
	v_add3_u32 v1, v1, v41, v40                                // 000000005418: d6550001 04a25301
	v_add3_u32 v3, v3, v42, v11                                // 000000005420: d6550003 042e5503
	v_mul_lo_u32 v51, s25, v46                                 // 000000005428: d72c0033 02025c19
	v_add3_u32 v5, v5, v43, v10                                // 000000005430: d6550005 042a5705
	v_mul_lo_u32 v50, s24, v47                                 // 000000005438: d72c0032 02025e18
	v_mad_co_u64_u32 v[10:11], null, s24, v46, v[38:39]        // 000000005440: d6fe7c0a 049a5c18
	v_mad_co_u64_u32 v[12:13], null, v12, s38, 0               // 000000005448: d6fe7c0c 02004d0c
	v_lshlrev_b64_e32 v[40:41], 2, v[6:7]                      // 000000005450: 3e500c82
	v_lshlrev_b64_e32 v[42:43], 2, v[0:1]                      // 000000005454: 3e540082
	v_lshlrev_b64_e32 v[44:45], 2, v[2:3]                      // 000000005458: 3e580482
	v_lshlrev_b64_e32 v[46:47], 2, v[4:5]                      // 00000000545c: 3e5c0882
	v_lshlrev_b64_e32 v[48:49], 2, v[8:9]                      // 000000005460: 3e601082
	v_add3_u32 v15, v15, v59, v58                              // 000000005464: d655000f 04ea770f
	v_mul_lo_u32 v58, v54, s39                                 // 00000000546c: d72c003a 02004f36
	v_mad_co_u64_u32 v[0:1], null, v54, s38, 0                 // 000000005474: d6fe7c00 02004d36
	v_mul_lo_u32 v54, v30, s39                                 // 00000000547c: d72c0036 02004f1e
	v_mad_co_u64_u32 v[2:3], null, v30, s38, 0                 // 000000005484: d6fe7c02 02004d1e
	v_mul_lo_u32 v30, v28, s39                                 // 00000000548c: d72c001e 02004f1c
	v_mad_co_u64_u32 v[4:5], null, v28, s38, 0                 // 000000005494: d6fe7c04 02004d1c
	v_mul_lo_u32 v28, v26, s39                                 // 00000000549c: d72c001c 02004f1a
	v_mad_co_u64_u32 v[6:7], null, v26, s38, 0                 // 0000000054a4: d6fe7c06 02004d1a
	v_mul_lo_u32 v25, v25, s38                                 // 0000000054ac: d72c0019 02004d19
	v_mul_lo_u32 v26, v24, s39                                 // 0000000054b4: d72c001a 02004f18
	v_mad_co_u64_u32 v[8:9], null, v24, s38, 0                 // 0000000054bc: d6fe7c08 02004d18
	v_add3_u32 v11, v51, v11, v50                              // 0000000054c4: d655000b 04ca1733
	v_add3_u32 v13, v13, v57, v53                              // 0000000054cc: d655000d 04d6730d
	v_mul_lo_u32 v57, v52, s38                                 // 0000000054d4: d72c0039 02004d34
	v_mul_lo_u32 v29, v29, s38                                 // 0000000054dc: d72c001d 02004d1d
	v_mul_lo_u32 v31, v31, s38                                 // 0000000054e4: d72c001f 02004d1f
	v_mul_lo_u32 v27, v27, s38                                 // 0000000054ec: d72c001b 02004d1b
	v_add_co_u32 v97, vcc_lo, s34, v10                         // 0000000054f4: d7006a61 02021422
	s_wait_alu depctr_va_vcc(0)                                // 0000000054fc: bf88ff9d
	v_add_co_ci_u32_e64 v98, null, s35, v11, vcc_lo            // 000000005500: d5207c62 01aa1623
	v_add3_u32 v9, v9, v26, v25                                // 000000005508: d6550009 04663509
	v_mul_lo_u32 v24, s24, v56                                 // 000000005510: d72c0018 02027018
	v_mul_lo_u32 v25, s25, v55                                 // 000000005518: d72c0019 02026e19
	v_mad_co_u64_u32 v[10:11], null, s24, v55, v[38:39]        // 000000005520: d6fe7c0a 049a6e18
	v_add3_u32 v1, v1, v58, v57                                // 000000005528: d6550001 04e67501
	v_add3_u32 v5, v5, v30, v29                                // 000000005530: d6550005 04763d05
	v_add3_u32 v3, v3, v54, v31                                // 000000005538: d6550003 047e6d03
	v_add3_u32 v7, v7, v28, v27                                // 000000005540: d6550007 046e3907
	v_lshlrev_b64_e32 v[62:63], 2, v[8:9]                      // 000000005548: 3e7c1082
	v_add_co_u32 v9, s2, s36, v64                              // 00000000554c: d7000209 02028024
	v_lshlrev_b64_e32 v[54:55], 2, v[0:1]                      // 000000005554: 3e6c0082
	v_lshlrev_b64_e32 v[58:59], 2, v[4:5]                      // 000000005558: 3e740882
	v_add3_u32 v8, v25, v11, v24                               // 00000000555c: d6550008 04621719
	v_mul_lo_u32 v4, v19, s38                                  // 000000005564: d72c0004 02004d13
	v_mul_lo_u32 v5, v18, s39                                  // 00000000556c: d72c0005 02004f12
	v_mad_co_u64_u32 v[0:1], null, v18, s38, 0                 // 000000005574: d6fe7c00 02004d12
	s_wait_alu depctr_va_sdst(0)                               // 00000000557c: bf88f19f
	v_add_co_ci_u32_e64 v11, null, s37, 0, s2                  // 000000005580: d5207c0b 00090025
	v_lshlrev_b64_e32 v[56:57], 2, v[2:3]                      // 000000005588: 3e700482
	v_lshlrev_b64_e32 v[60:61], 2, v[6:7]                      // 00000000558c: 3e780c82
	v_mul_lo_u32 v6, v17, s38                                  // 000000005590: d72c0006 02004d11
	v_mul_lo_u32 v7, v16, s39                                  // 000000005598: d72c0007 02004f10
	v_mad_co_u64_u32 v[2:3], null, v16, s38, 0                 // 0000000055a0: d6fe7c02 02004d10
	v_add_co_u32 v16, vcc_lo, v9, 16                           // 0000000055a8: d7006a10 02012109
	s_wait_alu depctr_va_vcc(0)                                // 0000000055b0: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, 0, v11, vcc_lo              // 0000000055b4: d5207c11 01aa1680
	v_add3_u32 v1, v1, v5, v4                                  // 0000000055bc: d6550001 04120b01
	v_mul_lo_u32 v11, s24, v11                                 // 0000000055c4: d72c000b 02021618
	v_mul_lo_u32 v18, s25, v9                                  // 0000000055cc: d72c0012 02021219
	v_mad_co_u64_u32 v[4:5], null, s24, v9, v[38:39]           // 0000000055d4: d6fe7c04 049a1218
	v_lshlrev_b64_e32 v[50:51], 2, v[12:13]                    // 0000000055dc: 3e641882
	v_lshlrev_b64_e32 v[52:53], 2, v[14:15]                    // 0000000055e0: 3e681c82
	v_mul_lo_u32 v23, v23, s38                                 // 0000000055e4: d72c0017 02004d17
	v_mul_lo_u32 v26, v22, s39                                 // 0000000055ec: d72c001a 02004f16
	v_mad_co_u64_u32 v[12:13], null, v22, s38, 0               // 0000000055f4: d6fe7c0c 02004d16
	v_mul_lo_u32 v21, v21, s38                                 // 0000000055fc: d72c0015 02004d15
	v_mul_lo_u32 v22, v20, s39                                 // 000000005604: d72c0016 02004f14
	v_mad_co_u64_u32 v[14:15], null, v20, s38, 0               // 00000000560c: d6fe7c0e 02004d14
	v_add3_u32 v3, v3, v7, v6                                  // 000000005614: d6550003 041a0f03
	v_mul_lo_u32 v9, s24, v17                                  // 00000000561c: d72c0009 02022218
	v_mul_lo_u32 v17, s25, v16                                 // 000000005624: d72c0011 02022019
	v_mad_co_u64_u32 v[6:7], null, s24, v16, v[38:39]          // 00000000562c: d6fe7c06 049a2018
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]                      // 000000005634: 3e840082
	v_add3_u32 v0, v18, v5, v11                                // 000000005638: d6550000 042e0b12
	v_add3_u32 v13, v13, v26, v23                              // 000000005640: d655000d 045e350d
	v_add3_u32 v15, v15, v22, v21                              // 000000005648: d655000f 04562d0f
	v_add_co_u32 v104, vcc_lo, s34, v10                        // 000000005650: d7006a68 02021422
	s_wait_alu depctr_va_vcc(0)                                // 000000005658: bf88ff9d
	v_add_co_ci_u32_e64 v105, null, s35, v8, vcc_lo            // 00000000565c: d5207c69 01aa1023
	v_add3_u32 v1, v17, v7, v9                                 // 000000005664: d6550001 04260f11
	v_add_co_u32 v106, vcc_lo, s30, v4                         // 00000000566c: d7006a6a 0202081e
	s_wait_alu depctr_va_vcc(0)                                // 000000005674: bf88ff9d
	v_add_co_ci_u32_e64 v107, null, s31, v0, vcc_lo            // 000000005678: d5207c6b 01aa001f
	v_add_co_u32 v108, vcc_lo, s30, v6                         // 000000005680: d7006a6c 02020c1e
	v_lshlrev_b64_e32 v[38:39], 2, v[12:13]                    // 000000005688: 3e4c1882
	v_lshlrev_b64_e32 v[64:65], 2, v[14:15]                    // 00000000568c: 3e801c82
	v_lshlrev_b64_e32 v[68:69], 2, v[2:3]                      // 000000005690: 3e880482
	s_wait_alu depctr_va_vcc(0)                                // 000000005694: bf88ff9d
	v_add_co_ci_u32_e64 v109, null, s31, v1, vcc_lo            // 000000005698: d5207c6d 01aa021f
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v84, 0            // 0000000056a0: ca100080 65540080
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v82, 0             // 0000000056a8: ca100080 63520080
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v80, 0             // 0000000056b0: ca100080 5f500080
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v92, 0             // 0000000056b8: ca100080 555c0080
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v90, 0             // 0000000056c0: ca100080 535a0080
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v88, 0             // 0000000056c8: ca100080 51580080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v86, 0             // 0000000056d0: ca100080 4f560080
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v78, 0             // 0000000056d8: ca100080 4d4e0080
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v76, 0             // 0000000056e0: ca100080 5d4c0080
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v74, 0             // 0000000056e8: ca100080 5b4a0080
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v72, 0             // 0000000056f0: ca100080 59480080
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v70, 0             // 0000000056f8: ca100080 57460080
	v_mov_b32_e32 v75, 0                                       // 000000005700: 7e960280
	v_mov_b32_e32 v73, 0                                       // 000000005704: 7e920280
	v_mov_b32_e32 v71, 0                                       // 000000005708: 7e8e0280
	s_lshl_b64 s[2:3], s[6:7], 2                               // 00000000570c: 84828206
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000005710: 84848204
	s_mov_b64 s[8:9], 0                                        // 000000005714: be880180
	s_delay_alu instid0(salu_cycle_1)                          // 000000005718: bf870009
	v_add_co_u32 v111, vcc_lo, v106, s8                        // 00000000571c: d7006a6f 0200116a
	s_wait_alu depctr_va_vcc(0)                                // 000000005724: bf88ff9d
	v_add_co_ci_u32_e64 v112, null, s9, v107, vcc_lo           // 000000005728: d5207c70 01aad609
	v_add_co_u32 v113, vcc_lo, v108, s8                        // 000000005730: d7006a71 0200116c
	s_wait_alu depctr_va_vcc(0)                                // 000000005738: bf88ff9d
	v_add_co_ci_u32_e64 v114, null, s9, v109, vcc_lo           // 00000000573c: d5207c72 01aada09
	v_add_co_u32 v117, vcc_lo, v104, s8                        // 000000005744: d7006a75 02001168
	s_wait_alu depctr_va_vcc(0)                                // 00000000574c: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, s9, v105, vcc_lo           // 000000005750: d5207c76 01aad209
	v_add_co_u32 v119, vcc_lo, v97, s8                         // 000000005758: d7006a77 02001161
	s_wait_alu depctr_va_vcc(0)                                // 000000005760: bf88ff9d
	v_add_co_ci_u32_e64 v120, null, s9, v98, vcc_lo            // 000000005764: d5207c78 01aac409
	s_clause 0x1                                               // 00000000576c: bf850001
	global_load_b64 v[0:1], v[111:112], off                    // 000000005770: ee05407c 00000000 0000006f
	global_load_b64 v[115:116], v[113:114], off                // 00000000577c: ee05407c 00000073 00000071
	s_clause 0x1                                               // 000000005788: bf850001
	global_load_b64 v[2:3], v[117:118], off                    // 00000000578c: ee05407c 00000002 00000075
	global_load_b64 v[121:122], v[119:120], off                // 000000005798: ee05407c 00000079 00000077
	s_add_nc_u64 s[6:7], s[8:9], 0x80                          // 0000000057a4: a986ff08 00000080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057ac: bf88ff9e
	s_add_nc_u64 s[8:9], s[28:29], s[2:3]                      // 0000000057b0: a988021c
	s_wait_loadcnt 0x1                                         // 0000000057b4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[0:1], v[2:3], 0    // 0000000057b8: cc464018 1a020500
	s_wait_loadcnt 0x0                                         // 0000000057c0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[0:1], v[121:122], 0// 0000000057c4: cc464010 1a02f300
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[115:116], v[2:3], 0 // 0000000057cc: cc464008 1a020573
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[115:116], v[121:122], 0// 0000000057d4: cc464000 1a02f373
	s_clause 0x1                                               // 0000000057dc: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:16      // 0000000057e0: ee05407c 00000073 0000106f
	global_load_b64 v[121:122], v[113:114], off offset:16      // 0000000057ec: ee05407c 00000079 00001071
	s_clause 0x1                                               // 0000000057f8: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:16      // 0000000057fc: ee05407c 0000007b 00001075
	global_load_b64 v[125:126], v[119:120], off offset:16      // 000000005808: ee05407c 0000007d 00001077
	s_wait_loadcnt 0x1                                         // 000000005814: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 000000005818: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 000000005820: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 000000005824: cc464010 1c42fb73
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 00000000582c: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 000000005834: cc464000 1c02fb79
	s_clause 0x1                                               // 00000000583c: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:32      // 000000005840: ee05407c 00000073 0000206f
	global_load_b64 v[121:122], v[113:114], off offset:32      // 00000000584c: ee05407c 00000079 00002071
	s_clause 0x1                                               // 000000005858: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:32      // 00000000585c: ee05407c 0000007b 00002075
	global_load_b64 v[125:126], v[119:120], off offset:32      // 000000005868: ee05407c 0000007d 00002077
	s_wait_loadcnt 0x1                                         // 000000005874: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 000000005878: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 000000005880: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 000000005884: cc464010 1c42fb73
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 00000000588c: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 000000005894: cc464000 1c02fb79
	s_clause 0x1                                               // 00000000589c: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:48      // 0000000058a0: ee05407c 00000073 0000306f
	global_load_b64 v[121:122], v[113:114], off offset:48      // 0000000058ac: ee05407c 00000079 00003071
	s_clause 0x1                                               // 0000000058b8: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:48      // 0000000058bc: ee05407c 0000007b 00003075
	global_load_b64 v[125:126], v[119:120], off offset:48      // 0000000058c8: ee05407c 0000007d 00003077
	s_wait_loadcnt 0x1                                         // 0000000058d4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 0000000058d8: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 0000000058e0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 0000000058e4: cc464010 1c42fb73
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 0000000058ec: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 0000000058f4: cc464000 1c02fb79
	s_clause 0x1                                               // 0000000058fc: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:64      // 000000005900: ee05407c 00000073 0000406f
	global_load_b64 v[121:122], v[113:114], off offset:64      // 00000000590c: ee05407c 00000079 00004071
	s_clause 0x1                                               // 000000005918: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:64      // 00000000591c: ee05407c 0000007b 00004075
	global_load_b64 v[125:126], v[119:120], off offset:64      // 000000005928: ee05407c 0000007d 00004077
	s_wait_loadcnt 0x1                                         // 000000005934: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 000000005938: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 000000005940: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 000000005944: cc464010 1c42fb73
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 00000000594c: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 000000005954: cc464000 1c02fb79
	s_clause 0x1                                               // 00000000595c: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:80      // 000000005960: ee05407c 00000073 0000506f
	global_load_b64 v[121:122], v[113:114], off offset:80      // 00000000596c: ee05407c 00000079 00005071
	s_clause 0x1                                               // 000000005978: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:80      // 00000000597c: ee05407c 0000007b 00005075
	global_load_b64 v[125:126], v[119:120], off offset:80      // 000000005988: ee05407c 0000007d 00005077
	s_wait_loadcnt 0x1                                         // 000000005994: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 000000005998: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 0000000059a0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 0000000059a4: cc464010 1c42fb73
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 0000000059ac: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 0000000059b4: cc464000 1c02fb79
	s_clause 0x1                                               // 0000000059bc: bf850001
	global_load_b64 v[115:116], v[111:112], off offset:96      // 0000000059c0: ee05407c 00000073 0000606f
	global_load_b64 v[121:122], v[113:114], off offset:96      // 0000000059cc: ee05407c 00000079 00006071
	s_clause 0x1                                               // 0000000059d8: bf850001
	global_load_b64 v[123:124], v[117:118], off offset:96      // 0000000059dc: ee05407c 0000007b 00006075
	global_load_b64 v[125:126], v[119:120], off offset:96      // 0000000059e8: ee05407c 0000007d 00006077
	s_wait_loadcnt 0x1                                         // 0000000059f4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 0000000059f8: cc464018 1c62f773
	s_wait_loadcnt 0x0                                         // 000000005a00: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[115:116], v[125:126], v[16:23]// 000000005a04: cc464010 1c42fb73
	s_clause 0x1                                               // 000000005a0c: bf850001
	global_load_b64 v[111:112], v[111:112], off offset:112     // 000000005a10: ee05407c 0000006f 0000706f
	global_load_b64 v[113:114], v[113:114], off offset:112     // 000000005a1c: ee05407c 00000071 00007071
	s_clause 0x1                                               // 000000005a28: bf850001
	global_load_b64 v[115:116], v[117:118], off offset:112     // 000000005a2c: ee05407c 00000073 00007075
	global_load_b64 v[117:118], v[119:120], off offset:112     // 000000005a38: ee05407c 00000075 00007077
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[121:122], v[123:124], v[8:15]// 000000005a44: cc464008 1c22f779
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[121:122], v[125:126], v[0:7]// 000000005a4c: cc464000 1c02fb79
	s_wait_loadcnt 0x1                                         // 000000005a54: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[111:112], v[115:116], v[24:31]// 000000005a58: cc464018 1c62e76f
	s_wait_loadcnt 0x0                                         // 000000005a60: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[111:112], v[117:118], v[16:23]// 000000005a64: cc464010 1c42eb6f
	v_add_co_u32 v111, vcc_lo, s26, v68                        // 000000005a6c: d7006a6f 0202881a
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[113:114], v[115:116], v[8:15]// 000000005a74: cc464008 1c22e771
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[113:114], v[117:118], v[0:7]// 000000005a7c: cc464000 1c02eb71
	global_load_b32 v114, v110, s[8:9]                         // 000000005a84: ee050008 00000072 0000006e
	s_wait_alu depctr_va_vcc(0)                                // 000000005a90: bf88ff9d
	v_add_co_ci_u32_e64 v112, null, s27, v69, vcc_lo           // 000000005a94: d5207c70 01aa8a1b
	s_load_b32 s8, s[28:29], 0x0                               // 000000005a9c: f400020e f8000000
	s_add_nc_u64 s[28:29], s[28:29], s[4:5]                    // 000000005aa4: a99c041c
	global_load_b32 v115, v[111:112], off                      // 000000005aa8: ee05007c 00000073 0000006f
	s_wait_loadcnt 0x1                                         // 000000005ab4: bfc00001
	s_wait_kmcnt 0x0                                           // 000000005ab8: bfc70000
	v_cndmask_b32_e64 v111, s8, v114, s0                       // 000000005abc: d501006f 0002e408
	s_wait_loadcnt 0x0                                         // 000000005ac4: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ac8: bf870091
	v_mul_f32_e32 v112, v115, v111                             // 000000005acc: 10e0df73
	v_mul_f32_e32 v24, v24, v112                               // 000000005ad0: 1030e118
	v_add_co_u32 v112, vcc_lo, s26, v66                        // 000000005ad4: d7006a70 0202841a
	s_wait_alu depctr_va_vcc(0)                                // 000000005adc: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, s27, v67, vcc_lo           // 000000005ae0: d5207c71 01aa861b
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 000000005ae8: bf8700c3
	v_add_f32_e32 v103, v103, v24                              // 000000005aec: 06ce3167
	global_load_b32 v112, v[112:113], off                      // 000000005af0: ee05007c 00000070 00000070
	s_wait_loadcnt 0x0                                         // 000000005afc: bfc00000
	v_mul_f32_e32 v24, v111, v112                              // 000000005b00: 1030e16f
	v_mul_f32_e32 v24, v25, v24                                // 000000005b04: 10303119
	s_delay_alu instid0(valu_dep_1)                            // 000000005b08: bf870001
	v_add_f32_e32 v102, v102, v24                              // 000000005b0c: 06cc3166
	v_add_co_u32 v24, vcc_lo, s26, v64                         // 000000005b10: d7006a18 0202801a
	s_wait_alu depctr_va_vcc(0)                                // 000000005b18: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v65, vcc_lo            // 000000005b1c: d5207c19 01aa821b
	global_load_b32 v113, v[24:25], off                        // 000000005b24: ee05007c 00000071 00000018
	s_wait_loadcnt 0x0                                         // 000000005b30: bfc00000
	v_mul_f32_e32 v24, v111, v113                              // 000000005b34: 1030e36f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b38: bf870091
	v_mul_f32_e32 v24, v26, v24                                // 000000005b3c: 1030311a
	v_add_f32_e32 v101, v101, v24                              // 000000005b40: 06ca3165
	v_add_co_u32 v24, vcc_lo, s26, v38                         // 000000005b44: d7006a18 02024c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005b4c: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v39, vcc_lo            // 000000005b50: d5207c19 01aa4e1b
	global_load_b32 v26, v[24:25], off                         // 000000005b58: ee05007c 0000001a 00000018
	s_wait_loadcnt 0x0                                         // 000000005b64: bfc00000
	v_mul_f32_e32 v24, v111, v26                               // 000000005b68: 1030356f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b6c: bf870091
	v_mul_f32_e32 v24, v27, v24                                // 000000005b70: 1030311b
	v_add_f32_e32 v100, v100, v24                              // 000000005b74: 06c83164
	v_add_co_u32 v24, vcc_lo, s26, v62                         // 000000005b78: d7006a18 02027c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005b80: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v63, vcc_lo            // 000000005b84: d5207c19 01aa7e1b
	global_load_b32 v27, v[24:25], off                         // 000000005b8c: ee05007c 0000001b 00000018
	s_wait_loadcnt 0x0                                         // 000000005b98: bfc00000
	v_mul_f32_e32 v24, v111, v27                               // 000000005b9c: 1030376f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ba0: bf870091
	v_mul_f32_e32 v24, v28, v24                                // 000000005ba4: 1030311c
	v_add_f32_e32 v99, v99, v24                                // 000000005ba8: 06c63163
	v_add_co_u32 v24, vcc_lo, s26, v60                         // 000000005bac: d7006a18 0202781a
	s_wait_alu depctr_va_vcc(0)                                // 000000005bb4: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v61, vcc_lo            // 000000005bb8: d5207c19 01aa7a1b
	global_load_b32 v28, v[24:25], off                         // 000000005bc0: ee05007c 0000001c 00000018
	s_wait_loadcnt 0x0                                         // 000000005bcc: bfc00000
	v_mul_f32_e32 v24, v111, v28                               // 000000005bd0: 1030396f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005bd4: bf870091
	v_mul_f32_e32 v24, v29, v24                                // 000000005bd8: 1030311d
	v_add_f32_e32 v96, v96, v24                                // 000000005bdc: 06c03160
	v_add_co_u32 v24, vcc_lo, s26, v58                         // 000000005be0: d7006a18 0202741a
	s_wait_alu depctr_va_vcc(0)                                // 000000005be8: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v59, vcc_lo            // 000000005bec: d5207c19 01aa761b
	global_load_b32 v29, v[24:25], off                         // 000000005bf4: ee05007c 0000001d 00000018
	s_wait_loadcnt 0x0                                         // 000000005c00: bfc00000
	v_mul_f32_e32 v24, v111, v29                               // 000000005c04: 10303b6f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c08: bf870091
	v_mul_f32_e32 v24, v30, v24                                // 000000005c0c: 1030311e
	v_add_f32_e32 v95, v95, v24                                // 000000005c10: 06be315f
	v_add_co_u32 v24, vcc_lo, s26, v56                         // 000000005c14: d7006a18 0202701a
	s_wait_alu depctr_va_vcc(0)                                // 000000005c1c: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s27, v57, vcc_lo            // 000000005c20: d5207c19 01aa721b
	global_load_b32 v25, v[24:25], off                         // 000000005c28: ee05007c 00000019 00000018
	s_wait_loadcnt 0x0                                         // 000000005c34: bfc00000
	v_mul_f32_e32 v24, v111, v25                               // 000000005c38: 1030336f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c3c: bf870091
	v_mul_f32_e32 v24, v31, v24                                // 000000005c40: 1030311f
	v_add_f32_e32 v94, v94, v24                                // 000000005c44: 06bc315e
	v_cndmask_b32_e64 v24, s8, v114, s1                        // 000000005c48: d5010018 0006e408
	v_cmp_lt_u64_e64 s8, s[6:7], s[24:25]                      // 000000005c50: d4590008 02003006
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000005c58: bf870092
	v_mul_f32_e32 v30, v115, v24                               // 000000005c5c: 103c3173
	v_mul_f32_e32 v16, v16, v30                                // 000000005c60: 10203d10
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000005c64: bf8700a1
	v_add_f32_e32 v85, v85, v16                                // 000000005c68: 06aa2155
	v_mul_f32_e32 v16, v24, v112                               // 000000005c6c: 1020e118
	v_mul_f32_e32 v16, v17, v16                                // 000000005c70: 10202111
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000005c74: bf8700a1
	v_add_f32_e32 v84, v84, v16                                // 000000005c78: 06a82154
	v_mul_f32_e32 v16, v24, v113                               // 000000005c7c: 1020e318
	v_mul_f32_e32 v16, v18, v16                                // 000000005c80: 10202112
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c84: bf870091
	v_dual_add_f32 v83, v83, v16 :: v_dual_mul_f32 v16, v24, v26// 000000005c88: c9062153 53103518
	v_mul_f32_e32 v16, v19, v16                                // 000000005c90: 10202113
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000005c94: bf8700a1
	v_add_f32_e32 v82, v82, v16                                // 000000005c98: 06a42152
	v_mul_f32_e32 v16, v24, v27                                // 000000005c9c: 10203718
	v_mul_f32_e32 v16, v20, v16                                // 000000005ca0: 10202114
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000005ca4: bf8700a1
	v_add_f32_e32 v81, v81, v16                                // 000000005ca8: 06a22151
	v_mul_f32_e32 v16, v24, v28                                // 000000005cac: 10203918
	v_mul_f32_e32 v16, v21, v16                                // 000000005cb0: 10202115
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000005cb4: bf8700a1
	v_add_f32_e32 v80, v80, v16                                // 000000005cb8: 06a02150
	v_mul_f32_e32 v16, v24, v29                                // 000000005cbc: 10203b18
	v_mul_f32_e32 v16, v22, v16                                // 000000005cc0: 10202116
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005cc4: bf870091
	v_dual_add_f32 v79, v79, v16 :: v_dual_mul_f32 v16, v24, v25// 000000005cc8: c906214f 4f103318
	v_mul_f32_e32 v16, v23, v16                                // 000000005cd0: 10202117
	s_delay_alu instid0(valu_dep_1)                            // 000000005cd4: bf870001
	v_add_f32_e32 v77, v77, v16                                // 000000005cd8: 069a214d
	v_add_co_u32 v16, vcc_lo, s26, v54                         // 000000005cdc: d7006a10 02026c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005ce4: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s27, v55, vcc_lo            // 000000005ce8: d5207c11 01aa6e1b
	global_load_b32 v18, v[16:17], off                         // 000000005cf0: ee05007c 00000012 00000010
	s_wait_loadcnt 0x0                                         // 000000005cfc: bfc00000
	v_mul_f32_e32 v16, v111, v18                               // 000000005d00: 1020256f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000005d04: bf8701c1
	v_mul_f32_e32 v8, v8, v16                                  // 000000005d08: 10102108
	v_add_co_u32 v16, vcc_lo, s26, v52                         // 000000005d0c: d7006a10 0202681a
	s_wait_alu depctr_va_vcc(0)                                // 000000005d14: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s27, v53, vcc_lo            // 000000005d18: d5207c11 01aa6a1b
	v_add_f32_e32 v93, v93, v8                                 // 000000005d20: 06ba115d
	global_load_b32 v16, v[16:17], off                         // 000000005d24: ee05007c 00000010 00000010
	s_wait_loadcnt 0x0                                         // 000000005d30: bfc00000
	v_mul_f32_e32 v8, v111, v16                                // 000000005d34: 1010216f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005d38: bf870091
	v_mul_f32_e32 v8, v9, v8                                   // 000000005d3c: 10101109
	v_add_f32_e32 v92, v92, v8                                 // 000000005d40: 06b8115c
	v_add_co_u32 v8, vcc_lo, s26, v50                          // 000000005d44: d7006a08 0202641a
	s_wait_alu depctr_va_vcc(0)                                // 000000005d4c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v51, vcc_lo             // 000000005d50: d5207c09 01aa661b
	global_load_b32 v17, v[8:9], off                           // 000000005d58: ee05007c 00000011 00000008
	s_wait_loadcnt 0x0                                         // 000000005d64: bfc00000
	v_mul_f32_e32 v8, v111, v17                                // 000000005d68: 1010236f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005d6c: bf870091
	v_mul_f32_e32 v8, v10, v8                                  // 000000005d70: 1010110a
	v_add_f32_e32 v91, v91, v8                                 // 000000005d74: 06b6115b
	v_add_co_u32 v8, vcc_lo, s26, v48                          // 000000005d78: d7006a08 0202601a
	s_wait_alu depctr_va_vcc(0)                                // 000000005d80: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v49, vcc_lo             // 000000005d84: d5207c09 01aa621b
	global_load_b32 v10, v[8:9], off                           // 000000005d8c: ee05007c 0000000a 00000008
	s_wait_loadcnt 0x0                                         // 000000005d98: bfc00000
	v_mul_f32_e32 v8, v111, v10                                // 000000005d9c: 1010156f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005da0: bf870091
	v_mul_f32_e32 v8, v11, v8                                  // 000000005da4: 1010110b
	v_add_f32_e32 v90, v90, v8                                 // 000000005da8: 06b4115a
	v_add_co_u32 v8, vcc_lo, s26, v46                          // 000000005dac: d7006a08 02025c1a
	s_wait_alu depctr_va_vcc(0)                                // 000000005db4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v47, vcc_lo             // 000000005db8: d5207c09 01aa5e1b
	global_load_b32 v11, v[8:9], off                           // 000000005dc0: ee05007c 0000000b 00000008
	s_wait_loadcnt 0x0                                         // 000000005dcc: bfc00000
	v_mul_f32_e32 v8, v111, v11                                // 000000005dd0: 1010176f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005dd4: bf870091
	v_mul_f32_e32 v8, v12, v8                                  // 000000005dd8: 1010110c
	v_add_f32_e32 v89, v89, v8                                 // 000000005ddc: 06b21159
	v_add_co_u32 v8, vcc_lo, s26, v44                          // 000000005de0: d7006a08 0202581a
	s_wait_alu depctr_va_vcc(0)                                // 000000005de8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v45, vcc_lo             // 000000005dec: d5207c09 01aa5a1b
	global_load_b32 v12, v[8:9], off                           // 000000005df4: ee05007c 0000000c 00000008
	s_wait_loadcnt 0x0                                         // 000000005e00: bfc00000
	v_mul_f32_e32 v8, v111, v12                                // 000000005e04: 1010196f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e08: bf870091
	v_mul_f32_e32 v8, v13, v8                                  // 000000005e0c: 1010110d
	v_add_f32_e32 v88, v88, v8                                 // 000000005e10: 06b01158
	v_add_co_u32 v8, vcc_lo, s26, v42                          // 000000005e14: d7006a08 0202541a
	s_wait_alu depctr_va_vcc(0)                                // 000000005e1c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v43, vcc_lo             // 000000005e20: d5207c09 01aa561b
	global_load_b32 v13, v[8:9], off                           // 000000005e28: ee05007c 0000000d 00000008
	s_wait_loadcnt 0x0                                         // 000000005e34: bfc00000
	v_mul_f32_e32 v8, v111, v13                                // 000000005e38: 10101b6f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e3c: bf870091
	v_mul_f32_e32 v8, v14, v8                                  // 000000005e40: 1010110e
	v_add_f32_e32 v87, v87, v8                                 // 000000005e44: 06ae1157
	v_add_co_u32 v8, vcc_lo, s26, v40                          // 000000005e48: d7006a08 0202501a
	s_wait_alu depctr_va_vcc(0)                                // 000000005e50: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s27, v41, vcc_lo             // 000000005e54: d5207c09 01aa521b
	s_add_nc_u64 s[26:27], s[26:27], 4                         // 000000005e5c: a99a841a
	s_and_b32 vcc_lo, exec_lo, s8                              // 000000005e60: 8b6a087e
	s_mov_b64 s[8:9], s[6:7]                                   // 000000005e64: be880106
	global_load_b32 v8, v[8:9], off                            // 000000005e68: ee05007c 00000008 00000008
	s_wait_loadcnt 0x0                                         // 000000005e74: bfc00000
	v_mul_f32_e32 v9, v111, v8                                 // 000000005e78: 1012116f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e7c: bf870091
	v_mul_f32_e32 v9, v15, v9                                  // 000000005e80: 1012130f
	v_dual_add_f32 v86, v86, v9 :: v_dual_mul_f32 v9, v24, v18 // 000000005e84: c9061356 56082518
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e8c: bf870091
	v_mul_f32_e32 v0, v0, v9                                   // 000000005e90: 10001300
	v_add_f32_e32 v78, v78, v0                                 // 000000005e94: 069c014e
	v_mul_f32_e32 v0, v24, v16                                 // 000000005e98: 10002118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e9c: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 000000005ea0: 10000101
	v_add_f32_e32 v76, v76, v0                                 // 000000005ea4: 0698014c
	v_mul_f32_e32 v0, v24, v17                                 // 000000005ea8: 10002318
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005eac: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 000000005eb0: 10000102
	v_dual_add_f32 v75, v75, v0 :: v_dual_mul_f32 v0, v24, v10 // 000000005eb4: c906014b 4b001518
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ebc: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 000000005ec0: 10000103
	v_add_f32_e32 v74, v74, v0                                 // 000000005ec4: 0694014a
	v_mul_f32_e32 v0, v24, v11                                 // 000000005ec8: 10001718
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ecc: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 000000005ed0: 10000104
	v_add_f32_e32 v73, v73, v0                                 // 000000005ed4: 06920149
	v_mul_f32_e32 v0, v24, v12                                 // 000000005ed8: 10001918
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005edc: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 000000005ee0: 10000105
	v_add_f32_e32 v72, v72, v0                                 // 000000005ee4: 06900148
	v_mul_f32_e32 v0, v24, v13                                 // 000000005ee8: 10001b18
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005eec: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 000000005ef0: 10000106
	v_add_f32_e32 v71, v71, v0                                 // 000000005ef4: 068e0147
	v_mul_f32_e32 v0, v24, v8                                  // 000000005ef8: 10001118
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005efc: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 000000005f00: 10000107
	v_add_f32_e32 v70, v70, v0                                 // 000000005f04: 068c0146
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f08: bf88ff9e
	s_cbranch_vccnz 65026                                      // 000000005f0c: bfa4fe02 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3c18>
	v_mul_lo_u32 v2, s23, v36                                  // 000000005f10: d72c0002 02024817
	v_mul_lo_u32 v3, s22, v37                                  // 000000005f18: d72c0003 02024a16
	v_mad_co_u64_u32 v[0:1], null, s22, v36, 0                 // 000000005f20: d6fe7c00 02024816
	v_bfe_u32 v4, v103, 16, 1                                  // 000000005f28: d6100004 02052167
	v_or_b32_e32 v5, 0x400000, v103                            // 000000005f30: 380aceff 00400000
	v_bfe_u32 v6, v102, 16, 1                                  // 000000005f38: d6100006 02052166
	v_cmp_u_f32_e32 vcc_lo, v103, v103                         // 000000005f40: 7c30cf67
	v_or_b32_e32 v7, 0x400000, v102                            // 000000005f44: 380eccff 00400000
	v_add3_u32 v4, v4, v103, 0x7fff                            // 000000005f4c: d6550004 03fecf04 00007fff
	s_lshl_b64 s[0:1], s[22:23], 1                             // 000000005f58: 84808116
	v_add3_u32 v1, v1, v3, v2                                  // 000000005f5c: d6550001 040a0701
	v_add3_u32 v6, v6, v102, 0x7fff                            // 000000005f64: d6550006 03fecd06 00007fff
	v_lshlrev_b64_e32 v[2:3], 1, v[34:35]                      // 000000005f70: 3e044481
	s_wait_alu depctr_va_vcc(0)                                // 000000005f74: bf88ff9d
	v_cndmask_b32_e32 v8, v4, v5, vcc_lo                       // 000000005f78: 02100b04
	v_or_b32_e32 v11, 0x400000, v101                           // 000000005f7c: 3816caff 00400000
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000005f84: 3e000081
	v_bfe_u32 v13, v100, 16, 1                                 // 000000005f88: d610000d 02052164
	v_or_b32_e32 v14, 0x400000, v100                           // 000000005f90: 381cc8ff 00400000
	v_bfe_u32 v19, v95, 16, 1                                  // 000000005f98: d6100013 0205215f
	v_or_b32_e32 v20, 0x400000, v95                            // 000000005fa0: 3828beff 00400000
	v_or_b32_e32 v17, 0x400000, v96                            // 000000005fa8: 3822c0ff 00400000
	v_add_co_u32 v4, vcc_lo, s18, v0                           // 000000005fb0: d7006a04 02020012
	s_wait_alu depctr_va_vcc(0)                                // 000000005fb8: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s19, v1, vcc_lo              // 000000005fbc: d5207c05 01aa0213
	v_cmp_u_f32_e32 vcc_lo, v102, v102                         // 000000005fc4: 7c30cd66
	v_add3_u32 v13, v13, v100, 0x7fff                          // 000000005fc8: d655000d 03fec90d 00007fff
	v_add3_u32 v19, v19, v95, 0x7fff                           // 000000005fd4: d6550013 03febf13 00007fff
	v_mul_lo_u32 v22, s23, v32                                 // 000000005fe0: d72c0016 02024017
	v_mul_lo_u32 v23, s22, v33                                 // 000000005fe8: d72c0017 02024216
	s_wait_alu depctr_va_vcc(0)                                // 000000005ff0: bf88ff9d
	v_cndmask_b32_e32 v9, v6, v7, vcc_lo                       // 000000005ff4: 02120f06
	v_add_co_u32 v0, vcc_lo, v4, v2                            // 000000005ff8: d7006a00 02020504
	s_wait_alu depctr_va_vcc(0)                                // 000000006000: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v5, v3, vcc_lo               // 000000006004: d5207c01 01aa0705
	s_wait_alu depctr_sa_sdst(0)                               // 00000000600c: bf88ff9e
	v_add_co_u32 v7, vcc_lo, v4, s0                            // 000000006010: d7006a07 02000104
	v_bfe_u32 v6, v101, 16, 1                                  // 000000006018: d6100006 02052165
	s_wait_alu depctr_va_vcc(0)                                // 000000006020: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v5, vcc_lo              // 000000006024: d5207c0a 01aa0a01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000602c: bf870193
	v_add_co_u32 v4, vcc_lo, v7, v2                            // 000000006030: d7006a04 02020507
	v_add3_u32 v6, v6, v101, 0x7fff                            // 000000006038: d6550006 03fecb06 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006044: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000006048: bf870003
	v_add_co_ci_u32_e64 v5, null, v10, v3, vcc_lo              // 00000000604c: d5207c05 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 000000006054: 7c30cb65
	v_or_b32_e32 v24, 0x400000, v94                            // 000000006058: 3830bcff 00400000
	v_bfe_u32 v25, v92, 16, 1                                  // 000000006060: d6100019 0205215c
	v_or_b32_e32 v26, 0x400000, v92                            // 000000006068: 3834b8ff 00400000
	v_or_b32_e32 v29, 0x400000, v90                            // 000000006070: 383ab4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006078: bf88ff9d
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo                     // 00000000607c: 02161706
	v_add_co_u32 v12, vcc_lo, v7, s0                           // 000000006080: d7006a0c 02000107
	s_wait_alu depctr_va_vcc(0)                                // 000000006088: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 00000000608c: d5207c0a 01aa1401
	v_add3_u32 v25, v25, v92, 0x7fff                           // 000000006094: d6550019 03feb919 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000060a0: bf8701a3
	v_add_co_u32 v6, vcc_lo, v12, v2                           // 0000000060a4: d7006a06 0202050c
	s_wait_alu depctr_va_vcc(0)                                // 0000000060ac: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v10, v3, vcc_lo              // 0000000060b0: d5207c07 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v100, v100                         // 0000000060b8: 7c30c964
	v_bfe_u32 v31, v89, 16, 1                                  // 0000000060bc: d610001f 02052159
	v_bfe_u32 v37, v86, 16, 1                                  // 0000000060c4: d6100025 02052156
	v_or_b32_e32 v38, 0x400000, v86                            // 0000000060cc: 384cacff 00400000
	v_or_b32_e32 v35, 0x400000, v87                            // 0000000060d4: 3846aeff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000060dc: bf88ff9d
	v_cndmask_b32_e32 v14, v13, v14, vcc_lo                    // 0000000060e0: 021c1d0d
	s_clause 0x2                                               // 0000000060e4: bf850002
	global_store_d16_hi_b16 v[0:1], v8, off                    // 0000000060e8: ee09407c 04000000 00000000
	global_store_d16_hi_b16 v[4:5], v9, off                    // 0000000060f4: ee09407c 04800000 00000004
	global_store_d16_hi_b16 v[6:7], v11, off                   // 000000006100: ee09407c 05800000 00000006
	v_bfe_u32 v8, v99, 16, 1                                   // 00000000610c: d6100008 02052163
	v_add_co_u32 v11, vcc_lo, v12, s0                          // 000000006114: d7006a0b 0200010c
	s_wait_alu depctr_va_vcc(0)                                // 00000000611c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 000000006120: d5207c0a 01aa1401
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006128: bf870193
	v_add3_u32 v12, v8, v99, 0x7fff                            // 00000000612c: d655000c 03fec708 00007fff
	v_add_co_u32 v8, vcc_lo, v11, v2                           // 000000006138: d7006a08 0202050b
	v_or_b32_e32 v13, 0x400000, v99                            // 000000006140: 381ac6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006148: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v10, v3, vcc_lo              // 00000000614c: d5207c09 01aa070a
	v_cmp_u_f32_e32 vcc_lo, v99, v99                           // 000000006154: 7c30c763
	v_add3_u32 v31, v31, v89, 0x7fff                           // 000000006158: d655001f 03feb31f 00007fff
	v_add3_u32 v37, v37, v86, 0x7fff                           // 000000006164: d6550025 03fead25 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006170: bf88ff9d
	v_cndmask_b32_e32 v15, v12, v13, vcc_lo                    // 000000006174: 021e1b0c
	v_add_co_u32 v13, vcc_lo, v11, s0                          // 000000006178: d7006a0d 0200010b
	v_bfe_u32 v12, v96, 16, 1                                  // 000000006180: d610000c 02052160
	s_wait_alu depctr_va_vcc(0)                                // 000000006188: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s1, v10, vcc_lo             // 00000000618c: d5207c10 01aa1401
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006194: bf870193
	v_add_co_u32 v10, vcc_lo, v13, v2                          // 000000006198: d7006a0a 0202050d
	v_add3_u32 v12, v12, v96, 0x7fff                           // 0000000061a0: d655000c 03fec10c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000061ac: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000061b0: bf870003
	v_add_co_ci_u32_e64 v11, null, v16, v3, vcc_lo             // 0000000061b4: d5207c0b 01aa0710
	v_cmp_u_f32_e32 vcc_lo, v96, v96                           // 0000000061bc: 7c30c160
	s_wait_alu depctr_va_vcc(0)                                // 0000000061c0: bf88ff9d
	v_cndmask_b32_e32 v17, v12, v17, vcc_lo                    // 0000000061c4: 0222230c
	v_add_co_u32 v18, vcc_lo, v13, s0                          // 0000000061c8: d7006a12 0200010d
	s_wait_alu depctr_va_vcc(0)                                // 0000000061d0: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s1, v16, vcc_lo             // 0000000061d4: d5207c10 01aa2001
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000061dc: bf870122
	v_add_co_u32 v12, vcc_lo, v18, v2                          // 0000000061e0: d7006a0c 02020512
	s_wait_alu depctr_va_vcc(0)                                // 0000000061e8: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v16, v3, vcc_lo             // 0000000061ec: d5207c0d 01aa0710
	v_cmp_u_f32_e32 vcc_lo, v95, v95                           // 0000000061f4: 7c30bf5f
	s_wait_alu depctr_va_vcc(0)                                // 0000000061f8: bf88ff9d
	v_cndmask_b32_e32 v20, v19, v20, vcc_lo                    // 0000000061fc: 02282913
	s_clause 0x2                                               // 000000006200: bf850002
	global_store_d16_hi_b16 v[8:9], v14, off                   // 000000006204: ee09407c 07000000 00000008
	global_store_d16_hi_b16 v[10:11], v15, off                 // 000000006210: ee09407c 07800000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 00000000621c: ee09407c 08800000 0000000c
	v_bfe_u32 v14, v94, 16, 1                                  // 000000006228: d610000e 0205215e
	v_add_co_u32 v18, vcc_lo, v18, s0                          // 000000006230: d7006a12 02000112
	s_wait_alu depctr_va_vcc(0)                                // 000000006238: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v16, vcc_lo             // 00000000623c: d5207c13 01aa2001
	s_delay_alu instid0(valu_dep_3)                            // 000000006244: bf870003
	v_add3_u32 v21, v14, v94, 0x7fff                           // 000000006248: d6550015 03febd0e 00007fff
	v_mad_co_u64_u32 v[14:15], null, s22, v32, 0               // 000000006254: d6fe7c0e 02024016
	v_add_co_u32 v16, vcc_lo, v18, v2                          // 00000000625c: d7006a10 02020512
	s_wait_alu depctr_va_vcc(0)                                // 000000006264: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v19, v3, vcc_lo             // 000000006268: d5207c11 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v94, v94                           // 000000006270: 7c30bd5e
	v_or_b32_e32 v32, 0x400000, v89                            // 000000006274: 3840b2ff 00400000
	v_add3_u32 v15, v15, v23, v22                              // 00000000627c: d655000f 045a2f0f
	v_bfe_u32 v22, v93, 16, 1                                  // 000000006284: d6100016 0205215d
	s_wait_alu depctr_va_vcc(0)                                // 00000000628c: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v24, vcc_lo                    // 000000006290: 022a3115
	v_add_co_u32 v18, vcc_lo, v18, s0                          // 000000006294: d7006a12 02000112
	s_wait_alu depctr_va_vcc(0)                                // 00000000629c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s1, v19, vcc_lo             // 0000000062a0: d5207c13 01aa2601
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 0000000062a8: 3e1c1c81
	s_delay_alu instid0(valu_dep_3)                            // 0000000062ac: bf870003
	v_add_co_u32 v18, vcc_lo, v18, v2                          // 0000000062b0: d7006a12 02020512
	v_add3_u32 v22, v22, v93, 0x7fff                           // 0000000062b8: d6550016 03febb16 00007fff
	v_or_b32_e32 v23, 0x400000, v93                            // 0000000062c4: 382ebaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000062cc: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v19, v3, vcc_lo             // 0000000062d0: d5207c13 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v93, v93                           // 0000000062d8: 7c30bb5d
	s_wait_alu depctr_va_vcc(0)                                // 0000000062dc: bf88ff9d
	v_cndmask_b32_e32 v22, v22, v23, vcc_lo                    // 0000000062e0: 022c2f16
	v_add_co_u32 v23, vcc_lo, s18, v14                         // 0000000062e4: d7006a17 02021c12
	s_wait_alu depctr_va_vcc(0)                                // 0000000062ec: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s19, v15, vcc_lo            // 0000000062f0: d5207c18 01aa1e13
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000062f8: bf870122
	v_add_co_u32 v14, vcc_lo, v23, v2                          // 0000000062fc: d7006a0e 02020517
	s_wait_alu depctr_va_vcc(0)                                // 000000006304: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v24, v3, vcc_lo             // 000000006308: d5207c0f 01aa0718
	v_cmp_u_f32_e32 vcc_lo, v92, v92                           // 000000006310: 7c30b95c
	s_clause 0x2                                               // 000000006314: bf850002
	global_store_d16_hi_b16 v[16:17], v20, off                 // 000000006318: ee09407c 0a000000 00000010
	global_store_d16_hi_b16 v[18:19], v21, off                 // 000000006324: ee09407c 0a800000 00000012
	global_store_d16_hi_b16 v[14:15], v22, off                 // 000000006330: ee09407c 0b000000 0000000e
	v_bfe_u32 v20, v91, 16, 1                                  // 00000000633c: d6100014 0205215b
	s_wait_alu depctr_va_vcc(0)                                // 000000006344: bf88ff9d
	v_cndmask_b32_e32 v26, v25, v26, vcc_lo                    // 000000006348: 02343519
	v_add_co_u32 v22, vcc_lo, v23, s0                          // 00000000634c: d7006a16 02000117
	s_wait_alu depctr_va_vcc(0)                                // 000000006354: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s1, v24, vcc_lo             // 000000006358: d5207c17 01aa3001
	v_add3_u32 v24, v20, v91, 0x7fff                           // 000000006360: d6550018 03feb714 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 00000000636c: bf870003
	v_add_co_u32 v20, vcc_lo, v22, v2                          // 000000006370: d7006a14 02020516
	v_or_b32_e32 v25, 0x400000, v91                            // 000000006378: 3832b6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006380: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v23, v3, vcc_lo             // 000000006384: d5207c15 01aa0717
	v_cmp_u_f32_e32 vcc_lo, v91, v91                           // 00000000638c: 7c30b75b
	s_wait_alu depctr_va_vcc(0)                                // 000000006390: bf88ff9d
	v_cndmask_b32_e32 v27, v24, v25, vcc_lo                    // 000000006394: 02363318
	v_add_co_u32 v25, vcc_lo, v22, s0                          // 000000006398: d7006a19 02000116
	v_bfe_u32 v24, v90, 16, 1                                  // 0000000063a0: d6100018 0205215a
	s_wait_alu depctr_va_vcc(0)                                // 0000000063a8: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v23, vcc_lo             // 0000000063ac: d5207c1c 01aa2e01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000063b4: bf870193
	v_add_co_u32 v22, vcc_lo, v25, v2                          // 0000000063b8: d7006a16 02020519
	v_add3_u32 v24, v24, v90, 0x7fff                           // 0000000063c0: d6550018 03feb518 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000063cc: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000063d0: bf870003
	v_add_co_ci_u32_e64 v23, null, v28, v3, vcc_lo             // 0000000063d4: d5207c17 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v90, v90                           // 0000000063dc: 7c30b55a
	s_wait_alu depctr_va_vcc(0)                                // 0000000063e0: bf88ff9d
	v_cndmask_b32_e32 v29, v24, v29, vcc_lo                    // 0000000063e4: 023a3b18
	v_add_co_u32 v30, vcc_lo, v25, s0                          // 0000000063e8: d7006a1e 02000119
	s_wait_alu depctr_va_vcc(0)                                // 0000000063f0: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 0000000063f4: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000063fc: bf870122
	v_add_co_u32 v24, vcc_lo, v30, v2                          // 000000006400: d7006a18 0202051e
	s_wait_alu depctr_va_vcc(0)                                // 000000006408: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v28, v3, vcc_lo             // 00000000640c: d5207c19 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 000000006414: 7c30b359
	s_wait_alu depctr_va_vcc(0)                                // 000000006418: bf88ff9d
	v_cndmask_b32_e32 v32, v31, v32, vcc_lo                    // 00000000641c: 0240411f
	s_clause 0x2                                               // 000000006420: bf850002
	global_store_d16_hi_b16 v[20:21], v26, off                 // 000000006424: ee09407c 0d000000 00000014
	global_store_d16_hi_b16 v[22:23], v27, off                 // 000000006430: ee09407c 0d800000 00000016
	global_store_d16_hi_b16 v[24:25], v29, off                 // 00000000643c: ee09407c 0e800000 00000018
	v_bfe_u32 v26, v88, 16, 1                                  // 000000006448: d610001a 02052158
	v_add_co_u32 v29, vcc_lo, v30, s0                          // 000000006450: d7006a1d 0200011e
	s_wait_alu depctr_va_vcc(0)                                // 000000006458: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s1, v28, vcc_lo             // 00000000645c: d5207c1c 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000006464: bf870193
	v_add3_u32 v30, v26, v88, 0x7fff                           // 000000006468: d655001e 03feb11a 00007fff
	v_add_co_u32 v26, vcc_lo, v29, v2                          // 000000006474: d7006a1a 0202051d
	v_or_b32_e32 v31, 0x400000, v88                            // 00000000647c: 383eb0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006484: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v28, v3, vcc_lo             // 000000006488: d5207c1b 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v88, v88                           // 000000006490: 7c30b158
	s_wait_alu depctr_va_vcc(0)                                // 000000006494: bf88ff9d
	v_cndmask_b32_e32 v33, v30, v31, vcc_lo                    // 000000006498: 02423f1e
	v_add_co_u32 v31, vcc_lo, v29, s0                          // 00000000649c: d7006a1f 0200011d
	v_bfe_u32 v30, v87, 16, 1                                  // 0000000064a4: d610001e 02052157
	s_wait_alu depctr_va_vcc(0)                                // 0000000064ac: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v28, vcc_lo             // 0000000064b0: d5207c22 01aa3801
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000064b8: bf870193
	v_add_co_u32 v28, vcc_lo, v31, v2                          // 0000000064bc: d7006a1c 0202051f
	v_add3_u32 v30, v30, v87, 0x7fff                           // 0000000064c4: d655001e 03feaf1e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000064d0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000064d4: bf870003
	v_add_co_ci_u32_e64 v29, null, v34, v3, vcc_lo             // 0000000064d8: d5207c1d 01aa0722
	v_cmp_u_f32_e32 vcc_lo, v87, v87                           // 0000000064e0: 7c30af57
	s_wait_alu depctr_va_vcc(0)                                // 0000000064e4: bf88ff9d
	v_cndmask_b32_e32 v35, v30, v35, vcc_lo                    // 0000000064e8: 0246471e
	v_add_co_u32 v36, vcc_lo, v31, s0                          // 0000000064ec: d7006a24 0200011f
	s_wait_alu depctr_va_vcc(0)                                // 0000000064f4: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v34, vcc_lo             // 0000000064f8: d5207c22 01aa4401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000006500: bf870122
	v_add_co_u32 v30, vcc_lo, v36, v2                          // 000000006504: d7006a1e 02020524
	s_wait_alu depctr_va_vcc(0)                                // 00000000650c: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, v34, v3, vcc_lo             // 000000006510: d5207c1f 01aa0722
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 000000006518: 7c30ad56
	s_clause 0x2                                               // 00000000651c: bf850002
	global_store_d16_hi_b16 v[26:27], v32, off                 // 000000006520: ee09407c 10000000 0000001a
	global_store_d16_hi_b16 v[28:29], v33, off                 // 00000000652c: ee09407c 10800000 0000001c
	global_store_d16_hi_b16 v[30:31], v35, off                 // 000000006538: ee09407c 11800000 0000001e
	v_bfe_u32 v33, v85, 16, 1                                  // 000000006544: d6100021 02052155
	s_wait_alu depctr_va_vcc(0)                                // 00000000654c: bf88ff9d
	v_cndmask_b32_e32 v32, v37, v38, vcc_lo                    // 000000006550: 02404d25
	v_add_co_u32 v35, vcc_lo, v36, s0                          // 000000006554: d7006a23 02000124
	s_wait_alu depctr_va_vcc(0)                                // 00000000655c: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s1, v34, vcc_lo             // 000000006560: d5207c22 01aa4401
	v_add3_u32 v33, v33, v85, 0x7fff                           // 000000006568: d6550021 03feab21 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006574: bf870003
	v_add_co_u32 v2, vcc_lo, v35, v2                           // 000000006578: d7006a02 02020523
	v_or_b32_e32 v36, 0x400000, v85                            // 000000006580: 3848aaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006588: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v34, v3, vcc_lo              // 00000000658c: d5207c03 01aa0722
	v_bfe_u32 v34, v84, 16, 1                                  // 000000006594: d6100022 02052154
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 00000000659c: 7c30ab55
	v_bfe_u32 v35, v83, 16, 1                                  // 0000000065a0: d6100023 02052153
	global_store_d16_hi_b16 v[2:3], v32, off                   // 0000000065a8: ee09407c 10000000 00000002
	v_add3_u32 v32, v34, v84, 0x7fff                           // 0000000065b4: d6550020 03fea922 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000065c0: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v36, vcc_lo                    // 0000000065c4: 02424921
	v_or_b32_e32 v34, 0x400000, v84                            // 0000000065c8: 3844a8ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v84, v84                           // 0000000065d0: 7c30a954
	global_store_d16_hi_b16 v[0:1], v33, off offset:32         // 0000000065d4: ee09407c 10800000 00002000
	v_add3_u32 v0, v35, v83, 0x7fff                            // 0000000065e0: d6550000 03fea723 00007fff
	v_or_b32_e32 v1, 0x400000, v83                             // 0000000065ec: 3802a6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000065f4: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 0000000065f8: 02404520
	v_bfe_u32 v33, v82, 16, 1                                  // 0000000065fc: d6100021 02052152
	v_cmp_u_f32_e32 vcc_lo, v83, v83                           // 000000006604: 7c30a753
	global_store_d16_hi_b16 v[4:5], v32, off offset:32         // 000000006608: ee09407c 10000000 00002004
	v_add3_u32 v4, v33, v82, 0x7fff                            // 000000006614: d6550004 03fea521 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006620: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006624: 02000300
	v_bfe_u32 v1, v81, 16, 1                                   // 000000006628: d6100001 02052151
	v_or_b32_e32 v5, 0x400000, v82                             // 000000006630: 380aa4ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000006638: 7c30a552
	global_store_d16_hi_b16 v[6:7], v0, off offset:32          // 00000000663c: ee09407c 00000000 00002006
	v_add3_u32 v0, v1, v81, 0x7fff                             // 000000006648: d6550000 03fea301 00007fff
	v_or_b32_e32 v1, 0x400000, v81                             // 000000006654: 3802a2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000665c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000006660: 02080b04
	v_bfe_u32 v5, v80, 16, 1                                   // 000000006664: d6100005 02052150
	v_cmp_u_f32_e32 vcc_lo, v81, v81                           // 00000000666c: 7c30a351
	v_or_b32_e32 v7, 0x400000, v72                             // 000000006670: 380e90ff 00400000
	v_bfe_u32 v6, v71, 16, 1                                   // 000000006678: d6100006 02052147
	global_store_d16_hi_b16 v[8:9], v4, off offset:32          // 000000006680: ee09407c 02000000 00002008
	v_add3_u32 v4, v5, v80, 0x7fff                             // 00000000668c: d6550004 03fea105 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006698: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000669c: 02000300
	v_bfe_u32 v1, v79, 16, 1                                   // 0000000066a0: d6100001 0205214f
	v_or_b32_e32 v5, 0x400000, v80                             // 0000000066a8: 380aa0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v80, v80                           // 0000000066b0: 7c30a150
	v_add3_u32 v6, v6, v71, 0x7fff                             // 0000000066b4: d6550006 03fe8f06 00007fff
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 0000000066c0: ee09407c 00000000 0000200a
	v_add3_u32 v0, v1, v79, 0x7fff                             // 0000000066cc: d6550000 03fe9f01 00007fff
	v_or_b32_e32 v1, 0x400000, v79                             // 0000000066d8: 38029eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000066e0: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000066e4: 02080b04
	v_bfe_u32 v5, v77, 16, 1                                   // 0000000066e8: d6100005 0205214d
	v_cmp_u_f32_e32 vcc_lo, v79, v79                           // 0000000066f0: 7c309f4f
	v_or_b32_e32 v8, 0x400000, v71                             // 0000000066f4: 38108eff 00400000
	v_or_b32_e32 v9, 0x400000, v70                             // 0000000066fc: 38128cff 00400000
	global_store_d16_hi_b16 v[12:13], v4, off offset:32        // 000000006704: ee09407c 02000000 0000200c
	v_add3_u32 v4, v5, v77, 0x7fff                             // 000000006710: d6550004 03fe9b05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000671c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006720: 02000300
	v_bfe_u32 v1, v78, 16, 1                                   // 000000006724: d6100001 0205214e
	v_or_b32_e32 v5, 0x400000, v77                             // 00000000672c: 380a9aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000006734: 7c309b4d
	global_store_d16_hi_b16 v[16:17], v0, off offset:32        // 000000006738: ee09407c 00000000 00002010
	v_add3_u32 v0, v1, v78, 0x7fff                             // 000000006744: d6550000 03fe9d01 00007fff
	v_or_b32_e32 v1, 0x400000, v78                             // 000000006750: 38029cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006758: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 00000000675c: 02080b04
	v_bfe_u32 v5, v76, 16, 1                                   // 000000006760: d6100005 0205214c
	v_cmp_u_f32_e32 vcc_lo, v78, v78                           // 000000006768: 7c309d4e
	global_store_d16_hi_b16 v[18:19], v4, off offset:32        // 00000000676c: ee09407c 02000000 00002012
	v_add3_u32 v4, v5, v76, 0x7fff                             // 000000006778: d6550004 03fe9905 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006784: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006788: 02000300
	v_bfe_u32 v1, v75, 16, 1                                   // 00000000678c: d6100001 0205214b
	v_or_b32_e32 v5, 0x400000, v76                             // 000000006794: 380a98ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v76, v76                           // 00000000679c: 7c30994c
	global_store_d16_hi_b16 v[14:15], v0, off offset:32        // 0000000067a0: ee09407c 00000000 0000200e
	v_add3_u32 v0, v1, v75, 0x7fff                             // 0000000067ac: d6550000 03fe9701 00007fff
	v_or_b32_e32 v1, 0x400000, v75                             // 0000000067b8: 380296ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000067c0: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000067c4: 02080b04
	v_bfe_u32 v5, v74, 16, 1                                   // 0000000067c8: d6100005 0205214a
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 0000000067d0: 7c30974b
	global_store_d16_hi_b16 v[20:21], v4, off offset:32        // 0000000067d4: ee09407c 02000000 00002014
	v_add3_u32 v4, v5, v74, 0x7fff                             // 0000000067e0: d6550004 03fe9505 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000067ec: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000067f0: 02000300
	v_bfe_u32 v1, v73, 16, 1                                   // 0000000067f4: d6100001 02052149
	v_or_b32_e32 v5, 0x400000, v74                             // 0000000067fc: 380a94ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v74, v74                           // 000000006804: 7c30954a
	global_store_d16_hi_b16 v[22:23], v0, off offset:32        // 000000006808: ee09407c 00000000 00002016
	v_add3_u32 v0, v1, v73, 0x7fff                             // 000000006814: d6550000 03fe9301 00007fff
	v_or_b32_e32 v1, 0x400000, v73                             // 000000006820: 380292ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006828: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 00000000682c: 02080b04
	v_bfe_u32 v5, v72, 16, 1                                   // 000000006830: d6100005 02052148
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 000000006838: 7c309349
	s_delay_alu instid0(valu_dep_2)                            // 00000000683c: bf870002
	v_add3_u32 v5, v5, v72, 0x7fff                             // 000000006840: d6550005 03fe9105 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000684c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006850: 02000300
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 000000006854: 7c309148
	v_bfe_u32 v1, v70, 16, 1                                   // 000000006858: d6100001 02052146
	s_wait_alu depctr_va_vcc(0)                                // 000000006860: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000006864: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000006868: 7c308f47
	s_delay_alu instid0(valu_dep_3)                            // 00000000686c: bf870003
	v_add3_u32 v1, v1, v70, 0x7fff                             // 000000006870: d6550001 03fe8d01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000687c: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo                       // 000000006880: 020c1106
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000006884: 7c308d46
	s_wait_alu depctr_va_vcc(0)                                // 000000006888: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo                       // 00000000688c: 02021301
	s_clause 0x3                                               // 000000006890: bf850003
	global_store_d16_hi_b16 v[24:25], v4, off offset:32        // 000000006894: ee09407c 02000000 00002018
	global_store_d16_hi_b16 v[26:27], v0, off offset:32        // 0000000068a0: ee09407c 00000000 0000201a
	global_store_d16_hi_b16 v[28:29], v5, off offset:32        // 0000000068ac: ee09407c 02800000 0000201c
	global_store_d16_hi_b16 v[30:31], v6, off offset:32        // 0000000068b8: ee09407c 03000000 0000201e
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 0000000068c4: ee09407c 00800000 00002002
	s_nop 0                                                    // 0000000068d0: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000068d4: bfb60003
	s_endpgm                                                   // 0000000068d8: bfb00000
	s_code_end                                                 // 0000000068dc: bf9f0000
	s_code_end                                                 // 0000000068e0: bf9f0000
	s_code_end                                                 // 0000000068e4: bf9f0000
	s_code_end                                                 // 0000000068e8: bf9f0000
	s_code_end                                                 // 0000000068ec: bf9f0000
	s_code_end                                                 // 0000000068f0: bf9f0000
	s_code_end                                                 // 0000000068f4: bf9f0000
	s_code_end                                                 // 0000000068f8: bf9f0000
	s_code_end                                                 // 0000000068fc: bf9f0000
	s_code_end                                                 // 000000006900: bf9f0000
	s_code_end                                                 // 000000006904: bf9f0000
	s_code_end                                                 // 000000006908: bf9f0000
	s_code_end                                                 // 00000000690c: bf9f0000
	s_code_end                                                 // 000000006910: bf9f0000
	s_code_end                                                 // 000000006914: bf9f0000
	s_code_end                                                 // 000000006918: bf9f0000
	s_code_end                                                 // 00000000691c: bf9f0000
	s_code_end                                                 // 000000006920: bf9f0000
	s_code_end                                                 // 000000006924: bf9f0000
	s_code_end                                                 // 000000006928: bf9f0000
	s_code_end                                                 // 00000000692c: bf9f0000
	s_code_end                                                 // 000000006930: bf9f0000
	s_code_end                                                 // 000000006934: bf9f0000
	s_code_end                                                 // 000000006938: bf9f0000
	s_code_end                                                 // 00000000693c: bf9f0000
	s_code_end                                                 // 000000006940: bf9f0000
	s_code_end                                                 // 000000006944: bf9f0000
	s_code_end                                                 // 000000006948: bf9f0000
	s_code_end                                                 // 00000000694c: bf9f0000
	s_code_end                                                 // 000000006950: bf9f0000
	s_code_end                                                 // 000000006954: bf9f0000
	s_code_end                                                 // 000000006958: bf9f0000
	s_code_end                                                 // 00000000695c: bf9f0000
	s_code_end                                                 // 000000006960: bf9f0000
	s_code_end                                                 // 000000006964: bf9f0000
	s_code_end                                                 // 000000006968: bf9f0000
	s_code_end                                                 // 00000000696c: bf9f0000
	s_code_end                                                 // 000000006970: bf9f0000
	s_code_end                                                 // 000000006974: bf9f0000
	s_code_end                                                 // 000000006978: bf9f0000
	s_code_end                                                 // 00000000697c: bf9f0000
	s_code_end                                                 // 000000006980: bf9f0000
	s_code_end                                                 // 000000006984: bf9f0000
	s_code_end                                                 // 000000006988: bf9f0000
	s_code_end                                                 // 00000000698c: bf9f0000
	s_code_end                                                 // 000000006990: bf9f0000
	s_code_end                                                 // 000000006994: bf9f0000
	s_code_end                                                 // 000000006998: bf9f0000
	s_code_end                                                 // 00000000699c: bf9f0000
	s_code_end                                                 // 0000000069a0: bf9f0000
	s_code_end                                                 // 0000000069a4: bf9f0000
	s_code_end                                                 // 0000000069a8: bf9f0000
	s_code_end                                                 // 0000000069ac: bf9f0000
	s_code_end                                                 // 0000000069b0: bf9f0000
	s_code_end                                                 // 0000000069b4: bf9f0000
	s_code_end                                                 // 0000000069b8: bf9f0000
	s_code_end                                                 // 0000000069bc: bf9f0000
	s_code_end                                                 // 0000000069c0: bf9f0000
	s_code_end                                                 // 0000000069c4: bf9f0000
	s_code_end                                                 // 0000000069c8: bf9f0000
	s_code_end                                                 // 0000000069cc: bf9f0000
	s_code_end                                                 // 0000000069d0: bf9f0000
	s_code_end                                                 // 0000000069d4: bf9f0000
	s_code_end                                                 // 0000000069d8: bf9f0000
	s_code_end                                                 // 0000000069dc: bf9f0000
	s_code_end                                                 // 0000000069e0: bf9f0000
	s_code_end                                                 // 0000000069e4: bf9f0000
	s_code_end                                                 // 0000000069e8: bf9f0000
	s_code_end                                                 // 0000000069ec: bf9f0000
	s_code_end                                                 // 0000000069f0: bf9f0000
	s_code_end                                                 // 0000000069f4: bf9f0000
	s_code_end                                                 // 0000000069f8: bf9f0000
	s_code_end                                                 // 0000000069fc: bf9f0000
	s_code_end                                                 // 000000006a00: bf9f0000
	s_code_end                                                 // 000000006a04: bf9f0000
	s_code_end                                                 // 000000006a08: bf9f0000
	s_code_end                                                 // 000000006a0c: bf9f0000
	s_code_end                                                 // 000000006a10: bf9f0000
	s_code_end                                                 // 000000006a14: bf9f0000
	s_code_end                                                 // 000000006a18: bf9f0000
	s_code_end                                                 // 000000006a1c: bf9f0000
	s_code_end                                                 // 000000006a20: bf9f0000
	s_code_end                                                 // 000000006a24: bf9f0000
	s_code_end                                                 // 000000006a28: bf9f0000
	s_code_end                                                 // 000000006a2c: bf9f0000
	s_code_end                                                 // 000000006a30: bf9f0000
	s_code_end                                                 // 000000006a34: bf9f0000
	s_code_end                                                 // 000000006a38: bf9f0000
	s_code_end                                                 // 000000006a3c: bf9f0000
	s_code_end                                                 // 000000006a40: bf9f0000
	s_code_end                                                 // 000000006a44: bf9f0000
	s_code_end                                                 // 000000006a48: bf9f0000
	s_code_end                                                 // 000000006a4c: bf9f0000
	s_code_end                                                 // 000000006a50: bf9f0000
	s_code_end                                                 // 000000006a54: bf9f0000
	s_code_end                                                 // 000000006a58: bf9f0000
	s_code_end                                                 // 000000006a5c: bf9f0000
	s_code_end                                                 // 000000006a60: bf9f0000
	s_code_end                                                 // 000000006a64: bf9f0000
	s_code_end                                                 // 000000006a68: bf9f0000
	s_code_end                                                 // 000000006a6c: bf9f0000
	s_code_end                                                 // 000000006a70: bf9f0000
	s_code_end                                                 // 000000006a74: bf9f0000
	s_code_end                                                 // 000000006a78: bf9f0000
	s_code_end                                                 // 000000006a7c: bf9f0000
