
/tmp/tmpy7urfhwo.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_28d379a9237322d1>:
	s_clause 0x6                                               // 000000001b00: bf850006
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b04: f4004400 f80000c8
	s_load_b64 s[20:21], s[0:1], 0xa8                          // 000000001b0c: f4002500 f80000a8
	s_load_b64 s[28:29], s[0:1], 0xd8                          // 000000001b14: f4002700 f80000d8
	s_load_b64 s[34:35], s[0:1], 0x8                           // 000000001b1c: f4002880 f8000008
	s_load_b64 s[30:31], s[0:1], 0x30                          // 000000001b24: f4002780 f8000030
	s_load_b64 s[22:23], s[0:1], 0x58                          // 000000001b2c: f4002580 f8000058
	s_load_b64 s[24:25], s[0:1], 0x80                          // 000000001b34: f4002600 f8000080
	s_mov_b32 s2, ttmp9                                        // 000000001b3c: be820075
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b40: 86039f75
	s_mov_b32 s4, ttmp7                                        // 000000001b44: be840073
	s_ashr_i32 s5, ttmp7, 31                                   // 000000001b48: 86059f73
	s_lshl_b64 s[36:37], s[2:3], 5                             // 000000001b4c: 84a48502
	s_delay_alu instid0(salu_cycle_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000001b50: bf8700c9
	v_dual_mov_b32 v33, s37 :: v_dual_and_b32 v76, 15, v0      // 000000001b54: ca240025 214c008f
	s_lshl_b64 s[38:39], s[4:5], 5                             // 000000001b5c: 84a68504
	s_add_nc_u64 s[2:3], s[36:37], 32                          // 000000001b60: a982a024
	s_add_nc_u64 s[0:1], s[38:39], 32                          // 000000001b64: a980a026
	v_or_b32_e32 v32, s36, v76                                 // 000000001b68: 38409824
	v_mov_b32_e32 v35, s37                                     // 000000001b6c: 7e460225
	v_bfe_u32 v54, v0, 4, 1                                    // 000000001b70: d6100036 02050900
	s_or_b32 s33, s38, 16                                      // 000000001b78: 8c219026
	s_delay_alu instid0(valu_dep_3)                            // 000000001b7c: bf870003
	v_or_b32_e32 v34, 16, v32                                  // 000000001b80: 38444090
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	v_cmp_gt_i64_e64 s0, s[0:1], s[16:17]                      // 000000001b88: d4540000 02002000
	v_cmp_gt_i64_e64 s1, s[2:3], s[18:19]                      // 000000001b90: d4540001 02002402
	v_cmp_gt_i64_e64 s48, 0x80, s[28:29]                       // 000000001b98: d4540030 020038ff 00000080
	s_and_b32 s26, s28, 0xffffff80                             // 000000001ba4: 8b1aff1c ffffff80
	s_mov_b32 s27, s29                                         // 000000001bac: be9b001d
	s_or_b32 s0, s0, s1                                        // 000000001bb0: 8c000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bb4: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001bb8: 8b6a007e
	s_mov_b32 s0, -1                                           // 000000001bbc: be8000c1
	s_cbranch_vccz 3762                                        // 000000001bc0: bfa30eb2 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3b8c>
	s_and_b32 s0, s48, exec_lo                                 // 000000001bc4: 8b007e30
	s_cselect_b32 s0, 1, 0                                     // 000000001bc8: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bcc: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001bd0: bf078100
	s_cbranch_scc1 7                                           // 000000001bd4: bfa20007 <tessera_rocm_scaled_matmul_28d379a9237322d1+0xf4>
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v36, 3, v54    // 000000001bd8: ca220080 25246c83
	v_mov_b32_e32 v39, s39                                     // 000000001be0: 7e4e0227
	s_mov_b32 s0, 0                                            // 000000001be4: be800080
	s_delay_alu instid0(valu_dep_2)                            // 000000001be8: bf870002
	v_or_b32_e32 v38, s38, v36                                 // 000000001bec: 384c4826
	s_branch 1                                                 // 000000001bf0: bfa00001 <tessera_rocm_scaled_matmul_28d379a9237322d1+0xf8>
	s_mov_b32 s0, -1                                           // 000000001bf4: be8000c1
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v83, 0             // 000000001bf8: ca100080 50520080
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c00: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000001c04: 8b007e00
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v94, 0             // 000000001c08: ca100080 575e0080
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v100, 0            // 000000001c10: ca100080 5b640080
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v62, 0             // 000000001c18: ca100080 613e0080
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v64, 0            // 000000001c20: ca100080 69400080
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0             // 000000001c28: ca100080 41420080
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0             // 000000001c30: ca100080 43440080
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0             // 000000001c38: ca100080 45460080
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0             // 000000001c40: ca100080 47480080
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0             // 000000001c48: ca100080 494a0080
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v78, 0             // 000000001c50: ca100080 4b4e0080
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v56, 0             // 000000001c58: ca100080 4d380080
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v58, 0             // 000000001c60: ca100080 4f3a0080
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v60, 0             // 000000001c68: ca100080 373c0080
	v_mov_b32_e32 v57, 0                                       // 000000001c70: 7e720280
	v_mov_b32_e32 v59, 0                                       // 000000001c74: 7e760280
	v_mov_b32_e32 v61, 0                                       // 000000001c78: 7e7a0280
	v_mov_b32_e32 v63, 0                                       // 000000001c7c: 7e7e0280
	s_cselect_b32 s0, 1, 0                                     // 000000001c80: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 000000001c84: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 000000001c88: bf078100
	s_cbranch_scc1 2584                                        // 000000001c8c: bfa20a18 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x29f0>
	s_add_nc_u64 s[0:1], s[18:19], 0x7f                        // 000000001c90: a980ff12 0000007f
	v_dual_mov_b32 v1, s39 :: v_dual_lshlrev_b32 v36, 3, v54   // 000000001c98: ca220027 01246c83
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ca0: bf88ff9e
	s_lshr_b64 s[14:15], s[0:1], 7                             // 000000001ca4: 858e8700
	s_lshr_b64 s[2:3], s[36:37], 7                             // 000000001ca8: 85828724
	s_add_nc_u64 s[0:1], s[14:15], -1                          // 000000001cac: a980c10e
	v_or_b32_e32 v0, s38, v76                                  // 000000001cb0: 38009826
	v_or_b32_e32 v38, s38, v36                                 // 000000001cb4: 384c4826
	v_mov_b32_e32 v39, s39                                     // 000000001cb8: 7e4e0227
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cbc: bf88ff9e
	v_cmp_lt_u64_e64 s4, s[2:3], s[0:1]                        // 000000001cc0: d4590004 02000002
	v_or_b32_e32 v8, 1, v36                                    // 000000001cc8: 38104881
	v_or_b32_e32 v11, 2, v36                                   // 000000001ccc: 38164882
	s_lshr_b64 s[8:9], s[28:29], 7                             // 000000001cd0: 8588871c
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[38:39]                // 000000001cd4: 7ca84c10
	v_or_b32_e32 v2, s33, v76                                  // 000000001cd8: 38049821
	s_and_b32 s4, s4, exec_lo                                  // 000000001cdc: 8b047e04
	s_cselect_b32 s6, s2, s0                                   // 000000001ce0: 98060002
	v_or_b32_e32 v4, s38, v8                                   // 000000001ce4: 38081026
	v_cmp_gt_i64_e64 s0, s[16:17], v[0:1]                      // 000000001ce8: d4540000 02020010
	v_or_b32_e32 v0, s38, v11                                  // 000000001cf0: 38001626
	v_dual_mov_b32 v5, s39 :: v_dual_cndmask_b32 v6, 0, v38    // 000000001cf4: ca120027 05064c80
	v_cndmask_b32_e64 v7, 0, s39, vcc_lo                       // 000000001cfc: d5010007 01a84e80
	s_cselect_b32 s7, s3, s1                                   // 000000001d04: 98070103
	s_lshr_b32 s5, s29, 7                                      // 000000001d08: 8505871d
	s_delay_alu instid0(valu_dep_2)                            // 000000001d0c: bf870002
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000001d10: 7ca80810
	v_mul_lo_u32 v9, s5, v6                                    // 000000001d14: d72c0009 02020c05
	v_mul_lo_u32 v10, s8, v7                                   // 000000001d1c: d72c000a 02020e08
	v_mad_co_u64_u32 v[6:7], null, s8, v6, 0                   // 000000001d24: d6fe7c06 02020c08
	v_mov_b32_e32 v3, s39                                      // 000000001d2c: 7e060227
	v_or_b32_e32 v12, 3, v36                                   // 000000001d30: 38184883
	s_wait_alu depctr_va_vcc(0)                                // 000000001d34: bf88ff9d
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_cndmask_b32 v4, 0, v4// 000000001d38: ca520a80 05040880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001d40: 7ca80010
	v_cmp_gt_i64_e64 s1, s[16:17], v[2:3]                      // 000000001d44: d4540001 02020410
	v_or_b32_e32 v8, s33, v8                                   // 000000001d4c: 38101021
	v_add3_u32 v7, v7, v10, v9                                 // 000000001d50: d6550007 04261507
	v_mul_lo_u32 v9, s5, v4                                    // 000000001d58: d72c0009 02020805
	v_mul_lo_u32 v10, s8, v5                                   // 000000001d60: d72c000a 02020a08
	v_mad_co_u64_u32 v[2:3], null, s8, v4, 0                   // 000000001d68: d6fe7c02 02020808
	s_wait_alu depctr_va_vcc(0)                                // 000000001d70: bf88ff9d
	v_dual_cndmask_b32 v13, 0, v1 :: v_dual_cndmask_b32 v14, 0, v0// 000000001d74: ca520280 0d0e0080
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001d7c: 3e000c82
	v_mov_b32_e32 v5, s39                                      // 000000001d80: 7e0a0227
	v_or_b32_e32 v4, s38, v12                                  // 000000001d84: 38081826
	v_cmp_gt_i64_e64 s2, s[18:19], v[32:33]                    // 000000001d88: d4540002 02024012
	v_mad_co_u64_u32 v[6:7], null, s8, v14, 0                  // 000000001d90: d6fe7c06 02021c08
	v_add3_u32 v3, v3, v10, v9                                 // 000000001d98: d6550003 04261503
	v_mul_lo_u32 v10, s8, v13                                  // 000000001da0: d72c000a 02021a08
	v_or_b32_e32 v13, 4, v36                                   // 000000001da8: 381a4884
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000001dac: 7ca80810
	v_add_co_u32 v81, s4, s22, v0                              // 000000001db0: d7000451 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001db8: bf88f19f
	v_add_co_ci_u32_e64 v82, null, s23, v1, s4                 // 000000001dbc: d5207c52 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000001dc4: 3e000482
	v_mov_b32_e32 v3, s39                                      // 000000001dc8: 7e060227
	v_or_b32_e32 v2, s38, v13                                  // 000000001dcc: 38041a26
	v_mul_lo_u32 v9, s5, v14                                   // 000000001dd0: d72c0009 02021c05
	s_wait_alu depctr_va_vcc(0)                                // 000000001dd8: bf88ff9d
	v_dual_mov_b32 v37, 0 :: v_dual_cndmask_b32 v4, 0, v4      // 000000001ddc: ca120080 25040880
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo                        // 000000001de4: 020a0a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001de8: 7ca80410
	v_or_b32_e32 v14, 5, v36                                   // 000000001dec: 381c4885
	v_add_co_u32 v84, s4, s22, v0                              // 000000001df0: d7000454 02020016
	v_add3_u32 v7, v7, v10, v9                                 // 000000001df8: d6550007 04261507
	v_mul_lo_u32 v9, s5, v4                                    // 000000001e00: d72c0009 02020805
	v_mul_lo_u32 v10, s8, v5                                   // 000000001e08: d72c000a 02020a08
	v_mad_co_u64_u32 v[4:5], null, s8, v4, 0                   // 000000001e10: d6fe7c04 02020808
	s_wait_alu depctr_va_vcc(0)                                // 000000001e18: bf88ff9d
	v_cndmask_b32_e32 v16, 0, v2, vcc_lo                       // 000000001e1c: 02200480
	v_or_b32_e32 v2, s38, v14                                  // 000000001e20: 38041c26
	v_cndmask_b32_e32 v15, 0, v3, vcc_lo                       // 000000001e24: 021e0680
	s_wait_alu depctr_va_sdst(0)                               // 000000001e28: bf88f19f
	v_add_co_ci_u32_e64 v85, null, s23, v1, s4                 // 000000001e2c: d5207c55 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001e34: 3e000c82
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001e38: 7ca80410
	v_add3_u32 v5, v5, v10, v9                                 // 000000001e3c: d6550005 04261505
	v_mul_lo_u32 v10, s8, v15                                  // 000000001e44: d72c000a 02021e08
	v_or_b32_e32 v15, 6, v36                                   // 000000001e4c: 381e4886
	v_mul_lo_u32 v9, s5, v16                                   // 000000001e50: d72c0009 02022005
	v_mad_co_u64_u32 v[6:7], null, s8, v16, 0                  // 000000001e58: d6fe7c06 02022008
	s_wait_alu depctr_va_vcc(0)                                // 000000001e60: bf88ff9d
	v_dual_cndmask_b32 v17, 0, v2 :: v_dual_mov_b32 v100, v37  // 000000001e64: ca500480 11640125
	v_or_b32_e32 v2, s38, v15                                  // 000000001e6c: 38041e26
	v_cndmask_b32_e32 v16, 0, v3, vcc_lo                       // 000000001e70: 02200680
	v_add_co_u32 v86, s4, s22, v0                              // 000000001e74: d7000456 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001e7c: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s23, v1, s4                 // 000000001e80: d5207c58 00120217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001e88: 7ca80410
	v_add3_u32 v7, v7, v10, v9                                 // 000000001e8c: d6550007 04261507
	v_mul_lo_u32 v10, s8, v16                                  // 000000001e94: d72c000a 02022008
	v_or_b32_e32 v16, 7, v36                                   // 000000001e9c: 38204887
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001ea0: 3e000882
	v_mul_lo_u32 v9, s5, v17                                   // 000000001ea4: d72c0009 02022205
	s_wait_alu depctr_va_vcc(0)                                // 000000001eac: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v2, vcc_lo                       // 000000001eb0: 02240480
	v_mad_co_u64_u32 v[4:5], null, s8, v17, 0                  // 000000001eb4: d6fe7c04 02022208
	v_or_b32_e32 v2, s38, v16                                  // 000000001ebc: 38042026
	v_cndmask_b32_e32 v17, 0, v3, vcc_lo                       // 000000001ec0: 02220680
	v_add_co_u32 v89, s4, s22, v0                              // 000000001ec4: d7000459 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001ecc: bf88f19f
	v_add_co_ci_u32_e64 v90, null, s23, v1, s4                 // 000000001ed0: d5207c5a 00120217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001ed8: 7ca80410
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001edc: 3e000c82
	v_add3_u32 v5, v5, v10, v9                                 // 000000001ee0: d6550005 04261505
	v_mul_lo_u32 v9, s5, v18                                   // 000000001ee8: d72c0009 02022405
	v_mul_lo_u32 v10, s8, v17                                  // 000000001ef0: d72c000a 02022208
	v_mad_co_u64_u32 v[6:7], null, s8, v18, 0                  // 000000001ef8: d6fe7c06 02022408
	s_wait_alu depctr_va_vcc(0)                                // 000000001f00: bf88ff9d
	v_cndmask_b32_e32 v18, 0, v2, vcc_lo                       // 000000001f04: 02240480
	v_or_b32_e32 v2, s33, v36                                  // 000000001f08: 38044821
	v_add_co_u32 v92, s4, s22, v0                              // 000000001f0c: d700045c 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001f14: bf88f19f
	v_add_co_ci_u32_e64 v93, null, s23, v1, s4                 // 000000001f18: d5207c5d 00120217
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001f20: 3e000882
	v_dual_cndmask_b32 v17, 0, v3 :: v_dual_mov_b32 v94, v37   // 000000001f24: ca500680 115e0125
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[2:3]                  // 000000001f2c: 7ca80410
	v_add3_u32 v7, v7, v10, v9                                 // 000000001f30: d6550007 04261507
	v_dual_mov_b32 v9, s39 :: v_dual_mov_b32 v80, v37          // 000000001f38: ca100027 09500125
	v_mul_lo_u32 v10, s5, v18                                  // 000000001f40: d72c000a 02022405
	v_mul_lo_u32 v17, s8, v17                                  // 000000001f48: d72c0011 02022208
	v_mad_co_u64_u32 v[4:5], null, s8, v18, 0                  // 000000001f50: d6fe7c04 02022408
	v_add_co_u32 v95, s4, s22, v0                              // 000000001f58: d700045f 02020016
	s_wait_alu depctr_va_sdst(0)                               // 000000001f60: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s23, v1, s4                 // 000000001f64: d5207c60 00120217
	s_wait_alu depctr_va_vcc(0)                                // 000000001f6c: bf88ff9d
	v_cndmask_b32_e32 v2, 0, v2, vcc_lo                        // 000000001f70: 02040480
	v_cndmask_b32_e64 v3, 0, s39, vcc_lo                       // 000000001f74: d5010003 01a84e80
	v_lshlrev_b64_e32 v[0:1], 2, v[6:7]                        // 000000001f7c: 3e000c82
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[8:9]                  // 000000001f80: 7ca81010
	v_dual_mov_b32 v7, s39 :: v_dual_mov_b32 v70, v37          // 000000001f84: ca100027 07460125
	v_or_b32_e32 v6, s33, v11                                  // 000000001f8c: 380c1621
	v_add3_u32 v5, v5, v17, v10                                // 000000001f90: d6550005 042a2305
	v_mul_lo_u32 v9, s5, v2                                    // 000000001f98: d72c0009 02020405
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa0: bf88ff9d
	v_cndmask_b32_e64 v17, 0, s39, vcc_lo                      // 000000001fa4: d5010011 01a84e80
	v_cndmask_b32_e32 v8, 0, v8, vcc_lo                        // 000000001fac: 02101080
	v_add_co_u32 v98, vcc_lo, s22, v0                          // 000000001fb0: d7006a62 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000001fb8: bf88ff9d
	v_add_co_ci_u32_e64 v99, null, s23, v1, vcc_lo             // 000000001fbc: d5207c63 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000001fc4: 7ca80c10
	v_mul_lo_u32 v10, s8, v3                                   // 000000001fc8: d72c000a 02020608
	v_mad_co_u64_u32 v[2:3], null, s8, v2, 0                   // 000000001fd0: d6fe7c02 02020408
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000001fd8: 3e000882
	v_mad_co_u64_u32 v[4:5], null, s8, v8, 0                   // 000000001fdc: d6fe7c04 02021008
	s_wait_alu depctr_va_vcc(0)                                // 000000001fe4: bf88ff9d
	v_dual_mov_b32 v68, v37 :: v_dual_cndmask_b32 v11, 0, v6   // 000000001fe8: ca120125 440a0c80
	v_or_b32_e32 v6, s33, v12                                  // 000000001ff0: 380c1821
	v_mov_b32_e32 v66, v37                                     // 000000001ff4: 7e840325
	v_mov_b32_e32 v64, v37                                     // 000000001ff8: 7e800325
	v_add3_u32 v3, v3, v10, v9                                 // 000000001ffc: d6550003 04261503
	v_mul_lo_u32 v9, s5, v8                                    // 000000002004: d72c0009 02021005
	v_mul_lo_u32 v10, s8, v17                                  // 00000000200c: d72c000a 02022208
	v_cndmask_b32_e64 v8, 0, s39, vcc_lo                       // 000000002014: d5010008 01a84e80
	v_add_co_u32 v101, vcc_lo, s22, v0                         // 00000000201c: d7006a65 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002024: bf88ff9d
	v_add_co_ci_u32_e64 v102, null, s23, v1, vcc_lo            // 000000002028: d5207c66 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000002030: 7ca80c10
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000002034: 3e000482
	v_add3_u32 v5, v5, v10, v9                                 // 000000002038: d6550005 04261505
	v_mul_lo_u32 v9, s5, v11                                   // 000000002040: d72c0009 02021605
	v_mad_co_u64_u32 v[2:3], null, s8, v11, 0                  // 000000002048: d6fe7c02 02021608
	v_mul_lo_u32 v8, s8, v8                                    // 000000002050: d72c0008 02021008
	s_wait_alu depctr_va_vcc(0)                                // 000000002058: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v6, vcc_lo                       // 00000000205c: 02160c80
	v_or_b32_e32 v6, s33, v13                                  // 000000002060: 380c1a21
	v_cndmask_b32_e64 v10, 0, s39, vcc_lo                      // 000000002064: d501000a 01a84e80
	v_add_co_u32 v103, vcc_lo, s22, v0                         // 00000000206c: d7006a67 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002074: bf88ff9d
	v_add_co_ci_u32_e64 v104, null, s23, v1, vcc_lo            // 000000002078: d5207c68 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000002080: 7ca80c10
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000002084: 3e000882
	v_add3_u32 v3, v3, v8, v9                                  // 000000002088: d6550003 04261103
	v_mul_lo_u32 v8, s5, v11                                   // 000000002090: d72c0008 02021605
	v_mad_co_u64_u32 v[4:5], null, s8, v11, 0                  // 000000002098: d6fe7c04 02021608
	v_mul_lo_u32 v9, s8, v10                                   // 0000000020a0: d72c0009 02021408
	s_wait_alu depctr_va_vcc(0)                                // 0000000020a8: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v6, vcc_lo                       // 0000000020ac: 02160c80
	v_or_b32_e32 v6, s33, v14                                  // 0000000020b0: 380c1c21
	v_cndmask_b32_e64 v10, 0, s39, vcc_lo                      // 0000000020b4: d501000a 01a84e80
	v_add_co_u32 v106, vcc_lo, s22, v0                         // 0000000020bc: d7006a6a 02020016
	s_wait_alu depctr_va_vcc(0)                                // 0000000020c4: bf88ff9d
	v_add_co_ci_u32_e64 v107, null, s23, v1, vcc_lo            // 0000000020c8: d5207c6b 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 0000000020d0: 7ca80c10
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 0000000020d4: 3e000482
	v_add3_u32 v5, v5, v9, v8                                  // 0000000020d8: d6550005 04221305
	v_mul_lo_u32 v8, s5, v11                                   // 0000000020e0: d72c0008 02021605
	v_mad_co_u64_u32 v[2:3], null, s8, v11, 0                  // 0000000020e8: d6fe7c02 02021608
	v_mul_lo_u32 v9, s8, v10                                   // 0000000020f0: d72c0009 02021408
	s_wait_alu depctr_va_vcc(0)                                // 0000000020f8: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v6 :: v_dual_mov_b32 v62, v37   // 0000000020fc: ca500c80 0b3e0125
	v_or_b32_e32 v6, s33, v15                                  // 000000002104: 380c1e21
	v_cndmask_b32_e64 v10, 0, s39, vcc_lo                      // 000000002108: d501000a 01a84e80
	v_add_co_u32 v108, vcc_lo, s22, v0                         // 000000002110: d7006a6c 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002118: bf88ff9d
	v_add_co_ci_u32_e64 v109, null, s23, v1, vcc_lo            // 00000000211c: d5207c6d 01aa0217
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 000000002124: 7ca80c10
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000002128: 3e000882
	v_add3_u32 v3, v3, v9, v8                                  // 00000000212c: d6550003 04221303
	v_mul_lo_u32 v8, s5, v11                                   // 000000002134: d72c0008 02021605
	v_mul_lo_u32 v9, s8, v10                                   // 00000000213c: d72c0009 02021408
	v_mad_co_u64_u32 v[4:5], null, s8, v11, 0                  // 000000002144: d6fe7c04 02021608
	s_wait_alu depctr_va_vcc(0)                                // 00000000214c: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v6 :: v_dual_mov_b32 v78, v37   // 000000002150: ca500c80 0b4e0125
	v_or_b32_e32 v6, s33, v16                                  // 000000002158: 380c2021
	v_cndmask_b32_e64 v10, 0, s39, vcc_lo                      // 00000000215c: d501000a 01a84e80
	v_add_co_u32 v110, vcc_lo, s22, v0                         // 000000002164: d7006a6e 02020016
	s_wait_alu depctr_va_vcc(0)                                // 00000000216c: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, s23, v1, vcc_lo            // 000000002170: d5207c6f 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 000000002178: 3e000482
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[6:7]                  // 00000000217c: 7ca80c10
	v_add3_u32 v5, v5, v9, v8                                  // 000000002180: d6550005 04221305
	v_mul_lo_u32 v7, s5, v11                                   // 000000002188: d72c0007 02021605
	v_mul_lo_u32 v8, s8, v10                                   // 000000002190: d72c0008 02021408
	v_mad_co_u64_u32 v[2:3], null, s8, v11, 0                  // 000000002198: d6fe7c02 02021608
	v_cmp_gt_i64_e64 s3, s[18:19], v[34:35]                    // 0000000021a0: d4540003 02024412
	s_wait_alu depctr_va_vcc(0)                                // 0000000021a8: bf88ff9d
	v_cndmask_b32_e64 v9, 0, s39, vcc_lo                       // 0000000021ac: d5010009 01a84e80
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo                        // 0000000021b4: 020c0c80
	v_add_co_u32 v112, vcc_lo, s22, v0                         // 0000000021b8: d7006a70 02020016
	s_wait_alu depctr_va_vcc(0)                                // 0000000021c0: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, s23, v1, vcc_lo            // 0000000021c4: d5207c71 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 0000000021cc: 3e000882
	v_add3_u32 v3, v3, v8, v7                                  // 0000000021d0: d6550003 041e1103
	v_mul_lo_u32 v7, s5, v6                                    // 0000000021d8: d72c0007 02020c05
	v_mul_lo_u32 v8, s8, v9                                    // 0000000021e0: d72c0008 02021208
	v_mad_co_u64_u32 v[4:5], null, s8, v6, 0                   // 0000000021e8: d6fe7c04 02020c08
	v_dual_mov_b32 v105, v37 :: v_dual_mov_b32 v74, v37        // 0000000021f0: ca100125 694a0125
	v_add_co_u32 v114, vcc_lo, s22, v0                         // 0000000021f8: d7006a72 02020016
	s_wait_alu depctr_va_vcc(0)                                // 000000002200: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, s23, v1, vcc_lo            // 000000002204: d5207c73 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[2:3]                        // 00000000220c: 3e000482
	v_add_co_u32 v2, s4, s36, v76                              // 000000002210: d7000402 02029824
	s_wait_alu depctr_va_sdst(0)                               // 000000002218: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s37, 0, s4                   // 00000000221c: d5207c03 00110025
	v_add3_u32 v5, v5, v8, v7                                  // 000000002224: d6550005 041e1105
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000222c: bf8701a3
	v_add_co_u32 v6, vcc_lo, v2, 16                            // 000000002230: d7006a06 02012102
	s_wait_alu depctr_va_vcc(0)                                // 000000002238: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo                // 00000000223c: d5207c07 01aa0680
	v_add_co_u32 v116, vcc_lo, s22, v0                         // 000000002244: d7006a74 02020016
	s_wait_alu depctr_va_vcc(0)                                // 00000000224c: bf88ff9d
	v_add_co_ci_u32_e64 v117, null, s23, v1, vcc_lo            // 000000002250: d5207c75 01aa0217
	v_lshlrev_b64_e32 v[0:1], 2, v[4:5]                        // 000000002258: 3e000882
	v_add_co_u32 v5, s4, s38, v76                              // 00000000225c: d7000405 02029826
	v_mul_lo_u32 v4, s28, v7                                   // 000000002264: d72c0004 02020e1c
	s_wait_alu depctr_va_sdst(0)                               // 00000000226c: bf88f19f
	v_add_co_ci_u32_e64 v7, null, s39, 0, s4                   // 000000002270: d5207c07 00110027
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_3)// 000000002278: bf8701b3
	v_add_co_u32 v8, vcc_lo, v5, 16                            // 00000000227c: d7006a08 02012105
	v_mad_co_u64_u32 v[40:41], null, s28, v6, v[36:37]         // 000000002284: d6fe7c28 04920c1c
	s_wait_alu depctr_va_vcc(0)                                // 00000000228c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, 0, v7, vcc_lo                // 000000002290: d5207c09 01aa0e80
	v_mul_lo_u32 v6, s29, v6                                   // 000000002298: d72c0006 02020c1d
	v_mad_co_u64_u32 v[42:43], null, s28, v2, v[36:37]         // 0000000022a0: d6fe7c2a 0492041c
	v_mul_lo_u32 v3, s28, v3                                   // 0000000022a8: d72c0003 0202061c
	v_mul_lo_u32 v2, s29, v2                                   // 0000000022b0: d72c0002 0202041d
	v_mad_co_u64_u32 v[44:45], null, s28, v8, v[36:37]         // 0000000022b8: d6fe7c2c 0492101c
	v_mul_lo_u32 v9, s28, v9                                   // 0000000022c0: d72c0009 0202121c
	v_mul_lo_u32 v8, s29, v8                                   // 0000000022c8: d72c0008 0202101d
	v_mad_co_u64_u32 v[46:47], null, s28, v5, v[36:37]         // 0000000022d0: d6fe7c2e 04920a1c
	v_mul_lo_u32 v7, s28, v7                                   // 0000000022d8: d72c0007 02020e1c
	v_mul_lo_u32 v5, s29, v5                                   // 0000000022e0: d72c0005 02020a1d
	v_add_co_u32 v118, vcc_lo, s22, v0                         // 0000000022e8: d7006a76 02020016
	s_wait_alu depctr_va_vcc(0)                                // 0000000022f0: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s23, v1, vcc_lo            // 0000000022f4: d5207c77 01aa0217
	v_add3_u32 v41, v6, v41, v4                                // 0000000022fc: d6550029 04125306
	v_add3_u32 v43, v2, v43, v3                                // 000000002304: d655002b 040e5702
	v_add3_u32 v45, v8, v45, v9                                // 00000000230c: d655002d 04265b08
	v_add3_u32 v47, v5, v47, v7                                // 000000002314: d655002f 041e5f05
	v_dual_mov_b32 v97, v37 :: v_dual_mov_b32 v72, v37         // 00000000231c: ca100125 61480125
	v_dual_mov_b32 v91, v37 :: v_dual_mov_b32 v60, v37         // 000000002324: ca100125 5b3c0125
	v_dual_mov_b32 v87, v37 :: v_dual_mov_b32 v58, v37         // 00000000232c: ca100125 573a0125
	v_dual_mov_b32 v83, v37 :: v_dual_mov_b32 v56, v37         // 000000002334: ca100125 53380125
	v_mov_b32_e32 v69, v37                                     // 00000000233c: 7e8a0325
	v_mov_b32_e32 v67, v37                                     // 000000002340: 7e860325
	v_mov_b32_e32 v65, v37                                     // 000000002344: 7e820325
	v_mov_b32_e32 v79, v37                                     // 000000002348: 7e9e0325
	v_mov_b32_e32 v77, v37                                     // 00000000234c: 7e9a0325
	v_mov_b32_e32 v75, v37                                     // 000000002350: 7e960325
	v_mov_b32_e32 v73, v37                                     // 000000002354: 7e920325
	v_mov_b32_e32 v71, v37                                     // 000000002358: 7e8e0325
	v_mov_b32_e32 v63, v37                                     // 00000000235c: 7e7e0325
	v_mov_b32_e32 v61, v37                                     // 000000002360: 7e7a0325
	v_mov_b32_e32 v59, v37                                     // 000000002364: 7e760325
	v_mov_b32_e32 v57, v37                                     // 000000002368: 7e720325
	v_mov_b32_e32 v55, v37                                     // 00000000236c: 7e6e0325
	s_mov_b64 s[44:45], 0                                      // 000000002370: beac0180
	s_lshl_b64 s[40:41], s[6:7], 2                             // 000000002374: 84a88206
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, v37           // 000000002378: ca100080 18180125
	v_dual_mov_b32 v26, v37 :: v_dual_mov_b32 v27, v37         // 000000002380: ca100125 1a1a0125
	v_dual_mov_b32 v28, v37 :: v_dual_mov_b32 v29, v37         // 000000002388: ca100125 1c1c0125
	v_dual_mov_b32 v30, v37 :: v_dual_mov_b32 v31, v37         // 000000002390: ca100125 1e1e0125
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, v37           // 000000002398: ca100080 10100125
	v_dual_mov_b32 v18, v37 :: v_dual_mov_b32 v19, v37         // 0000000023a0: ca100125 12120125
	v_dual_mov_b32 v20, v37 :: v_dual_mov_b32 v21, v37         // 0000000023a8: ca100125 14140125
	v_dual_mov_b32 v22, v37 :: v_dual_mov_b32 v23, v37         // 0000000023b0: ca100125 16160125
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v9, v37             // 0000000023b8: ca100080 08080125
	v_dual_mov_b32 v10, v37 :: v_dual_mov_b32 v11, v37         // 0000000023c0: ca100125 0a0a0125
	v_dual_mov_b32 v12, v37 :: v_dual_mov_b32 v13, v37         // 0000000023c8: ca100125 0c0c0125
	v_dual_mov_b32 v14, v37 :: v_dual_mov_b32 v15, v37         // 0000000023d0: ca100125 0e0e0125
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, v37             // 0000000023d8: ca100080 00000125
	v_dual_mov_b32 v2, v37 :: v_dual_mov_b32 v3, v37           // 0000000023e0: ca100125 02020125
	v_dual_mov_b32 v4, v37 :: v_dual_mov_b32 v5, v37           // 0000000023e8: ca100125 04040125
	v_dual_mov_b32 v6, v37 :: v_dual_mov_b32 v7, v37           // 0000000023f0: ca100125 06060125
	s_add_nc_u64 s[42:43], s[44:45], 0x80                      // 0000000023f8: a9aaff2c 00000080
	s_mov_b64 s[46:47], s[44:45]                               // 000000002400: beae012c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002404: bf88ff9e
	v_add_co_u32 v48, s4, v36, s46                             // 000000002408: d7000430 02005d24
	s_wait_alu depctr_va_sdst(0)                               // 000000002410: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, s47, s4                  // 000000002414: d5207c31 00105e80
	v_add_co_u32 v122, vcc_lo, v46, s46                        // 00000000241c: d7006a7a 02005d2e
	s_wait_alu depctr_va_vcc(0)                                // 000000002424: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s47, v47, vcc_lo           // 000000002428: d5207c7b 01aa5e2f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 000000002430: bf8700c3
	v_cmp_gt_i64_e64 s10, s[28:29], v[48:49]                   // 000000002434: d454000a 0202601c
	s_and_b32 vcc_lo, s0, s10                                  // 00000000243c: 8b6a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000002440: bf88ff9e
	v_dual_cndmask_b32 v51, 0, v123 :: v_dual_cndmask_b32 v50, 0, v122// 000000002444: ca52f680 3332f480
	v_add_co_u32 v50, s4, s34, v50                             // 00000000244c: d7000432 02026422
	s_wait_alu depctr_va_sdst(0)                               // 000000002454: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002458: bf870002
	v_add_co_ci_u32_e64 v51, null, s35, v51, s4                // 00000000245c: d5207c33 00126623
	global_load_d16_u8 v50, v[50:51], off                      // 000000002464: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 000000002470: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 000000002474: d65d0032 01aa6480
	v_add_co_u32 v53, vcc_lo, v122, 1                          // 00000000247c: d7006a35 0201037a
	s_wait_alu depctr_va_vcc(0)                                // 000000002484: bf88ff9d
	v_add_co_ci_u32_e64 v120, null, 0, v123, vcc_lo            // 000000002488: d5207c78 01aaf680
	v_add_co_u32 v51, vcc_lo, v48, 1                           // 000000002490: d7006a33 02010330
	s_wait_alu depctr_va_vcc(0)                                // 000000002498: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, 0, v49, vcc_lo              // 00000000249c: d5207c34 01aa6280
	v_and_b16 v50.l, 0xff, v50.l                               // 0000000024a4: d7620032 020264ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000024b0: bf8700c2
	v_cmp_gt_i64_e64 s9, s[28:29], v[51:52]                    // 0000000024b4: d4540009 0202661c
	s_and_b32 vcc_lo, s0, s9                                   // 0000000024bc: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024c0: bf88ff9e
	v_dual_cndmask_b32 v52, 0, v120 :: v_dual_cndmask_b32 v51, 0, v53// 0000000024c4: ca52f080 34326a80
	v_add_co_u32 v51, s4, s34, v51                             // 0000000024cc: d7000433 02026622
	s_wait_alu depctr_va_sdst(0)                               // 0000000024d4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000024d8: bf870002
	v_add_co_ci_u32_e64 v52, null, s35, v52, s4                // 0000000024dc: d5207c34 00126823
	global_load_d16_hi_u8 v50, v[51:52], off                   // 0000000024e4: ee08407c 00000032 00000033
	s_wait_loadcnt 0x0                                         // 0000000024f0: bfc00000
	v_cndmask_b16 v52.l, 0, v50.h, vcc_lo                      // 0000000024f4: d65d1034 01aa6480
	v_add_co_u32 v51, vcc_lo, v122, 2                          // 0000000024fc: d7006a33 0201057a
	s_wait_alu depctr_va_vcc(0)                                // 000000002504: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v123, vcc_lo             // 000000002508: d5207c35 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 2                          // 000000002510: d7006a78 02010530
	s_wait_alu depctr_va_vcc(0)                                // 000000002518: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 00000000251c: d5207c79 01aa6280
	v_lshlrev_b16 v52.l, 8, v52.l                              // 000000002524: d7380034 02026888
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000252c: bf870112
	v_cmp_gt_i64_e64 s8, s[28:29], v[120:121]                  // 000000002530: d4540008 0202f01c
	v_or_b16 v50.l, v50.l, v52.l                               // 000000002538: d7630032 02026932
	s_and_b32 vcc_lo, s0, s8                                   // 000000002540: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 000000002544: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 000000002548: 02666680
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 00000000254c: 026a6a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002550: bf870122
	v_add_co_u32 v120, s4, s34, v51                            // 000000002554: d7000478 02026622
	s_wait_alu depctr_va_sdst(0)                               // 00000000255c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v53, s4               // 000000002560: d5207c79 00126a23
	global_load_d16_hi_u8 v50, v[120:121], off                 // 000000002568: ee08407c 00000032 00000078
	s_wait_loadcnt 0x0                                         // 000000002574: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, vcc_lo                      // 000000002578: d65d5032 01aa6480
	v_add_co_u32 v51, vcc_lo, v122, 3                          // 000000002580: d7006a33 0201077a
	s_wait_alu depctr_va_vcc(0)                                // 000000002588: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v123, vcc_lo             // 00000000258c: d5207c35 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 3                          // 000000002594: d7006a78 02010730
	s_wait_alu depctr_va_vcc(0)                                // 00000000259c: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 0000000025a0: d5207c79 01aa6280
	v_and_b16 v50.h, 0xff, v50.h op_sel:[0,1,1]                // 0000000025a8: d7625032 020264ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000025b4: bf870152
	v_cmp_gt_i64_e64 s7, s[28:29], v[120:121]                  // 0000000025b8: d4540007 0202f01c
	s_and_b32 vcc_lo, s0, s7                                   // 0000000025c0: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025c4: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 0000000025c8: 02666680
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 0000000025cc: 026a6a80
	v_add_co_u32 v120, s4, s34, v51                            // 0000000025d0: d7000478 02026622
	s_wait_alu depctr_va_sdst(0)                               // 0000000025d8: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000025dc: bf870002
	v_add_co_ci_u32_e64 v121, null, s35, v53, s4               // 0000000025e0: d5207c79 00126a23
	global_load_d16_u8 v51, v[120:121], off                    // 0000000025e8: ee07807c 00000033 00000078
	s_wait_loadcnt 0x0                                         // 0000000025f4: bfc00000
	v_cndmask_b16 v52.h, 0, v51.l, vcc_lo                      // 0000000025f8: d65d4034 01aa6680
	v_add_co_u32 v51, vcc_lo, v122, 4                          // 000000002600: d7006a33 0201097a
	s_wait_alu depctr_va_vcc(0)                                // 000000002608: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v123, vcc_lo             // 00000000260c: d5207c35 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 4                          // 000000002614: d7006a78 02010930
	s_wait_alu depctr_va_vcc(0)                                // 00000000261c: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 000000002620: d5207c79 01aa6280
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 000000002628: d7385034 02026888
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002630: bf870112
	v_cmp_gt_i64_e64 s6, s[28:29], v[120:121]                  // 000000002634: d4540006 0202f01c
	v_or_b16 v50.h, v50.h, v52.h op_sel:[1,1,1]                // 00000000263c: d7635832 02026932
	s_and_b32 vcc_lo, s0, s6                                   // 000000002644: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 000000002648: bf88ff9e
	v_cndmask_b32_e32 v51, 0, v51, vcc_lo                      // 00000000264c: 02666680
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 000000002650: 026a6a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002654: bf870122
	v_add_co_u32 v120, s4, s34, v51                            // 000000002658: d7000478 02026622
	s_wait_alu depctr_va_sdst(0)                               // 000000002660: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v53, s4               // 000000002664: d5207c79 00126a23
	global_load_d16_u8 v51, v[120:121], off                    // 00000000266c: ee07807c 00000033 00000078
	s_wait_loadcnt 0x0                                         // 000000002678: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, vcc_lo                      // 00000000267c: d65d0033 01aa6680
	v_add_co_u32 v53, vcc_lo, v122, 5                          // 000000002684: d7006a35 02010b7a
	s_wait_alu depctr_va_vcc(0)                                // 00000000268c: bf88ff9d
	v_add_co_ci_u32_e64 v124, null, 0, v123, vcc_lo            // 000000002690: d5207c7c 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 5                          // 000000002698: d7006a78 02010b30
	s_wait_alu depctr_va_vcc(0)                                // 0000000026a0: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 0000000026a4: d5207c79 01aa6280
	v_and_b16 v51.l, 0xff, v51.l                               // 0000000026ac: d7620033 020266ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000026b8: bf870152
	v_cmp_gt_i64_e64 s5, s[28:29], v[120:121]                  // 0000000026bc: d4540005 0202f01c
	s_and_b32 vcc_lo, s0, s5                                   // 0000000026c4: 8b6a0500
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026c8: bf88ff9e
	v_cndmask_b32_e32 v53, 0, v53, vcc_lo                      // 0000000026cc: 026a6a80
	v_cndmask_b32_e32 v121, 0, v124, vcc_lo                    // 0000000026d0: 02f2f880
	v_add_co_u32 v120, s4, s34, v53                            // 0000000026d4: d7000478 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 0000000026dc: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000026e0: bf870002
	v_add_co_ci_u32_e64 v121, null, s35, v121, s4              // 0000000026e4: d5207c79 0012f223
	global_load_d16_hi_u8 v51, v[120:121], off                 // 0000000026ec: ee08407c 00000033 00000078
	s_wait_loadcnt 0x0                                         // 0000000026f8: bfc00000
	v_cndmask_b16 v53.l, 0, v51.h, vcc_lo                      // 0000000026fc: d65d1035 01aa6680
	v_add_co_u32 v124, vcc_lo, v122, 6                         // 000000002704: d7006a7c 02010d7a
	s_wait_alu depctr_va_vcc(0)                                // 00000000270c: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, 0, v123, vcc_lo            // 000000002710: d5207c7d 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 6                          // 000000002718: d7006a78 02010d30
	s_wait_alu depctr_va_vcc(0)                                // 000000002720: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 000000002724: d5207c79 01aa6280
	v_lshlrev_b16 v53.l, 8, v53.l                              // 00000000272c: d7380035 02026a88
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000002734: bf870112
	v_cmp_gt_i64_e64 s4, s[28:29], v[120:121]                  // 000000002738: d4540004 0202f01c
	v_or_b16 v51.l, v51.l, v53.l                               // 000000002740: d7630033 02026b33
	s_and_b32 vcc_lo, s0, s4                                   // 000000002748: 8b6a0400
	s_wait_alu depctr_sa_sdst(0)                               // 00000000274c: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v124 :: v_dual_cndmask_b32 v121, 0, v125// 000000002750: ca52f880 7878fa80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002758: bf870121
	v_add_co_u32 v120, s11, s34, v120                          // 00000000275c: d7000b78 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 000000002764: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s11             // 000000002768: d5207c79 002ef223
	global_load_d16_hi_u8 v51, v[120:121], off                 // 000000002770: ee08407c 00000033 00000078
	s_wait_loadcnt 0x0                                         // 00000000277c: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, vcc_lo                      // 000000002780: d65d5033 01aa6680
	v_add_co_u32 v124, vcc_lo, v122, 7                         // 000000002788: d7006a7c 02010f7a
	s_wait_alu depctr_va_vcc(0)                                // 000000002790: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, 0, v123, vcc_lo            // 000000002794: d5207c7d 01aaf680
	v_add_co_u32 v120, vcc_lo, v48, 7                          // 00000000279c: d7006a78 02010f30
	s_wait_alu depctr_va_vcc(0)                                // 0000000027a4: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v49, vcc_lo             // 0000000027a8: d5207c79 01aa6280
	v_and_b16 v51.h, 0xff, v51.h op_sel:[0,1,1]                // 0000000027b0: d7625033 020266ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_2)// 0000000027bc: bf870152
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[120:121]              // 0000000027c0: 7ca8f01c
	s_and_b32 s11, s0, vcc_lo                                  // 0000000027c4: 8b0b6a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027c8: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v124, s11                       // 0000000027cc: d5010078 002ef880
	v_cndmask_b32_e64 v121, 0, v125, s11                       // 0000000027d4: d5010079 002efa80
	v_add_co_u32 v120, s12, s34, v120                          // 0000000027dc: d7000c78 0202f022
	s_wait_alu depctr_va_sdst(0)                               // 0000000027e4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000027e8: bf870002
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 0000000027ec: d5207c79 0032f223
	global_load_d16_hi_u8 v53, v[120:121], off                 // 0000000027f4: ee08407c 00000035 00000078
	s_wait_loadcnt 0x0                                         // 000000002800: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 000000002804: d65d5035 002e6a80
	v_add_co_u32 v124, s11, v44, s46                           // 00000000280c: d7000b7c 02005d2c
	s_wait_alu depctr_va_sdst(0)                               // 000000002814: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s47, v45, s11              // 000000002818: d5207c7d 002e5a2f
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_2)// 000000002820: bf870143
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 000000002824: d7385035 02026a88
	s_and_b32 s11, s1, s10                                     // 00000000282c: 8b0b0a01
	s_wait_alu depctr_sa_sdst(0)                               // 000000002830: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v124, s11                        // 000000002834: d5010034 002ef880
	v_or_b16 v51.h, v51.h, v53.h op_sel:[1,1,1]                // 00000000283c: d7635833 02026b33
	v_cndmask_b32_e64 v53, 0, v125, s11                        // 000000002844: d5010035 002efa80
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 00000000284c: bf870123
	v_add_co_u32 v52, s12, s34, v52                            // 000000002850: d7000c34 02026822
	s_wait_alu depctr_va_sdst(0)                               // 000000002858: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s35, v53, s12               // 00000000285c: d5207c35 00326a23
	global_load_d16_u8 v52, v[52:53], off                      // 000000002864: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 000000002870: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, s11                         // 000000002874: d65d0034 002e6880
	v_add_co_u32 v53, s11, v124, 1                             // 00000000287c: d7000b35 0201037c
	s_wait_alu depctr_va_sdst(0)                               // 000000002884: bf88f19f
	v_add_co_ci_u32_e64 v120, null, 0, v125, s11               // 000000002888: d5207c78 002efa80
	s_and_b32 s11, s1, s9                                      // 000000002890: 8b0b0901
	v_and_b16 v52.l, 0xff, v52.l                               // 000000002894: d7620034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028a0: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000028a4: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v120, s11                       // 0000000028ac: d5010079 002ef080
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000028b4: bf870122
	v_add_co_u32 v120, s12, s34, v53                           // 0000000028b8: d7000c78 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 0000000028c0: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s35, v121, s12             // 0000000028c4: d5207c79 0032f223
	global_load_d16_hi_u8 v52, v[120:121], off                 // 0000000028cc: ee08407c 00000034 00000078
	s_wait_loadcnt 0x0                                         // 0000000028d8: bfc00000
	v_cndmask_b16 v120.l, 0, v52.h, s11                        // 0000000028dc: d65d1078 002e6880
	v_add_co_u32 v53, s11, v124, 2                             // 0000000028e4: d7000b35 0201057c
	s_wait_alu depctr_va_sdst(0)                               // 0000000028ec: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v125, s11               // 0000000028f0: d5207c79 002efa80
	s_and_b32 s11, s1, s8                                      // 0000000028f8: 8b0b0801
	v_lshlrev_b16 v120.l, 8, v120.l                            // 0000000028fc: d7380078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002904: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000002908: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000002910: d5010079 002ef280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002918: bf870193
	v_or_b16 v52.l, v52.l, v120.l                              // 00000000291c: d7630034 0202f134
	v_add_co_u32 v126, s12, s34, v53                           // 000000002924: d7000c7e 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 00000000292c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002930: bf870003
	v_add_co_ci_u32_e64 v127, null, s35, v121, s12             // 000000002934: d5207c7f 0032f223
	global_load_d16_hi_u8 v52, v[126:127], off                 // 00000000293c: ee08407c 00000034 0000007e
	s_wait_loadcnt 0x0                                         // 000000002948: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s11                         // 00000000294c: d65d5034 002e6880
	v_add_co_u32 v53, s11, v124, 3                             // 000000002954: d7000b35 0201077c
	s_wait_alu depctr_va_sdst(0)                               // 00000000295c: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v125, s11               // 000000002960: d5207c79 002efa80
	s_and_b32 s11, s1, s7                                      // 000000002968: 8b0b0701
	v_and_b16 v52.h, 0xff, v52.h op_sel:[0,1,1]                // 00000000296c: d7625034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002978: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 00000000297c: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000002984: d5010079 002ef280
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000298c: bf870122
	v_add_co_u32 v126, s12, s34, v53                           // 000000002990: d7000c7e 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002998: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v121, s12             // 00000000299c: d5207c7f 0032f223
	global_load_d16_u8 v53, v[126:127], off                    // 0000000029a4: ee07807c 00000035 0000007e
	s_wait_loadcnt 0x0                                         // 0000000029b0: bfc00000
	v_cndmask_b16 v120.h, 0, v53.l, s11                        // 0000000029b4: d65d4078 002e6a80
	v_add_co_u32 v53, s11, v124, 4                             // 0000000029bc: d7000b35 0201097c
	s_wait_alu depctr_va_sdst(0)                               // 0000000029c4: bf88f19f
	v_add_co_ci_u32_e64 v121, null, 0, v125, s11               // 0000000029c8: d5207c79 002efa80
	s_and_b32 s11, s1, s6                                      // 0000000029d0: 8b0b0601
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 0000000029d4: d7385078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029dc: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000029e0: d5010035 002e6a80
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 0000000029e8: d5010079 002ef280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000029f0: bf870193
	v_or_b16 v52.h, v52.h, v120.h op_sel:[1,1,1]               // 0000000029f4: d7635834 0202f134
	v_add_co_u32 v126, s12, s34, v53                           // 0000000029fc: d7000c7e 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 000000002a04: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002a08: bf870003
	v_add_co_ci_u32_e64 v127, null, s35, v121, s12             // 000000002a0c: d5207c7f 0032f223
	global_load_d16_u8 v53, v[126:127], off                    // 000000002a14: ee07807c 00000035 0000007e
	s_wait_loadcnt 0x0                                         // 000000002a20: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s11                         // 000000002a24: d65d0035 002e6a80
	v_add_co_u32 v121, s11, v124, 5                            // 000000002a2c: d7000b79 02010b7c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a34: bf88f19f
	v_add_co_ci_u32_e64 v126, null, 0, v125, s11               // 000000002a38: d5207c7e 002efa80
	s_and_b32 s11, s1, s5                                      // 000000002a40: 8b0b0501
	v_and_b16 v53.l, 0xff, v53.l                               // 000000002a44: d7620035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a50: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000002a54: d5010079 002ef280
	v_cndmask_b32_e64 v127, 0, v126, s11                       // 000000002a5c: d501007f 002efc80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002a64: bf870122
	v_add_co_u32 v126, s12, s34, v121                          // 000000002a68: d7000c7e 0202f222
	s_wait_alu depctr_va_sdst(0)                               // 000000002a70: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 000000002a74: d5207c7f 0032fe23
	global_load_d16_hi_u8 v53, v[126:127], off                 // 000000002a7c: ee08407c 00000035 0000007e
	s_wait_loadcnt 0x0                                         // 000000002a88: bfc00000
	v_cndmask_b16 v121.l, 0, v53.h, s11                        // 000000002a8c: d65d1079 002e6a80
	v_add_co_u32 v126, s11, v124, 6                            // 000000002a94: d7000b7e 02010d7c
	s_wait_alu depctr_va_sdst(0)                               // 000000002a9c: bf88f19f
	v_add_co_ci_u32_e64 v127, null, 0, v125, s11               // 000000002aa0: d5207c7f 002efa80
	s_and_b32 s11, s1, s4                                      // 000000002aa8: 8b0b0401
	v_lshlrev_b16 v121.l, 8, v121.l                            // 000000002aac: d7380079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ab4: bf88ff9e
	v_cndmask_b32_e64 v126, 0, v126, s11                       // 000000002ab8: d501007e 002efc80
	v_cndmask_b32_e64 v127, 0, v127, s11                       // 000000002ac0: d501007f 002efe80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ac8: bf870193
	v_or_b16 v53.l, v53.l, v121.l                              // 000000002acc: d7630035 0202f335
	v_add_co_u32 v126, s12, s34, v126                          // 000000002ad4: d7000c7e 0202fc22
	s_wait_alu depctr_va_sdst(0)                               // 000000002adc: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000002ae0: bf870003
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 000000002ae4: d5207c7f 0032fe23
	global_load_d16_hi_u8 v53, v[126:127], off                 // 000000002aec: ee08407c 00000035 0000007e
	s_wait_loadcnt 0x0                                         // 000000002af8: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 000000002afc: d65d5035 002e6a80
	v_add_co_u32 v126, s11, v124, 7                            // 000000002b04: d7000b7e 02010f7c
	s_wait_alu depctr_va_sdst(0)                               // 000000002b0c: bf88f19f
	v_add_co_ci_u32_e64 v127, null, 0, v125, s11               // 000000002b10: d5207c7f 002efa80
	s_and_b32 s11, s1, vcc_lo                                  // 000000002b18: 8b0b6a01
	v_and_b16 v53.h, 0xff, v53.h op_sel:[0,1,1]                // 000000002b1c: d7625035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b28: bf88ff9e
	v_cndmask_b32_e64 v126, 0, v126, s11                       // 000000002b2c: d501007e 002efc80
	v_cndmask_b32_e64 v127, 0, v127, s11                       // 000000002b34: d501007f 002efe80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002b3c: bf870122
	v_add_co_u32 v126, s12, s34, v126                          // 000000002b40: d7000c7e 0202fc22
	s_wait_alu depctr_va_sdst(0)                               // 000000002b48: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s35, v127, s12             // 000000002b4c: d5207c7f 0032fe23
	global_load_d16_hi_u8 v121, v[126:127], off                // 000000002b54: ee08407c 00000079 0000007e
	s_wait_loadcnt 0x0                                         // 000000002b60: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 000000002b64: d65d5079 002ef280
	v_add_co_u32 v126, s11, v42, s46                           // 000000002b6c: d7000b7e 02005d2a
	s_wait_alu depctr_va_sdst(0)                               // 000000002b74: bf88f19f
	v_add_co_ci_u32_e64 v127, null, s47, v43, s11              // 000000002b78: d5207c7f 002e562f
	s_delay_alu instid0(valu_dep_3)                            // 000000002b80: bf870003
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000002b84: d7385079 0202f288
	s_and_b32 s11, s2, s10                                     // 000000002b8c: 8b0b0a02
	s_and_b32 s10, s3, s10                                     // 000000002b90: 8b0a0a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b94: bf88ff9e
	v_cndmask_b32_e64 v120, 0, v126, s11                       // 000000002b98: d5010078 002efc80
	v_or_b16 v53.h, v53.h, v121.h op_sel:[1,1,1]               // 000000002ba0: d7635835 0202f335
	v_cndmask_b32_e64 v121, 0, v127, s11                       // 000000002ba8: d5010079 002efe80
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 000000002bb0: bf870123
	v_add_co_u32 v120, s12, s30, v120                          // 000000002bb4: d7000c78 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000002bbc: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s12             // 000000002bc0: d5207c79 0032f21f
	global_load_d16_u8 v120, v[120:121], off                   // 000000002bc8: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 000000002bd4: bfc00000
	v_cndmask_b16 v120.l, 0, v120.l, s11                       // 000000002bd8: d65d0078 002ef080
	v_add_co_u32 v121, s11, v126, 1                            // 000000002be0: d7000b79 0201037e
	s_wait_alu depctr_va_sdst(0)                               // 000000002be8: bf88f19f
	v_add_co_ci_u32_e64 v128, null, 0, v127, s11               // 000000002bec: d5207c80 002efe80
	s_and_b32 s11, s2, s9                                      // 000000002bf4: 8b0b0902
	v_and_b16 v120.l, 0xff, v120.l                             // 000000002bf8: d7620078 0202f0ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c04: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000002c08: d5010079 002ef280
	v_cndmask_b32_e64 v129, 0, v128, s11                       // 000000002c10: d5010081 002f0080
	s_and_b32 s9, s3, s9                                       // 000000002c18: 8b090903
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002c1c: bf870122
	v_add_co_u32 v128, s12, s30, v121                          // 000000002c20: d7000c80 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c28: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s12             // 000000002c2c: d5207c81 0033021f
	global_load_d16_hi_u8 v120, v[128:129], off                // 000000002c34: ee08407c 00000078 00000080
	s_wait_loadcnt 0x0                                         // 000000002c40: bfc00000
	v_cndmask_b16 v120.h, 0, v120.h, s11                       // 000000002c44: d65d5078 002ef080
	v_add_co_u32 v121, s11, v126, 2                            // 000000002c4c: d7000b79 0201057e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c54: bf88f19f
	v_add_co_ci_u32_e64 v128, null, 0, v127, s11               // 000000002c58: d5207c80 002efe80
	s_and_b32 s11, s2, s8                                      // 000000002c60: 8b0b0802
	v_lshlrev_b16 v120.h, 8, v120.h op_sel:[0,1,1]             // 000000002c64: d7385078 0202f088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002c6c: bf88ff9e
	v_cndmask_b32_e64 v121, 0, v121, s11                       // 000000002c70: d5010079 002ef280
	v_cndmask_b32_e64 v129, 0, v128, s11                       // 000000002c78: d5010081 002f0080
	s_and_b32 s8, s3, s8                                       // 000000002c80: 8b080803
	v_or_b16 v132.l, v120.l, v120.h op_sel:[0,1,0]             // 000000002c84: d7631084 0202f178
	s_delay_alu instid0(valu_dep_3)                            // 000000002c8c: bf870003
	v_add_co_u32 v128, s12, s30, v121                          // 000000002c90: d7000c80 0202f21e
	s_wait_alu depctr_va_sdst(0)                               // 000000002c98: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s12             // 000000002c9c: d5207c81 0033021f
	global_load_d16_u8 v121, v[128:129], off                   // 000000002ca4: ee07807c 00000079 00000080
	s_wait_loadcnt 0x0                                         // 000000002cb0: bfc00000
	v_cndmask_b16 v121.l, 0, v121.l, s11                       // 000000002cb4: d65d0079 002ef280
	v_add_co_u32 v128, s11, v126, 3                            // 000000002cbc: d7000b80 0201077e
	s_wait_alu depctr_va_sdst(0)                               // 000000002cc4: bf88f19f
	v_add_co_ci_u32_e64 v129, null, 0, v127, s11               // 000000002cc8: d5207c81 002efe80
	s_and_b32 s11, s2, s7                                      // 000000002cd0: 8b0b0702
	v_and_b16 v121.l, 0xff, v121.l                             // 000000002cd4: d7620079 0202f2ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ce0: bf88ff9e
	v_cndmask_b32_e64 v128, 0, v128, s11                       // 000000002ce4: d5010080 002f0080
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 000000002cec: d5010081 002f0280
	s_and_b32 s7, s3, s7                                       // 000000002cf4: 8b070703
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cf8: bf870122
	v_add_co_u32 v128, s12, s30, v128                          // 000000002cfc: d7000c80 0203001e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d04: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s12             // 000000002d08: d5207c81 0033021f
	global_load_d16_hi_u8 v121, v[128:129], off                // 000000002d10: ee08407c 00000079 00000080
	s_wait_loadcnt 0x0                                         // 000000002d1c: bfc00000
	v_cndmask_b16 v121.h, 0, v121.h, s11                       // 000000002d20: d65d5079 002ef280
	v_add_co_u32 v128, s11, v126, 4                            // 000000002d28: d7000b80 0201097e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d30: bf88f19f
	v_add_co_ci_u32_e64 v129, null, 0, v127, s11               // 000000002d34: d5207c81 002efe80
	s_and_b32 s11, s2, s6                                      // 000000002d3c: 8b0b0602
	v_lshlrev_b16 v121.h, 8, v121.h op_sel:[0,1,1]             // 000000002d40: d7385079 0202f288
	s_wait_alu depctr_sa_sdst(0)                               // 000000002d48: bf88ff9e
	v_cndmask_b32_e64 v128, 0, v128, s11                       // 000000002d4c: d5010080 002f0080
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 000000002d54: d5010081 002f0280
	s_and_b32 s6, s3, s6                                       // 000000002d5c: 8b060603
	v_or_b16 v132.h, v121.l, v121.h op_sel:[0,1,1]             // 000000002d60: d7635084 0202f379
	s_delay_alu instid0(valu_dep_3)                            // 000000002d68: bf870003
	v_add_co_u32 v128, s12, s30, v128                          // 000000002d6c: d7000c80 0203001e
	s_wait_alu depctr_va_sdst(0)                               // 000000002d74: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s12             // 000000002d78: d5207c81 0033021f
	global_load_d16_u8 v128, v[128:129], off                   // 000000002d80: ee07807c 00000080 00000080
	s_wait_loadcnt 0x0                                         // 000000002d8c: bfc00000
	v_cndmask_b16 v128.l, 0, v128.l, s11                       // 000000002d90: d65d0080 002f0080
	v_add_co_u32 v129, s11, v126, 5                            // 000000002d98: d7000b81 02010b7e
	s_wait_alu depctr_va_sdst(0)                               // 000000002da0: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v127, s11               // 000000002da4: d5207c82 002efe80
	s_and_b32 s11, s2, s5                                      // 000000002dac: 8b0b0502
	v_and_b16 v128.l, 0xff, v128.l                             // 000000002db0: d7620080 020300ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002dbc: bf88ff9e
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 000000002dc0: d5010081 002f0280
	v_cndmask_b32_e64 v130, 0, v130, s11                       // 000000002dc8: d5010082 002f0480
	s_and_b32 s5, s3, s5                                       // 000000002dd0: 8b050503
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002dd4: bf870122
	v_add_co_u32 v129, s12, s30, v129                          // 000000002dd8: d7000c81 0203021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002de0: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s31, v130, s12             // 000000002de4: d5207c82 0033041f
	global_load_d16_hi_u8 v128, v[129:130], off                // 000000002dec: ee08407c 00000080 00000081
	s_wait_loadcnt 0x0                                         // 000000002df8: bfc00000
	v_cndmask_b16 v128.h, 0, v128.h, s11                       // 000000002dfc: d65d5080 002f0080
	v_add_co_u32 v129, s11, v126, 6                            // 000000002e04: d7000b81 02010d7e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e0c: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v127, s11               // 000000002e10: d5207c82 002efe80
	s_and_b32 s11, s2, s4                                      // 000000002e18: 8b0b0402
	v_lshlrev_b16 v128.h, 8, v128.h op_sel:[0,1,1]             // 000000002e1c: d7385080 02030088
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e24: bf88ff9e
	v_cndmask_b32_e64 v129, 0, v129, s11                       // 000000002e28: d5010081 002f0280
	v_cndmask_b32_e64 v130, 0, v130, s11                       // 000000002e30: d5010082 002f0480
	s_and_b32 s4, s3, s4                                       // 000000002e38: 8b040403
	v_or_b16 v133.l, v128.l, v128.h op_sel:[0,1,0]             // 000000002e3c: d7631085 02030180
	s_delay_alu instid0(valu_dep_3)                            // 000000002e44: bf870003
	v_add_co_u32 v129, s12, s30, v129                          // 000000002e48: d7000c81 0203021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e50: bf88f19f
	v_add_co_ci_u32_e64 v130, null, s31, v130, s12             // 000000002e54: d5207c82 0033041f
	global_load_d16_u8 v129, v[129:130], off                   // 000000002e5c: ee07807c 00000081 00000081
	s_wait_loadcnt 0x0                                         // 000000002e68: bfc00000
	v_cndmask_b16 v129.l, 0, v129.l, s11                       // 000000002e6c: d65d0081 002f0280
	v_add_co_u32 v130, s11, v126, 7                            // 000000002e74: d7000b82 02010f7e
	s_wait_alu depctr_va_sdst(0)                               // 000000002e7c: bf88f19f
	v_add_co_ci_u32_e64 v131, null, 0, v127, s11               // 000000002e80: d5207c83 002efe80
	s_and_b32 s11, s2, vcc_lo                                  // 000000002e88: 8b0b6a02
	v_and_b16 v129.l, 0xff, v129.l                             // 000000002e8c: d7620081 020302ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000002e98: bf88ff9e
	v_cndmask_b32_e64 v130, 0, v130, s11                       // 000000002e9c: d5010082 002f0480
	v_cndmask_b32_e64 v131, 0, v131, s11                       // 000000002ea4: d5010083 002f0680
	s_and_b32 vcc_lo, s3, vcc_lo                               // 000000002eac: 8b6a6a03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002eb0: bf870122
	v_add_co_u32 v130, s12, s30, v130                          // 000000002eb4: d7000c82 0203041e
	s_wait_alu depctr_va_sdst(0)                               // 000000002ebc: bf88f19f
	v_add_co_ci_u32_e64 v131, null, s31, v131, s12             // 000000002ec0: d5207c83 0033061f
	global_load_d16_hi_u8 v129, v[130:131], off                // 000000002ec8: ee08407c 00000081 00000082
	s_wait_loadcnt 0x0                                         // 000000002ed4: bfc00000
	v_cndmask_b16 v129.h, 0, v129.h, s11                       // 000000002ed8: d65d5081 002f0280
	v_add_co_u32 v120, s11, v40, s46                           // 000000002ee0: d7000b78 02005d28
	s_wait_alu depctr_va_sdst(0)                               // 000000002ee8: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s47, v41, s11              // 000000002eec: d5207c79 002e522f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ef4: bf870193
	v_lshlrev_b16 v129.h, 8, v129.h op_sel:[0,1,1]             // 000000002ef8: d7385081 02030288
	v_cndmask_b32_e64 v128, 0, v120, s10                       // 000000002f00: d5010080 002af080
	s_add_nc_u64 s[46:47], s[46:47], 32                        // 000000002f08: a9aea02e
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002f0c: bf8701a2
	v_or_b16 v133.h, v129.l, v129.h op_sel:[0,1,1]             // 000000002f10: d7635085 02030381
	v_cndmask_b32_e64 v129, 0, v121, s10                       // 000000002f18: d5010081 002af280
	v_add_co_u32 v128, s11, s30, v128                          // 000000002f20: d7000b80 0203001e
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002f28: bf8701a3
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[50:51], v[132:133], v[24:31]// 000000002f2c: cc464018 1c630932
	s_wait_alu depctr_va_sdst(0)                               // 000000002f34: bf88f19f
	v_add_co_ci_u32_e64 v129, null, s31, v129, s11             // 000000002f38: d5207c81 002f021f
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[52:53], v[132:133], v[8:15]// 000000002f40: cc464008 1c230934
	global_load_d16_u8 v128, v[128:129], off                   // 000000002f48: ee07807c 00000080 00000080
	s_wait_loadcnt 0x0                                         // 000000002f54: bfc00000
	v_cndmask_b16 v128.l, 0, v128.l, s10                       // 000000002f58: d65d0080 002b0080
	v_add_co_u32 v129, s10, v120, 1                            // 000000002f60: d7000a81 02010378
	s_wait_alu depctr_va_sdst(0)                               // 000000002f68: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v121, s10               // 000000002f6c: d5207c82 002af280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002f74: bf870193
	v_and_b16 v128.l, 0xff, v128.l                             // 000000002f78: d7620080 020300ff 000000ff
	v_cndmask_b32_e64 v129, 0, v129, s9                        // 000000002f84: d5010081 00270280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002f8c: bf870113
	v_cndmask_b32_e64 v130, 0, v130, s9                        // 000000002f90: d5010082 00270480
	v_add_co_u32 v129, s10, s30, v129                          // 000000002f98: d7000a81 0203021e
	s_wait_alu depctr_va_sdst(0)                               // 000000002fa0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000002fa4: bf870002
	v_add_co_ci_u32_e64 v130, null, s31, v130, s10             // 000000002fa8: d5207c82 002b041f
	global_load_d16_hi_u8 v128, v[129:130], off                // 000000002fb0: ee08407c 00000080 00000081
	s_wait_loadcnt 0x0                                         // 000000002fbc: bfc00000
	v_cndmask_b16 v128.h, 0, v128.h, s9                        // 000000002fc0: d65d5080 00270080
	v_add_co_u32 v129, s9, v120, 2                             // 000000002fc8: d7000981 02010578
	s_wait_alu depctr_va_sdst(0)                               // 000000002fd0: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v121, s9                // 000000002fd4: d5207c82 0026f280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002fdc: bf870193
	v_lshlrev_b16 v128.h, 8, v128.h op_sel:[0,1,1]             // 000000002fe0: d7385080 02030088
	v_cndmask_b32_e64 v129, 0, v129, s8                        // 000000002fe8: d5010081 00230280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000002ff0: bf870113
	v_cndmask_b32_e64 v130, 0, v130, s8                        // 000000002ff4: d5010082 00230480
	v_add_co_u32 v129, s9, s30, v129                           // 000000002ffc: d7000981 0203021e
	s_wait_alu depctr_va_sdst(0)                               // 000000003004: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003008: bf870002
	v_add_co_ci_u32_e64 v130, null, s31, v130, s9              // 00000000300c: d5207c82 0027041f
	global_load_d16_u8 v129, v[129:130], off                   // 000000003014: ee07807c 00000081 00000081
	s_wait_loadcnt 0x0                                         // 000000003020: bfc00000
	v_cndmask_b16 v129.l, 0, v129.l, s8                        // 000000003024: d65d0081 00230280
	v_add_co_u32 v130, s8, v120, 3                             // 00000000302c: d7000882 02010778
	s_wait_alu depctr_va_sdst(0)                               // 000000003034: bf88f19f
	v_add_co_ci_u32_e64 v131, null, 0, v121, s8                // 000000003038: d5207c83 0022f280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003040: bf870193
	v_and_b16 v129.l, 0xff, v129.l                             // 000000003044: d7620081 020302ff 000000ff
	v_cndmask_b32_e64 v130, 0, v130, s7                        // 000000003050: d5010082 001f0480
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003058: bf870113
	v_cndmask_b32_e64 v131, 0, v131, s7                        // 00000000305c: d5010083 001f0680
	v_add_co_u32 v130, s8, s30, v130                           // 000000003064: d7000882 0203041e
	s_wait_alu depctr_va_sdst(0)                               // 00000000306c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003070: bf870002
	v_add_co_ci_u32_e64 v131, null, s31, v131, s8              // 000000003074: d5207c83 0023061f
	global_load_d16_hi_u8 v129, v[130:131], off                // 00000000307c: ee08407c 00000081 00000082
	s_wait_loadcnt 0x0                                         // 000000003088: bfc00000
	v_cndmask_b16 v129.h, 0, v129.h, s7                        // 00000000308c: d65d5081 001f0280
	v_add_co_u32 v130, s7, v120, 4                             // 000000003094: d7000782 02010978
	s_wait_alu depctr_va_sdst(0)                               // 00000000309c: bf88f19f
	v_add_co_ci_u32_e64 v131, null, 0, v121, s7                // 0000000030a0: d5207c83 001ef280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000030a8: bf870193
	v_lshlrev_b16 v129.h, 8, v129.h op_sel:[0,1,1]             // 0000000030ac: d7385081 02030288
	v_cndmask_b32_e64 v130, 0, v130, s6                        // 0000000030b4: d5010082 001b0480
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 0000000030bc: bf870113
	v_cndmask_b32_e64 v131, 0, v131, s6                        // 0000000030c0: d5010083 001b0680
	v_add_co_u32 v130, s7, s30, v130                           // 0000000030c8: d7000782 0203041e
	s_wait_alu depctr_va_sdst(0)                               // 0000000030d0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000030d4: bf870002
	v_add_co_ci_u32_e64 v131, null, s31, v131, s7              // 0000000030d8: d5207c83 001f061f
	global_load_d16_u8 v130, v[130:131], off                   // 0000000030e0: ee07807c 00000082 00000082
	s_wait_loadcnt 0x0                                         // 0000000030ec: bfc00000
	v_cndmask_b16 v130.l, 0, v130.l, s6                        // 0000000030f0: d65d0082 001b0480
	v_add_co_u32 v131, s6, v120, 5                             // 0000000030f8: d7000683 02010b78
	s_wait_alu depctr_va_sdst(0)                               // 000000003100: bf88f19f
	v_add_co_ci_u32_e64 v134, null, 0, v121, s6                // 000000003104: d5207c86 001af280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000310c: bf870193
	v_and_b16 v130.l, 0xff, v130.l                             // 000000003110: d7620082 020304ff 000000ff
	v_cndmask_b32_e64 v131, 0, v131, s5                        // 00000000311c: d5010083 00170680
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003124: bf870113
	v_cndmask_b32_e64 v135, 0, v134, s5                        // 000000003128: d5010087 00170c80
	v_add_co_u32 v134, s6, s30, v131                           // 000000003130: d7000686 0203061e
	s_wait_alu depctr_va_sdst(0)                               // 000000003138: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000313c: bf870002
	v_add_co_ci_u32_e64 v135, null, s31, v135, s6              // 000000003140: d5207c87 001b0e1f
	global_load_d16_hi_u8 v130, v[134:135], off                // 000000003148: ee08407c 00000082 00000086
	s_wait_loadcnt 0x0                                         // 000000003154: bfc00000
	v_cndmask_b16 v130.h, 0, v130.h, s5                        // 000000003158: d65d5082 00170480
	v_add_co_u32 v131, s5, v120, 6                             // 000000003160: d7000583 02010d78
	s_wait_alu depctr_va_sdst(0)                               // 000000003168: bf88f19f
	v_add_co_ci_u32_e64 v134, null, 0, v121, s5                // 00000000316c: d5207c86 0016f280
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003174: bf870193
	v_lshlrev_b16 v130.h, 8, v130.h op_sel:[0,1,1]             // 000000003178: d7385082 02030488
	v_cndmask_b32_e64 v131, 0, v131, s4                        // 000000003180: d5010083 00130680
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_2)// 000000003188: bf870113
	v_cndmask_b32_e64 v135, 0, v134, s4                        // 00000000318c: d5010087 00130c80
	v_add_co_u32 v134, s5, s30, v131                           // 000000003194: d7000586 0203061e
	s_wait_alu depctr_va_sdst(0)                               // 00000000319c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000031a0: bf870002
	v_add_co_ci_u32_e64 v135, null, s31, v135, s5              // 0000000031a4: d5207c87 00170e1f
	global_load_d16_u8 v131, v[134:135], off                   // 0000000031ac: ee07807c 00000083 00000086
	s_wait_loadcnt 0x0                                         // 0000000031b8: bfc00000
	v_cndmask_b16 v131.l, 0, v131.l, s4                        // 0000000031bc: d65d0083 00130680
	v_add_co_u32 v134, s4, v120, 7                             // 0000000031c4: d7000486 02010f78
	s_wait_alu depctr_va_sdst(0)                               // 0000000031cc: bf88f19f
	v_add_co_ci_u32_e64 v135, null, 0, v121, s4                // 0000000031d0: d5207c87 0012f280
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031d8: bf870123
	v_and_b16 v131.l, 0xff, v131.l                             // 0000000031dc: d7620083 020306ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031e8: bf88ff9e
	v_dual_cndmask_b32 v134, 0, v134 :: v_dual_cndmask_b32 v135, 0, v135// 0000000031ec: ca530c80 86870e80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031f4: bf870121
	v_add_co_u32 v134, s4, s30, v134                           // 0000000031f8: d7000486 02030c1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003200: bf88f19f
	v_add_co_ci_u32_e64 v135, null, s31, v135, s4              // 000000003204: d5207c87 00130e1f
	global_load_d16_hi_u8 v131, v[134:135], off                // 00000000320c: ee08407c 00000083 00000086
	s_wait_loadcnt 0x0                                         // 000000003218: bfc00000
	v_cndmask_b16 v131.h, 0, v131.h, vcc_lo                    // 00000000321c: d65d5083 01ab0680
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003224: bf870091
	v_lshlrev_b16 v131.h, 8, v131.h op_sel:[0,1,1]             // 000000003228: d7385083 02030688
	v_or_b16 v131.h, v131.l, v131.h op_sel:[0,1,1]             // 000000003230: d7635083 02030783
	v_or_b16 v131.l, v130.l, v130.h op_sel:[0,1,0]             // 000000003238: d7631083 02030582
	v_or_b16 v130.h, v129.l, v129.h op_sel:[0,1,1]             // 000000003240: d7635082 02030381
	v_or_b16 v130.l, v128.l, v128.h op_sel:[0,1,0]             // 000000003248: d7631082 02030180
	s_delay_alu instid0(valu_dep_1)                            // 000000003250: bf870001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[50:51], v[130:131], v[16:23]// 000000003254: cc464010 1c430532
	v_add_co_u32 v50, vcc_lo, v48, 16                          // 00000000325c: d7006a32 02012130
	s_wait_alu depctr_va_vcc(0)                                // 000000003264: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, 0, v49, vcc_lo              // 000000003268: d5207c33 01aa6280
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[52:53], v[130:131], v[0:7]// 000000003270: cc464000 1c030534
	v_add_co_u32 v52, vcc_lo, v122, 16                         // 000000003278: d7006a34 0201217a
	s_delay_alu instid0(valu_dep_3)                            // 000000003280: bf870003
	v_cmp_gt_i64_e64 s8, s[28:29], v[50:51]                    // 000000003284: d4540008 0202641c
	s_wait_alu depctr_va_vcc(0)                                // 00000000328c: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v123, vcc_lo             // 000000003290: d5207c35 01aaf680
	s_and_b32 vcc_lo, s0, s8                                   // 000000003298: 8b6a0800
	s_wait_alu depctr_sa_sdst(0)                               // 00000000329c: bf88ff9e
	v_dual_cndmask_b32 v50, 0, v52 :: v_dual_cndmask_b32 v51, 0, v53// 0000000032a0: ca526880 32326a80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000032a8: bf870121
	v_add_co_u32 v50, s4, s34, v50                             // 0000000032ac: d7000432 02026422
	s_wait_alu depctr_va_sdst(0)                               // 0000000032b4: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s35, v51, s4                // 0000000032b8: d5207c33 00126623
	global_load_d16_u8 v50, v[50:51], off                      // 0000000032c0: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 0000000032cc: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, vcc_lo                      // 0000000032d0: d65d0032 01aa6480
	v_add_co_u32 v53, vcc_lo, v122, 17                         // 0000000032d8: d7006a35 0201237a
	s_wait_alu depctr_va_vcc(0)                                // 0000000032e0: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, 0, v123, vcc_lo            // 0000000032e4: d5207c80 01aaf680
	v_add_co_u32 v51, vcc_lo, v48, 17                          // 0000000032ec: d7006a33 02012330
	s_wait_alu depctr_va_vcc(0)                                // 0000000032f4: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, 0, v49, vcc_lo              // 0000000032f8: d5207c34 01aa6280
	v_and_b16 v50.l, 0xff, v50.l                               // 000000003300: d7620032 020264ff 000000ff
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000330c: bf8700c2
	v_cmp_gt_i64_e64 s9, s[28:29], v[51:52]                    // 000000003310: d4540009 0202661c
	s_and_b32 vcc_lo, s0, s9                                   // 000000003318: 8b6a0900
	s_wait_alu depctr_sa_sdst(0)                               // 00000000331c: bf88ff9e
	v_dual_cndmask_b32 v51, 0, v53 :: v_dual_cndmask_b32 v52, 0, v128// 000000003320: ca526a80 33350080
	v_add_co_u32 v51, s4, s34, v51                             // 000000003328: d7000433 02026622
	s_wait_alu depctr_va_sdst(0)                               // 000000003330: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003334: bf870002
	v_add_co_ci_u32_e64 v52, null, s35, v52, s4                // 000000003338: d5207c34 00126823
	global_load_d16_hi_u8 v50, v[51:52], off                   // 000000003340: ee08407c 00000032 00000033
	s_wait_loadcnt 0x0                                         // 00000000334c: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, vcc_lo                      // 000000003350: d65d5032 01aa6480
	v_add_co_u32 v53, vcc_lo, v122, 18                         // 000000003358: d7006a35 0201257a
	s_wait_alu depctr_va_vcc(0)                                // 000000003360: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, 0, v123, vcc_lo            // 000000003364: d5207c80 01aaf680
	v_add_co_u32 v51, vcc_lo, v48, 18                          // 00000000336c: d7006a33 02012530
	s_wait_alu depctr_va_vcc(0)                                // 000000003374: bf88ff9d
	v_add_co_ci_u32_e64 v52, null, 0, v49, vcc_lo              // 000000003378: d5207c34 01aa6280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000003380: bf8700c1
	v_cmp_gt_i64_e64 s10, s[28:29], v[51:52]                   // 000000003384: d454000a 0202661c
	s_and_b32 vcc_lo, s0, s10                                  // 00000000338c: 8b6a0a00
	s_wait_alu depctr_sa_sdst(0)                               // 000000003390: bf88ff9e
	v_dual_cndmask_b32 v51, 0, v53 :: v_dual_cndmask_b32 v52, 0, v128// 000000003394: ca526a80 33350080
	v_add_co_u32 v51, s4, s34, v51                             // 00000000339c: d7000433 02026622
	s_wait_alu depctr_va_sdst(0)                               // 0000000033a4: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 0000000033a8: bf870002
	v_add_co_ci_u32_e64 v52, null, s35, v52, s4                // 0000000033ac: d5207c34 00126823
	global_load_d16_u8 v51, v[51:52], off                      // 0000000033b4: ee07807c 00000033 00000033
	s_wait_loadcnt 0x0                                         // 0000000033c0: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, vcc_lo                      // 0000000033c4: d65d0033 01aa6680
	v_add_co_u32 v128, vcc_lo, v122, 19                        // 0000000033cc: d7006a80 0201277a
	s_wait_alu depctr_va_vcc(0)                                // 0000000033d4: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, 0, v123, vcc_lo            // 0000000033d8: d5207c81 01aaf680
	v_add_co_u32 v52, vcc_lo, v48, 19                          // 0000000033e0: d7006a34 02012730
	s_wait_alu depctr_va_vcc(0)                                // 0000000033e8: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v49, vcc_lo              // 0000000033ec: d5207c35 01aa6280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_1)// 0000000033f4: bf8700c1
	v_cmp_gt_i64_e64 s7, s[28:29], v[52:53]                    // 0000000033f8: d4540007 0202681c
	s_and_b32 vcc_lo, s0, s7                                   // 000000003400: 8b6a0700
	s_wait_alu depctr_sa_sdst(0)                               // 000000003404: bf88ff9e
	v_dual_cndmask_b32 v52, 0, v128 :: v_dual_cndmask_b32 v53, 0, v129// 000000003408: ca530080 34350280
	v_add_co_u32 v52, s4, s34, v52                             // 000000003410: d7000434 02026822
	s_wait_alu depctr_va_sdst(0)                               // 000000003418: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 00000000341c: bf870002
	v_add_co_ci_u32_e64 v53, null, s35, v53, s4                // 000000003420: d5207c35 00126a23
	global_load_d16_hi_u8 v51, v[52:53], off                   // 000000003428: ee08407c 00000033 00000034
	s_wait_loadcnt 0x0                                         // 000000003434: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, vcc_lo                      // 000000003438: d65d5033 01aa6680
	v_add_co_u32 v128, vcc_lo, v122, 20                        // 000000003440: d7006a80 0201297a
	s_wait_alu depctr_va_vcc(0)                                // 000000003448: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, 0, v123, vcc_lo            // 00000000344c: d5207c81 01aaf680
	v_add_co_u32 v52, vcc_lo, v48, 20                          // 000000003454: d7006a34 02012930
	s_wait_alu depctr_va_vcc(0)                                // 00000000345c: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, 0, v49, vcc_lo              // 000000003460: d5207c35 01aa6280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_1)// 000000003468: bf8700c1
	v_cmp_gt_i64_e64 s6, s[28:29], v[52:53]                    // 00000000346c: d4540006 0202681c
	s_and_b32 vcc_lo, s0, s6                                   // 000000003474: 8b6a0600
	s_wait_alu depctr_sa_sdst(0)                               // 000000003478: bf88ff9e
	v_dual_cndmask_b32 v52, 0, v128 :: v_dual_cndmask_b32 v53, 0, v129// 00000000347c: ca530080 34350280
	v_add_co_u32 v52, s4, s34, v52                             // 000000003484: d7000434 02026822
	s_wait_alu depctr_va_sdst(0)                               // 00000000348c: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003490: bf870002
	v_add_co_ci_u32_e64 v53, null, s35, v53, s4                // 000000003494: d5207c35 00126a23
	global_load_d16_u8 v52, v[52:53], off                      // 00000000349c: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 0000000034a8: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, vcc_lo                      // 0000000034ac: d65d0034 01aa6880
	v_add_co_u32 v53, vcc_lo, v122, 21                         // 0000000034b4: d7006a35 02012b7a
	s_wait_alu depctr_va_vcc(0)                                // 0000000034bc: bf88ff9d
	v_add_co_ci_u32_e64 v130, null, 0, v123, vcc_lo            // 0000000034c0: d5207c82 01aaf680
	v_add_co_u32 v128, vcc_lo, v48, 21                         // 0000000034c8: d7006a80 02012b30
	s_wait_alu depctr_va_vcc(0)                                // 0000000034d0: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, 0, v49, vcc_lo             // 0000000034d4: d5207c81 01aa6280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 0000000034dc: bf870151
	v_cmp_gt_i64_e32 vcc_lo, s[28:29], v[128:129]              // 0000000034e0: 7ca9001c
	s_and_b32 s4, s0, vcc_lo                                   // 0000000034e4: 8b046a00
	s_wait_alu depctr_sa_sdst(0)                               // 0000000034e8: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s4                          // 0000000034ec: d5010035 00126a80
	v_cndmask_b32_e64 v129, 0, v130, s4                        // 0000000034f4: d5010081 00130480
	v_add_co_u32 v128, s5, s34, v53                            // 0000000034fc: d7000580 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 000000003504: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003508: bf870002
	v_add_co_ci_u32_e64 v129, null, s35, v129, s5              // 00000000350c: d5207c81 00170223
	global_load_d16_hi_u8 v52, v[128:129], off                 // 000000003514: ee08407c 00000034 00000080
	s_wait_loadcnt 0x0                                         // 000000003520: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s4                          // 000000003524: d65d5034 00126880
	v_add_co_u32 v53, s4, v122, 22                             // 00000000352c: d7000435 02012d7a
	s_wait_alu depctr_va_sdst(0)                               // 000000003534: bf88f19f
	v_add_co_ci_u32_e64 v130, null, 0, v123, s4                // 000000003538: d5207c82 0012f680
	v_add_co_u32 v128, s4, v48, 22                             // 000000003540: d7000480 02012d30
	s_wait_alu depctr_va_sdst(0)                               // 000000003548: bf88f19f
	v_add_co_ci_u32_e64 v129, null, 0, v49, s4                 // 00000000354c: d5207c81 00126280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 000000003554: bf870151
	v_cmp_gt_i64_e64 s4, s[28:29], v[128:129]                  // 000000003558: d4540004 0203001c
	s_and_b32 s5, s0, s4                                       // 000000003560: 8b050400
	s_wait_alu depctr_sa_sdst(0)                               // 000000003564: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s5                          // 000000003568: d5010035 00166a80
	v_cndmask_b32_e64 v129, 0, v130, s5                        // 000000003570: d5010081 00170480
	v_add_co_u32 v128, s11, s34, v53                           // 000000003578: d7000b80 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 000000003580: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003584: bf870002
	v_add_co_ci_u32_e64 v129, null, s35, v129, s11             // 000000003588: d5207c81 002f0223
	global_load_d16_u8 v53, v[128:129], off                    // 000000003590: ee07807c 00000035 00000080
	s_wait_loadcnt 0x0                                         // 00000000359c: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s5                          // 0000000035a0: d65d0035 00166a80
	v_add_co_u32 v122, s5, v122, 23                            // 0000000035a8: d700057a 02012f7a
	s_wait_alu depctr_va_sdst(0)                               // 0000000035b0: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v123, s5                // 0000000035b4: d5207c7b 0016f680
	v_add_co_u32 v48, s5, v48, 23                              // 0000000035bc: d7000530 02012f30
	s_wait_alu depctr_va_sdst(0)                               // 0000000035c4: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v49, s5                  // 0000000035c8: d5207c31 00166280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_4) | instid1(valu_dep_2)// 0000000035d0: bf870151
	v_cmp_gt_i64_e64 s5, s[28:29], v[48:49]                    // 0000000035d4: d4540005 0202601c
	s_and_b32 s11, s0, s5                                      // 0000000035dc: 8b0b0500
	s_wait_alu depctr_sa_sdst(0)                               // 0000000035e0: bf88ff9e
	v_cndmask_b32_e64 v48, 0, v122, s11                        // 0000000035e4: d5010030 002ef480
	v_cndmask_b32_e64 v49, 0, v123, s11                        // 0000000035ec: d5010031 002ef680
	v_add_co_u32 v48, s12, s34, v48                            // 0000000035f4: d7000c30 02026022
	s_wait_alu depctr_va_sdst(0)                               // 0000000035fc: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_4) | instid1(valu_dep_1)// 000000003600: bf8700d2
	v_add_co_ci_u32_e64 v49, null, s35, v49, s12               // 000000003604: d5207c31 00326223
	global_load_d16_u8 v48, v[48:49], off                      // 00000000360c: ee07807c 00000030 00000030
	s_wait_loadcnt 0x0                                         // 000000003618: bfc00000
	v_and_b16 v48.h, 0xff, v53.l op_sel:[0,0,1]                // 00000000361c: d7624030 02026aff 000000ff
	v_cndmask_b16 v48.l, 0, v48.l, s11                         // 000000003628: d65d0030 002e6080
	v_lshlrev_b16 v48.l, 8, v48.l                              // 000000003630: d7380030 02026088
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003638: bf8700b1
	v_or_b16 v49.h, v48.h, v48.l op_sel:[1,0,1]                // 00000000363c: d7634831 02026130
	v_lshlrev_b16 v48.l, 8, v52.h op_sel:[0,1,0]               // 000000003644: d7381030 02026888
	v_and_b16 v48.h, 0xff, v52.l op_sel:[0,0,1]                // 00000000364c: d7624030 020268ff 000000ff
	v_or_b16 v49.l, v48.h, v48.l op_sel:[1,0,0]                // 000000003658: d7630831 02026130
	v_lshlrev_b16 v48.l, 8, v51.h op_sel:[0,1,0]               // 000000003660: d7381030 02026688
	v_and_b16 v48.h, 0xff, v51.l op_sel:[0,0,1]                // 000000003668: d7624030 020266ff 000000ff
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000003674: bf8700a1
	v_or_b16 v48.h, v48.h, v48.l op_sel:[1,0,1]                // 000000003678: d7634830 02026130
	v_lshlrev_b16 v48.l, 8, v50.h op_sel:[0,1,0]               // 000000003680: d7381030 02026488
	v_or_b16 v48.l, v50.l, v48.l                               // 000000003688: d7630030 02026132
	v_add_co_u32 v50, s11, v124, 16                            // 000000003690: d7000b32 0201217c
	s_wait_alu depctr_va_sdst(0)                               // 000000003698: bf88f19f
	v_add_co_ci_u32_e64 v51, null, 0, v125, s11                // 00000000369c: d5207c33 002efa80
	s_and_b32 s11, s1, s8                                      // 0000000036a4: 8b0b0801
	s_wait_alu depctr_sa_sdst(0)                               // 0000000036a8: bf88ff9e
	v_cndmask_b32_e64 v50, 0, v50, s11                         // 0000000036ac: d5010032 002e6480
	v_cndmask_b32_e64 v51, 0, v51, s11                         // 0000000036b4: d5010033 002e6680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000036bc: bf870122
	v_add_co_u32 v50, s12, s34, v50                            // 0000000036c0: d7000c32 02026422
	s_wait_alu depctr_va_sdst(0)                               // 0000000036c8: bf88f19f
	v_add_co_ci_u32_e64 v51, null, s35, v51, s12               // 0000000036cc: d5207c33 00326623
	global_load_d16_u8 v50, v[50:51], off                      // 0000000036d4: ee07807c 00000032 00000032
	s_wait_loadcnt 0x0                                         // 0000000036e0: bfc00000
	v_cndmask_b16 v50.l, 0, v50.l, s11                         // 0000000036e4: d65d0032 002e6480
	v_add_co_u32 v51, s11, v124, 17                            // 0000000036ec: d7000b33 0201237c
	s_wait_alu depctr_va_sdst(0)                               // 0000000036f4: bf88f19f
	v_add_co_ci_u32_e64 v52, null, 0, v125, s11                // 0000000036f8: d5207c34 002efa80
	s_and_b32 s11, s1, s9                                      // 000000003700: 8b0b0901
	v_and_b16 v50.l, 0xff, v50.l                               // 000000003704: d7620032 020264ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003710: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s11                         // 000000003714: d5010033 002e6680
	v_cndmask_b32_e64 v52, 0, v52, s11                         // 00000000371c: d5010034 002e6880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003724: bf870122
	v_add_co_u32 v51, s12, s34, v51                            // 000000003728: d7000c33 02026622
	s_wait_alu depctr_va_sdst(0)                               // 000000003730: bf88f19f
	v_add_co_ci_u32_e64 v52, null, s35, v52, s12               // 000000003734: d5207c34 00326823
	global_load_d16_hi_u8 v50, v[51:52], off                   // 00000000373c: ee08407c 00000032 00000033
	s_wait_loadcnt 0x0                                         // 000000003748: bfc00000
	v_cndmask_b16 v52.l, 0, v50.h, s11                         // 00000000374c: d65d1034 002e6480
	v_add_co_u32 v51, s11, v124, 18                            // 000000003754: d7000b33 0201257c
	s_wait_alu depctr_va_sdst(0)                               // 00000000375c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v125, s11                // 000000003760: d5207c35 002efa80
	s_and_b32 s11, s1, s10                                     // 000000003768: 8b0b0a01
	v_lshlrev_b16 v52.l, 8, v52.l                              // 00000000376c: d7380034 02026888
	s_wait_alu depctr_sa_sdst(0)                               // 000000003774: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s11                         // 000000003778: d5010033 002e6680
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003780: d5010035 002e6a80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003788: bf870193
	v_or_b16 v50.l, v50.l, v52.l                               // 00000000378c: d7630032 02026932
	v_add_co_u32 v122, s12, s34, v51                           // 000000003794: d7000c7a 02026622
	s_wait_alu depctr_va_sdst(0)                               // 00000000379c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 0000000037a0: bf870003
	v_add_co_ci_u32_e64 v123, null, s35, v53, s12              // 0000000037a4: d5207c7b 00326a23
	global_load_d16_hi_u8 v50, v[122:123], off                 // 0000000037ac: ee08407c 00000032 0000007a
	s_wait_loadcnt 0x0                                         // 0000000037b8: bfc00000
	v_cndmask_b16 v50.h, 0, v50.h, s11                         // 0000000037bc: d65d5032 002e6480
	v_add_co_u32 v51, s11, v124, 19                            // 0000000037c4: d7000b33 0201277c
	s_wait_alu depctr_va_sdst(0)                               // 0000000037cc: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v125, s11                // 0000000037d0: d5207c35 002efa80
	s_and_b32 s11, s1, s7                                      // 0000000037d8: 8b0b0701
	v_and_b16 v50.h, 0xff, v50.h op_sel:[0,1,1]                // 0000000037dc: d7625032 020264ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037e8: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s11                         // 0000000037ec: d5010033 002e6680
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000037f4: d5010035 002e6a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000037fc: bf870122
	v_add_co_u32 v122, s12, s34, v51                           // 000000003800: d7000c7a 02026622
	s_wait_alu depctr_va_sdst(0)                               // 000000003808: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s35, v53, s12              // 00000000380c: d5207c7b 00326a23
	global_load_d16_u8 v51, v[122:123], off                    // 000000003814: ee07807c 00000033 0000007a
	s_wait_loadcnt 0x0                                         // 000000003820: bfc00000
	v_cndmask_b16 v52.h, 0, v51.l, s11                         // 000000003824: d65d4034 002e6680
	v_add_co_u32 v51, s11, v124, 20                            // 00000000382c: d7000b33 0201297c
	s_wait_alu depctr_va_sdst(0)                               // 000000003834: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v125, s11                // 000000003838: d5207c35 002efa80
	s_and_b32 s11, s1, s6                                      // 000000003840: 8b0b0601
	v_lshlrev_b16 v52.h, 8, v52.h op_sel:[0,1,1]               // 000000003844: d7385034 02026888
	s_wait_alu depctr_sa_sdst(0)                               // 00000000384c: bf88ff9e
	v_cndmask_b32_e64 v51, 0, v51, s11                         // 000000003850: d5010033 002e6680
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003858: d5010035 002e6a80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003860: bf870193
	v_or_b16 v50.h, v50.h, v52.h op_sel:[1,1,1]                // 000000003864: d7635832 02026932
	v_add_co_u32 v122, s12, s34, v51                           // 00000000386c: d7000c7a 02026622
	s_wait_alu depctr_va_sdst(0)                               // 000000003874: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003878: bf870003
	v_add_co_ci_u32_e64 v123, null, s35, v53, s12              // 00000000387c: d5207c7b 00326a23
	global_load_d16_u8 v51, v[122:123], off                    // 000000003884: ee07807c 00000033 0000007a
	s_wait_loadcnt 0x0                                         // 000000003890: bfc00000
	v_cndmask_b16 v51.l, 0, v51.l, s11                         // 000000003894: d65d0033 002e6680
	v_add_co_u32 v53, s11, v124, 21                            // 00000000389c: d7000b35 02012b7c
	s_wait_alu depctr_va_sdst(0)                               // 0000000038a4: bf88f19f
	v_add_co_ci_u32_e64 v122, null, 0, v125, s11               // 0000000038a8: d5207c7a 002efa80
	s_and_b32 s11, s1, vcc_lo                                  // 0000000038b0: 8b0b6a01
	v_and_b16 v51.l, 0xff, v51.l                               // 0000000038b4: d7620033 020266ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038c0: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 0000000038c4: d5010035 002e6a80
	v_cndmask_b32_e64 v123, 0, v122, s11                       // 0000000038cc: d501007b 002ef480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000038d4: bf870122
	v_add_co_u32 v122, s12, s34, v53                           // 0000000038d8: d7000c7a 02026a22
	s_wait_alu depctr_va_sdst(0)                               // 0000000038e0: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s35, v123, s12             // 0000000038e4: d5207c7b 0032f623
	global_load_d16_hi_u8 v51, v[122:123], off                 // 0000000038ec: ee08407c 00000033 0000007a
	s_wait_loadcnt 0x0                                         // 0000000038f8: bfc00000
	v_cndmask_b16 v53.l, 0, v51.h, s11                         // 0000000038fc: d65d1035 002e6680
	v_add_co_u32 v122, s11, v124, 22                           // 000000003904: d7000b7a 02012d7c
	s_wait_alu depctr_va_sdst(0)                               // 00000000390c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v125, s11               // 000000003910: d5207c7b 002efa80
	s_and_b32 s11, s1, s4                                      // 000000003918: 8b0b0401
	v_lshlrev_b16 v53.l, 8, v53.l                              // 00000000391c: d7380035 02026a88
	s_wait_alu depctr_sa_sdst(0)                               // 000000003924: bf88ff9e
	v_cndmask_b32_e64 v122, 0, v122, s11                       // 000000003928: d501007a 002ef480
	v_cndmask_b32_e64 v123, 0, v123, s11                       // 000000003930: d501007b 002ef680
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003938: bf870193
	v_or_b16 v51.l, v51.l, v53.l                               // 00000000393c: d7630033 02026b33
	v_add_co_u32 v122, s12, s34, v122                          // 000000003944: d7000c7a 0202f422
	s_wait_alu depctr_va_sdst(0)                               // 00000000394c: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003950: bf870003
	v_add_co_ci_u32_e64 v123, null, s35, v123, s12             // 000000003954: d5207c7b 0032f623
	global_load_d16_hi_u8 v51, v[122:123], off                 // 00000000395c: ee08407c 00000033 0000007a
	s_wait_loadcnt 0x0                                         // 000000003968: bfc00000
	v_cndmask_b16 v51.h, 0, v51.h, s11                         // 00000000396c: d65d5033 002e6680
	v_add_co_u32 v122, s11, v124, 23                           // 000000003974: d7000b7a 02012f7c
	s_wait_alu depctr_va_sdst(0)                               // 00000000397c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v125, s11               // 000000003980: d5207c7b 002efa80
	s_and_b32 s11, s1, s5                                      // 000000003988: 8b0b0501
	v_and_b16 v51.h, 0xff, v51.h op_sel:[0,1,1]                // 00000000398c: d7625033 020266ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003998: bf88ff9e
	v_cndmask_b32_e64 v122, 0, v122, s11                       // 00000000399c: d501007a 002ef480
	v_cndmask_b32_e64 v123, 0, v123, s11                       // 0000000039a4: d501007b 002ef680
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000039ac: bf870122
	v_add_co_u32 v122, s12, s34, v122                          // 0000000039b0: d7000c7a 0202f422
	s_wait_alu depctr_va_sdst(0)                               // 0000000039b8: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s35, v123, s12             // 0000000039bc: d5207c7b 0032f623
	global_load_d16_hi_u8 v53, v[122:123], off                 // 0000000039c4: ee08407c 00000035 0000007a
	s_wait_loadcnt 0x0                                         // 0000000039d0: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 0000000039d4: d65d5035 002e6a80
	v_add_co_u32 v52, s11, v126, 16                            // 0000000039dc: d7000b34 0201217e
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 0000000039e4: bf870092
	v_lshlrev_b16 v53.h, 8, v53.h op_sel:[0,1,1]               // 0000000039e8: d7385035 02026a88
	v_or_b16 v51.h, v51.h, v53.h op_sel:[1,1,1]                // 0000000039f0: d7635833 02026b33
	s_wait_alu depctr_va_sdst(0)                               // 0000000039f8: bf88f19f
	v_add_co_ci_u32_e64 v53, null, 0, v127, s11                // 0000000039fc: d5207c35 002efe80
	s_and_b32 s11, s2, s8                                      // 000000003a04: 8b0b0802
	s_and_b32 s8, s3, s8                                       // 000000003a08: 8b080803
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a0c: bf88ff9e
	v_cndmask_b32_e64 v52, 0, v52, s11                         // 000000003a10: d5010034 002e6880
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003a18: d5010035 002e6a80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003a20: bf870122
	v_add_co_u32 v52, s12, s30, v52                            // 000000003a24: d7000c34 0202681e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a2c: bf88f19f
	v_add_co_ci_u32_e64 v53, null, s31, v53, s12               // 000000003a30: d5207c35 00326a1f
	global_load_d16_u8 v52, v[52:53], off                      // 000000003a38: ee07807c 00000034 00000034
	s_wait_loadcnt 0x0                                         // 000000003a44: bfc00000
	v_cndmask_b16 v52.l, 0, v52.l, s11                         // 000000003a48: d65d0034 002e6880
	v_add_co_u32 v53, s11, v126, 17                            // 000000003a50: d7000b35 0201237e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a58: bf88f19f
	v_add_co_ci_u32_e64 v122, null, 0, v127, s11               // 000000003a5c: d5207c7a 002efe80
	s_and_b32 s11, s2, s9                                      // 000000003a64: 8b0b0902
	v_and_b16 v52.l, 0xff, v52.l                               // 000000003a68: d7620034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003a74: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003a78: d5010035 002e6a80
	v_cndmask_b32_e64 v123, 0, v122, s11                       // 000000003a80: d501007b 002ef480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003a88: bf870122
	v_add_co_u32 v122, s12, s30, v53                           // 000000003a8c: d7000c7a 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003a94: bf88f19f
	v_add_co_ci_u32_e64 v123, null, s31, v123, s12             // 000000003a98: d5207c7b 0032f61f
	global_load_d16_hi_u8 v52, v[122:123], off                 // 000000003aa0: ee08407c 00000034 0000007a
	s_wait_loadcnt 0x0                                         // 000000003aac: bfc00000
	v_cndmask_b16 v122.l, 0, v52.h, s11                        // 000000003ab0: d65d107a 002e6880
	v_add_co_u32 v53, s11, v126, 18                            // 000000003ab8: d7000b35 0201257e
	s_wait_alu depctr_va_sdst(0)                               // 000000003ac0: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v127, s11               // 000000003ac4: d5207c7b 002efe80
	s_and_b32 s11, s2, s10                                     // 000000003acc: 8b0b0a02
	v_lshlrev_b16 v122.l, 8, v122.l                            // 000000003ad0: d738007a 0202f488
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ad8: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003adc: d5010035 002e6a80
	v_cndmask_b32_e64 v124, 0, v123, s11                       // 000000003ae4: d501007c 002ef680
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003aec: bf870193
	v_or_b16 v52.l, v52.l, v122.l                              // 000000003af0: d7630034 0202f534
	v_add_co_u32 v123, s12, s30, v53                           // 000000003af8: d7000c7b 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b00: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003b04: bf870003
	v_add_co_ci_u32_e64 v124, null, s31, v124, s12             // 000000003b08: d5207c7c 0032f81f
	global_load_d16_hi_u8 v52, v[123:124], off                 // 000000003b10: ee08407c 00000034 0000007b
	s_wait_loadcnt 0x0                                         // 000000003b1c: bfc00000
	v_cndmask_b16 v52.h, 0, v52.h, s11                         // 000000003b20: d65d5034 002e6880
	v_add_co_u32 v53, s11, v126, 19                            // 000000003b28: d7000b35 0201277e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b30: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v127, s11               // 000000003b34: d5207c7b 002efe80
	s_and_b32 s11, s2, s7                                      // 000000003b3c: 8b0b0702
	v_and_b16 v52.h, 0xff, v52.h op_sel:[0,1,1]                // 000000003b40: d7625034 020268ff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003b4c: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003b50: d5010035 002e6a80
	v_cndmask_b32_e64 v124, 0, v123, s11                       // 000000003b58: d501007c 002ef680
	s_and_b32 s7, s3, s7                                       // 000000003b60: 8b070703
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003b64: bf870122
	v_add_co_u32 v123, s12, s30, v53                           // 000000003b68: d7000c7b 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b70: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s31, v124, s12             // 000000003b74: d5207c7c 0032f81f
	global_load_d16_u8 v53, v[123:124], off                    // 000000003b7c: ee07807c 00000035 0000007b
	s_wait_loadcnt 0x0                                         // 000000003b88: bfc00000
	v_cndmask_b16 v122.h, 0, v53.l, s11                        // 000000003b8c: d65d407a 002e6a80
	v_add_co_u32 v53, s11, v126, 20                            // 000000003b94: d7000b35 0201297e
	s_wait_alu depctr_va_sdst(0)                               // 000000003b9c: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v127, s11               // 000000003ba0: d5207c7b 002efe80
	s_and_b32 s11, s2, s6                                      // 000000003ba8: 8b0b0602
	v_lshlrev_b16 v122.h, 8, v122.h op_sel:[0,1,1]             // 000000003bac: d738507a 0202f488
	s_wait_alu depctr_sa_sdst(0)                               // 000000003bb4: bf88ff9e
	v_cndmask_b32_e64 v53, 0, v53, s11                         // 000000003bb8: d5010035 002e6a80
	v_cndmask_b32_e64 v124, 0, v123, s11                       // 000000003bc0: d501007c 002ef680
	s_and_b32 s6, s3, s6                                       // 000000003bc8: 8b060603
	v_or_b16 v52.h, v52.h, v122.h op_sel:[1,1,1]               // 000000003bcc: d7635834 0202f534
	s_delay_alu instid0(valu_dep_3)                            // 000000003bd4: bf870003
	v_add_co_u32 v123, s12, s30, v53                           // 000000003bd8: d7000c7b 02026a1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003be0: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s31, v124, s12             // 000000003be4: d5207c7c 0032f81f
	global_load_d16_u8 v53, v[123:124], off                    // 000000003bec: ee07807c 00000035 0000007b
	s_wait_loadcnt 0x0                                         // 000000003bf8: bfc00000
	v_cndmask_b16 v53.l, 0, v53.l, s11                         // 000000003bfc: d65d0035 002e6a80
	v_add_co_u32 v123, s11, v126, 21                           // 000000003c04: d7000b7b 02012b7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c0c: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v127, s11               // 000000003c10: d5207c7c 002efe80
	s_and_b32 s11, s2, vcc_lo                                  // 000000003c18: 8b0b6a02
	v_and_b16 v53.l, 0xff, v53.l                               // 000000003c1c: d7620035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c28: bf88ff9e
	v_cndmask_b32_e64 v123, 0, v123, s11                       // 000000003c2c: d501007b 002ef680
	v_cndmask_b32_e64 v124, 0, v124, s11                       // 000000003c34: d501007c 002ef880
	s_and_b32 vcc_lo, s3, vcc_lo                               // 000000003c3c: 8b6a6a03
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003c40: bf870122
	v_add_co_u32 v123, s12, s30, v123                          // 000000003c44: d7000c7b 0202f61e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c4c: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s31, v124, s12             // 000000003c50: d5207c7c 0032f81f
	global_load_d16_hi_u8 v53, v[123:124], off                 // 000000003c58: ee08407c 00000035 0000007b
	s_wait_loadcnt 0x0                                         // 000000003c64: bfc00000
	v_cndmask_b16 v123.l, 0, v53.h, s11                        // 000000003c68: d65d107b 002e6a80
	v_add_co_u32 v124, s11, v126, 22                           // 000000003c70: d7000b7c 02012d7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003c78: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s11               // 000000003c7c: d5207c7d 002efe80
	s_and_b32 s11, s2, s4                                      // 000000003c84: 8b0b0402
	v_lshlrev_b16 v123.l, 8, v123.l                            // 000000003c88: d738007b 0202f688
	s_wait_alu depctr_sa_sdst(0)                               // 000000003c90: bf88ff9e
	v_cndmask_b32_e64 v124, 0, v124, s11                       // 000000003c94: d501007c 002ef880
	v_cndmask_b32_e64 v125, 0, v125, s11                       // 000000003c9c: d501007d 002efa80
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003ca4: bf870193
	v_or_b16 v53.l, v53.l, v123.l                              // 000000003ca8: d7630035 0202f735
	v_add_co_u32 v124, s12, s30, v124                          // 000000003cb0: d7000c7c 0202f81e
	s_wait_alu depctr_va_sdst(0)                               // 000000003cb8: bf88f19f
	s_delay_alu instid0(valu_dep_3)                            // 000000003cbc: bf870003
	v_add_co_ci_u32_e64 v125, null, s31, v125, s12             // 000000003cc0: d5207c7d 0032fa1f
	global_load_d16_hi_u8 v53, v[124:125], off                 // 000000003cc8: ee08407c 00000035 0000007c
	s_wait_loadcnt 0x0                                         // 000000003cd4: bfc00000
	v_cndmask_b16 v53.h, 0, v53.h, s11                         // 000000003cd8: d65d5035 002e6a80
	v_add_co_u32 v124, s11, v126, 23                           // 000000003ce0: d7000b7c 02012f7e
	s_wait_alu depctr_va_sdst(0)                               // 000000003ce8: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v127, s11               // 000000003cec: d5207c7d 002efe80
	s_and_b32 s11, s2, s5                                      // 000000003cf4: 8b0b0502
	v_and_b16 v53.h, 0xff, v53.h op_sel:[0,1,1]                // 000000003cf8: d7625035 02026aff 000000ff
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d04: bf88ff9e
	v_cndmask_b32_e64 v124, 0, v124, s11                       // 000000003d08: d501007c 002ef880
	v_cndmask_b32_e64 v125, 0, v125, s11                       // 000000003d10: d501007d 002efa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003d18: bf870122
	v_add_co_u32 v124, s12, s30, v124                          // 000000003d1c: d7000c7c 0202f81e
	s_wait_alu depctr_va_sdst(0)                               // 000000003d24: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s12             // 000000003d28: d5207c7d 0032fa1f
	global_load_d16_hi_u8 v123, v[124:125], off                // 000000003d30: ee08407c 0000007b 0000007c
	s_wait_loadcnt 0x0                                         // 000000003d3c: bfc00000
	v_cndmask_b16 v123.h, 0, v123.h, s11                       // 000000003d40: d65d507b 002ef680
	v_add_co_u32 v122, s11, v120, 16                           // 000000003d48: d7000b7a 02012178
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003d50: bf870112
	v_lshlrev_b16 v123.h, 8, v123.h op_sel:[0,1,1]             // 000000003d54: d738507b 0202f688
	v_cndmask_b32_e64 v122, 0, v122, s8                        // 000000003d5c: d501007a 0022f480
	s_delay_alu instid0(valu_dep_2) | instskip(skip_2) | instid1(valu_dep_3)// 000000003d64: bf8701b2
	v_or_b16 v53.h, v53.h, v123.h op_sel:[1,1,1]               // 000000003d68: d7635835 0202f735
	s_wait_alu depctr_va_sdst(0)                               // 000000003d70: bf88f19f
	v_add_co_ci_u32_e64 v123, null, 0, v121, s11               // 000000003d74: d5207c7b 002ef280
	v_add_co_u32 v122, s11, s30, v122                          // 000000003d7c: d7000b7a 0202f41e
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003d84: bf870193
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[48:49], v[52:53], v[24:31]// 000000003d88: cc464018 1c626930
	v_cndmask_b32_e64 v123, 0, v123, s8                        // 000000003d90: d501007b 0022f680
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[50:51], v[52:53], v[8:15]// 000000003d98: cc464008 1c226932
	s_wait_alu depctr_va_sdst(0)                               // 000000003da0: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003da4: bf870002
	v_add_co_ci_u32_e64 v123, null, s31, v123, s11             // 000000003da8: d5207c7b 002ef61f
	global_load_d16_u8 v122, v[122:123], off                   // 000000003db0: ee07807c 0000007a 0000007a
	s_wait_loadcnt 0x0                                         // 000000003dbc: bfc00000
	v_cndmask_b16 v122.l, 0, v122.l, s8                        // 000000003dc0: d65d007a 0022f480
	v_add_co_u32 v123, s8, v120, 17                            // 000000003dc8: d700087b 02012378
	s_wait_alu depctr_va_sdst(0)                               // 000000003dd0: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v121, s8                // 000000003dd4: d5207c7c 0022f280
	s_and_b32 s8, s3, s9                                       // 000000003ddc: 8b080903
	s_wait_alu depctr_sa_sdst(0)                               // 000000003de0: bf88ff9e
	v_cndmask_b32_e64 v123, 0, v123, s8                        // 000000003de4: d501007b 0022f680
	v_cndmask_b32_e64 v124, 0, v124, s8                        // 000000003dec: d501007c 0022f880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003df4: bf870122
	v_add_co_u32 v123, s9, s30, v123                           // 000000003df8: d700097b 0202f61e
	s_wait_alu depctr_va_sdst(0)                               // 000000003e00: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s31, v124, s9              // 000000003e04: d5207c7c 0026f81f
	global_load_d16_hi_u8 v122, v[123:124], off                // 000000003e0c: ee08407c 0000007a 0000007b
	s_wait_loadcnt 0x0                                         // 000000003e18: bfc00000
	v_cndmask_b16 v122.h, 0, v122.h, s8                        // 000000003e1c: d65d507a 0022f480
	v_add_co_u32 v123, s8, v120, 18                            // 000000003e24: d700087b 02012578
	s_wait_alu depctr_va_sdst(0)                               // 000000003e2c: bf88f19f
	v_add_co_ci_u32_e64 v124, null, 0, v121, s8                // 000000003e30: d5207c7c 0022f280
	s_and_b32 s8, s3, s10                                      // 000000003e38: 8b080a03
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e3c: bf88ff9e
	v_cndmask_b32_e64 v123, 0, v123, s8                        // 000000003e40: d501007b 0022f680
	v_cndmask_b32_e64 v124, 0, v124, s8                        // 000000003e48: d501007c 0022f880
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003e50: bf870122
	v_add_co_u32 v123, s9, s30, v123                           // 000000003e54: d700097b 0202f61e
	s_wait_alu depctr_va_sdst(0)                               // 000000003e5c: bf88f19f
	v_add_co_ci_u32_e64 v124, null, s31, v124, s9              // 000000003e60: d5207c7c 0026f81f
	global_load_d16_u8 v123, v[123:124], off                   // 000000003e68: ee07807c 0000007b 0000007b
	s_wait_loadcnt 0x0                                         // 000000003e74: bfc00000
	v_cndmask_b16 v123.l, 0, v123.l, s8                        // 000000003e78: d65d007b 0022f680
	v_add_co_u32 v124, s8, v120, 19                            // 000000003e80: d700087c 02012778
	s_wait_alu depctr_va_sdst(0)                               // 000000003e88: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v121, s8                // 000000003e8c: d5207c7d 0022f280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003e94: bf870112
	v_cndmask_b32_e64 v124, 0, v124, s7                        // 000000003e98: d501007c 001ef880
	v_cndmask_b32_e64 v125, 0, v125, s7                        // 000000003ea0: d501007d 001efa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003ea8: bf870122
	v_add_co_u32 v124, s8, s30, v124                           // 000000003eac: d700087c 0202f81e
	s_wait_alu depctr_va_sdst(0)                               // 000000003eb4: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s8              // 000000003eb8: d5207c7d 0022fa1f
	global_load_d16_hi_u8 v123, v[124:125], off                // 000000003ec0: ee08407c 0000007b 0000007c
	s_wait_loadcnt 0x0                                         // 000000003ecc: bfc00000
	v_cndmask_b16 v123.h, 0, v123.h, s7                        // 000000003ed0: d65d507b 001ef680
	v_add_co_u32 v124, s7, v120, 20                            // 000000003ed8: d700077c 02012978
	s_wait_alu depctr_va_sdst(0)                               // 000000003ee0: bf88f19f
	v_add_co_ci_u32_e64 v125, null, 0, v121, s7                // 000000003ee4: d5207c7d 001ef280
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000003eec: bf870112
	v_cndmask_b32_e64 v124, 0, v124, s6                        // 000000003ef0: d501007c 001af880
	v_cndmask_b32_e64 v125, 0, v125, s6                        // 000000003ef8: d501007d 001afa80
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003f00: bf870122
	v_add_co_u32 v124, s7, s30, v124                           // 000000003f04: d700077c 0202f81e
	s_wait_alu depctr_va_sdst(0)                               // 000000003f0c: bf88f19f
	v_add_co_ci_u32_e64 v125, null, s31, v125, s7              // 000000003f10: d5207c7d 001efa1f
	global_load_d16_u8 v124, v[124:125], off                   // 000000003f18: ee07807c 0000007c 0000007c
	s_wait_loadcnt 0x0                                         // 000000003f24: bfc00000
	v_cndmask_b16 v124.l, 0, v124.l, s6                        // 000000003f28: d65d007c 001af880
	v_add_co_u32 v125, s6, v120, 21                            // 000000003f30: d700067d 02012b78
	s_wait_alu depctr_va_sdst(0)                               // 000000003f38: bf88f19f
	v_add_co_ci_u32_e64 v126, null, 0, v121, s6                // 000000003f3c: d5207c7e 001af280
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003f44: bf870091
	v_dual_cndmask_b32 v125, 0, v125 :: v_dual_cndmask_b32 v126, 0, v126// 000000003f48: ca52fa80 7d7efc80
	v_add_co_u32 v125, s6, s30, v125                           // 000000003f50: d700067d 0202fa1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003f58: bf88f19f
	s_delay_alu instid0(valu_dep_2)                            // 000000003f5c: bf870002
	v_add_co_ci_u32_e64 v126, null, s31, v126, s6              // 000000003f60: d5207c7e 001afc1f
	global_load_d16_hi_u8 v124, v[125:126], off                // 000000003f68: ee08407c 0000007c 0000007d
	s_wait_loadcnt 0x0                                         // 000000003f74: bfc00000
	v_cndmask_b16 v124.h, 0, v124.h, vcc_lo                    // 000000003f78: d65d507c 01aaf880
	v_add_co_u32 v125, vcc_lo, v120, 22                        // 000000003f80: d7006a7d 02012d78
	s_wait_alu depctr_va_vcc(0)                                // 000000003f88: bf88ff9d
	v_add_co_ci_u32_e64 v126, null, 0, v121, vcc_lo            // 000000003f8c: d5207c7e 01aaf280
	s_and_b32 vcc_lo, s3, s4                                   // 000000003f94: 8b6a0403
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f98: bf88ff9e
	v_dual_cndmask_b32 v125, 0, v125 :: v_dual_cndmask_b32 v126, 0, v126// 000000003f9c: ca52fa80 7d7efc80
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003fa4: bf870121
	v_add_co_u32 v125, s4, s30, v125                           // 000000003fa8: d700047d 0202fa1e
	s_wait_alu depctr_va_sdst(0)                               // 000000003fb0: bf88f19f
	v_add_co_ci_u32_e64 v126, null, s31, v126, s4              // 000000003fb4: d5207c7e 0012fc1f
	global_load_d16_u8 v125, v[125:126], off                   // 000000003fbc: ee07807c 0000007d 0000007d
	s_wait_loadcnt 0x0                                         // 000000003fc8: bfc00000
	v_cndmask_b16 v125.l, 0, v125.l, vcc_lo                    // 000000003fcc: d65d007d 01aafa80
	v_add_co_u32 v120, vcc_lo, v120, 23                        // 000000003fd4: d7006a78 02012f78
	s_wait_alu depctr_va_vcc(0)                                // 000000003fdc: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v121, vcc_lo            // 000000003fe0: d5207c79 01aaf280
	s_and_b32 vcc_lo, s3, s5                                   // 000000003fe8: 8b6a0503
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fec: bf88ff9e
	v_dual_cndmask_b32 v120, 0, v120 :: v_dual_cndmask_b32 v121, 0, v121// 000000003ff0: ca52f080 7878f280
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003ff8: bf870121
	v_add_co_u32 v120, s4, s30, v120                           // 000000003ffc: d7000478 0202f01e
	s_wait_alu depctr_va_sdst(0)                               // 000000004004: bf88f19f
	v_add_co_ci_u32_e64 v121, null, s31, v121, s4              // 000000004008: d5207c79 0012f21f
	v_cmp_lt_u64_e64 s4, s[46:47], s[42:43]                    // 000000004010: d4590004 0200542e
	global_load_d16_u8 v120, v[120:121], off                   // 000000004018: ee07807c 00000078 00000078
	s_wait_loadcnt 0x0                                         // 000000004024: bfc00000
	v_and_b16 v120.h, 0xff, v125.l op_sel:[0,0,1]              // 000000004028: d7624078 0202faff 000000ff
	v_cndmask_b16 v120.l, 0, v120.l, vcc_lo                    // 000000004034: d65d0078 01aaf080
	s_and_b32 vcc_lo, exec_lo, s4                              // 00000000403c: 8b6a047e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004040: bf870091
	v_lshlrev_b16 v120.l, 8, v120.l                            // 000000004044: d7380078 0202f088
	v_or_b16 v125.h, v120.h, v120.l op_sel:[1,0,1]             // 00000000404c: d763487d 0202f178
	v_lshlrev_b16 v120.l, 8, v124.h op_sel:[0,1,0]             // 000000004054: d7381078 0202f888
	v_and_b16 v120.h, 0xff, v124.l op_sel:[0,0,1]              // 00000000405c: d7624078 0202f8ff 000000ff
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000004068: bf8700b1
	v_or_b16 v125.l, v120.h, v120.l op_sel:[1,0,0]             // 00000000406c: d763087d 0202f178
	v_lshlrev_b16 v120.l, 8, v123.h op_sel:[0,1,0]             // 000000004074: d7381078 0202f688
	v_and_b16 v120.h, 0xff, v123.l op_sel:[0,0,1]              // 00000000407c: d7624078 0202f6ff 000000ff
	v_or_b16 v124.h, v120.h, v120.l op_sel:[1,0,1]             // 000000004088: d763487c 0202f178
	v_lshlrev_b16 v120.l, 8, v122.h op_sel:[0,1,0]             // 000000004090: d7381078 0202f488
	v_and_b16 v120.h, 0xff, v122.l op_sel:[0,0,1]              // 000000004098: d7624078 0202f4ff 000000ff
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000040a4: bf870091
	v_or_b16 v124.l, v120.h, v120.l op_sel:[1,0,0]             // 0000000040a8: d763087c 0202f178
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[48:49], v[124:125], v[16:23]// 0000000040b0: cc464010 1c42f930
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[50:51], v[124:125], v[0:7]// 0000000040b8: cc464000 1c02f932
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040c0: bf88ff9e
	s_cbranch_vccnz 63695                                      // 0000000040c4: bfa4f8cf <tessera_rocm_scaled_matmul_28d379a9237322d1+0x904>
	s_lshr_b64 s[4:5], s[44:45], 5                             // 0000000040c8: 8584852c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000040cc: bf88ff9e
	v_add_co_u32 v48, vcc_lo, v81, s4                          // 0000000040d0: d7006a30 02000951
	s_wait_alu depctr_va_vcc(0)                                // 0000000040d8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v82, vcc_lo             // 0000000040dc: d5207c31 01aaa405
	v_add_co_u32 v50, vcc_lo, v84, s4                          // 0000000040e4: d7006a32 02000954
	s_wait_alu depctr_va_vcc(0)                                // 0000000040ec: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s5, v85, vcc_lo             // 0000000040f0: d5207c33 01aaaa05
	v_add_co_u32 v52, vcc_lo, v86, s4                          // 0000000040f8: d7006a34 02000956
	s_wait_alu depctr_va_vcc(0)                                // 000000004100: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s5, v88, vcc_lo             // 000000004104: d5207c35 01aab005
	v_add_co_u32 v120, vcc_lo, v89, s4                         // 00000000410c: d7006a78 02000959
	s_wait_alu depctr_va_vcc(0)                                // 000000004114: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s5, v90, vcc_lo            // 000000004118: d5207c79 01aab405
	v_add_co_u32 v122, vcc_lo, v92, s4                         // 000000004120: d7006a7a 0200095c
	s_wait_alu depctr_va_vcc(0)                                // 000000004128: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s5, v93, vcc_lo            // 00000000412c: d5207c7b 01aaba05
	v_add_co_u32 v124, vcc_lo, v95, s4                         // 000000004134: d7006a7c 0200095f
	s_wait_alu depctr_va_vcc(0)                                // 00000000413c: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s5, v96, vcc_lo            // 000000004140: d5207c7d 01aac005
	v_add_co_u32 v126, vcc_lo, v98, s4                         // 000000004148: d7006a7e 02000962
	s_wait_alu depctr_va_vcc(0)                                // 000000004150: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s5, v99, vcc_lo            // 000000004154: d5207c7f 01aac605
	v_add_co_u32 v128, vcc_lo, v101, s4                        // 00000000415c: d7006a80 02000965
	s_wait_alu depctr_va_vcc(0)                                // 000000004164: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, s5, v102, vcc_lo           // 000000004168: d5207c81 01aacc05
	s_clause 0x7                                               // 000000004170: bf850007
	global_load_b32 v130, v[48:49], off                        // 000000004174: ee05007c 00000082 00000030
	global_load_b32 v131, v[50:51], off                        // 000000004180: ee05007c 00000083 00000032
	global_load_b32 v132, v[52:53], off                        // 00000000418c: ee05007c 00000084 00000034
	global_load_b32 v133, v[120:121], off                      // 000000004198: ee05007c 00000085 00000078
	global_load_b32 v134, v[122:123], off                      // 0000000041a4: ee05007c 00000086 0000007a
	global_load_b32 v135, v[124:125], off                      // 0000000041b0: ee05007c 00000087 0000007c
	global_load_b32 v136, v[126:127], off                      // 0000000041bc: ee05007c 00000088 0000007e
	global_load_b32 v137, v[128:129], off                      // 0000000041c8: ee05007c 00000089 00000080
	v_add_co_u32 v48, vcc_lo, v103, s4                         // 0000000041d4: d7006a30 02000967
	s_wait_alu depctr_va_vcc(0)                                // 0000000041dc: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v104, vcc_lo            // 0000000041e0: d5207c31 01aad005
	v_add_co_u32 v50, vcc_lo, v106, s4                         // 0000000041e8: d7006a32 0200096a
	s_wait_alu depctr_va_vcc(0)                                // 0000000041f0: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s5, v107, vcc_lo            // 0000000041f4: d5207c33 01aad605
	v_add_co_u32 v52, vcc_lo, v108, s4                         // 0000000041fc: d7006a34 0200096c
	s_wait_alu depctr_va_vcc(0)                                // 000000004204: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s5, v109, vcc_lo            // 000000004208: d5207c35 01aada05
	v_add_co_u32 v120, vcc_lo, v110, s4                        // 000000004210: d7006a78 0200096e
	s_wait_alu depctr_va_vcc(0)                                // 000000004218: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s5, v111, vcc_lo           // 00000000421c: d5207c79 01aade05
	v_add_co_u32 v122, vcc_lo, v112, s4                        // 000000004224: d7006a7a 02000970
	s_wait_alu depctr_va_vcc(0)                                // 00000000422c: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s5, v113, vcc_lo           // 000000004230: d5207c7b 01aae205
	v_add_co_u32 v124, vcc_lo, v114, s4                        // 000000004238: d7006a7c 02000972
	s_wait_alu depctr_va_vcc(0)                                // 000000004240: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s5, v115, vcc_lo           // 000000004244: d5207c7d 01aae605
	v_add_co_u32 v126, vcc_lo, v116, s4                        // 00000000424c: d7006a7e 02000974
	s_wait_alu depctr_va_vcc(0)                                // 000000004254: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s5, v117, vcc_lo           // 000000004258: d5207c7f 01aaea05
	v_add_co_u32 v128, vcc_lo, v118, s4                        // 000000004260: d7006a80 02000976
	s_wait_alu depctr_va_vcc(0)                                // 000000004268: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, s5, v119, vcc_lo           // 00000000426c: d5207c81 01aaee05
	s_clause 0x7                                               // 000000004274: bf850007
	global_load_b32 v48, v[48:49], off                         // 000000004278: ee05007c 00000030 00000030
	global_load_b32 v49, v[50:51], off                         // 000000004284: ee05007c 00000031 00000032
	global_load_b32 v50, v[52:53], off                         // 000000004290: ee05007c 00000032 00000034
	global_load_b32 v51, v[120:121], off                       // 00000000429c: ee05007c 00000033 00000078
	global_load_b32 v52, v[122:123], off                       // 0000000042a8: ee05007c 00000034 0000007a
	global_load_b32 v53, v[124:125], off                       // 0000000042b4: ee05007c 00000035 0000007c
	global_load_b32 v120, v[126:127], off                      // 0000000042c0: ee05007c 00000078 0000007e
	global_load_b32 v121, v[128:129], off                      // 0000000042cc: ee05007c 00000079 00000080
	s_lshr_b64 s[4:5], s[44:45], 7                             // 0000000042d8: 8584872c
	s_mov_b64 s[44:45], s[42:43]                               // 0000000042dc: beac012a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042e0: bf88ff9e
	s_mul_u64 s[4:5], s[4:5], s[14:15]                         // 0000000042e4: aa840e04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042e8: bf88ff9e
	s_lshl_b64 s[4:5], s[4:5], 2                               // 0000000042ec: 84848204
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042f0: bf88ff9e
	s_add_nc_u64 s[4:5], s[24:25], s[4:5]                      // 0000000042f4: a9840418
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042f8: bf88ff9e
	s_add_nc_u64 s[6:7], s[4:5], s[40:41]                      // 0000000042fc: a9862804
	s_clause 0x1                                               // 000000004300: bf850001
	s_load_b32 s6, s[6:7], 0x0                                 // 000000004304: f4000183 f8000000
	s_load_b32 s4, s[4:5], 0x0                                 // 00000000430c: f4000102 f8000000
	v_cmp_lt_i64_e64 s5, s[42:43], s[26:27]                    // 000000004314: d4510005 0200342a
	s_and_b32 vcc_lo, exec_lo, s5                              // 00000000431c: 8b6a057e
	s_wait_kmcnt 0x0                                           // 000000004320: bfc70000
	v_mov_b32_e32 v122, s6                                     // 000000004324: 7ef40206
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_1)// 000000004328: bf8700a1
	v_cndmask_b32_e64 v123, s4, v122, s2                       // 00000000432c: d501007b 000af404
	s_wait_loadcnt 0xf                                         // 000000004334: bfc0000f
	v_mul_f32_e32 v124, v130, v123                             // 000000004338: 10f8f782
	s_wait_loadcnt 0xe                                         // 00000000433c: bfc0000e
	v_mul_f32_e32 v125, v123, v131                             // 000000004340: 10fb077b
	v_cndmask_b32_e64 v122, s4, v122, s3                       // 000000004344: d501007a 000ef404
	s_wait_loadcnt 0xc                                         // 00000000434c: bfc0000c
	v_dual_mul_f32 v126, v123, v132 :: v_dual_mul_f32 v127, v123, v133// 000000004350: c8c7097b 7e7f0b7b
	s_wait_loadcnt 0xa                                         // 000000004358: bfc0000a
	v_dual_mul_f32 v128, v123, v134 :: v_dual_mul_f32 v129, v123, v135// 00000000435c: c8c70d7b 80810f7b
	s_wait_loadcnt 0x9                                         // 000000004364: bfc00009
	v_dual_mul_f32 v138, v123, v136 :: v_dual_mul_f32 v131, v122, v131// 000000004368: c8c7117b 8a83077a
	s_wait_loadcnt 0x8                                         // 000000004370: bfc00008
	v_dual_mul_f32 v139, v123, v137 :: v_dual_mul_f32 v130, v130, v122// 000000004374: c8c7137b 8b82f582
	v_dual_mul_f32 v132, v122, v132 :: v_dual_mul_f32 v133, v122, v133// 00000000437c: c8c7097a 84850b7a
	v_dual_mul_f32 v134, v122, v134 :: v_dual_mul_f32 v135, v122, v135// 000000004384: c8c70d7a 86870f7a
	v_dual_mul_f32 v136, v122, v136 :: v_dual_mul_f32 v137, v122, v137// 00000000438c: c8c7117a 8889137a
	v_dual_mul_f32 v24, v24, v124 :: v_dual_mul_f32 v27, v27, v127// 000000004394: c8c6f918 181aff1b
	v_dual_mul_f32 v25, v25, v125 :: v_dual_mul_f32 v26, v26, v126// 00000000439c: c8c6fb19 191afd1a
	v_dual_mul_f32 v29, v29, v129 :: v_dual_mul_f32 v28, v28, v128// 0000000043a4: c8c7031d 1d1d011c
	v_dual_mul_f32 v31, v31, v139 :: v_dual_mul_f32 v30, v30, v138// 0000000043ac: c8c7171f 1f1f151e
	v_dual_mul_f32 v17, v17, v131 :: v_dual_mul_f32 v18, v18, v132// 0000000043b4: c8c70711 11130912
	v_mul_f32_e32 v21, v21, v135                               // 0000000043bc: 102b0f15
	v_dual_mul_f32 v19, v19, v133 :: v_dual_mul_f32 v20, v20, v134// 0000000043c0: c8c70b13 13150d14
	v_mul_f32_e32 v23, v23, v137                               // 0000000043c8: 102f1317
	v_dual_add_f32 v105, v105, v24 :: v_dual_add_f32 v100, v100, v25// 0000000043cc: c9083169 69643364
	s_wait_loadcnt 0x6                                         // 0000000043d4: bfc00006
	v_dual_mul_f32 v140, v123, v48 :: v_dual_mul_f32 v141, v123, v49// 0000000043d8: c8c6617b 8c8c637b
	s_wait_loadcnt 0x4                                         // 0000000043e0: bfc00004
	v_dual_mul_f32 v142, v123, v50 :: v_dual_mul_f32 v143, v123, v51// 0000000043e4: c8c6657b 8e8e677b
	s_wait_loadcnt 0x2                                         // 0000000043ec: bfc00002
	v_dual_mul_f32 v144, v123, v52 :: v_dual_mul_f32 v145, v123, v53// 0000000043f0: c8c6697b 90906b7b
	s_wait_loadcnt 0x1                                         // 0000000043f8: bfc00001
	v_dual_mul_f32 v146, v123, v120 :: v_dual_mul_f32 v49, v122, v49// 0000000043fc: c8c6f17b 9230637a
	s_wait_loadcnt 0x0                                         // 000000004404: bfc00000
	v_dual_mul_f32 v123, v123, v121 :: v_dual_mul_f32 v48, v122, v48// 000000004408: c8c6f37b 7b30617a
	v_dual_mul_f32 v51, v122, v51 :: v_dual_mul_f32 v50, v122, v50// 000000004410: c8c6677a 3332657a
	v_dual_mul_f32 v53, v122, v53 :: v_dual_mul_f32 v52, v122, v52// 000000004418: c8c66b7a 3534697a
	v_dual_mul_f32 v121, v122, v121 :: v_dual_mul_f32 v120, v122, v120// 000000004420: c8c6f37a 7978f17a
	v_mul_f32_e32 v16, v16, v130                               // 000000004428: 10210510
	v_dual_mul_f32 v22, v22, v136 :: v_dual_mul_f32 v9, v9, v141// 00000000442c: c8c71116 16091b09
	v_dual_mul_f32 v8, v8, v140 :: v_dual_mul_f32 v11, v11, v143// 000000004434: c8c71908 080b1f0b
	v_dual_mul_f32 v10, v10, v142 :: v_dual_mul_f32 v13, v13, v145// 00000000443c: c8c71d0a 0a0d230d
	v_dual_mul_f32 v12, v12, v144 :: v_dual_mul_f32 v15, v15, v123// 000000004444: c8c7210c 0c0ef70f
	v_dual_mul_f32 v14, v14, v146 :: v_dual_mul_f32 v1, v1, v49// 00000000444c: c8c7250e 0e006301
	v_dual_mul_f32 v0, v0, v48 :: v_dual_mul_f32 v3, v3, v51   // 000000004454: c8c66100 00026703
	v_dual_mul_f32 v2, v2, v50 :: v_dual_mul_f32 v5, v5, v53   // 00000000445c: c8c66502 02046b05
	v_dual_mul_f32 v4, v4, v52 :: v_dual_mul_f32 v7, v7, v121  // 000000004464: c8c66904 0406f307
	v_dual_mul_f32 v6, v6, v120 :: v_dual_add_f32 v97, v97, v26// 00000000446c: c8c8f106 06603561
	v_dual_add_f32 v94, v94, v27 :: v_dual_add_f32 v91, v91, v28// 000000004474: c908375e 5e5a395b
	v_dual_add_f32 v87, v87, v29 :: v_dual_add_f32 v80, v80, v31// 00000000447c: c9083b57 57503f50
	v_dual_add_f32 v83, v83, v30 :: v_dual_add_f32 v70, v70, v16// 000000004484: c9083d53 53462146
	v_dual_add_f32 v69, v69, v17 :: v_dual_add_f32 v68, v68, v18// 00000000448c: c9082345 45442544
	v_dual_add_f32 v67, v67, v19 :: v_dual_add_f32 v66, v66, v20// 000000004494: c9082743 43422942
	v_dual_add_f32 v65, v65, v21 :: v_dual_add_f32 v64, v64, v22// 00000000449c: c9082b41 41402d40
	v_dual_add_f32 v62, v62, v23 :: v_dual_add_f32 v79, v79, v8// 0000000044a4: c9082f3e 3e4e114f
	v_dual_add_f32 v78, v78, v9 :: v_dual_add_f32 v77, v77, v10// 0000000044ac: c908134e 4e4c154d
	v_dual_add_f32 v75, v75, v11 :: v_dual_add_f32 v74, v74, v12// 0000000044b4: c908174b 4b4a194a
	v_dual_add_f32 v73, v73, v13 :: v_dual_add_f32 v72, v72, v14// 0000000044bc: c9081b49 49481d48
	v_dual_add_f32 v71, v71, v15 :: v_dual_add_f32 v60, v60, v2// 0000000044c4: c9081f47 473c053c
	v_dual_add_f32 v63, v63, v0 :: v_dual_add_f32 v56, v56, v6 // 0000000044cc: c908013f 3f380d38
	v_dual_add_f32 v61, v61, v1 :: v_dual_add_f32 v58, v58, v4 // 0000000044d4: c908033d 3d3a093a
	v_add_f32_e32 v59, v59, v3                                 // 0000000044dc: 0676073b
	v_add_f32_e32 v57, v57, v5                                 // 0000000044e0: 06720b39
	v_add_f32_e32 v55, v55, v7                                 // 0000000044e4: 066e0f37
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044e8: bf88ff9e
	s_cbranch_vccnz 63394                                      // 0000000044ec: bfa4f7a2 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x878>
	v_mul_lo_u32 v2, s19, v38                                  // 0000000044f0: d72c0002 02024c13
	v_mul_lo_u32 v3, s18, v39                                  // 0000000044f8: d72c0003 02024e12
	v_mad_co_u64_u32 v[0:1], null, s18, v38, 0                 // 000000004500: d6fe7c00 02024c12
	v_sub_co_u32 v14, vcc_lo, s16, v38                         // 000000004508: d7016a0e 02024c10
	s_wait_alu depctr_va_vcc(0)                                // 000000004510: bf88ff9d
	v_sub_co_ci_u32_e64 v15, null, s17, v39, vcc_lo            // 000000004514: d5217c0f 01aa4e11
	v_cmp_gt_i64_e64 s3, s[18:19], v[32:33]                    // 00000000451c: d4540003 02024012
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_3)// 000000004524: bf870194
	v_add3_u32 v1, v1, v3, v2                                  // 000000004528: d6550001 040a0701
	v_cmp_lt_i64_e32 vcc_lo, 0, v[14:15]                       // 000000004530: 7ca21c80
	s_delay_alu instid0(valu_dep_2)                            // 000000004534: bf870002
	v_lshlrev_b64_e32 v[6:7], 1, v[0:1]                        // 000000004538: 3e0c0081
	s_and_b32 s0, vcc_lo, s3                                   // 00000000453c: 8b00036a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004540: bf88ff9e
	s_and_saveexec_b32 s1, s0                                  // 000000004544: be812000
	s_cbranch_execz 28                                         // 000000004548: bfa5001c <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2abc>
	v_lshlrev_b64_e32 v[0:1], 1, v[32:33]                      // 00000000454c: 3e004081
	v_add_co_u32 v3, s0, s20, v6                               // 000000004550: d7000003 02020c14
	v_bfe_u32 v2, v105, 16, 1                                  // 000000004558: d6100002 02052169
	s_wait_alu depctr_va_sdst(0)                               // 000000004560: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s21, v7, s0                  // 000000004564: d5207c04 00020e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000456c: bf870193
	v_add_co_u32 v0, s0, v3, v0                                // 000000004570: d7000000 02020103
	v_add3_u32 v2, v2, v105, 0x7fff                            // 000000004578: d6550002 03fed302 00007fff
	v_or_b32_e32 v5, 0x400000, v105                            // 000000004584: 380ad2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 00000000458c: bf88f19f
	v_add_co_ci_u32_e64 v1, null, v4, v1, s0                   // 000000004590: d5207c01 00020304
	v_cmp_u_f32_e64 s0, v105, v105                             // 000000004598: d4180000 0202d369
	s_wait_alu depctr_va_sdst(0)                               // 0000000045a0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000045a4: bf870001
	v_cndmask_b32_e64 v2, v2, v5, s0                           // 0000000045a8: d5010002 00020b02
	global_store_d16_hi_b16 v[0:1], v2, off                    // 0000000045b0: ee09407c 01000000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000045c0: 8c7e017e
	v_add_co_u32 v0, s0, s18, v32                              // 0000000045c4: d7000000 02024012
	s_wait_alu depctr_va_sdst(0)                               // 0000000045cc: bf88f19f
	v_add_co_ci_u32_e64 v1, null, s19, v33, s0                 // 0000000045d0: d5207c01 00024213
	v_cmp_lt_i64_e64 s0, 1, v[14:15]                           // 0000000045d8: d4510000 02021c81
	s_delay_alu instid0(valu_dep_2)                            // 0000000045e0: bf870002
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000045e4: 3e000081
	s_and_b32 s1, s0, s3                                       // 0000000045e8: 8b010300
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045ec: bf88ff9e
	s_and_saveexec_b32 s2, s1                                  // 0000000045f0: be822001
	s_cbranch_execz 27                                         // 0000000045f4: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2b64>
	v_bfe_u32 v2, v100, 16, 1                                  // 0000000045f8: d6100002 02052164
	v_add_co_u32 v3, s1, s20, v6                               // 000000004600: d7000103 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004608: bf88f19f
	v_add_co_ci_u32_e64 v4, null, s21, v7, s1                  // 00000000460c: d5207c04 00060e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004614: bf870193
	v_add3_u32 v5, v2, v100, 0x7fff                            // 000000004618: d6550005 03fec902 00007fff
	v_add_co_u32 v2, s1, v3, v0                                // 000000004624: d7000102 02020103
	v_or_b32_e32 v8, 0x400000, v100                            // 00000000462c: 3810c8ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004634: bf88f19f
	v_add_co_ci_u32_e64 v3, null, v4, v1, s1                   // 000000004638: d5207c03 00060304
	v_cmp_u_f32_e64 s1, v100, v100                             // 000000004640: d4180001 0202c964
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 00000000464c: bf870001
	v_cndmask_b32_e64 v4, v5, v8, s1                           // 000000004650: d5010004 00061105
	global_store_d16_hi_b16 v[2:3], v4, off                    // 000000004658: ee09407c 02000000 00000002
	s_wait_alu depctr_sa_sdst(0)                               // 000000004664: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004668: 8c7e027e
	s_lshl_b64 s[4:5], s[18:19], 1                             // 00000000466c: 84848112
	s_wait_alu depctr_sa_sdst(0)                               // 000000004670: bf88ff9e
	v_add_co_u32 v2, s1, s4, v32                               // 000000004674: d7000102 02024004
	s_wait_alu depctr_va_sdst(0)                               // 00000000467c: bf88f19f
	v_add_co_ci_u32_e64 v3, null, s5, v33, s1                  // 000000004680: d5207c03 00064205
	v_cmp_lt_i64_e64 s1, 2, v[14:15]                           // 000000004688: d4510001 02021c82
	s_delay_alu instid0(valu_dep_2)                            // 000000004690: bf870002
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000004694: 3e040481
	s_and_b32 s2, s1, s3                                       // 000000004698: 8b020301
	s_wait_alu depctr_sa_sdst(0)                               // 00000000469c: bf88ff9e
	s_and_saveexec_b32 s4, s2                                  // 0000000046a0: be842002
	s_cbranch_execz 27                                         // 0000000046a4: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2c14>
	v_bfe_u32 v4, v97, 16, 1                                   // 0000000046a8: d6100004 02052161
	v_add_co_u32 v5, s2, s20, v6                               // 0000000046b0: d7000205 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000046b8: bf88f19f
	v_add_co_ci_u32_e64 v8, null, s21, v7, s2                  // 0000000046bc: d5207c08 000a0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000046c4: bf870193
	v_add3_u32 v9, v4, v97, 0x7fff                             // 0000000046c8: d6550009 03fec304 00007fff
	v_add_co_u32 v4, s2, v5, v2                                // 0000000046d4: d7000204 02020505
	v_or_b32_e32 v10, 0x400000, v97                            // 0000000046dc: 3814c2ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000046e4: bf88f19f
	v_add_co_ci_u32_e64 v5, null, v8, v3, s2                   // 0000000046e8: d5207c05 000a0708
	v_cmp_u_f32_e64 s2, v97, v97                               // 0000000046f0: d4180002 0202c361
	s_wait_alu depctr_va_sdst(0)                               // 0000000046f8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000046fc: bf870001
	v_cndmask_b32_e64 v8, v9, v10, s2                          // 000000004700: d5010008 000a1509
	global_store_d16_hi_b16 v[4:5], v8, off                    // 000000004708: ee09407c 04000000 00000004
	s_wait_alu depctr_sa_sdst(0)                               // 000000004714: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004718: 8c7e047e
	v_mad_co_u64_u32 v[8:9], null, s18, 3, v[32:33]            // 00000000471c: d6fe7c08 04810612
	v_cmp_lt_i64_e64 s2, 3, v[14:15]                           // 000000004724: d4510002 02021c83
	s_and_b32 s4, s2, s3                                       // 00000000472c: 8b040302
	v_mad_co_u64_u32 v[9:10], null, s19, 3, v[9:10]            // 000000004730: d6fe7c09 04250613
	s_delay_alu instid0(valu_dep_1)                            // 000000004738: bf870001
	v_lshlrev_b64_e32 v[4:5], 1, v[8:9]                        // 00000000473c: 3e081081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004740: bf88ff9e
	s_and_saveexec_b32 s5, s4                                  // 000000004744: be852004
	s_cbranch_execz 27                                         // 000000004748: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2cb8>
	v_bfe_u32 v8, v94, 16, 1                                   // 00000000474c: d6100008 0205215e
	v_add_co_u32 v9, s4, s20, v6                               // 000000004754: d7000409 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 00000000475c: bf88f19f
	v_add_co_ci_u32_e64 v10, null, s21, v7, s4                 // 000000004760: d5207c0a 00120e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004768: bf870193
	v_add3_u32 v11, v8, v94, 0x7fff                            // 00000000476c: d655000b 03febd08 00007fff
	v_add_co_u32 v8, s4, v9, v4                                // 000000004778: d7000408 02020909
	v_or_b32_e32 v12, 0x400000, v94                            // 000000004780: 3818bcff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004788: bf88f19f
	v_add_co_ci_u32_e64 v9, null, v10, v5, s4                  // 00000000478c: d5207c09 00120b0a
	v_cmp_u_f32_e64 s4, v94, v94                               // 000000004794: d4180004 0202bd5e
	s_wait_alu depctr_va_sdst(0)                               // 00000000479c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000047a0: bf870001
	v_cndmask_b32_e64 v10, v11, v12, s4                        // 0000000047a4: d501000a 0012190b
	global_store_d16_hi_b16 v[8:9], v10, off                   // 0000000047ac: ee09407c 05000000 00000008
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047b8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000047bc: 8c7e057e
	s_lshl_b64 s[4:5], s[18:19], 2                             // 0000000047c0: 84848212
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047c4: bf88ff9e
	v_add_co_u32 v8, s4, s4, v32                               // 0000000047c8: d7000408 02024004
	s_wait_alu depctr_va_sdst(0)                               // 0000000047d0: bf88f19f
	v_add_co_ci_u32_e64 v9, null, s5, v33, s4                  // 0000000047d4: d5207c09 00124205
	v_cmp_lt_i64_e64 s4, 4, v[14:15]                           // 0000000047dc: d4510004 02021c84
	s_delay_alu instid0(valu_dep_2)                            // 0000000047e4: bf870002
	v_lshlrev_b64_e32 v[8:9], 1, v[8:9]                        // 0000000047e8: 3e101081
	s_and_b32 s5, s4, s3                                       // 0000000047ec: 8b050304
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047f0: bf88ff9e
	s_and_saveexec_b32 s6, s5                                  // 0000000047f4: be862005
	s_cbranch_execz 27                                         // 0000000047f8: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2d68>
	v_bfe_u32 v10, v91, 16, 1                                  // 0000000047fc: d610000a 0205215b
	v_add_co_u32 v11, s5, s20, v6                              // 000000004804: d700050b 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 00000000480c: bf88f19f
	v_add_co_ci_u32_e64 v12, null, s21, v7, s5                 // 000000004810: d5207c0c 00160e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004818: bf870193
	v_add3_u32 v13, v10, v91, 0x7fff                           // 00000000481c: d655000d 03feb70a 00007fff
	v_add_co_u32 v10, s5, v11, v8                              // 000000004828: d700050a 0202110b
	v_or_b32_e32 v16, 0x400000, v91                            // 000000004830: 3820b6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004838: bf88f19f
	v_add_co_ci_u32_e64 v11, null, v12, v9, s5                 // 00000000483c: d5207c0b 0016130c
	v_cmp_u_f32_e64 s5, v91, v91                               // 000000004844: d4180005 0202b75b
	s_wait_alu depctr_va_sdst(0)                               // 00000000484c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004850: bf870001
	v_cndmask_b32_e64 v12, v13, v16, s5                        // 000000004854: d501000c 0016210d
	global_store_d16_hi_b16 v[10:11], v12, off                 // 00000000485c: ee09407c 06000000 0000000a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004868: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s6                              // 00000000486c: 8c7e067e
	v_mad_co_u64_u32 v[10:11], null, s18, 5, v[32:33]          // 000000004870: d6fe7c0a 04810a12
	v_cmp_lt_i64_e64 s5, 5, v[14:15]                           // 000000004878: d4510005 02021c85
	s_and_b32 s6, s5, s3                                       // 000000004880: 8b060305
	v_mad_co_u64_u32 v[11:12], null, s19, 5, v[11:12]          // 000000004884: d6fe7c0b 042d0a13
	s_delay_alu instid0(valu_dep_1)                            // 00000000488c: bf870001
	v_lshlrev_b64_e32 v[10:11], 1, v[10:11]                    // 000000004890: 3e141481
	s_wait_alu depctr_sa_sdst(0)                               // 000000004894: bf88ff9e
	s_and_saveexec_b32 s7, s6                                  // 000000004898: be872006
	s_cbranch_execz 27                                         // 00000000489c: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2e0c>
	v_bfe_u32 v12, v87, 16, 1                                  // 0000000048a0: d610000c 02052157
	v_add_co_u32 v13, s6, s20, v6                              // 0000000048a8: d700060d 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000048b0: bf88f19f
	v_add_co_ci_u32_e64 v16, null, s21, v7, s6                 // 0000000048b4: d5207c10 001a0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000048bc: bf870193
	v_add3_u32 v17, v12, v87, 0x7fff                           // 0000000048c0: d6550011 03feaf0c 00007fff
	v_add_co_u32 v12, s6, v13, v10                             // 0000000048cc: d700060c 0202150d
	v_or_b32_e32 v18, 0x400000, v87                            // 0000000048d4: 3824aeff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 0000000048dc: bf88f19f
	v_add_co_ci_u32_e64 v13, null, v16, v11, s6                // 0000000048e0: d5207c0d 001a1710
	v_cmp_u_f32_e64 s6, v87, v87                               // 0000000048e8: d4180006 0202af57
	s_wait_alu depctr_va_sdst(0)                               // 0000000048f0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 0000000048f4: bf870001
	v_cndmask_b32_e64 v16, v17, v18, s6                        // 0000000048f8: d5010010 001a2511
	global_store_d16_hi_b16 v[12:13], v16, off                 // 000000004900: ee09407c 08000000 0000000c
	s_wait_alu depctr_sa_sdst(0)                               // 00000000490c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s7                              // 000000004910: 8c7e077e
	v_mad_co_u64_u32 v[16:17], null, s18, 6, v[32:33]          // 000000004914: d6fe7c10 04810c12
	v_cmp_lt_i64_e64 s6, 6, v[14:15]                           // 00000000491c: d4510006 02021c86
	s_and_b32 s7, s6, s3                                       // 000000004924: 8b070306
	v_mad_co_u64_u32 v[17:18], null, s19, 6, v[17:18]          // 000000004928: d6fe7c11 04450c13
	s_delay_alu instid0(valu_dep_1)                            // 000000004930: bf870001
	v_lshlrev_b64_e32 v[12:13], 1, v[16:17]                    // 000000004934: 3e182081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004938: bf88ff9e
	s_and_saveexec_b32 s8, s7                                  // 00000000493c: be882007
	s_cbranch_execz 27                                         // 000000004940: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2eb0>
	v_bfe_u32 v16, v83, 16, 1                                  // 000000004944: d6100010 02052153
	v_add_co_u32 v17, s7, s20, v6                              // 00000000494c: d7000711 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 000000004954: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v7, s7                 // 000000004958: d5207c12 001e0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004960: bf870193
	v_add3_u32 v19, v16, v83, 0x7fff                           // 000000004964: d6550013 03fea710 00007fff
	v_add_co_u32 v16, s7, v17, v12                             // 000000004970: d7000710 02021911
	v_or_b32_e32 v20, 0x400000, v83                            // 000000004978: 3828a6ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004980: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v13, s7                // 000000004984: d5207c11 001e1b12
	v_cmp_u_f32_e64 s7, v83, v83                               // 00000000498c: d4180007 0202a753
	s_wait_alu depctr_va_sdst(0)                               // 000000004994: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004998: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s7                        // 00000000499c: d5010012 001e2913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 0000000049a4: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s8                              // 0000000049b4: 8c7e087e
	v_mad_co_u64_u32 v[16:17], null, s18, 7, v[32:33]          // 0000000049b8: d6fe7c10 04810e12
	v_cmp_lt_i64_e64 s7, 7, v[14:15]                           // 0000000049c0: d4510007 02021c87
	s_and_b32 s8, s7, s3                                       // 0000000049c8: 8b080307
	v_mad_co_u64_u32 v[17:18], null, s19, 7, v[17:18]          // 0000000049cc: d6fe7c11 04450e13
	s_delay_alu instid0(valu_dep_1)                            // 0000000049d4: bf870001
	v_lshlrev_b64_e32 v[14:15], 1, v[16:17]                    // 0000000049d8: 3e1c2081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000049dc: bf88ff9e
	s_and_saveexec_b32 s9, s8                                  // 0000000049e0: be892008
	s_cbranch_execz 27                                         // 0000000049e4: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x2f54>
	v_bfe_u32 v16, v80, 16, 1                                  // 0000000049e8: d6100010 02052150
	v_add_co_u32 v17, s8, s20, v6                              // 0000000049f0: d7000811 02020c14
	s_wait_alu depctr_va_sdst(0)                               // 0000000049f8: bf88f19f
	v_add_co_ci_u32_e64 v18, null, s21, v7, s8                 // 0000000049fc: d5207c12 00220e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004a04: bf870193
	v_add3_u32 v19, v16, v80, 0x7fff                           // 000000004a08: d6550013 03fea110 00007fff
	v_add_co_u32 v16, s8, v17, v14                             // 000000004a14: d7000810 02021d11
	v_or_b32_e32 v20, 0x400000, v80                            // 000000004a1c: 3828a0ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004a24: bf88f19f
	v_add_co_ci_u32_e64 v17, null, v18, v15, s8                // 000000004a28: d5207c11 00221f12
	v_cmp_u_f32_e64 s8, v80, v80                               // 000000004a30: d4180008 0202a150
	s_wait_alu depctr_va_sdst(0)                               // 000000004a38: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004a3c: bf870001
	v_cndmask_b32_e64 v18, v19, v20, s8                        // 000000004a40: d5010012 00222913
	global_store_d16_hi_b16 v[16:17], v18, off                 // 000000004a48: ee09407c 09000000 00000010
	s_wait_alu depctr_sa_sdst(0)                               // 000000004a54: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s9                              // 000000004a58: 8c7e097e
	v_or_b32_e32 v18, s33, v36                                 // 000000004a5c: 38244821
	v_or_b32_e32 v19, s39, v37                                 // 000000004a60: 38264a27
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 000000004a64: bf870112
	v_mul_lo_u32 v20, s19, v18                                 // 000000004a68: d72c0014 02022413
	v_mul_lo_u32 v21, s18, v19                                 // 000000004a70: d72c0015 02022612
	v_mad_co_u64_u32 v[16:17], null, s18, v18, 0               // 000000004a78: d6fe7c10 02022412
	v_sub_co_u32 v18, s8, s16, v18                             // 000000004a80: d7010812 02022410
	s_wait_alu depctr_va_sdst(0)                               // 000000004a88: bf88f19f
	v_sub_co_ci_u32_e64 v19, null, s17, v19, s8                // 000000004a8c: d5217c13 00222611
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_4)// 000000004a94: bf870211
	v_cmp_lt_i64_e64 s8, 0, v[18:19]                           // 000000004a98: d4510008 02022480
	v_add3_u32 v17, v17, v21, v20                              // 000000004aa0: d6550011 04522b11
	s_delay_alu instid0(valu_dep_1)                            // 000000004aa8: bf870001
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000004aac: 3e202081
	s_and_b32 s9, s8, s3                                       // 000000004ab0: 8b090308
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ab4: bf88ff9e
	s_and_saveexec_b32 s10, s9                                 // 000000004ab8: be8a2009
	s_cbranch_execz 28                                         // 000000004abc: bfa5001c <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3030>
	v_lshlrev_b64_e32 v[20:21], 1, v[32:33]                    // 000000004ac0: 3e284081
	v_add_co_u32 v23, s9, s20, v16                             // 000000004ac4: d7000917 02022014
	v_bfe_u32 v22, v79, 16, 1                                  // 000000004acc: d6100016 0205214f
	s_wait_alu depctr_va_sdst(0)                               // 000000004ad4: bf88f19f
	v_add_co_ci_u32_e64 v24, null, s21, v17, s9                // 000000004ad8: d5207c18 00262215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ae0: bf870193
	v_add_co_u32 v20, s9, v23, v20                             // 000000004ae4: d7000914 02022917
	v_add3_u32 v22, v22, v79, 0x7fff                           // 000000004aec: d6550016 03fe9f16 00007fff
	v_or_b32_e32 v25, 0x400000, v79                            // 000000004af8: 38329eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b00: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v24, v21, s9                // 000000004b04: d5207c15 00262b18
	v_cmp_u_f32_e64 s9, v79, v79                               // 000000004b0c: d4180009 02029f4f
	s_wait_alu depctr_va_sdst(0)                               // 000000004b14: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004b18: bf870001
	v_cndmask_b32_e64 v22, v22, v25, s9                        // 000000004b1c: d5010016 00263316
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004b24: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b30: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s10                             // 000000004b34: 8c7e0a7e
	v_cmp_lt_i64_e64 s9, 1, v[18:19]                           // 000000004b38: d4510009 02022481
	s_and_b32 s10, s9, s3                                      // 000000004b40: 8b0a0309
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b44: bf88ff9e
	s_and_saveexec_b32 s11, s10                                // 000000004b48: be8b200a
	s_cbranch_execz 27                                         // 000000004b4c: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x30bc>
	v_bfe_u32 v20, v78, 16, 1                                  // 000000004b50: d6100014 0205214e
	v_add_co_u32 v21, s10, s20, v16                            // 000000004b58: d7000a15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004b60: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s10               // 000000004b64: d5207c16 002a2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004b6c: bf870193
	v_add3_u32 v23, v20, v78, 0x7fff                           // 000000004b70: d6550017 03fe9d14 00007fff
	v_add_co_u32 v20, s10, v21, v0                             // 000000004b7c: d7000a14 02020115
	v_or_b32_e32 v24, 0x400000, v78                            // 000000004b84: 38309cff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004b8c: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v1, s10                // 000000004b90: d5207c15 002a0316
	v_cmp_u_f32_e64 s10, v78, v78                              // 000000004b98: d418000a 02029d4e
	s_wait_alu depctr_va_sdst(0)                               // 000000004ba0: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ba4: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s10                       // 000000004ba8: d5010016 002a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004bb0: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bbc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s11                             // 000000004bc0: 8c7e0b7e
	v_cmp_lt_i64_e64 s10, 2, v[18:19]                          // 000000004bc4: d451000a 02022482
	s_and_b32 s11, s10, s3                                     // 000000004bcc: 8b0b030a
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bd0: bf88ff9e
	s_and_saveexec_b32 s12, s11                                // 000000004bd4: be8c200b
	s_cbranch_execz 27                                         // 000000004bd8: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3148>
	v_bfe_u32 v20, v77, 16, 1                                  // 000000004bdc: d6100014 0205214d
	v_add_co_u32 v21, s11, s20, v16                            // 000000004be4: d7000b15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004bec: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s11               // 000000004bf0: d5207c16 002e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004bf8: bf870193
	v_add3_u32 v23, v20, v77, 0x7fff                           // 000000004bfc: d6550017 03fe9b14 00007fff
	v_add_co_u32 v20, s11, v21, v2                             // 000000004c08: d7000b14 02020515
	v_or_b32_e32 v24, 0x400000, v77                            // 000000004c10: 38309aff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004c18: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v3, s11                // 000000004c1c: d5207c15 002e0716
	v_cmp_u_f32_e64 s11, v77, v77                              // 000000004c24: d418000b 02029b4d
	s_wait_alu depctr_va_sdst(0)                               // 000000004c2c: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004c30: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s11                       // 000000004c34: d5010016 002e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004c3c: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c48: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s12                             // 000000004c4c: 8c7e0c7e
	v_cmp_lt_i64_e64 s11, 3, v[18:19]                          // 000000004c50: d451000b 02022483
	s_and_b32 s12, s11, s3                                     // 000000004c58: 8b0c030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c5c: bf88ff9e
	s_and_saveexec_b32 s13, s12                                // 000000004c60: be8d200c
	s_cbranch_execz 27                                         // 000000004c64: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x31d4>
	v_bfe_u32 v20, v75, 16, 1                                  // 000000004c68: d6100014 0205214b
	v_add_co_u32 v21, s12, s20, v16                            // 000000004c70: d7000c15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004c78: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s12               // 000000004c7c: d5207c16 00322215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004c84: bf870193
	v_add3_u32 v23, v20, v75, 0x7fff                           // 000000004c88: d6550017 03fe9714 00007fff
	v_add_co_u32 v20, s12, v21, v4                             // 000000004c94: d7000c14 02020915
	v_or_b32_e32 v24, 0x400000, v75                            // 000000004c9c: 383096ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004ca4: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v5, s12                // 000000004ca8: d5207c15 00320b16
	v_cmp_u_f32_e64 s12, v75, v75                              // 000000004cb0: d418000c 0202974b
	s_wait_alu depctr_va_sdst(0)                               // 000000004cb8: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004cbc: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s12                       // 000000004cc0: d5010016 00323117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004cc8: ee09407c 0b000000 00000014
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cd4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s13                             // 000000004cd8: 8c7e0d7e
	v_cmp_lt_i64_e64 s12, 4, v[18:19]                          // 000000004cdc: d451000c 02022484
	s_and_b32 s13, s12, s3                                     // 000000004ce4: 8b0d030c
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ce8: bf88ff9e
	s_and_saveexec_b32 s14, s13                                // 000000004cec: be8e200d
	s_cbranch_execz 27                                         // 000000004cf0: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3260>
	v_bfe_u32 v20, v74, 16, 1                                  // 000000004cf4: d6100014 0205214a
	v_add_co_u32 v21, s13, s20, v16                            // 000000004cfc: d7000d15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004d04: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s13               // 000000004d08: d5207c16 00362215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004d10: bf870193
	v_add3_u32 v23, v20, v74, 0x7fff                           // 000000004d14: d6550017 03fe9514 00007fff
	v_add_co_u32 v20, s13, v21, v8                             // 000000004d20: d7000d14 02021115
	v_or_b32_e32 v24, 0x400000, v74                            // 000000004d28: 383094ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004d30: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v9, s13                // 000000004d34: d5207c15 00361316
	v_cmp_u_f32_e64 s13, v74, v74                              // 000000004d3c: d418000d 0202954a
	s_wait_alu depctr_va_sdst(0)                               // 000000004d44: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004d48: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s13                       // 000000004d4c: d5010016 00363117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004d54: ee09407c 0b000000 00000014
	s_or_b32 exec_lo, exec_lo, s14                             // 000000004d60: 8c7e0e7e
	v_cmp_lt_i64_e64 s13, 5, v[18:19]                          // 000000004d64: d451000d 02022485
	s_and_b32 s14, s13, s3                                     // 000000004d6c: 8b0e030d
	s_delay_alu instid0(salu_cycle_1)                          // 000000004d70: bf870009
	s_and_saveexec_b32 s15, s14                                // 000000004d74: be8f200e
	s_cbranch_execz 27                                         // 000000004d78: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x32e8>
	v_bfe_u32 v20, v73, 16, 1                                  // 000000004d7c: d6100014 02052149
	v_add_co_u32 v21, s14, s20, v16                            // 000000004d84: d7000e15 02022014
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_3)// 000000004d8c: bf870191
	v_add_co_ci_u32_e64 v22, null, s21, v17, s14               // 000000004d90: d5207c16 003a2215
	v_add3_u32 v23, v20, v73, 0x7fff                           // 000000004d98: d6550017 03fe9314 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004da4: bf870003
	v_add_co_u32 v20, s14, v21, v10                            // 000000004da8: d7000e14 02021515
	v_or_b32_e32 v24, 0x400000, v73                            // 000000004db0: 383092ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004db8: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v11, s14               // 000000004dbc: d5207c15 003a1716
	v_cmp_u_f32_e64 s14, v73, v73                              // 000000004dc4: d418000e 02029349
	s_wait_alu depctr_va_sdst(0)                               // 000000004dcc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004dd0: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s14                       // 000000004dd4: d5010016 003a3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004ddc: ee09407c 0b000000 00000014
	s_or_b32 exec_lo, exec_lo, s15                             // 000000004de8: 8c7e0f7e
	v_cmp_lt_i64_e64 s14, 6, v[18:19]                          // 000000004dec: d451000e 02022486
	s_and_b32 s15, s14, s3                                     // 000000004df4: 8b0f030e
	s_wait_alu depctr_sa_sdst(0)                               // 000000004df8: bf88ff9e
	s_and_saveexec_b32 s40, s15                                // 000000004dfc: bea8200f
	s_cbranch_execz 27                                         // 000000004e00: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3370>
	v_bfe_u32 v20, v72, 16, 1                                  // 000000004e04: d6100014 02052148
	v_add_co_u32 v21, s15, s20, v16                            // 000000004e0c: d7000f15 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004e14: bf88f19f
	v_add_co_ci_u32_e64 v22, null, s21, v17, s15               // 000000004e18: d5207c16 003e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004e20: bf870193
	v_add3_u32 v23, v20, v72, 0x7fff                           // 000000004e24: d6550017 03fe9114 00007fff
	v_add_co_u32 v20, s15, v21, v12                            // 000000004e30: d7000f14 02021915
	v_or_b32_e32 v24, 0x400000, v72                            // 000000004e38: 383090ff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004e40: bf88f19f
	v_add_co_ci_u32_e64 v21, null, v22, v13, s15               // 000000004e44: d5207c15 003e1b16
	v_cmp_u_f32_e64 s15, v72, v72                              // 000000004e4c: d418000f 02029148
	s_wait_alu depctr_va_sdst(0)                               // 000000004e54: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004e58: bf870001
	v_cndmask_b32_e64 v22, v23, v24, s15                       // 000000004e5c: d5010016 003e3117
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000004e64: ee09407c 0b000000 00000014
	s_or_b32 exec_lo, exec_lo, s40                             // 000000004e70: 8c7e287e
	v_cmp_lt_i64_e64 s15, 7, v[18:19]                          // 000000004e74: d451000f 02022487
	s_and_b32 s3, s15, s3                                      // 000000004e7c: 8b03030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000004e80: bf88ff9e
	s_and_saveexec_b32 s40, s3                                 // 000000004e84: bea82003
	s_cbranch_execz 27                                         // 000000004e88: bfa5001b <tessera_rocm_scaled_matmul_28d379a9237322d1+0x33f8>
	v_bfe_u32 v18, v71, 16, 1                                  // 000000004e8c: d6100012 02052147
	v_add_co_u32 v19, s3, s20, v16                             // 000000004e94: d7000313 02022014
	s_wait_alu depctr_va_sdst(0)                               // 000000004e9c: bf88f19f
	v_add_co_ci_u32_e64 v20, null, s21, v17, s3                // 000000004ea0: d5207c14 000e2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004ea8: bf870193
	v_add3_u32 v21, v18, v71, 0x7fff                           // 000000004eac: d6550015 03fe8f12 00007fff
	v_add_co_u32 v18, s3, v19, v14                             // 000000004eb8: d7000312 02021d13
	v_or_b32_e32 v22, 0x400000, v71                            // 000000004ec0: 382c8eff 00400000
	s_wait_alu depctr_va_sdst(0)                               // 000000004ec8: bf88f19f
	v_add_co_ci_u32_e64 v19, null, v20, v15, s3                // 000000004ecc: d5207c13 000e1f14
	v_cmp_u_f32_e64 s3, v71, v71                               // 000000004ed4: d4180003 02028f47
	s_wait_alu depctr_va_sdst(0)                               // 000000004edc: bf88f19f
	s_delay_alu instid0(valu_dep_1)                            // 000000004ee0: bf870001
	v_cndmask_b32_e64 v20, v21, v22, s3                        // 000000004ee4: d5010014 000e2d15
	global_store_d16_hi_b16 v[18:19], v20, off                 // 000000004eec: ee09407c 0a000000 00000012
	s_or_b32 exec_lo, exec_lo, s40                             // 000000004ef8: 8c7e287e
	v_cmp_gt_i64_e64 s3, s[18:19], v[34:35]                    // 000000004efc: d4540003 02024412
	s_and_b32 s41, vcc_lo, s3                                  // 000000004f04: 8b29036a
	s_delay_alu instid0(salu_cycle_1)                          // 000000004f08: bf870009
	s_and_saveexec_b32 s40, s41                                // 000000004f0c: bea82029
	s_cbranch_execz 25                                         // 000000004f10: bfa50019 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3478>
	v_lshlrev_b64_e32 v[18:19], 1, v[32:33]                    // 000000004f14: 3e244081
	v_add_co_u32 v21, vcc_lo, s20, v6                          // 000000004f18: d7006a15 02020c14
	v_bfe_u32 v20, v70, 16, 1                                  // 000000004f20: d6100014 02052146
	s_wait_alu depctr_va_vcc(0)                                // 000000004f28: bf88ff9d
	v_add_co_ci_u32_e64 v22, null, s21, v7, vcc_lo             // 000000004f2c: d5207c16 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004f34: bf870193
	v_add_co_u32 v18, vcc_lo, v21, v18                         // 000000004f38: d7006a12 02022515
	v_add3_u32 v20, v20, v70, 0x7fff                           // 000000004f40: d6550014 03fe8d14 00007fff
	v_or_b32_e32 v23, 0x400000, v70                            // 000000004f4c: 382e8cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004f54: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v22, v19, vcc_lo            // 000000004f58: d5207c13 01aa2716
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000004f60: 7c308d46
	s_wait_alu depctr_va_vcc(0)                                // 000000004f64: bf88ff9d
	v_cndmask_b32_e32 v20, v20, v23, vcc_lo                    // 000000004f68: 02282f14
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004f6c: ee09407c 0a000000 00002012
	s_or_b32 exec_lo, exec_lo, s40                             // 000000004f78: 8c7e287e
	s_and_b32 s40, s0, s3                                      // 000000004f7c: 8b280300
	s_delay_alu instid0(salu_cycle_1)                          // 000000004f80: bf870009
	s_and_saveexec_b32 s0, s40                                 // 000000004f84: be802028
	s_cbranch_execz 24                                         // 000000004f88: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x34ec>
	v_bfe_u32 v18, v69, 16, 1                                  // 000000004f8c: d6100012 02052145
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000004f94: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000004f9c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000004fa0: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004fa8: bf870193
	v_add3_u32 v21, v18, v69, 0x7fff                           // 000000004fac: d6550015 03fe8b12 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v0                          // 000000004fb8: d7006a12 02020113
	v_or_b32_e32 v22, 0x400000, v69                            // 000000004fc0: 382c8aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004fc8: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v1, vcc_lo             // 000000004fcc: d5207c13 01aa0314
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000004fd4: 7c308b45
	s_wait_alu depctr_va_vcc(0)                                // 000000004fd8: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000004fdc: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000004fe0: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004ff0: 8c7e007e
	s_and_b32 s1, s1, s3                                       // 000000004ff4: 8b010301
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ff8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004ffc: be802001
	s_cbranch_execz 24                                         // 000000005000: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3564>
	v_bfe_u32 v18, v68, 16, 1                                  // 000000005004: d6100012 02052144
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 00000000500c: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000005014: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000005018: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005020: bf870193
	v_add3_u32 v21, v18, v68, 0x7fff                           // 000000005024: d6550015 03fe8912 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v2                          // 000000005030: d7006a12 02020513
	v_or_b32_e32 v22, 0x400000, v68                            // 000000005038: 382c88ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005040: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v3, vcc_lo             // 000000005044: d5207c13 01aa0714
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 00000000504c: 7c308944
	s_wait_alu depctr_va_vcc(0)                                // 000000005050: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000005054: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000005058: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000005064: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005068: 8c7e007e
	s_and_b32 s1, s2, s3                                       // 00000000506c: 8b010302
	s_wait_alu depctr_sa_sdst(0)                               // 000000005070: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005074: be802001
	s_cbranch_execz 24                                         // 000000005078: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x35dc>
	v_bfe_u32 v18, v67, 16, 1                                  // 00000000507c: d6100012 02052143
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000005084: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 00000000508c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000005090: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005098: bf870193
	v_add3_u32 v21, v18, v67, 0x7fff                           // 00000000509c: d6550015 03fe8712 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v4                          // 0000000050a8: d7006a12 02020913
	v_or_b32_e32 v22, 0x400000, v67                            // 0000000050b0: 382c86ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000050b8: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v5, vcc_lo             // 0000000050bc: d5207c13 01aa0b14
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 0000000050c4: 7c308743
	s_wait_alu depctr_va_vcc(0)                                // 0000000050c8: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 0000000050cc: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 0000000050d0: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050dc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050e0: 8c7e007e
	s_and_b32 s1, s4, s3                                       // 0000000050e4: 8b010304
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050e8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000050ec: be802001
	s_cbranch_execz 24                                         // 0000000050f0: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3654>
	v_bfe_u32 v18, v66, 16, 1                                  // 0000000050f4: d6100012 02052142
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 0000000050fc: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 000000005104: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000005108: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005110: bf870193
	v_add3_u32 v21, v18, v66, 0x7fff                           // 000000005114: d6550015 03fe8512 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v8                          // 000000005120: d7006a12 02021113
	v_or_b32_e32 v22, 0x400000, v66                            // 000000005128: 382c84ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005130: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v9, vcc_lo             // 000000005134: d5207c13 01aa1314
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 00000000513c: 7c308542
	s_wait_alu depctr_va_vcc(0)                                // 000000005140: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000005144: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000005148: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000005154: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005158: 8c7e007e
	s_and_b32 s1, s5, s3                                       // 00000000515c: 8b010305
	s_wait_alu depctr_sa_sdst(0)                               // 000000005160: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005164: be802001
	s_cbranch_execz 24                                         // 000000005168: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x36cc>
	v_bfe_u32 v18, v65, 16, 1                                  // 00000000516c: d6100012 02052141
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 000000005174: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 00000000517c: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 000000005180: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005188: bf870193
	v_add3_u32 v21, v18, v65, 0x7fff                           // 00000000518c: d6550015 03fe8312 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v10                         // 000000005198: d7006a12 02021513
	v_or_b32_e32 v22, 0x400000, v65                            // 0000000051a0: 382c82ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000051a8: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v11, vcc_lo            // 0000000051ac: d5207c13 01aa1714
	v_cmp_u_f32_e32 vcc_lo, v65, v65                           // 0000000051b4: 7c308341
	s_wait_alu depctr_va_vcc(0)                                // 0000000051b8: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 0000000051bc: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 0000000051c0: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051cc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000051d0: 8c7e007e
	s_and_b32 s1, s6, s3                                       // 0000000051d4: 8b010306
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051d8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000051dc: be802001
	s_cbranch_execz 24                                         // 0000000051e0: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3744>
	v_bfe_u32 v18, v64, 16, 1                                  // 0000000051e4: d6100012 02052140
	v_add_co_u32 v19, vcc_lo, s20, v6                          // 0000000051ec: d7006a13 02020c14
	s_wait_alu depctr_va_vcc(0)                                // 0000000051f4: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v7, vcc_lo             // 0000000051f8: d5207c14 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005200: bf870193
	v_add3_u32 v21, v18, v64, 0x7fff                           // 000000005204: d6550015 03fe8112 00007fff
	v_add_co_u32 v18, vcc_lo, v19, v12                         // 000000005210: d7006a12 02021913
	v_or_b32_e32 v22, 0x400000, v64                            // 000000005218: 382c80ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005220: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v20, v13, vcc_lo            // 000000005224: d5207c13 01aa1b14
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 00000000522c: 7c308140
	s_wait_alu depctr_va_vcc(0)                                // 000000005230: bf88ff9d
	v_cndmask_b32_e32 v20, v21, v22, vcc_lo                    // 000000005234: 02282d15
	global_store_d16_hi_b16 v[18:19], v20, off offset:32       // 000000005238: ee09407c 0a000000 00002012
	s_wait_alu depctr_sa_sdst(0)                               // 000000005244: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005248: 8c7e007e
	s_and_b32 s1, s7, s3                                       // 00000000524c: 8b010307
	s_wait_alu depctr_sa_sdst(0)                               // 000000005250: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005254: be802001
	s_cbranch_execz 24                                         // 000000005258: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x37bc>
	v_add_co_u32 v6, vcc_lo, s20, v6                           // 00000000525c: d7006a06 02020c14
	v_bfe_u32 v18, v62, 16, 1                                  // 000000005264: d6100012 0205213e
	s_wait_alu depctr_va_vcc(0)                                // 00000000526c: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo              // 000000005270: d5207c07 01aa0e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000005278: bf870193
	v_add_co_u32 v6, vcc_lo, v6, v14                           // 00000000527c: d7006a06 02021d06
	v_add3_u32 v18, v18, v62, 0x7fff                           // 000000005284: d6550012 03fe7d12 00007fff
	v_or_b32_e32 v19, 0x400000, v62                            // 000000005290: 38267cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005298: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo              // 00000000529c: d5207c07 01aa1f07
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 0000000052a4: 7c307d3e
	s_wait_alu depctr_va_vcc(0)                                // 0000000052a8: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo                    // 0000000052ac: 02242712
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 0000000052b0: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000052c0: 8c7e007e
	s_and_b32 s1, s8, s3                                       // 0000000052c4: 8b010308
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052c8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000052cc: be802001
	s_cbranch_execz 25                                         // 0000000052d0: bfa50019 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3838>
	v_lshlrev_b64_e32 v[6:7], 1, v[32:33]                      // 0000000052d4: 3e0c4081
	v_add_co_u32 v19, vcc_lo, s20, v16                         // 0000000052d8: d7006a13 02022014
	v_bfe_u32 v18, v63, 16, 1                                  // 0000000052e0: d6100012 0205213f
	s_wait_alu depctr_va_vcc(0)                                // 0000000052e8: bf88ff9d
	v_add_co_ci_u32_e64 v20, null, s21, v17, vcc_lo            // 0000000052ec: d5207c14 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000052f4: bf870193
	v_add_co_u32 v6, vcc_lo, v19, v6                           // 0000000052f8: d7006a06 02020d13
	v_add3_u32 v18, v18, v63, 0x7fff                           // 000000005300: d6550012 03fe7f12 00007fff
	v_or_b32_e32 v21, 0x400000, v63                            // 00000000530c: 382a7eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005314: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v20, v7, vcc_lo              // 000000005318: d5207c07 01aa0f14
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000005320: 7c307f3f
	s_wait_alu depctr_va_vcc(0)                                // 000000005324: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo                    // 000000005328: 02242b12
	global_store_d16_hi_b16 v[6:7], v18, off offset:32         // 00000000532c: ee09407c 09000000 00002006
	s_wait_alu depctr_sa_sdst(0)                               // 000000005338: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000533c: 8c7e007e
	s_and_b32 s1, s9, s3                                       // 000000005340: 8b010309
	s_wait_alu depctr_sa_sdst(0)                               // 000000005344: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005348: be802001
	s_cbranch_execz 24                                         // 00000000534c: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x38b0>
	v_add_co_u32 v7, vcc_lo, s20, v16                          // 000000005350: d7006a07 02022014
	v_bfe_u32 v6, v61, 16, 1                                   // 000000005358: d6100006 0205213d
	s_wait_alu depctr_va_vcc(0)                                // 000000005360: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s21, v17, vcc_lo            // 000000005364: d5207c12 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000536c: bf870193
	v_add_co_u32 v0, vcc_lo, v7, v0                            // 000000005370: d7006a00 02020107
	v_add3_u32 v6, v6, v61, 0x7fff                             // 000000005378: d6550006 03fe7b06 00007fff
	v_or_b32_e32 v19, 0x400000, v61                            // 000000005384: 38267aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000538c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v18, v1, vcc_lo              // 000000005390: d5207c01 01aa0312
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000005398: 7c307b3d
	s_wait_alu depctr_va_vcc(0)                                // 00000000539c: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v19, vcc_lo                      // 0000000053a0: 020c2706
	global_store_d16_hi_b16 v[0:1], v6, off offset:32          // 0000000053a4: ee09407c 03000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053b0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000053b4: 8c7e007e
	s_and_b32 s1, s10, s3                                      // 0000000053b8: 8b01030a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053bc: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000053c0: be802001
	s_cbranch_execz 24                                         // 0000000053c4: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3928>
	v_bfe_u32 v0, v60, 16, 1                                   // 0000000053c8: d6100000 0205213c
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 0000000053d0: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000053d8: bf88ff9d
	v_add_co_ci_u32_e64 v6, null, s21, v17, vcc_lo             // 0000000053dc: d5207c06 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000053e4: bf870193
	v_add3_u32 v7, v0, v60, 0x7fff                             // 0000000053e8: d6550007 03fe7900 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v2                            // 0000000053f4: d7006a00 02020501
	v_or_b32_e32 v18, 0x400000, v60                            // 0000000053fc: 382478ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000005404: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v6, v3, vcc_lo               // 000000005408: d5207c01 01aa0706
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 000000005410: 7c30793c
	s_wait_alu depctr_va_vcc(0)                                // 000000005414: bf88ff9d
	v_cndmask_b32_e32 v2, v7, v18, vcc_lo                      // 000000005418: 02042507
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000541c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005428: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000542c: 8c7e007e
	s_and_b32 s1, s11, s3                                      // 000000005430: 8b01030b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005434: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005438: be802001
	s_cbranch_execz 24                                         // 00000000543c: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x39a0>
	v_bfe_u32 v0, v59, 16, 1                                   // 000000005440: d6100000 0205213b
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005448: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005450: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 000000005454: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000545c: bf870193
	v_add3_u32 v3, v0, v59, 0x7fff                             // 000000005460: d6550003 03fe7700 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v4                            // 00000000546c: d7006a00 02020901
	v_or_b32_e32 v6, 0x400000, v59                             // 000000005474: 380c76ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000547c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v5, vcc_lo               // 000000005480: d5207c01 01aa0b02
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000005488: 7c30773b
	s_wait_alu depctr_va_vcc(0)                                // 00000000548c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v6, vcc_lo                       // 000000005490: 02040d03
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005494: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054a0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000054a4: 8c7e007e
	s_and_b32 s1, s12, s3                                      // 0000000054a8: 8b01030c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054ac: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000054b0: be802001
	s_cbranch_execz 24                                         // 0000000054b4: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3a18>
	v_bfe_u32 v0, v58, 16, 1                                   // 0000000054b8: d6100000 0205213a
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 0000000054c0: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000054c8: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 0000000054cc: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000054d4: bf870193
	v_add3_u32 v3, v0, v58, 0x7fff                             // 0000000054d8: d6550003 03fe7500 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v8                            // 0000000054e4: d7006a00 02021101
	v_or_b32_e32 v4, 0x400000, v58                             // 0000000054ec: 380874ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000054f4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v9, vcc_lo               // 0000000054f8: d5207c01 01aa1302
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000005500: 7c30753a
	s_wait_alu depctr_va_vcc(0)                                // 000000005504: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005508: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 00000000550c: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005518: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000551c: 8c7e007e
	s_and_b32 s1, s13, s3                                      // 000000005520: 8b01030d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005524: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005528: be802001
	s_cbranch_execz 24                                         // 00000000552c: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3a90>
	v_bfe_u32 v0, v57, 16, 1                                   // 000000005530: d6100000 02052139
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005538: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005540: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 000000005544: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000554c: bf870193
	v_add3_u32 v3, v0, v57, 0x7fff                             // 000000005550: d6550003 03fe7300 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v10                           // 00000000555c: d7006a00 02021501
	v_or_b32_e32 v4, 0x400000, v57                             // 000000005564: 380872ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000556c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v11, vcc_lo              // 000000005570: d5207c01 01aa1702
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000005578: 7c307339
	s_wait_alu depctr_va_vcc(0)                                // 00000000557c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005580: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005584: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005590: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005594: 8c7e007e
	s_and_b32 s1, s14, s3                                      // 000000005598: 8b01030e
	s_wait_alu depctr_sa_sdst(0)                               // 00000000559c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000055a0: be802001
	s_cbranch_execz 24                                         // 0000000055a4: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3b08>
	v_bfe_u32 v0, v56, 16, 1                                   // 0000000055a8: d6100000 02052138
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 0000000055b0: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 0000000055b8: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 0000000055bc: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000055c4: bf870193
	v_add3_u32 v3, v0, v56, 0x7fff                             // 0000000055c8: d6550003 03fe7100 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v12                           // 0000000055d4: d7006a00 02021901
	v_or_b32_e32 v4, 0x400000, v56                             // 0000000055dc: 380870ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000055e4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v13, vcc_lo              // 0000000055e8: d5207c01 01aa1b02
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 0000000055f0: 7c307138
	s_wait_alu depctr_va_vcc(0)                                // 0000000055f4: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 0000000055f8: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 0000000055fc: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005608: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000560c: 8c7e007e
	s_and_b32 s1, s15, s3                                      // 000000005610: 8b01030f
	s_wait_alu depctr_sa_sdst(0)                               // 000000005614: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005618: be802001
	s_cbranch_execz 24                                         // 00000000561c: bfa50018 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3b80>
	v_bfe_u32 v0, v55, 16, 1                                   // 000000005620: d6100000 02052137
	v_add_co_u32 v1, vcc_lo, s20, v16                          // 000000005628: d7006a01 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000005630: bf88ff9d
	v_add_co_ci_u32_e64 v2, null, s21, v17, vcc_lo             // 000000005634: d5207c02 01aa2215
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000563c: bf870193
	v_add3_u32 v3, v0, v55, 0x7fff                             // 000000005640: d6550003 03fe6f00 00007fff
	v_add_co_u32 v0, vcc_lo, v1, v14                           // 00000000564c: d7006a00 02021d01
	v_or_b32_e32 v4, 0x400000, v55                             // 000000005654: 38086eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000565c: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v2, v15, vcc_lo              // 000000005660: d5207c01 01aa1f02
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000005668: 7c306f37
	s_wait_alu depctr_va_vcc(0)                                // 00000000566c: bf88ff9d
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo                       // 000000005670: 02040903
	global_store_d16_hi_b16 v[0:1], v2, off offset:32          // 000000005674: ee09407c 01000000 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005680: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005684: 8c7e007e
	s_mov_b32 s0, 0                                            // 000000005688: be800080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000568c: bf88ff9e
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000005690: 8b6a007e
	s_wait_alu depctr_sa_sdst(0)                               // 000000005694: bf88ff9e
	s_cbranch_vccz 48                                          // 000000005698: bfa30030 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3c5c>
	s_and_b32 s0, s48, exec_lo                                 // 00000000569c: 8b007e30
	s_cselect_b32 s0, 1, 0                                     // 0000000056a0: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056a4: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000056a8: bf078100
	s_cbranch_scc1 46                                          // 0000000056ac: bfa2002e <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3c68>
	v_dual_mov_b32 v49, s39 :: v_dual_lshlrev_b32 v0, 3, v54   // 0000000056b0: ca220027 31006c83
	v_mov_b32_e32 v51, s39                                     // 0000000056b8: 7e660227
	v_mov_b32_e32 v47, s39                                     // 0000000056bc: 7e5e0227
	v_mov_b32_e32 v45, s39                                     // 0000000056c0: 7e5a0227
	s_delay_alu instid0(valu_dep_4)                            // 0000000056c4: bf870004
	v_or_b32_e32 v1, 1, v0                                     // 0000000056c8: 38020081
	v_or_b32_e32 v2, 2, v0                                     // 0000000056cc: 38040082
	v_or_b32_e32 v3, 3, v0                                     // 0000000056d0: 38060083
	v_or_b32_e32 v4, 4, v0                                     // 0000000056d4: 38080084
	v_or_b32_e32 v5, 5, v0                                     // 0000000056d8: 380a0085
	v_or_b32_e32 v6, 6, v0                                     // 0000000056dc: 380c0086
	v_or_b32_e32 v7, 7, v0                                     // 0000000056e0: 380e0087
	v_or_b32_e32 v48, s38, v0                                  // 0000000056e4: 38600026
	v_or_b32_e32 v50, s38, v1                                  // 0000000056e8: 38640226
	v_or_b32_e32 v46, s38, v2                                  // 0000000056ec: 385c0426
	v_or_b32_e32 v44, s38, v3                                  // 0000000056f0: 38580626
	v_mov_b32_e32 v39, s39                                     // 0000000056f4: 7e4e0227
	v_or_b32_e32 v38, s38, v4                                  // 0000000056f8: 384c0826
	v_mov_b32_e32 v43, s39                                     // 0000000056fc: 7e560227
	v_or_b32_e32 v42, s38, v5                                  // 000000005700: 38540a26
	v_mov_b32_e32 v41, s39                                     // 000000005704: 7e520227
	v_or_b32_e32 v40, s38, v6                                  // 000000005708: 38500c26
	v_mov_b32_e32 v37, s39                                     // 00000000570c: 7e4a0227
	v_or_b32_e32 v36, s38, v7                                  // 000000005710: 38480e26
	v_or_b32_e32 v30, s33, v0                                  // 000000005714: 383c0021
	v_mov_b32_e32 v31, s39                                     // 000000005718: 7e3e0227
	v_mov_b32_e32 v27, s39                                     // 00000000571c: 7e360227
	v_or_b32_e32 v26, s33, v1                                  // 000000005720: 38340221
	v_mov_b32_e32 v25, s39                                     // 000000005724: 7e320227
	v_or_b32_e32 v24, s33, v2                                  // 000000005728: 38300421
	v_mov_b32_e32 v21, s39                                     // 00000000572c: 7e2a0227
	v_or_b32_e32 v20, s33, v3                                  // 000000005730: 38280621
	v_mov_b32_e32 v29, s39                                     // 000000005734: 7e3a0227
	v_or_b32_e32 v28, s33, v4                                  // 000000005738: 38380821
	v_mov_b32_e32 v23, s39                                     // 00000000573c: 7e2e0227
	v_or_b32_e32 v22, s33, v5                                  // 000000005740: 382c0a21
	v_mov_b32_e32 v19, s39                                     // 000000005744: 7e260227
	v_or_b32_e32 v18, s33, v6                                  // 000000005748: 38240c21
	v_mov_b32_e32 v17, s39                                     // 00000000574c: 7e220227
	v_or_b32_e32 v16, s33, v7                                  // 000000005750: 38200e21
	s_mov_b32 s0, 0                                            // 000000005754: be800080
	s_branch 4                                                 // 000000005758: bfa00004 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x3c6c>
	s_nop 0                                                    // 00000000575c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000005760: bfb60003
	s_endpgm                                                   // 000000005764: bfb00000
	s_mov_b32 s0, -1                                           // 000000005768: be8000c1
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v109, 0           // 00000000576c: ca100080 6a6c0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000005774: bf88ff9e
	s_and_b32 s0, s0, exec_lo                                  // 000000005778: 8b007e00
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v111, 0           // 00000000577c: ca100080 6c6e0080
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v113, 0           // 000000005784: ca100080 6e700080
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v91, 0            // 00000000578c: ca100080 705a0080
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v95, 0            // 000000005794: ca100080 725e0080
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v97, 0             // 00000000579c: ca100080 58600080
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v99, 0             // 0000000057a4: ca100080 5c620080
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v101, 0            // 0000000057ac: ca100080 5e640080
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v103, 0            // 0000000057b4: ca100080 60660080
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v105, 0            // 0000000057bc: ca100080 62680080
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v107, 0           // 0000000057c4: ca100080 646a0080
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v85, 0            // 0000000057cc: ca100080 66540080
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v87, 0            // 0000000057d4: ca100080 68560080
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v89, 0             // 0000000057dc: ca100080 34580080
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v93, 0             // 0000000057e4: ca100080 545c0080
	v_mov_b32_e32 v86, 0                                       // 0000000057ec: 7eac0280
	v_mov_b32_e32 v90, 0                                       // 0000000057f0: 7eb40280
	s_cselect_b32 s0, 1, 0                                     // 0000000057f4: 98008081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057f8: bf88ff9e
	s_cmp_lg_u32 s0, 1                                         // 0000000057fc: bf078100
	s_cbranch_scc1 910                                         // 000000005800: bfa2038e <tessera_rocm_scaled_matmul_28d379a9237322d1+0x4b3c>
	v_dual_mov_b32 v49, s39 :: v_dual_lshlrev_b32 v52, 3, v54  // 000000005804: ca220027 31346c83
	s_add_nc_u64 s[0:1], s[18:19], 0x7f                        // 00000000580c: a980ff12 0000007f
	v_dual_mov_b32 v41, s39 :: v_dual_mov_b32 v112, 0          // 000000005814: ca100027 29700080
	s_delay_alu instid0(valu_dep_2)                            // 00000000581c: bf870002
	v_or_b32_e32 v48, s38, v52                                 // 000000005820: 38606826
	v_or_b32_e32 v0, 1, v52                                    // 000000005824: 38006881
	v_or_b32_e32 v3, 4, v52                                    // 000000005828: 38066884
	v_or_b32_e32 v2, 3, v52                                    // 00000000582c: 38046883
	s_wait_alu depctr_sa_sdst(0)                               // 000000005830: bf88ff9e
	s_lshr_b64 s[4:5], s[0:1], 7                               // 000000005834: 85848700
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[48:49]                // 000000005838: 7ca86010
	v_or_b32_e32 v50, s38, v0                                  // 00000000583c: 38640026
	v_or_b32_e32 v38, s38, v3                                  // 000000005840: 384c0626
	v_mov_b32_e32 v51, s39                                     // 000000005844: 7e660227
	v_or_b32_e32 v30, s33, v52                                 // 000000005848: 383c6821
	v_mov_b32_e32 v45, s39                                     // 00000000584c: 7e5a0227
	s_wait_alu depctr_va_vcc(0)                                // 000000005850: bf88ff9d
	v_cndmask_b32_e32 v77, 0, v48, vcc_lo                      // 000000005854: 029a6080
	v_cndmask_b32_e64 v78, 0, s39, vcc_lo                      // 000000005858: d501004e 01a84e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[50:51]                // 000000005860: 7ca86410
	v_or_b32_e32 v44, s38, v2                                  // 000000005864: 38580426
	v_or_b32_e32 v5, 6, v52                                    // 000000005868: 380a6886
	s_lshr_b64 s[0:1], s[36:37], 7                             // 00000000586c: 85808724
	s_wait_alu depctr_sa_sdst(0)                               // 000000005870: bf88ff9e
	s_add_nc_u64 s[2:3], s[4:5], -1                            // 000000005874: a982c104
	v_or_b32_e32 v6, 7, v52                                    // 000000005878: 380c6887
	s_wait_alu depctr_sa_sdst(0)                               // 00000000587c: bf88ff9e
	v_cmp_lt_u64_e64 s6, s[0:1], s[2:3]                        // 000000005880: d4590006 02000400
	s_wait_alu depctr_va_vcc(0)                                // 000000005888: bf88ff9d
	v_dual_mov_b32 v53, 0 :: v_dual_cndmask_b32 v80, 0, v51    // 00000000588c: ca120080 35506680
	v_or_b32_e32 v40, s38, v5                                  // 000000005894: 38500a26
	v_or_b32_e32 v20, s33, v2                                  // 000000005898: 38280421
	v_cndmask_b32_e32 v79, 0, v50, vcc_lo                      // 00000000589c: 029e6480
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[44:45]                // 0000000058a0: 7ca85810
	v_or_b32_e32 v16, s33, v6                                  // 0000000058a4: 38200c21
	v_mov_b32_e32 v39, s39                                     // 0000000058a8: 7e4e0227
	s_lshr_b64 s[8:9], s[28:29], 7                             // 0000000058ac: 8588871c
	s_and_b32 s6, s6, exec_lo                                  // 0000000058b0: 8b067e06
	s_cselect_b32 s6, s0, s2                                   // 0000000058b4: 98060200
	v_cmp_gt_i64_e64 s2, s[16:17], v[40:41]                    // 0000000058b8: d4540002 02025010
	s_wait_alu depctr_va_vcc(0)                                // 0000000058c0: bf88ff9d
	v_dual_cndmask_b32 v83, 0, v44 :: v_dual_mov_b32 v114, 0   // 0000000058c4: ca505880 53720080
	v_cndmask_b32_e32 v84, 0, v45, vcc_lo                      // 0000000058cc: 02a85a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[38:39]                // 0000000058d0: 7ca84c10
	v_dual_mov_b32 v27, s39 :: v_dual_mov_b32 v96, 0           // 0000000058d4: ca100027 1b600080
	v_or_b32_e32 v26, s33, v0                                  // 0000000058dc: 38340021
	v_dual_mov_b32 v37, s39 :: v_dual_mov_b32 v110, 0          // 0000000058e0: ca100027 256e0080
	v_or_b32_e32 v36, s38, v6                                  // 0000000058e8: 38480c26
	s_wait_alu depctr_va_sdst(0)                               // 0000000058ec: bf88f19f
	v_cndmask_b32_e64 v70, 0, v40, s2                          // 0000000058f0: d5010046 000a5080
	v_cndmask_b32_e64 v71, 0, v41, s2                          // 0000000058f8: d5010047 000a5280
	v_cmp_gt_i64_e64 s2, s[16:17], v[26:27]                    // 000000005900: d4540002 02023410
	s_wait_alu depctr_va_vcc(0)                                // 000000005908: bf88ff9d
	v_dual_cndmask_b32 v66, 0, v38 :: v_dual_cndmask_b32 v67, 0, v39// 00000000590c: ca524c80 42424e80
	v_mov_b32_e32 v108, 0                                      // 000000005914: 7ed80280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[36:37]                // 000000005918: 7ca84810
	v_dual_mov_b32 v31, s39 :: v_dual_mov_b32 v98, 0           // 00000000591c: ca100027 1f620080
	s_wait_alu depctr_va_sdst(0)                               // 000000005924: bf88f19f
	v_cndmask_b32_e64 v14, 0, v26, s2                          // 000000005928: d501000e 000a3480
	v_cndmask_b32_e64 v15, 0, s39, s2                          // 000000005930: d501000f 00084e80
	s_wait_alu depctr_va_vcc(0)                                // 000000005938: bf88ff9d
	v_dual_mov_b32 v47, s39 :: v_dual_cndmask_b32 v72, 0, v36  // 00000000593c: ca120027 2f484880
	v_dual_cndmask_b32 v73, 0, v37 :: v_dual_mov_b32 v106, 0   // 000000005944: ca504a80 496a0080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[30:31]                // 00000000594c: 7ca83c10
	v_dual_mov_b32 v21, s39 :: v_dual_mov_b32 v92, 0           // 000000005950: ca100027 155c0080
	s_cselect_b32 s7, s1, s3                                   // 000000005958: 98070301
	s_lshr_b32 s9, s29, 7                                      // 00000000595c: 8509871d
	v_or_b32_e32 v1, 2, v52                                    // 000000005960: 38026882
	s_wait_alu depctr_sa_sdst(0)                               // 000000005964: bf88ff9e
	v_mul_lo_u32 v87, v15, s8                                  // 000000005968: d72c0057 0200110f
	v_mul_lo_u32 v88, v14, s9                                  // 000000005970: d72c0058 0200130e
	v_mad_co_u64_u32 v[14:15], null, v14, s8, 0                // 000000005978: d6fe7c0e 0200110e
	s_wait_alu depctr_va_vcc(0)                                // 000000005980: bf88ff9d
	v_cndmask_b32_e32 v74, 0, v30, vcc_lo                      // 000000005984: 02943c80
	v_cndmask_b32_e64 v75, 0, s39, vcc_lo                      // 000000005988: d501004b 01a84e80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[20:21]                // 000000005990: 7ca82810
	v_mov_b32_e32 v17, s39                                     // 000000005994: 7e220227
	v_dual_mov_b32 v25, s39 :: v_dual_mov_b32 v94, 0           // 000000005998: ca100027 195e0080
	v_or_b32_e32 v24, s33, v1                                  // 0000000059a0: 38300221
	v_or_b32_e32 v4, 5, v52                                    // 0000000059a4: 38086885
	s_wait_alu depctr_va_vcc(0)                                // 0000000059a8: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v20, vcc_lo                       // 0000000059ac: 02102880
	v_cndmask_b32_e64 v9, 0, s39, vcc_lo                       // 0000000059b0: d5010009 01a84e80
	v_add3_u32 v15, v15, v88, v87                              // 0000000059b8: d655000f 055eb10f
	v_mov_b32_e32 v88, 0                                       // 0000000059c0: 7eb00280
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[16:17]                // 0000000059c4: 7ca82010
	v_or_b32_e32 v46, s38, v1                                  // 0000000059c8: 385c0226
	v_cmp_gt_i64_e64 s3, s[16:17], v[24:25]                    // 0000000059cc: d4540003 02023010
	v_dual_mov_b32 v29, s39 :: v_dual_mov_b32 v104, 0          // 0000000059d4: ca100027 1d680080
	v_or_b32_e32 v28, s33, v3                                  // 0000000059dc: 38380621
	v_dual_mov_b32 v23, s39 :: v_dual_mov_b32 v102, 0          // 0000000059e0: ca100027 17660080
	v_dual_mov_b32 v19, s39 :: v_dual_mov_b32 v100, 0          // 0000000059e8: ca100027 13640080
	v_or_b32_e32 v18, s33, v5                                  // 0000000059f0: 38240a21
	v_or_b32_e32 v22, s33, v4                                  // 0000000059f4: 382c0821
	s_wait_alu depctr_va_vcc(0)                                // 0000000059f8: bf88ff9d
	v_cndmask_b32_e32 v0, 0, v16, vcc_lo                       // 0000000059fc: 02002080
	v_cndmask_b32_e64 v1, 0, s39, vcc_lo                       // 000000005a00: d5010001 01a84e80
	v_cmp_gt_i64_e64 s1, s[16:17], v[46:47]                    // 000000005a08: d4540001 02025c10
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[28:29]                // 000000005a10: 7ca83810
	v_mov_b32_e32 v43, s39                                     // 000000005a14: 7e560227
	v_or_b32_e32 v42, s38, v4                                  // 000000005a18: 38540826
	s_wait_alu depctr_va_sdst(0)                               // 000000005a1c: bf88f19f
	v_cndmask_b32_e64 v12, 0, v24, s3                          // 000000005a20: d501000c 000e3080
	v_cndmask_b32_e64 v13, 0, s39, s3                          // 000000005a28: d501000d 000c4e80
	v_cmp_gt_i64_e64 s2, s[16:17], v[18:19]                    // 000000005a30: d4540002 02022410
	v_cmp_gt_i64_e64 s3, s[16:17], v[22:23]                    // 000000005a38: d4540003 02022c10
	v_mul_lo_u32 v2, v1, s8                                    // 000000005a40: d72c0002 02001101
	v_mul_lo_u32 v3, v0, s9                                    // 000000005a48: d72c0003 02001300
	v_mad_co_u64_u32 v[0:1], null, v0, s8, 0                   // 000000005a50: d6fe7c00 02001100
	v_cndmask_b32_e64 v81, 0, v46, s1                          // 000000005a58: d5010051 00065c80
	v_cndmask_b32_e64 v82, 0, v47, s1                          // 000000005a60: d5010052 00065e80
	s_wait_alu depctr_va_vcc(0)                                // 000000005a68: bf88ff9d
	v_cndmask_b32_e32 v6, 0, v28, vcc_lo                       // 000000005a6c: 020c3880
	v_cmp_gt_i64_e64 s1, s[16:17], v[42:43]                    // 000000005a70: d4540001 02025410
	s_wait_alu depctr_va_sdst(0)                               // 000000005a78: bf88f19f
	v_cndmask_b32_e64 v4, 0, v18, s2                           // 000000005a7c: d5010004 000a2480
	v_cndmask_b32_e64 v5, 0, s39, s2                           // 000000005a84: d5010005 00084e80
	v_cndmask_b32_e64 v7, 0, s39, vcc_lo                       // 000000005a8c: d5010007 01a84e80
	v_cndmask_b32_e64 v10, 0, v22, s3                          // 000000005a94: d501000a 000e2c80
	v_cndmask_b32_e64 v11, 0, s39, s3                          // 000000005a9c: d501000b 000c4e80
	v_cndmask_b32_e64 v68, 0, v42, s1                          // 000000005aa4: d5010044 00065480
	v_cndmask_b32_e64 v69, 0, v43, s1                          // 000000005aac: d5010045 00065680
	v_cmp_gt_i64_e64 s1, s[18:19], v[34:35]                    // 000000005ab4: d4540001 02024412
	v_add3_u32 v1, v1, v3, v2                                  // 000000005abc: d6550001 040a0701
	v_mul_lo_u32 v34, v5, s8                                   // 000000005ac4: d72c0022 02001105
	v_mul_lo_u32 v35, v4, s9                                   // 000000005acc: d72c0023 02001304
	v_mad_co_u64_u32 v[2:3], null, v4, s8, 0                   // 000000005ad4: d6fe7c02 02001104
	v_mul_lo_u32 v11, v11, s8                                  // 000000005adc: d72c000b 0200110b
	v_mul_lo_u32 v54, v10, s9                                  // 000000005ae4: d72c0036 0200130a
	v_mad_co_u64_u32 v[4:5], null, v10, s8, 0                  // 000000005aec: d6fe7c04 0200110a
	v_mul_lo_u32 v10, v7, s8                                   // 000000005af4: d72c000a 02001107
	v_mul_lo_u32 v55, v6, s9                                   // 000000005afc: d72c0037 02001306
	v_mad_co_u64_u32 v[6:7], null, v6, s8, 0                   // 000000005b04: d6fe7c06 02001106
	v_add_co_u32 v85, s2, s38, v76                             // 000000005b0c: d7000255 02029826
	s_wait_alu depctr_va_sdst(0)                               // 000000005b14: bf88f19f
	v_add_co_ci_u32_e64 v86, null, s39, 0, s2                  // 000000005b18: d5207c56 00090027
	v_add3_u32 v5, v5, v54, v11                                // 000000005b20: d6550005 042e6d05
	s_delay_alu instid0(valu_dep_3)                            // 000000005b28: bf870003
	v_mul_lo_u32 v63, s29, v85                                 // 000000005b2c: d72c003f 0202aa1d
	v_mul_lo_u32 v64, v13, s8                                  // 000000005b34: d72c0040 0200110d
	v_add3_u32 v7, v7, v55, v10                                // 000000005b3c: d6550007 042a6f07
	v_mad_co_u64_u32 v[10:11], null, s28, v85, v[52:53]        // 000000005b44: d6fe7c0a 04d2aa1c
	v_mul_lo_u32 v62, s28, v86                                 // 000000005b4c: d72c003e 0202ac1c
	v_mul_lo_u32 v65, v12, s9                                  // 000000005b54: d72c0041 0200130c
	v_mad_co_u64_u32 v[12:13], null, v12, s8, 0                // 000000005b5c: d6fe7c0c 0200110c
	v_mul_lo_u32 v56, v9, s8                                   // 000000005b64: d72c0038 02001109
	v_mul_lo_u32 v57, v8, s9                                   // 000000005b6c: d72c0039 02001308
	v_mad_co_u64_u32 v[8:9], null, v8, s8, 0                   // 000000005b74: d6fe7c08 02001108
	v_add3_u32 v3, v3, v35, v34                                // 000000005b7c: d6550003 048a4703
	v_lshlrev_b64_e32 v[34:35], 2, v[0:1]                      // 000000005b84: 3e440082
	v_add3_u32 v0, v63, v11, v62                               // 000000005b88: d6550000 04fa173f
	v_add_co_u32 v115, vcc_lo, s34, v10                        // 000000005b90: d7006a73 02021422
	v_add3_u32 v13, v13, v65, v64                              // 000000005b98: d655000d 0502830d
	v_lshlrev_b64_e32 v[58:59], 2, v[6:7]                      // 000000005ba0: 3e740c82
	v_add3_u32 v9, v9, v57, v56                                // 000000005ba4: d6550009 04e27309
	v_lshlrev_b64_e32 v[56:57], 2, v[4:5]                      // 000000005bac: 3e700882
	v_lshlrev_b64_e32 v[64:65], 2, v[14:15]                    // 000000005bb0: 3e801c82
	v_mul_lo_u32 v14, v71, s8                                  // 000000005bb4: d72c000e 02001147
	v_mul_lo_u32 v15, v70, s9                                  // 000000005bbc: d72c000f 02001346
	v_mad_co_u64_u32 v[4:5], null, v70, s8, 0                  // 000000005bc4: d6fe7c04 02001146
	v_mul_lo_u32 v69, v69, s8                                  // 000000005bcc: d72c0045 02001145
	v_mul_lo_u32 v70, v68, s9                                  // 000000005bd4: d72c0046 02001344
	v_mad_co_u64_u32 v[6:7], null, v68, s8, 0                  // 000000005bdc: d6fe7c06 02001144
	v_lshlrev_b64_e32 v[54:55], 2, v[2:3]                      // 000000005be4: 3e6c0482
	s_wait_alu depctr_va_vcc(0)                                // 000000005be8: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, s35, v0, vcc_lo            // 000000005bec: d5207c74 01aa0023
	v_lshlrev_b64_e32 v[62:63], 2, v[12:13]                    // 000000005bf4: 3e7c1882
	v_mul_lo_u32 v10, v75, s8                                  // 000000005bf8: d72c000a 0200114b
	v_mul_lo_u32 v11, v74, s9                                  // 000000005c00: d72c000b 0200134a
	v_mad_co_u64_u32 v[0:1], null, v74, s8, 0                  // 000000005c08: d6fe7c00 0200114a
	v_mul_lo_u32 v12, v73, s8                                  // 000000005c10: d72c000c 02001149
	v_mul_lo_u32 v13, v72, s9                                  // 000000005c18: d72c000d 02001348
	v_mad_co_u64_u32 v[2:3], null, v72, s8, 0                  // 000000005c20: d6fe7c02 02001148
	v_lshlrev_b64_e32 v[60:61], 2, v[8:9]                      // 000000005c28: 3e781082
	v_mul_lo_u32 v67, v67, s8                                  // 000000005c2c: d72c0043 02001143
	v_mul_lo_u32 v68, v66, s9                                  // 000000005c34: d72c0044 02001342
	v_mad_co_u64_u32 v[8:9], null, v66, s8, 0                  // 000000005c3c: d6fe7c08 02001142
	v_add_co_u32 v66, vcc_lo, v85, 16                          // 000000005c44: d7006a42 02012155
	s_wait_alu depctr_va_vcc(0)                                // 000000005c4c: bf88ff9d
	v_add_co_ci_u32_e64 v71, null, 0, v86, vcc_lo              // 000000005c50: d5207c47 01aaac80
	v_add3_u32 v5, v5, v15, v14                                // 000000005c58: d6550005 043a1f05
	v_add3_u32 v7, v7, v70, v69                                // 000000005c60: d6550007 05168d07
	v_add3_u32 v1, v1, v11, v10                                // 000000005c68: d6550001 042a1701
	v_add3_u32 v3, v3, v13, v12                                // 000000005c70: d6550003 04321b03
	v_mad_co_u64_u32 v[10:11], null, s28, v66, v[52:53]        // 000000005c78: d6fe7c0a 04d2841c
	v_mul_lo_u32 v12, s28, v71                                 // 000000005c80: d72c000c 02028e1c
	v_mul_lo_u32 v13, s29, v66                                 // 000000005c88: d72c000d 0202841d
	v_add3_u32 v9, v9, v68, v67                                // 000000005c90: d6550009 050e8909
	v_add_co_u32 v76, s2, s36, v76                             // 000000005c98: d700024c 02029824
	v_lshlrev_b64_e32 v[70:71], 2, v[4:5]                      // 000000005ca0: 3e8c0882
	v_lshlrev_b64_e32 v[72:73], 2, v[6:7]                      // 000000005ca4: 3e900c82
	v_mul_lo_u32 v15, v79, s9                                  // 000000005ca8: d72c000f 0200134f
	v_mad_co_u64_u32 v[4:5], null, v79, s8, 0                  // 000000005cb0: d6fe7c04 0200114f
	v_mul_lo_u32 v79, v77, s9                                  // 000000005cb8: d72c004f 0200134d
	v_mad_co_u64_u32 v[6:7], null, v77, s8, 0                  // 000000005cc0: d6fe7c06 0200114d
	s_wait_alu depctr_va_sdst(0)                               // 000000005cc8: bf88f19f
	v_add_co_ci_u32_e64 v77, null, s37, 0, s2                  // 000000005ccc: d5207c4d 00090025
	v_lshlrev_b64_e32 v[66:67], 2, v[0:1]                      // 000000005cd4: 3e840082
	v_lshlrev_b64_e32 v[74:75], 2, v[8:9]                      // 000000005cd8: 3e941082
	v_mul_lo_u32 v8, v84, s8                                   // 000000005cdc: d72c0008 02001154
	v_mul_lo_u32 v9, v83, s9                                   // 000000005ce4: d72c0009 02001353
	v_mad_co_u64_u32 v[0:1], null, v83, s8, 0                  // 000000005cec: d6fe7c00 02001153
	v_mul_lo_u32 v14, v80, s8                                  // 000000005cf4: d72c000e 02001150
	v_add_co_u32 v80, vcc_lo, v76, 16                          // 000000005cfc: d7006a50 0201214c
	v_lshlrev_b64_e32 v[68:69], 2, v[2:3]                      // 000000005d04: 3e880482
	v_add3_u32 v13, v13, v11, v12                              // 000000005d08: d655000d 0432170d
	v_mul_lo_u32 v11, v82, s8                                  // 000000005d10: d72c000b 02001152
	v_mul_lo_u32 v12, v81, s9                                  // 000000005d18: d72c000c 02001351
	v_mad_co_u64_u32 v[2:3], null, v81, s8, 0                  // 000000005d20: d6fe7c02 02001151
	s_wait_alu depctr_va_vcc(0)                                // 000000005d28: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, 0, v77, vcc_lo              // 000000005d2c: d5207c51 01aa9a80
	v_add3_u32 v1, v1, v9, v8                                  // 000000005d34: d6550001 04221301
	v_add3_u32 v5, v5, v15, v14                                // 000000005d3c: d6550005 043a1f05
	v_mad_co_u64_u32 v[8:9], null, s28, v80, v[52:53]          // 000000005d44: d6fe7c08 04d2a01c
	s_delay_alu instid0(valu_dep_4)                            // 000000005d4c: bf870004
	v_mul_lo_u32 v14, s28, v81                                 // 000000005d50: d72c000e 0202a21c
	v_mul_lo_u32 v15, s29, v80                                 // 000000005d58: d72c000f 0202a01d
	v_mul_lo_u32 v78, v78, s8                                  // 000000005d60: d72c004e 0200114e
	v_add3_u32 v3, v3, v12, v11                                // 000000005d68: d6550003 042e1903
	v_mad_co_u64_u32 v[11:12], null, s28, v76, v[52:53]        // 000000005d70: d6fe7c0b 04d2981c
	v_mul_lo_u32 v52, s28, v77                                 // 000000005d78: d72c0034 02029a1c
	v_mul_lo_u32 v84, s29, v76                                 // 000000005d80: d72c0054 0202981d
	v_lshlrev_b64_e32 v[76:77], 2, v[0:1]                      // 000000005d88: 3e980082
	v_add_co_u32 v117, vcc_lo, s34, v10                        // 000000005d8c: d7006a75 02021422
	v_add3_u32 v0, v15, v9, v14                                // 000000005d94: d6550000 043a130f
	v_add3_u32 v7, v7, v79, v78                                // 000000005d9c: d6550007 053a9f07
	s_wait_alu depctr_va_vcc(0)                                // 000000005da4: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, s35, v13, vcc_lo           // 000000005da8: d5207c76 01aa1a23
	v_add3_u32 v1, v84, v12, v52                               // 000000005db0: d6550001 04d21954
	v_add_co_u32 v119, vcc_lo, s30, v8                         // 000000005db8: d7006a77 0202101e
	s_wait_alu depctr_va_vcc(0)                                // 000000005dc0: bf88ff9d
	v_add_co_ci_u32_e64 v120, null, s31, v0, vcc_lo            // 000000005dc4: d5207c78 01aa001f
	v_add_co_u32 v121, vcc_lo, s30, v11                        // 000000005dcc: d7006a79 0202161e
	v_cmp_gt_i64_e64 s0, s[18:19], v[32:33]                    // 000000005dd4: d4540000 02024012
	v_lshlrev_b64_e32 v[78:79], 2, v[2:3]                      // 000000005ddc: 3e9c0482
	v_lshlrev_b64_e32 v[80:81], 2, v[4:5]                      // 000000005de0: 3ea00882
	v_lshlrev_b64_e32 v[82:83], 2, v[6:7]                      // 000000005de4: 3ea40c82
	s_wait_alu depctr_va_vcc(0)                                // 000000005de8: bf88ff9d
	v_add_co_ci_u32_e64 v122, null, s31, v1, vcc_lo            // 000000005dec: d5207c7a 01aa021f
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v90, 0            // 000000005df4: ca100080 715a0080
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v86, 0            // 000000005dfc: ca100080 6f560080
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v84, 0            // 000000005e04: ca100080 6d540080
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v52, 0             // 000000005e0c: ca100080 61340080
	v_mov_b32_e32 v95, 0                                       // 000000005e14: 7ebe0280
	v_mov_b32_e32 v91, 0                                       // 000000005e18: 7eb60280
	v_mov_b32_e32 v107, 0                                      // 000000005e1c: 7ed60280
	v_mov_b32_e32 v105, 0                                      // 000000005e20: 7ed20280
	v_mov_b32_e32 v103, 0                                      // 000000005e24: 7ece0280
	v_mov_b32_e32 v101, 0                                      // 000000005e28: 7eca0280
	v_mov_b32_e32 v99, 0                                       // 000000005e2c: 7ec60280
	v_mov_b32_e32 v93, 0                                       // 000000005e30: 7eba0280
	v_mov_b32_e32 v89, 0                                       // 000000005e34: 7eb20280
	v_mov_b32_e32 v87, 0                                       // 000000005e38: 7eae0280
	v_mov_b32_e32 v85, 0                                       // 000000005e3c: 7eaa0280
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000005e40: 84828206
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000005e44: 84848204
	s_mov_b64 s[8:9], 0                                        // 000000005e48: be880180
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e4c: bf88ff9e
	v_add_co_u32 v140, vcc_lo, v115, s8                        // 000000005e50: d7006a8c 02001173
	s_wait_alu depctr_va_vcc(0)                                // 000000005e58: bf88ff9d
	v_add_co_ci_u32_e64 v141, null, s9, v116, vcc_lo           // 000000005e5c: d5207c8d 01aae809
	v_add_co_u32 v142, vcc_lo, v117, s8                        // 000000005e64: d7006a8e 02001175
	s_wait_alu depctr_va_vcc(0)                                // 000000005e6c: bf88ff9d
	v_add_co_ci_u32_e64 v143, null, s9, v118, vcc_lo           // 000000005e70: d5207c8f 01aaec09
	v_add_co_u32 v146, vcc_lo, v121, s8                        // 000000005e78: d7006a92 02001179
	s_wait_alu depctr_va_vcc(0)                                // 000000005e80: bf88ff9d
	v_add_co_ci_u32_e64 v147, null, s9, v122, vcc_lo           // 000000005e84: d5207c93 01aaf409
	v_add_co_u32 v148, vcc_lo, v119, s8                        // 000000005e8c: d7006a94 02001177
	s_wait_alu depctr_va_vcc(0)                                // 000000005e94: bf88ff9d
	v_add_co_ci_u32_e64 v149, null, s9, v120, vcc_lo           // 000000005e98: d5207c95 01aaf009
	s_clause 0x1                                               // 000000005ea0: bf850001
	global_load_b64 v[0:1], v[140:141], off                    // 000000005ea4: ee05407c 00000000 0000008c
	global_load_b64 v[144:145], v[142:143], off                // 000000005eb0: ee05407c 00000090 0000008e
	s_clause 0x1                                               // 000000005ebc: bf850001
	global_load_b64 v[2:3], v[146:147], off                    // 000000005ec0: ee05407c 00000002 00000092
	global_load_b64 v[150:151], v[148:149], off                // 000000005ecc: ee05407c 00000096 00000094
	s_add_nc_u64 s[6:7], s[8:9], 0x80                          // 000000005ed8: a986ff08 00000080
	s_add_nc_u64 s[8:9], s[24:25], s[2:3]                      // 000000005ee0: a9880218
	s_wait_loadcnt 0x1                                         // 000000005ee4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[0:1], v[2:3], 0  // 000000005ee8: cc46407c 1a020500
	s_wait_loadcnt 0x0                                         // 000000005ef0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[0:1], v[150:151], 0// 000000005ef4: cc464084 1a032d00
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[144:145], v[2:3], 0 // 000000005efc: cc464008 1a020590
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[144:145], v[150:151], 0// 000000005f04: cc464000 1a032d90
	s_clause 0x1                                               // 000000005f0c: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:16      // 000000005f10: ee05407c 00000090 0000108c
	global_load_b64 v[150:151], v[142:143], off offset:16      // 000000005f1c: ee05407c 00000096 0000108e
	s_clause 0x1                                               // 000000005f28: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:16      // 000000005f2c: ee05407c 00000098 00001092
	global_load_b64 v[154:155], v[148:149], off offset:16      // 000000005f38: ee05407c 0000009a 00001094
	s_wait_loadcnt 0x1                                         // 000000005f44: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 000000005f48: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 000000005f50: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 000000005f54: cc464084 1e133590
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 000000005f5c: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 000000005f64: cc464000 1c033596
	s_clause 0x1                                               // 000000005f6c: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:32      // 000000005f70: ee05407c 00000090 0000208c
	global_load_b64 v[150:151], v[142:143], off offset:32      // 000000005f7c: ee05407c 00000096 0000208e
	s_clause 0x1                                               // 000000005f88: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:32      // 000000005f8c: ee05407c 00000098 00002092
	global_load_b64 v[154:155], v[148:149], off offset:32      // 000000005f98: ee05407c 0000009a 00002094
	s_wait_loadcnt 0x1                                         // 000000005fa4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 000000005fa8: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 000000005fb0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 000000005fb4: cc464084 1e133590
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 000000005fbc: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 000000005fc4: cc464000 1c033596
	s_clause 0x1                                               // 000000005fcc: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:48      // 000000005fd0: ee05407c 00000090 0000308c
	global_load_b64 v[150:151], v[142:143], off offset:48      // 000000005fdc: ee05407c 00000096 0000308e
	s_clause 0x1                                               // 000000005fe8: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:48      // 000000005fec: ee05407c 00000098 00003092
	global_load_b64 v[154:155], v[148:149], off offset:48      // 000000005ff8: ee05407c 0000009a 00003094
	s_wait_loadcnt 0x1                                         // 000000006004: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 000000006008: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 000000006010: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 000000006014: cc464084 1e133590
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 00000000601c: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 000000006024: cc464000 1c033596
	s_clause 0x1                                               // 00000000602c: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:64      // 000000006030: ee05407c 00000090 0000408c
	global_load_b64 v[150:151], v[142:143], off offset:64      // 00000000603c: ee05407c 00000096 0000408e
	s_clause 0x1                                               // 000000006048: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:64      // 00000000604c: ee05407c 00000098 00004092
	global_load_b64 v[154:155], v[148:149], off offset:64      // 000000006058: ee05407c 0000009a 00004094
	s_wait_loadcnt 0x1                                         // 000000006064: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 000000006068: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 000000006070: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 000000006074: cc464084 1e133590
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 00000000607c: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 000000006084: cc464000 1c033596
	s_clause 0x1                                               // 00000000608c: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:80      // 000000006090: ee05407c 00000090 0000508c
	global_load_b64 v[150:151], v[142:143], off offset:80      // 00000000609c: ee05407c 00000096 0000508e
	s_clause 0x1                                               // 0000000060a8: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:80      // 0000000060ac: ee05407c 00000098 00005092
	global_load_b64 v[154:155], v[148:149], off offset:80      // 0000000060b8: ee05407c 0000009a 00005094
	s_wait_loadcnt 0x1                                         // 0000000060c4: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 0000000060c8: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 0000000060d0: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 0000000060d4: cc464084 1e133590
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 0000000060dc: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 0000000060e4: cc464000 1c033596
	s_clause 0x1                                               // 0000000060ec: bf850001
	global_load_b64 v[144:145], v[140:141], off offset:96      // 0000000060f0: ee05407c 00000090 0000608c
	global_load_b64 v[150:151], v[142:143], off offset:96      // 0000000060fc: ee05407c 00000096 0000608e
	s_clause 0x1                                               // 000000006108: bf850001
	global_load_b64 v[152:153], v[146:147], off offset:96      // 00000000610c: ee05407c 00000098 00006092
	global_load_b64 v[154:155], v[148:149], off offset:96      // 000000006118: ee05407c 0000009a 00006094
	s_wait_loadcnt 0x1                                         // 000000006124: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[144:145], v[152:153], v[124:131]// 000000006128: cc46407c 1df33190
	s_wait_loadcnt 0x0                                         // 000000006130: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[144:145], v[154:155], v[132:139]// 000000006134: cc464084 1e133590
	s_clause 0x1                                               // 00000000613c: bf850001
	global_load_b64 v[140:141], v[140:141], off offset:112     // 000000006140: ee05407c 0000008c 0000708c
	global_load_b64 v[142:143], v[142:143], off offset:112     // 00000000614c: ee05407c 0000008e 0000708e
	s_clause 0x1                                               // 000000006158: bf850001
	global_load_b64 v[144:145], v[146:147], off offset:112     // 00000000615c: ee05407c 00000090 00007092
	global_load_b64 v[146:147], v[148:149], off offset:112     // 000000006168: ee05407c 00000092 00007094
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[150:151], v[152:153], v[8:15]// 000000006174: cc464008 1c233196
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[150:151], v[154:155], v[0:7]// 00000000617c: cc464000 1c033596
	s_wait_loadcnt 0x1                                         // 000000006184: bfc00001
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[140:141], v[144:145], v[124:131]// 000000006188: cc46407c 1df3218c
	s_wait_loadcnt 0x0                                         // 000000006190: bfc00000
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[140:141], v[146:147], v[132:139]// 000000006194: cc464084 1e13258c
	v_add_co_u32 v140, vcc_lo, s22, v82                        // 00000000619c: d7006a8c 0202a416
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[142:143], v[144:145], v[8:15]// 0000000061a4: cc464008 1c23218e
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[142:143], v[146:147], v[0:7]// 0000000061ac: cc464000 1c03258e
	global_load_b32 v142, v53, s[8:9]                          // 0000000061b4: ee050008 0000008e 00000035
	s_wait_alu depctr_va_vcc(0)                                // 0000000061c0: bf88ff9d
	v_add_co_ci_u32_e64 v141, null, s23, v83, vcc_lo           // 0000000061c4: d5207c8d 01aaa617
	s_load_b32 s8, s[24:25], 0x0                               // 0000000061cc: f400020c f8000000
	s_add_nc_u64 s[24:25], s[24:25], s[4:5]                    // 0000000061d4: a9980418
	global_load_b32 v143, v[140:141], off                      // 0000000061d8: ee05007c 0000008f 0000008c
	s_wait_loadcnt 0x1                                         // 0000000061e4: bfc00001
	s_wait_kmcnt 0x0                                           // 0000000061e8: bfc70000
	v_cndmask_b32_e64 v123, s8, v142, s0                       // 0000000061ec: d501007b 00031c08
	s_wait_loadcnt 0x0                                         // 0000000061f4: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000061f8: bf870091
	v_mul_f32_e32 v140, v143, v123                             // 0000000061fc: 1118f78f
	v_mul_f32_e32 v124, v124, v140                             // 000000006200: 10f9197c
	v_add_co_u32 v140, vcc_lo, s22, v80                        // 000000006204: d7006a8c 0202a016
	s_wait_alu depctr_va_vcc(0)                                // 00000000620c: bf88ff9d
	v_add_co_ci_u32_e64 v141, null, s23, v81, vcc_lo           // 000000006210: d5207c8d 01aaa217
	s_delay_alu instid0(valu_dep_3) | instskip(skip_3) | instid1(valu_dep_1)// 000000006218: bf8700c3
	v_add_f32_e32 v114, v114, v124                             // 00000000621c: 06e4f972
	global_load_b32 v140, v[140:141], off                      // 000000006220: ee05007c 0000008c 0000008c
	s_wait_loadcnt 0x0                                         // 00000000622c: bfc00000
	v_mul_f32_e32 v124, v123, v140                             // 000000006230: 10f9197b
	v_mul_f32_e32 v124, v125, v124                             // 000000006234: 10f8f97d
	s_delay_alu instid0(valu_dep_1)                            // 000000006238: bf870001
	v_add_f32_e32 v113, v113, v124                             // 00000000623c: 06e2f971
	v_add_co_u32 v124, vcc_lo, s22, v78                        // 000000006240: d7006a7c 02029c16
	s_wait_alu depctr_va_vcc(0)                                // 000000006248: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v79, vcc_lo           // 00000000624c: d5207c7d 01aa9e17
	global_load_b32 v141, v[124:125], off                      // 000000006254: ee05007c 0000008d 0000007c
	s_wait_loadcnt 0x0                                         // 000000006260: bfc00000
	v_mul_f32_e32 v124, v123, v141                             // 000000006264: 10f91b7b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006268: bf870091
	v_mul_f32_e32 v124, v126, v124                             // 00000000626c: 10f8f97e
	v_add_f32_e32 v112, v112, v124                             // 000000006270: 06e0f970
	v_add_co_u32 v124, vcc_lo, s22, v76                        // 000000006274: d7006a7c 02029816
	s_wait_alu depctr_va_vcc(0)                                // 00000000627c: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v77, vcc_lo           // 000000006280: d5207c7d 01aa9a17
	global_load_b32 v126, v[124:125], off                      // 000000006288: ee05007c 0000007e 0000007c
	s_wait_loadcnt 0x0                                         // 000000006294: bfc00000
	v_mul_f32_e32 v124, v123, v126                             // 000000006298: 10f8fd7b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000629c: bf870091
	v_mul_f32_e32 v124, v127, v124                             // 0000000062a0: 10f8f97f
	v_add_f32_e32 v111, v111, v124                             // 0000000062a4: 06def96f
	v_add_co_u32 v124, vcc_lo, s22, v74                        // 0000000062a8: d7006a7c 02029416
	s_wait_alu depctr_va_vcc(0)                                // 0000000062b0: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v75, vcc_lo           // 0000000062b4: d5207c7d 01aa9617
	global_load_b32 v127, v[124:125], off                      // 0000000062bc: ee05007c 0000007f 0000007c
	s_wait_loadcnt 0x0                                         // 0000000062c8: bfc00000
	v_mul_f32_e32 v124, v123, v127                             // 0000000062cc: 10f8ff7b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062d0: bf870091
	v_mul_f32_e32 v124, v128, v124                             // 0000000062d4: 10f8f980
	v_add_f32_e32 v110, v110, v124                             // 0000000062d8: 06dcf96e
	v_add_co_u32 v124, vcc_lo, s22, v72                        // 0000000062dc: d7006a7c 02029016
	s_wait_alu depctr_va_vcc(0)                                // 0000000062e4: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v73, vcc_lo           // 0000000062e8: d5207c7d 01aa9217
	global_load_b32 v128, v[124:125], off                      // 0000000062f0: ee05007c 00000080 0000007c
	s_wait_loadcnt 0x0                                         // 0000000062fc: bfc00000
	v_mul_f32_e32 v124, v123, v128                             // 000000006300: 10f9017b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006304: bf870091
	v_mul_f32_e32 v124, v129, v124                             // 000000006308: 10f8f981
	v_add_f32_e32 v109, v109, v124                             // 00000000630c: 06daf96d
	v_add_co_u32 v124, vcc_lo, s22, v70                        // 000000006310: d7006a7c 02028c16
	s_wait_alu depctr_va_vcc(0)                                // 000000006318: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v71, vcc_lo           // 00000000631c: d5207c7d 01aa8e17
	global_load_b32 v129, v[124:125], off                      // 000000006324: ee05007c 00000081 0000007c
	s_wait_loadcnt 0x0                                         // 000000006330: bfc00000
	v_mul_f32_e32 v124, v123, v129                             // 000000006334: 10f9037b
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_3)// 000000006338: bf8701b1
	v_mul_f32_e32 v124, v130, v124                             // 00000000633c: 10f8f982
	v_cndmask_b32_e64 v130, s8, v142, s1                       // 000000006340: d5010082 00071c08
	v_cmp_lt_i64_e64 s8, s[6:7], s[26:27]                      // 000000006348: d4510008 02003406
	v_add_f32_e32 v108, v108, v124                             // 000000006350: 06d8f96c
	v_add_co_u32 v124, vcc_lo, s22, v68                        // 000000006354: d7006a7c 02028816
	s_wait_alu depctr_va_vcc(0)                                // 00000000635c: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v69, vcc_lo           // 000000006360: d5207c7d 01aa8a17
	global_load_b32 v124, v[124:125], off                      // 000000006368: ee05007c 0000007c 0000007c
	s_wait_loadcnt 0x0                                         // 000000006374: bfc00000
	v_dual_mul_f32 v125, v123, v124 :: v_dual_mul_f32 v124, v130, v124// 000000006378: c8c6f97b 7d7cf982
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_2)// 000000006380: bf870111
	v_mul_f32_e32 v125, v131, v125                             // 000000006384: 10fafb83
	v_mul_f32_e32 v124, v139, v124                             // 000000006388: 10f8f98b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_2)// 00000000638c: bf870112
	v_dual_add_f32 v106, v106, v125 :: v_dual_mul_f32 v125, v143, v130// 000000006390: c906fb6a 6a7d058f
	v_add_f32_e32 v88, v88, v124                               // 000000006398: 06b0f958
	v_add_co_u32 v124, vcc_lo, s22, v66                        // 00000000639c: d7006a7c 02028416
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_1)// 0000000063a4: bf870093
	v_mul_f32_e32 v125, v132, v125                             // 0000000063a8: 10fafb84
	v_add_f32_e32 v98, v98, v125                               // 0000000063ac: 06c4fb62
	v_mul_f32_e32 v125, v130, v140                             // 0000000063b0: 10fb1982
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063b4: bf870091
	v_mul_f32_e32 v125, v133, v125                             // 0000000063b8: 10fafb85
	v_add_f32_e32 v97, v97, v125                               // 0000000063bc: 06c2fb61
	v_mul_f32_e32 v125, v130, v141                             // 0000000063c0: 10fb1b82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063c4: bf870091
	v_mul_f32_e32 v125, v134, v125                             // 0000000063c8: 10fafb86
	v_dual_add_f32 v96, v96, v125 :: v_dual_mul_f32 v125, v130, v126// 0000000063cc: c906fb60 607cfd82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063d4: bf870091
	v_mul_f32_e32 v125, v135, v125                             // 0000000063d8: 10fafb87
	v_add_f32_e32 v95, v95, v125                               // 0000000063dc: 06befb5f
	v_mul_f32_e32 v125, v130, v127                             // 0000000063e0: 10faff82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063e4: bf870091
	v_mul_f32_e32 v125, v136, v125                             // 0000000063e8: 10fafb88
	v_add_f32_e32 v94, v94, v125                               // 0000000063ec: 06bcfb5e
	v_mul_f32_e32 v125, v130, v128                             // 0000000063f0: 10fb0182
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000063f4: bf870091
	v_mul_f32_e32 v125, v137, v125                             // 0000000063f8: 10fafb89
	v_add_f32_e32 v92, v92, v125                               // 0000000063fc: 06b8fb5c
	v_mul_f32_e32 v125, v130, v129                             // 000000006400: 10fb0382
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006404: bf870091
	v_mul_f32_e32 v125, v138, v125                             // 000000006408: 10fafb8a
	v_add_f32_e32 v91, v91, v125                               // 00000000640c: 06b6fb5b
	s_wait_alu depctr_va_vcc(0)                                // 000000006410: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v67, vcc_lo           // 000000006414: d5207c7d 01aa8617
	global_load_b32 v126, v[124:125], off                      // 00000000641c: ee05007c 0000007e 0000007c
	s_wait_loadcnt 0x0                                         // 000000006428: bfc00000
	v_mul_f32_e32 v124, v123, v126                             // 00000000642c: 10f8fd7b
	s_delay_alu instid0(valu_dep_1) | instskip(skip_3) | instid1(valu_dep_3)// 000000006430: bf8701c1
	v_mul_f32_e32 v8, v8, v124                                 // 000000006434: 1010f908
	v_add_co_u32 v124, vcc_lo, s22, v64                        // 000000006438: d7006a7c 02028016
	s_wait_alu depctr_va_vcc(0)                                // 000000006440: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s23, v65, vcc_lo           // 000000006444: d5207c7d 01aa8217
	v_add_f32_e32 v107, v107, v8                               // 00000000644c: 06d6116b
	global_load_b32 v124, v[124:125], off                      // 000000006450: ee05007c 0000007c 0000007c
	s_wait_loadcnt 0x0                                         // 00000000645c: bfc00000
	v_mul_f32_e32 v8, v123, v124                               // 000000006460: 1010f97b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006464: bf870091
	v_mul_f32_e32 v8, v9, v8                                   // 000000006468: 10101109
	v_add_f32_e32 v105, v105, v8                               // 00000000646c: 06d21169
	v_add_co_u32 v8, vcc_lo, s22, v62                          // 000000006470: d7006a08 02027c16
	s_wait_alu depctr_va_vcc(0)                                // 000000006478: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v63, vcc_lo             // 00000000647c: d5207c09 01aa7e17
	global_load_b32 v125, v[8:9], off                          // 000000006484: ee05007c 0000007d 00000008
	s_wait_loadcnt 0x0                                         // 000000006490: bfc00000
	v_mul_f32_e32 v8, v123, v125                               // 000000006494: 1010fb7b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006498: bf870091
	v_mul_f32_e32 v8, v10, v8                                  // 00000000649c: 1010110a
	v_add_f32_e32 v104, v104, v8                               // 0000000064a0: 06d01168
	v_add_co_u32 v8, vcc_lo, s22, v60                          // 0000000064a4: d7006a08 02027816
	s_wait_alu depctr_va_vcc(0)                                // 0000000064ac: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v61, vcc_lo             // 0000000064b0: d5207c09 01aa7a17
	global_load_b32 v10, v[8:9], off                           // 0000000064b8: ee05007c 0000000a 00000008
	s_wait_loadcnt 0x0                                         // 0000000064c4: bfc00000
	v_mul_f32_e32 v8, v123, v10                                // 0000000064c8: 1010157b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000064cc: bf870091
	v_mul_f32_e32 v8, v11, v8                                  // 0000000064d0: 1010110b
	v_add_f32_e32 v103, v103, v8                               // 0000000064d4: 06ce1167
	v_add_co_u32 v8, vcc_lo, s22, v58                          // 0000000064d8: d7006a08 02027416
	s_wait_alu depctr_va_vcc(0)                                // 0000000064e0: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v59, vcc_lo             // 0000000064e4: d5207c09 01aa7617
	global_load_b32 v11, v[8:9], off                           // 0000000064ec: ee05007c 0000000b 00000008
	s_wait_loadcnt 0x0                                         // 0000000064f8: bfc00000
	v_mul_f32_e32 v8, v123, v11                                // 0000000064fc: 1010177b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006500: bf870091
	v_mul_f32_e32 v8, v12, v8                                  // 000000006504: 1010110c
	v_add_f32_e32 v102, v102, v8                               // 000000006508: 06cc1166
	v_add_co_u32 v8, vcc_lo, s22, v56                          // 00000000650c: d7006a08 02027016
	s_wait_alu depctr_va_vcc(0)                                // 000000006514: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v57, vcc_lo             // 000000006518: d5207c09 01aa7217
	global_load_b32 v12, v[8:9], off                           // 000000006520: ee05007c 0000000c 00000008
	s_wait_loadcnt 0x0                                         // 00000000652c: bfc00000
	v_mul_f32_e32 v8, v123, v12                                // 000000006530: 1010197b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006534: bf870091
	v_mul_f32_e32 v8, v13, v8                                  // 000000006538: 1010110d
	v_add_f32_e32 v101, v101, v8                               // 00000000653c: 06ca1165
	v_add_co_u32 v8, vcc_lo, s22, v54                          // 000000006540: d7006a08 02026c16
	s_wait_alu depctr_va_vcc(0)                                // 000000006548: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v55, vcc_lo             // 00000000654c: d5207c09 01aa6e17
	global_load_b32 v13, v[8:9], off                           // 000000006554: ee05007c 0000000d 00000008
	s_wait_loadcnt 0x0                                         // 000000006560: bfc00000
	v_mul_f32_e32 v8, v123, v13                                // 000000006564: 10101b7b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006568: bf870091
	v_mul_f32_e32 v8, v14, v8                                  // 00000000656c: 1010110e
	v_add_f32_e32 v100, v100, v8                               // 000000006570: 06c81164
	v_add_co_u32 v8, vcc_lo, s22, v34                          // 000000006574: d7006a08 02024416
	s_wait_alu depctr_va_vcc(0)                                // 00000000657c: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s23, v35, vcc_lo             // 000000006580: d5207c09 01aa4617
	s_add_nc_u64 s[22:23], s[22:23], 4                         // 000000006588: a9968416
	s_and_b32 vcc_lo, exec_lo, s8                              // 00000000658c: 8b6a087e
	s_mov_b64 s[8:9], s[6:7]                                   // 000000006590: be880106
	global_load_b32 v8, v[8:9], off                            // 000000006594: ee05007c 00000008 00000008
	s_wait_loadcnt 0x0                                         // 0000000065a0: bfc00000
	v_mul_f32_e32 v9, v123, v8                                 // 0000000065a4: 1012117b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065a8: bf870091
	v_mul_f32_e32 v9, v15, v9                                  // 0000000065ac: 1012130f
	v_add_f32_e32 v99, v99, v9                                 // 0000000065b0: 06c61363
	v_mul_f32_e32 v9, v130, v126                               // 0000000065b4: 1012fd82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065b8: bf870091
	v_mul_f32_e32 v0, v0, v9                                   // 0000000065bc: 10001300
	v_add_f32_e32 v93, v93, v0                                 // 0000000065c0: 06ba015d
	v_mul_f32_e32 v0, v130, v124                               // 0000000065c4: 1000f982
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065c8: bf870091
	v_mul_f32_e32 v0, v1, v0                                   // 0000000065cc: 10000101
	v_add_f32_e32 v90, v90, v0                                 // 0000000065d0: 06b4015a
	v_mul_f32_e32 v0, v130, v125                               // 0000000065d4: 1000fb82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065d8: bf870091
	v_mul_f32_e32 v0, v2, v0                                   // 0000000065dc: 10000102
	v_dual_add_f32 v89, v89, v0 :: v_dual_mul_f32 v0, v130, v10// 0000000065e0: c9060159 59001582
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065e8: bf870091
	v_mul_f32_e32 v0, v3, v0                                   // 0000000065ec: 10000103
	v_dual_add_f32 v87, v87, v0 :: v_dual_mul_f32 v0, v130, v11// 0000000065f0: c9060157 57001782
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000065f8: bf870091
	v_mul_f32_e32 v0, v4, v0                                   // 0000000065fc: 10000104
	v_add_f32_e32 v86, v86, v0                                 // 000000006600: 06ac0156
	v_mul_f32_e32 v0, v130, v12                                // 000000006604: 10001982
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006608: bf870091
	v_mul_f32_e32 v0, v5, v0                                   // 00000000660c: 10000105
	v_dual_add_f32 v85, v85, v0 :: v_dual_mul_f32 v0, v130, v13// 000000006610: c9060155 55001b82
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006618: bf870091
	v_mul_f32_e32 v0, v6, v0                                   // 00000000661c: 10000106
	v_add_f32_e32 v84, v84, v0                                 // 000000006620: 06a80154
	v_mul_f32_e32 v0, v130, v8                                 // 000000006624: 10001182
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006628: bf870091
	v_mul_f32_e32 v0, v7, v0                                   // 00000000662c: 10000107
	v_add_f32_e32 v52, v52, v0                                 // 000000006630: 06680134
	s_wait_alu depctr_sa_sdst(0)                               // 000000006634: bf88ff9e
	s_cbranch_vccnz 65028                                      // 000000006638: bfa4fe04 <tessera_rocm_scaled_matmul_28d379a9237322d1+0x434c>
	v_mul_lo_u32 v2, s19, v48                                  // 00000000663c: d72c0002 02026013
	v_mul_lo_u32 v3, s18, v49                                  // 000000006644: d72c0003 02026212
	v_mad_co_u64_u32 v[0:1], null, s18, v48, 0                 // 00000000664c: d6fe7c00 02026012
	v_bfe_u32 v8, v114, 16, 1                                  // 000000006654: d6100008 02052172
	v_mul_lo_u32 v6, s19, v50                                  // 00000000665c: d72c0006 02026413
	v_mul_lo_u32 v7, s18, v51                                  // 000000006664: d72c0007 02026612
	v_or_b32_e32 v9, 0x400000, v114                            // 00000000666c: 3812e4ff 00400000
	v_lshlrev_b64_e32 v[4:5], 1, v[32:33]                      // 000000006674: 3e084081
	v_add3_u32 v8, v8, v114, 0x7fff                            // 000000006678: d6550008 03fee508 00007fff
	v_mul_lo_u32 v11, s19, v46                                 // 000000006684: d72c000b 02025c13
	v_add3_u32 v1, v1, v3, v2                                  // 00000000668c: d6550001 040a0701
	v_mad_co_u64_u32 v[2:3], null, s18, v50, 0                 // 000000006694: d6fe7c02 02026412
	v_bfe_u32 v10, v113, 16, 1                                 // 00000000669c: d610000a 02052171
	v_or_b32_e32 v12, 0x400000, v113                           // 0000000066a4: 3818e2ff 00400000
	v_mul_lo_u32 v13, s19, v44                                 // 0000000066ac: d72c000d 02025813
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 0000000066b4: 3e000081
	v_mul_lo_u32 v14, s18, v45                                 // 0000000066b8: d72c000e 02025a12
	v_add3_u32 v10, v10, v113, 0x7fff                          // 0000000066c0: d655000a 03fee30a 00007fff
	v_bfe_u32 v32, v111, 16, 1                                 // 0000000066cc: d6100020 0205216f
	v_add3_u32 v3, v3, v7, v6                                  // 0000000066d4: d6550003 041a0f03
	v_mad_co_u64_u32 v[6:7], null, s18, v46, 0                 // 0000000066dc: d6fe7c06 02025c12
	v_add_co_u32 v0, vcc_lo, s20, v0                           // 0000000066e4: d7006a00 02020014
	s_wait_alu depctr_va_vcc(0)                                // 0000000066ec: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo              // 0000000066f0: d5207c01 01aa0215
	v_cmp_u_f32_e32 vcc_lo, v114, v114                         // 0000000066f8: 7c30e572
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 0000000066fc: 3e040481
	v_add3_u32 v32, v32, v111, 0x7fff                          // 000000006700: d6550020 03fedf20 00007fff
	v_or_b32_e32 v33, 0x400000, v111                           // 00000000670c: 3842deff 00400000
	v_mul_lo_u32 v34, s18, v43                                 // 000000006714: d72c0022 02025612
	s_wait_alu depctr_va_vcc(0)                                // 00000000671c: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo                       // 000000006720: 02101308
	v_mul_lo_u32 v9, s18, v47                                  // 000000006724: d72c0009 02025e12
	v_add_co_u32 v0, vcc_lo, v0, v4                            // 00000000672c: d7006a00 02020900
	s_wait_alu depctr_va_vcc(0)                                // 000000006734: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v1, v5, vcc_lo               // 000000006738: d5207c01 01aa0b01
	v_cmp_u_f32_e32 vcc_lo, v113, v113                         // 000000006740: 7c30e371
	v_mul_lo_u32 v37, s18, v37                                 // 000000006744: d72c0025 02024a12
	v_add3_u32 v7, v7, v9, v11                                 // 00000000674c: d6550007 042e1307
	global_store_d16_hi_b16 v[0:1], v8, off                    // 000000006754: ee09407c 04000000 00000000
	s_wait_alu depctr_va_vcc(0)                                // 000000006760: bf88ff9d
	v_cndmask_b32_e32 v12, v10, v12, vcc_lo                    // 000000006764: 0218190a
	v_add_co_u32 v8, vcc_lo, s20, v2                           // 000000006768: d7006a08 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006770: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, s21, v3, vcc_lo              // 000000006774: d5207c09 01aa0615
	v_lshlrev_b64_e32 v[2:3], 1, v[6:7]                        // 00000000677c: 3e040c81
	v_bfe_u32 v10, v112, 16, 1                                 // 000000006780: d610000a 02052170
	v_add_co_u32 v6, vcc_lo, v8, v4                            // 000000006788: d7006a06 02020908
	s_wait_alu depctr_va_vcc(0)                                // 000000006790: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v9, v5, vcc_lo               // 000000006794: d5207c07 01aa0b09
	s_delay_alu instid0(valu_dep_3)                            // 00000000679c: bf870003
	v_add3_u32 v8, v10, v112, 0x7fff                           // 0000000067a0: d6550008 03fee10a 00007fff
	v_add_co_u32 v10, vcc_lo, s20, v2                          // 0000000067ac: d7006a0a 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000067b4: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s21, v3, vcc_lo             // 0000000067b8: d5207c0b 01aa0615
	v_mad_co_u64_u32 v[2:3], null, s18, v44, 0                 // 0000000067c0: d6fe7c02 02025812
	v_or_b32_e32 v9, 0x400000, v112                            // 0000000067c8: 3812e0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v112, v112                         // 0000000067d0: 7c30e170
	s_wait_alu depctr_va_vcc(0)                                // 0000000067d4: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 0000000067d8: bf870002
	v_cndmask_b32_e32 v15, v8, v9, vcc_lo                      // 0000000067dc: 021e1308
	v_add_co_u32 v8, vcc_lo, v10, v4                           // 0000000067e0: d7006a08 0202090a
	s_wait_alu depctr_va_vcc(0)                                // 0000000067e8: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v11, v5, vcc_lo              // 0000000067ec: d5207c09 01aa0b0b
	v_add3_u32 v3, v3, v14, v13                                // 0000000067f4: d6550003 04361d03
	v_mul_lo_u32 v13, s19, v38                                 // 0000000067fc: d72c000d 02024c13
	v_mul_lo_u32 v14, s18, v39                                 // 000000006804: d72c000e 02024e12
	v_mad_co_u64_u32 v[10:11], null, s18, v38, 0               // 00000000680c: d6fe7c0a 02024c12
	v_cmp_u_f32_e32 vcc_lo, v111, v111                         // 000000006814: 7c30df6f
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006818: 3e040481
	s_clause 0x1                                               // 00000000681c: bf850001
	global_store_d16_hi_b16 v[6:7], v12, off                   // 000000006820: ee09407c 06000000 00000006
	global_store_d16_hi_b16 v[8:9], v15, off                   // 00000000682c: ee09407c 07800000 00000008
	v_bfe_u32 v38, v109, 16, 1                                 // 000000006838: d6100026 0205216d
	s_wait_alu depctr_va_vcc(0)                                // 000000006840: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v33, vcc_lo                    // 000000006844: 02404320
	v_mul_lo_u32 v33, s19, v42                                 // 000000006848: d72c0021 02025413
	v_add3_u32 v11, v11, v14, v13                              // 000000006850: d655000b 04361d0b
	v_add_co_u32 v12, vcc_lo, s20, v2                          // 000000006858: d7006a0c 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006860: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, s21, v3, vcc_lo             // 000000006864: d5207c0d 01aa0615
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 00000000686c: bf8701d3
	v_lshlrev_b64_e32 v[2:3], 1, v[10:11]                      // 000000006870: 3e041481
	v_bfe_u32 v14, v110, 16, 1                                 // 000000006874: d610000e 0205216e
	v_add_co_u32 v10, vcc_lo, v12, v4                          // 00000000687c: d7006a0a 0202090c
	s_wait_alu depctr_va_vcc(0)                                // 000000006884: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v13, v5, vcc_lo             // 000000006888: d5207c0b 01aa0b0d
	v_add3_u32 v12, v14, v110, 0x7fff                          // 000000006890: d655000c 03fedd0e 00007fff
	v_add_co_u32 v14, vcc_lo, s20, v2                          // 00000000689c: d7006a0e 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000068a4: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, s21, v3, vcc_lo             // 0000000068a8: d5207c0f 01aa0615
	v_mad_co_u64_u32 v[2:3], null, s18, v42, 0                 // 0000000068b0: d6fe7c02 02025412
	v_or_b32_e32 v13, 0x400000, v110                           // 0000000068b8: 381adcff 00400000
	v_cmp_u_f32_e32 vcc_lo, v110, v110                         // 0000000068c0: 7c30dd6e
	v_add3_u32 v38, v38, v109, 0x7fff                          // 0000000068c4: d6550026 03fedb26 00007fff
	v_or_b32_e32 v39, 0x400000, v109                           // 0000000068d0: 384edaff 00400000
	global_store_d16_hi_b16 v[10:11], v32, off                 // 0000000068d8: ee09407c 10000000 0000000a
	s_wait_alu depctr_va_vcc(0)                                // 0000000068e4: bf88ff9d
	v_cndmask_b32_e32 v35, v12, v13, vcc_lo                    // 0000000068e8: 02461b0c
	v_add_co_u32 v12, vcc_lo, v14, v4                          // 0000000068ec: d7006a0c 0202090e
	s_wait_alu depctr_va_vcc(0)                                // 0000000068f4: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v15, v5, vcc_lo             // 0000000068f8: d5207c0d 01aa0b0f
	v_add3_u32 v3, v3, v34, v33                                // 000000006900: d6550003 04864503
	v_mul_lo_u32 v33, s19, v40                                 // 000000006908: d72c0021 02025013
	v_mul_lo_u32 v34, s18, v41                                 // 000000006910: d72c0022 02025212
	v_mad_co_u64_u32 v[14:15], null, s18, v40, 0               // 000000006918: d6fe7c0e 02025012
	v_cmp_u_f32_e32 vcc_lo, v109, v109                         // 000000006920: 7c30db6d
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006924: 3e040481
	global_store_d16_hi_b16 v[12:13], v35, off                 // 000000006928: ee09407c 11800000 0000000c
	v_mul_lo_u32 v40, s18, v27                                 // 000000006934: d72c0028 02023612
	s_wait_alu depctr_va_vcc(0)                                // 00000000693c: bf88ff9d
	v_cndmask_b32_e32 v35, v38, v39, vcc_lo                    // 000000006940: 02464f26
	v_mul_lo_u32 v39, s19, v36                                 // 000000006944: d72c0027 02024813
	v_add3_u32 v15, v15, v34, v33                              // 00000000694c: d655000f 0486450f
	v_add_co_u32 v32, vcc_lo, s20, v2                          // 000000006954: d7006a20 02020414
	s_wait_alu depctr_va_vcc(0)                                // 00000000695c: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s21, v3, vcc_lo             // 000000006960: d5207c21 01aa0615
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000006968: bf8701d3
	v_lshlrev_b64_e32 v[2:3], 1, v[14:15]                      // 00000000696c: 3e041c81
	v_bfe_u32 v34, v108, 16, 1                                 // 000000006970: d6100022 0205216c
	v_add_co_u32 v14, vcc_lo, v32, v4                          // 000000006978: d7006a0e 02020920
	s_wait_alu depctr_va_vcc(0)                                // 000000006980: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v33, v5, vcc_lo             // 000000006984: d5207c0f 01aa0b21
	v_add3_u32 v32, v34, v108, 0x7fff                          // 00000000698c: d6550020 03fed922 00007fff
	v_add_co_u32 v34, vcc_lo, s20, v2                          // 000000006998: d7006a22 02020414
	s_wait_alu depctr_va_vcc(0)                                // 0000000069a0: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, s21, v3, vcc_lo             // 0000000069a4: d5207c26 01aa0615
	v_mad_co_u64_u32 v[2:3], null, s18, v36, 0                 // 0000000069ac: d6fe7c02 02024812
	v_or_b32_e32 v33, 0x400000, v108                           // 0000000069b4: 3842d8ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v108, v108                         // 0000000069bc: 7c30d96c
	global_store_d16_hi_b16 v[14:15], v35, off                 // 0000000069c0: ee09407c 11800000 0000000e
	s_wait_alu depctr_va_vcc(0)                                // 0000000069cc: bf88ff9d
	v_cndmask_b32_e32 v36, v32, v33, vcc_lo                    // 0000000069d0: 02484320
	v_add_co_u32 v32, vcc_lo, v34, v4                          // 0000000069d4: d7006a20 02020922
	s_wait_alu depctr_va_vcc(0)                                // 0000000069dc: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, v38, v5, vcc_lo             // 0000000069e0: d5207c21 01aa0b26
	v_add3_u32 v3, v3, v37, v39                                // 0000000069e8: d6550003 049e4b03
	v_mul_lo_u32 v37, s19, v30                                 // 0000000069f0: d72c0025 02023c13
	v_mul_lo_u32 v38, s18, v31                                 // 0000000069f8: d72c0026 02023e12
	v_mad_co_u64_u32 v[30:31], null, s18, v30, 0               // 000000006a00: d6fe7c1e 02023c12
	v_bfe_u32 v34, v106, 16, 1                                 // 000000006a08: d6100022 0205216a
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006a10: 3e040481
	v_or_b32_e32 v39, 0x400000, v106                           // 000000006a14: 384ed4ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v106, v106                         // 000000006a1c: 7c30d56a
	global_store_d16_hi_b16 v[32:33], v36, off                 // 000000006a20: ee09407c 12000000 00000020
	v_add3_u32 v34, v34, v106, 0x7fff                          // 000000006a2c: d6550022 03fed522 00007fff
	v_add3_u32 v31, v31, v38, v37                              // 000000006a38: d655001f 04964d1f
	v_bfe_u32 v37, v107, 16, 1                                 // 000000006a40: d6100025 0205216b
	s_wait_alu depctr_va_vcc(0)                                // 000000006a48: bf88ff9d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_4) | instid1(valu_dep_3)// 000000006a4c: bf8701d3
	v_cndmask_b32_e32 v34, v34, v39, vcc_lo                    // 000000006a50: 02444f22
	v_add_co_u32 v35, vcc_lo, s20, v2                          // 000000006a54: d7006a23 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006a5c: bf88ff9d
	v_add_co_ci_u32_e64 v36, null, s21, v3, vcc_lo             // 000000006a60: d5207c24 01aa0615
	v_lshlrev_b64_e32 v[2:3], 1, v[30:31]                      // 000000006a68: 3e043c81
	v_add_co_u32 v30, vcc_lo, v35, v4                          // 000000006a6c: d7006a1e 02020923
	s_wait_alu depctr_va_vcc(0)                                // 000000006a74: bf88ff9d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_4)// 000000006a78: bf870223
	v_add_co_ci_u32_e64 v31, null, v36, v5, vcc_lo             // 000000006a7c: d5207c1f 01aa0b24
	v_add3_u32 v35, v37, v107, 0x7fff                          // 000000006a84: d6550023 03fed725 00007fff
	v_add_co_u32 v37, vcc_lo, s20, v2                          // 000000006a90: d7006a25 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006a98: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, s21, v3, vcc_lo             // 000000006a9c: d5207c26 01aa0615
	v_mul_lo_u32 v39, s19, v26                                 // 000000006aa4: d72c0027 02023413
	v_mad_co_u64_u32 v[2:3], null, s18, v26, 0                 // 000000006aac: d6fe7c02 02023412
	v_or_b32_e32 v36, 0x400000, v107                           // 000000006ab4: 3848d6ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v107, v107                         // 000000006abc: 7c30d76b
	global_store_d16_hi_b16 v[30:31], v34, off                 // 000000006ac0: ee09407c 11000000 0000001e
	s_wait_alu depctr_va_vcc(0)                                // 000000006acc: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v36, vcc_lo                    // 000000006ad0: 02464923
	v_add_co_u32 v26, vcc_lo, v37, v4                          // 000000006ad4: d7006a1a 02020925
	s_wait_alu depctr_va_vcc(0)                                // 000000006adc: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v38, v5, vcc_lo             // 000000006ae0: d5207c1b 01aa0b26
	v_add3_u32 v3, v3, v40, v39                                // 000000006ae8: d6550003 049e5103
	v_mul_lo_u32 v37, s19, v24                                 // 000000006af0: d72c0025 02023013
	v_mul_lo_u32 v38, s18, v25                                 // 000000006af8: d72c0026 02023212
	v_mad_co_u64_u32 v[24:25], null, s18, v24, 0               // 000000006b00: d6fe7c18 02023012
	v_bfe_u32 v36, v105, 16, 1                                 // 000000006b08: d6100024 02052169
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006b10: 3e040481
	v_or_b32_e32 v39, 0x400000, v105                           // 000000006b14: 384ed2ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v105, v105                         // 000000006b1c: 7c30d369
	global_store_d16_hi_b16 v[26:27], v35, off                 // 000000006b20: ee09407c 11800000 0000001a
	v_add3_u32 v36, v36, v105, 0x7fff                          // 000000006b2c: d6550024 03fed324 00007fff
	v_mul_lo_u32 v40, s18, v21                                 // 000000006b38: d72c0028 02022a12
	v_add3_u32 v25, v25, v38, v37                              // 000000006b40: d6550019 04964d19
	v_bfe_u32 v37, v104, 16, 1                                 // 000000006b48: d6100025 02052168
	s_wait_alu depctr_va_vcc(0)                                // 000000006b50: bf88ff9d
	v_cndmask_b32_e32 v34, v36, v39, vcc_lo                    // 000000006b54: 02444f24
	v_add_co_u32 v35, vcc_lo, s20, v2                          // 000000006b58: d7006a23 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006b60: bf88ff9d
	v_add_co_ci_u32_e64 v36, null, s21, v3, vcc_lo             // 000000006b64: d5207c24 01aa0615
	v_lshlrev_b64_e32 v[2:3], 1, v[24:25]                      // 000000006b6c: 3e043081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000006b70: bf8701a3
	v_add_co_u32 v24, vcc_lo, v35, v4                          // 000000006b74: d7006a18 02020923
	s_wait_alu depctr_va_vcc(0)                                // 000000006b7c: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v36, v5, vcc_lo             // 000000006b80: d5207c19 01aa0b24
	v_add3_u32 v35, v37, v104, 0x7fff                          // 000000006b88: d6550023 03fed125 00007fff
	s_delay_alu instid0(valu_dep_4)                            // 000000006b94: bf870004
	v_add_co_u32 v37, vcc_lo, s20, v2                          // 000000006b98: d7006a25 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006ba0: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, s21, v3, vcc_lo             // 000000006ba4: d5207c26 01aa0615
	v_mul_lo_u32 v39, s19, v20                                 // 000000006bac: d72c0027 02022813
	v_mad_co_u64_u32 v[2:3], null, s18, v20, 0                 // 000000006bb4: d6fe7c02 02022812
	v_or_b32_e32 v36, 0x400000, v104                           // 000000006bbc: 3848d0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v104, v104                         // 000000006bc4: 7c30d168
	s_wait_alu depctr_va_vcc(0)                                // 000000006bc8: bf88ff9d
	s_delay_alu instid0(valu_dep_2)                            // 000000006bcc: bf870002
	v_cndmask_b32_e32 v35, v35, v36, vcc_lo                    // 000000006bd0: 02464923
	v_add_co_u32 v20, vcc_lo, v37, v4                          // 000000006bd4: d7006a14 02020925
	s_wait_alu depctr_va_vcc(0)                                // 000000006bdc: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v38, v5, vcc_lo             // 000000006be0: d5207c15 01aa0b26
	v_add3_u32 v3, v3, v40, v39                                // 000000006be8: d6550003 049e5103
	v_mul_lo_u32 v37, s19, v28                                 // 000000006bf0: d72c0025 02023813
	v_mul_lo_u32 v38, s18, v29                                 // 000000006bf8: d72c0026 02023a12
	v_mad_co_u64_u32 v[28:29], null, s18, v28, 0               // 000000006c00: d6fe7c1c 02023812
	v_bfe_u32 v36, v103, 16, 1                                 // 000000006c08: d6100024 02052167
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006c10: 3e040481
	v_or_b32_e32 v39, 0x400000, v103                           // 000000006c14: 384eceff 00400000
	v_cmp_u_f32_e32 vcc_lo, v103, v103                         // 000000006c1c: 7c30cf67
	global_store_d16_hi_b16 v[24:25], v34, off                 // 000000006c20: ee09407c 11000000 00000018
	v_add3_u32 v36, v36, v103, 0x7fff                          // 000000006c2c: d6550024 03fecf24 00007fff
	global_store_d16_hi_b16 v[20:21], v35, off                 // 000000006c38: ee09407c 11800000 00000014
	v_add3_u32 v29, v29, v38, v37                              // 000000006c44: d655001d 04964d1d
	v_bfe_u32 v37, v102, 16, 1                                 // 000000006c4c: d6100025 02052166
	v_mul_lo_u32 v40, s18, v23                                 // 000000006c54: d72c0028 02022e12
	s_wait_alu depctr_va_vcc(0)                                // 000000006c5c: bf88ff9d
	v_cndmask_b32_e32 v34, v36, v39, vcc_lo                    // 000000006c60: 02444f24
	v_add_co_u32 v35, vcc_lo, s20, v2                          // 000000006c64: d7006a23 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006c6c: bf88ff9d
	v_add_co_ci_u32_e64 v36, null, s21, v3, vcc_lo             // 000000006c70: d5207c24 01aa0615
	v_lshlrev_b64_e32 v[2:3], 1, v[28:29]                      // 000000006c78: 3e043881
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000006c7c: bf8701a3
	v_add_co_u32 v28, vcc_lo, v35, v4                          // 000000006c80: d7006a1c 02020923
	s_wait_alu depctr_va_vcc(0)                                // 000000006c88: bf88ff9d
	v_add_co_ci_u32_e64 v29, null, v36, v5, vcc_lo             // 000000006c8c: d5207c1d 01aa0b24
	v_add3_u32 v35, v37, v102, 0x7fff                          // 000000006c94: d6550023 03fecd25 00007fff
	s_delay_alu instid0(valu_dep_4)                            // 000000006ca0: bf870004
	v_add_co_u32 v37, vcc_lo, s20, v2                          // 000000006ca4: d7006a25 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006cac: bf88ff9d
	v_add_co_ci_u32_e64 v38, null, s21, v3, vcc_lo             // 000000006cb0: d5207c26 01aa0615
	v_mul_lo_u32 v39, s19, v22                                 // 000000006cb8: d72c0027 02022c13
	v_mad_co_u64_u32 v[2:3], null, s18, v22, 0                 // 000000006cc0: d6fe7c02 02022c12
	v_or_b32_e32 v36, 0x400000, v102                           // 000000006cc8: 3848ccff 00400000
	v_cmp_u_f32_e32 vcc_lo, v102, v102                         // 000000006cd0: 7c30cd66
	s_wait_alu depctr_va_vcc(0)                                // 000000006cd4: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_4)// 000000006cd8: bf870212
	v_cndmask_b32_e32 v35, v35, v36, vcc_lo                    // 000000006cdc: 02464923
	v_add3_u32 v3, v3, v40, v39                                // 000000006ce0: d6550003 049e5103
	v_add_co_u32 v22, vcc_lo, v37, v4                          // 000000006ce8: d7006a16 02020925
	v_bfe_u32 v36, v101, 16, 1                                 // 000000006cf0: d6100024 02052165
	s_wait_alu depctr_va_vcc(0)                                // 000000006cf8: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, v38, v5, vcc_lo             // 000000006cfc: d5207c17 01aa0b26
	v_mul_lo_u32 v37, s19, v18                                 // 000000006d04: d72c0025 02022413
	v_mul_lo_u32 v38, s18, v19                                 // 000000006d0c: d72c0026 02022612
	v_mad_co_u64_u32 v[18:19], null, s18, v18, 0               // 000000006d14: d6fe7c12 02022412
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]                        // 000000006d1c: 3e040481
	v_add3_u32 v36, v36, v101, 0x7fff                          // 000000006d20: d6550024 03fecb24 00007fff
	v_or_b32_e32 v39, 0x400000, v101                           // 000000006d2c: 384ecaff 00400000
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 000000006d34: 7c30cb65
	s_clause 0x1                                               // 000000006d38: bf850001
	global_store_d16_hi_b16 v[28:29], v34, off                 // 000000006d3c: ee09407c 11000000 0000001c
	global_store_d16_hi_b16 v[22:23], v35, off                 // 000000006d48: ee09407c 11800000 00000016
	v_bfe_u32 v35, v100, 16, 1                                 // 000000006d54: d6100023 02052164
	v_mul_lo_u32 v40, s18, v17                                 // 000000006d5c: d72c0028 02022212
	v_add3_u32 v19, v19, v38, v37                              // 000000006d64: d6550013 04964d13
	s_wait_alu depctr_va_vcc(0)                                // 000000006d6c: bf88ff9d
	v_cndmask_b32_e32 v34, v36, v39, vcc_lo                    // 000000006d70: 02444f24
	v_add_co_u32 v36, vcc_lo, s20, v2                          // 000000006d74: d7006a24 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006d7c: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s21, v3, vcc_lo             // 000000006d80: d5207c25 01aa0615
	v_lshlrev_b64_e32 v[2:3], 1, v[18:19]                      // 000000006d88: 3e042481
	v_mul_lo_u32 v39, s19, v16                                 // 000000006d8c: d72c0027 02022013
	v_mad_co_u64_u32 v[16:17], null, s18, v16, 0               // 000000006d94: d6fe7c10 02022012
	v_add_co_u32 v18, vcc_lo, v36, v4                          // 000000006d9c: d7006a12 02020924
	v_add3_u32 v35, v35, v100, 0x7fff                          // 000000006da4: d6550023 03fec923 00007fff
	v_or_b32_e32 v38, 0x400000, v100                           // 000000006db0: 384cc8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006db8: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v37, v5, vcc_lo             // 000000006dbc: d5207c13 01aa0b25
	v_cmp_u_f32_e32 vcc_lo, v100, v100                         // 000000006dc4: 7c30c964
	v_add3_u32 v17, v17, v40, v39                              // 000000006dc8: d6550011 049e5111
	v_bfe_u32 v36, v99, 16, 1                                  // 000000006dd0: d6100024 02052163
	v_or_b32_e32 v37, 0x400000, v99                            // 000000006dd8: 384ac6ff 00400000
	global_store_d16_hi_b16 v[18:19], v34, off                 // 000000006de0: ee09407c 11000000 00000012
	s_wait_alu depctr_va_vcc(0)                                // 000000006dec: bf88ff9d
	v_cndmask_b32_e32 v35, v35, v38, vcc_lo                    // 000000006df0: 02464d23
	v_add_co_u32 v2, vcc_lo, s20, v2                           // 000000006df4: d7006a02 02020414
	s_wait_alu depctr_va_vcc(0)                                // 000000006dfc: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo              // 000000006e00: d5207c03 01aa0615
	v_lshlrev_b64_e32 v[16:17], 1, v[16:17]                    // 000000006e08: 3e202081
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000006e0c: bf8701a3
	v_add_co_u32 v2, vcc_lo, v2, v4                            // 000000006e10: d7006a02 02020902
	s_wait_alu depctr_va_vcc(0)                                // 000000006e18: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v3, v5, vcc_lo               // 000000006e1c: d5207c03 01aa0b03
	v_add3_u32 v36, v36, v99, 0x7fff                           // 000000006e24: d6550024 03fec724 00007fff
	v_cmp_u_f32_e32 vcc_lo, v99, v99                           // 000000006e30: 7c30c763
	global_store_d16_hi_b16 v[2:3], v35, off                   // 000000006e34: ee09407c 11800000 00000002
	v_bfe_u32 v35, v98, 16, 1                                  // 000000006e40: d6100023 02052162
	s_wait_alu depctr_va_vcc(0)                                // 000000006e48: bf88ff9d
	v_cndmask_b32_e32 v34, v36, v37, vcc_lo                    // 000000006e4c: 02444b24
	v_add_co_u32 v16, vcc_lo, s20, v16                         // 000000006e50: d7006a10 02022014
	s_wait_alu depctr_va_vcc(0)                                // 000000006e58: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo            // 000000006e5c: d5207c11 01aa2215
	v_add3_u32 v35, v35, v98, 0x7fff                           // 000000006e64: d6550023 03fec523 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000006e70: bf870003
	v_add_co_u32 v4, vcc_lo, v16, v4                           // 000000006e74: d7006a04 02020910
	v_or_b32_e32 v36, 0x400000, v98                            // 000000006e7c: 3848c4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006e84: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v17, v5, vcc_lo              // 000000006e88: d5207c05 01aa0b11
	v_bfe_u32 v16, v97, 16, 1                                  // 000000006e90: d6100010 02052161
	v_cmp_u_f32_e32 vcc_lo, v98, v98                           // 000000006e98: 7c30c562
	global_store_d16_hi_b16 v[4:5], v34, off                   // 000000006e9c: ee09407c 11000000 00000004
	v_or_b32_e32 v34, 0x400000, v97                            // 000000006ea8: 3844c2ff 00400000
	v_add3_u32 v16, v16, v97, 0x7fff                           // 000000006eb0: d6550010 03fec310 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006ebc: bf88ff9d
	v_cndmask_b32_e32 v17, v35, v36, vcc_lo                    // 000000006ec0: 02224923
	v_bfe_u32 v35, v96, 16, 1                                  // 000000006ec4: d6100023 02052160
	v_cmp_u_f32_e32 vcc_lo, v97, v97                           // 000000006ecc: 7c30c361
	global_store_d16_hi_b16 v[0:1], v17, off offset:32         // 000000006ed0: ee09407c 08800000 00002000
	v_add3_u32 v0, v35, v96, 0x7fff                            // 000000006edc: d6550000 03fec123 00007fff
	v_or_b32_e32 v1, 0x400000, v96                             // 000000006ee8: 3802c0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006ef0: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v34, vcc_lo                    // 000000006ef4: 02204510
	v_bfe_u32 v17, v95, 16, 1                                  // 000000006ef8: d6100011 0205215f
	v_cmp_u_f32_e32 vcc_lo, v96, v96                           // 000000006f00: 7c30c160
	global_store_d16_hi_b16 v[6:7], v16, off offset:32         // 000000006f04: ee09407c 08000000 00002006
	v_add3_u32 v6, v17, v95, 0x7fff                            // 000000006f10: d6550006 03febf11 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006f1c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006f20: 02000300
	v_bfe_u32 v1, v94, 16, 1                                   // 000000006f24: d6100001 0205215e
	v_or_b32_e32 v7, 0x400000, v95                             // 000000006f2c: 380ebeff 00400000
	v_cmp_u_f32_e32 vcc_lo, v95, v95                           // 000000006f34: 7c30bf5f
	global_store_d16_hi_b16 v[8:9], v0, off offset:32          // 000000006f38: ee09407c 00000000 00002008
	v_add3_u32 v0, v1, v94, 0x7fff                             // 000000006f44: d6550000 03febd01 00007fff
	v_or_b32_e32 v1, 0x400000, v94                             // 000000006f50: 3802bcff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006f58: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000006f5c: 020c0f06
	v_bfe_u32 v7, v92, 16, 1                                   // 000000006f60: d6100007 0205215c
	v_cmp_u_f32_e32 vcc_lo, v94, v94                           // 000000006f68: 7c30bd5e
	v_bfe_u32 v8, v84, 16, 1                                   // 000000006f6c: d6100008 02052154
	v_or_b32_e32 v9, 0x400000, v85                             // 000000006f74: 3812aaff 00400000
	global_store_d16_hi_b16 v[10:11], v6, off offset:32        // 000000006f7c: ee09407c 03000000 0000200a
	v_add3_u32 v6, v7, v92, 0x7fff                             // 000000006f88: d6550006 03feb907 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000006f94: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000006f98: 02000300
	v_bfe_u32 v1, v91, 16, 1                                   // 000000006f9c: d6100001 0205215b
	v_or_b32_e32 v7, 0x400000, v92                             // 000000006fa4: 380eb8ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v92, v92                           // 000000006fac: 7c30b95c
	v_add3_u32 v8, v8, v84, 0x7fff                             // 000000006fb0: d6550008 03fea908 00007fff
	global_store_d16_hi_b16 v[12:13], v0, off offset:32        // 000000006fbc: ee09407c 00000000 0000200c
	v_add3_u32 v0, v1, v91, 0x7fff                             // 000000006fc8: d6550000 03feb701 00007fff
	v_or_b32_e32 v1, 0x400000, v91                             // 000000006fd4: 3802b6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000006fdc: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000006fe0: 020c0f06
	v_bfe_u32 v7, v88, 16, 1                                   // 000000006fe4: d6100007 02052158
	v_cmp_u_f32_e32 vcc_lo, v91, v91                           // 000000006fec: 7c30b75b
	v_or_b32_e32 v10, 0x400000, v84                            // 000000006ff0: 3814a8ff 00400000
	v_or_b32_e32 v11, 0x400000, v52                            // 000000006ff8: 381668ff 00400000
	global_store_d16_hi_b16 v[14:15], v6, off offset:32        // 000000007000: ee09407c 03000000 0000200e
	v_add3_u32 v6, v7, v88, 0x7fff                             // 00000000700c: d6550006 03feb107 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000007018: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000701c: 02000300
	v_bfe_u32 v1, v93, 16, 1                                   // 000000007020: d6100001 0205215d
	v_or_b32_e32 v7, 0x400000, v88                             // 000000007028: 380eb0ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v88, v88                           // 000000007030: 7c30b158
	global_store_d16_hi_b16 v[32:33], v0, off offset:32        // 000000007034: ee09407c 00000000 00002020
	v_add3_u32 v0, v1, v93, 0x7fff                             // 000000007040: d6550000 03febb01 00007fff
	v_or_b32_e32 v1, 0x400000, v93                             // 00000000704c: 3802baff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000007054: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000007058: 020c0f06
	v_bfe_u32 v7, v90, 16, 1                                   // 00000000705c: d6100007 0205215a
	v_cmp_u_f32_e32 vcc_lo, v93, v93                           // 000000007064: 7c30bb5d
	global_store_d16_hi_b16 v[30:31], v6, off offset:32        // 000000007068: ee09407c 03000000 0000201e
	v_add3_u32 v6, v7, v90, 0x7fff                             // 000000007074: d6550006 03feb507 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000007080: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000007084: 02000300
	v_bfe_u32 v1, v89, 16, 1                                   // 000000007088: d6100001 02052159
	v_or_b32_e32 v7, 0x400000, v90                             // 000000007090: 380eb4ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v90, v90                           // 000000007098: 7c30b55a
	global_store_d16_hi_b16 v[26:27], v0, off offset:32        // 00000000709c: ee09407c 00000000 0000201a
	v_add3_u32 v0, v1, v89, 0x7fff                             // 0000000070a8: d6550000 03feb301 00007fff
	v_or_b32_e32 v1, 0x400000, v89                             // 0000000070b4: 3802b2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000070bc: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 0000000070c0: 020c0f06
	v_bfe_u32 v7, v87, 16, 1                                   // 0000000070c4: d6100007 02052157
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 0000000070cc: 7c30b359
	global_store_d16_hi_b16 v[24:25], v6, off offset:32        // 0000000070d0: ee09407c 03000000 00002018
	v_add3_u32 v6, v7, v87, 0x7fff                             // 0000000070dc: d6550006 03feaf07 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000070e8: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000070ec: 02000300
	v_bfe_u32 v1, v86, 16, 1                                   // 0000000070f0: d6100001 02052156
	v_or_b32_e32 v7, 0x400000, v87                             // 0000000070f8: 380eaeff 00400000
	v_cmp_u_f32_e32 vcc_lo, v87, v87                           // 000000007100: 7c30af57
	global_store_d16_hi_b16 v[20:21], v0, off offset:32        // 000000007104: ee09407c 00000000 00002014
	v_add3_u32 v0, v1, v86, 0x7fff                             // 000000007110: d6550000 03fead01 00007fff
	v_or_b32_e32 v1, 0x400000, v86                             // 00000000711c: 3802acff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000007124: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v7, vcc_lo                       // 000000007128: 020c0f06
	v_bfe_u32 v7, v85, 16, 1                                   // 00000000712c: d6100007 02052155
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 000000007134: 7c30ad56
	s_delay_alu instid0(valu_dep_2)                            // 000000007138: bf870002
	v_add3_u32 v7, v7, v85, 0x7fff                             // 00000000713c: d6550007 03feab07 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000007148: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000714c: 02000300
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 000000007150: 7c30ab55
	v_bfe_u32 v1, v52, 16, 1                                   // 000000007154: d6100001 02052134
	s_wait_alu depctr_va_vcc(0)                                // 00000000715c: bf88ff9d
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 000000007160: 020e1307
	v_cmp_u_f32_e32 vcc_lo, v84, v84                           // 000000007164: 7c30a954
	s_delay_alu instid0(valu_dep_3)                            // 000000007168: bf870003
	v_add3_u32 v1, v1, v52, 0x7fff                             // 00000000716c: d6550001 03fe6901 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000007178: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v10, vcc_lo                      // 00000000717c: 02101508
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 000000007180: 7c306934
	s_wait_alu depctr_va_vcc(0)                                // 000000007184: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v11, vcc_lo                      // 000000007188: 02021701
	s_clause 0x4                                               // 00000000718c: bf850004
	global_store_d16_hi_b16 v[28:29], v6, off offset:32        // 000000007190: ee09407c 03000000 0000201c
	global_store_d16_hi_b16 v[22:23], v0, off offset:32        // 00000000719c: ee09407c 00000000 00002016
	global_store_d16_hi_b16 v[18:19], v7, off offset:32        // 0000000071a8: ee09407c 03800000 00002012
	global_store_d16_hi_b16 v[2:3], v8, off offset:32          // 0000000071b4: ee09407c 04000000 00002002
	global_store_d16_hi_b16 v[4:5], v1, off offset:32          // 0000000071c0: ee09407c 00800000 00002004
	s_nop 0                                                    // 0000000071cc: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 0000000071d0: bfb60003
	s_endpgm                                                   // 0000000071d4: bfb00000
	s_code_end                                                 // 0000000071d8: bf9f0000
	s_code_end                                                 // 0000000071dc: bf9f0000
	s_code_end                                                 // 0000000071e0: bf9f0000
	s_code_end                                                 // 0000000071e4: bf9f0000
	s_code_end                                                 // 0000000071e8: bf9f0000
	s_code_end                                                 // 0000000071ec: bf9f0000
	s_code_end                                                 // 0000000071f0: bf9f0000
	s_code_end                                                 // 0000000071f4: bf9f0000
	s_code_end                                                 // 0000000071f8: bf9f0000
	s_code_end                                                 // 0000000071fc: bf9f0000
	s_code_end                                                 // 000000007200: bf9f0000
	s_code_end                                                 // 000000007204: bf9f0000
	s_code_end                                                 // 000000007208: bf9f0000
	s_code_end                                                 // 00000000720c: bf9f0000
	s_code_end                                                 // 000000007210: bf9f0000
	s_code_end                                                 // 000000007214: bf9f0000
	s_code_end                                                 // 000000007218: bf9f0000
	s_code_end                                                 // 00000000721c: bf9f0000
	s_code_end                                                 // 000000007220: bf9f0000
	s_code_end                                                 // 000000007224: bf9f0000
	s_code_end                                                 // 000000007228: bf9f0000
	s_code_end                                                 // 00000000722c: bf9f0000
	s_code_end                                                 // 000000007230: bf9f0000
	s_code_end                                                 // 000000007234: bf9f0000
	s_code_end                                                 // 000000007238: bf9f0000
	s_code_end                                                 // 00000000723c: bf9f0000
	s_code_end                                                 // 000000007240: bf9f0000
	s_code_end                                                 // 000000007244: bf9f0000
	s_code_end                                                 // 000000007248: bf9f0000
	s_code_end                                                 // 00000000724c: bf9f0000
	s_code_end                                                 // 000000007250: bf9f0000
	s_code_end                                                 // 000000007254: bf9f0000
	s_code_end                                                 // 000000007258: bf9f0000
	s_code_end                                                 // 00000000725c: bf9f0000
	s_code_end                                                 // 000000007260: bf9f0000
	s_code_end                                                 // 000000007264: bf9f0000
	s_code_end                                                 // 000000007268: bf9f0000
	s_code_end                                                 // 00000000726c: bf9f0000
	s_code_end                                                 // 000000007270: bf9f0000
	s_code_end                                                 // 000000007274: bf9f0000
	s_code_end                                                 // 000000007278: bf9f0000
	s_code_end                                                 // 00000000727c: bf9f0000
	s_code_end                                                 // 000000007280: bf9f0000
	s_code_end                                                 // 000000007284: bf9f0000
	s_code_end                                                 // 000000007288: bf9f0000
	s_code_end                                                 // 00000000728c: bf9f0000
	s_code_end                                                 // 000000007290: bf9f0000
	s_code_end                                                 // 000000007294: bf9f0000
	s_code_end                                                 // 000000007298: bf9f0000
	s_code_end                                                 // 00000000729c: bf9f0000
	s_code_end                                                 // 0000000072a0: bf9f0000
	s_code_end                                                 // 0000000072a4: bf9f0000
	s_code_end                                                 // 0000000072a8: bf9f0000
	s_code_end                                                 // 0000000072ac: bf9f0000
	s_code_end                                                 // 0000000072b0: bf9f0000
	s_code_end                                                 // 0000000072b4: bf9f0000
	s_code_end                                                 // 0000000072b8: bf9f0000
	s_code_end                                                 // 0000000072bc: bf9f0000
	s_code_end                                                 // 0000000072c0: bf9f0000
	s_code_end                                                 // 0000000072c4: bf9f0000
	s_code_end                                                 // 0000000072c8: bf9f0000
	s_code_end                                                 // 0000000072cc: bf9f0000
	s_code_end                                                 // 0000000072d0: bf9f0000
	s_code_end                                                 // 0000000072d4: bf9f0000
	s_code_end                                                 // 0000000072d8: bf9f0000
	s_code_end                                                 // 0000000072dc: bf9f0000
	s_code_end                                                 // 0000000072e0: bf9f0000
	s_code_end                                                 // 0000000072e4: bf9f0000
	s_code_end                                                 // 0000000072e8: bf9f0000
	s_code_end                                                 // 0000000072ec: bf9f0000
	s_code_end                                                 // 0000000072f0: bf9f0000
	s_code_end                                                 // 0000000072f4: bf9f0000
	s_code_end                                                 // 0000000072f8: bf9f0000
	s_code_end                                                 // 0000000072fc: bf9f0000
	s_code_end                                                 // 000000007300: bf9f0000
	s_code_end                                                 // 000000007304: bf9f0000
	s_code_end                                                 // 000000007308: bf9f0000
	s_code_end                                                 // 00000000730c: bf9f0000
	s_code_end                                                 // 000000007310: bf9f0000
	s_code_end                                                 // 000000007314: bf9f0000
	s_code_end                                                 // 000000007318: bf9f0000
	s_code_end                                                 // 00000000731c: bf9f0000
	s_code_end                                                 // 000000007320: bf9f0000
	s_code_end                                                 // 000000007324: bf9f0000
	s_code_end                                                 // 000000007328: bf9f0000
	s_code_end                                                 // 00000000732c: bf9f0000
	s_code_end                                                 // 000000007330: bf9f0000
	s_code_end                                                 // 000000007334: bf9f0000
	s_code_end                                                 // 000000007338: bf9f0000
	s_code_end                                                 // 00000000733c: bf9f0000
	s_code_end                                                 // 000000007340: bf9f0000
	s_code_end                                                 // 000000007344: bf9f0000
	s_code_end                                                 // 000000007348: bf9f0000
	s_code_end                                                 // 00000000734c: bf9f0000
	s_code_end                                                 // 000000007350: bf9f0000
	s_code_end                                                 // 000000007354: bf9f0000
	s_code_end                                                 // 000000007358: bf9f0000
	s_code_end                                                 // 00000000735c: bf9f0000
	s_code_end                                                 // 000000007360: bf9f0000
	s_code_end                                                 // 000000007364: bf9f0000
	s_code_end                                                 // 000000007368: bf9f0000
	s_code_end                                                 // 00000000736c: bf9f0000
	s_code_end                                                 // 000000007370: bf9f0000
	s_code_end                                                 // 000000007374: bf9f0000
	s_code_end                                                 // 000000007378: bf9f0000
	s_code_end                                                 // 00000000737c: bf9f0000
