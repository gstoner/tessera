
/tmp/tmp5724k7_2.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_afded1274d72297c>:
	s_clause 0x1                                               // 000000001b00: bf850001
	s_load_b64 s[8:9], s[0:1], 0xd8                            // 000000001b04: f4002200 f80000d8
	s_load_b64 s[12:13], s[0:1], 0x8                           // 000000001b0c: f4002300 f8000008
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b14: 320a0081
	s_mov_b32 s2, ttmp9                                        // 000000001b18: be820075
	s_mov_b32 s10, ttmp7                                       // 000000001b1c: be8a0073
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b20: 86039f75
	s_ashr_i32 s11, ttmp7, 31                                  // 000000001b24: 860b9f73
	s_clause 0x3                                               // 000000001b28: bf850003
	s_load_b64 s[14:15], s[0:1], 0x30                          // 000000001b2c: f4002380 f8000030
	s_load_b64 s[4:5], s[0:1], 0x58                            // 000000001b34: f4002100 f8000058
	s_load_b64 s[20:21], s[0:1], 0x80                          // 000000001b3c: f4002500 f8000080
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b44: f4004400 f80000c8
	s_lshl_b64 s[6:7], s[2:3], 7                               // 000000001b4c: 84868702
	s_lshl_b64 s[2:3], s[10:11], 7                             // 000000001b50: 8482870a
	v_and_b32_e32 v6, 0x60, v5                                 // 000000001b54: 360c0aff 00000060
	v_or_b32_e32 v1, s2, v5                                    // 000000001b5c: 38020a02
	v_lshlrev_b32_e32 v4, 4, v0                                // 000000001b60: 30080084
	v_mov_b32_e32 v14, 0                                       // 000000001b64: 7e1c0280
	v_mul_u32_u24_e32 v10, 48, v5                              // 000000001b68: 16140ab0
	v_dual_mov_b32 v105, 0 :: v_dual_and_b32 v28, 8, v5        // 000000001b6c: ca240080 691c0a88
	s_delay_alu instid0(valu_dep_4)                            // 000000001b74: bf870004
	v_and_b32_e32 v11, 16, v4                                  // 000000001b78: 36160890
	v_or_b32_e32 v4, s6, v5                                    // 000000001b7c: 38080a06
	v_mov_b32_e32 v101, 0                                      // 000000001b80: 7eca0280
	s_wait_kmcnt 0x0                                           // 000000001b84: bfc70000
	v_mul_lo_u32 v9, s9, v1                                    // 000000001b88: d72c0009 02020209
	v_mad_co_u64_u32 v[2:3], null, s8, v1, s[12:13]            // 000000001b90: d6fe7c02 00320208
	v_mov_b32_e32 v1, s3                                       // 000000001b98: 7e020203
	v_or_b32_e32 v7, 16, v6                                    // 000000001b9c: 380e0c90
	v_lshlrev_b32_e32 v8, 1, v0                                // 000000001ba0: 30100081
	v_or_b32_e32 v26, s2, v6                                   // 000000001ba4: 38340c02
	v_and_b32_e32 v0, 15, v0                                   // 000000001ba8: 3600008f
	v_add_nc_u32_e32 v15, v10, v11                             // 000000001bac: 4a1e170a
	v_or_b32_e32 v36, s2, v7                                   // 000000001bb0: 38480e02
	s_mul_i32 s2, s8, s3                                       // 000000001bb4: 96020308
	v_mul_lo_u32 v10, s9, v4                                   // 000000001bb8: d72c000a 02020809
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bc0: bf88ff9e
	v_add3_u32 v9, v9, v3, s2                                  // 000000001bc4: d6550009 000a0709
	v_mad_co_u64_u32 v[3:4], null, s8, v4, s[14:15]            // 000000001bcc: d6fe7c03 003a0808
	v_or_b32_e32 v6, v6, v0                                    // 000000001bd4: 380c0106
	s_mul_i32 s2, s8, s7                                       // 000000001bd8: 96020708
	v_add_co_u32 v16, vcc_lo, v2, v11                          // 000000001bdc: d7006a10 02021702
	s_delay_alu instid0(valu_dep_1)                            // 000000001be4: bf870001
	v_add_co_ci_u32_e64 v17, null, 0, v9, vcc_lo               // 000000001be8: d5207c11 01aa1280
	v_or_b32_e32 v31, 1, v28                                   // 000000001bf0: 383e3881
	s_wait_alu depctr_sa_sdst(0)                               // 000000001bf4: bf88ff9e
	v_add3_u32 v2, v10, v4, s2                                 // 000000001bf8: d6550002 000a090a
	v_mul_u32_u24_e32 v4, 48, v6                               // 000000001c00: 16080cb0
	v_and_or_b32 v6, v8, 64, v0                                // 000000001c04: d6570006 04018108
	v_add_co_u32 v18, vcc_lo, v3, v11                          // 000000001c0c: d7006a12 02021703
	s_wait_alu depctr_va_vcc(0)                                // 000000001c14: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, 0, v2, vcc_lo               // 000000001c18: d5207c13 01aa0480
	s_delay_alu instid0(valu_dep_3)                            // 000000001c20: bf870003
	v_or_b32_e32 v30, 32, v6                                   // 000000001c24: 383c0ca0
	v_or_b32_e32 v32, 48, v6                                   // 000000001c28: 38400cb0
	v_or_b32_e32 v20, v4, v28                                  // 000000001c2c: 38283904
	v_or_b32_e32 v29, 16, v6                                   // 000000001c30: 383a0c90
	s_lshr_b64 s[22:23], s[8:9], 5                             // 000000001c34: 85968508
	v_mul_u32_u24_e32 v3, 48, v30                              // 000000001c38: 16063cb0
	v_mul_u32_u24_e32 v4, 48, v32                              // 000000001c3c: 160840b0
	s_lshr_b32 s8, s9, 5                                       // 000000001c40: 85088509
	v_mul_u32_u24_e32 v2, 48, v29                              // 000000001c44: 16043ab0
	v_mov_b32_e32 v97, 0                                       // 000000001c48: 7ec20280
	v_or_b32_e32 v39, v3, v28                                  // 000000001c4c: 384e3903
	v_mov_b32_e32 v3, s7                                       // 000000001c50: 7e060207
	v_or_b32_e32 v7, v7, v0                                    // 000000001c54: 380e0107
	v_mul_u32_u24_e32 v0, 48, v6                               // 000000001c58: 16000cb0
	v_or_b32_e32 v40, v4, v28                                  // 000000001c5c: 38503904
	v_or_b32_e32 v4, v31, v26                                  // 000000001c60: 3808351f
	v_dual_mov_b32 v50, 0 :: v_dual_add_nc_u32 v119, 0x1800, v39// 000000001c64: ca200080 32764eff 00001800
	s_delay_alu instid0(valu_dep_4) | instskip(skip_3) | instid1(valu_dep_4)// 000000001c70: bf870244
	v_or_b32_e32 v37, v0, v28                                  // 000000001c74: 384a3900
	v_or_b32_e32 v0, v26, v28                                  // 000000001c78: 3800391a
	v_mul_u32_u24_e32 v5, 48, v7                               // 000000001c7c: 160a0eb0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v94, 0             // 000000001c80: ca100080 5d5e0080
	v_dual_mov_b32 v54, 0 :: v_dual_add_nc_u32 v117, 0x1800, v37// 000000001c88: ca200080 36744aff 00001800
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 000000001c94: bf870214
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000001c98: 7ca80010
	v_or_b32_e32 v21, v5, v28                                  // 000000001c9c: 382a3905
	v_mov_b32_e32 v5, s3                                       // 000000001ca0: 7e0a0203
	v_mov_b32_e32 v13, s3                                      // 000000001ca4: 7e1a0203
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v92, 0             // 000000001ca8: ca100080 4b5c0080
	s_wait_alu depctr_va_vcc(0)                                // 000000001cb0: bf88ff9d
	v_dual_cndmask_b32 v7, 0, v1 :: v_dual_cndmask_b32 v8, 0, v0// 000000001cb4: ca520280 07080080
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000001cbc: 7ca80810
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v90, 0             // 000000001cc0: ca100080 475a0080
	s_delay_alu instid0(valu_dep_3)                            // 000000001cc8: bf870003
	v_mul_lo_u32 v12, s22, v7                                  // 000000001ccc: d72c000c 02020e16
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cd4: bf88ff9e
	v_mul_lo_u32 v9, s8, v8                                    // 000000001cd8: d72c0009 02021008
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v86, 0             // 000000001ce0: ca100080 45560080
	s_wait_alu depctr_va_vcc(0)                                // 000000001ce8: bf88ff9d
	v_cndmask_b32_e32 v11, 0, v4, vcc_lo                       // 000000001cec: 02160880
	v_or_b32_e32 v38, v2, v28                                  // 000000001cf0: 384c3902
	v_or_b32_e32 v2, s6, v6                                    // 000000001cf4: 38040c06
	v_or_b32_e32 v33, 2, v28                                   // 000000001cf8: 38423882
	v_mad_co_u64_u32 v[6:7], null, s22, v8, 0                  // 000000001cfc: d6fe7c06 02021016
	v_or_b32_e32 v34, 3, v28                                   // 000000001d04: 38443883
	v_cndmask_b32_e32 v10, 0, v5, vcc_lo                       // 000000001d08: 02140a80
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[2:3]                  // 000000001d0c: 7ca80412
	v_or_b32_e32 v4, v33, v26                                  // 000000001d10: 38083521
	v_mul_lo_u32 v22, s8, v11                                  // 000000001d14: d72c0016 02021608
	v_or_b32_e32 v35, 4, v28                                   // 000000001d1c: 38463884
	v_mul_lo_u32 v23, s22, v10                                 // 000000001d20: d72c0017 02021416
	v_add3_u32 v7, v7, v12, v9                                 // 000000001d28: d6550007 04261907
	v_or_b32_e32 v12, v34, v26                                 // 000000001d30: 38183522
	v_cmp_gt_i64_e64 s2, s[16:17], v[4:5]                      // 000000001d34: d4540002 02020810
	s_wait_alu depctr_va_vcc(0)                                // 000000001d3c: bf88ff9d
	v_cndmask_b32_e32 v8, 0, v2, vcc_lo                        // 000000001d40: 02100480
	v_cndmask_b32_e64 v9, 0, s7, vcc_lo                        // 000000001d44: d5010009 01a80e80
	v_mad_co_u64_u32 v[10:11], null, s22, v11, 0               // 000000001d4c: d6fe7c0a 02021616
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000001d54: 7ca81810
	v_or_b32_e32 v43, 6, v28                                   // 000000001d58: 38563886
	s_wait_alu depctr_va_sdst(0)                               // 000000001d5c: bf88f19f
	v_cndmask_b32_e64 v24, 0, v5, s2                           // 000000001d60: d5010018 000a0a80
	v_cndmask_b32_e64 v25, 0, v4, s2                           // 000000001d68: d5010019 000a0880
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000001d70: 3e080c82
	v_or_b32_e32 v44, 7, v28                                   // 000000001d74: 38583887
	s_wait_alu depctr_va_vcc(0)                                // 000000001d78: bf88ff9d
	v_cndmask_b32_e32 v41, 0, v12, vcc_lo                      // 000000001d7c: 02521880
	v_or_b32_e32 v12, v35, v26                                 // 000000001d80: 38183523
	v_add3_u32 v11, v11, v23, v22                              // 000000001d84: d655000b 045a2f0b
	v_mul_lo_u32 v27, s8, v25                                  // 000000001d8c: d72c001b 02023208
	v_mul_lo_u32 v24, s22, v24                                 // 000000001d94: d72c0018 02023016
	v_mad_co_u64_u32 v[6:7], null, s22, v25, 0                 // 000000001d9c: d6fe7c06 02023216
	v_cndmask_b32_e32 v25, 0, v13, vcc_lo                      // 000000001da4: 02321a80
	v_add_co_u32 v22, vcc_lo, s4, v4                           // 000000001da8: d7006a16 02020804
	s_wait_alu depctr_va_vcc(0)                                // 000000001db0: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s5, v5, vcc_lo              // 000000001db4: d5207c17 01aa0a05
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000001dbc: 7ca81810
	v_lshlrev_b64_e32 v[4:5], 2, v[10:11]                      // 000000001dc0: 3e081482
	v_add3_u32 v7, v7, v24, v27                                // 000000001dc4: d6550007 046e3107
	v_mul_lo_u32 v42, s22, v25                                 // 000000001dcc: d72c002a 02023216
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v82, 0             // 000000001dd4: ca100080 43520080
	s_wait_alu depctr_va_vcc(0)                                // 000000001ddc: bf88ff9d
	v_cndmask_b32_e32 v12, 0, v12, vcc_lo                      // 000000001de0: 02181880
	v_mul_lo_u32 v27, s8, v41                                  // 000000001de4: d72c001b 02025208
	v_mad_co_u64_u32 v[10:11], null, s22, v41, 0               // 000000001dec: d6fe7c0a 02025216
	v_cndmask_b32_e32 v13, 0, v13, vcc_lo                      // 000000001df4: 021a1a80
	v_add_co_u32 v24, vcc_lo, s4, v4                           // 000000001df8: d7006a18 02020804
	v_or_b32_e32 v41, 5, v28                                   // 000000001e00: 38523885
	s_wait_alu depctr_va_vcc(0)                                // 000000001e04: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, s5, v5, vcc_lo              // 000000001e08: d5207c19 01aa0a05
	v_lshlrev_b64_e32 v[4:5], 2, v[6:7]                        // 000000001e10: 3e080c82
	v_add3_u32 v11, v11, v42, v27                              // 000000001e14: d655000b 046e550b
	v_mul_lo_u32 v27, s8, v12                                  // 000000001e1c: d72c001b 02021808
	v_mul_lo_u32 v42, s22, v13                                 // 000000001e24: d72c002a 02021a16
	v_mad_co_u64_u32 v[6:7], null, s22, v12, 0                 // 000000001e2c: d6fe7c06 02021816
	v_or_b32_e32 v12, v41, v26                                 // 000000001e34: 38183529
	v_mov_b32_e32 v13, s3                                      // 000000001e38: 7e1a0203
	v_add_co_u32 v73, vcc_lo, s4, v4                           // 000000001e3c: d7006a49 02020804
	s_wait_alu depctr_va_vcc(0)                                // 000000001e44: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, s5, v5, vcc_lo              // 000000001e48: d5207c4a 01aa0a05
	v_or_b32_e32 v4, v43, v26                                  // 000000001e50: 3808352b
	v_mov_b32_e32 v5, s3                                       // 000000001e54: 7e0a0203
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000001e58: 7ca81810
	v_add3_u32 v7, v7, v42, v27                                // 000000001e5c: d6550007 046e5507
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 000000001e64: 3e141482
	v_add_nc_u32_e32 v118, 0x1800, v38                         // 000000001e68: 4aec4cff 00001800
	v_cmp_gt_i64_e64 s2, s[16:17], v[4:5]                      // 000000001e70: d4540002 02020810
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v68, 0             // 000000001e78: ca100080 3f440080
	s_wait_alu depctr_va_vcc(0)                                // 000000001e80: bf88ff9d
	v_dual_cndmask_b32 v27, 0, v13 :: v_dual_cndmask_b32 v42, 0, v12// 000000001e84: ca521a80 1b2a1880
	v_or_b32_e32 v12, v44, v26                                 // 000000001e8c: 3818352c
	s_wait_alu depctr_va_sdst(0)                               // 000000001e90: bf88f19f
	v_cndmask_b32_e64 v26, 0, v5, s2                           // 000000001e94: d501001a 000a0a80
	v_cndmask_b32_e64 v45, 0, v4, s2                           // 000000001e9c: d501002d 000a0880
	v_mul_lo_u32 v47, s22, v27                                 // 000000001ea4: d72c002f 02023616
	v_mul_lo_u32 v46, s8, v42                                  // 000000001eac: d72c002e 02025408
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000001eb4: 7ca81810
	v_mad_co_u64_u32 v[4:5], null, s22, v42, 0                 // 000000001eb8: d6fe7c04 02025416
	v_mul_lo_u32 v42, s8, v45                                  // 000000001ec0: d72c002a 02025a08
	v_mul_lo_u32 v48, s22, v26                                 // 000000001ec8: d72c0030 02023416
	v_mad_co_u64_u32 v[26:27], null, s22, v45, 0               // 000000001ed0: d6fe7c1a 02025a16
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 000000001ed8: 3e0c0c82
	s_wait_alu depctr_va_vcc(0)                                // 000000001edc: bf88ff9d
	v_dual_cndmask_b32 v12, 0, v12 :: v_dual_cndmask_b32 v13, 0, v13// 000000001ee0: ca521880 0c0c1a80
	v_add_co_u32 v76, vcc_lo, s4, v10                          // 000000001ee8: d7006a4c 02021404
	v_add3_u32 v5, v5, v47, v46                                // 000000001ef0: d6550005 04ba5f05
	s_wait_alu depctr_va_vcc(0)                                // 000000001ef8: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s5, v11, vcc_lo             // 000000001efc: d5207c4d 01aa1605
	v_mul_lo_u32 v45, s8, v12                                  // 000000001f04: d72c002d 02021808
	v_mul_lo_u32 v13, s22, v13                                 // 000000001f0c: d72c000d 02021a16
	v_mad_co_u64_u32 v[10:11], null, s22, v12, 0               // 000000001f14: d6fe7c0a 02021816
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]                        // 000000001f1c: 3e080882
	v_add3_u32 v27, v27, v48, v42                              // 000000001f20: d655001b 04aa611b
	v_add_co_u32 v78, vcc_lo, s4, v6                           // 000000001f28: d7006a4e 02020c04
	s_wait_alu depctr_va_vcc(0)                                // 000000001f30: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, s5, v7, vcc_lo              // 000000001f34: d5207c4f 01aa0e05
	s_delay_alu instid0(valu_dep_3)                            // 000000001f3c: bf870003
	v_lshlrev_b64_e32 v[6:7], 2, v[26:27]                      // 000000001f40: 3e0c3482
	v_add3_u32 v11, v11, v13, v45                              // 000000001f44: d655000b 04b61b0b
	v_add_co_u32 v80, vcc_lo, s4, v4                           // 000000001f4c: d7006a50 02020804
	s_wait_alu depctr_va_vcc(0)                                // 000000001f54: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, s5, v5, vcc_lo              // 000000001f58: d5207c51 01aa0a05
	v_or_b32_e32 v12, s6, v29                                  // 000000001f60: 38183a06
	v_mov_b32_e32 v5, s3                                       // 000000001f64: 7e0a0203
	v_mov_b32_e32 v13, s7                                      // 000000001f68: 7e1a0207
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]                    // 000000001f6c: 3e141482
	v_add_co_u32 v83, vcc_lo, s4, v6                           // 000000001f70: d7006a53 02020c04
	s_wait_alu depctr_va_vcc(0)                                // 000000001f78: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, s5, v7, vcc_lo              // 000000001f7c: d5207c54 01aa0e05
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[12:13]                // 000000001f84: 7ca81812
	v_or_b32_e32 v4, v36, v28                                  // 000000001f88: 38083924
	v_add_co_u32 v87, s2, s4, v10                              // 000000001f8c: d7000257 02021404
	s_wait_alu depctr_va_sdst(0)                               // 000000001f94: bf88f19f
	v_add_co_ci_u32_e64 v88, null, s5, v11, s2                 // 000000001f98: d5207c58 000a1605
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa0: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v13 :: v_dual_cndmask_b32 v10, 0, v12// 000000001fa4: ca521a80 0b0a1880
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[4:5]                  // 000000001fac: 7ca80810
	v_mov_b32_e32 v7, s7                                       // 000000001fb0: 7e0e0207
	v_or_b32_e32 v6, s6, v30                                   // 000000001fb4: 380c3c06
	v_mov_b32_e32 v13, s3                                      // 000000001fb8: 7e1a0203
	v_or_b32_e32 v12, v36, v31                                 // 000000001fbc: 38183f24
	s_wait_alu depctr_va_vcc(0)                                // 000000001fc0: bf88ff9d
	v_dual_mov_b32 v29, s7 :: v_dual_cndmask_b32 v26, 0, v5    // 000000001fc4: ca120007 1d1a0a80
	v_cndmask_b32_e32 v28, 0, v4, vcc_lo                       // 000000001fcc: 02380880
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[6:7]                  // 000000001fd0: 7ca80c12
	v_cmp_gt_i64_e64 s2, s[16:17], v[12:13]                    // 000000001fd4: d4540002 02021810
	v_mov_b32_e32 v106, 0                                      // 000000001fdc: 7ed40280
	v_mul_lo_u32 v42, s22, v26                                 // 000000001fe0: d72c002a 02023416
	v_mad_co_u64_u32 v[30:31], null, s22, v28, 0               // 000000001fe8: d6fe7c1e 02023816
	s_wait_alu depctr_va_vcc(0)                                // 000000001ff0: bf88ff9d
	v_dual_mov_b32 v100, 0 :: v_dual_cndmask_b32 v27, 0, v7    // 000000001ff4: ca120080 641a0e80
	v_mul_lo_u32 v7, s8, v28                                   // 000000001ffc: d72c0007 02023808
	s_wait_alu depctr_va_sdst(0)                               // 000000002004: bf88f19f
	v_cndmask_b32_e64 v45, 0, v12, s2                          // 000000002008: d501002d 000a1880
	v_or_b32_e32 v12, v36, v33                                 // 000000002010: 38184324
	v_or_b32_e32 v28, s6, v32                                  // 000000002014: 38384006
	v_cndmask_b32_e64 v32, 0, v13, s2                          // 000000002018: d5010020 000a1a80
	v_cndmask_b32_e32 v26, 0, v6, vcc_lo                       // 000000002020: 02340c80
	v_mul_lo_u32 v33, s8, v45                                  // 000000002024: d72c0021 02025a08
	v_cmp_gt_i64_e64 s2, s[16:17], v[12:13]                    // 00000000202c: d4540002 02021810
	v_add3_u32 v31, v31, v42, v7                               // 000000002034: d655001f 041e551f
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[28:29]                // 00000000203c: 7ca83812
	v_mad_co_u64_u32 v[6:7], null, s22, v45, 0                 // 000000002040: d6fe7c06 02025a16
	v_mul_lo_u32 v32, s22, v32                                 // 000000002048: d72c0020 02024016
	v_mov_b32_e32 v102, 0                                      // 000000002050: 7ecc0280
	s_wait_alu depctr_va_sdst(0)                               // 000000002054: bf88f19f
	v_cndmask_b32_e64 v42, 0, v13, s2                          // 000000002058: d501002a 000a1a80
	v_cndmask_b32_e64 v45, 0, v12, s2                          // 000000002060: d501002d 000a1880
	v_lshlrev_b64_e32 v[12:13], 2, v[30:31]                    // 000000002068: 3e183c82
	v_mov_b32_e32 v31, s3                                      // 00000000206c: 7e3e0203
	v_or_b32_e32 v30, v36, v34                                 // 000000002070: 383c4524
	s_wait_alu depctr_va_vcc(0)                                // 000000002074: bf88ff9d
	v_dual_cndmask_b32 v29, 0, v29 :: v_dual_add_nc_u32 v120, 0x1800, v40// 000000002078: ca603a80 1d7850ff 00001800
	v_cndmask_b32_e32 v28, 0, v28, vcc_lo                      // 000000002084: 02383880
	v_add_co_u32 v95, s2, s4, v12                              // 000000002088: d700025f 02021804
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[30:31]                // 000000002090: 7ca83c10
	s_wait_alu depctr_va_sdst(0)                               // 000000002094: bf88f19f
	v_add_co_ci_u32_e64 v96, null, s5, v13, s2                 // 000000002098: d5207c60 000a1a05
	v_mov_b32_e32 v13, s3                                      // 0000000020a0: 7e1a0203
	v_or_b32_e32 v12, v36, v35                                 // 0000000020a4: 38184724
	v_add3_u32 v7, v7, v32, v33                                // 0000000020a8: d6550007 04864107
	v_mul_lo_u32 v34, s8, v45                                  // 0000000020b0: d72c0022 02025a08
	v_mul_lo_u32 v42, s22, v42                                 // 0000000020b8: d72c002a 02025416
	v_mad_co_u64_u32 v[32:33], null, s22, v45, 0               // 0000000020c0: d6fe7c20 02025a16
	s_wait_alu depctr_va_vcc(0)                                // 0000000020c8: bf88ff9d
	v_dual_cndmask_b32 v31, 0, v31 :: v_dual_cndmask_b32 v30, 0, v30// 0000000020cc: ca523e80 1f1e3c80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 0000000020d4: 7ca81810
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]                        // 0000000020d8: 3e0c0c82
	v_mov_b32_e32 v72, 0                                       // 0000000020dc: 7e900280
	s_delay_alu instid0(valu_dep_4)                            // 0000000020e0: bf870004
	v_mul_lo_u32 v35, s22, v31                                 // 0000000020e4: d72c0023 02023e16
	v_mov_b32_e32 v70, 0                                       // 0000000020ec: 7e8c0280
	v_add3_u32 v33, v33, v42, v34                              // 0000000020f0: d6550021 048a5521
	v_mul_lo_u32 v34, s8, v30                                  // 0000000020f8: d72c0022 02023c08
	v_mad_co_u64_u32 v[30:31], null, s22, v30, 0               // 000000002100: d6fe7c1e 02023c16
	s_wait_alu depctr_va_vcc(0)                                // 000000002108: bf88ff9d
	v_cndmask_b32_e32 v45, 0, v12, vcc_lo                      // 00000000210c: 025a1880
	v_or_b32_e32 v12, v36, v41                                 // 000000002110: 38185324
	v_add_co_u32 v98, s2, s4, v6                               // 000000002114: d7000262 02020c04
	s_wait_alu depctr_va_sdst(0)                               // 00000000211c: bf88f19f
	v_add_co_ci_u32_e64 v99, null, s5, v7, s2                  // 000000002120: d5207c63 000a0e05
	v_lshlrev_b64_e32 v[6:7], 2, v[32:33]                      // 000000002128: 3e0c4082
	v_cndmask_b32_e32 v42, 0, v13, vcc_lo                      // 00000000212c: 02541a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000002130: 7ca81810
	v_add3_u32 v31, v31, v35, v34                              // 000000002134: d655001f 048a471f
	v_mov_b32_e32 v35, s3                                      // 00000000213c: 7e460203
	v_or_b32_e32 v34, v36, v43                                 // 000000002140: 38445724
	v_add_co_u32 v103, s2, s4, v6                              // 000000002144: d7000267 02020c04
	s_wait_alu depctr_va_sdst(0)                               // 00000000214c: bf88f19f
	v_add_co_ci_u32_e64 v104, null, s5, v7, s2                 // 000000002150: d5207c68 000a0e05
	v_lshlrev_b64_e32 v[6:7], 2, v[30:31]                      // 000000002158: 3e0c3c82
	s_wait_alu depctr_va_vcc(0)                                // 00000000215c: bf88ff9d
	v_dual_cndmask_b32 v31, 0, v12 :: v_dual_mov_b32 v66, 0    // 000000002160: ca501880 1f420080
	v_or_b32_e32 v12, v36, v44                                 // 000000002168: 38185924
	v_mul_lo_u32 v41, s8, v45                                  // 00000000216c: d72c0029 02025a08
	v_mul_lo_u32 v42, s22, v42                                 // 000000002174: d72c002a 02025416
	v_mad_co_u64_u32 v[32:33], null, s22, v45, 0               // 00000000217c: d6fe7c20 02025a16
	v_cmp_gt_i64_e64 s2, s[16:17], v[34:35]                    // 000000002184: d4540002 02024410
	v_cndmask_b32_e32 v30, 0, v13, vcc_lo                      // 00000000218c: 023c1a80
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[12:13]                // 000000002190: 7ca81810
	v_mul_lo_u32 v36, s8, v31                                  // 000000002194: d72c0024 02023e08
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0             // 00000000219c: ca100080 38380080
	v_mov_b32_e32 v64, 0                                       // 0000000021a4: 7e800280
	s_wait_alu depctr_va_sdst(0)                               // 0000000021a8: bf88f19f
	v_cndmask_b32_e64 v35, 0, v35, s2                          // 0000000021ac: d5010023 000a4680
	v_cndmask_b32_e64 v34, 0, v34, s2                          // 0000000021b4: d5010022 000a4480
	v_add3_u32 v33, v33, v42, v41                              // 0000000021bc: d6550021 04a65521
	s_wait_alu depctr_va_vcc(0)                                // 0000000021c4: bf88ff9d
	v_dual_cndmask_b32 v13, 0, v13 :: v_dual_cndmask_b32 v12, 0, v12// 0000000021c8: ca521a80 0d0c1880
	v_mul_lo_u32 v41, s22, v30                                 // 0000000021d0: d72c0029 02023c16
	v_mad_co_u64_u32 v[30:31], null, s22, v31, 0               // 0000000021d8: d6fe7c1e 02023e16
	v_mul_lo_u32 v42, s8, v34                                  // 0000000021e0: d72c002a 02024408
	v_mul_lo_u32 v43, s22, v35                                 // 0000000021e8: d72c002b 02024616
	v_mad_co_u64_u32 v[34:35], null, s22, v34, 0               // 0000000021f0: d6fe7c22 02024416
	v_add_co_u32 v107, vcc_lo, s4, v6                          // 0000000021f8: d7006a6b 02020c04
	s_wait_alu depctr_va_vcc(0)                                // 000000002200: bf88ff9d
	v_add_co_ci_u32_e64 v108, null, s5, v7, vcc_lo             // 000000002204: d5207c6c 01aa0e05
	v_lshlrev_b64_e32 v[6:7], 2, v[32:33]                      // 00000000220c: 3e0c4082
	v_mul_lo_u32 v32, s8, v12                                  // 000000002210: d72c0020 02021808
	v_mul_lo_u32 v33, s22, v13                                 // 000000002218: d72c0021 02021a16
	v_mad_co_u64_u32 v[12:13], null, s22, v12, 0               // 000000002220: d6fe7c0c 02021816
	v_add3_u32 v31, v31, v41, v36                              // 000000002228: d655001f 0492531f
	v_add3_u32 v35, v35, v43, v42                              // 000000002230: d6550023 04aa5723
	v_add_co_u32 v109, vcc_lo, s4, v6                          // 000000002238: d7006a6d 02020c04
	s_wait_alu depctr_va_vcc(0)                                // 000000002240: bf88ff9d
	v_add_co_ci_u32_e64 v110, null, s5, v7, vcc_lo             // 000000002244: d5207c6e 01aa0e05
	v_lshlrev_b64_e32 v[30:31], 2, v[30:31]                    // 00000000224c: 3e3c3c82
	v_add3_u32 v13, v13, v33, v32                              // 000000002250: d655000d 0482430d
	v_lshlrev_b64_e32 v[6:7], 2, v[34:35]                      // 000000002258: 3e0c4482
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v62, 0             // 00000000225c: ca100080 373e0080
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v60, 0             // 000000002264: ca100080 353c0080
	s_delay_alu instid0(valu_dep_4)                            // 00000000226c: bf870004
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]                    // 000000002270: 3e181882
	v_add_co_u32 v111, vcc_lo, s4, v30                         // 000000002274: d7006a6f 02023c04
	s_wait_alu depctr_va_vcc(0)                                // 00000000227c: bf88ff9d
	v_add_co_ci_u32_e64 v112, null, s5, v31, vcc_lo            // 000000002280: d5207c70 01aa3e05
	v_add_co_u32 v113, vcc_lo, s4, v6                          // 000000002288: d7006a71 02020c04
	s_wait_alu depctr_va_vcc(0)                                // 000000002290: bf88ff9d
	v_add_co_ci_u32_e64 v114, null, s5, v7, vcc_lo             // 000000002294: d5207c72 01aa0e05
	v_add_co_u32 v115, vcc_lo, s4, v12                         // 00000000229c: d7006a73 02021804
	s_wait_alu depctr_va_vcc(0)                                // 0000000022a4: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, s5, v13, vcc_lo            // 0000000022a8: d5207c74 01aa1a05
	v_lshlrev_b64_e32 v[6:7], 2, v[8:9]                        // 0000000022b0: 3e0c1082
	v_lshlrev_b64_e32 v[8:9], 2, v[10:11]                      // 0000000022b4: 3e101482
	v_lshlrev_b64_e32 v[10:11], 2, v[26:27]                    // 0000000022b8: 3e143482
	v_lshlrev_b64_e32 v[12:13], 2, v[28:29]                    // 0000000022bc: 3e183882
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v58, 0             // 0000000022c0: ca100080 333a0080
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v52, 0             // 0000000022c8: ca100080 2f340080
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v48, 0             // 0000000022d0: ca100080 29300080
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0             // 0000000022d8: ca100080 28260080
	v_mov_b32_e32 v46, 0                                       // 0000000022e0: 7e5c0280
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0             // 0000000022e4: ca100080 26240080
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v35, 0             // 0000000022ec: ca100080 2c220080
	v_mov_b32_e32 v42, 0                                       // 0000000022f4: 7e540280
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v31, 0             // 0000000022f8: ca100080 221e0080
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v91, 0             // 000000002300: ca100080 245a0080
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v89, 0             // 000000002308: ca100080 20580080
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v85, 0             // 000000002310: ca100080 1e540080
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v65, 0             // 000000002318: ca100080 1c400080
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v61, 0             // 000000002320: ca100080 1a3c0080
	v_mov_b32_e32 v59, 0                                       // 000000002328: 7e760280
	v_mov_b32_e32 v49, 0                                       // 00000000232c: 7e620280
	v_mov_b32_e32 v45, 0                                       // 000000002330: 7e5a0280
	v_mov_b32_e32 v43, 0                                       // 000000002334: 7e560280
	v_mov_b32_e32 v33, 0                                       // 000000002338: 7e420280
	v_mov_b32_e32 v29, 0                                       // 00000000233c: 7e3a0280
	v_mov_b32_e32 v27, 0                                       // 000000002340: 7e360280
	s_mov_b64 s[24:25], 0                                      // 000000002344: be980180
	s_delay_alu instid0(salu_cycle_1)                          // 000000002348: bf870009
	s_lshl_b64 s[2:3], s[24:25], 5                             // 00000000234c: 84828518
	s_lshl_b64 s[16:17], s[24:25], 2                           // 000000002350: 84908218
	s_wait_alu depctr_sa_sdst(0)                               // 000000002354: bf88ff9e
	v_add_co_u32 v121, vcc_lo, v16, s2                         // 000000002358: d7006a79 02000510
	v_add_co_u32 v125, s2, v18, s2                             // 000000002360: d700027d 02000512
	s_wait_alu depctr_va_vcc(0)                                // 000000002368: bf88ff9d
	v_add_co_ci_u32_e64 v122, null, s3, v17, vcc_lo            // 00000000236c: d5207c7a 01aa2203
	s_wait_alu depctr_va_sdst(0)                               // 000000002374: bf88f19f
	v_add_co_ci_u32_e64 v126, null, s3, v19, s2                // 000000002378: d5207c7e 000a2603
	s_mul_u64 s[2:3], s[24:25], s[18:19]                       // 000000002380: aa821218
	global_load_b128 v[121:124], v[121:122], off               // 000000002384: ee05c07c 00000079 00000079
	s_wait_alu depctr_sa_sdst(0)                               // 000000002390: bf88ff9e
	s_lshl_b64 s[26:27], s[2:3], 2                             // 000000002394: 849a8202
	global_load_b128 v[125:128], v[125:126], off               // 000000002398: ee05c07c 0000007d 0000007d
	v_add_co_u32 v129, vcc_lo, v22, s16                        // 0000000023a4: d7006a81 02002116
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023ac: bf88ff9e
	s_add_nc_u64 s[26:27], s[20:21], s[26:27]                  // 0000000023b0: a99a1a14
	v_add_co_u32 v131, s2, v24, s16                            // 0000000023b4: d7000283 02002118
	v_add_co_u32 v133, s3, v73, s16                            // 0000000023bc: d7000385 02002149
	s_wait_alu depctr_va_vcc(0)                                // 0000000023c4: bf88ff9d
	v_add_co_ci_u32_e64 v130, null, s17, v23, vcc_lo           // 0000000023c8: d5207c82 01aa2e11
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023d0: bf88ff9e
	v_add_co_u32 v161, vcc_lo, s26, v6                         // 0000000023d4: d7006aa1 02020c1a
	s_wait_alu depctr_va_sdst(0)                               // 0000000023dc: bf88f19f
	v_add_co_ci_u32_e64 v132, null, s17, v25, s2               // 0000000023e0: d5207c84 000a3211
	v_add_co_ci_u32_e64 v134, null, s17, v74, s3               // 0000000023e8: d5207c86 000e9411
	s_wait_alu depctr_va_vcc(0)                                // 0000000023f0: bf88ff9d
	v_add_co_ci_u32_e64 v162, null, s27, v7, vcc_lo            // 0000000023f4: d5207ca2 01aa0e1b
	v_add_co_u32 v135, s4, v76, s16                            // 0000000023fc: d7000487 0200214c
	v_add_co_u32 v137, s5, v78, s16                            // 000000002404: d7000589 0200214e
	v_add_co_u32 v139, s6, v80, s16                            // 00000000240c: d700068b 02002150
	v_add_co_u32 v141, s7, v83, s16                            // 000000002414: d700078d 02002153
	v_add_co_u32 v143, s8, v87, s16                            // 00000000241c: d700088f 02002157
	v_add_co_u32 v163, s2, s26, v8                             // 000000002424: d70002a3 0202101a
	s_wait_alu depctr_va_sdst(0)                               // 00000000242c: bf88f19f
	v_add_co_ci_u32_e64 v136, null, s17, v77, s4               // 000000002430: d5207c88 00129a11
	v_add_co_ci_u32_e64 v138, null, s17, v79, s5               // 000000002438: d5207c8a 00169e11
	v_add_co_ci_u32_e64 v140, null, s17, v81, s6               // 000000002440: d5207c8c 001aa211
	v_add_co_ci_u32_e64 v142, null, s17, v84, s7               // 000000002448: d5207c8e 001ea811
	v_add_co_ci_u32_e64 v144, null, s17, v88, s8               // 000000002450: d5207c90 0022b011
	v_add_co_ci_u32_e64 v164, null, s27, v9, s2                // 000000002458: d5207ca4 000a121b
	s_barrier_signal -1                                        // 000000002460: be804ec1
	s_barrier_wait 0xffff                                      // 000000002464: bf94ffff
	v_add_co_u32 v165, s3, s26, v10                            // 000000002468: d70003a5 0202141a
	v_add_co_u32 v167, s4, s26, v12                            // 000000002470: d70004a7 0202181a
	s_wait_alu depctr_va_sdst(0)                               // 000000002478: bf88f19f
	v_add_co_ci_u32_e64 v166, null, s27, v11, s3               // 00000000247c: d5207ca6 000e161b
	v_add_co_ci_u32_e64 v168, null, s27, v13, s4               // 000000002484: d5207ca8 00121a1b
	v_add_co_u32 v145, s9, v95, s16                            // 00000000248c: d7000991 0200215f
	v_add_co_u32 v147, s10, v98, s16                           // 000000002494: d7000a93 02002162
	v_add_co_u32 v149, s11, v103, s16                          // 00000000249c: d7000b95 02002167
	v_add_co_u32 v151, s12, v107, s16                          // 0000000024a4: d7000c97 0200216b
	v_add_co_u32 v153, s13, v109, s16                          // 0000000024ac: d7000d99 0200216d
	v_add_co_u32 v155, s14, v111, s16                          // 0000000024b4: d7000e9b 0200216f
	v_add_co_u32 v157, s15, v113, s16                          // 0000000024bc: d7000f9d 02002171
	v_add_co_u32 v159, s16, v115, s16                          // 0000000024c4: d700109f 02002173
	s_wait_alu depctr_va_sdst(0)                               // 0000000024cc: bf88f19f
	v_add_co_ci_u32_e64 v146, null, s17, v96, s9               // 0000000024d0: d5207c92 0026c011
	v_add_co_ci_u32_e64 v148, null, s17, v99, s10              // 0000000024d8: d5207c94 002ac611
	v_add_co_ci_u32_e64 v150, null, s17, v104, s11             // 0000000024e0: d5207c96 002ed011
	v_add_co_ci_u32_e64 v152, null, s17, v108, s12             // 0000000024e8: d5207c98 0032d811
	v_add_co_ci_u32_e64 v154, null, s17, v110, s13             // 0000000024f0: d5207c9a 0036dc11
	v_add_co_ci_u32_e64 v156, null, s17, v112, s14             // 0000000024f8: d5207c9c 003ae011
	v_add_co_ci_u32_e64 v158, null, s17, v114, s15             // 000000002500: d5207c9e 003ee411
	v_add_co_ci_u32_e64 v160, null, s17, v116, s16             // 000000002508: d5207ca0 0042e811
	s_add_nc_u64 s[24:25], s[24:25], 1                         // 000000002510: a9988118
	s_delay_alu instid0(salu_cycle_1)                          // 000000002514: bf870009
	s_cmp_lg_u64 s[24:25], s[22:23]                            // 000000002518: bf111618
	s_wait_loadcnt 0x1                                         // 00000000251c: bfc00001
	ds_store_b128 v15, v[121:124]                              // 000000002520: db7c0000 0000790f
	s_wait_loadcnt 0x0                                         // 000000002528: bfc00000
	ds_store_b128 v15, v[125:128] offset:6144                  // 00000000252c: db7c1800 00007d0f
	s_wait_dscnt 0x0                                           // 000000002534: bfc60000
	s_barrier_signal -1                                        // 000000002538: be804ec1
	s_barrier_wait 0xffff                                      // 00000000253c: bf94ffff
	global_load_b32 v195, v[129:130], off                      // 000000002540: ee05007c 000000c3 00000081
	global_load_b32 v196, v[161:162], off                      // 00000000254c: ee05007c 000000c4 000000a1
	s_clause 0x6                                               // 000000002558: bf850006
	global_load_b32 v197, v[131:132], off                      // 00000000255c: ee05007c 000000c5 00000083
	global_load_b32 v198, v[133:134], off                      // 000000002568: ee05007c 000000c6 00000085
	global_load_b32 v199, v[135:136], off                      // 000000002574: ee05007c 000000c7 00000087
	global_load_b32 v200, v[137:138], off                      // 000000002580: ee05007c 000000c8 00000089
	global_load_b32 v201, v[139:140], off                      // 00000000258c: ee05007c 000000c9 0000008b
	global_load_b32 v202, v[141:142], off                      // 000000002598: ee05007c 000000ca 0000008d
	global_load_b32 v203, v[143:144], off                      // 0000000025a4: ee05007c 000000cb 0000008f
	s_clause 0x2                                               // 0000000025b0: bf850002
	global_load_b32 v204, v[163:164], off                      // 0000000025b4: ee05007c 000000cc 000000a3
	global_load_b32 v205, v[165:166], off                      // 0000000025c0: ee05007c 000000cd 000000a5
	global_load_b32 v206, v[167:168], off                      // 0000000025cc: ee05007c 000000ce 000000a7
	s_clause 0x7                                               // 0000000025d8: bf850007
	global_load_b32 v207, v[145:146], off                      // 0000000025dc: ee05007c 000000cf 00000091
	global_load_b32 v208, v[147:148], off                      // 0000000025e8: ee05007c 000000d0 00000093
	global_load_b32 v209, v[149:150], off                      // 0000000025f4: ee05007c 000000d1 00000095
	global_load_b32 v210, v[151:152], off                      // 000000002600: ee05007c 000000d2 00000097
	global_load_b32 v211, v[153:154], off                      // 00000000260c: ee05007c 000000d3 00000099
	global_load_b32 v212, v[155:156], off                      // 000000002618: ee05007c 000000d4 0000009b
	global_load_b32 v213, v[157:158], off                      // 000000002624: ee05007c 000000d5 0000009d
	global_load_b32 v214, v[159:160], off                      // 000000002630: ee05007c 000000d6 0000009f
	ds_load_2addr_b64 v[167:170], v20 offset1:2                // 00000000263c: d9dc0200 a7000014
	ds_load_2addr_b64 v[175:178], v117 offset1:2               // 000000002644: d9dc0200 af000075
	ds_load_2addr_b64 v[179:182], v118 offset1:2               // 00000000264c: d9dc0200 b3000076
	ds_load_2addr_b64 v[183:186], v119 offset1:2               // 000000002654: d9dc0200 b7000077
	ds_load_2addr_b64 v[187:190], v120 offset1:2               // 00000000265c: d9dc0200 bb000078
	ds_load_2addr_b64 v[191:194], v21 offset1:2                // 000000002664: d9dc0200 bf000015
	s_wait_dscnt 0x4                                           // 00000000266c: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[167:168], v[175:176], 0// 000000002670: cc464079 1a035fa7
	s_wait_dscnt 0x3                                           // 000000002678: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[167:168], v[179:180], 0// 00000000267c: cc464081 1a0367a7
	s_wait_dscnt 0x2                                           // 000000002684: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[167:168], v[183:184], 0// 000000002688: cc464089 1a036fa7
	s_wait_dscnt 0x1                                           // 000000002690: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[167:168], v[187:188], 0// 000000002694: cc464091 1a0377a7
	s_wait_dscnt 0x0                                           // 00000000269c: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[153:160], v[191:192], v[175:176], 0// 0000000026a0: cc464099 1a035fbf
	v_wmma_f32_16x16x16_fp8_fp8 v[161:168], v[191:192], v[179:180], 0// 0000000026a8: cc4640a1 1a0367bf
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[169:170], v[177:178], v[121:128]// 0000000026b0: cc464079 1de763a9
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[169:170], v[181:182], v[129:136]// 0000000026b8: cc464081 1e076ba9
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[169:170], v[185:186], v[137:144]// 0000000026c0: cc464089 1e2773a9
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[169:170], v[189:190], v[145:152]// 0000000026c8: cc464091 1e477ba9
	v_wmma_f32_16x16x16_fp8_fp8 v[169:176], v[191:192], v[183:184], 0// 0000000026d0: cc4640a9 1a036fbf
	v_wmma_f32_16x16x16_fp8_fp8 v[153:160], v[193:194], v[177:178], v[153:160]// 0000000026d8: cc464099 1e6763c1
	v_wmma_f32_16x16x16_fp8_fp8 v[161:168], v[193:194], v[181:182], v[161:168]// 0000000026e0: cc4640a1 1e876bc1
	v_wmma_f32_16x16x16_fp8_fp8 v[177:184], v[191:192], v[187:188], 0// 0000000026e8: cc4640b1 1a0377bf
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_2)// 0000000026f0: bf870114
	v_wmma_f32_16x16x16_fp8_fp8 v[169:176], v[193:194], v[185:186], v[169:176]// 0000000026f4: cc4640a9 1ea773c1
	v_wmma_f32_16x16x16_fp8_fp8 v[177:184], v[193:194], v[189:190], v[177:184]// 0000000026fc: cc4640b1 1ec77bc1
	s_wait_loadcnt 0x11                                        // 000000002704: bfc00011
	v_dual_mul_f32 v185, v195, v196 :: v_dual_mul_f32 v186, v196, v197// 000000002708: c8c789c3 b9bb8bc4
	s_wait_loadcnt 0xf                                         // 000000002710: bfc0000f
	v_dual_mul_f32 v187, v196, v198 :: v_dual_mul_f32 v188, v196, v199// 000000002714: c8c78dc4 bbbd8fc4
	s_wait_loadcnt 0xd                                         // 00000000271c: bfc0000d
	v_dual_mul_f32 v189, v196, v200 :: v_dual_mul_f32 v190, v196, v201// 000000002720: c8c791c4 bdbf93c4
	s_wait_loadcnt 0xb                                         // 000000002728: bfc0000b
	v_dual_mul_f32 v191, v196, v202 :: v_dual_mul_f32 v192, v196, v203// 00000000272c: c8c795c4 bfc197c4
	s_wait_loadcnt 0xa                                         // 000000002734: bfc0000a
	v_dual_mul_f32 v193, v195, v204 :: v_dual_mul_f32 v194, v197, v204// 000000002738: c8c799c3 c1c399c5
	v_dual_mul_f32 v121, v121, v185 :: v_dual_mul_f32 v122, v122, v186// 000000002740: c8c77379 797b757a
	v_mul_f32_e32 v123, v123, v187                             // 000000002748: 10f7777b
	v_dual_mul_f32 v185, v201, v204 :: v_dual_mul_f32 v186, v202, v204// 00000000274c: c8c799c9 b9bb99ca
	v_mul_f32_e32 v187, v203, v204                             // 000000002754: 117799cb
	v_dual_mul_f32 v215, v198, v204 :: v_dual_mul_f32 v216, v199, v204// 000000002758: c8c799c6 d7d999c7
	v_mul_f32_e32 v217, v200, v204                             // 000000002760: 11b399c8
	v_dual_mul_f32 v124, v124, v188 :: v_dual_mul_f32 v125, v125, v189// 000000002764: c8c7797c 7c7d7b7d
	v_dual_mul_f32 v126, v126, v190 :: v_dual_mul_f32 v127, v127, v191// 00000000276c: c8c77d7e 7e7f7f7f
	v_mul_f32_e32 v128, v128, v192                             // 000000002774: 11018180
	s_wait_loadcnt 0x9                                         // 000000002778: bfc00009
	v_dual_mul_f32 v188, v195, v205 :: v_dual_mul_f32 v189, v197, v205// 00000000277c: c8c79bc3 bcbd9bc5
	v_dual_mul_f32 v190, v198, v205 :: v_dual_mul_f32 v191, v199, v205// 000000002784: c8c79bc6 bebf9bc7
	v_mul_f32_e32 v192, v200, v205                             // 00000000278c: 11819bc8
	v_dual_mul_f32 v129, v129, v193 :: v_dual_mul_f32 v130, v130, v194// 000000002790: c8c78381 81838582
	v_dual_mul_f32 v134, v134, v185 :: v_dual_mul_f32 v135, v135, v186// 000000002798: c8c77386 86877587
	v_dual_mul_f32 v136, v136, v187 :: v_dual_mul_f32 v185, v201, v205// 0000000027a0: c8c77788 88b99bc9
	v_dual_mul_f32 v186, v202, v205 :: v_dual_mul_f32 v187, v203, v205// 0000000027a8: c8c79bca babb9bcb
	s_wait_loadcnt 0x8                                         // 0000000027b0: bfc00008
	v_dual_mul_f32 v193, v195, v206 :: v_dual_mul_f32 v194, v197, v206// 0000000027b4: c8c79dc3 c1c39dc5
	v_mul_f32_e32 v195, v198, v206                             // 0000000027bc: 11879dc6
	v_dual_mul_f32 v197, v199, v206 :: v_dual_mul_f32 v198, v200, v206// 0000000027c0: c8c79dc7 c5c79dc8
	v_dual_mul_f32 v199, v201, v206 :: v_dual_mul_f32 v200, v202, v206// 0000000027c8: c8c79dc9 c7c99dca
	v_mul_f32_e32 v201, v203, v206                             // 0000000027d0: 11939dcb
	v_dual_mul_f32 v131, v131, v215 :: v_dual_mul_f32 v132, v132, v216// 0000000027d4: c8c7af83 8385b184
	s_wait_loadcnt 0x7                                         // 0000000027dc: bfc00007
	v_dual_mul_f32 v133, v133, v217 :: v_dual_mul_f32 v202, v196, v207// 0000000027e0: c8c7b385 85cb9fc4
	s_wait_loadcnt 0x6                                         // 0000000027e8: bfc00006
	v_mul_f32_e32 v203, v196, v208                             // 0000000027ec: 1197a1c4
	s_wait_loadcnt 0x4                                         // 0000000027f0: bfc00004
	v_dual_mul_f32 v215, v196, v209 :: v_dual_mul_f32 v216, v196, v210// 0000000027f4: c8c7a3c4 d7d9a5c4
	s_wait_loadcnt 0x3                                         // 0000000027fc: bfc00003
	v_mul_f32_e32 v217, v196, v211                             // 000000002800: 11b3a7c4
	v_dual_mul_f32 v137, v137, v188 :: v_dual_mul_f32 v138, v138, v189// 000000002804: c8c77989 898b7b8a
	v_dual_mul_f32 v139, v139, v190 :: v_dual_mul_f32 v140, v140, v191// 00000000280c: c8c77d8b 8b8d7f8c
	v_dual_mul_f32 v141, v141, v192 :: v_dual_mul_f32 v142, v142, v185// 000000002814: c8c7818d 8d8f738e
	v_dual_mul_f32 v143, v143, v186 :: v_dual_mul_f32 v144, v144, v187// 00000000281c: c8c7758f 8f917790
	s_wait_loadcnt 0x1                                         // 000000002824: bfc00001
	v_dual_mul_f32 v185, v196, v212 :: v_dual_mul_f32 v186, v196, v213// 000000002828: c8c7a9c4 b9bbabc4
	s_wait_loadcnt 0x0                                         // 000000002830: bfc00000
	v_mul_f32_e32 v187, v196, v214                             // 000000002834: 1177adc4
	v_dual_mul_f32 v188, v204, v207 :: v_dual_mul_f32 v189, v204, v208// 000000002838: c8c79fcc bcbda1cc
	v_dual_mul_f32 v190, v204, v209 :: v_dual_mul_f32 v191, v204, v210// 000000002840: c8c7a3cc bebfa5cc
	v_mul_f32_e32 v192, v204, v211                             // 000000002848: 1181a7cc
	v_dual_mul_f32 v196, v204, v212 :: v_dual_mul_f32 v145, v145, v193// 00000000284c: c8c7a9cc c4918391
	v_dual_mul_f32 v146, v146, v194 :: v_dual_mul_f32 v147, v147, v195// 000000002854: c8c78592 92938793
	v_dual_mul_f32 v148, v148, v197 :: v_dual_mul_f32 v149, v149, v198// 00000000285c: c8c78b94 94958d95
	v_dual_mul_f32 v150, v150, v199 :: v_dual_mul_f32 v151, v151, v200// 000000002864: c8c78f96 96979197
	v_mul_f32_e32 v152, v152, v201                             // 00000000286c: 11319398
	v_dual_mul_f32 v193, v204, v213 :: v_dual_mul_f32 v194, v204, v214// 000000002870: c8c7abcc c1c3adcc
	v_mul_f32_e32 v195, v205, v207                             // 000000002878: 11879fcd
	v_dual_mul_f32 v197, v205, v208 :: v_dual_mul_f32 v198, v205, v209// 00000000287c: c8c7a1cd c5c7a3cd
	v_dual_mul_f32 v199, v205, v210 :: v_dual_mul_f32 v200, v205, v211// 000000002884: c8c7a5cd c7c9a7cd
	v_dual_mul_f32 v201, v205, v212 :: v_dual_mul_f32 v204, v205, v213// 00000000288c: c8c7a9cd c9cdabcd
	v_mul_f32_e32 v205, v205, v214                             // 000000002894: 119badcd
	v_dual_mul_f32 v207, v206, v207 :: v_dual_mul_f32 v208, v206, v208// 000000002898: c8c79fce cfd1a1ce
	v_dual_mul_f32 v209, v206, v209 :: v_dual_mul_f32 v210, v206, v210// 0000000028a0: c8c7a3ce d1d3a5ce
	v_dual_mul_f32 v211, v206, v211 :: v_dual_mul_f32 v212, v206, v212// 0000000028a8: c8c7a7ce d3d5a9ce
	v_dual_mul_f32 v213, v206, v213 :: v_dual_mul_f32 v206, v206, v214// 0000000028b0: c8c7abce d5cfadce
	v_dual_mul_f32 v153, v153, v202 :: v_dual_mul_f32 v154, v154, v203// 0000000028b8: c8c79599 999b979a
	v_dual_mul_f32 v155, v155, v215 :: v_dual_mul_f32 v156, v156, v216// 0000000028c0: c8c7af9b 9b9db19c
	v_mul_f32_e32 v157, v157, v217                             // 0000000028c8: 113bb39d
	v_dual_mul_f32 v158, v158, v185 :: v_dual_mul_f32 v159, v159, v186// 0000000028cc: c8c7739e 9e9f759f
	v_dual_mul_f32 v160, v160, v187 :: v_dual_mul_f32 v161, v161, v188// 0000000028d4: c8c777a0 a0a179a1
	v_dual_mul_f32 v162, v162, v189 :: v_dual_mul_f32 v163, v163, v190// 0000000028dc: c8c77ba2 a2a37da3
	v_dual_mul_f32 v164, v164, v191 :: v_dual_mul_f32 v165, v165, v192// 0000000028e4: c8c77fa4 a4a581a5
	v_dual_mul_f32 v166, v166, v196 :: v_dual_mul_f32 v167, v167, v193// 0000000028ec: c8c789a6 a6a783a7
	v_dual_mul_f32 v168, v168, v194 :: v_dual_mul_f32 v169, v169, v195// 0000000028f4: c8c785a8 a8a987a9
	v_dual_mul_f32 v170, v170, v197 :: v_dual_mul_f32 v171, v171, v198// 0000000028fc: c8c78baa aaab8dab
	v_dual_mul_f32 v172, v172, v199 :: v_dual_mul_f32 v173, v173, v200// 000000002904: c8c78fac acad91ad
	v_dual_mul_f32 v174, v174, v201 :: v_dual_mul_f32 v175, v175, v204// 00000000290c: c8c793ae aeaf99af
	v_dual_mul_f32 v176, v176, v205 :: v_dual_mul_f32 v177, v177, v207// 000000002914: c8c79bb0 b0b19fb1
	v_dual_mul_f32 v178, v178, v208 :: v_dual_mul_f32 v179, v179, v209// 00000000291c: c8c7a1b2 b2b3a3b3
	v_dual_mul_f32 v180, v180, v210 :: v_dual_mul_f32 v181, v181, v211// 000000002924: c8c7a5b4 b4b5a7b5
	v_dual_mul_f32 v182, v182, v212 :: v_dual_mul_f32 v183, v183, v213// 00000000292c: c8c7a9b6 b6b7abb7
	v_mul_f32_e32 v184, v184, v206                             // 000000002934: 11719db8
	v_add_f32_e32 v14, v14, v121                               // 000000002938: 061cf30e
	v_dual_add_f32 v106, v106, v122 :: v_dual_add_f32 v105, v105, v123// 00000000293c: c908f56a 6a68f769
	v_dual_add_f32 v102, v102, v124 :: v_dual_add_f32 v101, v101, v125// 000000002944: c908f966 6664fb65
	v_dual_add_f32 v100, v100, v126 :: v_dual_add_f32 v97, v97, v127// 00000000294c: c908fd64 6460ff61
	v_add_f32_e32 v93, v93, v128                               // 000000002954: 06bb015d
	v_dual_add_f32 v75, v75, v129 :: v_dual_add_f32 v72, v72, v130// 000000002958: c909034b 4b490548
	v_dual_add_f32 v71, v71, v131 :: v_dual_add_f32 v70, v70, v132// 000000002960: c9090747 47470946
	v_add_f32_e32 v69, v69, v133                               // 000000002968: 068b0b45
	v_dual_add_f32 v67, v67, v134 :: v_dual_add_f32 v66, v66, v135// 00000000296c: c9090d43 43430f42
	v_add_f32_e32 v63, v63, v136                               // 000000002974: 067f113f
	v_dual_add_f32 v57, v57, v137 :: v_dual_add_f32 v56, v56, v138// 000000002978: c9091339 39391538
	v_dual_add_f32 v55, v55, v139 :: v_dual_add_f32 v54, v54, v140// 000000002980: c9091737 37371936
	v_add_f32_e32 v53, v53, v141                               // 000000002988: 066b1b35
	v_dual_add_f32 v51, v51, v142 :: v_dual_add_f32 v50, v50, v143// 00000000298c: c9091d33 33331f32
	v_add_f32_e32 v47, v47, v144                               // 000000002994: 065f212f
	v_dual_add_f32 v41, v41, v145 :: v_dual_add_f32 v40, v40, v146// 000000002998: c9092329 29292528
	v_dual_add_f32 v39, v39, v147 :: v_dual_add_f32 v38, v38, v148// 0000000029a0: c9092727 27272926
	v_add_f32_e32 v37, v37, v149                               // 0000000029a8: 064b2b25
	v_dual_add_f32 v35, v35, v150 :: v_dual_add_f32 v34, v34, v151// 0000000029ac: c9092d23 23232f22
	v_dual_add_f32 v31, v31, v152 :: v_dual_add_f32 v94, v94, v153// 0000000029b4: c909311f 1f5f335e
	v_dual_add_f32 v92, v92, v154 :: v_dual_add_f32 v91, v91, v155// 0000000029bc: c909355c 5c5b375b
	v_dual_add_f32 v90, v90, v156 :: v_dual_add_f32 v89, v89, v157// 0000000029c4: c909395a 5a593b59
	v_dual_add_f32 v86, v86, v158 :: v_dual_add_f32 v85, v85, v159// 0000000029cc: c9093d56 56553f55
	v_add_f32_e32 v82, v82, v160                               // 0000000029d4: 06a54152
	v_dual_add_f32 v68, v68, v161 :: v_dual_add_f32 v65, v65, v162// 0000000029d8: c9094344 44414541
	v_add_f32_e32 v64, v64, v163                               // 0000000029e0: 06814740
	v_dual_add_f32 v62, v62, v164 :: v_dual_add_f32 v61, v61, v165// 0000000029e4: c909493e 3e3d4b3d
	v_dual_add_f32 v60, v60, v166 :: v_dual_add_f32 v59, v59, v167// 0000000029ec: c9094d3c 3c3b4f3b
	v_add_f32_e32 v58, v58, v168                               // 0000000029f4: 0675513a
	v_dual_add_f32 v52, v52, v169 :: v_dual_add_f32 v49, v49, v170// 0000000029f8: c9095334 34315531
	v_add_f32_e32 v48, v48, v171                               // 000000002a00: 06615730
	v_dual_add_f32 v46, v46, v172 :: v_dual_add_f32 v45, v45, v173// 000000002a04: c909592e 2e2d5b2d
	v_dual_add_f32 v44, v44, v174 :: v_dual_add_f32 v43, v43, v175// 000000002a0c: c9095d2c 2c2b5f2b
	v_add_f32_e32 v42, v42, v176                               // 000000002a14: 0655612a
	v_dual_add_f32 v36, v36, v177 :: v_dual_add_f32 v33, v33, v178// 000000002a18: c9096324 24216521
	v_add_f32_e32 v32, v32, v179                               // 000000002a20: 06416720
	v_dual_add_f32 v30, v30, v180 :: v_dual_add_f32 v29, v29, v181// 000000002a24: c909691e 1e1d6b1d
	v_dual_add_f32 v28, v28, v182 :: v_dual_add_f32 v27, v27, v183// 000000002a2c: c9096d1c 1c1b6f1b
	v_add_f32_e32 v26, v26, v184                               // 000000002a34: 0635711a
	s_cbranch_scc1 65091                                       // 000000002a38: bfa2fe43 <tessera_rocm_scaled_matmul_lds_afded1274d72297c+0x848>
	s_load_b64 s[2:3], s[0:1], 0xa8                            // 000000002a3c: f4002080 f80000a8
	v_mul_lo_u32 v6, s19, v0                                   // 000000002a44: d72c0006 02020013
	v_mul_lo_u32 v7, s18, v1                                   // 000000002a4c: d72c0007 02020212
	v_mad_co_u64_u32 v[0:1], null, s18, v0, 0                  // 000000002a54: d6fe7c00 02020012
	v_bfe_u32 v8, v14, 16, 1                                   // 000000002a5c: d6100008 0205210e
	v_or_b32_e32 v9, 0x400000, v14                             // 000000002a64: 38121cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v14, v14                           // 000000002a6c: 7c301d0e
	v_lshlrev_b64_e32 v[73:74], 1, v[2:3]                      // 000000002a70: 3e920481
	v_bfe_u32 v2, v106, 16, 1                                  // 000000002a74: d6100002 0205216a
	v_or_b32_e32 v3, 0x400000, v106                            // 000000002a7c: 3806d4ff 00400000
	s_lshl_b64 s[0:1], s[18:19], 1                             // 000000002a84: 84808112
	v_add3_u32 v1, v1, v7, v6                                  // 000000002a88: d6550001 041a0f01
	v_add3_u32 v6, v8, v14, 0x7fff                             // 000000002a90: d6550006 03fe1d08 00007fff
	v_add3_u32 v2, v2, v106, 0x7fff                            // 000000002a9c: d6550002 03fed502 00007fff
	v_or_b32_e32 v11, 0x400000, v105                           // 000000002aa8: 3816d2ff 00400000
	v_bfe_u32 v13, v102, 16, 1                                 // 000000002ab0: d610000d 02052166
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000002ab8: 3e000081
	s_wait_alu depctr_va_vcc(0)                                // 000000002abc: bf88ff9d
	v_cndmask_b32_e32 v8, v6, v9, vcc_lo                       // 000000002ac0: 02101306
	v_bfe_u32 v6, v105, 16, 1                                  // 000000002ac4: d6100006 02052169
	v_or_b32_e32 v14, 0x400000, v102                           // 000000002acc: 381cccff 00400000
	v_add3_u32 v13, v13, v102, 0x7fff                          // 000000002ad4: d655000d 03fecd0d 00007fff
	v_or_b32_e32 v17, 0x400000, v100                           // 000000002ae0: 3822c8ff 00400000
	s_wait_kmcnt 0x0                                           // 000000002ae8: bfc70000
	v_add_co_u32 v0, vcc_lo, s2, v0                            // 000000002aec: d7006a00 02020002
	s_wait_alu depctr_va_vcc(0)                                // 000000002af4: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo               // 000000002af8: d5207c01 01aa0203
	v_cmp_u_f32_e32 vcc_lo, v106, v106                         // 000000002b00: 7c30d56a
	v_add3_u32 v6, v6, v105, 0x7fff                            // 000000002b04: d6550006 03fed306 00007fff
	v_bfe_u32 v19, v97, 16, 1                                  // 000000002b10: d6100013 02052161
	v_or_b32_e32 v20, 0x400000, v97                            // 000000002b18: 3828c2ff 00400000
	v_mul_lo_u32 v21, s18, v5                                  // 000000002b20: d72c0015 02020a12
	s_wait_alu depctr_va_vcc(0)                                // 000000002b28: bf88ff9d
	v_cndmask_b32_e32 v9, v2, v3, vcc_lo                       // 000000002b2c: 02120702
	v_add_co_u32 v2, vcc_lo, v0, v73                           // 000000002b30: d7006a02 02029300
	s_wait_alu depctr_va_vcc(0)                                // 000000002b38: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v1, v74, vcc_lo              // 000000002b3c: d5207c03 01aa9501
	v_add_co_u32 v7, vcc_lo, v0, s0                            // 000000002b44: d7006a07 02000100
	s_wait_alu depctr_va_vcc(0)                                // 000000002b4c: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v1, vcc_lo              // 000000002b50: d5207c0a 01aa0201
	v_add3_u32 v19, v19, v97, 0x7fff                           // 000000002b58: d6550013 03fec313 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002b64: bf8701a3
	v_add_co_u32 v0, vcc_lo, v7, v73                           // 000000002b68: d7006a00 02029307
	s_wait_alu depctr_va_vcc(0)                                // 000000002b70: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v10, v74, vcc_lo             // 000000002b74: d5207c01 01aa950a
	v_cmp_u_f32_e32 vcc_lo, v105, v105                         // 000000002b7c: 7c30d369
	v_or_b32_e32 v22, 0x400000, v93                            // 000000002b80: 382cbaff 00400000
	v_or_b32_e32 v23, 0x400000, v94                            // 000000002b88: 382ebcff 00400000
	v_or_b32_e32 v24, 0x400000, v92                            // 000000002b90: 3830b8ff 00400000
	v_or_b32_e32 v77, 0x400000, v90                            // 000000002b98: 389ab4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002ba0: bf88ff9d
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo                     // 000000002ba4: 02161706
	v_add_co_u32 v12, vcc_lo, v7, s0                           // 000000002ba8: d7006a0c 02000107
	s_wait_alu depctr_va_vcc(0)                                // 000000002bb0: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 000000002bb4: d5207c0a 01aa1401
	v_bfe_u32 v79, v89, 16, 1                                  // 000000002bbc: d610004f 02052159
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002bc4: bf8701a3
	v_add_co_u32 v6, vcc_lo, v12, v73                          // 000000002bc8: d7006a06 0202930c
	s_wait_alu depctr_va_vcc(0)                                // 000000002bd0: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v10, v74, vcc_lo             // 000000002bd4: d5207c07 01aa950a
	v_cmp_u_f32_e32 vcc_lo, v102, v102                         // 000000002bdc: 7c30cd66
	s_clause 0x2                                               // 000000002be0: bf850002
	global_store_d16_hi_b16 v[2:3], v8, off                    // 000000002be4: ee09407c 04000000 00000002
	global_store_d16_hi_b16 v[0:1], v9, off                    // 000000002bf0: ee09407c 04800000 00000000
	global_store_d16_hi_b16 v[6:7], v11, off                   // 000000002bfc: ee09407c 05800000 00000006
	v_bfe_u32 v8, v101, 16, 1                                  // 000000002c08: d6100008 02052165
	v_add3_u32 v79, v79, v89, 0x7fff                           // 000000002c10: d655004f 03feb34f 00007fff
	v_or_b32_e32 v80, 0x400000, v89                            // 000000002c1c: 38a0b2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c24: bf88ff9d
	v_cndmask_b32_e32 v14, v13, v14, vcc_lo                    // 000000002c28: 021c1d0d
	v_add_co_u32 v11, vcc_lo, v12, s0                          // 000000002c2c: d7006a0b 0200010c
	s_wait_alu depctr_va_vcc(0)                                // 000000002c34: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 000000002c38: d5207c0a 01aa1401
	v_add3_u32 v12, v8, v101, 0x7fff                           // 000000002c40: d655000c 03fecb08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002c4c: bf870003
	v_add_co_u32 v8, vcc_lo, v11, v73                          // 000000002c50: d7006a08 0202930b
	v_or_b32_e32 v13, 0x400000, v101                           // 000000002c58: 381acaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c60: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v10, v74, vcc_lo             // 000000002c64: d5207c09 01aa950a
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 000000002c6c: 7c30cb65
	v_or_b32_e32 v84, 0x400000, v85                            // 000000002c70: 38a8aaff 00400000
	v_or_b32_e32 v87, 0x400000, v82                            // 000000002c78: 38aea4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002c80: bf88ff9d
	v_cndmask_b32_e32 v15, v12, v13, vcc_lo                    // 000000002c84: 021e1b0c
	v_bfe_u32 v12, v100, 16, 1                                 // 000000002c88: d610000c 02052164
	v_add_co_u32 v11, vcc_lo, v11, s0                          // 000000002c90: d7006a0b 0200010b
	s_wait_alu depctr_va_vcc(0)                                // 000000002c98: bf88ff9d
	v_add_co_ci_u32_e64 v10, null, s1, v10, vcc_lo             // 000000002c9c: d5207c0a 01aa1401
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ca4: bf870193
	v_add3_u32 v16, v12, v100, 0x7fff                          // 000000002ca8: d6550010 03fec90c 00007fff
	v_add_co_u32 v12, vcc_lo, v11, v73                         // 000000002cb4: d7006a0c 0202930b
	s_wait_alu depctr_va_vcc(0)                                // 000000002cbc: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002cc0: bf870003
	v_add_co_ci_u32_e64 v13, null, v10, v74, vcc_lo            // 000000002cc4: d5207c0d 01aa950a
	v_cmp_u_f32_e32 vcc_lo, v100, v100                         // 000000002ccc: 7c30c964
	s_wait_alu depctr_va_vcc(0)                                // 000000002cd0: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v17, vcc_lo                    // 000000002cd4: 02202310
	v_add_co_u32 v17, vcc_lo, v11, s0                          // 000000002cd8: d7006a11 0200010b
	s_wait_alu depctr_va_vcc(0)                                // 000000002ce0: bf88ff9d
	v_add_co_ci_u32_e64 v18, null, s1, v10, vcc_lo             // 000000002ce4: d5207c12 01aa1401
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002cec: bf870122
	v_add_co_u32 v10, vcc_lo, v17, v73                         // 000000002cf0: d7006a0a 02029311
	s_wait_alu depctr_va_vcc(0)                                // 000000002cf8: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, v18, v74, vcc_lo            // 000000002cfc: d5207c0b 01aa9512
	v_cmp_u_f32_e32 vcc_lo, v97, v97                           // 000000002d04: 7c30c361
	s_clause 0x2                                               // 000000002d08: bf850002
	global_store_d16_hi_b16 v[8:9], v14, off                   // 000000002d0c: ee09407c 07000000 00000008
	global_store_d16_hi_b16 v[12:13], v15, off                 // 000000002d18: ee09407c 07800000 0000000c
	global_store_d16_hi_b16 v[10:11], v16, off                 // 000000002d24: ee09407c 08000000 0000000a
	v_bfe_u32 v14, v93, 16, 1                                  // 000000002d30: d610000e 0205215d
	s_wait_alu depctr_va_vcc(0)                                // 000000002d38: bf88ff9d
	v_cndmask_b32_e32 v20, v19, v20, vcc_lo                    // 000000002d3c: 02282913
	v_add_co_u32 v16, vcc_lo, v17, s0                          // 000000002d40: d7006a10 02000111
	s_wait_alu depctr_va_vcc(0)                                // 000000002d48: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v18, vcc_lo             // 000000002d4c: d5207c11 01aa2401
	v_mul_lo_u32 v19, s19, v4                                  // 000000002d54: d72c0013 02020813
	v_mad_co_u64_u32 v[4:5], null, s18, v4, 0                  // 000000002d5c: d6fe7c04 02020812
	v_add3_u32 v18, v14, v93, 0x7fff                           // 000000002d64: d6550012 03febb0e 00007fff
	v_add_co_u32 v14, vcc_lo, v16, v73                         // 000000002d70: d7006a0e 02029310
	s_wait_alu depctr_va_vcc(0)                                // 000000002d78: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v17, v74, vcc_lo            // 000000002d7c: d5207c0f 01aa9511
	v_cmp_u_f32_e32 vcc_lo, v93, v93                           // 000000002d84: 7c30bb5d
	v_add3_u32 v5, v5, v21, v19                                // 000000002d88: d6550005 044e2b05
	s_wait_alu depctr_va_vcc(0)                                // 000000002d90: bf88ff9d
	v_cndmask_b32_e32 v22, v18, v22, vcc_lo                    // 000000002d94: 022c2d12
	v_add_co_u32 v19, vcc_lo, v16, s0                          // 000000002d98: d7006a13 02000110
	v_bfe_u32 v18, v94, 16, 1                                  // 000000002da0: d6100012 0205215e
	s_wait_alu depctr_va_vcc(0)                                // 000000002da8: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, s1, v17, vcc_lo             // 000000002dac: d5207c15 01aa2201
	v_lshlrev_b64_e32 v[16:17], 1, v[4:5]                      // 000000002db4: 3e200881
	v_add_co_u32 v4, vcc_lo, v19, v73                          // 000000002db8: d7006a04 02029313
	v_add3_u32 v18, v18, v94, 0x7fff                           // 000000002dc0: d6550012 03febd12 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002dcc: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v21, v74, vcc_lo             // 000000002dd0: d5207c05 01aa9515
	v_cmp_u_f32_e32 vcc_lo, v94, v94                           // 000000002dd8: 7c30bd5e
	s_wait_alu depctr_va_vcc(0)                                // 000000002ddc: bf88ff9d
	v_cndmask_b32_e32 v21, v18, v23, vcc_lo                    // 000000002de0: 022a2f12
	v_add_co_u32 v16, vcc_lo, s2, v16                          // 000000002de4: d7006a10 02022002
	s_wait_alu depctr_va_vcc(0)                                // 000000002dec: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s3, v17, vcc_lo             // 000000002df0: d5207c11 01aa2203
	v_bfe_u32 v23, v92, 16, 1                                  // 000000002df8: d6100017 0205215c
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002e00: bf8701a3
	v_add_co_u32 v18, vcc_lo, v16, v73                         // 000000002e04: d7006a12 02029310
	s_wait_alu depctr_va_vcc(0)                                // 000000002e0c: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v17, v74, vcc_lo            // 000000002e10: d5207c13 01aa9511
	s_delay_alu instid0(valu_dep_3)                            // 000000002e18: bf870003
	v_add3_u32 v23, v23, v92, 0x7fff                           // 000000002e1c: d6550017 03feb917 00007fff
	v_cmp_u_f32_e32 vcc_lo, v92, v92                           // 000000002e28: 7c30b95c
	s_clause 0x2                                               // 000000002e2c: bf850002
	global_store_d16_hi_b16 v[14:15], v20, off                 // 000000002e30: ee09407c 0a000000 0000000e
	global_store_d16_hi_b16 v[4:5], v22, off                   // 000000002e3c: ee09407c 0b000000 00000004
	global_store_d16_hi_b16 v[18:19], v21, off                 // 000000002e48: ee09407c 0a800000 00000012
	v_bfe_u32 v20, v91, 16, 1                                  // 000000002e54: d6100014 0205215b
	s_wait_alu depctr_va_vcc(0)                                // 000000002e5c: bf88ff9d
	v_cndmask_b32_e32 v22, v23, v24, vcc_lo                    // 000000002e60: 022c3117
	v_add_co_u32 v16, vcc_lo, v16, s0                          // 000000002e64: d7006a10 02000110
	s_wait_alu depctr_va_vcc(0)                                // 000000002e6c: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v17, vcc_lo             // 000000002e70: d5207c11 01aa2201
	v_add3_u32 v23, v20, v91, 0x7fff                           // 000000002e78: d6550017 03feb714 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002e84: bf870003
	v_add_co_u32 v20, vcc_lo, v16, v73                         // 000000002e88: d7006a14 02029310
	v_or_b32_e32 v24, 0x400000, v91                            // 000000002e90: 3830b6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002e98: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v17, v74, vcc_lo            // 000000002e9c: d5207c15 01aa9511
	v_cmp_u_f32_e32 vcc_lo, v91, v91                           // 000000002ea4: 7c30b75b
	s_wait_alu depctr_va_vcc(0)                                // 000000002ea8: bf88ff9d
	v_cndmask_b32_e32 v23, v23, v24, vcc_lo                    // 000000002eac: 022e3117
	v_bfe_u32 v24, v90, 16, 1                                  // 000000002eb0: d6100018 0205215a
	v_add_co_u32 v16, vcc_lo, v16, s0                          // 000000002eb8: d7006a10 02000110
	s_wait_alu depctr_va_vcc(0)                                // 000000002ec0: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, s1, v17, vcc_lo             // 000000002ec4: d5207c11 01aa2201
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ecc: bf870193
	v_add3_u32 v76, v24, v90, 0x7fff                           // 000000002ed0: d655004c 03feb518 00007fff
	v_add_co_u32 v24, vcc_lo, v16, v73                         // 000000002edc: d7006a18 02029310
	s_wait_alu depctr_va_vcc(0)                                // 000000002ee4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ee8: bf870003
	v_add_co_ci_u32_e64 v25, null, v17, v74, vcc_lo            // 000000002eec: d5207c19 01aa9511
	v_cmp_u_f32_e32 vcc_lo, v90, v90                           // 000000002ef4: 7c30b55a
	s_wait_alu depctr_va_vcc(0)                                // 000000002ef8: bf88ff9d
	v_cndmask_b32_e32 v76, v76, v77, vcc_lo                    // 000000002efc: 02989b4c
	v_add_co_u32 v77, vcc_lo, v16, s0                          // 000000002f00: d7006a4d 02000110
	s_wait_alu depctr_va_vcc(0)                                // 000000002f08: bf88ff9d
	v_add_co_ci_u32_e64 v78, null, s1, v17, vcc_lo             // 000000002f0c: d5207c4e 01aa2201
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f14: bf870122
	v_add_co_u32 v16, vcc_lo, v77, v73                         // 000000002f18: d7006a10 0202934d
	s_wait_alu depctr_va_vcc(0)                                // 000000002f20: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v78, v74, vcc_lo            // 000000002f24: d5207c11 01aa954e
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 000000002f2c: 7c30b359
	s_clause 0x2                                               // 000000002f30: bf850002
	global_store_d16_hi_b16 v[20:21], v22, off                 // 000000002f34: ee09407c 0b000000 00000014
	global_store_d16_hi_b16 v[24:25], v23, off                 // 000000002f40: ee09407c 0b800000 00000018
	global_store_d16_hi_b16 v[16:17], v76, off                 // 000000002f4c: ee09407c 26000000 00000010
	v_bfe_u32 v22, v86, 16, 1                                  // 000000002f58: d6100016 02052156
	s_wait_alu depctr_va_vcc(0)                                // 000000002f60: bf88ff9d
	v_cndmask_b32_e32 v80, v79, v80, vcc_lo                    // 000000002f64: 02a0a14f
	v_add_co_u32 v76, vcc_lo, v77, s0                          // 000000002f68: d7006a4c 0200014d
	s_wait_alu depctr_va_vcc(0)                                // 000000002f70: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s1, v78, vcc_lo             // 000000002f74: d5207c4d 01aa9c01
	v_add3_u32 v78, v22, v86, 0x7fff                           // 000000002f7c: d655004e 03fead16 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002f88: bf870003
	v_add_co_u32 v22, vcc_lo, v76, v73                         // 000000002f8c: d7006a16 0202934c
	v_or_b32_e32 v79, 0x400000, v86                            // 000000002f94: 389eacff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002f9c: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, v77, v74, vcc_lo            // 000000002fa0: d5207c17 01aa954d
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 000000002fa8: 7c30ad56
	v_bfe_u32 v86, v82, 16, 1                                  // 000000002fac: d6100056 02052152
	s_wait_alu depctr_va_vcc(0)                                // 000000002fb4: bf88ff9d
	v_cndmask_b32_e32 v81, v78, v79, vcc_lo                    // 000000002fb8: 02a29f4e
	v_add_co_u32 v79, vcc_lo, v76, s0                          // 000000002fbc: d7006a4f 0200014c
	v_bfe_u32 v78, v85, 16, 1                                  // 000000002fc4: d610004e 02052155
	s_wait_alu depctr_va_vcc(0)                                // 000000002fcc: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s1, v77, vcc_lo             // 000000002fd0: d5207c53 01aa9a01
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002fd8: bf870193
	v_add_co_u32 v76, vcc_lo, v79, v73                         // 000000002fdc: d7006a4c 0202934f
	v_add3_u32 v78, v78, v85, 0x7fff                           // 000000002fe4: d655004e 03feab4e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002ff0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ff4: bf870003
	v_add_co_ci_u32_e64 v77, null, v83, v74, vcc_lo            // 000000002ff8: d5207c4d 01aa9553
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 000000003000: 7c30ab55
	v_add3_u32 v86, v86, v82, 0x7fff                           // 000000003004: d6550056 03fea556 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003010: bf88ff9d
	v_cndmask_b32_e32 v84, v78, v84, vcc_lo                    // 000000003014: 02a8a94e
	v_add_co_u32 v85, vcc_lo, v79, s0                          // 000000003018: d7006a55 0200014f
	s_wait_alu depctr_va_vcc(0)                                // 000000003020: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s1, v83, vcc_lo             // 000000003024: d5207c53 01aaa601
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000302c: bf870122
	v_add_co_u32 v78, vcc_lo, v85, v73                         // 000000003030: d7006a4e 02029355
	s_wait_alu depctr_va_vcc(0)                                // 000000003038: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, v83, v74, vcc_lo            // 00000000303c: d5207c4f 01aa9553
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000003044: 7c30a552
	s_clause 0x2                                               // 000000003048: bf850002
	global_store_d16_hi_b16 v[22:23], v80, off                 // 00000000304c: ee09407c 28000000 00000016
	global_store_d16_hi_b16 v[76:77], v81, off                 // 000000003058: ee09407c 28800000 0000004c
	global_store_d16_hi_b16 v[78:79], v84, off                 // 000000003064: ee09407c 2a000000 0000004e
	v_bfe_u32 v81, v75, 16, 1                                  // 000000003070: d6100051 0205214b
	v_or_b32_e32 v84, 0x400000, v75                            // 000000003078: 38a896ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003080: bf88ff9d
	v_cndmask_b32_e32 v80, v86, v87, vcc_lo                    // 000000003084: 02a0af56
	v_add_co_u32 v82, vcc_lo, v85, s0                          // 000000003088: d7006a52 02000155
	s_wait_alu depctr_va_vcc(0)                                // 000000003090: bf88ff9d
	v_add_co_ci_u32_e64 v83, null, s1, v83, vcc_lo             // 000000003094: d5207c53 01aaa601
	v_add3_u32 v81, v81, v75, 0x7fff                           // 00000000309c: d6550051 03fe9751 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000030a8: bf8701a3
	v_add_co_u32 v73, vcc_lo, v82, v73                         // 0000000030ac: d7006a49 02029352
	s_wait_alu depctr_va_vcc(0)                                // 0000000030b4: bf88ff9d
	v_add_co_ci_u32_e64 v74, null, v83, v74, vcc_lo            // 0000000030b8: d5207c4a 01aa9553
	v_bfe_u32 v82, v72, 16, 1                                  // 0000000030c0: d6100052 02052148
	v_cmp_u_f32_e32 vcc_lo, v75, v75                           // 0000000030c8: 7c30974b
	global_store_d16_hi_b16 v[73:74], v80, off                 // 0000000030cc: ee09407c 28000000 00000049
	v_add3_u32 v80, v82, v72, 0x7fff                           // 0000000030d8: d6550050 03fe9152 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000030e4: bf88ff9d
	v_cndmask_b32_e32 v75, v81, v84, vcc_lo                    // 0000000030e8: 0296a951
	v_bfe_u32 v81, v71, 16, 1                                  // 0000000030ec: d6100051 02052147
	v_or_b32_e32 v82, 0x400000, v72                            // 0000000030f4: 38a490ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 0000000030fc: 7c309148
	global_store_d16_hi_b16 v[2:3], v75, off offset:32         // 000000003100: ee09407c 25800000 00002002
	v_add3_u32 v75, v81, v71, 0x7fff                           // 00000000310c: d655004b 03fe8f51 00007fff
	v_or_b32_e32 v81, 0x400000, v71                            // 000000003118: 38a28eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003120: bf88ff9d
	v_cndmask_b32_e32 v72, v80, v82, vcc_lo                    // 000000003124: 0290a550
	v_bfe_u32 v80, v70, 16, 1                                  // 000000003128: d6100050 02052146
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 000000003130: 7c308f47
	global_store_d16_hi_b16 v[0:1], v72, off offset:32         // 000000003134: ee09407c 24000000 00002000
	v_add3_u32 v72, v80, v70, 0x7fff                           // 000000003140: d6550048 03fe8d50 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000314c: bf88ff9d
	v_cndmask_b32_e32 v71, v75, v81, vcc_lo                    // 000000003150: 028ea34b
	v_bfe_u32 v75, v69, 16, 1                                  // 000000003154: d610004b 02052145
	v_or_b32_e32 v80, 0x400000, v70                            // 00000000315c: 38a08cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 000000003164: 7c308d46
	global_store_d16_hi_b16 v[6:7], v71, off offset:32         // 000000003168: ee09407c 23800000 00002006
	v_add3_u32 v71, v75, v69, 0x7fff                           // 000000003174: d6550047 03fe8b4b 00007fff
	v_or_b32_e32 v75, 0x400000, v69                            // 000000003180: 38968aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003188: bf88ff9d
	v_cndmask_b32_e32 v70, v72, v80, vcc_lo                    // 00000000318c: 028ca148
	v_bfe_u32 v72, v67, 16, 1                                  // 000000003190: d6100048 02052143
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 000000003198: 7c308b45
	global_store_d16_hi_b16 v[8:9], v70, off offset:32         // 00000000319c: ee09407c 23000000 00002008
	v_add3_u32 v70, v72, v67, 0x7fff                           // 0000000031a8: d6550046 03fe8748 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000031b4: bf88ff9d
	v_cndmask_b32_e32 v69, v71, v75, vcc_lo                    // 0000000031b8: 028a9747
	v_bfe_u32 v71, v66, 16, 1                                  // 0000000031bc: d6100047 02052142
	v_or_b32_e32 v72, 0x400000, v67                            // 0000000031c4: 389086ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 0000000031cc: 7c308743
	global_store_d16_hi_b16 v[12:13], v69, off offset:32       // 0000000031d0: ee09407c 22800000 0000200c
	v_add3_u32 v69, v71, v66, 0x7fff                           // 0000000031dc: d6550045 03fe8547 00007fff
	v_or_b32_e32 v71, 0x400000, v66                            // 0000000031e8: 388e84ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000031f0: bf88ff9d
	v_cndmask_b32_e32 v67, v70, v72, vcc_lo                    // 0000000031f4: 02869146
	v_bfe_u32 v70, v63, 16, 1                                  // 0000000031f8: d6100046 0205213f
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 000000003200: 7c308542
	global_store_d16_hi_b16 v[10:11], v67, off offset:32       // 000000003204: ee09407c 21800000 0000200a
	v_add3_u32 v67, v70, v63, 0x7fff                           // 000000003210: d6550043 03fe7f46 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000321c: bf88ff9d
	v_cndmask_b32_e32 v66, v69, v71, vcc_lo                    // 000000003220: 02848f45
	v_bfe_u32 v69, v68, 16, 1                                  // 000000003224: d6100045 02052144
	v_or_b32_e32 v70, 0x400000, v63                            // 00000000322c: 388c7eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v63, v63                           // 000000003234: 7c307f3f
	global_store_d16_hi_b16 v[14:15], v66, off offset:32       // 000000003238: ee09407c 21000000 0000200e
	v_add3_u32 v66, v69, v68, 0x7fff                           // 000000003244: d6550042 03fe8945 00007fff
	v_or_b32_e32 v69, 0x400000, v68                            // 000000003250: 388a88ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003258: bf88ff9d
	v_cndmask_b32_e32 v63, v67, v70, vcc_lo                    // 00000000325c: 027e8d43
	v_bfe_u32 v67, v65, 16, 1                                  // 000000003260: d6100043 02052141
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 000000003268: 7c308944
	v_bfe_u32 v68, v64, 16, 1                                  // 00000000326c: d6100044 02052140
	global_store_d16_hi_b16 v[4:5], v63, off offset:32         // 000000003274: ee09407c 1f800000 00002004
	v_add3_u32 v63, v67, v65, 0x7fff                           // 000000003280: d655003f 03fe8343 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000328c: bf88ff9d
	v_cndmask_b32_e32 v66, v66, v69, vcc_lo                    // 000000003290: 02848b42
	v_or_b32_e32 v67, 0x400000, v65                            // 000000003294: 388682ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v65, v65                           // 00000000329c: 7c308341
	v_bfe_u32 v65, v62, 16, 1                                  // 0000000032a0: d6100041 0205213e
	global_store_d16_hi_b16 v[18:19], v66, off offset:32       // 0000000032a8: ee09407c 21000000 00002012
	v_add3_u32 v66, v68, v64, 0x7fff                           // 0000000032b4: d6550042 03fe8144 00007fff
	v_or_b32_e32 v68, 0x400000, v64                            // 0000000032c0: 388880ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000032c8: bf88ff9d
	v_cndmask_b32_e32 v63, v63, v67, vcc_lo                    // 0000000032cc: 027e873f
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 0000000032d0: 7c308140
	global_store_d16_hi_b16 v[20:21], v63, off offset:32       // 0000000032d4: ee09407c 1f800000 00002014
	s_wait_alu depctr_va_vcc(0)                                // 0000000032e0: bf88ff9d
	v_cndmask_b32_e32 v64, v66, v68, vcc_lo                    // 0000000032e4: 02808942
	v_bfe_u32 v66, v61, 16, 1                                  // 0000000032e8: d6100042 0205213d
	v_add3_u32 v63, v65, v62, 0x7fff                           // 0000000032f0: d655003f 03fe7d41 00007fff
	v_or_b32_e32 v65, 0x400000, v62                            // 0000000032fc: 38827cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000003304: 7c307d3e
	global_store_d16_hi_b16 v[24:25], v64, off offset:32       // 000000003308: ee09407c 20000000 00002018
	v_add3_u32 v64, v66, v61, 0x7fff                           // 000000003314: d6550040 03fe7b42 00007fff
	v_or_b32_e32 v66, 0x400000, v61                            // 000000003320: 38847aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003328: bf88ff9d
	v_cndmask_b32_e32 v62, v63, v65, vcc_lo                    // 00000000332c: 027c833f
	v_bfe_u32 v63, v60, 16, 1                                  // 000000003330: d610003f 0205213c
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000003338: 7c307b3d
	global_store_d16_hi_b16 v[16:17], v62, off offset:32       // 00000000333c: ee09407c 1f000000 00002010
	v_add3_u32 v62, v63, v60, 0x7fff                           // 000000003348: d655003e 03fe793f 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003354: bf88ff9d
	v_cndmask_b32_e32 v61, v64, v66, vcc_lo                    // 000000003358: 027a8540
	v_bfe_u32 v64, v59, 16, 1                                  // 00000000335c: d6100040 0205213b
	v_or_b32_e32 v63, 0x400000, v60                            // 000000003364: 387e78ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v60, v60                           // 00000000336c: 7c30793c
	global_store_d16_hi_b16 v[22:23], v61, off offset:32       // 000000003370: ee09407c 1e800000 00002016
	v_add3_u32 v61, v64, v59, 0x7fff                           // 00000000337c: d655003d 03fe7740 00007fff
	v_or_b32_e32 v64, 0x400000, v59                            // 000000003388: 388076ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003390: bf88ff9d
	v_cndmask_b32_e32 v60, v62, v63, vcc_lo                    // 000000003394: 02787f3e
	v_bfe_u32 v62, v58, 16, 1                                  // 000000003398: d610003e 0205213a
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 0000000033a0: 7c30773b
	global_store_d16_hi_b16 v[76:77], v60, off offset:32       // 0000000033a4: ee09407c 1e000000 0000204c
	v_add3_u32 v60, v62, v58, 0x7fff                           // 0000000033b0: d655003c 03fe753e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000033bc: bf88ff9d
	v_cndmask_b32_e32 v59, v61, v64, vcc_lo                    // 0000000033c0: 0276813d
	v_bfe_u32 v61, v57, 16, 1                                  // 0000000033c4: d610003d 02052139
	v_or_b32_e32 v62, 0x400000, v58                            // 0000000033cc: 387c74ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 0000000033d4: 7c30753a
	global_store_d16_hi_b16 v[78:79], v59, off offset:32       // 0000000033d8: ee09407c 1d800000 0000204e
	v_add3_u32 v59, v61, v57, 0x7fff                           // 0000000033e4: d655003b 03fe733d 00007fff
	v_or_b32_e32 v61, 0x400000, v57                            // 0000000033f0: 387a72ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000033f8: bf88ff9d
	v_cndmask_b32_e32 v58, v60, v62, vcc_lo                    // 0000000033fc: 02747d3c
	v_bfe_u32 v60, v56, 16, 1                                  // 000000003400: d610003c 02052138
	v_cmp_u_f32_e32 vcc_lo, v57, v57                           // 000000003408: 7c307339
	global_store_d16_hi_b16 v[73:74], v58, off offset:32       // 00000000340c: ee09407c 1d000000 00002049
	v_add3_u32 v58, v60, v56, 0x7fff                           // 000000003418: d655003a 03fe713c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003424: bf88ff9d
	v_cndmask_b32_e32 v57, v59, v61, vcc_lo                    // 000000003428: 02727b3b
	v_bfe_u32 v59, v55, 16, 1                                  // 00000000342c: d610003b 02052137
	v_or_b32_e32 v60, 0x400000, v56                            // 000000003434: 387870ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v56, v56                           // 00000000343c: 7c307138
	global_store_d16_hi_b16 v[2:3], v57, off offset:64         // 000000003440: ee09407c 1c800000 00004002
	v_add3_u32 v57, v59, v55, 0x7fff                           // 00000000344c: d6550039 03fe6f3b 00007fff
	v_or_b32_e32 v59, 0x400000, v55                            // 000000003458: 38766eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003460: bf88ff9d
	v_cndmask_b32_e32 v56, v58, v60, vcc_lo                    // 000000003464: 0270793a
	v_bfe_u32 v58, v54, 16, 1                                  // 000000003468: d610003a 02052136
	v_cmp_u_f32_e32 vcc_lo, v55, v55                           // 000000003470: 7c306f37
	global_store_d16_hi_b16 v[0:1], v56, off offset:64         // 000000003474: ee09407c 1c000000 00004000
	v_add3_u32 v56, v58, v54, 0x7fff                           // 000000003480: d6550038 03fe6d3a 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000348c: bf88ff9d
	v_cndmask_b32_e32 v55, v57, v59, vcc_lo                    // 000000003490: 026e7739
	v_bfe_u32 v57, v53, 16, 1                                  // 000000003494: d6100039 02052135
	v_or_b32_e32 v58, 0x400000, v54                            // 00000000349c: 38746cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v54, v54                           // 0000000034a4: 7c306d36
	global_store_d16_hi_b16 v[6:7], v55, off offset:64         // 0000000034a8: ee09407c 1b800000 00004006
	v_add3_u32 v55, v57, v53, 0x7fff                           // 0000000034b4: d6550037 03fe6b39 00007fff
	v_or_b32_e32 v57, 0x400000, v53                            // 0000000034c0: 38726aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000034c8: bf88ff9d
	v_cndmask_b32_e32 v54, v56, v58, vcc_lo                    // 0000000034cc: 026c7538
	v_bfe_u32 v56, v51, 16, 1                                  // 0000000034d0: d6100038 02052133
	v_cmp_u_f32_e32 vcc_lo, v53, v53                           // 0000000034d8: 7c306b35
	global_store_d16_hi_b16 v[8:9], v54, off offset:64         // 0000000034dc: ee09407c 1b000000 00004008
	v_add3_u32 v54, v56, v51, 0x7fff                           // 0000000034e8: d6550036 03fe6738 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000034f4: bf88ff9d
	v_cndmask_b32_e32 v53, v55, v57, vcc_lo                    // 0000000034f8: 026a7337
	v_bfe_u32 v55, v50, 16, 1                                  // 0000000034fc: d6100037 02052132
	v_or_b32_e32 v56, 0x400000, v51                            // 000000003504: 387066ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v51, v51                           // 00000000350c: 7c306733
	global_store_d16_hi_b16 v[12:13], v53, off offset:64       // 000000003510: ee09407c 1a800000 0000400c
	v_add3_u32 v53, v55, v50, 0x7fff                           // 00000000351c: d6550035 03fe6537 00007fff
	v_or_b32_e32 v55, 0x400000, v50                            // 000000003528: 386e64ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003530: bf88ff9d
	v_cndmask_b32_e32 v51, v54, v56, vcc_lo                    // 000000003534: 02667136
	v_bfe_u32 v54, v47, 16, 1                                  // 000000003538: d6100036 0205212f
	v_cmp_u_f32_e32 vcc_lo, v50, v50                           // 000000003540: 7c306532
	global_store_d16_hi_b16 v[10:11], v51, off offset:64       // 000000003544: ee09407c 19800000 0000400a
	v_add3_u32 v51, v54, v47, 0x7fff                           // 000000003550: d6550033 03fe5f36 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000355c: bf88ff9d
	v_cndmask_b32_e32 v50, v53, v55, vcc_lo                    // 000000003560: 02646f35
	v_bfe_u32 v53, v52, 16, 1                                  // 000000003564: d6100035 02052134
	v_or_b32_e32 v54, 0x400000, v47                            // 00000000356c: 386c5eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v47, v47                           // 000000003574: 7c305f2f
	global_store_d16_hi_b16 v[14:15], v50, off offset:64       // 000000003578: ee09407c 19000000 0000400e
	v_add3_u32 v50, v53, v52, 0x7fff                           // 000000003584: d6550032 03fe6935 00007fff
	v_or_b32_e32 v53, 0x400000, v52                            // 000000003590: 386a68ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003598: bf88ff9d
	v_cndmask_b32_e32 v47, v51, v54, vcc_lo                    // 00000000359c: 025e6d33
	v_bfe_u32 v51, v49, 16, 1                                  // 0000000035a0: d6100033 02052131
	v_cmp_u_f32_e32 vcc_lo, v52, v52                           // 0000000035a8: 7c306934
	v_bfe_u32 v52, v48, 16, 1                                  // 0000000035ac: d6100034 02052130
	global_store_d16_hi_b16 v[4:5], v47, off offset:64         // 0000000035b4: ee09407c 17800000 00004004
	v_add3_u32 v47, v51, v49, 0x7fff                           // 0000000035c0: d655002f 03fe6333 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000035cc: bf88ff9d
	v_cndmask_b32_e32 v50, v50, v53, vcc_lo                    // 0000000035d0: 02646b32
	v_or_b32_e32 v51, 0x400000, v49                            // 0000000035d4: 386662ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v49, v49                           // 0000000035dc: 7c306331
	v_bfe_u32 v49, v46, 16, 1                                  // 0000000035e0: d6100031 0205212e
	global_store_d16_hi_b16 v[18:19], v50, off offset:64       // 0000000035e8: ee09407c 19000000 00004012
	v_add3_u32 v50, v52, v48, 0x7fff                           // 0000000035f4: d6550032 03fe6134 00007fff
	v_or_b32_e32 v52, 0x400000, v48                            // 000000003600: 386860ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003608: bf88ff9d
	v_cndmask_b32_e32 v47, v47, v51, vcc_lo                    // 00000000360c: 025e672f
	v_cmp_u_f32_e32 vcc_lo, v48, v48                           // 000000003610: 7c306130
	global_store_d16_hi_b16 v[20:21], v47, off offset:64       // 000000003614: ee09407c 17800000 00004014
	s_wait_alu depctr_va_vcc(0)                                // 000000003620: bf88ff9d
	v_cndmask_b32_e32 v48, v50, v52, vcc_lo                    // 000000003624: 02606932
	v_bfe_u32 v50, v45, 16, 1                                  // 000000003628: d6100032 0205212d
	v_add3_u32 v47, v49, v46, 0x7fff                           // 000000003630: d655002f 03fe5d31 00007fff
	v_or_b32_e32 v49, 0x400000, v46                            // 00000000363c: 38625cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v46, v46                           // 000000003644: 7c305d2e
	global_store_d16_hi_b16 v[24:25], v48, off offset:64       // 000000003648: ee09407c 18000000 00004018
	v_add3_u32 v48, v50, v45, 0x7fff                           // 000000003654: d6550030 03fe5b32 00007fff
	v_or_b32_e32 v50, 0x400000, v45                            // 000000003660: 38645aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003668: bf88ff9d
	v_cndmask_b32_e32 v46, v47, v49, vcc_lo                    // 00000000366c: 025c632f
	v_bfe_u32 v47, v44, 16, 1                                  // 000000003670: d610002f 0205212c
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 000000003678: 7c305b2d
	global_store_d16_hi_b16 v[16:17], v46, off offset:64       // 00000000367c: ee09407c 17000000 00004010
	v_add3_u32 v46, v47, v44, 0x7fff                           // 000000003688: d655002e 03fe592f 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003694: bf88ff9d
	v_cndmask_b32_e32 v45, v48, v50, vcc_lo                    // 000000003698: 025a6530
	v_bfe_u32 v48, v43, 16, 1                                  // 00000000369c: d6100030 0205212b
	v_or_b32_e32 v47, 0x400000, v44                            // 0000000036a4: 385e58ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 0000000036ac: 7c30592c
	global_store_d16_hi_b16 v[22:23], v45, off offset:64       // 0000000036b0: ee09407c 16800000 00004016
	v_add3_u32 v45, v48, v43, 0x7fff                           // 0000000036bc: d655002d 03fe5730 00007fff
	v_or_b32_e32 v48, 0x400000, v43                            // 0000000036c8: 386056ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000036d0: bf88ff9d
	v_cndmask_b32_e32 v44, v46, v47, vcc_lo                    // 0000000036d4: 02585f2e
	v_bfe_u32 v46, v42, 16, 1                                  // 0000000036d8: d610002e 0205212a
	v_cmp_u_f32_e32 vcc_lo, v43, v43                           // 0000000036e0: 7c30572b
	global_store_d16_hi_b16 v[76:77], v44, off offset:64       // 0000000036e4: ee09407c 16000000 0000404c
	v_add3_u32 v44, v46, v42, 0x7fff                           // 0000000036f0: d655002c 03fe552e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000036fc: bf88ff9d
	v_cndmask_b32_e32 v43, v45, v48, vcc_lo                    // 000000003700: 0256612d
	v_bfe_u32 v45, v41, 16, 1                                  // 000000003704: d610002d 02052129
	v_or_b32_e32 v46, 0x400000, v42                            // 00000000370c: 385c54ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v42, v42                           // 000000003714: 7c30552a
	global_store_d16_hi_b16 v[78:79], v43, off offset:64       // 000000003718: ee09407c 15800000 0000404e
	v_add3_u32 v43, v45, v41, 0x7fff                           // 000000003724: d655002b 03fe532d 00007fff
	v_or_b32_e32 v45, 0x400000, v41                            // 000000003730: 385a52ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003738: bf88ff9d
	v_cndmask_b32_e32 v42, v44, v46, vcc_lo                    // 00000000373c: 02545d2c
	v_bfe_u32 v44, v40, 16, 1                                  // 000000003740: d610002c 02052128
	v_cmp_u_f32_e32 vcc_lo, v41, v41                           // 000000003748: 7c305329
	global_store_d16_hi_b16 v[73:74], v42, off offset:64       // 00000000374c: ee09407c 15000000 00004049
	v_add3_u32 v42, v44, v40, 0x7fff                           // 000000003758: d655002a 03fe512c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003764: bf88ff9d
	v_cndmask_b32_e32 v41, v43, v45, vcc_lo                    // 000000003768: 02525b2b
	v_bfe_u32 v43, v39, 16, 1                                  // 00000000376c: d610002b 02052127
	v_or_b32_e32 v44, 0x400000, v40                            // 000000003774: 385850ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v40, v40                           // 00000000377c: 7c305128
	global_store_d16_hi_b16 v[2:3], v41, off offset:96         // 000000003780: ee09407c 14800000 00006002
	v_add3_u32 v2, v43, v39, 0x7fff                            // 00000000378c: d6550002 03fe4f2b 00007fff
	v_or_b32_e32 v3, 0x400000, v39                             // 000000003798: 38064eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000037a0: bf88ff9d
	v_cndmask_b32_e32 v40, v42, v44, vcc_lo                    // 0000000037a4: 0250592a
	v_bfe_u32 v41, v38, 16, 1                                  // 0000000037a8: d6100029 02052126
	v_cmp_u_f32_e32 vcc_lo, v39, v39                           // 0000000037b0: 7c304f27
	global_store_d16_hi_b16 v[0:1], v40, off offset:96         // 0000000037b4: ee09407c 14000000 00006000
	v_add3_u32 v0, v41, v38, 0x7fff                            // 0000000037c0: d6550000 03fe4d29 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000037cc: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 0000000037d0: 02040702
	v_bfe_u32 v3, v37, 16, 1                                   // 0000000037d4: d6100003 02052125
	v_or_b32_e32 v1, 0x400000, v38                             // 0000000037dc: 38024cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v38, v38                           // 0000000037e4: 7c304d26
	global_store_d16_hi_b16 v[6:7], v2, off offset:96          // 0000000037e8: ee09407c 01000000 00006006
	v_add3_u32 v2, v3, v37, 0x7fff                             // 0000000037f4: d6550002 03fe4b03 00007fff
	v_or_b32_e32 v3, 0x400000, v37                             // 000000003800: 38064aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003808: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000380c: 02000300
	v_bfe_u32 v1, v35, 16, 1                                   // 000000003810: d6100001 02052123
	v_cmp_u_f32_e32 vcc_lo, v37, v37                           // 000000003818: 7c304b25
	v_or_b32_e32 v6, 0x400000, v27                             // 00000000381c: 380c36ff 00400000
	v_or_b32_e32 v7, 0x400000, v26                             // 000000003824: 380e34ff 00400000
	global_store_d16_hi_b16 v[8:9], v0, off offset:96          // 00000000382c: ee09407c 00000000 00006008
	v_add3_u32 v0, v1, v35, 0x7fff                             // 000000003838: d6550000 03fe4701 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003844: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003848: 02040702
	v_bfe_u32 v3, v34, 16, 1                                   // 00000000384c: d6100003 02052122
	v_or_b32_e32 v1, 0x400000, v35                             // 000000003854: 380246ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v35, v35                           // 00000000385c: 7c304723
	global_store_d16_hi_b16 v[12:13], v2, off offset:96        // 000000003860: ee09407c 01000000 0000600c
	v_add3_u32 v2, v3, v34, 0x7fff                             // 00000000386c: d6550002 03fe4503 00007fff
	v_or_b32_e32 v3, 0x400000, v34                             // 000000003878: 380644ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003880: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003884: 02000300
	v_bfe_u32 v1, v31, 16, 1                                   // 000000003888: d6100001 0205211f
	v_cmp_u_f32_e32 vcc_lo, v34, v34                           // 000000003890: 7c304522
	global_store_d16_hi_b16 v[10:11], v0, off offset:96        // 000000003894: ee09407c 00000000 0000600a
	v_add3_u32 v0, v1, v31, 0x7fff                             // 0000000038a0: d6550000 03fe3f01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000038ac: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 0000000038b0: 02040702
	v_bfe_u32 v3, v36, 16, 1                                   // 0000000038b4: d6100003 02052124
	v_or_b32_e32 v1, 0x400000, v31                             // 0000000038bc: 38023eff 00400000
	v_cmp_u_f32_e32 vcc_lo, v31, v31                           // 0000000038c4: 7c303f1f
	global_store_d16_hi_b16 v[14:15], v2, off offset:96        // 0000000038c8: ee09407c 01000000 0000600e
	v_add3_u32 v2, v3, v36, 0x7fff                             // 0000000038d4: d6550002 03fe4903 00007fff
	v_or_b32_e32 v3, 0x400000, v36                             // 0000000038e0: 380648ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000038e8: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000038ec: 02000300
	v_bfe_u32 v1, v33, 16, 1                                   // 0000000038f0: d6100001 02052121
	v_cmp_u_f32_e32 vcc_lo, v36, v36                           // 0000000038f8: 7c304924
	global_store_d16_hi_b16 v[4:5], v0, off offset:96          // 0000000038fc: ee09407c 00000000 00006004
	v_add3_u32 v0, v1, v33, 0x7fff                             // 000000003908: d6550000 03fe4301 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003914: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 000000003918: 02040702
	v_bfe_u32 v3, v32, 16, 1                                   // 00000000391c: d6100003 02052120
	v_or_b32_e32 v1, 0x400000, v33                             // 000000003924: 380242ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v33, v33                           // 00000000392c: 7c304321
	v_bfe_u32 v4, v27, 16, 1                                   // 000000003930: d6100004 0205211b
	global_store_d16_hi_b16 v[18:19], v2, off offset:96        // 000000003938: ee09407c 01000000 00006012
	v_add3_u32 v2, v3, v32, 0x7fff                             // 000000003944: d6550002 03fe4103 00007fff
	v_or_b32_e32 v3, 0x400000, v32                             // 000000003950: 380640ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003958: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000395c: 02000300
	v_bfe_u32 v1, v30, 16, 1                                   // 000000003960: d6100001 0205211e
	v_cmp_u_f32_e32 vcc_lo, v32, v32                           // 000000003968: 7c304120
	v_or_b32_e32 v5, 0x400000, v28                             // 00000000396c: 380a38ff 00400000
	v_add3_u32 v4, v4, v27, 0x7fff                             // 000000003974: d6550004 03fe3704 00007fff
	global_store_d16_hi_b16 v[20:21], v0, off offset:96        // 000000003980: ee09407c 00000000 00006014
	v_add3_u32 v0, v1, v30, 0x7fff                             // 00000000398c: d6550000 03fe3d01 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003998: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 00000000399c: 02040702
	v_bfe_u32 v3, v29, 16, 1                                   // 0000000039a0: d6100003 0205211d
	v_or_b32_e32 v1, 0x400000, v30                             // 0000000039a8: 38023cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 0000000039b0: 7c303d1e
	global_store_d16_hi_b16 v[24:25], v2, off offset:96        // 0000000039b4: ee09407c 01000000 00006018
	v_add3_u32 v2, v3, v29, 0x7fff                             // 0000000039c0: d6550002 03fe3b03 00007fff
	v_or_b32_e32 v3, 0x400000, v29                             // 0000000039cc: 38063aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000039d4: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000039d8: 02000300
	v_bfe_u32 v1, v28, 16, 1                                   // 0000000039dc: d6100001 0205211c
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 0000000039e4: 7c303b1d
	s_delay_alu instid0(valu_dep_2)                            // 0000000039e8: bf870002
	v_add3_u32 v1, v1, v28, 0x7fff                             // 0000000039ec: d6550001 03fe3901 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000039f8: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo                       // 0000000039fc: 02040702
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 000000003a00: 7c30391c
	v_bfe_u32 v3, v26, 16, 1                                   // 000000003a04: d6100003 0205211a
	s_wait_alu depctr_va_vcc(0)                                // 000000003a0c: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo                       // 000000003a10: 02020b01
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 000000003a14: 7c30371b
	s_delay_alu instid0(valu_dep_3)                            // 000000003a18: bf870003
	v_add3_u32 v3, v3, v26, 0x7fff                             // 000000003a1c: d6550003 03fe3503 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003a28: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       // 000000003a2c: 02080d04
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 000000003a30: 7c30351a
	s_wait_alu depctr_va_vcc(0)                                // 000000003a34: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v7, vcc_lo                       // 000000003a38: 02060f03
	s_clause 0x3                                               // 000000003a3c: bf850003
	global_store_d16_hi_b16 v[16:17], v0, off offset:96        // 000000003a40: ee09407c 00000000 00006010
	global_store_d16_hi_b16 v[22:23], v2, off offset:96        // 000000003a4c: ee09407c 01000000 00006016
	global_store_d16_hi_b16 v[76:77], v1, off offset:96        // 000000003a58: ee09407c 00800000 0000604c
	global_store_d16_hi_b16 v[78:79], v4, off offset:96        // 000000003a64: ee09407c 02000000 0000604e
	global_store_d16_hi_b16 v[73:74], v3, off offset:96        // 000000003a70: ee09407c 01800000 00006049
	s_nop 0                                                    // 000000003a7c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000003a80: bfb60003
	s_endpgm                                                   // 000000003a84: bfb00000
	s_code_end                                                 // 000000003a88: bf9f0000
	s_code_end                                                 // 000000003a8c: bf9f0000
	s_code_end                                                 // 000000003a90: bf9f0000
	s_code_end                                                 // 000000003a94: bf9f0000
	s_code_end                                                 // 000000003a98: bf9f0000
	s_code_end                                                 // 000000003a9c: bf9f0000
	s_code_end                                                 // 000000003aa0: bf9f0000
	s_code_end                                                 // 000000003aa4: bf9f0000
	s_code_end                                                 // 000000003aa8: bf9f0000
	s_code_end                                                 // 000000003aac: bf9f0000
	s_code_end                                                 // 000000003ab0: bf9f0000
	s_code_end                                                 // 000000003ab4: bf9f0000
	s_code_end                                                 // 000000003ab8: bf9f0000
	s_code_end                                                 // 000000003abc: bf9f0000
	s_code_end                                                 // 000000003ac0: bf9f0000
	s_code_end                                                 // 000000003ac4: bf9f0000
	s_code_end                                                 // 000000003ac8: bf9f0000
	s_code_end                                                 // 000000003acc: bf9f0000
	s_code_end                                                 // 000000003ad0: bf9f0000
	s_code_end                                                 // 000000003ad4: bf9f0000
	s_code_end                                                 // 000000003ad8: bf9f0000
	s_code_end                                                 // 000000003adc: bf9f0000
	s_code_end                                                 // 000000003ae0: bf9f0000
	s_code_end                                                 // 000000003ae4: bf9f0000
	s_code_end                                                 // 000000003ae8: bf9f0000
	s_code_end                                                 // 000000003aec: bf9f0000
	s_code_end                                                 // 000000003af0: bf9f0000
	s_code_end                                                 // 000000003af4: bf9f0000
	s_code_end                                                 // 000000003af8: bf9f0000
	s_code_end                                                 // 000000003afc: bf9f0000
	s_code_end                                                 // 000000003b00: bf9f0000
	s_code_end                                                 // 000000003b04: bf9f0000
	s_code_end                                                 // 000000003b08: bf9f0000
	s_code_end                                                 // 000000003b0c: bf9f0000
	s_code_end                                                 // 000000003b10: bf9f0000
	s_code_end                                                 // 000000003b14: bf9f0000
	s_code_end                                                 // 000000003b18: bf9f0000
	s_code_end                                                 // 000000003b1c: bf9f0000
	s_code_end                                                 // 000000003b20: bf9f0000
	s_code_end                                                 // 000000003b24: bf9f0000
	s_code_end                                                 // 000000003b28: bf9f0000
	s_code_end                                                 // 000000003b2c: bf9f0000
	s_code_end                                                 // 000000003b30: bf9f0000
	s_code_end                                                 // 000000003b34: bf9f0000
	s_code_end                                                 // 000000003b38: bf9f0000
	s_code_end                                                 // 000000003b3c: bf9f0000
	s_code_end                                                 // 000000003b40: bf9f0000
	s_code_end                                                 // 000000003b44: bf9f0000
	s_code_end                                                 // 000000003b48: bf9f0000
	s_code_end                                                 // 000000003b4c: bf9f0000
	s_code_end                                                 // 000000003b50: bf9f0000
	s_code_end                                                 // 000000003b54: bf9f0000
	s_code_end                                                 // 000000003b58: bf9f0000
	s_code_end                                                 // 000000003b5c: bf9f0000
	s_code_end                                                 // 000000003b60: bf9f0000
	s_code_end                                                 // 000000003b64: bf9f0000
	s_code_end                                                 // 000000003b68: bf9f0000
	s_code_end                                                 // 000000003b6c: bf9f0000
	s_code_end                                                 // 000000003b70: bf9f0000
	s_code_end                                                 // 000000003b74: bf9f0000
	s_code_end                                                 // 000000003b78: bf9f0000
	s_code_end                                                 // 000000003b7c: bf9f0000
	s_code_end                                                 // 000000003b80: bf9f0000
	s_code_end                                                 // 000000003b84: bf9f0000
	s_code_end                                                 // 000000003b88: bf9f0000
	s_code_end                                                 // 000000003b8c: bf9f0000
	s_code_end                                                 // 000000003b90: bf9f0000
	s_code_end                                                 // 000000003b94: bf9f0000
	s_code_end                                                 // 000000003b98: bf9f0000
	s_code_end                                                 // 000000003b9c: bf9f0000
	s_code_end                                                 // 000000003ba0: bf9f0000
	s_code_end                                                 // 000000003ba4: bf9f0000
	s_code_end                                                 // 000000003ba8: bf9f0000
	s_code_end                                                 // 000000003bac: bf9f0000
	s_code_end                                                 // 000000003bb0: bf9f0000
	s_code_end                                                 // 000000003bb4: bf9f0000
	s_code_end                                                 // 000000003bb8: bf9f0000
	s_code_end                                                 // 000000003bbc: bf9f0000
	s_code_end                                                 // 000000003bc0: bf9f0000
	s_code_end                                                 // 000000003bc4: bf9f0000
	s_code_end                                                 // 000000003bc8: bf9f0000
	s_code_end                                                 // 000000003bcc: bf9f0000
	s_code_end                                                 // 000000003bd0: bf9f0000
	s_code_end                                                 // 000000003bd4: bf9f0000
	s_code_end                                                 // 000000003bd8: bf9f0000
	s_code_end                                                 // 000000003bdc: bf9f0000
	s_code_end                                                 // 000000003be0: bf9f0000
	s_code_end                                                 // 000000003be4: bf9f0000
	s_code_end                                                 // 000000003be8: bf9f0000
	s_code_end                                                 // 000000003bec: bf9f0000
	s_code_end                                                 // 000000003bf0: bf9f0000
	s_code_end                                                 // 000000003bf4: bf9f0000
	s_code_end                                                 // 000000003bf8: bf9f0000
	s_code_end                                                 // 000000003bfc: bf9f0000
	s_code_end                                                 // 000000003c00: bf9f0000
	s_code_end                                                 // 000000003c04: bf9f0000
	s_code_end                                                 // 000000003c08: bf9f0000
	s_code_end                                                 // 000000003c0c: bf9f0000
	s_code_end                                                 // 000000003c10: bf9f0000
	s_code_end                                                 // 000000003c14: bf9f0000
	s_code_end                                                 // 000000003c18: bf9f0000
	s_code_end                                                 // 000000003c1c: bf9f0000
	s_code_end                                                 // 000000003c20: bf9f0000
	s_code_end                                                 // 000000003c24: bf9f0000
	s_code_end                                                 // 000000003c28: bf9f0000
	s_code_end                                                 // 000000003c2c: bf9f0000
	s_code_end                                                 // 000000003c30: bf9f0000
	s_code_end                                                 // 000000003c34: bf9f0000
	s_code_end                                                 // 000000003c38: bf9f0000
	s_code_end                                                 // 000000003c3c: bf9f0000
	s_code_end                                                 // 000000003c40: bf9f0000
	s_code_end                                                 // 000000003c44: bf9f0000
	s_code_end                                                 // 000000003c48: bf9f0000
	s_code_end                                                 // 000000003c4c: bf9f0000
	s_code_end                                                 // 000000003c50: bf9f0000
	s_code_end                                                 // 000000003c54: bf9f0000
	s_code_end                                                 // 000000003c58: bf9f0000
	s_code_end                                                 // 000000003c5c: bf9f0000
	s_code_end                                                 // 000000003c60: bf9f0000
	s_code_end                                                 // 000000003c64: bf9f0000
	s_code_end                                                 // 000000003c68: bf9f0000
	s_code_end                                                 // 000000003c6c: bf9f0000
	s_code_end                                                 // 000000003c70: bf9f0000
	s_code_end                                                 // 000000003c74: bf9f0000
	s_code_end                                                 // 000000003c78: bf9f0000
	s_code_end                                                 // 000000003c7c: bf9f0000
