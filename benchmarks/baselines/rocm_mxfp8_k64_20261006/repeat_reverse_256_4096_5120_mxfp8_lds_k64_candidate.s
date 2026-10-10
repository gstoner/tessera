
/tmp/tmpg3ngvsop.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

disassembly of section .text:

0000000000001b00 <tessera_rocm_scaled_matmul_lds_533dbb9ffad73571>:
	s_clause 0x5                                               // 000000001b00: bf850005
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001b04: f4002280 f8000008
	s_load_b64 s[8:9], s[0:1], 0x30                            // 000000001b0c: f4002200 f8000030
	s_load_b64 s[4:5], s[0:1], 0x58                            // 000000001b14: f4002100 f8000058
	s_load_b64 s[44:45], s[0:1], 0x80                          // 000000001b1c: f4002b00 f8000080
	s_load_b64 s[12:13], s[0:1], 0xd8                          // 000000001b24: f4002300 f80000d8
	s_load_b128 s[40:43], s[0:1], 0xc8                         // 000000001b2c: f4004a00 f80000c8
	v_lshrrev_b32_e32 v1, 1, v0                                // 000000001b34: 32020081
	s_mov_b32 s6, ttmp7                                        // 000000001b38: be860073
	s_ashr_i32 s7, ttmp7, 31                                   // 000000001b3c: 86079f73
	v_dual_mov_b32 v45, 0 :: v_dual_and_b32 v2, 15, v0         // 000000001b40: ca240080 2d02008f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_3)// 000000001b48: bf8701c2
	v_and_b32_e32 v3, 0x60, v1                                 // 000000001b4c: 360602ff 00000060
	s_lshl_b64 s[16:17], s[6:7], 7                             // 000000001b54: 84908706
	v_and_b32_e32 v10, 8, v1                                   // 000000001b58: 36140288
	v_dual_mov_b32 v41, s17 :: v_dual_lshlrev_b32 v4, 4, v0    // 000000001b5c: ca220011 29040084
	v_or_b32_e32 v6, s16, v3                                   // 000000001b64: 380c0610
	v_or_b32_e32 v5, 16, v3                                    // 000000001b68: 380a0690
	v_or_b32_e32 v3, v3, v2                                    // 000000001b6c: 38060503
	v_lshrrev_b32_e32 v8, 2, v0                                // 000000001b70: 32100082
	v_and_b32_e32 v0, 47, v0                                   // 000000001b74: 360000af
	v_or_b32_e32 v40, v6, v10                                  // 000000001b78: 38501506
	s_mov_b32 s2, ttmp9                                        // 000000001b7c: be820075
	v_mul_u32_u24_e32 v1, 0x50, v3                             // 000000001b80: 160206ff 00000050
	s_ashr_i32 s3, ttmp9, 31                                   // 000000001b88: 86039f75
	v_or_b32_e32 v2, v5, v2                                    // 000000001b8c: 38040505
	s_wait_kmcnt 0x0                                           // 000000001b90: bfc70000
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[40:41]                // 000000001b94: 7ca85028
	s_lshl_b64 s[14:15], s[2:3], 6                             // 000000001b98: 848e8602
	s_delay_alu instid0(salu_cycle_1)                          // 000000001b9c: bf870009
	v_dual_mov_b32 v47, s15 :: v_dual_and_b32 v44, 48, v4      // 000000001ba0: ca24000f 2f2c08b0
	v_mul_u32_u24_e32 v4, 0x50, v8                             // 000000001ba8: 160810ff 00000050
	v_mul_u32_u24_e32 v2, 0x50, v2                             // 000000001bb0: 160404ff 00000050
	v_cndmask_b32_e32 v13, 0, v40, vcc_lo                      // 000000001bb8: 021a5080
	v_or_b32_e32 v12, 1, v10                                   // 000000001bbc: 38181481
	v_or_b32_e32 v55, v1, v10                                  // 000000001bc0: 386e1501
	v_mov_b32_e32 v1, s17                                      // 000000001bc4: 7e020211
	v_or_b32_e32 v11, 16, v0                                   // 000000001bc8: 38160090
	v_or_b32_e32 v46, s14, v0                                  // 000000001bcc: 385c000e
	v_mul_u32_u24_e32 v3, 0x50, v0                             // 000000001bd0: 160600ff 00000050
	v_or_b32_e32 v0, v12, v6                                   // 000000001bd8: 38000d0c
	v_or_b32_e32 v15, 2, v10                                   // 000000001bdc: 381e1482
	v_add_nc_u32_e32 v54, v4, v44                              // 000000001be0: 4a6c5904
	v_mul_u32_u24_e32 v4, 0x50, v11                            // 000000001be4: 160816ff 00000050
	v_or_b32_e32 v16, 3, v10                                   // 000000001bec: 38201483
	v_cndmask_b32_e32 v14, 0, v41, vcc_lo                      // 000000001bf0: 021c5280
	v_cmp_gt_i64_e32 vcc_lo, s[42:43], v[46:47]                // 000000001bf4: 7ca85c2a
	v_or_b32_e32 v56, v2, v10                                  // 000000001bf8: 38701502
	v_or_b32_e32 v57, v10, v3                                  // 000000001bfc: 3872070a
	v_or_b32_e32 v2, v15, v6                                   // 000000001c00: 38040d0f
	v_mov_b32_e32 v3, s17                                      // 000000001c04: 7e060211
	v_cmp_gt_i64_e64 s2, s[40:41], v[0:1]                      // 000000001c08: d4540002 02020028
	v_or_b32_e32 v9, s16, v5                                   // 000000001c10: 38120a10
	v_or_b32_e32 v60, v4, v10                                  // 000000001c14: 38781504
	v_or_b32_e32 v4, v16, v6                                   // 000000001c18: 38080d10
	v_mov_b32_e32 v5, s17                                      // 000000001c1c: 7e0a0211
	v_or_b32_e32 v20, 4, v10                                   // 000000001c20: 38281484
	s_wait_alu depctr_va_vcc(0)                                // 000000001c24: bf88ff9d
	v_cndmask_b32_e64 v63, 0, s15, vcc_lo                      // 000000001c28: d501003f 01a81e80
	v_cndmask_b32_e32 v65, 0, v46, vcc_lo                      // 000000001c30: 02825c80
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[2:3]                  // 000000001c34: 7ca80428
	v_cndmask_b32_e64 v17, 0, v0, s2                           // 000000001c38: d5010011 000a0080
	v_cndmask_b32_e64 v18, 0, v1, s2                           // 000000001c40: d5010012 000a0280
	v_cmp_gt_i64_e64 s2, s[40:41], v[4:5]                      // 000000001c48: d4540002 02020828
	v_or_b32_e32 v0, v20, v6                                   // 000000001c50: 38000d14
	v_or_b32_e32 v22, 5, v10                                   // 000000001c54: 382c1485
	v_or_b32_e32 v25, 6, v10                                   // 000000001c58: 38321486
	s_wait_alu depctr_va_vcc(0)                                // 000000001c5c: bf88ff9d
	v_cndmask_b32_e32 v19, 0, v2, vcc_lo                       // 000000001c60: 02260480
	v_or_b32_e32 v42, v9, v10                                  // 000000001c64: 38541509
	v_cndmask_b32_e32 v21, 0, v3, vcc_lo                       // 000000001c68: 022a0680
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[0:1]                  // 000000001c6c: 7ca80028
	s_wait_alu depctr_va_sdst(0)                               // 000000001c70: bf88f19f
	v_cndmask_b32_e64 v23, 0, v4, s2                           // 000000001c74: d5010017 000a0880
	v_or_b32_e32 v2, v22, v6                                   // 000000001c7c: 38040d16
	v_or_b32_e32 v4, v25, v6                                   // 000000001c80: 38080d19
	v_or_b32_e32 v26, 7, v10                                   // 000000001c84: 38341487
	v_cndmask_b32_e64 v24, 0, v5, s2                           // 000000001c88: d5010018 000a0a80
	s_wait_alu depctr_va_vcc(0)                                // 000000001c90: bf88ff9d
	v_cndmask_b32_e32 v27, 0, v0, vcc_lo                       // 000000001c94: 02360080
	v_cmp_gt_i64_e64 s2, s[40:41], v[2:3]                      // 000000001c98: d4540002 02020428
	v_cndmask_b32_e32 v28, 0, v1, vcc_lo                       // 000000001ca0: 02380280
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[4:5]                  // 000000001ca4: 7ca80828
	v_or_b32_e32 v0, s14, v11                                  // 000000001ca8: 3800160e
	v_mov_b32_e32 v43, s17                                     // 000000001cac: 7e560211
	v_mov_b32_e32 v1, s15                                      // 000000001cb0: 7e02020f
	v_or_b32_e32 v6, v26, v6                                   // 000000001cb4: 380c0d1a
	v_mov_b32_e32 v7, s17                                      // 000000001cb8: 7e0e0211
	s_wait_alu depctr_va_sdst(0)                               // 000000001cbc: bf88f19f
	v_cndmask_b32_e64 v30, 0, v3, s2                           // 000000001cc0: d501001e 000a0680
	v_or_b32_e32 v3, v9, v15                                   // 000000001cc8: 38061f09
	s_wait_alu depctr_va_vcc(0)                                // 000000001ccc: bf88ff9d
	v_dual_cndmask_b32 v11, 0, v4 :: v_dual_mov_b32 v4, s17    // 000000001cd0: ca500880 0b040011
	v_cndmask_b32_e32 v31, 0, v5, vcc_lo                       // 000000001cd8: 023e0a80
	v_cmp_gt_i64_e32 vcc_lo, s[42:43], v[0:1]                  // 000000001cdc: 7ca8002a
	v_cndmask_b32_e64 v29, 0, v2, s2                           // 000000001ce0: d501001d 000a0480
	v_cmp_gt_i64_e64 s2, s[40:41], v[6:7]                      // 000000001ce8: d4540002 02020c28
	v_mov_b32_e32 v2, s17                                      // 000000001cf0: 7e040211
	v_or_b32_e32 v1, v9, v12                                   // 000000001cf4: 38021909
	v_or_b32_e32 v5, v9, v16                                   // 000000001cf8: 380a2109
	s_wait_alu depctr_va_vcc(0)                                // 000000001cfc: bf88ff9d
	v_cndmask_b32_e64 v75, 0, s15, vcc_lo                      // 000000001d00: d501004b 01a81e80
	v_cndmask_b32_e32 v76, 0, v0, vcc_lo                       // 000000001d08: 02980080
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[42:43]                // 000000001d0c: 7ca85428
	s_wait_alu depctr_va_sdst(0)                               // 000000001d10: bf88f19f
	v_cndmask_b32_e64 v32, 0, v6, s2                           // 000000001d14: d5010020 000a0c80
	v_mov_b32_e32 v6, s17                                      // 000000001d1c: 7e0c0211
	v_cndmask_b32_e64 v33, 0, v7, s2                           // 000000001d20: d5010021 000a0e80
	v_cmp_gt_i64_e64 s2, s[40:41], v[1:2]                      // 000000001d28: d4540002 02020228
	v_or_b32_e32 v0, v9, v20                                   // 000000001d30: 38002909
	s_wait_alu depctr_va_vcc(0)                                // 000000001d34: bf88ff9d
	v_cndmask_b32_e32 v10, 0, v42, vcc_lo                      // 000000001d38: 02145480
	v_cndmask_b32_e32 v12, 0, v43, vcc_lo                      // 000000001d3c: 02185680
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[3:4]                  // 000000001d40: 7ca80628
	v_mov_b32_e32 v124, v45                                    // 000000001d44: 7ef8032d
	s_wait_alu depctr_va_sdst(0)                               // 000000001d48: bf88f19f
	v_cndmask_b32_e64 v15, 0, v1, s2                           // 000000001d4c: d501000f 000a0280
	v_mov_b32_e32 v1, s17                                      // 000000001d54: 7e020211
	v_cndmask_b32_e64 v16, 0, v2, s2                           // 000000001d58: d5010010 000a0480
	v_cmp_gt_i64_e64 s2, s[40:41], v[5:6]                      // 000000001d60: d4540002 02020a28
	s_wait_alu depctr_va_vcc(0)                                // 000000001d68: bf88ff9d
	v_cndmask_b32_e32 v20, 0, v3, vcc_lo                       // 000000001d6c: 02280680
	v_cndmask_b32_e32 v34, 0, v4, vcc_lo                       // 000000001d70: 02440880
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[0:1]                  // 000000001d74: 7ca80028
	v_mov_b32_e32 v3, s17                                      // 000000001d78: 7e060211
	v_or_b32_e32 v2, v9, v22                                   // 000000001d7c: 38042d09
	v_or_b32_e32 v4, v9, v25                                   // 000000001d80: 38083309
	s_wait_alu depctr_va_sdst(0)                               // 000000001d84: bf88f19f
	v_cndmask_b32_e64 v35, 0, v5, s2                           // 000000001d88: d5010023 000a0a80
	s_wait_alu depctr_va_vcc(0)                                // 000000001d90: bf88ff9d
	v_dual_mov_b32 v5, s17 :: v_dual_cndmask_b32 v22, 0, v0    // 000000001d94: ca120011 05160080
	v_dual_cndmask_b32 v25, 0, v1 :: v_dual_mov_b32 v114, v45  // 000000001d9c: ca500280 1972012d
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[2:3]                  // 000000001da4: 7ca80428
	v_or_b32_e32 v0, v9, v26                                   // 000000001da8: 38003509
	v_cndmask_b32_e64 v36, 0, v6, s2                           // 000000001dac: d5010024 000a0c80
	v_cmp_gt_i64_e64 s2, s[40:41], v[4:5]                      // 000000001db4: d4540002 02020828
	v_or_b32_e32 v6, s16, v8                                   // 000000001dbc: 380c1010
	s_lshr_b64 s[6:7], s[12:13], 5                             // 000000001dc0: 8586850c
	s_wait_alu depctr_va_vcc(0)                                // 000000001dc4: bf88ff9d
	v_dual_cndmask_b32 v37, 0, v2 :: v_dual_mov_b32 v94, v45   // 000000001dc8: ca500480 255e012d
	v_dual_cndmask_b32 v9, 0, v3 :: v_dual_mov_b32 v74, v45    // 000000001dd0: ca500680 094a012d
	v_cmp_gt_i64_e32 vcc_lo, s[40:41], v[0:1]                  // 000000001dd8: 7ca80028
	s_wait_alu depctr_va_sdst(0)                               // 000000001ddc: bf88f19f
	v_cndmask_b32_e64 v7, 0, v4, s2                            // 000000001de0: d5010007 000a0880
	v_mul_lo_u32 v4, s13, v6                                   // 000000001de8: d72c0004 02020c0d
	v_mad_co_u64_u32 v[2:3], null, s12, v6, v[44:45]           // 000000001df0: d6fe7c02 04b20c0c
	v_cndmask_b32_e64 v26, 0, v5, s2                           // 000000001df8: d501001a 000a0a80
	v_or_b32_e32 v5, s14, v8                                   // 000000001e00: 380a100e
	s_wait_alu depctr_va_vcc(0)                                // 000000001e04: bf88ff9d
	v_cndmask_b32_e32 v38, 0, v0, vcc_lo                       // 000000001e08: 024c0080
	v_or_b32_e32 v0, 64, v6                                    // 000000001e0c: 38000cc0
	v_cndmask_b32_e32 v6, 0, v1, vcc_lo                        // 000000001e10: 020c0280
	s_mul_i32 s2, s12, s17                                     // 000000001e14: 9602110c
	v_mul_lo_u32 v50, s13, v5                                  // 000000001e18: d72c0032 02020a0d
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e20: bf88ff9e
	v_add3_u32 v8, v4, v3, s2                                  // 000000001e24: d6550008 000a0704
	v_mul_lo_u32 v39, s13, v0                                  // 000000001e2c: d72c0027 0202000d
	v_mad_co_u64_u32 v[0:1], null, s12, v0, v[44:45]           // 000000001e34: d6fe7c00 04b2000c
	v_mad_co_u64_u32 v[3:4], null, s12, v5, v[44:45]           // 000000001e3c: d6fe7c03 04b20a0c
	v_add_co_u32 v48, vcc_lo, s10, v2                          // 000000001e44: d7006a30 0202040a
	s_wait_alu depctr_va_vcc(0)                                // 000000001e4c: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s11, v8, vcc_lo             // 000000001e50: d5207c31 01aa100b
	s_mul_i32 s3, s12, s15                                     // 000000001e58: 96030f0c
	v_mul_lo_u32 v14, s6, v14                                  // 000000001e5c: d72c000e 02021c06
	v_add3_u32 v8, v39, v1, s2                                 // 000000001e64: d6550008 000a0327
	s_lshr_b32 s2, s13, 5                                      // 000000001e6c: 8502850d
	v_mad_co_u64_u32 v[1:2], null, s6, v13, s[4:5]             // 000000001e70: d6fe7c01 00121a06
	s_wait_alu depctr_sa_sdst(0)                               // 000000001e78: bf88ff9e
	v_mul_lo_u32 v44, s2, v13                                  // 000000001e7c: d72c002c 02021a02
	v_add3_u32 v39, v50, v4, s3                                // 000000001e84: d6550027 000e0932
	v_mul_lo_u32 v13, s6, v18                                  // 000000001e8c: d72c000d 02022406
	v_mul_lo_u32 v18, s2, v17                                  // 000000001e94: d72c0012 02022202
	v_mad_co_u64_u32 v[4:5], null, s6, v17, s[4:5]             // 000000001e9c: d6fe7c04 00122206
	v_add_co_u32 v50, vcc_lo, s10, v0                          // 000000001ea4: d7006a32 0202000a
	s_wait_alu depctr_va_vcc(0)                                // 000000001eac: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s11, v8, vcc_lo             // 000000001eb0: d5207c33 01aa100b
	v_add3_u32 v0, v44, v2, v14                                // 000000001eb8: d6550000 043a052c
	v_add_co_u32 v52, vcc_lo, s8, v3                           // 000000001ec0: d7006a34 02020608
	v_add3_u32 v5, v18, v5, v13                                // 000000001ec8: d6550005 04360b12
	v_mul_lo_u32 v8, s6, v21                                   // 000000001ed0: d72c0008 02022a06
	v_mul_lo_u32 v13, s2, v19                                  // 000000001ed8: d72c000d 02022602
	v_mad_co_u64_u32 v[2:3], null, s6, v19, s[4:5]             // 000000001ee0: d6fe7c02 00122606
	s_wait_alu depctr_va_vcc(0)                                // 000000001ee8: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s9, v39, vcc_lo             // 000000001eec: d5207c35 01aa4e09
	v_add_co_u32 v85, vcc_lo, v1, 1                            // 000000001ef4: d7006a55 02010301
	s_wait_alu depctr_va_vcc(0)                                // 000000001efc: bf88ff9d
	v_add_co_ci_u32_e64 v86, null, 0, v0, vcc_lo               // 000000001f00: d5207c56 01aa0080
	v_add_co_u32 v87, vcc_lo, v4, 1                            // 000000001f08: d7006a57 02010304
	s_wait_alu depctr_va_vcc(0)                                // 000000001f10: bf88ff9d
	v_add_co_ci_u32_e64 v88, null, 0, v5, vcc_lo               // 000000001f14: d5207c58 01aa0a80
	v_mul_lo_u32 v4, s6, v24                                   // 000000001f1c: d72c0004 02023006
	v_mul_lo_u32 v5, s2, v23                                   // 000000001f24: d72c0005 02022e02
	v_mad_co_u64_u32 v[0:1], null, s6, v23, s[4:5]             // 000000001f2c: d6fe7c00 00122e06
	v_add3_u32 v3, v13, v3, v8                                 // 000000001f34: d6550003 0422070d
	v_add_co_u32 v89, vcc_lo, v65, s42                         // 000000001f3c: d7006a59 02005541
	s_wait_alu depctr_va_vcc(0)                                // 000000001f44: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s43, v63, vcc_lo            // 000000001f48: d5207c5a 01aa7e2b
	v_add_co_u32 v91, vcc_lo, v2, 1                            // 000000001f50: d7006a5b 02010302
	s_wait_alu depctr_va_vcc(0)                                // 000000001f58: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, 0, v3, vcc_lo               // 000000001f5c: d5207c5d 01aa0680
	v_add3_u32 v5, v5, v1, v4                                  // 000000001f64: d6550005 04120305
	v_mul_lo_u32 v8, s6, v28                                   // 000000001f6c: d72c0008 02023806
	v_mul_lo_u32 v13, s2, v27                                  // 000000001f74: d72c000d 02023602
	v_mad_co_u64_u32 v[1:2], null, s6, v27, s[4:5]             // 000000001f7c: d6fe7c01 00123606
	v_mul_lo_u32 v6, s6, v6                                    // 000000001f84: d72c0006 02020c06
	v_mul_lo_u32 v14, s2, v38                                  // 000000001f8c: d72c000e 02024c02
	v_mad_co_u64_u32 v[3:4], null, s6, v38, s[4:5]             // 000000001f94: d6fe7c03 00124c06
	v_add_co_u32 v95, vcc_lo, v0, 1                            // 000000001f9c: d7006a5f 02010300
	s_wait_alu depctr_va_vcc(0)                                // 000000001fa4: bf88ff9d
	v_add_co_ci_u32_e64 v96, null, 0, v5, vcc_lo               // 000000001fa8: d5207c60 01aa0a80
	v_add3_u32 v0, v13, v2, v8                                 // 000000001fb0: d6550000 0422050d
	v_mul_lo_u32 v8, s6, v30                                   // 000000001fb8: d72c0008 02023c06
	v_mul_lo_u32 v13, s2, v29                                  // 000000001fc0: d72c000d 02023a02
	v_add3_u32 v2, v14, v4, v6                                 // 000000001fc8: d6550002 041a090e
	v_mad_co_u64_u32 v[4:5], null, s6, v29, s[4:5]             // 000000001fd0: d6fe7c04 00123a06
	v_add_co_u32 v97, vcc_lo, v76, s42                         // 000000001fd8: d7006a61 0200554c
	v_mul_lo_u32 v14, s6, v26                                  // 000000001fe0: d72c000e 02023406
	v_mul_lo_u32 v17, s2, v7                                   // 000000001fe8: d72c0011 02020e02
	v_mad_co_u64_u32 v[6:7], null, s6, v7, s[4:5]              // 000000001ff0: d6fe7c06 00120e06
	s_wait_alu depctr_va_vcc(0)                                // 000000001ff8: bf88ff9d
	v_add_co_ci_u32_e64 v98, null, s43, v75, vcc_lo            // 000000001ffc: d5207c62 01aa962b
	v_add_co_u32 v99, vcc_lo, v1, 1                            // 000000002004: d7006a63 02010301
	s_wait_alu depctr_va_vcc(0)                                // 00000000200c: bf88ff9d
	v_add_co_ci_u32_e64 v100, null, 0, v0, vcc_lo              // 000000002010: d5207c64 01aa0080
	v_add_co_u32 v102, vcc_lo, v3, 1                           // 000000002018: d7006a66 02010303
	s_wait_alu depctr_va_vcc(0)                                // 000000002020: bf88ff9d
	v_add_co_ci_u32_e64 v103, null, 0, v2, vcc_lo              // 000000002024: d5207c67 01aa0480
	v_add3_u32 v5, v13, v5, v8                                 // 00000000202c: d6550005 04220b0d
	v_mul_lo_u32 v8, s6, v31                                   // 000000002034: d72c0008 02023e06
	v_mul_lo_u32 v13, s2, v11                                  // 00000000203c: d72c000d 02021602
	v_mad_co_u64_u32 v[0:1], null, s6, v11, s[4:5]             // 000000002044: d6fe7c00 00121606
	v_mul_lo_u32 v9, s6, v9                                    // 00000000204c: d72c0009 02021206
	v_mul_lo_u32 v11, s2, v37                                  // 000000002054: d72c000b 02024a02
	v_mad_co_u64_u32 v[2:3], null, s6, v37, s[4:5]             // 00000000205c: d6fe7c02 00124a06
	v_add3_u32 v7, v17, v7, v14                                // 000000002064: d6550007 043a0f11
	v_add_co_u32 v104, vcc_lo, v4, 1                           // 00000000206c: d7006a68 02010304
	s_wait_alu depctr_va_vcc(0)                                // 000000002074: bf88ff9d
	v_add_co_ci_u32_e64 v106, null, 0, v5, vcc_lo              // 000000002078: d5207c6a 01aa0a80
	v_add_co_u32 v107, vcc_lo, v6, 1                           // 000000002080: d7006a6b 02010306
	s_wait_alu depctr_va_vcc(0)                                // 000000002088: bf88ff9d
	v_add_co_ci_u32_e64 v108, null, 0, v7, vcc_lo              // 00000000208c: d5207c6c 01aa0e80
	v_add3_u32 v1, v13, v1, v8                                 // 000000002094: d6550001 0422030d
	v_add3_u32 v7, v11, v3, v9                                 // 00000000209c: d6550007 0426070b
	v_mul_lo_u32 v8, s6, v33                                   // 0000000020a4: d72c0008 02024206
	v_mul_lo_u32 v9, s2, v32                                   // 0000000020ac: d72c0009 02024002
	v_mad_co_u64_u32 v[3:4], null, s6, v32, s[4:5]             // 0000000020b4: d6fe7c03 00124006
	v_mul_lo_u32 v11, s6, v25                                  // 0000000020bc: d72c000b 02023206
	v_mul_lo_u32 v13, s2, v22                                  // 0000000020c4: d72c000d 02022c02
	v_mad_co_u64_u32 v[5:6], null, s6, v22, s[4:5]             // 0000000020cc: d6fe7c05 00122c06
	v_add_co_u32 v110, vcc_lo, v0, 1                           // 0000000020d4: d7006a6e 02010300
	s_wait_alu depctr_va_vcc(0)                                // 0000000020dc: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, 0, v1, vcc_lo              // 0000000020e0: d5207c6f 01aa0280
	v_add_co_u32 v112, vcc_lo, v2, 1                           // 0000000020e8: d7006a70 02010302
	v_add3_u32 v2, v9, v4, v8                                  // 0000000020f0: d6550002 04220909
	v_mul_lo_u32 v8, s6, v12                                   // 0000000020f8: d72c0008 02021806
	v_mul_lo_u32 v9, s2, v10                                   // 000000002100: d72c0009 02021402
	v_mad_co_u64_u32 v[0:1], null, s6, v10, s[4:5]             // 000000002108: d6fe7c00 00121406
	s_wait_alu depctr_va_vcc(0)                                // 000000002110: bf88ff9d
	v_add_co_ci_u32_e64 v113, null, 0, v7, vcc_lo              // 000000002114: d5207c71 01aa0e80
	v_add3_u32 v4, v13, v6, v11                                // 00000000211c: d6550004 042e0d0d
	v_mul_lo_u32 v10, s6, v36                                  // 000000002124: d72c000a 02024806
	v_mul_lo_u32 v11, s2, v35                                  // 00000000212c: d72c000b 02024602
	v_mad_co_u64_u32 v[6:7], null, s6, v35, s[4:5]             // 000000002134: d6fe7c06 00124606
	v_add_co_u32 v115, vcc_lo, v3, 1                           // 00000000213c: d7006a73 02010303
	s_wait_alu depctr_va_vcc(0)                                // 000000002144: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, 0, v2, vcc_lo              // 000000002148: d5207c74 01aa0480
	v_add_co_u32 v117, vcc_lo, v5, 1                           // 000000002150: d7006a75 02010305
	v_add3_u32 v5, v9, v1, v8                                  // 000000002158: d6550005 04220309
	v_mul_lo_u32 v8, s6, v16                                   // 000000002160: d72c0008 02022006
	v_mul_lo_u32 v9, s2, v15                                   // 000000002168: d72c0009 02021e02
	v_mad_co_u64_u32 v[1:2], null, s6, v15, s[4:5]             // 000000002170: d6fe7c01 00121e06
	s_wait_alu depctr_va_vcc(0)                                // 000000002178: bf88ff9d
	v_add_co_ci_u32_e64 v118, null, 0, v4, vcc_lo              // 00000000217c: d5207c76 01aa0880
	v_add3_u32 v7, v11, v7, v10                                // 000000002184: d6550007 042a0f0b
	v_mul_lo_u32 v10, s6, v34                                  // 00000000218c: d72c000a 02024406
	v_mul_lo_u32 v11, s2, v20                                  // 000000002194: d72c000b 02022802
	v_mad_co_u64_u32 v[3:4], null, s6, v20, s[4:5]             // 00000000219c: d6fe7c03 00122806
	v_add_co_u32 v120, vcc_lo, v0, 1                           // 0000000021a4: d7006a78 02010300
	v_add3_u32 v0, v9, v2, v8                                  // 0000000021ac: d6550000 04220509
	s_wait_alu depctr_va_vcc(0)                                // 0000000021b4: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, 0, v5, vcc_lo              // 0000000021b8: d5207c79 01aa0a80
	v_add_co_u32 v122, vcc_lo, v6, 1                           // 0000000021c0: d7006a7a 02010306
	v_add3_u32 v2, v11, v4, v10                                // 0000000021c8: d6550002 042a090b
	s_wait_alu depctr_va_vcc(0)                                // 0000000021d0: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, 0, v7, vcc_lo              // 0000000021d4: d5207c7b 01aa0e80
	v_add_co_u32 v125, vcc_lo, v1, 1                           // 0000000021dc: d7006a7d 02010301
	s_wait_alu depctr_va_vcc(0)                                // 0000000021e4: bf88ff9d
	v_add_co_ci_u32_e64 v126, null, 0, v0, vcc_lo              // 0000000021e8: d5207c7e 01aa0080
	v_add_co_u32 v127, vcc_lo, v3, 1                           // 0000000021f0: d7006a7f 02010303
	s_wait_alu depctr_va_vcc(0)                                // 0000000021f8: bf88ff9d
	v_add_co_ci_u32_e64 v128, null, 0, v2, vcc_lo              // 0000000021fc: d5207c80 01aa0480
	v_dual_mov_b32 v119, v45 :: v_dual_mov_b32 v72, v45        // 000000002204: ca10012d 7748012d
	v_dual_mov_b32 v109, v45 :: v_dual_mov_b32 v70, v45        // 00000000220c: ca10012d 6d46012d
	v_dual_mov_b32 v105, v45 :: v_dual_mov_b32 v92, v45        // 000000002214: ca10012d 695c012d
	v_dual_mov_b32 v101, v45 :: v_dual_mov_b32 v84, v45        // 00000000221c: ca10012d 6554012d
	v_dual_mov_b32 v77, v45 :: v_dual_mov_b32 v82, v45         // 000000002224: ca10012d 4d52012d
	v_dual_mov_b32 v73, v45 :: v_dual_mov_b32 v80, v45         // 00000000222c: ca10012d 4950012d
	v_dual_mov_b32 v71, v45 :: v_dual_mov_b32 v78, v45         // 000000002234: ca10012d 474e012d
	v_dual_mov_b32 v69, v45 :: v_dual_mov_b32 v68, v45         // 00000000223c: ca10012d 4544012d
	v_dual_mov_b32 v67, v45 :: v_dual_mov_b32 v66, v45         // 000000002244: ca10012d 4342012d
	v_dual_mov_b32 v83, v45 :: v_dual_mov_b32 v64, v45         // 00000000224c: ca10012d 5340012d
	v_dual_mov_b32 v81, v45 :: v_dual_mov_b32 v62, v45         // 000000002254: ca10012d 513e012d
	v_dual_mov_b32 v79, v45 :: v_dual_mov_b32 v58, v45         // 00000000225c: ca10012d 4f3a012d
	v_dual_mov_b32 v61, v45 :: v_dual_mov_b32 v44, v45         // 000000002264: ca10012d 3d2c012d
	v_mov_b32_e32 v59, v45                                     // 00000000226c: 7e76032d
	s_and_b32 s48, s6, -2                                      // 000000002270: 8b30c206
	s_mov_b32 s49, s7                                          // 000000002274: beb10007
	s_lshl_b64 s[46:47], s[42:43], 1                           // 000000002278: 84ae812a
	s_mov_b64 s[50:51], 0                                      // 00000000227c: beb20180
	global_load_b128 v[0:3], v[48:49], off                     // 000000002280: ee05c07c 00000000 00000030
	global_load_b128 v[4:7], v[50:51], off                     // 00000000228c: ee05c07c 00000004 00000032
	global_load_b128 v[8:11], v[52:53], off                    // 000000002298: ee05c07c 00000008 00000034
	v_add_co_u32 v14, vcc_lo, s44, v65                         // 0000000022a4: d7006a0e 0202822c
	v_add_co_u32 v32, s10, s44, v76                            // 0000000022ac: d7000a20 0202982c
	s_wait_alu depctr_va_vcc(0)                                // 0000000022b4: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, s45, v63, vcc_lo            // 0000000022b8: d5207c0f 01aa7e2d
	s_wait_alu depctr_va_sdst(0)                               // 0000000022c0: bf88f19f
	v_add_co_ci_u32_e64 v33, null, s45, v75, s10               // 0000000022c4: d5207c21 002a962d
	v_add_co_u32 v16, s2, v85, s50                             // 0000000022cc: d7000210 02006555
	v_add_co_u32 v18, s3, v87, s50                             // 0000000022d4: d7000312 02006557
	v_add_co_u32 v20, s4, v91, s50                             // 0000000022dc: d7000414 0200655b
	v_add_co_u32 v22, s5, v95, s50                             // 0000000022e4: d7000516 0200655f
	v_add_co_u32 v24, s6, v99, s50                             // 0000000022ec: d7000618 02006563
	v_add_co_u32 v26, s7, v104, s50                            // 0000000022f4: d700071a 02006568
	v_add_co_u32 v28, s8, v110, s50                            // 0000000022fc: d700081c 0200656e
	v_add_co_u32 v30, s9, v115, s50                            // 000000002304: d700091e 02006573
	v_add_co_u32 v34, s11, v120, s50                           // 00000000230c: d7000b22 02006578
	v_add_co_u32 v36, s12, v125, s50                           // 000000002314: d7000c24 0200657d
	v_add_co_u32 v38, s13, v127, s50                           // 00000000231c: d7000d26 0200657f
	v_add_co_u32 v129, s14, v122, s50                          // 000000002324: d7000e81 0200657a
	v_add_co_u32 v131, s15, v117, s50                          // 00000000232c: d7000f83 02006575
	v_add_co_u32 v133, s16, v112, s50                          // 000000002334: d7001085 02006570
	v_add_co_u32 v135, s17, v107, s50                          // 00000000233c: d7001187 0200656b
	v_add_co_u32 v137, s18, v102, s50                          // 000000002344: d7001289 02006566
	v_add_co_u32 v139, s19, s44, v89                           // 00000000234c: d700138b 0202b22c
	v_add_co_u32 v141, s20, s44, v97                           // 000000002354: d700148d 0202c22c
	s_barrier_signal -1                                        // 00000000235c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002360: bf94ffff
	s_wait_alu depctr_va_sdst(0)                               // 000000002364: bf88f19f
	v_add_co_ci_u32_e64 v17, null, s51, v86, s2                // 000000002368: d5207c11 000aac33
	v_add_co_ci_u32_e64 v19, null, s51, v88, s3                // 000000002370: d5207c13 000eb033
	v_add_co_ci_u32_e64 v21, null, s51, v93, s4                // 000000002378: d5207c15 0012ba33
	v_add_co_ci_u32_e64 v23, null, s51, v96, s5                // 000000002380: d5207c17 0016c033
	v_add_co_ci_u32_e64 v25, null, s51, v100, s6               // 000000002388: d5207c19 001ac833
	v_add_co_ci_u32_e64 v27, null, s51, v106, s7               // 000000002390: d5207c1b 001ed433
	v_add_co_ci_u32_e64 v29, null, s51, v111, s8               // 000000002398: d5207c1d 0022de33
	v_add_co_ci_u32_e64 v31, null, s51, v116, s9               // 0000000023a0: d5207c1f 0026e833
	v_add_co_ci_u32_e64 v35, null, s51, v121, s11              // 0000000023a8: d5207c23 002ef233
	v_add_co_ci_u32_e64 v37, null, s51, v126, s12              // 0000000023b0: d5207c25 0032fc33
	v_add_co_ci_u32_e64 v39, null, s51, v128, s13              // 0000000023b8: d5207c27 00370033
	v_add_co_ci_u32_e64 v130, null, s51, v123, s14             // 0000000023c0: d5207c82 003af633
	v_add_co_ci_u32_e64 v132, null, s51, v118, s15             // 0000000023c8: d5207c84 003eec33
	v_add_co_ci_u32_e64 v134, null, s51, v113, s16             // 0000000023d0: d5207c86 0042e233
	v_add_co_ci_u32_e64 v136, null, s51, v108, s17             // 0000000023d8: d5207c88 0046d833
	v_add_co_ci_u32_e64 v138, null, s51, v103, s18             // 0000000023e0: d5207c8a 004ace33
	v_add_co_ci_u32_e64 v140, null, s45, v90, s19              // 0000000023e8: d5207c8c 004eb42d
	v_add_co_ci_u32_e64 v142, null, s45, v98, s20              // 0000000023f0: d5207c8e 0052c42d
	v_add_nc_u32_e32 v12, 0x2800, v57                          // 0000000023f8: 4a1872ff 00002800
	v_add_nc_u32_e32 v13, 0x2800, v60                          // 000000002400: 4a1a78ff 00002800
	v_add_co_u32 v48, s38, v48, 64                             // 000000002408: d7002630 02018130
	v_add_co_u32 v50, s39, v50, 64                             // 000000002410: d7002732 02018132
	v_add_co_u32 v52, s40, v52, 64                             // 000000002418: d7002834 02018134
	s_wait_alu depctr_va_sdst(0)                               // 000000002420: bf88f19f
	v_add_co_ci_u32_e64 v49, null, 0, v49, s38                 // 000000002424: d5207c31 009a6280
	v_add_co_ci_u32_e64 v51, null, 0, v51, s39                 // 00000000242c: d5207c33 009e6680
	v_add_co_ci_u32_e64 v53, null, 0, v53, s40                 // 000000002434: d5207c35 00a26a80
	s_add_nc_u64 s[50:51], s[50:51], 2                         // 00000000243c: a9b28232
	s_add_nc_u64 s[44:45], s[44:45], s[46:47]                  // 000000002440: a9ac2e2c
	s_wait_loadcnt 0x2                                         // 000000002444: bfc00002
	ds_store_b128 v54, v[0:3]                                  // 000000002448: db7c0000 00000036
	s_wait_loadcnt 0x1                                         // 000000002450: bfc00001
	ds_store_b128 v54, v[4:7] offset:5120                      // 000000002454: db7c1400 00000436
	s_wait_loadcnt 0x0                                         // 00000000245c: bfc00000
	ds_store_b128 v54, v[8:11] offset:10240                    // 000000002460: db7c2800 00000836
	s_wait_dscnt 0x0                                           // 000000002468: bfc60000
	s_barrier_signal -1                                        // 00000000246c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002470: bf94ffff
	s_clause 0x1                                               // 000000002474: bf850001
	global_load_u8 v169, v[14:15], off                         // 000000002478: ee04007c 000000a9 0000000e
	global_load_u8 v170, v[32:33], off                         // 000000002484: ee04007c 000000aa 00000020
	s_clause 0x3                                               // 000000002490: bf850003
	global_load_u8 v171, v[16:17], off offset:-1               // 000000002494: ee04007c 000000ab ffffff10
	global_load_u8 v172, v[18:19], off offset:-1               // 0000000024a0: ee04007c 000000ac ffffff12
	global_load_u8 v173, v[20:21], off offset:-1               // 0000000024ac: ee04007c 000000ad ffffff14
	global_load_u8 v174, v[22:23], off offset:-1               // 0000000024b8: ee04007c 000000ae ffffff16
	global_load_u8 v175, v[139:140], off                       // 0000000024c4: ee04007c 000000af 0000008b
	global_load_u8 v176, v[24:25], off offset:-1               // 0000000024d0: ee04007c 000000b0 ffffff18
	global_load_u8 v177, v[141:142], off                       // 0000000024dc: ee04007c 000000b1 0000008d
	s_clause 0x1a                                              // 0000000024e8: bf85001a
	global_load_u8 v178, v[26:27], off offset:-1               // 0000000024ec: ee04007c 000000b2 ffffff1a
	global_load_u8 v179, v[28:29], off offset:-1               // 0000000024f8: ee04007c 000000b3 ffffff1c
	global_load_u8 v180, v[30:31], off offset:-1               // 000000002504: ee04007c 000000b4 ffffff1e
	global_load_u8 v181, v[34:35], off offset:-1               // 000000002510: ee04007c 000000b5 ffffff22
	global_load_u8 v182, v[36:37], off offset:-1               // 00000000251c: ee04007c 000000b6 ffffff24
	global_load_u8 v183, v[38:39], off offset:-1               // 000000002528: ee04007c 000000b7 ffffff26
	global_load_u8 v184, v[129:130], off offset:-1             // 000000002534: ee04007c 000000b8 ffffff81
	global_load_u8 v185, v[131:132], off offset:-1             // 000000002540: ee04007c 000000b9 ffffff83
	global_load_u8 v186, v[133:134], off offset:-1             // 00000000254c: ee04007c 000000ba ffffff85
	global_load_u8 v187, v[135:136], off offset:-1             // 000000002558: ee04007c 000000bb ffffff87
	global_load_u8 v188, v[137:138], off offset:-1             // 000000002564: ee04007c 000000bc ffffff89
	global_load_u8 v189, v[16:17], off                         // 000000002570: ee04007c 000000bd 00000010
	global_load_u8 v190, v[18:19], off                         // 00000000257c: ee04007c 000000be 00000012
	global_load_u8 v191, v[20:21], off                         // 000000002588: ee04007c 000000bf 00000014
	global_load_u8 v192, v[22:23], off                         // 000000002594: ee04007c 000000c0 00000016
	global_load_u8 v193, v[24:25], off                         // 0000000025a0: ee04007c 000000c1 00000018
	global_load_u8 v194, v[26:27], off                         // 0000000025ac: ee04007c 000000c2 0000001a
	global_load_u8 v195, v[28:29], off                         // 0000000025b8: ee04007c 000000c3 0000001c
	global_load_u8 v196, v[30:31], off                         // 0000000025c4: ee04007c 000000c4 0000001e
	global_load_u8 v197, v[34:35], off                         // 0000000025d0: ee04007c 000000c5 00000022
	global_load_u8 v198, v[36:37], off                         // 0000000025dc: ee04007c 000000c6 00000024
	global_load_u8 v199, v[38:39], off                         // 0000000025e8: ee04007c 000000c7 00000026
	global_load_u8 v200, v[129:130], off                       // 0000000025f4: ee04007c 000000c8 00000081
	global_load_u8 v201, v[131:132], off                       // 000000002600: ee04007c 000000c9 00000083
	global_load_u8 v202, v[133:134], off                       // 00000000260c: ee04007c 000000ca 00000085
	global_load_u8 v203, v[135:136], off                       // 000000002618: ee04007c 000000cb 00000087
	global_load_u8 v204, v[137:138], off                       // 000000002624: ee04007c 000000cc 00000089
	ds_load_2addr_b64 v[0:3], v55 offset1:2                    // 000000002630: d9dc0200 00000037
	ds_load_2addr_b64 v[4:7], v12 offset1:2                    // 000000002638: d9dc0200 0400000c
	ds_load_2addr_b64 v[8:11], v13 offset1:2                   // 000000002640: d9dc0200 0800000d
	ds_load_2addr_b64 v[14:17], v56 offset1:2                  // 000000002648: d9dc0200 0e000038
	ds_load_2addr_b64 v[153:156], v12 offset0:4 offset1:6      // 000000002650: d9dc0604 9900000c
	ds_load_2addr_b64 v[157:160], v55 offset0:4 offset1:6      // 000000002658: d9dc0604 9d000037
	ds_load_2addr_b64 v[161:164], v13 offset0:4 offset1:6      // 000000002660: d9dc0604 a100000d
	ds_load_2addr_b64 v[165:168], v56 offset0:4 offset1:6      // 000000002668: d9dc0604 a5000038
	s_wait_dscnt 0x6                                           // 000000002670: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[0:1], v[4:5], 0  // 000000002674: cc464081 1a020900
	s_wait_dscnt 0x5                                           // 00000000267c: bfc60005
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[0:1], v[8:9], 0  // 000000002680: cc464089 1a021100
	s_wait_dscnt 0x4                                           // 000000002688: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[14:15], v[4:5], 0// 00000000268c: cc464091 1a02090e
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[14:15], v[8:9], 0  // 000000002694: cc464020 1a02110e
	v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[2:3], v[6:7], v[129:136]// 00000000269c: cc464081 1e060d02
	v_wmma_f32_16x16x16_fp8_fp8 v[137:144], v[2:3], v[10:11], v[137:144]// 0000000026a4: cc464089 1e261502
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000026ac: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[145:152], v[16:17], v[6:7], v[145:152]// 0000000026b0: cc464091 1e460d10
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[10:11], v[32:39]// 0000000026b8: cc464020 1c821510
	s_wait_dscnt 0x2                                           // 0000000026c0: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[157:158], v[153:154], 0// 0000000026c4: cc464018 1a03339d
	s_wait_dscnt 0x1                                           // 0000000026cc: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[157:158], v[161:162], 0// 0000000026d0: cc464010 1a03439d
	s_wait_dscnt 0x0                                           // 0000000026d8: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[165:166], v[153:154], 0// 0000000026dc: cc464008 1a0333a5
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[165:166], v[161:162], 0// 0000000026e4: cc464000 1a0343a5
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[159:160], v[155:156], v[24:31]// 0000000026ec: cc464018 1c63379f
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[159:160], v[163:164], v[16:23]// 0000000026f4: cc464010 1c43479f
	s_delay_alu instid0(valu_dep_4) | instskip(next) | instid1(valu_dep_4)// 0000000026fc: bf870214
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[167:168], v[155:156], v[8:15]// 000000002700: cc464008 1c2337a7
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[167:168], v[163:164], v[0:7]// 000000002708: cc464000 1c0347a7
	s_wait_loadcnt 0x23                                        // 000000002710: bfc00023
	v_add_nc_u32_e32 v153, 0xffffff02, v169                    // 000000002714: 4b3352ff ffffff02
	s_wait_loadcnt 0x22                                        // 00000000271c: bfc00022
	v_add_nc_u32_e32 v154, 0xffffff02, v170                    // 000000002720: 4b3554ff ffffff02
	s_wait_loadcnt 0x21                                        // 000000002728: bfc00021
	v_cmp_eq_u32_e64 s2, 0xff, v171                            // 00000000272c: d44a0002 020356ff 000000ff
	v_cmp_eq_u32_e32 vcc_lo, 0xff, v169                        // 000000002738: 7c9552ff 000000ff
	s_wait_loadcnt 0x20                                        // 000000002740: bfc00020
	v_cmp_eq_u32_e64 s3, 0xff, v172                            // 000000002744: d44a0003 020358ff 000000ff
	s_wait_loadcnt 0x1f                                        // 000000002750: bfc0001f
	v_cmp_eq_u32_e64 s4, 0xff, v173                            // 000000002754: d44a0004 02035aff 000000ff
	s_wait_loadcnt 0x1e                                        // 000000002760: bfc0001e
	v_cmp_eq_u32_e64 s5, 0xff, v174                            // 000000002764: d44a0005 02035cff 000000ff
	s_wait_loadcnt 0x1d                                        // 000000002770: bfc0001d
	v_add_nc_u32_e32 v155, 0xffffff02, v175                    // 000000002774: 4b375eff ffffff02
	s_wait_loadcnt 0x1c                                        // 00000000277c: bfc0001c
	v_cmp_eq_u32_e64 s6, 0xff, v176                            // 000000002780: d44a0006 020360ff 000000ff
	s_wait_loadcnt 0x1b                                        // 00000000278c: bfc0001b
	v_add_nc_u32_e32 v156, 0xffffff02, v177                    // 000000002790: 4b3962ff ffffff02
	s_wait_loadcnt 0x1a                                        // 000000002798: bfc0001a
	v_cmp_eq_u32_e64 s7, 0xff, v178                            // 00000000279c: d44a0007 020364ff 000000ff
	s_wait_loadcnt 0x19                                        // 0000000027a8: bfc00019
	v_cmp_eq_u32_e64 s8, 0xff, v179                            // 0000000027ac: d44a0008 020366ff 000000ff
	s_wait_loadcnt 0x18                                        // 0000000027b8: bfc00018
	v_cmp_eq_u32_e64 s9, 0xff, v180                            // 0000000027bc: d44a0009 020368ff 000000ff
	v_cmp_eq_u32_e64 s10, 0xff, v170                           // 0000000027c8: d44a000a 020354ff 000000ff
	s_wait_loadcnt 0x17                                        // 0000000027d4: bfc00017
	v_cmp_eq_u32_e64 s11, 0xff, v181                           // 0000000027d8: d44a000b 02036aff 000000ff
	s_wait_loadcnt 0x16                                        // 0000000027e4: bfc00016
	v_cmp_eq_u32_e64 s12, 0xff, v182                           // 0000000027e8: d44a000c 02036cff 000000ff
	s_wait_loadcnt 0x15                                        // 0000000027f4: bfc00015
	v_cmp_eq_u32_e64 s13, 0xff, v183                           // 0000000027f8: d44a000d 02036eff 000000ff
	s_wait_loadcnt 0x14                                        // 000000002804: bfc00014
	v_cmp_eq_u32_e64 s14, 0xff, v184                           // 000000002808: d44a000e 020370ff 000000ff
	s_wait_loadcnt 0x13                                        // 000000002814: bfc00013
	v_cmp_eq_u32_e64 s15, 0xff, v185                           // 000000002818: d44a000f 020372ff 000000ff
	s_wait_loadcnt 0x12                                        // 000000002824: bfc00012
	v_cmp_eq_u32_e64 s16, 0xff, v186                           // 000000002828: d44a0010 020374ff 000000ff
	v_cmp_eq_u32_e64 s20, 0xff, v175                           // 000000002834: d44a0014 02035eff 000000ff
	v_cmp_eq_u32_e64 s28, 0xff, v177                           // 000000002840: d44a001c 020362ff 000000ff
	v_add_nc_u32_e32 v157, v153, v171                          // 00000000284c: 4b3b5799
	v_add_nc_u32_e32 v158, v153, v172                          // 000000002850: 4b3d5999
	v_add_nc_u32_e32 v159, v153, v173                          // 000000002854: 4b3f5b99
	v_add_nc_u32_e32 v160, v153, v174                          // 000000002858: 4b415d99
	v_add_nc_u32_e32 v161, v153, v176                          // 00000000285c: 4b436199
	v_add_nc_u32_e32 v162, v153, v178                          // 000000002860: 4b456599
	v_add_nc_u32_e32 v163, v153, v179                          // 000000002864: 4b476799
	v_add_nc_u32_e32 v164, v153, v180                          // 000000002868: 4b496999
	v_add_nc_u32_e32 v165, v154, v171                          // 00000000286c: 4b4b579a
	v_add_nc_u32_e32 v166, v154, v172                          // 000000002870: 4b4d599a
	v_add_nc_u32_e32 v167, v154, v173                          // 000000002874: 4b4f5b9a
	v_add_nc_u32_e32 v168, v154, v174                          // 000000002878: 4b515d9a
	v_add_nc_u32_e32 v169, v154, v176                          // 00000000287c: 4b53619a
	v_add_nc_u32_e32 v170, v154, v178                          // 000000002880: 4b55659a
	v_add_nc_u32_e32 v171, v154, v179                          // 000000002884: 4b57679a
	v_add_nc_u32_e32 v172, v154, v180                          // 000000002888: 4b59699a
	v_add_nc_u32_e32 v173, v153, v181                          // 00000000288c: 4b5b6b99
	v_add_nc_u32_e32 v174, v153, v182                          // 000000002890: 4b5d6d99
	v_add_nc_u32_e32 v175, v153, v183                          // 000000002894: 4b5f6f99
	v_add_nc_u32_e32 v176, v153, v184                          // 000000002898: 4b617199
	v_add_nc_u32_e32 v177, v153, v185                          // 00000000289c: 4b637399
	v_add_nc_u32_e32 v178, v153, v186                          // 0000000028a0: 4b657599
	s_wait_loadcnt 0x11                                        // 0000000028a4: bfc00011
	v_add_nc_u32_e32 v179, v153, v187                          // 0000000028a8: 4b677799
	s_wait_loadcnt 0x10                                        // 0000000028ac: bfc00010
	v_add_nc_u32_e32 v153, v153, v188                          // 0000000028b0: 4b337999
	v_add_nc_u32_e32 v180, v154, v181                          // 0000000028b4: 4b696b9a
	v_add_nc_u32_e32 v181, v154, v182                          // 0000000028b8: 4b6b6d9a
	v_add_nc_u32_e32 v182, v154, v183                          // 0000000028bc: 4b6d6f9a
	v_add_nc_u32_e32 v183, v154, v184                          // 0000000028c0: 4b6f719a
	v_add_nc_u32_e32 v184, v154, v185                          // 0000000028c4: 4b71739a
	v_add_nc_u32_e32 v185, v154, v186                          // 0000000028c8: 4b73759a
	v_add_nc_u32_e32 v186, v154, v187                          // 0000000028cc: 4b75779a
	v_add_nc_u32_e32 v154, v154, v188                          // 0000000028d0: 4b35799a
	v_cmp_eq_u32_e64 s17, 0xff, v187                           // 0000000028d4: d44a0011 020376ff 000000ff
	v_cmp_eq_u32_e64 s18, 0xff, v188                           // 0000000028e0: d44a0012 020378ff 000000ff
	s_wait_loadcnt 0xf                                         // 0000000028ec: bfc0000f
	v_cmp_eq_u32_e64 s19, 0xff, v189                           // 0000000028f0: d44a0013 02037aff 000000ff
	s_wait_loadcnt 0xe                                         // 0000000028fc: bfc0000e
	v_cmp_eq_u32_e64 s21, 0xff, v190                           // 000000002900: d44a0015 02037cff 000000ff
	s_wait_loadcnt 0xd                                         // 00000000290c: bfc0000d
	v_cmp_eq_u32_e64 s22, 0xff, v191                           // 000000002910: d44a0016 02037eff 000000ff
	s_wait_loadcnt 0xc                                         // 00000000291c: bfc0000c
	v_cmp_eq_u32_e64 s23, 0xff, v192                           // 000000002920: d44a0017 020380ff 000000ff
	s_wait_loadcnt 0xb                                         // 00000000292c: bfc0000b
	v_cmp_eq_u32_e64 s24, 0xff, v193                           // 000000002930: d44a0018 020382ff 000000ff
	s_wait_loadcnt 0xa                                         // 00000000293c: bfc0000a
	v_cmp_eq_u32_e64 s25, 0xff, v194                           // 000000002940: d44a0019 020384ff 000000ff
	s_wait_loadcnt 0x9                                         // 00000000294c: bfc00009
	v_cmp_eq_u32_e64 s26, 0xff, v195                           // 000000002950: d44a001a 020386ff 000000ff
	s_wait_loadcnt 0x8                                         // 00000000295c: bfc00008
	v_cmp_eq_u32_e64 s27, 0xff, v196                           // 000000002960: d44a001b 020388ff 000000ff
	s_wait_loadcnt 0x7                                         // 00000000296c: bfc00007
	v_cmp_eq_u32_e64 s29, 0xff, v197                           // 000000002970: d44a001d 02038aff 000000ff
	s_wait_loadcnt 0x6                                         // 00000000297c: bfc00006
	v_cmp_eq_u32_e64 s30, 0xff, v198                           // 000000002980: d44a001e 02038cff 000000ff
	s_wait_loadcnt 0x5                                         // 00000000298c: bfc00005
	v_cmp_eq_u32_e64 s31, 0xff, v199                           // 000000002990: d44a001f 02038eff 000000ff
	s_wait_loadcnt 0x4                                         // 00000000299c: bfc00004
	v_cmp_eq_u32_e64 s33, 0xff, v200                           // 0000000029a0: d44a0021 020390ff 000000ff
	s_wait_loadcnt 0x3                                         // 0000000029ac: bfc00003
	v_cmp_eq_u32_e64 s34, 0xff, v201                           // 0000000029b0: d44a0022 020392ff 000000ff
	s_wait_loadcnt 0x2                                         // 0000000029bc: bfc00002
	v_cmp_eq_u32_e64 s35, 0xff, v202                           // 0000000029c0: d44a0023 020394ff 000000ff
	s_wait_loadcnt 0x1                                         // 0000000029cc: bfc00001
	v_cmp_eq_u32_e64 s36, 0xff, v203                           // 0000000029d0: d44a0024 020396ff 000000ff
	v_add_nc_u32_e32 v187, v155, v189                          // 0000000029dc: 4b777b9b
	v_add_nc_u32_e32 v188, v155, v190                          // 0000000029e0: 4b797d9b
	v_add_nc_u32_e32 v205, v155, v191                          // 0000000029e4: 4b9b7f9b
	v_add_nc_u32_e32 v206, v155, v192                          // 0000000029e8: 4b9d819b
	v_add_nc_u32_e32 v207, v155, v193                          // 0000000029ec: 4b9f839b
	v_add_nc_u32_e32 v208, v155, v194                          // 0000000029f0: 4ba1859b
	v_add_nc_u32_e32 v209, v155, v195                          // 0000000029f4: 4ba3879b
	v_add_nc_u32_e32 v210, v155, v196                          // 0000000029f8: 4ba5899b
	v_add_nc_u32_e32 v189, v156, v189                          // 0000000029fc: 4b7b7b9c
	v_add_nc_u32_e32 v190, v156, v190                          // 000000002a00: 4b7d7d9c
	v_add_nc_u32_e32 v191, v156, v191                          // 000000002a04: 4b7f7f9c
	v_add_nc_u32_e32 v192, v156, v192                          // 000000002a08: 4b81819c
	v_add_nc_u32_e32 v193, v156, v193                          // 000000002a0c: 4b83839c
	v_add_nc_u32_e32 v194, v156, v194                          // 000000002a10: 4b85859c
	v_add_nc_u32_e32 v195, v156, v195                          // 000000002a14: 4b87879c
	v_add_nc_u32_e32 v196, v156, v196                          // 000000002a18: 4b89899c
	v_add_nc_u32_e32 v211, v155, v197                          // 000000002a1c: 4ba78b9b
	v_add_nc_u32_e32 v212, v155, v198                          // 000000002a20: 4ba98d9b
	v_add_nc_u32_e32 v213, v155, v199                          // 000000002a24: 4bab8f9b
	v_add_nc_u32_e32 v214, v155, v200                          // 000000002a28: 4bad919b
	v_add_nc_u32_e32 v215, v155, v201                          // 000000002a2c: 4baf939b
	v_add_nc_u32_e32 v216, v155, v202                          // 000000002a30: 4bb1959b
	v_add_nc_u32_e32 v217, v155, v203                          // 000000002a34: 4bb3979b
	s_wait_loadcnt 0x0                                         // 000000002a38: bfc00000
	v_add_nc_u32_e32 v155, v155, v204                          // 000000002a3c: 4b37999b
	v_add_nc_u32_e32 v197, v156, v197                          // 000000002a40: 4b8b8b9c
	v_add_nc_u32_e32 v198, v156, v198                          // 000000002a44: 4b8d8d9c
	v_add_nc_u32_e32 v199, v156, v199                          // 000000002a48: 4b8f8f9c
	v_add_nc_u32_e32 v200, v156, v200                          // 000000002a4c: 4b91919c
	v_add_nc_u32_e32 v201, v156, v201                          // 000000002a50: 4b93939c
	v_add_nc_u32_e32 v202, v156, v202                          // 000000002a54: 4b95959c
	v_add_nc_u32_e32 v203, v156, v203                          // 000000002a58: 4b97979c
	v_add_nc_u32_e32 v156, v156, v204                          // 000000002a5c: 4b39999c
	v_ldexp_f32 v129, v129, v157                               // 000000002a60: d71c0081 02033b81
	v_ldexp_f32 v130, v130, v158                               // 000000002a68: d71c0082 02033d82
	v_ldexp_f32 v131, v131, v159                               // 000000002a70: d71c0083 02033f83
	v_ldexp_f32 v132, v132, v160                               // 000000002a78: d71c0084 02034184
	v_ldexp_f32 v133, v133, v161                               // 000000002a80: d71c0085 02034385
	v_ldexp_f32 v134, v134, v162                               // 000000002a88: d71c0086 02034586
	v_ldexp_f32 v135, v135, v163                               // 000000002a90: d71c0087 02034787
	v_ldexp_f32 v136, v136, v164                               // 000000002a98: d71c0088 02034988
	v_ldexp_f32 v137, v137, v165                               // 000000002aa0: d71c0089 02034b89
	v_ldexp_f32 v138, v138, v166                               // 000000002aa8: d71c008a 02034d8a
	v_ldexp_f32 v139, v139, v167                               // 000000002ab0: d71c008b 02034f8b
	v_ldexp_f32 v140, v140, v168                               // 000000002ab8: d71c008c 0203518c
	v_ldexp_f32 v141, v141, v169                               // 000000002ac0: d71c008d 0203538d
	v_ldexp_f32 v142, v142, v170                               // 000000002ac8: d71c008e 0203558e
	v_ldexp_f32 v143, v143, v171                               // 000000002ad0: d71c008f 0203578f
	v_ldexp_f32 v144, v144, v172                               // 000000002ad8: d71c0090 02035990
	v_ldexp_f32 v145, v145, v173                               // 000000002ae0: d71c0091 02035b91
	v_ldexp_f32 v146, v146, v174                               // 000000002ae8: d71c0092 02035d92
	v_ldexp_f32 v147, v147, v175                               // 000000002af0: d71c0093 02035f93
	v_ldexp_f32 v148, v148, v176                               // 000000002af8: d71c0094 02036194
	v_ldexp_f32 v149, v149, v177                               // 000000002b00: d71c0095 02036395
	v_ldexp_f32 v150, v150, v178                               // 000000002b08: d71c0096 02036596
	v_ldexp_f32 v151, v151, v179                               // 000000002b10: d71c0097 02036797
	v_ldexp_f32 v152, v152, v153                               // 000000002b18: d71c0098 02033398
	v_ldexp_f32 v32, v32, v180                                 // 000000002b20: d71c0020 02036920
	v_ldexp_f32 v33, v33, v181                                 // 000000002b28: d71c0021 02036b21
	v_ldexp_f32 v34, v34, v182                                 // 000000002b30: d71c0022 02036d22
	v_ldexp_f32 v35, v35, v183                                 // 000000002b38: d71c0023 02036f23
	v_ldexp_f32 v36, v36, v184                                 // 000000002b40: d71c0024 02037124
	v_ldexp_f32 v37, v37, v185                                 // 000000002b48: d71c0025 02037325
	v_ldexp_f32 v38, v38, v186                                 // 000000002b50: d71c0026 02037526
	v_ldexp_f32 v39, v39, v154                                 // 000000002b58: d71c0027 02033527
	v_cmp_eq_u32_e64 s37, 0xff, v204                           // 000000002b60: d44a0025 020398ff 000000ff
	s_or_b32 s38, s2, vcc_lo                                   // 000000002b6c: 8c266a02
	s_or_b32 s39, vcc_lo, s3                                   // 000000002b70: 8c27036a
	s_or_b32 s40, vcc_lo, s4                                   // 000000002b74: 8c28046a
	s_or_b32 s41, vcc_lo, s5                                   // 000000002b78: 8c29056a
	s_or_b32 s52, vcc_lo, s6                                   // 000000002b7c: 8c34066a
	s_or_b32 s53, vcc_lo, s7                                   // 000000002b80: 8c35076a
	s_or_b32 s54, vcc_lo, s8                                   // 000000002b84: 8c36086a
	s_or_b32 s55, vcc_lo, s9                                   // 000000002b88: 8c37096a
	s_or_b32 s2, s2, s10                                       // 000000002b8c: 8c020a02
	s_or_b32 s3, s3, s10                                       // 000000002b90: 8c030a03
	s_or_b32 s4, s4, s10                                       // 000000002b94: 8c040a04
	s_or_b32 s5, s5, s10                                       // 000000002b98: 8c050a05
	s_or_b32 s6, s6, s10                                       // 000000002b9c: 8c060a06
	s_or_b32 s7, s7, s10                                       // 000000002ba0: 8c070a07
	s_or_b32 s8, s8, s10                                       // 000000002ba4: 8c080a08
	s_or_b32 s9, s9, s10                                       // 000000002ba8: 8c090a09
	s_or_b32 s56, vcc_lo, s11                                  // 000000002bac: 8c380b6a
	s_or_b32 s57, vcc_lo, s12                                  // 000000002bb0: 8c390c6a
	s_or_b32 s58, vcc_lo, s13                                  // 000000002bb4: 8c3a0d6a
	s_or_b32 s59, vcc_lo, s14                                  // 000000002bb8: 8c3b0e6a
	s_or_b32 s60, vcc_lo, s15                                  // 000000002bbc: 8c3c0f6a
	s_or_b32 s61, vcc_lo, s16                                  // 000000002bc0: 8c3d106a
	s_or_b32 s62, vcc_lo, s17                                  // 000000002bc4: 8c3e116a
	s_or_b32 s63, vcc_lo, s18                                  // 000000002bc8: 8c3f126a
	s_or_b32 s11, s10, s11                                     // 000000002bcc: 8c0b0b0a
	s_or_b32 s12, s10, s12                                     // 000000002bd0: 8c0c0c0a
	s_or_b32 s13, s10, s13                                     // 000000002bd4: 8c0d0d0a
	s_or_b32 s14, s10, s14                                     // 000000002bd8: 8c0e0e0a
	s_or_b32 s15, s10, s15                                     // 000000002bdc: 8c0f0f0a
	s_or_b32 s16, s10, s16                                     // 000000002be0: 8c10100a
	s_or_b32 s17, s10, s17                                     // 000000002be4: 8c11110a
	s_or_b32 s10, s10, s18                                     // 000000002be8: 8c0a120a
	v_ldexp_f32 v24, v24, v187                                 // 000000002bec: d71c0018 02037718
	v_ldexp_f32 v25, v25, v188                                 // 000000002bf4: d71c0019 02037919
	v_ldexp_f32 v26, v26, v205                                 // 000000002bfc: d71c001a 02039b1a
	v_ldexp_f32 v27, v27, v206                                 // 000000002c04: d71c001b 02039d1b
	v_ldexp_f32 v28, v28, v207                                 // 000000002c0c: d71c001c 02039f1c
	v_ldexp_f32 v29, v29, v208                                 // 000000002c14: d71c001d 0203a11d
	v_ldexp_f32 v30, v30, v209                                 // 000000002c1c: d71c001e 0203a31e
	v_ldexp_f32 v31, v31, v210                                 // 000000002c24: d71c001f 0203a51f
	v_ldexp_f32 v16, v16, v189                                 // 000000002c2c: d71c0010 02037b10
	v_ldexp_f32 v17, v17, v190                                 // 000000002c34: d71c0011 02037d11
	v_ldexp_f32 v18, v18, v191                                 // 000000002c3c: d71c0012 02037f12
	v_ldexp_f32 v19, v19, v192                                 // 000000002c44: d71c0013 02038113
	v_ldexp_f32 v20, v20, v193                                 // 000000002c4c: d71c0014 02038314
	v_ldexp_f32 v21, v21, v194                                 // 000000002c54: d71c0015 02038515
	v_ldexp_f32 v22, v22, v195                                 // 000000002c5c: d71c0016 02038716
	v_ldexp_f32 v23, v23, v196                                 // 000000002c64: d71c0017 02038917
	v_ldexp_f32 v8, v8, v211                                   // 000000002c6c: d71c0008 0203a708
	v_ldexp_f32 v9, v9, v212                                   // 000000002c74: d71c0009 0203a909
	v_ldexp_f32 v10, v10, v213                                 // 000000002c7c: d71c000a 0203ab0a
	v_ldexp_f32 v11, v11, v214                                 // 000000002c84: d71c000b 0203ad0b
	v_ldexp_f32 v12, v12, v215                                 // 000000002c8c: d71c000c 0203af0c
	v_ldexp_f32 v13, v13, v216                                 // 000000002c94: d71c000d 0203b10d
	v_ldexp_f32 v14, v14, v217                                 // 000000002c9c: d71c000e 0203b30e
	v_ldexp_f32 v15, v15, v155                                 // 000000002ca4: d71c000f 0203370f
	v_ldexp_f32 v0, v0, v197                                   // 000000002cac: d71c0000 02038b00
	v_ldexp_f32 v1, v1, v198                                   // 000000002cb4: d71c0001 02038d01
	v_ldexp_f32 v2, v2, v199                                   // 000000002cbc: d71c0002 02038f02
	v_ldexp_f32 v3, v3, v200                                   // 000000002cc4: d71c0003 02039103
	v_ldexp_f32 v4, v4, v201                                   // 000000002ccc: d71c0004 02039304
	v_ldexp_f32 v5, v5, v202                                   // 000000002cd4: d71c0005 02039505
	v_ldexp_f32 v6, v6, v203                                   // 000000002cdc: d71c0006 02039706
	v_ldexp_f32 v7, v7, v156                                   // 000000002ce4: d71c0007 02033907
	s_wait_alu depctr_sa_sdst(0)                               // 000000002cec: bf88ff9e
	v_cndmask_b32_e64 v129, v129, 0x7fc00000, s38              // 000000002cf0: d5010081 0099ff81 7fc00000
	v_cndmask_b32_e64 v130, v130, 0x7fc00000, s39              // 000000002cfc: d5010082 009dff82 7fc00000
	v_cndmask_b32_e64 v131, v131, 0x7fc00000, s40              // 000000002d08: d5010083 00a1ff83 7fc00000
	v_cndmask_b32_e64 v132, v132, 0x7fc00000, s41              // 000000002d14: d5010084 00a5ff84 7fc00000
	v_cndmask_b32_e64 v133, v133, 0x7fc00000, s52              // 000000002d20: d5010085 00d1ff85 7fc00000
	v_cndmask_b32_e64 v134, v134, 0x7fc00000, s53              // 000000002d2c: d5010086 00d5ff86 7fc00000
	v_cndmask_b32_e64 v135, v135, 0x7fc00000, s54              // 000000002d38: d5010087 00d9ff87 7fc00000
	v_cndmask_b32_e64 v136, v136, 0x7fc00000, s55              // 000000002d44: d5010088 00ddff88 7fc00000
	v_cndmask_b32_e64 v137, v137, 0x7fc00000, s2               // 000000002d50: d5010089 0009ff89 7fc00000
	v_cndmask_b32_e64 v138, v138, 0x7fc00000, s3               // 000000002d5c: d501008a 000dff8a 7fc00000
	v_cndmask_b32_e64 v139, v139, 0x7fc00000, s4               // 000000002d68: d501008b 0011ff8b 7fc00000
	v_cndmask_b32_e64 v140, v140, 0x7fc00000, s5               // 000000002d74: d501008c 0015ff8c 7fc00000
	v_cndmask_b32_e64 v141, v141, 0x7fc00000, s6               // 000000002d80: d501008d 0019ff8d 7fc00000
	v_cndmask_b32_e64 v142, v142, 0x7fc00000, s7               // 000000002d8c: d501008e 001dff8e 7fc00000
	v_cndmask_b32_e64 v143, v143, 0x7fc00000, s8               // 000000002d98: d501008f 0021ff8f 7fc00000
	v_cndmask_b32_e64 v144, v144, 0x7fc00000, s9               // 000000002da4: d5010090 0025ff90 7fc00000
	v_cndmask_b32_e64 v145, v145, 0x7fc00000, s56              // 000000002db0: d5010091 00e1ff91 7fc00000
	v_cndmask_b32_e64 v146, v146, 0x7fc00000, s57              // 000000002dbc: d5010092 00e5ff92 7fc00000
	v_cndmask_b32_e64 v147, v147, 0x7fc00000, s58              // 000000002dc8: d5010093 00e9ff93 7fc00000
	v_cndmask_b32_e64 v148, v148, 0x7fc00000, s59              // 000000002dd4: d5010094 00edff94 7fc00000
	v_cndmask_b32_e64 v149, v149, 0x7fc00000, s60              // 000000002de0: d5010095 00f1ff95 7fc00000
	v_cndmask_b32_e64 v150, v150, 0x7fc00000, s61              // 000000002dec: d5010096 00f5ff96 7fc00000
	v_cndmask_b32_e64 v151, v151, 0x7fc00000, s62              // 000000002df8: d5010097 00f9ff97 7fc00000
	v_cndmask_b32_e64 v152, v152, 0x7fc00000, s63              // 000000002e04: d5010098 00fdff98 7fc00000
	v_cndmask_b32_e64 v32, v32, 0x7fc00000, s11                // 000000002e10: d5010020 002dff20 7fc00000
	v_cndmask_b32_e64 v33, v33, 0x7fc00000, s12                // 000000002e1c: d5010021 0031ff21 7fc00000
	v_cndmask_b32_e64 v34, v34, 0x7fc00000, s13                // 000000002e28: d5010022 0035ff22 7fc00000
	v_cndmask_b32_e64 v35, v35, 0x7fc00000, s14                // 000000002e34: d5010023 0039ff23 7fc00000
	v_cndmask_b32_e64 v36, v36, 0x7fc00000, s15                // 000000002e40: d5010024 003dff24 7fc00000
	v_cndmask_b32_e64 v37, v37, 0x7fc00000, s16                // 000000002e4c: d5010025 0041ff25 7fc00000
	v_cndmask_b32_e64 v38, v38, 0x7fc00000, s17                // 000000002e58: d5010026 0045ff26 7fc00000
	v_cndmask_b32_e64 v39, v39, 0x7fc00000, s10                // 000000002e64: d5010027 0029ff27 7fc00000
	s_or_b32 s18, s20, s21                                     // 000000002e70: 8c121514
	s_or_b32 s64, s20, s22                                     // 000000002e74: 8c401614
	s_or_b32 s65, s20, s23                                     // 000000002e78: 8c411714
	s_or_b32 s66, s20, s24                                     // 000000002e7c: 8c421814
	s_or_b32 s67, s20, s25                                     // 000000002e80: 8c431914
	s_or_b32 s68, s20, s26                                     // 000000002e84: 8c441a14
	s_or_b32 s69, s20, s27                                     // 000000002e88: 8c451b14
	s_or_b32 s70, s19, s28                                     // 000000002e8c: 8c461c13
	s_or_b32 s21, s21, s28                                     // 000000002e90: 8c151c15
	s_or_b32 s22, s22, s28                                     // 000000002e94: 8c161c16
	s_or_b32 s23, s23, s28                                     // 000000002e98: 8c171c17
	s_or_b32 s24, s24, s28                                     // 000000002e9c: 8c181c18
	s_or_b32 s25, s25, s28                                     // 000000002ea0: 8c191c19
	s_or_b32 s26, s26, s28                                     // 000000002ea4: 8c1a1c1a
	s_or_b32 s27, s27, s28                                     // 000000002ea8: 8c1b1c1b
	s_or_b32 s71, s20, s29                                     // 000000002eac: 8c471d14
	s_or_b32 s72, s20, s30                                     // 000000002eb0: 8c481e14
	s_or_b32 s73, s20, s31                                     // 000000002eb4: 8c491f14
	s_or_b32 s74, s20, s33                                     // 000000002eb8: 8c4a2114
	s_or_b32 s75, s20, s34                                     // 000000002ebc: 8c4b2214
	s_or_b32 s76, s20, s35                                     // 000000002ec0: 8c4c2314
	s_or_b32 s77, s20, s36                                     // 000000002ec4: 8c4d2414
	s_or_b32 s78, s20, s37                                     // 000000002ec8: 8c4e2514
	s_or_b32 s19, s19, s20                                     // 000000002ecc: 8c131413
	s_or_b32 s20, s28, s29                                     // 000000002ed0: 8c141d1c
	s_or_b32 s29, s28, s30                                     // 000000002ed4: 8c1d1e1c
	s_or_b32 s30, s28, s31                                     // 000000002ed8: 8c1e1f1c
	s_or_b32 s31, s28, s33                                     // 000000002edc: 8c1f211c
	s_or_b32 s33, s28, s34                                     // 000000002ee0: 8c21221c
	s_or_b32 s34, s28, s35                                     // 000000002ee4: 8c22231c
	s_or_b32 s35, s28, s36                                     // 000000002ee8: 8c23241c
	s_or_b32 s28, s28, s37                                     // 000000002eec: 8c1c251c
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ef0: bf88ff9e
	v_cndmask_b32_e64 v25, v25, 0x7fc00000, s18                // 000000002ef4: d5010019 0049ff19 7fc00000
	v_cndmask_b32_e64 v26, v26, 0x7fc00000, s64                // 000000002f00: d501001a 0101ff1a 7fc00000
	v_cndmask_b32_e64 v27, v27, 0x7fc00000, s65                // 000000002f0c: d501001b 0105ff1b 7fc00000
	v_cndmask_b32_e64 v28, v28, 0x7fc00000, s66                // 000000002f18: d501001c 0109ff1c 7fc00000
	v_cndmask_b32_e64 v29, v29, 0x7fc00000, s67                // 000000002f24: d501001d 010dff1d 7fc00000
	v_cndmask_b32_e64 v30, v30, 0x7fc00000, s68                // 000000002f30: d501001e 0111ff1e 7fc00000
	v_cndmask_b32_e64 v31, v31, 0x7fc00000, s69                // 000000002f3c: d501001f 0115ff1f 7fc00000
	v_cndmask_b32_e64 v16, v16, 0x7fc00000, s70                // 000000002f48: d5010010 0119ff10 7fc00000
	v_cndmask_b32_e64 v17, v17, 0x7fc00000, s21                // 000000002f54: d5010011 0055ff11 7fc00000
	v_cndmask_b32_e64 v18, v18, 0x7fc00000, s22                // 000000002f60: d5010012 0059ff12 7fc00000
	v_cndmask_b32_e64 v19, v19, 0x7fc00000, s23                // 000000002f6c: d5010013 005dff13 7fc00000
	v_cndmask_b32_e64 v20, v20, 0x7fc00000, s24                // 000000002f78: d5010014 0061ff14 7fc00000
	v_cndmask_b32_e64 v21, v21, 0x7fc00000, s25                // 000000002f84: d5010015 0065ff15 7fc00000
	v_cndmask_b32_e64 v22, v22, 0x7fc00000, s26                // 000000002f90: d5010016 0069ff16 7fc00000
	v_cndmask_b32_e64 v23, v23, 0x7fc00000, s27                // 000000002f9c: d5010017 006dff17 7fc00000
	v_cndmask_b32_e64 v8, v8, 0x7fc00000, s71                  // 000000002fa8: d5010008 011dff08 7fc00000
	v_cndmask_b32_e64 v9, v9, 0x7fc00000, s72                  // 000000002fb4: d5010009 0121ff09 7fc00000
	v_cndmask_b32_e64 v10, v10, 0x7fc00000, s73                // 000000002fc0: d501000a 0125ff0a 7fc00000
	v_cndmask_b32_e64 v11, v11, 0x7fc00000, s74                // 000000002fcc: d501000b 0129ff0b 7fc00000
	v_cndmask_b32_e64 v12, v12, 0x7fc00000, s75                // 000000002fd8: d501000c 012dff0c 7fc00000
	v_cndmask_b32_e64 v13, v13, 0x7fc00000, s76                // 000000002fe4: d501000d 0131ff0d 7fc00000
	v_cndmask_b32_e64 v14, v14, 0x7fc00000, s77                // 000000002ff0: d501000e 0135ff0e 7fc00000
	v_cndmask_b32_e64 v15, v15, 0x7fc00000, s78                // 000000002ffc: d501000f 0139ff0f 7fc00000
	v_cndmask_b32_e64 v24, v24, 0x7fc00000, s19                // 000000003008: d5010018 004dff18 7fc00000
	v_cndmask_b32_e64 v0, v0, 0x7fc00000, s20                  // 000000003014: d5010000 0051ff00 7fc00000
	v_cndmask_b32_e64 v1, v1, 0x7fc00000, s29                  // 000000003020: d5010001 0075ff01 7fc00000
	v_cndmask_b32_e64 v2, v2, 0x7fc00000, s30                  // 00000000302c: d5010002 0079ff02 7fc00000
	v_cndmask_b32_e64 v3, v3, 0x7fc00000, s31                  // 000000003038: d5010003 007dff03 7fc00000
	v_cndmask_b32_e64 v4, v4, 0x7fc00000, s33                  // 000000003044: d5010004 0085ff04 7fc00000
	v_cndmask_b32_e64 v5, v5, 0x7fc00000, s34                  // 000000003050: d5010005 0089ff05 7fc00000
	v_cndmask_b32_e64 v6, v6, 0x7fc00000, s35                  // 00000000305c: d5010006 008dff06 7fc00000
	v_cndmask_b32_e64 v7, v7, 0x7fc00000, s28                  // 000000003068: d5010007 0071ff07 7fc00000
	v_dual_add_f32 v45, v45, v129 :: v_dual_add_f32 v124, v124, v130// 000000003074: c909032d 2d7d057c
	v_dual_add_f32 v119, v119, v131 :: v_dual_add_f32 v114, v114, v132// 00000000307c: c9090777 77730972
	v_add_f32_e32 v109, v109, v133                             // 000000003084: 06db0b6d
	v_add_f32_e32 v105, v105, v134                             // 000000003088: 06d30d69
	v_dual_add_f32 v101, v101, v135 :: v_dual_add_f32 v94, v94, v136// 00000000308c: c9090f65 655f115e
	v_dual_add_f32 v77, v77, v137 :: v_dual_add_f32 v74, v74, v138// 000000003094: c909134d 4d4b154a
	v_dual_add_f32 v73, v73, v139 :: v_dual_add_f32 v72, v72, v140// 00000000309c: c9091749 49491948
	v_dual_add_f32 v71, v71, v141 :: v_dual_add_f32 v70, v70, v142// 0000000030a4: c9091b47 47471d46
	v_add_f32_e32 v69, v69, v143                               // 0000000030ac: 068b1f45
	v_dual_add_f32 v67, v67, v144 :: v_dual_add_f32 v92, v92, v145// 0000000030b0: c9092143 435d235c
	v_dual_add_f32 v84, v84, v146 :: v_dual_add_f32 v83, v83, v147// 0000000030b8: c9092554 54532753
	v_dual_add_f32 v82, v82, v148 :: v_dual_add_f32 v81, v81, v149// 0000000030c0: c9092952 52512b51
	v_dual_add_f32 v80, v80, v150 :: v_dual_add_f32 v79, v79, v151// 0000000030c8: c9092d50 504f2f4f
	v_add_f32_e32 v78, v78, v152                               // 0000000030d0: 069d314e
	v_dual_add_f32 v32, v68, v32 :: v_dual_add_f32 v33, v66, v33// 0000000030d4: c9084144 20204342
	v_dual_add_f32 v34, v64, v34 :: v_dual_add_f32 v35, v62, v35// 0000000030dc: c9084540 2222473e
	v_dual_add_f32 v36, v61, v36 :: v_dual_add_f32 v37, v59, v37// 0000000030e4: c908493d 24244b3b
	v_dual_add_f32 v38, v58, v38 :: v_dual_add_f32 v39, v44, v39// 0000000030ec: c9084d3a 26264f2c
	v_dual_add_f32 v124, v124, v25 :: v_dual_add_f32 v119, v119, v26// 0000000030f4: c908337c 7c763577
	v_dual_add_f32 v114, v114, v27 :: v_dual_add_f32 v109, v109, v28// 0000000030fc: c9083772 726c396d
	v_add_f32_e32 v105, v105, v29                              // 000000003104: 06d23b69
	v_dual_add_f32 v101, v101, v30 :: v_dual_add_f32 v94, v94, v31// 000000003108: c9083d65 655e3f5e
	v_dual_add_f32 v77, v77, v16 :: v_dual_add_f32 v74, v74, v17// 000000003110: c908214d 4d4a234a
	v_dual_add_f32 v73, v73, v18 :: v_dual_add_f32 v72, v72, v19// 000000003118: c9082549 49482748
	v_dual_add_f32 v71, v71, v20 :: v_dual_add_f32 v70, v70, v21// 000000003120: c9082947 47462b46
	v_add_f32_e32 v69, v69, v22                                // 000000003128: 068a2d45
	v_dual_add_f32 v67, v67, v23 :: v_dual_add_f32 v92, v92, v8// 00000000312c: c9082f43 435c115c
	v_dual_add_f32 v84, v84, v9 :: v_dual_add_f32 v83, v83, v10// 000000003134: c9081354 54521553
	v_dual_add_f32 v82, v82, v11 :: v_dual_add_f32 v81, v81, v12// 00000000313c: c9081752 52501951
	v_dual_add_f32 v80, v80, v13 :: v_dual_add_f32 v79, v79, v14// 000000003144: c9081b50 504e1d4f
	v_dual_add_f32 v78, v78, v15 :: v_dual_add_f32 v45, v45, v24// 00000000314c: c9081f4e 4e2c312d
	v_add_f32_e32 v68, v32, v0                                 // 000000003154: 06880120
	v_add_f32_e32 v66, v33, v1                                 // 000000003158: 06840321
	v_add_f32_e32 v64, v34, v2                                 // 00000000315c: 06800522
	v_dual_add_f32 v62, v35, v3 :: v_dual_add_f32 v61, v36, v4 // 000000003160: c9080723 3e3c0924
	v_dual_add_f32 v59, v37, v5 :: v_dual_add_f32 v58, v38, v6 // 000000003168: c9080b25 3b3a0d26
	v_add_f32_e32 v44, v39, v7                                 // 000000003170: 06580f27
	s_cmp_lg_u64 s[48:49], s[50:51]                            // 000000003174: bf113230
	s_cbranch_scc1 64577                                       // 000000003178: bfa2fc41 <tessera_rocm_scaled_matmul_lds_533dbb9ffad73571+0x780>
	s_load_b64 s[0:1], s[0:1], 0xa8                            // 00000000317c: f4002000 f80000a8
	v_mul_lo_u32 v2, s43, v40                                  // 000000003184: d72c0002 0202502b
	v_mul_lo_u32 v3, s42, v41                                  // 00000000318c: d72c0003 0202522a
	v_mad_co_u64_u32 v[0:1], null, s42, v40, 0                 // 000000003194: d6fe7c00 0202502a
	v_bfe_u32 v4, v45, 16, 1                                   // 00000000319c: d6100004 0205212d
	v_or_b32_e32 v5, 0x400000, v45                             // 0000000031a4: 380a5aff 00400000
	v_cmp_u_f32_e32 vcc_lo, v45, v45                           // 0000000031ac: 7c305b2d
	v_bfe_u32 v6, v124, 16, 1                                  // 0000000031b0: d6100006 0205217c
	v_or_b32_e32 v7, 0x400000, v124                            // 0000000031b8: 380ef8ff 00400000
	v_add3_u32 v4, v4, v45, 0x7fff                             // 0000000031c0: d6550004 03fe5b04 00007fff
	v_bfe_u32 v8, v119, 16, 1                                  // 0000000031cc: d6100008 02052177
	v_add3_u32 v1, v1, v3, v2                                  // 0000000031d4: d6550001 040a0701
	v_lshlrev_b64_e32 v[2:3], 1, v[46:47]                      // 0000000031dc: 3e045c81
	v_add3_u32 v6, v6, v124, 0x7fff                            // 0000000031e0: d6550006 03fef906 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000031ec: bf88ff9d
	v_cndmask_b32_e32 v9, v4, v5, vcc_lo                       // 0000000031f0: 02120b04
	v_add3_u32 v8, v8, v119, 0x7fff                            // 0000000031f4: d6550008 03feef08 00007fff
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]                        // 000000003200: 3e000081
	v_or_b32_e32 v11, 0x400000, v119                           // 000000003204: 3816eeff 00400000
	v_bfe_u32 v13, v114, 16, 1                                 // 00000000320c: d610000d 02052172
	v_or_b32_e32 v14, 0x400000, v114                           // 000000003214: 381ce4ff 00400000
	v_or_b32_e32 v17, 0x400000, v105                           // 00000000321c: 3822d2ff 00400000
	v_bfe_u32 v19, v101, 16, 1                                 // 000000003224: d6100013 02052165
	s_wait_kmcnt 0x0                                           // 00000000322c: bfc70000
	v_add_co_u32 v4, vcc_lo, s0, v0                            // 000000003230: d7006a04 02020000
	s_wait_alu depctr_va_vcc(0)                                // 000000003238: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, s1, v1, vcc_lo               // 00000000323c: d5207c05 01aa0201
	v_cmp_u_f32_e32 vcc_lo, v124, v124                         // 000000003244: 7c30f97c
	v_add3_u32 v13, v13, v114, 0x7fff                          // 000000003248: d655000d 03fee50d 00007fff
	v_add3_u32 v19, v19, v101, 0x7fff                          // 000000003254: d6550013 03fecb13 00007fff
	v_or_b32_e32 v20, 0x400000, v101                           // 000000003260: 3828caff 00400000
	v_mul_lo_u32 v22, s43, v42                                 // 000000003268: d72c0016 0202542b
	s_wait_alu depctr_va_vcc(0)                                // 000000003270: bf88ff9d
	v_cndmask_b32_e32 v10, v6, v7, vcc_lo                      // 000000003274: 02140f06
	v_add_co_u32 v0, vcc_lo, v4, v2                            // 000000003278: d7006a00 02020504
	s_wait_alu depctr_va_vcc(0)                                // 000000003280: bf88ff9d
	v_add_co_ci_u32_e64 v1, null, v5, v3, vcc_lo               // 000000003284: d5207c01 01aa0705
	v_add_co_u32 v6, vcc_lo, v4, s46                           // 00000000328c: d7006a06 02005d04
	s_wait_alu depctr_va_vcc(0)                                // 000000003294: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, s47, v5, vcc_lo              // 000000003298: d5207c07 01aa0a2f
	v_mul_lo_u32 v23, s42, v43                                 // 0000000032a0: d72c0017 0202562a
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000032a8: bf8701a3
	v_add_co_u32 v4, vcc_lo, v6, v2                            // 0000000032ac: d7006a04 02020506
	s_wait_alu depctr_va_vcc(0)                                // 0000000032b4: bf88ff9d
	v_add_co_ci_u32_e64 v5, null, v7, v3, vcc_lo               // 0000000032b8: d5207c05 01aa0707
	v_cmp_u_f32_e32 vcc_lo, v119, v119                         // 0000000032c0: 7c30ef77
	v_or_b32_e32 v24, 0x400000, v94                            // 0000000032c4: 3830bcff 00400000
	v_bfe_u32 v25, v84, 16, 1                                  // 0000000032cc: d6100019 02052154
	v_or_b32_e32 v26, 0x400000, v84                            // 0000000032d4: 3834a8ff 00400000
	v_or_b32_e32 v29, 0x400000, v82                            // 0000000032dc: 383aa4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000032e4: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v11, vcc_lo                      // 0000000032e8: 02101708
	v_add_co_u32 v11, vcc_lo, v6, s46                          // 0000000032ec: d7006a0b 02005d06
	s_wait_alu depctr_va_vcc(0)                                // 0000000032f4: bf88ff9d
	v_add_co_ci_u32_e64 v12, null, s47, v7, vcc_lo             // 0000000032f8: d5207c0c 01aa0e2f
	v_add3_u32 v25, v25, v84, 0x7fff                           // 000000003300: d6550019 03fea919 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000330c: bf8701a3
	v_add_co_u32 v6, vcc_lo, v11, v2                           // 000000003310: d7006a06 0202050b
	s_wait_alu depctr_va_vcc(0)                                // 000000003318: bf88ff9d
	v_add_co_ci_u32_e64 v7, null, v12, v3, vcc_lo              // 00000000331c: d5207c07 01aa070c
	v_cmp_u_f32_e32 vcc_lo, v114, v114                         // 000000003324: 7c30e572
	s_clause 0x2                                               // 000000003328: bf850002
	global_store_d16_hi_b16 v[0:1], v9, off                    // 00000000332c: ee09407c 04800000 00000000
	global_store_d16_hi_b16 v[4:5], v10, off                   // 000000003338: ee09407c 05000000 00000004
	global_store_d16_hi_b16 v[6:7], v8, off                    // 000000003344: ee09407c 04000000 00000006
	v_bfe_u32 v8, v109, 16, 1                                  // 000000003350: d6100008 0205216d
	v_bfe_u32 v31, v81, 16, 1                                  // 000000003358: d610001f 02052151
	v_or_b32_e32 v32, 0x400000, v81                            // 000000003360: 3840a2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003368: bf88ff9d
	v_cndmask_b32_e32 v14, v13, v14, vcc_lo                    // 00000000336c: 021c1d0d
	v_add_co_u32 v10, vcc_lo, v11, s46                         // 000000003370: d7006a0a 02005d0b
	s_wait_alu depctr_va_vcc(0)                                // 000000003378: bf88ff9d
	v_add_co_ci_u32_e64 v11, null, s47, v12, vcc_lo            // 00000000337c: d5207c0b 01aa182f
	v_add3_u32 v12, v8, v109, 0x7fff                           // 000000003384: d655000c 03fedb08 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003390: bf870003
	v_add_co_u32 v8, vcc_lo, v10, v2                           // 000000003394: d7006a08 0202050a
	v_or_b32_e32 v13, 0x400000, v109                           // 00000000339c: 381adaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000033a4: bf88ff9d
	v_add_co_ci_u32_e64 v9, null, v11, v3, vcc_lo              // 0000000033a8: d5207c09 01aa070b
	v_cmp_u_f32_e32 vcc_lo, v109, v109                         // 0000000033b0: 7c30db6d
	v_add3_u32 v31, v31, v81, 0x7fff                           // 0000000033b4: d655001f 03fea31f 00007fff
	v_or_b32_e32 v35, 0x400000, v79                            // 0000000033c0: 38469eff 00400000
	v_bfe_u32 v37, v78, 16, 1                                  // 0000000033c8: d6100025 0205214e
	v_or_b32_e32 v38, 0x400000, v78                            // 0000000033d0: 384c9cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000033d8: bf88ff9d
	v_cndmask_b32_e32 v15, v12, v13, vcc_lo                    // 0000000033dc: 021e1b0c
	v_add_co_u32 v13, vcc_lo, v10, s46                         // 0000000033e0: d7006a0d 02005d0a
	v_bfe_u32 v12, v105, 16, 1                                 // 0000000033e8: d610000c 02052169
	s_wait_alu depctr_va_vcc(0)                                // 0000000033f0: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s47, v11, vcc_lo            // 0000000033f4: d5207c10 01aa162f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000033fc: bf870193
	v_add_co_u32 v10, vcc_lo, v13, v2                          // 000000003400: d7006a0a 0202050d
	v_add3_u32 v12, v12, v105, 0x7fff                          // 000000003408: d655000c 03fed30c 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003414: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003418: bf870003
	v_add_co_ci_u32_e64 v11, null, v16, v3, vcc_lo             // 00000000341c: d5207c0b 01aa0710
	v_cmp_u_f32_e32 vcc_lo, v105, v105                         // 000000003424: 7c30d369
	v_add3_u32 v37, v37, v78, 0x7fff                           // 000000003428: d6550025 03fe9d25 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003434: bf88ff9d
	v_cndmask_b32_e32 v17, v12, v17, vcc_lo                    // 000000003438: 0222230c
	v_add_co_u32 v18, vcc_lo, v13, s46                         // 00000000343c: d7006a12 02005d0d
	s_wait_alu depctr_va_vcc(0)                                // 000000003444: bf88ff9d
	v_add_co_ci_u32_e64 v16, null, s47, v16, vcc_lo            // 000000003448: d5207c10 01aa202f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003450: bf870122
	v_add_co_u32 v12, vcc_lo, v18, v2                          // 000000003454: d7006a0c 02020512
	s_wait_alu depctr_va_vcc(0)                                // 00000000345c: bf88ff9d
	v_add_co_ci_u32_e64 v13, null, v16, v3, vcc_lo             // 000000003460: d5207c0d 01aa0710
	v_cmp_u_f32_e32 vcc_lo, v101, v101                         // 000000003468: 7c30cb65
	s_clause 0x2                                               // 00000000346c: bf850002
	global_store_d16_hi_b16 v[8:9], v14, off                   // 000000003470: ee09407c 07000000 00000008
	global_store_d16_hi_b16 v[10:11], v15, off                 // 00000000347c: ee09407c 07800000 0000000a
	global_store_d16_hi_b16 v[12:13], v17, off                 // 000000003488: ee09407c 08800000 0000000c
	v_bfe_u32 v14, v94, 16, 1                                  // 000000003494: d610000e 0205215e
	s_wait_alu depctr_va_vcc(0)                                // 00000000349c: bf88ff9d
	v_cndmask_b32_e32 v20, v19, v20, vcc_lo                    // 0000000034a0: 02282913
	v_add_co_u32 v18, vcc_lo, v18, s46                         // 0000000034a4: d7006a12 02005d12
	s_wait_alu depctr_va_vcc(0)                                // 0000000034ac: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s47, v16, vcc_lo            // 0000000034b0: d5207c13 01aa202f
	v_add3_u32 v21, v14, v94, 0x7fff                           // 0000000034b8: d6550015 03febd0e 00007fff
	v_mad_co_u64_u32 v[14:15], null, s42, v42, 0               // 0000000034c4: d6fe7c0e 0202542a
	v_add_co_u32 v16, vcc_lo, v18, v2                          // 0000000034cc: d7006a10 02020512
	s_wait_alu depctr_va_vcc(0)                                // 0000000034d4: bf88ff9d
	v_add_co_ci_u32_e64 v17, null, v19, v3, vcc_lo             // 0000000034d8: d5207c11 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v94, v94                           // 0000000034e0: 7c30bd5e
	s_delay_alu instid0(valu_dep_4)                            // 0000000034e4: bf870004
	v_add3_u32 v15, v15, v23, v22                              // 0000000034e8: d655000f 045a2f0f
	v_bfe_u32 v22, v92, 16, 1                                  // 0000000034f0: d6100016 0205215c
	s_wait_alu depctr_va_vcc(0)                                // 0000000034f8: bf88ff9d
	v_cndmask_b32_e32 v21, v21, v24, vcc_lo                    // 0000000034fc: 022a3115
	v_add_co_u32 v18, vcc_lo, v18, s46                         // 000000003500: d7006a12 02005d12
	s_wait_alu depctr_va_vcc(0)                                // 000000003508: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, s47, v19, vcc_lo            // 00000000350c: d5207c13 01aa262f
	v_lshlrev_b64_e32 v[14:15], 1, v[14:15]                    // 000000003514: 3e1c1c81
	s_delay_alu instid0(valu_dep_3)                            // 000000003518: bf870003
	v_add_co_u32 v18, vcc_lo, v18, v2                          // 00000000351c: d7006a12 02020512
	v_add3_u32 v22, v22, v92, 0x7fff                           // 000000003524: d6550016 03feb916 00007fff
	v_or_b32_e32 v23, 0x400000, v92                            // 000000003530: 382eb8ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003538: bf88ff9d
	v_add_co_ci_u32_e64 v19, null, v19, v3, vcc_lo             // 00000000353c: d5207c13 01aa0713
	v_cmp_u_f32_e32 vcc_lo, v92, v92                           // 000000003544: 7c30b95c
	s_wait_alu depctr_va_vcc(0)                                // 000000003548: bf88ff9d
	v_cndmask_b32_e32 v22, v22, v23, vcc_lo                    // 00000000354c: 022c2f16
	v_add_co_u32 v23, vcc_lo, s0, v14                          // 000000003550: d7006a17 02021c00
	s_wait_alu depctr_va_vcc(0)                                // 000000003558: bf88ff9d
	v_add_co_ci_u32_e64 v24, null, s1, v15, vcc_lo             // 00000000355c: d5207c18 01aa1e01
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003564: bf870122
	v_add_co_u32 v14, vcc_lo, v23, v2                          // 000000003568: d7006a0e 02020517
	s_wait_alu depctr_va_vcc(0)                                // 000000003570: bf88ff9d
	v_add_co_ci_u32_e64 v15, null, v24, v3, vcc_lo             // 000000003574: d5207c0f 01aa0718
	v_cmp_u_f32_e32 vcc_lo, v84, v84                           // 00000000357c: 7c30a954
	s_clause 0x2                                               // 000000003580: bf850002
	global_store_d16_hi_b16 v[16:17], v20, off                 // 000000003584: ee09407c 0a000000 00000010
	global_store_d16_hi_b16 v[18:19], v21, off                 // 000000003590: ee09407c 0a800000 00000012
	global_store_d16_hi_b16 v[14:15], v22, off                 // 00000000359c: ee09407c 0b000000 0000000e
	v_bfe_u32 v20, v83, 16, 1                                  // 0000000035a8: d6100014 02052153
	s_wait_alu depctr_va_vcc(0)                                // 0000000035b0: bf88ff9d
	v_cndmask_b32_e32 v26, v25, v26, vcc_lo                    // 0000000035b4: 02343519
	v_add_co_u32 v22, vcc_lo, v23, s46                         // 0000000035b8: d7006a16 02005d17
	s_wait_alu depctr_va_vcc(0)                                // 0000000035c0: bf88ff9d
	v_add_co_ci_u32_e64 v23, null, s47, v24, vcc_lo            // 0000000035c4: d5207c17 01aa302f
	v_add3_u32 v24, v20, v83, 0x7fff                           // 0000000035cc: d6550018 03fea714 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000035d8: bf870003
	v_add_co_u32 v20, vcc_lo, v22, v2                          // 0000000035dc: d7006a14 02020516
	v_or_b32_e32 v25, 0x400000, v83                            // 0000000035e4: 3832a6ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000035ec: bf88ff9d
	v_add_co_ci_u32_e64 v21, null, v23, v3, vcc_lo             // 0000000035f0: d5207c15 01aa0717
	v_cmp_u_f32_e32 vcc_lo, v83, v83                           // 0000000035f8: 7c30a753
	s_wait_alu depctr_va_vcc(0)                                // 0000000035fc: bf88ff9d
	v_cndmask_b32_e32 v27, v24, v25, vcc_lo                    // 000000003600: 02363318
	v_add_co_u32 v25, vcc_lo, v22, s46                         // 000000003604: d7006a19 02005d16
	v_bfe_u32 v24, v82, 16, 1                                  // 00000000360c: d6100018 02052152
	s_wait_alu depctr_va_vcc(0)                                // 000000003614: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s47, v23, vcc_lo            // 000000003618: d5207c1c 01aa2e2f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003620: bf870193
	v_add_co_u32 v22, vcc_lo, v25, v2                          // 000000003624: d7006a16 02020519
	v_add3_u32 v24, v24, v82, 0x7fff                           // 00000000362c: d6550018 03fea518 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003638: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000363c: bf870003
	v_add_co_ci_u32_e64 v23, null, v28, v3, vcc_lo             // 000000003640: d5207c17 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000003648: 7c30a552
	s_wait_alu depctr_va_vcc(0)                                // 00000000364c: bf88ff9d
	v_cndmask_b32_e32 v29, v24, v29, vcc_lo                    // 000000003650: 023a3b18
	v_add_co_u32 v30, vcc_lo, v25, s46                         // 000000003654: d7006a1e 02005d19
	s_wait_alu depctr_va_vcc(0)                                // 00000000365c: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s47, v28, vcc_lo            // 000000003660: d5207c1c 01aa382f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003668: bf870122
	v_add_co_u32 v24, vcc_lo, v30, v2                          // 00000000366c: d7006a18 0202051e
	s_wait_alu depctr_va_vcc(0)                                // 000000003674: bf88ff9d
	v_add_co_ci_u32_e64 v25, null, v28, v3, vcc_lo             // 000000003678: d5207c19 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v81, v81                           // 000000003680: 7c30a351
	s_clause 0x2                                               // 000000003684: bf850002
	global_store_d16_hi_b16 v[20:21], v26, off                 // 000000003688: ee09407c 0d000000 00000014
	global_store_d16_hi_b16 v[22:23], v27, off                 // 000000003694: ee09407c 0d800000 00000016
	global_store_d16_hi_b16 v[24:25], v29, off                 // 0000000036a0: ee09407c 0e800000 00000018
	v_bfe_u32 v26, v80, 16, 1                                  // 0000000036ac: d610001a 02052150
	s_wait_alu depctr_va_vcc(0)                                // 0000000036b4: bf88ff9d
	v_cndmask_b32_e32 v32, v31, v32, vcc_lo                    // 0000000036b8: 0240411f
	v_add_co_u32 v29, vcc_lo, v30, s46                         // 0000000036bc: d7006a1d 02005d1e
	s_wait_alu depctr_va_vcc(0)                                // 0000000036c4: bf88ff9d
	v_add_co_ci_u32_e64 v28, null, s47, v28, vcc_lo            // 0000000036c8: d5207c1c 01aa382f
	v_add3_u32 v30, v26, v80, 0x7fff                           // 0000000036d0: d655001e 03fea11a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000036dc: bf870003
	v_add_co_u32 v26, vcc_lo, v29, v2                          // 0000000036e0: d7006a1a 0202051d
	v_or_b32_e32 v31, 0x400000, v80                            // 0000000036e8: 383ea0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000036f0: bf88ff9d
	v_add_co_ci_u32_e64 v27, null, v28, v3, vcc_lo             // 0000000036f4: d5207c1b 01aa071c
	v_cmp_u_f32_e32 vcc_lo, v80, v80                           // 0000000036fc: 7c30a150
	s_wait_alu depctr_va_vcc(0)                                // 000000003700: bf88ff9d
	v_cndmask_b32_e32 v33, v30, v31, vcc_lo                    // 000000003704: 02423f1e
	v_add_co_u32 v31, vcc_lo, v29, s46                         // 000000003708: d7006a1f 02005d1d
	v_bfe_u32 v30, v79, 16, 1                                  // 000000003710: d610001e 0205214f
	s_wait_alu depctr_va_vcc(0)                                // 000000003718: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s47, v28, vcc_lo            // 00000000371c: d5207c22 01aa382f
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000003724: bf870193
	v_add_co_u32 v28, vcc_lo, v31, v2                          // 000000003728: d7006a1c 0202051f
	v_add3_u32 v30, v30, v79, 0x7fff                           // 000000003730: d655001e 03fe9f1e 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000373c: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003740: bf870003
	v_add_co_ci_u32_e64 v29, null, v34, v3, vcc_lo             // 000000003744: d5207c1d 01aa0722
	v_cmp_u_f32_e32 vcc_lo, v79, v79                           // 00000000374c: 7c309f4f
	s_wait_alu depctr_va_vcc(0)                                // 000000003750: bf88ff9d
	v_cndmask_b32_e32 v35, v30, v35, vcc_lo                    // 000000003754: 0246471e
	v_add_co_u32 v36, vcc_lo, v31, s46                         // 000000003758: d7006a24 02005d1f
	s_wait_alu depctr_va_vcc(0)                                // 000000003760: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s47, v34, vcc_lo            // 000000003764: d5207c22 01aa442f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000376c: bf870122
	v_add_co_u32 v30, vcc_lo, v36, v2                          // 000000003770: d7006a1e 02020524
	s_wait_alu depctr_va_vcc(0)                                // 000000003778: bf88ff9d
	v_add_co_ci_u32_e64 v31, null, v34, v3, vcc_lo             // 00000000377c: d5207c1f 01aa0722
	v_cmp_u_f32_e32 vcc_lo, v78, v78                           // 000000003784: 7c309d4e
	s_clause 0x2                                               // 000000003788: bf850002
	global_store_d16_hi_b16 v[26:27], v32, off                 // 00000000378c: ee09407c 10000000 0000001a
	global_store_d16_hi_b16 v[28:29], v33, off                 // 000000003798: ee09407c 10800000 0000001c
	global_store_d16_hi_b16 v[30:31], v35, off                 // 0000000037a4: ee09407c 11800000 0000001e
	v_bfe_u32 v33, v77, 16, 1                                  // 0000000037b0: d6100021 0205214d
	s_wait_alu depctr_va_vcc(0)                                // 0000000037b8: bf88ff9d
	v_cndmask_b32_e32 v32, v37, v38, vcc_lo                    // 0000000037bc: 02404d25
	v_add_co_u32 v35, vcc_lo, v36, s46                         // 0000000037c0: d7006a23 02005d24
	s_wait_alu depctr_va_vcc(0)                                // 0000000037c8: bf88ff9d
	v_add_co_ci_u32_e64 v34, null, s47, v34, vcc_lo            // 0000000037cc: d5207c22 01aa442f
	v_add3_u32 v33, v33, v77, 0x7fff                           // 0000000037d4: d6550021 03fe9b21 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 0000000037e0: bf870003
	v_add_co_u32 v2, vcc_lo, v35, v2                           // 0000000037e4: d7006a02 02020523
	v_or_b32_e32 v36, 0x400000, v77                            // 0000000037ec: 38489aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000037f4: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, v34, v3, vcc_lo              // 0000000037f8: d5207c03 01aa0722
	v_bfe_u32 v34, v74, 16, 1                                  // 000000003800: d6100022 0205214a
	v_cmp_u_f32_e32 vcc_lo, v77, v77                           // 000000003808: 7c309b4d
	v_bfe_u32 v35, v73, 16, 1                                  // 00000000380c: d6100023 02052149
	global_store_d16_hi_b16 v[2:3], v32, off                   // 000000003814: ee09407c 10000000 00000002
	v_add3_u32 v32, v34, v74, 0x7fff                           // 000000003820: d6550020 03fe9522 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000382c: bf88ff9d
	v_cndmask_b32_e32 v33, v33, v36, vcc_lo                    // 000000003830: 02424921
	v_or_b32_e32 v34, 0x400000, v74                            // 000000003834: 384494ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v74, v74                           // 00000000383c: 7c30954a
	global_store_d16_hi_b16 v[0:1], v33, off offset:32         // 000000003840: ee09407c 10800000 00002000
	v_add3_u32 v0, v35, v73, 0x7fff                            // 00000000384c: d6550000 03fe9323 00007fff
	v_or_b32_e32 v1, 0x400000, v73                             // 000000003858: 380292ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003860: bf88ff9d
	v_cndmask_b32_e32 v32, v32, v34, vcc_lo                    // 000000003864: 02404520
	v_bfe_u32 v33, v72, 16, 1                                  // 000000003868: d6100021 02052148
	v_cmp_u_f32_e32 vcc_lo, v73, v73                           // 000000003870: 7c309349
	global_store_d16_hi_b16 v[4:5], v32, off offset:32         // 000000003874: ee09407c 10000000 00002004
	v_add3_u32 v4, v33, v72, 0x7fff                            // 000000003880: d6550004 03fe9121 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 00000000388c: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003890: 02000300
	v_bfe_u32 v1, v71, 16, 1                                   // 000000003894: d6100001 02052147
	v_or_b32_e32 v5, 0x400000, v72                             // 00000000389c: 380a90ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v72, v72                           // 0000000038a4: 7c309148
	global_store_d16_hi_b16 v[6:7], v0, off offset:32          // 0000000038a8: ee09407c 00000000 00002006
	v_add3_u32 v0, v1, v71, 0x7fff                             // 0000000038b4: d6550000 03fe8f01 00007fff
	v_or_b32_e32 v1, 0x400000, v71                             // 0000000038c0: 38028eff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000038c8: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000038cc: 02080b04
	v_bfe_u32 v5, v70, 16, 1                                   // 0000000038d0: d6100005 02052146
	v_cmp_u_f32_e32 vcc_lo, v71, v71                           // 0000000038d8: 7c308f47
	v_bfe_u32 v6, v58, 16, 1                                   // 0000000038dc: d6100006 0205213a
	v_or_b32_e32 v7, 0x400000, v59                             // 0000000038e4: 380e76ff 00400000
	global_store_d16_hi_b16 v[8:9], v4, off offset:32          // 0000000038ec: ee09407c 02000000 00002008
	v_add3_u32 v4, v5, v70, 0x7fff                             // 0000000038f8: d6550004 03fe8d05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003904: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003908: 02000300
	v_bfe_u32 v1, v69, 16, 1                                   // 00000000390c: d6100001 02052145
	v_or_b32_e32 v5, 0x400000, v70                             // 000000003914: 380a8cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v70, v70                           // 00000000391c: 7c308d46
	v_add3_u32 v6, v6, v58, 0x7fff                             // 000000003920: d6550006 03fe7506 00007fff
	global_store_d16_hi_b16 v[10:11], v0, off offset:32        // 00000000392c: ee09407c 00000000 0000200a
	v_add3_u32 v0, v1, v69, 0x7fff                             // 000000003938: d6550000 03fe8b01 00007fff
	v_or_b32_e32 v1, 0x400000, v69                             // 000000003944: 38028aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000394c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003950: 02080b04
	v_bfe_u32 v5, v67, 16, 1                                   // 000000003954: d6100005 02052143
	v_cmp_u_f32_e32 vcc_lo, v69, v69                           // 00000000395c: 7c308b45
	v_or_b32_e32 v8, 0x400000, v58                             // 000000003960: 381074ff 00400000
	v_or_b32_e32 v9, 0x400000, v44                             // 000000003968: 381258ff 00400000
	global_store_d16_hi_b16 v[12:13], v4, off offset:32        // 000000003970: ee09407c 02000000 0000200c
	v_add3_u32 v4, v5, v67, 0x7fff                             // 00000000397c: d6550004 03fe8705 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003988: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 00000000398c: 02000300
	v_bfe_u32 v1, v68, 16, 1                                   // 000000003990: d6100001 02052144
	v_or_b32_e32 v5, 0x400000, v67                             // 000000003998: 380a86ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v67, v67                           // 0000000039a0: 7c308743
	global_store_d16_hi_b16 v[16:17], v0, off offset:32        // 0000000039a4: ee09407c 00000000 00002010
	v_add3_u32 v0, v1, v68, 0x7fff                             // 0000000039b0: d6550000 03fe8901 00007fff
	v_or_b32_e32 v1, 0x400000, v68                             // 0000000039bc: 380288ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000039c4: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 0000000039c8: 02080b04
	v_bfe_u32 v5, v66, 16, 1                                   // 0000000039cc: d6100005 02052142
	v_cmp_u_f32_e32 vcc_lo, v68, v68                           // 0000000039d4: 7c308944
	global_store_d16_hi_b16 v[18:19], v4, off offset:32        // 0000000039d8: ee09407c 02000000 00002012
	v_add3_u32 v4, v5, v66, 0x7fff                             // 0000000039e4: d6550004 03fe8505 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000039f0: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 0000000039f4: 02000300
	v_bfe_u32 v1, v64, 16, 1                                   // 0000000039f8: d6100001 02052140
	v_or_b32_e32 v5, 0x400000, v66                             // 000000003a00: 380a84ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 000000003a08: 7c308542
	global_store_d16_hi_b16 v[14:15], v0, off offset:32        // 000000003a0c: ee09407c 00000000 0000200e
	v_add3_u32 v0, v1, v64, 0x7fff                             // 000000003a18: d6550000 03fe8101 00007fff
	v_or_b32_e32 v1, 0x400000, v64                             // 000000003a24: 380280ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a2c: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003a30: 02080b04
	v_bfe_u32 v5, v62, 16, 1                                   // 000000003a34: d6100005 0205213e
	v_cmp_u_f32_e32 vcc_lo, v64, v64                           // 000000003a3c: 7c308140
	global_store_d16_hi_b16 v[20:21], v4, off offset:32        // 000000003a40: ee09407c 02000000 00002014
	v_add3_u32 v4, v5, v62, 0x7fff                             // 000000003a4c: d6550004 03fe7d05 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003a58: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003a5c: 02000300
	v_bfe_u32 v1, v61, 16, 1                                   // 000000003a60: d6100001 0205213d
	v_or_b32_e32 v5, 0x400000, v62                             // 000000003a68: 380a7cff 00400000
	v_cmp_u_f32_e32 vcc_lo, v62, v62                           // 000000003a70: 7c307d3e
	global_store_d16_hi_b16 v[22:23], v0, off offset:32        // 000000003a74: ee09407c 00000000 00002016
	v_add3_u32 v0, v1, v61, 0x7fff                             // 000000003a80: d6550000 03fe7b01 00007fff
	v_or_b32_e32 v1, 0x400000, v61                             // 000000003a8c: 38027aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a94: bf88ff9d
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo                       // 000000003a98: 02080b04
	v_bfe_u32 v5, v59, 16, 1                                   // 000000003a9c: d6100005 0205213b
	v_cmp_u_f32_e32 vcc_lo, v61, v61                           // 000000003aa4: 7c307b3d
	s_delay_alu instid0(valu_dep_2)                            // 000000003aa8: bf870002
	v_add3_u32 v5, v5, v59, 0x7fff                             // 000000003aac: d6550005 03fe7705 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003ab8: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v1, vcc_lo                       // 000000003abc: 02000300
	v_cmp_u_f32_e32 vcc_lo, v59, v59                           // 000000003ac0: 7c30773b
	v_bfe_u32 v1, v44, 16, 1                                   // 000000003ac4: d6100001 0205212c
	s_wait_alu depctr_va_vcc(0)                                // 000000003acc: bf88ff9d
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000003ad0: 020a0f05
	v_cmp_u_f32_e32 vcc_lo, v58, v58                           // 000000003ad4: 7c30753a
	s_delay_alu instid0(valu_dep_3)                            // 000000003ad8: bf870003
	v_add3_u32 v1, v1, v44, 0x7fff                             // 000000003adc: d6550001 03fe5901 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003ae8: bf88ff9d
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo                       // 000000003aec: 020c1106
	v_cmp_u_f32_e32 vcc_lo, v44, v44                           // 000000003af0: 7c30592c
	s_wait_alu depctr_va_vcc(0)                                // 000000003af4: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v9, vcc_lo                       // 000000003af8: 02021301
	s_clause 0x3                                               // 000000003afc: bf850003
	global_store_d16_hi_b16 v[24:25], v4, off offset:32        // 000000003b00: ee09407c 02000000 00002018
	global_store_d16_hi_b16 v[26:27], v0, off offset:32        // 000000003b0c: ee09407c 00000000 0000201a
	global_store_d16_hi_b16 v[28:29], v5, off offset:32        // 000000003b18: ee09407c 02800000 0000201c
	global_store_d16_hi_b16 v[30:31], v6, off offset:32        // 000000003b24: ee09407c 03000000 0000201e
	global_store_d16_hi_b16 v[2:3], v1, off offset:32          // 000000003b30: ee09407c 00800000 00002002
	s_nop 0                                                    // 000000003b3c: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 000000003b40: bfb60003
	s_endpgm                                                   // 000000003b44: bfb00000
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
	s_code_end                                                 // 000000003c80: bf9f0000
	s_code_end                                                 // 000000003c84: bf9f0000
	s_code_end                                                 // 000000003c88: bf9f0000
	s_code_end                                                 // 000000003c8c: bf9f0000
	s_code_end                                                 // 000000003c90: bf9f0000
	s_code_end                                                 // 000000003c94: bf9f0000
	s_code_end                                                 // 000000003c98: bf9f0000
	s_code_end                                                 // 000000003c9c: bf9f0000
	s_code_end                                                 // 000000003ca0: bf9f0000
	s_code_end                                                 // 000000003ca4: bf9f0000
	s_code_end                                                 // 000000003ca8: bf9f0000
	s_code_end                                                 // 000000003cac: bf9f0000
	s_code_end                                                 // 000000003cb0: bf9f0000
	s_code_end                                                 // 000000003cb4: bf9f0000
	s_code_end                                                 // 000000003cb8: bf9f0000
	s_code_end                                                 // 000000003cbc: bf9f0000
	s_code_end                                                 // 000000003cc0: bf9f0000
	s_code_end                                                 // 000000003cc4: bf9f0000
	s_code_end                                                 // 000000003cc8: bf9f0000
	s_code_end                                                 // 000000003ccc: bf9f0000
	s_code_end                                                 // 000000003cd0: bf9f0000
	s_code_end                                                 // 000000003cd4: bf9f0000
	s_code_end                                                 // 000000003cd8: bf9f0000
	s_code_end                                                 // 000000003cdc: bf9f0000
	s_code_end                                                 // 000000003ce0: bf9f0000
	s_code_end                                                 // 000000003ce4: bf9f0000
	s_code_end                                                 // 000000003ce8: bf9f0000
	s_code_end                                                 // 000000003cec: bf9f0000
	s_code_end                                                 // 000000003cf0: bf9f0000
	s_code_end                                                 // 000000003cf4: bf9f0000
	s_code_end                                                 // 000000003cf8: bf9f0000
	s_code_end                                                 // 000000003cfc: bf9f0000
