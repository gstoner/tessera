
/tmp/tmpetemsy3x.hsaco:	file format elf64-amdgpu

disassembly of section .text:

0000000000001b00 <packed_folded_w4a8>:
	v_dual_mov_b32 v56, 0 :: v_dual_and_b32 v71, 0xc0, v0      // 000000001b00: ca240080 384600ff 000000c0
	v_lshlrev_b32_e32 v2, 4, v0                                // 000000001b0c: 30040084
	v_lshrrev_b32_e32 v1, 2, v0                                // 000000001b10: 32020082
	s_clause 0x1                                               // 000000001b14: bf850001
	s_load_b64 s[2:3], s[0:1], 0xd8                            // 000000001b18: f4002080 f80000d8
	s_load_b128 s[16:19], s[0:1], 0xc8                         // 000000001b20: f4004400 f80000c8
	v_mov_b32_e32 v59, v56                                     // 000000001b28: 7e760338
	s_clause 0x2                                               // 000000001b2c: bf850002
	s_load_b64 s[6:7], s[0:1], 0x8                             // 000000001b30: f4002180 f8000008
	s_load_b64 s[4:5], s[0:1], 0x30                            // 000000001b38: f4002100 f8000030
	s_load_b64 s[8:9], s[0:1], 0x80                            // 000000001b40: f4002200 f8000080
	v_dual_mov_b32 v2, v56 :: v_dual_and_b32 v3, 48, v2        // 000000001b48: ca240138 020204b0
	v_mul_u32_u24_e32 v5, 0x50, v1                             // 000000001b50: 160a02ff 00000050
	v_dual_mov_b32 v57, v56 :: v_dual_and_b32 v70, 15, v0      // 000000001b58: ca240138 3946008f
	v_or_b32_e32 v80, 16, v71                                  // 000000001b60: 38a08e90
	v_or_b32_e32 v102, 32, v71                                 // 000000001b64: 38cc8ea0
	s_delay_alu instid0(valu_dep_4) | instskip(skip_2) | instid1(valu_dep_4)// 000000001b68: bf870234
	v_dual_mov_b32 v4, v56 :: v_dual_add_nc_u32 v73, v5, v3    // 000000001b6c: ca200138 04480705
	v_lshrrev_b32_e32 v5, 1, v0                                // 000000001b74: 320a0081
	v_dual_mov_b32 v63, v56 :: v_dual_and_b32 v6, 0xcf, v0     // 000000001b78: ca240138 3f0600ff 000000cf
	v_or_b32_e32 v7, v102, v70                                 // 000000001b84: 380e8d66
	s_mov_b32 s12, ttmp7                                       // 000000001b88: be8c0073
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000001b8c: bf870193
	v_dual_mov_b32 v25, v56 :: v_dual_and_b32 v138, 8, v5      // 000000001b90: ca240138 198a0a88
	v_mul_u32_u24_e32 v5, 0x50, v6                             // 000000001b98: 160a0cff 00000050
	v_or_b32_e32 v6, v80, v70                                  // 000000001ba0: 380c8d50
	s_ashr_i32 s13, ttmp7, 31                                  // 000000001ba4: 860d9f73
	v_or_b32_e32 v130, 48, v71                                 // 000000001ba8: 39048eb0
	s_lshl_b64 s[22:23], s[12:13], 8                           // 000000001bac: 8496880c
	v_or_b32_e32 v74, v138, v5                                 // 000000001bb0: 38940b8a
	v_mul_u32_u24_e32 v6, 0x50, v6                             // 000000001bb4: 160c0cff 00000050
	v_mul_u32_u24_e32 v5, 0x50, v7                             // 000000001bbc: 160a0eff 00000050
	v_or_b32_e32 v11, s22, v1                                  // 000000001bc4: 38160216
	s_mov_b32 s10, ttmp9                                       // 000000001bc8: be8a0075
	s_ashr_i32 s11, ttmp9, 31                                  // 000000001bcc: 860b9f75
	v_or_b32_e32 v7, v130, v70                                 // 000000001bd0: 380e8d82
	v_dual_mov_b32 v27, v56 :: v_dual_and_b32 v8, 47, v0       // 000000001bd4: ca240138 1b0800af
	v_or_b32_e32 v75, v6, v138                                 // 000000001bdc: 38971506
	v_or_b32_e32 v76, v5, v138                                 // 000000001be0: 38991505
	s_wait_kmcnt 0x0                                           // 000000001be4: bfc70000
	v_mul_lo_u32 v13, s3, v11                                  // 000000001be8: d72c000d 02021603
	v_mad_co_u64_u32 v[5:6], null, s2, v11, v[3:4]             // 000000001bf0: d6fe7c05 040e1602
	s_lshl_b64 s[24:25], s[10:11], 6                           // 000000001bf8: 8498860a
	s_lshr_b64 s[10:11], s[2:3], 5                             // 000000001bfc: 858a8502
	v_or_b32_e32 v9, s24, v1                                   // 000000001c00: 38120218
	s_mul_u64 s[10:11], s[10:11], s[18:19]                     // 000000001c04: aa8a120a
	v_mul_u32_u24_e32 v7, 0x50, v7                             // 000000001c08: 160e0eff 00000050
	s_add_nc_u64 s[14:15], s[8:9], s[10:11]                    // 000000001c10: a98e0a08
	s_lshr_b64 s[10:11], s[2:3], 4                             // 000000001c14: 858a8402
	v_mul_u32_u24_e32 v8, 0x50, v8                             // 000000001c18: 161010ff 00000050
	v_or_b32_e32 v14, 64, v11                                  // 000000001c20: 381c16c0
	v_mov_b32_e32 v10, s25                                     // 000000001c24: 7e140219
	s_mul_i32 s11, s2, s23                                     // 000000001c28: 960b1702
	v_add_co_u32 v64, vcc_lo, s14, v9                          // 000000001c2c: d7006a40 0202120e
	v_add3_u32 v6, v13, v6, s11                                // 000000001c34: d6550006 002e0d0d
	v_or_b32_e32 v77, v7, v138                                 // 000000001c3c: 389b1507
	v_or_b32_e32 v15, v138, v8                                 // 000000001c40: 381e118a
	v_mul_lo_u32 v16, s3, v14                                  // 000000001c44: d72c0010 02021c03
	v_mad_co_u64_u32 v[7:8], null, s2, v14, v[3:4]             // 000000001c4c: d6fe7c07 040e1c02
	v_add_co_ci_u32_e64 v65, null, s15, v10, vcc_lo            // 000000001c54: d5207c41 01aa140f
	v_add_co_u32 v78, vcc_lo, s6, v5                           // 000000001c5c: d7006a4e 02020a06
	v_or_b32_e32 v13, 0x80, v11                                // 000000001c64: 381a16ff 00000080
	s_wait_alu depctr_va_vcc(0)                                // 000000001c6c: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, s7, v6, vcc_lo              // 000000001c70: d5207c4f 01aa0c07
	v_and_b32_e32 v5, 3, v0                                    // 000000001c78: 360a0083
	v_alignbit_b32 v10, v10, v9, 4                             // 000000001c7c: d616000a 0212130a
	v_dual_mov_b32 v6, v56 :: v_dual_add_nc_u32 v87, 0x5000, v15// 000000001c84: ca200138 06561eff 00005000
	s_lshr_b32 s12, s3, 4                                      // 000000001c90: 850c8403
	v_add3_u32 v14, v16, v8, s11                               // 000000001c94: d655000e 002e1110
	v_mul_lo_u32 v16, s3, v13                                  // 000000001c9c: d72c0010 02021a03
	v_mad_co_u64_u32 v[8:9], null, s2, v13, v[3:4]             // 000000001ca4: d6fe7c08 040e1a02
	v_mul_lo_u32 v13, s12, v10                                 // 000000001cac: d72c000d 0202140c
	v_mad_co_u64_u32 v[5:6], null, s10, v10, v[5:6]            // 000000001cb4: d6fe7c05 0416140a
	s_lshr_b32 s12, s25, 4                                     // 000000001cbc: 850c8419
	v_dual_mov_b32 v61, v56 :: v_dual_and_b32 v72, 32, v0      // 000000001cc0: ca240138 3d4800a0
	v_or_b32_e32 v11, 0xc0, v11                                // 000000001cc8: 381616ff 000000c0
	s_wait_alu depctr_sa_sdst(0)                               // 000000001cd0: bf88ff9e
	s_mul_i32 s10, s10, s12                                    // 000000001cd4: 960a0c0a
	v_add_co_u32 v81, vcc_lo, s6, v7                           // 000000001cd8: d7006a51 02020e06
	v_bfe_u32 v7, v0, 1, 1                                     // 000000001ce0: d6100007 02050300
	s_wait_alu depctr_sa_sdst(0)                               // 000000001ce8: bf88ff9e
	v_add3_u32 v6, v13, v6, s10                                // 000000001cec: d6550006 002a0d0d
	v_or_b32_e32 v160, 16, v72                                 // 000000001cf4: 39409090
	v_mul_lo_u32 v10, s3, v11                                  // 000000001cf8: d72c000a 02021603
	v_mad_co_u64_u32 v[3:4], null, s2, v11, v[3:4]             // 000000001d00: d6fe7c03 040e1602
	v_mad_co_u64_u32 v[1:2], null, s18, v7, v[1:2]             // 000000001d08: d6fe7c01 04060e12
	v_lshlrev_b64_e32 v[5:6], 7, v[5:6]                        // 000000001d10: 3e0a0a87
	v_or_b32_e32 v12, v160, v70                                // 000000001d14: 38188da0
	v_add3_u32 v9, v16, v9, s11                                // 000000001d18: d6550009 002e1310
	s_wait_alu depctr_va_vcc(0)                                // 000000001d20: bf88ff9d
	v_add_co_ci_u32_e64 v82, null, s7, v14, vcc_lo             // 000000001d24: d5207c52 01aa1c07
	v_add_co_u32 v83, vcc_lo, s6, v8                           // 000000001d2c: d7006a53 02021006
	v_add3_u32 v4, v10, v4, s11                                // 000000001d34: d6550004 002e090a
	v_and_or_b32 v0, v0, 60, v5                                // 000000001d3c: d6570000 04157900
	v_mul_u32_u24_e32 v12, 0x50, v12                           // 000000001d44: 161818ff 00000050
	s_wait_alu depctr_va_vcc(0)                                // 000000001d4c: bf88ff9d
	v_add_co_ci_u32_e64 v84, null, s7, v9, vcc_lo              // 000000001d50: d5207c54 01aa1207
	v_add_co_u32 v85, vcc_lo, s6, v3                           // 000000001d58: d7006a55 02020606
	v_mad_co_u64_u32 v[2:3], null, s19, v7, v[2:3]             // 000000001d60: d6fe7c02 040a0e13
	s_wait_alu depctr_va_vcc(0)                                // 000000001d68: bf88ff9d
	v_add_co_ci_u32_e64 v86, null, s7, v4, vcc_lo              // 000000001d6c: d5207c56 01aa0807
	v_add_co_u32 v0, vcc_lo, s4, v0                            // 000000001d74: d7006a00 02020004
	s_add_nc_u64 s[2:3], s[8:9], s[24:25]                      // 000000001d7c: a9821808
	v_or_b32_e32 v12, v12, v138                                // 000000001d80: 3819150c
	s_wait_alu depctr_va_vcc(0)                                // 000000001d84: bf88ff9d
	v_add_co_ci_u32_e64 v3, null, s5, v6, vcc_lo               // 000000001d88: d5207c03 01aa0c05
	s_wait_alu depctr_sa_sdst(0)                               // 000000001d90: bf88ff9e
	v_add_co_u32 v66, vcc_lo, s2, v1                           // 000000001d94: d7006a42 02020202
	s_wait_alu depctr_va_vcc(0)                                // 000000001d9c: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s3, v2, vcc_lo              // 000000001da0: d5207c43 01aa0403
	v_add_co_u32 v68, vcc_lo, 0x43, v0                         // 000000001da8: d7006a44 020200ff 00000043
	s_wait_alu depctr_va_vcc(0)                                // 000000001db4: bf88ff9d
	v_add_co_ci_u32_e64 v69, null, 0, v3, vcc_lo               // 000000001db8: d5207c45 01aa0680
	v_dual_mov_b32 v29, v56 :: v_dual_add_nc_u32 v88, 0x5000, v12// 000000001dc0: ca200138 1d5818ff 00005000
	v_dual_mov_b32 v58, v56 :: v_dual_mov_b32 v31, v56         // 000000001dcc: ca100138 3a1e0138
	v_dual_mov_b32 v60, v56 :: v_dual_mov_b32 v49, v56         // 000000001dd4: ca100138 3c300138
	v_dual_mov_b32 v62, v56 :: v_dual_mov_b32 v51, v56         // 000000001ddc: ca100138 3e320138
	v_dual_mov_b32 v24, v56 :: v_dual_mov_b32 v53, v56         // 000000001de4: ca100138 18340138
	v_dual_mov_b32 v26, v56 :: v_dual_mov_b32 v55, v56         // 000000001dec: ca100138 1a360138
	v_dual_mov_b32 v28, v56 :: v_dual_mov_b32 v17, v56         // 000000001df4: ca100138 1c100138
	v_dual_mov_b32 v30, v56 :: v_dual_mov_b32 v19, v56         // 000000001dfc: ca100138 1e120138
	v_dual_mov_b32 v48, v56 :: v_dual_mov_b32 v21, v56         // 000000001e04: ca100138 30140138
	v_dual_mov_b32 v50, v56 :: v_dual_mov_b32 v23, v56         // 000000001e0c: ca100138 32160138
	v_dual_mov_b32 v52, v56 :: v_dual_mov_b32 v41, v56         // 000000001e14: ca100138 34280138
	v_dual_mov_b32 v54, v56 :: v_dual_mov_b32 v43, v56         // 000000001e1c: ca100138 362a0138
	v_dual_mov_b32 v16, v56 :: v_dual_mov_b32 v45, v56         // 000000001e24: ca100138 102c0138
	v_dual_mov_b32 v18, v56 :: v_dual_mov_b32 v47, v56         // 000000001e2c: ca100138 122e0138
	v_dual_mov_b32 v20, v56 :: v_dual_mov_b32 v9, v56          // 000000001e34: ca100138 14080138
	v_dual_mov_b32 v22, v56 :: v_dual_mov_b32 v11, v56         // 000000001e3c: ca100138 160a0138
	v_dual_mov_b32 v40, v56 :: v_dual_mov_b32 v13, v56         // 000000001e44: ca100138 280c0138
	v_dual_mov_b32 v42, v56 :: v_dual_mov_b32 v15, v56         // 000000001e4c: ca100138 2a0e0138
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v33, v56         // 000000001e54: ca100138 2c200138
	v_dual_mov_b32 v46, v56 :: v_dual_mov_b32 v35, v56         // 000000001e5c: ca100138 2e220138
	v_dual_mov_b32 v8, v56 :: v_dual_mov_b32 v37, v56          // 000000001e64: ca100138 08240138
	v_dual_mov_b32 v10, v56 :: v_dual_mov_b32 v39, v56         // 000000001e6c: ca100138 0a260138
	v_dual_mov_b32 v12, v56 :: v_dual_mov_b32 v1, v56          // 000000001e74: ca100138 0c000138
	v_dual_mov_b32 v14, v56 :: v_dual_mov_b32 v3, v56          // 000000001e7c: ca100138 0e020138
	v_dual_mov_b32 v32, v56 :: v_dual_mov_b32 v5, v56          // 000000001e84: ca100138 20040138
	v_dual_mov_b32 v34, v56 :: v_dual_mov_b32 v7, v56          // 000000001e8c: ca100138 22060138
	v_mov_b32_e32 v36, v56                                     // 000000001e94: 7e480338
	v_mov_b32_e32 v38, v56                                     // 000000001e98: 7e4c0338
	v_mov_b32_e32 v0, v56                                      // 000000001e9c: 7e000338
	v_mov_b32_e32 v2, v56                                      // 000000001ea0: 7e040338
	v_mov_b32_e32 v4, v56                                      // 000000001ea4: 7e080338
	v_mov_b32_e32 v6, v56                                      // 000000001ea8: 7e0c0338
	s_lshl_b64 s[20:21], s[18:19], 1                           // 000000001eac: 84948112
	s_mov_b64 s[26:27], 0                                      // 000000001eb0: be9a0180
	global_load_u8 v107, v[66:67], off                         // 000000001eb4: ee04007c 0000006b 00000042
	global_load_u8 v108, v[64:65], off                         // 000000001ec0: ee04007c 0000006c 00000040
	global_load_d16_u8 v101, v[66:67], off                     // 000000001ecc: ee07807c 00000065 00000042
	global_load_d16_hi_u8 v101, v[64:65], off                  // 000000001ed8: ee08407c 00000065 00000040
	v_add_co_u32 v89, vcc_lo, v78, s26                         // 000000001ee4: d7006a59 0200354e
	s_wait_alu depctr_va_vcc(0)                                // 000000001eec: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s27, v79, vcc_lo            // 000000001ef0: d5207c5a 01aa9e1b
	v_add_co_u32 v93, s2, v81, s26                             // 000000001ef8: d700025d 02003551
	s_wait_alu depctr_va_sdst(0)                               // 000000001f00: bf88f19f
	v_add_co_ci_u32_e64 v94, null, s27, v82, s2                // 000000001f04: d5207c5e 000aa41b
	global_load_b128 v[89:92], v[89:90], off                   // 000000001f0c: ee05c07c 00000059 00000059
	s_clause 0x1                                               // 000000001f18: bf850001
	global_load_b32 v109, v[68:69], off offset:-67             // 000000001f1c: ee05007c 0000006d ffffbd44
	global_load_b32 v110, v[68:69], off offset:-3              // 000000001f28: ee05007c 0000006e fffffd44
	global_load_b128 v[93:96], v[93:94], off                   // 000000001f34: ee05c07c 0000005d 0000005d
	v_add_co_u32 v97, s3, v83, s26                             // 000000001f40: d7000361 02003553
	v_add_co_u32 v103, s4, v85, s26                            // 000000001f48: d7000467 02003555
	s_wait_alu depctr_va_sdst(0)                               // 000000001f50: bf88f19f
	v_add_co_ci_u32_e64 v98, null, s27, v84, s3                // 000000001f54: d5207c62 000ea81b
	v_add_co_ci_u32_e64 v104, null, s27, v86, s4               // 000000001f5c: d5207c68 0012ac1b
	v_add_co_u32 v66, vcc_lo, v66, s20                         // 000000001f64: d7006a42 02002942
	s_clause 0x1                                               // 000000001f6c: bf850001
	global_load_b128 v[97:100], v[97:98], off                  // 000000001f70: ee05c07c 00000061 00000061
	global_load_b128 v[103:106], v[103:104], off               // 000000001f7c: ee05c07c 00000067 00000067
	s_wait_alu depctr_va_vcc(0)                                // 000000001f88: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s21, v67, vcc_lo            // 000000001f8c: d5207c43 01aa8615
	s_barrier_signal -1                                        // 000000001f94: be804ec1
	s_barrier_wait 0xffff                                      // 000000001f98: bf94ffff
	v_add_co_u32 v68, s2, 0x200, v68                           // 000000001f9c: d7000244 020288ff 00000200
	s_wait_alu depctr_va_sdst(0)                               // 000000001fa8: bf88f19f
	v_add_co_ci_u32_e64 v69, null, 0, v69, s2                  // 000000001fac: d5207c45 000a8a80
	s_add_nc_u64 s[26:27], s[26:27], 64                        // 000000001fb4: a99ac01a
	s_wait_loadcnt 0x8                                         // 000000001fb8: bfc00008
	v_sub_nc_u32_e32 v111, v108, v107                          // 000000001fbc: 4cded76c
	s_wait_loadcnt 0x6                                         // 000000001fc0: bfc00006
	v_cmp_eq_u16_e64 s2, 0, v101.l                             // 000000001fc4: d43a0002 0202ca80
	v_cmp_eq_u16_e32 vcc_lo, v101.h, v101.l                    // 000000001fcc: 7c74cbe5
	s_delay_alu instid0(valu_dep_3)                            // 000000001fd0: bf870003
	v_cmp_ne_u32_e64 s3, 2, v111                               // 000000001fd4: d44d0003 0202de82
	v_cmp_ne_u32_e64 s4, 3, v111                               // 000000001fdc: d44d0004 0202de83
	s_wait_alu depctr_va_vcc(0)                                // 000000001fe4: bf88ff9d
	v_cndmask_b32_e64 v133, 0, 0x3c383000, vcc_lo              // 000000001fe8: d5010085 01a9fe80 3c383000
	v_cndmask_b32_e64 v134, 0, 0x4c484440, vcc_lo              // 000000001ff4: d5010086 01a9fe80 4c484440
	v_cmp_ne_u32_e32 vcc_lo, 1, v111                           // 000000002000: 7c9ade81
	v_cmp_ne_u32_e64 s5, 4, v111                               // 000000002004: d44d0005 0202de84
	s_wait_loadcnt 0x5                                         // 00000000200c: bfc00005
	ds_store_b128 v73, v[89:92]                                // 000000002010: db7c0000 00005949
	v_cmp_ne_u32_e64 s6, 5, v111                               // 000000002018: d44d0006 0202de85
	v_cmp_ne_u32_e64 s7, 6, v111                               // 000000002020: d44d0007 0202de86
	s_wait_alu depctr_va_vcc(0)                                // 000000002028: bf88ff9d
	v_cndmask_b32_e32 v91, 0x44403c38, v134, vcc_lo            // 00000000202c: 02b70cff 44403c38
	v_cndmask_b32_e32 v92, 0x34302800, v133, vcc_lo            // 000000002034: 02b90aff 34302800
	v_cmp_ne_u32_e64 s8, 7, v111                               // 00000000203c: d44d0008 0202de87
	v_cmp_ne_u32_e64 s9, 8, v111                               // 000000002044: d44d0009 0202de88
	v_cmp_ne_u32_e64 s10, 9, v111                              // 00000000204c: d44d000a 0202de89
	s_wait_alu depctr_va_sdst(0)                               // 000000002054: bf88f19f
	v_cndmask_b32_e64 v91, 0x3c383430, v91, s3                 // 000000002058: d501005b 000eb6ff 3c383430
	v_cndmask_b32_e64 v92, 0x2c282000, v92, s3                 // 000000002064: d501005c 000eb8ff 2c282000
	v_cmp_ne_u32_e64 s11, 10, v111                             // 000000002070: d44d000b 0202de8a
	v_cmp_ne_u32_e64 s12, 11, v111                             // 000000002078: d44d000c 0202de8b
	v_cmp_ne_u32_e64 s13, 12, v111                             // 000000002080: d44d000d 0202de8c
	v_cndmask_b32_e64 v91, 0x34302c28, v91, s4                 // 000000002088: d501005b 0012b6ff 34302c28
	v_cndmask_b32_e64 v92, 0x24201800, v92, s4                 // 000000002094: d501005c 0012b8ff 24201800
	s_wait_loadcnt 0x2                                         // 0000000020a0: bfc00002
	ds_store_b128 v73, v[93:96] offset:5120                    // 0000000020a4: db7c1400 00005d49
	v_lshrrev_b32_e32 v117, 1, v109                            // 0000000020ac: 32eada81
	v_lshrrev_b32_e32 v126, 5, v110                            // 0000000020b0: 32fcdc85
	v_cndmask_b32_e64 v91, 0x2c282420, v91, s5                 // 0000000020b4: d501005b 0016b6ff 2c282420
	v_cndmask_b32_e64 v92, 0x1c181000, v92, s5                 // 0000000020c0: d501005c 0016b8ff 1c181000
	v_lshrrev_b32_e32 v118, 5, v109                            // 0000000020cc: 32ecda85
	v_lshrrev_b32_e32 v127, 9, v110                            // 0000000020d0: 32fedc89
	v_lshrrev_b32_e32 v119, 9, v109                            // 0000000020d4: 32eeda89
	v_cndmask_b32_e64 v91, 0x24201c18, v91, s6                 // 0000000020d8: d501005b 001ab6ff 24201c18
	v_cndmask_b32_e64 v92, 0x14100800, v92, s6                 // 0000000020e4: d501005c 001ab8ff 14100800
	v_lshrrev_b32_e32 v128, 13, v110                           // 0000000020f0: 3300dc8d
	s_and_b32 vcc_lo, s13, s12                                 // 0000000020f4: 8b6a0c0d
	v_lshrrev_b32_e32 v120, 13, v109                           // 0000000020f8: 32f0da8d
	v_cndmask_b32_e64 v91, 0x1c181410, v91, s7                 // 0000000020fc: d501005b 001eb6ff 1c181410
	v_cndmask_b32_e64 v92, 0xc080400, v92, s7                  // 000000002108: d501005c 001eb8ff 0c080400
	v_lshrrev_b32_e32 v123, 25, v109                           // 000000002114: 32f6da99
	v_lshrrev_b32_e32 v129, 17, v110                           // 000000002118: 3302dc91
	v_lshlrev_b32_e32 v116, 3, v109                            // 00000000211c: 30e8da83
	v_cndmask_b32_e64 v91, 0x14100c08, v91, s8                 // 000000002120: d501005b 0022b6ff 14100c08
	v_cndmask_b32_e64 v92, 0x6040200, v92, s8                  // 00000000212c: d501005c 0022b8ff 06040200
	v_lshrrev_b32_e32 v121, 17, v109                           // 000000002138: 32f2da91
	v_lshrrev_b32_e32 v131, 21, v110                           // 00000000213c: 3306dc95
	v_lshrrev_b32_e32 v112, 8, v109                            // 000000002140: 32e0da88
	v_cndmask_b32_e64 v91, 0xc080604, v91, s9                  // 000000002144: d501005b 0026b6ff 0c080604
	v_cndmask_b32_e64 v92, 0x3020100, v92, s9                  // 000000002150: d501005c 0026b8ff 03020100
	v_lshrrev_b32_e32 v113, 24, v109                           // 00000000215c: 32e2da98
	v_lshrrev_b32_e32 v114, 8, v110                            // 000000002160: 32e4dc88
	v_lshrrev_b32_e32 v115, 24, v110                           // 000000002164: 32e6dc98
	v_cndmask_b32_e64 v91, 0x6040302, v91, s10                 // 000000002168: d501005b 002ab6ff 06040302
	v_cndmask_b32_e64 v92, 0x2010000, v92, s10                 // 000000002174: d501005c 002ab8ff 02010000
	v_lshrrev_b32_e32 v122, 21, v109                           // 000000002180: 32f4da95
	v_lshrrev_b32_e32 v125, 1, v110                            // 000000002184: 32fadc81
	v_lshrrev_b32_e32 v132, 25, v110                           // 000000002188: 3308dc99
	v_cndmask_b32_e64 v91, 0x3020201, v91, s11                 // 00000000218c: d501005b 002eb6ff 03020201
	v_cndmask_b32_e64 v92, 0x1000000, v92, s11                 // 000000002198: d501005c 002eb8ff 01000000
	v_and_b32_e32 v117, 56, v117                               // 0000000021a4: 36eaeab8
	v_and_b32_e32 v126, 56, v126                               // 0000000021a8: 36fcfcb8
	v_and_b32_e32 v118, 56, v118                               // 0000000021ac: 36ececb8
	v_cndmask_b32_e64 v93, 0x2010100, v91, s12                 // 0000000021b0: d501005d 0032b6ff 02010100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021bc: bf88ff9e
	v_cndmask_b32_e32 v91, 0, v92, vcc_lo                      // 0000000021c0: 02b6b880
	v_and_b32_e32 v127, 56, v127                               // 0000000021c4: 36fefeb8
	v_and_b32_e32 v119, 56, v119                               // 0000000021c8: 36eeeeb8
	v_and_b32_e32 v128, 56, v128                               // 0000000021cc: 370100b8
	v_cndmask_b32_e64 v92, 0x1000000, v93, s13                 // 0000000021d0: d501005c 0036baff 01000000
	v_and_b32_e32 v120, 56, v120                               // 0000000021dc: 36f0f0b8
	v_and_b32_e32 v123, 56, v123                               // 0000000021e0: 36f6f6b8
	v_and_b32_e32 v129, 56, v129                               // 0000000021e4: 370302b8
	v_lshlrev_b32_e32 v124, 3, v110                            // 0000000021e8: 30f8dc83
	v_and_b32_e32 v121, 56, v121                               // 0000000021ec: 36f2f2b8
	v_and_b32_e32 v131, 56, v131                               // 0000000021f0: 370706b8
	v_lshrrev_b64 v[93:94], v116, v[91:92]                     // 0000000021f4: d73d005d 0202b774
	v_lshlrev_b16 v101.h, 4, v109.l op_sel:[0,0,1]             // 0000000021fc: d7384065 0202da84
	v_and_b16 v101.l, 0x80, v109.l                             // 000000002204: d7620065 0202daff 00000080
	v_lshlrev_b16 v107.l, 4, v109.h op_sel:[0,1,0]             // 000000002210: d738106b 0202da84
	v_and_b16 v107.h, 0x80, v109.h op_sel:[0,1,1]              // 000000002218: d762506b 0202daff 00000080
	v_lshlrev_b16 v108.l, 4, v110.l                            // 000000002224: d738006c 0202dc84
	v_and_b16 v108.h, 0x80, v110.l op_sel:[0,0,1]              // 00000000222c: d762406c 0202dcff 00000080
	v_lshlrev_b16 v109.l, 4, v110.h op_sel:[0,1,0]             // 000000002238: d738106d 0202dc84
	v_and_b16 v109.h, 0x80, v110.h op_sel:[0,1,1]              // 000000002240: d762506d 0202dcff 00000080
	v_lshlrev_b16 v110.l, 4, v112.l                            // 00000000224c: d738006e 0202e084
	v_and_b16 v110.h, 0x80, v112.l op_sel:[0,0,1]              // 000000002254: d762406e 0202e0ff 00000080
	v_and_b32_e32 v122, 56, v122                               // 000000002260: 36f4f4b8
	v_lshlrev_b16 v111.l, 4, v113.l                            // 000000002264: d738006f 0202e284
	v_and_b16 v111.h, 0x80, v113.l op_sel:[0,0,1]              // 00000000226c: d762406f 0202e2ff 00000080
	v_and_b32_e32 v125, 56, v125                               // 000000002278: 36fafab8
	v_lshlrev_b16 v112.l, 4, v114.l                            // 00000000227c: d7380070 0202e484
	v_and_b16 v112.h, 0x80, v114.l op_sel:[0,0,1]              // 000000002284: d7624070 0202e4ff 00000080
	v_lshlrev_b16 v113.l, 4, v115.l                            // 000000002290: d7380071 0202e684
	v_and_b32_e32 v132, 56, v132                               // 000000002298: 370908b8
	v_and_b16 v113.h, 0x80, v115.l op_sel:[0,0,1]              // 00000000229c: d7624071 0202e6ff 00000080
	v_lshrrev_b64 v[94:95], v117, v[91:92]                     // 0000000022a8: d73d005e 0202b775
	v_lshrrev_b64 v[114:115], v126, v[91:92]                   // 0000000022b0: d73d0072 0202b77e
	v_lshrrev_b64 v[95:96], v118, v[91:92]                     // 0000000022b8: d73d005f 0202b776
	v_lshrrev_b64 v[115:116], v127, v[91:92]                   // 0000000022c0: d73d0073 0202b77f
	s_wait_loadcnt 0x1                                         // 0000000022c8: bfc00001
	ds_store_b128 v73, v[97:100] offset:10240                  // 0000000022cc: db7c2800 00006149
	v_lshrrev_b64 v[96:97], v119, v[91:92]                     // 0000000022d4: d73d0060 0202b777
	v_lshrrev_b64 v[116:117], v128, v[91:92]                   // 0000000022dc: d73d0074 0202b780
	s_wait_loadcnt 0x0                                         // 0000000022e4: bfc00000
	ds_store_b128 v73, v[103:106] offset:15360                 // 0000000022e8: db7c3c00 00006749
	v_lshrrev_b64 v[97:98], v120, v[91:92]                     // 0000000022f0: d73d0061 0202b778
	v_lshrrev_b64 v[103:104], v123, v[91:92]                   // 0000000022f8: d73d0067 0202b77b
	v_lshrrev_b64 v[117:118], v129, v[91:92]                   // 000000002300: d73d0075 0202b781
	v_lshrrev_b64 v[98:99], v121, v[91:92]                     // 000000002308: d73d0062 0202b779
	v_lshrrev_b64 v[104:105], v124, v[91:92]                   // 000000002310: d73d0068 0202b77c
	v_lshrrev_b64 v[118:119], v131, v[91:92]                   // 000000002318: d73d0076 0202b783
	v_lshrrev_b64 v[99:100], v122, v[91:92]                    // 000000002320: d73d0063 0202b77a
	v_lshrrev_b64 v[105:106], v125, v[91:92]                   // 000000002328: d73d0069 0202b77d
	v_lshrrev_b64 v[119:120], v132, v[91:92]                   // 000000002330: d73d0077 0202b784
	v_and_b16 v101.h, 0x80, v101.h op_sel:[0,1,1]              // 000000002338: d7625065 0202caff 00000080
	v_and_b16 v107.l, 0x80, v107.l                             // 000000002344: d762006b 0202d6ff 00000080
	v_and_b16 v108.l, 0x80, v108.l                             // 000000002350: d762006c 0202d8ff 00000080
	v_and_b16 v109.l, 0x80, v109.l                             // 00000000235c: d762006d 0202daff 00000080
	v_and_b16 v89.l, 0x80, v110.l                              // 000000002368: d7620059 0202dcff 00000080
	v_and_b16 v89.h, 0x80, v111.l op_sel:[0,0,1]               // 000000002374: d7624059 0202deff 00000080
	v_and_b16 v90.l, 0x80, v112.l                              // 000000002380: d762005a 0202e0ff 00000080
	v_and_b16 v90.h, 0x80, v113.l op_sel:[0,0,1]               // 00000000238c: d762405a 0202e2ff 00000080
	v_or_b16 v91.l, v101.h, v93.l op_sel:[1,0,0]               // 000000002398: d763085b 0202bb65
	v_or_b16 v91.h, v101.l, v94.l op_sel:[0,0,1]               // 0000000023a0: d763405b 0202bd65
	v_or_b16 v89.l, v89.l, v95.l                               // 0000000023a8: d7630059 0202bf59
	v_or_b16 v92.l, v110.h, v96.l op_sel:[1,0,0]               // 0000000023b0: d763085c 0202c16e
	v_or_b16 v92.h, v107.l, v97.l op_sel:[0,0,1]               // 0000000023b8: d763405c 0202c36b
	v_or_b16 v93.l, v107.h, v98.l op_sel:[1,0,0]               // 0000000023c0: d763085d 0202c56b
	v_or_b16 v89.h, v89.h, v99.l op_sel:[1,0,1]                // 0000000023c8: d7634859 0202c759
	v_or_b16 v93.h, v111.h, v103.l op_sel:[1,0,1]              // 0000000023d0: d763485d 0202cf6f
	v_or_b16 v94.l, v108.l, v104.l                             // 0000000023d8: d763005e 0202d16c
	v_or_b16 v94.h, v108.h, v105.l op_sel:[1,0,1]              // 0000000023e0: d763485e 0202d36c
	v_or_b16 v90.l, v90.l, v114.l                              // 0000000023e8: d763005a 0202e55a
	v_or_b16 v95.l, v112.h, v115.l op_sel:[1,0,0]              // 0000000023f0: d763085f 0202e770
	v_or_b16 v95.h, v109.l, v116.l op_sel:[0,0,1]              // 0000000023f8: d763405f 0202e96d
	v_or_b16 v96.l, v109.h, v117.l op_sel:[1,0,0]              // 000000002400: d7630860 0202eb6d
	v_or_b16 v90.h, v90.h, v118.l op_sel:[1,0,1]               // 000000002408: d763485a 0202ed5a
	v_or_b16 v96.h, v113.h, v119.l op_sel:[1,0,1]              // 000000002410: d7634860 0202ef71
	v_cndmask_b16 v91.l, v91.l, 0, s2                          // 000000002418: d65d005b 0009015b
	v_cndmask_b16 v91.h, v91.h, 0, s2                          // 000000002420: d65d485b 0009015b
	v_cndmask_b16 v89.l, v89.l, 0, s2                          // 000000002428: d65d0059 00090159
	v_cndmask_b16 v92.l, v92.l, 0, s2                          // 000000002430: d65d005c 0009015c
	v_cndmask_b16 v92.h, v92.h, 0, s2                          // 000000002438: d65d485c 0009015c
	v_cndmask_b16 v93.l, v93.l, 0, s2                          // 000000002440: d65d005d 0009015d
	v_cndmask_b16 v89.h, v89.h, 0, s2                          // 000000002448: d65d4859 00090159
	v_cndmask_b16 v93.h, v93.h, 0, s2                          // 000000002450: d65d485d 0009015d
	v_cndmask_b16 v94.l, v94.l, 0, s2                          // 000000002458: d65d005e 0009015e
	v_cndmask_b16 v94.h, v94.h, 0, s2                          // 000000002460: d65d485e 0009015e
	v_cndmask_b16 v90.l, v90.l, 0, s2                          // 000000002468: d65d005a 0009015a
	v_cndmask_b16 v96.h, v96.h, 0, s2                          // 000000002470: d65d4860 00090160
	v_cndmask_b16 v90.h, v90.h, 0, s2                          // 000000002478: d65d485a 0009015a
	v_cndmask_b16 v96.l, v96.l, 0, s2                          // 000000002480: d65d0060 00090160
	v_cndmask_b16 v95.h, v95.h, 0, s2                          // 000000002488: d65d485f 0009015f
	v_cndmask_b16 v95.l, v95.l, 0, s2                          // 000000002490: d65d005f 0009015f
	v_lshlrev_b16 v96.h, 8, v96.h op_sel:[0,1,1]               // 000000002498: d7385060 0202c088
	v_and_b16 v90.h, 0xff, v90.h op_sel:[0,1,1]                // 0000000024a0: d762505a 0202b4ff 000000ff
	v_lshlrev_b16 v96.l, 8, v96.l                              // 0000000024ac: d7380060 0202c088
	v_and_b16 v95.h, 0xff, v95.h op_sel:[0,1,1]                // 0000000024b4: d762505f 0202beff 000000ff
	v_lshlrev_b16 v95.l, 8, v95.l                              // 0000000024c0: d738005f 0202be88
	v_and_b16 v90.l, 0xff, v90.l                               // 0000000024c8: d762005a 0202b4ff 000000ff
	v_lshlrev_b16 v94.h, 8, v94.h op_sel:[0,1,1]               // 0000000024d4: d738505e 0202bc88
	v_and_b16 v94.l, 0xff, v94.l                               // 0000000024dc: d762005e 0202bcff 000000ff
	v_lshlrev_b16 v93.h, 8, v93.h op_sel:[0,1,1]               // 0000000024e8: d738505d 0202ba88
	v_and_b16 v89.h, 0xff, v89.h op_sel:[0,1,1]                // 0000000024f0: d7625059 0202b2ff 000000ff
	v_lshlrev_b16 v93.l, 8, v93.l                              // 0000000024fc: d738005d 0202ba88
	v_and_b16 v97.l, 0xff, v92.h op_sel:[0,1,0]                // 000000002504: d7621061 0202b8ff 000000ff
	v_lshlrev_b16 v97.h, 8, v92.l op_sel:[0,0,1]               // 000000002510: d7384061 0202b888
	v_and_b16 v89.l, 0xff, v89.l                               // 000000002518: d7620059 0202b2ff 000000ff
	v_lshlrev_b16 v98.l, 8, v91.h op_sel:[0,1,0]               // 000000002524: d7381062 0202b688
	v_and_b16 v98.h, 0xff, v91.l op_sel:[0,0,1]                // 00000000252c: d7624062 0202b6ff 000000ff
	v_or_b16 v92.h, v90.h, v96.h op_sel:[1,1,1]                // 000000002538: d763585c 0202c15a
	v_or_b16 v92.l, v95.h, v96.l op_sel:[1,0,0]                // 000000002540: d763085c 0202c15f
	v_or_b16 v91.h, v90.l, v95.l op_sel:[0,0,1]                // 000000002548: d763405b 0202bf5a
	v_or_b16 v91.l, v94.l, v94.h op_sel:[0,1,0]                // 000000002550: d763105b 0202bd5e
	v_or_b16 v90.h, v89.h, v93.h op_sel:[1,1,1]                // 000000002558: d763585a 0202bb59
	v_or_b16 v90.l, v97.l, v93.l                               // 000000002560: d763005a 0202bb61
	v_or_b16 v89.h, v89.l, v97.h op_sel:[0,1,1]                // 000000002568: d7635059 0202c359
	v_or_b16 v89.l, v98.h, v98.l op_sel:[1,0,0]                // 000000002570: d7630859 0202c562
	s_cmp_lg_u64 s[26:27], 0x400                               // 000000002578: bf11ff1a 00000400
	ds_store_b128 v73, v[89:92] offset:20480                   // 000000002580: db7c5000 00005949
	s_wait_dscnt 0x0                                           // 000000002588: bfc60000
	s_barrier_signal -1                                        // 00000000258c: be804ec1
	s_barrier_wait 0xffff                                      // 000000002590: bf94ffff
	ds_load_2addr_b64 v[89:92], v74 offset1:2                  // 000000002594: d9dc0200 5900004a
	ds_load_2addr_b64 v[93:96], v87 offset1:2                  // 00000000259c: d9dc0200 5d000057
	ds_load_2addr_b64 v[97:100], v88 offset1:2                 // 0000000025a4: d9dc0200 61000058
	ds_load_2addr_b64 v[103:106], v75 offset1:2                // 0000000025ac: d9dc0200 6700004b
	ds_load_2addr_b64 v[107:110], v76 offset1:2                // 0000000025b4: d9dc0200 6b00004c
	ds_load_2addr_b64 v[111:114], v77 offset1:2                // 0000000025bc: d9dc0200 6f00004d
	ds_load_2addr_b64 v[115:118], v74 offset0:4 offset1:6      // 0000000025c4: d9dc0604 7300004a
	ds_load_2addr_b64 v[119:122], v87 offset0:4 offset1:6      // 0000000025cc: d9dc0604 77000057
	ds_load_2addr_b64 v[123:126], v88 offset0:4 offset1:6      // 0000000025d4: d9dc0604 7b000058
	ds_load_2addr_b64 v[131:134], v75 offset0:4 offset1:6      // 0000000025dc: d9dc0604 8300004b
	ds_load_2addr_b64 v[139:142], v76 offset0:4 offset1:6      // 0000000025e4: d9dc0604 8b00004c
	ds_load_2addr_b64 v[143:146], v77 offset0:4 offset1:6      // 0000000025ec: d9dc0604 8f00004d
	s_wait_dscnt 0xa                                           // 0000000025f4: bfc6000a
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[89:90], v[93:94], v[56:63]// 0000000025f8: cc464038 1ce2bb59
	s_wait_dscnt 0x9                                           // 000000002600: bfc60009
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[89:90], v[97:98], v[24:31]// 000000002604: cc464018 1c62c359
	s_wait_dscnt 0x8                                           // 00000000260c: bfc60008
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[103:104], v[93:94], v[48:55]// 000000002610: cc464030 1cc2bb67
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[103:104], v[97:98], v[16:23]// 000000002618: cc464010 1c42c367
	s_wait_dscnt 0x7                                           // 000000002620: bfc60007
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[107:108], v[93:94], v[40:47]// 000000002624: cc464028 1ca2bb6b
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[107:108], v[97:98], v[8:15]// 00000000262c: cc464008 1c22c36b
	s_wait_dscnt 0x6                                           // 000000002634: bfc60006
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[111:112], v[93:94], v[32:39]// 000000002638: cc464020 1c82bb6f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[111:112], v[97:98], v[0:7]// 000000002640: cc464000 1c02c36f
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[91:92], v[95:96], v[56:63]// 000000002648: cc464038 1ce2bf5b
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[91:92], v[99:100], v[24:31]// 000000002650: cc464018 1c62c75b
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[105:106], v[95:96], v[48:55]// 000000002658: cc464030 1cc2bf69
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[105:106], v[99:100], v[16:23]// 000000002660: cc464010 1c42c769
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[109:110], v[95:96], v[40:47]// 000000002668: cc464028 1ca2bf6d
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[109:110], v[99:100], v[8:15]// 000000002670: cc464008 1c22c76d
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[113:114], v[95:96], v[32:39]// 000000002678: cc464020 1c82bf71
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[113:114], v[99:100], v[0:7]// 000000002680: cc464000 1c02c771
	s_wait_dscnt 0x4                                           // 000000002688: bfc60004
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[115:116], v[119:120], v[56:63]// 00000000268c: cc464038 1ce2ef73
	s_wait_dscnt 0x3                                           // 000000002694: bfc60003
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[115:116], v[123:124], v[24:31]// 000000002698: cc464018 1c62f773
	s_wait_dscnt 0x2                                           // 0000000026a0: bfc60002
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[131:132], v[119:120], v[48:55]// 0000000026a4: cc464030 1cc2ef83
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[131:132], v[123:124], v[16:23]// 0000000026ac: cc464010 1c42f783
	s_wait_dscnt 0x1                                           // 0000000026b4: bfc60001
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[139:140], v[119:120], v[40:47]// 0000000026b8: cc464028 1ca2ef8b
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[139:140], v[123:124], v[8:15]// 0000000026c0: cc464008 1c22f78b
	s_wait_dscnt 0x0                                           // 0000000026c8: bfc60000
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[143:144], v[119:120], v[32:39]// 0000000026cc: cc464020 1c82ef8f
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[143:144], v[123:124], v[0:7]// 0000000026d4: cc464000 1c02f78f
	v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[117:118], v[121:122], v[56:63]// 0000000026dc: cc464038 1ce2f375
	v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[117:118], v[125:126], v[24:31]// 0000000026e4: cc464018 1c62fb75
	v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[133:134], v[121:122], v[48:55]// 0000000026ec: cc464030 1cc2f385
	v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[133:134], v[125:126], v[16:23]// 0000000026f4: cc464010 1c42fb85
	v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[141:142], v[121:122], v[40:47]// 0000000026fc: cc464028 1ca2f38d
	v_wmma_f32_16x16x16_fp8_fp8 v[8:15], v[141:142], v[125:126], v[8:15]// 000000002704: cc464008 1c22fb8d
	v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[145:146], v[121:122], v[32:39]// 00000000270c: cc464020 1c82f391
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[145:146], v[125:126], v[0:7]// 000000002714: cc464000 1c02fb91
	s_cbranch_scc1 64997                                       // 00000000271c: bfa2fde5 <packed_folded_w4a8+0x3b4>
	v_or_b32_e32 v74, s22, v71                                 // 000000002720: 38948e16
	v_or_b32_e32 v64, s24, v70                                 // 000000002724: 38808c18
	v_mov_b32_e32 v79, s23                                     // 000000002728: 7e9e0217
	s_load_b64 s[4:5], s[0:1], 0x58                            // 00000000272c: f4002100 f8000058
	v_mov_b32_e32 v77, s25                                     // 000000002734: 7e9a0219
	v_or_b32_e32 v78, v138, v74                                // 000000002738: 389c958a
	v_or_b32_e32 v76, v64, v72                                 // 00000000273c: 38989140
	v_mov_b32_e32 v75, s23                                     // 000000002740: 7e960217
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002744: bf870193
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[78:79]                // 000000002748: 7ca89c10
	v_cmp_gt_i64_e64 s2, s[18:19], v[76:77]                    // 00000000274c: d4540002 02029812
	s_wait_alu depctr_va_vcc(0)                                // 000000002754: bf88ff9d
	v_dual_cndmask_b32 v66, 0, v79 :: v_dual_cndmask_b32 v65, 0, v78// 000000002758: ca529e80 42409c80
	s_wait_alu depctr_va_sdst(0)                               // 000000002760: bf88f19f
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_3)// 000000002764: bf8701a2
	v_cndmask_b32_e64 v69, 0, v76, s2                          // 000000002768: d5010045 000a9880
	v_cndmask_b32_e64 v68, 0, v77, s2                          // 000000002770: d5010044 000a9a80
	v_lshlrev_b64_e32 v[66:67], 2, v[65:66]                    // 000000002778: 3e848282
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 00000000277c: bf8701a3
	v_add_co_u32 v124, vcc_lo, s14, v69                        // 000000002780: d7006a7c 02028a0e
	s_wait_alu depctr_va_vcc(0)                                // 000000002788: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s15, v68, vcc_lo           // 00000000278c: d5207c7d 01aa880f
	s_wait_kmcnt 0x0                                           // 000000002794: bfc70000
	s_delay_alu instid0(valu_dep_3)                            // 000000002798: bf870003
	v_add_co_u32 v66, vcc_lo, s4, v66                          // 00000000279c: d7006a42 02028404
	s_wait_alu depctr_va_vcc(0)                                // 0000000027a4: bf88ff9d
	v_add_co_ci_u32_e64 v67, null, s5, v67, vcc_lo             // 0000000027a8: d5207c43 01aa8605
	global_load_u8 v65, v[124:125], off                        // 0000000027b0: ee04007c 00000041 0000007c
	global_load_b32 v68, v[66:67], off                         // 0000000027bc: ee05007c 00000044 00000042
	s_wait_loadcnt 0x1                                         // 0000000027c8: bfc00001
	v_lshlrev_b32_e32 v83, 23, v65                             // 0000000027cc: 30a68297
	v_mov_b32_e32 v65, s25                                     // 0000000027d0: 7e820219
	s_wait_loadcnt 0x0                                         // 0000000027d4: bfc00000
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 0000000027d8: bf870092
	v_mul_f32_e32 v69, v68, v83                                // 0000000027dc: 108aa744
	v_cmp_class_f32_e64 s2, v69, 0x198                         // 0000000027e0: d47e0002 0201ff45 00000198
	v_mul_f32_e32 v81, v56, v69                                // 0000000027ec: 10a28b38
	s_xor_b32 s3, s2, -1                                       // 0000000027f0: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027f4: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 0000000027f8: be822003
	s_cbranch_execnz 2729                                      // 0000000027fc: bfa60aa9 <packed_folded_w4a8+0x37a4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002800: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002804: 8c7e027e
	v_or_b32_e32 v144, 1, v138                                 // 000000002808: 39211481
	v_mov_b32_e32 v69, v75                                     // 00000000280c: 7e8a034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002810: bf870092
	v_or_b32_e32 v68, v144, v74                                // 000000002814: 38889590
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[68:69]                // 000000002818: 7ca88810
	s_wait_alu depctr_va_vcc(0)                                // 00000000281c: bf88ff9d
	v_dual_cndmask_b32 v69, 0, v69 :: v_dual_cndmask_b32 v68, 0, v68// 000000002820: ca528a80 45448880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002828: bf870091
	v_lshlrev_b64_e32 v[68:69], 2, v[68:69]                    // 00000000282c: 3e888882
	v_add_co_u32 v68, vcc_lo, s4, v68                          // 000000002830: d7006a44 02028804
	s_wait_alu depctr_va_vcc(0)                                // 000000002838: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 00000000283c: bf8700c2
	v_add_co_ci_u32_e64 v69, null, s5, v69, vcc_lo             // 000000002840: d5207c45 01aa8a05
	global_load_b32 v56, v[68:69], off                         // 000000002848: ee05007c 00000038 00000044
	s_wait_loadcnt 0x0                                         // 000000002854: bfc00000
	v_mul_f32_e32 v70, v56, v83                                // 000000002858: 108ca738
	v_cmp_class_f32_e64 s2, v70, 0x198                         // 00000000285c: d47e0002 0201ff46 00000198
	v_mul_f32_e32 v82, v57, v70                                // 000000002868: 10a48d39
	s_xor_b32 s3, s2, -1                                       // 00000000286c: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002870: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 000000002874: be822003
	s_cbranch_execnz 2715                                      // 000000002878: bfa60a9b <packed_folded_w4a8+0x37e8>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000287c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002880: 8c7e027e
	v_or_b32_e32 v145, 2, v138                                 // 000000002884: 39231482
	v_mov_b32_e32 v57, v75                                     // 000000002888: 7e72034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 00000000288c: bf870092
	v_or_b32_e32 v56, v145, v74                                // 000000002890: 38709591
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[56:57]                // 000000002894: 7ca87010
	s_wait_alu depctr_va_vcc(0)                                // 000000002898: bf88ff9d
	v_dual_cndmask_b32 v57, 0, v57 :: v_dual_cndmask_b32 v56, 0, v56// 00000000289c: ca527280 39387080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000028a4: bf870091
	v_lshlrev_b64_e32 v[56:57], 2, v[56:57]                    // 0000000028a8: 3e707082
	v_add_co_u32 v56, vcc_lo, s4, v56                          // 0000000028ac: d7006a38 02027004
	s_wait_alu depctr_va_vcc(0)                                // 0000000028b4: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000028b8: bf8700c2
	v_add_co_ci_u32_e64 v57, null, s5, v57, vcc_lo             // 0000000028bc: d5207c39 01aa7205
	global_load_b32 v70, v[56:57], off                         // 0000000028c4: ee05007c 00000046 00000038
	s_wait_loadcnt 0x0                                         // 0000000028d0: bfc00000
	v_mul_f32_e32 v71, v70, v83                                // 0000000028d4: 108ea746
	v_cmp_class_f32_e64 s2, v71, 0x198                         // 0000000028d8: d47e0002 0201ff47 00000198
	v_mul_f32_e32 v84, v58, v71                                // 0000000028e4: 10a88f3a
	s_xor_b32 s3, s2, -1                                       // 0000000028e8: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028ec: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 0000000028f0: be822003
	s_cbranch_execnz 2701                                      // 0000000028f4: bfa60a8d <packed_folded_w4a8+0x382c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000028f8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000028fc: 8c7e027e
	v_or_b32_e32 v146, 3, v138                                 // 000000002900: 39251483
	v_mov_b32_e32 v71, v75                                     // 000000002904: 7e8e034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002908: bf870092
	v_or_b32_e32 v70, v146, v74                                // 00000000290c: 388c9592
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[70:71]                // 000000002910: 7ca88c10
	s_wait_alu depctr_va_vcc(0)                                // 000000002914: bf88ff9d
	v_dual_cndmask_b32 v71, 0, v71 :: v_dual_cndmask_b32 v70, 0, v70// 000000002918: ca528e80 47468c80
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002920: bf870091
	v_lshlrev_b64_e32 v[70:71], 2, v[70:71]                    // 000000002924: 3e8c8c82
	v_add_co_u32 v70, vcc_lo, s4, v70                          // 000000002928: d7006a46 02028c04
	s_wait_alu depctr_va_vcc(0)                                // 000000002930: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002934: bf8700c2
	v_add_co_ci_u32_e64 v71, null, s5, v71, vcc_lo             // 000000002938: d5207c47 01aa8e05
	global_load_b32 v58, v[70:71], off                         // 000000002940: ee05007c 0000003a 00000046
	s_wait_loadcnt 0x0                                         // 00000000294c: bfc00000
	v_mul_f32_e32 v72, v58, v83                                // 000000002950: 1090a73a
	v_cmp_class_f32_e64 s2, v72, 0x198                         // 000000002954: d47e0002 0201ff48 00000198
	v_mul_f32_e32 v85, v59, v72                                // 000000002960: 10aa913b
	s_xor_b32 s3, s2, -1                                       // 000000002964: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002968: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 00000000296c: be822003
	s_cbranch_execnz 2687                                      // 000000002970: bfa60a7f <packed_folded_w4a8+0x3870>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002974: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002978: 8c7e027e
	v_or_b32_e32 v147, 4, v138                                 // 00000000297c: 39271484
	v_mov_b32_e32 v59, v75                                     // 000000002980: 7e76034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002984: bf870092
	v_or_b32_e32 v58, v147, v74                                // 000000002988: 38749593
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[58:59]                // 00000000298c: 7ca87410
	s_wait_alu depctr_va_vcc(0)                                // 000000002990: bf88ff9d
	v_dual_cndmask_b32 v59, 0, v59 :: v_dual_cndmask_b32 v58, 0, v58// 000000002994: ca527680 3b3a7480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000299c: bf870091
	v_lshlrev_b64_e32 v[58:59], 2, v[58:59]                    // 0000000029a0: 3e747482
	v_add_co_u32 v58, vcc_lo, s4, v58                          // 0000000029a4: d7006a3a 02027404
	s_wait_alu depctr_va_vcc(0)                                // 0000000029ac: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 0000000029b0: bf8700c2
	v_add_co_ci_u32_e64 v59, null, s5, v59, vcc_lo             // 0000000029b4: d5207c3b 01aa7605
	global_load_b32 v72, v[58:59], off                         // 0000000029bc: ee05007c 00000048 0000003a
	s_wait_loadcnt 0x0                                         // 0000000029c8: bfc00000
	v_mul_f32_e32 v73, v72, v83                                // 0000000029cc: 1092a748
	v_cmp_class_f32_e64 s2, v73, 0x198                         // 0000000029d0: d47e0002 0201ff49 00000198
	v_mul_f32_e32 v86, v60, v73                                // 0000000029dc: 10ac933c
	s_xor_b32 s3, s2, -1                                       // 0000000029e0: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029e4: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 0000000029e8: be822003
	s_cbranch_execnz 2673                                      // 0000000029ec: bfa60a71 <packed_folded_w4a8+0x38b4>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029f0: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000029f4: 8c7e027e
	v_or_b32_e32 v152, 5, v138                                 // 0000000029f8: 39311485
	v_mov_b32_e32 v73, v75                                     // 0000000029fc: 7e92034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002a00: bf870092
	v_or_b32_e32 v72, v152, v74                                // 000000002a04: 38909598
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[72:73]                // 000000002a08: 7ca89010
	s_wait_alu depctr_va_vcc(0)                                // 000000002a0c: bf88ff9d
	v_dual_cndmask_b32 v73, 0, v73 :: v_dual_cndmask_b32 v72, 0, v72// 000000002a10: ca529280 49489080
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002a18: bf870091
	v_lshlrev_b64_e32 v[72:73], 2, v[72:73]                    // 000000002a1c: 3e909082
	v_add_co_u32 v72, vcc_lo, s4, v72                          // 000000002a20: d7006a48 02029004
	s_wait_alu depctr_va_vcc(0)                                // 000000002a28: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002a2c: bf8700c2
	v_add_co_ci_u32_e64 v73, null, s5, v73, vcc_lo             // 000000002a30: d5207c49 01aa9205
	global_load_b32 v60, v[72:73], off                         // 000000002a38: ee05007c 0000003c 00000048
	s_wait_loadcnt 0x0                                         // 000000002a44: bfc00000
	v_mul_f32_e32 v87, v60, v83                                // 000000002a48: 10aea73c
	v_cmp_class_f32_e64 s2, v87, 0x198                         // 000000002a4c: d47e0002 0201ff57 00000198
	v_mul_f32_e32 v87, v61, v87                                // 000000002a58: 10aeaf3d
	s_xor_b32 s3, s2, -1                                       // 000000002a5c: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a60: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 000000002a64: be822003
	s_cbranch_execnz 2659                                      // 000000002a68: bfa60a63 <packed_folded_w4a8+0x38f8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002a6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002a70: 8c7e027e
	v_or_b32_e32 v153, 6, v138                                 // 000000002a74: 39331486
	v_mov_b32_e32 v61, v75                                     // 000000002a78: 7e7a034b
	s_delay_alu instid0(valu_dep_2) | instskip(next) | instid1(valu_dep_1)// 000000002a7c: bf870092
	v_or_b32_e32 v60, v153, v74                                // 000000002a80: 38789599
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[60:61]                // 000000002a84: 7ca87810
	s_wait_alu depctr_va_vcc(0)                                // 000000002a88: bf88ff9d
	v_dual_cndmask_b32 v61, 0, v61 :: v_dual_cndmask_b32 v60, 0, v60// 000000002a8c: ca527a80 3d3c7880
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002a94: bf870091
	v_lshlrev_b64_e32 v[60:61], 2, v[60:61]                    // 000000002a98: 3e787882
	v_add_co_u32 v60, vcc_lo, s4, v60                          // 000000002a9c: d7006a3c 02027804
	s_wait_alu depctr_va_vcc(0)                                // 000000002aa4: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002aa8: bf8700c2
	v_add_co_ci_u32_e64 v61, null, s5, v61, vcc_lo             // 000000002aac: d5207c3d 01aa7a05
	global_load_b32 v89, v[60:61], off                         // 000000002ab4: ee05007c 00000059 0000003c
	s_wait_loadcnt 0x0                                         // 000000002ac0: bfc00000
	v_mul_f32_e32 v88, v89, v83                                // 000000002ac4: 10b0a759
	v_cmp_class_f32_e64 s2, v88, 0x198                         // 000000002ac8: d47e0002 0201ff58 00000198
	v_mul_f32_e32 v88, v62, v88                                // 000000002ad4: 10b0b13e
	s_xor_b32 s3, s2, -1                                       // 000000002ad8: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002adc: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 000000002ae0: be822003
	s_cbranch_execnz 2645                                      // 000000002ae4: bfa60a55 <packed_folded_w4a8+0x393c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002ae8: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002aec: 8c7e027e
	v_or_b32_e32 v154, 7, v138                                 // 000000002af0: 39351487
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002af4: bf870091
	v_or_b32_e32 v74, v154, v74                                // 000000002af8: 3894959a
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[74:75]                // 000000002afc: 7ca89410
	s_wait_alu depctr_va_vcc(0)                                // 000000002b00: bf88ff9d
	v_dual_cndmask_b32 v75, 0, v75 :: v_dual_cndmask_b32 v74, 0, v74// 000000002b04: ca529680 4b4a9480
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002b0c: bf870091
	v_lshlrev_b64_e32 v[74:75], 2, v[74:75]                    // 000000002b10: 3e949482
	v_add_co_u32 v74, vcc_lo, s4, v74                          // 000000002b14: d7006a4a 02029404
	s_wait_alu depctr_va_vcc(0)                                // 000000002b1c: bf88ff9d
	s_delay_alu instid0(valu_dep_2) | instskip(skip_3) | instid1(valu_dep_1)// 000000002b20: bf8700c2
	v_add_co_ci_u32_e64 v75, null, s5, v75, vcc_lo             // 000000002b24: d5207c4b 01aa9605
	global_load_b32 v62, v[74:75], off                         // 000000002b2c: ee05007c 0000003e 0000004a
	s_wait_loadcnt 0x0                                         // 000000002b38: bfc00000
	v_mul_f32_e32 v89, v62, v83                                // 000000002b3c: 10b2a73e
	v_cmp_class_f32_e64 s2, v89, 0x198                         // 000000002b40: d47e0002 0201ff59 00000198
	v_mul_f32_e32 v89, v63, v89                                // 000000002b4c: 10b2b33f
	s_xor_b32 s3, s2, -1                                       // 000000002b50: 8d03c102
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b54: bf88ff9e
	s_and_saveexec_b32 s2, s3                                  // 000000002b58: be822003
	s_cbranch_execnz 2632                                      // 000000002b5c: bfa60a48 <packed_folded_w4a8+0x3980>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002b60: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s2                              // 000000002b64: 8c7e027e
	s_load_b64 s[2:3], s[0:1], 0xa8                            // 000000002b68: f4002080 f80000a8
	v_mul_lo_u32 v83, s19, v78                                 // 000000002b70: d72c0053 02029c13
	v_mul_lo_u32 v79, s18, v79                                 // 000000002b78: d72c004f 02029e12
	v_mad_co_u64_u32 v[62:63], null, s18, v78, 0               // 000000002b80: d6fe7c3e 02029c12
	v_bfe_u32 v78, v81, 16, 1                                  // 000000002b88: d610004e 02052151
	v_lshlrev_b64_e32 v[110:111], 1, v[76:77]                  // 000000002b90: 3edc9881
	v_or_b32_e32 v76, 0x400000, v82                            // 000000002b94: 3898a4ff 00400000
	v_or_b32_e32 v98, s22, v80                                 // 000000002b9c: 38c4a016
	v_bfe_u32 v91, v86, 16, 1                                  // 000000002ba0: d610005b 02052156
	v_add3_u32 v77, v78, v81, 0x7fff                           // 000000002ba8: d655004d 03fea34e 00007fff
	v_or_b32_e32 v92, 0x400000, v86                            // 000000002bb4: 38b8acff 00400000
	v_add3_u32 v63, v63, v79, v83                              // 000000002bbc: d655003f 054e9f3f
	v_bfe_u32 v83, v82, 16, 1                                  // 000000002bc4: d6100053 02052152
	v_or_b32_e32 v79, 0x400000, v81                            // 000000002bcc: 389ea2ff 00400000
	v_add3_u32 v91, v91, v86, 0x7fff                           // 000000002bd4: d655005b 03fead5b 00007fff
	v_or_b32_e32 v95, 0x400000, v89                            // 000000002be0: 38beb2ff 00400000
	v_lshlrev_b64_e32 v[62:63], 1, v[62:63]                    // 000000002be8: 3e7c7c81
	v_add3_u32 v78, v83, v82, 0x7fff                           // 000000002bec: d655004e 03fea553 00007fff
	v_mov_b32_e32 v101, s23                                    // 000000002bf8: 7eca0217
	v_or_b32_e32 v100, v98, v138                               // 000000002bfc: 38c91562
	v_mov_b32_e32 v99, s23                                     // 000000002c00: 7ec60217
	s_wait_kmcnt 0x0                                           // 000000002c04: bfc70000
	v_add_co_u32 v83, vcc_lo, s2, v62                          // 000000002c08: d7006a53 02027c02
	s_wait_alu depctr_va_vcc(0)                                // 000000002c10: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s3, v63, vcc_lo             // 000000002c14: d5207c5a 01aa7e03
	v_cmp_u_f32_e32 vcc_lo, v81, v81                           // 000000002c1c: 7c30a351
	s_wait_alu depctr_va_vcc(0)                                // 000000002c20: bf88ff9d
	v_cndmask_b32_e32 v77, v77, v79, vcc_lo                    // 000000002c24: 029a9f4d
	v_add_co_u32 v62, vcc_lo, v83, v110                        // 000000002c28: d7006a3e 0202dd53
	s_wait_alu depctr_va_vcc(0)                                // 000000002c30: bf88ff9d
	v_add_co_ci_u32_e64 v63, null, v90, v111, vcc_lo           // 000000002c34: d5207c3f 01aadf5a
	v_cmp_u_f32_e32 vcc_lo, v82, v82                           // 000000002c3c: 7c30a552
	v_bfe_u32 v79, v84, 16, 1                                  // 000000002c40: d610004f 02052154
	v_or_b32_e32 v82, 0x400000, v84                            // 000000002c48: 38a4a8ff 00400000
	global_store_d16_hi_b16 v[62:63], v77, off                 // 000000002c50: ee09407c 26800000 0000003e
	s_wait_alu depctr_va_vcc(0)                                // 000000002c5c: bf88ff9d
	v_cndmask_b32_e32 v80, v78, v76, vcc_lo                    // 000000002c60: 02a0994e
	v_add_co_u32 v76, vcc_lo, v83, s20                         // 000000002c64: d7006a4c 02002953
	s_wait_alu depctr_va_vcc(0)                                // 000000002c6c: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s21, v90, vcc_lo            // 000000002c70: d5207c4d 01aab415
	v_add3_u32 v81, v79, v84, 0x7fff                           // 000000002c78: d6550051 03fea94f 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000002c84: bf8701a3
	v_add_co_u32 v78, vcc_lo, v76, v110                        // 000000002c88: d7006a4e 0202dd4c
	s_wait_alu depctr_va_vcc(0)                                // 000000002c90: bf88ff9d
	v_add_co_ci_u32_e64 v79, null, v77, v111, vcc_lo           // 000000002c94: d5207c4f 01aadf4d
	v_cmp_u_f32_e32 vcc_lo, v84, v84                           // 000000002c9c: 7c30a954
	v_or_b32_e32 v90, 0x400000, v85                            // 000000002ca0: 38b4aaff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002ca8: bf88ff9d
	v_cndmask_b32_e32 v81, v81, v82, vcc_lo                    // 000000002cac: 02a2a551
	v_bfe_u32 v82, v85, 16, 1                                  // 000000002cb0: d6100052 02052155
	v_add_co_u32 v76, vcc_lo, v76, s20                         // 000000002cb8: d7006a4c 0200294c
	s_wait_alu depctr_va_vcc(0)                                // 000000002cc0: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, s21, v77, vcc_lo            // 000000002cc4: d5207c4d 01aa9a15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002ccc: bf870193
	v_add3_u32 v84, v82, v85, 0x7fff                           // 000000002cd0: d6550054 03feab52 00007fff
	v_add_co_u32 v82, vcc_lo, v76, v110                        // 000000002cdc: d7006a52 0202dd4c
	s_wait_alu depctr_va_vcc(0)                                // 000000002ce4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002ce8: bf870003
	v_add_co_ci_u32_e64 v83, null, v77, v111, vcc_lo           // 000000002cec: d5207c53 01aadf4d
	v_cmp_u_f32_e32 vcc_lo, v85, v85                           // 000000002cf4: 7c30ab55
	s_wait_alu depctr_va_vcc(0)                                // 000000002cf8: bf88ff9d
	v_cndmask_b32_e32 v84, v84, v90, vcc_lo                    // 000000002cfc: 02a8b554
	v_add_co_u32 v85, vcc_lo, v76, s20                         // 000000002d00: d7006a55 0200294c
	s_wait_alu depctr_va_vcc(0)                                // 000000002d08: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s21, v77, vcc_lo            // 000000002d0c: d5207c5a 01aa9a15
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000002d14: bf870122
	v_add_co_u32 v76, vcc_lo, v85, v110                        // 000000002d18: d7006a4c 0202dd55
	s_wait_alu depctr_va_vcc(0)                                // 000000002d20: bf88ff9d
	v_add_co_ci_u32_e64 v77, null, v90, v111, vcc_lo           // 000000002d24: d5207c4d 01aadf5a
	v_cmp_u_f32_e32 vcc_lo, v86, v86                           // 000000002d2c: 7c30ad56
	s_clause 0x2                                               // 000000002d30: bf850002
	global_store_d16_hi_b16 v[78:79], v80, off                 // 000000002d34: ee09407c 28000000 0000004e
	global_store_d16_hi_b16 v[82:83], v81, off                 // 000000002d40: ee09407c 28800000 00000052
	global_store_d16_hi_b16 v[76:77], v84, off                 // 000000002d4c: ee09407c 2a000000 0000004c
	v_bfe_u32 v80, v87, 16, 1                                  // 000000002d58: d6100050 02052157
	s_wait_alu depctr_va_vcc(0)                                // 000000002d60: bf88ff9d
	v_cndmask_b32_e32 v92, v91, v92, vcc_lo                    // 000000002d64: 02b8b95b
	v_add_co_u32 v84, vcc_lo, v85, s20                         // 000000002d68: d7006a54 02002955
	s_wait_alu depctr_va_vcc(0)                                // 000000002d70: bf88ff9d
	v_add_co_ci_u32_e64 v85, null, s21, v90, vcc_lo            // 000000002d74: d5207c55 01aab415
	v_add3_u32 v86, v80, v87, 0x7fff                           // 000000002d7c: d6550056 03feaf50 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000002d88: bf870003
	v_add_co_u32 v80, vcc_lo, v84, v110                        // 000000002d8c: d7006a50 0202dd54
	v_or_b32_e32 v90, 0x400000, v87                            // 000000002d94: 38b4aeff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002d9c: bf88ff9d
	v_add_co_ci_u32_e64 v81, null, v85, v111, vcc_lo           // 000000002da0: d5207c51 01aadf55
	v_cmp_u_f32_e32 vcc_lo, v87, v87                           // 000000002da8: 7c30af57
	v_or_b32_e32 v91, 0x400000, v88                            // 000000002dac: 38b6b0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000002db4: bf88ff9d
	v_cndmask_b32_e32 v93, v86, v90, vcc_lo                    // 000000002db8: 02bab556
	v_add_co_u32 v87, vcc_lo, v84, s20                         // 000000002dbc: d7006a57 02002954
	v_bfe_u32 v86, v88, 16, 1                                  // 000000002dc4: d6100056 02052158
	s_wait_alu depctr_va_vcc(0)                                // 000000002dcc: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s21, v85, vcc_lo            // 000000002dd0: d5207c5a 01aaaa15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002dd8: bf870193
	v_add_co_u32 v84, vcc_lo, v87, v110                        // 000000002ddc: d7006a54 0202dd57
	v_add3_u32 v86, v86, v88, 0x7fff                           // 000000002de4: d6550056 03feb156 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000002df0: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000002df4: bf870003
	v_add_co_ci_u32_e64 v85, null, v90, v111, vcc_lo           // 000000002df8: d5207c55 01aadf5a
	v_cmp_u_f32_e32 vcc_lo, v88, v88                           // 000000002e00: 7c30b158
	s_wait_alu depctr_va_vcc(0)                                // 000000002e04: bf88ff9d
	v_cndmask_b32_e32 v94, v86, v91, vcc_lo                    // 000000002e08: 02bcb756
	v_bfe_u32 v86, v89, 16, 1                                  // 000000002e0c: d6100056 02052159
	v_add_co_u32 v88, vcc_lo, v87, s20                         // 000000002e14: d7006a58 02002957
	s_wait_alu depctr_va_vcc(0)                                // 000000002e1c: bf88ff9d
	v_add_co_ci_u32_e64 v90, null, s21, v90, vcc_lo            // 000000002e20: d5207c5a 01aab415
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000002e28: bf870193
	v_add3_u32 v91, v86, v89, 0x7fff                           // 000000002e2c: d655005b 03feb356 00007fff
	v_add_co_u32 v86, vcc_lo, v88, v110                        // 000000002e38: d7006a56 0202dd58
	s_wait_alu depctr_va_vcc(0)                                // 000000002e40: bf88ff9d
	s_delay_alu instid0(valu_dep_3) | instskip(skip_2) | instid1(valu_dep_1)// 000000002e44: bf8700b3
	v_add_co_ci_u32_e64 v87, null, v90, v111, vcc_lo           // 000000002e48: d5207c57 01aadf5a
	v_cmp_u_f32_e32 vcc_lo, v89, v89                           // 000000002e50: 7c30b359
	v_add_co_u32 v88, s0, v88, s20                             // 000000002e54: d7000058 02002958
	v_add_co_ci_u32_e64 v89, null, s21, v90, s0                // 000000002e5c: d5207c59 0002b415
	s_wait_alu depctr_va_vcc(0)                                // 000000002e64: bf88ff9d
	v_cndmask_b32_e32 v95, v91, v95, vcc_lo                    // 000000002e68: 02bebf5b
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[100:101]              // 000000002e6c: 7ca8c810
	s_wait_alu depctr_va_vcc(0)                                // 000000002e70: bf88ff9d
	v_dual_cndmask_b32 v91, 0, v101 :: v_dual_cndmask_b32 v90, 0, v100// 000000002e74: ca52ca80 5b5ac880
	v_add_co_u32 v88, vcc_lo, v88, v110                        // 000000002e7c: d7006a58 0202dd58
	s_wait_alu depctr_va_vcc(0)                                // 000000002e84: bf88ff9d
	v_add_co_ci_u32_e64 v89, null, v89, v111, vcc_lo           // 000000002e88: d5207c59 01aadf59
	s_delay_alu instid0(valu_dep_3)                            // 000000002e90: bf870003
	v_lshlrev_b64_e32 v[90:91], 2, v[90:91]                    // 000000002e94: 3eb4b482
	s_clause 0x3                                               // 000000002e98: bf850003
	global_store_d16_hi_b16 v[80:81], v92, off                 // 000000002e9c: ee09407c 2e000000 00000050
	global_store_d16_hi_b16 v[84:85], v93, off                 // 000000002ea8: ee09407c 2e800000 00000054
	global_store_d16_hi_b16 v[86:87], v94, off                 // 000000002eb4: ee09407c 2f000000 00000056
	global_store_d16_hi_b16 v[88:89], v95, off                 // 000000002ec0: ee09407c 2f800000 00000058
	v_add_co_u32 v90, vcc_lo, s4, v90                          // 000000002ecc: d7006a5a 0202b404
	s_wait_alu depctr_va_vcc(0)                                // 000000002ed4: bf88ff9d
	v_add_co_ci_u32_e64 v91, null, s5, v91, vcc_lo             // 000000002ed8: d5207c5b 01aab605
	global_load_u8 v93, v[124:125], off                        // 000000002ee0: ee04007c 0000005d 0000007c
	global_load_b32 v92, v[90:91], off                         // 000000002eec: ee05007c 0000005c 0000005a
	s_wait_loadcnt 0x1                                         // 000000002ef8: bfc00001
	v_lshlrev_b32_e32 v105, 23, v93                            // 000000002efc: 30d2ba97
	s_wait_loadcnt 0x0                                         // 000000002f00: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000002f04: bf870091
	v_mul_f32_e32 v93, v92, v105                               // 000000002f08: 10bad35c
	v_cmp_class_f32_e64 s0, v93, 0x198                         // 000000002f0c: d47e0000 0201ff5d 00000198
	v_mul_f32_e32 v103, v48, v93                               // 000000002f18: 10cebb30
	s_xor_b32 s1, s0, -1                                       // 000000002f1c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f20: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000002f24: be802001
	s_cbranch_execnz 2406                                      // 000000002f28: bfa60966 <packed_folded_w4a8+0x39c4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002f30: 8c7e007e
	v_or_b32_e32 v92, v144, v98                                // 000000002f34: 38b8c590
	v_mov_b32_e32 v93, v99                                     // 000000002f38: 7eba0363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000002f3c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[92:93]                // 000000002f40: 7ca8b810
	s_wait_alu depctr_va_vcc(0)                                // 000000002f44: bf88ff9d
	v_dual_cndmask_b32 v93, 0, v93 :: v_dual_cndmask_b32 v92, 0, v92// 000000002f48: ca52ba80 5d5cb880
	v_lshlrev_b64_e32 v[92:93], 2, v[92:93]                    // 000000002f50: 3eb8b882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002f54: bf870121
	v_add_co_u32 v92, vcc_lo, s4, v92                          // 000000002f58: d7006a5c 0202b804
	s_wait_alu depctr_va_vcc(0)                                // 000000002f60: bf88ff9d
	v_add_co_ci_u32_e64 v93, null, s5, v93, vcc_lo             // 000000002f64: d5207c5d 01aaba05
	global_load_b32 v48, v[92:93], off                         // 000000002f6c: ee05007c 00000030 0000005c
	s_wait_loadcnt 0x0                                         // 000000002f78: bfc00000
	v_mul_f32_e32 v94, v48, v105                               // 000000002f7c: 10bcd330
	s_delay_alu instid0(valu_dep_1)                            // 000000002f80: bf870001
	v_cmp_class_f32_e64 s0, v94, 0x198                         // 000000002f84: d47e0000 0201ff5e 00000198
	v_mul_f32_e32 v104, v49, v94                               // 000000002f90: 10d0bd31
	s_xor_b32 s1, s0, -1                                       // 000000002f94: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000002f98: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000002f9c: be802001
	s_cbranch_execnz 2393                                      // 000000002fa0: bfa60959 <packed_folded_w4a8+0x3a08>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002fa4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000002fa8: 8c7e007e
	v_or_b32_e32 v48, v145, v98                                // 000000002fac: 3860c591
	v_mov_b32_e32 v49, v99                                     // 000000002fb0: 7e620363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000002fb4: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[48:49]                // 000000002fb8: 7ca86010
	s_wait_alu depctr_va_vcc(0)                                // 000000002fbc: bf88ff9d
	v_dual_cndmask_b32 v49, 0, v49 :: v_dual_cndmask_b32 v48, 0, v48// 000000002fc0: ca526280 31306080
	v_lshlrev_b64_e32 v[48:49], 2, v[48:49]                    // 000000002fc8: 3e606082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000002fcc: bf870121
	v_add_co_u32 v48, vcc_lo, s4, v48                          // 000000002fd0: d7006a30 02026004
	s_wait_alu depctr_va_vcc(0)                                // 000000002fd8: bf88ff9d
	v_add_co_ci_u32_e64 v49, null, s5, v49, vcc_lo             // 000000002fdc: d5207c31 01aa6205
	global_load_b32 v94, v[48:49], off                         // 000000002fe4: ee05007c 0000005e 00000030
	s_wait_loadcnt 0x0                                         // 000000002ff0: bfc00000
	v_mul_f32_e32 v95, v94, v105                               // 000000002ff4: 10bed35e
	s_delay_alu instid0(valu_dep_1)                            // 000000002ff8: bf870001
	v_cmp_class_f32_e64 s0, v95, 0x198                         // 000000002ffc: d47e0000 0201ff5f 00000198
	v_mul_f32_e32 v106, v50, v95                               // 000000003008: 10d4bf32
	s_xor_b32 s1, s0, -1                                       // 00000000300c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003010: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003014: be802001
	s_cbranch_execnz 2380                                      // 000000003018: bfa6094c <packed_folded_w4a8+0x3a4c>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000301c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003020: 8c7e007e
	v_or_b32_e32 v94, v146, v98                                // 000000003024: 38bcc592
	v_mov_b32_e32 v95, v99                                     // 000000003028: 7ebe0363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 00000000302c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[94:95]                // 000000003030: 7ca8bc10
	s_wait_alu depctr_va_vcc(0)                                // 000000003034: bf88ff9d
	v_dual_cndmask_b32 v95, 0, v95 :: v_dual_cndmask_b32 v94, 0, v94// 000000003038: ca52be80 5f5ebc80
	v_lshlrev_b64_e32 v[94:95], 2, v[94:95]                    // 000000003040: 3ebcbc82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003044: bf870121
	v_add_co_u32 v94, vcc_lo, s4, v94                          // 000000003048: d7006a5e 0202bc04
	s_wait_alu depctr_va_vcc(0)                                // 000000003050: bf88ff9d
	v_add_co_ci_u32_e64 v95, null, s5, v95, vcc_lo             // 000000003054: d5207c5f 01aabe05
	global_load_b32 v50, v[94:95], off                         // 00000000305c: ee05007c 00000032 0000005e
	s_wait_loadcnt 0x0                                         // 000000003068: bfc00000
	v_mul_f32_e32 v96, v50, v105                               // 00000000306c: 10c0d332
	s_delay_alu instid0(valu_dep_1)                            // 000000003070: bf870001
	v_cmp_class_f32_e64 s0, v96, 0x198                         // 000000003074: d47e0000 0201ff60 00000198
	v_mul_f32_e32 v108, v51, v96                               // 000000003080: 10d8c133
	s_xor_b32 s1, s0, -1                                       // 000000003084: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003088: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000308c: be802001
	s_cbranch_execnz 2367                                      // 000000003090: bfa6093f <packed_folded_w4a8+0x3a90>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003094: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003098: 8c7e007e
	v_or_b32_e32 v50, v147, v98                                // 00000000309c: 3864c593
	v_mov_b32_e32 v51, v99                                     // 0000000030a0: 7e660363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 0000000030a4: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[50:51]                // 0000000030a8: 7ca86410
	s_wait_alu depctr_va_vcc(0)                                // 0000000030ac: bf88ff9d
	v_dual_cndmask_b32 v51, 0, v51 :: v_dual_cndmask_b32 v50, 0, v50// 0000000030b0: ca526680 33326480
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]                    // 0000000030b8: 3e646482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000030bc: bf870121
	v_add_co_u32 v50, vcc_lo, s4, v50                          // 0000000030c0: d7006a32 02026404
	s_wait_alu depctr_va_vcc(0)                                // 0000000030c8: bf88ff9d
	v_add_co_ci_u32_e64 v51, null, s5, v51, vcc_lo             // 0000000030cc: d5207c33 01aa6605
	global_load_b32 v96, v[50:51], off                         // 0000000030d4: ee05007c 00000060 00000032
	s_wait_loadcnt 0x0                                         // 0000000030e0: bfc00000
	v_mul_f32_e32 v97, v96, v105                               // 0000000030e4: 10c2d360
	s_delay_alu instid0(valu_dep_1)                            // 0000000030e8: bf870001
	v_cmp_class_f32_e64 s0, v97, 0x198                         // 0000000030ec: d47e0000 0201ff61 00000198
	v_mul_f32_e32 v109, v52, v97                               // 0000000030f8: 10dac334
	s_xor_b32 s1, s0, -1                                       // 0000000030fc: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003100: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003104: be802001
	s_cbranch_execnz 2354                                      // 000000003108: bfa60932 <packed_folded_w4a8+0x3ad4>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000310c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003110: 8c7e007e
	v_or_b32_e32 v96, v152, v98                                // 000000003114: 38c0c598
	v_mov_b32_e32 v97, v99                                     // 000000003118: 7ec20363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 00000000311c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[96:97]                // 000000003120: 7ca8c010
	s_wait_alu depctr_va_vcc(0)                                // 000000003124: bf88ff9d
	v_dual_cndmask_b32 v97, 0, v97 :: v_dual_cndmask_b32 v96, 0, v96// 000000003128: ca52c280 6160c080
	v_lshlrev_b64_e32 v[96:97], 2, v[96:97]                    // 000000003130: 3ec0c082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003134: bf870121
	v_add_co_u32 v96, vcc_lo, s4, v96                          // 000000003138: d7006a60 0202c004
	s_wait_alu depctr_va_vcc(0)                                // 000000003140: bf88ff9d
	v_add_co_ci_u32_e64 v97, null, s5, v97, vcc_lo             // 000000003144: d5207c61 01aac205
	global_load_b32 v52, v[96:97], off                         // 00000000314c: ee05007c 00000034 00000060
	s_wait_loadcnt 0x0                                         // 000000003158: bfc00000
	v_mul_f32_e32 v107, v52, v105                              // 00000000315c: 10d6d334
	s_delay_alu instid0(valu_dep_1)                            // 000000003160: bf870001
	v_cmp_class_f32_e64 s0, v107, 0x198                        // 000000003164: d47e0000 0201ff6b 00000198
	v_mul_f32_e32 v112, v53, v107                              // 000000003170: 10e0d735
	s_xor_b32 s1, s0, -1                                       // 000000003174: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003178: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000317c: be802001
	s_cbranch_execnz 2341                                      // 000000003180: bfa60925 <packed_folded_w4a8+0x3b18>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003184: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003188: 8c7e007e
	v_or_b32_e32 v52, v153, v98                                // 00000000318c: 3868c599
	v_mov_b32_e32 v53, v99                                     // 000000003190: 7e6a0363
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003194: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[52:53]                // 000000003198: 7ca86810
	s_wait_alu depctr_va_vcc(0)                                // 00000000319c: bf88ff9d
	v_dual_cndmask_b32 v53, 0, v53 :: v_dual_cndmask_b32 v52, 0, v52// 0000000031a0: ca526a80 35346880
	v_lshlrev_b64_e32 v[52:53], 2, v[52:53]                    // 0000000031a8: 3e686882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000031ac: bf870121
	v_add_co_u32 v52, vcc_lo, s4, v52                          // 0000000031b0: d7006a34 02026804
	s_wait_alu depctr_va_vcc(0)                                // 0000000031b8: bf88ff9d
	v_add_co_ci_u32_e64 v53, null, s5, v53, vcc_lo             // 0000000031bc: d5207c35 01aa6a05
	global_load_b32 v107, v[52:53], off                        // 0000000031c4: ee05007c 0000006b 00000034
	s_wait_loadcnt 0x0                                         // 0000000031d0: bfc00000
	v_mul_f32_e32 v113, v107, v105                             // 0000000031d4: 10e2d36b
	s_delay_alu instid0(valu_dep_1)                            // 0000000031d8: bf870001
	v_cmp_class_f32_e64 s0, v113, 0x198                        // 0000000031dc: d47e0000 0201ff71 00000198
	v_mul_f32_e32 v113, v54, v113                              // 0000000031e8: 10e2e336
	s_xor_b32 s1, s0, -1                                       // 0000000031ec: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031f0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000031f4: be802001
	s_cbranch_execnz 2328                                      // 0000000031f8: bfa60918 <packed_folded_w4a8+0x3b5c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003200: 8c7e007e
	v_or_b32_e32 v98, v154, v98                                // 000000003204: 38c4c59a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003208: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[98:99]                // 00000000320c: 7ca8c410
	s_wait_alu depctr_va_vcc(0)                                // 000000003210: bf88ff9d
	v_dual_cndmask_b32 v99, 0, v99 :: v_dual_cndmask_b32 v98, 0, v98// 000000003214: ca52c680 6362c480
	v_lshlrev_b64_e32 v[98:99], 2, v[98:99]                    // 00000000321c: 3ec4c482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003220: bf870121
	v_add_co_u32 v98, vcc_lo, s4, v98                          // 000000003224: d7006a62 0202c404
	s_wait_alu depctr_va_vcc(0)                                // 00000000322c: bf88ff9d
	v_add_co_ci_u32_e64 v99, null, s5, v99, vcc_lo             // 000000003230: d5207c63 01aac605
	global_load_b32 v54, v[98:99], off                         // 000000003238: ee05007c 00000036 00000062
	s_wait_loadcnt 0x0                                         // 000000003244: bfc00000
	v_mul_f32_e32 v107, v54, v105                              // 000000003248: 10d6d336
	s_delay_alu instid0(valu_dep_1)                            // 00000000324c: bf870001
	v_cmp_class_f32_e64 s0, v107, 0x198                        // 000000003250: d47e0000 0201ff6b 00000198
	v_mul_f32_e32 v114, v55, v107                              // 00000000325c: 10e4d737
	s_xor_b32 s1, s0, -1                                       // 000000003260: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003264: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003268: be802001
	s_cbranch_execnz 2316                                      // 00000000326c: bfa6090c <packed_folded_w4a8+0x3ba0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003270: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003274: 8c7e007e
	v_mul_lo_u32 v105, s19, v100                               // 000000003278: d72c0069 0202c813
	v_mul_lo_u32 v101, s18, v101                               // 000000003280: d72c0065 0202ca12
	v_mad_co_u64_u32 v[54:55], null, s18, v100, 0              // 000000003288: d6fe7c36 0202c812
	v_bfe_u32 v100, v103, 16, 1                                // 000000003290: d6100064 02052167
	v_cmp_u_f32_e32 vcc_lo, v103, v103                         // 000000003298: 7c30cf67
	v_or_b32_e32 v126, s22, v102                               // 00000000329c: 38fccc16
	v_bfe_u32 v102, v104, 16, 1                                // 0000000032a0: d6100066 02052168
	v_bfe_u32 v107, v106, 16, 1                                // 0000000032a8: d610006b 0205216a
	v_add3_u32 v100, v100, v103, 0x7fff                        // 0000000032b0: d6550064 03fecf64 00007fff
	v_or_b32_e32 v116, 0x400000, v108                          // 0000000032bc: 38e8d8ff 00400000
	v_add3_u32 v55, v55, v101, v105                            // 0000000032c4: d6550037 05a6cb37
	v_or_b32_e32 v101, 0x400000, v103                          // 0000000032cc: 38caceff 00400000
	v_or_b32_e32 v105, 0x400000, v104                          // 0000000032d4: 38d2d0ff 00400000
	v_add3_u32 v102, v102, v104, 0x7fff                        // 0000000032dc: d6550066 03fed166 00007fff
	v_bfe_u32 v117, v109, 16, 1                                // 0000000032e8: d6100075 0205216d
	v_lshlrev_b64_e32 v[54:55], 1, v[54:55]                    // 0000000032f0: 3e6c6c81
	s_wait_alu depctr_va_vcc(0)                                // 0000000032f4: bf88ff9d
	v_cndmask_b32_e32 v100, v100, v101, vcc_lo                 // 0000000032f8: 02c8cb64
	v_or_b32_e32 v118, 0x400000, v109                          // 0000000032fc: 38ecdaff 00400000
	v_or_b32_e32 v121, 0x400000, v114                          // 000000003304: 38f2e4ff 00400000
	v_add3_u32 v117, v117, v109, 0x7fff                        // 00000000330c: d6550075 03fedb75 00007fff
	v_mov_b32_e32 v129, s23                                    // 000000003318: 7f020217
	v_add_co_u32 v54, vcc_lo, s2, v54                          // 00000000331c: d7006a36 02026c02
	s_wait_alu depctr_va_vcc(0)                                // 000000003324: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s3, v55, vcc_lo             // 000000003328: d5207c37 01aa6e03
	v_cmp_u_f32_e32 vcc_lo, v104, v104                         // 000000003330: 7c30d168
	v_or_b32_e32 v128, v126, v138                              // 000000003334: 3901157e
	s_wait_alu depctr_va_vcc(0)                                // 000000003338: bf88ff9d
	v_dual_mov_b32 v127, s23 :: v_dual_cndmask_b32 v104, v102, v105// 00000000333c: ca120017 7f68d366
	v_add_co_u32 v102, vcc_lo, v54, v110                       // 000000003344: d7006a66 0202dd36
	s_wait_alu depctr_va_vcc(0)                                // 00000000334c: bf88ff9d
	v_add_co_ci_u32_e64 v103, null, v55, v111, vcc_lo          // 000000003350: d5207c67 01aadf37
	v_add_co_u32 v54, vcc_lo, v54, s20                         // 000000003358: d7006a36 02002936
	s_wait_alu depctr_va_vcc(0)                                // 000000003360: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s21, v55, vcc_lo            // 000000003364: d5207c37 01aa6e15
	global_store_d16_hi_b16 v[102:103], v100, off              // 00000000336c: ee09407c 32000000 00000066
	v_add_co_u32 v100, vcc_lo, v54, v110                       // 000000003378: d7006a64 0202dd36
	v_add3_u32 v105, v107, v106, 0x7fff                        // 000000003380: d6550069 03fed56b 00007fff
	v_or_b32_e32 v107, 0x400000, v106                          // 00000000338c: 38d6d4ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003394: bf88ff9d
	v_add_co_ci_u32_e64 v101, null, v55, v111, vcc_lo          // 000000003398: d5207c65 01aadf37
	v_cmp_u_f32_e32 vcc_lo, v106, v106                         // 0000000033a0: 7c30d56a
	v_bfe_u32 v106, v108, 16, 1                                // 0000000033a4: d610006a 0205216c
	s_wait_alu depctr_va_vcc(0)                                // 0000000033ac: bf88ff9d
	v_cndmask_b32_e32 v105, v105, v107, vcc_lo                 // 0000000033b0: 02d2d769
	v_add_co_u32 v54, vcc_lo, v54, s20                         // 0000000033b4: d7006a36 02002936
	s_wait_alu depctr_va_vcc(0)                                // 0000000033bc: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, s21, v55, vcc_lo            // 0000000033c0: d5207c37 01aa6e15
	v_add3_u32 v115, v106, v108, 0x7fff                        // 0000000033c8: d6550073 03fed96a 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000033d4: bf8701a3
	v_add_co_u32 v106, vcc_lo, v54, v110                       // 0000000033d8: d7006a6a 0202dd36
	s_wait_alu depctr_va_vcc(0)                                // 0000000033e0: bf88ff9d
	v_add_co_ci_u32_e64 v107, null, v55, v111, vcc_lo          // 0000000033e4: d5207c6b 01aadf37
	v_cmp_u_f32_e32 vcc_lo, v108, v108                         // 0000000033ec: 7c30d96c
	s_wait_alu depctr_va_vcc(0)                                // 0000000033f0: bf88ff9d
	v_cndmask_b32_e32 v108, v115, v116, vcc_lo                 // 0000000033f4: 02d8e973
	v_add_co_u32 v115, vcc_lo, v54, s20                        // 0000000033f8: d7006a73 02002936
	s_wait_alu depctr_va_vcc(0)                                // 000000003400: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, s21, v55, vcc_lo           // 000000003404: d5207c74 01aa6e15
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 00000000340c: bf870122
	v_add_co_u32 v54, vcc_lo, v115, v110                       // 000000003410: d7006a36 0202dd73
	s_wait_alu depctr_va_vcc(0)                                // 000000003418: bf88ff9d
	v_add_co_ci_u32_e64 v55, null, v116, v111, vcc_lo          // 00000000341c: d5207c37 01aadf74
	v_cmp_u_f32_e32 vcc_lo, v109, v109                         // 000000003424: 7c30db6d
	s_clause 0x2                                               // 000000003428: bf850002
	global_store_d16_hi_b16 v[100:101], v104, off              // 00000000342c: ee09407c 34000000 00000064
	global_store_d16_hi_b16 v[106:107], v105, off              // 000000003438: ee09407c 34800000 0000006a
	global_store_d16_hi_b16 v[54:55], v108, off                // 000000003444: ee09407c 36000000 00000036
	v_bfe_u32 v104, v112, 16, 1                                // 000000003450: d6100068 02052170
	s_wait_alu depctr_va_vcc(0)                                // 000000003458: bf88ff9d
	v_cndmask_b32_e32 v118, v117, v118, vcc_lo                 // 00000000345c: 02eced75
	v_add_co_u32 v108, vcc_lo, v115, s20                       // 000000003460: d7006a6c 02002973
	s_wait_alu depctr_va_vcc(0)                                // 000000003468: bf88ff9d
	v_add_co_ci_u32_e64 v109, null, s21, v116, vcc_lo          // 00000000346c: d5207c6d 01aae815
	v_add3_u32 v115, v104, v112, 0x7fff                        // 000000003474: d6550073 03fee168 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003480: bf870003
	v_add_co_u32 v104, vcc_lo, v108, v110                      // 000000003484: d7006a68 0202dd6c
	v_or_b32_e32 v116, 0x400000, v112                          // 00000000348c: 38e8e0ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003494: bf88ff9d
	v_add_co_ci_u32_e64 v105, null, v109, v111, vcc_lo         // 000000003498: d5207c69 01aadf6d
	v_cmp_u_f32_e32 vcc_lo, v112, v112                         // 0000000034a0: 7c30e170
	v_bfe_u32 v112, v113, 16, 1                                // 0000000034a4: d6100070 02052171
	v_or_b32_e32 v117, 0x400000, v113                          // 0000000034ac: 38eae2ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000034b4: bf88ff9d
	v_cndmask_b32_e32 v119, v115, v116, vcc_lo                 // 0000000034b8: 02eee973
	v_add_co_u32 v115, vcc_lo, v108, s20                       // 0000000034bc: d7006a73 0200296c
	s_wait_alu depctr_va_vcc(0)                                // 0000000034c4: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, s21, v109, vcc_lo          // 0000000034c8: d5207c74 01aada15
	v_add3_u32 v112, v112, v113, 0x7fff                        // 0000000034d0: d6550070 03fee370 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 0000000034dc: bf8701a3
	v_add_co_u32 v108, vcc_lo, v115, v110                      // 0000000034e0: d7006a6c 0202dd73
	s_wait_alu depctr_va_vcc(0)                                // 0000000034e8: bf88ff9d
	v_add_co_ci_u32_e64 v109, null, v116, v111, vcc_lo         // 0000000034ec: d5207c6d 01aadf74
	v_cmp_u_f32_e32 vcc_lo, v113, v113                         // 0000000034f4: 7c30e371
	s_wait_alu depctr_va_vcc(0)                                // 0000000034f8: bf88ff9d
	v_cndmask_b32_e32 v120, v112, v117, vcc_lo                 // 0000000034fc: 02f0eb70
	v_bfe_u32 v112, v114, 16, 1                                // 000000003500: d6100070 02052172
	v_add_co_u32 v115, vcc_lo, v115, s20                       // 000000003508: d7006a73 02002973
	s_wait_alu depctr_va_vcc(0)                                // 000000003510: bf88ff9d
	v_add_co_ci_u32_e64 v116, null, s21, v116, vcc_lo          // 000000003514: d5207c74 01aae815
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 00000000351c: bf870193
	v_add3_u32 v117, v112, v114, 0x7fff                        // 000000003520: d6550075 03fee570 00007fff
	v_add_co_u32 v112, vcc_lo, v115, v110                      // 00000000352c: d7006a70 0202dd73
	s_wait_alu depctr_va_vcc(0)                                // 000000003534: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 000000003538: bf870003
	v_add_co_ci_u32_e64 v113, null, v116, v111, vcc_lo         // 00000000353c: d5207c71 01aadf74
	v_cmp_u_f32_e32 vcc_lo, v114, v114                         // 000000003544: 7c30e572
	v_add_co_u32 v114, s0, v115, s20                           // 000000003548: d7000072 02002973
	s_wait_alu depctr_va_sdst(0)                               // 000000003550: bf88f19f
	v_add_co_ci_u32_e64 v115, null, s21, v116, s0              // 000000003554: d5207c73 0002e815
	s_wait_alu depctr_va_vcc(0)                                // 00000000355c: bf88ff9d
	v_cndmask_b32_e32 v121, v117, v121, vcc_lo                 // 000000003560: 02f2f375
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[128:129]              // 000000003564: 7ca90010
	s_wait_alu depctr_va_vcc(0)                                // 000000003568: bf88ff9d
	v_dual_cndmask_b32 v117, 0, v129 :: v_dual_cndmask_b32 v116, 0, v128// 00000000356c: ca530280 75750080
	v_add_co_u32 v114, vcc_lo, v114, v110                      // 000000003574: d7006a72 0202dd72
	s_wait_alu depctr_va_vcc(0)                                // 00000000357c: bf88ff9d
	v_add_co_ci_u32_e64 v115, null, v115, v111, vcc_lo         // 000000003580: d5207c73 01aadf73
	s_delay_alu instid0(valu_dep_3)                            // 000000003588: bf870003
	v_lshlrev_b64_e32 v[116:117], 2, v[116:117]                // 00000000358c: 3ee8e882
	s_clause 0x3                                               // 000000003590: bf850003
	global_store_d16_hi_b16 v[104:105], v118, off              // 000000003594: ee09407c 3b000000 00000068
	global_store_d16_hi_b16 v[108:109], v119, off              // 0000000035a0: ee09407c 3b800000 0000006c
	global_store_d16_hi_b16 v[112:113], v120, off              // 0000000035ac: ee09407c 3c000000 00000070
	global_store_d16_hi_b16 v[114:115], v121, off              // 0000000035b8: ee09407c 3c800000 00000072
	v_add_co_u32 v116, vcc_lo, s4, v116                        // 0000000035c4: d7006a74 0202e804
	s_wait_alu depctr_va_vcc(0)                                // 0000000035cc: bf88ff9d
	v_add_co_ci_u32_e64 v117, null, s5, v117, vcc_lo           // 0000000035d0: d5207c75 01aaea05
	global_load_u8 v119, v[124:125], off                       // 0000000035d8: ee04007c 00000077 0000007c
	global_load_b32 v118, v[116:117], off                      // 0000000035e4: ee05007c 00000076 00000074
	s_wait_loadcnt 0x1                                         // 0000000035f0: bfc00001
	v_lshlrev_b32_e32 v133, 23, v119                           // 0000000035f4: 310aee97
	s_wait_loadcnt 0x0                                         // 0000000035f8: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000035fc: bf870091
	v_mul_f32_e32 v119, v118, v133                             // 000000003600: 10ef0b76
	v_cmp_class_f32_e64 s0, v119, 0x198                        // 000000003604: d47e0000 0201ff77 00000198
	v_mul_f32_e32 v131, v40, v119                              // 000000003610: 1106ef28
	s_xor_b32 s1, s0, -1                                       // 000000003614: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003618: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000361c: be802001
	s_cbranch_execnz 2096                                      // 000000003620: bfa60830 <packed_folded_w4a8+0x3be4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003624: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003628: 8c7e007e
	v_or_b32_e32 v118, v144, v126                              // 00000000362c: 38ecfd90
	v_mov_b32_e32 v119, v127                                   // 000000003630: 7eee037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003634: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[118:119]              // 000000003638: 7ca8ec10
	s_wait_alu depctr_va_vcc(0)                                // 00000000363c: bf88ff9d
	v_dual_cndmask_b32 v119, 0, v119 :: v_dual_cndmask_b32 v118, 0, v118// 000000003640: ca52ee80 7776ec80
	v_lshlrev_b64_e32 v[118:119], 2, v[118:119]                // 000000003648: 3eecec82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000364c: bf870121
	v_add_co_u32 v118, vcc_lo, s4, v118                        // 000000003650: d7006a76 0202ec04
	s_wait_alu depctr_va_vcc(0)                                // 000000003658: bf88ff9d
	v_add_co_ci_u32_e64 v119, null, s5, v119, vcc_lo           // 00000000365c: d5207c77 01aaee05
	global_load_b32 v40, v[118:119], off                       // 000000003664: ee05007c 00000028 00000076
	s_wait_loadcnt 0x0                                         // 000000003670: bfc00000
	v_mul_f32_e32 v120, v40, v133                              // 000000003674: 10f10b28
	s_delay_alu instid0(valu_dep_1)                            // 000000003678: bf870001
	v_cmp_class_f32_e64 s0, v120, 0x198                        // 00000000367c: d47e0000 0201ff78 00000198
	v_mul_f32_e32 v132, v41, v120                              // 000000003688: 1108f129
	s_xor_b32 s1, s0, -1                                       // 00000000368c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003690: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003694: be802001
	s_cbranch_execnz 2083                                      // 000000003698: bfa60823 <packed_folded_w4a8+0x3c28>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000369c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000036a0: 8c7e007e
	v_or_b32_e32 v40, v145, v126                               // 0000000036a4: 3850fd91
	v_mov_b32_e32 v41, v127                                    // 0000000036a8: 7e52037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 0000000036ac: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[40:41]                // 0000000036b0: 7ca85010
	s_wait_alu depctr_va_vcc(0)                                // 0000000036b4: bf88ff9d
	v_dual_cndmask_b32 v41, 0, v41 :: v_dual_cndmask_b32 v40, 0, v40// 0000000036b8: ca525280 29285080
	v_lshlrev_b64_e32 v[40:41], 2, v[40:41]                    // 0000000036c0: 3e505082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000036c4: bf870121
	v_add_co_u32 v40, vcc_lo, s4, v40                          // 0000000036c8: d7006a28 02025004
	s_wait_alu depctr_va_vcc(0)                                // 0000000036d0: bf88ff9d
	v_add_co_ci_u32_e64 v41, null, s5, v41, vcc_lo             // 0000000036d4: d5207c29 01aa5205
	global_load_b32 v120, v[40:41], off                        // 0000000036dc: ee05007c 00000078 00000028
	s_wait_loadcnt 0x0                                         // 0000000036e8: bfc00000
	v_mul_f32_e32 v121, v120, v133                             // 0000000036ec: 10f30b78
	s_delay_alu instid0(valu_dep_1)                            // 0000000036f0: bf870001
	v_cmp_class_f32_e64 s0, v121, 0x198                        // 0000000036f4: d47e0000 0201ff79 00000198
	v_mul_f32_e32 v134, v42, v121                              // 000000003700: 110cf32a
	s_xor_b32 s1, s0, -1                                       // 000000003704: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003708: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000370c: be802001
	s_cbranch_execnz 2070                                      // 000000003710: bfa60816 <packed_folded_w4a8+0x3c6c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003714: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003718: 8c7e007e
	v_or_b32_e32 v120, v146, v126                              // 00000000371c: 38f0fd92
	v_mov_b32_e32 v121, v127                                   // 000000003720: 7ef2037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003724: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[120:121]              // 000000003728: 7ca8f010
	s_wait_alu depctr_va_vcc(0)                                // 00000000372c: bf88ff9d
	v_dual_cndmask_b32 v121, 0, v121 :: v_dual_cndmask_b32 v120, 0, v120// 000000003730: ca52f280 7978f080
	v_lshlrev_b64_e32 v[120:121], 2, v[120:121]                // 000000003738: 3ef0f082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000373c: bf870121
	v_add_co_u32 v120, vcc_lo, s4, v120                        // 000000003740: d7006a78 0202f004
	s_wait_alu depctr_va_vcc(0)                                // 000000003748: bf88ff9d
	v_add_co_ci_u32_e64 v121, null, s5, v121, vcc_lo           // 00000000374c: d5207c79 01aaf205
	global_load_b32 v42, v[120:121], off                       // 000000003754: ee05007c 0000002a 00000078
	s_wait_loadcnt 0x0                                         // 000000003760: bfc00000
	v_mul_f32_e32 v122, v42, v133                              // 000000003764: 10f50b2a
	s_delay_alu instid0(valu_dep_1)                            // 000000003768: bf870001
	v_cmp_class_f32_e64 s0, v122, 0x198                        // 00000000376c: d47e0000 0201ff7a 00000198
	v_mul_f32_e32 v136, v43, v122                              // 000000003778: 1110f52b
	s_xor_b32 s1, s0, -1                                       // 00000000377c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003780: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003784: be802001
	s_cbranch_execnz 2057                                      // 000000003788: bfa60809 <packed_folded_w4a8+0x3cb0>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000378c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003790: 8c7e007e
	v_or_b32_e32 v42, v147, v126                               // 000000003794: 3854fd93
	v_mov_b32_e32 v43, v127                                    // 000000003798: 7e56037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 00000000379c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[42:43]                // 0000000037a0: 7ca85410
	s_wait_alu depctr_va_vcc(0)                                // 0000000037a4: bf88ff9d
	v_dual_cndmask_b32 v43, 0, v43 :: v_dual_cndmask_b32 v42, 0, v42// 0000000037a8: ca525680 2b2a5480
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]                    // 0000000037b0: 3e545482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000037b4: bf870121
	v_add_co_u32 v42, vcc_lo, s4, v42                          // 0000000037b8: d7006a2a 02025404
	s_wait_alu depctr_va_vcc(0)                                // 0000000037c0: bf88ff9d
	v_add_co_ci_u32_e64 v43, null, s5, v43, vcc_lo             // 0000000037c4: d5207c2b 01aa5605
	global_load_b32 v122, v[42:43], off                        // 0000000037cc: ee05007c 0000007a 0000002a
	s_wait_loadcnt 0x0                                         // 0000000037d8: bfc00000
	v_mul_f32_e32 v123, v122, v133                             // 0000000037dc: 10f70b7a
	s_delay_alu instid0(valu_dep_1)                            // 0000000037e0: bf870001
	v_cmp_class_f32_e64 s0, v123, 0x198                        // 0000000037e4: d47e0000 0201ff7b 00000198
	v_mul_f32_e32 v137, v44, v123                              // 0000000037f0: 1112f72c
	s_xor_b32 s1, s0, -1                                       // 0000000037f4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000037f8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000037fc: be802001
	s_cbranch_execnz 2044                                      // 000000003800: bfa607fc <packed_folded_w4a8+0x3cf4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003804: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003808: 8c7e007e
	v_or_b32_e32 v122, v152, v126                              // 00000000380c: 38f4fd98
	v_mov_b32_e32 v123, v127                                   // 000000003810: 7ef6037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003814: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[122:123]              // 000000003818: 7ca8f410
	s_wait_alu depctr_va_vcc(0)                                // 00000000381c: bf88ff9d
	v_dual_cndmask_b32 v123, 0, v123 :: v_dual_cndmask_b32 v122, 0, v122// 000000003820: ca52f680 7b7af480
	v_lshlrev_b64_e32 v[122:123], 2, v[122:123]                // 000000003828: 3ef4f482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 00000000382c: bf870121
	v_add_co_u32 v122, vcc_lo, s4, v122                        // 000000003830: d7006a7a 0202f404
	s_wait_alu depctr_va_vcc(0)                                // 000000003838: bf88ff9d
	v_add_co_ci_u32_e64 v123, null, s5, v123, vcc_lo           // 00000000383c: d5207c7b 01aaf605
	global_load_b32 v44, v[122:123], off                       // 000000003844: ee05007c 0000002c 0000007a
	s_wait_loadcnt 0x0                                         // 000000003850: bfc00000
	v_mul_f32_e32 v135, v44, v133                              // 000000003854: 110f0b2c
	s_delay_alu instid0(valu_dep_1)                            // 000000003858: bf870001
	v_cmp_class_f32_e64 s0, v135, 0x198                        // 00000000385c: d47e0000 0201ff87 00000198
	v_mul_f32_e32 v139, v45, v135                              // 000000003868: 11170f2d
	s_xor_b32 s1, s0, -1                                       // 00000000386c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003870: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003874: be802001
	s_cbranch_execnz 2031                                      // 000000003878: bfa607ef <packed_folded_w4a8+0x3d38>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000387c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003880: 8c7e007e
	v_or_b32_e32 v44, v153, v126                               // 000000003884: 3858fd99
	v_mov_b32_e32 v45, v127                                    // 000000003888: 7e5a037f
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 00000000388c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[44:45]                // 000000003890: 7ca85810
	s_wait_alu depctr_va_vcc(0)                                // 000000003894: bf88ff9d
	v_dual_cndmask_b32 v45, 0, v45 :: v_dual_cndmask_b32 v44, 0, v44// 000000003898: ca525a80 2d2c5880
	v_lshlrev_b64_e32 v[44:45], 2, v[44:45]                    // 0000000038a0: 3e585882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 0000000038a4: bf870121
	v_add_co_u32 v44, vcc_lo, s4, v44                          // 0000000038a8: d7006a2c 02025804
	s_wait_alu depctr_va_vcc(0)                                // 0000000038b0: bf88ff9d
	v_add_co_ci_u32_e64 v45, null, s5, v45, vcc_lo             // 0000000038b4: d5207c2d 01aa5a05
	global_load_b32 v135, v[44:45], off                        // 0000000038bc: ee05007c 00000087 0000002c
	s_wait_loadcnt 0x0                                         // 0000000038c8: bfc00000
	v_mul_f32_e32 v140, v135, v133                             // 0000000038cc: 11190b87
	s_delay_alu instid0(valu_dep_1)                            // 0000000038d0: bf870001
	v_cmp_class_f32_e64 s0, v140, 0x198                        // 0000000038d4: d47e0000 0201ff8c 00000198
	v_mul_f32_e32 v140, v46, v140                              // 0000000038e0: 1119192e
	s_xor_b32 s1, s0, -1                                       // 0000000038e4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038e8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000038ec: be802001
	s_cbranch_execnz 2018                                      // 0000000038f0: bfa607e2 <packed_folded_w4a8+0x3d7c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000038f4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000038f8: 8c7e007e
	v_or_b32_e32 v126, v154, v126                              // 0000000038fc: 38fcfd9a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003900: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[126:127]              // 000000003904: 7ca8fc10
	s_wait_alu depctr_va_vcc(0)                                // 000000003908: bf88ff9d
	v_dual_cndmask_b32 v127, 0, v127 :: v_dual_cndmask_b32 v126, 0, v126// 00000000390c: ca52fe80 7f7efc80
	v_lshlrev_b64_e32 v[126:127], 2, v[126:127]                // 000000003914: 3efcfc82
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003918: bf870121
	v_add_co_u32 v126, vcc_lo, s4, v126                        // 00000000391c: d7006a7e 0202fc04
	s_wait_alu depctr_va_vcc(0)                                // 000000003924: bf88ff9d
	v_add_co_ci_u32_e64 v127, null, s5, v127, vcc_lo           // 000000003928: d5207c7f 01aafe05
	global_load_b32 v46, v[126:127], off                       // 000000003930: ee05007c 0000002e 0000007e
	s_wait_loadcnt 0x0                                         // 00000000393c: bfc00000
	v_mul_f32_e32 v135, v46, v133                              // 000000003940: 110f0b2e
	s_delay_alu instid0(valu_dep_1)                            // 000000003944: bf870001
	v_cmp_class_f32_e64 s0, v135, 0x198                        // 000000003948: d47e0000 0201ff87 00000198
	v_mul_f32_e32 v141, v47, v135                              // 000000003954: 111b0f2f
	s_xor_b32 s1, s0, -1                                       // 000000003958: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000395c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003960: be802001
	s_cbranch_execnz 2006                                      // 000000003964: bfa607d6 <packed_folded_w4a8+0x3dc0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003968: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000396c: 8c7e007e
	v_mul_lo_u32 v133, s19, v128                               // 000000003970: d72c0085 02030013
	v_mul_lo_u32 v129, s18, v129                               // 000000003978: d72c0081 02030212
	v_mad_co_u64_u32 v[46:47], null, s18, v128, 0              // 000000003980: d6fe7c2e 02030012
	v_bfe_u32 v128, v131, 16, 1                                // 000000003988: d6100080 02052183
	v_cmp_u_f32_e32 vcc_lo, v131, v131                         // 000000003990: 7c310783
	v_or_b32_e32 v148, s22, v130                               // 000000003994: 39290416
	v_bfe_u32 v130, v132, 16, 1                                // 000000003998: d6100082 02052184
	v_bfe_u32 v135, v134, 16, 1                                // 0000000039a0: d6100087 02052186
	v_add3_u32 v128, v128, v131, 0x7fff                        // 0000000039a8: d6550080 03ff0780 00007fff
	v_or_b32_e32 v143, 0x400000, v136                          // 0000000039b4: 391f10ff 00400000
	v_add3_u32 v47, v47, v129, v133                            // 0000000039bc: d655002f 0617032f
	v_or_b32_e32 v129, 0x400000, v131                          // 0000000039c4: 390306ff 00400000
	v_or_b32_e32 v133, 0x400000, v132                          // 0000000039cc: 390b08ff 00400000
	v_add3_u32 v130, v130, v132, 0x7fff                        // 0000000039d4: d6550082 03ff0982 00007fff
	v_bfe_u32 v149, v137, 16, 1                                // 0000000039e0: d6100095 02052189
	v_lshlrev_b64_e32 v[46:47], 1, v[46:47]                    // 0000000039e8: 3e5c5c81
	s_wait_alu depctr_va_vcc(0)                                // 0000000039ec: bf88ff9d
	v_cndmask_b32_e32 v128, v128, v129, vcc_lo                 // 0000000039f0: 03010380
	v_or_b32_e32 v150, 0x400000, v137                          // 0000000039f4: 392d12ff 00400000
	v_or_b32_e32 v157, 0x400000, v141                          // 0000000039fc: 393b1aff 00400000
	v_add3_u32 v149, v149, v137, 0x7fff                        // 000000003a04: d6550095 03ff1395 00007fff
	v_mov_b32_e32 v151, s23                                    // 000000003a10: 7f2e0217
	v_add_co_u32 v46, vcc_lo, s2, v46                          // 000000003a14: d7006a2e 02025c02
	s_wait_alu depctr_va_vcc(0)                                // 000000003a1c: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s3, v47, vcc_lo             // 000000003a20: d5207c2f 01aa5e03
	v_cmp_u_f32_e32 vcc_lo, v132, v132                         // 000000003a28: 7c310984
	s_wait_alu depctr_va_vcc(0)                                // 000000003a2c: bf88ff9d
	v_cndmask_b32_e32 v132, v130, v133, vcc_lo                 // 000000003a30: 03090b82
	v_add_co_u32 v130, vcc_lo, v46, v110                       // 000000003a34: d7006a82 0202dd2e
	s_wait_alu depctr_va_vcc(0)                                // 000000003a3c: bf88ff9d
	v_add_co_ci_u32_e64 v131, null, v47, v111, vcc_lo          // 000000003a40: d5207c83 01aadf2f
	v_add_co_u32 v46, vcc_lo, v46, s20                         // 000000003a48: d7006a2e 0200292e
	s_wait_alu depctr_va_vcc(0)                                // 000000003a50: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s21, v47, vcc_lo            // 000000003a54: d5207c2f 01aa5e15
	global_store_d16_hi_b16 v[130:131], v128, off              // 000000003a5c: ee09407c 40000000 00000082
	v_add_co_u32 v128, vcc_lo, v46, v110                       // 000000003a68: d7006a80 0202dd2e
	v_add3_u32 v133, v135, v134, 0x7fff                        // 000000003a70: d6550085 03ff0d87 00007fff
	v_or_b32_e32 v135, 0x400000, v134                          // 000000003a7c: 390f0cff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003a84: bf88ff9d
	v_add_co_ci_u32_e64 v129, null, v47, v111, vcc_lo          // 000000003a88: d5207c81 01aadf2f
	v_cmp_u_f32_e32 vcc_lo, v134, v134                         // 000000003a90: 7c310d86
	v_bfe_u32 v134, v136, 16, 1                                // 000000003a94: d6100086 02052188
	s_wait_alu depctr_va_vcc(0)                                // 000000003a9c: bf88ff9d
	v_cndmask_b32_e32 v133, v133, v135, vcc_lo                 // 000000003aa0: 030b0f85
	v_add_co_u32 v46, vcc_lo, v46, s20                         // 000000003aa4: d7006a2e 0200292e
	s_wait_alu depctr_va_vcc(0)                                // 000000003aac: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, s21, v47, vcc_lo            // 000000003ab0: d5207c2f 01aa5e15
	v_add3_u32 v142, v134, v136, 0x7fff                        // 000000003ab8: d655008e 03ff1186 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003ac4: bf8701a3
	v_add_co_u32 v134, vcc_lo, v46, v110                       // 000000003ac8: d7006a86 0202dd2e
	s_wait_alu depctr_va_vcc(0)                                // 000000003ad0: bf88ff9d
	v_add_co_ci_u32_e64 v135, null, v47, v111, vcc_lo          // 000000003ad4: d5207c87 01aadf2f
	v_cmp_u_f32_e32 vcc_lo, v136, v136                         // 000000003adc: 7c311188
	s_wait_alu depctr_va_vcc(0)                                // 000000003ae0: bf88ff9d
	v_cndmask_b32_e32 v136, v142, v143, vcc_lo                 // 000000003ae4: 03111f8e
	v_add_co_u32 v142, vcc_lo, v46, s20                        // 000000003ae8: d7006a8e 0200292e
	s_wait_alu depctr_va_vcc(0)                                // 000000003af0: bf88ff9d
	v_add_co_ci_u32_e64 v143, null, s21, v47, vcc_lo           // 000000003af4: d5207c8f 01aa5e15
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 000000003afc: bf870122
	v_add_co_u32 v46, vcc_lo, v142, v110                       // 000000003b00: d7006a2e 0202dd8e
	s_wait_alu depctr_va_vcc(0)                                // 000000003b08: bf88ff9d
	v_add_co_ci_u32_e64 v47, null, v143, v111, vcc_lo          // 000000003b0c: d5207c2f 01aadf8f
	v_cmp_u_f32_e32 vcc_lo, v137, v137                         // 000000003b14: 7c311389
	s_clause 0x2                                               // 000000003b18: bf850002
	global_store_d16_hi_b16 v[128:129], v132, off              // 000000003b1c: ee09407c 42000000 00000080
	global_store_d16_hi_b16 v[134:135], v133, off              // 000000003b28: ee09407c 42800000 00000086
	global_store_d16_hi_b16 v[46:47], v136, off                // 000000003b34: ee09407c 44000000 0000002e
	v_bfe_u32 v132, v139, 16, 1                                // 000000003b40: d6100084 0205218b
	s_wait_alu depctr_va_vcc(0)                                // 000000003b48: bf88ff9d
	v_cndmask_b32_e32 v149, v149, v150, vcc_lo                 // 000000003b4c: 032b2d95
	v_add_co_u32 v136, vcc_lo, v142, s20                       // 000000003b50: d7006a88 0200298e
	s_wait_alu depctr_va_vcc(0)                                // 000000003b58: bf88ff9d
	v_add_co_ci_u32_e64 v137, null, s21, v143, vcc_lo          // 000000003b5c: d5207c89 01ab1e15
	v_add3_u32 v142, v132, v139, 0x7fff                        // 000000003b64: d655008e 03ff1784 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000003b70: bf870003
	v_add_co_u32 v132, vcc_lo, v136, v110                      // 000000003b74: d7006a84 0202dd88
	v_or_b32_e32 v143, 0x400000, v139                          // 000000003b7c: 391f16ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003b84: bf88ff9d
	v_add_co_ci_u32_e64 v133, null, v137, v111, vcc_lo         // 000000003b88: d5207c85 01aadf89
	v_cmp_u_f32_e32 vcc_lo, v139, v139                         // 000000003b90: 7c31178b
	v_bfe_u32 v139, v140, 16, 1                                // 000000003b94: d610008b 0205218c
	v_or_b32_e32 v150, 0x400000, v140                          // 000000003b9c: 392d18ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000003ba4: bf88ff9d
	v_cndmask_b32_e32 v155, v142, v143, vcc_lo                 // 000000003ba8: 03371f8e
	v_add_co_u32 v142, vcc_lo, v136, s20                       // 000000003bac: d7006a8e 02002988
	s_wait_alu depctr_va_vcc(0)                                // 000000003bb4: bf88ff9d
	v_add_co_ci_u32_e64 v143, null, s21, v137, vcc_lo          // 000000003bb8: d5207c8f 01ab1215
	v_add3_u32 v139, v139, v140, 0x7fff                        // 000000003bc0: d655008b 03ff198b 00007fff
	s_delay_alu instid0(valu_dep_3) | instskip(skip_1) | instid1(valu_dep_3)// 000000003bcc: bf8701a3
	v_add_co_u32 v136, vcc_lo, v142, v110                      // 000000003bd0: d7006a88 0202dd8e
	s_wait_alu depctr_va_vcc(0)                                // 000000003bd8: bf88ff9d
	v_add_co_ci_u32_e64 v137, null, v143, v111, vcc_lo         // 000000003bdc: d5207c89 01aadf8f
	v_cmp_u_f32_e32 vcc_lo, v140, v140                         // 000000003be4: 7c31198c
	s_wait_alu depctr_va_vcc(0)                                // 000000003be8: bf88ff9d
	v_cndmask_b32_e32 v156, v139, v150, vcc_lo                 // 000000003bec: 03392d8b
	v_add_co_u32 v140, vcc_lo, v142, s20                       // 000000003bf0: d7006a8c 0200298e
	v_bfe_u32 v139, v141, 16, 1                                // 000000003bf8: d610008b 0205218d
	s_wait_alu depctr_va_vcc(0)                                // 000000003c00: bf88ff9d
	v_add_co_ci_u32_e64 v142, null, s21, v143, vcc_lo          // 000000003c04: d5207c8e 01ab1e15
	v_or_b32_e32 v150, v148, v138                              // 000000003c0c: 392d1594
	v_add_co_u32 v138, vcc_lo, v140, v110                      // 000000003c10: d7006a8a 0202dd8c
	v_add3_u32 v143, v139, v141, 0x7fff                        // 000000003c18: d655008f 03ff1b8b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000003c24: bf88ff9d
	v_add_co_ci_u32_e64 v139, null, v142, v111, vcc_lo         // 000000003c28: d5207c8b 01aadf8e
	v_cmp_u_f32_e32 vcc_lo, v141, v141                         // 000000003c30: 7c311b8d
	v_add_co_u32 v140, s0, v140, s20                           // 000000003c34: d700008c 0200298c
	s_wait_alu depctr_va_sdst(0)                               // 000000003c3c: bf88f19f
	v_add_co_ci_u32_e64 v141, null, s21, v142, s0              // 000000003c40: d5207c8d 00031c15
	s_wait_alu depctr_va_vcc(0)                                // 000000003c48: bf88ff9d
	v_cndmask_b32_e32 v157, v143, v157, vcc_lo                 // 000000003c4c: 033b3b8f
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[150:151]              // 000000003c50: 7ca92c10
	s_wait_alu depctr_va_vcc(0)                                // 000000003c54: bf88ff9d
	v_dual_cndmask_b32 v143, 0, v151 :: v_dual_cndmask_b32 v142, 0, v150// 000000003c58: ca532e80 8f8f2c80
	v_add_co_u32 v140, vcc_lo, v140, v110                      // 000000003c60: d7006a8c 0202dd8c
	s_wait_alu depctr_va_vcc(0)                                // 000000003c68: bf88ff9d
	v_add_co_ci_u32_e64 v141, null, v141, v111, vcc_lo         // 000000003c6c: d5207c8d 01aadf8d
	s_delay_alu instid0(valu_dep_3)                            // 000000003c74: bf870003
	v_lshlrev_b64_e32 v[142:143], 2, v[142:143]                // 000000003c78: 3f1d1c82
	s_clause 0x3                                               // 000000003c7c: bf850003
	global_store_d16_hi_b16 v[132:133], v149, off              // 000000003c80: ee09407c 4a800000 00000084
	global_store_d16_hi_b16 v[136:137], v155, off              // 000000003c8c: ee09407c 4d800000 00000088
	global_store_d16_hi_b16 v[138:139], v156, off              // 000000003c98: ee09407c 4e000000 0000008a
	global_store_d16_hi_b16 v[140:141], v157, off              // 000000003ca4: ee09407c 4e800000 0000008c
	v_mov_b32_e32 v149, s23                                    // 000000003cb0: 7f2a0217
	v_add_co_u32 v142, vcc_lo, s4, v142                        // 000000003cb4: d7006a8e 02031c04
	s_wait_alu depctr_va_vcc(0)                                // 000000003cbc: bf88ff9d
	v_add_co_ci_u32_e64 v143, null, s5, v143, vcc_lo           // 000000003cc0: d5207c8f 01ab1e05
	global_load_u8 v125, v[124:125], off                       // 000000003cc8: ee04007c 0000007d 0000007c
	global_load_b32 v124, v[142:143], off                      // 000000003cd4: ee05007c 0000007c 0000008e
	s_wait_loadcnt 0x1                                         // 000000003ce0: bfc00001
	v_lshlrev_b32_e32 v156, 23, v125                           // 000000003ce4: 3138fa97
	s_wait_loadcnt 0x0                                         // 000000003ce8: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000003cec: bf870091
	v_mul_f32_e32 v125, v124, v156                             // 000000003cf0: 10fb397c
	v_cmp_class_f32_e64 s0, v125, 0x198                        // 000000003cf4: d47e0000 0201ff7d 00000198
	v_mul_f32_e32 v155, v32, v125                              // 000000003d00: 1136fb20
	s_xor_b32 s1, s0, -1                                       // 000000003d04: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d08: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003d0c: be802001
	s_cbranch_execnz 1788                                      // 000000003d10: bfa606fc <packed_folded_w4a8+0x3e04>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d14: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003d18: 8c7e007e
	v_or_b32_e32 v124, v144, v148                              // 000000003d1c: 38f92990
	v_mov_b32_e32 v125, v149                                   // 000000003d20: 7efa0395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003d24: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[124:125]              // 000000003d28: 7ca8f810
	s_wait_alu depctr_va_vcc(0)                                // 000000003d2c: bf88ff9d
	v_dual_cndmask_b32 v125, 0, v125 :: v_dual_cndmask_b32 v124, 0, v124// 000000003d30: ca52fa80 7d7cf880
	v_lshlrev_b64_e32 v[124:125], 2, v[124:125]                // 000000003d38: 3ef8f882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003d3c: bf870121
	v_add_co_u32 v124, vcc_lo, s4, v124                        // 000000003d40: d7006a7c 0202f804
	s_wait_alu depctr_va_vcc(0)                                // 000000003d48: bf88ff9d
	v_add_co_ci_u32_e64 v125, null, s5, v125, vcc_lo           // 000000003d4c: d5207c7d 01aafa05
	global_load_b32 v32, v[124:125], off                       // 000000003d54: ee05007c 00000020 0000007c
	s_wait_loadcnt 0x0                                         // 000000003d60: bfc00000
	v_mul_f32_e32 v144, v32, v156                              // 000000003d64: 11213920
	s_delay_alu instid0(valu_dep_1)                            // 000000003d68: bf870001
	v_cmp_class_f32_e64 s0, v144, 0x198                        // 000000003d6c: d47e0000 0201ff90 00000198
	v_mul_f32_e32 v157, v33, v144                              // 000000003d78: 113b2121
	s_xor_b32 s1, s0, -1                                       // 000000003d7c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d80: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003d84: be802001
	s_cbranch_execnz 1775                                      // 000000003d88: bfa606ef <packed_folded_w4a8+0x3e48>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003d8c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003d90: 8c7e007e
	v_or_b32_e32 v32, v145, v148                               // 000000003d94: 38412991
	v_mov_b32_e32 v33, v149                                    // 000000003d98: 7e420395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003d9c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[32:33]                // 000000003da0: 7ca84010
	s_wait_alu depctr_va_vcc(0)                                // 000000003da4: bf88ff9d
	v_dual_cndmask_b32 v33, 0, v33 :: v_dual_cndmask_b32 v32, 0, v32// 000000003da8: ca524280 21204080
	v_lshlrev_b64_e32 v[32:33], 2, v[32:33]                    // 000000003db0: 3e404082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003db4: bf870121
	v_add_co_u32 v32, vcc_lo, s4, v32                          // 000000003db8: d7006a20 02024004
	s_wait_alu depctr_va_vcc(0)                                // 000000003dc0: bf88ff9d
	v_add_co_ci_u32_e64 v33, null, s5, v33, vcc_lo             // 000000003dc4: d5207c21 01aa4205
	global_load_b32 v144, v[32:33], off                        // 000000003dcc: ee05007c 00000090 00000020
	s_wait_loadcnt 0x0                                         // 000000003dd8: bfc00000
	v_mul_f32_e32 v145, v144, v156                             // 000000003ddc: 11233990
	s_delay_alu instid0(valu_dep_1)                            // 000000003de0: bf870001
	v_cmp_class_f32_e64 s0, v145, 0x198                        // 000000003de4: d47e0000 0201ff91 00000198
	v_mul_f32_e32 v158, v34, v145                              // 000000003df0: 113d2322
	s_xor_b32 s1, s0, -1                                       // 000000003df4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003df8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003dfc: be802001
	s_cbranch_execnz 1762                                      // 000000003e00: bfa606e2 <packed_folded_w4a8+0x3e8c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e04: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e08: 8c7e007e
	v_or_b32_e32 v144, v146, v148                              // 000000003e0c: 39212992
	v_mov_b32_e32 v145, v149                                   // 000000003e10: 7f220395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003e14: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[144:145]              // 000000003e18: 7ca92010
	s_wait_alu depctr_va_vcc(0)                                // 000000003e1c: bf88ff9d
	v_dual_cndmask_b32 v145, 0, v145 :: v_dual_cndmask_b32 v144, 0, v144// 000000003e20: ca532280 91912080
	v_lshlrev_b64_e32 v[144:145], 2, v[144:145]                // 000000003e28: 3f212082
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003e2c: bf870121
	v_add_co_u32 v144, vcc_lo, s4, v144                        // 000000003e30: d7006a90 02032004
	s_wait_alu depctr_va_vcc(0)                                // 000000003e38: bf88ff9d
	v_add_co_ci_u32_e64 v145, null, s5, v145, vcc_lo           // 000000003e3c: d5207c91 01ab2205
	global_load_b32 v34, v[144:145], off                       // 000000003e44: ee05007c 00000022 00000090
	s_wait_loadcnt 0x0                                         // 000000003e50: bfc00000
	v_mul_f32_e32 v146, v34, v156                              // 000000003e54: 11253922
	s_delay_alu instid0(valu_dep_1)                            // 000000003e58: bf870001
	v_cmp_class_f32_e64 s0, v146, 0x198                        // 000000003e5c: d47e0000 0201ff92 00000198
	v_mul_f32_e32 v159, v35, v146                              // 000000003e68: 113f2523
	s_xor_b32 s1, s0, -1                                       // 000000003e6c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e70: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003e74: be802001
	s_cbranch_execnz 1749                                      // 000000003e78: bfa606d5 <packed_folded_w4a8+0x3ed0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003e7c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003e80: 8c7e007e
	v_or_b32_e32 v34, v147, v148                               // 000000003e84: 38452993
	v_mov_b32_e32 v35, v149                                    // 000000003e88: 7e460395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003e8c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[34:35]                // 000000003e90: 7ca84410
	s_wait_alu depctr_va_vcc(0)                                // 000000003e94: bf88ff9d
	v_dual_cndmask_b32 v35, 0, v35 :: v_dual_cndmask_b32 v34, 0, v34// 000000003e98: ca524680 23224480
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]                    // 000000003ea0: 3e444482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003ea4: bf870121
	v_add_co_u32 v34, vcc_lo, s4, v34                          // 000000003ea8: d7006a22 02024404
	s_wait_alu depctr_va_vcc(0)                                // 000000003eb0: bf88ff9d
	v_add_co_ci_u32_e64 v35, null, s5, v35, vcc_lo             // 000000003eb4: d5207c23 01aa4605
	global_load_b32 v146, v[34:35], off                        // 000000003ebc: ee05007c 00000092 00000022
	s_wait_loadcnt 0x0                                         // 000000003ec8: bfc00000
	v_mul_f32_e32 v147, v146, v156                             // 000000003ecc: 11273992
	s_delay_alu instid0(valu_dep_1)                            // 000000003ed0: bf870001
	v_cmp_class_f32_e64 s0, v147, 0x198                        // 000000003ed4: d47e0000 0201ff93 00000198
	v_mul_f32_e32 v161, v36, v147                              // 000000003ee0: 11432724
	s_xor_b32 s1, s0, -1                                       // 000000003ee4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ee8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003eec: be802001
	s_cbranch_execnz 1736                                      // 000000003ef0: bfa606c8 <packed_folded_w4a8+0x3f14>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003ef4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003ef8: 8c7e007e
	v_or_b32_e32 v146, v152, v148                              // 000000003efc: 39252998
	v_mov_b32_e32 v147, v149                                   // 000000003f00: 7f260395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003f04: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[146:147]              // 000000003f08: 7ca92410
	s_wait_alu depctr_va_vcc(0)                                // 000000003f0c: bf88ff9d
	v_dual_cndmask_b32 v147, 0, v147 :: v_dual_cndmask_b32 v146, 0, v146// 000000003f10: ca532680 93932480
	v_lshlrev_b64_e32 v[146:147], 2, v[146:147]                // 000000003f18: 3f252482
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003f1c: bf870121
	v_add_co_u32 v146, vcc_lo, s4, v146                        // 000000003f20: d7006a92 02032404
	s_wait_alu depctr_va_vcc(0)                                // 000000003f28: bf88ff9d
	v_add_co_ci_u32_e64 v147, null, s5, v147, vcc_lo           // 000000003f2c: d5207c93 01ab2605
	global_load_b32 v36, v[146:147], off                       // 000000003f34: ee05007c 00000024 00000092
	s_wait_loadcnt 0x0                                         // 000000003f40: bfc00000
	v_mul_f32_e32 v152, v36, v156                              // 000000003f44: 11313924
	s_delay_alu instid0(valu_dep_1)                            // 000000003f48: bf870001
	v_cmp_class_f32_e64 s0, v152, 0x198                        // 000000003f4c: d47e0000 0201ff98 00000198
	v_mul_f32_e32 v162, v37, v152                              // 000000003f58: 11453125
	s_xor_b32 s1, s0, -1                                       // 000000003f5c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f60: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003f64: be802001
	s_cbranch_execnz 1723                                      // 000000003f68: bfa606bb <packed_folded_w4a8+0x3f58>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003f6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003f70: 8c7e007e
	v_or_b32_e32 v36, v153, v148                               // 000000003f74: 38492999
	v_mov_b32_e32 v37, v149                                    // 000000003f78: 7e4a0395
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003f7c: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[36:37]                // 000000003f80: 7ca84810
	s_wait_alu depctr_va_vcc(0)                                // 000000003f84: bf88ff9d
	v_dual_cndmask_b32 v37, 0, v37 :: v_dual_cndmask_b32 v36, 0, v36// 000000003f88: ca524a80 25244880
	v_lshlrev_b64_e32 v[36:37], 2, v[36:37]                    // 000000003f90: 3e484882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000003f94: bf870121
	v_add_co_u32 v36, vcc_lo, s4, v36                          // 000000003f98: d7006a24 02024804
	s_wait_alu depctr_va_vcc(0)                                // 000000003fa0: bf88ff9d
	v_add_co_ci_u32_e64 v37, null, s5, v37, vcc_lo             // 000000003fa4: d5207c25 01aa4a05
	global_load_b32 v152, v[36:37], off                        // 000000003fac: ee05007c 00000098 00000024
	s_wait_loadcnt 0x0                                         // 000000003fb8: bfc00000
	v_mul_f32_e32 v153, v152, v156                             // 000000003fbc: 11333998
	s_delay_alu instid0(valu_dep_1)                            // 000000003fc0: bf870001
	v_cmp_class_f32_e64 s0, v153, 0x198                        // 000000003fc4: d47e0000 0201ff99 00000198
	v_mul_f32_e32 v163, v38, v153                              // 000000003fd0: 11473326
	s_xor_b32 s1, s0, -1                                       // 000000003fd4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fd8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000003fdc: be802001
	s_cbranch_execnz 1710                                      // 000000003fe0: bfa606ae <packed_folded_w4a8+0x3f9c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003fe4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000003fe8: 8c7e007e
	v_or_b32_e32 v148, v154, v148                              // 000000003fec: 3929299a
	s_delay_alu instid0(valu_dep_1) | instskip(skip_2) | instid1(valu_dep_1)// 000000003ff0: bf8700b1
	v_cmp_gt_i64_e32 vcc_lo, s[16:17], v[148:149]              // 000000003ff4: 7ca92810
	s_wait_alu depctr_va_vcc(0)                                // 000000003ff8: bf88ff9d
	v_dual_cndmask_b32 v149, 0, v149 :: v_dual_cndmask_b32 v148, 0, v148// 000000003ffc: ca532a80 95952880
	v_lshlrev_b64_e32 v[148:149], 2, v[148:149]                // 000000004004: 3f292882
	s_delay_alu instid0(valu_dep_1) | instskip(skip_1) | instid1(valu_dep_2)// 000000004008: bf870121
	v_add_co_u32 v148, vcc_lo, s4, v148                        // 00000000400c: d7006a94 02032804
	s_wait_alu depctr_va_vcc(0)                                // 000000004014: bf88ff9d
	v_add_co_ci_u32_e64 v149, null, s5, v149, vcc_lo           // 000000004018: d5207c95 01ab2a05
	global_load_b32 v38, v[148:149], off                       // 000000004020: ee05007c 00000026 00000094
	s_wait_loadcnt 0x0                                         // 00000000402c: bfc00000
	v_mul_f32_e32 v152, v38, v156                              // 000000004030: 11313926
	s_delay_alu instid0(valu_dep_1)                            // 000000004034: bf870001
	v_cmp_class_f32_e64 s0, v152, 0x198                        // 000000004038: d47e0000 0201ff98 00000198
	v_mul_f32_e32 v164, v39, v152                              // 000000004044: 11493127
	s_xor_b32 s1, s0, -1                                       // 000000004048: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000404c: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004050: be802001
	s_cbranch_execnz 1698                                      // 000000004054: bfa606a2 <packed_folded_w4a8+0x3fe0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004058: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000405c: 8c7e007e
	v_mul_lo_u32 v152, s19, v150                               // 000000004060: d72c0098 02032c13
	v_mul_lo_u32 v151, s18, v151                               // 000000004068: d72c0097 02032e12
	v_mad_co_u64_u32 v[38:39], null, s18, v150, 0              // 000000004070: d6fe7c26 02032c12
	v_bfe_u32 v150, v155, 16, 1                                // 000000004078: d6100096 0205219b
	v_or_b32_e32 v153, 0x400000, v155                          // 000000004080: 393336ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v155, v155                         // 000000004088: 7c31379b
	v_or_b32_e32 v154, 0x400000, v157                          // 00000000408c: 39353aff 00400000
	v_or_b32_e32 v165, 0x400000, v159                          // 000000004094: 394b3eff 00400000
	v_add3_u32 v150, v150, v155, 0x7fff                        // 00000000409c: d6550096 03ff3796 00007fff
	v_or_b32_e32 v155, 0x400000, v158                          // 0000000040a8: 39373cff 00400000
	v_add3_u32 v39, v39, v151, v152                            // 0000000040b0: d6550027 06632f27
	v_bfe_u32 v151, v157, 16, 1                                // 0000000040b8: d6100097 0205219d
	v_bfe_u32 v152, v158, 16, 1                                // 0000000040c0: d6100098 0205219e
	s_wait_alu depctr_va_vcc(0)                                // 0000000040c8: bf88ff9d
	v_cndmask_b32_e32 v150, v150, v153, vcc_lo                 // 0000000040cc: 032d3396
	v_bfe_u32 v166, v161, 16, 1                                // 0000000040d0: d61000a6 020521a1
	v_lshlrev_b64_e32 v[38:39], 1, v[38:39]                    // 0000000040d8: 3e4c4c81
	v_add3_u32 v151, v151, v157, 0x7fff                        // 0000000040dc: d6550097 03ff3b97 00007fff
	v_add3_u32 v156, v152, v158, 0x7fff                        // 0000000040e8: d655009c 03ff3d98 00007fff
	v_or_b32_e32 v167, 0x400000, v161                          // 0000000040f4: 394f42ff 00400000
	v_add3_u32 v166, v166, v161, 0x7fff                        // 0000000040fc: d65500a6 03ff43a6 00007fff
	v_or_b32_e32 v168, 0x400000, v163                          // 000000004108: 395146ff 00400000
	v_add_co_u32 v38, vcc_lo, s2, v38                          // 000000004110: d7006a26 02024c02
	s_wait_alu depctr_va_vcc(0)                                // 000000004118: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s3, v39, vcc_lo             // 00000000411c: d5207c27 01aa4e03
	v_cmp_u_f32_e32 vcc_lo, v157, v157                         // 000000004124: 7c313b9d
	v_or_b32_e32 v64, v64, v160                                // 000000004128: 38814140
	v_or_b32_e32 v169, 0x400000, v164                          // 00000000412c: 395348ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004134: bf88ff9d
	v_cndmask_b32_e32 v154, v151, v154, vcc_lo                 // 000000004138: 03353597
	v_add_co_u32 v152, vcc_lo, v38, v110                       // 00000000413c: d7006a98 0202dd26
	s_wait_alu depctr_va_vcc(0)                                // 000000004144: bf88ff9d
	v_add_co_ci_u32_e64 v153, null, v39, v111, vcc_lo          // 000000004148: d5207c99 01aadf27
	v_add_co_u32 v38, vcc_lo, v38, s20                         // 000000004150: d7006a26 02002926
	s_wait_alu depctr_va_vcc(0)                                // 000000004158: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v39, vcc_lo            // 00000000415c: d5207c27 01aa4e15
	global_store_d16_hi_b16 v[152:153], v150, off              // 000000004164: ee09407c 4b000000 00000098
	v_add_co_u32 v150, vcc_lo, v38, v110                       // 000000004170: d7006a96 0202dd26
	s_wait_alu depctr_va_vcc(0)                                // 000000004178: bf88ff9d
	v_add_co_ci_u32_e64 v151, null, v39, v111, vcc_lo          // 00000000417c: d5207c97 01aadf27
	v_cmp_u_f32_e32 vcc_lo, v158, v158                         // 000000004184: 7c313d9e
	s_wait_alu depctr_va_vcc(0)                                // 000000004188: bf88ff9d
	v_cndmask_b32_e32 v155, v156, v155, vcc_lo                 // 00000000418c: 0337379c
	v_bfe_u32 v156, v159, 16, 1                                // 000000004190: d610009c 0205219f
	v_add_co_u32 v38, vcc_lo, v38, s20                         // 000000004198: d7006a26 02002926
	s_wait_alu depctr_va_vcc(0)                                // 0000000041a0: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, s21, v39, vcc_lo            // 0000000041a4: d5207c27 01aa4e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000041ac: bf870193
	v_add3_u32 v158, v156, v159, 0x7fff                        // 0000000041b0: d655009e 03ff3f9c 00007fff
	v_add_co_u32 v156, vcc_lo, v38, v110                       // 0000000041bc: d7006a9c 0202dd26
	s_wait_alu depctr_va_vcc(0)                                // 0000000041c4: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000041c8: bf870003
	v_add_co_ci_u32_e64 v157, null, v39, v111, vcc_lo          // 0000000041cc: d5207c9d 01aadf27
	v_cmp_u_f32_e32 vcc_lo, v159, v159                         // 0000000041d4: 7c313f9f
	s_wait_alu depctr_va_vcc(0)                                // 0000000041d8: bf88ff9d
	v_cndmask_b32_e32 v158, v158, v165, vcc_lo                 // 0000000041dc: 033d4b9e
	v_add_co_u32 v159, vcc_lo, v38, s20                        // 0000000041e0: d7006a9f 02002926
	s_wait_alu depctr_va_vcc(0)                                // 0000000041e8: bf88ff9d
	v_add_co_ci_u32_e64 v165, null, s21, v39, vcc_lo           // 0000000041ec: d5207ca5 01aa4e15
	s_delay_alu instid0(valu_dep_2) | instskip(skip_1) | instid1(valu_dep_2)// 0000000041f4: bf870122
	v_add_co_u32 v38, vcc_lo, v159, v110                       // 0000000041f8: d7006a26 0202dd9f
	s_wait_alu depctr_va_vcc(0)                                // 000000004200: bf88ff9d
	v_add_co_ci_u32_e64 v39, null, v165, v111, vcc_lo          // 000000004204: d5207c27 01aadfa5
	v_cmp_u_f32_e32 vcc_lo, v161, v161                         // 00000000420c: 7c3143a1
	s_clause 0x2                                               // 000000004210: bf850002
	global_store_d16_hi_b16 v[150:151], v154, off              // 000000004214: ee09407c 4d000000 00000096
	global_store_d16_hi_b16 v[156:157], v155, off              // 000000004220: ee09407c 4d800000 0000009c
	global_store_d16_hi_b16 v[38:39], v158, off                // 00000000422c: ee09407c 4f000000 00000026
	v_bfe_u32 v154, v162, 16, 1                                // 000000004238: d610009a 020521a2
	s_wait_alu depctr_va_vcc(0)                                // 000000004240: bf88ff9d
	v_cndmask_b32_e32 v166, v166, v167, vcc_lo                 // 000000004244: 034d4fa6
	v_add_co_u32 v158, vcc_lo, v159, s20                       // 000000004248: d7006a9e 0200299f
	s_wait_alu depctr_va_vcc(0)                                // 000000004250: bf88ff9d
	v_add_co_ci_u32_e64 v159, null, s21, v165, vcc_lo          // 000000004254: d5207c9f 01ab4a15
	v_add3_u32 v161, v154, v162, 0x7fff                        // 00000000425c: d65500a1 03ff459a 00007fff
	s_delay_alu instid0(valu_dep_3)                            // 000000004268: bf870003
	v_add_co_u32 v154, vcc_lo, v158, v110                      // 00000000426c: d7006a9a 0202dd9e
	v_or_b32_e32 v165, 0x400000, v162                          // 000000004274: 394b44ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000427c: bf88ff9d
	v_add_co_ci_u32_e64 v155, null, v159, v111, vcc_lo         // 000000004280: d5207c9b 01aadf9f
	v_cmp_u_f32_e32 vcc_lo, v162, v162                         // 000000004288: 7c3145a2
	s_wait_alu depctr_va_vcc(0)                                // 00000000428c: bf88ff9d
	v_cndmask_b32_e32 v162, v161, v165, vcc_lo                 // 000000004290: 03454ba1
	v_add_co_u32 v165, vcc_lo, v158, s20                       // 000000004294: d7006aa5 0200299e
	v_bfe_u32 v161, v163, 16, 1                                // 00000000429c: d61000a1 020521a3
	s_wait_alu depctr_va_vcc(0)                                // 0000000042a4: bf88ff9d
	v_add_co_ci_u32_e64 v167, null, s21, v159, vcc_lo          // 0000000042a8: d5207ca7 01ab3e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 0000000042b0: bf870193
	v_add_co_u32 v158, vcc_lo, v165, v110                      // 0000000042b4: d7006a9e 0202dda5
	v_add3_u32 v161, v161, v163, 0x7fff                        // 0000000042bc: d65500a1 03ff47a1 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 0000000042c8: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 0000000042cc: bf870003
	v_add_co_ci_u32_e64 v159, null, v167, v111, vcc_lo         // 0000000042d0: d5207c9f 01aadfa7
	v_cmp_u_f32_e32 vcc_lo, v163, v163                         // 0000000042d8: 7c3147a3
	s_wait_alu depctr_va_vcc(0)                                // 0000000042dc: bf88ff9d
	v_cndmask_b32_e32 v163, v161, v168, vcc_lo                 // 0000000042e0: 034751a1
	v_add_co_u32 v165, vcc_lo, v165, s20                       // 0000000042e4: d7006aa5 020029a5
	v_bfe_u32 v161, v164, 16, 1                                // 0000000042ec: d61000a1 020521a4
	s_wait_alu depctr_va_vcc(0)                                // 0000000042f4: bf88ff9d
	v_add_co_ci_u32_e64 v167, null, s21, v167, vcc_lo          // 0000000042f8: d5207ca7 01ab4e15
	s_delay_alu instid0(valu_dep_3) | instskip(next) | instid1(valu_dep_3)// 000000004300: bf870193
	v_add_co_u32 v160, vcc_lo, v165, v110                      // 000000004304: d7006aa0 0202dda5
	v_add3_u32 v168, v161, v164, 0x7fff                        // 00000000430c: d65500a8 03ff49a1 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004318: bf88ff9d
	s_delay_alu instid0(valu_dep_3)                            // 00000000431c: bf870003
	v_add_co_ci_u32_e64 v161, null, v167, v111, vcc_lo         // 000000004320: d5207ca1 01aadfa7
	v_cmp_u_f32_e32 vcc_lo, v164, v164                         // 000000004328: 7c3149a4
	v_add_co_u32 v165, s0, v165, s20                           // 00000000432c: d70000a5 020029a5
	s_wait_alu depctr_va_sdst(0)                               // 000000004334: bf88f19f
	v_add_co_ci_u32_e64 v167, null, s21, v167, s0              // 000000004338: d5207ca7 00034e15
	s_wait_alu depctr_va_vcc(0)                                // 000000004340: bf88ff9d
	v_cndmask_b32_e32 v164, v168, v169, vcc_lo                 // 000000004344: 034953a8
	v_cmp_gt_i64_e32 vcc_lo, s[18:19], v[64:65]                // 000000004348: 7ca88012
	s_wait_alu depctr_va_vcc(0)                                // 00000000434c: bf88ff9d
	v_dual_cndmask_b32 v169, 0, v64 :: v_dual_cndmask_b32 v168, 0, v65// 000000004350: ca528080 a9a88280
	v_add_co_u32 v64, vcc_lo, v165, v110                       // 000000004358: d7006a40 0202dda5
	s_wait_alu depctr_va_vcc(0)                                // 000000004360: bf88ff9d
	v_add_co_ci_u32_e64 v65, null, v167, v111, vcc_lo          // 000000004364: d5207c41 01aadfa7
	s_delay_alu instid0(valu_dep_3)                            // 00000000436c: bf870003
	v_add_co_u32 v110, vcc_lo, s14, v169                       // 000000004370: d7006a6e 0203520e
	s_wait_alu depctr_va_vcc(0)                                // 000000004378: bf88ff9d
	v_add_co_ci_u32_e64 v111, null, s15, v168, vcc_lo          // 00000000437c: d5207c6f 01ab500f
	s_clause 0x3                                               // 000000004384: bf850003
	global_store_d16_hi_b16 v[154:155], v166, off              // 000000004388: ee09407c 53000000 0000009a
	global_store_d16_hi_b16 v[158:159], v162, off              // 000000004394: ee09407c 51000000 0000009e
	global_store_d16_hi_b16 v[160:161], v163, off              // 0000000043a0: ee09407c 51800000 000000a0
	global_store_d16_hi_b16 v[64:65], v164, off                // 0000000043ac: ee09407c 52000000 00000040
	global_load_u8 v163, v[110:111], off                       // 0000000043b8: ee04007c 000000a3 0000006e
	global_load_b32 v162, v[66:67], off                        // 0000000043c4: ee05007c 000000a2 00000042
	s_wait_loadcnt 0x1                                         // 0000000043d0: bfc00001
	v_lshlrev_b32_e32 v67, 23, v163                            // 0000000043d4: 30874697
	s_wait_loadcnt 0x0                                         // 0000000043d8: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000043dc: bf870091
	v_mul_f32_e32 v66, v162, v67                               // 0000000043e0: 108487a2
	v_cmp_class_f32_e64 s0, v66, 0x198                         // 0000000043e4: d47e0000 0201ff42 00000198
	v_mul_f32_e32 v66, v24, v66                                // 0000000043f0: 10848518
	s_xor_b32 s1, s0, -1                                       // 0000000043f4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043f8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000043fc: be802001
	s_cbranch_execnz 1480                                      // 000000004400: bfa605c8 <packed_folded_w4a8+0x4024>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004404: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004408: 8c7e007e
	global_load_b32 v68, v[68:69], off                         // 00000000440c: ee05007c 00000044 00000044
	s_wait_loadcnt 0x0                                         // 000000004418: bfc00000
	v_mul_f32_e32 v24, v68, v67                                // 00000000441c: 10308744
	s_delay_alu instid0(valu_dep_1)                            // 000000004420: bf870001
	v_cmp_class_f32_e64 s0, v24, 0x198                         // 000000004424: d47e0000 0201ff18 00000198
	v_mul_f32_e32 v24, v25, v24                                // 000000004430: 10303119
	s_xor_b32 s1, s0, -1                                       // 000000004434: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004438: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000443c: be802001
	s_cbranch_execnz 1481                                      // 000000004440: bfa605c9 <packed_folded_w4a8+0x4068>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004444: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004448: 8c7e007e
	global_load_b32 v56, v[56:57], off                         // 00000000444c: ee05007c 00000038 00000038
	s_wait_loadcnt 0x0                                         // 000000004458: bfc00000
	v_mul_f32_e32 v25, v56, v67                                // 00000000445c: 10328738
	s_delay_alu instid0(valu_dep_1)                            // 000000004460: bf870001
	v_cmp_class_f32_e64 s0, v25, 0x198                         // 000000004464: d47e0000 0201ff19 00000198
	v_mul_f32_e32 v25, v26, v25                                // 000000004470: 1032331a
	s_xor_b32 s1, s0, -1                                       // 000000004474: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004478: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000447c: be802001
	s_cbranch_execnz 1482                                      // 000000004480: bfa605ca <packed_folded_w4a8+0x40ac>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004484: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004488: 8c7e007e
	global_load_b32 v56, v[70:71], off                         // 00000000448c: ee05007c 00000038 00000046
	s_wait_loadcnt 0x0                                         // 000000004498: bfc00000
	v_mul_f32_e32 v26, v56, v67                                // 00000000449c: 10348738
	s_delay_alu instid0(valu_dep_1)                            // 0000000044a0: bf870001
	v_cmp_class_f32_e64 s0, v26, 0x198                         // 0000000044a4: d47e0000 0201ff1a 00000198
	v_mul_f32_e32 v26, v27, v26                                // 0000000044b0: 1034351b
	s_xor_b32 s1, s0, -1                                       // 0000000044b4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044b8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000044bc: be802001
	s_cbranch_execnz 1483                                      // 0000000044c0: bfa605cb <packed_folded_w4a8+0x40f0>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000044c8: 8c7e007e
	global_load_b32 v56, v[58:59], off                         // 0000000044cc: ee05007c 00000038 0000003a
	s_wait_loadcnt 0x0                                         // 0000000044d8: bfc00000
	v_mul_f32_e32 v27, v56, v67                                // 0000000044dc: 10368738
	s_delay_alu instid0(valu_dep_1)                            // 0000000044e0: bf870001
	v_cmp_class_f32_e64 s0, v27, 0x198                         // 0000000044e4: d47e0000 0201ff1b 00000198
	v_mul_f32_e32 v27, v28, v27                                // 0000000044f0: 1036371c
	s_xor_b32 s1, s0, -1                                       // 0000000044f4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000044f8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000044fc: be802001
	s_cbranch_execnz 1484                                      // 000000004500: bfa605cc <packed_folded_w4a8+0x4134>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004504: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004508: 8c7e007e
	global_load_b32 v56, v[72:73], off                         // 00000000450c: ee05007c 00000038 00000048
	s_wait_loadcnt 0x0                                         // 000000004518: bfc00000
	v_mul_f32_e32 v28, v56, v67                                // 00000000451c: 10388738
	s_delay_alu instid0(valu_dep_1)                            // 000000004520: bf870001
	v_cmp_class_f32_e64 s0, v28, 0x198                         // 000000004524: d47e0000 0201ff1c 00000198
	v_mul_f32_e32 v28, v29, v28                                // 000000004530: 1038391d
	s_xor_b32 s1, s0, -1                                       // 000000004534: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004538: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000453c: be802001
	s_cbranch_execnz 1485                                      // 000000004540: bfa605cd <packed_folded_w4a8+0x4178>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004544: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004548: 8c7e007e
	global_load_b32 v56, v[60:61], off                         // 00000000454c: ee05007c 00000038 0000003c
	s_wait_loadcnt 0x0                                         // 000000004558: bfc00000
	v_mul_f32_e32 v29, v56, v67                                // 00000000455c: 103a8738
	s_delay_alu instid0(valu_dep_1)                            // 000000004560: bf870001
	v_cmp_class_f32_e64 s0, v29, 0x198                         // 000000004564: d47e0000 0201ff1d 00000198
	v_mul_f32_e32 v29, v30, v29                                // 000000004570: 103a3b1e
	s_xor_b32 s1, s0, -1                                       // 000000004574: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004578: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 00000000457c: be802001
	s_cbranch_execnz 1486                                      // 000000004580: bfa605ce <packed_folded_w4a8+0x41bc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004584: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004588: 8c7e007e
	global_load_b32 v56, v[74:75], off                         // 00000000458c: ee05007c 00000038 0000004a
	s_wait_loadcnt 0x0                                         // 000000004598: bfc00000
	v_mul_f32_e32 v30, v56, v67                                // 00000000459c: 103c8738
	s_delay_alu instid0(valu_dep_1)                            // 0000000045a0: bf870001
	v_cmp_class_f32_e64 s0, v30, 0x198                         // 0000000045a4: d47e0000 0201ff1e 00000198
	v_mul_f32_e32 v30, v31, v30                                // 0000000045b0: 103c3d1f
	s_xor_b32 s1, s0, -1                                       // 0000000045b4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045b8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000045bc: be802001
	s_cbranch_execnz 1487                                      // 0000000045c0: bfa605cf <packed_folded_w4a8+0x4200>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000045c4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000045c8: 8c7e007e
	v_bfe_u32 v31, v66, 16, 1                                  // 0000000045cc: d610001f 02052142
	v_or_b32_e32 v56, 0x400000, v66                            // 0000000045d4: 387084ff 00400000
	v_bfe_u32 v57, v24, 16, 1                                  // 0000000045dc: d6100039 02052118
	v_cmp_u_f32_e32 vcc_lo, v66, v66                           // 0000000045e4: 7c308542
	v_or_b32_e32 v58, 0x400000, v24                            // 0000000045e8: 387430ff 00400000
	v_add3_u32 v31, v31, v66, 0x7fff                           // 0000000045f0: d655001f 03fe851f 00007fff
	v_bfe_u32 v59, v25, 16, 1                                  // 0000000045fc: d610003b 02052119
	v_add3_u32 v57, v57, v24, 0x7fff                           // 000000004604: d6550039 03fe3139 00007fff
	v_or_b32_e32 v60, 0x400000, v25                            // 000000004610: 387832ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004618: bf88ff9d
	v_cndmask_b32_e32 v31, v31, v56, vcc_lo                    // 00000000461c: 023e711f
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 000000004620: 7c303118
	v_bfe_u32 v56, v26, 16, 1                                  // 000000004624: d6100038 0205211a
	v_add3_u32 v59, v59, v25, 0x7fff                           // 00000000462c: d655003b 03fe333b 00007fff
	global_store_d16_hi_b16 v[62:63], v31, off offset:32       // 000000004638: ee09407c 0f800000 0000203e
	s_wait_alu depctr_va_vcc(0)                                // 000000004644: bf88ff9d
	v_cndmask_b32_e32 v24, v57, v58, vcc_lo                    // 000000004648: 02307539
	v_cmp_u_f32_e32 vcc_lo, v25, v25                           // 00000000464c: 7c303319
	v_bfe_u32 v31, v27, 16, 1                                  // 000000004650: d610001f 0205211b
	v_or_b32_e32 v57, 0x400000, v29                            // 000000004658: 38723aff 00400000
	v_or_b32_e32 v58, 0x400000, v30                            // 000000004660: 38743cff 00400000
	global_store_d16_hi_b16 v[78:79], v24, off offset:32       // 000000004668: ee09407c 0c000000 0000204e
	s_wait_alu depctr_va_vcc(0)                                // 000000004674: bf88ff9d
	v_cndmask_b32_e32 v25, v59, v60, vcc_lo                    // 000000004678: 0232793b
	v_add3_u32 v24, v56, v26, 0x7fff                           // 00000000467c: d6550018 03fe3538 00007fff
	v_or_b32_e32 v56, 0x400000, v26                            // 000000004688: 387034ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v26, v26                           // 000000004690: 7c30351a
	v_bfe_u32 v26, v28, 16, 1                                  // 000000004694: d610001a 0205211c
	global_store_d16_hi_b16 v[82:83], v25, off offset:32       // 00000000469c: ee09407c 0c800000 00002052
	v_add3_u32 v25, v31, v27, 0x7fff                           // 0000000046a8: d6550019 03fe371f 00007fff
	v_or_b32_e32 v31, 0x400000, v27                            // 0000000046b4: 383e36ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000046bc: bf88ff9d
	v_cndmask_b32_e32 v24, v24, v56, vcc_lo                    // 0000000046c0: 02307118
	v_cmp_u_f32_e32 vcc_lo, v27, v27                           // 0000000046c4: 7c30371b
	v_bfe_u32 v56, v29, 16, 1                                  // 0000000046c8: d6100038 0205211d
	v_add3_u32 v26, v26, v28, 0x7fff                           // 0000000046d0: d655001a 03fe391a 00007fff
	v_bfe_u32 v27, v30, 16, 1                                  // 0000000046dc: d610001b 0205211e
	s_wait_alu depctr_va_vcc(0)                                // 0000000046e4: bf88ff9d
	v_cndmask_b32_e32 v25, v25, v31, vcc_lo                    // 0000000046e8: 02323f19
	v_or_b32_e32 v31, 0x400000, v28                            // 0000000046ec: 383e38ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v28, v28                           // 0000000046f4: 7c30391c
	v_add3_u32 v56, v56, v29, 0x7fff                           // 0000000046f8: d6550038 03fe3b38 00007fff
	v_add3_u32 v27, v27, v30, 0x7fff                           // 000000004704: d655001b 03fe3d1b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004710: bf88ff9d
	v_cndmask_b32_e32 v26, v26, v31, vcc_lo                    // 000000004714: 02343f1a
	v_cmp_u_f32_e32 vcc_lo, v29, v29                           // 000000004718: 7c303b1d
	s_wait_alu depctr_va_vcc(0)                                // 00000000471c: bf88ff9d
	v_cndmask_b32_e32 v28, v56, v57, vcc_lo                    // 000000004720: 02387338
	v_cmp_u_f32_e32 vcc_lo, v30, v30                           // 000000004724: 7c303d1e
	s_wait_alu depctr_va_vcc(0)                                // 000000004728: bf88ff9d
	v_cndmask_b32_e32 v27, v27, v58, vcc_lo                    // 00000000472c: 0236751b
	s_clause 0x3                                               // 000000004730: bf850003
	global_store_d16_hi_b16 v[76:77], v24, off offset:32       // 000000004734: ee09407c 0c000000 0000204c
	global_store_d16_hi_b16 v[80:81], v25, off offset:32       // 000000004740: ee09407c 0c800000 00002050
	global_store_d16_hi_b16 v[84:85], v26, off offset:32       // 00000000474c: ee09407c 0d000000 00002054
	global_store_d16_hi_b16 v[86:87], v28, off offset:32       // 000000004758: ee09407c 0e000000 00002056
	global_store_d16_hi_b16 v[88:89], v27, off offset:32       // 000000004764: ee09407c 0d800000 00002058
	global_load_u8 v24, v[110:111], off                        // 000000004770: ee04007c 00000018 0000006e
	global_load_b32 v26, v[90:91], off                         // 00000000477c: ee05007c 0000001a 0000005a
	s_wait_loadcnt 0x1                                         // 000000004788: bfc00001
	v_lshlrev_b32_e32 v25, 23, v24                             // 00000000478c: 30323097
	s_wait_loadcnt 0x0                                         // 000000004790: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004794: bf870091
	v_mul_f32_e32 v24, v26, v25                                // 000000004798: 1030331a
	v_cmp_class_f32_e64 s0, v24, 0x198                         // 00000000479c: d47e0000 0201ff18 00000198
	v_mul_f32_e32 v24, v16, v24                                // 0000000047a8: 10303110
	s_xor_b32 s1, s0, -1                                       // 0000000047ac: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047b0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000047b4: be802001
	s_cbranch_execnz 1378                                      // 0000000047b8: bfa60562 <packed_folded_w4a8+0x4244>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000047c0: 8c7e007e
	global_load_b32 v26, v[92:93], off                         // 0000000047c4: ee05007c 0000001a 0000005c
	s_wait_loadcnt 0x0                                         // 0000000047d0: bfc00000
	v_mul_f32_e32 v16, v26, v25                                // 0000000047d4: 1020331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000047d8: bf870001
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 0000000047dc: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v17, v16                                // 0000000047e8: 10202111
	s_xor_b32 s1, s0, -1                                       // 0000000047ec: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047f0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000047f4: be802001
	s_cbranch_execnz 1379                                      // 0000000047f8: bfa60563 <packed_folded_w4a8+0x4288>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004800: 8c7e007e
	global_load_b32 v26, v[48:49], off                         // 000000004804: ee05007c 0000001a 00000030
	s_wait_loadcnt 0x0                                         // 000000004810: bfc00000
	v_mul_f32_e32 v17, v26, v25                                // 000000004814: 1022331a
	s_delay_alu instid0(valu_dep_1)                            // 000000004818: bf870001
	v_cmp_class_f32_e64 s0, v17, 0x198                         // 00000000481c: d47e0000 0201ff11 00000198
	v_mul_f32_e32 v17, v18, v17                                // 000000004828: 10222312
	s_xor_b32 s1, s0, -1                                       // 00000000482c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004830: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004834: be802001
	s_cbranch_execnz 1380                                      // 000000004838: bfa60564 <packed_folded_w4a8+0x42cc>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000483c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004840: 8c7e007e
	global_load_b32 v26, v[94:95], off                         // 000000004844: ee05007c 0000001a 0000005e
	s_wait_loadcnt 0x0                                         // 000000004850: bfc00000
	v_mul_f32_e32 v18, v26, v25                                // 000000004854: 1024331a
	s_delay_alu instid0(valu_dep_1)                            // 000000004858: bf870001
	v_cmp_class_f32_e64 s0, v18, 0x198                         // 00000000485c: d47e0000 0201ff12 00000198
	v_mul_f32_e32 v18, v19, v18                                // 000000004868: 10242513
	s_xor_b32 s1, s0, -1                                       // 00000000486c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004870: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004874: be802001
	s_cbranch_execnz 1381                                      // 000000004878: bfa60565 <packed_folded_w4a8+0x4310>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000487c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004880: 8c7e007e
	global_load_b32 v26, v[50:51], off                         // 000000004884: ee05007c 0000001a 00000032
	s_wait_loadcnt 0x0                                         // 000000004890: bfc00000
	v_mul_f32_e32 v19, v26, v25                                // 000000004894: 1026331a
	s_delay_alu instid0(valu_dep_1)                            // 000000004898: bf870001
	v_cmp_class_f32_e64 s0, v19, 0x198                         // 00000000489c: d47e0000 0201ff13 00000198
	v_mul_f32_e32 v19, v20, v19                                // 0000000048a8: 10262714
	s_xor_b32 s1, s0, -1                                       // 0000000048ac: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048b0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000048b4: be802001
	s_cbranch_execnz 1382                                      // 0000000048b8: bfa60566 <packed_folded_w4a8+0x4354>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048bc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000048c0: 8c7e007e
	global_load_b32 v26, v[96:97], off                         // 0000000048c4: ee05007c 0000001a 00000060
	s_wait_loadcnt 0x0                                         // 0000000048d0: bfc00000
	v_mul_f32_e32 v20, v26, v25                                // 0000000048d4: 1028331a
	s_delay_alu instid0(valu_dep_1)                            // 0000000048d8: bf870001
	v_cmp_class_f32_e64 s0, v20, 0x198                         // 0000000048dc: d47e0000 0201ff14 00000198
	v_mul_f32_e32 v20, v21, v20                                // 0000000048e8: 10282915
	s_xor_b32 s1, s0, -1                                       // 0000000048ec: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048f0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000048f4: be802001
	s_cbranch_execnz 1383                                      // 0000000048f8: bfa60567 <packed_folded_w4a8+0x4398>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048fc: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004900: 8c7e007e
	global_load_b32 v26, v[52:53], off                         // 000000004904: ee05007c 0000001a 00000034
	s_wait_loadcnt 0x0                                         // 000000004910: bfc00000
	v_mul_f32_e32 v21, v26, v25                                // 000000004914: 102a331a
	s_delay_alu instid0(valu_dep_1)                            // 000000004918: bf870001
	v_cmp_class_f32_e64 s0, v21, 0x198                         // 00000000491c: d47e0000 0201ff15 00000198
	v_mul_f32_e32 v21, v22, v21                                // 000000004928: 102a2b16
	s_xor_b32 s1, s0, -1                                       // 00000000492c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004930: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004934: be802001
	s_cbranch_execnz 1384                                      // 000000004938: bfa60568 <packed_folded_w4a8+0x43dc>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000493c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004940: 8c7e007e
	global_load_b32 v26, v[98:99], off                         // 000000004944: ee05007c 0000001a 00000062
	s_wait_loadcnt 0x0                                         // 000000004950: bfc00000
	v_mul_f32_e32 v22, v26, v25                                // 000000004954: 102c331a
	s_delay_alu instid0(valu_dep_1)                            // 000000004958: bf870001
	v_cmp_class_f32_e64 s0, v22, 0x198                         // 00000000495c: d47e0000 0201ff16 00000198
	v_mul_f32_e32 v22, v23, v22                                // 000000004968: 102c2d17
	s_xor_b32 s1, s0, -1                                       // 00000000496c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004970: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004974: be802001
	s_cbranch_execnz 1385                                      // 000000004978: bfa60569 <packed_folded_w4a8+0x4420>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000497c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004980: 8c7e007e
	v_bfe_u32 v23, v24, 16, 1                                  // 000000004984: d6100017 02052118
	v_or_b32_e32 v25, 0x400000, v24                            // 00000000498c: 383230ff 00400000
	v_bfe_u32 v26, v16, 16, 1                                  // 000000004994: d610001a 02052110
	v_cmp_u_f32_e32 vcc_lo, v24, v24                           // 00000000499c: 7c303118
	v_or_b32_e32 v27, 0x400000, v16                            // 0000000049a0: 383620ff 00400000
	v_add3_u32 v23, v23, v24, 0x7fff                           // 0000000049a8: d6550017 03fe3117 00007fff
	v_bfe_u32 v28, v17, 16, 1                                  // 0000000049b4: d610001c 02052111
	v_add3_u32 v26, v26, v16, 0x7fff                           // 0000000049bc: d655001a 03fe211a 00007fff
	v_or_b32_e32 v29, 0x400000, v17                            // 0000000049c8: 383a22ff 00400000
	v_bfe_u32 v24, v18, 16, 1                                  // 0000000049d0: d6100018 02052112
	s_wait_alu depctr_va_vcc(0)                                // 0000000049d8: bf88ff9d
	v_cndmask_b32_e32 v23, v23, v25, vcc_lo                    // 0000000049dc: 022e3317
	v_cmp_u_f32_e32 vcc_lo, v16, v16                           // 0000000049e0: 7c302110
	v_add3_u32 v25, v28, v17, 0x7fff                           // 0000000049e4: d6550019 03fe231c 00007fff
	global_store_d16_hi_b16 v[102:103], v23, off offset:32     // 0000000049f0: ee09407c 0b800000 00002066
	s_wait_alu depctr_va_vcc(0)                                // 0000000049fc: bf88ff9d
	v_cndmask_b32_e32 v16, v26, v27, vcc_lo                    // 000000004a00: 0220371a
	v_cmp_u_f32_e32 vcc_lo, v17, v17                           // 000000004a04: 7c302311
	v_bfe_u32 v23, v19, 16, 1                                  // 000000004a08: d6100017 02052113
	v_or_b32_e32 v26, 0x400000, v22                            // 000000004a10: 38342cff 00400000
	global_store_d16_hi_b16 v[100:101], v16, off offset:32     // 000000004a18: ee09407c 08000000 00002064
	s_wait_alu depctr_va_vcc(0)                                // 000000004a24: bf88ff9d
	v_cndmask_b32_e32 v17, v25, v29, vcc_lo                    // 000000004a28: 02223b19
	v_add3_u32 v16, v24, v18, 0x7fff                           // 000000004a2c: d6550010 03fe2518 00007fff
	v_or_b32_e32 v24, 0x400000, v18                            // 000000004a38: 383024ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v18, v18                           // 000000004a40: 7c302512
	v_bfe_u32 v18, v20, 16, 1                                  // 000000004a44: d6100012 02052114
	global_store_d16_hi_b16 v[106:107], v17, off offset:32     // 000000004a4c: ee09407c 08800000 0000206a
	v_add3_u32 v17, v23, v19, 0x7fff                           // 000000004a58: d6550011 03fe2717 00007fff
	v_or_b32_e32 v23, 0x400000, v19                            // 000000004a64: 382e26ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a6c: bf88ff9d
	v_cndmask_b32_e32 v16, v16, v24, vcc_lo                    // 000000004a70: 02203110
	v_cmp_u_f32_e32 vcc_lo, v19, v19                           // 000000004a74: 7c302713
	v_bfe_u32 v24, v21, 16, 1                                  // 000000004a78: d6100018 02052115
	v_add3_u32 v18, v18, v20, 0x7fff                           // 000000004a80: d6550012 03fe2912 00007fff
	v_bfe_u32 v19, v22, 16, 1                                  // 000000004a8c: d6100013 02052116
	v_or_b32_e32 v25, 0x400000, v21                            // 000000004a94: 38322aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004a9c: bf88ff9d
	v_cndmask_b32_e32 v17, v17, v23, vcc_lo                    // 000000004aa0: 02222f11
	v_or_b32_e32 v23, 0x400000, v20                            // 000000004aa4: 382e28ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v20, v20                           // 000000004aac: 7c302914
	v_add3_u32 v24, v24, v21, 0x7fff                           // 000000004ab0: d6550018 03fe2b18 00007fff
	v_add3_u32 v19, v19, v22, 0x7fff                           // 000000004abc: d6550013 03fe2d13 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004ac8: bf88ff9d
	v_cndmask_b32_e32 v18, v18, v23, vcc_lo                    // 000000004acc: 02242f12
	v_cmp_u_f32_e32 vcc_lo, v21, v21                           // 000000004ad0: 7c302b15
	s_wait_alu depctr_va_vcc(0)                                // 000000004ad4: bf88ff9d
	v_cndmask_b32_e32 v20, v24, v25, vcc_lo                    // 000000004ad8: 02283318
	v_cmp_u_f32_e32 vcc_lo, v22, v22                           // 000000004adc: 7c302d16
	s_wait_alu depctr_va_vcc(0)                                // 000000004ae0: bf88ff9d
	v_cndmask_b32_e32 v19, v19, v26, vcc_lo                    // 000000004ae4: 02263513
	s_clause 0x3                                               // 000000004ae8: bf850003
	global_store_d16_hi_b16 v[54:55], v16, off offset:32       // 000000004aec: ee09407c 08000000 00002036
	global_store_d16_hi_b16 v[104:105], v17, off offset:32     // 000000004af8: ee09407c 08800000 00002068
	global_store_d16_hi_b16 v[108:109], v18, off offset:32     // 000000004b04: ee09407c 09000000 0000206c
	global_store_d16_hi_b16 v[112:113], v20, off offset:32     // 000000004b10: ee09407c 0a000000 00002070
	global_store_d16_hi_b16 v[114:115], v19, off offset:32     // 000000004b1c: ee09407c 09800000 00002072
	global_load_u8 v16, v[110:111], off                        // 000000004b28: ee04007c 00000010 0000006e
	global_load_b32 v18, v[116:117], off                       // 000000004b34: ee05007c 00000012 00000074
	s_wait_loadcnt 0x1                                         // 000000004b40: bfc00001
	v_lshlrev_b32_e32 v17, 23, v16                             // 000000004b44: 30222097
	s_wait_loadcnt 0x0                                         // 000000004b48: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004b4c: bf870091
	v_mul_f32_e32 v16, v18, v17                                // 000000004b50: 10202312
	v_cmp_class_f32_e64 s0, v16, 0x198                         // 000000004b54: d47e0000 0201ff10 00000198
	v_mul_f32_e32 v16, v8, v16                                 // 000000004b60: 10202108
	s_xor_b32 s1, s0, -1                                       // 000000004b64: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b68: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004b6c: be802001
	s_cbranch_execnz 1276                                      // 000000004b70: bfa604fc <packed_folded_w4a8+0x4464>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004b74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004b78: 8c7e007e
	global_load_b32 v18, v[118:119], off                       // 000000004b7c: ee05007c 00000012 00000076
	s_wait_loadcnt 0x0                                         // 000000004b88: bfc00000
	v_mul_f32_e32 v8, v18, v17                                 // 000000004b8c: 10102312
	s_delay_alu instid0(valu_dep_1)                            // 000000004b90: bf870001
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000004b94: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v9, v8                                   // 000000004ba0: 10101109
	s_xor_b32 s1, s0, -1                                       // 000000004ba4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ba8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004bac: be802001
	s_cbranch_execnz 1277                                      // 000000004bb0: bfa604fd <packed_folded_w4a8+0x44a8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004bb8: 8c7e007e
	global_load_b32 v18, v[40:41], off                         // 000000004bbc: ee05007c 00000012 00000028
	s_wait_loadcnt 0x0                                         // 000000004bc8: bfc00000
	v_mul_f32_e32 v9, v18, v17                                 // 000000004bcc: 10122312
	s_delay_alu instid0(valu_dep_1)                            // 000000004bd0: bf870001
	v_cmp_class_f32_e64 s0, v9, 0x198                          // 000000004bd4: d47e0000 0201ff09 00000198
	v_mul_f32_e32 v9, v10, v9                                  // 000000004be0: 1012130a
	s_xor_b32 s1, s0, -1                                       // 000000004be4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004be8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004bec: be802001
	s_cbranch_execnz 1278                                      // 000000004bf0: bfa604fe <packed_folded_w4a8+0x44ec>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004bf4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004bf8: 8c7e007e
	global_load_b32 v18, v[120:121], off                       // 000000004bfc: ee05007c 00000012 00000078
	s_wait_loadcnt 0x0                                         // 000000004c08: bfc00000
	v_mul_f32_e32 v10, v18, v17                                // 000000004c0c: 10142312
	s_delay_alu instid0(valu_dep_1)                            // 000000004c10: bf870001
	v_cmp_class_f32_e64 s0, v10, 0x198                         // 000000004c14: d47e0000 0201ff0a 00000198
	v_mul_f32_e32 v10, v11, v10                                // 000000004c20: 1014150b
	s_xor_b32 s1, s0, -1                                       // 000000004c24: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c28: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004c2c: be802001
	s_cbranch_execnz 1279                                      // 000000004c30: bfa604ff <packed_folded_w4a8+0x4530>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004c38: 8c7e007e
	global_load_b32 v18, v[42:43], off                         // 000000004c3c: ee05007c 00000012 0000002a
	s_wait_loadcnt 0x0                                         // 000000004c48: bfc00000
	v_mul_f32_e32 v11, v18, v17                                // 000000004c4c: 10162312
	s_delay_alu instid0(valu_dep_1)                            // 000000004c50: bf870001
	v_cmp_class_f32_e64 s0, v11, 0x198                         // 000000004c54: d47e0000 0201ff0b 00000198
	v_mul_f32_e32 v11, v12, v11                                // 000000004c60: 1016170c
	s_xor_b32 s1, s0, -1                                       // 000000004c64: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c68: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004c6c: be802001
	s_cbranch_execnz 1280                                      // 000000004c70: bfa60500 <packed_folded_w4a8+0x4574>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004c74: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004c78: 8c7e007e
	global_load_b32 v18, v[122:123], off                       // 000000004c7c: ee05007c 00000012 0000007a
	s_wait_loadcnt 0x0                                         // 000000004c88: bfc00000
	v_mul_f32_e32 v12, v18, v17                                // 000000004c8c: 10182312
	s_delay_alu instid0(valu_dep_1)                            // 000000004c90: bf870001
	v_cmp_class_f32_e64 s0, v12, 0x198                         // 000000004c94: d47e0000 0201ff0c 00000198
	v_mul_f32_e32 v12, v13, v12                                // 000000004ca0: 1018190d
	s_xor_b32 s1, s0, -1                                       // 000000004ca4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ca8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004cac: be802001
	s_cbranch_execnz 1281                                      // 000000004cb0: bfa60501 <packed_folded_w4a8+0x45b8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cb4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004cb8: 8c7e007e
	global_load_b32 v18, v[44:45], off                         // 000000004cbc: ee05007c 00000012 0000002c
	s_wait_loadcnt 0x0                                         // 000000004cc8: bfc00000
	v_mul_f32_e32 v13, v18, v17                                // 000000004ccc: 101a2312
	s_delay_alu instid0(valu_dep_1)                            // 000000004cd0: bf870001
	v_cmp_class_f32_e64 s0, v13, 0x198                         // 000000004cd4: d47e0000 0201ff0d 00000198
	v_mul_f32_e32 v13, v14, v13                                // 000000004ce0: 101a1b0e
	s_xor_b32 s1, s0, -1                                       // 000000004ce4: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ce8: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004cec: be802001
	s_cbranch_execnz 1282                                      // 000000004cf0: bfa60502 <packed_folded_w4a8+0x45fc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004cf4: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004cf8: 8c7e007e
	global_load_b32 v18, v[126:127], off                       // 000000004cfc: ee05007c 00000012 0000007e
	s_wait_loadcnt 0x0                                         // 000000004d08: bfc00000
	v_mul_f32_e32 v14, v18, v17                                // 000000004d0c: 101c2312
	s_delay_alu instid0(valu_dep_1)                            // 000000004d10: bf870001
	v_cmp_class_f32_e64 s0, v14, 0x198                         // 000000004d14: d47e0000 0201ff0e 00000198
	v_mul_f32_e32 v14, v15, v14                                // 000000004d20: 101c1d0f
	s_xor_b32 s1, s0, -1                                       // 000000004d24: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d28: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004d2c: be802001
	s_cbranch_execnz 1283                                      // 000000004d30: bfa60503 <packed_folded_w4a8+0x4640>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004d34: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004d38: 8c7e007e
	v_bfe_u32 v15, v16, 16, 1                                  // 000000004d3c: d610000f 02052110
	v_or_b32_e32 v17, 0x400000, v16                            // 000000004d44: 382220ff 00400000
	v_bfe_u32 v18, v8, 16, 1                                   // 000000004d4c: d6100012 02052108
	v_cmp_u_f32_e32 vcc_lo, v16, v16                           // 000000004d54: 7c302110
	v_or_b32_e32 v19, 0x400000, v8                             // 000000004d58: 382610ff 00400000
	v_add3_u32 v15, v15, v16, 0x7fff                           // 000000004d60: d655000f 03fe210f 00007fff
	v_bfe_u32 v20, v9, 16, 1                                   // 000000004d6c: d6100014 02052109
	v_add3_u32 v18, v18, v8, 0x7fff                            // 000000004d74: d6550012 03fe1112 00007fff
	v_or_b32_e32 v21, 0x400000, v9                             // 000000004d80: 382a12ff 00400000
	v_bfe_u32 v16, v10, 16, 1                                  // 000000004d88: d6100010 0205210a
	s_wait_alu depctr_va_vcc(0)                                // 000000004d90: bf88ff9d
	v_cndmask_b32_e32 v15, v15, v17, vcc_lo                    // 000000004d94: 021e230f
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 000000004d98: 7c301108
	v_add3_u32 v17, v20, v9, 0x7fff                            // 000000004d9c: d6550011 03fe1314 00007fff
	global_store_d16_hi_b16 v[130:131], v15, off offset:32     // 000000004da8: ee09407c 07800000 00002082
	s_wait_alu depctr_va_vcc(0)                                // 000000004db4: bf88ff9d
	v_cndmask_b32_e32 v8, v18, v19, vcc_lo                     // 000000004db8: 02102712
	v_cmp_u_f32_e32 vcc_lo, v9, v9                             // 000000004dbc: 7c301309
	v_bfe_u32 v15, v11, 16, 1                                  // 000000004dc0: d610000f 0205210b
	v_or_b32_e32 v18, 0x400000, v14                            // 000000004dc8: 38241cff 00400000
	global_store_d16_hi_b16 v[128:129], v8, off offset:32      // 000000004dd0: ee09407c 04000000 00002080
	s_wait_alu depctr_va_vcc(0)                                // 000000004ddc: bf88ff9d
	v_cndmask_b32_e32 v9, v17, v21, vcc_lo                     // 000000004de0: 02122b11
	v_add3_u32 v8, v16, v10, 0x7fff                            // 000000004de4: d6550008 03fe1510 00007fff
	v_or_b32_e32 v16, 0x400000, v10                            // 000000004df0: 382014ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v10, v10                           // 000000004df8: 7c30150a
	v_bfe_u32 v10, v12, 16, 1                                  // 000000004dfc: d610000a 0205210c
	global_store_d16_hi_b16 v[134:135], v9, off offset:32      // 000000004e04: ee09407c 04800000 00002086
	v_add3_u32 v9, v15, v11, 0x7fff                            // 000000004e10: d6550009 03fe170f 00007fff
	v_or_b32_e32 v15, 0x400000, v11                            // 000000004e1c: 381e16ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e24: bf88ff9d
	v_cndmask_b32_e32 v8, v8, v16, vcc_lo                      // 000000004e28: 02102108
	v_cmp_u_f32_e32 vcc_lo, v11, v11                           // 000000004e2c: 7c30170b
	v_bfe_u32 v16, v13, 16, 1                                  // 000000004e30: d6100010 0205210d
	v_add3_u32 v10, v10, v12, 0x7fff                           // 000000004e38: d655000a 03fe190a 00007fff
	v_bfe_u32 v11, v14, 16, 1                                  // 000000004e44: d610000b 0205210e
	v_or_b32_e32 v17, 0x400000, v13                            // 000000004e4c: 38221aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 000000004e54: bf88ff9d
	v_cndmask_b32_e32 v9, v9, v15, vcc_lo                      // 000000004e58: 02121f09
	v_or_b32_e32 v15, 0x400000, v12                            // 000000004e5c: 381e18ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v12, v12                           // 000000004e64: 7c30190c
	v_add3_u32 v16, v16, v13, 0x7fff                           // 000000004e68: d6550010 03fe1b10 00007fff
	v_add3_u32 v11, v11, v14, 0x7fff                           // 000000004e74: d655000b 03fe1d0b 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000004e80: bf88ff9d
	v_cndmask_b32_e32 v10, v10, v15, vcc_lo                    // 000000004e84: 02141f0a
	v_cmp_u_f32_e32 vcc_lo, v13, v13                           // 000000004e88: 7c301b0d
	s_wait_alu depctr_va_vcc(0)                                // 000000004e8c: bf88ff9d
	v_cndmask_b32_e32 v12, v16, v17, vcc_lo                    // 000000004e90: 02182310
	v_cmp_u_f32_e32 vcc_lo, v14, v14                           // 000000004e94: 7c301d0e
	s_wait_alu depctr_va_vcc(0)                                // 000000004e98: bf88ff9d
	v_cndmask_b32_e32 v11, v11, v18, vcc_lo                    // 000000004e9c: 0216250b
	s_clause 0x3                                               // 000000004ea0: bf850003
	global_store_d16_hi_b16 v[46:47], v8, off offset:32        // 000000004ea4: ee09407c 04000000 0000202e
	global_store_d16_hi_b16 v[132:133], v9, off offset:32      // 000000004eb0: ee09407c 04800000 00002084
	global_store_d16_hi_b16 v[136:137], v10, off offset:32     // 000000004ebc: ee09407c 05000000 00002088
	global_store_d16_hi_b16 v[138:139], v12, off offset:32     // 000000004ec8: ee09407c 06000000 0000208a
	global_store_d16_hi_b16 v[140:141], v11, off offset:32     // 000000004ed4: ee09407c 05800000 0000208c
	global_load_u8 v8, v[110:111], off                         // 000000004ee0: ee04007c 00000008 0000006e
	global_load_b32 v10, v[142:143], off                       // 000000004eec: ee05007c 0000000a 0000008e
	s_wait_loadcnt 0x1                                         // 000000004ef8: bfc00001
	v_lshlrev_b32_e32 v9, 23, v8                               // 000000004efc: 30121097
	s_wait_loadcnt 0x0                                         // 000000004f00: bfc00000
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000004f04: bf870091
	v_mul_f32_e32 v8, v10, v9                                  // 000000004f08: 1010130a
	v_cmp_class_f32_e64 s0, v8, 0x198                          // 000000004f0c: d47e0000 0201ff08 00000198
	v_mul_f32_e32 v8, v0, v8                                   // 000000004f18: 10101100
	s_xor_b32 s1, s0, -1                                       // 000000004f1c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f20: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004f24: be802001
	s_cbranch_execnz 1174                                      // 000000004f28: bfa60496 <packed_folded_w4a8+0x4684>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f2c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004f30: 8c7e007e
	global_load_b32 v10, v[124:125], off                       // 000000004f34: ee05007c 0000000a 0000007c
	s_wait_loadcnt 0x0                                         // 000000004f40: bfc00000
	v_mul_f32_e32 v0, v10, v9                                  // 000000004f44: 1000130a
	s_delay_alu instid0(valu_dep_1)                            // 000000004f48: bf870001
	v_cmp_class_f32_e64 s0, v0, 0x198                          // 000000004f4c: d47e0000 0201ff00 00000198
	v_mul_f32_e32 v0, v1, v0                                   // 000000004f58: 10000101
	s_xor_b32 s1, s0, -1                                       // 000000004f5c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f60: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004f64: be802001
	s_cbranch_execnz 1175                                      // 000000004f68: bfa60497 <packed_folded_w4a8+0x46c8>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004f6c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004f70: 8c7e007e
	global_load_b32 v10, v[32:33], off                         // 000000004f74: ee05007c 0000000a 00000020
	s_wait_loadcnt 0x0                                         // 000000004f80: bfc00000
	v_mul_f32_e32 v1, v10, v9                                  // 000000004f84: 1002130a
	s_delay_alu instid0(valu_dep_1)                            // 000000004f88: bf870001
	v_cmp_class_f32_e64 s0, v1, 0x198                          // 000000004f8c: d47e0000 0201ff01 00000198
	v_mul_f32_e32 v1, v2, v1                                   // 000000004f98: 10020302
	s_xor_b32 s1, s0, -1                                       // 000000004f9c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fa0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004fa4: be802001
	s_cbranch_execnz 1176                                      // 000000004fa8: bfa60498 <packed_folded_w4a8+0x470c>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004fb0: 8c7e007e
	global_load_b32 v10, v[144:145], off                       // 000000004fb4: ee05007c 0000000a 00000090
	s_wait_loadcnt 0x0                                         // 000000004fc0: bfc00000
	v_mul_f32_e32 v2, v10, v9                                  // 000000004fc4: 1004130a
	s_delay_alu instid0(valu_dep_1)                            // 000000004fc8: bf870001
	v_cmp_class_f32_e64 s0, v2, 0x198                          // 000000004fcc: d47e0000 0201ff02 00000198
	v_mul_f32_e32 v2, v3, v2                                   // 000000004fd8: 10040503
	s_xor_b32 s1, s0, -1                                       // 000000004fdc: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fe0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000004fe4: be802001
	s_cbranch_execnz 1177                                      // 000000004fe8: bfa60499 <packed_folded_w4a8+0x4750>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004fec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000004ff0: 8c7e007e
	global_load_b32 v10, v[34:35], off                         // 000000004ff4: ee05007c 0000000a 00000022
	s_wait_loadcnt 0x0                                         // 000000005000: bfc00000
	v_mul_f32_e32 v3, v10, v9                                  // 000000005004: 1006130a
	s_delay_alu instid0(valu_dep_1)                            // 000000005008: bf870001
	v_cmp_class_f32_e64 s0, v3, 0x198                          // 00000000500c: d47e0000 0201ff03 00000198
	v_mul_f32_e32 v3, v4, v3                                   // 000000005018: 10060704
	s_xor_b32 s1, s0, -1                                       // 00000000501c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005020: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005024: be802001
	s_cbranch_execnz 1178                                      // 000000005028: bfa6049a <packed_folded_w4a8+0x4794>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000502c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005030: 8c7e007e
	global_load_b32 v10, v[146:147], off                       // 000000005034: ee05007c 0000000a 00000092
	s_wait_loadcnt 0x0                                         // 000000005040: bfc00000
	v_mul_f32_e32 v4, v10, v9                                  // 000000005044: 1008130a
	s_delay_alu instid0(valu_dep_1)                            // 000000005048: bf870001
	v_cmp_class_f32_e64 s0, v4, 0x198                          // 00000000504c: d47e0000 0201ff04 00000198
	v_mul_f32_e32 v4, v5, v4                                   // 000000005058: 10080905
	s_xor_b32 s1, s0, -1                                       // 00000000505c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005060: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 000000005064: be802001
	s_cbranch_execnz 1179                                      // 000000005068: bfa6049b <packed_folded_w4a8+0x47d8>
	s_wait_alu depctr_sa_sdst(0)                               // 00000000506c: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005070: 8c7e007e
	global_load_b32 v10, v[36:37], off                         // 000000005074: ee05007c 0000000a 00000024
	s_wait_loadcnt 0x0                                         // 000000005080: bfc00000
	v_mul_f32_e32 v5, v10, v9                                  // 000000005084: 100a130a
	s_delay_alu instid0(valu_dep_1)                            // 000000005088: bf870001
	v_cmp_class_f32_e64 s0, v5, 0x198                          // 00000000508c: d47e0000 0201ff05 00000198
	v_mul_f32_e32 v5, v6, v5                                   // 000000005098: 100a0b06
	s_xor_b32 s1, s0, -1                                       // 00000000509c: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050a0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000050a4: be802001
	s_cbranch_execnz 1180                                      // 0000000050a8: bfa6049c <packed_folded_w4a8+0x481c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050ac: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050b0: 8c7e007e
	global_load_b32 v10, v[148:149], off                       // 0000000050b4: ee05007c 0000000a 00000094
	s_wait_loadcnt 0x0                                         // 0000000050c0: bfc00000
	v_mul_f32_e32 v6, v10, v9                                  // 0000000050c4: 100c130a
	s_delay_alu instid0(valu_dep_1)                            // 0000000050c8: bf870001
	v_cmp_class_f32_e64 s0, v6, 0x198                          // 0000000050cc: d47e0000 0201ff06 00000198
	v_mul_f32_e32 v6, v7, v6                                   // 0000000050d8: 100c0d07
	s_xor_b32 s1, s0, -1                                       // 0000000050dc: 8d01c100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050e0: bf88ff9e
	s_and_saveexec_b32 s0, s1                                  // 0000000050e4: be802001
	s_cbranch_execnz 1181                                      // 0000000050e8: bfa6049d <packed_folded_w4a8+0x4860>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050ec: bf88ff9e
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000050f0: 8c7e007e
	v_bfe_u32 v7, v8, 16, 1                                    // 0000000050f4: d6100007 02052108
	v_or_b32_e32 v9, 0x400000, v8                              // 0000000050fc: 381210ff 00400000
	v_bfe_u32 v10, v0, 16, 1                                   // 000000005104: d610000a 02052100
	v_cmp_u_f32_e32 vcc_lo, v8, v8                             // 00000000510c: 7c301108
	v_or_b32_e32 v11, 0x400000, v0                             // 000000005110: 381600ff 00400000
	v_add3_u32 v7, v7, v8, 0x7fff                              // 000000005118: d6550007 03fe1107 00007fff
	v_bfe_u32 v12, v1, 16, 1                                   // 000000005124: d610000c 02052101
	v_add3_u32 v10, v10, v0, 0x7fff                            // 00000000512c: d655000a 03fe010a 00007fff
	v_or_b32_e32 v13, 0x400000, v1                             // 000000005138: 381a02ff 00400000
	v_bfe_u32 v8, v2, 16, 1                                    // 000000005140: d6100008 02052102
	s_wait_alu depctr_va_vcc(0)                                // 000000005148: bf88ff9d
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 00000000514c: 020e1307
	v_cmp_u_f32_e32 vcc_lo, v0, v0                             // 000000005150: 7c300100
	v_add3_u32 v9, v12, v1, 0x7fff                             // 000000005154: d6550009 03fe030c 00007fff
	global_store_d16_hi_b16 v[152:153], v7, off offset:32      // 000000005160: ee09407c 03800000 00002098
	s_wait_alu depctr_va_vcc(0)                                // 00000000516c: bf88ff9d
	v_cndmask_b32_e32 v0, v10, v11, vcc_lo                     // 000000005170: 0200170a
	v_cmp_u_f32_e32 vcc_lo, v1, v1                             // 000000005174: 7c300301
	v_bfe_u32 v7, v3, 16, 1                                    // 000000005178: d6100007 02052103
	v_or_b32_e32 v10, 0x400000, v6                             // 000000005180: 38140cff 00400000
	global_store_d16_hi_b16 v[150:151], v0, off offset:32      // 000000005188: ee09407c 00000000 00002096
	s_wait_alu depctr_va_vcc(0)                                // 000000005194: bf88ff9d
	v_cndmask_b32_e32 v1, v9, v13, vcc_lo                      // 000000005198: 02021b09
	v_add3_u32 v0, v8, v2, 0x7fff                              // 00000000519c: d6550000 03fe0508 00007fff
	v_or_b32_e32 v8, 0x400000, v2                              // 0000000051a8: 381004ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v2, v2                             // 0000000051b0: 7c300502
	v_bfe_u32 v2, v4, 16, 1                                    // 0000000051b4: d6100002 02052104
	global_store_d16_hi_b16 v[156:157], v1, off offset:32      // 0000000051bc: ee09407c 00800000 0000209c
	v_add3_u32 v1, v7, v3, 0x7fff                              // 0000000051c8: d6550001 03fe0707 00007fff
	v_or_b32_e32 v7, 0x400000, v3                              // 0000000051d4: 380e06ff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 0000000051dc: bf88ff9d
	v_cndmask_b32_e32 v0, v0, v8, vcc_lo                       // 0000000051e0: 02001100
	v_cmp_u_f32_e32 vcc_lo, v3, v3                             // 0000000051e4: 7c300703
	v_bfe_u32 v8, v5, 16, 1                                    // 0000000051e8: d6100008 02052105
	v_add3_u32 v2, v2, v4, 0x7fff                              // 0000000051f0: d6550002 03fe0902 00007fff
	v_bfe_u32 v3, v6, 16, 1                                    // 0000000051fc: d6100003 02052106
	v_or_b32_e32 v9, 0x400000, v5                              // 000000005204: 38120aff 00400000
	s_wait_alu depctr_va_vcc(0)                                // 00000000520c: bf88ff9d
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo                       // 000000005210: 02020f01
	v_or_b32_e32 v7, 0x400000, v4                              // 000000005214: 380e08ff 00400000
	v_cmp_u_f32_e32 vcc_lo, v4, v4                             // 00000000521c: 7c300904
	v_add3_u32 v8, v8, v5, 0x7fff                              // 000000005220: d6550008 03fe0b08 00007fff
	v_add3_u32 v3, v3, v6, 0x7fff                              // 00000000522c: d6550003 03fe0d03 00007fff
	s_wait_alu depctr_va_vcc(0)                                // 000000005238: bf88ff9d
	v_cndmask_b32_e32 v2, v2, v7, vcc_lo                       // 00000000523c: 02040f02
	v_cmp_u_f32_e32 vcc_lo, v5, v5                             // 000000005240: 7c300b05
	s_wait_alu depctr_va_vcc(0)                                // 000000005244: bf88ff9d
	v_cndmask_b32_e32 v4, v8, v9, vcc_lo                       // 000000005248: 02081308
	v_cmp_u_f32_e32 vcc_lo, v6, v6                             // 00000000524c: 7c300d06
	s_wait_alu depctr_va_vcc(0)                                // 000000005250: bf88ff9d
	v_cndmask_b32_e32 v3, v3, v10, vcc_lo                      // 000000005254: 02061503
	s_clause 0x3                                               // 000000005258: bf850003
	global_store_d16_hi_b16 v[38:39], v0, off offset:32        // 00000000525c: ee09407c 00000000 00002026
	global_store_d16_hi_b16 v[154:155], v1, off offset:32      // 000000005268: ee09407c 00800000 0000209a
	global_store_d16_hi_b16 v[158:159], v2, off offset:32      // 000000005274: ee09407c 01000000 0000209e
	global_store_d16_hi_b16 v[160:161], v4, off offset:32      // 000000005280: ee09407c 02000000 000020a0
	global_store_d16_hi_b16 v[64:65], v3, off offset:32        // 00000000528c: ee09407c 01800000 00002040
	s_nop 0                                                    // 000000005298: bf800000
	s_sendmsg sendmsg(msg_dealloc_vgprs)                       // 00000000529c: bfb60003
	s_endpgm                                                   // 0000000052a0: bfb00000
	v_cvt_f64_f32_e32 v[69:70], v56                            // 0000000052a4: 7e8a2138
	v_cvt_f64_f32_e32 v[71:72], v83                            // 0000000052a8: 7e8e2153
	v_cvt_f64_f32_e32 v[81:82], v68                            // 0000000052ac: 7ea22144
	v_cmp_eq_f32_e32 vcc_lo, 0, v56                            // 0000000052b0: 7c247080
	v_cmp_class_f32_e64 s3, v68, 0x1f8                         // 0000000052b4: d47e0003 0201ff44 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 0000000052c0: 8b03036a
	v_mul_f64_e32 v[69:70], v[69:70], v[71:72]                 // 0000000052c4: 0c8a8f45
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000052c8: bf870091
	v_mul_f64_e32 v[69:70], v[69:70], v[81:82]                 // 0000000052cc: 0c8aa345
	v_cvt_f32_f64_e32 v69, v[69:70]                            // 0000000052d0: 7e8a1f45
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052d4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000052d8: bf870001
	v_cndmask_b32_e64 v81, v69, 0, s3                          // 0000000052dc: d5010051 000d0145
	s_branch 62790                                             // 0000000052e4: bfa0f546 <packed_folded_w4a8+0xd00>
	v_cvt_f64_f32_e32 v[70:71], v57                            // 0000000052e8: 7e8c2139
	v_cvt_f64_f32_e32 v[72:73], v83                            // 0000000052ec: 7e902153
	v_cvt_f64_f32_e32 v[84:85], v56                            // 0000000052f0: 7ea82138
	v_cmp_eq_f32_e32 vcc_lo, 0, v57                            // 0000000052f4: 7c247280
	v_cmp_class_f32_e64 s3, v56, 0x1f8                         // 0000000052f8: d47e0003 0201ff38 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 000000005304: 8b03036a
	v_mul_f64_e32 v[70:71], v[70:71], v[72:73]                 // 000000005308: 0c8c9146
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000530c: bf870091
	v_mul_f64_e32 v[70:71], v[70:71], v[84:85]                 // 000000005310: 0c8ca946
	v_cvt_f32_f64_e32 v70, v[70:71]                            // 000000005314: 7e8c1f46
	s_wait_alu depctr_sa_sdst(0)                               // 000000005318: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000531c: bf870001
	v_cndmask_b32_e64 v82, v70, 0, s3                          // 000000005320: d5010052 000d0146
	s_branch 62804                                             // 000000005328: bfa0f554 <packed_folded_w4a8+0xd7c>
	v_cvt_f64_f32_e32 v[71:72], v58                            // 00000000532c: 7e8e213a
	v_cvt_f64_f32_e32 v[84:85], v83                            // 000000005330: 7ea82153
	v_cvt_f64_f32_e32 v[86:87], v70                            // 000000005334: 7eac2146
	v_cmp_eq_f32_e32 vcc_lo, 0, v58                            // 000000005338: 7c247480
	v_cmp_class_f32_e64 s3, v70, 0x1f8                         // 00000000533c: d47e0003 0201ff46 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 000000005348: 8b03036a
	v_mul_f64_e32 v[71:72], v[71:72], v[84:85]                 // 00000000534c: 0c8ea947
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005350: bf870091
	v_mul_f64_e32 v[71:72], v[71:72], v[86:87]                 // 000000005354: 0c8ead47
	v_cvt_f32_f64_e32 v71, v[71:72]                            // 000000005358: 7e8e1f47
	s_wait_alu depctr_sa_sdst(0)                               // 00000000535c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005360: bf870001
	v_cndmask_b32_e64 v84, v71, 0, s3                          // 000000005364: d5010054 000d0147
	s_branch 62818                                             // 00000000536c: bfa0f562 <packed_folded_w4a8+0xdf8>
	v_cvt_f64_f32_e32 v[72:73], v59                            // 000000005370: 7e90213b
	v_cvt_f64_f32_e32 v[85:86], v83                            // 000000005374: 7eaa2153
	v_cvt_f64_f32_e32 v[87:88], v58                            // 000000005378: 7eae213a
	v_cmp_eq_f32_e32 vcc_lo, 0, v59                            // 00000000537c: 7c247680
	v_cmp_class_f32_e64 s3, v58, 0x1f8                         // 000000005380: d47e0003 0201ff3a 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 00000000538c: 8b03036a
	v_mul_f64_e32 v[72:73], v[72:73], v[85:86]                 // 000000005390: 0c90ab48
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005394: bf870091
	v_mul_f64_e32 v[72:73], v[72:73], v[87:88]                 // 000000005398: 0c90af48
	v_cvt_f32_f64_e32 v72, v[72:73]                            // 00000000539c: 7e901f48
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053a0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000053a4: bf870001
	v_cndmask_b32_e64 v85, v72, 0, s3                          // 0000000053a8: d5010055 000d0148
	s_branch 62832                                             // 0000000053b0: bfa0f570 <packed_folded_w4a8+0xe74>
	v_cvt_f64_f32_e32 v[86:87], v60                            // 0000000053b4: 7eac213c
	v_cvt_f64_f32_e32 v[88:89], v83                            // 0000000053b8: 7eb02153
	v_cvt_f64_f32_e32 v[90:91], v72                            // 0000000053bc: 7eb42148
	v_cmp_eq_f32_e32 vcc_lo, 0, v60                            // 0000000053c0: 7c247880
	v_cmp_class_f32_e64 s3, v72, 0x1f8                         // 0000000053c4: d47e0003 0201ff48 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 0000000053d0: 8b03036a
	v_mul_f64_e32 v[86:87], v[86:87], v[88:89]                 // 0000000053d4: 0cacb156
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000053d8: bf870091
	v_mul_f64_e32 v[86:87], v[86:87], v[90:91]                 // 0000000053dc: 0cacb556
	v_cvt_f32_f64_e32 v73, v[86:87]                            // 0000000053e0: 7e921f56
	s_wait_alu depctr_sa_sdst(0)                               // 0000000053e4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000053e8: bf870001
	v_cndmask_b32_e64 v86, v73, 0, s3                          // 0000000053ec: d5010056 000d0149
	s_branch 62846                                             // 0000000053f4: bfa0f57e <packed_folded_w4a8+0xef0>
	v_cvt_f64_f32_e32 v[87:88], v61                            // 0000000053f8: 7eae213d
	v_cvt_f64_f32_e32 v[89:90], v83                            // 0000000053fc: 7eb22153
	v_cvt_f64_f32_e32 v[91:92], v60                            // 000000005400: 7eb6213c
	v_cmp_eq_f32_e32 vcc_lo, 0, v61                            // 000000005404: 7c247a80
	v_cmp_class_f32_e64 s3, v60, 0x1f8                         // 000000005408: d47e0003 0201ff3c 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 000000005414: 8b03036a
	v_mul_f64_e32 v[87:88], v[87:88], v[89:90]                 // 000000005418: 0caeb357
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000541c: bf870091
	v_mul_f64_e32 v[87:88], v[87:88], v[91:92]                 // 000000005420: 0caeb757
	v_cvt_f32_f64_e32 v87, v[87:88]                            // 000000005424: 7eae1f57
	s_wait_alu depctr_sa_sdst(0)                               // 000000005428: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000542c: bf870001
	v_cndmask_b32_e64 v87, v87, 0, s3                          // 000000005430: d5010057 000d0157
	s_branch 62860                                             // 000000005438: bfa0f58c <packed_folded_w4a8+0xf6c>
	v_cvt_f64_f32_e32 v[90:91], v62                            // 00000000543c: 7eb4213e
	v_cvt_f64_f32_e32 v[92:93], v83                            // 000000005440: 7eb82153
	v_cvt_f64_f32_e32 v[94:95], v89                            // 000000005444: 7ebc2159
	v_cmp_eq_f32_e32 vcc_lo, 0, v62                            // 000000005448: 7c247c80
	v_cmp_class_f32_e64 s3, v89, 0x1f8                         // 00000000544c: d47e0003 0201ff59 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 000000005458: 8b03036a
	v_mul_f64_e32 v[90:91], v[90:91], v[92:93]                 // 00000000545c: 0cb4b95a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005460: bf870091
	v_mul_f64_e32 v[90:91], v[90:91], v[94:95]                 // 000000005464: 0cb4bd5a
	v_cvt_f32_f64_e32 v88, v[90:91]                            // 000000005468: 7eb01f5a
	s_wait_alu depctr_sa_sdst(0)                               // 00000000546c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005470: bf870001
	v_cndmask_b32_e64 v88, v88, 0, s3                          // 000000005474: d5010058 000d0158
	s_branch 62874                                             // 00000000547c: bfa0f59a <packed_folded_w4a8+0xfe8>
	v_cvt_f64_f32_e32 v[89:90], v63                            // 000000005480: 7eb2213f
	v_cvt_f64_f32_e32 v[91:92], v83                            // 000000005484: 7eb62153
	v_cvt_f64_f32_e32 v[93:94], v62                            // 000000005488: 7eba213e
	v_cmp_eq_f32_e32 vcc_lo, 0, v63                            // 00000000548c: 7c247e80
	v_cmp_class_f32_e64 s3, v62, 0x1f8                         // 000000005490: d47e0003 0201ff3e 000001f8
	s_and_b32 s3, vcc_lo, s3                                   // 00000000549c: 8b03036a
	v_mul_f64_e32 v[89:90], v[89:90], v[91:92]                 // 0000000054a0: 0cb2b759
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000054a4: bf870091
	v_mul_f64_e32 v[89:90], v[89:90], v[93:94]                 // 0000000054a8: 0cb2bb59
	v_cvt_f32_f64_e32 v83, v[89:90]                            // 0000000054ac: 7ea61f59
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054b0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000054b4: bf870001
	v_cndmask_b32_e64 v89, v83, 0, s3                          // 0000000054b8: d5010059 000d0153
	s_branch 62887                                             // 0000000054c0: bfa0f5a7 <packed_folded_w4a8+0x1060>
	v_cvt_f64_f32_e32 v[93:94], v48                            // 0000000054c4: 7eba2130
	v_cvt_f64_f32_e32 v[95:96], v105                           // 0000000054c8: 7ebe2169
	v_cvt_f64_f32_e32 v[103:104], v92                          // 0000000054cc: 7ece215c
	v_cmp_eq_f32_e32 vcc_lo, 0, v48                            // 0000000054d0: 7c246080
	v_cmp_class_f32_e64 s1, v92, 0x1f8                         // 0000000054d4: d47e0001 0201ff5c 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000054e0: 8b01016a
	v_mul_f64_e32 v[93:94], v[93:94], v[95:96]                 // 0000000054e4: 0cbabf5d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000054e8: bf870091
	v_mul_f64_e32 v[93:94], v[93:94], v[103:104]               // 0000000054ec: 0cbacf5d
	v_cvt_f32_f64_e32 v93, v[93:94]                            // 0000000054f0: 7eba1f5d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054f4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000054f8: bf870001
	v_cndmask_b32_e64 v103, v93, 0, s1                         // 0000000054fc: d5010067 0005015d
	s_branch 63113                                             // 000000005504: bfa0f689 <packed_folded_w4a8+0x142c>
	v_cvt_f64_f32_e32 v[94:95], v49                            // 000000005508: 7ebc2131
	v_cvt_f64_f32_e32 v[96:97], v105                           // 00000000550c: 7ec02169
	v_cvt_f64_f32_e32 v[106:107], v48                          // 000000005510: 7ed42130
	v_cmp_eq_f32_e32 vcc_lo, 0, v49                            // 000000005514: 7c246280
	v_cmp_class_f32_e64 s1, v48, 0x1f8                         // 000000005518: d47e0001 0201ff30 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005524: 8b01016a
	v_mul_f64_e32 v[94:95], v[94:95], v[96:97]                 // 000000005528: 0cbcc15e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000552c: bf870091
	v_mul_f64_e32 v[94:95], v[94:95], v[106:107]               // 000000005530: 0cbcd55e
	v_cvt_f32_f64_e32 v94, v[94:95]                            // 000000005534: 7ebc1f5e
	s_wait_alu depctr_sa_sdst(0)                               // 000000005538: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000553c: bf870001
	v_cndmask_b32_e64 v104, v94, 0, s1                         // 000000005540: d5010068 0005015e
	s_branch 63126                                             // 000000005548: bfa0f696 <packed_folded_w4a8+0x14a4>
	v_cvt_f64_f32_e32 v[95:96], v50                            // 00000000554c: 7ebe2132
	v_cvt_f64_f32_e32 v[106:107], v105                         // 000000005550: 7ed42169
	v_cvt_f64_f32_e32 v[108:109], v94                          // 000000005554: 7ed8215e
	v_cmp_eq_f32_e32 vcc_lo, 0, v50                            // 000000005558: 7c246480
	v_cmp_class_f32_e64 s1, v94, 0x1f8                         // 00000000555c: d47e0001 0201ff5e 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005568: 8b01016a
	v_mul_f64_e32 v[95:96], v[95:96], v[106:107]               // 00000000556c: 0cbed55f
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005570: bf870091
	v_mul_f64_e32 v[95:96], v[95:96], v[108:109]               // 000000005574: 0cbed95f
	v_cvt_f32_f64_e32 v95, v[95:96]                            // 000000005578: 7ebe1f5f
	s_wait_alu depctr_sa_sdst(0)                               // 00000000557c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005580: bf870001
	v_cndmask_b32_e64 v106, v95, 0, s1                         // 000000005584: d501006a 0005015f
	s_branch 63139                                             // 00000000558c: bfa0f6a3 <packed_folded_w4a8+0x151c>
	v_cvt_f64_f32_e32 v[96:97], v51                            // 000000005590: 7ec02133
	v_cvt_f64_f32_e32 v[107:108], v105                         // 000000005594: 7ed62169
	v_cvt_f64_f32_e32 v[112:113], v50                          // 000000005598: 7ee02132
	v_cmp_eq_f32_e32 vcc_lo, 0, v51                            // 00000000559c: 7c246680
	v_cmp_class_f32_e64 s1, v50, 0x1f8                         // 0000000055a0: d47e0001 0201ff32 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000055ac: 8b01016a
	v_mul_f64_e32 v[96:97], v[96:97], v[107:108]               // 0000000055b0: 0cc0d760
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000055b4: bf870091
	v_mul_f64_e32 v[96:97], v[96:97], v[112:113]               // 0000000055b8: 0cc0e160
	v_cvt_f32_f64_e32 v96, v[96:97]                            // 0000000055bc: 7ec01f60
	s_wait_alu depctr_sa_sdst(0)                               // 0000000055c0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000055c4: bf870001
	v_cndmask_b32_e64 v108, v96, 0, s1                         // 0000000055c8: d501006c 00050160
	s_branch 63152                                             // 0000000055d0: bfa0f6b0 <packed_folded_w4a8+0x1594>
	v_cvt_f64_f32_e32 v[112:113], v52                          // 0000000055d4: 7ee02134
	v_cvt_f64_f32_e32 v[114:115], v105                         // 0000000055d8: 7ee42169
	v_cvt_f64_f32_e32 v[116:117], v96                          // 0000000055dc: 7ee82160
	v_cmp_eq_f32_e32 vcc_lo, 0, v52                            // 0000000055e0: 7c246880
	v_cmp_class_f32_e64 s1, v96, 0x1f8                         // 0000000055e4: d47e0001 0201ff60 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000055f0: 8b01016a
	v_mul_f64_e32 v[112:113], v[112:113], v[114:115]           // 0000000055f4: 0ce0e570
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000055f8: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[116:117]           // 0000000055fc: 0ce0e970
	v_cvt_f32_f64_e32 v97, v[112:113]                          // 000000005600: 7ec21f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000005604: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005608: bf870001
	v_cndmask_b32_e64 v109, v97, 0, s1                         // 00000000560c: d501006d 00050161
	s_branch 63165                                             // 000000005614: bfa0f6bd <packed_folded_w4a8+0x160c>
	v_cvt_f64_f32_e32 v[112:113], v53                          // 000000005618: 7ee02135
	v_cvt_f64_f32_e32 v[114:115], v105                         // 00000000561c: 7ee42169
	v_cvt_f64_f32_e32 v[116:117], v52                          // 000000005620: 7ee82134
	v_cmp_eq_f32_e32 vcc_lo, 0, v53                            // 000000005624: 7c246a80
	v_cmp_class_f32_e64 s1, v52, 0x1f8                         // 000000005628: d47e0001 0201ff34 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005634: 8b01016a
	v_mul_f64_e32 v[112:113], v[112:113], v[114:115]           // 000000005638: 0ce0e570
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000563c: bf870091
	v_mul_f64_e32 v[112:113], v[112:113], v[116:117]           // 000000005640: 0ce0e970
	v_cvt_f32_f64_e32 v107, v[112:113]                         // 000000005644: 7ed61f70
	s_wait_alu depctr_sa_sdst(0)                               // 000000005648: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000564c: bf870001
	v_cndmask_b32_e64 v112, v107, 0, s1                        // 000000005650: d5010070 0005016b
	s_branch 63178                                             // 000000005658: bfa0f6ca <packed_folded_w4a8+0x1684>
	v_cvt_f64_f32_e32 v[113:114], v54                          // 00000000565c: 7ee22136
	v_cvt_f64_f32_e32 v[115:116], v105                         // 000000005660: 7ee62169
	v_cvt_f64_f32_e32 v[117:118], v107                         // 000000005664: 7eea216b
	v_cmp_eq_f32_e32 vcc_lo, 0, v54                            // 000000005668: 7c246c80
	v_cmp_class_f32_e64 s1, v107, 0x1f8                        // 00000000566c: d47e0001 0201ff6b 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005678: 8b01016a
	v_mul_f64_e32 v[113:114], v[113:114], v[115:116]           // 00000000567c: 0ce2e771
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005680: bf870091
	v_mul_f64_e32 v[113:114], v[113:114], v[117:118]           // 000000005684: 0ce2eb71
	v_cvt_f32_f64_e32 v113, v[113:114]                         // 000000005688: 7ee21f71
	s_wait_alu depctr_sa_sdst(0)                               // 00000000568c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005690: bf870001
	v_cndmask_b32_e64 v113, v113, 0, s1                        // 000000005694: d5010071 00050171
	s_branch 63191                                             // 00000000569c: bfa0f6d7 <packed_folded_w4a8+0x16fc>
	v_cvt_f64_f32_e32 v[114:115], v55                          // 0000000056a0: 7ee42137
	v_cvt_f64_f32_e32 v[116:117], v105                         // 0000000056a4: 7ee82169
	v_cvt_f64_f32_e32 v[118:119], v54                          // 0000000056a8: 7eec2136
	v_cmp_eq_f32_e32 vcc_lo, 0, v55                            // 0000000056ac: 7c246e80
	v_cmp_class_f32_e64 s1, v54, 0x1f8                         // 0000000056b0: d47e0001 0201ff36 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000056bc: 8b01016a
	v_mul_f64_e32 v[114:115], v[114:115], v[116:117]           // 0000000056c0: 0ce4e972
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000056c4: bf870091
	v_mul_f64_e32 v[114:115], v[114:115], v[118:119]           // 0000000056c8: 0ce4ed72
	v_cvt_f32_f64_e32 v105, v[114:115]                         // 0000000056cc: 7ed21f72
	s_wait_alu depctr_sa_sdst(0)                               // 0000000056d0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000056d4: bf870001
	v_cndmask_b32_e64 v114, v105, 0, s1                        // 0000000056d8: d5010072 00050169
	s_branch 63203                                             // 0000000056e0: bfa0f6e3 <packed_folded_w4a8+0x1770>
	v_cvt_f64_f32_e32 v[119:120], v40                          // 0000000056e4: 7eee2128
	v_cvt_f64_f32_e32 v[121:122], v133                         // 0000000056e8: 7ef22185
	v_cvt_f64_f32_e32 v[131:132], v118                         // 0000000056ec: 7f062176
	v_cmp_eq_f32_e32 vcc_lo, 0, v40                            // 0000000056f0: 7c245080
	v_cmp_class_f32_e64 s1, v118, 0x1f8                        // 0000000056f4: d47e0001 0201ff76 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005700: 8b01016a
	v_mul_f64_e32 v[119:120], v[119:120], v[121:122]           // 000000005704: 0ceef377
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005708: bf870091
	v_mul_f64_e32 v[119:120], v[119:120], v[131:132]           // 00000000570c: 0cef0777
	v_cvt_f32_f64_e32 v119, v[119:120]                         // 000000005710: 7eee1f77
	s_wait_alu depctr_sa_sdst(0)                               // 000000005714: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005718: bf870001
	v_cndmask_b32_e64 v131, v119, 0, s1                        // 00000000571c: d5010083 00050177
	s_branch 63423                                             // 000000005724: bfa0f7bf <packed_folded_w4a8+0x1b24>
	v_cvt_f64_f32_e32 v[120:121], v41                          // 000000005728: 7ef02129
	v_cvt_f64_f32_e32 v[122:123], v133                         // 00000000572c: 7ef42185
	v_cvt_f64_f32_e32 v[134:135], v40                          // 000000005730: 7f0c2128
	v_cmp_eq_f32_e32 vcc_lo, 0, v41                            // 000000005734: 7c245280
	v_cmp_class_f32_e64 s1, v40, 0x1f8                         // 000000005738: d47e0001 0201ff28 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005744: 8b01016a
	v_mul_f64_e32 v[120:121], v[120:121], v[122:123]           // 000000005748: 0cf0f578
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000574c: bf870091
	v_mul_f64_e32 v[120:121], v[120:121], v[134:135]           // 000000005750: 0cf10d78
	v_cvt_f32_f64_e32 v120, v[120:121]                         // 000000005754: 7ef01f78
	s_wait_alu depctr_sa_sdst(0)                               // 000000005758: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000575c: bf870001
	v_cndmask_b32_e64 v132, v120, 0, s1                        // 000000005760: d5010084 00050178
	s_branch 63436                                             // 000000005768: bfa0f7cc <packed_folded_w4a8+0x1b9c>
	v_cvt_f64_f32_e32 v[121:122], v42                          // 00000000576c: 7ef2212a
	v_cvt_f64_f32_e32 v[134:135], v133                         // 000000005770: 7f0c2185
	v_cvt_f64_f32_e32 v[136:137], v120                         // 000000005774: 7f102178
	v_cmp_eq_f32_e32 vcc_lo, 0, v42                            // 000000005778: 7c245480
	v_cmp_class_f32_e64 s1, v120, 0x1f8                        // 00000000577c: d47e0001 0201ff78 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005788: 8b01016a
	v_mul_f64_e32 v[121:122], v[121:122], v[134:135]           // 00000000578c: 0cf30d79
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005790: bf870091
	v_mul_f64_e32 v[121:122], v[121:122], v[136:137]           // 000000005794: 0cf31179
	v_cvt_f32_f64_e32 v121, v[121:122]                         // 000000005798: 7ef21f79
	s_wait_alu depctr_sa_sdst(0)                               // 00000000579c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000057a0: bf870001
	v_cndmask_b32_e64 v134, v121, 0, s1                        // 0000000057a4: d5010086 00050179
	s_branch 63449                                             // 0000000057ac: bfa0f7d9 <packed_folded_w4a8+0x1c14>
	v_cvt_f64_f32_e32 v[122:123], v43                          // 0000000057b0: 7ef4212b
	v_cvt_f64_f32_e32 v[135:136], v133                         // 0000000057b4: 7f0e2185
	v_cvt_f64_f32_e32 v[139:140], v42                          // 0000000057b8: 7f16212a
	v_cmp_eq_f32_e32 vcc_lo, 0, v43                            // 0000000057bc: 7c245680
	v_cmp_class_f32_e64 s1, v42, 0x1f8                         // 0000000057c0: d47e0001 0201ff2a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000057cc: 8b01016a
	v_mul_f64_e32 v[122:123], v[122:123], v[135:136]           // 0000000057d0: 0cf50f7a
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000057d4: bf870091
	v_mul_f64_e32 v[122:123], v[122:123], v[139:140]           // 0000000057d8: 0cf5177a
	v_cvt_f32_f64_e32 v122, v[122:123]                         // 0000000057dc: 7ef41f7a
	s_wait_alu depctr_sa_sdst(0)                               // 0000000057e0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000057e4: bf870001
	v_cndmask_b32_e64 v136, v122, 0, s1                        // 0000000057e8: d5010088 0005017a
	s_branch 63462                                             // 0000000057f0: bfa0f7e6 <packed_folded_w4a8+0x1c8c>
	v_cvt_f64_f32_e32 v[139:140], v44                          // 0000000057f4: 7f16212c
	v_cvt_f64_f32_e32 v[141:142], v133                         // 0000000057f8: 7f1a2185
	v_cvt_f64_f32_e32 v[148:149], v122                         // 0000000057fc: 7f28217a
	v_cmp_eq_f32_e32 vcc_lo, 0, v44                            // 000000005800: 7c245880
	v_cmp_class_f32_e64 s1, v122, 0x1f8                        // 000000005804: d47e0001 0201ff7a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005810: 8b01016a
	v_mul_f64_e32 v[139:140], v[139:140], v[141:142]           // 000000005814: 0d171b8b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005818: bf870091
	v_mul_f64_e32 v[139:140], v[139:140], v[148:149]           // 00000000581c: 0d17298b
	v_cvt_f32_f64_e32 v123, v[139:140]                         // 000000005820: 7ef61f8b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005824: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005828: bf870001
	v_cndmask_b32_e64 v137, v123, 0, s1                        // 00000000582c: d5010089 0005017b
	s_branch 63475                                             // 000000005834: bfa0f7f3 <packed_folded_w4a8+0x1d04>
	v_cvt_f64_f32_e32 v[139:140], v45                          // 000000005838: 7f16212d
	v_cvt_f64_f32_e32 v[141:142], v133                         // 00000000583c: 7f1a2185
	v_cvt_f64_f32_e32 v[148:149], v44                          // 000000005840: 7f28212c
	v_cmp_eq_f32_e32 vcc_lo, 0, v45                            // 000000005844: 7c245a80
	v_cmp_class_f32_e64 s1, v44, 0x1f8                         // 000000005848: d47e0001 0201ff2c 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005854: 8b01016a
	v_mul_f64_e32 v[139:140], v[139:140], v[141:142]           // 000000005858: 0d171b8b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000585c: bf870091
	v_mul_f64_e32 v[139:140], v[139:140], v[148:149]           // 000000005860: 0d17298b
	v_cvt_f32_f64_e32 v135, v[139:140]                         // 000000005864: 7f0e1f8b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005868: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000586c: bf870001
	v_cndmask_b32_e64 v139, v135, 0, s1                        // 000000005870: d501008b 00050187
	s_branch 63488                                             // 000000005878: bfa0f800 <packed_folded_w4a8+0x1d7c>
	v_cvt_f64_f32_e32 v[140:141], v46                          // 00000000587c: 7f18212e
	v_cvt_f64_f32_e32 v[142:143], v133                         // 000000005880: 7f1c2185
	v_cvt_f64_f32_e32 v[148:149], v135                         // 000000005884: 7f282187
	v_cmp_eq_f32_e32 vcc_lo, 0, v46                            // 000000005888: 7c245c80
	v_cmp_class_f32_e64 s1, v135, 0x1f8                        // 00000000588c: d47e0001 0201ff87 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005898: 8b01016a
	v_mul_f64_e32 v[140:141], v[140:141], v[142:143]           // 00000000589c: 0d191d8c
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000058a0: bf870091
	v_mul_f64_e32 v[140:141], v[140:141], v[148:149]           // 0000000058a4: 0d19298c
	v_cvt_f32_f64_e32 v140, v[140:141]                         // 0000000058a8: 7f181f8c
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058ac: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000058b0: bf870001
	v_cndmask_b32_e64 v140, v140, 0, s1                        // 0000000058b4: d501008c 0005018c
	s_branch 63501                                             // 0000000058bc: bfa0f80d <packed_folded_w4a8+0x1df4>
	v_cvt_f64_f32_e32 v[141:142], v47                          // 0000000058c0: 7f1a212f
	v_cvt_f64_f32_e32 v[148:149], v133                         // 0000000058c4: 7f282185
	v_cvt_f64_f32_e32 v[150:151], v46                          // 0000000058c8: 7f2c212e
	v_cmp_eq_f32_e32 vcc_lo, 0, v47                            // 0000000058cc: 7c245e80
	v_cmp_class_f32_e64 s1, v46, 0x1f8                         // 0000000058d0: d47e0001 0201ff2e 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000058dc: 8b01016a
	v_mul_f64_e32 v[141:142], v[141:142], v[148:149]           // 0000000058e0: 0d1b298d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000058e4: bf870091
	v_mul_f64_e32 v[141:142], v[141:142], v[150:151]           // 0000000058e8: 0d1b2d8d
	v_cvt_f32_f64_e32 v133, v[141:142]                         // 0000000058ec: 7f0a1f8d
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058f0: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000058f4: bf870001
	v_cndmask_b32_e64 v141, v133, 0, s1                        // 0000000058f8: d501008d 00050185
	s_branch 63513                                             // 000000005900: bfa0f819 <packed_folded_w4a8+0x1e68>
	v_cvt_f64_f32_e32 v[157:158], v32                          // 000000005904: 7f3a2120
	v_cvt_f64_f32_e32 v[161:162], v156                         // 000000005908: 7f42219c
	v_cvt_f64_f32_e32 v[163:164], v124                         // 00000000590c: 7f46217c
	v_cmp_eq_f32_e32 vcc_lo, 0, v32                            // 000000005910: 7c244080
	v_cmp_class_f32_e64 s1, v124, 0x1f8                        // 000000005914: d47e0001 0201ff7c 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005920: 8b01016a
	v_mul_f64_e32 v[157:158], v[157:158], v[161:162]           // 000000005924: 0d3b439d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005928: bf870091
	v_mul_f64_e32 v[157:158], v[157:158], v[163:164]           // 00000000592c: 0d3b479d
	v_cvt_f32_f64_e32 v125, v[157:158]                         // 000000005930: 7efa1f9d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005934: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005938: bf870001
	v_cndmask_b32_e64 v155, v125, 0, s1                        // 00000000593c: d501009b 0005017d
	s_branch 63731                                             // 000000005944: bfa0f8f3 <packed_folded_w4a8+0x2214>
	v_cvt_f64_f32_e32 v[157:158], v33                          // 000000005948: 7f3a2121
	v_cvt_f64_f32_e32 v[161:162], v156                         // 00000000594c: 7f42219c
	v_cvt_f64_f32_e32 v[163:164], v32                          // 000000005950: 7f462120
	v_cmp_eq_f32_e32 vcc_lo, 0, v33                            // 000000005954: 7c244280
	v_cmp_class_f32_e64 s1, v32, 0x1f8                         // 000000005958: d47e0001 0201ff20 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005964: 8b01016a
	v_mul_f64_e32 v[157:158], v[157:158], v[161:162]           // 000000005968: 0d3b439d
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 00000000596c: bf870091
	v_mul_f64_e32 v[157:158], v[157:158], v[163:164]           // 000000005970: 0d3b479d
	v_cvt_f32_f64_e32 v144, v[157:158]                         // 000000005974: 7f201f9d
	s_wait_alu depctr_sa_sdst(0)                               // 000000005978: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000597c: bf870001
	v_cndmask_b32_e64 v157, v144, 0, s1                        // 000000005980: d501009d 00050190
	s_branch 63744                                             // 000000005988: bfa0f900 <packed_folded_w4a8+0x228c>
	v_cvt_f64_f32_e32 v[158:159], v34                          // 00000000598c: 7f3c2122
	v_cvt_f64_f32_e32 v[161:162], v156                         // 000000005990: 7f42219c
	v_cvt_f64_f32_e32 v[163:164], v144                         // 000000005994: 7f462190
	v_cmp_eq_f32_e32 vcc_lo, 0, v34                            // 000000005998: 7c244480
	v_cmp_class_f32_e64 s1, v144, 0x1f8                        // 00000000599c: d47e0001 0201ff90 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000059a8: 8b01016a
	v_mul_f64_e32 v[158:159], v[158:159], v[161:162]           // 0000000059ac: 0d3d439e
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000059b0: bf870091
	v_mul_f64_e32 v[158:159], v[158:159], v[163:164]           // 0000000059b4: 0d3d479e
	v_cvt_f32_f64_e32 v145, v[158:159]                         // 0000000059b8: 7f221f9e
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059bc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000059c0: bf870001
	v_cndmask_b32_e64 v158, v145, 0, s1                        // 0000000059c4: d501009e 00050191
	s_branch 63757                                             // 0000000059cc: bfa0f90d <packed_folded_w4a8+0x2304>
	v_cvt_f64_f32_e32 v[161:162], v35                          // 0000000059d0: 7f422123
	v_cvt_f64_f32_e32 v[163:164], v156                         // 0000000059d4: 7f46219c
	v_cvt_f64_f32_e32 v[165:166], v34                          // 0000000059d8: 7f4a2122
	v_cmp_eq_f32_e32 vcc_lo, 0, v35                            // 0000000059dc: 7c244680
	v_cmp_class_f32_e64 s1, v34, 0x1f8                         // 0000000059e0: d47e0001 0201ff22 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000059ec: 8b01016a
	v_mul_f64_e32 v[161:162], v[161:162], v[163:164]           // 0000000059f0: 0d4347a1
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000059f4: bf870091
	v_mul_f64_e32 v[161:162], v[161:162], v[165:166]           // 0000000059f8: 0d434ba1
	v_cvt_f32_f64_e32 v146, v[161:162]                         // 0000000059fc: 7f241fa1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a00: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005a04: bf870001
	v_cndmask_b32_e64 v159, v146, 0, s1                        // 000000005a08: d501009f 00050192
	s_branch 63770                                             // 000000005a10: bfa0f91a <packed_folded_w4a8+0x237c>
	v_cvt_f64_f32_e32 v[161:162], v36                          // 000000005a14: 7f422124
	v_cvt_f64_f32_e32 v[163:164], v156                         // 000000005a18: 7f46219c
	v_cvt_f64_f32_e32 v[165:166], v146                         // 000000005a1c: 7f4a2192
	v_cmp_eq_f32_e32 vcc_lo, 0, v36                            // 000000005a20: 7c244880
	v_cmp_class_f32_e64 s1, v146, 0x1f8                        // 000000005a24: d47e0001 0201ff92 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005a30: 8b01016a
	v_mul_f64_e32 v[161:162], v[161:162], v[163:164]           // 000000005a34: 0d4347a1
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005a38: bf870091
	v_mul_f64_e32 v[161:162], v[161:162], v[165:166]           // 000000005a3c: 0d434ba1
	v_cvt_f32_f64_e32 v147, v[161:162]                         // 000000005a40: 7f261fa1
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a44: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005a48: bf870001
	v_cndmask_b32_e64 v161, v147, 0, s1                        // 000000005a4c: d50100a1 00050193
	s_branch 63783                                             // 000000005a54: bfa0f927 <packed_folded_w4a8+0x23f4>
	v_cvt_f64_f32_e32 v[162:163], v37                          // 000000005a58: 7f442125
	v_cvt_f64_f32_e32 v[164:165], v156                         // 000000005a5c: 7f48219c
	v_cvt_f64_f32_e32 v[166:167], v36                          // 000000005a60: 7f4c2124
	v_cmp_eq_f32_e32 vcc_lo, 0, v37                            // 000000005a64: 7c244a80
	v_cmp_class_f32_e64 s1, v36, 0x1f8                         // 000000005a68: d47e0001 0201ff24 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005a74: 8b01016a
	v_mul_f64_e32 v[162:163], v[162:163], v[164:165]           // 000000005a78: 0d4549a2
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005a7c: bf870091
	v_mul_f64_e32 v[162:163], v[162:163], v[166:167]           // 000000005a80: 0d454da2
	v_cvt_f32_f64_e32 v152, v[162:163]                         // 000000005a84: 7f301fa2
	s_wait_alu depctr_sa_sdst(0)                               // 000000005a88: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005a8c: bf870001
	v_cndmask_b32_e64 v162, v152, 0, s1                        // 000000005a90: d50100a2 00050198
	s_branch 63796                                             // 000000005a98: bfa0f934 <packed_folded_w4a8+0x246c>
	v_cvt_f64_f32_e32 v[163:164], v38                          // 000000005a9c: 7f462126
	v_cvt_f64_f32_e32 v[165:166], v156                         // 000000005aa0: 7f4a219c
	v_cvt_f64_f32_e32 v[167:168], v152                         // 000000005aa4: 7f4e2198
	v_cmp_eq_f32_e32 vcc_lo, 0, v38                            // 000000005aa8: 7c244c80
	v_cmp_class_f32_e64 s1, v152, 0x1f8                        // 000000005aac: d47e0001 0201ff98 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005ab8: 8b01016a
	v_mul_f64_e32 v[163:164], v[163:164], v[165:166]           // 000000005abc: 0d474ba3
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ac0: bf870091
	v_mul_f64_e32 v[163:164], v[163:164], v[167:168]           // 000000005ac4: 0d474fa3
	v_cvt_f32_f64_e32 v153, v[163:164]                         // 000000005ac8: 7f321fa3
	s_wait_alu depctr_sa_sdst(0)                               // 000000005acc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ad0: bf870001
	v_cndmask_b32_e64 v163, v153, 0, s1                        // 000000005ad4: d50100a3 00050199
	s_branch 63809                                             // 000000005adc: bfa0f941 <packed_folded_w4a8+0x24e4>
	v_cvt_f64_f32_e32 v[152:153], v39                          // 000000005ae0: 7f302127
	v_cvt_f64_f32_e32 v[164:165], v156                         // 000000005ae4: 7f48219c
	v_cvt_f64_f32_e32 v[166:167], v38                          // 000000005ae8: 7f4c2126
	v_cmp_eq_f32_e32 vcc_lo, 0, v39                            // 000000005aec: 7c244e80
	v_cmp_class_f32_e64 s1, v38, 0x1f8                         // 000000005af0: d47e0001 0201ff26 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005afc: 8b01016a
	v_mul_f64_e32 v[152:153], v[152:153], v[164:165]           // 000000005b00: 0d314998
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b04: bf870091
	v_mul_f64_e32 v[152:153], v[152:153], v[166:167]           // 000000005b08: 0d314d98
	v_cvt_f32_f64_e32 v152, v[152:153]                         // 000000005b0c: 7f301f98
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b10: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005b14: bf870001
	v_cndmask_b32_e64 v164, v152, 0, s1                        // 000000005b18: d50100a4 00050198
	s_branch 63821                                             // 000000005b20: bfa0f94d <packed_folded_w4a8+0x2558>
	v_cvt_f64_f32_e32 v[163:164], v24                          // 000000005b24: 7f462118
	v_cvt_f64_f32_e32 v[165:166], v67                          // 000000005b28: 7f4a2143
	v_cvt_f64_f32_e32 v[167:168], v162                         // 000000005b2c: 7f4e21a2
	v_cmp_eq_f32_e32 vcc_lo, 0, v24                            // 000000005b30: 7c243080
	v_cmp_class_f32_e64 s1, v162, 0x1f8                        // 000000005b34: d47e0001 0201ffa2 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005b40: 8b01016a
	v_mul_f64_e32 v[163:164], v[163:164], v[165:166]           // 000000005b44: 0d474ba3
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b48: bf870091
	v_mul_f64_e32 v[163:164], v[163:164], v[167:168]           // 000000005b4c: 0d474fa3
	v_cvt_f32_f64_e32 v66, v[163:164]                          // 000000005b50: 7e841fa3
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b54: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005b58: bf870001
	v_cndmask_b32_e64 v66, v66, 0, s1                          // 000000005b5c: d5010042 00050142
	s_branch 64039                                             // 000000005b64: bfa0fa27 <packed_folded_w4a8+0x2904>
	v_cvt_f64_f32_e32 v[162:163], v25                          // 000000005b68: 7f442119
	v_cvt_f64_f32_e32 v[164:165], v67                          // 000000005b6c: 7f482143
	v_cvt_f64_f32_e32 v[166:167], v68                          // 000000005b70: 7f4c2144
	v_cmp_eq_f32_e32 vcc_lo, 0, v25                            // 000000005b74: 7c243280
	v_cmp_class_f32_e64 s1, v68, 0x1f8                         // 000000005b78: d47e0001 0201ff44 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005b84: 8b01016a
	v_mul_f64_e32 v[162:163], v[162:163], v[164:165]           // 000000005b88: 0d4549a2
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005b8c: bf870091
	v_mul_f64_e32 v[162:163], v[162:163], v[166:167]           // 000000005b90: 0d454da2
	v_cvt_f32_f64_e32 v24, v[162:163]                          // 000000005b94: 7e301fa2
	s_wait_alu depctr_sa_sdst(0)                               // 000000005b98: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005b9c: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s1                          // 000000005ba0: d5010018 00050118
	s_branch 64038                                             // 000000005ba8: bfa0fa26 <packed_folded_w4a8+0x2944>
	v_cvt_f64_f32_e32 v[68:69], v26                            // 000000005bac: 7e88211a
	v_cvt_f64_f32_e32 v[162:163], v67                          // 000000005bb0: 7f442143
	v_cvt_f64_f32_e32 v[164:165], v56                          // 000000005bb4: 7f482138
	v_cmp_eq_f32_e32 vcc_lo, 0, v26                            // 000000005bb8: 7c243480
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005bbc: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005bc8: 8b01016a
	v_mul_f64_e32 v[68:69], v[68:69], v[162:163]               // 000000005bcc: 0c894544
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005bd0: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[164:165]               // 000000005bd4: 0c894944
	v_cvt_f32_f64_e32 v25, v[68:69]                            // 000000005bd8: 7e321f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005bdc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005be0: bf870001
	v_cndmask_b32_e64 v25, v25, 0, s1                          // 000000005be4: d5010019 00050119
	s_branch 64037                                             // 000000005bec: bfa0fa25 <packed_folded_w4a8+0x2984>
	v_cvt_f64_f32_e32 v[68:69], v27                            // 000000005bf0: 7e88211b
	v_cvt_f64_f32_e32 v[70:71], v67                            // 000000005bf4: 7e8c2143
	v_cvt_f64_f32_e32 v[162:163], v56                          // 000000005bf8: 7f442138
	v_cmp_eq_f32_e32 vcc_lo, 0, v27                            // 000000005bfc: 7c243680
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005c00: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005c0c: 8b01016a
	v_mul_f64_e32 v[68:69], v[68:69], v[70:71]                 // 000000005c10: 0c888d44
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c14: bf870091
	v_mul_f64_e32 v[68:69], v[68:69], v[162:163]               // 000000005c18: 0c894544
	v_cvt_f32_f64_e32 v26, v[68:69]                            // 000000005c1c: 7e341f44
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c20: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005c24: bf870001
	v_cndmask_b32_e64 v26, v26, 0, s1                          // 000000005c28: d501001a 0005011a
	s_branch 64036                                             // 000000005c30: bfa0fa24 <packed_folded_w4a8+0x29c4>
	v_cvt_f64_f32_e32 v[57:58], v28                            // 000000005c34: 7e72211c
	v_cvt_f64_f32_e32 v[68:69], v67                            // 000000005c38: 7e882143
	v_cvt_f64_f32_e32 v[70:71], v56                            // 000000005c3c: 7e8c2138
	v_cmp_eq_f32_e32 vcc_lo, 0, v28                            // 000000005c40: 7c243880
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005c44: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005c50: 8b01016a
	v_mul_f64_e32 v[57:58], v[57:58], v[68:69]                 // 000000005c54: 0c728939
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c58: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[70:71]                 // 000000005c5c: 0c728d39
	v_cvt_f32_f64_e32 v27, v[57:58]                            // 000000005c60: 7e361f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000005c64: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005c68: bf870001
	v_cndmask_b32_e64 v27, v27, 0, s1                          // 000000005c6c: d501001b 0005011b
	s_branch 64035                                             // 000000005c74: bfa0fa23 <packed_folded_w4a8+0x2a04>
	v_cvt_f64_f32_e32 v[57:58], v29                            // 000000005c78: 7e72211d
	v_cvt_f64_f32_e32 v[68:69], v67                            // 000000005c7c: 7e882143
	v_cvt_f64_f32_e32 v[70:71], v56                            // 000000005c80: 7e8c2138
	v_cmp_eq_f32_e32 vcc_lo, 0, v29                            // 000000005c84: 7c243a80
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005c88: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005c94: 8b01016a
	v_mul_f64_e32 v[57:58], v[57:58], v[68:69]                 // 000000005c98: 0c728939
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005c9c: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[70:71]                 // 000000005ca0: 0c728d39
	v_cvt_f32_f64_e32 v28, v[57:58]                            // 000000005ca4: 7e381f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ca8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005cac: bf870001
	v_cndmask_b32_e64 v28, v28, 0, s1                          // 000000005cb0: d501001c 0005011c
	s_branch 64034                                             // 000000005cb8: bfa0fa22 <packed_folded_w4a8+0x2a44>
	v_cvt_f64_f32_e32 v[57:58], v30                            // 000000005cbc: 7e72211e
	v_cvt_f64_f32_e32 v[59:60], v67                            // 000000005cc0: 7e762143
	v_cvt_f64_f32_e32 v[68:69], v56                            // 000000005cc4: 7e882138
	v_cmp_eq_f32_e32 vcc_lo, 0, v30                            // 000000005cc8: 7c243c80
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005ccc: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005cd8: 8b01016a
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 000000005cdc: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ce0: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[68:69]                 // 000000005ce4: 0c728939
	v_cvt_f32_f64_e32 v29, v[57:58]                            // 000000005ce8: 7e3a1f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000005cec: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005cf0: bf870001
	v_cndmask_b32_e64 v29, v29, 0, s1                          // 000000005cf4: d501001d 0005011d
	s_branch 64033                                             // 000000005cfc: bfa0fa21 <packed_folded_w4a8+0x2a84>
	v_cvt_f64_f32_e32 v[57:58], v31                            // 000000005d00: 7e72211f
	v_cvt_f64_f32_e32 v[59:60], v67                            // 000000005d04: 7e762143
	v_cvt_f64_f32_e32 v[67:68], v56                            // 000000005d08: 7e862138
	v_cmp_eq_f32_e32 vcc_lo, 0, v31                            // 000000005d0c: 7c243e80
	v_cmp_class_f32_e64 s1, v56, 0x1f8                         // 000000005d10: d47e0001 0201ff38 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005d1c: 8b01016a
	v_mul_f64_e32 v[57:58], v[57:58], v[59:60]                 // 000000005d20: 0c727739
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005d24: bf870091
	v_mul_f64_e32 v[57:58], v[57:58], v[67:68]                 // 000000005d28: 0c728739
	v_cvt_f32_f64_e32 v30, v[57:58]                            // 000000005d2c: 7e3c1f39
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d30: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005d34: bf870001
	v_cndmask_b32_e64 v30, v30, 0, s1                          // 000000005d38: d501001e 0005011e
	s_branch 64032                                             // 000000005d40: bfa0fa20 <packed_folded_w4a8+0x2ac4>
	v_cvt_f64_f32_e32 v[27:28], v16                            // 000000005d44: 7e362110
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005d48: 7e3a2119
	v_cvt_f64_f32_e32 v[56:57], v26                            // 000000005d4c: 7e70211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v16                            // 000000005d50: 7c242080
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005d54: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005d60: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005d64: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005d68: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[56:57]                 // 000000005d6c: 0c36711b
	v_cvt_f32_f64_e32 v24, v[27:28]                            // 000000005d70: 7e301f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005d74: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005d78: bf870001
	v_cndmask_b32_e64 v24, v24, 0, s1                          // 000000005d7c: d5010018 00050118
	s_branch 64141                                             // 000000005d84: bfa0fa8d <packed_folded_w4a8+0x2cbc>
	v_cvt_f64_f32_e32 v[27:28], v17                            // 000000005d88: 7e362111
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005d8c: 7e3a2119
	v_cvt_f64_f32_e32 v[56:57], v26                            // 000000005d90: 7e70211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v17                            // 000000005d94: 7c242280
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005d98: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005da4: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005da8: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005dac: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[56:57]                 // 000000005db0: 0c36711b
	v_cvt_f32_f64_e32 v16, v[27:28]                            // 000000005db4: 7e201f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005db8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005dbc: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s1                          // 000000005dc0: d5010010 00050110
	s_branch 64140                                             // 000000005dc8: bfa0fa8c <packed_folded_w4a8+0x2cfc>
	v_cvt_f64_f32_e32 v[27:28], v18                            // 000000005dcc: 7e362112
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005dd0: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005dd4: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v18                            // 000000005dd8: 7c242480
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005ddc: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005de8: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005dec: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005df0: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005df4: 0c36611b
	v_cvt_f32_f64_e32 v17, v[27:28]                            // 000000005df8: 7e221f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005dfc: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005e00: bf870001
	v_cndmask_b32_e64 v17, v17, 0, s1                          // 000000005e04: d5010011 00050111
	s_branch 64139                                             // 000000005e0c: bfa0fa8b <packed_folded_w4a8+0x2d3c>
	v_cvt_f64_f32_e32 v[27:28], v19                            // 000000005e10: 7e362113
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005e14: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005e18: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v19                            // 000000005e1c: 7c242680
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005e20: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005e2c: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005e30: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e34: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005e38: 0c36611b
	v_cvt_f32_f64_e32 v18, v[27:28]                            // 000000005e3c: 7e241f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e40: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005e44: bf870001
	v_cndmask_b32_e64 v18, v18, 0, s1                          // 000000005e48: d5010012 00050112
	s_branch 64138                                             // 000000005e50: bfa0fa8a <packed_folded_w4a8+0x2d7c>
	v_cvt_f64_f32_e32 v[27:28], v20                            // 000000005e54: 7e362114
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005e58: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005e5c: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v20                            // 000000005e60: 7c242880
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005e64: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005e70: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005e74: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005e78: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005e7c: 0c36611b
	v_cvt_f32_f64_e32 v19, v[27:28]                            // 000000005e80: 7e261f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005e84: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005e88: bf870001
	v_cndmask_b32_e64 v19, v19, 0, s1                          // 000000005e8c: d5010013 00050113
	s_branch 64137                                             // 000000005e94: bfa0fa89 <packed_folded_w4a8+0x2dbc>
	v_cvt_f64_f32_e32 v[27:28], v21                            // 000000005e98: 7e362115
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005e9c: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005ea0: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v21                            // 000000005ea4: 7c242a80
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005ea8: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005eb4: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005eb8: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005ebc: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005ec0: 0c36611b
	v_cvt_f32_f64_e32 v20, v[27:28]                            // 000000005ec4: 7e281f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005ec8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005ecc: bf870001
	v_cndmask_b32_e64 v20, v20, 0, s1                          // 000000005ed0: d5010014 00050114
	s_branch 64136                                             // 000000005ed8: bfa0fa88 <packed_folded_w4a8+0x2dfc>
	v_cvt_f64_f32_e32 v[27:28], v22                            // 000000005edc: 7e362116
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005ee0: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005ee4: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v22                            // 000000005ee8: 7c242c80
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005eec: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005ef8: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005efc: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f00: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005f04: 0c36611b
	v_cvt_f32_f64_e32 v21, v[27:28]                            // 000000005f08: 7e2a1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f0c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f10: bf870001
	v_cndmask_b32_e64 v21, v21, 0, s1                          // 000000005f14: d5010015 00050115
	s_branch 64135                                             // 000000005f1c: bfa0fa87 <packed_folded_w4a8+0x2e3c>
	v_cvt_f64_f32_e32 v[27:28], v23                            // 000000005f20: 7e362117
	v_cvt_f64_f32_e32 v[29:30], v25                            // 000000005f24: 7e3a2119
	v_cvt_f64_f32_e32 v[48:49], v26                            // 000000005f28: 7e60211a
	v_cmp_eq_f32_e32 vcc_lo, 0, v23                            // 000000005f2c: 7c242e80
	v_cmp_class_f32_e64 s1, v26, 0x1f8                         // 000000005f30: d47e0001 0201ff1a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005f3c: 8b01016a
	v_mul_f64_e32 v[27:28], v[27:28], v[29:30]                 // 000000005f40: 0c363b1b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f44: bf870091
	v_mul_f64_e32 v[27:28], v[27:28], v[48:49]                 // 000000005f48: 0c36611b
	v_cvt_f32_f64_e32 v22, v[27:28]                            // 000000005f4c: 7e2c1f1b
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f50: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f54: bf870001
	v_cndmask_b32_e64 v22, v22, 0, s1                          // 000000005f58: d5010016 00050116
	s_branch 64134                                             // 000000005f60: bfa0fa86 <packed_folded_w4a8+0x2e7c>
	v_cvt_f64_f32_e32 v[19:20], v8                             // 000000005f64: 7e262108
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000005f68: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000005f6c: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v8                             // 000000005f70: 7c241080
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000005f74: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005f80: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000005f84: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005f88: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000005f8c: 0c262f13
	v_cvt_f32_f64_e32 v16, v[19:20]                            // 000000005f90: 7e201f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000005f94: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005f98: bf870001
	v_cndmask_b32_e64 v16, v16, 0, s1                          // 000000005f9c: d5010010 00050110
	s_branch 64243                                             // 000000005fa4: bfa0faf3 <packed_folded_w4a8+0x3074>
	v_cvt_f64_f32_e32 v[19:20], v9                             // 000000005fa8: 7e262109
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000005fac: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000005fb0: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v9                             // 000000005fb4: 7c241280
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000005fb8: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000005fc4: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000005fc8: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000005fcc: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000005fd0: 0c262f13
	v_cvt_f32_f64_e32 v8, v[19:20]                             // 000000005fd4: 7e101f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000005fd8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000005fdc: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s1                            // 000000005fe0: d5010008 00050108
	s_branch 64242                                             // 000000005fe8: bfa0faf2 <packed_folded_w4a8+0x30b4>
	v_cvt_f64_f32_e32 v[19:20], v10                            // 000000005fec: 7e26210a
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000005ff0: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000005ff4: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v10                            // 000000005ff8: 7c241480
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000005ffc: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006008: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000600c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006010: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000006014: 0c262f13
	v_cvt_f32_f64_e32 v9, v[19:20]                             // 000000006018: 7e121f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000601c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006020: bf870001
	v_cndmask_b32_e64 v9, v9, 0, s1                            // 000000006024: d5010009 00050109
	s_branch 64241                                             // 00000000602c: bfa0faf1 <packed_folded_w4a8+0x30f4>
	v_cvt_f64_f32_e32 v[19:20], v11                            // 000000006030: 7e26210b
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000006034: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000006038: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v11                            // 00000000603c: 7c241680
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000006040: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 00000000604c: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000006050: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006054: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000006058: 0c262f13
	v_cvt_f32_f64_e32 v10, v[19:20]                            // 00000000605c: 7e141f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000006060: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006064: bf870001
	v_cndmask_b32_e64 v10, v10, 0, s1                          // 000000006068: d501000a 0005010a
	s_branch 64240                                             // 000000006070: bfa0faf0 <packed_folded_w4a8+0x3134>
	v_cvt_f64_f32_e32 v[19:20], v12                            // 000000006074: 7e26210c
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000006078: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 00000000607c: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v12                            // 000000006080: 7c241880
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000006084: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006090: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000006094: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006098: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 00000000609c: 0c262f13
	v_cvt_f32_f64_e32 v11, v[19:20]                            // 0000000060a0: 7e161f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060a4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000060a8: bf870001
	v_cndmask_b32_e64 v11, v11, 0, s1                          // 0000000060ac: d501000b 0005010b
	s_branch 64239                                             // 0000000060b4: bfa0faef <packed_folded_w4a8+0x3174>
	v_cvt_f64_f32_e32 v[19:20], v13                            // 0000000060b8: 7e26210d
	v_cvt_f64_f32_e32 v[21:22], v17                            // 0000000060bc: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 0000000060c0: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v13                            // 0000000060c4: 7c241a80
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 0000000060c8: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000060d4: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 0000000060d8: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000060dc: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 0000000060e0: 0c262f13
	v_cvt_f32_f64_e32 v12, v[19:20]                            // 0000000060e4: 7e181f13
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060e8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000060ec: bf870001
	v_cndmask_b32_e64 v12, v12, 0, s1                          // 0000000060f0: d501000c 0005010c
	s_branch 64238                                             // 0000000060f8: bfa0faee <packed_folded_w4a8+0x31b4>
	v_cvt_f64_f32_e32 v[19:20], v14                            // 0000000060fc: 7e26210e
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000006100: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000006104: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v14                            // 000000006108: 7c241c80
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 00000000610c: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006118: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 00000000611c: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006120: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000006124: 0c262f13
	v_cvt_f32_f64_e32 v13, v[19:20]                            // 000000006128: 7e1a1f13
	s_wait_alu depctr_sa_sdst(0)                               // 00000000612c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006130: bf870001
	v_cndmask_b32_e64 v13, v13, 0, s1                          // 000000006134: d501000d 0005010d
	s_branch 64237                                             // 00000000613c: bfa0faed <packed_folded_w4a8+0x31f4>
	v_cvt_f64_f32_e32 v[19:20], v15                            // 000000006140: 7e26210f
	v_cvt_f64_f32_e32 v[21:22], v17                            // 000000006144: 7e2a2111
	v_cvt_f64_f32_e32 v[23:24], v18                            // 000000006148: 7e2e2112
	v_cmp_eq_f32_e32 vcc_lo, 0, v15                            // 00000000614c: 7c241e80
	v_cmp_class_f32_e64 s1, v18, 0x1f8                         // 000000006150: d47e0001 0201ff12 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 00000000615c: 8b01016a
	v_mul_f64_e32 v[19:20], v[19:20], v[21:22]                 // 000000006160: 0c262b13
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006164: bf870091
	v_mul_f64_e32 v[19:20], v[19:20], v[23:24]                 // 000000006168: 0c262f13
	v_cvt_f32_f64_e32 v14, v[19:20]                            // 00000000616c: 7e1c1f13
	s_wait_alu depctr_sa_sdst(0)                               // 000000006170: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006174: bf870001
	v_cndmask_b32_e64 v14, v14, 0, s1                          // 000000006178: d501000e 0005010e
	s_branch 64236                                             // 000000006180: bfa0faec <packed_folded_w4a8+0x3234>
	v_cvt_f64_f32_e32 v[11:12], v0                             // 000000006184: 7e162100
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006188: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 00000000618c: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v0                             // 000000006190: 7c240080
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 000000006194: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000061a0: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000061a4: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000061a8: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000061ac: 0c161f0b
	v_cvt_f32_f64_e32 v8, v[11:12]                             // 0000000061b0: 7e101f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061b4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061b8: bf870001
	v_cndmask_b32_e64 v8, v8, 0, s1                            // 0000000061bc: d5010008 00050108
	s_branch 64345                                             // 0000000061c4: bfa0fb59 <packed_folded_w4a8+0x342c>
	v_cvt_f64_f32_e32 v[11:12], v1                             // 0000000061c8: 7e162101
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000061cc: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000061d0: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v1                             // 0000000061d4: 7c240280
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 0000000061d8: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000061e4: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000061e8: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000061ec: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000061f0: 0c161f0b
	v_cvt_f32_f64_e32 v0, v[11:12]                             // 0000000061f4: 7e001f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000061f8: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000061fc: bf870001
	v_cndmask_b32_e64 v0, v0, 0, s1                            // 000000006200: d5010000 00050100
	s_branch 64344                                             // 000000006208: bfa0fb58 <packed_folded_w4a8+0x346c>
	v_cvt_f64_f32_e32 v[11:12], v2                             // 00000000620c: 7e162102
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006210: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000006214: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v2                             // 000000006218: 7c240480
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 00000000621c: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006228: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000622c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006230: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000006234: 0c161f0b
	v_cvt_f32_f64_e32 v1, v[11:12]                             // 000000006238: 7e021f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000623c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006240: bf870001
	v_cndmask_b32_e64 v1, v1, 0, s1                            // 000000006244: d5010001 00050101
	s_branch 64343                                             // 00000000624c: bfa0fb57 <packed_folded_w4a8+0x34ac>
	v_cvt_f64_f32_e32 v[11:12], v3                             // 000000006250: 7e162103
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006254: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000006258: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v3                             // 00000000625c: 7c240680
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 000000006260: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 00000000626c: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000006270: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006274: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000006278: 0c161f0b
	v_cvt_f32_f64_e32 v2, v[11:12]                             // 00000000627c: 7e041f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006280: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006284: bf870001
	v_cndmask_b32_e64 v2, v2, 0, s1                            // 000000006288: d5010002 00050102
	s_branch 64342                                             // 000000006290: bfa0fb56 <packed_folded_w4a8+0x34ec>
	v_cvt_f64_f32_e32 v[11:12], v4                             // 000000006294: 7e162104
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006298: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 00000000629c: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v4                             // 0000000062a0: 7c240880
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 0000000062a4: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000062b0: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000062b4: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062b8: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 0000000062bc: 0c161f0b
	v_cvt_f32_f64_e32 v3, v[11:12]                             // 0000000062c0: 7e061f0b
	s_wait_alu depctr_sa_sdst(0)                               // 0000000062c4: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 0000000062c8: bf870001
	v_cndmask_b32_e64 v3, v3, 0, s1                            // 0000000062cc: d5010003 00050103
	s_branch 64341                                             // 0000000062d4: bfa0fb55 <packed_folded_w4a8+0x352c>
	v_cvt_f64_f32_e32 v[11:12], v5                             // 0000000062d8: 7e162105
	v_cvt_f64_f32_e32 v[13:14], v9                             // 0000000062dc: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 0000000062e0: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v5                             // 0000000062e4: 7c240a80
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 0000000062e8: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 0000000062f4: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 0000000062f8: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 0000000062fc: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000006300: 0c161f0b
	v_cvt_f32_f64_e32 v4, v[11:12]                             // 000000006304: 7e081f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006308: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 00000000630c: bf870001
	v_cndmask_b32_e64 v4, v4, 0, s1                            // 000000006310: d5010004 00050104
	s_branch 64340                                             // 000000006318: bfa0fb54 <packed_folded_w4a8+0x356c>
	v_cvt_f64_f32_e32 v[11:12], v6                             // 00000000631c: 7e162106
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006320: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000006324: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v6                             // 000000006328: 7c240c80
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 00000000632c: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 000000006338: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 00000000633c: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006340: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000006344: 0c161f0b
	v_cvt_f32_f64_e32 v5, v[11:12]                             // 000000006348: 7e0a1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 00000000634c: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006350: bf870001
	v_cndmask_b32_e64 v5, v5, 0, s1                            // 000000006354: d5010005 00050105
	s_branch 64339                                             // 00000000635c: bfa0fb53 <packed_folded_w4a8+0x35ac>
	v_cvt_f64_f32_e32 v[11:12], v7                             // 000000006360: 7e162107
	v_cvt_f64_f32_e32 v[13:14], v9                             // 000000006364: 7e1a2109
	v_cvt_f64_f32_e32 v[15:16], v10                            // 000000006368: 7e1e210a
	v_cmp_eq_f32_e32 vcc_lo, 0, v7                             // 00000000636c: 7c240e80
	v_cmp_class_f32_e64 s1, v10, 0x1f8                         // 000000006370: d47e0001 0201ff0a 000001f8
	s_and_b32 s1, vcc_lo, s1                                   // 00000000637c: 8b01016a
	v_mul_f64_e32 v[11:12], v[11:12], v[13:14]                 // 000000006380: 0c161b0b
	s_delay_alu instid0(valu_dep_1) | instskip(next) | instid1(valu_dep_1)// 000000006384: bf870091
	v_mul_f64_e32 v[11:12], v[11:12], v[15:16]                 // 000000006388: 0c161f0b
	v_cvt_f32_f64_e32 v6, v[11:12]                             // 00000000638c: 7e0c1f0b
	s_wait_alu depctr_sa_sdst(0)                               // 000000006390: bf88ff9e
	s_delay_alu instid0(valu_dep_1)                            // 000000006394: bf870001
	v_cndmask_b32_e64 v6, v6, 0, s1                            // 000000006398: d5010006 00050106
	s_branch 64338                                             // 0000000063a0: bfa0fb52 <packed_folded_w4a8+0x35ec>
	s_code_end                                                 // 0000000063a4: bf9f0000
	s_code_end                                                 // 0000000063a8: bf9f0000
	s_code_end                                                 // 0000000063ac: bf9f0000
	s_code_end                                                 // 0000000063b0: bf9f0000
	s_code_end                                                 // 0000000063b4: bf9f0000
	s_code_end                                                 // 0000000063b8: bf9f0000
	s_code_end                                                 // 0000000063bc: bf9f0000
	s_code_end                                                 // 0000000063c0: bf9f0000
	s_code_end                                                 // 0000000063c4: bf9f0000
	s_code_end                                                 // 0000000063c8: bf9f0000
	s_code_end                                                 // 0000000063cc: bf9f0000
	s_code_end                                                 // 0000000063d0: bf9f0000
	s_code_end                                                 // 0000000063d4: bf9f0000
	s_code_end                                                 // 0000000063d8: bf9f0000
	s_code_end                                                 // 0000000063dc: bf9f0000
	s_code_end                                                 // 0000000063e0: bf9f0000
	s_code_end                                                 // 0000000063e4: bf9f0000
	s_code_end                                                 // 0000000063e8: bf9f0000
	s_code_end                                                 // 0000000063ec: bf9f0000
	s_code_end                                                 // 0000000063f0: bf9f0000
	s_code_end                                                 // 0000000063f4: bf9f0000
	s_code_end                                                 // 0000000063f8: bf9f0000
	s_code_end                                                 // 0000000063fc: bf9f0000
	s_code_end                                                 // 000000006400: bf9f0000
	s_code_end                                                 // 000000006404: bf9f0000
	s_code_end                                                 // 000000006408: bf9f0000
	s_code_end                                                 // 00000000640c: bf9f0000
	s_code_end                                                 // 000000006410: bf9f0000
	s_code_end                                                 // 000000006414: bf9f0000
	s_code_end                                                 // 000000006418: bf9f0000
	s_code_end                                                 // 00000000641c: bf9f0000
	s_code_end                                                 // 000000006420: bf9f0000
	s_code_end                                                 // 000000006424: bf9f0000
	s_code_end                                                 // 000000006428: bf9f0000
	s_code_end                                                 // 00000000642c: bf9f0000
	s_code_end                                                 // 000000006430: bf9f0000
	s_code_end                                                 // 000000006434: bf9f0000
	s_code_end                                                 // 000000006438: bf9f0000
	s_code_end                                                 // 00000000643c: bf9f0000
	s_code_end                                                 // 000000006440: bf9f0000
	s_code_end                                                 // 000000006444: bf9f0000
	s_code_end                                                 // 000000006448: bf9f0000
	s_code_end                                                 // 00000000644c: bf9f0000
	s_code_end                                                 // 000000006450: bf9f0000
	s_code_end                                                 // 000000006454: bf9f0000
	s_code_end                                                 // 000000006458: bf9f0000
	s_code_end                                                 // 00000000645c: bf9f0000
	s_code_end                                                 // 000000006460: bf9f0000
	s_code_end                                                 // 000000006464: bf9f0000
	s_code_end                                                 // 000000006468: bf9f0000
	s_code_end                                                 // 00000000646c: bf9f0000
	s_code_end                                                 // 000000006470: bf9f0000
	s_code_end                                                 // 000000006474: bf9f0000
	s_code_end                                                 // 000000006478: bf9f0000
	s_code_end                                                 // 00000000647c: bf9f0000
	s_code_end                                                 // 000000006480: bf9f0000
	s_code_end                                                 // 000000006484: bf9f0000
	s_code_end                                                 // 000000006488: bf9f0000
	s_code_end                                                 // 00000000648c: bf9f0000
	s_code_end                                                 // 000000006490: bf9f0000
	s_code_end                                                 // 000000006494: bf9f0000
	s_code_end                                                 // 000000006498: bf9f0000
	s_code_end                                                 // 00000000649c: bf9f0000
	s_code_end                                                 // 0000000064a0: bf9f0000
	s_code_end                                                 // 0000000064a4: bf9f0000
	s_code_end                                                 // 0000000064a8: bf9f0000
	s_code_end                                                 // 0000000064ac: bf9f0000
	s_code_end                                                 // 0000000064b0: bf9f0000
	s_code_end                                                 // 0000000064b4: bf9f0000
	s_code_end                                                 // 0000000064b8: bf9f0000
	s_code_end                                                 // 0000000064bc: bf9f0000
	s_code_end                                                 // 0000000064c0: bf9f0000
	s_code_end                                                 // 0000000064c4: bf9f0000
	s_code_end                                                 // 0000000064c8: bf9f0000
	s_code_end                                                 // 0000000064cc: bf9f0000
	s_code_end                                                 // 0000000064d0: bf9f0000
	s_code_end                                                 // 0000000064d4: bf9f0000
	s_code_end                                                 // 0000000064d8: bf9f0000
	s_code_end                                                 // 0000000064dc: bf9f0000
	s_code_end                                                 // 0000000064e0: bf9f0000
	s_code_end                                                 // 0000000064e4: bf9f0000
	s_code_end                                                 // 0000000064e8: bf9f0000
	s_code_end                                                 // 0000000064ec: bf9f0000
	s_code_end                                                 // 0000000064f0: bf9f0000
	s_code_end                                                 // 0000000064f4: bf9f0000
	s_code_end                                                 // 0000000064f8: bf9f0000
	s_code_end                                                 // 0000000064fc: bf9f0000
	s_code_end                                                 // 000000006500: bf9f0000
	s_code_end                                                 // 000000006504: bf9f0000
	s_code_end                                                 // 000000006508: bf9f0000
	s_code_end                                                 // 00000000650c: bf9f0000
	s_code_end                                                 // 000000006510: bf9f0000
	s_code_end                                                 // 000000006514: bf9f0000
	s_code_end                                                 // 000000006518: bf9f0000
	s_code_end                                                 // 00000000651c: bf9f0000
	s_code_end                                                 // 000000006520: bf9f0000
	s_code_end                                                 // 000000006524: bf9f0000
	s_code_end                                                 // 000000006528: bf9f0000
	s_code_end                                                 // 00000000652c: bf9f0000
	s_code_end                                                 // 000000006530: bf9f0000
	s_code_end                                                 // 000000006534: bf9f0000
	s_code_end                                                 // 000000006538: bf9f0000
	s_code_end                                                 // 00000000653c: bf9f0000
	s_code_end                                                 // 000000006540: bf9f0000
	s_code_end                                                 // 000000006544: bf9f0000
	s_code_end                                                 // 000000006548: bf9f0000
	s_code_end                                                 // 00000000654c: bf9f0000
	s_code_end                                                 // 000000006550: bf9f0000
	s_code_end                                                 // 000000006554: bf9f0000
	s_code_end                                                 // 000000006558: bf9f0000
	s_code_end                                                 // 00000000655c: bf9f0000
	s_code_end                                                 // 000000006560: bf9f0000
	s_code_end                                                 // 000000006564: bf9f0000
	s_code_end                                                 // 000000006568: bf9f0000
	s_code_end                                                 // 00000000656c: bf9f0000
	s_code_end                                                 // 000000006570: bf9f0000
	s_code_end                                                 // 000000006574: bf9f0000
	s_code_end                                                 // 000000006578: bf9f0000
	s_code_end                                                 // 00000000657c: bf9f0000
