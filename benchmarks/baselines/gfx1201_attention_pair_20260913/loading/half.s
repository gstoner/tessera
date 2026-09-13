
/tmp/gfx1201-attention-loading/half.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001b00 <attention_forward>:
	v_mov_b32_e32 v1, 0                                        // 000000001B00: 7E020280
	s_mov_b32 s4, ttmp7                                        // 000000001B04: BE840073
	s_mov_b32 s3, 0                                            // 000000001B08: BE830080
	v_dual_mov_b32 v13, v0 :: v_dual_mov_b32 v2, v0            // 000000001B0C: CA100100 0D020100
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B14: BF870002
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v14, v1            // 000000001B18: CA100101 030E0101
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001B20: BF870001
	v_cmp_lt_u64_e32 vcc_lo, 0x3df, v[2:3]                     // 000000001B24: 7CB204FF 000003DF
	v_lshlrev_b32_e32 v4, 2, v2                                // 000000001B2C: 30080482
	v_add_co_u32 v2, s2, v2, 32                                // 000000001B30: D7000202 02014102
	s_wait_alu depctr_va_sdst(0)                               // 000000001B38: BF88F19F
	v_add_co_ci_u32_e64 v3, null, 0, v3, s2                    // 000000001B3C: D5207C03 000A0680
	s_or_b32 s3, vcc_lo, s3                                    // 000000001B44: 8C03036A
	ds_store_b32 v4, v1                                        // 000000001B48: D8340000 00000104
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B50: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s3                        // 000000001B54: 917E037E
	s_cbranch_execnz 65521                                     // 000000001B58: BFA6FFF1 <attention_forward+0x20>
	s_or_b32 exec_lo, exec_lo, s3                              // 000000001B5C: 8C7E037E
	v_cmp_gt_u32_e64 s2, 16, v0                                // 000000001B60: D44C0002 02020090
	v_lshlrev_b32_e32 v96, 2, v0                               // 000000001B68: 30C00082
	s_and_saveexec_b32 s3, s2                                  // 000000001B6C: BE832002
	s_cbranch_execz 9                                          // 000000001B70: BFA50009 <attention_forward+0x98>
	v_dual_mov_b32 v2, 0xf149f2ca :: v_dual_lshlrev_b32 v1, 2, v0// 000000001B74: CA2200FF 02000082 F149F2CA
	v_mov_b32_e32 v3, 0                                        // 000000001B80: 7E060280
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B84: BF870002
	v_add_nc_u32_e32 v1, 0x1400, v1                            // 000000001B88: 4A0202FF 00001400
	ds_store_2addr_b32 v1, v2, v3 offset1:16                   // 000000001B90: D8381000 00030201
	s_wait_alu depctr_sa_sdst(0)                               // 000000001B98: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000001B9C: 8C7E037E
	s_clause 0x1                                               // 000000001BA0: BF850001
	s_load_b128 s[20:23], s[0:1], 0xa0                         // 000000001BA4: F4004500 F80000A0
	s_load_b64 s[10:11], s[0:1], 0xb8                          // 000000001BAC: F4002280 F80000B8
	s_mov_b32 s6, ttmp9                                        // 000000001BB4: BE860075
	s_ashr_i32 s7, ttmp9, 31                                   // 000000001BB8: 86079F75
	s_ashr_i32 s5, s4, 31                                      // 000000001BBC: 86059F04
	s_lshl_b64 s[26:27], s[6:7], 4                             // 000000001BC0: 849A8406
	s_wait_dscnt 0x0                                           // 000000001BC4: BFC60000
	s_barrier_signal -1                                        // 000000001BC8: BE804EC1
	s_mov_b32 s19, 0                                           // 000000001BCC: BE930080
	s_mov_b64 s[30:31], 0                                      // 000000001BD0: BE9E0180
	s_wait_kmcnt 0x0                                           // 000000001BD4: BFC70000
	v_cmp_gt_i64_e64 s3, s[22:23], s[20:21]                    // 000000001BD8: D4540003 02002816
	s_add_nc_u64 s[6:7], s[22:23], 15                          // 000000001BE0: A9868F16
	s_sub_nc_u64 s[8:9], s[22:23], s[20:21]                    // 000000001BE4: AA081416
	s_lshr_b64 s[6:7], s[6:7], 4                               // 000000001BE8: 85868406
	s_mul_u64 s[24:25], s[20:21], s[4:5]                       // 000000001BEC: AA980414
	s_add_nc_u64 s[14:15], s[6:7], -1                          // 000000001BF0: A98EC106
	s_and_b32 s3, s3, exec_lo                                  // 000000001BF4: 8B037E03
	s_cselect_b32 s9, s9, 0                                    // 000000001BF8: 98098009
	s_cselect_b32 s8, s8, 0                                    // 000000001BFC: 98088008
	s_barrier_wait 0xffff                                      // 000000001C00: BF94FFFF
	s_add_nc_u64 s[8:9], s[8:9], s[26:27]                      // 000000001C04: A9881A08
	global_inv scope:SCOPE_SE                                  // 000000001C08: EE0AC07C 00040000 00000000
	s_add_nc_u64 s[12:13], s[8:9], 15                          // 000000001C14: A98C8F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001C18: BF870499
	s_lshr_b64 s[12:13], s[12:13], 4                           // 000000001C1C: 858C840C
	v_cmp_lt_i64_e64 s3, s[14:15], s[12:13]                    // 000000001C20: D4510003 0200180E
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C28: BF88FF9E
	s_and_b32 s3, s3, exec_lo                                  // 000000001C2C: 8B037E03
	s_cselect_b32 s13, s15, s13                                // 000000001C30: 980D0D0F
	s_cselect_b32 s12, s14, s12                                // 000000001C34: 980C0C0E
	s_cmp_lg_u64 s[10:11], 0                                   // 000000001C38: BF11800A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001C3C: BF88FF9E
	s_add_nc_u64 s[10:11], s[12:13], 1                         // 000000001C40: A98A810C
	s_cselect_b32 s33, -1, 0                                   // 000000001C44: 982180C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001C48: BF8704B9
	s_and_b32 s3, s33, exec_lo                                 // 000000001C4C: 8B037E21
	s_cselect_b32 s29, s11, s7                                 // 000000001C50: 981D070B
	s_cselect_b32 s28, s10, s6                                 // 000000001C54: 981C060A
	s_cmp_eq_u64 s[28:29], 0                                   // 000000001C58: BF10801C
	s_cbranch_scc1 3247                                        // 000000001C5C: BFA20CAF <attention_forward+0x341c>
	v_dual_mov_b32 v2, s27 :: v_dual_and_b32 v15, 15, v0       // 000000001C60: CA24001B 020E008F
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001C68: F4002280 F8000008
	s_mul_u64 s[34:35], s[22:23], s[4:5]                       // 000000001C70: AAA20416
	v_lshrrev_b32_e32 v6, 4, v0                                // 000000001C74: 320C0084
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001C78: BF870002
	v_or_b32_e32 v1, s26, v15                                  // 000000001C7C: 38021E1A
	s_lshl_b64 s[6:7], s[34:35], 6                             // 000000001C80: 84868622
	s_clause 0x2                                               // 000000001C84: BF850002
	s_load_b64 s[36:37], s[0:1], 0x30                          // 000000001C88: F4002900 F8000030
	s_load_b64 s[38:39], s[0:1], 0x58                          // 000000001C90: F4002980 F8000058
	s_load_b32 s42, s[0:1], 0xb0                               // 000000001C98: F4000A80 F80000B0
	v_lshlrev_b32_e32 v11, 2, v0                               // 000000001CA0: 30160082
	v_or_b32_e32 v124, 16, v15                                 // 000000001CA4: 38F81E90
	v_add_co_u32 v3, vcc_lo, s24, v1                           // 000000001CA8: D7006A03 02020218
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 000000001CB0: BF870241
	v_add_co_ci_u32_e64 v4, null, s25, v2, vcc_lo              // 000000001CB4: D5207C04 01AA0419
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[1:2]                  // 000000001CBC: 7CA80214
	v_or_b32_e32 v126, 32, v15                                 // 000000001CC0: 38FC1EA0
	v_or_b32_e32 v128, 48, v15                                 // 000000001CC4: 39001EB0
	v_lshlrev_b64_e32 v[3:4], 6, v[3:4]                        // 000000001CC8: 3E060686
	v_mov_b32_e32 v16, 0                                       // 000000001CCC: 7E200280
	v_add_co_u32 v97, s3, s6, v0                               // 000000001CD0: D7000361 02020006
	s_wait_alu depctr_va_sdst(0)                               // 000000001CD8: BF88F19F
	v_add_co_ci_u32_e64 v98, null, s7, 0, s3                   // 000000001CDC: D5207C62 000D0007
	s_wait_alu depctr_va_vcc(0)                                // 000000001CE4: BF88FF9D
	v_dual_cndmask_b32 v2, 0, v4 :: v_dual_cndmask_b32 v1, 0, v3// 000000001CE8: CA520880 02000680
	v_cmp_ne_u32_e64 s3, 0, v6                                 // 000000001CF0: D44D0003 02020C80
	v_lshlrev_b32_e32 v107, 6, v0                              // 000000001CF8: 30D60086
	v_or_b32_e32 v108, 0x1400, v11                             // 000000001CFC: 38D816FF 00001400
	v_add_nc_u32_e32 v109, 0x1440, v11                         // 000000001D04: 4ADA16FF 00001440
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]                        // 000000001D0C: 3E080281
	v_or_b32_e32 v1, 16, v3                                    // 000000001D10: 38020690
	v_add_nc_u32_e32 v110, 0x1480, v11                         // 000000001D14: 4ADC16FF 00001480
	v_lshl_add_u32 v117, v6, 5, 0x1480                         // 000000001D1C: D6460075 03FD0B06 00001480
	v_or_b32_e32 v129, s6, v124                                // 000000001D28: 3902F806
	v_or_b32_e32 v131, s6, v126                                // 000000001D2C: 3906FC06
	s_wait_kmcnt 0x0                                           // 000000001D30: BFC70000
	v_add_co_u32 v17, s4, s10, v4                              // 000000001D34: D7000411 0202080A
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001D3C: 02020280
	v_add_co_ci_u32_e64 v18, null, s11, v5, s4                 // 000000001D40: D5207C12 00120A0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001D48: BF870193
	v_add_co_u32 v19, s4, v17, 28                              // 000000001D4C: D7000413 02013911
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]                        // 000000001D54: 3E080281
	s_wait_alu depctr_va_sdst(0)                               // 000000001D58: BF88F19F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001D5C: BF870003
	v_add_co_ci_u32_e64 v20, null, 0, v18, s4                  // 000000001D60: D5207C14 00122480
	v_add_co_u32 v21, s4, v17, 30                              // 000000001D68: D7000415 02013D11
	v_or_b32_e32 v1, 32, v3                                    // 000000001D70: 380206A0
	s_wait_alu depctr_va_sdst(0)                               // 000000001D74: BF88F19F
	v_add_co_ci_u32_e64 v22, null, 0, v18, s4                  // 000000001D78: D5207C16 00122480
	v_add_co_u32 v23, s4, v17, 12                              // 000000001D80: D7000417 02011911
	s_wait_alu depctr_va_sdst(0)                               // 000000001D88: BF88F19F
	v_add_co_ci_u32_e64 v24, null, 0, v18, s4                  // 000000001D8C: D5207C18 00122480
	v_add_co_u32 v25, s4, s10, v4                              // 000000001D94: D7000419 0202080A
	s_wait_alu depctr_va_sdst(0)                               // 000000001D9C: BF88F19F
	v_add_co_ci_u32_e64 v26, null, s11, v5, s4                 // 000000001DA0: D5207C1A 00120A0B
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001DA8: 02020280
	v_add_co_u32 v27, s4, v17, 14                              // 000000001DAC: D700041B 02011D11
	s_wait_alu depctr_va_sdst(0)                               // 000000001DB4: BF88F19F
	v_add_co_ci_u32_e64 v28, null, 0, v18, s4                  // 000000001DB8: D5207C1C 00122480
	v_add_co_u32 v29, s4, v25, 28                              // 000000001DC0: D700041D 02013919
	s_wait_alu depctr_va_sdst(0)                               // 000000001DC8: BF88F19F
	v_add_co_ci_u32_e64 v30, null, 0, v26, s4                  // 000000001DCC: D5207C1E 00123480
	v_add_co_u32 v31, s4, v25, 30                              // 000000001DD4: D700041F 02013D19
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]                        // 000000001DDC: 3E080281
	v_or_b32_e32 v1, 48, v3                                    // 000000001DE0: 380206B0
	s_wait_alu depctr_va_sdst(0)                               // 000000001DE4: BF88F19F
	v_add_co_ci_u32_e64 v32, null, 0, v26, s4                  // 000000001DE8: D5207C20 00123480
	v_add_co_u32 v33, s4, v25, 12                              // 000000001DF0: D7000421 02011919
	s_wait_alu depctr_va_sdst(0)                               // 000000001DF8: BF88F19F
	v_add_co_ci_u32_e64 v34, null, 0, v26, s4                  // 000000001DFC: D5207C22 00123480
	v_add_co_u32 v35, s4, v25, 14                              // 000000001E04: D7000423 02011D19
	s_wait_alu depctr_va_sdst(0)                               // 000000001E0C: BF88F19F
	v_add_co_ci_u32_e64 v36, null, 0, v26, s4                  // 000000001E10: D5207C24 00123480
	v_add_co_u32 v37, s4, s10, v4                              // 000000001E18: D7000425 0202080A
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo                        // 000000001E20: 02020280
	s_wait_alu depctr_va_sdst(0)                               // 000000001E24: BF88F19F
	v_add_co_ci_u32_e64 v38, null, s11, v5, s4                 // 000000001E28: D5207C26 00120A0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001E30: BF870193
	v_add_co_u32 v39, s4, v37, 28                              // 000000001E34: D7000427 02013925
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 000000001E3C: 3E020281
	s_wait_alu depctr_va_sdst(0)                               // 000000001E40: BF88F19F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001E44: BF870003
	v_add_co_ci_u32_e64 v40, null, 0, v38, s4                  // 000000001E48: D5207C28 00124C80
	v_add_co_u32 v41, s4, v37, 30                              // 000000001E50: D7000429 02013D25
	s_wait_alu depctr_va_sdst(0)                               // 000000001E58: BF88F19F
	v_add_co_ci_u32_e64 v42, null, 0, v38, s4                  // 000000001E5C: D5207C2A 00124C80
	v_add_co_u32 v43, s4, v37, 12                              // 000000001E64: D700042B 02011925
	s_wait_alu depctr_va_sdst(0)                               // 000000001E6C: BF88F19F
	v_add_co_ci_u32_e64 v44, null, 0, v38, s4                  // 000000001E70: D5207C2C 00124C80
	v_add_co_u32 v45, s4, s10, v1                              // 000000001E78: D700042D 0202020A
	s_wait_alu depctr_va_sdst(0)                               // 000000001E80: BF88F19F
	v_add_co_ci_u32_e64 v46, null, s11, v2, s4                 // 000000001E84: D5207C2E 0012040B
	v_dual_mov_b32 v130, s7 :: v_dual_lshlrev_b32 v1, 3, v6    // 000000001E8C: CA220007 82000C83
	v_lshlrev_b32_e32 v2, 2, v15                               // 000000001E94: 30041E82
	v_add_co_u32 v47, s4, v37, 14                              // 000000001E98: D700042F 02011D25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001EA0: BF870223
	v_or_b32_e32 v4, 1, v1                                     // 000000001EA4: 38080281
	v_or_b32_e32 v5, 2, v1                                     // 000000001EA8: 380A0282
	v_lshl_or_b32 v3, v6, 9, v2                                // 000000001EAC: D6560003 04091306
	v_or_b32_e32 v7, 3, v1                                     // 000000001EB4: 380E0283
	v_or_b32_e32 v8, 4, v1                                     // 000000001EB8: 38100284
	s_wait_alu depctr_va_sdst(0)                               // 000000001EBC: BF88F19F
	v_add_co_ci_u32_e64 v48, null, 0, v38, s4                  // 000000001EC0: D5207C30 00124C80
	v_dual_mov_b32 v132, s7 :: v_dual_add_nc_u32 v99, 0x1000, v3// 000000001EC8: CA200007 846206FF 00001000
	v_lshl_or_b32 v3, v4, 6, v2                                // 000000001ED4: D6560003 04090D04
	v_add_co_u32 v49, s4, v45, 28                              // 000000001EDC: D7000431 0201392D
	s_wait_alu depctr_va_sdst(0)                               // 000000001EE4: BF88F19F
	v_add_co_ci_u32_e64 v50, null, 0, v46, s4                  // 000000001EE8: D5207C32 00125C80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001EF0: BF870003
	v_add_nc_u32_e32 v100, 0x1000, v3                          // 000000001EF4: 4AC806FF 00001000
	v_lshl_or_b32 v3, v5, 6, v2                                // 000000001EFC: D6560003 04090D05
	v_add_co_u32 v51, s4, v45, 30                              // 000000001F04: D7000433 02013D2D
	v_or_b32_e32 v9, 5, v1                                     // 000000001F0C: 38120285
	s_wait_alu depctr_va_sdst(0)                               // 000000001F10: BF88F19F
	v_add_co_ci_u32_e64 v52, null, 0, v46, s4                  // 000000001F14: D5207C34 00125C80
	v_add_nc_u32_e32 v101, 0x1000, v3                          // 000000001F1C: 4ACA06FF 00001000
	v_lshl_or_b32 v3, v7, 6, v2                                // 000000001F24: D6560003 04090D07
	v_add_co_u32 v53, s4, v45, 12                              // 000000001F2C: D7000435 0201192D
	s_wait_alu depctr_va_sdst(0)                               // 000000001F34: BF88F19F
	v_add_co_ci_u32_e64 v54, null, 0, v46, s4                  // 000000001F38: D5207C36 00125C80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001F40: BF870003
	v_dual_mov_b32 v127, s7 :: v_dual_add_nc_u32 v102, 0x1000, v3// 000000001F44: CA200007 7F6606FF 00001000
	v_lshl_or_b32 v3, v8, 6, v2                                // 000000001F50: D6560003 04090D08
	v_add_co_u32 v55, s4, v45, 14                              // 000000001F58: D7000437 02011D2D
	v_or_b32_e32 v10, 6, v1                                    // 000000001F60: 38140286
	s_wait_alu depctr_va_sdst(0)                               // 000000001F64: BF88F19F
	v_add_co_ci_u32_e64 v56, null, 0, v46, s4                  // 000000001F68: D5207C38 00125C80
	v_add_nc_u32_e32 v103, 0x1000, v3                          // 000000001F70: 4ACE06FF 00001000
	v_lshl_or_b32 v3, v9, 6, v2                                // 000000001F78: D6560003 04090D09
	v_add_co_u32 v57, s4, v1, s8                               // 000000001F80: D7000439 02001101
	s_wait_alu depctr_va_sdst(0)                               // 000000001F88: BF88F19F
	v_add_co_ci_u32_e64 v58, null, 0, s9, s4                   // 000000001F8C: D5207C3A 00101280
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001F94: BF870003
	v_add_nc_u32_e32 v104, 0x1000, v3                          // 000000001F98: 4AD006FF 00001000
	v_lshl_or_b32 v3, v10, 6, v2                               // 000000001FA0: D6560003 04090D0A
	v_or_b32_e32 v1, 7, v1                                     // 000000001FA8: 38020287
	v_add_co_u32 v59, s4, v57, 1                               // 000000001FAC: D700043B 02010339
	s_wait_alu depctr_va_sdst(0)                               // 000000001FB4: BF88F19F
	v_add_co_ci_u32_e64 v60, null, 0, v58, s4                  // 000000001FB8: D5207C3C 00127480
	v_add_co_u32 v61, s4, v57, 2                               // 000000001FC0: D700043D 02010539
	s_wait_alu depctr_va_sdst(0)                               // 000000001FC8: BF88F19F
	v_add_co_ci_u32_e64 v62, null, 0, v58, s4                  // 000000001FCC: D5207C3E 00127480
	v_add_co_u32 v63, s4, v57, 3                               // 000000001FD4: D700043F 02010739
	v_add_nc_u32_e32 v105, 0x1000, v3                          // 000000001FDC: 4AD206FF 00001000
	v_lshl_or_b32 v3, v1, 6, v2                                // 000000001FE4: D6560003 04090D01
	s_wait_alu depctr_va_sdst(0)                               // 000000001FEC: BF88F19F
	v_add_co_ci_u32_e64 v64, null, 0, v58, s4                  // 000000001FF0: D5207C40 00127480
	v_add_co_u32 v65, s4, v57, 4                               // 000000001FF8: D7000441 02010939
	s_wait_alu depctr_va_sdst(0)                               // 000000002000: BF88F19F
	v_add_co_ci_u32_e64 v66, null, 0, v58, s4                  // 000000002004: D5207C42 00127480
	v_add_co_u32 v67, s4, v57, 5                               // 00000000200C: D7000443 02010B39
	v_add_nc_u32_e32 v106, 0x1000, v3                          // 000000002014: 4AD406FF 00001000
	v_lshlrev_b32_e32 v3, 6, v15                               // 00000000201C: 30061E86
	s_wait_alu depctr_va_sdst(0)                               // 000000002020: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v58, s4                  // 000000002024: D5207C44 00127480
	v_add_co_u32 v69, s4, v57, 6                               // 00000000202C: D7000445 02010D39
	s_wait_alu depctr_va_sdst(0)                               // 000000002034: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v58, s4                  // 000000002038: D5207C46 00127480
	v_add_co_u32 v71, s4, v57, 7                               // 000000002040: D7000447 02010F39
	s_wait_alu depctr_va_sdst(0)                               // 000000002048: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v58, s4                  // 00000000204C: D5207C48 00127480
	v_or_b32_e32 v111, 0x1000, v3                              // 000000002054: 38DE06FF 00001000
	v_or_b32_e32 v112, 0x1038, v3                              // 00000000205C: 38E006FF 00001038
	v_or_b32_e32 v113, 0x103c, v3                              // 000000002064: 38E206FF 0000103C
	v_or_b32_e32 v114, 0x1018, v3                              // 00000000206C: 38E406FF 00001018
	v_or_b32_e32 v115, 0x101c, v3                              // 000000002074: 38E606FF 0000101C
	v_lshl_or_b32 v116, v6, 11, v2                             // 00000000207C: D6560074 04091706
	v_lshl_or_b32 v118, v4, 8, v2                              // 000000002084: D6560076 04091104
	v_lshl_or_b32 v119, v5, 8, v2                              // 00000000208C: D6560077 04091105
	v_lshl_or_b32 v120, v7, 8, v2                              // 000000002094: D6560078 04091107
	v_lshl_or_b32 v121, v8, 8, v2                              // 00000000209C: D6560079 04091108
	v_lshl_or_b32 v122, v9, 8, v2                              // 0000000020A4: D656007A 04091109
	v_lshl_or_b32 v123, v10, 8, v2                             // 0000000020AC: D656007B 0409110A
	v_lshl_or_b32 v125, v1, 8, v2                              // 0000000020B4: D656007D 04091101
	v_or_b32_e32 v133, s6, v128                                // 0000000020BC: 390B0006
	s_branch 263                                               // 0000000020C0: BFA00107 <attention_forward+0x9e0>
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000020C4: 8C7E057E
	v_or_b32_e32 v87, s40, v87                                 // 0000000020C8: 38AEAE28
	v_or_b32_e32 v88, s41, v88                                 // 0000000020CC: 38B0B029
	v_or_b32_e32 v85, s40, v85                                 // 0000000020D0: 38AAAA28
	v_or_b32_e32 v86, s41, v86                                 // 0000000020D4: 38ACAC29
	v_or_b32_e32 v81, s40, v81                                 // 0000000020D8: 38A2A228
	v_add_co_u32 v90, s4, v87, s34                             // 0000000020DC: D700045A 02004557
	s_wait_alu depctr_va_sdst(0)                               // 0000000020E4: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v88, s4                // 0000000020E8: D5207C5B 0012B023
	v_add_co_u32 v92, s4, v85, s34                             // 0000000020F0: D700045C 02004555
	s_wait_alu depctr_va_sdst(0)                               // 0000000020F8: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v86, s4                // 0000000020FC: D5207C5D 0012AC23
	v_or_b32_e32 v82, s41, v82                                 // 000000002104: 38A4A429
	v_cmp_gt_i64_e64 s4, s[22:23], v[87:88]                    // 000000002108: D4540004 0202AE16
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 000000002110: 3EB4B486
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002114: BF870004
	v_lshlrev_b64_e32 v[87:88], 6, v[92:93]                    // 000000002118: 3EAEB886
	v_add_co_u32 v92, s5, v81, s34                             // 00000000211C: D700055C 02004551
	v_lshlrev_b64_e32 v[83:84], 1, v[83:84]                    // 000000002124: 3EA6A681
	s_wait_alu depctr_va_sdst(0)                               // 000000002128: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v82, s5                // 00000000212C: D5207C5D 0016A423
	v_or_b32_e32 v90, v90, v128                                // 000000002134: 38B5015A
	v_cndmask_b32_e64 v91, 0, v91, s4                          // 000000002138: D501005B 0012B680
	v_cmp_gt_i64_e64 s6, s[22:23], v[81:82]                    // 000000002140: D4540006 0202A216
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002148: BF870004
	v_lshlrev_b64_e32 v[92:93], 6, v[92:93]                    // 00000000214C: 3EB8B886
	v_add_co_u32 v83, s5, s38, v83                             // 000000002150: D7000553 0202A626
	s_wait_alu depctr_va_sdst(0)                               // 000000002158: BF88F19F
	v_add_co_ci_u32_e64 v84, null, s39, v84, s5                // 00000000215C: D5207C54 0016A827
	v_cmp_gt_i64_e64 s5, s[22:23], v[85:86]                    // 000000002164: D4540005 0202AA16
	v_or_b32_e32 v85, v87, v128                                // 00000000216C: 38AB0157
	v_cndmask_b32_e64 v90, 0, v90, s4                          // 000000002170: D501005A 0012B480
	v_or_b32_e32 v87, v92, v128                                // 000000002178: 38AF015C
	v_or_b32_e32 v79, s40, v79                                 // 00000000217C: 389E9E28
	v_or_b32_e32 v80, s41, v80                                 // 000000002180: 38A0A029
	s_wait_alu depctr_va_sdst(0)                               // 000000002184: BF88F19F
	v_cndmask_b32_e64 v86, 0, v88, s5                          // 000000002188: D5010056 0016B080
	v_cndmask_b32_e64 v85, 0, v85, s5                          // 000000002190: D5010055 0016AA80
	v_lshlrev_b64_e32 v[81:82], 1, v[90:91]                    // 000000002198: 3EA2B481
	v_cndmask_b32_e64 v88, 0, v93, s6                          // 00000000219C: D5010058 001ABA80
	v_cndmask_b32_e64 v87, 0, v87, s6                          // 0000000021A4: D5010057 001AAE80
	v_or_b32_e32 v77, s40, v77                                 // 0000000021AC: 389A9A28
	v_lshlrev_b64_e32 v[85:86], 1, v[85:86]                    // 0000000021B0: 3EAAAA81
	v_or_b32_e32 v78, s41, v78                                 // 0000000021B4: 389C9C29
	v_add_co_u32 v81, s7, s38, v81                             // 0000000021B8: D7000751 0202A226
	v_lshlrev_b64_e32 v[87:88], 1, v[87:88]                    // 0000000021C0: 3EAEAE81
	s_wait_alu depctr_va_sdst(0)                               // 0000000021C4: BF88F19F
	v_add_co_ci_u32_e64 v82, null, s39, v82, s7                // 0000000021C8: D5207C52 001EA427
	v_add_co_u32 v90, s7, v79, s34                             // 0000000021D0: D700075A 0200454F
	s_wait_alu depctr_va_sdst(0)                               // 0000000021D8: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v80, s7                // 0000000021DC: D5207C5B 001EA023
	v_add_co_u32 v85, s7, s38, v85                             // 0000000021E4: D7000755 0202AA26
	v_or_b32_e32 v75, s40, v75                                 // 0000000021EC: 38969628
	s_wait_alu depctr_va_sdst(0)                               // 0000000021F0: BF88F19F
	v_add_co_ci_u32_e64 v86, null, s39, v86, s7                // 0000000021F4: D5207C56 001EAC27
	v_add_co_u32 v87, s7, s38, v87                             // 0000000021FC: D7000757 0202AE26
	v_or_b32_e32 v76, s41, v76                                 // 000000002204: 38989829
	s_wait_alu depctr_va_sdst(0)                               // 000000002208: BF88F19F
	v_add_co_ci_u32_e64 v88, null, s39, v88, s7                // 00000000220C: D5207C58 001EB027
	v_cmp_gt_i64_e64 s7, s[22:23], v[79:80]                    // 000000002214: D4540007 02029E16
	v_add_co_u32 v79, s8, v77, s34                             // 00000000221C: D700084F 0200454D
	s_wait_alu depctr_va_sdst(0)                               // 000000002224: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s35, v78, s8                // 000000002228: D5207C50 00229C23
	v_add_co_u32 v92, s8, v75, s34                             // 000000002230: D700085C 0200454B
	s_wait_alu depctr_va_sdst(0)                               // 000000002238: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v76, s8                // 00000000223C: D5207C5D 00229823
	v_or_b32_e32 v73, s40, v73                                 // 000000002244: 38929228
	v_or_b32_e32 v74, s41, v74                                 // 000000002248: 38949429
	v_cmp_gt_i64_e64 s8, s[22:23], v[77:78]                    // 00000000224C: D4540008 02029A16
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002254: BF870004
	v_lshlrev_b64_e32 v[77:78], 6, v[92:93]                    // 000000002258: 3E9AB886
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 00000000225C: 3EB4B486
	v_add_co_u32 v92, s9, v73, s34                             // 000000002260: D700095C 02004549
	s_wait_alu depctr_va_sdst(0)                               // 000000002268: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v74, s9                // 00000000226C: D5207C5D 00269423
	v_lshlrev_b64_e32 v[79:80], 6, v[79:80]                    // 000000002274: 3E9E9E86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000002278: BF870224
	v_or_b32_e32 v90, v90, v128                                // 00000000227C: 38B5015A
	v_cmp_gt_i64_e64 s9, s[22:23], v[75:76]                    // 000000002280: D4540009 02029616
	v_lshlrev_b64_e32 v[75:76], 6, v[92:93]                    // 000000002288: 3E96B886
	v_or_b32_e32 v77, v77, v128                                // 00000000228C: 389B014D
	v_cndmask_b32_e64 v91, 0, v91, s7                          // 000000002290: D501005B 001EB680
	v_or_b32_e32 v79, v79, v128                                // 000000002298: 389F014F
	v_cndmask_b32_e64 v90, 0, v90, s7                          // 00000000229C: D501005A 001EB480
	v_cmp_gt_i64_e64 s10, s[22:23], v[73:74]                   // 0000000022A4: D454000A 02029216
	v_or_b32_e32 v75, v75, v128                                // 0000000022AC: 3897014B
	v_cndmask_b32_e64 v80, 0, v80, s8                          // 0000000022B0: D5010050 0022A080
	v_cndmask_b32_e64 v79, 0, v79, s8                          // 0000000022B8: D501004F 00229E80
	s_wait_alu depctr_va_sdst(0)                               // 0000000022C0: BF88F19F
	v_cndmask_b32_e64 v78, 0, v78, s9                          // 0000000022C4: D501004E 00269C80
	v_cndmask_b32_e64 v77, 0, v77, s9                          // 0000000022CC: D501004D 00269A80
	v_lshlrev_b64_e32 v[90:91], 1, v[90:91]                    // 0000000022D4: 3EB4B481
	v_cndmask_b32_e64 v76, 0, v76, s10                         // 0000000022D8: D501004C 002A9880
	v_cndmask_b32_e64 v75, 0, v75, s10                         // 0000000022E0: D501004B 002A9680
	v_lshlrev_b64_e32 v[79:80], 1, v[79:80]                    // 0000000022E8: 3E9E9E81
	v_lshlrev_b64_e32 v[73:74], 1, v[77:78]                    // 0000000022EC: 3E929A81
	s_wait_dscnt 0x1                                           // 0000000022F0: BFC60001
	v_cvt_f16_f32_e32 v12.l, v12                               // 0000000022F4: 7E18150C
	v_add_co_u32 v90, s11, s38, v90                            // 0000000022F8: D7000B5A 0202B426
	v_lshlrev_b64_e32 v[75:76], 1, v[75:76]                    // 000000002300: 3E969681
	s_wait_alu depctr_va_sdst(0)                               // 000000002304: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s39, v91, s11               // 000000002308: D5207C5B 002EB627
	v_add_co_u32 v77, s11, s38, v79                            // 000000002310: D7000B4D 02029E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002318: BF88F19F
	v_add_co_ci_u32_e64 v78, null, s39, v80, s11               // 00000000231C: D5207C4E 002EA027
	v_add_co_u32 v79, s11, s38, v73                            // 000000002324: D7000B4F 02029226
	s_wait_alu depctr_va_sdst(0)                               // 00000000232C: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s39, v74, s11               // 000000002330: D5207C50 002E9427
	v_add_co_u32 v92, s11, s38, v75                            // 000000002338: D7000B5C 02029626
	s_wait_alu depctr_va_sdst(0)                               // 000000002340: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s39, v76, s11               // 000000002344: D5207C5D 002E9827
	s_clause 0x7                                               // 00000000234C: BF850007
	global_load_d16_b16 v73, v[83:84], off                     // 000000002350: EE08007C 00000049 00000053
	global_load_d16_hi_b16 v73, v[81:82], off                  // 00000000235C: EE08C07C 00000049 00000051
	global_load_d16_b16 v74, v[85:86], off                     // 000000002368: EE08007C 0000004A 00000055
	global_load_d16_hi_b16 v74, v[87:88], off                  // 000000002374: EE08C07C 0000004A 00000057
	global_load_d16_b16 v75, v[90:91], off                     // 000000002380: EE08007C 0000004B 0000005A
	global_load_d16_hi_b16 v75, v[77:78], off                  // 00000000238C: EE08C07C 0000004B 0000004D
	global_load_d16_b16 v76, v[79:80], off                     // 000000002398: EE08007C 0000004C 0000004F
	global_load_d16_hi_b16 v76, v[92:93], off                  // 0000000023A4: EE08C07C 0000004C 0000005C
	v_add_nc_u32_e32 v90, 0xc0, v116                           // 0000000023B0: 4AB4E8FF 000000C0
	ds_load_2addr_b32 v[77:78], v116 offset0:48 offset1:112    // 0000000023B8: D8DC7030 4D000074
	ds_load_2addr_b32 v[79:80], v116 offset0:176 offset1:240   // 0000000023C0: D8DCF0B0 4F000074
	ds_load_2addr_stride64_b32 v[85:86], v90 offset0:4 offset1:5// 0000000023C8: D8E00504 5500005A
	ds_load_2addr_stride64_b32 v[87:88], v90 offset0:6 offset1:7// 0000000023D0: D8E00706 5700005A
	s_wait_dscnt 0x4                                           // 0000000023D8: BFC60004
	v_cvt_f16_f32_e32 v12.h, v89                               // 0000000023DC: 7F181559
	s_add_nc_u64 s[30:31], s[30:31], 1                         // 0000000023E0: A99E811E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000023E4: BF870009
	s_cmp_lg_u64 s[30:31], s[28:29]                            // 0000000023E8: BF111C1E
	s_wait_dscnt 0x3                                           // 0000000023EC: BFC60003
	v_dual_mul_f32 v1, v1, v77 :: v_dual_mul_f32 v2, v2, v78   // 0000000023F0: C8C69B01 01029D02
	s_wait_dscnt 0x2                                           // 0000000023F8: BFC60002
	v_dual_mul_f32 v3, v3, v79 :: v_dual_mul_f32 v4, v4, v80   // 0000000023FC: C8C69F03 0304A104
	s_wait_dscnt 0x1                                           // 000000002404: BFC60001
	v_dual_mul_f32 v5, v5, v85 :: v_dual_mul_f32 v6, v6, v86   // 000000002408: C8C6AB05 0506AD06
	s_wait_dscnt 0x0                                           // 000000002410: BFC60000
	v_dual_mul_f32 v7, v7, v87 :: v_dual_mul_f32 v8, v8, v88   // 000000002414: C8C6AF07 0708B108
	s_wait_loadcnt 0x6                                         // 00000000241C: BFC00006
	s_wait_alu depctr_sa_sdst(0)                               // 000000002420: BF88FF9E
	v_cndmask_b16 v81.l, 0, v73.l, s12                         // 000000002424: D65D0051 00329280
	v_cndmask_b16 v81.h, 0, v73.h, s4                          // 00000000242C: D65D5051 00129280
	s_wait_loadcnt 0x4                                         // 000000002434: BFC00004
	v_cndmask_b16 v82.l, 0, v74.l, s5                          // 000000002438: D65D0052 00169480
	v_cndmask_b16 v82.h, 0, v74.h, s6                          // 000000002440: D65D5052 001A9480
	s_wait_loadcnt 0x2                                         // 000000002448: BFC00002
	v_cndmask_b16 v83.l, 0, v75.l, s7                          // 00000000244C: D65D0053 001E9680
	v_cndmask_b16 v83.h, 0, v75.h, s8                          // 000000002454: D65D5053 00229680
	s_wait_loadcnt 0x0                                         // 00000000245C: BFC00000
	v_cndmask_b16 v84.l, 0, v76.l, s9                          // 000000002460: D65D0054 00269880
	v_cndmask_b16 v84.h, 0, v76.h, s10                         // 000000002468: D65D5054 002A9880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002470: BF870091
	v_wmma_f32_16x16x16_f16 v[73:80], v[9:12], v[81:84], 0     // 000000002474: CC404049 1A02A309
	v_dual_add_f32 v1, v73, v1 :: v_dual_add_f32 v2, v74, v2   // 00000000247C: C9080349 0102054A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002484: BF870192
	v_dual_add_f32 v3, v75, v3 :: v_dual_add_f32 v4, v76, v4   // 000000002488: C908074B 0304094C
	v_dual_add_f32 v5, v77, v5 :: v_dual_add_f32 v6, v78, v6   // 000000002490: C9080B4D 05060D4E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002498: BF870004
	v_dual_add_f32 v7, v79, v7 :: v_dual_add_f32 v8, v80, v8   // 00000000249C: C9080F4F 07081150
	ds_store_2addr_b32 v116, v1, v2 offset0:48 offset1:112     // 0000000024A4: D8387030 00020174
	ds_store_2addr_b32 v116, v3, v4 offset0:176 offset1:240    // 0000000024AC: D838F0B0 00040374
	ds_store_2addr_stride64_b32 v90, v5, v6 offset0:4 offset1:5// 0000000024B4: D83C0504 0006055A
	ds_store_2addr_stride64_b32 v90, v7, v8 offset0:6 offset1:7// 0000000024BC: D83C0706 0008075A
	s_wait_dscnt 0x0                                           // 0000000024C4: BFC60000
	s_barrier_signal -1                                        // 0000000024C8: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000024CC: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000024D0: EE0AC07C 00040000 00000000
	s_cbranch_scc0 2703                                        // 0000000024DC: BFA10A8F <attention_forward+0x341c>
	s_and_saveexec_b32 s4, s3                                  // 0000000024E0: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000024E4: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 0000000024E8: 8D04047E
	s_cbranch_execz 16                                         // 0000000024EC: BFA50010 <attention_forward+0xa30>
	global_load_b96 v[1:3], v[17:18], off offset:16            // 0000000024F0: EE05807C 00000001 00001011
	s_wait_loadcnt 0x0                                         // 0000000024FC: BFC00000
	v_cndmask_b16 v9.l, 0, v1.l, vcc_lo                        // 000000002500: D65D0009 01AA0280
	v_cndmask_b16 v9.h, 0, v1.h, vcc_lo                        // 000000002508: D65D5009 01AA0280
	v_cndmask_b16 v10.l, 0, v2.l, vcc_lo                       // 000000002510: D65D000A 01AA0480
	v_cndmask_b16 v10.h, 0, v2.h, vcc_lo                       // 000000002518: D65D500A 01AA0480
	v_cndmask_b16 v11.l, 0, v3.l, vcc_lo                       // 000000002520: D65D000B 01AA0680
	v_cndmask_b16 v11.h, 0, v3.h, vcc_lo                       // 000000002528: D65D500B 01AA0680
	s_wait_alu depctr_sa_sdst(0)                               // 000000002530: BF88FF9E
	s_or_saveexec_b32 s4, s4                                   // 000000002534: BE842204
	v_dual_mov_b32 v1, v19 :: v_dual_mov_b32 v2, v20           // 000000002538: CA100113 01020114
	v_dual_mov_b32 v3, v21 :: v_dual_mov_b32 v4, v22           // 000000002540: CA100115 03040116
	s_wait_alu depctr_sa_sdst(0)                               // 000000002548: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s4                             // 00000000254C: 8D7E047E
	s_cbranch_execz 20                                         // 000000002550: BFA50014 <attention_forward+0xaa4>
	global_load_b96 v[3:5], v[17:18], off                      // 000000002554: EE05807C 00000003 00000011
	v_dual_mov_b32 v1, v23 :: v_dual_mov_b32 v2, v24           // 000000002560: CA100117 01020118
	s_wait_loadcnt 0x0                                         // 000000002568: BFC00000
	v_cndmask_b16 v9.l, 0, v3.l, vcc_lo                        // 00000000256C: D65D0009 01AA0680
	v_cndmask_b16 v9.h, 0, v3.h, vcc_lo                        // 000000002574: D65D5009 01AA0680
	v_cndmask_b16 v10.l, 0, v4.l, vcc_lo                       // 00000000257C: D65D000A 01AA0880
	v_cndmask_b16 v10.h, 0, v4.h, vcc_lo                       // 000000002584: D65D500A 01AA0880
	v_cndmask_b16 v11.l, 0, v5.l, vcc_lo                       // 00000000258C: D65D000B 01AA0A80
	v_cndmask_b16 v11.h, 0, v5.h, vcc_lo                       // 000000002594: D65D500B 01AA0A80
	v_dual_mov_b32 v3, v27 :: v_dual_mov_b32 v4, v28           // 00000000259C: CA10011B 0304011C
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000025A4: 8C7E047E
	global_load_d16_b16 v1, v[1:2], off                        // 0000000025A8: EE08007C 00000001 00000001
	global_load_d16_hi_b16 v1, v[3:4], off                     // 0000000025B4: EE08C07C 00000001 00000003
	s_lshl_b64 s[40:41], s[30:31], 4                           // 0000000025C0: 84A8841E
	v_dual_mov_b32 v4, 14 :: v_dual_mov_b32 v5, 0              // 0000000025C4: CA10008E 04040080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025CC: BF88FF9E
	v_or_b32_e32 v73, s40, v15                                 // 0000000025D0: 38921E28
	v_dual_mov_b32 v3, s41 :: v_dual_mov_b32 v74, s41          // 0000000025D4: CA100029 034A0029
	v_dual_mov_b32 v6, 12 :: v_dual_mov_b32 v7, 0              // 0000000025DC: CA10008C 06060080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000025E4: BF8701A3
	v_add_co_u32 v2, s4, v73, s34                              // 0000000025E8: D7000402 02004549
	s_wait_alu depctr_va_sdst(0)                               // 0000000025F0: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s35, v3, s4                  // 0000000025F4: D5207C03 00120623
	v_cmp_gt_i64_e64 s4, s[22:23], v[73:74]                    // 0000000025FC: D4540004 02029216
	v_dual_mov_b32 v80, 8 :: v_dual_mov_b32 v81, 0             // 000000002604: CA100088 50500080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)// 00000000260C: BF870253
	v_lshlrev_b64_e32 v[77:78], 6, v[2:3]                      // 000000002610: 3E9A0486
	v_dual_mov_b32 v82, 6 :: v_dual_mov_b32 v83, 0             // 000000002614: CA100086 52520080
	v_dual_mov_b32 v84, 4 :: v_dual_mov_b32 v85, 0             // 00000000261C: CA100084 54540080
	v_mov_b32_e32 v79, 0                                       // 000000002624: 7E9E0280
	s_wait_alu depctr_va_sdst(0)                               // 000000002628: BF88F19F
	v_cndmask_b32_e64 v76, 0, v78, s4                          // 00000000262C: D501004C 00129C80
	v_cndmask_b32_e64 v75, 0, v77, s4                          // 000000002634: D501004B 00129A80
	v_mov_b32_e32 v78, 10                                      // 00000000263C: 7E9C028A
	v_dual_mov_b32 v86, 2 :: v_dual_mov_b32 v87, 0             // 000000002640: CA100082 56560080
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002648: BF870093
	v_lshlrev_b64_e32 v[2:3], 1, v[75:76]                      // 00000000264C: 3E049681
	v_add_co_u32 v2, s5, s36, v2                               // 000000002650: D7000502 02020424
	s_wait_alu depctr_va_sdst(0)                               // 000000002658: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000265C: BF870092
	v_add_co_ci_u32_e64 v3, null, s37, v3, s5                  // 000000002660: D5207C03 00160625
	v_dual_mov_b32 v88, v2 :: v_dual_mov_b32 v89, v3           // 000000002668: CA100102 58580103
	s_and_saveexec_b32 s6, s3                                  // 000000002670: BE862003
	s_cbranch_execz 19                                         // 000000002674: BFA50013 <attention_forward+0xbc4>
	v_add_co_u32 v88, s5, v2, 16                               // 000000002678: D7000558 02012102
	v_dual_mov_b32 v4, 30 :: v_dual_mov_b32 v5, 0              // 000000002680: CA10009E 04040080
	v_dual_mov_b32 v6, 28 :: v_dual_mov_b32 v7, 0              // 000000002688: CA10009C 06060080
	v_dual_mov_b32 v78, 26 :: v_dual_mov_b32 v79, 0            // 000000002690: CA10009A 4E4E0080
	v_dual_mov_b32 v80, 24 :: v_dual_mov_b32 v81, 0            // 000000002698: CA100098 50500080
	v_dual_mov_b32 v82, 22 :: v_dual_mov_b32 v83, 0            // 0000000026A0: CA100096 52520080
	v_dual_mov_b32 v84, 20 :: v_dual_mov_b32 v85, 0            // 0000000026A8: CA100094 54540080
	v_dual_mov_b32 v86, 18 :: v_dual_mov_b32 v87, 0            // 0000000026B0: CA100092 56560080
	s_wait_alu depctr_va_sdst(0)                               // 0000000026B8: BF88F19F
	v_add_co_ci_u32_e64 v89, null, 0, v3, s5                   // 0000000026BC: D5207C59 00160680
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026C4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000026C8: 8C7E067E
	v_add_co_u32 v86, s5, v2, v86                              // 0000000026CC: D7000556 0202AD02
	s_wait_alu depctr_va_sdst(0)                               // 0000000026D4: BF88F19F
	v_add_co_ci_u32_e64 v87, null, v3, v87, s5                 // 0000000026D8: D5207C57 0016AF03
	v_add_co_u32 v84, s5, v2, v84                              // 0000000026E0: D7000554 0202A902
	s_wait_alu depctr_va_sdst(0)                               // 0000000026E8: BF88F19F
	v_add_co_ci_u32_e64 v85, null, v3, v85, s5                 // 0000000026EC: D5207C55 0016AB03
	v_add_co_u32 v82, s5, v2, v82                              // 0000000026F4: D7000552 0202A502
	s_wait_alu depctr_va_sdst(0)                               // 0000000026FC: BF88F19F
	v_add_co_ci_u32_e64 v83, null, v3, v83, s5                 // 000000002700: D5207C53 0016A703
	v_add_co_u32 v80, s5, v2, v80                              // 000000002708: D7000550 0202A102
	s_wait_alu depctr_va_sdst(0)                               // 000000002710: BF88F19F
	v_add_co_ci_u32_e64 v81, null, v3, v81, s5                 // 000000002714: D5207C51 0016A303
	v_add_co_u32 v78, s5, v2, v78                              // 00000000271C: D700054E 02029D02
	s_wait_alu depctr_va_sdst(0)                               // 000000002724: BF88F19F
	v_add_co_ci_u32_e64 v79, null, v3, v79, s5                 // 000000002728: D5207C4F 00169F03
	v_add_co_u32 v6, s5, v2, v6                                // 000000002730: D7000506 02020D02
	global_load_d16_b16 v8, v[88:89], off                      // 000000002738: EE08007C 00000008 00000058
	s_wait_alu depctr_va_sdst(0)                               // 000000002744: BF88F19F
	v_add_co_ci_u32_e64 v7, null, v3, v7, s5                   // 000000002748: D5207C07 00160F03
	v_add_co_u32 v88, s5, v2, v4                               // 000000002750: D7000558 02020902
	s_wait_alu depctr_va_sdst(0)                               // 000000002758: BF88F19F
	v_add_co_ci_u32_e64 v89, null, v3, v5, s5                  // 00000000275C: D5207C59 00160B03
	s_clause 0x6                                               // 000000002764: BF850006
	global_load_d16_b16 v2, v[86:87], off                      // 000000002768: EE08007C 00000002 00000056
	global_load_d16_hi_b16 v2, v[84:85], off                   // 000000002774: EE08C07C 00000002 00000054
	global_load_d16_b16 v3, v[82:83], off                      // 000000002780: EE08007C 00000003 00000052
	global_load_d16_hi_b16 v3, v[80:81], off                   // 00000000278C: EE08C07C 00000003 00000050
	global_load_d16_b16 v4, v[78:79], off                      // 000000002798: EE08007C 00000004 0000004E
	global_load_d16_hi_b16 v4, v[6:7], off                     // 0000000027A4: EE08C07C 00000004 00000006
	global_load_d16_b16 v5, v[88:89], off                      // 0000000027B0: EE08007C 00000005 00000058
	s_wait_loadcnt 0x8                                         // 0000000027BC: BFC00008
	v_cndmask_b16 v12.l, 0, v1.l, vcc_lo                       // 0000000027C0: D65D000C 01AA0280
	v_cndmask_b16 v12.h, 0, v1.h, vcc_lo                       // 0000000027C8: D65D500C 01AA0280
	s_wait_loadcnt 0x7                                         // 0000000027D0: BFC00007
	v_cndmask_b16 v78.l, 0, v8.l, s4                           // 0000000027D4: D65D004E 00121080
	s_wait_loadcnt 0x5                                         // 0000000027DC: BFC00005
	v_cndmask_b16 v78.h, 0, v2.l, s4                           // 0000000027E0: D65D404E 00120480
	v_cndmask_b16 v79.l, 0, v2.h, s4                           // 0000000027E8: D65D104F 00120480
	s_wait_loadcnt 0x3                                         // 0000000027F0: BFC00003
	v_cndmask_b16 v79.h, 0, v3.l, s4                           // 0000000027F4: D65D404F 00120680
	v_cndmask_b16 v80.l, 0, v3.h, s4                           // 0000000027FC: D65D1050 00120680
	s_wait_loadcnt 0x1                                         // 000000002804: BFC00001
	v_cndmask_b16 v80.h, 0, v4.l, s4                           // 000000002808: D65D4050 00120880
	v_cndmask_b16 v81.l, 0, v4.h, s4                           // 000000002810: D65D1051 00120880
	s_wait_loadcnt 0x0                                         // 000000002818: BFC00000
	v_cndmask_b16 v81.h, 0, v5.l, s4                           // 00000000281C: D65D4051 00120A80
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002824: BF870001
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:12], v[78:81], 0       // 000000002828: CC404001 1A029D09
	s_and_saveexec_b32 s5, s3                                  // 000000002830: BE852003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002834: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000002838: 8D05057E
	s_cbranch_execz 16                                         // 00000000283C: BFA50010 <attention_forward+0xd80>
	global_load_b96 v[9:11], v[25:26], off offset:16           // 000000002840: EE05807C 00000009 00001019
	s_wait_loadcnt 0x0                                         // 00000000284C: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 000000002850: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 000000002858: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 000000002860: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 000000002868: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 000000002870: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 000000002878: D65D500B 01AA1680
	s_wait_alu depctr_sa_sdst(0)                               // 000000002880: BF88FF9E
	s_or_saveexec_b32 s5, s5                                   // 000000002884: BE852205
	v_dual_mov_b32 v79, v30 :: v_dual_mov_b32 v78, v29         // 000000002888: CA10011E 4F4E011D
	v_dual_mov_b32 v81, v32 :: v_dual_mov_b32 v80, v31         // 000000002890: CA100120 5150011F
	s_wait_alu depctr_sa_sdst(0)                               // 000000002898: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 00000000289C: 8D7E057E
	s_cbranch_execz 20                                         // 0000000028A0: BFA50014 <attention_forward+0xdf4>
	global_load_b96 v[9:11], v[25:26], off                     // 0000000028A4: EE05807C 00000009 00000019
	v_dual_mov_b32 v79, v34 :: v_dual_mov_b32 v78, v33         // 0000000028B0: CA100122 4F4E0121
	v_dual_mov_b32 v81, v36 :: v_dual_mov_b32 v80, v35         // 0000000028B8: CA100124 51500123
	s_wait_loadcnt 0x0                                         // 0000000028C0: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 0000000028C4: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 0000000028CC: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 0000000028D4: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 0000000028DC: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 0000000028E4: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 0000000028EC: D65D500B 01AA1680
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000028F4: 8C7E057E
	global_load_d16_b16 v12, v[78:79], off                     // 0000000028F8: EE08007C 0000000C 0000004E
	global_load_d16_hi_b16 v12, v[80:81], off                  // 000000002904: EE08C07C 0000000C 00000050
	v_or_b32_e32 v75, 16, v77                                  // 000000002910: 38969A90
	v_dual_mov_b32 v80, 14 :: v_dual_mov_b32 v81, 0            // 000000002914: CA10008E 50500080
	v_dual_mov_b32 v82, 12 :: v_dual_mov_b32 v83, 0            // 00000000291C: CA10008C 52520080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000002924: BF8701B3
	v_cndmask_b32_e64 v75, 0, v75, s4                          // 000000002928: D501004B 00129680
	v_dual_mov_b32 v84, 10 :: v_dual_mov_b32 v85, 0            // 000000002930: CA10008A 54540080
	v_dual_mov_b32 v86, 8 :: v_dual_mov_b32 v87, 0             // 000000002938: CA100088 56560080
	v_lshlrev_b64_e32 v[78:79], 1, v[75:76]                    // 000000002940: 3E9C9681
	v_dual_mov_b32 v88, 6 :: v_dual_mov_b32 v89, 0             // 000000002944: CA100086 58580080
	v_dual_mov_b32 v90, 4 :: v_dual_mov_b32 v91, 0             // 00000000294C: CA100084 5A5A0080
	v_dual_mov_b32 v92, 2 :: v_dual_mov_b32 v93, 0             // 000000002954: CA100082 5C5C0080
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 00000000295C: BF8700B4
	v_add_co_u32 v78, s5, s36, v78                             // 000000002960: D700054E 02029C24
	s_wait_alu depctr_va_sdst(0)                               // 000000002968: BF88F19F
	v_add_co_ci_u32_e64 v79, null, s37, v79, s5                // 00000000296C: D5207C4F 00169E25
	v_dual_mov_b32 v94, v78 :: v_dual_mov_b32 v95, v79         // 000000002974: CA10014E 5E5E014F
	s_and_saveexec_b32 s6, s3                                  // 00000000297C: BE862003
	s_cbranch_execz 19                                         // 000000002980: BFA50013 <attention_forward+0xed0>
	v_add_co_u32 v94, s5, v78, 16                              // 000000002984: D700055E 0201214E
	v_dual_mov_b32 v80, 30 :: v_dual_mov_b32 v81, 0            // 00000000298C: CA10009E 50500080
	v_dual_mov_b32 v82, 28 :: v_dual_mov_b32 v83, 0            // 000000002994: CA10009C 52520080
	v_dual_mov_b32 v84, 26 :: v_dual_mov_b32 v85, 0            // 00000000299C: CA10009A 54540080
	v_dual_mov_b32 v86, 24 :: v_dual_mov_b32 v87, 0            // 0000000029A4: CA100098 56560080
	v_dual_mov_b32 v88, 22 :: v_dual_mov_b32 v89, 0            // 0000000029AC: CA100096 58580080
	v_dual_mov_b32 v90, 20 :: v_dual_mov_b32 v91, 0            // 0000000029B4: CA100094 5A5A0080
	v_dual_mov_b32 v92, 18 :: v_dual_mov_b32 v93, 0            // 0000000029BC: CA100092 5C5C0080
	s_wait_alu depctr_va_sdst(0)                               // 0000000029C4: BF88F19F
	v_add_co_ci_u32_e64 v95, null, 0, v79, s5                  // 0000000029C8: D5207C5F 00169E80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029D0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000029D4: 8C7E067E
	v_add_co_u32 v92, s5, v78, v92                             // 0000000029D8: D700055C 0202B94E
	s_wait_alu depctr_va_sdst(0)                               // 0000000029E0: BF88F19F
	v_add_co_ci_u32_e64 v93, null, v79, v93, s5                // 0000000029E4: D5207C5D 0016BB4F
	v_add_co_u32 v90, s5, v78, v90                             // 0000000029EC: D700055A 0202B54E
	s_wait_alu depctr_va_sdst(0)                               // 0000000029F4: BF88F19F
	v_add_co_ci_u32_e64 v91, null, v79, v91, s5                // 0000000029F8: D5207C5B 0016B74F
	v_add_co_u32 v88, s5, v78, v88                             // 000000002A00: D7000558 0202B14E
	s_wait_alu depctr_va_sdst(0)                               // 000000002A08: BF88F19F
	v_add_co_ci_u32_e64 v89, null, v79, v89, s5                // 000000002A0C: D5207C59 0016B34F
	v_add_co_u32 v86, s5, v78, v86                             // 000000002A14: D7000556 0202AD4E
	s_wait_alu depctr_va_sdst(0)                               // 000000002A1C: BF88F19F
	v_add_co_ci_u32_e64 v87, null, v79, v87, s5                // 000000002A20: D5207C57 0016AF4F
	v_add_co_u32 v84, s5, v78, v84                             // 000000002A28: D7000554 0202A94E
	s_wait_alu depctr_va_sdst(0)                               // 000000002A30: BF88F19F
	v_add_co_ci_u32_e64 v85, null, v79, v85, s5                // 000000002A34: D5207C55 0016AB4F
	v_add_co_u32 v82, s5, v78, v82                             // 000000002A3C: D7000552 0202A54E
	global_load_d16_b16 v75, v[94:95], off                     // 000000002A44: EE08007C 0000004B 0000005E
	s_wait_alu depctr_va_sdst(0)                               // 000000002A50: BF88F19F
	v_add_co_ci_u32_e64 v83, null, v79, v83, s5                // 000000002A54: D5207C53 0016A74F
	v_add_co_u32 v94, s5, v78, v80                             // 000000002A5C: D700055E 0202A14E
	s_wait_alu depctr_va_sdst(0)                               // 000000002A64: BF88F19F
	v_add_co_ci_u32_e64 v95, null, v79, v81, s5                // 000000002A68: D5207C5F 0016A34F
	s_clause 0x6                                               // 000000002A70: BF850006
	global_load_d16_hi_b16 v75, v[92:93], off                  // 000000002A74: EE08C07C 0000004B 0000005C
	global_load_d16_b16 v79, v[90:91], off                     // 000000002A80: EE08007C 0000004F 0000005A
	global_load_d16_hi_b16 v79, v[88:89], off                  // 000000002A8C: EE08C07C 0000004F 00000058
	global_load_d16_b16 v80, v[86:87], off                     // 000000002A98: EE08007C 00000050 00000056
	global_load_d16_hi_b16 v80, v[84:85], off                  // 000000002AA4: EE08C07C 00000050 00000054
	global_load_d16_b16 v81, v[82:83], off                     // 000000002AB0: EE08007C 00000051 00000052
	global_load_d16_hi_b16 v81, v[94:95], off                  // 000000002ABC: EE08C07C 00000051 0000005E
	s_wait_loadcnt 0x8                                         // 000000002AC8: BFC00008
	v_cndmask_b16 v12.l, 0, v12.l, vcc_lo                      // 000000002ACC: D65D000C 01AA1880
	v_cndmask_b16 v12.h, 0, v12.h, vcc_lo                      // 000000002AD4: D65D500C 01AA1880
	s_wait_loadcnt 0x6                                         // 000000002ADC: BFC00006
	v_cndmask_b16 v78.l, 0, v75.l, s4                          // 000000002AE0: D65D004E 00129680
	v_cndmask_b16 v78.h, 0, v75.h, s4                          // 000000002AE8: D65D504E 00129680
	s_wait_loadcnt 0x4                                         // 000000002AF0: BFC00004
	v_cndmask_b16 v79.l, 0, v79.l, s4                          // 000000002AF4: D65D004F 00129E80
	v_cndmask_b16 v79.h, 0, v79.h, s4                          // 000000002AFC: D65D504F 00129E80
	s_wait_loadcnt 0x2                                         // 000000002B04: BFC00002
	v_cndmask_b16 v80.l, 0, v80.l, s4                          // 000000002B08: D65D0050 0012A080
	v_cndmask_b16 v80.h, 0, v80.h, s4                          // 000000002B10: D65D5050 0012A080
	s_wait_loadcnt 0x0                                         // 000000002B18: BFC00000
	v_cndmask_b16 v81.l, 0, v81.l, s4                          // 000000002B1C: D65D0051 0012A280
	v_cndmask_b16 v81.h, 0, v81.h, s4                          // 000000002B24: D65D5051 0012A280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002B2C: BF870001
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:12], v[78:81], v[1:8]  // 000000002B30: CC404001 1C069D09
	s_and_saveexec_b32 s5, s3                                  // 000000002B38: BE852003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B3C: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000002B40: 8D05057E
	s_cbranch_execz 16                                         // 000000002B44: BFA50010 <attention_forward+0x1088>
	global_load_b96 v[9:11], v[37:38], off offset:16           // 000000002B48: EE05807C 00000009 00001025
	s_wait_loadcnt 0x0                                         // 000000002B54: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 000000002B58: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 000000002B60: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 000000002B68: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 000000002B70: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 000000002B78: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 000000002B80: D65D500B 01AA1680
	s_wait_alu depctr_sa_sdst(0)                               // 000000002B88: BF88FF9E
	s_or_saveexec_b32 s5, s5                                   // 000000002B8C: BE852205
	v_dual_mov_b32 v79, v40 :: v_dual_mov_b32 v78, v39         // 000000002B90: CA100128 4F4E0127
	v_dual_mov_b32 v81, v42 :: v_dual_mov_b32 v80, v41         // 000000002B98: CA10012A 51500129
	s_wait_alu depctr_sa_sdst(0)                               // 000000002BA0: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 000000002BA4: 8D7E057E
	s_cbranch_execz 20                                         // 000000002BA8: BFA50014 <attention_forward+0x10fc>
	global_load_b96 v[9:11], v[37:38], off                     // 000000002BAC: EE05807C 00000009 00000025
	v_dual_mov_b32 v79, v44 :: v_dual_mov_b32 v78, v43         // 000000002BB8: CA10012C 4F4E012B
	v_dual_mov_b32 v81, v48 :: v_dual_mov_b32 v80, v47         // 000000002BC0: CA100130 5150012F
	s_wait_loadcnt 0x0                                         // 000000002BC8: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 000000002BCC: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 000000002BD4: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 000000002BDC: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 000000002BE4: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 000000002BEC: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 000000002BF4: D65D500B 01AA1680
	s_or_b32 exec_lo, exec_lo, s5                              // 000000002BFC: 8C7E057E
	global_load_d16_b16 v12, v[78:79], off                     // 000000002C00: EE08007C 0000000C 0000004E
	global_load_d16_hi_b16 v12, v[80:81], off                  // 000000002C0C: EE08C07C 0000000C 00000050
	v_or_b32_e32 v75, 32, v77                                  // 000000002C18: 38969AA0
	v_dual_mov_b32 v80, 14 :: v_dual_mov_b32 v81, 0            // 000000002C1C: CA10008E 50500080
	v_dual_mov_b32 v82, 12 :: v_dual_mov_b32 v83, 0            // 000000002C24: CA10008C 52520080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000002C2C: BF8701B3
	v_cndmask_b32_e64 v75, 0, v75, s4                          // 000000002C30: D501004B 00129680
	v_dual_mov_b32 v84, 10 :: v_dual_mov_b32 v85, 0            // 000000002C38: CA10008A 54540080
	v_dual_mov_b32 v86, 8 :: v_dual_mov_b32 v87, 0             // 000000002C40: CA100088 56560080
	v_lshlrev_b64_e32 v[78:79], 1, v[75:76]                    // 000000002C48: 3E9C9681
	v_dual_mov_b32 v88, 6 :: v_dual_mov_b32 v89, 0             // 000000002C4C: CA100086 58580080
	v_dual_mov_b32 v90, 4 :: v_dual_mov_b32 v91, 0             // 000000002C54: CA100084 5A5A0080
	v_dual_mov_b32 v92, 2 :: v_dual_mov_b32 v93, 0             // 000000002C5C: CA100082 5C5C0080
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000002C64: BF8700B4
	v_add_co_u32 v78, s5, s36, v78                             // 000000002C68: D700054E 02029C24
	s_wait_alu depctr_va_sdst(0)                               // 000000002C70: BF88F19F
	v_add_co_ci_u32_e64 v79, null, s37, v79, s5                // 000000002C74: D5207C4F 00169E25
	v_dual_mov_b32 v94, v78 :: v_dual_mov_b32 v95, v79         // 000000002C7C: CA10014E 5E5E014F
	s_and_saveexec_b32 s6, s3                                  // 000000002C84: BE862003
	s_cbranch_execz 19                                         // 000000002C88: BFA50013 <attention_forward+0x11d8>
	v_add_co_u32 v94, s5, v78, 16                              // 000000002C8C: D700055E 0201214E
	v_dual_mov_b32 v80, 30 :: v_dual_mov_b32 v81, 0            // 000000002C94: CA10009E 50500080
	v_dual_mov_b32 v82, 28 :: v_dual_mov_b32 v83, 0            // 000000002C9C: CA10009C 52520080
	v_dual_mov_b32 v84, 26 :: v_dual_mov_b32 v85, 0            // 000000002CA4: CA10009A 54540080
	v_dual_mov_b32 v86, 24 :: v_dual_mov_b32 v87, 0            // 000000002CAC: CA100098 56560080
	v_dual_mov_b32 v88, 22 :: v_dual_mov_b32 v89, 0            // 000000002CB4: CA100096 58580080
	v_dual_mov_b32 v90, 20 :: v_dual_mov_b32 v91, 0            // 000000002CBC: CA100094 5A5A0080
	v_dual_mov_b32 v92, 18 :: v_dual_mov_b32 v93, 0            // 000000002CC4: CA100092 5C5C0080
	s_wait_alu depctr_va_sdst(0)                               // 000000002CCC: BF88F19F
	v_add_co_ci_u32_e64 v95, null, 0, v79, s5                  // 000000002CD0: D5207C5F 00169E80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CD8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002CDC: 8C7E067E
	v_add_co_u32 v92, s5, v78, v92                             // 000000002CE0: D700055C 0202B94E
	s_wait_alu depctr_va_sdst(0)                               // 000000002CE8: BF88F19F
	v_add_co_ci_u32_e64 v93, null, v79, v93, s5                // 000000002CEC: D5207C5D 0016BB4F
	v_add_co_u32 v90, s5, v78, v90                             // 000000002CF4: D700055A 0202B54E
	s_wait_alu depctr_va_sdst(0)                               // 000000002CFC: BF88F19F
	v_add_co_ci_u32_e64 v91, null, v79, v91, s5                // 000000002D00: D5207C5B 0016B74F
	v_add_co_u32 v88, s5, v78, v88                             // 000000002D08: D7000558 0202B14E
	s_wait_alu depctr_va_sdst(0)                               // 000000002D10: BF88F19F
	v_add_co_ci_u32_e64 v89, null, v79, v89, s5                // 000000002D14: D5207C59 0016B34F
	v_add_co_u32 v86, s5, v78, v86                             // 000000002D1C: D7000556 0202AD4E
	s_wait_alu depctr_va_sdst(0)                               // 000000002D24: BF88F19F
	v_add_co_ci_u32_e64 v87, null, v79, v87, s5                // 000000002D28: D5207C57 0016AF4F
	v_add_co_u32 v84, s5, v78, v84                             // 000000002D30: D7000554 0202A94E
	s_wait_alu depctr_va_sdst(0)                               // 000000002D38: BF88F19F
	v_add_co_ci_u32_e64 v85, null, v79, v85, s5                // 000000002D3C: D5207C55 0016AB4F
	v_add_co_u32 v82, s5, v78, v82                             // 000000002D44: D7000552 0202A54E
	global_load_d16_b16 v75, v[94:95], off                     // 000000002D4C: EE08007C 0000004B 0000005E
	s_wait_alu depctr_va_sdst(0)                               // 000000002D58: BF88F19F
	v_add_co_ci_u32_e64 v83, null, v79, v83, s5                // 000000002D5C: D5207C53 0016A74F
	v_add_co_u32 v94, s5, v78, v80                             // 000000002D64: D700055E 0202A14E
	s_wait_alu depctr_va_sdst(0)                               // 000000002D6C: BF88F19F
	v_add_co_ci_u32_e64 v95, null, v79, v81, s5                // 000000002D70: D5207C5F 0016A34F
	s_clause 0x6                                               // 000000002D78: BF850006
	global_load_d16_hi_b16 v75, v[92:93], off                  // 000000002D7C: EE08C07C 0000004B 0000005C
	global_load_d16_b16 v79, v[90:91], off                     // 000000002D88: EE08007C 0000004F 0000005A
	global_load_d16_hi_b16 v79, v[88:89], off                  // 000000002D94: EE08C07C 0000004F 00000058
	global_load_d16_b16 v80, v[86:87], off                     // 000000002DA0: EE08007C 00000050 00000056
	global_load_d16_hi_b16 v80, v[84:85], off                  // 000000002DAC: EE08C07C 00000050 00000054
	global_load_d16_b16 v81, v[82:83], off                     // 000000002DB8: EE08007C 00000051 00000052
	global_load_d16_hi_b16 v81, v[94:95], off                  // 000000002DC4: EE08C07C 00000051 0000005E
	s_wait_loadcnt 0x8                                         // 000000002DD0: BFC00008
	v_cndmask_b16 v12.l, 0, v12.l, vcc_lo                      // 000000002DD4: D65D000C 01AA1880
	v_cndmask_b16 v12.h, 0, v12.h, vcc_lo                      // 000000002DDC: D65D500C 01AA1880
	s_wait_loadcnt 0x6                                         // 000000002DE4: BFC00006
	v_cndmask_b16 v78.l, 0, v75.l, s4                          // 000000002DE8: D65D004E 00129680
	v_cndmask_b16 v78.h, 0, v75.h, s4                          // 000000002DF0: D65D504E 00129680
	s_wait_loadcnt 0x4                                         // 000000002DF8: BFC00004
	v_cndmask_b16 v79.l, 0, v79.l, s4                          // 000000002DFC: D65D004F 00129E80
	v_cndmask_b16 v79.h, 0, v79.h, s4                          // 000000002E04: D65D504F 00129E80
	s_wait_loadcnt 0x2                                         // 000000002E0C: BFC00002
	v_cndmask_b16 v80.l, 0, v80.l, s4                          // 000000002E10: D65D0050 0012A080
	v_cndmask_b16 v80.h, 0, v80.h, s4                          // 000000002E18: D65D5050 0012A080
	s_wait_loadcnt 0x0                                         // 000000002E20: BFC00000
	v_cndmask_b16 v81.l, 0, v81.l, s4                          // 000000002E24: D65D0051 0012A280
	v_cndmask_b16 v81.h, 0, v81.h, s4                          // 000000002E2C: D65D5051 0012A280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002E34: BF870001
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:12], v[78:81], v[1:8]  // 000000002E38: CC404001 1C069D09
	s_and_saveexec_b32 s5, s3                                  // 000000002E40: BE852003
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E44: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000002E48: 8D05057E
	s_cbranch_execz 16                                         // 000000002E4C: BFA50010 <attention_forward+0x1390>
	global_load_b96 v[9:11], v[45:46], off offset:16           // 000000002E50: EE05807C 00000009 0000102D
	s_wait_loadcnt 0x0                                         // 000000002E5C: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 000000002E60: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 000000002E68: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 000000002E70: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 000000002E78: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 000000002E80: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 000000002E88: D65D500B 01AA1680
	s_wait_alu depctr_sa_sdst(0)                               // 000000002E90: BF88FF9E
	s_or_saveexec_b32 s5, s5                                   // 000000002E94: BE852205
	v_dual_mov_b32 v79, v50 :: v_dual_mov_b32 v78, v49         // 000000002E98: CA100132 4F4E0131
	v_dual_mov_b32 v81, v52 :: v_dual_mov_b32 v80, v51         // 000000002EA0: CA100134 51500133
	s_wait_alu depctr_sa_sdst(0)                               // 000000002EA8: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 000000002EAC: 8D7E057E
	s_cbranch_execz 20                                         // 000000002EB0: BFA50014 <attention_forward+0x1404>
	global_load_b96 v[9:11], v[45:46], off                     // 000000002EB4: EE05807C 00000009 0000002D
	v_dual_mov_b32 v79, v54 :: v_dual_mov_b32 v78, v53         // 000000002EC0: CA100136 4F4E0135
	v_dual_mov_b32 v81, v56 :: v_dual_mov_b32 v80, v55         // 000000002EC8: CA100138 51500137
	s_wait_loadcnt 0x0                                         // 000000002ED0: BFC00000
	v_cndmask_b16 v9.l, 0, v9.l, vcc_lo                        // 000000002ED4: D65D0009 01AA1280
	v_cndmask_b16 v9.h, 0, v9.h, vcc_lo                        // 000000002EDC: D65D5009 01AA1280
	v_cndmask_b16 v10.l, 0, v10.l, vcc_lo                      // 000000002EE4: D65D000A 01AA1480
	v_cndmask_b16 v10.h, 0, v10.h, vcc_lo                      // 000000002EEC: D65D500A 01AA1480
	v_cndmask_b16 v11.l, 0, v11.l, vcc_lo                      // 000000002EF4: D65D000B 01AA1680
	v_cndmask_b16 v11.h, 0, v11.h, vcc_lo                      // 000000002EFC: D65D500B 01AA1680
	s_or_b32 exec_lo, exec_lo, s5                              // 000000002F04: 8C7E057E
	global_load_d16_b16 v12, v[78:79], off                     // 000000002F08: EE08007C 0000000C 0000004E
	global_load_d16_hi_b16 v12, v[80:81], off                  // 000000002F14: EE08C07C 0000000C 00000050
	v_or_b32_e32 v75, 48, v77                                  // 000000002F20: 38969AB0
	v_dual_mov_b32 v77, 14 :: v_dual_mov_b32 v78, 0            // 000000002F24: CA10008E 4D4E0080
	v_dual_mov_b32 v79, 12 :: v_dual_mov_b32 v80, 0            // 000000002F2C: CA10008C 4F500080
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000002F34: BF8701B3
	v_cndmask_b32_e64 v75, 0, v75, s4                          // 000000002F38: D501004B 00129680
	v_dual_mov_b32 v81, 10 :: v_dual_mov_b32 v82, 0            // 000000002F40: CA10008A 51520080
	v_dual_mov_b32 v83, 8 :: v_dual_mov_b32 v84, 0             // 000000002F48: CA100088 53540080
	v_lshlrev_b64_e32 v[75:76], 1, v[75:76]                    // 000000002F50: 3E969681
	v_dual_mov_b32 v85, 6 :: v_dual_mov_b32 v86, 0             // 000000002F54: CA100086 55560080
	v_dual_mov_b32 v87, 4 :: v_dual_mov_b32 v88, 0             // 000000002F5C: CA100084 57580080
	v_dual_mov_b32 v89, 2 :: v_dual_mov_b32 v90, 0             // 000000002F64: CA100082 595A0080
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000002F6C: BF8700B4
	v_add_co_u32 v75, s5, s36, v75                             // 000000002F70: D700054B 02029624
	s_wait_alu depctr_va_sdst(0)                               // 000000002F78: BF88F19F
	v_add_co_ci_u32_e64 v76, null, s37, v76, s5                // 000000002F7C: D5207C4C 00169825
	v_dual_mov_b32 v91, v75 :: v_dual_mov_b32 v92, v76         // 000000002F84: CA10014B 5B5C014C
	s_and_saveexec_b32 s6, s3                                  // 000000002F8C: BE862003
	s_cbranch_execz 19                                         // 000000002F90: BFA50013 <attention_forward+0x14e0>
	v_add_co_u32 v91, s5, v75, 16                              // 000000002F94: D700055B 0201214B
	v_dual_mov_b32 v77, 30 :: v_dual_mov_b32 v78, 0            // 000000002F9C: CA10009E 4D4E0080
	v_dual_mov_b32 v79, 28 :: v_dual_mov_b32 v80, 0            // 000000002FA4: CA10009C 4F500080
	v_dual_mov_b32 v81, 26 :: v_dual_mov_b32 v82, 0            // 000000002FAC: CA10009A 51520080
	v_dual_mov_b32 v83, 24 :: v_dual_mov_b32 v84, 0            // 000000002FB4: CA100098 53540080
	v_dual_mov_b32 v85, 22 :: v_dual_mov_b32 v86, 0            // 000000002FBC: CA100096 55560080
	v_dual_mov_b32 v87, 20 :: v_dual_mov_b32 v88, 0            // 000000002FC4: CA100094 57580080
	v_dual_mov_b32 v89, 18 :: v_dual_mov_b32 v90, 0            // 000000002FCC: CA100092 595A0080
	s_wait_alu depctr_va_sdst(0)                               // 000000002FD4: BF88F19F
	v_add_co_ci_u32_e64 v92, null, 0, v76, s5                  // 000000002FD8: D5207C5C 00169880
	s_wait_alu depctr_sa_sdst(0)                               // 000000002FE0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000002FE4: 8C7E067E
	v_add_co_u32 v89, s5, v75, v89                             // 000000002FE8: D7000559 0202B34B
	s_wait_alu depctr_va_sdst(0)                               // 000000002FF0: BF88F19F
	v_add_co_ci_u32_e64 v90, null, v76, v90, s5                // 000000002FF4: D5207C5A 0016B54C
	v_add_co_u32 v87, s5, v75, v87                             // 000000002FFC: D7000557 0202AF4B
	s_wait_alu depctr_va_sdst(0)                               // 000000003004: BF88F19F
	v_add_co_ci_u32_e64 v88, null, v76, v88, s5                // 000000003008: D5207C58 0016B14C
	v_add_co_u32 v85, s5, v75, v85                             // 000000003010: D7000555 0202AB4B
	s_wait_alu depctr_va_sdst(0)                               // 000000003018: BF88F19F
	v_add_co_ci_u32_e64 v86, null, v76, v86, s5                // 00000000301C: D5207C56 0016AD4C
	v_add_co_u32 v83, s5, v75, v83                             // 000000003024: D7000553 0202A74B
	s_wait_alu depctr_va_sdst(0)                               // 00000000302C: BF88F19F
	v_add_co_ci_u32_e64 v84, null, v76, v84, s5                // 000000003030: D5207C54 0016A94C
	v_add_co_u32 v81, s5, v75, v81                             // 000000003038: D7000551 0202A34B
	s_wait_alu depctr_va_sdst(0)                               // 000000003040: BF88F19F
	v_add_co_ci_u32_e64 v82, null, v76, v82, s5                // 000000003044: D5207C52 0016A54C
	v_add_co_u32 v79, s5, v75, v79                             // 00000000304C: D700054F 02029F4B
	global_load_d16_b16 v91, v[91:92], off                     // 000000003054: EE08007C 0000005B 0000005B
	s_wait_alu depctr_va_sdst(0)                               // 000000003060: BF88F19F
	v_add_co_ci_u32_e64 v80, null, v76, v80, s5                // 000000003064: D5207C50 0016A14C
	v_add_co_u32 v92, s5, v75, v77                             // 00000000306C: D700055C 02029B4B
	s_wait_alu depctr_va_sdst(0)                               // 000000003074: BF88F19F
	v_add_co_ci_u32_e64 v93, null, v76, v78, s5                // 000000003078: D5207C5D 00169D4C
	s_clause 0x6                                               // 000000003080: BF850006
	global_load_d16_b16 v75, v[89:90], off                     // 000000003084: EE08007C 0000004B 00000059
	global_load_d16_hi_b16 v75, v[87:88], off                  // 000000003090: EE08C07C 0000004B 00000057
	global_load_d16_b16 v76, v[85:86], off                     // 00000000309C: EE08007C 0000004C 00000055
	global_load_d16_hi_b16 v76, v[83:84], off                  // 0000000030A8: EE08C07C 0000004C 00000053
	global_load_d16_b16 v77, v[81:82], off                     // 0000000030B4: EE08007C 0000004D 00000051
	global_load_d16_hi_b16 v77, v[79:80], off                  // 0000000030C0: EE08C07C 0000004D 0000004F
	global_load_d16_b16 v78, v[92:93], off                     // 0000000030CC: EE08007C 0000004E 0000005C
	s_wait_loadcnt 0x8                                         // 0000000030D8: BFC00008
	v_cndmask_b16 v12.l, 0, v12.l, vcc_lo                      // 0000000030DC: D65D000C 01AA1880
	v_cndmask_b16 v12.h, 0, v12.h, vcc_lo                      // 0000000030E4: D65D500C 01AA1880
	v_cmp_le_i64_e64 s5, s[22:23], v[73:74]                    // 0000000030EC: D4530005 02029216
	v_cmp_lt_i64_e64 s6, v[57:58], v[73:74]                    // 0000000030F4: D4510006 02029339
	v_cmp_lt_i64_e64 s7, v[59:60], v[73:74]                    // 0000000030FC: D4510007 0202933B
	v_cmp_lt_i64_e64 s8, v[61:62], v[73:74]                    // 000000003104: D4510008 0202933D
	v_cmp_lt_i64_e64 s9, v[63:64], v[73:74]                    // 00000000310C: D4510009 0202933F
	v_cmp_lt_i64_e64 s10, v[65:66], v[73:74]                   // 000000003114: D451000A 02029341
	v_cmp_lt_i64_e64 s11, v[67:68], v[73:74]                   // 00000000311C: D451000B 02029343
	v_cmp_lt_i64_e64 s12, v[69:70], v[73:74]                   // 000000003124: D451000C 02029345
	v_cmp_lt_i64_e64 s13, v[71:72], v[73:74]                   // 00000000312C: D451000D 02029347
	s_wait_loadcnt 0x7                                         // 000000003134: BFC00007
	v_cndmask_b16 v73.l, 0, v91.l, s4                          // 000000003138: D65D0049 0012B680
	s_wait_loadcnt 0x5                                         // 000000003140: BFC00005
	v_cndmask_b16 v73.h, 0, v75.l, s4                          // 000000003144: D65D4049 00129680
	v_cndmask_b16 v74.l, 0, v75.h, s4                          // 00000000314C: D65D104A 00129680
	s_wait_loadcnt 0x3                                         // 000000003154: BFC00003
	v_cndmask_b16 v74.h, 0, v76.l, s4                          // 000000003158: D65D404A 00129880
	v_cndmask_b16 v75.l, 0, v76.h, s4                          // 000000003160: D65D104B 00129880
	s_wait_loadcnt 0x1                                         // 000000003168: BFC00001
	v_cndmask_b16 v75.h, 0, v77.l, s4                          // 00000000316C: D65D404B 00129A80
	v_cndmask_b16 v76.l, 0, v77.h, s4                          // 000000003174: D65D104C 00129A80
	s_wait_loadcnt 0x0                                         // 00000000317C: BFC00000
	v_cndmask_b16 v76.h, 0, v78.l, s4                          // 000000003180: D65D404C 00129C80
	s_and_b32 s4, s33, s6                                      // 000000003188: 8B040621
	s_and_b32 s6, s33, s7                                      // 00000000318C: 8B060721
	s_wait_alu depctr_sa_sdst(0)                               // 000000003190: BF88FF9E
	s_or_b32 s4, s5, s4                                        // 000000003194: 8C040405
	s_and_b32 s7, s33, s8                                      // 000000003198: 8B070821
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:12], v[73:76], v[1:8]  // 00000000319C: CC404001 1C069309
	s_and_b32 s8, s33, s9                                      // 0000000031A4: 8B080921
	s_and_b32 s9, s33, s10                                     // 0000000031A8: 8B090A21
	s_and_b32 s10, s33, s11                                    // 0000000031AC: 8B0A0B21
	s_and_b32 s11, s33, s12                                    // 0000000031B0: 8B0B0C21
	v_dual_mul_f32 v1, s42, v1 :: v_dual_mul_f32 v2, s42, v2   // 0000000031B4: C8C6022A 0102042A
	v_dual_mul_f32 v3, s42, v3 :: v_dual_mul_f32 v4, s42, v4   // 0000000031BC: C8C6062A 0304082A
	v_dual_mul_f32 v5, s42, v5 :: v_dual_mul_f32 v6, s42, v6   // 0000000031C4: C8C60A2A 05060C2A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031CC: BF88FF9E
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000031D0: BF870003
	v_cndmask_b32_e64 v1, v1, 0xf149f2ca, s4                   // 0000000031D4: D5010001 0011FF01 F149F2CA
	s_or_b32 s4, s5, s6                                        // 0000000031E0: 8C040605
	v_dual_mul_f32 v7, s42, v7 :: v_dual_mul_f32 v8, s42, v8   // 0000000031E4: C8C60E2A 0708102A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000031EC: BF88FF9E
	v_cndmask_b32_e64 v2, v2, 0xf149f2ca, s4                   // 0000000031F0: D5010002 0011FF02 F149F2CA
	s_or_b32 s4, s5, s7                                        // 0000000031FC: 8C040705
	s_and_b32 s12, s33, s13                                    // 000000003200: 8B0C0D21
	s_wait_alu depctr_sa_sdst(0)                               // 000000003204: BF88FF9E
	v_cndmask_b32_e64 v3, v3, 0xf149f2ca, s4                   // 000000003208: D5010003 0011FF03 F149F2CA
	s_or_b32 s4, s5, s8                                        // 000000003214: 8C040805
	s_wait_alu depctr_sa_sdst(0)                               // 000000003218: BF88FF9E
	v_cndmask_b32_e64 v4, v4, 0xf149f2ca, s4                   // 00000000321C: D5010004 0011FF04 F149F2CA
	s_or_b32 s4, s5, s9                                        // 000000003228: 8C040905
	s_wait_alu depctr_sa_sdst(0)                               // 00000000322C: BF88FF9E
	v_cndmask_b32_e64 v5, v5, 0xf149f2ca, s4                   // 000000003230: D5010005 0011FF05 F149F2CA
	s_or_b32 s4, s5, s10                                       // 00000000323C: 8C040A05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003240: BF88FF9E
	v_cndmask_b32_e64 v6, v6, 0xf149f2ca, s4                   // 000000003244: D5010006 0011FF06 F149F2CA
	s_or_b32 s4, s5, s11                                       // 000000003250: 8C040B05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003254: BF88FF9E
	v_cndmask_b32_e64 v7, v7, 0xf149f2ca, s4                   // 000000003258: D5010007 0011FF07 F149F2CA
	s_or_b32 s4, s5, s12                                       // 000000003264: 8C040C05
	s_wait_alu depctr_sa_sdst(0)                               // 000000003268: BF88FF9E
	v_cndmask_b32_e64 v8, v8, 0xf149f2ca, s4                   // 00000000326C: D5010008 0011FF08 F149F2CA
	ds_store_b32 v99, v1                                       // 000000003278: D8340000 00000163
	ds_store_b32 v100, v2                                      // 000000003280: D8340000 00000264
	ds_store_b32 v101, v3                                      // 000000003288: D8340000 00000365
	ds_store_b32 v102, v4                                      // 000000003290: D8340000 00000466
	ds_store_b32 v103, v5                                      // 000000003298: D8340000 00000567
	ds_store_b32 v104, v6                                      // 0000000032A0: D8340000 00000668
	ds_store_b32 v105, v7                                      // 0000000032A8: D8340000 00000769
	ds_store_b32 v106, v8                                      // 0000000032B0: D8340000 0000086A
	s_wait_dscnt 0x0                                           // 0000000032B8: BFC60000
	s_barrier_signal -1                                        // 0000000032BC: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000032C0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000032C4: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s43, s2                                 // 0000000032D0: BEAB2002
	s_cbranch_execz 572                                        // 0000000032D4: BFA5023C <attention_forward+0x20c8>
	ds_load_b128 v[73:76], v107 offset:4096                    // 0000000032D8: DBFC1000 4900006B
	ds_load_b128 v[77:80], v107 offset:4112                    // 0000000032E0: DBFC1010 4D00006B
	ds_load_b128 v[5:8], v107 offset:4128                      // 0000000032E8: DBFC1020 0500006B
	ds_load_b32 v11, v109                                      // 0000000032F0: D8D80000 0B00006D
	s_wait_dscnt 0x3                                           // 0000000032F8: BFC60003
	v_max_num_f32_e32 v1, v73, v73                             // 0000000032FC: 2C029349
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003300: BF870091
	v_max_num_f32_e32 v1, 0xf149f2ca, v1                       // 000000003304: 2C0202FF F149F2CA
	v_max3_num_f32 v9, v1, v74, v75                            // 00000000330C: D62A0009 052E9501
	ds_load_b128 v[1:4], v107 offset:4144                      // 000000003314: DBFC1030 0100006B
	s_wait_dscnt 0x3                                           // 00000000331C: BFC60003
	v_max3_num_f32 v9, v9, v76, v77                            // 000000003320: D62A0009 05369909
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000003328: BF8700C1
	v_max3_num_f32 v10, v9, v78, v79                           // 00000000332C: D62A000A 053E9D09
	ds_load_b32 v9, v108                                       // 000000003334: D8D80000 0900006C
	s_wait_dscnt 0x3                                           // 00000000333C: BFC60003
	v_max3_num_f32 v10, v10, v80, v5                           // 000000003340: D62A000A 0416A10A
	v_max3_num_f32 v10, v10, v6, v7                            // 000000003348: D62A000A 041E0D0A
	s_wait_dscnt 0x1                                           // 000000003350: BFC60001
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003354: BF870091
	v_max3_num_f32 v10, v10, v8, v1                            // 000000003358: D62A000A 0406110A
	v_max3_num_f32 v10, v10, v2, v3                            // 000000003360: D62A000A 040E050A
	s_wait_dscnt 0x0                                           // 000000003368: BFC60000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000336C: BF870091
	v_max3_num_f32 v10, v9, v10, v4                            // 000000003370: D62A000A 04121509
	v_sub_f32_e32 v12, v9, v10                                 // 000000003378: 08181509
	v_dual_sub_f32 v73, v73, v10 :: v_dual_sub_f32 v76, v76, v10// 00000000337C: C94A1549 494C154C
	v_dual_sub_f32 v75, v75, v10 :: v_dual_sub_f32 v78, v78, v10// 000000003384: C94A154B 4B4E154E
	v_dual_sub_f32 v74, v74, v10 :: v_dual_sub_f32 v5, v5, v10 // 00000000338C: C94A154A 4A041505
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000003394: BF870193
	v_dual_mul_f32 v82, 0x3fb8aa3b, v73 :: v_dual_mul_f32 v81, 0x3fb8aa3b, v12// 000000003398: C8C692FF 525018FF 3FB8AA3B
	v_dual_mul_f32 v84, 0x3fb8aa3b, v75 :: v_dual_mul_f32 v87, 0x3fb8aa3b, v78// 0000000033A4: C8C696FF 54569CFF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 0000000033B0: BF870233
	v_dual_mul_f32 v90, 0x3fb8aa3b, v5 :: v_dual_sub_f32 v77, v77, v10// 0000000033B4: C8CA0AFF 5A4C154D 3FB8AA3B
	v_sub_f32_e32 v80, v80, v10                                // 0000000033C0: 08A01550
	v_mul_f32_e32 v83, 0x3fb8aa3b, v74                         // 0000000033C4: 10A694FF 3FB8AA3B
	v_fma_f32 v135, 0x3fb8aa3b, v75, -v84                      // 0000000033CC: D6130087 855296FF 3FB8AA3B
	v_fma_f32 v141, 0x3fb8aa3b, v78, -v87                      // 0000000033D8: D613008D 855E9CFF 3FB8AA3B
	v_rndne_f32_e32 v142, v87                                  // 0000000033E4: 7F1C4757
	v_mul_f32_e32 v86, 0x3fb8aa3b, v77                         // 0000000033E8: 10AC9AFF 3FB8AA3B
	v_cmp_ngt_f32_e64 s13, 0xc2ce8ed0, v78                     // 0000000033F0: D41B000D 02029CFF C2CE8ED0
	v_cmp_nlt_f32_e64 s16, 0x42b17218, v78                     // 0000000033FC: D41E0010 02029CFF 42B17218
	v_fma_f32 v95, 0x3fb8aa3b, v74, -v83                       // 000000003408: D613005F 854E94FF 3FB8AA3B
	v_fmac_f32_e32 v135, 0x32a5705f, v75                       // 000000003414: 570E96FF 32A5705F
	v_fmac_f32_e32 v141, 0x32a5705f, v78                       // 00000000341C: 571A9CFF 32A5705F
	v_sub_f32_e32 v78, v87, v142                               // 000000003424: 089D1D57
	v_dual_sub_f32 v79, v79, v10 :: v_dual_sub_f32 v6, v6, v10 // 000000003428: C94A154F 4F061506
	v_fmac_f32_e32 v95, 0x32a5705f, v74                        // 000000003430: 56BE94FF 32A5705F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000003438: BF870223
	v_dual_mul_f32 v85, 0x3fb8aa3b, v76 :: v_dual_add_f32 v78, v78, v141// 00000000343C: C8C898FF 554F1B4E 3FB8AA3B
	v_cvt_i32_f32_e32 v87, v142                                // 000000003448: 7EAE118E
	v_mul_f32_e32 v88, 0x3fb8aa3b, v79                         // 00000000344C: 10B09EFF 3FB8AA3B
	v_fma_f32 v93, 0x3fb8aa3b, v73, -v82                       // 000000003454: D613005D 854A92FF 3FB8AA3B
	v_rndne_f32_e32 v134, v83                                  // 000000003460: 7F0C4753
	v_exp_f32_e32 v78, v78                                     // 000000003464: 7E9C4B4E
	v_mul_f32_e32 v89, 0x3fb8aa3b, v80                         // 000000003468: 10B2A0FF 3FB8AA3B
	v_fma_f32 v137, 0x3fb8aa3b, v76, -v85                      // 000000003470: D6130089 855698FF 3FB8AA3B
	v_rndne_f32_e32 v138, v85                                  // 00000000347C: 7F144755
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v74                      // 000000003480: D41B0005 020294FF C2CE8ED0
	v_cmp_nlt_f32_e64 s8, 0x42b17218, v74                      // 00000000348C: D41E0008 020294FF 42B17218
	v_cmp_ngt_f32_e64 s9, 0xc2ce8ed0, v76                      // 000000003498: D41B0009 020298FF C2CE8ED0
	v_cmp_nlt_f32_e64 s12, 0x42b17218, v76                     // 0000000034A4: D41E000C 020298FF 42B17218
	v_rndne_f32_e32 v94, v82                                   // 0000000034B0: 7EBC4752
	v_ldexp_f32 v78, v78, v87                                  // 0000000034B4: D71C004E 0202AF4E
	v_fma_f32 v143, 0x3fb8aa3b, v79, -v88                      // 0000000034BC: D613008F 85629EFF 3FB8AA3B
	v_rndne_f32_e32 v144, v88                                  // 0000000034C8: 7F204758
	v_dual_fmac_f32 v93, 0x32a5705f, v73 :: v_dual_sub_f32 v74, v83, v134// 0000000034CC: C80A92FF 5D4B0D53 32A5705F
	v_rndne_f32_e32 v146, v89                                  // 0000000034D8: 7F244759
	v_dual_fmac_f32 v137, 0x32a5705f, v76 :: v_dual_sub_f32 v76, v85, v138// 0000000034DC: C80A98FF 894D1555 32A5705F
	s_wait_alu depctr_va_sdst(0)                               // 0000000034E8: BF88F19F
	v_cndmask_b32_e64 v78, 0, v78, s13                         // 0000000034EC: D501004E 00369C80
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v73                      // 0000000034F4: D41B0004 020292FF C2CE8ED0
	v_cmp_nlt_f32_e64 s7, 0x42b17218, v73                      // 000000003500: D41E0007 020292FF 42B17218
	v_cmp_ngt_f32_e64 s15, 0xc2ce8ed0, v79                     // 00000000350C: D41B000F 02029EFF C2CE8ED0
	v_cmp_nlt_f32_e64 s17, 0x42b17218, v79                     // 000000003518: D41E0011 02029EFF 42B17218
	v_fmac_f32_e32 v143, 0x32a5705f, v79                       // 000000003524: 571E9EFF 32A5705F
	v_cndmask_b32_e64 v78, 0x7f800000, v78, s16                // 00000000352C: D501004E 00429CFF 7F800000
	v_dual_sub_f32 v79, v88, v144 :: v_dual_sub_f32 v88, v89, v146// 000000003538: C94B2158 4F592559
	v_dual_sub_f32 v73, v82, v94 :: v_dual_add_f32 v76, v76, v137// 000000003540: C948BD52 494D134C
	v_cvt_i32_f32_e32 v85, v138                                // 000000003548: 7EAA118A
	v_rndne_f32_e32 v136, v84                                  // 00000000354C: 7F104754
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v75                      // 000000003550: D41B0006 020296FF C2CE8ED0
	v_cmp_nlt_f32_e64 s10, 0x42b17218, v75                     // 00000000355C: D41E000A 020296FF 42B17218
	v_exp_f32_e32 v76, v76                                     // 000000003568: 7E984B4C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 00000000356C: BF870243
	v_dual_add_f32 v74, v74, v95 :: v_dual_sub_f32 v75, v84, v136// 000000003570: C90ABF4A 4A4B1154
	v_cvt_i32_f32_e32 v82, v94                                 // 000000003578: 7EA4115E
	v_cvt_i32_f32_e32 v83, v134                                // 00000000357C: 7EA61186
	v_cvt_i32_f32_e32 v84, v136                                // 000000003580: 7EA81188
	v_exp_f32_e32 v74, v74                                     // 000000003584: 7E944B4A
	v_add_f32_e32 v75, v75, v135                               // 000000003588: 06970F4B
	v_fma_f32 v139, 0x3fb8aa3b, v77, -v86                      // 00000000358C: D613008B 855A9AFF 3FB8AA3B
	s_delay_alu instid0(TRANS32_DEP_2)                         // 000000003598: BF870006
	v_ldexp_f32 v76, v76, v85                                  // 00000000359C: D71C004C 0202AB4C
	v_rndne_f32_e32 v140, v86                                  // 0000000035A4: 7F184756
	v_cmp_ngt_f32_e64 s11, 0xc2ce8ed0, v77                     // 0000000035A8: D41B000B 02029AFF C2CE8ED0
	v_exp_f32_e32 v75, v75                                     // 0000000035B4: 7E964B4B
	v_cmp_nlt_f32_e64 s14, 0x42b17218, v77                     // 0000000035B8: D41E000E 02029AFF 42B17218
	v_cndmask_b32_e64 v76, 0, v76, s9                          // 0000000035C4: D501004C 00269880
	v_ldexp_f32 v74, v74, v83                                  // 0000000035CC: D71C004A 0202A74A
	v_cvt_i32_f32_e32 v83, v144                                // 0000000035D4: 7EA61190
	v_fma_f32 v145, 0x3fb8aa3b, v80, -v89                      // 0000000035D8: D6130091 8566A0FF 3FB8AA3B
	v_cvt_i32_f32_e32 v85, v146                                // 0000000035E4: 7EAA1192
	v_cndmask_b32_e64 v76, 0x7f800000, v76, s12                // 0000000035E8: D501004C 003298FF 7F800000
	v_add_f32_e32 v73, v73, v93                                // 0000000035F4: 0692BB49
	v_ldexp_f32 v75, v75, v84                                  // 0000000035F8: D71C004B 0202A94B
	v_cndmask_b32_e64 v74, 0, v74, s5                          // 000000003600: D501004A 00169480
	v_cmp_ngt_f32_e64 s18, 0xc2ce8ed0, v80                     // 000000003608: D41B0012 0202A0FF C2CE8ED0
	v_fma_f32 v91, 0x3fb8aa3b, v12, -v81                       // 000000003614: D613005B 854618FF 3FB8AA3B
	v_exp_f32_e32 v73, v73                                     // 000000003620: 7E924B49
	s_wait_alu depctr_va_sdst(0)                               // 000000003624: BF88F19F
	v_cndmask_b32_e64 v75, 0, v75, s6                          // 000000003628: D501004B 001A9680
	v_cndmask_b32_e64 v74, 0x7f800000, v74, s8                 // 000000003630: D501004A 002294FF 7F800000
	v_rndne_f32_e32 v92, v81                                   // 00000000363C: 7EB84751
	v_dual_sub_f32 v7, v7, v10 :: v_dual_sub_f32 v2, v2, v10   // 000000003640: C94A1507 07021502
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000003648: BF870224
	v_cndmask_b32_e64 v75, 0x7f800000, v75, s10                // 00000000364C: D501004B 002A96FF 7F800000
	v_fma_f32 v147, 0x3fb8aa3b, v5, -v90                       // 000000003658: D6130093 856A0AFF 3FB8AA3B
	v_sub_f32_e32 v81, v81, v92                                // 000000003664: 08A2B951
	v_ldexp_f32 v73, v73, v82                                  // 000000003668: D71C0049 0202A549
	v_dual_sub_f32 v3, v3, v10 :: v_dual_sub_f32 v8, v8, v10   // 000000003670: C94A1503 03081508
	v_sub_f32_e32 v1, v1, v10                                  // 000000003678: 08021501
	v_mul_f32_e32 v87, 0x3fb8aa3b, v6                          // 00000000367C: 10AE0CFF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003684: BF870004
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000003688: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v80                      // 000000003690: D41E0004 0202A0FF 42B17218
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v7                       // 00000000369C: D41B0005 02020EFF C2CE8ED0
	v_cmp_ngt_f32_e64 s6, 0xc2ce8ed0, v12                      // 0000000036A8: D41B0006 020218FF C2CE8ED0
	v_rndne_f32_e32 v89, v87                                   // 0000000036B4: 7EB24757
	v_cndmask_b32_e64 v73, 0x7f800000, v73, s7                 // 0000000036B8: D5010049 001E92FF 7F800000
	v_dual_fmac_f32 v139, 0x32a5705f, v77 :: v_dual_sub_f32 v4, v4, v10// 0000000036C4: C80A9AFF 8B041504 32A5705F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000036D0: BF870122
	v_dual_add_f32 v82, v73, v74 :: v_dual_sub_f32 v77, v86, v140// 0000000036D4: C90A9549 524D1956
	v_cvt_i32_f32_e32 v86, v140                                // 0000000036DC: 7EAC118C
	v_dual_add_f32 v82, v75, v82 :: v_dual_add_f32 v77, v77, v139// 0000000036E0: C908A54B 524D174D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000036E8: BF870111
	v_add_f32_e32 v82, v76, v82                                // 0000000036EC: 06A4A54C
	v_exp_f32_e32 v77, v77                                     // 0000000036F0: 7E9A4B4D
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000036F4: BF870125
	v_ldexp_f32 v77, v77, v86                                  // 0000000036F8: D71C004D 0202AD4D
	v_cvt_i32_f32_e32 v86, v92                                 // 000000003700: 7EAC115C
	v_cndmask_b32_e64 v77, 0, v77, s11                         // 000000003704: D501004D 002E9A80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000370C: BF870091
	v_cndmask_b32_e64 v77, 0x7f800000, v77, s14                // 000000003710: D501004D 003A9AFF 7F800000
	v_add_f32_e32 v82, v77, v82                                // 00000000371C: 06A4A54D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003720: BF870091
	v_dual_add_f32 v82, v78, v82 :: v_dual_add_f32 v79, v79, v143// 000000003724: C908A54E 524F1F4F
	v_exp_f32_e32 v79, v79                                     // 00000000372C: 7E9E4B4F
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003730: BF870125
	v_ldexp_f32 v79, v79, v83                                  // 000000003734: D71C004F 0202A74F
	v_rndne_f32_e32 v83, v90                                   // 00000000373C: 7EA6475A
	v_cndmask_b32_e64 v79, 0, v79, s15                         // 000000003740: D501004F 003E9E80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003748: BF870091
	v_cndmask_b32_e64 v79, 0x7f800000, v79, s17                // 00000000374C: D501004F 00469EFF 7F800000
	v_dual_fmac_f32 v145, 0x32a5705f, v80 :: v_dual_add_f32 v82, v79, v82// 000000003758: C808A0FF 9152A54F 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003764: BF870121
	v_add_f32_e32 v84, v88, v145                               // 000000003768: 06A92358
	v_fma_f32 v88, 0x3fb8aa3b, v6, -v87                        // 00000000376C: D6130058 855E0CFF 3FB8AA3B
	v_exp_f32_e32 v84, v84                                     // 000000003778: 7EA84B54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000000377C: BF870291
	v_fmac_f32_e32 v88, 0x32a5705f, v6                         // 000000003780: 56B00CFF 32A5705F
	v_ldexp_f32 v84, v84, v85                                  // 000000003788: D71C0054 0202AB54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000003790: BF8700A1
	v_cndmask_b32_e64 v84, 0, v84, s18                         // 000000003794: D5010054 004AA880
	s_wait_alu depctr_va_sdst(0)                               // 00000000379C: BF88F19F
	v_cndmask_b32_e64 v80, 0x7f800000, v84, s4                 // 0000000037A0: D5010050 0012A8FF 7F800000
	v_fmac_f32_e32 v91, 0x32a5705f, v12                        // 0000000037AC: 56B618FF 32A5705F
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v5                       // 0000000037B4: D41B0004 02020AFF C2CE8ED0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 0000000037C0: BF870223
	v_dual_add_f32 v82, v80, v82 :: v_dual_sub_f32 v85, v90, v83// 0000000037C4: C90AA550 5254A75A
	v_mul_f32_e32 v90, 0x3fb8aa3b, v7                          // 0000000037CC: 10B40EFF 3FB8AA3B
	v_add_f32_e32 v81, v81, v91                                // 0000000037D4: 06A2B751
	v_cvt_i32_f32_e32 v83, v83                                 // 0000000037D8: 7EA61153
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000037DC: BF8700A3
	v_rndne_f32_e32 v91, v90                                   // 0000000037E0: 7EB6475A
	v_fmac_f32_e32 v147, 0x32a5705f, v5                        // 0000000037E4: 57260AFF 32A5705F
	v_add_f32_e32 v85, v85, v147                               // 0000000037EC: 06AB2755
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 0000000037F0: BF8701C1
	v_exp_f32_e32 v84, v85                                     // 0000000037F4: 7EA84B55
	v_sub_f32_e32 v85, v87, v89                                // 0000000037F8: 08AAB357
	v_cvt_i32_f32_e32 v89, v89                                 // 0000000037FC: 7EB21159
	v_fma_f32 v87, 0x3fb8aa3b, v7, -v90                        // 000000003800: D6130057 856A0EFF 3FB8AA3B
	v_dual_add_f32 v85, v85, v88 :: v_dual_sub_f32 v88, v90, v91// 00000000380C: C90AB155 5558B75A
	v_mul_f32_e32 v90, 0x3fb8aa3b, v8                          // 000000003814: 10B410FF 3FB8AA3B
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000381C: BF870195
	v_ldexp_f32 v83, v84, v83                                  // 000000003820: D71C0053 0202A754
	v_exp_f32_e32 v84, v85                                     // 000000003828: 7EA84B55
	s_wait_alu depctr_va_sdst(0)                               // 00000000382C: BF88F19F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000003830: BF8700B1
	v_cndmask_b32_e64 v83, 0, v83, s4                          // 000000003834: D5010053 0012A680
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v5                       // 00000000383C: D41E0004 02020AFF 42B17218
	s_wait_alu depctr_va_sdst(0)                               // 000000003848: BF88F19F
	v_cndmask_b32_e64 v5, 0x7f800000, v83, s4                  // 00000000384C: D5010005 0012A6FF 7F800000
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000003858: BF870235
	v_ldexp_f32 v83, v84, v89                                  // 00000000385C: D71C0053 0202B354
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v6                       // 000000003864: D41B0004 02020CFF C2CE8ED0
	v_cvt_i32_f32_e32 v84, v91                                 // 000000003870: 7EA8115B
	v_add_f32_e32 v82, v5, v82                                 // 000000003874: 06A4A505
	s_wait_alu depctr_va_sdst(0)                               // 000000003878: BF88F19F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 00000000387C: BF8700B3
	v_cndmask_b32_e64 v83, 0, v83, s4                          // 000000003880: D5010053 0012A680
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v6                       // 000000003888: D41E0004 02020CFF 42B17218
	s_wait_alu depctr_va_sdst(0)                               // 000000003894: BF88F19F
	v_cndmask_b32_e64 v6, 0x7f800000, v83, s4                  // 000000003898: D5010006 0012A6FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v7                       // 0000000038A4: D41E0004 02020EFF 42B17218
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000038B0: BF870092
	v_dual_add_f32 v82, v6, v82 :: v_dual_fmac_f32 v87, 0x32a5705f, v7// 0000000038B4: C900A506 52560EFF 32A5705F
	v_add_f32_e32 v85, v88, v87                                // 0000000038C0: 06AAAF58
	v_fma_f32 v87, 0x3fb8aa3b, v8, -v90                        // 0000000038C4: D6130057 856A10FF 3FB8AA3B
	v_rndne_f32_e32 v88, v90                                   // 0000000038D0: 7EB0475A
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000038D4: BF870113
	v_exp_f32_e32 v85, v85                                     // 0000000038D8: 7EAA4B55
	v_fmac_f32_e32 v87, 0x32a5705f, v8                         // 0000000038DC: 56AE10FF 32A5705F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000038E4: BF870132
	v_sub_f32_e32 v90, v90, v88                                // 0000000038E8: 08B4B15A
	v_exp_f32_e32 v81, v81                                     // 0000000038EC: 7EA24B51
	v_cvt_i32_f32_e32 v83, v88                                 // 0000000038F0: 7EA61158
	v_add_f32_e32 v87, v90, v87                                // 0000000038F4: 06AEAF5A
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000038F8: BF870116
	v_ldexp_f32 v84, v85, v84                                  // 0000000038FC: D71C0054 0202A955
	v_exp_f32_e32 v85, v87                                     // 000000003904: 7EAA4B57
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000003908: BF8701A6
	v_ldexp_f32 v81, v81, v86                                  // 00000000390C: D71C0051 0202AD51
	v_mul_f32_e32 v86, 0x3fb8aa3b, v1                          // 000000003914: 10AC02FF 3FB8AA3B
	v_cndmask_b32_e64 v84, 0, v84, s5                          // 00000000391C: D5010054 0016A880
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v8                       // 000000003924: D41B0005 020210FF C2CE8ED0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000003930: BF870214
	v_cndmask_b32_e64 v81, 0, v81, s6                          // 000000003934: D5010051 001AA280
	v_rndne_f32_e32 v87, v86                                   // 00000000393C: 7EAE4756
	s_wait_alu depctr_va_sdst(0)                               // 000000003940: BF88F19F
	v_cndmask_b32_e64 v7, 0x7f800000, v84, s4                  // 000000003944: D5010007 0012A8FF 7F800000
	v_ldexp_f32 v83, v85, v83                                  // 000000003950: D71C0053 0202A755
	v_fma_f32 v85, 0x3fb8aa3b, v1, -v86                        // 000000003958: D6130055 855A02FF 3FB8AA3B
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v8                       // 000000003964: D41E0004 020210FF 42B17218
	v_sub_f32_e32 v84, v86, v87                                // 000000003970: 08A8AF56
	v_mul_f32_e32 v86, 0x3fb8aa3b, v2                          // 000000003974: 10AC04FF 3FB8AA3B
	v_cndmask_b32_e64 v83, 0, v83, s5                          // 00000000397C: D5010053 0016A680
	v_fmac_f32_e32 v85, 0x32a5705f, v1                         // 000000003984: 56AA02FF 32A5705F
	v_cvt_i32_f32_e32 v87, v87                                 // 00000000398C: 7EAE1157
	v_add_f32_e32 v82, v7, v82                                 // 000000003990: 06A4A507
	v_rndne_f32_e32 v88, v86                                   // 000000003994: 7EB04756
	s_wait_alu depctr_va_sdst(0)                               // 000000003998: BF88F19F
	v_cndmask_b32_e64 v8, 0x7f800000, v83, s4                  // 00000000399C: D5010008 0012A6FF 7F800000
	v_add_f32_e32 v84, v84, v85                                // 0000000039A8: 06A8AB54
	v_fma_f32 v85, 0x3fb8aa3b, v2, -v86                        // 0000000039AC: D6130055 855A04FF 3FB8AA3B
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v2                       // 0000000039B8: D41B0005 020204FF C2CE8ED0
	v_sub_f32_e32 v86, v86, v88                                // 0000000039C4: 08ACB156
	v_cvt_i32_f32_e32 v88, v88                                 // 0000000039C8: 7EB01158
	v_exp_f32_e32 v83, v84                                     // 0000000039CC: 7EA64B54
	v_fmac_f32_e32 v85, 0x32a5705f, v2                         // 0000000039D0: 56AA04FF 32A5705F
	v_add_f32_e32 v82, v8, v82                                 // 0000000039D8: 06A4A508
	v_cmp_nlt_f32_e64 s6, 0x42b17218, v12                      // 0000000039DC: D41E0006 020218FF 42B17218
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)// 0000000039E8: BF8702A3
	v_dual_add_f32 v85, v86, v85 :: v_dual_mul_f32 v86, 0x3fb8aa3b, v4// 0000000039EC: C906AB56 555608FF 3FB8AA3B
	v_mul_f32_e32 v84, 0x3fb8aa3b, v3                          // 0000000039F8: 10A806FF 3FB8AA3B
	v_ldexp_f32 v83, v83, v87                                  // 000000003A00: D71C0053 0202AF53
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v1                       // 000000003A08: D41B0004 020202FF C2CE8ED0
	s_wait_alu depctr_va_sdst(0)                               // 000000003A14: BF88F19F
	v_cndmask_b32_e64 v81, 0x7f800000, v81, s6                 // 000000003A18: D5010051 001AA2FF 7F800000
	v_rndne_f32_e32 v91, v86                                   // 000000003A24: 7EB64756
	v_fma_f32 v87, 0x3fb8aa3b, v4, -v86                        // 000000003A28: D6130057 855A08FF 3FB8AA3B
	v_fma_f32 v89, 0x3fb8aa3b, v3, -v84                        // 000000003A34: D6130059 855206FF 3FB8AA3B
	v_rndne_f32_e32 v90, v84                                   // 000000003A40: 7EB44754
	v_cndmask_b32_e64 v83, 0, v83, s4                          // 000000003A44: D5010053 0012A680
	v_sub_f32_e32 v86, v86, v91                                // 000000003A4C: 08ACB756
	v_exp_f32_e32 v85, v85                                     // 000000003A50: 7EAA4B55
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v1                       // 000000003A54: D41E0004 020202FF 42B17218
	v_dual_fmac_f32 v89, 0x32a5705f, v3 :: v_dual_sub_f32 v84, v84, v90// 000000003A60: C80A06FF 5954B554 32A5705F
	s_wait_alu depctr_va_sdst(0)                               // 000000003A6C: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000003A70: BF8701A2
	v_cndmask_b32_e64 v1, 0x7f800000, v83, s4                  // 000000003A74: D5010001 0012A6FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v2                       // 000000003A80: D41E0004 020204FF 42B17218
	v_add_f32_e32 v84, v84, v89                                // 000000003A8C: 06A8B354
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000003A90: BF870215
	v_ldexp_f32 v85, v85, v88                                  // 000000003A94: D71C0055 0202B155
	v_dual_fmac_f32 v87, 0x32a5705f, v4 :: v_dual_add_f32 v82, v1, v82// 000000003A9C: C80808FF 5752A501 32A5705F
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003AA8: BF870113
	v_exp_f32_e32 v84, v84                                     // 000000003AAC: 7EA84B54
	v_cndmask_b32_e64 v83, 0, v85, s5                          // 000000003AB0: D5010053 0016AA80
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v3                       // 000000003AB8: D41B0005 020206FF C2CE8ED0
	s_wait_alu depctr_va_sdst(0)                               // 000000003AC4: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000003AC8: BF870122
	v_cndmask_b32_e64 v2, 0x7f800000, v83, s4                  // 000000003ACC: D5010002 0012A6FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v3                       // 000000003AD8: D41E0004 020206FF 42B17218
	v_add_f32_e32 v12, v2, v82                                 // 000000003AE4: 0618A502
	v_add_f32_e32 v86, v86, v87                                // 000000003AE8: 06ACAF56
	v_cvt_i32_f32_e32 v87, v90                                 // 000000003AEC: 7EAE115A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003AF0: BF870092
	v_exp_f32_e32 v85, v86                                     // 000000003AF4: 7EAA4B56
	v_ldexp_f32 v84, v84, v87                                  // 000000003AF8: D71C0054 0202AF54
	v_cvt_i32_f32_e32 v86, v91                                 // 000000003B00: 7EAC115B
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003B04: BF870002
	v_cndmask_b32_e64 v83, 0, v84, s5                          // 000000003B08: D5010053 0016A880
	v_cmp_ngt_f32_e64 s5, 0xc2ce8ed0, v4                       // 000000003B10: D41B0005 020208FF C2CE8ED0
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_3)   // 000000003B1C: BF870185
	v_ldexp_f32 v84, v85, v86                                  // 000000003B20: D71C0054 0202AD55
	s_wait_alu depctr_va_sdst(0)                               // 000000003B28: BF88F19F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000003B2C: BF8701A3
	v_cndmask_b32_e64 v3, 0x7f800000, v83, s4                  // 000000003B30: D5010003 0012A6FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v4                       // 000000003B3C: D41E0004 020208FF 42B17218
	v_cndmask_b32_e64 v82, 0, v84, s5                          // 000000003B48: D5010052 0016A880
	v_cmp_nge_f32_e64 s5, 0xf149f2ca, v9                       // 000000003B50: D4190005 020212FF F149F2CA
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000003B5C: BF8701A4
	v_add_f32_e32 v9, v3, v12                                  // 000000003B60: 06121903
	s_wait_alu depctr_va_sdst(0)                               // 000000003B64: BF88F19F
	v_cndmask_b32_e64 v4, 0x7f800000, v82, s4                  // 000000003B68: D5010004 0012A4FF 7F800000
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003B74: BF870113
	v_cndmask_b32_e64 v12, 0, v81, s5                          // 000000003B78: D501000C 0016A280
	v_add_f32_e32 v9, v4, v9                                   // 000000003B80: 06121304
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003B84: BF870092
	v_mul_f32_e32 v11, v12, v11                                // 000000003B88: 1016170C
	v_add_f32_e32 v9, v11, v9                                  // 000000003B8C: 0612130B
	ds_store_b128 v107, v[73:76] offset:4096                   // 000000003B90: DB7C1000 0000496B
	ds_store_b128 v107, v[77:80] offset:4112                   // 000000003B98: DB7C1010 00004D6B
	ds_store_b128 v107, v[5:8] offset:4128                     // 000000003BA0: DB7C1020 0000056B
	ds_store_b128 v107, v[1:4] offset:4144                     // 000000003BA8: DB7C1030 0000016B
	ds_store_b32 v109, v9                                      // 000000003BB0: D8340000 0000096D
	ds_store_b32 v108, v10                                     // 000000003BB8: D8340000 00000A6C
	ds_store_b32 v110, v12                                     // 000000003BC0: D8340000 00000C6E
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BC8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s43                             // 000000003BCC: 8C7E2B7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000003BD0: BFC80000
	s_barrier_signal -1                                        // 000000003BD4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003BD8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003BDC: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s4, s3                                  // 000000003BE8: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003BEC: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000003BF0: 8D04047E
	s_cbranch_execz 12                                         // 000000003BF4: BFA5000C <attention_forward+0x2128>
	ds_load_b128 v[1:4], v111 offset:32                        // 000000003BF8: DBFC0020 0100006F
	ds_load_b64 v[5:6], v111 offset:48                         // 000000003C00: D9D80030 0500006F
	s_wait_dscnt 0x1                                           // 000000003C08: BFC60001
	v_cvt_f16_f32_e32 v9.l, v1                                 // 000000003C0C: 7E121501
	v_cvt_f16_f32_e32 v9.h, v2                                 // 000000003C10: 7F121502
	v_cvt_f16_f32_e32 v10.l, v3                                // 000000003C14: 7E141503
	v_cvt_f16_f32_e32 v10.h, v4                                // 000000003C18: 7F141504
	s_wait_dscnt 0x0                                           // 000000003C1C: BFC60000
	v_cvt_f16_f32_e32 v11.l, v5                                // 000000003C20: 7E161505
	v_cvt_f16_f32_e32 v11.h, v6                                // 000000003C24: 7F161506
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C28: BF88FF9E
	s_or_saveexec_b32 s4, s4                                   // 000000003C2C: BE842204
	v_dual_mov_b32 v1, v112 :: v_dual_mov_b32 v2, v113         // 000000003C30: CA100170 01020171
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C38: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s4                             // 000000003C3C: 8D7E047E
	s_cbranch_execz 14                                         // 000000003C40: BFA5000E <attention_forward+0x217c>
	ds_load_b128 v[2:5], v111                                  // 000000003C44: DBFC0000 0200006F
	ds_load_b64 v[6:7], v111 offset:16                         // 000000003C4C: D9D80010 0600006F
	v_mov_b32_e32 v1, v114                                     // 000000003C54: 7E020372
	s_wait_dscnt 0x1                                           // 000000003C58: BFC60001
	v_cvt_f16_f32_e32 v9.l, v2                                 // 000000003C5C: 7E121502
	v_cvt_f16_f32_e32 v9.h, v3                                 // 000000003C60: 7F121503
	v_cvt_f16_f32_e32 v10.l, v4                                // 000000003C64: 7E141504
	v_cvt_f16_f32_e32 v10.h, v5                                // 000000003C68: 7F141505
	s_wait_dscnt 0x0                                           // 000000003C6C: BFC60000
	v_cvt_f16_f32_e32 v11.l, v6                                // 000000003C70: 7E161506
	v_cvt_f16_f32_e32 v11.h, v7                                // 000000003C74: 7F161507
	v_mov_b32_e32 v2, v115                                     // 000000003C78: 7E040373
	s_or_b32 exec_lo, exec_lo, s4                              // 000000003C7C: 8C7E047E
	ds_load_b32 v12, v1                                        // 000000003C80: D8D80000 0C000001
	ds_load_b32 v83, v2                                        // 000000003C88: D8D80000 53000002
	s_and_saveexec_b32 s4, s3                                  // 000000003C90: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000003C94: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000003C98: 8D04047E
	s_or_b32 s6, s40, 8                                        // 000000003C9C: 8C068828
	s_mov_b32 s7, s41                                          // 000000003CA0: BE870029
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CA4: BF88FF9E
	s_add_nc_u64 s[8:9], s[6:7], s[34:35]                      // 000000003CA8: A9882206
	v_cmp_lt_i64_e64 s12, s[6:7], s[22:23]                     // 000000003CAC: D451000C 02002C06
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CB4: BF88FF9E
	s_lshl_b64 s[8:9], s[8:9], 6                               // 000000003CB8: 84888608
	s_wait_alu depctr_sa_sdst(0)                               // 000000003CBC: BF88FF9E
	v_or_b32_e32 v1, s8, v15                                   // 000000003CC0: 38021E08
	v_cndmask_b32_e64 v76, 0, s9, s12                          // 000000003CC4: D501004C 00301280
	s_delay_alu instid0(VALU_DEP_2)                            // 000000003CCC: BF870002
	v_cndmask_b32_e64 v75, 0, v1, s12                          // 000000003CD0: D501004B 00320280
	s_or_saveexec_b32 s5, s4                                   // 000000003CD8: BE852204
	v_dual_mov_b32 v1, 15 :: v_dual_mov_b32 v2, 0              // 000000003CDC: CA10008F 01020080
	v_dual_mov_b32 v5, 14 :: v_dual_mov_b32 v6, 0              // 000000003CE4: CA10008E 05060080
	v_dual_mov_b32 v7, 13 :: v_dual_mov_b32 v8, 0              // 000000003CEC: CA10008D 07080080
	v_dual_mov_b32 v73, 12 :: v_dual_mov_b32 v74, 0            // 000000003CF4: CA10008C 494A0080
	v_dual_mov_b32 v77, 11 :: v_dual_mov_b32 v78, 0            // 000000003CFC: CA10008B 4D4E0080
	v_dual_mov_b32 v79, 10 :: v_dual_mov_b32 v80, 0            // 000000003D04: CA10008A 4F500080
	v_dual_mov_b32 v81, 9 :: v_dual_mov_b32 v82, 0             // 000000003D0C: CA100089 51520080
	v_dual_mov_b32 v3, v15 :: v_dual_mov_b32 v4, v16           // 000000003D14: CA10010F 03040110
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D1C: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 000000003D20: 8D7E057E
	s_cbranch_execz 35                                         // 000000003D24: BFA50023 <attention_forward+0x22b4>
	s_lshl_b64 s[6:7], s[30:31], 10                            // 000000003D28: 84868A1E
	v_dual_mov_b32 v1, 7 :: v_dual_mov_b32 v2, 0               // 000000003D2C: CA100087 01020080
	s_wait_alu depctr_sa_sdst(0)                               // 000000003D34: BF88FF9E
	v_add_co_u32 v3, s4, s6, v97                               // 000000003D38: D7000403 0202C206
	s_wait_alu depctr_va_sdst(0)                               // 000000003D40: BF88F19F
	v_add_co_ci_u32_e64 v4, null, s7, v98, s4                  // 000000003D44: D5207C04 0012C407
	v_cmp_lt_i64_e64 s4, s[40:41], s[22:23]                    // 000000003D4C: D4510004 02002C28
	v_dual_mov_b32 v5, 6 :: v_dual_mov_b32 v6, 0               // 000000003D54: CA100086 05060080
	v_dual_mov_b32 v7, 5 :: v_dual_mov_b32 v8, 0               // 000000003D5C: CA100085 07080080
	v_dual_mov_b32 v73, 4 :: v_dual_mov_b32 v74, 0             // 000000003D64: CA100084 494A0080
	s_wait_alu depctr_va_sdst(0)                               // 000000003D6C: BF88F19F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003D70: BF870004
	v_cndmask_b32_e64 v76, 0, v4, s4                           // 000000003D74: D501004C 00120880
	v_cndmask_b32_e64 v75, 0, v3, s4                           // 000000003D7C: D501004B 00120680
	v_dual_mov_b32 v77, 3 :: v_dual_mov_b32 v78, 0             // 000000003D84: CA100083 4D4E0080
	v_dual_mov_b32 v79, 2 :: v_dual_mov_b32 v80, 0             // 000000003D8C: CA100082 4F500080
	v_dual_mov_b32 v81, 1 :: v_dual_mov_b32 v82, 0             // 000000003D94: CA100081 51520080
	v_dual_mov_b32 v3, v13 :: v_dual_mov_b32 v4, v14           // 000000003D9C: CA10010D 0304010E
	s_and_not1_b32 s6, s12, exec_lo                            // 000000003DA4: 91067E0C
	s_and_b32 s4, s4, exec_lo                                  // 000000003DA8: 8B047E04
	s_wait_alu depctr_sa_sdst(0)                               // 000000003DAC: BF88FF9E
	s_or_b32 s12, s6, s4                                       // 000000003DB0: 8C0C0406
	s_or_b32 exec_lo, exec_lo, s5                              // 000000003DB4: 8C7E057E
	v_or_b32_e32 v81, s40, v81                                 // 000000003DB8: 38A2A228
	v_or_b32_e32 v82, s41, v82                                 // 000000003DBC: 38A4A429
	v_or_b32_e32 v79, s40, v79                                 // 000000003DC0: 389E9E28
	v_or_b32_e32 v80, s41, v80                                 // 000000003DC4: 38A0A029
	v_or_b32_e32 v77, s40, v77                                 // 000000003DC8: 389A9A28
	v_add_co_u32 v84, s4, v81, s34                             // 000000003DCC: D7000454 02004551
	s_wait_alu depctr_va_sdst(0)                               // 000000003DD4: BF88F19F
	v_add_co_ci_u32_e64 v85, null, s35, v82, s4                // 000000003DD8: D5207C55 0012A423
	v_add_co_u32 v86, s4, v79, s34                             // 000000003DE0: D7000456 0200454F
	s_wait_alu depctr_va_sdst(0)                               // 000000003DE8: BF88F19F
	v_add_co_ci_u32_e64 v87, null, s35, v80, s4                // 000000003DEC: D5207C57 0012A023
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000003DF4: BF870233
	v_lshlrev_b64_e32 v[84:85], 6, v[84:85]                    // 000000003DF8: 3EA8A886
	v_cmp_gt_i64_e64 s4, s[22:23], v[81:82]                    // 000000003DFC: D4540004 0202A216
	v_or_b32_e32 v78, s41, v78                                 // 000000003E04: 389C9C29
	v_lshlrev_b64_e32 v[81:82], 6, v[86:87]                    // 000000003E08: 3EA2AC86
	v_lshlrev_b64_e32 v[75:76], 1, v[75:76]                    // 000000003E0C: 3E969681
	v_cmp_gt_i64_e64 s5, s[22:23], v[79:80]                    // 000000003E10: D4540005 02029E16
	v_or_b32_e32 v85, v85, v4                                  // 000000003E18: 38AA0955
	v_or_b32_e32 v84, v84, v3                                  // 000000003E1C: 38A80754
	v_or_b32_e32 v73, s40, v73                                 // 000000003E20: 38929228
	v_or_b32_e32 v87, v81, v3                                  // 000000003E24: 38AE0751
	v_add_co_u32 v81, s6, v77, s34                             // 000000003E28: D7000651 0200454D
	s_wait_alu depctr_va_sdst(0)                               // 000000003E30: BF88F19F
	v_cndmask_b32_e64 v85, 0, v85, s4                          // 000000003E34: D5010055 0012AA80
	v_cndmask_b32_e64 v84, 0, v84, s4                          // 000000003E3C: D5010054 0012A880
	v_or_b32_e32 v86, v82, v4                                  // 000000003E44: 38AC0952
	v_add_co_ci_u32_e64 v82, null, s35, v78, s6                // 000000003E48: D5207C52 001A9C23
	v_add_co_u32 v75, s6, s38, v75                             // 000000003E50: D700064B 02029626
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000003E58: BF870194
	v_lshlrev_b64_e32 v[79:80], 1, v[84:85]                    // 000000003E5C: 3E9EA881
	v_lshlrev_b64_e32 v[81:82], 6, v[81:82]                    // 000000003E60: 3EA2A286
	v_or_b32_e32 v74, s41, v74                                 // 000000003E64: 38949429
	s_wait_alu depctr_va_sdst(0)                               // 000000003E68: BF88F19F
	v_add_co_ci_u32_e64 v76, null, s39, v76, s6                // 000000003E6C: D5207C4C 001A9827
	v_cndmask_b32_e64 v85, 0, v86, s5                          // 000000003E74: D5010055 0016AC80
	v_add_co_u32 v79, s6, s38, v79                             // 000000003E7C: D700064F 02029E26
	s_wait_alu depctr_va_sdst(0)                               // 000000003E84: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s39, v80, s6                // 000000003E88: D5207C50 001AA027
	v_or_b32_e32 v88, v81, v3                                  // 000000003E90: 38B00751
	v_add_co_u32 v81, s6, v73, s34                             // 000000003E94: D7000651 02004549
	v_or_b32_e32 v86, v82, v4                                  // 000000003E9C: 38AC0952
	s_wait_alu depctr_va_sdst(0)                               // 000000003EA0: BF88F19F
	v_add_co_ci_u32_e64 v82, null, s35, v74, s6                // 000000003EA4: D5207C52 001A9423
	v_cmp_gt_i64_e64 s6, s[22:23], v[77:78]                    // 000000003EAC: D4540006 02029A16
	v_or_b32_e32 v7, s40, v7                                   // 000000003EB4: 380E0E28
	v_or_b32_e32 v8, s41, v8                                   // 000000003EB8: 38101029
	v_cndmask_b32_e64 v84, 0, v87, s5                          // 000000003EBC: D5010054 0016AE80
	v_lshlrev_b64_e32 v[81:82], 6, v[81:82]                    // 000000003EC4: 3EA2A286
	v_or_b32_e32 v5, s40, v5                                   // 000000003EC8: 380A0A28
	s_wait_alu depctr_va_sdst(0)                               // 000000003ECC: BF88F19F
	v_cndmask_b32_e64 v78, 0, v86, s6                          // 000000003ED0: D501004E 001AAC80
	v_add_co_u32 v86, s7, v7, s34                              // 000000003ED8: D7000756 02004507
	s_wait_alu depctr_va_sdst(0)                               // 000000003EE0: BF88F19F
	v_add_co_ci_u32_e64 v87, null, s35, v8, s7                 // 000000003EE4: D5207C57 001E1023
	v_lshlrev_b64_e32 v[84:85], 1, v[84:85]                    // 000000003EEC: 3EA8A881
	v_cmp_gt_i64_e64 s7, s[22:23], v[73:74]                    // 000000003EF0: D4540007 02029216
	v_cndmask_b32_e64 v77, 0, v88, s6                          // 000000003EF8: D501004D 001AB080
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003F00: BF870004
	v_lshlrev_b64_e32 v[73:74], 6, v[86:87]                    // 000000003F04: 3E92AC86
	v_or_b32_e32 v88, v82, v4                                  // 000000003F08: 38B00952
	v_or_b32_e32 v6, s41, v6                                   // 000000003F0C: 380C0C29
	v_or_b32_e32 v89, v81, v3                                  // 000000003F10: 38B20751
	v_add_co_u32 v81, s8, s38, v84                             // 000000003F14: D7000851 0202A826
	s_wait_alu depctr_va_sdst(0)                               // 000000003F1C: BF88F19F
	v_add_co_ci_u32_e64 v82, null, s39, v85, s8                // 000000003F20: D5207C52 0022AA27
	v_cndmask_b32_e64 v85, 0, v88, s7                          // 000000003F28: D5010055 001EB080
	v_or_b32_e32 v88, v73, v3                                  // 000000003F30: 38B00749
	v_add_co_u32 v73, s8, v5, s34                              // 000000003F34: D7000849 02004505
	v_or_b32_e32 v86, v74, v4                                  // 000000003F3C: 38AC094A
	s_wait_alu depctr_va_sdst(0)                               // 000000003F40: BF88F19F
	v_add_co_ci_u32_e64 v74, null, s35, v6, s8                 // 000000003F44: D5207C4A 00220C23
	v_cmp_gt_i64_e64 s8, s[22:23], v[7:8]                      // 000000003F4C: D4540008 02020E16
	v_or_b32_e32 v1, s40, v1                                   // 000000003F54: 38020228
	v_or_b32_e32 v2, s41, v2                                   // 000000003F58: 38040429
	v_lshlrev_b64_e32 v[77:78], 1, v[77:78]                    // 000000003F5C: 3E9A9A81
	v_lshlrev_b64_e32 v[73:74], 6, v[73:74]                    // 000000003F60: 3E929286
	v_cndmask_b32_e64 v84, 0, v89, s7                          // 000000003F64: D5010054 001EB280
	s_wait_alu depctr_va_sdst(0)                               // 000000003F6C: BF88F19F
	v_cndmask_b32_e64 v8, 0, v86, s8                           // 000000003F70: D5010008 0022AC80
	v_add_co_u32 v86, s9, v1, s34                              // 000000003F78: D7000956 02004501
	s_wait_alu depctr_va_sdst(0)                               // 000000003F80: BF88F19F
	v_add_co_ci_u32_e64 v87, null, s35, v2, s9                 // 000000003F84: D5207C57 00260423
	v_cmp_gt_i64_e64 s9, s[22:23], v[5:6]                      // 000000003F8C: D4540009 02020A16
	v_add_co_u32 v77, s10, s38, v77                            // 000000003F94: D7000A4D 02029A26
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003F9C: BF870003
	v_lshlrev_b64_e32 v[5:6], 6, v[86:87]                      // 000000003FA0: 3E0AAC86
	v_cndmask_b32_e64 v7, 0, v88, s8                           // 000000003FA4: D5010007 0022B080
	v_or_b32_e32 v88, v74, v4                                  // 000000003FAC: 38B0094A
	v_or_b32_e32 v89, v73, v3                                  // 000000003FB0: 38B20749
	s_wait_alu depctr_va_sdst(0)                               // 000000003FB4: BF88F19F
	v_add_co_ci_u32_e64 v78, null, s39, v78, s10               // 000000003FB8: D5207C4E 002A9C27
	v_cmp_gt_i64_e64 s10, s[22:23], v[1:2]                     // 000000003FC0: D454000A 02020216
	v_or_b32_e32 v6, v6, v4                                    // 000000003FC8: 380C0906
	v_or_b32_e32 v5, v5, v3                                    // 000000003FCC: 380A0705
	v_lshlrev_b64_e32 v[73:74], 1, v[84:85]                    // 000000003FD0: 3E92A881
	v_cndmask_b32_e64 v85, 0, v88, s9                          // 000000003FD4: D5010055 0026B080
	v_cndmask_b32_e64 v84, 0, v89, s9                          // 000000003FDC: D5010054 0026B280
	s_wait_alu depctr_va_sdst(0)                               // 000000003FE4: BF88F19F
	v_cndmask_b32_e64 v6, 0, v6, s10                           // 000000003FE8: D5010006 002A0C80
	v_cndmask_b32_e64 v5, 0, v5, s10                           // 000000003FF0: D5010005 002A0A80
	v_lshlrev_b64_e32 v[7:8], 1, v[7:8]                        // 000000003FF8: 3E0E0E81
	v_add_co_u32 v1, s11, s38, v73                             // 000000003FFC: D7000B01 02029226
	v_lshlrev_b64_e32 v[3:4], 1, v[84:85]                      // 000000004004: 3E06A881
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004008: BF870004
	v_lshlrev_b64_e32 v[5:6], 1, v[5:6]                        // 00000000400C: 3E0A0A81
	s_wait_alu depctr_va_sdst(0)                               // 000000004010: BF88F19F
	v_add_co_ci_u32_e64 v2, null, s39, v74, s11                // 000000004014: D5207C02 002E9427
	v_add_co_u32 v7, s11, s38, v7                              // 00000000401C: D7000B07 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000004024: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s11                 // 000000004028: D5207C08 002E1027
	v_add_co_u32 v3, s11, s38, v3                              // 000000004030: D7000B03 02020626
	s_wait_alu depctr_va_sdst(0)                               // 000000004038: BF88F19F
	v_add_co_ci_u32_e64 v4, null, s39, v4, s11                 // 00000000403C: D5207C04 002E0827
	v_add_co_u32 v5, s11, s38, v5                              // 000000004044: D7000B05 02020A26
	s_wait_alu depctr_va_sdst(0)                               // 00000000404C: BF88F19F
	v_add_co_ci_u32_e64 v6, null, s39, v6, s11                 // 000000004050: D5207C06 002E0C27
	s_clause 0x7                                               // 000000004058: BF850007
	global_load_d16_b16 v73, v[75:76], off                     // 00000000405C: EE08007C 00000049 0000004B
	global_load_d16_hi_b16 v73, v[79:80], off                  // 000000004068: EE08C07C 00000049 0000004F
	global_load_d16_b16 v74, v[81:82], off                     // 000000004074: EE08007C 0000004A 00000051
	global_load_d16_hi_b16 v74, v[77:78], off                  // 000000004080: EE08C07C 0000004A 0000004D
	global_load_d16_b16 v75, v[1:2], off                       // 00000000408C: EE08007C 0000004B 00000001
	global_load_d16_hi_b16 v75, v[7:8], off                    // 000000004098: EE08C07C 0000004B 00000007
	global_load_d16_b16 v76, v[3:4], off                       // 0000000040A4: EE08007C 0000004C 00000003
	global_load_d16_hi_b16 v76, v[5:6], off                    // 0000000040B0: EE08C07C 0000004C 00000005
	ds_load_b128 v[1:4], v117                                  // 0000000040BC: DBFC0000 01000075
	ds_load_b32 v77, v116                                      // 0000000040C4: D8D80000 4D000074
	ds_load_b32 v78, v118                                      // 0000000040CC: D8D80000 4E000076
	ds_load_b32 v79, v119                                      // 0000000040D4: D8D80000 4F000077
	ds_load_b32 v80, v120                                      // 0000000040DC: D8D80000 50000078
	ds_load_b128 v[5:8], v117 offset:16                        // 0000000040E4: DBFC0010 05000075
	ds_load_b32 v85, v121                                      // 0000000040EC: D8D80000 55000079
	ds_load_b32 v86, v122                                      // 0000000040F4: D8D80000 5600007A
	ds_load_b32 v87, v123                                      // 0000000040FC: D8D80000 5700007B
	ds_load_b32 v88, v125                                      // 000000004104: D8D80000 5800007D
	s_wait_dscnt 0xb                                           // 00000000410C: BFC6000B
	v_cvt_f16_f32_e32 v12.l, v12                               // 000000004110: 7E18150C
	s_wait_dscnt 0xa                                           // 000000004114: BFC6000A
	v_cvt_f16_f32_e32 v12.h, v83                               // 000000004118: 7F181553
	s_wait_dscnt 0x7                                           // 00000000411C: BFC60007
	v_dual_mul_f32 v89, v77, v1 :: v_dual_mul_f32 v90, v78, v2 // 000000004120: C8C6034D 595A054E
	s_wait_dscnt 0x5                                           // 000000004128: BFC60005
	v_dual_mul_f32 v91, v79, v3 :: v_dual_mul_f32 v92, v80, v4 // 00000000412C: C8C6074F 5B5C0950
	s_wait_loadcnt 0x6                                         // 000000004134: BFC00006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004138: BF88FF9E
	v_cndmask_b16 v81.l, 0, v73.l, s12                         // 00000000413C: D65D0051 00329280
	v_cndmask_b16 v81.h, 0, v73.h, s4                          // 000000004144: D65D5051 00129280
	s_wait_loadcnt 0x4                                         // 00000000414C: BFC00004
	v_cndmask_b16 v82.l, 0, v74.l, s5                          // 000000004150: D65D0052 00169480
	v_cndmask_b16 v82.h, 0, v74.h, s6                          // 000000004158: D65D5052 001A9480
	s_wait_loadcnt 0x2                                         // 000000004160: BFC00002
	v_cndmask_b16 v83.l, 0, v75.l, s7                          // 000000004164: D65D0053 001E9680
	v_cndmask_b16 v83.h, 0, v75.h, s8                          // 00000000416C: D65D5053 00229680
	s_wait_loadcnt 0x0                                         // 000000004174: BFC00000
	v_cndmask_b16 v84.l, 0, v76.l, s9                          // 000000004178: D65D0054 00269880
	v_cndmask_b16 v84.h, 0, v76.h, s10                         // 000000004180: D65D5054 002A9880
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004188: BF870001
	v_wmma_f32_16x16x16_f16 v[73:80], v[9:12], v[81:84], 0     // 00000000418C: CC404049 1A02A309
	s_wait_dscnt 0x2                                           // 000000004194: BFC60002
	v_dual_mul_f32 v9, v85, v5 :: v_dual_mul_f32 v10, v86, v6  // 000000004198: C8C60B55 090A0D56
	s_wait_dscnt 0x0                                           // 0000000041A0: BFC60000
	v_dual_mul_f32 v11, v87, v7 :: v_dual_mul_f32 v12, v88, v8 // 0000000041A4: C8C60F57 0B0C1158
	v_dual_add_f32 v73, v73, v89 :: v_dual_add_f32 v74, v74, v90// 0000000041AC: C908B349 494AB54A
	v_dual_add_f32 v75, v75, v91 :: v_dual_add_f32 v76, v76, v92// 0000000041B4: C908B74B 4B4CB94C
	v_dual_add_f32 v9, v77, v9 :: v_dual_add_f32 v10, v78, v10 // 0000000041BC: C908134D 090A154E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000041C4: BF870004
	v_dual_add_f32 v11, v79, v11 :: v_dual_add_f32 v12, v80, v12// 0000000041C8: C908174F 0B0C1950
	ds_store_b32 v118, v74                                     // 0000000041D0: D8340000 00004A76
	ds_store_b32 v119, v75                                     // 0000000041D8: D8340000 00004B77
	ds_store_b32 v120, v76                                     // 0000000041E0: D8340000 00004C78
	ds_store_b32 v121, v9                                      // 0000000041E8: D8340000 00000979
	ds_store_b32 v122, v10                                     // 0000000041F0: D8340000 00000A7A
	ds_store_b32 v123, v11                                     // 0000000041F8: D8340000 00000B7B
	ds_store_b32 v116, v73                                     // 000000004200: D8340000 00004974
	ds_store_b32 v125, v12                                     // 000000004208: D8340000 00000C7D
	s_and_saveexec_b32 s4, s3                                  // 000000004210: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004214: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000004218: 8D04047E
	s_cbranch_execz 12                                         // 00000000421C: BFA5000C <attention_forward+0x2750>
	ds_load_b128 v[9:12], v111 offset:32                       // 000000004220: DBFC0020 0900006F
	ds_load_b64 v[73:74], v111 offset:48                       // 000000004228: D9D80030 4900006F
	s_wait_dscnt 0x1                                           // 000000004230: BFC60001
	v_cvt_f16_f32_e32 v9.l, v9                                 // 000000004234: 7E121509
	v_cvt_f16_f32_e32 v9.h, v10                                // 000000004238: 7F12150A
	v_cvt_f16_f32_e32 v10.l, v11                               // 00000000423C: 7E14150B
	v_cvt_f16_f32_e32 v10.h, v12                               // 000000004240: 7F14150C
	s_wait_dscnt 0x0                                           // 000000004244: BFC60000
	v_cvt_f16_f32_e32 v11.l, v73                               // 000000004248: 7E161549
	v_cvt_f16_f32_e32 v11.h, v74                               // 00000000424C: 7F16154A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004250: BF88FF9E
	s_or_saveexec_b32 s4, s4                                   // 000000004254: BE842204
	v_dual_mov_b32 v12, v112 :: v_dual_mov_b32 v73, v113       // 000000004258: CA100170 0C480171
	s_wait_alu depctr_sa_sdst(0)                               // 000000004260: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s4                             // 000000004264: 8D7E047E
	s_cbranch_execz 14                                         // 000000004268: BFA5000E <attention_forward+0x27a4>
	ds_load_b128 v[73:76], v111                                // 00000000426C: DBFC0000 4900006F
	ds_load_b64 v[77:78], v111 offset:16                       // 000000004274: D9D80010 4D00006F
	v_mov_b32_e32 v12, v114                                    // 00000000427C: 7E180372
	s_wait_dscnt 0x1                                           // 000000004280: BFC60001
	v_cvt_f16_f32_e32 v9.l, v73                                // 000000004284: 7E121549
	v_cvt_f16_f32_e32 v9.h, v74                                // 000000004288: 7F12154A
	v_cvt_f16_f32_e32 v10.l, v75                               // 00000000428C: 7E14154B
	v_cvt_f16_f32_e32 v10.h, v76                               // 000000004290: 7F14154C
	s_wait_dscnt 0x0                                           // 000000004294: BFC60000
	v_cvt_f16_f32_e32 v11.l, v77                               // 000000004298: 7E16154D
	v_cvt_f16_f32_e32 v11.h, v78                               // 00000000429C: 7F16154E
	v_mov_b32_e32 v73, v115                                    // 0000000042A0: 7E920373
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000042A4: 8C7E047E
	ds_load_b32 v12, v12                                       // 0000000042A8: D8D80000 0C00000C
	ds_load_b32 v89, v73                                       // 0000000042B0: D8D80000 59000049
	s_and_saveexec_b32 s4, s3                                  // 0000000042B8: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042BC: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 0000000042C0: 8D04047E
	s_or_b32 s6, s40, 8                                        // 0000000042C4: 8C068828
	s_mov_b32 s7, s41                                          // 0000000042C8: BE870029
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042CC: BF88FF9E
	s_add_nc_u64 s[8:9], s[6:7], s[34:35]                      // 0000000042D0: A9882206
	v_cmp_lt_i64_e64 s12, s[6:7], s[22:23]                     // 0000000042D4: D451000C 02002C06
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042DC: BF88FF9E
	s_lshl_b64 s[8:9], s[8:9], 6                               // 0000000042E0: 84888608
	s_wait_alu depctr_sa_sdst(0)                               // 0000000042E4: BF88FF9E
	v_or_b32_e32 v73, s8, v124                                 // 0000000042E8: 3892F808
	v_cndmask_b32_e64 v84, 0, s9, s12                          // 0000000042EC: D5010054 00301280
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000042F4: BF870002
	v_cndmask_b32_e64 v83, 0, v73, s12                         // 0000000042F8: D5010053 00329280
	s_or_saveexec_b32 s5, s4                                   // 000000004300: BE852204
	v_dual_mov_b32 v73, 15 :: v_dual_mov_b32 v74, 0            // 000000004304: CA10008F 494A0080
	v_dual_mov_b32 v75, 14 :: v_dual_mov_b32 v76, 0            // 00000000430C: CA10008E 4B4C0080
	v_dual_mov_b32 v77, 13 :: v_dual_mov_b32 v78, 0            // 000000004314: CA10008D 4D4E0080
	v_dual_mov_b32 v79, 12 :: v_dual_mov_b32 v80, 0            // 00000000431C: CA10008C 4F500080
	v_dual_mov_b32 v81, 11 :: v_dual_mov_b32 v82, 0            // 000000004324: CA10008B 51520080
	v_dual_mov_b32 v85, 10 :: v_dual_mov_b32 v86, 0            // 00000000432C: CA10008A 55560080
	v_dual_mov_b32 v87, 9 :: v_dual_mov_b32 v88, 0             // 000000004334: CA100089 57580080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000433C: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 000000004340: 8D7E057E
	s_cbranch_execz 33                                         // 000000004344: BFA50021 <attention_forward+0x28cc>
	s_lshl_b64 s[6:7], s[30:31], 10                            // 000000004348: 84868A1E
	v_dual_mov_b32 v73, 7 :: v_dual_mov_b32 v74, 0             // 00000000434C: CA100087 494A0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004354: BF88FF9E
	v_add_co_u32 v75, s4, s6, v129                             // 000000004358: D700044B 02030206
	s_wait_alu depctr_va_sdst(0)                               // 000000004360: BF88F19F
	v_add_co_ci_u32_e64 v76, null, s7, v127, s4                // 000000004364: D5207C4C 0012FE07
	v_cmp_lt_i64_e64 s4, s[40:41], s[22:23]                    // 00000000436C: D4510004 02002C28
	v_dual_mov_b32 v77, 5 :: v_dual_mov_b32 v78, 0             // 000000004374: CA100085 4D4E0080
	v_dual_mov_b32 v79, 4 :: v_dual_mov_b32 v80, 0             // 00000000437C: CA100084 4F500080
	v_dual_mov_b32 v81, 3 :: v_dual_mov_b32 v82, 0             // 000000004384: CA100083 51520080
	s_wait_alu depctr_va_sdst(0)                               // 00000000438C: BF88F19F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004390: BF870004
	v_cndmask_b32_e64 v84, 0, v76, s4                          // 000000004394: D5010054 00129880
	v_cndmask_b32_e64 v83, 0, v75, s4                          // 00000000439C: D5010053 00129680
	v_dual_mov_b32 v75, 6 :: v_dual_mov_b32 v76, 0             // 0000000043A4: CA100086 4B4C0080
	v_dual_mov_b32 v85, 2 :: v_dual_mov_b32 v86, 0             // 0000000043AC: CA100082 55560080
	v_dual_mov_b32 v87, 1 :: v_dual_mov_b32 v88, 0             // 0000000043B4: CA100081 57580080
	s_and_not1_b32 s6, s12, exec_lo                            // 0000000043BC: 91067E0C
	s_and_b32 s4, s4, exec_lo                                  // 0000000043C0: 8B047E04
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043C4: BF88FF9E
	s_or_b32 s12, s6, s4                                       // 0000000043C8: 8C0C0406
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000043CC: 8C7E057E
	v_or_b32_e32 v87, s40, v87                                 // 0000000043D0: 38AEAE28
	v_or_b32_e32 v88, s41, v88                                 // 0000000043D4: 38B0B029
	v_or_b32_e32 v85, s40, v85                                 // 0000000043D8: 38AAAA28
	v_or_b32_e32 v86, s41, v86                                 // 0000000043DC: 38ACAC29
	v_or_b32_e32 v81, s40, v81                                 // 0000000043E0: 38A2A228
	v_add_co_u32 v90, s4, v87, s34                             // 0000000043E4: D700045A 02004557
	s_wait_alu depctr_va_sdst(0)                               // 0000000043EC: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v88, s4                // 0000000043F0: D5207C5B 0012B023
	v_add_co_u32 v92, s4, v85, s34                             // 0000000043F8: D700045C 02004555
	s_wait_alu depctr_va_sdst(0)                               // 000000004400: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v86, s4                // 000000004404: D5207C5D 0012AC23
	v_or_b32_e32 v82, s41, v82                                 // 00000000440C: 38A4A429
	v_cmp_gt_i64_e64 s4, s[22:23], v[87:88]                    // 000000004410: D4540004 0202AE16
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 000000004418: 3EB4B486
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000441C: BF870004
	v_lshlrev_b64_e32 v[87:88], 6, v[92:93]                    // 000000004420: 3EAEB886
	v_add_co_u32 v92, s5, v81, s34                             // 000000004424: D700055C 02004551
	v_lshlrev_b64_e32 v[83:84], 1, v[83:84]                    // 00000000442C: 3EA6A681
	s_wait_alu depctr_va_sdst(0)                               // 000000004430: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v82, s5                // 000000004434: D5207C5D 0016A423
	v_or_b32_e32 v90, v90, v124                                // 00000000443C: 38B4F95A
	v_cndmask_b32_e64 v91, 0, v91, s4                          // 000000004440: D501005B 0012B680
	v_cmp_gt_i64_e64 s6, s[22:23], v[81:82]                    // 000000004448: D4540006 0202A216
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004450: BF870004
	v_lshlrev_b64_e32 v[92:93], 6, v[92:93]                    // 000000004454: 3EB8B886
	v_add_co_u32 v83, s5, s38, v83                             // 000000004458: D7000553 0202A626
	s_wait_alu depctr_va_sdst(0)                               // 000000004460: BF88F19F
	v_add_co_ci_u32_e64 v84, null, s39, v84, s5                // 000000004464: D5207C54 0016A827
	v_cmp_gt_i64_e64 s5, s[22:23], v[85:86]                    // 00000000446C: D4540005 0202AA16
	v_or_b32_e32 v85, v87, v124                                // 000000004474: 38AAF957
	v_cndmask_b32_e64 v90, 0, v90, s4                          // 000000004478: D501005A 0012B480
	v_or_b32_e32 v87, v92, v124                                // 000000004480: 38AEF95C
	v_or_b32_e32 v79, s40, v79                                 // 000000004484: 389E9E28
	v_or_b32_e32 v80, s41, v80                                 // 000000004488: 38A0A029
	s_wait_alu depctr_va_sdst(0)                               // 00000000448C: BF88F19F
	v_cndmask_b32_e64 v86, 0, v88, s5                          // 000000004490: D5010056 0016B080
	v_cndmask_b32_e64 v85, 0, v85, s5                          // 000000004498: D5010055 0016AA80
	v_lshlrev_b64_e32 v[81:82], 1, v[90:91]                    // 0000000044A0: 3EA2B481
	v_cndmask_b32_e64 v88, 0, v93, s6                          // 0000000044A4: D5010058 001ABA80
	v_cndmask_b32_e64 v87, 0, v87, s6                          // 0000000044AC: D5010057 001AAE80
	v_or_b32_e32 v77, s40, v77                                 // 0000000044B4: 389A9A28
	v_lshlrev_b64_e32 v[85:86], 1, v[85:86]                    // 0000000044B8: 3EAAAA81
	v_or_b32_e32 v78, s41, v78                                 // 0000000044BC: 389C9C29
	v_add_co_u32 v81, s7, s38, v81                             // 0000000044C0: D7000751 0202A226
	v_lshlrev_b64_e32 v[87:88], 1, v[87:88]                    // 0000000044C8: 3EAEAE81
	s_wait_alu depctr_va_sdst(0)                               // 0000000044CC: BF88F19F
	v_add_co_ci_u32_e64 v82, null, s39, v82, s7                // 0000000044D0: D5207C52 001EA427
	v_add_co_u32 v90, s7, v79, s34                             // 0000000044D8: D700075A 0200454F
	s_wait_alu depctr_va_sdst(0)                               // 0000000044E0: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v80, s7                // 0000000044E4: D5207C5B 001EA023
	v_add_co_u32 v85, s7, s38, v85                             // 0000000044EC: D7000755 0202AA26
	v_or_b32_e32 v75, s40, v75                                 // 0000000044F4: 38969628
	s_wait_alu depctr_va_sdst(0)                               // 0000000044F8: BF88F19F
	v_add_co_ci_u32_e64 v86, null, s39, v86, s7                // 0000000044FC: D5207C56 001EAC27
	v_add_co_u32 v87, s7, s38, v87                             // 000000004504: D7000757 0202AE26
	v_or_b32_e32 v76, s41, v76                                 // 00000000450C: 38989829
	s_wait_alu depctr_va_sdst(0)                               // 000000004510: BF88F19F
	v_add_co_ci_u32_e64 v88, null, s39, v88, s7                // 000000004514: D5207C58 001EB027
	v_cmp_gt_i64_e64 s7, s[22:23], v[79:80]                    // 00000000451C: D4540007 02029E16
	v_add_co_u32 v79, s8, v77, s34                             // 000000004524: D700084F 0200454D
	s_wait_alu depctr_va_sdst(0)                               // 00000000452C: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s35, v78, s8                // 000000004530: D5207C50 00229C23
	v_add_co_u32 v92, s8, v75, s34                             // 000000004538: D700085C 0200454B
	s_wait_alu depctr_va_sdst(0)                               // 000000004540: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v76, s8                // 000000004544: D5207C5D 00229823
	v_or_b32_e32 v73, s40, v73                                 // 00000000454C: 38929228
	v_or_b32_e32 v74, s41, v74                                 // 000000004550: 38949429
	v_cmp_gt_i64_e64 s8, s[22:23], v[77:78]                    // 000000004554: D4540008 02029A16
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000455C: BF870004
	v_lshlrev_b64_e32 v[77:78], 6, v[92:93]                    // 000000004560: 3E9AB886
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 000000004564: 3EB4B486
	v_add_co_u32 v92, s9, v73, s34                             // 000000004568: D700095C 02004549
	s_wait_alu depctr_va_sdst(0)                               // 000000004570: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v74, s9                // 000000004574: D5207C5D 00269423
	v_lshlrev_b64_e32 v[79:80], 6, v[79:80]                    // 00000000457C: 3E9E9E86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000004580: BF870224
	v_or_b32_e32 v90, v90, v124                                // 000000004584: 38B4F95A
	v_cmp_gt_i64_e64 s9, s[22:23], v[75:76]                    // 000000004588: D4540009 02029616
	v_lshlrev_b64_e32 v[75:76], 6, v[92:93]                    // 000000004590: 3E96B886
	v_or_b32_e32 v77, v77, v124                                // 000000004594: 389AF94D
	v_cndmask_b32_e64 v91, 0, v91, s7                          // 000000004598: D501005B 001EB680
	v_or_b32_e32 v79, v79, v124                                // 0000000045A0: 389EF94F
	v_cndmask_b32_e64 v90, 0, v90, s7                          // 0000000045A4: D501005A 001EB480
	v_cmp_gt_i64_e64 s10, s[22:23], v[73:74]                   // 0000000045AC: D454000A 02029216
	v_or_b32_e32 v75, v75, v124                                // 0000000045B4: 3896F94B
	v_cndmask_b32_e64 v80, 0, v80, s8                          // 0000000045B8: D5010050 0022A080
	v_cndmask_b32_e64 v79, 0, v79, s8                          // 0000000045C0: D501004F 00229E80
	s_wait_alu depctr_va_sdst(0)                               // 0000000045C8: BF88F19F
	v_cndmask_b32_e64 v78, 0, v78, s9                          // 0000000045CC: D501004E 00269C80
	v_cndmask_b32_e64 v77, 0, v77, s9                          // 0000000045D4: D501004D 00269A80
	v_lshlrev_b64_e32 v[90:91], 1, v[90:91]                    // 0000000045DC: 3EB4B481
	v_cndmask_b32_e64 v76, 0, v76, s10                         // 0000000045E0: D501004C 002A9880
	v_cndmask_b32_e64 v75, 0, v75, s10                         // 0000000045E8: D501004B 002A9680
	v_lshlrev_b64_e32 v[79:80], 1, v[79:80]                    // 0000000045F0: 3E9E9E81
	v_lshlrev_b64_e32 v[73:74], 1, v[77:78]                    // 0000000045F4: 3E929A81
	s_wait_dscnt 0x1                                           // 0000000045F8: BFC60001
	v_cvt_f16_f32_e32 v12.l, v12                               // 0000000045FC: 7E18150C
	v_add_co_u32 v90, s11, s38, v90                            // 000000004600: D7000B5A 0202B426
	v_lshlrev_b64_e32 v[75:76], 1, v[75:76]                    // 000000004608: 3E969681
	s_wait_alu depctr_va_sdst(0)                               // 00000000460C: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s39, v91, s11               // 000000004610: D5207C5B 002EB627
	v_add_co_u32 v77, s11, s38, v79                            // 000000004618: D7000B4D 02029E26
	s_wait_alu depctr_va_sdst(0)                               // 000000004620: BF88F19F
	v_add_co_ci_u32_e64 v78, null, s39, v80, s11               // 000000004624: D5207C4E 002EA027
	v_add_co_u32 v79, s11, s38, v73                            // 00000000462C: D7000B4F 02029226
	s_wait_alu depctr_va_sdst(0)                               // 000000004634: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s39, v74, s11               // 000000004638: D5207C50 002E9427
	v_add_co_u32 v92, s11, s38, v75                            // 000000004640: D7000B5C 02029626
	s_wait_alu depctr_va_sdst(0)                               // 000000004648: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s39, v76, s11               // 00000000464C: D5207C5D 002E9827
	s_clause 0x7                                               // 000000004654: BF850007
	global_load_d16_b16 v73, v[83:84], off                     // 000000004658: EE08007C 00000049 00000053
	global_load_d16_hi_b16 v73, v[81:82], off                  // 000000004664: EE08C07C 00000049 00000051
	global_load_d16_b16 v74, v[85:86], off                     // 000000004670: EE08007C 0000004A 00000055
	global_load_d16_hi_b16 v74, v[87:88], off                  // 00000000467C: EE08C07C 0000004A 00000057
	global_load_d16_b16 v75, v[90:91], off                     // 000000004688: EE08007C 0000004B 0000005A
	global_load_d16_hi_b16 v75, v[77:78], off                  // 000000004694: EE08C07C 0000004B 0000004D
	global_load_d16_b16 v76, v[79:80], off                     // 0000000046A0: EE08007C 0000004C 0000004F
	global_load_d16_hi_b16 v76, v[92:93], off                  // 0000000046AC: EE08C07C 0000004C 0000005C
	v_add_nc_u32_e32 v90, 64, v116                             // 0000000046B8: 4AB4E8C0
	ds_load_2addr_b32 v[77:78], v116 offset0:16 offset1:80     // 0000000046BC: D8DC5010 4D000074
	ds_load_2addr_b32 v[79:80], v116 offset0:144 offset1:208   // 0000000046C4: D8DCD090 4F000074
	ds_load_2addr_stride64_b32 v[85:86], v90 offset0:4 offset1:5// 0000000046CC: D8E00504 5500005A
	ds_load_2addr_stride64_b32 v[87:88], v90 offset0:6 offset1:7// 0000000046D4: D8E00706 5700005A
	s_wait_dscnt 0x4                                           // 0000000046DC: BFC60004
	v_cvt_f16_f32_e32 v12.h, v89                               // 0000000046E0: 7F181559
	s_wait_dscnt 0x2                                           // 0000000046E4: BFC60002
	v_dual_mul_f32 v89, v1, v77 :: v_dual_mul_f32 v92, v3, v79 // 0000000046E8: C8C69B01 595C9F03
	v_mul_f32_e32 v91, v2, v78                                 // 0000000046F0: 10B69D02
	v_mul_f32_e32 v93, v4, v80                                 // 0000000046F4: 10BAA104
	s_wait_loadcnt 0x6                                         // 0000000046F8: BFC00006
	s_wait_alu depctr_sa_sdst(0)                               // 0000000046FC: BF88FF9E
	v_cndmask_b16 v81.l, 0, v73.l, s12                         // 000000004700: D65D0051 00329280
	v_cndmask_b16 v81.h, 0, v73.h, s4                          // 000000004708: D65D5051 00129280
	s_wait_loadcnt 0x4                                         // 000000004710: BFC00004
	v_cndmask_b16 v82.l, 0, v74.l, s5                          // 000000004714: D65D0052 00169480
	v_cndmask_b16 v82.h, 0, v74.h, s6                          // 00000000471C: D65D5052 001A9480
	s_wait_loadcnt 0x2                                         // 000000004724: BFC00002
	v_cndmask_b16 v83.l, 0, v75.l, s7                          // 000000004728: D65D0053 001E9680
	v_cndmask_b16 v83.h, 0, v75.h, s8                          // 000000004730: D65D5053 00229680
	s_wait_loadcnt 0x0                                         // 000000004738: BFC00000
	v_cndmask_b16 v84.l, 0, v76.l, s9                          // 00000000473C: D65D0054 00269880
	v_cndmask_b16 v84.h, 0, v76.h, s10                         // 000000004744: D65D5054 002A9880
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000474C: BF870001
	v_wmma_f32_16x16x16_f16 v[73:80], v[9:12], v[81:84], 0     // 000000004750: CC404049 1A02A309
	s_wait_dscnt 0x1                                           // 000000004758: BFC60001
	v_mul_f32_e32 v10, v6, v86                                 // 00000000475C: 1014AD06
	s_wait_dscnt 0x0                                           // 000000004760: BFC60000
	v_dual_mul_f32 v12, v8, v88 :: v_dual_mul_f32 v9, v5, v85  // 000000004764: C8C6B108 0C08AB05
	v_add_f32_e32 v74, v74, v91                                // 00000000476C: 0694B74A
	v_dual_mul_f32 v11, v7, v87 :: v_dual_add_f32 v76, v76, v93// 000000004770: C8C8AF07 0B4CBB4C
	v_dual_add_f32 v73, v73, v89 :: v_dual_add_f32 v10, v78, v10// 000000004778: C908B349 490A154E
	v_add_f32_e32 v75, v75, v92                                // 000000004780: 0696B94B
	v_dual_add_f32 v9, v77, v9 :: v_dual_add_f32 v12, v80, v12 // 000000004784: C908134D 090C1950
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000478C: BF870004
	v_add_f32_e32 v11, v79, v11                                // 000000004790: 0616174F
	ds_store_2addr_b32 v116, v73, v74 offset0:16 offset1:80    // 000000004794: D8385010 004A4974
	ds_store_2addr_b32 v116, v75, v76 offset0:144 offset1:208  // 00000000479C: D838D090 004C4B74
	ds_store_2addr_stride64_b32 v90, v9, v10 offset0:4 offset1:5// 0000000047A4: D83C0504 000A095A
	ds_store_2addr_stride64_b32 v90, v11, v12 offset0:6 offset1:7// 0000000047AC: D83C0706 000C0B5A
	s_and_saveexec_b32 s4, s3                                  // 0000000047B4: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047B8: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 0000000047BC: 8D04047E
	s_cbranch_execz 12                                         // 0000000047C0: BFA5000C <attention_forward+0x2cf4>
	ds_load_b128 v[9:12], v111 offset:32                       // 0000000047C4: DBFC0020 0900006F
	ds_load_b64 v[73:74], v111 offset:48                       // 0000000047CC: D9D80030 4900006F
	s_wait_dscnt 0x1                                           // 0000000047D4: BFC60001
	v_cvt_f16_f32_e32 v9.l, v9                                 // 0000000047D8: 7E121509
	v_cvt_f16_f32_e32 v9.h, v10                                // 0000000047DC: 7F12150A
	v_cvt_f16_f32_e32 v10.l, v11                               // 0000000047E0: 7E14150B
	v_cvt_f16_f32_e32 v10.h, v12                               // 0000000047E4: 7F14150C
	s_wait_dscnt 0x0                                           // 0000000047E8: BFC60000
	v_cvt_f16_f32_e32 v11.l, v73                               // 0000000047EC: 7E161549
	v_cvt_f16_f32_e32 v11.h, v74                               // 0000000047F0: 7F16154A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047F4: BF88FF9E
	s_or_saveexec_b32 s4, s4                                   // 0000000047F8: BE842204
	v_dual_mov_b32 v12, v112 :: v_dual_mov_b32 v73, v113       // 0000000047FC: CA100170 0C480171
	s_wait_alu depctr_sa_sdst(0)                               // 000000004804: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s4                             // 000000004808: 8D7E047E
	s_cbranch_execz 14                                         // 00000000480C: BFA5000E <attention_forward+0x2d48>
	ds_load_b128 v[73:76], v111                                // 000000004810: DBFC0000 4900006F
	ds_load_b64 v[77:78], v111 offset:16                       // 000000004818: D9D80010 4D00006F
	v_mov_b32_e32 v12, v114                                    // 000000004820: 7E180372
	s_wait_dscnt 0x1                                           // 000000004824: BFC60001
	v_cvt_f16_f32_e32 v9.l, v73                                // 000000004828: 7E121549
	v_cvt_f16_f32_e32 v9.h, v74                                // 00000000482C: 7F12154A
	v_cvt_f16_f32_e32 v10.l, v75                               // 000000004830: 7E14154B
	v_cvt_f16_f32_e32 v10.h, v76                               // 000000004834: 7F14154C
	s_wait_dscnt 0x0                                           // 000000004838: BFC60000
	v_cvt_f16_f32_e32 v11.l, v77                               // 00000000483C: 7E16154D
	v_cvt_f16_f32_e32 v11.h, v78                               // 000000004840: 7F16154E
	v_mov_b32_e32 v73, v115                                    // 000000004844: 7E920373
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004848: 8C7E047E
	ds_load_b32 v12, v12                                       // 00000000484C: D8D80000 0C00000C
	ds_load_b32 v89, v73                                       // 000000004854: D8D80000 59000049
	s_and_saveexec_b32 s4, s3                                  // 00000000485C: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004860: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000004864: 8D04047E
	s_or_b32 s6, s40, 8                                        // 000000004868: 8C068828
	s_mov_b32 s7, s41                                          // 00000000486C: BE870029
	s_wait_alu depctr_sa_sdst(0)                               // 000000004870: BF88FF9E
	s_add_nc_u64 s[8:9], s[6:7], s[34:35]                      // 000000004874: A9882206
	v_cmp_lt_i64_e64 s12, s[6:7], s[22:23]                     // 000000004878: D451000C 02002C06
	s_wait_alu depctr_sa_sdst(0)                               // 000000004880: BF88FF9E
	s_lshl_b64 s[8:9], s[8:9], 6                               // 000000004884: 84888608
	s_wait_alu depctr_sa_sdst(0)                               // 000000004888: BF88FF9E
	v_or_b32_e32 v73, s8, v126                                 // 00000000488C: 3892FC08
	v_cndmask_b32_e64 v84, 0, s9, s12                          // 000000004890: D5010054 00301280
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004898: BF870002
	v_cndmask_b32_e64 v83, 0, v73, s12                         // 00000000489C: D5010053 00329280
	s_or_saveexec_b32 s5, s4                                   // 0000000048A4: BE852204
	v_dual_mov_b32 v73, 15 :: v_dual_mov_b32 v74, 0            // 0000000048A8: CA10008F 494A0080
	v_dual_mov_b32 v75, 14 :: v_dual_mov_b32 v76, 0            // 0000000048B0: CA10008E 4B4C0080
	v_dual_mov_b32 v77, 13 :: v_dual_mov_b32 v78, 0            // 0000000048B8: CA10008D 4D4E0080
	v_dual_mov_b32 v79, 12 :: v_dual_mov_b32 v80, 0            // 0000000048C0: CA10008C 4F500080
	v_dual_mov_b32 v81, 11 :: v_dual_mov_b32 v82, 0            // 0000000048C8: CA10008B 51520080
	v_dual_mov_b32 v85, 10 :: v_dual_mov_b32 v86, 0            // 0000000048D0: CA10008A 55560080
	v_dual_mov_b32 v87, 9 :: v_dual_mov_b32 v88, 0             // 0000000048D8: CA100089 57580080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048E0: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 0000000048E4: 8D7E057E
	s_cbranch_execz 33                                         // 0000000048E8: BFA50021 <attention_forward+0x2e70>
	s_lshl_b64 s[6:7], s[30:31], 10                            // 0000000048EC: 84868A1E
	v_dual_mov_b32 v73, 7 :: v_dual_mov_b32 v74, 0             // 0000000048F0: CA100087 494A0080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000048F8: BF88FF9E
	v_add_co_u32 v75, s4, s6, v131                             // 0000000048FC: D700044B 02030606
	s_wait_alu depctr_va_sdst(0)                               // 000000004904: BF88F19F
	v_add_co_ci_u32_e64 v76, null, s7, v130, s4                // 000000004908: D5207C4C 00130407
	v_cmp_lt_i64_e64 s4, s[40:41], s[22:23]                    // 000000004910: D4510004 02002C28
	v_dual_mov_b32 v77, 5 :: v_dual_mov_b32 v78, 0             // 000000004918: CA100085 4D4E0080
	v_dual_mov_b32 v79, 4 :: v_dual_mov_b32 v80, 0             // 000000004920: CA100084 4F500080
	v_dual_mov_b32 v81, 3 :: v_dual_mov_b32 v82, 0             // 000000004928: CA100083 51520080
	s_wait_alu depctr_va_sdst(0)                               // 000000004930: BF88F19F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004934: BF870004
	v_cndmask_b32_e64 v84, 0, v76, s4                          // 000000004938: D5010054 00129880
	v_cndmask_b32_e64 v83, 0, v75, s4                          // 000000004940: D5010053 00129680
	v_dual_mov_b32 v75, 6 :: v_dual_mov_b32 v76, 0             // 000000004948: CA100086 4B4C0080
	v_dual_mov_b32 v85, 2 :: v_dual_mov_b32 v86, 0             // 000000004950: CA100082 55560080
	v_dual_mov_b32 v87, 1 :: v_dual_mov_b32 v88, 0             // 000000004958: CA100081 57580080
	s_and_not1_b32 s6, s12, exec_lo                            // 000000004960: 91067E0C
	s_and_b32 s4, s4, exec_lo                                  // 000000004964: 8B047E04
	s_wait_alu depctr_sa_sdst(0)                               // 000000004968: BF88FF9E
	s_or_b32 s12, s6, s4                                       // 00000000496C: 8C0C0406
	s_or_b32 exec_lo, exec_lo, s5                              // 000000004970: 8C7E057E
	v_or_b32_e32 v87, s40, v87                                 // 000000004974: 38AEAE28
	v_or_b32_e32 v88, s41, v88                                 // 000000004978: 38B0B029
	v_or_b32_e32 v85, s40, v85                                 // 00000000497C: 38AAAA28
	v_or_b32_e32 v86, s41, v86                                 // 000000004980: 38ACAC29
	v_or_b32_e32 v81, s40, v81                                 // 000000004984: 38A2A228
	v_add_co_u32 v90, s4, v87, s34                             // 000000004988: D700045A 02004557
	s_wait_alu depctr_va_sdst(0)                               // 000000004990: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v88, s4                // 000000004994: D5207C5B 0012B023
	v_add_co_u32 v92, s4, v85, s34                             // 00000000499C: D700045C 02004555
	s_wait_alu depctr_va_sdst(0)                               // 0000000049A4: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v86, s4                // 0000000049A8: D5207C5D 0012AC23
	v_or_b32_e32 v82, s41, v82                                 // 0000000049B0: 38A4A429
	v_cmp_gt_i64_e64 s4, s[22:23], v[87:88]                    // 0000000049B4: D4540004 0202AE16
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 0000000049BC: 3EB4B486
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049C0: BF870004
	v_lshlrev_b64_e32 v[87:88], 6, v[92:93]                    // 0000000049C4: 3EAEB886
	v_add_co_u32 v92, s5, v81, s34                             // 0000000049C8: D700055C 02004551
	v_lshlrev_b64_e32 v[83:84], 1, v[83:84]                    // 0000000049D0: 3EA6A681
	s_wait_alu depctr_va_sdst(0)                               // 0000000049D4: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v82, s5                // 0000000049D8: D5207C5D 0016A423
	v_or_b32_e32 v90, v90, v126                                // 0000000049E0: 38B4FD5A
	v_cndmask_b32_e64 v91, 0, v91, s4                          // 0000000049E4: D501005B 0012B680
	v_cmp_gt_i64_e64 s6, s[22:23], v[81:82]                    // 0000000049EC: D4540006 0202A216
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000049F4: BF870004
	v_lshlrev_b64_e32 v[92:93], 6, v[92:93]                    // 0000000049F8: 3EB8B886
	v_add_co_u32 v83, s5, s38, v83                             // 0000000049FC: D7000553 0202A626
	s_wait_alu depctr_va_sdst(0)                               // 000000004A04: BF88F19F
	v_add_co_ci_u32_e64 v84, null, s39, v84, s5                // 000000004A08: D5207C54 0016A827
	v_cmp_gt_i64_e64 s5, s[22:23], v[85:86]                    // 000000004A10: D4540005 0202AA16
	v_or_b32_e32 v85, v87, v126                                // 000000004A18: 38AAFD57
	v_cndmask_b32_e64 v90, 0, v90, s4                          // 000000004A1C: D501005A 0012B480
	v_or_b32_e32 v87, v92, v126                                // 000000004A24: 38AEFD5C
	v_or_b32_e32 v79, s40, v79                                 // 000000004A28: 389E9E28
	v_or_b32_e32 v80, s41, v80                                 // 000000004A2C: 38A0A029
	s_wait_alu depctr_va_sdst(0)                               // 000000004A30: BF88F19F
	v_cndmask_b32_e64 v86, 0, v88, s5                          // 000000004A34: D5010056 0016B080
	v_cndmask_b32_e64 v85, 0, v85, s5                          // 000000004A3C: D5010055 0016AA80
	v_lshlrev_b64_e32 v[81:82], 1, v[90:91]                    // 000000004A44: 3EA2B481
	v_cndmask_b32_e64 v88, 0, v93, s6                          // 000000004A48: D5010058 001ABA80
	v_cndmask_b32_e64 v87, 0, v87, s6                          // 000000004A50: D5010057 001AAE80
	v_or_b32_e32 v77, s40, v77                                 // 000000004A58: 389A9A28
	v_lshlrev_b64_e32 v[85:86], 1, v[85:86]                    // 000000004A5C: 3EAAAA81
	v_or_b32_e32 v78, s41, v78                                 // 000000004A60: 389C9C29
	v_add_co_u32 v81, s7, s38, v81                             // 000000004A64: D7000751 0202A226
	v_lshlrev_b64_e32 v[87:88], 1, v[87:88]                    // 000000004A6C: 3EAEAE81
	s_wait_alu depctr_va_sdst(0)                               // 000000004A70: BF88F19F
	v_add_co_ci_u32_e64 v82, null, s39, v82, s7                // 000000004A74: D5207C52 001EA427
	v_add_co_u32 v90, s7, v79, s34                             // 000000004A7C: D700075A 0200454F
	s_wait_alu depctr_va_sdst(0)                               // 000000004A84: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s35, v80, s7                // 000000004A88: D5207C5B 001EA023
	v_add_co_u32 v85, s7, s38, v85                             // 000000004A90: D7000755 0202AA26
	v_or_b32_e32 v75, s40, v75                                 // 000000004A98: 38969628
	s_wait_alu depctr_va_sdst(0)                               // 000000004A9C: BF88F19F
	v_add_co_ci_u32_e64 v86, null, s39, v86, s7                // 000000004AA0: D5207C56 001EAC27
	v_add_co_u32 v87, s7, s38, v87                             // 000000004AA8: D7000757 0202AE26
	v_or_b32_e32 v76, s41, v76                                 // 000000004AB0: 38989829
	s_wait_alu depctr_va_sdst(0)                               // 000000004AB4: BF88F19F
	v_add_co_ci_u32_e64 v88, null, s39, v88, s7                // 000000004AB8: D5207C58 001EB027
	v_cmp_gt_i64_e64 s7, s[22:23], v[79:80]                    // 000000004AC0: D4540007 02029E16
	v_add_co_u32 v79, s8, v77, s34                             // 000000004AC8: D700084F 0200454D
	s_wait_alu depctr_va_sdst(0)                               // 000000004AD0: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s35, v78, s8                // 000000004AD4: D5207C50 00229C23
	v_add_co_u32 v92, s8, v75, s34                             // 000000004ADC: D700085C 0200454B
	s_wait_alu depctr_va_sdst(0)                               // 000000004AE4: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v76, s8                // 000000004AE8: D5207C5D 00229823
	v_or_b32_e32 v73, s40, v73                                 // 000000004AF0: 38929228
	v_or_b32_e32 v74, s41, v74                                 // 000000004AF4: 38949429
	v_cmp_gt_i64_e64 s8, s[22:23], v[77:78]                    // 000000004AF8: D4540008 02029A16
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004B00: BF870004
	v_lshlrev_b64_e32 v[77:78], 6, v[92:93]                    // 000000004B04: 3E9AB886
	v_lshlrev_b64_e32 v[90:91], 6, v[90:91]                    // 000000004B08: 3EB4B486
	v_add_co_u32 v92, s9, v73, s34                             // 000000004B0C: D700095C 02004549
	s_wait_alu depctr_va_sdst(0)                               // 000000004B14: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s35, v74, s9                // 000000004B18: D5207C5D 00269423
	v_lshlrev_b64_e32 v[79:80], 6, v[79:80]                    // 000000004B20: 3E9E9E86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000004B24: BF870224
	v_or_b32_e32 v90, v90, v126                                // 000000004B28: 38B4FD5A
	v_cmp_gt_i64_e64 s9, s[22:23], v[75:76]                    // 000000004B2C: D4540009 02029616
	v_lshlrev_b64_e32 v[75:76], 6, v[92:93]                    // 000000004B34: 3E96B886
	v_or_b32_e32 v77, v77, v126                                // 000000004B38: 389AFD4D
	v_cndmask_b32_e64 v91, 0, v91, s7                          // 000000004B3C: D501005B 001EB680
	v_or_b32_e32 v79, v79, v126                                // 000000004B44: 389EFD4F
	v_cndmask_b32_e64 v90, 0, v90, s7                          // 000000004B48: D501005A 001EB480
	v_cmp_gt_i64_e64 s10, s[22:23], v[73:74]                   // 000000004B50: D454000A 02029216
	v_or_b32_e32 v75, v75, v126                                // 000000004B58: 3896FD4B
	v_cndmask_b32_e64 v80, 0, v80, s8                          // 000000004B5C: D5010050 0022A080
	v_cndmask_b32_e64 v79, 0, v79, s8                          // 000000004B64: D501004F 00229E80
	s_wait_alu depctr_va_sdst(0)                               // 000000004B6C: BF88F19F
	v_cndmask_b32_e64 v78, 0, v78, s9                          // 000000004B70: D501004E 00269C80
	v_cndmask_b32_e64 v77, 0, v77, s9                          // 000000004B78: D501004D 00269A80
	v_lshlrev_b64_e32 v[90:91], 1, v[90:91]                    // 000000004B80: 3EB4B481
	v_cndmask_b32_e64 v76, 0, v76, s10                         // 000000004B84: D501004C 002A9880
	v_cndmask_b32_e64 v75, 0, v75, s10                         // 000000004B8C: D501004B 002A9680
	v_lshlrev_b64_e32 v[79:80], 1, v[79:80]                    // 000000004B94: 3E9E9E81
	v_lshlrev_b64_e32 v[73:74], 1, v[77:78]                    // 000000004B98: 3E929A81
	s_wait_dscnt 0x1                                           // 000000004B9C: BFC60001
	v_cvt_f16_f32_e32 v12.l, v12                               // 000000004BA0: 7E18150C
	v_add_co_u32 v90, s11, s38, v90                            // 000000004BA4: D7000B5A 0202B426
	v_lshlrev_b64_e32 v[75:76], 1, v[75:76]                    // 000000004BAC: 3E969681
	s_wait_alu depctr_va_sdst(0)                               // 000000004BB0: BF88F19F
	v_add_co_ci_u32_e64 v91, null, s39, v91, s11               // 000000004BB4: D5207C5B 002EB627
	v_add_co_u32 v77, s11, s38, v79                            // 000000004BBC: D7000B4D 02029E26
	s_wait_alu depctr_va_sdst(0)                               // 000000004BC4: BF88F19F
	v_add_co_ci_u32_e64 v78, null, s39, v80, s11               // 000000004BC8: D5207C4E 002EA027
	v_add_co_u32 v79, s11, s38, v73                            // 000000004BD0: D7000B4F 02029226
	s_wait_alu depctr_va_sdst(0)                               // 000000004BD8: BF88F19F
	v_add_co_ci_u32_e64 v80, null, s39, v74, s11               // 000000004BDC: D5207C50 002E9427
	v_add_co_u32 v92, s11, s38, v75                            // 000000004BE4: D7000B5C 02029626
	s_wait_alu depctr_va_sdst(0)                               // 000000004BEC: BF88F19F
	v_add_co_ci_u32_e64 v93, null, s39, v76, s11               // 000000004BF0: D5207C5D 002E9827
	s_clause 0x7                                               // 000000004BF8: BF850007
	global_load_d16_b16 v73, v[83:84], off                     // 000000004BFC: EE08007C 00000049 00000053
	global_load_d16_hi_b16 v73, v[81:82], off                  // 000000004C08: EE08C07C 00000049 00000051
	global_load_d16_b16 v74, v[85:86], off                     // 000000004C14: EE08007C 0000004A 00000055
	global_load_d16_hi_b16 v74, v[87:88], off                  // 000000004C20: EE08C07C 0000004A 00000057
	global_load_d16_b16 v75, v[90:91], off                     // 000000004C2C: EE08007C 0000004B 0000005A
	global_load_d16_hi_b16 v75, v[77:78], off                  // 000000004C38: EE08C07C 0000004B 0000004D
	global_load_d16_b16 v76, v[79:80], off                     // 000000004C44: EE08007C 0000004C 0000004F
	global_load_d16_hi_b16 v76, v[92:93], off                  // 000000004C50: EE08C07C 0000004C 0000005C
	v_add_nc_u32_e32 v90, 0x80, v116                           // 000000004C5C: 4AB4E8FF 00000080
	ds_load_2addr_b32 v[77:78], v116 offset0:32 offset1:96     // 000000004C64: D8DC6020 4D000074
	ds_load_2addr_b32 v[79:80], v116 offset0:160 offset1:224   // 000000004C6C: D8DCE0A0 4F000074
	ds_load_2addr_stride64_b32 v[85:86], v90 offset0:4 offset1:5// 000000004C74: D8E00504 5500005A
	ds_load_2addr_stride64_b32 v[87:88], v90 offset0:6 offset1:7// 000000004C7C: D8E00706 5700005A
	s_wait_dscnt 0x4                                           // 000000004C84: BFC60004
	v_cvt_f16_f32_e32 v12.h, v89                               // 000000004C88: 7F181559
	s_wait_dscnt 0x2                                           // 000000004C8C: BFC60002
	v_dual_mul_f32 v89, v1, v77 :: v_dual_mul_f32 v92, v3, v79 // 000000004C90: C8C69B01 595C9F03
	v_mul_f32_e32 v91, v2, v78                                 // 000000004C98: 10B69D02
	v_mul_f32_e32 v93, v4, v80                                 // 000000004C9C: 10BAA104
	s_wait_loadcnt 0x6                                         // 000000004CA0: BFC00006
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CA4: BF88FF9E
	v_cndmask_b16 v81.l, 0, v73.l, s12                         // 000000004CA8: D65D0051 00329280
	v_cndmask_b16 v81.h, 0, v73.h, s4                          // 000000004CB0: D65D5051 00129280
	s_wait_loadcnt 0x4                                         // 000000004CB8: BFC00004
	v_cndmask_b16 v82.l, 0, v74.l, s5                          // 000000004CBC: D65D0052 00169480
	v_cndmask_b16 v82.h, 0, v74.h, s6                          // 000000004CC4: D65D5052 001A9480
	s_wait_loadcnt 0x2                                         // 000000004CCC: BFC00002
	v_cndmask_b16 v83.l, 0, v75.l, s7                          // 000000004CD0: D65D0053 001E9680
	v_cndmask_b16 v83.h, 0, v75.h, s8                          // 000000004CD8: D65D5053 00229680
	s_wait_loadcnt 0x0                                         // 000000004CE0: BFC00000
	v_cndmask_b16 v84.l, 0, v76.l, s9                          // 000000004CE4: D65D0054 00269880
	v_cndmask_b16 v84.h, 0, v76.h, s10                         // 000000004CEC: D65D5054 002A9880
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004CF4: BF870001
	v_wmma_f32_16x16x16_f16 v[73:80], v[9:12], v[81:84], 0     // 000000004CF8: CC404049 1A02A309
	s_wait_dscnt 0x1                                           // 000000004D00: BFC60001
	v_mul_f32_e32 v10, v6, v86                                 // 000000004D04: 1014AD06
	s_wait_dscnt 0x0                                           // 000000004D08: BFC60000
	v_dual_mul_f32 v12, v8, v88 :: v_dual_mul_f32 v9, v5, v85  // 000000004D0C: C8C6B108 0C08AB05
	v_add_f32_e32 v74, v74, v91                                // 000000004D14: 0694B74A
	v_dual_mul_f32 v11, v7, v87 :: v_dual_add_f32 v76, v76, v93// 000000004D18: C8C8AF07 0B4CBB4C
	v_dual_add_f32 v73, v73, v89 :: v_dual_add_f32 v10, v78, v10// 000000004D20: C908B349 490A154E
	v_add_f32_e32 v75, v75, v92                                // 000000004D28: 0696B94B
	v_dual_add_f32 v9, v77, v9 :: v_dual_add_f32 v12, v80, v12 // 000000004D2C: C908134D 090C1950
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004D34: BF870004
	v_add_f32_e32 v11, v79, v11                                // 000000004D38: 0616174F
	ds_store_2addr_b32 v116, v73, v74 offset0:32 offset1:96    // 000000004D3C: D8386020 004A4974
	ds_store_2addr_b32 v116, v75, v76 offset0:160 offset1:224  // 000000004D44: D838E0A0 004C4B74
	ds_store_2addr_stride64_b32 v90, v9, v10 offset0:4 offset1:5// 000000004D4C: D83C0504 000A095A
	ds_store_2addr_stride64_b32 v90, v11, v12 offset0:6 offset1:7// 000000004D54: D83C0706 000C0B5A
	s_and_saveexec_b32 s4, s3                                  // 000000004D5C: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D60: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000004D64: 8D04047E
	s_cbranch_execz 12                                         // 000000004D68: BFA5000C <attention_forward+0x329c>
	ds_load_b128 v[9:12], v111 offset:32                       // 000000004D6C: DBFC0020 0900006F
	ds_load_b64 v[73:74], v111 offset:48                       // 000000004D74: D9D80030 4900006F
	s_wait_dscnt 0x1                                           // 000000004D7C: BFC60001
	v_cvt_f16_f32_e32 v9.l, v9                                 // 000000004D80: 7E121509
	v_cvt_f16_f32_e32 v9.h, v10                                // 000000004D84: 7F12150A
	v_cvt_f16_f32_e32 v10.l, v11                               // 000000004D88: 7E14150B
	v_cvt_f16_f32_e32 v10.h, v12                               // 000000004D8C: 7F14150C
	s_wait_dscnt 0x0                                           // 000000004D90: BFC60000
	v_cvt_f16_f32_e32 v11.l, v73                               // 000000004D94: 7E161549
	v_cvt_f16_f32_e32 v11.h, v74                               // 000000004D98: 7F16154A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D9C: BF88FF9E
	s_or_saveexec_b32 s4, s4                                   // 000000004DA0: BE842204
	v_dual_mov_b32 v12, v112 :: v_dual_mov_b32 v73, v113       // 000000004DA4: CA100170 0C480171
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DAC: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s4                             // 000000004DB0: 8D7E047E
	s_cbranch_execz 14                                         // 000000004DB4: BFA5000E <attention_forward+0x32f0>
	ds_load_b128 v[73:76], v111                                // 000000004DB8: DBFC0000 4900006F
	ds_load_b64 v[77:78], v111 offset:16                       // 000000004DC0: D9D80010 4D00006F
	v_mov_b32_e32 v12, v114                                    // 000000004DC8: 7E180372
	s_wait_dscnt 0x1                                           // 000000004DCC: BFC60001
	v_cvt_f16_f32_e32 v9.l, v73                                // 000000004DD0: 7E121549
	v_cvt_f16_f32_e32 v9.h, v74                                // 000000004DD4: 7F12154A
	v_cvt_f16_f32_e32 v10.l, v75                               // 000000004DD8: 7E14154B
	v_cvt_f16_f32_e32 v10.h, v76                               // 000000004DDC: 7F14154C
	s_wait_dscnt 0x0                                           // 000000004DE0: BFC60000
	v_cvt_f16_f32_e32 v11.l, v77                               // 000000004DE4: 7E16154D
	v_cvt_f16_f32_e32 v11.h, v78                               // 000000004DE8: 7F16154E
	v_mov_b32_e32 v73, v115                                    // 000000004DEC: 7E920373
	s_or_b32 exec_lo, exec_lo, s4                              // 000000004DF0: 8C7E047E
	ds_load_b32 v12, v12                                       // 000000004DF4: D8D80000 0C00000C
	ds_load_b32 v89, v73                                       // 000000004DFC: D8D80000 59000049
	s_and_saveexec_b32 s4, s3                                  // 000000004E04: BE842003
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E08: BF88FF9E
	s_xor_b32 s4, exec_lo, s4                                  // 000000004E0C: 8D04047E
	s_or_b32 s6, s40, 8                                        // 000000004E10: 8C068828
	s_mov_b32 s7, s41                                          // 000000004E14: BE870029
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E18: BF88FF9E
	s_add_nc_u64 s[8:9], s[6:7], s[34:35]                      // 000000004E1C: A9882206
	v_cmp_lt_i64_e64 s12, s[6:7], s[22:23]                     // 000000004E20: D451000C 02002C06
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E28: BF88FF9E
	s_lshl_b64 s[8:9], s[8:9], 6                               // 000000004E2C: 84888608
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E30: BF88FF9E
	v_or_b32_e32 v73, s8, v128                                 // 000000004E34: 38930008
	v_cndmask_b32_e64 v84, 0, s9, s12                          // 000000004E38: D5010054 00301280
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004E40: BF870002
	v_cndmask_b32_e64 v83, 0, v73, s12                         // 000000004E44: D5010053 00329280
	s_or_saveexec_b32 s5, s4                                   // 000000004E4C: BE852204
	v_dual_mov_b32 v73, 15 :: v_dual_mov_b32 v74, 0            // 000000004E50: CA10008F 494A0080
	v_dual_mov_b32 v75, 14 :: v_dual_mov_b32 v76, 0            // 000000004E58: CA10008E 4B4C0080
	v_dual_mov_b32 v77, 13 :: v_dual_mov_b32 v78, 0            // 000000004E60: CA10008D 4D4E0080
	v_dual_mov_b32 v79, 12 :: v_dual_mov_b32 v80, 0            // 000000004E68: CA10008C 4F500080
	v_dual_mov_b32 v81, 11 :: v_dual_mov_b32 v82, 0            // 000000004E70: CA10008B 51520080
	v_dual_mov_b32 v85, 10 :: v_dual_mov_b32 v86, 0            // 000000004E78: CA10008A 55560080
	v_dual_mov_b32 v87, 9 :: v_dual_mov_b32 v88, 0             // 000000004E80: CA100089 57580080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E88: BF88FF9E
	s_xor_b32 exec_lo, exec_lo, s5                             // 000000004E8C: 8D7E057E
	s_cbranch_execz 62604                                      // 000000004E90: BFA5F48C <attention_forward+0x5c4>
	s_lshl_b64 s[6:7], s[30:31], 10                            // 000000004E94: 84868A1E
	v_dual_mov_b32 v73, 7 :: v_dual_mov_b32 v74, 0             // 000000004E98: CA100087 494A0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EA0: BF88FF9E
	v_add_co_u32 v75, s4, s6, v133                             // 000000004EA4: D700044B 02030A06
	s_wait_alu depctr_va_sdst(0)                               // 000000004EAC: BF88F19F
	v_add_co_ci_u32_e64 v76, null, s7, v132, s4                // 000000004EB0: D5207C4C 00130807
	v_cmp_lt_i64_e64 s4, s[40:41], s[22:23]                    // 000000004EB8: D4510004 02002C28
	v_dual_mov_b32 v77, 5 :: v_dual_mov_b32 v78, 0             // 000000004EC0: CA100085 4D4E0080
	v_dual_mov_b32 v79, 4 :: v_dual_mov_b32 v80, 0             // 000000004EC8: CA100084 4F500080
	v_dual_mov_b32 v81, 3 :: v_dual_mov_b32 v82, 0             // 000000004ED0: CA100083 51520080
	s_wait_alu depctr_va_sdst(0)                               // 000000004ED8: BF88F19F
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004EDC: BF870004
	v_cndmask_b32_e64 v84, 0, v76, s4                          // 000000004EE0: D5010054 00129880
	v_cndmask_b32_e64 v83, 0, v75, s4                          // 000000004EE8: D5010053 00129680
	v_dual_mov_b32 v75, 6 :: v_dual_mov_b32 v76, 0             // 000000004EF0: CA100086 4B4C0080
	v_dual_mov_b32 v85, 2 :: v_dual_mov_b32 v86, 0             // 000000004EF8: CA100082 55560080
	v_dual_mov_b32 v87, 1 :: v_dual_mov_b32 v88, 0             // 000000004F00: CA100081 57580080
	s_and_not1_b32 s6, s12, exec_lo                            // 000000004F08: 91067E0C
	s_and_b32 s4, s4, exec_lo                                  // 000000004F0C: 8B047E04
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F10: BF88FF9E
	s_or_b32 s12, s6, s4                                       // 000000004F14: 8C0C0406
	s_branch 62570                                             // 000000004F18: BFA0F46A <attention_forward+0x5c4>
	s_load_b64 s[4:5], s[0:1], 0x80                            // 000000004F1C: F4002100 F8000080
	s_branch 18                                                // 000000004F24: BFA00012 <attention_forward+0x3470>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F28: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000004F2C: 8C7E067E
	v_add_co_u32 v1, vcc_lo, v13, 32                           // 000000004F30: D7006A01 0201410D
	s_wait_alu depctr_va_vcc(0)                                // 000000004F38: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, 0, v14, vcc_lo               // 000000004F3C: D5207C02 01AA1C80
	v_cmp_lt_u64_e32 vcc_lo, 0x3df, v[13:14]                   // 000000004F44: 7CB21AFF 000003DF
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000004F4C: BF870193
	v_dual_mov_b32 v13, v1 :: v_dual_add_nc_u32 v96, 0x80, v96 // 000000004F50: CA200101 0D60C0FF 00000080
	v_mov_b32_e32 v14, v2                                      // 000000004F5C: 7E1C0302
	s_or_b32 s19, vcc_lo, s19                                  // 000000004F60: 8C13136A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F64: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s19                       // 000000004F68: 917E137E
	s_cbranch_execz 68                                         // 000000004F6C: BFA50044 <attention_forward+0x3580>
	v_lshrrev_b64 v[1:2], 6, v[13:14]                          // 000000004F70: D73D0001 02021A86
	s_mov_b32 s6, exec_lo                                      // 000000004F78: BE86007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004F7C: BF870111
	v_or_b32_e32 v2, s27, v2                                   // 000000004F80: 3804041B
	v_or_b32_e32 v1, s26, v1                                   // 000000004F84: 3802021A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F88: BF870001
	v_cmpx_gt_i64_e32 s[20:21], v[1:2]                         // 000000004F8C: 7DA80214
	s_cbranch_execz 65509                                      // 000000004F90: BFA5FFE5 <attention_forward+0x3428>
	v_alignbit_b32 v3, v14, v13, 6                             // 000000004F94: D6160003 021A1B0E
	v_add_co_u32 v1, s3, v1, s24                               // 000000004F9C: D7000301 02003101
	s_wait_alu depctr_va_sdst(0)                               // 000000004FA4: BF88F19F
	v_add_co_ci_u32_e64 v2, null, s25, v2, s3                  // 000000004FA8: D5207C02 000E0419
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004FB0: BF870003
	v_lshlrev_b32_e32 v3, 2, v3                                // 000000004FB4: 30060682
	ds_load_b32 v4, v96                                        // 000000004FB8: D8D80000 04000060
	ds_load_b32 v3, v3 offset:5184                             // 000000004FC0: D8D81440 03000003
	v_lshlrev_b64_e32 v[1:2], 8, v[1:2]                        // 000000004FC8: 3E020288
	s_wait_dscnt 0x0                                           // 000000004FCC: BFC60000
	v_div_scale_f32 v5, null, v3, v3, v4                       // 000000004FD0: D6FC7C05 04120703
	v_div_scale_f32 v8, vcc_lo, v4, v3, v4                     // 000000004FD8: D6FC6A08 04120704
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004FE0: BF870292
	v_rcp_f32_e32 v6, v5                                       // 000000004FE4: 7E0C5505
	v_fma_f32 v7, -v5, v6, 1.0                                 // 000000004FE8: D6130007 23CA0D05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FF0: BF870091
	v_fmac_f32_e32 v6, v7, v6                                  // 000000004FF4: 560C0D07
	v_mul_f32_e32 v7, v8, v6                                   // 000000004FF8: 100E0D08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FFC: BF870091
	v_fma_f32 v9, -v5, v7, v8                                  // 000000005000: D6130009 24220F05
	v_fmac_f32_e32 v7, v9, v6                                  // 000000005008: 560E0D09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000500C: BF870131
	v_fma_f32 v5, -v5, v7, v8                                  // 000000005010: D6130005 24220F05
	v_and_b32_e32 v8, 63, v13                                  // 000000005018: 36101ABF
	s_wait_alu depctr_va_vcc(0)                                // 00000000501C: BF88FF9D
	v_div_fmas_f32 v5, v5, v6, v7                              // 000000005020: D6370005 041E0D05
	s_wait_kmcnt 0x0                                           // 000000005028: BFC70000
	v_add_co_u32 v1, vcc_lo, s4, v1                            // 00000000502C: D7006A01 02020204
	v_lshlrev_b32_e32 v6, 2, v8                                // 000000005034: 300C1082
	s_wait_alu depctr_va_vcc(0)                                // 000000005038: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, s5, v2, vcc_lo               // 00000000503C: D5207C02 01AA0405
	v_div_fixup_f32 v4, v5, v3, v4                             // 000000005044: D6270004 04120705
	v_cmp_lt_f32_e32 vcc_lo, 0, v3                             // 00000000504C: 7C220680
	s_wait_alu depctr_va_vcc(0)                                // 000000005050: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005054: BF870002
	v_cndmask_b32_e32 v3, 0, v4, vcc_lo                        // 000000005058: 02060880
	v_add_co_u32 v1, vcc_lo, v1, v6                            // 00000000505C: D7006A01 02020D01
	s_wait_alu depctr_va_vcc(0)                                // 000000005064: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo                // 000000005068: D5207C02 01AA0480
	global_store_b32 v[1:2], v3, off                           // 000000005070: EE06807C 01800000 00000001
	s_branch 65450                                             // 00000000507C: BFA0FFAA <attention_forward+0x3428>
	s_or_b32 exec_lo, exec_lo, s19                             // 000000005080: 8C7E137E
	s_and_saveexec_b32 s3, s2                                  // 000000005084: BE832002
	s_cbranch_execz 60                                         // 000000005088: BFA5003C <attention_forward+0x367c>
	v_mov_b32_e32 v2, s27                                      // 00000000508C: 7E04021B
	v_or_b32_e32 v1, s26, v0                                   // 000000005090: 3802001A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005094: BF870001
	v_cmp_gt_i64_e32 vcc_lo, s[20:21], v[1:2]                  // 000000005098: 7CA80214
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 00000000509C: 8B7E6A7E
	s_cbranch_execz 54                                         // 0000000050A0: BFA50036 <attention_forward+0x367c>
	v_lshlrev_b32_e32 v0, 2, v0                                // 0000000050A4: 30000082
	s_load_b64 s[0:1], s[0:1], 0xc8                            // 0000000050A8: F4002000 F80000C8
	s_lshl_b64 s[2:3], s[24:25], 2                             // 0000000050B0: 84828218
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000050B4: BF870001
	v_add_nc_u32_e32 v0, 0x1400, v0                            // 0000000050B8: 4A0000FF 00001400
	ds_load_2addr_b32 v[3:4], v0 offset1:16                    // 0000000050C0: D8DC1000 03000000
	s_wait_kmcnt 0x0                                           // 0000000050C8: BFC70000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050CC: BF88FF9E
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 0000000050D0: A9800200
	s_wait_dscnt 0x0                                           // 0000000050D4: BFC60000
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v4                      // 0000000050D8: 7C2808FF 00800000
	s_wait_alu depctr_va_vcc(0)                                // 0000000050E0: BF88FF9D
	v_cndmask_b32_e64 v0, 0, 32, vcc_lo                        // 0000000050E4: D5010000 01A94080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000050EC: BF870091
	v_ldexp_f32 v0, v4, v0                                     // 0000000050F0: D71C0000 02020104
	v_log_f32_e32 v0, v0                                       // 0000000050F8: 7E004F00
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000050FC: BF870095
	v_mul_f32_e32 v4, 0x3f317217, v0                           // 000000005100: 100800FF 3F317217
	v_fma_f32 v5, 0x3f317217, v0, -v4                          // 000000005108: D6130005 841200FF 3F317217
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005114: BF870091
	v_fmamk_f32 v5, v0, 0x3377d1cf, v5                         // 000000005118: 580A0B00 3377D1CF
	v_add_f32_e32 v4, v4, v5                                   // 000000005120: 06080B04
	v_cndmask_b32_e64 v5, 0, 0x41b17218, vcc_lo                // 000000005124: D5010005 01A9FE80 41B17218
	v_cmp_gt_f32_e64 vcc_lo, 0x7f800000, |v0|                  // 000000005130: D414026A 020200FF 7F800000
	s_wait_alu depctr_va_vcc(0)                                // 00000000513C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005140: BF870093
	v_cndmask_b32_e32 v0, v0, v4, vcc_lo                       // 000000005144: 02000900
	v_sub_f32_e32 v4, v0, v5                                   // 000000005148: 08080B00
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]                        // 00000000514C: 3E000282
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005150: BF870112
	v_add_f32_e32 v2, v3, v4                                   // 000000005154: 06040903
	v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000005158: D7006A00 02020000
	s_wait_alu depctr_va_vcc(0)                                // 000000005160: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005164: BF870003
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo               // 000000005168: D5207C01 01AA0201
	global_store_b32 v[0:1], v2, off                           // 000000005170: EE06807C 01000000 00000000
	s_nop 0                                                    // 00000000517C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000005180: BFB60003
	s_endpgm                                                   // 000000005184: BFB00000
	s_code_end                                                 // 000000005188: BF9F0000
	s_code_end                                                 // 00000000518C: BF9F0000
	s_code_end                                                 // 000000005190: BF9F0000
	s_code_end                                                 // 000000005194: BF9F0000
	s_code_end                                                 // 000000005198: BF9F0000
	s_code_end                                                 // 00000000519C: BF9F0000
	s_code_end                                                 // 0000000051A0: BF9F0000
	s_code_end                                                 // 0000000051A4: BF9F0000
	s_code_end                                                 // 0000000051A8: BF9F0000
	s_code_end                                                 // 0000000051AC: BF9F0000
	s_code_end                                                 // 0000000051B0: BF9F0000
	s_code_end                                                 // 0000000051B4: BF9F0000
	s_code_end                                                 // 0000000051B8: BF9F0000
	s_code_end                                                 // 0000000051BC: BF9F0000
	s_code_end                                                 // 0000000051C0: BF9F0000
	s_code_end                                                 // 0000000051C4: BF9F0000
	s_code_end                                                 // 0000000051C8: BF9F0000
	s_code_end                                                 // 0000000051CC: BF9F0000
	s_code_end                                                 // 0000000051D0: BF9F0000
	s_code_end                                                 // 0000000051D4: BF9F0000
	s_code_end                                                 // 0000000051D8: BF9F0000
	s_code_end                                                 // 0000000051DC: BF9F0000
	s_code_end                                                 // 0000000051E0: BF9F0000
	s_code_end                                                 // 0000000051E4: BF9F0000
	s_code_end                                                 // 0000000051E8: BF9F0000
	s_code_end                                                 // 0000000051EC: BF9F0000
	s_code_end                                                 // 0000000051F0: BF9F0000
	s_code_end                                                 // 0000000051F4: BF9F0000
	s_code_end                                                 // 0000000051F8: BF9F0000
	s_code_end                                                 // 0000000051FC: BF9F0000
	s_code_end                                                 // 000000005200: BF9F0000
	s_code_end                                                 // 000000005204: BF9F0000
	s_code_end                                                 // 000000005208: BF9F0000
	s_code_end                                                 // 00000000520C: BF9F0000
	s_code_end                                                 // 000000005210: BF9F0000
	s_code_end                                                 // 000000005214: BF9F0000
	s_code_end                                                 // 000000005218: BF9F0000
	s_code_end                                                 // 00000000521C: BF9F0000
	s_code_end                                                 // 000000005220: BF9F0000
	s_code_end                                                 // 000000005224: BF9F0000
	s_code_end                                                 // 000000005228: BF9F0000
	s_code_end                                                 // 00000000522C: BF9F0000
	s_code_end                                                 // 000000005230: BF9F0000
	s_code_end                                                 // 000000005234: BF9F0000
	s_code_end                                                 // 000000005238: BF9F0000
	s_code_end                                                 // 00000000523C: BF9F0000
	s_code_end                                                 // 000000005240: BF9F0000
	s_code_end                                                 // 000000005244: BF9F0000
	s_code_end                                                 // 000000005248: BF9F0000
	s_code_end                                                 // 00000000524C: BF9F0000
	s_code_end                                                 // 000000005250: BF9F0000
	s_code_end                                                 // 000000005254: BF9F0000
	s_code_end                                                 // 000000005258: BF9F0000
	s_code_end                                                 // 00000000525C: BF9F0000
	s_code_end                                                 // 000000005260: BF9F0000
	s_code_end                                                 // 000000005264: BF9F0000
	s_code_end                                                 // 000000005268: BF9F0000
	s_code_end                                                 // 00000000526C: BF9F0000
	s_code_end                                                 // 000000005270: BF9F0000
	s_code_end                                                 // 000000005274: BF9F0000
	s_code_end                                                 // 000000005278: BF9F0000
	s_code_end                                                 // 00000000527C: BF9F0000
	s_code_end                                                 // 000000005280: BF9F0000
	s_code_end                                                 // 000000005284: BF9F0000
	s_code_end                                                 // 000000005288: BF9F0000
	s_code_end                                                 // 00000000528C: BF9F0000
	s_code_end                                                 // 000000005290: BF9F0000
	s_code_end                                                 // 000000005294: BF9F0000
	s_code_end                                                 // 000000005298: BF9F0000
	s_code_end                                                 // 00000000529C: BF9F0000
	s_code_end                                                 // 0000000052A0: BF9F0000
	s_code_end                                                 // 0000000052A4: BF9F0000
	s_code_end                                                 // 0000000052A8: BF9F0000
	s_code_end                                                 // 0000000052AC: BF9F0000
	s_code_end                                                 // 0000000052B0: BF9F0000
	s_code_end                                                 // 0000000052B4: BF9F0000
	s_code_end                                                 // 0000000052B8: BF9F0000
	s_code_end                                                 // 0000000052BC: BF9F0000
	s_code_end                                                 // 0000000052C0: BF9F0000
	s_code_end                                                 // 0000000052C4: BF9F0000
	s_code_end                                                 // 0000000052C8: BF9F0000
	s_code_end                                                 // 0000000052CC: BF9F0000
	s_code_end                                                 // 0000000052D0: BF9F0000
	s_code_end                                                 // 0000000052D4: BF9F0000
	s_code_end                                                 // 0000000052D8: BF9F0000
	s_code_end                                                 // 0000000052DC: BF9F0000
	s_code_end                                                 // 0000000052E0: BF9F0000
	s_code_end                                                 // 0000000052E4: BF9F0000
	s_code_end                                                 // 0000000052E8: BF9F0000
	s_code_end                                                 // 0000000052EC: BF9F0000
	s_code_end                                                 // 0000000052F0: BF9F0000
	s_code_end                                                 // 0000000052F4: BF9F0000
	s_code_end                                                 // 0000000052F8: BF9F0000
	s_code_end                                                 // 0000000052FC: BF9F0000
	s_code_end                                                 // 000000005300: BF9F0000
	s_code_end                                                 // 000000005304: BF9F0000
	s_code_end                                                 // 000000005308: BF9F0000
	s_code_end                                                 // 00000000530C: BF9F0000
	s_code_end                                                 // 000000005310: BF9F0000
	s_code_end                                                 // 000000005314: BF9F0000
	s_code_end                                                 // 000000005318: BF9F0000
	s_code_end                                                 // 00000000531C: BF9F0000
	s_code_end                                                 // 000000005320: BF9F0000
	s_code_end                                                 // 000000005324: BF9F0000
	s_code_end                                                 // 000000005328: BF9F0000
	s_code_end                                                 // 00000000532C: BF9F0000
	s_code_end                                                 // 000000005330: BF9F0000
	s_code_end                                                 // 000000005334: BF9F0000
	s_code_end                                                 // 000000005338: BF9F0000
	s_code_end                                                 // 00000000533C: BF9F0000
	s_code_end                                                 // 000000005340: BF9F0000
	s_code_end                                                 // 000000005344: BF9F0000
	s_code_end                                                 // 000000005348: BF9F0000
	s_code_end                                                 // 00000000534C: BF9F0000
	s_code_end                                                 // 000000005350: BF9F0000
	s_code_end                                                 // 000000005354: BF9F0000
	s_code_end                                                 // 000000005358: BF9F0000
	s_code_end                                                 // 00000000535C: BF9F0000
	s_code_end                                                 // 000000005360: BF9F0000
	s_code_end                                                 // 000000005364: BF9F0000
	s_code_end                                                 // 000000005368: BF9F0000
	s_code_end                                                 // 00000000536C: BF9F0000
	s_code_end                                                 // 000000005370: BF9F0000
	s_code_end                                                 // 000000005374: BF9F0000
	s_code_end                                                 // 000000005378: BF9F0000
	s_code_end                                                 // 00000000537C: BF9F0000
