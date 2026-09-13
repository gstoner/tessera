
/tmp/gfx1201-attention-loading/baseline.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001b00 <attention_forward>:
	v_mov_b32_e32 v1, 0                                        // 000000001B00: 7E020280
	s_mov_b32 s6, ttmp7                                        // 000000001B04: BE860073
	s_mov_b32 s3, 0                                            // 000000001B08: BE830080
	v_dual_mov_b32 v17, v0 :: v_dual_mov_b32 v2, v0            // 000000001B0C: CA100100 11020100
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001B14: BF870002
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v18, v1            // 000000001B18: CA100101 03120101
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
	v_lshlrev_b32_e32 v43, 2, v0                               // 000000001B68: 30560082
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
	s_load_b128 s[24:27], s[0:1], 0xa0                         // 000000001BA4: F4004600 F80000A0
	s_load_b64 s[10:11], s[0:1], 0xb8                          // 000000001BAC: F4002280 F80000B8
	s_mov_b32 s4, ttmp9                                        // 000000001BB4: BE840075
	s_ashr_i32 s5, ttmp9, 31                                   // 000000001BB8: 86059F75
	s_ashr_i32 s7, s6, 31                                      // 000000001BBC: 86079F06
	s_lshl_b64 s[28:29], s[4:5], 4                             // 000000001BC0: 849C8404
	s_wait_dscnt 0x0                                           // 000000001BC4: BFC60000
	s_barrier_signal -1                                        // 000000001BC8: BE804EC1
	s_mov_b64 s[34:35], 15                                     // 000000001BCC: BEA2018F
	s_mov_b32 s21, 0                                           // 000000001BD0: BE950080
	s_wait_kmcnt 0x0                                           // 000000001BD4: BFC70000
	v_cmp_gt_i64_e64 s3, s[26:27], s[24:25]                    // 000000001BD8: D4540003 0200301A
	s_add_nc_u64 s[4:5], s[26:27], 15                          // 000000001BE0: A9848F1A
	s_sub_nc_u64 s[8:9], s[26:27], s[24:25]                    // 000000001BE4: AA08181A
	s_lshr_b64 s[4:5], s[4:5], 4                               // 000000001BE8: 85848404
	s_mul_u64 s[22:23], s[24:25], s[6:7]                       // 000000001BEC: AA960618
	s_add_nc_u64 s[14:15], s[4:5], -1                          // 000000001BF0: A98EC104
	s_and_b32 s3, s3, exec_lo                                  // 000000001BF4: 8B037E03
	s_cselect_b32 s9, s9, 0                                    // 000000001BF8: 98098009
	s_cselect_b32 s8, s8, 0                                    // 000000001BFC: 98088008
	s_barrier_wait 0xffff                                      // 000000001C00: BF94FFFF
	s_add_nc_u64 s[8:9], s[8:9], s[28:29]                      // 000000001C04: A9881C08
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
	s_cselect_b32 s31, s11, s5                                 // 000000001C50: 981F050B
	s_cselect_b32 s30, s10, s4                                 // 000000001C54: 981E040A
	s_cmp_eq_u64 s[30:31], 0                                   // 000000001C58: BF10801E
	s_cbranch_scc1 3163                                        // 000000001C5C: BFA20C5B <attention_forward+0x32cc>
	v_and_b32_e32 v44, 15, v0                                  // 000000001C60: 3658008F
	v_mov_b32_e32 v2, s29                                      // 000000001C64: 7E04021D
	s_load_b64 s[10:11], s[0:1], 0x8                           // 000000001C68: F4002280 F8000008
	v_lshrrev_b32_e32 v8, 4, v0                                // 000000001C70: 32100084
	s_clause 0x2                                               // 000000001C74: BF850002
	s_load_b64 s[36:37], s[0:1], 0x30                          // 000000001C78: F4002900 F8000030
	s_load_b64 s[38:39], s[0:1], 0x58                          // 000000001C80: F4002980 F8000058
	s_load_b32 s42, s[0:1], 0xb0                               // 000000001C88: F4000A80 F80000B0
	v_or_b32_e32 v1, s28, v44                                  // 000000001C90: 3802581C
	v_lshlrev_b32_e32 v53, 6, v0                               // 000000001C94: 306A0086
	v_cmp_eq_u32_e64 s3, 0, v8                                 // 000000001C98: D44A0003 02021080
	v_lshl_add_u32 v59, v8, 5, 0x1480                          // 000000001CA0: D646003B 03FD0B08 00001480
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001CAC: BF870094
	v_add_co_u32 v3, vcc_lo, s22, v1                           // 000000001CB0: D7006A03 02020216
	v_add_co_ci_u32_e64 v4, null, s23, v2, vcc_lo              // 000000001CB8: D5207C04 01AA0417
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[1:2]                  // 000000001CC0: 7CA80218
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001CC4: BF8700A2
	v_lshlrev_b64_e32 v[3:4], 6, v[3:4]                        // 000000001CC8: 3E060686
	s_wait_alu depctr_va_vcc(0)                                // 000000001CCC: BF88FF9D
	v_cndmask_b32_e32 v1, 0, v3, vcc_lo                        // 000000001CD0: 02020680
	v_or_b32_e32 v6, 16, v3                                    // 000000001CD4: 380C0690
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001CD8: BF870003
	v_cndmask_b32_e32 v2, 0, v4, vcc_lo                        // 000000001CDC: 02040880
	v_or_b32_e32 v11, 32, v3                                   // 000000001CE0: 381606A0
	v_lshlrev_b32_e32 v10, 3, v8                               // 000000001CE4: 30141083
	v_or_b32_e32 v14, 48, v3                                   // 000000001CE8: 381C06B0
	v_lshlrev_b32_e32 v9, 2, v44                               // 000000001CEC: 30125882
	v_lshlrev_b64_e32 v[4:5], 1, v[1:2]                        // 000000001CF0: 3E080281
	v_cndmask_b32_e32 v1, 0, v6, vcc_lo                        // 000000001CF4: 02020C80
	v_lshlrev_b32_e32 v57, 6, v44                              // 000000001CF8: 30725886
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001CFC: BF870224
	v_lshl_or_b32 v12, v8, 9, v9                               // 000000001D00: D656000C 04251308
	v_lshl_or_b32 v58, v8, 11, v9                              // 000000001D08: D656003A 04251708
	v_lshlrev_b64_e32 v[6:7], 1, v[1:2]                        // 000000001D10: 3E0C0281
	v_cndmask_b32_e32 v1, 0, v11, vcc_lo                       // 000000001D14: 02021680
	s_wait_kmcnt 0x0                                           // 000000001D18: BFC70000
	v_add_co_u32 v19, s4, s10, v4                              // 000000001D1C: D7000413 0202080A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001D24: BF870191
	v_add_co_ci_u32_e64 v20, null, s11, v5, s4                 // 000000001D28: D5207C14 00120A0B
	v_lshlrev_b64_e32 v[3:4], 1, v[1:2]                        // 000000001D30: 3E060281
	v_cndmask_b32_e32 v1, 0, v14, vcc_lo                       // 000000001D34: 02021C80
	v_add_co_u32 v21, s4, s10, v6                              // 000000001D38: D7000415 02020C0A
	v_or_b32_e32 v13, 1, v10                                   // 000000001D40: 381A1481
	s_wait_alu depctr_va_sdst(0)                               // 000000001D44: BF88F19F
	v_add_co_ci_u32_e64 v22, null, s11, v7, s4                 // 000000001D48: D5207C16 00120E0B
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 000000001D50: 3E020281
	v_add_co_u32 v23, s4, s10, v3                              // 000000001D54: D7000417 0202060A
	s_wait_alu depctr_va_sdst(0)                               // 000000001D5C: BF88F19F
	v_add_co_ci_u32_e64 v24, null, s11, v4, s4                 // 000000001D60: D5207C18 0012080B
	v_or_b32_e32 v3, 3, v10                                    // 000000001D68: 38061483
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001D6C: BF870004
	v_add_co_u32 v25, s4, s10, v1                              // 000000001D70: D7000419 0202020A
	s_wait_alu depctr_va_sdst(0)                               // 000000001D78: BF88F19F
	v_add_co_ci_u32_e64 v26, null, s11, v2, s4                 // 000000001D7C: D5207C1A 0012040B
	v_lshl_or_b32 v1, v13, 6, v9                               // 000000001D84: D6560001 04250D0D
	v_or_b32_e32 v2, 2, v10                                    // 000000001D8C: 38041482
	v_or_b32_e32 v4, 4, v10                                    // 000000001D90: 38081484
	v_or_b32_e32 v5, 5, v10                                    // 000000001D94: 380A1485
	v_add_co_u32 v27, s4, v10, s8                              // 000000001D98: D700041B 0200110A
	v_add_nc_u32_e32 v46, 0x1000, v1                           // 000000001DA0: 4A5C02FF 00001000
	v_lshl_or_b32 v1, v2, 6, v9                                // 000000001DA8: D6560001 04250D02
	s_wait_alu depctr_va_sdst(0)                               // 000000001DB0: BF88F19F
	v_add_co_ci_u32_e64 v28, null, 0, s9, s4                   // 000000001DB4: D5207C1C 00101280
	v_or_b32_e32 v6, 6, v10                                    // 000000001DBC: 380C1486
	v_add_co_u32 v29, s4, v27, 1                               // 000000001DC0: D700041D 0201031B
	v_add_nc_u32_e32 v47, 0x1000, v1                           // 000000001DC8: 4A5E02FF 00001000
	v_lshl_or_b32 v1, v3, 6, v9                                // 000000001DD0: D6560001 04250D03
	s_wait_alu depctr_va_sdst(0)                               // 000000001DD8: BF88F19F
	v_add_co_ci_u32_e64 v30, null, 0, v28, s4                  // 000000001DDC: D5207C1E 00123880
	v_add_co_u32 v31, s4, v27, 2                               // 000000001DE4: D700041F 0201051B
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)// 000000001DEC: BF8701D3
	v_add_nc_u32_e32 v48, 0x1000, v1                           // 000000001DF0: 4A6002FF 00001000
	v_lshl_or_b32 v1, v4, 6, v9                                // 000000001DF8: D6560001 04250D04
	s_wait_alu depctr_va_sdst(0)                               // 000000001E00: BF88F19F
	v_add_co_ci_u32_e64 v32, null, 0, v28, s4                  // 000000001E04: D5207C20 00123880
	v_add_co_u32 v33, s4, v27, 3                               // 000000001E0C: D7000421 0201071B
	v_add_nc_u32_e32 v49, 0x1000, v1                           // 000000001E14: 4A6202FF 00001000
	v_lshl_or_b32 v1, v5, 6, v9                                // 000000001E1C: D6560001 04250D05
	v_or_b32_e32 v7, 7, v10                                    // 000000001E24: 380E1487
	s_wait_alu depctr_va_sdst(0)                               // 000000001E28: BF88F19F
	v_add_co_ci_u32_e64 v34, null, 0, v28, s4                  // 000000001E2C: D5207C22 00123880
	v_add_co_u32 v35, s4, v27, 4                               // 000000001E34: D7000423 0201091B
	v_add_nc_u32_e32 v50, 0x1000, v1                           // 000000001E3C: 4A6402FF 00001000
	v_lshl_or_b32 v1, v6, 6, v9                                // 000000001E44: D6560001 04250D06
	s_wait_alu depctr_va_sdst(0)                               // 000000001E4C: BF88F19F
	v_add_co_ci_u32_e64 v36, null, 0, v28, s4                  // 000000001E50: D5207C24 00123880
	v_add_co_u32 v37, s4, v27, 5                               // 000000001E58: D7000425 02010B1B
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001E60: BF870003
	v_add_nc_u32_e32 v51, 0x1000, v1                           // 000000001E64: 4A6602FF 00001000
	v_lshl_or_b32 v1, v7, 6, v9                                // 000000001E6C: D6560001 04250D07
	v_lshlrev_b32_e32 v10, 2, v0                               // 000000001E74: 30140082
	s_wait_alu depctr_va_sdst(0)                               // 000000001E78: BF88F19F
	v_add_co_ci_u32_e64 v38, null, 0, v28, s4                  // 000000001E7C: D5207C26 00123880
	v_add_co_u32 v39, s4, v27, 6                               // 000000001E84: D7000427 02010D1B
	s_wait_alu depctr_va_sdst(0)                               // 000000001E8C: BF88F19F
	v_add_co_ci_u32_e64 v40, null, 0, v28, s4                  // 000000001E90: D5207C28 00123880
	v_add_co_u32 v41, s4, v27, 7                               // 000000001E98: D7000429 02010F1B
	v_add_nc_u32_e32 v45, 0x1000, v12                          // 000000001EA0: 4A5A18FF 00001000
	s_wait_alu depctr_va_sdst(0)                               // 000000001EA8: BF88F19F
	v_add_co_ci_u32_e64 v42, null, 0, v28, s4                  // 000000001EAC: D5207C2A 00123880
	v_add_nc_u32_e32 v52, 0x1000, v1                           // 000000001EB4: 4A6802FF 00001000
	v_or_b32_e32 v54, 0x1400, v10                              // 000000001EBC: 386C14FF 00001400
	v_add_nc_u32_e32 v55, 0x1440, v10                          // 000000001EC4: 4A6E14FF 00001440
	v_add_nc_u32_e32 v56, 0x1480, v10                          // 000000001ECC: 4A7014FF 00001480
	v_lshl_or_b32 v60, v13, 8, v9                              // 000000001ED4: D656003C 0425110D
	v_lshl_or_b32 v61, v2, 8, v9                               // 000000001EDC: D656003D 04251102
	v_lshl_or_b32 v62, v3, 8, v9                               // 000000001EE4: D656003E 04251103
	v_lshl_or_b32 v63, v4, 8, v9                               // 000000001EEC: D656003F 04251104
	v_lshl_or_b32 v64, v5, 8, v9                               // 000000001EF4: D6560040 04251105
	v_lshl_or_b32 v65, v6, 8, v9                               // 000000001EFC: D6560041 04251106
	v_lshl_or_b32 v66, v7, 8, v9                               // 000000001F04: D6560042 04251107
	s_mul_u64 s[4:5], s[26:27], s[6:7]                         // 000000001F0C: AA84061A
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F10: BF88FF9E
	s_lshl_b64 s[40:41], s[4:5], 6                             // 000000001F14: 84A88604
	s_branch 1766                                              // 000000001F18: BFA006E6 <attention_forward+0x1fb4>
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F1C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000001F20: 8C7E057E
	s_add_nc_u64 s[4:5], s[34:35], -15                         // 000000001F24: A984CF22
	v_add_co_u32 v5, s6, v44, s40                              // 000000001F28: D7000605 0200512C
	s_wait_alu depctr_va_sdst(0)                               // 000000001F30: BF88F19F
	v_add_co_ci_u32_e64 v6, null, 0, s41, s6                   // 000000001F34: D5207C06 00185280
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F3C: BF88FF9E
	v_cmp_lt_i64_e64 s4, s[4:5], s[26:27]                      // 000000001F40: D4510004 02003404
	s_add_nc_u64 s[6:7], s[34:35], -7                          // 000000001F48: A986C722
	v_add_co_u32 v3, s5, 0x200, v5                             // 000000001F4C: D7000503 02020AFF 00000200
	s_wait_alu depctr_va_sdst(0)                               // 000000001F58: BF88F19F
	v_add_co_ci_u32_e64 v4, null, 0, v6, s5                    // 000000001F5C: D5207C04 00160C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001F64: BF88FF9E
	v_cmp_lt_i64_e64 s5, s[6:7], s[26:27]                      // 000000001F68: D4510005 02003406
	v_cndmask_b32_e64 v2, 0, v6, s4                            // 000000001F70: D5010002 00120C80
	v_cndmask_b32_e64 v1, 0, v5, s4                            // 000000001F78: D5010001 00120A80
	s_add_nc_u64 s[8:9], s[34:35], -14                         // 000000001F80: A988CE22
	v_add_co_u32 v7, s6, v5, 64                                // 000000001F84: D7000607 02018105
	v_cndmask_b32_e64 v4, 0, v4, s5                            // 000000001F8C: D5010004 00160880
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001F94: BF870003
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]                        // 000000001F98: 3E020281
	v_cndmask_b32_e64 v3, 0, v3, s5                            // 000000001F9C: D5010003 00160680
	s_wait_alu depctr_va_sdst(0)                               // 000000001FA4: BF88F19F
	v_add_co_ci_u32_e64 v8, null, 0, v6, s6                    // 000000001FA8: D5207C08 001A0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000001FB0: BF88FF9E
	v_cmp_lt_i64_e64 s6, s[8:9], s[26:27]                      // 000000001FB4: D4510006 02003408
	s_add_nc_u64 s[8:9], s[34:35], -6                          // 000000001FBC: A988C622
	v_lshlrev_b64_e32 v[3:4], 1, v[3:4]                        // 000000001FC0: 3E060681
	v_add_co_u32 v1, s7, s38, v1                               // 000000001FC4: D7000701 02020226
	s_wait_alu depctr_va_sdst(0)                               // 000000001FCC: BF88F19F
	v_add_co_ci_u32_e64 v2, null, s39, v2, s7                  // 000000001FD0: D5207C02 001E0427
	v_cndmask_b32_e64 v8, 0, v8, s6                            // 000000001FD8: D5010008 001A1080
	v_cndmask_b32_e64 v7, 0, v7, s6                            // 000000001FE0: D5010007 001A0E80
	v_add_co_u32 v11, s7, 0x240, v5                            // 000000001FE8: D700070B 02020AFF 00000240
	s_wait_alu depctr_va_sdst(0)                               // 000000001FF4: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s7                   // 000000001FF8: D5207C0C 001E0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002000: BF88FF9E
	v_cmp_lt_i64_e64 s7, s[8:9], s[26:27]                      // 000000002004: D4510007 02003408
	v_add_co_u32 v9, s8, s38, v3                               // 00000000200C: D7000809 02020626
	s_wait_alu depctr_va_sdst(0)                               // 000000002014: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v4, s8                 // 000000002018: D5207C0A 00220827
	v_lshlrev_b64_e32 v[3:4], 1, v[7:8]                        // 000000002020: 3E060E81
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002024: BF870004
	v_cndmask_b32_e64 v8, 0, v12, s7                           // 000000002028: D5010008 001E1880
	v_cndmask_b32_e64 v7, 0, v11, s7                           // 000000002030: D5010007 001E1680
	s_add_nc_u64 s[10:11], s[34:35], -13                       // 000000002038: A98ACD22
	v_add_co_u32 v13, s8, 0x80, v5                             // 00000000203C: D700080D 02020AFF 00000080
	s_wait_alu depctr_va_sdst(0)                               // 000000002048: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s8                   // 00000000204C: D5207C0E 00220C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002054: BF88FF9E
	v_cmp_lt_i64_e64 s8, s[10:11], s[26:27]                    // 000000002058: D4510008 0200340A
	v_add_co_u32 v11, s9, s38, v3                              // 000000002060: D700090B 02020626
	s_wait_alu depctr_va_sdst(0)                               // 000000002068: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v4, s9                 // 00000000206C: D5207C0C 00260827
	v_lshlrev_b64_e32 v[3:4], 1, v[7:8]                        // 000000002074: 3E060E81
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002078: BF870004
	v_cndmask_b32_e64 v8, 0, v14, s8                           // 00000000207C: D5010008 00221C80
	v_cndmask_b32_e64 v7, 0, v13, s8                           // 000000002084: D5010007 00221A80
	s_add_nc_u64 s[10:11], s[34:35], -5                        // 00000000208C: A98AC522
	v_add_co_u32 v15, s9, 0x280, v5                            // 000000002090: D700090F 02020AFF 00000280
	s_wait_alu depctr_va_sdst(0)                               // 00000000209C: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s9                   // 0000000020A0: D5207C10 00260C80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020A8: BF88FF9E
	v_cmp_lt_i64_e64 s9, s[10:11], s[26:27]                    // 0000000020AC: D4510009 0200340A
	s_add_nc_u64 s[12:13], s[34:35], -12                       // 0000000020B4: A98CCC22
	v_add_co_u32 v67, s10, 0xc0, v5                            // 0000000020B8: D7000A43 02020AFF 000000C0
	v_add_co_u32 v13, s11, s38, v3                             // 0000000020C4: D7000B0D 02020626
	s_wait_alu depctr_va_sdst(0)                               // 0000000020CC: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s10                  // 0000000020D0: D5207C44 002A0C80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000020D8: BF88FF9E
	v_cmp_lt_i64_e64 s10, s[12:13], s[26:27]                   // 0000000020DC: D451000A 0200340C
	v_add_co_ci_u32_e64 v14, null, s39, v4, s11                // 0000000020E4: D5207C0E 002E0827
	v_lshlrev_b64_e32 v[3:4], 1, v[7:8]                        // 0000000020EC: 3E060E81
	v_cndmask_b32_e64 v8, 0, v16, s9                           // 0000000020F0: D5010008 00262080
	v_cndmask_b32_e64 v7, 0, v15, s9                           // 0000000020F8: D5010007 00261E80
	s_wait_alu depctr_va_sdst(0)                               // 000000002100: BF88F19F
	v_cndmask_b32_e64 v15, 0, v67, s10                         // 000000002104: D501000F 002A8680
	v_cndmask_b32_e64 v16, 0, v68, s10                         // 00000000210C: D5010010 002A8880
	s_add_nc_u64 s[12:13], s[34:35], -4                        // 000000002114: A98CC422
	v_add_co_u32 v67, s11, s38, v3                             // 000000002118: D7000B43 02020626
	s_wait_alu depctr_va_sdst(0)                               // 000000002120: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v4, s11                // 000000002124: D5207C44 002E0827
	v_lshlrev_b64_e32 v[3:4], 1, v[7:8]                        // 00000000212C: 3E060E81
	v_add_co_u32 v69, s11, 0x2c0, v5                           // 000000002130: D7000B45 02020AFF 000002C0
	s_wait_alu depctr_va_sdst(0)                               // 00000000213C: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s11                  // 000000002140: D5207C46 002E0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002148: BF88FF9E
	v_cmp_lt_i64_e64 s11, s[12:13], s[26:27]                   // 00000000214C: D451000B 0200340C
	v_lshlrev_b64_e32 v[7:8], 1, v[15:16]                      // 000000002154: 3E0E1E81
	v_add_co_u32 v15, s12, s38, v3                             // 000000002158: D7000C0F 02020626
	s_wait_alu depctr_va_sdst(0)                               // 000000002160: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v4, s12                // 000000002164: D5207C10 00320827
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 00000000216C: BF870154
	v_cndmask_b32_e64 v4, 0, v70, s11                          // 000000002170: D5010004 002E8C80
	v_cndmask_b32_e64 v3, 0, v69, s11                          // 000000002178: D5010003 002E8A80
	s_wait_loadcnt_dscnt 0x0                                   // 000000002180: BFC80000
	s_barrier_signal -1                                        // 000000002184: BE804EC1
	v_add_co_u32 v7, s12, s38, v7                              // 000000002188: D7000C07 02020E26
	v_lshlrev_b64_e32 v[3:4], 1, v[3:4]                        // 000000002190: 3E060681
	s_wait_alu depctr_va_sdst(0)                               // 000000002194: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s12                 // 000000002198: D5207C08 00321027
	s_add_nc_u64 s[14:15], s[34:35], -11                       // 0000000021A0: A98ECB22
	v_add_co_u32 v71, s12, 0x100, v5                           // 0000000021A4: D7000C47 02020AFF 00000100
	s_wait_alu depctr_va_sdst(0)                               // 0000000021B0: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s12                  // 0000000021B4: D5207C48 00320C80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021BC: BF88FF9E
	v_cmp_lt_i64_e64 s12, s[14:15], s[26:27]                   // 0000000021C0: D451000C 0200340E
	v_add_co_u32 v69, s13, s38, v3                             // 0000000021C8: D7000D45 02020626
	s_wait_alu depctr_va_sdst(0)                               // 0000000021D0: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v4, s13                // 0000000021D4: D5207C46 00360827
	s_add_nc_u64 s[14:15], s[34:35], -3                        // 0000000021DC: A98EC322
	v_add_co_u32 v73, s13, 0x300, v5                           // 0000000021E0: D7000D49 02020AFF 00000300
	s_wait_alu depctr_va_sdst(0)                               // 0000000021EC: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s13                  // 0000000021F0: D5207C4A 00360C80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000021F8: BF88FF9E
	v_cmp_lt_i64_e64 s13, s[14:15], s[26:27]                   // 0000000021FC: D451000D 0200340E
	v_cndmask_b32_e64 v72, 0, v72, s12                         // 000000002204: D5010048 00329080
	v_cndmask_b32_e64 v71, 0, v71, s12                         // 00000000220C: D5010047 00328E80
	s_barrier_wait 0xffff                                      // 000000002214: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002218: EE0AC07C 00040000 00000000
	s_clause 0x7                                               // 000000002224: BF850007
	global_load_d16_b16 v4, v[1:2], off                        // 000000002228: EE08007C 00000004 00000001
	global_load_d16_hi_b16 v4, v[9:10], off                    // 000000002234: EE08C07C 00000004 00000009
	global_load_d16_b16 v1, v[11:12], off                      // 000000002240: EE08007C 00000001 0000000B
	global_load_d16_hi_b16 v1, v[13:14], off                   // 00000000224C: EE08C07C 00000001 0000000D
	global_load_d16_b16 v2, v[67:68], off                      // 000000002258: EE08007C 00000002 00000043
	global_load_d16_hi_b16 v2, v[15:16], off                   // 000000002264: EE08C07C 00000002 0000000F
	global_load_d16_b16 v3, v[7:8], off                        // 000000002270: EE08007C 00000003 00000007
	global_load_d16_hi_b16 v3, v[69:70], off                   // 00000000227C: EE08C07C 00000003 00000045
	s_wait_alu depctr_va_sdst(0)                               // 000000002288: BF88F19F
	v_cndmask_b32_e64 v10, 0, v74, s13                         // 00000000228C: D501000A 00369480
	v_cndmask_b32_e64 v9, 0, v73, s13                          // 000000002294: D5010009 00369280
	v_lshlrev_b64_e32 v[7:8], 1, v[71:72]                      // 00000000229C: 3E0E8E81
	s_add_nc_u64 s[16:17], s[34:35], -10                       // 0000000022A0: A990CA22
	v_add_co_u32 v11, s14, 0x140, v5                           // 0000000022A4: D7000E0B 02020AFF 00000140
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000022B0: BF870003
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 0000000022B4: 3E121281
	s_wait_alu depctr_va_sdst(0)                               // 0000000022B8: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s14                  // 0000000022BC: D5207C0C 003A0C80
	v_add_co_u32 v7, s15, s38, v7                              // 0000000022C4: D7000F07 02020E26
	s_wait_alu depctr_sa_sdst(0)                               // 0000000022CC: BF88FF9E
	v_cmp_lt_i64_e64 s14, s[16:17], s[26:27]                   // 0000000022D0: D451000E 02003410
	s_wait_alu depctr_va_sdst(0)                               // 0000000022D8: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s15                 // 0000000022DC: D5207C08 003E1027
	s_add_nc_u64 s[16:17], s[34:35], -2                        // 0000000022E4: A990C222
	v_add_co_u32 v13, s15, 0x340, v5                           // 0000000022E8: D7000F0D 02020AFF 00000340
	s_wait_alu depctr_va_sdst(0)                               // 0000000022F4: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s15                  // 0000000022F8: D5207C0E 003E0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000002300: BF88FF9E
	v_cmp_lt_i64_e64 s15, s[16:17], s[26:27]                   // 000000002304: D451000F 02003410
	v_add_co_u32 v9, s16, s38, v9                              // 00000000230C: D7001009 02021226
	s_wait_alu depctr_va_sdst(0)                               // 000000002314: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s16               // 000000002318: D5207C0A 00421427
	s_add_nc_u64 s[18:19], s[34:35], -9                        // 000000002320: A992C922
	v_add_co_u32 v15, s16, 0x180, v5                           // 000000002324: D700100F 02020AFF 00000180
	v_cndmask_b32_e64 v12, 0, v12, s14                         // 000000002330: D501000C 003A1880
	v_cndmask_b32_e64 v11, 0, v11, s14                         // 000000002338: D501000B 003A1680
	s_wait_alu depctr_va_sdst(0)                               // 000000002340: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s16                  // 000000002344: D5207C10 00420C80
	s_wait_alu depctr_sa_sdst(0)                               // 00000000234C: BF88FF9E
	v_cmp_lt_i64_e64 s16, s[18:19], s[26:27]                   // 000000002350: D4510010 02003412
	v_cndmask_b32_e64 v14, 0, v14, s15                         // 000000002358: D501000E 003E1C80
	v_cndmask_b32_e64 v13, 0, v13, s15                         // 000000002360: D501000D 003E1A80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000002368: 3E161681
	s_add_nc_u64 s[18:19], s[34:35], -1                        // 00000000236C: A992C122
	s_add_nc_u64 s[44:45], s[34:35], -8                        // 000000002370: A9ACC822
	v_cndmask_b32_e64 v16, 0, v16, s16                         // 000000002374: D5010010 00422080
	v_cndmask_b32_e64 v15, 0, v15, s16                         // 00000000237C: D501000F 00421E80
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000002384: 3E1A1A81
	v_add_co_u32 v11, s17, s38, v11                            // 000000002388: D700110B 02021626
	s_wait_alu depctr_va_sdst(0)                               // 000000002390: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s17               // 000000002394: D5207C0C 00461827
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 00000000239C: 3E1E1E81
	v_add_co_u32 v67, s17, 0x380, v5                           // 0000000023A0: D7001143 02020AFF 00000380
	s_wait_alu depctr_va_sdst(0)                               // 0000000023AC: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s17                  // 0000000023B0: D5207C44 00460C80
	s_wait_alu depctr_sa_sdst(0)                               // 0000000023B8: BF88FF9E
	v_cmp_lt_i64_e64 s17, s[18:19], s[26:27]                   // 0000000023BC: D4510011 02003412
	v_add_co_u32 v69, s18, 0x1c0, v5                           // 0000000023C4: D7001245 02020AFF 000001C0
	v_add_co_u32 v13, s19, s38, v13                            // 0000000023D0: D700130D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 0000000023D8: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s18                  // 0000000023DC: D5207C46 004A0C80
	v_cmp_lt_i64_e64 s18, s[44:45], s[26:27]                   // 0000000023E4: D4510012 0200342C
	v_add_co_ci_u32_e64 v14, null, s39, v14, s19               // 0000000023EC: D5207C0E 004E1C27
	v_add_co_u32 v15, s19, s38, v15                            // 0000000023F4: D700130F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 0000000023FC: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s19               // 000000002400: D5207C10 004E2027
	v_add_co_u32 v71, s19, 0x3c0, v5                           // 000000002408: D7001347 02020AFF 000003C0
	v_cndmask_b32_e64 v68, 0, v68, s17                         // 000000002414: D5010044 00468880
	v_cndmask_b32_e64 v67, 0, v67, s17                         // 00000000241C: D5010043 00468680
	s_wait_alu depctr_va_sdst(0)                               // 000000002424: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s19                  // 000000002428: D5207C48 004E0C80
	v_cmp_lt_i64_e64 s19, s[34:35], s[26:27]                   // 000000002430: D4510013 02003422
	v_cndmask_b32_e64 v70, 0, v70, s18                         // 000000002438: D5010046 004A8C80
	v_cndmask_b32_e64 v69, 0, v69, s18                         // 000000002440: D5010045 004A8A80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 000000002448: 3E868681
	v_add_nc_u32_e32 v109, 0x400, v58                          // 00000000244C: 4ADA74FF 00000400
	s_add_nc_u64 s[30:31], s[30:31], -1                        // 000000002454: A99EC11E
	s_wait_alu depctr_va_sdst(0)                               // 000000002458: BF88F19F
	v_cndmask_b32_e64 v72, 0, v72, s19                         // 00000000245C: D5010048 004E9080
	v_cndmask_b32_e64 v71, 0, v71, s19                         // 000000002464: D5010047 004E8E80
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 00000000246C: 3E8A8A81
	v_add_co_u32 v67, s20, s38, v67                            // 000000002470: D7001443 02028626
	s_wait_alu depctr_va_sdst(0)                               // 000000002478: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 00000000247C: D5207C44 00528827
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 000000002484: 3E8E8E81
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002488: BF870004
	v_add_co_u32 v69, s20, s38, v69                            // 00000000248C: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002494: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 000000002498: D5207C46 00528C27
	v_add_co_u32 v73, s20, v5, 16                              // 0000000024A0: D7001449 02012105
	s_wait_alu depctr_va_sdst(0)                               // 0000000024A8: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 0000000024AC: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 0000000024B4: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 0000000024BC: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 0000000024C0: D5207C48 00529027
	v_add_co_u32 v79, s20, 0x210, v5                           // 0000000024C8: D700144F 02020AFF 00000210
	s_wait_alu depctr_va_sdst(0)                               // 0000000024D4: BF88F19F
	v_add_co_ci_u32_e64 v80, null, 0, v6, s20                  // 0000000024D8: D5207C50 00520C80
	v_cndmask_b32_e64 v74, 0, v74, s4                          // 0000000024E0: D501004A 00129480
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 0000000024E8: D5010049 00129280
	s_clause 0x7                                               // 0000000024F0: BF850007
	global_load_d16_b16 v75, v[7:8], off                       // 0000000024F4: EE08007C 0000004B 00000007
	global_load_d16_hi_b16 v75, v[9:10], off                   // 000000002500: EE08C07C 0000004B 00000009
	global_load_d16_b16 v76, v[11:12], off                     // 00000000250C: EE08007C 0000004C 0000000B
	global_load_d16_hi_b16 v76, v[13:14], off                  // 000000002518: EE08C07C 0000004C 0000000D
	global_load_d16_b16 v77, v[15:16], off                     // 000000002524: EE08007C 0000004D 0000000F
	global_load_d16_hi_b16 v77, v[67:68], off                  // 000000002530: EE08C07C 0000004D 00000043
	global_load_d16_b16 v78, v[69:70], off                     // 00000000253C: EE08007C 0000004E 00000045
	global_load_d16_hi_b16 v78, v[71:72], off                  // 000000002548: EE08C07C 0000004E 00000047
	v_cndmask_b32_e64 v10, 0, v80, s5                          // 000000002554: D501000A 0016A080
	v_cndmask_b32_e64 v9, 0, v79, s5                           // 00000000255C: D5010009 00169E80
	v_add_co_u32 v11, s20, 0x50, v5                            // 000000002564: D700140B 02020AFF 00000050
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 000000002570: 3E0E9281
	s_wait_alu depctr_va_sdst(0)                               // 000000002574: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 000000002578: D5207C0C 00520C80
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000002580: 3E121281
	v_cndmask_b32_e64 v11, 0, v11, s6                          // 000000002584: D501000B 001A1680
	s_add_nc_u64 s[34:35], s[34:35], 16                        // 00000000258C: A9A29022
	v_add_co_u32 v7, s20, s38, v7                              // 000000002590: D7001407 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002598: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 00000000259C: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x250, v5                           // 0000000025A4: D700140D 02020AFF 00000250
	s_wait_alu depctr_va_sdst(0)                               // 0000000025B0: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 0000000025B4: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 0000000025BC: D7001409 02021226
	v_cndmask_b32_e64 v12, 0, v12, s6                          // 0000000025C4: D501000C 001A1880
	s_wait_alu depctr_va_sdst(0)                               // 0000000025CC: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 0000000025D0: D5207C0A 00521427
	v_add_co_u32 v15, s20, 0x90, v5                            // 0000000025D8: D700140F 02020AFF 00000090
	s_wait_alu depctr_va_sdst(0)                               // 0000000025E4: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 0000000025E8: D5207C10 00520C80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 0000000025F0: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s7                          // 0000000025F4: D501000E 001E1C80
	v_cndmask_b32_e64 v13, 0, v13, s7                          // 0000000025FC: D501000D 001E1A80
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002604: BF870004
	v_cndmask_b32_e64 v16, 0, v16, s8                          // 000000002608: D5010010 00222080
	v_cndmask_b32_e64 v15, 0, v15, s8                          // 000000002610: D501000F 00221E80
	s_cmp_lg_u64 s[30:31], 0                                   // 000000002618: BF11801E
	v_add_co_u32 v11, s20, s38, v11                            // 00000000261C: D700140B 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000002624: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 000000002628: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 00000000262C: D5207C0C 00521827
	v_add_co_u32 v67, s20, 0x290, v5                           // 000000002634: D7001443 02020AFF 00000290
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 000000002640: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002644: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 000000002648: D5207C44 00520C80
	v_add_co_u32 v69, s20, 0xd0, v5                            // 000000002650: D7001445 02020AFF 000000D0
	s_wait_alu depctr_va_sdst(0)                               // 00000000265C: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 000000002660: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 000000002668: D700140D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002670: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 000000002674: D5207C0E 00521C27
	v_add_co_u32 v15, s20, s38, v15                            // 00000000267C: D700140F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002684: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 000000002688: D5207C10 00522027
	v_add_co_u32 v71, s20, 0x2d0, v5                           // 000000002690: D7001447 02020AFF 000002D0
	v_cndmask_b32_e64 v68, 0, v68, s9                          // 00000000269C: D5010044 00268880
	v_cndmask_b32_e64 v67, 0, v67, s9                          // 0000000026A4: D5010043 00268680
	s_wait_alu depctr_va_sdst(0)                               // 0000000026AC: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 0000000026B0: D5207C48 00520C80
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 0000000026B8: D5010046 002A8C80
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 0000000026C0: D5010045 002A8A80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 0000000026C8: 3E868681
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)// 0000000026CC: BF870244
	v_cndmask_b32_e64 v72, 0, v72, s11                         // 0000000026D0: D5010048 002E9080
	v_cndmask_b32_e64 v71, 0, v71, s11                         // 0000000026D8: D5010047 002E8E80
	s_add_nc_u64 s[40:41], s[40:41], 0x400                     // 0000000026E0: A9A8FF28 00000400
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 0000000026E8: 3E8A8A81
	v_add_co_u32 v67, s20, s38, v67                            // 0000000026EC: D7001443 02028626
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 0000000026F4: BF870233
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 0000000026F8: 3E8E8E81
	s_wait_alu depctr_va_sdst(0)                               // 0000000026FC: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000002700: D5207C44 00528827
	v_add_co_u32 v69, s20, s38, v69                            // 000000002708: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002710: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 000000002714: D5207C46 00528C27
	v_add_co_u32 v73, s20, 0x110, v5                           // 00000000271C: D7001449 02020AFF 00000110
	s_wait_alu depctr_va_sdst(0)                               // 000000002728: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 00000000272C: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 000000002734: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 00000000273C: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 000000002740: D5207C48 00529027
	v_add_co_u32 v83, s20, 0x310, v5                           // 000000002748: D7001453 02020AFF 00000310
	s_wait_alu depctr_va_sdst(0)                               // 000000002754: BF88F19F
	v_add_co_ci_u32_e64 v84, null, 0, v6, s20                  // 000000002758: D5207C54 00520C80
	v_cndmask_b32_e64 v74, 0, v74, s12                         // 000000002760: D501004A 00329480
	v_cndmask_b32_e64 v73, 0, v73, s12                         // 000000002768: D5010049 00329280
	s_clause 0x7                                               // 000000002770: BF850007
	global_load_d16_b16 v79, v[7:8], off                       // 000000002774: EE08007C 0000004F 00000007
	global_load_d16_hi_b16 v79, v[9:10], off                   // 000000002780: EE08C07C 0000004F 00000009
	global_load_d16_b16 v80, v[11:12], off                     // 00000000278C: EE08007C 00000050 0000000B
	global_load_d16_hi_b16 v80, v[13:14], off                  // 000000002798: EE08C07C 00000050 0000000D
	global_load_d16_b16 v81, v[15:16], off                     // 0000000027A4: EE08007C 00000051 0000000F
	global_load_d16_hi_b16 v81, v[67:68], off                  // 0000000027B0: EE08C07C 00000051 00000043
	global_load_d16_b16 v82, v[69:70], off                     // 0000000027BC: EE08007C 00000052 00000045
	global_load_d16_hi_b16 v82, v[71:72], off                  // 0000000027C8: EE08C07C 00000052 00000047
	v_cndmask_b32_e64 v10, 0, v84, s13                         // 0000000027D4: D501000A 0036A880
	v_cndmask_b32_e64 v9, 0, v83, s13                          // 0000000027DC: D5010009 0036A680
	v_add_co_u32 v11, s20, 0x150, v5                           // 0000000027E4: D700140B 02020AFF 00000150
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 0000000027F0: 3E0E9281
	s_wait_alu depctr_va_sdst(0)                               // 0000000027F4: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 0000000027F8: D5207C0C 00520C80
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000002800: 3E121281
	v_cndmask_b32_e64 v11, 0, v11, s14                         // 000000002804: D501000B 003A1680
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000280C: BF870004
	v_add_co_u32 v7, s20, s38, v7                              // 000000002810: D7001407 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002818: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 00000000281C: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x350, v5                           // 000000002824: D700140D 02020AFF 00000350
	s_wait_alu depctr_va_sdst(0)                               // 000000002830: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 000000002834: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 00000000283C: D7001409 02021226
	v_cndmask_b32_e64 v12, 0, v12, s14                         // 000000002844: D501000C 003A1880
	s_wait_alu depctr_va_sdst(0)                               // 00000000284C: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 000000002850: D5207C0A 00521427
	v_add_co_u32 v15, s20, 0x190, v5                           // 000000002858: D700140F 02020AFF 00000190
	s_wait_alu depctr_va_sdst(0)                               // 000000002864: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 000000002868: D5207C10 00520C80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000002870: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s15                         // 000000002874: D501000E 003E1C80
	v_cndmask_b32_e64 v13, 0, v13, s15                         // 00000000287C: D501000D 003E1A80
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000002884: BF870234
	v_cndmask_b32_e64 v16, 0, v16, s16                         // 000000002888: D5010010 00422080
	v_cndmask_b32_e64 v15, 0, v15, s16                         // 000000002890: D501000F 00421E80
	v_add_co_u32 v11, s20, s38, v11                            // 000000002898: D700140B 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 0000000028A0: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 0000000028A4: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 0000000028A8: D5207C0C 00521827
	v_add_co_u32 v67, s20, 0x390, v5                           // 0000000028B0: D7001443 02020AFF 00000390
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 0000000028BC: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 0000000028C0: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 0000000028C4: D5207C44 00520C80
	v_add_co_u32 v69, s20, 0x1d0, v5                           // 0000000028CC: D7001445 02020AFF 000001D0
	s_wait_alu depctr_va_sdst(0)                               // 0000000028D8: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 0000000028DC: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 0000000028E4: D700140D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 0000000028EC: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 0000000028F0: D5207C0E 00521C27
	v_add_co_u32 v15, s20, s38, v15                            // 0000000028F8: D700140F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002900: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 000000002904: D5207C10 00522027
	v_add_co_u32 v71, s20, 0x3d0, v5                           // 00000000290C: D7001447 02020AFF 000003D0
	v_cndmask_b32_e64 v68, 0, v68, s17                         // 000000002918: D5010044 00468880
	v_cndmask_b32_e64 v67, 0, v67, s17                         // 000000002920: D5010043 00468680
	s_wait_alu depctr_va_sdst(0)                               // 000000002928: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 00000000292C: D5207C48 00520C80
	v_cndmask_b32_e64 v70, 0, v70, s18                         // 000000002934: D5010046 004A8C80
	v_cndmask_b32_e64 v69, 0, v69, s18                         // 00000000293C: D5010045 004A8A80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 000000002944: 3E868681
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000002948: BF870224
	v_cndmask_b32_e64 v72, 0, v72, s19                         // 00000000294C: D5010048 004E9080
	v_cndmask_b32_e64 v71, 0, v71, s19                         // 000000002954: D5010047 004E8E80
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 00000000295C: 3E8A8A81
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002960: BF870194
	v_add_co_u32 v67, s20, s38, v67                            // 000000002964: D7001443 02028626
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 00000000296C: 3E8E8E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002970: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000002974: D5207C44 00528827
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000297C: BF870004
	v_add_co_u32 v69, s20, s38, v69                            // 000000002980: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002988: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 00000000298C: D5207C46 00528C27
	v_add_co_u32 v73, s20, v5, 32                              // 000000002994: D7001449 02014105
	s_wait_alu depctr_va_sdst(0)                               // 00000000299C: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 0000000029A0: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 0000000029A8: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 0000000029B0: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 0000000029B4: D5207C48 00529027
	v_add_co_u32 v87, s20, 0x220, v5                           // 0000000029BC: D7001457 02020AFF 00000220
	s_wait_alu depctr_va_sdst(0)                               // 0000000029C8: BF88F19F
	v_add_co_ci_u32_e64 v88, null, 0, v6, s20                  // 0000000029CC: D5207C58 00520C80
	v_cndmask_b32_e64 v74, 0, v74, s4                          // 0000000029D4: D501004A 00129480
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 0000000029DC: D5010049 00129280
	s_clause 0x7                                               // 0000000029E4: BF850007
	global_load_d16_b16 v83, v[7:8], off                       // 0000000029E8: EE08007C 00000053 00000007
	global_load_d16_hi_b16 v83, v[9:10], off                   // 0000000029F4: EE08C07C 00000053 00000009
	global_load_d16_b16 v84, v[11:12], off                     // 000000002A00: EE08007C 00000054 0000000B
	global_load_d16_hi_b16 v84, v[13:14], off                  // 000000002A0C: EE08C07C 00000054 0000000D
	global_load_d16_b16 v85, v[15:16], off                     // 000000002A18: EE08007C 00000055 0000000F
	global_load_d16_hi_b16 v85, v[67:68], off                  // 000000002A24: EE08C07C 00000055 00000043
	global_load_d16_b16 v86, v[69:70], off                     // 000000002A30: EE08007C 00000056 00000045
	global_load_d16_hi_b16 v86, v[71:72], off                  // 000000002A3C: EE08C07C 00000056 00000047
	v_cndmask_b32_e64 v10, 0, v88, s5                          // 000000002A48: D501000A 0016B080
	v_cndmask_b32_e64 v9, 0, v87, s5                           // 000000002A50: D5010009 0016AE80
	v_add_co_u32 v11, s20, 0x60, v5                            // 000000002A58: D700140B 02020AFF 00000060
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 000000002A64: 3E0E9281
	s_wait_alu depctr_va_sdst(0)                               // 000000002A68: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 000000002A6C: D5207C0C 00520C80
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000002A74: 3E121281
	v_cndmask_b32_e64 v11, 0, v11, s6                          // 000000002A78: D501000B 001A1680
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002A80: BF870004
	v_add_co_u32 v7, s20, s38, v7                              // 000000002A84: D7001407 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002A8C: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 000000002A90: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x260, v5                           // 000000002A98: D700140D 02020AFF 00000260
	s_wait_alu depctr_va_sdst(0)                               // 000000002AA4: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 000000002AA8: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 000000002AB0: D7001409 02021226
	v_cndmask_b32_e64 v12, 0, v12, s6                          // 000000002AB8: D501000C 001A1880
	s_wait_alu depctr_va_sdst(0)                               // 000000002AC0: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 000000002AC4: D5207C0A 00521427
	v_add_co_u32 v15, s20, 0xa0, v5                            // 000000002ACC: D700140F 02020AFF 000000A0
	s_wait_alu depctr_va_sdst(0)                               // 000000002AD8: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 000000002ADC: D5207C10 00520C80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000002AE4: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s7                          // 000000002AE8: D501000E 001E1C80
	v_cndmask_b32_e64 v13, 0, v13, s7                          // 000000002AF0: D501000D 001E1A80
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002AF8: BF870004
	v_cndmask_b32_e64 v16, 0, v16, s8                          // 000000002AFC: D5010010 00222080
	v_cndmask_b32_e64 v15, 0, v15, s8                          // 000000002B04: D501000F 00221E80
	s_wait_loadcnt 0x1e                                        // 000000002B0C: BFC0001E
	v_cndmask_b16 v4.l, 0, v4.l, s4                            // 000000002B10: D65D0004 00120880
	v_add_co_u32 v11, s20, s38, v11                            // 000000002B18: D700140B 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000002B20: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 000000002B24: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 000000002B28: D5207C0C 00521827
	v_add_co_u32 v67, s20, 0x2a0, v5                           // 000000002B30: D7001443 02020AFF 000002A0
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 000000002B3C: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002B40: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 000000002B44: D5207C44 00520C80
	v_add_co_u32 v69, s20, 0xe0, v5                            // 000000002B4C: D7001445 02020AFF 000000E0
	s_wait_alu depctr_va_sdst(0)                               // 000000002B58: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 000000002B5C: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 000000002B64: D700140D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002B6C: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 000000002B70: D5207C0E 00521C27
	v_add_co_u32 v15, s20, s38, v15                            // 000000002B78: D700140F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002B80: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 000000002B84: D5207C10 00522027
	v_add_co_u32 v71, s20, 0x2e0, v5                           // 000000002B8C: D7001447 02020AFF 000002E0
	v_cndmask_b32_e64 v68, 0, v68, s9                          // 000000002B98: D5010044 00268880
	v_cndmask_b32_e64 v67, 0, v67, s9                          // 000000002BA0: D5010043 00268680
	s_wait_alu depctr_va_sdst(0)                               // 000000002BA8: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 000000002BAC: D5207C48 00520C80
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 000000002BB4: D5010046 002A8C80
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 000000002BBC: D5010045 002A8A80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 000000002BC4: 3E868681
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002BC8: BF870004
	v_cndmask_b32_e64 v72, 0, v72, s11                         // 000000002BCC: D5010048 002E9080
	v_cndmask_b32_e64 v71, 0, v71, s11                         // 000000002BD4: D5010047 002E8E80
	v_cndmask_b16 v4.h, 0, v4.h, s5                            // 000000002BDC: D65D5004 00160880
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 000000002BE4: 3E8A8A81
	s_wait_loadcnt 0x1c                                        // 000000002BE8: BFC0001C
	v_cndmask_b16 v1.l, 0, v1.l, s6                            // 000000002BEC: D65D0001 001A0280
	v_add_co_u32 v67, s20, s38, v67                            // 000000002BF4: D7001443 02028626
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 000000002BFC: 3E8E8E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002C00: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000002C04: D5207C44 00528827
	v_add_co_u32 v69, s20, s38, v69                            // 000000002C0C: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002C14: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 000000002C18: D5207C46 00528C27
	v_add_co_u32 v73, s20, 0x120, v5                           // 000000002C20: D7001449 02020AFF 00000120
	s_wait_alu depctr_va_sdst(0)                               // 000000002C2C: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 000000002C30: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 000000002C38: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002C40: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 000000002C44: D5207C48 00529027
	v_add_co_u32 v91, s20, 0x320, v5                           // 000000002C4C: D700145B 02020AFF 00000320
	s_wait_alu depctr_va_sdst(0)                               // 000000002C58: BF88F19F
	v_add_co_ci_u32_e64 v92, null, 0, v6, s20                  // 000000002C5C: D5207C5C 00520C80
	v_cndmask_b32_e64 v74, 0, v74, s12                         // 000000002C64: D501004A 00329480
	v_cndmask_b32_e64 v73, 0, v73, s12                         // 000000002C6C: D5010049 00329280
	s_clause 0x7                                               // 000000002C74: BF850007
	global_load_d16_b16 v87, v[7:8], off                       // 000000002C78: EE08007C 00000057 00000007
	global_load_d16_hi_b16 v87, v[9:10], off                   // 000000002C84: EE08C07C 00000057 00000009
	global_load_d16_b16 v88, v[11:12], off                     // 000000002C90: EE08007C 00000058 0000000B
	global_load_d16_hi_b16 v88, v[13:14], off                  // 000000002C9C: EE08C07C 00000058 0000000D
	global_load_d16_b16 v89, v[15:16], off                     // 000000002CA8: EE08007C 00000059 0000000F
	global_load_d16_hi_b16 v89, v[67:68], off                  // 000000002CB4: EE08C07C 00000059 00000043
	global_load_d16_b16 v90, v[69:70], off                     // 000000002CC0: EE08007C 0000005A 00000045
	global_load_d16_hi_b16 v90, v[71:72], off                  // 000000002CCC: EE08C07C 0000005A 00000047
	v_cndmask_b32_e64 v10, 0, v92, s13                         // 000000002CD8: D501000A 0036B880
	v_cndmask_b32_e64 v9, 0, v91, s13                          // 000000002CE0: D5010009 0036B680
	v_add_co_u32 v11, s20, 0x160, v5                           // 000000002CE8: D700140B 02020AFF 00000160
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 000000002CF4: 3E0E9281
	s_wait_alu depctr_va_sdst(0)                               // 000000002CF8: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 000000002CFC: D5207C0C 00520C80
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000002D04: 3E121281
	v_cndmask_b32_e64 v11, 0, v11, s14                         // 000000002D08: D501000B 003A1680
	v_cndmask_b16 v1.h, 0, v1.h, s7                            // 000000002D10: D65D5001 001E0280
	v_add_co_u32 v7, s20, s38, v7                              // 000000002D18: D7001407 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002D20: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 000000002D24: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x360, v5                           // 000000002D2C: D700140D 02020AFF 00000360
	s_wait_alu depctr_va_sdst(0)                               // 000000002D38: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 000000002D3C: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 000000002D44: D7001409 02021226
	v_cndmask_b32_e64 v12, 0, v12, s14                         // 000000002D4C: D501000C 003A1880
	s_wait_alu depctr_va_sdst(0)                               // 000000002D54: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 000000002D58: D5207C0A 00521427
	v_add_co_u32 v15, s20, 0x1a0, v5                           // 000000002D60: D700140F 02020AFF 000001A0
	s_wait_alu depctr_va_sdst(0)                               // 000000002D6C: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 000000002D70: D5207C10 00520C80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000002D78: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s15                         // 000000002D7C: D501000E 003E1C80
	v_cndmask_b32_e64 v13, 0, v13, s15                         // 000000002D84: D501000D 003E1A80
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002D8C: BF870004
	v_cndmask_b32_e64 v16, 0, v16, s16                         // 000000002D90: D5010010 00422080
	v_cndmask_b32_e64 v15, 0, v15, s16                         // 000000002D98: D501000F 00421E80
	s_wait_loadcnt 0x22                                        // 000000002DA0: BFC00022
	v_cndmask_b16 v2.l, 0, v2.l, s8                            // 000000002DA4: D65D0002 00220480
	v_add_co_u32 v11, s20, s38, v11                            // 000000002DAC: D700140B 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000002DB4: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 000000002DB8: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 000000002DBC: D5207C0C 00521827
	v_add_co_u32 v67, s20, 0x3a0, v5                           // 000000002DC4: D7001443 02020AFF 000003A0
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 000000002DD0: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002DD4: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 000000002DD8: D5207C44 00520C80
	v_add_co_u32 v69, s20, 0x1e0, v5                           // 000000002DE0: D7001445 02020AFF 000001E0
	s_wait_alu depctr_va_sdst(0)                               // 000000002DEC: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 000000002DF0: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 000000002DF8: D700140D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002E00: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 000000002E04: D5207C0E 00521C27
	v_add_co_u32 v15, s20, s38, v15                            // 000000002E0C: D700140F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002E14: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 000000002E18: D5207C10 00522027
	v_add_co_u32 v71, s20, 0x3e0, v5                           // 000000002E20: D7001447 02020AFF 000003E0
	v_cndmask_b32_e64 v68, 0, v68, s17                         // 000000002E2C: D5010044 00468880
	v_cndmask_b32_e64 v67, 0, v67, s17                         // 000000002E34: D5010043 00468680
	s_wait_alu depctr_va_sdst(0)                               // 000000002E3C: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 000000002E40: D5207C48 00520C80
	v_cndmask_b32_e64 v70, 0, v70, s18                         // 000000002E48: D5010046 004A8C80
	v_cndmask_b32_e64 v69, 0, v69, s18                         // 000000002E50: D5010045 004A8A80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 000000002E58: 3E868681
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002E5C: BF870004
	v_cndmask_b32_e64 v72, 0, v72, s19                         // 000000002E60: D5010048 004E9080
	v_cndmask_b32_e64 v71, 0, v71, s19                         // 000000002E68: D5010047 004E8E80
	v_cndmask_b16 v2.h, 0, v2.h, s9                            // 000000002E70: D65D5002 00260480
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 000000002E78: 3E8A8A81
	s_wait_loadcnt 0x20                                        // 000000002E7C: BFC00020
	v_cndmask_b16 v3.l, 0, v3.l, s10                           // 000000002E80: D65D0003 002A0680
	v_add_co_u32 v67, s20, s38, v67                            // 000000002E88: D7001443 02028626
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 000000002E90: 3E8E8E81
	s_wait_alu depctr_va_sdst(0)                               // 000000002E94: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000002E98: D5207C44 00528827
	v_add_co_u32 v69, s20, s38, v69                            // 000000002EA0: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 000000002EA8: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 000000002EAC: D5207C46 00528C27
	v_add_co_u32 v73, s20, v5, 48                              // 000000002EB4: D7001449 02016105
	s_wait_alu depctr_va_sdst(0)                               // 000000002EBC: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 000000002EC0: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 000000002EC8: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002ED0: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 000000002ED4: D5207C48 00529027
	v_add_co_u32 v95, s20, 0x230, v5                           // 000000002EDC: D700145F 02020AFF 00000230
	s_wait_alu depctr_va_sdst(0)                               // 000000002EE8: BF88F19F
	v_add_co_ci_u32_e64 v96, null, 0, v6, s20                  // 000000002EEC: D5207C60 00520C80
	v_cndmask_b32_e64 v74, 0, v74, s4                          // 000000002EF4: D501004A 00129480
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000002EFC: D5010049 00129280
	s_clause 0x7                                               // 000000002F04: BF850007
	global_load_d16_b16 v91, v[7:8], off                       // 000000002F08: EE08007C 0000005B 00000007
	global_load_d16_hi_b16 v91, v[9:10], off                   // 000000002F14: EE08C07C 0000005B 00000009
	global_load_d16_b16 v92, v[11:12], off                     // 000000002F20: EE08007C 0000005C 0000000B
	global_load_d16_hi_b16 v92, v[13:14], off                  // 000000002F2C: EE08C07C 0000005C 0000000D
	global_load_d16_b16 v93, v[15:16], off                     // 000000002F38: EE08007C 0000005D 0000000F
	global_load_d16_hi_b16 v93, v[67:68], off                  // 000000002F44: EE08C07C 0000005D 00000043
	global_load_d16_b16 v94, v[69:70], off                     // 000000002F50: EE08007C 0000005E 00000045
	global_load_d16_hi_b16 v94, v[71:72], off                  // 000000002F5C: EE08C07C 0000005E 00000047
	v_cndmask_b32_e64 v10, 0, v96, s5                          // 000000002F68: D501000A 0016C080
	v_cndmask_b32_e64 v9, 0, v95, s5                           // 000000002F70: D5010009 0016BE80
	v_add_co_u32 v11, s20, 0x70, v5                            // 000000002F78: D700140B 02020AFF 00000070
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 000000002F84: 3E0E9281
	s_wait_alu depctr_va_sdst(0)                               // 000000002F88: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 000000002F8C: D5207C0C 00520C80
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000002F94: 3E121281
	v_cndmask_b32_e64 v11, 0, v11, s6                          // 000000002F98: D501000B 001A1680
	v_cndmask_b16 v3.h, 0, v3.h, s11                           // 000000002FA0: D65D5003 002E0680
	v_add_co_u32 v7, s20, s38, v7                              // 000000002FA8: D7001407 02020E26
	s_wait_alu depctr_va_sdst(0)                               // 000000002FB0: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 000000002FB4: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x270, v5                           // 000000002FBC: D700140D 02020AFF 00000270
	v_cndmask_b32_e64 v12, 0, v12, s6                          // 000000002FC8: D501000C 001A1880
	s_wait_alu depctr_va_sdst(0)                               // 000000002FD0: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 000000002FD4: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 000000002FDC: D7001409 02021226
	s_wait_alu depctr_va_sdst(0)                               // 000000002FE4: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 000000002FE8: D5207C0A 00521427
	v_add_co_u32 v15, s20, 0xb0, v5                            // 000000002FF0: D700140F 02020AFF 000000B0
	s_wait_alu depctr_va_sdst(0)                               // 000000002FFC: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 000000003000: D5207C10 00520C80
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000003008: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s7                          // 00000000300C: D501000E 001E1C80
	v_cndmask_b32_e64 v13, 0, v13, s7                          // 000000003014: D501000D 001E1A80
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 00000000301C: BF870234
	v_cndmask_b32_e64 v16, 0, v16, s8                          // 000000003020: D5010010 00222080
	v_cndmask_b32_e64 v15, 0, v15, s8                          // 000000003028: D501000F 00221E80
	v_add_co_u32 v11, s20, s38, v11                            // 000000003030: D700140B 02021626
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000003038: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 00000000303C: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 000000003040: D5207C0C 00521827
	v_add_co_u32 v67, s20, 0x2b0, v5                           // 000000003048: D7001443 02020AFF 000002B0
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 000000003054: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 000000003058: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 00000000305C: D5207C44 00520C80
	v_add_co_u32 v69, s20, 0xf0, v5                            // 000000003064: D7001445 02020AFF 000000F0
	s_wait_alu depctr_va_sdst(0)                               // 000000003070: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 000000003074: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 00000000307C: D700140D 02021A26
	s_wait_alu depctr_va_sdst(0)                               // 000000003084: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 000000003088: D5207C0E 00521C27
	v_cndmask_b32_e64 v68, 0, v68, s9                          // 000000003090: D5010044 00268880
	v_cndmask_b32_e64 v67, 0, v67, s9                          // 000000003098: D5010043 00268680
	v_add_co_u32 v15, s20, s38, v15                            // 0000000030A0: D700140F 02021E26
	v_cndmask_b32_e64 v70, 0, v70, s10                         // 0000000030A8: D5010046 002A8C80
	v_cndmask_b32_e64 v69, 0, v69, s10                         // 0000000030B0: D5010045 002A8A80
	s_wait_alu depctr_va_sdst(0)                               // 0000000030B8: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 0000000030BC: D5207C10 00522027
	v_add_co_u32 v71, s20, 0x2f0, v5                           // 0000000030C4: D7001447 02020AFF 000002F0
	s_wait_alu depctr_va_sdst(0)                               // 0000000030D0: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 0000000030D4: D5207C48 00520C80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 0000000030DC: 3E868681
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 0000000030E0: 3E8A8A81
	v_cndmask_b32_e64 v71, 0, v71, s11                         // 0000000030E4: D5010047 002E8E80
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000030EC: BF870214
	v_cndmask_b32_e64 v72, 0, v72, s11                         // 0000000030F0: D5010048 002E9080
	v_add_co_u32 v67, s20, s38, v67                            // 0000000030F8: D7001443 02028626
	s_wait_alu depctr_va_sdst(0)                               // 000000003100: BF88F19F
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000003104: D5207C44 00528827
	v_add_co_u32 v69, s20, s38, v69                            // 00000000310C: D7001445 02028A26
	v_lshlrev_b64_e32 v[71:72], 1, v[71:72]                    // 000000003114: 3E8E8E81
	s_wait_alu depctr_va_sdst(0)                               // 000000003118: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 00000000311C: D5207C46 00528C27
	v_add_co_u32 v73, s20, 0x130, v5                           // 000000003124: D7001449 02020AFF 00000130
	s_wait_alu depctr_va_sdst(0)                               // 000000003130: BF88F19F
	v_add_co_ci_u32_e64 v74, null, 0, v6, s20                  // 000000003134: D5207C4A 00520C80
	v_add_co_u32 v71, s20, s38, v71                            // 00000000313C: D7001447 02028E26
	s_wait_alu depctr_va_sdst(0)                               // 000000003144: BF88F19F
	v_add_co_ci_u32_e64 v72, null, s39, v72, s20               // 000000003148: D5207C48 00529027
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003150: BF870003
	v_cndmask_b32_e64 v74, 0, v74, s12                         // 000000003154: D501004A 00329480
	v_cndmask_b32_e64 v73, 0, v73, s12                         // 00000000315C: D5010049 00329280
	v_add_co_u32 v99, s20, 0x330, v5                           // 000000003164: D7001463 02020AFF 00000330
	s_wait_alu depctr_va_sdst(0)                               // 000000003170: BF88F19F
	v_add_co_ci_u32_e64 v100, null, 0, v6, s20                 // 000000003174: D5207C64 00520C80
	s_clause 0x7                                               // 00000000317C: BF850007
	global_load_d16_b16 v95, v[7:8], off                       // 000000003180: EE08007C 0000005F 00000007
	global_load_d16_hi_b16 v95, v[9:10], off                   // 00000000318C: EE08C07C 0000005F 00000009
	global_load_d16_b16 v96, v[11:12], off                     // 000000003198: EE08007C 00000060 0000000B
	global_load_d16_hi_b16 v96, v[13:14], off                  // 0000000031A4: EE08C07C 00000060 0000000D
	global_load_d16_b16 v97, v[15:16], off                     // 0000000031B0: EE08007C 00000061 0000000F
	global_load_d16_hi_b16 v97, v[67:68], off                  // 0000000031BC: EE08C07C 00000061 00000043
	global_load_d16_b16 v98, v[69:70], off                     // 0000000031C8: EE08007C 00000062 00000045
	global_load_d16_hi_b16 v98, v[71:72], off                  // 0000000031D4: EE08C07C 00000062 00000047
	v_lshlrev_b64_e32 v[7:8], 1, v[73:74]                      // 0000000031E0: 3E0E9281
	v_cndmask_b32_e64 v9, 0, v99, s13                          // 0000000031E4: D5010009 0036C680
	v_cndmask_b32_e64 v10, 0, v100, s13                        // 0000000031EC: D501000A 0036C880
	v_add_co_u32 v11, s20, 0x170, v5                           // 0000000031F4: D700140B 02020AFF 00000170
	s_wait_alu depctr_va_sdst(0)                               // 000000003200: BF88F19F
	v_add_co_ci_u32_e64 v12, null, 0, v6, s20                  // 000000003204: D5207C0C 00520C80
	v_add_co_u32 v7, s20, s38, v7                              // 00000000320C: D7001407 02020E26
	v_lshlrev_b64_e32 v[9:10], 1, v[9:10]                      // 000000003214: 3E121281
	s_wait_alu depctr_va_sdst(0)                               // 000000003218: BF88F19F
	v_add_co_ci_u32_e64 v8, null, s39, v8, s20                 // 00000000321C: D5207C08 00521027
	v_add_co_u32 v13, s20, 0x370, v5                           // 000000003224: D700140D 02020AFF 00000370
	v_cndmask_b32_e64 v12, 0, v12, s14                         // 000000003230: D501000C 003A1880
	v_cndmask_b32_e64 v11, 0, v11, s14                         // 000000003238: D501000B 003A1680
	s_wait_alu depctr_va_sdst(0)                               // 000000003240: BF88F19F
	v_add_co_ci_u32_e64 v14, null, 0, v6, s20                  // 000000003244: D5207C0E 00520C80
	v_add_co_u32 v9, s20, s38, v9                              // 00000000324C: D7001409 02021226
	s_wait_alu depctr_va_sdst(0)                               // 000000003254: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s39, v10, s20               // 000000003258: D5207C0A 00521427
	v_lshlrev_b64_e32 v[11:12], 1, v[11:12]                    // 000000003260: 3E161681
	v_cndmask_b32_e64 v14, 0, v14, s15                         // 000000003264: D501000E 003E1C80
	v_cndmask_b32_e64 v13, 0, v13, s15                         // 00000000326C: D501000D 003E1A80
	v_add_co_u32 v15, s20, 0x1b0, v5                           // 000000003274: D700140F 02020AFF 000001B0
	s_wait_alu depctr_va_sdst(0)                               // 000000003280: BF88F19F
	v_add_co_ci_u32_e64 v16, null, 0, v6, s20                  // 000000003284: D5207C10 00520C80
	v_add_co_u32 v67, s20, 0x3b0, v5                           // 00000000328C: D7001443 02020AFF 000003B0
	v_lshlrev_b64_e32 v[13:14], 1, v[13:14]                    // 000000003298: 3E1A1A81
	s_wait_alu depctr_va_sdst(0)                               // 00000000329C: BF88F19F
	v_add_co_ci_u32_e64 v68, null, 0, v6, s20                  // 0000000032A0: D5207C44 00520C80
	v_add_co_u32 v11, s20, s38, v11                            // 0000000032A8: D700140B 02021626
	v_cndmask_b32_e64 v16, 0, v16, s16                         // 0000000032B0: D5010010 00422080
	v_cndmask_b32_e64 v15, 0, v15, s16                         // 0000000032B8: D501000F 00421E80
	s_wait_alu depctr_va_sdst(0)                               // 0000000032C0: BF88F19F
	v_add_co_ci_u32_e64 v12, null, s39, v12, s20               // 0000000032C4: D5207C0C 00521827
	v_add_co_u32 v69, s20, 0x1f0, v5                           // 0000000032CC: D7001445 02020AFF 000001F0
	s_wait_alu depctr_va_sdst(0)                               // 0000000032D8: BF88F19F
	v_add_co_ci_u32_e64 v70, null, 0, v6, s20                  // 0000000032DC: D5207C46 00520C80
	v_add_co_u32 v13, s20, s38, v13                            // 0000000032E4: D700140D 02021A26
	v_lshlrev_b64_e32 v[15:16], 1, v[15:16]                    // 0000000032EC: 3E1E1E81
	s_wait_alu depctr_va_sdst(0)                               // 0000000032F0: BF88F19F
	v_add_co_ci_u32_e64 v14, null, s39, v14, s20               // 0000000032F4: D5207C0E 00521C27
	v_add_co_u32 v71, s20, 0x3f0, v5                           // 0000000032FC: D7001447 02020AFF 000003F0
	v_cndmask_b32_e64 v70, 0, v70, s18                         // 000000003308: D5010046 004A8C80
	v_cndmask_b32_e64 v69, 0, v69, s18                         // 000000003310: D5010045 004A8A80
	s_wait_alu depctr_va_sdst(0)                               // 000000003318: BF88F19F
	v_add_co_ci_u32_e64 v72, null, 0, v6, s20                  // 00000000331C: D5207C48 00520C80
	v_cndmask_b32_e64 v68, 0, v68, s17                         // 000000003324: D5010044 00468880
	v_cndmask_b32_e64 v67, 0, v67, s17                         // 00000000332C: D5010043 00468680
	v_add_co_u32 v5, s20, s38, v15                             // 000000003334: D7001405 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 00000000333C: BF88F19F
	v_add_co_ci_u32_e64 v6, null, s39, v16, s20                // 000000003340: D5207C06 00522027
	v_lshlrev_b64_e32 v[15:16], 1, v[69:70]                    // 000000003348: 3E1E8A81
	v_cndmask_b32_e64 v70, 0, v72, s19                         // 00000000334C: D5010046 004E9080
	v_cndmask_b32_e64 v69, 0, v71, s19                         // 000000003354: D5010045 004E8E80
	v_lshlrev_b64_e32 v[67:68], 1, v[67:68]                    // 00000000335C: 3E868681
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000003360: BF870112
	v_lshlrev_b64_e32 v[69:70], 1, v[69:70]                    // 000000003364: 3E8A8A81
	v_add_co_u32 v67, s20, s38, v67                            // 000000003368: D7001443 02028626
	s_wait_alu depctr_va_sdst(0)                               // 000000003370: BF88F19F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003374: BF870003
	v_add_co_ci_u32_e64 v68, null, s39, v68, s20               // 000000003378: D5207C44 00528827
	v_add_co_u32 v15, s20, s38, v15                            // 000000003380: D700140F 02021E26
	s_wait_alu depctr_va_sdst(0)                               // 000000003388: BF88F19F
	v_add_co_ci_u32_e64 v16, null, s39, v16, s20               // 00000000338C: D5207C10 00522027
	v_add_co_u32 v69, s20, s38, v69                            // 000000003394: D7001445 02028A26
	s_wait_alu depctr_va_sdst(0)                               // 00000000339C: BF88F19F
	v_add_co_ci_u32_e64 v70, null, s39, v70, s20               // 0000000033A0: D5207C46 00528C27
	s_clause 0x7                                               // 0000000033A8: BF850007
	global_load_d16_b16 v99, v[7:8], off                       // 0000000033AC: EE08007C 00000063 00000007
	global_load_d16_hi_b16 v99, v[9:10], off                   // 0000000033B8: EE08C07C 00000063 00000009
	global_load_d16_b16 v100, v[11:12], off                    // 0000000033C4: EE08007C 00000064 0000000B
	global_load_d16_hi_b16 v100, v[13:14], off                 // 0000000033D0: EE08C07C 00000064 0000000D
	global_load_d16_b16 v101, v[5:6], off                      // 0000000033DC: EE08007C 00000065 00000005
	global_load_d16_hi_b16 v101, v[67:68], off                 // 0000000033E8: EE08C07C 00000065 00000043
	global_load_d16_b16 v102, v[15:16], off                    // 0000000033F4: EE08007C 00000066 0000000F
	global_load_d16_hi_b16 v102, v[69:70], off                 // 000000003400: EE08C07C 00000066 00000045
	ds_load_b128 v[5:8], v57 offset:4128                       // 00000000340C: DBFC1020 05000039
	ds_load_b128 v[9:12], v57 offset:4096                      // 000000003414: DBFC1000 09000039
	ds_load_b128 v[13:16], v57 offset:4112                     // 00000000341C: DBFC1010 0D000039
	ds_load_b128 v[67:70], v57 offset:4144                     // 000000003424: DBFC1030 43000039
	s_wait_dscnt 0x2                                           // 00000000342C: BFC60002
	v_cndmask_b32_e64 v5, v5, v9, s3                           // 000000003430: D5010005 000E1305
	v_cndmask_b32_e64 v6, v6, v10, s3                          // 000000003438: D5010006 000E1506
	v_cndmask_b32_e64 v7, v7, v11, s3                          // 000000003440: D5010007 000E1707
	v_cndmask_b32_e64 v8, v8, v12, s3                          // 000000003448: D5010008 000E1908
	v_cndmask_b16 v9.l, v4.h, v4.l, s3                         // 000000003450: D65D0809 000E0904
	v_cvt_f16_f32_e32 v71.l, v5                                // 000000003458: 7E8E1505
	v_cvt_f16_f32_e32 v71.h, v6                                // 00000000345C: 7F8E1506
	v_cvt_f16_f32_e32 v72.l, v7                                // 000000003460: 7E901507
	v_cvt_f16_f32_e32 v72.h, v8                                // 000000003464: 7F901508
	s_wait_dscnt 0x0                                           // 000000003468: BFC60000
	v_cndmask_b32_e64 v5, v67, v13, s3                         // 00000000346C: D5010005 000E1B43
	v_cndmask_b32_e64 v6, v68, v14, s3                         // 000000003474: D5010006 000E1D44
	v_cndmask_b32_e64 v7, v69, v15, s3                         // 00000000347C: D5010007 000E1F45
	v_cndmask_b32_e64 v8, v70, v16, s3                         // 000000003484: D5010008 000E2146
	ds_load_b128 v[67:70], v59                                 // 00000000348C: DBFC0000 4300003B
	ds_load_b32 v13, v58                                       // 000000003494: D8D80000 0D00003A
	ds_load_b32 v14, v60                                       // 00000000349C: D8D80000 0E00003C
	ds_load_b32 v15, v61                                       // 0000000034A4: D8D80000 0F00003D
	v_cvt_f16_f32_e32 v73.l, v5                                // 0000000034AC: 7E921505
	s_wait_loadcnt 0x36                                        // 0000000034B0: BFC00036
	v_cndmask_b16 v4.l, 0, v75.l, s12                          // 0000000034B4: D65D0004 00329680
	v_cndmask_b16 v4.h, 0, v75.h, s13                          // 0000000034BC: D65D5004 00369680
	s_wait_loadcnt 0x34                                        // 0000000034C4: BFC00034
	v_cndmask_b16 v5.l, 0, v76.l, s14                          // 0000000034C8: D65D0005 003A9880
	v_cndmask_b16 v5.h, 0, v76.h, s15                          // 0000000034D0: D65D5005 003E9880
	v_cndmask_b16 v9.h, v1.h, v1.l, s3                         // 0000000034D8: D65D4809 000E0301
	v_cndmask_b16 v10.l, v2.h, v2.l, s3                        // 0000000034E0: D65D080A 000E0502
	s_wait_loadcnt 0x32                                        // 0000000034E8: BFC00032
	v_cndmask_b16 v1.l, 0, v77.l, s16                          // 0000000034EC: D65D0001 00429A80
	v_cndmask_b16 v1.h, 0, v77.h, s17                          // 0000000034F4: D65D5001 00469A80
	s_wait_loadcnt 0x30                                        // 0000000034FC: BFC00030
	v_cndmask_b16 v2.l, 0, v78.l, s18                          // 000000003500: D65D0002 004A9C80
	v_cndmask_b16 v2.h, 0, v78.h, s19                          // 000000003508: D65D5002 004E9C80
	ds_load_b128 v[75:78], v59 offset:16                       // 000000003510: DBFC0010 4B00003B
	ds_load_b32 v16, v62                                       // 000000003518: D8D80000 1000003E
	ds_load_b32 v103, v63                                      // 000000003520: D8D80000 6700003F
	ds_load_b32 v104, v64                                      // 000000003528: D8D80000 68000040
	ds_load_b32 v105, v65                                      // 000000003530: D8D80000 69000041
	ds_load_b32 v106, v66                                      // 000000003538: D8D80000 6A000042
	v_cvt_f16_f32_e32 v73.h, v6                                // 000000003540: 7F921506
	v_cvt_f16_f32_e32 v74.l, v7                                // 000000003544: 7E941507
	v_cvt_f16_f32_e32 v74.h, v8                                // 000000003548: 7F941508
	v_cndmask_b16 v10.h, v3.h, v3.l, s3                        // 00000000354C: D65D480A 000E0703
	v_cndmask_b16 v11.l, v4.h, v4.l, s3                        // 000000003554: D65D080B 000E0904
	v_cndmask_b16 v11.h, v5.h, v5.l, s3                        // 00000000355C: D65D480B 000E0B05
	v_cndmask_b16 v12.l, v1.h, v1.l, s3                        // 000000003564: D65D080C 000E0301
	v_cndmask_b16 v12.h, v2.h, v2.l, s3                        // 00000000356C: D65D480C 000E0502
	s_delay_alu instid0(VALU_DEP_1)                            // 000000003574: BF870001
	v_wmma_f32_16x16x16_f16 v[1:8], v[71:74], v[9:12], 0       // 000000003578: CC404001 1A021347
	s_wait_dscnt 0x7                                           // 000000003580: BFC60007
	v_dual_mul_f32 v9, v13, v67 :: v_dual_mul_f32 v10, v14, v68// 000000003584: C8C6870D 090A890E
	s_wait_dscnt 0x6                                           // 00000000358C: BFC60006
	v_mul_f32_e32 v11, v15, v69                                // 000000003590: 10168B0F
	s_wait_dscnt 0x1                                           // 000000003594: BFC60001
	v_mul_f32_e32 v12, v105, v77                               // 000000003598: 10189B69
	v_dual_add_f32 v9, v1, v9 :: v_dual_add_f32 v10, v2, v10   // 00000000359C: C9081301 090A1502
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000035A4: BF870133
	v_add_f32_e32 v11, v3, v11                                 // 0000000035A8: 06161703
	v_mul_f32_e32 v3, v104, v76                                // 0000000035AC: 10069968
	v_dual_mul_f32 v1, v16, v70 :: v_dual_mul_f32 v2, v103, v75// 0000000035B0: C8C68D10 01029767
	v_dual_add_f32 v7, v7, v12 :: v_dual_add_f32 v6, v6, v3    // 0000000035B8: C9081907 07060706
	s_wait_dscnt 0x0                                           // 0000000035C0: BFC60000
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000035C4: BF870192
	v_dual_mul_f32 v13, v106, v78 :: v_dual_add_f32 v14, v4, v1// 0000000035C8: C8C89D6A 0D0E0304
	v_add_f32_e32 v5, v5, v2                                   // 0000000035D0: 060A0505
	s_wait_loadcnt 0x2e                                        // 0000000035D4: BFC0002E
	v_cndmask_b16 v1.l, 0, v79.l, s4                           // 0000000035D8: D65D0001 00129E80
	v_cndmask_b16 v1.h, 0, v79.h, s5                           // 0000000035E0: D65D5001 00169E80
	v_add_f32_e32 v8, v8, v13                                  // 0000000035E8: 06101B08
	ds_store_b32 v60, v10                                      // 0000000035EC: D8340000 00000A3C
	ds_store_b32 v61, v11                                      // 0000000035F4: D8340000 00000B3D
	ds_store_b32 v62, v14                                      // 0000000035FC: D8340000 00000E3E
	ds_store_b32 v63, v5                                       // 000000003604: D8340000 0000053F
	ds_store_b32 v64, v6                                       // 00000000360C: D8340000 00000640
	ds_store_b32 v65, v7                                       // 000000003614: D8340000 00000741
	ds_store_b32 v58, v9                                       // 00000000361C: D8340000 0000093A
	ds_store_b32 v66, v8                                       // 000000003624: D8340000 00000842
	s_wait_loadcnt 0x2c                                        // 00000000362C: BFC0002C
	v_cndmask_b16 v2.l, 0, v80.l, s6                           // 000000003630: D65D0002 001AA080
	v_cndmask_b16 v2.h, 0, v80.h, s7                           // 000000003638: D65D5002 001EA080
	s_wait_loadcnt 0x2a                                        // 000000003640: BFC0002A
	v_cndmask_b16 v3.l, 0, v81.l, s8                           // 000000003644: D65D0003 0022A280
	v_cndmask_b16 v3.h, 0, v81.h, s9                           // 00000000364C: D65D5003 0026A280
	s_wait_loadcnt 0x28                                        // 000000003654: BFC00028
	v_cndmask_b16 v4.l, 0, v82.l, s10                          // 000000003658: D65D0004 002AA480
	v_cndmask_b16 v4.h, 0, v82.h, s11                          // 000000003660: D65D5004 002EA480
	ds_load_2addr_b32 v[79:80], v58 offset0:16 offset1:32      // 000000003668: D8DC2010 4F00003A
	ds_load_2addr_b32 v[81:82], v58 offset0:48 offset1:80      // 000000003670: D8DC5030 5100003A
	ds_load_2addr_b32 v[103:104], v58 offset0:144 offset1:160  // 000000003678: D8DCA090 6700003A
	ds_load_2addr_b32 v[105:106], v58 offset0:176 offset1:208  // 000000003680: D8DCD0B0 6900003A
	ds_load_2addr_b32 v[107:108], v109 offset0:16 offset1:32   // 000000003688: D8DC2010 6B00006D
	v_cndmask_b16 v9.l, v1.h, v1.l, s3                         // 000000003690: D65D0809 000E0301
	v_cndmask_b16 v9.h, v2.h, v2.l, s3                         // 000000003698: D65D4809 000E0502
	s_wait_loadcnt 0x26                                        // 0000000036A0: BFC00026
	v_cndmask_b16 v1.l, 0, v83.l, s12                          // 0000000036A4: D65D0001 0032A680
	v_cndmask_b16 v1.h, 0, v83.h, s13                          // 0000000036AC: D65D5001 0036A680
	s_wait_loadcnt 0x24                                        // 0000000036B4: BFC00024
	v_cndmask_b16 v2.l, 0, v84.l, s14                          // 0000000036B8: D65D0002 003AA880
	v_cndmask_b16 v2.h, 0, v84.h, s15                          // 0000000036C0: D65D5002 003EA880
	ds_load_2addr_b32 v[83:84], v109 offset0:48 offset1:80     // 0000000036C8: D8DC5030 5300006D
	v_cndmask_b16 v10.l, v3.h, v3.l, s3                        // 0000000036D0: D65D080A 000E0703
	v_cndmask_b16 v10.h, v4.h, v4.l, s3                        // 0000000036D8: D65D480A 000E0904
	s_wait_loadcnt 0x22                                        // 0000000036E0: BFC00022
	v_cndmask_b16 v3.l, 0, v85.l, s16                          // 0000000036E4: D65D0003 0042AA80
	v_cndmask_b16 v3.h, 0, v85.h, s17                          // 0000000036EC: D65D5003 0046AA80
	s_wait_loadcnt 0x20                                        // 0000000036F4: BFC00020
	v_cndmask_b16 v4.l, 0, v86.l, s18                          // 0000000036F8: D65D0004 004AAC80
	v_cndmask_b16 v4.h, 0, v86.h, s19                          // 000000003700: D65D5004 004EAC80
	v_cndmask_b16 v11.l, v1.h, v1.l, s3                        // 000000003708: D65D080B 000E0301
	v_cndmask_b16 v11.h, v2.h, v2.l, s3                        // 000000003710: D65D480B 000E0502
	v_cndmask_b16 v12.l, v3.h, v3.l, s3                        // 000000003718: D65D080C 000E0703
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000003720: BF870094
	v_cndmask_b16 v12.h, v4.h, v4.l, s3                        // 000000003724: D65D480C 000E0904
	v_wmma_f32_16x16x16_f16 v[1:8], v[71:74], v[9:12], 0       // 00000000372C: CC404001 1A021347
	s_wait_dscnt 0x3                                           // 000000003734: BFC60003
	v_dual_mul_f32 v10, v68, v82 :: v_dual_mul_f32 v11, v69, v103// 000000003738: C8C6A544 0A0ACF45
	s_wait_dscnt 0x1                                           // 000000003740: BFC60001
	v_mul_f32_e32 v13, v75, v107                               // 000000003744: 101AD74B
	v_dual_mul_f32 v9, v67, v79 :: v_dual_mul_f32 v12, v70, v106// 000000003748: C8C69F43 090CD546
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000003750: BF870193
	v_dual_add_f32 v103, v2, v10 :: v_dual_add_f32 v106, v3, v11// 000000003754: C9081502 676A1703
	v_add_f32_e32 v110, v5, v13                                // 00000000375C: 06DC1B05
	s_wait_dscnt 0x0                                           // 000000003760: BFC60000
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003764: BF870003
	v_dual_add_f32 v82, v1, v9 :: v_dual_mul_f32 v11, v76, v84 // 000000003768: C9061301 520AA94C
	s_wait_loadcnt 0x1e                                        // 000000003770: BFC0001E
	v_cndmask_b16 v1.l, 0, v87.l, s4                           // 000000003774: D65D0001 0012AE80
	v_cndmask_b16 v1.h, 0, v87.h, s5                           // 00000000377C: D65D5001 0016AE80
	ds_load_2addr_b32 v[84:85], v109 offset0:144 offset1:160   // 000000003784: D8DCA090 5400006D
	ds_load_2addr_b32 v[86:87], v109 offset0:176 offset1:208   // 00000000378C: D8DCD0B0 5600006D
	s_wait_loadcnt 0x1c                                        // 000000003794: BFC0001C
	v_cndmask_b16 v2.l, 0, v88.l, s6                           // 000000003798: D65D0002 001AB080
	v_cndmask_b16 v2.h, 0, v88.h, s7                           // 0000000037A0: D65D5002 001EB080
	s_wait_loadcnt 0x1a                                        // 0000000037A8: BFC0001A
	v_cndmask_b16 v3.l, 0, v89.l, s8                           // 0000000037AC: D65D0003 0022B280
	v_cndmask_b16 v3.h, 0, v89.h, s9                           // 0000000037B4: D65D5003 0026B280
	v_add_f32_e32 v107, v4, v12                                // 0000000037BC: 06D61904
	v_cndmask_b16 v1.l, v1.h, v1.l, s3                         // 0000000037C0: D65D0801 000E0301
	v_cndmask_b16 v1.h, v2.h, v2.l, s3                         // 0000000037C8: D65D4801 000E0502
	s_wait_loadcnt 0x18                                        // 0000000037D0: BFC00018
	v_cndmask_b16 v2.h, 0, v90.l, s10                          // 0000000037D4: D65D4002 002AB480
	v_cndmask_b16 v2.l, v3.h, v3.l, s3                         // 0000000037DC: D65D0802 000E0703
	v_cndmask_b16 v3.l, 0, v90.h, s11                          // 0000000037E4: D65D1003 002EB480
	ds_load_2addr_b32 v[88:89], v58 offset0:96 offset1:112     // 0000000037EC: D8DC7060 5800003A
	s_wait_loadcnt 0x16                                        // 0000000037F4: BFC00016
	v_cndmask_b16 v3.h, 0, v91.l, s12                          // 0000000037F8: D65D4003 0032B680
	v_cndmask_b16 v4.l, 0, v91.h, s13                          // 000000003800: D65D1004 0036B680
	s_wait_loadcnt 0x14                                        // 000000003808: BFC00014
	v_cndmask_b16 v4.h, 0, v92.l, s14                          // 00000000380C: D65D4004 003AB880
	v_cndmask_b16 v5.l, 0, v92.h, s15                          // 000000003814: D65D1005 003EB880
	s_wait_loadcnt 0x12                                        // 00000000381C: BFC00012
	v_cndmask_b16 v5.h, 0, v93.l, s16                          // 000000003820: D65D4005 0042BA80
	v_cndmask_b16 v9.l, 0, v93.h, s17                          // 000000003828: D65D1009 0046BA80
	s_wait_loadcnt 0x10                                        // 000000003830: BFC00010
	v_cndmask_b16 v9.h, 0, v94.l, s18                          // 000000003834: D65D4009 004ABC80
	ds_load_2addr_b32 v[90:91], v58 offset0:224 offset1:240    // 00000000383C: D8DCF0E0 5A00003A
	v_cndmask_b16 v10.l, 0, v94.h, s19                         // 000000003844: D65D100A 004EBC80
	v_cndmask_b16 v2.h, v3.l, v2.h, s3                         // 00000000384C: D65D5002 000E0503
	v_cndmask_b16 v3.l, v4.l, v3.h, s3                         // 000000003854: D65D1003 000E0704
	v_cndmask_b16 v3.h, v5.l, v4.h, s3                         // 00000000385C: D65D5003 000E0905
	v_cndmask_b16 v4.l, v9.l, v5.h, s3                         // 000000003864: D65D1004 000E0B09
	v_cndmask_b16 v4.h, v10.l, v9.h, s3                        // 00000000386C: D65D5004 000E130A
	v_add_f32_e32 v92, v6, v11                                 // 000000003874: 06B81706
	s_wait_dscnt 0x2                                           // 000000003878: BFC60002
	v_mul_f32_e32 v6, v78, v87                                 // 00000000387C: 100CAF4E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003880: BF870003
	v_wmma_f32_16x16x16_f16 v[9:16], v[71:74], v[1:4], 0       // 000000003884: CC404009 1A020347
	v_mul_f32_e32 v1, v67, v80                                 // 00000000388C: 1002A143
	ds_load_2addr_b32 v[79:80], v109 offset0:96 offset1:112    // 000000003890: D8DC7060 4F00006D
	v_mul_f32_e32 v2, v69, v104                                // 000000003898: 1004D145
	v_mul_f32_e32 v5, v77, v84                                 // 00000000389C: 100AA94D
	v_add_f32_e32 v93, v8, v6                                  // 0000000038A0: 06BA0D08
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000038A4: BF870193
	v_add_f32_e32 v2, v11, v2                                  // 0000000038A8: 0604050B
	v_add_f32_e32 v84, v7, v5                                  // 0000000038AC: 06A80B07
	v_mul_f32_e32 v5, v75, v108                                // 0000000038B0: 100AD94B
	s_wait_dscnt 0x1                                           // 0000000038B4: BFC60001
	v_dual_mul_f32 v3, v68, v88 :: v_dual_mul_f32 v4, v70, v90 // 0000000038B8: C8C6B144 0304B546
	ds_store_2addr_b32 v58, v106, v2 offset0:144 offset1:160   // 0000000038C0: D838A090 00026A3A
	ds_load_2addr_b32 v[87:88], v109 offset0:224 offset1:240   // 0000000038C8: D8DCF0E0 5700006D
	v_add_f32_e32 v6, v13, v5                                  // 0000000038D0: 060C0B0D
	v_add_f32_e32 v1, v9, v1                                   // 0000000038D4: 06020309
	ds_store_2addr_b32 v109, v110, v6 offset0:16 offset1:32    // 0000000038D8: D8382010 00066E6D
	ds_store_2addr_b32 v58, v82, v1 offset0:16 offset1:32      // 0000000038E0: D8382010 0001523A
	v_add_f32_e32 v82, v10, v3                                 // 0000000038E8: 06A4070A
	s_wait_dscnt 0x4                                           // 0000000038EC: BFC60004
	v_mul_f32_e32 v7, v76, v79                                 // 0000000038F0: 100E9F4C
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000038F4: BF870001
	v_dual_add_f32 v90, v12, v4 :: v_dual_add_f32 v13, v14, v7 // 0000000038F8: C908090C 5A0C0F0E
	v_mul_f32_e32 v14, v77, v85                                // 000000003900: 101CAB4D
	s_wait_loadcnt 0xe                                         // 000000003904: BFC0000E
	v_cndmask_b16 v1.l, 0, v95.l, s4                           // 000000003908: D65D0001 0012BE80
	v_cndmask_b16 v1.h, 0, v95.h, s5                           // 000000003910: D65D5001 0016BE80
	s_wait_loadcnt 0xc                                         // 000000003918: BFC0000C
	v_cndmask_b16 v2.l, 0, v96.l, s6                           // 00000000391C: D65D0002 001AC080
	v_cndmask_b16 v2.h, 0, v96.h, s7                           // 000000003924: D65D5002 001EC080
	s_wait_loadcnt 0xa                                         // 00000000392C: BFC0000A
	v_cndmask_b16 v3.l, 0, v97.l, s8                           // 000000003930: D65D0003 0022C280
	v_cndmask_b16 v3.h, 0, v97.h, s9                           // 000000003938: D65D5003 0026C280
	v_cndmask_b16 v9.l, v1.h, v1.l, s3                         // 000000003940: D65D0809 000E0301
	s_wait_loadcnt 0x8                                         // 000000003948: BFC00008
	v_cndmask_b16 v1.l, 0, v98.l, s10                          // 00000000394C: D65D0001 002AC480
	v_cndmask_b16 v9.h, v2.h, v2.l, s3                         // 000000003954: D65D4809 000E0502
	v_cndmask_b16 v1.h, 0, v98.h, s11                          // 00000000395C: D65D5001 002EC480
	v_cndmask_b16 v10.l, v3.h, v3.l, s3                        // 000000003964: D65D080A 000E0703
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000396C: BF870002
	v_cndmask_b16 v10.h, v1.h, v1.l, s3                        // 000000003970: D65D480A 000E0301
	s_wait_loadcnt 0x6                                         // 000000003978: BFC00006
	v_cndmask_b16 v2.l, 0, v99.l, s12                          // 00000000397C: D65D0002 0032C680
	v_cndmask_b16 v2.h, 0, v99.h, s13                          // 000000003984: D65D5002 0036C680
	s_wait_loadcnt 0x4                                         // 00000000398C: BFC00004
	v_cndmask_b16 v3.l, 0, v100.l, s14                         // 000000003990: D65D0003 003AC880
	v_cndmask_b16 v3.h, 0, v100.h, s15                         // 000000003998: D65D5003 003EC880
	s_wait_loadcnt 0x2                                         // 0000000039A0: BFC00002
	v_cndmask_b16 v4.l, 0, v101.l, s16                         // 0000000039A4: D65D0004 0042CA80
	v_cndmask_b16 v4.h, 0, v101.h, s17                         // 0000000039AC: D65D5004 0046CA80
	s_wait_loadcnt 0x0                                         // 0000000039B4: BFC00000
	v_cndmask_b16 v5.l, 0, v102.l, s18                         // 0000000039B8: D65D0005 004ACC80
	v_cndmask_b16 v5.h, 0, v102.h, s19                         // 0000000039C0: D65D5005 004ECC80
	v_cndmask_b16 v11.l, v2.h, v2.l, s3                        // 0000000039C8: D65D080B 000E0502
	v_cndmask_b16 v11.h, v3.h, v3.l, s3                        // 0000000039D0: D65D480B 000E0703
	v_cndmask_b16 v12.l, v4.h, v4.l, s3                        // 0000000039D8: D65D080C 000E0904
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000039E0: BF870094
	v_cndmask_b16 v12.h, v5.h, v5.l, s3                        // 0000000039E4: D65D480C 000E0B05
	v_wmma_f32_16x16x16_f16 v[1:8], v[71:74], v[9:12], 0       // 0000000039EC: CC404001 1A021347
	v_mul_f32_e32 v9, v67, v81                                 // 0000000039F4: 1012A343
	v_mul_f32_e32 v11, v69, v105                               // 0000000039F8: 1016D345
	v_mul_f32_e32 v10, v68, v89                                // 0000000039FC: 1014B344
	v_add_f32_e32 v12, v15, v14                                // 000000003A00: 06181D0F
	s_wait_dscnt 0x2                                           // 000000003A04: BFC60002
	v_dual_mul_f32 v14, v78, v87 :: v_dual_add_f32 v1, v1, v9  // 000000003A08: C8C8AF4E 0E001301
	s_delay_alu instid0(VALU_DEP_3)                            // 000000003A10: BF870003
	v_dual_add_f32 v3, v3, v11 :: v_dual_add_f32 v2, v2, v10   // 000000003A14: C9081703 03021502
	ds_store_2addr_b32 v58, v1, v103 offset0:48 offset1:80     // 000000003A1C: D8385030 0067013A
	ds_store_2addr_b32 v58, v82, v2 offset0:96 offset1:112     // 000000003A24: D8387060 0002523A
	ds_store_2addr_b32 v58, v3, v107 offset0:176 offset1:208   // 000000003A2C: D838D0B0 006B033A
	v_dual_mul_f32 v1, v70, v91 :: v_dual_mul_f32 v10, v77, v86// 000000003A34: C8C6B746 010AAD4D
	v_dual_mul_f32 v2, v75, v83 :: v_dual_mul_f32 v3, v76, v80 // 000000003A3C: C8C6A74B 0202A14C
	v_mul_f32_e32 v11, v78, v88                                // 000000003A44: 1016B14E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000003A48: BF870193
	v_dual_add_f32 v1, v4, v1 :: v_dual_add_f32 v4, v7, v10    // 000000003A4C: C9080304 01041507
	v_dual_add_f32 v2, v5, v2 :: v_dual_add_f32 v3, v6, v3     // 000000003A54: C9080505 02020706
	v_add_f32_e32 v9, v16, v14                                 // 000000003A5C: 06121D10
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003A60: BF870004
	v_add_f32_e32 v5, v8, v11                                  // 000000003A64: 060A1708
	ds_store_2addr_b32 v109, v84, v12 offset0:144 offset1:160  // 000000003A68: D838A090 000C546D
	ds_store_2addr_b32 v58, v90, v1 offset0:224 offset1:240    // 000000003A70: D838F0E0 00015A3A
	ds_store_2addr_b32 v109, v2, v92 offset0:48 offset1:80     // 000000003A78: D8385030 005C026D
	ds_store_2addr_b32 v109, v13, v3 offset0:96 offset1:112    // 000000003A80: D8387060 00030D6D
	ds_store_2addr_b32 v109, v4, v93 offset0:176 offset1:208   // 000000003A88: D838D0B0 005D046D
	ds_store_2addr_b32 v109, v9, v5 offset0:224 offset1:240    // 000000003A90: D838F0E0 0005096D
	s_wait_dscnt 0x0                                           // 000000003A98: BFC60000
	s_barrier_signal -1                                        // 000000003A9C: BE804EC1
	s_barrier_wait 0xffff                                      // 000000003AA0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000003AA4: EE0AC07C 00040000 00000000
	s_cbranch_scc0 1222                                        // 000000003AB0: BFA104C6 <attention_forward+0x32cc>
	s_wait_alu depctr_sa_sdst(0)                               // 000000003AB4: BF88FF9E
	v_add_co_u32 v83, s4, v44, s34                             // 000000003AB8: D7000453 0200452C
	s_wait_alu depctr_va_sdst(0)                               // 000000003AC0: BF88F19F
	v_add_co_ci_u32_e64 v84, null, 0, s35, s4                  // 000000003AC4: D5207C54 00104680
	s_clause 0x7                                               // 000000003ACC: BF850007
	global_load_b128 v[1:4], v[19:20], off                     // 000000003AD0: EE05C07C 00000001 00000013
	global_load_b128 v[5:8], v[19:20], off offset:16           // 000000003ADC: EE05C07C 00000005 00001013
	global_load_b128 v[9:12], v[21:22], off                    // 000000003AE8: EE05C07C 00000009 00000015
	global_load_b128 v[13:16], v[21:22], off offset:16         // 000000003AF4: EE05C07C 0000000D 00001015
	global_load_b128 v[67:70], v[23:24], off                   // 000000003B00: EE05C07C 00000043 00000017
	global_load_b128 v[71:74], v[23:24], off offset:16         // 000000003B0C: EE05C07C 00000047 00001017
	global_load_b128 v[75:78], v[25:26], off                   // 000000003B18: EE05C07C 0000004B 00000019
	global_load_b128 v[79:82], v[25:26], off offset:16         // 000000003B24: EE05C07C 0000004F 00001019
	v_add_co_u32 v115, s4, v83, -15                            // 000000003B30: D7000473 02019F53
	s_wait_alu depctr_va_sdst(0)                               // 000000003B38: BF88F19F
	v_add_co_ci_u32_e64 v116, null, -1, v84, s4                // 000000003B3C: D5207C74 0012A8C1
	v_add_co_u32 v101, s4, v57, s40                            // 000000003B44: D7000465 02005139
	s_wait_alu depctr_va_sdst(0)                               // 000000003B4C: BF88F19F
	v_add_co_ci_u32_e64 v102, null, 0, s41, s4                 // 000000003B50: D5207C66 00105280
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000003B58: BF870193
	v_cmp_gt_i64_e64 s4, s[26:27], v[115:116]                  // 000000003B5C: D4540004 0202E61A
	v_add_co_u32 v85, s5, v101, 16                             // 000000003B64: D7000555 02012165
	s_wait_alu depctr_va_sdst(0)                               // 000000003B6C: BF88F19F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000003B70: BF870223
	v_add_co_ci_u32_e64 v86, null, 0, v102, s5                 // 000000003B74: D5207C56 0016CC80
	v_cmp_lt_i64_e64 s6, v[27:28], v[115:116]                  // 000000003B7C: D4510006 0202E71B
	v_cndmask_b32_e64 v84, 0, v102, s4                         // 000000003B84: D5010054 0012CC80
	v_cndmask_b32_e64 v83, 0, v101, s4                         // 000000003B8C: D5010053 0012CA80
	s_delay_alu instid0(VALU_DEP_4)                            // 000000003B94: BF870004
	v_cndmask_b32_e64 v90, 0, v86, s4                          // 000000003B98: D501005A 0012AC80
	v_cndmask_b32_e64 v89, 0, v85, s4                          // 000000003BA0: D5010059 0012AA80
	v_cmp_lt_i64_e64 s8, v[29:30], v[115:116]                  // 000000003BA8: D4510008 0202E71D
	v_cmp_lt_i64_e64 s10, v[31:32], v[115:116]                 // 000000003BB0: D451000A 0202E71F
	v_lshlrev_b64_e32 v[83:84], 1, v[83:84]                    // 000000003BB8: 3EA6A681
	v_cmp_lt_i64_e64 s12, v[33:34], v[115:116]                 // 000000003BBC: D451000C 0202E721
	v_lshlrev_b64_e32 v[91:92], 1, v[89:90]                    // 000000003BC4: 3EB6B281
	v_cmp_lt_i64_e64 s7, v[35:36], v[115:116]                  // 000000003BC8: D4510007 0202E723
	v_cmp_lt_i64_e64 s9, v[37:38], v[115:116]                  // 000000003BD0: D4510009 0202E725
	v_cmp_lt_i64_e64 s11, v[39:40], v[115:116]                 // 000000003BD8: D451000B 0202E727
	v_add_co_u32 v87, s5, s36, v83                             // 000000003BE0: D7000557 0202A624
	s_wait_alu depctr_va_sdst(0)                               // 000000003BE8: BF88F19F
	v_add_co_ci_u32_e64 v88, null, s37, v84, s5                // 000000003BEC: D5207C58 0016A825
	v_add_co_u32 v93, s5, v101, 32                             // 000000003BF4: D700055D 02014165
	s_wait_alu depctr_va_sdst(0)                               // 000000003BFC: BF88F19F
	v_add_co_ci_u32_e64 v94, null, 0, v102, s5                 // 000000003C00: D5207C5E 0016CC80
	v_add_co_u32 v95, s5, s36, v91                             // 000000003C08: D700055F 0202B624
	s_wait_alu depctr_va_sdst(0)                               // 000000003C10: BF88F19F
	v_add_co_ci_u32_e64 v96, null, s37, v92, s5                // 000000003C14: D5207C60 0016B825
	v_add_co_u32 v101, s5, v101, 48                            // 000000003C1C: D7000565 02016165
	v_cndmask_b32_e64 v98, 0, v94, s4                          // 000000003C24: D5010062 0012BC80
	v_cndmask_b32_e64 v97, 0, v93, s4                          // 000000003C2C: D5010061 0012BA80
	s_wait_alu depctr_va_sdst(0)                               // 000000003C34: BF88F19F
	v_add_co_ci_u32_e64 v102, null, 0, v102, s5                // 000000003C38: D5207C66 0016CC80
	v_cndmask_b32_e64 v105, 0, v101, s4                        // 000000003C40: D5010069 0012CA80
	s_clause 0x1                                               // 000000003C48: BF850001
	global_load_b128 v[83:86], v[87:88], off                   // 000000003C4C: EE05C07C 00000053 00000057
	global_load_b128 v[91:94], v[95:96], off                   // 000000003C58: EE05C07C 0000005B 0000005F
	v_lshlrev_b64_e32 v[99:100], 1, v[97:98]                   // 000000003C64: 3EC6C281
	v_cndmask_b32_e64 v106, 0, v102, s4                        // 000000003C68: D501006A 0012CC80
	s_clause 0x1                                               // 000000003C70: BF850001
	global_load_b128 v[87:90], v[87:88], off offset:16         // 000000003C74: EE05C07C 00000057 00001057
	global_load_b128 v[95:98], v[95:96], off offset:16         // 000000003C80: EE05C07C 0000005F 0000105F
	v_cmp_lt_i64_e64 s13, v[41:42], v[115:116]                 // 000000003C8C: D451000D 0202E729
	s_and_b32 s7, s33, s7                                      // 000000003C94: 8B070721
	v_lshlrev_b64_e32 v[107:108], 1, v[105:106]                // 000000003C98: 3ED6D281
	v_add_co_u32 v103, s5, s36, v99                            // 000000003C9C: D7000567 0202C624
	s_wait_alu depctr_va_sdst(0)                               // 000000003CA4: BF88F19F
	v_add_co_ci_u32_e64 v104, null, s37, v100, s5              // 000000003CA8: D5207C68 0016C825
	s_and_b32 s9, s33, s9                                      // 000000003CB0: 8B090921
	v_add_co_u32 v111, s5, s36, v107                           // 000000003CB4: D700056F 0202D624
	s_clause 0x1                                               // 000000003CBC: BF850001
	global_load_b128 v[99:102], v[103:104], off                // 000000003CC0: EE05C07C 00000063 00000067
	global_load_b128 v[103:106], v[103:104], off offset:16     // 000000003CCC: EE05C07C 00000067 00001067
	s_wait_alu depctr_va_sdst(0)                               // 000000003CD8: BF88F19F
	v_add_co_ci_u32_e64 v112, null, s37, v108, s5              // 000000003CDC: D5207C70 0016D825
	s_clause 0x1                                               // 000000003CE4: BF850001
	global_load_b128 v[107:110], v[111:112], off               // 000000003CE8: EE05C07C 0000006B 0000006F
	global_load_b128 v[111:114], v[111:112], off offset:16     // 000000003CF4: EE05C07C 0000006F 0000106F
	v_cmp_le_i64_e64 s5, s[26:27], v[115:116]                  // 000000003D00: D4530005 0202E61A
	s_and_b32 s11, s33, s11                                    // 000000003D08: 8B0B0B21
	s_wait_loadcnt 0xf                                         // 000000003D0C: BFC0000F
	v_cndmask_b16 v1.l, 0, v1.l, vcc_lo                        // 000000003D10: D65D0001 01AA0280
	s_wait_loadcnt 0xe                                         // 000000003D18: BFC0000E
	v_cndmask_b16 v5.l, 0, v5.l, vcc_lo                        // 000000003D1C: D65D0005 01AA0A80
	v_cndmask_b16 v1.h, 0, v1.h, vcc_lo                        // 000000003D24: D65D5001 01AA0280
	v_cndmask_b16 v5.h, 0, v5.h, vcc_lo                        // 000000003D2C: D65D5005 01AA0A80
	v_cndmask_b16 v2.l, 0, v2.l, vcc_lo                        // 000000003D34: D65D0002 01AA0480
	v_cndmask_b16 v6.l, 0, v6.l, vcc_lo                        // 000000003D3C: D65D0006 01AA0C80
	v_cndmask_b16 v2.h, 0, v2.h, vcc_lo                        // 000000003D44: D65D5002 01AA0480
	v_cndmask_b16 v6.h, 0, v6.h, vcc_lo                        // 000000003D4C: D65D5006 01AA0C80
	v_cndmask_b16 v3.l, 0, v3.l, vcc_lo                        // 000000003D54: D65D0003 01AA0680
	v_cndmask_b16 v7.l, 0, v7.l, vcc_lo                        // 000000003D5C: D65D0007 01AA0E80
	v_cndmask_b16 v3.h, 0, v3.h, vcc_lo                        // 000000003D64: D65D5003 01AA0680
	v_cndmask_b16 v7.h, 0, v7.h, vcc_lo                        // 000000003D6C: D65D5007 01AA0E80
	v_cndmask_b16 v4.l, 0, v4.l, vcc_lo                        // 000000003D74: D65D0004 01AA0880
	v_cndmask_b16 v8.l, 0, v8.l, vcc_lo                        // 000000003D7C: D65D0008 01AA1080
	v_cndmask_b16 v4.h, 0, v4.h, vcc_lo                        // 000000003D84: D65D5004 01AA0880
	v_cndmask_b16 v8.h, 0, v8.h, vcc_lo                        // 000000003D8C: D65D5008 01AA1080
	s_wait_loadcnt 0xd                                         // 000000003D94: BFC0000D
	v_cndmask_b16 v115.l, 0, v9.l, vcc_lo                      // 000000003D98: D65D0073 01AA1280
	v_cndmask_b16 v115.h, 0, v9.h, vcc_lo                      // 000000003DA0: D65D5073 01AA1280
	v_cndmask_b16 v116.l, 0, v10.l, vcc_lo                     // 000000003DA8: D65D0074 01AA1480
	v_cndmask_b16 v116.h, 0, v10.h, vcc_lo                     // 000000003DB0: D65D5074 01AA1480
	v_cndmask_b16 v117.l, 0, v11.l, vcc_lo                     // 000000003DB8: D65D0075 01AA1680
	v_cndmask_b16 v117.h, 0, v11.h, vcc_lo                     // 000000003DC0: D65D5075 01AA1680
	v_cndmask_b16 v118.l, 0, v12.l, vcc_lo                     // 000000003DC8: D65D0076 01AA1880
	v_cndmask_b16 v118.h, 0, v12.h, vcc_lo                     // 000000003DD0: D65D5076 01AA1880
	s_wait_loadcnt 0xb                                         // 000000003DD8: BFC0000B
	v_cndmask_b16 v67.l, 0, v67.l, vcc_lo                      // 000000003DDC: D65D0043 01AA8680
	s_wait_loadcnt 0xa                                         // 000000003DE4: BFC0000A
	v_cndmask_b16 v71.l, 0, v71.l, vcc_lo                      // 000000003DE8: D65D0047 01AA8E80
	v_cndmask_b16 v67.h, 0, v67.h, vcc_lo                      // 000000003DF0: D65D5043 01AA8680
	v_cndmask_b16 v71.h, 0, v71.h, vcc_lo                      // 000000003DF8: D65D5047 01AA8E80
	v_cndmask_b16 v68.l, 0, v68.l, vcc_lo                      // 000000003E00: D65D0044 01AA8880
	v_cndmask_b16 v72.l, 0, v72.l, vcc_lo                      // 000000003E08: D65D0048 01AA9080
	v_cndmask_b16 v68.h, 0, v68.h, vcc_lo                      // 000000003E10: D65D5044 01AA8880
	v_cndmask_b16 v72.h, 0, v72.h, vcc_lo                      // 000000003E18: D65D5048 01AA9080
	v_cndmask_b16 v69.l, 0, v69.l, vcc_lo                      // 000000003E20: D65D0045 01AA8A80
	v_cndmask_b16 v73.l, 0, v73.l, vcc_lo                      // 000000003E28: D65D0049 01AA9280
	v_cndmask_b16 v69.h, 0, v69.h, vcc_lo                      // 000000003E30: D65D5045 01AA8A80
	v_cndmask_b16 v73.h, 0, v73.h, vcc_lo                      // 000000003E38: D65D5049 01AA9280
	v_cndmask_b16 v70.l, 0, v70.l, vcc_lo                      // 000000003E40: D65D0046 01AA8C80
	v_cndmask_b16 v74.l, 0, v74.l, vcc_lo                      // 000000003E48: D65D004A 01AA9480
	v_cndmask_b16 v70.h, 0, v70.h, vcc_lo                      // 000000003E50: D65D5046 01AA8C80
	v_cndmask_b16 v74.h, 0, v74.h, vcc_lo                      // 000000003E58: D65D504A 01AA9480
	s_wait_loadcnt 0x9                                         // 000000003E60: BFC00009
	v_cndmask_b16 v75.l, 0, v75.l, vcc_lo                      // 000000003E64: D65D004B 01AA9680
	v_cndmask_b16 v75.h, 0, v75.h, vcc_lo                      // 000000003E6C: D65D504B 01AA9680
	v_cndmask_b16 v76.l, 0, v76.l, vcc_lo                      // 000000003E74: D65D004C 01AA9880
	v_cndmask_b16 v76.h, 0, v76.h, vcc_lo                      // 000000003E7C: D65D504C 01AA9880
	v_cndmask_b16 v77.l, 0, v77.l, vcc_lo                      // 000000003E84: D65D004D 01AA9A80
	v_cndmask_b16 v77.h, 0, v77.h, vcc_lo                      // 000000003E8C: D65D504D 01AA9A80
	v_cndmask_b16 v78.l, 0, v78.l, vcc_lo                      // 000000003E94: D65D004E 01AA9C80
	v_cndmask_b16 v78.h, 0, v78.h, vcc_lo                      // 000000003E9C: D65D504E 01AA9C80
	s_wait_loadcnt 0x8                                         // 000000003EA4: BFC00008
	v_cndmask_b16 v79.l, 0, v79.l, vcc_lo                      // 000000003EA8: D65D004F 01AA9E80
	v_cndmask_b16 v79.h, 0, v79.h, vcc_lo                      // 000000003EB0: D65D504F 01AA9E80
	v_cndmask_b16 v80.l, 0, v80.l, vcc_lo                      // 000000003EB8: D65D0050 01AAA080
	v_cndmask_b16 v80.h, 0, v80.h, vcc_lo                      // 000000003EC0: D65D5050 01AAA080
	v_cndmask_b16 v81.l, 0, v81.l, vcc_lo                      // 000000003EC8: D65D0051 01AAA280
	v_cndmask_b16 v81.h, 0, v81.h, vcc_lo                      // 000000003ED0: D65D5051 01AAA280
	v_cndmask_b16 v82.l, 0, v82.l, vcc_lo                      // 000000003ED8: D65D0052 01AAA480
	v_cndmask_b16 v82.h, 0, v82.h, vcc_lo                      // 000000003EE0: D65D5052 01AAA480
	v_cndmask_b16 v9.l, v5.l, v1.l, s3                         // 000000003EE8: D65D0009 000E0305
	v_cndmask_b16 v9.h, v5.h, v1.h, s3                         // 000000003EF0: D65D5809 000E0305
	v_cndmask_b16 v10.l, v6.l, v2.l, s3                        // 000000003EF8: D65D000A 000E0506
	v_cndmask_b16 v10.h, v6.h, v2.h, s3                        // 000000003F00: D65D580A 000E0506
	v_cndmask_b16 v11.l, v7.l, v3.l, s3                        // 000000003F08: D65D000B 000E0707
	v_cndmask_b16 v11.h, v7.h, v3.h, s3                        // 000000003F10: D65D580B 000E0707
	v_cndmask_b16 v12.l, v8.l, v4.l, s3                        // 000000003F18: D65D000C 000E0908
	v_cndmask_b16 v12.h, v8.h, v4.h, s3                        // 000000003F20: D65D580C 000E0908
	s_wait_loadcnt 0x7                                         // 000000003F28: BFC00007
	v_cndmask_b16 v1.l, 0, v83.l, s4                           // 000000003F2C: D65D0001 0012A680
	v_cndmask_b16 v1.h, 0, v83.h, s4                           // 000000003F34: D65D5001 0012A680
	v_cndmask_b16 v2.l, 0, v84.l, s4                           // 000000003F3C: D65D0002 0012A880
	v_cndmask_b16 v2.h, 0, v84.h, s4                           // 000000003F44: D65D5002 0012A880
	v_cndmask_b16 v3.l, 0, v85.l, s4                           // 000000003F4C: D65D0003 0012AA80
	v_cndmask_b16 v3.h, 0, v85.h, s4                           // 000000003F54: D65D5003 0012AA80
	v_cndmask_b16 v4.l, 0, v86.l, s4                           // 000000003F5C: D65D0004 0012AC80
	v_cndmask_b16 v4.h, 0, v86.h, s4                           // 000000003F64: D65D5004 0012AC80
	s_wait_loadcnt 0x5                                         // 000000003F6C: BFC00005
	v_cndmask_b16 v5.l, 0, v87.l, s4                           // 000000003F70: D65D0005 0012AE80
	v_cndmask_b16 v5.h, 0, v87.h, s4                           // 000000003F78: D65D5005 0012AE80
	v_cndmask_b16 v6.l, 0, v88.l, s4                           // 000000003F80: D65D0006 0012B080
	v_cndmask_b16 v6.h, 0, v88.h, s4                           // 000000003F88: D65D5006 0012B080
	v_cndmask_b16 v7.l, 0, v89.l, s4                           // 000000003F90: D65D0007 0012B280
	v_cndmask_b16 v7.h, 0, v89.h, s4                           // 000000003F98: D65D5007 0012B280
	v_cndmask_b16 v8.l, 0, v90.l, s4                           // 000000003FA0: D65D0008 0012B480
	v_cndmask_b16 v8.h, 0, v90.h, s4                           // 000000003FA8: D65D5008 0012B480
	v_cndmask_b16 v13.l, 0, v13.l, vcc_lo                      // 000000003FB0: D65D000D 01AA1A80
	v_cndmask_b16 v13.h, 0, v13.h, vcc_lo                      // 000000003FB8: D65D500D 01AA1A80
	v_cndmask_b16 v14.l, 0, v14.l, vcc_lo                      // 000000003FC0: D65D000E 01AA1C80
	v_cndmask_b16 v14.h, 0, v14.h, vcc_lo                      // 000000003FC8: D65D500E 01AA1C80
	v_cndmask_b16 v15.l, 0, v15.l, vcc_lo                      // 000000003FD0: D65D000F 01AA1E80
	v_cndmask_b16 v15.h, 0, v15.h, vcc_lo                      // 000000003FD8: D65D500F 01AA1E80
	v_cndmask_b16 v16.l, 0, v16.l, vcc_lo                      // 000000003FE0: D65D0010 01AA2080
	v_cndmask_b16 v16.h, 0, v16.h, vcc_lo                      // 000000003FE8: D65D5010 01AA2080
	v_cndmask_b16 v67.l, v71.l, v67.l, s3                      // 000000003FF0: D65D0043 000E8747
	v_cndmask_b16 v67.h, v71.h, v67.h, s3                      // 000000003FF8: D65D5843 000E8747
	v_cndmask_b16 v68.l, v72.l, v68.l, s3                      // 000000004000: D65D0044 000E8948
	v_cndmask_b16 v68.h, v72.h, v68.h, s3                      // 000000004008: D65D5844 000E8948
	v_cndmask_b16 v69.l, v73.l, v69.l, s3                      // 000000004010: D65D0045 000E8B49
	v_cndmask_b16 v69.h, v73.h, v69.h, s3                      // 000000004018: D65D5845 000E8B49
	v_cndmask_b16 v70.l, v74.l, v70.l, s3                      // 000000004020: D65D0046 000E8D4A
	v_cndmask_b16 v70.h, v74.h, v70.h, s3                      // 000000004028: D65D5846 000E8D4A
	v_cndmask_b16 v71.l, v79.l, v75.l, s3                      // 000000004030: D65D0047 000E974F
	v_cndmask_b16 v71.h, v79.h, v75.h, s3                      // 000000004038: D65D5847 000E974F
	v_cndmask_b16 v72.l, v80.l, v76.l, s3                      // 000000004040: D65D0048 000E9950
	v_cndmask_b16 v72.h, v80.h, v76.h, s3                      // 000000004048: D65D5848 000E9950
	v_cndmask_b16 v73.l, v81.l, v77.l, s3                      // 000000004050: D65D0049 000E9B51
	v_cndmask_b16 v73.h, v81.h, v77.h, s3                      // 000000004058: D65D5849 000E9B51
	v_cndmask_b16 v74.l, v82.l, v78.l, s3                      // 000000004060: D65D004A 000E9D52
	v_cndmask_b16 v74.h, v82.h, v78.h, s3                      // 000000004068: D65D584A 000E9D52
	v_cndmask_b16 v79.l, 0, v91.l, s4                          // 000000004070: D65D004F 0012B680
	v_cndmask_b16 v79.h, 0, v91.h, s4                          // 000000004078: D65D504F 0012B680
	v_cndmask_b16 v80.l, 0, v92.l, s4                          // 000000004080: D65D0050 0012B880
	v_cndmask_b16 v80.h, 0, v92.h, s4                          // 000000004088: D65D5050 0012B880
	v_cndmask_b16 v81.l, 0, v93.l, s4                          // 000000004090: D65D0051 0012BA80
	v_cndmask_b16 v81.h, 0, v93.h, s4                          // 000000004098: D65D5051 0012BA80
	v_cndmask_b16 v82.l, 0, v94.l, s4                          // 0000000040A0: D65D0052 0012BC80
	v_cndmask_b16 v82.h, 0, v94.h, s4                          // 0000000040A8: D65D5052 0012BC80
	s_wait_loadcnt 0x4                                         // 0000000040B0: BFC00004
	v_cndmask_b16 v83.l, 0, v95.l, s4                          // 0000000040B4: D65D0053 0012BE80
	v_cndmask_b16 v83.h, 0, v95.h, s4                          // 0000000040BC: D65D5053 0012BE80
	v_cndmask_b16 v84.l, 0, v96.l, s4                          // 0000000040C4: D65D0054 0012C080
	v_cndmask_b16 v84.h, 0, v96.h, s4                          // 0000000040CC: D65D5054 0012C080
	v_cndmask_b16 v85.l, 0, v97.l, s4                          // 0000000040D4: D65D0055 0012C280
	v_cndmask_b16 v85.h, 0, v97.h, s4                          // 0000000040DC: D65D5055 0012C280
	v_cndmask_b16 v86.l, 0, v98.l, s4                          // 0000000040E4: D65D0056 0012C480
	v_cndmask_b16 v86.h, 0, v98.h, s4                          // 0000000040EC: D65D5056 0012C480
	v_cndmask_b16 v75.l, v5.l, v1.l, s3                        // 0000000040F4: D65D004B 000E0305
	v_cndmask_b16 v75.h, v5.h, v1.h, s3                        // 0000000040FC: D65D584B 000E0305
	v_cndmask_b16 v76.l, v6.l, v2.l, s3                        // 000000004104: D65D004C 000E0506
	v_cndmask_b16 v76.h, v6.h, v2.h, s3                        // 00000000410C: D65D584C 000E0506
	v_cndmask_b16 v77.l, v7.l, v3.l, s3                        // 000000004114: D65D004D 000E0707
	v_cndmask_b16 v77.h, v7.h, v3.h, s3                        // 00000000411C: D65D584D 000E0707
	v_cndmask_b16 v78.l, v8.l, v4.l, s3                        // 000000004124: D65D004E 000E0908
	v_cndmask_b16 v78.h, v8.h, v4.h, s3                        // 00000000412C: D65D584E 000E0908
	v_cndmask_b16 v13.l, v13.l, v115.l, s3                     // 000000004134: D65D000D 000EE70D
	v_cndmask_b16 v13.h, v13.h, v115.h, s3                     // 00000000413C: D65D580D 000EE70D
	v_cndmask_b16 v14.l, v14.l, v116.l, s3                     // 000000004144: D65D000E 000EE90E
	v_cndmask_b16 v14.h, v14.h, v116.h, s3                     // 00000000414C: D65D580E 000EE90E
	v_cndmask_b16 v15.l, v15.l, v117.l, s3                     // 000000004154: D65D000F 000EEB0F
	v_cndmask_b16 v15.h, v15.h, v117.h, s3                     // 00000000415C: D65D580F 000EEB0F
	v_cndmask_b16 v16.l, v16.l, v118.l, s3                     // 000000004164: D65D0010 000EED10
	v_cndmask_b16 v16.h, v16.h, v118.h, s3                     // 00000000416C: D65D5810 000EED10
	s_wait_loadcnt 0x3                                         // 000000004174: BFC00003
	v_cndmask_b16 v87.l, 0, v99.l, s4                          // 000000004178: D65D0057 0012C680
	v_cndmask_b16 v87.h, 0, v99.h, s4                          // 000000004180: D65D5057 0012C680
	v_cndmask_b16 v88.l, 0, v100.l, s4                         // 000000004188: D65D0058 0012C880
	v_cndmask_b16 v88.h, 0, v100.h, s4                         // 000000004190: D65D5058 0012C880
	v_cndmask_b16 v89.l, 0, v101.l, s4                         // 000000004198: D65D0059 0012CA80
	v_cndmask_b16 v89.h, 0, v101.h, s4                         // 0000000041A0: D65D5059 0012CA80
	v_cndmask_b16 v90.l, 0, v102.l, s4                         // 0000000041A8: D65D005A 0012CC80
	v_cndmask_b16 v90.h, 0, v102.h, s4                         // 0000000041B0: D65D505A 0012CC80
	s_wait_loadcnt 0x2                                         // 0000000041B8: BFC00002
	v_cndmask_b16 v91.l, 0, v103.l, s4                         // 0000000041BC: D65D005B 0012CE80
	v_cndmask_b16 v91.h, 0, v103.h, s4                         // 0000000041C4: D65D505B 0012CE80
	v_cndmask_b16 v92.l, 0, v104.l, s4                         // 0000000041CC: D65D005C 0012D080
	v_cndmask_b16 v92.h, 0, v104.h, s4                         // 0000000041D4: D65D505C 0012D080
	v_cndmask_b16 v93.l, 0, v105.l, s4                         // 0000000041DC: D65D005D 0012D280
	v_cndmask_b16 v93.h, 0, v105.h, s4                         // 0000000041E4: D65D505D 0012D280
	v_cndmask_b16 v94.l, 0, v106.l, s4                         // 0000000041EC: D65D005E 0012D480
	v_cndmask_b16 v94.h, 0, v106.h, s4                         // 0000000041F4: D65D505E 0012D480
	v_cndmask_b16 v79.l, v83.l, v79.l, s3                      // 0000000041FC: D65D004F 000E9F53
	v_cndmask_b16 v79.h, v83.h, v79.h, s3                      // 000000004204: D65D584F 000E9F53
	v_cndmask_b16 v80.l, v84.l, v80.l, s3                      // 00000000420C: D65D0050 000EA154
	v_cndmask_b16 v80.h, v84.h, v80.h, s3                      // 000000004214: D65D5850 000EA154
	v_cndmask_b16 v81.l, v85.l, v81.l, s3                      // 00000000421C: D65D0051 000EA355
	v_cndmask_b16 v81.h, v85.h, v81.h, s3                      // 000000004224: D65D5851 000EA355
	v_cndmask_b16 v82.l, v86.l, v82.l, s3                      // 00000000422C: D65D0052 000EA556
	v_cndmask_b16 v82.h, v86.h, v82.h, s3                      // 000000004234: D65D5852 000EA556
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:12], v[75:78], 0       // 00000000423C: CC404001 1A029709
	s_wait_loadcnt 0x1                                         // 000000004244: BFC00001
	v_cndmask_b16 v95.l, 0, v107.l, s4                         // 000000004248: D65D005F 0012D680
	v_cndmask_b16 v95.h, 0, v107.h, s4                         // 000000004250: D65D505F 0012D680
	v_cndmask_b16 v96.l, 0, v108.l, s4                         // 000000004258: D65D0060 0012D880
	v_cndmask_b16 v96.h, 0, v108.h, s4                         // 000000004260: D65D5060 0012D880
	v_cndmask_b16 v97.l, 0, v109.l, s4                         // 000000004268: D65D0061 0012DA80
	v_cndmask_b16 v97.h, 0, v109.h, s4                         // 000000004270: D65D5061 0012DA80
	v_cndmask_b16 v98.l, 0, v110.l, s4                         // 000000004278: D65D0062 0012DC80
	v_cndmask_b16 v98.h, 0, v110.h, s4                         // 000000004280: D65D5062 0012DC80
	s_wait_loadcnt 0x0                                         // 000000004288: BFC00000
	v_cndmask_b16 v99.l, 0, v111.l, s4                         // 00000000428C: D65D0063 0012DE80
	v_cndmask_b16 v99.h, 0, v111.h, s4                         // 000000004294: D65D5063 0012DE80
	v_cndmask_b16 v100.l, 0, v112.l, s4                        // 00000000429C: D65D0064 0012E080
	v_cndmask_b16 v100.h, 0, v112.h, s4                        // 0000000042A4: D65D5064 0012E080
	v_cndmask_b16 v101.l, 0, v113.l, s4                        // 0000000042AC: D65D0065 0012E280
	v_cndmask_b16 v101.h, 0, v113.h, s4                        // 0000000042B4: D65D5065 0012E280
	v_cndmask_b16 v102.l, 0, v114.l, s4                        // 0000000042BC: D65D0066 0012E480
	v_cndmask_b16 v102.h, 0, v114.h, s4                        // 0000000042C4: D65D5066 0012E480
	v_cndmask_b16 v83.l, v91.l, v87.l, s3                      // 0000000042CC: D65D0053 000EAF5B
	v_cndmask_b16 v83.h, v91.h, v87.h, s3                      // 0000000042D4: D65D5853 000EAF5B
	v_cndmask_b16 v84.l, v92.l, v88.l, s3                      // 0000000042DC: D65D0054 000EB15C
	v_cndmask_b16 v84.h, v92.h, v88.h, s3                      // 0000000042E4: D65D5854 000EB15C
	v_cndmask_b16 v85.l, v93.l, v89.l, s3                      // 0000000042EC: D65D0055 000EB35D
	v_cndmask_b16 v85.h, v93.h, v89.h, s3                      // 0000000042F4: D65D5855 000EB35D
	v_cndmask_b16 v86.l, v94.l, v90.l, s3                      // 0000000042FC: D65D0056 000EB55E
	v_cndmask_b16 v86.h, v94.h, v90.h, s3                      // 000000004304: D65D5856 000EB55E
	v_wmma_f32_16x16x16_f16 v[1:8], v[13:16], v[79:82], v[1:8] // 00000000430C: CC404001 1C069F0D
	v_cndmask_b16 v9.l, v99.l, v95.l, s3                       // 000000004314: D65D0009 000EBF63
	v_cndmask_b16 v9.h, v99.h, v95.h, s3                       // 00000000431C: D65D5809 000EBF63
	v_cndmask_b16 v10.l, v100.l, v96.l, s3                     // 000000004324: D65D000A 000EC164
	v_cndmask_b16 v10.h, v100.h, v96.h, s3                     // 00000000432C: D65D580A 000EC164
	v_cndmask_b16 v11.l, v101.l, v97.l, s3                     // 000000004334: D65D000B 000EC365
	v_cndmask_b16 v11.h, v101.h, v97.h, s3                     // 00000000433C: D65D580B 000EC365
	v_cndmask_b16 v12.l, v102.l, v98.l, s3                     // 000000004344: D65D000C 000EC566
	v_cndmask_b16 v12.h, v102.h, v98.h, s3                     // 00000000434C: D65D580C 000EC566
	v_wmma_f32_16x16x16_f16 v[1:8], v[67:70], v[83:86], v[1:8] // 000000004354: CC404001 1C06A743
	s_and_b32 s4, s33, s6                                      // 00000000435C: 8B040621
	s_and_b32 s6, s33, s8                                      // 000000004360: 8B060821
	s_wait_alu depctr_sa_sdst(0)                               // 000000004364: BF88FF9E
	s_or_b32 s4, s5, s4                                        // 000000004368: 8C040405
	s_and_b32 s8, s33, s10                                     // 00000000436C: 8B080A21
	v_wmma_f32_16x16x16_f16 v[1:8], v[71:74], v[9:12], v[1:8]  // 000000004370: CC404001 1C061347
	s_and_b32 s10, s33, s12                                    // 000000004378: 8B0A0C21
	s_and_b32 s12, s33, s13                                    // 00000000437C: 8B0C0D21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004380: BF870111
	v_dual_mul_f32 v1, s42, v1 :: v_dual_mul_f32 v2, s42, v2   // 000000004384: C8C6022A 0102042A
	v_dual_mul_f32 v3, s42, v3 :: v_dual_mul_f32 v4, s42, v4   // 00000000438C: C8C6062A 0304082A
	v_dual_mul_f32 v5, s42, v5 :: v_dual_mul_f32 v6, s42, v6   // 000000004394: C8C60A2A 05060C2A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000439C: BF88FF9E
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000043A0: BF870003
	v_cndmask_b32_e64 v1, v1, 0xf149f2ca, s4                   // 0000000043A4: D5010001 0011FF01 F149F2CA
	s_or_b32 s4, s5, s6                                        // 0000000043B0: 8C040605
	v_dual_mul_f32 v7, s42, v7 :: v_dual_mul_f32 v8, s42, v8   // 0000000043B4: C8C60E2A 0708102A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043BC: BF88FF9E
	v_cndmask_b32_e64 v2, v2, 0xf149f2ca, s4                   // 0000000043C0: D5010002 0011FF02 F149F2CA
	s_or_b32 s4, s5, s8                                        // 0000000043CC: 8C040805
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043D0: BF88FF9E
	v_cndmask_b32_e64 v3, v3, 0xf149f2ca, s4                   // 0000000043D4: D5010003 0011FF03 F149F2CA
	s_or_b32 s4, s5, s10                                       // 0000000043E0: 8C040A05
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043E4: BF88FF9E
	v_cndmask_b32_e64 v4, v4, 0xf149f2ca, s4                   // 0000000043E8: D5010004 0011FF04 F149F2CA
	s_or_b32 s4, s5, s7                                        // 0000000043F4: 8C040705
	s_wait_alu depctr_sa_sdst(0)                               // 0000000043F8: BF88FF9E
	v_cndmask_b32_e64 v5, v5, 0xf149f2ca, s4                   // 0000000043FC: D5010005 0011FF05 F149F2CA
	s_or_b32 s4, s5, s9                                        // 000000004408: 8C040905
	s_wait_alu depctr_sa_sdst(0)                               // 00000000440C: BF88FF9E
	v_cndmask_b32_e64 v6, v6, 0xf149f2ca, s4                   // 000000004410: D5010006 0011FF06 F149F2CA
	s_or_b32 s4, s5, s11                                       // 00000000441C: 8C040B05
	s_wait_alu depctr_sa_sdst(0)                               // 000000004420: BF88FF9E
	v_cndmask_b32_e64 v7, v7, 0xf149f2ca, s4                   // 000000004424: D5010007 0011FF07 F149F2CA
	s_or_b32 s4, s5, s12                                       // 000000004430: 8C040C05
	s_wait_alu depctr_sa_sdst(0)                               // 000000004434: BF88FF9E
	v_cndmask_b32_e64 v8, v8, 0xf149f2ca, s4                   // 000000004438: D5010008 0011FF08 F149F2CA
	ds_store_b32 v45, v1                                       // 000000004444: D8340000 0000012D
	ds_store_b32 v46, v2                                       // 00000000444C: D8340000 0000022E
	ds_store_b32 v47, v3                                       // 000000004454: D8340000 0000032F
	ds_store_b32 v48, v4                                       // 00000000445C: D8340000 00000430
	ds_store_b32 v49, v5                                       // 000000004464: D8340000 00000531
	ds_store_b32 v50, v6                                       // 00000000446C: D8340000 00000632
	ds_store_b32 v51, v7                                       // 000000004474: D8340000 00000733
	ds_store_b32 v52, v8                                       // 00000000447C: D8340000 00000834
	s_wait_dscnt 0x0                                           // 000000004484: BFC60000
	s_barrier_signal -1                                        // 000000004488: BE804EC1
	s_barrier_wait 0xffff                                      // 00000000448C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004490: EE0AC07C 00040000 00000000
	s_and_saveexec_b32 s5, s2                                  // 00000000449C: BE852002
	s_cbranch_execz 63134                                      // 0000000044A0: BFA5F69E <attention_forward+0x41c>
	ds_load_b128 v[6:9], v53 offset:4096                       // 0000000044A4: DBFC1000 06000035
	ds_load_b128 v[10:13], v53 offset:4112                     // 0000000044AC: DBFC1010 0A000035
	ds_load_b128 v[67:70], v53 offset:4128                     // 0000000044B4: DBFC1020 43000035
	ds_load_b32 v15, v55                                       // 0000000044BC: D8D80000 0F000037
	s_wait_dscnt 0x3                                           // 0000000044C4: BFC60003
	v_max_num_f32_e32 v1, v6, v6                               // 0000000044C8: 2C020D06
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000044CC: BF870091
	v_max_num_f32_e32 v1, 0xf149f2ca, v1                       // 0000000044D0: 2C0202FF F149F2CA
	v_max3_num_f32 v5, v1, v7, v8                              // 0000000044D8: D62A0005 04220F01
	ds_load_b128 v[1:4], v53 offset:4144                       // 0000000044E0: DBFC1030 01000035
	s_wait_dscnt 0x3                                           // 0000000044E8: BFC60003
	v_max3_num_f32 v5, v5, v9, v10                             // 0000000044EC: D62A0005 042A1305
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000044F4: BF8700C1
	v_max3_num_f32 v14, v5, v11, v12                           // 0000000044F8: D62A000E 04321705
	ds_load_b32 v5, v54                                        // 000000004500: D8D80000 05000036
	s_wait_dscnt 0x3                                           // 000000004508: BFC60003
	v_max3_num_f32 v14, v14, v13, v67                          // 00000000450C: D62A000E 050E1B0E
	v_max3_num_f32 v14, v14, v68, v69                          // 000000004514: D62A000E 0516890E
	s_wait_dscnt 0x1                                           // 00000000451C: BFC60001
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004520: BF870091
	v_max3_num_f32 v14, v14, v70, v1                           // 000000004524: D62A000E 04068D0E
	v_max3_num_f32 v14, v14, v2, v3                            // 00000000452C: D62A000E 040E050E
	s_wait_dscnt 0x0                                           // 000000004534: BFC60000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004538: BF870091
	v_max3_num_f32 v14, v5, v14, v4                            // 00000000453C: D62A000E 04121D05
	v_sub_f32_e32 v16, v5, v14                                 // 000000004544: 08201D05
	v_dual_sub_f32 v6, v6, v14 :: v_dual_sub_f32 v7, v7, v14   // 000000004548: C94A1D06 06061D07
	v_dual_sub_f32 v10, v10, v14 :: v_dual_sub_f32 v11, v11, v14// 000000004550: C94A1D0A 0A0A1D0B
	v_dual_sub_f32 v8, v8, v14 :: v_dual_sub_f32 v9, v9, v14   // 000000004558: C94A1D08 08081D09
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000004560: BF870212
	v_dual_mul_f32 v71, 0x3fb8aa3b, v16 :: v_dual_mul_f32 v76, 0x3fb8aa3b, v10// 000000004564: C8C620FF 474C14FF 3FB8AA3B
	v_mul_f32_e32 v72, 0x3fb8aa3b, v6                          // 000000004570: 10900CFF 3FB8AA3B
	v_dual_sub_f32 v12, v12, v14 :: v_dual_sub_f32 v13, v13, v14// 000000004578: C94A1D0C 0C0C1D0D
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000004580: BF870194
	v_dual_mul_f32 v73, 0x3fb8aa3b, v7 :: v_dual_mul_f32 v74, 0x3fb8aa3b, v8// 000000004584: C8C60EFF 494A10FF 3FB8AA3B
	v_fma_f32 v83, 0x3fb8aa3b, v6, -v72                        // 000000004590: D6130053 85220CFF 3FB8AA3B
	v_rndne_f32_e32 v84, v72                                   // 00000000459C: 7EA84748
	v_fma_f32 v91, 0x3fb8aa3b, v10, -v76                       // 0000000045A0: D613005B 853214FF 3FB8AA3B
	v_rndne_f32_e32 v92, v76                                   // 0000000045AC: 7EB8474C
	v_dual_mul_f32 v75, 0x3fb8aa3b, v9 :: v_dual_mul_f32 v78, 0x3fb8aa3b, v12// 0000000045B0: C8C612FF 4B4E18FF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000045BC: BF870004
	v_dual_mul_f32 v77, 0x3fb8aa3b, v11 :: v_dual_sub_f32 v72, v72, v84// 0000000045C0: C8CA16FF 4D48A948 3FB8AA3B
	v_fma_f32 v81, 0x3fb8aa3b, v16, -v71                       // 0000000045CC: D6130051 851E20FF 3FB8AA3B
	v_rndne_f32_e32 v82, v71                                   // 0000000045D8: 7EA44747
	v_fma_f32 v85, 0x3fb8aa3b, v7, -v73                        // 0000000045DC: D6130055 85260EFF 3FB8AA3B
	v_rndne_f32_e32 v86, v73                                   // 0000000045E8: 7EAC4749
	v_fma_f32 v87, 0x3fb8aa3b, v8, -v74                        // 0000000045EC: D6130057 852A10FF 3FB8AA3B
	v_rndne_f32_e32 v88, v74                                   // 0000000045F8: 7EB0474A
	v_dual_fmac_f32 v83, 0x32a5705f, v6 :: v_dual_sub_f32 v76, v76, v92// 0000000045FC: C80A0CFF 534CB94C 32A5705F
	v_fmac_f32_e32 v91, 0x32a5705f, v10                        // 000000004608: 56B614FF 32A5705F
	v_rndne_f32_e32 v96, v78                                   // 000000004610: 7EC0474E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004614: BF870003
	v_dual_fmac_f32 v81, 0x32a5705f, v16 :: v_dual_add_f32 v72, v72, v83// 000000004618: C80820FF 5148A748 32A5705F
	v_dual_sub_f32 v71, v71, v82 :: v_dual_sub_f32 v74, v74, v88// 000000004624: C94AA547 474AB14A
	v_fmac_f32_e32 v85, 0x32a5705f, v7                         // 00000000462C: 56AA0EFF 32A5705F
	v_dual_fmac_f32 v87, 0x32a5705f, v8 :: v_dual_add_f32 v76, v76, v91// 000000004634: C80810FF 574CB74C 32A5705F
	v_sub_f32_e32 v73, v73, v86                                // 000000004640: 0892AD49
	v_fma_f32 v95, 0x3fb8aa3b, v12, -v78                       // 000000004644: D613005F 853A18FF 3FB8AA3B
	v_sub_f32_e32 v78, v78, v96                                // 000000004650: 089CC14E
	s_delay_alu instid0(VALU_DEP_4)                            // 000000004654: BF870004
	v_add_f32_e32 v74, v74, v87                                // 000000004658: 0694AF4A
	v_exp_f32_e32 v72, v72                                     // 00000000465C: 7E904B48
	v_add_f32_e32 v73, v73, v85                                // 000000004660: 0692AB49
	v_cvt_i32_f32_e32 v84, v84                                 // 000000004664: 7EA81154
	v_cvt_i32_f32_e32 v86, v86                                 // 000000004668: 7EAC1156
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v6                       // 00000000466C: D41B0004 02020CFF C2CE8ED0
	v_exp_f32_e32 v74, v74                                     // 000000004678: 7E944B4A
	v_exp_f32_e32 v73, v73                                     // 00000000467C: 7E924B49
	v_cvt_i32_f32_e32 v88, v88                                 // 000000004680: 7EB01158
	v_fma_f32 v89, 0x3fb8aa3b, v9, -v75                        // 000000004684: D6130059 852E12FF 3FB8AA3B
	v_ldexp_f32 v72, v72, v84                                  // 000000004690: D71C0048 0202A948
	v_rndne_f32_e32 v90, v75                                   // 000000004698: 7EB4474B
	v_dual_fmac_f32 v95, 0x32a5705f, v12 :: v_dual_sub_f32 v70, v70, v14// 00000000469C: C80A18FF 5F461D46 32A5705F
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000046A8: BF870004
	v_fmac_f32_e32 v89, 0x32a5705f, v9                         // 0000000046AC: 56B212FF 32A5705F
	s_wait_alu depctr_va_sdst(0)                               // 0000000046B4: BF88F19F
	v_cndmask_b32_e64 v72, 0, v72, s4                          // 0000000046B8: D5010048 00129080
	v_ldexp_f32 v73, v73, v86                                  // 0000000046C0: D71C0049 0202AD49
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v7                       // 0000000046C8: D41B0004 02020EFF C2CE8ED0
	v_ldexp_f32 v74, v74, v88                                  // 0000000046D4: D71C004A 0202B14A
	v_dual_sub_f32 v69, v69, v14 :: v_dual_add_f32 v78, v78, v95// 0000000046DC: C9481D45 454EBF4E
	v_exp_f32_e32 v76, v76                                     // 0000000046E4: 7E984B4C
	s_wait_alu depctr_va_sdst(0)                               // 0000000046E8: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 0000000046EC: D5010049 00129280
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v8                       // 0000000046F4: D41B0004 020210FF C2CE8ED0
	v_cvt_i32_f32_e32 v92, v92                                 // 000000004700: 7EB8115C
	v_fma_f32 v93, 0x3fb8aa3b, v11, -v77                       // 000000004704: D613005D 853616FF 3FB8AA3B
	v_rndne_f32_e32 v94, v77                                   // 000000004710: 7EBC474D
	v_exp_f32_e32 v78, v78                                     // 000000004714: 7E9C4B4E
	s_wait_alu depctr_va_sdst(0)                               // 000000004718: BF88F19F
	v_cndmask_b32_e64 v74, 0, v74, s4                          // 00000000471C: D501004A 00129480
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v6                       // 000000004724: D41E0004 02020CFF 42B17218
	v_ldexp_f32 v76, v76, v92                                  // 000000004730: D71C004C 0202B94C
	v_sub_f32_e32 v77, v77, v94                                // 000000004738: 089ABD4D
	v_cvt_i32_f32_e32 v94, v94                                 // 00000000473C: 7EBC115E
	v_cvt_i32_f32_e32 v83, v96                                 // 000000004740: 7EA61160
	s_wait_alu depctr_va_sdst(0)                               // 000000004744: BF88F19F
	v_cndmask_b32_e64 v6, 0x7f800000, v72, s4                  // 000000004748: D5010006 001290FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v7                       // 000000004754: D41E0004 02020EFF 42B17218
	v_dual_sub_f32 v67, v67, v14 :: v_dual_sub_f32 v68, v68, v14// 000000004760: C94A1D43 43441D44
	v_ldexp_f32 v78, v78, v83                                  // 000000004768: D71C004E 0202A74E
	v_mul_f32_e32 v79, 0x3fb8aa3b, v13                         // 000000004770: 109E1AFF 3FB8AA3B
	s_wait_alu depctr_va_sdst(0)                               // 000000004778: BF88F19F
	v_cndmask_b32_e64 v7, 0x7f800000, v73, s4                  // 00000000477C: D5010007 001292FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v9                       // 000000004788: D41B0004 020212FF C2CE8ED0
	v_dual_mul_f32 v80, 0x3fb8aa3b, v67 :: v_dual_sub_f32 v1, v1, v14// 000000004794: C8CA86FF 50001D01 3FB8AA3B
	v_fma_f32 v97, 0x3fb8aa3b, v13, -v79                       // 0000000047A0: D6130061 853E1AFF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000047AC: BF870004
	v_add_f32_e32 v73, v6, v7                                  // 0000000047B0: 06920F06
	v_sub_f32_e32 v75, v75, v90                                // 0000000047B4: 0896B54B
	v_cvt_i32_f32_e32 v90, v90                                 // 0000000047B8: 7EB4115A
	v_rndne_f32_e32 v98, v79                                   // 0000000047BC: 7EC4474F
	v_fma_f32 v99, 0x3fb8aa3b, v67, -v80                       // 0000000047C0: D6130063 854286FF 3FB8AA3B
	v_rndne_f32_e32 v100, v80                                  // 0000000047CC: 7EC84750
	v_add_f32_e32 v75, v75, v89                                // 0000000047D0: 0696B34B
	v_dual_add_f32 v71, v71, v81 :: v_dual_sub_f32 v2, v2, v14 // 0000000047D4: C90AA347 47021D02
	v_sub_f32_e32 v79, v79, v98                                // 0000000047DC: 089EC54F
	v_fmac_f32_e32 v99, 0x32a5705f, v67                        // 0000000047E0: 56C686FF 32A5705F
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(TRANS32_DEP_2)// 0000000047E8: BF870354
	v_exp_f32_e32 v75, v75                                     // 0000000047EC: 7E964B4B
	v_sub_f32_e32 v3, v3, v14                                  // 0000000047F0: 08061D03
	v_cvt_i32_f32_e32 v81, v100                                // 0000000047F4: 7EA21164
	v_exp_f32_e32 v71, v71                                     // 0000000047F8: 7E8E4B47
	v_sub_f32_e32 v4, v4, v14                                  // 0000000047FC: 08081D04
	v_ldexp_f32 v75, v75, v90                                  // 000000004800: D71C004B 0202B54B
	s_wait_alu depctr_va_sdst(0)                               // 000000004808: BF88F19F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000480C: BF870141
	v_cndmask_b32_e64 v72, 0, v75, s4                          // 000000004810: D5010048 00129680
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v8                       // 000000004818: D41E0004 020210FF 42B17218
	v_sub_f32_e32 v75, v80, v100                               // 000000004824: 0896C950
	s_wait_alu depctr_va_sdst(0)                               // 000000004828: BF88F19F
	v_cndmask_b32_e64 v8, 0x7f800000, v74, s4                  // 00000000482C: D5010008 001294FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v10                      // 000000004838: D41B0004 020214FF C2CE8ED0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004844: BF870123
	v_add_f32_e32 v75, v75, v99                                // 000000004848: 0696C74B
	s_wait_alu depctr_va_sdst(0)                               // 00000000484C: BF88F19F
	v_cndmask_b32_e64 v74, 0, v76, s4                          // 000000004850: D501004A 00129880
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v9                       // 000000004858: D41E0004 020212FF 42B17218
	v_mul_f32_e32 v76, 0x3fb8aa3b, v68                         // 000000004864: 109888FF 3FB8AA3B
	s_wait_alu depctr_va_sdst(0)                               // 00000000486C: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000004870: BF870132
	v_cndmask_b32_e64 v9, 0x7f800000, v72, s4                  // 000000004874: D5010009 001290FF 7F800000
	v_add_f32_e32 v72, v8, v73                                 // 000000004880: 06909308
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v11                      // 000000004884: D41B0004 020216FF C2CE8ED0
	v_dual_add_f32 v72, v9, v72 :: v_dual_fmac_f32 v93, 0x32a5705f, v11// 000000004890: C9009109 485C16FF 32A5705F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000489C: BF870091
	v_add_f32_e32 v77, v77, v93                                // 0000000048A0: 069ABB4D
	v_exp_f32_e32 v77, v77                                     // 0000000048A4: 7E9A4B4D
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000048A8: BF8700A5
	v_ldexp_f32 v77, v77, v94                                  // 0000000048AC: D71C004D 0202BD4D
	s_wait_alu depctr_va_sdst(0)                               // 0000000048B4: BF88F19F
	v_cndmask_b32_e64 v73, 0, v77, s4                          // 0000000048B8: D5010049 00129A80
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v10                      // 0000000048C0: D41E0004 020214FF 42B17218
	v_fma_f32 v77, 0x3fb8aa3b, v68, -v76                       // 0000000048CC: D613004D 853288FF 3FB8AA3B
	s_wait_alu depctr_va_sdst(0)                               // 0000000048D8: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000048DC: BF8701A2
	v_cndmask_b32_e64 v10, 0x7f800000, v74, s4                 // 0000000048E0: D501000A 001294FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v12                      // 0000000048EC: D41B0004 020218FF C2CE8ED0
	v_fmac_f32_e32 v77, 0x32a5705f, v68                        // 0000000048F8: 569A88FF 32A5705F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000004900: BF8701A3
	v_add_f32_e32 v72, v10, v72                                // 000000004904: 0690910A
	s_wait_alu depctr_va_sdst(0)                               // 000000004908: BF88F19F
	v_cndmask_b32_e64 v74, 0, v78, s4                          // 00000000490C: D501004A 00129C80
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v11                      // 000000004914: D41E0004 020216FF 42B17218
	v_rndne_f32_e32 v78, v76                                   // 000000004920: 7E9C474C
	s_wait_alu depctr_va_sdst(0)                               // 000000004924: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000004928: BF8701B2
	v_cndmask_b32_e64 v11, 0x7f800000, v73, s4                 // 00000000492C: D501000B 001292FF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v12                      // 000000004938: D41E0004 020218FF 42B17218
	v_cvt_i32_f32_e32 v73, v98                                 // 000000004944: 7E921162
	v_add_f32_e32 v72, v11, v72                                // 000000004948: 0690910B
	s_wait_alu depctr_va_sdst(0)                               // 00000000494C: BF88F19F
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000004950: BF8701C3
	v_cndmask_b32_e64 v12, 0x7f800000, v74, s4                 // 000000004954: D501000C 001294FF 7F800000
	v_fmac_f32_e32 v97, 0x32a5705f, v13                        // 000000004960: 56C21AFF 32A5705F
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v13                      // 000000004968: D41B0004 02021AFF C2CE8ED0
	v_cvt_i32_f32_e32 v74, v82                                 // 000000004974: 7E941152
	v_dual_add_f32 v72, v12, v72 :: v_dual_add_f32 v79, v79, v97// 000000004978: C908910C 484EC34F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004980: BF870112
	v_ldexp_f32 v71, v71, v74                                  // 000000004984: D71C0047 02029547
	v_exp_f32_e32 v79, v79                                     // 00000000498C: 7E9E4B4F
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000004990: BF870135
	v_ldexp_f32 v73, v79, v73                                  // 000000004994: D71C0049 0202934F
	v_mul_f32_e32 v79, 0x3fb8aa3b, v69                         // 00000000499C: 109E8AFF 3FB8AA3B
	s_wait_alu depctr_va_sdst(0)                               // 0000000049A4: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 0000000049A8: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v13                      // 0000000049B0: D41E0004 02021AFF 42B17218
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000049BC: BF870123
	v_rndne_f32_e32 v80, v79                                   // 0000000049C0: 7EA0474F
	s_wait_alu depctr_va_sdst(0)                               // 0000000049C4: BF88F19F
	v_cndmask_b32_e64 v13, 0x7f800000, v73, s4                 // 0000000049C8: D501000D 001292FF 7F800000
	v_exp_f32_e32 v73, v75                                     // 0000000049D4: 7E924B4B
	v_sub_f32_e32 v75, v76, v78                                // 0000000049D8: 08969D4C
	v_fma_f32 v76, 0x3fb8aa3b, v69, -v79                       // 0000000049DC: D613004C 853E8AFF 3FB8AA3B
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v67                      // 0000000049E8: D41B0004 020286FF C2CE8ED0
	v_cvt_i32_f32_e32 v78, v78                                 // 0000000049F4: 7E9C114E
	v_cvt_i32_f32_e32 v74, v80                                 // 0000000049F8: 7E941150
	v_add_f32_e32 v75, v75, v77                                // 0000000049FC: 06969B4B
	v_dual_fmac_f32 v76, 0x32a5705f, v69 :: v_dual_sub_f32 v77, v79, v80// 000000004A00: C80A8AFF 4C4CA14F 32A5705F
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000004A0C: BF870225
	v_ldexp_f32 v73, v73, v81                                  // 000000004A10: D71C0049 0202A349
	v_mul_f32_e32 v79, 0x3fb8aa3b, v70                         // 000000004A18: 109E8CFF 3FB8AA3B
	v_exp_f32_e32 v75, v75                                     // 000000004A20: 7E964B4B
	v_add_f32_e32 v72, v13, v72                                // 000000004A24: 0690910D
	v_add_f32_e32 v76, v77, v76                                // 000000004A28: 0698994D
	s_wait_alu depctr_va_sdst(0)                               // 000000004A2C: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000004A30: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v67                      // 000000004A38: D41E0004 020286FF 42B17218
	v_fma_f32 v77, 0x3fb8aa3b, v70, -v79                       // 000000004A44: D613004D 853E8CFF 3FB8AA3B
	v_rndne_f32_e32 v81, v79                                   // 000000004A50: 7EA2474F
	v_exp_f32_e32 v76, v76                                     // 000000004A54: 7E984B4C
	s_wait_alu depctr_va_sdst(0)                               // 000000004A58: BF88F19F
	v_cndmask_b32_e64 v67, 0x7f800000, v73, s4                 // 000000004A5C: D5010043 001292FF 7F800000
	v_ldexp_f32 v73, v75, v78                                  // 000000004A68: D71C0049 02029D4B
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v68                      // 000000004A70: D41B0004 020288FF C2CE8ED0
	v_fmac_f32_e32 v77, 0x32a5705f, v70                        // 000000004A7C: 569A8CFF 32A5705F
	v_dual_sub_f32 v79, v79, v81 :: v_dual_mul_f32 v78, 0x3fb8aa3b, v3// 000000004A84: C946A34F 4F4E06FF 3FB8AA3B
	v_add_f32_e32 v72, v67, v72                                // 000000004A90: 06909143
	s_wait_alu depctr_va_sdst(0)                               // 000000004A94: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000004A98: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v68                      // 000000004AA0: D41E0004 020288FF 42B17218
	v_add_f32_e32 v75, v79, v77                                // 000000004AAC: 06969B4F
	v_ldexp_f32 v74, v76, v74                                  // 000000004AB0: D71C004A 0202954C
	v_dual_mul_f32 v76, 0x3fb8aa3b, v1 :: v_dual_mul_f32 v77, 0x3fb8aa3b, v2// 000000004AB8: C8C602FF 4C4C04FF 3FB8AA3B
	s_wait_alu depctr_va_sdst(0)                               // 000000004AC4: BF88F19F
	v_cndmask_b32_e64 v68, 0x7f800000, v73, s4                 // 000000004AC8: D5010044 001292FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v69                      // 000000004AD4: D41B0004 02028AFF C2CE8ED0
	v_exp_f32_e32 v75, v75                                     // 000000004AE0: 7E964B4B
	v_fma_f32 v79, 0x3fb8aa3b, v3, -v78                        // 000000004AE4: D613004F 853A06FF 3FB8AA3B
	v_rndne_f32_e32 v80, v78                                   // 000000004AF0: 7EA0474E
	v_add_f32_e32 v72, v68, v72                                // 000000004AF4: 06909144
	s_wait_alu depctr_va_sdst(0)                               // 000000004AF8: BF88F19F
	v_cndmask_b32_e64 v73, 0, v74, s4                          // 000000004AFC: D5010049 00129480
	v_cvt_i32_f32_e32 v74, v81                                 // 000000004B04: 7E941151
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v69                      // 000000004B08: D41E0004 02028AFF 42B17218
	v_fmac_f32_e32 v79, 0x32a5705f, v3                         // 000000004B14: 569E06FF 32A5705F
	s_wait_alu depctr_va_sdst(0)                               // 000000004B1C: BF88F19F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004B20: BF870002
	v_cndmask_b32_e64 v69, 0x7f800000, v73, s4                 // 000000004B24: D5010045 001292FF 7F800000
	v_ldexp_f32 v73, v75, v74                                  // 000000004B30: D71C0049 0202954B
	v_fma_f32 v74, 0x3fb8aa3b, v1, -v76                        // 000000004B38: D613004A 853202FF 3FB8AA3B
	v_rndne_f32_e32 v75, v76                                   // 000000004B44: 7E96474C
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v70                      // 000000004B48: D41B0004 02028CFF C2CE8ED0
	v_add_f32_e32 v72, v69, v72                                // 000000004B54: 06909145
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000004B58: BF870214
	v_fmac_f32_e32 v74, 0x32a5705f, v1                         // 000000004B5C: 569402FF 32A5705F
	v_sub_f32_e32 v76, v76, v75                                // 000000004B64: 0898974C
	s_wait_alu depctr_va_sdst(0)                               // 000000004B68: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000004B6C: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v70                      // 000000004B74: D41E0004 02028CFF 42B17218
	v_cvt_i32_f32_e32 v75, v75                                 // 000000004B80: 7E96114B
	s_wait_alu depctr_va_sdst(0)                               // 000000004B84: BF88F19F
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004B88: BF870002
	v_cndmask_b32_e64 v70, 0x7f800000, v73, s4                 // 000000004B8C: D5010046 001292FF 7F800000
	v_add_f32_e32 v73, v76, v74                                // 000000004B98: 0692954C
	v_fma_f32 v74, 0x3fb8aa3b, v2, -v77                        // 000000004B9C: D613004A 853604FF 3FB8AA3B
	v_rndne_f32_e32 v76, v77                                   // 000000004BA8: 7E98474D
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v1                       // 000000004BAC: D41B0004 020202FF C2CE8ED0
	v_add_f32_e32 v72, v70, v72                                // 000000004BB8: 06909146
	v_exp_f32_e32 v73, v73                                     // 000000004BBC: 7E924B49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004BC0: BF870123
	v_dual_fmac_f32 v74, 0x32a5705f, v2 :: v_dual_sub_f32 v77, v77, v76// 000000004BC4: C80A04FF 4A4C994D 32A5705F
	v_cvt_i32_f32_e32 v76, v76                                 // 000000004BD0: 7E98114C
	v_dual_add_f32 v74, v77, v74 :: v_dual_sub_f32 v77, v78, v80// 000000004BD4: C90A954D 4A4CA14E
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000004BDC: BF8701A5
	v_ldexp_f32 v73, v73, v75                                  // 000000004BE0: D71C0049 02029749
	v_mul_f32_e32 v75, 0x3fb8aa3b, v4                          // 000000004BE8: 109608FF 3FB8AA3B
	v_exp_f32_e32 v74, v74                                     // 000000004BF0: 7E944B4A
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004BF4: BF870003
	v_add_f32_e32 v77, v77, v79                                // 000000004BF8: 069A9F4D
	s_wait_alu depctr_va_sdst(0)                               // 000000004BFC: BF88F19F
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000004C00: D5010049 00129280
	v_fma_f32 v78, 0x3fb8aa3b, v4, -v75                        // 000000004C08: D613004E 852E08FF 3FB8AA3B
	v_rndne_f32_e32 v81, v75                                   // 000000004C14: 7EA2474B
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v1                       // 000000004C18: D41E0004 020202FF 42B17218
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004C24: BF870292
	v_dual_fmac_f32 v78, 0x32a5705f, v4 :: v_dual_sub_f32 v75, v75, v81// 000000004C28: C80A08FF 4E4AA34B 32A5705F
	v_ldexp_f32 v74, v74, v76                                  // 000000004C34: D71C004A 0202994A
	s_wait_alu depctr_va_sdst(0)                               // 000000004C3C: BF88F19F
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004C40: BF870003
	v_cndmask_b32_e64 v1, 0x7f800000, v73, s4                  // 000000004C44: D5010001 001292FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v2                       // 000000004C50: D41B0004 020204FF C2CE8ED0
	v_exp_f32_e32 v76, v77                                     // 000000004C5C: 7E984B4D
	v_add_f32_e32 v73, v75, v78                                // 000000004C60: 06929D4B
	v_cvt_i32_f32_e32 v75, v80                                 // 000000004C64: 7E961150
	v_add_f32_e32 v72, v1, v72                                 // 000000004C68: 06909101
	s_wait_alu depctr_va_sdst(0)                               // 000000004C6C: BF88F19F
	v_cndmask_b32_e64 v74, 0, v74, s4                          // 000000004C70: D501004A 00129480
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v2                       // 000000004C78: D41E0004 020204FF 42B17218
	v_exp_f32_e32 v73, v73                                     // 000000004C84: 7E924B49
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004C88: BF870126
	v_ldexp_f32 v75, v76, v75                                  // 000000004C8C: D71C004B 0202974C
	s_wait_alu depctr_va_sdst(0)                               // 000000004C94: BF88F19F
	v_cndmask_b32_e64 v2, 0x7f800000, v74, s4                  // 000000004C98: D5010002 001294FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v3                       // 000000004CA4: D41B0004 020206FF C2CE8ED0
	v_cvt_i32_f32_e32 v74, v81                                 // 000000004CB0: 7E941151
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000004CB4: BF8701A3
	v_add_f32_e32 v72, v2, v72                                 // 000000004CB8: 06909102
	s_wait_alu depctr_va_sdst(0)                               // 000000004CBC: BF88F19F
	v_cndmask_b32_e64 v75, 0, v75, s4                          // 000000004CC0: D501004B 00129680
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v16                      // 000000004CC8: D41B0004 020220FF C2CE8ED0
	v_ldexp_f32 v73, v73, v74                                  // 000000004CD4: D71C0049 02029549
	s_wait_alu depctr_va_sdst(0)                               // 000000004CDC: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000004CE0: BF8700B2
	v_cndmask_b32_e64 v71, 0, v71, s4                          // 000000004CE4: D5010047 00128E80
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v3                       // 000000004CEC: D41E0004 020206FF 42B17218
	s_wait_alu depctr_va_sdst(0)                               // 000000004CF8: BF88F19F
	v_cndmask_b32_e64 v3, 0x7f800000, v75, s4                  // 000000004CFC: D5010003 001296FF 7F800000
	v_cmp_ngt_f32_e64 s4, 0xc2ce8ed0, v4                       // 000000004D08: D41B0004 020208FF C2CE8ED0
	s_wait_alu depctr_va_sdst(0)                               // 000000004D14: BF88F19F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000004D18: BF8700B1
	v_cndmask_b32_e64 v73, 0, v73, s4                          // 000000004D1C: D5010049 00129280
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v16                      // 000000004D24: D41E0004 020220FF 42B17218
	s_wait_alu depctr_va_sdst(0)                               // 000000004D30: BF88F19F
	v_cndmask_b32_e64 v16, 0x7f800000, v71, s4                 // 000000004D34: D5010010 00128EFF 7F800000
	v_cmp_nlt_f32_e64 s4, 0x42b17218, v4                       // 000000004D40: D41E0004 020208FF 42B17218
	v_add_f32_e32 v71, v3, v72                                 // 000000004D4C: 068E9103
	s_wait_alu depctr_va_sdst(0)                               // 000000004D50: BF88F19F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000004D54: BF8700B2
	v_cndmask_b32_e64 v4, 0x7f800000, v73, s4                  // 000000004D58: D5010004 001292FF 7F800000
	v_cmp_nge_f32_e64 s4, 0xf149f2ca, v5                       // 000000004D64: D4190004 02020AFF F149F2CA
	s_wait_alu depctr_va_sdst(0)                               // 000000004D70: BF88F19F
	v_cndmask_b32_e64 v5, 0, v16, s4                           // 000000004D74: D5010005 00122080
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004D7C: BF870113
	v_add_f32_e32 v16, v4, v71                                 // 000000004D80: 06208F04
	v_mul_f32_e32 v15, v5, v15                                 // 000000004D84: 101E1F05
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004D88: BF870001
	v_add_f32_e32 v15, v15, v16                                // 000000004D8C: 061E210F
	ds_store_b128 v53, v[6:9] offset:4096                      // 000000004D90: DB7C1000 00000635
	ds_store_b128 v53, v[10:13] offset:4112                    // 000000004D98: DB7C1010 00000A35
	ds_store_b128 v53, v[67:70] offset:4128                    // 000000004DA0: DB7C1020 00004335
	ds_store_b128 v53, v[1:4] offset:4144                      // 000000004DA8: DB7C1030 00000135
	ds_store_b32 v55, v15                                      // 000000004DB0: D8340000 00000F37
	ds_store_b32 v54, v14                                      // 000000004DB8: D8340000 00000E36
	ds_store_b32 v56, v5                                       // 000000004DC0: D8340000 00000538
	s_branch 62548                                             // 000000004DC8: BFA0F454 <attention_forward+0x41c>
	s_load_b64 s[4:5], s[0:1], 0x80                            // 000000004DCC: F4002100 F8000080
	s_branch 18                                                // 000000004DD4: BFA00012 <attention_forward+0x3320>
	s_wait_alu depctr_sa_sdst(0)                               // 000000004DD8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000004DDC: 8C7E067E
	v_add_co_u32 v1, vcc_lo, v17, 32                           // 000000004DE0: D7006A01 02014111
	s_wait_alu depctr_va_vcc(0)                                // 000000004DE8: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, 0, v18, vcc_lo               // 000000004DEC: D5207C02 01AA2480
	v_cmp_lt_u64_e32 vcc_lo, 0x3df, v[17:18]                   // 000000004DF4: 7CB222FF 000003DF
	v_add_nc_u32_e32 v43, 0x80, v43                            // 000000004DFC: 4A5656FF 00000080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004E04: BF870003
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v2           // 000000004E08: CA100101 11120102
	s_or_b32 s21, vcc_lo, s21                                  // 000000004E10: 8C15156A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E14: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s21                       // 000000004E18: 917E157E
	s_cbranch_execz 68                                         // 000000004E1C: BFA50044 <attention_forward+0x3430>
	v_lshrrev_b64 v[1:2], 6, v[17:18]                          // 000000004E20: D73D0001 02022286
	s_mov_b32 s6, exec_lo                                      // 000000004E28: BE86007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004E2C: BF870111
	v_or_b32_e32 v2, s29, v2                                   // 000000004E30: 3804041D
	v_or_b32_e32 v1, s28, v1                                   // 000000004E34: 3802021C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004E38: BF870001
	v_cmpx_gt_i64_e32 s[24:25], v[1:2]                         // 000000004E3C: 7DA80218
	s_cbranch_execz 65509                                      // 000000004E40: BFA5FFE5 <attention_forward+0x32d8>
	v_alignbit_b32 v3, v18, v17, 6                             // 000000004E44: D6160003 021A2312
	v_add_co_u32 v1, s3, v1, s22                               // 000000004E4C: D7000301 02002D01
	s_wait_alu depctr_va_sdst(0)                               // 000000004E54: BF88F19F
	v_add_co_ci_u32_e64 v2, null, s23, v2, s3                  // 000000004E58: D5207C02 000E0417
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004E60: BF870003
	v_lshlrev_b32_e32 v3, 2, v3                                // 000000004E64: 30060682
	ds_load_b32 v4, v43                                        // 000000004E68: D8D80000 0400002B
	ds_load_b32 v3, v3 offset:5184                             // 000000004E70: D8D81440 03000003
	v_lshlrev_b64_e32 v[1:2], 8, v[1:2]                        // 000000004E78: 3E020288
	s_wait_dscnt 0x0                                           // 000000004E7C: BFC60000
	v_div_scale_f32 v5, null, v3, v3, v4                       // 000000004E80: D6FC7C05 04120703
	v_div_scale_f32 v8, vcc_lo, v4, v3, v4                     // 000000004E88: D6FC6A08 04120704
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000004E90: BF870292
	v_rcp_f32_e32 v6, v5                                       // 000000004E94: 7E0C5505
	v_fma_f32 v7, -v5, v6, 1.0                                 // 000000004E98: D6130007 23CA0D05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004EA0: BF870091
	v_fmac_f32_e32 v6, v7, v6                                  // 000000004EA4: 560C0D07
	v_mul_f32_e32 v7, v8, v6                                   // 000000004EA8: 100E0D08
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004EAC: BF870091
	v_fma_f32 v9, -v5, v7, v8                                  // 000000004EB0: D6130009 24220F05
	v_fmac_f32_e32 v7, v9, v6                                  // 000000004EB8: 560E0D09
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000004EBC: BF870131
	v_fma_f32 v5, -v5, v7, v8                                  // 000000004EC0: D6130005 24220F05
	v_and_b32_e32 v8, 63, v17                                  // 000000004EC8: 361022BF
	s_wait_alu depctr_va_vcc(0)                                // 000000004ECC: BF88FF9D
	v_div_fmas_f32 v5, v5, v6, v7                              // 000000004ED0: D6370005 041E0D05
	s_wait_kmcnt 0x0                                           // 000000004ED8: BFC70000
	v_add_co_u32 v1, vcc_lo, s4, v1                            // 000000004EDC: D7006A01 02020204
	v_lshlrev_b32_e32 v6, 2, v8                                // 000000004EE4: 300C1082
	s_wait_alu depctr_va_vcc(0)                                // 000000004EE8: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, s5, v2, vcc_lo               // 000000004EEC: D5207C02 01AA0405
	v_div_fixup_f32 v4, v5, v3, v4                             // 000000004EF4: D6270004 04120705
	v_cmp_lt_f32_e32 vcc_lo, 0, v3                             // 000000004EFC: 7C220680
	s_wait_alu depctr_va_vcc(0)                                // 000000004F00: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000004F04: BF870002
	v_cndmask_b32_e32 v3, 0, v4, vcc_lo                        // 000000004F08: 02060880
	v_add_co_u32 v1, vcc_lo, v1, v6                            // 000000004F0C: D7006A01 02020D01
	s_wait_alu depctr_va_vcc(0)                                // 000000004F14: BF88FF9D
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo                // 000000004F18: D5207C02 01AA0480
	global_store_b32 v[1:2], v3, off                           // 000000004F20: EE06807C 01800000 00000001
	s_branch 65450                                             // 000000004F2C: BFA0FFAA <attention_forward+0x32d8>
	s_or_b32 exec_lo, exec_lo, s21                             // 000000004F30: 8C7E157E
	s_and_saveexec_b32 s3, s2                                  // 000000004F34: BE832002
	s_cbranch_execz 60                                         // 000000004F38: BFA5003C <attention_forward+0x352c>
	v_mov_b32_e32 v2, s29                                      // 000000004F3C: 7E04021D
	v_or_b32_e32 v1, s28, v0                                   // 000000004F40: 3802001C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F44: BF870001
	v_cmp_gt_i64_e32 vcc_lo, s[24:25], v[1:2]                  // 000000004F48: 7CA80218
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000004F4C: 8B7E6A7E
	s_cbranch_execz 54                                         // 000000004F50: BFA50036 <attention_forward+0x352c>
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000004F54: 30000082
	s_load_b64 s[0:1], s[0:1], 0xc8                            // 000000004F58: F4002000 F80000C8
	s_lshl_b64 s[2:3], s[22:23], 2                             // 000000004F60: 84828216
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F64: BF870001
	v_add_nc_u32_e32 v0, 0x1400, v0                            // 000000004F68: 4A0000FF 00001400
	ds_load_2addr_b32 v[3:4], v0 offset1:16                    // 000000004F70: D8DC1000 03000000
	s_wait_kmcnt 0x0                                           // 000000004F78: BFC70000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F7C: BF88FF9E
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]                        // 000000004F80: A9800200
	s_wait_dscnt 0x0                                           // 000000004F84: BFC60000
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v4                      // 000000004F88: 7C2808FF 00800000
	s_wait_alu depctr_va_vcc(0)                                // 000000004F90: BF88FF9D
	v_cndmask_b32_e64 v0, 0, 32, vcc_lo                        // 000000004F94: D5010000 01A94080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004F9C: BF870091
	v_ldexp_f32 v0, v4, v0                                     // 000000004FA0: D71C0000 02020104
	v_log_f32_e32 v0, v0                                       // 000000004FA8: 7E004F00
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FAC: BF870095
	v_mul_f32_e32 v4, 0x3f317217, v0                           // 000000004FB0: 100800FF 3F317217
	v_fma_f32 v5, 0x3f317217, v0, -v4                          // 000000004FB8: D6130005 841200FF 3F317217
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FC4: BF870091
	v_fmamk_f32 v5, v0, 0x3377d1cf, v5                         // 000000004FC8: 580A0B00 3377D1CF
	v_add_f32_e32 v4, v4, v5                                   // 000000004FD0: 06080B04
	v_cndmask_b32_e64 v5, 0, 0x41b17218, vcc_lo                // 000000004FD4: D5010005 01A9FE80 41B17218
	v_cmp_gt_f32_e64 vcc_lo, 0x7f800000, |v0|                  // 000000004FE0: D414026A 020200FF 7F800000
	s_wait_alu depctr_va_vcc(0)                                // 000000004FEC: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004FF0: BF870093
	v_cndmask_b32_e32 v0, v0, v4, vcc_lo                       // 000000004FF4: 02000900
	v_sub_f32_e32 v4, v0, v5                                   // 000000004FF8: 08080B00
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]                        // 000000004FFC: 3E000282
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005000: BF870112
	v_add_f32_e32 v2, v3, v4                                   // 000000005004: 06040903
	v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000005008: D7006A00 02020000
	s_wait_alu depctr_va_vcc(0)                                // 000000005010: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005014: BF870003
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo               // 000000005018: D5207C01 01AA0201
	global_store_b32 v[0:1], v2, off                           // 000000005020: EE06807C 01000000 00000000
	s_nop 0                                                    // 00000000502C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000005030: BFB60003
	s_endpgm                                                   // 000000005034: BFB00000
	s_code_end                                                 // 000000005038: BF9F0000
	s_code_end                                                 // 00000000503C: BF9F0000
	s_code_end                                                 // 000000005040: BF9F0000
	s_code_end                                                 // 000000005044: BF9F0000
	s_code_end                                                 // 000000005048: BF9F0000
	s_code_end                                                 // 00000000504C: BF9F0000
	s_code_end                                                 // 000000005050: BF9F0000
	s_code_end                                                 // 000000005054: BF9F0000
	s_code_end                                                 // 000000005058: BF9F0000
	s_code_end                                                 // 00000000505C: BF9F0000
	s_code_end                                                 // 000000005060: BF9F0000
	s_code_end                                                 // 000000005064: BF9F0000
	s_code_end                                                 // 000000005068: BF9F0000
	s_code_end                                                 // 00000000506C: BF9F0000
	s_code_end                                                 // 000000005070: BF9F0000
	s_code_end                                                 // 000000005074: BF9F0000
	s_code_end                                                 // 000000005078: BF9F0000
	s_code_end                                                 // 00000000507C: BF9F0000
	s_code_end                                                 // 000000005080: BF9F0000
	s_code_end                                                 // 000000005084: BF9F0000
	s_code_end                                                 // 000000005088: BF9F0000
	s_code_end                                                 // 00000000508C: BF9F0000
	s_code_end                                                 // 000000005090: BF9F0000
	s_code_end                                                 // 000000005094: BF9F0000
	s_code_end                                                 // 000000005098: BF9F0000
	s_code_end                                                 // 00000000509C: BF9F0000
	s_code_end                                                 // 0000000050A0: BF9F0000
	s_code_end                                                 // 0000000050A4: BF9F0000
	s_code_end                                                 // 0000000050A8: BF9F0000
	s_code_end                                                 // 0000000050AC: BF9F0000
	s_code_end                                                 // 0000000050B0: BF9F0000
	s_code_end                                                 // 0000000050B4: BF9F0000
	s_code_end                                                 // 0000000050B8: BF9F0000
	s_code_end                                                 // 0000000050BC: BF9F0000
	s_code_end                                                 // 0000000050C0: BF9F0000
	s_code_end                                                 // 0000000050C4: BF9F0000
	s_code_end                                                 // 0000000050C8: BF9F0000
	s_code_end                                                 // 0000000050CC: BF9F0000
	s_code_end                                                 // 0000000050D0: BF9F0000
	s_code_end                                                 // 0000000050D4: BF9F0000
	s_code_end                                                 // 0000000050D8: BF9F0000
	s_code_end                                                 // 0000000050DC: BF9F0000
	s_code_end                                                 // 0000000050E0: BF9F0000
	s_code_end                                                 // 0000000050E4: BF9F0000
	s_code_end                                                 // 0000000050E8: BF9F0000
	s_code_end                                                 // 0000000050EC: BF9F0000
	s_code_end                                                 // 0000000050F0: BF9F0000
	s_code_end                                                 // 0000000050F4: BF9F0000
	s_code_end                                                 // 0000000050F8: BF9F0000
	s_code_end                                                 // 0000000050FC: BF9F0000
	s_code_end                                                 // 000000005100: BF9F0000
	s_code_end                                                 // 000000005104: BF9F0000
	s_code_end                                                 // 000000005108: BF9F0000
	s_code_end                                                 // 00000000510C: BF9F0000
	s_code_end                                                 // 000000005110: BF9F0000
	s_code_end                                                 // 000000005114: BF9F0000
	s_code_end                                                 // 000000005118: BF9F0000
	s_code_end                                                 // 00000000511C: BF9F0000
	s_code_end                                                 // 000000005120: BF9F0000
	s_code_end                                                 // 000000005124: BF9F0000
	s_code_end                                                 // 000000005128: BF9F0000
	s_code_end                                                 // 00000000512C: BF9F0000
	s_code_end                                                 // 000000005130: BF9F0000
	s_code_end                                                 // 000000005134: BF9F0000
	s_code_end                                                 // 000000005138: BF9F0000
	s_code_end                                                 // 00000000513C: BF9F0000
	s_code_end                                                 // 000000005140: BF9F0000
	s_code_end                                                 // 000000005144: BF9F0000
	s_code_end                                                 // 000000005148: BF9F0000
	s_code_end                                                 // 00000000514C: BF9F0000
	s_code_end                                                 // 000000005150: BF9F0000
	s_code_end                                                 // 000000005154: BF9F0000
	s_code_end                                                 // 000000005158: BF9F0000
	s_code_end                                                 // 00000000515C: BF9F0000
	s_code_end                                                 // 000000005160: BF9F0000
	s_code_end                                                 // 000000005164: BF9F0000
	s_code_end                                                 // 000000005168: BF9F0000
	s_code_end                                                 // 00000000516C: BF9F0000
	s_code_end                                                 // 000000005170: BF9F0000
	s_code_end                                                 // 000000005174: BF9F0000
	s_code_end                                                 // 000000005178: BF9F0000
	s_code_end                                                 // 00000000517C: BF9F0000
	s_code_end                                                 // 000000005180: BF9F0000
	s_code_end                                                 // 000000005184: BF9F0000
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
