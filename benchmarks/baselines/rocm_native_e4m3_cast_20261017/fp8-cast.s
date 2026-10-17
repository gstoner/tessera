
/home/angstorms/scratch/rocm-native-gate-up-views-20261016/block-quant-probe/fp8-cast.hsaco:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1201"

Disassembly of section .text:

0000000000001700 <fp8_cast>:
	s_load_b64 s[4:5], s[0:1], 0x50                            // 000000001700: F4002100 F8000050
	s_mov_b32 s2, ttmp9                                        // 000000001708: BE820075
	s_ashr_i32 s3, ttmp9, 31                                   // 00000000170C: 86039F75
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001710: BF870499
	s_lshl_b64 s[2:3], s[2:3], 8                               // 000000001714: 84828802
	v_or_b32_e32 v0, s2, v0                                    // 000000001718: 38000002
	v_mov_b32_e32 v1, s3                                       // 00000000171C: 7E020203
	s_mov_b32 s2, exec_lo                                      // 000000001720: BE82007E
	s_wait_kmcnt 0x0                                           // 000000001724: BFC70000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001728: BF870001
	v_cmpx_gt_u64_e32 s[4:5], v[0:1]                           // 00000000172C: 7DB80004
	s_cbranch_execz 29                                         // 000000001730: BFA5001D <fp8_cast+0xa8>
	s_load_b64 s[2:3], s[0:1], 0x8                             // 000000001734: F4002080 F8000008
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]                        // 00000000173C: 3E040082
	s_load_b64 s[0:1], s[0:1], 0x30                            // 000000001740: F4002000 F8000030
	s_wait_kmcnt 0x0                                           // 000000001748: BFC70000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000174C: BF870091
	v_add_co_u32 v2, vcc_lo, s2, v2                            // 000000001750: D7006A02 02020402
	v_add_co_ci_u32_e64 v3, null, s3, v3, vcc_lo               // 000000001758: D5207C03 01AA0603
	v_add_co_u32 v0, vcc_lo, s0, v0                            // 000000001760: D7006A00 02020000
	s_wait_alu depctr_va_vcc(0)                                // 000000001768: BF88FF9D
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo               // 00000000176C: D5207C01 01AA0201
	global_load_b32 v2, v[2:3], off                            // 000000001774: EE05007C 00000002 00000002
	v_mov_b32_e32 v3, 0                                        // 000000001780: 7E060280
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001784: BF870131
	v_mov_b16_e32 v4.l, v3.l                                   // 000000001788: 7E083903
	v_mov_b16_e32 v4.h, v3.l                                   // 00000000178C: 7F083903
	s_wait_loadcnt 0x0                                         // 000000001790: BFC00000
	v_cvt_pk_fp8_f32 v4.l, v2, v2                              // 000000001794: D7690004 02020502
	global_store_b8 v[0:1], v4, off                            // 00000000179C: EE06007C 02000000 00000000
	s_endpgm                                                   // 0000000017A8: BFB00000
	s_code_end                                                 // 0000000017AC: BF9F0000
	s_code_end                                                 // 0000000017B0: BF9F0000
	s_code_end                                                 // 0000000017B4: BF9F0000
	s_code_end                                                 // 0000000017B8: BF9F0000
	s_code_end                                                 // 0000000017BC: BF9F0000
	s_code_end                                                 // 0000000017C0: BF9F0000
	s_code_end                                                 // 0000000017C4: BF9F0000
	s_code_end                                                 // 0000000017C8: BF9F0000
	s_code_end                                                 // 0000000017CC: BF9F0000
	s_code_end                                                 // 0000000017D0: BF9F0000
	s_code_end                                                 // 0000000017D4: BF9F0000
	s_code_end                                                 // 0000000017D8: BF9F0000
	s_code_end                                                 // 0000000017DC: BF9F0000
	s_code_end                                                 // 0000000017E0: BF9F0000
	s_code_end                                                 // 0000000017E4: BF9F0000
	s_code_end                                                 // 0000000017E8: BF9F0000
	s_code_end                                                 // 0000000017EC: BF9F0000
	s_code_end                                                 // 0000000017F0: BF9F0000
	s_code_end                                                 // 0000000017F4: BF9F0000
	s_code_end                                                 // 0000000017F8: BF9F0000
	s_code_end                                                 // 0000000017FC: BF9F0000
	s_code_end                                                 // 000000001800: BF9F0000
	s_code_end                                                 // 000000001804: BF9F0000
	s_code_end                                                 // 000000001808: BF9F0000
	s_code_end                                                 // 00000000180C: BF9F0000
	s_code_end                                                 // 000000001810: BF9F0000
	s_code_end                                                 // 000000001814: BF9F0000
	s_code_end                                                 // 000000001818: BF9F0000
	s_code_end                                                 // 00000000181C: BF9F0000
	s_code_end                                                 // 000000001820: BF9F0000
	s_code_end                                                 // 000000001824: BF9F0000
	s_code_end                                                 // 000000001828: BF9F0000
	s_code_end                                                 // 00000000182C: BF9F0000
	s_code_end                                                 // 000000001830: BF9F0000
	s_code_end                                                 // 000000001834: BF9F0000
	s_code_end                                                 // 000000001838: BF9F0000
	s_code_end                                                 // 00000000183C: BF9F0000
	s_code_end                                                 // 000000001840: BF9F0000
	s_code_end                                                 // 000000001844: BF9F0000
	s_code_end                                                 // 000000001848: BF9F0000
	s_code_end                                                 // 00000000184C: BF9F0000
	s_code_end                                                 // 000000001850: BF9F0000
	s_code_end                                                 // 000000001854: BF9F0000
	s_code_end                                                 // 000000001858: BF9F0000
	s_code_end                                                 // 00000000185C: BF9F0000
	s_code_end                                                 // 000000001860: BF9F0000
	s_code_end                                                 // 000000001864: BF9F0000
	s_code_end                                                 // 000000001868: BF9F0000
	s_code_end                                                 // 00000000186C: BF9F0000
	s_code_end                                                 // 000000001870: BF9F0000
	s_code_end                                                 // 000000001874: BF9F0000
	s_code_end                                                 // 000000001878: BF9F0000
	s_code_end                                                 // 00000000187C: BF9F0000
	s_code_end                                                 // 000000001880: BF9F0000
	s_code_end                                                 // 000000001884: BF9F0000
	s_code_end                                                 // 000000001888: BF9F0000
	s_code_end                                                 // 00000000188C: BF9F0000
	s_code_end                                                 // 000000001890: BF9F0000
	s_code_end                                                 // 000000001894: BF9F0000
	s_code_end                                                 // 000000001898: BF9F0000
	s_code_end                                                 // 00000000189C: BF9F0000
	s_code_end                                                 // 0000000018A0: BF9F0000
	s_code_end                                                 // 0000000018A4: BF9F0000
	s_code_end                                                 // 0000000018A8: BF9F0000
	s_code_end                                                 // 0000000018AC: BF9F0000
	s_code_end                                                 // 0000000018B0: BF9F0000
	s_code_end                                                 // 0000000018B4: BF9F0000
	s_code_end                                                 // 0000000018B8: BF9F0000
	s_code_end                                                 // 0000000018BC: BF9F0000
	s_code_end                                                 // 0000000018C0: BF9F0000
	s_code_end                                                 // 0000000018C4: BF9F0000
	s_code_end                                                 // 0000000018C8: BF9F0000
	s_code_end                                                 // 0000000018CC: BF9F0000
	s_code_end                                                 // 0000000018D0: BF9F0000
	s_code_end                                                 // 0000000018D4: BF9F0000
	s_code_end                                                 // 0000000018D8: BF9F0000
	s_code_end                                                 // 0000000018DC: BF9F0000
	s_code_end                                                 // 0000000018E0: BF9F0000
	s_code_end                                                 // 0000000018E4: BF9F0000
	s_code_end                                                 // 0000000018E8: BF9F0000
	s_code_end                                                 // 0000000018EC: BF9F0000
	s_code_end                                                 // 0000000018F0: BF9F0000
	s_code_end                                                 // 0000000018F4: BF9F0000
	s_code_end                                                 // 0000000018F8: BF9F0000
	s_code_end                                                 // 0000000018FC: BF9F0000
	s_code_end                                                 // 000000001900: BF9F0000
	s_code_end                                                 // 000000001904: BF9F0000
	s_code_end                                                 // 000000001908: BF9F0000
	s_code_end                                                 // 00000000190C: BF9F0000
	s_code_end                                                 // 000000001910: BF9F0000
	s_code_end                                                 // 000000001914: BF9F0000
	s_code_end                                                 // 000000001918: BF9F0000
	s_code_end                                                 // 00000000191C: BF9F0000
	s_code_end                                                 // 000000001920: BF9F0000
	s_code_end                                                 // 000000001924: BF9F0000
	s_code_end                                                 // 000000001928: BF9F0000
	s_code_end                                                 // 00000000192C: BF9F0000
	s_code_end                                                 // 000000001930: BF9F0000
	s_code_end                                                 // 000000001934: BF9F0000
	s_code_end                                                 // 000000001938: BF9F0000
	s_code_end                                                 // 00000000193C: BF9F0000
	s_code_end                                                 // 000000001940: BF9F0000
	s_code_end                                                 // 000000001944: BF9F0000
	s_code_end                                                 // 000000001948: BF9F0000
	s_code_end                                                 // 00000000194C: BF9F0000
	s_code_end                                                 // 000000001950: BF9F0000
	s_code_end                                                 // 000000001954: BF9F0000
	s_code_end                                                 // 000000001958: BF9F0000
	s_code_end                                                 // 00000000195C: BF9F0000
	s_code_end                                                 // 000000001960: BF9F0000
	s_code_end                                                 // 000000001964: BF9F0000
	s_code_end                                                 // 000000001968: BF9F0000
	s_code_end                                                 // 00000000196C: BF9F0000
	s_code_end                                                 // 000000001970: BF9F0000
	s_code_end                                                 // 000000001974: BF9F0000
	s_code_end                                                 // 000000001978: BF9F0000
	s_code_end                                                 // 00000000197C: BF9F0000
