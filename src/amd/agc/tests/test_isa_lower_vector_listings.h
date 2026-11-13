static const struct listing_case vector_listings[] = {
   {"mac/fused/gfx10.3",
    "pc_0:\n"
    "   v0 = v_fmac_f32 v1, v2, v0\n"
    "   s_endpgm\n"},
   {"mac/fused/gfx11",
    "pc_0:\n"
    "   v0 = v_fmac_f32 v1, v2, v0\n"
    "   s_endpgm\n"},
   {"mac/fused/gfx12",
    "pc_0:\n"
    "   v0 = v_fmac_f32 v1, v2, v0\n"
    "   s_endpgm\n"},
   {"mad/fused_modifiers/gfx10.3",
    "pc_0:\n"
    "   v0 = v_fma_f32 -v1, |v2|, v3 clamp\n"
    "   s_endpgm\n"},
   {"mad/fused_modifiers/gfx11",
    "pc_0:\n"
    "   v0 = v_fma_f32 -v1, |v2|, v3 clamp\n"
    "   s_endpgm\n"},
   {"mad/fused_modifiers/gfx12",
    "pc_0:\n"
    "   v0 = v_fma_f32 -v1, |v2|, v3 clamp\n"
    "   s_endpgm\n"},
   {"mad_legacy/unfused/gfx10.3",
    "pc_0:\n"
    "   v4 = v_mul_legacy_f32 v1, |v2| vop3\n"
    "   v0 = v_add_f32 v4, -v3 vop3\n"
    "   s_endpgm\n"},
   {"mac_legacy/unfused/gfx10.3",
    "pc_0:\n"
    "   v3 = v_mul_legacy_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v3, v0 vop3\n"
    "   s_endpgm\n"},
   {"mad_legacy/unfused/gfx11",
    "pc_0:\n"
    "   v4 = v_mul_legacy_f32 v1, |v2| vop3\n"
    "   v0 = v_add_f32 v4, -v3 vop3\n"
    "   s_endpgm\n"},
   {"mac_legacy/unfused/gfx11",
    "pc_0:\n"
    "   v3 = v_mul_legacy_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v3, v0 vop3\n"
    "   s_endpgm\n"},
   {"mad_legacy/unfused/gfx11.5",
    "pc_0:\n"
    "   v4 = v_mul_legacy_f32 v1, |v2| vop3\n"
    "   v0 = v_add_f32 v4, -v3 vop3\n"
    "   s_endpgm\n"},
   {"mac_legacy/unfused/gfx11.5",
    "pc_0:\n"
    "   v3 = v_mul_legacy_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v3, v0 vop3\n"
    "   s_endpgm\n"},
   {"mad_legacy/unfused/gfx12",
    "pc_0:\n"
    "   v4 = v_mul_legacy_f32 v1, |v2| vop3\n"
    "   v0 = v_add_f32 v4, -v3 vop3\n"
    "   s_endpgm\n"},
   {"mac_legacy/unfused/gfx12",
    "pc_0:\n"
    "   v3 = v_mul_legacy_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v3, v0 vop3\n"
    "   s_endpgm\n"},
   {"mad_legacy/flushes_denormals",
    "pc_0:\n"
    "   s_denorm_mode imm:12\n"
    "   v4 = v_mul_legacy_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v4, v3 vop3\n"
    "   s_denorm_mode imm:15\n"
    "   s_endpgm\n"},
   {"mad/exact",
    "pc_0:\n"
    "   v4 = v_mul_f32 v1, v2 vop3\n"
    "   v0 = v_add_f32 v4, v3 vop3\n"
    "   s_endpgm\n"},
   {"sdwa/expanded_gfx11",
    "pc_0:\n"
    "   v3 = v_bfe_u32 v1, 0, 16\n"
    "   v4 = v_bfe_u32 v2, 16, 8\n"
    "   v5 = v_add_f32 v3, v4 vop3\n"
    "   v5 = v_lshlrev_b32 16, v5\n"
    "   v0 = v_bfi_b32 0xffff0000, v5, v0\n"
    "   s_endpgm\n"},
   {"sdwa/kept_gfx103",
    "pc_0:\n"
    "   v0[16:32] = v_add_f32 v1, v2 dst_sel:uword1 dst_preserve src0_sel:uword0 src1_sel:ubyte2\n"
    "   s_endpgm\n"},
   {"sdwa/move_neg",
    "pc_0:\n"
    "   v29 = v_xor_b32 0x80000000, v62\n"
    "   s_endpgm\n"},
   {"sdwa/move_abs",
    "pc_0:\n"
    "   v29 = v_and_b32 0x7fffffff, v62\n"
    "   s_endpgm\n"},
   {"sdwa/move_neg_abs",
    "pc_0:\n"
    "   v29 = v_or_b32 0x80000000, v62\n"
    "   s_endpgm\n"},
   {"sdwa/float_modifiers_stay_on_the_instruction",
    "pc_0:\n"
    "   v2 = v_mul_f32 -v1, v0 vop3\n"
    "   s_endpgm\n"},
   {"waitcnt/gfx11",
    "pc_0:\n"
    "   s_waitcnt lgkmcnt(0) vmcnt(0)\n"
    "   s_endpgm\n"},
   {"waitcnt/gfx12_counters",
    "pc_0:\n"
    "   s4 = s_load_dword s[0-1], 0 storage:buffer\n"
    "   v1 = buffer_load_dword s[8-11], v0, 0 offen storage:buffer\n"
    "   s_wait_loadcnt imm:0\n"
    "   s_wait_kmcnt imm:0\n"
    "   s_endpgm\n"},
   {"compare/cmpx_gfx11",
    "pc_0:\n"
    "   exec_lo = v_cmpx_eq_f32 0, v1\n"
    "   s_endpgm\n"},
   {"compare/cmpx_vop3_gfx11",
    "pc_0:\n"
    "   exec_lo = v_cmpx_eq_f32 0, v1 vop3\n"
    "   s_endpgm\n"},
   {"compare/eq_u32_gfx11",
    "pc_0:\n"
    "   vcc_lo = v_cmp_eq_u32 0, v1\n"
    "   s_endpgm\n"},
   {"compare/eq_u32_gfx103",
    "pc_0:\n"
    "   vcc_lo = v_cmp_eq_u32 0, v1\n"
    "   s_endpgm\n"},
   {"compare/constant_false_gfx12",
    "pc_0:\n"
    "   vcc_lo = s_mov_b32 0\n"
    "   s_endpgm\n"},
   {"compare/constant_true_gfx12",
    "pc_0:\n"
    "   vcc = s_mov_b64 exec\n"
    "   s_endpgm\n"},
   {"compare/constant_true_cmpx_gfx12",
    "pc_0:\n"
    "   s_endpgm\n"},
   {"export/parameter_without_ring_gfx11",
    "refused: exp at 0x0 is unsupported on gfx11: parameter exports require the attribute ring\n"},
   {"lds/add_src2/gfx10.3",
    "pc_0:\n"
    "   v2 = v_add_u32 0x408, v0\n"
    "   v3 = ds_read_b32 v2 storage:shared\n"
    "   s_waitcnt lgkmcnt(0)\n"
    "   ds_add_u32 v0, v3 storage:shared\n"
    "   s_waitcnt lgkmcnt(0)\n"
    "   s_endpgm\n"},
   {"lds/add_src2/gfx11",
    "pc_0:\n"
    "   v2 = v_add_u32 0x408, v0\n"
    "   v3 = ds_read_b32 v2 storage:shared\n"
    "   s_waitcnt lgkmcnt(0)\n"
    "   ds_add_u32 v0, v3 storage:shared\n"
    "   s_waitcnt lgkmcnt(0)\n"
    "   s_endpgm\n"},
   {"lds/add_src2/gfx12",
    "pc_0:\n"
    "   v2 = v_add_u32 0x408, v0\n"
    "   v3 = ds_read_b32 v2 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "   ds_add_u32 v0, v3 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "   s_endpgm\n"},
   {"thread_ids/unpacked_gfx11",
    "prologue:\n"
    "   v1 = v_bfe_u32 v0, 10, 10\n"
    "   v0 = v_and_b32 0x3ff, v0\n"
    "pc_0:\n"
    "   v2 = v_add_u32 v0, v1\n"
    "   s_endpgm\n"},
   {"thread_ids/separate_gfx103",
    "pc_0:\n"
    "   v2 = v_add_u32 v0, v1\n"
    "   s_endpgm\n"},
   {"thread_ids/not_compute_gfx11",
    "pc_0:\n"
    "   v2 = v_add_u32 v0, v1\n"
    "   s_endpgm\n"},
   {"t16/high_vgpr_gfx11",
    "pc_0:\n"
    "   v200 = v_add_f16 v1, v2 vop3\n"
    "   s_endpgm\n"},
   {"t16/low_vgpr_gfx11",
    "pc_0:\n"
    "   v0 = v_add_f16 v1, v2\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn/gfx10.3",
    "pc_0:\n"
    "   v4 = ds_cmpst_rtn_b32 v2, v5, v6 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn/gfx11",
    "pc_0:\n"
    "   v4 = ds_cmpst_rtn_b32 v2, v6, v5 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn_b64/gfx11",
    "pc_0:\n"
    "   v[4-5] = ds_cmpst_rtn_b64 v2, v[6-7], v[8-9] storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn/gfx11.5",
    "pc_0:\n"
    "   v4 = ds_cmpst_rtn_b32 v2, v6, v5 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn_b64/gfx11.5",
    "pc_0:\n"
    "   v[4-5] = ds_cmpst_rtn_b64 v2, v[6-7], v[8-9] storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn/gfx12",
    "pc_0:\n"
    "   v4 = ds_cmpst_rtn_b32 v2, v6, v5 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_rtn_b64/gfx12",
    "pc_0:\n"
    "   v[4-5] = ds_cmpst_rtn_b64 v2, v[6-7], v[8-9] storage:shared\n"
    "   s_endpgm\n"},
   {"lds/add_rtn_gfx11",
    "pc_0:\n"
    "   v4 = ds_add_rtn_u32 v2, v5 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_f32/gfx11",
    "pc_0:\n"
    "   v4 = ds_cmpst_rtn_f32 v2, v6, v5 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/cmpst_f32/gfx12",
    "pc_0:\n"
    "   s4 = s_mov_b32 exec_lo\n"
    "   s6 = s_mov_b32 vcc_lo\n"
    "   s8 = s_cselect_b32 1, 0, scc\n"
    "   v7 = ds_read_b32 v2 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "local_1:\n"
    "   s_nop\n"
    "   vcc_lo = v_cmp_eq_f32 v7, v5\n"
    "   exec_lo, scc = s_and_b32 exec_lo, vcc_lo\n"
    "   s_cbranch_execz -> local_3\n"
    ":\n"
    "   v8 = ds_cmpst_rtn_b32 v2, v6, v7 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "   vcc_lo = v_cmp_eq_u32 v8, v7\n"
    "   v7 = v_mov_b32 v8\n"
    "   exec_lo, scc = s_andn2_b32 exec_lo, vcc_lo\n"
    "   s_cbranch_execnz -> local_1\n"
    "local_3:\n"
    "   s_nop\n"
    "   exec_lo = s_mov_b32 s4\n"
    "   vcc_lo = s_mov_b32 s6\n"
    "   scc = s_cmp_lg_u32 s8, 0\n"
    "   v4 = v_mov_b32 v7\n"
    "   s_endpgm\n"},
   {"lds/wrap_rtn/gfx11",
    "pc_0:\n"
    "   v4 = ds_wrap_rtn_b32 v2, v5, v6 storage:shared\n"
    "   s_endpgm\n"},
   {"lds/wrap_rtn/gfx12",
    "pc_0:\n"
    "   s4 = s_mov_b32 exec_lo\n"
    "   s6 = s_mov_b32 vcc_lo\n"
    "   s8 = s_cselect_b32 1, 0, scc\n"
    "   v7 = ds_read_b32 v2 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "local_1:\n"
    "   s_nop\n"
    "   vcc_lo = v_cmp_ge_u32 v7, v5\n"
    "   v8 = v_sub_u32 v7, v5\n"
    "   v9 = v_add_u32 v7, v6\n"
    "   v9 = v_cndmask_b32 v9, v8, vcc_lo\n"
    "   v8 = ds_cmpst_rtn_b32 v2, v9, v7 storage:shared\n"
    "   s_wait_dscnt imm:0\n"
    "   vcc_lo = v_cmp_eq_u32 v8, v7\n"
    "   v7 = v_mov_b32 v8\n"
    "   exec_lo, scc = s_andn2_b32 exec_lo, vcc_lo\n"
    "   s_cbranch_execnz -> local_1\n"
    "local_3:\n"
    "   s_nop\n"
    "   exec_lo = s_mov_b32 s4\n"
    "   vcc_lo = s_mov_b32 s6\n"
    "   scc = s_cmp_lg_u32 s8, 0\n"
    "   v4 = v_mov_b32 v7\n"
    "   s_endpgm\n"},
   {"tables/vop3_0x16b_gfx103",
    "pc_0:\n"
    "   v0 = v_mul_lo_u32 v1, v2\n"
    "   s_endpgm\n"},
   {"tables/vop3_0x16b_gfx11",
    "pc_0:\n"
    "   v0 = v_mul_lo_i32 v1, v2\n"
    "   s_endpgm\n"},
   {"tables/endpgm_ordered_ps_done/gfx11",
    "refused: sop1_op_0x0 at 0x0 is unsupported on gfx1013: no ACO opcode for this target encoding\n"},
   {"tables/endpgm_ordered_ps_done/gfx11.5",
    "refused: sop1_op_0x0 at 0x0 is unsupported on gfx1013: no ACO opcode for this target encoding\n"},
   {"tables/endpgm_ordered_ps_done/gfx12",
    "refused: s_endpgm_ordered_ps_done at 0x0 is unsupported on gfx12: instruction was removed on the target\n"},
   {"tables/sopp_0x17_gfx12",
    "pc_0:\n"
    "pc_4:\n"
    "   s_endpgm\n"},
   {"tables/sopp_0x17_gfx11",
    "pc_0:\n"
    "   s_cbranch_cdbgsys -> pc_4\n"
    "pc_4:\n"
    "   s_endpgm\n"},
   {"tables/vop1_0x1b_gfx12",
    "pc_0:\n"
    "   v_pipeflush\n"
    "   s_endpgm\n"},
   {"mad64/overlap_gfx11",
    "pc_0:\n"
    "   v[6-7], s4 = v_mad_u64_u32 v2, v3, v[4-5]\n"
    "   v4 = v_mov_b32 v6\n"
    "   v5 = v_mov_b32 v7\n"
    "   s_endpgm\n"},
   {"mad64/overlap/gfx11.5",
    "pc_0:\n"
    "   v[4-5], s4 = v_mad_u64_u32 v2, v3, v[4-5]\n"
    "   s_endpgm\n"},
   {"mad64/overlap/gfx12",
    "pc_0:\n"
    "   v[4-5], s4 = v_mad_u64_u32 v2, v3, v[4-5]\n"
    "   s_endpgm\n"},
   {"mad64/disjoint_gfx11",
    "pc_0:\n"
    "   v[4-5], s4 = v_mad_u64_u32 v2, v3, v[6-7]\n"
    "   s_endpgm\n"},
   {"prologue/own_block_gfx115",
    "prologue:\n"
    "   v1 = v_bfe_u32 v0, 10, 10\n"
    "   v0 = v_and_b32 0x3ff, v0\n"
    "pc_0:\n"
    "   v3 = v_add_u32 v0, v1\n"
    "   scc = s_cmp_eq_u32 s0, 0\n"
    "   s_cbranch_scc0 -> pc_0\n"
    "pc_c:\n"
    "   s_endpgm\n"},
   {"prologue/none_gfx103",
    "pc_0:\n"
    "   v3 = v_add_u32 v0, v1\n"
    "   scc = s_cmp_eq_u32 s0, 0\n"
    "   s_cbranch_scc0 -> pc_0\n"
    "pc_c:\n"
    "   s_endpgm\n"},
   {"thread_ids/x_only",
    "pc_0:\n"
    "   v4 = v_mov_b32 v0\n"
    "   s_endpgm\n"},
   {"thread_ids/x_and_z",
    "prologue:\n"
    "   v0 = v_and_b32 0x3ff, v0\n"
    "pc_0:\n"
    "   v4 = v_mov_b32 v0\n"
    "   s_endpgm\n"},
   {"waits/entry_issued_types_gfx12",
    "pc_0:\n"
    "   v1 = buffer_load_dword s[8-11], v0, 0 offen storage:buffer\n"
    "   v[0-3] = image_sample s[8-15], s[16-19],  v1: undef, v[4-5] 2d storage:image\n"
    "   v2 = buffer_load_dword s[8-11], v0, 0 offen storage:buffer\n"
    "   s_wait_loadcnt imm:1\n"
    "   s_wait_samplecnt imm:0\n"
    "   s_endpgm\n"},
   {"waits/unknown_block_gfx12",
    "pc_0:\n"
    "   s_cbranch_scc0 -> pc_4\n"
    "pc_4:\n"
    "   v[0-3] = image_sample s[8-15], s[16-19],  v1: undef, v[4-5] 2d storage:image\n"
    "   s_wait_loadcnt imm:0\n"
    "   s_wait_samplecnt imm:1\n"
    "   s_wait_bvhcnt imm:0\n"
    "   s_wait_expcnt imm:0\n"
    "   s_wait_dscnt imm:0\n"
    "   s_wait_kmcnt imm:0\n"
    "   s_wait_storecnt imm:0\n"
    "   s_endpgm\n"},
   {"waits/scalar_loads_drain_gfx12",
    "pc_0:\n"
    "   s4 = s_load_dword s[0-1], 0 storage:buffer\n"
    "   v[2-3] = ds_read2_b32 v2 offset0:1 offset1:2 storage:shared\n"
    "   v[4-5] = ds_read2_b32 v2 offset0:1 offset1:2 storage:shared\n"
    "   s_wait_kmcnt imm:0\n"
    "   s_wait_dscnt imm:1\n"
    "   s_endpgm\n"},
};
