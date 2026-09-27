// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::data_type::NumericType;
use crate::ir::*;
use crate::ops::{CmpAccumOp, CmpOp, CmpResultType, LogicOp, MuxOp,
                 OpCSel, OpFCmp, OpICmp, OpMux, OpShiftLop, ShiftOp};
use rustc_hash::{FxHashMap, FxHashSet};

struct Comparison {
    src_type: DataType,
    cmp_op: CmpOp,
    srcs: [Src; 2],
}

impl Comparison {
    fn from_instr(instr: &Instr) -> Option<(SSAValue, Self)> {
        let (dst, src_type, res_type, cmp_op, srcs, accum_op) = match &instr.op {
            Op::FCmp(op) => (&op.dst, op.src_type, op.res_type,
                            op.cmp_op, &op.srcs, op.accum_op),
            Op::ICmp(op) => (&op.dst, op.src_type, op.res_type,
                            op.cmp_op, &op.srcs, op.accum_op),
            _ => return None,
        };
        if !matches!(src_type, DataType::F32 | DataType::S32 | DataType::U32)
            || res_type != CmpResultType::M1 || accum_op != CmpAccumOp::None
            || dst.lanes != DstLanes::All || instr.flow != FlowCtrl::NONE
            || !matches!(cmp_op, CmpOp::Eq | CmpOp::Ne | CmpOp::Lt
                         | CmpOp::Le | CmpOp::Gt | CmpOp::Ge)
        {
            return None;
        }
        let DstRef::SSA(vec) = &dst.dst_ref else { return None; };
        if vec.comps() != 1 || vec[0].bits() != 32 {
            return None;
        }
        Some((vec[0], Self { src_type, cmp_op, srcs: srcs.clone() }))
    }

    fn merge_select(&self, op: &OpCSel, model: &dyn Model) -> Option<Op> {
        let mut merged = OpCSel {
            dst: op.dst.clone(),
            cmp_type: self.src_type,
            cmp_op: self.cmp_op,
            cmp_srcs: self.srcs.clone(),
            sel_srcs: op.sel_srcs.clone(),
        };
        if op.cmp_op == CmpOp::Eq {
            merged.sel_srcs.swap(0, 1);
        }
        let merged: Op = merged.into();
        supported_op(merged, model)
    }

    fn merge_accum(&self, op: &OpShiftLop, accum: &Src,
                   model: &dyn Model) -> Option<Op> {
        let accum_op = match op.logic_op {
            LogicOp::And => CmpAccumOp::And,
            LogicOp::Or => CmpAccumOp::Or,
            _ => return None,
        };
        let merged = if self.src_type.is_float_type() {
            OpFCmp {
                dst: op.dst.clone(), src_type: self.src_type,
                res_type: CmpResultType::M1, cmp_op: self.cmp_op,
                srcs: self.srcs.clone(), accum: accum.clone(), accum_op,
            }.into()
        } else {
            OpICmp {
                dst: op.dst.clone(), src_type: self.src_type,
                res_type: CmpResultType::M1, cmp_op: self.cmp_op,
                srcs: self.srcs.clone(), accum: accum.clone(), accum_op,
            }.into()
        };
        supported_op(merged, model)
    }
}

fn supported_op(op: Op, model: &dyn Model) -> Option<Op> {
    if !model.op_is_supported(&op)
        || op.dsts().iter().any(|dst| !model.op_dst_supports_lanes(&op, dst.lanes))
        || op.srcs().iter().any(|src| {
            !model.op_src_supports_swizzle(&op, src, src.swizzle)
                || !model.op_src_supports_mod(&op, src, src.src_mod)
        })
    {
        None
    } else {
        Some(op)
    }
}

fn plain_scalar_ssa(src: &Src) -> Option<SSAValue> {
    if !src.swizzle.is_none() || !src.src_mod.is_none() {
        return None;
    }
    let vec = src.src_ref.as_ssa()?;
    (vec.comps() == 1 && vec[0].bits() == 32).then(|| vec[0])
}

#[derive(Default)]
struct BooleanMasks {
    values: FxHashSet<SSAValue>,
}

impl BooleanMasks {
    fn contains(&self, src: &Src) -> bool {
        if let Some(value) = src.resolve_imm(DataType::I32) {
            return value == 0 || value == u64::from(u32::MAX);
        }
        if src.swizzle.is_none()
            && matches!(src.src_mod, SrcMod::None | SrcMod::BNot)
        {
            if let Some(vec) = src.src_ref.as_ssa() {
                return vec.comps() == 1 && self.values.contains(&vec[0]);
            }
        }
        false
    }

    fn insert_instr(&mut self, instr: &Instr) -> bool {
        let dsts = instr.dsts();
        if dsts.len() != 1 || dsts[0].lanes != DstLanes::All {
            return false;
        }
        let Some(vec) = dsts[0].dst_ref.as_ssa() else { return false; };
        if vec.comps() != 1 || vec[0].bits() != 32 {
            return false;
        }
        let known = match &instr.op {
            Op::FCmp(op) => op.src_type == DataType::F32
                && op.res_type == CmpResultType::M1,
            Op::ICmp(op) => matches!(op.src_type, DataType::S32 | DataType::U32)
                && op.res_type == CmpResultType::M1,
            Op::Copy(op) => op.dst_type.total_bits() == 32 && self.contains(&op.src),
            Op::CSel(op) => matches!(op.cmp_type, DataType::F32 | DataType::S32 | DataType::U32)
                && op.sel_srcs.iter().all(|src| self.contains(src)),
            Op::ShiftLop(op) => op.dst_type == DataType::U32
                && op.shift_op == ShiftOp::None
                && matches!(op.logic_op, LogicOp::And | LogicOp::Or | LogicOp::Xor)
                && self.contains(&op.src0) && self.contains(&op.src2),
            _ => false,
        };
        known && self.values.insert(vec[0])
    }

    fn for_shader(s: &Shader<'_>) -> Self {
        let mut masks = Self::default();
        loop {
            let mut progress = false;
            for instr in s.blocks.iter().flat_map(|block| &block.instrs) {
                progress |= masks.insert_instr(instr);
            }
            if !progress {
                return masks;
            }
        }
    }
}

fn accum_inputs(op: &OpShiftLop) -> Option<[(&Src, &Src); 2]> {
    if op.dst_type != DataType::U32 || op.dst.lanes != DstLanes::All
        || op.shift_op != ShiftOp::None || op.not_result
        || !matches!(op.logic_op, LogicOp::And | LogicOp::Or)
    {
        return None;
    }
    Some([(&op.src0, &op.src2), (&op.src2, &op.src0)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use crate::ops::OpFCmp;
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};
    use crate::ops::{OpCopy, OpRegOut};
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn logic(dst: Dst, src0: Src, src2: Src, logic_op: LogicOp) -> OpShiftLop {
        OpShiftLop {
            dst, dst_type: DataType::U32, shift_op: ShiftOp::None,
            logic_op, not_result: false, src0, shift: 0_u32.into(), src2,
        }
    }

    #[test]
    fn accumulated_compare_preserves_mask_semantics() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut alloc = SSAValueAllocator::default();
        for ty in [DataType::F32, DataType::S32, DataType::U32] {
            for cmp_op in [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le,
                           CmpOp::Gt, CmpOp::Ge] {
                for a in [0_u32, 0x80000000, 0x3f800000, 0xbf800000,
                          0x7fc00000, 0x7f800000, 0xffffffff] {
                    for b in [0_u32, 0x80000000, 0x3f800000, 0x7fc00000] {
                        for accum in [0_u32, u32::MAX] {
                            for lop in [LogicOp::And, LogicOp::Or] {
                                let cmp = Comparison { src_type: ty, cmp_op,
                                    srcs: [a.into(), b.into()] };
                                let op = logic(alloc.alloc_ref(32).into(),
                                               alloc.alloc_ref(32).into(), accum.into(), lop);
                                assert!(accum_inputs(&op).is_some());
                                let merged = cmp.merge_accum(&op, &op.src2, model.as_ref()).unwrap();
                                let (result, accum_op) = match merged {
                                    Op::FCmp(x) => (x.cmp_op.fold_data(x.src_type, a.into(), b.into()), x.accum_op),
                                    Op::ICmp(x) => (x.cmp_op.fold_data(x.src_type, a.into(), b.into()), x.accum_op),
                                    _ => panic!("Expected comparison"),
                                };
                                let mask = CmpResultType::M1.fold(cmp_op.fold_data(ty, a.into(), b.into()), 32);
                                let expected = if lop == LogicOp::And { mask & accum } else { mask | accum };
                                let actual = CmpResultType::M1.fold(accum_op.fold(result, accum != 0), 32);
                                assert_eq!(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn boolean_masks_reject_arbitrary_values() {
        let mut alloc = SSAValueAllocator::default();
        let mut masks = BooleanMasks::default();
        for value in [0_u32, u32::MAX] {
            assert!(masks.contains(&value.into()));
        }
        for value in [1_u32, 2, 0x80000000, 0x7fffffff, 0xfffffffe] {
            assert!(!masks.contains(&value.into()));
        }
        let value = alloc.alloc_ref(32);
        assert!(!masks.contains(&value.clone().into()));
        let copy: Instr = OpCopy { dst: value.clone().into(), dst_type: DataType::I32,
                                  src: u32::MAX.into() }.into();
        assert!(masks.insert_instr(&copy));
        let mut src: Src = value.into();
        assert!(masks.contains(&src));
        src.src_mod = SrcMod::BNot;
        assert!(masks.contains(&src));
        assert!(plain_scalar_ssa(&src).is_none());
        let mut op = logic(alloc.alloc_ref(32).into(), src, 0_u32.into(), LogicOp::And);
        op.not_result = true;
        assert!(accum_inputs(&op).is_none());
        op.not_result = false;
        op.shift_op = ShiftOp::LShift;
        assert!(accum_inputs(&op).is_none());
    }

    #[test]
    fn compare_logic_pass_respects_uses_and_mask_proof() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        for extra_use in [false, true] {
            for unknown_accum in [false, true] {
                for reverse in [false, true] {
                    let mut alloc = SSAValueAllocator::default();
                    let cmp_dst = alloc.alloc_ref(32);
                    let accum_dst = alloc.alloc_ref(32);
                    let result = alloc.alloc_ref(32);
                    let cmp: Instr = OpFCmp {
                        dst: cmp_dst.clone().into(), src_type: DataType::F32,
                        res_type: CmpResultType::M1, cmp_op: CmpOp::Lt,
                        srcs: [0_f32.into(), 1_f32.into()], accum: 0_u32.into(),
                        accum_op: CmpAccumOp::None,
                    }.into();
                    let accum: Instr = OpCopy {
                        dst: accum_dst.clone().into(), dst_type: DataType::I32,
                        src: if unknown_accum { 1_u32.into() } else { u32::MAX.into() },
                    }.into();
                    let (lhs, rhs) = if reverse { (accum_dst, cmp_dst.clone()) }
                        else { (cmp_dst.clone(), accum_dst) };
                    let mut instrs = vec![cmp, accum,
                        logic(result.clone().into(), lhs.into(), rhs.into(), LogicOp::And).into(),
                        OpRegOut { reg: RegRef::new(0, RegRange::Regs(1)),
                                   src_type: DataType::I32, src: result.into() }.into()];
                    if extra_use {
                        instrs.push(OpRegOut { reg: RegRef::new(1, RegRange::Regs(1)),
                                              src_type: DataType::I32, src: cmp_dst.into() }.into());
                    }
                    let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
                    cfg.add_node(0, BasicBlock { label: LabelAllocator::default().alloc(), instrs });
                    let mut s = Shader { model: model.as_ref(), ssa_alloc: alloc,
                        phi_alloc: Default::default(), blocks: cfg.as_cfg(false),
                        info: ShaderInfo::default(), constant_pool: None };
                    s.opt_exec_units();
                    let fused = s.blocks[0].instrs.iter().filter(|instr|
                        matches!(&instr.op, Op::FCmp(op) if op.accum_op == CmpAccumOp::And)).count();
                    assert_eq!(fused, usize::from(!extra_use && !unknown_accum));
                }
            }
        }
    }

    #[test]
    fn compare_select_truth_values() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut alloc = SSAValueAllocator::default();
        let predicate = alloc.alloc_ref(32);
        for ty in [DataType::F32, DataType::S32, DataType::U32] {
            for cmp_op in [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le,
                           CmpOp::Gt, CmpOp::Ge] {
                for test in [CmpOp::Eq, CmpOp::Ne] {
                    for a in [0_u32, 0x80000000, 0x3f800000, 0xbf800000,
                              0x7fc00000, 0xffc00000, 0x7f800000, 0xffffffff] {
                        for b in [0_u32, 0x80000000, 0x3f800000, 0x7fc00000] {
                            let cmp = Comparison {
                                src_type: ty, cmp_op,
                                srcs: [a.into(), b.into()],
                            };
                            let sel = OpCSel {
                                dst: alloc.alloc_ref(32).into(),
                                cmp_type: DataType::S32,
                                cmp_op: test,
                                cmp_srcs: [predicate.clone().into(), 0_u32.into()],
                                sel_srcs: [123_u32.into(), 456_u32.into()],
                            };
                            assert!(select_predicate(&sel).is_some());
                            let merged = cmp.merge_select(&sel, model.as_ref()).unwrap();
                            let Op::CSel(merged) = merged else { unreachable!(); };
                            let original_cmp = cmp_op.fold_data(ty, a.into(), b.into());
                            for result_type in [CmpResultType::M1] {
                                let result = result_type.fold(original_cmp, 32);
                                let take_true = test.fold_data(DataType::S32,
                                                              result.into(), 0);
                                let expected = if take_true { 123 } else { 456 };
                                let take_true = merged.cmp_op.fold_data(ty, a.into(), b.into());
                                let actual = merged.sel_srcs[usize::from(!take_true)]
                                    .resolve_imm(DataType::I32).unwrap();
                                assert_eq!(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn predicate_modifiers_are_rejected() {
        let mut alloc = SSAValueAllocator::default();
        let mut sel = OpCSel {
            dst: DstRef::None.into(),
            cmp_type: DataType::S32,
            cmp_op: CmpOp::Ne,
            cmp_srcs: [alloc.alloc_ref(32).into(), 0_u32.into()],
            sel_srcs: [1_u32.into(), 2_u32.into()],
        };
        assert!(select_predicate(&sel).is_some());
        sel.cmp_srcs[0].src_mod = SrcMod::BNot;
        assert!(select_predicate(&sel).is_none());
        sel.cmp_srcs[0].src_mod = SrcMod::None;
        sel.cmp_op = CmpOp::Gt;
        assert!(select_predicate(&sel).is_none());
        sel.cmp_op = CmpOp::Ne;
        sel.cmp_srcs[1] = 1_u32.into();
        assert!(select_predicate(&sel).is_none());
    }

    #[test]
    fn accumulated_and_nonboolean_compare_are_rejected() {
        let mut alloc = SSAValueAllocator::default();
        let mut cmp = OpFCmp {
            dst: alloc.alloc_ref(32).into(),
            src_type: DataType::F32,
            res_type: CmpResultType::M1,
            cmp_op: CmpOp::Lt,
            srcs: [0_f32.into(), 1_f32.into()],
            accum: 0_u32.into(),
            accum_op: CmpAccumOp::None,
        };
        assert!(Comparison::from_instr(&cmp.clone().into()).is_some());
        cmp.accum_op = CmpAccumOp::Or;
        assert!(Comparison::from_instr(&cmp.clone().into()).is_none());
        cmp.accum_op = CmpAccumOp::None;
        for res_type in [CmpResultType::C, CmpResultType::I1, CmpResultType::F1] {
            cmp.res_type = res_type;
            assert!(Comparison::from_instr(&cmp.clone().into()).is_none());
        }
        cmp.res_type = CmpResultType::M1;
        cmp.cmp_op = CmpOp::Total;
        assert!(Comparison::from_instr(&cmp.into()).is_none());
    }
}

fn select_predicate(op: &OpCSel) -> Option<SSAValue> {
    if !matches!(op.cmp_type, DataType::S32 | DataType::U32)
        || !matches!(op.cmp_op, CmpOp::Eq | CmpOp::Ne)
        || !op.cmp_srcs[1].is_zero()
    {
        return None;
    }
    let src = &op.cmp_srcs[0];
    if !src.swizzle.is_none() || !src.src_mod.is_none() {
        return None;
    }
    let vec = src.src_ref.as_ssa()?;
    if vec.comps() != 1 || vec[0].bits() != 32 {
        return None;
    }
    Some(vec[0])
}

/// MUX executes in the SFU unit, while CSEL executes in the CVT unit.  CVT is
/// always wider than the SFU unit by at least 4x, this is always an optimization.
fn try_replace_mux_csel(op: &OpMux) -> Option<OpCSel> {
    if !matches!(op.dst_type, DataType::I32 | DataType::V2I16) {
        return None;
    };

    if op.src0.swizzle != Swizzle::NONE
        || op.src1.swizzle != Swizzle::NONE
        || op.sel.swizzle != Swizzle::NONE
    {
        return None;
    }

    let (cmp_op, num) = match op.mux_op {
        MuxOp::Neg => (CmpOp::Ge, NumericType::SignedInteger),
        MuxOp::IntZero => (CmpOp::Ne, NumericType::SignedInteger),
        MuxOp::FpZero => (CmpOp::Ne, NumericType::Float),
        MuxOp::Bit => return None,
    };

    Some(OpCSel {
        dst: op.dst.clone(),
        cmp_type: DataType::get(op.dst_type.comps(), num, op.dst_type.bits()),
        cmp_op,
        cmp_srcs: [op.sel.clone(), 0u32.into()],
        sel_srcs: [op.src1.clone(), op.src0.clone()],
    })
}

fn translate_instr(i: &mut Instr) {
    if let Op::Mux(op) = &mut i.op {
        if let Some(csel) = try_replace_mux_csel(op) {
            i.op = csel.into();
        }
    }
}

impl Shader<'_> {
    /// Tries to replace operations with cheaper, equivalent ones.  The only
    /// example right now is MUX being replaced with CSEL (SFU -> CVT), we might
    /// want this list to be expanded in the future.  Must run before
    /// legalization as we are inserting a 0 source that might conflict with
    /// other FAUs (in v9-v10).
    pub fn opt_exec_units(&mut self) {
        let mut uses: FxHashMap<SSAValue, usize> = FxHashMap::default();
        for block in self.blocks.iter() {
            for instr in &block.instrs {
                for ssa in instr.iter_ssa_uses() {
                    *uses.entry(*ssa).or_default() += 1;
                }
            }
        }
        let model = self.model;
        let masks = BooleanMasks::for_shader(self);
        let mut progress = false;
        for block in self.blocks.iter_mut() {
            let mut comparisons = FxHashMap::default();
            for instr in &mut block.instrs {
                translate_instr(instr);
                if let Op::CSel(op) = &instr.op {
                    if let Some(ssa) = select_predicate(op) {
                        if let Some(cmp) = comparisons.get(&ssa) {
                            let cmp: &Comparison = cmp;
                            if let Some(merged) = cmp.merge_select(op, model) {
                                instr.op = merged;
                                progress = true;
                            }
                        }
                    }
                }
                if instr.flow == FlowCtrl::NONE {
                    if let Op::ShiftLop(op) = &instr.op {
                        if let Some(inputs) = accum_inputs(op) {
                            let merged = inputs.into_iter().find_map(|(src, accum)| {
                                let ssa = plain_scalar_ssa(src)?;
                                if !masks.contains(accum) {
                                    return None;
                                }
                                let cmp = comparisons.get(&ssa)?;
                                cmp.merge_accum(op, accum, model)
                            });
                            if let Some(merged) = merged {
                                instr.op = merged;
                                progress = true;
                            }
                        }
                    }
                }
                if let Some((ssa, cmp)) = Comparison::from_instr(instr) {
                    if uses.get(&ssa) == Some(&1) {
                        comparisons.insert(ssa, cmp);
                    }
                }
            }
        }
        if progress {
            self.opt_dce();
        }
    }
}
