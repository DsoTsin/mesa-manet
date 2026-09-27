use crate::ir::*;
use crate::ops::{FClamp, FRound, OpCopy};
use rustc_hash::FxHashMap;

#[derive(Eq, Hash, PartialEq)]
enum ScalarValue {
    SSA(SSAValue),
    Imm(u32),
    FAU(FAUPage, u16),
}

#[derive(Eq, Hash, PartialEq)]
struct ScalarSource {
    value: ScalarValue,
    src_mod: SrcMod,
}

impl ScalarSource {
    fn new(src: &Src) -> Option<Self> {
        if !src.swizzle.is_none() {
            return None;
        }
        let value = match &src.src_ref {
            SrcRef::SSA(vec) if vec.comps() == 1 && vec[0].bits() == 32
                && !vec[0].is_mem() => ScalarValue::SSA(vec[0]),
            SrcRef::Zero => ScalarValue::Imm(0),
            SrcRef::Imm32(imm) => ScalarValue::Imm(imm.get()),
            SrcRef::FAU(fau) if !fau.load64 && !fau.page.is_special() =>
                ScalarValue::FAU(fau.page, fau.idx),
            _ => return None,
        };
        Some(Self { value, src_mod: src.src_mod })
    }
}

#[derive(Eq, Hash, PartialEq)]
enum ExpressionKind {
    FAdd(FRound, FClamp),
    Fma(FRound, FClamp),
    FmaRScale(FRound, FClamp),
    FSinTable(bool),
    FCosTable(bool),
}

#[derive(Eq, Hash, PartialEq)]
struct Expression {
    kind: ExpressionKind,
    srcs: Vec<ScalarSource>,
}

impl Expression {
    fn new(instr: &Instr) -> Option<(SSAValue, Self)> {
        if instr.flow != FlowCtrl::NONE {
            return None;
        }
        let kind = match &instr.op {
            Op::FAdd(op) if op.dst_type == DataType::F32 =>
                ExpressionKind::FAdd(op.round, op.clamp),
            Op::Fma(op) if op.dst_type == DataType::F32 =>
                ExpressionKind::Fma(op.round, op.clamp),
            Op::FmaRScale(op) => ExpressionKind::FmaRScale(op.round, op.clamp),
            Op::FSinTable(op) => ExpressionKind::FSinTable(op.offset),
            Op::FCosTable(op) => ExpressionKind::FCosTable(op.offset),
            _ => return None,
        };
        let [dst] = instr.dsts() else { return None; };
        if dst.lanes != DstLanes::All {
            return None;
        }
        let DstRef::SSA(vec) = &dst.dst_ref else { return None; };
        if vec.comps() != 1 || vec[0].bits() != 32 || vec[0].is_mem() {
            return None;
        }
        let srcs = instr.srcs().iter().map(ScalarSource::new)
            .collect::<Option<Vec<_>>>()?;
        Some((vec[0], Self { kind, srcs }))
    }
}

fn cse_block(instrs: &mut [Instr]) -> bool {
    let mut expressions = FxHashMap::default();
    let mut copies = FxHashMap::default();
    let mut progress = false;
    for instr in instrs {
        for src in instr.iter_ssa_uses_mut() {
            if let Some(&copy) = copies.get(src) {
                *src = copy;
            }
        }
        let Some((dst, expression)) = Expression::new(instr) else {
            continue;
        };
        if let Some(&existing) = expressions.get(&expression) {
            copies.insert(dst, existing);
            instr.op = OpCopy {
                dst: dst.into(),
                dst_type: DataType::I32,
                src: Src::from(existing),
            }.into();
            progress = true;
        } else {
            expressions.insert(expression, dst);
        }
    }
    progress
}

impl Shader<'_> {
    pub fn opt_cse(&mut self) -> bool {
        let mut progress = false;
        for block in self.blocks.iter_mut() {
            progress |= cse_block(&mut block.instrs);
        }
        progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{SSAInstrBuilder, SSABuilder};
    use crate::model::model_for_gpu_id;
    use crate::ops::{OpFma, OpFmaRScale, OpFSinTable};
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};

    #[test]
    fn sin_cos_share_reduction_and_tables() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut alloc = SSAValueAllocator::default();
        let input = alloc.alloc_ssa(32);
        let sin = alloc.alloc_ssa(32);
        let cos = alloc.alloc_ssa(32);
        let mut b = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
        b.fsincos_32_to(sin.into(), input.into(), false);
        b.fsincos_32_to(cos.into(), input.into(), true);
        b.fadd_32(sin.into(), cos.into());
        let mut instrs = b.into_vec();
        assert!(cse_block(&mut instrs));
        assert_eq!(instrs.iter().filter(|i| matches!(i.op, Op::FSinTable(_))).count(), 1);
        assert_eq!(instrs.iter().filter(|i| matches!(i.op, Op::FCosTable(_))).count(), 1);
        assert_eq!(instrs.iter().filter(|i| matches!(i.op, Op::FmaRScale(_))).count(), 1);
        assert_eq!(instrs.iter().filter(|i| matches!(i.op, Op::Copy(_))).count(), 6);
        assert!(!cse_block(&mut instrs));
    }

    #[test]
    fn arithmetic_semantics_are_part_of_key() {
        let mut alloc = SSAValueAllocator::default();
        let input = alloc.alloc_ssa(32);
        let base = OpFma {
            dst: alloc.alloc_ssa(32).into(), dst_type: DataType::F32,
            round: FRound::NearestEven, clamp: FClamp::None,
            srcs: [input.into(), 1_f32.into(), 0_u32.into()],
        };
        let key = Expression::new(&base.clone().into()).unwrap().1;
        let mut variants = Vec::new();
        for round in [FRound::Up, FRound::Down, FRound::TowardsZero, FRound::NearestValue] {
            let mut op = base.clone();
            op.round = round;
            variants.push(op);
        }
        for clamp in [FClamp::ZeroToInf, FClamp::NegOneToOne, FClamp::ZeroToOne] {
            let mut op = base.clone();
            op.clamp = clamp;
            variants.push(op);
        }
        for bits in [0x80000000_u32, 0x7fc00000, 0x7fc00001] {
            let mut op = base.clone();
            op.srcs[2] = bits.into();
            variants.push(op);
        }
        let mut op = base.clone();
        op.srcs[0] = op.srcs[0].clone().fneg();
        variants.push(op);
        let mut op = base.clone();
        op.srcs.swap(0, 1);
        variants.push(op);
        let keys: Vec<_> = variants.into_iter()
            .map(|op| Expression::new(&op.into()).unwrap().1).collect();
        for (idx, other) in keys.iter().enumerate() {
            assert!(key != *other);
            assert!(keys[..idx].iter().all(|k| k != other));
        }
    }

    #[test]
    fn table_offset_and_block_scope_are_preserved() {
        let mut alloc = SSAValueAllocator::default();
        let input = alloc.alloc_ssa(32);
        let mut instrs: Vec<Instr> = [false, true, false].into_iter().map(|offset|
            OpFSinTable { dst: alloc.alloc_ssa(32).into(), src: input.into(), offset }.into()
        ).collect();
        assert!(cse_block(&mut instrs));
        assert!(matches!(instrs[0].op, Op::FSinTable(_)));
        assert!(matches!(instrs[1].op, Op::FSinTable(_)));
        assert!(matches!(instrs[2].op, Op::Copy(_)));
        let mut next: Vec<Instr> = vec![OpFSinTable {
            dst: alloc.alloc_ssa(32).into(), src: input.into(), offset: false,
        }.into()];
        assert!(!cse_block(&mut next));
    }

    #[test]
    fn special_sources_and_partial_destinations_are_rejected() {
        let mut alloc = SSAValueAllocator::default();
        let mut instr: Instr = OpFSinTable {
            dst: alloc.alloc_ssa(32).into(), src: FAURef::user_i32(0).into(), offset: false,
        }.into();
        assert!(Expression::new(&instr).is_some());
        instr.srcs_mut()[0].swizzle = Swizzle::H00;
        assert!(Expression::new(&instr).is_none());
        instr.srcs_mut()[0] = FAURef::user_i64(0).into();
        assert!(Expression::new(&instr).is_none());
        let mut special = FAURef::user_i32(0);
        special.page = FAUPage::Special0;
        instr.srcs_mut()[0] = special.into();
        assert!(Expression::new(&instr).is_none());
        instr.srcs_mut()[0] = 0_u32.into();
        instr.dsts_mut()[0].lanes = DstLanes::H0;
        assert!(Expression::new(&instr).is_none());
        instr.dsts_mut()[0].lanes = DstLanes::All;
        instr.flow.set_reconverge();
        assert!(Expression::new(&instr).is_none());
    }

    #[test]
    fn scale_operand_is_part_of_key() {
        let mut alloc = SSAValueAllocator::default();
        let input = alloc.alloc_ssa(32);
        let mut instrs: Vec<Instr> = [0_u32, u32::MAX, 0].into_iter().map(|scale|
            OpFmaRScale {
                dst: alloc.alloc_ssa(32).into(), round: FRound::NearestEven,
                clamp: FClamp::None,
                srcs: [input.into(), input.into(), (-0_f32).into()],
                scale: scale.into(),
            }.into()
        ).collect();
        assert!(cse_block(&mut instrs));
        assert!(matches!(instrs[0].op, Op::FmaRScale(_)));
        assert!(matches!(instrs[1].op, Op::FmaRScale(_)));
        assert!(matches!(instrs[2].op, Op::Copy(_)));
    }

    #[test]
    fn fau_identity_ignores_display_metadata() {
        let mut first = FAURef::user_i32(2);
        first.imm32 = Some(0);
        let mut other = first;
        other.imm32 = Some(1);
        let key = ScalarSource::new(&first.into()).unwrap();
        assert!(key == ScalarSource::new(&other.into()).unwrap());
        other.idx = 3;
        assert!(key != ScalarSource::new(&other.into()).unwrap());
        other.idx = 2;
        other.page = FAUPage::SmallConst;
        assert!(key != ScalarSource::new(&other.into()).unwrap());
    }
}
