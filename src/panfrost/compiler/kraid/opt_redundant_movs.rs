// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::ra::instr_clobbered_regs;
use rustc_hash::FxHashMap;

fn const_mov(instr: &Instr) -> Option<(u8, u32)> {
    let Op::Mov(op) = &instr.op else {
        return None;
    };
    if instr.flow != FlowCtrl::NONE
        || op.dst_type != DataType::I32
        || op.dst.lanes != DstLanes::All
        || !op.src.swizzle.is_none()
        || !op.src.src_mod.is_none()
    {
        return None;
    }
    let DstRef::Reg(reg) = &op.dst.dst_ref else {
        return None;
    };
    if reg.range != RegRange::Regs(1) {
        return None;
    }
    let value = match &op.src.src_ref {
        SrcRef::Zero => 0,
        SrcRef::Imm32(v) => v.get(),
        _ => return None,
    };
    Some((reg.idx, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn shader(model: &dyn Model, instrs: Vec<Instr>) -> Shader<'_> {
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
        cfg.add_node(0, BasicBlock { label: LabelAllocator::default().alloc(), instrs });
        Shader { model, ssa_alloc: Default::default(), phi_alloc: Default::default(),
                 blocks: cfg.as_cfg(false), info: ShaderInfo::default(), constant_pool: None }
    }

    fn mov(dst: u8, src: u8) -> Instr {
        crate::ops::OpMov {
            dst: RegRef::new(dst, RegRange::Regs(1)).into(),
            dst_type: DataType::I32,
            src: RegRef::new(src, RegRange::Regs(1)).into(),
        }.into()
    }

    fn execute(instrs: &[Instr], regs: &mut [u32; 128]) {
        for instr in instrs {
            match &instr.op {
                Op::Mov(op) => {
                    let reg = op.dst.dst_ref.as_reg().unwrap();
                    let value = match &op.src.src_ref {
                        SrcRef::Reg(r) => regs[usize::from(r.idx)],
                        SrcRef::Imm32(v) => v.get(),
                        SrcRef::Zero => 0,
                        _ => panic!(),
                    };
                    let dst = &mut regs[usize::from(reg.idx)];
                    if op.dst_type == DataType::I16 {
                        *dst = (*dst & 0xffff0000) | (value & 0xffff);
                    } else {
                        *dst = value;
                    }
                }
                Op::RtCall(_) => {
                    for value in &mut regs[..20] {
                        *value = value.wrapping_mul(1234567).wrapping_add(98765);
                    }
                }
                _ => panic!(),
            }
        }
    }

    #[test]
    fn register_roundtrip_and_partial_write() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut s = shader(model.as_ref(), vec![mov(28, 18), mov(35, 18), mov(18, 28)]);
        assert!(s.opt_redundant_movs());
        assert_eq!(s.blocks[0].instrs.len(), 2);
        let mut partial = mov(18, 7);
        let Op::Mov(op) = &mut partial.op else { panic!() };
        op.dst_type = DataType::I16;
        op.dst = RegRef::new(18, RegRange::Half0).into();
        let mut s = shader(model.as_ref(), vec![mov(28, 18), partial, mov(18, 28)]);
        assert!(!s.opt_redundant_movs());
    }

    #[test]
    fn register_copy_differential_with_clobbers() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut random = 0x615a12d1_u32;
        for _ in 0..128 {
            let mut instrs = Vec::new();
            for _ in 0..256 {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let dst = ((random >> 8) & 63) as u8;
                let src = ((random >> 20) & 63) as u8;
                let mut instr = mov(dst, src);
                if random & 7 == 0 {
                    let Op::Mov(op) = &mut instr.op else { panic!() };
                    op.src = random.into();
                } else if random & 7 == 1 {
                    let Op::Mov(op) = &mut instr.op else { panic!() };
                    op.dst = RegRef::new(dst, RegRange::Half0).into();
                    op.dst_type = DataType::I16;
                }
                instrs.push(instr);
                instrs.push(mov(src, dst));
                if random & 31 == 2 {
                    instrs.push(crate::ops::OpRtCall { tls_args: false, dst: DstRef::None.into(),
                        arg_mask: 0,
                        target: RegRef::new(11, RegRange::Regs(1)).into(),
                        args: std::array::from_fn(|_| 0_u32.into()),
                        stack: RegRef::new(8, RegRange::Regs(2)).into(),
                    }.into());
                }
            }
            let mut expected = std::array::from_fn(|i| (i as u32).wrapping_mul(random));
            let mut actual = expected;
            execute(&instrs, &mut expected);
            let mut s = shader(model.as_ref(), instrs);
            s.opt_redundant_movs();
            execute(&s.blocks[0].instrs, &mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn register_copy_keeps_flow_control() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut last = mov(18, 28);
        last.flow.set_reconverge();
        let mut s = shader(model.as_ref(), vec![mov(28, 18), last]);
        assert!(!s.opt_redundant_movs());
        assert!(s.blocks[0].instrs[1].flow.get_reconverge());
    }
}

fn forget_reg(known: &mut FxHashMap<u8, u32>, reg: &RegRef) {
    let bytes = reg.byte_range();
    for w in (bytes.start / 4)..bytes.end.div_ceil(4) {
        known.remove(&u8::try_from(w).unwrap());
    }
}

fn reg_mov(instr: &Instr) -> Option<(u8, u8)> {
    let Op::Mov(op) = &instr.op else { return None };
    if instr.flow != FlowCtrl::NONE
        || op.dst_type != DataType::I32
        || op.dst.lanes != DstLanes::All
        || !op.src.swizzle.is_none()
        || !op.src.src_mod.is_none()
    {
        return None;
    }
    let DstRef::Reg(dst) = &op.dst.dst_ref else { return None };
    let SrcRef::Reg(src) = &op.src.src_ref else { return None };
    if dst.range != RegRange::Regs(1) || src.range != RegRange::Regs(1) {
        return None;
    }
    Some((dst.idx, src.idx))
}

impl Shader<'_> {
    pub fn opt_redundant_movs(&mut self) -> bool {
        let model = self.model;
        let mut progress = false;
        let mut exit: Vec<Option<FxHashMap<u8, u32>>> =
            vec![None; self.blocks.len()];
        for bi in 0..self.blocks.len() {
            let mut known: Option<FxHashMap<u8, u32>> = None;
            for &pi in self.blocks.pred_indices(bi) {
                let Some(Some(pred)) = (pi < bi).then(|| exit[pi].as_ref())
                else {
                    known = Some(Default::default());
                    break;
                };
                known = Some(match known {
                    None => pred.clone(),
                    Some(mut cur) => {
                        cur.retain(|k, v| pred.get(k) == Some(v));
                        cur
                    }
                });
            }
            let mut known = known.unwrap_or_default();
            let mut values: [usize; 128] = std::array::from_fn(|i| i);
            let mut next = values.len();
            self.blocks[bi].instrs.retain(|instr| {
                if instr.flow != FlowCtrl::NONE
                    || matches!(instr.op, Op::Branch(_) | Op::Jump(_)
                        | Op::JumpEx(_) | Op::Discard(_) | Op::WMask(_))
                {
                    known.clear();
                    for value in &mut values {
                        *value = next;
                        next += 1;
                    }
                }
                if let Some((dst, src)) = reg_mov(instr) {
                    if values[usize::from(dst)] == values[usize::from(src)]
                        || known.get(&dst).is_some_and(|v| known.get(&src) == Some(v))
                    {
                        progress = true;
                        return false;
                    }
                    values[usize::from(dst)] = values[usize::from(src)];
                    if let Some(value) = known.get(&src).copied() {
                        known.insert(dst, value);
                    } else {
                        known.remove(&dst);
                    }
                    return true;
                }
                if let Some((idx, value)) = const_mov(instr) {
                    if known.get(&idx) == Some(&value) {
                        progress = true;
                        return false;
                    }
                    known.insert(idx, value);
                    values[usize::from(idx)] = next;
                    next += 1;
                    return true;
                }
                for dst in instr.dsts() {
                    if let DstRef::Reg(reg) = &dst.dst_ref {
                        forget_reg(&mut known, reg);
                        let bytes = reg.byte_range();
                        for word in (bytes.start / 4)..bytes.end.div_ceil(4) {
                            values[usize::from(word)] = next;
                            next += 1;
                        }
                    }
                }
                for reg in instr_clobbered_regs(model, &instr.op) {
                    forget_reg(&mut known, &reg);
                    let bytes = reg.byte_range();
                    for word in (bytes.start / 4)..bytes.end.div_ceil(4) {
                        values[usize::from(word)] = next;
                        next += 1;
                    }
                }
                true
            });
            exit[bi] = Some(known);
        }
        progress
    }
}
