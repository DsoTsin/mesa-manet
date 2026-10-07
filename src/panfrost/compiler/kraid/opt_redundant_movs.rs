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

fn forget_reg(known: &mut FxHashMap<u8, u32>, reg: &RegRef) {
    let bytes = reg.byte_range();
    for w in (bytes.start / 4)..bytes.end.div_ceil(4) {
        known.remove(&u8::try_from(w).unwrap());
    }
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
            self.blocks[bi].instrs.retain(|instr| {
                if let Some((idx, value)) = const_mov(instr) {
                    if known.get(&idx) == Some(&value) {
                        progress = true;
                        return false;
                    }
                    known.insert(idx, value);
                    return true;
                }
                for dst in instr.dsts() {
                    if let DstRef::Reg(reg) = &dst.dst_ref {
                        forget_reg(&mut known, reg);
                    }
                }
                for reg in instr_clobbered_regs(model, &instr.op) {
                    forget_reg(&mut known, &reg);
                }
                true
            });
            exit[bi] = Some(known);
        }
        progress
    }
}
