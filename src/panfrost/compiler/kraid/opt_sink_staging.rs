// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::liveness::Liveness;
use crate::ops::MemoryEffect;
use rustc_hash::{FxHashMap, FxHashSet};

fn movable(model: &dyn Model, instr: &Instr) -> bool {
    instr.flow == FlowCtrl::NONE
        && !model.op_is_message(&instr.op)
        && instr.op.can_eliminate()
        && instr.op.memory_effect() == MemoryEffect::None
        && !matches!(
            instr.op,
            Op::Clper(_)
                | Op::PhiDst(_)
                | Op::PhiSrc(_)
                | Op::RegIn(_)
                | Op::RegOut(_)
                | Op::Branch(_)
                | Op::ScheduleBarrier(_)
        )
}

fn has_staging_vec(model: &dyn Model, instr: &Instr) -> bool {
    model.op_is_message(&instr.op)
        && instr.srcs().iter().any(|src| {
            matches!(&src.src_ref, SrcRef::SSA(vec) if vec.comps() >= 2)
                && model.op_src_is_staging_reg(&instr.op, src)
        })
}

fn sink_block(
    model: &dyn Model,
    block: &mut BasicBlock,
    live_out: &dyn Fn(&SSAValue) -> bool,
) -> bool {
    let mut progress = false;
    let mut sunk: FxHashSet<SSAValue> = FxHashSet::default();
    loop {
        let body = block.body_ip_range();
        let mut def_ip: FxHashMap<SSAValue, usize> = FxHashMap::default();
        let mut first_use: FxHashMap<SSAValue, usize> = FxHashMap::default();
        let mut last_use: FxHashMap<SSAValue, usize> = FxHashMap::default();
        for (ip, instr) in block.instrs.iter().enumerate() {
            for ssa in instr.iter_ssa_uses() {
                first_use.entry(*ssa).or_insert(ip);
                last_use.insert(*ssa, ip);
            }
            for ssa in instr.iter_ssa_defs() {
                def_ip.insert(*ssa, ip);
            }
        }

        let mut found = None;
        'search: for t in body.clone() {
            let target = &block.instrs[t];
            let is_msg = model.op_is_message(&target.op);
            let is_sunk = target.iter_ssa_defs().any(|d| sunk.contains(d));
            if !is_msg && !is_sunk {
                continue;
            }
            for src in target.srcs() {
                let SrcRef::SSA(vec) = &src.src_ref else {
                    continue;
                };
                if !is_sunk
                    && (vec.comps() < 2
                        || !model.op_src_is_staging_reg(&target.op, src))
                {
                    continue;
                }
                for ssa in vec.iter() {
                    let Some(&d) = def_ip.get(ssa) else {
                        continue;
                    };
                    if !body.contains(&d)
                        || d + 1 >= t
                        || first_use.get(ssa) != Some(&t)
                    {
                        continue;
                    }
                    let def = &block.instrs[d];
                    if !movable(model, def)
                        || def.iter_ssa_defs().any(|x| {
                            sunk.contains(x)
                                || first_use.get(x).is_some_and(|&u| u < t)
                        })
                    {
                        continue;
                    }
                    if !block.instrs[d + 1..t]
                        .iter()
                        .any(|i| has_staging_vec(model, i))
                    {
                        continue;
                    }
                    let mut kills = FxHashSet::default();
                    for s in def.iter_ssa_uses() {
                        if last_use.get(s) == Some(&d) && !live_out(s) {
                            kills.insert(*s);
                        }
                    }
                    if kills.len() > 1 {
                        continue;
                    }
                    found = Some((d, t));
                    break 'search;
                }
            }
        }

        let Some((d, t)) = found else {
            break;
        };
        let instr = block.instrs.remove(d);
        for x in instr.iter_ssa_defs() {
            sunk.insert(*x);
        }
        block.instrs.insert(t - 1, instr);
        progress = true;
    }
    progress
}

impl Shader<'_> {
    pub fn opt_sink_staging(&mut self) -> bool {
        let live = Liveness::for_shader(self);
        let model = self.model;
        let mut progress = false;
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            let bl = live.block(bi);
            progress |= sink_block(model, block, &|s| bl.is_live_out(s));
        }
        progress
    }
}
