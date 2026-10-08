// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::liveness::*;
use crate::ops::{MemoryEffect, VaryingUpdateMode};
use crate::ssa_value::SSAValueAllocator;
use compiler::bitset::BitSet;
use rustc_hash::FxHashMap;
use std::ops::Range;

fn is_mem(op: &Op) -> bool {
    match op.memory_effect() {
        // We ignore ConstRead since the value of constant memory should never
        // be affected by other memory ops
        MemoryEffect::None | MemoryEffect::ConstRead => false,
        MemoryEffect::Read | MemoryEffect::Write | MemoryEffect::ReadWrite => {
            true
        }
    }
}

fn is_mem_write(op: &Op) -> bool {
    match op.memory_effect() {
        MemoryEffect::None | MemoryEffect::ConstRead | MemoryEffect::Read => {
            false
        }
        MemoryEffect::Write | MemoryEffect::ReadWrite => true,
    }
}

fn is_barrier(op: &Op) -> bool {
    matches!(op, Op::Barrier(_) | Op::ScheduleBarrier(_))
}

fn respects_barrier(op: &Op) -> bool {
    // ALU can freely slide past barriers
    is_barrier(op) || is_mem(op)
}

fn writes_var_hidden(op: &Op) -> bool {
    match op.var_update_mode() {
        VaryingUpdateMode::Store | VaryingUpdateMode::Clobber => true,
        VaryingUpdateMode::Retrieve | VaryingUpdateMode::None => false,
    }
}

fn uses_var_hidden(op: &Op) -> bool {
    op.var_update_mode() != VaryingUpdateMode::None
}

struct DepTracker {
    /// For each instruction, the bitset of instructions which depend on it
    deps: Vec<BitSet<usize>>,
    /// For each instruction, the number of instructions it depends on
    count: Vec<u32>,
    /// Bitset of instructions for which count == 0
    ready: BitSet<usize>,
}

impl DepTracker {
    fn add_ip(&mut self, ip: usize) {
        debug_assert!(self.count[ip] == 0);
        self.ready.insert(ip);
    }

    fn add_dep(&mut self, ip: usize, dep_ip: usize) {
        if self.deps[ip].insert(dep_ip) {
            self.count[dep_ip] += 1;
            self.ready.remove(dep_ip);
        }
    }

    fn ready(&self) -> impl Iterator<Item = usize> + use<'_> {
        self.ready.iter()
    }

    fn remove_ip(&mut self, ip: usize) {
        debug_assert!(self.count[ip] == 0);
        debug_assert!(self.ready.contains(ip));

        self.ready.remove(ip);
        for dep_ip in self.deps[ip].iter() {
            self.count[dep_ip] -= 1;
            if self.count[dep_ip] == 0 {
                self.ready.insert(dep_ip);
            }
        }
        self.deps[ip].clear();
    }

    fn for_block(block: &BasicBlock, body_range: Range<usize>) -> DepTracker {
        let instr_count = block.instrs.len();
        let mut deps = DepTracker {
            deps: (0..instr_count).map(|_| BitSet::new()).collect(),
            count: (0..instr_count).map(|_| 0_u32).collect(),
            ready: BitSet::new(),
        };

        // Map from an SSAValue to its definition in this block, if any.
        let mut def_ip: FxHashMap<SSAValue, usize> = Default::default();

        // IP of the last memory instruction
        let mut bar_ip = usize::MAX;

        // IP of the last barrier instruction
        let mut mem_ip = usize::MAX;

        // IP of the last instruction to access the varying hidden register
        let mut var_ip = usize::MAX;

        // IP of the last discard instruction
        let mut discard_ip = usize::MAX;

        for ip in body_range.clone() {
            let instr = &block.instrs[ip];
            deps.add_ip(ip);

            for ssa in instr.iter_ssa_uses() {
                if let Some(&def_ip) = def_ip.get(ssa) {
                    deps.add_dep(ip, def_ip);
                }
            }

            if bar_ip != usize::MAX && respects_barrier(&instr.op) {
                deps.add_dep(ip, bar_ip);
            }
            if is_barrier(&instr.op) {
                bar_ip = ip;
            }

            // Capture WaW and RaW hazards
            if mem_ip != usize::MAX && is_mem(&instr.op) {
                deps.add_dep(ip, mem_ip);
            }
            if is_mem_write(&instr.op) {
                mem_ip = ip;
            }

            // Capture WaW and RaW hazards
            if var_ip != usize::MAX && uses_var_hidden(&instr.op) {
                deps.add_dep(ip, var_ip);
            }
            if writes_var_hidden(&instr.op) {
                var_ip = ip;
            }

            if discard_ip != usize::MAX && instr.reads_discard() {
                deps.add_dep(ip, discard_ip);
            }
            if instr.writes_discard() {
                discard_ip = ip;
            }

            for ssa in instr.iter_ssa_defs() {
                def_ip.insert(*ssa, ip);
            }
        }

        bar_ip = usize::MAX;
        mem_ip = usize::MAX;
        var_ip = usize::MAX;
        discard_ip = usize::MAX;
        for ip in body_range.clone().rev() {
            let instr = &block.instrs[ip];

            if bar_ip != usize::MAX && respects_barrier(&instr.op) {
                deps.add_dep(bar_ip, ip);
            }
            if is_barrier(&instr.op) {
                bar_ip = ip;
            }

            // Capture WaR hazards
            if mem_ip != usize::MAX && is_mem(&instr.op) {
                deps.add_dep(mem_ip, ip);
            }
            if is_mem_write(&instr.op) {
                mem_ip = ip;
            }

            // Capture WaR hazards
            if var_ip != usize::MAX && uses_var_hidden(&instr.op) {
                deps.add_dep(var_ip, ip);
            }
            if writes_var_hidden(&instr.op) {
                var_ip = ip;
            }

            if discard_ip != usize::MAX && instr.reads_discard() {
                deps.add_dep(discard_ip, ip);
            }
            if instr.writes_discard() {
                discard_ip = ip;
            }
        }

        deps
    }
}

pub(crate) fn pressure_schedule(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    b: &BasicBlock,
    live_out: &BitSet<u32>,
) -> (Vec<usize>, u32) {
    let breadth = pressure_schedule_order(model, ssa_alloc, b, live_out, false);
    let depth = pressure_schedule_order(model, ssa_alloc, b, live_out, true);
    if depth.1 < breadth.1 { depth } else { breadth }
}

fn pressure_schedule_order(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    b: &BasicBlock,
    live_out: &BitSet<u32>,
    depth_first: bool,
) -> (Vec<usize>, u32) {
    let body_range = b.body_ip_range();

    let mut deps = DepTracker::for_block(b, body_range.clone());

    let mut live = LiveSet::new();
    for idx in live_out.iter() {
        live.insert(ssa_alloc.lookup_by_idx(idx));
    }

    // We're not going to schedule the postlude but we need to account for it
    // in the live set or our estimates will be all out of whack.
    let mut max_live = LiveBytes::default();
    for ip in body_range.end..b.instrs.len() {
        let bytes = live.insert_instr_bottom_up(model, &b.instrs[ip]);
        max_live = max_live.max(bytes);
    }

    let mut end_ip = body_range.end;
    let mut schedule: Vec<_> = (0..b.instrs.len())
        .map(|ip| {
            if body_range.contains(&ip) {
                usize::MAX
            } else {
                ip
            }
        })
        .collect();

    let mut live_since: FxHashMap<SSAValue, usize> = Default::default();
    loop {
        let mut best_ip = usize::MAX;
        let mut best_pressure = i32::MAX;
        let mut best_since = 0;
        for ip in deps.ready() {
            let mut rel_pressure = 0_i32;
            let mut since = 0;
            for ssa in b.instrs[ip].iter_ssa_defs() {
                if live.contains(ssa) {
                    rel_pressure -= i32::from(ssa.bytes());
                }
                since = since.max(live_since.get(ssa).copied().unwrap_or(0));
            }
            for ssa in b.instrs[ip].iter_ssa_uses() {
                if !live.contains(ssa) {
                    rel_pressure += i32::from(ssa.bytes());
                }
            }
            let better = if depth_first {
                rel_pressure < best_pressure
                    || (rel_pressure == best_pressure && since >= best_since)
            } else {
                rel_pressure <= best_pressure
            };
            if better {
                best_ip = ip;
                best_pressure = rel_pressure;
                best_since = since;
            }
        }

        if best_ip == usize::MAX {
            break;
        }

        // Assert that no instruction gets placed twice
        assert!(schedule[best_ip] == usize::MAX);
        end_ip -= 1;
        schedule[best_ip] = end_ip;

        if depth_first {
            for ssa in b.instrs[best_ip].iter_ssa_uses() {
                if !live.contains(ssa) {
                    live_since.insert(*ssa, body_range.end - end_ip);
                }
            }
        }

        let bytes = live.insert_instr_bottom_up(model, &b.instrs[best_ip]);
        max_live = max_live.max(bytes);

        deps.remove_ip(best_ip);
    }

    // Assert we placed all of them
    assert!(end_ip == body_range.start);

    (schedule, max_live.reg)
}

fn pressure_schedule_block(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    b: &mut BasicBlock,
    bl: &BlockLiveness,
) {
    let (schedule, max_live) =
        pressure_schedule(model, ssa_alloc, b, bl.live_out_set());

    // Replace with the new scheduling if it's better.
    if max_live < bl.max_live_bytes().reg {
        // SAFETY:
        //
        // We already asserted that we placed each instruction exactly once.
        unsafe {
            b.reorder_instrs(|ip| Some(schedule[ip]), b.instrs.len());
        }
    }
}

fn message_schedule_region(model: &dyn Model, instr: &Instr) -> bool {
    instr.flow == FlowCtrl::NONE
        && instr.op.can_eliminate()
        && instr.op.var_update_mode() == VaryingUpdateMode::None
        && !instr.reads_discard() && !instr.writes_discard()
        && instr.iter_reg_uses().next().is_none()
        && instr.iter_reg_defs().next().is_none()
        && (matches!(instr.op, Op::Load(_) | Op::LdPka(_))
            || (!model.op_is_message(&instr.op)
                && instr.op.memory_effect() == MemoryEffect::None))
}

fn message_schedule_order(instrs: &[Instr]) -> Vec<usize> {
    let mut successors = vec![Vec::new(); instrs.len()];
    let mut pending = vec![0; instrs.len()];
    let mut defs: FxHashMap<SSAValue, usize> = FxHashMap::default();
    for (ip, instr) in instrs.iter().enumerate() {
        let mut predecessors = BitSet::new();
        for ssa in instr.iter_ssa_uses() {
            if let Some(&pred) = defs.get(ssa) {
                if predecessors.insert(pred) {
                    successors[pred].push(ip);
                    pending[ip] += 1;
                }
            }
        }
        for ssa in instr.iter_ssa_defs() {
            assert!(defs.insert(*ssa, ip).is_none());
        }
    }
    let mut distance = vec![usize::MAX; instrs.len()];
    for ip in (0..instrs.len()).rev() {
        distance[ip] = if matches!(instrs[ip].op, Op::Load(_) | Op::LdPka(_)) {
            0
        } else {
            successors[ip].iter().map(|&next| distance[next].saturating_add(1))
                .min().unwrap_or(usize::MAX)
        };
    }
    let mut ready: Vec<_> = pending.iter().enumerate()
        .filter_map(|(ip, &count)| (count == 0).then_some(ip)).collect();
    let mut order = Vec::with_capacity(instrs.len());
    while !ready.is_empty() {
        let best = (0..ready.len()).min_by_key(|&idx| (distance[ready[idx]], ready[idx])).unwrap();
        let ip = ready.swap_remove(best);
        order.push(ip);
        for &next in &successors[ip] {
            pending[next] -= 1;
            if pending[next] == 0 {
                ready.push(next);
            }
        }
    }
    assert_eq!(order.len(), instrs.len());
    order
}

impl Shader<'_> {
    pub fn schedule_for_message_loads(&mut self) {
        if self.model.arch() != 15 {
            return;
        }
        for block in self.blocks.iter_mut() {
            let body = block.body_ip_range();
            let mut start = body.start;
            while start < body.end {
                let end = start + block.instrs[start..body.end].iter()
                    .take_while(|instr| message_schedule_region(self.model, instr)).count();
                if end > start + 1 && block.instrs[start..end].iter()
                    .any(|instr| matches!(instr.op, Op::Load(_) | Op::LdPka(_)))
                {
                    let order = message_schedule_order(&block.instrs[start..end]);
                    let mut instrs: Vec<_> = block.instrs.drain(start..end).map(Some).collect();
                    block.instrs.splice(start..start,
                        order.into_iter().map(|ip| instrs[ip].take().unwrap()));
                }
                start = end + 1;
            }
        }
    }

    pub fn schedule_for_pressure(&mut self) {
        let live = Liveness::for_shader(self);
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            pressure_schedule_block(
                self.model,
                &self.ssa_alloc,
                block,
                live.block(bi),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa_value::AllocSSA;
    use crate::flow::FlowWaitBit;
    use crate::model::model_for_gpu_id;
    use crate::ops::{MemAccess, OpBarrier, OpIAdd, OpLdPka, OpLoad, OpMov};

    #[test]
    fn message_schedule_issues_address_producers_before_consumers() {
        let mut alloc = SSAValueAllocator::default();
        let a = alloc.alloc_ref(32);
        let used_a = alloc.alloc_ref(32);
        let addr = alloc.alloc_ref(64);
        let b = alloc.alloc_ref(32);
        let used_b = alloc.alloc_ref(32);
        let instrs = vec![
            Instr::from(OpLdPka { dst: a.clone().into(), dst_type: DataType::I32,
                                 access: MemAccess::None, handle: 0_u32.into(), offset: 0_u32.into() }),
            OpMov { dst: used_a.into(), dst_type: DataType::I32, src: a.into() }.into(),
            OpIAdd { dst: addr.clone().into(), dst_type: DataType::U64,
                     saturate: false, srcs: [0_u32.into(), 0_u32.into()] }.into(),
            OpLoad { dst: b.clone().into(), dst_type: DataType::I32, is_tls: false,
                     access: MemAccess::None, addr: addr.into(), offset: 0 }.into(),
            OpMov { dst: used_b.into(), dst_type: DataType::I32, src: b.into() }.into(),
        ];
        assert_eq!(message_schedule_order(&instrs), vec![0, 2, 3, 1, 4]);
    }

    #[test]
    fn message_schedule_preserves_load_dependencies() {
        let mut alloc = SSAValueAllocator::default();
        let handle = alloc.alloc_ref(32);
        let offset = alloc.alloc_ref(32);
        let value = alloc.alloc_ref(32);
        let used_value = alloc.alloc_ref(32);
        let instrs = vec![
            Instr::from(OpLdPka { dst: handle.clone().into(), dst_type: DataType::I32,
                                 access: MemAccess::None, handle: 0_u32.into(), offset: 0_u32.into() }),
            OpMov { dst: offset.clone().into(), dst_type: DataType::I32, src: handle.clone().into() }.into(),
            OpLdPka { dst: value.clone().into(), dst_type: DataType::I32,
                      access: MemAccess::None, handle: handle.into(), offset: offset.into() }.into(),
            OpMov { dst: used_value.into(), dst_type: DataType::I32, src: value.into() }.into(),
        ];
        assert_eq!(message_schedule_order(&instrs), vec![0, 1, 2, 3]);
    }

    #[test]
    fn message_schedule_regions_exclude_flow_and_physical_registers() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let reg = RegRef::new(0, RegRange::Regs(1));
        let physical = Instr::from(OpMov { dst: reg.into(), dst_type: DataType::I32, src: reg.into() });
        assert!(!message_schedule_region(model.as_ref(), &physical));
        assert!(!message_schedule_region(model.as_ref(), &OpBarrier {}.into()));
        let mut alloc = SSAValueAllocator::default();
        let mut flow = Instr::from(OpMov { dst: alloc.alloc_ref(32).into(),
                                          dst_type: DataType::I32, src: 0_u32.into() });
        assert!(message_schedule_region(model.as_ref(), &flow));
        flow.flow.set_wait_bit(FlowWaitBit::Slot0);
        assert!(!message_schedule_region(model.as_ref(), &flow));
    }
}
