// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use crate::debug::{DEBUG, DebugFlags};
use crate::flow::FlowWaitBit;
use crate::ir::*;
use crate::ops::{MemoryEffect, VaryingUpdateMode};
use std::cmp::Reverse;

#[derive(Default)]
struct Slot {
    count: usize,
    wait_ip: Option<usize>,
}

fn slot_wait_bit(slot: usize) -> FlowWaitBit {
    match slot {
        0 => FlowWaitBit::Slot0,
        1 => FlowWaitBit::Slot1,
        2 => FlowWaitBit::Slot2,
        _ => unreachable!(),
    }
}

fn fixed_wait_slot(i: &Instr) -> Option<usize> {
    match i.op {
        Op::ATest(_) => Some(0),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct MemFootprint {
    base: Option<RegRef>,
    start: i32,
    end: i32,
}

impl MemFootprint {
    fn of(op: &Op) -> MemFootprint {
        let (addr, offset, bits) = match op {
            Op::Load(op) => (&op.addr, op.offset, op.dst_type.total_bits()),
            Op::Store(op) => (&op.addr, op.offset, op.src_type.total_bits()),
            _ => {
                return MemFootprint {
                    base: None,
                    start: 0,
                    end: 0,
                };
            }
        };
        let base = match &addr.src_ref {
            SrcRef::Reg(reg)
                if addr.src_mod.is_none() && addr.swizzle == Swizzle::NONE =>
            {
                Some(*reg)
            }
            _ => None,
        };
        let start = i32::from(offset);
        MemFootprint {
            base,
            start,
            end: start + (i32::from(bits) + 7) / 8,
        }
    }

    fn may_alias(&self, other: &MemFootprint) -> bool {
        match (self.base, other.base) {
            (Some(a), Some(b)) if a == b => {
                self.start < other.end && other.start < self.end
            }
            _ => true,
        }
    }

    fn invalidate_written(&mut self, written: &RegRef) {
        if let Some(base) = self.base {
            let a = base.reg_range();
            let b = written.reg_range();
            if a.start < b.end && b.start < a.end {
                self.base = None;
            }
        }
    }
}

fn next_aliasing(
    accesses: &[(usize, MemFootprint)],
    footprint: &MemFootprint,
) -> Option<usize> {
    accesses
        .iter()
        .rev()
        .find(|(_, other)| footprint.may_alias(other))
        .map(|(ip, _)| *ip)
}

fn calc_message_deadlines_in_bb(
    model: &dyn Model,
    block: &BasicBlock,
) -> Vec<Option<usize>> {
    let reg_count = model.max_reg_count();
    let mut deadlines = vec![None; block.instrs.len()];
    let mut next_access = vec![None; reg_count as usize];
    let mut loads: Vec<(usize, MemFootprint)> = Vec::new();
    let mut stores: Vec<(usize, MemFootprint)> = Vec::new();
    let mut next_barrier = None;
    let mut next_ld_var = None;

    for (ip, instr) in block.instrs.iter().enumerate().rev() {
        let effect = instr.op.memory_effect();
        let var_usage = instr.op.var_update_mode();

        for reg in instr.op.iter_reg_defs() {
            for (_, access) in loads.iter_mut().chain(stores.iter_mut()) {
                access.invalidate_written(reg);
            }
        }

        if model.op_is_message(&instr.op) {
            let next_reg_access = instr
                .op
                .iter_reg_defs()
                .flat_map(RegRef::reg_range)
                .filter_map(|reg| next_access[usize::from(reg)])
                .min();

            let footprint = MemFootprint::of(&instr.op);
            let next_mem_hazard = match effect {
                MemoryEffect::None | MemoryEffect::ConstRead => None,
                MemoryEffect::Read => next_aliasing(&stores, &footprint),
                MemoryEffect::Write | MemoryEffect::ReadWrite => [
                    next_aliasing(&loads, &footprint),
                    next_aliasing(&stores, &footprint),
                ]
                .into_iter()
                .flatten()
                .min(),
            };

            // All LD_VAR has a hidden register, we only care about WaR/WaW.
            // RaW is handled for us in hw
            let next_hidden_reg_hazard = match var_usage {
                VaryingUpdateMode::Store | VaryingUpdateMode::Clobber => {
                    next_ld_var
                }
                _ => None,
            };

            deadlines[ip] = [
                next_reg_access,
                next_mem_hazard,
                next_barrier,
                next_hidden_reg_hazard,
            ]
            .into_iter()
            .flatten()
            .min();
        }

        // BARRIER waits for all message slots. Slot7 is waited on across warps
        // but general message slots are per-warp. Drain them before BARRIER so
        // a later load cannot overtake an earlier store from another warp.
        // Virtual barriers have no HW instruction to carry the wait, so they
        // wait on the preceding instruction as well.
        if matches!(instr.op, Op::Barrier(_) | Op::ScheduleBarrier(_)) {
            next_barrier = Some(ip);
        }

        let footprint = MemFootprint::of(&instr.op);
        match effect {
            MemoryEffect::None | MemoryEffect::ConstRead => (),
            MemoryEffect::Read => loads.push((ip, footprint)),
            MemoryEffect::Write => stores.push((ip, footprint)),
            MemoryEffect::ReadWrite => {
                loads.push((ip, footprint));
                stores.push((ip, footprint));
            }
        }

        if var_usage != VaryingUpdateMode::None {
            next_ld_var = Some(ip);
        }

        for reg in instr.op.iter_reg_defs().chain(instr.op.iter_reg_uses()) {
            for reg_index in reg.reg_range() {
                next_access[reg_index as usize] = Some(ip);
            }
        }
    }

    deadlines
}

impl Shader<'_> {
    pub fn assign_message_slots(&mut self) {
        if DEBUG.contains(DebugFlags::SERIAL) {
            for b in self.blocks.iter_mut() {
                for i in b.instrs.iter_mut() {
                    if matches!(&i.op, Op::Barrier(_)) {
                        i.flow.set_wait_bit(FlowWaitBit::Barrier);
                    } else if self.model.op_is_message(&i.op) {
                        i.flow.set_msg_slot_idx(0);
                        i.flow.set_wait_bit(FlowWaitBit::Slot0);
                    }
                }
                // Remove virtual schedule barriers.
                b.instrs.retain(|instr| {
                    !matches!(&instr.op, Op::ScheduleBarrier(_))
                });
            }
            return;
        }

        for block in self.blocks.iter_mut() {
            // Track whole registers as HW can have race conditions when a
            // pending message read/writes part of a register that is accessed
            // by another instruction.
            let deadlines = calc_message_deadlines_in_bb(self.model, block);
            let mut slots: [Slot; 3] = Default::default();

            for ip in 0..block.instrs.len() {
                // No need to insert wait at the end of the shader.
                if block.instrs[ip].flow.get_end_shader() {
                    break;
                }

                // Insert wait on the previous instruction if deadline is
                // current instruction.
                for slot_idx in 0..3 {
                    let needs_wait = slots[slot_idx].wait_ip == Some(ip);
                    if !needs_wait {
                        continue;
                    }

                    debug_assert!(!matches!(
                        &block.instrs[ip - 1].op,
                        Op::Barrier(_) | Op::ScheduleBarrier(_)
                    ));
                    debug_assert!(slots[slot_idx].count > 0 && ip > 0);

                    slots[slot_idx] = Slot::default();
                    block.instrs[ip - 1]
                        .flow
                        .set_wait_bit(slot_wait_bit(slot_idx));
                }

                let is_barrier = matches!(&block.instrs[ip].op, Op::Barrier(_));
                let is_message = self.model.op_is_message(&block.instrs[ip].op);

                if is_barrier {
                    #[cfg(debug_assertions)]
                    for slot in &slots {
                        debug_assert_eq!(slot.count, 0);
                    }

                    block.instrs[ip].flow.set_wait_bit(FlowWaitBit::Barrier);
                    continue;
                } else if !is_message {
                    continue;
                }

                // Prefer a slot with the same wait point, otherwise choose the least-used slot.
                let deadline = deadlines[ip];
                let slot_idx = fixed_wait_slot(&block.instrs[ip])
                    .unwrap_or_else(|| {
                        slots
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, slot)| {
                                (Reverse(slot.wait_ip == deadline), slot.count)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap()
                    });

                block.instrs[ip].flow.set_msg_slot_idx(slot_idx as u8);

                let slot = &mut slots[slot_idx];
                slot.count += 1;
                if let Some(deadline) = deadline {
                    slot.wait_ip = Some(
                        slot.wait_ip.map_or(deadline, |old| old.min(deadline)),
                    );
                }
            }

            // Currently the analysis is only for one basic block at a time. If
            // we are not at the end of the shader we need to insert waits at
            // the last instruction of the block.
            if let Some(last) = block.instrs.last_mut() {
                if !last.flow.get_end_shader() {
                    for (slot_idx, slot) in slots.iter().enumerate() {
                        if slot.count > 0 {
                            last.flow.set_wait_bit(slot_wait_bit(slot_idx));
                        }
                    }
                }
            }

            // Remove virtual schedule barriers.
            block
                .instrs
                .retain(|instr| !matches!(&instr.op, Op::ScheduleBarrier(_)));
        }
    }
}
