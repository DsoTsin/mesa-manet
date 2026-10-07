// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use crate::flow::FlowWaitBit;
use crate::ir::*;
use crate::liveness::{PhysicalLiveness, RegAccess};
use crate::model::RegByteSet;
use std::ops::Range;

use compiler::dataflow::ForwardDataflow;

const SLOT_WAITS: [FlowWaitBit; 3] =
    [FlowWaitBit::Slot0, FlowWaitBit::Slot1, FlowWaitBit::Slot2];

#[derive(Clone, Copy, PartialEq)]
struct PendingStagingReads([RegByteSet; 3]);

impl PendingStagingReads {
    fn new() -> Self {
        Self([RegByteSet::new(); 3])
    }

    fn union_with(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= *b;
        }
    }

    fn any(&self) -> RegByteSet {
        self.0[0] | self.0[1] | self.0[2]
    }

    fn step(&mut self, model: &dyn Model, instr: &Instr) {
        if let Some(slot) = instr.flow.get_msg_slot_idx() {
            if model.op_is_message(&instr.op) {
                for src in instr.srcs() {
                    if !model.op_src_is_staging_reg(&instr.op, src) {
                        continue;
                    }
                    if let Some(reg) = src.src_ref.as_reg() {
                        self.0[usize::from(slot)].insert_range(reg.byte_range());
                    }
                }
            }
        }
        for (pending, bit) in self.0.iter_mut().zip(SLOT_WAITS) {
            if instr.flow.get_wait_bit(bit) {
                *pending = RegByteSet::new();
            }
        }
    }
}

fn discard_footprints(model: &dyn Model, instr: &Instr) -> Vec<Option<Range<u16>>> {
    instr
        .srcs()
        .iter()
        .map(|src| {
            let reg = src.src_ref.as_reg()?;
            if !matches!(reg.range, RegRange::Regs(_)) {
                return None;
            }
            if model.op_src_is_64bit(&instr.op, src) {
                let start = u16::from(reg.idx & !1) * 4;
                let bytes = reg.byte_range();
                Some(start.min(bytes.start)..(start + 8).max(bytes.end))
            } else {
                Some(reg.byte_range())
            }
        })
        .collect()
}

fn mark_dead_sources(model: &dyn Model, b: &mut BasicBlock, mut live: RegByteSet) {
    for instr in b.instrs.iter_mut().rev() {
        let access = RegAccess::for_instr(model, instr);
        let retained = live - access.writes;
        let footprints = discard_footprints(model, instr);
        for (src, footprint) in instr.srcs_mut().iter_mut().zip(footprints) {
            src.last_use = footprint
                .is_some_and(|bytes| !retained.contains_any_in_range(bytes));
        }
        live = access.live_before(live);
    }
}

fn keep_async_staging_reads(
    model: &dyn Model,
    b: &mut BasicBlock,
    mut pending: PendingStagingReads,
) {
    for instr in b.instrs.iter_mut() {
        let busy = pending.any();
        let footprints = discard_footprints(model, instr);
        let keep: Vec<bool> = instr
            .srcs()
            .iter()
            .zip(footprints)
            .map(|(src, footprint)| {
                src.last_use
                    && (model.op_src_is_staging_reg(&instr.op, src)
                        || footprint.is_some_and(|bytes| {
                            busy.contains_any_in_range(bytes)
                        }))
            })
            .collect();
        for (src, keep) in instr.srcs_mut().iter_mut().zip(keep) {
            if keep {
                src.last_use = false;
            }
        }
        pending.step(model, instr);
    }
}

fn mark_block_last_use(
    model: &dyn Model,
    b: &mut BasicBlock,
    live_out: RegByteSet,
    pending_in: PendingStagingReads,
) {
    mark_dead_sources(model, b, live_out);
    keep_async_staging_reads(model, b, pending_in);
}

impl Shader<'_> {
    pub fn mark_last_use(&mut self) {
        let model = self.model;
        let physical = PhysicalLiveness::for_shader(self);

        let mut pending_in = vec![PendingStagingReads::new(); self.blocks.len()];
        let mut pending_out = pending_in.clone();
        ForwardDataflow {
            cfg: &self.blocks,
            block_in: &mut pending_in,
            block_out: &mut pending_out,
            transfer: |_, block, output, input| {
                let mut pending = *input;
                for instr in &block.instrs {
                    pending.step(model, instr);
                }
                let changed = pending != *output;
                *output = pending;
                changed
            },
            join: |input, output| input.union_with(output),
        }
        .solve();

        for (bi, b) in self.blocks.iter_mut().enumerate() {
            mark_block_last_use(model, b, physical.live_out[bi], pending_in[bi]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use crate::ops::OpMov;

    fn reg(idx: u8) -> RegRef {
        RegRef::new(idx, RegRange::Regs(1))
    }

    fn mov(dst: u8, src: u8) -> Instr {
        OpMov { dst: reg(dst).into(), dst_type: DataType::I32, src: reg(src).into() }.into()
    }

    #[test]
    fn last_use_follows_partial_writes_and_live_out() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut instrs = vec![mov(2, 0), mov(3, 0), mov(0, 1), mov(4, 0)];
        instrs[0].srcs_mut()[0].last_use = true;
        instrs[2].dsts_mut()[0].lanes = DstLanes::H0;
        let mut b = BasicBlock { label: LabelAllocator::default().alloc(), instrs };
        let mut live = RegByteSet::new();
        live.insert_range(0..4);
        mark_block_last_use(model.as_ref(), &mut b, live, PendingStagingReads::new());
        assert!(!b.instrs[0].srcs()[0].last_use);
        assert!(!b.instrs[1].srcs()[0].last_use);
        assert!(b.instrs[2].srcs()[0].last_use);
        assert!(!b.instrs[3].srcs()[0].last_use);
        mark_block_last_use(model.as_ref(), &mut b, RegByteSet::new(), PendingStagingReads::new());
        assert!(!b.instrs[1].srcs()[0].last_use);
        assert!(b.instrs[3].srcs()[0].last_use);
    }

    #[test]
    fn last_use_is_kept_while_async_staging_read_is_pending() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let instrs = vec![mov(5, 4), mov(6, 7)];
        let mut b = BasicBlock { label: LabelAllocator::default().alloc(), instrs };
        let mut pending = PendingStagingReads::new();
        pending.0[1].insert_range(reg(4).byte_range());
        mark_block_last_use(model.as_ref(), &mut b, RegByteSet::new(), pending);
        assert!(!b.instrs[0].srcs()[0].last_use);
        assert!(b.instrs[1].srcs()[0].last_use);
    }
}
