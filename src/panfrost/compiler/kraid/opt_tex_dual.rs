// SPDX-License-Identifier: MIT

use crate::debug::{DEBUG, DebugFlags};
use crate::ir::*;
use crate::liveness::{LiveSet, Liveness};
use crate::ops::*;
use crate::schedule::pressure_schedule;
use crate::ssa_value::SSAValueAllocator;
use compiler::bitset::BitSet;
use rustc_hash::FxHashSet;

const MAX_DUAL_REGS: u8 = 8;
const REG_GRANULE_BYTES: u32 = 16 * 4;
const RELAXED_HEADROOM_BYTES: u32 = 6 * 4;
const MAX_SCHEDULED_BLOCK: usize = 4096;
const SCHEDULE_COST: usize = 4;
const SEARCH_BUDGET: usize = 1 << 21;

#[derive(Clone, Copy)]
pub enum TexDualPolicy {
    Strict,
    Relaxed,
}

struct Single<'a> {
    op: &'a OpTexSingle,
    handle: u32,
    dst: &'a SSARef,
}

fn single(instr: &Instr) -> Option<Single<'_>> {
    let Op::TexSingle(op) = &instr.op else {
        return None;
    };
    if instr.flow != FlowCtrl::NONE
        || op.wide_indices
        || op.dim != TexDim::Tex2D
        || op.projection_enable
        || op.array_enable
        || op.texel_offset
        || op.compare_enable
        || !matches!(
            op.lod_mode,
            TexLodMode::None
                | TexLodMode::Computed
                | TexLodMode::ComputedForceDelta
        )
        || op.dst_type.bits() != 32
        || op.dst_type.comps() != op.write_mask.comps()
    {
        return None;
    }
    if !matches!(&op.data.src_ref, SrcRef::SSA(_)) {
        return None;
    }
    let handle = &op.handle;
    if !handle.swizzle.is_none() || !handle.src_mod.is_none() {
        return None;
    }
    let handle = match &handle.src_ref {
        SrcRef::Zero => 0,
        SrcRef::Imm64(v) if v.get() >> 32 == 0 => v.get() as u32,
        _ => return None,
    };
    let DstRef::SSA(dst) = &op.dst.dst_ref else {
        return None;
    };
    if op.dst.lanes != DstLanes::All
        || dst.iter().any(|ssa| ssa.bits() != 32 || ssa.is_mem())
        || dst.comps() != op.dst_type.total_bits().div_ceil(32)
    {
        return None;
    }
    Some(Single { op, handle, dst })
}

fn compatible(a: &OpTexSingle, b: &OpTexSingle) -> bool {
    a.data.src_ref == b.data.src_ref
        && a.dim == b.dim
        && a.projection_enable == b.projection_enable
        && a.array_enable == b.array_enable
        && a.texel_offset == b.texel_offset
        && a.compare_enable == b.compare_enable
        && a.lod_mode == b.lod_mode
}

fn can_be_secondary(s: &Single) -> bool {
    s.op.write_mask.is_prefix() && s.handle != 0
}

fn is_fence(instr: &Instr) -> bool {
    matches!(&instr.op, Op::Barrier(_) | Op::ScheduleBarrier(_))
}

fn build_dual(primary: &Single, secondary: &Single) -> Instr {
    let p = primary.op;
    let s = secondary.op;
    let dst: SSARef =
        primary.dst.iter().chain(secondary.dst.iter()).copied().collect();
    let handle = u64::from(primary.handle) | (u64::from(secondary.handle) << 32);
    OpTexDual {
        dst: dst.into(),
        primary_type: p.dst_type,
        primary_mask: p.write_mask,
        secondary_type: s.dst_type,
        secondary_mask: s.write_mask,
        skip: p.skip && s.skip,
        dim: p.dim,
        projection_enable: p.projection_enable,
        array_enable: p.array_enable,
        texel_offset: p.texel_offset,
        compare_enable: p.compare_enable,
        lod_mode: p.lod_mode,
        data: p.data.clone(),
        handle: handle.into(),
    }
    .into()
}

fn block_max_live(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    live_out: &BitSet<u32>,
    instrs: &[&Instr],
) -> u32 {
    let mut live = LiveSet::new();
    for idx in live_out.iter() {
        live.insert(ssa_alloc.lookup_by_idx(idx));
    }
    let mut max = live.bytes().reg;
    for instr in instrs.iter().rev() {
        max = max.max(live.insert_instr_bottom_up(model, instr).reg);
    }
    max
}

fn scheduled_max_live(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    live_out: &BitSet<u32>,
    block: &BasicBlock,
) -> u32 {
    let view: Vec<&Instr> = block.instrs.iter().collect();
    let current = block_max_live(model, ssa_alloc, live_out, &view);
    let (_, scheduled) = pressure_schedule(model, ssa_alloc, block, live_out);
    current.min(scheduled)
}

fn uses_any(instr: &Instr, vec: &SSARef) -> bool {
    instr.iter_ssa_uses().any(|ssa| vec.iter().any(|v| v == ssa))
}

struct Pairing {
    ia: usize,
    ib: usize,
    at_a: bool,
    dual: Instr,
}

struct PairSearch<'a> {
    model: &'a dyn Model,
    ssa_alloc: &'a SSAValueAllocator,
    live_out: &'a BitSet<u32>,
    limit: u32,
    budget: usize,
    rejected: FxHashSet<(SSAValue, SSAValue)>,
}

impl PairSearch<'_> {
    fn fits(
        &mut self,
        block: &BasicBlock,
        dual: &Instr,
        at: usize,
        removed: usize,
    ) -> bool {
        let len = block.instrs.len();
        if len > self.budget {
            return false;
        }
        self.budget -= len;
        let view: Vec<&Instr> = block
            .instrs
            .iter()
            .enumerate()
            .filter(|&(ip, _)| ip != removed)
            .map(|(ip, instr)| if ip == at { dual } else { instr })
            .collect();
        if block_max_live(self.model, self.ssa_alloc, self.live_out, &view)
            <= self.limit
        {
            return true;
        }
        if len > MAX_SCHEDULED_BLOCK || len * SCHEDULE_COST > self.budget {
            return false;
        }
        self.budget -= len * SCHEDULE_COST;
        let paired = BasicBlock {
            label: block.label,
            instrs: view.into_iter().cloned().collect(),
        };
        let (_, scheduled) =
            pressure_schedule(self.model, self.ssa_alloc, &paired, self.live_out);
        scheduled <= self.limit
    }

    fn find(&mut self, block: &BasicBlock) -> Option<Pairing> {
        let instrs = &block.instrs;
        for ib in 0..instrs.len() {
            let Some(b) = single(&instrs[ib]) else {
                continue;
            };
            let mut crosses_discard = false;
            for ia in (0..ib).rev() {
                if is_fence(&instrs[ia]) {
                    break;
                }
                let Some(a) = single(&instrs[ia]) else {
                    crosses_discard |= instrs[ia].writes_discard();
                    continue;
                };
                if !compatible(a.op, b.op)
                    || a.dst.comps() + b.dst.comps() > MAX_DUAL_REGS
                {
                    continue;
                }
                let key = (a.dst[0], b.dst[0]);
                if self.rejected.contains(&key) {
                    continue;
                }
                let dual = if can_be_secondary(&b) {
                    build_dual(&a, &b)
                } else if can_be_secondary(&a) {
                    build_dual(&b, &a)
                } else {
                    continue;
                };

                let at_a = if self.fits(block, &dual, ia, ib) {
                    true
                } else if !crosses_discard
                    && !instrs[ia + 1..ib].iter().any(|i| uses_any(i, a.dst))
                    && self.fits(block, &dual, ib, ia)
                {
                    false
                } else {
                    self.rejected.insert(key);
                    continue;
                };
                self.rejected.clear();
                return Some(Pairing { ia, ib, at_a, dual });
            }
        }
        None
    }
}

fn block_has_candidates(b: &BasicBlock) -> bool {
    let mut seen: Vec<&OpTexSingle> = Vec::new();
    for instr in b.instrs.iter() {
        if is_fence(instr) {
            seen.clear();
            continue;
        }
        let Some(s) = single(instr) else {
            continue;
        };
        if seen.iter().any(|a| compatible(a, s.op)) {
            return true;
        }
        seen.push(s.op);
    }
    false
}

impl Shader<'_> {
    pub fn has_tex_dual_candidates(&self) -> bool {
        self.model.arch() >= 15
            && !DEBUG.contains(DebugFlags::NO_TEX_DUAL)
            && self.blocks.iter().any(|b| block_has_candidates(b))
    }

    pub fn tex_dual_signature(&self) -> Vec<(usize, usize, SSARef)> {
        let mut sig = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            for (ip, instr) in b.instrs.iter().enumerate() {
                if let Op::TexDual(op) = &instr.op {
                    if let Some(dst) = op.dst.dst_ref.as_ssa() {
                        sig.push((bi, ip, dst.clone()));
                    }
                }
            }
        }
        sig
    }

    pub fn opt_tex_dual(&mut self, policy: TexDualPolicy) -> bool {
        if !self.has_tex_dual_candidates() {
            return false;
        }

        let live = Liveness::for_shader(self);
        let live_out: Vec<BitSet<u32>> = (0..self.blocks.len())
            .map(|bi| live.block(bi).live_out_set().clone())
            .collect();
        let limit = (0..self.blocks.len())
            .map(|bi| {
                scheduled_max_live(
                    self.model,
                    &self.ssa_alloc,
                    &live_out[bi],
                    &self.blocks[bi],
                )
            })
            .max()
            .unwrap_or(0);
        let limit = match policy {
            TexDualPolicy::Strict => limit,
            TexDualPolicy::Relaxed => (limit.next_multiple_of(REG_GRANULE_BYTES)
                - RELAXED_HEADROOM_BYTES)
                .max(limit),
        };

        let mut progress = false;
        let mut budget = SEARCH_BUDGET;
        for bi in 0..self.blocks.len() {
            if !block_has_candidates(&self.blocks[bi]) {
                continue;
            }
            let mut search = PairSearch {
                model: self.model,
                ssa_alloc: &self.ssa_alloc,
                live_out: &live_out[bi],
                limit,
                budget,
                rejected: FxHashSet::default(),
            };
            while let Some(p) = search.find(&self.blocks[bi]) {
                let instrs = &mut self.blocks[bi].instrs;
                let (at, removed) = if p.at_a { (p.ia, p.ib) } else { (p.ib, p.ia) };
                instrs[at] = p.dual;
                instrs.remove(removed);
                progress = true;
            }
            budget = search.budget;
        }
        progress
    }
}
