// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;

use crate::builder::*;
use crate::ir::*;
use crate::liveness::*;
use crate::model::{FAUModel, RegByteSet};
use crate::ra;
use crate::small_constants;
use compiler::bitset::{BitSet, ConstBitSet};
use compiler::enum_as_u8::EnumAsU8;
use compiler::smallvec::SmallVec;
use kraid_proc_macros::EnumAsU8;

fn move_src_to_tmp(b: &mut impl SSABuilder, src: &mut Src, bytes: u8) {
    debug_assert!(bytes > 0 && bytes <= 8);
    if let SrcRef::FAU(fau) = &src.src_ref {
        if fau.is_zext() {
            let tmp = b.alloc_ref(64);
            b.copy_i32_to(tmp[0].into(), FAURef::user_i32(fau.idx & !1).into());
            b.copy_i32_to(tmp[1].into(), SrcRef::Zero.into());
            src.src_ref = tmp.into();
            return;
        }
    }
    let tmp = b.alloc_ref((bytes * 8).into());
    let src_ref = std::mem::replace(&mut src.src_ref, tmp.clone().into());
    b.copy_to(tmp.into(), DataType::i(bytes * 8), src_ref.into());
}

/// Legalizes immediate sources by ensuring the following:
///
///  1. For 32-bit immediates, ensure that the instruction supports it.
///
///  2. For 64-bit immediates, only OpCopy supports them.
fn legalize_imm_src(b: &mut impl SSABuilder, op: &mut Op, src_idx: usize) {
    let src = &op.srcs()[src_idx];

    match &src.src_ref {
        SrcRef::Imm32(imm32) => {
            if !b.model().op_src_supports_imm32(op, src, (*imm32).into()) {
                let bytes = src.src_ref.bytes_read();
                move_src_to_tmp(b, &mut op.srcs_mut()[src_idx], bytes);
            }
        }
        SrcRef::Imm64(_) if !matches!(op, Op::Copy(_)) => {
            // No real instruction can read 64-bit immediates.  If any
            // instruction is left reading them, lower it into a copy.
            let bytes = src.src_ref.bytes_read();
            move_src_to_tmp(b, &mut op.srcs_mut()[src_idx], bytes);
        }
        _ => {}
    }
}

#[derive(Default)]
struct SSAValueSet {
    set: BitSet<SSAValue>,
    vec: Vec<SSAValue>,
}

impl SSAValueSet {
    fn new() -> SSAValueSet {
        Default::default()
    }

    fn clear(&mut self) {
        for ssa in self.vec.drain(..) {
            self.set.remove(ssa);
        }
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn insert(&mut self, ssa: SSAValue) -> bool {
        if self.set.insert(ssa) {
            self.vec.push(ssa);
            true
        } else {
            false
        }
    }
}

/// Legalizes fixed-reg sources by ensuring the following:
///
///  1. For the given instruction, ensure that no SSAValue is used multiple
///     times in fixed sources.  It's fine if an SSAValue is used in a fixed
///     source and also in a non-fixed source as the fixed source takes
///     priority.
///
///  2. For all fixed-reg sources, they must consume SSA values so we have
///     something to fix.  Fixed-reg sources cannot consume immediates or FAU.
///
///  3. For any fixed-reg source which is in the clobber set, this instruction
///     must kill the source.
fn legalize_fixed_srcs(
    b: &mut impl SSABuilder,
    bl: &BlockLiveness,
    instr: &mut Instr,
    ip: usize,
    ssa_used: &mut SSAValueSet,
) {
    debug_assert!(ssa_used.is_empty());

    let mut clobbered = RegByteSet::new();
    for reg in ra::instr_clobbered_regs(b.model(), &instr.op) {
        clobbered.insert_range(reg.byte_range());
    }
    for dst in instr.dsts() {
        if let Some(reg) = b.model().op_fixed_dst_reg(&instr.op, dst) {
            clobbered.insert_range(reg.byte_range());
        }
    }

    for src_idx in 0..instr.srcs().len() {
        let src = &instr.srcs()[src_idx];
        let src_type = instr.src_type(src);

        let Some(reg) = b.model().op_fixed_src_reg(&instr.op, src) else {
            continue;
        };

        let src = &mut instr.srcs_mut()[src_idx];
        if let SrcRef::SSA(vec) = &mut src.src_ref {
            for (ssa, bytes) in vec.iter_mut_zip_bytes(reg.byte_range()) {
                let duplicate = !ssa_used.insert(*ssa);
                if duplicate
                    || (clobbered.contains_any_in_range(bytes)
                        && bl.is_live_after_ip(ssa, ip))
                {
                    *ssa = b.copy_ssa(*ssa);
                }
            }
        } else {
            let bytes = if src_type == DataType::SR {
                assert!(src.src_ref != SrcRef::Zero);
                src.src_ref.bytes_read()
            } else {
                src_type.total_bytes()
            };
            move_src_to_tmp(b, src, bytes);
        }
    }

    ssa_used.clear();
}

/// Legalizes vector sources by ensuring the following
///
///  1. For any vector source (SSARef::comps() > 1), all the SSAValues
///     referenced by the SSARef are unique.
///
///  2. For any two sources, either they are identical or they have no
///     SSAValues in common.
fn legalize_vec_srcs(
    b: &mut impl SSABuilder,
    instr: &mut Instr,
    ssa_used: &mut SSAValueSet,
) {
    debug_assert!(ssa_used.is_empty());
    let srcs = instr.srcs_mut();

    let mut duplicates = [!0_usize; Instr::MAX_SRC_COUNT];
    debug_assert!(srcs.len() <= duplicates.len());
    for i in 0..srcs.len() {
        let (srcs_before_i, srcs_after_i) = srcs.split_at_mut(i);
        let SrcRef::SSA(vec) = &mut srcs_after_i[0].src_ref else {
            continue;
        };

        for (j, sb) in srcs_before_i.iter().enumerate() {
            if let SrcRef::SSA(sb_vec) = &sb.src_ref {
                if sb_vec == vec {
                    duplicates[i] = j;
                    break;
                }
            }
        }
        if duplicates[i] != !0_usize {
            continue;
        }

        for ssa in vec {
            if !ssa_used.insert(*ssa) {
                *ssa = b.copy_ssa(*ssa);
            }
        }
    }

    for i in 0..srcs.len() {
        if duplicates[i] != !0_usize {
            srcs[i].src_ref = srcs[duplicates[i]].src_ref.clone();
        }
    }

    ssa_used.clear();
}

#[repr(u8)]
#[derive(Clone, Copy, EnumAsU8, PartialEq)]
enum HWFAUPage {
    User0,
    User1,
    User2,
    User3,
    Special0,
    Special1,
    Special3,
    // This one has to go last
    SmallConst,
}

struct FAUSlot {
    page: FAUPage,
    /// 64-bit index (fau.index >> 1)
    idx64: u16,
    use_count: u8,
    words_used: u8,
    scalar_uses: [u8; 2],
    planned_page: Option<HWFAUPage>,
}

impl FAUSlot {
    fn hw_page(&self, fau_model: &FAUModel) -> HWFAUPage {
        if let Some(page) = self.planned_page {
            return page;
        }
        match self.page {
            FAUPage::User => match fau_model.user_page_idx(self.idx64 << 1) {
                0 => HWFAUPage::User0,
                1 => HWFAUPage::User1,
                2 => HWFAUPage::User2,
                3 => HWFAUPage::User3,
                _ => panic!("Invalid user FAU page"),
            },
            FAUPage::Special0 => HWFAUPage::Special0,
            FAUPage::Special1 => HWFAUPage::Special1,
            FAUPage::Special3 => HWFAUPage::Special3,
            FAUPage::SmallConst => HWFAUPage::SmallConst,
            FAUPage::Virtual => panic!("Unallocated virtual FAU"),
        }
    }
}

/// Legalizes FAU sources by ensuring the following
///
///  1. The combined amount of FAU (including small constants) is at most
///     64 bits, (not including k0 for arch >= v12).
///
///  2. All FAUs come from the same page
///
///  3. Message instructions are not allowed to read from LaneId or anything
///     in FAUPage::Special3.
///
///  4. Instructions executed within the execution engine limit the number of
///     FAU indices they can encode (1 in v13, 2 in v14)
struct LegalizeFAU<'a> {
    fau_model: &'a FAUModel,
    slots: SmallVec<FAUSlot>,
    inval_src: ConstBitSet<1, usize>,
    src64_w1: ConstBitSet<1, usize>,
    reads_k0: bool,
    max_words: u32,
}

pub fn fau_word_limit(model: &dyn Model, op: &Op) -> u32 {
    match op {
        Op::CSel(csel)
            if csel.cmp_srcs.iter().any(|src| {
                matches!(&src.src_ref, SrcRef::Imm32(imm)
                    if model.op_src_supports_imm32(op, src, imm.get()))
            }) =>
        {
            1
        }
        _ => 2,
    }
}

impl LegalizeFAU<'_> {
    fn new<'a>(fau_model: &'a FAUModel) -> LegalizeFAU<'a> {
        LegalizeFAU {
            fau_model,
            slots: Default::default(),
            inval_src: ConstBitSet::new(),
            src64_w1: ConstBitSet::new(),
            reads_k0: false,
            max_words: 2,
        }
    }

    fn extract_srcs(&mut self, model: &dyn Model, op: &Op, planning: bool) {
        for (src_idx, src) in op.srcs().iter().enumerate() {
            if matches!(src.src_ref, SrcRef::Zero) {
                self.reads_k0 = true;
            }
            let SrcRef::FAU(fau) = src.src_ref else {
                continue;
            };
            if planning && fau.page == FAUPage::Virtual {
                continue;
            }
            self.add_fau(model, op, src_idx, src, &fau, None);
        }
    }

    fn extract_planned_srcs(
        &mut self,
        model: &dyn Model,
        op: &Op,
    ) -> SmallVec<(usize, FAURef)> {
        const PLANNED_BASE: u16 = 0xff00;

        self.extract_srcs(model, op, true);
        let user_page = self
            .slots
            .iter()
            .find(|slot| slot.page == FAUPage::User)
            .map_or(HWFAUPage::User0, |slot| slot.hw_page(self.fau_model));

        let mut keys: SmallVec<(u64, u16)> = Default::default();
        let mut next = PLANNED_BASE;
        let mut planned: SmallVec<(usize, FAURef)> = Default::default();
        for (src_idx, src) in op.srcs().iter().enumerate() {
            let (key, wide) = match &src.src_ref {
                SrcRef::FAU(fau) if fau.page == FAUPage::Virtual => {
                    ((1_u64 << 32) | u64::from(fau.idx), fau.load64)
                }
                SrcRef::Imm32(imm) => {
                    if model.op_src_supports_imm32(op, src, imm.get())
                        || model.op_src_is_64bit(op, src)
                    {
                        continue;
                    }
                    if let Some(fau) =
                        small_constants::small_const_fau(model, op, src)
                    {
                        self.add_fau(model, op, src_idx, src, &fau, None);
                        planned.push((src_idx, fau));
                        continue;
                    }
                    (u64::from(imm.get()), false)
                }
                _ => continue,
            };
            let idx = match keys.iter().find(|(k, _)| *k == key) {
                Some((_, idx)) => *idx,
                None => {
                    if wide {
                        next = (next + 1) & !1;
                    }
                    let idx = next;
                    next += if wide { 2 } else { 1 };
                    keys.push((key, idx));
                    idx
                }
            };
            let fau = if wide {
                FAURef::user_i64(idx)
            } else {
                FAURef::user_i32(idx)
            };
            self.add_fau(model, op, src_idx, src, &fau, Some(user_page));
            planned.push((src_idx, fau));
        }
        planned
    }

    fn add_fau(
        &mut self,
        model: &dyn Model,
        op: &Op,
        src_idx: usize,
        src: &Src,
        fau: &FAURef,
        planned_page: Option<HWFAUPage>,
    ) {
        {
            let idx64 = fau.idx >> 1;
            let word = fau.idx & 1;
            let zext = fau.is_zext();
            let is64 = fau.page != FAUPage::SmallConst
                && !zext
                && model.op_src_is_64bit(&op, src);

            if is64 && word == 1 {
                self.src64_w1.insert(src_idx);

                // We can't handle .w1 in most 64-bit ops.  If we can't compose
                // with W1 (byte swizzles) or if the resulting swizzle isn't
                // supported, we have to mark this source invalid.
                if !Swizzle::replicate_word(1)
                    .swizzle(src.swizzle)
                    .is_some_and(|s| model.op_src_supports_swizzle(op, src, s))
                {
                    self.inval_src.insert(src_idx);
                    return;
                }
            }

            let slot = self
                .slots
                .iter_mut()
                .find(|x| x.page == fau.page && x.idx64 == idx64);
            let slot = match slot {
                Some(slot) => slot,
                None => self.slots.push_mut(FAUSlot {
                    page: fau.page,
                    idx64,
                    use_count: 0,
                    words_used: 0,
                    scalar_uses: [0; 2],
                    planned_page,
                }),
            };

            slot.use_count += 1;
            if !fau.load64 {
                slot.scalar_uses[usize::from(word)] += 1;
            }
            slot.words_used |= if zext {
                0b01
            } else if is64 || fau.load64 {
                0b11
            } else {
                1 << word
            };
        }
    }

    fn fau_retained(&self, fau: &FAURef) -> bool {
        let idx64 = fau.idx >> 1;
        let words = if fau.is_zext() {
            0b01
        } else if fau.load64 {
            0b11
        } else {
            1 << (fau.idx & 1)
        };
        self.slots
            .iter()
            .any(|s| {
                s.page == fau.page && s.idx64 == idx64
                    && (s.words_used & words) == words
            })
    }

    /// Once we filtered all slots, we can apply it back to the Op
    /// all FAU srcs that have been retained are ok, the rest require
    /// legalization.
    fn apply_legalization(&self, b: &mut impl SSABuilder, op: &mut Op) {
        // TODO: use only one registers if the same FAU is read twice
        for (src_idx, src) in op.srcs_mut().iter_mut().enumerate() {
            let SrcRef::FAU(fau) = &mut src.src_ref else {
                continue;
            };
            if self.inval_src.contains(src_idx) || !self.fau_retained(fau) {
                move_src_to_tmp(b, src, src.src_ref.bytes_read());
            } else if self.src64_w1.contains(src_idx) {
                // We already checked that this swizzle is supported
                fau.idx &= !1;
                fau.load64 = true;
                src.swizzle =
                    Swizzle::replicate_word(1).swizzle(src.swizzle).unwrap();
            }
        }
    }

    fn sort_bandwidth(&mut self) {
        // Prioritize on the bandwidth of each index
        self.slots.sort_by_key(|slot| {
            Reverse((slot.words_used.count_ones(), slot.use_count))
        });
    }

    /// Instructions can always only read from the same page
    /// for FAU-RAM or FAU-special (does not apply to small-consts)
    fn just_one_page(&mut self) {
        const _: () = {
            // SmallConst has to come last
            assert!(
                HWFAUPage::MAX_DISCRIMINANT == (HWFAUPage::SmallConst as u8)
            );
        };

        let mut pages_read = <HWFAUPage as EnumAsU8>::VariantSet::new();
        let mut page_words_read = [0_u8; HWFAUPage::SmallConst as u8 as usize];
        for fau in self.slots.iter() {
            let page = fau.hw_page(self.fau_model);
            let words = fau.words_used.count_ones() as u8;
            if page != HWFAUPage::SmallConst {
                pages_read.insert(page);
                page_words_read[usize::from(page.as_u8())] += words;
            }
        }

        if pages_read.len() <= 1 {
            return;
        }

        // Select the page with the most words used
        let selected_page = page_words_read
            .iter()
            .enumerate()
            .max_by_key(|(_page, count)| **count)
            .unwrap()
            .0 as u8;

        // Filter sources
        self.slots.retain(|fau| {
            fau.page.is_small_const()
                || fau.hw_page(self.fau_model).as_u8() == selected_page
        });
    }

    fn just_one_index(&mut self) {
        // Keep all constants and only the first non-const slot
        let mut found = false;
        self.slots.retain(|fau| {
            if fau.page.is_small_const() {
                true
            } else if !found {
                found = true;
                true
            } else {
                false
            }
        });
    }

    /// The combined amount of FAU (including small constants) is at most
    /// 64 bits, not including k0 (SrcRef::Zero).
    fn limit_bandwidth(&mut self) {
        // Limit the bandwidth to 64 bits (2 words)
        // Instructions can only use one distinct special FAU
        let mut special_taken = false;
        let mut used_words = 0;

        // On some archs (< v12) k0 counts towards the bandwidth, but we must
        // not legalize it (it could be part of an aliased instruction, thrown
        // away by the encoder).
        if self.reads_k0 && !self.fau_model.is_zero_free {
            used_words += 1;
        }
        for fau in self.slots.iter_mut() {
            let mut words = fau.words_used.count_ones();
            if used_words + words > self.max_words {
                let word = usize::from(fau.scalar_uses[1] > fau.scalar_uses[0]);
                if used_words < self.max_words && fau.scalar_uses[word] > 0 {
                    fau.words_used = 1 << word;
                    words = 1;
                } else {
                    fau.words_used = 0;
                    continue;
                }
            }
            if fau.page.is_special() {
                if special_taken {
                    fau.words_used = 0;
                    continue;
                } else {
                    special_taken = true;
                }
            }
            used_words += words;
        }
        self.slots.retain(|fau| fau.words_used != 0);
        debug_assert!(used_words <= self.max_words);
    }

    fn legalize_message(&mut self) {
        // Messages cannot read from special page 3 or from warp_id
        self.slots.retain(|fau| {
            let is_warp_id =
                fau.page == FAUPage::Special0 && fau.idx64 == 0b0010;
            fau.page != FAUPage::Special3 && !is_warp_id
        });

        // No limit on the number of FAU indices
        // We can use as many constants as we want
        // The only legalization required is that there must be one page
        self.just_one_page();
    }

    fn is_trivial(&self) -> bool {
        if self.slots.len() == 0 {
            return true; // Nothing to legalize
        }

        // 1 slot is always legal (unless we have a 0 in v9-v10)
        self.slots.len() == 1
            && self.slots[0].words_used.count_ones() <= self.max_words
            && !(self.reads_k0 && !self.fau_model.is_zero_free)
    }

    fn legalize_execution_unit(&mut self) {
        if self.is_trivial() {
            return;
        }

        self.sort_bandwidth();
        if self.fau_model.single_fau_ram_index {
            self.just_one_index();
        } else {
            self.just_one_page();
        }

        self.limit_bandwidth();
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum CopyKey {
    Imm(u64),
    FAU(FAUPage, u16, bool),
}

pub fn src_copy_keys(model: &dyn Model, op: &Op) -> Vec<CopyKey> {
    let mut ctx = LegalizeFAU::new(model.fau());
    ctx.max_words = fau_word_limit(model, op);
    let planned = ctx.extract_planned_srcs(model, op);
    if !ctx.slots.is_empty() {
        if model.op_is_message(op) {
            ctx.legalize_message();
        } else {
            ctx.legalize_execution_unit();
        }
    }
    let placed = |src_idx: usize, fau: &FAURef| {
        !ctx.inval_src.contains(src_idx) && ctx.fau_retained(fau)
    };
    op.srcs()
        .iter()
        .enumerate()
        .filter_map(|(src_idx, src)| match &src.src_ref {
            SrcRef::FAU(fau) if fau.page == FAUPage::Virtual => {
                match planned.iter().find(|(idx, _)| *idx == src_idx) {
                    Some((_, planned)) if placed(src_idx, planned) => None,
                    _ => Some(CopyKey::FAU(fau.page, fau.idx, fau.load64)),
                }
            }
            SrcRef::FAU(fau) if !placed(src_idx, fau) => {
                Some(CopyKey::FAU(fau.page, fau.idx, fau.load64))
            }
            SrcRef::Imm32(v) => {
                if model.op_src_supports_imm32(op, src, v.get()) {
                    return None;
                }
                match planned.iter().find(|(idx, _)| *idx == src_idx) {
                    Some((_, fau)) if placed(src_idx, fau) => None,
                    _ => Some(CopyKey::Imm(v.get().into())),
                }
            }
            SrcRef::Imm64(v) => Some(CopyKey::Imm(v.get())),
            _ => None,
        })
        .collect()
}

fn legalize_fau_srcs(
    b: &mut impl SSABuilder,
    fau_model: &FAUModel,
    op: &mut Op,
) {
    let mut ctx = LegalizeFAU::new(fau_model);
    ctx.max_words = fau_word_limit(b.model(), op);

    ctx.extract_srcs(b.model(), op, false);
    if ctx.slots.is_empty() && ctx.inval_src.is_empty() {
        return;
    }

    // We can have zero legal slots even if some sources are FAU in the case
    // where we early-discard a 64-bit source.  But, in this case, further
    // legalization is unnecessary.
    if !ctx.slots.is_empty() {
        if b.model().op_is_message(op) {
            ctx.legalize_message();
        } else {
            ctx.legalize_execution_unit();
        }
    }

    ctx.apply_legalization(b, op);
}

impl Shader<'_> {
    pub fn legalize(&mut self) {
        let live = Liveness::for_shader(self);

        let model = self.model;
        let fau = model.fau();
        let mut ssa_used = SSAValueSet::new();
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            let bl = live.block(bi);
            let mut count = 0..;
            block.map_instrs(|mut instr| {
                let ip = count.next().unwrap();
                let mut b = SSAInstrBuilder::new(model, &mut self.ssa_alloc);
                legalize_vec_srcs(&mut b, &mut instr, &mut ssa_used);
                legalize_fixed_srcs(&mut b, bl, &mut instr, ip, &mut ssa_used);
                for src_idx in 0..instr.srcs().len() {
                    legalize_imm_src(&mut b, &mut instr, src_idx);
                }
                legalize_fau_srcs(&mut b, fau, &mut instr);
                b.push_instr(instr);
                b.into_mapped()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use crate::ops::{CmpOp, FClamp, FRound, OpCSel, OpFAdd};
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};

    #[test]
    fn user_fau_page_boundaries() {
        for (gpu_id, variant, page_words) in [
            (0xa0000000, 0, 64_u16),
            (0x0f08_0000_f000_0000, 4, 128_u16),
        ] {
            let model = model_for_gpu_id(gpu_id, variant).unwrap();
            for (a, b) in [(62, 64), (0, 126), (126, 128), (128, 254)] {
                let mut alloc = SSAValueAllocator::default();
                let mut op: Op = OpFAdd {
                    dst: alloc.alloc_ref(32).into(),
                    dst_type: DataType::F32,
                    round: FRound::NearestEven,
                    clamp: FClamp::None,
                    srcs: [FAURef::user_i32(a).into(), FAURef::user_i32(b).into()],
                }.into();
                let mut builder = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
                legalize_fau_srcs(&mut builder, model.fau(), &mut op);
                let same_page = a / page_words == b / page_words;
                let retains_both = same_page && !model.fau().single_fau_ram_index;
                assert_eq!(builder.into_vec().len(), usize::from(!retains_both));
            }
        }
    }

    #[test]
    fn scalar_word_bandwidth() {
        for arch in [9, 10, 12, 13, 14] {
            let model = model_for_gpu_id(arch << 28, 0).unwrap();
            for uses in [[1, 1], [1, 2], [3, 1], [0, 1], [1, 0], [0, 0]] {
                let mut ctx = LegalizeFAU::new(model.fau());
                ctx.reads_k0 = true;
                ctx.slots.push(FAUSlot {
                    page: FAUPage::User,
                    idx64: 4,
                    use_count: uses[0] + uses[1] + 1,
                    words_used: 0b11,
                    scalar_uses: uses,
                    planned_page: None,
                });
                ctx.legalize_execution_unit();
                let full = arch >= 12;
                let any = uses != [0, 0];
                let selected = u16::from(uses[1] > uses[0]);
                for word in 0..2 {
                    assert_eq!(ctx.fau_retained(&FAURef::user_i32(8 + word)),
                               full || (any && word == selected));
                }
                assert_eq!(ctx.fau_retained(&FAURef::user_i64(8)), full);
            }
        }
    }

    #[test]
    fn full_pair_without_zero() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut ctx = LegalizeFAU::new(model.fau());
        ctx.slots.push(FAUSlot {
            page: FAUPage::User,
            idx64: 4,
            use_count: 1,
            words_used: 0b11,
            scalar_uses: [0, 0],
            planned_page: None,
        });
        ctx.legalize_execution_unit();
        assert!(ctx.fau_retained(&FAURef::user_i64(8)));
    }

    #[test]
    fn single_word_limit() {
        let model = model_for_gpu_id(0x0f08_0000_f000_0000, 4).unwrap();
        let mut ctx = LegalizeFAU::new(model.fau());
        ctx.max_words = 1;
        for idx64 in [4, 5] {
            ctx.slots.push(FAUSlot {
                page: FAUPage::User,
                idx64,
                use_count: 1,
                words_used: 0b01,
                scalar_uses: [1, 0],
                planned_page: None,
            });
        }
        ctx.legalize_execution_unit();
        let words: u32 = ctx.slots.iter().map(|s| s.words_used.count_ones()).sum();
        assert_eq!(words, 1);

        let mut ctx = LegalizeFAU::new(model.fau());
        ctx.max_words = 1;
        ctx.slots.push(FAUSlot {
            page: FAUPage::User,
            idx64: 4,
            use_count: 3,
            words_used: 0b11,
            scalar_uses: [1, 2],
            planned_page: None,
        });
        ctx.legalize_execution_unit();
        assert_eq!(ctx.slots.len(), 1);
        assert_eq!(ctx.slots[0].words_used, 0b10);
    }

    #[test]
    fn single_slot_with_zero() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut ctx = LegalizeFAU::new(model.fau());
        ctx.reads_k0 = true;
        for idx64 in [4, 5] {
            ctx.slots.push(FAUSlot {
                page: FAUPage::User,
                idx64,
                use_count: 2,
                words_used: 0b11,
                scalar_uses: [1, 1],
                planned_page: None,
            });
        }
        ctx.legalize_execution_unit();
        assert_eq!(ctx.slots.len(), 1);
        assert_eq!(ctx.slots[0].words_used.count_ones(), 1);
        assert!(!ctx.fau_retained(&FAURef::user_i32(10)));
    }

    #[test]
    fn csel_pair_and_zero() {
        for arch in [10, 12] {
            let model = model_for_gpu_id(arch << 28, 0).unwrap();
            let mut alloc = SSAValueAllocator::default();
            let cmp = alloc.alloc_ref(32);
            let dst = alloc.alloc_ref(32);
            let mut op: Op = OpCSel {
                dst: dst.into(),
                cmp_type: DataType::S32,
                cmp_op: CmpOp::Ne,
                cmp_srcs: [cmp.into(), 0_u32.into()],
                sel_srcs: [FAURef::user_i32(8).into(),
                           FAURef::user_i32(9).into()],
            }.into();
            let mut b = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
            legalize_fau_srcs(&mut b, model.fau(), &mut op);
            assert_eq!(b.into_vec().len(), usize::from(arch == 10));
            let retained = op.srcs().iter()
                .filter(|s| matches!(s.src_ref, SrcRef::FAU(_))).count();
            assert_eq!(retained, if arch == 10 { 1 } else { 2 });
        }
    }
}
