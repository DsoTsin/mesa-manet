// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;
use std::num::{NonZeroU32, NonZeroU64};

use crate::ir::*;
use crate::ops::{MemAccess, OpLdPka};
use crate::ssa_value::AllocSSA;
use kraid_bindings::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ConstEntry {
    I32(NonZeroU32),
    I64(NonZeroU64),
}

impl ConstEntry {
    fn from_src(model: &dyn Model, op: &Op, src: &Src) -> Option<Self> {
        match &src.src_ref {
            SrcRef::Imm32(v) => (!model.op_src_supports_imm32(op, src, v.get()))
                .then_some(ConstEntry::I32(*v)),
            SrcRef::Imm64(v) => Some(ConstEntry::I64(*v)),
            _ => None,
        }
    }

    fn words(self) -> u32 {
        match self {
            ConstEntry::I32(_) => 1,
            ConstEntry::I64(_) => 2,
        }
    }

    fn word_values(self) -> [u32; 2] {
        match self {
            ConstEntry::I32(v) => [v.get(), 0],
            ConstEntry::I64(v) => [v.get() as u32, (v.get() >> 32) as u32],
        }
    }

    fn high_is_zero(self) -> bool {
        matches!(self, ConstEntry::I64(v) if v.get() >> 32 == 0)
    }
}

fn texture_handle(op: &Op, src: &Src) -> bool {
    let handle = match op {
        Op::TexDual(op) => &op.handle,
        Op::TexFetch(op) => &op.handle,
        Op::TexGather(op) => &op.handle,
        Op::TexGradient(op) => &op.handle,
        Op::TexSingle(op) => &op.handle,
        _ => return false,
    };
    std::ptr::eq(src, handle)
}

fn needs_even_word(model: &dyn Model, op: &Op, src: &Src) -> bool {
    model.op_src_is_64bit(op, src)
        && !Swizzle::replicate_word(1)
            .swizzle(src.swizzle)
            .is_some_and(|s| model.op_src_supports_swizzle(op, src, s))
}

const MAX_FAU_WORDS: usize = 256;
const FREE: u16 = u16::MAX;
const RESERVED: u16 = u16::MAX - 1;

const LOAD_WEIGHT: u64 = 5;

fn loop_weight(depth: u32) -> u64 {
    1_u64 << (4 * depth.min(6))
}

fn value_root(values: &[pan_fau_value], mut i: usize) -> usize {
    while values[i].alias >= 0 {
        i = values[i].alias as usize;
    }
    i
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Item {
    Value(usize),
    Const(ConstEntry),
}

struct Candidate {
    item: Item,
    size: u32,
    align: u32,
    weight: u64,
    zext: bool,
    late: bool,
}

#[derive(Clone, Copy)]
enum Unit {
    One(usize),
    Pair(usize, usize),
}

struct Occupancy {
    owner: [u16; MAX_FAU_WORDS],
    max: u32,
}

impl Occupancy {
    fn new(reserved: u32, max: u32) -> Self {
        let mut owner = [FREE; MAX_FAU_WORDS];
        for w in 0..reserved {
            owner[w as usize] = RESERVED;
        }
        Occupancy { owner, max }
    }

    fn free(&self, at: u32, size: u32) -> bool {
        at + size <= self.max
            && (at..at + size).all(|w| self.owner[w as usize] == FREE)
    }

    fn take(&mut self, at: u32, size: u32, c: usize) {
        for w in at..at + size {
            self.owner[w as usize] = u16::try_from(c).unwrap();
        }
    }

    fn release(&mut self, at: u32, size: u32) {
        for w in at..at + size {
            self.owner[w as usize] = FREE;
        }
    }

    fn first_fit(&self, begin: u32, end: u32, size: u32, align: u32) -> Option<u32> {
        let mut at = begin.next_multiple_of(align);
        while at + size <= end {
            if self.free(at, size) {
                return Some(at);
            }
            at += align;
        }
        None
    }
}

fn set_bit(bits: &mut [u32], bit: u32) {
    bits[(bit / 32) as usize] |= 1 << (bit % 32);
}

fn clear_bit(bits: &mut [u32], bit: u32) {
    bits[(bit / 32) as usize] &= !(1 << (bit % 32));
}

struct Allocator {
    page_words: u32,
    single_index: bool,
    reserved: u32,
    max: u32,
    cands: Vec<Candidate>,
    index: FxHashMap<Item, usize>,
    adj: Vec<FxHashMap<usize, u64>>,
    fixed: Vec<u64>,
    place: Vec<Option<u32>>,
    occ: Occupancy,
}

impl Allocator {
    fn page_of(&self, word: u32) -> u32 {
        word / self.page_words
    }

    fn pages(&self) -> Vec<(u32, u32)> {
        if self.reserved >= self.max {
            return Vec::new();
        }
        let first = self.page_of(self.reserved);
        let last = self.page_of(self.max - 1);
        (first..=last)
            .map(|p| {
                let begin = (p * self.page_words).max(self.reserved);
                let end = ((p + 1) * self.page_words).min(self.max);
                (begin, end)
            })
            .filter(|(b, e)| b < e)
            .collect()
    }

    fn edge(&self, a: usize, b: usize) -> u64 {
        self.adj[a].get(&b).copied().unwrap_or(0)
    }

    fn need(&self, c: usize) -> u32 {
        self.cands[c].size.next_multiple_of(self.cands[c].align)
    }

    fn is_value(&self, c: usize) -> bool {
        matches!(self.cands[c].item, Item::Value(_))
    }

    fn assign_pages(&self, key: &dyn Fn(usize) -> (u8, u64)) -> Vec<usize> {
        let pages = self.pages();
        let n = self.cands.len();
        if pages.len() <= 1 {
            return vec![0; n];
        }
        let total: u32 = (0..n).map(|c| self.need(c)).sum();
        if total <= pages[0].1 - pages[0].0 {
            return vec![0; n];
        }

        let mut values_left: u32 =
            (0..n).filter(|&c| self.is_value(c)).map(|c| self.need(c)).sum();
        let mut capacity_left: u32 = pages.iter().map(|(b, e)| e - b).sum();
        let mut assigned: Vec<Option<usize>> = vec![None; n];
        let prefix_page = self.reserved > 0
            && self.page_of(self.reserved - 1) == self.page_of(pages[0].0);

        for (pi, &(begin, end)) in pages.iter().enumerate() {
            let mut cap = end - begin;
            let mut gain: Vec<u64> = if pi == 0 && prefix_page {
                self.fixed.clone()
            } else {
                vec![0; n]
            };
            loop {
                let mut best: Option<usize> = None;
                for c in 0..n {
                    if assigned[c].is_some() {
                        continue;
                    }
                    let need = self.need(c);
                    if need > cap {
                        continue;
                    }
                    if !self.is_value(c) && capacity_left - need < values_left {
                        continue;
                    }
                    let better = match best {
                        None => true,
                        Some(b) => {
                            (gain[c], self.cands[c].weight, Reverse(key(c)))
                                > (gain[b], self.cands[b].weight, Reverse(key(b)))
                        }
                    };
                    if better {
                        best = Some(c);
                    }
                }
                let Some(c) = best else {
                    break;
                };
                let need = self.need(c);
                assigned[c] = Some(pi);
                cap -= need;
                capacity_left -= need;
                if self.is_value(c) {
                    values_left -= need;
                }
                for (&d, &w) in self.adj[c].iter() {
                    gain[d] += w;
                }
            }
            capacity_left -= cap;
        }
        let mut assigned: Vec<usize> = assigned
            .into_iter()
            .map(|p| p.unwrap_or(pages.len() - 1))
            .collect();
        self.refine_pages(&pages, &mut assigned, prefix_page);
        assigned
    }

    fn refine_pages(
        &self,
        pages: &[(u32, u32)],
        assigned: &mut [usize],
        prefix_page: bool,
    ) {
        const MAX_ROUNDS: usize = 256;
        let n = self.cands.len();
        let np = pages.len();
        let caps: Vec<u32> = pages.iter().map(|(b, e)| e - b).collect();
        let mut used = vec![0_u32; np];
        let mut aff = vec![vec![0_i64; np]; n];
        for c in 0..n {
            used[assigned[c]] += self.need(c);
            if prefix_page {
                aff[c][0] += self.fixed[c] as i64;
            }
            for (&d, &w) in self.adj[c].iter() {
                aff[c][assigned[d]] += w as i64;
            }
        }
        let movers: Vec<usize> = (0..n)
            .filter(|&c| !self.adj[c].is_empty() || self.fixed[c] > 0)
            .collect();
        for _ in 0..MAX_ROUNDS {
            let mut best: Option<(i64, usize, Option<usize>, usize)> = None;
            for &a in &movers {
                let pa = assigned[a];
                for pb in 0..np {
                    if pb == pa {
                        continue;
                    }
                    let gain = aff[a][pb] - aff[a][pa];
                    if gain <= 0 {
                        continue;
                    }
                    if used[pb] + self.need(a) <= caps[pb]
                        && best.is_none_or(|(g, ..)| gain > g)
                    {
                        best = Some((gain, a, None, pb));
                    }
                    for &b in &movers {
                        if assigned[b] != pb {
                            continue;
                        }
                        if used[pa] - self.need(a) + self.need(b) > caps[pa]
                            || used[pb] - self.need(b) + self.need(a) > caps[pb]
                        {
                            continue;
                        }
                        let g = gain + aff[b][pa] - aff[b][pb]
                            - 2 * self.edge(a, b) as i64;
                        if g > 0 && best.is_none_or(|(bg, ..)| g > bg) {
                            best = Some((g, a, Some(b), pb));
                        }
                    }
                }
            }
            let Some((_, a, b, pb)) = best else {
                break;
            };
            let pa = assigned[a];
            let relocate = |x: usize, from: usize, to: usize,
                                assigned: &mut [usize],
                                used: &mut [u32],
                                aff: &mut [Vec<i64>]| {
                assigned[x] = to;
                used[from] -= self.need(x);
                used[to] += self.need(x);
                for (&d, &w) in self.adj[x].iter() {
                    aff[d][from] -= w as i64;
                    aff[d][to] += w as i64;
                }
            };
            relocate(a, pa, pb, assigned, &mut used, &mut aff);
            if let Some(b) = b {
                relocate(b, pb, pa, assigned, &mut used, &mut aff);
            }
        }
    }

    fn match_pairs(&self, assigned: &[usize], keys: &[(u8, u64)]) -> Vec<(usize, usize)> {
        if !self.single_index {
            return Vec::new();
        }
        let single = |c: usize| self.cands[c].size == 1;
        let mut edges: Vec<(u64, usize, usize)> = Vec::new();
        for a in 0..self.cands.len() {
            if !single(a) {
                continue;
            }
            for (&b, &w) in self.adj[a].iter() {
                if b <= a || !single(b) || assigned[a] != assigned[b] {
                    continue;
                }
                if self.cands[a].align == 2 && self.cands[b].align == 2 {
                    continue;
                }
                edges.push((w, a, b));
            }
        }
        edges.sort_by_key(|&(w, a, b)| (Reverse(w), keys[a].min(keys[b]), a, b));
        let mut matched = vec![false; self.cands.len()];
        let mut pairs = Vec::new();
        for (_, a, b) in edges {
            if matched[a] || matched[b] {
                continue;
            }
            matched[a] = true;
            matched[b] = true;
            let low_first = |c: usize| {
                (
                    Reverse(self.cands[c].align),
                    !self.is_value(c),
                    keys[c],
                )
            };
            if low_first(a) <= low_first(b) {
                pairs.push((a, b));
            } else {
                pairs.push((b, a));
            }
        }
        pairs
    }

    fn page_order(&self, page: usize) -> Vec<(u32, u32)> {
        let pages = self.pages();
        let mut order = vec![pages[page]];
        order.extend(
            pages
                .iter()
                .enumerate()
                .filter(|&(p, _)| p != page)
                .map(|(_, r)| *r),
        );
        order
    }

    fn shape(&self, unit: Unit) -> (u32, u32) {
        match unit {
            Unit::One(c) => (self.cands[c].size, self.cands[c].align),
            Unit::Pair(..) => (2, 2),
        }
    }

    fn commit(&mut self, unit: Unit, at: u32) {
        match unit {
            Unit::One(c) => {
                self.occ.take(at, self.cands[c].size, c);
                self.place[c] = Some(at);
            }
            Unit::Pair(lo, hi) => {
                self.occ.take(at, 1, lo);
                self.occ.take(at + 1, 1, hi);
                self.place[lo] = Some(at);
                self.place[hi] = Some(at + 1);
            }
        }
    }

    fn first_fit(&self, unit: Unit, page: usize) -> Option<u32> {
        let (size, align) = self.shape(unit);
        self.page_order(page)
            .into_iter()
            .find_map(|(begin, end)| self.occ.first_fit(begin, end, size, align))
    }

    fn reclaim_window(&self, c: usize, page: usize) -> Option<(u32, Vec<usize>)> {
        let size = self.cands[c].size;
        let align = self.cands[c].align;
        for (begin, end) in self.page_order(page) {
            let mut best: Option<(u64, u32, Vec<usize>)> = None;
            let mut at = begin.next_multiple_of(align);
            while at + size <= end {
                let mut owners: Vec<usize> = Vec::new();
                let mut ok = true;
                for w in at..at + size {
                    match self.occ.owner[w as usize] {
                        FREE => {}
                        RESERVED => ok = false,
                        o => {
                            let o = usize::from(o);
                            if self.is_value(o) {
                                ok = false;
                            } else if !owners.contains(&o) {
                                owners.push(o);
                            }
                        }
                    }
                }
                if ok {
                    let cost: u64 = owners.iter().map(|&o| self.cands[o].weight).sum();
                    if best.as_ref().is_none_or(|(bc, ..)| cost < *bc) {
                        best = Some((cost, at, owners));
                    }
                }
                at += align;
            }
            if let Some((_, at, owners)) = best {
                return Some((at, owners));
            }
        }
        None
    }

    fn place_one(&mut self, c: usize, page: usize) -> bool {
        if let Some(at) = self.first_fit(Unit::One(c), page) {
            self.commit(Unit::One(c), at);
            return true;
        }
        if !self.is_value(c) {
            return false;
        }
        let Some((at, owners)) = self.reclaim_window(c, page) else {
            return false;
        };
        for o in owners {
            let from = self.place[o].take().unwrap();
            self.occ.release(from, self.cands[o].size);
        }
        self.commit(Unit::One(c), at);
        true
    }

    fn place_unit(&mut self, unit: Unit, pages: &[usize]) {
        match unit {
            Unit::One(c) => {
                self.place_one(c, pages[c]);
            }
            Unit::Pair(lo, hi) => {
                if let Some(at) = self.first_fit(unit, pages[lo]) {
                    self.commit(unit, at);
                    return;
                }
                let mut order = [lo, hi];
                order.sort_by_key(|&c| !self.is_value(c));
                for c in order {
                    self.place_one(c, pages[c]);
                }
            }
        }
    }
}

impl Shader<'_> {
    fn rewrite_fau_srcs(
        &mut self,
        virt: Option<&pan_fau_virtual>,
        consts: &FxHashMap<ConstEntry, (u32, bool)>,
    ) {
        let model = self.model;
        for block in self.blocks.iter_mut() {
            for instr in block.instrs.iter_mut() {
                for src_idx in 0..instr.srcs().len() {
                    let src = &instr.srcs()[src_idx];
                    let new_ref = match &src.src_ref {
                        SrcRef::FAU(fau) if fau.page == FAUPage::Virtual => {
                            let virt = virt.expect("Virtual FAU without a table");
                            let at = virt.map[usize::from(fau.idx)];
                            assert!(at >= 0, "Unplaced virtual FAU word");
                            let at = u16::try_from(at).unwrap();
                            let mut placed = if fau.load64 {
                                FAURef::user_i64(at)
                            } else {
                                FAURef::user_i32(at)
                            };
                            placed.special = fau.special;
                            placed.imm32 = fau.imm32;
                            SrcRef::FAU(placed)
                        }
                        _ => {
                            let Some(entry) =
                                ConstEntry::from_src(model, &instr.op, src)
                            else {
                                continue;
                            };
                            let Some(&(at, zext)) = consts.get(&entry) else {
                                continue;
                            };
                            let at = u16::try_from(at).unwrap();
                            SrcRef::FAU(match entry {
                                ConstEntry::I32(_) => FAURef::user_i32(at),
                                ConstEntry::I64(_) if zext => {
                                    FAURef::user_i64_zext(at)
                                }
                                ConstEntry::I64(_) => FAURef::user_i64(at),
                            })
                        }
                    };
                    instr.srcs_mut()[src_idx].src_ref = new_ref;
                }
            }
        }
    }

    fn append_consts(
        &mut self,
        fau: &mut pan_fau_layout,
    ) -> FxHashMap<ConstEntry, (u32, bool)> {
        let mut entries: Vec<ConstEntry> = Vec::new();
        for instr in self.blocks.iter().flat_map(|b| b.instrs.iter()) {
            for src in instr.srcs() {
                if let Some(entry) = ConstEntry::from_src(self.model, &instr.op, src)
                {
                    if !entries.contains(&entry) {
                        entries.push(entry);
                    }
                }
            }
        }
        let mut consts = FxHashMap::default();
        for entry in entries {
            let start = fau.count.next_multiple_of(entry.words());
            if start + entry.words() > fau.max {
                continue;
            }
            for w in fau.count..start + entry.words() {
                fau.words[w as usize].constant = 0;
                set_bit(&mut fau.is_const, w);
                clear_bit(&mut fau.is_pilot, w);
            }
            let words = entry.word_values();
            for k in 0..entry.words() {
                fau.words[(start + k) as usize].constant = words[k as usize];
            }
            consts.insert(entry, (start, false));
            fau.count = start + entry.words();
        }
        consts
    }

    fn collect_candidates(
        &self,
        values: &[pan_fau_value],
        value_of: &[u16],
        dropped: &FxHashSet<usize>,
        promote: bool,
        fau: &pan_fau_layout,
    ) -> (Allocator, Vec<Vec<usize>>) {
        let model = self.model;
        let fau_model = model.fau();
        let zext_ok = model.arch() >= 14;

        let mut alloc = Allocator {
            page_words: u32::from(fau_model.user_page_words()),
            single_index: fau_model.single_fau_ram_index,
            reserved: fau.reserved,
            max: fau.max,
            cands: Vec::new(),
            index: FxHashMap::default(),
            adj: Vec::new(),
            fixed: Vec::new(),
            place: Vec::new(),
            occ: Occupancy::new(fau.reserved, fau.max),
        };

        for (i, value) in values.iter().enumerate() {
            if value.alias >= 0 || dropped.contains(&i) {
                continue;
            }
            alloc.index.insert(Item::Value(i), alloc.cands.len());
            alloc.cands.push(Candidate {
                item: Item::Value(i),
                size: u32::from(value.size),
                align: 1,
                weight: 0,
                zext: false,
                late: false,
            });
        }

        let mut blocks_of: Vec<Vec<usize>> = vec![Vec::new(); alloc.cands.len()];
        let mut even_only: FxHashSet<ConstEntry> = FxHashSet::default();
        let mut dual_handles: FxHashSet<ConstEntry> = FxHashSet::default();
        let mut other_consts: FxHashSet<ConstEntry> = FxHashSet::default();
        let mut wide_values: FxHashSet<usize> = FxHashSet::default();
        let mut even_values: FxHashSet<usize> = FxHashSet::default();
        let mut co_uses: Vec<(Vec<usize>, bool, u64)> = Vec::new();
        let mut full_width: FxHashSet<ConstEntry> = FxHashSet::default();
        if zext_ok && promote {
            for block in self.blocks.iter() {
                for instr in block.instrs.iter() {
                    for src in instr.srcs() {
                        if let Some(entry) =
                            ConstEntry::from_src(model, &instr.op, src)
                        {
                            if entry.high_is_zero()
                                && !texture_handle(&instr.op, src)
                            {
                                full_width.insert(entry);
                            }
                        }
                    }
                }
            }
        }
        for bi in 0..self.blocks.len() {
            let w = loop_weight(u32::try_from(self.blocks.loop_depth(bi)).unwrap());
            for instr in self.blocks[bi].instrs.iter() {
                let mut items: Vec<usize> = Vec::new();
                let mut fixed = false;
                for src in instr.srcs() {
                    let item = match &src.src_ref {
                        SrcRef::FAU(f) if f.page == FAUPage::Virtual => {
                            let vi = usize::from(value_of[usize::from(f.idx)]);
                            let root = value_root(values, vi);
                            if f.load64 {
                                wide_values.insert(root);
                            } else if (f.idx - values[vi].word) % 2 == 0
                                && needs_even_word(model, &instr.op, src)
                            {
                                even_values.insert(root);
                            }
                            Item::Value(root)
                        }
                        SrcRef::FAU(f) if f.page == FAUPage::User => {
                            fixed = true;
                            continue;
                        }
                        _ => {
                            if !promote {
                                continue;
                            }
                            let Some(entry) =
                                ConstEntry::from_src(model, &instr.op, src)
                            else {
                                continue;
                            };
                            if needs_even_word(model, &instr.op, src) {
                                even_only.insert(entry);
                            }
                            if matches!(&instr.op, Op::TexDual(_))
                                && texture_handle(&instr.op, src)
                            {
                                dual_handles.insert(entry);
                            } else {
                                other_consts.insert(entry);
                            }
                            Item::Const(entry)
                        }
                    };
                    let c = match alloc.index.get(&item) {
                        Some(&c) => c,
                        None => {
                            let Item::Const(entry) = item else {
                                unreachable!();
                            };
                            let zext = zext_ok
                                && entry.high_is_zero()
                                && !full_width.contains(&entry);
                            alloc.index.insert(item, alloc.cands.len());
                            alloc.cands.push(Candidate {
                                item,
                                size: if zext { 1 } else { entry.words() },
                                align: entry.words(),
                                weight: 0,
                                zext,
                                late: false,
                            });
                            blocks_of.push(Vec::new());
                            alloc.cands.len() - 1
                        }
                    };
                    if !items.contains(&c) {
                        items.push(c);
                    }
                }
                for &c in &items {
                    alloc.cands[c].weight += w;
                    if blocks_of[c].last() != Some(&bi) {
                        blocks_of[c].push(bi);
                    }
                }
                if items.len() >= 2 || (fixed && !items.is_empty()) {
                    co_uses.push((items, fixed, w));
                }
            }
        }

        for c in alloc.cands.iter_mut() {
            match c.item {
                Item::Const(entry) => {
                    if even_only.contains(&entry) {
                        c.align = 2;
                    }
                    c.late = dual_handles.contains(&entry)
                        && !other_consts.contains(&entry);
                }
                Item::Value(v) => {
                    if wide_values.contains(&v)
                        || even_values.contains(&v)
                        || c.size >= 2
                    {
                        c.align = 2;
                    }
                }
            }
        }

        let n = alloc.cands.len();
        alloc.adj = vec![FxHashMap::default(); n];
        alloc.fixed = vec![0; n];
        alloc.place = vec![None; n];
        for (items, fixed, w) in &co_uses {
            for (i, &a) in items.iter().enumerate() {
                if *fixed {
                    alloc.fixed[a] += w;
                }
                for &b in &items[i + 1..] {
                    *alloc.adj[a].entry(b).or_default() += w;
                    *alloc.adj[b].entry(a).or_default() += w;
                }
            }
        }
        (alloc, blocks_of)
    }

    fn fallback_values(
        &self,
        alloc: &Allocator,
        blocks_of: &[Vec<usize>],
        values: &[pan_fau_value],
    ) -> FxHashSet<usize> {
        let mut dropped = FxHashSet::default();
        let n = alloc.cands.len();
        let capacity = alloc.max.saturating_sub(alloc.reserved);
        let demand: u32 = (0..n).map(|c| alloc.need(c)).sum();
        if demand <= capacity {
            return dropped;
        }
        let benefit = |blocks: &[usize]| -> u64 {
            blocks
                .iter()
                .map(|&bi| loop_weight(u32::try_from(self.blocks.loop_depth(bi)).unwrap()))
                .sum()
        };
        let mut must = 0_u32;
        let mut groups: FxHashMap<(u32, u16), (Vec<usize>, Vec<usize>, u32)> =
            FxHashMap::default();
        let mut items: Vec<(u64, u32, Option<(u32, u16)>, usize)> = Vec::new();
        for c in 0..n {
            match alloc.cands[c].item {
                Item::Value(v) => {
                    let value = &values[v];
                    if !value.loadable || blocks_of[c].is_empty() {
                        must += alloc.need(c);
                        continue;
                    }
                    let group = groups
                        .entry((value.handle, value.ubo.offset / 16))
                        .or_default();
                    group.0.push(v);
                    for &bi in &blocks_of[c] {
                        if !group.1.contains(&bi) {
                            group.1.push(bi);
                        }
                    }
                    group.2 += alloc.need(c);
                }
                Item::Const(_) => {
                    items.push((benefit(&blocks_of[c]), alloc.need(c), None, c));
                }
            }
        }
        for (&key, (_, blocks, words)) in groups.iter() {
            items.push((LOAD_WEIGHT * benefit(blocks), *words, Some(key), 0));
        }
        items.sort_by(|a, b| {
            (u128::from(b.0) * u128::from(a.1))
                .cmp(&(u128::from(a.0) * u128::from(b.1)))
                .then_with(|| b.2.is_some().cmp(&a.2.is_some()))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });
        let mut avail = capacity.saturating_sub(must);
        for (_, words, group, _) in items {
            if words <= avail {
                avail -= words;
            } else if let Some(key) = group {
                dropped.extend(groups[&key].0.iter().copied());
            }
        }
        dropped
    }

    fn emit_fallback_loads(
        &mut self,
        values: &[pan_fau_value],
        value_of: &[u16],
        dropped: &FxHashSet<usize>,
    ) {
        let lookup = |f: &FAURef| -> Option<((u32, u16), u32, u32)> {
            if f.page != FAUPage::Virtual {
                return None;
            }
            let v = value_root(values, usize::from(value_of[usize::from(f.idx)]));
            if !dropped.contains(&v) {
                return None;
            }
            let value = &values[v];
            assert!(value.loadable);
            let word = u32::from(value.ubo.offset) / 4 + u32::from(f.idx - value.word);
            Some((
                (value.handle, value.ubo.offset / 16),
                word,
                word + u32::from(f.load64),
            ))
        };
        for bi in 0..self.blocks.len() {
            let mut spans: FxHashMap<(u32, u16), (u32, u32, usize)> = FxHashMap::default();
            for (ip, instr) in self.blocks[bi].instrs.iter().enumerate() {
                for src in instr.srcs() {
                    let SrcRef::FAU(f) = &src.src_ref else {
                        continue;
                    };
                    let Some((key, lo, hi)) = lookup(f) else {
                        continue;
                    };
                    let span = spans.entry(key).or_insert((lo, hi, ip));
                    span.0 = span.0.min(lo);
                    span.1 = span.1.max(hi);
                }
            }
            if spans.is_empty() {
                continue;
            }
            let mut keys: Vec<(u32, u16)> = spans.keys().copied().collect();
            keys.sort_unstable();
            let mut loads: FxHashMap<(u32, u16), (u32, SSARef)> = FxHashMap::default();
            let mut inserts: Vec<(usize, Instr)> = Vec::new();
            for key in keys {
                let (lo, hi, ip) = spans[&key];
                let bits = 32 * (hi - lo + 1);
                let dst = self.ssa_alloc.alloc_ref(u16::try_from(bits).unwrap());
                inserts.push((
                    ip,
                    OpLdPka {
                        dst: dst.clone().into(),
                        dst_type: DataType::i(u8::try_from(bits).unwrap()),
                        access: MemAccess::Const,
                        offset: (lo * 4).into(),
                        handle: key.0.into(),
                    }
                    .into(),
                ));
                loads.insert(key, (lo, dst));
            }
            for instr in self.blocks[bi].instrs.iter_mut() {
                for src in instr.srcs_mut() {
                    let SrcRef::FAU(f) = &src.src_ref else {
                        continue;
                    };
                    let load64 = f.load64;
                    let Some((key, word, _)) = lookup(f) else {
                        continue;
                    };
                    let (lo, dst) = &loads[&key];
                    let k = usize::try_from(word - lo).unwrap();
                    src.src_ref = if load64 {
                        SSARef::from([dst[k], dst[k + 1]]).into()
                    } else {
                        SSARef::from(dst[k]).into()
                    };
                }
            }
            inserts.sort_by_key(|&(ip, _)| Reverse(ip));
            for (ip, instr) in inserts {
                self.blocks[bi].instrs.insert(ip, instr);
            }
        }
    }

    pub fn allocate_fau(
        &mut self,
        inputs: &pan_compile_inputs,
        fau: &mut pan_fau_layout,
    ) {
        let promote = inputs.fau.promote_immediates;
        let virt = unsafe { inputs.fau.virt.as_mut() };
        if virt.is_none() && !promote {
            return;
        }
        assert!(fau.max as usize <= MAX_FAU_WORDS);

        let values: Vec<pan_fau_value> = virt
            .as_deref()
            .map_or(Vec::new(), |v| v.values[..v.value_count as usize].to_vec());
        let value_of: Vec<u16> = virt
            .as_deref()
            .map_or(Vec::new(), |v| v.value_of[..v.word_count as usize].to_vec());

        if let Some(v) = virt.as_deref() {
            if v.placed {
                let dropped: FxHashSet<usize> = (0..values.len())
                    .filter(|&i| {
                        values[i].alias < 0 && v.map[usize::from(values[i].word)] < 0
                    })
                    .collect();
                if !dropped.is_empty() {
                    self.emit_fallback_loads(&values, &value_of, &dropped);
                }
                let consts = if promote {
                    self.append_consts(fau)
                } else {
                    FxHashMap::default()
                };
                self.rewrite_fau_srcs(Some(v), &consts);
                return;
            }
        }

        let mut dropped: FxHashSet<usize> = FxHashSet::default();
        let (mut alloc, blocks_of) =
            self.collect_candidates(&values, &value_of, &dropped, promote, fau);
        let fallback = self.fallback_values(&alloc, &blocks_of, &values);
        if !fallback.is_empty() {
            self.emit_fallback_loads(&values, &value_of, &fallback);
            dropped = fallback;
            alloc = self
                .collect_candidates(&values, &value_of, &dropped, promote, fau)
                .0;
        }
        let n = alloc.cands.len();

        let keys: Vec<(u8, u64)> = alloc
            .cands
            .iter()
            .enumerate()
            .map(|(c, cand)| match cand.item {
                Item::Value(v) if u32::from(values[v].source) == PAN_FAU_VALUE_PILOT => {
                    (0_u8, v as u64)
                }
                Item::Value(v) => (
                    1,
                    (u64::from(values[v].ubo.ubo) << 32) | u64::from(values[v].ubo.offset),
                ),
                Item::Const(_) => (2, c as u64),
            })
            .collect();
        let pages = alloc.assign_pages(&|c| keys[c]);

        let pairs = alloc.match_pairs(&pages, &keys);
        let mut paired = vec![false; n];
        let mut units: Vec<Unit> = Vec::new();
        for &(lo, hi) in &pairs {
            paired[lo] = true;
            paired[hi] = true;
            units.push(Unit::Pair(lo, hi));
        }
        units.extend((0..n).filter(|&c| !paired[c]).map(Unit::One));
        units.sort_by_key(|&unit| {
            let (size, _) = alloc.shape(unit);
            let (kind, lead) = match unit {
                Unit::Pair(lo, hi) => (2_u8, if keys[lo] <= keys[hi] { lo } else { hi }),
                Unit::One(c) => (u8::from(size >= 2), c),
            };
            let weight = match unit {
                Unit::Pair(lo, hi) => alloc.cands[lo].weight + alloc.cands[hi].weight,
                Unit::One(c) => alloc.cands[c].weight,
            };
            let rank = if alloc.is_value(lead) { 0 } else { weight };
            (
                alloc.cands[lead].late,
                Reverse(kind),
                Reverse(size),
                keys[lead].0,
                Reverse(rank),
                keys[lead],
            )
        });
        for unit in units {
            alloc.place_unit(unit, &pages);
        }
        for c in 0..n {
            if alloc.place[c].is_none() && !alloc.is_value(c) {
                alloc.place_one(c, pages[c]);
            }
        }
        assert!(
            (0..n).all(|c| !alloc.is_value(c) || alloc.place[c].is_some()),
            "FAU values do not fit"
        );

        let mut end = fau.reserved;
        for (c, at) in alloc.place.iter().enumerate() {
            if let Some(at) = at {
                end = end.max(at + alloc.cands[c].size);
            }
        }

        for w in fau.reserved..end {
            fau.words[w as usize].constant = 0;
            set_bit(&mut fau.is_const, w);
            clear_bit(&mut fau.is_pilot, w);
        }

        let mut const_words: FxHashMap<ConstEntry, (u32, bool)> = FxHashMap::default();
        for (c, at) in alloc.place.iter().enumerate() {
            let Some(at) = *at else {
                continue;
            };
            let cand = &alloc.cands[c];
            match cand.item {
                Item::Value(v) => {
                    let value = &values[v];
                    for k in 0..u32::from(value.size) {
                        let w = at + k;
                        if u32::from(value.source) == PAN_FAU_VALUE_PILOT {
                            clear_bit(&mut fau.is_const, w);
                            set_bit(&mut fau.is_pilot, w);
                        } else {
                            fau.words[w as usize].relocation = pan_ubo_relocation {
                                ubo: value.ubo.ubo,
                                offset: value.ubo.offset
                                    + 4 * u16::try_from(k).unwrap(),
                            };
                            clear_bit(&mut fau.is_const, w);
                            clear_bit(&mut fau.is_pilot, w);
                        }
                    }
                }
                Item::Const(entry) => {
                    let words = entry.word_values();
                    for k in 0..cand.size {
                        fau.words[(at + k) as usize].constant = words[k as usize];
                    }
                    const_words.insert(entry, (at, cand.zext));
                }
            }
        }
        fau.count = end;

        let mut placed_virt = None;
        if let Some(v) = virt {
            for (i, value) in values.iter().enumerate() {
                let root = value_root(&values, i);
                if dropped.contains(&root) {
                    continue;
                }
                let c = alloc.index[&Item::Value(root)];
                let at = alloc.place[c].unwrap();
                for k in 0..u32::from(value.size) {
                    v.map[usize::from(value.word) + k as usize] =
                        i16::try_from(at + k).unwrap();
                }
            }
            v.placed = true;
            placed_virt = Some(&*v);
        }

        self.rewrite_fau_srcs(placed_virt, &const_words);
    }
}
