use crate::ir::*;
use crate::liveness::{PhysicalLiveness, RegAccess};
use crate::model::RegByteSet;
use crate::parallel_copy::ParallelCopy;
use crate::ssa_value::SSAValueAllocator;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
struct Hole {
    start: usize,
    end: usize,
    available: bool,
}

struct FreeIntervals {
    halves: Vec<Vec<Hole>>,
}

impl FreeIntervals {
    fn for_block(access: &[RegAccess], live_out: RegByteSet, regs: u8) -> Self {
        let mut busy = vec![RegByteSet::new(); access.len()];
        let mut live = live_out;
        for (ip, a) in access.iter().enumerate().rev() {
            busy[ip] = live | a.reads | a.writes;
            live = a.live_before(live);
        }
        let halves = (0..usize::from(regs) * 2).map(|half| {
            let mut holes = Vec::new();
            let mut start = 0;
            for (ip, occupied) in busy.iter().enumerate() {
                let byte = (half * 2) as u16;
                if occupied.contains(byte) || occupied.contains(byte + 1) {
                    if start < ip {
                        holes.push(Hole { start, end: ip, available: true });
                    }
                    start = ip + 1;
                }
            }
            if start < busy.len() {
                holes.push(Hole { start, end: busy.len(), available: true });
            }
            holes
        }).collect();
        Self { halves }
    }

    fn select(&self, at: usize, bytes: u8) -> Option<(RegRef, Vec<(usize, usize)>, usize)> {
        let width = usize::from(bytes / 2);
        let mut best = None;
        let mut best_end = 0;
        for first in (0..self.halves.len()).step_by(width) {
            let mut selected = Vec::with_capacity(width);
            let mut end = usize::MAX;
            for half in first..first + width {
                let holes = &self.halves[half];
                let index = holes.partition_point(|hole| hole.end <= at);
                let Some(hole) = holes.get(index) else { break; };
                if !hole.available || hole.start > at {
                    break;
                }
                end = end.min(hole.end);
                selected.push((half, index));
            }
            if selected.len() == width && end > best_end {
                let start = (first * 2) as u16;
                let reg = RegRef::from_byte_range(start..start + u16::from(bytes)).unwrap();
                best = Some((reg, selected, end));
                best_end = end;
            }
        }
        best
    }

    fn plan(&mut self, accesses: &[Access], bytes: u8) -> Option<Vec<Move>> {
        let mut claimed: Vec<(usize, usize)> = Vec::new();
        let mut moves = Vec::new();
        let mut at = accesses[0].ip;
        let mut reg = accesses[0].reg;
        for load in &accesses[1..] {
            while at < load.ip {
                let Some((next, holes, end)) = self.select(at, bytes) else {
                    for (half, index) in claimed {
                        self.halves[half][index].available = true;
                    }
                    return None;
                };
                moves.push(Move { ip: at, src: reg, dst: next });
                for &(half, index) in &holes {
                    self.halves[half][index].available = false;
                }
                claimed.extend(holes);
                at = end;
                reg = next;
            }
            moves.push(Move { ip: load.ip, src: reg, dst: load.reg });
        }
        if moves.iter().filter(|m| m.src != m.dst).count() > accesses.len() {
            for (half, index) in claimed {
                self.halves[half][index].available = true;
            }
            return None;
        }
        Some(moves)
    }
}

#[derive(Clone, Copy)]
struct Access {
    block: usize,
    ip: usize,
    store: bool,
    reg: RegRef,
}

#[derive(Clone, Copy)]
struct Move {
    ip: usize,
    src: RegRef,
    dst: RegRef,
}

#[derive(Default)]
struct Slot {
    invalid: bool,
    accesses: Vec<Access>,
}

impl Slot {
    fn is_candidate(&self) -> bool {
        let Some(first) = self.accesses.first() else { return false; };
        !self.invalid && first.store && self.accesses.len() > 1 &&
            self.accesses[1..].iter().all(|a| a.block == first.block && !a.store)
    }

    fn access(instr: &Instr, mem: MemRef, store: bool, block: usize, ip: usize) -> Option<Access> {
        let Op::Copy(copy) = &instr.op else { return None; };
        if instr.flow != FlowCtrl::NONE || !copy.src.src_mod.is_none() ||
            !matches!(mem.range, 2 | 4) || copy.dst_type.total_bits() != mem.range * 8 {
            return None;
        }
        let reg = if store {
            let reg = *copy.src.src_ref.as_reg()?;
            if copy.src.swizzle != Swizzle::from(reg.range) {
                return None;
            }
            reg
        } else {
            let reg = *copy.dst.dst_ref.as_reg()?;
            if copy.dst.lanes != Dst::from(reg).lanes || copy.src.swizzle != Src::from(mem).swizzle {
                return None;
            }
            reg
        };
        if reg.bytes() != mem.range {
            return None;
        }
        Some(Access { block, ip, store, reg })
    }
}

impl Shader<'_> {
    pub fn unspill(&mut self, tls_base: u32) {
        let mut slots: BTreeMap<(u16, u8), Slot> = BTreeMap::new();
        for (block, b) in self.blocks.iter().enumerate() {
            for (ip, instr) in b.instrs.iter().enumerate() {
                let sources = instr.srcs().iter().filter_map(|src| src.src_ref.as_mem().copied()).map(|mem| (mem, false));
                let dests = instr.dsts().iter().filter_map(|dst| dst.dst_ref.as_mem().copied()).map(|mem| (mem, true));
                for (mem, store) in sources.chain(dests) {
                    let slot = slots.entry((mem.offset, mem.range)).or_default();
                    if u32::from(mem.offset) < tls_base {
                        slot.invalid = true;
                    }
                    if let Some(access) = Slot::access(instr, mem, store, block, ip) {
                        slot.accesses.push(access);
                    } else {
                        slot.invalid = true;
                    }
                }
            }
        }
        let keys: Vec<_> = slots.keys().copied().collect();
        for (i, &(offset, bytes)) in keys.iter().enumerate() {
            for &other in &keys[i + 1..] {
                if other.0 >= offset + u16::from(bytes) {
                    break;
                }
                slots.get_mut(&(offset, bytes)).unwrap().invalid = true;
                slots.get_mut(&other).unwrap().invalid = true;
            }
        }
        if !slots.values().any(Slot::is_candidate) {
            return;
        }
        let PhysicalLiveness { access, live_out } = PhysicalLiveness::for_shader(self);
        for (bi, b) in self.blocks.iter_mut().enumerate() {
            let mut free = FreeIntervals::for_block(&access[bi], live_out[bi], self.info.registers_used);
            let mut moves: Vec<Vec<Move>> = vec![Vec::new(); b.instrs.len()];
            let mut remove = vec![false; b.instrs.len()];
            for (&(_, bytes), slot) in &slots {
                if !slot.is_candidate() || slot.accesses[0].block != bi {
                    continue;
                }
                let Some(plan) = free.plan(&slot.accesses, bytes) else { continue; };
                for m in plan {
                    moves[m.ip].push(m);
                }
                for a in &slot.accesses {
                    remove[a.ip] = true;
                }
            }
            let instrs = std::mem::take(&mut b.instrs);
            for (ip, instr) in instrs.into_iter().enumerate() {
                if !moves[ip].is_empty() {
                    let mut copy = ParallelCopy::new(self.model, false);
                    for m in &moves[ip] {
                        copy.add_copy(m.dst.into(), m.src.into());
                    }
                    b.instrs.extend(copy.into_instrs::<SSAValueAllocator>(None));
                }
                if !remove[ip] {
                    b.instrs.push(instr);
                }
            }
        }
        let mut tls_size = tls_base;
        for b in &self.blocks {
            for instr in &b.instrs {
                let sources = instr.srcs().iter().filter_map(|src| src.src_ref.as_mem());
                let dests = instr.dsts().iter().filter_map(|dst| dst.dst_ref.as_mem());
                for mem in sources.chain(dests) {
                    tls_size = tls_size.max(u32::from(mem.offset) + u32::from(mem.range));
                }
            }
        }
        self.info.tls_size = tls_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use crate::ops::OpCopy;
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn reg(index: u8) -> RegRef {
        RegRef::new(index, RegRange::Regs(1))
    }

    fn copy(dst: impl Into<Dst>, src: impl Into<Src>, bits: u8) -> Instr {
        OpCopy { dst: dst.into(), dst_type: DataType::i(bits), src: src.into() }.into()
    }

    fn shader(model: &dyn Model, blocks: Vec<Vec<Instr>>, regs: u8) -> Shader<'_> {
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
        let mut labels = LabelAllocator::default();
        for (i, instrs) in blocks.into_iter().enumerate() {
            cfg.add_node(i, BasicBlock { label: labels.alloc(), instrs });
            if i > 0 {
                cfg.add_edge(i - 1, i);
            }
        }
        Shader {
            model, ssa_alloc: Default::default(), phi_alloc: Default::default(),
            blocks: cfg.as_cfg(false),
            info: ShaderInfo { registers_used: regs, tls_size: 4, ..Default::default() },
            constant_pool: None,
        }
    }

    #[test]
    fn selection_uses_longest_aligned_intersection() {
        let free = FreeIntervals { halves: vec![
            vec![Hole { start: 0, end: 4, available: true }],
            vec![Hole { start: 0, end: 9, available: true }],
            vec![Hole { start: 0, end: 7, available: true }],
            vec![Hole { start: 0, end: 7, available: true }],
        ] };
        let (r16, _, end16) = free.select(0, 2).unwrap();
        assert!(r16 == RegRef::new(0, RegRange::Half1));
        assert_eq!(end16, 9);
        let (r32, _, end32) = free.select(0, 4).unwrap();
        assert!(r32 == reg(1));
        assert_eq!(end32, 7);
        assert!(free.select(9, 4).is_none());
    }

    #[test]
    fn planning_moves_at_boundaries_and_rolls_back_failure() {
        let half = |start, end| vec![Hole { start, end, available: true }];
        let mut free = FreeIntervals { halves: vec![half(0, 3), half(0, 3), half(2, 6), half(2, 6)] };
        let accesses = [
            Access { block: 0, ip: 0, store: true, reg: reg(2) },
            Access { block: 0, ip: 2, store: false, reg: reg(0) },
            Access { block: 0, ip: 5, store: false, reg: reg(2) },
        ];
        let plan = free.plan(&accesses, 4).unwrap();
        assert_eq!(plan.iter().map(|m| (m.ip, m.src.idx, m.dst.idx)).collect::<Vec<_>>(),
                   [(0, 2, 0), (2, 0, 0), (3, 0, 1), (5, 1, 2)]);
        assert!(free.halves.iter().all(|h| !h[0].available));
        let mut free = FreeIntervals { halves: vec![half(0, 3), half(0, 3), half(4, 6), half(4, 6)] };
        let original = free.halves.clone();
        assert!(free.plan(&accesses, 4).is_none());
        assert_eq!(free.halves, original);
        let mut free = FreeIntervals { halves: vec![half(0, 3), half(0, 3), half(2, 6), half(2, 6)] };
        let original = free.halves.clone();
        assert!(free.plan(&[accesses[0], accesses[2]], 4).is_none());
        assert_eq!(free.halves, original);
    }

    #[test]
    fn unspill_removes_single_store_and_multiple_loads() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mem = MemRef { offset: 0, range: 4 };
        let mut s = shader(model.as_ref(), vec![vec![
            copy(mem, reg(0), 32), copy(reg(0), 42_u32, 32),
            copy(reg(1), mem, 32), copy(reg(2), mem, 32),
        ]], 4);
        s.unspill(0);
        assert_eq!(s.info.tls_size, 0);
        let mut values = [11_u32, 0, 0, 0];
        for instr in &s.blocks[0].instrs {
            let Op::Copy(op) = &instr.op else { panic!("Unexpected instruction"); };
            let dst = op.dst.dst_ref.as_reg().unwrap();
            let value = match &op.src.src_ref {
                SrcRef::Reg(src) => values[usize::from(src.idx)],
                src => u32::try_from(src).unwrap(),
            };
            values[usize::from(dst.idx)] = value;
        }
        assert_eq!(&values[..3], &[42, 11, 11]);
    }

    #[test]
    fn unspill_handles_half_registers_and_tls_floor() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mem = MemRef { offset: 16, range: 2 };
        let src = RegRef::new(0, RegRange::Half1);
        let dst = RegRef::new(1, RegRange::Half0);
        let mut s = shader(model.as_ref(), vec![vec![copy(mem, src, 16), copy(dst, mem, 16)]], 2);
        s.info.tls_size = 18;
        s.unspill(16);
        assert_eq!(s.info.tls_size, 16);
        for instr in &s.blocks[0].instrs {
            assert!(instr.srcs().iter().all(|src| src.src_ref.as_mem().is_none()));
            assert!(instr.dsts().iter().all(|dst| dst.dst_ref.as_mem().is_none()));
        }
    }

    #[test]
    fn successor_live_in_is_unavailable() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mem = MemRef { offset: 0, range: 4 };
        let mut s = shader(model.as_ref(), vec![
            vec![copy(mem, reg(0), 32), copy(reg(0), 42_u32, 32), copy(reg(0), mem, 32)],
            vec![copy(reg(0), reg(1), 32)],
        ], 2);
        s.unspill(0);
        assert_eq!(s.info.tls_size, 4);
        assert!(s.blocks[0].instrs[0].dsts()[0].dst_ref.as_mem().is_some());
    }

    #[test]
    fn cross_block_and_overlapping_slots_are_rejected() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let word = MemRef { offset: 0, range: 4 };
        let half = MemRef { offset: 2, range: 2 };
        let mut s = shader(model.as_ref(), vec![
            vec![copy(word, reg(0), 32)], vec![copy(reg(1), word, 32)],
        ], 4);
        s.unspill(0);
        assert_eq!(s.info.tls_size, 4);
        let mut s = shader(model.as_ref(), vec![vec![
            copy(word, reg(0), 32), copy(RegRef::new(1, RegRange::Half0), half, 16),
            copy(reg(2), word, 32),
        ]], 4);
        s.unspill(0);
        assert_eq!(s.info.tls_size, 4);
        assert_eq!(s.blocks[0].instrs.len(), 3);
    }

    #[test]
    fn partial_writes_preserve_other_live_bytes() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut instr = copy(reg(0), reg(1), 32);
        instr.dsts_mut()[0].lanes = DstLanes::H0;
        let access = RegAccess::for_instr(model.as_ref(), &instr);
        let before = access.live_before(RegByteSet::from(0..4));
        assert!(!before.contains(0));
        assert!(!before.contains(1));
        assert!(before.contains(2));
        assert!(before.contains(3));
    }

    #[test]
    fn randomized_programs_preserve_observable_bytes() {
        fn next(seed: &mut u32) -> u32 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed
        }
        fn range(r: std::ops::Range<u16>) -> std::ops::Range<usize> {
            usize::from(r.start)..usize::from(r.end)
        }
        fn run(instrs: &[Instr], initial: &[u8; 48]) -> [u8; 128] {
            let mut regs = *initial;
            let mut memory = [0_u8; 128];
            for instr in instrs {
                match &instr.op {
                    Op::Copy(op) => {
                        let bytes = usize::from(op.dst_type.total_bits() / 8);
                        let value = match &op.src.src_ref {
                            SrcRef::Reg(r) => regs[range(r.byte_range())].to_vec(),
                            SrcRef::Mem(m) => memory[range(m.byte_range())].to_vec(),
                            r => u32::try_from(r).unwrap().to_le_bytes()[..bytes].to_vec(),
                        };
                        assert_eq!(bytes, value.len());
                        match &op.dst.dst_ref {
                            DstRef::Reg(r) => regs[range(r.byte_range())].copy_from_slice(&value),
                            DstRef::Mem(m) => memory[range(m.byte_range())].copy_from_slice(&value),
                            _ => panic!("Unexpected copy destination"),
                        }
                    },
                    Op::ShiftLop(op) => {
                        assert!(op.shift_op == crate::ops::ShiftOp::None);
                        assert!(op.logic_op == crate::ops::LogicOp::Xor);
                        assert!(!op.not_result);
                        let dst = range(op.dst.dst_ref.as_reg().unwrap().byte_range());
                        let src = range(op.src0.src_ref.as_reg().unwrap().byte_range());
                        assert_eq!(dst.len(), src.len());
                        let old = regs;
                        for (d, s) in dst.zip(src) {
                            regs[d] ^= old[s];
                        }
                    },
                    _ => panic!("Unexpected instruction"),
                }
            }
            memory
        }
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut seed = 0x2da290;
        let mut changed = 0;
        for case in 0..3000 {
            let mut initial = [0_u8; 48];
            for byte in &mut initial {
                *byte = next(&mut seed) as u8;
            }
            let mut slots = Vec::new();
            let mut instrs = Vec::new();
            for index in 0..8 {
                let bytes = if next(&mut seed) & 1 == 0 { 2 } else { 4 };
                let mem = MemRef { offset: 64 + index * 4, range: bytes };
                let half = (next(&mut seed) % 2) as u8;
                let source = RegRef::new((next(&mut seed) % 12) as u8,
                    if bytes == 4 { RegRange::Regs(1) } else if half == 0 { RegRange::Half0 } else { RegRange::Half1 });
                instrs.push(copy(mem, source, bytes * 8));
                slots.push(mem);
            }
            for _ in 0..64 {
                let dst = (next(&mut seed) % 12) as u8;
                match next(&mut seed) % 3 {
                    0 => {
                        let mem = slots[(next(&mut seed) % 8) as usize];
                        let half = next(&mut seed) & 1;
                        let dst = RegRef::new(dst, if mem.range == 4 { RegRange::Regs(1) }
                            else if half == 0 { RegRange::Half0 } else { RegRange::Half1 });
                        instrs.push(copy(dst, mem, mem.range * 8));
                    },
                    1 => instrs.push(copy(reg(dst), next(&mut seed), 32)),
                    _ => instrs.push(copy(reg(dst), reg((next(&mut seed) % 12) as u8), 32)),
                }
            }
            for index in 0..12 {
                instrs.push(copy(MemRef { offset: u16::from(index) * 4, range: 4 }, reg(index), 32));
            }
            let loads = instrs.iter().filter(|i|
                i.srcs().iter().any(|src| src.src_ref.as_mem().is_some())).count();
            let expected = run(&instrs, &initial);
            let mut s = shader(model.as_ref(), vec![instrs], 12);
            s.info.tls_size = 96;
            s.unspill(64);
            let actual = run(&s.blocks[0].instrs, &initial);
            assert_eq!(&expected[..48], &actual[..48], "case {case}");
            assert_eq!(s.info.registers_used, 12);
            assert!(s.info.tls_size >= 64);
            changed += usize::from(s.blocks[0].instrs.iter().filter(|i|
                i.srcs().iter().any(|src| src.src_ref.as_mem().is_some())).count() < loads);
        }
        assert!(changed > 0);
    }
}
