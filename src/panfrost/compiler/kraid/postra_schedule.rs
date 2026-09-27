use crate::ir::*;
use crate::isa::ExecUnit;
use crate::liveness::{PhysicalLiveness, RegAccess};
use crate::model::{RegByteSet, MAX_REG_BYTES};
use crate::ops::{MemoryEffect, VaryingUpdateMode};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const UNITS: usize = MAX_REG_BYTES / 2;

fn alu_unit(model: &dyn Model, instr: &Instr) -> Option<ExecUnit> {
    if instr.flow != FlowCtrl::NONE || !instr.op.can_eliminate()
        || instr.op.memory_effect() != MemoryEffect::None
        || instr.op.var_update_mode() != VaryingUpdateMode::None
        || instr.reads_discard() || instr.writes_discard()
    {
        return None;
    }
    match model.op_exec_unit(&instr.op)? {
        unit @ (ExecUnit::Fma | ExecUnit::Cvt | ExecUnit::Sfu) => Some(unit),
        ExecUnit::Msg => None,
    }
}

fn halves(bytes: RegByteSet) -> Vec<usize> {
    let mut units: Vec<_> = bytes.iter().map(|byte| usize::from(byte / 2)).collect();
    units.dedup();
    units
}

#[derive(Clone, Default)]
struct ValueUses {
    remaining: usize,
    live_out: bool,
}

struct Node {
    successors: Vec<(usize, u32)>,
    predecessors: usize,
    height: u32,
    reads: Vec<(usize, usize)>,
    writes: Vec<usize>,
}

struct ScheduleDAG {
    nodes: Vec<Node>,
    values: Vec<ValueUses>,
}

impl ScheduleDAG {
    fn new(access: &[RegAccess], units: &[ExecUnit], live_out: RegByteSet) -> Self {
        let mut predecessors = vec![BTreeMap::<usize, u32>::new(); access.len()];
        let mut last_writer = vec![None; MAX_REG_BYTES];
        let mut readers = vec![Vec::new(); MAX_REG_BYTES];
        let mut versions: Vec<_> = (0..UNITS).collect();
        let mut values = vec![ValueUses::default(); UNITS];
        let mut nodes = Vec::with_capacity(access.len());
        for (ip, a) in access.iter().enumerate() {
            let mut add_dep = |pred: usize, latency: u32| {
                if pred != ip {
                    predecessors[ip].entry(pred).and_modify(|v| *v = (*v).max(latency)).or_insert(latency);
                }
            };
            for byte in a.reads.iter().map(usize::from) {
                if let Some(pred) = last_writer[byte] {
                    let latency = if units[pred] == ExecUnit::Sfu && units[ip] == ExecUnit::Sfu { 2 } else { 1 };
                    add_dep(pred, latency);
                }
                readers[byte].push(ip);
            }
            for byte in a.writes.iter().map(usize::from) {
                if let Some(pred) = last_writer[byte] {
                    add_dep(pred, 0);
                }
                for pred in readers[byte].drain(..) {
                    add_dep(pred, 0);
                }
                last_writer[byte] = Some(ip);
            }
            let reads = halves(a.reads).into_iter().map(|unit| {
                let version = versions[unit];
                values[version].remaining += 1;
                (unit, version)
            }).collect();
            let writes = halves(a.writes);
            for &unit in &writes {
                let partial = !a.writes.contains((unit * 2) as u16)
                    || !a.writes.contains((unit * 2 + 1) as u16);
                if partial {
                    values[versions[unit]].live_out = true;
                }
                versions[unit] = values.len();
                values.push(ValueUses { remaining: 0, live_out: partial });
            }
            nodes.push(Node { successors: Vec::new(), predecessors: predecessors[ip].len(),
                              height: 0, reads, writes });
        }
        for unit in halves(live_out) {
            values[versions[unit]].live_out = true;
        }
        for (ip, deps) in predecessors.into_iter().enumerate() {
            for (pred, latency) in deps {
                nodes[pred].successors.push((ip, latency));
            }
        }
        for ip in (0..nodes.len()).rev() {
            nodes[ip].height = nodes[ip].successors.iter()
                .map(|&(next, latency)| nodes[next].height + latency).max().unwrap_or(0);
        }
        Self { nodes, values }
    }

    fn cycles(&self, order: impl IntoIterator<Item = usize>) -> u32 {
        let mut depth = vec![0; self.nodes.len()];
        let mut clock = 0;
        for ip in order {
            clock = clock.max(depth[ip]);
            for &(next, latency) in &self.nodes[ip].successors {
                depth[next] = depth[next].max(clock + latency);
            }
            clock += 1;
        }
        clock
    }

    fn schedule(&self) -> Vec<usize> {
        let mut state = ScheduleState {
            clock: 0, depth: vec![0; self.nodes.len()],
            last_def: vec![None; UNITS], last_read: vec![None; UNITS],
            values: self.values.clone(),
        };
        let mut pending: Vec<_> = self.nodes.iter().map(|n| n.predecessors).collect();
        let mut ready: Vec<_> = pending.iter().enumerate().filter_map(|(ip, &n)| (n == 0).then_some(ip)).collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while !ready.is_empty() {
            let mut best = 0;
            for i in 1..ready.len() {
                if state.compare(self, ready[i], ready[best]) == Ordering::Greater {
                    best = i;
                }
            }
            let ip = ready.swap_remove(best);
            let node = &self.nodes[ip];
            state.clock = state.clock.max(state.depth[ip]);
            for &(unit, version) in &node.reads {
                state.values[version].remaining -= 1;
                state.last_read[unit] = Some(state.clock);
            }
            for &unit in &node.writes {
                state.last_def[unit] = Some(state.clock);
            }
            for &(next, latency) in &node.successors {
                state.depth[next] = state.depth[next].max(state.clock + latency);
                pending[next] -= 1;
                if pending[next] == 0 {
                    ready.push(next);
                }
            }
            state.clock += 1;
            order.push(ip);
        }
        assert_eq!(order.len(), self.nodes.len());
        if state.clock > self.cycles(0..self.nodes.len()) {
            (0..self.nodes.len()).collect()
        } else {
            order
        }
    }
}

struct ScheduleState {
    clock: u32,
    depth: Vec<u32>,
    last_def: Vec<Option<u32>>,
    last_read: Vec<Option<u32>>,
    values: Vec<ValueUses>,
}

impl ScheduleState {
    fn forwarding(&self, node: &Node) -> Vec<(u32, bool)> {
        let mut score: Vec<_> = node.reads.iter().filter_map(|&(unit, version)| {
            let cycle = self.last_def[unit]?;
            let value = &self.values[version];
            (self.clock.wrapping_sub(cycle) <= 3).then_some((cycle, value.remaining == 1 && !value.live_out))
        }).collect();
        score.sort_by_key(|&(cycle, _)| std::cmp::Reverse(cycle));
        score
    }

    fn reuse(&self, node: &Node) -> Vec<u32> {
        let mut score: Vec<_> = node.reads.iter().filter_map(|&(unit, _)| {
            let cycle = self.last_read[unit]?;
            (self.clock.wrapping_sub(cycle) <= 3).then_some(cycle)
        }).collect();
        score.sort_by_key(|&cycle| std::cmp::Reverse(cycle));
        score
    }

    fn compare(&self, dag: &ScheduleDAG, a: usize, b: usize) -> Ordering {
        if self.depth[a] > self.clock || self.depth[b] > self.clock {
            let order = self.depth[b].cmp(&self.depth[a]);
            if order != Ordering::Equal {
                return order;
            }
        }
        self.forwarding(&dag.nodes[a]).cmp(&self.forwarding(&dag.nodes[b]))
            .then_with(|| self.reuse(&dag.nodes[a]).cmp(&self.reuse(&dag.nodes[b])))
            .then_with(|| dag.nodes[a].height.cmp(&dag.nodes[b].height))
            .then_with(|| b.cmp(&a))
    }
}

fn refresh_last_use(model: &dyn Model, b: &mut BasicBlock, mut live: RegByteSet) {
    for instr in b.instrs.iter_mut().rev() {
        let access = RegAccess::for_instr(model, instr);
        let retained = live - access.writes;
        for src in instr.srcs_mut() {
            src.last_use = match src.src_ref.as_reg() {
                Some(reg) if matches!(reg.range, RegRange::Regs(_)) => {
                    !reg.byte_range().any(|byte| retained.contains(byte))
                }
                _ => false,
            };
        }
        live = access.live_before(live);
    }
}

impl Shader<'_> {
    pub fn schedule_after_ra(&mut self) {
        if self.model.arch() != 10 {
            return;
        }
        let physical = PhysicalLiveness::for_shader(self);
        for (bi, b) in self.blocks.iter_mut().enumerate() {
            let access = &physical.access[bi];
            let mut live = physical.live_out[bi];
            let mut at_end = vec![RegByteSet::new(); b.instrs.len()];
            for ip in (0..b.instrs.len()).rev() {
                at_end[ip] = live;
                live = access[ip].live_before(live);
            }
            let mut old_to_new: Vec<_> = (0..b.instrs.len()).collect();
            let mut start = 0;
            let mut changed = false;
            while start < b.instrs.len() {
                let mut units = Vec::new();
                for instr in &b.instrs[start..] {
                    let Some(unit) = alu_unit(self.model, instr) else { break; };
                    units.push(unit);
                }
                let end = start + units.len();
                if units.len() > 1 {
                    let dag = ScheduleDAG::new(&access[start..end], &units, at_end[end - 1]);
                    for (new, old) in dag.schedule().into_iter().enumerate() {
                        old_to_new[start + old] = start + new;
                        changed |= old != new;
                    }
                }
                start = end + 1;
            }
            if changed {
                let count = b.instrs.len();
                unsafe { b.reorder_instrs(|ip| Some(old_to_new[ip]), count); }
                refresh_last_use(self.model, b, physical.live_out[bi]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::FlowWaitBit;
    use crate::model::model_for_gpu_id;
    use crate::ops::{OpFRcp, OpMov, OpNop};
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn access(reads: &[u16], writes: &[u16]) -> RegAccess {
        let mut a = RegAccess { reads: RegByteSet::new(), writes: RegByteSet::new() };
        for &byte in reads { a.reads.insert(byte); }
        for &byte in writes { a.writes.insert(byte); }
        a
    }

    fn reg(idx: u8) -> RegRef {
        RegRef::new(idx, RegRange::Regs(1))
    }

    fn mov(dst: u8, src: u8) -> Instr {
        OpMov { dst: reg(dst).into(), dst_type: DataType::I32, src: reg(src).into() }.into()
    }

    fn shader(model: &dyn Model, instrs: Vec<Instr>) -> Shader<'_> {
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
        cfg.add_node(0, BasicBlock { label: LabelAllocator::default().alloc(), instrs });
        Shader { model, ssa_alloc: Default::default(), phi_alloc: Default::default(),
                 blocks: cfg.as_cfg(false), info: ShaderInfo::default(), constant_pool: None }
    }

    #[test]
    fn sfu_chain_uses_independent_work() {
        let dag = ScheduleDAG::new(&[
            access(&[0, 1], &[4, 5]), access(&[4, 5], &[8, 9]),
            access(&[12, 13], &[16, 17]),
        ], &[ExecUnit::Sfu, ExecUnit::Sfu, ExecUnit::Fma], RegByteSet::new());
        assert_eq!(dag.cycles(0..3), 4);
        let order = dag.schedule();
        assert_eq!(order, vec![0, 2, 1]);
        assert_eq!(dag.cycles(order), 3);
    }

    #[test]
    fn physical_aliases_preserve_raw_war_waw() {
        let dag = ScheduleDAG::new(&[
            access(&[0, 1, 2, 3], &[4, 5, 6, 7]),
            access(&[4, 5], &[0]),
            access(&[0, 1], &[5]),
            access(&[5], &[1]),
        ], &[ExecUnit::Cvt; 4], RegByteSet::new());
        let order = dag.schedule();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn forwarding_respects_last_use_and_window() {
        let mut live = RegByteSet::new();
        live.insert_range(0..2);
        let dag = ScheduleDAG::new(&[
            access(&[0, 1], &[4, 5]), access(&[2, 3], &[6, 7]),
        ], &[ExecUnit::Fma; 2], live);
        let mut state = ScheduleState { clock: 3, depth: vec![0; 2],
            last_def: vec![Some(0); UNITS], last_read: vec![Some(0); UNITS], values: dag.values.clone() };
        assert_eq!(state.forwarding(&dag.nodes[0]), vec![(0, false)]);
        assert_eq!(state.forwarding(&dag.nodes[1]), vec![(0, true)]);
        assert_eq!(state.compare(&dag, 1, 0), Ordering::Greater);
        state.clock = 4;
        assert!(state.forwarding(&dag.nodes[0]).is_empty());
        assert!(state.reuse(&dag.nodes[1]).is_empty());
    }

    #[test]
    fn last_use_is_recomputed_with_partial_writes_and_live_out() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let mut instrs = vec![mov(2, 0), mov(3, 0), mov(0, 1), mov(4, 0)];
        instrs[0].srcs_mut()[0].last_use = true;
        instrs[2].dsts_mut()[0].lanes = DstLanes::H0;
        let mut b = BasicBlock { label: LabelAllocator::default().alloc(), instrs };
        let mut live = RegByteSet::new();
        live.insert_range(0..4);
        refresh_last_use(model.as_ref(), &mut b, live);
        assert!(!b.instrs[0].srcs()[0].last_use);
        assert!(!b.instrs[1].srcs()[0].last_use);
        assert!(b.instrs[2].srcs()[0].last_use);
        assert!(!b.instrs[3].srcs()[0].last_use);
        refresh_last_use(model.as_ref(), &mut b, RegByteSet::new());
        assert!(!b.instrs[1].srcs()[0].last_use);
        assert!(b.instrs[3].srcs()[0].last_use);
    }

    #[test]
    fn shader_schedule_preserves_flow_boundaries() {
        let model = model_for_gpu_id(0xa0000000, 0).unwrap();
        let rcp = |dst, src| Instr::from(OpFRcp { dst: reg(dst).into(),
            dst_type: DataType::F32, src: reg(src).into() });
        let mut barrier = Instr::from(OpNop {});
        barrier.flow.set_wait_bit(FlowWaitBit::Slot0);
        let mut s = shader(model.as_ref(), vec![rcp(1, 0), rcp(2, 1), mov(4, 3),
                                               barrier, rcp(5, 4), rcp(6, 5), mov(8, 7)]);
        s.schedule_after_ra();
        assert!(matches!(s.blocks[0].instrs[1].op, Op::Mov(_)));
        assert!(matches!(s.blocks[0].instrs[3].op, Op::Nop(_)));
        assert!(s.blocks[0].instrs[3].flow != FlowCtrl::NONE);
        assert!(matches!(s.blocks[0].instrs[5].op, Op::Mov(_)));
    }

    #[test]
    fn randomized_physical_programs_preserve_all_bytes() {
        let mut random = 0x243f6a88_u32;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };
        let execute = |program: &[RegAccess], order: &[usize], initial: &[u8; 32]| {
            let mut bytes = *initial;
            for &ip in order {
                let a = &program[ip];
                let value = a.reads.iter().fold(ip as u8, |v, byte|
                    v.rotate_left(1).wrapping_add(bytes[usize::from(byte)]));
                for byte in a.writes.iter() {
                    bytes[usize::from(byte)] = value.wrapping_add(byte as u8);
                }
            }
            bytes
        };
        for _ in 0..6000 {
            let count = 5 + next() as usize % 40;
            let mut program = Vec::new();
            let mut units = Vec::new();
            for _ in 0..count {
                let mut a = access(&[], &[]);
                for _ in 0..(next() % 4) {
                    let width = 1_u16 << (next() % 3);
                    let start = (next() as u16 % 32) & !(width - 1);
                    a.reads.insert_range(start..start + width);
                }
                let width = 1_u16 << (next() % 3);
                let start = (next() as u16 % 32) & !(width - 1);
                a.writes.insert_range(start..start + width);
                program.push(a);
                units.push(match next() % 3 { 0 => ExecUnit::Fma, 1 => ExecUnit::Cvt, _ => ExecUnit::Sfu });
            }
            let mut initial = [0; 32];
            for v in &mut initial { *v = next() as u8; }
            let dag = ScheduleDAG::new(&program, &units, RegByteSet::new());
            let order = dag.schedule();
            let original: Vec<_> = (0..count).collect();
            assert_eq!(execute(&program, &original, &initial), execute(&program, &order, &initial));
            assert!(dag.cycles(order) <= dag.cycles(original));
        }
    }
}
