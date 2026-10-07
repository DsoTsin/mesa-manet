// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::ops::*;
use crate::phi::Phi;
use compiler::bitset::BitSet;

fn lod_uses_helpers(mode: TexLodMode) -> bool {
    matches!(
        mode,
        TexLodMode::Computed
            | TexLodMode::ComputedForceDelta
            | TexLodMode::ComputedBias
            | TexLodMode::ComputedBiasForceDelta
    )
}

fn uses_helpers(op: &Op) -> bool {
    match op {
        Op::TexSingle(op) => lod_uses_helpers(op.lod_mode),
        Op::TexDual(op) => lod_uses_helpers(op.lod_mode),
        Op::TexGradient(op) => {
            op.coord_mode != TexGradientCoordMode::Derivative
        }
        Op::Clper(_) | Op::WMask(_) => true,
        _ => false,
    }
}

fn force_delta_mode(mode: TexLodMode) -> Option<TexLodMode> {
    match mode {
        TexLodMode::Computed | TexLodMode::ComputedForceDelta => {
            Some(TexLodMode::ComputedForceDelta)
        }
        _ => None,
    }
}

fn implicit_lod(op: &Op) -> Option<(TexLodMode, &Src, TexDim, bool)> {
    match op {
        Op::TexSingle(op) if lod_uses_helpers(op.lod_mode) => {
            Some((op.lod_mode, &op.data, op.dim, op.projection_enable))
        }
        Op::TexDual(op) if lod_uses_helpers(op.lod_mode) => {
            Some((op.lod_mode, &op.data, op.dim, op.projection_enable))
        }
        _ => None,
    }
}

fn set_force_delta(op: &mut Op) {
    match op {
        Op::TexSingle(op) => {
            if let Some(mode) = force_delta_mode(op.lod_mode) {
                op.lod_mode = mode;
            }
        }
        Op::TexDual(op) => {
            if let Some(mode) = force_delta_mode(op.lod_mode) {
                op.lod_mode = mode;
            }
        }
        _ => {}
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct RegMask([u64; 4]);

impl RegMask {
    fn set(&mut self, reg: u8) {
        self.0[usize::from(reg / 64)] |= 1 << (reg % 64);
    }

    fn clear(&mut self, reg: u8) {
        self.0[usize::from(reg / 64)] &= !(1 << (reg % 64));
    }

    fn test(&self, reg: u8) -> bool {
        self.0[usize::from(reg / 64)] & (1 << (reg % 64)) != 0
    }

    fn union(&mut self, other: &RegMask) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= b;
        }
    }
}

fn reg_indices(reg: &RegRef) -> std::ops::Range<u8> {
    let bytes = reg.byte_range();
    let start = u8::try_from(bytes.start / 4).unwrap();
    let end = u8::try_from(bytes.end.div_ceil(4)).unwrap();
    start..end
}

fn lod_coord_regs(op: &Op) -> Option<Option<std::ops::Range<u8>>> {
    let (_, data, dim, proj) = implicit_lod(op)?;
    let comps = match dim {
        TexDim::Tex1D => 1,
        TexDim::Tex2D => 2,
        TexDim::Tex3D => 3,
        TexDim::Cube => {
            if proj {
                3
            } else {
                2
            }
        }
    } + u8::from(proj && dim != TexDim::Cube);
    Some(match data.src_ref.as_reg() {
        Some(reg) if matches!(reg.range, RegRange::Regs(_)) => {
            let regs = reg_indices(reg);
            Some(regs.start..regs.end.min(regs.start + comps))
        }
        _ => None,
    })
}

fn must_execute(
    instr: &Instr,
    need: &RegMask,
    succ_alive: bool,
    force_delta: bool,
) -> bool {
    match &instr.op {
        Op::Clper(_) | Op::WMask(_) => return true,
        Op::TexGradient(op) => {
            if op.coord_mode != TexGradientCoordMode::Derivative {
                return true;
            }
        }
        Op::Branch(op) => {
            if succ_alive && !op.is_unconditional() {
                return true;
            }
        }
        _ => {}
    }
    if let Some((mode, ..)) = implicit_lod(&instr.op) {
        if !force_delta || force_delta_mode(mode).is_none() {
            return true;
        }
    }
    instr
        .iter_reg_defs()
        .any(|reg| reg_indices(reg).any(|r| need.test(r)))
}

fn set_skip(op: &mut Op, skip: bool) {
    match op {
        Op::TexSingle(op) => op.skip = skip,
        Op::TexDual(op) => op.skip = skip,
        Op::TexFetch(op) => op.skip = skip,
        Op::TexGather(op) => op.skip = skip,
        _ => {}
    }
}

fn can_carry_discard(instr: &Instr) -> bool {
    let mut flow = instr.flow;
    flow.take_msg_slot_idx();
    flow.take_discard();
    flow == FlowCtrl::NONE
        && !matches!(
            instr.op,
            Op::Branch(_) | Op::Jump(_) | Op::BlendCall(_) | Op::PilotJump(_)
        )
}

fn place_discard(block: &mut BasicBlock, after: Option<usize>, is_exit: bool) {
    let from = after.unwrap_or(0);
    if let Some(ip) = (from..block.instrs.len())
        .find(|&ip| can_carry_discard(&block.instrs[ip]))
    {
        block.instrs[ip].flow.set_discard();
        return;
    }
    if is_exit {
        return;
    }
    let mut nop = Instr::from(OpNop {});
    nop.flow.set_discard();
    block.instrs.insert(after.map_or(0, |ip| ip + 1), nop);
}

impl Shader<'_> {
    pub fn mark_helper_skip(&mut self) {
        if !self.info.is_fragment {
            return;
        }

        let mut needed: BitSet<SSAValue> = BitSet::new();
        let mut phis: BitSet<Phi> = BitSet::new();
        for block in self.blocks.iter() {
            for instr in &block.instrs {
                if uses_helpers(&instr.op) || matches!(instr.op, Op::Branch(_))
                {
                    for ssa in instr.iter_ssa_uses() {
                        needed.insert(*ssa);
                    }
                }
            }
        }

        loop {
            let mut progress = false;
            for block in self.blocks.iter().rev() {
                for instr in block.instrs.iter().rev() {
                    let live = match &instr.op {
                        Op::PhiSrc(op) => phis.contains(op.phi),
                        Op::PhiDst(op) => {
                            if instr
                                .iter_ssa_defs()
                                .any(|d| needed.contains(*d))
                            {
                                progress |= phis.insert(op.phi);
                            }
                            false
                        }
                        _ => instr.iter_ssa_defs().any(|d| needed.contains(*d)),
                    };
                    if live {
                        for ssa in instr.iter_ssa_uses() {
                            progress |= needed.insert(*ssa);
                        }
                    }
                }
            }
            if !progress {
                break;
            }
        }

        for block in self.blocks.iter_mut() {
            for instr in &mut block.instrs {
                let exec = instr.iter_ssa_defs().any(|d| needed.contains(*d));
                set_skip(&mut instr.op, !exec);
            }
        }
    }

    pub fn mark_helper_terminate(&mut self) {
        if !self.info.is_fragment || self.info.is_blend || self.is_empty() {
            return;
        }

        let force_delta = self.model.arch() >= 15;
        let n = self.blocks.len();
        let mut alive = vec![false; n];
        let mut need_in = vec![RegMask::default(); n];
        let mut exec: Vec<Vec<bool>> = self
            .blocks
            .iter()
            .map(|b| vec![false; b.instrs.len()])
            .collect();
        loop {
            let mut changed = false;
            for bi in (0..n).rev() {
                let succs = self.blocks.succ_indices(bi);
                let succ_alive = succs.iter().any(|&s| alive[s]);
                let mut need = RegMask::default();
                for &s in succs {
                    need.union(&need_in[s]);
                }
                let mut block_alive = succ_alive;
                for (ip, instr) in
                    self.blocks[bi].instrs.iter().enumerate().rev()
                {
                    let e = must_execute(instr, &need, succ_alive, force_delta);
                    exec[bi][ip] = e;
                    let uses: Vec<u8> = if e {
                        block_alive = true;
                        for reg in instr.iter_reg_defs() {
                            if matches!(reg.range, RegRange::Regs(_)) {
                                for r in reg_indices(reg) {
                                    need.clear(r);
                                }
                            }
                        }
                        instr.iter_reg_uses().flat_map(reg_indices).collect()
                    } else {
                        match lod_coord_regs(&instr.op) {
                            Some(Some(regs)) => regs.collect(),
                            Some(None) => instr
                                .iter_reg_uses()
                                .flat_map(reg_indices)
                                .collect(),
                            None => Vec::new(),
                        }
                    };
                    for r in uses {
                        need.set(r);
                    }
                }
                if need != need_in[bi] || block_alive != alive[bi] {
                    need_in[bi] = need;
                    alive[bi] = block_alive;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for bi in 0..n {
            let mut points: Vec<Option<usize>> = Vec::new();
            let succ_alive =
                self.blocks.succ_indices(bi).iter().any(|&s| alive[s]);
            let dead_from = if alive[bi] {
                if succ_alive {
                    continue;
                }
                let Some(ip) = exec[bi].iter().rposition(|&e| e) else {
                    continue;
                };
                points.push(Some(ip));
                ip + 1
            } else {
                let preds = self.blocks.pred_indices(bi);
                if bi == 0
                    || preds.iter().any(|&p| {
                        self.blocks.succ_indices(p).iter().any(|&s| alive[s])
                    })
                {
                    points.push(None);
                }
                0
            };
            let is_exit = self.blocks.succ_indices(bi).is_empty();
            for (ip, instr) in self.blocks[bi]
                .instrs
                .iter_mut()
                .enumerate()
                .skip(dead_from)
            {
                if force_delta {
                    set_force_delta(&mut instr.op);
                }
                if matches!(instr.op, Op::Discard(_)) {
                    points.push(Some(ip));
                }
            }

            for after in points.into_iter().rev() {
                place_discard(&mut self.blocks[bi], after, is_exit);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::FlowWaitBit;
    use crate::model::model_for_gpu_id;
    use crate::phi::PhiAllocator;
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    const GPU_ID: u64 = 0x0f08_0000_f000_0000;

    fn shader<'a>(
        model: &'a dyn Model,
        alloc: SSAValueAllocator,
        blocks: Vec<Vec<Instr>>,
        edges: &[(usize, usize)],
    ) -> Shader<'a> {
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> =
            CFGBuilder::new();
        let mut labels = LabelAllocator::default();
        for (i, instrs) in blocks.into_iter().enumerate() {
            cfg.add_node(
                i,
                BasicBlock {
                    label: labels.alloc(),
                    instrs,
                },
            );
        }
        for &(from, to) in edges {
            cfg.add_edge(from, to);
        }
        Shader {
            model,
            ssa_alloc: alloc,
            phi_alloc: Default::default(),
            blocks: cfg.as_cfg(false),
            info: ShaderInfo {
                is_fragment: true,
                ..Default::default()
            },
            constant_pool: None,
        }
    }

    fn texture(dst: SSARef, data: Src, lod_mode: TexLodMode) -> Instr {
        OpTexSingle {
            dst: dst.into(),
            dst_type: DataType::A32,
            skip: true,
            dim: TexDim::Tex2D,
            projection_enable: false,
            write_mask: TexWriteMask::new(1),
            wide_indices: false,
            array_enable: false,
            texel_offset: false,
            compare_enable: false,
            lod_mode,
            data,
            handle: 0_u32.into(),
        }
        .into()
    }

    fn quad(dst: Dst, data: Src) -> Instr {
        OpClper {
            dst,
            subgroup: SubgroupSize::Subgroup4,
            lane_op: ClperLaneOp::Xor,
            inactive: ClperInactiveResult::Zero,
            data,
            lane: 1_u32.into(),
        }
        .into()
    }

    fn branch(cond: Src) -> Instr {
        OpBranch {
            not: false,
            cond,
            combine_op: BranchCombineOp::None,
            label: LabelAllocator::default().alloc(),
        }
        .into()
    }

    fn demote() -> Instr {
        OpDiscard {
            cmp_op: CmpOp::Lt,
            srcs: [0_u32.into(), 1_u32.into()],
        }
        .into()
    }

    fn skips(s: &Shader<'_>) -> Vec<bool> {
        s.blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter_map(|i| match &i.op {
                Op::TexSingle(op) => Some(op.skip),
                _ => None,
            })
            .collect()
    }

    fn discard_count(s: &Shader<'_>) -> usize {
        s.blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter(|i| i.flow.get_discard())
            .count()
    }

    #[test]
    fn skip_depends_on_later_helper_users() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        for (lod, first_skip) in [
            (TexLodMode::Computed, false),
            (TexLodMode::ComputedForceDelta, false),
            (TexLodMode::ComputedBias, false),
            (TexLodMode::ComputedBiasForceDelta, false),
            (TexLodMode::Explicit, true),
        ] {
            let mut alloc = SSAValueAllocator::default();
            let a = alloc.alloc_ref(32);
            let b = alloc.alloc_ref(32);
            let mut s = shader(
                model.as_ref(),
                alloc,
                vec![vec![
                    texture(a.clone(), 0_u32.into(), TexLodMode::Computed),
                    texture(b, a.into(), lod),
                ]],
                &[],
            );
            s.mark_helper_skip();
            assert_eq!(skips(&s), [first_skip, true]);
        }
    }

    #[test]
    fn skip_keeps_quad_and_branch_dependencies() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        for branch_user in [false, true] {
            let mut alloc = SSAValueAllocator::default();
            let a = alloc.alloc_ref(32);
            let b = alloc.alloc_ref(32);
            let c = alloc.alloc_ref(32);
            let user = if branch_user {
                branch(b.clone().into())
            } else {
                quad(c.into(), b.clone().into())
            };
            let mut s = shader(
                model.as_ref(),
                alloc,
                vec![vec![
                    texture(a.clone(), 0_u32.into(), TexLodMode::Explicit),
                    OpCopy {
                        dst: b.into(),
                        dst_type: DataType::I32,
                        src: a.into(),
                    }
                    .into(),
                    user,
                ]],
                &[],
            );
            s.mark_helper_skip();
            assert_eq!(skips(&s), [false]);
        }
    }

    #[test]
    fn skip_propagates_through_shared_loop_phi() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        for helper_user in [false, true] {
            let mut alloc = SSAValueAllocator::default();
            let [a, b, p, q] = [(); 4].map(|_| alloc.alloc_ref(32));
            let mut phis = PhiAllocator::default();
            let phi = phis.alloc(32);
            let user = if helper_user {
                quad(q.into(), p.clone().into())
            } else {
                OpRegOut {
                    reg: RegRef::new(0, RegRange::Regs(1)),
                    src_type: DataType::I32,
                    src: p.clone().into(),
                }
                .into()
            };
            let mut s = shader(
                model.as_ref(),
                alloc,
                vec![
                    vec![
                        texture(a.clone(), 0_u32.into(), TexLodMode::Explicit),
                        OpPhiSrc {
                            phi,
                            src_type: DataType::I32,
                            src: a.into(),
                        }
                        .into(),
                    ],
                    vec![
                        OpPhiDst {
                            phi,
                            dst: p.clone().into(),
                            dst_type: DataType::I32,
                        }
                        .into(),
                        texture(b.clone(), p.into(), TexLodMode::Explicit),
                        OpPhiSrc {
                            phi,
                            src_type: DataType::I32,
                            src: b.into(),
                        }
                        .into(),
                        user,
                    ],
                ],
                &[(0, 1), (1, 1)],
            );
            s.phi_alloc = phis;
            s.mark_helper_skip();
            assert_eq!(skips(&s), [!helper_user, !helper_user]);
        }
    }

    #[test]
    fn discard_reuses_an_existing_carrier() {
        let mut instr = Instr::from(OpNop {});
        instr.flow.set_msg_slot_idx(1);
        instr.flow.set_discard();
        let mut block = BasicBlock {
            label: LabelAllocator::default().alloc(),
            instrs: vec![instr, OpNop {}.into()],
        };
        place_discard(&mut block, None, false);
        assert_eq!(block.instrs.len(), 2);
        assert!(block.instrs[0].flow.get_discard());
        assert_eq!(block.instrs[0].flow.get_msg_slot_idx(), Some(1));
        assert!(!block.instrs[1].flow.get_discard());
    }

    #[test]
    fn discard_preserves_conflicting_flow() {
        for flow_kind in 0..3 {
            let mut instr = Instr::from(OpNop {});
            match flow_kind {
                0 => instr.flow.set_wait_bit(FlowWaitBit::Slot0),
                1 => instr.flow.set_reconverge(),
                _ => instr.flow.set_end_shader(),
            }
            let original = instr.flow;
            let mut block = BasicBlock {
                label: LabelAllocator::default().alloc(),
                instrs: vec![instr],
            };
            place_discard(&mut block, None, true);
            assert_eq!(block.instrs.len(), 1);
            assert!(block.instrs[0].flow == original);
        }
    }

    #[test]
    fn discard_inserts_after_helper_before_branch() {
        let mut helper =
            quad(RegRef::new(0, RegRange::Regs(1)).into(), 0_u32.into());
        helper.flow.set_wait_bit(FlowWaitBit::Slot0);
        let original = helper.flow;
        let mut block = BasicBlock {
            label: LabelAllocator::default().alloc(),
            instrs: vec![helper, branch(1_u32.into())],
        };
        place_discard(&mut block, Some(0), false);
        assert_eq!(block.instrs.len(), 3);
        assert!(block.instrs[0].flow == original);
        assert!(matches!(block.instrs[1].op, Op::Nop(_)));
        assert!(block.instrs[1].flow.get_discard());
        assert!(!block.instrs[2].flow.get_discard());
    }

    #[test]
    fn terminate_does_not_repeat_discard_after_demote() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![vec![demote(), OpNop {}.into()]],
            &[],
        );
        s.mark_helper_terminate();
        assert_eq!(discard_count(&s), 1);
        assert!(s.blocks[0].instrs[0].flow.get_discard());
        s.mark_helper_terminate();
        assert_eq!(discard_count(&s), 1);
        assert_eq!(s.blocks[0].instrs.len(), 2);
    }

    #[test]
    fn terminate_keeps_helpers_after_demote() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![vec![
                demote(),
                quad(RegRef::new(0, RegRange::Regs(1)).into(), 0_u32.into()),
                OpNop {}.into(),
            ]],
            &[],
        );
        s.mark_helper_terminate();
        assert!(!s.blocks[0].instrs[0].flow.get_discard());
        assert!(s.blocks[0].instrs[1].flow.get_discard());
        assert_eq!(discard_count(&s), 1);
        s.mark_helper_terminate();
        assert_eq!(discard_count(&s), 1);
    }

    #[test]
    fn terminate_handles_multiple_predecessors_of_dead_block() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![
                vec![OpNop {}.into()],
                vec![quad(
                    RegRef::new(0, RegRange::Regs(1)).into(),
                    0_u32.into(),
                )],
                vec![OpNop {}.into()],
            ],
            &[(0, 1), (0, 2), (1, 2)],
        );
        s.mark_helper_terminate();
        assert!(!s.blocks[0].instrs[0].flow.get_discard());
        assert!(s.blocks[1].instrs[0].flow.get_discard());
        assert!(s.blocks[2].instrs[0].flow.get_discard());
    }

    #[test]
    fn terminate_keeps_helpers_on_loop_backedge() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![
                vec![OpNop {}.into()],
                vec![quad(
                    RegRef::new(0, RegRange::Regs(1)).into(),
                    0_u32.into(),
                )],
                vec![demote()],
                vec![OpNop {}.into()],
            ],
            &[(0, 1), (1, 2), (2, 1), (2, 3)],
        );
        s.mark_helper_terminate();
        for bi in 0..3 {
            assert!(!s.blocks[bi].instrs[0].flow.get_discard());
        }
        assert!(s.blocks[3].instrs[0].flow.get_discard());
    }

    #[test]
    fn helper_passes_respect_shader_stage() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![vec![OpNop {}.into()]],
            &[],
        );
        s.info.is_fragment = false;
        s.mark_helper_terminate();
        assert_eq!(discard_count(&s), 0);
        s.info.is_fragment = true;
        s.info.is_blend = true;
        s.mark_helper_terminate();
        assert_eq!(discard_count(&s), 0);

        let mut alloc = SSAValueAllocator::default();
        let a = alloc.alloc_ref(32);
        let b = alloc.alloc_ref(32);
        let mut s = shader(
            model.as_ref(),
            alloc,
            vec![vec![
                texture(a.clone(), 0_u32.into(), TexLodMode::Computed),
                quad(b.into(), a.into()),
            ]],
            &[],
        );
        s.info.is_fragment = false;
        s.mark_helper_skip();
        assert_eq!(skips(&s), [true]);
    }

    const G610_GPU_ID: u64 = 0xa807_0000;

    fn reg(idx: u8, n: u8) -> RegRef {
        RegRef::new(idx, RegRange::Regs(n))
    }

    fn def(idx: u8) -> Instr {
        OpCopy {
            dst: reg(idx, 1).into(),
            dst_type: DataType::I32,
            src: 0_u32.into(),
        }
        .into()
    }

    fn sample(dst: RegRef, data: RegRef, array: bool) -> Instr {
        OpTexSingle {
            dst: dst.into(),
            dst_type: DataType::A32,
            skip: true,
            dim: TexDim::Tex2D,
            projection_enable: false,
            write_mask: TexWriteMask::new(1),
            wide_indices: false,
            array_enable: array,
            texel_offset: false,
            compare_enable: false,
            lod_mode: TexLodMode::Computed,
            data: data.into(),
            handle: 0_u32.into(),
        }
        .into()
    }

    fn force_delta(s: &Shader<'_>) -> Vec<bool> {
        s.blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter_map(|i| match &i.op {
                Op::TexSingle(op) => {
                    Some(op.lod_mode == TexLodMode::ComputedForceDelta)
                }
                _ => None,
            })
            .collect()
    }

    fn discards(s: &Shader<'_>, bi: usize) -> Vec<bool> {
        s.blocks[bi]
            .instrs
            .iter()
            .map(|i| i.flow.get_discard())
            .collect()
    }

    fn coordinate_block() -> Vec<Instr> {
        vec![
            def(4),
            def(5),
            def(8),
            sample(reg(10, 1), reg(4, 2), false),
            OpNop {}.into(),
        ]
    }

    #[test]
    fn force_delta_terminates_after_coordinates() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![coordinate_block()],
            &[],
        );
        s.mark_helper_terminate();
        assert_eq!(discards(&s, 0), [false, true, false, false, false]);
        assert_eq!(force_delta(&s), [true]);
    }

    #[test]
    fn force_delta_is_v15_only() {
        let model = model_for_gpu_id(G610_GPU_ID, 0).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![coordinate_block()],
            &[],
        );
        s.mark_helper_terminate();
        assert_eq!(discards(&s, 0), [false, false, false, true, false]);
        assert_eq!(force_delta(&s), [false]);
    }

    #[test]
    fn force_delta_ignores_array_layer() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![vec![
                def(4),
                def(5),
                def(6),
                sample(reg(10, 1), reg(4, 3), true),
                OpNop {}.into(),
            ]],
            &[],
        );
        s.mark_helper_terminate();
        assert_eq!(discards(&s, 0), [false, true, false, false, false]);
        assert_eq!(force_delta(&s), [true]);
    }

    #[test]
    fn force_delta_keeps_dependent_coordinates() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![vec![
                def(4),
                def(5),
                sample(reg(6, 2), reg(4, 2), false),
                sample(reg(10, 1), reg(6, 2), false),
                OpNop {}.into(),
            ]],
            &[],
        );
        s.mark_helper_terminate();
        assert_eq!(discards(&s, 0), [false, false, true, false, false]);
        assert_eq!(force_delta(&s), [false, true]);
    }

    #[test]
    fn force_delta_across_blocks() {
        let model = model_for_gpu_id(GPU_ID, 4).unwrap();
        let mut s = shader(
            model.as_ref(),
            Default::default(),
            vec![
                vec![def(1), def(4), def(5), branch(reg(1, 1).into())],
                vec![sample(reg(10, 1), reg(4, 2), false)],
                vec![OpNop {}.into()],
            ],
            &[(0, 1), (0, 2), (1, 2)],
        );
        s.mark_helper_terminate();
        assert_eq!(discards(&s, 0), [false, false, true, false]);
        assert_eq!(discards(&s, 1), [false]);
        assert_eq!(force_delta(&s), [true]);
    }
}
