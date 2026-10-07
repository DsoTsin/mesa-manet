// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use compiler::bitset::BitSet;
use rustc_hash::FxHashMap;

use crate::ir::*;
use crate::ops::MemAccess;

fn can_eliminate_instr(instr: &Instr) -> bool {
    instr.flow == FlowCtrl::NONE && instr.op.can_eliminate()
}

impl Op {
    fn trim_load_dst(&mut self, live: &BitSet<SSAValue>) {
        let (dst, ty, access) = match self {
            Op::LdCvt(op) => (&mut op.dst, &mut op.dst_type, op.access),
            Op::Load(op) if !op.is_tls => {
                (&mut op.dst, &mut op.dst_type, op.access)
            }
            _ => return,
        };
        if access == MemAccess::Force || dst.lanes != DstLanes::All {
            return;
        }
        let DstRef::SSA(vec) = &dst.dst_ref else { return };
        let Some(last) = vec.iter().rposition(|ssa| live.contains(*ssa)) else {
            return;
        };
        let words = last + 1;
        if words == vec.len() || vec[0].bits() != 32 {
            return;
        }
        let bits = (words * 32) as u8;
        *ty = if ty.comps() == 1 {
            DataType::i(bits)
        } else {
            DataType::v(bits / ty.bits(), ty.scalar_type())
        };
        dst.dst_ref = DstRef::SSA(vec[..words].try_into().unwrap());
    }
}

impl Shader<'_> {
    pub fn opt_dce(&mut self) {
        // TODO: optimize the case when the shader has no loops
        // TODO: can these be Vec<Option<&Instr>>?
        // Yeah we store references directly, I slipped some money to
        // the borrow checker, it's fine
        let mut ssa_map = FxHashMap::default();
        let mut phi_map: FxHashMap<_, Vec<_>> = FxHashMap::default();
        let mut work_queue = Vec::new();

        // O(N)
        // Create ssa-map and phi-map and gather the roots in a queue
        for block in self.blocks.iter() {
            for instr in block.instrs.iter() {
                if !can_eliminate_instr(instr) {
                    work_queue.push(instr);
                }
                if let Op::PhiSrc(phi) = &instr.op {
                    phi_map.entry(phi.phi).or_default().push(instr);
                    continue;
                }
                for dst in instr.dsts() {
                    match &dst.dst_ref {
                        DstRef::None => (),
                        DstRef::SSA(vec) => {
                            for ssa in vec {
                                ssa_map.insert(*ssa, instr);
                            }
                        }
                        DstRef::Reg(_) | DstRef::Mem(_) => {
                            panic!("opt_dce() must be run in SSA form");
                        }
                    }
                }
            }
        }

        let mut live_ssa_set = BitSet::new();
        let mut live_phi_set = BitSet::new();
        // Mark
        // Walk the graph backwards and mark sources as live
        // then push their dsts in the working set
        while let Some(instr) = work_queue.pop() {
            if let Op::PhiDst(x) = &instr.op {
                if live_phi_set.insert(x.phi) {
                    work_queue.extend(phi_map.get(&x.phi).unwrap());
                }
                continue;
            }
            for src_ssa in instr.iter_ssa_uses() {
                if live_ssa_set.insert(*src_ssa) {
                    work_queue.push(*ssa_map.get(src_ssa).unwrap());
                }
            }
        }
        drop(ssa_map);
        drop(phi_map);
        drop(work_queue);

        // Sweep
        self.map_instrs(|mut instr, _| {
            if !can_eliminate_instr(&instr) {
                return [instr].into();
            }
            let live = if let Op::PhiSrc(phi) = &instr.op {
                live_phi_set.contains(phi.phi)
            } else {
                instr.dsts().iter().any(|dst| match &dst.dst_ref {
                    DstRef::None => false,
                    DstRef::SSA(vec) => {
                        vec.iter().any(|ssa| live_ssa_set.contains(*ssa))
                    }
                    DstRef::Reg(_) | DstRef::Mem(_) => true,
                })
            };

            if live {
                instr.op.trim_load_dst(&live_ssa_set);
                [instr].into()
            } else {
                [].into()
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{OpLdCvt, OpLoad};
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};

    #[test]
    fn load_demand_preserves_live_components() {
        for ty in [DataType::V4F32, DataType::V3U32, DataType::V4F16, DataType::V3S16] {
            let mut alloc = SSAValueAllocator::default();
            let dst = alloc.alloc_ref(ty.total_bits().into());
            for mask in 1..(1 << dst.len()) {
                let mut live = BitSet::new();
                for (i, ssa) in dst.iter().enumerate() {
                    if mask & (1 << i) != 0 { live.insert(*ssa); }
                }
                let mut op: Op = OpLdCvt {
                    dst: dst.clone().into(), dst_type: ty, access: MemAccess::Const,
                    addr: 0_u64.into(), cvt: 0_u32.into(), offset: 0,
                }.into();
                op.trim_load_dst(&live);
                let Op::LdCvt(op) = op else { panic!() };
                let words = dst.iter().rposition(|ssa| live.contains(*ssa)).unwrap() + 1;
                let DstRef::SSA(trimmed) = op.dst.dst_ref else { panic!() };
                assert!(trimmed.as_slice() == &dst[..words]);
                assert_eq!(op.dst_type.total_bits().div_ceil(32) as usize, words);
                assert!(op.dst_type.scalar_type() == ty.scalar_type());
            }
        }
    }

    #[test]
    fn load_demand_respects_forced_access_and_tls() {
        let mut alloc = SSAValueAllocator::default();
        let dst = alloc.alloc_ref(128);
        let mut live = BitSet::new();
        live.insert(dst[0]);
        for (access, tls, expected) in [(MemAccess::Const, false, 32), (MemAccess::Force, false, 128), (MemAccess::Const, true, 128)] {
            let mut op: Op = OpLoad {
                dst: dst.clone().into(), dst_type: DataType::I128,
                access, is_tls: tls, addr: 0_u64.into(), offset: 0,
            }.into();
            op.trim_load_dst(&live);
            let Op::Load(op) = op else { panic!() };
            assert_eq!(op.dst_type.total_bits(), expected);
        }
    }
}
