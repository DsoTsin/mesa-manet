use super::*;
use crate::model::model_for_gpu_id;
use crate::ops::{MemAccess, OpRegIn, OpRtCall, OpStore};
use compiler::cfg::CFGBuilder;

fn call() -> Instr {
    OpRtCall { tls_args: false, dst: DstRef::None.into(),
        arg_mask: 0,
        target: RegRef::new(11, RegRange::Regs(1)).into(),
        args: std::array::from_fn(|_| 0_u32.into()),
        stack: RegRef::new(8, RegRange::Regs(2)).into(),
    }
    .into()
}

fn execute_copies(code: &[Instr], regs: &mut [u8; 512]) {
    for instr in code {
        let Op::Copy(op) = &instr.op else {
            panic!("{instr}")
        };
        assert!(op.src.src_mod.is_none() && op.src.swizzle.is_none());
        let src = op.src.src_ref.as_reg().unwrap().byte_range();
        let dst = op.dst.dst_ref.as_reg().unwrap().byte_range();
        let value = regs[usize::from(src.start)..usize::from(src.end)].to_vec();
        regs[usize::from(dst.start)..usize::from(dst.end)]
            .copy_from_slice(&value);
    }
}

#[test]
fn entry_preservation_respects_liveness_capacity_and_branch_paths() {
    let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
    for mask in [0, 1, 2, 3, 0x10000, 0x55555, 0xaaaaa, 0xfffff] {
        for (enabled, has_call, limit) in [
            (true, true, 256),
            (true, true, 128),
            (true, false, 256),
            (false, true, 256),
        ] {
            let mut ssa_alloc = SSAValueAllocator::default();
            let inputs: Vec<_> =
                (0..32).map(|_| ssa_alloc.alloc_ssa(32)).collect();
            let mut entry = Vec::new();
            for (reg, &ssa) in inputs.iter().enumerate() {
                entry.push(
                    OpRegIn {
                        dst: ssa.into(),
                        dst_type: DataType::I32,
                        reg: RegRef::new(reg as u8, RegRange::Regs(1)),
                        preload: None,
                    }
                    .into(),
                );
            }
            let store = |ssa: SSAValue| -> Instr {
                OpStore {
                    src_type: DataType::I32,
                    data: ssa.into(),
                    addr: 0_u64.into(),
                    offset: 0,
                    is_tls: false,
                    is_psiz: false,
                    access: MemAccess::None,
                }
                .into()
            };
            entry.extend(inputs.iter().copied().map(store));
            let tail = inputs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i >= 20 || mask & (1 << i) != 0)
                .map(|(_, &ssa)| store(ssa))
                .collect();
            let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> =
                CFGBuilder::new();
            let mut labels = LabelAllocator::default();
            for (i, instrs) in [
                entry,
                if has_call {
                    vec![call(), call()]
                } else {
                    vec![]
                },
                vec![],
                tail,
            ]
            .into_iter()
            .enumerate()
            {
                cfg.add_node(
                    i,
                    BasicBlock {
                        label: labels.alloc(),
                        instrs,
                    },
                );
            }
            for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
                cfg.add_edge(a, b);
            }
            let s = Shader {
                model: model.as_ref(),
                ssa_alloc,
                phi_alloc: Default::default(),
                blocks: cfg.as_cfg(false),
                info: ShaderInfo::default(),
                constant_pool: None,
            };
            let live = Liveness::for_shader(&s);
            let mut arena = Arena::new_reg(model.as_ref(), limit);
            arena.rt_call_placement = enabled;
            let affinities = AffinityMap {
                ssa_affinities: SSAValueIndexedVec::with_count(32),
                phi_webs: SSAValueIndexedVec::with_count(32),
            };
            let mut alloc =
                GlobalRegAlloc::new(model.as_ref(), &arena, &affinities);
            let mut copies = ParallelCopy::new(model.as_ref(), false);
            alloc.start_shader(&s.blocks, &live, &mut copies);
            let code: Vec<_> =
                copies.into_instrs::<SSAValueAllocator>(None).collect();
            let original = std::array::from_fn(|i| (i as u8).wrapping_mul(37));
            let mut actual = original;
            execute_copies(&code, &mut actual);
            for (i, ssa) in inputs.iter().enumerate() {
                let bytes = alloc.local.ssa_bytes(ssa);
                assert_eq!(
                    &actual[usize::from(bytes.start)..usize::from(bytes.end)],
                    &original[i * 4..i * 4 + 4]
                );
                let moved = enabled
                    && has_call
                    && limit == 256
                    && i < 20
                    && mask & (1 << i) != 0;
                if moved {
                    assert!(bytes.start >= 80);
                    assert_eq!(bytes.start % 8, (i * 4 % 8) as u16);
                } else {
                    assert_eq!(bytes.start, (i * 4) as u16);
                }
            }
            assert!(arena.bytes_used() <= limit);
        }
    }
}

#[test]
fn paired_eviction_preserves_all_values_with_fragmented_free_space() {
    let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
    for fragmented in 0..3 {
        let mut arena = Arena::new_reg(model.as_ref(), 256);
        arena.rt_call_placement = true;
        let affinities = AffinityMap {
            ssa_affinities: SSAValueIndexedVec::with_count(64),
            phi_webs: SSAValueIndexedVec::with_count(64),
        };
        let mut alloc = LocalRegAlloc::new(model.as_ref(), &arena, &affinities);
        let used: Vec<u16> = (0..64)
            .filter(|r| {
                *r < 20
                    || match fragmented {
                        0 => false,
                        1 => r % 2 == 0,
                        _ => r % 4 < 2,
                    }
            })
            .collect();
        for &reg in &used {
            alloc.assign_idx_bytes(u32::from(reg), reg * 4..reg * 4 + 4);
        }
        let mut instr = call();
        let mut copies = ParallelCopy::new(model.as_ref(), false);
        alloc.alloc_regs_instr(
            0,
            &mut instr,
            &mut copies,
            &BlockLiveness::default(),
        );
        let code: Vec<_> =
            copies.into_instrs::<SSAValueAllocator>(None).collect();
        let original = std::array::from_fn(|i| (i as u8).wrapping_mul(43));
        let mut actual = original;
        execute_copies(&code, &mut actual);
        actual[..80].fill(0xcc);
        for reg in used {
            let bytes = alloc.idx_bytes(u32::from(reg));
            assert!(bytes.start >= 80);
            assert_eq!(
                &actual[usize::from(bytes.start)..usize::from(bytes.end)],
                &original[usize::from(reg) * 4..usize::from(reg) * 4 + 4]
            );
        }
    }
}
