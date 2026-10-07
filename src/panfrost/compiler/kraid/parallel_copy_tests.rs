use super::*;
use crate::model::model_for_gpu_id;
use crate::ssa_value::SSAValueAllocator;

fn trace() -> Instr {
    OpRtTrace {
        dst: RegRef::new(24, RegRange::Regs(2)).into(),
        state: 0_u64.into(),
        root: 0_u64.into(),
        data: RegRef::new(0, RegRange::Regs(10)).into(),
        resume: false,
    }
    .into()
}

fn execute(instrs: &[Instr], values: &mut [u8; 128]) {
    for instr in instrs {
        match &instr.op {
            Op::Copy(op) => {
                let dst = op.dst.dst_ref.as_reg().unwrap().byte_range();
                let SrcRef::Reg(src) = op.src.src_ref else {
                    panic!()
                };
                let src = src.byte_range();
                let tmp = values[usize::from(src.start)..usize::from(src.end)]
                    .to_vec();
                values[usize::from(dst.start)..usize::from(dst.end)]
                    .copy_from_slice(&tmp);
            }
            Op::ShiftLop(op) => {
                assert!(op.logic_op == LogicOp::Xor);
                assert!(op.shift_op == ShiftOp::None);
                assert!(!op.not_result);
                let dst = op.dst.dst_ref.as_reg().unwrap().byte_range();
                let SrcRef::Reg(src) = op.src0.src_ref else {
                    panic!()
                };
                let src = src.byte_range();
                let tmp = values[usize::from(src.start)..usize::from(src.end)]
                    .to_vec();
                for (i, d) in dst.enumerate() {
                    values[usize::from(d)] ^= tmp[i % tmp.len()];
                }
            }
            _ => panic!("unexpected copy instruction: {instr}"),
        }
    }
}

#[test]
fn scratch_rotates_staging_registers() {
    let model = model_for_gpu_id(0x0f08_0000_f000_0000, 4).unwrap();
    let mut copy = ParallelCopy::new(model.as_ref(), false);
    for (dst, src) in [(7, 8), (8, 9), (9, 10), (10, 7)] {
        copy.add_copy(
            RegRef::new(dst, RegRange::Regs(1)).into(),
            RegRef::new(src, RegRange::Regs(1)).into(),
        );
    }
    copy.set_scratch_from_instr(&trace());
    let instrs: Vec<_> = copy.into_instrs::<SSAValueAllocator>(None).collect();
    assert_eq!(instrs.len(), 5);
    assert!(instrs.iter().all(|i| matches!(i.op, Op::Copy(_))));
}

#[test]
fn scratch_preserves_parallel_copy_permutations() {
    let model = model_for_gpu_id(0x0f08_0000_f000_0000, 4).unwrap();
    for width in [2_u16, 4] {
        for seed in 0..4096_u32 {
            let mut state = seed + 1;
            let mut permutation = [0, 1, 2, 3, 4, 5, 6, 7];
            for i in (1..8).rev() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                permutation.swap(i, state as usize % (i + 1));
            }
            let mut copy = ParallelCopy::new(model.as_ref(), false);
            let mut initial = [0_u8; 128];
            for (i, value) in initial.iter_mut().enumerate() {
                *value = (i as u8).wrapping_mul(17).wrapping_add(seed as u8);
            }
            let mut expected = initial;
            for (dst, &src) in permutation.iter().enumerate() {
                let d = dst as u16 * width;
                let s = src * width;
                copy.add_copy(
                    RegRef::from_byte_range(d..d + width).unwrap().into(),
                    RegRef::from_byte_range(s..s + width).unwrap().into(),
                );
                expected[usize::from(d)..usize::from(d + width)]
                    .copy_from_slice(
                        &initial[usize::from(s)..usize::from(s + width)],
                    );
            }
            copy.set_scratch_from_instr(&trace());
            let scratch = copy.scratch.clone().unwrap();
            let instrs: Vec<_> =
                copy.into_instrs::<SSAValueAllocator>(None).collect();
            let mut actual = initial;
            execute(&instrs, &mut actual);
            for i in 0..actual.len() {
                if !scratch.contains(&(i as u16)) {
                    assert_eq!(
                        actual[i], expected[i],
                        "seed {seed}, width {width}, byte {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn scratch_does_not_clobber_trace_inputs_or_copies() {
    let model = model_for_gpu_id(0x0f08_0000_f000_0000, 4).unwrap();
    let mut copy = ParallelCopy::new(model.as_ref(), false);
    let mut instr = trace();
    let Op::RtTrace(op) = &mut instr.op else {
        panic!()
    };
    op.state = RegRef::new(24, RegRange::Regs(2)).into();
    copy.set_scratch_from_instr(&instr);
    assert!(copy.scratch.is_none());
    for (dst, src) in [(24, 0), (25, 1)] {
        copy.add_copy(
            RegRef::new(dst, RegRange::Regs(1)).into(),
            RegRef::new(src, RegRange::Regs(1)).into(),
        );
    }
    copy.set_scratch_from_instr(&trace());
    assert!(copy.scratch.is_none());
}

#[test]
fn scratch_is_restricted_to_v15_register_copies() {
    let v10 = model_for_gpu_id(0xa0000000, 0).unwrap();
    let mut copy = ParallelCopy::new(v10.as_ref(), false);
    copy.set_scratch_from_instr(&trace());
    assert!(copy.scratch.is_none());
    let v15 = model_for_gpu_id(0x0f08_0000_f000_0000, 4).unwrap();
    let mut copy = ParallelCopy::new(v15.as_ref(), true);
    copy.set_scratch_from_instr(&trace());
    assert!(copy.scratch.is_none());
}
