use crate::ir::*;
use crate::ops::*;

impl Shader<'_> {
    pub fn lower_rt_calls(&mut self) {
        let model = self.model;
        self.map_instrs(|instr, _| {
            let (target, mode) = match instr.op {
                Op::RtCall(op) => (op.target, JumpStackMode::Call),
                Op::RtReturn(op) => (op.target, JumpStackMode::Return),
                _ => return [instr].into(),
            };
            assert!(model.arch() >= 15);
            let pc = model.fau().special(SpecialFAU::Pc).unwrap();
            let mut jump = Instr::from(OpJumpEx {
                not: true,
                cond: 0_u32.into(),
                target_lo: target,
                target_hi: pc.word(1).into(),
                combine_op: BranchCombineOp::None,
                stack_mode: mode,
                diverge: false,
            });
            jump.flow = instr.flow;
            jump.flow.take_end_shader();
            jump.flow.take_reconverge();
            let mut result = Vec::new();
            if mode == JumpStackMode::Call {
                assert!(!instr.flow.get_end_shader());
                result.push(Instr::from(OpIAdd {
                    dst: RegRef::new(10, RegRange::Regs(1)).into(),
                    dst_type: DataType::I32,
                    saturate: false,
                    srcs: [pc.word(0).into(), 8_u32.into()],
                }));
            }
            result.push(jump);
            if instr.flow.get_reconverge() {
                assert!(mode == JumpStackMode::Call);
                let mut nop = Instr::from(OpNop {});
                nop.flow.set_reconverge();
                result.push(nop);
            }
            result.into()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{Builder, SSAInstrBuilder};
    use crate::liveness::PhysicalLiveness;
    use crate::model::model_for_gpu_id;
    use crate::ssa_value::AllocSSA;
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn shader(model: &dyn Model, instrs: Vec<Instr>) -> Shader<'_> {
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> =
            CFGBuilder::new();
        cfg.add_node(
            0,
            BasicBlock {
                label: LabelAllocator::default().alloc(),
                instrs,
            },
        );
        Shader {
            model,
            ssa_alloc: Default::default(),
            phi_alloc: Default::default(),
            blocks: cfg.as_cfg(false),
            info: ShaderInfo::default(),
            constant_pool: None,
        }
    }

    fn call() -> Instr {
        let mut target: Src = RegRef::new(11, RegRange::Regs(1)).into();
        target.last_use = true;
        OpRtCall { tls_args: false, dst: DstRef::None.into(),
            arg_mask: 0xffff,
            target,
            args: std::array::from_fn(|i| {
                RegRef::new((if i < 8 { i } else { i + 4 }) as u8,
                            RegRange::Regs(1)).into()
            }),
            stack: RegRef::new(8, RegRange::Regs(2)).into(),
        }
        .into()
    }

    #[test]
    fn low_pressure_v15_shaders_include_fixed_registers_in_arena() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for reg in [0, 1, 3, 4, 5, 6, 12, 15, 16, 17, 18, 19, 31, 63, 127] {
            let mut s = shader(model.as_ref(), Vec::new());
            let value = s.ssa_alloc.alloc_ssa(32);
            s.blocks[0].instrs = vec![
                OpRegIn { dst: value.into(), dst_type: DataType::I32,
                    reg: RegRef::new(reg, RegRange::Regs(1)), preload: None,
                }.into(),
                OpStore { data: value.into(), src_type: DataType::I32,
                    addr: model.fau().special(SpecialFAU::ThreadLocalPointer).unwrap().into(),
                    offset: 0, is_tls: false, is_psiz: false,
                    access: MemAccess::Force,
                }.into(),
            ];
            s.assign_registers(0, false, false, false);
            s.lower_copy();
            assert!(u32::from(s.info.registers_used) >= u32::from(reg) + 1);
            s.assign_message_slots();
            assert!(!model.encode_shader(&s).is_empty());
        }
    }

    #[test]
    fn native_tls_base_is_independent_of_spill_base() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let tls = model.fau().special(SpecialFAU::ThreadLocalPointer).unwrap();
        let raw_tls = model.fau().special(SpecialFAU::RtThreadLocalPointer).unwrap();
        assert!((tls.page, tls.idx) == (raw_tls.page, raw_tls.idx));
        let mut s = shader(model.as_ref(), vec![
            OpMov {
                dst: RegRef::new(28, RegRange::Regs(1)).into(),
                dst_type: DataType::I32, src: raw_tls.word(0).into(),
            }.into(),
            OpMov {
                dst: RegRef::new(29, RegRange::Regs(1)).into(),
                dst_type: DataType::I32, src: raw_tls.word(1).into(),
            }.into(),
            OpStore {
                src_type: DataType::I32,
                data: RegRef::new(30, RegRange::Regs(1)).into(),
                addr: tls.into(), offset: 0, is_tls: true, is_psiz: false,
                access: MemAccess::Force,
            }.into(),
            OpRtReturn {
                root: false, result_count: 0,
                values: std::array::from_fn(|_| 0_u32.into()),
                target: RegRef::new(10, RegRange::Regs(1)).into(),
                stack: RegRef::new(8, RegRange::Regs(2)).into(),
            }.into(),
        ]);
        s.info.rt_frame_tls = true;
        s.info.rt_frame_prefix = 32;
        s.info.tls_size = 64;
        s.lower_rt_frames();
        assert!(s.blocks[0].instrs.iter().any(|i| matches!(&i.op,
            Op::Store(op) if op.is_tls && op.addr.src_ref.as_reg().unwrap().idx == 20)));
        s.assign_message_slots();
        s.lower_rt_calls();
        let raw = s.blocks[0].instrs.iter().filter(|i| matches!(&i.op,
            Op::Mov(op) if matches!(op.src.src_ref, SrcRef::FAU(f)
                if f.special == Some(SpecialFAU::RtThreadLocalPointer)))).count();
        assert_eq!(raw, 2);
        assert!(!model.encode_shader(&s).is_empty());
    }

    #[test]
    fn native_call_matches_vendor_adr_and_call() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut s = shader(model.as_ref(), vec![call()]);
        s.lower_rt_calls();
        for instr in &s.blocks[0].instrs {
            assert!(crate::stats::exec_class(model.as_ref(), &instr.op).is_some());
        }
        let code = model.encode_shader(&s);
        let words: Vec<u64> = code
            .chunks_exact(2)
            .map(|w| u64::from(w[0]) | u64::from(w[1]) << 32)
            .collect();
        assert_eq!(words[0], 0xc06d0a00000008fe);
        assert_eq!(words[1] & 0x03ffffffffffffff, 0x02ad001200ff8bc0);
        assert_eq!(s.blocks[0].instrs[1].flow.wait & 7, 0);
    }

    #[test]
    fn native_arguments_fit_preload_set_and_clobber_range() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut set = PreloadRegSet::default();
        let mut used = [false; 20];
        assert!(!PreloadReg::GlobalId0.is_rt_arg());
        assert!(!PreloadReg::BlendReturnAddr.is_rt_arg());
        for arg in [
            PreloadReg::RtFrame, PreloadReg::RtPayload, PreloadReg::RtRoot,
            PreloadReg::RtRay, PreloadReg::RtStack, PreloadReg::RtReturn,
            PreloadReg::RtRecord, PreloadReg::RtShaderRecord,
            PreloadReg::RtTmax, PreloadReg::RtHitKind, PreloadReg::RtDepth,
            PreloadReg::RtFrameLayout,
        ] {
            assert!(arg.is_rt_arg());
            set.insert(arg);
            assert!(set.contains(arg));
            for r in model.preload_reg(arg).unwrap().reg_range() {
                assert!(!used[usize::from(r)]);
                used[usize::from(r)] = true;
            }
        }
        assert!(!used[11] && used[19]);
    }

    #[test]
    fn tls_arguments_match_dense_preloads() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut instr = call();
        let Op::RtCall(op) = &mut instr.op else { panic!() };
        op.tls_args = true;
        let args = [
            (PreloadReg::RtFrameOffset, 0),
            (PreloadReg::RtPayloadOffset, 2),
            (PreloadReg::RtRootOffset, 4),
            (PreloadReg::RtRayOffset, 6),
            (PreloadReg::RtLaunch0, 1),
            (PreloadReg::RtLaunch1, 3),
            (PreloadReg::RtLaunch2, 5),
        ];
        let mut used = [false; 20];
        for (physical, (preload, arg)) in args.into_iter().enumerate() {
            let Op::RtCall(op) = &instr.op else { panic!() };
            let source = model.op_fixed_src_reg(&instr.op, &op.args[arg]).unwrap();
            let incoming = model.preload_reg(preload).unwrap();
            assert_eq!(source.idx, physical as u8);
            assert_eq!(incoming.idx, source.idx);
            assert_eq!(incoming.bytes(), 4);
            assert!(preload.is_rt_arg());
            assert!(!used[usize::from(source.idx)]);
            used[usize::from(source.idx)] = true;
        }
    }

    #[test]
    fn call_returns_before_pending_reconvergence() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut c = call();
        c.flow.set_reconverge();
        let mut s = shader(model.as_ref(), vec![c]);
        s.lower_rt_calls();
        assert_eq!(s.blocks[0].instrs.len(), 3);
        assert!(!s.blocks[0].instrs[1].flow.get_reconverge());
        assert!(s.blocks[0].instrs[2].flow.get_reconverge());
    }

    #[test]
    fn only_root_returns_skip_caller_preservation() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for root in [false, true] {
            let mut s = shader(model.as_ref(), vec![
                OpMov {
                    dst: RegRef::new(20, RegRange::Regs(1)).into(),
                    dst_type: DataType::I32, src: 7_u32.into(),
                }.into(),
                OpRtReturn { result_count: 0, values: std::array::from_fn(|_| 0_u32.into()),
                    root,
                    target: RegRef::new(10, RegRange::Regs(1)).into(),
                    stack: RegRef::new(8, RegRange::Regs(2)).into(),
                }.into(),
            ]);
            s.lower_rt_frames();
            assert_eq!(s.info.rt_frame_size, if root { 0 } else { 16 });
            assert_eq!(s.blocks[0].instrs.len() == 2, root);
        }
    }

    #[test]
    fn native_return_drains_messages_without_ending_shader() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let mut ret: Instr = OpRtReturn { result_count: 0, values: std::array::from_fn(|_| 0_u32.into()),
            root: false,
            target: RegRef::new(10, RegRange::Regs(1)).into(),
            stack: RegRef::new(8, RegRange::Regs(2)).into(),
        }
        .into();
        ret.flow.set_end_shader();
        let mut s = shader(model.as_ref(), vec![ret]);
        s.lower_rt_calls();
        let i = &s.blocks[0].instrs[0];
        assert!(!i.flow.get_end_shader());
        assert_eq!(i.flow.wait & 7, 0);
        let Op::JumpEx(op) = &i.op else { panic!() };
        assert!(op.stack_mode == JumpStackMode::Return);
        let code = model.encode_shader(&s);
        let word = u64::from(code[0]) | u64::from(code[1]) << 32;
        assert_eq!(word & 0x03ffffffffffffff, 0x02ad001000ff0ac0);
    }

    #[test]
    fn rt_transfers_wait_for_messages_before_clearing_end_shader() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let addr = || Src::from(RegRef::new(8, RegRange::Regs(2)));
        for returning in [false, true] {
            for kind in 0..5 {
                let message = match kind {
                    0 => Instr::from(OpLoad {
                        dst: RegRef::new(28, RegRange::Regs(1)).into(),
                        dst_type: DataType::I32, is_tls: false,
                        access: MemAccess::Force, addr: addr(), offset: 0,
                    }),
                    1 => Instr::from(OpStore {
                        data: RegRef::new(28, RegRange::Regs(1)).into(),
                        src_type: DataType::I32, is_tls: false, is_psiz: false,
                        access: MemAccess::Force, addr: addr(), offset: 0,
                    }),
                    2 => Instr::from(OpLdMulti {
                        dst: RegRef::new(28, RegRange::Regs(8)).into(),
                        addr: addr(), offset: 0,
                    }),
                    3 => Instr::from(OpStMulti {
                        data: RegRef::new(28, RegRange::Regs(8)).into(),
                        addr: addr(), offset: 0,
                    }),
                    _ => Instr::from(OpRtTrace {
                        dst: RegRef::new(28, RegRange::Regs(2)).into(),
                        state: addr(), root: addr(),
                        data: RegRef::new(32, RegRange::Regs(16)).into(),
                        resume: false,
                    }),
                };
                let mut transfer = if returning {
                    Instr::from(OpRtReturn {
                        root: false, result_count: 0,
                        values: std::array::from_fn(|_| 0_u32.into()),
                        target: RegRef::new(10, RegRange::Regs(1)).into(),
                        stack: addr(),
                    })
                } else {
                    call()
                };
                if returning {
                    transfer.flow.set_end_shader();
                }
                let mut s = shader(model.as_ref(), vec![message, transfer]);
                s.assign_message_slots();
                assert_eq!(s.blocks[0].instrs[0].flow.wait & 7, 1);
                s.lower_rt_calls();
                let last = s.blocks[0].instrs.last().unwrap();
                assert!(matches!(&last.op, Op::JumpEx(_)));
                assert!(!last.flow.get_end_shader());
                assert_eq!(last.flow.wait & 7, 0);
                assert!(!model.encode_shader(&s).is_empty());
            }
        }
    }

    #[test]
    fn native_call_register_allocation_preserves_live_values() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for result_count in [0, 1, 3] {
        for (count, mask) in [
            (1, 0xffff), (12, 0xffff), (32, 0xffff), (64, 0xffff),
            (104, 0xffff), (120, 0xffff), (120, 0), (120, 1),
            (120, 0x8000), (120, 0x55aa),
        ] {
            let mut s = shader(model.as_ref(), Vec::new());
            let mut b = SSAInstrBuilder::new(model.as_ref(), &mut s.ssa_alloc);
            let addr = b.alloc_ref(64);
            b.push_op(OpRegIn {
                dst: addr.clone().into(),
                dst_type: DataType::I64,
                reg: RegRef::new(0, RegRange::Regs(2)),
                preload: None,
            });
            let mut values = Vec::new();
            for i in 0..count {
                let v = b.alloc_ssa(32);
                b.push_op(OpLoad {
                    dst: v.into(),
                    dst_type: DataType::I32,
                    addr: addr.clone().into(),
                    offset: i * 4,
                    is_tls: false,
                    access: MemAccess::None,
                });
                values.push(v);
            }
            let result = (result_count != 0).then(|| b.alloc_ref(result_count * 32));
            b.push_op(OpRtCall {
                tls_args: false,
                dst: result.clone().map_or(DstRef::None, DstRef::SSA).into(),
                arg_mask: mask,
                target: values[0].into(),
                args: std::array::from_fn(|i| {
                    if mask & (1 << i) != 0 {
                        values[i % values.len()].into()
                    } else {
                        0_u32.into()
                    }
                }),
                stack: addr.clone().into(),
            });
            if let Some(result) = result {
                for (i, v) in result.iter().enumerate() {
                    b.push_op(OpStore {
                        src_type: DataType::I32,
                        data: (*v).into(),
                        addr: addr.clone().into(),
                        offset: 1024 + (i * 4) as i16,
                        is_tls: false,
                        is_psiz: false,
                        access: MemAccess::None,
                    });
                }
            }
            for (i, v) in values.into_iter().enumerate() {
                b.push_op(OpStore {
                    src_type: DataType::I32,
                    data: v.into(),
                    addr: addr.clone().into(),
                    offset: (i * 4).try_into().unwrap(),
                    is_tls: false,
                    is_psiz: false,
                    access: MemAccess::None,
                });
            }
            s.blocks[0].instrs = b.into_vec();
            s.widen_alu_ops();
            s.legalize();
            s.assign_registers(0, false, false, false);
            s.lower_copy();
            let live = PhysicalLiveness::for_shader(&s);
            let mut after = live.live_out[0];
            let mut calls = 0;
            for (ip, instr) in s.blocks[0].instrs.iter().enumerate().rev() {
                if let Op::RtCall(op) = &instr.op {
                    for (i, src) in op.args.iter().enumerate() {
                        if mask & (1 << i) != 0 {
                            assert_eq!(src.src_ref.as_reg().unwrap().idx,
                                       (if i < 8 { i } else { i + 4 }) as u8);
                        } else {
                            assert!(src.src_ref == SrcRef::Zero);
                        }
                    }
                    assert_eq!(op.dst.dst_ref.bytes_written(), (result_count * 4) as u8);
                    if result_count != 0 {
                        assert_eq!(op.dst.dst_ref.as_reg().unwrap().idx, 0);
                    }
                    for byte in (result_count * 4) as u16..80 {
                        assert!(
                            !after.contains(byte),
                            "live value in clobbered byte {byte}"
                        );
                    }
                    for src in instr.srcs() {
                        assert!(
                            src.src_ref.as_reg()
                                == model
                                    .op_fixed_src_reg(&instr.op, src)
                                    .as_ref()
                        );
                    }
                    calls += 1;
                }
                after = live.access[0][ip].live_before(after);
            }
            assert_eq!(calls, 1);
        }
        }
    }

    #[test]
    fn native_spills_have_a_reserved_frame_base() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for root in [false, true] {
            let mut s = shader(model.as_ref(), Vec::new());
            s.info.tls_size = 64;
            let mut b = SSAInstrBuilder::new(model.as_ref(), &mut s.ssa_alloc);
            let addr = b.alloc_ref(64);
            b.push_op(OpRegIn {
                dst: addr.clone().into(), dst_type: DataType::I64,
                reg: RegRef::new(8, RegRange::Regs(2)), preload: None,
            });
            let mut values = Vec::new();
            for i in 0..120 {
                let v = b.alloc_ssa(32);
                b.push_op(OpLoad {
                    dst: v.into(), dst_type: DataType::I32,
                    addr: addr.clone().into(), offset: i * 4,
                    is_tls: false, access: MemAccess::None,
                });
                values.push(v);
            }
            b.push_op(OpRtCall { tls_args: false, dst: DstRef::None.into(),
                arg_mask: 0, target: 0_u32.into(),
                args: std::array::from_fn(|_| 0_u32.into()),
                stack: addr.clone().into(),
            });
            for (i, v) in values.into_iter().enumerate() {
                b.push_op(OpStore {
                    src_type: DataType::I32, data: v.into(),
                    addr: addr.clone().into(), offset: (i * 4) as i16,
                    is_tls: false, is_psiz: false, access: MemAccess::None,
                });
            }
            b.push_op(OpRtReturn { result_count: 0, values: std::array::from_fn(|_| 0_u32.into()), root, target: 0_u32.into(), stack: addr.into() });
            s.blocks[0].instrs = b.into_vec();
            s.legalize();
            s.assign_registers(0, false, false, false);
            assert!(s.info.rt_frame_tls);
            for instr in &s.blocks[0].instrs {
                for reg in instr.iter_reg_defs().chain(instr.iter_reg_uses()) {
                    assert!(reg.reg_range().all(|r| r != 20 && r != 21));
                }
            }
            s.lower_copy();
            s.lower_rt_frames();
            assert_eq!(s.info.tls_size, 0);
            assert!(s.info.rt_frame_size >= 64);
            for instr in &s.blocks[0].instrs {
                assert!(!instr.srcs().iter().any(|src| matches!(src.src_ref,
                    SrcRef::FAU(fau) if fau.special == Some(SpecialFAU::ThreadLocalPointer))));
            }
            s.assign_message_slots();
            s.insert_required_waits();
            s.lower_rt_calls();
            assert!(!model.encode_shader(&s).is_empty());
        }
    }
}
