use crate::builder::{Builder, InstrBuilder};
use crate::ir::*;
use crate::ops::*;

fn address(b: &mut impl Builder, base: Src, offset: u32) -> (Src, i16) {
    if let Ok(offset) = i16::try_from(offset) {
        return (base, offset);
    }
    b.push_op(OpMov {
        dst: RegRef::new(26, RegRange::Regs(1)).into(),
        dst_type: DataType::I32,
        src: offset.into(),
    });
    let dst = RegRef::new(24, RegRange::Regs(2));
    b.push_op(OpIAdd {
        dst: dst.into(),
        dst_type: DataType::U64,
        saturate: false,
        srcs: [
            base,
            Src::from(RegRef::new(26, RegRange::Regs(2)))
                .swizzle(Swizzle::widen_u32(0)),
        ],
    });
    (dst.into(), 0)
}

fn transfer(
    b: &mut impl Builder,
    load: bool,
    reg: RegRef,
    base: Src,
    offset: u32,
    is_tls: bool,
) {
    let (addr, offset) = address(b, base, offset);
    if reg.bytes() > 16 {
        if load {
            b.push_op(OpLdMulti { dst: reg.into(), addr, offset });
        } else {
            b.push_op(OpStMulti { data: reg.into(), addr, offset });
        }
        return;
    }
    let ty = DataType::i(reg.bytes() * 8);
    if load {
        b.push_op(OpLoad {
            dst: reg.into(),
            dst_type: ty,
            addr,
            offset,
            is_tls,
            access: MemAccess::Force,
        });
    } else {
        b.push_op(OpStore {
            src_type: ty,
            data: reg.into(),
            addr,
            offset,
            is_tls,
            is_psiz: false,
            access: MemAccess::Force,
        });
    }
}

fn frame_code(
    model: &dyn Model,
    saved: &[u8],
    restore: bool,
    multi: bool,
) -> Vec<Instr> {
    let mut b = InstrBuilder::new(model);
    let stack: Src = RegRef::new(8, RegRange::Regs(2)).into();
    let mut start = 0;
    while start < saved.len() {
        let mut end = start + 1;
        while end < saved.len()
            && saved[start] % 2 == 0
            && end - start < if multi { 16 } else { 4 }
            && saved[end] == saved[end - 1] + 1
        {
            end += 1;
        }
        let reg =
            RegRef::new(saved[start], RegRange::Regs((end - start) as u8));
        transfer(
            &mut b,
            restore,
            reg,
            stack.clone(),
            (start * 4) as u32,
            false,
        );
        start = end;
    }
    b.into_vec()
}

impl Shader<'_> {
    pub fn lower_rt_frames(&mut self) {
        let callee = self
            .blocks
            .iter()
            .any(|b| b.instrs.iter().any(|i| matches!(&i.op, Op::RtReturn(ret) if !ret.root)));
        if !callee && !self.info.rt_frame_tls && self.info.rt_frame_prefix == 0 {
            return;
        }
        assert!(self.model.arch() >= 15);
        let mut written = [false; 128];
        for instr in self.blocks.iter().flat_map(|b| &b.instrs) {
            for reg in instr.iter_reg_defs() {
                for r in reg.reg_range() {
                    if callee && r >= 20 {
                        written[usize::from(r)] = true;
                    }
                }
            }
        }
        if self.info.rt_frame_tls && callee {
            written[20..22].fill(true);
        }
        let saved: Vec<_> =
            (20..128).filter(|&r| written[r]).map(|r| r as u8).collect();
        let tls_offset = (saved.len() as u32 * 4).next_multiple_of(16);
        let tls_size = self.info.tls_size.next_multiple_of(16);
        assert!(tls_size == 0 || self.info.rt_frame_tls);
        self.info.tls_size = 0;
        self.info.rt_frame_size = self.info.rt_frame_prefix + tls_offset + tls_size;
        let mut prologue = Vec::new();
        if self.info.rt_frame_prefix != 0 {
            prologue.push(OpMov {
                dst: RegRef::new(11, RegRange::Regs(1)).into(),
                dst_type: DataType::I32,
                src: self.info.rt_frame_prefix.into(),
            }.into());
            prologue.push(OpIAdd {
                dst: RegRef::new(8, RegRange::Regs(2)).into(),
                dst_type: DataType::U64,
                saturate: false,
                srcs: [RegRef::new(8, RegRange::Regs(2)).into(),
                    Src::from(RegRef::new(10, RegRange::Regs(2)))
                        .swizzle(Swizzle::widen_u32(1))],
            }.into());
        }
        prologue.extend(frame_code(self.model, &saved, false, self.info.rt_frame_multi));
        let epilogue =
            frame_code(self.model, &saved, true, self.info.rt_frame_multi);
        if self.info.rt_frame_tls {
            for instr in self.blocks.iter_mut().flat_map(|b| &mut b.instrs) {
                for src in instr.srcs_mut() {
                    if let SrcRef::FAU(fau) = src.src_ref {
                        if fau.special == Some(SpecialFAU::ThreadLocalPointer) {
                            assert!(!fau.is_zext());
                            src.src_ref = RegRef::new(20 + (fau.idx & 1) as u8,
                                RegRange::Regs(if fau.load64 { 2 } else { 1 })).into();
                        }
                    }
                }
            }
            let mut b = InstrBuilder::new(self.model);
            let base = RegRef::new(20, RegRange::Regs(2));
            b.push_op(OpMov {
                dst: RegRef::new(20, RegRange::Regs(1)).into(),
                dst_type: DataType::I32, src: tls_offset.into(),
            });
            b.push_op(OpIAdd {
                dst: base.into(), dst_type: DataType::U64, saturate: false,
                srcs: [RegRef::new(8, RegRange::Regs(2)).into(),
                       Src::from(base).swizzle(Swizzle::widen_u32(0))],
            });
            prologue.extend(b.into_vec());
        }
        if tls_size != 0 {
            self.info.registers_used = self.info.registers_used.max(32);
        }
        let frame_size = self.info.rt_frame_size;
        self.map_instrs(|instr, _| {
            if let Op::RtFrameSize(op) = instr.op {
                return [Instr::from(OpMov {
                    dst: op.dst,
                    dst_type: DataType::I32,
                    src: frame_size.into(),
                })].into();
            }
            if matches!(instr.op, Op::RtReturn(_)) {
                let mut code = epilogue.clone();
                code.push(instr);
                code.into()
            } else {
                [instr].into()
            }
        });
        self.blocks[0].instrs.splice(..0, prologue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    const TLS: u64 = 0x100000;
    const STACK: u64 = 0x1ffffff00;

    #[test]
    fn native_frame_prefix_and_result_are_disjoint() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for prefix in [0, 16, 256, 65536] {
            let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
            cfg.add_node(0, BasicBlock {
                label: LabelAllocator::default().alloc(),
                instrs: vec![
                    OpMov {
                        dst: RegRef::new(29, RegRange::Regs(1)).into(),
                        dst_type: DataType::I32, src: 42_u32.into(),
                    }.into(),
                    OpRtFrameSize { dst: RegRef::new(0, RegRange::Regs(1)).into() }.into(),
                    OpRtReturn {
                        root: false, result_count: 1,
                        values: [RegRef::new(0, RegRange::Regs(1)).into(), 0_u32.into(), 0_u32.into()],
                        target: RegRef::new(10, RegRange::Regs(1)).into(),
                        stack: RegRef::new(8, RegRange::Regs(2)).into(),
                    }.into(),
                ],
            });
            let mut shader = Shader {
                model: model.as_ref(), ssa_alloc: Default::default(),
                phi_alloc: Default::default(), blocks: cfg.as_cfg(false),
                info: ShaderInfo { rt_frame_prefix: prefix, rt_frame_multi: true,
                                   ..Default::default() }, constant_pool: None,
            };
            shader.lower_rt_frames();
            assert_eq!(shader.info.rt_frame_size, prefix + 16);
            let mut regs = std::array::from_fn(|i| (i as u32).wrapping_mul(0x1020304));
            regs[8] = STACK as u32;
            regs[9] = (STACK >> 32) as u32;
            let original = regs;
            let mut stack = vec![0xcc; (prefix + 16) as usize];
            let code = &shader.blocks[0].instrs;
            execute(&code[..code.len() - 1], &mut regs, &mut [], &mut stack);
            assert_eq!(regs[0], prefix + 16);
            assert_eq!(regs[29], original[29]);
            assert_eq!(regs[10], original[10]);
            assert_eq!(u64::from(regs[8]) | u64::from(regs[9]) << 32, STACK + u64::from(prefix));
            assert!(stack[..prefix as usize].iter().all(|&v| v == 0xcc));
            shader.assign_message_slots();
            shader.insert_required_waits();
            shader.lower_rt_calls();
            assert!(!model.encode_shader(&shader).is_empty());
        }
    }

    #[test]
    fn native_frame_odd_register_runs_encode() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        for saved in [
            vec![29, 30, 31, 32],
            vec![21, 22, 23, 25, 26, 27, 28, 29],
            vec![123, 124, 125, 126, 127],
            (20..128).filter(|r| r % 7 != 0).collect(),
        ] {
            for (restore, multi) in [(false, false), (true, false), (false, true), (true, true)] {
                let code = frame_code(model.as_ref(), &saved, restore, multi);
                let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
                cfg.add_node(0, BasicBlock {
                    label: LabelAllocator::default().alloc(), instrs: code,
                });
                let mut shader = Shader {
                    model: model.as_ref(), ssa_alloc: Default::default(),
                    phi_alloc: Default::default(), blocks: cfg.as_cfg(false),
                    info: ShaderInfo::default(), constant_pool: None,
                };
                shader.assign_message_slots();
                shader.insert_required_waits();
                assert!(!model.encode_shader(&shader).is_empty());
            }
            let mut regs = std::array::from_fn(|i| (i as u32).wrapping_mul(0x1020304));
            regs[8] = STACK as u32;
            regs[9] = (STACK >> 32) as u32;
            let original = regs;
            let mut stack = vec![0xcc; saved.len() * 4];
            execute(&frame_code(model.as_ref(), &saved, false, true),
                    &mut regs, &mut [], &mut stack);
            for &r in &saved {
                regs[usize::from(r)] = 0xdeadbeef;
            }
            execute(&frame_code(model.as_ref(), &saved, true, true),
                    &mut regs, &mut [], &mut stack);
            assert_eq!(regs, original);
        }
    }

    fn source(src: &Src, regs: &[u32; 128]) -> u64 {
        assert!(src.src_mod.is_none());
        match &src.src_ref {
            SrcRef::Zero => 0,
            SrcRef::Imm32(x) => u64::from(x.get()),
            SrcRef::FAU(fau) => {
                assert!(fau.special == Some(SpecialFAU::ThreadLocalPointer));
                TLS
            }
            SrcRef::Reg(reg) => {
                let low = u64::from(regs[usize::from(reg.idx)]);
                if reg.bytes() == 4 || src.swizzle == Swizzle::widen_u32(0) {
                    low
                } else if src.swizzle == Swizzle::widen_u32(1) {
                    u64::from(regs[usize::from(reg.idx) + 1])
                } else {
                    assert!(src.swizzle.is_none());
                    low | u64::from(regs[usize::from(reg.idx) + 1]) << 32
                }
            }
            _ => panic!(),
        }
    }

    fn execute(
        code: &[Instr],
        regs: &mut [u32; 128],
        tls: &mut [u8],
        stack: &mut [u8],
    ) {
        for instr in code {
            match &instr.op {
                Op::Mov(op) => {
                    regs[usize::from(op.dst.dst_ref.as_reg().unwrap().idx)] =
                        source(&op.src, regs) as u32;
                }
                Op::IAdd(op) => {
                    assert!(op.dst_type == DataType::U64 && !op.saturate);
                    let value = source(&op.srcs[0], regs)
                        .wrapping_add(source(&op.srcs[1], regs));
                    let r = usize::from(op.dst.dst_ref.as_reg().unwrap().idx);
                    regs[r] = value as u32;
                    regs[r + 1] = (value >> 32) as u32;
                }
                Op::Load(_) | Op::Store(_) | Op::LdMulti(_) | Op::StMulti(_) => {
                    let (load, reg, addr, offset) = match &instr.op {
                        Op::Load(op) => (
                            true,
                            op.dst.dst_ref.as_reg().unwrap(),
                            &op.addr,
                            op.offset,
                        ),
                        Op::Store(op) => (
                            false,
                            op.data.src_ref.as_reg().unwrap(),
                            &op.addr,
                            op.offset,
                        ),
                        Op::LdMulti(op) => (true, op.dst.dst_ref.as_reg().unwrap(), &op.addr, op.offset),
                        Op::StMulti(op) => (false, op.data.src_ref.as_reg().unwrap(), &op.addr, op.offset),
                        _ => unreachable!(),
                    };
                    let addr = source(addr, regs)
                        .wrapping_add_signed(i64::from(offset));
                    let (memory, base) = if addr >= STACK {
                        (&mut *stack, STACK)
                    } else {
                        (&mut *tls, TLS)
                    };
                    let offset = usize::try_from(addr - base).unwrap();
                    for (i, r) in reg.reg_range().enumerate() {
                        let bytes =
                            &mut memory[offset + i * 4..offset + (i + 1) * 4];
                        if load {
                            regs[usize::from(r)] =
                                u32::from_le_bytes(bytes.try_into().unwrap());
                        } else {
                            bytes.copy_from_slice(
                                &regs[usize::from(r)].to_le_bytes(),
                            );
                        }
                    }
                }
                _ => panic!("unexpected frame instruction: {instr}"),
            }
        }
    }

    #[test]
    fn native_frame_restores_nested_preservation() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let saved: Vec<u8> = (20..80).chain([95, 127]).collect();
        for multi in [false, true] {
            let mut regs = std::array::from_fn(|i| (i as u32).wrapping_mul(0x1020304));
            let mut snapshots = Vec::new();
            let mut stack = vec![0xcc; 4 * 512];
            let save = frame_code(model.as_ref(), &saved, false, multi);
            let restore = frame_code(model.as_ref(), &saved, true, multi);
            for depth in 0..4 {
                let base = STACK + depth * 512;
                regs[8] = base as u32;
                regs[9] = (base >> 32) as u32;
                snapshots.push(regs);
                execute(&save, &mut regs, &mut [], &mut stack);
                for &r in &saved {
                    regs[usize::from(r)] ^= 0x12345678 + depth as u32;
                }
            }
            for depth in (0..4).rev() {
                regs[8] = snapshots[depth][8];
                regs[9] = snapshots[depth][9];
                execute(&restore, &mut regs, &mut [], &mut stack);
                assert_eq!(regs, snapshots[depth]);
            }
        }
    }
}
