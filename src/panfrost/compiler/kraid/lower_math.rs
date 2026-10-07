use crate::builder::*;
use crate::ir::*;
use crate::ops::*;
use std::f32::consts::PI;

pub(crate) trait MathBuilder: SSABuilder {
    fn expand_exp(&mut self, dst: Dst, arg: Src, log2_base: Src) {
        let scale = self.alloc_ssa(32);

        self.push_op(OpFmaRScale {
            dst: scale.into(),
            round: FRound::NearestEven,
            clamp: FClamp::None,
            srcs: [arg.clone(), log2_base, Src::fneg_zero(32)],
            scale: 24_u32.into(),
        });

        let scale_fixp = self.alloc_ssa(32);
        self.push_op(OpF32ToI32 {
            dst: scale_fixp.into(),
            dst_type: DataType::S32,
            src: scale.into(),
            round: FRound::NearestEven,
        });

        self.push_op(OpFExp32 {
            dst,
            expx: scale_fixp.into(),
            expf: scale.into(),
        });
    }

    fn expand_log(&mut self, dst: Dst, arg: Src) {
        let frexp = self.alloc_ssa(32);
        self.push_op(OpFrexpE {
            dst: frexp.into(),
            src_type: DataType::F32,
            src: arg.clone(),
            mode: FrexpMode::Log,
            neg_result: false,
        });
        let frexpi = self.alloc_ssa(32);
        self.push_op(OpIToF32 {
            dst: frexpi.into(),
            src_type: DataType::S32,
            src: frexp.into(),
            round: FRound::NearestEven,
        });

        let flogd = self.alloc_ssa(32);
        self.push_op(OpFLogD {
            dst: flogd.into(),
            src: arg.clone(),
        });
        let lscale = self.alloc_ssa(32);
        self.push_op(OpFAddLScale {
            dst: lscale.into(),
            round: FRound::NearestEven,
            clamp: FClamp::None,
            srcs: [Src::from(-1.0), arg],
        });

        self.fma_32_to(dst, flogd.into(), lscale.into(), frexpi.into());
    }

    fn expand_sincos(&mut self, dst: Dst, src: Src, is_cos: bool) {
        let two_over_pi: f32 = 2.0 / PI;
        let mpi_over_two: f32 = -PI / 2.0;
        let sincos_bias: f32 = 786432.0;

        let x_u6 =
            self.fma_32(src.clone(), two_over_pi.into(), sincos_bias.into());
        let e_part = self.fadd_32(x_u6.into(), Src::from(sincos_bias).fneg());
        let e = self.fma_32(e_part.into(), mpi_over_two.into(), src);

        let sinx = self.alloc_ssa(32);
        let cosx = self.alloc_ssa(32);
        self.push_op(OpFSinTable {
            dst: sinx.into(),
            src: x_u6.into(),
            offset: false,
        });
        self.push_op(OpFCosTable {
            dst: cosx.into(),
            src: x_u6.into(),
            offset: false,
        });

        let sinx = Src::from(sinx);
        let cosx = Src::from(cosx);

        let f = if is_cos { cosx.clone() } else { sinx.clone() };
        let fd = if is_cos { sinx.fneg() } else { cosx };
        let mfdd = f.clone();

        let e2_over2 = self.alloc_ssa(32);
        self.push_op(OpFmaRScale {
            dst: e2_over2.into(),
            round: FRound::NearestEven,
            clamp: FClamp::None,
            srcs: [e.into(), e.into(), Src::from(-0.0)],
            scale: Src::from(-1i32 as u32),
        });

        let quadratic =
            self.fma_32(Src::from(e2_over2).fneg(), mfdd, Src::from(-0.0));

        let partial = self.alloc_ssa(32);
        self.push_op(OpFma {
            dst: partial.into(),
            dst_type: DataType::F32,
            round: FRound::NearestEven,
            clamp: FClamp::NegOneToOne,
            srcs: [e.into(), fd, quadratic.into()],
        });

        self.fadd_32_to(dst, partial.into(), f);
    }
    fn lower_math_op(&mut self, op: OpFMath) {
        let [src, scale] = op.srcs;
        let native: Op = OpFNative { dst: op.dst.clone(), kind: op.kind, src: src.clone() }.into();
        if !matches!(op.kind, FMathKind::Sin | FMathKind::Cos) && self.model().op_is_supported(&native) {
            let src = if op.kind == FMathKind::Exp2 && scale.resolve_imm(DataType::F32) != Some(1.0f32.to_bits() as u64) {
                self.fma_32(src, scale, Src::fneg_zero(32)).into()
            } else { src };
            self.push_op(OpFNative { dst: op.dst, kind: op.kind, src });
            return;
        }
        match op.kind {
            FMathKind::Exp2 => self.expand_exp(op.dst, src, scale),
            FMathKind::Log2 => self.expand_log(op.dst, src),
            FMathKind::Sqrt => {
                let t = self.alloc_ssa(32);
                self.push_op(OpFRsq { dst: t.into(), dst_type: DataType::F32, src });
                self.push_op(OpFRcp { dst: op.dst, dst_type: DataType::F32, src: t.into() });
            }
            FMathKind::Sin | FMathKind::Cos => {
                let native: Op = OpFSinCos {
                    dst: op.dst.clone(), remainder: src.clone(), quadrant: 0_u32.into(),
                    is_cos: op.kind == FMathKind::Cos,
                }.into();
                if self.model().op_is_supported(&native) {
                    let bias = 12582912.0f32;
                    let quadrant = self.fma_32(src.clone(), (2.0f32 / PI).into(), bias.into());
                    let reduced = self.fadd_32(quadrant.into(), (-bias).into());
                    let remainder = self.fma_32(reduced.into(), (-PI / 2.0).into(), src);
                    self.push_op(OpFSinCos { dst: op.dst, remainder: remainder.into(), quadrant: quadrant.into(), is_cos: op.kind == FMathKind::Cos });
                } else {
                    self.expand_sincos(op.dst, src, op.kind == FMathKind::Cos);
                }
            }
        }
    }
}

impl<T: SSABuilder> MathBuilder for T {}

impl Shader<'_> {
    pub fn lower_math(&mut self) {
        let model = self.model;
        self.map_instrs(|instr, alloc| {
            let Op::FMath(op) = &instr.op else { return [instr].into(); };
            assert!(instr.flow == FlowCtrl::NONE);
            let mut b = SSAInstrBuilder::new(model, alloc);
            b.lower_math_op((**op).clone());
            b.into_mapped()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;
    use crate::ssa_value::{AllocSSA, SSAValueAllocator};

    #[test]
    fn target_math_selection() {
        for (gpu, variant, native) in [(0xa0000000, 0, false), (0xf080000f0000000, 4, true)] {
            let model = model_for_gpu_id(gpu, variant).unwrap();
            for kind in [FMathKind::Exp2, FMathKind::Log2, FMathKind::Sqrt, FMathKind::Sin, FMathKind::Cos] {
                let mut alloc = SSAValueAllocator::default();
                let input = alloc.alloc_ssa(32);
                let dst = alloc.alloc_ssa(32);
                let mut b = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
                b.lower_math_op(OpFMath { dst: dst.into(), kind, srcs: [input.into(), 1_f32.into()] });
                let instrs = b.into_vec();
                assert!(instrs.iter().all(|i| model.op_is_supported(&i.op)));
                let expected = match (native, kind) {
                    (true, FMathKind::Sin | FMathKind::Cos) => 4,
                    (true, _) => 1,
                    (false, FMathKind::Exp2) => 3,
                    (false, FMathKind::Log2) => 5,
                    (false, FMathKind::Sqrt) => 2,
                    (false, _) => 9,
                };
                assert_eq!(instrs.len(), expected);
                let DstRef::SSA(result) = &instrs.last().unwrap().dsts()[0].dst_ref else { panic!() };
                assert!(result.as_slice() == &[dst]);
            }
        }
    }

    #[test]
    fn scaled_exp_keeps_multiplier() {
        let model = model_for_gpu_id(0xf080000f0000000, 4).unwrap();
        for scale in [0.0_f32, -0.0, 1.0, -1.0, f32::INFINITY, f32::NAN, std::f32::consts::LOG2_E] {
            let mut alloc = SSAValueAllocator::default();
            let input = alloc.alloc_ssa(32);
            let dst = alloc.alloc_ssa(32);
            let mut b = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
            b.fexp_32_to(dst.into(), input.into(), scale.into());
            let semantic = b.into_vec();
            let Op::FMath(op) = &semantic[0].op else { panic!() };
            let mut b = SSAInstrBuilder::new(model.as_ref(), &mut alloc);
            b.lower_math_op((**op).clone());
            let instrs = b.into_vec();
            assert_eq!(instrs.len(), if scale == 1.0 { 1 } else { 2 });
            if scale != 1.0 {
                let Op::Fma(op) = &instrs[0].op else { panic!() };
                assert_eq!(op.srcs[1].resolve_imm(DataType::F32), Some(scale.to_bits() as u64));
                assert_eq!(op.srcs[2].resolve_imm(DataType::F32), Some(0x80000000));
            }
        }
    }

    #[test]
    fn sincos_reduction_preserves_quadrants() {
        for i in -32768..=32768 {
            let x = i as f32 * (PI / 32768.0);
            let q = x.mul_add(2.0 / PI, 12582912.0);
            let r = (q - 12582912.0).mul_add(-PI / 2.0, x);
            assert!(r.abs() <= PI / 4.0 + 0.000001);
            let (sin, cos) = match q.to_bits() & 3 {
                0 => (r.sin(), r.cos()),
                1 => (r.cos(), -r.sin()),
                2 => (-r.sin(), -r.cos()),
                _ => (-r.cos(), r.sin()),
            };
            assert!((sin - x.sin()).abs() < 0.000001);
            assert!((cos - x.cos()).abs() < 0.000001);
        }
    }
}
