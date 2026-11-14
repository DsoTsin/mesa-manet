use crate::ir::*;
use crate::ops::*;

fn selected_half(swizzle: Swizzle, lane: u8) -> Option<u8> {
    if swizzle == Swizzle::H00 {
        Some(0)
    } else if swizzle == Swizzle::H11 {
        Some(1)
    } else if swizzle == Swizzle::H01 {
        Some(lane)
    } else if swizzle == Swizzle::H10 {
        Some(1 - lane)
    } else {
        None
    }
}

fn pack_source(low: &Src, high: &Src) -> Option<Src> {
    if low.src_mod != high.src_mod {
        return None;
    }
    let reference = match (&low.src_ref, &high.src_ref) {
        (SrcRef::Reg(a), SrcRef::Reg(b))
            if a.idx == b.idx
                && matches!(
                    a.range,
                    RegRange::Half0 | RegRange::Half1 | RegRange::Regs(1)
                )
                && matches!(
                    b.range,
                    RegRange::Half0 | RegRange::Half1 | RegRange::Regs(1)
                ) =>
        {
            SrcRef::Reg(RegRef::new(a.idx, RegRange::Regs(1)))
        }
        (SrcRef::FAU(_), SrcRef::FAU(_)) | (SrcRef::Zero, SrcRef::Zero)
            if low.src_ref == high.src_ref =>
        {
            low.src_ref.clone()
        }
        _ => return None,
    };
    let lo = selected_half(low.swizzle, 0)?;
    let hi = selected_half(high.swizzle, 1)?;
    let mut src = low.clone();
    src.src_ref = reference;
    src.swizzle = match (lo, hi) {
        (0, 0) => Swizzle::H00,
        (0, 1) => Swizzle::H01,
        (1, 0) => Swizzle::H10,
        (1, 1) => Swizzle::H11,
        _ => unreachable!(),
    };
    Some(src)
}

fn pack_pair(
    first: &Instr,
    second: &Instr,
    model: &dyn Model,
) -> Option<Instr> {
    if first.flow != FlowCtrl::NONE || second.flow != FlowCtrl::NONE {
        return None;
    }
    match (&first.op, &second.op) {
        (Op::FAdd(a), Op::FAdd(b))
            if a.dst_type == DataType::V2F16
                && b.dst_type == a.dst_type
                && a.round == b.round
                && a.clamp == b.clamp => {}
        (Op::Fma(a), Op::Fma(b))
            if a.dst_type == DataType::V2F16
                && b.dst_type == a.dst_type
                && a.round == b.round
                && a.clamp == b.clamp => {}
        _ => return None,
    }
    let a = &first.dsts()[0];
    let b = &second.dsts()[0];
    let ra = a.dst_ref.as_reg()?;
    let rb = b.dst_ref.as_reg()?;
    if ra.idx != rb.idx {
        return None;
    }
    let (low, high) = match (a.lanes, b.lanes) {
        (DstLanes::H0, DstLanes::H1) => (first, second),
        (DstLanes::H1, DstLanes::H0) => (second, first),
        _ => return None,
    };
    for src in first.srcs().iter().chain(second.srcs()) {
        if src.src_ref.as_reg().is_some_and(|reg| reg.idx == ra.idx) {
            return None;
        }
    }
    let mut op = low.op.clone();
    op.dsts_mut()[0] = RegRef::new(ra.idx, RegRange::Regs(1)).into();
    for (index, src) in op.srcs_mut().iter_mut().enumerate() {
        *src = pack_source(&low.srcs()[index], &high.srcs()[index])?;
    }
    if !model.op_is_supported(&op)
        || !model.op_dst_supports_lanes(&op, DstLanes::All)
        || op.srcs().iter().any(|src| {
            !model.op_src_supports_swizzle(&op, src, src.swizzle)
                || !model.op_src_supports_mod(&op, src, src.src_mod)
        })
    {
        return None;
    }
    Some(op.into())
}

impl Shader<'_> {
    pub fn opt_pack_halves(&mut self) -> bool {
        if self.model.arch() != 15 {
            return false;
        }
        let mut progress = false;
        for block in self.blocks.iter_mut() {
            let mut instrs =
                std::mem::take(&mut block.instrs).into_iter().peekable();
            while let Some(first) = instrs.next() {
                if let Some(second) = instrs.peek() {
                    if let Some(packed) = pack_pair(&first, second, self.model)
                    {
                        instrs.next();
                        block.instrs.push(packed);
                        progress = true;
                        continue;
                    }
                }
                block.instrs.push(first);
            }
        }
        progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::model_for_gpu_id;

    fn add(lane: u8) -> Instr {
        OpFAdd {
            dst: RegRef::new(
                7,
                if lane == 0 {
                    RegRange::Half0
                } else {
                    RegRange::Half1
                },
            )
            .into(),
            dst_type: DataType::V2F16,
            srcs: [
                Src::from(RegRef::new(2, RegRange::Regs(1))).swizzle(
                    if lane == 0 {
                        Swizzle::H00
                    } else {
                        Swizzle::H11
                    },
                ),
                Src::from(RegRef::new(3, RegRange::Regs(1))).swizzle(
                    if lane == 0 {
                        Swizzle::H00
                    } else {
                        Swizzle::H11
                    },
                ),
            ],
            round: FRound::NearestEven,
            clamp: FClamp::None,
        }
        .into()
    }

    #[test]
    fn joins_two_independent_half_writes() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let first = add(0);
        let second = add(1);
        let packed = pack_pair(&first, &second, model.as_ref()).unwrap();
        assert_eq!(packed.dsts()[0].lanes, DstLanes::All);
        assert!(packed.srcs().iter().all(|src| src.swizzle == Swizzle::H01));
        assert!(pack_pair(&second, &first, model.as_ref()).is_some());
    }

    #[test]
    fn rejects_aliases_and_rounding_changes() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let first = add(0);
        let mut second = add(1);
        if let Op::FAdd(op) = &mut second.op {
            op.round = FRound::TowardsZero;
        }
        assert!(pack_pair(&first, &second, model.as_ref()).is_none());
        let mut second = add(1);
        if let Op::FAdd(op) = &mut second.op {
            op.srcs[0].src_ref = RegRef::new(7, RegRange::Regs(1)).into();
        }
        assert!(pack_pair(&first, &second, model.as_ref()).is_none());
        assert!(pack_pair(&first, &first, model.as_ref()).is_none());
    }

    #[test]
    fn composes_half_swizzles_without_changing_lane_bits() {
        let swizzles = [Swizzle::H00, Swizzle::H01, Swizzle::H10, Swizzle::H11];
        let values = [0, 0x80000000, 0x7c007e00, 0x0001ffff, 0x12345678];
        for low_swizzle in swizzles {
            for high_swizzle in swizzles {
                let low = Src::from(RegRef::new(2, RegRange::Regs(1)))
                    .swizzle(low_swizzle);
                let high = Src::from(RegRef::new(2, RegRange::Regs(1)))
                    .swizzle(high_swizzle);
                let packed = pack_source(&low, &high).unwrap();
                for value in values {
                    let expected = (low_swizzle.fold_u32(value).unwrap()
                        & 0xffff)
                        | (high_swizzle.fold_u32(value).unwrap() & 0xffff0000);
                    assert_eq!(
                        packed.swizzle.fold_u32(value).unwrap(),
                        expected
                    );
                }
            }
        }
        let low =
            Src::from(RegRef::new(2, RegRange::Half0)).swizzle(Swizzle::H00);
        let high =
            Src::from(RegRef::new(2, RegRange::Half1)).swizzle(Swizzle::H11);
        let packed = pack_source(&low, &high).unwrap();
        assert_eq!(packed.src_ref.as_reg().unwrap().range.bytes(), 4);
        assert!(packed.swizzle == Swizzle::H01);
    }
}
