// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::liveness::{LiveSet, Liveness};
use rustc_hash::FxHashMap;

const MAX_GAP: usize = 64;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum CopyKey {
    Imm(u64, u8),
    FAU(FAUPage, u16, bool, u8),
}

fn copy_key(instr: &Instr) -> Option<(CopyKey, SSARef)> {
    if instr.flow != FlowCtrl::NONE {
        return None;
    }
    let Op::Copy(op) = &instr.op else {
        return None;
    };
    if op.dst.lanes != DstLanes::All
        || !op.src.swizzle.is_none()
        || op.src.src_mod != SrcMod::None
    {
        return None;
    }
    let DstRef::SSA(vec) = &op.dst.dst_ref else {
        return None;
    };
    let comps = vec.comps();
    if comps > 2 || vec.iter().any(|ssa| ssa.bits() != 32 || ssa.is_mem()) {
        return None;
    }
    let key = match &op.src.src_ref {
        SrcRef::Zero => CopyKey::Imm(0, comps),
        SrcRef::Imm32(v) if comps == 1 => CopyKey::Imm(v.get().into(), comps),
        SrcRef::Imm64(v) if comps == 2 => CopyKey::Imm(v.get(), comps),
        SrcRef::FAU(fau)
            if matches!(fau.page, FAUPage::User | FAUPage::SmallConst)
                && fau.load64 == (comps == 2) =>
        {
            CopyKey::FAU(fau.page, fau.idx, fau.load64, comps)
        }
        _ => return None,
    };
    Some((key, vec.clone()))
}

#[derive(Clone, Copy)]
struct Use {
    block: usize,
    ip: usize,
    src_idx: usize,
}

fn plain_uses(
    model: &dyn Model,
    block: usize,
    instrs: &[Instr],
    def_ip: usize,
    vec: &SSARef,
    uses: &FxHashMap<SSAValue, Vec<Use>>,
) -> Option<(usize, Vec<usize>)> {
    let mut last = def_ip;
    let mut ips = Vec::new();
    for ssa in vec.iter() {
        for u in uses.get(ssa).map(|v| v.as_slice()).unwrap_or(&[]) {
            if u.block != block || u.ip <= def_ip {
                return None;
            }
            let instr = &instrs[u.ip];
            if matches!(&instr.op, Op::PhiSrc(_) | Op::PhiDst(_) | Op::RegOut(_))
            {
                return None;
            }
            let src = &instr.srcs()[u.src_idx];
            let SrcRef::SSA(src_vec) = &src.src_ref else {
                return None;
            };
            if src_vec != vec
                || model.op_fixed_src_reg(&instr.op, src).is_some()
            {
                return None;
            }
            last = last.max(u.ip);
            ips.push(u.ip);
        }
    }
    Some((last, ips))
}

fn rename_is_legal(instr: &Instr, from: &SSARef, to: &SSARef) -> bool {
    instr.srcs().iter().all(|src| match &src.src_ref {
        SrcRef::SSA(vec) => {
            vec == from
                || vec == to
                || !vec.iter().any(|ssa| to.iter().any(|t| t == ssa))
        }
        _ => true,
    })
}

const FULL_OCCUPANCY_BYTES: u32 = 32 * 4;

fn block_pressure(s: &Shader, live: &Liveness, bi: usize) -> Vec<u32> {
    let bl = live.block(bi);
    let mut set = LiveSet::from_iter(
        bl.live_in_set().iter().map(|idx| s.ssa_alloc.lookup_by_idx(idx)),
    );
    s.blocks[bi]
        .instrs
        .iter()
        .enumerate()
        .map(|(ip, instr)| set.insert_instr_top_down(s.model, ip, instr, bl).reg)
        .collect()
}

impl Shader<'_> {
    pub fn opt_share_copies(&mut self) -> bool {
        let model = self.model;
        let live = Liveness::for_shader(self);
        let limit = if live.max_live_bytes().reg <= FULL_OCCUPANCY_BYTES {
            FULL_OCCUPANCY_BYTES
        } else {
            u32::MAX
        };
        let mut pressures: Vec<Vec<u32>> = (0..self.blocks.len())
            .map(|bi| block_pressure(self, &live, bi))
            .collect();
        let mut uses: FxHashMap<SSAValue, Vec<Use>> = FxHashMap::default();
        for (block, b) in self.blocks.iter().enumerate() {
            for (ip, instr) in b.instrs.iter().enumerate() {
                for (src_idx, src) in instr.srcs().iter().enumerate() {
                    if let SrcRef::SSA(vec) = &src.src_ref {
                        for ssa in vec.iter() {
                            uses.entry(*ssa).or_default().push(Use {
                                block,
                                ip,
                                src_idx,
                            });
                        }
                    }
                }
            }
        }

        let mut progress = false;
        for (block, b) in self.blocks.iter_mut().enumerate() {
            let instrs = &mut b.instrs;
            let pressure = &mut pressures[block];
            let mut recent: FxHashMap<CopyKey, (SSARef, usize)> =
                FxHashMap::default();
            let mut removed = vec![false; instrs.len()];
            for ip in 0..instrs.len() {
                let Some((key, dst)) = copy_key(&instrs[ip]) else {
                    continue;
                };
                let Some((last, use_ips)) =
                    plain_uses(model, block, instrs, ip, &dst, &uses)
                else {
                    continue;
                };

                if let Some((existing, existing_last)) = recent.get_mut(&key) {
                    let gap = (*existing_last + 1).min(ip)..ip;
                    let extra = u32::from(existing.bytes());
                    let fits = pressure[gap.clone()]
                        .iter()
                        .all(|p| p + extra <= limit);
                    if ip.saturating_sub(*existing_last) <= MAX_GAP
                        && fits
                        && use_ips.iter().all(|&uip| {
                            rename_is_legal(&instrs[uip], &dst, existing)
                        })
                    {
                        for &uip in &use_ips {
                            for src in instrs[uip].srcs_mut() {
                                if let SrcRef::SSA(vec) = &mut src.src_ref {
                                    if *vec == dst {
                                        *vec = existing.clone();
                                    }
                                }
                            }
                        }
                        for p in &mut pressure[gap] {
                            *p += extra;
                        }
                        *existing_last = (*existing_last).max(last);
                        removed[ip] = true;
                        progress = true;
                        continue;
                    }
                }
                recent.insert(key, (dst, last));
            }

            if progress {
                let mut ip = 0;
                instrs.retain(|_| {
                    let keep = !removed[ip];
                    ip += 1;
                    keep
                });
            }
        }

        progress
    }
}
