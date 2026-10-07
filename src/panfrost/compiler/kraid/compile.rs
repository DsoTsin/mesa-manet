// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::data_type::NumericType;
use crate::debug::*;
use crate::decode::{Args, disassemble};
use crate::ir::*;
use crate::model::model_for_gpu_id;
use crate::ops::OpNop;
use crate::opt_tex_dual::TexDualPolicy;
use crate::spill::SPILL_LOOP_EXIT_DISTANCE;
use compiler::bindings::*;
use compiler::memstream::MemStream;
use kraid_bindings::*;

use std::collections::HashMap;
use std::sync::OnceLock;

fn nir_opts(arch: u8, merge_wg: bool) -> nir_shader_compiler_options {
    nir_shader_compiler_options {
        lower_scmp: true,
        lower_flrp16: true,
        lower_flrp32: true,
        lower_flrp64: true,
        lower_ffract: arch < 11,
        lower_fmod: true,
        lower_fdiv: true,
        lower_isign: true,
        lower_find_lsb: true,
        lower_ifind_msb: true,
        lower_fdph: true,
        lower_fsqrt: arch < 15,

        lower_fsign: true,

        lower_bitfield_insert: true,
        lower_bitfield_extract: true,
        lower_bitfield_extract8: true,
        lower_bitfield_extract16: true,
        has_bitfield_select: true,

        lower_pack_64_4x16: true,
        lower_pack_half_2x16: true,
        lower_pack_unorm_2x16: true,
        lower_pack_snorm_2x16: true,
        lower_pack_unorm_4x8: true,
        lower_pack_snorm_4x8: true,
        lower_unpack_half_2x16: true,
        lower_unpack_unorm_2x16: true,
        lower_unpack_snorm_2x16: true,
        lower_unpack_unorm_4x8: true,
        lower_unpack_snorm_4x8: true,
        has_pack_32_4x8: true,

        lower_doubles_options: nir_lower_dmod,
        lower_int64_options: !(nir_lower_iadd64
            | nir_lower_icmp64
            | nir_lower_ineg64
            | nir_lower_logic64
            | nir_lower_shift64
            | nir_lower_imul_2x32_64),
        lower_fisnormal: true,
        lower_uadd_carry: true,
        lower_usub_borrow: true,

        has_ldexp: true,
        has_isub: true,
        support_16bit_alu: true,
        vectorize_vec2_16bit: true,
        float_mul_add16: nir_float_muladd_support_has_ffma
            | nir_float_muladd_support_fuse,
        float_mul_add32: nir_float_muladd_support_has_ffma
            | nir_float_muladd_support_fuse,
        float_mul_add64: nir_float_muladd_support_has_ffma
            | nir_float_muladd_support_fuse,

        lower_uniforms_to_ubo: true,

        has_cs_global_id: true,
        lower_cs_local_index_to_id: true,
        lower_device_index_to_zero: true,
        max_unroll_iterations: 32,
        skip_partial_unroll: true,
        max_unroll_cost: 384,
        max_samples: 16,
        force_indirect_unrolling: (nir_var_shader_in
            | nir_var_shader_out
            | nir_var_function_temp),
        force_indirect_unrolling_sampler: true,
        scalarize_ddx: true,
        support_indirect_inputs: 0,
        lower_hadd: arch >= 11,
        lower_hadd64: true,
        discard_is_demote: true,
        has_udot_4x8: true,
        has_udot_4x8_sat: true,
        has_sdot_4x8: true,
        has_sdot_4x8_sat: true,
        has_sudot_4x8: true,
        has_sudot_4x8_sat: true,

        divergence_analysis_options: if merge_wg {
            nir_divergence_across_subgroups
                | nir_divergence_multiple_workgroup_per_compute_subgroup
        } else {
            0
        },
        lower_mediump_io: Some(pan_nir_lower_mediump_io),
        io_options: nir_io_has_intrinsics | nir_io_non_interpolated_as_uint,
        ..Default::default()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kraid_get_nir_shader_compiler_options(
    arch: u8,
    merge_wg: bool,
) -> *const nir_shader_compiler_options {
    static OPTS: OnceLock<HashMap<(u8, bool), nir_shader_compiler_options>> =
        OnceLock::new();

    let opts = OPTS
        .get_or_init(|| {
            let mut map = HashMap::new();
            for arch in 9..=15 {
                for merge_wg in [false, true] {
                    let opts = nir_opts(arch, merge_wg);
                    map.insert((arch, merge_wg), opts);
                }
            }
            map
        })
        .get(&(arch, merge_wg))
        .expect("Unsupported GPU arch");

    opts as *const _
}

fn dynarray_append_vec<T: Copy>(buf: &mut util_dynarray, vec: Vec<T>) {
    unsafe {
        let p = util_dynarray_grow_bytes(
            buf,
            vec.len().try_into().unwrap(),
            std::mem::size_of::<T>(),
        );
        assert!(!p.is_null(), "util_dynarray_grow_bytes() failed");
        std::ptr::copy_nonoverlapping(vec.as_ptr(), p as *mut T, vec.len());
    }
}

fn write_back_info(
    model: &dyn Model,
    src: &ShaderInfo,
    nir: &nir_shader,
    dst: &mut pan_shader_info,
    idvs: kraid_idvs_mode,
) {
    if idvs == KRAID_IDVS_VARYING {
        let vs = unsafe { &mut dst.__bindgen_anon_1.vs };

        vs.secondary_work_reg_count = src.registers_used.into();
        vs.secondary_preload = src.register_preload;
    } else {
        dst.work_reg_count = src.registers_used.into();
        dst.preload = src.register_preload;
    }

    dst.tls_size = dst.tls_size.max(src.tls_size);
    dst.has_shader_clk_instr |= src.has_ld_gclk;

    if model.arch() >= 9 {
        if nir.info.stage() == MESA_SHADER_FRAGMENT {
            let bifrost_info = unsafe { dst.__bindgen_anon_2.bifrost.as_mut() };
            bifrost_info.uses_flat_shading = src.uses_flat_shading;

            let translate_color = |dt: &Option<DataType>| {
                let Some(dt) = dt else {
                    return nir_type_invalid;
                };
                let num_type = match dt.num_type() {
                    NumericType::SignedInteger => nir_type_int,
                    NumericType::UnsignedInteger => nir_type_uint,
                    NumericType::Float => nir_type_float,
                    _ => panic!("Invalid color data type"),
                };
                num_type | (dt.bits() as nir_alu_type)
            };

            for (i, btype) in src.blend_types.iter().enumerate() {
                bifrost_info.blend[i].type_ = translate_color(btype);
            }
            bifrost_info.blend_src1_type = translate_color(&src.blend1_type);
            let fs = unsafe { &mut dst.__bindgen_anon_1.fs };
            fs.fixed_function_blend = src.fixed_function_blend;
            fs.hsr.ld_tile = src.hsr.ld_tile;
            fs.hsr.wait_or_tile_access_before_atest_zsemit =
                src.hsr.wait_or_tile_access_before_atest_zsemit;
            fs.hsr.varying_before_atest_zsemit =
                src.hsr.varying_before_atest_zsemit;
            fs.hsr.centroid_interpolation = src.hsr.centroid_interpolation;
        }
    } else {
        panic!("Unsupported GPU generation");
    }

    if nir.info.stage() == MESA_SHADER_VERTEX && idvs == KRAID_IDVS_ALL {
        let secondary_mask =
            unsafe { val_ex_fifo_varying_bits() } | (1 << VARYING_SLOT_POS);
        dst.__bindgen_anon_1.vs.secondary_enable =
            (nir.info.outputs_written & !secondary_mask) != 0;
    }
}

/// v9 reuses psiz writes as line width when drawing lines.
/// We cannot know what we're drawing at compile time so we need to create
/// a variant without psiz writes to be selected when we aren't drawing points
fn encode_no_psiz_variant(
    nir: &nir_shader,
    s: &mut Shader,
    model: &dyn Model,
    binary: &mut util_dynarray,
    info: &mut pan_shader_info,
    idvs: kraid_idvs_mode,
) {
    // TODO: v10+ HW should ignore psiz writes, investigate
    if nir.info.internal
        || nir.info.stage() != MESA_SHADER_VERTEX
        || (nir.info.outputs_written & (1 << VARYING_SLOT_PSIZ)) == 0
        || !(idvs == KRAID_IDVS_POSITION || idvs == KRAID_IDVS_ALL)
    {
        return;
    }

    info.__bindgen_anon_1.vs.no_psiz_offset = binary.size;
    // Find the store with is_psiz set
    let store = s
        .blocks
        .iter_mut()
        .rev()
        .flat_map(|b| b.instrs.iter_mut().rev())
        .find(|i| matches!(&i.op, Op::Store(s) if s.is_psiz))
        .expect("No psiz write found");

    // Patch it out, but preserve flow
    store.op = Op::Nop(OpNop {});
    s.write_instrumentation_counts();

    let bin = model.encode_shader(s);
    dynarray_append_vec(binary, bin);
}

fn print_disassembly(bin: &[u32], arch: u8) {
    if !DEBUG.contains(DebugFlags::PRINT) {
        return;
    }
    let mut lock = std::io::stderr().lock();
    let args = Args {
        arch,
        print_offset: true,
        print_hexdump: true,
    };
    let mut instrs: Vec<u64> = Default::default();
    for idx in (0..bin.len()).step_by(2) {
        let instr = (bin[idx] as u64) | (bin[idx + 1] as u64) << 32;
        if instr == 0 {
            break;
        }
        instrs.push(instr);
    }
    let _ = disassemble(&mut lock, &args, &instrs);
}

fn finish_shader(
    s: &mut Shader,
    loop_exit_distance: usize,
    const_pref: bool,
    anti_frag: bool,
    live_src: bool,
) {
    s.assign_registers(loop_exit_distance, const_pref, anti_frag, live_src);
    pass!(s.lower_copy());
    pass!(s.opt_redundant_movs());
    pass!(s.hoist_message_loads());
    pass!(s.schedule_after_ra());

    // These have to happen after register allocation because they may add
    // critical edges.
    pass!(s.opt_jump_thread());
    pass!(s.opt_branch_invert());
    pass!(s.opt_fall_through());

    // These have to happen last since we can't remove any instructions after
    // they've completed.
    pass!(s.assign_message_slots());
    pass!(s.insert_required_waits());
    pass!(s.mark_reconvergence());
    pass!(s.opt_flow());
    pass!(s.mark_helper_terminate());
    pass!(s.mark_last_use());
}

fn allocation_better(c: &Shader, b: &Shader) -> bool {
    c.info.registers_used <= b.info.registers_used
        && c.spill_aware_cost() < b.spill_aware_cost()
}

fn message_allocation_cost(s: &Shader) -> [usize; 4] {
    let mut cost = [0; 4];
    for instr in s.blocks.iter().flat_map(|b| &b.instrs) {
        cost[0] += 1;
        match &instr.op {
            Op::Store(op) if op.is_tls => cost[1] += 1,
            Op::Load(op) if op.is_tls => cost[2] += 1,
            _ => (),
        }
        if matches!(instr.op, Op::Load(_) | Op::LdPka(_)) && instr.flow.wait & 7 != 0 {
            cost[3] += 1;
        }
    }
    cost
}

fn message_allocation_better(c: &Shader, b: &Shader) -> bool {
    let cc = message_allocation_cost(c);
    let bc = message_allocation_cost(b);
    c.info.registers_used <= b.info.registers_used
        && cc.iter().zip(&bc).all(|(c, b)| c <= b)
        && (cc[3] < bc[3] || allocation_better(c, b))
}

fn select_allocation<'a>(mut s: Shader<'a>) -> Shader<'a> {
    pass!(s.schedule_for_pressure());
    pass!(s.legalize());
    let huge = s.blocks.iter().map(|b| b.instrs.len()).sum::<usize>() > 10000;
    let mut sunk = (!huge).then(|| s.clone());
    let has_sunk = sunk.as_mut().is_some_and(|c| pass!(c.opt_sink_staging()));
    if let Some(c) = sunk.as_mut().filter(|_| has_sunk) {
        pass!(c.legalize());
    }
    pass!(s.opt_share_copies());
    let better = allocation_better;
    let base = s;
    let mut messages = (!huge && base.model.arch() == 15).then(|| base.clone());
    if let Some(messages) = &mut messages {
        pass!(messages.schedule_for_message_loads());
        pass!(messages.legalize());
    }
    let mut s = base.clone();
    finish_shader(&mut s, 0, false, false, false);
    let mut distance = 0;
    if base.needs_spilling() {
        let mut c = base.clone();
        finish_shader(&mut c, SPILL_LOOP_EXIT_DISTANCE, false, false, false);
        if better(&c, &s) {
            s = c;
            distance = SPILL_LOOP_EXIT_DISTANCE;
        }
    }
    let mut c = base.clone();
    finish_shader(&mut c, distance, true, false, false);
    if better(&c, &s) {
        s = c;
    }
    if !huge {
        let mut c = base.clone();
        finish_shader(&mut c, distance, true, true, false);
        if better(&c, &s) {
            s = c;
        }
        let mut c = base;
        finish_shader(&mut c, distance, true, true, true);
        if better(&c, &s) {
            s = c;
        }
    }
    if let Some(mut sunk) = sunk.filter(|_| has_sunk) {
        pass!(sunk.opt_share_copies());
        finish_shader(&mut sunk, distance, true, true, false);
        if better(&sunk, &s) {
            s = sunk;
        }
    }
    if let Some(messages) = messages {
        for (const_pref, anti_frag, live_src) in [(false, false, false),
                                                (true, true, false), (true, true, true)] {
            let mut c = messages.clone();
            finish_shader(&mut c, distance, const_pref, anti_frag, live_src);
            if message_allocation_better(&c, &s) {
                s = c;
            }
        }
    }
    s
}

fn allocate<'a>(
    mut s: Shader<'a>,
    inputs: &pan_compile_inputs,
    fau: &mut pan_fau_layout,
) -> Shader<'a> {
    pass!(s.allocate_fau(inputs, fau));
    pass!(s.opt_share_csel_zero());
    select_allocation(s)
}

struct VariantCost {
    regs: u8,
    spill_cost: u64,
    instrs: usize,
    spills: usize,
    fills: usize,
    tls: u32,
}

impl VariantCost {
    fn new(s: &Shader) -> Self {
        let mut cost = VariantCost {
            regs: s.info.registers_used,
            spill_cost: s.spill_aware_cost(),
            instrs: 0,
            spills: 0,
            fills: 0,
            tls: s.info.tls_size,
        };
        for instr in s.blocks.iter().flat_map(|b| b.instrs.iter()) {
            cost.instrs += 1;
            match &instr.op {
                Op::Store(op) if op.is_tls => cost.spills += 1,
                Op::Load(op) if op.is_tls => cost.fills += 1,
                _ => (),
            }
        }
        cost
    }

    fn no_worse_than(&self, other: &VariantCost) -> bool {
        self.regs <= other.regs
            && self.spill_cost <= other.spill_cost
            && self.instrs <= other.instrs
            && self.spills <= other.spills
            && self.fills <= other.fills
            && self.tls <= other.tls
    }

    fn key(&self) -> (u64, usize, u8, usize, usize, u32) {
        (
            self.spill_cost,
            self.instrs,
            self.regs,
            self.spills,
            self.fills,
            self.tls,
        )
    }
}

fn fau_values_match(a: &pan_fau_layout, b: &pan_fau_layout) -> bool {
    if a.reserved != b.reserved || a.is_pilot != b.is_pilot {
        return false;
    }
    let is_value = |fau: &pan_fau_layout, w: usize| {
        w < fau.count as usize && (fau.is_const[w / 32] >> (w % 32)) & 1 == 0
    };
    let words = a.count.max(b.count) as usize;
    (a.reserved as usize..words).all(|w| {
        let value = is_value(a, w);
        value == is_value(b, w)
            && (!value || unsafe { a.words[w].constant == b.words[w].constant })
    })
}

fn select_tex_dual<'a>(
    paired: Vec<Shader<'a>>,
    plain: Shader<'a>,
    inputs: &pan_compile_inputs,
    fau: &mut pan_fau_layout,
) -> Shader<'a> {
    let virt = inputs.fau.virt;
    let read_virt = || (!virt.is_null()).then(|| unsafe { *virt });
    let write_virt = |v: Option<pan_fau_virtual>| {
        if let Some(v) = v {
            unsafe { *virt = v };
        }
    };

    let fau_in = *fau;
    let virt_in = read_virt();
    let plain = allocate(plain, inputs, fau);
    let plain_cost = VariantCost::new(&plain);
    let fau_plain = *fau;
    let virt_plain = read_virt();

    let mut best = None;
    for variant in paired {
        *fau = fau_in;
        write_virt(virt_in);
        let variant = allocate(variant, inputs, fau);
        let virt_variant = read_virt();
        let same_values = fau_values_match(fau, &fau_plain)
            && match (&virt_variant, &virt_plain) {
                (Some(a), Some(b)) => a.map == b.map && a.placed == b.placed,
                (None, None) => true,
                _ => false,
            };
        let cost = VariantCost::new(&variant);
        if same_values
            && cost.no_worse_than(&plain_cost)
            && best
                .as_ref()
                .is_none_or(|(_, c, _, _): &(_, VariantCost, _, _)| {
                    cost.key() < c.key()
                })
        {
            best = Some((variant, cost, *fau, virt_variant));
        }
    }

    match best {
        Some((variant, _, fau_variant, virt_variant)) => {
            *fau = fau_variant;
            write_virt(virt_variant);
            variant
        }
        None => {
            *fau = fau_plain;
            write_virt(virt_plain);
            plain
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kraid_compile_nir(
    nir: &mut nir_shader,
    inputs: &pan_compile_inputs,
    binary: &mut util_dynarray,
    info: &mut pan_shader_info,
    idvs: kraid_idvs_mode,
) {
    let model = model_for_gpu_id(inputs.gpu_id, inputs.gpu_variant).unwrap();

    if DEBUG.contains(DebugFlags::PRINT) {
        eprint!("{}", nir.to_string().unwrap());
    }

    let mut s = Shader::from_nir(model.as_ref(), nir, inputs);
    s.run_pass("after translation from NIR", |_| {});

    pass!(s.opt_cse());
    pass!(s.opt_copy_prop());
    pass!(s.lower_math());
    pass!(s.remat_constants());
    pass!(s.widen_alu_ops());
    pass!(s.legalize_src_swizzles());
    pass!(s.opt_copy_prop());
    while pass!(s.opt_dst_mod_prop()) {
        pass!(s.opt_copy_prop());
    }
    pass!(s.lower_mkvec_swz());
    pass!(s.opt_var());
    if pass!(s.opt_cse()) {
        pass!(s.opt_copy_prop());
    }
    pass!(s.opt_dce());
    pass!(s.opt_select_immediates());
    pass!(s.opt_normalize_consts());
    pass!(s.lower_small_constants());
    pass!(s.opt_exec_units());
    pass!(s.mark_helper_skip());
    let mut s = if s.has_tex_dual_candidates() {
        let huge = s.blocks.iter().map(|b| b.instrs.len()).sum::<usize>() > 10000;
        let policies: &[TexDualPolicy] = if huge {
            &[TexDualPolicy::Strict]
        } else {
            &[TexDualPolicy::Relaxed, TexDualPolicy::Strict]
        };
        let mut paired: Vec<Shader> = Vec::new();
        for &policy in policies {
            let mut variant = s.clone();
            if pass!(variant.opt_tex_dual(policy))
                && !paired.iter().any(|p| {
                    p.tex_dual_signature() == variant.tex_dual_signature()
                })
            {
                paired.push(variant);
            }
        }
        if paired.is_empty() {
            allocate(s, inputs, &mut info.fau)
        } else {
            select_tex_dual(paired, s, inputs, &mut info.fau)
        }
    } else {
        allocate(s, inputs, &mut info.fau)
    };

    let stats = if !s.is_empty() {
        let mut stats = s.get_stats(nir.scratch_size);
        s.write_instrumentation_counts();
        pass!(s.lower_blend_call());
        if model.arch() >= 13 && nir.info.stage() == MESA_SHADER_FRAGMENT {
            s.info.hsr = crate::encode_v9::v9_gather_hsr_info(&s, model.arch());
        }

        let bin = model.encode_shader(&s);
        print_disassembly(&bin, model.arch());
        let code_size = std::mem::size_of_val(&bin[..]);
        dynarray_append_vec(binary, bin);

        encode_no_psiz_variant(nir, &mut s, model.as_ref(), binary, info, idvs);

        if stats.isa == PAN_STAT_VALHALL {
            stats.__bindgen_anon_1.valhall.code_size =
                code_size.try_into().unwrap();
        } else {
            panic!("Unsupported ISA");
        }
        stats
    } else {
        pan_stats {
            isa: PAN_STAT_VALHALL,
            __bindgen_anon_1: pan_stats__bindgen_ty_1 {
                valhall: valhall_stats::default(),
            },
        }
    };

    if idvs == KRAID_IDVS_VARYING {
        info.stats_idvs_varying = stats;
    } else {
        info.stats = stats;
    }

    write_back_info(model.as_ref(), &s.info, nir, info, idvs);
    unsafe { pan_shader_update_info(info, nir, inputs) };

    if DEBUG.contains(DebugFlags::STATS) {
        let mut stream = MemStream::new().expect("Failed to open memstream");

        unsafe {
            // Both binding crates generate their own stdio FILE, those are the
            // same type underneath
            let f = stream.c_file() as *mut kraid_bindings::FILE;
            let prefix =
                kraid_shader_stage_name(nir.info.stage(), inputs.is_blend);

            pan_stats_verbose_prologue(
                f,
                prefix,
                inputs.gpu_id,
                inputs.gpu_variant,
                model.arch().into(),
            );
            // TODO: min/max statistics
            pan_valhall_stats_verbose(
                f,
                &info.stats.__bindgen_anon_1.valhall,
                std::ptr::null(),
                std::ptr::null(),
                info.tls_size,
            );
            pan_stats_verbose_epilogue(f, info);
        }

        eprint!("{}", stream.take_utf8_string_lossy().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::FlowWaitBit;
    use crate::ops::{MemAccess, OpLoad, OpStore};
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn cost_shader(model: &dyn Model, cost: [usize; 4], regs: u8) -> Shader<'_> {
        let mut instrs = Vec::new();
        for _ in 0..cost[1] {
            instrs.push(Instr::from(OpStore { src_type: DataType::I32, is_tls: true,
                is_psiz: false, access: MemAccess::None, data: 0_u32.into(),
                addr: 0_u32.into(), offset: 0 }));
        }
        for ip in 0..cost[2] + cost[3] {
            let mut load = Instr::from(OpLoad { dst: RegRef::new(0, RegRange::Regs(1)).into(),
                dst_type: DataType::I32, is_tls: ip < cost[2], access: MemAccess::None,
                addr: 0_u32.into(), offset: 0 });
            if ip >= cost[2] {
                load.flow.set_wait_bit(FlowWaitBit::Slot0);
            }
            instrs.push(load);
        }
        assert!(instrs.len() <= cost[0]);
        instrs.resize_with(cost[0], || OpNop {}.into());
        let mut cfg: CFGBuilder<usize, BasicBlock, FxBuildHasher> = CFGBuilder::new();
        cfg.add_node(0, BasicBlock { label: LabelAllocator::default().alloc(), instrs });
        let mut info = ShaderInfo::default();
        info.registers_used = regs;
        Shader { model, ssa_alloc: Default::default(), phi_alloc: Default::default(),
                 blocks: cfg.as_cfg(false), info, constant_pool: None }
    }

    #[test]
    fn message_candidate_never_trades_registers_instructions_spills_or_fills() {
        let model = model_for_gpu_id(0x0f080000f0000000, 4).unwrap();
        let baseline = cost_shader(model.as_ref(), [6, 1, 1, 2], 32);
        let cost = [6, 1, 1, 1];
        assert!(message_allocation_better(&cost_shader(model.as_ref(), cost, 32), &baseline));
        assert!(!message_allocation_better(&cost_shader(model.as_ref(), cost, 33), &baseline));
        for field in 0..3 {
            let mut worse = cost;
            worse[field] += 1;
            assert!(!message_allocation_better(&cost_shader(model.as_ref(), worse, 32), &baseline));
        }
        assert!(!message_allocation_better(&cost_shader(model.as_ref(), [6, 1, 1, 3], 32), &baseline));
        assert!(!message_allocation_better(&cost_shader(model.as_ref(), [5, 1, 1, 3], 32), &baseline));
    }
}
