#include "panvk_nir_rt_pipeline.h"
#include "nir_builder.h"
#include "nir_constant_expressions.h"
#include <stdio.h>

static const nir_shader_compiler_options options = {0};

static nir_const_value
evaluate(nir_def *def, unsigned comp, const uint64_t *args)
{
   if (nir_def_is_const(def))
      return nir_def_as_load_const(def)->value[comp];
   if (nir_def_instr(def)->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
      if (intr->intrinsic == nir_intrinsic_load_rt_tls_base_pan)
         return nir_const_value_for_uint(args[32], 64);
      if (intr->intrinsic == nir_intrinsic_load_rt_frame_size_pan)
         return nir_const_value_for_uint(args[35], 32);
      if (intr->intrinsic == nir_intrinsic_load_global_invocation_id ||
          intr->intrinsic == nir_intrinsic_load_ray_launch_id)
         return nir_const_value_for_uint(args[17 + comp], 32);
       if (intr->intrinsic == nir_intrinsic_load_global ||
           intr->intrinsic == nir_intrinsic_load_global_constant ||
           intr->intrinsic == nir_intrinsic_load_push_constant) {
          uint64_t address;
          if (intr->intrinsic == nir_intrinsic_load_push_constant) {
             unsigned relative = nir_intrinsic_base(intr) - args[33];
             unsigned offset = relative < PANVK_RT_DISPATCH_PARAMS_SIZE
                ? 104 + relative : 88 + relative - PANVK_RT_DISPATCH_PARAMS_SIZE;
             address = args[34] + offset + evaluate(intr->src[0].ssa, 0, args).u32;
          } else {
             address = evaluate(intr->src[0].ssa, 0, args).u64;
          }
          address += comp * (def->bit_size / 8);
         uint64_t value = address * UINT64_C(0x9e3779b97f4a7c15) ^ (address >> 32);
         if (def->bit_size < 64)
            value &= (UINT64_C(1) << def->bit_size) - 1;
         return nir_const_value_for_uint(value, def->bit_size);
      }
      assert(intr->intrinsic == nir_intrinsic_load_rt_arg_pan);
      return nir_const_value_for_uint(args[nir_intrinsic_base(intr)], def->bit_size);
   }
   nir_alu_instr *alu = nir_def_as_alu(def);
   nir_const_value values[16][16] = {0}, *sources[16], output[16] = {0};
   const nir_op_info *op = &nir_op_infos[alu->op];
   for (unsigned s = 0; s < op->num_inputs; s++) {
      sources[s] = values[s];
      unsigned comps = op->input_sizes[s] ? op->input_sizes[s] : def->num_components;
      for (unsigned c = 0; c < comps; c++)
         values[s][c] = evaluate(alu->src[s].src.ssa, alu->src[s].swizzle[c], args);
   }
   nir_eval_const_opcode(alu->op, output, NULL, def->num_components, def->bit_size,
                         sources, 0);
   return output[comp];
}

static void
call_values(nir_intrinsic_instr *call, const uint64_t *input, uint64_t *output)
{
   const unsigned bases[] = {0, 5, 2, 6};
   for (unsigned i = 0; i < 4; i++)
      output[bases[i]] = evaluate(call->src[1].ssa, i, input).u64;
   for (unsigned i = 0; i < 2; i++) {
      output[7 + i] = evaluate(call->src[2].ssa, 2 * i, input).u32 |
         ((uint64_t)evaluate(call->src[2].ssa, 2 * i + 1, input).u32 << 32);
   }
   for (unsigned i = 0; i < 4; i++)
      output[9 + i] = evaluate(call->src[2].ssa, i + 4, input).u32;
}

static uint64_t
output_value(nir_shader *nir, const uint64_t *args)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_global)
               return evaluate(intr->src[0].ssa, 0, args).u64;
         }
      }
   }
   UNREACHABLE("missing result");
}

static nir_intrinsic_instr *
make_call(nir_builder *b, nir_def *frame, nir_def *payload, nir_def *root,
          nir_def *depth, mesa_shader_stage stage)
{
   nir_def *z = nir_imm_int(b, 0);
   nir_def *hit[] = {z, z, z, z, z, z, depth,
      nir_load_rt_arg_pan(b, 32, .base = 12)};
   nir_rt_call_pan(b, nir_imm_int(b, 0x1234),
      nir_vec4(b, frame, payload, root, nir_load_rt_arg_pan(b, 64, .base = 6)),
      nir_vec(b, hit, 8), nir_imm_int64(b, 0), nir_imm_int(b, 0),
      .base = stage, .rt_arg_mask = 0xffff);
   return nir_instr_as_intrinsic(nir_builder_last_instr(b));
}

static void
check(unsigned mode, nir_op operation, uint64_t constant)
{
   nir_builder entry = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *frame = nir_load_rt_arg_pan(&entry, 64, .base = 0);
   nir_def *base = nir_load_rt_arg_pan(&entry, 64, .base = 5);
   unsigned const_bits = operation == nir_op_ishl || operation == nir_op_ushr ||
                         operation == nir_op_ishr ? 32 : 64;
   nir_def *imm = nir_imm_intN_t(&entry, constant, const_bits);
   nir_def *payload = nir_build_alu(&entry, operation,
      mode == 7 ? imm : base, mode == 7 ? base : imm, NULL, NULL);
   if (mode == 4)
      nir_store_global(&entry, payload, nir_imm_int64(&entry, 0x2000), .align_mul = 8);
   nir_def *root = mode == 1 ? nir_load_rt_arg_pan(&entry, 64, .base = 2) : frame;
   nir_intrinsic_instr *calls[2];
   nir_def *depth = mode == 6 ? nir_build_alu(&entry, operation,
      nir_load_rt_arg_pan(&entry, 32, .base = 11), nir_imm_int(&entry, constant), NULL, NULL)
      : nir_imm_int(&entry, 77);
   calls[0] = make_call(&entry, frame, payload, root, depth, MESA_SHADER_MISS);
   unsigned call_count = 1;
   if (mode == 2) {
      calls[1] = make_call(&entry, frame, base, nir_load_rt_arg_pan(&entry, 64, .base = 2),
                           nir_imm_int(&entry, 78), MESA_SHADER_MISS);
      call_count++;
   }
   nir_def *value = nir_ixor(&callee, nir_load_rt_arg_pan(&callee, 64, .base = 0),
                                      nir_load_rt_arg_pan(&callee, 64, .base = 2));
   value = nir_ixor(&callee, value, nir_load_rt_arg_pan(&callee, 64, .base = 5));
   value = nir_ixor(&callee, value, nir_u2u64(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 11)));
   value = nir_ixor(&callee, value, nir_u2u64(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 12)));
   nir_store_global(&callee, value, nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   nir_shader *original = nir_shader_clone(NULL, callee.shader);
   const uint64_t inputs[] = {0, 1, UINT32_MAX, UINT64_MAX, 0x8000000000000000ull,
                              0x123456789abcdef0ull};
   uint64_t expected[2][ARRAY_SIZE(inputs)];
   for (unsigned c = 0; c < call_count; c++) {
      for (unsigned i = 0; i < ARRAY_SIZE(inputs); i++) {
         uint64_t in[13] = {0}, actual[13] = {0};
         in[0] = inputs[i]; in[2] = ~inputs[i]; in[5] = inputs[i]; in[11] = (uint32_t)inputs[i]; in[12] = (uint32_t)~inputs[i];
         call_values(calls[c], in, actual);
         expected[c][i] = output_value(original, actual);
      }
   }
   struct panvk_nir_rt_pipeline_stage stage = {.stage = MESA_SHADER_MISS};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {
      .stages = &stage, .stage_count = mode == 5 ? 0 : 1,
      .flags = mode == 3 ? VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR : 0,
   };
   struct panvk_nir_rt_pipeline_result result = {.nir = entry.shader, .functions = functions};
   panvk_nir_opt_rt_calls(&info, &result);
   nir_validate_shader(entry.shader, "call arguments");
   nir_validate_shader(callee.shader, "callee arguments");
   for (unsigned c = 0; c < call_count; c++) {
      for (unsigned i = 0; i < ARRAY_SIZE(inputs); i++) {
         uint64_t in[13] = {0}, actual[13] = {0};
         in[0] = inputs[i]; in[2] = ~inputs[i]; in[5] = inputs[i]; in[11] = (uint32_t)inputs[i]; in[12] = (uint32_t)~inputs[i];
         call_values(calls[c], in, actual);
         assert(output_value(callee.shader, actual) == expected[c][i]);
         if (mode == 0 && (nir_intrinsic_rt_arg_mask(calls[c]) & 0xc))
            assert(actual[5] == inputs[i]);
         if (mode == 6 && (nir_intrinsic_rt_arg_mask(calls[c]) & 0x4000))
            assert(actual[11] == (uint32_t)inputs[i]);
      }
      unsigned mask = nir_intrinsic_rt_arg_mask(calls[c]);
      if (mode == 3 || mode == 5)
         assert(mask == 0xffff);
      else {
         assert(!(mask & 0x3fc0));
         if (mode == 0)
            assert(!(mask & 0x4030));
         if (mode == 2)
            assert((mask & 0x4030) == 0x4030);
      }
   }
   ralloc_free(original);
   ralloc_free(entry.shader);
   ralloc_free(callee.shader);
}

static void
check_targets(bool recursive)
{
   nir_builder entry = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder a = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *ef = nir_load_rt_arg_pan(&entry, 64, .base = 0);
   nir_def *ep = nir_load_rt_arg_pan(&entry, 64, .base = 5);
   nir_def *er = nir_load_rt_arg_pan(&entry, 64, .base = 2);
   nir_intrinsic_instr *outer = make_call(&entry, ef, ep, recursive ? ef : er,
      nir_load_rt_arg_pan(&entry, 32, .base = 11), MESA_SHADER_MISS);
   nir_def *af = nir_load_rt_arg_pan(&a, 64, .base = 0);
   nir_def *ap = nir_load_rt_arg_pan(&a, 64, .base = 5);
   nir_def *ar = nir_load_rt_arg_pan(&a, 64, .base = 2);
   nir_intrinsic_instr *inner = NULL;
   if (recursive) {
      inner = make_call(&a, af, ap, af, nir_load_rt_arg_pan(&a, 32, .base = 11),
                         MESA_SHADER_CALLABLE);
      nir_def *bf = nir_load_rt_arg_pan(&b, 64, .base = 0);
      make_call(&b, bf,
         nir_load_rt_arg_pan(&b, 64, .base = 5), bf,
         nir_load_rt_arg_pan(&b, 32, .base = 11), MESA_SHADER_MISS);
   }
   nir_store_global(&a, recursive ? nir_ixor(&a, af, ar) : af,
                     nir_imm_int64(&a, 0x1000), .align_mul = 8);
   nir_store_global(&b, nir_load_rt_arg_pan(&b, 64, .base = 2),
                     nir_imm_int64(&b, 0x1000), .align_mul = 8);
   struct panvk_nir_rt_pipeline_stage stages[] = {
      {.stage = MESA_SHADER_MISS},
      {.stage = recursive ? MESA_SHADER_CALLABLE : MESA_SHADER_MISS},
   };
   nir_shader *functions[] = {a.shader, b.shader};
   struct panvk_nir_rt_pipeline_info info = {.stages = stages, .stage_count = 2};
   struct panvk_nir_rt_pipeline_result result = {.nir = entry.shader, .functions = functions};
   panvk_nir_opt_rt_calls(&info, &result);
   unsigned mask = nir_intrinsic_rt_arg_mask(outer);
   assert((mask & 0x33) == 0x33);
   if (!recursive)
      assert(mask == 0x33);
   else
      assert((nir_intrinsic_rt_arg_mask(inner) & 0x33) == 0x33);
   nir_validate_shader(entry.shader, "target union entry");
   nir_validate_shader(a.shader, "target union callee A");
   nir_validate_shader(b.shader, "target union callee B");
   ralloc_free(entry.shader);
   ralloc_free(a.shader);
   ralloc_free(b.shader);
}

static unsigned
count_calls(nir_shader *nir)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            count += instr->type == nir_instr_type_intrinsic &&
               (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_rt_call_pan ||
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_rt_call_result_pan);
         }
      }
   }
   return count;
}

static void
check_argument_chain(unsigned mode, uint64_t constant)
{
   nir_builder b[4];
   for (unsigned i = 0; i < ARRAY_SIZE(b); i++)
      b[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   const mesa_shader_stage kinds[] = {
      MESA_SHADER_MISS, MESA_SHADER_CLOSEST_HIT, MESA_SHADER_CALLABLE,
   };
   nir_intrinsic_instr *calls[4];
   calls[0] = make_call(&b[0], nir_imm_int64(&b[0], constant),
      nir_imm_int64(&b[0], constant ^ 0xa5), nir_imm_int64(&b[0], constant + 9),
      nir_imm_int(&b[0], constant), kinds[0]);
   if (mode == 1)
      calls[3] = make_call(&b[0], nir_imm_int64(&b[0], constant),
         nir_imm_int64(&b[0], constant ^ 0xa6), nir_imm_int64(&b[0], constant + 9),
         nir_imm_int(&b[0], constant), kinds[0]);
   for (unsigned i = 1; i < 3; i++)
      calls[i] = make_call(&b[i], nir_load_rt_arg_pan(&b[i], 64, .base = 0),
         nir_load_rt_arg_pan(&b[i], 64, .base = 5),
         nir_load_rt_arg_pan(&b[i], 64, .base = 2),
         nir_load_rt_arg_pan(&b[i], 32, .base = 11), kinds[i]);
   nir_def *value = nir_load_rt_arg_pan(&b[3], 64, .base = 0);
   value = nir_ixor(&b[3], value, nir_load_rt_arg_pan(&b[3], 64, .base = 5));
   value = nir_iadd(&b[3], value, nir_load_rt_arg_pan(&b[3], 64, .base = 2));
   value = nir_ixor(&b[3], value,
      nir_u2u64(&b[3], nir_load_rt_arg_pan(&b[3], 32, .base = 11)));
   nir_store_global(&b[3], value, nir_imm_int64(&b[3], 0x1000), .align_mul = 8);
   if (mode == 2)
      make_call(&b[3], nir_load_rt_arg_pan(&b[3], 64, .base = 0),
         nir_load_rt_arg_pan(&b[3], 64, .base = 5),
         nir_load_rt_arg_pan(&b[3], 64, .base = 2),
         nir_load_rt_arg_pan(&b[3], 32, .base = 11), kinds[0]);
   struct panvk_nir_rt_pipeline_stage stages[3];
   nir_shader *functions[3];
   for (unsigned i = 0; i < 3; i++) {
      stages[i] = (struct panvk_nir_rt_pipeline_stage){.stage = kinds[i]};
      functions[i] = b[i + 1].shader;
   }
   struct panvk_nir_rt_pipeline_info info = {
      .native = true, .stages = stages, .stage_count = 3,
      .flags = mode == 3 ? VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR : 0,
   };
   struct panvk_nir_rt_pipeline_result result = {.nir = b[0].shader, .functions = functions};
   uint64_t expected[2][32];
   for (unsigned n = 0; n < 32; n++) {
      for (unsigned c = 0; c < (mode == 1 ? 2 : 1); c++) {
         uint64_t args[13] = {[6] = n * UINT64_C(0x9e3779b97f4a7c15), [12] = n};
         call_values(c ? calls[3] : calls[0], args, args);
         call_values(calls[1], args, args);
         call_values(calls[2], args, args);
         expected[c][n] = output_value(b[3].shader, args);
      }
   }
   panvk_nir_opt_rt_calls(&info, &result);
   for (unsigned n = 0; n < 32; n++) {
      for (unsigned c = 0; c < (mode == 1 ? 2 : 1); c++) {
         uint64_t args[13] = {[6] = n * UINT64_C(0x9e3779b97f4a7c15), [12] = n};
         call_values(c ? calls[3] : calls[0], args, args);
         call_values(calls[1], args, args);
         call_values(calls[2], args, args);
         assert(output_value(b[3].shader, args) == expected[c][n]);
      }
   }
   unsigned loads = 0;
   nir_foreach_block(block, b[3].impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_rt_arg_pan)
            loads++;
      }
   }
   if (mode == 0)
      assert(loads == 0);
   else if (mode == 1)
      assert(loads == 1);
   else
      assert(loads != 0);
   assert(!panvk_nir_opt_rt_calls(&info, &result));
   for (unsigned i = 0; i < ARRAY_SIZE(b); i++) {
      nir_validate_shader(b[i].shader, "RT argument chain");
      ralloc_free(b[i].shader);
   }
}

static void
check_argument_canonicalization(void)
{
   nir_builder entry = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *frame = nir_iadd_imm(&entry, nir_load_rt_arg_pan(&entry, 64, .base = 0), 16);
   nir_def *root = frame;
   nir_intrinsic_instr *call = make_call(&entry, frame, nir_imm_int64(&entry, 0),
      root, nir_imm_int(&entry, 0), MESA_SHADER_MISS);
   nir_store_global(&callee,
      nir_ixor(&callee, nir_load_rt_arg_pan(&callee, 64, .base = 0),
                         nir_load_rt_arg_pan(&callee, 64, .base = 2)),
      nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   struct panvk_nir_rt_pipeline_stage stage = {.stage = MESA_SHADER_MISS};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = &stage, .stage_count = 1};
   struct panvk_nir_rt_pipeline_result result = {.nir = entry.shader, .functions = functions};
   panvk_nir_opt_rt_calls(&info, &result);
   assert(nir_intrinsic_rt_arg_mask(call) == 0);
   assert(!panvk_nir_opt_rt_calls(&info, &result));
   for (unsigned n = 0; n < 64; n++) {
      uint64_t args[13] = {n * UINT64_C(0x9e3779b97f4a7c15)};
      call_values(call, args, args);
      assert(output_value(callee.shader, args) == 0);
   }
   nir_validate_shader(entry.shader, "RT argument CSE entry");
   nir_validate_shader(callee.shader, "RT argument CSE callee");
   ralloc_free(entry.shader);
   ralloc_free(callee.shader);
}

static void
check_lto(unsigned mode)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *args[4], *hit[8];
   const unsigned bases[] = {0, 5, 2, 6};
   for (unsigned i = 0; i < 4; i++)
      args[i] = nir_load_rt_arg_pan(&caller, 64, .base = bases[i]);
   for (unsigned i = 0; i < 2; i++) {
      nir_def *value = nir_unpack_64_2x32(&caller,
         nir_load_rt_arg_pan(&caller, 64, .base = 7 + i));
      hit[2 * i] = nir_channel(&caller, value, 0);
      hit[2 * i + 1] = nir_channel(&caller, value, 1);
   }
   for (unsigned i = 4; i < 8; i++)
      hit[i] = nir_load_rt_arg_pan(&caller, 32, .base = 5 + i);
   mesa_shader_stage kind = mode == 7 ? MESA_SHADER_RAYGEN : MESA_SHADER_MISS;
   nir_rt_call_pan(&caller, nir_imm_int(&caller, 7), nir_vec(&caller, args, 4),
      nir_vec(&caller, hit, 8), nir_imm_int64(&caller, 0), nir_imm_int(&caller, 0),
      .base = kind, .rt_arg_mask = 0xffff);
   nir_intrinsic_instr *call = nir_instr_as_intrinsic(nir_builder_last_instr(&caller));
   nir_barrier(&caller, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_DEVICE, .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_ssbo);
   nir_intrinsic_instr *user_barrier = nir_instr_as_intrinsic(nir_builder_last_instr(&caller));
   nir_def *value = nir_imm_int64(&callee, 13);
   for (unsigned base = 0; base < 13; base++) {
      if (base == 1 || base == 3 || base == 4)
         continue;
      value = nir_iadd(&callee, nir_imul_imm(&callee, value, 17),
         nir_u2u64(&callee, nir_load_rt_arg_pan(&callee, base >= 9 ? 32 : 64, .base = base)));
   }
   if (mode == 8)
      value = nir_iadd(&callee, value, nir_load_rt_arg_pan(&callee, 64, .base = 3));
   nir_store_global(&callee, value, nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   if (mode == 6)
      make_call(&callee, nir_imm_int64(&callee, 0), nir_imm_int64(&callee, 0),
         nir_imm_int64(&callee, 0), nir_imm_int(&callee, 0), kind);
   if (mode == 9) {
      for (unsigned i = 0; i < 300; i++)
         nir_store_global(&callee, value, nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   }
   nir_rt_return_pan(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 4),
      nir_load_rt_arg_pan(&callee, 64, .base = 3), .base = mode == 7);
   if (mode == 5)
      callee.shader->info.float_controls_execution_mode = FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32;
   struct panvk_nir_rt_pipeline_stage stages[2] = {{.stage = kind}, {.stage = kind}};
   nir_shader *functions[] = {callee.shader, callee.shader};
   struct panvk_nir_rt_pipeline_info info = {
      .stages = stages, .stage_count = mode == 4 ? 2 : 1,
      .native = true, .disable_lto = mode == 1,
      .flags = mode == 2 ? VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR :
               mode == 3 ? VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT : 0,
   };
   struct panvk_nir_rt_pipeline_result result = {.nir = caller.shader, .functions = functions};
   uint64_t expected[128], inputs[128][13];
   uint64_t random = 0x9e3779b97f4a7c15ull;
   for (unsigned n = 0; n < ARRAY_SIZE(expected); n++) {
      for (unsigned a = 0; a < 13; a++) {
         random ^= random << 13;
         random ^= random >> 7;
         random ^= random << 17;
         inputs[n][a] = a >= 9 || a == 4 ? (uint32_t)random : random;
      }
      uint64_t actual[13] = {0};
      call_values(call, inputs[n], actual);
      actual[3] = inputs[n][3];
      expected[n] = output_value(callee.shader, actual);
   }
   bool progress = panvk_nir_opt_rt_lto(&info, &result);
   bool inlined = mode == 0 || mode == 6 || mode == 9;
   bool remaining = !inlined || mode == 6;
   assert(progress == inlined);
   assert(count_calls(caller.shader) == remaining);
   if (progress) {
      for (unsigned n = 0; n < ARRAY_SIZE(expected); n++)
         assert(output_value(caller.shader, inputs[n]) == expected[n]);
   }
   assert(panvk_nir_lower_rt_call_barriers(caller.shader) == remaining);
   unsigned barriers = 0;
   nir_foreach_block(block, caller.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_barrier)
            continue;
         barriers++;
         if (intr != user_barrier) {
            assert(nir_intrinsic_memory_modes(intr) == nir_var_mem_global);
            assert(nir_intrinsic_execution_scope(intr) == SCOPE_NONE);
         }
      }
   }
   assert(barriers == 1 + 2 * remaining);
   assert(nir_intrinsic_memory_modes(user_barrier) == nir_var_mem_ssbo);
   assert(nir_intrinsic_execution_scope(user_barrier) == SCOPE_WORKGROUP);
   ralloc_free(callee.shader);
   nir_validate_shader(caller.shader, "LTO caller after callee release");
   ralloc_free(caller.shader);
}

static bool
replace_target_index(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_rt_arg_pan || nir_intrinsic_base(intr) != 11)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_int(b, *(uint32_t *)data));
   return true;
}

static void
check_lto_multiple_targets(unsigned components, bool partial, bool direct)
{
   nir_builder shaders[3];
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++)
      shaders[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   mesa_shader_stage kind = components ? MESA_SHADER_INTERSECTION : MESA_SHADER_MISS;
   nir_intrinsic_instr *call = make_call(&shaders[0],
      nir_load_rt_arg_pan(&shaders[0], 64, .base = 0),
      nir_load_rt_arg_pan(&shaders[0], 64, .base = 5),
      nir_load_rt_arg_pan(&shaders[0], 64, .base = 2), nir_imm_int(&shaders[0], 0), kind);
   shaders[0].cursor = nir_before_instr(&call->instr);
   nir_src_rewrite(&call->src[4], nir_load_rt_arg_pan(&shaders[0], 32, .base = 11));
   if (components) {
      nir_def *value = nir_rt_call_result_pan(&shaders[0], components,
         call->src[0].ssa, call->src[1].ssa, call->src[2].ssa, call->src[3].ssa,
         call->src[4].ssa, .base = kind, .rt_arg_mask = 0xffff);
      nir_instr_remove(&call->instr);
      shaders[0].cursor = nir_after_impl(shaders[0].impl);
      nir_def *checksum = nir_imm_int64(&shaders[0], 0);
      for (unsigned c = 0; c < components; c++)
         checksum = nir_iadd(&shaders[0], nir_imul_imm(&shaders[0], checksum, 17),
            nir_u2u64(&shaders[0], nir_channel(&shaders[0], value, c)));
      nir_store_global(&shaders[0], checksum, nir_imm_int64(&shaders[0], 0x1000), .align_mul = 8);
   }
   shaders[0].cursor = nir_after_impl(shaders[0].impl);
   nir_rt_return_pan(&shaders[0], nir_load_rt_arg_pan(&shaders[0], 32, .base = 4),
      nir_load_rt_arg_pan(&shaders[0], 64, .base = 3), .stack_size = 64);
   for (unsigned i = 1; i < ARRAY_SIZE(shaders); i++) {
      if (components) {
         nir_def *values[3];
         for (unsigned c = 0; c < components; c++)
            values[c] = nir_imm_int(&shaders[i], 111 * i + c);
         nir_rt_return_value_pan(&shaders[i], nir_load_rt_arg_pan(&shaders[i], 32, .base = 4),
            nir_load_rt_arg_pan(&shaders[i], 64, .base = 3), nir_vec(&shaders[i], values, components));
      } else {
         nir_store_global(&shaders[i], nir_imm_int64(&shaders[i], 111 * i),
            nir_imm_int64(&shaders[i], 0x1000), .align_mul = 8);
         nir_rt_return_pan(&shaders[i], nir_load_rt_arg_pan(&shaders[i], 32, .base = 4),
            nir_load_rt_arg_pan(&shaders[i], 64, .base = 3));
      }
   }
   if (partial) {
      shaders[2].cursor = nir_before_impl(shaders[2].impl);
      nir_store_global(&shaders[2], nir_load_rt_arg_pan(&shaders[2], 64, .base = 3),
         nir_imm_int64(&shaders[2], 0x2000), .align_mul = 8);
   }
   struct panvk_nir_rt_pipeline_stage stages[2] = {{.stage = kind}, {.stage = kind}};
   nir_shader *functions[] = {shaders[1].shader, shaders[2].shader};
   struct panvk_nir_rt_pipeline_info info = {
      .native = true, .direct_handles = direct, .stages = stages, .stage_count = 2,
   };
   struct panvk_nir_rt_pipeline_result result = {.nir = shaders[0].shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result) == direct);
   assert(count_calls(shaders[0].shader) == (!direct || partial));
   const uint32_t indices[] = {0, 1, 2, 3, UINT32_MAX};
   for (unsigned n = 0; n < ARRAY_SIZE(indices); n++) {
      nir_shader *copy = nir_shader_clone(NULL, shaders[0].shader);
      uint32_t index = indices[n];
      if (direct && !partial && (index < 1 || index > 2)) {
         ralloc_free(copy);
         continue;
      }
      nir_shader_intrinsics_pass(copy, replace_target_index, nir_metadata_control_flow, &index);
      bool progress;
      do {
         progress = nir_opt_copy_prop(copy);
         progress |= nir_opt_constant_folding(copy);
         progress |= nir_opt_dead_cf(copy);
         progress |= nir_opt_remove_phis(copy);
         progress |= nir_opt_dce(copy);
      } while (progress);
      bool inlined = direct && (index == 1 || (index == 2 && !partial));
      assert(count_calls(copy) == !inlined);
      if (inlined) {
         uint64_t expected = 0, args[13] = {0};
         if (components) {
            for (unsigned c = 0; c < components; c++)
               expected = expected * 17 + 111 * index + c;
         } else {
            expected = 111 * index;
         }
         assert(output_value(copy, args) == expected);
      }
      nir_validate_shader(copy, "RT multiple target specialization");
      ralloc_free(copy);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++) {
      nir_validate_shader(shaders[i].shader, "RT guarded runtime LTO");
      ralloc_free(shaders[i].shader);
   }
}

static void
check_lto_chain(bool cycle)
{
   nir_builder shaders[3];
   for (unsigned i = 0; i < 3; i++)
      shaders[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   for (unsigned i = 0; i < (cycle ? 3 : 2); i++) {
      nir_builder *b = &shaders[i];
      nir_def *f = nir_load_rt_arg_pan(b, 64, .base = 0);
      make_call(b, f, f, f, nir_imm_int(b, 0),
         i == 1 ? MESA_SHADER_CALLABLE : MESA_SHADER_INTERSECTION);
   }
   nir_store_global(&shaders[2], nir_load_rt_arg_pan(&shaders[2], 64, .base = 0),
      nir_imm_int64(&shaders[2], 0x1000), .align_mul = 8);
   struct panvk_nir_rt_pipeline_stage stages[] = {
      {.stage = MESA_SHADER_INTERSECTION}, {.stage = MESA_SHADER_CALLABLE},
   };
   nir_shader *functions[] = {shaders[1].shader, shaders[2].shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = stages, .stage_count = 2};
   struct panvk_nir_rt_pipeline_result result = {.nir = shaders[0].shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result));
   for (unsigned i = 0; i < 3; i++) {
      assert(count_calls(shaders[i].shader) == cycle);
      nir_validate_shader(shaders[i].shader, "LTO chain");
   }
   for (unsigned i = 0; i < 3; i++)
      ralloc_free(shaders[i].shader);
}

static void
check_lto_caller_stage(mesa_shader_stage kind)
{
   nir_builder entry = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   mesa_shader_stage target = kind == MESA_SHADER_ANY_HIT ? MESA_SHADER_MISS : MESA_SHADER_ANY_HIT;
   nir_def *frame = nir_load_rt_arg_pan(&caller, 64, .base = 0);
   make_call(&caller, frame, frame, frame, nir_imm_int(&caller, 0), target);
   nir_store_global(&callee, nir_imm_int64(&callee, 17),
                     nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   struct panvk_nir_rt_pipeline_stage stages[] = {{.stage = kind}, {.stage = target}};
   nir_shader *functions[] = {caller.shader, callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = stages, .stage_count = 2};
   struct panvk_nir_rt_pipeline_result result = {.nir = entry.shader, .functions = functions};
   bool allowed = kind == MESA_SHADER_RAYGEN || kind == MESA_SHADER_INTERSECTION ||
                  kind == MESA_SHADER_ANY_HIT;
   assert(panvk_nir_opt_rt_lto(&info, &result) == allowed);
   assert(count_calls(caller.shader) == !allowed);
   nir_validate_shader(caller.shader, "LTO caller stage guard");
   if (allowed)
      assert(output_value(caller.shader, NULL) == 17);
   ralloc_free(callee.shader);
   ralloc_free(caller.shader);
   ralloc_free(entry.shader);
}

static void
check_lto_boundary(unsigned count)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *zero = nir_imm_int64(&caller, 0);
   make_call(&caller, zero, zero, zero, nir_imm_int(&caller, 0), MESA_SHADER_MISS);
   nir_def *address = nir_imm_int64(&callee, 0x1000);
   nir_def *value = nir_imm_int(&callee, 17);
   for (unsigned i = 2; i < count; i++)
      nir_store_global(&callee, value, address, .align_mul = 4);
   struct panvk_nir_rt_pipeline_stage stage = {.stage = MESA_SHADER_MISS};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = &stage, .stage_count = 1};
   struct panvk_nir_rt_pipeline_result result = {.nir = caller.shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result));
   assert(count_calls(caller.shader) == 0);
   nir_validate_shader(caller.shader, "LTO size boundary");
   ralloc_free(callee.shader);
   ralloc_free(caller.shader);
}

static void
check_lto_variables(void)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_variable *uniform = nir_variable_create(callee.shader, nir_var_uniform,
      glsl_uint_type(), "value");
   nir_store_global(&callee, nir_load_var(&callee, uniform),
      nir_imm_int64(&callee, 0x1000), .align_mul = 4);
   nir_def *zero = nir_imm_int64(&caller, 0);
   make_call(&caller, zero, zero, zero, nir_imm_int(&caller, 0), MESA_SHADER_MISS);
   struct panvk_nir_rt_pipeline_stage stage = {.stage = MESA_SHADER_MISS};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = &stage, .stage_count = 1};
   struct panvk_nir_rt_pipeline_result result = {.nir = caller.shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result));
   assert(!exec_list_is_empty(&caller.shader->variables));
   nir_foreach_variable_in_shader(var, caller.shader)
      assert(var != uniform);
   ralloc_free(callee.shader);
   nir_validate_shader(caller.shader, "LTO shader variable ownership");
   ralloc_free(caller.shader);
}

static void
check_lto_frames(unsigned prefix, unsigned callee_prefix, unsigned calls, bool overflow)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *frame = nir_load_rt_arg_pan(&caller, 64, .base = 0);
   for (unsigned i = 0; i < calls; i++)
      make_call(&caller, frame, frame, frame, nir_imm_int(&caller, 0), MESA_SHADER_CLOSEST_HIT);
   nir_rt_return_pan(&caller, nir_load_rt_arg_pan(&caller, 32, .base = 4),
      nir_load_rt_arg_pan(&caller, 64, .base = 3), .stack_size = prefix);
   frame = nir_load_rt_arg_pan(&callee, 64, .base = 0);
   nir_def *child = nir_iadd(&callee, frame, nir_u2u64(&callee, nir_load_rt_frame_size_pan(&callee)));
   make_call(&callee, child, frame, child, nir_imm_int(&callee, 0), MESA_SHADER_CLOSEST_HIT);
   nir_store_global(&callee, frame, child, .align_mul = 8);
   nir_rt_return_pan(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 4),
      nir_load_rt_arg_pan(&callee, 64, .base = 3), .stack_size = callee_prefix);
   struct panvk_nir_rt_pipeline_stage stage = {.stage = MESA_SHADER_CLOSEST_HIT};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = &stage, .stage_count = 1};
   struct panvk_nir_rt_pipeline_result result = {.nir = caller.shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result) == !overflow);
   assert(count_calls(caller.shader) == calls);
   if (!overflow) {
      uint64_t input[36] = {0};
      const uint64_t bases[] = {0, UINT64_C(0x1fffffff0), UINT64_MAX - 1024};
      for (unsigned n = 0; n < ARRAY_SIZE(bases); n++) {
         input[0] = bases[n];
         input[35] = (uint64_t)prefix + callee_prefix + 512;
         unsigned stores = 0;
         nir_foreach_block(block, caller.impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_global) {
                  stores++;
                  assert(evaluate(intr->src[0].ssa, 0, input).u64 == input[0] + prefix);
                  assert(evaluate(intr->src[1].ssa, 0, input).u64 == input[0] + input[35]);
               } else if (intr->intrinsic == nir_intrinsic_rt_return_pan) {
                  assert(nir_intrinsic_stack_size(intr) == prefix + callee_prefix);
               } else if (intr->intrinsic == nir_intrinsic_rt_call_pan) {
                  uint64_t actual[13] = {0};
                  call_values(intr, input, actual);
                  assert(actual[0] == input[0] + input[35]);
                  assert(actual[5] == input[0] + prefix);
               }
            }
         }
         assert(stores == calls);
      }
   }
   ralloc_free(callee.shader);
   nir_validate_shader(caller.shader, "RT inlined frame ownership");
   ralloc_free(caller.shader);
}

static void
check_lto_call_budget(unsigned calls, bool partial_last, unsigned nested)
{
   nir_builder shaders[3];
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++)
      shaders[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *zero = nir_imm_int64(&shaders[0], 0);
   for (unsigned i = 0; i < calls; i++)
      make_call(&shaders[0], zero, zero, zero, nir_imm_int(&shaders[0], 0), MESA_SHADER_MISS);
   if (partial_last)
      make_call(&shaders[0], zero, zero, zero, nir_imm_int(&shaders[0], 0), MESA_SHADER_CLOSEST_HIT);
   else
      make_call(&shaders[0], zero, zero, zero, nir_imm_int(&shaders[0], 0), MESA_SHADER_MISS);
   nir_store_global(&shaders[1], nir_imm_int(&shaders[1], 1),
      nir_imm_int64(&shaders[1], 0x1000), .align_mul = 4);
   for (unsigned i = 0; i < nested; i++) {
      nir_def *frame = nir_load_rt_arg_pan(&shaders[1], 64, .base = 0);
      make_call(&shaders[1], frame, frame, frame, nir_imm_int(&shaders[1], 0), MESA_SHADER_MISS);
   }
   nir_store_global(&shaders[2], nir_load_rt_arg_pan(&shaders[2], 64, .base = 3),
      nir_imm_int64(&shaders[2], 0x1000), .align_mul = 8);
   struct panvk_nir_rt_pipeline_stage stages[] = {
      {.stage = MESA_SHADER_MISS}, {.stage = MESA_SHADER_CLOSEST_HIT},
   };
   nir_shader *functions[] = {shaders[1].shader, shaders[2].shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = stages, .stage_count = 2};
   struct panvk_nir_rt_pipeline_result result = {.nir = shaders[0].shader, .functions = functions};
   bool allowed = calls + !partial_last + nested <= 25 || (partial_last && !nested);
   assert(panvk_nir_opt_rt_lto(&info, &result) == allowed);
   assert(count_calls(shaders[0].shader) ==
      (allowed ? (calls + !partial_last) * nested + partial_last : calls + 1));
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++) {
      nir_validate_shader(shaders[i].shader, "RT callsite count budget");
      ralloc_free(shaders[i].shader);
   }
}

static void
check_lto_payload(bool escape, bool dead)
{
   nir_builder shaders[3];
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++)
      shaders[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_variable *payload = nir_local_variable_create(shaders[0].impl, glsl_uint_type(), "payload");
   nir_store_var(&shaders[0], payload, nir_imm_int(&shaders[0], 17), 1);
   nir_def *zero = nir_imm_int64(&shaders[0], 0);
   make_call(&shaders[0], zero, nir_u2u64(&shaders[0], &nir_build_deref_var(&shaders[0], payload)->def),
      zero, nir_imm_int(&shaders[0], 0), MESA_SHADER_INTERSECTION);
   nir_store_global(&shaders[0], nir_u2u64(&shaders[0], nir_load_var(&shaders[0], payload)),
      nir_imm_int64(&shaders[0], 0x1000), .align_mul = 8);
   for (unsigned i = 1; i < ARRAY_SIZE(shaders); i++) {
      nir_builder *b = &shaders[i];
      nir_def *address = nir_load_rt_arg_pan(b, 64, .base = 5);
      if (i == 1 || escape) {
         if (i == 2 && dead)
            nir_push_if(b, nir_imm_false(b));
         make_call(b, nir_load_rt_arg_pan(b, 64, .base = 0), address,
            nir_load_rt_arg_pan(b, 64, .base = 2), nir_imm_int(b, 0),
            i == 1 ? MESA_SHADER_ANY_HIT : MESA_SHADER_CALLABLE);
         if (i == 2 && dead)
            nir_pop_if(b, NULL);
      }
      nir_deref_instr *deref = nir_build_deref_cast(b, address, nir_var_mem_global, glsl_uint_type(), 0);
      nir_store_deref(b, deref, nir_iadd_imm(b, nir_load_deref(b, deref), i == 1 ? 7 : 5), 1);
      nir_rt_return_pan(b, nir_load_rt_arg_pan(b, 32, .base = 4),
         nir_load_rt_arg_pan(b, 64, .base = 3));
   }
   struct panvk_nir_rt_pipeline_stage stages[] = {
      {.stage = MESA_SHADER_INTERSECTION}, {.stage = MESA_SHADER_ANY_HIT},
   };
   nir_shader *functions[] = {shaders[1].shader, shaders[2].shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = stages, .stage_count = 2};
   struct panvk_nir_rt_pipeline_result result = {.nir = shaders[0].shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &result));
   panvk_nir_opt_rt_private_io(shaders[0].shader);
   nir_remove_dead_variables(shaders[0].shader, nir_var_function_temp, NULL);
   bool escaped = escape && !dead;
   assert(count_calls(shaders[0].shader) == escaped);
   assert(exec_list_is_empty(&shaders[0].impl->locals) == !escaped);
   if (!escaped) {
      uint64_t input[36] = {0};
      assert(output_value(shaders[0].shader, input) == 29);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++) {
      nir_validate_shader(shaders[i].shader, "RT transitive payload promotion");
      ralloc_free(shaders[i].shader);
   }
}

static void
check_private_io(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   unsigned comps = mode == 5 ? 2 : 1;
   const struct glsl_type *type = comps == 2 ? glsl_uvec2_type() : glsl_uint_type();
   nir_variable *var = nir_local_variable_create(b.impl, type, "payload");
   nir_def *initial = nir_load_rt_arg_pan(&b, 32, .base = 9);
   if (comps == 2)
      initial = nir_vec2(&b, initial, nir_load_rt_arg_pan(&b, 32, .base = 10));
   nir_store_var(&b, var, initial, BITFIELD_MASK(comps));
   nir_deref_instr *source = nir_build_deref_var(&b, var);
   nir_deref_instr *cast = nir_build_deref_cast(&b, nir_u2u64(&b, &source->def),
      nir_var_mem_global, mode == 1 ? glsl_float_type() : type, mode == 4 ? 4 : 0);
   if (mode == 3)
      cast->cast.align_mul = 4;
   if (mode == 2)
      nir_store_global(&b, &cast->def, nir_imm_int64(&b, 0x2000), .align_mul = 8);
   nir_def *value = nir_iadd_imm(&b, nir_load_deref(&b, cast), 17);
   nir_store_deref(&b, cast, value, BITFIELD_MASK(comps));
   value = nir_load_var(&b, var);
   if (comps == 2)
      value = nir_iadd(&b, nir_channel(&b, value, 0), nir_channel(&b, value, 1));
   nir_store_global(&b, nir_u2u64(&b, value), nir_imm_int64(&b, 0x1000), .align_mul = 8);
   nir_validate_shader(b.shader, "RT private IO test input");
   panvk_nir_opt_rt_private_io(b.shader);
   unsigned casts = 0, reloads = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref) {
               nir_deref_instr *d = nir_instr_as_deref(instr);
               casts += d->deref_type == nir_deref_type_cast && d->modes == nir_var_mem_global;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref)
                  reloads += nir_deref_mode_is(nir_src_as_deref(intr->src[0]),
                                               nir_var_function_temp);
            }
         }
      }
   }
   bool allowed = mode == 0 || mode == 5;
   assert(casts == !allowed);
   assert(allowed || reloads);
   if (allowed) {
      for (unsigned n = 0; n < 128; n++) {
         uint64_t input[13] = {0};
         input[9] = UINT32_MAX - n;
         input[10] = n * 0x9e3779b9u;
         uint32_t expected = input[9] + 17;
         if (comps == 2)
            expected += (uint32_t)(input[10] + 17);
         assert(output_value(b.shader, input) == expected);
      }
   }
   ralloc_free(b.shader);
}

static void
check_tls_arguments(unsigned mask)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   const unsigned bases[] = {0, 5, 2, 6};
   nir_def *frame = nir_load_rt_arg_pan(&caller, 64, .base = 0);
   nir_intrinsic_instr *call = make_call(&caller, frame,
      nir_load_rt_arg_pan(&caller, 64, .base = 5),
      nir_load_rt_arg_pan(&caller, 64, .base = 2),
      nir_load_rt_arg_pan(&caller, 32, .base = 11), MESA_SHADER_MISS);
   nir_intrinsic_set_rt_arg_mask(call, mask);
   nir_def *value = nir_imm_int64(&callee, 0);
   for (unsigned a = 0; a < 4; a++)
      value = nir_iadd(&callee, nir_imul_imm(&callee, value, 17),
                       nir_load_rt_arg_pan(&callee, 64, .base = bases[a]));
   nir_store_global(&callee, value, nir_imm_int64(&callee, 0x1000), .align_mul = 8);
   assert(panvk_nir_lower_rt_tls(caller.shader));
   assert(panvk_nir_lower_rt_tls(callee.shader));
   assert(nir_intrinsic_rt_arg_mask(call) == ((mask & ~0xaa) | ((mask & 0xaa) >> 1)));
   assert(nir_intrinsic_rt_call_flags(call) == 1);
   const uint64_t tls[] = {0, 0xfffffff0, UINT64_C(0x12345678ffffff00), UINT64_MAX - 31};
   const uint32_t offsets[] = {0, 1, 16, 65535, UINT32_MAX};
   for (unsigned t = 0; t < ARRAY_SIZE(tls); t++) {
      for (unsigned n = 0; n < ARRAY_SIZE(offsets); n++) {
         uint64_t input[33] = {0}, actual[13] = {0}, expected = 0;
         input[32] = tls[t];
         for (unsigned a = 0; a < 4; a++) {
            unsigned offset = offsets[(n + a) % ARRAY_SIZE(offsets)];
            input[13 + a] = offset;
            expected = expected * 17 + (tls[t] + offset);
         }
         call_values(call, input, actual);
         for (unsigned a = 0; a < 4; a++)
            assert(actual[bases[a]] == input[13 + a]);
         assert(output_value(callee.shader, input) == expected);
      }
   }
   ralloc_free(caller.shader);
   ralloc_free(callee.shader);
}

static void
check_launch_arguments(unsigned mode)
{
   nir_builder shaders[4];
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++)
      shaders[i] = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   const mesa_shader_stage kinds[] = {MESA_SHADER_RAYGEN, MESA_SHADER_CLOSEST_HIT, MESA_SHADER_MISS};
   nir_intrinsic_instr *calls[4];
   for (unsigned i = 0; i < 3; i++)
      calls[i] = make_call(&shaders[i], nir_load_rt_arg_pan(&shaders[i], 64, .base = 0),
         nir_load_rt_arg_pan(&shaders[i], 64, .base = 5),
         nir_load_rt_arg_pan(&shaders[i], 64, .base = 2),
         nir_imm_int(&shaders[i], 0), kinds[i]);
   unsigned mask = mode < 3 ? BITFIELD_BIT(mode) : mode == 4 ? 0 : 7;
   nir_def *checksum = nir_imm_int64(&shaders[3], 0);
   if (mask) {
      nir_def *launch = nir_load_ray_launch_id(&shaders[3]);
      for (unsigned c = 0; c < 3; c++) {
         if (mask & BITFIELD_BIT(c))
            checksum = nir_iadd(&shaders[3], nir_imul_imm(&shaders[3], checksum, 17),
               nir_u2u64(&shaders[3], nir_channel(&shaders[3], launch, c)));
      }
   }
   nir_store_global(&shaders[3], checksum, nir_imm_int64(&shaders[3], 0x1000), .align_mul = 8);
   if (mode >= 6)
      calls[3] = make_call(&shaders[3], nir_load_rt_arg_pan(&shaders[3], 64, .base = 0),
         nir_load_rt_arg_pan(&shaders[3], 64, .base = 5),
         nir_load_rt_arg_pan(&shaders[3], 64, .base = 2),
         nir_imm_int(&shaders[3], 0), mode == 6 ? MESA_SHADER_CALLABLE : kinds[0]);
   struct panvk_nir_rt_pipeline_stage stages[3];
   nir_shader *functions[3];
   for (unsigned i = 0; i < 3; i++) {
      stages[i] = (struct panvk_nir_rt_pipeline_stage){.stage = kinds[i]};
      functions[i] = shaders[i + 1].shader;
   }
   struct panvk_nir_rt_pipeline_info info = {
      .native = true, .tls = mode != 5, .stages = stages, .stage_count = 3,
   };
   struct panvk_nir_rt_pipeline_result result = {.nir = shaders[0].shader, .functions = functions};
   uint64_t expected[32], inputs[32][35] = {0};
   for (unsigned n = 0; n < 32; n++) {
      for (unsigned c = 0; c < 3; c++)
         inputs[n][17 + c] = n == 0 ? 0 : n == 1 ? UINT32_MAX :
            (uint32_t)(UINT64_C(0x9e3779b97f4a7c15) * (n + c));
      expected[n] = output_value(shaders[3].shader, inputs[n]);
   }
   if (info.tls) {
      for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++)
         panvk_nir_lower_rt_tls(shaders[i].shader);
   }
   assert(panvk_nir_lower_rt_launch_id(&info, &result) == (info.tls && mask != 0));
   unsigned propagated = mode == 6 ? 7 : mask;
   for (unsigned n = 0; n < 32; n++) {
      uint64_t actual[35];
      memcpy(actual, inputs[n], sizeof(actual));
      if (info.tls) {
         for (unsigned i = 0; i < 3; i++) {
            uint64_t next[35] = {0};
            for (unsigned c = 0; c < 3; c++) {
               uint64_t word = evaluate(calls[i]->src[1].ssa, c, actual).u64;
               assert((uint32_t)(word >> 32) ==
                  (propagated & BITFIELD_BIT(c) ? inputs[n][17 + c] : 0));
               assert(!!(nir_intrinsic_rt_arg_mask(calls[i]) & BITFIELD_BIT(2 * c + 1)) ==
                  !!(propagated & BITFIELD_BIT(c)));
               next[17 + c] = word >> 32;
            }
            memcpy(actual, next, sizeof(actual));
         }
      }
      assert(output_value(shaders[3].shader, actual) == expected[n]);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(shaders); i++) {
      nir_validate_shader(shaders[i].shader, "RT launch argument chain");
      ralloc_free(shaders[i].shader);
   }
}

static void
check_lto_result(unsigned components, bool private_frame)
{
   nir_builder caller = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_builder callee = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   mesa_shader_stage kind = components == 1 ? MESA_SHADER_ANY_HIT : MESA_SHADER_INTERSECTION;
   nir_def *args[4], *hit[8];
   const unsigned bases[] = {0, 5, 2, 6};
   for (unsigned i = 0; i < 4; i++)
      args[i] = nir_load_rt_arg_pan(&caller, 64, .base = bases[i]);
   for (unsigned i = 0; i < 8; i++)
      hit[i] = nir_load_rt_arg_pan(&caller, 32, .base = 9 + i % 4);
   nir_def *result = nir_rt_call_result_pan(&caller, components,
      nir_imm_int(&caller, 7), nir_vec(&caller, args, 4), nir_vec(&caller, hit, 8),
      nir_imm_int64(&caller, 0), nir_imm_int(&caller, 0), .base = kind, .rt_arg_mask = 0xffff);
   nir_def *checksum = nir_imm_int64(&caller, 0);
   nir_def *values[3];
   for (unsigned i = 0; i < components; i++) {
      checksum = nir_iadd(&caller, nir_imul_imm(&caller, checksum, 17),
         nir_u2u64(&caller, nir_channel(&caller, result, i)));
      values[i] = nir_ixor(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 9 + i),
                          nir_imm_int(&callee, 123 + i));
      if (private_frame)
         values[i] = nir_iadd(&callee, values[i],
            nir_u2u32(&callee, nir_load_rt_arg_pan(&callee, 64, .base = 0)));
   }
   nir_store_global(&caller, checksum, nir_imm_int64(&caller, 0x1000), .align_mul = 8);
   nir_rt_return_pan(&caller, nir_load_rt_arg_pan(&caller, 32, .base = 4),
      nir_load_rt_arg_pan(&caller, 64, .base = 3), .stack_size = 64);
   nir_rt_return_value_pan(&callee, nir_load_rt_arg_pan(&callee, 32, .base = 4),
      nir_load_rt_arg_pan(&callee, 64, .base = 3), nir_vec(&callee, values, components),
      .stack_size = private_frame ? 32 : 0);
   struct panvk_nir_rt_pipeline_stage stage = {.stage = kind};
   nir_shader *functions[] = {callee.shader};
   struct panvk_nir_rt_pipeline_info info = {.native = true, .stages = &stage, .stage_count = 1};
   struct panvk_nir_rt_pipeline_result output = {.nir = caller.shader, .functions = functions};
   assert(panvk_nir_opt_rt_lto(&info, &output));
   assert(count_calls(caller.shader) == 0);
   ralloc_free(callee.shader);
   nir_validate_shader(caller.shader, "LTO typed result");
   uint64_t random = 0x9e3779b97f4a7c15ull;
   for (unsigned n = 0; n < 128; n++) {
      uint64_t input[13] = {0}, expected = 0;
      input[0] = UINT64_C(0x1fffffff0) + n;
      for (unsigned i = 0; i < components; i++) {
         random ^= random << 13;
         random ^= random >> 7;
         random ^= random << 17;
         input[9 + i] = (uint32_t)random;
         uint32_t value = input[9 + i] ^ (123 + i);
         if (private_frame)
            value += input[0] + 64;
         expected = expected * 17 + value;
      }
      assert(output_value(caller.shader, input) == expected);
   }
   ralloc_free(caller.shader);
}

static void
check_private_rt_call(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_variable *var = mode & 1 ?
      nir_variable_create(b.shader, nir_var_shader_temp, glsl_uint_type(), "payload") :
      nir_local_variable_create(b.impl, glsl_uint_type(), "payload");
   nir_store_var(&b, var, nir_imm_int(&b, 0), 1);
   nir_def *payload = nir_u2u64(&b, &nir_build_deref_var(&b, var)->def);
   nir_def *zero = nir_imm_int64(&b, 0);
   nir_intrinsic_instr *call = make_call(&b, zero, payload, zero,
      nir_imm_int(&b, 0), MESA_SHADER_MISS);
   if (mode & 2) {
      b.cursor = nir_before_instr(&call->instr);
      nir_rt_call_result_pan(&b, 1, call->src[0].ssa, call->src[1].ssa,
         call->src[2].ssa, call->src[3].ssa, call->src[4].ssa, .base = MESA_SHADER_ANY_HIT,
         .rt_arg_mask = 0xffff);
      nir_instr_remove(&call->instr);
   }
   b.cursor = nir_after_impl(b.impl);
   nir_store_global(&b, nir_load_var(&b, var), nir_imm_int64(&b, 0x1000),
      .align_mul = 4);
   nir_validate_shader(b.shader, "RT private call input");
   panvk_nir_opt_rt_private_io(b.shader);
   unsigned reloads = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref)
            reloads += nir_deref_mode_is(nir_src_as_deref(intr->src[0]),
                                         var->data.mode);
      }
   }
   assert(reloads == 1);
   ralloc_free(b.shader);
}

static void
check_load_range(unsigned bits, unsigned mask, bool constant)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *base = nir_load_rt_arg_pan(&b, 64, .base = 0);
   nir_def *load = constant ?
      nir_load_global_constant(&b, 4, bits, base, .align_mul = 16) :
      nir_load_global(&b, 4, bits, base, .align_mul = 16);
   nir_def *value = nir_imm_int64(&b, 0);
   for (unsigned i = 0; i < 4; i++) {
      if (mask & BITFIELD_BIT(i))
         value = nir_iadd(&b, nir_imul_imm(&b, value, 17),
                            nir_u2u64(&b, nir_channel(&b, load, i)));
   }
   nir_store_global(&b, value, nir_imm_int64(&b, 0x1000), .align_mul = 8);
   nir_shader *original = nir_shader_clone(NULL, b.shader);
   nir_opt_shrink_vectors(b.shader, true);
   nir_validate_shader(b.shader, "RT load range narrowing");
   for (unsigned i = 0; i < 32; i++) {
      uint64_t args[13] = {0};
      args[0] = (UINT64_C(0xffffffff00000000) +
                 UINT64_C(0x1000000010) * i) & ~UINT64_C(15);
      assert(output_value(b.shader, args) == output_value(original, args));
   }
   ralloc_free(original);
   ralloc_free(b.shader);
}

static void
check_dispatch_params(unsigned offset, unsigned comps, unsigned bits, bool native)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *dispatch = nir_load_rt_arg_pan(&b, 64, .base = 0);
   unsigned sysval = 368;
   nir_def *load = panvk_nir_load_rt_dispatch(&b, dispatch, offset, comps, bits,
                                              native ? sysval : 0);
   bool promoted = native && bits != 16 && offset % (bits / 8) == 0 &&
      ((offset >= 104 && offset + comps * (bits / 8) <= 160) ||
       (offset >= 88 && offset + comps * (bits / 8) <= 100));
   assert(nir_def_as_intrinsic(load)->intrinsic ==
          (promoted ? nir_intrinsic_load_push_constant : nir_intrinsic_load_global_constant));
   if (promoted) {
      unsigned relative = offset >= 104 ? offset - 104 :
                          PANVK_RT_DISPATCH_PARAMS_SIZE + offset - 88;
      assert(nir_intrinsic_base(nir_def_as_intrinsic(load)) == sysval + relative);
      assert(nir_intrinsic_range(nir_def_as_intrinsic(load)) == comps * (bits / 8));
   } else {
      unsigned align = bits == 64 ? 8 : 4;
      assert(nir_intrinsic_align_mul(nir_def_as_intrinsic(load)) == align);
      assert(nir_intrinsic_align_offset(nir_def_as_intrinsic(load)) == offset % align);
   }
   nir_def *value = nir_imm_int64(&b, 0);
   for (unsigned c = 0; c < comps; c++)
      value = nir_iadd(&b, nir_imul_imm(&b, value, 17),
                          nir_u2u64(&b, nir_channel(&b, load, c)));
   nir_store_global(&b, value, nir_imm_int64(&b, 0x1000), .align_mul = 8);
   nir_validate_shader(b.shader, "RT dispatch parameters");
   for (unsigned i = 0; i < 32; i++) {
      uint64_t args[35] = {0};
      args[0] = (UINT64_C(0xffffffff00000000) +
                 UINT64_C(0x1000000010) * i) & ~UINT64_C(15);
      args[33] = sysval;
      args[34] = args[0];
      uint64_t expected = 0;
      for (unsigned c = 0; c < comps; c++) {
         uint64_t address = args[0] + offset + c * (bits / 8);
         uint64_t v = address * UINT64_C(0x9e3779b97f4a7c15) ^ (address >> 32);
         if (bits != 64)
            v &= (UINT64_C(1) << bits) - 1;
         expected = expected * 17 + v;
      }
      assert(output_value(b.shader, args) == expected);
   }
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const uint32_t indices[] = {0, 1, 2};
   assert(panvk_rt_native_shader_indices(indices, 3) ==
          (UINT64_C(1) | (UINT64_C(2) << 21) | (UINT64_C(3) << 42)));
   assert(panvk_rt_native_shader_indices(indices, 0) == 0);
   const uint32_t maximum[] = {PANVK_RT_NATIVE_SHADER_INDEX_MASK - 1, 0, UINT32_MAX};
   assert(panvk_rt_native_shader_indices(maximum, PANVK_RT_NATIVE_SHADER_INDEX_MASK) ==
          (UINT64_C(0x1fffff) | (UINT64_C(1) << 21)));
   assert(panvk_rt_native_shader_indices(maximum, PANVK_RT_NATIVE_SHADER_INDEX_MASK + 1) == 0);
   const nir_op ops[] = {nir_op_iadd, nir_op_isub, nir_op_iand, nir_op_ior,
                         nir_op_ixor, nir_op_ishl, nir_op_ushr, nir_op_ishr};
   for (unsigned mode = 0; mode < 8; mode++) {
      for (unsigned op = 0; op < ARRAY_SIZE(ops); op++) {
         if (mode == 7 && op >= 5)
            continue;
         check(mode, ops[op], 7);
         check(mode, ops[op], UINT64_MAX);
      }
   }
   check_targets(false);
   check_targets(true);
   check_argument_canonicalization();
   const uint64_t chain_constants[] = {0, 1, UINT32_MAX, UINT64_MAX,
      UINT64_C(0x8000000000000000), UINT64_C(0x123456789abcdef0)};
   for (unsigned mode = 0; mode < 4; mode++) {
      for (unsigned i = 0; i < ARRAY_SIZE(chain_constants); i++)
         check_argument_chain(mode, chain_constants[i]);
   }
   for (unsigned mode = 0; mode < 10; mode++)
      check_lto(mode);
   for (unsigned components = 0; components <= 3; components++) {
      if (components == 2)
         continue;
      check_lto_multiple_targets(components, false, true);
      check_lto_multiple_targets(components, true, true);
      check_lto_multiple_targets(components, false, false);
   }
   puts("9 guarded LTO scenarios and 36 valid-target/partial-fallback specializations passed");
   check_lto_chain(false);
   check_lto_chain(true);
   for (mesa_shader_stage s = MESA_SHADER_RAYGEN; s <= MESA_SHADER_CALLABLE; s++)
      check_lto_caller_stage(s);
   check_lto_boundary(255);
   check_lto_boundary(256);
   check_lto_boundary(257);
   check_lto_variables();
   check_lto_frames(64, 224, 1, false);
   check_lto_frames(64, 224, 3, false);
   check_lto_frames(0, 0, 2, false);
   check_lto_frames(UINT32_MAX - 15, 32, 2, true);
   for (unsigned n = 23; n <= 26; n++) {
      check_lto_call_budget(n, false, 0);
      check_lto_call_budget(n, true, 0);
   }
   check_lto_call_budget(23, false, 1);
   check_lto_call_budget(24, false, 1);
   check_lto_call_budget(26, true, 1);
   check_lto_payload(false, false);
   check_lto_payload(true, false);
   check_lto_payload(true, true);
   puts("Non-leaf RT inlining, shared frames, overflow, call budgets and transitive payload promotion passed");
   for (unsigned mode = 0; mode < 8; mode++)
      check_launch_arguments(mode);
   puts("8 RT launch argument scenarios and 256 output comparisons passed");
   check_lto_result(1, false);
   check_lto_result(3, false);
   check_lto_result(1, true);
   check_lto_result(3, true);
   const unsigned masks[] = {0, 1, 2, 3, 0xffff, 0xaaaa};
   for (unsigned i = 0; i < ARRAY_SIZE(masks); i++)
      check_tls_arguments(masks[i]);
   for (unsigned mode = 0; mode < 6; mode++)
      check_private_io(mode);
   for (unsigned mode = 0; mode < 4; mode++)
      check_private_rt_call(mode);
   for (unsigned bits = 16; bits <= 64; bits *= 2) {
      for (unsigned mask = 1; mask < 16; mask++) {
         check_load_range(bits, mask, false);
         check_load_range(bits, mask, true);
      }
   }
   const unsigned offsets[] = {0, 88, 92, 96, 100, 104, 108, 112, 120, 128, 136,
                                144, 152, 156, 160, 168};
   unsigned dispatch_scenarios = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(offsets); i++) {
      for (unsigned bits = 16; bits <= 64; bits *= 2) {
         for (unsigned comps = 1; comps <= 4; comps++) {
            check_dispatch_params(offsets[i], comps, bits, false);
            check_dispatch_params(offsets[i], comps, bits, true);
            dispatch_scenarios += 2;
         }
      }
   }
   printf("%u RT dispatch parameter scenarios passed\n", dispatch_scenarios);
   puts("25 RT propagation/canonicalization scenarios and 1024 output comparisons passed");
   glsl_type_singleton_decref();
   puts("124 RT argument scenarios, 26 LTO scenarios, 120 TLS address scenarios, 10 private IO scenarios and 90 load range scenarios passed");
   return 0;
}
