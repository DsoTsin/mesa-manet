#include "nvk_nir_lower_ray_queries.h"

#include "bvh/nvk_bvh.h"
#include "compiler/spirv/spirv.h"
#include "nir/nir_builder.h"
#include "nir/nir_deref.h"
#include "util/hash_table.h"

enum rq_field {
   rq_as,
   rq_flags,
   rq_mask,
   rq_origin,
   rq_direction,
   rq_tmin,
   rq_active,
   rq_started,
   rq_stack0,
   rq_stack1,
   rq_stack2,
   rq_instance,
   rq_item_address,
   rq_item_count,
   rq_item_range,
   rq_candidate,
   rq_committed,
   rq_field_count,
};

enum rq_hit_field {
   rq_hit_type,
   rq_hit_t,
   rq_hit_barycentrics,
   rq_hit_primitive,
   rq_hit_geometry,
   rq_hit_instance_index,
   rq_hit_instance_node,
   rq_hit_triangle_pointer,
   rq_hit_front_face,
   rq_hit_opaque,
   rq_hit_field_count,
};

static const struct glsl_type *
rq_intersection_type(void)
{
   struct glsl_struct_field fields[rq_hit_field_count] = {
      [rq_hit_type] = { .type = glsl_uint_type(), .name = "type" },
      [rq_hit_t] = { .type = glsl_float_type(), .name = "t" },
      [rq_hit_barycentrics] = { .type = glsl_vec_type(2), .name = "barycentrics" },
      [rq_hit_primitive] = { .type = glsl_uint_type(), .name = "primitive" },
      [rq_hit_geometry] = { .type = glsl_uint_type(), .name = "geometry" },
      [rq_hit_instance_index] = { .type = glsl_uint_type(), .name = "instance_index" },
      [rq_hit_instance_node] = { .type = glsl_uint64_t_type(), .name = "instance_node" },
      [rq_hit_triangle_pointer] = { .type = glsl_uvec_type(2), .name = "triangle_pointer" },
      [rq_hit_front_face] = { .type = glsl_bool_type(), .name = "front_face" },
      [rq_hit_opaque] = { .type = glsl_bool_type(), .name = "opaque" },
   };
   return glsl_struct_type(fields, rq_hit_field_count, "nvk_query_intersection", false);
}

static const struct glsl_type *
rq_state_type(void)
{
   const struct glsl_type *intersection = rq_intersection_type();
   struct glsl_struct_field fields[rq_field_count] = {
      [rq_as] = { .type = glsl_uint64_t_type(), .name = "acceleration_structure" },
      [rq_flags] = { .type = glsl_uint_type(), .name = "flags" },
      [rq_mask] = { .type = glsl_uint_type(), .name = "mask" },
      [rq_origin] = { .type = glsl_vec_type(3), .name = "origin" },
      [rq_direction] = { .type = glsl_vec_type(3), .name = "direction" },
      [rq_tmin] = { .type = glsl_float_type(), .name = "tmin" },
      [rq_active] = { .type = glsl_bool_type(), .name = "active" },
      [rq_started] = { .type = glsl_bool_type(), .name = "started" },
      [rq_stack0] = { .type = glsl_uvec_type(4), .name = "stack0" },
      [rq_stack1] = { .type = glsl_uvec_type(4), .name = "stack1" },
      [rq_stack2] = { .type = glsl_uvec_type(4), .name = "stack2" },
      [rq_instance] = { .type = glsl_uvec_type(4), .name = "instance" },
      [rq_item_address] = { .type = glsl_uint64_t_type(), .name = "item_address" },
      [rq_item_count] = { .type = glsl_uint_type(), .name = "item_count" },
      [rq_item_range] = { .type = glsl_bool_type(), .name = "item_range" },
      [rq_candidate] = { .type = intersection, .name = "candidate" },
      [rq_committed] = { .type = intersection, .name = "committed" },
   };
   return glsl_struct_type(fields, rq_field_count, "nvk_ray_query", false);
}

static nir_def *
rq_load(nir_builder *b, nir_deref_instr *query, enum rq_field field)
{
   return nir_load_deref(b, nir_build_deref_struct(b, query, field));
}

static void
rq_store(nir_builder *b, nir_deref_instr *query, enum rq_field field,
         nir_def *value)
{
   nir_store_deref(b, nir_build_deref_struct(b, query, field), value,
                   BITFIELD_MASK(value->num_components));
}

static nir_deref_instr *
rq_hit(nir_builder *b, nir_deref_instr *query, bool committed)
{
   return nir_build_deref_struct(b, query, committed ? rq_committed : rq_candidate);
}

static nir_def *
rq_hit_load(nir_builder *b, nir_deref_instr *hit, enum rq_hit_field field)
{
   return nir_load_deref(b, nir_build_deref_struct(b, hit, field));
}

static void
rq_hit_store(nir_builder *b, nir_deref_instr *hit, enum rq_hit_field field,
             nir_def *value)
{
   nir_store_deref(b, nir_build_deref_struct(b, hit, field), value,
                   BITFIELD_MASK(value->num_components));
}

static nir_def *
rq_global(nir_builder *b, nir_def *address, unsigned components,
           unsigned bit_size, unsigned offset)
{
   return nir_load_global(b, components, bit_size,
                          nir_iadd_imm(b, address, offset), .align_mul = 4,
                          .access = ACCESS_NON_WRITEABLE);
}

static nir_def *
rq_instance_data(nir_builder *b, nir_deref_instr *query, nir_def *instance_index)
{
   nir_def *as = rq_load(b, query, rq_as);
   nir_def *offset = rq_global(b, as, 1, 64, 48);
   nir_def *unit = nir_imul_imm(b, nir_ushr_imm(b, instance_index, 4),
                              NVK_BVH_INSTANCE_DATA_UNIT_SIZE);
   return nir_iadd(b, nir_iadd(b, as, offset), nir_u2u64(b, unit));
}

static nir_def *
rq_instance_slot(nir_builder *b, nir_def *instance_index)
{
   return nir_iand_imm(b, instance_index, NVK_BVH_INSTANCE_DATA_UNIT_SLOTS - 1);
}

static nir_def *
rq_instance_blas(nir_builder *b, nir_deref_instr *query, nir_def *instance_index)
{
   nir_def *unit = rq_instance_data(b, query, instance_index);
   nir_def *slot = rq_instance_slot(b, instance_index);
   nir_def *address = nir_iadd(b, unit, nir_u2u64(b, nir_ishl_imm(b, slot, 3)));
   return rq_global(b, address, 1, 64, 128);
}

static nir_def *
rq_instance_flags(nir_builder *b, nir_deref_instr *query, nir_def *instance_index)
{
   nir_def *unit = rq_instance_data(b, query, instance_index);
   nir_def *slot = rq_instance_slot(b, instance_index);
   nir_def *address = nir_iadd(b, unit, nir_u2u64(b, nir_ishl_imm(b, slot, 3)));
   return nir_iand_imm(b, nir_ushr_imm(b, rq_global(b, address, 1, 32, 0), 8), 0xff);
}

static void
rq_set_instance(nir_builder *b, nir_deref_instr *query, nir_deref_instr *candidate)
{
   nir_def *instance = rq_load(b, query, rq_instance);
   nir_def *index = nir_iadd_imm(b, nir_channel(b, instance, 0), -1);
   nir_def *as = rq_load(b, query, rq_as);
   nir_def *offset = rq_global(b, as, 1, 64, 56);
   nir_def *unit = rq_instance_data(b, query, index);
   nir_def *slot = rq_instance_slot(b, index);
   nir_def *metadata = nir_iadd(b, unit,
                               nir_u2u64(b, nir_ishl_imm(b, slot, 3)));
   nir_def *rank = rq_global(b, metadata, 1, 32, 1024);
   nir_def *node = nir_iadd(b, nir_iadd(b, as, offset),
                           nir_ishl_imm(b, nir_u2u64(b, rank), 6));
   rq_hit_store(b, candidate, rq_hit_instance_index, index);
   rq_hit_store(b, candidate, rq_hit_instance_node, node);
}

static void
rq_commit(nir_builder *b, nir_deref_instr *query, nir_def *t)
{
   nir_deref_instr *candidate = rq_hit(b, query, false);
   nir_deref_instr *committed = rq_hit(b, query, true);
   for (unsigned field = 0; field < rq_hit_field_count; field++)
      rq_hit_store(b, committed, field, rq_hit_load(b, candidate, field));
   rq_hit_store(b, committed, rq_hit_t, t);
   nir_def *terminate = nir_test_mask(b, rq_load(b, query, rq_flags),
                                     SpvRayFlagsTerminateOnFirstHitKHRMask);
   nir_push_if(b, terminate);
   rq_store(b, query, rq_active, nir_imm_false(b));
   rq_store(b, query, rq_item_count, nir_imm_int(b, 0));
   rq_store(b, query, rq_item_range, nir_imm_false(b));
   nir_pop_if(b, NULL);
}

static void
rq_initialize(nir_builder *b, nir_deref_instr *query, nir_intrinsic_instr *intr)
{
   nir_def *as = intr->src[1].ssa;
   if (as->num_components == 2)
      as = nir_pack_64_2x32(b, as);
   as = nir_u2u64(b, as);
   rq_store(b, query, rq_as, as);
   rq_store(b, query, rq_flags, intr->src[2].ssa);
   rq_store(b, query, rq_mask, nir_iand_imm(b, intr->src[3].ssa, 0xff));
   rq_store(b, query, rq_origin, intr->src[4].ssa);
   rq_store(b, query, rq_tmin, intr->src[5].ssa);
   rq_store(b, query, rq_direction, intr->src[6].ssa);
   rq_store(b, query, rq_started, nir_imm_false(b));
   rq_store(b, query, rq_active, nir_imm_false(b));
   rq_store(b, query, rq_item_address, nir_imm_int64(b, 0));
   rq_store(b, query, rq_item_count, nir_imm_int(b, 0));
   rq_store(b, query, rq_item_range, nir_imm_false(b));
   rq_store(b, query, rq_stack0, nir_imm_zero(b, 4, 32));
   rq_store(b, query, rq_stack1, nir_imm_zero(b, 4, 32));
   rq_store(b, query, rq_stack2, nir_imm_zero(b, 4, 32));
   rq_store(b, query, rq_instance, nir_imm_zero(b, 4, 32));
   nir_deref_instr *candidate = rq_hit(b, query, false);
   nir_deref_instr *committed = rq_hit(b, query, true);
   rq_hit_store(b, candidate, rq_hit_type, nir_imm_int(b, 0));
   rq_hit_store(b, committed, rq_hit_type, nir_imm_int(b, 0));
   rq_hit_store(b, committed, rq_hit_t, intr->src[7].ssa);
   nir_def *valid = nir_iand(b, nir_ine_imm(b, as, 0),
                            nir_ine_imm(b, rq_load(b, query, rq_mask), 0));
   nir_push_if(b, valid);
   nir_def *root = nir_iadd(b, as, rq_global(b, as, 1, 64, 32));
   nir_def *root_words = nir_unpack_64_2x32(b, root);
   rq_store(b, query, rq_stack0,
            nir_vec4(b, nir_channel(b, root_words, 0),
                        nir_ior_imm(b, nir_channel(b, root_words, 1), 0x10000000),
                        nir_imm_int(b, 0), nir_imm_int(b, 0)));
   rq_store(b, query, rq_active,
            nir_ine_imm(b, rq_global(b, as, 1, 32,
                                     NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET), 0));
   nir_pop_if(b, NULL);
}

static nir_def *
rq_opacity(nir_builder *b, nir_deref_instr *query, nir_def *geometry,
            nir_def *instance_index)
{
   nir_def *flags = rq_load(b, query, rq_flags);
   nir_def *instance_flags = rq_instance_flags(b, query, instance_index);
   nir_def *opaque = nir_test_mask(b, geometry, 0x80000000);
   opaque = nir_bcsel(b, nir_test_mask(b, instance_flags, 8), nir_imm_false(b), opaque);
   opaque = nir_bcsel(b, nir_test_mask(b, instance_flags, 4), nir_imm_true(b), opaque);
   opaque = nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsNoOpaqueKHRMask),
                      nir_imm_false(b), opaque);
   return nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsOpaqueKHRMask),
                       nir_imm_true(b), opaque);
}

static void
rq_next_item(nir_builder *b, nir_deref_instr *query, nir_variable *result)
{
   nir_deref_instr *candidate = rq_hit(b, query, false);
   nir_def *address = rq_load(b, query, rq_item_address);
   nir_def *metadata = rq_global(b, address, 2, 32, 0);
   rq_store(b, query, rq_item_address, nir_iadd_imm(b, address, 8));
   rq_store(b, query, rq_item_count,
            nir_iadd_imm(b, rq_load(b, query, rq_item_count), -1));
   rq_set_instance(b, query, candidate);
   nir_def *geometry = nir_channel(b, metadata, 1);
   nir_def *opaque = rq_opacity(b, query, geometry,
                                rq_hit_load(b, candidate, rq_hit_instance_index));
   nir_def *flags = rq_load(b, query, rq_flags);
   nir_def *culled = nir_bcsel(b, opaque,
      nir_test_mask(b, flags, SpvRayFlagsCullOpaqueKHRMask),
      nir_test_mask(b, flags, SpvRayFlagsCullNoOpaqueKHRMask));
   nir_push_if(b, nir_inot(b, culled));
   rq_hit_store(b, candidate, rq_hit_type, nir_imm_int(b, 2));
   rq_hit_store(b, candidate, rq_hit_primitive, nir_channel(b, metadata, 0));
   rq_hit_store(b, candidate, rq_hit_geometry, geometry);
   rq_hit_store(b, candidate, rq_hit_opaque, opaque);
   nir_store_var(b, result, nir_imm_true(b), 1);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
}

static void
rq_drain_item_ranges(nir_builder *b, nir_deref_instr *query)
{
   nir_push_loop(b);
   nir_def *stack1 = rq_load(b, query, rq_stack1);
   nir_def *stack2 = rq_load(b, query, rq_stack2);
   nir_def *metadata = nir_channel(b, stack1, 1);
   nir_def *pointer = nir_iand_imm(b, nir_ushr_imm(b, metadata, 28), 7);
   nir_def *second = nir_ieq_imm(b, pointer, 2);
   nir_def *entry_lo = nir_bcsel(b, second, nir_channel(b, stack2, 0),
                                             nir_channel(b, stack1, 2));
   nir_def *entry_hi = nir_bcsel(b, second, nir_channel(b, stack2, 1),
                                             nir_channel(b, stack1, 3));
   nir_def *has_entries = nir_test_mask(b, metadata, 0x0e000000);
   nir_def *next_range = nir_ieq_imm(b,
      nir_iand_imm(b, nir_ushr_imm(b, entry_hi, 17), 0xf8), 0x28);
   nir_def *available = nir_iand(b, has_entries,
      nir_iand(b, nir_ine_imm(b, pointer, 3), next_range));
   nir_push_if(b, nir_inot(b, available));
   rq_store(b, query, rq_item_range, nir_imm_false(b));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);

   nir_def *updated_metadata = nir_iadd_imm(b, metadata, 0x0e000000);
   rq_store(b, query, rq_stack1,
      nir_vec4(b, nir_channel(b, stack1, 0), updated_metadata,
                  nir_channel(b, stack1, 2), nir_channel(b, stack1, 3)));
   nir_def *within_range = nir_fge(b,
      rq_hit_load(b, rq_hit(b, query, true), rq_hit_t), entry_lo);
   nir_push_if(b, within_range);
   nir_def *base = nir_pack_64_2x32(b,
      nir_vec2(b, nir_channel(b, stack1, 0),
                  nir_iand_imm(b, updated_metadata, 0x7fff)));
   nir_def *offset = nir_ishr_imm(b, nir_ishl_imm(b, entry_hi, 21), 21);
   nir_def *address = nir_iadd(b, nir_ishl_imm(b, base, 3),
      nir_i2i64(b, nir_ishl_imm(b, offset, 3)));
   nir_def *count = nir_iand_imm(b, nir_ushr_imm(b, entry_hi, 11), 0x3f);
   rq_store(b, query, rq_item_address, address);
   rq_store(b, query, rq_item_count, count);
   nir_push_if(b, nir_ine_imm(b, count, 0));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
}

struct rq_ttu_result {
   nir_def *hit;
   nir_def *triangle_pointer;
};

static struct rq_ttu_result
rq_traverse(nir_builder *b, nir_deref_instr *query, bool opacity_micromap)
{
   nir_def *flags = rq_load(b, query, rq_flags);
   nir_def *mask = rq_load(b, query, rq_mask);
   nir_def *primary = nir_ior_imm(b,
      nir_ior(b, nir_ishl_imm(b, mask, 24),
         nir_ior(b, nir_iand_imm(b, flags, 0x30),
                    nir_iand_imm(b, nir_ishl_imm(b, flags, 5), 0x80))),
      0x30207);
   nir_def *secondary = nir_ior_imm(b, nir_iand_imm(b, primary, 0xff0308b7), 0x400);
   nir_def *modes = nir_ior_imm(b,
      nir_ior(b,
         nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsCullOpaqueKHRMask),
                    nir_imm_int(b, 1), nir_iand_imm(b, flags, 2)),
         nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsCullNoOpaqueKHRMask),
                    nir_imm_int(b, 16),
                    nir_ishl_imm(b, nir_iand_imm(b, flags, 1), 5))),
      0x50400400);
   modes = nir_ior(b, modes, opacity_micromap ?
      nir_bcsel(b,
         nir_test_mask(b, flags, SpvRayFlagsForceOpacityMicromap2StateEXTMask),
         nir_imm_int(b, 0x05000000), nir_imm_int(b, 0)) :
      nir_imm_int(b, 0x0a000000));
   modes = nir_ior(b, modes,
      nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsSkipTrianglesKHRMask),
                 nir_imm_int(b, 0x5000), nir_imm_int(b, 0)));
   modes = nir_ior(b, modes,
      nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsSkipAABBsKHRMask),
                 nir_imm_int(b, 0x50000), nir_imm_int(b, 0)));
   nir_def *parameters = nir_ishl_imm(b, mask, 16);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *primary_words = nir_vec4(b, primary, nir_ior_imm(b, modes, 0x55000), parameters, zero);
   nir_def *secondary_words = nir_vec4(b, secondary, modes, parameters, zero);
   nir_def *origin = rq_load(b, query, rq_origin);
   nir_def *direction = rq_load(b, query, rq_direction);
   nir_def *origin_tmin = nir_vec4(b, nir_channel(b, origin, 0),
      nir_channel(b, origin, 1), nir_channel(b, origin, 2), rq_load(b, query, rq_tmin));
   nir_def *direction_tmax = nir_vec4(b, nir_channel(b, direction, 0),
      nir_channel(b, direction, 1), nir_channel(b, direction, 2),
      rq_hit_load(b, rq_hit(b, query, true), rq_hit_t));
   nir_def *restore = rq_load(b, query, rq_started);
   nir_def *initialize = nir_inot(b, restore);
   nir_def *always = nir_imm_true(b);
   nir_def *stack0 = rq_load(b, query, rq_stack0);
   nir_def *stack1 = rq_load(b, query, rq_stack1);
   nir_def *stack2 = rq_load(b, query, rq_stack2);

   nir_ttu_open_nv(b, .flags = 1);
   nir_ttu_store_nv(b, always, primary_words, .base = 0);
   nir_ttu_store_nv(b, always, origin_tmin, .base = 64);
   nir_ttu_store_nv(b, always, direction_tmax, .base = 80);
   nir_ttu_store_nv(b, always, secondary_words, .base = 128);
   nir_ttu_store_nv(b, restore, stack0, .base = 256);
   nir_ttu_store_nv(b, restore, stack1, .base = 384);
   nir_ttu_store_nv(b, restore, stack2, .base = 400);
   nir_ttu_store_nv(b, initialize, stack0, .base = 512);
   nir_ttu_go_nv(b);
   nir_def *hit = nir_ttu_load_nv(b, .base = 768);
   nir_def *triangle_pointer = nir_ttu_load_nv(b, .base = 784);
   nir_def *instance = nir_ttu_load_nv(b, .base = 800);
   nir_def *next_stack0 = nir_ttu_load_nv(b, .base = 896);
   nir_def *next_stack1 = nir_ttu_load_nv(b, .base = 960);
   nir_def *next_stack2 = nir_ttu_load_nv(b, .base = 976, .flags = 1);

   rq_store(b, query, rq_stack0, next_stack0);
   rq_store(b, query, rq_stack1, next_stack1);
   rq_store(b, query, rq_stack2, next_stack2);
   rq_store(b, query, rq_instance, instance);
   rq_store(b, query, rq_started, nir_imm_true(b));
   rq_store(b, query, rq_active, nir_ine_imm(b, nir_channel(b, next_stack0, 1), 0));
   return (struct rq_ttu_result) { .hit = hit, .triangle_pointer = triangle_pointer };
}

static nir_def *
rq_proceed(nir_builder *b, nir_deref_instr *query, bool opacity_micromap)
{
   nir_variable *result = nir_local_variable_create(b->impl, glsl_bool_type(), "query_has_candidate");
   nir_store_var(b, result, nir_imm_false(b), 1);
   nir_deref_instr *candidate = rq_hit(b, query, false);
   rq_hit_store(b, candidate, rq_hit_type, nir_imm_int(b, 0));
   nir_loop *loop = nir_push_loop(b);
   nir_loop_add_continue_construct(loop);
   nir_push_if(b, nir_iand(b, rq_load(b, query, rq_item_range),
      nir_ieq_imm(b, rq_load(b, query, rq_item_count), 0)));
   rq_drain_item_ranges(b, query);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_ine_imm(b, rq_load(b, query, rq_item_count), 0));
   rq_next_item(b, query, result);
   nir_jump(b, nir_jump_continue);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_inot(b, rq_load(b, query, rq_active)));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   struct rq_ttu_result traversal = rq_traverse(b, query, opacity_micromap);
   nir_def *hit = traversal.hit;
   nir_def *kind = nir_channel(b, hit, 3);
   nir_push_if(b, nir_ige_imm(b, kind, 0));
   rq_set_instance(b, query, candidate);
   nir_def *instance_index = rq_hit_load(b, candidate, rq_hit_instance_index);
   nir_def *blas = rq_instance_blas(b, query, instance_index);
   nir_def *primitive_offset = rq_global(b, blas, 1, 64, NVK_BVH_PRIMITIVE_DATA_OFFSET);
   nir_def *primitive_address = nir_iadd(b, nir_iadd(b, blas, primitive_offset),
      nir_ishl_imm(b, nir_u2u64(b, nir_channel(b, hit, 0)), 3));
   nir_def *metadata = rq_global(b, primitive_address, 2, 32, 0);
   nir_def *t = nir_iand_imm(b, nir_channel(b, hit, 1), 0x7fffffff);
   nir_def *opaque = nir_ige_imm(b, nir_channel(b, hit, 1), 0);
   rq_hit_store(b, candidate, rq_hit_type, nir_imm_int(b, 1));
   rq_hit_store(b, candidate, rq_hit_t, t);
   rq_hit_store(b, candidate, rq_hit_barycentrics,
      nir_iand_imm(b, nir_channels(b, hit, 0xc), 0x7fffffff));
   rq_hit_store(b, candidate, rq_hit_primitive, nir_channel(b, metadata, 0));
   rq_hit_store(b, candidate, rq_hit_geometry, nir_channel(b, metadata, 1));
   rq_hit_store(b, candidate, rq_hit_triangle_pointer,
      nir_channels(b, traversal.triangle_pointer, 0x3));
   rq_hit_store(b, candidate, rq_hit_front_face,
      nir_ige_imm(b, nir_channel(b, hit, 2), 0));
   rq_hit_store(b, candidate, rq_hit_opaque, opaque);
   nir_push_if(b, opaque);
   rq_commit(b, query, t);
   nir_push_else(b, NULL);
   nir_store_var(b, result, nir_imm_true(b), 1);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   nir_def *item_range = nir_ieq_imm(b, nir_iand_imm(b, kind, 0xfe000000), 0xa0000000);
   nir_push_if(b, item_range);
   nir_def *address = nir_pack_64_2x32(b, nir_vec2(b, nir_channel(b, hit, 0),
      nir_iand_imm(b, nir_channel(b, hit, 1), 0x7fff)));
   rq_store(b, query, rq_item_address, nir_ishl_imm(b, address, 3));
   rq_store(b, query, rq_item_count, nir_iand_imm(b, kind, 0x3f));
   rq_store(b, query, rq_item_range, nir_imm_true(b));
   nir_push_else(b, NULL);
   rq_store(b, query, rq_active, nir_imm_false(b));
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
   return nir_load_var(b, result);
}

static nir_def *
rq_matrix_column(nir_builder *b, nir_deref_instr *query, nir_deref_instr *hit,
                  bool object_to_world, unsigned column)
{
   nir_def *address;
   if (object_to_world) {
      nir_def *index = rq_hit_load(b, hit, rq_hit_instance_index);
      address = nir_iadd_imm(b, rq_instance_data(b, query, index), 256);
      address = nir_iadd(b, address,
         nir_u2u64(b, nir_imul_imm(b, rq_instance_slot(b, index), 48)));
   } else {
      address = nir_iadd_imm(b, rq_hit_load(b, hit, rq_hit_instance_node), 16);
   }
   nir_def *components[3];
   for (unsigned row = 0; row < 3; row++)
      components[row] = rq_global(b, address, 1, 32, row * 16 + column * 4);
   return nir_vec(b, components, 3);
}

static nir_def *
rq_transform_ray(nir_builder *b, nir_deref_instr *query, nir_deref_instr *hit,
                  bool origin)
{
   nir_def *ray = rq_load(b, query, origin ? rq_origin : rq_direction);
   nir_def *result = origin ? rq_matrix_column(b, query, hit, false, 3)
                           : nir_imm_zero(b, 3, 32);
   for (unsigned column = 0; column < 3; column++)
      result = nir_fadd(b, result,
         nir_fmul(b, rq_matrix_column(b, query, hit, false, column),
                     nir_channel(b, ray, column)));
   return result;
}

static nir_def *
rq_triangle_position(nir_builder *b, nir_deref_instr *hit, unsigned vertex)
{
   nir_def *pointer = rq_hit_load(b, hit, rq_hit_triangle_pointer);
   nir_def *request = nir_vec4(b,
      nir_iand_imm(b, nir_channel(b, pointer, 0), 0xffffff80),
      nir_channel(b, pointer, 1), nir_imm_int(b, 0), nir_imm_int(b, 0));
   nir_def *zero = nir_imm_zero(b, 4, 32);
   nir_def *always = nir_imm_true(b);
   nir_ttu_open_nv(b);
   nir_ttu_store_nv(b, nir_imm_false(b), zero, .base = 0);
   nir_ttu_store_nv(b, always, request, .base = 640);
   nir_ttu_go_nv(b);
   nir_def *vertices0 = nir_ttu_load_nv(b, .base = 832);
   nir_def *vertices1 = nir_ttu_load_nv(b, .base = 848);
   nir_def *vertices2 = nir_ttu_load_nv(b, .base = 864, .flags = 1);
   nir_def *all[9] = {
      nir_channel(b, vertices0, 0), nir_channel(b, vertices0, 1),
      nir_channel(b, vertices0, 2), nir_channel(b, vertices0, 3),
      nir_channel(b, vertices1, 0), nir_channel(b, vertices1, 1),
      nir_channel(b, vertices1, 2), nir_channel(b, vertices1, 3),
      nir_channel(b, vertices2, 0),
   };
   return nir_vec(b, &all[vertex * 3], 3);
}

static nir_def *
rq_load_value(nir_builder *b, nir_deref_instr *query, nir_intrinsic_instr *intr)
{
   const bool committed = nir_intrinsic_committed(intr);
   const unsigned column = nir_intrinsic_column(intr);
   nir_deref_instr *hit = rq_hit(b, query, committed);
   switch (nir_intrinsic_ray_query_value(intr)) {
   case nir_ray_query_value_flags:
      return rq_load(b, query, rq_flags);
   case nir_ray_query_value_tmin:
      return rq_load(b, query, rq_tmin);
   case nir_ray_query_value_world_ray_origin:
      return rq_load(b, query, rq_origin);
   case nir_ray_query_value_world_ray_direction:
      return rq_load(b, query, rq_direction);
   case nir_ray_query_value_intersection_type: {
      nir_def *type = rq_hit_load(b, hit, rq_hit_type);
      return committed ? type : nir_iadd_imm(b, type, -1);
   }
   case nir_ray_query_value_intersection_t:
      return rq_hit_load(b, hit, rq_hit_t);
   case nir_ray_query_value_intersection_barycentrics:
      return rq_hit_load(b, hit, rq_hit_barycentrics);
   case nir_ray_query_value_intersection_front_face:
      return rq_hit_load(b, hit, rq_hit_front_face);
   case nir_ray_query_value_intersection_candidate_aabb_opaque:
      return rq_hit_load(b, hit, rq_hit_opaque);
   case nir_ray_query_value_intersection_geometry_index:
      return nir_iand_imm(b, rq_hit_load(b, hit, rq_hit_geometry), 0x0fffffff);
   case nir_ray_query_value_intersection_primitive_index:
      return rq_hit_load(b, hit, rq_hit_primitive);
   case nir_ray_query_value_intersection_instance_id:
      return rq_hit_load(b, hit, rq_hit_instance_index);
   case nir_ray_query_value_intersection_instance_custom_index:
   case nir_ray_query_value_intersection_instance_sbt_index: {
      nir_def *index = rq_hit_load(b, hit, rq_hit_instance_index);
      nir_def *unit = rq_instance_data(b, query, index);
      nir_def *slot = nir_ishl_imm(b, rq_instance_slot(b, index), 3);
      nir_def *address = nir_iadd(b, unit, nir_u2u64(b, slot));
      unsigned offset = nir_intrinsic_ray_query_value(intr) ==
         nir_ray_query_value_intersection_instance_custom_index ? 1028 : 4;
      return nir_iand_imm(b, rq_global(b, address, 1, 32, offset), 0xffffff);
   }
   case nir_ray_query_value_intersection_object_ray_origin:
      return rq_transform_ray(b, query, hit, true);
   case nir_ray_query_value_intersection_object_ray_direction:
      return rq_transform_ray(b, query, hit, false);
   case nir_ray_query_value_intersection_object_to_world:
      return rq_matrix_column(b, query, hit, true, column);
   case nir_ray_query_value_intersection_world_to_object:
      return rq_matrix_column(b, query, hit, false, column);
   case nir_ray_query_value_intersection_triangle_vertex_positions:
      return rq_triangle_position(b, hit, column);
   default:
      UNREACHABLE("Invalid ray query value");
   }
}

static nir_deref_instr *
rq_deref(nir_builder *b, nir_def *query, struct hash_table *variables)
{
   nir_deref_path path;
   nir_deref_path_init(&path, nir_def_as_deref(query), NULL);
   assert(path.path[0]->deref_type == nir_deref_type_var);
   struct hash_entry *entry = _mesa_hash_table_search(variables, path.path[0]->var);
   assert(entry);
   nir_deref_instr *deref = nir_build_deref_var(b, entry->data);
   for (nir_deref_instr **element = &path.path[1]; *element; element++) {
      assert((*element)->deref_type == nir_deref_type_array);
      deref = nir_build_deref_array(b, deref, (*element)->arr.index.ssa);
   }
   nir_deref_path_finish(&path);
   return deref;
}

static void
rq_create_variable(nir_shader *shader, nir_function_impl *impl,
                    nir_variable *variable, struct hash_table *variables)
{
   const struct glsl_type *type = glsl_type_wrap_in_arrays(rq_state_type(), variable->type);
   nir_variable *state = impl ? nir_local_variable_create(impl, type, "nvk_query")
      : nir_variable_create(shader, nir_var_shader_temp, type, "nvk_query");
   _mesa_hash_table_insert(variables, variable, state);
}

bool
nvk_nir_lower_ray_queries_with_omm(nir_shader *shader, bool opacity_micromap)
{
   struct hash_table *variables = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_variable_in_shader(variable, shader) {
      if (variable->data.ray_query)
         rq_create_variable(shader, NULL, variable, variables);
   }
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_function_temp_variable(variable, impl) {
         if (variable->data.ray_query)
            rq_create_variable(shader, impl, variable, variables);
      }
   }
   if (variables->entries == 0) {
      ralloc_free(variables);
      return false;
   }
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(intr->intrinsic))
               continue;
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *query = rq_deref(&b, intr->src[0].ssa, variables);
            nir_def *replacement = NULL;
            switch (intr->intrinsic) {
            case nir_intrinsic_rq_initialize:
               rq_initialize(&b, query, intr);
               break;
            case nir_intrinsic_rq_proceed:
               replacement = rq_proceed(&b, query, opacity_micromap);
               break;
            case nir_intrinsic_rq_confirm_intersection:
               rq_commit(&b, query, rq_hit_load(&b, rq_hit(&b, query, false), rq_hit_t));
               break;
            case nir_intrinsic_rq_generate_intersection: {
               nir_def *t = intr->src[1].ssa;
               nir_def *in_range = nir_iand(&b,
                  nir_fge(&b, t, rq_load(&b, query, rq_tmin)),
                  nir_fge(&b, rq_hit_load(&b, rq_hit(&b, query, true), rq_hit_t), t));
               nir_push_if(&b, in_range);
               rq_commit(&b, query, t);
               nir_pop_if(&b, NULL);
               break;
            }
            case nir_intrinsic_rq_terminate:
               rq_store(&b, query, rq_active, nir_imm_false(&b));
               rq_store(&b, query, rq_item_count, nir_imm_int(&b, 0));
               rq_store(&b, query, rq_item_range, nir_imm_false(&b));
               break;
            case nir_intrinsic_rq_load:
               replacement = rq_load_value(&b, query, intr);
               break;
            default:
               UNREACHABLE("Invalid ray query intrinsic");
            }
            if (replacement)
               nir_def_rewrite_uses(&intr->def, replacement);
            nir_instr_remove(instr);
         }
      }
      nir_progress(true, impl, nir_metadata_none);
   }
   ralloc_free(variables);
   NIR_PASS(_, shader, nir_lower_continue_constructs);
   NIR_PASS(_, shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   return true;
}

bool
nvk_nir_lower_ray_queries(nir_shader *shader)
{
   return nvk_nir_lower_ray_queries_with_omm(shader, true);
}
