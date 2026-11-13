#include "nvk_nir_ser.h"
#include "nir_builder.h"
#include "nir_deref.h"

struct ser_lower_options {
   bool reorder;
   bool triangle_positions;
};

static nir_deref_instr *
ser_word(nir_builder *b, nir_def *object, unsigned byte)
{
   nir_deref_instr *parent = nir_src_as_deref(nir_src_for_ssa(object));
   nir_deref_instr *array = nir_build_deref_cast(b, object, parent->modes,
      glsl_array_type(glsl_uint_type(), NVK_HIT_OBJECT_SIZE / 4, 4), 4);
   return nir_build_deref_array_imm(b, array, byte / 4);
}

static nir_def *
ser_load(nir_builder *b, nir_def *object, unsigned byte, unsigned components)
{
   nir_def *values[4];
   for (unsigned i = 0; i < components; i++)
      values[i] = nir_load_deref(b, ser_word(b, object, byte + i * 4));
   return nir_vec(b, values, components);
}

static void
ser_store(nir_builder *b, nir_def *object, unsigned byte, nir_def *value)
{
   for (unsigned i = 0; i < value->num_components; i++)
      nir_store_deref(b, ser_word(b, object, byte + i * 4), nir_channel(b, value, i), 1);
}

static void
ser_attributes(nir_builder *b, nir_def *object, nir_deref_instr *attribute,
               unsigned byte, bool to_object)
{
   const struct glsl_type *type = attribute->type;
   if (glsl_type_is_struct(type)) {
      unsigned offset = 0;
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         const struct glsl_type *field = glsl_get_struct_field(type, i);
         unsigned size, alignment;
         glsl_get_natural_size_align_bytes(field, &size, &alignment);
         offset = ALIGN_POT(offset, alignment);
         ser_attributes(b, object, nir_build_deref_struct(b, attribute, i), byte + offset, to_object);
         offset += size;
      }
      return;
   }
   if (glsl_type_is_array_or_matrix(type)) {
      unsigned size, alignment;
      glsl_get_natural_size_align_bytes(glsl_get_array_element(type), &size, &alignment);
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         ser_attributes(b, object, nir_build_deref_array_imm(b, attribute, i),
                        byte + i * ALIGN_POT(size, alignment), to_object);
      return;
   }
   unsigned components = glsl_get_vector_elements(type);
   unsigned bits = glsl_type_is_boolean(type) ? 32 : glsl_get_bit_size(type);
   assert(byte + components * bits / 8 <= 32);
   if (to_object) {
      nir_def *value = nir_load_deref(b, attribute);
      if (value->bit_size == 1)
         value = nir_b2i32(b, value);
      for (unsigned i = 0; i < components; i++) {
         unsigned offset = byte + i * bits / 8;
         nir_def *scalar = nir_channel(b, value, i);
         if (bits == 64) {
            ser_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE + offset,
                      nir_unpack_64_2x32(b, scalar));
         } else if (bits == 32) {
            ser_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE + offset, scalar);
         } else {
            unsigned shift = (offset & 3u) * 8;
            nir_def *word = ser_load(b, object, NVK_HIT_OBJECT_ATTRIBUTE + (offset & ~3u), 1);
            word = nir_bfi(b, nir_imm_int(b, ((1u << bits) - 1u) << shift),
                           nir_ishl_imm(b, nir_u2u32(b, scalar), shift), word);
            ser_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE + (offset & ~3u), word);
         }
      }
   } else {
      nir_def *values[4];
      for (unsigned i = 0; i < components; i++) {
         unsigned offset = byte + i * bits / 8;
         nir_def *word = ser_load(b, object, NVK_HIT_OBJECT_ATTRIBUTE + (offset & ~3u), 1);
         if (bits == 64)
            values[i] = nir_pack_64_2x32_split(b, word,
               ser_load(b, object, NVK_HIT_OBJECT_ATTRIBUTE + offset + 4, 1));
         else
            values[i] = nir_u2uN(b, nir_ushr_imm(b, word, (offset & 3u) * 8), bits);
      }
      nir_def *value = nir_vec(b, values, components);
      if (glsl_type_is_boolean(type))
         value = nir_i2b(b, value);
      nir_store_deref(b, attribute, value, BITFIELD_MASK(components));
   }
}

static nir_def *
ser_query_load(nir_builder *b, nir_def *query, nir_ray_query_value value,
               unsigned components, unsigned bits, unsigned column)
{
   return nir_rq_load(b, components, bits, query, .ray_query_value = value,
                     .committed = true, .column = column);
}

static bool
ser_lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct ser_lower_options *options = data;
   nir_def *object;
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_record_empty:
   case nir_intrinsic_hit_object_record_miss:
   case nir_intrinsic_hit_object_record_from_query:
   case nir_intrinsic_hit_object_load:
   case nir_intrinsic_hit_object_load_attributes:
   case nir_intrinsic_hit_object_set_sbt_index:
      object = intr->src[0].ssa;
      break;
   case nir_intrinsic_reorder_thread:
      if (options->reorder)
         return false;
      nir_instr_remove(&intr->instr);
      return true;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_record_empty:
      ser_store(b, object, NVK_HIT_OBJECT_TYPE, nir_imm_int(b, NVK_HIT_OBJECT_EMPTY));
      break;
   case nir_intrinsic_hit_object_set_sbt_index:
      ser_store(b, object, NVK_HIT_OBJECT_SBT_INDEX, intr->src[1].ssa);
      break;
   case nir_intrinsic_hit_object_load_attributes:
      ser_attributes(b, object, nir_src_as_deref(intr->src[1]), 0, false);
      break;
   case nir_intrinsic_hit_object_record_miss:
      ser_store(b, object, 72, intr->src[1].ssa);
      ser_store(b, object, NVK_HIT_OBJECT_SBT_INDEX, intr->src[2].ssa);
      ser_store(b, object, 96, intr->src[3].ssa);
      ser_store(b, object, 92, intr->src[4].ssa);
      ser_store(b, object, 112, intr->src[5].ssa);
      ser_store(b, object, 108, intr->src[6].ssa);
      ser_store(b, object, NVK_HIT_OBJECT_TIME, nir_imm_int(b, 0));
      ser_store(b, object, NVK_HIT_OBJECT_TYPE, nir_imm_int(b, NVK_HIT_OBJECT_MISS));
      break;
   case nir_intrinsic_hit_object_record_from_query: {
      nir_def *query = intr->src[1].ssa;
      static const struct {
         nir_ray_query_value field;
         unsigned offset, components, bits;
      } ray[] = {
         {nir_ray_query_value_flags, 72, 1, 32},
         {nir_ray_query_value_tmin, 92, 1, 32},
         {nir_ray_query_value_world_ray_origin, 96, 3, 32},
         {nir_ray_query_value_intersection_t, 108, 1, 32},
         {nir_ray_query_value_world_ray_direction, 112, 3, 32},
      }, hit[] = {
         {nir_ray_query_value_intersection_t, 0, 1, 32},
         {nir_ray_query_value_intersection_primitive_index, 12, 1, 32},
         {nir_ray_query_value_intersection_instance_id, 16, 1, 32},
         {nir_ray_query_value_intersection_geometry_index, 20, 1, 32},
         {nir_ray_query_value_intersection_instance_custom_index, 24, 1, 32},
         {nir_ray_query_value_intersection_instance_sbt_index, 28, 1, 32},
         {nir_ray_query_value_intersection_object_ray_origin, 40, 3, 32},
         {nir_ray_query_value_intersection_object_ray_direction, 52, 3, 32},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(ray); i++)
         ser_store(b, object, ray[i].offset, ser_query_load(b, query, ray[i].field,
                  ray[i].components, ray[i].bits, 0));
      ser_store(b, object, NVK_HIT_OBJECT_SBT_INDEX, intr->src[2].ssa);
      ser_store(b, object, NVK_HIT_OBJECT_TIME, nir_imm_int(b, 0));
      nir_def *type = ser_query_load(b, query, nir_ray_query_value_intersection_type, 1, 32, 0);
      ser_store(b, object, NVK_HIT_OBJECT_TYPE, nir_bcsel(b, nir_ieq_imm(b, type, 0),
         nir_imm_int(b, NVK_HIT_OBJECT_MISS), nir_imm_int(b, NVK_HIT_OBJECT_HIT)));
      nir_push_if(b, nir_ine_imm(b, type, 0));
      for (unsigned i = 0; i < ARRAY_SIZE(hit); i++)
         ser_store(b, object, NVK_HIT_OBJECT_RECORD + hit[i].offset,
            ser_query_load(b, query, hit[i].field, hit[i].components, hit[i].bits, 0));
      for (unsigned column = 0; column < 4; column++) {
         ser_store(b, object, NVK_HIT_OBJECT_RECORD + 64 + column * 12,
            ser_query_load(b, query, nir_ray_query_value_intersection_object_to_world, 3, 32, column));
         ser_store(b, object, NVK_HIT_OBJECT_RECORD + 112 + column * 12,
            ser_query_load(b, query, nir_ray_query_value_intersection_world_to_object, 3, 32, column));
      }
      for (unsigned byte = 0; byte < 32; byte += 4)
         ser_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE + byte, nir_imm_int(b, 0));
      nir_push_if(b, nir_ieq_imm(b, type, 1));
      nir_def *bary = ser_query_load(b, query, nir_ray_query_value_intersection_barycentrics, 2, 32, 0);
      ser_store(b, object, NVK_HIT_OBJECT_RECORD + 4, bary);
      ser_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE, bary);
      nir_def *front = ser_query_load(b, query, nir_ray_query_value_intersection_front_face, 1, 1, 0);
      ser_store(b, object, NVK_HIT_OBJECT_RECORD + 32, nir_b2i32(b, front));
      ser_store(b, object, 52, nir_bcsel(b, front, nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff)));
      if (options->triangle_positions) {
         for (unsigned vertex = 0; vertex < 3; vertex++)
            ser_store(b, object, NVK_HIT_OBJECT_RECORD + 160 + vertex * 12,
               ser_query_load(b, query, nir_ray_query_value_intersection_triangle_vertex_positions, 3, 32, vertex));
      }
      nir_push_else(b, NULL);
      ser_attributes(b, object, nir_src_as_deref(intr->src[3]), 0, true);
      ser_store(b, object, 52, intr->src[4].ssa);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_hit_object_load: {
      unsigned column = nir_intrinsic_column(intr);
      unsigned offset;
      switch ((nir_hit_object_value)nir_intrinsic_base(intr)) {
      case nir_hit_object_value_tmin: offset = 92; break;
      case nir_hit_object_value_tmax: offset = 108; break;
      case nir_hit_object_value_flags: offset = 72; break;
      case nir_hit_object_value_world_origin: offset = 96; break;
      case nir_hit_object_value_world_direction: offset = 112; break;
      case nir_hit_object_value_object_origin: offset = NVK_HIT_OBJECT_RECORD + 40; break;
      case nir_hit_object_value_object_direction: offset = NVK_HIT_OBJECT_RECORD + 52; break;
      case nir_hit_object_value_object_to_world: offset = NVK_HIT_OBJECT_RECORD + 64 + column * 12; break;
      case nir_hit_object_value_world_to_object: offset = NVK_HIT_OBJECT_RECORD + 112 + column * 12; break;
      case nir_hit_object_value_instance_id: offset = NVK_HIT_OBJECT_RECORD + 16; break;
      case nir_hit_object_value_instance_custom_index: offset = NVK_HIT_OBJECT_RECORD + 24; break;
      case nir_hit_object_value_geometry_index: offset = NVK_HIT_OBJECT_RECORD + 20; break;
      case nir_hit_object_value_primitive_index: offset = NVK_HIT_OBJECT_RECORD + 12; break;
      case nir_hit_object_value_hit_kind: offset = 52; break;
      case nir_hit_object_value_current_time: offset = NVK_HIT_OBJECT_TIME; break;
      case nir_hit_object_value_sbt_index: offset = NVK_HIT_OBJECT_SBT_INDEX; break;
      case nir_hit_object_value_triangle_positions: offset = NVK_HIT_OBJECT_RECORD + 160 + column * 12; break;
      case nir_hit_object_value_shader_record: return false;
      case nir_hit_object_value_is_empty:
      case nir_hit_object_value_is_hit:
      case nir_hit_object_value_is_miss:
         value = nir_ieq_imm(b, ser_load(b, object, NVK_HIT_OBJECT_TYPE, 1),
            nir_intrinsic_base(intr) == nir_hit_object_value_is_empty ? NVK_HIT_OBJECT_EMPTY :
            nir_intrinsic_base(intr) == nir_hit_object_value_is_hit ? NVK_HIT_OBJECT_HIT : NVK_HIT_OBJECT_MISS);
         break;
      default: UNREACHABLE("Invalid hit object field");
      }
      if (!value)
         value = ser_load(b, object, offset, intr->def.num_components);
      break;
   }
   default: UNREACHABLE("Invalid hit object operation");
   }
   if (value)
      nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

bool
nvk_nir_lower_hit_objects_with_reorder(nir_shader *nir, bool reorder)
{
   struct ser_lower_options options = {.reorder = reorder};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            options.triangle_positions |=
               intr->intrinsic == nir_intrinsic_hit_object_load &&
               nir_intrinsic_base(intr) == nir_hit_object_value_triangle_positions;
         }
      }
   }
   return nir_shader_intrinsics_pass(nir, ser_lower, nir_metadata_none, &options);
}

bool
nvk_nir_lower_hit_objects(nir_shader *nir)
{
   return nvk_nir_lower_hit_objects_with_reorder(nir, false);
}
