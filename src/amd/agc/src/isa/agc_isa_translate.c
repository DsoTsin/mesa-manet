#include "agc_isa_translate.h"

#include <stdlib.h>
#include <string.h>

#include "agc_lower.h"
#include "agc_isa_regalloc.h"

#define FNV_OFFSET UINT64_C(0xcbf29ce484222325)
#define FNV_PRIME UINT64_C(0x00000100000001b3)

static uint64_t
fnv_bytes(uint64_t h, const uint8_t *bytes, size_t count)
{
   for (size_t i = 0; i < count; i++)
      h = (h ^ bytes[i]) * FNV_PRIME;
   return h;
}

static uint64_t
agc_isa_guest_hash(const uint32_t *words, size_t count)
{
   uint64_t h = FNV_OFFSET;
   const uint64_t length = count;
   uint8_t bytes[8];
   for (unsigned i = 0; i < 8; i++)
      bytes[i] = (uint8_t)(length >> (8 * i));
   h = fnv_bytes(h, bytes, 8);
   for (size_t w = 0; w < count; w++) {
      for (unsigned i = 0; i < 4; i++)
         bytes[i] = (uint8_t)(words[w] >> (8 * i));
      h = fnv_bytes(h, bytes, 4);
   }
   uint64_t z = h;
   z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
   return z ^ (z >> 31);
}

static bool
fail(struct agc_isa_error *error, enum agc_isa_error_kind kind, const char *reason)
{
   memset(error, 0, sizeof(*error));
   error->kind = kind;
   error->reason = reason;
   return false;
}

static bool
null_exports(const struct agc_isa_ir *ir)
{
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      if (inst->fields == AGC_ISA_EXP && inst->exp.target == AGC_ISA_EXP_TARGET_NULL)
         return true;
   }
   return false;
}

static bool
sop1_literal_at(const uint32_t *code, size_t words, bool has_sgpr, uint8_t sgpr, uint32_t literal, uint32_t *offset)
{
   for (size_t i = 0; i + 1 < words; i++) {
      const uint32_t w = code[i];
      if ((w >> 23) != 0x17d || (w & 0xff) != 0xff || code[i + 1] != literal)
         continue;
      if (has_sgpr && ((w >> 16) & 0x7f) != sgpr)
         continue;
      *offset = (uint32_t)(i + 1);
      return true;
   }
   return false;
}

struct descriptor_arrays {
   struct agc_shader_descriptor_use *uses;
   struct agc_shader_descriptor_table *tables;
   struct agc_shader_table_link *links;
};

static bool
convert_uses(const struct agc_isa_descriptor_use *in, size_t count, struct descriptor_arrays *out)
{
   size_t table_count = 0;
   size_t link_count = 0;
   for (size_t i = 0; i < count; i++) {
      table_count += in[i].source.table_count;
      for (uint32_t t = 0; t < in[i].source.table_count; t++)
         link_count += in[i].source.tables[t].link_count;
   }
   out->uses = calloc(count ? count : 1, sizeof(*out->uses));
   out->tables = calloc(table_count ? table_count : 1, sizeof(*out->tables));
   out->links = calloc(link_count ? link_count : 1, sizeof(*out->links));
   if (!out->uses || !out->tables || !out->links)
      return false;
   size_t next_table = 0;
   size_t next_link = 0;
   for (size_t i = 0; i < count; i++) {
      const struct agc_isa_descriptor_use *u = &in[i];
      struct agc_shader_descriptor_use *v = &out->uses[i];
      v->pc = u->pc;
      v->kind = (enum agc_shader_resource_kind)u->kind;
      v->base_sgpr = u->base_sgpr;
      v->dword_count = u->dword_count;
      v->access = (enum agc_shader_descriptor_access)u->access;
      v->source.kind = (enum agc_shader_source_kind)u->source.kind;
      v->source.first = u->source.first;
      v->source.tables = &out->tables[next_table];
      v->source.table_count = u->source.table_count;
      for (uint32_t t = 0; t < u->source.table_count; t++) {
         const struct agc_isa_descriptor_table *a = &u->source.tables[t];
         struct agc_shader_descriptor_table *b = &out->tables[next_table++];
         b->root = a->root;
         b->root_form = (enum agc_shader_address_form)a->root_form;
         b->offset = a->offset;
         b->links = &out->links[next_link];
         b->link_count = a->link_count;
         for (uint32_t l = 0; l < a->link_count; l++) {
            out->links[next_link].offset = a->links[l].offset;
            out->links[next_link].form = (enum agc_shader_address_form)a->links[l].form;
            next_link++;
         }
      }
      v->has_buffer_access = u->has_buffer_access;
      if (u->has_buffer_access) {
         v->buffer_access.scalar = u->buffer_access.scalar;
         v->buffer_access.indexed = u->buffer_access.indexed;
         v->buffer_access.vector_offset = u->buffer_access.vector_offset;
         v->buffer_access.immediate_offset = u->buffer_access.immediate_offset;
         v->buffer_access.scalar_offset.kind =
            (enum agc_shader_scalar_offset_kind)u->buffer_access.scalar_offset.kind;
         v->buffer_access.scalar_offset.sgpr = u->buffer_access.scalar_offset.sgpr;
         v->buffer_access.scalar_offset.bound = u->buffer_access.scalar_offset.bound;
      }
   }
   return true;
}

static void
free_uses(struct descriptor_arrays *arrays)
{
   free(arrays->uses);
   free(arrays->tables);
   free(arrays->links);
   memset(arrays, 0, sizeof(*arrays));
}

static struct agc_shader_binding *
convert_bindings(const struct agc_isa_resource_binding *in, size_t count)
{
   struct agc_shader_binding *out = calloc(count ? count : 1, sizeof(*out));
   for (size_t i = 0; out && i < count; i++) {
      out[i].kind = (enum agc_shader_resource_kind)in[i].kind;
      out[i].base_sgpr = in[i].base_sgpr;
      out[i].dword_count = in[i].dword_count;
      out[i].writable = in[i].writable;
   }
   return out;
}

static bool
assemble(const struct agc_shader_request *request, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
         const struct agc_isa_encoded *encoded, struct agc_shader_artifact *out, struct agc_isa_error *error)
{
   const struct agc_shader_options *options = &request->options;
   const struct agc_shader_interface *iface = &request->iface;
   const enum agc_isa_target target = (enum agc_isa_target)options->target;
   struct agc_shader_artifact a;
   memset(&a, 0, sizeof(a));
   a.code = encoded->code;
   a.code_count = (uint32_t)encoded->code_count;
   a.program_words = (uint32_t)encoded->program_words;
   a.options = *options;
   a.precision = request->precision;

   struct agc_isa_shader_config config;
   agc_isa_regalloc_finish(regs, &ir->metadata, options, iface, &config);
   a.pgm_rsrc1 = config.pgm_rsrc1;
   a.pgm_rsrc2 = config.pgm_rsrc2;
   a.num_sgprs = config.num_sgprs;
   a.num_vgprs = config.num_vgprs;
   a.lds_bytes = config.lds_bytes;
   a.scratch_bytes_per_lane = config.scratch_bytes_per_lane;
   a.user_sgpr_count = config.user_sgpr_count;

   const size_t guest_words = ir->code_words < request->word_count ? ir->code_words : request->word_count;
   a.guest_words = (uint32_t)guest_words;
   a.guest_hash = agc_isa_guest_hash(request->words, guest_words);

   struct agc_shader_binding *bindings = convert_bindings(ir->metadata.bindings, ir->metadata.binding_count);
   struct agc_shader_binding *preconverted = calloc(encoded->preconverted_count ? encoded->preconverted_count : 1,
                                                    sizeof(*preconverted));
   struct agc_shader_external_target *external =
      calloc(ir->flow.external_count ? ir->flow.external_count : 1, sizeof(*external));
   struct descriptor_arrays uses = {0};
   struct descriptor_arrays pointers = {0};
   bool ok = bindings && preconverted && external &&
             convert_uses(ir->metadata.descriptor_uses, ir->metadata.descriptor_use_count, &uses) &&
             convert_uses(ir->metadata.pointer_uses, ir->metadata.pointer_use_count, &pointers);
   if (ok) {
      memcpy(preconverted, encoded->preconverted, encoded->preconverted_count * sizeof(*preconverted));
      for (size_t i = 0; i < ir->flow.external_count; i++) {
         external[i].pc = ir->flow.external[i].pc;
         external[i].call = ir->flow.external[i].call;
         external[i].has_sgpr = ir->flow.external[i].has_sgpr;
         external[i].sgpr = ir->flow.external[i].sgpr;
      }
      a.bindings = bindings;
      a.binding_count = (uint32_t)ir->metadata.binding_count;
      a.preconverted_descriptors = preconverted;
      a.preconverted_descriptor_count = (uint32_t)encoded->preconverted_count;
      a.external_targets = external;
      a.external_target_count = (uint32_t)ir->flow.external_count;
      a.descriptor_uses = uses.uses;
      a.descriptor_use_count = (uint32_t)ir->metadata.descriptor_use_count;
      a.pointer_uses = pointers.uses;
      a.pointer_use_count = (uint32_t)ir->metadata.pointer_use_count;
      a.exports.color_mask = ir->metadata.exports.color_mask;
      a.exports.depth_mask = ir->metadata.exports.depth_mask;
      a.exports.position_mask = ir->metadata.exports.position_mask;
      a.exports.param_mask = ir->metadata.exports.param_mask;
      a.exports.primitive = ir->metadata.exports.primitive;
      a.exports.compressed = ir->metadata.exports.compressed;
      a.exports.valid_mask = ir->metadata.exports.valid_mask;
      a.has_user_data_tail_sgpr = ir->metadata.has_user_data_tail_sgpr;
      a.user_data_tail_sgpr = ir->metadata.user_data_tail_sgpr;
      a.uses_pc_relative = ir->metadata.uses_pc_relative;
      a.has_buffer_fill = ir->metadata.has_buffer_fill;
      a.buffer_fill.buffer = ir->metadata.buffer_fill.buffer;
      a.buffer_fill.value = ir->metadata.buffer_fill.value;

      if (options->stage == AGC_SHADER_STAGE_FRAGMENT && agc_isa_target_is_gfx11_plus(target)) {
         a.has_spi_shader_col_format =
            agc_isa_pixel_col_format_for(iface->has_ps_col_format, iface->ps_col_format, iface->has_ps_z_format,
                                         iface->ps_z_format, null_exports(ir), &a.spi_shader_col_format);
      }
      if (options->stage == AGC_SHADER_STAGE_VERTEX && agc_isa_target_is_gfx11_plus(target) &&
          iface->vertex_stage == AGC_SHADER_VERTEX_STAGE_ES_GS) {
         a.has_ngg = true;
         a.ngg.param_count = agc_isa_export_info_param_count(&ir->metadata.exports);
         a.ngg.spi_vs_out_config = agc_isa_ngg_spi_vs_out_config(a.ngg.param_count);
         memcpy(a.ngg.registers, agc_isa_ngg_program_registers, sizeof(a.ngg.registers));
         if (encoded->has_attribute_ring) {
            a.ngg.has_attribute_ring_relocations = true;
            for (unsigned k = 0; k < 2 && ok; k++) {
               ok = sop1_literal_at(encoded->code, encoded->program_words, encoded->attribute_ring.has_sgpr,
                                    (uint8_t)(encoded->attribute_ring.sgpr + k), encoded->attribute_ring.words[k],
                                    &a.ngg.attribute_ring_relocations[k]);
            }
            if (!ok)
               fail(error, AGC_ISA_ERROR_UNSUPPORTED, "the attribute ring descriptor literals are not in the code");
         }
      }
   } else {
      fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, NULL);
   }

   uint8_t *bytes = NULL;
   size_t size = 0;
   if (ok && agc_shader_artifact_serialize(&a, &bytes, &size) != AGC_SHADER_FILE_OK)
      ok = fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, "the translation does not serialize");
   if (ok && agc_shader_artifact_deserialize(bytes, size, out) != AGC_SHADER_FILE_OK)
      ok = fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, "the translation does not read back");
   free(bytes);
   free(bindings);
   free(preconverted);
   free(external);
   free_uses(&uses);
   free_uses(&pointers);
   return ok;
}

bool
agc_isa_translate(const struct agc_shader_request *request, struct agc_shader_artifact *artifact,
                  struct agc_isa_error *error)
{
   memset(error, 0, sizeof(*error));
   memset(artifact, 0, sizeof(*artifact));
   const enum agc_isa_target target = (enum agc_isa_target)request->options.target;
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   struct agc_isa_encoded encoded;
   agc_isa_encoded_init(&encoded);
   bool ok = agc_isa_analyze_linked(target, request->words, request->word_count, &request->options, &request->iface,
                                    &request->linkage, &ir, error);
   struct agc_isa_regalloc regs;
   if (ok) {
      agc_isa_regalloc_init(&regs, &ir.metadata, &request->options, &request->iface);
      struct agc_isa_lowerer lowerer;
      agc_isa_lowerer_init(&lowerer, &request->options, request->precision);
      agc_isa_lowerer_with_interface(&lowerer, &ir, &request->iface);
      ok = agc_isa_lower_encode(&lowerer, &ir, &regs, &encoded, error);
   }
   if (ok)
      ok = assemble(request, &ir, &regs, &encoded, artifact, error);
   agc_isa_encoded_finish(&encoded);
   agc_isa_ir_finish(&ir);
   return ok;
}

bool
agc_isa_translate_listing(const struct agc_shader_request *request, char **listing, struct agc_isa_error *error)
{
   memset(error, 0, sizeof(*error));
   *listing = NULL;
   const enum agc_isa_target target = (enum agc_isa_target)request->options.target;
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   bool ok = agc_isa_analyze_linked(target, request->words, request->word_count, &request->options, &request->iface,
                                    &request->linkage, &ir, error);
   if (ok) {
      struct agc_isa_regalloc regs;
      agc_isa_regalloc_init(&regs, &ir.metadata, &request->options, &request->iface);
      struct agc_isa_lowerer lowerer;
      agc_isa_lowerer_init(&lowerer, &request->options, request->precision);
      agc_isa_lowerer_with_interface(&lowerer, &ir, &request->iface);
      ok = agc_isa_lower_listing(&lowerer, &ir, &regs, listing, error);
   }
   agc_isa_ir_finish(&ir);
   return ok;
}
