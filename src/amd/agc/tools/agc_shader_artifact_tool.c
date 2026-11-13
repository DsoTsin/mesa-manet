#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_shader_artifact.h"
#include "util/os_file.h"

struct agc_tool_text {
   char *data;
   size_t size;
   size_t capacity;
};

struct agc_tool_names {
   char *storage;
   char **names;
   size_t count;
};

struct agc_tool_args {
   const char *positional[4];
   unsigned positional_count;
   const char *dump;
   const char *programs;
   const char *summary;
   const char *requests;
   const char *reference;
   bool digest;
};

static void AGC_PRINTF(2, 3)
agc_tool_put(struct agc_tool_text *text, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   int length = vsnprintf(NULL, 0, format, args);
   va_end(args);
   if (length < 0)
      abort();
   if (text->size + (size_t)length + 1 > text->capacity) {
      size_t capacity = text->capacity ? text->capacity : 1024;
      while (capacity < text->size + (size_t)length + 1)
         capacity *= 2;
      text->data = realloc(text->data, capacity);
      if (!text->data)
         abort();
      text->capacity = capacity;
   }
   va_start(args, format);
   vsnprintf(text->data + text->size, (size_t)length + 1, format, args);
   va_end(args);
   text->size += (size_t)length;
}

static uint64_t
agc_tool_text_digest(const struct agc_tool_text *text)
{
   uint64_t hash = UINT64_C(0xcbf29ce484222325);
   for (size_t i = 0; i < text->size; i++)
      hash = (hash ^ (uint8_t)text->data[i]) * UINT64_C(0x100000001b3);
   return hash;
}

static uint64_t
agc_tool_hash_byte(uint64_t hash, uint8_t byte)
{
   return (hash ^ byte) * UINT64_C(0x100000001b3);
}

static uint64_t
agc_tool_hash_words(const uint32_t *words, uint32_t count)
{
   uint64_t hash = UINT64_C(0xcbf29ce484222325);
   uint64_t length = count;
   for (unsigned i = 0; i < 8; i++)
      hash = agc_tool_hash_byte(hash, (uint8_t)(length >> (8 * i)));
   for (uint32_t w = 0; w < count; w++) {
      for (unsigned i = 0; i < 4; i++)
         hash = agc_tool_hash_byte(hash, (uint8_t)(words[w] >> (8 * i)));
   }
   hash = (hash ^ (hash >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
   hash = (hash ^ (hash >> 27)) * UINT64_C(0x94d049bb133111eb);
   return hash ^ (hash >> 31);
}

static void
agc_tool_put_optional_dec(struct agc_tool_text *text, const char *name, bool present, unsigned value)
{
   if (present)
      agc_tool_put(text, "%s %u\n", name, value);
   else
      agc_tool_put(text, "%s -\n", name);
}

static void
agc_tool_put_optional_hex(struct agc_tool_text *text, const char *name, bool present, uint64_t value)
{
   if (present)
      agc_tool_put(text, "%s 0x%llx\n", name, (unsigned long long)value);
   else
      agc_tool_put(text, "%s -\n", name);
}

static void
agc_tool_dump_request(struct agc_tool_text *text, const char *name, const struct agc_shader_request *request)
{
   const struct agc_shader_interface *i = &request->iface;

   agc_tool_put(text, "request %s\n", name);
   agc_tool_put(text, "compiler_version %u\n", request->compiler_version);
   agc_tool_put(text, "target %u stage %u wave_size %u ray_unit %u precision %u\n",
                (unsigned)request->options.target, (unsigned)request->options.stage,
                (unsigned)request->options.wave_size, (unsigned)request->options.ray_unit,
                (unsigned)request->precision);
   agc_tool_put_optional_dec(text, "user_sgpr_count", i->has_user_sgpr_count, i->user_sgpr_count);
   agc_tool_put_optional_hex(text, "guest_pgm_rsrc1", i->has_guest_pgm_rsrc1, i->guest_pgm_rsrc1);
   agc_tool_put_optional_hex(text, "guest_pgm_rsrc2", i->has_guest_pgm_rsrc2, i->guest_pgm_rsrc2);
   agc_tool_put(text, "lds_bytes 0x%x\n", i->lds_bytes);
   agc_tool_put(text, "scratch_bytes_per_lane 0x%x\n", i->scratch_bytes_per_lane);
   agc_tool_put_optional_hex(text, "guest_code_address", i->has_guest_code_address, i->guest_code_address);
   agc_tool_put_optional_hex(text, "ps_input_ena", i->has_ps_input_ena, i->ps_input_ena);
   agc_tool_put_optional_hex(text, "ps_input_addr", i->has_ps_input_addr, i->ps_input_addr);
   agc_tool_put_optional_hex(text, "ps_col_format", i->has_ps_col_format, i->ps_col_format);
   agc_tool_put_optional_hex(text, "ps_z_format", i->has_ps_z_format, i->ps_z_format);
   agc_tool_put(text, "dual_source_blend %u\n", (unsigned)i->dual_source_blend);
   agc_tool_put_optional_dec(text, "param_exports", i->has_param_exports, i->param_exports);
   agc_tool_put_optional_hex(text, "descriptor_lut", i->has_descriptor_lut, i->descriptor_lut);
   agc_tool_put(text, "preconvert_user_descriptors %u\n", (unsigned)i->preconvert_user_descriptors);
   if (i->has_attribute_ring)
      agc_tool_put(text, "attribute_ring 0x%llx 0x%x\n", (unsigned long long)i->attribute_ring.address,
                   i->attribute_ring.bytes);
   else
      agc_tool_put(text, "attribute_ring -\n");
   agc_tool_put_optional_hex(text, "gds_buffer", i->has_gds_buffer, i->gds_buffer);
   agc_tool_put_optional_hex(text, "mip_stats_table", i->has_mip_stats_table, i->mip_stats_table);
   agc_tool_put_optional_hex(text, "image_reports", i->has_image_reports, i->image_reports);
   agc_tool_put(text, "vertex_stage %u\n", (unsigned)i->vertex_stage);
   agc_tool_put(text, "native_dcc %u\n", (unsigned)i->native_dcc);
   agc_tool_put(text, "words %u %016llx\n", request->word_count,
                (unsigned long long)agc_tool_hash_words(request->words, request->word_count));
   agc_tool_put(text, "regions %u\n", request->linkage.region_count);
   for (uint32_t r = 0; r < request->linkage.region_count; r++) {
      const struct agc_shader_code_region *region = &request->linkage.regions[r];
      agc_tool_put(text, "region 0x%llx %u %016llx\n", (unsigned long long)region->address, region->word_count,
                   (unsigned long long)agc_tool_hash_words(region->words, region->word_count));
   }
   agc_tool_put_optional_hex(text, "second_half", request->linkage.has_second_half, request->linkage.second_half);
}

static void
agc_tool_dump_bindings(struct agc_tool_text *text, const char *title, const struct agc_shader_binding *bindings,
                       uint32_t count)
{
   agc_tool_put(text, "%s %u\n", title, count);
   for (uint32_t i = 0; i < count; i++)
      agc_tool_put(text, "binding %u %u %u %u\n", (unsigned)bindings[i].kind, (unsigned)bindings[i].base_sgpr,
                   (unsigned)bindings[i].dword_count, (unsigned)bindings[i].writable);
}

static void
agc_tool_dump_uses(struct agc_tool_text *text, const char *title, const struct agc_shader_descriptor_use *uses,
                   uint32_t count)
{
   agc_tool_put(text, "%s %u\n", title, count);
   for (uint32_t i = 0; i < count; i++) {
      const struct agc_shader_descriptor_use *use = &uses[i];
      const struct agc_shader_buffer_access *access = &use->buffer_access;

      agc_tool_put(text, "use 0x%x %u %u %u %u source %u ", use->pc, (unsigned)use->kind,
                   (unsigned)use->base_sgpr, (unsigned)use->dword_count, (unsigned)use->access,
                   (unsigned)use->source.kind);
      if (use->source.kind == AGC_SHADER_SOURCE_USER_SGPRS)
         agc_tool_put(text, "%u", (unsigned)use->source.first);
      else
         agc_tool_put(text, "-");
      agc_tool_put(text, " tables %u access", use->source.table_count);
      if (use->has_buffer_access) {
         agc_tool_put(text, " %u %u %u 0x%x %u ", (unsigned)access->scalar, (unsigned)access->indexed,
                      (unsigned)access->vector_offset, access->immediate_offset,
                      (unsigned)access->scalar_offset.kind);
         if (access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_BOUNDED)
            agc_tool_put(text, "0x%x\n", access->scalar_offset.bound);
         else if (access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_USER_SGPR)
            agc_tool_put(text, "%u\n", (unsigned)access->scalar_offset.sgpr);
         else
            agc_tool_put(text, "-\n");
      } else {
         agc_tool_put(text, " -\n");
      }
      for (uint32_t t = 0; t < use->source.table_count; t++) {
         const struct agc_shader_descriptor_table *table = &use->source.tables[t];
         agc_tool_put(text, "table %u %u %lld links %u\n", (unsigned)table->root, (unsigned)table->root_form,
                      (long long)table->offset, table->link_count);
         for (uint32_t l = 0; l < table->link_count; l++)
            agc_tool_put(text, "link %lld %u\n", (long long)table->links[l].offset,
                         (unsigned)table->links[l].form);
      }
   }
}

static void
agc_tool_dump_artifact(struct agc_tool_text *text, const char *name, const struct agc_shader_artifact *a)
{
   const struct agc_shader_error *e = &a->error;

   agc_tool_put(text, "artifact %s\n", name);
   agc_tool_put(text,
                "error %u pc 0x%x word 0x%x branch_target 0x%x opcode 0x%x vgprs %u family %u target %u "
                "compiler %u requested %u\n",
                (unsigned)e->kind, e->pc, e->word, e->branch_target, (unsigned)e->opcode, (unsigned)e->vgprs,
                (unsigned)e->family, (unsigned)e->target, (unsigned)e->compiler, (unsigned)e->requested);
   if (e->reason)
      agc_tool_put(text, "reason %zu %s\n", strlen(e->reason), e->reason);
   else
      agc_tool_put(text, "reason -\n");
   if (e->kind != AGC_SHADER_ERROR_NONE)
      return;

   agc_tool_put(text, "options %u %u %u %u precision %u\n", (unsigned)a->options.target,
                (unsigned)a->options.stage, (unsigned)a->options.wave_size, (unsigned)a->options.ray_unit,
                (unsigned)a->precision);
   agc_tool_put(text, "pgm_rsrc1 0x%x pgm_rsrc2 0x%x num_sgprs %u num_vgprs %u\n", a->pgm_rsrc1, a->pgm_rsrc2,
                (unsigned)a->num_sgprs, (unsigned)a->num_vgprs);
   agc_tool_put(text, "lds_bytes 0x%x scratch_bytes_per_lane 0x%x user_sgpr_count %u\n", a->lds_bytes,
                a->scratch_bytes_per_lane, (unsigned)a->user_sgpr_count);
   agc_tool_put(text, "program_words %u guest_words %u guest_hash %016llx\n", a->program_words, a->guest_words,
                (unsigned long long)a->guest_hash);
   agc_tool_put(text, "code %u %016llx\n", a->code_count,
                (unsigned long long)agc_tool_hash_words(a->code, a->code_count));
   agc_tool_put(text, "exports 0x%x 0x%x 0x%x 0x%x %u %u %u\n", a->exports.color_mask,
                (unsigned)a->exports.depth_mask, (unsigned)a->exports.position_mask, a->exports.param_mask,
                (unsigned)a->exports.primitive, (unsigned)a->exports.compressed, (unsigned)a->exports.valid_mask);
   agc_tool_put_optional_dec(text, "user_data_tail_sgpr", a->has_user_data_tail_sgpr, a->user_data_tail_sgpr);
   agc_tool_put(text, "uses_pc_relative %u\n", (unsigned)a->uses_pc_relative);
   agc_tool_put_optional_hex(text, "spi_shader_col_format", a->has_spi_shader_col_format,
                             a->spi_shader_col_format);
   if (a->has_ngg) {
      agc_tool_put(text, "ngg %u 0x%x relocations ", a->ngg.param_count, a->ngg.spi_vs_out_config);
      if (a->ngg.has_attribute_ring_relocations)
         agc_tool_put(text, "%u %u", a->ngg.attribute_ring_relocations[0], a->ngg.attribute_ring_relocations[1]);
      else
         agc_tool_put(text, "-");
      agc_tool_put(text, " registers");
      for (unsigned i = 0; i < AGC_SHADER_NGG_REGISTER_COUNT; i++)
         agc_tool_put(text, " 0x%x:%u", a->ngg.registers[i].reg, (unsigned)a->ngg.registers[i].source);
      agc_tool_put(text, "\n");
   } else {
      agc_tool_put(text, "ngg -\n");
   }
   if (a->has_buffer_fill)
      agc_tool_put(text, "buffer_fill %u %u\n", (unsigned)a->buffer_fill.buffer, (unsigned)a->buffer_fill.value);
   else
      agc_tool_put(text, "buffer_fill -\n");
   agc_tool_dump_bindings(text, "bindings", a->bindings, a->binding_count);
   agc_tool_dump_bindings(text, "preconverted", a->preconverted_descriptors, a->preconverted_descriptor_count);
   agc_tool_put(text, "external_targets %u\n", a->external_target_count);
   for (uint32_t i = 0; i < a->external_target_count; i++) {
      const struct agc_shader_external_target *target = &a->external_targets[i];
      agc_tool_put(text, "external 0x%x %u ", target->pc, (unsigned)target->call);
      if (target->has_sgpr)
         agc_tool_put(text, "%u\n", (unsigned)target->sgpr);
      else
         agc_tool_put(text, "-\n");
   }
   agc_tool_dump_uses(text, "descriptor_uses", a->descriptor_uses, a->descriptor_use_count);
   agc_tool_dump_uses(text, "pointer_uses", a->pointer_uses, a->pointer_use_count);
}

static bool
agc_tool_read(const char *directory, const char *name, const char *extension, uint8_t **data, size_t *size)
{
   size_t length = strlen(directory) + strlen(name) + strlen(extension) + 2;
   char *path = malloc(length);

   if (!path)
      abort();
   snprintf(path, length, "%s/%s%s", directory, name, extension);
   *data = (uint8_t *)os_read_file(path, size);
   if (!*data)
      fprintf(stderr, "%s: cannot read\n", path);
   free(path);
   return *data != NULL;
}

static bool
agc_tool_read_names(const char *path, struct agc_tool_names *names)
{
   size_t size;
   size_t lines = 1;

   memset(names, 0, sizeof(*names));
   names->storage = os_read_file(path, &size);
   if (!names->storage) {
      fprintf(stderr, "%s: cannot read\n", path);
      return false;
   }
   for (size_t i = 0; i < size; i++)
      lines += names->storage[i] == '\n';
   names->names = calloc(lines, sizeof(*names->names));
   if (!names->names)
      abort();
   char *line = names->storage;
   for (size_t i = 0; i <= size; i++) {
      if (i < size && names->storage[i] != '\n')
         continue;
      names->storage[i] = 0;
      if (i > 0 && names->storage[i - 1] == '\r')
         names->storage[i - 1] = 0;
      if (*line)
         names->names[names->count++] = line;
      line = names->storage + i + 1;
   }
   return true;
}

static void
agc_tool_free_names(struct agc_tool_names *names)
{
   free(names->names);
   free(names->storage);
}

static FILE *
agc_tool_open_output(const char *path)
{
   FILE *file;

   if (!path)
      return NULL;
   file = fopen(path, "wb");
   if (!file) {
      fprintf(stderr, "%s: cannot write\n", path);
      exit(2);
   }
   return file;
}

static void
agc_tool_write_dump(FILE *file, const char *name, const struct agc_tool_text *text, bool digest)
{
   if (!file)
      return;
   if (digest)
      fprintf(file, "%s %016llx\n", name, (unsigned long long)agc_tool_text_digest(text));
   else
      fwrite(text->data, 1, text->size, file);
}

static bool
agc_tool_same(const uint8_t *a, size_t a_size, const uint8_t *b, size_t b_size)
{
   return a_size == b_size && memcmp(a, b, a_size) == 0;
}

static int
agc_tool_check_requests(const struct agc_tool_args *args)
{
   struct agc_tool_names names;
   struct agc_tool_text text = {0};
   FILE *dump = agc_tool_open_output(args->dump);
   size_t ok = 0, rejected = 0, mismatch = 0, unreadable = 0, key_named = 0, key_mismatch = 0;

   if (!agc_tool_read_names(args->positional[1], &names))
      return 2;
   for (size_t n = 0; n < names.count; n++) {
      struct agc_shader_request request;
      struct agc_shader_key key, bytes_key, named;
      uint8_t *data, *again;
      size_t size, again_size;

      if (!agc_tool_read(args->positional[0], names.names[n], AGC_SHADER_REQUEST_EXTENSION, &data, &size)) {
         unreadable++;
         continue;
      }
      enum agc_shader_file_status status = agc_shader_request_deserialize(data, size, &request);
      if (status != AGC_SHADER_FILE_OK) {
         fprintf(stderr, "%s: request rejected: %s\n", names.names[n], agc_shader_file_status_name(status));
         rejected++;
         free(data);
         continue;
      }
      status = agc_shader_request_serialize(&request, &again, &again_size);
      if (status == AGC_SHADER_FILE_OK && agc_tool_same(data, size, again, again_size)) {
         ok++;
      } else {
         fprintf(stderr, "%s: request differs after a round trip\n", names.names[n]);
         mismatch++;
      }
      agc_shader_key_of_bytes(data, size, &bytes_key);
      if (agc_shader_request_key(&request, &key) != AGC_SHADER_FILE_OK ||
          memcmp(&key, &bytes_key, sizeof(key)) != 0) {
         fprintf(stderr, "%s: key of the request and of its bytes differ\n", names.names[n]);
         key_mismatch++;
      } else if (agc_shader_key_from_hex(names.names[n], &named)) {
         key_named++;
         if (memcmp(&key, &named, sizeof(key)) != 0) {
            fprintf(stderr, "%s: the name is not the key\n", names.names[n]);
            key_mismatch++;
         }
      }
      text.size = 0;
      agc_tool_dump_request(&text, names.names[n], &request);
      agc_tool_write_dump(dump, names.names[n], &text, args->digest);
      agc_shader_request_free(&request);
      free(again);
      free(data);
   }
   if (dump)
      fclose(dump);
   printf("requests %zu round_trip_ok %zu rejected %zu byte_mismatch %zu unreadable %zu key_named %zu "
          "key_mismatch %zu\n",
          names.count, ok, rejected, mismatch, unreadable, key_named, key_mismatch);
   agc_tool_free_names(&names);
   free(text.data);
   return rejected || mismatch || unreadable || key_mismatch ? 1 : 0;
}

static bool
agc_tool_matches_request(const struct agc_shader_artifact *artifact, const struct agc_shader_request *request)
{
   if (artifact->error.kind != AGC_SHADER_ERROR_NONE)
      return true;
   return artifact->options.target == request->options.target &&
          artifact->options.stage == request->options.stage &&
          artifact->options.wave_size == request->options.wave_size &&
          artifact->options.ray_unit == request->options.ray_unit && artifact->precision == request->precision &&
          artifact->guest_words <= request->word_count &&
          artifact->guest_hash == agc_tool_hash_words(request->words, artifact->guest_words);
}

static int
agc_tool_check_artifacts(const struct agc_tool_args *args)
{
   struct agc_tool_names names;
   struct agc_tool_text text = {0};
   FILE *dump = agc_tool_open_output(args->dump);
   FILE *programs = agc_tool_open_output(args->programs);
   FILE *summary = agc_tool_open_output(args->summary);
   size_t kinds[AGC_SHADER_ERROR_KIND_COUNT] = {0};
   size_t ok = 0, rejected = 0, mismatch = 0, unreadable = 0, request_checked = 0, request_mismatch = 0;

   if (!agc_tool_read_names(args->positional[1], &names))
      return 2;
   for (size_t n = 0; n < names.count; n++) {
      const char *name = names.names[n];
      struct agc_shader_artifact artifact;
      uint8_t *data, *again;
      size_t size, again_size;

      if (!agc_tool_read(args->positional[0], name, AGC_SHADER_ARTIFACT_EXTENSION, &data, &size)) {
         unreadable++;
         continue;
      }
      enum agc_shader_file_status status = agc_shader_artifact_deserialize(data, size, &artifact);
      if (status != AGC_SHADER_FILE_OK) {
         fprintf(stderr, "%s: artifact rejected: %s\n", name, agc_shader_file_status_name(status));
         rejected++;
         free(data);
         continue;
      }
      status = agc_shader_artifact_serialize(&artifact, &again, &again_size);
      if (status == AGC_SHADER_FILE_OK && agc_tool_same(data, size, again, again_size)) {
         ok++;
      } else {
         fprintf(stderr, "%s: artifact differs after a round trip\n", name);
         mismatch++;
      }
      kinds[artifact.error.kind]++;
      text.size = 0;
      agc_tool_dump_artifact(&text, name, &artifact);
      agc_tool_write_dump(dump, name, &text, args->digest);
      if (programs && artifact.error.kind == AGC_SHADER_ERROR_NONE) {
         fprintf(programs, "%s ok", name);
         for (uint32_t w = 0; w < artifact.program_words; w++)
            fprintf(programs, " %08x", artifact.code[w]);
         fputc('\n', programs);
      } else if (programs) {
         fprintf(programs, "%s error %s\n", name, agc_shader_error_kind_name(artifact.error.kind));
      }
      if (summary && artifact.error.kind == AGC_SHADER_ERROR_NONE)
         fprintf(summary, "%s\t%016llx\t%u\t%x\t%x\t%u\t%u\n", name,
                 (unsigned long long)agc_tool_hash_words(artifact.code, artifact.code_count),
                 artifact.program_words, artifact.pgm_rsrc1, artifact.pgm_rsrc2, (unsigned)artifact.num_sgprs,
                 (unsigned)artifact.num_vgprs);
      else if (summary)
         fprintf(summary, "%s\terror\t%s\n", name, agc_shader_error_kind_name(artifact.error.kind));
      if (args->requests) {
         struct agc_shader_request request;
         uint8_t *request_data;
         size_t request_size;

         if (agc_tool_read(args->requests, name, AGC_SHADER_REQUEST_EXTENSION, &request_data, &request_size)) {
            request_checked++;
            if (agc_shader_request_deserialize(request_data, request_size, &request) != AGC_SHADER_FILE_OK ||
                !agc_tool_matches_request(&artifact, &request)) {
               fprintf(stderr, "%s: artifact does not belong to its request\n", name);
               request_mismatch++;
            }
            agc_shader_request_free(&request);
            free(request_data);
         } else {
            request_mismatch++;
         }
      }
      agc_shader_artifact_free(&artifact);
      free(again);
      free(data);
   }
   if (dump)
      fclose(dump);
   if (programs)
      fclose(programs);
   if (summary)
      fclose(summary);
   printf("artifacts %zu round_trip_ok %zu rejected %zu byte_mismatch %zu unreadable %zu\n", names.count, ok,
          rejected, mismatch, unreadable);
   printf("translated %zu\n", kinds[AGC_SHADER_ERROR_NONE]);
   for (unsigned kind = 1; kind < AGC_SHADER_ERROR_KIND_COUNT; kind++) {
      if (kinds[kind])
         printf("error %s %zu\n", agc_shader_error_kind_name((enum agc_shader_error_kind)kind), kinds[kind]);
   }
   if (args->requests)
      printf("request_checked %zu request_mismatch %zu\n", request_checked, request_mismatch);
   agc_tool_free_names(&names);
   free(text.data);
   return rejected || mismatch || unreadable || request_mismatch ? 1 : 0;
}

static bool
agc_tool_cache_file_same(const char *directory, const char *subdirectory, const struct agc_shader_key *key,
                         const char *extension, const uint8_t *data, size_t size)
{
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   char *folder;
   uint8_t *stored;
   size_t stored_size;
   size_t length = strlen(directory) + strlen(subdirectory) + 2;
   bool same;

   agc_shader_key_hex(key, hex);
   folder = malloc(length);
   if (!folder)
      abort();
   snprintf(folder, length, "%s%s%s", directory, *subdirectory ? "/" : "", subdirectory);
   same = agc_tool_read(folder, hex, extension, &stored, &stored_size) &&
          agc_tool_same(data, size, stored, stored_size);
   free(stored);
   free(folder);
   return same;
}

static bool
agc_tool_load_request(const char *directory, const char *name, struct agc_shader_request *request,
                      struct agc_shader_key *key, uint8_t **data, size_t *size)
{
   memset(request, 0, sizeof(*request));
   if (!agc_tool_read(directory, name, AGC_SHADER_REQUEST_EXTENSION, data, size))
      return false;
   if (agc_shader_request_deserialize(*data, *size, request) == AGC_SHADER_FILE_OK &&
       agc_shader_request_key(request, key) == AGC_SHADER_FILE_OK)
      return true;
   fprintf(stderr, "%s: not a valid request\n", name);
   agc_shader_request_free(request);
   free(*data);
   *data = NULL;
   return false;
}

static int
agc_tool_cache_miss(const struct agc_tool_args *args)
{
   struct agc_tool_names names;
   struct agc_shader_cache *cache;
   size_t not_found = 0, found = 0, invalid = 0, identical = 0, failed = 0;

   if (!agc_tool_read_names(args->positional[1], &names))
      return 2;
   if (agc_shader_cache_open(args->positional[2], &cache) != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: cannot open the cache\n", args->positional[2]);
      return 2;
   }
   for (size_t n = 0; n < names.count; n++) {
      struct agc_shader_request request;
      struct agc_shader_artifact artifact;
      struct agc_shader_key key;
      uint8_t *data;
      size_t size;

      if (!agc_tool_load_request(args->positional[0], names.names[n], &request, &key, &data, &size)) {
         failed++;
         continue;
      }
      enum agc_shader_file_status status = agc_shader_cache_lookup(cache, &key, &artifact);
      if (status == AGC_SHADER_FILE_NOT_FOUND)
         not_found++;
      else if (status == AGC_SHADER_FILE_OK)
         found++;
      else
         invalid++;
      agc_shader_artifact_free(&artifact);
      status = agc_shader_cache_miss(cache, &key, &request);
      if (status != AGC_SHADER_FILE_OK) {
         fprintf(stderr, "%s: miss failed: %s\n", names.names[n], agc_shader_file_status_name(status));
         failed++;
      } else if (agc_tool_cache_file_same(args->positional[2], AGC_SHADER_REQUEST_DIRECTORY, &key,
                                          AGC_SHADER_REQUEST_EXTENSION, data, size)) {
         identical++;
      } else {
         fprintf(stderr, "%s: the request written on a miss differs from the source\n", names.names[n]);
         failed++;
      }
      agc_shader_request_free(&request);
      free(data);
   }
   agc_shader_cache_close(cache);
   printf("cache_miss %zu not_found %zu found %zu invalid %zu written_identical %zu failed %zu\n", names.count,
          not_found, found, invalid, identical, failed);
   agc_tool_free_names(&names);
   return failed || invalid ? 1 : 0;
}

static int
agc_tool_cache_lookup(const struct agc_tool_args *args)
{
   struct agc_tool_names names;
   struct agc_shader_cache *cache;
   size_t found = 0, not_found = 0, invalid = 0, identical = 0, different = 0, failed = 0;

   if (!agc_tool_read_names(args->positional[1], &names))
      return 2;
   if (agc_shader_cache_open(args->positional[2], &cache) != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: cannot open the cache\n", args->positional[2]);
      return 2;
   }
   for (size_t n = 0; n < names.count; n++) {
      struct agc_shader_request request;
      struct agc_shader_artifact artifact;
      struct agc_shader_key key;
      uint8_t *data;
      size_t size;

      if (!agc_tool_load_request(args->positional[0], names.names[n], &request, &key, &data, &size)) {
         failed++;
         continue;
      }
      enum agc_shader_file_status status = agc_shader_cache_lookup(cache, &key, &artifact);
      if (status == AGC_SHADER_FILE_NOT_FOUND) {
         not_found++;
      } else if (status != AGC_SHADER_FILE_OK) {
         fprintf(stderr, "%s: lookup failed: %s\n", names.names[n], agc_shader_file_status_name(status));
         invalid++;
      } else {
         found++;
         if (!agc_tool_matches_request(&artifact, &request)) {
            fprintf(stderr, "%s: the cached artifact does not belong to the request\n", names.names[n]);
            failed++;
         }
         if (args->reference) {
            uint8_t *again = NULL, *reference = NULL;
            size_t again_size, reference_size;

            if (agc_shader_artifact_serialize(&artifact, &again, &again_size) == AGC_SHADER_FILE_OK &&
                agc_tool_read(args->reference, names.names[n], AGC_SHADER_ARTIFACT_EXTENSION, &reference,
                              &reference_size) &&
                agc_tool_same(again, again_size, reference, reference_size)) {
               identical++;
            } else {
               fprintf(stderr, "%s: the cached artifact differs from the reference\n", names.names[n]);
               different++;
            }
            free(reference);
            free(again);
         }
      }
      agc_shader_artifact_free(&artifact);
      agc_shader_request_free(&request);
      free(data);
   }
   agc_shader_cache_close(cache);
   printf("cache_lookup %zu found %zu not_found %zu invalid %zu reference_identical %zu reference_mismatch %zu "
          "failed %zu\n",
          names.count, found, not_found, invalid, identical, different, failed);
   agc_tool_free_names(&names);
   return not_found || invalid || different || failed ? 1 : 0;
}

static int
agc_tool_cache_store(const struct agc_tool_args *args)
{
   struct agc_tool_names names;
   struct agc_shader_cache *cache;
   size_t identical = 0, failed = 0;

   if (!agc_tool_read_names(args->positional[1], &names))
      return 2;
   if (agc_shader_cache_open(args->positional[2], &cache) != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: cannot open the cache\n", args->positional[2]);
      return 2;
   }
   for (size_t n = 0; n < names.count; n++) {
      struct agc_shader_artifact artifact, back;
      struct agc_shader_key key;
      uint8_t *data;
      size_t size;

      if (!agc_shader_key_from_hex(names.names[n], &key) ||
          !agc_tool_read(args->positional[0], names.names[n], AGC_SHADER_ARTIFACT_EXTENSION, &data, &size)) {
         fprintf(stderr, "%s: not an artifact named by a key\n", names.names[n]);
         failed++;
         continue;
      }
      memset(&back, 0, sizeof(back));
      if (agc_shader_artifact_deserialize(data, size, &artifact) == AGC_SHADER_FILE_OK &&
          agc_shader_cache_store(cache, &key, &artifact) == AGC_SHADER_FILE_OK &&
          agc_tool_cache_file_same(args->positional[2], "", &key, AGC_SHADER_ARTIFACT_EXTENSION, data, size) &&
          agc_shader_cache_lookup(cache, &key, &back) == AGC_SHADER_FILE_OK) {
         identical++;
      } else {
         fprintf(stderr, "%s: store failed\n", names.names[n]);
         failed++;
      }
      agc_shader_artifact_free(&back);
      agc_shader_artifact_free(&artifact);
      free(data);
   }
   agc_shader_cache_close(cache);
   printf("cache_store %zu stored_identical %zu failed %zu\n", names.count, identical, failed);
   agc_tool_free_names(&names);
   return failed ? 1 : 0;
}

static int
agc_tool_key(const struct agc_tool_args *args)
{
   struct agc_shader_request request;
   struct agc_shader_key key;
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   enum agc_shader_file_status status = agc_shader_request_read_file(args->positional[0], &request);

   if (status == AGC_SHADER_FILE_OK)
      status = agc_shader_request_key(&request, &key);
   agc_shader_request_free(&request);
   if (status != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: %s\n", args->positional[0], agc_shader_file_status_name(status));
      return 1;
   }
   agc_shader_key_hex(&key, hex);
   printf("%s\n", hex);
   return 0;
}

static int
agc_tool_usage(void)
{
   fprintf(stderr,
           "usage:\n"
           "  agc_shader_artifact_tool check-requests <dir> <names file> [--dump FILE [--digest]]\n"
           "  agc_shader_artifact_tool check-artifacts <dir> <names file> [--dump FILE [--digest]]\n"
           "                           [--programs FILE] [--summary FILE] [--requests DIR]\n"
           "  agc_shader_artifact_tool cache-miss <requests dir> <names file> <cache dir>\n"
           "  agc_shader_artifact_tool cache-lookup <requests dir> <names file> <cache dir> [--reference DIR]\n"
           "  agc_shader_artifact_tool cache-store <artifacts dir> <names file> <cache dir>\n"
           "  agc_shader_artifact_tool key <request file>\n");
   return 2;
}

int
main(int argc, char **argv)
{
   struct agc_tool_args args = {0};
   const char *command = argc > 1 ? argv[1] : "";

   for (int i = 2; i < argc; i++) {
      const char **option = NULL;

      if (!strcmp(argv[i], "--dump"))
         option = &args.dump;
      else if (!strcmp(argv[i], "--programs"))
         option = &args.programs;
      else if (!strcmp(argv[i], "--summary"))
         option = &args.summary;
      else if (!strcmp(argv[i], "--requests"))
         option = &args.requests;
      else if (!strcmp(argv[i], "--reference"))
         option = &args.reference;
      if (option) {
         if (++i >= argc)
            return agc_tool_usage();
         *option = argv[i];
      } else if (!strcmp(argv[i], "--digest")) {
         args.digest = true;
      } else if (argv[i][0] == '-' && argv[i][1] == '-') {
         return agc_tool_usage();
      } else if (args.positional_count < 4) {
         args.positional[args.positional_count++] = argv[i];
      } else {
         return agc_tool_usage();
      }
   }
   if (!strcmp(command, "check-requests") && args.positional_count == 2)
      return agc_tool_check_requests(&args);
   if (!strcmp(command, "check-artifacts") && args.positional_count == 2)
      return agc_tool_check_artifacts(&args);
   if (!strcmp(command, "cache-miss") && args.positional_count == 3)
      return agc_tool_cache_miss(&args);
   if (!strcmp(command, "cache-lookup") && args.positional_count == 3)
      return agc_tool_cache_lookup(&args);
   if (!strcmp(command, "cache-store") && args.positional_count == 3)
      return agc_tool_cache_store(&args);
   if (!strcmp(command, "key") && args.positional_count == 1)
      return agc_tool_key(&args);
   return agc_tool_usage();
}
