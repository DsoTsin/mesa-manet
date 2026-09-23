/*
 * Copyright © 2020 Google, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nir_serialize.h"

#include "ir3_compiler.h"
#include "ir3_nir.h"
#include <type_traits>

#define debug 0

/*
 * Shader disk-cache implementation.
 *
 * Note that at least in the EGL_ANDROID_blob_cache, we should never
 * rely on inter-dependencies between different cache entries:
 *
 *    No guarantees are made as to whether a given key/value pair is present in
 *    the cache after the set call.  If a different value has been associated
 *    with the given key in the past then it is undefined which value, if any,
 *    is associated with the key after the set call.  Note that while there are
 *    no guarantees, the cache implementation should attempt to cache the most
 *    recently set value for a given key.
 *
 * for this reason, because binning pass variants share const_state with
 * their draw-pass counterpart, both variants are serialized together.
 */

void
ir3_disk_cache_init(struct ir3_compiler *compiler)
{
   if (ir3_shader_debug & IR3_DBG_NOCACHE)
      return;

   const char *renderer = fd_dev_name(compiler->dev_id);
   const struct build_id_note *note =
      build_id_find_nhdr_for_addr((const void *)ir3_disk_cache_init);
   unsigned build_id_len = build_id_length(note);
   assert(note && build_id_len == BUILD_ID_EXPECTED_HASH_LENGTH); /* sha1 */

   const uint8_t *id_sha1 = build_id_data(note);
   assert(id_sha1);

   blake3_hasher ctx;
   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, id_sha1, build_id_len);
   _mesa_blake3_update(&ctx, &compiler->options.uche_trap_base,
                     sizeof(compiler->options.uche_trap_base));
   _mesa_blake3_final(&ctx, blake3);

   char timestamp[BLAKE3_HEX_LEN];
   _mesa_blake3_format(timestamp, blake3);

   uint64_t driver_flags = ir3_shader_debug_hash_key();
   compiler->disk_cache = disk_cache_create(renderer, timestamp, driver_flags);
}

void
ir3_disk_cache_init_shader_key(struct ir3_compiler *compiler,
                               struct ir3_shader *shader)
{
   if (!compiler->disk_cache && !ir3_shader_bisect_need_shader_key())
      return;

   blake3_hasher ctx;

   _mesa_blake3_init(&ctx);

   /* Serialize the NIR to a binary blob that we can hash for the disk
    * cache.  Drop unnecessary information (like variable names)
    * so the serialized NIR is smaller, and also to let us detect more
    * isomorphic shaders when hashing, increasing cache hits.
    */
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, shader->nir, true);
   _mesa_blake3_update(&ctx, blob.data, blob.size);
   blob_finish(&blob);

   _mesa_blake3_update(&ctx, &shader->options.api_wavesize,
                     sizeof(shader->options.api_wavesize));
   _mesa_blake3_update(&ctx, &shader->options.real_wavesize,
                     sizeof(shader->options.real_wavesize));
   _mesa_blake3_update(&ctx, &shader->options.nir_options,
                     sizeof(shader->options.nir_options));

   /* Note that on some gens stream-out is lowered in ir3 to stg.  For later
    * gens we maybe don't need to include stream-out in the cache key.
    */
   _mesa_blake3_update(&ctx, &shader->stream_output,
                     sizeof(shader->stream_output));

   _mesa_blake3_final(&ctx, shader->cache_key);
}

static void
compute_variant_key(struct ir3_shader *shader, struct ir3_shader_variant *v,
                    cache_key cache_key)
{
   struct blob blob;
   blob_init(&blob);

   blob_write_bytes(&blob, &shader->cache_key, sizeof(shader->cache_key));
   blob_write_bytes(&blob, &v->key, sizeof(v->key));
   blob_write_uint8(&blob, v->binning_pass);

   disk_cache_compute_key(shader->compiler->disk_cache, blob.data, blob.size,
                          cache_key);

   blob_finish(&blob);
}

template <typename IO>
static void
process_variant(IO &io, struct ir3_shader_variant *v)
{
   io.bytes(VARIANT_CACHE_PTR(v), VARIANT_CACHE_SIZE);

   /*
    * pointers need special handling:
    */

   if constexpr (IO::is_read())
      v->bin = (uint32_t *)rzalloc_size(v, v->info.size);
   io.bytes(v->bin, v->info.size);

   /* No handling of constant_data, it's already baked into bin at this point. */

   if (!v->binning_pass)
      io.bytes(v->const_state, sizeof(*v->const_state));

   if (!v->compiler->info->props.load_shader_consts_via_preamble) {
      io.u32(v->imm_state.size);
      uint32_t immeds_sz = v->imm_state.size * sizeof(v->imm_state.values[0]);

      if constexpr (IO::is_read()) {
         v->imm_state.count = v->imm_state.size;
         v->imm_state.values = (uint32_t *)ralloc_size(v, immeds_sz);
      }

      io.bytes(v->imm_state.values, immeds_sz);
   }
}

static void
ir3_shader_variant_init(struct ir3_shader_variant *v,
                        struct ir3_compiler *compiler)
{
   v->id = 0;
   v->compiler = compiler;
   v->binning_pass = false;
   v->nonbinning = NULL;
   v->binning = NULL;
   v->const_state =
      (struct ir3_const_state *)rzalloc_size(v, sizeof(*v->const_state));
}

static void
ir3_shader_variant_binning_alloc(struct ir3_shader_variant *v)
{
   v->binning =
      (struct ir3_shader_variant *)rzalloc_size(v, sizeof(*v->binning));
   v->binning->id = 0;
   v->binning->compiler = v->compiler;
   v->binning->binning_pass = true;
   v->binning->nonbinning = v;
   v->binning->key = v->key;
   v->binning->type = MESA_SHADER_VERTEX;
   v->binning->mergedregs = v->mergedregs;
   v->binning->const_state = v->const_state;
}

template <typename IO>
void
ir3_process_variant(IO &io, struct ir3_shader_variant *v)
{
   if constexpr (IO::is_read())
      ir3_shader_variant_init(v, io.compiler);

   io.bytes(&v->key, sizeof(v->key));
   io.enum_t(v->type);
   io.boolean(v->mergedregs);
   process_variant(io, v);

   if (v->type == MESA_SHADER_VERTEX && ir3_has_binning_vs(&v->key)) {
      if constexpr (IO::is_read())
         ir3_shader_variant_binning_alloc(v);

      process_variant(io, v->binning);
   }
}

void
ir3_blob_write::variant(const struct ir3_shader_variant *&v)
{
   ir3_process_variant(*this, (struct ir3_shader_variant *)v);
}

void
ir3_blob_read::variant(const struct ir3_shader_variant *&v_out)
{
   struct ir3_shader_variant *v =
      (struct ir3_shader_variant *)rzalloc_size(NULL, sizeof(*v));

   ir3_process_variant(*this, v);
   v_out = v;
}

bool
ir3_disk_cache_retrieve(struct ir3_shader *shader,
                        struct ir3_shader_variant *v)
{
   if (!shader->compiler->disk_cache)
      return false;

   cache_key cache_key;

   compute_variant_key(shader, v, cache_key);

   if (debug) {
      char blake3[BLAKE3_HEX_LEN];
      _mesa_blake3_format(blake3, cache_key);
      fprintf(stderr, "[mesa disk cache] retrieving variant %s: ", blake3);
   }

   size_t size;
   void *buffer = disk_cache_get(shader->compiler->disk_cache, cache_key, &size);

   if (debug)
      fprintf(stderr, "%s\n", buffer ? "found" : "missing");

   if (!buffer)
      return false;

   struct blob_reader blob;
   blob_reader_init(&blob, buffer, size);
   ir3_blob_read reader{&blob, shader->compiler};

   process_variant(reader, v);

   if (v->binning)
      process_variant(reader, v->binning);

   free(buffer);

   return true;
}

void
ir3_disk_cache_store(struct ir3_shader *shader,
                     struct ir3_shader_variant *v)
{
   if (!shader->compiler->disk_cache)
      return;

   cache_key cache_key;

   compute_variant_key(shader, v, cache_key);

   if (debug) {
      char blake3[BLAKE3_HEX_LEN];
      _mesa_blake3_format(blake3, cache_key);
      fprintf(stderr, "[mesa disk cache] storing variant %s\n", blake3);
   }

   struct blob blob;
   blob_init(&blob);
   ir3_blob_write writer{&blob};

   process_variant(writer, v);

   if (v->binning)
      process_variant(writer, v->binning);

   disk_cache_put(shader->compiler->disk_cache, cache_key, blob.data, blob.size, NULL);
   blob_finish(&blob);
}
