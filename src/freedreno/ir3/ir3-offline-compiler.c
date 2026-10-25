#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define kDefaultMaxIdBound ir3_offline_spirv_default_max_id_bound
#include <spirv-tools/libspirv.h>
#undef kDefaultMaxIdBound
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "spirv/nir_spirv.h"
#include "spirv/spirv.h"
#include "util/half_float.h"
#include "util/mesa-blake3.h"
#include "util/u_string.h"
#include "git_sha1.h"
#include "ir3-offline-compiler.h"
#include "vk_util.h"

#define MAX_SETS 8

struct options {
   const char *input, *prefix, *entry;
   int stage;
   struct ir3_offline_binding bindings[256];
   unsigned binding_count;
   VkSpecializationMapEntry specs[128];
   uint64_t spec_data[128];
   unsigned spec_count;
};

static bool
number(const char *str, uint64_t *out)
{
   char *end;
   errno = 0;
   if (!str[0] || str[0] == '-')
      return false;
   *out = strtoull(str, &end, 0);
   return !errno && !*end;
}

static void
json_string(FILE *f, const char *str)
{
   fputc('"', f);
   for (const unsigned char *s = (const unsigned char *)str; *s; s++) {
      if (*s == '"' || *s == '\\')
         fprintf(f, "\\%c", *s);
      else if (*s < 32)
         fprintf(f, "\\u%04x", *s);
      else
         fputc(*s, f);
   }
   fputc('"', f);
}

static void
hash_hex(const void *data, size_t size, char out[BLAKE3_HEX_LEN])
{
   blake3_hash hash;
   _mesa_blake3_compute(data, size, hash);
   _mesa_blake3_format(out, hash);
}

static int
stage_from_model(unsigned model)
{
   switch (model) {
   case SpvExecutionModelVertex:
      return MESA_SHADER_VERTEX;
   case SpvExecutionModelFragment:
      return MESA_SHADER_FRAGMENT;
   case SpvExecutionModelGLCompute:
      return MESA_SHADER_COMPUTE;
   default:
      return -1;
   }
}

static bool
select_entry(const uint32_t *words, size_t count, struct options *opts)
{
   if (count < 5 || words[0] != SpvMagicNumber || words[4] != 0)
      return false;
   unsigned matches = 0;
   int stage = -1;
   for (size_t i = 5; i < count;) {
      unsigned wc = words[i] >> 16, op = words[i] & 0xffff;
      if (!wc || wc > count - i)
         return false;
      if (op == SpvOpEntryPoint) {
         if (wc < 4 || !memchr(&words[i + 3], 0, (wc - 3) * 4))
            return false;
         int candidate = stage_from_model(words[i + 1]);
         if (!strcmp((const char *)&words[i + 3], opts->entry) &&
             (opts->stage < 0 || opts->stage == candidate)) {
            matches++;
            stage = candidate;
         }
      }
      i += wc;
   }
   if (matches != 1 || stage < 0)
      return false;
   opts->stage = stage;
   return true;
}

static bool
descriptor_type(const char *str, VkDescriptorType *type)
{
   static const struct {
      const char *name;
      VkDescriptorType type;
   } types[] = {
      {"sampler", VK_DESCRIPTOR_TYPE_SAMPLER},
      {"combined-image-sampler", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
      {"sampled-image", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE},
      {"storage-image", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
      {"uniform-texel-buffer", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER},
      {"storage-texel-buffer", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER},
      {"uniform-buffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
      {"storage-buffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
      {"uniform-buffer-dynamic", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC},
      {"storage-buffer-dynamic", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC},
      {"input-attachment", VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT},
      {"acceleration-structure", VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(types); i++)
      if (!strcmp(str, types[i].name)) {
         *type = types[i].type;
         return true;
      }
   return false;
}

static bool
add_binding(struct options *opts, const char *arg)
{
   char *copy = strdup(arg), *save = NULL;
   if (!copy)
      return false;
   char *parts[5] = {0};
   for (unsigned i = 0; i < 5; i++)
      parts[i] = strtok_r(i ? NULL : copy, ":", &save);
   uint64_t set, binding, count;
   VkDescriptorType type;
   bool ok = parts[0] && parts[1] && parts[2] && parts[3] && !parts[4] &&
             number(parts[0], &set) && number(parts[1], &binding) &&
             descriptor_type(parts[2], &type) && number(parts[3], &count) &&
             set < MAX_SETS && binding < 4096 && count > 0 && count <= 65536 &&
             opts->binding_count < ARRAY_SIZE(opts->bindings);
   if (ok) {
      for (unsigned i = 0; i < opts->binding_count; i++)
         if (opts->bindings[i].set == set &&
             opts->bindings[i].vk.binding == binding)
            ok = false;
   }
   if (ok)
      opts->bindings[opts->binding_count++] = (struct ir3_offline_binding){
         .set = set,
         .overridden = true,
         .vk = {.binding = binding,
                .descriptorType = type,
                .descriptorCount = count},
      };
   free(copy);
   return ok;
}

static bool
add_spec(struct options *opts, const char *arg)
{
   char *copy = strdup(arg), *save = NULL;
   if (!copy)
      return false;
   char *parts[4] = {0};
   for (unsigned i = 0; i < 4; i++)
      parts[i] = strtok_r(i ? NULL : copy, ":=", &save);
   uint64_t id, size, value;
   bool ok = parts[0] && parts[1] && parts[2] && !parts[3] &&
             number(parts[0], &id) && id <= UINT32_MAX &&
             number(parts[1], &size) &&
             (size == 1 || size == 2 || size == 4 || size == 8) &&
             number(parts[2], &value) &&
             (size == 8 || value < (UINT64_C(1) << (8 * size))) &&
             opts->spec_count < ARRAY_SIZE(opts->specs);
   if (ok)
      for (unsigned i = 0; i < opts->spec_count; i++)
         if (opts->specs[i].constantID == id)
            ok = false;
   if (ok) {
      unsigned i = opts->spec_count++;
      opts->specs[i] = (VkSpecializationMapEntry){
         .constantID = id,
         .offset = i * sizeof(uint64_t),
         .size = size,
      };
      memcpy(&opts->spec_data[i], &value, size);
   }
   free(copy);
   return ok;
}

static bool
typed_spec(struct options *opts, const uint32_t *words, size_t count,
           const char *arg)
{
   const char *equal = strchr(arg, '=');
   char id_text[32];
   if (!equal || equal == arg || equal - arg >= sizeof(id_text) || !equal[1])
      return false;
   memcpy(id_text, arg, equal - arg);
   id_text[equal - arg] = 0;
   uint64_t id;
   if (!number(id_text, &id) || id > UINT32_MAX)
      return false;
   uint32_t constant = 0, type = 0;
   for (size_t i = 5; i < count; i += words[i] >> 16)
      if ((words[i] & 0xffff) == SpvOpDecorate && (words[i] >> 16) >= 4 &&
          words[i + 2] == SpvDecorationSpecId && words[i + 3] == id)
         constant = words[i + 1];
   for (size_t i = 5; i < count; i += words[i] >> 16) {
      unsigned op = words[i] & 0xffff;
      if ((op == SpvOpSpecConstant || op == SpvOpSpecConstantTrue ||
           op == SpvOpSpecConstantFalse) &&
          words[i + 2] == constant)
         type = words[i + 1];
   }
   if (!constant || !type)
      return false;
   const char *text = equal + 1;
   uint64_t bits = 0;
   unsigned size = 0;
   for (size_t i = 5; i < count; i += words[i] >> 16) {
      if ((words[i] >> 16) < 2 || words[i + 1] != type)
         continue;
      unsigned op = words[i] & 0xffff;
      char *end;
      errno = 0;
      if (op == SpvOpTypeBool) {
         if (strcmp(text, "true") && strcmp(text, "false") &&
             strcmp(text, "1") && strcmp(text, "0"))
            return false;
         bits = !strcmp(text, "true") || !strcmp(text, "1");
         size = 4;
      } else if (op == SpvOpTypeInt) {
         unsigned width = words[i + 2];
         if (width != 8 && width != 16 && width != 32 && width != 64)
            return false;
         size = width / 8;
         if (words[i + 3]) {
            int64_t value = strtoll(text, &end, 0);
            if (errno || !*text || *end ||
                (width < 64 && (value < -(INT64_C(1) << (width - 1)) ||
                                value >= (INT64_C(1) << (width - 1)))))
               return false;
            bits = (uint64_t)value;
            if (width < 64)
               bits &= (UINT64_C(1) << width) - 1;
         } else if (!number(text, &bits) ||
                    (width < 64 && bits >= (UINT64_C(1) << width))) {
            return false;
         }
      } else if (op == SpvOpTypeFloat) {
         unsigned width = words[i + 2];
         if (width == 16 || width == 32) {
            float value = strtof(text, &end);
            if (errno || !*text || *end)
               return false;
            if (width == 16)
               bits = _mesa_float_to_half(value);
            else
               memcpy(&bits, &value, sizeof(value));
         } else if (width == 64) {
            double value = strtod(text, &end);
            if (errno || !*text || *end)
               return false;
            memcpy(&bits, &value, sizeof(value));
         } else {
            return false;
         }
         size = width / 8;
      } else {
         continue;
      }
      break;
   }
   char raw[96];
   snprintf(raw, sizeof(raw), "%" PRIu64 ":%u:0x%" PRIx64, id, size, bits);
   return size && add_spec(opts, raw);
}

static bool
read_spirv(const char *input, uint32_t **words, size_t *bytes)
{
   FILE *f = !strcmp(input, "-") ? stdin : fopen(input, "rb");
   if (!f) {
      fprintf(stderr, "Cannot read %s: %s\n", input, strerror(errno));
      return false;
   }
#ifdef _WIN32
   if (f == stdin && _setmode(_fileno(stdin), _O_BINARY) < 0)
      return false;
#endif
   size_t capacity = 65536;
   const size_t limit = 256 * 1024 * 1024;
   uint8_t *data = malloc(capacity);
   size_t used = 0;
   bool ok = data != NULL;
   while (ok) {
      used += fread(data + used, 1, capacity - used, f);
      if (ferror(f)) {
         ok = false;
         break;
      }
      if (feof(f))
         break;
      if (used == capacity) {
         if (capacity == limit) {
            ok = fgetc(f) == EOF && !ferror(f);
            break;
         }
         capacity *= 2;
         uint8_t *next = realloc(data, capacity);
         if (!next) {
            ok = false;
            break;
         }
         data = next;
      }
   }
   if (f != stdin)
      ok &= fclose(f) == 0;
   if (!ok || used < 20 || used % 4) {
      free(data);
      fputs("Invalid SPIR-V input size or read failure\n", stderr);
      return false;
   }
   *words = (uint32_t *)data;
   *bytes = used;
   return true;
}

static bool
reflect(nir_shader *nir, struct options *opts)
{
   nir_foreach_variable_with_modes(
      var, nir,
      nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_uniform | nir_var_image) {
      VkDescriptorType type;
      const struct glsl_type *base = glsl_without_array(var->type);
      if (var->data.mode == nir_var_mem_ubo)
         type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      else if (var->data.mode == nir_var_mem_ssbo)
         type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      else if (var->data.resource_type == nir_resource_type_acceleration_structure)
         type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
      else if (glsl_type_is_image(base))
         type = (glsl_get_sampler_dim(base) == GLSL_SAMPLER_DIM_SUBPASS ||
                 glsl_get_sampler_dim(base) == GLSL_SAMPLER_DIM_SUBPASS_MS)
                   ? VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
                : glsl_get_sampler_dim(base) == GLSL_SAMPLER_DIM_BUF
                   ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                   : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      else if (glsl_type_is_sampler(base) || glsl_type_is_texture(base))
         type = glsl_type_is_bare_sampler(base) ? VK_DESCRIPTOR_TYPE_SAMPLER
                : glsl_get_sampler_dim(base) == GLSL_SAMPLER_DIM_BUF
                   ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                : glsl_type_is_texture(base)
                   ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                   : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      else
         continue;

      unsigned count =
         glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type) : 1;
      struct ir3_offline_binding *binding = NULL;
      for (unsigned i = 0; i < opts->binding_count; i++)
         if (opts->bindings[i].set == var->data.descriptor_set &&
             opts->bindings[i].vk.binding == var->data.binding)
            binding = &opts->bindings[i];
      if (binding) {
         VkDescriptorType actual = binding->vk.descriptorType;
         bool dynamic = (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                         actual == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
                        (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                         actual == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
         if ((actual != type && !dynamic) ||
             (count && binding->vk.descriptorCount < count))
            return false;
      } else {
         if (var->data.descriptor_set >= MAX_SETS ||
             var->data.binding >= 4096 || !count || count > 65536 ||
             opts->binding_count == ARRAY_SIZE(opts->bindings)) {
            fprintf(stderr, "Cannot reflect descriptor set=%u binding=%u count=%u type=%u\n",
                    var->data.descriptor_set, var->data.binding, count, type);
            return false;
         }
         binding = &opts->bindings[opts->binding_count++];
         *binding = (struct ir3_offline_binding){
            .set = var->data.descriptor_set,
            .vk = {.binding = var->data.binding,
                   .descriptorType = type,
                   .descriptorCount = count},
         };
      }
   }
   for (unsigned i = 0; i < opts->binding_count; i++)
      opts->bindings[i].vk.stageFlags = mesa_to_vk_shader_stage(opts->stage);
   return true;
}

static FILE *
output_file(const char *prefix, const char *suffix, const char *mode)
{
   char *path;
   if (asprintf(&path, "%s.%s", prefix, suffix) < 0)
      return NULL;
   FILE *f = fopen(path, mode);
   if (!f)
      fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
   free(path);
   return f;
}

static bool
close_output(FILE *f)
{
   bool ok = !ferror(f);
   return fclose(f) == 0 && ok;
}


static bool
target(const char *name, struct fd_dev_id *id)
{
   uint64_t number_value;
   const char *numeric = name;
   if (name[0] == 'a' || name[0] == 'A')
      numeric++;
   bool numeric_name = number(numeric, &number_value);
   for (unsigned i = 0;; i++) {
      struct fd_dev_id candidate;
      const char *model = fd_dev_name_at(i, &candidate);
      if (!model)
         return false;
      unsigned model_number = 0;
      if (numeric_name) {
         if (sscanf(model, "FD%u", &model_number) != 1)
            sscanf(model, "Adreno (TM) %u", &model_number);
      }
      if (!strcmp(name, model) || (numeric_name && model_number == number_value)) {
         *id = candidate;
         return true;
      }
   }
}

static bool
dump_variant(FILE *report, const struct options *opts,
             struct ir3_shader_variant *v, unsigned index, const char *kind)
{
   char suffix[96], hash[BLAKE3_HEX_LEN];
   size_t bytes = v->info.sizedwords * sizeof(uint32_t);
   hash_hex(v->bin, bytes, hash);
   if (opts->prefix) {
      snprintf(suffix, sizeof(suffix), "%u.%s.isa.txt", index, kind);
      FILE *f = output_file(opts->prefix, suffix, "w");
      if (!f)
         return false;
      if (!v->disasm_info.disasm) {
         fclose(f);
         return false;
      }
      fputs(v->disasm_info.disasm, f);
      if (!close_output(f))
         return false;
      snprintf(suffix, sizeof(suffix), "%u.%s.bin", index, kind);
      f = output_file(opts->prefix, suffix, "wb");
      if (!f)
         return false;
      bool written = fwrite(v->bin, 1, bytes, f) == bytes;
      if (!close_output(f) || !written)
         return false;
   }
   const struct ir3_info *s = &v->info;
   fprintf(report,
           "%s{\"kind\":\"%s\",\"binary_blake3\":\"%s\",\"code_bytes\":%zu,"
           "\"instructions\":%u,\"preamble_instructions\":%u,\"nops\":%u,"
           "\"mov\":%u,\"conversion\":%u,\"loops\":%u,\"private_stores\":%u,"
           "\"private_loads\":%u,\"private_bytes\":%u,\"full_registers_vec4\":%d,\"half_registers_vec4\":%d,"
           "\"max_waves\":%d,\"subgroup_size\":%u,\"constlen_vec4\":%u,\"merged_register_file\":%s,"
           "\"ss\":%u,\"sy\":%u,\"ss_stall_estimate\":%u,\"sy_stall_estimate\":%u,"
           "\"categories\":[%u,%u,%u,%u,%u,%u,%u,%u]}",
           index ? "," : "", kind, hash, bytes, s->instrs_count,
           s->preamble_instrs_count, s->nops_count, s->mov_count, s->cov_count,
           s->loops, s->stp_count, s->ldp_count, v->pvtmem_size, s->max_reg + 1,
           s->max_half_reg + 1, s->max_waves, s->subgroup_size, s->constlen,
           v->mergedregs ? "true" : "false",
           s->ss, s->sy, s->sstall, s->systall,
           s->instrs_per_cat[0], s->instrs_per_cat[1], s->instrs_per_cat[2],
           s->instrs_per_cat[3], s->instrs_per_cat[4], s->instrs_per_cat[5],
           s->instrs_per_cat[6], s->instrs_per_cat[7]);
   return !ferror(report);
}

static void
usage(FILE *f)
{
   fputs("Usage: ir3-offline-compiler [options] shader.spv\n"
         "  -c, --gpu NAME         Adreno name or number (default a830)\n"
         "  --gpu-id N             Legacy GPU ID\n"
         "  --chip-id N            Exact chip ID, including revision/fuse\n"
         "  --list                 List known Turnip targets\n"
         "  -n, --entry NAME       Entry point (default main)\n"
         "  --stage STAGE          vertex, fragment, compute (auto by entry)\n"
         "  --binding S:B:TYPE:N   Override descriptor type/count\n"
         "  --spec ID:SIZE:BITS    Raw specialization bits (1/2/4/8 bytes)\n"
         "  --sconst ID=VALUE      Typed specialization constant\n"
         "  --subgroup-size N     Required subgroup size\n"
         "  --push-constant-size N  Pipeline range in bytes (default 256)\n"
         "  --samples N           1, 2, 4 or 8, sample interpolation enabled\n"
         "  --robust              Robust buffer access 2 lowering\n"
         "  -o, --output PREFIX   Write ISA, raw code, NIR and JSON\n"
         "  --json                Print JSON statistics (default)\n"
         "  -h, --help            Print this help\n", f);
}

int
main(int argc, char **argv)
{
   struct options opts = {.entry = "main", .stage = -1};
   struct ir3_offline_options compile_options = {.samples = 1, .push_constant_size = 256};
   const char *model = "a830", *typed[128];
   unsigned typed_count = 0;
   bool explicit_id = false, list = false;
   for (int i = 1; i < argc; i++) {
      const char *arg = argv[i];
      if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
         usage(stdout);
         return 0;
      }
      if (!strcmp(arg, "--list")) { list = true; continue; }
      if (!strcmp(arg, "--json")) continue;
      if (!strcmp(arg, "--robust")) { compile_options.robust = true; continue; }
      if (arg[0] != '-' || !strcmp(arg, "-")) {
         if (opts.input) goto invalid;
         opts.input = arg;
         continue;
      }
      if (++i >= argc) goto invalid;
      const char *value = argv[i];
      uint64_t n;
      if (!strcmp(arg, "-c") || !strcmp(arg, "--gpu")) {
         model = value;
      } else if (!strcmp(arg, "-n") || !strcmp(arg, "--entry") || !strcmp(arg, "--name")) {
         opts.entry = value;
      } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
         opts.prefix = value;
      } else if (!strcmp(arg, "--stage")) {
         opts.stage = !strcmp(value, "vertex") ? MESA_SHADER_VERTEX :
                      !strcmp(value, "fragment") ? MESA_SHADER_FRAGMENT :
                      !strcmp(value, "compute") ? MESA_SHADER_COMPUTE : -1;
         if (opts.stage < 0) goto invalid;
      } else if (!strcmp(arg, "--binding")) {
         if (!add_binding(&opts, value)) goto invalid;
      } else if (!strcmp(arg, "--spec")) {
         if (!add_spec(&opts, value)) goto invalid;
      } else if (!strcmp(arg, "--sconst")) {
         if (typed_count == ARRAY_SIZE(typed)) goto invalid;
         typed[typed_count++] = value;
      } else if (!strcmp(arg, "--gpu-id")) {
         if (!number(value, &n) || !n || n > UINT32_MAX) goto invalid;
         compile_options.dev_id.gpu_id = n;
         explicit_id = true;
      } else if (!strcmp(arg, "--chip-id")) {
         if (!number(value, &n) || !n) goto invalid;
         compile_options.dev_id.chip_id = n;
         explicit_id = true;
      } else if (!strcmp(arg, "--subgroup-size")) {
         if (!number(value, &n) || n > 128 || !n) goto invalid;
         compile_options.subgroup_size = n;
      } else if (!strcmp(arg, "--push-constant-size")) {
         if (!number(value, &n) || n > 256 || n % 4) goto invalid;
         compile_options.push_constant_size = n;
      } else if (!strcmp(arg, "--samples")) {
         if (!number(value, &n) || (n != 1 && n != 2 && n != 4 && n != 8)) goto invalid;
         compile_options.samples = n;
      } else {
         goto invalid;
      }
   }
   if (list) {
      struct fd_dev_id id;
      for (unsigned i = 0; (model = fd_dev_name_at(i, &id)); i++) {
         const struct fd_dev_info *info = fd_dev_info_raw(&id);
         if (info && info->chip >= 6 && info->chip <= 8)
            printf("%s gpu_id=%u chip_id=0x%" PRIx64 "\n", model, id.gpu_id, id.chip_id);
      }
      return 0;
   }
   if (!opts.input) goto invalid;
   if (!explicit_id && !target(model, &compile_options.dev_id)) {
      fprintf(stderr, "Unknown GPU: %s; use --list\n", model);
      return 1;
   }
   const struct fd_dev_info *info = fd_dev_info_raw(&compile_options.dev_id);
   if (!info || info->chip < 6 || info->chip > 8) {
      fputs("Unsupported Turnip target; use --list\n", stderr);
      return 1;
   }
   if (compile_options.subgroup_size &&
       compile_options.subgroup_size != info->threadsize_base &&
       (compile_options.subgroup_size != info->threadsize_base * 2 ||
        !info->props.supports_double_threadsize)) {
      fputs("Unsupported subgroup size for this target\n", stderr);
      return 1;
   }
   uint32_t *words;
   size_t bytes;
   if (!read_spirv(opts.input, &words, &bytes)) return 1;
   bool ok = false;
   struct ir3_offline_context *ctx = NULL;
   spv_context validator = spvContextCreate(SPV_ENV_VULKAN_1_3);
   spv_diagnostic diagnostic = NULL;
   spv_const_binary_t binary = {words, bytes / 4};
   if (spvValidate(validator, &binary, &diagnostic) != SPV_SUCCESS) {
      if (diagnostic) spvDiagnosticPrint(diagnostic);
      fputs("SPIR-V validation failed\n", stderr);
      goto cleanup;
   }
   if (!select_entry(words, bytes / 4, &opts)) {
      fputs("Invalid, missing, ambiguous or unsupported entry point; supported stages: vertex, fragment, compute. RT pipeline stages require a separate Turnip ABI\n", stderr);
      goto cleanup;
   }
   for (unsigned i = 0; i < typed_count; i++)
      if (!typed_spec(&opts, words, bytes / 4, typed[i])) {
         fputs("Invalid typed specialization constant\n", stderr);
         goto cleanup;
      }
   compile_options.stage = opts.stage;
   ctx = ir3_offline_context_create(&compile_options);
   if (!ctx) { fputs("Compiler context initialization failed\n", stderr); goto cleanup; }
   VkSpecializationInfo spec = {.mapEntryCount = opts.spec_count,
      .pMapEntries = opts.specs, .dataSize = sizeof(uint64_t) * opts.spec_count,
      .pData = opts.spec_data};
   nir_shader *nir = ir3_offline_to_nir(ctx, words, bytes, opts.entry, &spec);
   if (!nir) { fputs("Turnip SPIR-V translation failed\n", stderr); goto cleanup; }
   if (!reflect(nir, &opts)) {
      fputs("Descriptor reflection failed; supply --binding for runtime arrays\n", stderr);
      ralloc_free(nir);
      goto cleanup;
   }
   if (opts.prefix) {
      FILE *f = output_file(opts.prefix, "initial.nir", "w");
      if (!f) { ralloc_free(nir); goto cleanup; }
      nir_print_shader(nir, f);
      if (!close_output(f)) { ralloc_free(nir); goto cleanup; }
   }
   struct ir3_shader_variant *v = ir3_offline_compile(ctx, nir, opts.bindings, opts.binding_count);
   if (!v) { fputs("IR3 compilation failed\n", stderr); goto cleanup; }
   char input_hash[BLAKE3_HEX_LEN];
   hash_hex(words, bytes, input_hash);
   FILE *report = opts.prefix ? output_file(opts.prefix, "json", "w") : stdout;
   if (!report) goto cleanup;
   fputs("{\"compiler\":\"Turnip IR3\",\"mesa_git\":", report);
   json_string(report, MESA_GIT_SHA1);
   fputs(",\"gpu\":", report);
   json_string(report, fd_dev_name(&compile_options.dev_id));
   fprintf(report, ",\"gpu_id\":%u,\"chip_id\":\"0x%" PRIx64 "\",\"stage\":",
           compile_options.dev_id.gpu_id, compile_options.dev_id.chip_id);
   json_string(report, _mesa_shader_stage_to_string(opts.stage));
   fputs(",\"entry\":", report);
   json_string(report, opts.entry);
   fprintf(report, ",\"input_blake3\":\"%s\",\"cache\":false,\"pipeline_state\":\"reflected fixture\","
           "\"samples\":%u,\"robust\":%s,\"subgroup_request\":%u,\"push_constant_range_bytes\":%u,\"bindings\":[",
           input_hash, compile_options.samples, compile_options.robust ? "true" : "false",
           compile_options.subgroup_size, compile_options.push_constant_size);
   for (unsigned i = 0; i < opts.binding_count; i++) {
      const struct ir3_offline_binding *b = &opts.bindings[i];
      fprintf(report, "%s{\"set\":%u,\"binding\":%u,\"type\":%u,\"count\":%u,\"explicit\":%s}",
              i ? "," : "", b->set, b->vk.binding, b->vk.descriptorType,
              b->vk.descriptorCount, b->overridden ? "true" : "false");
   }
   fputs("],\"specialization\":[", report);
   for (unsigned i = 0; i < opts.spec_count; i++)
      fprintf(report, "%s{\"id\":%u,\"size\":%zu,\"bits\":\"0x%" PRIx64 "\"}",
              i ? "," : "", opts.specs[i].constantID, opts.specs[i].size, opts.spec_data[i]);
   fputs("],\"variants\":[", report);
   ok = dump_variant(report, &opts, v, 0, "main");
   if (ok && v->binning) ok = dump_variant(report, &opts, v->binning, 1, "binning");
   fputs("]}\n", report);
   ok = report == stdout ? !ferror(report) && !fflush(report) : close_output(report) && ok;
   if (ok && opts.prefix && v->disasm_info.nir) {
      FILE *f = output_file(opts.prefix, "final.nir", "w");
      if (!f) { ok = false; goto cleanup; }
      fputs(v->disasm_info.nir, f);
      ok = close_output(f);
   }
cleanup:
   ir3_offline_context_destroy(ctx);
   spvDiagnosticDestroy(diagnostic);
   spvContextDestroy(validator);
   free(words);
   return ok ? 0 : 1;
invalid:
   fputs("Invalid arguments\n", stderr);
   usage(stderr);
   return 1;
}
