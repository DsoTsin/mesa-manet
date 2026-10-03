#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spirv-tools/libspirv.h>

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
#include "nir.h"
#include "vk_alloc.h"
#include "vk_nir.h"
#include "vk_util.h"

#include "kraidoc_context.h"
#include "kraidoc_report.h"
#include "pan_model.h"
#include "pan_shader.h"
#include "panvk_descriptor_set_layout.h"
#include "panvk_shader.h"

#define DECLARE_ARCH(arch)                                                      \
   extern const struct vk_device_shader_ops panvk_v##arch##_offline_shader_ops; \
   VkResult panvk_v##arch##_compile_shader_offline(                             \
      struct panvk_device *, struct vk_shader_compile_info *, bool,             \
      struct vk_shader **);                                                     \
   VkResult panvk_v##arch##_CreateDescriptorSetLayout(                          \
      VkDevice, const VkDescriptorSetLayoutCreateInfo *,                        \
      const VkAllocationCallbacks *, VkDescriptorSetLayout *)

DECLARE_ARCH(9);
DECLARE_ARCH(10);
DECLARE_ARCH(11);

struct arch_ops {
   const struct vk_device_shader_ops *frontend;
   VkResult (*compile)(struct panvk_device *, struct vk_shader_compile_info *,
                       bool, struct vk_shader **);
   VkResult (*create_layout)(VkDevice, const VkDescriptorSetLayoutCreateInfo *,
                             const VkAllocationCallbacks *,
                             VkDescriptorSetLayout *);
};

struct binding {
   unsigned set;
   VkDescriptorSetLayoutBinding vk;
   bool explicit;
};

struct options {
   const char *input, *prefix, *output, *entry, *cpu;
   uint64_t gpu_id;
   uint32_t gpu_variant, cores;
   int stage;
   bool json, nir, preamble, legacy, detailed, list, info;
   struct binding bindings[256];
   unsigned binding_count;
   VkSpecializationMapEntry specs[128];
   uint64_t spec_data[128];
   unsigned spec_count;
   const char *sconst[128];
   unsigned sconst_count;
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
      opts->bindings[opts->binding_count++] = (struct binding){
         .set = set,
         .explicit = true,
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
      struct binding *binding = NULL;
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
             opts->binding_count == ARRAY_SIZE(opts->bindings))
            return false;
         binding = &opts->bindings[opts->binding_count++];
         *binding = (struct binding){
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
dump_program(const struct options *opts, FILE *report, unsigned index,
             const char *kind, const struct panvk_shader_variant *variant,
             unsigned offset, unsigned size, const struct pan_stats *stats,
             bool *comma)
{
   char suffix[96], hash[BLAKE3_HEX_LEN];
   const uint8_t *data = (const uint8_t *)variant->bin_ptr + offset;
   hash_hex(data, size, hash);
   snprintf(suffix, sizeof(suffix), "%u.%s.isa.txt", index, kind);
   FILE *f = output_file(opts->prefix, suffix, "w");
   if (!f)
      return false;
   pan_disassemble(f, data, size, opts->gpu_id, false);
   if (!close_output(f))
      return false;
   snprintf(suffix, sizeof(suffix), "%u.%s.bin", index, kind);
   f = output_file(opts->prefix, suffix, "wb");
   if (!f)
      return false;
   fwrite(data, 1, size, f);
   if (!close_output(f))
      return false;
   if (*comma)
      fputc(',', report);
   *comma = true;
   fprintf(report,
           "{\"variant\":%u,\"kind\":\"%s\",\"instructions\":%u,"
           "\"spills\":%u,\"fills\":%u,\"work_reg_count\":%u,"
           "\"tls_size\":%u,\"wls_size\":%u,\"binary_bytes\":%u,"
           "\"binary_blake3\":\"%s\"}",
           index, kind, stats->valhall.instrs, stats->valhall.spills,
           stats->valhall.fills, variant->info.work_reg_count,
           variant->info.tls_size, variant->info.wls_size, size, hash);
   if (opts->legacy && !opts->json)
      printf("%u %s: %u instructions, %u registers, %u spills, %u fills, "
             "%u bytes\n",
             index, kind, stats->valhall.instrs, variant->info.work_reg_count,
             stats->valhall.spills, stats->valhall.fills, size);
   return !ferror(report);
}

static void
write_context(FILE *report, const struct options *opts, const char *input_hash)
{
   fprintf(report,
           "{\"compiler\":\"Kraid\",\"mesa_version\":\"%s\",\"mesa_git_sha\":"
           "\"%s\","
           "\"input_blake3\":\"%s\",\"input\":",
           PACKAGE_VERSION, MESA_GIT_SHA1[0] ? MESA_GIT_SHA1 : "unknown",
           input_hash);
   json_string(report, opts->input);
   fputs(",\"entry\":", report);
   json_string(report, opts->entry);
   fprintf(report,
           ",\"stage\":\"%s\",\"gpu_id\":\"0x%" PRIx64 "\","
           "\"gpu_variant\":%u,\"shader_cores\":%u,"
           "\"subgroup_size\":16,\"preamble_enabled\":%s,"
           "\"compiler_flags\":%u,"
           "\"abi\":\"PanVK\",\"graphics_state\":null,"
           "\"robustness\":\"disabled\","
           "\"original_pipeline_state\":false,\"specialization\":[",
           _mesa_shader_stage_to_string(opts->stage), opts->gpu_id,
           opts->gpu_variant, opts->cores, opts->preamble ? "true" : "false",
           pan_get_compiler_flags(pan_arch(opts->gpu_id)));
   for (unsigned i = 0; i < opts->spec_count; i++)
      fprintf(report, "%s{\"id\":%u,\"bytes\":%zu,\"bits\":\"0x%" PRIx64 "\"}",
              i ? "," : "", opts->specs[i].constantID, opts->specs[i].size,
              opts->spec_data[i]);
   fputs("],\"bindings\":[", report);
   for (unsigned i = 0; i < opts->binding_count; i++) {
      const struct binding *b = &opts->bindings[i];
      fprintf(report,
              "%s{\"set\":%u,\"binding\":%u,\"type\":%u,\"count\":%u,"
              "\"source\":\"%s\"}",
              i ? "," : "", b->set, b->vk.binding, b->vk.descriptorType,
              b->vk.descriptorCount,
              b->explicit ? "argument" : "NIR reflection");
   }
   fputs("],\"programs\":[", report);
}

static void
write_report_context(FILE *f, const void *options, const char *input_hash)
{
   write_context(f, options, input_hash);
   fputs("]}", f);
}

static bool
write_artifacts(const struct options *opts, struct panvk_shader *pan_shader,
                const char *input_hash)
{
   if (!opts->prefix)
      return true;
   FILE *report = output_file(opts->prefix, "report.json", "w");
   if (!report)
      return false;
   write_context(report, opts, input_hash);
   bool comma = false;
   bool ok = true;
   panvk_shader_foreach_variant(pan_shader, variant) {
      if (!variant->bin_size)
         continue;
      unsigned index = variant - pan_shader->variants;
      unsigned split = variant->info.stage == MESA_SHADER_VERTEX
                          ? variant->info.vs.secondary_offset
                          : 0;
      ok &= dump_program(opts, report, index, "main", variant, 0,
                         split ? split : variant->bin_size,
                         &variant->info.stats, &comma);
      if (split)
         ok &= dump_program(opts, report, index, "varying", variant, split,
                            variant->bin_size - split,
                            &variant->info.stats_idvs_varying, &comma);
      if (variant->preamble)
         ok &= dump_program(opts, report, index, "pilot", variant->preamble, 0,
                            variant->preamble->bin_size,
                            &variant->preamble->info.stats, &comma);
      if (opts->nir && variant->nir_str) {
         char suffix[96];
         snprintf(suffix, sizeof(suffix), "%u.main.nir.txt", index);
         FILE *f = output_file(opts->prefix, suffix, "w");
         if (f) {
            fputs(variant->nir_str, f);
            ok &= close_output(f);
         } else {
            ok = false;
         }
      }
   }
   fputs("]}\n", report);
   ok &= close_output(report);
   if (opts->legacy && opts->json && ok) {
      FILE *f = output_file(opts->prefix, "report.json", "r");
      if (!f) {
         ok = false;
      } else {
         int c;
         while ((c = fgetc(f)) != EOF)
            fputc(c, stdout);
         ok &= close_output(f) && !ferror(stdout);
      }
   }
   return ok;
}

static bool
compile(const uint32_t *words, size_t bytes, struct options *opts)
{
   struct arch_ops ops;
   switch (pan_arch(opts->gpu_id)) {
#define ARCH_CASE(arch)                                                        \
   case arch:                                                                  \
      ops = (struct arch_ops){&panvk_v##arch##_offline_shader_ops,             \
                              panvk_v##arch##_compile_shader_offline,          \
                              panvk_v##arch##_CreateDescriptorSetLayout};      \
      break
      ARCH_CASE(9);
      ARCH_CASE(10);
      ARCH_CASE(11);
#undef ARCH_CASE
   default:
      return false;
   }
   bool ok = false;
   nir_shader *nir = NULL;
   struct vk_shader *shader = NULL;
   struct vk_descriptor_set_layout *layouts[MAX_SETS] = {0};
   unsigned set_count = 0;
   struct panvk_instance instance = {0};
   instance.vk.app_info.api_version = VK_API_VERSION_1_4;
   struct kraidoc_target kmod = {0};
   kmod.props.gpu_id = opts->gpu_id;
   kmod.props.gpu_variant = opts->gpu_variant;
   kmod.props.shader_present = BITFIELD64_MASK(opts->cores);
   struct panvk_physical_device physical = {0};
   physical.kmod.dev = &kmod;
   physical.model = pan_get_model(opts->gpu_id, opts->gpu_variant);
   physical.vk.instance = &instance.vk;
   memset(&physical.vk.supported_features, 1,
          sizeof(physical.vk.supported_features));
   physical.vk.properties.apiVersion = VK_API_VERSION_1_4;
   physical.vk.properties.subgroupSize = 16;
   struct panvk_device device = {0};
   device.vk.physical = &physical.vk;
   device.vk.alloc = *vk_default_allocator();
   vk_object_base_init(&device.vk, &device.vk.base, VK_OBJECT_TYPE_DEVICE);
   device.vk.base.client_visible = true;
   vk_object_base_instance_init(&instance.vk, &instance.vk.base,
                                VK_OBJECT_TYPE_INSTANCE);
   instance.vk.base.client_visible = true;
   list_inithead(&instance.vk.debug_report.callbacks);
   list_inithead(&instance.vk.debug_utils.callbacks);
   glsl_type_singleton_init_or_ref();
   struct vk_pipeline_robustness_state robustness = {
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   struct spirv_to_nir_options spirv_options =
      ops.frontend->get_spirv_options(&physical.vk, opts->stage, &robustness);
   VkSpecializationInfo spec = {
      .mapEntryCount = opts->spec_count,
      .pMapEntries = opts->specs,
      .dataSize = opts->spec_count * sizeof(uint64_t),
      .pData = opts->spec_data,
   };
   nir = vk_spirv_to_nir(
      &device.vk, words, bytes, opts->stage, opts->entry, &spec, &spirv_options,
      ops.frontend->get_nir_options(&physical.vk, opts->stage, &robustness),
      false, NULL);
   if (!nir) {
      fputs("SPIR-V translation failed\n", stderr);
      goto cleanup;
   }
   if (!reflect(nir, opts)) {
      fputs("Descriptor reflection failed; supply --binding for runtime arrays "
            "or a compatible layout\n",
            stderr);
      goto cleanup;
   }
   for (unsigned i = 0; i < opts->binding_count; i++)
      set_count = MAX2(set_count, opts->bindings[i].set + 1);
   for (unsigned set = 0; set < set_count; set++) {
      VkDescriptorSetLayoutBinding bindings[256];
      unsigned count = 0;
      for (unsigned i = 0; i < opts->binding_count; i++)
         if (opts->bindings[i].set == set)
            bindings[count++] = opts->bindings[i].vk;
      VkDescriptorSetLayoutCreateInfo create = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = count,
         .pBindings = bindings,
      };
      VkDescriptorSetLayout handle;
      if (ops.create_layout(panvk_device_to_handle(&device), &create, NULL,
                            &handle) != VK_SUCCESS)
         goto cleanup;
      layouts[set] = &panvk_descriptor_set_layout_from_handle(handle)->vk;
   }
   ops.frontend->preprocess_nir(&physical.vk, nir, &robustness);
   struct vk_shader_compile_info info = {
      .stage = opts->stage,
      .flags = opts->nir
                  ? VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA
                  : 0,
      .next_stage_mask =
         opts->stage == MESA_SHADER_VERTEX ? VK_SHADER_STAGE_FRAGMENT_BIT : 0,
      .nir = nir,
      .robustness = &robustness,
      .set_layout_count = set_count,
      .set_layouts = layouts,
   };
   VkResult result = ops.compile(&device, &info, opts->preamble, &shader);
   if (result != VK_SUCCESS) {
      fprintf(stderr, "Kraid compilation failed: %d\n", result);
      goto cleanup;
   }
   char input_hash[BLAKE3_HEX_LEN];
   hash_hex(words, bytes, input_hash);
   struct panvk_shader *pan_shader =
      container_of(shader, struct panvk_shader, vk);
   ok = write_artifacts(opts, pan_shader, input_hash);
   if (ok && !opts->legacy) {
      FILE *report = opts->output ? fopen(opts->output, "w") : stdout;
      if (!report) {
         fprintf(stderr, "Cannot open %s: %s\n", opts->output, strerror(errno));
         ok = false;
      } else {
         struct kraidoc_report_context ctx = {
            .filename = opts->input,
            .entrypoint = opts->entry,
            .input_hash = input_hash,
            .gpu_id = opts->gpu_id,
            .gpu_variant = opts->gpu_variant,
            .shader_cores = opts->cores,
            .stage = opts->stage,
            .preamble_enabled = opts->preamble,
            .context = opts,
            .write_context = write_report_context,
         };
         ok = kraidoc_write_report(report, opts->json, opts->detailed, &ctx,
                                   pan_shader);
         if (report != stdout)
            ok &= close_output(report);
      }
   }
cleanup:
   if (shader)
      shader->ops->destroy(&device.vk, shader, NULL);
   ralloc_free(nir);
   for (unsigned set = 0; set < set_count; set++)
      if (layouts[set])
         vk_descriptor_set_layout_unref(&device.vk, layouts[set]);
   glsl_type_singleton_decref();
   return ok;
}

static void
usage(FILE *f)
{
   fputs(
      "Usage: kraidoc [options] shader.spv\n"
      "  -c, --core NAME          Mali-G610 (default), G710, G310v1..v5,\n"
      "                          G57, G68, G615, G715\n"
      "  --gpu-id N --gpu-variant N  Exact known Valhall target\n"
      "  --shader-cores N         CPU compile context, default 4\n"
      "  -n, --name NAME          Entry point, default main (--entry alias)\n"
      "  -v/--vertex -f/--fragment -C/--compute\n"
      "  --stage STAGE            vertex, fragment, compute (auto by entry)\n"
      "  -o, --output FILE        Statistics report (stdout by default)\n"
      "  --format text|json       malioc statistics output format\n"
      "  -d, --detailed           Detailed statistics\n"
      "  -S, --sconst ID=VALUE    Typed specialization value; repeatable\n"
      "  --isa-prefix PREFIX      Save ISA, binaries and Kraid JSON\n"
      "  --report-format malioc|kraid  Report schema; kraid keeps -o PREFIX\n"
      "  --spec ID:BYTES:BITS      Specialization raw bits; repeatable\n"
      "  --binding SET:BINDING:TYPE:COUNT  Descriptor layout override\n"
      "  --no-preamble --nir -l/--list -i/--info --version --help\n"
      "  --vulkan --spirv         Vulkan SPIR-V input (- for stdin)\n",
      f);
}

int
main(int argc, char **argv)
{
   struct options opts = {
      .cpu = "Mali-G610",
      .entry = "main",
      .cores = 4,
      .stage = -1,
      .preamble = true,
   };
   bool positional = false;
   for (int i = 1; i < argc; i++) {
      const char *arg = argv[i];
      if (positional || arg[0] != '-' || !strcmp(arg, "-")) {
         if (opts.input)
            goto invalid;
         opts.input = arg;
         continue;
      }
      if (!strcmp(arg, "--") || !strcmp(arg, "--ignore_rest")) {
         positional = true;
         continue;
      }
      if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
         usage(stdout);
         return 0;
      }
      if (!strcmp(arg, "--version")) {
         printf("kraidoc %s (%s)\n", PACKAGE_VERSION,
                MESA_GIT_SHA1[0] ? MESA_GIT_SHA1 : "unknown");
         return 0;
      }
      if (!strcmp(arg, "--list") || !strcmp(arg, "-l")) {
         opts.list = true;
         continue;
      }
      if (!strcmp(arg, "--info") || !strcmp(arg, "-i")) {
         opts.info = true;
         continue;
      }
      if (!strcmp(arg, "--detailed") || !strcmp(arg, "-d")) {
         opts.detailed = true;
         continue;
      }
      if (!strcmp(arg, "--vulkan") || !strcmp(arg, "--spirv"))
         continue;
      if (!strcmp(arg, "--vertex") || !strcmp(arg, "-v") ||
          !strcmp(arg, "--fragment") || !strcmp(arg, "-f") ||
          !strcmp(arg, "--compute") || !strcmp(arg, "-C")) {
         int stage = !strcmp(arg, "--vertex") || !strcmp(arg, "-v")
                        ? MESA_SHADER_VERTEX
                     : !strcmp(arg, "--fragment") || !strcmp(arg, "-f")
                        ? MESA_SHADER_FRAGMENT
                        : MESA_SHADER_COMPUTE;
         if (opts.stage >= 0 && opts.stage != stage)
            goto invalid;
         opts.stage = stage;
         continue;
      }
      if (!strcmp(arg, "--opengles") || !strcmp(arg, "--opencl") ||
          !strcmp(arg, "--kernel") || !strcmp(arg, "-k") ||
          !strcmp(arg, "--tessellation_control") || !strcmp(arg, "-t") ||
          !strcmp(arg, "--tessellation_evaluation") || !strcmp(arg, "-e") ||
          !strcmp(arg, "--geometry") || !strcmp(arg, "-g") ||
          !strncmp(arg, "--ray_", 6)) {
         fprintf(stderr,
                 "%s is unsupported; select Vulkan vertex, fragment, "
                 "or compute SPIR-V\n",
                 arg);
         return 2;
      }
      if (!strcmp(arg, "--no-preamble")) {
         opts.preamble = false;
         continue;
      }
      if (!strcmp(arg, "--nir")) {
         opts.nir = true;
         continue;
      }
      const char *value = NULL;
      char option_name[64];
      const char *equal = strchr(arg, '=');
      if (arg[1] == '-' && equal) {
         if (equal - arg >= sizeof(option_name))
            goto invalid;
         memcpy(option_name, arg, equal - arg);
         option_name[equal - arg] = 0;
         arg = option_name;
         value = equal + 1;
      } else if (arg[1] != '-' && arg[2] && strchr("cnoS", arg[1])) {
         option_name[0] = '-';
         option_name[1] = arg[1];
         option_name[2] = 0;
         value = arg + 2;
         arg = option_name;
      } else {
         if (++i == argc)
            goto invalid;
         value = argv[i];
      }
      uint64_t number_value;
      if (!strcmp(arg, "-c") || !strcmp(arg, "--core"))
         opts.cpu = value;
      else if (!strcmp(arg, "-n") || !strcmp(arg, "--entry") ||
               !strcmp(arg, "--name"))
         opts.entry = value;
      else if (!strcmp(arg, "-o") || !strcmp(arg, "--output"))
         opts.output = value;
      else if (!strcmp(arg, "--isa-prefix"))
         opts.prefix = value;
      else if (!strcmp(arg, "--format")) {
         if (strcmp(value, "json") && strcmp(value, "text"))
            goto invalid;
         opts.json = !strcmp(value, "json");
      } else if (!strcmp(arg, "--report-format")) {
         if (strcmp(value, "malioc") && strcmp(value, "kraid"))
            goto invalid;
         opts.legacy = !strcmp(value, "kraid");
      } else if (!strcmp(arg, "--stage")) {
         opts.stage = !strcmp(value, "vertex")     ? MESA_SHADER_VERTEX
                      : !strcmp(value, "fragment") ? MESA_SHADER_FRAGMENT
                      : !strcmp(value, "compute")  ? MESA_SHADER_COMPUTE
                                                   : -1;
         if (opts.stage < 0)
            goto invalid;
      } else if (!strcmp(arg, "--spec")) {
         if (!add_spec(&opts, value))
            goto invalid;
      } else if (!strcmp(arg, "--sconst") || !strcmp(arg, "-S")) {
         if (opts.sconst_count == ARRAY_SIZE(opts.sconst))
            goto invalid;
         opts.sconst[opts.sconst_count++] = value;
      } else if (!strcmp(arg, "--binding")) {
         if (!add_binding(&opts, value))
            goto invalid;
      } else if (!strcmp(arg, "--gpu-id")) {
         if (!number(value, &opts.gpu_id) || !opts.gpu_id)
            goto invalid;
      } else if (!strcmp(arg, "--gpu-variant")) {
         if (!number(value, &number_value) || number_value > UINT32_MAX)
            goto invalid;
         opts.gpu_variant = number_value;
      } else if (!strcmp(arg, "--shader-cores")) {
         if (!number(value, &number_value) || !number_value ||
             number_value > 64)
            goto invalid;
         opts.cores = number_value;
      } else {
         goto invalid;
      }
   }
   if (!opts.input && !opts.list && !opts.info)
      goto invalid;
   if (opts.list) {
      FILE *f = opts.output ? fopen(opts.output, "w") : stdout;
      if (!f)
         return 2;
      bool ok = kraidoc_write_list(f, opts.json);
      if (f != stdout)
         ok &= close_output(f);
      return ok ? 0 : 1;
   }
   if (!opts.gpu_id) {
      bool found = false;
      for (unsigned i = 0; i < pan_model_count; i++) {
         const struct pan_model *m = &pan_model_list[i];
         if (strcmp(opts.cpu, m->name) && strcmp(opts.cpu, m->name + 5))
            continue;
         unsigned arch = m->gpu_prod_id >> 16;
         if (arch < 9 || arch > 11)
            continue;
         opts.gpu_id = (uint64_t)(arch << 12 | (m->gpu_prod_id & 0xfff)) << 16;
         opts.gpu_variant = m->gpu_variant;
         found = true;
         break;
      }
      if (!found) {
         fprintf(stderr, "Unknown Valhall target: %s\n", opts.cpu);
         return 2;
      }
   }
   if (pan_arch(opts.gpu_id) < 9 || pan_arch(opts.gpu_id) > 11 ||
       !pan_get_model(opts.gpu_id, opts.gpu_variant)) {
      fputs("Only known Valhall GPU IDs and variants are supported\n", stderr);
      return 2;
   }
   if (opts.info) {
      FILE *f = opts.output ? fopen(opts.output, "w") : stdout;
      if (!f)
         return 2;
      bool ok = kraidoc_write_info(f, opts.json, opts.gpu_id, opts.gpu_variant);
      if (f != stdout)
         ok &= close_output(f);
      return ok ? 0 : 1;
   }
   uint32_t *words = NULL;
   size_t size = 0;
   if (!read_spirv(opts.input, &words, &size))
      return 2;
   spv_target_env target_env;
   if (!spvParseTargetEnv("vulkan1.4", &target_env) &&
       !spvParseTargetEnv("vulkan1.3", &target_env)) {
      fputs("SPIRV-Tools must support Vulkan 1.3 or newer\n", stderr);
      free(words);
      return 2;
   }
   spv_context context = spvContextCreate(target_env);
   spv_diagnostic diagnostic = NULL;
   spv_result_t valid =
      spvValidateBinary(context, words, size / 4, &diagnostic);
   if (valid != SPV_SUCCESS)
      fprintf(stderr, "SPIR-V validation failed: %s\n",
              diagnostic ? diagnostic->error : "unknown validation error");
   spvDiagnosticDestroy(diagnostic);
   spvContextDestroy(context);
   if (valid != SPV_SUCCESS) {
      free(words);
      return 2;
   }
   if (!select_entry(words, size / 4, &opts)) {
      fputs("Invalid SPIR-V or missing/ambiguous/unsupported entry point\n",
            stderr);
      free(words);
      return 2;
   }
   for (unsigned i = 0; i < opts.sconst_count; i++) {
      if (!typed_spec(&opts, words, size / 4, opts.sconst[i])) {
         fprintf(stderr, "Invalid or duplicate specialization value: %s\n",
                 opts.sconst[i]);
         free(words);
         return 2;
      }
   }
   if (!pan_use_kraid(pan_arch(opts.gpu_id), opts.stage, false) ||
       !pan_use_kraid(pan_arch(opts.gpu_id), opts.stage, true)) {
      fputs(
         "Kraid must be enabled for the selected stage and internal shaders; "
         "remove PAN_USE_KRAID or set it to all\n",
         stderr);
      free(words);
      return 2;
   }
   char *default_prefix = NULL;
   if (opts.legacy && !opts.prefix)
      opts.prefix = opts.output;
   if (!opts.prefix && (opts.legacy || opts.nir)) {
      if (asprintf(&default_prefix, "%s.kraid", opts.input) < 0) {
         free(words);
         return 2;
      }
      opts.prefix = default_prefix;
   }
   bool ok = compile(words, size, &opts);
   free(default_prefix);
   free(words);
   return ok ? 0 : 1;
invalid:
   usage(stderr);
   return 2;
}
