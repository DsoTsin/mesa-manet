#include <inttypes.h>

#include "git_sha1.h"
#include "kraidoc_report.h"
#include "pan_model.h"
#include "panvk_shader.h"

static void
string(FILE *f, const char *text)
{
   fputc('"', f);
   for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
      if (*p == '"' || *p == '\\')
         fprintf(f, "\\%c", *p);
      else if (*p < 32)
         fprintf(f, "\\u%04x", *p);
      else
         fputc(*p, f);
   }
   fputc('"', f);
}

static void
property(FILE *f, const char *name, const char *display, uint64_t value,
         bool comma)
{
   fprintf(f,
           "%s{\"name\":\"%s\",\"display_name\":\"%s\","
           "\"description\":\"%s\",\"value\":%" PRIu64 "}",
           comma ? "," : "", name, display, display, value);
}

static void
boolean(FILE *f, const char *name, bool value)
{
   fprintf(f,
           ",{\"name\":\"%s\",\"display_name\":\"%s\","
           "\"description\":\"%s\",\"value\":%s}",
           name, name, name, value ? "true" : "false");
}

static const char *const pipelines[] = {
   "arith_total", "arith_fma", "arith_cvt", "arith_sfu",
   "load_store",  "varying",   "texture",
};

static const char *const displays[] = {
   "Arithmetic", "Arith FMA", "Arith CVT", "Arith SFU",
   "Load/Store", "Varying",   "Texture",
};

static void
producer(FILE *f, const char *schema)
{
   unsigned major = 0, minor = 0, patch = 0;
   sscanf(PACKAGE_VERSION, "%u.%u.%u", &major, &minor, &patch);
   fprintf(
      f, "\"producer\":{\"name\":\"kraidoc\",\"version\":[%u,%u,%u],\"build\":",
      major, minor, patch);
   string(f, MESA_GIT_SHA1[0] ? MESA_GIT_SHA1 : "unknown");
   fprintf(f, "},\"schema\":{\"name\":\"%s\",\"version\":0}", schema);
}

bool
kraidoc_write_list(FILE *f, bool json)
{
   if (json)
      fputs("{\"cores\":[", f);
   bool comma = false;
   for (unsigned i = 0; i < pan_model_count; i++) {
      const struct pan_model *model = &pan_model_list[i];
      unsigned arch = model->gpu_prod_id >> 16;
      if (arch < 9 || arch > 11)
         continue;
      if (json) {
         fprintf(f, "%s{\"apis\":[\"Vulkan\"],\"core\":", comma ? "," : "");
         string(f, model->name);
         fputc('}', f);
      } else {
         fprintf(f, "%s variant %u (Valhall %u): Vulkan SPIR-V\n", model->name,
                 model->gpu_variant, arch);
      }
      comma = true;
   }
   if (json) {
      fputs("],", f);
      producer(f, "list");
      fputs("}\n", f);
   }
   return !ferror(f);
}

bool
kraidoc_write_info(FILE *f, bool json, uint64_t gpu_id, uint32_t gpu_variant)
{
   const struct pan_model *model = pan_get_model(gpu_id, gpu_variant);
   if (json) {
      fputs("{\"hardware\":{\"architecture\":\"Valhall\",\"core\":", f);
      string(f, model->name);
      fprintf(
         f,
         ",\"revision\":\"r%up%u\"},\"driver\":\"Mesa %s\","
         "\"apis\":{\"vulkan\":{\"max_version\":1.6},"
         "\"opengles\":{\"max_version\":null},\"opencl\":{\"max_version\":null}},"
         "\"kraid\":{\"input\":\"SPIR-V\",\"stages\":[\"vertex\",\"fragment\",\"compute\"]},",
         (unsigned)PAN_VERSION_MAJOR(gpu_id), (unsigned)PAN_VERSION_MINOR(gpu_id), PACKAGE_VERSION);
      producer(f, "info");
      fputs("}\n", f);
   } else {
      fprintf(f,
              "Hardware: %s r%up%u\nArchitecture: Valhall\nDriver: Mesa %s\n"
              "API: Vulkan\nInput: SPIR-V 1.0 through 1.6\n"
              "Stages: vertex, fragment, compute\n",
              model->name, (unsigned)PAN_VERSION_MAJOR(gpu_id), (unsigned)PAN_VERSION_MINOR(gpu_id),
              PACKAGE_VERSION);
   }
   return !ferror(f);
}

static void
cycles(FILE *f, const struct valhall_stats *s, bool compute)
{
   float values[] = {s->alu, s->fma, s->cvt, s->sfu, s->ls, s->v, s->t};
   fputs("{\"pipelines\":[", f);
   bool comma = false;
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++) {
      if (compute && i == 5)
         continue;
      fprintf(f, "%s\"%s\"", comma ? "," : "", pipelines[i]);
      comma = true;
   }
   fputs("],\"total_cycles\":{\"cycle_count\":[", f);
   comma = false;
   float bound = MAX4(s->alu, s->ls, s->v, s->t);
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++) {
      if (compute && i == 5)
         continue;
      fprintf(f, "%s%.9g", comma ? "," : "", values[i]);
      comma = true;
   }
   fputs("],\"bound_pipelines\":[", f);
   comma = false;
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++) {
      if ((i >= 1 && i <= 3) || (compute && i == 5) || bound == 0 ||
          values[i] != bound)
         continue;
      fprintf(f, "%s\"%s\"", comma ? "," : "", pipelines[i]);
      comma = true;
   }
   fputs("]},\"shortest_path_cycles\":null,\"longest_path_cycles\":null}", f);
}

static void
program(FILE *f, bool json, bool detailed, const char *name, unsigned index,
        const struct panvk_shader_variant *variant,
        const struct pan_stats *stats, bool pilot, bool comma)
{
   const struct valhall_stats *s = &stats->valhall;
   unsigned occupancy = s->threads * 50;
   unsigned stack = s->stack_alloca_bytes + s->stack_spill_bytes;
   if (!json) {
      fprintf(f, "\n%s shader (variant %u)\n=================\n\n", name,
              index);
      fprintf(f,
              "Work registers: %u (%u%% occupancy)\n"
              "Uniform registers: %u\nShared storage size: %u bytes\n"
              "Stack size: %u bytes\nStack spilling: %s\n",
              s->registers_used, occupancy, s->uniforms_used * 2,
              variant->info.wls_size, stack,
              s->stack_spill_bytes ? "true" : "false");
      fprintf(f, "Code size: %u bytes\nLoops: %u\nSpills/fills: %u/%u\n",
              s->code_size, s->loops, s->spills, s->fills);
      const char *labels[] = {"A", "FMA", "CVT", "SFU", "LS", "V", "T"};
      float values[] = {s->alu, s->fma, s->cvt, s->sfu, s->ls, s->v, s->t};
      bool compute = variant->info.stage == MESA_SHADER_COMPUTE;
      fprintf(f, "\n%-25s", "");
      for (unsigned i = 0; i < ARRAY_SIZE(values); i++)
         if (!((!detailed && i >= 1 && i <= 3) || (compute && i == 5)))
            fprintf(f, " %8s", labels[i]);
      fputs("    Bound\n", f);
      fprintf(f, "%-25s", "Total instruction cycles:");
      for (unsigned i = 0; i < ARRAY_SIZE(values); i++)
         if (!((!detailed && i >= 1 && i <= 3) || (compute && i == 5)))
            fprintf(f, " %8.3f", values[i]);
      float bound = MAX4(s->alu, s->ls, s->v, s->t);
      for (unsigned i = 0; i < ARRAY_SIZE(values); i++)
         if (!(i >= 1 && i <= 3) && !(compute && i == 5) && bound > 0 &&
             values[i] == bound)
            fprintf(f, " %s", labels[i]);
      fputc('\n', f);
      if (detailed)
         fprintf(f,
                 "Stack alloca: %u bytes\nStack spill: %u bytes\n"
                 "Instructions: %u\nMaximum threads: %u\n",
                 s->stack_alloca_bytes, s->stack_spill_bytes, s->instrs,
                 s->threads);
      return;
   }
   fprintf(f,
           "%s{\"name\":\"%s\",\"kraid_variant\":%u,\"is_pilot\":%s,"
           "\"properties\":[",
           comma ? "," : "", name, index, pilot ? "true" : "false");
   property(f, "work_registers_used", "Work Registers Used", s->registers_used,
            false);
   property(f, "thread_occupancy", "Thread Occupancy", occupancy, true);
   property(f, "uniform_registers_used", "Uniform Registers Used",
            s->uniforms_used * 2, true);
   property(f, "shared_storage_size", "Shared Storage Size",
            variant->info.wls_size, true);
   property(f, "stack_size", "Stack Size", stack, true);
   boolean(f, "has_stack_spilling", s->stack_spill_bytes != 0);
   property(f, "stack_spill_bytes", "Stack Spill Size (bytes)",
            s->stack_spill_bytes, true);
   property(f, "stack_alloca_bytes", "Stack Alloca Size (bytes)",
            s->stack_alloca_bytes, true);
   property(f, "instructions", "Instructions", s->instrs, true);
   property(f, "code_size", "Code Size", s->code_size, true);
   property(f, "constant_data_size", "Constant Data Size",
            s->constant_data_size, true);
   property(f, "loops", "Loops", s->loops, true);
   property(f, "spills", "Spills", s->spills, true);
   property(f, "fills", "Fills", s->fills, true);
   property(f, "maximum_threads", "Maximum Threads", s->threads, true);
   fputs("],\"performance\":", f);
   cycles(f, s, variant->info.stage == MESA_SHADER_COMPUTE);
   fputc('}', f);
}

bool
kraidoc_write_report(FILE *f, bool json, bool detailed,
                     const struct kraidoc_report_context *ctx,
                     const struct panvk_shader *shader)
{
   const struct pan_model *model = pan_get_model(ctx->gpu_id, ctx->gpu_variant);
   const char *stage = ctx->stage == MESA_SHADER_VERTEX     ? "Vertex"
                       : ctx->stage == MESA_SHADER_FRAGMENT ? "Fragment"
                                                            : "Compute";
   unsigned major = 0, minor = 0, patch = 0;
   sscanf(PACKAGE_VERSION, "%u.%u.%u", &major, &minor, &patch);
   if (json) {
      fprintf(f,
              "{\"producer\":{\"name\":\"kraidoc\",\"version\":[%u,%u,%u],"
              "\"build\":",
              major, minor, patch);
      string(f, MESA_GIT_SHA1[0] ? MESA_GIT_SHA1 : "unknown");
      fputs(
         "},\"schema\":{\"name\":\"performance\",\"version\":2},\"shaders\":[{"
         "\"filename\":",
         f);
      string(f, ctx->filename);
      fputs(",\"driver\":", f);
      string(f, "Mesa " PACKAGE_VERSION);
      fputs(",\"hardware\":{\"architecture\":\"Valhall\",\"core\":", f);
      string(f, model->name);
      fprintf(f, ",\"revision\":\"r%up%u\",\"pipelines\":[",
              (unsigned)PAN_VERSION_MAJOR(ctx->gpu_id), (unsigned)PAN_VERSION_MINOR(ctx->gpu_id));
      for (unsigned i = 0; i < ARRAY_SIZE(pipelines); i++)
         fprintf(
            f,
            "%s{\"name\":\"%s\",\"display_name\":\"%s\",\"description\":\"%s\"}",
            i ? "," : "", pipelines[i], displays[i], displays[i]);
      fprintf(
         f,
         "]},\"shader\":{\"api\":\"Vulkan\",\"type\":\"%s\"},"
         "\"notes\":[\"Cycles use Mesa's Valhall static cost model.\","
         "\"Path cycle bounds and 16-bit arithmetic percentages are unavailable.\"],"
         "\"warnings\":[],\"properties\":[],\"kraid\":",
         stage);
      ctx->write_context(f, ctx->context, ctx->input_hash);
      fputs(",\"variants\":[", f);
   } else {
      fprintf(f,
              "kraidoc %s\n\nConfiguration\n=============\n\n"
              "Hardware: %s r%up%u\nArchitecture: Valhall\n"
              "Driver: Mesa %s\nShader type: Vulkan %s\n",
              PACKAGE_VERSION, model->name, (unsigned)PAN_VERSION_MAJOR(ctx->gpu_id),
              (unsigned)PAN_VERSION_MINOR(ctx->gpu_id), PACKAGE_VERSION, stage);
   }
   bool comma = false;
   panvk_shader_foreach_variant_const(shader, variant)
   {
      if (!variant->bin_size)
         continue;
      unsigned index = variant - shader->variants;
      bool idvs = ctx->stage == MESA_SHADER_VERTEX && variant->info.vs.idvs;
      program(f, json, detailed, idvs ? "Position" : "Main", index, variant,
              &variant->info.stats, false, comma);
      comma = true;
      if (idvs && variant->info.vs.secondary_offset)
         program(f, json, detailed, "Varying", index, variant,
                 &variant->info.stats_idvs_varying, false, true);
      if (variant->preamble)
         program(f, json, detailed, "Pilot", index, variant->preamble,
                 &variant->preamble->info.stats, true, true);
   }
   if (json)
      fputs("]}]}\n", f);
   else
      fputs("\nCycle estimates use Mesa's Valhall static cost model.\n"
            "Path cycle bounds and 16-bit arithmetic percentages: N/A\n",
            f);
   return !ferror(f);
}
