#include "agc_program.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include "c11/threads.h"
#include "util/u_atomic.h"

#include "ac_gpu_info.h"

#include "agc_debug.h"
#include "agc_driver.h"
#include "agc_driver_ops.h"
#include "agc_memory.h"
#include "agc_registers.h"
#include "agc_runtime.h"
#include "agc_shader_artifact.h"
#include "agc_surface.h"
#include "agc_isa_translate.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define REG_COMPUTE_PGM_RSRC1 0x212u
#define REG_COMPUTE_PGM_RSRC2 0x213u
#define REG_PS_RSRC1 0x00au
#define REG_PS_RSRC2 0x00bu
#define REG_GS_RSRC1 0x08au
#define REG_GS_RSRC2 0x08bu
#define REG_HS_RSRC1 0x10au
#define REG_HS_RSRC2 0x10bu
#define REG_SPI_VS_OUT_CONFIG 0x1b1u
#define REG_SPI_PS_INPUT_ENA 0x1b3u
#define REG_SPI_PS_INPUT_ADDR 0x1b4u
#define REG_SPI_PS_IN_CONTROL 0x1b6u
#define REG_SPI_SHADER_Z_FORMAT 0x1c4u
#define REG_SPI_SHADER_COL_FORMAT 0x1c5u

#define CS_W32_EN (UINT32_C(1) << 15)
#define PS_W32_EN (UINT32_C(1) << 15)
#define HS_W32_EN (UINT32_C(1) << 21)
#define GS_W32_EN (UINT32_C(1) << 22)
#define RSRC2_SCRATCH_EN UINT32_C(1)

#define CODE_ALIGNMENT 256u
#define ORACLE_TIMEOUT_MS 120000u
#define PLACED_INITIAL_SLOTS 16384u

struct placed {
   struct agc_shader_key key;
   uint64_t va;
   uint32_t pgm_rsrc1;
   uint32_t pgm_rsrc2;
   bool has_col_format;
   uint32_t col_format;
   bool has_vs_out_config;
   uint32_t vs_out_config;
   bool used;
};

enum translator {
   TRANSLATOR_ORACLE,
   TRANSLATOR_ACO,
   TRANSLATOR_AUTO,
};

static struct {
   mtx_t lock;
   bool ready;
   bool failed;
   enum translator translator;
   struct agc_shader_cache *cache;
   char directory[1024];
   char oracle[1024];
   struct placed *placed;
   uint32_t placed_slots;
   uint32_t placed_used;
   struct agc_program_counters counters;
} programs;

static once_flag programs_once = ONCE_FLAG_INIT;

static void
programs_init_lock(void)
{
   mtx_init(&programs.lock, mtx_plain);
}

static void
default_directory(char *out, size_t bytes)
{
   const char *local = getenv("LOCALAPPDATA");
   if (local && *local)
      snprintf(out, bytes, "%s\\AnyPS5\\agc-shader-cache", local);
   else
      snprintf(out, bytes, "agc-shader-cache");
}

static bool
programs_ready(const char *function)
{
   if (programs.ready)
      return true;
   if (programs.failed)
      return false;
   const char *mode = getenv("AGC_SHADER_TRANSLATOR");
   programs.translator = !mode || !*mode || !strcmp(mode, "oracle") ? TRANSLATOR_ORACLE
                         : !strcmp(mode, "auto")                     ? TRANSLATOR_AUTO
                                                                     : TRANSLATOR_ACO;
   const char *directory = getenv("AGC_SHADER_CACHE");
   if (directory && *directory)
      snprintf(programs.directory, sizeof(programs.directory), "%s", directory);
   else
      default_directory(programs.directory, sizeof(programs.directory));
   if (programs.translator != TRANSLATOR_ORACLE) {
      size_t length = strlen(programs.directory);
      snprintf(programs.directory + length, sizeof(programs.directory) - length, "-aco-v3");
   }
   const char *oracle = getenv("AGC_SHADER_ORACLE");
   if (oracle && *oracle)
      snprintf(programs.oracle, sizeof(programs.oracle), "%s", oracle);
   enum agc_shader_file_status status = agc_shader_cache_open(programs.directory, &programs.cache);
   if (status != AGC_SHADER_FILE_OK) {
      agc_report(function, "the shader cache %s cannot be opened: %s", programs.directory,
                 agc_shader_file_status_name(status));
      programs.failed = true;
      return false;
   }
   programs.placed = calloc(PLACED_INITIAL_SLOTS, sizeof(*programs.placed));
   if (!programs.placed) {
      programs.failed = true;
      return false;
   }
   programs.placed_slots = PLACED_INITIAL_SLOTS;
   programs.ready = true;
   return true;
}

static uint32_t
key_slot(const struct agc_shader_key *key, uint32_t slots)
{
   uint32_t slot;
   memcpy(&slot, key->bytes, sizeof(slot));
   return slot & (slots - 1);
}

static bool
placed_grow(void)
{
   uint32_t slots = programs.placed_slots * 2;
   struct placed *grown = calloc(slots, sizeof(*grown));
   if (!grown)
      return false;
   for (uint32_t i = 0; i < programs.placed_slots; i++) {
      if (!programs.placed[i].used)
         continue;
      uint32_t slot = key_slot(&programs.placed[i].key, slots);
      while (grown[slot].used)
         slot = (slot + 1) & (slots - 1);
      grown[slot] = programs.placed[i];
   }
   free(programs.placed);
   programs.placed = grown;
   programs.placed_slots = slots;
   return true;
}

static struct placed *
placed_find(const struct agc_shader_key *key, bool insert)
{
   if (insert && (uint64_t)(programs.placed_used + 1) * 4 > (uint64_t)programs.placed_slots * 3 && !placed_grow())
      return NULL;
   uint32_t mask = programs.placed_slots - 1;
   for (uint32_t slot = key_slot(key, programs.placed_slots);; slot = (slot + 1) & mask) {
      struct placed *entry = &programs.placed[slot];
      if (!entry->used) {
         if (!insert)
            return NULL;
         entry->used = true;
         entry->key = *key;
         programs.placed_used++;
         return entry;
      }
      if (!memcmp(entry->key.bytes, key->bytes, sizeof(key->bytes)))
         return entry;
   }
}

static bool
find_register(const struct agc_register *registers, uint32_t count, uint32_t offset, uint32_t *value)
{
   for (uint32_t i = 0; registers && i < count; i++) {
      if (registers[i].offset == offset) {
         *value = registers[i].value;
         return true;
      }
   }
   return false;
}

static uint32_t
set_register(struct agc_register *registers, uint32_t count, uint32_t offset, uint32_t value)
{
   uint32_t written = 0;
   for (uint32_t i = 0; registers && i < count; i++) {
      if (registers[i].offset == offset) {
         registers[i].value = value;
         written++;
      }
   }
   return written;
}

static enum agc_shader_isa
host_isa(void)
{
   switch (agc_host_current()) {
   case AGC_HOST_GFX103:
      return AGC_SHADER_ISA_GFX103;
   case AGC_HOST_GFX11:
      return AGC_SHADER_ISA_GFX11;
   case AGC_HOST_GFX115:
      return AGC_SHADER_ISA_GFX115;
   case AGC_HOST_GFX12:
      return AGC_SHADER_ISA_GFX12;
   default:
      return AGC_SHADER_ISA_COUNT;
   }
}

struct stage_layout {
   enum agc_shader_stage stage;
   enum agc_shader_vertex_stage vertex_stage;
   uint32_t rsrc1;
   uint32_t rsrc2;
   uint8_t wave_size;
};

static bool
stage_layout(const struct agc_shader *shader, struct stage_layout *layout)
{
   const struct agc_shader_specials *specials = shader->specials;
   uint32_t stages = specials ? specials->vgt_shader_stages_en.value : 0;
   uint32_t ps_control = 0;
   memset(layout, 0, sizeof(*layout));
   switch (shader->kind) {
   case AGC_SHADER_CS:
      layout->stage = AGC_SHADER_STAGE_COMPUTE;
      layout->rsrc1 = REG_COMPUTE_PGM_RSRC1;
      layout->rsrc2 = REG_COMPUTE_PGM_RSRC2;
      layout->wave_size = specials && (specials->dispatch_modifier & CS_W32_EN) ? 32 : 64;
      return true;
   case AGC_SHADER_PS:
      find_register(shader->cx_registers, shader->num_cx_registers, REG_SPI_PS_IN_CONTROL, &ps_control);
      layout->stage = AGC_SHADER_STAGE_FRAGMENT;
      layout->rsrc1 = REG_PS_RSRC1;
      layout->rsrc2 = REG_PS_RSRC2;
      layout->wave_size = ps_control & PS_W32_EN ? 32 : 64;
      return true;
   case AGC_SHADER_GS:
      layout->stage = AGC_SHADER_STAGE_VERTEX;
      layout->vertex_stage = AGC_SHADER_VERTEX_STAGE_ES_GS;
      layout->rsrc1 = REG_GS_RSRC1;
      layout->rsrc2 = REG_GS_RSRC2;
      layout->wave_size = stages & GS_W32_EN ? 32 : 64;
      return true;
   case AGC_SHADER_HS:
      layout->stage = AGC_SHADER_STAGE_VERTEX;
      layout->vertex_stage = AGC_SHADER_VERTEX_STAGE_LS_HS;
      layout->rsrc1 = REG_HS_RSRC1;
      layout->rsrc2 = REG_HS_RSRC2;
      layout->wave_size = stages & HS_W32_EN ? 32 : 64;
      return true;
   default:
      return false;
   }
}

static void
build_request(const struct agc_shader *shader, const struct stage_layout *layout, uint64_t guest_code,
              const uint32_t *words, uint32_t word_count, const struct agc_program_link *link,
              struct agc_shader_code_region *region, struct agc_shader_request *request)
{
   const struct radeon_info *info = agc_driver_info();
   struct agc_shader_interface *iface = &request->iface;
   uint32_t rsrc1 = 0;
   uint32_t rsrc2 = 0;
   memset(request, 0, sizeof(*request));
   find_register(shader->sh_registers, shader->num_sh_registers, layout->rsrc1, &rsrc1);
   find_register(shader->sh_registers, shader->num_sh_registers, layout->rsrc2, &rsrc2);
   request->compiler_version = AGC_SHADER_COMPILER_VERSION;
   request->options.target = host_isa();
   request->options.stage = layout->stage;
   request->options.wave_size = layout->wave_size;
   request->options.ray_unit = AGC_SHADER_RAY_UNIT_HOST;
   request->precision = AGC_SHADER_PRECISION_EXACT;
   request->words = words;
   request->word_count = word_count;
   iface->has_guest_pgm_rsrc1 = true;
   iface->guest_pgm_rsrc1 = rsrc1;
   iface->has_guest_pgm_rsrc2 = true;
   iface->guest_pgm_rsrc2 = rsrc2;
   iface->scratch_bytes_per_lane = rsrc2 & RSRC2_SCRATCH_EN ? (uint32_t)shader->scratch_size_dw_per_thread * 4 : 0;
   iface->has_guest_code_address = true;
   iface->guest_code_address = guest_code;
   iface->has_descriptor_lut = true;
   iface->descriptor_lut = AGC_RUNTIME_DESCRIPTOR_LUT_VA;
   iface->has_gds_buffer = true;
   iface->gds_buffer = AGC_RUNTIME_GDS_VA;
   iface->has_image_reports = true;
   iface->image_reports = AGC_RUNTIME_IMAGE_REPORTS_VA;
   iface->vertex_stage = layout->vertex_stage;
   if (layout->stage == AGC_SHADER_STAGE_VERTEX && !layout->vertex_stage && shader->specials &&
       ((shader->specials->vgt_shader_stages_en.value >> 3) & 3) == 1)
      iface->es_input = AGC_SHADER_ES_DOMAIN;
   iface->native_dcc = (request->options.target == AGC_SHADER_ISA_GFX11 ||
                        request->options.target == AGC_SHADER_ISA_GFX115) &&
                       agc_surface_native_dcc();
   if (layout->stage == AGC_SHADER_STAGE_FRAGMENT) {
      const struct agc_register *cx = shader->cx_registers;
      uint32_t count = shader->num_cx_registers;
      iface->has_ps_input_ena = true;
      find_register(cx, count, REG_SPI_PS_INPUT_ENA, &iface->ps_input_ena);
      iface->has_ps_input_addr = true;
      find_register(cx, count, REG_SPI_PS_INPUT_ADDR, &iface->ps_input_addr);
      iface->has_ps_col_format = true;
      find_register(cx, count, REG_SPI_SHADER_COL_FORMAT, &iface->ps_col_format);
      iface->has_ps_z_format = true;
      find_register(cx, count, REG_SPI_SHADER_Z_FORMAT, &iface->ps_z_format);
      iface->has_mip_stats_table = true;
      iface->mip_stats_table = AGC_RUNTIME_MIP_STATS_VA;
   }
   if (layout->stage == AGC_SHADER_STAGE_VERTEX && layout->vertex_stage == AGC_SHADER_VERTEX_STAGE_ES_GS && info) {
      iface->has_attribute_ring = true;
      iface->attribute_ring.address = AGC_RUNTIME_ATTRIBUTE_RING_VA;
      iface->attribute_ring.bytes = info->attribute_ring_size_per_se * (info->max_se ? info->max_se : 1);
   }
   if (link && link->second_words) {
      region->address = link->second_half;
      region->words = link->second_words;
      region->word_count = link->second_word_count;
      request->linkage.regions = region;
      request->linkage.region_count = 1;
      request->linkage.has_second_half = true;
      request->linkage.second_half = link->second_half;
   }
}

static bool
run_oracle(const char *request_path, const char *artifact_path, const char *function)
{
#ifdef _WIN32
   char command[3200];
   snprintf(command, sizeof(command), "\"%s\" translate \"%s\" \"%s\"", programs.oracle, request_path,
            artifact_path);
   STARTUPINFOA startup = {.cb = sizeof(startup)};
   PROCESS_INFORMATION process = {0};
   if (!CreateProcessA(NULL, command, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &startup, &process)) {
      agc_report(function, "the shader oracle %s could not be started (error %lu)", programs.oracle,
                 (unsigned long)GetLastError());
      return false;
   }
   DWORD waited = WaitForSingleObject(process.hProcess, ORACLE_TIMEOUT_MS);
   DWORD code = 1;
   if (waited == WAIT_OBJECT_0)
      GetExitCodeProcess(process.hProcess, &code);
   else
      TerminateProcess(process.hProcess, 1);
   CloseHandle(process.hThread);
   CloseHandle(process.hProcess);
   if (waited != WAIT_OBJECT_0 || code != 0) {
      agc_report(function, "the shader oracle failed on %s (%s, exit code %lu)", request_path,
                 waited == WAIT_OBJECT_0 ? "finished" : "timed out", (unsigned long)code);
      return false;
   }
   return true;
#else
   (void)request_path;
   (void)artifact_path;
   agc_report(function, "the shader oracle is only available on Windows");
   return false;
#endif
}

static bool
translate(const struct agc_shader_request *request, const struct agc_shader_key *key,
          struct agc_shader_artifact *artifact, const char *function)
{
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   agc_shader_key_hex(key, hex);
   enum agc_shader_file_status status = agc_shader_cache_lookup(programs.cache, key, artifact);
   if (status == AGC_SHADER_FILE_OK) {
      programs.counters.cache_hits++;
      return true;
   }
   if (status != AGC_SHADER_FILE_NOT_FOUND)
      agc_report(function, "cached translation %s is unusable (%s); translating again", hex,
                 agc_shader_file_status_name(status));
   if (programs.translator != TRANSLATOR_ORACLE) {
      struct agc_isa_error isa_error;
      if (agc_isa_translate(request, artifact, &isa_error)) {
         programs.counters.aco_translations++;
         status = agc_shader_cache_store(programs.cache, key, artifact);
         if (status != AGC_SHADER_FILE_OK)
            agc_report(function, "translation %s could not be cached: %s", hex, agc_shader_file_status_name(status));
         return true;
      }
      programs.counters.aco_fallbacks++;
      agc_stat(function, "shader %s is not translated in the driver (error %u at pc 0x%x%s%s)%s", hex,
               (unsigned)isa_error.kind, isa_error.pc, isa_error.reason ? ": " : "",
               isa_error.reason ? isa_error.reason : "",
               programs.translator == TRANSLATOR_AUTO ? "; the oracle translates it" : "");
      if (programs.translator == TRANSLATOR_ACO)
         return false;
   }
   status = agc_shader_cache_miss(programs.cache, key, request);
   if (status != AGC_SHADER_FILE_OK) {
      agc_report(function, "the translation request %s could not be written: %s", hex,
                 agc_shader_file_status_name(status));
      return false;
   }
   if (!programs.oracle[0]) {
      agc_report(function, "shader %s is not translated yet and no translator is configured: run the oracle on %s\\%s",
                 hex, programs.directory, AGC_SHADER_REQUEST_DIRECTORY);
      return false;
   }
   char request_path[1400];
   char artifact_path[1400];
   snprintf(request_path, sizeof(request_path), "%s\\%s\\%s%s", programs.directory, AGC_SHADER_REQUEST_DIRECTORY,
            hex, AGC_SHADER_REQUEST_EXTENSION);
   snprintf(artifact_path, sizeof(artifact_path), "%s\\%s.oracle%s", programs.directory, hex,
            AGC_SHADER_ARTIFACT_EXTENSION);
   programs.counters.oracle_runs++;
   if (!run_oracle(request_path, artifact_path, function))
      return false;
   status = agc_shader_artifact_read_file(artifact_path, artifact);
   if (status != AGC_SHADER_FILE_OK) {
      agc_report(function, "the oracle's translation %s is unreadable: %s", hex, agc_shader_file_status_name(status));
      return false;
   }
   status = agc_shader_cache_store(programs.cache, key, artifact);
   if (status != AGC_SHADER_FILE_OK)
      agc_report(function, "translation %s could not be cached: %s", hex, agc_shader_file_status_name(status));
   remove(artifact_path);
   const char *keep = getenv("AGC_SHADER_KEEP_REQUESTS");
   if (!keep || !*keep || *keep == '0')
      remove(request_path);
   return true;
}

static int32_t
place(struct agc_shader *shader, const struct stage_layout *layout, const struct agc_shader_key *key,
      const struct agc_shader_artifact *artifact, struct placed *entry, const char *function)
{
   if (artifact->error.kind != AGC_SHADER_ERROR_NONE) {
      char hex[AGC_SHADER_KEY_HEX_SIZE];
      agc_shader_key_hex(key, hex);
      agc_fatal(function, "shader %s at 0x%llx cannot be translated (%s at pc 0x%x%s%s); the request is in %s\\%s",
                hex, (unsigned long long)shader->code, agc_shader_error_kind_name(artifact->error.kind),
                artifact->error.pc, artifact->error.reason ? ": " : "",
                artifact->error.reason ? artifact->error.reason : "", programs.directory,
                AGC_SHADER_REQUEST_DIRECTORY);
   }
   struct agc_mem mem;
   uint64_t bytes = (uint64_t)artifact->code_count * 4;
   if (!agc_mem_suballoc(bytes, CODE_ALIGNMENT, AGC_BO_EXECUTE, &mem)) {
      agc_report(function, "out of driver memory for %llu bytes of shader code", (unsigned long long)bytes);
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   memcpy(mem.cpu, artifact->code, bytes);
   _mm_sfence();
   entry->va = mem.va;
   entry->pgm_rsrc1 = artifact->pgm_rsrc1;
   entry->pgm_rsrc2 = artifact->pgm_rsrc2;
   entry->has_col_format = layout->stage == AGC_SHADER_STAGE_FRAGMENT && artifact->has_spi_shader_col_format;
   entry->col_format = artifact->spi_shader_col_format;
   entry->has_vs_out_config = artifact->has_ngg;
   entry->vs_out_config = artifact->ngg.spi_vs_out_config;
   programs.counters.translated++;
   programs.counters.placed_bytes += bytes;
   if (artifact->external_target_count) {
      programs.counters.external_targets += artifact->external_target_count;
      agc_report(function, "shader at 0x%llx jumps to %u targets outside itself; those waves end there",
                 (unsigned long long)shader->code, artifact->external_target_count);
   }
   return 0;
}

static void
rewrite(struct agc_shader *shader, const struct stage_layout *layout, const struct placed *entry,
        const char *function)
{
   if (!set_register(shader->sh_registers, shader->num_sh_registers, layout->rsrc1, entry->pgm_rsrc1) ||
       !set_register(shader->sh_registers, shader->num_sh_registers, layout->rsrc2, entry->pgm_rsrc2))
      agc_report(function, "the shader at 0x%llx has no resource registers to update", (unsigned long long)shader->code);
   if (entry->has_col_format &&
       !set_register(shader->cx_registers, shader->num_cx_registers, REG_SPI_SHADER_COL_FORMAT, entry->col_format))
      agc_report(function, "the pixel shader at 0x%llx has no SPI_SHADER_COL_FORMAT to update",
                 (unsigned long long)shader->code);
   if (entry->has_vs_out_config &&
       !set_register(shader->cx_registers, shader->num_cx_registers, REG_SPI_VS_OUT_CONFIG, entry->vs_out_config))
      agc_report(function, "the geometry shader at 0x%llx has no SPI_VS_OUT_CONFIG to update",
                 (unsigned long long)shader->code);
}

int32_t
agc_program_translate_linked(struct agc_shader *shader, uint64_t guest_code, const uint32_t *words,
                             uint32_t word_count, const struct agc_program_link *link, uint64_t *host_code)
{
   static const char function[] = "sceAgcCreateShader";
   struct stage_layout layout;
   if (!stage_layout(shader, &layout)) {
      *host_code = guest_code;
      return 0;
   }
   char error[256];
   if (!agc_driver_start(error, sizeof(error)))
      agc_fatal(function, "the GPU is unavailable: %s", error);

   call_once(&programs_once, programs_init_lock);
   mtx_lock(&programs.lock);
   if (!programs_ready(function)) {
      mtx_unlock(&programs.lock);
      agc_fatal(function, "the shader cache is unavailable");
   }
   struct agc_shader_request request;
   struct agc_shader_code_region region;
   build_request(shader, &layout, guest_code, words, word_count, link, &region, &request);
   struct agc_shader_key key;
   if (agc_shader_request_key(&request, &key) != AGC_SHADER_FILE_OK) {
      mtx_unlock(&programs.lock);
      agc_fatal(function, "the translation request of the shader at 0x%llx cannot be encoded",
                (unsigned long long)guest_code);
   }
   struct placed *entry = placed_find(&key, false);
   if (!entry || !entry->va) {
      struct agc_shader_artifact artifact;
      memset(&artifact, 0, sizeof(artifact));
      if (!translate(&request, &key, &artifact, function)) {
         mtx_unlock(&programs.lock);
         agc_fatal(function, "the shader at 0x%llx has no translation (cache %s)", (unsigned long long)guest_code,
                   programs.directory);
      }
      entry = placed_find(&key, true);
      if (!entry) {
         mtx_unlock(&programs.lock);
         agc_fatal(function, "out of memory for more than %u translated programs", programs.placed_used);
      }
      int32_t result = place(shader, &layout, &key, &artifact, entry, function);
      agc_shader_artifact_free(&artifact);
      if (result) {
         entry->used = false;
         programs.placed_used--;
         mtx_unlock(&programs.lock);
         return result;
      }
   }
   struct placed copy = *entry;
   mtx_unlock(&programs.lock);
   rewrite(shader, &layout, &copy, function);
   *host_code = copy.va;
   return 0;
}

int32_t
agc_program_translate(struct agc_shader *shader, uint64_t guest_code, uint64_t *host_code)
{
   return agc_program_translate_linked(shader, guest_code, (const uint32_t *)(uintptr_t)guest_code,
                                       shader->shader_size / 4, NULL, host_code);
}

void
agc_program_install(void)
{
   agc_shader_set_program_hook(agc_program_translate);
}

void
agc_program_get_counters(struct agc_program_counters *counters)
{
   call_once(&programs_once, programs_init_lock);
   mtx_lock(&programs.lock);
   *counters = programs.counters;
   mtx_unlock(&programs.lock);
}
