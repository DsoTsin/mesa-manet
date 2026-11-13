#include <inttypes.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"
#include "agc_lower.h"
#include "agc_isa_translate.h"
#include "agc_shader_artifact.h"

#ifdef _WIN32
#include <crtdbg.h>
#include <windows.h>
#endif

static bool
write_bytes(const char *path, const uint8_t *data, size_t size)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      return false;
   bool ok = fwrite(data, 1, size, f) == size;
   return fclose(f) == 0 && ok;
}

static void
describe_error(const struct agc_isa_error *error, char *text, size_t bytes)
{
   snprintf(text, bytes, "error kind %u at pc 0x%x (family %u opcode 0x%x)%s%s", (unsigned)error->kind, error->pc,
            error->family, error->opcode, error->reason ? ": " : "", error->reason ? error->reason : "");
}

static uint64_t
fnv64(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   uint64_t h = UINT64_C(0xcbf29ce484222325);
   for (size_t i = 0; i < size; i++)
      h = (h ^ bytes[i]) * UINT64_C(0x00000100000001b3);
   return h;
}

static uint64_t
artifact_hash(const struct agc_shader_artifact *artifact)
{
   uint8_t *bytes = NULL;
   size_t size = 0;
   if (agc_shader_artifact_serialize(artifact, &bytes, &size) != AGC_SHADER_FILE_OK)
      return 0;
   uint64_t h = fnv64(bytes, size);
   free(bytes);
   return h;
}

static unsigned
stage_index(const struct agc_shader_request *request)
{
   return request->options.stage == AGC_SHADER_STAGE_COMPUTE ? 0 :
          request->options.stage == AGC_SHADER_STAGE_FRAGMENT ? 1 :
          request->iface.vertex_stage == AGC_SHADER_VERTEX_STAGE_LS_HS ? 3 : 2;
}

static int
command_translate(const char *request_path, const char *out_path)
{
   struct agc_shader_request request;
   memset(&request, 0, sizeof(request));
   if (agc_shader_request_read_file(request_path, &request) != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: unreadable request\n", request_path);
      return 2;
   }
   struct agc_shader_artifact artifact = {0};
   struct agc_isa_error error = {0};
   int status = 0;
   if (!agc_isa_translate(&request, &artifact, &error)) {
      char text[512];
      describe_error(&error, text, sizeof(text));
      fprintf(stderr, "%s\n", text);
      status = 1;
   } else {
      printf("%u words, program %u, code %016" PRIx64 ", artifact %016" PRIx64 "\n", artifact.code_count,
             artifact.program_words, fnv64(artifact.code, (size_t)artifact.code_count * 4), artifact_hash(&artifact));
      if (!write_bytes(out_path, (const uint8_t *)artifact.code, (size_t)artifact.code_count * 4))
         status = 2;
   }
   agc_shader_artifact_free(&artifact);
   agc_shader_request_free(&request);
   return status;
}

static int
command_listing(const char *request_path)
{
   struct agc_shader_request request;
   memset(&request, 0, sizeof(request));
   if (agc_shader_request_read_file(request_path, &request) != AGC_SHADER_FILE_OK) {
      fprintf(stderr, "%s: unreadable request\n", request_path);
      return 2;
   }
   char *listing = NULL;
   struct agc_isa_error error = {0};
   int status = 0;
   if (agc_isa_translate_listing(&request, &listing, &error)) {
      fputs(listing, stdout);
   } else {
      char text[512];
      describe_error(&error, text, sizeof(text));
      fprintf(stderr, "%s\n", text);
      status = 1;
   }
   free(listing);
   agc_shader_request_free(&request);
   return status;
}

#ifdef _WIN32
static int
command_translate_corpus(const char *requests, const char *report_path, const char *artifacts)
{
   char pattern[1024];
   snprintf(pattern, sizeof(pattern), "%s\\*.req", requests);
   WIN32_FIND_DATAA found;
   HANDLE search = FindFirstFileA(pattern, &found);
   if (search == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "no requests in %s\n", requests);
      return 2;
   }
   FILE *report = fopen(report_path, "w");
   if (!report) {
      FindClose(search);
      return 2;
   }
   unsigned total = 0;
   unsigned translated = 0;
   unsigned stages[4][2] = {{0}};
   do {
      char path[1400];
      snprintf(path, sizeof(path), "%s\\%s", requests, found.cFileName);
      struct agc_shader_request request = {0};
      struct agc_shader_artifact artifact = {0};
      struct agc_isa_error error = {0};
      total++;
      if (agc_shader_request_read_file(path, &request) != AGC_SHADER_FILE_OK) {
         fprintf(report, "%s: unreadable request\n", found.cFileName);
         continue;
      }
      unsigned stage = stage_index(&request);
      bool ok = agc_isa_translate(&request, &artifact, &error);
      stages[stage][0]++;
      stages[stage][1] += ok;
      translated += ok;
      if (ok) {
         fprintf(report, "%s: stage %u, %u words, code %016" PRIx64 ", artifact %016" PRIx64 "\n", found.cFileName,
                 stage, artifact.code_count, fnv64(artifact.code, (size_t)artifact.code_count * 4),
                 artifact_hash(&artifact));
         if (artifacts) {
            char out[1400];
            snprintf(out, sizeof(out), "%s\\%.*s.art", artifacts, (int)(strlen(found.cFileName) - 4),
                     found.cFileName);
            if (agc_shader_artifact_write_file(&artifact, out) != AGC_SHADER_FILE_OK)
               fprintf(report, "%s: the artifact could not be written\n", found.cFileName);
         }
      } else {
         char text[AGC_ISA_ERROR_TEXT_SIZE];
         describe_error(&error, text, sizeof(text));
         fprintf(report, "%s: stage %u, refused: %s\n", found.cFileName, stage, text);
      }
      agc_shader_artifact_free(&artifact);
      agc_shader_request_free(&request);
   } while (FindNextFileA(search, &found));
   FindClose(search);
   fclose(report);
   static const char *const names[] = {"compute", "pixel", "ES/GS", "LS/HS"};
   printf("%u requests, %u translated\n", total, translated);
   for (unsigned s = 0; s < 4; s++)
      printf("   %s: %u/%u\n", names[s], stages[s][1], stages[s][0]);
   uint64_t reports[2];
   agc_isa_lowering_report_counts(reports);
   printf("   image descriptor conversions: %" PRIu64 ", %" PRIu64 " of them report the image\n", reports[0],
          reports[1]);
   return translated == total ? 0 : 1;
}

static uint64_t fuzz_state;
static const uint32_t *fuzz_words;
static size_t fuzz_word_count;
static unsigned fuzz_index;
static unsigned fuzz_stage;
static unsigned fuzz_wave;
static int fuzz_only = -1;

static jmp_buf fuzz_jump;
static bool fuzz_verbose;
static bool fuzz_wide;
static bool fuzz_wide_registers;
static bool fuzz_dual_source;

#define FUZZ_TEMPLATES 8
#define FUZZ_REGION_WORDS (8 * AGC_ISA_MAX_INSTRUCTION_WORDS + 1)

static void
fuzz_abort(int sig)
{
   (void)sig;
   if (fuzz_verbose) {
      fprintf(stderr, "abort at index %u (stage %u wave %u):", fuzz_index, fuzz_stage, fuzz_wave);
      for (size_t w = 0; w < fuzz_word_count; w++)
         fprintf(stderr, " %08x", fuzz_words[w]);
      fprintf(stderr, "\n");
      fflush(stderr);
   }
   signal(SIGABRT, fuzz_abort);
   longjmp(fuzz_jump, 1);
}

static int
fuzz_run(const struct agc_shader_request *request, struct agc_shader_artifact *artifact, struct agc_isa_error *error)
{
   if (setjmp(fuzz_jump))
      return 2;
   return agc_isa_translate(request, artifact, error) ? 1 : 0;
}

static uint32_t
fuzz_next(void)
{
   fuzz_state ^= fuzz_state >> 12;
   fuzz_state ^= fuzz_state << 25;
   fuzz_state ^= fuzz_state >> 27;
   return (uint32_t)((fuzz_state * UINT64_C(0x2545f4914f6cdd1d)) >> 32);
}

static bool
fuzz_registers_fit(const struct agc_isa_inst *inst)
{
   if (fuzz_wide_registers)
      return true;
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   for (unsigned pass = 0; pass < 2; pass++) {
      unsigned count = pass ? agc_isa_inst_defs(inst, true, ranges) : agc_isa_inst_uses(inst, true, ranges);
      for (unsigned i = 0; i < count; i++) {
         unsigned last = agc_isa_reg_range_last(ranges[i]);
         if (ranges[i].file == AGC_ISA_REG_SCALAR && last >= 40 && ranges[i].first < AGC_ISA_SGPR_LIMIT)
            return false;
         if (ranges[i].file == AGC_ISA_REG_VECTOR && last >= 48)
            return false;
      }
   }
   return true;
}

static bool
fuzz_flow_instruction(uint32_t *words, bool *branch)
{
   static const uint8_t branches[7] = {
      AGC_ISA_SOPP_BRANCH,        AGC_ISA_SOPP_CBRANCH_SCC0,  AGC_ISA_SOPP_CBRANCH_SCC1,  AGC_ISA_SOPP_CBRANCH_VCCZ,
      AGC_ISA_SOPP_CBRANCH_VCCNZ, AGC_ISA_SOPP_CBRANCH_EXECZ, AGC_ISA_SOPP_CBRANCH_EXECNZ,
   };
   if (fuzz_next() % 12 != 0)
      return false;
   static const uint8_t specials[4] = {106, 108, 124, 126};
   unsigned pick = fuzz_next() % 12;
   uint32_t sgpr = 2 * (fuzz_next() % 20);
   if (fuzz_next() % 8 == 0)
      sgpr = specials[fuzz_next() % 4];
   if (pick < 7) {
      words[0] = 0xbf800000u | ((uint32_t)branches[pick] << 16);
      *branch = true;
   } else if (pick == 7) {
      words[0] = 0xb0000000u | ((uint32_t)AGC_ISA_SOPK_CALL_B64 << 23) | (sgpr << 16);
      *branch = true;
   } else if (pick == 8) {
      words[0] = 0xbe800000u | (sgpr << 16) | ((uint32_t)AGC_ISA_SOP1_GETPC_B64 << 8);
   } else if (pick < 11) {
      uint32_t op = pick == 9 ? AGC_ISA_SOP1_SETPC_B64 : AGC_ISA_SOP1_SWAPPC_B64;
      words[0] = 0xbe800000u | (sgpr << 16) | (op << 8) | (2 * (fuzz_next() % 20));
   } else {
      words[0] = 0xbf810000u;
   }
   return true;
}

static void
fuzz_bias(uint32_t prefix, uint32_t *words)
{
   static const uint16_t vop3_ops[] = {0x102, 0x10d, 0x106, 0x11f, 0x12b, 0x140, 0x141, 0x14b, 0x176, 0x177};
   if (prefix == 0xcc000000u && fuzz_next() % 3 == 0) {
      words[0] = (words[0] & ~(0x7fu << 16)) | ((0x13u + fuzz_next() % 7) << 16);
      if (fuzz_next() % 2) {
         words[0] = (words[0] & ~((7u << 8) | (1u << 13))) | (1u << 14);
         words[1] &= ~(7u << 29);
      }
   } else if (prefix == 0xd4000000u && fuzz_next() % 6 == 0) {
      words[0] = (words[0] & ~(0x3ffu << 16)) |
                 ((uint32_t)vop3_ops[fuzz_next() % (sizeof(vop3_ops) / sizeof(vop3_ops[0]))] << 16);
      if (fuzz_next() % 2) {
         words[0] &= ~((7u << 8) | (0xfu << 11));
         words[1] &= ~(3u << 27);
      }
   } else if (prefix == 0xf8000000u) {
      if (!fuzz_wide_registers) {
         words[1] = 0;
         for (unsigned c = 0; c < 4; c++)
            words[1] |= (fuzz_next() % 48) << (8 * c);
      }
      if (fuzz_dual_source && fuzz_next() % 2)
         words[0] = (words[0] & ~(0x3fu << 4)) | ((fuzz_next() % 2) << 4);
   } else if ((prefix == 0xe0000000u || prefix == 0xe8000000u) && fuzz_next() % 3 == 0) {
      words[1] = (words[1] & 0x00ffffffu) | ((fuzz_next() % 16) << 24);
   } else if (prefix == 0xb0000000u) {
      static const uint8_t portable[6] = {1, 2, 5, 6, 23, 24};
      uint32_t op = (words[0] >> 23) & 0x1f;
      bool hwreg = op == AGC_ISA_SOPK_GETREG_B32 || op == AGC_ISA_SOPK_SETREG_B32 ||
                   op == AGC_ISA_SOPK_SETREG_IMM32_B32;
      if (hwreg && fuzz_next() % 4 != 0)
         words[0] = (words[0] & ~0x3fu) | portable[fuzz_next() % 6];
      bool counter = op >= AGC_ISA_SOPK_WAITCNT_VSCNT && op <= AGC_ISA_SOPK_WAITCNT_LGKMCNT;
      if (counter && fuzz_next() % 8 != 0) {
         uint32_t limit = op == AGC_ISA_SOPK_WAITCNT_EXPCNT ? 8 : 64;
         words[0] = (words[0] & ~((0x7fu << 16) | 0xffffu)) | (125u << 16) | (fuzz_next() % limit);
      }
   } else if ((prefix == 0x80000000u || prefix == 0xbe800000u || prefix == 0xbf000000u) && fuzz_next() % 6 == 0) {
      words[0] |= 0xffu;
      words[1] = (uint32_t)((int32_t)(fuzz_next() % 96) - 20);
   }
}

static size_t
fuzz_export(uint32_t *words, unsigned target)
{
   words[0] = 0xf8000000u | (fuzz_next() & 0x1c0fu) | (target << 4);
   words[1] = 0;
   for (unsigned c = 0; c < 4; c++)
      words[1] |= (fuzz_next() % 48) << (8 * c);
   return 2;
}

static size_t
fuzz_instruction(uint32_t *words, bool *branch)
{
   *branch = false;
   if (fuzz_wide && fuzz_flow_instruction(words, branch))
      return 1;
   static const struct {
      uint8_t weight;
      uint32_t prefix;
      uint32_t mask;
      bool vector;
   } kinds[] = {
      {12, 0x00000000u, 0x7fffffffu, true}, {6, 0x7e000000u, 0x01ffffffu, true},  {6, 0x7c000000u, 0x01ffffffu, true},
      {8, 0x80000000u, 0x3fffffffu, false}, {4, 0xb0000000u, 0x0fffffffu, false}, {6, 0xbe800000u, 0x007fffffu, false},
      {4, 0xbf000000u, 0x007fffffu, false}, {3, 0xbf800000u, 0x007fffffu, false}, {3, 0xc8000000u, 0x03ffffffu, false},
      {6, 0xcc000000u, 0x03ffffffu, false}, {14, 0xd4000000u, 0x03ffffffu, false}, {8, 0xd8000000u, 0x03ffffffu, false},
      {8, 0xdc000000u, 0x03ffffffu, false}, {8, 0xe0000000u, 0x03ffffffu, false},  {4, 0xe8000000u, 0x03ffffffu, false},
      {14, 0xf0000000u, 0x03ffffffu, false}, {6, 0xf4000000u, 0x03ffffffu, false},  {4, 0xf8000000u, 0x03ffffffu, false},
   };
   unsigned total = 0;
   for (unsigned k = 0; k < sizeof(kinds) / sizeof(kinds[0]); k++)
      total += kinds[k].weight;
   for (;;) {
      unsigned pick = fuzz_next() % total;
      unsigned k = 0;
      while (pick >= kinds[k].weight)
         pick -= kinds[k++].weight;
      words[0] = kinds[k].prefix | (fuzz_next() & kinds[k].mask);
      for (unsigned w = 1; w < AGC_ISA_MAX_INSTRUCTION_WORDS; w++)
         words[w] = fuzz_next();
      if (kinds[k].vector && fuzz_next() % 4 == 0) {
         static const uint32_t codes[] = {249, 250, 233, 234, 255};
         words[0] = (words[0] & ~0x1ffu) | codes[fuzz_next() % 5];
      }
      if (fuzz_wide)
         fuzz_bias(kinds[k].prefix, words);
      struct agc_isa_inst inst;
      struct agc_isa_error error;
      if (!agc_isa_decode_at(words, AGC_ISA_MAX_INSTRUCTION_WORDS, 0, &inst, &error))
         continue;
      if (inst.flags & (AGC_ISA_FLAG_BRANCH | AGC_ISA_FLAG_COND_BRANCH | AGC_ISA_FLAG_END | AGC_ISA_FLAG_JUMP |
                        AGC_ISA_FLAG_CALL | AGC_ISA_FLAG_PC_RELATIVE))
         continue;
      if (!inst.word_count || inst.word_count > AGC_ISA_MAX_INSTRUCTION_WORDS || !fuzz_registers_fit(&inst))
         continue;
      return inst.word_count;
   }
}

static void
fuzz_interface(struct agc_shader_request *request, unsigned stage)
{
   struct agc_shader_interface *f = &request->iface;
   if (fuzz_next() % 2) {
      f->has_descriptor_lut = fuzz_next() % 4 != 0;
      f->descriptor_lut = UINT64_C(0x180000000) + (fuzz_next() & 0xfffff000u);
      f->preconvert_user_descriptors = fuzz_next() % 2;
      f->native_dcc = fuzz_next() % 2;
   }
   if (fuzz_next() % 2) {
      f->has_gds_buffer = fuzz_next() % 2;
      f->gds_buffer = UINT64_C(0x200000000) + (fuzz_next() & 0xfffffff0u);
   }
   if (fuzz_next() % 2) {
      f->has_mip_stats_table = fuzz_next() % 2;
      f->mip_stats_table = UINT64_C(0x280000000) + (fuzz_next() & 0xfffffff0u);
   }
   if (fuzz_next() % 2) {
      f->has_image_reports = fuzz_next() % 2;
      f->image_reports = UINT64_C(0x300000000) + (fuzz_next() & 0xfffffff0u);
   }
   if (fuzz_next() % 2) {
      f->has_guest_code_address = fuzz_next() % 2;
      f->guest_code_address = UINT64_C(0x900000000) + ((uint64_t)(fuzz_next() & 0xffffffu) << 2);
   }
   fuzz_dual_source = false;
   if (stage == 1) {
      if (fuzz_next() % 2) {
         f->dual_source_blend = fuzz_next() % 2;
         f->has_ps_col_format = fuzz_next() % 2;
         f->ps_col_format = fuzz_next();
         f->has_ps_z_format = fuzz_next() % 2;
         f->ps_z_format = fuzz_next() % 16;
      }
      fuzz_dual_source = f->dual_source_blend;
   } else if (stage >= 2 && fuzz_next() % 2) {
      f->has_attribute_ring = fuzz_next() % 2;
      f->attribute_ring.address = UINT64_C(0x380000000) + (fuzz_next() & 0xfffff000u);
      f->attribute_ring.bytes = 0x1000u << (fuzz_next() % 8);
      f->es_input = fuzz_next() % 2 ? AGC_SHADER_ES_DOMAIN : AGC_SHADER_ES_VERTEX;
   }
   if (fuzz_next() % 4 == 0) {
      f->has_user_sgpr_count = fuzz_next() % 2;
      f->user_sgpr_count = (uint8_t)(fuzz_next() % 17);
   }
   if (fuzz_next() % 4 == 0)
      f->scratch_bytes_per_lane = fuzz_next() % 2 ? 0 : 64u << (fuzz_next() % 4);
   if (fuzz_next() % 3 == 0) {
      f->has_guest_pgm_rsrc2 = fuzz_next() % 4 != 0;
      f->guest_pgm_rsrc2 = fuzz_next();
      if (fuzz_next() % 2)
         f->guest_pgm_rsrc2 = (f->guest_pgm_rsrc2 & ~0x3eu) | ((fuzz_next() % 17) << 1);
   }
   if (fuzz_next() % 4 == 0) {
      f->has_guest_pgm_rsrc1 = fuzz_next() % 2;
      f->guest_pgm_rsrc1 = fuzz_next();
   }
}

static void
fuzz_linkage(struct agc_shader_request *request, struct agc_shader_code_region regions[2],
             uint32_t storage[2][FUZZ_REGION_WORDS])
{
   memset(&request->linkage, 0, sizeof(request->linkage));
   if (fuzz_next() % 8 != 0)
      return;
   unsigned count = 1 + fuzz_next() % 2;
   for (unsigned r = 0; r < count; r++) {
      unsigned instructions = 1 + fuzz_next() % 6;
      size_t size = 0;
      for (unsigned i = 0; i < instructions; i++) {
         bool branch;
         size_t words = fuzz_instruction(storage[r] + size, &branch);
         if (branch)
            storage[r][size] = 0xbf800000u;
         size += words;
      }
      storage[r][size++] = fuzz_next() % 2 ? 0xbf810000u : 0xbe802000u | (2 * (fuzz_next() % 20));
      regions[r].address = UINT64_C(0xa00000000) + r * UINT64_C(0x10000) + ((uint64_t)(fuzz_next() & 0xfffu) << 2);
      regions[r].words = storage[r];
      regions[r].word_count = (uint32_t)size;
   }
   request->linkage.regions = regions;
   request->linkage.region_count = count;
   request->linkage.has_second_half = fuzz_next() % 2;
   request->linkage.second_half = UINT64_C(0xb00000000) + ((uint64_t)(fuzz_next() & 0xffffffu) << 2);
   if (fuzz_next() % 8 != 0)
      request->linkage.second_half = regions[fuzz_next() % count].address;
   if (fuzz_next() % 8 != 0) {
      request->iface.has_guest_code_address = true;
      request->iface.guest_code_address = UINT64_C(0x900000000) + ((uint64_t)(fuzz_next() & 0xffffffu) << 2);
   }
}

static uint64_t
error_hash(const struct agc_isa_error *error)
{
   uint32_t fields[4] = {(uint32_t)error->kind, error->pc, error->family, error->opcode};
   uint64_t h = fnv64(fields, sizeof(fields));
   if (error->reason)
      h ^= fnv64(error->reason, strlen(error->reason));
   return h;
}

static int
command_fuzz(const char *requests, uint64_t seed, unsigned count, unsigned length)
{
   char pattern[1024];
   snprintf(pattern, sizeof(pattern), "%s\\*.req", requests);
   WIN32_FIND_DATAA found;
   HANDLE search = FindFirstFileA(pattern, &found);
   if (search == INVALID_HANDLE_VALUE) {
      fprintf(stderr, "no requests in %s\n", requests);
      return 2;
   }
   static struct agc_shader_request templates[4][FUZZ_TEMPLATES];
   unsigned have[4] = {0, 0, 0, 0};
   unsigned wanted = fuzz_wide ? FUZZ_TEMPLATES : 1;
   unsigned missing = 4 * wanted;
   do {
      char path[1400];
      snprintf(path, sizeof(path), "%s\\%s", requests, found.cFileName);
      struct agc_shader_request request = {0};
      if (agc_shader_request_read_file(path, &request) != AGC_SHADER_FILE_OK)
         continue;
      unsigned stage = stage_index(&request);
      if (have[stage] == wanted) {
         agc_shader_request_free(&request);
         continue;
      }
      templates[stage][have[stage]++] = request;
      missing--;
   } while (missing && FindNextFileA(search, &found));
   FindClose(search);
   static struct agc_shader_code_region regions[2];
   static uint32_t region_words[2][FUZZ_REGION_WORDS];
   unsigned ran = 0, translated = 0, refused = 0, aborted = 0;
   uint64_t hash = UINT64_C(0xcbf29ce484222325);
   uint32_t words[64 * AGC_ISA_MAX_INSTRUCTION_WORDS + 1];
   static struct {
      char text[160];
      unsigned count;
   } reasons[256];
   unsigned reason_count = 0;
   signal(SIGABRT, fuzz_abort);
   _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
   _set_error_mode(_OUT_TO_STDERR);
   if (!fuzz_verbose)
      freopen("NUL", "w", stderr);
   for (unsigned n = 0; n < count; n++) {
      if (fuzz_only >= 0 && n != (unsigned)fuzz_only)
         continue;
      fuzz_state = (seed + n) * UINT64_C(0x9e3779b97f4a7c15) + 1;
      unsigned stage = fuzz_next() % 4;
      if (!have[stage])
         continue;
      struct agc_shader_request request = templates[stage][fuzz_wide ? fuzz_next() % have[stage] : 0];
      memset(&request.linkage, 0, sizeof(request.linkage));
      fuzz_wide_registers = false;
      if (fuzz_wide) {
         fuzz_wide_registers = fuzz_next() % 16 == 0;
         fuzz_interface(&request, stage);
         fuzz_linkage(&request, regions, region_words);
      }
      unsigned instructions = 1 + fuzz_next() % (length ? length : 1);
      size_t size = 0;
      uint16_t starts[65];
      bool branches[64];
      unsigned emitted = 0;
      for (unsigned i = 0; i < instructions && i < 64; i++) {
         starts[emitted] = (uint16_t)size;
         size += fuzz_instruction(words + size, &branches[emitted]);
         emitted++;
      }
      if (fuzz_wide && fuzz_dual_source && fuzz_next() % 2) {
         unsigned first = fuzz_next() % 2;
         for (unsigned k = 0; k < 2 && emitted < 64; k++) {
            starts[emitted] = (uint16_t)size;
            branches[emitted++] = false;
            size += fuzz_export(words + size, k ? !first : first);
         }
      }
      starts[emitted] = (uint16_t)size;
      for (unsigned i = 0; i < emitted; i++) {
         if (!branches[i])
            continue;
         int delta = (int)starts[fuzz_next() % (emitted + 1)] - (int)(starts[i] + 1);
         words[starts[i]] = (words[starts[i]] & 0xffff0000u) | (uint16_t)delta;
      }
      words[size++] = 0xbf810000u;
      request.words = words;
      request.word_count = (uint32_t)size;
      request.options.wave_size = fuzz_next() % 2 ? 64 : 32;
      request.precision = fuzz_next() % 2 ? AGC_SHADER_PRECISION_EXACT : AGC_SHADER_PRECISION_FUSED;
      struct agc_shader_artifact artifact = {0};
      struct agc_isa_error error = {0};
      fuzz_words = words;
      fuzz_word_count = size;
      fuzz_index = n;
      fuzz_stage = stage;
      fuzz_wave = request.options.wave_size;
      int result = fuzz_run(&request, &artifact, &error);
      ran++;
      uint64_t item;
      if (result == 2) {
         aborted++;
         item = UINT64_C(0xab027ed);
         if (getenv("AGC_FUZZ_ABORTS") && aborted <= 40)
            printf("abort at index %u\n", n);
      } else if (result == 1) {
         translated++;
         item = artifact_hash(&artifact);
         if (fuzz_verbose)
            printf("index %u (stage %u wave %u): %u words, artifact %016" PRIx64 "\n", n, stage,
                   (unsigned)request.options.wave_size, artifact.code_count, item);
         agc_shader_artifact_free(&artifact);
      } else {
         refused++;
         item = error_hash(&error);
         const char *reason = error.reason ? error.reason : "(no reason)";
         if (fuzz_verbose) {
            char text[AGC_ISA_ERROR_TEXT_SIZE + 512];
            describe_error(&error, text, sizeof(text));
            printf("index %u (stage %u wave %u): %s\n", n, stage, (unsigned)request.options.wave_size, text);
         }
         unsigned r = 0;
         while (r < reason_count && strcmp(reasons[r].text, reason))
            r++;
         if (r == reason_count && reason_count < 256) {
            snprintf(reasons[r].text, sizeof(reasons[r].text), "%s", reason);
            reasons[r].count = 0;
            reason_count++;
         }
         if (r < reason_count)
            reasons[r].count++;
      }
      hash = (hash ^ item) * UINT64_C(0x00000100000001b3);
   }
   for (unsigned k = 0; k < 4; k++) {
      for (unsigned t = 0; t < have[k]; t++)
         agc_shader_request_free(&templates[k][t]);
   }
   if (getenv("AGC_FUZZ_REASONS")) {
      for (unsigned r = 0; r < reason_count; r++)
         printf("%8u  %s\n", reasons[r].count, reasons[r].text);
   }
   printf("%u programs: %u translated, %u refused, %u aborted, hash %016" PRIx64 "\n", ran, translated, refused,
          aborted, hash);
   return 0;
}
#endif

int
main(int argc, char **argv)
{
   if (argc == 4 && !strcmp(argv[1], "translate"))
      return command_translate(argv[2], argv[3]);
   if (argc == 3 && !strcmp(argv[1], "listing"))
      return command_listing(argv[2]);
#ifdef _WIN32
   if ((argc == 4 || argc == 5) && !strcmp(argv[1], "translate-corpus"))
      return command_translate_corpus(argv[2], argv[3], argc == 5 ? argv[4] : NULL);
   if (argc >= 6 && !strcmp(argv[1], "fuzz-wide")) {
      fuzz_wide = true;
      argv[1] = "fuzz";
   }
   if (argc == 7 && !strcmp(argv[1], "fuzz")) {
      fuzz_only = atoi(argv[6]);
      fuzz_verbose = true;
      argc--;
   }
   if (argc == 6 && !strcmp(argv[1], "fuzz"))
      return command_fuzz(argv[2], strtoull(argv[3], NULL, 0), (unsigned)strtoul(argv[4], NULL, 0),
                          (unsigned)strtoul(argv[5], NULL, 0));
#endif
   fprintf(stderr, "usage: agc_isa_lower_tool translate <request> <out.bin>\n"
                   "       agc_isa_lower_tool listing <request>\n"
                   "       agc_isa_lower_tool translate-corpus <requests dir> <report> [artifacts dir]\n"
                   "       agc_isa_lower_tool fuzz <requests dir> <seed> <count> <max instructions> [index]\n"
                   "       agc_isa_lower_tool fuzz-wide <requests dir> <seed> <count> <max instructions> [index]\n");
   return 2;
}
