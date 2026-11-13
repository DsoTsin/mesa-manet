#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <dirent.h>
#endif

#include "agc_isa.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

struct text {
   char *data;
   size_t size;
   size_t capacity;
};

static void
die(const char *format, ...)
{
   va_list args;
   va_start(args, format);
   vfprintf(stderr, format, args);
   va_end(args);
   fputc('\n', stderr);
   exit(2);
}

static void *
checked(void *pointer)
{
   if (!pointer)
      die("out of memory");
   return pointer;
}

static void
text_reserve(struct text *text, size_t extra)
{
   if (text->size + extra <= text->capacity)
      return;
   size_t capacity = text->capacity ? text->capacity * 2 : 1 << 16;
   while (capacity < text->size + extra)
      capacity *= 2;
   text->data = checked(realloc(text->data, capacity));
   text->capacity = capacity;
}

static void
put(struct text *text, const char *format, ...) AGC_PRINTF(2, 3);

static void
put(struct text *text, const char *format, ...)
{
   text_reserve(text, 512);
   va_list args;
   va_start(args, format);
   int written = vsnprintf(text->data + text->size, text->capacity - text->size, format, args);
   va_end(args);
   if (written < 0 || (size_t)written >= text->capacity - text->size)
      die("a formatted item does not fit");
   text->size += (size_t)written;
}

static void
put_str(struct text *text, const char *string)
{
   size_t length = strlen(string);
   text_reserve(text, length + 1);
   memcpy(text->data + text->size, string, length);
   text->size += length;
}

static void
put_char(struct text *text, char c)
{
   text_reserve(text, 1);
   text->data[text->size++] = c;
}

static uint64_t
fnv(const char *data, size_t size)
{
   uint64_t hash = UINT64_C(0xcbf29ce484222325);
   for (size_t i = 0; i < size; i++)
      hash = (hash ^ (uint8_t)data[i]) * UINT64_C(0x100000001b3);
   return hash;
}

static void
flush(struct text *text)
{
   fwrite(text->data, 1, text->size, stdout);
   text->size = 0;
}

static void
put_operand(struct text *text, struct agc_isa_operand op)
{
   static const char *const names[AGC_ISA_OPERAND_KIND_COUNT] = {
      [AGC_ISA_OPERAND_NONE] = "none",
      [AGC_ISA_OPERAND_VCC_LO] = "vcc_lo",
      [AGC_ISA_OPERAND_VCC_HI] = "vcc_hi",
      [AGC_ISA_OPERAND_M0] = "m0",
      [AGC_ISA_OPERAND_NULL] = "null",
      [AGC_ISA_OPERAND_EXEC_LO] = "exec_lo",
      [AGC_ISA_OPERAND_EXEC_HI] = "exec_hi",
      [AGC_ISA_OPERAND_SHARED_BASE] = "shared_base",
      [AGC_ISA_OPERAND_SHARED_LIMIT] = "shared_limit",
      [AGC_ISA_OPERAND_PRIVATE_BASE] = "private_base",
      [AGC_ISA_OPERAND_PRIVATE_LIMIT] = "private_limit",
      [AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID] = "pops_exiting_wave_id",
      [AGC_ISA_OPERAND_VCC_Z] = "vccz",
      [AGC_ISA_OPERAND_EXEC_Z] = "execz",
      [AGC_ISA_OPERAND_SCC] = "scc",
      [AGC_ISA_OPERAND_LDS_DIRECT] = "lds_direct",
   };
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      put(text, "s%u", op.reg);
      break;
   case AGC_ISA_OPERAND_VGPR:
      put(text, "v%u", op.reg);
      break;
   case AGC_ISA_OPERAND_TTMP:
      put(text, "ttmp%u", op.reg);
      break;
   case AGC_ISA_OPERAND_INLINE_INT:
      put(text, "int(%d)", op.inline_int);
      break;
   case AGC_ISA_OPERAND_INLINE_FLOAT:
      put(text, "float(%u)", op.inline_float);
      break;
   case AGC_ISA_OPERAND_LITERAL:
      put(text, "lit(0x%x)", op.literal);
      break;
   case AGC_ISA_OPERAND_RAW:
      put(text, "raw(0x%x)", op.raw);
      break;
   default:
      put_str(text, op.kind < AGC_ISA_OPERAND_KIND_COUNT ? names[op.kind] : "?");
      break;
   }
}

static void
put_sources(struct text *text, const struct agc_isa_operand src[3])
{
   put_char(text, '[');
   for (unsigned i = 0; i < 3; i++) {
      if (i)
         put_char(text, ',');
      put_operand(text, src[i]);
   }
   put_char(text, ']');
}

static void
put_ext(struct text *text, const struct agc_isa_vop_ext *ext)
{
   const struct agc_isa_sdwa *s = &ext->sdwa;
   const struct agc_isa_dpp16 *d = &ext->dpp16;
   switch (ext->kind) {
   case AGC_ISA_VOP_EXT_SDWA:
      put(text,
          "sdwa(dst_sel=%u,dst_unused=%u,clamp=%u,omod=%u,src0_sel=%u,src0_sext=%u,src0_neg=%u,src0_abs=%u,"
          "src1_sel=%u,src1_sext=%u,src1_neg=%u,src1_abs=%u)",
          s->dst_sel, s->dst_unused, s->clamp, s->omod, s->src0_sel, s->src0_sext, s->src0_neg, s->src0_abs,
          s->src1_sel, s->src1_sext, s->src1_neg, s->src1_abs);
      break;
   case AGC_ISA_VOP_EXT_DPP16:
      put(text,
          "dpp16(ctrl=0x%x,fetch_inactive=%u,bound_ctrl=%u,src0_neg=%u,src0_abs=%u,src1_neg=%u,src1_abs=%u,"
          "bank_mask=0x%x,row_mask=0x%x)",
          d->ctrl, d->fetch_inactive, d->bound_ctrl, d->src0_neg, d->src0_abs, d->src1_neg, d->src1_abs,
          d->bank_mask, d->row_mask);
      break;
   case AGC_ISA_VOP_EXT_DPP8:
      put(text, "dpp8(selects=0x%x,fetch_inactive=%u)", ext->dpp8.selects, ext->dpp8.fetch_inactive);
      break;
   default:
      put_str(text, "none");
      break;
   }
}

static const char *
kind_name(unsigned kind)
{
   static const char *const names[4] = {"load", "store", "atomic", "control"};
   return kind < 4 ? names[kind] : "?";
}

static void
put_fields(struct text *text, const struct agc_isa_inst *inst)
{
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      put_str(text, "sop1 sdst=");
      put_operand(text, inst->sop1.sdst);
      put_str(text, " src0=");
      put_operand(text, inst->sop1.src0);
      break;
   case AGC_ISA_SOP2:
      put_str(text, "sop2 sdst=");
      put_operand(text, inst->sop2.sdst);
      put_str(text, " src0=");
      put_operand(text, inst->sop2.src0);
      put_str(text, " src1=");
      put_operand(text, inst->sop2.src1);
      break;
   case AGC_ISA_SOPK:
      put_str(text, "sopk sdst=");
      put_operand(text, inst->sopk.sdst);
      put(text, " simm16=0x%x imm32=", inst->sopk.simm16);
      if (inst->sopk.has_imm32)
         put(text, "0x%x", inst->sopk.imm32);
      else
         put_str(text, "none");
      break;
   case AGC_ISA_SOPC:
      put_str(text, "sopc src0=");
      put_operand(text, inst->sopc.src0);
      put_str(text, " src1=");
      put_operand(text, inst->sopc.src1);
      break;
   case AGC_ISA_SOPP:
      put(text, "sopp simm16=0x%x", inst->sopp.simm16);
      break;
   case AGC_ISA_VOP1:
      put_str(text, "vop1 vdst=");
      put_operand(text, inst->vop1.vdst);
      put_str(text, " src0=");
      put_operand(text, inst->vop1.src0);
      put_str(text, " ext=");
      put_ext(text, &inst->vop1.ext);
      break;
   case AGC_ISA_VOP2:
      put_str(text, "vop2 vdst=");
      put_operand(text, inst->vop2.vdst);
      put_str(text, " src0=");
      put_operand(text, inst->vop2.src0);
      put_str(text, " vsrc1=");
      put_operand(text, inst->vop2.vsrc1);
      put_str(text, " constant=");
      if (inst->vop2.has_constant)
         put(text, "0x%x", inst->vop2.constant);
      else
         put_str(text, "none");
      put_str(text, " ext=");
      put_ext(text, &inst->vop2.ext);
      break;
   case AGC_ISA_VOPC:
      put_str(text, "vopc sdst=");
      put_operand(text, inst->vopc.sdst);
      put_str(text, " src0=");
      put_operand(text, inst->vopc.src0);
      put_str(text, " vsrc1=");
      put_operand(text, inst->vopc.vsrc1);
      put_str(text, " ext=");
      put_ext(text, &inst->vopc.ext);
      break;
   case AGC_ISA_VOP3: {
      const struct agc_isa_vop3 *v = &inst->vop3;
      put_str(text, "vop3 vdst=");
      put_operand(text, v->vdst);
      put_str(text, " sdst=");
      put_operand(text, v->sdst);
      put_str(text, " src=");
      put_sources(text, v->src);
      put(text, " abs=%u neg=%u opsel=%u clamp=%u omod=%u", v->abs, v->neg, v->opsel, v->clamp, v->omod);
      break;
   }
   case AGC_ISA_VOP3P: {
      const struct agc_isa_vop3p *v = &inst->vop3p;
      put_str(text, "vop3p vdst=");
      put_operand(text, v->vdst);
      put_str(text, " src=");
      put_sources(text, v->src);
      put(text, " neg=%u neg_hi=%u opsel=%u opsel_hi=%u clamp=%u", v->neg, v->neg_hi, v->opsel, v->opsel_hi,
          v->clamp);
      break;
   }
   case AGC_ISA_VINTRP:
      put(text, "vintrp vdst=%u vsrc=%u attr=%u chan=%u", inst->vintrp.vdst, inst->vintrp.vsrc, inst->vintrp.attr,
          inst->vintrp.chan);
      break;
   case AGC_ISA_SMEM: {
      const struct agc_isa_smem *s = &inst->smem;
      put_str(text, "smem sdata=");
      put_operand(text, s->sdata);
      put(text, " sbase=%u offset=%d soffset=", s->sbase, s->offset);
      put_operand(text, s->soffset);
      put(text, " glc=%u dlc=%u dwords=%u buffer=%u", s->glc, s->dlc, s->dwords, s->buffer);
      break;
   }
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF: {
      const struct agc_isa_buffer *b = &inst->buffer;
      put(text, "%s vdata=%u vaddr=%u srsrc=%u soffset=", inst->fields == AGC_ISA_MUBUF ? "mubuf" : "mtbuf",
          b->vdata, b->vaddr, b->srsrc);
      put_operand(text, b->soffset);
      put(text, " offset=%u offen=%u idxen=%u glc=%u dlc=%u slc=%u tfe=%u lds=%u format=%u kind=%s data_dwords=%u",
          b->offset, b->offen, b->idxen, b->glc, b->dlc, b->slc, b->tfe, b->lds, b->format, kind_name(b->kind),
          b->data_dwords);
      break;
   }
   case AGC_ISA_FLAT: {
      static const char *const segments[3] = {"flat", "scratch", "global"};
      const struct agc_isa_flat *f = &inst->flat;
      put(text, "flat segment=%s vdst=%u vdata=%u vaddr=%u saddr=", f->segment < 3 ? segments[f->segment] : "?",
          f->vdst, f->vdata, f->vaddr);
      put_operand(text, f->saddr);
      put(text, " uses_vaddr=%u offset=%d glc=%u dlc=%u slc=%u lds=%u kind=%s data_dwords=%u", f->uses_vaddr,
          f->offset, f->glc, f->dlc, f->slc, f->lds, kind_name(f->kind), f->data_dwords);
      break;
   }
   case AGC_ISA_DS: {
      const struct agc_isa_ds *d = &inst->ds;
      put(text,
          "ds vdst=%u addr=%u data0=%u data1=%u offset0=%u offset1=%u gds=%u dst_dwords=%u data0_dwords=%u "
          "data1_dwords=%u uses_addr=%u writes_addr=%u",
          d->vdst, d->addr, d->data0, d->data1, d->offset0, d->offset1, d->gds, d->dst_dwords, d->data0_dwords,
          d->data1_dwords, d->uses_addr, d->writes_addr);
      break;
   }
   case AGC_ISA_MIMG: {
      const struct agc_isa_image *m = &inst->mimg;
      put(text,
          "mimg dim=%u dmask=%u unorm=%u glc=%u slc=%u dlc=%u r128=%u tfe=%u lwe=%u a16=%u d16=%u vdata=%u addr=[",
          m->dim, m->dmask, m->unorm, m->glc, m->slc, m->dlc, m->r128, m->tfe, m->lwe, m->a16, m->d16, m->vdata);
      for (unsigned i = 0; i < AGC_ISA_MAX_NSA_ADDRESSES; i++)
         put(text, i ? ",%u" : "%u", m->addr[i]);
      put(text,
          "] nsa=%u nsa_words=%u addr_dwords=%u bvh8=%u srsrc=%u ssamp=%u uses_sampler=%u kind=%s data_dwords=%u",
          m->nsa, m->nsa_words, m->addr_dwords, m->bvh8, m->srsrc, m->ssamp, m->uses_sampler, kind_name(m->kind),
          m->data_dwords);
      break;
   }
   case AGC_ISA_EXP: {
      const struct agc_isa_export *e = &inst->exp;
      put(text, "exp target=%u en=%u compr=%u done=%u vm=%u row_en=%u vsrc=[%u,%u,%u,%u]", e->target, e->en,
          e->compr, e->done, e->vm, e->row_en, e->vsrc[0], e->vsrc[1], e->vsrc[2], e->vsrc[3]);
      break;
   }
   case AGC_ISA_LDSDIR:
      put(text, "ldsdir vdst=%u attr=%u chan=%u wait_vdst=%u", inst->ldsdir.vdst, inst->ldsdir.attr,
          inst->ldsdir.chan, inst->ldsdir.wait_vdst);
      break;
   case AGC_ISA_VINTERP: {
      const struct agc_isa_vinterp *v = &inst->vinterp;
      put(text, "vinterp vdst=%u src=", v->vdst);
      put_sources(text, v->src);
      put(text, " neg=%u opsel=%u clamp=%u wait_exp=%u", v->neg, v->opsel, v->clamp, v->wait_exp);
      break;
   }
   default:
      put_str(text, "?");
      break;
   }
}

static void
put_ranges(struct text *text, const char *label, const struct agc_isa_reg_range *ranges, unsigned count)
{
   put(text, " %s=[", label);
   for (unsigned i = 0; i < count; i++)
      put(text, i ? ",%c%u+%u" : "%c%u+%u", ranges[i].file == AGC_ISA_REG_SCALAR ? 's' : 'v', ranges[i].first,
          ranges[i].count);
   put_char(text, ']');
}

static void
put_inst(struct text *text, const struct agc_isa_inst *inst, const uint32_t *words, bool regs)
{
   char scratch[AGC_ISA_MNEMONIC_SIZE];
   put(text, "%08x %s family=%s opcode=0x%x flags=0x%llx branch=", inst->pc, agc_isa_inst_mnemonic(inst, scratch),
       agc_isa_family_name((enum agc_isa_family)inst->family), inst->opcode, (unsigned long long)inst->flags);
   if (inst->has_branch_target)
      put(text, "0x%x", inst->branch_target);
   else
      put_str(text, "none");
   put_str(text, " words=[");
   for (unsigned i = 0; i < inst->word_count; i++)
      put(text, i ? ",%08x" : "%08x", words[inst->pc / 4 + i]);
   put_str(text, "] ");
   put_fields(text, inst);
   if (regs) {
      struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
      put_ranges(text, "defs32", ranges, agc_isa_inst_defs(inst, false, ranges));
      put_ranges(text, "uses32", ranges, agc_isa_inst_uses(inst, false, ranges));
      put_ranges(text, "defs64", ranges, agc_isa_inst_defs(inst, true, ranges));
      put_ranges(text, "uses64", ranges, agc_isa_inst_uses(inst, true, ranges));
   }
   put_char(text, '\n');
}

static void
put_result(struct text *text, const char *name, bool ok, const struct agc_isa_ir *ir,
           const struct agc_isa_error *error, const uint32_t *words, bool regs)
{
   if (!ok) {
      char message[AGC_ISA_ERROR_TEXT_SIZE];
      agc_isa_error_format(error, message, sizeof(message));
      put(text, "error %s %s\n", name, message);
      return;
   }
   for (size_t i = 0; i < ir->instruction_count; i++)
      put_inst(text, &ir->instructions[i], words, regs);
   put(text, "end %s code_words=%zu instructions=%zu\n", name, ir->code_words, ir->instruction_count);
}

struct file_list {
   char **paths;
   size_t count;
   size_t capacity;
};

static void
file_list_add(struct file_list *list, const char *directory, const char *name)
{
   if (list->count == list->capacity) {
      list->capacity = list->capacity ? list->capacity * 2 : 1024;
      list->paths = checked(realloc(list->paths, list->capacity * sizeof(*list->paths)));
   }
   size_t length = strlen(name) + (directory ? strlen(directory) + 1 : 0) + 1;
   char *path = checked(malloc(length));
   if (directory)
      snprintf(path, length, "%s/%s", directory, name);
   else
      snprintf(path, length, "%s", name);
   list->paths[list->count++] = path;
}

static int
compare_strings(const void *a, const void *b)
{
   return strcmp(*(char *const *)a, *(char *const *)b);
}

static bool
ends_with(const char *text, const char *suffix)
{
   size_t length = strlen(text);
   size_t suffix_length = strlen(suffix);
   return length >= suffix_length && !strcmp(text + length - suffix_length, suffix);
}

static bool
add_directory(struct file_list *list, const char *directory)
{
   size_t first = list->count;
#ifdef _WIN32
   char pattern[1024];
   snprintf(pattern, sizeof(pattern), "%s/*.bin", directory);
   struct _finddata_t found;
   intptr_t handle = _findfirst(pattern, &found);
   if (handle == -1) {
      snprintf(pattern, sizeof(pattern), "%s/*", directory);
      handle = _findfirst(pattern, &found);
      if (handle == -1)
         return false;
      _findclose(handle);
      return true;
   }
   do {
      if (!(found.attrib & _A_SUBDIR) && ends_with(found.name, ".bin"))
         file_list_add(list, directory, found.name);
   } while (_findnext(handle, &found) == 0);
   _findclose(handle);
#else
   DIR *dir = opendir(directory);
   if (!dir)
      return false;
   for (struct dirent *entry = readdir(dir); entry; entry = readdir(dir)) {
      if (ends_with(entry->d_name, ".bin"))
         file_list_add(list, directory, entry->d_name);
   }
   closedir(dir);
#endif
   qsort(list->paths + first, list->count - first, sizeof(*list->paths), compare_strings);
   return true;
}

static void
stem_of(const char *path, char *stem, size_t size)
{
   const char *name = path;
   for (const char *c = path; *c; c++) {
      if (*c == '/' || *c == '\\')
         name = c + 1;
   }
   snprintf(stem, size, "%s", name);
   if (ends_with(stem, ".bin"))
      stem[strlen(stem) - 4] = '\0';
}

static uint32_t *
read_words(const char *path, size_t *word_count)
{
   FILE *file = fopen(path, "rb");
   if (!file)
      die("%s: cannot open", path);
   fseek(file, 0, SEEK_END);
   long size = ftell(file);
   fseek(file, 0, SEEK_SET);
   if (size < 0)
      die("%s: cannot size", path);
   uint32_t *words = checked(malloc((size_t)size + 4));
   if (fread(words, 1, (size_t)size, file) != (size_t)size)
      die("%s: cannot read", path);
   fclose(file);
   *word_count = (size_t)size / 4;
   return words;
}

struct scope_line {
   char *name;
   size_t sequence;
   uint32_t *entries;
   size_t entry_count;
   bool through_jumps;
};

struct scope_file {
   struct scope_line *lines;
   size_t count;
};

static int
compare_scope_lines(const void *a, const void *b)
{
   const struct scope_line *x = a;
   const struct scope_line *y = b;
   int order = strcmp(x->name, y->name);
   if (order)
      return order;
   return x->sequence < y->sequence ? -1 : x->sequence > y->sequence;
}

static int
compare_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a;
   uint32_t y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

static size_t
sort_unique(uint32_t *values, size_t count)
{
   qsort(values, count, sizeof(*values), compare_u32);
   size_t kept = 0;
   for (size_t i = 0; i < count; i++) {
      if (!kept || values[kept - 1] != values[i])
         values[kept++] = values[i];
   }
   return kept;
}

static void
read_scopes(const char *path, struct scope_file *scopes)
{
   FILE *file = fopen(path, "rb");
   if (!file)
      die("%s: cannot open", path);
   fseek(file, 0, SEEK_END);
   long size = ftell(file);
   fseek(file, 0, SEEK_SET);
   char *data = checked(malloc((size_t)size + 1));
   if (fread(data, 1, (size_t)size, file) != (size_t)size)
      die("%s: cannot read", path);
   fclose(file);
   data[size] = '\0';
   size_t capacity = 0;
   for (char *line = data; *line;) {
      char *end = strchr(line, '\n');
      char *next = end ? end + 1 : line + strlen(line);
      if (end)
         *end = '\0';
      size_t length = strlen(line);
      if (length && line[length - 1] == '\r')
         line[length - 1] = '\0';
      char *flag = strchr(line, '\t');
      char *entries = flag ? strchr(flag + 1, '\t') : NULL;
      if (entries) {
         *flag++ = '\0';
         *entries++ = '\0';
         if (scopes->count == capacity) {
            capacity = capacity ? capacity * 2 : 256;
            scopes->lines = checked(realloc(scopes->lines, capacity * sizeof(*scopes->lines)));
         }
         struct scope_line *scope = &scopes->lines[scopes->count];
         memset(scope, 0, sizeof(*scope));
         scope->name = line;
         scope->sequence = scopes->count++;
         scope->through_jumps = !strcmp(flag, "1");
         if (strcmp(entries, "-")) {
            size_t count = 1;
            for (const char *c = entries; *c; c++)
               count += *c == ',';
            scope->entries = checked(malloc(count * sizeof(uint32_t)));
            for (char *item = entries; item;) {
               char *comma = strchr(item, ',');
               if (comma)
                  *comma = '\0';
               scope->entries[scope->entry_count++] = (uint32_t)strtoul(item, NULL, 16);
               item = comma ? comma + 1 : NULL;
            }
            scope->entry_count = sort_unique(scope->entries, scope->entry_count);
         }
      }
      line = next;
   }
   qsort(scopes->lines, scopes->count, sizeof(*scopes->lines), compare_scope_lines);
}

static size_t
first_scope(const struct scope_file *scopes, const char *name)
{
   size_t lo = 0;
   size_t hi = scopes->count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (strcmp(scopes->lines[mid].name, name) < 0)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo;
}

struct options {
   bool digest;
   bool regs;
   bool has_host;
   enum agc_isa_target host;
   const char *scopes;
   size_t shard;
   size_t shards;
   unsigned bench;
   const char *fuzz;
   uint64_t seed;
   uint64_t start;
   uint64_t count;
   uint64_t block;
};

static enum agc_isa_target
target_of(const char *name)
{
   for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
      if (!strcmp(name, agc_isa_target_name((enum agc_isa_target)target)))
         return (enum agc_isa_target)target;
   }
   die("unknown target %s", name);
   return AGC_ISA_TARGET_GFX1013;
}

static void
put_scope_header(struct text *text, const char *name, const struct agc_isa_decode_scope *scope)
{
   put(text, "scoped %s through_jumps=%u entries=[", name, scope->through_jumps);
   for (size_t i = 0; i < scope->entry_count; i++)
      put(text, i ? ",0x%x" : "0x%x", scope->entries[i]);
   put_str(text, "]\n");
}

static void
program_text(struct text *text, const char *name, const uint32_t *words, size_t word_count,
             const struct options *options, const struct agc_isa_host_tables *host,
             const struct scope_file *scopes, struct agc_isa_ir *ir)
{
   struct agc_isa_error error;
   put(text, "program %s words=%zu\n", name, word_count);
   bool ok = host ? agc_isa_host_decode(host, words, word_count, ir, &error)
                  : agc_isa_decode(words, word_count, ir, &error);
   put_result(text, name, ok, ir, &error, words, options->regs);
   if (!scopes)
      return;
   for (size_t i = first_scope(scopes, name); i < scopes->count && !strcmp(scopes->lines[i].name, name); i++) {
      const struct scope_line *line = &scopes->lines[i];
      struct agc_isa_decode_scope scope = {line->entries, line->entry_count, line->through_jumps};
      put_scope_header(text, name, &scope);
      ok = agc_isa_decode_scoped(words, word_count, &scope, ir, &error);
      put_result(text, name, ok, ir, &error, words, options->regs);
   }
}

static double
now_seconds(void)
{
   struct timespec time;
   timespec_get(&time, TIME_UTC);
   return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static void
run_bench(const struct file_list *files, const struct options *options)
{
   uint32_t **programs = checked(malloc((files->count + 1) * sizeof(*programs)));
   size_t *sizes = checked(malloc((files->count + 1) * sizeof(*sizes)));
   size_t total_words = 0;
   for (size_t i = 0; i < files->count; i++) {
      programs[i] = read_words(files->paths[i], &sizes[i]);
      total_words += sizes[i];
   }
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   size_t instructions = 0;
   size_t failed = 0;
   double started = now_seconds();
   for (unsigned round = 0; round < options->bench; round++) {
      for (size_t i = 0; i < files->count; i++) {
         struct agc_isa_error error;
         if (agc_isa_decode(programs[i], sizes[i], &ir, &error))
            instructions += ir.instruction_count;
         else
            failed++;
      }
   }
   double seconds = now_seconds() - started;
   double runs = (double)files->count * options->bench;
   printf("bench: %zu programs, %zu words, %u rounds, %zu instructions, %zu failed, %.3f s, %.0f programs/s, "
          "%.1f M words/s\n",
          files->count, total_words, options->bench, instructions / options->bench, failed / options->bench, seconds,
          runs / seconds, (double)total_words * options->bench / seconds / 1e6);
   agc_isa_ir_finish(&ir);
}

static void
run_dump(const struct file_list *files, const struct options *options)
{
   struct agc_isa_host_tables *host = NULL;
   if (options->has_host) {
      host = checked(malloc(sizeof(*host)));
      agc_isa_host_tables_init(host, options->host);
   }
   struct scope_file scopes = {0};
   if (options->scopes)
      read_scopes(options->scopes, &scopes);
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   struct text text = {0};
   for (size_t i = 0; i < files->count; i++) {
      char name[256];
      stem_of(files->paths[i], name, sizeof(name));
      size_t word_count;
      uint32_t *words = read_words(files->paths[i], &word_count);
      text.size = 0;
      program_text(&text, name, words, word_count, options, host, options->scopes ? &scopes : NULL, &ir);
      if (options->digest)
         printf("%s %016llx\n", name, (unsigned long long)fnv(text.data, text.size));
      else
         flush(&text);
      free(words);
   }
   agc_isa_ir_finish(&ir);
   free(host);
}

struct rng {
   uint64_t state;
};

static uint64_t
rng_next(struct rng *rng)
{
   rng->state += UINT64_C(0x9e3779b97f4a7c15);
   uint64_t z = rng->state;
   z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
   return z ^ (z >> 31);
}

static const uint32_t fuzz_prefixes[24][2] = {
   {0x00000000, 0x80000000}, {0x00000000, 0x80000000}, {0x7c000000, 0xfe000000}, {0x7e000000, 0xfe000000},
   {0x80000000, 0xc0000000}, {0xb0000000, 0xf0000000}, {0xbe800000, 0xff800000}, {0xbf000000, 0xff800000},
   {0xbf800000, 0xff800000}, {0xc8000000, 0xfc000000}, {0xcc000000, 0xfc000000}, {0xcd000000, 0xff000000},
   {0xce000000, 0xff000000}, {0xd4000000, 0xfc000000}, {0xd4000000, 0xfc000000}, {0xd8000000, 0xfc000000},
   {0xdc000000, 0xfc000000}, {0xe0000000, 0xfc000000}, {0xe8000000, 0xfc000000}, {0xf0000000, 0xfc000000},
   {0xf4000000, 0xfc000000}, {0xf8000000, 0xfc000000}, {0xf8000000, 0xffffe000}, {0x00000000, 0x00000000},
};

struct fuzz_template {
   unsigned count;
   uint32_t words[5];
};

static const struct fuzz_template fuzz_templates[30] = {
   {4, {0xf1949f05, 0x00012010, 0x18151411, 0x0000301c}},
   {2, {0xdb840010, 0x03080201}},
   {2, {0x060004f9, 0x02041501}},
   {2, {0x7e0002fa, 0xff00b101}},
   {2, {0x7e0002e9, 0x00123401}},
   {2, {0x7e0002ea, 0x00123401}},
   {2, {0xd54b8200, 0x240e0501}},
   {2, {0xd70f6a00, 0x00020501}},
   {3, {0xd5030000, 0x000202ff, 0x40490fdb}},
   {1, {0x7d840280}},
   {1, {0x7c280280}},
   {2, {0x7d8404f9, 0x06868501}},
   {1, {0xc8080500}},
   {2, {0xf4080100, 0xfa000010}},
   {2, {0xe0301000, 0x80020100}},
   {2, {0xea6b2000, 0x80020400}},
   {2, {0xdc308000, 0x017d0002}},
   {2, {0xdc708ff8, 0x00040302}},
   {2, {0xd8dc0201, 0x00000002}},
   {2, {0xf800180f, 0x03020100}},
   {2, {0xf0800f08, 0x00820004}},
   {3, {0xf0800f0a, 0x00820004, 0x00000007}},
   {2, {0xf1989f01, 0x00010004}},
   {5, {0xf1989f07, 0x00010004, 0x0c0b0a09, 0x100f0e0d, 0x00001411}},
   {2, {0xba800001, 0x12345678}},
   {1, {0xbb00ffff}},
   {1, {0xbefd2104}},
   {2, {0x580202ff, 0x3f800000}},
   {2, {0xcc160000, 0x60020501}},
   {1, {0xbf8c0f7f}},
};

static void
fuzz_window(struct rng *rng, uint32_t w[6])
{
   static const uint32_t codes[8] = {249, 250, 233, 234, 255, 0x80, 0x100, 0x7d};
   for (unsigned i = 0; i < 6; i++)
      w[i] = (uint32_t)rng_next(rng);
   uint64_t pick = rng_next(rng);
   if ((pick & 3) == 3) {
      const struct fuzz_template *t = &fuzz_templates[(pick >> 2) % ARRAY_SIZE(fuzz_templates)];
      memcpy(w, t->words, t->count * sizeof(uint32_t));
      uint64_t flips = (pick >> 16) & 3;
      for (uint64_t i = 0; i < flips; i++) {
         uint64_t flipped = rng_next(rng) % (32 * (uint64_t)t->count);
         w[flipped / 32] ^= UINT32_C(1) << (flipped % 32);
      }
      return;
   }
   const uint32_t *prefix = fuzz_prefixes[(pick >> 2) % ARRAY_SIZE(fuzz_prefixes)];
   w[0] = (w[0] & ~prefix[1]) | prefix[0];
   uint64_t tweak = rng_next(rng);
   switch (tweak & 7) {
   case 0:
      w[0] |= 0xffff;
      break;
   case 1:
      w[0] = (w[0] & ~UINT32_C(0x1ff)) | codes[(tweak >> 3) & 7];
      break;
   case 2:
      for (unsigned k = 0; k < 3; k++) {
         if ((tweak >> (3 + k)) & 1)
            w[1] = (w[1] & ~(UINT32_C(0x1ff) << (9 * k))) | (UINT32_C(255) << (9 * k));
      }
      break;
   case 3:
      w[0] &= prefix[1] | (uint32_t)(tweak >> 8);
      w[1] &= (uint32_t)(tweak >> 32);
      break;
   default:
      break;
   }
}

static size_t
fuzz_program(struct rng *rng, uint32_t words[48], uint32_t entries[4], size_t *entry_count, bool *through_jumps)
{
   size_t length = 1 + (size_t)(rng_next(rng) % 40);
   size_t count = 0;
   while (count < length) {
      uint64_t r = rng_next(rng);
      uint32_t small = (uint16_t)((int)((r >> 8) % 17) - 8);
      switch (r % 16) {
      case 0:
      case 1:
         words[count++] = 0xbf810000;
         break;
      case 2:
         words[count++] = 0xbf820000 | small;
         break;
      case 3:
      case 4:
         words[count++] = 0xbf840000 | small;
         break;
      case 5:
         words[count++] = 0xbe802004;
         break;
      case 6:
      case 7:
      case 8:
         words[count++] = 0x7e020200;
         break;
      case 9:
         words[count++] = 0x7e0202ff;
         break;
      case 10:
         words[count++] = 0xffffffff;
         break;
      case 11:
         words[count++] = (uint32_t)(r >> 32);
         break;
      case 12:
         words[count++] = 0xbb000000 | small;
         break;
      case 13:
         words[count++] = 0xd54b0000;
         words[count++] = 0x040e0501;
         break;
      case 14:
         words[count++] = 0xbf800000;
         break;
      default:
         words[count++] = 0xbf850000 | small;
         break;
      }
   }
   uint64_t r = rng_next(rng);
   *through_jumps = (r & 1) != 0;
   uint64_t wanted = (r >> 1) % 4;
   size_t found = 0;
   for (uint64_t i = 0; i < wanted; i++) {
      uint64_t e = rng_next(rng);
      uint32_t pc = (uint32_t)((e % ((uint64_t)count + 2)) * 4);
      if ((e >> 32) % 8 == 0)
         pc += 2;
      entries[found++] = pc;
   }
   *entry_count = sort_unique(entries, found);
   return count;
}

static void
fuzz_case(struct text *text, const struct options *options, const struct agc_isa_host_tables *host,
          struct agc_isa_ir *ir, uint64_t index)
{
   struct rng rng = {options->seed ^ (index * UINT64_C(0xd6e8feb86659fd93))};
   struct agc_isa_error error;
   if (!strcmp(options->fuzz, "scoped")) {
      uint32_t words[48];
      uint32_t entries[4];
      struct agc_isa_decode_scope scope = {entries, 0, false};
      size_t count = fuzz_program(&rng, words, entries, &scope.entry_count, &scope.through_jumps);
      put(text, "case %llu words=[", (unsigned long long)index);
      for (size_t i = 0; i < count; i++)
         put(text, i ? ",%08x" : "%08x", words[i]);
      put(text, "] through_jumps=%u entries=[", scope.through_jumps);
      for (size_t i = 0; i < scope.entry_count; i++)
         put(text, i ? ",0x%x" : "0x%x", entries[i]);
      put_str(text, "]\n");
      bool ok = agc_isa_decode_scoped(words, count, &scope, ir, &error);
      put_result(text, "case", ok, ir, &error, words, options->regs);
      return;
   }
   uint32_t w[6];
   fuzz_window(&rng, w);
   put(text, "case %llu [%08x,%08x,%08x,%08x,%08x,%08x]\n", (unsigned long long)index, w[0], w[1], w[2], w[3], w[4],
       w[5]);
   if (!host) {
      struct agc_isa_inst inst;
      if (agc_isa_decode_at(w, 6, 0, &inst, &error)) {
         put_inst(text, &inst, w, options->regs);
      } else {
         char message[AGC_ISA_ERROR_TEXT_SIZE];
         agc_isa_error_format(&error, message, sizeof(message));
         put(text, "error %s\n", message);
      }
      return;
   }
   for (size_t n = 1; n <= 6; n++) {
      bool ok = agc_isa_host_decode(host, w, n, ir, &error);
      if (!ok && error.kind == AGC_ISA_ERROR_TRUNCATED && error.pc == 0) {
         if (n == 6)
            put_str(text, "truncated\n");
         continue;
      }
      put_result(text, "case", ok, ir, &error, w, options->regs);
      break;
   }
}

static void
run_fuzz(const struct options *options)
{
   struct agc_isa_host_tables *host = NULL;
   if (!strncmp(options->fuzz, "host:", 5)) {
      host = checked(malloc(sizeof(*host)));
      agc_isa_host_tables_init(host, target_of(options->fuzz + 5));
   }
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   struct text text = {0};
   uint64_t end = options->start + options->count;
   for (uint64_t index = options->start; index < end;) {
      uint64_t stop = (index / options->block + 1) * options->block;
      if (stop > end)
         stop = end;
      text.size = 0;
      for (uint64_t i = index; i < stop; i++)
         fuzz_case(&text, options, host, &ir, i);
      if (options->digest)
         printf("block %llu %016llx\n", (unsigned long long)index, (unsigned long long)fnv(text.data, text.size));
      else
         flush(&text);
      index = stop;
   }
   agc_isa_ir_finish(&ir);
   free(host);
}

static void
put_rule(struct text *text, struct agc_isa_xlat x)
{
   static const char *const names[AGC_ISA_RULE_COUNT] = {
      "none",    "bvh8",    "bvh_stack8", "mac",     "mac_legacy", "mad",        "mad_legacy",      "madmk",
      "madak",   "dot2c",   "dot4c",      "dot_signed", "cmpk",    "hwreg",      "sendmsg",         "waitcnt",
      "waitcnt_counter",    "barrier",    "mem_time", "interp",    "wave_id",    "find_first_zero", "dot2",
   };
   put_str(text, x.rule < AGC_ISA_RULE_COUNT ? names[x.rule] : "?");
   if (x.rule == AGC_ISA_RULE_DOT_SIGNED || x.rule == AGC_ISA_RULE_CMPK || x.rule == AGC_ISA_RULE_HWREG ||
       x.rule == AGC_ISA_RULE_SENDMSG || x.rule == AGC_ISA_RULE_DOT2)
      put(text, ",0x%x", x.value);
}

static void
put_xlat(struct text *text, struct agc_isa_xlat x)
{
   switch (x.kind) {
   case AGC_ISA_XLAT_SAME:
      put_str(text, "same");
      break;
   case AGC_ISA_XLAT_OP:
      put(text, "op(0x%x)", x.value);
      break;
   case AGC_ISA_XLAT_RULE:
      put_str(text, "rule(");
      put_rule(text, x);
      put_char(text, ')');
      break;
   case AGC_ISA_XLAT_DROP:
      put_str(text, "drop");
      break;
   default:
      put_str(text, "no");
      break;
   }
}

static void
put_raw_xlats(struct text *text, struct agc_isa_xlat gfx103, struct agc_isa_xlat gfx11, struct agc_isa_xlat gfx12)
{
   put_str(text, " raw=[");
   put_xlat(text, gfx103);
   put_char(text, ',');
   put_xlat(text, gfx11);
   put_char(text, ',');
   put_xlat(text, gfx12);
   put_char(text, ']');
}

static void
put_optional_operand(struct text *text, bool present, struct agc_isa_operand op)
{
   if (present)
      put_operand(text, op);
   else
      put_str(text, "none");
}

static void
operand_line(struct text *text, struct agc_isa_operand op)
{
   put_str(text, "operand ");
   put_operand(text, op);
   put_str(text, " codes=[");
   for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
      uint16_t code;
      if (target)
         put_char(text, ',');
      if (agc_isa_operand_code(op, (enum agc_isa_target)target, &code))
         put(text, "0x%x", code);
      else
         put_char(text, 'x');
   }
   put_str(text, "] scalar=");
   uint16_t index;
   if (agc_isa_operand_scalar_index(op, &index))
      put(text, "%u", index);
   else
      put_char(text, 'x');
   put_str(text, " vgpr=");
   if (op.kind == AGC_ISA_OPERAND_VGPR)
      put(text, "%u", op.reg);
   else
      put_char(text, 'x');
   put(text, " is_vgpr=%u literal=", agc_isa_operand_is_vgpr(op));
   if (op.kind == AGC_ISA_OPERAND_LITERAL)
      put(text, "0x%x\n", op.literal);
   else
      put_str(text, "x\n");
}

static void
put_order(struct text *text, const char *name, size_t count, const void *table, size_t stride, size_t op_offset)
{
   put(text, "order %s [", name);
   for (size_t i = 0; i < count; i++) {
      uint16_t op;
      memcpy(&op, (const char *)table + i * stride + op_offset, sizeof(op));
      put(text, i ? ",0x%x" : "0x%x", op);
   }
   put_str(text, "]\n");
}

static void
run_tables(void)
{
   static const enum agc_isa_family op_families[9] = {
      AGC_ISA_SOP1, AGC_ISA_SOP2, AGC_ISA_SOPK, AGC_ISA_SOPC,  AGC_ISA_SOPP,
      AGC_ISA_VOP1, AGC_ISA_VOP2, AGC_ISA_VOP3, AGC_ISA_VOP3P,
   };
   static const char *const mem_names[AGC_ISA_MEM_TABLE_COUNT] = {
      "SMEM", "VMEM_COMMON", "MUBUF_ONLY", "MTBUF", "FLAT_ONLY",
   };
   static const char *const cmp_types[9] = {"f16", "f32", "f64", "i16", "u16", "i32", "u32", "i64", "u64"};
   struct text text = {0};
   size_t count;
   for (unsigned f = 0; f < 9; f++) {
      const struct agc_isa_op_info *ops = agc_isa_op_table(op_families[f], &count);
      put_order(&text, agc_isa_family_name(op_families[f]), count, ops, sizeof(*ops),
                offsetof(struct agc_isa_op_info, op));
   }
   for (unsigned t = 0; t < AGC_ISA_MEM_TABLE_COUNT; t++) {
      const struct agc_isa_mem_op *ops = agc_isa_mem_table((enum agc_isa_mem_table)t, &count);
      put_order(&text, mem_names[t], count, ops, sizeof(*ops), offsetof(struct agc_isa_mem_op, op));
   }
   const struct agc_isa_ds_op *ds_ops = agc_isa_ds_table(&count);
   put_order(&text, "DS", count, ds_ops, sizeof(*ds_ops), offsetof(struct agc_isa_ds_op, op));
   const struct agc_isa_image_op *image_ops = agc_isa_image_table(&count);
   put_order(&text, "MIMG", count, image_ops, sizeof(*image_ops), offsetof(struct agc_isa_image_op, op));

   for (unsigned family = 0; family < AGC_ISA_FAMILY_COUNT; family++) {
      for (unsigned op = 0; op < 0x400; op++) {
         const struct agc_isa_op_info *info = agc_isa_op_lookup((enum agc_isa_family)family, (uint16_t)op);
         if (!info)
            continue;
         put(&text, "op %s 0x%x %s flags=0x%llx", agc_isa_family_name((enum agc_isa_family)family), op, info->name,
             (unsigned long long)info->flags);
         put_raw_xlats(&text, info->gfx103, info->gfx11, info->gfx12);
         put_str(&text, " t=[");
         for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
            if (target)
               put_char(&text, ',');
            put_xlat(&text, agc_isa_op_translate(info, (enum agc_isa_target)target));
         }
         put_str(&text, "]\n");
      }
   }
   for (unsigned family = 0; family < AGC_ISA_FAMILY_COUNT; family++) {
      for (unsigned op = 0; op < 0x400; op++) {
         char scratch[AGC_ISA_MNEMONIC_SIZE];
         put(&text, "mnemonic %s 0x%x %s\n", agc_isa_family_name((enum agc_isa_family)family), op,
             agc_isa_mnemonic((enum agc_isa_family)family, (uint16_t)op, scratch));
      }
   }
   for (unsigned op = 0; op < 0x400; op++) {
      enum agc_isa_family base_family;
      uint16_t base;
      uint64_t flags = 0;
      put(&text, "vop3 0x%x base=", op);
      if (agc_isa_vop3_base((uint16_t)op, &base_family, &base))
         put(&text, "%s:0x%x", agc_isa_family_name(base_family), base);
      else
         put_str(&text, "none");
      put_str(&text, " flags=");
      if (agc_isa_vop3_flags((uint16_t)op, &flags))
         put(&text, "0x%llx", (unsigned long long)flags);
      else
         put_str(&text, "none");
      put(&text, " sources=%u\n", agc_isa_vop3_source_count((uint16_t)op, flags));
   }
   for (unsigned bits = 0; bits < 64; bits++) {
      unsigned sources;
      if (agc_isa_flag_source_count(UINT64_C(1) << bits, &sources))
         put(&text, "flag_sources %u %u\n", bits, sources);
      else
         put(&text, "flag_sources %u none\n", bits);
   }
   for (unsigned op = 0; op < 0x400; op++) {
      const struct agc_isa_vopc_info *info = agc_isa_vopc_decode((uint16_t)op);
      if (!info) {
         put(&text, "vopc 0x%x none\n", op);
         continue;
      }
      bool constant;
      put(&text, "vopc 0x%x %s type=%s cond=", op, info->name, info->type < 9 ? cmp_types[info->type] : "?");
      if (info->cond == AGC_ISA_CMP_COND_CLASS)
         put_str(&text, "class");
      else
         put(&text, "%u", info->cond);
      put(&text, " exec=%u flags=0x%llx constant=", info->exec, (unsigned long long)info->flags);
      if (agc_isa_vopc_is_constant_result(info, &constant))
         put(&text, "%u", constant);
      else
         put_str(&text, "none");
      put(&text, " gfx11=0x%x\n", info->gfx11_opcode);
   }
   static const char *const mem_labels[4] = {"smem", "mubuf", "mtbuf", "flat"};
   for (unsigned which = 0; which < 4; which++) {
      for (unsigned op = 0; op < 0x400; op++) {
         const struct agc_isa_mem_op *m = which == 0   ? agc_isa_smem_op((uint16_t)op)
                                          : which == 1 ? agc_isa_mubuf_op((uint16_t)op)
                                          : which == 2 ? agc_isa_mtbuf_op((uint16_t)op)
                                                       : agc_isa_flat_op((uint16_t)op);
         if (!m)
            continue;
         put(&text, "mem %s 0x%x %s kind=%s dwords=%u flags=0x%llx buffer=%u", mem_labels[which], op, m->name,
             kind_name(m->kind), m->dwords, (unsigned long long)m->flags, agc_isa_is_buffer_smem(m));
         put_raw_xlats(&text, m->gfx103, m->gfx11, m->gfx12);
         put_str(&text, " t=[");
         for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
            if (target)
               put_char(&text, ',');
            put_xlat(&text, agc_isa_mem_op_translate(m, (enum agc_isa_target)target));
         }
         put_str(&text, "]\n");
      }
   }
   for (unsigned op = 0; op < 0x400; op++) {
      const struct agc_isa_ds_op *d = agc_isa_ds_op((uint16_t)op);
      if (!d)
         continue;
      put(&text, "ds 0x%x %s dst=%u data0=%u data1=%u flags=0x%x t=[", op, d->name, d->dst, d->data0, d->data1,
          d->flags);
      for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
         if (target)
            put_char(&text, ',');
         put_xlat(&text, agc_isa_ds_op_translate(d, (enum agc_isa_target)target));
      }
      put_str(&text, "] swaps=[");
      for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++)
         put(&text, target ? ",%u" : "%u", agc_isa_ds_op_swaps_data(d, (enum agc_isa_target)target));
      put_str(&text, "]\n");
   }
   for (unsigned op = 0; op < 0x400; op++) {
      const struct agc_isa_image_op *image = agc_isa_image_op((uint16_t)op);
      if (!image)
         continue;
      const struct agc_isa_image_op *plain = agc_isa_image_op_plain(image);
      put(&text, "image 0x%x %s flags=0x%x kind=%s sampler=%u plain=", op, image->name, image->flags,
          kind_name(agc_isa_image_op_kind(image)), agc_isa_image_op_uses_sampler(image));
      if (plain)
         put(&text, "0x%x", plain->op);
      else
         put_str(&text, "none");
      put_raw_xlats(&text, image->gfx103, image->gfx11, image->gfx12);
      put_str(&text, " t=[");
      for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
         if (target)
            put_char(&text, ',');
         put_xlat(&text, agc_isa_image_op_translate(image, (enum agc_isa_target)target));
      }
      put_str(&text, "] addr=[");
      for (unsigned dim = 0; dim <= 8; dim++) {
         for (unsigned a16 = 0; a16 < 2; a16++)
            put(&text, "%u,", agc_isa_image_address_dwords(image->flags, (uint8_t)dim, a16 != 0));
      }
      put_str(&text, "] data=[");
      for (unsigned dmask = 0; dmask < 16; dmask++) {
         for (unsigned d16 = 0; d16 < 2; d16++)
            put(&text, "%u,", agc_isa_image_data_dwords(image->flags, (uint8_t)dmask, d16 != 0));
      }
      put_str(&text, "]\n");
   }
   put_str(&text, "vsample ");
   for (unsigned op = 0; op < 0x100; op++)
      put(&text, "%u%u", agc_isa_gfx12_vsample((uint16_t)op, false), agc_isa_gfx12_vsample((uint16_t)op, true));
   put_str(&text, "\ncoords [");
   for (unsigned dim = 0; dim <= 255; dim++)
      put(&text, "%u,", agc_isa_image_coordinates((uint8_t)dim));
   put_str(&text, "]\n");
   for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
      put(&text, "format %s [", agc_isa_target_name((enum agc_isa_target)target));
      for (unsigned format = 0; format <= 255; format++) {
         uint8_t remapped;
         if (agc_isa_remap_buffer_format((uint8_t)format, (enum agc_isa_target)target, &remapped))
            put(&text, "%u,", remapped);
         else
            put_str(&text, "x,");
      }
      put_str(&text, "]\n");
   }
   for (unsigned code = 0; code < 0x400; code++) {
      struct agc_isa_operand op;
      put(&text, "code 0x%x from=", code);
      put_optional_operand(&text, agc_isa_operand_from_code((uint16_t)code, &op), op);
      put_str(&text, " dst=");
      put_optional_operand(&text, agc_isa_operand_scalar_destination((uint16_t)code, &op), op);
      put_str(&text, " raw=");
      put_operand(&text, agc_isa_operand_from_code_or_raw((uint16_t)code));
      put_char(&text, '\n');
   }
   for (unsigned n = 0; n <= 255; n++) {
      operand_line(&text, agc_isa_operand_sgpr((uint8_t)n));
      operand_line(&text, agc_isa_operand_vgpr((uint8_t)n));
      operand_line(&text, agc_isa_operand_ttmp((uint8_t)n));
      operand_line(&text, agc_isa_operand_inline_int((int8_t)n));
      operand_line(&text, agc_isa_operand_inline_float((uint8_t)n));
   }
   const struct agc_isa_operand singles[20] = {
      agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO),
      agc_isa_operand_special(AGC_ISA_OPERAND_VCC_HI),
      agc_isa_operand_special(AGC_ISA_OPERAND_M0),
      agc_isa_operand_special(AGC_ISA_OPERAND_NULL),
      agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_LO),
      agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_HI),
      agc_isa_operand_literal(0),
      agc_isa_operand_literal(0xffffffff),
      agc_isa_operand_special(AGC_ISA_OPERAND_SHARED_BASE),
      agc_isa_operand_special(AGC_ISA_OPERAND_SHARED_LIMIT),
      agc_isa_operand_special(AGC_ISA_OPERAND_PRIVATE_BASE),
      agc_isa_operand_special(AGC_ISA_OPERAND_PRIVATE_LIMIT),
      agc_isa_operand_special(AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID),
      agc_isa_operand_special(AGC_ISA_OPERAND_VCC_Z),
      agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_Z),
      agc_isa_operand_special(AGC_ISA_OPERAND_SCC),
      agc_isa_operand_special(AGC_ISA_OPERAND_LDS_DIRECT),
      agc_isa_operand_raw(0),
      agc_isa_operand_raw(0x1ff),
      agc_isa_operand_raw(0xffff),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(singles); i++)
      operand_line(&text, singles[i]);
   for (unsigned code = 0; code <= 255; code++) {
      uint32_t bits;
      if (agc_isa_inline_float_bits((uint8_t)code, &bits))
         put(&text, "float_bits %u 0x%x\n", code, bits);
   }
   uint32_t values[128];
   unsigned value_count = 0;
   for (int value = -20; value <= 70; value++)
      values[value_count++] = (uint32_t)value;
   for (unsigned code = 0; code <= 255; code++) {
      uint32_t bits;
      if (agc_isa_inline_float_bits((uint8_t)code, &bits))
         values[value_count++] = bits;
   }
   values[value_count++] = 0x12345678;
   values[value_count++] = 0x80000000;
   values[value_count++] = 0x7fffffff;
   values[value_count++] = 0x3f000001;
   for (unsigned i = 0; i < value_count; i++) {
      struct agc_isa_operand inline_op;
      put(&text, "constant 0x%x ", values[i]);
      put_operand(&text, agc_isa_operand_constant(values[i]));
      put_char(&text, ' ');
      put_optional_operand(&text, agc_isa_operand_inline_32(values[i], &inline_op), inline_op);
      put_char(&text, '\n');
   }
   for (unsigned family = 0; family < AGC_ISA_FAMILY_COUNT; family++)
      put(&text, "family %s scalar_alu=%u vector_alu=%u vector_memory=%u\n",
          agc_isa_family_name((enum agc_isa_family)family), agc_isa_family_is_scalar_alu((enum agc_isa_family)family),
          agc_isa_family_is_vector_alu((enum agc_isa_family)family),
          agc_isa_family_is_vector_memory((enum agc_isa_family)family));
   for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++)
      put(&text, "target %s gfx11_plus=%u gfx11=%u gfx12=%u line=%u\n",
          agc_isa_target_name((enum agc_isa_target)target), agc_isa_target_is_gfx11_plus((enum agc_isa_target)target),
          agc_isa_target_is_gfx11((enum agc_isa_target)target), agc_isa_target_is_gfx12((enum agc_isa_target)target),
          agc_isa_target_instruction_cache_line_bytes((enum agc_isa_target)target));
   for (unsigned major = 9; major <= 13; major++) {
      for (unsigned minor = 0; minor <= 6; minor++) {
         for (unsigned stepping = 0; stepping <= 4; stepping++) {
            enum agc_isa_target target;
            put(&text, "gfx_ip %u.%u.%u %s\n", major, minor, stepping,
                agc_isa_target_from_gfx_ip(major, minor, stepping, &target) ? agc_isa_target_name(target) : "none");
         }
      }
   }
   struct agc_isa_host_tables *host = checked(malloc(sizeof(*host)));
   for (unsigned target = 0; target < AGC_ISA_TARGET_COUNT; target++) {
      const char *target_name = agc_isa_target_name((enum agc_isa_target)target);
      agc_isa_host_tables_init(host, (enum agc_isa_target)target);
      for (unsigned family = 0; family < AGC_ISA_FAMILY_COUNT; family++) {
         for (unsigned op = 0; op < 0x400; op++) {
            uint16_t guest;
            if (agc_isa_host_guest_opcode(host, (enum agc_isa_family)family, (uint16_t)op, &guest))
               put(&text, "host %s %s 0x%x 0x%x\n", target_name, agc_isa_family_name((enum agc_isa_family)family), op,
                   guest);
         }
      }
      put(&text, "host_format %s [", target_name);
      for (unsigned format = 0; format <= 255; format++) {
         uint8_t guest;
         if (agc_isa_host_guest_format(host, (uint8_t)format, &guest))
            put(&text, "%u,", guest);
         else
            put_str(&text, "x,");
      }
      put(&text, "]\nsigned_dot %s [", target_name);
      for (unsigned op = 0; op < 0x400; op++) {
         if (agc_isa_host_is_signed_dot(host, (uint16_t)op))
            put(&text, "0x%x,", op);
      }
      put_str(&text, "]\n");
   }
   free(host);
   const struct agc_isa_error errors[] = {
      {.kind = AGC_ISA_ERROR_UNSUPPORTED},
      {.kind = AGC_ISA_ERROR_INVALID_OPTIONS, .reason = "wave_size must be 32 or 64"},
      {.kind = AGC_ISA_ERROR_TARGET_MISMATCH, .compiler = AGC_ISA_TARGET_GFX115, .requested = AGC_ISA_TARGET_GFX12},
      {.kind = AGC_ISA_ERROR_EMPTY_PROGRAM},
      {.kind = AGC_ISA_ERROR_TRUNCATED, .pc = 0x40},
      {.kind = AGC_ISA_ERROR_INVALID_ENCODING, .pc = 8, .word = 0xfff},
      {.kind = AGC_ISA_ERROR_UNKNOWN_OPCODE, .pc = 4, .family = AGC_ISA_MIMG, .opcode = 0xe4},
      {.kind = AGC_ISA_ERROR_INVALID_OPERAND, .pc = 12, .reason = "reserved source operand encoding"},
      {.kind = AGC_ISA_ERROR_INVALID_BRANCH, .pc = 0, .branch_target = 68},
      {.kind = AGC_ISA_ERROR_MISSING_END_PROGRAM},
      {.kind = AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION, .pc = 0x10, .family = AGC_ISA_VOP3, .opcode = 0x141,
       .target = AGC_ISA_TARGET_GFX12, .reason = "no fused form"},
      {.kind = AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION, .pc = 0x14, .family = AGC_ISA_SOP1, .opcode = 0x7f,
       .target = AGC_ISA_TARGET_GFX103, .reason = "unknown"},
      {.kind = AGC_ISA_ERROR_BRANCH_OUT_OF_RANGE, .pc = 0x100},
      {.kind = AGC_ISA_ERROR_OUT_OF_REGISTERS, .vgprs = 300},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(errors); i++) {
      char message[AGC_ISA_ERROR_TEXT_SIZE];
      uint32_t pc;
      agc_isa_error_format(&errors[i], message, sizeof(message));
      put(&text, "error unsupported=%u pc=", agc_isa_error_is_unsupported(&errors[i]));
      if (agc_isa_error_pc(&errors[i], &pc))
         put(&text, "0x%x", pc);
      else
         put_str(&text, "none");
      put(&text, " text=%s\n", message);
   }
   flush(&text);
}

static uint64_t
number_of(const char *text)
{
   return strtoull(text, NULL, 0);
}

int
main(int argc, char **argv)
{
#ifdef _WIN32
   _setmode(_fileno(stdout), _O_BINARY);
#endif
   struct options options = {.shards = 1, .block = 4096};
   struct file_list files = {0};
   const char *mode = NULL;
   for (int i = 1; i < argc; i++) {
      const char *arg = argv[i];
      bool has_value = i + 1 < argc;
      if (!strcmp(arg, "--digest")) {
         options.digest = true;
      } else if (!strcmp(arg, "--regs")) {
         options.regs = true;
      } else if (!strcmp(arg, "--tables")) {
         mode = arg;
      } else if (!strcmp(arg, "--host") && has_value) {
         options.has_host = true;
         options.host = target_of(argv[++i]);
      } else if (!strcmp(arg, "--scopes") && has_value) {
         options.scopes = argv[++i];
      } else if (!strcmp(arg, "--shard") && has_value) {
         const char *value = argv[++i];
         const char *slash = strchr(value, '/');
         if (!slash)
            die("--shard takes K/N");
         options.shard = (size_t)number_of(value);
         options.shards = (size_t)number_of(slash + 1);
         if (!options.shards || options.shard >= options.shards)
            die("--shard takes K/N with K below N");
      } else if (!strcmp(arg, "--bench") && has_value) {
         options.bench = (unsigned)number_of(argv[++i]);
      } else if (!strcmp(arg, "--fuzz") && has_value) {
         options.fuzz = argv[++i];
      } else if (!strcmp(arg, "--seed") && has_value) {
         options.seed = number_of(argv[++i]);
      } else if (!strcmp(arg, "--start") && has_value) {
         options.start = number_of(argv[++i]);
      } else if (!strcmp(arg, "--count") && has_value) {
         options.count = number_of(argv[++i]);
      } else if (!strcmp(arg, "--block") && has_value) {
         options.block = number_of(argv[++i]);
         if (!options.block)
            die("--block takes a positive count");
      } else if (arg[0] == '-' && arg[1] == '-') {
         die("unknown or incomplete option %s", arg);
      } else if (!add_directory(&files, arg)) {
         file_list_add(&files, NULL, arg);
      }
   }
   if (mode && !strcmp(mode, "--tables")) {
      run_tables();
      return 0;
   }
   if (options.fuzz) {
      run_fuzz(&options);
      return 0;
   }
   if (!files.count) {
      fprintf(stderr,
              "usage: agc_isa_dump [--digest] [--regs] [--host TARGET] [--scopes FILE] [--shard K/N] [--bench ROUNDS] "
              "PROGRAM|DIRECTORY...\n"
              "       agc_isa_dump --tables\n"
              "       agc_isa_dump --fuzz inst|scoped|host:TARGET --seed S --start A --count C [--block B] [--digest]\n");
      return 2;
   }
   struct file_list selected = {0};
   for (size_t i = 0; i < files.count; i++) {
      if (i % options.shards == options.shard)
         file_list_add(&selected, NULL, files.paths[i]);
   }
   if (options.bench)
      run_bench(&selected, &options);
   else
      run_dump(&selected, &options);
   return 0;
}
