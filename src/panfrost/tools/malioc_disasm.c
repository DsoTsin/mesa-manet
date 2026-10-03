#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/stat.h>
#endif

#include "valhall/disassemble.h"

static const char library_sha256[] =
   "23f5873642b6794f2754d926b652bf43b2f4e22aa71f3c4980580e54cbfbb0fd";

struct stage {
   const char *name, *chunk;
   uint32_t model, backend;
};

static const struct stage stages[] = {
   {"vertex", "CVER", 0, 0},
   {"fragment", "CFRA", 4, 4},
   {"compute", "CCOM", 5, 5},
   {"ray_generation", NULL, 5313, 7},
   {"ray_intersection", NULL, 5314, 8},
   {"ray_anyhit", NULL, 5315, 9},
   {"ray_closest_hit", NULL, 5316, 10},
   {"ray_miss", NULL, 5317, 11},
   {"callable", NULL, 5318, 12},
};

struct arguments {
   const char *input, *prefix, *entrypoint, *library, *core, *revision;
   const struct stage *stage;
   unsigned arch;
   bool no_idvs, extract_only, mbs2, raw, instrument;
};

struct blob {
   uint8_t *data;
   size_t size;
};

struct reader {
   const uint8_t *data;
   size_t pos, end;
};

struct shader_binary {
   uint32_t header[8];
   const uint8_t *code;
   size_t size, ebin_offset, objc_offset;
};

struct shader_binaries {
   struct shader_binary *items;
   uint32_t count;
};

struct key_values {
   uint32_t count;
   const char **values;
};

struct compiler_output {
   uint32_t count;
   struct key_values *variants;
   void *opaque10, *opaque18;
   uint32_t error_count;
   const char **errors;
   uint32_t warning_count;
   const char **warnings;
};

static bool
fail(const char *format, ...)
{
   va_list args;
   fputs("malioc ISA dump failed: ", stderr);
   va_start(args, format);
   vfprintf(stderr, format, args);
   va_end(args);
   fputc('\n', stderr);
   return false;
}

#ifdef _WIN32
static wchar_t *
wide_path(const char *path)
{
   int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
   if (!count)
      return NULL;
   wchar_t *wide = malloc((size_t)count * sizeof(*wide));
   if (wide && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
                                   wide, count)) {
      free(wide);
      return NULL;
   }
   return wide;
}
#endif

static FILE *
open_file(const char *path, const char *mode)
{
#ifdef _WIN32
   wchar_t *wide = wide_path(path);
   FILE *file = wide ? _wfopen(wide, !strcmp(mode, "rb") ? L"rb" : L"wb") : NULL;
   free(wide);
   return file;
#else
   return fopen(path, mode);
#endif
}

static bool
read_file(const char *path, struct blob *blob)
{
   FILE *file = open_file(path, "rb");
   if (!file)
      return fail("cannot open %s: %s", path, strerror(errno));
#ifdef _WIN32
   int seek_status = _fseeki64(file, 0, SEEK_END);
   int64_t length = _ftelli64(file);
#else
   int seek_status = fseek(file, 0, SEEK_END);
   int64_t length = ftell(file);
#endif
   bool ok = false;
   if (seek_status || length < 0 || (uint64_t)length > SIZE_MAX - 8) {
      fail("invalid file length: %s", path);
      goto done;
   }
   rewind(file);
   blob->size = (size_t)length;
   blob->data = calloc(1, blob->size + 8);
   if (!blob->data) {
      fail("out of memory reading %s", path);
      goto done;
   }
   if (fread(blob->data, 1, blob->size, file) != blob->size || ferror(file)) {
      fail("cannot read %s", path);
      free(blob->data);
      memset(blob, 0, sizeof(*blob));
      goto done;
   }
   ok = true;
done:
   fclose(file);
   return ok;
}

static char *
output_path(const char *prefix, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   int suffix = vsnprintf(NULL, 0, format, args);
   va_end(args);
   size_t length = strlen(prefix);
   if (suffix < 0 || length > SIZE_MAX - (size_t)suffix - 1)
      return NULL;
   char *path = malloc(length + (size_t)suffix + 1);
   if (!path)
      return NULL;
   memcpy(path, prefix, length);
   va_start(args, format);
   vsnprintf(path + length, (size_t)suffix + 1, format, args);
   va_end(args);
   return path;
}

static bool
make_parents(const char *prefix)
{
   char *path = output_path(prefix, "%s", "");
   if (!path)
      return fail("out of memory creating output directories");
   bool ok = true;
   for (size_t i = 1; path[i]; ++i) {
      if (path[i] != '/' && path[i] != '\\')
         continue;
      if (i == 2 && path[1] == ':')
         continue;
      char separator = path[i];
      path[i] = 0;
#ifdef _WIN32
      wchar_t *wide = wide_path(path);
      bool created = wide && CreateDirectoryW(wide, NULL);
      DWORD error = GetLastError();
      free(wide);
      if (!created && error != ERROR_ALREADY_EXISTS) {
         ok = fail("cannot create directory %s (Windows error %lu)", path, error);
         break;
      }
#else
      if (mkdir(path, 0777) && errno != EEXIST) {
         ok = fail("cannot create directory %s: %s", path, strerror(errno));
         break;
      }
#endif
      path[i] = separator;
   }
   free(path);
   return ok;
}

static bool
write_file(const char *path, const void *data, size_t size, unsigned disasm_arch)
{
   if (!path)
      return fail("out of memory creating output path");
   FILE *file = open_file(path, "wb");
   if (!file)
      return fail("cannot create %s: %s", path, strerror(errno));
   bool ok;
   if (disasm_arch) {
      uint8_t *padded = calloc(1, size + 8);
      if (!padded) {
         fail("out of memory disassembling %s", path);
         fclose(file);
         return false;
      }
      memcpy(padded, data, size);
      disassemble_valhall(file, padded, size, disasm_arch, false);
      free(padded);
      ok = !ferror(file);
   } else {
      ok = fwrite(data, 1, size, file) == size;
   }
   if (fclose(file))
      ok = false;
   return ok || fail("cannot write %s", path);
}

static uint32_t
load_u32(const uint8_t *data)
{
   return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
          ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static bool
take(struct reader *reader, size_t size, const uint8_t **data)
{
   if (size > reader->end - reader->pos)
      return fail("truncated MBS2 data at 0x%zx", reader->pos);
   if (data)
      *data = reader->data + reader->pos;
   reader->pos += size;
   return true;
}

static bool
read_u32(struct reader *reader, uint32_t *value)
{
   const uint8_t *data;
   if (!take(reader, 4, &data))
      return false;
   *value = load_u32(data);
   return true;
}

static bool
read_chunk(struct reader *reader, const char *expected, struct reader *payload,
           const uint8_t **tag, size_t *offset)
{
   size_t start = reader->pos;
   const uint8_t *header;
   if (!take(reader, 8, &header))
      return false;
   if (expected && memcmp(header, expected, 4))
      return fail("expected %.4s chunk at 0x%zx", expected, start);
   struct reader child = {reader->data, reader->pos, reader->pos};
   if (!take(reader, load_u32(header + 4), NULL))
      return false;
   child.end = reader->pos;
   if (payload)
      *payload = child;
   if (tag)
      *tag = header;
   if (offset)
      *offset = start;
   return true;
}

static bool
extract_binaries(const struct blob *blob, const struct stage *stage,
                 struct shader_binaries *binaries)
{
   struct reader file = {blob->data, 0, blob->size}, root, shader, common;
   uint32_t version, count;
   const uint8_t *fields;
   if (!read_chunk(&file, "MBS2", &root, NULL, NULL) ||
       !read_u32(&root, &version))
      return false;
   if (file.pos != file.end || version != 54)
      return fail("unsupported MBS2 version or trailing data");
   if (!read_chunk(&root, "VEHW", NULL, NULL, NULL) ||
       !read_chunk(&root, stage->chunk, &shader, NULL, NULL) ||
       !read_chunk(&shader, "CMMN", &common, NULL, NULL) ||
       !read_chunk(&common, "VELA", NULL, NULL, NULL))
      return false;
   for (unsigned i = 0; i < 6; ++i) {
      if (!read_chunk(&common, "SSYM", NULL, NULL, NULL))
         return false;
   }
   if (!read_chunk(&common, "UBUF", NULL, NULL, NULL) ||
       !take(&common, 4, &fields))
      return false;
   if (fields[0] || fields[1])
      return fail("nonzero CMMN reserved field");
   if (!read_u32(&common, &count))
      return false;
   if (count > (common.end - common.pos) / 16)
      return fail("invalid FCST count");
   for (uint32_t i = 0; i < count; ++i) {
      struct reader constant;
      if (!read_chunk(&common, "FCST", &constant, NULL, NULL) ||
          !take(&constant, 8, NULL))
         return false;
   }
   if (!read_u32(&common, &count))
      return false;
   if (count > (common.end - common.pos) / 40)
      return fail("invalid EBIN count");
   binaries->items = calloc(count, sizeof(*binaries->items));
   if (count && !binaries->items)
      return fail("out of memory parsing EBINs");
   binaries->count = count;
   bool has_code = false;
   for (uint32_t i = 0; i < count; ++i) {
      struct shader_binary *binary = &binaries->items[i];
      struct reader ebin;
      if (!read_chunk(&common, "EBIN", &ebin, NULL, &binary->ebin_offset) ||
          !take(&ebin, 32, &fields))
         return false;
      for (unsigned j = 0; j < 8; ++j)
         binary->header[j] = load_u32(fields + j * 4);
      while (ebin.pos < ebin.end) {
         struct reader child;
         const uint8_t *tag;
         size_t offset;
         if (!read_chunk(&ebin, NULL, &child, &tag, &offset))
            return false;
         if (memcmp(tag, "OBJC", 4))
            continue;
         if (binary->code)
            return fail("multiple OBJC chunks in EBIN at 0x%zx", binary->ebin_offset);
         binary->code = blob->data + child.pos;
         binary->size = child.end - child.pos;
         binary->objc_offset = offset;
      }
      if (!binary->code || binary->size % 8)
         return fail("missing or unaligned Valhall OBJC in EBIN at 0x%zx",
                     binary->ebin_offset);
      has_code |= binary->size != 0;
   }
   return has_code || fail("MBS2 contains no shader code");
}

static bool
spirv_stage(const struct blob *blob, const char *entrypoint,
            const struct stage **selected)
{
   if (blob->size < 20 || blob->size % 4 || blob->size > UINT32_MAX)
      return fail("invalid SPIR-V length");
   if (load_u32(blob->data) != 0x07230203 || !load_u32(blob->data + 12) ||
       load_u32(blob->data + 16))
      return fail("invalid SPIR-V header");
   uint32_t matches = 0, model = UINT32_MAX;
   for (size_t pos = 20; pos < blob->size;) {
      uint32_t instruction = load_u32(blob->data + pos);
      size_t size = (instruction >> 16) * 4;
      if (!size || size > blob->size - pos)
         return fail("truncated SPIR-V instruction at word %zu", pos / 4);
      if ((instruction & 0xffff) == 15) {
         if (size < 16)
            return fail("invalid OpEntryPoint");
         const char *name = (const char *)blob->data + pos + 12;
         if (!memchr(name, 0, size - 12))
            return fail("unterminated OpEntryPoint name");
         uint32_t execution_model = load_u32(blob->data + pos + 4);
         if (!strcmp(name, entrypoint) &&
             (!*selected || execution_model == (*selected)->model)) {
            ++matches;
            model = execution_model;
         }
      }
      pos += size;
   }
   if (matches != 1)
      return fail("entrypoint '%s' is missing, ambiguous or mismatches --stage", entrypoint);
   for (unsigned i = 0; i < sizeof(stages) / sizeof(stages[0]); ++i) {
      if (stages[i].model == model) {
         *selected = &stages[i];
         return true;
      }
   }
   return fail("unsupported SPIR-V execution model %" PRIu32, model);
}

#ifdef _WIN32
struct compiler_input {
   const void *source;
   uint32_t count;
   const char **options;
   const char *core, *revision;
};

struct compilation_result {
   uint32_t errors, warnings;
   const char *log;
   const void *binary;
   uint64_t size;
   void *opaque20, *opaque28;
   void (*free_memory)(void *);
   uint32_t shader_count, reserved3c;
   void *auxiliary;
};

struct core {
   const char *name, *revision;
   uint64_t id;
};

_Static_assert(sizeof(void *) == 8, "malioc requires Windows x64");
_Static_assert(sizeof(struct compiler_input) == 40, "malioc input ABI");
_Static_assert(sizeof(struct key_values) == 16, "malioc key/value ABI");
_Static_assert(sizeof(struct compiler_output) == 64, "malioc output ABI");
_Static_assert(sizeof(struct compilation_result) == 72, "malioc result ABI");
_Static_assert(sizeof(struct core) == 24, "malioc core map ABI");

#define CHECK_OFFSET(type, field, offset) \
   _Static_assert(offsetof(struct type, field) == offset, #type "." #field " ABI")
CHECK_OFFSET(compiler_input, count, 0x08);
CHECK_OFFSET(compiler_input, options, 0x10);
CHECK_OFFSET(compiler_input, core, 0x18);
CHECK_OFFSET(compiler_input, revision, 0x20);
CHECK_OFFSET(key_values, values, 0x08);
CHECK_OFFSET(compiler_output, variants, 0x08);
CHECK_OFFSET(compiler_output, error_count, 0x20);
CHECK_OFFSET(compiler_output, errors, 0x28);
CHECK_OFFSET(compiler_output, warning_count, 0x30);
CHECK_OFFSET(compiler_output, warnings, 0x38);
CHECK_OFFSET(compilation_result, warnings, 0x04);
CHECK_OFFSET(compilation_result, log, 0x08);
CHECK_OFFSET(compilation_result, binary, 0x10);
CHECK_OFFSET(compilation_result, size, 0x18);
CHECK_OFFSET(compilation_result, opaque20, 0x20);
CHECK_OFFSET(compilation_result, opaque28, 0x28);
CHECK_OFFSET(compilation_result, free_memory, 0x30);
CHECK_OFFSET(compilation_result, shader_count, 0x38);
CHECK_OFFSET(compilation_result, auxiliary, 0x40);
#undef CHECK_OFFSET

struct compiler {
   HMODULE dll, crt;
   uint32_t core_id;
   int (*compile)(const struct compiler_input *, struct compiler_output *);
   int (*release)(struct compiler_output *);
   void (*init_options)(void *, uint32_t, uint32_t);
   bool (*set_option)(void *, int, int);
   int (*backend)(void *, const void *, size_t, void *, void *, int, uint8_t,
                  uint32_t, void *, void *, struct compilation_result **);
   void (*free_result)(struct compilation_result *);
   FARPROC allocate, deallocate;
};

static bool
sha256(const struct blob *blob, char hex[65])
{
   BCRYPT_ALG_HANDLE algorithm = NULL;
   BCRYPT_HASH_HANDLE hash = NULL;
   uint8_t digest[32];
   NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                                 NULL, 0);
   if (status >= 0)
      status = BCryptCreateHash(algorithm, &hash, NULL, 0, NULL, 0, 0);
   for (size_t pos = 0; status >= 0 && pos < blob->size;) {
      ULONG size = (ULONG)((blob->size - pos > UINT32_MAX) ? UINT32_MAX : blob->size - pos);
      status = BCryptHashData(hash, blob->data + pos, size, 0);
      pos += size;
   }
   if (status >= 0)
      status = BCryptFinishHash(hash, digest, sizeof(digest), 0);
   if (hash)
      BCryptDestroyHash(hash);
   if (algorithm)
      BCryptCloseAlgorithmProvider(algorithm, 0);
   if (status < 0)
      return fail("BCrypt SHA256 failed: 0x%08lx", (unsigned long)status);
   for (unsigned i = 0; i < sizeof(digest); ++i)
      snprintf(hex + i * 2, 3, "%02x", digest[i]);
   return true;
}

static void
close_compiler(struct compiler *compiler)
{
   if (compiler->dll)
      FreeLibrary(compiler->dll);
   if (compiler->crt)
      FreeLibrary(compiler->crt);
}

static bool
open_compiler(struct compiler *compiler, const struct arguments *args)
{
   char hash[65];
   struct blob library = {0};
   if (!read_file(args->library, &library))
      return false;
   bool verified = sha256(&library, hash);
   free(library.data);
   if (!verified)
      return false;
   if (strcmp(hash, library_sha256))
      return fail("unsupported compiler DLL: expected the analyzed malioc 2026.5 r56p1 binary");
   wchar_t *path = wide_path(args->library);
   DWORD length = path ? GetFullPathNameW(path, 0, NULL, NULL) : 0;
   wchar_t *absolute = length ? malloc((size_t)length * sizeof(*absolute)) : NULL;
   DWORD written = absolute ? GetFullPathNameW(path, length, absolute, NULL) : 0;
   if (written && written < length)
      compiler->dll = LoadLibraryExW(absolute, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
   free(path);
   free(absolute);
   if (!compiler->dll)
      return fail("cannot load compiler DLL (Windows error %lu)", GetLastError());
   compiler->crt = LoadLibraryExW(L"ucrtbase.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
   if (!compiler->crt)
      return fail("cannot load ucrtbase.dll (Windows error %lu)", GetLastError());
#define BIND(member, symbol) \
   do { \
      FARPROC address = GetProcAddress(compiler->dll, symbol); \
      if (!address) \
         return fail("compiler export not found: %s", symbol); \
      memcpy(&compiler->member, &address, sizeof(address)); \
   } while (0)
   BIND(compile, "malioc_compile");
   BIND(release, "malioc_release_outputs");
   BIND(init_options, "cmpbe_v2_init_options");
   BIND(set_option, "cmpbe_v2_set_option_value");
   BIND(backend, "cmpbe_v2_compile_single_shader");
   BIND(free_result, "cmpbe_v2_free_compilation_result");
#undef BIND
   compiler->allocate = GetProcAddress(compiler->crt, "malloc");
   compiler->deallocate = GetProcAddress(compiler->crt, "free");
   const uint32_t *count = (const uint32_t *)(uintptr_t)GetProcAddress(compiler->dll, "mali_compiler_n_cores");
   const struct core *cores = (const struct core *)(uintptr_t)GetProcAddress(compiler->dll, "mali_compiler_core_map");
   if (!compiler->allocate || !compiler->deallocate || !count || !cores)
      return fail("missing compiler core map or CRT exports");
   for (uint32_t i = 0; i < *count; ++i) {
      if (!strcmp(cores[i].name, args->core) && !strcmp(cores[i].revision, args->revision)) {
         if (cores[i].id > UINT32_MAX)
            return fail("unsupported compiler core ID");
         compiler->core_id = (uint32_t)cores[i].id;
         return true;
      }
   }
   return fail("unsupported core/revision: %s %s", args->core, args->revision);
}

static bool
compile_shader(struct compiler *compiler, const struct arguments *args,
               const struct blob *source, struct blob *binary,
               struct compiler_output *report, bool *report_owned)
{
   char length[32];
   snprintf(length, sizeof(length), "%zu", source->size);
   bool idvs = args->stage->model == 0 && !args->no_idvs;
   const char *values[] = {
      "compiler_type", "openglessl", "shader_type", args->stage->name,
      "spirv", "true", "spirv_binary_length", length,
      "spirv_entrypoint_name", args->entrypoint, "request-pilots", "true",
      "request-idvs", idvs ? "true" : "false",
   };
   struct compiler_input input = {
      source->data, sizeof(values) / sizeof(values[0]), values, args->core, args->revision,
   };
   int status = compiler->compile(&input, report);
   if (status)
      return fail("malioc_compile failed with status %d", status);
   *report_owned = true;
   if (report->error_count) {
      for (uint32_t i = 0; i < report->error_count; ++i)
         fail("%s", report->errors[i]);
      return false;
   }
   if (!report->count)
      return fail("compiler returned no variants");
   for (uint32_t i = 0; i < report->count; ++i) {
      const struct key_values *kv = &report->variants[i];
      bool valhall = false;
      if (kv->count % 2)
         return fail("invalid malioc variant key/value count");
      for (uint32_t j = 0; j < kv->count; j += 2) {
         if (!strcmp(kv->values[j], "architecture"))
            valhall = !strcmp(kv->values[j + 1], "valhall");
      }
      if (!valhall)
         return fail("ISA dumping currently supports Valhall targets only");
   }
   union { uint64_t alignment; uint8_t data[0xe0]; } context = {0};
   union { uint64_t alignment; uint8_t data[48]; } source_info = {0};
   union { uint64_t alignment; uint8_t data[104]; } options = {0};
   uint64_t variant[3] = {0};
   memcpy(context.data + 0x88, &compiler->allocate, sizeof(compiler->allocate));
   memcpy(context.data + 0x90, &compiler->deallocate, sizeof(compiler->deallocate));
   memcpy(context.data + 0xb0, &compiler->core_id, sizeof(compiler->core_id));
   context.data[0xca] = context.data[0xcf] = context.data[0xd2] = context.data[0xd4] = 1;
   context.data[0xd1] = idvs;
   memcpy(source_info.data, &args->entrypoint, sizeof(args->entrypoint));
   compiler->init_options(options.data, compiler->core_id, 0);
   options.data[0x2a] = options.data[0x2b] = 1;
   if (args->instrument && !compiler->set_option(options.data, 24, 1))
      return fail("cannot enable dynamic counters instrumentation");
   struct compilation_result *result = NULL;
   status = compiler->backend(context.data, source->data, source->size,
                              source_info.data, options.data, args->stage->backend,
                              7, 1, variant, NULL, &result);
   if (!result)
      return fail("backend compilation failed with status %d, no result", status);
   bool ok = false;
   if (status || result->errors || !result->binary || !result->size) {
      fail("backend status=%d, errors=%" PRIu32 ": %s", status,
           result->errors, result->log ? result->log : "");
   } else if (result->size > SIZE_MAX - 8) {
      fail("backend binary size exceeds address space");
   } else {
      binary->size = (size_t)result->size;
      binary->data = calloc(1, binary->size + 8);
      if (binary->data) {
         memcpy(binary->data, result->binary, binary->size);
         ok = true;
      } else {
         fail("out of memory copying compiler output");
      }
   }
   compiler->free_result(result);
   return ok;
}
#endif

static void
json_string(FILE *file, const char *value)
{
   fputc('"', file);
   for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
      if (*p == '"' || *p == '\\')
         fprintf(file, "\\%c", *p);
      else if (*p < 32)
         fprintf(file, "\\u%04x", *p);
      else
         fputc(*p, file);
   }
   fputc('"', file);
}

static bool
write_outputs(const struct arguments *args, const struct blob *mbs2,
              const struct shader_binaries *binaries,
              const struct compiler_output *report, const char *source_hash)
{
   if (!make_parents(args->prefix))
      return false;
   char *path = output_path(args->prefix, ".mbs2.bin");
   bool ok = write_file(path, mbs2->data, mbs2->size, 0);
   free(path);
   if (!ok)
      return false;
   for (uint32_t i = 0; i < binaries->count; ++i) {
      const struct shader_binary *binary = &binaries->items[i];
      path = output_path(args->prefix, ".ebin%" PRIu32 ".objc", i);
      ok = write_file(path, binary->code, binary->size, 0);
      free(path);
      if (!ok)
         return false;
      path = output_path(args->prefix, ".ebin%" PRIu32 ".isa.txt", i);
      if (!args->extract_only)
         ok = write_file(path, binary->code, binary->size, args->arch);
      printf("%s EBIN %" PRIu32 ": %zu bytes\n", args->stage->name, i, binary->size);
      free(path);
      if (!ok)
         return false;
   }
   path = output_path(args->prefix, ".json");
   FILE *file = path ? open_file(path, "wb") : NULL;
   if (!file) {
      fail("cannot create output JSON: %s", path ? path : "out of memory");
      free(path);
      return false;
   }
   fputs("{\n  \"stage\": ", file);
   json_string(file, args->stage->name);
   if (!args->mbs2) {
      fputs(",\n  \"entrypoint\": ", file);
      json_string(file, args->entrypoint);
      fputs(",\n  \"core\": ", file);
      json_string(file, args->core);
      fputs(",\n  \"revision\": ", file);
      json_string(file, args->revision);
      fprintf(file, ",\n  \"idvs\": %s,\n  \"source_sha256\": ",
              args->stage->model == 0 && !args->no_idvs ? "true" : "false");
      json_string(file, source_hash);
      fputs(",\n  \"library_sha256\": ", file);
      json_string(file, library_sha256);
   }
   fputs(",\n  \"variants\": [", file);
   for (uint32_t i = 0; i < report->count; ++i) {
      const struct key_values *kv = &report->variants[i];
      fputs(i ? ",\n    {" : "\n    {", file);
      for (uint32_t j = 0; j < kv->count; j += 2) {
         fputs(j ? ",\n      " : "\n      ", file);
         json_string(file, kv->values[j]);
         fputs(": ", file);
         json_string(file, kv->values[j + 1]);
      }
      fputs("\n    }", file);
   }
   fputs("\n  ],\n  \"warnings\": [", file);
   for (uint32_t i = 0; i < report->warning_count; ++i) {
      if (i)
         fputs(", ", file);
      json_string(file, report->warnings[i]);
   }
   fputs("],\n  \"binaries\": [", file);
   for (uint32_t i = 0; i < binaries->count; ++i) {
      const struct shader_binary *binary = &binaries->items[i];
      fprintf(file, "%s\n    {\"index\": %" PRIu32 ", \"ebin_offset\": %zu, "
              "\"objc_offset\": %zu, \"size\": %zu, \"header\": [",
              i ? "," : "", i, binary->ebin_offset, binary->objc_offset, binary->size);
      for (unsigned j = 0; j < 8; ++j)
         fprintf(file, "%s%" PRIu32, j ? ", " : "", binary->header[j]);
      fputs("], \"binary\": ", file);
      char *objc = output_path(args->prefix, ".ebin%" PRIu32 ".objc", i);
      char *isa = output_path(args->prefix, ".ebin%" PRIu32 ".isa.txt", i);
      if (!objc || !isa) {
         free(objc);
         free(isa);
         fclose(file);
         free(path);
         return fail("out of memory serializing output paths");
      }
      json_string(file, objc);
      if (!args->extract_only) {
         fputs(", \"isa\": ", file);
         json_string(file, isa);
      }
      fputc('}', file);
      free(objc);
      free(isa);
   }
   fputs("\n  ]\n}\n", file);
   ok = !ferror(file);
   if (fclose(file))
      ok = false;
   if (!ok)
      fail("cannot write %s", path);
   free(path);
   return ok;
}

static void
usage(FILE *file)
{
   fputs("Usage: malioc_disasm input.spv -o PREFIX [options]\n"
         "Compile Vulkan SPIR-V and dump every Valhall ISA variant in one process.\n"
         "  -o, --output-prefix PREFIX   Output MBS2, OBJC, ISA and JSON files\n"
         "  -n, --entrypoint NAME        Entry point (default: main)\n"
         "  --stage vertex|fragment|compute  Default: infer from OpEntryPoint\n"
         "  --library PATH              Analyzed malioc 2026.5 r56p1 DLL\n"
         "  -c, --core NAME              Default: Mali-G610\n"
         "  --revision REV              Default: r0p0\n"
         "  --arch N                    Valhall ISA version for disassembly (default: 10)\n"
         "  --no-idvs                   Combine vertex Position/Varying variants\n"
         "  --instrument                Enable dynamic counters instrumentation\n"
         "  --extract-only              Skip text ISA\n"
         "  --mbs2                      Read existing MBS2; requires --stage\n"
         "  --raw                       Disassemble raw OBJC to stdout\n"
         "  -h, --help                  Show help\n", file);
}

static bool
parse_arguments(int argc, char **argv, struct arguments *args)
{
   args->entrypoint = "main";
   args->core = "Mali-G610";
   args->revision = "r0p0";
   args->arch = 10;
   for (int i = 1; i < argc; ++i) {
      const char *option = argv[i];
      const char **value = NULL;
      if (!strcmp(option, "-o") || !strcmp(option, "--output-prefix"))
         value = &args->prefix;
      else if (!strcmp(option, "-n") || !strcmp(option, "--entrypoint"))
         value = &args->entrypoint;
      else if (!strcmp(option, "-c") || !strcmp(option, "--core"))
         value = &args->core;
      else if (!strcmp(option, "--revision"))
         value = &args->revision;
      else if (!strcmp(option, "--library"))
         value = &args->library;
      else if (!strcmp(option, "--stage")) {
         if (++i == argc)
            return fail("--stage requires a value");
         args->stage = NULL;
         for (unsigned j = 0; j < sizeof(stages) / sizeof(stages[0]); ++j) {
            if (!strcmp(argv[i], stages[j].name))
               args->stage = &stages[j];
         }
         if (!args->stage)
            return fail("unsupported stage: %s", argv[i]);
      } else if (!strcmp(option, "--arch")) {
         if (++i == argc)
            return fail("--arch requires a value");
         args->arch = (unsigned)strtoul(argv[i], NULL, 10);
         if (args->arch < 9 || args->arch > 15)
            return fail("unsupported Valhall arch: %s", argv[i]);
      } else if (!strcmp(option, "--no-idvs"))
         args->no_idvs = true;
      else if (!strcmp(option, "--instrument"))
         args->instrument = true;
      else if (!strcmp(option, "--extract-only"))
         args->extract_only = true;
      else if (!strcmp(option, "--mbs2"))
         args->mbs2 = true;
      else if (!strcmp(option, "--raw"))
         args->raw = true;
      else if (option[0] == '-')
         return fail("unknown option: %s", option);
      else if (!args->input)
         args->input = option;
      else
         return fail("unexpected argument: %s", option);
      if (value) {
         if (++i == argc || !argv[i][0])
            return fail("%s requires a value", option);
         *value = argv[i];
      }
   }
   if (!args->input || (!args->raw && !args->prefix))
      return fail("input and -o PREFIX are required");
   if (args->raw && (args->mbs2 || args->prefix))
      return fail("--raw cannot be combined with --mbs2 or -o");
   if (args->mbs2 && !args->stage)
      return fail("--mbs2 requires --stage");
   return true;
}

static int
run(int argc, char **argv)
{
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
         usage(stdout);
         return 0;
      }
   }
   struct arguments args = {0};
   if (!parse_arguments(argc, argv, &args)) {
      usage(stderr);
      return 1;
   }
   struct blob source = {0}, binary = {0};
   struct shader_binaries binaries = {0};
   struct compiler_output report = {0};
   int status = 1;
#ifdef _WIN32
   struct compiler compiler = {0};
   bool report_owned = false;
   char hash[65] = {0};
   char *default_library = NULL;
#endif
   if (!read_file(args.input, &source))
      goto done;
   if (args.raw) {
      if (source.size % 8) {
         fail("raw Valhall OBJC length must be a multiple of 8");
         goto done;
      }
#ifdef _WIN32
      if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
         fail("cannot set raw ISA output to binary mode");
         goto done;
      }
#endif
      disassemble_valhall(stdout, source.data, source.size, args.arch, false);
      status = fflush(stdout) || ferror(stdout) ? 1 : 0;
      goto done;
   }
   if (args.mbs2) {
      binary = source;
      memset(&source, 0, sizeof(source));
   } else {
      if (!spirv_stage(&source, args.entrypoint, &args.stage))
         goto done;
#ifdef _WIN32
      if (!args.library) {
         const char *program_files = getenv("ProgramFiles");
         default_library = output_path(program_files ? program_files : "C:/Program Files",
            "/Arm/Arm Performance Studio 2026.5/mali_offline_compiler/graphics/Mali-Gxx_r56p1-00rel0.dll");
         args.library = default_library;
      }
      if (!args.library) {
         fail("out of memory creating DLL path");
         goto done;
      }
      if (!sha256(&source, hash) || !open_compiler(&compiler, &args) ||
          !compile_shader(&compiler, &args, &source, &binary, &report, &report_owned))
         goto done;
#else
      fail("the analyzed compiler ABI requires Windows x64");
      goto done;
#endif
   }
   if (!extract_binaries(&binary, args.stage, &binaries))
      goto done;
   if (write_outputs(&args, &binary, &binaries, &report,
#ifdef _WIN32
                     hash
#else
                     ""
#endif
                     ))
      status = 0;
done:
#ifdef _WIN32
   if (report_owned)
      compiler.release(&report);
   close_compiler(&compiler);
   free(default_library);
#endif
   free(binaries.items);
   free(binary.data);
   free(source.data);
   return status;
}

#ifdef _WIN32
int
wmain(int argc, wchar_t **wide_argv)
{
   char **argv = calloc((size_t)argc, sizeof(*argv));
   if (!argv)
      return 1;
   int status = 1;
   for (int i = 0; i < argc; ++i) {
      int size = WideCharToMultiByte(CP_UTF8, 0, wide_argv[i], -1, NULL, 0, NULL, NULL);
      argv[i] = size ? malloc(size) : NULL;
      if (!argv[i] || !WideCharToMultiByte(CP_UTF8, 0, wide_argv[i], -1,
                                         argv[i], size, NULL, NULL))
         goto done;
   }
   status = run(argc, argv);
done:
   for (int i = 0; i < argc; ++i)
      free(argv[i]);
   free(argv);
   return status;
}
#else
int
main(int argc, char **argv)
{
   return run(argc, argv);
}
#endif
