#ifndef AGC_TEST_ISA_LISTING_H
#define AGC_TEST_ISA_LISTING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"
#include "test.h"

struct listing_case {
   const char *name;
   const char *text;
};

struct listing_set {
   const struct listing_case *cases;
   size_t count;
   FILE *update;
};

static inline void
listing_set_open(struct listing_set *set, const struct listing_case *cases, size_t count, const char *file,
                 const char *array)
{
   set->cases = cases;
   set->count = count;
   set->update = NULL;
   const char *dir = getenv("AGC_ISA_LISTINGS");
   if (!dir)
      return;
   char path[1024];
   snprintf(path, sizeof(path), "%s/%s", dir, file);
   set->update = fopen(path, "wb");
   if (set->update)
      fprintf(set->update, "static const struct listing_case %s[] = {\n", array);
}

static inline void
listing_set_close(struct listing_set *set)
{
   if (!set->update)
      return;
   fputs("};\n", set->update);
   fclose(set->update);
   set->update = NULL;
}

static inline void
listing_record(FILE *out, const char *name, const char *text)
{
   fprintf(out, "   {\"%s\",", name);
   const char *line = text;
   while (*line) {
      const char *end = strchr(line, '\n');
      size_t length = end ? (size_t)(end - line) : strlen(line);
      fprintf(out, "\n    \"%.*s\\n\"", (int)length, line);
      line += length + (end ? 1 : 0);
   }
   if (!*text)
      fputs("\n    \"\"", out);
   fputs("},\n", out);
}

static inline bool
listing_matches(struct listing_set *set, const char *name, const char *actual)
{
   if (set->update) {
      listing_record(set->update, name, actual);
      return true;
   }
   for (size_t i = 0; i < set->count; i++) {
      if (strcmp(set->cases[i].name, name))
         continue;
      if (!strcmp(set->cases[i].text, actual))
         return true;
      fprintf(stderr, "%s: the listing changed\n--- expected\n%s--- actual\n%s---\n", name, set->cases[i].text, actual);
      return false;
   }
   fprintf(stderr, "%s: no recorded listing\n--- actual\n%s---\n", name, actual);
   return false;
}

static inline char *
listing_of_error(const struct agc_isa_error *error)
{
   char text[AGC_ISA_ERROR_TEXT_SIZE];
   agc_isa_error_format(error, text, sizeof(text));
   size_t size = strlen(text) + 16;
   char *out = malloc(size);
   if (out)
      snprintf(out, size, "refused: %s\n", text);
   return out;
}

static inline const char *
listing_block(const char *listing, const char *label)
{
   size_t length = strlen(label);
   const char *line = listing;
   while (line && *line) {
      if (!strncmp(line, label, length) && line[length] == ':')
         return line;
      line = strchr(line, '\n');
      if (line)
         line++;
   }
   return NULL;
}

static inline unsigned
listing_lines(const char *listing, const char *needle, bool in_block)
{
   unsigned count = 0;
   const char *line = listing;
   bool first = true;
   while (line && *line) {
      const char *end = strchr(line, '\n');
      size_t length = end ? (size_t)(end - line) : strlen(line);
      bool label = line[0] != ' ';
      if (in_block && label && !first)
         break;
      first = false;
      if (!label) {
         for (size_t at = 0; at + strlen(needle) <= length; at++) {
            if (!strncmp(line + at, needle, strlen(needle))) {
               count++;
               break;
            }
         }
      }
      line = end ? end + 1 : NULL;
   }
   return count;
}

static inline bool
listing_has_line(const char *listing, const char *text)
{
   size_t wanted = strlen(text);
   const char *line = listing;
   while (line && *line) {
      const char *end = strchr(line, '\n');
      size_t length = end ? (size_t)(end - line) : strlen(line);
      const char *start = line;
      while (length && *start == ' ') {
         start++;
         length--;
      }
      if (length == wanted && !strncmp(start, text, wanted))
         return true;
      line = end ? end + 1 : NULL;
   }
   return false;
}

#endif
