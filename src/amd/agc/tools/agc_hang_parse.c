#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_debug.h"
#include "amd_family.h"

#define HANG_DUMP_MAGIC UINT64_C(0x31474e4148434741)

struct block {
   uint64_t va;
   uint32_t dwords;
   uint32_t *data;
};

static struct block *blocks;
static uint32_t block_count;

static void
lookup(void *data, uint64_t addr, struct ac_addr_info *info)
{
   (void)data;
   memset(info, 0, sizeof(*info));
   for (uint32_t i = 0; i < block_count; i++) {
      if (addr >= blocks[i].va && addr - blocks[i].va < (uint64_t)blocks[i].dwords * 4) {
         info->cpu_addr = (uint8_t *)blocks[i].data + (addr - blocks[i].va);
         info->valid = true;
         return;
      }
   }
}

static const struct block *
find(uint64_t va)
{
   for (uint32_t i = 0; i < block_count; i++) {
      if (blocks[i].va == va)
         return &blocks[i];
   }
   return NULL;
}

int
main(int argc, char **argv)
{
   if (argc != 2) {
      fprintf(stderr, "usage: %s agc-hang-*.bin\n", argv[0]);
      return 2;
   }
   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      fprintf(stderr, "cannot open %s\n", argv[1]);
      return 2;
   }
   uint64_t magic = 0;
   uint32_t head[6] = {0};
   if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != HANG_DUMP_MAGIC || fread(head, sizeof(head), 1, f) != 1) {
      fprintf(stderr, "%s is not a hang dump\n", argv[1]);
      return 2;
   }
   uint32_t root_count = head[3];
   uint64_t *roots = calloc(root_count ? root_count : 1, sizeof(*roots));
   uint32_t *root_dwords = calloc(root_count ? root_count : 1, sizeof(*root_dwords));
   for (uint32_t i = 0; i < root_count; i++) {
      if (fread(&roots[i], sizeof(roots[i]), 1, f) != 1 || fread(&root_dwords[i], sizeof(root_dwords[i]), 1, f) != 1) {
         fprintf(stderr, "truncated root list\n");
         return 2;
      }
   }
   uint32_t capacity = 0;
   for (;;) {
      uint64_t va;
      uint32_t dwords;
      uint32_t pad;
      if (fread(&va, sizeof(va), 1, f) != 1 || fread(&dwords, sizeof(dwords), 1, f) != 1 ||
          fread(&pad, sizeof(pad), 1, f) != 1)
         break;
      if (block_count == capacity) {
         capacity = capacity ? capacity * 2 : 64;
         blocks = realloc(blocks, capacity * sizeof(*blocks));
      }
      uint32_t *data = malloc((size_t)dwords * 4 + 4);
      if (fread(data, 4, dwords, f) != dwords) {
         fprintf(stderr, "truncated block at 0x%" PRIx64 "\n", va);
         free(data);
         break;
      }
      blocks[block_count++] = (struct block){va, dwords, data};
   }
   fclose(f);
   printf("gfx level %u, family %u, %s queue, %u roots, %u blocks\n", head[0], head[1],
          head[2] ? "compute" : "graphics", root_count, block_count);
   for (uint32_t i = 0; i < block_count; i++)
      printf("block 0x%" PRIx64 " %u dwords\n", blocks[i].va, blocks[i].dwords);
   for (uint32_t i = 0; i < root_count; i++) {
      const struct block *root = find(roots[i]);
      if (!root) {
         printf("root 0x%" PRIx64 " was not dumped\n", roots[i]);
         continue;
      }
      char name[64];
      snprintf(name, sizeof(name), "root %u at 0x%" PRIx64, i, roots[i]);
      struct ac_ib_parser parser = {
         .f = stdout,
         .ib = root->data,
         .num_dw = root_dwords[i] < root->dwords ? root_dwords[i] : root->dwords,
         .gfx_level = (enum amd_gfx_level)head[0],
         .family = (enum radeon_family)head[1],
         .ip_type = head[2] ? AMD_IP_COMPUTE : AMD_IP_GFX,
         .addr_callback = lookup,
      };
      ac_parse_ib(&parser, name);
   }
   return 0;
}
