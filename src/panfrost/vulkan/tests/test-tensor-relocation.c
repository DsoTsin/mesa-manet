#include "panvk_tensor_csf.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t
next(uint64_t *seed)
{
   *seed ^= *seed << 13;
   *seed ^= *seed >> 7;
   *seed ^= *seed << 17;
   return *seed;
}

static void
expected(uint8_t *out, const uint8_t *template_desc, const uint8_t *bound)
{
   for (unsigned i = 0; i < PANVK_TENSOR_DESCRIPTOR_SIZE; ++i)
      out[i] = i >= 12 && i < 32 ? bound[i] : template_desc[i];
}

static unsigned
test_cpu(void)
{
   uint64_t seed = 0x37c64f1a7b2d9081;
   unsigned count = 0;

   for (unsigned n = 0; n < 1024; ++n) {
      uint8_t storage[256], before[256], exp[64];

      for (unsigned i = 0; i < sizeof(storage); ++i)
         storage[i] = next(&seed);

      for (unsigned alias = 0; alias < 7; ++alias) {
         memcpy(before, storage, sizeof(storage));
         unsigned dest = (unsigned[]){160, 32, 96, 48, 80, 32, 96}[alias];
         unsigned tmpl = alias == 5 ? 96 : 32;
         unsigned bound = alias == 6 ? 32 : 96;

         expected(exp, before + tmpl, before + bound);
         panvk_tensor_desc_relocate(storage + dest, storage + tmpl,
                                    storage + bound);
         assert(!memcmp(storage + dest, exp, sizeof(exp)));
         assert(!memcmp(storage, before, dest));
         assert(!memcmp(storage + dest + 64, before + dest + 64,
                        sizeof(storage) - dest - 64));
         memcpy(storage, before, sizeof(storage));
         ++count;
      }

      for (unsigned offset = 32; offset <= 96; offset += 16) {
         memcpy(before, storage, sizeof(storage));
         expected(exp, before + offset, before + 64);
         panvk_tensor_desc_patch(storage + offset, storage + 64);
         assert(!memcmp(storage + offset, exp, sizeof(exp)));
         assert(!memcmp(storage, before, offset));
         assert(!memcmp(storage + offset + 64, before + offset + 64,
                        sizeof(storage) - offset - 64));
         memcpy(storage, before, sizeof(storage));
         ++count;
      }
   }

   return count;
}

static unsigned
test_cs(void)
{
   const uint64_t src = 0x123456780000, dst = 0x234567890000;
   uint64_t seed = 0x88f53a9913670d2b;
   unsigned count = 0;

   for (unsigned slot = 0; slot < CS_MAX_SB_COUNT; ++slot) {
      for (unsigned layout = 0; layout < 4; ++layout) {
         unsigned src_reg = 4 + layout * 16;
         unsigned dst_reg = src_reg + 2;
         unsigned scratch_reg = src_reg + 4;
         uint64_t words[32] = {0};
         struct cs_builder b;
         struct cs_builder_conf conf = {
            .nr_registers = 96,
            .nr_kernel_registers = 4,
            .ls_sb_slot = slot,
         };
         struct cs_buffer buffer = {
            .cpu = words,
            .gpu = 0,
            .capacity = ARRAY_SIZE(words),
         };

         cs_builder_init(&b, &conf, buffer);
         cs_move64_to(&b, cs_reg64(&b, src_reg), src);
         cs_move64_to(&b, cs_reg64(&b, dst_reg), dst);
         panvk_cs_patch_tensor_desc(&b, cs_reg64(&b, dst_reg),
                                    cs_reg64(&b, src_reg),
                                    cs_reg_tuple(&b, scratch_reg, 3));
         assert(BITSET_IS_EMPTY(b.cur_ls_tracker->pending_loads));
         assert(!b.cur_ls_tracker->pending_stores);
         cs_end(&b);
         assert(b.root_chunk.size == 10);

         uint32_t regs[256] = {0};
         uint8_t source[64], output[64], exp[64];
         for (unsigned i = 0; i < sizeof(source); ++i) {
            source[i] = next(&seed);
            output[i] = next(&seed);
         }
         expected(exp, output, source);

         for (unsigned i = 0; i < b.root_chunk.size; ++i) {
            uint64_t word = words[i];
            unsigned op = word >> 56;
            unsigned reg = word >> 48 & 255;
            if (op == 1) {
               regs[reg] = word;
               regs[reg + 1] = word >> 32 & 0xffff;
            } else if (op == 3) {
               assert((word >> 16 & 0xffff) == 1u << slot);
               assert(!(word >> 33 & 1));
            } else {
               assert(op == 0x14 || op == 0x15);
               unsigned areg = word >> 40 & 255;
               unsigned mask = word >> 16 & 0xffff;
               int offset = (int16_t)word;
               uint64_t address = regs[areg] | (uint64_t)regs[areg + 1] << 32;
               assert(address == (op == 0x14 ? src : dst));
               assert((offset == 24 && mask == 3) ||
                      (offset == 12 && mask == 7));
               for (unsigned j = 0; j < 16; ++j) {
                  if (!(mask & (1u << j)))
                     continue;
                  if (op == 0x14)
                     memcpy(&regs[reg + j], source + offset + j * 4, 4);
                  else
                     memcpy(output + offset + j * 4, &regs[reg + j], 4);
               }
            }
         }
         assert(!memcmp(output, exp, sizeof(exp)));
         if (!slot && !layout) {
            const uint64_t vendor[] = {
               0x0104123456780000, 0x0106234567890000,
               0x1408040000030018, 0x0300000000010000,
               0x1508060000030018, 0x0300000000010000,
               0x140804000007000c, 0x0300000000010000,
               0x150806000007000c, 0x0300000000010000,
            };
            assert(!memcmp(words, vendor, sizeof(vendor)));
         }
         cs_builder_fini(&b);
         ++count;
      }
   }

   return count;
}

int
main(void)
{
   printf("CPU relocation: %u; CS relocation: %u\n", test_cpu(), test_cs());
   return 0;
}
