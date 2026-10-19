#ifndef PANVK_DATA_GRAPH_CSF_H
#define PANVK_DATA_GRAPH_CSF_H

#include "genxml/cs_builder.h"

#if PAN_ARCH < 15
#error "Neural dispatch requires PAN_ARCH >= 15"
#endif

struct panvk_neural_program {
   uint32_t offset[4];
   uint32_t extent[4];
   uint32_t loops[6];
   uint32_t pipeline_id;
   uint32_t dispatch_id;
   uint64_t address;
   uint32_t size;
};

static inline void
panvk_cs_neural_invalidate(struct cs_builder *b, uint16_t endpoint_mask,
                           struct cs_index flush_id, uint8_t signal_slot)
{
   assert(!(endpoint_mask & BITFIELD_BIT(signal_slot)));
   cs_wait_slots(b, endpoint_mask);
   cs_move32_to(b, flush_id, 0);
   cs_flush_caches_with_neural(
      b, MALI_CS_FLUSH_MODE_NONE, MALI_CS_FLUSH_MODE_NONE,
      MALI_CS_OTHER_FLUSH_MODE_NONE, true, flush_id,
      cs_defer(0, signal_slot));
   cs_wait_slot(b, signal_slot);
}

static inline void
panvk_cs_neural_resources(struct cs_builder *b, const uint64_t address[4],
                          const uint32_t size[4])
{
   for (unsigned i = 0; i < 4; i++)
      cs_move32_to(b, cs_reg32(b, 22 + i * 4), size[i]);
   for (unsigned i = 0; i < 4; i++)
      cs_move64_to(b, cs_reg64(b, 20 + i * 4),
                   address[i] & UINT64_C(0xffffffffffff));
}

static inline void
panvk_cs_neural_program(struct cs_builder *b,
                        const struct panvk_neural_program *program)
{
   for (unsigned i = 0; i < 4; i++)
      cs_move32_to(b, cs_reg32(b, 8 + i), program->offset[i]);
   for (unsigned i = 0; i < 4; i++)
      cs_move32_to(b, cs_reg32(b, 12 + i), program->extent[i]);
   cs_move32_to(b, cs_reg32(b, 3), program->loops[3]);
   cs_move32_to(b, cs_reg32(b, 2), program->loops[2]);
   cs_bfins_imm32(b, cs_reg32(b, 0), cs_reg32(b, 0), 16, 16,
                  5 | ((program->pipeline_id & 255) << 8));
   cs_bfins_imm32(b, cs_reg32(b, 1), cs_reg32(b, 1), 24, 8,
                  (program->dispatch_id >> 16) & 255);
   cs_bfins_imm32(b, cs_reg32(b, 1), cs_reg32(b, 1), 8, 16,
                  program->dispatch_id & 65535);
   cs_move32_to(b, cs_reg32(b, 4), (program->loops[0] & 3) |
                ((program->loops[1] & 3) << 16));
   cs_move32_to(b, cs_reg32(b, 5), (program->loops[4] & 65535) |
                ((program->loops[5] & 3) << 16));
   uint32_t rounded_size = program->size + 7;
   cs_move32_to(b, cs_reg32(b, 18), rounded_size & 0x7f8);
   cs_move32_to(b, cs_reg32(b, 16), program->address);
   cs_move32_to(b, cs_reg32(b, 17),
                ((program->address >> 32) & 0xffffff) |
                ((rounded_size >> 3) << 24));
}

static inline void
panvk_cs_neural_program_deps(struct cs_builder *b,
                             const struct panvk_neural_program *program,
                             unsigned endpoint, uint16_t wait_mask)
{
   cs_select_endpoint_sb(b, endpoint);
   panvk_cs_neural_program(b, program);
   if (wait_mask)
      cs_wait_slots(b, wait_mask);
   cs_run_neural(b, 0, 16, 36);
}

#endif
