#ifndef PANVK_TENSOR_CSF_H
#define PANVK_TENSOR_CSF_H

#include "genxml/cs_builder.h"
#include "panvk_tensor_desc.h"

#if PAN_ARCH < 15
#error "Tensor descriptors require PAN_ARCH >= 15"
#endif

static inline void
panvk_cs_patch_tensor_desc(struct cs_builder *b, struct cs_index dest,
                           struct cs_index bound, struct cs_index scratch)
{
   assert(scratch.type == CS_INDEX_REGISTER);
   assert(scratch.size >= PANVK_TENSOR_DESC_STRIDE_WORDS);
   assert(dest.type == CS_INDEX_REGISTER && dest.size == 2);
   assert(bound.type == CS_INDEX_REGISTER && bound.size == 2);
   assert(scratch.reg + PANVK_TENSOR_DESC_STRIDE_WORDS <= dest.reg ||
          dest.reg + dest.size <= scratch.reg);
   assert(scratch.reg + PANVK_TENSOR_DESC_STRIDE_WORDS <= bound.reg ||
          bound.reg + bound.size <= scratch.reg);

   struct cs_index address =
      cs_extract_tuple(b, scratch, 0, PANVK_TENSOR_DESC_ADDRESS_WORDS);
   struct cs_index strides =
      cs_extract_tuple(b, scratch, 0, PANVK_TENSOR_DESC_STRIDE_WORDS);

   cs_load_to(b, address, bound, BITFIELD_MASK(PANVK_TENSOR_DESC_ADDRESS_WORDS),
              PANVK_TENSOR_DESC_ADDRESS_OFFSET);
   cs_wait_slot(b, b->conf.ls_sb_slot);
   cs_store(b, address, dest, BITFIELD_MASK(PANVK_TENSOR_DESC_ADDRESS_WORDS),
            PANVK_TENSOR_DESC_ADDRESS_OFFSET);
   cs_wait_slot(b, b->conf.ls_sb_slot);
   cs_load_to(b, strides, bound, BITFIELD_MASK(PANVK_TENSOR_DESC_STRIDE_WORDS),
              PANVK_TENSOR_DESC_STRIDE_OFFSET);
   cs_wait_slot(b, b->conf.ls_sb_slot);
   cs_store(b, strides, dest, BITFIELD_MASK(PANVK_TENSOR_DESC_STRIDE_WORDS),
            PANVK_TENSOR_DESC_STRIDE_OFFSET);
   cs_wait_slot(b, b->conf.ls_sb_slot);
}

#endif
