#ifndef PANVK_TENSOR_DESC_H
#define PANVK_TENSOR_DESC_H

#include <stdint.h>
#include <string.h>

#include "panvk_descriptor_limits.h"

enum {
   PANVK_TENSOR_DESC_STRIDE_OFFSET = 12,
   PANVK_TENSOR_DESC_STRIDE_WORDS = 3,
   PANVK_TENSOR_DESC_ADDRESS_OFFSET = 24,
   PANVK_TENSOR_DESC_ADDRESS_WORDS = 2,
};

static inline void
panvk_tensor_desc_patch(void *dest, const void *bound)
{
   uint32_t strides[PANVK_TENSOR_DESC_STRIDE_WORDS];
   uint32_t address[PANVK_TENSOR_DESC_ADDRESS_WORDS];

   memcpy(strides, (const char *)bound + PANVK_TENSOR_DESC_STRIDE_OFFSET,
          sizeof(strides));
   memcpy(address, (const char *)bound + PANVK_TENSOR_DESC_ADDRESS_OFFSET,
          sizeof(address));
   memcpy((char *)dest + PANVK_TENSOR_DESC_ADDRESS_OFFSET, address,
          sizeof(address));
   memcpy((char *)dest + PANVK_TENSOR_DESC_STRIDE_OFFSET, strides,
          sizeof(strides));
}

static inline void
panvk_tensor_desc_relocate(void *dest, const void *template_desc,
                           const void *bound)
{
   uint32_t desc[PANVK_TENSOR_DESCRIPTOR_SIZE / sizeof(uint32_t)];

   memcpy(desc, template_desc, sizeof(desc));
   panvk_tensor_desc_patch(desc, bound);
   memcpy(dest, desc, sizeof(desc));
}

#endif
