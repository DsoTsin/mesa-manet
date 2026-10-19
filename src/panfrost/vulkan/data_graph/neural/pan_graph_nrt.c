#include "pan_graph_nrt.h"

#include <limits.h>
#include <stddef.h>

static bool
valid_target(const struct pan_graph_nrt_target *target)
{
   return target && target->shared_size && target->alignment && target->banks &&
          target->bank_parts && target->banks <= target->shared_size &&
          target->bank_parts <= target->banks &&
          target->alignment <= target->shared_size &&
          !(target->shared_size % target->banks) &&
          !(target->shared_size % target->bank_parts);
}

static uint64_t
round_up(uint64_t value, uint64_t alignment)
{
   return (value + alignment - 1) / alignment * alignment;
}

static bool
multiply(uint64_t *value, uint64_t factor)
{
   if (*value > UINT64_MAX / factor)
      return false;
   *value *= factor;
   return true;
}

struct pan_graph_nrt_target
pan_graph_nrt_target_v15(void)
{
   return (struct pan_graph_nrt_target){65536, 32, 8, 1};
}

bool
pan_graph_nrt_allocator_init(struct pan_graph_nrt_allocator *allocator,
                             const struct pan_graph_nrt_target *target)
{
   if (!allocator || !valid_target(target))
      return false;
   *allocator = (struct pan_graph_nrt_allocator){.target = *target};
   return true;
}

enum pan_graph_nrt_status
pan_graph_nrt_allocate(struct pan_graph_nrt_allocator *allocator,
                       const struct pan_graph_nrt_request *request,
                       struct pan_graph_nrt_layout *layout)
{
   if (!allocator || !request || !layout || !valid_target(&allocator->target) ||
       (unsigned)request->type > PAN_GRAPH_NRT_WEIGHT || !request->buffers ||
       request->buffers > INT32_MAX || request->initial_format > 2 ||
       request->weight_format > 2 ||
       (request->element_bits != 8 && request->element_bits != 16 &&
        request->element_bits != 32) ||
       (request->type == PAN_GRAPH_NRT_WEIGHT && request->element_bits != 8) ||
       (uint64_t)allocator->front + allocator->back != allocator->used)
      return PAN_GRAPH_NRT_INVALID;

   const struct pan_graph_nrt_target *target = &allocator->target;
   const uint32_t limit = target->shared_size / target->bank_parts;
   if (allocator->used > limit)
      return PAN_GRAPH_NRT_INVALID;

   uint64_t shape[4], block[4] = {1, 1, 1, 1}, bytes = 1, secondary = 0;
   for (unsigned i = 0; i < 4; i++) {
      if (!request->dimensions[i] || request->dimensions[i] > INT32_MAX)
         return PAN_GRAPH_NRT_INVALID;
      shape[i] = request->dimensions[i];
   }

   struct pan_graph_nrt_layout result = {
      .buffers = request->buffers,
      .format = request->weight_format,
   };
   if (request->type == PAN_GRAPH_NRT_WEIGHT) {
      shape[0] = round_up(shape[0], request->narrow_weight ? 8 : 16);
      shape[3] = round_up(shape[3], 8);
   } else {
      result.format = request->element_bits == 8 && request->initial_format < 2
                         ? 0
                         : request->element_bits == 16 &&
                                request->initial_format < 2
                            ? 1
                            : request->element_bits == 32 &&
                                   !request->initial_format
                               ? 2
                               : 3;
      block[0] = ((uint64_t)target->alignment + request->element_bits / 2 - 1) /
                 (request->element_bits / 2);
      block[1] = 4;
      for (unsigned i = 0; i < 4; i++)
         shape[i] = round_up(shape[i], block[i]);
   }

   for (unsigned i = 0; i < 4; i++) {
      if (!multiply(&bytes, shape[i]))
         return PAN_GRAPH_NRT_OUT_OF_MEMORY;
   }
   if (request->type == PAN_GRAPH_NRT_WEIGHT) {
      const uint32_t elements[] = {28, 32, 16};
      if (!request->weight_format)
         secondary = (bytes / (8 * (uint64_t)target->alignment) +
                      !!(bytes % (8 * (uint64_t)target->alignment))) *
                     target->alignment;
      bytes = bytes / elements[request->weight_format] +
              !!(bytes % elements[request->weight_format]);
      if (!multiply(&bytes, target->alignment))
         return PAN_GRAPH_NRT_OUT_OF_MEMORY;
   } else if (!multiply(&bytes, request->element_bits / 8) ||
              !multiply(&bytes, request->wide ? 2 : 1)) {
      return PAN_GRAPH_NRT_OUT_OF_MEMORY;
   }

   if (bytes > limit || secondary > limit - bytes ||
       bytes + secondary > limit - allocator->used)
      return PAN_GRAPH_NRT_OUT_OF_MEMORY;

   const uint32_t bank_size = target->shared_size / target->banks;
   const uint32_t max_banks = target->banks < 8 ? target->banks : 8;
   uint64_t size = 0, reserved = 0;
   uint32_t banks;
   for (banks = 1; banks <= max_banks; banks++) {
      uint64_t alignment = banks * (uint64_t)target->alignment;
      size = round_up(bytes, alignment) + round_up(secondary, alignment);
      reserved = banks * (uint64_t)bank_size;
      if (size <= reserved / request->buffers)
         break;
   }
   if (banks > max_banks || reserved > limit - allocator->used)
      return PAN_GRAPH_NRT_OUT_OF_MEMORY;

   result.size = size;
   result.banks = banks;
   result.base = request->front
                    ? allocator->front / bank_size
                    : (target->shared_size - allocator->back - reserved) /
                         bank_size;
   if (request->type == PAN_GRAPH_NRT_NORMAL) {
      const uint64_t x = shape[0] / block[0];
      const uint64_t y = (shape[1] / block[1]) << request->wide;
      const uint64_t z = shape[2] / block[2];
      result.dimensions[0] = x * y * z;
      result.dimensions[1] = y;
      result.dimensions[2] = 1;
      result.dimensions[3] = z * y;
   }

   allocator->used += reserved;
   if (request->front)
      allocator->front += reserved;
   else
      allocator->back += reserved;
   *layout = result;
   return PAN_GRAPH_NRT_SUCCESS;
}

bool
pan_graph_nrt_accumulator_layout(
   const struct pan_graph_nrt_accumulator_request *request,
   struct pan_graph_nrt_layout *layout)
{
   if (!request || !layout || request->initial_format > 2)
      return false;
   for (unsigned i = 0; i < 4; i++) {
      if (!request->dimensions[i] || request->dimensions[i] > INT32_MAX)
         return false;
   }
   for (unsigned i = 0; i < 2; i++) {
      if (request->input_bits[i] != 8 && request->input_bits[i] != 16 &&
          request->input_bits[i] != 32)
         return false;
   }

   struct pan_graph_nrt_layout result = {.kind = 1, .buffers = 2};
   if (request->initial_format == 2) {
      result.format = request->producer_mode ? 5 : 6;
   } else if (request->producer_mode == 3 || request->producer_mode == 4) {
      result.format = request->input_bits[0] <= 8 && request->input_bits[1] <= 8
                         ? 0
                         : 2;
   } else {
      result.format = request->producer_mode == 0;
   }

   static const uint32_t blocks[4][3] = {
      {32, 2, 2}, {8, 4, 4}, {32, 4, 1}, {8, 16, 1},
   };
   const uint32_t type = request->schedule_type < 3 ? request->schedule_type : 3;
   const uint32_t *block = blocks[type];
   uint64_t tiles[3];
   for (unsigned i = 0; i < 3; i++) {
      uint32_t alignment = block[i] << (i && request->paired_axes);
      tiles[i] = ((uint64_t)request->dimensions[i] + alignment - 1) / alignment;
   }
   tiles[2] <<= result.format > 1;
   uint64_t volume = tiles[0];
   if (!multiply(&volume, tiles[1]) || !multiply(&volume, tiles[2]) ||
       !multiply(&volume, request->paired_axes ? 4 : 1) || volume > UINT32_MAX)
      return false;
   result.dimensions[0] = volume;
   result.dimensions[1] = result.format > 1 ? 2 : 1;
   result.dimensions[2] = tiles[2];
   result.dimensions[3] = tiles[1] * tiles[2];
   *layout = result;
   return true;
}

bool
pan_graph_nrt_kind2_layout(bool flag, struct pan_graph_nrt_layout *layout)
{
   if (!layout)
      return false;
   *layout = (struct pan_graph_nrt_layout){
      .kind = 2,
      .buffers = 2,
      .format = !flag,
   };
   return true;
}

bool
pan_graph_nrt_kind3_layout(uint32_t element_bits, uint32_t initial_format,
                          struct pan_graph_nrt_layout *layout)
{
   if (!layout || initial_format > 2 ||
       (element_bits != 8 && element_bits != 16 && element_bits != 32))
      return false;
   *layout = (struct pan_graph_nrt_layout){
      .kind = 3,
      .buffers = 1,
      .format = element_bits == 16 && !initial_format ? 4 : 0,
   };
   return true;
}
