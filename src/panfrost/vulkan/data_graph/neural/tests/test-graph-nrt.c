#include "pan_graph_nrt.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int
main(void)
{
   const struct pan_graph_nrt_target target = {4096, 64, 16, 1};
   struct pan_graph_nrt_allocator allocator;
   assert(pan_graph_nrt_allocator_init(&allocator, &target));
   struct pan_graph_nrt_request request = {
      .dimensions = {5, 3, 2, 1},
      .element_bits = 8,
      .buffers = 2,
      .front = true,
   };
   struct pan_graph_nrt_layout front, back;
   assert(pan_graph_nrt_allocate(&allocator, &request, &front) ==
          PAN_GRAPH_NRT_SUCCESS);
   assert(front.buffers == 2 && front.size == 128 && front.base == 0 &&
          front.banks == 1 && front.dimensions[0] == 2 &&
          front.dimensions[1] == 1 && front.dimensions[2] == 1 &&
          front.dimensions[3] == 2);
   request.front = false;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_SUCCESS);
   assert(back.base == 15 && allocator.used == 512 && allocator.front == 256 &&
          allocator.back == 256);

   struct pan_graph_nrt_allocator saved = allocator;
   struct pan_graph_nrt_layout sentinel;
   memset(&sentinel, 0xa5, sizeof(sentinel));
   back = sentinel;
   request.dimensions[2] = 4096;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_OUT_OF_MEMORY);
   assert(!memcmp(&saved, &allocator, sizeof(saved)) &&
          !memcmp(&sentinel, &back, sizeof(back)));
   request.dimensions[2] = 0;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_INVALID);
   assert(!memcmp(&saved, &allocator, sizeof(saved)) &&
          !memcmp(&sentinel, &back, sizeof(back)));

   for (unsigned i = 0; i < 4; i++)
      request.dimensions[i] = INT32_MAX;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_OUT_OF_MEMORY);
   assert(!memcmp(&saved, &allocator, sizeof(saved)));

   const struct pan_graph_nrt_target large = {65536, 64, 32, 2};
   assert(pan_graph_nrt_allocator_init(&allocator, &large));
   request = (struct pan_graph_nrt_request){
      .type = PAN_GRAPH_NRT_WEIGHT,
      .dimensions = {4, 5, 3, 1},
      .element_bits = 8,
      .weight_format = 1,
      .buffers = 2,
      .front = true,
   };
   assert(pan_graph_nrt_allocate(&allocator, &request, &front) ==
          PAN_GRAPH_NRT_SUCCESS);
   assert(front.size == 3840 && front.format == 1);

   assert(pan_graph_nrt_allocator_init(&allocator, &target));
   request = (struct pan_graph_nrt_request){
      .dimensions = {16, 4, 16, 1},
      .element_bits = 8,
      .buffers = 2,
      .front = true,
   };
   assert(pan_graph_nrt_allocate(&allocator, &request, &front) ==
          PAN_GRAPH_NRT_SUCCESS);
   assert(front.banks == 8 && front.size == 1024 && allocator.used == 2048);
   request.dimensions[2] = 17;
   saved = allocator;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_OUT_OF_MEMORY);
   assert(!memcmp(&saved, &allocator, sizeof(saved)));

   assert(pan_graph_nrt_allocator_init(&allocator, &target));
   request.dimensions[2] = 2;
   for (unsigned i = 0; i < 16; i++) {
      request.front = i & 1;
      assert(pan_graph_nrt_allocate(&allocator, &request, &front) ==
             PAN_GRAPH_NRT_SUCCESS);
   }
   assert(allocator.used == target.shared_size);
   saved = allocator;
   assert(pan_graph_nrt_allocate(&allocator, &request, &back) ==
          PAN_GRAPH_NRT_OUT_OF_MEMORY);
   assert(!memcmp(&saved, &allocator, sizeof(saved)));

   struct pan_graph_nrt_accumulator_request accumulator = {
      .dimensions = {4, 3, 1, 1},
      .schedule_type = 2,
      .producer_mode = 3,
      .input_bits = {8, 8},
   };
   assert(pan_graph_nrt_accumulator_layout(&accumulator, &front));
   assert(front.kind == 1 && front.buffers == 2 && !front.format);
   assert(front.dimensions[0] == 1 && front.dimensions[1] == 1 &&
          front.dimensions[2] == 1 && front.dimensions[3] == 1);
   accumulator.input_bits[1] = 16;
   assert(pan_graph_nrt_accumulator_layout(&accumulator, &front));
   assert(front.format == 2 && front.dimensions[0] == 2 &&
          front.dimensions[1] == 2);
   back = sentinel;
   accumulator.dimensions[0] = 0;
   assert(!pan_graph_nrt_accumulator_layout(&accumulator, &back));
   assert(!memcmp(&back, &sentinel, sizeof(back)));
   for (unsigned i = 0; i < 4; i++)
      accumulator.dimensions[i] = INT32_MAX;
   assert(!pan_graph_nrt_accumulator_layout(&accumulator, &back));
   assert(!memcmp(&back, &sentinel, sizeof(back)));
   assert(pan_graph_nrt_kind2_layout(false, &front));
   assert(front.kind == 2 && front.buffers == 2 && front.format == 1);
   assert(pan_graph_nrt_kind3_layout(16, 0, &front));
   assert(front.kind == 3 && front.buffers == 1 && front.format == 4);
   puts("NRT bank capacity, opposing allocations, overflow and failure guards pass");
   return 0;
}
