#include "../nvk_ser.h"
#include <cub/block/block_radix_sort.cuh>

using nvk_ser_radix = cub::BlockRadixSort<uint32_t, NVK_SER_SORT_THREADS,
   NVK_SER_SORT_ITEMS, uint32_t, 6>;

__device__ __forceinline__ nvk_ser_queue *
nvk_ser_get_queue(nvk_ser_kernel_args args)
{
   return reinterpret_cast<nvk_ser_queue *>(args.queue);
}

__device__ __forceinline__ void
nvk_ser_pcas(uint64_t address, uint32_t qmd, uint32_t command)
{
   const uint64_t packet = qmd | (uint64_t(command) << 32);
   asm volatile("red.release.gpu.global.add.u64 [%0], %1;"
      : : "l"(address), "l"(packet) : "memory");
}

__device__ __forceinline__ uint32_t
nvk_ser_packet_key(const nvk_ser_packet &packet)
{
   if (packet.owner == NVK_SER_INVALID_OWNER)
      return UINT32_MAX;
   const uint32_t bits = min(packet.hint_bits, 32u);
   const uint32_t mask = bits == 32u ? UINT32_MAX : (1u << bits) - 1u;
   const uint32_t key = ((packet.pc * 0x9e3779b9u) ^
      (packet.hit_object_key * 0x85ebca6bu) ^ (packet.hint & mask)) & 0xffffffu;
   return key == 0xffffffu ? 0xfffffeu : key;
}

extern "C" __global__ void
nvk_ser_init_native(nvk_ser_kernel_args args)
{
   nvk_ser_queue *queue = nvk_ser_get_queue(args);
   auto *packets = reinterpret_cast<nvk_ser_packet *>(queue->packets);
   auto *owners = reinterpret_cast<uint32_t *>(queue->resume_owner_ids);
   for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        index < queue->owner_capacity; index += blockDim.x * gridDim.x) {
      packets[index] = { NVK_SER_INVALID_OWNER, 0, 0, UINT32_MAX, index, 0, 0 };
      owners[index] = index;
   }
   if (blockIdx.x == 0 && threadIdx.x == 0) {
      auto *counters = reinterpret_cast<nvk_ser_counters *>(queue->counters);
      *counters = { queue->owner_capacity, 0, 0, 0, 0, 0, 0, queue->owner_capacity };
   }
}

extern "C" __global__ void
nvk_ser_sort_native(nvk_ser_kernel_args args)
{
   __shared__ typename nvk_ser_radix::TempStorage temporary;
   nvk_ser_queue *queue = nvk_ser_get_queue(args);
   const auto *packets = reinterpret_cast<const nvk_ser_packet *>(queue->packets);
   auto *owners = reinterpret_cast<uint32_t *>(queue->resume_owner_ids);
   auto *counters = reinterpret_cast<nvk_ser_counters *>(queue->counters);
   uint32_t keys[NVK_SER_SORT_ITEMS];
   uint32_t values[NVK_SER_SORT_ITEMS];
   const uint32_t base = blockIdx.x * NVK_SER_SORT_CAPACITY;
   for (uint32_t item = 0; item < NVK_SER_SORT_ITEMS; item++) {
      const uint32_t index = base + threadIdx.x * NVK_SER_SORT_ITEMS + item;
      const bool valid = index < queue->owner_capacity &&
         packets[index].owner != NVK_SER_INVALID_OWNER;
      keys[item] = valid ? nvk_ser_packet_key(packets[index]) : UINT32_MAX;
      values[item] = valid ? index : NVK_SER_INVALID_OWNER;
   }
   nvk_ser_radix(temporary).SortBlockedToStriped(keys, values, 0, 24);
   for (uint32_t item = 0; item < NVK_SER_SORT_ITEMS; item++) {
      const uint32_t index = base + threadIdx.x + item * blockDim.x;
      if (index < queue->owner_capacity)
         owners[index] = values[item];
   }
   __syncthreads();
   if (threadIdx.x == 0) {
      __threadfence();
      if (atomicAdd(&counters->sort_completed, 1u) == gridDim.x - 1u) {
         counters->arrived = 0;
         counters->scheduled = 0;
         counters->sort_completed = 0;
         counters->epoch++;
         counters->remaining = counters->alive;
         const uint32_t groups = queue->owner_capacity / NVK_SER_SHADE_THREADS;
         if (queue->flags & NVK_SER_DEBUG_STOP_BEFORE_SHADE) {
            counters->error = 2;
            __threadfence();
            atomicExch(&counters->completed, 1u);
         } else if (!groups || groups > 0x40000u || !queue->native.reflected_sked ||
             !queue->native.shade_qmd) {
            counters->error = 1;
            __threadfence();
            atomicExch(&counters->completed, 1u);
         } else {
            __threadfence();
            nvk_ser_pcas(queue->native.reflected_sked, queue->native.shade_qmd,
               6u | ((groups - 1u) << 14));
         }
      }
   }
}

extern "C" __global__ void
nvk_ser_tail_native(nvk_ser_kernel_args args)
{
   if (threadIdx.x == 0 && blockIdx.x == 0) {
      const nvk_ser_queue *queue = nvk_ser_get_queue(args);
      auto *counters = reinterpret_cast<nvk_ser_counters *>(queue->counters);
      __threadfence();
      atomicExch(&counters->completed, 1u);
   }
}
