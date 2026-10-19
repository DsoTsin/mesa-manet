#include "pan_graph_alloc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

struct alloc_slot {
   uint64_t size;
   uint64_t pressure;
   bool critical;
};

struct alloc_context {
   const struct pan_graph_alloc_request *requests;
   uint32_t count;
   struct alloc_slot *slots;
   uint32_t *starts;
   uint32_t *order;
};

struct graph_mt19937 {
   uint32_t values[624];
   unsigned index;
};

static uint32_t
random_next(struct graph_mt19937 *rng)
{
   unsigned i = rng->index;
   unsigned next = (i + 1) % 624;
   uint32_t joined = (rng->values[i] & 0x80000000) |
                     (rng->values[next] & 0x7fffffff);
   uint32_t value = rng->values[(i + 397) % 624] ^ (joined >> 1) ^
                    (rng->values[next] & 1 ? 0x9908b0df : 0);
   rng->values[i] = value;
   rng->index = next;
   value ^= value >> 11;
   value ^= (value << 7) & 0x9d2c5680;
   value ^= (value << 15) & 0xefc60000;
   return value ^ (value >> 18);
}

static bool
overlap(const struct pan_graph_alloc_request *a,
        const struct pan_graph_alloc_request *b)
{
   return a->first <= b->last && b->first <= a->last;
}

static bool
before(const struct alloc_context *ctx, uint32_t a, uint32_t b)
{
   if (ctx->slots[a].pressure != ctx->slots[b].pressure)
      return ctx->slots[a].pressure > ctx->slots[b].pressure;
   const struct pan_graph_alloc_request *ra = &ctx->requests[a];
   const struct pan_graph_alloc_request *rb = &ctx->requests[b];
   uint32_t da = ra->last - ra->first, db = rb->last - rb->first;
   if (da != db)
      return da > db;
   if (ra->first != rb->first)
      return ra->first < rb->first;
   if (ctx->slots[a].size != ctx->slots[b].size)
      return ctx->slots[a].size > ctx->slots[b].size;
   return a < b;
}

static uint64_t
place(const struct alloc_context *ctx, struct pan_graph_allocation *result,
      uint64_t ceiling)
{
   for (uint32_t i = 0; i < ctx->count; i++)
      result[i].offset = UINT64_MAX;
   uint64_t peak = 0;
   for (uint32_t position = 0; position < ctx->count; position++) {
      uint32_t id = ctx->order[position];
      uint64_t offset = 0, size = ctx->slots[id].size;
      int32_t predecessor = -1;
      bool changed;
      do {
         changed = false;
         for (unsigned phase = 0; phase < 2; phase++) {
            for (uint32_t k = 0; k < ctx->count; k++) {
               uint32_t j = phase ? ctx->starts[k] : k;
               bool earlier = ctx->requests[j].first <= ctx->requests[id].first;
               if (earlier != !phase || j == id || result[j].offset == UINT64_MAX ||
                   !overlap(&ctx->requests[id], &ctx->requests[j]))
                  continue;
               uint64_t end = result[j].offset + ctx->slots[j].size;
               if (offset < end && result[j].offset < offset + size) {
                  offset = end;
                  predecessor = j;
                  changed = true;
               }
            }
         }
      } while (changed);
      result[id] = (struct pan_graph_allocation){offset, position, predecessor};
      uint64_t end = offset + size;
      if (peak < end)
         peak = end;
      if (peak > ceiling)
         break;
   }
   return peak;
}

enum pan_graph_alloc_status
pan_graph_allocate(const struct pan_graph_alloc_request *requests, uint32_t count,
                   uint64_t alignment, uint32_t max_trials,
                   struct pan_graph_allocation *allocations,
                   struct pan_graph_alloc_stats *stats)
{
   if (!stats || !alignment || (alignment & (alignment - 1)) || count > INT32_MAX ||
       (count && (!requests || !allocations)))
      return PAN_GRAPH_ALLOC_INVALID;
   if (!count) {
      *stats = (struct pan_graph_alloc_stats){.iterations = 1};
      return PAN_GRAPH_ALLOC_SUCCESS;
   }
   uint64_t total = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (requests[i].first > requests[i].last ||
          requests[i].size > UINT64_MAX - (alignment - 1))
         return PAN_GRAPH_ALLOC_INVALID;
      uint64_t size = (requests[i].size + alignment - 1) & ~(alignment - 1);
      if (total > UINT64_MAX - size)
         return PAN_GRAPH_ALLOC_INVALID;
      total += size;
   }
   struct alloc_slot *slots = calloc(count, sizeof(*slots));
   uint32_t *indices = calloc(count, 4 * sizeof(*indices));
   struct pan_graph_allocation *best = calloc(count, sizeof(*best));
   struct pan_graph_allocation *candidate = calloc(count, sizeof(*candidate));
   if (!slots || !indices || !best || !candidate) {
      free(slots);
      free(indices);
      free(best);
      free(candidate);
      return PAN_GRAPH_ALLOC_NO_MEMORY;
   }
   struct alloc_context ctx = {requests, count, slots, indices, indices + count};
   uint32_t *choices = indices + (size_t)count * 2;
   uint32_t *nonoverlap = indices + (size_t)count * 3;
   for (uint32_t i = 0; i < count; i++) {
      slots[i].size = (requests[i].size + alignment - 1) & ~(alignment - 1);
      ctx.order[i] = ctx.starts[i] = i;
   }
   uint64_t lower = 0;
   for (uint32_t i = 0; i < count; i++) {
      uint32_t time = requests[i].first;
      uint64_t pressure = 0;
      for (uint32_t j = 0; j < count; j++) {
         if (requests[j].first <= time && time <= requests[j].last)
            pressure += slots[j].size;
      }
      if (lower < pressure)
         lower = pressure;
      for (uint32_t j = 0; j < count; j++) {
         if (requests[j].first <= time && time <= requests[j].last &&
             slots[j].pressure < pressure)
            slots[j].pressure = pressure;
      }
   }
   for (uint32_t i = 1; i < count; i++) {
      uint32_t id = ctx.order[i], j = i;
      while (j && before(&ctx, id, ctx.order[j - 1])) {
         ctx.order[j] = ctx.order[j - 1];
         j--;
      }
      ctx.order[j] = id;
      id = ctx.starts[i];
      j = i;
      while (j && requests[id].first < requests[ctx.starts[j - 1]].first) {
         ctx.starts[j] = ctx.starts[j - 1];
         j--;
      }
      ctx.starts[j] = id;
   }
   uint64_t peak = place(&ctx, best, UINT64_MAX);
   uint32_t iterations = 1;
   struct graph_mt19937 rng = {.values[0] = 5489};
   for (unsigned i = 1; i < 624; i++)
      rng.values[i] = 1812433253 * (rng.values[i - 1] ^ (rng.values[i - 1] >> 30)) + i;
   for (uint32_t trial = 0; trial < max_trials && peak > lower; trial++) {
      uint32_t tail = 0;
      for (uint32_t i = 0; i < count; i++) {
         slots[i].critical = false;
         if (best[i].offset + slots[i].size > best[tail].offset + slots[tail].size)
            tail = i;
      }
      for (uint32_t i = 0; i < count; i++) {
         if (i != tail && !overlap(&requests[i], &requests[tail]))
            continue;
         for (int32_t j = i; j >= 0; j = best[j].predecessor)
            slots[best[j].position].critical = true;
      }
      uint32_t n = 0, m = 0;
      for (uint32_t i = 0; i < count; i++) {
         if (!slots[i].critical)
            continue;
         choices[n++] = i;
         if (!overlap(&requests[ctx.order[i]], &requests[tail]))
            nonoverlap[m++] = i;
      }
      if (n < 2)
         break;
      uint32_t a = random_next(&rng), b = random_next(&rng), c = random_next(&rng);
      uint32_t first = a % 100 <= 29 && m ? nonoverlap[b % m] : choices[b % n];
      uint32_t second = choices[c % (n - 1)];
      if (first == second)
         second = choices[n - 1];
      uint32_t tmp = ctx.order[first];
      ctx.order[first] = ctx.order[second];
      ctx.order[second] = tmp;
      uint64_t next_peak = place(&ctx, candidate, peak);
      iterations++;
      if (next_peak <= peak) {
         peak = next_peak;
         memcpy(best, candidate, count * sizeof(*best));
      } else {
         tmp = ctx.order[first];
         ctx.order[first] = ctx.order[second];
         ctx.order[second] = tmp;
      }
   }
   memcpy(allocations, best, count * sizeof(*best));
   *stats = (struct pan_graph_alloc_stats){peak, lower, iterations};
   free(slots);
   free(indices);
   free(best);
   free(candidate);
   return PAN_GRAPH_ALLOC_SUCCESS;
}
