#include "nvk_bvh_planner.h"
#include "nvk_bvh_encode_wide.h"
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cuda/atomic>
#endif
#include <math.h>

namespace {

struct nvk_planner_temp {
   int32_t count;
   int32_t squeeze_tag;
};

struct nvk_planner_task {
   int32_t pair;
   int32_t budget;
};

__device__ __forceinline__ float nvk_planner_add(float a, float b)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
   return result;
#else
   return __fadd_rn(a, b);
#endif
}

__device__ __forceinline__ float nvk_planner_mul(float a, float b)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
   return result;
#else
   return __fmul_rn(a, b);
#endif
}

__device__ __forceinline__ float nvk_planner_fma(float a, float b, float c)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
   return result;
#else
   return __fmaf_rn(a, b, c);
#endif
}

__device__ __forceinline__ float nvk_planner_rcp(float a)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(a));
   return result;
#else
   return 1.0f / a;
#endif
}

__device__ __forceinline__ uint32_t nvk_planner_local()
{
   return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

struct nvk_planner {
   const nvk_bvh_planner_args &args;

   __device__ __forceinline__ nvk_bvh_wide_workspace *work()
   {
      return reinterpret_cast<nvk_bvh_wide_workspace *>(args.workspace);
   }

   __device__ __forceinline__ nvk_planner_temp *temporary()
   {
      return reinterpret_cast<nvk_planner_temp *>(args.workspace + NVK_BVH_WIDE_WORKSPACE_SIZE);
   }

   __device__ __forceinline__ nvk_planner_task *tasks()
   {
      return reinterpret_cast<nvk_planner_task *>(args.workspace + NVK_BVH_WIDE_WORKSPACE_SIZE +
                                                uint64_t(args.capacity) * sizeof(nvk_planner_temp));
   }

   __device__ __forceinline__ uint32_t *task_counter()
   {
      return reinterpret_cast<uint32_t *>(args.workspace + 24);
   }

   __device__ __forceinline__ int32_t *primitive_counter()
   {
      return reinterpret_cast<int32_t *>(args.workspace + 28);
   }

   __device__ __forceinline__ int32_t load_shared_word(uint64_t address)
   {
#ifdef __CUDA_ARCH__
      int32_t result;
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(result) : "l"(address) : "memory");
      return result;
#else
      return reinterpret_cast<const int32_t *>(address)[0];
#endif
   }

   __device__ __forceinline__ void store_shared_word(uint64_t address, int32_t value)
   {
#ifdef __CUDA_ARCH__
      asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
#else
      reinterpret_cast<int32_t *>(address)[0] = value;
#endif
   }

   __device__ __forceinline__ nvk_bvh_wide_node load_node(int32_t index)
   {
#ifdef __CUDA_ARCH__
      nvk_bvh_wide_node node;
      uint32_t cost;
      uint64_t address = args.nodes + uint64_t(uint32_t(index)) * 16;
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" :
                   "=r"(node.child_pair), "=r"(cost), "=r"(node.subtree_primitives), "=r"(node.owner_subtree_primitives) :
                   "l"(address) : "memory");
      node.subtree_cost = __uint_as_float(cost);
      return node;
#else
      return reinterpret_cast<const nvk_bvh_wide_node *>(args.nodes)[index];
#endif
   }

   __device__ __forceinline__ void store_node(int32_t index, nvk_bvh_wide_node node)
   {
#ifdef __CUDA_ARCH__
      uint64_t address = args.nodes + uint64_t(uint32_t(index)) * 16;
      asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" : :
                   "l"(address), "r"(node.child_pair), "r"(__float_as_uint(node.subtree_cost)),
                   "r"(node.subtree_primitives), "r"(node.owner_subtree_primitives) : "memory");
#else
      reinterpret_cast<nvk_bvh_wide_node *>(args.nodes)[index] = node;
#endif
   }

   __device__ __forceinline__ int32_t load_temp(int32_t pair, uint32_t field)
   {
      return load_shared_word(args.workspace + NVK_BVH_WIDE_WORKSPACE_SIZE + uint64_t(uint32_t(pair)) * 8 + field);
   }

   __device__ __forceinline__ void store_temp(int32_t pair, uint32_t field, int32_t value)
   {
      store_shared_word(args.workspace + NVK_BVH_WIDE_WORKSPACE_SIZE + uint64_t(uint32_t(pair)) * 8 + field, value);
   }

   __device__ __forceinline__ void store_owner(int32_t index, int32_t value)
   {
      store_shared_word(args.nodes + uint64_t(uint32_t(index)) * 16 + 12, value);
   }

   __device__ __forceinline__ uint32_t primitive_count()
   {
      return reinterpret_cast<const uint32_t *>(args.ir_header)[6];
   }

   __device__ __forceinline__ bool valid_pair(uint32_t pair)
   {
      return pair == 0 || (pair >= 2 && pair < primitive_count());
   }

   __device__ __forceinline__ uint64_t bounds_temporary(uint32_t pair)
   {
      return args.workspace + NVK_BVH_WIDE_WORKSPACE_SIZE + uint64_t(pair) * 8;
   }

   __device__ __forceinline__ void initialize_bounds(uint32_t pair)
   {
      if (work()->reserved0)
         return;
      if (pair < args.capacity) {
         store_shared_word(bounds_temporary(pair), 0);
         store_shared_word(bounds_temporary(pair) + 4, -1);
      }
      if (pair == 0) {
         uint32_t reference = uint32_t(load_shared_word(args.pair_to_ir));
         while (reference != 0xffffffffu && (reference & 3u) == 1) {
            uint64_t node = args.ir + (reference & ~3u);
            uint32_t left = uint32_t(load_shared_word(node + 24));
            reference = left != 0xffffffffu ? left : uint32_t(load_shared_word(node + 28));
         }
         *primitive_counter() = reference == 0xffffffffu ? 0 : int32_t(reference & 3u);
         if (primitive_count() > args.capacity)
            atomicOr(&work()->error, NVK_BVH_PLANNER_INVALID_TOPOLOGY);
      }
   }

   __device__ __forceinline__ void link_bounds(uint32_t pair)
   {
      if (work()->reserved0)
         return;
      if (pair >= args.capacity || !valid_pair(pair) || work()->error)
         return;
      uint32_t ready = 0;
      for (uint32_t child = 0; child < 2; child++) {
         int32_t child_pair = load_node(int32_t(pair * 2 + child)).child_pair;
         if (child_pair < 0) {
            ready++;
         } else if ((child_pair & 1) || !valid_pair(uint32_t(child_pair) / 2) ||
                    uint32_t(child_pair) / 2 >= args.capacity || uint32_t(child_pair) / 2 == pair) {
            atomicOr(&work()->error, NVK_BVH_PLANNER_INVALID_TOPOLOGY);
         } else {
            store_shared_word(bounds_temporary(uint32_t(child_pair) / 2) + 4,
                              int32_t(pair * 2 + child));
         }
      }
      store_shared_word(bounds_temporary(pair), int32_t(ready));
   }

   __device__ __forceinline__ uint32_t arrive_bounds(uint32_t pair)
   {
      uint32_t *address = reinterpret_cast<uint32_t *>(bounds_temporary(pair));
#ifdef __CUDA_ARCH__
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(*address);
      return ready.fetch_add(1, cuda::memory_order_acq_rel);
#else
      return atomicAdd(address, 1u);
#endif
   }

   __device__ __forceinline__ void rebuild_bounds(uint32_t pair)
   {
      if (work()->reserved0)
         return;
      if (pair >= args.capacity || !valid_pair(pair) || work()->error)
         return;
      for (;;) {
         if (arrive_bounds(pair) != 2)
            return;
         float minimum[3] = {3.402823466e38f, 3.402823466e38f, 3.402823466e38f};
         float maximum[3] = {-3.402823466e38f, -3.402823466e38f, -3.402823466e38f};
         uint32_t references[2] = {0xffffffffu, 0xffffffffu};
         for (uint32_t child = 0; child < 2; child++) {
            uint32_t index = pair * 2 + child;
            nvk_bvh_wide_node node = load_node(int32_t(index));
            uint32_t count = 0;
            if (node.child_pair >= 0) {
               uint32_t child_pair = uint32_t(node.child_pair);
               references[child] = uint32_t(load_shared_word(args.pair_to_ir + uint64_t(child_pair / 2) * 4));
               count = (uint32_t(load_node(int32_t(child_pair)).subtree_primitives) +
                        uint32_t(load_node(int32_t(child_pair + 1)).subtree_primitives)) & 0x7fffffffu;
            } else if (node.child_pair != int32_t(0x80000000u)) {
               uint32_t original = ~uint32_t(node.child_pair);
               if (original >= args.capacity) {
                  atomicOr(&work()->error, NVK_BVH_PLANNER_INVALID_TOPOLOGY);
                  return;
               }
               references[child] = original * args.ir_leaf_size | uint32_t(*primitive_counter());
               count = 1;
            }
            if (references[child] != 0xffffffffu) {
               uint64_t source = args.ir + (references[child] & ~3u);
               for (uint32_t axis = 0; axis < 3; axis++) {
                  minimum[axis] = fminf(minimum[axis], __uint_as_float(uint32_t(load_shared_word(source + axis * 4))));
                  maximum[axis] = fmaxf(maximum[axis], __uint_as_float(uint32_t(load_shared_word(source + 12 + axis * 4))));
               }
            }
            int32_t annotated = count ? int32_t((uint32_t(node.subtree_primitives) & 0x80000000u) | count) : 0;
            store_shared_word(args.nodes + uint64_t(index) * 16 + 8, annotated);
         }
         uint32_t reference = uint32_t(load_shared_word(args.pair_to_ir + uint64_t(pair) * 4));
         uint64_t destination = args.ir + (reference & ~3u);
         for (uint32_t axis = 0; axis < 3; axis++) {
            store_shared_word(destination + axis * 4, int32_t(__float_as_uint(minimum[axis])));
            store_shared_word(destination + 12 + axis * 4, int32_t(__float_as_uint(maximum[axis])));
         }
         store_shared_word(destination + 24, int32_t(references[0]));
         store_shared_word(destination + 28, int32_t(references[1]));
         int32_t parent = load_shared_word(bounds_temporary(pair) + 4);
         if (parent < 0)
            return;
         __threadfence();
         pair = uint32_t(parent) / 2;
      }
   }

   __device__ __forceinline__ float model_area()
   {
      uint32_t reference = reinterpret_cast<const uint32_t *>(args.pair_to_ir)[0];
      const float *bounds = reinterpret_cast<const float *>(args.ir + (reference & ~3u));
      float x = nvk_planner_add(bounds[3], -bounds[0]);
      float y = nvk_planner_add(bounds[4], -bounds[1]);
      float z = nvk_planner_add(bounds[5], -bounds[2]);
      float area = nvk_planner_fma(x, y, nvk_planner_fma(y, z, nvk_planner_mul(x, z)));
      float reciprocal = nvk_planner_rcp(fmaxf(x, fmaxf(y, z)));
      return nvk_planner_mul(area, nvk_planner_mul(reciprocal, reciprocal));
   }

   __device__ __forceinline__ int minimum_wide_nodes(int primitives)
   {
      return max(primitives - int(args.max_leaf_size + 1), 0) * 2 /
             int((args.branching_factor - 2) * args.max_leaf_size + 2) + 1;
   }

   __device__ __forceinline__ void initialize(uint32_t index)
   {
      if (args.flags & NVK_BVH_PLANNER_INIT_SQUEEZE) {
         if (index < args.capacity) {
            tasks()[index].pair = -1;
            tasks()[index].budget = -1;
         }
         if (index == 0) {
            work()->allocated_nodes = 0;
            *task_counter() = 0;
            *primitive_counter() = -1;
         }
         return;
      }
      if (index < args.capacity) {
         temporary()[index].count = 0;
         temporary()[index].squeeze_tag = 0;
      }
      if (index == 0) {
         work()->allocated_nodes = 0;
         work()->reserved0 = 0;
         work()->node_count = 0;
         work()->max_leaf_primitives = args.max_leaf_size;
         work()->branching_factor = args.branching_factor;
         work()->error = args.capacity < 2 || args.branching_factor < 3 || args.branching_factor > 16 ||
                         args.max_leaf_size == 0 || args.max_leaf_size > 16 || args.max_wide_nodes == 0 ?
                         NVK_BVH_PLANNER_INVALID_CONFIGURATION : 0;
         *task_counter() = 0;
         *primitive_counter() = -1;
      }
   }

   __device__ __forceinline__ uint32_t child_mask(uint32_t lane)
   {
      if (lane == 0)
         return 2;
      if (lane >= 16)
         return 0;
      uint32_t bit = 1u << (31 - __clz(lane));
      uint32_t mask = 1u << (lane + bit);
      return mask | (mask << bit);
   }

   __device__ __forceinline__ void choose_shape(nvk_bvh_wide_node &root, int &wide_nodes, uint32_t lane)
   {
      nvk_bvh_wide_node node = root;
      int index = -1;
      uint32_t trail = lane;
      int owner_limit = -1;
      if (trail > 1) {
         do {
            index = node.child_pair + int(trail & 1u);
            node = load_node(index);
            trail >>= 1;
            if (owner_limit == -1) {
               owner_limit = node.subtree_primitives & 0x7fffffff;
               if (owner_limit <= int(args.branching_factor))
                  owner_limit = 0;
               node.owner_subtree_primitives = 0x7fffffff;
            }
         } while (trail > 1 && node.child_pair >= 0 && node.owner_subtree_primitives >= owner_limit);
      }
      bool active = lane >= 1 && (lane < 4 || node.owner_subtree_primitives >= owner_limit) && node.child_pair >= 0;
      int delta_nodes = 0;
      float delta_cost = 0.0f;
      if (active) {
         nvk_bvh_wide_node left = load_node(node.child_pair);
         nvk_bvh_wide_node right = load_node(node.child_pair + 1);
         if (index != -1) {
            delta_nodes = load_temp(node.child_pair / 2, 0);
            delta_cost = node.subtree_cost;
         }
         if (left.child_pair >= 0)
            delta_nodes -= load_temp(left.child_pair / 2, 0);
         if (right.child_pair >= 0)
            delta_nodes -= load_temp(right.child_pair / 2, 0);
         delta_cost = nvk_planner_add(delta_cost, -left.subtree_cost);
         delta_cost = nvk_planner_add(delta_cost, -right.subtree_cost);
      }
      int removals = __popc(__ballot_sync(0xffffffffu, active)) - int(args.branching_factor - 1);
      uint32_t children = child_mask(lane);
      for (; removals > 0; removals--) {
         uint32_t active_mask = __ballot_sync(0xffffffffu, active);
         bool removable = active && (active_mask & children) == 0;
         uint32_t contribution = __float_as_uint(fmaxf(delta_cost, 0.0f));
         uint32_t best = __reduce_min_sync(0xffffffffu, removable ? contribution : 0xffffffffu);
         uint32_t tied = __ballot_sync(0xffffffffu, removable && contribution == best);
         uint32_t remove = 31u - __clz(tied);
         active = active && lane != remove;
      }
      wide_nodes = __reduce_add_sync(0xffffffffu, active ? -delta_nodes : 0) + 1;
      float cost = active ? -delta_cost : 0.0f;
      for (uint32_t step = 1; step < 32; step *= 2)
         cost = nvk_planner_add(cost, __shfl_xor_sync(0xffffffffu, cost, step));
      root.subtree_cost = cost;
      if (lane == 2 || lane == 3 || (active && lane != 1))
         store_owner(index, active ? root.subtree_primitives & 0x7fffffff : 0);
   }

   __device__ __forceinline__ void execute(uint32_t lane)
   {
      if (work()->error)
         return;
      uint32_t primitives = primitive_count();
      if (primitives <= 1) {
         if (blockIdx.x == 0 && nvk_planner_local() == 0) {
            for (uint32_t child = 0; child < 2; child++) {
               nvk_bvh_wide_node node = load_node(int32_t(child));
               bool valid = node.child_pair < 0 && node.child_pair != int32_t(0x80000000u);
               node.subtree_cost = valid ? nvk_planner_mul(node.subtree_cost, args.sah_primitive_cost) : 0.0f;
               node.subtree_primitives = valid ? int32_t(0x80000001u) : 0;
               node.owner_subtree_primitives = 0;
               store_node(int32_t(child), node);
            }
            temporary()[0].count = 1;
            temporary()[0].squeeze_tag = int(primitives);
            work()->node_count = 1;
            work()->reserved0 = 1;
            *primitive_counter() = -1;
         }
         return;
      }
      float root_area = model_area();
      int current = -1;
      do {
         uint32_t active_mask = __ballot_sync(0xffffffffu, current >= 0);
         uint32_t idle_mask = __ballot_sync(0xffffffffu, current == -1);
         bool fetching = active_mask == 0;
         uint32_t allocation = 0;
         uint32_t leader = 31u - __clz(idle_mask);
         if (fetching && idle_mask && lane == leader)
            allocation = atomicAdd(&work()->allocated_nodes, uint32_t(__popc(idle_mask)));
         allocation = __shfl_sync(0xffffffffu, allocation, idle_mask ? leader : 0);
         bool skip_internal = false;
         if (fetching && current == -1) {
            uint32_t higher_mask = lane == 31 ? 0 : ~((1u << (lane + 1)) - 1u);
            uint32_t index = allocation + __popc(idle_mask & higher_mask);
            current = index < (primitives <= 2 ? 2u : primitives * 2) ? int(index) : -2;
            skip_internal = primitives > 2;
         }
         nvk_bvh_wide_node node{};
         if (current >= 0 && current != 0x7fffffff) {
            node = load_node(current);
            if (skip_internal && (node.child_pair >= 0 || node.child_pair == int32_t(0x80000000u)))
               current = -1;
         }
         int parent = node.subtree_primitives == -1 ? 0x7fffffff : node.subtree_primitives;
         float area = current == 0x7fffffff ? root_area : node.subtree_cost;
         if (current >= 0) {
            if (node.child_pair < 0) {
               node.subtree_primitives = 1;
               node.owner_subtree_primitives = int32_t(__float_as_uint(nvk_planner_mul(area, args.sah_primitive_cost)));
            } else {
               nvk_bvh_wide_node left = load_node(node.child_pair);
               nvk_bvh_wide_node right = load_node(node.child_pair + 1);
               node.subtree_primitives = (left.subtree_primitives & 0x7fffffff) + (right.subtree_primitives & 0x7fffffff);
               node.owner_subtree_primitives = int32_t(__float_as_uint(nvk_planner_add(__uint_as_float(uint32_t(left.owner_subtree_primitives)),
                                                                                   __uint_as_float(uint32_t(right.owner_subtree_primitives)))));
               int left_tag = left.child_pair < 0 ? 1 : load_temp(left.child_pair / 2, 4);
               int right_tag = right.child_pair < 0 ? 1 : load_temp(right.child_pair / 2, 4);
               int tag = (left_tag == -1 ? int(args.max_leaf_size) : left_tag) +
                         (right_tag == -1 ? int(args.max_leaf_size) : right_tag);
               store_temp(node.child_pair / 2, 4, tag * 2 >= int(args.branching_factor * args.max_leaf_size + 2) ? -1 : tag);
            }
         }
         bool large = node.subtree_primitives > int(args.branching_factor);
         uint32_t large_mask = __ballot_sync(0xffffffffu, current >= 0 && large);
         int wide_nodes = 0;
         while (large_mask) {
            uint32_t source = 31u - __clz(large_mask);
            large_mask ^= 1u << source;
            nvk_bvh_wide_node candidate;
            candidate.child_pair = __shfl_sync(0xffffffffu, node.child_pair, source);
            candidate.subtree_cost = __shfl_sync(0xffffffffu, node.subtree_cost, source);
            candidate.subtree_primitives = __shfl_sync(0xffffffffu, node.subtree_primitives, source);
            candidate.owner_subtree_primitives = __shfl_sync(0xffffffffu, node.owner_subtree_primitives, source);
            int candidate_nodes;
            choose_shape(candidate, candidate_nodes, lane);
            if (lane == source) {
               node = candidate;
               wide_nodes = candidate_nodes;
            }
         }
         if (current >= 0 && !large) {
            node.subtree_cost = __uint_as_float(uint32_t(node.owner_subtree_primitives));
            wide_nodes = 1;
            if (node.child_pair >= 0) {
               store_owner(node.child_pair, 0);
               store_owner(node.child_pair + 1, 0);
            }
         }
         if (current >= 0) {
            node.subtree_cost = nvk_planner_fma(area, args.sah_node_cost, node.subtree_cost);
            if (current == 0x7fffffff) {
               work()->node_count = uint32_t(wide_nodes);
               work()->reserved0 = uint32_t(wide_nodes <= int(args.max_wide_nodes));
            } else {
               float leaf_cost = nvk_planner_mul(nvk_planner_mul(area, float(node.subtree_primitives)), args.sah_primitive_cost);
               if (node.child_pair < 0 || (node.subtree_primitives <= int(args.max_leaf_size) && leaf_cost <= node.subtree_cost)) {
                  node.subtree_primitives |= int32_t(0x80000000u);
                  node.subtree_cost = leaf_cost;
                  wide_nodes = 0;
               }
               store_node(current, node);
               if (node.child_pair >= 0)
                  store_temp(node.child_pair / 2, 0, wide_nodes);
            }
            __threadfence();
            if (current != 0x7fffffff && atomicAdd(&temporary()[uint32_t(current) / 2].count, 1) != 0)
               current = parent;
            else
               current = -1;
         }
      } while (__any_sync(0xffffffffu, current != -2));
      if (blockIdx.x == 0 && nvk_planner_local() == 0)
         *primitive_counter() = -1;
   }

   __device__ __forceinline__ void squeeze(uint32_t lane, volatile int32_t *inners,
                                         volatile int32_t *primitives, volatile int32_t *child_pairs,
                                         volatile int32_t *child_indices, volatile int32_t *child_counts)
   {
      if (work()->reserved0 || work()->error)
         return;
      uint32_t lower_mask = (1u << lane) - 1u;
      for (;;) {
         nvk_planner_task task{-1, 0};
         if (lane == 0) {
            uint32_t index = atomicAdd(&work()->allocated_nodes, 1u);
            if (index == 0) {
               bool needed = work()->node_count > args.max_wide_nodes;
               *primitive_counter() = needed ? int32_t(primitive_count()) : 0;
               *task_counter() = needed ? 1 : 0;
               if (needed) {
                  task = {0, int32_t(args.max_wide_nodes)};
                  work()->node_count = 0;
               }
            } else if (index < args.max_wide_nodes) {
               volatile nvk_planner_task *pending = tasks() + index;
               volatile int32_t *remaining = primitive_counter();
               while (*remaining != 0 && pending->pair == -1)
                  __nanosleep(200);
               task.pair = pending->pair;
               task.budget = pending->budget;
            }
         }
         __syncwarp();
         task.pair = __shfl_sync(0xffffffffu, task.pair, 0);
         task.budget = __shfl_sync(0xffffffffu, task.budget, 0);
         if (task.pair == -1)
            return;
         int total_primitives = int((uint32_t(load_node(task.pair).subtree_primitives) +
                                    uint32_t(load_node(task.pair + 1).subtree_primitives)) & 0x7fffffffu);
         if (total_primitives <= int(args.branching_factor)) {
            if (lane == 0) {
               atomicAdd(&work()->node_count, 1u);
               atomicAdd(primitive_counter(), -total_primitives);
            }
            continue;
         }
         int stack = 255;
         int inner_count = 0, primitive_items = 0, children = 0;
         nvk_bvh_wide_node node{};
         node.child_pair = task.pair;
         uint32_t trail = __brev(lane) >> 27;
         int index;
         do {
            index = node.child_pair + int(trail & 1u);
            node = load_node(index);
            trail >>= 1;
         } while (node.child_pair >= 0 && node.owner_subtree_primitives >= total_primitives);
         bool internal = trail == 0 && node.subtree_primitives >= 0;
         uint32_t mask = __ballot_sync(0xffffffffu, internal);
         uint32_t slot = __popc(mask & lower_mask);
         if (internal) {
            child_pairs[slot] = node.child_pair;
            child_counts[slot] = node.subtree_primitives;
            child_indices[slot] = index;
         }
         children = __popc(mask);
         __syncwarp();
         int original_budget = lane < uint32_t(children) ? load_temp(child_pairs[lane] / 2, 0) : 0;
         int budget = lane < uint32_t(children) ? min(original_budget, minimum_wide_nodes(child_counts[lane])) : 0;
         int total_budget = __reduce_add_sync(0xffffffffu, budget);
         bool needed = total_budget + 1 > task.budget;
         if (needed) {
            int collapsed_budget = budget;
            if (lane < uint32_t(children) && child_counts[lane] < int(args.max_leaf_size))
               collapsed_budget = 0;
            if (__reduce_add_sync(0xffffffffu, collapsed_budget) + 1 <= task.budget) {
               if (lane < uint32_t(children) && child_counts[lane] < int(args.max_leaf_size)) {
                  int32_t collapsed = load_node(child_indices[lane]).subtree_primitives | int32_t(0x80000000u);
                  store_shared_word(args.nodes + uint64_t(uint32_t(child_indices[lane])) * 16 + 8, collapsed);
                  store_owner(child_indices[lane], 0);
                  original_budget = budget = 0;
               }
               needed = false;
            }
         }
         if (needed) {
            children = 0;
            if (lane == 0)
               inners[254] = task.pair;
            stack = 254;
            __syncwarp();
            while (stack < 255) {
               int entries = min(255 - stack, 16);
               int pair = lane < uint32_t(entries * 2) ? inners[stack + lane / 2] : -1;
               __syncwarp();
               if (lane < uint32_t(entries * 2) && (lane & 1u) == 0)
                  inners[inner_count + lane / 2] = pair;
               stack += entries;
               inner_count += entries;
               nvk_bvh_wide_node candidate{};
               if (pair != -1)
                  candidate = load_node(pair + int(lane & 1u));
               bool primitive = pair != -1 && candidate.child_pair < 0;
               bool child = pair != -1 && !primitive && load_temp(candidate.child_pair / 2, 4) == -1;
               bool inner = pair != -1 && !primitive && !child;
               uint32_t primitive_mask = __ballot_sync(0xffffffffu, primitive);
               uint32_t child_mask = __ballot_sync(0xffffffffu, child);
               uint32_t inner_mask = __ballot_sync(0xffffffffu, inner);
               stack -= __popc(inner_mask);
               uint32_t primitive_index = uint32_t(primitive_items) + __popc(primitive_mask & lower_mask);
               uint32_t child_index = uint32_t(children) + __popc(child_mask & lower_mask);
               uint32_t inner_index = uint32_t(stack) + __popc(inner_mask & lower_mask);
               primitive_items += __popc(primitive_mask);
               children += __popc(child_mask);
               if (primitive)
                  primitives[primitive_index] = candidate.child_pair;
               if (child) {
                  child_pairs[child_index] = candidate.child_pair;
                  child_counts[child_index] = candidate.subtree_primitives & 0x7fffffff;
               }
               if (inner)
                  inners[inner_index] = candidate.child_pair;
               __syncwarp();
            }
            original_budget = lane < uint32_t(children) ? load_temp(child_pairs[lane] / 2, 0) : 0;
            budget = lane < uint32_t(children) ? min(original_budget, minimum_wide_nodes(child_counts[lane])) : 0;
         }
         total_budget = __reduce_add_sync(0xffffffffu, budget);
         int excess = task.budget - total_budget - 1;
         if (excess < 0) {
            budget = 0;
         } else {
            int prefix = original_budget - budget;
            for (uint32_t step = 1; step < 16; step *= 2) {
               int prior = __shfl_up_sync(0xffffffffu, prefix, step);
               if (lane >= step)
                  prefix += prior;
            }
            int total = max(__shfl_sync(0xffffffffu, prefix, 15), 1);
            float product = nvk_planner_mul(float(excess), float(prefix));
            int end = min(int(nvk_planner_fma(product, nvk_planner_rcp(float(total)), 0.5f)), excess);
            int begin = __shfl_sync(0xffffffffu, end, (lane - 1u) & 31u);
            budget += lane ? end - begin : end;
         }
         int child_primitives = lane < uint32_t(children) ? child_counts[lane] : 0;
         bool enqueue = lane < uint32_t(children) && budget < original_budget && child_primitives > int(args.branching_factor);
         uint32_t pending_mask = __ballot_sync(0xffffffffu, enqueue);
         uint32_t pending_base = 0;
         if (lane == 0 && pending_mask)
            pending_base = atomicAdd(task_counter(), uint32_t(__popc(pending_mask)));
         pending_base = __shfl_sync(0xffffffffu, pending_base, 0);
         uint32_t pending_index = pending_base + __popc(pending_mask & lower_mask);
         if (enqueue) {
            tasks()[pending_index].budget = budget;
            __threadfence();
            reinterpret_cast<volatile nvk_planner_task *>(tasks())[pending_index].pair = child_pairs[lane];
         }
         __syncwarp();
         __threadfence();
         int node_update = lane == 0 ? 1 : 0;
         int primitive_update = 0;
         if (!enqueue) {
            node_update += original_budget;
            primitive_update += child_primitives;
         }
         if (lane == 0)
            primitive_update += needed ? primitive_items : total_primitives;
         if (!needed)
            primitive_update -= child_primitives;
         node_update = __reduce_add_sync(0xffffffffu, node_update);
         primitive_update = __reduce_add_sync(0xffffffffu, primitive_update);
         if (lane == 0) {
            if (node_update)
               atomicAdd(&work()->node_count, uint32_t(node_update));
            if (primitive_update)
               atomicAdd(primitive_counter(), -primitive_update);
         }
         if (!needed)
            continue;
         int leaves = min(primitive_items, int(args.branching_factor) - children);
         float coefficient = nvk_planner_mul(float(primitive_items), nvk_planner_rcp(float(leaves)));
         int primitive_base = min(int(nvk_planner_fma(float(lane), coefficient, 0.5f)), primitive_items);
         int leaf_size = min(int(nvk_planner_fma(float(lane + 1), coefficient, 0.5f)), primitive_items) - primitive_base;
         int inner_prefix = max(leaf_size - 1, 0);
         for (uint32_t step = 1; step < 16; step *= 2) {
            int prior = __shfl_up_sync(0xffffffffu, inner_prefix, step);
            if (lane >= step)
               inner_prefix += prior;
         }
         int inner_base = inner_count - inner_prefix;
         for (int item = 1; item < leaf_size; item++) {
            int pair = inners[inner_base + item - 1];
            nvk_bvh_wide_node left{primitives[primitive_base], 0.0f, int32_t(uint32_t(item) | 0x80000000u), 0};
            nvk_bvh_wide_node right{primitives[primitive_base + item], 0.0f, int32_t(0x80000001u), 0};
            store_node(pair, left);
            store_node(pair + 1, right);
            primitives[primitive_base] = pair;
         }
         if (leaf_size) {
            child_pairs[children + lane] = primitives[primitive_base];
            child_counts[children + lane] = int32_t(uint32_t(leaf_size) | 0x80000000u);
         }
         children += leaves;
         __syncwarp();
         if (lane < uint32_t(children * 2 - 2)) {
            int child = int(lane) - (children - 2);
            nvk_bvh_wide_node replacement{};
            if (child < 0) {
               replacement.child_pair = inners[lane + 1];
               replacement.subtree_primitives = 0x3fffffff;
               replacement.owner_subtree_primitives = 0x7fffffff;
            } else {
               replacement.child_pair = child_pairs[child];
               replacement.subtree_primitives = child_counts[child];
               replacement.owner_subtree_primitives = 0;
            }
            store_node(inners[lane / 2] + int(lane & 1u), replacement);
         }
         __threadfence();
         __syncwarp();
      }
   }
};

}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_planner_init(nvk_bvh_planner_args args)
{
   nvk_planner planner{args};
   uint32_t threads = blockDim.x * blockDim.y * blockDim.z;
   planner.initialize(blockIdx.x * threads + nvk_planner_local());
}

extern "C" __global__ __launch_bounds__(512) void
nvk_bvh_planner_exec(nvk_bvh_planner_args args)
{
   nvk_planner planner{args};
   planner.execute(nvk_planner_local() & 31u);
}

extern "C" __global__ __launch_bounds__(64) void
nvk_bvh_planner_squeeze(nvk_bvh_planner_args args)
{
   __shared__ volatile int32_t inners[2][255];
   __shared__ volatile int32_t primitives[2][256];
   __shared__ volatile int32_t child_pairs[2][16];
   __shared__ volatile int32_t child_indices[2][16];
   __shared__ volatile int32_t child_counts[2][16];
   uint32_t warp = nvk_planner_local() / 32;
   nvk_planner planner{args};
   planner.squeeze(nvk_planner_local() & 31u, inners[warp], primitives[warp], child_pairs[warp], child_indices[warp], child_counts[warp]);
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_planner_bounds_init(nvk_bvh_planner_args args)
{
   nvk_planner planner{args};
   planner.initialize_bounds(blockIdx.x * blockDim.x + threadIdx.x);
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_planner_bounds_parents(nvk_bvh_planner_args args)
{
   nvk_planner planner{args};
   planner.link_bounds(blockIdx.x * blockDim.x + threadIdx.x);
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_planner_bounds(nvk_bvh_planner_args args)
{
   nvk_planner planner{args};
   planner.rebuild_bounds(blockIdx.x * blockDim.x + threadIdx.x);
}
