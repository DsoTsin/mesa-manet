#ifndef PANVK_RT_SHADER_GROUP_H
#define PANVK_RT_SHADER_GROUP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PANVK_RT_SHADER_GROUP_HANDLE_SIZE 32
#define PANVK_RT_SHADER_ADDRESS_MASK UINT64_C(0x0000ffffffffffff)

struct panvk_rt_shader_group_handle {
   uint64_t general_or_intersection;
   uint64_t any_hit;
   uint64_t closest_hit;
   uint64_t pilot;
};

struct panvk_rt_shader_group_stack_sizes {
   uint32_t general;
   uint32_t closest_hit;
   uint32_t any_hit;
   uint32_t intersection;
};

enum panvk_rt_shader_group_stack_selector {
   PANVK_RT_SHADER_GROUP_STACK_GENERAL = 0,
   PANVK_RT_SHADER_GROUP_STACK_CLOSEST_HIT = 1,
   PANVK_RT_SHADER_GROUP_STACK_ANY_HIT = 2,
   PANVK_RT_SHADER_GROUP_STACK_INTERSECTION = 3,
};

_Static_assert(sizeof(struct panvk_rt_shader_group_handle) ==
                  PANVK_RT_SHADER_GROUP_HANDLE_SIZE,
               "ray tracing shader group handle size");
_Static_assert(offsetof(struct panvk_rt_shader_group_handle, any_hit) == 8,
               "ray tracing any-hit handle offset");
_Static_assert(offsetof(struct panvk_rt_shader_group_handle, closest_hit) == 16,
               "ray tracing closest-hit handle offset");
_Static_assert(offsetof(struct panvk_rt_shader_group_handle, pilot) == 24,
               "ray tracing pilot handle offset");

uint64_t panvk_rt_shader_address_tag(uint64_t entry, uint32_t shader_index,
                                    bool tag_index);

uint64_t panvk_rt_shader_address_untag(uint64_t tagged_entry);

void panvk_rt_shader_group_init_general(
   struct panvk_rt_shader_group_handle *handle,
   struct panvk_rt_shader_group_stack_sizes *stack_sizes, uint64_t entry,
   uint32_t pilot_offset, uint32_t stack_size);

void panvk_rt_shader_group_init_hit(
   struct panvk_rt_shader_group_handle *handle,
   struct panvk_rt_shader_group_stack_sizes *stack_sizes,
   uint64_t intersection_entry, uint32_t intersection_stack_size,
   uint64_t any_hit_entry, uint32_t any_hit_stack_size,
   uint64_t closest_hit_entry, uint32_t closest_hit_stack_size);

uint64_t panvk_rt_shader_group_stack_size(
   const struct panvk_rt_shader_group_stack_sizes *stack_sizes,
   uint32_t selector);

#endif
