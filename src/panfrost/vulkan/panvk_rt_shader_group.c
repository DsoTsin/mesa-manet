#include "panvk_rt_shader_group.h"

#include <string.h>

uint64_t
panvk_rt_shader_address_tag(uint64_t entry, uint32_t shader_index, bool tag_index)
{
   if (!tag_index || shader_index == UINT32_MAX)
      return entry;

   return entry | ((uint64_t)shader_index << 48);
}

uint64_t
panvk_rt_shader_address_untag(uint64_t tagged_entry)
{
   return tagged_entry & PANVK_RT_SHADER_ADDRESS_MASK;
}

void
panvk_rt_shader_group_init_general(
   struct panvk_rt_shader_group_handle *handle,
   struct panvk_rt_shader_group_stack_sizes *stack_sizes, uint64_t entry,
   uint32_t pilot_offset, uint32_t stack_size)
{
   memset(handle, 0, sizeof(*handle));
   memset(stack_sizes, 0, sizeof(*stack_sizes));

   handle->general_or_intersection = entry;
   if (pilot_offset)
      handle->pilot = entry + pilot_offset;

   stack_sizes->general = stack_size;
}

void
panvk_rt_shader_group_init_hit(
   struct panvk_rt_shader_group_handle *handle,
   struct panvk_rt_shader_group_stack_sizes *stack_sizes,
   uint64_t intersection_entry, uint32_t intersection_stack_size,
   uint64_t any_hit_entry, uint32_t any_hit_stack_size,
   uint64_t closest_hit_entry, uint32_t closest_hit_stack_size)
{
   memset(handle, 0, sizeof(*handle));
   memset(stack_sizes, 0, sizeof(*stack_sizes));

   if (intersection_entry) {
      handle->general_or_intersection = intersection_entry;
      stack_sizes->intersection = intersection_stack_size;
   }

   if (any_hit_entry) {
      handle->any_hit = any_hit_entry;
      stack_sizes->any_hit = any_hit_stack_size;
   }

   if (closest_hit_entry) {
      handle->closest_hit = closest_hit_entry;
      stack_sizes->closest_hit = closest_hit_stack_size;
   }
}

uint64_t
panvk_rt_shader_group_stack_size(
   const struct panvk_rt_shader_group_stack_sizes *stack_sizes, uint32_t selector)
{
   switch (selector) {
   case PANVK_RT_SHADER_GROUP_STACK_GENERAL:
      return stack_sizes->general;
   case PANVK_RT_SHADER_GROUP_STACK_CLOSEST_HIT:
      return stack_sizes->closest_hit;
   case PANVK_RT_SHADER_GROUP_STACK_ANY_HIT:
      return stack_sizes->any_hit;
   case PANVK_RT_SHADER_GROUP_STACK_INTERSECTION:
      return stack_sizes->intersection;
   default:
      return 0;
   }
}
