#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/prim.h"
#include "genxml/gen_macros.h"
#include "xfb.h"

static uint32_t
xfb_index(global const struct panlib_xfb_params *p, uint32_t element)
{
   uint64_t id = (uint64_t)p->first_index + element;
   if (id >= p->index_range)
      return 0;
   uint64_t address = p->index_buffer + id * p->index_size;
   if (p->index_size == 1)
      return *(global const uint8_t *)address;
   if (p->index_size == 2)
      return *(global const uint16_t *)address;
   return *(global const uint32_t *)address;
}

static uint32_t
xfb_restart_index(uint32_t size)
{
   return size == 4 ? UINT32_MAX : (1u << (size * 8)) - 1;
}

uint32_t
GENX(panlib_xfb_vertex_id)(global const struct panlib_xfb_params *p,
                          uint32_t captured)
{
   enum mesa_prim mode = p->topology;
   uint64_t per_instance =
      (uint64_t)p->primitives_per_instance * p->vertices_per_primitive;
   uint32_t id = captured % per_instance;
   uint32_t primitive = id / p->vertices_per_primitive;
   uint32_t vertex = id % p->vertices_per_primitive;
   if (mode == MESA_PRIM_LINES_ADJACENCY ||
       mode == MESA_PRIM_LINE_STRIP_ADJACENCY)
      vertex++;
   if (mode == MESA_PRIM_TRIANGLES_ADJACENCY ||
       mode == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY)
      vertex *= 2;

   uint32_t segment_start = 0;
   uint32_t segment_primitives = p->primitives_per_instance;
   if (p->restart && p->index_size) {
      uint32_t restart = xfb_restart_index(p->index_size);
      for (uint64_t end = 0; end <= p->vertex_count; end++) {
         if (end != p->vertex_count && xfb_index(p, end) != restart)
            continue;
         segment_primitives =
            u_decomposed_prims_for_vertices(mode, end - segment_start);
         if (primitive < segment_primitives)
            break;
         primitive -= segment_primitives;
         segment_start = end + 1;
      }
   }

   uint32_t element = segment_start +
      poly_vertex_id_for_topology(mode, p->provoking_first, primitive,
                                   vertex, segment_primitives);
   uint32_t index = p->index_size ? xfb_index(p, element) : element;
   return index + p->first_vertex;
}

KERNEL(1)
panlib_xfb_setup(global struct panlib_xfb_params *p,
                 global const uint32_t *commands,
                 global const uint32_t *draw_count_buffer,
                 uint32_t draw_id, uint32_t max_draw_count)
{
   if (commands) {
      uint32_t draws = draw_count_buffer
                         ? min(*draw_count_buffer, max_draw_count)
                         : max_draw_count;
      p->vertex_count = draw_id < draws ? commands[0] : 0;
      p->instance_count = draw_id < draws ? commands[1] : 0;
      p->first_index = p->index_size ? commands[2] : 0;
      p->first_vertex = p->index_size ? commands[3] : commands[2];
      p->first_instance = commands[p->index_size ? 4 : 3];
      p->draw_id = draw_id;
   }

   enum mesa_prim mode = p->topology;
   uint32_t count = u_decomposed_prims_for_vertices(mode, p->vertex_count);
   if (p->restart && p->index_size) {
      count = 0;
      uint32_t start = 0;
      uint32_t restart = xfb_restart_index(p->index_size);
      for (uint64_t end = 0; end <= p->vertex_count; end++) {
         if (end != p->vertex_count && xfb_index(p, end) != restart)
            continue;
         count += u_decomposed_prims_for_vertices(mode, end - start);
         start = end + 1;
      }
   }

   p->primitives_per_instance = count;
   p->vertices_per_primitive = mesa_vertices_per_prim(u_reduced_prim(mode));
   uint64_t needed = (uint64_t)count * p->instance_count;
   uint64_t written = needed;
   global uint32_t *offsets = (global uint32_t *)p->offsets;

   for (uint32_t i = 0; i < 4; i++) {
      p->start[i] = offsets[i];
      if (!p->stride[i])
         continue;
      uint64_t last = (uint64_t)offsets[i] + p->output_end[i] +
                      (p->vertices_per_primitive - 1) * p->stride[i];
      uint64_t capacity = p->size[i] < last ? 0 :
         1 + (p->size[i] - last) /
             ((uint64_t)p->vertices_per_primitive * p->stride[i]);
      written = min(written, capacity);
   }

   p->captured_vertices = written * p->vertices_per_primitive;
   p->workgroups[0] = ((uint64_t)p->captured_vertices + 63) / 64;
   p->workgroups[1] = 1;
   p->workgroups[2] = 1;
   for (uint32_t i = 0; i < 4; i++) {
      if (p->stride[i])
         offsets[i] += p->captured_vertices * p->stride[i];
   }
   if (p->query) {
      global uint64_t *reports = (global uint64_t *)p->query;
      reports[0] += written;
      reports[1] += needed;
   }
}
