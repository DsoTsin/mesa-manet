#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "gs.h"

static uint
panlib_div_round_up(uint n, uint d)
{
   return d ? (n / d) + ((n % d) != 0) : 0;
}

KERNEL(1)
panlib_gs_setup_indirect(global struct poly_vertex_params *vp,
                         global struct poly_geometry_params *p,
                         global struct panvk_gs_raster_params *rp,
                         global uint32_t *grids,
                         global struct poly_heap *heap,
                         global const uint32_t *indirect,
                         global const uint32_t *draw_count_buffer,
                         uint32_t draw_index,
                         uint64_t in_index_buffer,
                         uint32_t in_index_buffer_range_el,
                         uint32_t in_index_size_B,
                         uint32_t prim,
                         uint32_t shape,
                         uint32_t max_indices,
                         uint32_t vertex_stride,
                         uint32_t varying_size_B,
                         uint32_t dynamic_vertices,
                         uint32_t clip,
                         uint32_t vs_wg_size_x,
                         uint32_t vs_wg_size_y,
                         uint32_t gs_wg_size_x,
                         global uint32_t *first_vertex_sysval,
                         global uint32_t *base_instance_sysval)
{
   const bool active =
      !draw_count_buffer || draw_index < *draw_count_buffer;
   const uint count = active ? indirect[0] : 0;
   const uint instance_count = active ? indirect[1] : 0;
   const uint first = active ? indirect[2] : 0;
   const uint prims = u_decomposed_prims_for_vertices(prim, count);
   const uint input_prims = prims * instance_count;
   const uint vertex_count = (input_prims * vertex_stride + 3) & ~3u;
   const uint index_count = input_prims * max_indices;

   poly_vertex_params_set_draw(vp, count, instance_count);
   vp->index_size_B = in_index_size_B;
   if (in_index_size_B) {
      vp->index_buffer =
         poly_index_buffer(in_index_buffer, in_index_buffer_range_el, first,
                           in_index_size_B);
      vp->index_buffer_range_el =
         poly_index_buffer_range_el(in_index_buffer_range_el, first);
   }

   poly_geometry_params_set_draw(p, prim, shape, max_indices, count,
                                 instance_count);

   global uchar *counts =
      poly_heap_alloc(heap, dynamic_vertices ? input_prims * 4 : 0);
   global uchar *vs_out =
      poly_heap_alloc(heap, poly_tcs_in_size(count * instance_count,
                                             vp->outputs));
   global uchar *pos = poly_heap_alloc(heap, vertex_count * 16);
   global uchar *clip_pos =
      clip ? poly_heap_alloc(heap, vertex_count * 16) : 0;
   global uchar *vary = poly_heap_alloc(heap, vertex_count * varying_size_B);
   global uchar *idx = poly_heap_alloc(heap, index_count * 4);

   vp->output_buffer = (uintptr_t)vs_out;
   p->count_buffer = dynamic_vertices ? (global uint *)counts : 0;
   p->output_index_buffer = (global uint *)idx;

   rp->index_buffer = (uintptr_t)idx;
   rp->position_buffer = (uintptr_t)pos;
   rp->varying_buffer = (uintptr_t)vary;
   rp->clip_position_buffer = (uintptr_t)clip_pos;
   rp->index_count = vertex_count ? index_count : 0;

   if (first_vertex_sysval)
      *first_vertex_sysval = active ? indirect[in_index_size_B ? 3 : 2] : 0;

   if (base_instance_sysval)
      *base_instance_sysval = active ? indirect[in_index_size_B ? 4 : 3] : 0;

   grids[0] = panlib_div_round_up(count, vs_wg_size_x);
   grids[1] = panlib_div_round_up(instance_count, vs_wg_size_y);
   grids[2] = 1;
   grids[3] = panlib_div_round_up(prims, gs_wg_size_x);
   grids[4] = instance_count;
   grids[5] = 1;
}

KERNEL(256)
panlib_prefix_sum_gs_params(constant struct poly_geometry_params *p)
{
   local uint scratch[32];

   poly_prefix_sum(scratch, p->count_buffer, p->input_primitives, 1, 0, 256);
}
