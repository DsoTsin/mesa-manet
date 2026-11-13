#include "nvk_nir_ser_queue.h"
#include "nvk_ser.h"
#include "nir_builder.h"

static nir_def *
ser_load(nir_builder *b, nir_def *address, unsigned offset, unsigned bits)
{
   return nir_load_global(b, 1, bits, nir_iadd_imm(b, address, offset),
      .align_mul = bits / 8, .access = ACCESS_COHERENT);
}

static void
ser_fence(nir_builder *b)
{
   nir_barrier(b, SCOPE_NONE, SCOPE_DEVICE, NIR_MEMORY_ACQ_REL,
      nir_var_mem_global);
}

static nir_def *
ser_atomic_add(nir_builder *b, nir_def *counters, unsigned offset, int value)
{
   return nir_global_atomic(b, 32, nir_iadd_imm(b, counters, offset),
      nir_imm_int(b, value), .atomic_op = nir_atomic_op_iadd,
      .access = ACCESS_COHERENT);
}

static void
ser_schedule(nir_builder *b, nir_def *queue, nir_def *counters)
{
   ser_fence(b);
   nir_def *remaining = ser_atomic_add(b, counters,
      offsetof(struct nvk_ser_counters, remaining), -1);
   nir_push_if(b, nir_ieq_imm(b, remaining, 1));
   nir_def *alive = ser_atomic_add(b, counters,
      offsetof(struct nvk_ser_counters, alive), 0);
   nir_def *prior = nir_global_atomic_swap(b, 32,
      nir_iadd_imm(b, counters, offsetof(struct nvk_ser_counters, scheduled)),
      nir_imm_int(b, 0), nir_imm_int(b, 1),
      .atomic_op = nir_atomic_op_cmpxchg, .access = ACCESS_COHERENT);
   nir_push_if(b, nir_ieq_imm(b, prior, 0));
   nir_def *sort = ser_load(b, queue,
      offsetof(struct nvk_ser_queue, native.sort_qmd), 32);
   nir_def *tail = ser_load(b, queue,
      offsetof(struct nvk_ser_queue, native.tail_qmd), 32);
   nir_def *qmd = nir_bcsel(b, nir_ieq_imm(b, alive, 0), tail, sort);
   nir_def *sked = ser_load(b, queue,
      offsetof(struct nvk_ser_queue, native.reflected_sked), 64);
   nir_def *flags = ser_load(b, queue, offsetof(struct nvk_ser_queue, flags), 32);
   qmd = nir_bcsel(b, nir_ine_imm(b,
      nir_iand_imm(b, flags, NVK_SER_DEBUG_PCAS_TAIL), 0), tail, qmd);
   nir_push_if(b, nir_iand(b, nir_ine_imm(b,
      nir_iand_imm(b, flags, NVK_SER_DEBUG_STOP_BEFORE_PCAS), 0),
      nir_ine_imm(b, alive, 0)));
   nir_store_global(b, nir_imm_int(b, 1), nir_iadd_imm(b, counters,
      offsetof(struct nvk_ser_counters, completed)), .align_mul = 4,
      .access = ACCESS_COHERENT);
   nir_push_else(b, NULL);
   nir_def *packet = nir_pack_64_2x32_split(b, qmd, nir_imm_int(b, 3));
   ser_fence(b);
   nir_global_atomic(b, 64, sked, packet,
      .atomic_op = nir_atomic_op_iadd, .access = ACCESS_COHERENT);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static void
ser_suspend(nir_builder *b, nir_def *queue, nir_def *owner, void *data)
{
   (void)owner;
   (void)data;
   nir_def *counters = ser_load(b, queue, offsetof(struct nvk_ser_queue, counters), 64);
   ser_fence(b);
   ser_atomic_add(b, counters, offsetof(struct nvk_ser_counters, arrived), 1);
   ser_schedule(b, queue, counters);
}

static void
ser_complete(nir_builder *b, nir_def *queue, nir_def *owner, void *data)
{
   (void)data;
   nir_def *packets = ser_load(b, queue, offsetof(struct nvk_ser_queue, packets), 64);
   nir_def *packet = nir_iadd(b, packets,
      nir_imul_imm(b, nir_u2u64(b, owner), sizeof(struct nvk_ser_packet)));
   nir_store_global(b, nir_imm_int(b, NVK_SER_INVALID_OWNER), packet,
      .align_mul = 4, .access = ACCESS_COHERENT);
   nir_def *counters = ser_load(b, queue, offsetof(struct nvk_ser_queue, counters), 64);
   ser_fence(b);
   ser_atomic_add(b, counters, offsetof(struct nvk_ser_counters, alive), -1);
   ser_schedule(b, queue, counters);
}

const struct nvk_nir_rt_ser_options nvk_nir_ser_queue_options = {
   .emit_suspend = ser_suspend,
   .emit_complete = ser_complete,
};
