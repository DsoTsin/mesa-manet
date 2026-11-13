#include "agc_cb_builders.h"
#include "agc_driver.h"
#include "agc_exports.h"
#include "sid.h"

#define AGC_DRAW_MODIFIER_MASK UINT64_C(0x1ffffffff)
#define AGC_DRAW_VERTEX_OFFSET (UINT64_C(1) << 0)
#define AGC_DRAW_START_INDEX (UINT64_C(1) << 1)
#define AGC_DRAW_INSTANCE_OFFSET (UINT64_C(1) << 2)
#define AGC_DRAW_DRAW_INDEX (UINT64_C(1) << 3)
#define AGC_DRAW_USER_VGPRS (UINT64_C(1) << 4)
#define AGC_DRAW_VERTEX_OFFSET_SHIFT 9
#define AGC_DRAW_START_INDEX_SHIFT 14
#define AGC_DRAW_INSTANCE_OFFSET_SHIFT 19
#define AGC_DRAW_DRAW_INDEX_SHIFT 24
#define AGC_DRAW_STAGE_SHIFT 29
#define AGC_GUEST_SH_USER_DATA_GS_0 0x08cu
#define AGC_GUEST_SH_USER_DATA_HS_0 0x10cu
#define AGC_INDEX_SIZE_MAX 2u
#define AGC_UCONFIG_INDEX_TYPE 2u
#define AGC_PREDICATION_OP_MAX 4u

static inline bool
draw_modifier_valid(uint64_t bits)
{
   return (bits & ~AGC_DRAW_MODIFIER_MASK) == 0;
}

static inline uint32_t
draw_initiator(bool indexed)
{
   return S_0287F0_SOURCE_SELECT(indexed ? V_0287F0_DI_SRC_SEL_DMA
                                         : V_0287F0_DI_SRC_SEL_AUTO_INDEX);
}

static uint32_t
draw_location(uint64_t bits, uint64_t enable, uint32_t shift, const char *function)
{
   if (!(bits & enable))
      return 0;
   uint32_t stage = (uint32_t)(bits >> AGC_DRAW_STAGE_SHIFT) & 7;
   uint32_t base = stage == 3 || stage == 5 ? AGC_GUEST_SH_USER_DATA_HS_0
                                            : AGC_GUEST_SH_USER_DATA_GS_0;
   uint32_t offset = base + ((uint32_t)(bits >> shift) & 0x1f);
   const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, AGC_REG_BANK_SH, offset);
   enum agc_reg_class cls = entry ? (enum agc_reg_class)entry->cls : AGC_REG_UNDEFINED;
   if (cls != AGC_REG_PASS && cls != AGC_REG_MOVE && cls != AGC_REG_ADDR) {
      agc_fatal(function, "draw parameter register sh:0x%03x has no host location (%s)", offset,
                agc_reg_class_name(cls));
   }
   return entry->host_offset;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndexAuto(struct agc_cb *cb, uint32_t count, uint64_t bits)
{
   if (AGC_UNLIKELY(!cb || !draw_modifier_valid(bits))) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDEX_AUTO);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DRAW_INDEX_AUTO, 1, 0);
   out[1] = count;
   out[2] = draw_initiator(false);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexAutoGetSize(void)
{
   return AGC_DWORDS_DRAW_INDEX_AUTO * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndex(struct agc_cb *cb, uint32_t count, uint64_t address, uint64_t bits)
{
   if (AGC_UNLIKELY(!cb || !address || !draw_modifier_valid(bits))) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDEX);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DRAW_INDEX_2, 4, 0);
   out[1] = count ? count : 1;
   out[2] = (uint32_t)address;
   out[3] = (uint32_t)(address >> 32);
   out[4] = count;
   out[5] = draw_initiator(true);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexGetSize(void)
{
   return AGC_DWORDS_DRAW_INDEX * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndexOffset(struct agc_cb *cb, uint32_t offset, uint32_t count, uint64_t bits)
{
   if (AGC_UNLIKELY(!cb || !draw_modifier_valid(bits))) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDEX_OFFSET);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t end = count > UINT32_MAX - offset ? UINT32_MAX : offset + count;
   out[0] = PKT3(PKT3_DRAW_INDEX_OFFSET_2, 3, 0);
   out[1] = end ? end : 1;
   out[2] = offset;
   out[3] = count;
   out[4] = draw_initiator(true);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexOffsetGetSize(void)
{
   return AGC_DWORDS_DRAW_INDEX_OFFSET * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndirect(struct agc_cb *cb, uint32_t offset, uint64_t bits)
{
   if (AGC_UNLIKELY(!cb || (offset & 3) || !draw_modifier_valid(bits))) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   uint32_t vertex = draw_location(bits, AGC_DRAW_VERTEX_OFFSET, AGC_DRAW_VERTEX_OFFSET_SHIFT,
                                   __func__);
   uint32_t instance = draw_location(bits, AGC_DRAW_INSTANCE_OFFSET,
                                     AGC_DRAW_INSTANCE_OFFSET_SHIFT, __func__);
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDIRECT);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DRAW_INDIRECT, 3, 0);
   out[1] = offset;
   out[2] = S_242_START_VTX_LOC(vertex);
   out[3] = S_243_START_INST_LOC(instance);
   out[4] = draw_initiator(false);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndirectGetSize(void)
{
   return AGC_DWORDS_DRAW_INDIRECT * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndexIndirect(struct agc_cb *cb, uint32_t offset, uint64_t bits)
{
   if (AGC_UNLIKELY(!cb || (offset & 3) || !draw_modifier_valid(bits))) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   uint32_t vertex = draw_location(bits, AGC_DRAW_VERTEX_OFFSET, AGC_DRAW_VERTEX_OFFSET_SHIFT,
                                   __func__);
   uint32_t start_index = draw_location(bits, AGC_DRAW_START_INDEX, AGC_DRAW_START_INDEX_SHIFT,
                                        __func__);
   uint32_t instance = draw_location(bits, AGC_DRAW_INSTANCE_OFFSET,
                                     AGC_DRAW_INSTANCE_OFFSET_SHIFT, __func__);
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDIRECT);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DRAW_INDEX_INDIRECT, 3, 0);
   out[1] = offset;
   out[2] = S_252_BASE_VTX_LOC(vertex) | S_252_START_INDX_LOC(start_index);
   out[3] = S_253_START_INST_LOC(instance) |
            S_253_START_INDX_ENABLE((bits & AGC_DRAW_START_INDEX) != 0);
   out[4] = draw_initiator(true);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexIndirectGetSize(void)
{
   return AGC_DWORDS_DRAW_INDIRECT * 4;
}

static uint32_t *
draw_indirect_multi(struct agc_cb *cb, bool indexed, uint32_t offset, uint32_t count_indirect,
                    uint32_t count, uint64_t count_address, uint32_t stride, uint64_t bits,
                    const char *function)
{
   bool valid = cb && count_indirect <= 1 && (offset & 3) == 0 && (stride & 3) == 0 &&
                stride >= (indexed ? 20u : 16u) && draw_modifier_valid(bits);
   if (valid)
      valid = count_indirect ? agc_address_valid(count_address, 4) : count_address == 0;
   if (AGC_UNLIKELY(!valid)) {
      agc_report(function, "invalid draw arguments");
      return NULL;
   }
   uint32_t vertex = draw_location(bits, AGC_DRAW_VERTEX_OFFSET, AGC_DRAW_VERTEX_OFFSET_SHIFT,
                                   function);
   uint32_t start_index = indexed ? draw_location(bits, AGC_DRAW_START_INDEX,
                                                  AGC_DRAW_START_INDEX_SHIFT, function)
                                  : 0;
   uint32_t instance = draw_location(bits, AGC_DRAW_INSTANCE_OFFSET,
                                     AGC_DRAW_INSTANCE_OFFSET_SHIFT, function);
   uint32_t draw_index = draw_location(bits, AGC_DRAW_DRAW_INDEX, AGC_DRAW_DRAW_INDEX_SHIFT,
                                       function);
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DRAW_INDIRECT_MULTI);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t control = S_2C4_DRAW_INDEX_LOC(draw_index) |
                      S_2C4_USE_VGPRS((bits & AGC_DRAW_USER_VGPRS) != 0) |
                      S_2C4_COUNT_INDIRECT_ENABLE(count_indirect) |
                      S_2C4_DRAW_INDEX_ENABLE((bits & AGC_DRAW_DRAW_INDEX) != 0);
   if (indexed) {
      out[0] = PKT3(PKT3_DRAW_INDEX_INDIRECT_MULTI, 8, 0);
      out[2] = S_382_BASE_VTX_LOC(vertex) | S_382_START_INDX_LOC(start_index);
      control |= S_384_START_INDX_ENABLE((bits & AGC_DRAW_START_INDEX) != 0);
   } else {
      out[0] = PKT3(PKT3_DRAW_INDIRECT_MULTI, 8, 0);
      out[2] = S_2C2_START_VTX_LOC(vertex);
   }
   out[1] = offset;
   out[3] = S_2C3_START_INST_LOC(instance);
   out[4] = control;
   out[5] = count;
   out[6] = (uint32_t)count_address;
   out[7] = (uint32_t)(count_address >> 32);
   out[8] = stride;
   out[9] = draw_initiator(indexed);
   agc_fill_nops(out + 10, AGC_DWORDS_DRAW_INDIRECT_MULTI - 10);
   return out;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndirectMulti(struct agc_cb *cb, uint32_t offset, uint32_t count_indirect,
                           uint32_t count, uint64_t count_address, uint32_t stride, uint64_t bits)
{
   return draw_indirect_multi(cb, false, offset, count_indirect, count, count_address, stride, bits,
                              __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndexIndirectMulti(struct agc_cb *cb, uint32_t offset, uint32_t count_indirect,
                                uint32_t count, uint64_t count_address, uint32_t stride,
                                uint64_t bits)
{
   return draw_indirect_multi(cb, true, offset, count_indirect, count, count_address, stride, bits,
                              __func__);
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndirectMultiGetSize(void)
{
   return AGC_DWORDS_DRAW_INDIRECT_MULTI * 4;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexIndirectMultiGetSize(void)
{
   return AGC_DWORDS_DRAW_INDIRECT_MULTI * 4;
}

uint32_t AGC_SYSV
sceAgcDcbDrawIndexMultiInstancedGetSize(void)
{
   return AGC_DWORDS_DRAW_INDEX_MULTI_INSTANCED * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDispatchIndirect(struct agc_cb *cb, uint32_t offset, uint32_t modifier)
{
   if (!cb || !agc_dispatch_modifier_valid(modifier)) {
      agc_report(__func__, "invalid dispatch arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DISPATCH_INDIRECT_DCB);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DISPATCH_INDIRECT, 1, 0) | PKT3_SHADER_TYPE_S(1);
   out[1] = offset;
   out[2] = agc_dispatch_initiator(modifier);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbDispatchIndirectGetSize(void)
{
   return AGC_DWORDS_DISPATCH_INDIRECT_DCB * 4;
}

static uint32_t *
set_index_size(struct agc_cb *cb, uint8_t size, uint8_t cache_policy, uint8_t object_id,
               const char *function)
{
   if (!cb || size > AGC_INDEX_SIZE_MAX || cache_policy > 3 || object_id > 1) {
      agc_report(function, "invalid index size arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_INDEX_SIZE);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_SET_UCONFIG_REG_INDEX, 1, 0);
   out[1] = ((R_03090C_VGT_INDEX_TYPE - CIK_UCONFIG_REG_OFFSET) >> 2) |
            (AGC_UCONFIG_INDEX_TYPE << 28);
   out[2] = S_028A7C_INDEX_TYPE(size) | S_028A7C_DISABLE_INSTANCE_PACKING(object_id);
   agc_record_note_draw_state(out, AGC_DRAW_STATE_INDEX_TYPE);
   return out;
}

uint32_t *AGC_SYSV
sceAgcDcbSetIndexSize(struct agc_cb *cb, uint8_t size, uint8_t cache_policy)
{
   return set_index_size(cb, size, cache_policy, 0, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetIndexSize_0600(struct agc_cb *cb, uint8_t size, uint8_t cache_policy, uint8_t object_id)
{
   return set_index_size(cb, size, cache_policy, object_id, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbSetIndexSizeGetSize(void)
{
   return AGC_DWORDS_INDEX_SIZE * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetIndexCount(struct agc_cb *cb, uint32_t count)
{
   uint32_t *out = agc_packet_alloc(cb, AGC_DWORDS_INDEX_COUNT, __func__);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_INDEX_BUFFER_SIZE, 0, 0);
   out[1] = count;
   agc_record_note_draw_state(out, AGC_DRAW_STATE_INDEX_SIZE);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetIndexCountGetSize(void)
{
   return AGC_DWORDS_INDEX_COUNT * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetIndexBuffer(struct agc_cb *cb, uint64_t address)
{
   if (!cb || (address & 1)) {
      agc_report(__func__, "null command buffer or misaligned index buffer");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_INDEX_BASE);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_INDEX_BASE, 1, 0);
   out[1] = (uint32_t)address;
   out[2] = (uint32_t)(address >> 32);
   agc_record_note_draw_state(out, AGC_DRAW_STATE_INDEX_BASE);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetIndexBufferGetSize(void)
{
   return AGC_DWORDS_INDEX_BASE * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetIndexIndirectArgs(struct agc_cb *cb, uint64_t address, uint32_t index)
{
   if (!cb || (address & 0xf) || index > 0xffff) {
      agc_report(__func__, "invalid index attribute arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_INDEX_INDIRECT_ARGS);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_INDEX_ATTRIBUTES_INDIRECT, 2, 0);
   out[1] = (uint32_t)address;
   out[2] = (uint32_t)(address >> 32);
   out[3] = S_913_ATTRIBUTE_INDEX(index);
   agc_record_note_draw_state(out, AGC_DRAW_STATE_INDEX_ATTRIBUTES);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetIndexIndirectArgsGetSize(void)
{
   return AGC_DWORDS_INDEX_INDIRECT_ARGS * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetNumInstances(struct agc_cb *cb, uint32_t count)
{
   uint32_t *out = agc_packet_alloc(cb, AGC_DWORDS_NUM_INSTANCES, __func__);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_NUM_INSTANCES, 0, 0);
   out[1] = count;
   agc_record_note_draw_state(out, AGC_DRAW_STATE_NUM_INSTANCES);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetNumInstancesGetSize(void)
{
   return AGC_DWORDS_NUM_INSTANCES * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetBaseIndirectArgs(struct agc_cb *cb, uint32_t shader_type, uint64_t address)
{
   if (!cb || shader_type > 1 || (address & 7) || address > AGC_ADDRESS_MAX) {
      agc_report(__func__, "invalid indirect base arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_SET_BASE);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_SET_BASE, 2, 0) | PKT3_SHADER_TYPE_S(shader_type);
   out[1] = S_111_BASE_INDEX(V_111_PATCH_TABLE_BASE);
   out[2] = (uint32_t)address;
   out[3] = (uint32_t)(address >> 32);
   agc_record_note_draw_state(out, shader_type ? AGC_DRAW_STATE_DISPATCH_BASE : AGC_DRAW_STATE_DRAW_BASE);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetBaseDispatchIndirectArgsGetSize(void)
{
   return AGC_DWORDS_SET_BASE * 4;
}

uint32_t AGC_SYSV
sceAgcDcbSetBaseDrawIndirectArgsGetSize(void)
{
   return AGC_DWORDS_SET_BASE * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetPredication(struct agc_cb *cb, uint8_t condition, uint8_t operation, uint8_t wait,
                        uint64_t address, uint32_t count)
{
   (void)count;
   if (!cb || condition > 1 || operation > AGC_PREDICATION_OP_MAX || wait > 1 ||
       (operation != V_201_CLEAR_PREDICATE && !agc_address_valid(address, 8))) {
      agc_report(__func__, "invalid predication arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_PREDICATION);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_SET_PREDICATION, 2, 0);
   if (operation == V_201_CLEAR_PREDICATE) {
      out[1] = S_201_PRED_OP(V_201_CLEAR_PREDICATE);
      out[2] = 0;
      out[3] = 0;
      return out;
   }
   bool occlusion = operation == V_201_SET_ZPASS_PREDICATE ||
                    operation == V_201_SET_PRIMCOUNT_PREDICATE;
   out[1] = S_201_PRED_BOOL(condition) | S_201_HINT(occlusion && wait) | S_201_PRED_OP(operation);
   out[2] = (uint32_t)address & ~UINT32_C(0xf);
   out[3] = (uint32_t)(address >> 32);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbSetZPassPredicationEnableGetSize(void)
{
   return AGC_DWORDS_PREDICATION * 4;
}

uint32_t AGC_SYSV
sceAgcDcbSetBoolPredicationEnableGetSize(void)
{
   return AGC_DWORDS_PREDICATION * 4;
}

uint32_t AGC_SYSV
sceAgcDcbSetPredicationDisableGetSize(void)
{
   return AGC_DWORDS_PREDICATION * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbStallCommandBufferParser(struct agc_cb *cb)
{
   uint32_t *out = agc_packet_alloc(cb, AGC_DWORDS_STALL_PARSER, __func__);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_PFP_SYNC_ME, 0, 0);
   out[1] = 0;
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbStallCommandBufferParserGetSize(void)
{
   return AGC_DWORDS_STALL_PARSER * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbWriteData(struct agc_cb *cb, uint8_t dst, uint8_t cache_policy, uint64_t address,
                   const uint32_t *data, uint32_t count, uint8_t increment, uint8_t confirm)
{
   return agc_emit_write_data(cb, false, dst, cache_policy, address, data, count, increment,
                              confirm, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbWriteDataGetSize(uint32_t count)
{
   return (count + AGC_DWORDS_WRITE_DATA_HEAD) * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbCopyData(struct agc_cb *cb, uint8_t dst, uint8_t dst_policy, uint64_t destination,
                  uint8_t src, uint8_t src_policy, uint64_t source, uint8_t item, uint8_t confirm)
{
   return agc_emit_copy_data(cb, false, dst, dst_policy, destination, src, src_policy, source, item,
                             confirm, __func__);
}

uint64_t AGC_SYSV
sceAgcDcbCopyDataGetSize(void)
{
   return AGC_DWORDS_COPY_DATA * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDmaData(struct agc_cb *cb, uint8_t engine, uint8_t dst, uint8_t dst_policy,
                 uint64_t destination, uint8_t src, uint8_t src_policy, uint64_t source,
                 uint32_t bytes, uint8_t wait, uint8_t confirm, uint8_t block)
{
   return agc_emit_dma_data(cb, false, engine, dst, dst_policy, destination, src, src_policy,
                            source, bytes, wait, confirm, block, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbDmaDataGetSize(void)
{
   return AGC_DWORDS_DMA_DATA * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbAtomicMem(struct agc_cb *cb, uint8_t engine, uint8_t op, uint8_t command,
                   uint8_t cache_policy, uint64_t address, uint64_t source, uint64_t compare,
                   uint32_t interval)
{
   return agc_emit_atomic_mem(cb, false, engine, op, command, cache_policy, address, source,
                              compare, interval, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbAtomicMemGetSize(void)
{
   return AGC_DWORDS_ATOMIC_MEM * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbWaitRegMem(struct agc_cb *cb, uint8_t size, uint8_t function, uint8_t operation,
                    uint8_t cache_policy, uint64_t address, uint64_t reference, uint64_t mask,
                    uint32_t poll)
{
   return agc_emit_wait_reg_mem(cb, false, size, function, operation, cache_policy, address,
                                reference, mask, poll, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbWaitOnAddressGetSize(uint32_t size)
{
   return agc_wait_dwords(size) * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbAcquireMem(struct agc_cb *cb, uint8_t engine, uint32_t cb_db, uint32_t gcr, uint64_t base,
                    uint64_t size, uint32_t poll)
{
   return agc_emit_acquire_mem(cb, false, engine, cb_db, gcr, base, size, poll, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbAcquireMemGetSize(void)
{
   return AGC_DWORDS_ACQUIRE_MEM * 4;
}

uint32_t AGC_SYSV
sceAgcDcbQueueEndOfShaderActionGetSize(void)
{
   return AGC_DWORDS_RELEASE_MEM * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbEventWrite(struct agc_cb *cb, uint8_t event, uint64_t address)
{
   return agc_emit_event_write(cb, false, event, address, __func__);
}

uint64_t AGC_SYSV
sceAgcDcbEventWriteGetSize(uint8_t event)
{
   return agc_event_write_host_dwords(false, event) * 4;
}

uint32_t AGC_SYSV
sceAgcDcbBeginOcclusionQueryGetSize(int32_t query_type)
{
   uint32_t dump = AGC_DWORDS_EVENT_WRITE_ADDRESS * 4;
   if (query_type == 0)
      return dump;
   if (query_type == 1)
      return (AGC_DWORDS_OCCLUSION_RESULTS + AGC_DWORDS_WRITE_DATA_HEAD) * 4 + dump;
   return 0;
}

uint32_t AGC_SYSV
sceAgcDcbEndOcclusionQueryGetSize(void)
{
   return AGC_DWORDS_EVENT_WRITE_ADDRESS * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbPrimeUtcl2(struct agc_cb *cb, uint32_t permission, uint32_t mode, uint64_t address,
                    uint32_t pages)
{
   (void)permission;
   (void)mode;
   (void)address;
   (void)pages;
   return agc_emit_nops(cb, AGC_DWORDS_PRIME_UTCL2, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbPrimeUtcl2GetSize(void)
{
   return AGC_DWORDS_PRIME_UTCL2 * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbJump(struct agc_cb *cb, uint8_t mode, uint8_t cache_policy, uint64_t target,
              uint32_t dwords)
{
   return agc_emit_jump(cb, false, mode, cache_policy, target, dwords, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbJumpGetSize(void)
{
   return AGC_DWORDS_JUMP * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbCondExec(struct agc_cb *cb, uint64_t address, uint32_t dwords)
{
   return agc_emit_cond_exec(cb, address, dwords, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbCondExecGetSize(void)
{
   return AGC_DWORDS_COND_EXEC * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbRewind(struct agc_cb *cb, uint32_t state)
{
   return agc_emit_rewind(cb, state, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbRewindGetSize(void)
{
   return AGC_DWORDS_REWIND * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbSetCxRegisterDirect(struct agc_cb *cb, struct agc_register reg)
{
   return agc_emit_set_register_range(cb, AGC_REG_BANK_CX, reg.offset, &reg.value, 1, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetShRegisterDirect(struct agc_cb *cb, struct agc_register reg)
{
   return agc_emit_set_register_range(cb, AGC_REG_BANK_SH, reg.offset, &reg.value, 1, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetUcRegisterDirect(struct agc_cb *cb, struct agc_register reg)
{
   return agc_emit_set_register_range(cb, AGC_REG_BANK_UC, reg.offset, &reg.value, 1, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbSetCxRegisterDirectGetSize(void)
{
   return agc_set_registers_bytes(1);
}

uint32_t AGC_SYSV
sceAgcDcbSetShRegisterDirectGetSize(void)
{
   return agc_set_registers_bytes(1);
}

uint32_t AGC_SYSV
sceAgcDcbSetUcRegisterDirectGetSize(void)
{
   return agc_set_registers_bytes(1);
}

uint32_t *AGC_SYSV
sceAgcDcbPushMarker(struct agc_cb *cb, const char *text, uint32_t color)
{
   (void)color;
   return agc_emit_marker(cb, text, false, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbPushMarkerSpan(struct agc_cb *cb, const char *text, uint32_t length, uint32_t color)
{
   (void)color;
   return agc_emit_marker_span(cb, text, length, false, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetMarker(struct agc_cb *cb, const char *text, uint32_t color)
{
   (void)color;
   return agc_emit_marker(cb, text, true, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetMarkerSpan(struct agc_cb *cb, const char *text, uint32_t length, uint32_t color)
{
   (void)color;
   return agc_emit_marker_span(cb, text, length, true, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbPopMarker(struct agc_cb *cb)
{
   return agc_emit_nops(cb, AGC_DWORDS_MARKER_POP, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetFlip(struct agc_cb *cb, uint32_t video_out, int32_t index, uint32_t mode,
                 int64_t argument)
{
   return agc_emit_flip(cb, video_out, index, mode, argument, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbWaitUntilSafeForRendering(struct agc_cb *cb, uint32_t video_out, int32_t index)
{
   return agc_emit_rendering_wait(cb, video_out, index, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetWorkloadsActive(struct agc_cb *cb, uint32_t stream, const uint32_t *ids,
                            uint32_t count)
{
   return agc_emit_workloads_active(cb, stream, ids, count, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetWorkloadComplete(struct agc_cb *cb, uint32_t stream, uint32_t id)
{
   return agc_emit_workload_complete(cb, stream, id, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbSetWorkloadStreamInactive(struct agc_cb *cb, uint32_t stream)
{
   return agc_emit_workload_stream_inactive(cb, stream, __func__);
}

#define AGC_CONTEXT_OP_CLEAR 0

static const uint8_t context_state_dwords[] = {5, 27, 27, 32};

static bool
driver_sequence(int32_t operation, uint64_t *va, uint32_t *dwords)
{
   if (operation < 0)
      return agc_driver_reset_sequence(va, dwords);
   return agc_driver_context_sequence((uint32_t)operation, va, dwords);
}

static uint32_t *
call_driver_sequence(struct agc_cb *cb, int32_t operation, uint32_t footprint, const char *function)
{
   uint64_t va = 0;
   uint32_t dwords = 0;
   if (!driver_sequence(operation, &va, &dwords)) {
      agc_driver_start(NULL, 0);
      if (!driver_sequence(operation, &va, &dwords)) {
         agc_report(function, "the driver has no state sequence to call");
         return NULL;
      }
   }
   uint32_t *out = agc_record_alloc(cb, footprint);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   out[1] = (uint32_t)va;
   out[2] = (uint32_t)(va >> 32);
   out[3] = S_3F3_IB_SIZE(dwords) | S_3F3_VALID(1);
   agc_fill_nops(out + AGC_DWORDS_JUMP, footprint - AGC_DWORDS_JUMP);
   agc_record_add_reloc(out, AGC_RELOC_CONTEXT_OP)->arg32 = operation < 0 ? 0 : (uint32_t)operation;
   return out;
}

uint32_t *AGC_SYSV
sceAgcDcbResetQueue(struct agc_cb *cb, uint32_t operation, uint32_t state)
{
   (void)state;
   if (!cb || operation > AGC_RESET_QUEUE_OPS_MAX) {
      agc_report(__func__, "null command buffer or invalid reset operations");
      return NULL;
   }
   return call_driver_sequence(cb, AGC_CONTEXT_OP_CLEAR, AGC_DWORDS_RESET_QUEUE_DCB, __func__);
}

uint32_t *AGC_SYSV
sceAgcDcbContextStateOp_0100(struct agc_cb *cb, uint32_t operation)
{
   if (!cb || operation >= sizeof(context_state_dwords)) {
      agc_report(__func__, "null command buffer or invalid context state operation");
      return NULL;
   }
   return call_driver_sequence(cb, (int32_t)operation, context_state_dwords[operation], __func__);
}

uint64_t AGC_SYSV
sceAgcDcbContextStateOpGetSize(uint32_t operation)
{
   return operation < sizeof(context_state_dwords) ? context_state_dwords[operation] * 4u : 0;
}

uint32_t *AGC_SYSV
sceAgcDcbAtomicGds_0900(struct agc_cb *cb, uint8_t engine, uint16_t op, uint8_t command,
                        uint8_t float_mode, uint32_t increment, uint32_t base, uint32_t size,
                        uint32_t offset, uint32_t dst, uint64_t source0, uint64_t source1)
{
   (void)size;
   return agc_emit_atomic_gds(cb, false, engine, op, command, float_mode, increment, base, offset,
                              dst, source0, source1, __func__);
}

uint32_t AGC_SYSV
sceAgcDcbAtomicGdsGetSize(void)
{
   return AGC_DWORDS_ATOMIC_GDS * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbMemSemaphore(struct agc_cb *cb, uint64_t address, uint8_t mailbox, uint8_t signal_type,
                      uint8_t operation)
{
   return agc_emit_mem_semaphore(cb, address, mailbox, signal_type, operation, __func__);
}

static void
lod_stats_copy(uint32_t *out, uint32_t header, uint64_t source, uint64_t destination, uint32_t bytes)
{
   out[0] = PKT3(PKT3_DMA_DATA, 5, 0);
   out[1] = header;
   out[2] = (uint32_t)source;
   out[3] = (uint32_t)(source >> 32);
   out[4] = (uint32_t)destination;
   out[5] = (uint32_t)(destination >> 32);
   out[6] = S_506_BYTE_COUNT(bytes);
}

uint32_t *AGC_SYSV
sceAgcDcbGetLodStats(struct agc_cb *cb, uint8_t cache_policy, uint64_t stats, uint32_t bytes,
                     uint32_t reset_count, uint8_t force_reset, uint8_t report_and_reset,
                     uint32_t interval)
{
   if (!cb || cache_policy > 3 || reset_count > 0xff || force_reset > 1 || report_and_reset > 1 ||
       interval > 0xff || (stats && !agc_address_valid(stats, 64))) {
      agc_report(__func__, "invalid LOD statistics arguments");
      return NULL;
   }
   if (!stats)
      return agc_emit_nops(cb, AGC_DWORDS_LOD_STATS, __func__);
   uint32_t limited = bytes < AGC_MIP_STATS_REPORT_BYTES ? bytes : AGC_MIP_STATS_REPORT_BYTES;
   uint32_t slots = limited > AGC_MIP_STATS_REPORT_SLOTS_OFFSET ? (limited - AGC_MIP_STATS_REPORT_SLOTS_OFFSET) & ~3u : 0;
   uint32_t host[3 * AGC_DWORDS_DMA_DATA];
   uint32_t used = 0;
   const uint32_t copy = S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                         S_501_CP_SYNC(1);
   if (slots) {
      lod_stats_copy(host + used, copy, AGC_RUNTIME_MIP_STATS_VA, stats + AGC_MIP_STATS_REPORT_SLOTS_OFFSET, slots);
      used += AGC_DWORDS_DMA_DATA;
   }
   lod_stats_copy(host + used, copy, AGC_RUNTIME_MIP_STATS_TEMPLATE_VA, AGC_RUNTIME_MIP_STATS_VA,
                  (uint32_t)AGC_RUNTIME_MIP_STATS_BYTES);
   used += AGC_DWORDS_DMA_DATA;
   host[used++] = PKT3(PKT3_WRITE_DATA, 3, 0);
   host[used++] = S_371_DST_SEL(V_371_MEMORY) | S_371_WR_CONFIRM(1);
   host[used++] = (uint32_t)stats;
   host[used++] = (uint32_t)(stats >> 32);
   host[used++] = 1;
   uint32_t *out = agc_emit_trampoline(cb, AGC_DWORDS_LOD_STATS, host, used, __func__);
   if (out)
      agc_record_note_write_value(out, stats, 1);
   return out;
}

uint32_t AGC_SYSV
sceAgcDcbGetLodStatsGetSize(void)
{
   return AGC_DWORDS_LOD_STATS * 4;
}

uint32_t *AGC_SYSV
sceAgcDcbDrawIndexMultiInstanced(struct agc_cb *cb, uint32_t count, uint64_t indices,
                                 uint64_t objects, uint32_t instances, uint64_t bits)
{
   (void)count;
   (void)indices;
   (void)objects;
   (void)instances;
   if (!cb || !draw_modifier_valid(bits)) {
      agc_report(__func__, "invalid draw arguments");
      return NULL;
   }
   agc_fatal(__func__, "needs a trampoline");
}
