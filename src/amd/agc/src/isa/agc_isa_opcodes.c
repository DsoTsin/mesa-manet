#include "agc_isa.h"

#include <ctype.h>
#include <stdio.h>

#include "agc_isa_tables.h"

#define AGC_ISA_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

struct op_table {
   const struct agc_isa_op_info *ops;
   const uint8_t *index;
   uint16_t count;
   uint16_t index_size;
};

#define OP_TABLE(name)                                                                                            \
   {                                                                                                              \
      agc_isa_##name##_ops, agc_isa_##name##_index, AGC_ISA_ARRAY_SIZE(agc_isa_##name##_ops),                    \
         AGC_ISA_ARRAY_SIZE(agc_isa_##name##_index)                                                               \
   }

static const struct op_table op_tables[AGC_ISA_FAMILY_COUNT] = {
   [AGC_ISA_SOP1] = OP_TABLE(sop1),   [AGC_ISA_SOP2] = OP_TABLE(sop2), [AGC_ISA_SOPK] = OP_TABLE(sopk),
   [AGC_ISA_SOPC] = OP_TABLE(sopc),   [AGC_ISA_SOPP] = OP_TABLE(sopp), [AGC_ISA_VOP1] = OP_TABLE(vop1),
   [AGC_ISA_VOP2] = OP_TABLE(vop2),   [AGC_ISA_VOP3] = OP_TABLE(vop3), [AGC_ISA_VOP3P] = OP_TABLE(vop3p),
};

const struct agc_isa_op_info *
agc_isa_op_table(enum agc_isa_family family, size_t *count)
{
   if ((unsigned)family >= AGC_ISA_FAMILY_COUNT) {
      *count = 0;
      return NULL;
   }
   *count = op_tables[family].count;
   return op_tables[family].ops;
}

const struct agc_isa_op_info *
agc_isa_op_lookup(enum agc_isa_family family, uint16_t op)
{
   if ((unsigned)family >= AGC_ISA_FAMILY_COUNT)
      return NULL;
   const struct op_table *table = &op_tables[family];
   if (op >= table->index_size || !table->index[op])
      return NULL;
   return &table->ops[table->index[op] - 1];
}

static struct agc_isa_xlat
resolve_xlat(struct agc_isa_xlat gfx103, struct agc_isa_xlat gfx11, struct agc_isa_xlat gfx12, uint16_t op,
             enum agc_isa_target target)
{
   struct agc_isa_xlat x = agc_isa_xlat_of(AGC_ISA_XLAT_SAME);
   switch (target) {
   case AGC_ISA_TARGET_GFX103:
      x = gfx103;
      break;
   case AGC_ISA_TARGET_GFX11:
   case AGC_ISA_TARGET_GFX115:
      x = gfx11;
      break;
   case AGC_ISA_TARGET_GFX12:
      x = gfx12;
      break;
   default:
      break;
   }
   return x.kind == AGC_ISA_XLAT_SAME ? agc_isa_xlat_op(op) : x;
}

struct agc_isa_xlat
agc_isa_op_translate(const struct agc_isa_op_info *info, enum agc_isa_target target)
{
   return resolve_xlat(info->gfx103, info->gfx11, info->gfx12, info->op, target);
}

bool
agc_isa_vop3_base(uint16_t op, enum agc_isa_family *family, uint16_t *base)
{
   if (op <= AGC_ISA_VOP3_VOPC_LAST) {
      *family = AGC_ISA_VOPC;
      *base = op;
   } else if (op >= AGC_ISA_VOP3_VOP2_BASE && op <= AGC_ISA_VOP3_VOP2_LAST) {
      *family = AGC_ISA_VOP2;
      *base = (uint16_t)(op - AGC_ISA_VOP3_VOP2_BASE);
   } else if (op >= AGC_ISA_VOP3_VOP1_BASE && op <= AGC_ISA_VOP3_VOP1_LAST) {
      *family = AGC_ISA_VOP1;
      *base = (uint16_t)(op - AGC_ISA_VOP3_VOP1_BASE);
   } else {
      return false;
   }
   return true;
}

bool
agc_isa_vop3_flags(uint16_t op, uint64_t *flags)
{
   enum agc_isa_family family;
   uint16_t base;
   if (!agc_isa_vop3_base(op, &family, &base)) {
      const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_VOP3, op);
      if (!info)
         return false;
      *flags = info->flags;
      return true;
   }
   if (family == AGC_ISA_VOPC) {
      const struct agc_isa_vopc_info *info = agc_isa_vopc_decode(base);
      if (!info)
         return false;
      *flags = info->flags;
      return true;
   }
   const struct agc_isa_op_info *info = agc_isa_op_lookup(family, base);
   if (!info || (info->flags & AGC_ISA_FLAG_NO_VOP3))
      return false;
   *flags = info->flags;
   return true;
}

bool
agc_isa_flag_source_count(uint64_t flags, unsigned *count)
{
   if (flags & AGC_ISA_FLAG_SRC3)
      *count = 3;
   else if (flags & AGC_ISA_FLAG_SRC2)
      *count = 2;
   else if (flags & AGC_ISA_FLAG_SRC1)
      *count = 1;
   else
      return false;
   return true;
}

unsigned
agc_isa_vop3_source_count(uint16_t op, uint64_t flags)
{
   unsigned count;
   if (agc_isa_flag_source_count(flags, &count))
      return count;
   enum agc_isa_family family;
   uint16_t base;
   if (!agc_isa_vop3_base(op, &family, &base))
      return 3;
   if (family == AGC_ISA_VOP1)
      return (flags & AGC_ISA_FLAG_NO_OPERANDS) ? 0 : 1;
   if (family == AGC_ISA_VOP2 && (flags & AGC_ISA_FLAG_CARRY_IN))
      return 3;
   return 2;
}

unsigned
agc_isa_source_count(const struct agc_isa_inst *inst)
{
   if (inst->fields == AGC_ISA_VOP3)
      return agc_isa_vop3_source_count(inst->opcode, inst->flags);
   unsigned count;
   if (agc_isa_flag_source_count(inst->flags, &count))
      return count;
   return inst->fields == AGC_ISA_VOP3P ? 3 : 2;
}

const struct agc_isa_vopc_info *
agc_isa_vopc_decode(uint16_t op)
{
   if (op >= AGC_ISA_ARRAY_SIZE(agc_isa_vopc_index) || !agc_isa_vopc_index[op])
      return NULL;
   return &agc_isa_vopc_ops[agc_isa_vopc_index[op] - 1];
}

bool
agc_isa_vopc_is_constant_result(const struct agc_isa_vopc_info *info, bool *result)
{
   bool is_float = info->type == AGC_ISA_CMP_F16 || info->type == AGC_ISA_CMP_F32 || info->type == AGC_ISA_CMP_F64;
   if (info->cond == 0)
      *result = false;
   else if (info->cond == (is_float ? 15 : 7))
      *result = true;
   else
      return false;
   return true;
}

static const struct {
   const struct agc_isa_mem_op *ops;
   size_t count;
} mem_tables[AGC_ISA_MEM_TABLE_COUNT] = {
   [AGC_ISA_MEM_TABLE_SMEM] = {agc_isa_smem_ops, AGC_ISA_ARRAY_SIZE(agc_isa_smem_ops)},
   [AGC_ISA_MEM_TABLE_VMEM_COMMON] = {agc_isa_vmem_common_ops, AGC_ISA_ARRAY_SIZE(agc_isa_vmem_common_ops)},
   [AGC_ISA_MEM_TABLE_MUBUF_ONLY] = {agc_isa_mubuf_only_ops, AGC_ISA_ARRAY_SIZE(agc_isa_mubuf_only_ops)},
   [AGC_ISA_MEM_TABLE_MTBUF] = {agc_isa_mtbuf_ops, AGC_ISA_ARRAY_SIZE(agc_isa_mtbuf_ops)},
   [AGC_ISA_MEM_TABLE_FLAT_ONLY] = {agc_isa_flat_only_ops, AGC_ISA_ARRAY_SIZE(agc_isa_flat_only_ops)},
};

const struct agc_isa_mem_op *
agc_isa_mem_table(enum agc_isa_mem_table table, size_t *count)
{
   if ((unsigned)table >= AGC_ISA_MEM_TABLE_COUNT) {
      *count = 0;
      return NULL;
   }
   *count = mem_tables[table].count;
   return mem_tables[table].ops;
}

const struct agc_isa_mem_op *
agc_isa_smem_op(uint16_t op)
{
   if (op >= AGC_ISA_ARRAY_SIZE(agc_isa_smem_index) || !agc_isa_smem_index[op])
      return NULL;
   return &agc_isa_smem_ops[agc_isa_smem_index[op] - 1];
}

static const struct agc_isa_mem_op *
chained_op(const uint8_t *index, size_t index_size, const struct agc_isa_mem_op *only, uint16_t op)
{
   if (op >= index_size || !index[op])
      return NULL;
   size_t position = index[op] - 1u;
   size_t common = AGC_ISA_ARRAY_SIZE(agc_isa_vmem_common_ops);
   return position < common ? &agc_isa_vmem_common_ops[position] : &only[position - common];
}

const struct agc_isa_mem_op *
agc_isa_mubuf_op(uint16_t op)
{
   return chained_op(agc_isa_mubuf_index, AGC_ISA_ARRAY_SIZE(agc_isa_mubuf_index), agc_isa_mubuf_only_ops, op);
}

const struct agc_isa_mem_op *
agc_isa_mtbuf_op(uint16_t op)
{
   if (op >= AGC_ISA_ARRAY_SIZE(agc_isa_mtbuf_index) || !agc_isa_mtbuf_index[op])
      return NULL;
   return &agc_isa_mtbuf_ops[agc_isa_mtbuf_index[op] - 1];
}

const struct agc_isa_mem_op *
agc_isa_flat_op(uint16_t op)
{
   return chained_op(agc_isa_flat_index, AGC_ISA_ARRAY_SIZE(agc_isa_flat_index), agc_isa_flat_only_ops, op);
}

bool
agc_isa_flat_only_op(uint16_t op)
{
   for (size_t i = 0; i < AGC_ISA_ARRAY_SIZE(agc_isa_flat_only_ops); i++) {
      if (agc_isa_flat_only_ops[i].op == op)
         return true;
   }
   return false;
}

struct agc_isa_xlat
agc_isa_mem_op_translate(const struct agc_isa_mem_op *op, enum agc_isa_target target)
{
   return resolve_xlat(op->gfx103, op->gfx11, op->gfx12, op->op, target);
}

const struct agc_isa_ds_op *
agc_isa_ds_table(size_t *count)
{
   *count = AGC_ISA_ARRAY_SIZE(agc_isa_ds_ops);
   return agc_isa_ds_ops;
}

const struct agc_isa_ds_op *
agc_isa_ds_op(uint16_t op)
{
   if (op >= AGC_ISA_ARRAY_SIZE(agc_isa_ds_index) || !agc_isa_ds_index[op])
      return NULL;
   return &agc_isa_ds_ops[agc_isa_ds_index[op] - 1];
}

struct agc_isa_xlat
agc_isa_ds_op_translate(const struct agc_isa_ds_op *op, enum agc_isa_target target)
{
   if (op->flags & AGC_ISA_DS_FLAG_STACK8)
      return agc_isa_xlat_rule(AGC_ISA_RULE_BVH_STACK8, 0);
   unsigned removed = 0;
   if (target == AGC_ISA_TARGET_GFX103 || agc_isa_target_is_gfx11(target))
      removed = AGC_ISA_DS_FLAG_SRC2;
   else if (target == AGC_ISA_TARGET_GFX12)
      removed = AGC_ISA_DS_FLAG_SRC2 | AGC_ISA_DS_FLAG_GWS | AGC_ISA_DS_FLAG_GDS_ONLY | AGC_ISA_DS_FLAG_NOT_GFX12;
   if (op->flags & removed)
      return agc_isa_xlat_of(AGC_ISA_XLAT_NO);
   if (agc_isa_target_is_gfx11_plus(target) && op->op == AGC_ISA_DS_ADD_RTN_F32)
      return agc_isa_xlat_op(AGC_ISA_DS_ADD_RTN_F32_GFX11);
   return agc_isa_xlat_op(op->op);
}

bool
agc_isa_ds_op_swaps_data(const struct agc_isa_ds_op *op, enum agc_isa_target target)
{
   return (op->flags & AGC_ISA_DS_FLAG_CMPST) && agc_isa_target_is_gfx11_plus(target);
}

const struct agc_isa_image_op *
agc_isa_image_table(size_t *count)
{
   *count = AGC_ISA_ARRAY_SIZE(agc_isa_image_ops);
   return agc_isa_image_ops;
}

const struct agc_isa_image_op *
agc_isa_image_op(uint16_t op)
{
   if (op >= AGC_ISA_ARRAY_SIZE(agc_isa_image_index) || !agc_isa_image_index[op])
      return NULL;
   return &agc_isa_image_ops[agc_isa_image_index[op] - 1];
}

const struct agc_isa_image_op *
agc_isa_image_op_plain(const struct agc_isa_image_op *op)
{
   if (!(op->flags & AGC_ISA_IMG_ADJUST))
      return NULL;
   return agc_isa_image_op((uint16_t)(op->op & ~AGC_ISA_IMG_ADJUST_OPCODE_BIT));
}

struct agc_isa_xlat
agc_isa_image_op_translate(const struct agc_isa_image_op *op, enum agc_isa_target target)
{
   if (op->flags & AGC_ISA_IMG_BVH8)
      return agc_isa_xlat_rule(AGC_ISA_RULE_BVH8, 0);
   if ((op->flags & AGC_ISA_IMG_ADJUST) && target != AGC_ISA_TARGET_GFX1013) {
      const struct agc_isa_image_op *plain = agc_isa_image_op_plain(op);
      return plain ? agc_isa_image_op_translate(plain, target) : agc_isa_xlat_of(AGC_ISA_XLAT_NO);
   }
   return resolve_xlat(op->gfx103, op->gfx11, op->gfx12, op->op, target);
}

uint8_t
agc_isa_image_coordinates(uint8_t dim)
{
   switch (dim) {
   case 0:
      return 1;
   case 1:
   case 4:
      return 2;
   case 2:
   case 3:
   case 5:
   case 6:
      return 3;
   default:
      return 4;
   }
}

static uint8_t
image_gradients(uint8_t dim)
{
   switch (dim) {
   case 0:
   case 4:
      return 1;
   case 2:
      return 3;
   default:
      return 2;
   }
}

uint8_t
agc_isa_image_address_dwords(uint32_t flags, uint8_t dim, bool a16)
{
   if (flags & AGC_ISA_IMG_BVH)
      return (uint8_t)((a16 ? 8 : 11) + ((flags & AGC_ISA_IMG_BVH64) ? 1 : 0));
   if (flags & AGC_ISA_IMG_RESINFO)
      return 1;
   unsigned dwords = ((flags & AGC_ISA_IMG_OFFSET) ? 1 : 0) + ((flags & AGC_ISA_IMG_BIAS) ? 1 : 0) +
                     ((flags & AGC_ISA_IMG_COMPARE) ? 1 : 0);
   if (flags & AGC_ISA_IMG_DERIV) {
      unsigned grads = image_gradients(dim);
      dwords += ((flags & AGC_ISA_IMG_G16) || a16) ? 2 * ((grads + 1) / 2) : 2 * grads;
   }
   unsigned trailing = agc_isa_image_coordinates(dim) + ((flags & AGC_ISA_IMG_LOD) ? 1 : 0) +
                       ((flags & AGC_ISA_IMG_CLAMP) ? 1 : 0) + ((flags & AGC_ISA_IMG_MIP) ? 1 : 0);
   return (uint8_t)(dwords + (a16 ? (trailing + 1) / 2 : trailing));
}

uint8_t
agc_isa_image_data_dwords(uint32_t flags, uint8_t dmask, bool d16)
{
   if (flags & AGC_ISA_IMG_BVH8)
      return 10;
   if (flags & AGC_ISA_IMG_BVH)
      return 4;
   unsigned components = 4;
   if (!(flags & (AGC_ISA_IMG_GATHER | AGC_ISA_IMG_MSAA)) || (flags & AGC_ISA_IMG_PACKED_GATHER)) {
      components = 0;
      for (unsigned bit = 0; bit < 8; bit++)
         components += (dmask >> bit) & 1u;
   }
   return (uint8_t)(d16 ? (components + 1) / 2 : components);
}

bool
agc_isa_remap_buffer_format(uint8_t format, enum agc_isa_target target, uint8_t *remapped)
{
   if (!agc_isa_target_is_gfx11_plus(target) || format <= 29) {
      *remapped = format;
      return format <= 77;
   }
   if (format == 36)
      *remapped = 30;
   else if (format == 43)
      *remapped = 31;
   else if (format == 44 || format == 45)
      *remapped = (uint8_t)(format - 12);
   else if (format >= 48 && format <= 77)
      *remapped = (uint8_t)(format - 14);
   else
      return false;
   return true;
}

static const char *
known_mnemonic(enum agc_isa_family family, uint16_t op)
{
   static const char *const vintrp[3] = {"v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32"};
   const struct agc_isa_op_info *info;
   const struct agc_isa_vopc_info *vopc;
   const struct agc_isa_mem_op *mem;
   const struct agc_isa_ds_op *ds;
   const struct agc_isa_image_op *image;
   enum agc_isa_family base_family;
   uint16_t base;
   switch (family) {
   case AGC_ISA_VOPC:
      vopc = agc_isa_vopc_decode(op);
      return vopc ? vopc->name : NULL;
   case AGC_ISA_VOP3:
      if (!agc_isa_vop3_base(op, &base_family, &base)) {
         info = agc_isa_op_lookup(AGC_ISA_VOP3, op);
         return info ? info->name : NULL;
      }
      if (base_family == AGC_ISA_VOPC)
         return known_mnemonic(AGC_ISA_VOPC, base);
      info = agc_isa_op_lookup(base_family, base);
      return info ? info->name : NULL;
   case AGC_ISA_VINTRP:
      return op <= AGC_ISA_VINTRP_MOV ? vintrp[op] : NULL;
   case AGC_ISA_EXP:
      return "exp";
   case AGC_ISA_SMEM:
      mem = agc_isa_smem_op(op);
      return mem ? mem->name : NULL;
   case AGC_ISA_MUBUF:
      return op < AGC_ISA_ARRAY_SIZE(agc_isa_mubuf_names) ? agc_isa_mubuf_names[op] : NULL;
   case AGC_ISA_MTBUF:
      mem = agc_isa_mtbuf_op(op);
      return mem ? mem->name : NULL;
   case AGC_ISA_FLAT:
      return op < AGC_ISA_ARRAY_SIZE(agc_isa_flat_names) ? agc_isa_flat_names[op] : NULL;
   case AGC_ISA_DS:
      ds = agc_isa_ds_op(op);
      return ds ? ds->name : NULL;
   case AGC_ISA_MIMG:
      image = agc_isa_image_op(op);
      return image ? image->name : NULL;
   default:
      info = agc_isa_op_lookup(family, op);
      return info ? info->name : NULL;
   }
}

const char *
agc_isa_mnemonic(enum agc_isa_family family, uint16_t op, char scratch[AGC_ISA_MNEMONIC_SIZE])
{
   const char *known = known_mnemonic(family, op);
   if (known)
      return known;
   const char *name = agc_isa_family_name(family);
   size_t length = 0;
   for (; name[length] && length < AGC_ISA_MNEMONIC_SIZE - 16; length++)
      scratch[length] = (char)tolower((unsigned char)name[length]);
   snprintf(scratch + length, AGC_ISA_MNEMONIC_SIZE - length, "_op_0x%x", (unsigned)op);
   return scratch;
}
