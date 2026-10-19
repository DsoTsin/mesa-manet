#include "pan_graph_tsu.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct row_operation {
   unsigned kind;
   unsigned target;
   uint64_t value;
   int32_t low;
   int32_t high;
};

struct row_operations {
   size_t count;
   size_t capacity;
   struct row_operation *data;
};

struct stage {
   unsigned rows;
   unsigned columns;
   uint64_t matrix[PAN_GRAPH_TSU_MAX_RANK][PAN_GRAPH_TSU_MAX_RANK];
   struct row_operations operations[PAN_GRAPH_TSU_MAX_RANK];
};

struct compiler {
   enum pan_graph_tsu_status status;
   unsigned initial_rank;
   unsigned rank;
   int64_t low[PAN_GRAPH_TSU_MAX_RANK];
   int64_t high[PAN_GRAPH_TSU_MAX_RANK];
   size_t stage_count;
   size_t stage_capacity;
   struct stage *stages;
   unsigned current;
   size_t count;
   size_t capacity;
   struct pan_graph_tsu_instr *instructions;
};

static bool
reserve(struct compiler *c, void **data, size_t *capacity, size_t count,
        size_t stride)
{
   if (count <= *capacity)
      return true;
   if (count > SIZE_MAX / stride) {
      c->status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
      return false;
   }
   size_t next = *capacity <= SIZE_MAX / 2 ? *capacity * 2 : count;
   if (next < count)
      next = count;
   if (next < 8)
      next = 8;
   if (next > SIZE_MAX / stride)
      next = count;
   void *result = realloc(*data, next * stride);
   if (!result) {
      c->status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
      return false;
   }
   *data = result;
   *capacity = next;
   return true;
}

static bool
checked_add(int64_t a, int64_t b, int64_t *result)
{
   if ((b > 0 && a > INT64_MAX - b) ||
       (b < 0 && a < INT64_MIN - b))
      return false;
   *result = a + b;
   return true;
}

static bool
checked_sub(int64_t a, int64_t b, int64_t *result)
{
   if ((b > 0 && a < INT64_MIN + b) ||
       (b < 0 && a > INT64_MAX + b))
      return false;
   *result = a - b;
   return true;
}

static bool
checked_mul(int64_t a, int64_t b, int64_t *result)
{
   uint64_t ua = a < 0 ? -(uint64_t)a : (uint64_t)a;
   uint64_t ub = b < 0 ? -(uint64_t)b : (uint64_t)b;
   bool negative = (a < 0) != (b < 0);
   uint64_t limit = negative ? (uint64_t)INT64_MAX + 1 : INT64_MAX;
   if (ub && ua > limit / ub)
      return false;
   uint64_t product = ua * ub;
   if (negative)
      *result = product == (uint64_t)INT64_MAX + 1 ? INT64_MIN
                                                : -(int64_t)product;
   else
      *result = product;
   return true;
}

static int64_t
floor_div(int64_t value, int64_t divisor)
{
   return value / divisor - (value % divisor < 0);
}

static struct stage *
new_stage(struct compiler *c, unsigned rank)
{
   if (!reserve(c, (void **)&c->stages, &c->stage_capacity,
                c->stage_count + 1, sizeof(*c->stages)))
      return NULL;
   struct stage *s = &c->stages[c->stage_count++];
   memset(s, 0, sizeof(*s));
   s->rows = s->columns = rank;
   for (unsigned i = 0; i < rank; i++)
      s->matrix[i][i] = 1;
   return s;
}

static struct stage *
ensure_stage(struct compiler *c)
{
   return c->stage_count ? &c->stages[c->stage_count - 1]
                         : new_stage(c, c->initial_rank);
}

static bool
append_row(struct compiler *c, struct stage *s, unsigned index,
           struct row_operation op)
{
   struct row_operations *row = &s->operations[index];
   if (!reserve(c, (void **)&row->data, &row->capacity, row->count + 1,
                sizeof(*row->data)))
      return false;
   row->data[row->count++] = op;
   return true;
}

static bool
blocked(const struct stage *s, unsigned index)
{
   const struct row_operations *row = &s->operations[index];
   return row->count && row->data[row->count - 1].kind == 8;
}

static bool
unblock(struct compiler *c, struct stage **s, unsigned index)
{
   if (blocked(*s, index))
      *s = new_stage(c, (*s)->rows);
   return *s != NULL;
}

static bool
plan_row(struct compiler *c, unsigned index, struct row_operation op)
{
   struct stage *s = ensure_stage(c);
   if (!s)
      return false;
   if (op.kind == 4 || op.kind == 5) {
      if (!unblock(c, &s, op.target) || !unblock(c, &s, index) ||
          !append_row(c, s, op.target, (struct row_operation){.kind = 8}))
         return false;
   } else if (op.kind == 3) {
      if (!unblock(c, &s, index) || !unblock(c, &s, index + 1) ||
          !append_row(c, s, index, (struct row_operation){.kind = 8}))
         return false;
      index++;
   } else if (!unblock(c, &s, index)) {
      return false;
   }
   return append_row(c, s, index, op);
}

static bool
plan_affine(struct compiler *c, const struct pan_graph_tsu_transform *t)
{
   unsigned rows = t->rows - 1, columns = t->columns - 1;
   if (t->rows < 2 || rows > PAN_GRAPH_TSU_MAX_RANK ||
       t->columns != c->rank + 1 || !t->low || !t->high)
      return false;
   bool row_multi = false, col_multi = false, identity = rows == columns;
   for (unsigned r = 0; r < rows; r++) {
      unsigned used = 0;
      for (unsigned j = 0; j < columns; j++) {
         size_t k = r * t->columns + j;
         if (t->low[k] < 0 || t->low[k] != t->high[k])
            return false;
         used += t->low[k] != 0;
         identity &= t->low[k] == (r == j);
      }
      row_multi |= used > 1;
   }
   for (unsigned j = 0; j < columns; j++) {
      unsigned used = 0;
      for (unsigned r = 0; r < rows; r++)
         used += t->low[r * t->columns + j] != 0;
      col_multi |= used > 1;
   }
   if (row_multi && col_multi)
      return false;
   for (unsigned j = 0; j <= columns; j++) {
      size_t k = rows * t->columns + j;
      if (t->low[k] != (j == columns) || t->high[k] != (j == columns))
         return false;
   }

   if (!identity) {
      struct stage *s = new_stage(c, rows);
      if (!s)
         return false;
      s->columns = columns;
      memset(s->matrix, 0, sizeof(s->matrix));
      for (unsigned r = 0; r < rows; r++)
         for (unsigned j = 0; j < columns; j++)
            s->matrix[r][j] = t->low[r * t->columns + j];
   }
   int64_t low[PAN_GRAPH_TSU_MAX_RANK], high[PAN_GRAPH_TSU_MAX_RANK];
   for (unsigned r = 0; r < rows; r++) {
      size_t k = r * t->columns + columns;
      int64_t bias = t->low[k], delta;
      if (!checked_sub(t->high[k], bias, &delta))
         return false;
      if ((bias && !plan_row(c, r, (struct row_operation){
                     .kind = 0, .value = (uint64_t)bias})) ||
          (delta && !plan_row(c, r, (struct row_operation){
                      .kind = 1, .value = (uint64_t)delta})))
         return false;
      low[r] = bias;
      high[r] = t->high[k];
      for (unsigned j = 0; j < columns; j++) {
         int64_t a, b;
         if (!checked_mul(t->low[r * t->columns + j], c->low[j], &a) ||
             !checked_mul(t->high[r * t->columns + j], c->high[j], &b) ||
             !checked_add(low[r], a, &low[r]) ||
             !checked_add(high[r], b, &high[r]))
            return false;
      }
   }
   c->rank = rows;
   memcpy(c->low, low, rows * sizeof(*low));
   memcpy(c->high, high, rows * sizeof(*high));
   return true;
}

static bool
plan(struct compiler *c, const struct pan_graph_tsu_transform *t)
{
   unsigned i = t->index;
   int64_t d = t->value;
   if (t->kind == PAN_GRAPH_TSU_AFFINE)
      return plan_affine(c, t);
   if (t->kind != PAN_GRAPH_TSU_DIVIDE && i >= c->rank)
      return false;
   switch (t->kind) {
   case PAN_GRAPH_TSU_DIVIDE:
      if (!t->divisors)
         return false;
      for (i = 0; i < c->rank; i++) {
         d = t->divisors[i];
         if (d <= 0 || d > UINT32_MAX)
            return false;
         if (d != 1 && !plan_row(c, i, (struct row_operation){
               .kind = 2, .value = d, .low = (int32_t)c->low[i],
               .high = (int32_t)c->high[i]}))
            return false;
         c->low[i] = floor_div(c->low[i], d);
         c->high[i] = floor_div(c->high[i], d);
      }
      return true;
   case PAN_GRAPH_TSU_SPLIT: {
      if (d <= 0 || d > UINT32_MAX || c->rank == PAN_GRAPH_TSU_MAX_RANK)
         return false;
      struct stage *s = ensure_stage(c);
      if (!s)
         return false;
      if (s->rows != c->rank)
         return false;
      memmove(&s->matrix[i + 1], &s->matrix[i],
              (s->rows - i) * sizeof(s->matrix[0]));
      memset(&s->matrix[i], 0, sizeof(s->matrix[0]));
      memmove(&s->operations[i + 1], &s->operations[i],
              (s->rows - i) * sizeof(s->operations[0]));
      s->operations[i] = (struct row_operations){0};
      s->rows++;
      if (!plan_row(c, i, (struct row_operation){.kind = 3, .value = d,
                      .low = (int32_t)c->low[i], .high = (int32_t)c->high[i]}))
         return false;
      int64_t lq = floor_div(c->low[i], d), hq = floor_div(c->high[i], d);
      int64_t lr = c->low[i] % d, hr = c->high[i] % d;
      lr = lr < 0 ? lr + d : lr;
      hr = hr < 0 ? hr + d : hr;
      if (lq != hq) {
         lr = 0;
         hr = d - 1;
      }
      memmove(&c->low[i + 2], &c->low[i + 1],
              (c->rank - i - 1) * sizeof(c->low[0]));
      memmove(&c->high[i + 2], &c->high[i + 1],
              (c->rank - i - 1) * sizeof(c->high[0]));
      c->low[i] = lq;
      c->high[i] = hq;
      c->low[i + 1] = lr;
      c->high[i + 1] = hr;
      c->rank++;
      return true;
   }
   case PAN_GRAPH_TSU_NEGATE: {
      if (c->low[i] == INT64_MIN || c->high[i] == INT64_MIN ||
          !plan_row(c, i, (struct row_operation){.kind = 6}))
         return false;
      int64_t low = -c->high[i];
      c->high[i] = -c->low[i];
      c->low[i] = low;
      return true;
   }
   case PAN_GRAPH_TSU_CLAMP:
      if (d < 0 || !plan_row(c, i, (struct row_operation){.kind = 7, .value = d}))
         return false;
      c->low[i] = c->low[i] < 0 ? 0 : c->low[i] > d ? d : c->low[i];
      c->high[i] = c->high[i] > d ? d : c->high[i];
      return true;
   case PAN_GRAPH_TSU_SUBTRACT: {
      unsigned j = t->other;
      if (d <= 0 || j >= c->rank)
         return false;
      int64_t a, b;
      if (!checked_mul(d, c->low[j], &a) ||
          !checked_mul(d, c->high[j], &b) ||
          !checked_sub(c->low[i], a, &c->low[i]) ||
          !checked_sub(c->high[i], b, &c->high[i]))
         return false;
      return plan_row(c, i > j ? i : j, (struct row_operation){
         .kind = j >= i ? 5 : 4, .value = d, .target = i < j ? i : j});
   }
   default:
      return false;
   }
}

static void
emit(struct compiler *c, unsigned opcode, unsigned reg, uint64_t immediate,
     unsigned shift, unsigned increment, unsigned flag)
{
   if (c->status != PAN_GRAPH_TSU_SUCCESS ||
       !reserve(c, (void **)&c->instructions, &c->capacity, c->count + 1,
                sizeof(*c->instructions)))
      return;
   c->instructions[c->count++] = (struct pan_graph_tsu_instr){
      .opcode = opcode, .reg = reg, .immediate = immediate, .shift = shift,
      .increment = increment, .flag = flag,
      .size = (opcode >= 5 && opcode <= 14) ||
              (opcode >= 18 && opcode <= 20) || opcode == 2 ? 2 : 1,
   };
}

static void
select_register(struct compiler *c, unsigned target)
{
   if (target == c->current)
      return;
   if (target < 8) {
      emit(c, 4, target, 0, 0, 0, 0);
   } else {
      emit(c, 4, 7, 0, 0, 0, 0);
      emit(c, 1, 7, 0, 0, 1, 0);
      for (unsigned n = 8; n < target; n++)
         emit(c, 2, 0, n, 0, 1, 0);
   }
   c->current = target;
}

static bool
alternate(struct compiler *c, unsigned dest, uint64_t value, bool initialized)
{
   for (int shift = 24; shift >= 0; shift -= 8) {
      uint64_t part = shift == 24 ? value >> shift : (value >> shift) & 255;
      if (!part)
         continue;
      select_register(c, dest);
      emit(c, initialized ? 7 : 6, 0, part, shift, 0, 1);
      initialized = true;
   }
   return initialized;
}

static bool
arithmetic(struct compiler *c, unsigned dest, unsigned source, uint64_t value,
           bool initialized)
{
   if (dest == source && value >= 256 && (value & UINT64_C(0xffffffffff00ffff)) &&
       (value & UINT64_C(0xffffffffffff00ff))) {
      select_register(c, dest);
      initialized = alternate(c, dest, value, initialized);
      select_register(c, dest);
      emit(c, 7, 0, 0, 0, 0, 0);
      return initialized;
   }
   for (int shift = 24; shift >= 0; shift -= 8) {
      uint64_t part = shift == 24 ? value >> shift : (value >> shift) & 255;
      if (!part)
         continue;
      select_register(c, dest);
      if (initialized) {
         if (!shift && part == 1 && dest != source)
            emit(c, 3, source, 0, 0, 0, 0);
         else
            emit(c, 5, source, part, shift, 0, 0);
      } else if (dest == source) {
         if (!shift && part == 2)
            emit(c, 3, source, 0, 0, 0, 0);
         else if (shift || part != 1)
            emit(c, 6, 0, part, shift, 0, 0);
      } else if (!shift && part == 1) {
         emit(c, 0, source, 0, 0, 0, 0);
      } else {
         emit(c, 17, 0, 0, 0, 0, 0);
         emit(c, 5, source, part, shift, 0, 0);
      }
      initialized = true;
   }
   return initialized;
}

static bool
coefficient(struct compiler *c, unsigned dest, unsigned source, uint64_t value,
            bool initialized)
{
   if (source < 8)
      return arithmetic(c, dest, source, value, initialized);
   unsigned temporary = dest == 0;
   select_register(c, temporary);
   emit(c, 2, 0, source, 0, 0, 0);
   initialized = arithmetic(c, source == dest ? 0 : dest, temporary, value,
                            initialized);
   select_register(c, temporary);
   emit(c, 2, 0, source, 0, 0, 0);
   return initialized;
}

static unsigned
bit_length(uint64_t value)
{
   unsigned length = 0;
   for (; value; value >>= 1)
      length++;
   return length;
}

static void
divide(struct compiler *c, unsigned dest, uint32_t divisor, int32_t low,
       int32_t high)
{
   select_register(c, dest);
   unsigned k = bit_length(divisor - 1);
   if (!(divisor & (divisor - 1))) {
      emit(c, 14, 0, k, 0, 0, 0);
      return;
   }
   int32_t bound = ~low > high ? ~low : high;
   unsigned precision = 7;
   while (precision < 31 && (int64_t)bound >= (INT64_C(1) << precision))
      precision += 8;
   uint64_t multiplier = ((UINT64_C(1) << (precision + k)) +
                          (UINT64_C(1) << k)) / divisor;
   uint32_t byte = multiplier & 255;
   bool initialized = byte != 0;
   emit(c, initialized ? 9 : 8, 0, byte, 0, 0, 1);
   for (unsigned shift = 8; shift <= precision; shift += 8) {
      byte = (multiplier >> shift) & 255;
      if (!initialized && byte) {
         emit(c, 11, 0, byte, 0, 0, 1);
         initialized = true;
      } else {
         emit(c, 10, 0, byte, 0, 0, 1);
      }
   }
   emit(c, 7, 0, 0, 0, 0, 0);
   if (k != 1)
      emit(c, 14, 0, k - 1, 0, 0, 0);
}

static void
emit_row(struct compiler *c, unsigned index, const struct row_operation *op)
{
   if (op->kind <= 1) {
      select_register(c, index);
      bool negative = op->value >> 63;
      uint64_t magnitude = negative && !op->kind ? -op->value : op->value;
      unsigned opcode = op->kind ? 18 : negative ? 20 : 19;
      for (int shift = 24; shift >= 0; shift -= 8) {
         uint64_t part = shift == 24 ? magnitude >> shift
                                    : (magnitude >> shift) & 255;
         if (part)
            emit(c, opcode, 0, part, shift, 0, 0);
      }
   } else if (op->kind == 2) {
      divide(c, index, op->value, op->low, op->high);
   } else if (op->kind == 3) {
      select_register(c, index - 1);
      emit(c, 0, index, 0, 0, 0, 0);
      divide(c, index - 1, op->value, op->low, op->high);
      alternate(c, index - 1, op->value, false);
      select_register(c, index);
      emit(c, 16, 0, 0, 0, 0, 0);
   } else if (op->kind == 4 || op->kind == 5) {
      alternate(c, op->kind == 4 ? op->target : index, op->value, false);
      select_register(c, op->kind == 4 ? index : op->target);
      emit(c, 16, 0, 0, 0, 0, 0);
   } else if (op->kind == 6) {
      select_register(c, index);
      emit(c, 15, 0, 0, 0, 0, 0);
   } else if (op->kind == 7) {
      select_register(c, index);
      unsigned shift = bit_length(op->value) & ~7u;
      emit(c, 12, 0, (op->value >> shift) & 255, shift, 0, 0);
      while (shift) {
         shift -= 8;
         emit(c, 13, 0, (op->value >> shift) & 255, shift, 0, 0);
      }
   }
}

static void
swap_columns(struct compiler *c, struct stage *s, unsigned n, unsigned a,
             unsigned b)
{
   select_register(c, a);
   emit(c, b < 8 ? 1 : 2, b < 8 ? b : 0, b < 8 ? 0 : b, 0, 0, 0);
   for (unsigned r = 0; r < n; r++) {
      uint64_t value = s->matrix[r][a];
      s->matrix[r][a] = s->matrix[r][b];
      s->matrix[r][b] = value;
   }
}

static unsigned
row_used(const struct stage *s, unsigned n, unsigned row)
{
   unsigned used = 0;
   for (unsigned j = 0; j < n; j++)
      used += s->matrix[row][j] != 0;
   return used;
}

static void
identity_row(struct stage *s, unsigned row, unsigned index)
{
   memset(s->matrix[row], 0, sizeof(s->matrix[row]));
   s->matrix[row][index] = 1;
}

static bool
is_identity_row(const struct stage *s, unsigned n, unsigned row)
{
   for (unsigned j = 0; j < n; j++) {
      if (s->matrix[row][j] != (row == j))
         return false;
   }
   return true;
}

static bool
schedule_stage(struct compiler *c, struct stage *s)
{
   unsigned n = s->rows > s->columns ? s->rows : s->columns;
   for (unsigned source = 0; source < s->rows; source++) {
      bool seen = false;
      for (unsigned dest = 0; dest < n; dest++) {
         uint64_t value = s->matrix[dest][source];
         if (!value)
            continue;
         if (!seen) {
            seen = true;
            continue;
         }
         if (row_used(s, n, dest) > 1)
            return false;
         if (source < 8) {
            arithmetic(c, dest, source, value, false);
         } else {
            unsigned temporary = dest == 0;
            select_register(c, temporary);
            emit(c, 2, 0, source, 0, 0, 0);
            arithmetic(c, source == dest ? 0 : dest, temporary,
                       s->matrix[dest][temporary], false);
            select_register(c, temporary);
            emit(c, 2, 0, source, 0, 0, 0);
         }
         identity_row(s, dest, dest);
      }
   }
   for (unsigned i = 0; i < s->rows; i++) {
      select_register(c, i);
      unsigned users = 0;
      for (unsigned r = 0; r < n; r++)
         users += s->matrix[r][i] != 0;
      if (users > 1)
         return false;
      while (!is_identity_row(s, n, i)) {
         bool later_user = false;
         for (unsigned r = i + 1; r < n; r++)
            later_user |= s->matrix[r][i] != 0;
         if (later_user) {
            for (unsigned j = i + 1; j < n; j++) {
               if (s->matrix[i][j]) {
                  swap_columns(c, s, n, i, j);
                  break;
               }
            }
         }
         unsigned working = i;
         for (unsigned r = 0; r < n; r++) {
            if (s->matrix[r][i]) {
               working = r;
               break;
            }
         }
         if (!row_used(s, n, working)) {
            emit(c, 17, 0, 0, 0, 0, 0);
         } else {
            unsigned pivot = 0;
            while (!s->matrix[working][pivot])
               pivot++;
            if (i != pivot && s->matrix[working][pivot] == 1)
               swap_columns(c, s, n, i, pivot);
            bool initialized = false;
            if (s->matrix[working][i])
               initialized = coefficient(c, i, i, s->matrix[working][i], false);
            for (unsigned j = i + 1; j < n; j++) {
               if (s->matrix[working][j])
                  initialized = coefficient(c, i, j, s->matrix[working][j],
                                            initialized);
            }
         }
         identity_row(s, working, i);
         if (i != working)
            swap_columns(c, s, n, i, working);
      }
      for (size_t j = 0; j < s->operations[i].count; j++)
         emit_row(c, i, &s->operations[i].data[j]);
   }
   for (unsigned i = s->rows; i < n; i++) {
      select_register(c, i);
      emit(c, 17, 0, 0, 0, 0, 0);
   }
   return true;
}

static bool
eligible(unsigned opcode)
{
   return opcode - 5 < 16 || opcode <= 3;
}

static unsigned
rule(unsigned index, const struct pan_graph_tsu_instr *a, unsigned sa,
     const struct pan_graph_tsu_instr *b, unsigned sb)
{
   switch (index) {
   case 0:
      return b->opcode == 4 && eligible(a->opcode) && !a->increment &&
             ((sa + 1) & 7) == sb ? 4 : 0;
   case 1:
      return b->opcode == 4 && a->increment == 1 ? 6 : 0;
   case 2:
      return a->opcode == 4 && (b->opcode == 4 || b->opcode == 21);
   case 3:
      return b->opcode == 4 && sa == sb ? 2 : 0;
   case 4:
      return b->opcode == 7 && !b->flag && !b->immediate && !b->shift &&
             !a->increment && a->flag == 1 && a->opcode >= 6 && a->opcode <= 10
                ? 5 : 0;
   case 5:
      return a->opcode == b->opcode && (a->opcode == 1 || a->opcode == 2) &&
             !a->increment && !b->increment &&
             (a->opcode == 1 ? a->reg == b->reg : a->immediate == b->immediate)
                ? 3 : 0;
   case 6:
      return a->opcode == 4 && b->opcode == 21;
   case 7:
      return eligible(a->opcode) && a->opcode != 1 && !a->increment && !a->flag &&
             eligible(b->opcode) && !b->flag &&
             (b->opcode == 17 || (b->opcode == 0 && b->reg != sa));
   case 8:
      return a->opcode == 1 &&
             ((a->reg == sa && a->increment != 1) ||
              (b->opcode == 21 && a->reg == (a->increment == 1 ? (sa - 1) & 7 : sa)));
   case 9:
      return a->opcode == 2 &&
             ((a->immediate == sa && a->increment != 1) ||
              (b->opcode == 21 && a->immediate ==
                 (a->increment == 1 ? (sa - 1) & 7 : sa)));
   default:
      return 0;
   }
}

static bool
optimize(struct compiler *c)
{
   if (!c->count)
      return true;
   if (c->count > SIZE_MAX / sizeof(unsigned))
      return false;
   unsigned *states = malloc(c->count * sizeof(*states));
   if (!states)
      return false;
   size_t count = 0;
   unsigned current = 0;
   for (size_t i = 0; i < c->count; i++) {
      struct pan_graph_tsu_instr raw = c->instructions[i];
      if (raw.opcode == 4)
         current = raw.reg;
      else if (raw.increment == 1)
         current = (current + 1) & 7;
      c->instructions[count] = raw;
      states[count++] = current;
      bool changed = true;
      while (changed && count >= 2) {
         changed = false;
         for (unsigned r = 0; r < 10 && count >= 2; r++) {
            struct pan_graph_tsu_instr *a = &c->instructions[count - 2];
            struct pan_graph_tsu_instr *b = &c->instructions[count - 1];
            unsigned action = rule(r, a, states[count - 2], b, states[count - 1]);
            if (!action)
               continue;
            changed = true;
            if (action == 6) {
               a->increment = 0;
               states[count - 2] = (states[count - 2] - 1) & 7;
               continue;
            }
            unsigned last = states[count - 1];
            states[count - 2] = last;
            if (action == 1)
               *a = *b;
            else if (action == 4)
               a->increment = 1;
            else if (action == 5)
               a->flag = 0;
            count--;
            if (action == 3) {
               count--;
               if (count)
                  states[count - 1] = last;
            }
         }
      }
   }
   free(states);
   c->count = count;
   return true;
}

enum pan_graph_tsu_status
pan_graph_tsu_compile(const struct pan_graph_tsu_info *info,
                      struct pan_graph_tsu_program *program)
{
   if (!info || !program || !info->rank || info->rank > PAN_GRAPH_TSU_MAX_RANK ||
       !info->low || !info->high || (info->count && !info->transforms))
      return PAN_GRAPH_TSU_INVALID;
   struct compiler c = {.rank = info->rank, .initial_rank = info->rank};
   memcpy(c.low, info->low, c.rank * sizeof(c.low[0]));
   memcpy(c.high, info->high, c.rank * sizeof(c.high[0]));
   for (unsigned i = 0; i < c.rank; i++) {
      if (c.low[i] > c.high[i])
         return PAN_GRAPH_TSU_INVALID;
   }
   for (size_t i = 0; i < info->count; i++) {
      if (!plan(&c, &info->transforms[i])) {
         if (c.status == PAN_GRAPH_TSU_SUCCESS)
            c.status = PAN_GRAPH_TSU_INVALID;
         break;
      }
   }
   for (size_t i = 0; i < c.stage_count && c.status == PAN_GRAPH_TSU_SUCCESS; i++) {
      if (!schedule_stage(&c, &c.stages[i]))
         c.status = PAN_GRAPH_TSU_UNSUPPORTED;
   }
   if (c.status == PAN_GRAPH_TSU_SUCCESS) {
      emit(&c, 21, 0, 0, 0, 0, 0);
      if (c.status == PAN_GRAPH_TSU_SUCCESS && !optimize(&c))
         c.status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   }
   for (size_t i = 0; i < c.stage_count; i++) {
      for (unsigned j = 0; j < PAN_GRAPH_TSU_MAX_RANK; j++)
         free(c.stages[i].operations[j].data);
   }
   free(c.stages);
   if (c.status == PAN_GRAPH_TSU_SUCCESS)
      *program = (struct pan_graph_tsu_program){c.count, c.instructions};
   else
      free(c.instructions);
   return c.status;
}

void
pan_graph_tsu_cleanup(struct pan_graph_tsu_program *program)
{
   free(program->instructions);
   *program = (struct pan_graph_tsu_program){0};
}

static bool
equal_instruction(const struct pan_graph_tsu_instr *a,
                  const struct pan_graph_tsu_instr *b)
{
   return a->opcode == b->opcode && a->increment == b->increment &&
          a->flag == b->flag && a->reg == b->reg &&
          a->immediate == b->immediate && a->shift == b->shift;
}

static uint32_t
find_program(const struct compiler *pool, const struct pan_graph_tsu_program *p)
{
   if (!pool->count || pool->count < p->count)
      return UINT32_MAX;
   if (!p->count)
      return 0;
   uint32_t offset = 0;
   for (size_t i = 0; i <= pool->count - p->count; i++) {
      for (size_t j = 0; j < p->count; j++) {
         if (!equal_instruction(&pool->instructions[i + j], &p->instructions[j]))
            break;
         if (p->instructions[j].opcode == 21 || j + 1 == p->count)
            return offset;
      }
      if (pool->instructions[i].size > UINT32_MAX - offset)
         return UINT32_MAX;
      offset += pool->instructions[i].size;
   }
   return UINT32_MAX;
}

enum pan_graph_tsu_status
pan_graph_tsu_link(const struct pan_graph_tsu_info *infos, size_t count,
                   struct pan_graph_tsu_program *program, uint32_t *offsets)
{
   if (!program || (count && (!infos || !offsets)))
      return PAN_GRAPH_TSU_INVALID;
   if (count > SIZE_MAX / sizeof(*offsets))
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   uint32_t *result = count ? malloc(count * sizeof(*result)) : NULL;
   if (count && !result)
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   struct compiler pool = {0};
   uint64_t bytes = 0;
   for (size_t i = 0; i < count; i++) {
      struct pan_graph_tsu_program p = {0};
      pool.status = pan_graph_tsu_compile(&infos[i], &p);
      if (pool.status != PAN_GRAPH_TSU_SUCCESS)
         break;
      size_t length = 0;
      for (size_t j = 0; j < p.count; j++)
         length += p.instructions[j].size;
      unsigned padding = (4 - (length & 3)) & 3;
      size_t capacity = p.count;
      if (p.count > SIZE_MAX - padding ||
          !reserve(&pool, (void **)&p.instructions, &capacity, p.count + padding,
                   sizeof(*p.instructions))) {
         pan_graph_tsu_cleanup(&p);
         pool.status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
         break;
      }
      for (unsigned j = 0; j < padding; j++)
         p.instructions[p.count++] = (struct pan_graph_tsu_instr){.opcode = 21, .size = 1};
      uint32_t offset = find_program(&pool, &p);
      if (offset == UINT32_MAX) {
         if (bytes + length + padding > UINT32_MAX ||
             p.count > SIZE_MAX - pool.count) {
            pan_graph_tsu_cleanup(&p);
            pool.status = PAN_GRAPH_TSU_UNSUPPORTED;
            break;
         }
         offset = bytes;
         if (!reserve(&pool, (void **)&pool.instructions, &pool.capacity,
                      pool.count + p.count, sizeof(*pool.instructions))) {
            pan_graph_tsu_cleanup(&p);
            break;
         }
         memcpy(pool.instructions + pool.count, p.instructions,
                p.count * sizeof(*p.instructions));
         pool.count += p.count;
         bytes += length + padding;
      }
      result[i] = offset;
      pan_graph_tsu_cleanup(&p);
   }
   if (pool.status == PAN_GRAPH_TSU_SUCCESS) {
      *program = (struct pan_graph_tsu_program){pool.count, pool.instructions};
      if (count)
         memcpy(offsets, result, count * sizeof(*result));
   } else {
      free(pool.instructions);
   }
   free(result);
   return pool.status;
}
