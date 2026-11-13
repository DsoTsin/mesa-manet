#include "agc_cb.h"
#include "test.h"

struct grow_log {
   unsigned calls;
   uint32_t last_dwords;
   uint32_t *next;
   uint32_t next_dwords;
   bool answer;
};

static bool AGC_SYSV
grow(struct agc_cb *cb, uint32_t dwords, void *user_data)
{
   struct grow_log *log = user_data;
   log->calls++;
   log->last_dwords = dwords;
   if (log->answer && log->next) {
      cb->bottom = log->next;
      cb->top = log->next + log->next_dwords;
      cb->up = cb->bottom;
      cb->down = cb->top;
   }
   return log->answer;
}

static void
test_cursor(void)
{
   _Alignas(64) uint32_t buffer[64];
   struct agc_cb cb;
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);
   CHECK(cb.bottom == buffer && cb.up == buffer);
   CHECK(cb.top == buffer + 64 && cb.down == buffer + 64);
   CHECK_EQ(agc_cb_remaining_dwords(&cb), 64);

   uint32_t *p = agc_cb_alloc_up(&cb, 5);
   CHECK(p == buffer);
   CHECK(cb.up == buffer + 5);
   CHECK_EQ(agc_cb_used_up_bytes(&cb), 20);
   CHECK_EQ(agc_cb_remaining_dwords(&cb), 59);

   cb.reserved_dwords = 16;
   CHECK_EQ(agc_cb_remaining_dwords(&cb), 43);
   CHECK(agc_cb_alloc_up(&cb, 44) == NULL);
   CHECK(cb.up == buffer + 5);
   CHECK(agc_cb_alloc_up(&cb, 43) == buffer + 5);
   CHECK_EQ(agc_cb_remaining_dwords(&cb), 0);
   cb.reserved_dwords = 200;
   CHECK_EQ(agc_cb_remaining_dwords(&cb), 0);

   agc_cb_reset(&cb);
   CHECK(cb.up == buffer && cb.down == buffer + 64);
}

static void
test_alloc_down(void)
{
   _Alignas(64) uint32_t buffer[64];
   struct agc_cb cb;
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   void *a = agc_cb_alloc_down(&cb, 6, 4);
   CHECK(a == buffer + 62);
   CHECK(cb.down == buffer + 62);
   CHECK_EQ(agc_cb_used_down_bytes(&cb), 8);

   void *b = agc_cb_alloc_down(&cb, 4, 16);
   CHECK(b == buffer + 60);
   CHECK(((uintptr_t)b & 15) == 0);

   void *c = agc_cb_alloc_down(&cb, 4, 64);
   CHECK(c == buffer + 48);
   CHECK(agc_cb_alloc_down(&cb, 0, 4) == NULL);
   CHECK(cb.down == buffer + 48);

   uint32_t *up = agc_cb_alloc_up(&cb, 1);
   CHECK(up == buffer);
   void *d = agc_cb_alloc_up_aligned(&cb, 8, 16);
   CHECK(d == buffer + 4);
   CHECK(cb.up == buffer + 6);
   CHECK(agc_cb_alloc_up_aligned(&cb, 4 * 41, 4) == buffer + 6);
   CHECK(agc_cb_alloc_up_aligned(&cb, 8, 4) == NULL);
   CHECK(agc_cb_alloc_down(&cb, 4, 4) == buffer + 47);
   CHECK(agc_cb_alloc_down(&cb, 4, 4) == NULL);
}

static void
test_grow(void)
{
   _Alignas(64) uint32_t first[8];
   _Alignas(64) uint32_t second[32];
   struct grow_log log = {.next = second, .next_dwords = 32, .answer = true};
   struct agc_cb cb;
   agc_cb_init(&cb, first, sizeof(first), grow, &log);
   cb.reserved_dwords = 2;

   CHECK(agc_cb_alloc_up(&cb, 6) == first);
   CHECK_EQ(log.calls, 0);
   uint32_t *p = agc_cb_alloc_up(&cb, 10);
   CHECK_EQ(log.calls, 1);
   CHECK_EQ(log.last_dwords, 12);
   CHECK(p == second);
   CHECK(cb.up == second + 10);

   log.next = NULL;
   CHECK(agc_cb_alloc_up(&cb, 40) == NULL);
   CHECK_EQ(log.calls, 2);
   CHECK_EQ(log.last_dwords, 42);

   log.answer = false;
   CHECK(agc_cb_alloc_down(&cb, 4 * 40, 4) == NULL);
   CHECK_EQ(log.calls, 3);
   CHECK(cb.up == second + 10);
}

int
main(void)
{
   test_cursor();
   test_alloc_down();
   test_grow();
   return TEST_RESULT();
}
