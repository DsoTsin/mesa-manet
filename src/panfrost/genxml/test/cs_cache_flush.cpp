#include "cs_builder.h"

#include <array>
#include <initializer_list>
#include <gtest/gtest.h>

class CsCacheFlushTest : public ::testing::Test {
protected:
   struct cs_builder b;
   std::array<uint64_t, 32> output{};

   CsCacheFlushTest()
   {
      struct cs_builder_conf conf = {
         .nr_registers = 96,
         .nr_kernel_registers = 4,
      };
      struct cs_buffer buffer = {
         .cpu = output.data(),
         .gpu = 0,
         .capacity = static_cast<uint32_t>(output.size()),
      };
      cs_builder_init(&b, &conf, buffer);
   }

   ~CsCacheFlushTest()
   {
      cs_builder_fini(&b);
   }

   void expect(std::initializer_list<uint64_t> expected)
   {
      cs_end(&b);
      ASSERT_EQ(b.root_chunk.size, expected.size());
      unsigned index = 0;
      for (uint64_t word : expected) {
         EXPECT_EQ(output[index], word) << index;
         index++;
      }
   }
};

TEST_F(CsCacheFlushTest, LegacyWrapperPreservesEncoding)
{
   cs_flush_caches(&b, MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
                   MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
                   MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, cs_reg32(&b, 84),
                   cs_defer(0, 0));
   expect({0x2400540000000233});
}

TEST_F(CsCacheFlushTest, ExplicitFalsePreservesEncoding)
{
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
      MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, false, cs_reg32(&b, 84),
      cs_defer(0, 0));
   expect({0x2400540000000233});
}

#if PAN_ARCH >= 15
TEST_F(CsCacheFlushTest, VendorPreambleAndCompletionWait)
{
   cs_move32_to(&b, cs_reg32(&b, 84), 0x1234);
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
      MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, true, cs_reg32(&b, 84),
      cs_defer(0, 0));
   cs_wait_slot(&b, 0);
   expect({0x0254000000001234, 0x2400540000008233, 0x0300000000010000});
}

TEST_F(CsCacheFlushTest, FullWaitBeforeTailAndFlushCompletion)
{
   cs_wait_slots(&b, 0xffff);
   cs_move32_to(&b, cs_reg32(&b, 73), 0);
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, true, cs_reg32(&b, 73),
      cs_defer(0, 0));
   cs_wait_slot(&b, 0);
   expect({0x03000000ffff0000, 0x0249000000000000,
           0x2400490000008211, 0x0300000000010000});
}

TEST_F(CsCacheFlushTest, NeuralOnlyMagniSequence)
{
   cs_wait_slots(&b, 0xfffc);
   cs_move32_to(&b, cs_reg32(&b, 84), 0);
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_NONE, MALI_CS_FLUSH_MODE_NONE,
      MALI_CS_OTHER_FLUSH_MODE_NONE, true, cs_reg32(&b, 84), cs_defer(0, 0));
   cs_wait_slot(&b, 0);
   expect({0x03000000fffc0000, 0x0254000000000000,
           0x2400540000008000, 0x0300000000010000});
}

TEST_F(CsCacheFlushTest, DirectDeferredMaskAndCompletionSlot)
{
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, true, cs_reg32(&b, 84),
      cs_defer(0xfffb, 2));
   cs_wait_slot(&b, 2);
   expect({0x24025400fffb8211, 0x0300000000040000});
}

TEST_F(CsCacheFlushTest, IndirectDeferredStateAndCompletionSlot)
{
   cs_move32_to(&b, cs_reg32(&b, 20), 0xfffb);
   cs_set_state(&b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, cs_reg32(&b, 20));
   cs_set_state_imm32(&b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, 2);
   cs_flush_caches_with_neural(
      &b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, true, cs_reg32(&b, 73),
      cs_defer_indirect());
   cs_wait_slot(&b, 2);
   expect({0x021400000000fffb, 0x1b00140900000000,
           0x1c00000200000002, 0x2410490000008211,
           0x0300000000040000});
}
#elif !defined(NDEBUG)
TEST_F(CsCacheFlushTest, RejectsNeuralOnOlderArchitecture)
{
   EXPECT_DEATH_IF_SUPPORTED(
      cs_flush_caches_with_neural(
         &b, MALI_CS_FLUSH_MODE_NONE, MALI_CS_FLUSH_MODE_NONE,
         MALI_CS_OTHER_FLUSH_MODE_NONE, true, cs_reg32(&b, 84), cs_defer(0, 0)),
      "!neural");
}
#endif
