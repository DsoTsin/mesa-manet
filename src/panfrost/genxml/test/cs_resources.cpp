#include "cs_builder.h"

#include <array>
#include <gtest/gtest.h>

class CsResourceTest : public ::testing::Test {
protected:
   struct cs_builder b;
   std::array<uint64_t, 4> output{};

   CsResourceTest()
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

   ~CsResourceTest()
   {
      cs_builder_fini(&b);
   }

   void expect(uint32_t resources, uint64_t word)
   {
      cs_req_res(&b, resources);
      cs_end(&b);
      ASSERT_EQ(b.root_chunk.size, 1);
      EXPECT_EQ(output[0], word);
   }
};

TEST_F(CsResourceTest, Compute)
{
   expect(CS_COMPUTE_RES, 0x2200000000000001);
}

TEST_F(CsResourceTest, Release)
{
   expect(0, 0x2200000000000000);
}

TEST_F(CsResourceTest, GraphicsAndCompute)
{
   expect(CS_COMPUTE_RES | CS_FRAG_RES | CS_TILER_RES | CS_IDVS_RES,
          0x220000000000000f);
}

#if PAN_ARCH >= 15
TEST_F(CsResourceTest, Neural)
{
   expect(CS_NEURAL_RES, 0x2200000000000020);
}

TEST_F(CsResourceTest, NeuralAndCompute)
{
   expect(CS_NEURAL_RES | CS_COMPUTE_RES, 0x2200000000000021);
}

TEST_F(CsResourceTest, NeuralAndRayTraversal)
{
   expect(CS_NEURAL_RES | CS_RT_RES, 0x2200000000000030);
}

TEST_F(CsResourceTest, VendorNeuralDispatch)
{
   cs_run_neural(&b, 0, 16, 36);
   cs_end(&b);
   ASSERT_EQ(b.root_chunk.size, 1);
   EXPECT_EQ(output[0], 0x0d00000024100000);
   EXPECT_EQ(b.req_resource_mask, CS_NEURAL_RES);
}

TEST_F(CsResourceTest, NeuralDispatchFields)
{
   cs_run_neural(&b, 32, 4, 8);
   cs_end(&b);
   ASSERT_EQ(b.root_chunk.size, 1);
   EXPECT_EQ(output[0], 0x0d20000008040000);
}
#endif
