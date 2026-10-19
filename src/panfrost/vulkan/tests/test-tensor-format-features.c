#include <assert.h>
#include <stdio.h>

#include "panvk_physical_device.h"
#include "panvk_tensor.h"

int
main(void)
{
   const VkFormat formats[] = {
      VK_FORMAT_R8_UINT,
      VK_FORMAT_R8_SINT,
      VK_FORMAT_R8_BOOL_ARM,
      VK_FORMAT_R16_UINT,
      VK_FORMAT_R16_SINT,
      VK_FORMAT_R16_SFLOAT,
      VK_FORMAT_R32_UINT,
      VK_FORMAT_R32_SINT,
      VK_FORMAT_R32_SFLOAT,
      VK_FORMAT_R64_UINT,
      VK_FORMAT_R64_SINT,
      VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM,
      VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM,
      VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM,
      VK_FORMAT_UNDEFINED,
      VK_FORMAT_R8G8_UINT,
      VK_FORMAT_R16G16_SINT,
      VK_FORMAT_R8G8B8A8_UNORM,
   };
   const VkTensorTilingARM tilings[] = {
      VK_TENSOR_TILING_LINEAR_ARM,
      VK_TENSOR_TILING_OPTIMAL_ARM,
      VK_TENSOR_TILING_BRICK_16_WIDE_ARM,
      VK_TENSOR_TILING_BRICK_8_WIDE_ARM,
      VK_TENSOR_TILING_BRICK_4_WIDE_ARM,
      VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_ARM,
      VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_64K_ARM,
   };
   unsigned checks = 0;
   for (unsigned tensors = 0; tensors < 2; tensors++) {
      for (unsigned graph = 0; graph < 2; graph++) {
         struct panvk_physical_device pdev = {0};
         pdev.vk.supported_extensions.ARM_tensors = tensors;
         pdev.vk.supported_features.dataGraph = graph;
         for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
            for (unsigned t = 0; t < ARRAY_SIZE(tilings); t++) {
               VkFormat format = formats[f];
               bool supported = format == VK_FORMAT_R8_UINT ||
                                format == VK_FORMAT_R8_SINT ||
                                format == VK_FORMAT_R32_UINT ||
                                format == VK_FORMAT_R32_SINT ||
                                format == VK_FORMAT_R32_SFLOAT;
               bool linear = tilings[t] == VK_TENSOR_TILING_LINEAR_ARM ||
                             tilings[t] == VK_TENSOR_TILING_OPTIMAL_ARM;
               VkFormatFeatureFlags2 flags = panvk_get_tensor_format_features(
                  &pdev, format, tilings[t]);
               bool actual = flags & VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM;
               bool expected = tensors && graph && supported && linear;
               if (actual != expected) {
                  fprintf(stderr,
                          "tensor graph format=%u tiling=%u tensors=%u graph=%u flags=%llu\n",
                          format, tilings[t], tensors, graph,
                          (unsigned long long)flags);
               }
               assert(actual == expected);
               if (!tensors)
                  assert(!flags);
               checks++;
            }
         }
      }
   }
   printf("tensor graph format checks=%u\n", checks);
   return 0;
}
