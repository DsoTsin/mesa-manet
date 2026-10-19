#define main tensor_test_main
#include "../../tests/tensor-device.c"
#undef main
#include <math.h>
#include <stdbool.h>

FN(CreateBuffer);
FN(DestroyBuffer);
FN(GetBufferMemoryRequirements);
FN(BindBufferMemory);
FN(CmdPipelineBarrier2);

struct graph_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceSize size;
   float *map;
};

static struct graph_buffer
make_graph_buffer(unsigned elements)
{
   struct graph_buffer b = {0};
   VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = elements * sizeof(float),
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   CHECK(CreateBuffer(device, &ci, NULL, &b.buffer));
   VkMemoryRequirements req;
   GetBufferMemoryRequirements(device, b.buffer, &req);
   uint32_t type = UINT32_MAX;
   for (unsigned i = 0; i < memory.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if ((req.memoryTypeBits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & flags) == flags) {
         type = i;
         break;
      }
   }
   if (type == UINT32_MAX)
      exit(1);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = req.size + 64,
                              .memoryTypeIndex = type};
   b.size = ai.allocationSize;
   CHECK(AllocateMemory(device, &ai, NULL, &b.memory));
   CHECK(BindBufferMemory(device, b.buffer, b.memory, 0));
   CHECK(MapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, (void **)&b.map));
   memset(b.map, 0xcd, b.size);
   return b;
}

int
main(int argc, char **argv)
{
   if (argc != 4)
      return 2;
   bool matmul = !strcmp(argv[3], "matmul");
   bool chain = !strcmp(argv[3], "chain");
   if (!matmul && !chain && strcmp(argv[3], "add"))
      return 2;
   setvbuf(stdout, NULL, _IONBF, 0);
   init(argv[1], 0);
   LOAD(CreateBuffer);
   LOAD(DestroyBuffer);
   LOAD(GetBufferMemoryRequirements);
   LOAD(BindBufferMemory);
   LOAD(CmdPipelineBarrier2);
   unsigned count[3] = {matmul ? 30 : 57, matmul ? 40 : 19, matmul ? 24 : 57};
   struct graph_buffer buffers[3];
   VkDescriptorSetLayoutBinding bindings[3];
   for (unsigned i = 0; i < 3; i++) {
      buffers[i] = make_graph_buffer(count[i]);
      bindings[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
      if (i < 2)
         for (unsigned j = 0; j < count[i]; j++)
            buffers[i].map[j] = ((int)((j * 13 + i * 7) % 41) - 20) * 0.125f;
   }
   VkDescriptorSetLayoutCreateInfo sli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(CreateDescriptorSetLayout(device, &sli, NULL, &sl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &sl};
   VkPipelineLayout layout;
   CHECK(CreatePipelineLayout(device, &pli, NULL, &layout));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
   VkDescriptorPoolCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
   VkDescriptorPool pool;
   CHECK(CreateDescriptorPool(device, &pi, NULL, &pool));
   VkDescriptorSetAllocateInfo sai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &sl};
   VkDescriptorSet set;
   CHECK(AllocateDescriptorSets(device, &sai, &set));
   for (unsigned i = 0; i < 3; i++) {
      VkDescriptorBufferInfo bi = {.buffer = buffers[i].buffer,
                                   .range = count[i] * sizeof(float)};
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi};
      UpdateDescriptorSets(device, 1, &w, 0, NULL);
   }
   FILE *file = fopen(argv[2], "rb");
   if (!file)
      return 1;
   fseek(file, 0, SEEK_END);
   long size = ftell(file);
   if (size < 20 || size % 4)
      return 1;
   rewind(file);
   uint32_t *code = malloc(size);
   if (!code || fread(code, 1, size, file) != size)
      return 1;
   fclose(file);
   VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code};
   VkShaderModule module;
   CHECK(CreateShaderModule(device, &mi, NULL, &module));
   free(code);
   VkComputePipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "partition_0_kernel"},
      .layout = layout};
   VkPipeline pipeline;
   CHECK(
      CreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline));
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family};
   VkCommandPool cp;
   CHECK(CreateCommandPool(device, &cpi, NULL, &cp));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(AllocateCommandBuffers(device, &cai, &cmd));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(BeginCommandBuffer(cmd, &begin));
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                         &set, 0, NULL);
   CmdDispatch(cmd, matmul ? 4 : 19, 3, matmul ? 2 : 1);
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
   VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                           .memoryBarrierCount = 1,
                           .pMemoryBarriers = &mb};
   CmdPipelineBarrier2(cmd, &dep);
   CHECK(EndCommandBuffer(cmd));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(QueueWaitIdle(queue));
   unsigned failures = 0;
   unsigned char *reference = malloc(buffers[2].size);
   memset(reference, 0xcd, buffers[2].size);
   for (unsigned i = 0; i < count[2]; i++) {
      float expected;
      if (matmul) {
         unsigned b = i / 12, m = i / 4 % 3, col = i % 4;
         expected = 0;
         for (unsigned k = 0; k < 5; k++)
            expected += buffers[0].map[b * 15 + m * 5 + k] *
                        buffers[1].map[b * 20 + k * 4 + col];
      } else {
         expected = buffers[0].map[i] + buffers[1].map[i % 19];
         if (chain)
            expected = fminf(fmaxf(expected, 0.0f), 1.0f);
      }
      memcpy(reference + i * sizeof(float), &expected, sizeof(expected));
      float actual = buffers[2].map[i];
      if (!isfinite(actual) || fabsf(actual - expected) > 1e-5f)
         failures++;
   }
   if (memcmp(reference + count[2] * sizeof(float),
              (char *)buffers[2].map + count[2] * sizeof(float),
              buffers[2].size - count[2] * sizeof(float)))
      failures++;
   free(reference);
   printf("KERNEL %s checks=%u failures=%u\n", argv[3], count[2], failures);
   DestroyCommandPool(device, cp, NULL);
   DestroyPipeline(device, pipeline, NULL);
   DestroyShaderModule(device, module, NULL);
   DestroyDescriptorPool(device, pool, NULL);
   DestroyPipelineLayout(device, layout, NULL);
   DestroyDescriptorSetLayout(device, sl, NULL);
   for (unsigned i = 0; i < 3; i++) {
      DestroyBuffer(device, buffers[i].buffer, NULL);
      UnmapMemory(device, buffers[i].memory);
      FreeMemory(device, buffers[i].memory, NULL);
   }
   DestroyDevice(device, NULL);
   return failures ? 1 : 0;
}
