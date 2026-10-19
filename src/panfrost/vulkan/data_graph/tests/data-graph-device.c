#define main tensor_test_main
#include "../../tests/tensor-device.c"
#undef main
#include <math.h>

FN(CreateDataGraphPipelinesARM);
FN(CreateDataGraphPipelineSessionARM);
FN(DestroyDataGraphPipelineSessionARM);
FN(GetDataGraphPipelineSessionBindPointRequirementsARM);
FN(GetDataGraphPipelineSessionMemoryRequirementsARM);
FN(BindDataGraphPipelineSessionMemoryARM);
FN(CmdDispatchDataGraphARM);
FN(GetDataGraphPipelineAvailablePropertiesARM);
FN(GetDataGraphPipelinePropertiesARM);
FN(CmdPipelineBarrier2);

int
main(int argc, char **argv)
{
   if (argc < 4)
      return 2;
   setvbuf(stdout, NULL, _IONBF, 0);
   init(argv[1], 0);
   tensor_extra_usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
   LOAD(CreateDataGraphPipelinesARM);
   LOAD(CreateDataGraphPipelineSessionARM);
   LOAD(DestroyDataGraphPipelineSessionARM);
   LOAD(GetDataGraphPipelineSessionBindPointRequirementsARM);
   LOAD(GetDataGraphPipelineSessionMemoryRequirementsARM);
   LOAD(BindDataGraphPipelineSessionMemoryARM);
   LOAD(CmdDispatchDataGraphARM);
   LOAD(GetDataGraphPipelineAvailablePropertiesARM);
   LOAD(GetDataGraphPipelinePropertiesARM);
   LOAD(CmdPipelineBarrier2);
   PFN_vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM query =
      (void *)tensor_gipa(
         instance, "vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM");
   uint32_t engines = 0;
   CHECK(query(physical, family, &engines, NULL));
   VkQueueFamilyDataGraphPropertiesARM *engine_props =
      calloc(engines, sizeof(*engine_props));
   for (unsigned i = 0; i < engines; i++)
      engine_props[i].sType =
         VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
   CHECK(query(physical, family, &engines, engine_props));
   for (unsigned i = 0; i < engines; i++)
      printf("ENGINE type=%d foreign=%u operation=%d name=%s version=%u\n",
             engine_props[i].engine.type, engine_props[i].engine.isForeign,
             engine_props[i].operation.operationType,
             engine_props[i].operation.name, engine_props[i].operation.version);
   free(engine_props);
   int matmul = !strcmp(argv[3], "matmul"), chain = !strcmp(argv[3], "chain");
   const char *spec_option = getenv("DG_TEST_SPECIALIZATION");
   const char *columns_option = getenv("DG_TEST_COLUMNS");
   unsigned columns = columns_option ? strtoul(columns_option, NULL, 10) : 19;
   if (!columns || columns > 4096)
      return 2;
   float clamp_max = spec_option && strcmp(spec_option, "default") ?
      strtof(spec_option, NULL) : 1.0f;
   if (!isfinite(clamp_max) || clamp_max < 0)
      return 2;
   unsigned rank = matmul ? 3 : 2;
   int64_t dims[3][4] = {{3, columns}, {1, columns}, {3, columns}};
   if (matmul) {
      int64_t shapes[3][4] = {{2, 3, 5}, {2, 5, 4}, {2, 3, 4}};
      memcpy(dims, shapes, sizeof(dims));
   }
   struct tensor tensors[3];
   unsigned total[3] = {1, 1, 1};
   VkTensorDescriptionARM desc[3];
   VkDataGraphPipelineResourceInfoARM resources[3];
   for (unsigned i = 0; i < 3; i++) {
      tensors[i] = make_tensor(VK_FORMAT_R32_SFLOAT, rank, dims[i], NULL);
      for (unsigned d = 0; d < rank; d++)
         total[i] *= dims[i][d];
      desc[i] = (VkTensorDescriptionARM){
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .format = VK_FORMAT_R32_SFLOAT,
         .dimensionCount = rank,
         .pDimensions = dims[i],
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};
      resources[i] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = &desc[i],
         .descriptorSet = 0,
         .binding = i};
      if (i < 2)
         for (unsigned j = 0; j < total[i]; j++)
            ((float *)tensors[i].map)[j] =
               ((int)((j * 13 + i * 7) % 41) - 20) * 0.125f;
   }
   VkDescriptorSetLayoutBinding bindings[3];
   for (unsigned i = 0; i < 3; i++)
      bindings[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_ALL};
   VkDescriptorSetLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(CreateDescriptorSetLayout(device, &li, NULL, &sl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &sl};
   VkPipelineLayout layout;
   CHECK(CreatePipelineLayout(device, &pli, NULL, &layout));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_TENSOR_ARM, 3};
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
      VkWriteDescriptorSetTensorARM tw = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
         .tensorViewCount = 1,
         .pTensorViews = &tensors[i].view};
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = &tw,
         .dstSet = set,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM};
      UpdateDescriptorSets(device, 1, &w, 0, NULL);
   }
   FILE *f = fopen(argv[2], "rb");
   if (!f)
      return 1;
   fseek(f, 0, SEEK_END);
   size_t size = ftell(f);
   rewind(f);
   uint32_t *code = malloc(size);
   if (fread(code, 1, size, f) != size)
      exit(1);
   fclose(f);
   VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code};
   VkShaderModule module;
   CHECK(CreateShaderModule(device, &mi, NULL, &module));
   free(code);
   VkDataGraphPipelineShaderModuleCreateInfoARM shader = {
      .sType =
         VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .module = module,
      .pName = "main"};
   uint8_t spec_data[13] = {0};
   uint32_t spec_columns = columns - 1;
   const char *boolean_option = getenv("DG_TEST_BOOLEAN");
   uint32_t spec_boolean = boolean_option ? strtoul(boolean_option, NULL, 10) : 1;
   memcpy(spec_data + 1, &clamp_max, sizeof(clamp_max));
   memcpy(spec_data + 5, &spec_columns, sizeof(spec_columns));
   memcpy(spec_data + 9, &spec_boolean, sizeof(spec_boolean));
   const VkSpecializationMapEntry spec_entries[] = {
      {.constantID = 42, .offset = 1, .size = 4},
      {.constantID = 43, .offset = 5, .size = 4},
      {.constantID = 999, .offset = 1, .size = 4},
      {.constantID = 44, .offset = 9, .size = 4},
   };
   const VkSpecializationInfo spec = {
      .mapEntryCount = 4,
      .pMapEntries = spec_entries,
      .dataSize = sizeof(spec_data),
      .pData = spec_data,
   };
   if (spec_option && strcmp(spec_option, "default"))
      shader.pSpecializationInfo = &spec;
   if (shader.pSpecializationInfo && !spec_boolean)
      clamp_max = 0;
   VkDataGraphPipelineCreateInfoARM ci = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .pNext = &shader,
      .layout = layout,
      .resourceInfoCount = 3,
      .pResourceInfos = resources};
   VkPipeline pipeline;
   CHECK(CreateDataGraphPipelinesARM(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                     &ci, NULL, &pipeline));
   VkDataGraphPipelineInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_INFO_ARM,
      .dataGraphPipeline = pipeline};
   uint32_t n = 0;
   CHECK(GetDataGraphPipelineAvailablePropertiesARM(device, &info, &n, NULL));
   VkDataGraphPipelinePropertyARM *props = calloc(n, sizeof(*props));
   CHECK(GetDataGraphPipelineAvailablePropertiesARM(device, &info, &n, props));
   for (unsigned i = 0; i < n; i++) {
      VkDataGraphPipelinePropertyQueryResultARM property = {
         .sType =
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT_ARM,
         .property = props[i]};
      CHECK(GetDataGraphPipelinePropertiesARM(device, &info, 1, &property));
      property.pData = calloc(property.dataSize + 1, 1);
      CHECK(GetDataGraphPipelinePropertiesARM(device, &info, 1, &property));
      printf("PROPERTY %d size=%zu text=%u\n", props[i], property.dataSize,
             property.isText);
      if (property.isText)
         printf("%s\n", (char *)property.pData);
      if (argc > 4) {
         char path[512];
         snprintf(path, sizeof(path), "%s/property-%d.bin", argv[4], props[i]);
         FILE *p = fopen(path, "wb");
         if (p) {
            fwrite(property.pData, 1, property.dataSize, p);
            fclose(p);
         }
      }
      free(property.pData);
   }
   free(props);
   VkDataGraphPipelineSessionCreateInfoARM sci = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
      .dataGraphPipeline = pipeline};
   VkDataGraphPipelineSessionARM session;
   CHECK(CreateDataGraphPipelineSessionARM(device, &sci, NULL, &session));
   VkDataGraphPipelineSessionBindPointRequirementsInfoARM sri = {
      .sType =
         VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
      .session = session};
   n = 0;
   CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &sri, &n,
                                                             NULL));
   VkDataGraphPipelineSessionBindPointRequirementARM *reqs =
      calloc(n, sizeof(*reqs));
   for (unsigned i = 0; i < n; i++)
      reqs[i].sType =
         VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
   CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &sri, &n,
                                                             reqs));
   VkDeviceMemory *allocations = NULL;
   unsigned allocations_count = 0;
   for (unsigned i = 0; i < n; i++)
      for (unsigned j = 0; j < reqs[i].numObjects; j++) {
         VkDataGraphPipelineSessionMemoryRequirementsInfoARM ri = {
            .sType =
               VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
            .session = session,
            .bindPoint = reqs[i].bindPoint,
            .objectIndex = j};
         VkMemoryRequirements2 mr = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
         GetDataGraphPipelineSessionMemoryRequirementsARM(device, &ri, &mr);
         printf("SESSION bind=%d object=%u size=%llu align=%llu types=%x\n",
                reqs[i].bindPoint, j,
                (unsigned long long)mr.memoryRequirements.size,
                (unsigned long long)mr.memoryRequirements.alignment,
                mr.memoryRequirements.memoryTypeBits);
         uint32_t mt = 0;
         while (!(mr.memoryRequirements.memoryTypeBits & (1u << mt)))
            mt++;
         VkMemoryAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.memoryRequirements.size,
            .memoryTypeIndex = mt};
         VkDeviceMemory mem;
         CHECK(AllocateMemory(device, &ai, NULL, &mem));
         allocations = realloc(allocations,
                               sizeof(*allocations) * (allocations_count + 1));
         allocations[allocations_count++] = mem;
         VkBindDataGraphPipelineSessionMemoryInfoARM bi = {
            .sType =
               VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
            .session = session,
            .bindPoint = reqs[i].bindPoint,
            .objectIndex = j,
            .memory = mem};
         CHECK(BindDataGraphPipelineSessionMemoryARM(device, 1, &bi));
      }
   free(reqs);
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
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline);
   CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, layout, 0,
                         1, &set, 0, NULL);
   VkDataGraphPipelineDispatchInfoARM dispatch = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM};
   CmdDispatchDataGraphARM(cmd, session, &dispatch);
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .srcAccessMask = VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
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
   const char *repeat_option = getenv("DG_TEST_REPEATS");
   unsigned repeats = repeat_option ? strtoul(repeat_option, NULL, 10) : 1;
   if (!repeats || repeats > 1000)
      return 2;
   for (unsigned i = 0; i < repeats; i++) {
      CHECK(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      CHECK(QueueWaitIdle(queue));
   }
   unsigned failures = 0;
   for (unsigned i = 0; i < total[2]; i++) {
      float expected;
      if (matmul) {
         unsigned b = i / 12, m = i / 4 % 3, col = i % 4;
         expected = 0;
         for (unsigned k = 0; k < 5; k++)
            expected += ((float *)tensors[0].map)[b * 15 + m * 5 + k] *
                        ((float *)tensors[1].map)[b * 20 + k * 4 + col];
      } else {
         expected =
            ((float *)tensors[0].map)[i] + ((float *)tensors[1].map)[i % columns];
         if (!strcmp(argv[3], "fanout"))
            expected = 3 * ((float *)tensors[0].map)[i] +
                       ((float *)tensors[1].map)[i % columns];
         if (!strcmp(argv[3], "mul"))
            expected = ((float *)tensors[0].map)[i] *
                       ((float *)tensors[1].map)[i % columns];
         if (chain)
            expected = fminf(fmaxf(expected, 0.0f), clamp_max);
      }
      float actual = ((float *)tensors[2].map)[i];
      if (!isfinite(actual) || fabsf(actual - expected) > 1e-5f) {
         if (failures < 8)
            printf("MISMATCH %u got=%g expected=%g\n", i, actual, expected);
         failures++;
      }
   }
   printf("GRAPH %s checks=%u failures=%u\n", argv[3], total[2], failures);
   printf("SUBMISSIONS %u\n", repeats);
   DestroyCommandPool(device, cp, NULL);
   DestroyDataGraphPipelineSessionARM(device, session, NULL);
   for (unsigned i = 0; i < allocations_count; i++)
      FreeMemory(device, allocations[i], NULL);
   free(allocations);
   DestroyPipeline(device, pipeline, NULL);
   DestroyShaderModule(device, module, NULL);
   DestroyDescriptorPool(device, pool, NULL);
   DestroyPipelineLayout(device, layout, NULL);
   DestroyDescriptorSetLayout(device, sl, NULL);
   for (unsigned i = 0; i < 3; i++)
      destroy_tensor(&tensors[i]);
   DestroyDevice(device, NULL);
   return failures ? 1 : 0;
}
