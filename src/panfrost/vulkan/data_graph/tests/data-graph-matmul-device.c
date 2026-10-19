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

static void *
element(struct tensor *t, unsigned rank, const int64_t *dims,
        const int64_t *strides, unsigned index)
{
   uint64_t offset = 0;
   for (unsigned i = rank; i-- > 0;) {
      offset += (index % dims[i]) * strides[i];
      index /= dims[i];
   }
   return (char *)t->map + offset;
}

static unsigned dtype;
static int a_zp,b_zp;
static float load_value(void *p, unsigned tensor) {
   if (dtype == 1) return *(const _Float16 *)p;
   if (dtype == 2 || (dtype == 3 && tensor == 2)) return *(const int32_t *)p;
   if (dtype == 3) return *(const int8_t *)p;
   return *(const float *)p;
}
static int32_t load_integer(void *p, unsigned tensor) {
   return dtype == 3 && tensor != 2 ? *(const int8_t *)p : *(const int32_t *)p;
}
static void store_value(void *p, unsigned tensor, float value) {
   if (dtype == 1) *(_Float16 *)p = value;
   else if (dtype == 2 || (dtype == 3 && tensor == 2)) *(int32_t *)p = value;
   else if (dtype == 3) *(int8_t *)p = value;
   else *(float *)p = value;
}
static unsigned
parameter(const char *name, unsigned fallback)
{
   const char *value = getenv(name);
   return value ? atoi(value) : fallback;
}

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
   int matmul = !strncmp(argv[3], "matmul", 6), chain = !strcmp(argv[3], "chain");
   unsigned batch = parameter("DG_TEST_BATCH", 2), rows = parameter("DG_TEST_M", 3),
            cols = parameter("DG_TEST_N", 4), inner = parameter("DG_TEST_K", 5);
   unsigned padded = parameter("DG_TEST_PADDED", 0);
   dtype = parameter("DG_TEST_DTYPE", 3);
   a_zp=getenv("DG_A_ZP")?atoi(getenv("DG_A_ZP")):0;
   b_zp=getenv("DG_B_ZP")?atoi(getenv("DG_B_ZP")):0;
   VkFormat formats[3];
   for (unsigned i = 0; i < 3; i++) formats[i] = dtype == 1 ? VK_FORMAT_R16_SFLOAT : dtype == 2 || (dtype == 3 && i == 2) ? VK_FORMAT_R32_SINT : dtype == 3 ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SFLOAT;
   PFN_vkGetPhysicalDeviceFormatProperties2 format_properties =
      (void *)tensor_gipa(instance, "vkGetPhysicalDeviceFormatProperties2");
   if (!format_properties)
      return 2;
   for (unsigned i = 0; i < 3; i++) {
      VkTensorFormatPropertiesARM tensor_properties = {
         .sType = VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM};
      VkFormatProperties2 properties = {
         .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
         .pNext = &tensor_properties};
      format_properties(physical, formats[i], &properties);
      if (!(tensor_properties.linearTilingTensorFeatures &
            VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM))
         return 2;
   }
   int twice = !strcmp(argv[3], "matmul-chain");
   if (twice && inner != cols) return 2;
   unsigned rank = matmul ? 3 : 2;
   int64_t dims[3][4] = {{3, 19}, {1, 19}, {3, 19}};
   if (matmul) {
      int64_t shapes[3][4] = {{batch, rows, inner}, {batch, inner, cols}, {batch, rows, cols}};
      memcpy(dims, shapes, sizeof(dims));
   }
   struct tensor tensors[3];
   unsigned total[3] = {1, 1, 1};
   VkTensorDescriptionARM desc[3];
   VkDataGraphPipelineResourceInfoARM resources[3];
   int64_t strides[3][4];
   for (unsigned i = 0; i < 3; i++) {
      strides[i][rank - 1] = dtype == 1 ? 2 : dtype == 3 && i < 2 ? 1 : 4;
      for (unsigned d = rank - 1; d-- > 0;)
         strides[i][d] = strides[i][d + 1] * dims[i][d + 1] + (padded ? 64 : 0);
   }
   for (unsigned i = 0; i < 3; i++) {
      tensors[i] = make_tensor(formats[i], rank, dims[i], padded ? strides[i] : NULL);
      for (unsigned d = 0; d < rank; d++)
         total[i] *= dims[i][d];
      desc[i] = (VkTensorDescriptionARM){
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .format = formats[i],
         .dimensionCount = rank,
         .pDimensions = dims[i],
         .pStrides = padded ? strides[i] : NULL,
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};
      resources[i] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = &desc[i],
         .descriptorSet = 0,
         .binding = i};
      if (i < 2)
         for (unsigned j = 0; j < total[i]; j++)
            store_value(element(&tensors[i], rank, dims[i], strides[i], j), i, ((int)((j * 13 + i * 7) % 9) - 4) * (dtype >= 2 ? 1.0f : 0.125f));
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
   printf("SESSION_REQUIREMENTS count=%u\n", n);
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
   unsigned repeats = parameter("DG_TEST_REPEATS", 1);
   if (!repeats || repeats > 1000)
      return 2;
   for (unsigned repeat = 0; repeat < repeats; repeat++) {
   CHECK(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(QueueWaitIdle(queue));
   }
   unsigned failures = 0;
   for (unsigned i = 0; i < total[2]; i++) {
      if (dtype >= 2) {
         uint32_t expected = 0;
         if (matmul) {
            unsigned b = i / (rows * cols), m = i / cols % rows, col = i % cols;
            for (unsigned k = 0; k < inner; k++) {
               int64_t left;
               if (twice) {
                  uint32_t intermediate = 0;
                  for (unsigned j = 0; j < inner; j++)
                     intermediate += (uint32_t)((int64_t)load_integer(element(&tensors[0], rank, dims[0], strides[0], b * rows * inner + m * inner + j), 0) *
                                                load_integer(element(&tensors[1], rank, dims[1], strides[1], b * inner * cols + j * cols + k), 1));
                  left = (int32_t)intermediate;
               } else {
                  left = load_integer(element(&tensors[0], rank, dims[0], strides[0], b * rows * inner + m * inner + k), 0);
               }
               int64_t right = load_integer(element(&tensors[1], rank, dims[1], strides[1], b * inner * cols + k * cols + col), 1);
               expected += (uint32_t)((left - a_zp) * (right - b_zp));
            }
         } else {
            expected = (uint32_t)load_integer(element(&tensors[0], rank, dims[0], strides[0], i), 0) +
                       (uint32_t)load_integer(element(&tensors[1], rank, dims[1], strides[1], i % dims[1][1]), 1);
            if (chain)
               expected = (int32_t)expected < 0 ? 0 : expected > 1 ? 1 : expected;
         }
         if (!strcmp(argv[3], "matmul-add")) expected *= 2;
         int32_t actual = load_integer(element(&tensors[2], rank, dims[2], strides[2], i), 2);
         if (actual != (int32_t)expected) {
            if (failures < 8)
               printf("MISMATCH %u got=%d expected=%d\n", i, actual, (int32_t)expected);
            failures++;
         }
         continue;
      }
      float expected;
      if (matmul) {
         unsigned b = i / (rows * cols), m = i / cols % rows, col = i % cols;
         expected = 0;
         for (unsigned k = 0; k < inner; k++) {
            float left;
            if (twice) {
               left = 0;
               for (unsigned j = 0; j < inner; j++)
                  left += load_value(element(&tensors[0], rank, dims[0], strides[0], b * rows * inner + m * inner + j), 0) *
                          load_value(element(&tensors[1], rank, dims[1], strides[1], b * inner * cols + j * cols + k), 1);
            } else {
               left = load_value(element(&tensors[0], rank, dims[0], strides[0], b * rows * inner + m * inner + k), 0);
            }
            expected += (left-a_zp) * (load_value(element(&tensors[1], rank, dims[1], strides[1], b * inner * cols + k * cols + col), 1)-b_zp);
         }
      } else {
         expected = load_value(element(&tensors[0], rank, dims[0], strides[0], i), 0) +
                    load_value(element(&tensors[1], rank, dims[1], strides[1], i % dims[1][1]), 1);
         if (chain)
            expected = fminf(fmaxf(expected, 0.0f), 1.0f);
      }
      if (!strcmp(argv[3], "matmul-add")) expected *= 2;
      float actual = load_value(element(&tensors[2], rank, dims[2], strides[2], i), 2);
      if (!isfinite(actual) ||
          (dtype >= 2 ? actual != expected :
           fabsf(actual - expected) > 1e-5f * fmaxf(1.0f, fabsf(expected)))) {
         if (failures < 8)
            printf("MISMATCH %u got=%g expected=%g\n", i, actual, expected);
         failures++;
      }
   }
   for (unsigned i = tensors[2].size - 64; i < tensors[2].size; i++)
      if (((unsigned char *)tensors[2].map)[i] != 0xcd) failures++;
   printf("GRAPH %s checks=%u failures=%u\n", argv[3], total[2], failures);
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
