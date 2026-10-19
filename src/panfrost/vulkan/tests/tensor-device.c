#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d at %d\n", #x, r, __LINE__);                   \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#define FN(n) static PFN_vk##n n
FN(GetDeviceProcAddr);
FN(CreateDevice);
FN(GetDeviceQueue);
FN(CreateTensorARM);
FN(DestroyTensorARM);
FN(GetTensorMemoryRequirementsARM);
FN(GetDeviceTensorMemoryRequirementsARM);
FN(BindTensorMemoryARM);
FN(CreateTensorViewARM);
FN(DestroyTensorViewARM);
FN(AllocateMemory);
FN(FreeMemory);
FN(MapMemory);
FN(UnmapMemory);
FN(CreateDescriptorSetLayout);
FN(DestroyDescriptorSetLayout);
FN(CreatePipelineLayout);
FN(DestroyPipelineLayout);
FN(CreateDescriptorPool);
FN(DestroyDescriptorPool);
FN(AllocateDescriptorSets);
FN(UpdateDescriptorSets);
FN(CreateShaderModule);
FN(DestroyShaderModule);
FN(CreateComputePipelines);
FN(DestroyPipeline);
FN(CreateCommandPool);
FN(DestroyCommandPool);
FN(AllocateCommandBuffers);
FN(BeginCommandBuffer);
FN(EndCommandBuffer);
FN(CmdBindPipeline);
FN(CmdBindDescriptorSets);
FN(CmdDispatch);
FN(QueueSubmit);
FN(QueueWaitIdle);
FN(DestroyDevice);
FN(CmdCopyTensorARM);
static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static VkQueue queue;
static uint32_t family;
static VkPhysicalDeviceMemoryProperties memory;
static VkTensorUsageFlagsARM tensor_extra_usage;
static PFN_vkGetInstanceProcAddr tensor_gipa;

struct module;
struct hwdev;
struct module {
   uint32_t tag;
   uint16_t a, b;
   const char *id, *name, *author;
   struct {
      int (*open)(const struct module *, const char *, struct hwdev **);
   } *methods;
   void *dso;
   uint64_t reserved[25];
};
struct hwdev {
   uint32_t tag, version;
   struct module *module;
   uint64_t reserved[12];
   int (*close)(struct hwdev *);
};
struct hwvk {
   struct hwdev common;
   PFN_vkEnumerateInstanceExtensionProperties enumerate;
   PFN_vkCreateInstance create;
   PFN_vkGetInstanceProcAddr gipa;
};

static void
init(const char *path, int caps)
{
   int system_loader = !strcmp(path, "system");
   void *lib =
      dlopen(system_loader ? "libvulkan.so" : path, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      fprintf(stderr, "%s\n", dlerror());
      exit(1);
   }
   VkResult (*negotiate)(uint32_t *) =
      dlsym(lib, "vk_icdNegotiateLoaderICDInterfaceVersion");
   if (negotiate) {
      uint32_t v = 5;
      CHECK(negotiate(&v));
   }
   PFN_vkGetInstanceProcAddr gipa =
      dlsym(lib, system_loader ? "vkGetInstanceProcAddr"
                               : "vk_icdGetInstanceProcAddr");
   if (!gipa) {
      struct module *m = dlsym(lib, "HMI");
      struct hwdev *h = NULL;
      if (!m || m->methods->open(m, "vk0", &h) || !h)
         exit(1);
      gipa = ((struct hwvk *)h)->gipa;
   }
   tensor_gipa = gipa;
   PFN_vkCreateInstance create = (void *)gipa(NULL, "vkCreateInstance");
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "manet-tensor-test",
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   CHECK(create(&ici, NULL, &instance));
   PFN_vkEnumeratePhysicalDevices ep =
      (void *)gipa(instance, "vkEnumeratePhysicalDevices");
   uint32_t n = 1;
   VkResult r = ep(instance, &n, &physical);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !n)
      exit(1);
   PFN_vkEnumerateDeviceExtensionProperties ex =
      (void *)gipa(instance, "vkEnumerateDeviceExtensionProperties");
   CHECK(ex(physical, NULL, &n, NULL));
   VkExtensionProperties *ext = calloc(n, sizeof(*ext));
   CHECK(ex(physical, NULL, &n, ext));
   int tensor = 0, graph = 0;
   for (unsigned i = 0; i < n; i++) {
      if (!strcmp(ext[i].extensionName, VK_ARM_TENSORS_EXTENSION_NAME))
         tensor = 1;
      if (!strcmp(ext[i].extensionName, VK_ARM_DATA_GRAPH_EXTENSION_NAME))
         graph = 1;
   }
   free(ext);
   PFN_vkGetPhysicalDeviceFeatures2 features =
      (void *)gipa(instance, "vkGetPhysicalDeviceFeatures2");
   VkPhysicalDeviceDataGraphFeaturesARM dg = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
   VkPhysicalDeviceTensorFeaturesARM tf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
      .pNext = &dg};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &tf};
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13};
   VkPhysicalDeviceFeatures2 f = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &f12};
   features(physical, &f);
   printf(
      "CAP tensor=%d shader=%u nonpacked=%u graph=%d dataGraph=%u graphModule=%u graphSpec=%u\n",
      tensor, tf.shaderTensorAccess, tf.tensorNonPacked, graph, dg.dataGraph,
      dg.dataGraphShaderModule, dg.dataGraphSpecializationConstants);
   if (caps)
      return;
   if (!tensor || !tf.shaderTensorAccess)
      exit(3);
   PFN_vkGetPhysicalDeviceQueueFamilyProperties q =
      (void *)gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
   q(physical, &n, NULL);
   VkQueueFamilyProperties *qp = calloc(n, sizeof(*qp));
   q(physical, &n, qp);
   family = UINT32_MAX;
   for (unsigned i = 0; i < n; i++)
      if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
         family = i;
         break;
      }
   free(qp);
   if (family == UINT32_MAX)
      exit(1);
   PFN_vkGetPhysicalDeviceMemoryProperties mp =
      (void *)gipa(instance, "vkGetPhysicalDeviceMemoryProperties");
   mp(physical, &memory);
   CreateDevice = (void *)gipa(instance, "vkCreateDevice");
   GetDeviceProcAddr = (void *)gipa(instance, "vkGetDeviceProcAddr");
   dg = (VkPhysicalDeviceDataGraphFeaturesARM){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM,
      .dataGraph = graph,
      .dataGraphSpecializationConstants =
         graph && getenv("DG_TEST_SPECIALIZATION") != NULL,
      .dataGraphShaderModule = graph};
   tf = (VkPhysicalDeviceTensorFeaturesARM){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
      .pNext = graph ? &dg : NULL,
      .tensorNonPacked = VK_TRUE,
      .shaderTensorAccess = VK_TRUE,
      .tensors = VK_TRUE};
   VkPhysicalDeviceVulkan13Features enable13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &tf,
      .maintenance4 = f13.maintenance4};
   VkPhysicalDeviceVulkan12Features enable12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &enable13,
      .shaderFloat16 = f12.shaderFloat16,
      .shaderInt8 = f12.shaderInt8,
      .vulkanMemoryModel = f12.vulkanMemoryModel};
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   const char *extensions[] = {VK_ARM_TENSORS_EXTENSION_NAME,
                               VK_ARM_DATA_GRAPH_EXTENSION_NAME,
                               VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
                               VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
   VkPhysicalDeviceFeatures basic = {.shaderInt16 = f.features.shaderInt16,
                                     .shaderInt64 = f.features.shaderInt64};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .pNext = &enable12,
                            .queueCreateInfoCount = 1,
                            .pQueueCreateInfos = &qi,
                            .enabledExtensionCount = graph ? 4 : 1,
                            .ppEnabledExtensionNames = extensions,
                            .pEnabledFeatures = &basic};
   CHECK(CreateDevice(physical, &di, NULL, &device));
#define LOAD(n)                                                                \
   n = (void *)GetDeviceProcAddr(device, "vk" #n);                             \
   if (!n) {                                                                   \
      fprintf(stderr, "missing vk%s\n", #n);                                   \
      exit(1);                                                                 \
   }
   LOAD(GetDeviceQueue);
   LOAD(CreateTensorARM);
   LOAD(DestroyTensorARM);
   LOAD(GetTensorMemoryRequirementsARM);
   LOAD(GetDeviceTensorMemoryRequirementsARM);
   LOAD(BindTensorMemoryARM);
   LOAD(CreateTensorViewARM);
   LOAD(DestroyTensorViewARM);
   LOAD(AllocateMemory);
   LOAD(FreeMemory);
   LOAD(MapMemory);
   LOAD(UnmapMemory);
   LOAD(CreateDescriptorSetLayout);
   LOAD(DestroyDescriptorSetLayout);
   LOAD(CreatePipelineLayout);
   LOAD(DestroyPipelineLayout);
   LOAD(CreateDescriptorPool);
   LOAD(DestroyDescriptorPool);
   LOAD(AllocateDescriptorSets);
   LOAD(UpdateDescriptorSets);
   LOAD(CreateShaderModule);
   LOAD(DestroyShaderModule);
   LOAD(CreateComputePipelines);
   LOAD(DestroyPipeline);
   LOAD(CreateCommandPool);
   LOAD(DestroyCommandPool);
   LOAD(AllocateCommandBuffers);
   LOAD(BeginCommandBuffer);
   LOAD(EndCommandBuffer);
   LOAD(CmdBindPipeline);
   LOAD(CmdBindDescriptorSets);
   LOAD(CmdDispatch);
   LOAD(QueueSubmit);
   LOAD(QueueWaitIdle);
   LOAD(DestroyDevice);
   LOAD(CmdCopyTensorARM);
   GetDeviceQueue(device, family, 0, &queue);
}

struct tensor {
   VkTensorARM tensor;
   VkTensorViewARM view;
   VkDeviceMemory mem;
   void *map;
   VkDeviceSize size;
};
static struct tensor
make_tensor(VkFormat format, unsigned rank, const int64_t *dims,
            const int64_t *strides)
{
   struct tensor t = {0};
   VkTensorDescriptionARM desc = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .format = format,
      .dimensionCount = rank,
      .pDimensions = dims,
      .pStrides = strides,
      .usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
               VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
               VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM | tensor_extra_usage};
   VkTensorCreateInfoARM ci = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
      .pDescription = &desc};
   CHECK(CreateTensorARM(device, &ci, NULL, &t.tensor));
   VkTensorMemoryRequirementsInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM,
      .tensor = t.tensor};
   VkMemoryRequirements2 req = {.sType =
                                   VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
   GetTensorMemoryRequirementsARM(device, &info, &req);
   VkDeviceTensorMemoryRequirementsARM direct = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_TENSOR_MEMORY_REQUIREMENTS_ARM,
      .pCreateInfo = &ci};
   VkMemoryRequirements2 req2 = {.sType =
                                    VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
   GetDeviceTensorMemoryRequirementsARM(device, &direct, &req2);
   if (memcmp(&req.memoryRequirements, &req2.memoryRequirements,
              sizeof(req.memoryRequirements)))
      exit(1);
   uint32_t mt = UINT32_MAX;
   for (unsigned i = 0; i < memory.memoryTypeCount; i++)
      if ((req.memoryRequirements.memoryTypeBits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & 6) == 6) {
         mt = i;
         break;
      }
   if (mt == UINT32_MAX)
      exit(1);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.memoryRequirements.size + 64,
      .memoryTypeIndex = mt};
   CHECK(AllocateMemory(device, &ai, NULL, &t.mem));
   t.size = ai.allocationSize;
   VkBindTensorMemoryInfoARM bi = {
      .sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM,
      .tensor = t.tensor,
      .memory = t.mem};
   CHECK(BindTensorMemoryARM(device, 1, &bi));
   CHECK(MapMemory(device, t.mem, 0, VK_WHOLE_SIZE, 0, &t.map));
   memset(t.map, 0xcd, t.size);
   VkTensorViewCreateInfoARM vi = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
      .tensor = t.tensor,
      .format = format};
   CHECK(CreateTensorViewARM(device, &vi, NULL, &t.view));
   return t;
}
static void
destroy_tensor(struct tensor *t)
{
   DestroyTensorViewARM(device, t->view, NULL);
   DestroyTensorARM(device, t->tensor, NULL);
   UnmapMemory(device, t->mem);
   FreeMemory(device, t->mem, NULL);
}

int
main(int argc, char **argv)
{
   if (argc < 2)
      return 2;
   init(argv[1], argc == 2);
   if (argc == 2)
      return 0;
   unsigned rank = atoi(argv[3]), bytes = atoi(argv[4]), width = atoi(argv[5]),
            padded = atoi(argv[6]), mode = atoi(argv[7]);
   VkFormat format = bytes == 1   ? VK_FORMAT_R8_UINT
                     : bytes == 2 ? VK_FORMAT_R16_UINT
                     : bytes == 4 ? VK_FORMAT_R32_UINT
                                  : VK_FORMAT_R64_UINT;
   int64_t dims[4] = {2, 3, 5, 19}, strides[4];
   dims[rank - 1] = 19;
   strides[rank - 1] = bytes;
   for (int d = rank - 2; d >= 0; d--)
      strides[d] = strides[d + 1] * dims[d + 1] + (padded ? 64 : 0);
   struct tensor tensors[2] = {
      make_tensor(format, rank, dims, padded ? strides : NULL),
      make_tensor(format, rank, dims, padded ? strides : NULL)};
   uint32_t total = 1;
   for (unsigned d = 0; d < rank; d++)
      total *= dims[d];
   if (!padded) {
      strides[rank - 1] = bytes;
      for (int d = rank - 2; d >= 0; d--)
         strides[d] = strides[d + 1] * dims[d + 1];
   }
   for (uint32_t i = 0; i < total; i++) {
      uint32_t idx = i;
      uint64_t off = 0;
      for (int d = rank - 1; d >= 0; d--) {
         off += (idx % dims[d]) * strides[d];
         idx /= dims[d];
      }
      uint64_t value = (uint64_t)i * 13 + 7;
      memcpy((char *)tensors[0].map + off, &value, bytes);
   }
   VkDescriptorSetLayoutBinding bindings[2];
   for (unsigned i = 0; i < 2; i++)
      bindings[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(CreateDescriptorSetLayout(device, &li, NULL, &sl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &sl};
   VkPipelineLayout layout;
   CHECK(CreatePipelineLayout(device, &pli, NULL, &layout));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_TENSOR_ARM, 2};
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
   for (unsigned i = 0; i < 2; i++) {
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
   FILE *file = fopen(argv[2], "rb");
   if (!file)
      return 1;
   fseek(file, 0, SEEK_END);
   size_t size = ftell(file);
   rewind(file);
   uint32_t *code = malloc(size);
   if (fread(code, 1, size, file) != size)
      exit(1);
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
                .pName = "main"},
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
   if (mode == 2) {
      VkTensorCopyARM region = {
         .sType = VK_STRUCTURE_TYPE_TENSOR_COPY_ARM,
         .dimensionCount = rank};
      VkCopyTensorInfoARM copy = {
         .sType = VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_ARM,
         .srcTensor = tensors[0].tensor,
         .dstTensor = tensors[1].tensor,
         .regionCount = 1,
         .pRegions = &region};
      CmdCopyTensorARM(cmd, &copy);
   } else {
      CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                            &set, 0, NULL);
      CmdDispatch(cmd, mode == 1 ? 1 : (total + 63) / 64, 1, 1);
   }
   CHECK(EndCommandBuffer(cmd));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(QueueWaitIdle(queue));
   unsigned failures = 0;
   unsigned count = mode == 1 ? width : total;
   unsigned char *reference = malloc(tensors[1].size);
   memset(reference, 0xcd, tensors[1].size);
   for (unsigned i = 0; i < count; i++) {
      uint32_t idx = i;
      uint64_t off = 0;
      for (int d = rank - 1; d >= 0; d--) {
         off += (idx % dims[d]) * strides[d];
         idx /= dims[d];
      }
      uint64_t expected = mode == 1 ? 0x3b : (uint64_t)i * 13 + 7, actual = 0;
      memcpy(reference + off, &expected, bytes);
      memcpy(&actual, (char *)tensors[1].map + off, bytes);
      if (bytes < 8)
         expected &= (1ull << (bytes * 8)) - 1;
      if (actual != expected) {
         if (failures < 8)
            printf("MISMATCH i=%u off=%llu got=%llx expected=%llx\n", i,
                   (unsigned long long)off, (unsigned long long)actual,
                   (unsigned long long)expected);
         failures++;
      }
   }
   if (memcmp(reference, tensors[1].map, tensors[1].size)) {
      printf("FAIL complete memory/padding/guard comparison\n");
      failures++;
   }
   free(reference);
   printf(
      "RESULT rank=%u bytes=%u width=%u padded=%u mode=%u checks=%u failures=%u\n",
      rank, bytes, width, padded, mode, count, failures);
   DestroyCommandPool(device, cp, NULL);
   DestroyPipeline(device, pipeline, NULL);
   DestroyShaderModule(device, module, NULL);
   DestroyDescriptorPool(device, pool, NULL);
   DestroyPipelineLayout(device, layout, NULL);
   DestroyDescriptorSetLayout(device, sl, NULL);
   for (unsigned i = 0; i < 2; i++)
      destroy_tensor(&tensors[i]);
   DestroyDevice(device, NULL);
   return failures ? 1 : 0;
}
