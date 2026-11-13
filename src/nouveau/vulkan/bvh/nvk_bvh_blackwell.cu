#if NVK_BVH_TARGET_SM != 120 || NVK_BVH_TARGET_ARCH != 5
#error Unsupported NVK BVH CUDA target
#endif

#include "nvk_bvh_build.cu"
#include "nvk_bvh_planner.cu"
#include "nvk_bvh_encode.cu"
#include "nvk_bvh_refit.cu"
#include "nvk_bvh_copy.cu"
#include "nvk_bvh_pack.cu"
#include "nvk_bvh_leaf.cu"
#include "nvk_bvh_omm.cu"
