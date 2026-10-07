/*
 * Copyright 2024 Google LLC
 * Copyright 2025 Arm Ltd.
 * Copyright 2026 NXP
 * SPDX-License-Identifier: MIT
 */

#include "panvk_utrace.h"

#include "kmod/pan_kmod.h"
#include "util/log.h"
#include "util/timespec.h"
#include "panvk_device.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "vk_sync.h"

static struct panvk_device *
to_dev(struct u_trace_context *utctx)
{
   return container_of(utctx, struct panvk_device, utrace.utctx);
}

void *
panvk_utrace_create_buffer(struct u_trace_context *utctx, uint64_t size_B)
{
   struct panvk_device *dev = to_dev(utctx);

   /* This memory is also used to write CSF commands, therefore we align to a
    * cache line. */
   const uint64_t alignment = 0x40;

   simple_mtx_lock(&dev->utrace.copy_buf_heap_lock);
   const uint64_t addr_dev =
      util_vma_heap_alloc(&dev->utrace.copy_buf_heap, size_B, alignment);
   simple_mtx_unlock(&dev->utrace.copy_buf_heap_lock);

   if (!addr_dev) {
      mesa_loge("Couldn't allocate utrace buffer (size = 0x%" PRIx64 ")."
                "Provide larger PANVK_UTRACE_CLONE_MEM_SIZE (current = 0x%" PRIx64 ")",
                size_B, dev->utrace.copy_buf_heap_bo->bo->size);
      return NULL;
   }

   struct panvk_utrace_buf *container = malloc(sizeof(struct panvk_utrace_buf));
   void *addr_host = dev->utrace.copy_buf_heap_bo->addr.host + addr_dev -
                     dev->utrace.copy_buf_heap_bo->addr.dev;

   *container = (struct panvk_utrace_buf){
      .host = addr_host,
      .dev = addr_dev,
      .size = size_B,
   };

   memset(addr_host, 0, size_B);

   return container;
}

void
panvk_utrace_delete_buffer(struct u_trace_context *utctx, void *buffer)
{
   /* u_trace_fini() frees the NULL container stored when a clone allocation
    * failed; nothing to release in that case. */
   if (!buffer)
      return;

   struct panvk_device *dev = to_dev(utctx);
   struct panvk_utrace_buf *buf = buffer;

   simple_mtx_lock(&dev->utrace.copy_buf_heap_lock);
   util_vma_heap_free(&dev->utrace.copy_buf_heap, buf->dev, buf->size);
   simple_mtx_unlock(&dev->utrace.copy_buf_heap_lock);

   free(buffer);
}

uint64_t
panvk_utrace_read_ts(struct u_trace_context *utctx, void *timestamps,
                     uint64_t offset_B, uint32_t flags, void *flush_data)
{
   struct panvk_device *dev = to_dev(utctx);
   const struct panvk_physical_device *pdev =
      to_panvk_physical_device(dev->vk.physical);
   const struct pan_kmod_dev_props *props = &pdev->kmod.dev->props;
   const struct panvk_utrace_buf *buf = timestamps;
   struct panvk_utrace_flush_data *data = flush_data;

   assert(props->timestamp_frequency);

   /* The buffer may be NULL if its clone-time allocation failed; the trace
    * point has no valid timestamp. */
   if (!buf) {
      mesa_loge("utrace: missing timestamp buffer (clone alloc failed); "
                "reporting no timestamp for this trace point");
      return U_TRACE_NO_TIMESTAMP;
   }

   struct panvk_utrace_submission *submission = data->submission;
   if (!submission->waited && !submission->failed) {
      submission->failed =
         vk_sync_wait(&dev->vk, submission->sync, submission->wait_value,
                      VK_SYNC_WAIT_COMPLETE, UINT64_MAX) != VK_SUCCESS;
      if (submission->failed)
          mesa_logw("failed to wait for utrace timestamps");
      submission->waited = true;
   }

   if (submission->failed)
      return U_TRACE_NO_TIMESTAMP;

   const uint64_t *ts_ptr = buf->host + offset_B;
   uint64_t ts = *ts_ptr;
   if (ts != U_TRACE_NO_TIMESTAMP)
      ts = pan_kmod_timestamp_cycles_to_ns(pdev->kmod.dev, ts);

   return ts;
}

const void *
panvk_utrace_get_data(struct u_trace_context *utctx, void *buffer,
                      uint64_t offset_B, uint32_t size_B)
{
   const struct panvk_utrace_buf *buf = buffer;

   /* The buffer may be NULL if its clone-time allocation failed; report
    * no data rather than dereferencing NULL. */
   if (!buf) {
      mesa_loge("utrace: missing indirect data buffer (clone alloc failed); "
                "reporting no data for this trace point");
      return NULL;
   }

   return buf->host + offset_B;
}

void
panvk_utrace_delete_flush_data(struct u_trace_context *utctx, void *flush_data)
{
   struct panvk_utrace_flush_data *data = flush_data;

   util_dynarray_foreach(&data->clone_cs_bufs, struct panvk_utrace_buf *, buf)
      panvk_utrace_delete_buffer(utctx, *buf);
   util_dynarray_fini(&data->clone_cs_bufs);

   panvk_utrace_submission_unref(utctx->pctx, data->submission);
}

void
panvk_utrace_submission_unref(struct panvk_device *dev,
                              struct panvk_utrace_submission *submission)
{
   if (submission && p_atomic_dec_zero(&submission->refs)) {
      if (submission->owns_sync)
         vk_sync_destroy(&dev->vk, submission->sync);
      free(submission);
   }
}

char *
panvk_utrace_label(enum u_trace_backend_type backend, const char *label)
{
   if (backend != U_TRACE_BACKEND_JSON)
      return strdup(label);

   char *escaped = malloc(strlen(label) * 6 + 1);
   if (!escaped)
      return NULL;

   char *out = escaped;
   for (const unsigned char *p = (const unsigned char *)label; *p; p++) {
      if (*p == '"' || *p == '\\') {
         *out++ = '\\';
         *out++ = *p;
      } else if (*p < 0x20) {
         snprintf(out, 7, "\\u%04x", *p);
         out += 6;
      } else {
         *out++ = *p;
      }
   }
   *out = 0;
   return escaped;
}
