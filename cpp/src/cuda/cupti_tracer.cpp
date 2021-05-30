#include "habitat_cupti.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <cupti.h>
#include <cuda_runtime.h>

#include "cupti_macros.h"

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

namespace {

void handleCuptiActivity(CUpti_Activity* record) {
  if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL && record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
    return;
  }
  CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4 *) record;
  habitat::cuda::KernelInstance instance(
    habitat::cuda::KernelMetadata(
      std::string(kernel->name),
      kernel->gridX * kernel->gridY * kernel->gridZ,
      kernel->blockX * kernel->blockY * kernel->blockZ,
      kernel->dynamicSharedMemory,
      kernel->staticSharedMemory,
      kernel->registersPerThread),
    kernel->end - kernel->start);
  habitat::cuda::CuptiManager::instance().newKernelInstance(std::move(instance));
}

void cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    std::cerr << "ERROR: Out of memory! (malloc in CUPTI)" << std::endl;
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void cuptiBufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        handleCuptiActivity(record);

      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;

      } else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      std::cerr << "WARNING: CUPTI dropped " << dropped << " activity records." << std::endl;
    }
  }

  free(buffer);
}

inline void enableCuptiRecording() {
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
}

inline void flushCupti() {
  cudaDeviceSynchronize();
  cuptiActivityFlushAll(0);
}

inline void disableCuptiRecording() {
  flushCupti();
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
}

}

namespace habitat {
namespace cuda {

CuptiTracer::CuptiTracer() {}

std::vector<KernelInstance>&& CuptiTracer::kernels() && {
  flushCupti();
  return std::move(kernels_);
}

std::vector<KernelInstance> CuptiTracer::lastKernels(size_t num_iterations) const {
  flushCupti();
  // NOTE: We assume that, after this point, no more kernels will be appended to the kernels_ vector.
  if (kernels_.size() % num_iterations != 0) {
    throw std::runtime_error("Recorded kernel size mismatch!");
  }

  size_t num_kernels = kernels_.size() / num_iterations;
  std::vector<KernelInstance> results;
  results.reserve(num_kernels);
  results.insert(results.begin(), kernels_.end() - num_kernels, kernels_.end());
  return results;
}

CuptiTracer::Ptr CuptiManager::allocateTracer() {
  if (!callbacks_bound_) {
    CUPTI_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
    callbacks_bound_ = true;
  }
  if (tracers_.size() == 0) {
    enableCuptiRecording();
  }
  CuptiTracer::Ptr ptr(new CuptiTracer(), &detail::cuptiTracerDeleter);
  tracers_.push_back(ptr.get());
  return ptr;
}

namespace detail {

void cuptiTracerDeleter(CuptiTracer* tracer) {
  CuptiManager& manager = CuptiManager::instance();
  auto it = std::find(manager.tracers_.begin(), manager.tracers_.end(), tracer);
  if (it == manager.tracers_.end()) {
    // Assertion failure
    throw std::runtime_error("Did not find CUPTI tracer in the manager's list when deleting!");
  }
  manager.tracers_.erase(it);
  if (manager.tracers_.size() == 0) {
    disableCuptiRecording();
  }
  delete tracer;
}

}
}
}
