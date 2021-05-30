#include "kernel.h"

namespace {

// We don't want any symbols from this header to be visible outside of this file
#include "cuda_occupancy.h"

cudaOccDeviceProp occDeviceProps(const habitat::cuda::DeviceProperties& properties) {
  cudaOccDeviceProp device_properties;
  device_properties.computeMajor = properties.compute_major;
  device_properties.computeMinor = properties.compute_minor;
  device_properties.maxThreadsPerBlock = properties.max_threads_per_block;
  device_properties.maxThreadsPerMultiprocessor = properties.max_threads_per_multiprocessor;
  device_properties.regsPerBlock = properties.regs_per_block;
  device_properties.regsPerMultiprocessor = properties.regs_per_multiprocessor;
  device_properties.warpSize = properties.warp_size;
  device_properties.sharedMemPerBlock = properties.shared_mem_per_block;
  device_properties.sharedMemPerMultiprocessor = properties.shared_mem_per_multiprocessor;
  device_properties.numSms = properties.num_sms;
  device_properties.sharedMemPerBlockOptin = properties.shared_mem_per_block_optin;
  return device_properties;
}

}

namespace habitat {
namespace cuda {

uint32_t KernelMetadata::threadBlockOccupancy(const DeviceProperties& device) const {
  return threadBlockOccupancy(device, registers_per_thread_);
}

uint32_t KernelMetadata::threadBlockOccupancy(
    const DeviceProperties& device, uint16_t registers_per_thread) const {
  cudaOccDeviceProp device_properties(occDeviceProps(device));
  cudaOccDeviceState device_state;
  cudaOccFuncAttributes attributes;
  attributes.maxThreadsPerBlock = INT_MAX;
  attributes.maxDynamicSharedSizeBytes = INT_MAX;
  attributes.numRegs = registers_per_thread;
  attributes.sharedSizeBytes = static_shared_memory_;

  int res;
  cudaOccResult result;
  if ((res = cudaOccMaxActiveBlocksPerMultiprocessor(
        &result,
        &device_properties,
        &attributes,
        &device_state,
        block_size_,
        dynamic_shared_memory_)) != CUDA_OCC_SUCCESS) {
    return 0;
  }

  return result.activeBlocksPerMultiprocessor;
}

bool operator==(const KernelMetadata& lhs, const KernelMetadata& rhs) {
  return lhs.num_blocks_ == rhs.num_blocks_ &&
    lhs.block_size_ == rhs.block_size_ &&
    lhs.dynamic_shared_memory_ == rhs.dynamic_shared_memory_ &&
    lhs.static_shared_memory_ == rhs.static_shared_memory_ &&
    lhs.registers_per_thread_ == rhs.registers_per_thread_ &&
    lhs.name_ == rhs.name_;
}

void KernelInstance::addMetric(std::string name, double value) {
  metrics_.push_back(std::make_pair(std::move(name), value));
}

}
}
