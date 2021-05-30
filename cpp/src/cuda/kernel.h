#pragma once

#include <string>
#include <cstdint>
#include <utility>
#include <vector>

#include "sampled_measurement.h"
#include "utils.h"

namespace habitat {
namespace cuda {

struct DeviceProperties {
  DeviceProperties(
      std::string name,
      int compute_major,
      int compute_minor,
      int max_threads_per_block,
      int max_threads_per_multiprocessor,
      int regs_per_block,
      int regs_per_multiprocessor,
      int warp_size,
      size_t shared_mem_per_block,
      size_t shared_mem_per_multiprocessor,
      int num_sms,
      size_t shared_mem_per_block_optin,
      int mem_bandwidth_gb,
      size_t base_clock_mhz,
      size_t peak_gflops_per_second)
        : name(std::move(name)),
          compute_major(compute_major),
          compute_minor(compute_minor),
          max_threads_per_block(max_threads_per_block),
          max_threads_per_multiprocessor(max_threads_per_multiprocessor),
          regs_per_block(regs_per_block),
          regs_per_multiprocessor(regs_per_multiprocessor),
          warp_size(warp_size),
          shared_mem_per_block(shared_mem_per_block),
          shared_mem_per_multiprocessor(shared_mem_per_multiprocessor),
          num_sms(num_sms),
          shared_mem_per_block_optin(shared_mem_per_block_optin),
          mem_bandwidth_gb(mem_bandwidth_gb),
          base_clock_mhz(base_clock_mhz),
          peak_gflops_per_second(peak_gflops_per_second) {}

  std::string name;
  int compute_major;
  int compute_minor;
  int max_threads_per_block;
  int max_threads_per_multiprocessor;
  int regs_per_block;
  int regs_per_multiprocessor;
  int warp_size;
  size_t shared_mem_per_block;
  size_t shared_mem_per_multiprocessor;
  int num_sms;
  size_t shared_mem_per_block_optin;
  int mem_bandwidth_gb;
  size_t base_clock_mhz;
  size_t peak_gflops_per_second;
};

class KernelMetadata {
 public:
  KernelMetadata(
      std::string name,
      int32_t num_blocks,
      int32_t block_size,
      int32_t dynamic_shared_memory,
      int32_t static_shared_memory,
      uint16_t registers_per_thread)
        : name_(std::move(name)),
          num_blocks_(num_blocks),
          block_size_(block_size),
          dynamic_shared_memory_(dynamic_shared_memory),
          static_shared_memory_(static_shared_memory),
          registers_per_thread_(registers_per_thread) {}

  const std::string& name() const {
    return name_;
  }

  int32_t numBlocks() const {
    return num_blocks_;
  }

  int32_t blockSize() const {
    return block_size_;
  }

  int32_t dynamicSharedMemory() const {
    return dynamic_shared_memory_;
  }

  int32_t staticSharedMemory() const {
    return static_shared_memory_;
  }

  uint16_t registersPerThread() const {
    return registers_per_thread_;
  }

  /**
   * Returns the theoretical thread block occupancy for this kernel when running on a
   * specified GPU. The return value is zero if an error occurred.
   */
  uint32_t threadBlockOccupancy(const DeviceProperties& device, uint16_t registers_per_thread) const;

  /**
   * Returns the theoretical thread block occupancy for this kernel when running on a specified
   * GPU using the same number of registers as the kernel on the measured device. The return
   * value is zero if an error occurred.
   */
  uint32_t threadBlockOccupancy(const DeviceProperties& device) const;

  friend bool operator==(const KernelMetadata& lhs, const KernelMetadata& rhs);

 private:
  std::string name_;
  int32_t num_blocks_;
  int32_t block_size_;
  int32_t dynamic_shared_memory_;
  int32_t static_shared_memory_;
  uint16_t registers_per_thread_;
};

class KernelInstance {
 public:
  KernelInstance(KernelMetadata metadata, uint64_t run_time_ns)
    : metadata_(std::move(metadata)), run_time_ns_(run_time_ns) {}

  const KernelMetadata& metadata() const {
    return metadata_;
  }

  uint64_t runTimeNs() const {
    return run_time_ns_;
  }

  void addMetric(std::string name, double value);

  const std::vector<std::pair<std::string, double>>& metrics() const {
    return metrics_;
  }

 private:
  KernelMetadata metadata_;
  uint64_t run_time_ns_;
  std::vector<std::pair<std::string, double>> metrics_;
};

}
}

namespace std {

// Allow KernelMetadata to serve as a key in std::unordered_map
template <>
struct hash<habitat::cuda::KernelMetadata> {
  std::size_t operator()(const habitat::cuda::KernelMetadata& m) const {
    return habitat::utils::hash_combine(
        m.name(),
        m.numBlocks(),
        m.blockSize(),
        m.dynamicSharedMemory(),
        m.staticSharedMemory(),
        m.registersPerThread());
  }
};

}
