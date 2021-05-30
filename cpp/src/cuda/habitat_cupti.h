#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "kernel.h"
#include "metrics.h"
#include "cupti_profiler.h"

namespace habitat {
namespace cuda {

class CuptiManager;
class CuptiTracer;

namespace detail {

void cuptiTracerDeleter(CuptiTracer* tracer);

}

/**
 * Accumulates kernel invocations during its lifetime.
 */
class CuptiTracer {
 public:
  using Ptr = std::unique_ptr<CuptiTracer, decltype(&detail::cuptiTracerDeleter)>;

  CuptiTracer(const CuptiTracer&) = delete;
  CuptiTracer& operator=(const CuptiTracer&) = delete;

  /**
   * Grab all the recorded kernels.
   */
  std::vector<KernelInstance>&& kernels() &&;

  /**
   * Assuming the same operation was recorded multiple times, this returns the kernels in the last operation.
   */
  std::vector<KernelInstance> lastKernels(size_t num_iterations) const;

 private:
  CuptiTracer();
  std::vector<KernelInstance> kernels_;

  friend class CuptiManager;
};

/**
 * Singleton that manages bindings to CUPTI.
 */
class CuptiManager {
 public:
  static CuptiManager& instance();
  CuptiTracer::Ptr allocateTracer();

  /**
   * Measures the specified metric for the kernels invoked by a given runnable.
   *
   * The metrics will be appended to the respective KernelInstances passed in to
   * this method.
   *
   * NOTE: The kernels invoked by the runnable must already exist as KernelInstances
   *       (i.e. the tracer must be used first). This is because some of the CUPTI
   *       metrics APIs require the execution time of the kernels.
   */
  void measureMetric(
      const std::string& metric_name,
      std::function<void(void)> runnable,
      std::vector<KernelInstance>& kernels);

  /**
   * Ensures CUPTI is unloaded from the process.
   *
   * This throws an exception if there are still CuptiTracers bound to the manager.
   */
  void unloadCupti();

  /**
   * This method is NOT intended to be called directly by end users.
   *
   * This method is called by CUPTI when recording a trace of the kernels that have been executed.
   */
  void newKernelInstance(KernelInstance info);

  /**
   * Use this method to control whether or not the manager should cache kernel metrics.
   * This can be useful because metrics gathering is very slow on Volta and newer generations.
   *
   * By default caching is disabled.
   */
  void setCacheMetrics(bool should_cache);

  /**
   * Returns whether or not the manager is caching kernel metrics.
   */
  bool isCachingMetrics() const;

  CuptiManager(const CuptiManager&) = delete;
  CuptiManager& operator=(const CuptiManager&) = delete;

 private:
  CuptiManager();
  std::unique_ptr<CuptiProfiler> profiler_;
  std::vector<CuptiTracer*> tracers_;
  std::unordered_map<KernelMetadata, std::unordered_map<std::string, double>> metrics_cache_;
  bool callbacks_bound_;
  bool should_cache_metrics_;

  friend void detail::cuptiTracerDeleter(CuptiTracer *tracer);
};

/**
 * RAII-helper to ensure CUPTI is unloaded.
 */
class CuptiGuard {
 public:
  ~CuptiGuard() {
    CuptiManager::instance().unloadCupti();
  }
};

}
}
