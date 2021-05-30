#include "habitat_cupti.h"

#include <cuda.h>
#include <cupti.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <utility>

#include "cupti_profiler.h"

namespace habitat {
namespace cuda {

CuptiManager::CuptiManager()
  : profiler_(CuptiProfiler::create()),
    callbacks_bound_(false),
    should_cache_metrics_(false) {}

CuptiManager& CuptiManager::instance() {
  static std::unique_ptr<CuptiManager> manager(new CuptiManager());
  return *manager;
}

// CuptiManager::allocateTracer() is defined in cupti_tracer.cpp

void CuptiManager::unloadCupti() {
  if (tracers_.size() > 0) {
    throw std::runtime_error("Cannot unload CUPTI because at least one tracer is still bound.");
  }
  cudaDeviceSynchronize();
  cuptiActivityFlushAll(0);
  cuptiFinalize();
  callbacks_bound_ = false;
}

void CuptiManager::newKernelInstance(KernelInstance info) {
  for (auto& tracer : tracers_) {
    tracer->kernels_.push_back(info);
  }
}

void CuptiManager::measureMetric(
    const std::string& metric_name,
    std::function<void(void)> runnable,
    std::vector<KernelInstance>& kernels) {
  if (tracers_.size() > 0) {
    throw std::runtime_error(
        "A CuptiTracer instance still exists. Metrics cannot be measured when tracing is being performed.");
  }

  // If the cache can fulfil the metrics request, just use the cache
  if (should_cache_metrics_ &&
    std::all_of(kernels.cbegin(), kernels.cend(), [&](auto& kernel) {
      auto it = metrics_cache_.find(kernel.metadata());
      return it != metrics_cache_.end() && it->second.count(metric_name) > 0;
    })) {

    for (auto& kernel : kernels) {
      auto it = metrics_cache_.find(kernel.metadata());
      double value = it->second.at(metric_name);
      kernel.addMetric(metric_name, value);
    }
    return;
  }

  // Otherwise, run the profiler
  profiler_->profile(metric_name, runnable, kernels);

  // Update the cache if needed
  if (should_cache_metrics_) {
    for (auto& kernel : kernels) {
      auto& metrics_map = metrics_cache_[kernel.metadata()];
      for (const auto& metric_pair : kernel.metrics()) {
        // NOTE: Map insert/emplace will only do the actual insertion if the key has not been used.
        //       Otherwise it just returns an iterator to the existing value.
        metrics_map.emplace(metric_pair);
      }
    }
  }
}

void CuptiManager::setCacheMetrics(bool should_cache) {
  if (should_cache_metrics_ && !should_cache) {
    metrics_cache_.clear();
  }
  should_cache_metrics_ = should_cache;
}

bool CuptiManager::isCachingMetrics() const {
  return should_cache_metrics_;
}

}
}
