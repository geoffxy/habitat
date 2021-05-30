#include "profiler.h"

#include "../cuda/habitat_cupti.h"

using habitat::cuda::CuptiManager;
using habitat::cuda::CuptiTracer;
using habitat::cuda::KernelInstance;

namespace {

std::vector<KernelInstance> measureRunTimes(
    const std::function<void()>& runnable) {
  std::vector<KernelInstance> kernels;
  {
    CuptiTracer::Ptr tracer = CuptiManager::instance().allocateTracer();
    runnable();
    kernels = std::move(*tracer).kernels();
  }
  return kernels;
}

}

namespace habitat {
namespace frontend {

void setCacheMetrics(bool should_cache) {
  CuptiManager::instance().setCacheMetrics(should_cache);
}

std::vector<KernelInstance> profile(std::function<void()> runnable) {
  return measureRunTimes(runnable);
}

std::vector<KernelInstance> profile(
    std::function<void()> runnable, const std::string& metric) {
  std::vector<cuda::KernelInstance> kernels(measureRunTimes(runnable));
  CuptiManager::instance().measureMetric(metric, std::move(runnable), kernels);
  return kernels;
}

}
}
