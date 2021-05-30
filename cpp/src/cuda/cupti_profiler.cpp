#include "cupti_profiler.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>

#include "legacy_cupti_profiler.h"
#include "new_cupti_profiler.h"
#include "cuda_macros.h"

namespace habitat {
namespace cuda {

std::unique_ptr<CuptiProfiler> CuptiProfiler::create() {
  cudaDeviceProp properties;
  RUNTIME_API_CALL(cudaGetDeviceProperties(&properties, 0));
  if (properties.major >= 7) {
    // The new profiler is used for Volta and newer GPUs
    return std::unique_ptr<CuptiProfiler>(new NewCuptiProfiler());
  } else {
    return std::unique_ptr<CuptiProfiler>(new LegacyCuptiProfiler());
  }
}

void CuptiProfiler::addMetrics(
    std::vector<KernelInstance>& kernels, const std::vector<KernelMetric>& metrics, const std::string& metric_name) {
  // Right now our "profiling model" is that only one metric is measured. NVIDIA's CUPTI documentation is
  // unfortunately not that clear, so we don't know exactly how profiled "ranges" map to kernels. Right now
  // we assume that the order of metric values matches the order in which the kernels were launched.
  if (kernels.size() != metrics.size()) {
    // Not sure how to proceed - we should throw an exception to be safe.
    throw std::runtime_error("Encountered a KernelInstance and metrics vector size mismatch!");
  }

  for (size_t i = 0; i < kernels.size(); i++) {
    kernels.at(i).addMetric(metric_name, metrics.at(i).metricValue());
  }
}

}
}
