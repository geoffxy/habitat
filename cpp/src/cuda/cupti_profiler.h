#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "kernel.h"
#include "metrics.h"

namespace habitat {
namespace cuda {

class CuptiManager;

class CuptiProfiler {
 public:
  virtual ~CuptiProfiler() {}

  /**
   * Perform the profiling to measure the requested metric.
   */
  virtual void profile(
      const std::string& metric_name,
      std::function<void(void)> runnable,
      std::vector<KernelInstance>& kernels) const = 0;

 protected:
  CuptiProfiler() {}

  /**
   * Utility function used to add measured kernel metrics to their associated KernelInstances.
   */
  static void addMetrics(
      std::vector<KernelInstance>& kernels,
      const std::vector<KernelMetric>& metrics,
      const std::string& metric_name);

 private:
  static std::unique_ptr<CuptiProfiler> create();
  friend class CuptiManager;
};

}
}
