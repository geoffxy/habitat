#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "cupti_profiler.h"
#include "kernel.h"

namespace habitat {
namespace cuda {

/**
 * Uses the legacy CUPTI metrics APIs (Pascal and older GPUs).
 */
class LegacyCuptiProfiler : public CuptiProfiler {
 public:
  void profile(
      const std::string& metric_name,
      std::function<void(void)> runnable,
      std::vector<KernelInstance>& kernels) const override;

 private:
  LegacyCuptiProfiler();
  ~LegacyCuptiProfiler();
  friend class CuptiProfiler;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}
}
