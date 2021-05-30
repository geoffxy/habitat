#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "cupti_profiler.h"
#include "kernel.h"
#include "metrics.h"

namespace habitat {
namespace cuda {

/**
 * Uses the new PerfWorks-based CUPTI profiling APIs (Volta and newer).
 */
class NewCuptiProfiler : public CuptiProfiler {
 public:
  void profile(
      const std::string& metric_name,
      std::function<void(void)> runnable,
      std::vector<KernelInstance>& kernels) const override;

 private:
  NewCuptiProfiler();
  ~NewCuptiProfiler();
  friend class CuptiProfiler;

  // We lazily initialize the profiler to prevent CUPTI from potentially
  // introducing overhead when profiling NOT is used.
  void initialize() const;

  class State;
  mutable std::unique_ptr<State> state_;
};

}
}
