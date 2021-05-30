#pragma once

#include <string>

namespace habitat {
namespace cuda {

class KernelMetric {
 public:
  KernelMetric(std::string kernel_name, double metric_value)
    : kernel_name_(std::move(kernel_name)),
      metric_value_(metric_value) {}

  const std::string& kernelName() const {
    return kernel_name_;
  }

  const double& metricValue() const {
    return metric_value_;
  }

 private:
  std::string kernel_name_;
  double metric_value_;
};

}
}
