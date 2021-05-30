#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../cuda/kernel.h"

namespace habitat {
namespace frontend {

void setCacheMetrics(bool should_cache);

std::vector<cuda::KernelInstance> profile(std::function<void()> runnable);

std::vector<cuda::KernelInstance> profile(
    std::function<void()> runnable, const std::string& metric);

}
}
