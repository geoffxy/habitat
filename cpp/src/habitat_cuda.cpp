#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>

#include "cuda/diagnostics.h"
#include "frontend/model_bindings.h"
#include "frontend/profiler.h"

namespace py = pybind11;

PYBIND11_MODULE(habitat_cuda, m) {
  habitat::frontend::bindModels(m);

  m.def("profile", [](py::function runnable_python, const std::string& metric) {
    std::function<void()> runnable = [runnable_python]() {
      runnable_python();
    };
    if (metric.size() == 0) {
      return habitat::frontend::profile(std::move(runnable));
    } else {
      return habitat::frontend::profile(std::move(runnable), metric);
    }
  }, py::arg("runnable"), py::arg("metric") = "", py::return_value_policy::move);

  m.def("set_cache_metrics", [](bool should_cache) {
    habitat::frontend::setCacheMetrics(should_cache);
  }, py::arg("should_cache"));

  m.def_submodule("_diagnostics")
    .def("run_flop_test", [](size_t num_blocks, size_t threads_per_block) {
      habitat::cuda::diagnostics::run_flop_test(num_blocks, threads_per_block);
    }, py::arg("num_blocks") = 8192, py::arg("threads_per_block") = 256);
}
