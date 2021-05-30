#include "model_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../cuda/kernel.h"

namespace py = pybind11;
using habitat::cuda::DeviceProperties;
using habitat::cuda::KernelInstance;
using habitat::cuda::KernelMetadata;

namespace habitat {
namespace frontend {

void bindModels(pybind11::module& m) {
  py::class_<DeviceProperties>(m, "DeviceProperties")
    .def(
        py::init<std::string, int, int, int, int, int, int, int, size_t, size_t, int, size_t, int, size_t, size_t>(),
        py::arg("name"),
        py::arg("compute_major"),
        py::arg("compute_minor"),
        py::arg("max_threads_per_block"),
        py::arg("max_threads_per_multiprocessor"),
        py::arg("regs_per_block"),
        py::arg("regs_per_multiprocessor"),
        py::arg("warp_size"),
        py::arg("shared_mem_per_block"),
        py::arg("shared_mem_per_multiprocessor"),
        py::arg("num_sms"),
        py::arg("shared_mem_per_block_optin"),
        py::arg("mem_bandwidth_gb"),
        py::arg("base_clock_mhz"),
        py::arg("peak_gflops_per_second"))
    .def("__repr__", [](const DeviceProperties& self) {
      return std::string("DeviceProperties(name=" + self.name + ")");
    }, py::return_value_policy::move)
    .def_property_readonly("name", [](const DeviceProperties& self) {
      return self.name;
    })
    .def_property_readonly("num_sms", [](const DeviceProperties& self) {
      return self.num_sms;
    })
    .def_property_readonly("mem_bandwidth_gb", [](const DeviceProperties& self) {
      return self.mem_bandwidth_gb;
    })
    .def_property_readonly("compute_capability", [](const DeviceProperties& self) {
      return py::make_tuple(self.compute_major, self.compute_minor);
    })
    .def_property_readonly("base_clock_mhz", [](const DeviceProperties& self) {
      return self.base_clock_mhz;
    })
    .def_property_readonly("peak_gflops_per_second", [](const DeviceProperties& self) {
      return self.peak_gflops_per_second;
    });

  py::class_<KernelInstance>(m, "KernelInstance")
    .def_property_readonly("name", [](const KernelInstance& self) {
      return self.metadata().name();
    }, py::return_value_policy::reference_internal)
    .def_property_readonly("run_time_ns", &KernelInstance::runTimeNs, py::return_value_policy::reference_internal)
    .def_property_readonly("num_blocks", [](const KernelInstance& self) { return self.metadata().numBlocks(); })
    .def_property_readonly("metrics", &KernelInstance::metrics, py::return_value_policy::reference_internal)
    .def_property_readonly("metadata", [](const KernelInstance& self) {
      py::dict metadata;
      const KernelMetadata& kernel_metadata = self.metadata();
      metadata["name"] = kernel_metadata.name();
      metadata["num_blocks"] = kernel_metadata.numBlocks();
      metadata["block_size"] = kernel_metadata.blockSize();
      metadata["static_shared_memory"] = kernel_metadata.staticSharedMemory();
      metadata["dynamic_shared_memory"] = kernel_metadata.dynamicSharedMemory();
      metadata["registers_per_thread"] = kernel_metadata.registersPerThread();
      return metadata;
    })
    .def("thread_block_occupancy", [](
        const KernelInstance& self, const DeviceProperties& device, int registers_per_thread) {
      if (registers_per_thread < 0) {
        return self.metadata().threadBlockOccupancy(device);
      } else {
        return self.metadata().threadBlockOccupancy(device, registers_per_thread);
      }
    }, py::arg("device"), py::arg("registers_per_thread") = -1);
}

}
}
