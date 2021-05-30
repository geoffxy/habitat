cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(habitat LANGUAGES C CXX CUDA)

# Include our custom find module files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find all dependencies
find_package(CUDAToolkit REQUIRED)
# We need CUPTI here because CUDAToolkit does not include the nvperf-related
# libraries and headers that we need.
find_package(CUPTI REQUIRED)
find_package(NVPerf REQUIRED)
add_subdirectory(external)

################################################################################
# Habitat Targets
################################################################################

# habitat_cuda: Builds a Python importable module that provides bindings to the
#               CUDA-related functionality used by Habitat.
set(HabitatCUDA "habitat_cuda")
pybind11_add_module(${HabitatCUDA} src/habitat_cuda.cpp)
set_property(TARGET ${HabitatCUDA} PROPERTY CXX_STANDARD 11)

# device_info: Builds a utility executable that prints information about the
#              underlying GPU device (e.g., number of SMs, memory bandwidth).
set(DeviceInfo "device_info")
add_executable(${DeviceInfo} src/device_info.cpp)
set_property(TARGET ${DeviceInfo} PROPERTY CXX_STANDARD 11)

################################################################################

# Add our source files
cmake_policy(SET CMP0076 NEW)
add_subdirectory(src)

# Specify dependencies for each target
target_link_libraries(${HabitatCUDA} PRIVATE habitat-cuda-lib)
set(HabitatCUDATransitiveDeps habitat-cuda-lib cupti_profilerhost_util)
target_link_libraries(${DeviceInfo} PRIVATE CUDA::cudart gflags::gflags)

# Since pybind11 modules are shared libraries, all the static libraries it
# depends on must be compiled as position independent code.
foreach(LIB ${HabitatCUDATransitiveDeps})
  set_property(TARGET ${LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

# Turn on all compile warnings
set(AllTargets ${HabitatCUDA})
foreach(TGT ${AllTargets})
  if(CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(${TGT} PRIVATE "-Wall")
  endif()
endforeach()
