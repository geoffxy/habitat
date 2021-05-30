#[=======================================================================[.rst:
FindNVPerf
---------

Finds the NVPerf library.

Note that NVPerf is usually distributed with CUPTI. Therefore to specify a
custom location, set the ``CUPTI_DIR`` environment variable. This find module
file will look under ``$CUPTI_DIR/lib/x64`` for the NVPerf shared library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``NVPerf_FOUND``
  True if the system has NVPerf.
``NVPerf_LIBRARIES``
  Libraries needed to link to NVPerf.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``NVPerf_HOST_LIBRARY``
  The path to the NVPerf host library.
``NVPerf_TARGET_LIBRARY``
  The path to the NVPerf target library.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(DEFINED ENV{CUDA_HOME})
  SET(CUPTI_CUDA_HOME "$ENV{CUDA_HOME}")
endif()

if(DEFINED ENV{CUPTI_DIR})
  SET(CUPTI_DIR "$ENV{CUPTI_DIR}")
endif()

find_library(NVPerf_HOST_LIBRARY nvperf_host
  HINTS
  ${CUPTI_DIR}/lib/x64
  ${CUPTI_CUDA_HOME}/extras/CUPTI/lib64
  /usr/local/cuda/extras/CUPTI/lib64
)

find_library(NVPerf_TARGET_LIBRARY nvperf_target
  HINTS
  ${CUPTI_DIR}/lib/x64
  ${CUPTI_CUDA_HOME}/extras/CUPTI/lib64
  /usr/local/cuda/extras/CUPTI/lib64
)

find_package_handle_standard_args(NVPerf
  DEFAULT_MSG
  NVPerf_HOST_LIBRARY
  NVPerf_TARGET_LIBRARY
)

if(NVPerf_FOUND)
  set(NVPerf_LIBRARIES ${NVPerf_HOST_LIBRARY} ${NVPerf_TARGET_LIBRARY})
  message(STATUS "Found NVPerf libraries: ${NVPerf_LIBRARIES}")
endif()
