#[=======================================================================[.rst:
FindCUPTI
---------

Finds the CUPTI library.

To specify a custom location, set the ``CUPTI_DIR`` environment variable.
This find module file will look under ``$CUPTI_DIR/include`` and ``$CUPTI_DIR/lib/x64``
for the headers and CUPTI shared library respectively.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CUPTI_FOUND``
  True if the system has CUPTI.
``CUPTI_INCLUDE_DIRS``
  Include directories needed to use CUPTI.
``CUPTI_LIBRARIES``
  Libraries needed to link to CUPTI.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CUPTI_INCLUDE_DIR``
  The directory containing ``cupti.h``.
``CUPTI_LIBRARY``
  The path to the CUPTI library.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(DEFINED ENV{CUDA_HOME})
  SET(CUPTI_CUDA_HOME "$ENV{CUDA_HOME}")
endif()

if(DEFINED ENV{CUPTI_DIR})
  SET(CUPTI_DIR "$ENV{CUPTI_DIR}")
endif()

find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS
  ${CUPTI_DIR}/include
  ${CUPTI_CUDA_HOME}/extras/CUPTI/include
  /usr/local/cuda/extras/CUPTI/include
)

find_library(CUPTI_LIBRARY cupti
  HINTS
  ${CUPTI_DIR}/lib/x64
  ${CUPTI_CUDA_HOME}/extras/CUPTI/lib64
  /usr/local/cuda/extras/CUPTI/lib64
)

find_package_handle_standard_args(CUPTI
  DEFAULT_MSG
  CUPTI_INCLUDE_DIR
  CUPTI_LIBRARY
)

if(CUPTI_FOUND)
  set(CUPTI_INCLUDE_DIRS ${CUPTI_INCLUDE_DIR})
  set(CUPTI_LIBRARIES ${CUPTI_LIBRARY})

  message(STATUS "Found CUPTI includes: ${CUPTI_INCLUDE_DIRS}")
  message(STATUS "Found CUPTI library: ${CUPTI_LIBRARIES}")
endif()
