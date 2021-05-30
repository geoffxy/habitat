set(CuptiProfilerHost "cupti_profilerhost_util")

add_library(${CuptiProfilerHost}
  src/profilerhost_util/Eval.cpp
  src/profilerhost_util/List.cpp
  src/profilerhost_util/Metric.cpp
)

target_link_libraries(${CuptiProfilerHost}
  PRIVATE
    CUDA::cupti
    ${NVPerf_LIBRARIES}
)

target_include_directories(${CuptiProfilerHost}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/c_util
    ${CMAKE_CURRENT_SOURCE_DIR}/include/profilerhost_util
  PRIVATE
    # We need to add this in addition to the CUDA::cupti
    # target above because this variable includes the
    # nvperf headers (and the CUDA::cupti target does not).
    ${CUPTI_INCLUDE_DIRS}
)
