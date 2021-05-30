#include "new_cupti_profiler.h"

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <cuda.h>
#include <Metric.h>
#include <Eval.h>

#include <stdio.h>
#include <stdlib.h>

#include "cupti_macros.h"
#include "metrics.h"

// NOTE: The code in this file is adapted from the CUPTI autorange_profiling sample.

#define NVPW_API_CALL(apiFuncCall)                                         \
do {                                                                       \
  NVPA_Status _status = apiFuncCall;                                       \
  if (_status != NVPA_STATUS_SUCCESS) {                                    \
    fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
            __FILE__, __LINE__, #apiFuncCall, _status);                    \
    throw std::runtime_error("NVPW API call failed!");                     \
  }                                                                        \
} while (0)

namespace {

class ProfilingSession {
 public:
  ProfilingSession(
      NVPA_MetricsContext* metrics_context,
      const std::string& chip_name,
      const std::string& metric,
      int num_kernels,
      const std::vector<uint8_t>& image_prefix,
      const std::vector<uint8_t>& config_image);

  void startProfiling();
  void stopProfiling();

  std::vector<habitat::cuda::KernelMetric> getMeasuredMetrics();

 private:
  void createCounterDataImage();

  CUpti_ProfilerReplayMode profiler_replay_mode_;
  CUpti_ProfilerRange profiler_range_;
  const std::string& metric_;
  NVPA_MetricsContext* metrics_context_;
  const std::string& chip_name_;
  int num_kernels_;

  const std::vector<uint8_t>& image_prefix_;
  const std::vector<uint8_t>& config_image_;

  // Data buffers
  std::vector<uint8_t> counter_data_image_;
  std::vector<uint8_t> counter_data_scratch_buffer_;
};

ProfilingSession::ProfilingSession(
    NVPA_MetricsContext* metrics_context,
    const std::string& chip_name,
    const std::string& metric,
    int num_kernels,
    const std::vector<uint8_t>& image_prefix,
    const std::vector<uint8_t>& config_image)
  : profiler_replay_mode_(CUPTI_KernelReplay),
    profiler_range_(CUPTI_AutoRange),
    metric_(metric),
    metrics_context_(metrics_context),
    chip_name_(chip_name),
    num_kernels_(num_kernels),
    image_prefix_(image_prefix),
    config_image_(config_image) {
  createCounterDataImage();
}

void ProfilingSession::createCounterDataImage() {
  CUpti_Profiler_CounterDataImageOptions options;
  options.pCounterDataPrefix = &image_prefix_[0];
  options.counterDataPrefixSize = image_prefix_.size();
  options.maxNumRanges = num_kernels_;
  options.maxNumRangeTreeNodes = num_kernels_;
  options.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculate_size_params =
      {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculate_size_params.pOptions = &options;
  calculate_size_params.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculate_size_params));

  CUpti_Profiler_CounterDataImage_Initialize_Params init_params =
      {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  init_params.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  init_params.pOptions = &options;
  init_params.counterDataImageSize = calculate_size_params.counterDataImageSize;

  counter_data_image_.resize(calculate_size_params.counterDataImageSize);
  init_params.pCounterDataImage = &counter_data_image_[0];
  CUPTI_CALL(cuptiProfilerCounterDataImageInitialize(&init_params));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratch_size_params =
      {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratch_size_params.counterDataImageSize = calculate_size_params.counterDataImageSize;
  scratch_size_params.pCounterDataImage = init_params.pCounterDataImage;
  CUPTI_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratch_size_params));

  counter_data_scratch_buffer_.resize(scratch_size_params.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params init_scratch_buffer =
      {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  init_scratch_buffer.counterDataImageSize = calculate_size_params.counterDataImageSize;

  init_scratch_buffer.pCounterDataImage = init_params.pCounterDataImage;
  init_scratch_buffer.counterDataScratchBufferSize = scratch_size_params.counterDataScratchBufferSize;
  init_scratch_buffer.pCounterDataScratchBuffer = &counter_data_scratch_buffer_[0];

  CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&init_scratch_buffer));
}

void ProfilingSession::startProfiling() {
  CUpti_Profiler_BeginSession_Params begin_session_params = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  begin_session_params.ctx = NULL;
  begin_session_params.counterDataImageSize = counter_data_image_.size();
  begin_session_params.pCounterDataImage = &counter_data_image_[0];
  begin_session_params.counterDataScratchBufferSize = counter_data_scratch_buffer_.size();
  begin_session_params.pCounterDataScratchBuffer = &counter_data_scratch_buffer_[0];
  begin_session_params.range = profiler_range_;
  begin_session_params.replayMode = profiler_replay_mode_;
  begin_session_params.maxRangesPerPass = num_kernels_;
  begin_session_params.maxLaunchesPerPass = num_kernels_;

  CUPTI_CALL(cuptiProfilerBeginSession(&begin_session_params));

  CUpti_Profiler_SetConfig_Params set_config_params = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  set_config_params.pConfig = &config_image_[0];
  set_config_params.configSize = config_image_.size();

  // NOTE: This is for KernelReplay mode (CUPTI performs the kernel replay for us)
  set_config_params.passIndex = 0;
  CUPTI_CALL(cuptiProfilerSetConfig(&set_config_params));

  CUpti_Profiler_EnableProfiling_Params enable_profiling_params = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
  CUPTI_CALL(cuptiProfilerEnableProfiling(&enable_profiling_params));
}

void ProfilingSession::stopProfiling() {
  CUpti_Profiler_DisableProfiling_Params disable_profiling_params =
      {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_CALL(cuptiProfilerDisableProfiling(&disable_profiling_params));

  CUpti_Profiler_UnsetConfig_Params unset_config_params = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_CALL(cuptiProfilerUnsetConfig(&unset_config_params));

  CUpti_Profiler_EndSession_Params end_session_params = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_CALL(cuptiProfilerEndSession(&end_session_params));
}

std::vector<habitat::cuda::KernelMetric> ProfilingSession::getMeasuredMetrics() {
  std::vector<NV::Metric::Eval::MetricNameValue> metric_name_value_map;
  bool succeeded = NV::Metric::Eval::GetMetricGpuValue(
      metrics_context_, chip_name_, counter_data_image_, {metric_}, metric_name_value_map);
  if (!succeeded) {
    return {};
  }

  // NOTE: This works because we only measure 1 metric right now
  std::vector<habitat::cuda::KernelMetric> measured_metrics;
  for (auto& metric : metric_name_value_map) {
    for (auto& range : metric.rangeNameMetricValueMap) {
      measured_metrics.push_back(habitat::cuda::KernelMetric(range.first, range.second));
    }
  }

  return measured_metrics;
}

class ProfilingGuard {
 public:
  explicit ProfilingGuard(ProfilingSession& session) : session_(session) {
    session_.startProfiling();
  }

  ~ProfilingGuard() {
    session_.stopProfiling();
  }

 private:
  ProfilingSession& session_;
};

}

namespace habitat {
namespace cuda {

class NewCuptiProfiler::State {
 public:
  State() : metrics_context_(nullptr) {
    CUpti_Profiler_Initialize_Params initialize_params = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerInitialize(&initialize_params));

    CUpti_Device_GetChipName_Params chip_name_params = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    // NOTE: We always use device id 0
    chip_name_params.deviceIndex = 0;
    CUPTI_CALL(cuptiDeviceGetChipName(&chip_name_params));
    chip_name_ = std::string(chip_name_params.pChipName);

    NVPW_InitializeHost_Params initialize_host_params = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_API_CALL(NVPW_InitializeHost(&initialize_host_params));

    NVPW_CUDA_MetricsContext_Create_Params metrics_context_create_params =
        {NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE};
    metrics_context_create_params.pChipName = chip_name_.c_str();
    NVPW_API_CALL(NVPW_CUDA_MetricsContext_Create(&metrics_context_create_params));
    metrics_context_ = metrics_context_create_params.pMetricsContext;
  }

  ~State() {
    NVPW_MetricsContext_Destroy_Params metrics_context_destroy_params =
        {NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE};
    metrics_context_destroy_params.pMetricsContext = metrics_context_;
    NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metrics_context_destroy_params);
    metrics_context_ = nullptr;

    CUpti_Profiler_DeInitialize_Params deinitialize_params = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    // NOTE: We don't error check because this is the destructor and so
    //       throwing an exception on an error here won't really help
    cuptiProfilerDeInitialize(&deinitialize_params);
  }

  const std::vector<uint8_t>& getConfigImage(const std::string& metric) const {
    auto it = config_images_.find(metric);
    if (it != config_images_.end()) {
      return it->second;
    }

    auto inserted = config_images_.emplace(std::make_pair<std::string, std::vector<uint8_t>>(std::string(metric), {}));
    if (!NV::Metric::Config::GetConfigImage(metrics_context_, chip_name_, {metric}, inserted.first->second)) {
      throw std::runtime_error("Failed to create config_image!");
    }
    return inserted.first->second;
  }

  const std::vector<uint8_t>& getImagePrefix(const std::string& metric) const {
    auto it = image_prefixes_.find(metric);
    if (it != image_prefixes_.end()) {
      return it->second;
    }

    auto inserted = image_prefixes_.emplace(
        std::make_pair<std::string, std::vector<uint8_t>>(std::string(metric), {}));
    if (!NV::Metric::Config::GetCounterDataPrefixImage(
          metrics_context_, chip_name_, {metric}, inserted.first->second)) {
      throw std::runtime_error("Failed to create counter_data_image_prefix!");
    }
    return inserted.first->second;
  }

  const std::string& chipName() const {
    return chip_name_;
  }

  NVPA_MetricsContext* metricsContext() const {
    return metrics_context_;
  }

 private:
  std::string chip_name_;
  mutable std::unordered_map<std::string, std::vector<uint8_t>> config_images_;
  mutable std::unordered_map<std::string, std::vector<uint8_t>> image_prefixes_;
  NVPA_MetricsContext* metrics_context_;
};

NewCuptiProfiler::NewCuptiProfiler() {}
NewCuptiProfiler::~NewCuptiProfiler() = default;

void NewCuptiProfiler::initialize() const {
  state_ = std::make_unique<State>();
}

void NewCuptiProfiler::profile(
    const std::string& metric_name,
    std::function<void(void)> runnable,
    std::vector<KernelInstance>& kernels) const {
  if (state_ == nullptr) {
    initialize();
  }

  ProfilingSession session(
      state_->metricsContext(),
      state_->chipName(),
      metric_name,
      kernels.size(),
      state_->getImagePrefix(metric_name),
      state_->getConfigImage(metric_name));
  {
    ProfilingGuard guard(session);
    runnable();
  }

  CuptiProfiler::addMetrics(kernels, session.getMeasuredMetrics(), metric_name);
}

}
}
