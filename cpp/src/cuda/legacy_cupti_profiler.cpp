#include "legacy_cupti_profiler.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cupti.h>
#include <cuda.h>

#include "cupti_exceptions.h"
#include "cupti_macros.h"
#include "metrics.h"

// NOTE: The code in this file is adapted from the CUPTI callback_metric sample.

#define DRIVER_API_CALL(apiFuncCall)                                       \
do {                                                                       \
  CUresult _status = apiFuncCall;                                          \
  if (_status != CUDA_SUCCESS) {                                           \
    fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
            __FILE__, __LINE__, #apiFuncCall, _status);                    \
    throw std::runtime_error("CUDA Driver API call failed!");              \
  }                                                                        \
} while (0)

namespace {

class MetricData {
 public:
  static MetricData fromMetricName(const std::string& metric_name, CUdevice device) {
    CUpti_MetricID metric_id;
    CUPTI_CALL(cuptiMetricGetIdFromName(device, metric_name.c_str(), &metric_id));

    uint32_t event_num;
    CUPTI_CALL(cuptiMetricGetNumEvents(metric_id, &event_num));

    CUcontext context;
    DRIVER_API_CALL(cuCtxGetCurrent(&context));

    CUpti_EventGroupSets* group_sets;
    CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metric_id), &metric_id, &group_sets));

    return MetricData(metric_id, event_num, group_sets);
  }

  CUpti_MetricID metricId() const {
    return metric_id_;
  }

  uint32_t eventNum() const {
    return event_num_;
  }

  CUpti_EventGroupSets* groupSets() const {
    return group_sets_;
  }

 private:
  MetricData(
      CUpti_MetricID metric_id,
      uint32_t event_num,
      CUpti_EventGroupSets* group_sets)
    : metric_id_(metric_id),
      event_num_(event_num),
      group_sets_(group_sets) {}

  CUpti_MetricID metric_id_;
  uint32_t event_num_;
  CUpti_EventGroupSets* group_sets_;
};

struct EventData {
  EventData(uint32_t num_events) {
    event_ids.reserve(num_events);
    event_values.reserve(num_events);
  }

  std::vector<CUpti_EventID> event_ids;
  std::vector<uint64_t> event_values;
};

class ProfilingSession {
 public:
  ProfilingSession(
      std::function<void(void)> runnable,
      const std::vector<habitat::cuda::KernelInstance>& kernels,
      const MetricData& metric_data,
      CUdevice device);

  std::vector<habitat::cuda::KernelMetric> getMeasuredMetrics();

  // NOTE: These callbacks should not be called explicitly. They are called by CUPTI.
  void handleKernelCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cb_info);
  static void profilingCallback(
      void* session, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cb_info);

 private:
  void profile();
  void startProfiling();
  void stopProfiling();

  class Guard {
   public:
    Guard(ProfilingSession& session) : session_(session) {
      session_.startProfiling();
    }

    ~Guard() {
      session_.stopProfiling();
    }

   private:
    ProfilingSession& session_;
  };
  friend class Guard;

  std::function<void(void)> runnable_;
  const std::vector<habitat::cuda::KernelInstance>& kernels_;
  const MetricData& metric_data_;

  CUdevice device_;
  CUpti_SubscriberHandle subscriber_;

  // We use one EventData per kernel to store its related events
  uint32_t pass_idx_;
  uint32_t kernel_idx_;
  std::vector<EventData> recorded_events_;
};

ProfilingSession::ProfilingSession(
    std::function<void(void)> runnable,
    const std::vector<habitat::cuda::KernelInstance>& kernels,
    const MetricData& metric_data,
    CUdevice device)
  : runnable_(runnable),
    kernels_(kernels),
    metric_data_(metric_data),
    device_(device),
    pass_idx_(0),
    kernel_idx_(0) {
  recorded_events_.resize(kernels_.size(), EventData(metric_data_.eventNum()));
  profile();
}

void ProfilingSession::profilingCallback(
    void* session, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cb_info) {
  ((ProfilingSession*) session)->handleKernelCallback(domain, cbid, cb_info);
}

void ProfilingSession::startProfiling() {
  CUPTI_CALL(cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)ProfilingSession::profilingCallback, this));
  CUPTI_CALL(cuptiEnableCallback(
      1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
  CUPTI_CALL(cuptiEnableCallback(
      1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
}

void ProfilingSession::stopProfiling() {
  CUPTI_CALL(cuptiUnsubscribe(subscriber_));
}

void ProfilingSession::profile() {
  ProfilingSession::Guard guard(*this);
  for (pass_idx_ = 0; pass_idx_ < metric_data_.groupSets()->numSets; pass_idx_++) {
    kernel_idx_ = 0;
    runnable_();
  }
}

void ProfilingSession::handleKernelCallback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *cb_info) {
  // This callback is enabled only for launch so we shouldn't see anything else.
  if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    throw std::runtime_error("Unexpected CUPTI callback in CuptiProfiler!");
  }

  CUpti_EventGroupSet* group_set = metric_data_.groupSets()->sets + pass_idx_;

  // On entry, enable all the event groups being collected this pass
  if (cb_info->callbackSite == CUPTI_API_ENTER) {
    cudaDeviceSynchronize();

    CUPTI_CALL(cuptiSetEventCollectionMode(cb_info->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
    for (uint32_t i = 0; i < group_set->numEventGroups; i++) {
      uint32_t all = 1;
      CUPTI_CALL(cuptiEventGroupSetAttribute(
          group_set->eventGroups[i],
          CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
          sizeof(all),
          &all));
      CUPTI_CALL(cuptiEventGroupEnable(group_set->eventGroups[i]));
    }
  }

  // On exit, read and record event values
  if (cb_info->callbackSite == CUPTI_API_EXIT) {
    cudaDeviceSynchronize();

    // for each group, read the event values from the group and record
    // in metricData
    for (uint32_t i = 0; i < group_set->numEventGroups; i++) {
      CUpti_EventGroup group = group_set->eventGroups[i];

      CUpti_EventDomainID group_domain;
      size_t group_domain_size = sizeof(group_domain);
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &group_domain_size, &group_domain));

      uint32_t num_total_instances;
      size_t num_total_instances_size = sizeof(num_total_instances);
      CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
          device_,
          group_domain,
          CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
          &num_total_instances_size,
          &num_total_instances));

      uint32_t num_instances;
      size_t num_instances_size = sizeof(num_instances);
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &num_instances_size, &num_instances));

      uint32_t num_events;
      size_t num_events_size = sizeof(num_events);
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &num_events_size, &num_events));

      std::vector<CUpti_EventID> event_ids;
      event_ids.resize(num_events, 0);
      size_t event_ids_size = event_ids.size() * sizeof(CUpti_EventID);
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &event_ids_size, event_ids.data()));

      std::vector<uint64_t> event_values;
      event_values.resize(num_instances, 0);
      size_t event_values_size = event_values.size() * sizeof(uint64_t);

      if (kernel_idx_ >= recorded_events_.size()) {
        throw std::runtime_error("CUPTI event collection callback was called too many times!");
      }

      EventData& events_for_kernel = recorded_events_[kernel_idx_];

      for (CUpti_EventID event_id : event_ids) {
        CUPTI_CALL(cuptiEventGroupReadEvent(
            group, CUPTI_EVENT_READ_FLAG_NONE, event_id, &event_values_size, event_values.data()));
        if (events_for_kernel.event_ids.size() >= metric_data_.eventNum()) {
          fprintf(stderr, "ERROR: Too many events collected, metric expects only %d\n",
                  (int) metric_data_.eventNum());
          throw std::runtime_error("CUPTI event collection error!");
        }

        // sum collect event values from all instances
        uint64_t sum = 0;
        for (uint64_t value : event_values) {
          sum += value;
        }

        // normalize the event value to represent the total number of
        // domain instances on the device
        events_for_kernel.event_ids.push_back(event_id);
        events_for_kernel.event_values.push_back( (sum * num_total_instances) / num_instances);
      }
    }

    // Finished processing the current kernel
    kernel_idx_ += 1;

    for (uint32_t i = 0; i < group_set->numEventGroups; i++) {
      CUPTI_CALL(cuptiEventGroupDisable(group_set->eventGroups[i]));
    }
  }
}

std::vector<habitat::cuda::KernelMetric> ProfilingSession::getMeasuredMetrics() {
  std::vector<habitat::cuda::KernelMetric> metrics;

  for (uint32_t i = 0; i < kernels_.size(); i++) {
    const habitat::cuda::KernelInstance& kernel = kernels_.at(i);
    EventData& events = recorded_events_.at(i);

    if (events.event_values.size() != metric_data_.eventNum()) {
      fprintf(stderr, "ERROR: Expected %u metric events, got %zu\n",
              metric_data_.eventNum(), events.event_values.size());
      return {};
    }

    double metric_value_double = 0.;

    try {
      // use all the collected events to calculate the metric value
      CUpti_MetricValue metric_value;
      CUPTI_CALL(cuptiMetricGetValue(
          device_,
          metric_data_.metricId(),
          events.event_ids.size() * sizeof(CUpti_EventID),
          events.event_ids.data(),
          events.event_values.size() * sizeof(uint64_t),
          events.event_values.data(),
          kernel.runTimeNs(),
          &metric_value));

      CUpti_MetricValueKind value_kind;
      size_t value_kind_size = sizeof(value_kind);
      CUPTI_CALL(cuptiMetricGetAttribute(
          metric_data_.metricId(),
          CUPTI_METRIC_ATTR_VALUE_KIND,
          &value_kind_size,
          &value_kind));

      switch (value_kind) {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
          metric_value_double = metric_value.metricValueDouble;
          break;

        case CUPTI_METRIC_VALUE_KIND_UINT64:
          metric_value_double = static_cast<double>(metric_value.metricValueUint64);
          break;

        case CUPTI_METRIC_VALUE_KIND_INT64:
          metric_value_double = static_cast<double>(metric_value.metricValueInt64);
          break;

        case CUPTI_METRIC_VALUE_KIND_PERCENT:
          metric_value_double = static_cast<double>(metric_value.metricValuePercent);
          break;

        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
          metric_value_double = static_cast<double>(metric_value.metricValueThroughput);
          break;

        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
          metric_value_double = static_cast<double>(metric_value.metricValueUtilizationLevel);
          break;

        default:
          fprintf(stderr, "ERROR: Unknown value kind when processing metric!\n");
          continue;
      }

    } catch (habitat::cuda::CuptiError& error) {
      // From the CUPTI documentation, CUPTI_ERROR_INVALID_METRIC_VALID means:
      //
      //   The computed metric value cannot be represented in the metric's value type.
      //   For example, if the metric value type is unsigned and the computed metric value is negative.
      //
      // If this happens, we just set the metric to 0.
      if (error.errorCode() != CUPTI_ERROR_INVALID_METRIC_VALUE) {
        throw;
      }
    }

    metrics.push_back(habitat::cuda::KernelMetric(kernel.metadata().name(), metric_value_double));
  }

  return metrics;
}

}

namespace habitat {
namespace cuda {

class LegacyCuptiProfiler::Impl {
 public:
  Impl() {
    DRIVER_API_CALL(cuDeviceGet(&device_, 0));
  }

  const MetricData& getMetricData(const std::string& metric_name) const {
    auto it = metric_name_to_data_.find(metric_name);
    if (it != metric_name_to_data_.end()) {
      return it->second;
    }

    MetricData data = MetricData::fromMetricName(metric_name, device_);
    auto result = metric_name_to_data_.emplace(std::make_pair(std::string(metric_name), std::move(data)));
    return result.first->second;
  }

  CUdevice device() const {
    return device_;
  }

 private:
  CUdevice device_;
  mutable std::unordered_map<std::string, MetricData> metric_name_to_data_;
};

LegacyCuptiProfiler::LegacyCuptiProfiler() : impl_(new Impl()) {}

LegacyCuptiProfiler::~LegacyCuptiProfiler() = default;

void LegacyCuptiProfiler::profile(
    const std::string& metric_name,
    std::function<void(void)> runnable,
    std::vector<KernelInstance>& kernels) const {
  ProfilingSession session(runnable, kernels, impl_->getMetricData(metric_name), impl_->device());
  CuptiProfiler::addMetrics(kernels, session.getMeasuredMetrics(), metric_name);
}

}
}
