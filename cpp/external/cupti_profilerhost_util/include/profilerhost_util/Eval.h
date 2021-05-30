#pragma once

#include <string>
#include <vector>
#include <nvperf_host.h>

namespace NV {
    namespace Metric {
        namespace Eval {
            struct MetricNameValue {
                std::string metricName;
                int numRanges;
                // <rangeName , metricValue> pair
                std::vector < std::pair<std::string, double> > rangeNameMetricValueMap;
            };


            /* Function to get aggregate metric value
             * @param[in]  chipName                 Chip name for which to get metric values
             * @param[in]  counterDataImage         Counter data image
             * @param[in]  metricNames              List of metrics to read from counter data image
             * @param[out] metricNameValueMap       Metric name value map
             */
            bool GetMetricGpuValue(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames, std::vector<MetricNameValue>& metricNameValueMap);

            bool PrintMetricValues(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames);

            }
    }
}
