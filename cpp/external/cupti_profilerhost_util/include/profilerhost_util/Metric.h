#pragma once

#include <string>
#include <vector>
#include <nvperf_host.h>

namespace NV {
    namespace Metric {
        namespace Config {
            /* Function to get Config image
            * @param[in]  chipName            Chip name for which configImage is to be generated
            * @param[in]  metricNames         List of metrics for which configImage is to be generated
            * @param[out] configImage         Generated configImage
            */
            bool GetConfigImage(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& configImage);

            /* Function to get CounterDataPrefix image
            * @param[in]  chipName                  Chip name for which counterDataImagePrefix is to be generated
            * @param[in]  metricNames               List of metrics for which counterDataImagePrefix is to be generated
            * @param[out] counterDataImagePrefix    Generated counterDataImagePrefix
            */
            bool GetCounterDataPrefixImage(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& counterDataImagePrefix);
        }
    }
}
