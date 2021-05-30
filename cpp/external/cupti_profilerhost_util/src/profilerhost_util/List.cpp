#include <List.h>
#include <iostream>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <ScopeExit.h>

#define RETURN_IF_NVPW_ERROR(retval, actual) \
    do { \
        if (NVPA_STATUS_SUCCESS != actual) { \
            fprintf(stderr, "FAILED: %s\n", #actual); \
            return retval; \
        } \
    } while (0)

namespace NV {
    namespace Metric {
        namespace Enum {
            bool ListSupportedChips() {
                NVPW_GetSupportedChipNames_Params getSupportedChipNames = { NVPW_GetSupportedChipNames_Params_STRUCT_SIZE };
                RETURN_IF_NVPW_ERROR(false, NVPW_GetSupportedChipNames(&getSupportedChipNames));
                std::cout << "\n Number of supported chips : " << getSupportedChipNames.numChipNames;
                std::cout << "\n List of supported chips : \n";

                for (size_t i = 0; i < getSupportedChipNames.numChipNames; i++) {
                    std::cout << " " << getSupportedChipNames.ppChipNames[i] << "\n";
                }

                return true;
            }

            bool ListMetrics(const char* chip, bool listSubMetrics) {

                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chip;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
                getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                getMetricNameBeginParams.hidePeakSubMetrics = !listSubMetrics;
                getMetricNameBeginParams.hidePerCycleSubMetrics = !listSubMetrics;
                getMetricNameBeginParams.hidePctOfPeakSubMetrics = !listSubMetrics;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricNames_Begin(&getMetricNameBeginParams));

                NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE };
                getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_GetMetricNames_End((NVPW_MetricsContext_GetMetricNames_End_Params *)&getMetricNameEndParams); });
                
                std::cout << getMetricNameBeginParams.numMetrics << " metrics in total on the chip\n Metrics List : \n";
                for (size_t i = 0; i < getMetricNameBeginParams.numMetrics; i++) {
                    std::cout << getMetricNameBeginParams.ppMetricNames[i] << "\n";
                }

                return true;
            }
        }
    }
}
