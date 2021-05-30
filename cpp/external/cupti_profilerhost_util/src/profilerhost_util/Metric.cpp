#include <Metric.h>
#include <Parser.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <iostream>
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
        namespace Config {

            bool GetRawMetricRequests(NVPA_MetricsContext* pMetricsContext,
                                      std::vector<std::string> metricNames,
                                      std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                                      std::vector<std::string>& temp) {
                std::string reqName;
                bool isolated = true;
                bool keepInstances = true;

                for (auto& metricName : metricNames)
                {
                    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
                    /* Bug in collection with collection of metrics without instances, keep it to true*/
                    keepInstances = true;
                    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
                    getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
                    getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));

                    for (const char** ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies)
                    {
                        temp.push_back(*ppMetricDependencies);
                    }
                    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
                    getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams));
                }

                for (auto& rawMetricName : temp)
                {
                    NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
                    metricRequest.pMetricName = rawMetricName.c_str();
                    metricRequest.isolated = isolated;
                    metricRequest.keepInstances = keepInstances;
                    rawMetricRequests.push_back(metricRequest);
                }

                return true;
            }

            bool GetConfigImage(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& configImage)
            {
                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                std::vector<std::string> temp;
                GetRawMetricRequests(metricsContext, metricNames, rawMetricRequests, temp);

                NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
                metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
                metricsConfigOptions.pChipName = chipName.c_str();
                NVPA_RawMetricsConfig* pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));

                NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
                rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
                SCOPE_EXIT([&]() { NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams); });

                NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
                beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

                NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
                addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

                NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
                endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

                NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
                generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

                NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
                getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                getConfigImageParams.bytesAllocated = 0;
                getConfigImageParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                configImage.resize(getConfigImageParams.bytesCopied);

                getConfigImageParams.bytesAllocated = configImage.size();
                getConfigImageParams.pBuffer = &configImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                return true;
            }

            bool GetCounterDataPrefixImage(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& counterDataImagePrefix)
            {
                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                std::vector<std::string> temp;
                GetRawMetricRequests(metricsContext, metricNames, rawMetricRequests, temp);

                NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
                counterDataBuilderCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

                NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
                counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                SCOPE_EXIT([&]() { NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams); });

                NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

                size_t counterDataPrefixSize = 0;
                NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
                getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                getCounterDataPrefixParams.bytesAllocated = 0;
                getCounterDataPrefixParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

                getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
                getCounterDataPrefixParams.pBuffer = &counterDataImagePrefix[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                return true;
            }
        }
    }
}
