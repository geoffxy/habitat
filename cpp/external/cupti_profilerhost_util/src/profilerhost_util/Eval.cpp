#include <Eval.h>
#include <Parser.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
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
        namespace Eval {
            std::string GetHwUnit(const std::string& metricName)
            {
                return metricName.substr(0, metricName.find("__", 0));
            }

            bool GetMetricGpuValue(NVPA_MetricsContext* metricsContext, std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames, std::vector<MetricNameValue>& metricNameValueMap) {
                if (!counterDataImage.size()) {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = &counterDataImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

                std::vector<std::string> reqName;
                reqName.resize(metricNames.size());

                bool isolated = true;
                bool keepInstances = true;
                std::vector<const char*> metricNamePtrs;
                metricNameValueMap.resize(metricNames.size());

                for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);
                    metricNamePtrs.push_back(reqName[metricIndex].c_str());
                    metricNameValueMap[metricIndex].metricName = metricNames[metricIndex];
                    metricNameValueMap[metricIndex].numRanges = getNumRangesParams.numRanges;
                }

                for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
                    std::vector<const char*> descriptionPtrs;

                    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                    getRangeDescParams.pCounterDataImage = &counterDataImage[0];
                    getRangeDescParams.rangeIndex = rangeIndex;
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                    descriptionPtrs.resize(getRangeDescParams.numDescriptions);

                    getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                    std::string rangeName;
                    for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                    {
                        if (descriptionIndex)
                        {
                            rangeName += "/";
                        }
                        rangeName += descriptionPtrs[descriptionIndex];
                    }

                    std::vector<double> gpuValues;
                    gpuValues.resize(metricNames.size());
                    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
                    setCounterDataParams.pMetricsContext = metricsContext;
                    setCounterDataParams.pCounterDataImage = &counterDataImage[0];
                    setCounterDataParams.isolated = true;
                    setCounterDataParams.rangeIndex = rangeIndex;
                    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

                    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
                    evalToGpuParams.pMetricsContext = metricsContext;
                    evalToGpuParams.numMetrics = metricNamePtrs.size();
                    evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
                    evalToGpuParams.pMetricValues = &gpuValues[0];
                    NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);
                    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                        metricNameValueMap[metricIndex].rangeNameMetricValueMap.push_back(std::make_pair(rangeName, gpuValues[metricIndex]));
                    }
                }

                return true;
            }

            bool PrintMetricValues(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames) {
                if (!counterDataImage.size()) {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = &counterDataImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

                std::vector<std::string> reqName;
                reqName.resize(metricNames.size());
                bool isolated = true;
                bool keepInstances = true;
                std::vector<const char*> metricNamePtrs;
                for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);
                    metricNamePtrs.push_back(reqName[metricIndex].c_str());
                }

                for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
                    std::vector<const char*> descriptionPtrs;

                    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                    getRangeDescParams.pCounterDataImage = &counterDataImage[0];
                    getRangeDescParams.rangeIndex = rangeIndex;
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                    
                    descriptionPtrs.resize(getRangeDescParams.numDescriptions);
                    
                    getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                    std::string rangeName;
                    for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                    {
                        if (descriptionIndex)
                        {
                            rangeName += "/";
                        }
                        rangeName += descriptionPtrs[descriptionIndex];
                    }

                    const bool isolated = true;
                    std::vector<double> gpuValues;
                    gpuValues.resize(metricNames.size());

                    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
                    setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    setCounterDataParams.pCounterDataImage = &counterDataImage[0];
                    setCounterDataParams.isolated = true;
                    setCounterDataParams.rangeIndex = rangeIndex;
                    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

                    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
                    evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    evalToGpuParams.numMetrics = metricNamePtrs.size();
                    evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
                    evalToGpuParams.pMetricValues = &gpuValues[0];
                    NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);

                    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                        std::cout << "rangeName: " << rangeName << "\tmetricName: " << metricNames[metricIndex] << "\tgpuValue: "  << gpuValues[metricIndex] << std::endl;
                    }
                }
                return true;
            }
        }
    }
}
