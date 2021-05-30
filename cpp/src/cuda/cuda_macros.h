#pragma once

#include <cuda.h>
#include <stdexcept>

// This header should only be included in source files (i.e. .cpp files) that contain CUDA API calls.

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        throw std::runtime_error("CUDA Runtime API call failed.");             \
    }                                                                          \
} while (0)
