#pragma once

#include <cupti.h>
#include <cstdio>
#include <stdexcept>

#include "cupti_exceptions.h"

// This header should only be included in source files (i.e. .cpp files) that contain CUPTI API calls.

#define CUPTI_CALL(call)                                                 \
do {                                                                     \
  CUptiResult _status = call;                                            \
  if (_status != CUPTI_SUCCESS) {                                        \
    const char* message;                                                 \
    cuptiGetResultString(_status, &message);                             \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
            __FILE__, __LINE__, #call, message);                         \
    throw habitat::cuda::CuptiError::from(_status);                      \
  }                                                                      \
} while (0)
