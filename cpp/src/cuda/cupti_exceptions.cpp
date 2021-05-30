#include "cupti_exceptions.h"

#include <string>

namespace habitat {
namespace cuda {

CuptiError::CuptiError(CUptiResult error_code, const char* error_message)
  : std::runtime_error(std::string(error_message)),
    error_code_(error_code) {}

CuptiError CuptiError::from(CUptiResult error_code) {
  const char* message;
  cuptiGetResultString(error_code, &message);
  return CuptiError(error_code, message);
}

}
}

