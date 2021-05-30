#pragma once

#include <stdexcept>
#include <cupti.h>

namespace habitat {
namespace cuda {

class CuptiError : public std::runtime_error {
 public:
  static CuptiError from(CUptiResult error_code);

  CUptiResult errorCode() const {
    return error_code_;
  }

 private:
  CuptiError(CUptiResult error_code, const char* error_message);
  CUptiResult error_code_;
};

}
}
