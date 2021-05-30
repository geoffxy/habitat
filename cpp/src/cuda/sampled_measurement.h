#pragma once

#include <stdexcept>
#include <algorithm>

namespace habitat {
namespace cuda {

template <typename T>
struct SampledMeasurement {
  SampledMeasurement(T median, T min, T max) : median(median), min(min), max(max) {
    if (min > max || max < median || min > median) {
      throw std::runtime_error("Invalid values passed into SampledMeasurement.");
    }
  }
  explicit SampledMeasurement(T value) : SampledMeasurement(value, value, value) {}

  template <typename V, typename Mapper>
  static SampledMeasurement<T> fromValues(std::vector<V>& values, Mapper mapper) {
    std::vector<T> mapped_values;
    mapped_values.resize(values.size());
    std::transform(values.begin(), values.end(), mapped_values.begin(), mapper);
    return SampledMeasurement<T>::fromValues(mapped_values);
  }

  static SampledMeasurement<T> fromValues(std::vector<T>& values) {
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;

    if (values.size() % 2 == 0) {
      T mid1 = values.at(mid);
      T mid2 = values.at(mid - 1);
      return SampledMeasurement<T>((mid1 + mid2) / 2, values.front(), values.back());

    } else {
      return SampledMeasurement<T>(values.at(mid), values.front(), values.back());
    }
  }

  T median;
  T min;
  T max;
};

}
}
