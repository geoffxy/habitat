#pragma once

namespace habitat {
namespace utils {

namespace detail {

// Combining hashes from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  hash_combine(seed, rest...);
}

}

template<typename... Values>
inline std::size_t hash_combine(Values... values) {
  std::size_t seed = 0;
  detail::hash_combine(seed, values...);
  return seed;
}

}
}
