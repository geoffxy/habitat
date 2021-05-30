#pragma once

namespace habitat {
namespace utils {

/**
 * Hashes and combines several values together.
 *
 * Usage:
 * std::size_t hash = habitat::utils::hash_combine("hello", "world", 1337);
 *
 */
template<typename... Values>
inline std::size_t hash_combine(Values... values);

}
}

#include "utils-inl.h"
