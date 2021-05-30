#pragma once

namespace habitat {
namespace cuda {
namespace diagnostics {

/**
 * Launches a single kernel that repeatedly performs 32-bit floating point adds.
 *
 * This diagnostic kernel is used to help us determine the peak performance
 * (GFLOP/s) of a device.
 */
void run_flop_test(size_t num_blocks = 8192, size_t threads_per_block = 256);

}
}
}
