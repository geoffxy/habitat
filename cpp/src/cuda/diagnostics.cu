#define SIZE 10000

namespace {

// We need this global variable to ensure that nvcc does not optimize away the
// operations inside flop_test().
__device__ float accum = 0.;

__global__ void flop_test() {
  float a = 0.1;
#pragma unroll
  for (size_t i = 0; i < SIZE; i++) {
    accum += a;
  }
}

}

namespace habitat {
namespace cuda {
namespace diagnostics {

void run_flop_test(size_t num_blocks, size_t threads_per_block) {
  flop_test<<<num_blocks, threads_per_block>>>();
}

}
}
}
