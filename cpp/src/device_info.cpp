#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <gflags/gflags.h>

DEFINE_int32(device, 0, "The ID of the device for which information should be extracted.");

int main(int argc, char* argv[]) {
  std::string usage("Utility that extracts usage information about the GPU(s) on this machine.\nUsage: ");
  usage += argv[0];
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1.0");
  gflags::ParseCommandLineFlags(&argc, &argv, /* remove_flags */ true);

  cudaDeviceProp props;
  memset(&props, 0, sizeof(cudaDeviceProp));
  cudaGetDeviceProperties(&props, FLAGS_device);

  std::cout << "compute_major: " << props.major << std::endl;
  std::cout << "compute_minor: " << props.minor << std::endl;
  std::cout << "max_threads_per_block: " << props.maxThreadsPerBlock << std::endl;
  std::cout << "max_threads_per_multiprocessor: " << props.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "regs_per_block: " << props.regsPerBlock << std::endl;
  std::cout << "regs_per_multiprocessor: " << props.regsPerMultiprocessor << std::endl;
  std::cout << "warp_size: " << props.warpSize << std::endl;
  std::cout << "shared_mem_per_block: " << props.sharedMemPerBlock << std::endl;
  std::cout << "shared_mem_per_multiprocessor: " << props.sharedMemPerMultiprocessor << std::endl;
  std::cout << "num_sms: " << props.multiProcessorCount << std::endl;
  std::cout << "shared_mem_per_block_optin: " << props.sharedMemPerBlockOptin << std::endl;

  return 0;
}
