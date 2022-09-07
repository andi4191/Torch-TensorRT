#include "cuda.h"
#include "cuda_runtime.h"

#include <iostream>
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"
namespace torch_tensorrt {
namespace core {
namespace runtime {

DeviceList::DeviceList() {
  util::initCuda();
  CUmoduleLoadingMode mode{CU_MODULE_LAZY_LOADING};
  CUresult res = cuModuleGetLoadingMode(&mode);
  assert(res == CUDA_SUCCESS);
  std::cout << "[DeviceList]: Check CUDA loading mode: " << mode << std::endl;
  int num_devices = 0;
  auto status = cudaGetDeviceCount(&num_devices);
  if (status != cudaSuccess) {
    LOG_WARNING("Unable to read CUDA capable devices. Return status: " << status);
  }

  for (int i = 0; i < num_devices; i++) {
    device_list[i] = CudaDevice(i, nvinfer1::DeviceType::kGPU);
  }

  // REVIEW: DO WE CARE ABOUT DLA?

  LOG_DEBUG("Runtime:\n Available CUDA Devices: \n" << this->dump_list());
}

void DeviceList::insert(int device_id, CudaDevice cuda_device) {
  device_list[device_id] = cuda_device;
}

CudaDevice DeviceList::find(int device_id) {
  return device_list[device_id];
}

DeviceList::DeviceMap DeviceList::get_devices() {
  return device_list;
}

std::string DeviceList::dump_list() {
  std::stringstream ss;
  for (auto it = device_list.begin(); it != device_list.end(); ++it) {
    ss << "    " << it->second << std::endl;
  }
  return ss.str();
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
