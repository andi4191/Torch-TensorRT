#include "torch/csrc/jit/api/module.h"

#include <iostream>
#include "core/compiler.h"
#include "core/util/prelude.h"
#include "cuda.h"
#include "torch_tensorrt/torch_tensorrt.h"
namespace torch_tensorrt {
// Defined in types.cpp
torch_tensorrt::core::runtime::CudaDevice to_internal_cuda_device(Device device);
namespace torchscript {
// Defined in compile_spec.cpp
torch_tensorrt::core::CompileSpec to_internal_compile_spec(CompileSpec external);

bool check_method_operator_support(const torch::jit::script::Module& module, std::string method_name) {
  return torch_tensorrt::core::CheckMethodOperatorSupport(module, method_name);
}

std::string convert_method_to_trt_engine(
    const torch::jit::script::Module& module,
    std::string method_name,
    CompileSpec info) {
  LOG_DEBUG(get_build_info());
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return torch_tensorrt::core::ConvertGraphToTRTEngine(module, method_name, to_internal_compile_spec(info));
}

torch::jit::script::Module compile(const torch::jit::script::Module& module, CompileSpec info) {
  torch_tensorrt::core::util::initCuda();
  CUmoduleLoadingMode mode = CU_MODULE_LAZY_LOADING;
  auto status = cuModuleGetLoadingMode(&mode);
  if (status != CUDA_SUCCESS) {
    std::cout << "Error using API  cuModuleGetLoadingMode. Return status:  " << status << std::endl;
  }
  std::cout << "[compile]: Check CUDA loading mode: " << mode << std::endl;
  LOG_DEBUG(get_build_info());
  assert(1 == 0);
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return torch_tensorrt::core::CompileGraph(module, to_internal_compile_spec(info));
}

torch::jit::Module embed_engine_in_new_module(const std::string& engine, Device device) {
  return torch_tensorrt::core::EmbedEngineInNewModule(engine, to_internal_cuda_device(device));
}

} // namespace torchscript

std::string get_build_info() {
  auto info = torch_tensorrt::core::util::get_build_info();
  return std::string("Torch-TensorRT Version: ") + TORCH_TENSORRT_VERSION + '\n' + info;
}

void dump_build_info() {
  std::cout << get_build_info() << std::endl;
}

void set_device(const int gpu_id) {
  // Want to export a much simpler (non CUDA header dependent) API
  torch_tensorrt::core::set_device(gpu_id);
}

static auto tensorrt_input_container = torch::class_<Input>("_torch_tensorrt", "Input").def(torch::init<>());
} // namespace torch_tensorrt
