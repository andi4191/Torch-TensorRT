#include <algorithm>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {

const std::string empty_string = std::string();

std::string slugify(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

TRTEngine::TRTEngine(std::string serialized_engine)
    : logger(std::string("[] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {
    std::string _name = "deserialized_trt";
    new (this) TRTEngine(_name, serialized_engine, empty_string);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info)
    : logger(std::string("[] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {
    std::string _name = "deserialized_trt";
    std::string device_info = serialized_info[0];
    std::string engine_info = serialized_info[1];

    new (this) TRTEngine(_name, engine_info, device_info);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine, std::string serialized_device_info = empty_string)
    : logger(std::string("[") + mod_name + std::string("_engine] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {

    // Deserialize device meta data if device_info is non-empty
    if (!serialized_device_info.empty())
    {
        auto cuda_device = deserialize_device(serialized_device_info);
        // Set CUDA device as configured in serialized meta data
        set_cuda_device(cuda_device);
    }

    rt = nvinfer1::createInferRuntime(logger);

    name = slugify(mod_name) + "_engine";

    cuda_engine = rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size());
    // Easy way to get a unique name for each engine, maybe there is a more descriptive way (using something associated with the graph maybe)
    id = reinterpret_cast<EngineID>(cuda_engine);

    exec_ctx = cuda_engine->createExecutionContext();

    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
        std::string name = cuda_engine->getBindingName(x);
        std::string idx_s = name.substr(name.find("_") + 1);
        uint64_t idx = static_cast<uint64_t>(std::stoi(idx_s));

        if(cuda_engine->bindingIsInput(x)) {
            inputs++;
            in_binding_map[x] = idx;
        } else {
            outputs++;
            out_binding_map[x] = idx;
        }
    }
    num_io = std::make_pair(inputs, outputs);

}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
    id = other.id;
    rt = other.rt;
    cuda_engine = other.cuda_engine;
    device_info = other.device_info;
    exec_ctx = other.exec_ctx;
    num_io = other.num_io;
    return (*this);
}

TRTEngine::~TRTEngine() {
    exec_ctx->destroy();
    cuda_engine->destroy();
    rt->destroy();
}


// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }

static auto TRTORCH_UNUSED TRTEngineTSRegistrtion = torch::class_<TRTEngine>("tensorrt", "Engine")
    .def(torch::init<std::string>())
    // TODO: .def("__call__", &TRTEngine::Run)
    // TODO: .def("run", &TRTEngine::Run)
    .def_pickle(
        [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> {
	    // Serialize TensorRT engine
	    auto serialized_trt_engine = self->cuda_engine->serialize();

	    // Adding device info related meta data to the serialized file
	    auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

	    CudaDevice cuda_device;
	    get_cuda_device(cuda_device);
	    std::vector<std::string> serialize_info;
	    serialize_info.push_back(serialize_device(cuda_device));
	    serialize_info.push_back(trt_engine);
	    return serialize_info;
        },
         [](std::vector<std::string> seralized_info) -> c10::intrusive_ptr<TRTEngine> {
            return c10::make_intrusive<TRTEngine>(std::move(seralized_info));
        }
    );


int CudaDevice::get_id(void) {
    return this->id;
}

void CudaDevice::set_id(int id) {
    this->id = id;
}

int CudaDevice::get_major(void) {
    return this->major;
}

void CudaDevice::set_major(int major) {
    this->major = major;
}

int CudaDevice::get_minor(void) {
    return this->minor;
}

void CudaDevice::set_minor(int minor) {
    this->minor = minor;
}

void set_cuda_device(CudaDevice& cuda_device) {
    TRTORCH_CHECK((cudaSetDevice(cuda_device.id) == cudaSuccess), "Unable to set device: " << cuda_device.id);
}

void get_cuda_device(CudaDevice& cuda_device) {
    TRTORCH_CHECK((cudaGetDevice(&cuda_device.id) == cudaSuccess), "Unable to get current device: " << cuda_device.id);
    cudaDeviceProp device_prop;
    TRTORCH_CHECK((cudaGetDeviceProperties(&device_prop, cuda_device.id) == cudaSuccess), "Unable to get CUDA properties from device:" << cuda_device.id);
    cuda_device.set_major(device_prop.major);
    cuda_device.set_minor(device_prop.minor);
}

std::string serialize_device(CudaDevice& cuda_device) {
    void *buffer = new char[sizeof(cuda_device)];
    void *ref_buf = buffer;

    int temp = cuda_device.get_id();
    memcpy(buffer, reinterpret_cast<int*>(&temp), sizeof(int));
    buffer = static_cast<char*>(buffer) + sizeof(int);

    temp = cuda_device.get_major();
    memcpy(buffer, reinterpret_cast<int*>(&temp), sizeof(int));
    buffer = static_cast<char*>(buffer) + sizeof(int);

    temp = cuda_device.get_minor();
    memcpy(buffer, reinterpret_cast<int*>(&temp), sizeof(int));
    buffer = static_cast<char*>(buffer) + sizeof(int);

    return std::string((const char*)ref_buf, sizeof(int)*3);
}

CudaDevice deserialize_device(std::string device_info) {
    CudaDevice ret;
    char *buffer = new char[device_info.size() + 1];
    std::copy(device_info.begin(), device_info.end(), buffer);
    int temp = 0;

    memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int));
    buffer += sizeof(int);
    ret.set_id(temp);

    memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int));
    buffer += sizeof(int);
    ret.set_major(temp);

    memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int));
    buffer += sizeof(int);
    ret.set_minor(temp);

    return ret;
}


} // namespace execution
} // namespace core
} // namespace trtorch
