#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto mm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({
    "aten::matmul(Tensor self, Tensor other) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto self = args[0].ITensorOrFreeze(ctx);
      LOG_DEBUG("self tensor shape: " << self->getDimensions());

      auto other = args[1].ITensorOrFreeze(ctx);
      LOG_DEBUG("other tensor shape: " << other->getDimensions());

      auto out = self;
      if ( self->getDimensions().nbDims > 2) {
        LOG_DEBUG("Hit this condition");
	auto dims = other->getDimensions();
	dims.d[0] = 1;
        auto reshape = ctx->net->addShuffle(*self);
	reshape->setReshapeDimensions(dims);
	std::string name = util::node_info(n) + "_[Reshape]";
	reshape->setName(name.c_str());
	out = reshape->getOutput(0);
      }
      auto check_dims = out->getDimensions();

      LOG_DEBUG("Output of the reshape layer" << check_dims.nbDims << " d [0]: " << check_dims.d[0] << " [1]: " << check_dims.d[1]);

      auto mm_layer = ctx->net->addMatrixMultiply(*out, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
      TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);
      mm_layer->setName(util::node_info(n).c_str());
      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
