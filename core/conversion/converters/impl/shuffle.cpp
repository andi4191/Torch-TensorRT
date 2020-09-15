#include "core/conversion/converters/converters.h"

#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static auto shuffle_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({
    "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensorOrFreeze(ctx);
      auto start_dim = args[1].unwrapToInt();
      auto end_dim = args[2].unwrapToInt();
      auto in_shape = util::toVec(in->getDimensions());
      std::vector<int64_t> out_shape;
      if (ctx->input_is_dynamic) {
        out_shape = std::vector<int64_t>({in_shape[0], -1});
      } else {
        out_shape = torch::flatten(torch::rand(in_shape), start_dim, end_dim).sizes().vec();
      }

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(out_shape));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  }).pattern({
    "aten::reshape(Tensor self, int[] shape) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensorOrFreeze(ctx);
      auto in_shape = util::toVec(in->getDimensions());
      std::vector<int64_t> new_shape;
      if (ctx->input_is_dynamic) {
        TRTORCH_THROW_ERROR("Resize is currently not support in dynamic input shape compilation");
      } else {
        new_shape = torch::reshape(torch::rand(in_shape), args[1].unwrapToIntList().vec()).sizes().vec();
      }

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(new_shape));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

      return true;
    }
  }).pattern({
    "aten::view(Tensor(a) self, int[] size) -> (Tensor(a))",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensorOrFreeze(ctx);
      auto in_shape = util::toVec(in->getDimensions());

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(args[1].unwrapToIntList().vec()));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

      return true;
    }
  }).pattern({
    "aten::permute(Tensor(a) self, int[] dims) -> (Tensor(a))",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensorOrFreeze(ctx);
      auto in_shape = util::toVec(in->getDimensions());
      auto new_order = args[1].unwrapToIntList().vec();

      LOG_DEBUG("Shuffle to: " << util::toDims(new_order));

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      nvinfer1::Permutation permute;
      std::copy(new_order.begin(), new_order.end(), permute.order);
      shuffle->setSecondTranspose(permute);
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

      return true;
    }
  }).pattern({
    "aten::t(Tensor self) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
        LOG_DEBUG("entry function");
      if (args[0].isIValue()) {
        LOG_DEBUG("if part");
	//auto t = args[0].ITensorOrFreeze(ctx).t();
	//auto t = args[0].IValue()->toTensor().t();
        auto t = args[0].unwrapToTensor().t();
	auto t_weights = Weights(ctx, t);
	auto layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
        TRTORCH_CHECK(layer, "Unable to create constant layer from node: " << *n);

	layer->setName(util::node_info(n).c_str());

	auto in = args[0].unwrapToTensor();
	auto out = in.t();
	auto in_wt = Weights(ctx, in);
	auto out_wt = Weights(ctx, out);

	LOG_DEBUG("Check input shape: " << in_wt.shape << " output shape: " << out_wt.shape); 
        auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
        LOG_DEBUG("If Checking Output tensor shape: " << out_tensor->getDimensions());
      }
      else {
        LOG_DEBUG("Else part");
        auto t = args[0].ITensor();
	auto in_shape = util::toVec(t->getDimensions());

	auto layer = ctx->net->addShuffle(*t);
        TRTORCH_CHECK(layer, "Unable to create shuffle layer from node: " << *n);

	nvinfer1::Permutation permute;
	std::copy(in_shape.rbegin(), in_shape.rend(), permute.order);
        LOG_DEBUG("shuffle option");
	layer->setSecondTranspose(permute);

	layer->setName(util::node_info(n).c_str());
        
	auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
        LOG_DEBUG("Checking Output tensor shape: " << out_tensor->getDimensions());
      }
      return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
