#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void UnpackLinear(std::shared_ptr<torch::jit::Graph>& graph) {
  //TensorRT implicitly adds a flatten layer infront of FC layers if necessary
  std::string linear_pattern = R"IR(
    graph(%x, %w, %b):
      %out: Tensor = aten::linear(%x, %w, %b)
      return (%out))IR";
  std::string mm_add_pattern = R"IR(
    graph(%x, %w, %b):
      %mm: Tensor = aten::matmul(%x, %w)
      %bias: Tensor = trt::const(%b)
      %1: int = prim::Constant[value=1]()
      %out: Tensor = aten::add_(%mm, %bias, %1)
      return (%out))IR";

  torch::jit::SubgraphRewriter unpack_linear;
  unpack_linear.RegisterRewritePattern(linear_pattern, mm_add_pattern);
  unpack_linear.runOnGraph(graph);
  LOG_GRAPH("Post unpack linear: " << *graph);
}


} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
