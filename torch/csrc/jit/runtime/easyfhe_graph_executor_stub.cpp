#include <torch/csrc/jit/runtime/graph_executor.h>

#include <c10/util/Exception.h>

C10_DEFINE_bool(
    torch_jit_enable_new_executor,
    false,
    "EasyFHE fast build disables the TorchScript graph executor")

C10_DEFINE_bool(
    torch_jit_execution_plan_reuse_code_graph,
    false,
    "EasyFHE fast build disables the TorchScript graph executor")

C10_DEFINE_bool(
    torch_jit_disable_warning_prints,
    false,
    "EasyFHE fast build disables the TorchScript graph executor")

namespace torch::jit {
namespace {
std::atomic<bool> profiling_mode{false};
std::atomic<bool> executor_mode{false};
std::atomic<size_t> num_profiled_runs{0};
} // namespace

EnableProfilingGuard::EnableProfilingGuard() = default;
EnableProfilingGuard::~EnableProfilingGuard() = default;

GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>&,
    std::string) {}

GraphExecutor::GraphExecutor(
    const std::shared_ptr<Graph>&,
    std::string,
    ExecutorExecutionMode) {}

void GraphExecutor::run(Stack&) {
  TORCH_CHECK(false, "TorchScript graph execution is disabled in EasyFHE fast build");
}

c10::intrusive_ptr<Future> GraphExecutor::runAsync(Stack&, TaskLauncher) {
  TORCH_CHECK(false, "TorchScript graph execution is disabled in EasyFHE fast build");
}

const ExecutionPlan& GraphExecutor::getPlanFor(
    Stack&,
    std::optional<size_t>) {
  static ExecutionPlan plan;
  TORCH_CHECK(false, "TorchScript graph execution is disabled in EasyFHE fast build");
  return plan;
}

const ExecutionPlan& GraphExecutor::getInputIndependentPlan() {
  static ExecutionPlan plan;
  TORCH_CHECK(false, "TorchScript graph execution is disabled in EasyFHE fast build");
  return plan;
}

GraphExecutorState GraphExecutor::getDebugState() {
  return GraphExecutorState();
}

void GraphExecutor::debugFlushCompilationCache() {}

bool GraphExecutor::isOptimized() const {
  return false;
}

Node* replaceBlockWithFallbackGraph(Block*, ArrayRef<Value*>) {
  TORCH_CHECK(false, "TorchScript graph execution is disabled in EasyFHE fast build");
}

void runRequiredPasses(const std::shared_ptr<Graph>&) {}

void debugSetFusionGroupInlining(bool) {}

bool getFusionGroupInlining() {
  return false;
}

void debugSetAutodiffSubgraphInlining(bool) {}

std::shared_ptr<Graph> lastExecutedOptimizedGraph() {
  return nullptr;
}

std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}

std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

std::atomic<size_t>& getNumProfiledRuns() {
  return num_profiled_runs;
}

size_t getBailoutDepth() {
  return 0;
}

bool IsNewExecutorEnabled() {
  return false;
}

namespace detail {
GraphExecutor* getGradExecutor(Operation&) {
  return nullptr;
}

GraphExecutor* getDifferentiableGraphOpExecutor(Operation&) {
  return nullptr;
}
} // namespace detail
} // namespace torch::jit
