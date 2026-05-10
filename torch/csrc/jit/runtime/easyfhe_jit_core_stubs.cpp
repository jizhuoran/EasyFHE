#include <ATen/core/function_schema.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <atomic>
#include <string>

namespace torch::jit {
void registerOperator(Operator&&) {}

void deregisterOperator(const c10::FunctionSchema&) {}

std::vector<StackEntry> currentCallstack() {
  return {};
}

std::vector<std::string> currentModuleHierarchy() {
  return {};
}

AttributeValue::Ptr GraphAttr::clone() const {
  return Ptr(new GraphAttr(name, value_));
}

std::unique_ptr<AttributeValue> GraphsAttr::clone() const {
  auto copy = value_;
  return Ptr(new GraphsAttr(name, std::move(copy)));
}

Value* insertConstant(
    Graph&,
    const IValue&,
    std::optional<SourceRange>,
    std::optional<ScopePtr>) {
  TORCH_CHECK(false, "JIT constant insertion is disabled in EasyFHE fast build");
}

std::optional<Value*> tryInsertConstant(
    Graph&,
    const IValue&,
    std::optional<SourceRange>,
    std::optional<ScopePtr>) {
  return std::nullopt;
}

std::optional<IValue> toIValue(const Value*) {
  return std::nullopt;
}

bool isBlockListedSchema(const FunctionSchema&) {
  return false;
}

Value* emitBuiltinCall(
    const SourceRange&,
    Graph&,
    Symbol,
    at::ArrayRef<NamedValue>,
    at::ArrayRef<NamedValue>,
    const std::optional<NamedValue>&) {
  TORCH_CHECK(false, "JIT builtin calls are disabled in EasyFHE fast build");
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol) {
  static const std::vector<std::shared_ptr<Operator>> empty;
  return empty;
}

std::shared_ptr<Operator> getOperatorForLiteral(const char*) {
  TORCH_CHECK(false, "JIT operator lookup is disabled in EasyFHE fast build");
}

std::shared_ptr<Graph> GraphFunction::optimized_graph() const {
  return graph_;
}

namespace tracer {
thread_local ArgumentStash ArgumentStash::stash;

std::atomic<bool>& getTracerStateWarnMode() {
  static std::atomic<bool> warn_mode{true};
  return warn_mode;
}

const char* WARN_PYTHON_DATAFLOW =
    " might cause the trace to be incorrect. Python dataflow tracing is disabled in EasyFHE fast build.";
const char* WARN_CONSTRUCTOR =
    " results are registered as constants in the trace. JIT tracing is disabled in EasyFHE fast build.";
const char* WARN_RESIZE =
    " cannot be represented because JIT tracing is disabled in EasyFHE fast build.";
const char* STRICT_TRACER_MSG =
    "Only tensors or tuples of tensors can be output from traced functions.";

std::vector<StackEntry> pythonCallstack() {
  return {};
}

TracingState::TracingState() : graph(std::make_shared<Graph>()) {
  enterFrame();
}

TracingState::~TracingState() = default;

void TracingState::setValue(const IValue&, Value*) {}

void TracingState::delValue(const IValue&) {}

Value* TracingState::getValue(const IValue&) {
  return nullptr;
}

Value* TracingState::getOutput(const IValue&, size_t) {
  return nullptr;
}

bool TracingState::hasValue(const IValue&) const {
  return false;
}

Node* TracingState::createNode(c10::Symbol op_name, size_t num_outputs) {
  return graph ? graph->create(op_name, num_outputs) : nullptr;
}

void TracingState::insertNode(Node* node) {
  if (graph && node) {
    graph->insertNode(node);
  }
}

const std::shared_ptr<TracingState>& getTracingState() {
  static thread_local std::shared_ptr<TracingState> state;
  return state;
}

void setTracingState(std::shared_ptr<TracingState> state) {
  const_cast<std::shared_ptr<TracingState>&>(getTracingState()) =
      std::move(state);
}

void ArgumentStash::stashIntArrayRefElem(
    const std::string&,
    size_t,
    size_t,
    const Variable&) {}

void ArgumentStash::stashValue(
    const std::string&,
    size_t,
    const Variable&,
    const c10::TypePtr&) {}

void recordSourceLocation(Node*) {}

void setRecordSourceLocation(void (*)(Node*)) {}

void setPythonCallstack(std::vector<StackEntry> (*)()) {}

void setValueTrace(const IValue&, Value*) {}

void delValueTrace(const IValue&) {}

std::function<void()> pauseTracing() {
  return []() {};
}

Value* getValueTrace(const IValue&) {
  return nullptr;
}

void abandon() {}

void addInputs(Node*, const char*, const at::Tensor&) {}

void addInputs(Node*, const char*, at::MemoryFormat) {}

void addOutput(Node*, const at::Tensor&) {}

autograd::Variable getSizeOf(const autograd::Variable&, int64_t) {
  return at::Tensor();
}

autograd::Variable getNumelOf(const autograd::Variable&) {
  return at::Tensor();
}

void _do_warn(const char*, const char*) {}

void setWarn(warn_fn_type) {}
} // namespace tracer

} // namespace torch::jit
