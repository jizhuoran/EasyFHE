#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/builtin_functions.h>
#include <torch/csrc/jit/frontend/versioned_symbols.h>
#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch::jit {

void NormalizeOps(const std::shared_ptr<Graph>&) {}

const std::unordered_map<Symbol, Symbol>& getOperatorAliasMap() {
  static const std::unordered_map<Symbol, Symbol> aliases;
  return aliases;
}

bool PeepholeOptimize(const std::shared_ptr<Graph>&, bool) {
  return false;
}

bool PeepholeOptimize(Block*, bool) {
  return false;
}

bool FuseAddMM(const std::shared_ptr<Graph>&) {
  return false;
}

bool ConstantPropagation(std::shared_ptr<Graph>&, bool) {
  return false;
}

bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>&) {
  return false;
}

std::optional<Stack> runNodeIfInputsAreConstant(const Node*, bool, AliasDb*) {
  return std::nullopt;
}

void EliminateDeadCode(const std::shared_ptr<Graph>&, DCESideEffectPolicy) {}

void EliminateDeadCode(Block*, bool, DCESideEffectPolicy) {}

void EliminateDeadCode(
    Block*,
    std::function<void(const std::unordered_set<const Value*>&)>,
    DCESideEffectPolicy) {}

void FixupTraceScopeBlocks(std::shared_ptr<Graph>&, Module*) {}

void InsertBailOuts(std::shared_ptr<Graph>) {}

std::shared_ptr<Graph> BuildBailOutGraphFrom(
    int64_t,
    const std::shared_ptr<Graph>& orig,
    const std::shared_ptr<Graph>&) {
  return orig;
}

void ConstantPooling(const std::shared_ptr<Graph>&) {}

void Autocast(const std::shared_ptr<Graph>&) {}

bool setAutocastMode(bool value) {
  return value;
}

bool autocastEnabled() {
  return false;
}

void LowerSimpleTuples(const std::shared_ptr<Graph>&) {}

void LowerAllTuples(const std::shared_ptr<Graph>&) {}

void LowerSimpleTuples(Block*) {}

bool ShapeSymbolTable::bindSymbolicShapes(
    at::IntArrayRef,
    const c10::SymbolicShape&) {
  return true;
}

int64_t registerFusion(const Node*) {
  return 0;
}

void runFusion(const int64_t, Stack&) {
  TORCH_CHECK(false, "JIT fusion is disabled in EasyFHE fast build");
}

bool canFuseOnCPU() {
  return false;
}

bool canFuseOnGPU() {
  return false;
}

void overrideCanFuseOnCPU(bool) {}

void overrideMustUseLLVMOnCPU(bool) {}

void overrideCanFuseOnGPU(bool) {}

std::vector<at::Tensor> debugLaunchGraph(Graph&, at::ArrayRef<at::Tensor>) {
  return {};
}

std::string debugGetFusedKernelCode(Graph&, at::ArrayRef<at::Tensor>) {
  return {};
}

size_t nCompiledKernels() {
  return 0;
}

uint64_t get_min_version_for_kind(const NodeKind&) {
  return 0;
}

namespace mobile {
void registerPrimOpsFunction(const std::string&, const std::function<void(Stack&)>&) {}
} // namespace mobile

std::vector<char> pickle_save(const IValue&) {
  return {};
}

std::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>&,
    size_t) {
  return std::nullopt;
}

bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>&,
    size_t) {
  return true;
}

bool isOpSymbolCurrent(const std::string&, size_t) {
  return true;
}

std::vector<std::string> loadPossibleHistoricOps(
    const std::string&,
    std::optional<size_t>) {
  return {};
}

uint64_t getMaxOperatorVersion() {
  return 0;
}

std::vector<UpgraderRange> getUpgradersRangeForOp(const std::string&) {
  return {};
}

void calculate_package_version_based_on_upgraders(bool) {}

bool get_version_calculator_flag() {
  return false;
}

const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map() {
  static const std::unordered_map<std::string, std::vector<UpgraderEntry>> versions;
  return versions;
}

void test_only_add_entry(const std::string&, UpgraderEntry) {}

void test_only_remove_entry(const std::string&) {}

void test_only_reset_flag() {}

const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol) {
  static const std::vector<Function*> functions;
  return functions;
}

c10::QualifiedName TypeNameUniquer::getUniqueName(c10::ConstNamedTypePtr t) {
  return t->name().value_or(c10::QualifiedName("__easyfhe_unnamed_type"));
}

} // namespace torch::jit
