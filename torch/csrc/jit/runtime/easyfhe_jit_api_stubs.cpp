#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>

#include <c10/util/Exception.h>

namespace torch::jit {

namespace {
ObjectPtr createEasyFHEDisabledModuleObject(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool = false) {
  if (class_name.prefix().empty()) {
    class_name = c10::QualifiedName("__torch__", class_name.name());
  }
  auto cls = ClassType::create(std::move(class_name), cu, /*is_module=*/true);
  cu->register_type(cls);
  return c10::ivalue::Object::create(
      c10::StrongTypePtr(std::move(cu), std::move(cls)), 0);
}

Module disabledJitSerializationModule() {
  TORCH_CHECK(false, "JIT serialization is disabled in EasyFHE fast build");
  return Module("__torch__.EasyFHEDisabled", std::make_shared<CompilationUnit>(), true);
}
} // namespace

bool& getInlineEverythingMode() {
  static bool inline_everything = false;
  return inline_everything;
}

named_attribute_list Module::named_attributes(bool recurse) const {
  return named_attribute_list(*this, recurse, false);
}

Object::Object(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

Object::Object(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle)
    : Object(createEasyFHEDisabledModuleObject(
          std::move(class_name),
          std::move(cu),
          shouldMangle)) {}

std::optional<Method> Object::find_method(const std::string&) const {
  return std::nullopt;
}

void Object::define(const std::string&, const ResolverPtr&) {
  TORCH_CHECK(false, "JIT object definition is disabled in EasyFHE fast build");
}

Object Object::copy() const {
  return Object(_ivalue()->copy());
}

Object Object::deepcopy() const {
  return Object(_ivalue()->deepcopy());
}

Module::Module(c10::QualifiedName class_name)
    : Object(createEasyFHEDisabledModuleObject(
          std::move(class_name),
          std::make_shared<CompilationUnit>())) {}

Module::Module(
    std::shared_ptr<CompilationUnit> cu,
    const c10::ClassTypePtr& type)
    : Object(c10::ivalue::Object::create(
          c10::StrongTypePtr(std::move(cu), type),
          type->numAttributes())) {}

Module::Module(
    c10::QualifiedName class_name,
    std::shared_ptr<CompilationUnit> cu,
    bool shouldMangle)
    : Object(createEasyFHEDisabledModuleObject(
          std::move(class_name),
          std::move(cu),
          shouldMangle)) {}

Method::Method(ObjectPtr owner, Function* function)
    : owner_(std::move(owner)), function_(function) {}

Module Method::owner() const {
  return Module(owner_);
}

ObjectPtr Method::raw_owner() const {
  return owner_;
}

void Method::run(Stack&) {
  TORCH_CHECK(false, "JIT Method execution is disabled in EasyFHE fast build");
}

c10::IValue Method::operator()(std::vector<c10::IValue>, const Kwargs&) const {
  TORCH_CHECK(false, "JIT Method execution is disabled in EasyFHE fast build");
}

c10::intrusive_ptr<c10::ivalue::Future> Method::run_async(
    std::vector<c10::IValue>,
    const Kwargs&,
    TaskLauncher) {
  TORCH_CHECK(false, "JIT Method execution is disabled in EasyFHE fast build");
}

void Method::setArgumentNames(std::vector<std::string>& argumentNames) const {
  argumentNames.clear();
}

Module load(std::istream&, std::optional<c10::Device>, bool) {
  return disabledJitSerializationModule();
}

Module load(
    std::istream&,
    std::optional<c10::Device>,
    ExtraFilesMap&,
    bool) {
  return disabledJitSerializationModule();
}

Module load(const std::string&, std::optional<c10::Device>, bool) {
  return disabledJitSerializationModule();
}

Module load(
    const std::string&,
    std::optional<c10::Device>,
    ExtraFilesMap&,
    bool) {
  return disabledJitSerializationModule();
}

Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface>,
    std::optional<c10::Device>,
    bool) {
  return disabledJitSerializationModule();
}

Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface>,
    std::optional<c10::Device>,
    ExtraFilesMap&,
    bool) {
  return disabledJitSerializationModule();
}

void ExportModule(
    const Module&,
    std::ostream&,
    const ExtraFilesMap&,
    bool,
    bool,
    bool) {
  TORCH_CHECK(false, "JIT serialization is disabled in EasyFHE fast build");
}

void ExportModule(
    const Module&,
    const std::string&,
    const ExtraFilesMap&,
    bool,
    bool,
    bool) {
  TORCH_CHECK(false, "JIT serialization is disabled in EasyFHE fast build");
}

void ExportModule(
    const Module&,
    const std::function<size_t(const void*, size_t)>&,
    const ExtraFilesMap&,
    bool,
    bool,
    bool) {
  TORCH_CHECK(false, "JIT serialization is disabled in EasyFHE fast build");
}

} // namespace torch::jit

namespace c10 {

torch::jit::Module IValue::toModule() const {
  return torch::jit::Module(toObject());
}

bool IValue::isModule() const {
  return isObject() && toObjectRef().type()->is_module();
}

} // namespace c10
