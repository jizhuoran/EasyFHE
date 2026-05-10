#include <torch/nn/module.h>

namespace torch::nn {
namespace {
std::string join_name(const std::string& prefix, const std::string& name) {
  return prefix.empty() ? name : prefix + "." + name;
}
} // namespace

Module::Module()
    : parameters_("Parameter"), buffers_("Buffer"), children_("Submodule") {}

Module::Module(std::string name) : Module() {
  name_ = std::move(name);
}

const std::string& Module::name() const noexcept {
  if (!name_.has_value()) {
    name_ = "Module";
  }
  return *name_;
}

std::shared_ptr<Module> Module::clone(const std::optional<Device>&) const {
  TORCH_CHECK(false, "C++ nn::Module::clone is disabled in EasyFHE");
}

std::vector<Tensor> Module::parameters(bool recurse) const {
  return named_parameters(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_parameters(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  for (const auto& parameter : parameters_) {
    if (parameter.value().defined()) {
      result.insert(parameter.key(), parameter.value());
    }
  }
  if (recurse) {
    for (const auto& child : children_) {
      for (const auto& parameter : child.value()->named_parameters(true)) {
        result.insert(join_name(child.key(), parameter.key()), parameter.value());
      }
    }
  }
  return result;
}

std::vector<Tensor> Module::buffers(bool recurse) const {
  return named_buffers(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_buffers(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  for (const auto& buffer : buffers_) {
    if (buffer.value().defined()) {
      result.insert(buffer.key(), buffer.value());
    }
  }
  if (recurse) {
    for (const auto& child : children_) {
      for (const auto& buffer : child.value()->named_buffers(true)) {
        result.insert(join_name(child.key(), buffer.key()), buffer.value());
      }
    }
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool include_self) const {
  return named_modules(std::string(), include_self).values();
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_modules(
    const std::string& name_prefix,
    bool include_self) const {
  OrderedDict<std::string, std::shared_ptr<Module>> result;
  if (include_self) {
    result.insert(name_prefix, const_cast<Module*>(this)->shared_from_this_checked());
  }
  for (const auto& child : children_) {
    const auto child_name = join_name(name_prefix, child.key());
    result.insert(child_name, child.value());
    for (const auto& module : child.value()->named_modules(child_name, false)) {
      result.insert(module.key(), module.value());
    }
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::children() const {
  return named_children().values();
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_children()
    const {
  return children_;
}

void Module::train(bool on) {
  is_training_ = on;
  for (auto& child : children_) {
    child.value()->train(on);
  }
}

void Module::eval() {
  train(false);
}

bool Module::is_training() const noexcept {
  return is_training_;
}

void Module::to(torch::Device, torch::Dtype, bool) {}

void Module::to(torch::Dtype, bool) {}

void Module::to(torch::Device, bool) {}

void Module::zero_grad(bool) {}

void Module::save(serialize::OutputArchive&) const {}

void Module::load(serialize::InputArchive&) {}

void Module::pretty_print(std::ostream& stream) const {
  stream << name();
}

bool Module::is_serializable() const {
  return false;
}

void Module::clone_(Module&, const std::optional<Device>&) {
  TORCH_CHECK(false, "C++ nn::Module::clone_ is disabled in EasyFHE");
}

void Module::apply(const ModuleApplyFunction& function) {
  function(*this);
}

void Module::apply(const ConstModuleApplyFunction& function) const {
  function(*this);
}

void Module::apply(
    const NamedModuleApplyFunction& function,
    const std::string& name_prefix) {
  function(name_prefix, *this);
}

void Module::apply(
    const ConstNamedModuleApplyFunction& function,
    const std::string& name_prefix) const {
  function(name_prefix, *this);
}

void Module::apply(const ModulePointerApplyFunction& function) const {
  function(shared_from_this_checked());
}

void Module::apply(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  function(name_prefix, shared_from_this_checked());
}

std::shared_ptr<Module> Module::shared_from_this_checked() const {
  try {
    return const_cast<Module*>(this)->shared_from_this();
  } catch (const std::bad_weak_ptr&) {
    TORCH_CHECK(false, "Module is not owned by a shared_ptr");
  }
}

void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  for (const auto& child : children_) {
    const auto child_name = join_name(name_prefix, child.key());
    function(child_name, child.value());
  }
}

void Module::pretty_print_recursive(std::ostream&, const std::string&) const {}

void Module::unregister_module(const std::string& name) {
  children_.erase(name);
}

std::ostream& operator<<(std::ostream& stream, const nn::Module& module) {
  return stream << module.name();
}

serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const std::shared_ptr<nn::Module>&) {
  return archive;
}

serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    const std::shared_ptr<nn::Module>&) {
  return archive;
}

} // namespace torch::nn
