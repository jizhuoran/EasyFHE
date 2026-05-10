#include <ATen/core/ATen_fwd.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/functional.h>
#include <ATen/native/Copy.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/core/Contiguity.h>
#include <c10/core/GradMode.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <optional>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace at::meta {

static inline c10::MemoryFormat cat_compute_output_memory_format(
    const MaterializedITensorListRef& inputs) {
  std::optional<c10::MemoryFormat> format = std::nullopt;
  for (const Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

TORCH_PRECOMPUTE_META_FUNC(cat)(const ITensorListRef& tensors, int64_t dim) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  auto materialized = tensors.materialize();

  native::check_cat_no_zero_dim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Checking names before the actual dimensions.
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);

  TORCH_CHECK_VALUE(
      !materialized.empty(),
      "torch.cat(): expected a non-empty list of Tensors");

  // Look for the first valid tensor.
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  bool all_contiguous = true;
  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;
  auto memory_format = cat_compute_output_memory_format(materialized);

  // Compute what the output dtype should be:
  const auto& result = maybe_get_output();
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // If the output tensor is defined, we need to take it into account
  // when computing the actual output dtype and the flags.
  if (is_out_defined) {
    // Check for type promotion, if the output tensor is defined.
    TORCH_CHECK_TYPE(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    out_dtype = result.scalar_type();
    all_contiguous = result.is_contiguous(memory_format);
  }

  // Fallback 'set_output' parameters.
  // (in case we don't find a valid tensor)
  DimVector sizes{0};
  TensorOptions options =
      materialized[0].get().options().dtype(out_dtype).memory_format(
          memory_format);

  // If we found a valid tensor, check whether the input tensors
  // are compatible, i.e. we can execute `cat` on them.
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    TORCH_CHECK_INDEX(
        dim <= materialized[valid].get().dim(),
        "torch.cat(): dimension ",
        dim,
        "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    size_t size_at_dim = 0;
    for (const auto i : c10::irange(materialized.size())) {
      const Tensor& t = materialized[i];
      all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
      if (!at::native::cat_should_skip_tensor(t)) {
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        size_at_dim += t.size(dim);
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        all_contiguous = false;
      }
    }

    // Actually set the output.
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options =
        materialized[valid].get().options().dtype(out_dtype).memory_format(
            memory_format);
  }

  set_output_raw_strided(0, sizes, {}, options, maybe_outnames);
  // Checks for overlaps between the inputs and the output tensor.
  if (is_out_defined && found_valid_tensor) {
    at::assert_no_internal_overlap(result);
    for (const Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }

  return TORCH_PRECOMPUTE_STRUCT(cat)()
      .set_dim(dim)
      .set_valid(valid)
      .set_all_contiguous(all_contiguous)
      .set_all_same_dtype(all_same_dtype)
      .set_all_same_sizes_and_stride(all_same_sizes_and_stride)
      .set_memory_format(memory_format);
}
} // namespace at::meta

namespace at::native {
namespace sparse_csr {
static inline bool is_sparse_compressed(const Tensor&) { return false; }
static inline bool is_sparse_compressed(Layout) { return false; }
static inline Layout flip_compressed_layout(Layout l) { return l; }
}
namespace sparse {
struct SparseTensorImpl;
static inline SparseTensorImpl* get_sparse_impl(const Tensor&) {
  TORCH_CHECK(false, "Sparse tensors not supported in EasyFHE");
  return nullptr;
}
}


static inline Tensor _sparse_coo_tensor_with_dims_and_tensors(
    int64_t, int64_t, c10::IntArrayRef, const Tensor& indices,
    const Tensor& values, const TensorOptions&, std::optional<bool> = std::nullopt) {
  TORCH_CHECK(false, "Sparse COO tensors not supported in EasyFHE");
  return Tensor();
}
static inline Tensor _sparse_coo_tensor_with_dims_and_tensors(
    int64_t, int64_t, c10::SymIntArrayRef, const Tensor& indices,
    const Tensor& values, const TensorOptions&, std::optional<bool> = std::nullopt) {
  TORCH_CHECK(false, "Sparse COO tensors not supported in EasyFHE");
  return Tensor();
}


DEFINE_DISPATCH(cat_serial_stub);
DEFINE_DISPATCH(stack_serial_stub);

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor) {
  TORCH_CHECK(shape_tensor.dim() == 1);
  std::vector<int64_t> shape;
  auto accessor = shape_tensor.accessor<int64_t, 1>();
  for (const auto i : c10::irange(shape_tensor.numel())) {
    shape.push_back(accessor[i]);
  }
  return self.reshape(IntArrayRef(shape));
}

Tensor _shape_as_tensor(const Tensor& self) {
  auto options = TensorOptions(at::kLong);
  auto sizes = self.sizes();
  auto result = at::empty({static_cast<int64_t>(sizes.size())}, options);
  std::memcpy(result.mutable_data_ptr<int64_t>(), sizes.data(), sizes.size() * sizeof(int64_t));
  return result;
}

Tensor& set_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(std::move(source), 0, new_size, {});
}

// unify with cuda implementation?  This is not done to avoid a dispatch in
// resize_impl_cpu_
Tensor& set_storage_cpu_(
    Tensor& result,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
  // We can reuse this kernel for the meta device.
  // We just need to make sure we don't actually try to resize the (null)
  // storage.
  at::native::resize_impl_cpu_(
      result.unsafeGetTensorImpl(),
      size,
      stride_opt,
      /*resize_storage=*/!result.is_meta());
  return result;
}

Tensor& set_storage_meta__symint(
    Tensor& result,
    Storage storage,
    c10::SymInt storage_offset,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  checkSetStorage(
      result,
      storage,
      storage_offset,
      size,
      stride,
      /*check_offset_in_bounds=*/false);

  c10::SymDimVector contiguous_strides;
  if (stride.data() == nullptr) {
    // TODO: dedupe this with empty() symbolic logic
    int64_t dim = size.size();
    contiguous_strides.resize(dim);
    if (dim > 0) {
      const auto last_idx = dim - 1;
      contiguous_strides.at(last_idx) = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        // TODO: max with 1
        contiguous_strides.at(i) =
            contiguous_strides.at(i + 1) * size.at(i + 1);
      }
    }
    stride = contiguous_strides;
  }

  // Run this before storage setting so we can access numel
  result.unsafeGetTensorImpl()->set_sizes_and_strides(
      size, stride, storage_offset);

  // Matches maybe_resize_storage_cpu no-numel behavior
  if (TORCH_GUARD_OR_TRUE(result.sym_numel().sym_ne(0))) {
    // maybe_resize_storage_cpu can handle no storage exists at all but
    // that should never be the case here
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_CHECK(
        storage.resizable(), "Trying to resize storage that is not resizable");
    // All meta data pointers are the same, so we don't have to "re" allocate
    // it.  TODO: Actually this might not quite be correct if we use special
    // pointers to track whether or not fake cuda tensors are pinned or not

    // TODO: When there are unbacked SymInts, we unconditionally skip the
    // setter.  This is technically wrong, but we cannot conveniently test
    // the real condition in many cases, because a lot of people are using
    // set_ just to swizzle metadata on a tensor, they didn't actually want
    // to see if they need to resize the storage.
    //
    // The old behavior was to unconditionally set_nbytes, but I think not
    // setting it is more safe.
    if (result.sym_numel().has_hint()) {
      const auto itemsize = result.dtype().itemsize();

      c10::SymInt new_size_bytes = result.is_contiguous()
          ? at::detail::computeStorageNbytesContiguous(
                size, itemsize, std::move(storage_offset))
          : at::detail::computeStorageNbytes(
                size, stride, itemsize, std::move(storage_offset));

      if (new_size_bytes.has_hint() && storage.sym_nbytes().has_hint() &&
          (new_size_bytes > storage.sym_nbytes())) {
        storage.set_nbytes(std::move(new_size_bytes));
      }
    }
  }
  return result;
}

Tensor& set__symint(
    Tensor& result,
    const Tensor& storage,
    c10::SymInt storage_offset,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  TORCH_CHECK(
      storage.is_contiguous(),
      "passed in tensor to be used as storage must be contiguous");
  return result.set__symint(
      storage.storage(),
      storage_offset + storage.sym_storage_offset(),
      size,
      stride);
}

Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return result.set__symint(
        source.storage(),
        source.sym_storage_offset(),
        source.sym_sizes(),
        source.sym_strides());
  }
  return result;
}

// this needs to be split along CPU/CUDA lines because we don't have a
// consistent way of getting the allocator to use for a device
// (c10::GetAllocator is not the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(Storage::use_byte_size_t(), 0, c10::GetAllocator(kCPU), true);
  result.set_(std::move(storage), 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

// We can't reuse the cpu kernel here because we don't want to use the cpu
// allocator.
Tensor& set_meta_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(), 0, c10::GetAllocator(kMeta), true);
  result.set_(std::move(storage), 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor sparse_broadcast_to(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "Sparse tensors not supported in EasyFHE");
  return self;
}

Tensor broadcast_to_symint(const Tensor& self, SymIntArrayRef size) {
  return self.expand_symint(size);
}

std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  return expand_outplace(tensors);
}

static void fastCatOutDim0(
    const Tensor& out,
    const MaterializedITensorListRef& inputs) {
  auto outBytes = out.nbytes();
  char* dataPtr = reinterpret_cast<char*>(out.data_ptr());
  size_t totalBytes = 0;
  for (const Tensor& input : inputs) {
    TORCH_CHECK(outBytes >= totalBytes);
    if (input.nbytes() > 0) {
      std::memcpy(dataPtr + totalBytes, input.const_data_ptr(), input.nbytes());
    }
    totalBytes += input.nbytes();
  }
  TORCH_CHECK(outBytes == totalBytes);
}

TORCH_IMPL_FUNC(cat_out_cpu)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();

  bool use_serial_kernel =
      result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
  ScalarType dtype = materialized[valid].get().scalar_type();
  bool serial_dtype = at::isFloatingType(dtype);
  // fast path for single thread when both inputs and result are contiguous and
  // not empty, and concat dim is 0
  if (use_serial_kernel && all_contiguous && all_same_dtype &&
      (MemoryFormat::Contiguous == memory_format)) {
    if (dim == 0) {
      fastCatOutDim0(result, materialized);
      return;
    }
    // TODO: Add fast cat for higher dimensions and support multi-threaded fast
    // cat
  }

  // fast path for single thread when both inputs and result are contiguous and
  // not empty
  if (use_serial_kernel && all_contiguous && all_same_dtype && serial_dtype) {
    cat_serial_stub(kCPU, result, materialized, dim);
    return;
  }

  int64_t offset = 0;
  if (all_same_sizes_and_stride && result.is_contiguous(memory_format) &&
      all_same_dtype) {
    const Tensor& source_slice = materialized[valid];
    auto slice_dim_size = source_slice.sizes()[dim];
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .resize_outputs(false)
                    .add_output(result_slice)
                    .add_const_input(source_slice)
                    .enforce_safe_casting_to_output(true)
                    .build();

    for (const Tensor& tensor : materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto source_data = static_cast<const char*>(tensor.const_data_ptr());
      auto result_data =
          static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, const_cast<char*>(source_data));
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (const Tensor& tensor : materialized) {
      if (cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.sizes()[dim];
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = TensorIteratorConfig()
                      .set_check_mem_overlap(false) // Already checked above
                      .resize_outputs(false)
                      .add_output(result_slice)
                      .add_const_input(tensor)
                      .promote_inputs_to_common_dtype(true)
                      .cast_common_dtype_to_outputs(true)
                      .enforce_safe_casting_to_output(true)
                      .build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }
}

// torch.concat, alias for torch.cat
Tensor& concat_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concat(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

// torch.concatenate, alias for torch.cat
Tensor& concatenate_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

Tensor concatenate(TensorList tensors, int64_t dim) {
  return at::cat(tensors, dim);
}

static bool sizes_match_except(
    IntArrayRef s1,
    IntArrayRef s2,
    int64_t dim_except /* should already be wrapped */) {
  if (s1.size() != s2.size()) {
    return false;
  }
  for (const auto i : c10::irange(static_cast<int64_t>(s1.size()))) {
    if (i != dim_except && s1[i] != s2[i]) {
      return false;
    }
  }
  return true;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
static void check_cat_sparse_dims(
    Tensor const& t, int64_t pos, IntArrayRef sizes,
    int64_t wrapped, int64_t sparse_dim, int64_t dense_dim) {
  TORCH_CHECK(false, "Sparse cat not supported in EasyFHE");
}

static Tensor cat_sparse_impl(
    const MaterializedITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(false, "Sparse cat not supported in EasyFHE");
  return tensors[0].get();
}

Tensor cat_sparse(const ITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(false, "Sparse cat not supported in EasyFHE");
  return at::Tensor();
}


Tensor block_diag(TensorList tensors) {
  Tensor result;
  if (tensors.empty()) {
    result = at::empty({1, 0});
    return result;
  }

  const Device& device = tensors[0].device();
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];

    TORCH_CHECK(
        tensor.device() == device,
        "torch.block_diag: input tensors must all be on the same device.",
        " Input 0 is on device ",
        device,
        " and input ",
        tensor_idx,
        " is on device ",
        tensor.device());
  }

  ScalarType output_scalar_type = native::result_type(tensors);
  int64_t result_dim0 = 0;
  int64_t result_dim1 = 0;
  std::vector<Tensor> tensors_2D(tensors.size());

  // Sum the dimensions of the tensors, check tensor sizes,
  // and expand all 0-D and 1-D tensors so that everything
  // is 2-D
  for (const auto tensor_idx : c10::irange(tensors.size())) {
    const Tensor& tensor = tensors[tensor_idx];
    int64_t ndims = tensor.dim();
    TORCH_CHECK(
        ndims <= 2,
        "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
        tensor_idx,
        " has ",
        ndims,
        " dimensions");

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (ndims == 2) {
      dim0 = tensor.size(0);
      dim1 = tensor.size(1);
      tensors_2D[tensor_idx] = tensor;
    } else if (ndims == 1) {
      // Switching dim 0 to dim 1 is intentional
      dim1 = tensor.size(0);
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    } else {
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    }
    result_dim0 += dim0;
    result_dim1 += dim1;
  }

  result = at::zeros(
      {result_dim0, result_dim1},
      tensors[0].options().dtype(output_scalar_type));

  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // Copy each tensor into the appropriate location in the result matrix
  for (const auto& tensor : tensors_2D) {
    int64_t dim0 = tensor.size(0);
    int64_t dim1 = tensor.size(1);
    result.slice(0, cur_dim0, cur_dim0 + dim0)
        .slice(1, cur_dim1, cur_dim1 + dim1)
        .copy_(tensor);

    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }

  return result;
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(
      chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.sym_size(dim);
  auto split_size = (dim_size + chunks - 1) / chunks;

  // We need to call split_with_sizes in the case where split_size and dimension
  // size are 0, because a call to split would discard the number of chunks
  // (because we can have an arbitrary number of 0-sized chunks adding up to 0).
  // So, call split_with_sizes with the correct number of chunks, eventually we
  // will do this for all cases.
  if (split_size == 0 && dim_size == 0) {
    std::vector<c10::SymInt> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.split_with_sizes_symint(split_sizes, dim);
  } else {
    return self.split_symint(std::move(split_size), dim);
  }
}

std::vector<Tensor> tensor_split_sections_symint(
    const Tensor& self,
    c10::SymInt sym_sections,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  // NB: intentional, sections specifies number of output tensors, which
  // cannot be polymorphic
  int64_t sections = sym_sections.guard_int(__FILE__, __LINE__);
  TORCH_CHECK(
      sections > 0, "number of sections must be larger than 0, got ", sections);
  const auto dim_size = self.sym_size(dim_);
  std::vector<Tensor> splits(sections);
  auto min_split_size = dim_size / sections;
  auto num_splits_one_extra = dim_size % sections;
  c10::SymInt start_idx = 0;
  for (const auto split_idx : c10::irange(sections)) {
    auto split_size = (num_splits_one_extra > split_idx) ? (min_split_size + 1)
                                                         : min_split_size;
    splits[split_idx] =
        at::slice_symint(self, dim_, start_idx, start_idx + split_size);
    start_idx += split_size;
  }
  return splits;
}

template <typename T>
static std::vector<Tensor> _tensor_split_indices(
    const Tensor& self,
    ArrayRef<T> indices,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  int64_t num_indices = indices.size();
  std::vector<Tensor> splits(num_indices + 1);
  T start_idx(0);
  for (const auto split_idx : c10::irange(num_indices)) {
    auto end_idx = indices[split_idx];
    splits[split_idx] = at::symint::slice<T>(self, dim_, start_idx, end_idx);
    start_idx = end_idx;
  }
  splits[num_indices] = at::symint::slice<T>(
      self, dim_, start_idx, at::symint::size<T>(self, dim_));
  return splits;
}

std::vector<Tensor> tensor_split(
    const Tensor& self,
    IntArrayRef indices,
    int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split_indices_symint(
    const Tensor& self,
    SymIntArrayRef indices,
    int64_t dim) {
  return _tensor_split_indices(self, indices, dim);
}

std::vector<Tensor> tensor_split(
    const Tensor& self,
    const Tensor& tensor_indices_or_sections,
    int64_t dim) {
  TORCH_CHECK(
      self.dim() > 0,
      "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
      self.dim(),
      " dims");
  auto split_device = tensor_indices_or_sections.device();
  TORCH_CHECK(
      split_device == kCPU,
      "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on ",
      split_device);
  auto split_dtype = tensor_indices_or_sections.scalar_type();
  TORCH_CHECK(
      split_dtype == at::kLong,
      "tensor_split expected tensor_indices_or_sections to have dtype of long, but got ",
      split_dtype);
  auto split_dim = tensor_indices_or_sections.dim();
  TORCH_CHECK(
      split_dim == 1 || split_dim == 0,
      "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with ",
      split_dim,
      " dims");

  if (split_dim == 0) {
    int64_t sections = tensor_indices_or_sections.item<int64_t>();
    return self.tensor_split(sections, dim);
  } else {
    auto indices_data = tensor_indices_or_sections.const_data_ptr<int64_t>();
    auto stride = tensor_indices_or_sections.stride(0);
    auto numel = tensor_indices_or_sections.numel();
    std::vector<int64_t> indices(numel);
    for (const auto offset : c10::irange(numel)) {
      // indices tensor could be non-contiguous
      indices[offset] = *(indices_data + offset * stride);
    }
    return self.tensor_split(indices, dim);
  }
}

std::vector<Tensor> unsafe_chunk(
    const Tensor& self,
    int64_t chunks,
    int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(
      chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);

  const auto dim_size = self.size(dim);
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  // See the comment above in chunk(...)
  if (split_size == 0 && dim_size == 0) {
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
    return self.unsafe_split_with_sizes(split_sizes, dim);
  } else {
    return self.unsafe_split(split_size, dim);
  }
}

Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}

Tensor diagonal(
    const Tensor& self,
    int64_t offset,
    int64_t dim1_,
    int64_t dim2_) {
  int64_t nDims = self.dim();
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(
      dim1 != dim2,
      "diagonal dimensions cannot be identical ",
      dim1_,
      ", ",
      dim2_);
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  NoNamesGuard no_names_guard;

  int64_t diag_size = 0;
  int64_t storage_offset = self.storage_offset();
  // compute storage offset and size for the diagonal
  // for positive values of offset (above the main diagonal)
  // "leftmost columns" (along dim2) are dropped
  // for negative values of offset (below the main diagonal)
  // "topmost rows" (along dim1) are dropped.
  // Note that we invert +/- in the second to absorb the negative
  // sign in the offset.
  if (offset >= 0) {
    diag_size = std::max<int64_t>(
        std::min(self.size(dim1), self.size(dim2) - offset), 0);
  } else {
    diag_size = std::max<int64_t>(
        std::min(self.size(dim1) + offset, self.size(dim2)), 0);
  }

  // NumPy allows you to specify offsets "off the end"; let's just be careful
  // not to set a ridiculous storage_offset in that case (technically it
  // shouldn't matter because there are no elements in the tensor, but let's be
  // kosher).
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * self.stride(dim2);
  } else {
    storage_offset -= offset * self.stride(dim1);
  }

  // construct new size and stride: we drop dim1 and dim2 (maximum first for not
  // changing the index of the minimum) the new ("joint") dimension is appended
  // to the end of the shape / stride to match numpy semantics
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1) + self.stride(dim2));

  // return view with new parameters
  auto result = self.as_strided(sizes, strides, storage_offset);

  no_names_guard.reset();
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}

Tensor diag_embed(
    const Tensor& self,
    int64_t offset,
    int64_t dim1_,
    int64_t dim2_) {
  int64_t nDims = self.dim() + 1;
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(
      dim1 != dim2,
      "diagonal dimensions cannot be identical ",
      dim1_,
      ", ",
      dim2_);
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  auto result = at::zeros(sizes, self.options());
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  return result;
}

Tensor expand(const Tensor& self, c10::IntArrayRef size, bool /*unused*/) {
  TORCH_CHECK(
      size.size() >= (size_t)self.dim(),
      "expand(",
      self.toString(),
      "{",
      self.sizes(),
      "}, size=",
      size,
      "): the number of sizes provided (",
      size.size(),
      ") ",
      "must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ")");
  TORCH_CHECK(
      !self.is_sparse(),
      "expand is unsupported for ",
      self.layout(),
      " tensors");

  auto expandedSizesAndStrides =
      inferExpandGeometry_dimvector(self.sizes(), self.strides(), size);

  auto result = self.as_strided(
      expandedSizesAndStrides.sizes, expandedSizesAndStrides.strides);
  namedinference::propagate_names_for_expand(result, self);
  return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand_symint(other.sym_sizes());
}

Tensor sum_to_size_symint(const Tensor& self, SymIntArrayRef size) {
  TORCH_CHECK(
      is_expandable_to(size, self.sym_sizes()),
      "size {",
      size,
      "} is not expandable to size {",
      self.sizes(),
      "}.");

  return sum_to(self, size);
}

// We currently do not support per-channel quant for unfold, diagonal, expand,
// permute.
// TODO: Make this an aten function and replace as_strided_qtensorimpl once that
// is done.
static Tensor make_qtensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    QuantizerPtr quantizer) {
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  setStrided(result, size, stride, self.storage_offset());
  return result;
}

Tensor as_strided_tensorimpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

template <typename T>
static void setStridedUnchecked(
    const Tensor& self,
    ArrayRef<T> size,
    ArrayRef<T> stride,
    T&& storage_offset) {
  checkAsStridedArgsAllowUnbackedSymInts(size, stride, storage_offset);
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_sizes_and_strides(size, stride, std::forward<T>(storage_offset));
}

Tensor as_strided_tensorimpl_meta_symint(
    const Tensor& self,
    SymIntArrayRef sym_size,
    SymIntArrayRef sym_stride,
    std::optional<c10::SymInt> sym_storage_offset_) {
  auto sym_storage_offset =
      sym_storage_offset_.value_or(self.sym_storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype());
  // NB: The reason this is unchecked is to ensure we don't generate
  // guards on the base storage itself when performing as_strided calls.
  // Although technically these guards are necessary, in practice they
  // cause a lot of guards that falsely refer to base symbols.  We will instead
  // rely on AOTAutograd to sort out if we actually have dependence on view
  // bases / storage size.
  setStridedUnchecked(
      result, sym_size, sym_stride, std::move(sym_storage_offset));
  return result;
}

Tensor as_strided_qtensorimpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE,
      "Setting strides is possible only on uniformly quantized tensor");
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  setStrided(result, size, stride, storage_offset);
  return result;
}

// This is an overloaded function similar to
// Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size,
// IntArrayRef stride, std::optional<int64_t> storage_offset_) and is currently
// not available through the dispatcher. The additional input, quantizer, is
// called by the select & slice methods.
// TODO: Make this function compatible with the dispatcher
static Tensor as_strided_qtensorimpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<int64_t> storage_offset_,
    QuantizerPtr quantizer) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  TORCH_CHECK(
      (quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE) ||
          (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE),
      "Setting strides is possible only on uniformly or per channel quantized tensors");
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  setStrided(result, size, stride, storage_offset);
  return result;
}

const Tensor& as_strided__symint(
    const Tensor& self,
    SymIntArrayRef size,
    SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.sym_storage_offset());
  setStrided(self, size, stride, std::move(storage_offset));
  return self;
}

// Should just use narrow_copy_out, but this API is used internally at Meta:
// https://github.com/pytorch/pytorch/pull/87045#issuecomment-1309353561
Tensor narrow_copy_dense_cpu(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  // narrow_copy_dense_cpu_out always resize output's size, so there only create
  // a zero size tensor.
  auto output = at::empty({0}, self.options());
  return narrow_copy_dense_cpu_out(self, dim, start, length, output);
}

Tensor narrow_copy_sparse(
    const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  TORCH_CHECK(false, "Sparse narrow_copy not supported in EasyFHE");
  return self;
}

Tensor& narrow_copy_dense_cpu_out(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length,
    Tensor& output) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(self.dtype() == output.dtype());

  auto self_contig = self.expect_contiguous();
  const auto self_sizes = self_contig->sizes();

  // wrap dim if negative and do bound check
  if (dim < 0) {
    dim = at::maybe_wrap_dim(dim, self_sizes.size());
  } else {
    TORCH_CHECK(dim < static_cast<int64_t>(self_sizes.size()));
  }

  // wrap start and do bound check
  const auto cur_size = self_sizes[dim];
  TORCH_CHECK_INDEX(
      -cur_size <= start && start <= cur_size,
      "start out of range (expected to be in range of [",
      -cur_size,
      ", ",
      cur_size,
      "], but got ",
      start,
      ")")
  if (start < 0) {
    start = start + cur_size;
  }
  TORCH_CHECK(
      length >= 0 && start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");

  // resize output
  auto output_sizes = self_sizes.vec();
  output_sizes[dim] = length;
  at::native::resize_(output, output_sizes);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int64_t unit = c10::size_from_dim_(dim + 1, self_sizes);
  const int64_t num_blocks = c10::size_to_dim_(dim, self_sizes);

  const auto itemsize = self_contig->dtype().itemsize();
  size_t src_nbytes = itemsize * self_contig->numel();
  size_t dst_nbytes = itemsize * output.numel();

  size_t src_block_size = unit * self_sizes[dim];
  size_t dst_block_size = unit * length;

  if (num_blocks == 0 || dst_block_size == 0) {
    return output;
  }

  const char* src_bytes =
      static_cast<const char*>(self_contig->const_data_ptr());
  char* dst_bytes = static_cast<char*>(output.data_ptr());

  size_t src_block_size_bytes = itemsize * src_block_size;
  size_t dst_block_size_bytes = itemsize * dst_block_size;
  size_t src_offset = unit * start;

  const char* src_offset_bytes = src_bytes + itemsize * src_offset;
  char* dst_offset_bytes = dst_bytes;

  for (const auto i : c10::irange(num_blocks)) {
    const char* local_src_offset_bytes =
        src_offset_bytes + i * src_block_size_bytes;
    char* local_dst_offset_bytes = dst_offset_bytes + i * dst_block_size_bytes;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<const void*>(
            local_src_offset_bytes + dst_block_size_bytes) <=
        static_cast<const void*>(src_bytes + src_nbytes));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        static_cast<void*>(local_dst_offset_bytes + dst_block_size_bytes) <=
        static_cast<void*>(dst_bytes + dst_nbytes));

    memcpy(
        local_dst_offset_bytes, local_src_offset_bytes, dst_block_size_bytes);
  }
  return output;
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
  auto cur_size = self.size(dim);
  TORCH_CHECK_INDEX(
      -cur_size <= start && start <= cur_size,
      "start out of range (expected to be in range of [",
      -cur_size,
      ", ",
      cur_size,
      "], but got ",
      start,
      ")")
  if (start < 0) {
    start = start + cur_size;
  }
  TORCH_CHECK(
      start <= cur_size - length,
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");
  return at::slice(self, dim, start, start + length, 1);
}

Tensor narrow_symint(
    const Tensor& self,
    int64_t dim,
    SymInt start,
    SymInt length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_SYM_CHECK(length.sym_ge(0), "narrow(): length must be non-negative.");
  auto cur_size = self.sym_size(dim);
  TORCH_CHECK_INDEX(
      ((-cur_size).sym_le(start).sym_and(start.sym_le(cur_size)))
          .expect_true(__FILE__, __LINE__),
      "start out of range (expected to be in range of [",
      -cur_size,
      ", ",
      cur_size,
      "], but got ",
      start,
      ")")

  auto cond1 = TORCH_GUARD_OR_FALSE(start.sym_lt(0));
  auto cond2 = TORCH_GUARD_OR_FALSE(start.sym_ge(0));

  if (cond1 || cond2) {
    if (cond1) {
      start = start + cur_size;
    }

    TORCH_SYM_CHECK(
        start.sym_le(cur_size - length),
        "start (",
        start,
        ") + length (",
        length,
        ") exceeds dimension size (",
        cur_size,
        ").");
    return at::slice_symint(self, dim, start, start + length, 1);
  }

  // Unbacked start handling!

  // Bounds check without converting start:
  // - If start < 0: need (start + cur_size) + length <= cur_size, i.e., start +
  // length <= 0
  // - If start >= 0: need start + length <= cur_size
  auto end = start + length;
  TORCH_SYM_CHECK(
      (start.sym_lt(0).sym_and((end).sym_le(0)))
          .sym_or(start.sym_ge(0).sym_and((end).sym_le(cur_size))),
      "start (",
      start,
      ") + length (",
      length,
      ") exceeds dimension size (",
      cur_size,
      ").");

  if (TORCH_GUARD_OR_FALSE(end.sym_ne(0))) {
    return at::slice_symint(self, dim, start, end, 1);
  } else {
    // Cannot statically determine the condition due to unbacked.
    // This is an interesting situation; when start is negative and
    // start + length == 0, slice and narrow do different things.
    // i.e., x.narrow(0, -2, 2) != x[-2:0]; in that case, we want to
    // pass curr_size instead of 0. Otherwise, they would do the same thing.
    // This says at runtime: if start < 0 and end == 0, then pass curr_size
    // instead of 0.

    auto use_different = start.sym_lt(0).sym_and(end.sym_eq(0)).toSymInt();
    auto result =
        at::slice_symint(self, dim, start, end + use_different * cur_size, 1);

    // Ensure slice allocated unbacked size is specialized to length.
    SymInt new_size = result.sym_size(dim);
    TORCH_SYM_CHECK(new_size.sym_eq(length), "")

    return result;
  }
}

// This overload exists purely for XLA, because they wanted to pass in
// "symbolic" start via Tensor.
Tensor narrow_tensor_symint(
    const Tensor& self,
    int64_t dim,
    const Tensor& start,
    SymInt length) {
  TORCH_CHECK(
      start.dim() == 0 &&
          isIntegralType(start.scalar_type(), /*includeBool=*/false),
      "start must be an 0-dim integral Tensor.");
  c10::SymInt st = start.item().toSymInt();
  return at::narrow_symint(self, dim, std::move(st), std::move(length));
}

std::
    tuple<DimVector, DimVector, std::vector<int64_t>> static _permute_size_stride_estimation(
        const Tensor& self,
        IntArrayRef dims) {
  const auto ndim = self.dim();
  TORCH_CHECK(
      ndim == static_cast<int64_t>(dims.size()),
      "permute(sparse_coo): number of dimensions in the tensor input ",
      "does not match the length of the desired ordering of dimensions ",
      "i.e. input.dim() = ",
      ndim,
      " is not equal to len(dims) = ",
      dims.size());

  const auto is_strided_layout = self.options().layout() == at::kStrided;
  const auto old_sizes = self.sizes();
  const auto old_strides = is_strided_layout ? self.strides() : IntArrayRef{};

  auto new_sizes = DimVector(ndim);
  auto new_strides = DimVector(is_strided_layout ? ndim : 0);
  auto wrapped_dims = std::vector<int64_t>(ndim);
  std::vector<bool> seen_dims(ndim);

  for (const auto i : c10::irange(ndim)) {
    const auto d = maybe_wrap_dim(dims[i], ndim);
    TORCH_CHECK(!seen_dims[d], "permute(): duplicate dims are not allowed.");
    seen_dims[d] = true;
    wrapped_dims[i] = d;
    new_sizes[i] = old_sizes[d];
    if (is_strided_layout) {
      new_strides[i] = old_strides[d];
    }
  }

  return std::make_tuple(new_sizes, new_strides, wrapped_dims);
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
  auto [new_sizes, new_strides, _] =
      _permute_size_stride_estimation(self, dims);
  return self.as_strided(new_sizes, new_strides);
}

Tensor permute_sparse_coo(const Tensor& self, IntArrayRef dims) {
  TORCH_CHECK(false, "Sparse COO permute not supported in EasyFHE");
  return self;
}

Tensor repeat(const Tensor& self, IntArrayRef repeats) {
  TORCH_CHECK(
      repeats.size() >= (size_t)self.dim(),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor xtensor = self.expand(padded_size);

  Tensor urtensor;
  if (self.is_quantized()) {
    TORCH_CHECK(false, "Quantized ops removed in EasyFHE");
  } else {
    urtensor = at::empty(target_size, self.options());
  }

  // return an empty tensor if one of the repeat dimensions is zero
  if (zero_tensor) {
    return urtensor;
  }

  for (const auto i : c10::irange(xtensor.dim())) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    auto size_i = xtensor.sizes()[i];
    auto step_i = std::max<int64_t>(size_i, 1);
    auto sizes = urtensor.sizes().vec();
    auto strides = urtensor.strides().vec();
    sizes.push_back(size_i);
    strides.push_back(strides[i]);
    sizes[i] = (sizes[i] - size_i) / step_i + 1;
    strides[i] *= step_i;
    urtensor = urtensor.as_strided(sizes, strides);
  }

  urtensor.copy_(xtensor.expand_as(urtensor));

  // Combine the dimensions to produce the target_size.
  // xtensor dims: [a0, ..., ad-1]
  // urtensor dims: [a0, ..., ad-1, b0, ..., bd-1]
  // b dims are produced by unfold.
  // Transform urtensor to [a0 * b0, ..., ad-1 * bd-1]
  const int64_t n_dims = xtensor.dim();
  auto range_a = at::arange(xtensor.dim(), at::TensorOptions(at::kLong));
  auto range_b = range_a + n_dims;
  auto stacked = stack({std::move(range_a), std::move(range_b)}, 1).flatten();
  auto permutation = IntArrayRef(stacked.const_data_ptr<int64_t>(), n_dims * 2);
  // Permute from [a0, ..., ad-1, b0, ..., bd-1] to [a0, b0, ..., ad-1, bd-1]
  urtensor = urtensor.permute(permutation);
  // Reshape from [a0, b0, ..., ad-1, bd-1] to [a0 * b0, ..., ad-1 * bd-1]
  urtensor = urtensor.reshape(target_size);

  return urtensor;
}

Tensor tile_symint(const Tensor& self, SymIntArrayRef reps) {
  // If self.size() > len(reps), reps is promoted to self.size() by prepending
  // 1’s to it to keep the same behaviour as `numpy.tile`.
  // Thus for a tensor of shape (2, 3, 4, 5), a dims of (2, 2) is treated
  // as (1, 1, 2, 2).
  const int64_t size_diff = self.dim() - static_cast<int64_t>(reps.size());
  if (size_diff > 0) {
    std::vector<c10::SymInt> new_reps(size_diff, 1);
    for (const auto i : c10::irange(reps.size())) {
      new_reps.emplace_back(reps[i]);
    }
    return self.repeat_symint(SymIntArrayRef(new_reps));
  }
  // `torch.tile` is equivalent to the already implemented `torch.Tensor.repeat`
  return self.repeat_symint(reps);
}

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
static Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const Vec& sizes,
    const Vec& strides) {
  // caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  Tensor self_;
  if (self.is_quantized()) {
    self_ = at::detail::make_tensor<QTensorImpl>(
        c10::TensorImpl::VIEW,
        Storage(self.storage()),
        self.key_set(),
        self.dtype(),
        get_qtensorimpl(self)->quantizer());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  } else {
    self_ = at::detail::make_tensor<TensorImpl>(
        c10::TensorImpl::VIEW,
        Storage(self.storage()),
        self.key_set(),
        self.dtype());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  }
  namedinference::propagate_names(self_, self);
  return self_;
}

// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and
// SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
static Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides) {
  // caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  Tensor self_;
  if (self.is_quantized()) {
    self_ = at::detail::make_tensor<QTensorImpl>(
        c10::TensorImpl::VIEW,
        Storage(self.storage()),
        self.key_set(),
        self.dtype(),
        get_qtensorimpl(self)->quantizer());
    self_.unsafeGetTensorImpl()->set_sizes_and_strides(
        sizes, strides, self.sym_storage_offset());
  } else {
    self_ = at::detail::make_tensor<TensorImpl>(
        c10::TensorImpl::VIEW,
        Storage(self.storage()),
        self.key_set(),
        self.dtype());
    self_.unsafeGetTensorImpl()->set_sizes_and_strides(
        sizes, strides, self.sym_storage_offset());
  }
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor reshape_symint(const Tensor& self, c10::SymIntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    TORCH_CHECK(false, "reshape is not implemented for sparse tensors");
  }

  if (self.is_contiguous_or_false() && !self.is_mkldnn()) {
    return self.view_symint(proposed_shape);
  }

  auto sym_numel = self.sym_numel();
  c10::SymDimVector shape = infer_size_dv(proposed_shape, sym_numel);

  if (self.is_mkldnn()) {
    TORCH_CHECK(false, "mkldnn reshape not supported in EasyFHE");
  }
  auto sym_sizes = self.sym_sizes();
  auto sym_strides = self.sym_strides();

  // `computeStride` returns the proper strides to use if this
  // `reshape` can be just a view.
  auto stride = at::detail::computeStride(sym_sizes, sym_strides, shape);

  // NB: Even though we have viewable geometry and the target strides here,
  //     we do not just call `as_strided` on `self` because the backward
  //     for `as_strided` is not as efficient as that of `view` (since the
  //     former is meant to handle general cases).
  //
  //     Similarly we don't call `view` because it duplicates some of the work
  //     we've already done, and instead call our internal/private operator
  //     `_reshape_alias` that essentially does the same thing as `view` and
  //     `as_strided` without any of the extra overhead.
  if (stride.has_value()) {
    // Temporary check to revert to the old behavior/view in cases where the
    // device is not supported (e.g. for XLA the operation is not supported
    // so we use `view` instead).
    //
    // We need to do the checks here instead of in `native_functions.yaml`
    // to preserve backwards compatibility.
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu() &&
        !at::isTensorSubclassLike(self)) {
      return self._reshape_alias_symint(shape, stride.value());
    } else {
      return self.view_symint(shape);
    }
  }
  return at::_unsafe_view_symint(
      self.clone(at::MemoryFormat::Contiguous), shape);
}

Tensor _reshape_copy_symint(
    const Tensor& self,
    c10::SymIntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    TORCH_CHECK(0, "_reshape_copy is not implemented for sparse tensors");
  }
  c10::SymDimVector shape = infer_size_dv(proposed_shape, self.sym_numel());

  if (self.is_mkldnn()) {
    TORCH_CHECK(0, "_reshape_copy not implemented for mkldnn tensors");
  }

  if (self.is_contiguous_or_false()) {
    return self.view_symint(shape).clone(at::MemoryFormat::Contiguous);
  } else {
    return at::_unsafe_view_symint(
        self.clone(at::MemoryFormat::Contiguous), shape);
  }
}

// Duplicate of above code for non-symbolic ints. Kept for BC purposes and to
// minimize breakages.
Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    TORCH_CHECK(false, "reshape is not implemented for sparse tensors");
  }
  DimVector shape = infer_size_dv(proposed_shape, self.numel());

  if (self.is_mkldnn()) {
    TORCH_CHECK(false, "mkldnn reshape not supported in EasyFHE");
  }

  // `computeStride` returns the proper strides to use if this
  // `reshape` can be just a view.
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);

  // NB: Even though we have viewable geometry and the target strides here,
  //     we do not just call `as_strided` on `self` because the backward
  //     for `as_strided` is not as efficient as that of `view` (since the
  //     former is meant to handle general cases).
  //
  //     Similarly we don't call `view` because it duplicates some of the work
  //     we've already done, and instead call our internal/private operator
  //     `_reshape_alias` that essentially does the same thing as `view` and
  //     `as_strided` without any of the extra overhead.
  if (stride.has_value()) {
    // Temporary check to revert to the old behavior/view in cases where the
    // device is not supported (e.g. for XLA the operation is not supported
    // so we use `view` instead).
    //
    // We need to do the checks here instead of in `native_functions.yaml`
    // to preserve backwards compatibility.
    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu()) {
      return self._reshape_alias(shape, stride.value());
    } else {
      return self.view(shape);
    }
  }
  return at::_unsafe_view(self.clone(at::MemoryFormat::Contiguous), shape);
}

Tensor _reshape_alias(
    const Tensor& self,
    IntArrayRef sizes,
    IntArrayRef strides) {
  // This is only used by `reshape` in cases where it would otherwise have
  // dispatched to `view`. This removes the overhead of calling `view` which
  // duplicates some of the work that's already been done (`infer_size_dv` and
  // `computeStride`).

  return alias_with_sizes_and_strides(self, sizes, strides);
}

Tensor reshape_as(const Tensor& self, const Tensor& other) {
  return self.reshape_symint(other.sym_sizes());
}

static Tensor select_sparse(const Tensor& self, int64_t dim, int64_t index) {
  TORCH_CHECK(false, "Sparse select not supported in EasyFHE");
  return self;
}

// this is an auxiliary function, called by the select&slice methods, that
// creates a new quantizer from the given input
// is_select is true if calling function is select()
static QuantizerPtr create_subtensor_quantizer(
    const Tensor& self,
    bool is_select,
    int64_t start,
    int64_t end,
    int64_t dim,
    int64_t step) {
  auto quantizer_prev = get_qtensorimpl(self)->quantizer();
  if (quantizer_prev->qscheme() == QScheme::PER_TENSOR_AFFINE) {
    return quantizer_prev;
  }
  QuantizerPtr quantizer;
  auto temp = static_cast<PerChannelAffineQuantizer*>(quantizer_prev.get());
  auto axis = temp->axis();
  auto scales = temp->scales();
  auto zero_points = temp->zero_points();
  if (dim == axis) {
    // Compute scales&zps for sub-tensor
    // *.select(0, start) could alternatively be replaced with *.slice(0, start,
    // end, step), but select has less overhead
    scales =
        is_select ? scales.select(0, start) : scales.slice(0, start, end, step);
    zero_points = is_select ? zero_points.select(0, start)
                            : zero_points.slice(0, start, end, step);
  }
  if (scales.numel() > 1) {
    // Axis only needs to be adjusted if the calling function is select(), since
    // select() reduces the number of dimensions of the tensor by 1, and remains
    // unchanged if calling function is slice()
    quantizer = make_per_channel_affine_quantizer(
        scales,
        zero_points,
        (is_select ? axis - 1 : axis),
        quantizer_prev->scalar_type());
  } else {
    quantizer = make_per_tensor_affine_quantizer(
        scales.item().to<double>(),
        zero_points.item().to<int64_t>(),
        quantizer_prev->scalar_type());
  }
  return quantizer;
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  return at::select_symint(self, dim, c10::SymInt{index});
}

Tensor select_symint(const Tensor& self, int64_t dim, c10::SymInt index) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.sym_sizes()[dim];
  // Note: `size < -index` is not equivalent to `size <= -1 - index` if index is
  // INT64_MIN For std::numeric_limits<int64_t>::min() result of unary minus is
  // undefined by the standard but in practice is equal to self. On the other
  // hand, indexing wrapping is valid for all negative int64_t values, as
  // x[INT64_MIN] is the same as x[INT64_MAX]
  if (size <= -1 - index || size <= index) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      TORCH_CHECK_INDEX(
          false,
          "select(): index ",
          index,
          " out of range for tensor of size ",
          self.sizes(),
          " at dimension ",
          self.names()[dim]);
    }
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    return select_sparse(self, dim, index.guard_int(__FILE__, __LINE__));
  }

  Tensor result;
  if (self.is_quantized()) {
    auto local_index = index.guard_int(__FILE__, __LINE__);

    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    auto storage_offset = self.storage_offset() + local_index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    auto quantizer = create_subtensor_quantizer(
        self, true, local_index, local_index + 1, dim, 1);
    result = as_strided_qtensorimpl(
        self, sizes, strides, storage_offset, std::move(quantizer));
  } else {
    std::vector<c10::SymInt> sizes(
        self.sym_sizes().begin(), self.sym_sizes().end());
    std::vector<c10::SymInt> strides(
        self.sym_strides().begin(), self.sym_strides().end());
    auto storage_offset = self.sym_storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    result = self.as_strided_symint(sizes, strides, storage_offset);
  }
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor select_backward_symint(
    const Tensor& grad,
    c10::SymIntArrayRef input_sizes,
    int64_t dim,
    c10::SymInt index) {
  auto grad_input = at::zeros_symint(input_sizes, grad.options());
  grad_input.select_symint(dim, std::move(index)).copy_(grad);
  return grad_input;
}

Tensor index_select_sparse_cpu(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  TORCH_CHECK(false, "Sparse index_select not supported in EasyFHE");
  return self;
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  // handle optional parameters
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");

  if (start_val < 0) {
    start_val += sizes[dim];
  }
  if (end_val < 0) {
    end_val += sizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= sizes[dim]) {
    start_val = sizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= sizes[dim]) {
    end_val = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start_val * strides[dim];
  auto len = end_val - start_val;
  sizes[dim] = (len + step - 1) / step; // round-up
  strides[dim] *= step;

  Tensor result;
  if (self.is_quantized()) {
    auto quantizer =
        create_subtensor_quantizer(self, false, start_val, end_val, dim, step);
    result = as_strided_qtensorimpl(
        self, sizes, strides, storage_offset, std::move(quantizer));
  } else {
    // NB: it is extremely important to perform a redispatch here for
    // the MPS backend; if you call directly to as_strided_tensorimpl,
    // the necessary metadata for MPS will not get setup and you will
    // get silently wrong results
    result = self.as_strided(sizes, strides, storage_offset);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor slice_inverse_symint(
    const Tensor& self,
    const Tensor& base,
    int64_t /* dim */,
    std::optional<SymInt> /* start */,
    std::optional<SymInt> /* end */,
    SymInt /* step */) {
  // assume self has enough to storage to be viewed with base's metadata
  return self.as_strided_symint(
      base.sym_sizes(), base.sym_strides(), base.sym_storage_offset());
}

Tensor slice_backward(
    const Tensor& grad,
    IntArrayRef input_sizes,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.slice(dim, start, end, step).copy_(grad);
  return grad_input;
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  const auto num_splits = get_num_splits(self, split_size, dim);
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size =
      split_size - (split_size * num_splits - self.size(dim));

  for (const auto i : c10::irange(num_splits)) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

std::vector<Tensor> split_symint(
    const Tensor& self,
    c10::SymIntArrayRef sizes,
    int64_t dim) {
  return at::split_with_sizes_symint(self, sizes, dim);
}

std::vector<Tensor> unsafe_split(
    const Tensor& self,
    int64_t split_size,
    int64_t dim) {
  auto result = at::native::split(self, split_size, dim);
  for (auto& t : result) {
    // TODO(Ailing): do we need to set version_counter here?
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(
          c10::VariableVersion(/*version=*/0));
    }
  }
  return result;
}

std::vector<Tensor> hsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(
      self.dim() >= 1,
      "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  int64_t dim = (self.dim() == 1) ? 0 : 1;
  TORCH_CHECK(
      split_size != 0 && self.sym_sizes()[dim] % split_size == 0,
      "torch.hsplit attempted to split along dimension ",
      dim,
      ", but the size of the dimension ",
      self.sizes()[dim],
      " is not divisible by the split_size ",
      split_size,
      "!");
  return at::tensor_split(self, split_size, dim);
}

std::vector<Tensor> vsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(
      self.dim() >= 2,
      "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  TORCH_CHECK(
      split_size != 0 && self.sym_sizes()[0] % split_size == 0,
      "torch.vsplit attempted to split along dimension ",
      0,
      ", but the size of the dimension ",
      self.sizes()[0],
      " is not divisible by the split_size ",
      split_size,
      "!");
  return at::tensor_split(self, split_size, 0);
}

std::vector<Tensor> dsplit(const Tensor& self, int64_t split_size) {
  TORCH_CHECK(
      self.dim() >= 3,
      "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  TORCH_CHECK(
      split_size != 0 && self.sym_sizes()[2] % split_size == 0,
      "torch.dsplit attempted to split along dimension ",
      2,
      ", but the size of the dimension ",
      self.sizes()[2],
      " is not divisible by the split_size ",
      split_size,
      "!");
  return at::tensor_split(self, split_size, 2);
}

std::vector<Tensor> split_with_sizes(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  const int64_t dim_size = self.size(dim);
  const int64_t num_splits = split_sizes.size();
  int64_t start_idx = 0;

  std::vector<Tensor> splits;
  splits.reserve(num_splits);
  for (const auto i : c10::irange(num_splits)) {
    auto length = split_sizes[i];
    TORCH_CHECK(
        length >= 0,
        "split_with_sizes expects split_sizes have only non-negative ",
        "entries, but got split_sizes=",
        split_sizes);
    splits.push_back(
        at::native::slice(self, dim, start_idx, start_idx + length, 1));
    start_idx += length;
  }
  TORCH_CHECK(
      start_idx == dim_size,
      "split_with_sizes expects split_sizes to sum exactly to ",
      dim_size,
      " (input tensor's size at dimension ",
      dim,
      "), ",
      "but got split_sizes=",
      split_sizes);
  return splits;
}

std::vector<Tensor> unsafe_split_with_sizes(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  auto result = at::native::split_with_sizes(self, split_sizes, dim);
  for (auto& t : result) {
    // TODO(Ailing): do we need to set version_counter here?
    if (!t.is_inference()) {
      t.unsafeGetTensorImpl()->set_version_counter(
          c10::VariableVersion(/*version=*/0));
    }
  }
  return result;
}

std::vector<Tensor> hsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(
      self.dim() >= 1,
      "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  return at::tensor_split(self, split_sizes, (self.dim() == 1) ? 0 : 1);
}

std::vector<Tensor> vsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(
      self.dim() >= 2,
      "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  return at::tensor_split(self, split_sizes, 0);
}

std::vector<Tensor> dsplit(const Tensor& self, IntArrayRef split_sizes) {
  TORCH_CHECK(
      self.dim() >= 3,
      "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ",
      self.dim(),
      " dimensions!")
  return at::tensor_split(self, split_sizes, 2);
}

// Precondition: tensors is non-empty
static inline std::vector<Tensor> get_stack_inputs(
    TensorList tensors,
    int64_t dim) {
  std::vector<Tensor> inputs(tensors.size());
  at::IntArrayRef entry_shape = tensors[0].sizes();
  inputs[0] = tensors[0].unsqueeze(dim);
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(
        tensors[i].sizes() == entry_shape,
        "stack expects each tensor to be equal size, but got ",
        entry_shape,
        " at entry 0 and ",
        tensors[i].sizes(),
        " at entry ",
        i);
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return inputs;
}

static bool inline maybe_native_stack(
    Tensor& result,
    TensorList tensors,
    int64_t dim) {
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  if (detail::CanUseNativeSerialStack<
          TensorList,
          /*skip_overlap_check*/ false>::call(result, tensors, dim)) {
    // compute the size of the result
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + dim, tensors.size());

    // skip resizing if size of result is same as expected
    // raise a warning while resizing if output has one or more elements
    // at::native::resize_output(result, result_sizes);
    // TODO: restore the above, see
    // https://github.com/pytorch/pytorch/issues/64709

    if (result.sizes() != result_sizes) {
      result.resize_(result_sizes);
    }

    stack_serial_stub(kCPU, result, tensors, dim);
    return true;
  }
  return false;
}

Tensor _stack(TensorList tensors, int64_t dim) {
  ScalarType high_type = result_type(tensors);
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  return at::native::_stack_out(get_stack_inputs(tensors, dim), dim, result);
}

Tensor _stack_cpu(TensorList tensors, int64_t dim) {
  ScalarType high_type = result_type(tensors);
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  return at::native::_stack_out_cpu(tensors, dim, result);
}

static void check_stack_inputs(TensorList tensors, int64_t dim) {
  at::IntArrayRef entry_shape = tensors[0].sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(
        tensors[i].sizes() == entry_shape,
        "stack expects each tensor to be equal size, but got ",
        entry_shape,
        " at entry 0 and ",
        tensors[i].sizes(),
        " at entry ",
        i);
  }
}

// Pads each tensor on `dim`-th dimension such that padded_dim % num_chunks ==
// 0.
static std::vector<Tensor> _pad_chunk(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks) {
  auto num_tensors = tensors.size();
  std::vector<Tensor> padded_tensors;
  padded_tensors.reserve(num_tensors);
  for (const auto& tensor : tensors) {
    auto tensor_size = tensor.sizes();
    std::vector<int64_t> padded_size(tensor_size.vec());
    padded_size[dim] =
        (tensor_size[dim] + num_chunks - 1) / num_chunks * num_chunks;
    Tensor padded_tensor = tensor;
    if (padded_size != tensor_size) {
      padded_tensor = tensor.new_zeros(padded_size);
      padded_tensor.narrow(dim, 0, tensor_size[dim]).copy_(tensor);
    }
    std::vector<int64_t> view_sizes(
        tensor_size.begin(), tensor_size.begin() + dim);
    view_sizes.insert(view_sizes.end(), {num_chunks, -1});
    padded_tensors.push_back(padded_tensor.reshape(view_sizes));
  }
  return padded_tensors;
}

Tensor _chunk_cat(TensorList tensors, int64_t dim, int64_t num_chunks) {
  auto wrapped_dim =
      at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  return at::cat(_pad_chunk(tensors, wrapped_dim, num_chunks), wrapped_dim + 1);
}

Tensor& _chunk_cat_out(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  auto wrapped_dim =
      at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  at::cat_out(
      out, _pad_chunk(tensors, wrapped_dim, num_chunks), wrapped_dim + 1);
  return out;
}

// TODO(msubkhankulov): refactor to use _stack
Tensor stack(TensorList tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "stack expects a non-empty TensorList");
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension() + 1);
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    check_stack_inputs(tensors, wrapped_dim);
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    auto out = at::cat(tensors, wrapped_dim);
    return out.view(result_sizes); // one can always split a dimension with view
  } else { // dim = tensors[0].ndimension() cannot be efficiently handled by
           // view
    return at::cat(get_stack_inputs(tensors, dim), dim);
  }
}

// CPU specific implementation
Tensor& _stack_out_cpu(TensorList tensors, int64_t dim, Tensor& result) {
  if (maybe_native_stack(result, tensors, dim)) {
    return result;
  } else {
    return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
  }
}

// default backend
Tensor& _stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  return at::cat_out(result, tensors, dim);
}

// TODO(msubkhankulov): refactor to use _stack_out
Tensor& stack_out(TensorList tensors, int64_t dim, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "stack expects a non-empty TensorList");
  auto wrapped_dim = maybe_wrap_dim(dim, tensors[0].ndimension() + 1);
  if (wrapped_dim < tensors[0].ndimension() && !tensors[0].is_sparse()) {
    check_stack_inputs(tensors, wrapped_dim);
    auto result_sizes = tensors[0].sizes().vec();
    result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
    at::native::resize_output(result, result_sizes);
    auto cat_sizes = tensors[0].sizes().vec();
    cat_sizes[wrapped_dim] *= tensors.size();
    auto strides =
        at::detail::computeStride(result.sizes(), result.strides(), cat_sizes);
    if (strides.has_value()) {
      // can take fast cat path
      auto result_view = result.view(cat_sizes);
      at::cat_out(result_view, tensors, wrapped_dim);
      return result;
    }
  }
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

Tensor hstack(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat(rep, 0);
  }
  return at::cat(rep, 1);
}

Tensor& hstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat_out(result, rep, 0);
  }
  return at::cat_out(result, rep, 1);
}

Tensor vstack(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat(rep, 0);
}

Tensor& vstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat_out(result, rep, 0);
}

Tensor dstack(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat(rep, 2);
}
Tensor& dstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat_out(result, rep, 2);
}

static inline Tensor& sparse_transpose_(
    Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "Sparse transpose not supported in EasyFHE");
  return self;
}

// torch.row_stack, alias for torch.vstack
Tensor& row_stack_out(TensorList tensors, Tensor& result) {
  return at::vstack_out(result, tensors);
}

Tensor row_stack(TensorList tensors) {
  return at::vstack(tensors);
}

static std::vector<Tensor> reshape_input_for_column_stack(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    // reshape 0D or 1D tensor t into (t.numel(), 1)
    if (input.dim() <= 1) {
      return input.reshape_symint({input.sym_numel(), 1});
    }
    return input;
  };
  std::transform(
      tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor& column_stack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(!tensors.empty(), "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack_out(result, reshaped_tensors);
}

Tensor column_stack(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack(reshaped_tensors);
}

static Tensor& propagate_transposed_names(
    Tensor& result,
    const Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(
      !(self.layout() == kSparseCsr || self.layout() == kSparseCsc ||
        self.layout() == kSparseBsr || self.layout() == kSparseBsc),
      "torch.transpose_: in-place transposition is not supported for ",
      self.layout(),
      " layout");

  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  // Sparse COO is an exceptional sparse format as it allows transpose
  // to be a view operation which is a convenient property for
  // in-place operations. For other sparse formats, the in-place
  // transpose would not be possible without shuffling the specified
  // values. So we don't support this as it would defeat the purpose
  // of in-place operations of being memory-efficient.
  if (self.is_sparse()) {
    return sparse_transpose_(self, dim0, dim1);
  }

  TORCH_CHECK(!self.is_mkldnn(), "MKLDNN transpose not supported in EasyFHE");

  SymDimVector sizes(self.sym_sizes().begin(), self.sym_sizes().end());
  std::swap(sizes[dim0], sizes[dim1]);
  SymDimVector strides(self.sym_strides().begin(), self.sym_strides().end());
  std::swap(strides[dim0], strides[dim1]);
  self.as_strided__symint(std::move(sizes), std::move(strides));
  return self;
}

namespace {
inline Tensor sparse_compressed_transpose(
    const Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  TORCH_CHECK(false, "Sparse compressed transpose not supported in EasyFHE");
  return self;
}
} // namespace

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);

  if (self.is_sparse()) {
    if (dim0 == dim1) {
      return self.clone();
    }
    Tensor self_clone = self.clone();
    return sparse_transpose_(self_clone, dim0, dim1);
  }
  if (self.layout() == kSparseBsr || self.layout() == kSparseCsr ||
      self.layout() == kSparseBsc || self.layout() == kSparseCsc) {
    return sparse_compressed_transpose(self, dim0, dim1);
  }

  if (self.is_mkldnn()) {
    TORCH_CHECK(false, "MKLDNN transpose not supported in EasyFHE");
  }

  // Transpose of a tensor is a view operation.
  if (dim0 == dim1) {
    return self.alias();
  }

  SymDimVector sizes(self.sym_sizes().begin(), self.sym_sizes().end());
  std::swap(sizes[dim0], sizes[dim1]);
  SymDimVector strides(self.sym_strides().begin(), self.sym_strides().end());
  std::swap(strides[dim0], strides[dim1]);
  auto result = self.as_strided_symint(sizes, strides);
  propagate_transposed_names(result, self, dim0, dim1);
  return result;
}

static void check_t(const Tensor& self, const char* fn) {
  if (self.is_sparse()) {
    TORCH_CHECK(false, "Sparse tensors not supported in EasyFHE");
  } else {
    TORCH_CHECK(
        self.dim() <= 2,
        fn,
        " expects a tensor with <= 2 dimensions, but self is ",
        self.dim(),
        "D");
  }
}

Tensor t(const Tensor& self) {
  check_t(self, "t()");
  return self.transpose(0, self.dim() < 2 ? 0 : 1);
}

Tensor& t_(Tensor& self) {
  check_t(self, "t_()");
  return self.transpose_(0, self.dim() < 2 ? 0 : 1);
}

std::tuple<SymDimVector, SymDimVector> static inferSqueezeGeometry(
    const Tensor& tensor) {
  SymDimVector sizes;
  SymDimVector strides;

  for (const auto d : c10::irange(tensor.dim())) {
    if (tensor.sym_sizes()[d] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }

  return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<SymDimVector, SymDimVector> static inferSqueezeGeometry(
    const Tensor& tensor,
    int64_t dim) {
  SymDimVector sizes;
  SymDimVector strides;

  for (const auto d : c10::irange(tensor.dim())) {
    if (d != dim || tensor.sym_sizes()[dim] != 1) {
      sizes.push_back(tensor.sym_sizes()[d]);
      strides.push_back(tensor.sym_strides()[d]);
    }
  }
  return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<SymDimVector, SymDimVector> static inferSqueezeGeometry(
    const Tensor& tensor,
    std::bitset<dim_bitset_size> dim_mask) {
  const auto ndim = tensor.dim();
  const auto sym_sizes = tensor.sym_sizes();
  const auto sym_strides = tensor.sym_strides();

  SymDimVector out_sizes, out_strides;
  for (const auto d : c10::irange(ndim)) {
    if (!dim_mask.test(d) || sym_sizes[d] != 1) {
      out_sizes.push_back(sym_sizes[d]);
      out_strides.push_back(sym_strides[d]);
    }
  }
  return std::make_tuple(std::move(out_sizes), std::move(out_strides));
}

namespace {
// Named type instead of a pair/tuple so that we can be sure to
// construct the vectors in place and get NRVO.
template <typename T>
struct InferUnsqueezeGeometryResult {
  SmallVector<T, kDimVectorStaticSize> sizes;
  SmallVector<T, kDimVectorStaticSize> strides;
  InferUnsqueezeGeometryResult(
      ArrayRef<T> tensor_sizes,
      ArrayRef<T> tensor_strides)
      : sizes(tensor_sizes.begin(), tensor_sizes.end()),
        strides(tensor_strides.begin(), tensor_strides.end()) {}
};

InferUnsqueezeGeometryResult<c10::SymInt> inferUnsqueezeGeometry_symint(
    const Tensor& tensor,
    int64_t dim) {
  InferUnsqueezeGeometryResult<c10::SymInt> result(
      tensor.sym_sizes(), tensor.sym_strides());
  c10::SymInt new_stride =
      dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}

InferUnsqueezeGeometryResult<int64_t> inferUnsqueezeGeometry(
    const Tensor& tensor,
    int64_t dim) {
  InferUnsqueezeGeometryResult<int64_t> result(
      tensor.sizes(), tensor.strides());
  int64_t new_stride =
      dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}

// dim is present if squeezing a single dimension and absent if squeezing all
// dimensions
Tensor squeeze_qtensor(const Tensor& self, c10::OptionalIntArrayRef dims) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  const auto ndim = self.dim();
  auto mask = dims.has_value()
      ? dim_list_to_bitset(dims, self.dim())
      : std::bitset<dim_bitset_size>((1ull << self.dim()) - 1);
  auto [sizes, strides] = inferSqueezeGeometry(self, mask);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    int64_t shift = 0;
    for (const auto d : c10::irange(ndim)) {
      if (mask.test(d) && self.sizes()[d] == 1) {
        TORCH_CHECK(
            axis != d,
            "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
        if (d < axis) {
          ++shift;
        }
      }
    }
    axis -= shift;
    quantizer = make_per_channel_affine_quantizer(
        per_channel_quantizer->scales(),
        per_channel_quantizer->zero_points(),
        axis,
        quantizer->scalar_type());
  }
  // TODO: quantized Tensor support for SymInt needs to be added but basic
  // building blocks are missing for now.
  auto result = make_qtensor(
      self,
      C10_AS_INTARRAYREF_SLOW(sizes),
      C10_AS_INTARRAYREF_SLOW(strides),
      std::move(quantizer));
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self, mask);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}
} // namespace

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  at::Tensor result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor squeeze_quantized(const Tensor& self) {
  return squeeze_qtensor(self, std::nullopt);
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, dims);
  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    return self.as_strided_symint(self.sym_sizes(), self.sym_strides());
  }
  auto g = inferSqueezeGeometry(self, dim);
  auto result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor squeeze_quantized(const Tensor& self, int64_t dim) {
  return squeeze_qtensor(self, dim);
}

Tensor squeeze(const Tensor& self, IntArrayRef dims) {
  auto mask = dim_list_to_bitset(dims, self.dim());
  auto g = inferSqueezeGeometry(self, mask);
  at::Tensor result = self.as_strided_symint(std::get<0>(g), std::get<1>(g));
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self, mask);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor squeeze_quantized(const Tensor& self, IntArrayRef dim) {
  return squeeze_qtensor(self, dim);
}

Tensor& squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  return self;
}

Tensor& squeeze_(Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, self.dim());

  if (dims == 0 || self.sym_sizes()[dim] != 1) {
    self.as_strided__symint(self.sym_sizes(), self.sym_strides());
    return self;
  }
  auto g = inferSqueezeGeometry(self, dim);
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  return self;
}

Tensor& squeeze_(Tensor& self, IntArrayRef dims) {
  auto mask = dim_list_to_bitset(dims, self.dim());
  auto g = inferSqueezeGeometry(self, mask);
  self.as_strided__symint(std::get<0>(g), std::get<1>(g));
  return self;
}

// NOTE [ Unsafe View ]
// _unsafe_view() differs from view() in that the returned tensor isn't treated
// as a view for the purposes of automatic differentiation. (It's not listed in
// VIEW_FUNCTIONS in gen_inplace_or_view_type.py).  It's only safe to use if the
// `self` tensor is temporary. For example, the viewed tensor here (a + b) is
// discarded immediately after viewing:
//
//  res = at::_unsafe_view(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.

static inline Tensor view_impl(const Tensor& self, IntArrayRef size) {
  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

Tensor _unsafe_view(const Tensor& self, IntArrayRef size) {
  return view_impl(self, size);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto g = inferUnsqueezeGeometry_symint(self, dim);
  return self.as_strided_symint(g.sizes, g.strides);
}

Tensor unsqueeze_sparse(Tensor const& self, int64_t dim) {
  TORCH_CHECK(false, "Sparse unsqueeze not supported in EasyFHE");
  return self;
}

Tensor unsqueeze_quantized(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto g = inferUnsqueezeGeometry(self, dim);
  auto quantizer = get_qtensorimpl(self)->quantizer();
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    quantizer = make_per_channel_affine_quantizer(
        per_channel_quantizer->scales(),
        per_channel_quantizer->zero_points(),
        axis,
        quantizer->scalar_type());
  }
  return make_qtensor(self, g.sizes, g.strides, std::move(quantizer));
}

Tensor& unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry_symint(self, dim);
  self.as_strided__symint(g.sizes, g.strides);
  return self;
}

Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");

  if (self.dim() == 0) {
    return self.reshape({1});
  }
  if (start_dim == end_dim) {
    return self;
  }

  auto slice_numel = c10::multiply_integers(
      self.sym_sizes().slice(start_dim, end_dim - start_dim + 1));
  std::vector<c10::SymInt> shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sym_sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (const auto i : c10::irange(end_dim + 1, self.dim())) {
    shape.push_back(self.sym_sizes()[i]);
  }

  return native::reshape_symint(self, shape);
}

Tensor ravel(const Tensor& self) {
  return self.contiguous().view(-1);
}

static inline void handle_unflatten_exception(
    const std::exception& e,
    const Tensor& self,
    int64_t dim,
    SymIntArrayRef sizes) {
  if (!strstr(e.what(), "is invalid for input of size")) {
    TORCH_CHECK(false, "unflatten got an unexpected error:\n", e.what());
  }

  if (self.has_names()) {
    TORCH_CHECK(
        false,
        "unflatten: Provided sizes ",
        sizes,
        " don't multiply up to the size of dim ",
        dim,
        " (",
        self.names()[dim],
        ": ",
        self.sym_size(dim),
        ") in Tensor",
        self.names());

  } else {
    TORCH_CHECK(
        false,
        "unflatten: Provided sizes ",
        sizes,
        " don't multiply up to the size of dim ",
        dim,
        " (",
        self.sym_size(dim),
        ") in the input tensor");
  }
}

static Tensor unflatten_impl(
    const Tensor& self,
    int64_t dim,
    SymIntArrayRef sizes,
    std::optional<DimnameList> names) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!sizes.empty(), "unflatten: sizes must be non-empty");
  TORCH_INTERNAL_ASSERT(!names || names->size() == sizes.size());
  if (self.has_names()) {
    TORCH_CHECK(
        names,
        "unflatten: input is a named tensor but no names were given for unflattened sizes");
  }

  SymDimVector inferred_size;
  try {
    inferred_size = at::infer_size_dv(sizes, self.sym_size(dim));
  } catch (const std::exception& e) {
    // at::infer_size would throw std::runtime_error for invalid size,
    // catch the runtime_error and display the error message in a more
    // user-friendly way for both tensors and named tensors
    handle_unflatten_exception(e, self, dim, sizes);
  }

  SymDimVector shape(self.sym_sizes().begin(), self.sym_sizes().end());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, inferred_size.begin(), inferred_size.end());

  Tensor result;
  {
    NoNamesGuard guard;
    result = self.view_symint(shape);
  }

  if (names) {
    auto outnames = self.names().vec();
    outnames.erase(outnames.begin() + dim);
    outnames.insert(outnames.begin() + dim, names->begin(), names->end());
    at::internal_set_names_inplace(result, outnames);
  }

  return result;
}

Tensor unflatten_symint(const Tensor& self, int64_t dim, SymIntArrayRef sizes) {
  return native::unflatten_impl(self, dim, sizes, std::nullopt);
}

Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view_symint(other.sym_sizes());
}

std::vector<Tensor> unbind(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  int64_t size = self.size(dim);
  std::vector<Tensor> tensors(size);
  for (const auto i : c10::irange(size)) {
    tensors[i] = self.select(dim, i);
  }
  return tensors;
}

std::vector<Tensor> meshgrid(TensorList tensors) {
  TORCH_WARN_ONCE(
      "torch.meshgrid: in an upcoming release, it will be required to pass the "
      "indexing argument.");
  return native::meshgrid(tensors, /*indexing=*/"ij");
}

std::vector<Tensor> meshgrid(TensorList tensors, std::string_view indexing) {
  int64_t size = tensors.size();
  TORCH_CHECK(size > 0, "meshgrid expects a non-empty TensorList");

  for (const auto i : c10::irange(size - 1)) {
    TORCH_CHECK(
        tensors[i].dtype() == tensors[i + 1].dtype(),
        "meshgrid expects all tensors to have the same dtype");
    TORCH_CHECK(
        tensors[i].device() == tensors[i + 1].device(),
        "meshgrid expects all tensors to have the same device");
  }

  // Input tensors is of type TensorList, which is an alias to a
  // constant array slice, which doesn't allow for mutations. We may
  // need to swap our first two elements if indexing is "ij", so we
  // unconditionally create a vector that we can reorder to keep the
  // implementation simple.
  //
  // We are not concerned with the performance of this relative to
  // constructor a grid for each input.
  std::vector<std::reference_wrapper<const Tensor>> tensor_refs(
      tensors.begin(), tensors.end());

  // Whether or not to swap the first two tensors.
  //
  // We only swap if there are at least two* input tensors (obviously)
  // and if indexing is "xy".
  //
  // A reminder about "xy" semantics: "xy" semantics implies that the
  // output grids are in the cartesian coordinate system. Thus the
  // first dimension is the "x" axis (corresponding to column) and the
  // second dimension is the "y" axis (corresponding to row). Tensors,
  // however, generally consider the first axis to be the row and the
  // second axis to be the columns. Thus we flip the two dimensions in
  // contrast to "ij" indexing.
  //
  // It turns out that it's easiest to implement this by just swapping
  // the first two inputs. However, the order of the outputs still
  // must correspond to the order of the inputs. Thus we also must
  // swap the outputs if we swapped the inputs.
  //
  // * Why do we even support this function for exactly one input?
  bool swap_first_and_second_tensors = false;

  if (indexing == "xy") {
    // We can only swap if there are multiple tensors.
    swap_first_and_second_tensors = size >= 2;
    if (swap_first_and_second_tensors) {
      std::swap(tensor_refs[0], tensor_refs[1]);
    }
  } else {
    // Only "xy" and "ij" are supported, and we already checked for
    // "xy" above. Only "ij" remains as a valid mode.
    TORCH_CHECK(
        indexing == "ij",
        "torch.meshgrid: indexing must be one of \"xy\" or \"ij\", "
        "but received: ",
        indexing);
  }

  std::vector<c10::SymInt> shape(size);
  for (const auto i : c10::irange(size)) {
    TORCH_CHECK(
        tensor_refs[i].get().dim() <= 1,
        "torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: ",
        tensor_refs[i]);
    shape[i] = tensor_refs[i]
                   .get()
                   .sym_numel(); // treat 0D tensors as if they were a 1D tensor
  }
  std::vector<Tensor> grids;
  grids.reserve(size);
  std::vector<c10::SymInt> view_shape(size, 1);
  for (const auto i : c10::irange(size)) {
    view_shape[i] = -1; // select this dimension to infer
    grids.push_back(
        tensor_refs[i].get().view_symint(view_shape).expand_symint(shape));
    view_shape[i] = 1; // restore to previous value
  }

  // Remember we need to also swap the outputs if we swapped the inputs.
  if (swap_first_and_second_tensors) {
    std::swap(grids[0], grids[1]);
  }
  return grids;
}

// Numpy-style `a.T`: returns the tensor
// with dims reversed
Tensor numpy_T(const Tensor& self) {
  const auto n = self.dim();
  if (n != 2 && n != 0) {
    TORCH_WARN_ONCE(
        "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated ",
        "and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices ",
        "or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.");
  }
  if (n == 0) {
    // Added in PyTorch 2.0
    TORCH_WARN_ONCE(
        "Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases.");
  }
  DimVector transpose_dims;
  for (int64_t i = n - 1; i >= 0; --i) {
    transpose_dims.push_back(i);
  }
  return self.permute(transpose_dims);
}

Tensor matrix_H(const Tensor& self) {
  const auto ndim = self.dim();
  if (ndim == 0) {
    // Added in PyTorch 2.0
    TORCH_WARN_ONCE(
        "Tensor.H is deprecated on 0-D tensors. Consider using x.conj().");
  }
  TORCH_CHECK(
      ndim == 2 || ndim == 0,
      "tensor.H is only supported on matrices (2-D tensors). Got ",
      ndim,
      "-D tensor.",
      ndim > 2 ? " For batches of matrices, consider using tensor.mH" : "");
  if (self.is_complex()) {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  } else {
    return ndim == 0 ? self : self.transpose(-2, -1);
  }
}

namespace {
Tensor _adjoint(
    const Tensor& self,
    const bool transpose,
    const char* const name) {
  const auto ndim = self.dim();
  TORCH_CHECK(
      ndim != 1,
      "tensor.",
      name,
      " is only supported on matrices or batches of matrices. Got 1-D tensor.");
  if (transpose || !self.is_complex()) {
    return ndim == 0 ? self : self.transpose(-2, -1);
  } else {
    return ndim == 0 ? self.conj() : self.transpose(-2, -1).conj();
  }
}
} // anonymous namespace

Tensor mT(const Tensor& self) {
  if (self.dim() == 0) {
    // Added in PyTorch 2.0
    TORCH_WARN_ONCE(
        "Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases.");
  }
  return _adjoint(self, /*transpose=*/true, "mT");
}

Tensor mH(const Tensor& self) {
  if (self.dim() == 0) {
    // Added in PyTorch 2.0
    TORCH_WARN_ONCE(
        "Tensor.mH is deprecated on 0-D tensors. Consider using x.conj().");
  }
  return _adjoint(self, /*transpose=*/false, "mH");
}

Tensor adjoint(const Tensor& self) {
  if (self.dim() == 0) {
    TORCH_WARN_ONCE(
        "adjoint() is deprecated on 0-D tensors. Consider using x.conj().");
  }
  return _adjoint(self, /*transpose=*/false, "adjoint()");
}

Tensor view(const Tensor& self, at::IntArrayRef size) {
  return view_impl(self, size);
}

Tensor alias(const Tensor& self) {
  return alias_with_sizes_and_strides(
      self, self.sym_sizes(), self.sym_strides());
}

Tensor detach(const Tensor& self) {
  // NB: detach() is not the same thing as alias()! The main difference is that
  // detach does not allow metadata change while alias does.
  return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
      // NB: The ADInplaceOrView logic will overwrite these with the
      // appropriate values if it runs; otherwise these are the values.
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/false));
}

Tensor diag(const Tensor& self, int64_t offset) {
  auto ndim = self.dim();
  TORCH_CHECK(
      ndim == 1 || ndim == 2,
      "diag(): Supports 1D or 2D tensors. Got ",
      self.dim(),
      "D");
  if (ndim == 1) {
    return at::diag_embed(self, offset);
  } else {
    // We return a copy of the diagonal
    return at::diagonal_copy(self, offset);
  }
}

Tensor& diag_out(const Tensor& self, int64_t offset, Tensor& out) {
  auto ndim = self.dim();
  TORCH_CHECK(
      ndim == 1 || ndim == 2,
      "Supports 1D or 2D tensors. Got ",
      self.dim(),
      "D");
  if (ndim == 1) {
    TORCH_CHECK(
        canCast(self.scalar_type(), out.scalar_type()),
        "diag: result type ",
        self.scalar_type(),
        " can't be cast to the desired out= type ",
        out.scalar_type());
    return at::diag_embed_out(out, self, offset);
  } else {
    return at::diagonal_copy_out(out, self, offset);
  }
}

Tensor diagonal_backward_symint(
    const Tensor& grad,
    SymIntArrayRef input_sizes,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  auto grad_input = at::zeros_symint(input_sizes, grad.options());
  auto diag = grad_input.diagonal(offset, dim1, dim2);
  diag.copy_(grad);
  return grad_input;
}

Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  TORCH_CHECK(
      src.size() == dst.size(),
      "movedim: Invalid source or destination dims: source (",
      src,
      " dims) should contain the same number of dims as destination (",
      dst,
      " dims)");

  size_t self_dim = self.dim();
  DimVector normalized_src(src.size());
  DimVector normalized_dst(dst.size());

  auto wrap_dims = [&self_dim](
                       const IntArrayRef& vec, DimVector& normalized_vec) {
    for (const auto i : c10::irange(vec.size())) {
      normalized_vec[i] = maybe_wrap_dim(vec[i], self_dim);
    }
  };

  wrap_dims(src, normalized_src);
  wrap_dims(dst, normalized_dst);

  auto all_unique = [](const DimVector& dims) {
    DimVector copy = dims;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    return duplicate == copy.end();
  };
  TORCH_CHECK(
      all_unique(normalized_src),
      "movedim: repeated dim in `source` (",
      src,
      ")");
  TORCH_CHECK(
      all_unique(normalized_dst),
      "movedim: repeated dim in `destination` (",
      dst,
      ")");

  // handle the case of scalar tensor as a no-op
  if (self_dim == 0)
    return self.alias();

  // TODO: The algorithm below can probably be optimized.
  // Reference:
  // https://github.com/pytorch/pytorch/pull/41480#discussion_r456100505

  // Algorithm Walkthrough
  // Example Input
  // Variable State:
  //     normalized_src = 0, 1
  //     normalized_dst = 2, 4
  //     self_dim = 5
  DimVector order(self_dim);
  DimVector source_dims(self_dim);
  DimVector destination_dims(self_dim);

  // We initialize two vectors to track update to the dims
  // `order` contains the final order of the dim positions.
  // Variable State:
  //     order = NA, NA, NA, NA, NA
  //     source_dims = 0, 1, 2, 3, 4
  //     destination_dims = 0, 1, 2, 3, 4
  std::iota(source_dims.begin(), source_dims.end(), 0);
  std::iota(destination_dims.begin(), destination_dims.end(), 0);

  // We mark and update position for the dim provided by user
  // i.e. `normalized_src` and `normalized_dims`
  // Variable State:
  //     order = NA, NA, 0, NA, 1
  //     source_dims = -1, -1, 2, 3, 4
  //     destination_dims = 0, 1, -1, 3, -1
  for (const auto i : c10::irange(src.size())) {
    order[normalized_dst[i]] = normalized_src[i];
    source_dims[normalized_src[i]] = -1;
    destination_dims[normalized_dst[i]] = -1;
  }

  // Remove the dims whose position we already know,
  // the ones marked with -1 in previous step
  // Variable State:
  //     source_dims = 2, 3, 4
  //     destination_dims = 0, 1, 3
  auto source_iter = std::remove(source_dims.begin(), source_dims.end(), -1);
  auto destination_iter =
      std::remove(destination_dims.begin(), destination_dims.end(), -1);

  int64_t rest_dim = self.dim() - src.size();
  TORCH_INTERNAL_ASSERT(
      std::distance(source_dims.begin(), source_iter) == rest_dim);
  TORCH_INTERNAL_ASSERT(
      std::distance(destination_dims.begin(), destination_iter) == rest_dim);

  // Update the position of the remaining dimensions.
  // `source_dims` now contains the original position
  // `destination_dims` contains the new position it will shifted to
  // after considering the user inputs.
  // Variable State:
  //     order = 2, 3, 0, 4, 1
  for (const auto i : c10::irange(rest_dim)) {
    order[destination_dims[i]] = source_dims[i];
  }

  return self.permute(order);
}

Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor moveaxis(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  return at::movedim(self, src, dst);
}

Tensor moveaxis(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose(axis0, axis1);
}

Tensor& swapaxes_(Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose_(axis0, axis1);
}

Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose(dim0, dim1);
}

Tensor& swapdims_(Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose_(dim0, dim1);
}

Tensor flatten_dense_tensors(TensorList tensors) {
  static auto flatten = [](const Tensor& t) {
    return t.contiguous().view({-1});
  };
  if (tensors.size() == 1)
    return flatten(tensors[0]);
  return at::cat(fmap(tensors, flatten));
}

std::vector<Tensor> unflatten_dense_tensors(
    const Tensor& flat,
    TensorList tensors) {
  std::vector<Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto& tensor : tensors) {
    auto numel = tensor.numel();
    // If unflatten an empty tensor, create a new empty tensor using
    // flat tensor Options.
    // This can avoid the unflattened empty tensor to share the same storage
    // with other unflatten tensors.
    if (numel == 0) {
      outputs.push_back(at::empty(tensor.sizes(), flat.options()));
    } else {
      outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
      offset += numel;
    }
  }
  return outputs;
}

// Clones a tensor by cloning the underlying storage that it came from,
// which allows us to replicate the exact strides/storage_offset in the cloned
// tensor. Note [*_scatter ops preserve strides] In order for functionalization
// to preserve stride correctness, the *_scatter operators that it calls must
// preserve the striding behavior of their inputs. Specifically, the output of
// *_scatter(base, mutated_view, ...) should have identical
// size/stride/storage_offset to "base".
at::Tensor clone_preserve_strides(const at::Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.has_storage());
  // In cases where the input tensor has internal memory overlap, we cannot
  // actually preserve the strides/storage_offset of the input tensor, because
  // *_scatter ops will try to copy_() into the cloned tensor.
  // However, this should **never** show up in functionalized user code;
  // most aten ops that try to mutate a tensor with internal memory overlap
  // would error anyway.
  //
  // The one place that this does come up is in autograd - if there's a
  // select_scatter in the forward, then autograd will generate one for the
  // backward. If the input to the select_scatter is grad_output, then this
  // could be an expanded tensor with internal overlap.
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    return self.clone();
  }
  auto dtype_size = c10::SymInt(self.dtype().itemsize());
  auto nbytes = self.storage().sym_nbytes();
  TORCH_INTERNAL_ASSERT(nbytes % dtype_size == 0);
  auto numel = nbytes / dtype_size;
  auto self_full_size = self.as_strided_symint({std::move(numel)}, {1}, 0);
  auto clone = self_full_size.clone();
  auto out = clone.as_strided_symint(
      self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());
  return out;
}

at::Tensor slice_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  // See Note [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  auto slice = output.slice(dim, start, end, step);
  TORCH_CHECK(
      slice.sizes() == src.sizes(),
      "expected src to have a size equal to the slice of self. src size = ",
      src.sizes(),
      ", slice size = ",
      slice.sizes());
  slice.copy_(src);
  return output;
}
at::Tensor select_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    c10::SymInt index) {
  auto output = clone_preserve_strides(self);
  auto slice = output.select_symint(dim, std::move(index));
  TORCH_CHECK(
      slice.sizes() == src.sizes(),
      "expected src to have a size equal to the slice of self. src size = ",
      src.sizes(),
      ", slice size = ",
      slice.sizes());
  slice.copy_(src);
  return output;
}
at::Tensor diagonal_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  // See Note [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  auto slice = output.diagonal(offset, dim1, dim2);
  TORCH_CHECK(
      slice.sizes() == src.sizes(),
      "expected src to have a size equal to the slice of self. src size = ",
      src.sizes(),
      ", slice size = ",
      slice.sizes());
  slice.copy_(src);
  return output;
}
at::Tensor as_strided_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  // See Note [as_strided_scatter backward support]
  TORCH_INTERNAL_ASSERT(
      !self.requires_grad() || self.is_contiguous(),
      "as_strided_scatter is currently only supported for contiguous inputs");
  // See Note [*_scatter ops preserve strides]
  auto output = clone_preserve_strides(self);
  auto slice =
      output.as_strided_symint(size, stride, std::move(storage_offset));
  TORCH_CHECK(
      slice.sym_sizes() == src.sym_sizes(),
      "expected src to have a size equal to the slice of self. src size = ",
      src.sym_sizes(),
      ", slice size = ",
      slice.sym_sizes());
  slice.copy_(src);
  return output;
}

// The default implementation of lift is a no-op.
// If TLS is set appropriately (for wrapper-tensor keys like Functionalize or
// functorch transforms), then we'll dispatch to one of their implementations,
// which will properly lift the tensor into a wrapper.
at::Tensor lift(const at::Tensor& self) {
  return self;
}

// See notes in native_functions.yaml
at::Tensor lift_fresh(const at::Tensor& self) {
  return self;
}

// Autogen kernels for tensor list ops dont work on XLA. TODO(jakeszwe)
void split_copy_Tensor_out(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim,
    at::TensorList out) {
  auto tmp = self.split(split_size, dim);

  TORCH_CHECK(
      out.size() == tmp.size(),
      "split_copy_Tensor_out() expected an out= argument of size ",
      tmp.size(),
      ", got size ",
      out.size());
  for (const auto i : c10::irange(out.size())) {
    out[i].copy_(tmp[i]);
  }
}

namespace {

void copy_tensor_array_to_out(
    const char* name,
    const std::vector<Tensor>& array,
    at::TensorList out) {
  TORCH_CHECK(
      out.size() == array.size(),
      name,
      " expected an out= argument of size ",
      array.size(),
      ", got size ",
      out.size());
  for (const auto i : c10::irange(out.size())) {
    if (resize_output_check(out[i], array[i].sizes())) {
      out[i].resize_(array[i].sizes());
    }
    TORCH_CHECK(
        out[i].dtype() == array[i].dtype(),
        "Expected out tensor to have dtype ",
        array[i].dtype(),
        ", but got ",
        out[i].dtype(),
        " instead");
    TORCH_CHECK(
        out[i].device() == array[i].device(),
        "Expected out tensor to have device ",
        array[i].device(),
        ", but got ",
        out[i].device(),
        " instead");
    out[i].copy_(array[i]);
  }
}

} // namespace

void split_with_sizes_copy_out(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  auto array = self.split_with_sizes(split_sizes, dim);
  copy_tensor_array_to_out("split_with_sizes_copy_out", array, out);
}

void unbind_copy_int_out(
    const at::Tensor& self,
    int64_t dim,
    at::TensorList out) {
  if (at::GradMode::is_enabled()) {
    for (const auto i : c10::irange(out.size())) {
      TORCH_CHECK(
          !out[i].requires_grad(),
          "unbind_copy(): functions with out=... arguments don't support automatic differentiation, "
          "but one of the arguments requires grad.");
    }
  }

  auto array = self.unbind(dim);
  copy_tensor_array_to_out("unbind_copy_int_out", array, out);
}

int64_t sparse_dim_default(const Tensor& self) {
  TORCH_CHECK(
      self.layout() == kStrided,
      "sparse_dim expected sparse or strided tensor layout but got ",
      self.layout());
  return 0;
}

int64_t dense_dim_default(const Tensor& self) {
  TORCH_CHECK(
      self.layout() == kStrided,
      "dense_dim expected sparse or strided tensor layout but got ",
      self.layout());
  return self.dim();
}

int64_t _nnz_default(const Tensor& self) {
  TORCH_CHECK(false, "_nnz not supported for non-sparse tensors");
  return 0;
}

bool is_coalesced_default(const Tensor& self) {
  return false;
}

Tensor& _coalesced_default_(Tensor& self, bool coalesced) {
  TORCH_CHECK(false, "_coalesced_ not supported for non-sparse tensors");
  return self;
}

Tensor _indices_default(const Tensor& self) {
  TORCH_CHECK(false, "_indices not supported for non-sparse tensors");
  return self;
}

Tensor _values_default(const Tensor& self) {
  TORCH_CHECK(false, "_values not supported for non-sparse tensors");
  return self;
}

} // namespace at::native
