cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})


if(NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-ignored-qualifiers")
  string(APPEND CMAKE_C_FLAGS " -Wno-ignored-qualifiers")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-absolute-value")
  string(APPEND CMAKE_C_FLAGS " -Wno-absolute-value")
endif(NOT MSVC)

# Can be compiled standalone
if(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  set(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  set(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  set(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  set(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
endif()

# These flag are used in Config but set externally. We must normalize them to
# 0/1 otherwise `#if ON` will be evaluated to false.
macro(set_bool OUT IN)
  if(${IN})
    set(${OUT} 1)
  else()
    set(${OUT} 0)
  endif()
endmacro()

function(rocm_generate_ck_conf SRC DST)
  set(CK_ENABLE_INT8  "ON")
  set(CK_ENABLE_FP16  "ON")
  set(CK_ENABLE_FP32  "ON")
  set(CK_ENABLE_FP64  "ON")
  set(CK_ENABLE_BF16  "ON")
  set(CK_ENABLE_FP8   "ON")
  set(CK_ENABLE_BF8   "ON")
  set(CK_USE_XDL  "ON")
  set(CK_USE_WMMA "ON")

  configure_file(
    "${SRC}"
    "${DST}"
    @ONLY
  )
endfunction()

set_bool(AT_BUILD_WITH_BLAS USE_BLAS)
set_bool(AT_BUILD_WITH_LAPACK USE_LAPACK)
set_bool(AT_BLAS_F2C BLAS_F2C)
set_bool(AT_BLAS_USE_CBLAS_DOT BLAS_USE_CBLAS_DOT)
set_bool(AT_MAGMA_ENABLED USE_MAGMA)
set_bool(CAFFE2_STATIC_LINK_CUDA_INT CAFFE2_STATIC_LINK_CUDA)
set_bool(AT_CUDNN_ENABLED CAFFE2_USE_CUDNN)
set_bool(AT_CUSPARSELT_ENABLED CAFFE2_USE_CUSPARSELT)
set_bool(AT_HIPSPARSELT_ENABLED CAFFE2_USE_HIPSPARSELT)

configure_file(Config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h")
# TODO: Do not generate CUDAConfig.h for ROCm BUILDS
# At the moment, `jit_macros.h` include CUDAConfig.h for both CUDA and HIP builds
if(USE_CUDA OR USE_ROCM)
  configure_file(cuda/CUDAConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDAConfig.h")
endif()
if(USE_ROCM)
  configure_file(hip/HIPConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/hip/HIPConfig.h")
endif()

# NB: If you edit these globs, you'll have to update setup.py package_data as well
file(GLOB_RECURSE ATen_CORE_HEADERS CONFIGURE_DEPENDS  "core/*.h")
file(GLOB_RECURSE ATen_CORE_SRCS CONFIGURE_DEPENDS "core/*.cpp")
set(ATen_TRANSFORMER_HEADERS)
if(NOT BUILD_LITE_INTERPRETER)
  file(GLOB_RECURSE ATen_CORE_TEST_SRCS CONFIGURE_DEPENDS "core/*_test.cpp")
endif()
EXCLUDE(ATen_CORE_SRCS "${ATen_CORE_SRCS}" ${ATen_CORE_TEST_SRCS})

file(GLOB base_h CONFIGURE_DEPENDS "*.h" "detail/*.h" "cpu/*.h" "cpu/vec/vec512/*.h" "cpu/vec/vec128/*.h" "cpu/vec/vec256/*.h" "cpu/vec/vec256/vsx/*.h" "cpu/vec/vec256/zarch/*.h" "cpu/vec/sve/*.h" "cpu/vec/*.h" "functorch/*.h" "accelerator/*.h")
file(GLOB base_cpp CONFIGURE_DEPENDS "*.cpp" "detail/*.cpp" "cpu/*.cpp" "functorch/*.cpp" "accelerator/*.cpp")
file(GLOB cuda_h CONFIGURE_DEPENDS "cuda/*.h" "cuda/detail/*.h" "cuda/*.cuh" "cuda/detail/*.cuh" "cuda/tunable/*.cuh" "cuda/tunable/*.h")
file(GLOB cuda_cpp CONFIGURE_DEPENDS "cuda/*.cpp" "cuda/detail/*.cpp" "cuda/tunable/*.cpp")
file(GLOB cuda_nvrtc_stub_h CONFIGURE_DEPENDS "cuda/nvrtc_stub/*.h")
file(GLOB cuda_nvrtc_stub_cpp CONFIGURE_DEPENDS "cuda/nvrtc_stub/*.cpp")
file(GLOB cuda_cu CONFIGURE_DEPENDS "cuda/*.cu" "cuda/detail/*.cu" "cuda/tunable/*.cu")
set(cudnn_h)
set(cudnn_cpp)
file(GLOB ops_h CONFIGURE_DEPENDS "ops/*.h")

# MTIA
file(GLOB mtia_h CONFIGURE_DEPENDS "mtia/*.h" "mtia/detail/*.h")
file(GLOB mtia_cpp CONFIGURE_DEPENDS "mtia/*.cpp" "mtia/detail/*.cpp")
file(GLOB_RECURSE native_mtia_cpp CONFIGURE_DEPENDS "native/mtia/*.cpp")
file(GLOB_RECURSE native_mtia_h CONFIGURE_DEPENDS "native/mtia/*.h")

set(xpu_h)
set(xpu_cpp)

set(hip_h)
set(hip_cpp)
set(hip_hip)
file(GLOB hip_nvrtc_stub_h CONFIGURE_DEPENDS "hip/nvrtc_stub/*.h")
file(GLOB hip_nvrtc_stub_cpp CONFIGURE_DEPENDS "hip/nvrtc_stub/*.cpp")
set(miopen_h)
set(miopen_cpp)

set(mkl_cpp)
file(GLOB mkldnn_cpp CONFIGURE_DEPENDS "mkldnn/*.cpp")

file(GLOB native_cpp CONFIGURE_DEPENDS "native/*.cpp")
set(native_mkl_cpp)
file(GLOB vulkan_cpp CONFIGURE_DEPENDS "vulkan/*.cpp")
set(native_vulkan_cpp)

file(GLOB native_eigen_cpp CONFIGURE_DEPENDS "native/sparse/eigen/*.cpp")

# Metal
file(GLOB metal_h CONFIGURE_DEPENDS "metal/*.h")
file(GLOB metal_cpp CONFIGURE_DEPENDS "metal/*.cpp")
file(GLOB_RECURSE native_metal_h CONFIGURE_DEPENDS "native/metal/*.h")
file(GLOB metal_test_srcs CONFIGURE_DEPENDS "native/metal/mpscnn/tests/*.mm")
file(GLOB_RECURSE native_metal_srcs CONFIGURE_DEPENDS "native/metal/*.mm" "native/metal/*.cpp")
EXCLUDE(native_metal_srcs "${native_metal_srcs}" ${metal_test_srcs})
file(GLOB metal_prepack_h CONFIGURE_DEPENDS "native/metal/MetalPrepackOpContext.h")
file(GLOB metal_prepack_cpp CONFIGURE_DEPENDS "native/metal/MetalPrepackOpRegister.cpp")

set(native_ao_sparse_cpp)
# MPS
set(mps_cpp)
set(mps_mm)
set(mps_h)
set(native_mps_cpp)
set(native_mps_mm)
set(native_mps_metal)
set(native_mps_h)
file(GLOB_RECURSE native_sparse_mps_mm CONFIGURE_DEPENDS "native/sparse/mps/*.mm")
file(GLOB_RECURSE native_mps_sparse_metal CONFIGURE_DEPENDS "native/sparse/mps/*.metal")

file(GLOB native_sparse_cpp CONFIGURE_DEPENDS "native/sparse/*.cpp")
set(native_quantized_cpp)
set(native_nested_cpp)
set(native_transformers_cpp)
file(GLOB_RECURSE native_fhe_cpp CONFIGURE_DEPENDS
  "native/fhe/cpu/*.cpp"
  "native/fhe/sampler/*.cpp"
  "native/fhe/sampler/third_party/blake2/utils/prng/*.c")

file(GLOB native_h CONFIGURE_DEPENDS "native/*.h")
set(native_ao_sparse_h)
set(native_quantized_h)
file(GLOB native_cpu_h CONFIGURE_DEPENDS "native/cpu/*.h")
file(GLOB native_utils_h CONFIGURE_DEPENDS "native/utils/*.h")
file(GLOB native_fhe_h CONFIGURE_DEPENDS "native/fhe/cpu/*.h")

file(GLOB native_cuda_cu CONFIGURE_DEPENDS "native/cuda/*.cu")
file(GLOB native_cuda_cpp CONFIGURE_DEPENDS "native/cuda/*.cpp")
file(GLOB native_cuda_h CONFIGURE_DEPENDS "native/cuda/*.h" "native/cuda/*.cuh")
file(GLOB native_cuda_linalg_cpp CONFIGURE_DEPENDS "native/cuda/linalg/*.cpp")
set(native_hip_h)
set(native_cudnn_cpp)
file(GLOB native_sparse_cuda_cu CONFIGURE_DEPENDS "native/sparse/cuda/*.cu")
file(GLOB native_sparse_cuda_cpp CONFIGURE_DEPENDS "native/sparse/cuda/*.cpp")
set(native_quantized_cuda_cu)
set(native_quantized_cuda_cpp)
set(native_quantized_cudnn_cpp)
set(native_nested_h)
set(native_nested_cuda_cu)
set(native_nested_cuda_cpp)
file(GLOB_RECURSE native_fhe_cuda_cu CONFIGURE_DEPENDS "native/fhe/cuda/*.cu")
file(GLOB_RECURSE native_fhe_cuda_cpp CONFIGURE_DEPENDS "native/fhe/cuda/*.cpp")

set(native_hip_hip)
set(native_hip_cpp)
set(native_hip_linalg_cpp)
set(native_miopen_cpp)
set(native_cudnn_hip_cpp)
set(native_nested_hip_hip)
set(native_nested_hip_cpp)
file(GLOB native_sparse_hip_hip CONFIGURE_DEPENDS "native/sparse/hip/*.hip")
file(GLOB native_sparse_hip_cpp CONFIGURE_DEPENDS "native/sparse/hip/*.cpp")
set(native_quantized_hip_hip)
set(native_quantized_hip_cpp)
set(native_transformers_cuda_cu)
set(native_transformers_cuda_cpp)
set(native_quantized_cudnn_hip_cpp)
file(GLOB native_utils_cpp CONFIGURE_DEPENDS "native/utils/*.cpp")
# MSLK
IF(USE_MSLK)
  set(MSLK_THIRD_PARTY ${PROJECT_SOURCE_DIR}/third_party/mslk/external/)
  set(MSLK_SRCS ${PROJECT_SOURCE_DIR}/third_party/mslk/csrc/gemm/)
  set(MSLK_INCLUDE ${PROJECT_SOURCE_DIR}/third_party/mslk/include/)

  if(USE_CUDA)
    # To avoid increasing the build time/binary size unnecessarily, use an allow-list of kernels to build.
    # If you want to integrate a kernel from MSLK into torch, you have to add it here.
    set(MSLK_CUTLASS_KERNELS_REGEX ".*(mx8mx8bf16_grouped|f4f4bf16_grouped|f4f4bf16).*")
    file(GLOB_RECURSE mslk_native_cuda_cu CONFIGURE_DEPENDS
      "${MSLK_SRCS}/cutlass/*.cu"
      "${MSLK_SRCS}/cutlass/**/*.cu")
    list(FILTER mslk_native_cuda_cu INCLUDE REGEX ${MSLK_CUTLASS_KERNELS_REGEX})

    # PyTorch is not built for 10.0a in CI, due to lack of portability,
    # so we need to explicitly build these files for 10.0a.
    foreach(cu_file ${mslk_native_cuda_cu})
      _BUILD_FOR_ADDITIONAL_ARCHS(
        "${cu_file}"
        "100a;103f;110a")
    endforeach()

    file(GLOB_RECURSE mslk_native_cuda_cpp CONFIGURE_DEPENDS
      "${MSLK_SRCS}/common/*.cpp"
    )

    # Combine all source files into a single list
    list(APPEND mslk_all_sources
      ${mslk_native_cuda_cu}
      ${mslk_native_cuda_cpp}
    )

    # Now, create the library and provide the sources at the same time
    add_library(mslk OBJECT ${mslk_all_sources})

    set_target_properties(mslk PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set(mslk_cuh
      "${MSLK_SRCS}/cutlass/mx8mx8bf16_grouped/"
      "${MSLK_SRCS}/cutlass/f4f4bf16_grouped/"
    )

    target_include_directories(mslk PRIVATE
      ${MSLK_THIRD_PARTY}/cutlass/include
      ${MSLK_THIRD_PARTY}/cutlass/tools/util/include
      ${mslk_cuh}
      ${MSLK_INCLUDE} # includes mslk/gemm/gemm_torch.h, mslk/utils/utils.h, mslk/utils/tuning_cache.hpp
    )

    # Add MSLK include directories for gemm_torch.h
    list(APPEND ATen_CUDA_INCLUDE ${MSLK_INCLUDE})
  elseif(USE_ROCM)
    # Path to vendored CK
    set(MSLK_CK_DIR "${MSLK_THIRD_PARTY}/composable_kernel")
    rocm_generate_ck_conf(
      "${MSLK_CK_DIR}/include/ck/config.h.in"
      "${MSLK_CK_DIR}/include/ck/config.h"
    )
    # Only include the kernels we want to build to avoid increasing binary size.
    file(GLOB_RECURSE mslk_native_rocm_hip CONFIGURE_DEPENDS
      "${MSLK_SRCS}/ck/fp8_rowwise_grouped/kernels/fp8_rowwise_grouped*.hip"
      "${MSLK_SRCS}/ck/fp8_rowwise_grouped/fp8_rowwise_grouped_gemm.hip")
    set_source_files_properties(${mslk_native_rocm_hip} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

    # Add additional HIPCC compiler flags for performance
    set(MSLK_EXTRA_HIPCC_FLAGS
      -mllvm
      -enable-post-misched=0
      -mllvm
      -greedy-reverse-local-assignment=1
      -fhip-new-launch-api)
    if(DEFINED ROCM_VERSION_DEV AND ROCM_VERSION_DEV VERSION_LESS "7.2.0")
        list(PREPEND MSLK_EXTRA_HIPCC_FLAGS -mllvm -amdgpu-coerce-illegal-types=1)
      endif()

    # Only compile for gfx942 and gfx950.
    set(HIP_CLANG_FLAGS_ORIGINAL ${HIP_CLANG_FLAGS})
    string(REGEX REPLACE "--offload-arch=[^ ]*" "" FILTERED_HIP_CLANG_FLAGS "${HIP_CLANG_FLAGS}")
    foreach(ARCH gfx942 gfx950)
      if(${ARCH} IN_LIST PYTORCH_ROCM_ARCH)
        list(APPEND FILTERED_HIP_CLANG_FLAGS --offload-arch=${ARCH})
      endif()
    endforeach()
    set(HIP_CLANG_FLAGS ${FILTERED_HIP_CLANG_FLAGS})

    hip_add_library(
      mslk STATIC
      ${mslk_native_rocm_hip}
      HIPCC_OPTIONS ${HIP_HCC_FLAGS} ${MSLK_EXTRA_HIPCC_FLAGS})
    set(HIP_CLANG_FLAGS ${HIP_CLANG_FLAGS_ORIGINAL})
    set_target_properties(mslk PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mslk PRIVATE MSLK_NO_EXTENDED_SHAPES)

    target_include_directories(mslk PRIVATE
      # MSLK version of Composable Kernel is used due to some customizations
      ${MSLK_THIRD_PARTY}/composable_kernel/include
      ${MSLK_THIRD_PARTY}/composable_kernel/library/include
      ${MSLK_THIRD_PARTY}/cutlass/include
      ${MSLK_THIRD_PARTY}/cutlass/tools/util/include
      ${MSLK_INCLUDE} # includes mslk/gemm/gemm_torch.h, mslk/utils/utils.h, mslk/utils/tuning_cache.hpp
    )

    # Add MSLK include directories for gemm_torch.h
    list(APPEND ATen_HIP_INCLUDE ${MSLK_INCLUDE})
  endif()

  # Ensure torch_cpu is built before mslk to avoid race conditions
  add_dependencies(mslk torch_cpu)
endif()

# KLEIDIAI
file(GLOB native_kleidiai CONFIGURE_DEPENDS "native/kleidiai/*.cpp")
file(GLOB native_kleidiai_h CONFIGURE_DEPENDS "native/kleidiai/*.h")

# Add files needed from jit folders
append_filelist("jit_core_headers" ATen_CORE_HEADERS)
append_filelist("jit_core_sources" ATen_CORE_SRCS)

# Quantized and NNAPI removed for EasyFHE

if(BUILD_LITE_INTERPRETER)
  set(all_cpu_cpp ${generated_sources} ${core_generated_sources} ${cpu_kernel_cpp})
  append_filelist("jit_core_sources" all_cpu_cpp)
  append_filelist("aten_cpu_source_non_codegen_list" all_cpu_cpp)
  append_filelist("aten_native_source_non_codegen_list" all_cpu_cpp)
else()
  set(
    all_cpu_cpp ${base_cpp} ${ATen_CORE_SRCS} ${native_cpp}
    ${native_sparse_cpp}
    ${native_quantized_cpp}
    ${native_mkl_cpp}
    ${native_fhe_cpp}
    ${native_utils_cpp} ${generated_sources} ${core_generated_sources}
    ${ATen_CPU_SRCS} ${ATen_NNAPI_SRCS} ${cpu_kernel_cpp}
  )
endif()

if(USE_LIGHTWEIGHT_DISPATCH)
  set(all_cpu_cpp ${all_cpu_cpp} ${generated_unboxing_sources})
endif()
if(AT_MKL_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkl_cpp})
endif()
if(AT_KLEIDIAI_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${native_kleidiai})
endif()
if(USE_VULKAN)
  set(all_cpu_cpp ${all_cpu_cpp} ${vulkan_cpp} ${native_vulkan_cpp} ${vulkan_generated_cpp})
else()
  set(all_cpu_cpp ${all_cpu_cpp} ${vulkan_cpp})
endif()
if(USE_EIGEN_SPARSE)
  set(all_cpu_cpp ${all_cpu_cpp} ${native_eigen_cpp})
endif()

if(USE_MTIA)
    set(ATen_MTIA_SRCS ${ATen_MTIA_SRCS} ${mtia_cpp} ${mtia_h} ${native_mtia_cpp} ${native_mtia_h})
endif()

if(USE_XPU)
  list(APPEND ATen_XPU_DEPENDENCY_LIBS xpu_mkldnn)

  list(APPEND ATen_XPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/native/mkldnn/xpu)
  list(APPEND ATen_XPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/native/mkldnn/xpu/detail)
  list(APPEND ATen_XPU_INCLUDE ${XPU_MKLDNN_INCLUDE})

  list(APPEND ATen_XPU_INCLUDE ${SYCL_INCLUDE_DIR})
  list(APPEND ATen_XPU_DEPENDENCY_LIBS ${SYCL_LIBRARY})
endif()

# Metal
if(USE_PYTORCH_METAL_EXPORT)
  # Add files needed from exporting metal models(optimized_for_mobile)
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp} ${metal_prepack_cpp})
elseif(APPLE AND USE_PYTORCH_METAL)
  # Compile Metal kernels
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp} ${native_metal_srcs})
else()
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp})
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "ATen doesn't not currently support simultaneously building with CUDA and ROCM")
endif()

if(USE_CUDA)
  list(APPEND ATen_CUDA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  # Next two lines are needed because TunableOp uses third-party/fmt
  list(APPEND ATen_CUDA_INCLUDE $<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>)
  list(APPEND ATen_CUDA_DEPENDENCY_LIBS fmt::fmt-header-only)
  list(APPEND ATen_CUDA_CU_SRCS
    ${cuda_cu}
    ${native_cuda_cu}
    ${native_transformers_cuda_cu}
    ${cuda_generated_sources}
    ${native_fhe_cuda_cu}
  )
  list(APPEND ATen_CUDA_CPP_SRCS
    ${cuda_cpp}
    ${native_cuda_cpp}
    ${native_cudnn_cpp}
    ${native_miopen_cpp}
    ${native_transformers_cuda_cpp}
    ${native_fhe_cuda_cpp}
  )
  set(ATen_CUDA_LINALG_SRCS ${native_cuda_linalg_cpp})
  if(NOT BUILD_LAZY_CUDA_LINALG)
    list(APPEND ATen_CUDA_CU_SRCS ${native_cuda_linalg_cpp})
  endif()
  if(CAFFE2_USE_CUDNN)
    list(APPEND ATen_CUDA_CPP_SRCS ${cudnn_cpp})
  endif()

  append_filelist("aten_cuda_cu_source_list" ATen_CUDA_CU_SRCS)
  append_filelist("aten_cuda_with_sort_by_key_source_list" ATen_CUDA_SRCS_W_SORT_BY_KEY)
  append_filelist("aten_cuda_cu_with_sort_by_key_source_list" ATen_CUDA_CU_SRCS_W_SORT_BY_KEY)

  exclude(ATen_CUDA_CPP_SRCS "${ATen_CUDA_CPP_SRCS}"
      ${ATen_CUDA_CU_SRCS}
      ${ATen_CUDA_SRCS_W_SORT_BY_KEY} ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
  exclude(ATen_CUDA_CU_SRCS "${ATen_CUDA_CU_SRCS}"
      ${ATen_CUDA_SRCS_W_SORT_BY_KEY} ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
endif()

if(USE_ROCM)
  if(USE_ROCM_CK_GEMM)
    list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/hip)
    list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/composable_kernel/include)
    list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/composable_kernel/library/include)
    list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/composable_kernel/example/ck_tile/01_fmha)
    list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/composable_kernel)
    # NOTE: The PyTorch build does not actually add_subdirectory
    # third_party/composable_kernel or use it as a CMake library. What is used
    # is header only, so this should be ok, except that the CMake build generates
    # a ck/config.h. We just do that part here. Without this, the ck.h from the
    # ROCM SDK may get accidentally used instead.
    rocm_generate_ck_conf(
      "${Torch_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/composable_kernel/ck/config.h"
    )
  endif()

  # Next two lines are needed because TunableOp uses third-party/fmt
  list(APPEND ATen_HIP_INCLUDE $<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>)
  list(APPEND ATen_HIP_DEPENDENCY_LIBS fmt::fmt-header-only)
  list(APPEND ATen_HIP_SRCS
    ${ATen_HIP_SRCS}
    ${hip_hip}
    ${native_hip_hip}
    ${native_nested_hip_hip}
    ${native_sparse_hip_hip}
    ${native_quantized_hip_hip}
  )
  if(NOT USE_ROCM_CK_GEMM)
    file(GLOB native_hip_bgemm CONFIGURE_DEPENDS "native/hip/bgemm_kernels/*.hip")
    file(GLOB native_hip_ck CONFIGURE_DEPENDS "native/hip/ck*.hip")
    exclude(ATen_HIP_SRCS "${ATen_HIP_SRCS}"
      ${native_hip_bgemm} ${native_hip_ck})
  endif()

  # TODO: Codegen separate files for HIP and use those (s/cuda_generated_sources/hip_generated_sources)
  list(APPEND all_hip_cpp
    ${native_nested_hip_cpp}
    ${native_sparse_hip_cpp}
    ${native_quantized_hip_cpp}
    ${native_quantized_cudnn_hip_cpp}
    ${hip_cpp}
    ${native_hip_cpp}
    ${native_hip_linalg_cpp}
    ${cuda_generated_sources}
    ${ATen_HIP_SRCS}
    ${native_miopen_cpp}
    ${native_cudnn_hip_cpp}
    ${miopen_cpp}
    ${all_hip_cpp}
  )
endif()

if(USE_XPU)
  list(APPEND ATen_XPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/xpu)
  list(APPEND ATen_XPU_SRCS ${xpu_cpp})
  list(APPEND ATen_XPU_SRCS ${xpu_generated_sources})
endif()

list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(BLAS_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

if(LAPACK_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  if(USE_CUDA AND MSVC)
    # Although Lapack provides CPU (and thus, one might expect that ATen_cuda
    # would not need this at all), some of our libraries (magma in particular)
    # backend to CPU BLAS/LAPACK implementations, and so it is very important
    # we get the *right* implementation, because even if the symbols are the
    # same, LAPACK implementations may have different calling conventions.
    # This caused https://github.com/pytorch/pytorch/issues/7353
    #
    # We do NOT do this on Linux, since we just rely on torch_cpu to
    # provide all of the symbols we need
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()
endif(LAPACK_FOUND)

if(UNIX AND NOT APPLE)
   include(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   if(NEED_LIBRT)
     list(APPEND ATen_CPU_DEPENDENCY_LIBS rt)
     set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   endif(NEED_LIBRT)
endif(UNIX AND NOT APPLE)

if(UNIX)
  include(CheckFunctionExists)
  set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  if(HAVE_MMAP)
    add_definitions(-DHAVE_MMAP=1)
  endif(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
  add_definitions(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  if(HAVE_SHM_OPEN)
    add_definitions(-DHAVE_SHM_OPEN=1)
  endif(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  if(HAVE_SHM_UNLINK)
    add_definitions(-DHAVE_SHM_UNLINK=1)
  endif(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  if(HAVE_MALLOC_USABLE_SIZE)
    add_definitions(-DHAVE_MALLOC_USABLE_SIZE=1)
  endif(HAVE_MALLOC_USABLE_SIZE)
  set(CMAKE_EXTRA_INCLUDE_FILES "fcntl.h")
  CHECK_FUNCTION_EXISTS(posix_fallocate HAVE_POSIX_FALLOCATE)
  if(HAVE_POSIX_FALLOCATE)
    add_definitions(-DHAVE_POSIX_FALLOCATE=1)
  endif(HAVE_POSIX_FALLOCATE)
endif(UNIX)

ADD_DEFINITIONS(-DUSE_EXTERNAL_MZCRC)

if(NOT MSVC)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS m)
endif()


if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x|ppc64le)$")
  list(APPEND ATen_CPU_DEPENDENCY_LIBS cpuinfo)
endif()

if(NOT EMSCRIPTEN AND NOT INTERN_BUILD_MOBILE)
  if(NOT MSVC)
    # Bump up optimization level for sleef to -O1, since at -O0 the compiler
    # excessively spills intermediate vector registers to the stack
    # and makes things run impossibly slowly
    set(OLD_CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    if(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
      string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
    else()
      set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    set(SLEEF_ARCH_AARCH64 ON)
  endif()

  if(NOT USE_SYSTEM_SLEEF)
    set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
    set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
    set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will be built." FORCE)
    set(SLEEF_DISABLE_OPENMP ON CACHE BOOL "OpenMP only used by sleef DFT and tests, both disabled" FORCE)
    if(WIN32)
      set(SLEEF_BUILD_WITH_LIBM OFF CACHE BOOL "Don't build sleef with libm for Windows." FORCE)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(DISABLE_SVE ON CACHE BOOL "Xcode's clang-12.5 crashes while trying to compile SVE code" FORCE)
      endif()
    endif()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/sleef" ${CMAKE_BINARY_DIR}/sleef)
    set_property(TARGET sleef PROPERTY FOLDER "dependencies")
    list(APPEND ATen_THIRD_PARTY_INCLUDE ${CMAKE_BINARY_DIR}/include)
    link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
  else()
    add_library(sleef SHARED IMPORTED)
    find_library(SLEEF_LIBRARY sleef)
    if(NOT SLEEF_LIBRARY)
      message(FATAL_ERROR "Cannot find sleef")
    endif()
    message("Found sleef: ${SLEEF_LIBRARY}")
    set_target_properties(sleef PROPERTIES IMPORTED_LOCATION "${SLEEF_LIBRARY}")
  endif()
  list(APPEND ATen_CPU_DEPENDENCY_LIBS sleef)

  if(NOT MSVC)
    set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  endif()
endif()

if(USE_CUDA AND NOT USE_ROCM)
  if($ENV{ATEN_STATIC_CUDA})
    if(CUDA_VERSION VERSION_LESS_EQUAL 12.9)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
          ${CUDA_LIBRARIES}
          CUDA::cusparse_static
          CUDA::cufft_static_nocallback)
    else()
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
          ${CUDA_LIBRARIES}
          CUDA::cusparse_static
          CUDA::cufft_static)
    endif()

   if(NOT BUILD_LAZY_CUDA_LINALG)
     list(APPEND ATen_CUDA_DEPENDENCY_LIBS
       CUDA::cusolver_static
       ${CUDAToolkit_LIBRARY_DIR}/libcusolver_lapack_static.a     # needed for libcusolver_static
     )
   endif()
  else()
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      CUDA::cusparse
      CUDA::cufft
    )
   if(NOT BUILD_LAZY_CUDA_LINALG)
     list(APPEND ATen_CUDA_DEPENDENCY_LIBS
       CUDA::cusolver
     )
   endif()
  endif()

  if(CAFFE2_USE_CUDNN)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
  endif()
  if($ENV{ATEN_STATIC_CUDA})
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      CUDA::culibos
      CUDA::cudart_static
    )
  endif($ENV{ATEN_STATIC_CUDA})
endif()


  if(USE_MAGMA)
    if(USE_CUDA AND NOT BUILD_LAZY_CUDA_LINALG)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS torch::magma)
    endif(USE_CUDA AND NOT BUILD_LAZY_CUDA_LINALG)
    if(USE_ROCM)
      list(APPEND ATen_HIP_DEPENDENCY_LIBS torch::magma)
    endif(USE_ROCM)
    if(MSVC)
      if($ENV{TH_BINARY_BUILD})
        # Do not do this on Linux: see Note [Extra MKL symbols for MAGMA in torch_cpu]
        # in caffe2/CMakeLists.txt
        list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
      endif($ENV{TH_BINARY_BUILD})
    endif(MSVC)
  endif(USE_MAGMA)

# Include CPU paths for CUDA/HIP as well
list(APPEND ATen_CUDA_INCLUDE ${ATen_CPU_INCLUDE})
list(APPEND ATen_HIP_INCLUDE ${ATen_CPU_INCLUDE})
list(APPEND ATen_VULKAN_INCLUDE ${ATen_CPU_INCLUDE})

# We have two libraries: libATen_cpu.so and libATen_cuda.so,
# with libATen_cuda.so depending on libATen_cpu.so.  The CPU library
# contains CPU code only.  libATen_cpu.so is invariant to the setting
# of USE_CUDA (it always builds the same way); libATen_cuda.so is only
# built when USE_CUDA=1 and CUDA is available.  (libATen_hip.so works
# the same way as libATen_cuda.so)
set(ATen_CPU_SRCS ${all_cpu_cpp})
list(APPEND ATen_CPU_DEPENDENCY_LIBS ATEN_CPU_FILES_GEN_LIB)

if(USE_CUDA)
  set(ATen_NVRTC_STUB_SRCS ${cuda_nvrtc_stub_cpp})
  list(APPEND ATen_CUDA_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
endif()

if(USE_MPS)
    include(../../../cmake/Metal.cmake)

    set(ATen_MPS_SRCS ${ATen_MPS_SRCS} ${mps_cpp} ${mps_mm} ${mps_h} ${native_mps_cpp} ${native_mps_mm} ${native_mps_h} ${native_sparse_mps_mm})

    if(CAN_COMPILE_METAL)
        foreach(SHADER ${native_mps_metal} ${native_mps_sparse_metal})
            cmake_path(GET SHADER STEM TGT_STEM)
            string(CONCAT TGT_BASIC ${TGT_STEM} "_31.air")
            list(APPEND AIR_BASIC ${TGT_BASIC})
            metal_to_air(${SHADER} ${TGT_BASIC} "-std=metal3.1")
        endforeach()
        air_to_metallib(kernels_basic.metallib ${AIR_BASIC})
        set(METALLIB_DEPS kernels_basic.metallib)
        if(CAN_COMPILE_METAL_40)
            foreach(SHADER ${native_mps_metal} ${native_mps_sparse_metal})
                cmake_path(GET SHADER STEM TGT_STEM)
                string(CONCAT TGT_40 ${TGT_STEM} "_40.air")
                list(APPEND AIR_40 ${TGT_40})
                metal_to_air(${SHADER} ${TGT_40} "-std=metal4.0")
            endforeach()
            air_to_metallib(kernels_40.metallib ${AIR_40})
            list(APPEND METALLIB_DEPS kernels_40.metallib)
        endif()
        add_custom_command(
                          COMMAND echo "// $$(date)" > metallib_dummy.cpp
                          DEPENDS ${METALLIB_DEPS}
                          OUTPUT metallib_dummy.cpp
                          COMMENT "Updating metallibs timestamp")
        add_custom_target(metallibs DEPENDS ${METALLIB_DEPS} metallib_dummy.cpp)
    else()
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/native/mps")
        foreach(SHADER ${native_mps_metal} ${native_mps_sparse_metal})
            cmake_path(GET SHADER STEM TGT_STEM)
            string(CONCAT SHADER_HDR_NAME  "${CMAKE_CURRENT_BINARY_DIR}" /native/mps/ ${TGT_STEM} "_metallib.h")
            metal_to_metallib_h(${SHADER} ${SHADER_HDR_NAME})
        endforeach()
    endif()
endif()

if(USE_ROCM)
  set(ATen_HIP_SRCS ${all_hip_cpp})
  # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
  # See NOTE [ ATen NVRTC Stub and HIP ]
  set(ATen_NVRTC_STUB_SRCS ${hip_nvrtc_stub_cpp})
  # NB: Instead of adding it to this list, we add it by hand
  # to caffe2_hip, because it needs to be a PRIVATE dependency
  # list(APPEND ATen_HIP_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
endif()

set(ATEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_INCLUDE_DIR}")
configure_file(ATenConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake"
  DESTINATION "${AT_INSTALL_SHARE_DIR}/cmake/ATen")

set(INSTALL_HEADERS ${base_h} ${ATen_CORE_HEADERS} ${native_nested_h} ${ATen_TRANSFORMER_HEADERS})
if(NOT INTERN_BUILD_MOBILE)
  list(APPEND INSTALL_HEADERS ${native_h} ${native_cpu_h} ${native_ao_sparse_h} ${native_quantized_h} ${native_fhe_h} ${cuda_h} ${native_cuda_h} ${native_hip_h} ${native_mtia_h} ${cudnn_h} ${hip_h} ${mtia_h} ${xpu_h} ${mps_h} ${native_kleidiai_h} ${native_mps_h} ${native_utils_h} ${miopen_h})
  # Metal
  if(USE_PYTORCH_METAL_EXPORT)
    # Add files needed from exporting metal models(optimized_for_mobile)
    list(APPEND INSTALL_HEADERS ${metal_h} ${metal_prepack_h})
  elseif(APPLE AND USE_PYTORCH_METAL)
    # Needed by Metal kernels
    list(APPEND INSTALL_HEADERS ${metal_h} ${native_metal_h})
  else()
    list(APPEND INSTALL_HEADERS ${metal_h})
  endif()
else()
  if(IOS AND USE_PYTORCH_METAL)
      list(APPEND INSTALL_HEADERS ${metal_h} ${native_metal_h})
  else()
      list(APPEND INSTALL_HEADERS ${metal_h} ${metal_prepack_h})
  endif()
endif()

# https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
foreach(HEADER  ${INSTALL_HEADERS})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "ATen/" HEADER_SUB ${HEADER})
  string(REPLACE "${Torch_SOURCE_DIR}/" "" HEADER_SUB ${HEADER_SUB})
  get_filename_component(DIR ${HEADER_SUB} DIRECTORY)
  install(FILES ${HEADER} DESTINATION "${AT_INSTALL_INCLUDE_DIR}/${DIR}")
endforeach()

# TODO: Install hip_generated_headers when we have it
foreach(HEADER ${generated_headers} ${cuda_generated_headers})
  # NB: Assumed to be flat
  install(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
endforeach()

message("AT_INSTALL_INCLUDE_DIR ${AT_INSTALL_INCLUDE_DIR}/ATen/core")
foreach(HEADER ${core_generated_headers})
  message("core header install: ${HEADER}")
  install(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/core)
endforeach()

install(FILES ${ops_h} ${ops_generated_headers}
  DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/ops)
install(FILES ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)

set(ATen_MOBILE_BENCHMARK_SRCS)

# Pass source, includes, and libs to parent
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
set(ATen_XPU_SRCS ${ATen_XPU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS ${ATen_CUDA_CU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_CPP_SRCS ${ATen_CUDA_CPP_SRCS} PARENT_SCOPE)
set(ATen_CUDA_LINALG_SRCS ${ATen_CUDA_LINALG_SRCS} PARENT_SCOPE)
set(ATen_CUDA_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_MPS_SRCS ${ATen_MPS_SRCS} PARENT_SCOPE)
set(ATen_MTIA_SRCS ${ATen_MTIA_SRCS} PARENT_SCOPE)
set(ATen_XPU_SRCS ${ATen_XPU_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_SRCS ${ATen_QUANTIZED_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_XPU_TEST_SRCS ${ATen_XPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_BENCHMARK_SRCS ${ATen_MOBILE_BENCHMARK_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_TEST_SRCS ${ATen_MOBILE_TEST_SRCS} ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_VEC_TEST_SRCS  ${ATen_VEC_TEST_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_TEST_SRCS ${ATen_QUANTIZED_TEST_SRCS} PARENT_SCOPE)
set(ATen_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_XPU_INCLUDE ${ATen_XPU_INCLUDE} PARENT_SCOPE)
set(ATen_VULKAN_INCLUDE ${ATen_VULKAN_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_XPU_DEPENDENCY_LIBS ${ATen_XPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
