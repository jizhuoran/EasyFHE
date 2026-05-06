#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201703L
#error C++17 or later compatible compiler is required to use PyTorch.
#endif

#include <torch/autograd.h>
#include <torch/cuda.h>
#include <torch/enum.h>
#include <torch/jit.h>
#include <torch/mps.h>
#include <torch/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/version.h>
#include <torch/xpu.h>
