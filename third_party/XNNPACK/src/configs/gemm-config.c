// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/gemm.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/igemm.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"

#if XNN_ARCH_WASMSIMD
  #include <emscripten.h>
#endif

#define XNN_MR_TO_INDEX(MR) (MR-1)


static struct xnn_gemm_config f16_gemm_config = {0};
static struct xnn_gemm_config f32_gemm_config = {0};
static struct xnn_gemm_config f32_gemm_nr2_config = {0};
static struct xnn_gemm_config f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config f32_qc8w_gemm_config = {0};
static struct xnn_gemm_config pf32_gemm_config = {0};
static struct xnn_gemm_config qd8_f16_qb4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f16_qc4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f16_qc8w_gemm_config = {0};
static struct xnn_gemm_config qd8_f32_qb4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config qd8_f32_qc8w_gemm_config = {0};
static struct xnn_gemm_config qp8_f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config qp8_f32_qb4w_gemm_config = {0};
static struct xnn_gemm_config qdu8_f32_qc4w_gemm_config = {0};
static struct xnn_gemm_config qdu8_f16_qc8w_gemm_config = {0};
static struct xnn_gemm_config qdu8_f32_qc8w_gemm_config = {0};
static struct xnn_gemm_config qdu8_f32_qb4w_gemm_config = {0};
static struct xnn_gemm_config qdu8_f16_qc4w_gemm_config = {0};
static struct xnn_gemm_config qs8_qc8w_gemm_config = {0};
static struct xnn_gemm_config qu8_gemm_config = {0};

XNN_INIT_ONCE_GUARD(f16_gemm);
XNN_INIT_ONCE_GUARD(f32_gemm);
XNN_INIT_ONCE_GUARD(f32_gemm_nr2);
XNN_INIT_ONCE_GUARD(f32_qc4w_gemm);
XNN_INIT_ONCE_GUARD(f32_qc8w_gemm);
XNN_INIT_ONCE_GUARD(pf32_gemm);
XNN_INIT_ONCE_GUARD(qd8_f16_qb4w_gemm);
XNN_INIT_ONCE_GUARD(qd8_f16_qc4w_gemm);
XNN_INIT_ONCE_GUARD(qd8_f16_qc8w_gemm);
XNN_INIT_ONCE_GUARD(qd8_f32_qb4w_gemm);
XNN_INIT_ONCE_GUARD(qd8_f32_qc4w_gemm);
XNN_INIT_ONCE_GUARD(qd8_f32_qc8w_gemm);
XNN_INIT_ONCE_GUARD(qp8_f32_qc4w_gemm);
XNN_INIT_ONCE_GUARD(qp8_f32_qb4w_gemm);
XNN_INIT_ONCE_GUARD(qdu8_f32_qc4w_gemm);
XNN_INIT_ONCE_GUARD(qdu8_f16_qc8w_gemm);
XNN_INIT_ONCE_GUARD(qdu8_f32_qc8w_gemm);
XNN_INIT_ONCE_GUARD(qdu8_f32_qb4w_gemm);
XNN_INIT_ONCE_GUARD(qdu8_f16_qc4w_gemm);
XNN_INIT_ONCE_GUARD(qs8_qc8w_gemm);
XNN_INIT_ONCE_GUARD(qu8_gemm);

static void init_f16_gemm_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm;
      f16_gemm_config.mr = 6;
      f16_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a55:
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
            f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
            f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
            f16_gemm_config.mr = 6;
            f16_gemm_config.nr = 16;
            break;
          case cpuinfo_uarch_cortex_a55r0:
          case cpuinfo_uarch_cortex_a75:
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
            f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
            f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
            f16_gemm_config.mr = 6;
            f16_gemm_config.nr = 16;
            break;
          case cpuinfo_uarch_exynos_m5:
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
            f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
            f16_gemm_config.mr = 4;
            f16_gemm_config.nr = 16;
            break;
          case cpuinfo_uarch_exynos_m4:
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
            f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
            f16_gemm_config.mr = 6;
            f16_gemm_config.nr = 16;
            break;
          default:
          case cpuinfo_uarch_cortex_a76:
          case cpuinfo_uarch_cortex_a77:
          case cpuinfo_uarch_cortex_a78:
          case cpuinfo_uarch_cortex_a510:
          case cpuinfo_uarch_cortex_a710:
          case cpuinfo_uarch_cortex_a715:
          case cpuinfo_uarch_cortex_x1:
          case cpuinfo_uarch_cortex_x2:
          case cpuinfo_uarch_cortex_x3:
          case cpuinfo_uarch_neoverse_n1:
          case cpuinfo_uarch_neoverse_n2:
          case cpuinfo_uarch_neoverse_v1:
          case cpuinfo_uarch_neoverse_v2:
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
            f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
            f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
            f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
            f16_gemm_config.mr = 6;
            f16_gemm_config.nr = 16;
            break;
        }

        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = f16_gemm_config.mr;
          const uint32_t nr = f16_gemm_config.nr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                if (mr == 6 && nr == 16) {
                  f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55;
                  f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
              case cpuinfo_uarch_cortex_a75:
                if (mr == 6 && nr == 16) {
                  f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0;
                  f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64);
        f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64);
        f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64);
        f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64);
        f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
        f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm;
        f16_gemm_config.mr = 6;
        f16_gemm_config.nr = 16;
      #endif  // XNN_ENABLE_ASSEMBLY
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast);
        f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast);
        f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_1x64__avx512fp16_broadcast);
        f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_igemm_minmax_ukernel_7x64__avx512fp16_broadcast);
        f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
        f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4;
        f16_gemm_config.mr = 7;
        f16_gemm_config.nr = 64;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_f32acc_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f16_f32acc_gemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_f32acc_igemm_minmax_ukernel_1x16__avx2_broadcast);
      f16_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f16_f32acc_igemm_minmax_ukernel_4x16__avx2_broadcast);
      f16_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      f16_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f16_gemm_gio_w;
      f16_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm;
      f16_gemm_config.mr = 4;
      f16_gemm_config.nr = 16;
    }
  #endif
}

#if XNN_ARCH_WASMSIMD
  EM_JS(int, hardware_concurrency, (void), {
    var concurrency = 1;
    try {
      concurrency = self.navigator.hardwareConcurrency;
    } catch(e) {
      // d8 environment doesn't provide `self`, thus we keep the default
    }
    return concurrency;
  });
  // A cpu with more than `kCoreCountThresholdForAdaptiveAvxOptimization` is
  // assumed to support AVX instructions.
  const int kCoreCountThresholdForAdaptiveAvxOptimization = 4;
#endif

static void init_pf32_gemm_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if (XNN_ENABLE_ARM_SME2 && hardware_config->use_arm_sme2) {
    #if XNN_ENABLE_ARM_SME2
      const size_t mr = xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr();
      const size_t nr = xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr();
      pf32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(nr)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2);
      pf32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      pf32_gemm_config.pack_weights_and_biases = xnn_pack_kai_f32_weights_and_biases;
      pf32_gemm_config.packed_stride_weights_and_biases = xnn_packed_stride_kai_f32_weights_and_biases;
      pf32_gemm_config.mr = mr;
      pf32_gemm_config.mr_packed = mr;
      pf32_gemm_config.nr = nr;
    #endif  // XNN_ENABLE_ARM_SME2
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

static void init_f32_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_uarch(0)->uarch) {
          case cpuinfo_uarch_cortex_a5:
          case cpuinfo_uarch_cortex_a7:
          case cpuinfo_uarch_krait:
          case cpuinfo_uarch_kryo:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a53:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a32:
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
          case cpuinfo_uarch_cortex_a73:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;

          default:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = f32_gemm_config.mr;
          const uint32_t nr = f32_gemm_config.nr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53;
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 4;
        f32_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
        switch (cpuinfo_get_core(0)->uarch) {
          case cpuinfo_uarch_cortex_a72:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a75:
          case cpuinfo_uarch_cortex_a76:
          case cpuinfo_uarch_exynos_m3:
          case cpuinfo_uarch_exynos_m4:
          case cpuinfo_uarch_neoverse_n1:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_exynos_m1:
          case cpuinfo_uarch_exynos_m2:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            f32_gemm_config.log2_sr = 2;
            break;
          case cpuinfo_uarch_cortex_a53:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a35:
          case cpuinfo_uarch_cortex_a55:
          case cpuinfo_uarch_kryo:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a73:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a77:
          case cpuinfo_uarch_exynos_m5:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 4;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_x3:
          case cpuinfo_uarch_neoverse_v2:
            // TODO(fbarchard): Implement asm with indexed inputs
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a78:
          case cpuinfo_uarch_cortex_a510:
          case cpuinfo_uarch_cortex_a710:
          case cpuinfo_uarch_cortex_a715:
          case cpuinfo_uarch_cortex_x1:
          case cpuinfo_uarch_cortex_x2:
          case cpuinfo_uarch_neoverse_n2:
          case cpuinfo_uarch_neoverse_v1:
          default:
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
            f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
            f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
            #if XNN_ENABLE_GEMM_M_SPECIALIZATION
              f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
              f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
            #endif
            f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
            f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
            f32_gemm_config.mr = 6;
            f32_gemm_config.nr = 8;
            break;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = f32_gemm_config.mr;
          const uint32_t nr = f32_gemm_config.nr;
          const uint32_t log2_sr = f32_gemm_config.log2_sr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  #endif
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                  #endif
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53;
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                if (mr == 6 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55;
                  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
                    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                  #endif
                } else if (mr == 4 && nr == 8 && log2_sr == 0) {
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53;
                  f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55;
                }
                break;
              default:
                break;
            }
          }
        }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x32__avx512f_broadcast);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x32__avx512f_broadcast);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_7x32__avx512f_broadcast);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_x32_packw_gemm_gio_ukernel_x32__avx512f_u8;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4_prfm;
        f32_gemm_config.mr = 7;
        f32_gemm_config.nr = 32;
      } else
    #endif
    if (hardware_config->use_x86_fma3) {
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_zen:
        case cpuinfo_uarch_dhyana:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4;
          f32_gemm_config.mr = 4;
          f32_gemm_config.nr = 16;
          f32_gemm_config.log2_sr = 2;
          break;
        default:
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast_prfm);
          f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_x32_packw_gemm_gio_ukernel_x16__avx_u8;
          f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4;
          f32_gemm_config.mr = 5;
          f32_gemm_config.nr = 16;
          break;
      }
    } else if (hardware_config->use_x86_avx) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_x32_packw_gemm_gio_ukernel_x16__avx_u8;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4;
      f32_gemm_config.mr = 5;
      f32_gemm_config.nr = 16;
    } else {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__sse_load1);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__sse_load1);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
      #else
        if (hardware_concurrency() > kCoreCountThresholdForAdaptiveAvxOptimization) {
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat);
        } else {
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
          f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
          f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat);
        }
      #endif

      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 8;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
        f32_gemm_config.mr = 6;
        f32_gemm_config.nr = 8;
      #else
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat);
        f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4;
        f32_gemm_config.mr = 5;
        f32_gemm_config.nr = 8;
      #endif
    }
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_2x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_2x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_2x4__scalar);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_2x4__scalar);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 2;
      f32_gemm_config.nr = 4;
    } else {
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
      f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__wasm);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__wasm);
      f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__wasm);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
      f32_gemm_config.mr = 4;
      f32_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_riscv_vector) {
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_7x4v__rvv);
      f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4v__rvv);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_7x4v__rvv);
      f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4v__rvv);
      f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8;
      f32_gemm_config.mr = 7;
      // nr is set to vlen * 4 / sizeof(float) = 4 * VLENB * 8 / 32 = VLENB
      f32_gemm_config.nr = hardware_config->vlenb;
      return;
    }
  #else
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_1x4__scalar);
    f32_gemm_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_1x4__scalar);
    f32_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_1x4__scalar);
    f32_gemm_config.relu.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_relu_ukernel_4x4__scalar);
    f32_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4;
    f32_gemm_config.mr = 4;
    f32_gemm_config.nr = 4;
  #endif
}

static void init_f32_gemm_nr2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
      f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128);
      f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm);
      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm);
        f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
        f32_gemm_nr2_config.mr = 4;
        f32_gemm_nr2_config.nr = 2;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64);
        f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
        f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm;
        f32_gemm_nr2_config.mr = 4;
        f32_gemm_nr2_config.nr = 2;
       #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__sse);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__sse);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
    f32_gemm_nr2_config.log2_kr = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86);
      #endif

      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
      f32_gemm_nr2_config.log2_kr = 2;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma);
      #else
        f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2c4__wasmsimd);
        f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm);
        f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm);
      #endif

      f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
      f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4;
      f32_gemm_nr2_config.mr = 4;
      f32_gemm_nr2_config.nr = 2;
      f32_gemm_nr2_config.log2_kr = 2;
    }
  #elif XNN_ARCH_WASM
    f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__wasm);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__wasm);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
  #else
    f32_gemm_nr2_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.linear.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_gemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_f32_igemm_minmax_ukernel_4x2__scalar);
    f32_gemm_nr2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_gemm_nr2_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_gemm_gio_w;
    f32_gemm_nr2_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4;
    f32_gemm_nr2_config.mr = 4;
    f32_gemm_nr2_config.nr = 2;
  #endif
}

static void init_f32_qc4w_gemm_config(void) {
    f32_qc4w_gemm_config.planes = 1;
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_lane_ld64);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_lane_ld64);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 6;
    f32_qc4w_gemm_config.nr = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast);
        f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast);
        f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
        f32_qc4w_gemm_config.mr = 7;
        f32_qc4w_gemm_config.nr = 32;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx2_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx_broadcast);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 3;
      f32_qc4w_gemm_config.nr = 16;
    } else {
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup);
      f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup);
      f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
      f32_qc4w_gemm_config.mr = 4;
      f32_qc4w_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #elif XNN_ARCH_WASM
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #else
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc4w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_qc4w_gemm_goi_w;
    f32_qc4w_gemm_config.mr = 4;
    f32_qc4w_gemm_config.nr = 4;
  #endif
}

static void init_f32_qc8w_gemm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 4;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      switch (cpuinfo_get_core(0)->uarch) {
        // TODO(fbarchard): fill in microkernels.
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55r0:
        case cpuinfo_uarch_cortex_a35:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
        case cpuinfo_uarch_cortex_a73:
        case cpuinfo_uarch_cortex_a77:
        case cpuinfo_uarch_exynos_m5:
        case cpuinfo_uarch_cortex_a78:
        case cpuinfo_uarch_cortex_x1:
        case cpuinfo_uarch_neoverse_n1:
        case cpuinfo_uarch_neoverse_v1:
        default:
          f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4);
          f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
          f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
          f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
          f32_qc8w_gemm_config.mr = 6;
          f32_qc8w_gemm_config.nr = 8;
      }
      #if XNN_MAX_UARCH_TYPES > 1
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = f32_gemm_config.mr;
        const uint32_t nr = f32_gemm_config.nr;
        const uint32_t log2_sr = f32_gemm_config.log2_sr;
        // TODO(fbarchard): fill in with microkernels.
        (void) mr;
        (void) nr;
        (void) log2_sr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }
          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
            default:
              break;
          }
        }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
      #else  // !XNN_ENABLE_ASSEMBLY
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_ENABLE_ASSEMBLY && !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x32__scalar_u2;
        f32_qc8w_gemm_config.mr = 7;
        f32_qc8w_gemm_config.nr = 32;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x16__scalar_u2;
      f32_qc8w_gemm_config.mr = 5;
      f32_qc8w_gemm_config.nr = 16;
    } else {
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup);
      f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup);
      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat);
      #else
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_splat);
      #endif

      f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
      f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
      f32_qc8w_gemm_config.mr = 4;
      f32_qc8w_gemm_config.nr = 8;
    } else {
      #if XNN_ARCH_WASMRELAXEDSIMD
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat);
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
        f32_qc8w_gemm_config.mr = 6;
        f32_qc8w_gemm_config.nr = 8;
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.linear.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat);
        f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat);
        f32_qc8w_gemm_config.relu.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_splat);
      #else
        f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
        f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2;
        f32_qc8w_gemm_config.mr = 5;
        f32_qc8w_gemm_config.nr = 8;
      #endif
    }
  #elif XNN_ARCH_WASM
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm);
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm);
    f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
    f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u2;
    f32_qc8w_gemm_config.mr = 4;
    f32_qc8w_gemm_config.nr = 4;
  #else
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    f32_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_f32_qs8w_gemm_gio_w;
    f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_x8_packw_gemm_goi_ukernel_x4__scalar_u2;
    f32_qc8w_gemm_config.mr = 4;
    f32_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qdu8_f16_qc4w_gemm_config(void) {
  // Use the same packing function throughout.
  qdu8_f16_qc4w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs4_weights_and_biases;
  qdu8_f16_qc4w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs4_weights_and_biases;
  qdu8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
  qdu8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
  #if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX256VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256vnni) {
        qdu8_f16_qc4w_gemm_config.arch = xnn_arch_x86_avx256vnni;
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni);
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni);
        qdu8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qdu8_f16_qc4w_gemm_config.mr = 8;
        qdu8_f16_qc4w_gemm_config.nr = 8;
        qdu8_f16_qc4w_gemm_config.log2_kr = 3;
        qdu8_f16_qc4w_gemm_config.planes = 2;
      } else
    #endif
    #if XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qdu8_f16_qc4w_gemm_config.arch = xnn_arch_x86_avxvnni;
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qdu8_f16_qc4w_gemm_config.mr = 5;
        qdu8_f16_qc4w_gemm_config.nr = 8;
        qdu8_f16_qc4w_gemm_config.log2_kr = 3;
        qdu8_f16_qc4w_gemm_config.planes = 2;
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
        qdu8_f16_qc4w_gemm_config.arch = xnn_arch_x86_avx256skx;
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm);
        qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm);
        qdu8_f16_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
        qdu8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
        qdu8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qdu8_f16_qc4w_gemm_config.mr = 8;
        qdu8_f16_qc4w_gemm_config.nr = 8;
        qdu8_f16_qc4w_gemm_config.log2_kr = 3;
        qdu8_f16_qc4w_gemm_config.planes = 2;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
        qdu8_f16_qc4w_gemm_config.arch = xnn_arch_x86_avx2;
      qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm);
      qdu8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm);
      qdu8_f16_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f16_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
      qdu8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
      qdu8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
      qdu8_f16_qc4w_gemm_config.mr = 4;
      qdu8_f16_qc4w_gemm_config.nr = 8;
      qdu8_f16_qc4w_gemm_config.log2_kr = 3;
      qdu8_f16_qc4w_gemm_config.planes = 2;
    }
  #endif
}

static void init_qd8_f16_qc4w_gemm_config(void) {
  // Use the same packing function throughout.
  qd8_f16_qc4w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs4_weights_and_biases;
  qd8_f16_qc4w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs4_weights_and_biases;
  qd8_f16_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;
  qd8_f16_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
          qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
          qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
          qd8_f16_qc4w_gemm_config.mr = 4;
          qd8_f16_qc4w_gemm_config.nr = 16;
          qd8_f16_qc4w_gemm_config.log2_kr = 2;
          qd8_f16_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.mr = 6;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.planes = 2;
      }
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.mr = 4;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.log2_kr = 3;
        qd8_f16_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.mr = 4;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.log2_kr = 2;
        qd8_f16_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qc4w_gemm_config.init.f16_qc4w = xnn_init_f16_qc4w_minmax_scalar_params;
        qd8_f16_qc4w_gemm_config.mr = 6;
        qd8_f16_qc4w_gemm_config.nr = 16;
        qd8_f16_qc4w_gemm_config.planes = 2;
    }
  #endif
}

static void init_qd8_f16_qb4w_gemm_config(void) {
  qd8_f16_qb4w_gemm_config.pack_gemm_goi_bl = (xnn_packw_gemm_goi_bl_ukernel_fn) xnn_pack_qs8_qb4w_gemm_goi_w;

  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
          qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
          qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
          qd8_f16_qb4w_gemm_config.mr = 4;
          qd8_f16_qb4w_gemm_config.nr = 16;
          qd8_f16_qb4w_gemm_config.log2_kr = 2;
          qd8_f16_qb4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else if (hardware_config->use_arm_neon_fp16_arith) {
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
        qd8_f16_qb4w_gemm_config.mr = 6;
        qd8_f16_qb4w_gemm_config.nr = 16;
        qd8_f16_qb4w_gemm_config.planes = 2;
      }
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
        qd8_f16_qb4w_gemm_config.mr = 4;
        qd8_f16_qb4w_gemm_config.nr = 16;
        qd8_f16_qb4w_gemm_config.log2_kr = 3;
        qd8_f16_qb4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
        qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
        qd8_f16_qb4w_gemm_config.mr = 4;
        qd8_f16_qb4w_gemm_config.nr = 16;
        qd8_f16_qb4w_gemm_config.log2_kr = 2;
        qd8_f16_qb4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else if (hardware_config->use_arm_neon_fp16_arith) {
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane);
        qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane);
        qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
        qd8_f16_qb4w_gemm_config.mr = 6;
        qd8_f16_qb4w_gemm_config.nr = 16;
        qd8_f16_qb4w_gemm_config.planes = 2;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qb4w_gemm_config.init.f16_qb4w = xnn_init_f16_qb4w_minmax_scalar_params;
      qd8_f16_qb4w_gemm_config.mr = 3;
      qd8_f16_qb4w_gemm_config.nr = 8;
      qd8_f16_qb4w_gemm_config.log2_kr = 3;
      qd8_f16_qb4w_gemm_config.planes = 2;
    }
  #endif
}

static void init_qd8_f32_qc4w_gemm_config(void) {
  // Use the same packing function throughout.
  qd8_f32_qc4w_gemm_config.pack_weights_and_biases = (xnn_pack_weights_and_biases_fn) xnn_pack_qs4_weights_and_biases;
  qd8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = (xnn_packed_stride_weights_and_biases_fn) xnn_packed_stride_qs4_weights_and_biases;
  qd8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;  // Ignored
  qd8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;  // Ignored
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot);
          qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot);
          qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
          qd8_f32_qc4w_gemm_config.mr = 4;
          qd8_f32_qc4w_gemm_config.nr = 16;
          qd8_f32_qc4w_gemm_config.log2_kr = 2;
          qd8_f32_qc4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.mr = 6;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.planes = 2;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.mr = 4;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.log2_kr = 3;
        qd8_f32_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.mr = 4;
        qd8_f32_qc4w_gemm_config.nr = 16;
        qd8_f32_qc4w_gemm_config.log2_kr = 2;
        qd8_f32_qc4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.mr = 6;
      qd8_f32_qc4w_gemm_config.nr = 16;
      qd8_f32_qc4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    #if XNN_ENABLE_AVX512AMX
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512amx) {
        qd8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx512amx;
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx);
        qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx);
        qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qd8_f32_qc4w_gemm_config.mr = 16;
        qd8_f32_qc4w_gemm_config.nr = 64;
        qd8_f32_qc4w_gemm_config.log2_kr = 2;
        qd8_f32_qc4w_gemm_config.planes = 2;
      } else
    #endif  // XNN_ENABLE_AVX512AMX
    {
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128);
      qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128);
      qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qd8_f32_qc4w_gemm_config.mr = 4;
      qd8_f32_qc4w_gemm_config.nr = 4;
      qd8_f32_qc4w_gemm_config.log2_kr = 3;
      qd8_f32_qc4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.log2_kr = 3;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #elif XNN_ARCH_WASM
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #else
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
    qd8_f32_qc4w_gemm_config.mr = 4;
    qd8_f32_qc4w_gemm_config.nr = 4;
    qd8_f32_qc4w_gemm_config.planes = 2;
  #endif
}

static void init_qp8_f32_qc4w_gemm_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  assert(hardware_config != NULL);
  if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
#if XNN_ENABLE_ARM_I8MM
    qp8_f32_qc4w_gemm_config.minmax.qp8gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_qp8gemm_ukernel(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot);
    qp8_f32_qc4w_gemm_config.minmax.qp8gemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_qp8gemm_ukernel(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2);
    qp8_f32_qc4w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qp8_f32_qc4w_gemm_config.pack_weights_and_biases = xnn_pack_kai_qs4_weights_and_biases;
    qp8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = xnn_packed_stride_kai_qs4_weights_and_biases;
    qp8_f32_qc4w_gemm_config.mr = 8;
    qp8_f32_qc4w_gemm_config.nr = 8;
    qp8_f32_qc4w_gemm_config.log2_kr = 4;
    qp8_f32_qc4w_gemm_config.log2_sr = 1;
    qp8_f32_qc4w_gemm_config.planes = 2;
    qp8_f32_qc4w_gemm_config.mr_packed = 4;
#endif  // XNN_ENABLE_ARM_I8MM
  } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
#if XNN_ENABLE_ARM_DOTPROD
    qp8_f32_qc4w_gemm_config.minmax.qp8gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_qp8gemm_ukernel(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot);
    qp8_f32_qc4w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qp8_f32_qc4w_gemm_config.pack_weights_and_biases = xnn_pack_kai_qs4_weights_and_biases;
    qp8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = xnn_packed_stride_kai_qs4_weights_and_biases;
    qp8_f32_qc4w_gemm_config.mr = 1;
    qp8_f32_qc4w_gemm_config.nr = 8;
    qp8_f32_qc4w_gemm_config.log2_kr = 4;
    qp8_f32_qc4w_gemm_config.log2_sr = 1;
    qp8_f32_qc4w_gemm_config.planes = 2;
    qp8_f32_qc4w_gemm_config.mr_packed = 1;
#endif  // XNN_ENABLE_ARM_DOTPROD
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

static void init_qp8_f32_qb4w_gemm_config(void) {
  #if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
    const struct xnn_hardware_config* hardware_config =
        xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qp8_f32_qb4w_gemm_config.minmax.qp8gemm_bl[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_qp8gemm_bl_ukernel(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot);
        qp8_f32_qb4w_gemm_config.minmax.qp8gemm_bl[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_qp8gemm_bl_ukernel(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_16x4c16s2__neoni8mm_mstep4);
        qp8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qp8_f32_qb4w_gemm_config.pack_weights_and_biases = xnn_pack_kai_qb4_weights_and_biases;
        qp8_f32_qb4w_gemm_config.packed_stride_weights_and_biases = xnn_packed_stride_kai_qb4_weights_and_biases;
        qp8_f32_qb4w_gemm_config.mr = 16;
        qp8_f32_qb4w_gemm_config.nr = 4;
        qp8_f32_qb4w_gemm_config.log2_kr = 4;
        qp8_f32_qb4w_gemm_config.log2_sr = 1;
        qp8_f32_qb4w_gemm_config.planes = 2;
        qp8_f32_qb4w_gemm_config.mr_packed = 4;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qp8_f32_qb4w_gemm_config.minmax.qp8gemm_bl[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_qp8gemm_bl_ukernel(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot);
        qp8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qp8_f32_qb4w_gemm_config.pack_weights_and_biases = xnn_pack_kai_qb4_weights_and_biases;
        qp8_f32_qb4w_gemm_config.packed_stride_weights_and_biases = xnn_packed_stride_kai_qb4_weights_and_biases;
        qp8_f32_qb4w_gemm_config.mr = 1;
        qp8_f32_qb4w_gemm_config.nr = 8;
        qp8_f32_qb4w_gemm_config.log2_kr = 4;
        qp8_f32_qb4w_gemm_config.log2_sr = 1;
        qp8_f32_qb4w_gemm_config.planes = 2;
        qp8_f32_qb4w_gemm_config.mr_packed = 1;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    }
  #endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
}

static void init_qdu8_f32_qb4w_gemm_config(void) {
  qdu8_f32_qb4w_gemm_config.pack_gemm_goi_bl = (xnn_packw_gemm_goi_bl_ukernel_fn) xnn_pack_qs8_qb4w_gemm_goi_w;
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512VNNIGFNI
      // Zen4 has gfni but is slower and 8x16 works better on zen4.  14x16 is faster on Sapphire Rapids
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnnigfni && cpuinfo_get_core(0)->uarch != cpuinfo_uarch_zen4) {
        qdu8_f32_qb4w_gemm_config.arch = xnn_arch_x86_avx512vnnigfni;
        qdu8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm);
        qdu8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(14)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm);
        qdu8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qdu8_f32_qb4w_gemm_config.mr = 14;
        qdu8_f32_qb4w_gemm_config.nr = 16;
        qdu8_f32_qb4w_gemm_config.log2_kr = 3;
        qdu8_f32_qb4w_gemm_config.planes = 2;
      } else
    #endif  // XNN_ENABLE_AVX512VNNIGFNI
    #if XNN_ENABLE_AVX512VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
        qdu8_f32_qb4w_gemm_config.arch = xnn_arch_x86_avx512vnni;
        qdu8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
        qdu8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm);
        qdu8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qdu8_f32_qb4w_gemm_config.mr = 8;
        qdu8_f32_qb4w_gemm_config.nr = 16;
        qdu8_f32_qb4w_gemm_config.log2_kr = 3;
        qdu8_f32_qb4w_gemm_config.planes = 2;
      }
    #else
      {}
    #endif
  #endif //XNN_ARCH_X86 || XNN_ARCH_X86_64
}

static void init_qd8_f32_qb4w_gemm_config(void) {
  qd8_f32_qb4w_gemm_config.pack_gemm_goi_bl = (xnn_packw_gemm_goi_bl_ukernel_fn) xnn_pack_qs8_qb4w_gemm_goi_w;

  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
        #if XNN_ENABLE_ARM_DOTPROD
          qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot);
          qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot);
          qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
          qd8_f32_qb4w_gemm_config.mr = 4;
          qd8_f32_qb4w_gemm_config.nr = 16;
          qd8_f32_qb4w_gemm_config.log2_kr = 2;
          qd8_f32_qb4w_gemm_config.planes = 2;
        #endif  // XNN_ENABLE_ARM_DOTPROD
      } else {
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
        qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qd8_f32_qb4w_gemm_config.mr = 6;
        qd8_f32_qb4w_gemm_config.nr = 16;
        qd8_f32_qb4w_gemm_config.planes = 2;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 4;
      qd8_f32_qb4w_gemm_config.nr = 4;
      qd8_f32_qb4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
      #if XNN_ENABLE_ARM_I8MM
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm);
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm);
        qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qd8_f32_qb4w_gemm_config.mr = 4;
        qd8_f32_qb4w_gemm_config.nr = 16;
        qd8_f32_qb4w_gemm_config.log2_kr = 3;
        qd8_f32_qb4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_I8MM
    } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
      #if XNN_ENABLE_ARM_DOTPROD
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot);
        qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot);
        qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
        qd8_f32_qb4w_gemm_config.mr = 4;
        qd8_f32_qb4w_gemm_config.nr = 16;
        qd8_f32_qb4w_gemm_config.log2_kr = 2;
        qd8_f32_qb4w_gemm_config.planes = 2;
      #endif  // XNN_ENABLE_ARM_DOTPROD
    } else {
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(6)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 6;
      qd8_f32_qb4w_gemm_config.nr = 16;
      qd8_f32_qb4w_gemm_config.planes = 2;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qd8_f32_qb4w_gemm_config.arch = xnn_arch_x86_avx2;
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 3;
      qd8_f32_qb4w_gemm_config.nr = 8;
      qd8_f32_qb4w_gemm_config.log2_kr = 3;
      qd8_f32_qb4w_gemm_config.planes = 2;
    } else if (hardware_config->use_x86_avx) {
      qd8_f32_qb4w_gemm_config.arch = xnn_arch_x86_avx;
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 4;
      qd8_f32_qb4w_gemm_config.nr = 4;
      qd8_f32_qb4w_gemm_config.log2_kr = 3;
      qd8_f32_qb4w_gemm_config.planes = 1;
    } else if (hardware_config->use_x86_sse4_1) {
      qd8_f32_qb4w_gemm_config.arch = xnn_arch_x86_sse4_1;
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 3;
      qd8_f32_qb4w_gemm_config.nr = 4;
      qd8_f32_qb4w_gemm_config.log2_kr = 3;
      qd8_f32_qb4w_gemm_config.planes = 1;
    } else {
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128);
      qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128);
      qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
      qd8_f32_qb4w_gemm_config.mr = 4;
      qd8_f32_qb4w_gemm_config.nr = 4;
      qd8_f32_qb4w_gemm_config.log2_kr = 3;
      qd8_f32_qb4w_gemm_config.planes = 1;
    }
  #else
    qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qb4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qb4w_gemm_config.init.f32_qb4w = xnn_init_f32_qb4w_minmax_scalar_params;
    qd8_f32_qb4w_gemm_config.mr = 4;
    qd8_f32_qb4w_gemm_config.nr = 4;
    qd8_f32_qb4w_gemm_config.planes = 2;
  #endif
}

static void init_qd8_f16_qc8w_gemm_config(void) {
  // Use the same packing function throughout.
  qd8_f16_qc8w_gemm_config.pack_weights_and_biases = (xnn_pack_weights_and_biases_fn)xnn_pack_qs8_weights_and_biases;
  qd8_f16_qc8w_gemm_config.packed_stride_weights_and_biases = (xnn_packed_stride_weights_and_biases_fn) xnn_packed_stride_qs8_weights_and_biases;
  qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
  qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 8;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f16_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f16_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f16_qc8w_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 8;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
           switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55);
                qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55);
                break;
              default:
                qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
                qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128);
                break;
            }
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f16_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f16_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f16_qc8w_gemm_config.log2_kr;
          // Avoid unused warnings.
          (void) mr;
          (void) nr;
          (void) log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith;
                    qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__neondotfp16arith);
            qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
            qd8_f16_qc8w_gemm_config.mr = 4;
            qd8_f16_qc8w_gemm_config.nr = 16;
            qd8_f16_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_fp16_arith) {
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith);
          qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
          qd8_f16_qc8w_gemm_config.mr = 2;
          qd8_f16_qc8w_gemm_config.nr = 8;
          qd8_f16_qc8w_gemm_config.log2_kr = 1;
          qd8_f16_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512AMX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512amx) {
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx);
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x64c4__avx512amx);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx);
        qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        qd8_f16_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qd8_f16_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qd8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qd8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar;
        qd8_f16_qc8w_gemm_config.mr = 16;
        qd8_f16_qc8w_gemm_config.nr = 64;
        qd8_f16_qc8w_gemm_config.log2_kr = 2;
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
        qd8_f16_qc8w_gemm_config.arch = xnn_arch_x86_avx256skx;
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx);
        qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx256skx);
        qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avx256skx);
        qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        qd8_f16_qc8w_gemm_config.mr = 5;
        qd8_f16_qc8w_gemm_config.nr = 8;
        qd8_f16_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qd8_f16_qc8w_gemm_config.arch = xnn_arch_x86_avx2;
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2);
      qd8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__avx2);
      qd8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
      qd8_f16_qc8w_gemm_config.mr = 3;
      qd8_f16_qc8w_gemm_config.nr = 8;
      qd8_f16_qc8w_gemm_config.log2_kr = 3;
    }
  #endif
}

static void init_qdu8_f16_qc8w_gemm_config(void) {
  // Use the same packing function throughout.
  qdu8_f16_qc8w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs8_weights_and_biases;
  qdu8_f16_qc8w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs8_weights_and_biases;
  qdu8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
  qdu8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    (void) hardware_config; // in case neither of the ifs are  compiled.
    #if XNN_ENABLE_AVX256VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256vnni) {
        qdu8_f16_qc8w_gemm_config.arch = xnn_arch_x86_avx256vnni;
        qdu8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni);
        qdu8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni);
        qdu8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni);
        qdu8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni);
        qdu8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        qdu8_f16_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qdu8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x8c8__avx256vnni_prfm;
        qdu8_f16_qc8w_gemm_config.mr = 8;
        qdu8_f16_qc8w_gemm_config.nr = 8;
        qdu8_f16_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        // AVX VNNI checked before AVX512SKX as it performs better with VNNI microkernels
        qdu8_f16_qc8w_gemm_config.arch = xnn_arch_x86_avxvnni;
        qdu8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f16_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f16_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f16_qc8w_gemm_config.init.f16 = xnn_init_f16_minmax_scalar_params;
        qdu8_f16_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f16_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qdu8_f16_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm;
        qdu8_f16_qc8w_gemm_config.mr = 5;
        qdu8_f16_qc8w_gemm_config.nr = 8;
        qdu8_f16_qc8w_gemm_config.log2_kr = 3;
      }
    #else
      {}
    #endif
  #endif //XNN_ARCH_X86 || XNN_ARCH_X86_64
}

static void init_qdu8_f32_qc8w_gemm_config(void) {
  // Use the same packing function throughout.
  qdu8_f32_qc8w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs8_weights_and_biases;
  qdu8_f32_qc8w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs8_weights_and_biases;
  qdu8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
  qdu8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    (void) hardware_config; // in case neither of the ifs are  compiled.
    #if XNN_ENABLE_AVX512VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
        qdu8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avx512vnni;
        qdu8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(10)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(10)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c8__avx512vnni_prfm);
        qdu8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qdu8_f32_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        #if XNN_ENABLE_AVX256VNNI
          qdu8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x16c8__avx256vnni_prfm;
        #else
          qdu8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x16c8__scalar;
        #endif
        qdu8_f32_qc8w_gemm_config.mr = 10;
        qdu8_f32_qc8w_gemm_config.nr = 16;
        qdu8_f32_qc8w_gemm_config.log2_kr = 3;
      }
    #endif
    #if XNN_ENABLE_AVXVNNI
     else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qdu8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avxvnni;
        qdu8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qdu8_f32_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qdu8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni_prfm;
        qdu8_f32_qc8w_gemm_config.mr = 5;
        qdu8_f32_qc8w_gemm_config.nr = 8;
        qdu8_f32_qc8w_gemm_config.log2_kr = 3;
      }
    #endif
  #endif //XNN_ARCH_X86 || XNN_ARCH_X86_64
}

static void init_qdu8_f32_qc4w_gemm_config(void) {
  // Use the same packing function throughout.
  qdu8_f32_qc4w_gemm_config.pack_weights_and_biases = (xnn_pack_weights_and_biases_fn) xnn_pack_qs4_weights_and_biases;
  qdu8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = (xnn_packed_stride_weights_and_biases_fn) xnn_packed_stride_qs4_weights_and_biases;
  qdu8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4w_gemm_gio_w;  // Ignored
  qdu8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4w_gemm_goi_w;  // Ignored
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512VNNIGFNI
      // Zen4 has gfni but is slower and 8x16 works better on zen4.  14x16 is faster on Sapphire Rapids
      // TODO(b/361288044): Re-enable once fixed.
      if (false && !XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnnigfni && cpuinfo_get_core(0)->uarch != cpuinfo_uarch_zen4) {
        qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx512vnnigfni;
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm);
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(14)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm);
        qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qdu8_f32_qc4w_gemm_config.mr = 14;
        qdu8_f32_qc4w_gemm_config.nr = 16;
        qdu8_f32_qc4w_gemm_config.log2_kr = 3;
        qdu8_f32_qc4w_gemm_config.planes = 2;
      } else
    #endif // XNN_ENABLE_AVX512VNNIGFNI
    #if XNN_ENABLE_AVX512VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
        qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx512vnni;
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm);
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm);
        qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qdu8_f32_qc4w_gemm_config.mr = 8;
        qdu8_f32_qc4w_gemm_config.nr = 16;
        qdu8_f32_qc4w_gemm_config.log2_kr = 3;
        qdu8_f32_qc4w_gemm_config.planes = 2;
      } else
    #endif
    #if XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avxvnni;
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm);
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm);
        qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qdu8_f32_qc4w_gemm_config.mr = 5;
        qdu8_f32_qc4w_gemm_config.nr = 8;
        qdu8_f32_qc4w_gemm_config.log2_kr = 3;
        qdu8_f32_qc4w_gemm_config.planes = 2;
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx512skx;
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd_prfm);
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm);
        qdu8_f32_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
        qdu8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
        qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qdu8_f32_qc4w_gemm_config.mr = 8;
        qdu8_f32_qc4w_gemm_config.nr = 16;
        qdu8_f32_qc4w_gemm_config.log2_kr = 3;
        qdu8_f32_qc4w_gemm_config.planes = 2;
    } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
        qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx256skx;
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm);
        qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm);
        qdu8_f32_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qdu8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
        qdu8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
        qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
        qdu8_f32_qc4w_gemm_config.mr = 8;
        qdu8_f32_qc4w_gemm_config.nr = 8;
        qdu8_f32_qc4w_gemm_config.log2_kr = 3;
        qdu8_f32_qc4w_gemm_config.planes = 2;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_avx2;
      qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm);
      qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm);
      qdu8_f32_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
      qdu8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
      qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qdu8_f32_qc4w_gemm_config.mr = 4;
      qdu8_f32_qc4w_gemm_config.nr = 8;
      qdu8_f32_qc4w_gemm_config.log2_kr = 3;
      qdu8_f32_qc4w_gemm_config.planes = 2;
    } else if (hardware_config->use_x86_ssse3) {
      qdu8_f32_qc4w_gemm_config.arch = xnn_arch_x86_ssse3;
      qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd);
      qdu8_f32_qc4w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd);
      qdu8_f32_qc4w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f32_qc4w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
      qdu8_f32_qc4w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_gio_w;
      qdu8_f32_qc4w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_qc4uw_gemm_goi_w;
      qdu8_f32_qc4w_gemm_config.init.f32_qc4w = xnn_init_f32_qc4w_minmax_scalar_params;
      qdu8_f32_qc4w_gemm_config.mr = 4;
      qdu8_f32_qc4w_gemm_config.nr = 4;
      qdu8_f32_qc4w_gemm_config.log2_kr = 3;
      qdu8_f32_qc4w_gemm_config.planes = 2;
    }
  #endif //XNN_ARCH_X86 || XNN_ARCH_X86_64
}

static void init_qd8_f32_qc8w_gemm_config(void) {
  // Use the same packing function throughout.
  qd8_f32_qc8w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs8_weights_and_biases;
  qd8_f32_qc8w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs8_weights_and_biases;
  qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
  qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        #if XNN_ENABLE_ARM_DOTPROD
          if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 8;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          } else
        #endif  // XNN_ENABLE_ARM_DOTPROD
        {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f32_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f32_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f32_qc8w_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 8;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.mr = 1;
      qd8_f32_qc8w_gemm_config.nr = 2;
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        #if XNN_ENABLE_ARM_I8MM
          if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          } else
        #endif  // XNN_ENABLE_ARM_I8MM
        #if XNN_ENABLE_ARM_DOTPROD
          if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
            switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              default:
                qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
            }
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          } else
        #endif  // XNN_ENABLE_ARM_DOTPROD
        {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qd8_f32_qc8w_gemm_config.mr;
          const uint32_t nr = qd8_f32_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qd8_f32_qc8w_gemm_config.log2_kr;
          // Avoid unused warnings.
          (void) mr;
          (void) nr;
          (void) log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot;
                    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot);
            qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot);
            qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
            qd8_f32_qc8w_gemm_config.mr = 4;
            qd8_f32_qc8w_gemm_config.nr = 16;
            qd8_f32_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal);
          qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
          qd8_f32_qc8w_gemm_config.mr = 2;
          qd8_f32_qc8w_gemm_config.nr = 8;
          qd8_f32_qc8w_gemm_config.log2_kr = 1;
          qd8_f32_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512AMX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512amx) {
        qd8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avx512amx;
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x64c4__avx512amx);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qd8_f32_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qd8_f32_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qd8_f32_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qd8_f32_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar;
        qd8_f32_qc8w_gemm_config.mr = 16;
        qd8_f32_qc8w_gemm_config.nr = 64;
        qd8_f32_qc8w_gemm_config.log2_kr = 2;
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        qd8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avx512skx;
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx_prfm);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx_prfm);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qd8_f32_qc8w_gemm_config.mr = 8;
        qd8_f32_qc8w_gemm_config.nr = 16;
        qd8_f32_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
        qd8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avx256skx;
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256skx);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(8)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256skx);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qd8_f32_qc8w_gemm_config.mr = 8;
        qd8_f32_qc8w_gemm_config.nr = 8;
        qd8_f32_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qd8_f32_qc8w_gemm_config.arch = xnn_arch_x86_avx2;
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avx2);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 8;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qd8_f32_qc8w_gemm_config.arch = xnn_arch_x86_sse4_1;
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse41_ld64);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld64);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_wasm_sdot) {
      if (hardware_config->is_x86) {
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmsdot);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__wasmsdot);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmsdot);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qd8_f32_qc8w_gemm_config.mr = 2;
        qd8_f32_qc8w_gemm_config.nr = 8;
        qd8_f32_qc8w_gemm_config.log2_kr = 3;
      } else {
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmsdot_u2);
        qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmsdot_u2);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot_u2);
        qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmsdot_u2);
        qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
        qd8_f32_qc8w_gemm_config.mr = 4;
        qd8_f32_qc8w_gemm_config.nr = 8;
        qd8_f32_qc8w_gemm_config.log2_kr = 3;
      }
    } else if (hardware_config->use_wasm_usdot) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmusdot_u2);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmusdot_u2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmusdot_u2);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmusdot_u2);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 8;
      qd8_f32_qc8w_gemm_config.log2_kr = 3;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
      qd8_f32_qc8w_gemm_config.log2_kr = 1;
      qd8_f32_qc8w_gemm_config.log2_sr = 2;
    }
  #elif XNN_ARCH_WASMSIMD
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qd8_f32_qc8w_gemm_config.mr = 4;
    qd8_f32_qc8w_gemm_config.nr = 4;
    qd8_f32_qc8w_gemm_config.log2_kr = 1;
    qd8_f32_qc8w_gemm_config.log2_sr = 2;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 2;
      qd8_f32_qc8w_gemm_config.nr = 2;
    } else {
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__wasm);
      qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__wasm);
      qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      qd8_f32_qc8w_gemm_config.mr = 4;
      qd8_f32_qc8w_gemm_config.nr = 4;
    }
  #else
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqgemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqgemm_ukernel((xnn_dqgemm_ukernel_fn) xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__scalar);
    qd8_f32_qc8w_gemm_config.minmax.dqigemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_dqigemm_ukernel((xnn_dqigemm_ukernel_fn) xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__scalar);
    qd8_f32_qc8w_gemm_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    qd8_f32_qc8w_gemm_config.mr = 4;
    qd8_f32_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qs8_qc8w_gemm_config(void) {
  // Use the same packing function throughout.
  qs8_qc8w_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qs8_weights_and_biases;
  qs8_qc8w_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qs8_weights_and_biases;
  qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
  qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_uarch(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
                qs8_qc8w_gemm_config.log2_kr = 2;
                break;
              default:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
                qs8_qc8w_gemm_config.log2_kr = 2;
                break;
            }
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_uarch(0)->uarch) {
            case cpuinfo_uarch_cortex_a5:
            case cpuinfo_uarch_cortex_a7:
            case cpuinfo_uarch_krait:
            case cpuinfo_uarch_kryo:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a32:
            case cpuinfo_uarch_cortex_a35:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a57:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;
            case cpuinfo_uarch_cortex_a72:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 1;
              qs8_qc8w_gemm_config.log2_sr = 2;
              break;
            case cpuinfo_uarch_exynos_m1:
            case cpuinfo_uarch_exynos_m2:
            case cpuinfo_uarch_exynos_m3:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 8;
              break;

            default:
              if (hardware_config->use_arm_neon_v8) {
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
              } else {
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
                qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
                qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
                qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
                qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
                qs8_qc8w_gemm_config.mr = 4;
                qs8_qc8w_gemm_config.nr = 8;
              }
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_qc8w_gemm_config.mr;
          const uint32_t nr = qs8_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qs8_qc8w_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 8 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot;
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55;
                    break;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53;
                }
                break;

              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 8;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else if (hardware_config->use_arm_neon_v8) {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 2;
      qs8_qc8w_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 3;
        }
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #else  // !XNN_PLATFORM_IOS && !XNN_PLATFORM_MAC
      #if XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            switch (cpuinfo_get_core(0)->uarch) {
              case cpuinfo_uarch_cortex_a55:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55);
                break;
              default:
                qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128);
                break;
            }
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          switch (cpuinfo_get_core(0)->uarch) {
            case cpuinfo_uarch_cortex_a35:
            case cpuinfo_uarch_kryo:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 4;
              qs8_qc8w_gemm_config.nr = 16;
              break;

            case cpuinfo_uarch_cortex_a72:
            case cpuinfo_uarch_cortex_a73:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 3;
              break;

            default:
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal);
              qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
              qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
              qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
              qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
              qs8_qc8w_gemm_config.mr = 2;
              qs8_qc8w_gemm_config.nr = 8;
              qs8_qc8w_gemm_config.log2_kr = 3;
              break;
          }
        }
        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qs8_qc8w_gemm_config.mr;
          const uint32_t nr = qs8_qc8w_gemm_config.nr;
          const uint32_t log2_kr = qs8_qc8w_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a55:
                #if XNN_ENABLE_ARM_DOTPROD
                  if (mr == 4 && nr == 16 && log2_kr == 2 && hardware_config->use_arm_neon_dot) {
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot;
                    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot;
                    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55;
                    break;
                  }
                #endif  // XNN_ENABLE_ARM_DOTPROD
              case cpuinfo_uarch_cortex_a53:
              case cpuinfo_uarch_cortex_a55r0:
                if (mr == 2 && nr == 8 && log2_kr == 3) {
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                  qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)].function[i] = (xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm;
                }
                break;

              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // !XNN_ENABLE_ASSEMBLY
        if (XNN_ENABLE_ARM_I8MM && hardware_config->use_arm_neon_i8mm) {
          #if XNN_ENABLE_ARM_I8MM
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 3;
          #endif  // XNN_ENABLE_ARM_I8MM && XNN_ENABLE_ARM_DOTPROD
        } else if (XNN_ENABLE_ARM_DOTPROD && hardware_config->use_arm_neon_dot) {
          #if XNN_ENABLE_ARM_DOTPROD
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot);
            qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot);
            qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
            qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
            qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
            qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
            qs8_qc8w_gemm_config.mr = 4;
            qs8_qc8w_gemm_config.nr = 16;
            qs8_qc8w_gemm_config.log2_kr = 2;
          #endif  // XNN_ENABLE_ARM_DOTPROD
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 1;
          qs8_qc8w_gemm_config.log2_sr = 2;
        }
      #endif  // XNN_ENABLE_ASSEMBLY
    #endif  // XNN_PLATFORM_IOS || XNN_PLATFORM_MAC || XNN_PLATFORM_WINDOWS
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512AMX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512amx) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__avx512amx);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x64c4__avx512amx);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(16)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x64c4__avx512amx);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x64c4__scalar;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 16;
        qs8_qc8w_gemm_config.nr = 64;
        qs8_qc8w_gemm_config.log2_kr = 2;
      } else
    #endif
    #if XNN_ENABLE_AVX512VNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512vnni) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
        #if XNN_ENABLE_AVX256VNNI
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__avx256vnni;
        #else
          qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x16c8__scalar;
        #endif
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 7;
        qs8_qc8w_gemm_config.nr = 16;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVXVNNIINT8 && XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnniint8 && hardware_config->use_x86_avxvnni) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_packw_gemm_goi_ukernel_x8c8__avxvnni;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 5;
        qs8_qc8w_gemm_config.nr = 8;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVXVNNI
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avxvnni) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(5)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x8c8__avxvnni;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 5;
        qs8_qc8w_gemm_config.nr = 8;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 7;
        qs8_qc8w_gemm_config.nr = 16;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx256skx) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx256skx);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx256skx);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 4;
        qs8_qc8w_gemm_config.nr = 8;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 8;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 3;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->use_wasm_sdot) {
        if (hardware_config->is_x86) {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmsdot);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmsdot);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 2;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 3;
        } else {
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2);
          qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2);
          qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2);
          qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
          qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
          qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
          qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
          qs8_qc8w_gemm_config.mr = 4;
          qs8_qc8w_gemm_config.nr = 8;
          qs8_qc8w_gemm_config.log2_kr = 3;
        }
      } else if (hardware_config->use_wasm_usdot) {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.packed_stride_weights_and_biases = NULL;  // Override the default packing function.
        qs8_qc8w_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_gio_w;
        qs8_qc8w_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_to_qu8_gemm_goi_w;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_to_qu8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_to_qu8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_to_qu8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 4;
        qs8_qc8w_gemm_config.nr = 8;
        qs8_qc8w_gemm_config.log2_kr = 3;
      } else {
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
        qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
        qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
        qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
        qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
        qs8_qc8w_gemm_config.mr = 4;
        qs8_qc8w_gemm_config.nr = 4;
        qs8_qc8w_gemm_config.log2_kr = 1;
        qs8_qc8w_gemm_config.log2_sr = 2;
      }
    #else
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 4;
      qs8_qc8w_gemm_config.nr = 4;
      qs8_qc8w_gemm_config.log2_kr = 1;
      qs8_qc8w_gemm_config.log2_sr = 2;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 2;
      qs8_qc8w_gemm_config.nr = 2;
    } else {
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
      qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
      qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
      qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
      qs8_qc8w_gemm_config.mr = 4;
      qs8_qc8w_gemm_config.nr = 4;
    }
  #else
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qs8_qc8w_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qs8_qc8w_gemm_config.init.qs8_qc8w = xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params;
    qs8_qc8w_gemm_config.pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
    qs8_qc8w_gemm_config.pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
    qs8_qc8w_gemm_config.pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;
    qs8_qc8w_gemm_config.mr = 3;
    qs8_qc8w_gemm_config.nr = 4;
  #endif
}

static void init_qu8_gemm_config(void) {
  // Use the same packing function throughout.
  qu8_gemm_config.pack_weights_and_biases =
      (xnn_pack_weights_and_biases_fn)xnn_pack_qu8_weights_and_biases;
  qu8_gemm_config.packed_stride_weights_and_biases =
      (xnn_packed_stride_weights_and_biases_fn)
          xnn_packed_stride_qu8_weights_and_biases;
  qu8_gemm_config.pack_gemm_gio = (xnn_packw_gemm_gio_ukernel_fn) xnn_pack_qu8_gemm_gio_w;
  qu8_gemm_config.pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qu8_gemm_goi_w;
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      #if XNN_ENABLE_ASSEMBLY
        switch (cpuinfo_get_uarch(0)->uarch) {
          case cpuinfo_uarch_cortex_a5:
          case cpuinfo_uarch_cortex_a7:
          case cpuinfo_uarch_krait:
          case cpuinfo_uarch_kryo:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a32:
          case cpuinfo_uarch_cortex_a35:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a53:
          case cpuinfo_uarch_cortex_a57:
          case cpuinfo_uarch_cortex_a72:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_cortex_a55r0:
          case cpuinfo_uarch_cortex_a55:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
          case cpuinfo_uarch_exynos_m1:
          case cpuinfo_uarch_exynos_m2:
          case cpuinfo_uarch_exynos_m3:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
          default:
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7);
            qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64);
            qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
            qu8_gemm_config.mr = 4;
            qu8_gemm_config.nr = 8;
            break;
        }

        #if XNN_MAX_UARCH_TYPES > 1
        {
          /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
          const uint32_t mr = qu8_gemm_config.mr;
          const uint32_t nr = qu8_gemm_config.nr;
          const uint32_t log2_kr = qu8_gemm_config.log2_kr;
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
            if (uarch_info == NULL) {
              /* No more microarchitectures in the system */
              break;
            }

            switch (uarch_info->uarch) {
              case cpuinfo_uarch_cortex_a53:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm;
                }
                break;
              case cpuinfo_uarch_cortex_a55r0:
              case cpuinfo_uarch_cortex_a55:
                if (mr == 4 && nr == 8 && log2_kr == 0) {
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7;
                  qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53;
                }
                break;
              default:
                break;
            }
          }
        }
        #endif  // XNN_MAX_UARCH_TYPES > 1
      #else  // XNN_ENABLE_ASSEMBLY
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane);
        qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
        qu8_gemm_config.mr = 3;
        qu8_gemm_config.nr = 8;
      #endif  // XNN_ENABLE_ASSEMBLY
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_armsimd32_params;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
      qu8_gemm_config.log2_kr = 2;
    }
  #elif XNN_ARCH_ARM64
    #if XNN_ENABLE_ASSEMBLY
      switch (cpuinfo_get_core(0)->uarch) {
        case cpuinfo_uarch_cortex_a53:
        case cpuinfo_uarch_cortex_a55r0:
        case cpuinfo_uarch_cortex_a55:
        case cpuinfo_uarch_kryo:
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu16_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu16_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu16_scalar_params;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          break;

        case cpuinfo_uarch_cortex_a57:
        case cpuinfo_uarch_cortex_a72:
        case cpuinfo_uarch_cortex_a73:
        case cpuinfo_uarch_cortex_a75:
        case cpuinfo_uarch_cortex_a76:
        case cpuinfo_uarch_exynos_m1:
        case cpuinfo_uarch_exynos_m2:
        case cpuinfo_uarch_exynos_m3:
        case cpuinfo_uarch_exynos_m4:
        case cpuinfo_uarch_neoverse_n1:
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          break;

        default:
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
          qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75);
          qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
          qu8_gemm_config.mr = 4;
          qu8_gemm_config.nr = 16;
          break;
      }
      #if XNN_MAX_UARCH_TYPES > 1
      {
        /* Choose micro-kernels for little cores according to micro-kernel specification for the big core */
        const uint32_t mr = qu8_gemm_config.mr;
        const uint32_t nr = qu8_gemm_config.nr;
        const uint32_t log2_kr = qu8_gemm_config.log2_kr;
        for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
          const struct cpuinfo_uarch_info* uarch_info = cpuinfo_get_uarch(i);
          if (uarch_info == NULL) {
            /* No more microarchitectures in the system */
            break;
          }

          switch (uarch_info->uarch) {
            case cpuinfo_uarch_cortex_a53:
            case cpuinfo_uarch_cortex_a55r0:
            case cpuinfo_uarch_cortex_a55:
              if (mr == 4 && nr == 16 && log2_kr == 0) {
                qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm;
                qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)].function[i] = (xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm;
              }
              break;
            default:
              break;
          }
        }
      }
      #endif  // XNN_MAX_UARCH_TYPES > 1
    #else  // !XNN_ENABLE_ASSEMBLY
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_rndnu_neon_params;
      qu8_gemm_config.mr = 4;
      qu8_gemm_config.nr = 16;
    #endif  // XNN_ENABLE_ASSEMBLY
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512SKX
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
        qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm);
        qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(7)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm);
        qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
        qu8_gemm_config.mr = 7;
        qu8_gemm_config.nr = 16;
        qu8_gemm_config.log2_kr = 3;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 8;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_avx) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 3;
      qu8_gemm_config.nr = 4;
      qu8_gemm_config.log2_kr = 3;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
    qu8_gemm_config.mr = 4;
    qu8_gemm_config.nr = 4;
    qu8_gemm_config.log2_kr = 1;
    qu8_gemm_config.log2_sr = 2;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(2)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 2;
      qu8_gemm_config.nr = 2;
    } else {
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic);
      qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(4)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic);
      qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
      qu8_gemm_config.mr = 4;
      qu8_gemm_config.nr = 4;
    }
  #else
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.gemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_gemm_ukernel((xnn_gemm_ukernel_fn) xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(1)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf);
    qu8_gemm_config.minmax.igemm[XNN_MR_TO_INDEX(3)] = xnn_init_hmp_igemm_ukernel((xnn_igemm_ukernel_fn) xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf);
    qu8_gemm_config.init.qu8 = xnn_init_qu8_conv_minmax_fp32_scalar_params;
    qu8_gemm_config.mr = 3;
    qu8_gemm_config.nr = 4;
  #endif
}

const struct xnn_gemm_config* xnn_init_f16_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_gemm);
  return &f16_gemm_config;
}

const struct xnn_gemm_config* xnn_init_pf32_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(pf32_gemm);
  return &pf32_gemm_config;
}

const struct xnn_gemm_config* xnn_init_f32_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_gemm);
  return &f32_gemm_config;
}

const struct xnn_gemm_config* xnn_init_f32_gemm_nr2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_gemm_nr2);
  return &f32_gemm_nr2_config;
}

const struct xnn_gemm_config* xnn_init_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_qc4w_gemm);
  return &f32_qc4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_f32_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_qc8w_gemm);
  return &f32_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f16_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(qd8_f16_qc8w_gemm);
  return &qd8_f16_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f16_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
  // there are no kernels on x86. qdu8_f16_qc4w kernels are used instead.
    return NULL;
  #endif

  XNN_INIT_ONCE(qd8_f16_qc4w_gemm);
  return &qd8_f16_qc4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qdu8_f16_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(qdu8_f16_qc4w_gemm);
  return &qdu8_f16_qc4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f16_qb4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(qd8_f16_qb4w_gemm);
  return &qd8_f16_qb4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qd8_f32_qc4w_gemm);
  return &qd8_f32_qc4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qdu8_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qdu8_f32_qc4w_gemm);
  return &qdu8_f32_qc4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f32_qb4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qd8_f32_qb4w_gemm);
  return &qd8_f32_qb4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qdu8_f32_qb4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qdu8_f32_qb4w_gemm);
  return &qdu8_f32_qb4w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qdu8_f16_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qdu8_f16_qc8w_gemm);
  return &qdu8_f16_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qdu8_f32_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qdu8_f32_qc8w_gemm);
  return &qdu8_f32_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qd8_f32_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qd8_f32_qc8w_gemm);
  return &qd8_f32_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qp8_f32_qc4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
XNN_INIT_ONCE(qp8_f32_qc4w_gemm);
  // Only return the config pointer if it actually provides a kernel.
  if (qp8_f32_qc4w_gemm_config.minmax.qp8gemm[0].function[0] != NULL) {
    return &qp8_f32_qc4w_gemm_config;
  }
  return NULL;
}

const struct xnn_gemm_config* xnn_init_qp8_f32_qb4w_gemm_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
XNN_INIT_ONCE(qp8_f32_qb4w_gemm);
  // Only return the config pointer if it actually provides a kernel.
  if (qp8_f32_qb4w_gemm_config.minmax.qp8gemm_bl[0].function[0] != NULL) {
    return &qp8_f32_qb4w_gemm_config;
  }
  return NULL;
}

const struct xnn_gemm_config* xnn_init_qs8_qc8w_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_qc8w_gemm);
  return &qs8_qc8w_gemm_config;
}

const struct xnn_gemm_config* xnn_init_qu8_gemm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_gemm);
  return &qu8_gemm_config;
}
