// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat result = x;
        v_if(x >= 0.0f) { result = 0.0f; }
        v_endif;

        sfpi::vFloat abs_x = sfpi::abs(x);
        sfpi::vFloat exp_neg_abs_x = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(-abs_x);
        sfpi::vFloat log1p_exp = exp_neg_abs_x;

        // For x >= 4, e = exp(-x) <= 0.0183. Five terms of
        // log1p(e) = e - e^2/2 + e^3/3 - e^4/4 + e^5/5
        // have a remainder below 6e-12 at the boundary, comfortably inside fp32 ULP.
        // This avoids the generic log1p reduction on the positive tail while retaining
        // the accurate identity used by the central/negative ranges.
        v_if(x >= 4.0f) {
            sfpi::vFloat e = exp_neg_abs_x;
            sfpi::vFloat p = 0.2f;
            p = p * e - 0.25f;
            p = p * e + 0.3333333333333333f;
            p = p * e - 0.5f;
            p = p * e + 1.0f;
            log1p_exp = p * e;
        }
        v_elseif(x < 4.0f) {
            log1p_exp = calculate_log1p_fp32<is_fp32_dest_acc_en>(exp_neg_abs_x);
        }
        v_endif;

        result = result - log1p_exp;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
void logsigmoid_init() {
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
