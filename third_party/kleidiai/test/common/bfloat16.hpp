//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <type_traits>

#include "test/common/type_traits.hpp"

extern "C" {

/// Converts single-precision floating-point to half-precision brain floating-point.
///
/// @params[in] value The single-precision floating-point value.
///
/// @return The half-precision brain floating-point value reinterpreted as 16-bit unsigned integer.
uint16_t kai_test_bfloat16_from_float(float value);

}  // extern "C"

namespace kai::test {

/// Half-precision brain floating-point.
class BFloat16 {
public:
    /// Constructor.
    BFloat16() = default;

    /// Creates a new object from the specified numeric value.
    explicit BFloat16(float value) : m_data(kai_test_bfloat16_from_float(value)) {
    }

    /// Assigns to the specified numeric value which will be converted to `bfloat16_t`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    BFloat16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
        m_data = kai_test_bfloat16_from_float(value_f32);
        return *this;
    }

    /// Converts to single-precision floating-point.
    explicit operator float() const {
        union {
            float f32;
            uint32_t u32;
        } data;

        data.u32 = static_cast<uint32_t>(m_data) << 16;

        return data.f32;
    }

private:
    /// Equality operator.
    [[nodiscard]] friend bool operator==(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data == rhs.m_data;
    }

    /// Inequality operator.
    [[nodiscard]] friend bool operator!=(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data != rhs.m_data;
    }

    /// Writes the value to the output stream.
    ///
    /// @param[in] os Output stream to be written to.
    /// @param[in] value Value to be written.
    ///
    /// @return The output stream.
    friend std::ostream& operator<<(std::ostream& os, BFloat16 value);

    uint16_t m_data;
};

}  // namespace kai::test
