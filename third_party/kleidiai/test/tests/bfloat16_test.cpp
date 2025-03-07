//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/bfloat16.hpp"

#include <gtest/gtest.h>

#include "test/common/cpu_info.hpp"

namespace kai::test {

TEST(BFloat16, SimpleTest) {
    if (!cpu_has_bf16()) {
        GTEST_SKIP();
    }

    ASSERT_EQ(static_cast<float>(BFloat16()), 0.0F);
    ASSERT_EQ(static_cast<float>(BFloat16(1.25F)), 1.25F);
    ASSERT_EQ(static_cast<float>(BFloat16(3)), 3.0F);

    ASSERT_FALSE(BFloat16(1.25F) == BFloat16(2.0F));
    ASSERT_TRUE(BFloat16(1.25F) == BFloat16(1.25F));
    ASSERT_FALSE(BFloat16(2.0F) == BFloat16(1.25F));

    ASSERT_TRUE(BFloat16(1.25F) != BFloat16(2.0F));
    ASSERT_FALSE(BFloat16(1.25F) != BFloat16(1.25F));
    ASSERT_TRUE(BFloat16(2.0F) != BFloat16(1.25F));
}

}  // namespace kai::test
