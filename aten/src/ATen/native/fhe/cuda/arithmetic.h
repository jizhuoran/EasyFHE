#pragma once

namespace at::native {

void vadd_mod(
    const size_t N,
    int64_t l,
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* mod);

void vsub_mod(
    const size_t N,
    int64_t l,
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* mod);

void vneg_mod(
    const size_t N,
    int64_t l,
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* mod);

} // namespace at::native