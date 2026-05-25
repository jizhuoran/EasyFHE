#include "ATen/native/fhe/sampler/third_party/blake2/utils/prng/blake2.h"

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <set>
#include <string_view>

namespace {

using UIntMatrix = std::vector<std::vector<uint64_t>>;

at::Tensor EmptyTensor(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).device(at::kCPU));
}

at::Tensor ScalarInt64Tensor(int64_t value) {
    auto out = EmptyTensor({}, at::kLong);
    *out.data_ptr<int64_t>() = value;
    return out;
}

double NowSeconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct NativeProfileStats {
    bool enabled = std::getenv("EASYFHE_NATIVE_PROFILE") != nullptr;
    uint64_t switchCalls = 0;
    uint64_t switchParts = 0;
    uint64_t rotationKeys = 0;
    double keygen = 0.0;
    double evalMult = 0.0;
    double rotationTotal = 0.0;
    double automorphism = 0.0;
    double switchTotal = 0.0;
    double uniform = 0.0;
    double gaussian = 0.0;
    double noiseScale = 0.0;
    double mulSub = 0.0;
    double pInjection = 0.0;
    double tupleReturn = 0.0;
    double encodeEncryptDecrypt = 0.0;

    void Reset() {
        const bool wasEnabled = enabled;
        *this                 = NativeProfileStats{};
        enabled               = wasEnabled;
    }

    void Report() const {
        if (!enabled) {
            return;
        }
        std::cerr << "[native-profile] keygen=" << keygen
                  << " eval_mult=" << evalMult
                  << " rotation_total=" << rotationTotal
                  << " rotation_keys=" << rotationKeys
                  << " switch_calls=" << switchCalls
                  << " switch_parts=" << switchParts
                  << " switch_total=" << switchTotal
                  << " uniform=" << uniform
                  << " gaussian=" << gaussian
                  << " noise_scale=" << noiseScale
                  << " mul_sub=" << mulSub
                  << " p_injection=" << pInjection
                  << " automorphism=" << automorphism
                  << " tuple_return=" << tupleReturn
                  << " encode_encrypt_decrypt=" << encodeEncryptDecrypt
                  << std::endl;
    }
};

NativeProfileStats gProfile;

uint32_t NativeThreadCount() {
    static const uint32_t count = []() {
        const char* env = std::getenv("EASYFHE_NATIVE_THREADS");
        if (env != nullptr) {
            const long parsed = std::strtol(env, nullptr, 10);
            return parsed > 0 ? static_cast<uint32_t>(parsed) : uint32_t{1};
        }
        const uint32_t hw = std::thread::hardware_concurrency();
        return hw == 0 ? uint32_t{1} : hw;
    }();
    return count;
}

template <class Function>
void SequentialFor(size_t count, Function fn) {
    for (size_t i = 0; i < count; ++i) {
        fn(i);
    }
}

template <class Function>
void ParallelForKeys(size_t count, Function fn) {
    const uint32_t threadCount = std::min<uint32_t>(NativeThreadCount(), static_cast<uint32_t>(count));
    if (threadCount <= 1 || count <= 1) {
        SequentialFor(count, fn);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threadCount - 1);
    const size_t block = (count + threadCount - 1) / threadCount;
    for (uint32_t t = 1; t < threadCount; ++t) {
        const size_t begin = t * block;
        const size_t end   = std::min(count, begin + block);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([=, &fn]() {
            for (size_t i = begin; i < end; ++i) {
                fn(i);
            }
        });
    }

    const size_t mainEnd = std::min(count, block);
    for (size_t i = 0; i < mainEnd; ++i) {
        fn(i);
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

enum class SecretKeyDistLocal {
    Gaussian,
    UniformTernary,
    SparseTernary,
};

enum class RandomModeLocal {
    Sequential,
    ParallelDeterministic,
};

enum class RotationRandomModeLocal {
    Fresh,
    ReuseByShape,
};

class FixedBlake2Prng {
public:
    using result_type = uint32_t;

    static constexpr result_type min() {
        return std::numeric_limits<result_type>::min();
    }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    FixedBlake2Prng() {
        seed_[0] = 1;
    }

    FixedBlake2Prng(std::array<result_type, 16> seed, uint64_t counter, size_t bufferIndex) : seed_(std::move(seed)) {
        if (bufferIndex == 0) {
            counter_     = counter;
            bufferIndex_ = 0;
        }
        else {
            if (counter == 0) {
                throw std::runtime_error("invalid Blake2 PRNG state");
            }
            counter_     = counter - 1;
            bufferIndex_ = 0;
            Generate();
            bufferIndex_ = bufferIndex;
        }
    }

    result_type operator()() {
        if (bufferIndex_ == buffer_.size()) {
            bufferIndex_ = 0;
        }
        if (bufferIndex_ == 0) {
            Generate();
        }
        return buffer_[bufferIndex_++];
    }

private:
    void Generate() {
        if (blake2xb(buffer_.data(),
                     buffer_.size() * sizeof(result_type),
                     &counter_,
                     sizeof(counter_),
                     seed_.data(),
                     seed_.size() * sizeof(result_type)) != 0) {
            throw std::runtime_error("EasyFHE fixed PRNG blake2xb failed");
        }
        ++counter_;
    }

    std::array<result_type, 16> seed_{};
    std::array<result_type, 1024> buffer_{};
    size_t bufferIndex_ = 0;
    uint64_t counter_   = 0;
};

void AppendUint64(std::string& out, uint64_t value) {
    for (uint32_t i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
    }
}

std::array<FixedBlake2Prng::result_type, 16> DeriveSeed(const std::string& label,
                                                        uint64_t domain,
                                                        uint64_t part) {
    std::array<FixedBlake2Prng::result_type, 16> master{};
    master[0] = 1;

    std::string material = "EasyFHE/native-sampler/v1/";
    material += label;
    AppendUint64(material, domain);
    AppendUint64(material, part);

    std::array<FixedBlake2Prng::result_type, 16> seed{};
    if (blake2xb(seed.data(),
                 seed.size() * sizeof(FixedBlake2Prng::result_type),
                 material.data(),
                 material.size(),
                 master.data(),
                 master.size() * sizeof(FixedBlake2Prng::result_type)) != 0) {
        throw std::runtime_error("EasyFHE native sampler seed derivation failed");
    }
    return seed;
}

struct NativeSamplingContext {
    explicit NativeSamplingContext(RandomModeLocal mode) : mode(mode) {}

    bool IsParallelDeterministic() const {
        return mode == RandomModeLocal::ParallelDeterministic;
    }

    FixedBlake2Prng SwitchPartPrng(const std::string& label, uint64_t domain, uint64_t part) const {
        return FixedBlake2Prng(DeriveSeed(label, domain, part), 0, 0);
    }

    RandomModeLocal mode = RandomModeLocal::Sequential;
};

SecretKeyDistLocal ParseSecretKeyDist(const std::string& value) {
    if (value == "GAUSSIAN")
        return SecretKeyDistLocal::Gaussian;
    if (value == "UNIFORM_TERNARY")
        return SecretKeyDistLocal::UniformTernary;
    if (value == "SPARSE_TERNARY")
        return SecretKeyDistLocal::SparseTernary;
    throw std::invalid_argument("unsupported secret_key_dist: " + value);
}

RandomModeLocal ParseRandomMode(const std::string& value) {
    if (value.empty()) {
        return RandomModeLocal::Sequential;
    }
    if (value == "sequential") {
        return RandomModeLocal::Sequential;
    }
    if (value == "parallel_deterministic") {
        return RandomModeLocal::ParallelDeterministic;
    }
    throw std::invalid_argument("unsupported native sampler random_mode: " + value);
}

RotationRandomModeLocal ParseRotationRandomMode(const std::string& value) {
    if (value.empty() || value == "fresh") {
        return RotationRandomModeLocal::Fresh;
    }
    if (value == "reuse_by_shape") {
        return RotationRandomModeLocal::ReuseByShape;
    }
    throw std::invalid_argument("unsupported native sampler rotation_random_mode: " + value);
}

std::vector<double> ValuesFromTensor(const at::Tensor& values) {
    if (!values.defined() || values.numel() == 0) {
        return {0.0};
    }
    const at::Tensor contiguous = values.contiguous();
    std::vector<double> out(static_cast<size_t>(contiguous.numel()));
    if (contiguous.scalar_type() == at::kDouble) {
        const auto* ptr = contiguous.data_ptr<double>();
        std::copy(ptr, ptr + contiguous.numel(), out.begin());
    }
    else if (contiguous.scalar_type() == at::kFloat) {
        const auto* ptr = contiguous.data_ptr<float>();
        for (int64_t i = 0; i < contiguous.numel(); ++i) {
            out[static_cast<size_t>(i)] = static_cast<double>(ptr[i]);
        }
    }
    else {
        throw std::invalid_argument("native sampler values must be a float32 or float64 tensor");
    }
    return out;
}

std::vector<std::vector<int32_t>> Int32VectorGroupsFromFlat(at::IntArrayRef values, at::IntArrayRef offsets) {
    if (values.empty()) {
        return {};
    }
    if (offsets.empty()) {
        std::vector<int32_t> group;
        group.reserve(values.size());
        for (auto value : values) {
            if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
                throw std::invalid_argument("rotation_indices contains a value outside int32 range");
            }
            group.push_back(static_cast<int32_t>(value));
        }
        return {std::move(group)};
    }
    if (offsets.front() != 0 || offsets.back() != static_cast<int64_t>(values.size())) {
        throw std::invalid_argument("rotation_group_offsets must start at 0 and end at len(rotation_indices)");
    }
    std::vector<std::vector<int32_t>> groups;
    groups.reserve(offsets.size() - 1);
    for (size_t groupIndex = 0; groupIndex + 1 < offsets.size(); ++groupIndex) {
        const int64_t begin = offsets[groupIndex];
        const int64_t end   = offsets[groupIndex + 1];
        if (begin < 0 || end < begin || end > static_cast<int64_t>(values.size())) {
            throw std::invalid_argument("rotation_group_offsets contains an invalid range");
        }
        std::vector<int32_t> group;
        group.reserve(static_cast<size_t>(end - begin));
        for (int64_t i = begin; i < end; ++i) {
            const int64_t value = values[static_cast<size_t>(i)];
            if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
                throw std::invalid_argument("rotation_indices contains a value outside int32 range");
            }
            group.push_back(static_cast<int32_t>(value));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

std::map<uint32_t, uint32_t> UInt32MapFromParallelArrays(at::IntArrayRef keys, at::IntArrayRef values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("rotation trim auto-index and limb arrays must have the same length");
    }
    std::map<uint32_t, uint32_t> out;
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto key = keys[i];
        const auto value = values[i];
        if (key < 0 || value < 0 || key > std::numeric_limits<uint32_t>::max() ||
            value > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("rotation trim arrays contain a value outside uint32 range");
        }
        out[static_cast<uint32_t>(key)] = static_cast<uint32_t>(value);
    }
    return out;
}


at::Tensor MatrixToTensor(const UIntMatrix& matrix) {
    const size_t rows = matrix.size();
    const size_t cols = rows == 0 ? 0 : matrix.front().size();
    at::Tensor out = EmptyTensor({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, at::kUInt64);
    auto* outPtr = out.data_ptr<uint64_t>();
    for (size_t row = 0; row < rows; ++row) {
        if (matrix[row].size() != cols) {
            throw std::runtime_error("ragged uint64 matrix cannot be converted to tensor");
        }
        std::memcpy(outPtr + row * cols, matrix[row].data(), cols * sizeof(uint64_t));
    }
    return out;
}

UIntMatrix MatrixFromTensor(const at::Tensor& tensor, const char* name) {
    const at::Tensor array = tensor.contiguous();
    if (array.dim() != 2 || array.scalar_type() != at::kUInt64) {
        throw std::invalid_argument(std::string(name) + " must be a 2D uint64 tensor");
    }
    const auto rows = static_cast<size_t>(array.size(0));
    const auto cols = static_cast<size_t>(array.size(1));
    const auto* ptr = array.data_ptr<uint64_t>();
    UIntMatrix out(rows, std::vector<uint64_t>(cols));
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(out[row].data(), ptr + row * cols, cols * sizeof(uint64_t));
    }
    return out;
}

std::vector<int64_t> Int64VectorFromTensor(const at::Tensor& tensor, const char* name) {
    const at::Tensor array = tensor.contiguous();
    if (array.dim() != 1 || array.scalar_type() != at::kLong) {
        throw std::invalid_argument(std::string(name) + " must be a 1D int64 tensor");
    }
    const auto size = static_cast<size_t>(array.size(0));
    const auto* ptr = array.data_ptr<int64_t>();
    std::vector<int64_t> out(size);
    std::copy(ptr, ptr + size, out.begin());
    return out;
}

at::Tensor Int64VectorToTensor(const std::vector<int64_t>& values) {
    at::Tensor out = EmptyTensor({static_cast<int64_t>(values.size())}, at::kLong);
    std::copy(values.begin(), values.end(), out.data_ptr<int64_t>());
    return out;
}

at::Tensor UInt32VectorToInt32Tensor(const std::vector<uint32_t>& values) {
    at::Tensor out = EmptyTensor({static_cast<int64_t>(values.size())}, at::kInt);
    auto* outPtr = out.data_ptr<int32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("uint32 vector value does not fit int32 tensor");
        }
        outPtr[i] = static_cast<int32_t>(values[i]);
    }
    return out;
}

at::Tensor MatrixVectorToTensor(const std::vector<UIntMatrix>& matrices) {
    const size_t parts = matrices.size();
    const size_t rows  = parts == 0 ? 0 : matrices.front().size();
    const size_t cols  = rows == 0 ? 0 : matrices.front().front().size();
    at::Tensor out = EmptyTensor(
        {static_cast<int64_t>(parts), static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, at::kUInt64);
    auto* outPtr = out.data_ptr<uint64_t>();
    for (size_t part = 0; part < parts; ++part) {
        if (matrices[part].size() != rows) {
            throw std::runtime_error("ragged uint64 matrix vector cannot be converted to tensor");
        }
        for (size_t row = 0; row < rows; ++row) {
            if (matrices[part][row].size() != cols) {
                throw std::runtime_error("ragged uint64 matrix vector cannot be converted to tensor");
            }
            std::memcpy(outPtr + (part * rows + row) * cols, matrices[part][row].data(), cols * sizeof(uint64_t));
        }
    }
    return out;
}

at::Tensor TrimmedSwitchKeyToTensor(const std::vector<UIntMatrix>& matrices,
                                              uint32_t limb,
                                              uint32_t qLimbs,
                                              uint32_t pLimbs,
                                              uint32_t dnum) {
    const uint32_t ringDim = matrices.empty() || matrices.front().empty() ? 0 : matrices.front().front().size();
    const uint32_t alpha   = (qLimbs + dnum - 1) / dnum;
    const uint32_t beta    = (limb + alpha - 1) / alpha;
    for (uint32_t part = 0; part < beta; ++part) {
        if (part >= matrices.size()) {
            throw std::runtime_error("not enough switch-key decomposition parts for requested trim");
        }
        for (uint32_t row = 0; row < limb + pLimbs; ++row) {
            const uint32_t sourceRow = row < limb ? row : qLimbs + (row - limb);
            if (sourceRow >= matrices[part].size()) {
                throw std::runtime_error("trimmed switch-key source row out of range");
            }
            if (matrices[part][sourceRow].size() != ringDim) {
                throw std::runtime_error("trimmed switch-key row has inconsistent ring dimension");
            }
        }
    }
    at::Tensor out = EmptyTensor(
        {static_cast<int64_t>(beta), static_cast<int64_t>(limb + pLimbs), static_cast<int64_t>(ringDim)}, at::kUInt64);
    auto* outPtr = out.data_ptr<uint64_t>();

    SequentialFor(static_cast<size_t>(beta) * (limb + pLimbs), [&](size_t flatRow) {
        const uint32_t part = static_cast<uint32_t>(flatRow / (limb + pLimbs));
        const uint32_t row = static_cast<uint32_t>(flatRow % (limb + pLimbs));
        const uint32_t sourceRow = row < limb ? row : qLimbs + (row - limb);
        const size_t offset = flatRow * ringDim;
        std::memcpy(outPtr + offset, matrices[part][sourceRow].data(), ringDim * sizeof(uint64_t));
    });
    return out;
}

at::Tensor VectorToTensor(const std::vector<uint64_t>& values) {
    at::Tensor out = EmptyTensor({static_cast<int64_t>(values.size())}, at::kUInt64);
    std::copy(values.begin(), values.end(), out.data_ptr<uint64_t>());
    return out;
}

at::Tensor VectorToTensor(const std::vector<double>& values) {
    at::Tensor out = EmptyTensor({static_cast<int64_t>(values.size())}, at::kDouble);
    std::copy(values.begin(), values.end(), out.data_ptr<double>());
    return out;
}

uint32_t GetMsb(uint64_t value) {
    uint32_t msb = 0;
    while (value != 0) {
        ++msb;
        value >>= 1;
    }
    return msb;
}

uint32_t ReverseBitsLocal(uint32_t value, uint32_t width) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < width; ++i) {
        result <<= 1;
        result |= (value & 1U);
        value >>= 1;
    }
    return result;
}

uint64_t ModMul(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(lhs) * rhs) % modulus);
}

uint64_t ShoupPrecon(uint64_t value, uint64_t modulus) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(value) << 64) / modulus);
}

uint64_t ModMulPrecon(uint64_t lhs, uint64_t rhs, uint64_t rhsPrecon, uint64_t modulus) {
    const uint64_t quotient = static_cast<uint64_t>((static_cast<unsigned __int128>(lhs) * rhsPrecon) >> 64);
    uint64_t result = lhs * rhs - quotient * modulus;
    if (result >= modulus) {
        result -= modulus;
    }
    if (result >= modulus) {
        result -= modulus;
    }
    return result;
}

std::vector<uint64_t> MakeScalarShoupTable(uint64_t scalar, const std::vector<uint64_t>& moduli) {
    std::vector<uint64_t> result;
    result.reserve(moduli.size());
    for (auto modulus : moduli) {
        result.push_back(ShoupPrecon(scalar % modulus, modulus));
    }
    return result;
}

uint64_t ModAdd(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    const uint64_t sum = lhs + rhs;
    if (sum < lhs || sum >= modulus) {
        return sum - modulus;
    }
    return sum;
}

uint64_t ModSub(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    if (lhs < rhs) {
        lhs += modulus;
    }
    return lhs - rhs;
}

uint64_t ModPow(uint64_t base, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1 % modulus;
    while (exponent > 0) {
        if ((exponent & 1U) != 0) {
            result = ModMul(result, base, modulus);
        }
        exponent >>= 1;
        if (exponent != 0) {
            base = ModMul(base, base, modulus);
        }
    }
    return result;
}

uint64_t ModInverse(uint64_t value, uint64_t modulus) {
    int64_t t = 0;
    int64_t newT = 1;
    int64_t r = static_cast<int64_t>(modulus);
    int64_t newR = static_cast<int64_t>(value % modulus);
    while (newR != 0) {
        const int64_t quotient = r / newR;
        const int64_t tmpT = t - quotient * newT;
        t = newT;
        newT = tmpT;
        const int64_t tmpR = r - quotient * newR;
        r = newR;
        newR = tmpR;
    }
    if (r > 1) {
        throw std::runtime_error("value is not invertible modulo modulus");
    }
    if (t < 0) {
        t += static_cast<int64_t>(modulus);
    }
    return static_cast<uint64_t>(t);
}

constexpr int64_t Max64BitValueLocal() {
    return static_cast<int64_t>((uint64_t{1} << 63) - (uint64_t{1} << 9) - 1);
}

bool Is64BitOverflowLocal(double value) {
    return std::abs(value) > static_cast<double>(Max64BitValueLocal());
}

std::vector<uint64_t> MakeRootOfUnityTable(uint64_t rootOfUnity, uint32_t ringDim, uint64_t modulus) {
    std::vector<uint64_t> table(ringDim);
    uint64_t x        = 1;
    const uint32_t msb = GetMsb(ringDim - 1);
    for (uint32_t i = 0; i < ringDim; ++i) {
        table[ReverseBitsLocal(i, msb)] = x;
        x = ModMul(x, rootOfUnity, modulus);
    }
    return table;
}

std::vector<uint64_t> MakeShoupTable(const std::vector<uint64_t>& values, uint64_t modulus) {
    std::vector<uint64_t> table(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        table[i] = ShoupPrecon(values[i], modulus);
    }
    return table;
}

void ForwardNTTInPlaceWithTable(std::vector<uint64_t>& values,
                                const std::vector<uint64_t>& rootTable,
                                const std::vector<uint64_t>& rootShoupTable,
                                uint64_t modulus) {
    const uint32_t n = static_cast<uint32_t>(values.size());
    if (n == 0 || (n & (n - 1)) != 0 || rootTable.size() != n || rootShoupTable.size() != n) {
        throw std::runtime_error("EasyFHE native NTT root table size mismatch");
    }
    uint32_t t           = n >> 1;
    uint32_t logt1       = GetMsb(t);
    for (uint32_t m = 1; m < n; m <<= 1) {
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t j1    = i << logt1;
            const uint32_t j2    = j1 + t;
            const uint64_t omega = rootTable[m + i];
            const uint64_t omegaPrecon = rootShoupTable[m + i];
            for (uint32_t indexLo = j1; indexLo < j2; ++indexLo) {
                const uint32_t indexHi = indexLo + t;
                const uint64_t loVal   = values[indexLo];
                const uint64_t factor  = ModMulPrecon(values[indexHi], omega, omegaPrecon, modulus);
                values[indexLo]        = ModAdd(loVal, factor, modulus);
                values[indexHi]        = ModSub(loVal, factor, modulus);
            }
        }
        t >>= 1;
        --logt1;
    }
}

void ForwardNTTInPlace(std::vector<uint64_t>& values, uint64_t rootOfUnity, uint64_t modulus) {
    const uint32_t n = static_cast<uint32_t>(values.size());
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("EasyFHE native NTT requires a non-empty power-of-two ring dimension");
    }

    const auto rootTable = MakeRootOfUnityTable(rootOfUnity, n, modulus);
    const auto rootShoupTable = MakeShoupTable(rootTable, modulus);
    ForwardNTTInPlaceWithTable(values, rootTable, rootShoupTable, modulus);
}

void InverseNTTInPlace(std::vector<uint64_t>& values, uint64_t rootOfUnity, uint64_t modulus) {
    const uint32_t n = static_cast<uint32_t>(values.size());
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("EasyFHE native iNTT requires a non-empty power-of-two ring dimension");
    }

    const uint64_t rootInv = ModPow(rootOfUnity, modulus - 2, modulus);
    const auto rootInvTable = MakeRootOfUnityTable(rootInv, n, modulus);
    uint32_t t = 1;
    uint32_t logt1 = 1;
    for (uint32_t m = n >> 1; m >= 1; m >>= 1) {
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t j1 = i << logt1;
            const uint32_t j2 = j1 + t;
            const uint64_t omega = rootInvTable[m + i];
            for (uint32_t indexLo = j1; indexLo < j2; ++indexLo) {
                const uint32_t indexHi = indexLo + t;
                const uint64_t loVal = values[indexLo];
                const uint64_t hiVal = values[indexHi];
                uint64_t omegaFactor = loVal;
                if (omegaFactor < hiVal) {
                    omegaFactor += modulus;
                }
                omegaFactor -= hiVal;
                values[indexLo] = ModAdd(loVal, hiVal, modulus);
                values[indexHi] = ModMul(omegaFactor, omega, modulus);
            }
        }
        if (m == 1) {
            break;
        }
        t <<= 1;
        ++logt1;
    }

    const uint64_t cyclOrderInv = ModPow((uint64_t{2} * n) % modulus, modulus - 2, modulus);
    for (auto& value : values) {
        value = ModMul(value, cyclOrderInv, modulus);
    }
}

void BitReverseComplexInPlace(std::vector<std::complex<double>>& values) {
    const uint32_t size = static_cast<uint32_t>(values.size());
    for (size_t i = 1, j = 0; i < size; ++i) {
        size_t bit = size >> 1;
        for (; j >= bit; bit >>= 1) {
            j -= bit;
        }
        j += bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
}

void FFTSpecialInvNative(std::vector<std::complex<double>>& values, uint32_t cyclOrder) {
    const uint32_t valsSize = static_cast<uint32_t>(values.size());
    const uint32_t halfSlots = valsSize;

    std::vector<uint32_t> rotGroup(halfSlots);
    uint32_t fivePows = 1;
    for (size_t i = 0; i < halfSlots; ++i) {
        rotGroup[i] = fivePows;
        fivePows *= 5;
        fivePows %= cyclOrder;
    }

    std::vector<std::complex<double>> ksiPows(cyclOrder + 1);
    for (size_t j = 0; j < cyclOrder; ++j) {
        const double angle = 2.0 * M_PI * j / cyclOrder;
        ksiPows[j].real(std::cos(angle));
        ksiPows[j].imag(std::sin(angle));
    }
    ksiPows[cyclOrder] = ksiPows[0];

    for (size_t len = valsSize; len >= 1; len >>= 1) {
        for (size_t i = 0; i < valsSize; i += len) {
            const size_t lenh = len >> 1;
            const size_t lenq = len << 2;
            const size_t gap  = cyclOrder / lenq;
            for (size_t j = 0; j < lenh; ++j) {
                const size_t idx       = (lenq - (rotGroup[j] % lenq)) * gap;
                std::complex<double> u = values[i + j] + values[i + j + lenh];
                std::complex<double> v = values[i + j] - values[i + j + lenh];
                v *= ksiPows[idx];
                values[i + j]        = u;
                values[i + j + lenh] = v;
            }
        }
    }
    BitReverseComplexInPlace(values);

    for (auto& value : values) {
        value /= valsSize;
    }
}

void FFTSpecialNative(std::vector<std::complex<double>>& values, uint32_t cyclOrder) {
    const uint32_t valsSize = static_cast<uint32_t>(values.size());
    const uint32_t halfSlots = valsSize;

    std::vector<uint32_t> rotGroup(halfSlots);
    uint32_t fivePows = 1;
    for (size_t i = 0; i < halfSlots; ++i) {
        rotGroup[i] = fivePows;
        fivePows *= 5;
        fivePows %= cyclOrder;
    }

    std::vector<std::complex<double>> ksiPows(cyclOrder + 1);
    for (size_t j = 0; j < cyclOrder; ++j) {
        const double angle = 2.0 * M_PI * j / cyclOrder;
        ksiPows[j].real(std::cos(angle));
        ksiPows[j].imag(std::sin(angle));
    }
    ksiPows[cyclOrder] = ksiPows[0];

    BitReverseComplexInPlace(values);
    for (size_t len = 2; len <= valsSize; len <<= 1) {
        const size_t lenh = len >> 1;
        const size_t lenq = len << 2;
        const size_t gap = cyclOrder / lenq;
        for (size_t i = 0; i < valsSize; i += len) {
            for (size_t j = 0; j < lenh; ++j) {
                const size_t idx = (rotGroup[j] % lenq) * gap;
                std::complex<double> u = values[i + j];
                std::complex<double> v = values[i + j + lenh];
                v *= ksiPows[idx];
                values[i + j] = u + v;
                values[i + j + lenh] = u - v;
            }
        }
    }
}

uint64_t GenerateUniformMod(FixedBlake2Prng& prng, uint64_t modulus) {
    if (modulus == 0) {
        throw std::runtime_error("cannot sample modulo zero");
    }

    constexpr uint32_t chunkWidth = std::numeric_limits<uint32_t>::digits;
    const uint32_t chunksPerValue = (GetMsb(modulus) - 1) / chunkWidth;
    const uint32_t shiftChunk     = chunksPerValue * chunkWidth;
    const uint32_t bound          = static_cast<uint32_t>(modulus >> shiftChunk);

    std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
    const typename std::uniform_int_distribution<uint32_t>::param_type boundParam(0, bound);
    while (true) {
        uint64_t result = 0;
        for (uint32_t i = 0, shift = 0; i < chunksPerValue; ++i, shift += chunkWidth) {
            result += uint64_t{prng()} << shift;
        }
        result += uint64_t{dist(prng, boundParam)} << shiftChunk;
        if (result < modulus) {
            return result;
        }
    }
}

uint64_t GenerateUniformMod(FixedBlake2Prng& prng,
                            uint64_t modulus,
                            std::uniform_int_distribution<uint32_t>& dist) {
    if (modulus == 0) {
        throw std::runtime_error("cannot sample modulo zero");
    }

    constexpr uint32_t chunkWidth = std::numeric_limits<uint32_t>::digits;
    const uint32_t chunksPerValue = (GetMsb(modulus) - 1) / chunkWidth;
    const uint32_t shiftChunk     = chunksPerValue * chunkWidth;
    const uint32_t bound          = static_cast<uint32_t>(modulus >> shiftChunk);

    (void)dist;
    auto drawBounded = [&]() {
        const uint32_t range = bound + 1;
        if (range == 0) {
            return prng();
        }
        if (range == 1) {
            return uint32_t{0};
        }
        uint32_t width = 32 - static_cast<uint32_t>(__builtin_clz(range)) - 1;
        if ((range & (std::numeric_limits<uint32_t>::max() >> (32 - width))) != 0) {
            ++width;
        }
        const uint32_t mask = width == 32 ? std::numeric_limits<uint32_t>::max() :
                                            (std::numeric_limits<uint32_t>::max() >> (32 - width));
        uint32_t value = 0;
        do {
            value = prng() & mask;
        } while (value >= range);
        return value;
    };

    while (true) {
        uint64_t result = 0;
        for (uint32_t i = 0, shift = 0; i < chunksPerValue; ++i, shift += chunkWidth) {
            result += uint64_t{prng()} << shift;
        }
        result += uint64_t{drawBounded()} << shiftChunk;
        if (result < modulus) {
            return result;
        }
    }
}

uint64_t Gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

bool MillerRabinPrimalityTestLocal(uint64_t p, FixedBlake2Prng& prng, uint32_t niter = 100);
void PrimeFactorizeLocal(uint64_t n, std::set<uint64_t>& primeFactors, FixedBlake2Prng& prng);

bool WitnessFunctionLocal(uint64_t a, uint64_t d, uint32_t s, uint64_t p) {
    uint64_t mod = ModPow(a, d, p);
    bool prevMod = false;
    for (uint32_t i = 0; i < s; ++i) {
        prevMod = (mod != 1 && mod != p - 1);
        mod = ModMul(mod, mod, p);
        if (mod == 1 && prevMod) {
            return true;
        }
    }
    return mod != 1;
}

bool MillerRabinPrimalityTestLocal(uint64_t p, FixedBlake2Prng& prng, uint32_t niter) {
    if (p == 2 || p == 3 || p == 5) {
        return true;
    }
    if (p < 2 || (p % 2) == 0) {
        return false;
    }

    uint64_t d = p - 1;
    uint32_t s = 0;
    while ((d % 2) == 0) {
        d >>= 1;
        ++s;
    }
    for (uint32_t i = 0; i < niter; ++i) {
        const uint64_t witness = (GenerateUniformMod(prng, p - 3) + 2) % p;
        if (WitnessFunctionLocal(witness, d, s, p)) {
            return false;
        }
    }
    return true;
}

uint64_t PollardRhoFactorizationLocal(uint64_t n, FixedBlake2Prng& prng) {
    if ((n % 2) == 0) {
        return 2;
    }
    uint64_t divisor = 1;
    const uint64_t c = GenerateUniformMod(prng, n);
    uint64_t x = GenerateUniformMod(prng, n);
    uint64_t xx = x;
    do {
        x = ModAdd(ModMul(x, x, n), c, n);
        xx = ModAdd(ModMul(xx, xx, n), c, n);
        xx = ModAdd(ModMul(xx, xx, n), c, n);
        divisor = Gcd((x > xx) ? x - xx : xx - x, n);
    } while (divisor == 1);
    return divisor;
}

void PrimeFactorizeLocal(uint64_t n, std::set<uint64_t>& primeFactors, FixedBlake2Prng& prng) {
    if (n == 0 || n == 1) {
        return;
    }
    if (MillerRabinPrimalityTestLocal(n, prng)) {
        primeFactors.insert(n);
        return;
    }

    const uint64_t divisor = PollardRhoFactorizationLocal(n, prng);
    PrimeFactorizeLocal(divisor, primeFactors, prng);
    PrimeFactorizeLocal(n / divisor, primeFactors, prng);
}

uint64_t FindGeneratorLocal(uint64_t q, FixedBlake2Prng& prng) {
    const uint64_t qm1 = q - 1;
    const uint64_t qm2 = q - 2;
    std::set<uint64_t> primeFactors;
    PrimeFactorizeLocal(qm1, primeFactors, prng);

    uint64_t gen = 0;
    size_t cnt = 0;
    do {
        cnt = 0;
        gen = GenerateUniformMod(prng, qm2) + 1;
        for (auto factor : primeFactors) {
            if (ModPow(gen, qm1 / factor, q) == 1) {
                break;
            }
            ++cnt;
        }
    } while (cnt != primeFactors.size());
    return gen;
}

std::vector<uint64_t> TotientListLocal(uint64_t n) {
    std::vector<uint64_t> result;
    result.reserve(n / 2);
    for (uint64_t i = 1; i < n; ++i) {
        if (Gcd(i, n) == 1) {
            result.push_back(i);
        }
    }
    return result;
}

uint64_t RootOfUnityLocal(uint64_t m, uint64_t modulo, FixedBlake2Prng& prng) {
    if (((modulo - 1) % m) != 0) {
        throw std::runtime_error("invalid root-of-unity modulus/cyclotomic order pair");
    }

    uint64_t result = 1;
    do {
        const uint64_t gen = FindGeneratorLocal(modulo, prng);
        result = ModPow(gen, (modulo - 1) / m, modulo);
    } while (result == 1);

    uint64_t x = ModMul(1, result, modulo);
    uint64_t minRoot = x;
    uint64_t curPowIdx = 1;
    for (auto nextPowIdx : TotientListLocal(m)) {
        const uint64_t diffPow = nextPowIdx - curPowIdx;
        for (uint64_t j = 0; j < diffPow; ++j) {
            x = ModMul(x, result, modulo);
        }
        if (x < minRoot && x != 1) {
            minRoot = x;
        }
        curPowIdx = nextPowIdx;
    }
    return minRoot;
}

uint64_t FirstPrimeLocal(uint32_t nBits, uint64_t m, FixedBlake2Prng& prng) {
    if (nBits >= 63) {
        throw std::runtime_error("local prime generation supports fewer than 63 bits");
    }
    const uint64_t q = uint64_t{1} << nBits;
    const uint64_t r = q % m;
    uint64_t qNew = q + 1 - r;
    if (r > 0) {
        qNew += m;
    }
    while (!MillerRabinPrimalityTestLocal(qNew, prng)) {
        const uint64_t prev = qNew;
        qNew += m;
        if (qNew < prev || qNew < q) {
            throw std::runtime_error("overflow growing prime candidate");
        }
    }
    return qNew;
}

uint64_t LastPrimeLocal(uint32_t nBits, uint64_t m, FixedBlake2Prng& prng) {
    if (nBits >= 63) {
        throw std::runtime_error("local prime generation supports fewer than 63 bits");
    }
    const uint64_t q = uint64_t{1} << nBits;
    const uint64_t r = q % m;
    uint64_t qNew = q + 1 - r;
    if (r < 2) {
        qNew -= m;
    }
    while (!MillerRabinPrimalityTestLocal(qNew, prng)) {
        const uint64_t prev = qNew;
        qNew -= m;
        if (qNew > prev || qNew > q) {
            throw std::runtime_error("overflow shrinking prime candidate");
        }
    }
    if (GetMsb(qNew) != nBits) {
        throw std::runtime_error("local LastPrime returned a prime with the wrong bit length");
    }
    return qNew;
}

uint64_t NextPrimeLocal(uint64_t q, uint64_t m, FixedBlake2Prng& prng) {
    uint64_t qNew = q + m;
    while (!MillerRabinPrimalityTestLocal(qNew, prng)) {
        const uint64_t prev = qNew;
        qNew += m;
        if (qNew < prev || qNew < q) {
            throw std::runtime_error("overflow growing prime candidate");
        }
    }
    return qNew;
}

uint64_t PreviousPrimeLocal(uint64_t q, uint64_t m, FixedBlake2Prng& prng) {
    uint64_t qNew = q - m;
    while (!MillerRabinPrimalityTestLocal(qNew, prng)) {
        const uint64_t prev = qNew;
        qNew -= m;
        if (qNew > prev || qNew > q) {
            throw std::runtime_error("overflow shrinking prime candidate");
        }
    }
    return qNew;
}

uint32_t ProductBitLength(const std::vector<uint64_t>& values, uint32_t begin, uint32_t endInclusive) {
    if (begin > endInclusive || endInclusive >= values.size()) {
        throw std::runtime_error("invalid product bit-length range");
    }
    if (endInclusive - begin < 2) {
        unsigned __int128 product = 1;
        for (uint32_t i = begin; i <= endInclusive; ++i) {
            product *= values[i];
        }
        uint32_t bits = 0;
        while (product != 0) {
            ++bits;
            product >>= 1;
        }
        return bits;
    }

    long double log2Product = 0.0L;
    for (uint32_t i = begin; i <= endInclusive; ++i) {
        log2Product += std::log2(static_cast<long double>(values[i]));
    }
    return static_cast<uint32_t>(std::floor(log2Product)) + 1;
}

class LocalDiscreteGaussian {
public:
    explicit LocalDiscreteGaussian(double stddev) {
        SetStd(stddev);
    }

    int64_t GenerateInt(FixedBlake2Prng& prng) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return GenerateInt(prng, distribution);
    }

    int64_t GenerateInt(FixedBlake2Prng& prng, std::uniform_real_distribution<double>& distribution) const {
        double seed = distribution(prng) - 0.5;
        double tmp  = std::abs(seed) - a_ / 2;
        return (tmp <= 0.0) ? 0 : FindInVector(vals_, tmp) * (seed > 0.0 ? 1 : -1);
    }

private:
    void SetStd(double stddev) {
        stddev_ = stddev;
        if (std::log2(stddev_) > 59) {
            throw std::runtime_error("standard deviation cannot exceed 59 bits");
        }
        if (stddev_ >= 300.0) {
            throw std::runtime_error("EasyFHE local DGG currently supports Peikert sampling only");
        }
        Initialize();
    }

    void Initialize() {
        double M = 12.00610553538285;
        int64_t fin = static_cast<int64_t>(std::ceil(stddev_ * M));
        vals_.resize(fin);
        double variance = 2 * stddev_ * stddev_;
        double cusum    = 0.0;
        for (int64_t x = 1; x <= fin; ++x) {
            vals_[x - 1] = (cusum += std::exp(-(static_cast<double>(x * x) / variance)));
        }
        a_ = 1.0 / (2 * cusum + 1.0);
        for (int64_t x = 0; x < fin; ++x) {
            vals_[x] *= a_;
        }
    }

    static int64_t FindInVector(const std::vector<double>& values, double search) {
        auto lower = std::lower_bound(values.begin(), values.end(), search);
        if (lower != values.end()) {
            return lower - values.begin() + 1;
        }
        throw std::runtime_error("EasyFHE local DGG search value not found");
    }

    double stddev_ = 1.0;
    double a_      = 0.0;
    std::vector<double> vals_;
};

std::vector<int32_t> GenerateTernaryIntVector(FixedBlake2Prng& prng, uint32_t size, uint32_t h) {
    if (h == 0) {
        std::uniform_int_distribution<int32_t> distribution(-1, 1);
        std::vector<int32_t> result(size);
        for (uint32_t i = 0; i < size; ++i) {
            result[i] = distribution(prng);
        }
        return result;
    }

    if (h > size) {
        h = size;
    }

    std::vector<int32_t> result(size);
    std::uniform_int_distribution<int32_t> indexDistribution(0, size - 1);
    std::bernoulli_distribution binaryDistribution(0.5);
    uint32_t counterPlus = 0;
    while ((counterPlus < h / 2 - 1) || (counterPlus > h / 2 + 1)) {
        counterPlus = 0;
        std::fill(result.begin(), result.end(), 0);

        uint32_t i = 0;
        while (i < h) {
            auto randomIndex = indexDistribution(prng);
            if (result[randomIndex] == 0) {
                if (binaryDistribution(prng)) {
                    result[randomIndex] = 1;
                    ++counterPlus;
                }
                else {
                    result[randomIndex] = -1;
                }
                ++i;
            }
        }
    }
    return result;
}

UIntMatrix MatrixFromSignedCoeffVector(const std::vector<uint64_t>& moduli,
                                       const std::vector<uint64_t>& roots,
                                       uint32_t ringDim,
                                       const std::vector<int64_t>& values) {
    if (values.size() != ringDim) {
        throw std::runtime_error("signed coefficient vector does not match ring dimension");
    }
    if (moduli.size() != roots.size()) {
        throw std::runtime_error("moduli/roots size mismatch");
    }

    UIntMatrix result;
    result.reserve(moduli.size());
    for (size_t limb = 0; limb < moduli.size(); ++limb) {
        const uint64_t modulus = moduli[limb];
        std::vector<uint64_t> coeffs(ringDim);
        for (size_t coeffIndex = 0; coeffIndex < values.size(); ++coeffIndex) {
            const auto value = values[coeffIndex];
            if (value >= 0) {
                const uint64_t unsignedValue = static_cast<uint64_t>(value);
                coeffs[coeffIndex] = unsignedValue < modulus ? unsignedValue : unsignedValue % modulus;
            }
            else {
                const uint64_t absValue = static_cast<uint64_t>(-value);
                const uint64_t reduced = absValue < modulus ? absValue : absValue % modulus;
                coeffs[coeffIndex] = reduced == 0 ? 0 : modulus - reduced;
            }
        }
        ForwardNTTInPlace(coeffs, roots[limb], modulus);
        result.emplace_back(std::move(coeffs));
    }
    return result;
}

UIntMatrix MatrixFromSignedCoeffVectorWithTables(const std::vector<uint64_t>& moduli,
                                                 const std::vector<std::vector<uint64_t>>& rootTables,
                                                 const std::vector<std::vector<uint64_t>>& rootShoupTables,
                                                 uint32_t ringDim,
                                                 const std::vector<int64_t>& values) {
    if (values.size() != ringDim) {
        throw std::runtime_error("signed coefficient vector does not match ring dimension");
    }
    if (moduli.size() != rootTables.size() || moduli.size() != rootShoupTables.size()) {
        throw std::runtime_error("moduli/root table size mismatch");
    }

    UIntMatrix result(moduli.size(), std::vector<uint64_t>(ringDim));
    SequentialFor(moduli.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        auto& coeffs = result[limb];
        for (size_t coeffIndex = 0; coeffIndex < values.size(); ++coeffIndex) {
            const auto value = values[coeffIndex];
            if (value >= 0) {
                const uint64_t unsignedValue = static_cast<uint64_t>(value);
                coeffs[coeffIndex] = unsignedValue < modulus ? unsignedValue : unsignedValue % modulus;
            }
            else {
                const uint64_t absValue = static_cast<uint64_t>(-value);
                const uint64_t reduced = absValue < modulus ? absValue : absValue % modulus;
                coeffs[coeffIndex] = reduced == 0 ? 0 : modulus - reduced;
            }
        }
        ForwardNTTInPlaceWithTable(coeffs, rootTables[limb], rootShoupTables[limb], modulus);
    });
    return result;
}

UIntMatrix MatrixFromSignedCoeffVector(const std::vector<uint64_t>& moduli,
                                       const std::vector<uint64_t>& roots,
                                       uint32_t ringDim,
                                       const std::vector<int32_t>& values) {
    std::vector<int64_t> widened(values.begin(), values.end());
    return MatrixFromSignedCoeffVector(moduli, roots, ringDim, widened);
}

void ValidateSameShape(const char* opName,
                       const UIntMatrix& lhs,
                       const UIntMatrix& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(std::string(opName) + " limb-count mismatch");
    }
    for (size_t limb = 0; limb < lhs.size(); ++limb) {
        if (lhs[limb].size() != rhs[limb].size()) {
            throw std::runtime_error(std::string(opName) + " ring-dimension mismatch");
        }
    }
}

UIntMatrix MatrixEvalAdd(const UIntMatrix& lhs, const UIntMatrix& rhs, const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalAdd", lhs, rhs);
    UIntMatrix result = lhs;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalAdd modulus count mismatch");
    }
    for (size_t limb = 0; limb < result.size(); ++limb) {
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            result[limb][coeff] = ModAdd(result[limb][coeff], rhs[limb][coeff], moduli[limb]);
        }
    }
    return result;
}

UIntMatrix MatrixEvalSub(const UIntMatrix& lhs, const UIntMatrix& rhs, const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalSub", lhs, rhs);
    UIntMatrix result = lhs;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalSub modulus count mismatch");
    }
    for (size_t limb = 0; limb < result.size(); ++limb) {
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            result[limb][coeff] = ModSub(result[limb][coeff], rhs[limb][coeff], moduli[limb]);
        }
    }
    return result;
}

UIntMatrix MatrixEvalMul(const UIntMatrix& lhs, const UIntMatrix& rhs, const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalMul", lhs, rhs);
    UIntMatrix result = lhs;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalMul modulus count mismatch");
    }
    SequentialFor(result.size(), [&](size_t limb) {
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            result[limb][coeff] = ModMul(result[limb][coeff], rhs[limb][coeff], moduli[limb]);
        }
    });
    return result;
}

UIntMatrix MatrixShoupPrecon(const UIntMatrix& value, const std::vector<uint64_t>& moduli) {
    if (value.size() != moduli.size()) {
        throw std::runtime_error("MatrixShoupPrecon modulus count mismatch");
    }
    UIntMatrix result = value;
    SequentialFor(result.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        for (auto& coeff : result[limb]) {
            coeff = ShoupPrecon(coeff, modulus);
        }
    });
    return result;
}

UIntMatrix MatrixEvalMulPrecon(const UIntMatrix& lhs,
                               const UIntMatrix& rhs,
                               const UIntMatrix& rhsPrecon,
                               const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalMulPrecon/lhs/rhs", lhs, rhs);
    ValidateSameShape("MatrixEvalMulPrecon/rhs/precon", rhs, rhsPrecon);
    UIntMatrix result = lhs;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalMulPrecon modulus count mismatch");
    }
    SequentialFor(result.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            result[limb][coeff] =
                ModMulPrecon(result[limb][coeff], rhs[limb][coeff], rhsPrecon[limb][coeff], modulus);
        }
    });
    return result;
}

UIntMatrix MatrixEvalScale(const UIntMatrix& value, uint64_t scalar, const std::vector<uint64_t>& moduli) {
    UIntMatrix result = value;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalScale modulus count mismatch");
    }
    for (size_t limb = 0; limb < result.size(); ++limb) {
        const uint64_t scalarMod = scalar % moduli[limb];
        for (auto& coeff : result[limb]) {
            coeff = ModMul(coeff, scalarMod, moduli[limb]);
        }
    }
    return result;
}

UIntMatrix MatrixEvalScalePrecon(const UIntMatrix& value,
                                 uint64_t scalar,
                                 const std::vector<uint64_t>& scalarPrecon,
                                 const std::vector<uint64_t>& moduli) {
    UIntMatrix result = value;
    if (result.size() != moduli.size() || result.size() != scalarPrecon.size()) {
        throw std::runtime_error("MatrixEvalScalePrecon modulus count mismatch");
    }
    SequentialFor(result.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        const uint64_t scalarMod = scalar % modulus;
        const uint64_t scalarShoup = scalarPrecon[limb];
        if (scalarMod == 1) {
            return;
        }
        for (auto& coeff : result[limb]) {
            coeff = ModMulPrecon(coeff, scalarMod, scalarShoup, modulus);
        }
    });
    return result;
}

UIntMatrix MatrixEvalScaledSubMul(const UIntMatrix& e,
                                  uint64_t scalar,
                                  const UIntMatrix& a,
                                  const UIntMatrix& s,
                                  const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalScaledSubMul/e/a", e, a);
    ValidateSameShape("MatrixEvalScaledSubMul/e/s", e, s);
    UIntMatrix result = e;
    if (result.size() != moduli.size()) {
        throw std::runtime_error("MatrixEvalScaledSubMul modulus count mismatch");
    }
    SequentialFor(result.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        const uint64_t scalarMod = scalar % modulus;
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            const uint64_t scaled = scalarMod == 1 ? result[limb][coeff] : ModMul(result[limb][coeff], scalarMod, modulus);
            result[limb][coeff] = ModSub(scaled, ModMul(a[limb][coeff], s[limb][coeff], modulus), modulus);
        }
    });
    return result;
}

UIntMatrix MatrixEvalScaledSubMulPrecon(const UIntMatrix& e,
                                        uint64_t scalar,
                                        const std::vector<uint64_t>& scalarPrecon,
                                        const UIntMatrix& a,
                                        const UIntMatrix& s,
                                        const UIntMatrix& sPrecon,
                                        const std::vector<uint64_t>& moduli) {
    ValidateSameShape("MatrixEvalScaledSubMulPrecon/e/a", e, a);
    ValidateSameShape("MatrixEvalScaledSubMulPrecon/e/s", e, s);
    ValidateSameShape("MatrixEvalScaledSubMulPrecon/s/precon", s, sPrecon);
    UIntMatrix result = e;
    if (result.size() != moduli.size() || result.size() != scalarPrecon.size()) {
        throw std::runtime_error("MatrixEvalScaledSubMulPrecon modulus count mismatch");
    }
    SequentialFor(result.size(), [&](size_t limb) {
        const uint64_t modulus = moduli[limb];
        const uint64_t scalarMod = scalar % modulus;
        const uint64_t scalarShoup = scalarPrecon[limb];
        for (size_t coeff = 0; coeff < result[limb].size(); ++coeff) {
            const uint64_t scaled = scalarMod == 1 ?
                result[limb][coeff] :
                ModMulPrecon(result[limb][coeff], scalarMod, scalarShoup, modulus);
            const uint64_t product =
                ModMulPrecon(a[limb][coeff], s[limb][coeff], sPrecon[limb][coeff], modulus);
            result[limb][coeff] = ModSub(scaled, product, modulus);
        }
    });
    return result;
}

UIntMatrix MatrixEvalScaleByLimb(const UIntMatrix& value,
                                 const std::vector<uint64_t>& scalars,
                                 const std::vector<uint64_t>& moduli) {
    UIntMatrix result = value;
    if (result.size() != moduli.size() || result.size() != scalars.size()) {
        throw std::runtime_error("MatrixEvalScaleByLimb modulus/scalar count mismatch");
    }
    for (size_t limb = 0; limb < result.size(); ++limb) {
        const uint64_t scalarMod = scalars[limb] % moduli[limb];
        for (auto& coeff : result[limb]) {
            coeff = ModMul(coeff, scalarMod, moduli[limb]);
        }
    }
    return result;
}

uint32_t FindAutomorphismIndex2nComplexLocal(int32_t index, uint32_t cyclOrder) {
    if (index == 0) {
        return 1;
    }
    if (index == static_cast<int32_t>(cyclOrder) - 1) {
        return static_cast<uint32_t>(index);
    }
    if ((cyclOrder == 0) || ((cyclOrder & (cyclOrder - 1)) != 0)) {
        throw std::runtime_error("FindAutomorphismIndex2nComplexLocal expects a power-of-two cyclotomic order");
    }

    const uint64_t generator = index < 0 ? ModInverse(5, cyclOrder) : 5;
    uint64_t result = generator;
    const uint32_t steps = static_cast<uint32_t>(std::abs(index));
    for (uint32_t j = 1; j < steps; ++j) {
        result = (result * generator) & (cyclOrder - 1);
    }
    return static_cast<uint32_t>(result);
}

std::vector<uint32_t> PrecomputeAutoMapLocal(uint32_t ringDim, uint32_t autoIndex) {
    const uint32_t cyclOrder = ringDim << 1;
    const uint32_t logm = GetMsb(cyclOrder) - 1;
    const uint32_t logn = logm - 1;
    std::vector<uint32_t> precomp(ringDim);
    for (uint32_t j = 0; j < ringDim; ++j) {
        const uint32_t jTmp = (j << 1) + 1;
        const uint32_t idx = (((jTmp * autoIndex) - (((jTmp * autoIndex) >> logm) << logm)) >> 1);
        const uint32_t jrev = ReverseBitsLocal(j, logn);
        const uint32_t idxrev = ReverseBitsLocal(idx, logn);
        precomp[jrev] = idxrev;
    }
    return precomp;
}

std::vector<uint32_t> InvertAutoMapLocal(const std::vector<uint32_t>& autoMap) {
    std::vector<uint32_t> inverse(autoMap.size());
    for (uint32_t i = 0; i < autoMap.size(); ++i) {
        const uint32_t mapped = autoMap[i];
        if (mapped >= autoMap.size()) {
            throw std::runtime_error("auto map contains an out-of-range index");
        }
        inverse[mapped] = i;
    }
    return inverse;
}

uint32_t NormalizeRotationIndexLocal(int32_t rotationIndex, uint32_t ringDim) {
    const int64_t normalized = rotationIndex < 0 ? static_cast<int64_t>(ringDim / 2) + rotationIndex : rotationIndex;
    if (normalized < 0 || normalized > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("normalized rotation index is outside uint32 range");
    }
    return static_cast<uint32_t>(normalized);
}

UIntMatrix AutomorphismTransformEval(const UIntMatrix& value, uint32_t autoIndex, uint32_t ringDim) {
    const auto precomp = PrecomputeAutoMapLocal(ringDim, autoIndex);
    UIntMatrix result = value;
    for (size_t limb = 0; limb < value.size(); ++limb) {
        if (value[limb].size() != ringDim) {
            throw std::runtime_error("AutomorphismTransformEval ring dimension mismatch");
        }
        for (uint32_t coeff = 0; coeff < ringDim; ++coeff) {
            result[limb][coeff] = value[limb][precomp[coeff]];
        }
    }
    return result;
}

uint64_t FitInt64ToNativeModulus(int64_t value, int64_t bigBound, uint64_t modulus) {
    const uint64_t encoded = static_cast<uint64_t>(value);
    if (encoded > static_cast<uint64_t>(bigBound >> 1)) {
        const uint64_t diffMod = (static_cast<uint64_t>(bigBound) - modulus) % modulus;
        const uint64_t valueMod = encoded % modulus;
        return ModSub(valueMod, diffMod, modulus);
    }
    return encoded % modulus;
}

UIntMatrix MatrixFromCkksEncodeCoeffs(const std::vector<uint64_t>& moduli,
                                      const std::vector<uint64_t>& roots,
                                      uint32_t ringDim,
                                      const std::vector<int64_t>& coeffs,
                                      int64_t bigBound,
                                      int64_t logApprox,
                                      uint64_t intPowP,
                                      uint32_t noiseScaleDeg) {
    const uint32_t dslots  = static_cast<uint32_t>(coeffs.size());
    if (dslots == 0 || ringDim % dslots != 0) {
        throw std::runtime_error("invalid CKKS encode slot count for native encoder");
    }
    if (moduli.size() != roots.size()) {
        throw std::runtime_error("encode moduli/roots size mismatch");
    }
    const uint32_t gap = ringDim / dslots;

    UIntMatrix result;
    result.reserve(moduli.size());
    for (size_t towerIndex = 0; towerIndex < moduli.size(); ++towerIndex) {
        const uint64_t modValue = moduli[towerIndex];
        std::vector<uint64_t> nativeCoeffs(ringDim, 0);
        for (uint32_t i = 0; i < dslots; ++i) {
            nativeCoeffs[gap * i] = FitInt64ToNativeModulus(coeffs[i], bigBound, modValue);
        }

        ForwardNTTInPlace(nativeCoeffs, roots[towerIndex], modValue);

        uint64_t correction = 1;
        if (noiseScaleDeg > 1) {
            correction = ModMul(correction, ModPow(intPowP % modValue, noiseScaleDeg - 1, modValue), modValue);
        }
        if (logApprox > 0) {
            correction = ModMul(correction, ModPow(2, static_cast<uint64_t>(logApprox), modValue), modValue);
        }
        if (correction != 1) {
            for (auto& value : nativeCoeffs) {
                value = ModMul(value, correction, modValue);
            }
        }
        result.emplace_back(std::move(nativeCoeffs));
    }
    return result;
}

UIntMatrix EncodeCkksPackedNative(const std::vector<uint64_t>& moduli,
                                const std::vector<uint64_t>& roots,
                                uint32_t ringDim,
                                const std::vector<double>& values,
                                uint32_t plaintextModulusBits,
                                uint32_t noiseScaleDeg,
                                uint32_t level,
                                uint32_t slots) {
    if (level != 0) {
        throw std::runtime_error("native CKKS encoder currently supports level=0 only");
    }
    if (slots < values.size()) {
        throw std::runtime_error("native CKKS encoder slot count is smaller than the input vector");
    }

    std::vector<std::complex<double>> inverse(values.begin(), values.end());
    inverse.resize(slots);
    FFTSpecialInvNative(inverse, ringDim * 2);

    const double scalingFactor = std::pow(2.0, plaintextModulusBits);
    int32_t logc = std::numeric_limits<int32_t>::min();
    for (uint32_t i = 0; i < slots; ++i) {
        inverse[i] *= scalingFactor;
        if (inverse[i].real() != 0.0) {
            logc = std::max<int32_t>(logc, static_cast<int32_t>(std::ceil(std::log2(std::abs(inverse[i].real())))));
        }
        if (inverse[i].imag() != 0.0) {
            logc = std::max<int32_t>(logc, static_cast<int32_t>(std::ceil(std::log2(std::abs(inverse[i].imag())))));
        }
    }
    logc = (logc == std::numeric_limits<int32_t>::min()) ? 0 : logc;
    if (logc < 0) {
        throw std::runtime_error("native CKKS encoder scaling factor is too small");
    }

    constexpr int32_t maxBitsInWord = 60;
    const int32_t logValid          = (logc <= maxBitsInWord) ? logc : maxBitsInWord;
    const int32_t logApprox         = logc - logValid;
    const double approxFactor       = std::pow(2.0, logApprox);

    std::vector<int64_t> temp(2 * slots);
    const int64_t maxBitValue = Max64BitValueLocal();
    for (uint32_t i = 0; i < slots; ++i) {
        const double dre = inverse[i].real() / approxFactor;
        const double dim = inverse[i].imag() / approxFactor;
        if (Is64BitOverflowLocal(dre) || Is64BitOverflowLocal(dim)) {
            throw std::runtime_error("native CKKS encoder overflow; try decreasing scaling factor");
        }
        const int64_t re = std::llround(dre);
        const int64_t im = std::llround(dim);
        temp[i]          = (re < 0) ? maxBitValue + re : re;
        temp[i + slots]  = (im < 0) ? maxBitValue + im : im;
    }

    const uint64_t intPowP = static_cast<uint64_t>(std::llround(scalingFactor));
    return MatrixFromCkksEncodeCoeffs(moduli, roots, ringDim, temp, maxBitValue, logApprox, intPowP, noiseScaleDeg);
}

UIntMatrix GenerateTernaryMatrix(FixedBlake2Prng& prng,
                                 const std::vector<uint64_t>& moduli,
                                 const std::vector<uint64_t>& roots,
                                 uint32_t ringDim,
                                 uint32_t h) {
    return MatrixFromSignedCoeffVector(moduli, roots, ringDim, GenerateTernaryIntVector(prng, ringDim, h));
}

UIntMatrix GenerateGaussianMatrix(FixedBlake2Prng& prng,
                                  const LocalDiscreteGaussian& dgg,
                                  const std::vector<uint64_t>& moduli,
                                  const std::vector<std::vector<uint64_t>>& rootTables,
                                  const std::vector<std::vector<uint64_t>>& rootShoupTables,
                                  uint32_t ringDim) {
    std::vector<int64_t> values(ringDim);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (uint32_t i = 0; i < ringDim; ++i) {
        values[i] = dgg.GenerateInt(prng, distribution);
    }
    return MatrixFromSignedCoeffVectorWithTables(moduli, rootTables, rootShoupTables, ringDim, values);
}

std::vector<uint64_t> GenerateUniformVector(FixedBlake2Prng& prng, uint64_t modulus, uint32_t ringDim) {
    if (modulus == 0) {
        throw std::runtime_error("cannot sample modulo zero");
    }
    constexpr uint32_t chunkWidth = std::numeric_limits<uint32_t>::digits;
    const uint32_t chunksPerValue = (GetMsb(modulus) - 1) / chunkWidth;
    const uint32_t shiftChunk     = chunksPerValue * chunkWidth;
    const uint32_t bound          = static_cast<uint32_t>(modulus >> shiftChunk);
    const uint32_t range          = bound + 1;
    uint32_t mask                 = std::numeric_limits<uint32_t>::max();
    if (range != 0 && range != 1) {
        uint32_t width = 32 - static_cast<uint32_t>(__builtin_clz(range)) - 1;
        if ((range & (std::numeric_limits<uint32_t>::max() >> (32 - width))) != 0) {
            ++width;
        }
        mask = width == 32 ? std::numeric_limits<uint32_t>::max() :
                              (std::numeric_limits<uint32_t>::max() >> (32 - width));
    }

    auto drawBounded = [&]() {
        if (range == 0) {
            return prng();
        }
        if (range == 1) {
            return uint32_t{0};
        }
        uint32_t value = 0;
        do {
            value = prng() & mask;
        } while (value >= range);
        return value;
    };

    std::vector<uint64_t> limb(ringDim);
    for (auto& value : limb) {
        while (true) {
            uint64_t candidate = 0;
            for (uint32_t i = 0, shift = 0; i < chunksPerValue; ++i, shift += chunkWidth) {
                candidate += uint64_t{prng()} << shift;
            }
            candidate += uint64_t{drawBounded()} << shiftChunk;
            if (candidate < modulus) {
                value = candidate;
                break;
            }
        }
    }
    return limb;
}

std::vector<uint64_t> MatrixRowFromSignedCoeffVectorWithTables(
    uint64_t modulus,
    const std::vector<uint64_t>& rootTable,
    const std::vector<uint64_t>& rootShoupTable,
    uint32_t ringDim,
    const std::vector<int64_t>& values) {
    if (values.size() != ringDim) {
        throw std::runtime_error("signed coefficient vector does not match ring dimension");
    }
    std::vector<uint64_t> coeffs(ringDim);
    for (size_t coeffIndex = 0; coeffIndex < values.size(); ++coeffIndex) {
        const auto value = values[coeffIndex];
        if (value >= 0) {
            const uint64_t unsignedValue = static_cast<uint64_t>(value);
            coeffs[coeffIndex] = unsignedValue < modulus ? unsignedValue : unsignedValue % modulus;
        }
        else {
            const uint64_t absValue = static_cast<uint64_t>(-value);
            const uint64_t reduced = absValue < modulus ? absValue : absValue % modulus;
            coeffs[coeffIndex] = reduced == 0 ? 0 : modulus - reduced;
        }
    }
    ForwardNTTInPlaceWithTable(coeffs, rootTable, rootShoupTable, modulus);
    return coeffs;
}

UIntMatrix GenerateUniformMatrix(FixedBlake2Prng& prng, const std::vector<uint64_t>& moduli, uint32_t ringDim) {
    UIntMatrix result;
    result.reserve(moduli.size());
    for (auto modulus : moduli) {
        result.emplace_back(GenerateUniformVector(prng, modulus, ringDim));
    }
    return result;
}

void ConsumeKeyTag(FixedBlake2Prng& prng) {
    std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
    for (size_t i = 0; i < 4; ++i) {
        (void)distribution(prng);
    }
}

struct ManualKeygenResult {
    UIntMatrix sk;
    UIntMatrix pkB;
    UIntMatrix pkA;
    UIntMatrix pkE;
    std::vector<int64_t> skCoeff;
};

struct ManualEncryptResult {
    UIntMatrix v;
    UIntMatrix e0;
    UIntMatrix e1;
    UIntMatrix ct0Zero;
    UIntMatrix ct1Zero;
    UIntMatrix ct0;
    UIntMatrix ct1;
};

struct ManualEvalMultKeyResult {
    UIntMatrix skSquared;
    UIntMatrix skExt;
    UIntMatrix pModq;
    std::vector<UIntMatrix> keyB;
    std::vector<UIntMatrix> keyA;
    std::vector<UIntMatrix> keyE;
};

struct ManualSwitchKeyResult {
    std::vector<UIntMatrix> keyB;
    std::vector<UIntMatrix> keyA;
    std::vector<UIntMatrix> keyE;
};

struct SwitchPartProfile {
    double uniform = 0.0;
    double gaussian = 0.0;
    double noiseScale = 0.0;
    double mulSub = 0.0;
    double pInjection = 0.0;
};

struct ManualRotationKey {
    uint32_t autoIndex = 0;
    ManualSwitchKeyResult key;
};

struct RotationKeyTensorResult {
    uint32_t autoIndex = 0;
    uint32_t rotationIndex = 0;
    at::Tensor keyB;
    at::Tensor keyA;
    at::Tensor autoMap;
    at::Tensor inverseAutoMap;
    uint32_t limb = 0;
    bool trimmed = false;
};

struct RotationRandomRows {
    std::vector<UIntMatrix> aByPart;
    std::vector<UIntMatrix> eByPart;
};

struct RotationRandomCache {
    std::map<uint32_t, RotationRandomRows> rowsByShape;
};

RotationKeyTensorResult RotationKeyToTensorResult(const ManualRotationKey& rotKey,
                                                  const std::map<uint32_t, uint32_t>& rotationTrimLimbsByAutoIndex,
                                                  uint32_t qLimbs,
                                                  uint32_t pLimbs,
                                                  uint32_t dnum) {
    auto trimIt       = rotationTrimLimbsByAutoIndex.find(rotKey.autoIndex);
    const bool trimKey = trimIt != rotationTrimLimbsByAutoIndex.end();
    RotationKeyTensorResult item;
    item.autoIndex = rotKey.autoIndex;
    if (trimKey) {
        const uint32_t limb = trimIt->second;
        item.keyB = TrimmedSwitchKeyToTensor(rotKey.key.keyB, limb, qLimbs, pLimbs, dnum);
        item.keyA = TrimmedSwitchKeyToTensor(rotKey.key.keyA, limb, qLimbs, pLimbs, dnum);
        item.limb = limb;
        item.trimmed = true;
    }
    else {
        item.keyB = MatrixVectorToTensor(rotKey.key.keyB);
        item.keyA = MatrixVectorToTensor(rotKey.key.keyA);
    }
    return item;
}

struct NativeCkksParams {
    std::vector<uint64_t> moduliQ;
    std::vector<uint64_t> rootsQ;
    std::vector<uint64_t> moduliP;
    std::vector<uint64_t> rootsP;
    std::vector<uint64_t> moduliQP;
    std::vector<uint64_t> rootsQP;
    std::vector<uint64_t> pModq;
    std::vector<std::vector<uint64_t>> rootTablesQ;
    std::vector<std::vector<uint64_t>> rootTablesQP;
    std::vector<std::vector<uint64_t>> rootShoupTablesQ;
    std::vector<std::vector<uint64_t>> rootShoupTablesQP;
    uint32_t ringDim = 0;
    uint32_t cyclOrder = 0;
    uint32_t numPartQ = 0;
    uint32_t numPerPartQ = 0;
};

struct NativeRuntimeConfig {
    SecretKeyDistLocal secretKeyDist = SecretKeyDistLocal::SparseTernary;
    double dggStd              = 3.19;
    uint64_t noiseScale        = 1;
};

struct NativeSamplerRequest {
    int64_t logN = 12;
    int64_t depth = 2;
    int64_t dcrtBits = 50;
    int64_t firstMod = 60;
    int64_t dnum = 3;
    std::string secretKeyDist = "SPARSE_TERNARY";
    bool includeEvalMultKey = false;
    bool includeEncryptTrace = true;
    int64_t scaleDeg = 1;
    int64_t level = 0;
    int64_t slots = 0;
    std::vector<std::vector<int32_t>> rotationIndexGroups;
    std::map<uint32_t, uint32_t> rotationTrimLimbsByAutoIndex;
    std::string rotationRandomMode = "fresh";
    std::string randomMode = "sequential";
    double standardDeviation = 3.19;
    uint64_t noiseScale = 1;
};

NativeRuntimeConfig GetRuntimeConfig(const NativeSamplerRequest& request) {
    NativeRuntimeConfig runtime;
    runtime.secretKeyDist = ParseSecretKeyDist(request.secretKeyDist);
    runtime.dggStd = request.standardDeviation;
    runtime.noiseScale = request.noiseScale;
    return runtime;
}

NativeCkksParams GenerateNativeCkksParams(const NativeSamplerRequest& request, FixedBlake2Prng& prng) {
    const auto logN             = static_cast<uint32_t>(request.logN);
    const auto depth            = static_cast<uint32_t>(request.depth);
    const auto dcrtBits         = static_cast<uint32_t>(request.dcrtBits);
    const auto firstMod         = static_cast<uint32_t>(request.firstMod);
    const auto dnum             = static_cast<uint32_t>(request.dnum);

    const uint32_t ringDim    = uint32_t{1} << logN;
    const uint32_t cyclOrder  = 2 * ringDim;
    const uint32_t numPrimesQ = depth + 1;

    std::vector<uint64_t> moduliQ(numPrimesQ);
    std::vector<uint64_t> rootsQ(numPrimesQ);
    uint64_t q = FirstPrimeLocal(dcrtBits, cyclOrder, prng);
    moduliQ[numPrimesQ - 1] = q;
    rootsQ[numPrimesQ - 1]  = RootOfUnityLocal(cyclOrder, moduliQ[numPrimesQ - 1], prng);

    uint64_t maxPrime = q;
    if (numPrimesQ > 1) {
        uint64_t qPrev = q;
        uint64_t qNext = q;
        for (int64_t i = static_cast<int64_t>(numPrimesQ) - 2, cnt = 0; i >= 1; --i, ++cnt) {
            if ((cnt % 2) == 0) {
                qPrev            = PreviousPrimeLocal(qPrev, cyclOrder, prng);
                moduliQ[size_t(i)] = qPrev;
            }
            else {
                qNext            = NextPrimeLocal(qNext, cyclOrder, prng);
                moduliQ[size_t(i)] = qNext;
            }
            if (moduliQ[size_t(i)] > maxPrime) {
                maxPrime = moduliQ[size_t(i)];
            }
            rootsQ[size_t(i)] = RootOfUnityLocal(cyclOrder, moduliQ[size_t(i)], prng);
        }
    }

    if (firstMod == dcrtBits) {
        moduliQ[0] = NextPrimeLocal(maxPrime, cyclOrder, prng);
    }
    else {
        moduliQ[0] = LastPrimeLocal(firstMod, cyclOrder, prng);
        if (std::find(moduliQ.begin() + 1, moduliQ.end(), moduliQ[0]) != moduliQ.end()) {
            moduliQ[0] = NextPrimeLocal(maxPrime, cyclOrder, prng);
        }
    }
    rootsQ[0] = RootOfUnityLocal(cyclOrder, moduliQ[0], prng);

    const uint32_t numPartQ   = std::min<uint32_t>(std::max<uint32_t>(dnum, 1), numPrimesQ);
    const uint32_t numPerPart = static_cast<uint32_t>(std::ceil(static_cast<double>(numPrimesQ) / numPartQ));
    uint32_t maxBits = 0;
    for (uint32_t part = 0; part < numPartQ; ++part) {
        const uint32_t startTower = part * numPerPart;
        const uint32_t endTower   = std::min<uint32_t>((part + 1) * numPerPart - 1, numPrimesQ - 1);
        maxBits = std::max<uint32_t>(maxBits, ProductBitLength(moduliQ, startTower, endTower));
    }
    const uint32_t auxBits    = 60;
    const uint32_t numPrimesP = static_cast<uint32_t>(std::ceil(static_cast<double>(maxBits) / auxBits));
    const uint64_t primeStep  = 2 * ringDim;
    std::vector<uint64_t> moduliP(numPrimesP);
    std::vector<uint64_t> rootsP(numPrimesP);
    uint64_t pPrev = FirstPrimeLocal(auxBits, primeStep, prng);
    for (uint32_t i = 0; i < numPrimesP; ++i) {
        bool foundInQ = false;
        do {
            moduliP[i] = PreviousPrimeLocal(pPrev, primeStep, prng);
            foundInQ   = std::find(moduliQ.begin(), moduliQ.end(), moduliP[i]) != moduliQ.end();
            pPrev      = moduliP[i];
        } while (foundInQ);
        rootsP[i] = RootOfUnityLocal(cyclOrder, moduliP[i], prng);
        pPrev     = moduliP[i];
    }

    std::vector<uint64_t> moduliQP;
    std::vector<uint64_t> rootsQP;
    moduliQP.reserve(moduliQ.size() + moduliP.size());
    rootsQP.reserve(rootsQ.size() + rootsP.size());
    moduliQP.insert(moduliQP.end(), moduliQ.begin(), moduliQ.end());
    moduliQP.insert(moduliQP.end(), moduliP.begin(), moduliP.end());
    rootsQP.insert(rootsQP.end(), rootsQ.begin(), rootsQ.end());
    rootsQP.insert(rootsQP.end(), rootsP.begin(), rootsP.end());

    std::vector<uint64_t> pModq(moduliQ.size(), 1);
    for (size_t limb = 0; limb < moduliQ.size(); ++limb) {
        for (auto p : moduliP) {
            pModq[limb] = ModMul(pModq[limb], p % moduliQ[limb], moduliQ[limb]);
        }
    }

    std::vector<std::vector<uint64_t>> rootTablesQ;
    std::vector<std::vector<uint64_t>> rootTablesQP;
    std::vector<std::vector<uint64_t>> rootShoupTablesQ;
    std::vector<std::vector<uint64_t>> rootShoupTablesQP;
    rootTablesQ.reserve(rootsQ.size());
    rootTablesQP.reserve(rootsQP.size());
    for (size_t i = 0; i < rootsQ.size(); ++i) {
        rootTablesQ.push_back(MakeRootOfUnityTable(rootsQ[i], ringDim, moduliQ[i]));
        rootShoupTablesQ.push_back(MakeShoupTable(rootTablesQ.back(), moduliQ[i]));
    }
    for (size_t i = 0; i < rootsQP.size(); ++i) {
        rootTablesQP.push_back(MakeRootOfUnityTable(rootsQP[i], ringDim, moduliQP[i]));
        rootShoupTablesQP.push_back(MakeShoupTable(rootTablesQP.back(), moduliQP[i]));
    }

    return NativeCkksParams{std::move(moduliQ),
                            std::move(rootsQ),
                            std::move(moduliP),
                            std::move(rootsP),
                            std::move(moduliQP),
                            std::move(rootsQP),
                            std::move(pModq),
                            std::move(rootTablesQ),
                            std::move(rootTablesQP),
                            std::move(rootShoupTablesQ),
                            std::move(rootShoupTablesQP),
                            ringDim,
                            cyclOrder,
                            numPartQ,
                            numPerPart};
}

UIntMatrix DecryptionPhase(const UIntMatrix& ct0,
                           const UIntMatrix& ct1,
                           const UIntMatrix& sk,
                           const std::vector<uint64_t>& moduli) {
    return MatrixEvalAdd(ct0, MatrixEvalMul(ct1, sk, moduli), moduli);
}


ManualKeygenResult ManualKeyGen(const NativeRuntimeConfig& runtime,
                                const NativeCkksParams& nativeParams,
                                FixedBlake2Prng& prng) {
    LocalDiscreteGaussian dgg(runtime.dggStd);

    std::vector<int64_t> skCoeff(nativeParams.ringDim);
    switch (runtime.secretKeyDist) {
        case SecretKeyDistLocal::Gaussian:
            for (uint32_t i = 0; i < nativeParams.ringDim; ++i) {
                skCoeff[i] = dgg.GenerateInt(prng);
            }
            break;
        case SecretKeyDistLocal::UniformTernary:
            {
                auto ternary = GenerateTernaryIntVector(prng, nativeParams.ringDim, 0);
                std::copy(ternary.begin(), ternary.end(), skCoeff.begin());
            }
            break;
        case SecretKeyDistLocal::SparseTernary:
            {
                auto ternary = GenerateTernaryIntVector(prng, nativeParams.ringDim, 192);
                std::copy(ternary.begin(), ternary.end(), skCoeff.begin());
            }
            break;
        default:
            throw std::runtime_error("unsupported secret key distribution for manual keygen");
    }

    UIntMatrix s = MatrixFromSignedCoeffVectorWithTables(
        nativeParams.moduliQ, nativeParams.rootTablesQ, nativeParams.rootShoupTablesQ, nativeParams.ringDim, skCoeff);
    UIntMatrix a = GenerateUniformMatrix(prng, nativeParams.moduliQ, nativeParams.ringDim);
    UIntMatrix e = GenerateGaussianMatrix(
        prng, dgg, nativeParams.moduliQ, nativeParams.rootTablesQ, nativeParams.rootShoupTablesQ, nativeParams.ringDim);
    UIntMatrix eScaled = MatrixEvalScale(e, runtime.noiseScale, nativeParams.moduliQ);
    UIntMatrix b = MatrixEvalSub(eScaled, MatrixEvalMul(a, s, nativeParams.moduliQ), nativeParams.moduliQ);

    ConsumeKeyTag(prng);

    return ManualKeygenResult{std::move(s), std::move(b), std::move(a), std::move(eScaled), std::move(skCoeff)};
}

ManualSwitchKeyResult ManualHybridKeySwitchGen(const NativeRuntimeConfig& runtime,
                                               const UIntMatrix& sOld,
                                               const UIntMatrix& sNewExt,
                                               const UIntMatrix& sNewExtShoup,
                                               const NativeCkksParams& nativeParams,
                                               FixedBlake2Prng& prng,
                                               const NativeSamplingContext& sampling,
                                               const std::string& switchLabel,
                                               uint64_t switchDomain,
                                               bool keepKeyE = true) {
    const double switchStart = NowSeconds();
    if (gProfile.enabled) {
        ++gProfile.switchCalls;
    }
    // HYBRID key switching uses a default per-part Gaussian generator, so the
    // eval-key noise follows the DGG default rather than the context parameter.
    LocalDiscreteGaussian dgg(1.0);

    std::vector<UIntMatrix> keyB(nativeParams.numPartQ);
    std::vector<UIntMatrix> keyA(nativeParams.numPartQ);
    std::vector<UIntMatrix> keyE(nativeParams.numPartQ);
    std::vector<SwitchPartProfile> partProfiles(nativeParams.numPartQ);
    const std::vector<uint64_t> noiseScaleShoup = MakeScalarShoupTable(runtime.noiseScale, nativeParams.moduliQP);

    auto generatePart = [&](uint32_t part, FixedBlake2Prng& partPrng) {
        SwitchPartProfile localProfile;
        double stageStart = NowSeconds();
        UIntMatrix a = GenerateUniformMatrix(partPrng, nativeParams.moduliQP, nativeParams.ringDim);
        if (gProfile.enabled) {
            localProfile.uniform += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }
        UIntMatrix e = GenerateGaussianMatrix(
            partPrng,
            dgg,
            nativeParams.moduliQP,
            nativeParams.rootTablesQP,
            nativeParams.rootShoupTablesQP,
            nativeParams.ringDim);
        if (gProfile.enabled) {
            localProfile.gaussian += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }
        UIntMatrix eScaled;
        UIntMatrix b;
        if (keepKeyE) {
            eScaled = MatrixEvalScalePrecon(e, runtime.noiseScale, noiseScaleShoup, nativeParams.moduliQP);
            if (gProfile.enabled) {
                localProfile.noiseScale += NowSeconds() - stageStart;
                stageStart = NowSeconds();
            }
            b = MatrixEvalSub(
                eScaled, MatrixEvalMulPrecon(a, sNewExt, sNewExtShoup, nativeParams.moduliQP), nativeParams.moduliQP);
        }
        else {
            b = MatrixEvalScaledSubMulPrecon(
                e, runtime.noiseScale, noiseScaleShoup, a, sNewExt, sNewExtShoup, nativeParams.moduliQP);
        }
        if (gProfile.enabled) {
            localProfile.mulSub += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        const uint32_t startPartIdx = nativeParams.numPerPartQ * part;
        const uint32_t endPartIdx = std::min<uint32_t>(startPartIdx + nativeParams.numPerPartQ,
                                                       static_cast<uint32_t>(nativeParams.moduliQ.size()));
        for (uint32_t limb = startPartIdx; limb < endPartIdx; ++limb) {
            const uint64_t modulus = nativeParams.moduliQ[limb];
            const uint64_t scalar = nativeParams.pModq[limb] % modulus;
            const uint64_t scalarShoup = ShoupPrecon(scalar, modulus);
            for (uint32_t coeff = 0; coeff < nativeParams.ringDim; ++coeff) {
                const uint64_t term = ModMulPrecon(sOld[limb][coeff], scalar, scalarShoup, modulus);
                b[limb][coeff] = ModAdd(b[limb][coeff], term, modulus);
            }
        }
        if (gProfile.enabled) {
            localProfile.pInjection += NowSeconds() - stageStart;
        }

        keyA[part] = std::move(a);
        keyB[part] = std::move(b);
        if (keepKeyE) {
            keyE[part] = std::move(eScaled);
        }
        partProfiles[part] = localProfile;
    };

    if (sampling.IsParallelDeterministic()) {
        SequentialFor(nativeParams.numPartQ, [&](size_t partIndex) {
            auto partPrng = sampling.SwitchPartPrng(switchLabel, switchDomain, partIndex);
            generatePart(static_cast<uint32_t>(partIndex), partPrng);
        });
    }
    else {
        for (uint32_t part = 0; part < nativeParams.numPartQ; ++part) {
            generatePart(part, prng);
        }
    }

    if (gProfile.enabled) {
        gProfile.switchParts += nativeParams.numPartQ;
        for (const auto& profile : partProfiles) {
            gProfile.uniform += profile.uniform;
            gProfile.gaussian += profile.gaussian;
            gProfile.noiseScale += profile.noiseScale;
            gProfile.mulSub += profile.mulSub;
            gProfile.pInjection += profile.pInjection;
        }
    }

    if (gProfile.enabled) {
        gProfile.switchTotal += NowSeconds() - switchStart;
    }
    return ManualSwitchKeyResult{std::move(keyB), std::move(keyA), std::move(keyE)};
}

std::vector<uint32_t> FullRotationRandomSourceRows(const NativeCkksParams& nativeParams) {
    std::vector<uint32_t> sourceRows(nativeParams.moduliQP.size());
    for (uint32_t row = 0; row < sourceRows.size(); ++row) {
        sourceRows[row] = row;
    }
    return sourceRows;
}

std::vector<uint32_t> TrimmedRotationRandomSourceRows(uint32_t limb, uint32_t qLimbs, uint32_t pLimbs) {
    std::vector<uint32_t> sourceRows;
    sourceRows.reserve(limb + pLimbs);
    for (uint32_t row = 0; row < limb; ++row) {
        sourceRows.push_back(row);
    }
    for (uint32_t row = 0; row < pLimbs; ++row) {
        sourceRows.push_back(qLimbs + row);
    }
    return sourceRows;
}

const RotationRandomRows& GetOrCreateRotationRandomRows(RotationRandomCache& cache,
                                                        const NativeCkksParams& nativeParams,
                                                        const NativeSamplingContext& sampling,
                                                        uint32_t shapeId,
                                                        uint32_t parts,
                                                        const std::vector<uint32_t>& sourceRows) {
    auto existing = cache.rowsByShape.find(shapeId);
    if (existing != cache.rowsByShape.end()) {
        return existing->second;
    }

    RotationRandomRows generated;
    generated.aByPart.resize(parts);
    generated.eByPart.resize(parts);
    std::vector<SwitchPartProfile> partProfiles(parts);
    LocalDiscreteGaussian dgg(1.0);

    SequentialFor(parts, [&](size_t partIndex) {
        const uint32_t part = static_cast<uint32_t>(partIndex);
        auto partPrng = sampling.SwitchPartPrng("rotation-reuse", shapeId, part);
        SwitchPartProfile localProfile;
        double stageStart = NowSeconds();

        UIntMatrix aRows;
        aRows.reserve(sourceRows.size());
        for (auto sourceRow : sourceRows) {
            aRows.push_back(GenerateUniformVector(partPrng, nativeParams.moduliQP[sourceRow], nativeParams.ringDim));
        }
        if (gProfile.enabled) {
            localProfile.uniform += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        std::vector<int64_t> gaussianCoeff(nativeParams.ringDim);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (uint32_t coeff = 0; coeff < nativeParams.ringDim; ++coeff) {
            gaussianCoeff[coeff] = dgg.GenerateInt(partPrng, distribution);
        }
        UIntMatrix eRows;
        eRows.reserve(sourceRows.size());
        for (auto sourceRow : sourceRows) {
            eRows.push_back(MatrixRowFromSignedCoeffVectorWithTables(nativeParams.moduliQP[sourceRow],
                                                                     nativeParams.rootTablesQP[sourceRow],
                                                                     nativeParams.rootShoupTablesQP[sourceRow],
                                                                     nativeParams.ringDim,
                                                                     gaussianCoeff));
        }
        if (gProfile.enabled) {
            localProfile.gaussian += NowSeconds() - stageStart;
        }

        generated.aByPart[part] = std::move(aRows);
        generated.eByPart[part] = std::move(eRows);
        partProfiles[part] = localProfile;
    });

    if (gProfile.enabled) {
        for (const auto& profile : partProfiles) {
            gProfile.uniform += profile.uniform;
            gProfile.gaussian += profile.gaussian;
        }
    }

    auto [it, inserted] = cache.rowsByShape.emplace(shapeId, std::move(generated));
    (void)inserted;
    return it->second;
}

RotationKeyTensorResult ManualTrimmedRotationSwitchKeyGenToTensorResult(const NativeRuntimeConfig& runtime,
                                                                        const UIntMatrix& sOld,
                                                                        const UIntMatrix& sNewExt,
                                                                        const UIntMatrix& sNewExtShoup,
                                                                        const NativeCkksParams& nativeParams,
                                                                        const NativeSamplingContext& sampling,
                                                                        uint32_t autoIndex,
                                                                        uint32_t limb,
                                                                        bool trimmed,
                                                                        RotationRandomCache* randomCache) {
    const double switchStart = NowSeconds();
    const uint32_t qLimbs = static_cast<uint32_t>(nativeParams.moduliQ.size());
    const uint32_t pLimbs = static_cast<uint32_t>(nativeParams.moduliP.size());
    const uint32_t ringDim = nativeParams.ringDim;
    if (limb == 0 || limb > qLimbs) {
        throw std::runtime_error("invalid trimmed rotation key limb count");
    }
    const uint32_t alpha = nativeParams.numPerPartQ;
    const uint32_t beta = (limb + alpha - 1) / alpha;

    if (gProfile.enabled) {
        ++gProfile.switchCalls;
    }

    at::Tensor keyB = EmptyTensor(
        {static_cast<int64_t>(beta), static_cast<int64_t>(limb + pLimbs), static_cast<int64_t>(ringDim)}, at::kUInt64);
    at::Tensor keyA = EmptyTensor(
        {static_cast<int64_t>(beta), static_cast<int64_t>(limb + pLimbs), static_cast<int64_t>(ringDim)}, at::kUInt64);
    auto* bPtr = keyB.data_ptr<uint64_t>();
    auto* aPtr = keyA.data_ptr<uint64_t>();
    auto tensorOffset = [=](uint32_t part, uint32_t row, uint32_t coeff) {
        return (static_cast<size_t>(part) * (limb + pLimbs) + row) * ringDim + coeff;
    };

    LocalDiscreteGaussian dgg(1.0);
    std::vector<SwitchPartProfile> partProfiles(beta);
    const std::vector<uint64_t> noiseScaleShoup = MakeScalarShoupTable(runtime.noiseScale, nativeParams.moduliQP);
    const RotationRandomRows* cachedRandom = nullptr;
    std::vector<uint32_t> cachedSourceRows;
    if (randomCache != nullptr) {
        cachedSourceRows = TrimmedRotationRandomSourceRows(limb, qLimbs, pLimbs);
        cachedRandom = &GetOrCreateRotationRandomRows(*randomCache, nativeParams, sampling, limb, beta, cachedSourceRows);
    }

    SequentialFor(beta, [&](size_t partIndex) {
        const uint32_t part = static_cast<uint32_t>(partIndex);
        auto partPrng = sampling.SwitchPartPrng("rotation", autoIndex, part);
        SwitchPartProfile localProfile;
        double stageStart = NowSeconds();

        UIntMatrix aRowsOwned;
        const UIntMatrix* aRowsPtr = nullptr;
        if (cachedRandom != nullptr) {
            aRowsPtr = &cachedRandom->aByPart[part];
        }
        else {
            aRowsOwned.resize(limb + pLimbs);
            for (uint32_t row = 0; row < limb + pLimbs; ++row) {
                const uint32_t sourceRow = row < limb ? row : qLimbs + (row - limb);
                aRowsOwned[row] = GenerateUniformVector(partPrng, nativeParams.moduliQP[sourceRow], ringDim);
            }
            aRowsPtr = &aRowsOwned;
        }
        const UIntMatrix& aRows = *aRowsPtr;
        for (uint32_t row = 0; row < limb + pLimbs; ++row) {
            for (uint32_t coeff = 0; coeff < ringDim; ++coeff) {
                aPtr[tensorOffset(part, row, coeff)] = aRows[row][coeff];
            }
        }
        if (gProfile.enabled && cachedRandom == nullptr) {
            localProfile.uniform += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        UIntMatrix eRowsOwned;
        const UIntMatrix* eRowsPtr = nullptr;
        if (cachedRandom != nullptr) {
            eRowsPtr = &cachedRandom->eByPart[part];
        }
        else {
            std::vector<int64_t> gaussianCoeff(ringDim);
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            for (uint32_t coeff = 0; coeff < ringDim; ++coeff) {
                gaussianCoeff[coeff] = dgg.GenerateInt(partPrng, distribution);
            }
            eRowsOwned.resize(limb + pLimbs);
            for (uint32_t row = 0; row < limb + pLimbs; ++row) {
                const uint32_t sourceRow = row < limb ? row : qLimbs + (row - limb);
                eRowsOwned[row] = MatrixRowFromSignedCoeffVectorWithTables(
                    nativeParams.moduliQP[sourceRow],
                    nativeParams.rootTablesQP[sourceRow],
                    nativeParams.rootShoupTablesQP[sourceRow],
                    ringDim,
                    gaussianCoeff);
            }
            eRowsPtr = &eRowsOwned;
        }
        const UIntMatrix& eRows = *eRowsPtr;
        if (gProfile.enabled && cachedRandom == nullptr) {
            localProfile.gaussian += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }
        else if (gProfile.enabled) {
            stageStart = NowSeconds();
        }

        for (uint32_t row = 0; row < limb + pLimbs; ++row) {
            const uint32_t sourceRow = row < limb ? row : qLimbs + (row - limb);
            const uint64_t modulus = nativeParams.moduliQP[sourceRow];
            const uint64_t scalarMod = runtime.noiseScale % modulus;
            const uint64_t scalarShoup = noiseScaleShoup[sourceRow];
            for (uint32_t coeff = 0; coeff < ringDim; ++coeff) {
                const uint64_t scaled = scalarMod == 1 ?
                    eRows[row][coeff] :
                    ModMulPrecon(eRows[row][coeff], scalarMod, scalarShoup, modulus);
                const uint64_t product = ModMulPrecon(
                    aRows[row][coeff],
                    sNewExt[sourceRow][coeff],
                    sNewExtShoup[sourceRow][coeff],
                    modulus);
                bPtr[tensorOffset(part, row, coeff)] =
                    ModSub(scaled, product, modulus);
            }
        }
        if (gProfile.enabled) {
            localProfile.mulSub += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        const uint32_t startPartIdx = nativeParams.numPerPartQ * part;
        const uint32_t endPartIdx = std::min<uint32_t>(startPartIdx + nativeParams.numPerPartQ, qLimbs);
        for (uint32_t sourceRow = startPartIdx; sourceRow < endPartIdx && sourceRow < limb; ++sourceRow) {
            const uint64_t modulus = nativeParams.moduliQ[sourceRow];
            const uint64_t scalar = nativeParams.pModq[sourceRow] % modulus;
            const uint64_t scalarShoup = ShoupPrecon(scalar, modulus);
            for (uint32_t coeff = 0; coeff < ringDim; ++coeff) {
                const uint64_t term = ModMulPrecon(sOld[sourceRow][coeff], scalar, scalarShoup, modulus);
                const size_t offset = tensorOffset(part, sourceRow, coeff);
                bPtr[offset] = ModAdd(bPtr[offset], term, modulus);
            }
        }
        if (gProfile.enabled) {
            localProfile.pInjection += NowSeconds() - stageStart;
        }
        partProfiles[part] = localProfile;
    });

    if (gProfile.enabled) {
        gProfile.switchParts += beta;
        for (const auto& profile : partProfiles) {
            gProfile.uniform += profile.uniform;
            gProfile.gaussian += profile.gaussian;
            gProfile.mulSub += profile.mulSub;
            gProfile.pInjection += profile.pInjection;
        }
        gProfile.switchTotal += NowSeconds() - switchStart;
    }

    RotationKeyTensorResult item;
    item.autoIndex = autoIndex;
    item.keyB = std::move(keyB);
    item.keyA = std::move(keyA);
    if (trimmed) {
        item.limb = limb;
        item.trimmed = true;
    }
    return item;
}

RotationKeyTensorResult ManualFullRotationSwitchKeyGenToTensorResult(const NativeRuntimeConfig& runtime,
                                                                     const UIntMatrix& sOld,
                                                                     const UIntMatrix& sNewExt,
                                                                     const UIntMatrix& sNewExtShoup,
                                                                     const NativeCkksParams& nativeParams,
                                                                     const NativeSamplingContext& sampling,
                                                                     uint32_t autoIndex,
                                                                     RotationRandomCache* randomCache) {
    const double switchStart = NowSeconds();
    const uint32_t rows = static_cast<uint32_t>(nativeParams.moduliQP.size());
    const uint32_t qLimbs = static_cast<uint32_t>(nativeParams.moduliQ.size());
    const uint32_t ringDim = nativeParams.ringDim;

    if (gProfile.enabled) {
        ++gProfile.switchCalls;
    }

    at::Tensor keyB = EmptyTensor(
        {static_cast<int64_t>(nativeParams.numPartQ), static_cast<int64_t>(rows), static_cast<int64_t>(ringDim)},
        at::kUInt64);
    at::Tensor keyA = EmptyTensor(
        {static_cast<int64_t>(nativeParams.numPartQ), static_cast<int64_t>(rows), static_cast<int64_t>(ringDim)},
        at::kUInt64);
    auto* bPtr = keyB.data_ptr<uint64_t>();
    auto* aPtr = keyA.data_ptr<uint64_t>();

    LocalDiscreteGaussian dgg(1.0);
    std::vector<SwitchPartProfile> partProfiles(nativeParams.numPartQ);
    const std::vector<uint64_t> noiseScaleShoup = MakeScalarShoupTable(runtime.noiseScale, nativeParams.moduliQP);
    const RotationRandomRows* cachedRandom = nullptr;
    std::vector<uint32_t> cachedSourceRows;
    if (randomCache != nullptr) {
        cachedSourceRows = FullRotationRandomSourceRows(nativeParams);
        cachedRandom = &GetOrCreateRotationRandomRows(*randomCache, nativeParams, sampling, 0, nativeParams.numPartQ, cachedSourceRows);
    }

    auto generatePart = [&](uint32_t part, FixedBlake2Prng& partPrng) {
        SwitchPartProfile localProfile;
        double stageStart = NowSeconds();
        UIntMatrix aOwned;
        const UIntMatrix* aPtrRows = nullptr;
        if (cachedRandom != nullptr) {
            aPtrRows = &cachedRandom->aByPart[part];
        }
        else {
            aOwned = GenerateUniformMatrix(partPrng, nativeParams.moduliQP, nativeParams.ringDim);
            aPtrRows = &aOwned;
        }
        const UIntMatrix& a = *aPtrRows;
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t offset = (static_cast<size_t>(part) * rows + row) * ringDim;
            std::memcpy(aPtr + offset, a[row].data(), ringDim * sizeof(uint64_t));
        }
        if (gProfile.enabled && cachedRandom == nullptr) {
            localProfile.uniform += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        UIntMatrix eOwned;
        const UIntMatrix* ePtrRows = nullptr;
        if (cachedRandom != nullptr) {
            ePtrRows = &cachedRandom->eByPart[part];
        }
        else {
            eOwned = GenerateGaussianMatrix(
                partPrng,
                dgg,
                nativeParams.moduliQP,
                nativeParams.rootTablesQP,
                nativeParams.rootShoupTablesQP,
                nativeParams.ringDim);
            ePtrRows = &eOwned;
        }
        const UIntMatrix& e = *ePtrRows;
        if (gProfile.enabled && cachedRandom == nullptr) {
            localProfile.gaussian += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }
        else if (gProfile.enabled) {
            stageStart = NowSeconds();
        }

        UIntMatrix b = MatrixEvalScaledSubMulPrecon(
            e, runtime.noiseScale, noiseScaleShoup, a, sNewExt, sNewExtShoup, nativeParams.moduliQP);
        if (gProfile.enabled) {
            localProfile.mulSub += NowSeconds() - stageStart;
            stageStart = NowSeconds();
        }

        const uint32_t startPartIdx = nativeParams.numPerPartQ * part;
        const uint32_t endPartIdx = std::min<uint32_t>(startPartIdx + nativeParams.numPerPartQ, qLimbs);
        for (uint32_t limb = startPartIdx; limb < endPartIdx; ++limb) {
            const uint64_t modulus = nativeParams.moduliQ[limb];
            const uint64_t scalar = nativeParams.pModq[limb] % modulus;
            const uint64_t scalarShoup = ShoupPrecon(scalar, modulus);
            for (uint32_t coeff = 0; coeff < nativeParams.ringDim; ++coeff) {
                const uint64_t term = ModMulPrecon(sOld[limb][coeff], scalar, scalarShoup, modulus);
                b[limb][coeff] = ModAdd(b[limb][coeff], term, modulus);
            }
        }
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t offset = (static_cast<size_t>(part) * rows + row) * ringDim;
            std::memcpy(bPtr + offset, b[row].data(), ringDim * sizeof(uint64_t));
        }
        if (gProfile.enabled) {
            localProfile.pInjection += NowSeconds() - stageStart;
        }
        partProfiles[part] = localProfile;
    };

    if (sampling.IsParallelDeterministic()) {
        SequentialFor(nativeParams.numPartQ, [&](size_t partIndex) {
            auto partPrng = sampling.SwitchPartPrng("rotation", autoIndex, partIndex);
            generatePart(static_cast<uint32_t>(partIndex), partPrng);
        });
    }
    else {
        throw std::runtime_error("full direct rotation key tuple path requires parallel_deterministic sampling");
    }

    if (gProfile.enabled) {
        gProfile.switchParts += nativeParams.numPartQ;
        for (const auto& profile : partProfiles) {
            gProfile.uniform += profile.uniform;
            gProfile.gaussian += profile.gaussian;
            gProfile.mulSub += profile.mulSub;
            gProfile.pInjection += profile.pInjection;
        }
        gProfile.switchTotal += NowSeconds() - switchStart;
    }

    RotationKeyTensorResult item;
    item.autoIndex = autoIndex;
    item.keyB = std::move(keyB);
    item.keyA = std::move(keyA);
    return item;
}

ManualEvalMultKeyResult ManualEvalMultKeyGen(const NativeRuntimeConfig& runtime,
                                             const ManualKeygenResult& key,
                                             const NativeCkksParams& nativeParams,
                                             FixedBlake2Prng& prng,
                                             const NativeSamplingContext& sampling) {
    // Sequential sampling preserves the historical key-tag consumption order
    // before generating the s^2 switching key.
    if (!sampling.IsParallelDeterministic()) {
        ConsumeKeyTag(prng);
    }

    UIntMatrix sOld = MatrixEvalMul(key.sk, key.sk, nativeParams.moduliQ);
    UIntMatrix sNewExt = MatrixFromSignedCoeffVectorWithTables(
        nativeParams.moduliQP, nativeParams.rootTablesQP, nativeParams.rootShoupTablesQP, nativeParams.ringDim, key.skCoeff);
    UIntMatrix sNewExtShoup = MatrixShoupPrecon(sNewExt, nativeParams.moduliQP);
    UIntMatrix pModqMatrix(1, nativeParams.pModq);
    ManualSwitchKeyResult switchKey = ManualHybridKeySwitchGen(
        runtime, sOld, sNewExt, sNewExtShoup, nativeParams, prng, sampling, "eval-mult", 0, true);

    return ManualEvalMultKeyResult{std::move(sOld),
                                   std::move(sNewExt),
                                   std::move(pModqMatrix),
                                   std::move(switchKey.keyB),
                                   std::move(switchKey.keyA),
                                   std::move(switchKey.keyE)};
}

std::vector<ManualRotationKey> ManualRotationKeyGen(const NativeRuntimeConfig& runtime,
                                                    const ManualKeygenResult& key,
                                                    const NativeCkksParams& nativeParams,
                                                    const std::vector<std::vector<int32_t>>& rotationIndexGroups,
                                                    FixedBlake2Prng& prng,
                                                    const NativeSamplingContext& sampling) {
    const UIntMatrix sExt = MatrixFromSignedCoeffVectorWithTables(
        nativeParams.moduliQP, nativeParams.rootTablesQP, nativeParams.rootShoupTablesQP, nativeParams.ringDim, key.skCoeff);
    const UIntMatrix sExtShoup = MatrixShoupPrecon(sExt, nativeParams.moduliQP);

    std::set<uint32_t> generatedAutoIndices;
    std::vector<ManualRotationKey> keys;
    for (const auto& group : rotationIndexGroups) {
        std::set<uint32_t> groupAutoIndices;
        for (auto rotationIndex : group) {
            groupAutoIndices.insert(FindAutomorphismIndex2nComplexLocal(rotationIndex, nativeParams.cyclOrder));
        }
        for (auto autoIndex : groupAutoIndices) {
            const bool forceRegenerate = autoIndex == nativeParams.cyclOrder - 1;
            if (!forceRegenerate && generatedAutoIndices.count(autoIndex) != 0) {
                continue;
            }
            if (!forceRegenerate) {
                generatedAutoIndices.insert(autoIndex);
            }
            // EvalAutomorphismKeyGen creates a temporary PrivateKeyImpl for the
            // inverse-permuted secret key before calling KeySwitchGen.
            if (!sampling.IsParallelDeterministic()) {
                ConsumeKeyTag(prng);
            }
            const uint32_t inverseAutoIndex = static_cast<uint32_t>(ModInverse(autoIndex, nativeParams.cyclOrder));
            UIntMatrix sNewExt = AutomorphismTransformEval(sExt, inverseAutoIndex, nativeParams.ringDim);
            UIntMatrix sNewExtShoup = AutomorphismTransformEval(sExtShoup, inverseAutoIndex, nativeParams.ringDim);
            keys.push_back(ManualRotationKey{
                autoIndex,
                ManualHybridKeySwitchGen(
                    runtime, key.sk, sNewExt, sNewExtShoup, nativeParams, prng, sampling, "rotation", autoIndex, false),
            });
        }
    }

    return keys;
}

std::vector<RotationKeyTensorResult> ManualRotationKeyGenToTensorList(
    const NativeRuntimeConfig& runtime,
    const ManualKeygenResult& key,
    const NativeCkksParams& nativeParams,
    const std::vector<std::vector<int32_t>>& rotationIndexGroups,
    const std::map<uint32_t, uint32_t>& rotationTrimLimbsByAutoIndex,
    uint32_t dnum,
    FixedBlake2Prng& prng,
    const NativeSamplingContext& sampling,
    RotationRandomModeLocal rotationRandomMode) {
    const double rotationStart = NowSeconds();
    if (rotationRandomMode == RotationRandomModeLocal::ReuseByShape && !sampling.IsParallelDeterministic()) {
        throw std::runtime_error("rotation_random_mode=reuse_by_shape requires random_mode=parallel_deterministic");
    }
    std::optional<RotationRandomCache> randomCache;
    if (rotationRandomMode == RotationRandomModeLocal::ReuseByShape) {
        std::cerr << "[native-warning] rotation_random_mode=reuse_by_shape reuses uniform/gaussian samples "
                     "across rotation keys with the same shape. This is for development/profiling only and "
                     "must not be used for production key material."
                  << std::endl;
        randomCache.emplace();
    }
    const UIntMatrix sExt = MatrixFromSignedCoeffVectorWithTables(
        nativeParams.moduliQP, nativeParams.rootTablesQP, nativeParams.rootShoupTablesQP, nativeParams.ringDim, key.skCoeff);
    const UIntMatrix sExtShoup = MatrixShoupPrecon(sExt, nativeParams.moduliQP);

    struct RotationKeyTask {
        uint32_t autoIndex = 0;
        uint32_t rotationIndex = 0;
    };

    std::set<uint32_t> generatedAutoIndices;
    std::vector<RotationKeyTask> tasks;
    for (const auto& group : rotationIndexGroups) {
        std::set<uint32_t> groupAutoIndices;
        std::map<uint32_t, uint32_t> rotationIndexByAutoIndex;
        for (auto rotationIndex : group) {
            const uint32_t autoIndex = FindAutomorphismIndex2nComplexLocal(rotationIndex, nativeParams.cyclOrder);
            groupAutoIndices.insert(autoIndex);
            rotationIndexByAutoIndex.emplace(autoIndex, NormalizeRotationIndexLocal(rotationIndex, nativeParams.ringDim));
        }
        for (auto autoIndex : groupAutoIndices) {
            const bool forceRegenerate = autoIndex == nativeParams.cyclOrder - 1;
            if (!forceRegenerate && generatedAutoIndices.count(autoIndex) != 0) {
                continue;
            }
            if (!forceRegenerate) {
                generatedAutoIndices.insert(autoIndex);
            }
            tasks.push_back(RotationKeyTask{autoIndex, rotationIndexByAutoIndex.at(autoIndex)});
        }
    }

    std::vector<RotationKeyTensorResult> keyList(tasks.size());
    auto generateTask = [&](size_t taskIndex, RotationRandomCache* taskRandomCache) -> RotationKeyTensorResult {
            const auto& task = tasks[taskIndex];
            const uint32_t autoIndex = task.autoIndex;

            if (!sampling.IsParallelDeterministic()) {
                ConsumeKeyTag(prng);
            }
            const uint32_t inverseAutoIndex = static_cast<uint32_t>(ModInverse(autoIndex, nativeParams.cyclOrder));
            double stageStart = NowSeconds();
            UIntMatrix sNewExt = AutomorphismTransformEval(sExt, inverseAutoIndex, nativeParams.ringDim);
            UIntMatrix sNewExtShoup = AutomorphismTransformEval(sExtShoup, inverseAutoIndex, nativeParams.ringDim);
            if (gProfile.enabled) {
                gProfile.automorphism += NowSeconds() - stageStart;
                ++gProfile.rotationKeys;
            }
            auto trimIt = rotationTrimLimbsByAutoIndex.find(autoIndex);
            auto finalizeItem = [&](RotationKeyTensorResult&& item) -> RotationKeyTensorResult {
                item.rotationIndex = task.rotationIndex;
                const auto autoMap = PrecomputeAutoMapLocal(nativeParams.ringDim, autoIndex);
                item.autoMap = UInt32VectorToInt32Tensor(autoMap);
                item.inverseAutoMap = UInt32VectorToInt32Tensor(InvertAutoMapLocal(autoMap));
                return item;
            };
            if (sampling.IsParallelDeterministic() && trimIt != rotationTrimLimbsByAutoIndex.end()) {
                RotationKeyTensorResult item = ManualTrimmedRotationSwitchKeyGenToTensorResult(
                    runtime,
                    key.sk,
                    sNewExt,
                    sNewExtShoup,
                    nativeParams,
                    sampling,
                    autoIndex,
                    trimIt->second,
                    true,
                    taskRandomCache);
                stageStart = NowSeconds();
                item = finalizeItem(std::move(item));
                if (gProfile.enabled) {
                    gProfile.tupleReturn += NowSeconds() - stageStart;
                }
                return item;
            }
            else if (sampling.IsParallelDeterministic()) {
                RotationKeyTensorResult item = ManualFullRotationSwitchKeyGenToTensorResult(
                    runtime,
                    key.sk,
                    sNewExt,
                    sNewExtShoup,
                    nativeParams,
                    sampling,
                    autoIndex,
                    taskRandomCache);
                stageStart = NowSeconds();
                item = finalizeItem(std::move(item));
                if (gProfile.enabled) {
                    gProfile.tupleReturn += NowSeconds() - stageStart;
                }
                return item;
            }
            else {
                ManualRotationKey rotKey{
                    autoIndex,
                    ManualHybridKeySwitchGen(
                        runtime, key.sk, sNewExt, sNewExtShoup, nativeParams, prng, sampling, "rotation", autoIndex, false),
                };
                stageStart = NowSeconds();
                RotationKeyTensorResult item =
                    finalizeItem(RotationKeyToTensorResult(rotKey,
                                                           rotationTrimLimbsByAutoIndex,
                                                           static_cast<uint32_t>(nativeParams.moduliQ.size()),
                                                           static_cast<uint32_t>(nativeParams.moduliP.size()),
                                                           dnum));
                if (gProfile.enabled) {
                    gProfile.tupleReturn += NowSeconds() - stageStart;
                }
                return item;
            }
    };

    const bool parallelKeyGen = sampling.IsParallelDeterministic() &&
                                rotationRandomMode == RotationRandomModeLocal::Fresh &&
                                !gProfile.enabled;
    if (parallelKeyGen) {
        ParallelForKeys(tasks.size(), [&](size_t taskIndex) {
            keyList[taskIndex] = generateTask(taskIndex, nullptr);
        });
    }
    else {
        for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
            keyList[taskIndex] = generateTask(taskIndex, randomCache ? &*randomCache : nullptr);
            if (gProfile.enabled && gProfile.rotationKeys % 10 == 0) {
                std::cerr << "[native-profile] generated rotation keys=" << gProfile.rotationKeys
                          << " elapsed_rotation=" << (NowSeconds() - rotationStart)
                          << std::endl;
            }
        }
    }

    if (gProfile.enabled) {
        gProfile.rotationTotal += NowSeconds() - rotationStart;
    }
    return keyList;
}

ManualEncryptResult ManualEncryptPublicZero(const NativeRuntimeConfig& runtime,
                                            const ManualKeygenResult& key,
                                            const NativeCkksParams& nativeParams,
                                            const UIntMatrix& plaintext,
                                            FixedBlake2Prng& prng) {
    LocalDiscreteGaussian dgg(runtime.dggStd);

    UIntMatrix v = runtime.secretKeyDist == SecretKeyDistLocal::Gaussian ?
                       GenerateGaussianMatrix(
                           prng, dgg, nativeParams.moduliQ, nativeParams.rootTablesQ, nativeParams.rootShoupTablesQ, nativeParams.ringDim) :
                       GenerateTernaryMatrix(prng, nativeParams.moduliQ, nativeParams.rootsQ, nativeParams.ringDim, 0);
    UIntMatrix e0 = GenerateGaussianMatrix(
        prng, dgg, nativeParams.moduliQ, nativeParams.rootTablesQ, nativeParams.rootShoupTablesQ, nativeParams.ringDim);
    UIntMatrix e1 = GenerateGaussianMatrix(
        prng, dgg, nativeParams.moduliQ, nativeParams.rootTablesQ, nativeParams.rootShoupTablesQ, nativeParams.ringDim);

    UIntMatrix e0Scaled = MatrixEvalScale(e0, runtime.noiseScale, nativeParams.moduliQ);
    UIntMatrix e1Scaled = MatrixEvalScale(e1, runtime.noiseScale, nativeParams.moduliQ);

    UIntMatrix c0Zero = MatrixEvalAdd(MatrixEvalMul(key.pkB, v, nativeParams.moduliQ), e0Scaled, nativeParams.moduliQ);
    UIntMatrix c1Zero = MatrixEvalAdd(MatrixEvalMul(key.pkA, v, nativeParams.moduliQ), e1Scaled, nativeParams.moduliQ);

    UIntMatrix c0 = MatrixEvalAdd(c0Zero, plaintext, nativeParams.moduliQ);
    UIntMatrix c1(c1Zero);

    return ManualEncryptResult{std::move(v),
                               std::move(e0Scaled),
                               std::move(e1Scaled),
                               std::move(c0Zero),
                               std::move(c1Zero),
                               std::move(c0),
                               std::move(c1)};
}

at::Tensor ManualCipherToTensor(const ManualEncryptResult& encrypted) {
    const auto& c0 = encrypted.ct0;
    const auto& c1 = encrypted.ct1;
    if (c0.size() != c1.size()) {
        throw std::runtime_error("manual ciphertext components have inconsistent limb count");
    }
    const size_t components = 2;
    const size_t limbs      = c0.size();
    const size_t ringDim    = limbs == 0 ? 0 : c0.front().size();
    at::Tensor out = EmptyTensor(
        {static_cast<int64_t>(components), static_cast<int64_t>(limbs), static_cast<int64_t>(ringDim)}, at::kUInt64);
    auto* outPtr = out.data_ptr<uint64_t>();
    for (size_t limb = 0; limb < limbs; ++limb) {
        if (c0[limb].size() != ringDim || c1[limb].size() != ringDim) {
            throw std::runtime_error("manual ciphertext components have inconsistent ring dimension");
        }
        std::memcpy(outPtr + limb * ringDim, c0[limb].data(), ringDim * sizeof(uint64_t));
        std::memcpy(outPtr + (limbs + limb) * ringDim, c1[limb].data(), ringDim * sizeof(uint64_t));
    }
    return out;
}

void AppendRotationKeyTensors(std::vector<at::Tensor>& out, const std::vector<RotationKeyTensorResult>& keys) {
    at::Tensor manifest = EmptyTensor({static_cast<int64_t>(keys.size())}, at::kLong);
    auto* manifestPtr = manifest.data_ptr<int64_t>();
    for (size_t i = 0; i < keys.size(); ++i) {
        manifestPtr[i] = static_cast<int64_t>(keys[i].rotationIndex);
    }
    out.push_back(std::move(manifest));
    for (const auto& key : keys) {
        out.push_back(key.keyB);
        out.push_back(key.keyA);
        out.push_back(key.autoMap);
        out.push_back(key.inverseAutoMap);
    }
}

std::vector<at::Tensor> SampleCkks(const NativeSamplerRequest& request, const at::Tensor& values) {
    gProfile.Reset();
    FixedBlake2Prng prng;
    NativeSamplingContext sampling(ParseRandomMode(request.randomMode));
    auto nativeParams       = GenerateNativeCkksParams(request, prng);
    auto runtime            = GetRuntimeConfig(request);
    double stageStart = NowSeconds();
    auto key               = ManualKeyGen(runtime, nativeParams, prng);
    if (gProfile.enabled) {
        gProfile.keygen += NowSeconds() - stageStart;
    }
    ManualEvalMultKeyResult evalMultKey;
    stageStart = NowSeconds();
    evalMultKey = ManualEvalMultKeyGen(runtime, key, nativeParams, prng, sampling);
    if (gProfile.enabled) {
        gProfile.evalMult += NowSeconds() - stageStart;
    }
    std::vector<double> inputValues;
    uint32_t actualSlots = 0;
    std::optional<UIntMatrix> ptx;
    std::optional<ManualEncryptResult> encrypted;
    std::optional<UIntMatrix> phase;
    if (request.includeEncryptTrace) {
        stageStart = NowSeconds();
        inputValues = ValuesFromTensor(values);
        const auto ringDim = nativeParams.ringDim;
        actualSlots =
            request.slots == 0 ? std::max<uint32_t>(static_cast<uint32_t>(inputValues.size()), static_cast<uint32_t>(ringDim / 2)) :
                         static_cast<uint32_t>(request.slots);

        ptx = EncodeCkksPackedNative(nativeParams.moduliQ,
                                     nativeParams.rootsQ,
                                     nativeParams.ringDim,
                                     inputValues,
                                     static_cast<uint32_t>(request.dcrtBits),
                                     static_cast<uint32_t>(request.scaleDeg),
                                     static_cast<uint32_t>(request.level),
                                     actualSlots);
        encrypted = ManualEncryptPublicZero(runtime, key, nativeParams, *ptx, prng);
        phase     = DecryptionPhase(encrypted->ct0, encrypted->ct1, key.sk, nativeParams.moduliQ);
        if (gProfile.enabled) {
            gProfile.encodeEncryptDecrypt += NowSeconds() - stageStart;
        }
    }

    std::vector<at::Tensor> out;
    out.reserve(10);
    out.push_back(VectorToTensor(nativeParams.moduliQ));
    out.push_back(VectorToTensor(nativeParams.rootsQ));
    out.push_back(VectorToTensor(nativeParams.moduliP));
    out.push_back(VectorToTensor(nativeParams.rootsP));
    out.push_back(MatrixToTensor(key.sk));
    out.push_back(Int64VectorToTensor(key.skCoeff));
    out.push_back(MatrixToTensor(key.pkB));
    out.push_back(MatrixToTensor(key.pkA));
    out.push_back(MatrixVectorToTensor(evalMultKey.keyB));
    out.push_back(MatrixVectorToTensor(evalMultKey.keyA));

    gProfile.Report();
    return out;
}

std::vector<at::Tensor> SampleRotationKeys(const NativeSamplerRequest& request,
                                           const at::Tensor& secretKey,
                                           const at::Tensor& secretKeyCoeff) {
    gProfile.Reset();
    FixedBlake2Prng prng;
    NativeSamplingContext sampling(ParseRandomMode(request.randomMode));
    const auto rotationRandomMode = ParseRotationRandomMode(request.rotationRandomMode);
    auto nativeParams = GenerateNativeCkksParams(request, prng);
    auto runtime      = GetRuntimeConfig(request);

    ManualKeygenResult key;
    key.sk      = MatrixFromTensor(secretKey, "secret_key");
    key.skCoeff = Int64VectorFromTensor(secretKeyCoeff, "secret_key_coeff");

    if (key.sk.size() != nativeParams.moduliQ.size()) {
        throw std::invalid_argument("secret_key limb count does not match generated CKKS params");
    }
    if (!key.sk.empty() && key.sk.front().size() != nativeParams.ringDim) {
        throw std::invalid_argument("secret_key ring dimension does not match generated CKKS params");
    }
    if (key.skCoeff.size() != nativeParams.ringDim) {
        throw std::invalid_argument("secret_key_coeff length does not match generated CKKS params");
    }

    auto rotationKeys = ManualRotationKeyGenToTensorList(runtime,
                                                         key,
                                                         nativeParams,
                                                         request.rotationIndexGroups,
                                                         request.rotationTrimLimbsByAutoIndex,
                                                         static_cast<uint32_t>(request.dnum),
                                                         prng,
                                                         sampling,
                                                         rotationRandomMode);
    std::vector<at::Tensor> out;
    out.reserve(1 + rotationKeys.size() * 4);
    AppendRotationKeyTensors(out, rotationKeys);
    gProfile.Report();
    return out;
}

NativeSamplerRequest MakeRequest(int64_t logN,
                                 int64_t depth,
                                 int64_t dcrtBits,
                                 int64_t firstMod,
                                 int64_t dnum,
                                 std::string_view secretKeyDist,
                                 bool includeEvalMultKey,
                                 bool includeEncryptTrace,
                                 int64_t scaleDeg,
                                 int64_t level,
                                 int64_t slots,
                                 at::IntArrayRef rotationIndices,
                                 at::IntArrayRef rotationGroupOffsets,
                                 at::IntArrayRef rotationTrimAutoIndices,
                                 at::IntArrayRef rotationTrimLimbs,
                                 std::string_view rotationRandomMode,
                                 std::string_view randomMode,
                                 double standardDeviation,
                                 int64_t noiseScale) {
    if (noiseScale < 0) {
        throw std::invalid_argument("noise_scale must be non-negative");
    }
    NativeSamplerRequest request;
    request.logN = logN;
    request.depth = depth;
    request.dcrtBits = dcrtBits;
    request.firstMod = firstMod;
    request.dnum = dnum;
    request.secretKeyDist = std::string(secretKeyDist);
    request.includeEvalMultKey = includeEvalMultKey;
    request.includeEncryptTrace = includeEncryptTrace;
    request.scaleDeg = scaleDeg;
    request.level = level;
    request.slots = slots;
    request.rotationIndexGroups = Int32VectorGroupsFromFlat(rotationIndices, rotationGroupOffsets);
    request.rotationTrimLimbsByAutoIndex = UInt32MapFromParallelArrays(rotationTrimAutoIndices, rotationTrimLimbs);
    request.rotationRandomMode = std::string(rotationRandomMode);
    request.randomMode = std::string(randomMode);
    request.standardDeviation = standardDeviation;
    request.noiseScale = static_cast<uint64_t>(noiseScale);
    return request;
}

}  // namespace

namespace at::native {

std::vector<Tensor> fhe_native_sample_ckks_cpu(const Tensor& values,
                                               int64_t logN,
                                               int64_t depth,
                                               int64_t dcrtBits,
                                               int64_t firstMod,
                                               int64_t dnum,
                                               std::string_view secretKeyDist,
                                               bool includeEvalMultKey,
                                               bool includeEncryptTrace,
                                               int64_t scaleDeg,
                                               int64_t level,
                                               int64_t slots,
                                               std::string_view randomMode,
                                               double standardDeviation,
                                               int64_t noiseScale) {
    auto request = MakeRequest(logN,
                               depth,
                               dcrtBits,
                               firstMod,
                               dnum,
                               secretKeyDist,
                               includeEvalMultKey,
                               includeEncryptTrace,
                               scaleDeg,
                               level,
                               slots,
                               at::IntArrayRef{},
                               at::IntArrayRef{},
                               at::IntArrayRef{},
                               at::IntArrayRef{},
                               "fresh",
                               randomMode,
                               standardDeviation,
                               noiseScale);
    return SampleCkks(request, values);
}

std::vector<Tensor> fhe_native_sample_rotation_keys_cpu(const Tensor& secretKey,
                                                        const Tensor& secretKeyCoeff,
                                                        int64_t logN,
                                                        int64_t depth,
                                                        int64_t dcrtBits,
                                                        int64_t firstMod,
                                                        int64_t dnum,
                                                        std::string_view secretKeyDist,
                                                        at::IntArrayRef rotationIndices,
                                                        at::IntArrayRef rotationGroupOffsets,
                                                        at::IntArrayRef rotationTrimAutoIndices,
                                                        at::IntArrayRef rotationTrimLimbs,
                                                        std::string_view rotationRandomMode,
                                                        std::string_view randomMode,
                                                        double standardDeviation,
                                                        int64_t noiseScale) {
    auto request = MakeRequest(logN,
                               depth,
                               dcrtBits,
                               firstMod,
                               dnum,
                               secretKeyDist,
                               false,
                               false,
                               1,
                               0,
                               0,
                               rotationIndices,
                               rotationGroupOffsets,
                               rotationTrimAutoIndices,
                               rotationTrimLimbs,
                               rotationRandomMode,
                               randomMode,
                               standardDeviation,
                               noiseScale);
    return SampleRotationKeys(request, secretKey, secretKeyCoeff);
}

}  // namespace at::native
