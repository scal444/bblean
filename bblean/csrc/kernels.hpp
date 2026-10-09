// Kernels of the similarity calculations (similarity.cpp): popcounts, Tanimoto
// similarity, iSIM and majority centroids
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Scalar popcount intrinsics:
#if defined(__SSE_4_2__) || defined(_M_SSE4_2)
// Compiler-portable, but *not available in systems that do not have SSE*
// (which should be almost no CPUs nowadays)
// Not actually vector instructions, they just live in the SSE header
// Should be *exactly as fast* as __(builtin_)popcnt(ll) (compile to the same
// code)
//
// nmmintrin.h is the SSE4.2 intrinsics (only) header for all compilers
// NOTE: This ifdef is probably overkill, almost all cases should be covered by
// the GCC|Clang|MSVC ifdefs, but it doesn't hurt to add it
#include <nmmintrin.h>
#define POPCOUNT_32 _mm_popcnt_u32
#define POPCOUNT_64 _mm_popcnt_u64
#elif defined(_MSC_VER)
// Windows (MSVC compiler)
#include <intrin.h>
#define POPCOUNT_32 __popcnt
#define POPCOUNT_64 __popcnt64
#elif defined(__GNUC__) || defined(__clang__)
// GCC | Clang
#define POPCOUNT_32 __builtin_popcount
#define POPCOUNT_64 __builtin_popcountll
#else
// If popcnt is not hardware supported numpy rolls out its own hand-coded
// version, fail for simplicity since it is not worth it to support those archs
#error "Popcount not supported in target architecture"
#endif

namespace bblean {

// Tanimoto similarities of n_samples rows of steps words to vec (T must be
// uint64_t or uint8_t), given the popcounts of the rows (cardinalities) and of
// vec. The similarity of two empty fingerprints is 0.
template <typename T>
inline void jt_sims(const T* arr, const T* vec, size_t n_samples, size_t steps, uint32_t vec_popcount,
                    const uint32_t* cardinalities, double* out) {
    for (size_t i{0}; i != n_samples; ++i) {  // not auto-vec by GCC
        const T* arr_row = arr + i * steps;
        uint32_t intersection{0};
        for (size_t j{0}; j != steps; ++j) {  // not auto-vec by GCC
            if constexpr (std::is_same_v<T, uint64_t>) {
                intersection += POPCOUNT_64(arr_row[j] & vec[j]);
            } else {
                intersection += POPCOUNT_32(arr_row[j] & vec[j]);
            }
        }
        auto denominator = cardinalities[i] + vec_popcount - intersection;
        // Cast is technically unnecessary since std::max promotes to double,
        // but added here for clarity (should compile to nop)
        out[i] = intersection / std::max(static_cast<double>(denominator), 1.0);
    }
}

// iSIM Tanimoto of n_objects fingerprints, from sum_kq and sum_kqsq, the sum and
// the sum of squares of their column-wise sum (n_objects must be >= 2)
inline double isim(uint64_t sum_kq, uint64_t sum_kqsq, int64_t n_objects) {
    if (sum_kq == 0) {
        return 1.0;
    }
    auto a = (sum_kqsq - sum_kq) / 2.0;
    return a / ((a + (n_objects * sum_kq)) - sum_kqsq);
}

// Majority vote centroid value of a column with sum linear_sum over n_samples
// fingerprints
template <typename T>
inline uint8_t centroid_value(T linear_sum, int64_t n_samples) {
    if (n_samples <= 1) {
        // Narrowing conversion: if n_samples <= 1 then linear_sum is guaranteed
        // to have a value that a uint8_t can hold (it should be 0 or 1)
        return static_cast<uint8_t>(linear_sum);
    }
    return (linear_sum >= n_samples * 0.5) ? 1 : 0;
}

}  // namespace bblean
