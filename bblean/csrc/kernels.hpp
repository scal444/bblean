// Kernels shared by the similarity calculations (similarity.cpp) and the
// BitBirch tree (bitbirch.cpp): popcounts, Tanimoto similarity, iSIM, majority
// centroids, the most dissimilar pair of a set of fingerprints, and the merge
// criteria
//
// Fingerprints are rows of 64 bit words: a packed (uint8) fingerprint viewed as
// 64 bit words, zero padded to a whole number of words. The order of the bits
// within the words does not matter for these calculations.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#define BBLEAN_NEON 1
#include <arm_neon.h>
#endif

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

using u64 = uint64_t;

inline uint32_t popcount(const u64* a, size_t w64) {
    uint32_t s = 0;
    for (size_t k = 0; k < w64; ++k) s += static_cast<uint32_t>(POPCOUNT_64(a[k]));
    return s;
}

#if defined(_MSC_VER) && !defined(__clang__)
}  // namespace bblean
#include <intrin.h>
namespace bblean {
inline int ctz64(u64 x) {
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
}
#else
inline int ctz64(u64 x) { return __builtin_ctzll(x); }
#endif

// Calls f(p) for every set bit p of a row
template <typename F>
inline void for_each_bit(const u64* a, size_t w64, F&& f) {
    for (size_t k = 0; k < w64; ++k)
        for (u64 w = a[k]; w; w &= w - 1) f(static_cast<int>(k * 64) + ctz64(w));
}

#if defined(BBLEAN_NEON)
// Query of V 16-byte vectors held in registers. Byte counts accumulate in u8
// lanes, at most 8 * V / 2 <= 128 per lane for V <= 32.
template <int V>
struct NeonQuery {
    uint8x16_t v[V];
    explicit NeonQuery(const u64* q) {
        const auto* p = reinterpret_cast<const uint8_t*>(q);
        for (int k = 0; k < V; ++k) v[k] = vld1q_u8(p + 16 * k);
    }
    uint32_t and_popcount(const u64* a) const {
        const auto* p = reinterpret_cast<const uint8_t*>(a);
        uint8x16_t acc0 = vcntq_u8(vandq_u8(vld1q_u8(p), v[0]));
        uint8x16_t acc1 = vcntq_u8(vandq_u8(vld1q_u8(p + 16), v[1]));
        for (int k = 2; k < V; k += 2) {
            acc0 = vaddq_u8(acc0, vcntq_u8(vandq_u8(vld1q_u8(p + 16 * k), v[k])));
            acc1 = vaddq_u8(acc1, vcntq_u8(vandq_u8(vld1q_u8(p + 16 * k + 16), v[k + 1])));
        }
        return vaddlvq_u8(vaddq_u8(acc0, acc1));
    }
};
#endif

// Calls f(i, popcount(row_i & query)) for each of the n rows (rows of w64 words)
template <typename F>
inline void for_each_intersection(const u64* rows, size_t n, size_t w64, const u64* query, F&& f) {
#if defined(BBLEAN_NEON)
    // 2048 and 1024 bit fingerprints: the query is kept in registers
    if (w64 == 32 || w64 == 16) {
        auto scan = [&](const auto& q) {
            for (size_t i = 0; i < n; ++i) f(i, q.and_popcount(rows + i * w64));
        };
        if (w64 == 32) {
            scan(NeonQuery<16>(query));
        } else {
            scan(NeonQuery<8>(query));
        }
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) {
        const u64* r = rows + i * w64;
        uint32_t s = 0;
        for (size_t k = 0; k < w64; ++k) s += static_cast<uint32_t>(POPCOUNT_64(r[k] & query[k]));
        f(i, s);
    }
}

// Tanimoto similarity from the popcounts of a & b, a and b. The similarity of two
// empty fingerprints is 0.
inline double jt_sim(uint32_t intersection, uint32_t pop_a, uint32_t pop_b) {
    const uint32_t denominator = pop_a + pop_b - intersection;
    return intersection / std::max(static_cast<double>(denominator), 1.0);
}

// Tanimoto similarities of n rows (with popcounts pops) to query
inline void jt_sims(const u64* rows, const uint32_t* pops, size_t n, size_t w64, const u64* query, double* out) {
    const uint32_t query_pop = popcount(query, w64);
    for_each_intersection(rows, n, w64, query,
                          [&](size_t i, uint32_t inter) { out[i] = jt_sim(inter, pops[i], query_pop); });
}

// Index of the first minimum / maximum of n values
inline size_t first_min(const double* v, size_t n) { return static_cast<size_t>(std::min_element(v, v + n) - v); }
inline size_t first_max(const double* v, size_t n) { return static_cast<size_t>(std::max_element(v, v + n) - v); }

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

template <typename T>
inline bool centroid_bit(T linear_sum, int64_t n_samples) {
    return centroid_value(linear_sum, n_samples) != 0;
}

// Sum and sum of squares of the column sums L (count columns) of n objects;
// on_centroid(j, L_j) is called for each column j in their majority centroid
template <typename T, typename F>
inline void column_sums(const T* L, size_t count, u64 n, u64& sum, u64& sum_sq, F&& on_centroid) {
    sum = sum_sq = 0;
    for (size_t j = 0; j < count; ++j) {
        const u64 v = static_cast<u64>(L[j]);
        sum += v;
        sum_sq += v * v;
        if (centroid_bit(v, static_cast<int64_t>(n))) on_centroid(j, v);
    }
}

// Complement of the iSIM radius of n objects (n >= 2) with column sums L, given
// sum, sum_sq, the popcount c_pop of the centroid c and S = sum_{j in c} L_j.
// Adding the centroid to the objects gives sums sum + c_pop and sum_sq + 2 S +
// c_pop.
inline double radius_compl(u64 sum, u64 sum_sq, u64 n, u64 c_pop, u64 S) {
    const double jt = isim(sum, sum_sq, static_cast<int64_t>(n));
    const double jt1 = isim(sum + c_pop, sum_sq + 2 * S + c_pop, static_cast<int64_t>(n + 1));
    return (jt1 * static_cast<double>(n + 1) - jt * static_cast<double>(n - 1)) / 2;
}

// Most dissimilar pair of n rows (O(N) approximation): fp1 is the first row least
// similar to the centroid of the rows, fp2 the first row least similar to fp1.
// The centroid uses the first n_columns bits only (the first n_columns / 8 bytes
// of the packed fingerprints). Writes the similarities of
// all rows to fp1 and fp2.
struct DissimilarPair {
    size_t fp1, fp2;
};

inline DissimilarPair most_dissimilar(const u64* rows, const uint32_t* pops, size_t n, size_t w64, size_t n_columns,
                                      double* sims1, double* sims2) {
    std::vector<u64> counts(w64 * 64, 0);
    for (size_t i = 0; i < n; ++i) for_each_bit(rows + i * w64, w64, [&](int p) { ++counts[p]; });
    std::vector<u64> centroid(w64, 0);
    for (size_t p = 0; p < n_columns; ++p)
        if (centroid_bit(counts[p], static_cast<int64_t>(n))) centroid[p >> 6] |= u64{1} << (p & 63);
    jt_sims(rows, pops, n, w64, centroid.data(), sims1);
    const size_t fp1 = first_min(sims1, n);
    jt_sims(rows, pops, n, w64, rows + fp1 * w64, sims1);
    const size_t fp2 = first_min(sims1, n);
    jt_sims(rows, pops, n, w64, rows + fp2 * w64, sims2);
    return {fp1, fp2};
}

// ------------------------------------------------------------ merge criteria

enum MergeKind : int {
    kDiameter = 0,
    kRadius = 1,
    kToleranceDiameter = 2,
    kFlexibleToleranceDiameter = 3,
    kToleranceRadius = 4,
    kToleranceLegacy = 5,
    kNever = 6,
};

struct MergeCriterion {
    int kind = kDiameter;
    double threshold = 0.65;
    double tolerance = 0.05;
    double decay = 0.0;  // size dependent tolerance (tolerance-* except legacy)
    double offset = 0.0;

    double tolerance_for(u64 old_n) const {
        const double t = tolerance * (std::exp(-decay * static_cast<double>(old_n)) - offset);
        return (0.0 > t) ? 0.0 : t;
    }
};

// Sizes and sums of the old cluster, the nominee and the merged cluster. The
// radius complements are only computed when the criterion needs them
// (new_rc() of the merged cluster, old_rc() of the old one).
struct MergeSums {
    u64 old_n, nom_n, new_n;
    u64 old_sum, old_sum_sq, new_sum, new_sum_sq;
};

template <typename NewRc, typename OldRc>
inline bool accept_merge(const MergeCriterion& c, const MergeSums& s, NewRc&& new_rc, OldRc&& old_rc) {
    const double thr = c.threshold;
    switch (c.kind) {
        case kDiameter:
            return isim(s.new_sum, s.new_sum_sq, static_cast<int64_t>(s.new_n)) >= thr;
        case kRadius:
            return new_rc() >= thr;
        case kToleranceDiameter:
        case kFlexibleToleranceDiameter: {
            const double new_dc = isim(s.new_sum, s.new_sum_sq, static_cast<int64_t>(s.new_n));
            if (new_dc < thr) return false;
            if (s.old_n == 1) return true;
            const double old_dc = isim(s.old_sum, s.old_sum_sq, static_cast<int64_t>(s.old_n));
            double ref = old_dc;
            if (c.kind == kFlexibleToleranceDiameter && thr < old_dc) ref = thr;  // min(old_dc, thr)
            return new_dc >= ref - c.tolerance_for(s.old_n);
        }
        case kToleranceRadius: {
            const double new_rc_v = new_rc();
            if (new_rc_v < thr) return false;
            if (s.old_n == 1) return true;
            return new_rc_v >= old_rc() - c.tolerance_for(s.old_n);
        }
        case kToleranceLegacy: {
            const double new_dc = isim(s.new_sum, s.new_sum_sq, static_cast<int64_t>(s.new_n));
            if (new_dc < thr) return false;
            if (s.old_n == 1 || s.nom_n != 1) return true;
            const double old_dc = isim(s.old_sum, s.old_sum_sq, static_cast<int64_t>(s.old_n));
            return (new_dc * static_cast<double>(s.new_n) - old_dc * static_cast<double>(s.old_n - 1)) / 2 >=
                   old_dc - c.tolerance;
        }
        default:
            return false;
    }
}

}  // namespace bblean
