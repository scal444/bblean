// Batched BitBirch fit (BitBirch(..., batch_size=..., n_threads=...)), included
// by bitbirch.cpp after the definition of Tree
//
// Multithreaded CPU version of the batched algorithm of nvMolKit's GPU BitBIRCH
// (bitBirch* kernels in nvMolKit src/bitbirch.cu). For the same input order,
// threshold, branching factor and batch size it builds the same tree, and gives
// the same clusters, as nvMolKit. This is a different algorithm from the serial
// insertion of Tree: the rows of a batch are routed against a snapshot of the
// tree and merged in rounds, leaves are split by their exact least similar
// pair, and the merge test is common >= threshold * (common + mismatches) (the
// 'diameter' criterion).
//
// Every round of a batch has the phases of the GPU version, separated by
// barriers:
//   route: descend the inner nodes (per row, read-only snapshot)
//   leaves (per routed leaf; on the GPU these are three phases that each touch
//     only that leaf, so they are fused per leaf, in GPU order): group
//     proposals against the snapshot -> group merges per proposed subcluster
//     (only after the first 16384 rows) -> remaining rows in batch order,
//     stopping when the leaf overflows
//   refresh: add the rows accepted this round to the summaries above them,
//     level by level (per changed node)
//   split seeds: per overflowing leaf (rows of the pair triangle in parallel)
//   repair: serial, by ascending leaf node id, with inner node split cascades
//   summaries of the split nodes (per node, chunks of bits in parallel)
//
// Exactness:
//  * Routing similarity: the GPU compares the correctly rounded FP32 quotient
//    |a & b| / |a | b| (1 if the union is empty; FP64 above 4096 bits).
//    Distinct fractions with denominators <= 4096 differ by more than an FP32
//    rounding step, so rounding keeps their order and ties; here the fractions
//    are compared exactly (cross multiplied). Closest subcluster: first maximum
//    in node order. Split seeds: first minimum in (i, j) pair order.
//  * Merge tests use exact integer terms (the GPU's FP64 partial sums are exact
//    integers below 2^53) and the same final FP64 comparison as the GPU,
//    common >= threshold * (common + mismatches), which is not always the same
//    as isim() >= threshold. With n members and column sums c:
//    common = (sum c^2 - sum c) / 2, mismatches = n sum c - sum c^2.
//  * Centroids: the majority rule of kernels.hpp (centroid_bit), updated
//    incrementally while n_samples only grows.
//  * Node ids are allocated serially in the same order as on the GPU, and
//    repair handles leaves by ascending node id.
//
// Storage: fingerprints are rows of 64 bit words (kernels.hpp), the same packed
// bytes that nvMolKit reads as 32 bit words. Linear sums are uint32 in the bit
// order of the words, like the sums of Tree. Each node keeps a contiguous copy
// of the centroids of its subclusters, in order (the GPU's "directory"), and
// each subcluster with a linear sum caches sum_kq and sum_kqsq, so that adding
// fingerprints only touches their set bits. After the fit, the tree is handed
// over to an empty Tree (hand_over), which then works as after a serial fit.
#pragma once

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#define BBLEAN_HAVE_MMAP 1
#endif

#include "kernels.hpp"

namespace bblean {

// ----------------------------------------------------------------- threads

inline void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// Threads that wait for work by spinning. A round has several short parallel
// phases separated by serial work, and waking sleeping threads (as OpenMP does
// by default) takes longer than the phases themselves. A phase waits only for
// the threads that joined it, so a descheduled idle thread never holds up a
// phase; idle threads yield the CPU after spinning for a while.
class SpinPool {
   public:
    explicit SpinPool(int threads) : threads_(std::max(1, threads)) {
        for (int id = 1; id < threads_; ++id) {
            workers_.emplace_back([this, id] { work(id); });
        }
    }
    SpinPool(const SpinPool&) = delete;
    SpinPool& operator=(const SpinPool&) = delete;
    ~SpinPool() {
        stop_.store(true);
        generation_.fetch_add(1);
        for (auto& w : workers_) {
            w.join();
        }
    }
    int size() const { return threads_; }

    // f(index, thread) for index in [0, count), handed out in chunks
    template <typename F>
    void parallel_for(int64_t count, int64_t chunk, F&& f) {
        if (count <= 0) {
            return;
        }
        if (threads_ == 1 || count <= chunk) {
            for (int64_t i = 0; i < count; ++i) {
                f(i, 0);
            }
            return;
        }
        std::atomic<int64_t> next{0};
        std::exception_ptr error;
        std::atomic<bool> failed{false};
        auto body = [&](int thread) {
            try {
                for (;;) {
                    const int64_t first =
                        next.fetch_add(chunk, std::memory_order_relaxed);
                    if (first >= count ||
                        failed.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const int64_t last = std::min(count, first + chunk);
                    for (int64_t i = first; i < last; ++i) {
                        f(i, thread);
                    }
                }
            } catch (...) {
                if (!failed.exchange(true)) {
                    error = std::current_exception();
                }
            }
        };
        job_ = body;
        open_.store(true);
        generation_.fetch_add(1);
        body(0);
        open_.store(false);
        while (active_.load() != 0) {
            cpu_relax();
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

   private:
    void work(int id) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            for (int spins = 0;
                 (g = generation_.load(std::memory_order_acquire)) == seen;
                 ++spins) {
                if (spins < 20000) {
                    cpu_relax();
                } else {
                    std::this_thread::yield();
                }
            }
            seen = g;
            if (stop_.load()) {
                return;
            }
            active_.fetch_add(1);
            if (open_.load()) {
                job_(id);
            }
            active_.fetch_sub(1);
        }
    }

    int threads_;
    std::vector<std::thread> workers_;
    std::function<void(int)> job_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> active_{0};
    std::atomic<bool> open_{false};
    std::atomic<bool> stop_{false};
};

// Zero filled array whose memory is only committed when used (anonymous mmap,
// with transparent huge pages where available: the tree arrays are read at
// random places, and with 4 KiB pages TLB misses take most of the time)
template <typename T>
class HugeArray {
   public:
    HugeArray() = default;
    explicit HugeArray(size_t count)
        : bytes_(std::max<size_t>(count * sizeof(T), 1)) {
#if defined(BBLEAN_HAVE_MMAP)
        void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            throw std::bad_alloc();
        }
#if defined(MADV_HUGEPAGE)
        madvise(p, bytes_, MADV_HUGEPAGE);
#endif
#else
        void* p = ::operator new(bytes_);
        std::memset(p, 0, bytes_);
#endif
        data_ = static_cast<T*>(p);
    }
    HugeArray(const HugeArray&) = delete;
    HugeArray& operator=(const HugeArray&) = delete;
    HugeArray& operator=(HugeArray&& o) noexcept {
        std::swap(data_, o.data_);
        std::swap(bytes_, o.bytes_);
        return *this;
    }
    ~HugeArray() {
        if (data_ == nullptr) {
            return;
        }
#if defined(BBLEAN_HAVE_MMAP)
        munmap(data_, bytes_);
#else
        ::operator delete(data_);
#endif
    }
    T* get() const { return data_; }

   private:
    T* data_ = nullptr;
    size_t bytes_ = 0;
};

// ------------------------------------------------------------------ engine

class BatchedFit {
   public:
    // Rows before this many are fitted without group proposals (nvMolKit's
    // orderedWarmupSize)
    static constexpr int kWarmup = 16384;
    // Words per task when summing linear sums in parallel
    static constexpr int kChunkWords = 4;

    BatchedFit(const uint64_t* fps, int n, int w64, double threshold, int bf,
               int batch_size, SpinPool& pool)
        : fps_(fps),
          n_(n),
          w64_(w64),
          n_bits_(w64 * 64),
          threshold_(threshold),
          bf_(bf),
          batch_size_(batch_size),
          pool_(pool) {
        const int min_node = bf / 2 + bf % 2;
        max_nodes_ = n / (min_node - 1) + 64;
        max_ents_ = n + max_nodes_ + 64;
        node_ents_.resize(max_nodes_);
        node_stride_ = bf + 2;
        node_cent_ = HugeArray<uint64_t>(static_cast<size_t>(max_nodes_) *
                                         node_stride_ * w64_);
        node_pop_ = HugeArray<uint32_t>(static_cast<size_t>(max_nodes_) *
                                        node_stride_);
        node_parent_.assign(max_nodes_, -1);
        node_parent_ent_.assign(max_nodes_, -1);
        node_leaf_.assign(max_nodes_, 0);
        delta_slot_.assign(max_nodes_, -1);
        dirty_.assign(max_nodes_, 0);
        leaf_range_.assign(max_nodes_, -1);
        ent_child_.resize(max_ents_);
        ent_slot_.resize(max_ents_);
        ent_fp_.resize(max_ents_);
        ent_node_.resize(max_ents_);
        ent_pos_.resize(max_ents_);
        ent_n_.resize(max_ents_);
        ent_pop_.resize(max_ents_);
        ent_sum_kq_.resize(max_ents_);
        ent_sum_kqsq_.resize(max_ents_);
        cent_ = HugeArray<uint64_t>(static_cast<size_t>(max_ents_) * w64_);
        // Every subcluster gets a linear sum at most once, so max_ents_ slots
        // are enough
        sums_ = HugeArray<uint32_t>(static_cast<size_t>(max_ents_) * n_bits_);
        fp_pop_.resize(n);
        pool_.parallel_for(n, 4096, [&](int64_t i, int) {
            fp_pop_[i] = popcount(fp(static_cast<int>(i)), w64_);
        });
        labels_.assign(n, -1);
        root_ = new_node(true, -1);
    }

    void run() {
        const int batch_cap = std::min(batch_size_, n_);
        keys_.resize(batch_cap);
        for (int begin = 0; begin < n_;) {
            const bool groups = begin >= kWarmup;
            const int boundary = begin < kWarmup ? std::min(kWarmup, n_) : n_;
            const int count = std::min(batch_cap, boundary - begin);
            int pending = count;
            while (pending > 0) {
                round(begin, count, groups);
                int p = 0;
                for (int i = begin; i < begin + count; ++i) {
                    p += labels_[i] < 0;
                }
                if (p >= pending) {
                    throw std::runtime_error(
                        "batched BitBirch fit failed to make progress");
                }
                pending = p;
            }
            begin += count;
        }
    }

    // Move the tree into an empty Tree (same nodes, subclusters, centroids
    // and linear sums). Leaves are chained in depth first order, and the
    // molecules of each subcluster are in row order, with indices ids[row] (or
    // start + row if ids is null).
    void hand_over(Tree& t, const int64_t* ids, int64_t start);

   private:
    // ---------------------------------------------------------- storage

    const uint64_t* fp(int mol) const {
        return fps_ + static_cast<size_t>(mol) * w64_;
    }
    uint64_t* cent(int e) {
        return cent_.get() + static_cast<size_t>(e) * w64_;
    }
    uint32_t* ls(int slot) {
        return sums_.get() + static_cast<size_t>(slot) * n_bits_;
    }
    uint64_t* node_cent(int node) {
        return node_cent_.get() +
               static_cast<size_t>(node) * node_stride_ * w64_;
    }
    uint32_t* node_pop(int node) {
        return node_pop_.get() + static_cast<size_t>(node) * node_stride_;
    }

    // Routing similarity as an exact fraction
    struct Frac {
        int64_t num, den;
    };
    static Frac frac(uint32_t inter, uint32_t uni) {
        return uni == 0 ? Frac{1, 1} : Frac{inter, uni};
    }
    static bool greater(Frac a, Frac b) {
        return a.num * b.den > b.num * a.den;
    }
    static bool less(Frac a, Frac b) { return a.num * b.den < b.num * a.den; }

    uint32_t and_popcount(const uint64_t* a, const uint64_t* b) const {
        uint32_t out = 0;
        for_each_intersection(a, 1, w64_, b,
                              [&](size_t, uint32_t inter) { out = inter; });
        return out;
    }

    int new_node(bool leaf, int parent) {
        const int node = num_nodes_++;
        if (node >= max_nodes_) {
            throw std::runtime_error("batched BitBirch: too many nodes");
        }
        node_ents_[node].clear();
        node_ents_[node].reserve(bf_ + 2);
        node_parent_[node] = parent;
        node_parent_ent_[node] = -1;
        node_leaf_[node] = leaf;
        return node;
    }

    int new_entry() {
        const int e = num_ents_.fetch_add(1, std::memory_order_relaxed);
        if (e >= max_ents_) {
            throw std::runtime_error("batched BitBirch: too many subclusters");
        }
        ent_child_[e] = -1;
        ent_slot_[e] = -1;
        ent_fp_[e] = -1;
        ent_n_[e] = 0;
        ent_node_[e] = -1;
        ent_pos_[e] = -1;
        ent_pop_[e] = 0;
        ent_sum_kq_[e] = 0;
        ent_sum_kqsq_[e] = 0;
        return e;
    }

    int parent_entry(int node) const {
        return node_parent_[node] < 0 ? -1 : node_parent_ent_[node];
    }

    void set_node_row(int node, int pos, int e) {
        std::memcpy(node_cent(node) + static_cast<size_t>(pos) * w64_, cent(e),
                    sizeof(uint64_t) * w64_);
        node_pop(node)[pos] = ent_pop_[e];
    }

    void append_to_node(int node, int e) {
        ent_node_[e] = node;
        ent_pos_[e] = static_cast<int>(node_ents_[node].size());
        if (ent_pos_[e] >= node_stride_) {
            throw std::runtime_error("batched BitBirch: node overflow");
        }
        node_ents_[node].push_back(e);
        set_node_row(node, ent_pos_[e], e);
    }

    // Replace the subclusters of a node (after a split)
    void set_node_entries(int node, std::vector<int>& ents) {
        node_ents_[node].swap(ents);
        for (size_t i = 0; i < node_ents_[node].size(); ++i) {
            const int e = node_ents_[node][i];
            ent_node_[e] = node;
            ent_pos_[e] = static_cast<int>(i);
            set_node_row(node, static_cast<int>(i), e);
        }
    }

    void sync_node_row(int e) {
        if (ent_node_[e] >= 0) {
            set_node_row(ent_node_[e], ent_pos_[e], e);
        }
    }

    // Give a subcluster its own linear sum (single fingerprints start from
    // their fingerprint)
    void materialize(int e) {
        if (ent_slot_[e] >= 0) {
            return;
        }
        const int slot = num_slots_.fetch_add(1, std::memory_order_relaxed);
        if (slot >= max_ents_) {
            throw std::runtime_error("batched BitBirch: too many linear sums");
        }
        ent_slot_[e] = slot;
        uint32_t* L = ls(slot);
        std::memset(L, 0, sizeof(uint32_t) * n_bits_);
        const int f = ent_fp_[e];
        if (f >= 0) {
            for_each_bit(fp(f), w64_, [&](int p) { L[p] = 1; });
            ent_sum_kq_[e] = ent_sum_kqsq_[e] = fp_pop_[f];
        } else {
            ent_sum_kq_[e] = ent_sum_kqsq_[e] = 0;
        }
        ent_fp_[e] = -1;
    }

    // Centroid from the linear sum
    void refresh_centroid(int e) {
        const uint32_t* L = ls(ent_slot_[e]);
        const int64_t n = ent_n_[e];
        uint64_t* c = cent(e);
        for (int k = 0; k < w64_; ++k) {
            uint64_t word = 0;
            for (int b = 0; b < 64; ++b) {
                word |= static_cast<uint64_t>(centroid_bit(L[k * 64 + b], n))
                        << b;
            }
            c[k] = word;
        }
        ent_pop_[e] = popcount(c, w64_);
        sync_node_row(e);
    }

    // Centroid after n_samples grew and only the sums of the bits visited by
    // for_touched grew. The majority threshold never decreases, so an
    // untouched bit that was clear stays clear: only set bits and touched bits
    // need checking.
    template <typename ForTouched>
    void update_centroid(int e, ForTouched&& for_touched) {
        const uint32_t* L = ls(ent_slot_[e]);
        const int64_t n = ent_n_[e];
        uint64_t* c = cent(e);
        for (int k = 0; k < w64_; ++k) {
            uint64_t keep = c[k];
            for (uint64_t w = c[k]; w; w &= w - 1) {
                const int b = ctz64(w);
                if (!centroid_bit(L[k * 64 + b], n)) {
                    keep &= ~(uint64_t{1} << b);
                }
            }
            c[k] = keep;
        }
        for_touched([&](int p) {
            const uint64_t bit = uint64_t{1} << (p & 63);
            if (centroid_bit(L[p], n)) {
                c[p >> 6] |= bit;
            } else {
                c[p >> 6] &= ~bit;
            }
        });
        ent_pop_[e] = popcount(c, w64_);
        sync_node_row(e);
    }

    void recompute_sums(int e) {
        const uint32_t* L = ls(ent_slot_[e]);
        uint64_t sum_kq = 0, sum_kqsq = 0;
        for (int p = 0; p < n_bits_; ++p) {
            const uint64_t v = L[p];
            sum_kq += v;
            sum_kqsq += v * v;
        }
        ent_sum_kq_[e] = sum_kq;
        ent_sum_kqsq_[e] = sum_kqsq;
    }

    // The merge test of the GPU version, from the sums of the merged cluster
    bool accept(uint64_t sum_kq, uint64_t sum_kqsq, uint64_t n) const {
        if (n <= 1) {
            return threshold_ <= 1.0;
        }
        const uint64_t common = (sum_kqsq - sum_kq) / 2;
        const double denominator =
            static_cast<double>(common + (n * sum_kq - sum_kqsq));
        return denominator > 0.0
                   ? static_cast<double>(common) >= threshold_ * denominator
                   : threshold_ <= 1.0;
    }

    // Merge test of subcluster e and molecule mol
    bool merge_test(int e, int mol) {
        const uint64_t* q = fp(mol);
        const uint64_t q_pop = fp_pop_[mol];
        uint64_t sum_kq, sum_kqsq, s = 0;  // s = sum_{j in q} L_j
        if (ent_fp_[e] >= 0) {
            // Single fingerprint f: L = f
            sum_kq = sum_kqsq = fp_pop_[ent_fp_[e]];
            s = and_popcount(fp(ent_fp_[e]), q);
        } else {
            const uint32_t* L = ls(ent_slot_[e]);
            sum_kq = ent_sum_kq_[e];
            sum_kqsq = ent_sum_kqsq_[e];
            for_each_bit(q, w64_, [&](int p) { s += L[p]; });
        }
        // Merged sums L + q: sum_kq + |q| and sum_kqsq + 2 s + |q|
        return accept(sum_kq + q_pop, sum_kqsq + 2 * s + q_pop,
                      static_cast<uint64_t>(ent_n_[e]) + 1);
    }

    void add_fingerprint(int e, int mol) {
        materialize(e);
        ++ent_n_[e];
        uint32_t* L = ls(ent_slot_[e]);
        const uint64_t* q = fp(mol);
        uint64_t s = 0;
        for_each_bit(q, w64_, [&](int p) { s += L[p]++; });
        ent_sum_kq_[e] += fp_pop_[mol];
        ent_sum_kqsq_[e] += 2 * s + fp_pop_[mol];
        update_centroid(e, [&](auto&& visit) { for_each_bit(q, w64_, visit); });
    }

    int new_leaf_entry(int mol) {
        const int e = new_entry();
        ent_n_[e] = 1;
        ent_fp_[e] = mol;
        std::memcpy(cent(e), fp(mol), sizeof(uint64_t) * w64_);
        ent_pop_[e] = fp_pop_[mol];
        return e;
    }

    // First subcluster (node order) with the largest routing similarity
    int closest(int node, int mol) {
        const uint32_t q_pop = fp_pop_[mol];
        const uint32_t* P = node_pop(node);
        int best = -1;
        Frac best_value = {-1, 1};
        for_each_intersection(
            node_cent(node), node_ents_[node].size(), w64_, fp(mol),
            [&](size_t i, uint32_t inter) {
                const Frac v = frac(inter, q_pop + P[i] - inter);
                if (greater(v, best_value)) {
                    best_value = v;
                    best = static_cast<int>(i);
                }
            });
        return node_ents_[node][best];
    }

    int num_chunks() const { return (w64_ + kChunkWords - 1) / kChunkWords; }

    // Set each target subcluster (node, target) to the sum of the subclusters
    // of the node. Targets are distinct; the work is split in word chunks.
    void summarize(const std::vector<std::pair<int, int>>& list) {
        if (list.empty()) {
            return;
        }
        for (const auto& [node, target] : list) {
            materialize(target);
            uint32_t count = 0;
            for (const int e : node_ents_[node]) {
                count += ent_n_[e];
            }
            ent_n_[target] = count;
        }
        const int chunks = num_chunks();
        pool_.parallel_for(
            static_cast<int64_t>(list.size()) * chunks, 1,
            [&](int64_t task, int) {
                const auto [node, target] = list[task / chunks];
                const int c = static_cast<int>(task % chunks);
                const int w0 = c * kChunkWords;
                const int w1 = std::min(w64_, w0 + kChunkWords);
                uint32_t* S = ls(ent_slot_[target]);
                std::memset(S + w0 * 64, 0, sizeof(uint32_t) * (w1 - w0) * 64);
                for (const int e : node_ents_[node]) {
                    if (ent_fp_[e] >= 0) {
                        for_each_bit(fp(ent_fp_[e]) + w0, w1 - w0,
                                     [&](int p) { ++S[w0 * 64 + p]; });
                    } else {
                        const uint32_t* L = ls(ent_slot_[e]);
                        for (int p = w0 * 64; p < w1 * 64; ++p) {
                            S[p] += L[p];
                        }
                    }
                }
            });
        pool_.parallel_for(static_cast<int64_t>(list.size()), 1,
                           [&](int64_t i, int) {
                               recompute_sums(list[i].second);
                               refresh_centroid(list[i].second);
                           });
    }

    // Row i of the seed search: first minimum over j > i of sim(i, j)
    std::pair<Frac, int> seed_row(int node, int i) {
        const uint64_t* C = node_cent(node);
        const uint32_t* P = node_pop(node);
        const int size = static_cast<int>(node_ents_[node].size());
        Frac best = {2, 1};
        int best_j = -1;
        for_each_intersection(
            C + static_cast<size_t>(i + 1) * w64_, size - i - 1, w64_,
            C + static_cast<size_t>(i) * w64_, [&](size_t k, uint32_t inter) {
                const int j = i + 1 + static_cast<int>(k);
                const Frac v = frac(inter, P[i] + P[j] - inter);
                if (less(v, best)) {
                    best = v;
                    best_j = j;
                }
            });
        return {best, best_j};
    }

    // Exact least similar pair (positions), ties to the smallest (i, j)
    std::pair<int, int> find_seeds(int node) {
        const int size = static_cast<int>(node_ents_[node].size());
        Frac best = {2, 1};
        int li = -1, ri = -1;
        for (int i = 0; i + 1 < size; ++i) {
            const auto [v, j] = seed_row(node, i);
            if (less(v, best)) {
                best = v;
                li = i;
                ri = j;
            }
        }
        return {li, ri};
    }

    // The two halves of a split with seeds at positions lp and rp: in node
    // order, each subcluster goes to the more similar seed (ties to the
    // smaller half, then the left one), and no half gets more than half
    // (rounded up) of the subclusters
    void split_plan(int node, int lp, int rp, std::vector<int>& lhs,
                    std::vector<int>& rhs) {
        const auto& E = node_ents_[node];
        const uint64_t* C = node_cent(node);
        const uint32_t* P = node_pop(node);
        const int size = static_cast<int>(E.size());
        const int max_group = (size + 1) / 2;
        std::vector<uint32_t> inter_l(size), inter_r(size);
        for_each_intersection(
            C, size, w64_, C + static_cast<size_t>(lp) * w64_,
            [&](size_t i, uint32_t inter) { inter_l[i] = inter; });
        for_each_intersection(
            C, size, w64_, C + static_cast<size_t>(rp) * w64_,
            [&](size_t i, uint32_t inter) { inter_r[i] = inter; });
        int nl = 1, nr = 1;
        lhs.assign(1, E[lp]);
        rhs.assign(1, E[rp]);
        for (int i = 0; i < size; ++i) {
            if (i == lp || i == rp) {
                continue;
            }
            const Frac lsim = frac(inter_l[i], P[i] + P[lp] - inter_l[i]);
            const Frac rsim = frac(inter_r[i], P[i] + P[rp] - inter_r[i]);
            const bool tie = !greater(lsim, rsim) && !less(lsim, rsim);
            bool to_left = greater(lsim, rsim) || (tie && nl <= nr);
            if (nl >= max_group) {
                to_left = false;
            } else if (nr >= max_group) {
                to_left = true;
            }
            if (to_left) {
                lhs.push_back(E[i]);
                ++nl;
            } else {
                rhs.push_back(E[i]);
                ++nr;
            }
        }
    }

    // Add the new sibling of a split node to its parent (or to a new root);
    // returns the parent
    int link_sibling(int node, int sibling) {
        const int parent = node_parent_[node];
        if (parent < 0) {
            const int new_root = new_node(false, -1);
            const int lhs = new_entry();
            const int rhs = new_entry();
            node_parent_[node] = node_parent_[sibling] = new_root;
            node_parent_ent_[node] = lhs;
            node_parent_ent_[sibling] = rhs;
            ent_child_[lhs] = node;
            ent_child_[rhs] = sibling;
            append_to_node(new_root, lhs);
            append_to_node(new_root, rhs);
            root_ = new_root;
            return new_root;
        }
        const int e = new_entry();
        ent_child_[e] = sibling;
        node_parent_ent_[sibling] = e;
        append_to_node(parent, e);
        return parent;
    }

    // Split an overflowing inner node, and its parents if they overflow
    void split_cascade(int node) {
        std::vector<int> lhs, rhs;
        while (static_cast<int>(node_ents_[node].size()) > bf_) {
            const auto [lp, rp] = find_seeds(node);
            const int old_parent = node_parent_[node];
            const int sibling =
                new_node(node_leaf_[node] != 0, node_parent_[node]);
            split_plan(node, lp, rp, lhs, rhs);
            set_node_entries(node, lhs);
            set_node_entries(sibling, rhs);
            if (!node_leaf_[node]) {
                for (const int e : node_ents_[node]) {
                    node_parent_[ent_child_[e]] = node;
                }
                for (const int e : node_ents_[sibling]) {
                    node_parent_[ent_child_[e]] = sibling;
                }
            }
            const int parent = link_sibling(node, sibling);
            summarize({{node, parent_entry(node)},
                       {sibling, parent_entry(sibling)}});
            if (old_parent < 0) {
                return;
            }
            node = parent;
        }
    }

    // Group merge of one proposed subcluster (proposals[first:end]): merge the
    // whole group if the merge test passes, otherwise test and add its
    // molecules one by one
    void merge_group(const std::vector<std::pair<int, int>>& proposals,
                     int first, int end, int thread) {
        std::vector<uint32_t>& delta = delta_[thread];
        std::vector<int>& touched = touched_[thread];
        const int e = proposals[first].first;
        const uint64_t n = static_cast<uint64_t>(ent_n_[e]) + (end - first);
        touched.clear();
        uint64_t added = 0;
        for (int i = first; i < end; ++i) {
            const int mol = proposals[i].second;
            added += fp_pop_[mol];
            for_each_bit(fp(mol), w64_, [&](int p) {
                if (delta[p]++ == 0) {
                    touched.push_back(p);
                }
            });
        }
        // Merged sums L + delta: only touched bits change sum_kqsq
        uint64_t sum_kq, sum_kqsq;
        if (ent_fp_[e] >= 0) {
            const uint64_t* f = fp(ent_fp_[e]);
            sum_kq = sum_kqsq = fp_pop_[ent_fp_[e]];
            for (const int p : touched) {
                const uint64_t l = (f[p >> 6] >> (p & 63)) & 1, d = delta[p];
                sum_kqsq += 2 * l * d + d * d;
            }
        } else {
            const uint32_t* L = ls(ent_slot_[e]);
            sum_kq = ent_sum_kq_[e];
            sum_kqsq = ent_sum_kqsq_[e];
            for (const int p : touched) {
                const uint64_t d = delta[p];
                sum_kqsq += 2 * static_cast<uint64_t>(L[p]) * d + d * d;
            }
        }
        sum_kq += added;
        if (accept(sum_kq, sum_kqsq, n)) {
            materialize(e);
            uint32_t* L = ls(ent_slot_[e]);
            for (const int p : touched) {
                L[p] += delta[p];
            }
            ent_n_[e] = static_cast<uint32_t>(n);
            ent_sum_kq_[e] = sum_kq;
            ent_sum_kqsq_[e] = sum_kqsq;
            update_centroid(e, [&](auto&& visit) {
                for (const int p : touched) {
                    visit(p);
                }
            });
            for (int i = first; i < end; ++i) {
                labels_[proposals[i].second] = e;
            }
        } else {
            for (int i = first; i < end; ++i) {
                const int mol = proposals[i].second;
                if (merge_test(e, mol)) {
                    add_fingerprint(e, mol);
                    labels_[mol] = e;
                }
            }
        }
        for (const int p : touched) {
            delta[p] = 0;
        }
    }

    // Sorted (leaf, molecule) pairs of the routed rows, and the ranges of
    // equal leaves
    void group_by_leaf(int begin, int count) {
        pairs_.clear();
        for (int o = 0; o < count; ++o) {
            if (keys_[o] != INT_MAX) {
                pairs_.emplace_back(keys_[o], begin + o);
            }
        }
        std::sort(pairs_.begin(), pairs_.end());
        ranges_.clear();
        for (size_t i = 0; i < pairs_.size();) {
            size_t j = i + 1;
            while (j < pairs_.size() && pairs_[j].first == pairs_[i].first) {
                ++j;
            }
            ranges_.emplace_back(static_cast<int>(i), static_cast<int>(j));
            i = j;
        }
    }

    // Leaf phase for the rows routed to one leaf (range r of pairs_)
    void fit_leaf(int r, bool groups, int thread) {
        const int node = pairs_[ranges_[r].first].first;
        if (groups && !node_ents_[node].empty()) {
            // Group proposals against the snapshot of the leaf, then group
            // merges per proposed subcluster
            auto& proposals = proposals_[thread];
            proposals.clear();
            for (int i = ranges_[r].first; i < ranges_[r].second; ++i) {
                const int mol = pairs_[i].second;
                const int e = closest(node, mol);
                if (merge_test(e, mol)) {
                    proposals.emplace_back(e, mol);
                }
            }
            std::sort(proposals.begin(), proposals.end());
            for (size_t first = 0; first < proposals.size();) {
                size_t end = first + 1;
                while (end < proposals.size() &&
                       proposals[end].first == proposals[first].first) {
                    ++end;
                }
                merge_group(proposals, static_cast<int>(first),
                            static_cast<int>(end), thread);
                first = end;
            }
        }
        // Remaining rows in batch order; the rest retry after a split
        for (int i = ranges_[r].first; i < ranges_[r].second; ++i) {
            const int mol = pairs_[i].second;
            if (labels_[mol] >= 0) {
                continue;
            }
            int e = node_ents_[node].empty() ? -1 : closest(node, mol);
            if (e >= 0 && merge_test(e, mol)) {
                add_fingerprint(e, mol);
            } else {
                e = new_leaf_entry(mol);
                append_to_node(node, e);
            }
            labels_[mol] = e;
            if (static_cast<int>(node_ents_[node].size()) > bf_) {
                break;
            }
        }
    }

    // Add the rows accepted this round to the summaries above them, level by
    // level (one task per changed node)
    void refresh_summaries(int num_leaves) {
        int leaf_depth = 0;
        for (int nd = 0; node_parent_[nd] >= 0; nd = node_parent_[nd]) {
            ++leaf_depth;
        }
        levels_.assign(leaf_depth + 1, {});
        for (int r = 0; r < num_leaves; ++r) {
            const int leaf = pairs_[ranges_[r].first].first;
            leaf_range_[leaf] = r;
            int level = leaf_depth;
            for (int ch = leaf; node_parent_[ch] >= 0;
                 ch = node_parent_[ch], --level) {
                if (!dirty_[ch]) {
                    dirty_[ch] = 1;
                    levels_[level].push_back(ch);
                }
            }
        }
        // The rows accepted below a changed node this round; each level adds
        // their set bits to the summary of the node
        size_t num_dirty = 0;
        for (int level = leaf_depth; level > 0; --level) {
            for (const int nd : levels_[level]) {
                delta_slot_[nd] = static_cast<int>(num_dirty++);
            }
        }
        if (accepted_.size() < num_dirty) {
            accepted_.resize(num_dirty);
        }
        for (int level = leaf_depth; level > 0; --level) {
            const auto& nodes = levels_[level];
            for (const int nd : nodes) {
                materialize(parent_entry(nd));
            }
            pool_.parallel_for(
                static_cast<int64_t>(nodes.size()), 1,
                [&](int64_t idx, int thread) {
                    refresh_summary(nodes[idx], thread);
                });
        }
        for (int level = leaf_depth; level > 0; --level) {
            for (const int nd : levels_[level]) {
                delta_slot_[nd] = -1;
                dirty_[nd] = 0;
            }
        }
    }

    void refresh_summary(int nd, int thread) {
        const int target = parent_entry(nd);
        std::vector<int>& acc = accepted_[delta_slot_[nd]];
        acc.clear();
        if (node_leaf_[nd]) {
            const int r = leaf_range_[nd];
            for (int i = ranges_[r].first; i < ranges_[r].second; ++i) {
                if (labels_[pairs_[i].second] >= 0) {
                    acc.push_back(pairs_[i].second);
                }
            }
        } else {
            for (const int e : node_ents_[nd]) {
                const int cs = delta_slot_[ent_child_[e]];
                if (cs >= 0) {
                    acc.insert(acc.end(), accepted_[cs].begin(),
                               accepted_[cs].end());
                }
            }
        }
        std::vector<uint32_t>& D = delta_[thread];
        std::vector<int>& touched = touched_[thread];
        touched.clear();
        for (const int mol : acc) {
            for_each_bit(fp(mol), w64_, [&](int p) {
                if (D[p]++ == 0) {
                    touched.push_back(p);
                }
            });
        }
        uint32_t* L = ls(ent_slot_[target]);
        uint64_t add_kq = 0, add_kqsq = 0;
        for (const int p : touched) {
            const uint64_t d = D[p], l = L[p];
            add_kq += d;
            add_kqsq += 2 * l * d + d * d;
            L[p] = static_cast<uint32_t>(l + d);
            D[p] = 0;
        }
        ent_n_[target] += static_cast<uint32_t>(acc.size());
        ent_sum_kq_[target] += add_kq;
        ent_sum_kqsq_[target] += add_kqsq;
        update_centroid(target, [&](auto&& visit) {
            for (const int p : touched) {
                visit(p);
            }
        });
    }

    // Seeds and halves of every overflowing leaf (rows of the pair triangles
    // in parallel, reduced in order)
    void plan_leaf_splits(int num_leaves) {
        split_leaves_.clear();
        for (int r = 0; r < num_leaves; ++r) {
            const int nd = pairs_[ranges_[r].first].first;
            leaf_range_[nd] = -1;
            if (static_cast<int>(node_ents_[nd].size()) > bf_) {
                split_leaves_.push_back(nd);  // ascending node id
            }
        }
        const int num_splits = static_cast<int>(split_leaves_.size());
        plan_l_.resize(num_splits);
        plan_r_.resize(num_splits);
        if (num_splits == 0) {
            return;
        }
        row_tasks_.clear();
        row_offset_.assign(num_splits + 1, 0);
        for (int s = 0; s < num_splits; ++s) {
            const int size =
                static_cast<int>(node_ents_[split_leaves_[s]].size());
            row_offset_[s + 1] = row_offset_[s] + size - 1;
            for (int i = 0; i + 1 < size; ++i) {
                row_tasks_.emplace_back(s, i);
            }
        }
        row_best_.resize(row_tasks_.size());
        pool_.parallel_for(static_cast<int64_t>(row_tasks_.size()), 2,
                           [&](int64_t t, int) {
                               row_best_[t] =
                                   seed_row(split_leaves_[row_tasks_[t].first],
                                            row_tasks_[t].second);
                           });
        pool_.parallel_for(num_splits, 1, [&](int64_t s, int) {
            Frac best = {2, 1};
            int lp = -1, rp = -1;
            for (int t = row_offset_[s]; t < row_offset_[s + 1]; ++t) {
                if (less(row_best_[t].first, best)) {
                    best = row_best_[t].first;
                    lp = t - row_offset_[s];
                    rp = row_best_[t].second;
                }
            }
            split_plan(split_leaves_[s], lp, rp, plan_l_[s], plan_r_[s]);
        });
    }

    // Split the leaves (serial, ascending leaf id), with inner split cascades
    void split_leaves() {
        std::vector<std::pair<int, int>> pending;
        for (size_t s = 0; s < split_leaves_.size(); ++s) {
            const int node = split_leaves_[s];
            const int sibling = new_node(true, node_parent_[node]);
            set_node_entries(node, plan_l_[s]);
            set_node_entries(sibling, plan_r_[s]);
            const int parent = link_sibling(node, sibling);
            pending.emplace_back(node, parent_entry(node));
            pending.emplace_back(sibling, parent_entry(sibling));
            if (static_cast<int>(node_ents_[parent].size()) > bf_) {
                summarize(pending);
                pending.clear();
                split_cascade(parent);
            }
        }
        summarize(pending);
    }

    void round(int begin, int count, bool groups) {
        const int T = pool_.size();
        if (delta_.size() != static_cast<size_t>(T)) {
            delta_.assign(T, std::vector<uint32_t>(n_bits_, 0));
            touched_.assign(T, {});
            proposals_.assign(T, {});
        }
        // Route against the snapshot of the tree (read only)
        pool_.parallel_for(count, 4, [&](int64_t o, int) {
            const int mol = begin + static_cast<int>(o);
            keys_[o] = INT_MAX;
            if (labels_[mol] >= 0) {
                return;
            }
            int node = root_;
            while (!node_leaf_[node]) {
                node = ent_child_[closest(node, mol)];
            }
            keys_[o] = node;
        });
        group_by_leaf(begin, count);
        const int num_leaves = static_cast<int>(ranges_.size());
        pool_.parallel_for(num_leaves, 1, [&](int64_t r, int thread) {
            fit_leaf(static_cast<int>(r), groups, thread);
        });
        refresh_summaries(num_leaves);
        plan_leaf_splits(num_leaves);
        split_leaves();
    }

    const uint64_t* fps_;
    int n_, w64_, n_bits_;
    double threshold_;
    int bf_, batch_size_;
    SpinPool& pool_;
    int max_nodes_ = 0, max_ents_ = 0;
    // nodes
    std::vector<std::vector<int>> node_ents_;
    int node_stride_ = 0;
    HugeArray<uint64_t> node_cent_;
    HugeArray<uint32_t> node_pop_;
    std::vector<int> node_parent_, node_parent_ent_, delta_slot_, leaf_range_;
    std::vector<uint8_t> node_leaf_, dirty_;
    int num_nodes_ = 0, root_ = -1;
    // subclusters
    std::atomic<int> num_ents_{0};
    std::vector<int> ent_child_, ent_slot_, ent_fp_, ent_node_, ent_pos_;
    std::vector<uint32_t> ent_n_, ent_pop_;
    std::vector<uint64_t> ent_sum_kq_, ent_sum_kqsq_;
    HugeArray<uint64_t> cent_;
    HugeArray<uint32_t> sums_;
    std::atomic<int> num_slots_{0};
    std::vector<uint32_t> fp_pop_;
    std::vector<int> labels_;  // subcluster of each row (-1: not yet placed)
    // round scratch
    std::vector<int> keys_, split_leaves_, row_offset_;
    std::vector<std::pair<int, int>> pairs_, ranges_, row_tasks_;
    std::vector<std::pair<Frac, int>> row_best_;
    std::vector<std::vector<int>> levels_, plan_l_, plan_r_, accepted_;
    // per thread scratch
    std::vector<std::vector<uint32_t>> delta_;
    std::vector<std::vector<int>> touched_;
    std::vector<std::vector<std::pair<int, int>>> proposals_;
};

inline void BatchedFit::hand_over(Tree& t, const int64_t* ids, int64_t start) {
    if (t.initialized() || !t.nodes_.empty() || !t.ents_.empty() ||
        !t.mem_id_.empty()) {
        throw std::runtime_error("a batched fit requires an empty tree");
    }
    if (t.w64_ != w64_) {
        throw std::runtime_error("fingerprint size mismatch");
    }
    // Number the nodes depth first (preorder), and their subclusters in that
    // order (serial), then fill nodes, subclusters and molecules in parallel
    const int num_ents = num_ents_.load();
    std::vector<int32_t> node_map(num_nodes_, -1), ent_map(num_ents, -1);
    std::vector<int> order, ents;
    std::vector<int32_t> leaves;
    std::vector<int> stack{root_};
    while (!stack.empty()) {
        const int nd = stack.back();
        stack.pop_back();
        node_map[nd] = static_cast<int32_t>(order.size());
        order.push_back(nd);
        if (node_leaf_[nd]) {
            leaves.push_back(node_map[nd]);
        }
        const auto& E = node_ents_[nd];
        for (const int e : E) {
            ent_map[e] = static_cast<int32_t>(ents.size());
            ents.push_back(e);
        }
        for (size_t i = E.size(); i-- > 0;) {
            if (ent_child_[E[i]] >= 0) {
                stack.push_back(ent_child_[E[i]]);
            }
        }
    }
    for (const int e : ents) {
        if (ent_slot_[e] < 0 && (ent_child_[e] >= 0 || ent_n_[e] != 1)) {
            throw std::runtime_error(
                "batched BitBirch: subcluster without linear sum");
        }
    }
    t.nodes_.resize(order.size());
    t.ents_.resize(ents.size());
    pool_.parallel_for(
        static_cast<int64_t>(order.size()), 1, [&](int64_t k, int) {
            const int nd = order[k];
            Node& node = t.nodes_[k];
            node.bf = bf_;
            node.leaf = node_leaf_[nd] != 0;
            const auto& E = node_ents_[nd];
            node.ents.reserve(bf_ + 1);
            for (const int e : E) {
                node.ents.push_back(ent_map[e]);
            }
            node.cent.assign(static_cast<size_t>(bf_ + 1) * w64_, 0);
            node.pop.assign(bf_ + 1, 0);
            std::memcpy(node.cent.data(), node_cent(nd),
                        sizeof(uint64_t) * w64_ * E.size());
            std::memcpy(node.pop.data(), node_pop(nd),
                        sizeof(uint32_t) * E.size());
        });
    // Subclusters: n_samples, children and linear sums (single fingerprints
    // keep an implicit linear sum, as in Tree)
    pool_.parallel_for(
        static_cast<int64_t>(ents.size()), 64, [&](int64_t k, int) {
            const int e = ents[k];
            Entry& entry = t.ents_[k];
            entry.n_samples = ent_n_[e];
            entry.child = ent_child_[e] >= 0 ? node_map[ent_child_[e]] : -1;
            if (ent_slot_[e] < 0) {
                entry.width = 0;
                entry.sum_kq = entry.sum_kqsq = fp_pop_[ent_fp_[e]];
                return;
            }
            const uint32_t* L = ls(ent_slot_[e]);
            entry.width = static_cast<uint8_t>(width_for(entry.n_samples));
            entry.sum.resize(static_cast<size_t>(n_bits_) * entry.width);
            uint64_t sum_kq = 0, sum_kqsq = 0;
            with_sum(entry.width, entry.sum.data(), [&](auto* S) {
                using V = std::remove_reference_t<decltype(*S)>;
                for (int p = 0; p < n_bits_; ++p) {
                    const uint64_t v = L[p];
                    S[p] = static_cast<V>(v);
                    sum_kq += v;
                    sum_kqsq += v * v;
                }
            });
            entry.sum_kq = sum_kq;
            entry.sum_kqsq = sum_kqsq;
        });
    // Molecules: member i is row i; the members of a subcluster are chained in
    // row order
    t.mem_id_.resize(n_);
    t.mem_next_.resize(n_);
    pool_.parallel_for(n_, 4096, [&](int64_t i, int) {
        t.mem_id_[i] = ids ? ids[i] : start + i;
    });
    std::vector<int64_t> head(ents.size(), -1), tail(ents.size(), -1);
    for (int i = n_ - 1; i >= 0; --i) {
        const int32_t e = ent_map[labels_[i]];
        t.mem_next_[i] = head[e];
        if (head[e] < 0) {
            tail[e] = i;
        }
        head[e] = i;
    }
    for (size_t k = 0; k < ents.size(); ++k) {
        t.ents_[k].head = head[k];
        t.ents_[k].tail = tail[k];
    }
    for (size_t i = 0; i < leaves.size(); ++i) {
        t.nodes_[leaves[i]].prev = i == 0 ? -1 : leaves[i - 1];
        t.nodes_[leaves[i]].next = i + 1 == leaves.size() ? -1 : leaves[i + 1];
    }
    t.first_leaf_ = leaves.empty() ? -1 : leaves[0];
    t.root_ = node_map[root_];
}

}  // namespace bblean
