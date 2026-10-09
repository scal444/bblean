// BitBirch tree, the C++ version of the python tree of bblean.BitBirch (which
// uses it when the extension is available, for the builtin merge criteria). It
// builds exactly the same tree.
//
// - Insertion: the closest subcluster of a node is the first maximum of the
//   Tanimoto similarity to the centroids of its subclusters. At a leaf the
//   nominee is merged with the closest subcluster if the merge criterion accepts
//   it, otherwise it is appended, and the node is split if it has more than
//   branching_factor subclusters. Subclusters on the path ("tracking"
//   subclusters) are updated after the insertion below them.
// - Splits: the most dissimilar pair of centroids of the node (kernels.hpp)
//   seeds the two new nodes, every subcluster goes to the node of the seed it
//   is more similar to (ties to the second one), the first seed to the first.
// - Centroids are majority votes of the linear sums (kernels.hpp).
// - Merge criteria: the builtin criteria of bblean.merges (kernels.hpp).
//
// Storage: each node keeps the centroids of its subclusters contiguously (in
// order) and their popcounts. Each subcluster ("entry") keeps n_samples, the
// sums T = sum(L) and Q = sum(L^2) of its linear sum L, and L itself, stored
// with the smallest unsigned type that holds n_samples. Entries formed by a
// single fingerprint (possibly weighted) store no linear sum: it is
// n_samples * fingerprint. Molecule indices are kept as linked lists (only for
// leaf entries), so merging clusters is O(1).
//
// Fingerprints are rows of 64 bit words (bit p is feature p ^ 7 of bblean's
// packed representation, see kernels.hpp). Linear sums are kept in this bit
// order, and converted to feature order on import / export.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels.hpp"

namespace py = pybind11;

namespace bblean {

using i64 = int64_t;

static inline int widthFor(u64 n) {
    if (n <= 0xFFull) return 1;
    if (n <= 0xFFFFull) return 2;
    if (n <= 0xFFFFFFFFull) return 4;
    return 8;
}

template <typename F>
static inline decltype(auto) withSum(uint8_t width, uint8_t* data, F&& f) {
    switch (width) {
        case 1:
            return f(data);
        case 2:
            return f(reinterpret_cast<uint16_t*>(data));
        case 4:
            return f(reinterpret_cast<uint32_t*>(data));
        default:
            return f(reinterpret_cast<u64*>(data));
    }
}

// count values as a numpy array of the smallest uint dtype that holds n
static py::array uintArray(const u64* values, size_t count, u64 n) {
    auto make = [&](auto tag) -> py::array {
        using V = decltype(tag);
        py::array_t<V> a(static_cast<py::ssize_t>(count));
        auto* p = a.mutable_data();
        for (size_t j = 0; j < count; ++j) p[j] = static_cast<V>(values[j]);
        return std::move(a);
    };
    switch (widthFor(n)) {
        case 1:
            return make(uint8_t{});
        case 2:
            return make(uint16_t{});
        case 4:
            return make(uint32_t{});
        default:
            return make(u64{});
    }
}

struct Entry {
    u64 n = 0;  // n_samples
    u64 T = 0;  // sum of the linear sum
    u64 Q = 0;  // sum of squares of the linear sum
    int32_t child = -1;
    uint8_t width = 0;  // bytes per linear sum element; 0: implicit (n * centroid)
    std::vector<uint8_t> sum;
    i64 head = -1, tail = -1;  // molecule index chain (leaf entries only)
};

struct Node {
    int bf = 0;
    bool leaf = false;
    int32_t prev = -1, next = -1;  // leaf chain, prev == -1 means "first leaf"
    std::vector<int32_t> ents;
    std::vector<u64> cent;      // (bf + 1) * w64
    std::vector<uint32_t> pop;  // (bf + 1)
};

// A subcluster about to be inserted: either one (weighted) fingerprint, whose
// linear sum is n * fp, or a dense linear sum (uint64, internal bit order).
struct Nominee {
    const u64* cent = nullptr;  // centroid (the fingerprint itself if single)
    uint32_t pop = 0;
    u64 n = 0, T = 0, Q = 0;
    bool dense = false;
    const u64* dsum = nullptr;
    i64 head = -1, tail = -1;
};

class Tree {
   public:
    explicit Tree(int nFeatures) {
        if (nFeatures <= 0 || nFeatures % 8 != 0)
            throw std::invalid_argument("n_features must be a positive multiple of 8");
        nFeatures_ = nFeatures;
        nBytes_ = nFeatures / 8;
        w64_ = (nBytes_ + 7) / 8;
        nb_ = w64_ * 64;
        tmpA_.assign(nb_, 0);
        tmpB_.assign(nb_, 0);
        tmpCent_.assign(w64_, 0);
        rowBuf_.assign(w64_, 0);
        tmpOut_.assign(nb_, 0);
    }

    int nFeatures() const { return nFeatures_; }
    int nBytes() const { return nBytes_; }
    bool initialized() const { return root_ >= 0 || firstLeaf_ >= 0; }
    bool onlyLeaves() const { return root_ < 0 && firstLeaf_ >= 0; }

    // ------------------------------------------------------------- fitting

    // Fit packed fingerprints (rows of n_features / 8 bytes). Molecule indices
    // are ids[i] if given, else start + i; weights (optional) per row.
    void fitPacked(const uint8_t* X, i64 rows, const i64* ids, i64 start, const u64* weights,
                   const MergeCriterion& merge, int bf) {
        prepare(merge, bf);
        for (i64 i = 0; i < rows; ++i) {
            u64* fp = rowBuf_.data();
            fp[w64_ - 1] = 0;
            std::memcpy(fp, X + i * nBytes_, nBytes_);
            const u64 w = weights ? weights[i] : 1;
            Nominee s;
            s.cent = fp;
            s.pop = popcount(fp, w64_);
            s.n = w;
            s.T = w * s.pop;
            s.Q = w * w * s.pop;
            s.head = s.tail = newMember(ids ? ids[i] : start + i);
            insertRoot(s, bf);
        }
    }

    // Fit buffers: rows of n_features + 1 unsigned values (linear sum in
    // feature order, then n_samples). Molecule indices: idx[offs[i]:offs[i+1]]
    // (offs may be null: no indices).
    template <typename T>
    void fitBuffers(const T* X, i64 rows, const i64* idx, const i64* offs, const MergeCriterion& merge, int bf) {
        prepare(merge, bf);
        std::vector<u64> dsum(nb_, 0);
        std::vector<u64> cent(w64_, 0);
        for (i64 i = 0; i < rows; ++i) {
            const T* buf = X + i * (nFeatures_ + 1);
            for (int p = 0; p < nFeatures_; ++p) dsum[p] = static_cast<u64>(buf[p ^ 7]);
            Nominee s;
            s.n = static_cast<u64>(buf[nFeatures_]);
            summarize(dsum.data(), s.n, cent.data(), s.pop, s.T, s.Q);
            s.cent = cent.data();
            s.dense = true;
            s.dsum = dsum.data();
            if (offs != nullptr) {
                for (i64 k = offs[i]; k < offs[i + 1]; ++k) {
                    const i64 m = newMember(idx[k]);
                    if (s.head < 0) {
                        s.head = m;
                    } else {
                        memNext_[s.tail] = m;
                    }
                    s.tail = m;
                }
            }
            insertRoot(s, bf);
        }
    }

    // Release internal nodes (BitBirch.delete_internal_nodes)
    void releaseInternal() {
        if (root_ < 0 || nodes_[root_].leaf) return;
        std::vector<char> isLeaf(nodes_.size(), 0);
        for (int32_t l = firstLeaf_; l >= 0; l = nodes_[l].next) isLeaf[l] = 1;
        for (size_t nd = 0; nd < nodes_.size(); ++nd) {
            if (isLeaf[nd]) continue;
            for (const int32_t e : nodes_[nd].ents) freeEntry(e);
            Node empty;
            std::swap(nodes_[nd], empty);
        }
        root_ = -1;
    }

    // -------------------------------------------------------------- export

    // Leaf entries in leaf order, or sorted by n_samples (descending, stable)
    std::vector<int32_t> leafEntries(bool sort) const {
        std::vector<int32_t> out;
        for (int32_t l = firstLeaf_; l >= 0; l = nodes_[l].next)
            out.insert(out.end(), nodes_[l].ents.begin(), nodes_[l].ents.end());
        if (sort)
            std::stable_sort(out.begin(), out.end(), [this](int32_t a, int32_t b) { return ents_[a].n > ents_[b].n; });
        return out;
    }

    std::vector<const u64*> leafCentroids(const std::vector<int32_t>& order) const {
        std::vector<const u64*> where(ents_.size(), nullptr);
        for (int32_t l = firstLeaf_; l >= 0; l = nodes_[l].next)
            for (size_t i = 0; i < nodes_[l].ents.size(); ++i)
                where[nodes_[l].ents[i]] = nodes_[l].cent.data() + i * w64_;
        std::vector<const u64*> out(order.size());
        for (size_t i = 0; i < order.size(); ++i) out[i] = where[order[i]];
        return out;
    }

    u64 entryN(int32_t e) const { return ents_[e].n; }
    i64 memberHead(int32_t e) const { return ents_[e].head; }
    i64 memberNext(i64 m) const { return memNext_[m]; }
    i64 memberId(i64 m) const { return memId_[m]; }

    // Linear sum of entry e (centroid cent) in feature order
    void linearSum(int32_t e, const u64* cent, u64* out) const { linearSumOf(ents_[e], cent, out); }

    // Molecule indices of a member chain
    py::list memberList(i64 head) const {
        py::list out;
        for (i64 m = head; m >= 0; m = memNext_[m]) out.append(memId_[m]);
        return out;
    }

    // ----------------------------------------------------------- pickling

    std::string serialize() const;
    static Tree deserialize(const std::string& s);

   private:
    void prepare(const MergeCriterion& merge, int bf) {
        if (onlyLeaves()) throw std::runtime_error("Internal nodes were released, call reset() before fit()");
        if (bf < 1) throw std::invalid_argument("branching_factor must be >= 1");
        merge_ = &merge;
        if (root_ < 0) {
            root_ = newNode(bf, true);
            firstLeaf_ = root_;
        }
    }

    i64 newMember(i64 id) {
        memId_.push_back(id);
        memNext_.push_back(-1);
        return static_cast<i64>(memId_.size()) - 1;
    }

    int32_t newNode(int bf, bool leaf) {
        Node nd;
        nd.bf = bf;
        nd.leaf = leaf;
        nd.ents.reserve(bf + 1);
        nd.cent.assign(static_cast<size_t>(bf + 1) * w64_, 0);
        nd.pop.assign(bf + 1, 0);
        nodes_.push_back(std::move(nd));
        return static_cast<int32_t>(nodes_.size()) - 1;
    }

    int32_t newEntry() {
        if (!freeEnts_.empty()) {
            const int32_t e = freeEnts_.back();
            freeEnts_.pop_back();
            ents_[e] = Entry();
            return e;
        }
        ents_.emplace_back();
        return static_cast<int32_t>(ents_.size()) - 1;
    }

    void freeEntry(int32_t e) {
        ents_[e] = Entry();
        freeEnts_.push_back(e);
    }

    u64* row(int32_t nid, int idx) { return nodes_[nid].cent.data() + static_cast<size_t>(idx) * w64_; }

    void setRow(int32_t nid, int idx, const u64* c, uint32_t pop) {
        std::memcpy(row(nid, idx), c, sizeof(u64) * w64_);
        nodes_[nid].pop[idx] = pop;
    }

    void appendEntry(int32_t nid, int32_t e, const u64* c, uint32_t pop) {
        Node& nd = nodes_[nid];
        const int idx = static_cast<int>(nd.ents.size());
        if (idx > nd.bf) throw std::runtime_error("node overflow");
        nd.ents.push_back(e);
        setRow(nid, idx, c, pop);
    }

    // f(getL) with getL(p) = L_p, specialized for the storage of E
    template <typename F>
    decltype(auto) withL(const Entry& E, const u64* cent, F&& f) const {
        if (E.width == 0) {
            const u64 n = E.n;
            return f([cent, n](int p) -> u64 { return ((cent[p >> 6] >> (p & 63)) & 1) ? n : 0; });
        }
        return withSum(E.width, const_cast<uint8_t*>(E.sum.data()),
                       [&](auto* L) { return f([L](int p) -> u64 { return static_cast<u64>(L[p]); }); });
    }

    void linearSumOf(const Entry& E, const u64* cent, u64* out) const {
        denseSum(E, cent, tmpOut_.data());
        for (int j = 0; j < nFeatures_; ++j) out[j] = tmpOut_[j ^ 7];
    }

    // Dense linear sum into out (uint64, internal order)
    void denseSum(const Entry& E, const u64* cent, u64* out) const {
        withL(E, cent, [&](auto getL) {
            for (int p = 0; p < nb_; ++p) out[p] = getL(p);
        });
    }

    // Centroid (with its popcount), T and Q of a dense linear sum of n samples
    void summarize(const u64* L, u64 n, u64* cent, uint32_t& pop, u64& T, u64& Q) const {
        std::fill(cent, cent + w64_, 0);
        column_sums(L, nb_, n, T, Q, [cent](size_t p, u64) { cent[p >> 6] |= u64{1} << (p & 63); });
        pop = popcount(cent, w64_);
    }

    // Store a dense linear sum with the width for n_samples n
    void storeSum(Entry& E, const u64* dense, u64 n) {
        E.width = static_cast<uint8_t>(widthFor(n));
        E.sum.assign(static_cast<size_t>(nb_) * E.width, 0);
        withSum(E.width, E.sum.data(), [&](auto* L) {
            using V = std::remove_reference_t<decltype(*L)>;
            for (int p = 0; p < nb_; ++p) L[p] = static_cast<V>(dense[p]);
        });
    }

    // Make sure the entry has an explicit linear sum of width for newN
    void ensureWidth(Entry& E, const u64* cent, u64 newN) {
        if (E.width != 0 && E.width >= widthFor(newN)) return;
        u64* tmp = tmpB_.data();
        denseSum(E, cent, tmp);
        storeSum(E, tmp, newN);
    }

    // Index of the closest entry (first maximum of the Tanimoto similarity)
    int closest(int32_t nid, const Nominee& s) {
        const Node& nd = nodes_[nid];
        const size_t m = nd.ents.size();
        sims1_.resize(m);
        jt_sims(nd.cent.data(), nd.pop.data(), m, w64_, s.cent, sims1_.data());
        return static_cast<int>(first_max(sims1_.data(), m));
    }

    // New centroid of entry E (centroid oc) after adding s; also returns
    // S = sum_{j in new centroid} L'_j (L' = merged linear sum), T' and Q'.
    void mergedStats(const Entry& E, const u64* oc, const Nominee& s, u64* nc, uint32_t& npop, u64& S, u64& T1,
                     u64& Q1) {
        const u64 n1 = E.n + s.n;
        u64 cross = 0;
        S = 0;
        if (!s.dense) {
            // L'_j = L_j + n_s * f_j; only bits in oc | fp can be set in the new centroid
            withL(E, oc, [&](auto getL) {
                const u64 sn = s.n;
                for (int k = 0; k < w64_; ++k) {
                    const u64 f = s.cent[k];
                    u64 out = 0;
                    for (u64 w = oc[k] | f; w; w &= w - 1) {
                        const int b = ctz64(w);
                        const u64 L0 = getL(k * 64 + b);
                        const u64 inF = (f >> b) & 1;
                        cross += inF * L0;
                        const u64 L1 = L0 + inF * sn;
                        if (centroid_bit(L1, n1)) {
                            out |= u64{1} << b;
                            S += L1;
                        }
                    }
                    nc[k] = out;
                }
            });
            cross *= s.n;
        } else {
            u64* L = tmpA_.data();
            denseSum(E, oc, L);
            std::fill(nc, nc + w64_, 0);
            for (int p = 0; p < nb_; ++p) {
                cross += L[p] * s.dsum[p];
                const u64 L1 = L[p] + s.dsum[p];
                if (centroid_bit(L1, n1)) {
                    nc[p >> 6] |= u64{1} << (p & 63);
                    S += L1;
                }
            }
        }
        npop = popcount(nc, w64_);
        T1 = E.T + s.T;
        Q1 = E.Q + s.Q + 2 * cross;
    }

    bool accept(const Entry& E, const u64* oc, uint32_t ocPop, const Nominee& s, u64 T1, u64 Q1, uint32_t npop,
                u64 S) const {
        const MergeSums sums{E.n, s.n, E.n + s.n, E.T, E.Q, T1, Q1};
        return accept_merge(
            *merge_, sums, [&] { return radius_compl(T1, Q1, E.n + s.n, npop, S); },
            [&] {
                u64 S0 = 0;
                withL(E, oc, [&](auto getL) { for_each_bit(oc, w64_, [&](int p) { S0 += getL(p); }); });
                return radius_compl(E.T, E.Q, E.n, ocPop, S0);
            });
    }

    // Add s to entry e at (nid, idx), with precomputed stats
    void absorb(int32_t nid, int idx, const Nominee& s, const u64* nc, uint32_t npop, u64 T1, u64 Q1, bool members) {
        Entry& E = ents_[nodes_[nid].ents[idx]];
        const u64 n1 = E.n + s.n;
        ensureWidth(E, row(nid, idx), n1);
        withSum(E.width, E.sum.data(), [&](auto* L) {
            using V = std::remove_reference_t<decltype(*L)>;
            if (!s.dense) {
                const V add = static_cast<V>(s.n);
                for_each_bit(s.cent, w64_, [&](int p) { L[p] = static_cast<V>(L[p] + add); });
            } else {
                for (int p = 0; p < nb_; ++p) L[p] = static_cast<V>(L[p] + s.dsum[p]);
            }
        });
        E.n = n1;
        E.T = T1;
        E.Q = Q1;
        if (members && s.head >= 0) {
            if (E.head < 0) {
                E.head = s.head;
            } else {
                memNext_[E.tail] = s.head;
            }
            E.tail = s.tail;
        }
        setRow(nid, idx, nc, npop);
    }

    void appendNominee(int32_t nid, const Nominee& s) {
        const int32_t e = newEntry();
        Entry& E = ents_[e];
        E.n = s.n;
        E.T = s.T;
        E.Q = s.Q;
        E.head = s.head;
        E.tail = s.tail;
        if (s.dense) storeSum(E, s.dsum, E.n);
        appendEntry(nid, e, s.cent, s.pop);
    }

    struct SplitOut {
        int32_t e1, e2;
        std::vector<u64> c1, c2;
        uint32_t p1, p2;
    };

    SplitOut split(int32_t nid) {
        const int m = static_cast<int>(nodes_[nid].ents.size());
        const bool leaf = nodes_[nid].leaf;
        const int bf = nodes_[nid].bf;
        const std::vector<int32_t> ents = nodes_[nid].ents;
        const std::vector<u64> rows(nodes_[nid].cent.begin(),
                                    nodes_[nid].cent.begin() + static_cast<size_t>(m) * w64_);
        const std::vector<uint32_t> pops(nodes_[nid].pop.begin(), nodes_[nid].pop.begin() + m);
        auto R = [&](int i) { return rows.data() + static_cast<size_t>(i) * w64_; };

        sims1_.resize(m);
        sims2_.resize(m);
        const size_t fp1 = most_dissimilar(rows.data(), pops.data(), m, w64_, nb_, sims1_.data(), sims2_.data()).fp1;

        // node1 is new, node2 is the old node (reused)
        const int32_t nid1 = newNode(bf, leaf);
        if (leaf) {
            Node& n2 = nodes_[nid];
            Node& n1 = nodes_[nid1];
            n1.prev = n2.prev;
            if (n2.prev >= 0) {
                nodes_[n2.prev].next = nid1;
            } else {
                firstLeaf_ = nid1;
            }
            n1.next = nid;
            n2.prev = nid1;
        }
        nodes_[nid].ents.clear();
        std::vector<u64> acc1(nb_, 0), acc2(nb_, 0);
        u64 n1 = 0, n2 = 0;
        u64* tmp = tmpA_.data();
        for (int i = 0; i < m; ++i) {
            const bool closer1 = static_cast<size_t>(i) == fp1 || sims1_[i] > sims2_[i];
            const Entry& E = ents_[ents[i]];
            std::vector<u64>& acc = closer1 ? acc1 : acc2;
            denseSum(E, R(i), tmp);
            for (int p = 0; p < nb_; ++p) acc[p] += tmp[p];
            if (closer1) {
                n1 += E.n;
                appendEntry(nid1, ents[i], R(i), pops[i]);
            } else {
                n2 += E.n;
                appendEntry(nid, ents[i], R(i), pops[i]);
            }
        }
        SplitOut out;
        out.e1 = makeSummary(nid1, n1, acc1, out.c1, out.p1);
        out.e2 = makeSummary(nid, n2, acc2, out.c2, out.p2);
        return out;
    }

    int32_t makeSummary(int32_t child, u64 n, const std::vector<u64>& acc, std::vector<u64>& c, uint32_t& pop) {
        const int32_t e = newEntry();
        Entry& E = ents_[e];
        E.child = child;
        E.n = n;
        c.resize(w64_);
        summarize(acc.data(), n, c.data(), pop, E.T, E.Q);
        storeSum(E, acc.data(), n);
        return e;
    }

    // Insert s below node nid; returns whether node nid must be split
    bool insert(int32_t nid, const Nominee& s) {
        if (nodes_[nid].ents.empty()) {
            appendNominee(nid, s);
            return false;
        }
        const int idx = closest(nid, s);
        const int32_t e = nodes_[nid].ents[idx];
        u64* nc = tmpCent_.data();
        uint32_t npop;
        u64 S, T1, Q1;
        if (ents_[e].child < 0) {
            const u64* oc = row(nid, idx);
            mergedStats(ents_[e], oc, s, nc, npop, S, T1, Q1);
            if (accept(ents_[e], oc, nodes_[nid].pop[idx], s, T1, Q1, npop, S)) {
                absorb(nid, idx, s, nc, npop, T1, Q1, true);
                return false;
            }
            appendNominee(nid, s);
            return static_cast<int>(nodes_[nid].ents.size()) > nodes_[nid].bf;
        }
        const int32_t child = ents_[e].child;
        if (insert(child, s)) {
            SplitOut o = split(child);
            nodes_[nid].ents[idx] = o.e1;
            setRow(nid, idx, o.c1.data(), o.p1);
            freeEntry(e);
            appendEntry(nid, o.e2, o.c2.data(), o.p2);
            return static_cast<int>(nodes_[nid].ents.size()) > nodes_[nid].bf;
        }
        // Update the tracking subcluster
        mergedStats(ents_[e], row(nid, idx), s, nc, npop, S, T1, Q1);
        absorb(nid, idx, s, nc, npop, T1, Q1, false);
        return false;
    }

    void insertRoot(const Nominee& s, int bf) {
        if (insert(root_, s)) {
            SplitOut o = split(root_);
            const int32_t r = newNode(bf, false);
            appendEntry(r, o.e1, o.c1.data(), o.p1);
            appendEntry(r, o.e2, o.c2.data(), o.p2);
            root_ = r;
        }
    }

    int nFeatures_ = 0, nBytes_ = 0, w64_ = 0, nb_ = 0;
    const MergeCriterion* merge_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<Entry> ents_;
    std::vector<int32_t> freeEnts_;
    std::vector<i64> memId_, memNext_;
    int32_t root_ = -1, firstLeaf_ = -1;
    // scratch
    std::vector<u64> tmpA_, tmpB_, tmpCent_, rowBuf_;
    mutable std::vector<u64> tmpOut_;
    std::vector<double> sims1_, sims2_;
};

// ---------------------------------------------------------------- pickling

namespace {
struct Writer {
    std::string out;
    template <typename T>
    void put(const T& v) {
        out.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <typename T>
    void vec(const std::vector<T>& v) {
        put<u64>(v.size());
        out.append(reinterpret_cast<const char*>(v.data()), sizeof(T) * v.size());
    }
};
struct Reader {
    const std::string& s;
    size_t pos = 0;
    template <typename T>
    T get() {
        if (pos + sizeof(T) > s.size()) throw std::runtime_error("corrupt BitBirch tree state");
        T v;
        std::memcpy(&v, s.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    template <typename T>
    void vec(std::vector<T>& v) {
        const u64 n = get<u64>();
        if (n > (s.size() - pos) / sizeof(T)) throw std::runtime_error("corrupt BitBirch tree state");
        v.resize(n);
        std::memcpy(v.data(), s.data() + pos, sizeof(T) * n);
        pos += sizeof(T) * n;
    }
};
constexpr u64 kMagic = 0x3165657274626262ull;  // "bbbtree1"
}  // namespace

std::string Tree::serialize() const {
    Writer w;
    w.put(kMagic);
    w.put<int32_t>(nFeatures_);
    w.put<int32_t>(root_);
    w.put<int32_t>(firstLeaf_);
    w.vec(memId_);
    w.vec(memNext_);
    w.vec(freeEnts_);
    w.put<u64>(nodes_.size());
    for (const Node& nd : nodes_) {
        w.put<int32_t>(nd.bf);
        w.put<uint8_t>(nd.leaf);
        w.put<int32_t>(nd.prev);
        w.put<int32_t>(nd.next);
        w.vec(nd.ents);
        w.vec(nd.cent);
        w.vec(nd.pop);
    }
    w.put<u64>(ents_.size());
    for (const Entry& E : ents_) {
        w.put(E.n);
        w.put(E.T);
        w.put(E.Q);
        w.put(E.child);
        w.put(E.width);
        w.put(E.head);
        w.put(E.tail);
        w.vec(E.sum);
    }
    return std::move(w.out);
}

Tree Tree::deserialize(const std::string& s) {
    Reader r{s};
    if (r.get<u64>() != kMagic) throw std::runtime_error("not a BitBirch tree state");
    Tree t(r.get<int32_t>());
    t.root_ = r.get<int32_t>();
    t.firstLeaf_ = r.get<int32_t>();
    r.vec(t.memId_);
    r.vec(t.memNext_);
    r.vec(t.freeEnts_);
    t.nodes_.resize(r.get<u64>());
    for (Node& nd : t.nodes_) {
        nd.bf = r.get<int32_t>();
        nd.leaf = r.get<uint8_t>() != 0;
        nd.prev = r.get<int32_t>();
        nd.next = r.get<int32_t>();
        r.vec(nd.ents);
        r.vec(nd.cent);
        r.vec(nd.pop);
    }
    t.ents_.resize(r.get<u64>());
    for (Entry& E : t.ents_) {
        E.n = r.get<u64>();
        E.T = r.get<u64>();
        E.Q = r.get<u64>();
        E.child = r.get<int32_t>();
        E.width = r.get<uint8_t>();
        E.head = r.get<i64>();
        E.tail = r.get<i64>();
        r.vec(E.sum);
    }
    return t;
}

}  // namespace bblean

// ----------------------------------------------------------------- bindings

namespace {

using bblean::i64;
using bblean::MergeCriterion;
using bblean::Tree;
using bblean::u64;

using I64Array = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using U64Array = py::array_t<u64, py::array::c_style | py::array::forcecast>;

// merge: (kind, threshold, tolerance, decay, offset) of a builtin criterion
MergeCriterion toCriterion(const py::tuple& t) {
    if (t.size() != 5) throw std::invalid_argument("invalid merge parameters");
    MergeCriterion c;
    c.kind = t[0].cast<int>();
    if (c.kind < bblean::kDiameter || c.kind > bblean::kNever) throw std::invalid_argument("unknown merge kind");
    c.threshold = t[1].cast<double>();
    c.tolerance = t[2].cast<double>();
    c.decay = t[3].cast<double>();
    c.offset = t[4].cast<double>();
    return c;
}

void fitPacked(Tree& tree, const py::array_t<uint8_t, py::array::c_style>& X, const py::object& ids, i64 start,
              const py::object& weights, const py::tuple& merge, int bf) {
    if (X.ndim() != 2) throw std::invalid_argument("fingerprints must be a 2D array");
    if (X.shape(1) != tree.nBytes())
        throw std::invalid_argument("fingerprints have " + std::to_string(X.shape(1)) + " bytes, expected " +
                                    std::to_string(tree.nBytes()));
    const i64 rows = X.shape(0);
    I64Array idArr;
    U64Array wArr;
    const i64* idPtr = nullptr;
    const u64* wPtr = nullptr;
    if (!ids.is_none()) {
        idArr = ids.cast<I64Array>();
        if (idArr.ndim() != 1 || idArr.shape(0) < rows) throw std::invalid_argument("not enough indices");
        idPtr = idArr.data();
    }
    if (!weights.is_none()) {
        wArr = weights.cast<U64Array>();
        if (wArr.ndim() != 1 || wArr.shape(0) < rows) throw std::invalid_argument("not enough weights");
        wPtr = wArr.data();
    }
    const MergeCriterion m = toCriterion(merge);
    py::gil_scoped_release release;
    tree.fitPacked(X.data(), rows, idPtr, start, wPtr, m, bf);
}

void fitBuffers(Tree& tree, const py::array& X, const py::object& idx, const py::object& offs, const py::tuple& merge,
               int bf) {
    if (X.ndim() != 2 || X.shape(1) != tree.nFeatures() + 1)
        throw std::invalid_argument("buffers must have shape (N, n_features + 1)");
    const auto kind = X.dtype().kind();
    if (kind != 'u' && kind != 'i') throw std::invalid_argument("buffers must have an integer dtype");
    const i64 rows = X.shape(0);
    I64Array idArr, offArr;
    const i64* idPtr = nullptr;
    const i64* offPtr = nullptr;
    if (!offs.is_none()) {
        idArr = idx.cast<I64Array>();
        offArr = offs.cast<I64Array>();
        if (offArr.ndim() != 1 || offArr.shape(0) < rows + 1) throw std::invalid_argument("not enough offsets");
        idPtr = idArr.data();
        offPtr = offArr.data();
        if (offPtr[0] < 0 || offPtr[rows] > idArr.size()) throw std::invalid_argument("offsets out of bounds");
        for (i64 i = 0; i < rows; ++i)
            if (offPtr[i] > offPtr[i + 1]) throw std::invalid_argument("offsets must be non-decreasing");
    }
    const MergeCriterion m = toCriterion(merge);
    auto contiguous = py::array::ensure(X, py::array::c_style);
    const void* data = contiguous.data();
    const auto size = X.dtype().itemsize();
    if (size != 1 && size != 2 && size != 4 && size != 8) throw std::invalid_argument("unsupported buffer dtype");
    py::gil_scoped_release release;
    switch (size) {
        case 1:
            return tree.fitBuffers(static_cast<const uint8_t*>(data), rows, idPtr, offPtr, m, bf);
        case 2:
            return tree.fitBuffers(static_cast<const uint16_t*>(data), rows, idPtr, offPtr, m, bf);
        case 4:
            return tree.fitBuffers(static_cast<const uint32_t*>(data), rows, idPtr, offPtr, m, bf);
        default:
            return tree.fitBuffers(static_cast<const u64*>(data), rows, idPtr, offPtr, m, bf);
    }
}

py::list clusterMolIds(const Tree& tree, bool sort) {
    const auto order = tree.leafEntries(sort);
    py::list out(order.size());
    for (size_t i = 0; i < order.size(); ++i) out[i] = tree.memberList(tree.memberHead(order[i]));
    return out;
}

py::array_t<u64> leafSizes(const Tree& tree, bool sort) {
    const auto order = tree.leafEntries(sort);
    py::array_t<u64> out(static_cast<py::ssize_t>(order.size()));
    auto* p = out.mutable_data();
    for (size_t i = 0; i < order.size(); ++i) p[i] = tree.entryN(order[i]);
    return out;
}

py::array_t<u64> assignments(const Tree& tree, i64 nMols, bool sort) {
    const auto order = tree.leafEntries(sort);
    py::array_t<u64> out(static_cast<py::ssize_t>(nMols));
    auto* p = out.mutable_data();
    std::fill(p, p + nMols, 0);
    for (size_t i = 0; i < order.size(); ++i)
        for (i64 m = tree.memberHead(order[i]); m >= 0; m = tree.memberNext(m)) {
            i64 id = tree.memberId(m);
            if (id < 0) id += nMols;  // python negative indexing
            if (id < 0 || id >= nMols) throw py::index_error("molecule index out of bounds for assignments");
            p[id] = static_cast<u64>(i + 1);
        }
    return out;
}

py::array_t<uint8_t> centroids(const Tree& tree, bool sort) {
    const auto order = tree.leafEntries(sort);
    const auto where = tree.leafCentroids(order);
    const int nBytes = tree.nBytes();
    py::array_t<uint8_t> out({static_cast<py::ssize_t>(order.size()), static_cast<py::ssize_t>(nBytes)});
    auto* p = out.mutable_data();
    for (size_t i = 0; i < order.size(); ++i) std::memcpy(p + i * nBytes, where[i], nBytes);
    return out;
}

// Linear sum + n_samples buffers of the leaf subclusters, with the smallest
// uint dtype that holds n_samples
py::list leafBuffers(const Tree& tree, bool sort) {
    const auto order = tree.leafEntries(sort);
    const auto where = tree.leafCentroids(order);
    const int F = tree.nFeatures();
    std::vector<u64> buf(F + 1);
    py::list out(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        tree.linearSum(order[i], where[i], buf.data());
        buf[F] = tree.entryN(order[i]);
        out[i] = bblean::uintArray(buf.data(), buf.size(), buf[F]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_cpp_bitbirch, m) {
    m.doc() = "BitBirch tree (C++ extension)";
    m.attr("MERGE_KINDS") = py::dict(
        py::arg("diameter") = int(bblean::kDiameter), py::arg("radius") = int(bblean::kRadius),
        py::arg("tolerance-diameter") = int(bblean::kToleranceDiameter),
        py::arg("flexible-tolerance-diameter") = int(bblean::kFlexibleToleranceDiameter),
        py::arg("tolerance-radius") = int(bblean::kToleranceRadius),
        py::arg("tolerance-legacy") = int(bblean::kToleranceLegacy), py::arg("never") = int(bblean::kNever));
    py::class_<Tree>(m, "Tree")
        .def(py::init<int>(), py::arg("n_features"))
        .def_property_readonly("n_features", &Tree::nFeatures)
        .def_property_readonly("initialized", &Tree::initialized)
        .def_property_readonly("only_leaves", &Tree::onlyLeaves)
        .def("fit_packed", &fitPacked, py::arg("X"), py::arg("ids"), py::arg("start"), py::arg("weights"),
             py::arg("merge"), py::arg("branching_factor"))
        .def("fit_buffers", &fitBuffers, py::arg("X"), py::arg("idx"), py::arg("offsets"), py::arg("merge"),
             py::arg("branching_factor"))
        .def("release_internal", &Tree::releaseInternal)
        .def("cluster_mol_ids", &clusterMolIds, py::arg("sort") = true)
        .def("leaf_sizes", &leafSizes, py::arg("sort") = true)
        .def("assignments", &assignments, py::arg("n_mols"), py::arg("sort") = true)
        .def("centroids", &centroids, py::arg("sort") = true)
        .def("leaf_buffers", &leafBuffers, py::arg("sort") = true)
        .def(py::pickle([](const Tree& t) { return py::bytes(t.serialize()); },
                        [](const py::bytes& b) { return Tree::deserialize(std::string(b)); }));
}
