// The elimination of one tree level's fronts, one thread block per (configuration, front).
//
// A front is a dense skew matrix of n <= MAXN slots; its strict upper triangle lives
// packed in shared memory for the whole elimination, so a step never touches global
// memory except to record its multipliers. Step t: a window of WINDOW candidate columns
// k (alive, fully summed, not failed since the last pivot; the lowest slots first) is
// scanned, a warp a column, for each column's largest alive entry m_k and its largest
// entry b_k in a candidate row r_k; the pivot pair is the window's k with the largest
// b_k / m_k among those >= u (at the root, where every row is a candidate, the largest
// b_k). A window with none passing marks its columns failed and the next is tried; a
// pivot clears the marks. Then
//   F <- F - c tau^T + tau c^T,   c = F[rest, r],  tau = F[k, rest] / a,  a = F[k, r].
// No pair passing ends the front: what is left alive is the parent's.
//
// The backward pass needs only the recorded (k, r, a, c, tau): with W the adjoint of
// the upper-triangle parameters held as a skew matrix, a step's update contributes
// cbar = -W tau and taubar = W c, which flow into rows and columns k and r.
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include "xla/ffi/api/ffi.h"

// The scalar of every front, contribution block, multiplier and record: double, or float when the library is
// built with -DPIVOTED_REAL=float (front_cuda.py builds one library per type and picks by the inputs' dtype).
// The pivots' log and sign accumulate in double in either build; the threshold u arrives as a double attribute
// and is cast. The block kernels' fp64 tensor-core close (wmma m8n8k4) is in the double build only; the U x U
// products (uu_tc_kernel) run on the fp64 tensor cores in either build, and so does the float build's global close
// when built with PIVOTED_GCLOSE_DMMA.
#ifndef PIVOTED_REAL
#define PIVOTED_REAL double
#endif
typedef PIVOTED_REAL real_t;
#if defined(PIVOTED_REAL_FLOAT)
#define FFI_REAL ffi::F32
#define RSUB_RN __fsub_rn
#define RADD_RN __fadd_rn
#else
#define FFI_REAL ffi::F64
#define RSUB_RN __dsub_rn
#define RADD_RN __dadd_rn
#endif

namespace ffi = xla::ffi;

// A block is T threads by the front's width. Large fronts retain their original 16-column logical window
// when the forward CTA is reduced from 512 to 256 threads; each physical warp then scans rows in waves.
constexpr int MAXN = 336;   // an upper bound; the device's shared memory decides (g_smem_limit); 336 holds a 320-wide level with its U x U block out
// A warp owns one front. Smaller CTAs reduce shared-memory allocation rounding while
// the default launch bound retains the same 16 resident warps' register budget.
#ifndef PIVOTED_WPB
#define PIVOTED_WPB 4
#endif
#if PIVOTED_WPB != 1 && PIVOTED_WPB != 2 && PIVOTED_WPB != 4
#error "PIVOTED_WPB must be 1, 2 or 4"
#endif
// An explicit FUSED_MIN_BLOCKS still overrides the default (e.g. 8 at WPB=4 asks
// for 32 warps' registers).
// The original block selector is unbounded: a launch bound can increase its register allocation.
// The per-warp selector declares its CTA size below so its 512-thread variant fits the register file.
#ifndef FUSED_MIN_BLOCKS
#define FUSED_MIN_BLOCKS (16 / PIVOTED_WPB)
#endif
// Each block-kernel warp chooses the same winning window column after the scan barrier, removing the serial
// thread-zero choice and its broadcast barrier. Zero restores that choice exactly, for comparison.
#ifndef PIVOTED_BSELECT_WARP
#define PIVOTED_BSELECT_WARP 0
#endif
// the block kernel's panel close on the tensor cores: the double build only (its scratch is real_t)
#if defined(PIVOTED_REAL_FLOAT)
#define PIVOTED_BWMMA 0
#else
#define PIVOTED_BWMMA 1
#endif
static int g_smem_limit = 0;    // bytes a block may use on this device, set by front_cuda_init
static int g_smem_per_sm = 0;   // the SM's whole shared memory
constexpr unsigned FULL = 0xffffffffu;
// Per front, a row of `counts` (int32, device memory): n_fully, slots_here, n_update. Device memory and not a
// by-value kernel parameter: a parameter array read at a runtime index is copied into every thread's local memory.

// the contribution's rows: the first min(count, slots_here) leftover candidates in slot order, padded to d_max,
// then the update slots n_fully .. n_fully + n_update - 1, padded to u_max; -1 marks padding. The forward and
// the backward kernel both call it on the alive flags after elimination, so they agree by construction.
template <int T>
__device__ int contribution_rows(const int* alive_slot, int n, int nf, int sh, int nu, int d_max, int u_max,
                                 int* rowslot, int* misc) {
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (warp == 0) {
    int found = 0;
    for (int base0 = 0; base0 < nf; base0 += 32) {
      int i = base0 + lane;
      bool ok = i < nf && alive_slot[i];
      unsigned msk = __ballot_sync(FULL, ok);
      int c = found + __popc(msk & ((1u << lane) - 1u));
      if (ok && c < d_max) rowslot[c] = c < sh ? i : -1;
      found += __popc(msk);
    }
    if (lane == 0) misc[0] = found;
    for (int c = (found < d_max ? found : d_max) + lane; c < d_max; c += 32) rowslot[c] = -1;
    for (int u = lane; u < u_max; u += 32) rowslot[d_max + u] = u < nu ? nf + u : -1;
  }
  __syncthreads();
  return misc[0];
}

__device__ __forceinline__ int pidx(int i, int j, int n) {   // i < j
  return i * (2 * n - i - 1) / 2 + (j - i - 1);
}

__device__ __forceinline__ real_t get(const real_t* P, int i, int j, int n) {
  if (i < j) return P[pidx(i, j, n)];
  if (i > j) return -P[pidx(j, i, n)];
  return 0.0;
}

// The warp's exact maximum of a non-negative real by the hardware reductions (sm_80+): the value's bits are
// monotone for non-negative reals, so the high word's maximum then, among the lanes holding it, the low word's;
// four instructions in place of five rounds of dependent shuffles.
__device__ __forceinline__ double warp_max_nonneg(double v) {
  const unsigned long long b = __double_as_longlong(v);
  const unsigned hi = unsigned(b >> 32), lo = unsigned(b);
  const unsigned mhi = __reduce_max_sync(FULL, hi);
  const unsigned mlo = __reduce_max_sync(FULL, hi == mhi ? lo : 0u);
  return __longlong_as_double((static_cast<unsigned long long>(mhi) << 32) | mlo);
}
__device__ __forceinline__ float warp_max_nonneg(float v) {
  const unsigned b = __float_as_uint(v);
  return __uint_as_float(__reduce_max_sync(FULL, b));
}
// The lowest index among the lanes whose value equals the maximum (-1 when no lane holds a candidate).
__device__ __forceinline__ int warp_argmin_at(bool at, int idx) {
  return __reduce_min_sync(FULL, at ? unsigned(idx) : 0xffffffffu) == 0xffffffffu ? -1
         : int(__reduce_min_sync(FULL, at ? unsigned(idx) : 0xffffffffu));
}

// The update block's Schur complement, deferred: the elimination skips every entry whose two indices
// are both in U — the last `nu` compact rows, the update slots, always alive — and the update-row parts of each
// step's c and tau, stored as (steps, ucap) a front, apply them at once at the end:
//   P[i][j] -= sum_q c_q[i] tau_q[j] - tau_q[i] c_q[j]   for nfc <= i < j < m.
// Exact: c_q and tau_q never read a U x U entry. `tid` strides by `T` over the packed U triangle.
__device__ void schur_uu_packed(real_t* P, int m, int nfc, int nu, int nsteps, const double* cub,
                                const double* tub, int ucap, int tid, int T) {
  const int pairs = nu * (nu - 1) / 2;
  int e = tid, ii = 0, jj = e + 1;
  while (ii < nu - 1 && jj >= nu) { jj = jj - nu + ii + 2; ++ii; }
  for (; e < pairs; e += T) {
    real_t acc = 0.0;
    for (int q = 0; q < nsteps; ++q) {
      const double* cq = cub + size_t(q) * ucap;
      const double* tq = tub + size_t(q) * ucap;
      acc += cq[ii] * tq[jj] - tq[ii] * cq[jj];
    }
    P[pidx(nfc + ii, nfc + jj, m)] -= acc;
    jj += T;
    while (ii < nu - 1 && jj >= nu) { jj = jj - nu + ii + 2; ++ii; }
  }
}

// the same on a full skew slab A (n x n, kept exactly skew)
__device__ void schur_uu_slab(real_t* A, int n, int nfc, int nu, int nsteps, const double* cub, const double* tub,
                              int ucap, int tid, int T) {
  const int pairs = nu * (nu - 1) / 2;
  int e = tid, ii = 0, jj = e + 1;
  while (ii < nu - 1 && jj >= nu) { jj = jj - nu + ii + 2; ++ii; }
  for (; e < pairs; e += T) {
    real_t acc = 0.0;
    for (int q = 0; q < nsteps; ++q) {
      const double* cq = cub + size_t(q) * ucap;
      const double* tq = tub + size_t(q) * ucap;
      acc += cq[ii] * tq[jj] - tq[ii] * cq[jj];
    }
    const real_t v = RSUB_RN(A[size_t(nfc + ii) * n + nfc + jj], acc);
    A[size_t(nfc + ii) * n + nfc + jj] = v;
    A[size_t(nfc + jj) * n + nfc + ii] = -v;
    jj += T;
    while (ii < nu - 1 && jj >= nu) { jj = jj - nu + ii + 2; ++ii; }
  }
}

// The block kernel's panel: up to PANEL_MAX pivot pairs whose rank-2 updates are held back and applied to the
// fronts' entries outside the up-to-date set U as one rank-2p update (see front_fwd_kernel).
constexpr int PANEL_MAX = 8;
constexpr int GPANEL_MAX = 16;   // the global kernels' panel: their close on the tensor cores makes a deeper one pay

struct Smem {
  int n;        // the level width (the planar panel's row count)
  real_t* PT;   // the open panel, planar: c_q[i] at PT[i * pmax + q], tau_q[i] at PT[pmax * n + i * pmax + q]
  real_t* scratch;   // the tensor-core close's tile a warp (16 x 16), with a panel of 4 or more
  real_t* P; real_t* c; real_t* tau; real_t* red_v; real_t* wkey; real_t* wm;
  int* alive; int* cand; int* failed; int* red_i; int* win; int* wr; int* misc;   // misc: nwin, k, r, m, nU
  int* cidx; int* slot;   // slot -> compact index (-1 dead), compact index -> slot
  int* inU; int* ulist;   // compact index -> in U; U's alive members
  int* smode;             // slot -> global mode id
};

// The sources of a level whose fronts assemble themselves (front_fwd_fused's tables, see there); by value
// The fused kernels' inputs: the sources (the flat free vector, the source levels' contribution blocks with their
// rows' validity and modes, the occupation) and the level's tables — the assembly PIECES (fast._pieces: per
// front, each site's packed Z block and each child's contribution block with a slot map), the bonds' constants
// (bd_pq (NB, 2) slot pairs p < q, bd_val), and the slots' alive and mode sources (asrc, msrc)
struct FusedIn {
  const real_t* free_; size_t fstride; const real_t* childC; size_t cstride;
  const int8_t* childV; const int32_t* childM; size_t vstride; const int8_t* occ; size_t ostride;
  const int32_t* pc_off; const int32_t* pc_desc; const int32_t* pc_map;
  const int32_t* bd_off; const int32_t* bd_pq; const real_t* bd_val;
  const int32_t* asrc; const int32_t* msrc;
  // the packed triangle's entries for every width m <= 320, as indices scaled by 16 (decoded by >> 4):
  // ijtab[ijoff[m] + lane * T + t] = (16 i) << 16 | 16 j for the entry e = lane + 32 t of width m, T = the
  // entries' count over 32 rounded up to a multiple of 4, 0 (the pair (0, 0): a null update) past the end; a lane
  // reads its entries four at a time, 16 bytes aligned, in place of the sweep's loop-carried decode
  const uint32_t* ijtab; const int32_t* ijoff;
  int tabn;   // the widest width the table covers (0: none — the global kernels; the old pair walks then)
};

// Front f's alive upper triangle from its pieces (the multifrontal extend-add), then its bonds: `store(ci, cj, v)`
// adds v to F[ci, cj] for compact indices ci < cj (cidx: slot -> compact index, -1 dead). Pair (r, c) of a piece
// is read once, the lanes along a row of the source block; the pieces overlap on the front, so each waits for
// the one before (WARP: one warp assembles; else the block). `tid` of `T` threads. (Staging the map in
// shared memory with more loads in flight a thread did not repay the registers it took.)
// The rows r < w of a block whose `key(r)` is >= 0, compacted in order into vl[k] = r and vc[k] = key(r): every warp
// walks the same ballots (so every lane knows the count), warp 0 writes. Returns the count.
template <class Key>
__device__ __forceinline__ int compact_rows(int w, Key key, int* vl, int* vc, int tid, int T) {
  const int lane = tid % 32, warp = tid / 32;
  int found = 0;
  for (int base0 = 0; base0 < w; base0 += 32) {
    const int r = base0 + lane;
    const int kv = r < w ? key(r) : -1;
    const unsigned msk = __ballot_sync(FULL, kv >= 0);
    if (kv >= 0 && warp == 0) { const int k = found + __popc(msk & ((1u << lane) - 1u)); vl[k] = r; vc[k] = kv; }
    found += __popc(msk);
  }
  if (T > 32) __syncthreads(); else __syncwarp();
  return found;
}

// The assembly's pair walk over a child's block: the entries a lane keeps in flight — their table words,
// slots and values loaded before any update is stored. The default is 4; the child blocks sit in L2 or
// HBM, so a lane's chain is this many loads a latency.
#ifndef PIVOTED_ASM_PIPE
#define PIVOTED_ASM_PIPE 4
#endif
template <bool WARP, class Store>
__device__ __forceinline__ void assemble_pieces(const FusedIn& fin, int f, const real_t* fr, const real_t* cc,
                                                const int* cidx, int tid, int T, Store store, int* vl = nullptr,
                                                int* vc = nullptr,   // vl, vc: scratch of the block's width (the global kernel passes none)
                                                int skipc = 0) {     // skipc: the first child pieces, done elsewhere (the global kernel's gather)
  const int lane = tid % 32, warp = tid / 32, W = T / 32;
  int sk = skipc;
  for (int p = fin.pc_off[f]; p < fin.pc_off[f + 1]; ++p) {
    const int base = fin.pc_desc[4 * p], w = fin.pc_desc[4 * p + 1], moff = fin.pc_desc[4 * p + 2];
    const int nown = fin.pc_desc[4 * p + 3];
    if (nown < 0 && sk > 0) { --sk; continue; }
    const int32_t* map = fin.pc_map + moff;
    const real_t* src = (nown >= 0 ? fr : cc) + base;
    if (nown < 0 && vl != nullptr && fin.tabn > 0 && w <= fin.tabn) {
      // a child's block: its rows that land on an alive slot of this front, compacted, then only their pairs (a < b
      // in the compact order is r < c in the block), each (a, b) from the width-L table
      const int L = compact_rows(w, [&](int r) { const int d = map[r]; return d >= 0 ? cidx[d] : -1; }, vl, vc, tid, T);
      const int E = L * (L - 1) / 2;
      const int T4 = ((E + 31) / 32 + 3) / 4 * 4;
      const uint32_t* tab = fin.ijtab + fin.ijoff[L] + lane * T4;
      // four pairs a lane in flight: their table words, slots and values loaded before any update is stored
      const int Tl = lane < E ? (E - lane + 31) / 32 : 0;     // the lane's entries e = lane + 32 t, t < Tl
      int t = warp;
      constexpr int K = PIVOTED_ASM_PIPE;
      for (; t + (K - 1) * W < Tl; t += K * W) {
        int ci[K], cj[K]; real_t val[K];
#pragma unroll
        for (int uu = 0; uu < K; ++uu) {
          const unsigned v = __ldg(tab + t + uu * W);
          const int a = (v >> 16) >> 4, b = (v & 0xffffu) >> 4;
          ci[uu] = vc[a]; cj[uu] = vc[b];
          val[uu] = src[vl[a] * w + vl[b]];
        }
#pragma unroll
        for (int uu = 0; uu < K; ++uu) {
          if (ci[uu] < cj[uu]) store(ci[uu], cj[uu], val[uu]); else store(cj[uu], ci[uu], -val[uu]);
        }
      }
      for (; t < Tl; t += W) {
        const unsigned v = __ldg(tab + t);
        const int a = (v >> 16) >> 4, b = (v & 0xffffu) >> 4;
        const int ci = vc[a], cj = vc[b];
        const real_t val = src[vl[a] * w + vl[b]];
        if (ci < cj) store(ci, cj, val); else store(cj, ci, -val);
      }
      if (WARP) __syncwarp(); else __syncthreads();
      continue;
    }
    const int pairs = w * (w - 1) / 2;
    int e = tid, r = 0, c = e + 1;    // (r, c) of pair e in the row-major strict upper triangle of w
    while (r < w - 1 && c >= w) { c = c - w + r + 2; ++r; }
    for (; e < pairs; e += T) {
      const int dr = map[r], dc = map[c];
      if (dr >= 0 && dc >= 0 && (nown < 0 || dr < nown || dc < nown)) {
        const int ci = cidx[dr], cj = cidx[dc];
        if (ci >= 0 && cj >= 0) {        // a site's block is packed, a child's square; both read at r < c only
          const real_t v = nown >= 0 ? src[e] : src[r * w + c];
          if (ci < cj) store(ci, cj, v); else store(cj, ci, -v);
        }
      }
      c += T;
      while (r < w - 1 && c >= w) { c = c - w + r + 2; ++r; }
    }
    if (WARP) __syncwarp(); else __syncthreads();
  }
  for (int q = fin.bd_off[f] + tid; q < fin.bd_off[f + 1]; q += T) {
    const int ci = cidx[fin.bd_pq[2 * q]], cj = cidx[fin.bd_pq[2 * q + 1]];
    if (ci >= 0 && cj >= 0) store(ci, cj, fin.bd_val[q]);
  }
}

#ifndef PIVOTED_BLOCK_COMPACT
#define PIVOTED_BLOCK_COMPACT 0   // positive: smaller triangle first, full-capacity retry on overflow
#endif
#ifndef PIVOTED_BLOCK_THREADS
#define PIVOTED_BLOCK_THREADS 0   // 128: fronts wider than 96; 256: fronts wider than 128; 0: original dispatch
#endif
#if PIVOTED_BLOCK_THREADS != 0 && PIVOTED_BLOCK_THREADS != 128 && PIVOTED_BLOCK_THREADS != 256
#error "PIVOTED_BLOCK_THREADS must be 0, 128 or 256"
#endif

template <int T>
__device__ Smem carve(char* base, int n, int pmax, int capacity = 0, int window = 0, int entries = 0) {
  constexpr int THREADS = T;
  const int WINDOW = window > 0 ? window : T / 32;
  Smem s;
  s.n = n;
  s.PT = reinterpret_cast<real_t*>(base);
  s.P = s.PT + 2 * size_t(pmax) * n;
  const int cap = capacity > 0 ? capacity : n;
  real_t* d = s.P + (entries > 0 ? entries : cap * (cap - 1) / 2);   // entries: the triangle's prefix
  s.c = d; s.tau = d + n; s.red_v = d + 2 * n; s.wkey = s.red_v + THREADS; s.wm = s.wkey + WINDOW;
  int* q = reinterpret_cast<int*>(s.wm + WINDOW);
  s.alive = q; s.cand = q + n; s.failed = q + 2 * n; s.red_i = q + 3 * n;
  s.win = s.red_i + THREADS; s.wr = s.win + WINDOW; s.misc = s.wr + WINDOW;
  s.cidx = s.misc + 8; s.slot = s.cidx + n; s.inU = s.slot + n; s.ulist = s.inU + n; s.smode = s.ulist + n;
  s.scratch = reinterpret_cast<real_t*>((reinterpret_cast<size_t>(s.smode + n) + 31) & ~size_t(31));
  return s;
}

size_t smem_bytes(int n, int THREADS, int pmax = 0, int capacity = 0, int window = 0, int entries = 0) {
  const int WINDOW = window > 0 ? window : THREADS / 32;
  const int cap = capacity > 0 ? capacity : n;
  const size_t tri = entries > 0 ? size_t(entries) : size_t(cap) * (cap - 1) / 2;
  // the tile scratch of the tensor-core close (a 16 x 16 tile a warp) only with a panel of 4 or more
  const size_t scratch = pmax >= 4 ? size_t(THREADS / 32) * 256 : 0;
  return sizeof(real_t) * (2 * size_t(pmax) * n + tri + 2 * n + THREADS + 2 * WINDOW + scratch)
       + sizeof(int) * (8 * n + THREADS + 2 * WINDOW + 8) + 32;
}

// The panel. An entry (i, j) is up to date when i or j is in U; the others still owe the open panel's tp pairs.
// A column joins U by paying them: A(i, k) -= sum_q c_q[i] tau_q[k] - tau_q[i] c_q[k] for every alive i outside U.
// Each term is written as the unblocked update writes it, so a panel of one pair computes what it computed.
template <int T>
__device__ void bring_in(Smem& s, int m, int pmax, int tp, int k) {
  if (tp > 0) {
    const int n_ = s.n;
    const real_t *ck = s.PT + size_t(k) * pmax, *tk = s.PT + size_t(pmax) * n_ + size_t(k) * pmax;
    for (int i = threadIdx.x; i < m; i += T) {
      if (i == k || !s.alive[i] || s.inU[i]) continue;
      const real_t *ci = s.PT + size_t(i) * pmax, *ti = s.PT + size_t(pmax) * n_ + size_t(i) * pmax;
      if (i < k) {
        real_t acc = ci[0] * tk[0] - ti[0] * ck[0];
        for (int q = 1; q < tp; ++q) acc += ci[q] * tk[q] - ti[q] * ck[q];
        s.P[pidx(i, k, m)] -= acc;
      } else {     // the stored entry is A(k, i)
        real_t acc = ck[0] * ti[0] - tk[0] * ci[0];
        for (int q = 1; q < tp; ++q) acc += ck[q] * ti[q] - tk[q] * ci[q];
        s.P[pidx(k, i, m)] -= acc;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) s.inU[k] = 1;
  __syncthreads();
}

// Closing the panel: the rank-2tp update of every alive entry outside U, by 8-row x 32-column tiles of the upper
// triangle (column block jb has 4 (jb + 1) of them); a lane holds its column's pairs in registers, a row's pairs
// are read as broadcasts. U empties.
template <int T>
__device__ void close_panel(Smem& s, int m, int pmax, int tp) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n_ = s.n;
  const real_t* Cp = s.PT;
  const real_t* Tp = s.PT + size_t(pmax) * n_;
  if (PIVOTED_BWMMA && pmax % 4 == 0 && tp > 0) {
    // the stale panel columns zeroed; the whole 16 x 16 tiles of the live square by wmma (C, T planar in shared
    // memory), each tile through the warp's scratch into the packed triangle where i < j and neither is in U
    const int mf = m / 16 * 16, tiles = mf / 16;
    for (int idx = threadIdx.x; idx < m * pmax; idx += T) {
      if (idx % pmax >= tp) { s.PT[idx] = 0.0; s.PT[size_t(pmax) * n_ + idx] = 0.0; }
    }
    __syncthreads();
#if !defined(PIVOTED_REAL_FLOAT)   // the fp64 tensor-core tiles; the float build never takes this branch (PIVOTED_BWMMA 0)
    using namespace nvcuda;
    real_t* sc = s.scratch + warp * 256;
    for (int tile = warp; tile < tiles * (tiles + 1) / 2; tile += T / 32) {
      int ti = 0, tj = tile;
      while (tj >= tiles - ti) { tj -= tiles - ti; ++ti; }
      tj += ti;
      wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc[2][2];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::fill_fragment(acc[p][q], 0.0);
      for (int k0 = 0; k0 < pmax; k0 += 4) {
        wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bb[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) wmma::load_matrix_sync(a[p], Cp + size_t(ti * 16 + p * 8) * pmax + k0, pmax);
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], Tp + size_t(tj * 16 + q * 8) * pmax + k0, pmax);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          wmma::load_matrix_sync(a[p], Tp + size_t(ti * 16 + p * 8) * pmax + k0, pmax);
#pragma unroll
          for (int e = 0; e < a[p].num_elements; ++e) a[p].x[e] = -a[p].x[e];
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], Cp + size_t(tj * 16 + q * 8) * pmax + k0, pmax);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
      }
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::store_matrix_sync(sc + p * 8 * 16 + q * 8, acc[p][q], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int i = ti * 16 + e / 16, j = tj * 16 + e % 16;
        if (i < j && s.alive[i] && s.alive[j] && !s.inU[i] && !s.inU[j]) s.P[pidx(i, j, m)] -= sc[e];
      }
      __syncwarp();
    }
#endif
    __syncthreads();
    const int rest = m - mf;      // the tail rows and columns past the last whole tile, the scalar form
    for (int e = threadIdx.x; e < m * rest; e += T) {
      const int j = mf + e % rest, i = e / rest;
      const int lo = i < j ? i : j, hi = i < j ? j : i;
      if (lo == hi || (i < mf && j < mf)) continue;
      if (i >= mf && j >= mf && i > j) continue;       // the tail's own square: each pair once
      if (!s.alive[lo] || !s.alive[hi] || s.inU[lo] || s.inU[hi]) continue;
      const real_t *clo = Cp + size_t(lo) * pmax, *tlo = Tp + size_t(lo) * pmax;
      const real_t *chi = Cp + size_t(hi) * pmax, *thi = Tp + size_t(hi) * pmax;
      real_t acc = clo[0] * thi[0] - tlo[0] * chi[0];
      for (int q = 1; q < tp; ++q) acc += clo[q] * thi[q] - tlo[q] * chi[q];
      s.P[pidx(lo, hi, m)] -= acc;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < m; c += T) s.inU[c] = 0;
    __syncthreads();
    return;
  }
  const int nb = (m + 31) / 32, items = 2 * nb * (nb + 1);
  for (int it = warp; it < items; it += T / 32) {
    int jb = int((sqrtf(2.0f * it + 1.0f) - 1.0f) * 0.5f);     // the largest jb with 2 jb (jb + 1) <= it
    while (2 * (jb + 1) * (jb + 2) <= it) ++jb;
    while (2 * jb * (jb + 1) > it) --jb;
    const int i0 = 8 * (it - 2 * jb * (jb + 1));
    const int j = 32 * jb + lane;
    const bool colok = j < m && s.alive[j] && !s.inU[j];
    real_t cj[PANEL_MAX], tj[PANEL_MAX];
#pragma unroll
    for (int q = 0; q < PANEL_MAX; ++q) {
      cj[q] = (q < tp && colok) ? Cp[size_t(j) * pmax + q] : 0.0;
      tj[q] = (q < tp && colok) ? Tp[size_t(j) * pmax + q] : 0.0;
    }
    const int i1 = min(i0 + 8, m);
    for (int i = i0; i < i1; ++i) {
      if (!s.alive[i] || s.inU[i]) continue;      // the same for every lane
      if (!colok || j <= i) continue;
      const real_t *ci = Cp + size_t(i) * pmax, *ti_ = Tp + size_t(i) * pmax;
      real_t acc = ci[0] * tj[0] - ti_[0] * cj[0];
#pragma unroll
      for (int q = 1; q < PANEL_MAX; ++q)
        if (q < tp) acc += ci[q] * tj[q] - ti_[q] * cj[q];
      s.P[pidx(i, j, m)] -= acc;
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < m; c += T) s.inU[c] = 0;
  __syncthreads();
}

// FUSED: the front assembles itself from `fin` (front_fwd_fused's tables) instead of reading F0, alive0 and mode;
// the per-front counts are then the level's table, read at the front's index.
// The block kernel's residency floor: blocks an SM the compiler must fit by registers (1 = its own
// choice, 128 registers at 256 threads = two blocks); with the U x U block out of shared memory the triangle no
// longer binds and 3 asks for 85 registers or fewer
#ifndef PIVOTED_BLOCK_MINBLOCKS
#define PIVOTED_BLOCK_MINBLOCKS 1
#endif
template <int T>
__global__ void
__launch_bounds__(T, PIVOTED_BLOCK_MINBLOCKS)   // always: at 512 threads the registers must fit 128 a thread or the launch is refused
front_fwd_kernel(const int32_t* counts, int nfront, int n, int steps, real_t u,
                                 int root, int record, int d_max, int u_max, real_t* Cout, int8_t* cvalid,
                                 int32_t* cmode, int32_t* count, int8_t* alive_out, real_t* logabs,
                                 real_t* sign, int32_t* pivots, real_t* growth, int32_t* rec_kr,
                                 real_t* rec_a, real_t* rec_c, real_t* rec_tau, double* cu, double* tu, int ucp,
                                 int pmax, int search_window, const FusedIn fin,
                                 int matrix_capacity, int fallback_only, int uuent) {
  // uuent: > 0 = the shared triangle holds its first uuent entries only — the rows below the update
  // boundary — and a deferred front's U x U block is gathered at the contribution write; 0 = the full triangle
  constexpr int THREADS = T;
  const int WINDOW = ((PIVOTED_BLOCK_THREADS == 256 && T == 256) ||
                      (PIVOTED_BLOCK_THREADS == 128 && T == 128)) && n > 128 ? 512 / 32 :
                     PIVOTED_BLOCK_THREADS == 128 && T == 128 && n > 96 ? 256 / 32 : T / 32;
  // Zero preserves the recorded window; a smaller positive value tries fewer columns,
  // with the same threshold and the same fallback over untested candidates.
  const int scan_window = search_window > 0 && search_window < WINDOW ? search_window : WINDOW;
  extern __shared__ __align__(16) char base[];
  const int b = blockIdx.x, f = b % nfront, t = threadIdx.x;
#if PIVOTED_BLOCK_COMPACT > 0
  if (fallback_only && count[b] >= 0) return;   // uniform across this front's whole CTA
#endif
  Smem s = carve<T>(base, n, pmax, matrix_capacity, WINDOW, uuent);
  const size_t vec = size_t(b) * n, cfg = size_t(b) / nfront;
  const int nf = counts[3 * f], sh_ = counts[3 * f + 1], nu_ = counts[3 * f + 2];
  const int warp = t / 32, lane = t % 32;
  // each slot's alive flag (into `failed`, free until the compaction is done) and mode
  for (int i = t; i < n; i += THREADS) {
    int a_, md;
    const int code = fin.asrc[f * n + i], mc = fin.msrc[f * n + i];
    if (code == -1) a_ = 1;
    else if (code == -2) a_ = 0;
    else if (code >= 0) a_ = fin.occ[cfg * fin.ostride + code] != 0;
    else a_ = fin.childV[cfg * fin.vstride + (-3 - code)] != 0;
    md = mc >= -1 ? mc : fin.childM[cfg * fin.vstride + (-3 - mc)];
    s.failed[i] = a_;
    s.smode[i] = md;
  }
  __syncthreads();
  // compact the alive slots, in slot order: dead rows (unoccupied physical modes, empty delay
  // slots, the level's padding) never enter a scan or an update
  if (warp == 0) {
    int found = 0;
    for (int base0 = 0; base0 < n; base0 += 32) {
      int i = base0 + lane;
      bool ok = i < n && s.failed[i] != 0;
      unsigned msk = __ballot_sync(FULL, ok);
      int c = found + __popc(msk & ((1u << lane) - 1u));
      if (i < n) s.cidx[i] = ok ? c : -1;
      if (ok) s.slot[c] = i;
      found += __popc(msk);
    }
    if (lane == 0) s.misc[3] = found;
  }
  __syncthreads();
  const int m = s.misc[3];
#if PIVOTED_BLOCK_COMPACT > 0
  // No matrix access or output record precedes this uniform decision. Logical maps,
  // panel vectors and delay slots retain their full n-wide capacity in both passes.
  if (m > matrix_capacity) {
    if (t == 0) count[b] = -1;
    return;
  }
#endif
  const int nfc = m - nu_;      // the update rows are the last nu_ compact rows (always alive)
  const bool defer = pmax == 1 && nu_ > 1;   // U x U waits for one Schur complement at the end
  const bool uu_out = uuent > 0 && defer;                // U x U not in shared memory at all
  const int tri = uu_out ? nfc * (2 * m - nfc - 1) / 2 : m * (m - 1) / 2;   // the triangle's entries in use
  for (int c = t; c < m; c += THREADS) {
    s.alive[c] = 1;
    s.cand[c] = s.slot[c] < nf;
    s.failed[c] = 0;
    s.inU[c] = 0;
  }
  {      // the alive upper triangle: zero, then the pieces and the bonds
    for (int e = t; e < tri; e += THREADS) s.P[e] = 0.0;
    __syncthreads();
    real_t* P = s.P;
    assemble_pieces<false>(fin, f, fin.free_ + cfg * fin.fstride, fin.childC + cfg * fin.cstride, s.cidx, t,
                           THREADS, [&](int ci, int cj, real_t v) { if (uu_out && ci >= nfc) return; P[pidx(ci, cj, m)] += v; },
                           s.failed, s.ulist);
    __syncthreads();
    for (int c = t; c < m; c += THREADS) s.failed[c] = 0;     // the assembly's scratch
  }
  __syncthreads();
  double la = 0.0, sg = 1.0; real_t big = 0.0;
  int step = 0, tp = 0;    // tp: pairs in the open panel
  for (; step < steps; ++step) {
#if PIVOTED_BSELECT_WARP
    int chosen_k = -1, chosen_r = -1;
#endif
    // rounds of WINDOW candidate columns until one passes or none is left untested
    while (true) {
      if (warp == 0) {      // the lowest untested candidates, by ballot
        int found = 0;
        for (int base0 = 0; base0 < m && found < scan_window; base0 += 32) {
          int i = base0 + lane;
          bool ok = i < m && s.cand[i] && !s.failed[i];
          unsigned msk = __ballot_sync(FULL, ok);
          int before = __popc(msk & ((1u << lane) - 1u));
          if (ok && found + before < scan_window) s.win[found + before] = i;
          found += __popc(msk);
        }
        if (lane == 0) s.misc[0] = found < scan_window ? found : scan_window;
      }
      __syncthreads();
      const int nwin = s.misc[0];
      if (nwin == 0) { if (t == 0) s.misc[1] = -1; __syncthreads(); break; }
      if (pmax > 1) {        // the scanned columns must be up to date; with no pair pending they are
        if (tp == 0) { if (t < nwin) s.inU[s.win[t]] = 1; __syncthreads(); }
        else for (int w = 0; w < nwin; ++w)
          if (!s.inU[s.win[w]]) bring_in<T>(s, m, pmax, tp, s.win[w]);
      }
      auto scan_column = [&](int w) {
        const int k = s.win[w];
        real_t mx = 0.0, bv = 0.0; int br = -1;
        for (int i = lane; i < m; i += 32) {
          if (!s.alive[i] || i == k) continue;
          real_t v = fabs(get(s.P, i, k, m));
          mx = fmax(mx, v);
          if (s.cand[i] && v > bv) { bv = v; br = i; }
        }
        mx = warp_max_nonneg(mx);
        {
          const real_t bmax = warp_max_nonneg(bv);
          br = warp_argmin_at(br >= 0 && bv == bmax, br);
          bv = bmax;
        }
        if (lane == 0) {
          real_t key = -1.0;
          if (br >= 0 && bv > 0.0) key = root ? bv : (bv / mx >= u ? bv / mx : -1.0);
          s.wkey[w] = key; s.wr[w] = br; s.wm[w] = mx;
        }
      };
      if (warp < nwin) scan_column(warp);
      // Only the opt-in large-front 256-thread CTA can need a second wave. Other
      // variants retain the original single scan branch.
      if constexpr (PIVOTED_BLOCK_THREADS == 256 && T == 256)
        if (warp + THREADS / 32 < nwin) scan_column(warp + THREADS / 32);
      // Four physical warps cover the same eight/sixteen logical candidates in waves.
      // All warps complete these scans before the shared-key selection barrier below.
      if constexpr (PIVOTED_BLOCK_THREADS == 128 && T == 128)
        for (int w = warp + THREADS / 32; w < nwin; w += THREADS / 32) scan_column(w);
      __syncthreads();
#if PIVOTED_BSELECT_WARP
      // The shared keys are complete. Each warp independently applies the old strict-greater comparison's
      // tie rule (the lowest window index), so the successful path needs no block broadcast of k and r.
      const real_t key = lane < nwin ? s.wkey[lane] : real_t(-1);
      const real_t best_key = warp_max_nonneg(fmax(key, real_t(0)));
      const int best = warp_argmin_at(key >= 0 && key == best_key, lane);
      if (warp == 0) {
        const real_t mx = warp_max_nonneg(lane < nwin ? s.wm[lane] : real_t(0));
        if (lane == 0) big = fmax(big, mx);
      }
      if (best >= 0) { chosen_k = s.win[best]; chosen_r = s.wr[best]; break; }
      if (t < nwin) s.failed[s.win[t]] = 1;
      // All warps must finish reading the old window before warp zero constructs the next one.
      __syncthreads();
#else
      if (t == 0) {
        int best = -1; real_t bk = -1.0;
        for (int w = 0; w < nwin; ++w) {
          big = fmax(big, s.wm[w]);
          if (s.wkey[w] > bk) { bk = s.wkey[w]; best = w; }
        }
        if (best >= 0) { s.misc[1] = s.win[best]; s.misc[2] = s.wr[best]; }
        else { s.misc[1] = -2; for (int w = 0; w < nwin; ++w) s.failed[s.win[w]] = 1; }
      }
      __syncthreads();
      if (s.misc[1] != -2) break;
#endif
    }
#if PIVOTED_BSELECT_WARP
    const int k = chosen_k, r = chosen_r;
    if (k < 0) break;      // no pair passes: the rest is the parent's
#else
    const int k = s.misc[1];
    if (k < 0) break;      // no pair passes: the rest is the parent's
    const int r = s.misc[2];
#endif
    if (pmax > 1 && !s.inU[r]) bring_in<T>(s, m, pmax, tp, r);     // c is column r: up to date first
    const real_t a = get(s.P, k, r, m);
    for (int i = t; i < m; i += THREADS) {
      bool rest = s.alive[i] && i != k && i != r;
      const real_t ci = rest ? get(s.P, i, r, m) : 0.0, ti = rest ? get(s.P, k, i, m) / a : 0.0;
      s.c[i] = ci; s.tau[i] = ti;
      s.PT[size_t(i) * pmax + tp] = ci; s.PT[size_t(pmax) * n + size_t(i) * pmax + tp] = ti;
      s.failed[i] = 0;      // the update may make a failed column pass
    }
    __syncthreads();
    const int sk = s.slot[k], sr = s.slot[r];
    if (t == 0) {
      la += log(fabs(a)); sg *= (a < 0 ? -1.0 : 1.0);
      s.alive[k] = 0; s.alive[r] = 0; s.cand[k] = 0; s.cand[r] = 0;   // the update reads c and tau only
      pivots[(size_t(b) * steps + step) * 2] = s.smode[sk];
      pivots[(size_t(b) * steps + step) * 2 + 1] = s.smode[sr];
      if (record) {
        rec_kr[(size_t(b) * steps + step) * 2] = sk; rec_kr[(size_t(b) * steps + step) * 2 + 1] = sr;
        rec_a[size_t(b) * steps + step] = a;
      }
    }
    if (record)              // the multipliers in slot coordinates, for the backward kernel
      for (int i = t; i < n; i += THREADS) {
        const int ci = s.cidx[i];
        rec_c[(size_t(b) * steps + step) * n + i] = ci >= 0 ? s.c[ci] : 0.0;
        rec_tau[(size_t(b) * steps + step) * n + i] = ci >= 0 ? s.tau[ci] : 0.0;
      }
    if (defer)               // the update rows' c and tau, for the deferred Schur complement
      for (int i = nfc + t; i < m; i += THREADS) {
        cu[(size_t(b) * steps + step) * ucp + i - nfc] = s.c[i];
        tu[(size_t(b) * steps + step) * ucp + i - nfc] = s.tau[i];
      }
    if (pmax == 1) {         // no panel: the rank-2 update of everything but U x U, a warp per folded row pair
      for (int p = warp; p < (m + 1) / 2; p += THREADS / 32) {
        for (int side = 0; side < 2; ++side) {
          int i = side ? m - 1 - p : p;
          if (side && i == p) break;
          if (defer && i >= nfc) continue;     // row i of U: its entries j > i are all in U
          real_t ci = s.c[i], ti = s.tau[i];
          if (ci == 0.0 && ti == 0.0) continue;
          for (int j = i + 1 + lane; j < m; j += 32)
            s.P[pidx(i, j, m)] -= ci * s.tau[j] - ti * s.c[j];
        }
      }
      __syncthreads();
      continue;
    }
    __syncthreads();
    // the rank-2 update now, only where it is read before the panel closes: the rows and columns of U
    if (warp == 0) {
      int found = 0;
      for (int base0 = 0; base0 < m; base0 += 32) {
        int i = base0 + lane;
        bool ok = i < m && s.alive[i] && s.inU[i];
        unsigned msk = __ballot_sync(FULL, ok);
        if (ok) s.ulist[found + __popc(msk & ((1u << lane) - 1u))] = i;
        found += __popc(msk);
      }
      if (lane == 0) s.misc[4] = found;
    }
    __syncthreads();
    const int nU = s.misc[4];
    // by stored rows, contiguous: U's rows whole, (u, j) for j > u; every other row at U's columns to its right
    for (int w = warp; w < nU; w += THREADS / 32) {
      const int uu = s.ulist[w];
      const real_t cu = s.c[uu], tu = s.tau[uu];
      for (int j = uu + 1 + lane; j < m; j += 32)
        s.P[pidx(uu, j, m)] -= cu * s.tau[j] - tu * s.c[j];
    }
    for (int i = warp; i < m; i += THREADS / 32) {
      if (!s.alive[i] || s.inU[i]) continue;
      const real_t ci = s.c[i], ti = s.tau[i];
      for (int w = lane; w < nU; w += 32) {
        const int uu = s.ulist[w];
        if (uu > i) s.P[pidx(i, uu, m)] -= ci * s.tau[uu] - ti * s.c[uu];
      }
    }
    __syncthreads();
    if (++tp == pmax) { close_panel<T>(s, m, pmax, tp); tp = 0; }
  }
  if (tp > 0) close_panel<T>(s, m, pmax, tp);
  if (defer && !uu_out) {
    __syncthreads();
    schur_uu_packed(s.P, m, nfc, nu_, step, cu + size_t(b) * steps * ucp, tu + size_t(b) * steps * ucp, ucp, t,
                    THREADS);
    __syncthreads();
  }
  for (int q = step + t; q < steps; q += THREADS) {   // the steps no pivot took
    pivots[(size_t(b) * steps + q) * 2] = -1; pivots[(size_t(b) * steps + q) * 2 + 1] = -1;
    if (record) { rec_kr[(size_t(b) * steps + q) * 2] = -1; rec_kr[(size_t(b) * steps + q) * 2 + 1] = -1; }
  }
  if (t == 0) {   // growth: the largest entry met in the scanned columns
    logabs[b] = la; sign[b] = sg; growth[b] = big;
  }
  // alive by slot, into the (now free) c vector's integer view: the contribution's rows read it
  int* alive_slot = s.failed;    // n ints, free after the elimination
  __syncthreads();
  for (int i = t; i < n; i += THREADS) {
    const int ci = s.cidx[i];
    const int a_ = ci >= 0 ? s.alive[ci] : 0;
    alive_out[vec + i] = a_;
    alive_slot[i] = a_;
  }
  __syncthreads();
  const int cd = d_max + u_max;
  int* rowslot = s.slot;         // cd <= n ints; the compact->slot map is no longer needed
  const int left = contribution_rows<T>(alive_slot, n, nf, sh_, nu_, d_max, u_max, rowslot,
                                        s.misc);
  if (t == 0) count[b] = left;
  const size_t cvec = size_t(b) * cd, cmat = size_t(b) * cd * cd;
  for (int c = t; c < cd; c += THREADS) {
    const int sl = rowslot[c];
    cvalid[cvec + c] = sl >= 0;
    cmode[cvec + c] = sl >= 0 ? s.smode[sl] : -1;
  }
  // U x U out of shared memory: the pairs with both indices in U are uu_tc_kernel's, launched after
  if (fin.tabn > 0) {   // the valid rows compacted, then only their pairs by the width-L table
    int* vl = s.failed; int* vc = s.ulist;
    const int L = compact_rows(cd, [&](int c) { const int sl = rowslot[c]; return sl >= 0 ? s.cidx[sl] : -1; }, vl, vc, t, THREADS);
    {
      const int E = L * (L - 1) / 2;
      const int T4 = ((E + 31) / 32 + 3) / 4 * 4;
      const uint32_t* tab = fin.ijtab + fin.ijoff[L] + lane * T4;
      for (int tt = warp; lane + 32 * tt < E; tt += THREADS / 32) {
        const unsigned v = __ldg(tab + tt);
        const int a = (v >> 16) >> 4, b = (v & 0xffffu) >> 4;
        const int ci = vc[a], cj = vc[b];
        if (uu_out && ci >= nfc && cj >= nfc) continue;
        Cout[cmat + size_t(vl[a]) * cd + vl[b]] = get(s.P, ci, cj, m);
      }
    }
  } else for (int e = t; e < cd * cd; e += THREADS) {   // the valid rows' strict upper triangle only
    const int r = e / cd, q = e - r * cd;
    const int sr_ = rowslot[r], sq = rowslot[q];
    if (r < q && sr_ >= 0 && sq >= 0) {
      const int ci = s.cidx[sr_], cj = s.cidx[sq];
      if (uu_out && ci >= nfc && cj >= nfc) continue;
      Cout[cmat + e] = get(s.P, ci, cj, m);
    }
  }
  __syncthreads();
}

constexpr int UU_THREADS = 256;
constexpr int UU_MAXN = 1024;   // the writer's inverse maps; a wider level is not deferred (guu_on), never silently left unwritten
// The deferred fronts' U x U block, after front_fwd_kernel ran with uu_out (or front_gfwd_kernel with guu_on): the
// entry (u, v), u < v < nu, of the contribution block's update rows is the children's entries at the parent's update
// slots nf + u and nf + v, less the Schur complement of the steps taken (pivots -1 past them). On the fp64 tensor
// cores (the float build too its records double), a block a front, a warp a 16 x 16 tile of the upper U x U
// (ti <= tj): the children's entries summed into the tile, then the Schur complement of the steps taken subtracted
// as (-C_u) T_v^T + T_u C_v^T with C_u(u, q) = cu[q * ucp + u] read column-major and T_v^T(q, v) = tu[q * ucp + v]
// row-major (ucp even, so a 4-step chunk starts 32-byte aligned), the steps past the last multiple of 4 in scalar;
// the tile's strict upper entries go to the contribution block's update rows.
__global__ void __launch_bounds__(UU_THREADS) uu_tc_kernel(const FusedIn fin, const int32_t* counts, int nfront,
    int n, int steps, int d_max, int u_max, real_t* Cout, const int32_t* pivots, const double* cu, const double* tu,
    int ucp) {
  __shared__ int inv[3][UU_MAXN];
  __shared__ __align__(32) double tilebuf[UU_THREADS / 32][256];
  __shared__ int sh_nsteps;
  const int b = blockIdx.x, f = b % nfront, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const size_t cfg = size_t(b) / nfront;
  const int nf = counts[3 * f], nu = counts[3 * f + 2];
  if (nu < 2) return;
  const int cd = d_max + u_max;
  int nch = 0; int ucw[3]; const real_t* ucb[3];
  for (int p = fin.pc_off[f]; p < fin.pc_off[f + 1] && nch < 3; ++p) {
    if (fin.pc_desc[4 * p + 3] >= 0) continue;
    ucw[nch] = fin.pc_desc[4 * p + 1];
    ucb[nch] = fin.childC + cfg * fin.cstride + fin.pc_desc[4 * p];
    const int32_t* map = fin.pc_map + fin.pc_desc[4 * p + 2];
    for (int u = t; u < nu; u += UU_THREADS) inv[nch][u] = -1;
    __syncthreads();
    for (int r = t; r < ucw[nch]; r += UU_THREADS) {
      const int d = map[r];
      if (d >= nf && d < nf + nu) inv[nch][d - nf] = r;
    }
    ++nch;
  }
  if (t == 0) {
    int q = 0;
    while (q < steps && pivots[(size_t(b) * steps + q) * 2] >= 0) ++q;
    sh_nsteps = q;
  }
  __syncthreads();
  const int nsteps = sh_nsteps, k4 = nsteps / 4 * 4;
  const double* cub = cu + size_t(b) * steps * ucp;
  const double* tub = tu + size_t(b) * steps * ucp;
  const size_t cmat = size_t(b) * cd * cd;
  const int tiles = (nu + 15) / 16;
  double* tb = tilebuf[warp];
  using namespace nvcuda;
  for (int tile = warp; tile < tiles * (tiles + 1) / 2; tile += UU_THREADS / 32) {
    int ti = 0, tj = tile;
    while (tj >= tiles - ti) { tj -= tiles - ti; ++ti; }
    tj += ti;
    // the children's sum, a lane eight entries
    for (int e = lane; e < 256; e += 32) {
      const int u = ti * 16 + e / 16, v = tj * 16 + e % 16;
      double acc = 0.0;
      if (u < nu && v < nu && u != v)
        for (int q = 0; q < nch; ++q) {
          const int ri = inv[q][u], rj = inv[q][v];
          if (ri >= 0 && rj >= 0) acc += ri < rj ? ucb[q][ri * ucw[q] + rj] : -ucb[q][rj * ucw[q] + ri];
        }
      tb[e] = acc;
    }
    __syncwarp();
    // the tile's rows and columns past nu read the recorded vectors' padding (u_max >= nu, the buffer ucp wide):
    // finite or not, those entries are never written out
    const bool full = (ucp % 4 == 0) && ti * 16 + 16 <= ucp && tj * 16 + 16 <= ucp;   // ucp % 4: every front's base 32-byte aligned
    if (full && k4 > 0) {
      wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc[2][2];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(acc[p][q], tb + p * 8 * 16 + q * 8, 16, wmma::mem_row_major);
      for (int k0 = 0; k0 < k4; k0 += 4) {
        wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::col_major> a[2];
        wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> bb[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {      // (-C_u) T_v^T
          wmma::load_matrix_sync(a[p], cub + size_t(k0) * ucp + ti * 16 + p * 8, ucp);
#pragma unroll
          for (int e = 0; e < a[p].num_elements; ++e) a[p].x[e] = -a[p].x[e];
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], tub + size_t(k0) * ucp + tj * 16 + q * 8, ucp);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
#pragma unroll
        for (int p = 0; p < 2; ++p) wmma::load_matrix_sync(a[p], tub + size_t(k0) * ucp + ti * 16 + p * 8, ucp);   // + T_u C_v^T
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], cub + size_t(k0) * ucp + tj * 16 + q * 8, ucp);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
      }
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::store_matrix_sync(tb + p * 8 * 16 + q * 8, acc[p][q], 16, wmma::mem_row_major);
      __syncwarp();
    }
    const int q0 = full ? k4 : 0;          // the steps the products did not take
    for (int e = lane; e < 256; e += 32) {
      const int u = ti * 16 + e / 16, v = tj * 16 + e % 16;
      if (u < v && v < nu) {
        double sch = 0.0;
        for (int q = q0; q < nsteps; ++q)
          sch += cub[size_t(q) * ucp + u] * tub[size_t(q) * ucp + v] - tub[size_t(q) * ucp + u] * cub[size_t(q) * ucp + v];
        Cout[cmat + size_t(d_max + u) * cd + d_max + v] = tb[e] - sch;
      }
    }
    __syncwarp();
  }
}

// The backward's adjoint gathered in the kernel: per front f, entries off[f] .. off[f+1] of (s, loc, sg) —
// the level's source s (1 .. fs the free vector, fs + 1 .. fs + cs the child blocks) reads W at loc = i * n + j
// (i < j) times sg. `bycount`: counts indexed by front (expand_dims calls).
struct GatherIn {
  const int32_t* off; const int32_t* s; const int32_t* loc; const real_t* sg; int fs; int cs; real_t* dfree;
  real_t* dchild; int bycount;
  int cbar_f; int ybar_f;   // Cbar, ybar unbatched (expand_dims of a cotangent the same for every configuration)
  // the update block out of shared memory: uuent > 0 = the shared triangle holds its
  // first uuent entries (the rows below the update boundary) only; ypre[(b steps + step) 2 u_max + which u_max + u]
  // = an update row's product over the update columns, bwd_uu_kernel's; U x U entries of the output are Cbar's
  int uuent; const real_t* ypre;
};
// The shared backward's register floor: 0 = no launch bounds (the compiler's own choice, two 512-thread blocks
// an SM); F > 0 = __launch_bounds__(T, F * 512 / T), the same register cap at every width template.
// An explicit floor of 1 is never passed: it lets the compiler spend 128 registers a thread (one block an SM).
#ifndef PIVOTED_BWD_MINBLOCKS
#define PIVOTED_BWD_MINBLOCKS 0
#endif
#if PIVOTED_BWD_MINBLOCKS > 0
#define BWD_BOUNDS(T) __launch_bounds__(T, PIVOTED_BWD_MINBLOCKS * 512 / T)
#else
#define BWD_BOUNDS(T)
#endif
// Every step's update-row products over the update columns: a block a front, A = Cbar_UU - Cbar_UU^T
// staged in shared memory once (an odd leading dimension against bank conflicts), then the WARPS over the steps with
// no further barrier — lanes over the update rows u, each summing A[u][v] vec[nf + v] over v, the step's tau and c
// read from global memory (broadcast loads). ypre[(b steps + s) 2 u_max + which u_max + u]; zero for a step not
// taken. (The alternatives, a block a front with the steps in chunks through shared memory or a warp a
// (front, step) reading A from global memory, are both latency chains.)
constexpr int BUU_THREADS = 256, BUU_WARPS = BUU_THREADS / 32;
__host__ __device__ inline size_t buu_smem_bytes(int u_max) { return sizeof(real_t) * size_t(u_max | 1) * u_max; }
__global__ void __launch_bounds__(BUU_THREADS) bwd_uu_kernel(const real_t* Cbar, const int32_t* rec_kr,
    const real_t* rec_tau, const real_t* rec_c, const int32_t* counts, int nfront, int n, int steps, int d_max,
    int u_max, int bycount, int cbar_f, real_t* ypre) {
  extern __shared__ __align__(16) char base[];
  real_t* A = reinterpret_cast<real_t*>(base);
  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, cix = bycount ? b % nfront : b;
  const int nf = counts[3 * cix], nu = counts[3 * cix + 2], cd = d_max + u_max, ld = nu | 1;
  if (nu == 0) return;
  const real_t* C = Cbar + size_t(cbar_f ? b % nfront : b) * cd * cd + size_t(d_max) * cd + d_max;
  for (int e = t; e < nu * nu; e += BUU_THREADS) {
    const int u = e / nu, v = e - u * nu;
    A[u * ld + v] = C[size_t(u) * cd + v] - C[size_t(v) * cd + u];
  }
  __syncthreads();
  for (int st = warp; st < steps; st += BUU_WARPS) {
    const size_t row = size_t(b) * steps + st;
    real_t* out = ypre + row * 2 * u_max;
    const bool taken = rec_kr[row * 2] >= 0;
    const real_t* tv = rec_tau + row * n + nf;
    const real_t* cv = rec_c + row * n + nf;
    for (int u = lane; u < u_max; u += 32) {
      real_t yt = 0.0, yc = 0.0;
      if (taken && u < nu) {
        const real_t* a = A + u * ld;
        for (int v = 0; v < nu; ++v) { const real_t av = a[v]; yt += av * tv[v]; yc += av * cv[v]; }
      }
      out[u] = yt; out[u_max + u] = yc;
    }
  }
}

#if !defined(PIVOTED_REAL_FLOAT)
// The same products on the fp64 tensor cores: a block a front, A = Cbar_UU - Cbar_UU^T staged in shared
// memory with nu padded to a multiple of 8 (p), the steps in chunks of BUU_TC_STEPS with their tau_U and c_U staged
// (zero for a step not taken, a column past nu, a step past the last), a warp an 8 x 8 tile (u, s) of Y_tau or Y_c =
// A [tau_U | c_U] by m8n8k4 DMMA, the k loop over p in fours — the B operand is the staged chunk read column-major
// (B(k, s) = V[s][k]); the tile goes through a per-warp scratch to ypre. Rows past nu and steps not taken are never
// read by the backward and are left as they are.
constexpr int BUU_TC_STEPS = 32;
__host__ __device__ inline int buu_pad(int nu) { return (nu + 7) / 8 * 8; }
__host__ __device__ inline size_t buu_tc_smem_bytes(int u_max) {
  const size_t p = buu_pad(u_max);
  return sizeof(double) * (p * p + 2 * size_t(BUU_TC_STEPS) * p + (BUU_THREADS / 32) * 64) + 32;
}
__global__ void __launch_bounds__(BUU_THREADS) bwd_uu_tc_kernel(const double* Cbar, const int32_t* rec_kr,
    const double* rec_tau, const double* rec_c, const int32_t* counts, int nfront, int n, int steps, int d_max,
    int u_max, int bycount, int cbar_f, double* ypre) {
  extern __shared__ __align__(32) char base[];
  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, cix = bycount ? b % nfront : b;
  const int nf = counts[3 * cix], nu = counts[3 * cix + 2], cd = d_max + u_max;
  if (nu == 0) return;
  const int p = buu_pad(nu);
  double* As = reinterpret_cast<double*>((reinterpret_cast<size_t>(base) + 31) & ~size_t(31));
  double* Ts = As + size_t(p) * p;
  double* Cs = Ts + size_t(BUU_TC_STEPS) * p;
  double* tile = Cs + size_t(BUU_TC_STEPS) * p + warp * 64;
  const double* C = Cbar + size_t(cbar_f ? b % nfront : b) * cd * cd + size_t(d_max) * cd + d_max;
  for (int e = t; e < p * p; e += BUU_THREADS) {
    const int u = e / p, v = e - u * p;
    As[e] = (u < nu && v < nu) ? C[size_t(u) * cd + v] - C[size_t(v) * cd + u] : 0.0;
  }
  using namespace nvcuda;
  const int tiles_u = p / 8, tiles_s = BUU_TC_STEPS / 8;
  for (int s0 = 0; s0 < steps; s0 += BUU_TC_STEPS) {
    const int ns = steps - s0 < BUU_TC_STEPS ? steps - s0 : BUU_TC_STEPS;
    __syncthreads();                      // the previous chunk's tiles are done with Ts and Cs
    for (int e = t; e < BUU_TC_STEPS * p; e += BUU_THREADS) {
      const int sq = e / p, v = e - sq * p;
      double tv = 0.0, cv = 0.0;
      if (sq < ns && v < nu) {
        const size_t row = size_t(b) * steps + s0 + sq;
        if (rec_kr[row * 2] >= 0) { tv = rec_tau[row * n + nf + v]; cv = rec_c[row * n + nf + v]; }
      }
      Ts[e] = tv; Cs[e] = cv;
    }
    __syncthreads();
    for (int tl = warp; tl < 2 * tiles_u * tiles_s; tl += BUU_THREADS / 32) {
      const int which = tl / (tiles_u * tiles_s), r = tl - which * tiles_u * tiles_s;
      const int ti = r / tiles_s, tj = r - ti * tiles_s;
      const double* V = which ? Cs : Ts;
      wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc;
      wmma::fill_fragment(acc, 0.0);
      for (int k = 0; k < p; k += 4) {
        wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bf;
        wmma::load_matrix_sync(a, As + size_t(ti) * 8 * p + k, p);
        wmma::load_matrix_sync(bf, V + size_t(tj) * 8 * p + k, p);
        wmma::mma_sync(acc, a, bf, acc);
      }
      wmma::store_matrix_sync(tile, acc, 8, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 64; e += 32) {
        const int u = ti * 8 + e / 8, sq = tj * 8 + e % 8;
        if (u < nu && sq < ns) ypre[(size_t(b) * steps + s0 + sq) * 2 * u_max + which * u_max + u] = tile[e];
      }
      __syncwarp();
    }
  }
}
#endif
// the producer: the tensor-core kernel where its shared memory fits (the double build), else the scalar one
static void bwd_uu_launch(size_t total, cudaStream_t stream, const real_t* Cbar, const int32_t* rec_kr,
    const real_t* rec_tau, const real_t* rec_c, const int32_t* counts, int nfront, int n, int steps, int d_max,
    int u_max, int bycount, int cbar_f, real_t* ypre) {
#if !defined(PIVOTED_REAL_FLOAT)
  if (buu_tc_smem_bytes(u_max) <= size_t(g_smem_limit)) {
    bwd_uu_tc_kernel<<<total, BUU_THREADS, buu_tc_smem_bytes(u_max), stream>>>(Cbar, rec_kr, rec_tau, rec_c, counts,
        nfront, n, steps, d_max, u_max, bycount, cbar_f, ypre);
    return;
  }
#endif
  bwd_uu_kernel<<<total, BUU_THREADS, buu_smem_bytes(u_max), stream>>>(Cbar, rec_kr, rec_tau, rec_c, counts, nfront, n,
      steps, d_max, u_max, bycount, cbar_f, ypre);
}
#ifndef PIVOTED_BWD_WORK
#define PIVOTED_BWD_WORK 0  // 1: skip inactive rows; 2: warp row products; 4: two-stage warp reduction; 8: live-column list; 16: G threads a row
#endif
template <int T>
__global__ void BWD_BOUNDS(T)
front_bwd_kernel(const int8_t* alive_out, const int32_t* rec_kr, const real_t* rec_a,
                                 const real_t* rec_c, const real_t* rec_tau, const real_t* Cbar,
                                 const real_t* ybar, const int32_t* counts, int nfront, int n, int steps,
                                 int d_max, int u_max, const GatherIn g) {
  constexpr int THREADS = T, WINDOW = T / 32;
  extern __shared__ __align__(16) char base[];
  Smem s = carve<T>(base, n, (PIVOTED_BWD_WORK & 8) ? 1 : 0, 0, 0, g.uuent);   // P holds W; c, tau hold the two products; PT the step's tau, c (bit 8)
  const int b = blockIdx.x, f = b % nfront, t = threadIdx.x;
  const size_t vec = size_t(b) * n;
  const bool uu_out = g.uuent > 0;                       // U x U not in shared memory
  for (int i = t; i < n; i += THREADS) s.alive[i] = alive_out[vec + i] != 0;
  for (int e = t; e < (uu_out ? g.uuent : n * (n - 1) / 2); e += THREADS) s.P[e] = 0.0;
  __syncthreads();
  // the contribution's adjoint onto the upper-triangle parameters: p_ij is F[i, j] = p, F[j, i] = -p, and the
  // valid rows' slots increase with the row, so the pair r < q is the pair rowslot[r] < rowslot[q]
  const int cd = d_max + u_max;
  int* rowslot = s.slot;
  const int cix = g.bycount ? f : b;
  const int nf0 = counts[3 * cix], nu0 = counts[3 * cix + 2];   // this front's update boundary and update rows
  contribution_rows<T>(s.alive, n, nf0, counts[3 * cix + 1], nu0, d_max, u_max, rowslot, s.misc);
  const size_t cmat = size_t(g.cbar_f ? f : b) * cd * cd;
  for (int e = t; e < cd * cd; e += THREADS) {
    const int r = e / cd, q = e - r * cd;
    const int s1 = rowslot[r], s2 = rowslot[q];
    if (r < q && s1 >= 0 && s2 >= 0 && !(uu_out && s1 >= nf0))
      s.P[pidx(s1, s2, n)] = Cbar[cmat + e];   // the cotangent's upper entry alone (the gather writes no lower one)
  }
  // W's entry (i < j): the shared triangle, or Cbar's update block for a U x U pair held out of it
  auto wval = [&](int i, int j) -> real_t {
    if (uu_out && i >= nf0) {
      if (j - nf0 >= nu0) return 0.0;
      return Cbar[cmat + size_t(d_max + i - nf0) * cd + d_max + j - nf0] - Cbar[cmat + size_t(d_max + j - nf0) * cd + d_max + i - nf0];
    }
    return s.P[pidx(i, j, n)];
  };
  __syncthreads();
  const real_t yb = ybar[g.ybar_f ? f : b];
  for (int step = steps - 1; step >= 0; --step) {
    const int k = rec_kr[(size_t(b) * steps + step) * 2], r = rec_kr[(size_t(b) * steps + step) * 2 + 1];
    if (k < 0) continue;
    const real_t a = rec_a[size_t(b) * steps + step];
    const real_t* c = rec_c + (size_t(b) * steps + step) * n;
    const real_t* tau = rec_tau + (size_t(b) * steps + step) * n;
    // the rest before this step: alive now, k and r excluded (they come back alive below)
#if PIVOTED_BWD_WORK & 8
    {   // the step's vectors in shared memory, the alive columns listed (their tau and c are the only nonzero ones)
      real_t* tsh = s.PT; real_t* csh = s.PT + n;
      for (int j = t; j < n; j += THREADS) { tsh[j] = tau[j]; csh[j] = c[j]; }
      if (t < 32) {
        int found = 0, fd = 0;
        for (int base0 = 0; base0 < n; base0 += 32) {
          const int j = base0 + t;
          const bool ok = j < n && s.alive[j];
          const unsigned msk = __ballot_sync(FULL, ok);
          if (ok) s.ulist[found + __popc(msk & ((1u << t) - 1u))] = j;
          found += __popc(msk);
          fd += __popc(__ballot_sync(FULL, ok && j < nf0));
        }
        if (t == 0) { s.misc[5] = found; s.misc[6] = fd; }
      }
      __syncthreads();
      const int na = s.misc[5], nd = s.misc[6];     // the live columns, and those below the update boundary
      const real_t* yp = uu_out ? g.ypre + (size_t(b) * steps + step) * 2 * u_max : nullptr;
#if PIVOTED_BWD_WORK & 16
      // G threads a row: the largest power of two <= THREADS / n, at most 8 — a group strides the live
      // columns and sums by shuffles; the rows' loop is uniform over the block so every lane reaches the shuffles
      const int G = THREADS >= 8 * n ? 8 : THREADS >= 4 * n ? 4 : THREADS >= 2 * n ? 2 : 1;
      const int rows = THREADS / G, part = t % G;
      for (int base0 = 0; base0 < n; base0 += rows) {
        const int i = base0 + t / G;
        const bool ok = i < n && s.alive[i];
        const bool iu = uu_out && i >= nf0 && ok;      // an update row: the D columns here, the U columns from ypre
        real_t y1 = iu && part == 0 ? yp[i - nf0] : 0.0, y2 = iu && part == 0 ? yp[u_max + i - nf0] : 0.0;
        if (ok)
          for (int q = part; q < (iu ? nd : na); q += G) {
            const int j = s.ulist[q];
            const real_t w = get(s.P, i, j, n);
            y1 += w * tsh[j]; y2 += w * csh[j];
          }
        for (int o = G >> 1; o; o >>= 1) {
          y1 += __shfl_xor_sync(FULL, y1, o);
          y2 += __shfl_xor_sync(FULL, y2, o);
        }
        if (i < n && part == 0) { s.c[i] = ok ? -y1 : 0.0; s.tau[i] = ok ? y2 : 0.0; }
      }
#else
      for (int i = t; i < n; i += THREADS) {
        if (!s.alive[i]) { s.c[i] = 0.0; s.tau[i] = 0.0; continue; }
        const bool iu = uu_out && i >= nf0;            // an update row: the D columns here, the U columns from ypre
        real_t y1 = iu ? yp[i - nf0] : 0.0, y2 = iu ? yp[u_max + i - nf0] : 0.0;
        const int qe = iu ? nd : na;
        for (int q = 0; q < qe; ++q) {
          const int j = s.ulist[q];
          const real_t w = get(s.P, i, j, n);
          y1 += w * tsh[j]; y2 += w * csh[j];
        }
        s.c[i] = -y1; s.tau[i] = y2;
      }
#endif
    }
#elif PIVOTED_BWD_WORK & 2
    const int lane = t % 32, warp = t / 32;
    for (int i = warp; i < n; i += THREADS / 32) {
      real_t y1 = 0.0, y2 = 0.0;
      if (s.alive[i]) {
        for (int j = lane; j < n; j += 32) {
          const real_t w = get(s.P, i, j, n);
          y1 += w * tau[j]; y2 += w * c[j];
        }
        for (int off = 16; off; off >>= 1) {
          y1 += __shfl_down_sync(FULL, y1, off);
          y2 += __shfl_down_sync(FULL, y2, off);
        }
      }
      if (lane == 0) { s.c[i] = -y1; s.tau[i] = y2; }
    }
#else
    for (int i = t; i < n; i += THREADS) {
      real_t y1 = 0.0, y2 = 0.0;
#if PIVOTED_BWD_WORK & 1
      if (!s.alive[i]) { s.c[i] = 0.0; s.tau[i] = 0.0; continue; }
#endif
      for (int j = 0; j < n; ++j) {
        real_t w = get(s.P, i, j, n);
        y1 += w * tau[j]; y2 += w * c[j];
      }
      s.c[i] = s.alive[i] ? -y1 : 0.0;     // cbar
      s.tau[i] = s.alive[i] ? y2 : 0.0;    // taubar
    }
#endif
    __syncthreads();
    // abar from tau = F[k, rest] / a and from log|a|
    real_t part = 0.0;
    for (int j = t; j < n; j += THREADS) part += s.tau[j] * tau[j];
    real_t tot;
#if PIVOTED_BWD_WORK & 4
    for (int off = 16; off; off >>= 1) part += __shfl_down_sync(FULL, part, off);
    if (t % 32 == 0) s.red_v[t / 32] = part;
    __syncthreads();
    // Every warp reads the same partials before the following barrier releases the scratch.
    tot = t % 32 < THREADS / 32 ? s.red_v[t % 32] : real_t(0);
    for (int off = 16; off; off >>= 1) tot += __shfl_down_sync(FULL, tot, off);
    tot = __shfl_sync(FULL, tot, 0);
#else
    s.red_v[t] = part; __syncthreads();
    for (int h = THREADS / 2; h > 0; h >>= 1) { if (t < h) s.red_v[t] += s.red_v[t + h]; __syncthreads(); }
    tot = s.red_v[0];
#endif
    __syncthreads();
    const real_t abar = -tot / a + yb / a;
    for (int i = t; i < n; i += THREADS) {
      if (!s.alive[i]) continue;
      // c_i = F[i, r]
      if (i < r) s.P[pidx(i, r, n)] += s.c[i]; else s.P[pidx(r, i, n)] -= s.c[i];
      // tau_i = F[k, i] / a
      if (k < i) s.P[pidx(k, i, n)] += s.tau[i] / a; else s.P[pidx(i, k, n)] -= s.tau[i] / a;
    }
    __syncthreads();
    if (t == 0) {
      if (k < r) s.P[pidx(k, r, n)] += abar; else s.P[pidx(r, k, n)] -= abar;
      s.alive[k] = 1; s.alive[r] = 1;
    }
    __syncthreads();
  }
  // the adjoint straight to the sources: no W0
  __syncthreads();
  const size_t cfg = size_t(b) / nfront;
  for (int e = g.off[f] + t; e < g.off[f + 1]; e += THREADS) {
    const int src = g.s[e], loc = g.loc[e], i = loc / n, j = loc - (loc / n) * n;
    const real_t v = wval(i, j) * g.sg[e];
    if (src <= g.fs) g.dfree[cfg * g.fs + src - 1] = v; else g.dchild[cfg * g.cs + src - 1 - g.fs] = v;
  }
}

// ---------------------------------------------------------------------------------------------------------
// The one-warp kernel's bound (a warp a front up to WARP_MAXN slots) and its fronts a block.
constexpr int WARP_MAXN = 96;
constexpr int WPB = PIVOTED_WPB;

// a warp's share, rounded to 16 bytes: 7n + 4 ints is odd for odd n and would misalign the next warp's doubles
#ifndef PIVOTED_WARP_COMPACT
#define PIVOTED_WARP_COMPACT 0   // positive: remove this many matrix-capacity slots, with a full-capacity fallback
#endif
__host__ __device__ inline size_t ct_offset(int n, int capacity = 0) {   // the (c, tau) pairs' offset in reals: 16-byte aligned
  const int cap = capacity > 0 ? capacity : n;
  const size_t o = size_t(cap) * (cap - 1) / 2 + 2 * size_t(n) + size_t(n) / 2 + 1;
  return (o * sizeof(real_t) + 15) / 16 * 16 / sizeof(real_t);
}
__host__ __device__ inline size_t fused_bytes(int n, int capacity = 0) {   // only the triangle shrinks; logical arrays stay n-wide
  const size_t b = sizeof(real_t) * (ct_offset(n, capacity) + 2 * size_t(n)) + sizeof(int) * (7 * size_t(n) + 4);
  return (b + 15) & ~size_t(15);
}

// The warp kernel with the front's ASSEMBLY inside: no front matrix in global memory. A slot's alive flag and
// mode come from `asrc` / `msrc` (-1 alive, -2 dead, >= 0 an occupation bit; <= -3 the child row -3-code of the
// previous level's contribution, its validity and mode); every upper-triangle target (slot_i < slot_j) of the
// front is const + sum_k sgn * source, a source >= 0 a free entry, <= -2 the child block entry -2-code, -1 none.
// Unbatched inputs (the tables) arrive with a leading dimension 1 (vmap's expand_dims) and are read at stride 0.
// A step: the candidate columns in slot order, a whole warp scanning each (lanes along the rows), the first that
// passes is the pivot column with its largest candidate row (the lowest on ties); the pivot's multipliers by one
// reciprocal; the rank-2 sweep over the packed triangle, lanes striding by 32. The pivots' logs are summed in one
// reduction at the front's end, and the contribution block is written through precomputed compact indices.
__global__ void __launch_bounds__(32 * WPB, FUSED_MIN_BLOCKS) front_fwd_fused(const FusedIn fin,
                                const int32_t* counts, int nfront, int n, int steps, real_t u, int root,
                                int record, int d_max, int u_max, size_t total, real_t* Cout, int8_t* cvalid,
                                int32_t* cmode, int32_t* count, int8_t* alive_out, real_t* logabs, real_t* sign,
                                int32_t* pivots, real_t* growth, int32_t* rec_kr, real_t* rec_a, real_t* rec_c,
                                real_t* rec_tau, double* cu, double* tu, int ucp,
                                int matrix_capacity, int fallback_only) {
  extern __shared__ __align__(16) char base[];
  const int lane = threadIdx.x % 32, wib = threadIdx.x / 32;
  const size_t b = size_t(blockIdx.x) * WPB + wib;
  if (b >= total) return;
#if PIVOTED_WARP_COMPACT > 0
  // Every lane of a warp sees one front's count. This kernel has no block barriers:
  // successful and inactive warps may exit even when a neighbour needs the fallback.
  if (fallback_only && count[b] >= 0) return;
#endif
  real_t* P = reinterpret_cast<real_t*>(base + wib * fused_bytes(n, matrix_capacity));
  real_t* cv = P + size_t(matrix_capacity) * (matrix_capacity - 1) / 2;
  real_t* tv = cv + n;
  real_t* pa = tv + n;                  // the pivots' a, a step (the log sum at the end)
  real_t* ct = P + ct_offset(n, matrix_capacity);  // the logical vectors/maps/records retain n slots
  int* q = reinterpret_cast<int*>(ct + 2 * n);
  int *alive = q, *cand = q + n, *failed = q + 2 * n, *cidx = q + 3 * n, *slot = q + 4 * n, *rowslot = q + 5 * n;
  int* smode = q + 6 * n;
  const size_t cfg = b / nfront;
  const int f = int(b % nfront);
  const size_t vec = b * n;
  const int nf = counts[3 * f], nu_f = counts[3 * f + 2];
  const real_t* fr = fin.free_ + cfg * fin.fstride;
  const real_t* cc = fin.childC + cfg * fin.cstride;
  const int8_t* cvv = fin.childV + cfg * fin.vstride;
  const int32_t* cmm = fin.childM + cfg * fin.vstride;
  const int8_t* oc = fin.occ + cfg * fin.ostride;
  // the slots' alive flags and modes, then the compaction
  for (int i = lane; i < n; i += 32) {
    const int code = fin.asrc[f * n + i], mc = fin.msrc[f * n + i];
    int a_;
    if (code == -1) a_ = 1;
    else if (code == -2) a_ = 0;
    else if (code >= 0) a_ = oc[code] != 0;
    else a_ = cvv[-3 - code] != 0;
    rowslot[i] = a_;
    smode[i] = mc >= -1 ? mc : cmm[-3 - mc];
  }
  __syncwarp();
  int m = 0;
  for (int base0 = 0; base0 < n; base0 += 32) {
    int i = base0 + lane;
    bool ok = i < n && rowslot[i] != 0;
    unsigned msk = __ballot_sync(FULL, ok);
    int c = m + __popc(msk & ((1u << lane) - 1u));
    if (i < n) cidx[i] = ok ? c : -1;
    if (ok) slot[c] = i;
    m += __popc(msk);
  }
  __syncwarp();
#if PIVOTED_WARP_COMPACT > 0
  if (m > matrix_capacity) {
    if (lane == 0) count[b] = -1;  // no matrix access or output record has occurred; the second launch replays this front
    return;
  }
#endif
  const int nfc = m - nu_f;           // the update rows are the last nu_f compact rows (always alive)
  const bool defer = nu_f > 1;   // U x U waits for one Schur complement at the end
  const int total_e = m * (m - 1) / 2;
  const int live_e = defer ? pidx(nfc, nfc + 1, m) : total_e;   // the packed triangle's U x U block is its tail
  for (int c = lane; c < m; c += 32) { alive[c] = 1; cand[c] = slot[c] < nf; failed[c] = 0; }
  for (int e = lane; e < m * (m - 1) / 2; e += 32) P[e] = 0.0;
  __syncwarp();
  assemble_pieces<true>(fin, f, fr, cc, cidx, lane, 32, [&](int ci, int cj, real_t v) { P[pidx(ci, cj, m)] += v; },
                        failed, rowslot);
  __syncwarp();
  for (int c = lane; c < m; c += 32) failed[c] = 0;     // the assembly's scratch
  __syncwarp();
  double la = 0.0, sg = 1.0; real_t big = 0.0;
  int step = 0;
  for (; step < steps; ++step) {
    int k = -1, r = -1;
    // candidates in slot order, a whole warp scanning each column (lanes along the rows), until one passes
    for (int kk = 0; k < 0 && kk < m; ++kk) {
      if (!cand[kk]) continue;
      real_t mx = 0.0, bv = 0.0; int br = -1;
      for (int i = lane; i < m; i += 32) {
        if (!alive[i] || i == kk) continue;
        real_t v = fabs(get(P, i, kk, m));
        mx = fmax(mx, v);
        if (cand[i] && v > bv) { bv = v; br = i; }
      }
      // the hardware reductions: the maximum, and the largest candidate's lowest row on ties
      mx = warp_max_nonneg(mx);
      const real_t bmax = warp_max_nonneg(bv);
      br = warp_argmin_at(br >= 0 && bv == bmax, br);
      bv = bmax;
      big = fmax(big, mx);
      if (br >= 0 && bv > 0.0 && (root || bv >= u * mx)) { k = kk; r = br; break; }
    }
    if (k < 0) break;      // no pair passes: the rest is the parent's
    const real_t a = get(P, k, r, m);
    const real_t inv = real_t(1) / a;
    for (int i = lane; i < m; i += 32) {
      bool rest = alive[i] && i != k && i != r;
      cv[i] = rest ? get(P, i, r, m) : 0.0;
      tv[i] = rest ? get(P, k, i, m) * inv : 0.0;
      failed[i] = 0;
    }
    __syncwarp();
    const int sk = slot[k], sr = slot[r];
    if (lane == 0) {
      alive[k] = 0; alive[r] = 0; cand[k] = 0; cand[r] = 0;
      pa[step] = a;
      pivots[(b * steps + step) * 2] = smode[sk];
      pivots[(b * steps + step) * 2 + 1] = smode[sr];
      if (record) {
        rec_kr[(b * steps + step) * 2] = sk; rec_kr[(b * steps + step) * 2 + 1] = sr;
        rec_a[b * steps + step] = a;
      }
    }
    if (record)
      for (int i = lane; i < n; i += 32) {
        const int ci = cidx[i];
        rec_c[(b * steps + step) * n + i] = ci >= 0 ? cv[ci] : 0.0;
        rec_tau[(b * steps + step) * n + i] = ci >= 0 ? tv[ci] : 0.0;
      }
    if (defer)
      for (int i = nfc + lane; i < m; i += 32) {
        cu[(b * steps + step) * ucp + i - nfc] = cv[i];
        tu[(b * steps + step) * ucp + i - nfc] = tv[i];
      }
    {   // the rank-2 update over the packed triangle but its U x U tail, lanes striding by 32 with (i, j) kept
      int e = lane, i = 0, j = lane + 1;
      while (i < m - 1 && j >= m) { j = j - m + i + 2; ++i; }   // the lane's first entry: row i holds j = i+1..m-1
      for (; e < live_e; e += 32) {
        const real_t v = P[e] - (cv[i] * tv[j] - tv[i] * cv[j]);
        P[e] = v;
        j += 32;
        while (i < m - 1 && j >= m) { j = j - m + i + 2; ++i; }
      }
    }
    __syncwarp();
  }
  if (defer) {
    __syncwarp();
    schur_uu_packed(P, m, nfc, nu_f, step, cu + b * steps * ucp, tu + b * steps * ucp, ucp, lane, 32);
    __syncwarp();
  }
  for (int qq = step + lane; qq < steps; qq += 32) {
    pivots[(b * steps + qq) * 2] = -1; pivots[(b * steps + qq) * 2 + 1] = -1;
    if (record) { rec_kr[(b * steps + qq) * 2] = -1; rec_kr[(b * steps + qq) * 2 + 1] = -1; }
  }
  {          // the pivots' logs, a lane a few, summed once
    double ls = 0.0; int neg = 0;
    for (int s2 = lane; s2 < step; s2 += 32) { const real_t av = pa[s2]; ls += log(fabs(double(av))); neg += av < 0; }
    for (int off = 16; off; off >>= 1) { ls += __shfl_down_sync(FULL, ls, off); neg += __shfl_down_sync(FULL, neg, off); }
    la = __shfl_sync(FULL, ls, 0); sg = (__shfl_sync(FULL, neg, 0) & 1) ? -1.0 : 1.0;
  }
  if (lane == 0) { logabs[b] = la; sign[b] = sg; growth[b] = big; }
  // alive by slot, then the contribution rows (the block kernel's rule), into `failed` and `rowslot`
  int* alive_slot = failed;
  __syncwarp();
  for (int i = lane; i < n; i += 32) {
    const int ci = cidx[i];
    const int a_ = ci >= 0 ? alive[ci] : 0;
    alive_out[vec + i] = a_;
  }
  __syncwarp();
  for (int i = lane; i < n; i += 32) { const int ci = cidx[i]; alive_slot[i] = ci >= 0 ? alive[ci] : 0; }
  __syncwarp();
  const int sh = counts[3 * f + 1], nu = counts[3 * f + 2];
  int left = 0;
  for (int base0 = 0; base0 < nf; base0 += 32) {
    int i = base0 + lane;
    bool ok = i < nf && alive_slot[i];
    unsigned msk = __ballot_sync(FULL, ok);
    int c = left + __popc(msk & ((1u << lane) - 1u));
    if (ok && c < d_max) rowslot[c] = c < sh ? i : -1;
    left += __popc(msk);
  }
  for (int c = (left < d_max ? left : d_max) + lane; c < d_max; c += 32) rowslot[c] = -1;
  for (int uu = lane; uu < u_max; uu += 32) rowslot[d_max + uu] = uu < nu ? nf + uu : -1;
  __syncwarp();
  if (lane == 0) count[b] = left;
  const int cd = d_max + u_max;
  const size_t cvec = b * cd, cmat = b * cd * cd;
  for (int c = lane; c < cd; c += 32) {
    const int sl = rowslot[c];
    cvalid[cvec + c] = sl >= 0;
    cmode[cvec + c] = sl >= 0 ? smode[sl] : -1;
  }
  {   // the valid rows compacted (into the scratch free now), then only their pairs by the width-L table (tabn = n here)
    int* vl = failed; int* vc = cand;
    const int L = compact_rows(cd, [&](int c) { const int sl = rowslot[c]; return sl >= 0 ? cidx[sl] : -1; }, vl, vc, lane, 32);
    const int E = L * (L - 1) / 2;
    const int T4 = ((E + 31) / 32 + 3) / 4 * 4;
    const uint32_t* tab = fin.ijtab + fin.ijoff[L] + lane * T4;
    for (int t = 0; lane + 32 * t < E; ++t) {
      const unsigned v = __ldg(tab + t);
      const int a = (v >> 16) >> 4, b = (v & 0xffffu) >> 4;
      Cout[cmat + size_t(vl[a]) * cd + vl[b]] = get(P, vc[a], vc[b], m);
    }
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------------------------------------
// Fronts wider than one block's shared memory (or any, when asked): the front lives in global memory, `A`, its
// compact alive part in full skew storage in a slab of n x n per front, A[i * n + j] = -A[j * n + i] exactly —
// every pair's update is one fixed expression (`pair_sum_p`, explicit roundings) for (min, max) and its negation for
// (max, min). The rows the elimination reads are cached in shared memory and kept up to date: a row joins the
// cache by paying the open panel's pending pairs; each pivot pair updates the cached rows at once and holds back
// its rank-2 update of the rest; every `pmax` pairs — or when the cache must make room, or at the end — the panel
// closes: one rank-2tp update of every uncached alive entry, the cached rows written back, the cache emptied.
// The pivot rule (the window of the lowest untested candidates, the threshold test, the delays), the records
// and the contribution block are the block kernel's. Assembly is front_fwd_fused's, from `fin`.
constexpr int GT = 512;
constexpr int GWINDOW = GT / 32;
// Tune only forward CTA scheduling. The logical search window, cache/panel sizing policy and backward's
// reduction tree remain those of the 512-thread baseline; fewer physical warps scan a window in waves.
#ifndef PIVOTED_GFWD_THREADS
#define PIVOTED_GFWD_THREADS 512
#endif
constexpr int GFT = PIVOTED_GFWD_THREADS;
// The global forward kernel's target blocks an SM: the host shrinks the panel and the row cache until GBLOCKS
// blocks' shared memory fit an SM (the launch bound's register floor is PIVOTED_GFWD_MINBLOCKS's, below). 1 = one
// block, as before.
#ifndef PIVOTED_GBLOCKS
#define PIVOTED_GBLOCKS 1
#endif
// The global forward kernel's update block under the tensor-core close: 1 = deferred — not gathered, not
// closed (tiles wholly inside skipped), written after the kernel by uu_tc_kernel from the children's blocks less
// the Schur complement of the recorded update rows' c and tau; 0 = gathered and closed with the rest, as before.
#ifndef PIVOTED_GUU
#define PIVOTED_GUU 0
#endif
static_assert(GFT == 128 || GFT == 256 || GFT == 512, "PIVOTED_GFWD_THREADS must be 128, 256 or 512");

__device__ __forceinline__ real_t pair_term(real_t a, real_t b, real_t c, real_t d) {   // a b - c d, fixed
  return __fma_rn(a, b, -__dmul_rn(c, d));
}

// sum_q c_q[i] tau_q[j] - tau_q[i] c_q[j] over the panel's tp pairs, in one fixed order, on the global kernel's planar
// panel: c_q[i] = ci[q], tau_q[i] = ti[q]
__device__ __forceinline__ real_t pair_sum_p(const real_t* ci, const real_t* ti, const real_t* cj, const real_t* tj,
                                             int tp) {
  real_t acc = pair_term(ci[0], tj[0], ti[0], cj[0]);
  for (int q = 1; q < tp; ++q) acc = RADD_RN(acc, pair_term(ci[q], tj[q], ti[q], cj[q]));
  return acc;
}
// The global kernel's panel close on the tensor cores (when n and the panel are multiples of 4; the scalar tiles
// otherwise): the panel's c (n x pmax) and tau (n x pmax) are planar in PT, so A -= C T^T - T C^T is two fp64 mma products
// (m8n8k4) over the 16 x 16 tiles of the live square — a tile's A block as the accumulator's initial value, the
// updated tile stored back, its negation stored column-major to the mirror tile — followed by the scalar form on the
// rows and columns past the last whole tile. Cached rows are written back by the caller afterwards, as before; the
// update block's Schur complement is not deferred on this path (the products do it at the tensor cores' rate).

// PIVOTED_GCLOSE_DMMA (0 by default; recipe.settings appends it to a float32 recipe for dmma=True): the float build's global close on the fp64
// tensor cores, every operand converted in its register (no staging); the double build is unchanged by it
#ifndef PIVOTED_GCLOSE_DMMA
#define PIVOTED_GCLOSE_DMMA 0
#endif
#if PIVOTED_GCLOSE_DMMA
// d = a b + c for one m8n8k4 fp64 tile product, the fragments one register a lane (PTX ISA, mma.m8n8k4 .f64)
__device__ __forceinline__ void dmma_f64(double& d0, double& d1, double a, double b, double c0, double c1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
               : "=d"(d0), "=d"(d1) : "d"(a), "d"(b), "d"(c0), "d"(c1));
}
#endif

// The global kernels' planar panel: row i's pmax coefficients at PT[i * ps + q] with ps = pmax + this pad.
// At ps = pmax = 8 doubles a warp's 32 rows sit on two banks, so every per-lane read of a row's panel (the row
// loads' and the scalar closes' pair sums, the pivot vectors' writes) is a 16-way conflict; ps = 10 makes it 2-way
// and keeps the tensor-core close's 16-byte leading dimension and 32-byte tile rows. The same values bit for bit.
#ifndef PIVOTED_GPT_PAD
#define PIVOTED_GPT_PAD 2
#endif
struct GSmem {
  real_t* PT; real_t* rows; real_t* c; real_t* tau; real_t* red_v; real_t* wkey; real_t* wm;
  int* alive; int* cand; int* failed; int* red_i; int* win; int* wr; int* misc;   // misc: nwin, k, r, m, ncached
  int* cidx; int* slot; int* smode; int* rowof; int* cached;   // rowof: compact -> cache slot; cached: slot -> compact
};

__host__ __device__ inline int gps(int pmax) { return pmax + PIVOTED_GPT_PAD; }   // the planar panel's row stride
__device__ GSmem gcarve(char* base, int n, int pmax, int ucap) {
  GSmem s;
  s.PT = reinterpret_cast<real_t*>(base);
  s.rows = s.PT + 2 * size_t(gps(pmax)) * n;
  s.c = s.rows + size_t(ucap) * n; s.tau = s.c + n; s.red_v = s.tau + n; s.wkey = s.red_v + GT;
  s.wm = s.wkey + GWINDOW;
  int* q = reinterpret_cast<int*>(s.wm + GWINDOW);
  s.alive = q; s.cand = q + n; s.failed = q + 2 * n; s.red_i = q + 3 * n;
  s.win = s.red_i + GT; s.wr = s.win + GWINDOW; s.misc = s.wr + GWINDOW;
  s.cidx = s.misc + 8; s.slot = s.cidx + n; s.smode = s.slot + n; s.rowof = s.smode + n; s.cached = s.rowof + n;
  return s;
}

size_t gsmem_bytes(int n, int pmax, int ucap, int threads = GT) {
  size_t b = sizeof(real_t) * (2 * size_t(gps(pmax)) * n + size_t(ucap) * n + 2 * n + GT + 2 * GWINDOW)
           + sizeof(int) * (7 * size_t(n) + GT + 2 * GWINDOW + 8 + ucap);
  return b;
}

// independently skip dead cache-load columns (1), dead update columns (2),
// and close tiles with no alive uncached pair (4). Zero restores the previous walks.
#ifndef PIVOTED_GLIVE
#define PIVOTED_GLIVE 0
#endif

// row k into the cache, its entries brought up to date: from a cached row's copy (exactly negated) or from A
// with the open panel's tp pairs applied
__device__ void gload(GSmem& s, const real_t* A, int m, int n, int pmax, int tp, int k) {
  const int sl = s.misc[4];
  real_t* rk = s.rows + size_t(sl) * n;
  const int ps = gps(pmax);
  const real_t* ck = s.PT + size_t(k) * ps;
  const real_t* tk = s.PT + size_t(ps) * n + size_t(k) * ps;
  for (int j = threadIdx.x; j < m; j += GFT) {
    real_t v;
    const int oj = s.rowof[j];
    if (j == k || ((PIVOTED_GLIVE & 1) && !s.alive[j])) v = 0.0;
    else if (oj >= 0) v = -s.rows[size_t(oj) * n + k];
    else {
      v = A[size_t(k) * n + j];
      if (tp > 0) {
        const real_t* cj = s.PT + size_t(j) * ps;
        const real_t* tj = s.PT + size_t(ps) * n + size_t(j) * ps;
        v = k < j ? RSUB_RN(v, pair_sum_p(ck, tk, cj, tj, tp)) : RADD_RN(v, pair_sum_p(cj, tj, ck, tk, tp));
      }
    }
    rk[j] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) { s.rowof[k] = sl; s.cached[sl] = k; s.misc[4] = sl + 1; }
  __syncthreads();
}

// close the panel: the held-back update of every uncached alive entry (tp > 0), the cached rows written back,
// the cache emptied. Tiles of 8 rows x 32 columns, a lane holding its column's pairs in registers.
__device__ void gclose(GSmem& s, real_t* A, int m, int n, int pmax, int tp, int nfc) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int ps = gps(pmax);                             // the planar row stride
  const real_t* Cp = s.PT;                              // c, n rows of stride ps
  const real_t* Tp = s.PT + size_t(ps) * n;             // tau, n rows of stride ps
  // the tensor-core close needs 32-byte-aligned tiles: the slab's row stride a multiple of 4 doubles, the panel of 4
  const bool tc = (n % 4 == 0) && (pmax % 4 == 0);
  if (tp > 0 && tc) {
    // the stale columns of the panel (q >= tp) and the rows past m to the tile edge: zero, so the products see them
    const int mf = m / 16 * 16, tiles = mf / 16;      // whole tiles only; the tail below is scalar
    for (int idx = threadIdx.x; idx < m * pmax; idx += GFT) {
      const int r = idx / pmax, q = idx % pmax;
      if (q >= tp) { s.PT[size_t(r) * ps + q] = 0.0; s.PT[size_t(ps) * n + size_t(r) * ps + q] = 0.0; }
    }
    if (PIVOTED_GLIVE & 4) {
      // Each 16-row block needs a close only if it has an alive, uncached row.
      // red_i is idle during the close; the existing barrier publishes these masks.
      for (int base0 = 0; base0 < m; base0 += GFT) {
        const int r = base0 + threadIdx.x;
        const unsigned live = __ballot_sync(FULL, r < m && s.alive[r] && s.rowof[r] < 0);
        if (lane == 0 && r < m) {
          s.red_i[r / 16] = (live & 0xffffu) != 0;
          if (r + 16 < m) s.red_i[r / 16 + 1] = (live >> 16) != 0;
        }
      }
    }
    __syncthreads();
    using namespace nvcuda;
    for (int tile = warp; tile < tiles * (tiles + 1) / 2; tile += GFT / 32) {
      int ti = 0, tj = tile;
      while (tj >= tiles - ti) { tj -= tiles - ti; ++ti; }
      if (PIVOTED_GUU && nfc < m && ti * 16 >= nfc) continue;   // wholly in the deferred U x U
      tj += ti;
      if ((PIVOTED_GLIVE & 4) && (!s.red_i[ti] || !s.red_i[tj])) continue;
#if defined(PIVOTED_REAL_FLOAT)
#if PIVOTED_GCLOSE_DMMA
      // PIVOTED_GCLOSE_DMMA (the float build; 0 by default): the close on the fp64 tensor cores with no staging —
      // mma.sync.m8n8k4.f64 by inline PTX, whose fragments are one register each (a: row lane / 4, column lane % 4 of
      // the 8 x 4 operand; b: row lane % 4, column lane / 4 of the 4 x 8; c, d: row lane / 4, columns 2 (lane % 4),
      // + 1 of the 8 x 8), so every operand is a float read from global memory and converted in the register, the
      // tile's entries the accumulators, rounded to float on the store. The same products as the CUDA-core close
      // below, summed by the tensor core instead of the fma chain; the diagonal tile's strict upper triangle written
      // and mirrored, as below, so the slab stays skew to the bit.
      {
        const int g = lane >> 2, tg = lane & 3;
        double acc[2][2][2];
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int i = ti * 16 + p * 8 + g, j = tj * 16 + q * 8 + 2 * tg;
            acc[p][q][0] = A[size_t(i) * n + j]; acc[p][q][1] = A[size_t(i) * n + j + 1];
          }
        for (int k0 = 0; k0 < tp; k0 += 4) {      // the panel's stale columns (q >= tp) are zero: whole k-steps
          double am[2], at[2], bt[2], bc[2];
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            const int i = ti * 16 + p * 8 + g;
            am[p] = -double(Cp[size_t(i) * ps + k0 + tg]); at[p] = Tp[size_t(i) * ps + k0 + tg];
          }
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int j = tj * 16 + q * 8 + g;
            bt[q] = Tp[size_t(j) * ps + k0 + tg]; bc[q] = Cp[size_t(j) * ps + k0 + tg];
          }
#pragma unroll
          for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int q = 0; q < 2; ++q) {     // A += (-C) T^T, then A += T C^T
              dmma_f64(acc[p][q][0], acc[p][q][1], am[p], bt[q], acc[p][q][0], acc[p][q][1]);
              dmma_f64(acc[p][q][0], acc[p][q][1], at[p], bc[q], acc[p][q][0], acc[p][q][1]);
            }
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int i = ti * 16 + p * 8 + g, j = tj * 16 + q * 8 + 2 * tg + e;
              if (ti == tj && j <= i) continue;
              const real_t v = real_t(acc[p][q][e]);
              A[size_t(i) * n + j] = v;
              A[size_t(j) * n + i] = -v;
            }
      }
      __syncwarp();
#else
      // the float build: the close on the CUDA cores — a lane owns row r = lane / 2 of the tile and the eight
      // columns (lane % 2) * 8 .., its row's c and tau in registers, each element's pair sum accumulated in double
      // from the float products (the staged fp64 path's digits), the tile and its mirror written; the diagonal
      // tile's strict upper triangle only, mirrored, so the slab stays skew to the bit
      {
        const int r = lane / 2, c0 = (lane % 2) * 8, i = ti * 16 + r;
        const real_t* ci = Cp + size_t(i) * ps;
        const real_t* ti_ = Tp + size_t(i) * ps;
        double cr[GPANEL_MAX], tr[GPANEL_MAX];
#pragma unroll
        for (int q = 0; q < GPANEL_MAX; ++q) { cr[q] = q < tp ? double(ci[q]) : 0.0; tr[q] = q < tp ? double(ti_[q]) : 0.0; }
#pragma unroll
        for (int cc = 0; cc < 8; ++cc) {
          const int c = c0 + cc, j = tj * 16 + c;
          if (ti == tj && c <= r) continue;
          const real_t* cj = Cp + size_t(j) * ps;
          const real_t* tj_ = Tp + size_t(j) * ps;
          double acc = double(A[size_t(i) * n + j]);
          for (int q = 0; q < tp; ++q) acc -= cr[q] * double(tj_[q]) - tr[q] * double(cj[q]);
          const real_t v = real_t(acc);
          A[size_t(i) * n + j] = v;
          A[size_t(j) * n + i] = -v;
        }
      }
      __syncwarp();
#endif
#else
      double* Ap = A + size_t(ti * 16) * n + tj * 16;
      const double *Ci = Cp + size_t(ti * 16) * ps, *Ti = Tp + size_t(ti * 16) * ps;
      const double *Cj = Cp + size_t(tj * 16) * ps, *Tj = Tp + size_t(tj * 16) * ps;
      const int lda = n, pld = ps;
      wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc[2][2];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          wmma::load_matrix_sync(acc[p][q], Ap + size_t(p * 8) * lda + q * 8, lda, wmma::mem_row_major);
      for (int k0 = 0; k0 < pmax; k0 += 4) {
        wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bb[2];
        // A += (-C) T^T
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          wmma::load_matrix_sync(a[p], Ci + size_t(p * 8) * pld + k0, pld);
#pragma unroll
          for (int e = 0; e < a[p].num_elements; ++e) a[p].x[e] = -a[p].x[e];
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], Tj + size_t(q * 8) * pld + k0, pld);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
        // A += T C^T
#pragma unroll
        for (int p = 0; p < 2; ++p) wmma::load_matrix_sync(a[p], Ti + size_t(p * 8) * pld + k0, pld);
#pragma unroll
        for (int q = 0; q < 2; ++q) wmma::load_matrix_sync(bb[q], Cj + size_t(q * 8) * pld + k0, pld);
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) wmma::mma_sync(acc[p][q], a[p], bb[q], acc[p][q]);
      }
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          wmma::store_matrix_sync(Ap + size_t(p * 8) * lda + q * 8, acc[p][q], lda, wmma::mem_row_major);
      if (ti != tj) {     // the mirror tile: the negation, stored column-major = the transpose
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int e = 0; e < acc[p][q].num_elements; ++e) acc[p][q].x[e] = -acc[p][q].x[e];
            wmma::store_matrix_sync(A + size_t(tj * 16 + q * 8) * n + ti * 16 + p * 8, acc[p][q], n, wmma::mem_col_major);
          }
      }
#endif
    }
    __syncthreads();
    // the rows and columns past the last whole tile: the scalar form, both triangles
    const int rest = m - mf;
    if (rest > 0) {
      for (int e = threadIdx.x; e < m * rest; e += GFT) {
        const int j = mf + e % rest, i = e / rest;      // (i, j) with j in the tail; i < m
        if (i == j) continue;
        if (PIVOTED_GUU && i >= nfc && j >= nfc) continue;   // the deferred U x U (nfc = m unless deferred)
        if ((PIVOTED_GLIVE & 4) && (!s.alive[i] || !s.alive[j] || s.rowof[i] >= 0 || s.rowof[j] >= 0)) continue;
        const real_t *ci = Cp + size_t(i) * ps, *ti_ = Tp + size_t(i) * ps;
        const real_t *cj = Cp + size_t(j) * ps, *tj = Tp + size_t(j) * ps;
        const real_t v = i < j ? RSUB_RN(A[size_t(i) * n + j], pair_sum_p(ci, ti_, cj, tj, tp))
                               : RADD_RN(A[size_t(i) * n + j], pair_sum_p(cj, tj, ci, ti_, tp));
        A[size_t(i) * n + j] = v;
        if (i < mf) A[size_t(j) * n + i] = -v;     // (j, i) is in the tail's own range when j >= mf and i >= mf
      }
    }
  } else if (tp > 0) {
    const int nb = (m + 31) / 32, items = nb * ((m + 7) / 8);
    for (int it = warp; it < items; it += GFT / 32) {
      const int jb = it % nb, i0 = 8 * (it / nb);
      const int j = 32 * jb + lane;
      const bool colok = j < m && s.alive[j] && s.rowof[j] < 0;
      real_t cj[GPANEL_MAX], tj[GPANEL_MAX];
#pragma unroll
      for (int q = 0; q < GPANEL_MAX; ++q) {
        cj[q] = (colok && q < tp) ? Cp[size_t(j) * ps + q] : 0.0;
        tj[q] = (colok && q < tp) ? Tp[size_t(j) * ps + q] : 0.0;
      }
      const int i1 = min(i0 + 8, m);
      for (int i = i0; i < i1; ++i) {
        if (!s.alive[i] || s.rowof[i] >= 0) continue;      // the same for every lane
        if (!colok || j == i) continue;
        if (i >= nfc && j >= nfc) continue;                  // U x U: the deferred Schur complement's
        const real_t *ci = Cp + size_t(i) * ps, *ti_ = Tp + size_t(i) * ps;
        real_t* e = A + size_t(i) * n + j;
        *e = i < j ? RSUB_RN(*e, pair_sum_p(ci, ti_, cj, tj, tp)) : RADD_RN(*e, pair_sum_p(cj, tj, ci, ti_, tp));
      }
    }
  }
  __syncthreads();
  const int nc = s.misc[4];
  for (int sl = warp; sl < nc; sl += GFT / 32) {
    const int u = s.cached[sl];
    if (u < 0 || !s.alive[u] || tp == 0) continue;     // with no pair held back the copy is A's own
    const real_t* ru = s.rows + size_t(sl) * n;
    for (int j = lane; j < m; j += 32) {
      if (!s.alive[j] || j == u) continue;
      const real_t v = ru[j];
      A[size_t(u) * n + j] = v;
      if (s.rowof[j] < 0) A[size_t(j) * n + u] = -v;
    }
  }
  __syncthreads();
  for (int sl = threadIdx.x; sl < nc; sl += GFT) {
    const int u = s.cached[sl];
    if (u >= 0) s.rowof[u] = -1;
    s.cached[sl] = -1;
  }
  __syncthreads();
  if (threadIdx.x == 0) s.misc[4] = 0;
  __syncthreads();
}

// The global forward's register floor: PIVOTED_GBLOCKS splits the shared memory between blocks an SM and was
// also the launch bound's floor — an explicit floor of 1 lets the compiler spend 128 registers a thread
// (one 512-thread block an SM). PIVOTED_GFWD_MINBLOCKS: 0 = __launch_bounds__(GFT) alone, F > 0 = a floor of F.
#ifndef PIVOTED_GFWD_MINBLOCKS
#define PIVOTED_GFWD_MINBLOCKS 0
#endif
#if PIVOTED_GFWD_MINBLOCKS > 0
#define GFWD_BOUNDS __launch_bounds__(GFT, PIVOTED_GFWD_MINBLOCKS)
#else
#define GFWD_BOUNDS __launch_bounds__(GFT)
#endif
__global__ void GFWD_BOUNDS front_gfwd_kernel(
    const int32_t* counts, int nfront, int n, int steps, real_t u, int root, int record, int d_max, int u_max,
    real_t* Cout, int8_t* cvalid, int32_t* cmode, int32_t* count, int8_t* alive_out, real_t* logabs, real_t* sign,
    int32_t* pivots, real_t* growth, int32_t* rec_kr, real_t* rec_a, real_t* rec_c, real_t* rec_tau, double* cu,
    double* tu, int ucp, real_t* work, int pmax, int ucap, int search_window, const FusedIn fin,
    int guu_on) {
  constexpr int THREADS = GFT, WINDOW = GWINDOW;
  // Zero preserves the recorded window; a smaller positive value tries fewer columns,
  // with the same threshold and the same fallback over untested candidates.
  const int scan_window = search_window > 0 && search_window < WINDOW ? search_window : WINDOW;
  extern __shared__ __align__(32) char base[];
  GSmem s = gcarve(base, n, pmax, ucap);
  const int b = blockIdx.x, f = b % nfront, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const size_t vec = size_t(b) * n, cfg = size_t(b) / nfront;
  real_t* A = work + size_t(b) * n * n;
  const int nf = counts[3 * f], sh_ = counts[3 * f + 1], nu_ = counts[3 * f + 2];
  for (int i = t; i < n; i += THREADS) {
    const int code = fin.asrc[f * n + i], mc = fin.msrc[f * n + i];
    int a_;
    if (code == -1) a_ = 1;
    else if (code == -2) a_ = 0;
    else if (code >= 0) a_ = fin.occ[cfg * fin.ostride + code] != 0;
    else a_ = fin.childV[cfg * fin.vstride + (-3 - code)] != 0;
    s.failed[i] = a_;
    s.smode[i] = mc >= -1 ? mc : fin.childM[cfg * fin.vstride + (-3 - mc)];
  }
  __syncthreads();
  if (warp == 0) {
    int found = 0;
    for (int base0 = 0; base0 < n; base0 += 32) {
      int i = base0 + lane;
      bool ok = i < n && s.failed[i] != 0;
      unsigned msk = __ballot_sync(FULL, ok);
      int c = found + __popc(msk & ((1u << lane) - 1u));
      if (i < n) s.cidx[i] = ok ? c : -1;
      if (ok) s.slot[c] = i;
      found += __popc(msk);
    }
    if (lane == 0) { s.misc[3] = found; s.misc[4] = 0; }
  }
  __syncthreads();
  const int m = s.misc[3];
  const bool tc_close = (n % 4 == 0) && (pmax % 4 == 0);
  // U x U deferred under the tensor-core close too: guu_on is the HANDLER's condition (PIVOTED_GUU, the
  // tensor-core close, n <= UU_MAXN), the one the writer kernel launches under — one condition, never two copies
  const bool guu = guu_on && nu_ > 1;
  const int nfc = (!tc_close || guu) ? m - nu_ : m;   // the update rows: deferred on the scalar close, or by guu
  for (int c = t; c < m; c += THREADS) {
    s.alive[c] = 1; s.cand[c] = s.slot[c] < nf; s.failed[c] = 0; s.rowof[c] = -1;
  }
  for (int c = t; c < ucap; c += THREADS) s.cached[c] = -1;
  int nch = 0;
  {
    // the children's blocks gathered — every entry of the alive square written once, coalesced, as the sum of the
    // children's entries at its two slots, no zeroing and no read-modify-write of the slab; the site pieces and the
    // bonds, few pairs, add after: inv[p][i] = the row of child p at compact index i (-1: none), in the
    // rows cache (free until the elimination); then every entry (i, j) of the alive square, four a thread in flight
    int* inv = reinterpret_cast<int*>(s.rows);
    int cw[4]; const real_t* cb[4];
    for (int p = fin.pc_off[f]; p < fin.pc_off[f + 1] && nch < 4; ++p) {
      if (fin.pc_desc[4 * p + 3] >= 0) continue;
      cw[nch] = fin.pc_desc[4 * p + 1];
      cb[nch] = fin.childC + cfg * fin.cstride + fin.pc_desc[4 * p];
      const int32_t* map = fin.pc_map + fin.pc_desc[4 * p + 2];
      for (int i = t; i < m; i += THREADS) inv[nch * n + i] = -1;
      __syncthreads();
      for (int r = t; r < cw[nch]; r += THREADS) {
        const int d = map[r];
        const int ci = d >= 0 ? s.cidx[d] : -1;
        if (ci >= 0) inv[nch * n + ci] = r;
      }
      ++nch;
    }
    __syncthreads();
    const int mm = m * m;
    for (int e0 = t; e0 < mm; e0 += 4 * THREADS) {
      real_t v[4]; int ii[4], jj[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int e = e0 + u * THREADS;
        v[u] = 0.0; ii[u] = -1; jj[u] = 0;
        if (e < mm) {
          const int i = e / m, j = e - i * m;
          if (guu && i >= nfc && j >= nfc) continue;    // U x U: uu_tc_kernel's
          ii[u] = i; jj[u] = j;
          if (i != j) for (int p = 0; p < nch; ++p) {
            const int ri = inv[p * n + i], rj = inv[p * n + j];
            if (ri >= 0 && rj >= 0) v[u] += ri < rj ? __ldg(cb[p] + ri * cw[p] + rj) : -__ldg(cb[p] + rj * cw[p] + ri);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) if (ii[u] >= 0) A[size_t(ii[u]) * n + jj[u]] = v[u];
    }
  }
  __syncthreads();
  assemble_pieces<false>(fin, f, fin.free_ + cfg * fin.fstride, fin.childC + cfg * fin.cstride, s.cidx, t, THREADS,
                         [&](int ci, int cj, real_t v) {
                           if (guu && ci >= nfc) return;     // ci < cj: both in U
                           A[size_t(ci) * n + cj] += v;
                           A[size_t(cj) * n + ci] -= v;
                         }, s.failed, s.failed, nch);
  __syncthreads();
  double la = 0.0, sg = 1.0; real_t big = 0.0;
  int step = 0, tp = 0;
  for (; step < steps; ++step) {
    while (true) {
      if (warp == 0) {      // the lowest untested candidates, by ballot
        int found = 0;
        for (int base0 = 0; base0 < m && found < scan_window; base0 += 32) {
          int i = base0 + lane;
          bool ok = i < m && s.cand[i] && !s.failed[i];
          unsigned msk = __ballot_sync(FULL, ok);
          int before = __popc(msk & ((1u << lane) - 1u));
          if (ok && found + before < scan_window) s.win[found + before] = i;
          found += __popc(msk);
        }
        if (lane == 0) s.misc[0] = found < scan_window ? found : scan_window;
      }
      __syncthreads();
      const int nwin = s.misc[0];
      if (nwin == 0) { if (t == 0) s.misc[1] = -1; __syncthreads(); break; }
      int need = 0;
      for (int w = 0; w < nwin; ++w) need += s.rowof[s.win[w]] < 0;
      if (s.misc[4] + need + 1 > ucap) {          // room for the window and a partner
        gclose(s, A, m, n, pmax, tp, nfc);
        tp = 0;
      }
      for (int w = 0; w < nwin; ++w)
        if (s.rowof[s.win[w]] < 0) gload(s, A, m, n, pmax, tp, s.win[w]);
      for (int w = warp; w < nwin; w += GFT / 32) {    // the same logical window, a warp per row in waves
        const int k = s.win[w];
        const real_t* rk = s.rows + size_t(s.rowof[k]) * n;
        real_t mx = 0.0, bv = 0.0; int br = -1;
        for (int i = lane; i < m; i += 32) {
          if (!s.alive[i] || i == k) continue;
          real_t v = fabs(rk[i]);
          mx = fmax(mx, v);
          if (s.cand[i] && v > bv) { bv = v; br = i; }
        }
        mx = warp_max_nonneg(mx);
        {
          const real_t bmax = warp_max_nonneg(bv);
          br = warp_argmin_at(br >= 0 && bv == bmax, br);
          bv = bmax;
        }
        if (lane == 0) {
          real_t key = -1.0;
          if (br >= 0 && bv > 0.0) key = root ? bv : (bv / mx >= u ? bv / mx : -1.0);
          s.wkey[w] = key; s.wr[w] = br; s.wm[w] = mx;
        }
      }
      __syncthreads();
      if (t == 0) {
        int best = -1; real_t bk = -1.0;
        for (int w = 0; w < nwin; ++w) {
          big = fmax(big, s.wm[w]);
          if (s.wkey[w] > bk) { bk = s.wkey[w]; best = w; }
        }
        if (best >= 0) { s.misc[1] = s.win[best]; s.misc[2] = s.wr[best]; }
        else { s.misc[1] = -2; for (int w = 0; w < nwin; ++w) s.failed[s.win[w]] = 1; }
      }
      __syncthreads();
      if (s.misc[1] != -2) break;
    }
    const int k = s.misc[1];
    if (k < 0) break;      // no pair passes: the rest is the parent's
    const int r = s.misc[2];
    if (s.rowof[r] < 0) gload(s, A, m, n, pmax, tp, r);
    const real_t* rk = s.rows + size_t(s.rowof[k]) * n;
    const real_t* rr = s.rows + size_t(s.rowof[r]) * n;
    const real_t a = rk[r];
    for (int i = t; i < m; i += THREADS) {
      const bool rest = s.alive[i] && i != k && i != r;
      const real_t ci = rest ? -rr[i] : 0.0, ti = rest ? rk[i] / a : 0.0;    // A[i, r], A[k, i] / a
      s.c[i] = ci; s.tau[i] = ti;
      s.PT[size_t(i) * gps(pmax) + tp] = ci; s.PT[size_t(gps(pmax)) * n + size_t(i) * gps(pmax) + tp] = ti;
      s.failed[i] = 0;
    }
    __syncthreads();
    const int sk = s.slot[k], sr = s.slot[r];
    if (t == 0) {
      la += log(fabs(a)); sg *= (a < 0 ? -1.0 : 1.0);
      s.alive[k] = 0; s.alive[r] = 0; s.cand[k] = 0; s.cand[r] = 0;
      pivots[(size_t(b) * steps + step) * 2] = s.smode[sk];
      pivots[(size_t(b) * steps + step) * 2 + 1] = s.smode[sr];
      if (record) {
        rec_kr[(size_t(b) * steps + step) * 2] = sk; rec_kr[(size_t(b) * steps + step) * 2 + 1] = sr;
        rec_a[size_t(b) * steps + step] = a;
      }
    }
    if (record)
      for (int i = t; i < n; i += THREADS) {
        const int ci = s.cidx[i];
        rec_c[(size_t(b) * steps + step) * n + i] = ci >= 0 ? s.c[ci] : 0.0;
        rec_tau[(size_t(b) * steps + step) * n + i] = ci >= 0 ? s.tau[ci] : 0.0;
      }
    for (int i = nfc + t; i < m; i += THREADS) {     // the update rows' c and tau, for the deferred Schur complement
      cu[(size_t(b) * steps + step) * ucp + i - nfc] = s.c[i];
      tu[(size_t(b) * steps + step) * ucp + i - nfc] = s.tau[i];
    }
    __syncthreads();
    // the cached rows now, every entry: (u, j) -= c_u tau_j - tau_u c_j, written for (min, max)
    const int nc = s.misc[4];
    for (int sl = warp; sl < nc; sl += THREADS / 32) {
      const int uu = s.cached[sl];
      if (uu < 0 || !s.alive[uu]) continue;
      real_t* ru = s.rows + size_t(sl) * n;
      const real_t cu = s.c[uu], tu = s.tau[uu];
      for (int j = lane; j < m; j += 32) {
        if (j == uu || ((PIVOTED_GLIVE & 2) && !s.alive[j])) continue;
        ru[j] = uu < j ? RSUB_RN(ru[j], pair_term(cu, s.tau[j], tu, s.c[j]))
                       : RADD_RN(ru[j], pair_term(s.c[j], tu, s.tau[j], cu));
      }
    }
    __syncthreads();
    if (++tp == pmax) { gclose(s, A, m, n, pmax, tp, nfc); tp = 0; }
  }
  gclose(s, A, m, n, pmax, tp, nfc);
  if (!tc_close && nu_ > 1) {
    schur_uu_slab(A, n, nfc, nu_, step, cu + size_t(b) * steps * ucp, tu + size_t(b) * steps * ucp, ucp, t, THREADS);
    __syncthreads();
  }
  for (int q = step + t; q < steps; q += THREADS) {   // the steps no pivot took
    pivots[(size_t(b) * steps + q) * 2] = -1; pivots[(size_t(b) * steps + q) * 2 + 1] = -1;
    if (record) { rec_kr[(size_t(b) * steps + q) * 2] = -1; rec_kr[(size_t(b) * steps + q) * 2 + 1] = -1; }
  }
  if (t == 0) { logabs[b] = la; sign[b] = sg; growth[b] = big; }
  int* alive_slot = s.failed;
  __syncthreads();
  for (int i = t; i < n; i += THREADS) {
    const int ci = s.cidx[i];
    const int a_ = ci >= 0 ? s.alive[ci] : 0;
    alive_out[vec + i] = a_;
    alive_slot[i] = a_;
  }
  __syncthreads();
  const int cd = d_max + u_max;
  int* rowslot = s.slot;
  int* cidx2 = s.rowof;           // the slot -> compact map, kept: `slot` becomes rowslot
  for (int i = t; i < n; i += THREADS) cidx2[i] = s.cidx[i];
  __syncthreads();
  const int left = contribution_rows<THREADS>(alive_slot, n, nf, sh_, nu_, d_max, u_max, rowslot, s.misc);
  if (t == 0) count[b] = left;
  const size_t cvec = size_t(b) * cd, cmat = size_t(b) * cd * cd;
  for (int c = t; c < cd; c += THREADS) {
    const int sl = rowslot[c];
    cvalid[cvec + c] = sl >= 0;
    cmode[cvec + c] = sl >= 0 ? s.smode[sl] : -1;
  }
  for (int e = t; e < cd * cd; e += THREADS) {   // the valid rows' strict upper triangle only
    const int r_ = e / cd, q_ = e - r_ * cd;
    const int s1 = rowslot[r_], s2 = rowslot[q_];
    if (r_ < q_ && s1 >= 0 && s2 >= 0 && !(guu && cidx2[s1] >= nfc && cidx2[s2] >= nfc))
      Cout[cmat + e] = A[size_t(cidx2[s1]) * n + cidx2[s2]];
  }
  __syncthreads();
}

#ifndef PIVOTED_GBWD_LIVE
#define PIVOTED_GBWD_LIVE 0   // 1 = the panel products over the live rows and columns only
#endif
// The backward pass of front_gfwd_kernel (or of any front), W in global memory: `W0`'s slab in full skew storage
// over slots, the steps replayed latest first in panels of up to `pmax`: one pass over W gives W tau_q and W c_q
// for the whole panel, and the panel's own updates, each a row k and a column r, enter the later steps' products
// as O(n) corrections; the updates are written to W when the panel ends.
__device__ __forceinline__ real_t gblock_sum(real_t v, real_t* red) {
  const int t = threadIdx.x;
  for (int off = 16; off; off >>= 1) v += __shfl_down_sync(FULL, v, off);
  __syncthreads();
  if (t % 32 == 0) red[t / 32] = v;
  __syncthreads();
  real_t tot = 0.0;
  for (int w = 0; w < GT / 32; ++w) tot += red[w];
  __syncthreads();
  return tot;
}

size_t gbwd_smem_bytes(int n, int pmax) {
  // TV (2p n), Y (2p n), CB, TB (p n each), DOT (p p 4), ABAR, A (p), red (GT/32); ints alive, rowslot (n), ks, rs (p),
  // misc 8
  return sizeof(real_t) * (6 * size_t(pmax) * n + 4 * size_t(pmax) * pmax + 2 * pmax + GT / 32)
       + sizeof(int) * (2 * size_t(n) + 2 * pmax + 8 + 2 * pmax);
}

// The global backward's register floor: 0 = __launch_bounds__(GT) alone (one 512-thread block an SM); F > 0 = a
// floor of F blocks. PM: the panel's compile-time cap — a row's accumulators
// (2 PM doubles) are sized by it, not by GPANEL_MAX, so a panel of 8 holds 32 registers of them, not 64.
#ifndef PIVOTED_GBWD_MINBLOCKS
#define PIVOTED_GBWD_MINBLOCKS 0
#endif
#if PIVOTED_GBWD_MINBLOCKS > 0
#define GBWD_BOUNDS __launch_bounds__(GT, PIVOTED_GBWD_MINBLOCKS)
#else
#define GBWD_BOUNDS __launch_bounds__(GT)
#endif
template <int PM>
__global__ void GBWD_BOUNDS front_gbwd_kernel(
    const int8_t* alive_out, const int32_t* rec_kr, const real_t* rec_a, const real_t* rec_c, const real_t* rec_tau,
    const real_t* Cbar, const real_t* ybar, const int32_t* counts, int nfront, int n, int steps, int d_max, int u_max,
    real_t* W0, int pmax, const GatherIn g) {
  extern __shared__ __align__(16) char base[];
  const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int cix = g.bycount ? b % nfront : b;
  real_t* TV = reinterpret_cast<real_t*>(base);             // [2q] tau_q, [2q + 1] c_q
  real_t* Y = TV + 2 * size_t(pmax) * n;                    // W_base tau_q, W_base c_q
  real_t* CB = Y + 2 * size_t(pmax) * n;                    // cbar_q
  real_t* TB = CB + size_t(pmax) * n;                       // taubar_q / a_q
  real_t* DOT = TB + size_t(pmax) * n;                      // [q'][q][4]: cb'.tau_q, cb'.c_q, tb'.tau_q, tb'.c_q
  real_t* ABAR = DOT + 4 * size_t(pmax) * pmax;
  real_t* AQ = ABAR + pmax;
  real_t* red = AQ + pmax;
  int* alive = reinterpret_cast<int*>(red + GT / 32);
  int* rowslot = alive + n;
  int* KS = rowslot + n; int* RS = KS + pmax; int* misc = RS + pmax; int* ST = misc + 8;
  const size_t vec = size_t(b) * n;
  real_t* W = W0 + size_t(b) * n * n;
  for (int i = t; i < n; i += GT) alive[i] = alive_out[vec + i] != 0;
  for (size_t e = t; e < size_t(n) * n; e += GT) W[e] = 0.0;
  __syncthreads();
  const int cd = d_max + u_max;
  contribution_rows<GT>(alive, n, counts[3 * cix], counts[3 * cix + 1], counts[3 * cix + 2], d_max, u_max, rowslot, misc);
  const size_t cmat = size_t(g.cbar_f ? b % nfront : b) * cd * cd;
  for (int e = t; e < cd * cd; e += GT) {
    const int r = e / cd, q = e - r * cd;
    const int s1 = rowslot[r], s2 = rowslot[q];
    if (r < q && s1 >= 0 && s2 >= 0) {
      const real_t v = Cbar[cmat + e];   // the cotangent's upper entry alone (the gather writes no lower one)
      W[size_t(s1) * n + s2] = v; W[size_t(s2) * n + s1] = -v;
    }
  }
  __syncthreads();
  const real_t yb = ybar[g.ybar_f ? b % nfront : b];
  int step = steps - 1;
  while (step >= 0) {
    if (t == 0) {        // the panel: up to pmax pivoting steps, latest first
      int np_ = 0, st = step;
      for (; st >= 0 && np_ < pmax; --st) {
        const int k = rec_kr[(size_t(b) * steps + st) * 2];
        if (k < 0) continue;
        KS[np_] = k; RS[np_] = rec_kr[(size_t(b) * steps + st) * 2 + 1]; ST[np_] = st;
        AQ[np_] = rec_a[size_t(b) * steps + st];
        ++np_;
      }
      misc[0] = np_; misc[1] = st;
    }
    __syncthreads();
    const int np_ = misc[0];
    step = misc[1];
    if (np_ == 0) break;
    for (int q = 0; q < np_; ++q)
      for (int i = t; i < n; i += GT) {
        TV[(2 * q) * size_t(n) + i] = rec_tau[(size_t(b) * steps + ST[q]) * n + i];
        TV[(2 * q + 1) * size_t(n) + i] = rec_c[(size_t(b) * steps + ST[q]) * n + i];
      }
    __syncthreads();
    // Y = W_base [tau_q, c_q]: a warp a row, lanes along it
#if PIVOTED_GBWD_LIVE
    // the live set (into rowslot, free after the contribution's adjoint): alive at the panel's start, or one of the
    // panel's own pivots — every step of the panel has zero tau and c elsewhere, and no other row's product is read
    if (warp == 0) {
      int found = 0;
      for (int base0 = 0; base0 < n; base0 += 32) {
        const int j = base0 + lane;
        bool ok = j < n && alive[j];
        if (j < n && !ok) for (int p = 0; p < np_; ++p) ok = ok || KS[p] == j || RS[p] == j;
        const unsigned msk = __ballot_sync(FULL, ok);
        if (ok) rowslot[found + __popc(msk & ((1u << lane) - 1u))] = j;
        found += __popc(msk);
      }
      if (lane == 0) misc[2] = found;
    }
    __syncthreads();
    const int nl = misc[2];
    for (int ii = warp; ii < nl; ii += GT / 32) {
      const int i = rowslot[ii];
      real_t acc[2 * PM];
#pragma unroll
      for (int q = 0; q < 2 * PM; ++q) acc[q] = 0.0;
      const real_t* wi = W + size_t(i) * n;
      for (int jj = lane; jj < nl; jj += 32) {
        const int j = rowslot[jj];
        const real_t w = wi[j];
#pragma unroll
        for (int q = 0; q < 2 * PM; ++q)
          if (q < 2 * np_) acc[q] += w * TV[q * size_t(n) + j];
      }
#else
    for (int i = warp; i < n; i += GT / 32) {
      real_t acc[2 * PM];
#pragma unroll
      for (int q = 0; q < 2 * PM; ++q) acc[q] = 0.0;
      const real_t* wi = W + size_t(i) * n;
      for (int j = lane; j < n; j += 32) {
        const real_t w = wi[j];
#pragma unroll
        for (int q = 0; q < 2 * PM; ++q)
          if (q < 2 * np_) acc[q] += w * TV[q * size_t(n) + j];
      }
#endif
#pragma unroll
      for (int q = 0; q < 2 * PM; ++q) {
        if (q >= 2 * np_) break;
        real_t v = acc[q];
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(FULL, v, off);
        if (lane == 0) Y[q * size_t(n) + i] = real_t(v);
      }
    }
    __syncthreads();
    for (int q = 0; q < np_; ++q) {
      const int k = KS[q], r = RS[q];
      const real_t a = AQ[q];
      const real_t* tq = TV + (2 * q) * size_t(n);
      const real_t* cq = TV + (2 * q + 1) * size_t(n);
      // the products with W as it stands at this step: the base, plus the panel's earlier (later-in-time) updates
      for (int i = t; i < n; i += GT) {
        real_t y1 = Y[(2 * q) * size_t(n) + i], y2 = Y[(2 * q + 1) * size_t(n) + i];
        for (int p = 0; p < q; ++p) {
          const int kp = KS[p], rp = RS[p];
          const real_t* cb = CB + size_t(p) * n;
          const real_t* tb = TB + size_t(p) * n;
          const real_t* d = DOT + (size_t(p) * pmax + q) * 4;
          const real_t ab = ABAR[p];
          // (dW v)_i = cb_i v_r - [i = r] cb.v + [i = k] tb.v - tb_i v_k + ab ([i = k] v_r - [i = r] v_k)
          y1 += cb[i] * tq[rp] - tb[i] * tq[kp];
          y2 += cb[i] * cq[rp] - tb[i] * cq[kp];
          if (i == rp) { y1 -= d[0] + ab * tq[kp]; y2 -= d[1] + ab * cq[kp]; }
          if (i == kp) { y1 += d[2] + ab * tq[rp]; y2 += d[3] + ab * cq[rp]; }
        }
        CB[size_t(q) * n + i] = alive[i] ? real_t(-y1) : real_t(0);     // cbar
        TB[size_t(q) * n + i] = alive[i] ? real_t(y2) : real_t(0);      // taubar, divided by a below
      }
      __syncthreads();
      real_t part = 0.0;
      for (int j = t; j < n; j += GT) part += TB[size_t(q) * n + j] * tq[j];
      const real_t tot = gblock_sum(real_t(part), red);
      for (int j = t; j < n; j += GT) TB[size_t(q) * n + j] /= a;
      if (t == 0) {
        ABAR[q] = -tot / a + yb / a;
        alive[k] = 1; alive[r] = 1;
      }
      __syncthreads();
      // this step's dots with the panel's earlier-in-time steps' vectors, a warp each
      for (int idx = warp; idx < 4 * (np_ - 1 - q); idx += GT / 32) {
        const int q2 = q + 1 + idx / 4, which = idx % 4;
        const real_t* x = (which < 2 ? CB : TB) + size_t(q) * n;
        const real_t* v = TV + (2 * q2 + (which & 1)) * size_t(n);
        real_t acc = 0.0;
        for (int j = lane; j < n; j += 32) acc += x[j] * v[j];
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(FULL, acc, off);
        if (lane == 0) DOT[(size_t(q) * pmax + q2) * 4 + which] = real_t(acc);
      }
      __syncthreads();
    }
    // the panel's updates into W, in the order they were made
    for (int q = 0; q < np_; ++q) {
      const int k = KS[q], r = RS[q];
      const real_t* cb = CB + size_t(q) * n;
      const real_t* tb = TB + size_t(q) * n;
      for (int i = t; i < n; i += GT) {
        if (cb[i] != 0.0) { W[size_t(i) * n + r] += cb[i]; W[size_t(r) * n + i] -= cb[i]; }
      }
      __syncthreads();
      for (int i = t; i < n; i += GT) {
        if (tb[i] != 0.0) { W[size_t(k) * n + i] += tb[i]; W[size_t(i) * n + k] -= tb[i]; }
      }
      __syncthreads();
      if (t == 0) { W[size_t(k) * n + r] += ABAR[q]; W[size_t(r) * n + k] -= ABAR[q]; }
      __syncthreads();
    }
  }
  if (g.off) {     // the adjoint straight to the sources
    __syncthreads();
    const int f = b % nfront;
    const size_t cfg = size_t(b) / nfront;
    for (int e = g.off[f] + t; e < g.off[f + 1]; e += GT) {
      const int src = g.s[e], loc = g.loc[e];
      const real_t v = W[loc] * g.sg[e];
      if (src <= g.fs) g.dfree[cfg * g.fs + src - 1] = v; else g.dchild[cfg * g.cs + src - 1 - g.fs] = v;
    }
  }
}


// leading-batch size of a buffer whose per-configuration part holds `per` elements
static size_t lead_of(size_t elements, size_t per) { return per ? elements / per : 1; }

static ffi::Error FusedForward(cudaStream_t stream, ffi::Buffer<FFI_REAL> free_, ffi::Buffer<FFI_REAL> childC,
                               ffi::Buffer<ffi::S8> childV, ffi::Buffer<ffi::S32> childM, ffi::Buffer<ffi::S8> occ,
                               ffi::Buffer<ffi::S32> pc_off, ffi::Buffer<ffi::S32> pc_desc, ffi::Buffer<ffi::S32> pc_map,
                               ffi::Buffer<ffi::S32> bd_off, ffi::Buffer<ffi::S32> bd_pq, ffi::Buffer<FFI_REAL> bd_val,
                               ffi::Buffer<ffi::S32> asrc, ffi::Buffer<ffi::S32> msrc, ffi::Buffer<ffi::S32> counts,
                               ffi::Buffer<ffi::U32> ijtab, ffi::Buffer<ffi::S32> ijoff, int64_t nfront,
                               int64_t n, int64_t fs, int64_t cs, int64_t vs, int64_t np_, int64_t steps, double u,
                               int64_t root, int64_t record, int64_t d_max, int64_t u_max, int64_t panel, int64_t search_window,
                               int64_t warp_max, int64_t uuent,
                               ffi::Result<ffi::Buffer<FFI_REAL>> Cout, ffi::Result<ffi::Buffer<ffi::S8>> cvalid,
                               ffi::Result<ffi::Buffer<ffi::S32>> cmode, ffi::Result<ffi::Buffer<ffi::S32>> count,
                               ffi::Result<ffi::Buffer<ffi::S8>> alive_out, ffi::Result<ffi::Buffer<FFI_REAL>> logabs,
                               ffi::Result<ffi::Buffer<FFI_REAL>> sign, ffi::Result<ffi::Buffer<ffi::S32>> pivots,
                               ffi::Result<ffi::Buffer<FFI_REAL>> growth, ffi::Result<ffi::Buffer<ffi::S32>> rec_kr,
                               ffi::Result<ffi::Buffer<FFI_REAL>> rec_a, ffi::Result<ffi::Buffer<FFI_REAL>> rec_c,
                               ffi::Result<ffi::Buffer<FFI_REAL>> rec_tau, ffi::Result<ffi::Buffer<ffi::F64>> cu,
                               ffi::Result<ffi::Buffer<ffi::F64>> tu) {
  if (d_max + u_max > n) return ffi::Error::InvalidArgument("contribution wider than the front");
  if (n > MAXN) return ffi::Error::InvalidArgument("front wider than MAXN");
  const size_t lf = lead_of(free_.element_count(), fs), lc = lead_of(childC.element_count(), cs);
  const size_t lv = lead_of(childV.element_count(), vs), lo = lead_of(occ.element_count(), np_);
  size_t B = 1;
  for (size_t l : {lf, lc, lv, lo}) if (l > B) B = l;
  for (size_t l : {lf, lc, lv, lo}) if (l != 1 && l != B) return ffi::Error::InvalidArgument("inconsistent batch");
  if (logabs->element_count() != B * size_t(nfront)) return ffi::Error::InvalidArgument("output batch mismatch");
  const size_t total = B * size_t(nfront);
  const int ucp = int(cu->dimensions()[cu->dimensions().size() - 1]);
  // a block a front, assembling itself: fronts wider than the warp kernel's bound (WARP_MAXN, or PIVOTED_WARP_MAX
  // below it — the narrow levels are occupancy-bound, a block holds 8 warps on a front where a warp holds 1)
  if (n > WARP_MAXN || n > warp_max) {
    const int TB = n > 96 && PIVOTED_BLOCK_THREADS == 128 ? 128 :
                   n > 128 && PIVOTED_BLOCK_THREADS == 256 ? 256 : n <= 32 ? 64 : n <= 64 ? 128 : n <= 128 ? 256 : 512;
    const int window = n > 128 ? 512 / 32 : n > 96 ? 256 / 32 : TB / 32;
    const int panel_threads = n > 96 && PIVOTED_BLOCK_THREADS == 128 ? 256 : TB;
    if (smem_bytes(int(n), panel_threads, 1, 0, window, int(uuent)) > size_t(g_smem_limit))
      return ffi::Error::InvalidArgument("front too wide for this GPU's shared memory");
    int pm = int(panel);
    pm = pm < 1 ? 1 : pm > PANEL_MAX ? PANEL_MAX : pm;
    while (pm > 1 && smem_bytes(int(n), panel_threads, pm, 0, window) > size_t(g_smem_limit)) --pm;
    const FusedIn fin{free_.typed_data(), lf == 1 ? 0 : size_t(fs), childC.typed_data(), lc == 1 ? 0 : size_t(cs),
                      childV.typed_data(), childM.typed_data(), lv == 1 ? 0 : size_t(vs), occ.typed_data(),
                      lo == 1 ? 0 : size_t(np_), pc_off.typed_data(), pc_desc.typed_data(), pc_map.typed_data(),
                      bd_off.typed_data(), bd_pq.typed_data(), bd_val.typed_data(), asrc.typed_data(), msrc.typed_data(),
                      ijtab.typed_data(), ijoff.typed_data(), n <= 320 ? int(n) : 0};
#define FFWD(T, CAP, FALLBACK) front_fwd_kernel<T><<<total, T, smem_bytes(int(n), T, pm, CAP, window, int(uuent)), stream>>>( \
        counts.typed_data(), int(nfront), int(n), int(steps), real_t(u), int(root), int(record), \
        int(d_max), int(u_max), Cout->typed_data(), cvalid->typed_data(), cmode->typed_data(), count->typed_data(), \
        alive_out->typed_data(), logabs->typed_data(), sign->typed_data(), pivots->typed_data(), \
        growth->typed_data(), rec_kr->typed_data(), rec_a->typed_data(), rec_c->typed_data(), rec_tau->typed_data(), \
        cu->typed_data(), tu->typed_data(), ucp, pm, int(search_window), fin, CAP, FALLBACK, int(uuent))
    int capacity = int(n);
#if PIVOTED_BLOCK_COMPACT > 0
    if (cs > 0) capacity = int(n) > PIVOTED_BLOCK_COMPACT ? int(n) - PIVOTED_BLOCK_COMPACT : 1;
#endif
    for (int pass = 0; pass < (capacity < n ? 2 : 1); ++pass) {
      const int cap = pass == 0 ? capacity : int(n);
      if (TB == 64) FFWD(64, cap, pass); else if (TB == 128) FFWD(128, cap, pass);
      else if (TB == 256) FFWD(256, cap, pass); else FFWD(512, cap, pass);
      const auto e = cudaGetLastError();
      if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    }
#undef FFWD
    if (uuent > 0 && n <= UU_MAXN) {
      uu_tc_kernel<<<total, UU_THREADS, 0, stream>>>(fin, counts.typed_data(), int(nfront), int(n), int(steps),
                                                     int(d_max), int(u_max), Cout->typed_data(), pivots->typed_data(),
                                                     cu->typed_data(), tu->typed_data(), ucp);
    }
    auto e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(e));
  }
  const size_t blocks = (total + WPB - 1) / WPB;
  const FusedIn fin{free_.typed_data(), lf == 1 ? 0 : size_t(fs), childC.typed_data(), lc == 1 ? 0 : size_t(cs),
                    childV.typed_data(), childM.typed_data(), lv == 1 ? 0 : size_t(vs), occ.typed_data(),
                    lo == 1 ? 0 : size_t(np_), pc_off.typed_data(), pc_desc.typed_data(), pc_map.typed_data(),
                    bd_off.typed_data(), bd_pq.typed_data(), bd_val.typed_data(), asrc.typed_data(), msrc.typed_data(),
                      ijtab.typed_data(), ijoff.typed_data(), n <= 320 ? int(n) : 0};
#define FUSED_LAUNCH(CAP, FALLBACK) front_fwd_fused<<<blocks, 32 * WPB, WPB * fused_bytes(int(n), CAP), stream>>>( \
      fin, counts.typed_data(), int(nfront), int(n), int(steps), real_t(u), int(root), int(record), \
      int(d_max), int(u_max), total, Cout->typed_data(), cvalid->typed_data(), cmode->typed_data(), \
      count->typed_data(), alive_out->typed_data(), logabs->typed_data(), sign->typed_data(), pivots->typed_data(), \
      growth->typed_data(), rec_kr->typed_data(), rec_a->typed_data(), rec_c->typed_data(), rec_tau->typed_data(), cu->typed_data(), tu->typed_data(), ucp, CAP, FALLBACK)
  int capacity = int(n);
#if PIVOTED_WARP_COMPACT > 0
  // Leaves have no child-delay padding to reclaim. Child-fed fronts try a smaller
  // triangle, keeping their full slot capacity and recording layout unchanged.
  if (cs > 0) capacity = int(n) > PIVOTED_WARP_COMPACT ? int(n) - PIVOTED_WARP_COMPACT : 1;
#endif
  for (int pass = 0; pass < (capacity < n ? 2 : 1); ++pass) {
    const int cap = pass == 0 ? capacity : int(n);
    FUSED_LAUNCH(cap, pass);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  }
#undef FUSED_LAUNCH
  auto e = cudaGetLastError();
  return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FrontFused, FusedForward,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<ffi::S8>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S8>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("nfront").Attr<int64_t>("n").Attr<int64_t>("fs").Attr<int64_t>("cs")
        .Attr<int64_t>("vs").Attr<int64_t>("np").Attr<int64_t>("steps").Attr<double>("u").Attr<int64_t>("root")
        .Attr<int64_t>("record").Attr<int64_t>("d_max").Attr<int64_t>("u_max").Attr<int64_t>("panel").Attr<int64_t>("search_window")
        .Attr<int64_t>("warp_max")
        .Attr<int64_t>("uuent")
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::S8>>().Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S8>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>(),
    {ffi::Traits::kCmdBufferCompatible});

// The global-memory forward: front_fwd_fused's inputs and outputs, plus the fronts' slabs (`work`, n x n each)
static ffi::Error GlobalForward(cudaStream_t stream, ffi::Buffer<FFI_REAL> free_, ffi::Buffer<FFI_REAL> childC,
                                ffi::Buffer<ffi::S8> childV, ffi::Buffer<ffi::S32> childM, ffi::Buffer<ffi::S8> occ,
                                ffi::Buffer<ffi::S32> pc_off, ffi::Buffer<ffi::S32> pc_desc, ffi::Buffer<ffi::S32> pc_map,
                                ffi::Buffer<ffi::S32> bd_off, ffi::Buffer<ffi::S32> bd_pq, ffi::Buffer<FFI_REAL> bd_val,
                                ffi::Buffer<ffi::S32> asrc, ffi::Buffer<ffi::S32> msrc, ffi::Buffer<ffi::S32> counts,
                                ffi::Buffer<ffi::U32> ijtab, ffi::Buffer<ffi::S32> ijoff, int64_t nfront,
                                int64_t n, int64_t fs, int64_t cs, int64_t vs, int64_t np_, int64_t steps, double u,
                                int64_t root, int64_t record, int64_t d_max, int64_t u_max, int64_t panel, int64_t search_window,
                                ffi::Result<ffi::Buffer<FFI_REAL>> Cout, ffi::Result<ffi::Buffer<ffi::S8>> cvalid,
                                ffi::Result<ffi::Buffer<ffi::S32>> cmode, ffi::Result<ffi::Buffer<ffi::S32>> count,
                                ffi::Result<ffi::Buffer<ffi::S8>> alive_out, ffi::Result<ffi::Buffer<FFI_REAL>> logabs,
                                ffi::Result<ffi::Buffer<FFI_REAL>> sign, ffi::Result<ffi::Buffer<ffi::S32>> pivots,
                                ffi::Result<ffi::Buffer<FFI_REAL>> growth, ffi::Result<ffi::Buffer<ffi::S32>> rec_kr,
                                ffi::Result<ffi::Buffer<FFI_REAL>> rec_a, ffi::Result<ffi::Buffer<FFI_REAL>> rec_c,
                                ffi::Result<ffi::Buffer<FFI_REAL>> rec_tau, ffi::Result<ffi::Buffer<ffi::F64>> cu,
                                ffi::Result<ffi::Buffer<ffi::F64>> tu, ffi::Result<ffi::Buffer<FFI_REAL>> work) {
  const size_t lf = lead_of(free_.element_count(), fs), lc = lead_of(childC.element_count(), cs);
  const size_t lv = lead_of(childV.element_count(), vs), lo = lead_of(occ.element_count(), np_);
  size_t B = 1;
  for (size_t l : {lf, lc, lv, lo}) if (l > B) B = l;
  for (size_t l : {lf, lc, lv, lo}) if (l != 1 && l != B) return ffi::Error::InvalidArgument("inconsistent batch");
  if (logabs->element_count() != B * size_t(nfront)) return ffi::Error::InvalidArgument("output batch mismatch");
  if (d_max + u_max > n) return ffi::Error::InvalidArgument("contribution wider than the front");
  const size_t total = B * size_t(nfront);
  // the panel and the row cache: as asked, then smaller until shared memory holds them
  int pm = int(panel);
  pm = pm < 1 ? 1 : pm > GPANEL_MAX ? GPANEL_MAX : pm;
  pm = pm < 4 ? 4 : (pm + 3) / 4 * 4;     // the tensor-core close reads the panel four columns at a time
  const int pstep = 4;
  int ucap = GWINDOW + 3 * pm + 8;
  // GBLOCKS blocks an SM share its shared memory (1 KB a block reserved by the hardware)
  const size_t glimit = PIVOTED_GBLOCKS > 1 ? size_t(g_smem_per_sm) / PIVOTED_GBLOCKS - 1024 : size_t(g_smem_limit);
  while (gsmem_bytes(int(n), pm, ucap) > glimit && pm > pstep) { pm -= pstep; ucap = GWINDOW + 3 * pm + 8; }
  while (gsmem_bytes(int(n), pm, ucap) > glimit && ucap > GWINDOW + 2) --ucap;   // a smaller cache: the closes come sooner
  const int active_window = search_window > 0 && search_window < GWINDOW ? int(search_window) : GWINDOW;
  // a narrower search window needs fewer cached rows: gload needs the window's rows and one partner, plus one
  // row of margin; only the cached data move, not the pivot rule
  while (gsmem_bytes(int(n), pm, ucap) > glimit && ucap > active_window + 2) --ucap;
  if (gsmem_bytes(int(n), pm, ucap) > glimit)
    return ffi::Error::InvalidArgument("front too wide even for the global-memory kernel's row cache");
  const FusedIn fin{free_.typed_data(), lf == 1 ? 0 : size_t(fs), childC.typed_data(), lc == 1 ? 0 : size_t(cs),
                    childV.typed_data(), childM.typed_data(), lv == 1 ? 0 : size_t(vs), occ.typed_data(),
                    lo == 1 ? 0 : size_t(np_), pc_off.typed_data(), pc_desc.typed_data(), pc_map.typed_data(),
                    bd_off.typed_data(), bd_pq.typed_data(), bd_val.typed_data(), asrc.typed_data(), msrc.typed_data()};
  // the update block deferred to uu_tc_kernel after the elimination: the kernel and the launch
  // share this one condition (a level wider than UU_MAXN is eliminated whole, as without PIVOTED_GUU)
  const int guu_on = PIVOTED_GUU && (n % 4 == 0) && (pm % 4 == 0) && n <= UU_MAXN;
  front_gfwd_kernel<<<total, GFT, gsmem_bytes(int(n), pm, ucap, GFT), stream>>>(
      counts.typed_data(), int(nfront), int(n), int(steps), real_t(u), int(root), int(record), int(d_max), int(u_max),
      Cout->typed_data(), cvalid->typed_data(), cmode->typed_data(), count->typed_data(), alive_out->typed_data(),
      logabs->typed_data(), sign->typed_data(), pivots->typed_data(), growth->typed_data(), rec_kr->typed_data(),
      rec_a->typed_data(), rec_c->typed_data(), rec_tau->typed_data(), cu->typed_data(), tu->typed_data(),
      int(cu->dimensions()[cu->dimensions().size() - 1]), work->typed_data(), pm, ucap, int(search_window), fin,
      guu_on);
  if (guu_on)
    uu_tc_kernel<<<total, UU_THREADS, 0, stream>>>(
                                                      fin, counts.typed_data(), int(nfront), int(n), int(steps),
                                                      int(d_max), int(u_max), Cout->typed_data(), pivots->typed_data(),
                                                      cu->typed_data(), tu->typed_data(),
                                                      int(cu->dimensions()[cu->dimensions().size() - 1]));
  auto e = cudaGetLastError();
  return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FrontGlobal, GlobalForward,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<ffi::S8>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S8>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("nfront").Attr<int64_t>("n").Attr<int64_t>("fs").Attr<int64_t>("cs")
        .Attr<int64_t>("vs").Attr<int64_t>("np").Attr<int64_t>("steps").Attr<double>("u").Attr<int64_t>("root")
        .Attr<int64_t>("record").Attr<int64_t>("d_max").Attr<int64_t>("u_max").Attr<int64_t>("panel").Attr<int64_t>("search_window")
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::S8>>().Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<ffi::S8>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<ffi::S32>>().Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<FFI_REAL>>()
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<FFI_REAL>>(),
    {ffi::Traits::kCmdBufferCompatible});

// the global backward at the panel's compile-time cap: 4, 8 or 16
static void gbwd_launch(size_t total, cudaStream_t stream, const int8_t* alive_out, const int32_t* rec_kr,
    const real_t* rec_a, const real_t* rec_c, const real_t* rec_tau, const real_t* Cbar, const real_t* ybar,
    const int32_t* counts, int nfront, int n, int steps, int d_max, int u_max, real_t* W0, int pm, const GatherIn g) {
  const size_t sm = gbwd_smem_bytes(n, pm);
  if (pm <= 4) front_gbwd_kernel<4><<<total, GT, sm, stream>>>(alive_out, rec_kr, rec_a, rec_c, rec_tau, Cbar, ybar, counts, nfront, n, steps, d_max, u_max, W0, pm, g);
  else if (pm <= 8) front_gbwd_kernel<8><<<total, GT, sm, stream>>>(alive_out, rec_kr, rec_a, rec_c, rec_tau, Cbar, ybar, counts, nfront, n, steps, d_max, u_max, W0, pm, g);
  else front_gbwd_kernel<16><<<total, GT, sm, stream>>>(alive_out, rec_kr, rec_a, rec_c, rec_tau, Cbar, ybar, counts, nfront, n, steps, d_max, u_max, W0, pm, g);
}

// The gathered-adjoint backwards: the shared and the global backward with the per-front source table,
// called with vmap's expand_dims (the tables and counts unbatched): the free vector's gradient accumulated in place
// (`dfree` aliases the accumulator `acc`, the running sum through the levels), the child blocks' zeroed unless
// `nomemset` (every entry a child's backward reads is one the table writes), then the sources' gradients written.
static ffi::Error BackwardS(cudaStream_t stream, ffi::Buffer<ffi::S8> alive_out, ffi::Buffer<ffi::S32> rec_kr,
                           ffi::Buffer<FFI_REAL> rec_a, ffi::Buffer<FFI_REAL> rec_c, ffi::Buffer<FFI_REAL> rec_tau,
                           ffi::Buffer<FFI_REAL> Cbar, ffi::Buffer<FFI_REAL> ybar, ffi::Buffer<ffi::S32> counts,
                           ffi::Buffer<ffi::S32> off, ffi::Buffer<ffi::S32> src, ffi::Buffer<ffi::S32> loc,
                           ffi::Buffer<FFI_REAL> sg, ffi::Buffer<FFI_REAL> acc,
                           int64_t steps, int64_t n, int64_t d_max, int64_t u_max, int64_t panel, int64_t glob,
                           int64_t fs, int64_t cs, int64_t uuent, int64_t nomemset,
                           ffi::Result<ffi::Buffer<FFI_REAL>> W0, ffi::Result<ffi::Buffer<FFI_REAL>> dfree,
                           ffi::Result<ffi::Buffer<FFI_REAL>> dchild, ffi::Result<ffi::Buffer<FFI_REAL>> ypre) {
  auto dc = counts.dimensions();
  const int nfront = int(dc[dc.size() - 2]);
  const size_t total = alive_out.element_count() / size_t(n);     // alive_out is always batched (a residual)
  if (!nomemset) cudaMemsetAsync(dchild->typed_data(), 0, dchild->size_bytes(), stream);   // nomemset: only mapped entries are ever read
  if (total == 0) return ffi::Error::Success();
  const size_t cd = size_t(d_max + u_max);
  const int cbar_f = Cbar.element_count() == size_t(nfront) * cd * cd && total != size_t(nfront);
  const int ybar_f = ybar.element_count() == size_t(nfront) && total != size_t(nfront);
  GatherIn g{off.typed_data(), src.typed_data(), loc.typed_data(), sg.typed_data(), int(fs), int(cs),
             dfree->typed_data(), dchild->typed_data(), 1, cbar_f, ybar_f};
  if (uuent > 0 && !glob) {    // the update block out of shared memory: its products first
    if (!(PIVOTED_BWD_WORK & 8)) return ffi::Error::InvalidArgument("uuent needs PIVOTED_BWD_WORK bit 8");
    bwd_uu_launch(total, stream, Cbar.typed_data(), rec_kr.typed_data(), rec_tau.typed_data(), rec_c.typed_data(),
        counts.typed_data(), nfront, int(n), int(steps), int(d_max), int(u_max), 1, cbar_f, ypre->typed_data());
    g.uuent = int(uuent); g.ypre = ypre->typed_data();
  }
  if (glob) {
    int pm = int(panel);
    pm = pm < 1 ? 1 : pm > GPANEL_MAX ? GPANEL_MAX : pm;
    while (pm > 1 && gbwd_smem_bytes(int(n), pm) > size_t(g_smem_limit)) --pm;
    if (gbwd_smem_bytes(int(n), pm) > size_t(g_smem_limit))
      return ffi::Error::InvalidArgument("front too wide for the global-memory backward kernel");
    gbwd_launch(total, stream, 
        alive_out.typed_data(), rec_kr.typed_data(), rec_a.typed_data(), rec_c.typed_data(), rec_tau.typed_data(),
        Cbar.typed_data(), ybar.typed_data(), counts.typed_data(), nfront, int(n), int(steps), int(d_max), int(u_max),
        W0->typed_data(), pm, g);
  } else {
#define BWDS(T) front_bwd_kernel<T><<<total, T, smem_bytes(int(n), T, (PIVOTED_BWD_WORK & 8) ? 1 : 0, 0, 0, g.uuent), stream>>>( \
      alive_out.typed_data(), rec_kr.typed_data(), rec_a.typed_data(), rec_c.typed_data(), rec_tau.typed_data(), \
      Cbar.typed_data(), ybar.typed_data(), counts.typed_data(), nfront, int(n), int(steps), int(d_max), int(u_max), g)
    if (n <= 64) BWDS(64); else if (n <= 128) BWDS(256); else BWDS(512);
#undef BWDS
  }
  auto e = cudaGetLastError();
  return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FrontBackwardS, BackwardS,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S8>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<FFI_REAL>>()
        .Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<FFI_REAL>>()
        .Arg<ffi::Buffer<FFI_REAL>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<FFI_REAL>>()
        .Arg<ffi::Buffer<FFI_REAL>>()
        .Attr<int64_t>("steps").Attr<int64_t>("n").Attr<int64_t>("d_max").Attr<int64_t>("u_max")
        .Attr<int64_t>("panel").Attr<int64_t>("glob").Attr<int64_t>("fs").Attr<int64_t>("cs").Attr<int64_t>("uuent")
        .Attr<int64_t>("nomemset")
        .Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<FFI_REAL>>().Ret<ffi::Buffer<FFI_REAL>>(),
    {ffi::Traits::kCmdBufferCompatible});

// The sign's permutation parity, one block a configuration: the alive modes in increasing id are sent,
// in order, to the valid pivots (mode ids, -1 absent) of the levels' pivot arrays taken in order; a dead mode and a
// pivot past the alive count are fixed points; the parity is (n - cycles) mod 2, the cycles by pointer doubling in
// shared memory. The value batched._parity_modes computes, without its concatenation, scatters and gathers.
constexpr int PAR_MAX_LEVELS = 16;
constexpr int PAR_THREADS = 512;
constexpr int PAR_CHUNK = 32;              // modes a thread at most: n <= PAR_THREADS * PAR_CHUNK = 16384
// P, a mode index, and CH, the modes a thread: <int, 16> up to 8192 modes, <16-bit, 32> above — four arrays a mode
// in shared memory; the 16-bit form, a little slower, only where the int form does not fit
struct ParitySeqs { const int32_t* p[PAR_MAX_LEVELS]; int S[PAR_MAX_LEVELS]; int L; };

// exclusive prefix of v over the block (every thread), `total` the block's sum; wsum: PAR_THREADS / 32 ints
__device__ __forceinline__ int block_excl_scan(int v, int* wsum, int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
  for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, x, o); if (lane >= o) x += y; }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = lane < PAR_THREADS / 32 ? wsum[lane] : 0;
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, w, o); if (lane >= o) w += y; }
    if (lane < PAR_THREADS / 32) wsum[lane] = w;
  }
  __syncthreads();
  const int before = (warp == 0 ? 0 : wsum[warp - 1]) + x - v;
  total = wsum[PAR_THREADS / 32 - 1];
  __syncthreads();
  return before;
}

template <typename P, int CH>
__global__ void __launch_bounds__(PAR_THREADS) parity_kernel(const int8_t* alive, int n, ParitySeqs seqs, int32_t* out) {
  typedef P par_t;
  extern __shared__ __align__(16) unsigned char psh_raw[];
  par_t* psh = reinterpret_cast<par_t*>(psh_raw);
  par_t* perm = psh;                 // n
  par_t* low = perm + n;             // n
  par_t* nxt = low + n;              // n
  par_t* mode_of_rank = nxt + n;     // n
  __shared__ int wsum[PAR_THREADS / 32];
  const int b = blockIdx.x, t = threadIdx.x;
  const int8_t* al = alive + size_t(b) * n;
  // the alive modes' ranks: each thread a contiguous chunk of the modes
  const int chunk = (n + PAR_THREADS - 1) / PAR_THREADS;
  const int i0 = t * chunk, i1 = min(i0 + chunk, n);
  int cnt = 0;
  for (int i = i0; i < i1; ++i) cnt += al[i] != 0;
  int n_alive;
  int r = block_excl_scan(cnt, wsum, n_alive);
  for (int i = i0; i < i1; ++i) { perm[i] = i; if (al[i]) mode_of_rank[r++] = i; }
  // the valid pivots' ranks over the levels' arrays in order, each thread a contiguous chunk of the concatenation
  int off[PAR_MAX_LEVELS + 1];
  off[0] = 0;
  for (int l = 0; l < seqs.L; ++l) off[l + 1] = off[l] + seqs.S[l];
  const int S = off[seqs.L];
  const int ch2 = (S + PAR_THREADS - 1) / PAR_THREADS;
  const int j0 = t * ch2, j1 = min(j0 + ch2, S);
  int l = 0;
  while (l < seqs.L && j0 >= off[l + 1]) ++l;
  int cv = 0, lc = l;
  for (int j = j0; j < j1; ++j) {
    while (j >= off[lc + 1]) ++lc;
    cv += seqs.p[lc][size_t(b) * seqs.S[lc] + (j - off[lc])] >= 0;
  }
  __syncthreads();                       // mode_of_rank complete
  int tot;
  int k = block_excl_scan(cv, wsum, tot);
  lc = l;
  for (int j = j0; j < j1; ++j) {
    while (j >= off[lc + 1]) ++lc;
    const int v = seqs.p[lc][size_t(b) * seqs.S[lc] + (j - off[lc])];
    if (v >= 0) { if (k < n_alive && v < n) perm[mode_of_rank[k]] = par_t(v); ++k; }
  }
  __syncthreads();
  for (int i = i0; i < i1; ++i) { low[i] = i; nxt[i] = perm[i]; }
  __syncthreads();
  int rounds = 1;
  while ((1 << rounds) < n) ++rounds;
  for (int round = 0; round < rounds; ++round) {
    int nl[CH], nn[CH];
#pragma unroll
    for (int u = 0; u < CH; ++u) {
      const int i = i0 + u;
      if (u < chunk && i < i1) { const int x = nxt[i]; nl[u] = min(int(low[i]), int(low[x])); nn[u] = nxt[x]; }
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < CH; ++u) {
      const int i = i0 + u;
      if (u < chunk && i < i1) { low[i] = par_t(nl[u]); nxt[i] = par_t(nn[u]); }
    }
    __syncthreads();
  }
  int c = 0;
  for (int i = i0; i < i1; ++i) c += low[i] == i;
  int cycles;
  block_excl_scan(c, wsum, cycles);
  if (t == 0) out[b] = (n - cycles) & 1;
}

static ffi::Error Parity(cudaStream_t stream, ffi::Buffer<ffi::S8> alive, ffi::RemainingArgs seqs,
                         ffi::Result<ffi::Buffer<ffi::S32>> out) {
  auto d = alive.dimensions();
  const int n = int(d[d.size() - 1]);
  size_t B = 1;
  for (size_t i = 0; i + 1 < d.size(); ++i) B *= d[i];
  if (n > PAR_THREADS * PAR_CHUNK) return ffi::Error::InvalidArgument("parity kernel: too many modes");
  ParitySeqs ps{};
  ps.L = int(seqs.size());
  if (ps.L > PAR_MAX_LEVELS) return ffi::Error::InvalidArgument("parity kernel: too many levels");
  for (int l = 0; l < ps.L; ++l) {
    auto buf = seqs.get<ffi::Buffer<ffi::S32>>(l);
    if (!buf.has_value()) return ffi::Error::InvalidArgument("parity kernel: a pivot array is not int32");
    auto dd = buf.value().dimensions();
    ps.S[l] = int(dd[dd.size() - 1]);
    size_t Bl = 1;
    for (size_t i = 0; i + 1 < dd.size(); ++i) Bl *= dd[i];
    if (Bl != B) return ffi::Error::InvalidArgument("parity kernel: a pivot array's batch differs from alive's");
    ps.p[l] = buf.value().typed_data();
  }
  if (B == 0) return ffi::Error::Success();
  if (n <= PAR_THREADS * PAR_CHUNK / 2)
    parity_kernel<int, PAR_CHUNK / 2><<<B, PAR_THREADS, size_t(4) * n * sizeof(int), stream>>>(alive.typed_data(), n, ps, out->typed_data());
  else
    parity_kernel<unsigned short, PAR_CHUNK><<<B, PAR_THREADS, size_t(4) * n * sizeof(unsigned short), stream>>>(alive.typed_data(), n, ps, out->typed_data());
  auto e = cudaGetLastError();
  return e == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FrontParity, Parity,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S8>>().RemainingArgs()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// whether the parity kernel's four arrays a mode for n modes fit the shared memory front_cuda_init grants it
extern "C" int front_cuda_parity_fits(int n) {
  if (n < 1 || n > PAR_THREADS * PAR_CHUNK) return 0;
  const size_t bytes = size_t(4) * n * (n <= PAR_THREADS * PAR_CHUNK / 2 ? sizeof(int) : sizeof(unsigned short));
  size_t want = size_t(4) * PAR_THREADS * PAR_CHUNK * sizeof(unsigned short);
  if (g_smem_limit - 4096 < int(want)) want = size_t(g_smem_limit - 4096);
  return bytes <= want;
}

// the widest front the shared-memory kernels (forward and backward) take on this device; wider ones go global
extern "C" int front_cuda_max_width() {
  int w = 0;
  for (int n = 1; n <= MAXN; ++n) {
    const int T = n <= 64 ? 64 : n <= 128 ? 256 : 512;
    if (smem_bytes(n, T, 1) <= size_t(g_smem_limit) && smem_bytes(n, T, 0) <= size_t(g_smem_limit)) w = n;
  }
  return w;
}

// dynamic shared memory above 48 KB, once, outside any stream capture
extern "C" int front_cuda_init() {
  int dev = 0, optin = 0, persm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceGetAttribute(&persm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  g_smem_limit = optin;
  g_smem_per_sm = persm;
  const int widths[3] = {64, 256, 512};
  const int ns[3] = {64, 128, MAXN};
  void* ffwd[3] = {(void*)front_fwd_kernel<64>, (void*)front_fwd_kernel<256>,
                   (void*)front_fwd_kernel<512>};
  void* bwd[3] = {(void*)front_bwd_kernel<64>, (void*)front_bwd_kernel<256>, (void*)front_bwd_kernel<512>};
  for (int i = 0; i < 3; ++i) {
    int bytes = int(smem_bytes(ns[i], widths[i], PANEL_MAX));
    if (bytes > optin) bytes = optin;
    // A 256-thread forward may now own any routed width. The backward and the
    // routing bound retain their original full-capacity, width-based choices.
    const int forward_bytes = PIVOTED_BLOCK_THREADS == 256 && widths[i] == 256 ? optin : bytes;
    cudaError_t e = cudaFuncSetAttribute(ffwd[i], cudaFuncAttributeMaxDynamicSharedMemorySize, forward_bytes);
    if (e != cudaSuccess) return 1000 * (i + 1) + int(e);
    e = cudaFuncSetAttribute(bwd[i], cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e != cudaSuccess) return 10000 * (i + 1) + int(e);
  }
#if PIVOTED_BLOCK_THREADS == 128
  // The new forward CTA can own every shared-memory width; backward dispatch and
  // max_width retain their original allocations and device routing bound.
  if (cudaFuncSetAttribute((void*)front_fwd_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           optin) != cudaSuccess) return 8;
#endif
  if (cudaFuncSetAttribute((void*)front_gfwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)
      != cudaSuccess) return 5;
  {   // the parity kernels' four arrays a mode, up to each form's widest configuration, beside the static scan buffer
    int want = 4 * PAR_THREADS * PAR_CHUNK * int(sizeof(unsigned short));   // 128 KB, both forms
    if (want > optin - 4096) want = optin - 4096;
    if (cudaFuncSetAttribute((void*)parity_kernel<int, PAR_CHUNK / 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, want)
        != cudaSuccess) return 9;
    if (cudaFuncSetAttribute((void*)parity_kernel<unsigned short, PAR_CHUNK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             want) != cudaSuccess) return 9;
  }
  if (cudaFuncSetAttribute((void*)front_gbwd_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess ||
      cudaFuncSetAttribute((void*)front_gbwd_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess ||
      cudaFuncSetAttribute((void*)front_gbwd_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess)
    return 6;
  if (cudaFuncSetAttribute((void*)bwd_uu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess)
    return 12;
#if !defined(PIVOTED_REAL_FLOAT)
  if (cudaFuncSetAttribute((void*)bwd_uu_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess)
    return 13;
#endif
  if (cudaFuncSetAttribute((void*)front_fwd_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           optin) != cudaSuccess) return 4;   // the device's own limit
  return 0;
}
