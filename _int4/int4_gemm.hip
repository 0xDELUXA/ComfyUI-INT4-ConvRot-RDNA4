// SPDX-License-Identifier: Apache-2.0
// Part of ComfyUI-INT4-ConvRot-RDNA4 - native int4 ConvRot W4A4 GEMM for AMD RDNA4 (gfx12).
//
// int4 convrot GEMM as a C-ABI shared library, callable from torch via ctypes.
//   D_int32[M][N] = Xint[M][K] @ Wint[N][K]^T  (comfy packed-int4 inputs)
//   out[m][n] = D * x_scale[m] * w_scale[n] (+ bias[n])
// Output dtype: 0 = float32, 1 = bfloat16.
// Fragment/lane layout validated in wmma_tile.hip; correctness in gemm_tile.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v2i load_frag(const int8_t* packed, int r, int rows,
                                         int khalf, int Kp) {
    if (r >= rows) return (v2i){0, 0};
    const int8_t* p = packed + (int64_t)r * Kp + khalf * 8;
    unsigned r0 = 0, r1 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) r0 |= (unsigned)(uint8_t)p[i]     << (8 * i);
    #pragma unroll
    for (int i = 0; i < 4; ++i) r1 |= (unsigned)(uint8_t)p[4 + i] << (8 * i);
    v2i out; out[0] = (int)r0; out[1] = (int)r1; return out;
}

template <typename OutT>
__device__ __forceinline__ void store_out(OutT* out, int64_t idx, float v);
template <> __device__ __forceinline__ void store_out<float>(float* out, int64_t idx, float v) { out[idx] = v; }
template <> __device__ __forceinline__ void store_out<__hip_bfloat16>(__hip_bfloat16* out, int64_t idx, float v) {
    out[idx] = __float2bfloat16(v);
}

template <typename OutT>
__global__ void int4_gemm_kernel(
        const int8_t* __restrict__ Xp, const int8_t* __restrict__ Wp,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        const float* __restrict__ bias, OutT* __restrict__ out,
        int M, int N, int K) {
    int Kp = K / 2;
    int nk32 = K / 32;
    int lane = threadIdx.x % 32;
    int mtile = blockIdx.y * 16;
    int ntile = blockIdx.x * 16;
    int khalf = lane / 16;
    int arow = mtile + (lane % 16);
    int bcol = ntile + (lane % 16);

    v8i acc = (v8i){0,0,0,0,0,0,0,0};
    for (int kk = 0; kk < nk32; ++kk) {
        int kbase = kk * 2 + khalf;
        v2i a = load_frag(Xp, arow, M, kbase, Kp);
        v2i b = load_frag(Wp, bcol, N, kbase, Kp);
        acc = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, acc, false);
    }

    int col = ntile + (lane % 16);
    if (col >= N) return;
    float ws = wscale[col];
    float bs = bias ? bias[col] : 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int row = mtile + e + 8 * (lane / 16);
        if (row >= M) continue;
        float v = (float)acc[e] * xscale[row] * ws + bs;
        store_out<OutT>(out, (int64_t)row * N + col, v);
    }
}

// ---------------------------------------------------------------------------
// Optimized tiled kernel: 64x64 macro-tile, 4 warps (2x2), LDS-staged A/B.
// Each warp owns a 32x32 quadrant = 2x2 WMMA tiles. K stepped by 32.
// LDS holds packed int4 directly (bytes), so global->LDS is plain byte copy and
// WMMA fragments load 8 bytes straight from LDS.
// ---------------------------------------------------------------------------
#define BM 64
#define BN 64
#define BKB 16   // BK/2 bytes per row per k-chunk (BK=32 int4)

__device__ __forceinline__ v2i lds_frag(const int8_t* smem, int row, int khalf) {
    const int8_t* p = smem + row * BKB + khalf * 8;
    unsigned r0 = 0, r1 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) r0 |= (unsigned)(uint8_t)p[i]     << (8 * i);
    #pragma unroll
    for (int i = 0; i < 4; ++i) r1 |= (unsigned)(uint8_t)p[4 + i] << (8 * i);
    v2i out; out[0] = (int)r0; out[1] = (int)r1; return out;
}

template <typename OutT>
__global__ __launch_bounds__(128) void int4_gemm_opt_kernel(
        const int8_t* __restrict__ Xp, const int8_t* __restrict__ Wp,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        const float* __restrict__ bias, OutT* __restrict__ out,
        int M, int N, int K) {
    __shared__ int8_t As[BM * BKB];   // 64*16 = 1024 B
    __shared__ int8_t Bs[BN * BKB];   // 1024 B

    int Kp = K / 2;
    int tid = threadIdx.x;            // 0..127
    int lane = tid % 32;
    int warp = tid / 32;              // 0..3
    int wm = warp / 2, wn = warp % 2; // warp row/col in 2x2
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    v8i acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = (v8i){0,0,0,0,0,0,0,0};

    int nchunk = K / 32;
    // each thread stages one 8-byte segment of As and one of Bs per chunk
    int srow = tid / 2;               // 0..63
    int soff = (tid % 2) * 8;         // 0 or 8

    for (int kc = 0; kc < nchunk; ++kc) {
        int gcol = kc * BKB + soff;   // byte column in global packed row
        // A: row m0+srow
        {
            int gr = m0 + srow;
            uint64_t v = 0;
            if (gr < M) v = *reinterpret_cast<const uint64_t*>(Xp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(As + srow * BKB + soff) = v;
        }
        // B: row n0+srow
        {
            int gr = n0 + srow;
            uint64_t v = 0;
            if (gr < N) v = *reinterpret_cast<const uint64_t*>(Wp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(Bs + srow * BKB + soff) = v;
        }
        __syncthreads();

        int khalf = lane / 16;
        int arow_base = wm * 32 + (lane % 16);
        int bcol_base = wn * 32 + (lane % 16);
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            v2i a = lds_frag(As, arow_base + ti * 16, khalf);
            #pragma unroll
            for (int tj = 0; tj < 2; ++tj) {
                v2i b = lds_frag(Bs, bcol_base + tj * 16, khalf);
                acc[ti][tj] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, acc[ti][tj], false);
            }
        }
        __syncthreads();
    }

    // store
    #pragma unroll
    for (int tj = 0; tj < 2; ++tj) {
        int col = n0 + wn * 32 + tj * 16 + (lane % 16);
        if (col >= N) continue;
        float ws = wscale[col];
        float bs = bias ? bias[col] : 0.0f;
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int row = m0 + wm * 32 + ti * 16 + e + 8 * (lane / 16);
                if (row >= M) continue;
                float v = (float)acc[ti][tj][e] * xscale[row] * ws + bs;
                store_out<OutT>(out, (int64_t)row * N + col, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 128x128 macro-tile, 8 warps (4x2), LDS-staged. Each warp owns 32 rows x 64
// cols = 2x4 = 8 WMMA tiles. Higher A/B reuse per LDS load than the 64 kernel.
// ---------------------------------------------------------------------------
#define BM2 128
#define BN2 128

template <typename OutT>
__global__ __launch_bounds__(256) void int4_gemm_opt128_kernel(
        const int8_t* __restrict__ Xp, const int8_t* __restrict__ Wp,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        const float* __restrict__ bias, OutT* __restrict__ out,
        int M, int N, int K) {
    __shared__ int8_t As[BM2 * BKB];   // 128*16 = 2048 B
    __shared__ int8_t Bs[BN2 * BKB];   // 2048 B

    int Kp = K / 2;
    int tid = threadIdx.x;             // 0..255
    int lane = tid % 32;
    int warp = tid / 32;               // 0..7
    int wm = warp / 2, wn = warp % 2;  // 4x2 warp grid
    int m0 = blockIdx.y * BM2;
    int n0 = blockIdx.x * BN2;

    v8i acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = (v8i){0,0,0,0,0,0,0,0};

    int nchunk = K / 32;
    int srow = tid / 2;                // 0..127
    int soff = (tid % 2) * 8;

    for (int kc = 0; kc < nchunk; ++kc) {
        int gcol = kc * BKB + soff;
        {
            int gr = m0 + srow;
            uint64_t v = 0;
            if (gr < M) v = *reinterpret_cast<const uint64_t*>(Xp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(As + srow * BKB + soff) = v;
        }
        {
            int gr = n0 + srow;
            uint64_t v = 0;
            if (gr < N) v = *reinterpret_cast<const uint64_t*>(Wp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(Bs + srow * BKB + soff) = v;
        }
        __syncthreads();

        int khalf = lane / 16;
        int arow_base = wm * 32 + (lane % 16);
        int bcol_base = wn * 64 + (lane % 16);
        v2i a0 = lds_frag(As, arow_base + 0, khalf);
        v2i a1 = lds_frag(As, arow_base + 16, khalf);
        #pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            v2i b = lds_frag(Bs, bcol_base + tj * 16, khalf);
            acc[0][tj] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a0, true, b, acc[0][tj], false);
            acc[1][tj] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a1, true, b, acc[1][tj], false);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tj = 0; tj < 4; ++tj) {
        int col = n0 + wn * 64 + tj * 16 + (lane % 16);
        if (col >= N) continue;
        float ws = wscale[col];
        float bs = bias ? bias[col] : 0.0f;
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int row = m0 + wm * 32 + ti * 16 + e + 8 * (lane / 16);
                if (row >= M) continue;
                float v = (float)acc[ti][tj][e] * xscale[row] * ws + bs;
                store_out<OutT>(out, (int64_t)row * N + col, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K int4 GEMM: partition the K reduction across blockIdx.z so small
// M*N tile grids (few output tiles, large K) still fill the GPU. Each block
// accumulates its K-slice and atomicAdds the scaled fp32 partial into a zeroed
// [M,N] fp32 workspace; bias + output-dtype cast are done in torch afterwards.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void int4_gemm_splitk_kernel(
        const int8_t* __restrict__ Xp, const int8_t* __restrict__ Wp,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        float* __restrict__ accum, int M, int N, int K, int nsplit) {
    __shared__ int8_t As[BM2 * BKB];
    __shared__ int8_t Bs[BN2 * BKB];

    int Kp = K / 2;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int wm = warp / 2, wn = warp % 2;
    int m0 = blockIdx.y * BM2;
    int n0 = blockIdx.x * BN2;

    int nchunk = K / 32;
    int chunks_per = (nchunk + nsplit - 1) / nsplit;
    int kc0 = blockIdx.z * chunks_per;
    int kc1 = kc0 + chunks_per; if (kc1 > nchunk) kc1 = nchunk;
    if (kc0 >= kc1) return;

    v8i acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = (v8i){0,0,0,0,0,0,0,0};

    int srow = tid / 2;
    int soff = (tid % 2) * 8;

    for (int kc = kc0; kc < kc1; ++kc) {
        int gcol = kc * BKB + soff;
        {
            int gr = m0 + srow; uint64_t v = 0;
            if (gr < M) v = *reinterpret_cast<const uint64_t*>(Xp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(As + srow * BKB + soff) = v;
        }
        {
            int gr = n0 + srow; uint64_t v = 0;
            if (gr < N) v = *reinterpret_cast<const uint64_t*>(Wp + (int64_t)gr * Kp + gcol);
            *reinterpret_cast<uint64_t*>(Bs + srow * BKB + soff) = v;
        }
        __syncthreads();
        int khalf = lane / 16;
        int arow_base = wm * 32 + (lane % 16);
        int bcol_base = wn * 64 + (lane % 16);
        v2i a0 = lds_frag(As, arow_base + 0, khalf);
        v2i a1 = lds_frag(As, arow_base + 16, khalf);
        #pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            v2i b = lds_frag(Bs, bcol_base + tj * 16, khalf);
            acc[0][tj] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a0, true, b, acc[0][tj], false);
            acc[1][tj] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a1, true, b, acc[1][tj], false);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tj = 0; tj < 4; ++tj) {
        int col = n0 + wn * 64 + tj * 16 + (lane % 16);
        if (col >= N) continue;
        float ws = wscale[col];
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int row = m0 + wm * 32 + ti * 16 + e + 8 * (lane / 16);
                if (row >= M) continue;
                float v = (float)acc[ti][tj][e] * xscale[row] * ws;
                atomicAdd(&accum[(int64_t)row * N + col], v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused ConvRot(256) rotation + rowwise int4 quantize + pack.
//   Regular Hadamard-256 = kron(H4,H4,H4,H4)/16, computed as 4 radix-4 butterfly
//   stages over the base-4 index digits (symmetric, matches eager _rotate_*).
//   One block per row: transform each 256-group in LDS, track rowwise absmax,
//   then quantize (scale=absmax/7, round-nearest-even, clamp[-7,7]) and pack.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float load_in(const void* x, int64_t idx, int in_dtype) {
    if (in_dtype == 0) return reinterpret_cast<const float*>(x)[idx];
    if (in_dtype == 1) return __bfloat162float(reinterpret_cast<const __hip_bfloat16*>(x)[idx]);
    return __half2float(reinterpret_cast<const __half*>(x)[idx]);
}

// Regular Hadamard-G (G in {16,64,256}) = kron(H4,...)/sqrt(G), computed as
// log4(G) radix-4 butterfly stages over the base-4 index digits (symmetric,
// matches eager _rotate_*). One block per row; 256/G groups processed per wave.
__global__ __launch_bounds__(256) void convrot_quant_int4_kernel(
        const void* __restrict__ x, int in_dtype,
        int8_t* __restrict__ qout, float* __restrict__ scaleout,
        int M, int K, int G) {
    const float h4[4][4] = {{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
    __shared__ float g[256];
    extern __shared__ __hip_bfloat16 rowbuf[];   // K bf16 = transformed row

    int row = blockIdx.x;
    int t = threadIdx.x;                          // 0..255
    int nstages = 0; while ((1 << (2 * nstages)) < G) nstages++;   // log4(G)
    int gpw = 256 / G;                            // groups per wave
    int glocal = t / G;                           // which group in this wave
    int e = t % G;                                // element within group
    int gbaseIdx = glocal * G;                    // this group's slice base in g[]
    float norm = rsqrtf((float)G);
    int ngrp = K / G;
    float lmax = 0.0f;

    for (int gbase = 0; gbase < ngrp; gbase += gpw) {
        int grp = gbase + glocal;
        bool active = grp < ngrp;
        g[t] = active ? load_in(x, (int64_t)row * K + (int64_t)grp * G + e, in_dtype) : 0.0f;
        __syncthreads();
        for (int stage = 0; stage < nstages; ++stage) {
            int stride = 1 << (2 * stage);
            int ds = (e / stride) & 3;
            int b = gbaseIdx + (e - ds * stride);
            float v0 = g[b], v1 = g[b + stride], v2 = g[b + 2 * stride], v3 = g[b + 3 * stride];
            float nv = h4[ds][0]*v0 + h4[ds][1]*v1 + h4[ds][2]*v2 + h4[ds][3]*v3;
            __syncthreads();
            g[t] = nv;
            __syncthreads();
        }
        if (active) {
            float tv = g[t] * norm;
            rowbuf[(int64_t)grp * G + e] = __float2bfloat16(tv);
            lmax = fmaxf(lmax, fabsf(__bfloat162float(__float2bfloat16(tv))));
        }
        __syncthreads();
    }

    // block reduce absmax over the 256 threads
    __shared__ float red[256];
    red[t] = lmax; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float rowmax = fmaxf(red[0], 1e-10f);
    float scale = rowmax / 7.0f;
    float inv = 7.0f / rowmax;
    if (t == 0) scaleout[row] = scale;

    // quantize + pack: each thread strides over output bytes [0, K/2)
    int Kp = K / 2;
    for (int jb = t; jb < Kp; jb += 256) {
        float a = __bfloat162float(rowbuf[2 * jb]);
        float b = __bfloat162float(rowbuf[2 * jb + 1]);
        int qa = (int)rintf(a * inv); qa = qa < -7 ? -7 : (qa > 7 ? 7 : qa);
        int qb = (int)rintf(b * inv); qb = qb < -7 ? -7 : (qb > 7 ? 7 : qb);
        qout[(int64_t)row * Kp + jb] = (int8_t)(((qa & 0xF) | ((qb & 0xF) << 4)) & 0xFF);
    }
}

// ===========================================================================
// int8 W8A8 ConvRot GEMM (native wmma_i32_16x16x16_iu8).
//
// Same output contract and tiling as the int4 path, but inputs are plain signed
// int8 [., K] (no nibble packing) so the global row stride is K bytes and each
// WMMA consumes 16 K-elements (K/16 chunks). The 16-byte-per-row LDS staging is
// byte-identical to int4 (int4: 32 nibbles = 16 B; int8: 16 int8 = 16 B), so the
// staging + lds_frag helpers are reused verbatim; only the builtin, chunk count,
// and global stride differ.
// ===========================================================================
template <typename OutT>
__global__ __launch_bounds__(256) void int8_gemm_opt128_kernel(
        const int8_t* __restrict__ X, const int8_t* __restrict__ W,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        const float* __restrict__ bias, OutT* __restrict__ out,
        int M, int N, int K) {
    __shared__ int8_t As[BM2 * BKB];   // 128*16 = 2048 B
    __shared__ int8_t Bs[BN2 * BKB];   // 2048 B

    int tid = threadIdx.x;             // 0..255
    int lane = tid % 32;
    int warp = tid / 32;               // 0..7
    int wm = warp / 2, wn = warp % 2;  // 4x2 warp grid
    int m0 = blockIdx.y * BM2;
    int n0 = blockIdx.x * BN2;

    v8i acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = (v8i){0,0,0,0,0,0,0,0};

    int nchunk = K / 16;               // int8: 16 K-elements (=16 bytes) per chunk
    int srow = tid / 2;                // 0..127
    int soff = (tid % 2) * 8;

    for (int kc = 0; kc < nchunk; ++kc) {
        int gcol = kc * BKB + soff;    // byte column in global row (stride K)
        {
            int gr = m0 + srow;
            uint64_t v = 0;
            if (gr < M) v = *reinterpret_cast<const uint64_t*>(X + (int64_t)gr * K + gcol);
            *reinterpret_cast<uint64_t*>(As + srow * BKB + soff) = v;
        }
        {
            int gr = n0 + srow;
            uint64_t v = 0;
            if (gr < N) v = *reinterpret_cast<const uint64_t*>(W + (int64_t)gr * K + gcol);
            *reinterpret_cast<uint64_t*>(Bs + srow * BKB + soff) = v;
        }
        __syncthreads();

        int khalf = lane / 16;
        int arow_base = wm * 32 + (lane % 16);
        int bcol_base = wn * 64 + (lane % 16);
        v2i a0 = lds_frag(As, arow_base + 0, khalf);
        v2i a1 = lds_frag(As, arow_base + 16, khalf);
        #pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            v2i b = lds_frag(Bs, bcol_base + tj * 16, khalf);
            acc[0][tj] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b, acc[0][tj], false);
            acc[1][tj] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b, acc[1][tj], false);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tj = 0; tj < 4; ++tj) {
        int col = n0 + wn * 64 + tj * 16 + (lane % 16);
        if (col >= N) continue;
        float ws = wscale[col];
        float bs = bias ? bias[col] : 0.0f;
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int row = m0 + wm * 32 + ti * 16 + e + 8 * (lane / 16);
                if (row >= M) continue;
                float v = (float)acc[ti][tj][e] * xscale[row] * ws + bs;
                store_out<OutT>(out, (int64_t)row * N + col, v);
            }
        }
    }
}

template <typename OutT>
__global__ __launch_bounds__(128) void int8_gemm_opt_kernel(
        const int8_t* __restrict__ X, const int8_t* __restrict__ W,
        const float* __restrict__ xscale, const float* __restrict__ wscale,
        const float* __restrict__ bias, OutT* __restrict__ out,
        int M, int N, int K) {
    __shared__ int8_t As[BM * BKB];    // 64*16 = 1024 B
    __shared__ int8_t Bs[BN * BKB];    // 1024 B

    int tid = threadIdx.x;             // 0..127
    int lane = tid % 32;
    int warp = tid / 32;               // 0..3
    int wm = warp / 2, wn = warp % 2;  // 2x2
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    v8i acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = (v8i){0,0,0,0,0,0,0,0};

    int nchunk = K / 16;
    int srow = tid / 2;                // 0..63
    int soff = (tid % 2) * 8;

    for (int kc = 0; kc < nchunk; ++kc) {
        int gcol = kc * BKB + soff;
        {
            int gr = m0 + srow;
            uint64_t v = 0;
            if (gr < M) v = *reinterpret_cast<const uint64_t*>(X + (int64_t)gr * K + gcol);
            *reinterpret_cast<uint64_t*>(As + srow * BKB + soff) = v;
        }
        {
            int gr = n0 + srow;
            uint64_t v = 0;
            if (gr < N) v = *reinterpret_cast<const uint64_t*>(W + (int64_t)gr * K + gcol);
            *reinterpret_cast<uint64_t*>(Bs + srow * BKB + soff) = v;
        }
        __syncthreads();

        int khalf = lane / 16;
        int arow_base = wm * 32 + (lane % 16);
        int bcol_base = wn * 32 + (lane % 16);
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            v2i a = lds_frag(As, arow_base + ti * 16, khalf);
            #pragma unroll
            for (int tj = 0; tj < 2; ++tj) {
                v2i b = lds_frag(Bs, bcol_base + tj * 16, khalf);
                acc[ti][tj] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[ti][tj], false);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tj = 0; tj < 2; ++tj) {
        int col = n0 + wn * 32 + tj * 16 + (lane % 16);
        if (col >= N) continue;
        float ws = wscale[col];
        float bs = bias ? bias[col] : 0.0f;
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int row = m0 + wm * 32 + ti * 16 + e + 8 * (lane / 16);
                if (row >= M) continue;
                float v = (float)acc[ti][tj][e] * xscale[row] * ws + bs;
                store_out<OutT>(out, (int64_t)row * N + col, v);
            }
        }
    }
}

// Fused ConvRot(G) rotation + rowwise int8 quantize (no packing). Identical
// rotation to convrot_quant_int4_kernel; differs only in the tail (scale=absmax/
// 127, plain int8 [M,K] output). One block per row.
__global__ __launch_bounds__(256) void convrot_quant_int8_kernel(
        const void* __restrict__ x, int in_dtype,
        int8_t* __restrict__ qout, float* __restrict__ scaleout,
        int M, int K, int G) {
    const float h4[4][4] = {{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
    __shared__ float g[256];
    extern __shared__ __hip_bfloat16 rowbuf[];   // K bf16 = transformed row

    int row = blockIdx.x;
    int t = threadIdx.x;                          // 0..255
    int nstages = 0; while ((1 << (2 * nstages)) < G) nstages++;   // log4(G)
    int gpw = 256 / G;                            // groups per wave
    int glocal = t / G;
    int e = t % G;
    int gbaseIdx = glocal * G;
    float norm = rsqrtf((float)G);
    int ngrp = K / G;
    float lmax = 0.0f;

    for (int gbase = 0; gbase < ngrp; gbase += gpw) {
        int grp = gbase + glocal;
        bool active = grp < ngrp;
        g[t] = active ? load_in(x, (int64_t)row * K + (int64_t)grp * G + e, in_dtype) : 0.0f;
        __syncthreads();
        for (int stage = 0; stage < nstages; ++stage) {
            int stride = 1 << (2 * stage);
            int ds = (e / stride) & 3;
            int b = gbaseIdx + (e - ds * stride);
            float v0 = g[b], v1 = g[b + stride], v2 = g[b + 2 * stride], v3 = g[b + 3 * stride];
            float nv = h4[ds][0]*v0 + h4[ds][1]*v1 + h4[ds][2]*v2 + h4[ds][3]*v3;
            __syncthreads();
            g[t] = nv;
            __syncthreads();
        }
        if (active) {
            float tv = g[t] * norm;
            rowbuf[(int64_t)grp * G + e] = __float2bfloat16(tv);
            lmax = fmaxf(lmax, fabsf(__bfloat162float(__float2bfloat16(tv))));
        }
        __syncthreads();
    }

    __shared__ float red[256];
    red[t] = lmax; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float rowmax = fmaxf(red[0], 1e-10f);
    float scale = rowmax / 127.0f;
    float inv = 127.0f / rowmax;
    if (t == 0) scaleout[row] = scale;

    for (int j = t; j < K; j += 256) {
        float a = __bfloat162float(rowbuf[j]);
        int qa = (int)rintf(a * inv); qa = qa < -127 ? -127 : (qa > 127 ? 127 : qa);
        qout[(int64_t)row * K + j] = (int8_t)qa;
    }
}

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// Split-K GEMM into a zeroed fp32 [M,N] accum (bias + cast applied in torch).
API int ck_int4_convrot_gemm_splitk(
        const void* Xp, const void* Wp, const void* xscale, const void* wscale,
        void* accum, int M, int N, int K, int nsplit, void* stream) {
    if (K % 32 != 0 || nsplit < 1) return -1;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2, nsplit);
    hipLaunchKernelGGL(int4_gemm_splitk_kernel, grid, dim3(256), 0, s,
                       reinterpret_cast<const int8_t*>(Xp), reinterpret_cast<const int8_t*>(Wp),
                       reinterpret_cast<const float*>(xscale), reinterpret_cast<const float*>(wscale),
                       reinterpret_cast<float*>(accum), M, N, K, nsplit);
    return (int)hipGetLastError();
}

// x: [M,K] in_dtype(0=f32,1=bf16,2=f16). qout: [M,K/2] int8. scaleout: [M] f32.
// Only convrot_groupsize G==256 supported here (the hot path); caller falls
// back to eager for other group sizes.
API int ck_convrot_quant_int4(
        const void* x, int in_dtype, void* qout, void* scaleout,
        int M, int K, int G, void* stream) {
    if (!(G == 16 || G == 64 || G == 256) || K % G != 0) return -1;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    size_t shmem = (size_t)K * sizeof(__hip_bfloat16);
    hipLaunchKernelGGL(convrot_quant_int4_kernel, dim3(M), dim3(256), shmem, s,
                       x, in_dtype, reinterpret_cast<int8_t*>(qout),
                       reinterpret_cast<float*>(scaleout), M, K, G);
    return (int)hipGetLastError();
}

// x: [M,K] in_dtype(0=f32,1=bf16,2=f16). qout: [M,K] int8 (plain, no packing).
// scaleout: [M] f32. G in {16,64,256}.
API int ck_convrot_quant_int8(
        const void* x, int in_dtype, void* qout, void* scaleout,
        int M, int K, int G, void* stream) {
    if (!(G == 16 || G == 64 || G == 256) || K % G != 0) return -1;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    size_t shmem = (size_t)K * sizeof(__hip_bfloat16);
    hipLaunchKernelGGL(convrot_quant_int8_kernel, dim3(M), dim3(256), shmem, s,
                       x, in_dtype, reinterpret_cast<int8_t*>(qout),
                       reinterpret_cast<float*>(scaleout), M, K, G);
    return (int)hipGetLastError();
}

// int8 W8A8 GEMM. X [M,K] int8, W [N,K] int8, xscale[M], wscale[N], bias[N]/null.
// Returns 0 on success, nonzero hipError_t on failure.
API int ck_int8_convrot_gemm(
        const void* X, const void* W, const void* xscale, const void* wscale,
        const void* bias, void* out, int M, int N, int K, int out_dtype,
        void* stream) {
    if (K % 16 != 0) return -1;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    const int8_t* x = reinterpret_cast<const int8_t*>(X);
    const int8_t* w = reinterpret_cast<const int8_t*>(W);
    const float* xs = reinterpret_cast<const float*>(xscale);
    const float* wsc = reinterpret_cast<const float*>(wscale);
    const float* b = reinterpret_cast<const float*>(bias);
    bool big = (M >= 128 && N >= 128);
    if (big) {
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        if (out_dtype == 0)
            hipLaunchKernelGGL(int8_gemm_opt128_kernel<float>, grid, dim3(256), 0, s,
                               x, w, xs, wsc, b, reinterpret_cast<float*>(out), M, N, K);
        else
            hipLaunchKernelGGL(int8_gemm_opt128_kernel<__hip_bfloat16>, grid, dim3(256), 0, s,
                               x, w, xs, wsc, b, reinterpret_cast<__hip_bfloat16*>(out), M, N, K);
    } else {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        if (out_dtype == 0)
            hipLaunchKernelGGL(int8_gemm_opt_kernel<float>, grid, dim3(128), 0, s,
                               x, w, xs, wsc, b, reinterpret_cast<float*>(out), M, N, K);
        else
            hipLaunchKernelGGL(int8_gemm_opt_kernel<__hip_bfloat16>, grid, dim3(128), 0, s,
                               x, w, xs, wsc, b, reinterpret_cast<__hip_bfloat16*>(out), M, N, K);
    }
    return (int)hipGetLastError();
}

// Returns 0 on success, nonzero hipError_t on failure.
API int ck_int4_convrot_gemm(
        const void* Xp, const void* Wp, const void* xscale, const void* wscale,
        const void* bias, void* out, int M, int N, int K, int out_dtype,
        void* stream) {
    if (K % 32 != 0) return -1;
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    const int8_t* xp = reinterpret_cast<const int8_t*>(Xp);
    const int8_t* wp = reinterpret_cast<const int8_t*>(Wp);
    const float* xs = reinterpret_cast<const float*>(xscale);
    const float* wsc = reinterpret_cast<const float*>(wscale);
    const float* b = reinterpret_cast<const float*>(bias);
    bool big = (M >= 128 && N >= 128);
    if (big) {
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        if (out_dtype == 0)
            hipLaunchKernelGGL(int4_gemm_opt128_kernel<float>, grid, dim3(256), 0, s,
                               xp, wp, xs, wsc, b, reinterpret_cast<float*>(out), M, N, K);
        else
            hipLaunchKernelGGL(int4_gemm_opt128_kernel<__hip_bfloat16>, grid, dim3(256), 0, s,
                               xp, wp, xs, wsc, b, reinterpret_cast<__hip_bfloat16*>(out), M, N, K);
    } else {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        if (out_dtype == 0)
            hipLaunchKernelGGL(int4_gemm_opt_kernel<float>, grid, dim3(128), 0, s,
                               xp, wp, xs, wsc, b, reinterpret_cast<float*>(out), M, N, K);
        else
            hipLaunchKernelGGL(int4_gemm_opt_kernel<__hip_bfloat16>, grid, dim3(128), 0, s,
                               xp, wp, xs, wsc, b, reinterpret_cast<__hip_bfloat16*>(out), M, N, K);
    }
    return (int)hipGetLastError();
}
