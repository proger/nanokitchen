/******************************************************************************
 * Based on code from https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan
 * Copyright (c) 2023, Tri Dao.
 * Apache License: https://github.com/state-spaces/mamba/blob/main/LICENSE (see copy in LICENSE.mamba)
 * Edited by Vol Kyrylov.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <torch/extension.h>

using input_t = float;
using weight_t = float;
using scan_t = float2;
using vec_t = uint4; // For loading 4 items at a time
static_assert(sizeof(vec_t) == 16);

static constexpr int kNThreads = 32;
static constexpr int kNItems = 4;
static_assert(kNItems % 4 == 0);
    // if (params.seqlen <= 128) {
    //     selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    // } else if (params.seqlen <= 256) {
    //     selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    // } else if (params.seqlen <= 512) {
    //     selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    // } else if (params.seqlen <= 1024) {
    //     selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    // } else {
    //     selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    // }
static constexpr int kNBytes = sizeof(input_t);
static_assert(kNBytes == 4);
static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
static constexpr int kNLoads = kNItems / kNElts;
static_assert(kNItems % kNElts == 0);
static_assert(kNLoads == 1);
static_assert(kNItems == 4);
// Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
static constexpr int kMinBlocks = 3;

template<typename scalar_t> struct SSMScanOp;

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(const float2 &ab0, const float2 &ab1) const {
        return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
    }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename scalar_t> struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;
    // Constructor
    __device__ SSMScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;
    }
};

using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>;
using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_DIRECT>;
using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                sizeof(typename BlockLoadVecT::TempStorage),
                                                sizeof(typename BlockStoreT::TempStorage),
                                                sizeof(typename BlockStoreVecT::TempStorage)});
static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);

struct Params {
    int batch;
    int dim;
    int seqlen;
    int n_chunks;
    signed long tokens_batch_stride;
    signed long tokens_d_stride;
    signed long gates_batch_stride;
    signed long gates_d_stride;
    signed long out_batch_stride;
    signed long out_d_stride;
    void* tokens_ptr;
    void* gates_ptr;
    void* x_ptr;
    void* out_ptr;
};

__global__ __launch_bounds__(kNThreads, kMinBlocks)
void simple_scan_kernel(Params params) {
    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load_vec = reinterpret_cast<BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<BlockStoreVecT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<BlockScanT::TempStorage*>(smem_ + kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    input_t *tokens = reinterpret_cast<input_t *>(params.tokens_ptr) + batch_id * params.tokens_batch_stride
        + dim_id * params.tokens_d_stride;
    input_t *gates = reinterpret_cast<input_t *>(params.gates_ptr) + batch_id * params.gates_batch_stride
        + dim_id * params.gates_d_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks;

    constexpr int kChunkSize = kNThreads * kNItems;
    constexpr int state_idx = 0;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t tokens_vals[kNItems], gates_vals[kNItems];
        __syncthreads();

        cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(tokens),
            reinterpret_cast<vec_t(&)[kNLoads]>(tokens_vals)
        );
        tokens += kChunkSize;
        cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_DIRECT>(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(gates),
            reinterpret_cast<vec_t(&)[kNLoads]>(gates_vals)
        );
        gates += kChunkSize;

        float out_vals[kNItems];

        __syncthreads();
        scan_t thread_data[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            thread_data[i] = make_float2(gates_vals[i], tokens_vals[i]);
        }

        // Initialize running total
        // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
        scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);

        SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
        cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>(smem_scan).InclusiveScan(
            thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
        );

        // There's a syncthreads in the scan op, so we don't need to sync here.
        // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
        if (threadIdx.x == 0) {
            smem_running_prefix[state_idx] = prefix_op.running_prefix;
            x[chunk + state_idx] = prefix_op.running_prefix;
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            out_vals[i] = thread_data[i].y;
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();

        input_t write_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { write_vals[i] = out_vals[i]; }
        cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_DIRECT>(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[kNLoads]>(write_vals)
        );
    }
}

void simple_scan_cuda(Params &params, cudaStream_t stream) {
    assert(params.seqlen % (kNThreads * kNItems) == 0);
    constexpr int kSmemSizeTotal = kSmemSize + sizeof(scan_t);
    // printf("smem_size = %d\n", kSmemSizeTotal);
    dim3 grid(params.batch, params.dim);
    if (kSmemSizeTotal >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            simple_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSizeTotal));
    }
    simple_scan_kernel<<<grid, kNThreads, kSmemSizeTotal, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor>
simple_scan_forward(const at::Tensor &tokens, const at::Tensor &gates) {
    auto input_type = tokens.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float);

    TORCH_CHECK(gates.scalar_type() == input_type);
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
    TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
    CHECK_SHAPE(tokens, batch_size, dim, seqlen);
    CHECK_SHAPE(gates, batch_size, dim, seqlen);

    const int n_chunks = (seqlen + 512 - 1) / 512;
    // const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(tokens);
    // Right now tokens have BHL layout and gates has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::zeros_like(gates) + 42;
    at::Tensor x;
    x = torch::zeros({batch_size, dim, n_chunks, 2}, tokens.options().dtype(input_type));

    Params params = {
        .batch = batch_size,
        .dim = dim,
        .seqlen = seqlen,
        .n_chunks = n_chunks,
        .tokens_batch_stride = tokens.stride(0),
        .tokens_d_stride = tokens.stride(1),
        .gates_batch_stride = gates.stride(0),
        .gates_d_stride = gates.stride(1),
        .out_batch_stride = out.stride(0),
        .out_d_stride = out.stride(1),
        .tokens_ptr = tokens.data_ptr(),
        .gates_ptr = gates.data_ptr(),
        .x_ptr = x.data_ptr(),
        .out_ptr = out.data_ptr(),
    };

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)tokens.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    simple_scan_cuda(params, stream);
    std::vector<at::Tensor> result = {out, x};
    return result;
}

