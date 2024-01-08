#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__device__ __inline__ float2 mempty() {
    return make_float2(1.0f, 0.0f);
}

__device__ __inline__ float2 mappend(float2 l, float2 r) {
    return make_float2(l.x * r.x, l.y * r.x + r.y);
}

template <int kNStepsPerThread, int kNWarpsPerBlock>
__global__ void scan(float* gates, float* tokens, float* result) {
    __shared__ float2 warp_last[kNWarpsPerBlock];

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    //
    // Perform a little bit  sequential computation in thread registers.
    // 
    float2 thread_acc[kNStepsPerThread];

    float2 acc = mempty();
    #pragma unroll
    for (int i = 0; i < kNStepsPerThread; ++i) {
        float g = gates[thread_id * kNStepsPerThread + i];
        float t = tokens[thread_id * kNStepsPerThread + i];
        acc = mappend(acc, make_float2(g, t));
        thread_acc[i] = acc;
    }

    __syncthreads();

    //
    // Stitch threads in a warp using shuffling
    //

    #pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2) {
        float recv_x = __shfl_up_sync(0xffffffff, acc.x, delta);
        float recv_y = __shfl_up_sync(0xffffffff, acc.y, delta);

        if (lane_id >= delta) {
            #pragma unroll
            for (int i = 0; i < kNStepsPerThread; ++i) {
                acc = mappend(thread_acc[i], make_float2(recv_x, recv_y));
                thread_acc[i] = acc;
            }
        }
    }
    __syncthreads();

    //
    // Stitch warps in a block using shared memory
    //

    if (lane_id == warpSize - 1) {
        warp_last[warp_id] = acc;
    }

    __syncthreads();

    if constexpr (kNWarpsPerBlock <= 1) {
        // the block is too small: use a sequential scan
        if (thread_id == 0) {
            float2 warp_acc = mempty();
            #pragma unroll
            for (int w = 0; w < kNWarpsPerBlock; ++w) {
                warp_acc = mappend(warp_acc, warp_last[w]);
                warp_last[w] = warp_acc;
            }
        }
    } else {
        // do a warp scan
        if (warp_id == 0) {
            float2 warp_acc = (lane_id < kNWarpsPerBlock) ? warp_last[lane_id] : mempty();

            #pragma unroll
            for (int delta = 1; delta < warpSize; delta *= 2) {
                float recv_x = __shfl_up_sync(0xffffffff, warp_acc.x, delta);
                float recv_y = __shfl_up_sync(0xffffffff, warp_acc.y, delta);

                warp_acc = (lane_id >= delta) ? mappend(warp_acc, make_float2(recv_x, recv_y)) : warp_acc;
            }

            if (lane_id < kNWarpsPerBlock) {
                warp_last[lane_id] = warp_acc;
            }
        }
    }

    __syncthreads();

    //
    // Add the last element of the previous warp to each element of the current warp.
    //

    float2 warp_from_left = (warp_id > 0) ? warp_last[warp_id - 1] : mempty();

    #pragma unroll
    for (int i = 0; i < kNStepsPerThread; ++i) {
        acc = mappend(thread_acc[i], warp_from_left);
        result[thread_id * kNStepsPerThread + i] = acc.y;
    }
}


at::Tensor
warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens) {
    TORCH_CHECK(tokens.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(gates.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
    TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto out = at::zeros_like(tokens) + 42;

    dim3 grid(1,1);

    if (seqlen == 32) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 64) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 128) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 4;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 256) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 8;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 512) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 1024) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 2048) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else if (seqlen == 4096) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        scan<kNStepsPerThread, kNWarpsPerBlock><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float2), stream>>>(
            gates.data_ptr<float>(),
            tokens.data_ptr<float>(),
            out.data_ptr<float>()
        );
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 4096");
    }

    return out;
}
