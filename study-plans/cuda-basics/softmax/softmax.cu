// Written for the current CUDA C++ programming model

#include <cuda_runtime.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int N
) {
    // solve() launches multiple blocks, but one block must own the whole row.
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ float shared[BLOCK_SIZE];

    const int tid = threadIdx.x;

    // Pass 1: global maximum.
    float local_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] =
                fmaxf(shared[tid], shared[tid + stride]);
        }

        __syncthreads();
    }

    const float max_value = shared[0];

    // Pass 2: exponentials and normalization sum.
    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        const float value = expf(input[i] - max_value);
        output[i] = value;
        local_sum += value;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    const float inverse_sum = 1.0f / shared[0];

    // Pass 3: normalize.
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] *= inverse_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}