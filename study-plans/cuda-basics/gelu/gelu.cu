#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i< N ){
        constexpr float inv_sqrt_2 = 0.7071067811865475f;
        output[i]  = 0.5f * input[i] * (1 + erff(input[i] * inv_sqrt_2));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
