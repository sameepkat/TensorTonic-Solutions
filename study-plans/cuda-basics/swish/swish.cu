#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    const int stride = gridDim.x * blockDim.x; 
    
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i+= stride){
        const float x = input[i];
        output[i] = x / (1.0f + __expf(-x));
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
