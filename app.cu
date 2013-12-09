#include <iostream>
#include "cudalist.cuh"

#define GRIDDIM 256
#define BLOCKDIM 64

__global__
void internal_memory(float * result) {

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    float memory[BLOCKDIM];
    culist<float, int> list(memory, blockDim.x);
    
    for (int i = 0; i < 100000; ++i) {
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
        }
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.pop_front();
            list.pop_back();
        }
    }
    
    for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
    }
    
    result[thid] = list[0];
}

__global__
void external_memory(float * result, float * memory) {

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    culist<float, int> list(memory+thid*blockDim.x, blockDim.x);
    
    for (int i = 0; i < 100000; ++i) {
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
        }
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.pop_front();
            list.pop_back();
        }
    }
    
    for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
    }
    
    result[thid] = list[0];
}

__global__
void shared_memory(float * result) {

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ float memory [BLOCKDIM*BLOCKDIM];
    culist<float, int> list(memory+threadIdx.x*blockDim.x, blockDim.x);
    
    for (int i = 0; i < 100000; ++i) {
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
        }
    
        for (int m = 0; m < blockDim.x/2; ++m) {
            list.pop_front();
            list.pop_back();
        }
    }
    
    for (int m = 0; m < blockDim.x/2; ++m) {
            list.push_front(threadIdx.x+m);
            list.push_back(threadIdx.x+m);
    }
    
    result[thid] = list[0];
}

int main() {

    float *Memory = NULL, *Result = NULL, *result = new float[GRIDDIM*BLOCKDIM];
    cudaMalloc(&Memory, sizeof(float)*GRIDDIM*BLOCKDIM*BLOCKDIM);
    cudaMalloc(&Result, sizeof(float)*GRIDDIM*BLOCKDIM);

    internal_memory<<<GRIDDIM, BLOCKDIM>>>(Result);                // fastest
    //external_memory<<<GRIDDIM, BLOCKDIM>>>(Result, Memory);      // meh
    //shared_memory<<<GRIDDIM, BLOCKDIM>>>(Result);                // meh^2

    cudaMemcpy(result, Result, sizeof(float)*GRIDDIM*BLOCKDIM, 
               cudaMemcpyDeviceToHost);
    
    for (int m = 0; m < GRIDDIM*BLOCKDIM; ++m)
        std::cout << result[m] << std::endl;
}
