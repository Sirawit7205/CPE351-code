#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define Nblock 1000
#define Nthread 1000
#define Ngrid 1

__global__ void setup(curandState *state){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(9999, index, 0, &state[index]);
}

__global__ void piEs(double *sum , int iteration, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int count = 0;
    for(i=0; i < iteration ;i++){
        double x = curand_uniform_double(&state[index]);
        double y = curand_uniform_double(&state[index]);
        if(x*x + y*y <= 1.0)
            count++;
    }
    sum[index] = 4.0*count / iteration;
}

int main(void){
	// dim3 dimGrid(Nblock,1,1);  // Grid dimensions
	// dim3 dimBlock(Nthread,1,1);  // Block dimensions
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    int i;
    int Athread = Nblock * Nthread;
    int iter = 1000000;
    double pi = 0.0;

    double *piSumHost =  (double *)malloc(Athread * sizeof(double));
    double *piSumDev;
    cudaMalloc((void**)&piSumDev, Athread);
    cudaMemset(piSumDev, 0, Athread);

    curandState *stateDev;
    cudaMalloc((void **)&stateDev, Athread);
	cudaEventRecord(start);
    setup <<<Ngrid, Nblock, Nthread>>> (stateDev);
    piEs  <<<Ngrid, Nblock, Nthread>>> (piSumDev, iter, stateDev);
	cudaEventRecord(stop);
    cudaMemcpy(piSumHost, piSumDev, Athread, cudaMemcpyDeviceToHost);

	float runningTime = 0;
	cudaEventElapsedTime(&runningTime, start, stop);
	printf("CUDA done, took %f ms\n", runningTime);

    for(i = 0; i<Athread; i++){
        pi += piSumHost[i];
    }
    pi /= Nblock;

    printf("Pi estimate = %.10lf\n", pi);

   free(piSumHost);
   cudaFree(piSumDev);
   return 0;
}