#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define Nblock 1024
#define Nthread 100
#define Ngrid 1
#define maxRound 5000

__global__ void setup(curandState *state){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(9999, index, 0, &state[index]);
}

__global__ void piEs(double *sum , int iteration, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int count;
    for(i=iteration; i-- ;){
        double x = curand_uniform_double(&state[index]);
        double y = curand_uniform_double(&state[index]);
        if(x*x + y*y <= 1.0)
            count++;
    }
    sum[index] = 4.0 * count / iteration;
}

int main(void){
    int i;
    int iter = 100000;
    int Athread = Nblock * Nthread;
    double step = iter / Athread;
    double pi = 0.0;

    float *piSumHost = (float *)malloc(Athread * sizeof(double));
    double *piSumDev;
    cudaMalloc((void**)&piSumDev, Athread);
    cudaMemset(piSumDev, 0, Athread);

    curandState *stateDev;
    cudaMalloc((void **)&stateDev, Athread);

    setup <<<Ngrid, Nblock>>> (stateDev);
    piEs  <<<Ngrid, Nblock>>> (piSumDev, iter, stateDev);

    cudaMemcpy(piSumHost, piSumDev, Athread, cudaMemcpyDeviceToHost);

    for(i = iter; i--;)
        pi += piSumHost[i];
    pi *= step;

    printf("Pi estimate = %.10lf\n", pi);

   free(piSumHost);
   cudaFree(piSumDev);
   return 0;
}