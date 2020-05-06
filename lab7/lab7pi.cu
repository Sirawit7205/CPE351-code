//60070501054
//60070501064
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define blocksize 1024
#define gridsize 1
#define threadsize 1024

__global__ void piEstimate(long long int *countStore, int *iterations)
{
	int rank = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = 0;
	long long int count = 0;
	int itr = iterations[0];
	double x, y;
	
	curandState state;
	curand_init(rank, 0, 0, &state);
	
	while(i < itr)
	{
		x = curand_uniform_double(&state);	
		y = curand_uniform_double(&state);	

		if(((x * x) + (y * y)) <= 1.0)
			count++;
			
		i++;
	}
	countStore[rank] +=count;
}

int main(int argc, char **argv)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int iterationsHost, *iterationsDev;
	long long int *countStoreHost, *countStoreDev;
	
	printf("Enter iterations: ");
	scanf("%d", &iterationsHost);
	
	countStoreHost = (long long int *)calloc(blocksize, sizeof(long long int));
	cudaMalloc((void **)&countStoreDev, sizeof(long long int) * blocksize);
	cudaMalloc((void **)&iterationsDev, sizeof(int));
	
	int i;
	
	cudaMemcpy(countStoreDev, countStoreHost, sizeof(long long int) * blocksize, cudaMemcpyHostToDevice);
	cudaMemcpy(iterationsDev, &iterationsHost, sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	piEstimate<<<gridsize, blocksize, threadsize>>>(countStoreDev, iterationsDev);
	cudaEventRecord(stop);
	cudaMemcpy(countStoreHost, countStoreDev, sizeof(long long int) * blocksize, cudaMemcpyDeviceToHost);
	
	float runningTime = 0;
	cudaEventElapsedTime(&runningTime, start, stop);
	printf("CUDA done, took %f ms\n", runningTime);
	
	/*int max = max_per_round;
	roundHost = iterationsHost / max;
	int rem = iterationsHost % max;
	for(i = 0; i < roundHost; i++)
	{
	//printf("Start round %d\n", i);
		cudaMemcpy(countStoreDev, countStoreHost, sizeof(long long int) * blocksize, cudaMemcpyHostToDevice);
		cudaMemcpy(roundDev, &i, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(iterationsDev, &max, sizeof(int), cudaMemcpyHostToDevice);
		piEstimate<<<gridsize, blocksize>>>(countStoreDev, roundDev, iterationsDev);
		cudaMemcpy(countStoreHost, countStoreDev, sizeof(long long int) * blocksize, cudaMemcpyDeviceToHost);
	}
	
	if(rem != 0)
	{
	//printf("Start rem round with %d iterations\n", rem);
		cudaMemcpy(countStoreDev, countStoreHost, sizeof(long long int) * blocksize, cudaMemcpyHostToDevice);
		cudaMemcpy(roundDev, &i, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(iterationsDev, &rem, sizeof(int), cudaMemcpyHostToDevice);
		piEstimate<<<gridsize, blocksize>>>(countStoreDev, roundDev, iterationsDev);
		cudaMemcpy(countStoreHost, countStoreDev, sizeof(long long int) * blocksize, cudaMemcpyDeviceToHost);
	}*/
	
	double pi = 0.0;
	for(i = 0; i < blocksize; i++)
		pi += countStoreHost[i];

	
	long long int totalPlot = blocksize * iterationsHost;
	pi = (pi / totalPlot) * 4;
	
	printf("Pi estimate for %d iterations = %.10lf\n", iterationsHost, pi);
	
	free(countStoreHost);
	cudaFree(countStoreDev);
	cudaFree(iterationsDev);
	
	return 0;
}