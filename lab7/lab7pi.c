#include <stdio.h>
#include <stdlib.h>

#define blocksize 1024
#define gridsize 1
#define seed 1234

__global__ void piEstimate(int *countStore, int *iterations)
{
	int i, count = 0;
	int itr = &iterations;
	double x, y;
	
	for(i = 0; i < itr; i++)
	{
		x = (rand() % (itr + 1)) / (double)itr;
		y = (rand() % (itr + 1)) / (double)itr;
		
		if(((x * x) + (y * y)) <= 1)
			count++;
		
		countStore[(blockIdx.x * blockDim.x) + threadIdx.x] = count;
	}
}

int main(int argc, char **argv)
{
	int iterationsHost, *iterationsDev;
	int *countStoreHost, *countStoreDev;
	
	srand(seed);
	
	printf("Enter iterations: ");
	scanf("%d", &iterationsHost);
	
	countStoreHost = (int *)calloc(blocksize, sizeof(int));
	cudaMalloc((void **)&countStoreDev, sizeof(int) * blocksize);
	cudaMalloc((void **)&iterationsDev, sizeof(int));
	
	cudaMemcpy(countStoreDev, countStoreHost, sizeof(int) * blocksize, cudaMemcpyHostToDevice);
	cudaMemcpy(iterationsDev, &iterationsHost, sizeof(int), cudaMemcpyHostToDevice);
	piEstimate<<<gridsize, blocksize>>>(countStoreDev, iterationsDev);
	cudaMemcpy(countStoreHost, countStoreDev, sizeof(int) * blocksize, cudaMemcpyDeviceToHost);
	
	int i;
	double pi = 0.0;
	for(i = 0; i < blocksize; i++)
		pi += countStoreHost[i];
	
	pi = (pi / (blocksize * iterationsHost)) * 4;
	
	printf("Pi estimate for %d iterations = %lf\n", iterationsHost, pi);
	
	return 0;
}