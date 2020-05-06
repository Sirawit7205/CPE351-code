#include <stdio.h>

__global__ void addVec(double *a_D, double *c_D)
{
	int t_rank;
	t_rank = blockIdx.x*blockDim.x + threadIdx.x;
	c_D[t_rank] = a_D[t_rank] + 1;
}

int main(int argc, char **argv)
{
	int i;
	int block_size = 1024, grid_size = 1;
	double *c_H, *c_D, *a_H, *a_D;
	
	a_H = (double *)malloc(block_size * sizeof(double));
	c_H = (double *)malloc(block_size * sizeof(double));
	cudaMalloc((void **)&a_D, sizeof(double) * block_size);
	cudaMalloc((void **)&c_D, sizeof(double) * block_size);
	
	for(i = 0; i < block_size; i++)
		a_H[i] = i;
	
	cudaMemcpy(a_D, a_H, sizeof(double) * block_size, cudaMemcpyHostToDevice);
	addVec<<<grid_size, block_size>>>(a_D, c_D);
	cudaMemcpy(c_H, c_D, sizeof(double) * block_size, cudaMemcpyDeviceToHost);
	
	for(i = 0; i < block_size; i++)
		printf("Vector %d is %lf\n", i, c_H[i]);
	
	free(a_H);
	free(c_H);
	cudaFree(a_D);
	cudaFree(c_D);
	
	return 0;
}
