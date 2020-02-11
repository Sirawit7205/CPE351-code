#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define INA "matAlarge.txt"
#define INB "matBlarge.txt"
#define OUT "outputlarge.txt"

int main(int argc, char *argv[])
{
	//MPI var
	int id;
	int p;
	int msg;
	MPI_Status status;

	//other var
	FILE *fpA, *fpB, *fpOut;
	float *ptrInputA, *ptrInputB, *ptrOutput;
	int i, j;
	int ROW, COL;
	int strpoint, worksize, remaining_worksize;
	double startTime, endTime;

	//init
	MPI_Init(&argc, &argv);

	//get world size and rank
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	//initialization at process 0
	if(id == 0)
	{
		//open input and output files
		fpA = fopen(INA, "r");
		fpB = fopen(INB, "r");
		fpOut = fopen(OUT, "w+");

		//read input size (read twice to skip headers on file B)
		fscanf(fpA, "%d", &ROW);
		fscanf(fpA, "%d", &COL);
		fscanf(fpB, "%d", &ROW);
		fscanf(fpB, "%d", &COL);

		//malloc storages
		ptrInputA = (float *)malloc(ROW * COL * sizeof(float));
		ptrInputB = (float *)malloc(ROW * COL * sizeof(float));

		//final result output buffer
		ptrOutput = (float *)malloc(ROW * COL * sizeof(float));

		//read inputs to buffer
		for(i = 0; i < ROW * COL; i++)
		{
			fscanf(fpA, "%f", &ptrInputA[i]);
			fscanf(fpB, "%f", &ptrInputB[i]);
		}

		//close input files
		fclose(fpA);
		fclose(fpB);

		//start timer
		startTime = MPI_Wtime();

		//assign work to each node, last node will get the remaining work
		worksize = (ROW * COL) / p;
		remaining_worksize = worksize + ((ROW * COL) - (worksize * p));
		strpoint = worksize;
		for(i = 1; i < p; i++)
		{
			//last node worksize is the largest
			if(i == p - 1)
			{
				MPI_Send(&remaining_worksize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&ptrInputA[strpoint], remaining_worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&ptrInputB[strpoint], remaining_worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
			//the rest are normal size
			else
			{
				MPI_Send(&worksize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&ptrInputA[strpoint], worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&ptrInputB[strpoint], worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}

			//move to the next memory starting point
			strpoint += worksize;
		}
	}

	//actual calculation
	if(id == 0)
	{
		//calculate
		for(i = 0; i < worksize; i++)
			ptrOutput[i] = ptrInputA[i] + ptrInputB[i];

		//read result from other nodes
		strpoint = worksize;
		for(i = 1; i < p; i++)
		{	
			//last node worksize is the largest
			if(i == p - 1)
			{
				MPI_Recv(&ptrOutput[strpoint], remaining_worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			}
			//the rest are normal size
			else
			{
				MPI_Recv(&ptrOutput[strpoint], worksize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
			}
			
			//move to the next memory starting point
			strpoint += worksize;
		}

		//stop timer
		endTime = MPI_Wtime();

		//write header to file
		fprintf(fpOut, "%d %d\n", ROW, COL);

		//write result to file
		for(i = 0; i < ROW * COL; i++)
		{
			fprintf(fpOut, "%.1f ", ptrOutput[i]);
			if(i % COL == (COL - 1))
				fprintf(fpOut, "\n");
		}

		//close output file
		fclose(fpOut);

		//print result
		printf("Done. Time usage is %lf\n", endTime - startTime);

	}
	else
	{
		//receiving worksize
		MPI_Recv(&worksize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		//malloc input and output buffer
		ptrInputA = (float *)malloc(worksize * sizeof(float));
		ptrInputB = (float *)malloc(worksize * sizeof(float));
		ptrOutput = (float *)malloc(worksize * sizeof(float));

		//receiving data
		MPI_Recv(ptrInputA, worksize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(ptrInputB, worksize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

		//calculate
		for(i = 0; i < worksize; i++)
			ptrOutput[i] = ptrInputA[i] + ptrInputB[i];

		//send result back
		MPI_Send(ptrOutput, worksize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

//stop
MPI_Finalize();
}
