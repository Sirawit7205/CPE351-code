#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW 5000
#define COL 5000

#define INA "outputA.txt"
#define INB "outputB.txt"
#define OUT "output.txt"

int main(int argc, char *argv[])
{
	//MPI var
	int id;
	int p;
	int msg;
	MPI_Status status;

	//other var
	FILE *fpA, *fpB, *fpOut;
	int *ptrInputA, *ptrInputB, *ptrOutput;
	int i, j;
	int strpoint, worksize;
	double startTime, endTime;

	//init
	MPI_Init(&argc, &argv);

	//get world size and rank
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	//initialization at process 0
	if(id == 0)
	{
		//malloc storages
		ptrInputA = (int *)malloc(ROW * COL * sizeof(int));
		ptrInputB = (int *)malloc(ROW * COL * sizeof(int));

		//open input and output files
		fpA = fopen(INA, "r");
		fpB = fopen(INB, "r");
		fpOut = fopen(OUT, "w+");

		//read inputs to buffer
		for(i = 0; i < ROW * COL; i++)
		{
			fscanf(fpA, "%d", &ptrInputA[i]);
			fscanf(fpB, "%d", &ptrInputB[i]);
		}

		//close input files
		fclose(fpA);
		fclose(fpB);

		//start timer
		startTime = MPI_Wtime();

		//assign work to each node
		worksize = (ROW * COL) / p;
		strpoint = worksize;
		for(i = 1; i < p; i++)
		{
			MPI_Send(&worksize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&ptrInputA[strpoint], worksize, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&ptrInputB[strpoint], worksize, MPI_INT, i, 0, MPI_COMM_WORLD);
			strpoint += worksize;
		}
	}

	//actual calculation
	if(id == 0)
	{
		//final result output buffer
		ptrOutput = (int *)malloc(ROW * COL * sizeof(int));

		//calculate
		for(i = 0; i < worksize; i++)
			ptrOutput[i] = ptrInputA[i] + ptrInputB[i];

		//read result from other nodes
		strpoint = worksize;
		for(i = 1; i < p; i++)
		{
			MPI_Recv(&ptrOutput[strpoint], worksize, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			strpoint += worksize;
		}

		//stop timer
		endTime = MPI_Wtime();

		//write result to file
		for(i = 0; i < ROW * COL; i++)
		{
			fprintf(fpOut, "%d ", ptrOutput[i]);
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
		ptrInputA = (int *)malloc(worksize * sizeof(int));
		ptrInputB = (int *)malloc(worksize * sizeof(int));
		ptrOutput = (int *)malloc(worksize * sizeof(int));

		//receiving data
		MPI_Recv(ptrInputA, worksize, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(ptrInputB, worksize, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

		//calculate
		for(i = 0; i < worksize; i++)
			ptrOutput[i] = ptrInputA[i] + ptrInputB[i];

		//send result back
		MPI_Send(ptrOutput, worksize, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

//stop
MPI_Finalize();
}
