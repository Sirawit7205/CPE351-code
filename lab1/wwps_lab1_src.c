// 60070501054
// 60070501064

#include <mpi.h>
#include <stdio.h>


int main(int argc, char *argv[])
{
	int id;
	int p;
	int i = 0;
	int msg;
	MPI_Status status;

	//init
	MPI_Init(&argc, &argv);

	//get size and rank
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(id == 0)
	{
		for(i = 1; i < p; i++)
		{
			MPI_Recv(&msg, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			printf("Hello rank 0, I'm rank %d\n", msg);
		}

	}
	else
	{
		MPI_Send(&id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	//stop
	MPI_Finalize();
}
