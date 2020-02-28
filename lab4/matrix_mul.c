#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MatrixA "matAsmall.txt"
#define MatrixB "matBsmall.txt"
#define MatrixOut "matout.txt"

//prototypes
void getWorksize(int procCount, int rowCount, int colCount, int *worksize, int *totalWorksize, int *displacement);
int getMatrixSize(int *rowA, int *colA, int *rowB, int *colB);
double** malloc2D(int row, int col);
void readInput(double **matA, double **matB);

int main(int argc, char* argv[])
{
  //mpi var
  MPI_Status status;
  MPI_Request request[10];

  //other var
  int p, id;
  int i;
  int rowA, colA, rowB, colB;
  double **matA, **matB, **matALocal;
  int *worksize, *totalWorksize, *displace;
  int localWorksize;

  //get arguments
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  //init by main process
  if(id == 0)
  {
    //get matrix size if the size if valid
    if(getMatrixSize(&rowA, &colA, &rowB, &colB) == -1)
    {
      printf("Matrix size error.\n");
      return 0;
    }
    printf("Matrix OK. Size of matrix: %dx%d and %dx%d\n", rowA, colA, rowB, colB);

    //malloc data, NOTE: B is transposed
    matA = malloc2D(rowA, colA);
    matB = malloc2D(colB, rowB);

    //read data to matrix
    readInput(matA, matB);
    printf("Matrix read OK.\n");

    //malloc worksize and displacement
    worksize = (int *)malloc(p * sizeof(int));
    totalWorksize = (int *)malloc(p * sizeof(int));
    displace = (int *)malloc(p * sizeof(int));

    //calculate size of work
    //getWorksize(p, rowA, colA, worksize, totalWorksize, displace);
    //calculate size
    int size = rowA / p;
    int rem = rowA % p;
    int curDisp = 0;

    //fill worksize and displacement arrays
    for(i = 0; i < p; i++)
    {
      //if this process has to handle remaining work
      if(i < rem)
        worksize[i] = size + 1;
      else
        worksize[i] = size;

      //calculate total worksize for Iscatterv
      totalWorksize[i] = worksize[i] * colA;
      
      //assign displacement then move to the next starting point
      displace[i] = curDisp;
      
      //next displacement is larger than normal
      if(i < rem)
        curDisp += (size + 1) * colA;
      else
        curDisp += size * colA;
    }
    printf("Worksize calculated.\n");
  }

  //send and receive the input data
  //broadcast matrix B sizing data
  MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //malloc location for matrix B
  if(id != 0)
    matB = malloc2D(colB, rowB);
  
  //broadcast matrix B
  MPI_Ibcast(&matB[0][0], rowB * colB, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request[2]);
  MPI_Wait(&request[2], &status);

  //broadcast matrix A column size
  MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //scatter matrix A row size (worksize)
  MPI_Scatter(&worksize[0], 1, MPI_INT, &localWorksize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //malloc location for matrix A
  matALocal = malloc2D(localWorksize, colA);

  //scatter matrix A
  MPI_Scatterv(&matA[0][0], &totalWorksize[0], &displace[0], MPI_DOUBLE, &matALocal[0][0], localWorksize * colA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //printf("Process %d: Data in matrix A is %lf\n", id, matA[10][10]);

  MPI_Finalize();
  return 0;
}

void getWorksize(int procCount, int rowCount, int colCount, int *worksize, int *totalWorksize, int *displacement)
{
  //calculate size
  int size = rowCount / procCount;
  int rem = rowCount % procCount;
  int i, curDisp = 0;

  //fill worksize and displacement arrays
  for(i = 0; i < procCount; i++)
  {
    //if this process has to handle remaining work
    if(i < rem)
      worksize[i] = size + 1;
    else
      worksize[i] = size;

    //calculate total worksize for Iscatterv
    totalWorksize[i] = worksize[i] * colCount;
    
    //assign displacement then move to the next starting point
    displacement[i] = curDisp;
    
    //next displacement is larger than normal
    if(i < rem)
      curDisp += (size + 1) * colCount;
    else
      curDisp += size * colCount;
  }
}

int getMatrixSize(int *rowA, int *colA, int *rowB, int *colB)
{
  FILE *fpA, *fpB;

  //open file
  fpA = fopen(MatrixA, "r");
  fpB = fopen(MatrixB, "r");

  //read size
  fscanf(fpA, "%d %d", rowA, colA);
  fscanf(fpB, "%d %d", rowB, colB);

  //close file
  fclose(fpA);
  fclose(fpB);

  //check validity
  if(*colA == *rowB)
    return 0;
  else
    return -1;
}

double** malloc2D(int row, int col)
{
  double *data = malloc(col * row * sizeof(double));
  double **arr = malloc(row * sizeof(double));
  int i;

  for (i = 0; i < row; i++)
      arr[i] = &(data[col * i]);
  return arr;
}

void readInput(double **matA, double **matB)
{
  FILE *fpA, *fpB;
  int rowA, colA, rowB, colB;
  int i, j;

  //open file
  fpA = fopen(MatrixA, "r");
  fpB = fopen(MatrixB, "r");

  //read input sizes
  fscanf(fpA, "%d %d", &rowA, &colA);
  fscanf(fpB, "%d %d", &rowB, &colB);

  //read data from file
  for(i = 0; i < rowA; i++)
  {
    for(j = 0; j < colA; j++)
    {
      fscanf(fpA, "%lf", &matA[i][j]);
    }
  }

  //NOTE: the row and col of B matrix is reversed (transpose) so the data is easier to work with
  for(i = 0; i < colB; i++)
  {
    for(j = 0; j < rowB; j++)
    {
      fscanf(fpB, "%lf", &matB[i][j]);
    }
  }

  //cleanup
  fclose(fpA);
  fclose(fpB);
}