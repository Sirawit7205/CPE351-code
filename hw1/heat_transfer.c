#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT argv[1]
#define OUTPUT argv[2]

#define matrixType double
#define matrixPtrType double

matrixPtrType **malloc2D(int row, int col);
void checkMatrixSize(char filename[], int *row, int *col);
void readMatrix(char filename[], matrixPtrType **mat);
void writeMatrix(char filename[], matrixPtrType **mat, int row, int col);

int main(int argc, char *argv[])
{
  //mpi vars
  int procCount, procId;

  //other vars
  int row, col;

  //init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);
  MPI_Comm_rank(MPI_COMM_WORLD, &procId);

  //get input size
  checkMatrixSize(INPUT, &row, &col);

  //malloc storage
  matrixPtrType **inputMat = malloc2D(row, col);

  //read input
  readMatrix(INPUT, inputMat);
  
  //calculate worksize
  int worksize, worksizeRem;
  int *assignRow, *assignSize, *assignDispl;
  if(procId == 0)
  {
    worksize = row / procCount;
    worksizeRem = row % procCount;

    assignRow = (int *)malloc(sizeof(int) * procCount);
    assignSize = (int *)malloc(sizeof(int) * procCount);
    assignDispl = (int *)malloc(sizeof(int) * procCount);

    int i;
    for(i = 0; i < procCount; i++)
    {
      //amount of rows
      if(i < worksizeRem)
        assignRow[i] = worksize + 1;
      else
        assignRow[i] = worksize;

      //amount of elements
      assignSize[i] = assignRow[i] * col;

      //displacements
      if(i == 0)
        assignDispl[i] = 0;
      else
        assignDispl[i] = assignDispl[i-1] + assignSize[i-1];

      //printf("For process %d, rows = %d size = %d displ = %d\n", i, assignRow[i], assignSize[i], assignDispl[i]);
    }
  }
}

matrixPtrType **malloc2D(int row, int col)
{
  matrixType *data = malloc(sizeof(matrixType) * row * col);
  matrixPtrType **ptr2d = malloc(sizeof(matrixPtrType) * row);

  int i;
  for(i = 0; i < row; i++)
    ptr2d[i] = &data[i * col];
  
  return ptr2d;
}

void checkMatrixSize(char filename[], int *row, int *col)
{
  FILE *fp;

  fp = fopen(filename, "r");
  fscanf(fp, "%d %d", row, col);
  fclose(fp);
}

void readMatrix(char filename[], matrixPtrType **mat)
{
  int row, col;

  FILE *fp;
  fp = fopen(filename, "r");

  fscanf(fp, "%d %d", &row, &col);

  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fscanf(fp, "%lf", &mat[i][j]);
    }
  }

  fclose(fp);
}

void writeMatrix(char filename[], matrixPtrType **mat, int row, int col)
{
  FILE *fp;
  fp = fopen(filename, "w+");

  fprintf(fp, "%d %d\n", row, col);

  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fprintf(fp, "%lf ", mat[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}