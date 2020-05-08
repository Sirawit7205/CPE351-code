#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//#define DEBUG
#define TIME_CAPTURE

#define MAT_A argv[1]
#define MAT_B argv[2]
#define MAT_OUT argv[3]

float **malloc2D(int row, int col)
{
  /*all memory required as a 1D array*/
  float *data1d = malloc(sizeof(float) * row * col);
  /*memory required for row pointers, 
    use sizeof(long) because float cannot hold 64-bit address space*/
  float **rowPtr = malloc(sizeof(long) * row);

  int i;
  for(i = 0; i < row; i++)
  {
    /*each pointer points to address of data1d*/
    rowPtr[i] = &(data1d[i * col]);
  }

  return rowPtr;
}

void checkMatrixSize(char *filename, int *row, int *col)
{
  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%d %d", row, col);
  fclose(fp);
}

void readFile(char *filename, float **buffer)
{
  FILE *fp = fopen(filename, "r");

  /*get matrix size*/
  int row, col;
  fscanf(fp, "%d %d", &row, &col);

  /*scanning*/
  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fscanf(fp, "%f", &buffer[i][j]);
    }
  }
  fclose(fp);
}

float StringToFloat(char *input)
{
  float value = 0.0, decimalMuliplier = 0.1, minusMultiplier = 1.0;
  int i = 0, decimalReached = 0;
  while('\0' != input[i])
  { 
    /*just skip minus sign for now*/
    if(input[i] == '-')
      minusMultiplier *= -1;
    /*when we reached decimal, formula will change*/
    else if(input[i] == '.')
      decimalReached = 1;
    /*left shift the previous value*/
    else if(0 == decimalReached)
      value = (value * 10) + (input[i] - '0');
    /*add decimal to current value*/
    else
    {
      value += (decimalMuliplier * (input[i] - '0'));
      decimalMuliplier /= 10;
    }
    i++;
  }

  return value * minusMultiplier;
}

void ReadFileWithString(char *filename, float **buffer)
{
  FILE *fp = fopen(filename, "r");

  /*get matrix size*/
  int row, col;
  fscanf(fp, "%d %d", &row, &col);

  /*scanning*/
  int i, j;
  char stringBuffer[20]={};
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fscanf(fp, "%s", stringBuffer);
      buffer[i][j] = StringToFloat(stringBuffer);
    }
  }
  fclose(fp);
}

void writeFile(char *filename, float **matrixOut, int row, int col)
{
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "%d %d\n", row, col);

  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fprintf(fp, "%.1f ", matrixOut[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}

void matrixAddition(float **matrixA, float **matrixB, float **matrixOut, int row, int col)
{
  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      matrixOut[i][j] = matrixA[i][j] + matrixB[i][j];
    }
  }
}

int main(int argc, char *argv[])
{
  /*vars that will be used across ranks*/
  int i, j;
  float **matrixA, **matrixB, **matrixOut;
  float **matrixALocal, **matrixBLocal, **matrixOutLocal;
  int row, col;

  /*timing*/
#ifdef TIME_CAPTURE
  double startTime, endTime;
#endif

  /*init mpi*/
  MPI_Init(&argc, &argv);
  
  /*get size and rank*/
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*let rank 0 read file and assign work,
    all other ranks calculate*/
  int normalJobSize, remainJobSize;
  int *count, *displ;
  if (0 == rank)
  {

#ifdef TIME_CAPTURE
    startTime = MPI_Wtime();
#endif

    /*input data size*/
    checkMatrixSize(MAT_A, &row, &col);

#ifdef DEBUG
    printf("[Rank 0] Input size = %dx%d\n", row, col);
#endif

    /*malloc space*/
    matrixA = malloc2D(row, col);
    matrixB = malloc2D(row, col);
    matrixOut = malloc2D(row, col);

    /*read file*/
    ReadFileWithString(MAT_A, matrixA);
    ReadFileWithString(MAT_B, matrixB);

#ifdef DEBUG
    printf("[Rank 0] Input read\n");
#endif

    /*calculate job size*/
    normalJobSize = (row / size) * col;
    remainJobSize = ((row / size) + (row % size)) * col;

#ifdef DEBUG
    printf("[Rank 0] job sizes are %d and %d\n", normalJobSize, remainJobSize);
#endif

    /*prepare scatter parameters*/
    int currentDispl = 0;
    count = (int *)malloc(sizeof(int) * size);
    displ = (int *)malloc(sizeof(int) * size);

    for(i = 0; i < size; i++)
    {
      count[i] = (size - 1 == i) ? remainJobSize : normalJobSize;
      displ[i] = currentDispl;
      currentDispl += normalJobSize;
    }
  }

  /*broadcast size first so all ranks can allocate memory*/
  MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] broadcasted size %dx%d\n", rank, row, col);
#endif

  /*malloc local storage*/
  int localRow = (rank == size - 1) ? ((row / size) + (row % size)) : (row / size);
  matrixALocal = malloc2D(localRow, col);
  matrixBLocal = malloc2D(localRow, col);
  matrixOutLocal = malloc2D(localRow, col);

#ifdef DEBUG
  printf("[Rank %d] local allocated size %dx%d\n", rank, localRow, col);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  /*scatter actual data*/
  MPI_Scatterv(&matrixA[0][0], count, displ, MPI_FLOAT, &matrixALocal[0][0], localRow * col, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(&matrixB[0][0], count, displ, MPI_FLOAT, &matrixBLocal[0][0], localRow * col, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Scattered. Test= %f %f\n", rank, matrixALocal[0][0], matrixBLocal[0][0]);
#endif


  /*actual calculation*/
  matrixAddition(matrixALocal, matrixBLocal, matrixOutLocal, localRow, col);

#ifdef DEBUG
    printf("[Rank %d] Calculated.\n", rank);
#endif

  /*send data back*/
  MPI_Gatherv(&matrixOutLocal[0][0], localRow * col, MPI_FLOAT, &matrixOut[0][0], count, displ, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
    printf("[Rank %d] Gathered.\n", rank);
#endif

  /*write data to file*/
  if(0 == rank)
  {
#ifdef DEBUG
    printf("[Rank 0] Recv data. Test= %f\n", matrixOut[0][0]);
#endif

    writeFile(MAT_OUT, matrixOut, row, col);

#ifdef DEBUG
    printf("[Rank 0] Output Write\n");
#endif

#ifdef TIME_CAPTURE
    endTime = MPI_Wtime();
    printf("[Time capture] Done in %lf seconds.\n", (endTime - startTime));
#endif
  }

  MPI_Finalize();
}