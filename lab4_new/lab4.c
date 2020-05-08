#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//#define DEBUG
#define TIME_CAPTURE

#define MAT_A argv[1]
#define MAT_B argv[2]
#define MAT_OUT argv[3]

static inline double **malloc2D(int row, int col)
{
  /*all memory required as a 1D array*/
  double *data1d = malloc(sizeof(double) * row * col);
  /*memory required for row pointers, 
    use sizeof(long) because float cannot hold 64-bit address space*/
  double **rowPtr = malloc(sizeof(long) * row);

  int i;
  for(i = 0; i < row; i++)
  {
    /*each pointer points to address of data1d*/
    rowPtr[i] = &(data1d[i * col]);
  }

  return rowPtr;
}

static inline void checkMatrixSize(char *filename, int *row, int *col)
{
  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%d %d", row, col);
  fclose(fp);
}

static inline double StringToFloat(char *input)
{
  double value = 0.0, decimalMuliplier = 0.1, minusMultiplier = 1.0;
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

void ReadFileWithString(char *filename, double **buffer)
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

void ReadFileWithStringAndTranspose(char *filename, double **buffer)
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
      buffer[j][i] = StringToFloat(stringBuffer);
    }
  }
  fclose(fp);
}

void writeFile(char *filename, double **matrixOut, int row, int col)
{
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "%d %d\n", row, col);

  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fprintf(fp, "%.0lf ", matrixOut[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}

void matrixMultiplication(double **matrixA, double **matrixB, double **matrixOut, int rowA, int colA, int rowB, int colB)
{
  int i, j, k;
  double temp = 0.0;

  /*slide matrixB in the outermost loop for locality*/
  for(int i = 0; i < colB; i++)
  {
    for(int j = 0; j < rowA; j++)
    {
      /*clear temp addition var*/
      temp = 0;
      for(int k = 0; k < colA; k++)
      {
        temp += (matrixA[j][k] * matrixB[i][k]);
      }
      /*save to output matrix*/
      matrixOut[j][i] = temp;
    }
  }
}

int main(int argc, char *argv[])
{
  /*some variables used in all ranks*/
  int i, j;
  int rank, size;
  double **matrixA, **matrixB, **matrixOut;
  double **matrixAlocal, **matrixOutlocal;
  int rowA, colA, rowB, colB;

#ifdef TIME_CAPTURE
  double startTime, endTime;
#endif

  /*mpi init then get rank and size*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*let rank 0 read file and assign work,
    all other ranks calculate*/
  int *count, *displ;
  int *countOut, *displOut;
  if(0 == rank)
  {

#ifdef TIME_CAPTURE
    startTime = MPI_Wtime();
#endif

    /*input data size*/
    /*we have to separate row and col for matrix A and B*/
    checkMatrixSize(MAT_A, &rowA, &colA);
    checkMatrixSize(MAT_B, &rowB, &colB);

#ifdef DEBUG
    printf("[Rank 0] Input size A: %dx%d\n", rowA, colA);
    printf("[Rank 0] Input size B: %dx%d\n", rowB, colB);
#endif

    /*malloc space*/
    /*matrix B will be transposed*/
    /*output matrix of multiplication to rowA * colB*/
    matrixA = malloc2D(rowA, colA);
    matrixB = malloc2D(colB, rowB);
    matrixOut = malloc2D(rowA, colB);

    /*read file*/
    ReadFileWithString(MAT_A, matrixA);
    ReadFileWithStringAndTranspose(MAT_B, matrixB);

#ifdef DEBUG
    printf("[Rank 0] Input read\n");
#endif

    /*calculate job size*/
    int normalJobSize = (rowA / size) * colA;
    int remainJobSize = ((rowA / size) + (rowA % size)) * colA;
    int normalJobSizeOut = (rowA / size) * colB;
    int remainJobSizeOut = ((rowA / size) + (rowA % size)) * colB;

#ifdef DEBUG
    printf("[Rank 0] job sizes are %d and %d\n", normalJobSize, remainJobSize);
#endif

    /*prepare scatter parameters*/
    int currentDispl = 0;
    int currentDisplOut = 0;
    count = (int *)malloc(sizeof(int) * size);
    displ = (int *)malloc(sizeof(int) * size);
    countOut = (int *)malloc(sizeof(int) * size);
    displOut = (int *)malloc(sizeof(int) * size);

    for(i = 0; i < size; i++)
    {
      count[i] = (size - 1 == i) ? remainJobSize : normalJobSize;
      displ[i] = currentDispl;
      countOut[i] = (size - 1 == i) ? remainJobSizeOut : normalJobSizeOut;
      displOut[i] = currentDisplOut;
      currentDispl += normalJobSize;
      currentDisplOut += normalJobSizeOut;
    }
  }

  /*broadcast matrix A and B size*/
  MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Broadcasted size A: %dx%d\n", rank, rowA, colA);
  printf("[Rank %d] Broadcasted size B: %dx%d\n", rank, rowB, colB);
#endif

  /*allocate local matrix memory*/
  int rowAlocal = (size - 1 == rank) ? ((rowA / size) + (rowA % size)) : (rowA / size);
  matrixAlocal = malloc2D(rowAlocal, colA);
  matrixOutlocal = malloc2D(rowAlocal, colB);

  if(0 != rank)
  {
    matrixB = malloc2D(colB, rowB);
  }

#ifdef DEBUG
  printf("[Rank %d] Allocated. A: %dx%d B: %dx%d OUT: %dx%d\n", rank, rowAlocal, colA, rowB, colB, rowAlocal, colB);
#endif

  /*wait*/
  MPI_Barrier(MPI_COMM_WORLD);

  /*broadcast matrix B*/
  MPI_Bcast(&matrixB[0][0], rowB * colB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /*scatter matrix A*/
  MPI_Scatterv(&matrixA[0][0], count, displ, MPI_DOUBLE, &matrixAlocal[0][0], rowAlocal * colA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Scattered and Broadcasted. Test= %lf %lf\n", rank, matrixAlocal[0][0], matrixB[0][0]);
#endif

  /*actual calculation*/
  matrixMultiplication(matrixAlocal, matrixB, matrixOutlocal, rowAlocal, colA, rowB, colB);

#ifdef DEBUG
  printf("[Rank %d] Calculated.\n", rank);
#endif  

  /*gather results back*/
  MPI_Gatherv(&matrixOutlocal[0][0], rowAlocal * colB, MPI_DOUBLE, &matrixOut[0][0], countOut, displOut, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Gathered.\n", rank);
#endif

  /*write data to file*/
  if(0 == rank)
  {
#ifdef DEBUG
    printf("[Rank 0] Recv data. Test= %lf\n", matrixOut[0][0]);
#endif

    writeFile(MAT_OUT, matrixOut, rowA, colB);

#ifdef DEBUG
    printf("[Rank 0] Output Write\n");
#endif

#ifdef TIME_CAPTURE
    endTime = MPI_Wtime();
    printf("[Time capture] Done in %lf seconds.\n", (endTime - startTime));
#endif
  }

  /*finalize*/
  MPI_Finalize();
}