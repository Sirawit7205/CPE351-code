#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEBUG
#define TIME_CAPTURE

#define MAT_IN argv[1]
#define MAT_OUT argv[2]
#define ITER 10

#define swap(a,b){int c = a; a = b; b = c;}

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

static inline void heatTransfer(double **matrix, int row, int col, int iter, int rank, int size)
{
  int i, j, k;
  int actualRow = row / 2;
  int offsetR = 0, offsetW = actualRow;
printf("%d %d\n", actualRow, col);
  if(0 == rank)
  {
    /*rank 0 avoid top and side, share bottom*/
    for(k = 0; k < iter; k++)
    {
      for(i = 0; i < actualRow; i++)
      {
        for(j = 0; j < col; j++)
        {
          if(i == 0 || j == 0 || j == col - 1)
            matrix[i + offsetW][j] = matrix[i + offsetR][j];
          else
            matrix[i + offsetW][j] = (matrix[i+offsetR][j] + matrix[i+offsetR-1][j] + matrix[i+offsetR+1][j] 
                                      + matrix[i+offsetR][j-1] + matrix[i+offsetR-1][j-1] + matrix[i+offsetR+1][j-1] 
                                      + matrix[i+offsetR][j+1] + matrix[i+offsetR-1][j+1] + matrix[i+offsetR+1][j+1]) / 9.0; 
        }
      }

#ifdef DEBUG
      printf("[Rank %d] Pass round %d\n", rank, k);
#endif

      /*exchange bottom row*/
      MPI_Sendrecv(&matrix[actualRow + offsetW - 2][0], col, MPI_DOUBLE, rank + 1, 0,
                    &matrix[actualRow + offsetW - 1][0], col, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /*swap write location*/
      swap(offsetR, offsetW);
    }
  }
  else if(size - 1 == rank)
  {
    /*rank size -1 avoid bottom and side, share top*/
    for(k = 0; k < iter; k++)
    {
      for(i = 0; i < actualRow; i++)
      {
        for(j = 0; j < col; j++)
        {
          if(i == actualRow - 1 || j == 0 || j == col - 1)
            matrix[i + offsetW][j] = matrix[i + offsetR][j];
          else
            matrix[i + offsetW][j] = (matrix[i+offsetR][j] + matrix[i+offsetR-1][j] + matrix[i+offsetR+1][j] 
                                      + matrix[i+offsetR][j-1] + matrix[i+offsetR-1][j-1] + matrix[i+offsetR+1][j-1] 
                                      + matrix[i+offsetR][j+1] + matrix[i+offsetR-1][j+1] + matrix[i+offsetR+1][j+1]) / 9.0; 
        }
      }

#ifdef DEBUG
      printf("[Rank %d] Pass round %d\n", rank, k);
#endif

      /*exchange top row*/
      MPI_Sendrecv(&matrix[offsetW + 1][0], col, MPI_DOUBLE, rank - 1, 0,
                    &matrix[offsetW][0], col, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /*swap write location*/
      swap(offsetR, offsetW);
    }
  }
  else
  {
    /*other ranks avoid side, share top and bottom*/
    for(k = 0; k < iter; k++)
    {
      for(i = 1; i < actualRow - 1; i++)
      {
        for(j = 1; j < col - 1; j++)
        {
          if(j == 0 || j == col - 1)
            matrix[i + offsetW][j] = matrix[i + offsetR][j];
          else
            matrix[i + offsetW][j] = (matrix[i+offsetR][j] + matrix[i+offsetR-1][j] + matrix[i+offsetR+1][j] 
                                      + matrix[i+offsetR][j-1] + matrix[i+offsetR-1][j-1] + matrix[i+offsetR+1][j-1] 
                                      + matrix[i+offsetR][j+1] + matrix[i+offsetR-1][j+1] + matrix[i+offsetR+1][j+1]) / 9.0; 
        }
      }

#ifdef DEBUG
      printf("[Rank %d] Pass round %d\n", rank, k);
#endif

      /*exchange top and bottom row*/
      MPI_Sendrecv(&matrix[offsetW + 1][0], col, MPI_DOUBLE, rank - 1, 0,
                    &matrix[offsetW][0], col, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Sendrecv(&matrix[actualRow + offsetW - 2][0], col, MPI_DOUBLE, rank + 1, 0,
                    &matrix[actualRow + offsetW - 1][0], col, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      /*swap write location*/
      swap(offsetR, offsetW);
    }
  }
}

int main(int argc, char *argv[])
{
  /*some variables used in all ranks*/
  int i, j;
  int rank, size;
  double **matrixA, **matrixOut;
  double **matrixALocal;
  int rowA, colA;

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
    checkMatrixSize(MAT_IN, &rowA, &colA);

#ifdef DEBUG
    printf("[Rank 0] Input size A: %dx%d\n", rowA, colA);
#endif

    /*malloc space*/
    matrixA = malloc2D(rowA, colA);
    matrixOut = malloc2D(rowA, colA);

    /*read file*/
    ReadFileWithString(MAT_IN, matrixA);

#ifdef DEBUG
    printf("[Rank 0] Input read\n");
#endif

    /*calculate job size*/
    int normalJobSize = (rowA / size) * colA;
    int remainJobSize = ((rowA / size) + (rowA % size)) * colA;

#ifdef DEBUG
    printf("[Rank 0] job sizes are %d and %d\n", normalJobSize, remainJobSize);
#endif

    /*prepare scatter parameters*/
    int currentDispl = 0;
    count = (int *)malloc(sizeof(int) * size);
    displ = (int *)malloc(sizeof(int) * size);

    for(i = 0; i < size; i++)
    {
      /*count*/
      if(0 == i)
        count[i] = normalJobSize + colA;
      else if(size - 1 == i)
        count[i] = remainJobSize + colA;
      else
        count[i] = normalJobSize + (colA * 2);
      
      /*displ*/
      displ[i] = currentDispl;
      if(0 == i)
        currentDispl += (normalJobSize + colA);
      else
        currentDispl += (normalJobSize + (colA * 2));
    }
  }

  /*broadcast size first so all ranks can allocate memory*/
  MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] broadcasted size %dx%d\n", rank, rowA, colA);
#endif

  /*malloc local storage, double the size to swing frames around*/
  int localRow;
  if(0 == rank)
    localRow = ((rowA / size) + 1) * 2;
  else if(size - 1 == rank)
    localRow = ((rowA / size) + (rowA % size) + 1) * 2;
  else
    localRow = ((rowA / size) + 2) * 2;
  matrixALocal = malloc2D(localRow, colA);

#ifdef DEBUG
  printf("[Rank %d] local allocated size %dx%d\n", rank, localRow, colA);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  /*scatter actual data*/
  MPI_Scatterv(&matrixA[0][0], count, displ, MPI_DOUBLE, &matrixALocal[0][0], localRow * colA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Scattered. Test= %lf\n", rank, matrixALocal[0][0]);
#endif

  heatTransfer(matrixALocal, localRow, colA, ITER, rank, size);

#ifdef DEBUG
  printf("[Rank %d] Calculated.\n", rank);
#endif

  /*finalize*/
  MPI_Finalize();
}