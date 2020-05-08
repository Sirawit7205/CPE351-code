#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

//#define DEBUG
#define TIME_CAPTURE

#define MAT_IN argv[1]
#define MAT_OUT argv[2]

#define swap(a, b){double _c; _c = a; a = b; b = _c;}

static inline void checkMatrixSize(char *filename, int *count)
{
  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%d", count);
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

static inline void ReadFileWithString(char *filename, double *buffer)
{
  FILE *fp = fopen(filename, "r");

  /*get matrix size*/
  int count;
  fscanf(fp, "%d", &count);

  /*scanning*/
  int i, j;
  char stringBuffer[20]={};
  for(i = 0; i < count; i++)
  {
    fscanf(fp, "%s", stringBuffer);
    buffer[i] = StringToFloat(stringBuffer);
  }
  fclose(fp);
}

static inline void writeFile(char *filename, double *matrixOut, int count)
{
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "%d\n", count);

  int i;
  for(i = 0; i < count; i++)
  {
    fprintf(fp, "%.4lf\n", matrixOut[i]);
  }

  fclose(fp);
}

static inline int partition(double *array, int left, int right)
{
  /*find pivot with median of three method*/
  double pivot = array[left];

  int i = --left, j = ++right;

  while(i < j)
  {
    do {i++;}
    while(array[i] < pivot);
    do {j--;}
    while(array[j] > pivot);

    if(j <= i)
    {
      return j;
    }
    else
    {
      swap(array[i], array[j]);
    }
  }
}

static inline void quickSort(double *array, int left, int right)
{
  if(left < right)
  {
    int pivot = partition(array, left, right);
    quickSort(array, left, pivot);
    quickSort(array, pivot + 1, right);
  }
}

static inline void *mergeArray(double *arrayA, double *arrayB, int countA, int countB)
{
  double *temp = malloc(sizeof(double) * (countA + countB));
  int i = 0, j = 0, k = 0;

  /*compare both*/
  while(i != countA && j != countB)
  {
    temp[k++] = (arrayA[i] < arrayB[j]) ? arrayA[i++] : arrayB[j++];
  }

  /*remaining of A and B*/
  while(i != countA)
  {
    temp[k++] = arrayA[i++];
  }

  while(j != countB)
  {
    temp[k++] = arrayB[j++];
  }

  return temp;
}

int main(int argc, char *argv[])
{
  /*some vars used by all ranks*/
  int i, j;
  int rank, size;
  int count, countLocal;
  int *scatterCount, *displ;
  double *arrayIn, *arrayOut, *arrayLocal;

#ifdef TIME_CAPTURE
  double startTime, endTime;
#endif

  /*mpi init then get rank and size*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int normalJobSize, remainJobSize;
  if(0 == rank)
  {
#ifdef TIME_CAPTURE
    startTime = MPI_Wtime();
#endif

    /*input data size*/
    checkMatrixSize(MAT_IN, &count);

#ifdef DEBUG
    printf("[Rank 0] Input size = %d\n", count);
#endif

    /*allocate memory*/
    arrayIn = (double *)malloc(sizeof(double) * count);

    /*read file*/
    ReadFileWithString(MAT_IN, arrayIn);

#ifdef DEBUG
    printf("[Rank 0] Input read\n");
#endif

    /*calculate job size*/
    normalJobSize = (count / size);
    remainJobSize = (count / size) + (count % size);

#ifdef DEBUG
    printf("[Rank 0] job sizes are %d and %d\n", normalJobSize, remainJobSize);
#endif

    /*prepare scatter parameters*/
    int currentDispl = 0;
    scatterCount = (int *)malloc(sizeof(int) * size);
    displ = (int *)malloc(sizeof(int) * size);

    for(i = 0; i < size; i++)
    {
      scatterCount[i] = (size - 1 == i) ? remainJobSize : normalJobSize;
      displ[i] = currentDispl;
      currentDispl += normalJobSize;
    }
  }

  /*broadcast size*/
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] broadcasted size %d\n", rank, count);
#endif

  /*malloc local storage*/
  countLocal = (size - 1 == rank) ? (count / size) + (count % size) : (count / size);
  arrayLocal = (double *)malloc(sizeof(double) * countLocal);

#ifdef DEBUG
  printf("[Rank %d] local allocated size %d\n", rank, countLocal);
#endif

  /*wait*/
  MPI_Barrier(MPI_COMM_WORLD);

  /*scatter data*/
  MPI_Scatterv(&arrayIn[0], scatterCount, displ, MPI_DOUBLE, &arrayLocal[0], countLocal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Scattered. Test= %lf\n", rank, arrayLocal[0]);
#endif

  /*partition at each process first*/
  int pivotLevel1 = partition(arrayLocal, 0, countLocal -1 );
  /*partition at each half again*/
  int pivotLevel2Left = partition(arrayLocal, 0, pivotLevel1);
  int pivotLevel2Right = partition(arrayLocal, pivotLevel1 + 1, countLocal - 1);

#ifdef DEBUG
  printf("[Rank %d] First 2 partition passed.\n", rank);
#endif

  /*divide partition works to threads*/
  // #pragma omp parallel num_threads(4)
  // {
  //   #pragma omp parallel sections
  //   {
  //     #pragma omp section
         quickSort(arrayLocal, 0, pivotLevel2Left);
  //     #pragma omp section
         quickSort(arrayLocal, pivotLevel2Left + 1, pivotLevel1);
  //     #pragma omp section
         quickSort(arrayLocal, pivotLevel1 + 1, pivotLevel2Right);
  //     #pragma omp section
         quickSort(arrayLocal, pivotLevel2Right + 1, countLocal - 1);
  //   }
  // }

#ifdef DEBUG
  printf("[Rank %d] Calculated.\n", rank);
#endif  

  /*send data back*/
  if(0 == rank)
  {
    /*copy rank 0 data back first*/
    int countOut = scatterCount[0];
    arrayOut = arrayLocal;
    
    /*merging back one by one*/
    for(i = 1; i < size; i++)
    {
      double *arrayTemp = malloc(sizeof(double) * scatterCount[i]);
      MPI_Recv(&arrayTemp[0], scatterCount[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      arrayOut = mergeArray(arrayOut, arrayTemp, countOut, scatterCount[i]);
      countOut += scatterCount[i];
    }

#ifdef DEBUG
  printf("[Rank 0] Merged.\n");
#endif  

  /*write output to file*/
  writeFile(MAT_OUT, arrayOut, count);

#ifdef DEBUG
    printf("[Rank 0] Output Write\n");
#endif

#ifdef TIME_CAPTURE
    endTime = MPI_Wtime();
    printf("[Time capture] Done in %lf seconds.\n", (endTime - startTime));
#endif

  }
  else
  {
    MPI_Send(&arrayLocal[0], countLocal, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

#ifdef DEBUG
  printf("[Rank %d] Sent data back.\n", rank);
#endif  
  }

  /*finalize*/
  MPI_Finalize();
}
