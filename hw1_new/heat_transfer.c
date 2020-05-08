#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT argv[1]
#define OUTPUT argv[2]
#define FRAMEIN argv[3]

#define matrixType float

matrixType **malloc2D(int row, int col);
void checkMatrixSize(char filename[], int *row, int *col);
void readMatrix(char filename[], matrixType **mat);
void writeMatrix(char filename[], matrixType **mat, int row, int col);

int main(int argc, char *argv[])
{
  //mpi vars
  int procCount, procId;
  double startTime, endTime;

  //other vars
  int row, col;
  int i, j;

  //init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);
  MPI_Comm_rank(MPI_COMM_WORLD, &procId);

  //capture start time
  //startTime = MPI_Wtime();

  //get input size
  checkMatrixSize(INPUT, &row, &col);

  //malloc storage
  matrixType **inputMat = malloc2D(row, col);
  matrixType **outputMat = malloc2D(row, col);

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

    for(i = 0; i < procCount; i++)
    {
      //amount of rows
      if(i < worksizeRem)
        assignRow[i] = worksize + 1;
      else
        assignRow[i] = worksize;

      //adjustment for overlapping rows
      //note that overlapping must be turned off on single-process runs
      if(procCount > 1)
      {
        if(i == 0 || i == procCount - 1)
          assignRow[i] += 1;
        else
          assignRow[i] += 2;
      }
      
      //amount of elements
      assignSize[i] = assignRow[i] * col;

      //displacements
      if(i == 0)
        assignDispl[i] = 0;
      else
        assignDispl[i] = assignDispl[i-1] + assignSize[i-1] - (2 * col);

      //printf("For process %d, rows = %d size = %d displ = %d\n", i, assignRow[i], assignSize[i], assignDispl[i]);
    }
  }

  int localRow;
  //broadcast and scatter worksizes
  MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(&assignRow[0], 1, MPI_INT, &localRow, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //printf("For process %d, rows = %d, cols = %d\n", procId, localRow, col);

  //allocate memory
  matrixType **localMatrix = malloc2D(localRow * 2, col);

  //scatter data to processes
  MPI_Scatterv(&inputMat[0][0], assignSize, assignDispl, MPI_FLOAT, &localMatrix[0][0], localRow * col, MPI_FLOAT, 0, MPI_COMM_WORLD);

  //calculate
  int round;                  //simulation frame
  int resDispl = localRow;    //location displacement of result storing buffer
  int datDispl = 0;           //location displacement of input data buffer
  int FRAME;

  //get frame count
  sscanf(FRAMEIN, "%d", &FRAME);

  //actual calculation
  for(round = 0; round < FRAME; round ++)
  {
    //printf("Process %d: Entering round %d\n", procId, round);
    //case based on overlapping types
    //first part, ignore LAST row
    if(procId == 0)
    {
      for(i = 0; i < localRow - 1; i++)
      {
        for(j = 0; j < col; j++)
        {
          //values at the edge doesn't change
          if(i == 0 || j == 0 || j == col -1)
            localMatrix[i + resDispl][j] = localMatrix[i + datDispl][j];
          //average values of 9 points
          else
            localMatrix[i + resDispl][j] = (localMatrix[i + datDispl - 1][j - 1] + localMatrix[i + datDispl - 1][j] + localMatrix[i + datDispl - 1][j + 1] +
                                            localMatrix[i + datDispl][j - 1] + localMatrix[i + datDispl][j] + localMatrix[i + datDispl][j + 1] +
                                            localMatrix[i + datDispl + 1][j - 1] + localMatrix[i + datDispl + 1][j] + localMatrix[i + datDispl + 1][j + 1])
                                            / 9.0;
        }
      }

      //send the LAST-1 row to the NEXT process
      //get the LAST row from the NEXT process
      if(procCount > 1)
      {
        MPI_Sendrecv(&localMatrix[localRow - 2 + resDispl][0], col, MPI_FLOAT, procId + 1, 0,
                      &localMatrix[localRow - 1 + resDispl][0], col, MPI_FLOAT, procId + 1, MPI_ANY_TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
      }
    }
    //last part, ignore FIRST row
    else if(procId == procCount - 1)
    {
      for(i = 1; i < localRow; i++)
      {
        for(j = 0; j < col; j++)
        {
          //values at the edge doesn't change
          if(i == localRow - 1 || j == 0 || j == col -1)
            localMatrix[i + resDispl][j] = localMatrix[i + datDispl][j];
          //average values of 9 points
          else
            localMatrix[i + resDispl][j] = (localMatrix[i + datDispl - 1][j - 1] + localMatrix[i + datDispl - 1][j] + localMatrix[i + datDispl - 1][j + 1] +
                                            localMatrix[i + datDispl][j - 1] + localMatrix[i + datDispl][j] + localMatrix[i + datDispl][j + 1] +
                                            localMatrix[i + datDispl + 1][j - 1] + localMatrix[i + datDispl + 1][j] + localMatrix[i + datDispl + 1][j + 1])
                                            / 9.0;
        }
      }

      //send the FIRST+1 row to the PREV process
      //get the FIRST row from the PREV process
      if(procCount > 1)
      {
        MPI_Sendrecv(&localMatrix[1 + resDispl][0], col, MPI_FLOAT, procId - 1, 0,
                      &localMatrix[0 + resDispl][0], col, MPI_FLOAT, procId - 1, MPI_ANY_TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
      }
    }
    //the rest, ignore FIRST and LAST row
    else
    {
      for(i = 1; i < localRow - 1; i++)
      {
        for(j = 0; j < col; j++)
        {
          //values at the edge doesn't change
          if(j == 0 || j == col -1)
            localMatrix[i + resDispl][j] = localMatrix[i + datDispl][j];
          //average values of 9 points
          else
            localMatrix[i + resDispl][j] = (localMatrix[i + datDispl - 1][j - 1] + localMatrix[i + datDispl - 1][j] + localMatrix[i + datDispl - 1][j + 1] +
                                            localMatrix[i + datDispl][j - 1] + localMatrix[i + datDispl][j] + localMatrix[i + datDispl][j + 1] +
                                            localMatrix[i + datDispl + 1][j - 1] + localMatrix[i + datDispl + 1][j] + localMatrix[i + datDispl + 1][j + 1])
                                            / 9.0;
        }
      }

      //send the FIRST+1 row to the PREV process
      //get the FIRST row from the PREV process
      if(procCount > 1)
      {
        MPI_Sendrecv(&localMatrix[1 + resDispl][0], col, MPI_FLOAT, procId - 1, 0,
                      &localMatrix[0 + resDispl][0], col, MPI_FLOAT, procId - 1, MPI_ANY_TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //send the LAST-1 row to the NEXT process
        //get the LAST row from the NEXT process
        MPI_Sendrecv(&localMatrix[localRow - 2 + resDispl][0], col, MPI_FLOAT, procId + 1, 0,
                      &localMatrix[localRow - 1 + resDispl][0], col, MPI_FLOAT, procId + 1, MPI_ANY_TAG,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);          
      }    
    }
    
    //swap displ values
    int temp = resDispl;
    resDispl = datDispl;
    datDispl = temp;
  }

  //undo last displ swap
  int temp = resDispl;
  resDispl = datDispl;
  datDispl = temp;

  //gathering results
  MPI_Gatherv(&localMatrix[resDispl][0], localRow * col, MPI_FLOAT, &outputMat[0][0], assignSize, assignDispl, MPI_FLOAT, 0, MPI_COMM_WORLD);

  //write output to file
  if(procId == 0)
  {
    writeMatrix(OUTPUT, outputMat, row, col);

    //capture end time
    //endTime = MPI_Wtime();

    //total time output
    //printf("Done. Total time usage = %lf\n", endTime - startTime);
  }

  MPI_Finalize();
}

matrixType **malloc2D(int row, int col)
{
  matrixType *data = malloc(sizeof(matrixType) * row * col);
  matrixType **ptr2d = malloc(sizeof(long) * row);

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

void readMatrix(char filename[], matrixType **mat)
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
      fscanf(fp, "%f", &mat[i][j]);
    }
  }

  fclose(fp);
}

void writeMatrix(char filename[], matrixType **mat, int row, int col)
{
  FILE *fp;
  fp = fopen(filename, "w+");

  fprintf(fp, "%d %d\n", row, col);

  int i, j;
  for(i = 0; i < row; i++)
  {
    for(j = 0; j < col; j++)
    {
      fprintf(fp, "%.0f ", mat[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}