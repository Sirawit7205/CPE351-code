#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
//60070501054
//60070501064

#include <time.h>

#define PR_COUNT argv[1]
#define INPUTA argv[2]
#define INPUTB argv[3]
#define OUTPUT argv[4]

int main(int argc, char *argv[])
{
  int i, ROW, COL, PROC_COUNT;
  float *ptrInputA, *ptrInputB, *ptrOutput;
  FILE *fpA, *fpB, *fpOut; 
  clock_t startTime, endTime, startCTime, endCTime;

  startTime = clock();
  
  sscanf(PR_COUNT, "%d", &PROC_COUNT);
  omp_set_dynamic(0);
  omp_set_num_threads(PROC_COUNT);

  //open input and output files
  fpA = fopen(INPUTA, "r");
  fpB = fopen(INPUTB, "r");
  fpOut = fopen(OUTPUT, "w+");

  //get matrix sizes
  fscanf(fpA, "%d %d", &ROW, &COL);
  fscanf(fpB, "%d %d", &ROW, &COL);

  //malloc storages
  ptrInputA = (float *)malloc(ROW * COL * sizeof(float));
  ptrInputB = (float *)malloc(ROW * COL * sizeof(float));
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

  //calculate
  startCTime = clock();
  #pragma omp parallel for
  for(i = 0; i < ROW * COL; i++)
  {
    ptrOutput[i] = ptrInputA[i] + ptrInputB[i];
  }
  endCTime = clock();

  //file output
  fprintf(fpOut, "%d %d\n", ROW, COL);
  for(i = 0; i < ROW * COL; i++)
  {
    fprintf(fpOut, "%.1f ", ptrOutput[i]);
    if(i % COL == (COL - 1))
      fprintf(fpOut, "\n");
  }

  //close output file
  fclose(fpOut);

  endTime = clock();

  printf("Total time usage: %lf\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
  printf(" Calc time usage: %lf\n", (double)(endCTime-startCTime)/CLOCKS_PER_SEC);
  
  return 0;
}