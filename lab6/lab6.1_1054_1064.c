//60070501054
//60070501064

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUTA argv[1]
#define INPUTB argv[2]
#define OUTPUT argv[3]

int main(int argc, char *argv[])
{
  int nt, rank;

  #pragma omp parallel private(nt, rank)
  {
    rank = omp_get_thread_num();
    printf("Hello world from thread %d\n", rank);

    if(rank == 0)
    {
      nt = omp_get_num_threads();
      printf("Number of threads = %d\n", nt);
    }
  }
}