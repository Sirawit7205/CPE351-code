#include <stdio.h>

#define ROW 5000
#define COL 5000

int main()
{
  int a, b, out, i, j;
  FILE *fpA, *fpB, *fpOut;

  fpA = fopen("outputA.txt", "r");
  fpB = fopen("outputB.txt", "r");
  fpOut = fopen("outputTest.txt", "w+");

  for(i = 0; i < ROW; i++)
  {
    for(j = 0; j < COL; j++)
    {
      fscanf(fpA, "%d", &a);
      fscanf(fpB, "%d", &b);
      out = a + b;
      fprintf(fpOut, "%d ", out);
    }
    fprintf(fpOut, "\n");
  }

return 0;
}