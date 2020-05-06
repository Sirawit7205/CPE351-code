#include <stdio.h>

#define MATSIZE argv[1]
#define OUTNAME argv[2]

int main(int argc, char *argv[])
{
  int i, j, SIZE;
  FILE *fp;

  sscanf(MATSIZE, "%d", &SIZE);
    
  fp = fopen(OUTNAME, "w+");

  fprintf(fp, "%d %d\n", SIZE, SIZE);
  for(i = 0; i < SIZE; i++)
  {
    for(j = 0; j < SIZE; j++)
    {
      /*if(i == 0 || i == SIZE - 1)
        fprintf(fp, "255 ");
      else if(j == 0 || j == SIZE - 1)
        fprintf(fp, "255 ");
      else
        fprintf(fp, "0 ");*/
      fprintf(fp, "%d ", j);
    }
    fprintf(fp, "\n");
  }
  return 0;
}