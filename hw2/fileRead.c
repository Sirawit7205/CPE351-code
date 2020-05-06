#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double storage[8000005]={};

int main()
{
  int count = 0;
  int i;
  clock_t startTime, endTime;

  startTime = clock();

  FILE *fp = fopen("largeinput.txt", "r");

  char *temp = malloc(20);
  fgets(temp, 20, fp);
  sscanf(temp, "%d", &count);

  //double *storage = malloc(sizeof(double) * count);
  

  for(i = 0; i < count; i++)
  {
    /*fgets(temp, 20, fp);
    sscanf(temp, "%lf", &storage[i]);*/
    fscanf(fp, "%lf", &storage[i]);
  }

  fclose(fp);

  endTime = clock();

  printf("Total time used to read: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

  startTime = clock();

  FILE *fp2 = fopen("output.txt", "w");

  fprintf(fp2, "%d\n", count);

  for(i = 0; i < count; i++)
  {
    /*sprintf(temp, "%.4lf", storage[i]);
    fputs(temp, fp2);*/
    fprintf(fp2, "%.4lf ", storage[i]);
    //fwrite(storage, sizeof(double), sizeof(storage), fp2);
  } 

  fclose(fp2);

  endTime = clock();

  printf("Total time used to write: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

  /*int fp = open("example.txt", O_RDONLY);

  read(fp, &row, sizeof(int));
  read(fp, &col, sizeof(int));

  printf("Row= %d, Col= %d\n", row, col);

  int* storage = malloc(row * col * sizeof(int));

  for(i = 0; i < row; i++)
  {
    printf("%d\n", read(fp, &storage[i * sizeof(int) * col], sizeof(int) * col));
    printf("%d %d %d %d\n",storage[i * col], storage[i * col + 1], storage[i * col + 2], storage[i * col + 3]);
  }

  close(fp);

  FILE *fp2;

  fp2 = fopen("output.txt", "w+");
  fprintf(fp2, "%d %d\n", row, col);
  for(i = 0; i < row * col; i++)
  {
    fprintf(fp2, "%f ", storage[i]);
  }

  fclose(fp2);*/

  return 0;
}