#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define row 5000
#define col 5000

int main() {
	int i, j;
	FILE *fpA, *fpB;

	srand(time(0));

	fpA = fopen("outputA.txt", "w+");
	fpB = fopen("outputB.txt", "w+");

	for(i = 0; i < row; i++) {
		for(j = 0; j < col; j++) {
			fprintf(fpA, "%d ", rand() % 100);
			fprintf(fpB, "%d ", rand() % 100);
		}
		fprintf(fpA, "\n");
		fprintf(fpB, "\n");
	}
fclose(fpA);
fclose(fpB);

return 0;
}
