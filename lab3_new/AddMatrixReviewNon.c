#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define debug if(1)
#define pass  printf("test pass\n");

#define matrixType float

#define inputMat1   argv[1]
#define inputMat2   argv[2]
#define outputPath  argv[3]

matrixType **malloc2D(int row, int col){
    matrixType *data = malloc(row * col * sizeof(matrixType));
    matrixType **arr = malloc(row * sizeof(long));
    int i;
    for(i=0; i<row; i++)
        arr[i] = &data[i*col];
    return arr;
}

float toFloat(const char *p){
    float r = 0.0;
    unsigned char buffer[39];
    int neg = 0;
    if(*p == '-'){neg=1; p++;}
    while(*p != '.'){
        r = (r*10.0) + (*p-'0');
        p++;
    }
    if (*p == '.'){
        float f = 0.0;
        int n=1;
        p++;
        while(*p >= '0' && *p <= '9'){
            f = (f*10.0) + (*p -'0');
            n *= 10;
            p++;
        }
        r += f/n;
    }
    return neg ? -r:r;
}

matrixType **fastReadFile(char *input, int *row, int *col){
    FILE *fp = fopen(input, "r");
    unsigned char buf[39];
    fscanf(fp,"%s",buf);
    *row = atoi(buf);
    fscanf(fp,"%s",buf);
    *col = atoi(buf);        

    matrixType **arr = malloc2D(*row, *col);
    printf("row: %d , col: %d\n",*row,*col);
    fgets(buf, sizeof(buf), fp);
    int i, j;
    for(i=0; i< *row; i++)
        for(j=0; j< *col; j++){
            fscanf(fp,"%s",buf);
            arr[i][j] = toFloat(buf);
        }
    fclose(fp);
    return arr;
}

matrixType **readFile(char *input, int *row, int *col){
    FILE *fp = fopen(input, "r");
    fscanf(fp,"%d %d",row, col);
    matrixType **arr = malloc2D(*row, *col);

    int i,j;
    for(i=0; i< *row; i++)
        for(j=0; j< *col; j++)
            fscanf(fp,"%f",&arr[i][j] );
    fclose(fp);
    return arr;
}

void writeFile(char *output,matrixType **arr,  int row, int col){
    FILE *fp = fopen(output,"w");
    fprintf(fp,"%d %d\n",row, col);

    int i,j;
    for(i=0; i<row; i++){
        for(j=0; j<col; j++){
            fprintf(fp,"%.1f ",arr[i][j]);
        }
        fputs("\n", fp);
    }
    fclose(fp);
}

matrixType *addMatrix1D(matrixType *arr1 , matrixType *arr2, int row, int col){
    int i;
    int size = row*col;
    matrixType *arrOut = malloc(size * sizeof(matrixType));
    for(i=0; i<size; i++){
        arrOut[i] = arr1[i] + arr2[i];
    }
    return arrOut;
}


int main(int argc, char **argv){
    int rank;
    int p;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if(rank == 0){
        int row1,col1;
        int row2,col2;
        double startTime, endRead, endCal, endWrite;
        startTime = MPI_Wtime();
        matrixType **matrix1 = fastReadFile(inputMat1,&row1, &col1);
        matrixType **matrix2 = fastReadFile(inputMat2,&row2, &col2);
        endRead = MPI_Wtime();

        // send
        int blockSize = row1*col1/p;
        int blockSizeRem = blockSize +((row1*col1) % p);
        int i;
        MPI_Request req[3];
        for(i=1; i<p; i++){
            int Size = i == p-1 ? blockSizeRem : blockSize;
            MPI_Send(&Size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Isend(&matrix1[0][0] + i*blockSize, Size, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Isend(&matrix2[0][0] + i*blockSize, Size, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &req[2]);
        }
        // cal
        matrixType **Result = malloc2D(row1,col1);

        matrixType *ptr1 = &matrix1[0][0];
        matrixType *ptr2 = &matrix2[0][0];
        matrixType *ptrRe = &Result[0][0];

        i = -1;
        while(++i < blockSize){
            ptrRe[i] = ptr1[i] + ptr2[i]; 
        }

        // merge data
        MPI_Request recv[p];
        for(i=1; i<p; i++){
            int Size = i == p-1 ? blockSizeRem : blockSize;
            MPI_Irecv(&Result[0][0] + i*blockSize, Size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &recv[i]); 
        }

        i = 1;
        while(i<p) MPI_Wait(&recv[i++],MPI_STATUS_IGNORE);
        endCal = MPI_Wtime();

        writeFile(outputPath, Result, row1, col1);
        endWrite = MPI_Wtime();
        
        //show usage time
        printf("Total usage time    : %lf\n", endWrite - startTime);
        printf("    Read time usage     : %lf\n", endRead - startTime);
        printf("    Cal time usage      : %lf\n",endCal - endRead);
        printf("    Write time usage    : %lf\n",endWrite - endCal);
    }
    else{
        int Size;
        MPI_Recv(&Size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrixType *m1 = malloc(Size * sizeof(matrixType));
        matrixType *m2 = malloc(Size * sizeof(matrixType));
        MPI_Request req[3];
        MPI_Irecv(m1, Size ,MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(m2, Size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Wait(&req[1],MPI_STATUS_IGNORE);
        MPI_Wait(&req[2],MPI_STATUS_IGNORE);
        MPI_Send(addMatrix1D(m1,m2,Size,1), Size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    

    MPI_Finalize();
    return 0;
}