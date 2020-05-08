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

static inline matrixType **malloc2D(int row, int col){
    matrixType *data = calloc(row * col , sizeof(matrixType));
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

void matrixShow(matrixType **arr, int row, int col){
    int i,j;
    for(i=0; i<row; i++){
        for(j=0; j<col; j++){
            printf("%.1f ",arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");        
}

void matrixShow1D(matrixType *arr, int size){
    int i;
    for(i=0; i<size; i++){
        printf("%.1f ",arr[i]);
    }
    printf("\n");        
}

matrixType **multipleMatrix2D(matrixType **m1, matrixType **m2, int row, int rc, int col){
    int i,j,k;
    matrixType **result = malloc2D(row,col);
    for(i=0; i<row; i++)
        for(k=0; k<rc; k++)
            for(j=0; j<col; j++){
                result[i][j] += m1[i][k] * m2[k][j]; 
            }
    return  result;
}


int main(int argc, char **argv){
    int rank;
    int p;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    double startTime, endRead, endCal, endWrite;

    int rowSize[p];
    int sendSize[p];
    int recvSize[p];
    int disp[p];
    int Rdisp[p];

    int row1, col1;
    int row2, col2;
    matrixType **matrix1;
    matrixType **matrix2;
    matrixType **Result;
    if(rank == 0){

        startTime = MPI_Wtime();
        matrix1 = readFile(inputMat1, &row1, &col1);
        matrix2 = readFile(inputMat2, &row2, &col2);
        Result  = (float **)malloc2D(row1,col2);
        endRead = MPI_Wtime();

        // calcualte send size
        int Integer = row1 / p;
        int Rem = row1 % p;
        int i;
        int dispIter = 0, RdispIter = 0; 
        printf("Inge : %d || rem : %d\n",Integer,Rem);
        for(i=0; i<p; i++){
            rowSize[i] = Integer;
            if(Rem){
                rowSize[i]++;
                Rem--;
            }
            recvSize[i] = rowSize[i] * col2;
            sendSize[i] = rowSize[i] * col1;

            Rdisp[i] = RdispIter;
            disp[i] = dispIter;

            RdispIter += recvSize[i];
            dispIter += sendSize[i];
        }
    }
    // send
    MPI_Bcast(&col1,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&col2,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank != 0) {
        matrix1 = malloc2D(1,1);
        matrix2 = malloc2D(col1,col2);
        Result  = malloc2D(1,1);
    }
    int priRowSize;
    MPI_Bcast(&matrix2[0][0],col1*col2,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(rowSize,1,MPI_INT,&priRowSize,1,MPI_INT,0,MPI_COMM_WORLD);

    matrixType **priMatrix = malloc2D(priRowSize,col1);
    MPI_Scatterv( &matrix1[0][0],sendSize,disp,MPI_FLOAT,&priMatrix[0][0],priRowSize*col1,MPI_FLOAT,0,MPI_COMM_WORLD);

    //calculate zone
    matrixType **priResult = multipleMatrix2D(priMatrix,matrix2,priRowSize,col1,col2); 
    MPI_Gatherv(&priResult[0][0],priRowSize*col2,MPI_FLOAT,&Result[0][0],recvSize,Rdisp,MPI_FLOAT,0,MPI_COMM_WORLD);

    if(rank == 0) {
        endCal = MPI_Wtime();
        writeFile(outputPath,Result,row1,col2);
        endWrite = MPI_Wtime();

        printf("Total usage time    : %lf\n", endWrite - startTime);
        printf("    Read time usage     : %lf\n", endRead - startTime);
        printf("    Cal time usage      : %lf\n",endCal - endRead);
        printf("    Write time usage    : %lf\n",endWrite - endCal);
    }
    MPI_Finalize();
    return 0;
}