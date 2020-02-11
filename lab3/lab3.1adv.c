#include <mpi.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>

#define Nrow 5237
#define Ncol 4831

#define MatrixA "matAlarge.txt"
#define MatrixB "matBlarge.txt"
#define MatrixOut "franceOut.txt"

double** malloc2d(int row,int col){
    double *data = malloc(col * row * sizeof(double));
    double **arr = malloc(row * row * sizeof(double));
    int i;
    for (i=0;i<row;i++)
        arr[i] = &(data[col*i]);
    return arr;
}   

void initMatrix(double **matrix, int row , int col){  
    int i,j;
    for(i = 0; i < row;i++){
        for(j=0;j<col;j++){
            matrix[i][j] = i+j;
        }
    }
}

void writeToFile(double **InMatrix,char *path,int row,int col){
    int i,j;
    FILE *fp = fopen(path,"w");
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            fprintf(fp,"%.1f ",InMatrix[i][j]);
        }
        fputs("\n",fp);
    }
    fclose(fp);
}

void readToMatrix(double **matrix, char *path, int row,int col){
    int i,j;
    FILE *fp = fopen(path,"r");

    fscanf(fp, "%d %d", &i, &j);

    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            fscanf(fp,"%lf",&matrix[i][j]);
        }
    }
}

double* plusMatrix(double *matrix1, double *matrix2 , double blockSize){
    double *result = (double*)malloc(blockSize * sizeof(double));
    int i;
    for(i = 0; i<blockSize; i++){
        result[i] = matrix1[i] + matrix2[i];
    }
    return result;
}

double main(int argc, char *argv[]){
    // mpi rank and size variable
    int p,id;
    int i;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double startTime, endTime;

    int blockSize = (Nrow * Ncol)/(p);

    //printf("Start Mpi @ Node %d :: blockSize >> %d :: mpi Node size >> %d \n",id,blockSize,p);
 
    if (id == 0){ // master
        double **matrix1 = malloc2d(Nrow,Ncol);
        double **matrix2 = malloc2d(Nrow,Ncol);
        double **result =  malloc2d(Nrow,Ncol);

        readToMatrix(matrix1,MatrixA,Nrow,Ncol);
        readToMatrix(matrix2,MatrixB,Nrow,Ncol);

        startTime = MPI_Wtime();

        // send data to anathor node 
        for (i = 1; i < p; i++){
            MPI_Send(&matrix1[0][0] + blockSize*i, blockSize , MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&matrix2[0][0] + blockSize*i, blockSize , MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            //printf("Send to Node >> %d\n",i);
        }
        // calculate
        memcpy(&result[0][0],plusMatrix(&matrix1[0][0],&matrix2[0][0],Nrow * Ncol),sizeof(double) * Nrow * Ncol);

        for (i=1;i<p;i++){
            MPI_Recv(&result[0][0]+blockSize*i,blockSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD, &status);     
            memcpy(&result[0][0]+blockSize*i,plusMatrix(&matrix1[0][0] + blockSize*i,&matrix2[0][0]+blockSize*i,blockSize),sizeof(double) * blockSize);
        }


       /*printf("final test :: \n");
       for(i = 0;i<Nrow;i++){
           int j;
           for(j=0;j<Ncol;j++){
             printf("%f ",matrix1[i][j]);
           }
           printf("\n");
       }
       printf("\n\n");
        for(i = 0;i<Nrow;i++){
           int j;
           for(j=0;j<Ncol;j++){
             printf("%f ",matrix2[i][j]);
           }
           printf("\n");
       }
       printf("\n\n");
        for(i = 0;i<Nrow;i++){
           int j;
           for(j=0;j<Ncol;j++){
             printf("%f ",result[i][j]);
           }
           printf("\n");
       }*/

        endTime = MPI_Wtime();

        writeToFile(result,MatrixOut,Nrow,Ncol);

        printf("Done. Time usage: %lf s\n", endTime - startTime);
    }
    else {
        double *matrix1 = (double*)malloc(blockSize * sizeof(double));
        double *matrix2 = (double*)malloc(blockSize * sizeof(double));
        double *result = (double*)malloc(blockSize * sizeof(double));

        MPI_Recv(matrix1,blockSize,MPI_DOUBLE, 0,0,MPI_COMM_WORLD, &status );
        MPI_Recv(matrix2,blockSize,MPI_DOUBLE , 0,1,MPI_COMM_WORLD, &status);
        //printf("Node %d :: Receive complete :: test data @ index 0 >> %f :: Matrix size >> %ld\n",id,matrix1[0], sizeof(matrix1));
        result = plusMatrix(matrix1,matrix2,blockSize);
        MPI_Send(result,blockSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
        free(matrix1);
        free(matrix2);
        //printf("Node %d >> END test value >> %f\n",id,result[0]);
    }
    MPI_Finalize();
}

