#include <mpi.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>

#define NArow 5
#define NAcol 2
#define NBrow 2
#define NBcol 8

#define MatrixA argv[1]
#define MatrixB argv[2]
#define MatrixOut argv[3]

double** malloc2d(int row,int col){
    double *data = malloc(col * row * sizeof(double));
    double **arr = malloc(row* sizeof(double));
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
    fprintf(fp,"%d %d\n",row,col);
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            fprintf(fp,"%.10lf ",InMatrix[i][j]);
        }
        fputs("\n",fp);
    }
    fclose(fp);
}

void readToMatrix(double **matrix, char *path){
    int i,j;
    int row,col;
    FILE *fp = fopen(path,"r");
    fscanf(fp,"%d %d",&row,&col);
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            fscanf(fp,"%lf",&matrix[i][j]);
        }
    }
    fclose(fp);
}

void checkMatrixSize( char *path,int *row,int *col){
    FILE *fp = fopen(path,"r");
    fscanf(fp,"%d %d",row,col);
    fclose(fp);
}

double* plusMatrix(double *matrix1, double *matrix2 , double blockSize){
    double *result = (double*)malloc(blockSize * sizeof(double));
    int i;
    for(i = 0; i<blockSize; i++){
        result[i] = matrix1[i] + matrix2[i];
    }
    return result;
}

void showMatrix(double **matrix,int row,int col){
    int i,j;
    for(i=0; i<row; i++){
        for(j=0; j<col; j++){
            printf("%.10f ",matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]){
    // mpi rank and size variable
    int p,id;
    int i,j,k;
    int row,col;
    MPI_Status status;
    MPI_Request req[4];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double startTime, endTime;
    // load
    int Arow = NArow,Acol = NAcol;
    int Brow = NBrow,Bcol = NBcol;
    checkMatrixSize(MatrixA,&Arow,&Acol);
    checkMatrixSize(MatrixB,&Brow,&Bcol);
    double **matrix1 = malloc2d(Arow,Acol);
    double **matrix2 = malloc2d(Brow,Bcol);
    double **result = malloc2d(Arow,Bcol);
    row = Arow;
    col = Bcol;
    // initial Matrix
    if (id == 0){

        // load from File
        //printf("Reading from file\n");
        readToMatrix(matrix1,MatrixA);
        readToMatrix(matrix2,MatrixB);
        //printf("Reading complete\n");

        // generate
        // printf("Generate matrix");
        // initMatrix(matrix1,NArow,NAcol);
        // initMatrix(matrix2,NBrow,NBcol);

        // printf("Matrix A\n");
        // showMatrix(matrix1,NArow,NAcol);
        // showMatrix(matrix2,NBrow,NBcol);
    }
    startTime = MPI_Wtime();

    int NumRow = Arow/p;
    int rem = Arow%p;

    int rowIndex = 0;
    int *displs = calloc(p,sizeof(int));
    int *sendcounts = calloc(p,sizeof(int)); 
    int *sendSize = calloc(p,sizeof(int)); 


    int recvIndex = 0;
    int *recvDispls = calloc(p,sizeof(int));
    int *recvSize = calloc(p,sizeof(int));

    if (id == 0){        
        // printf("\nbefore Send >> id : %d || Arow : %d || Acol : %d || rem : %d || allProcess : %d || blockSize : %d \n",id,Arow,Acol,rem,p,NumRow);
        // printf("Test rem\n");
        for(i=0; i < p; i++){
            sendcounts[i] = NumRow;
            if (rem>0){
                sendcounts[i]++;
                rem--;
            }
            sendSize[i] = sendcounts[i] * Acol;
            recvSize[i] = sendcounts[i] * Bcol;
            displs[i] = rowIndex;
            rowIndex += sendSize[i];
            recvDispls[i] = recvIndex;
            recvIndex += recvSize[i];
            // printf("Index : %d || startRowIndex : %d || sendcounts : %d || sendSize : %d ||recvSize : %d || recvDispls : %d \n",i,displs[i],sendcounts[i],sendSize[i],recvSize[i],recvDispls[i]);
        }
        // printf("End test rem\n");
    }
    MPI_Bcast(&Brow,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&Bcol,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&matrix2[0][0],Brow*Bcol,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&Acol,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(sendcounts,1,MPI_INT,&Arow,1,MPI_INT,0,MPI_COMM_WORLD);

    // printf("INPUT pid : %d || Arow : %d || Acol : %d \n",id,Arow,Acol);
    double **M1 = malloc2d(Arow,Acol);
    MPI_Scatterv(&matrix1[0][0],sendSize,displs,MPI_DOUBLE,&M1[0][0],Arow*Acol,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double **Mresult = malloc2d(Arow,Bcol);
    //MPI_Wait(&req[0],MPI_STATUS_IGNORE);


    for (i=0; i<Arow; i++){
        for (k = 0; k<Brow; k++){
            Mresult[i][j] = 0;
            for(j=0; j<Bcol; j++){
                Mresult[i][j] += M1[i][k] * matrix2[k][j];
            }
        }
    }

    // printf("OUTPUT pid : %d || Arow : %d || Bcol : %d \n",id,Arow,Bcol);
    MPI_Gatherv(&Mresult[0][0],Arow*Bcol,MPI_DOUBLE,&result[0][0],recvSize,recvDispls,MPI_DOUBLE,0,MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    if(id == 0){
        //printf("Time usage : %lf\n", endTime - startTime);
        // printf("Test Output \n");
        // showMatrix(result,NArow,NBcol);
        // printf("End Output \n");
        writeToFile(result,MatrixOut,row,col);
    }
   
    MPI_Finalize();

}

