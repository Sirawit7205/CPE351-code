#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define DEBUG 0 // 0 : debug off , 1: debug on
#define Debug if(DEBUG == 1) 
#define Rank0 if(rank == 0)

#define MatrixType float

#define inputFile argv[1]
#define outputFile argv[2]
#define Iteration atoi(argv[3])

static inline MatrixType **malloc2D(int row, int col){
    MatrixType *data = malloc(sizeof(MatrixType)*row*col);
    MatrixType **arr = malloc(sizeof(long)*(row));
    int i;
    for (i=0; i<row; i++) arr[i] = &data[i*col];
    return arr;
}

void test (){
    
}

MatrixType **calloc2D(int row, int col){
    MatrixType *data = calloc(row*col,sizeof(MatrixType));
    MatrixType **arr = malloc(sizeof(long)*row);
    int i;
    for (i=0; i<row; i++) arr[i] = &data[i*col];
    return arr;
}

void readToMatrix(MatrixType **matrix, char *path, int *row, int *col){
    int i,j;
    FILE *f = fopen(path,"r");
    fscanf(f,"%d %d",row,col);

    for (i=0; i< *row; i++)
        for (j=0; j< *col; j++)
            fscanf(f,"%f",&matrix[i][j]);
    fclose(f);
}

void writeToFile(MatrixType **matrix,char *path, int row, int col){
    int i,j;
    FILE *f = fopen(path,"w");
    fprintf(f,"%d %d\n",row,col);
    for (i=0; i<row; i++){
        for(j=0; j<col; j++)
            fprintf(f,"%.0f ",matrix[i][j]);
        fputs("\n",f);
    }
    fclose(f);
}

void readToMatrix1D(MatrixType *matrix, char *path, int row, int col){
    int i,j;
    FILE *f = fopen(path,"r");
    fscanf(f,"%d %d",&row,&col);

    for (i=0; i< row*col; i++){
        fscanf(f,"%f",&matrix[i]);
    }

    fclose(f);
}

void writeToFile1D(MatrixType *matrix,char *path, int row, int col){
    int i,j;
    int ri;
    FILE *f = fopen(path,"w");
    fprintf(f,"%d %d\n",row,col);
    for (i=0; i<row; i++){
        ri = i*col;
        for(j=0; j<col; j++)
            fprintf(f,"%.0f ",matrix[ri+j]);
        fputs("\n",f);
    }
    fclose(f);
}

void checkMatrixSize( char *path,int *row,int *col){
    FILE *fp = fopen(path,"r");
    fscanf(fp,"%d %d",row,col);
    fclose(fp);
} 

void showMatrix(MatrixType **matrix, int row, int col){
    int i,j;
    for (i=0; i<row; i++){
        for (j=0; j<col; j++){
            printf("%4.0f ",matrix[i][j]);
        }
        printf("\n");
    }
}

static inline MatrixType **heatCal(MatrixType **matrix, int row, int col){
    int i, j;
    int k,m;
    MatrixType **nextFrame = malloc2D(row, col);
    for (i=1; i<row-1; i++){
        nextFrame[i][0] = matrix[i][0];
        nextFrame[i][col-1] = matrix[i][col-1];
        for (j=1; j<col-1; j++){
            nextFrame[i][j] = (matrix[i][j] + matrix[i-1][j] + matrix[i+1][j] + matrix[i][j-1] + matrix[i-1][j-1] + matrix[i+1][j-1] + matrix[i][j+1] + matrix[i-1][j+1] + matrix[i+1][j+1])/9.0; 
            // nextFrame[i][j] /= 9.0;
        }
    }
    memcpy(nextFrame[0],matrix[0],sizeof(MatrixType) * col);
    memcpy(nextFrame[row-1],matrix[row-1],sizeof(MatrixType) * col);
   
    free(&matrix[0][0]);
    free(matrix);
    return nextFrame;
     
}

static inline MatrixType *heatCalOptSSS(MatrixType *ptr, int row, int col){
    int i;
    int k;
    MatrixType *nptr = malloc(sizeof(MatrixType)*row*col);

    MatrixType c[3];
    MatrixType b[3];
    MatrixType d[3]; 

    int rmRow = row%2;
    int lcondition = row-1-rmRow;
    for (i=1; i<lcondition; i+=2){
        int ri = i*col;
        int ri_ = ri+1;
        
        int rim = ri - col;
        int rim_ = rim+1;

        int rip = ri + col;
        int rip_ = rip+1;

        int riq = rip +col; 
        int riq_ = riq+1;

        nptr[ri] = ptr[ri];
        nptr[rip-1] = ptr[rip-1];

        nptr[rip] = ptr[rip];
        nptr[riq-1] = ptr[riq-1];
        k = 0;

        b[0] = (ptr[ri] + ptr[rip]);
        b[1] = (ptr[ri_] + ptr[rip_]);
        b[2] = (ptr[ri_+1] + ptr[rip_+1]);

        c[0] = (b[0] +  ptr[rim]);
        c[1] = (b[1] +  ptr[rim_]);
        c[2] = (b[2] +  ptr[rim_+1]);
        
        d[0] = (b[0] +  ptr[riq]);
        d[1] = (b[1] +  ptr[riq_]);
        d[2] = (b[2] +  ptr[riq_+1]);

        nptr[ri_] = (c[0]+c[1]+c[2])/9.0;
        nptr[rip_] =(d[0]+d[1]+d[2])/9.0;

        int cCondition = col-1;
        int j = 2;
        while(j<cCondition){
            b[k] =  (ptr[ri_ + j] + ptr[rip_ + j]);
            c[k] =  (b[k] +  ptr[rim_ + j]);
            d[k] =  (b[k] +  ptr[riq_ + j]);

            k++;
            if (k > 2) k=0;
            nptr[ri + j] = (c[0]+c[1]+c[2])/9.0;
            nptr[rip + j] =(d[0]+d[1]+d[2])/9.0;
            j+=1;
        }
    }
    if(rmRow){
        i = row-2;

        int ri = i*col;
        int ri_ = ri+1;
        int rip = ri + col; 
        int rip_ = rip+1;
        int rim = ri - col;
        int rim_ = rim+1;

        nptr[ri] = ptr[ri];
        nptr[rip-1] = ptr[rip-1];
        k = 0;

        c[0] = (ptr[ri] +  ptr[rim] + ptr[rip]);
        c[1] = (ptr[ri_]   +  ptr[rim_]   + ptr[rip_]);
        c[2] = (ptr[ri_+1] +  ptr[rim_+1] + ptr[rip_+1]);

        nptr[ri + 1] = (c[0]+c[1]+c[2])/9.0;

        int cCondition = col-1;
        int j = 2;
        while(j<cCondition){
            c[k] = (ptr[ri_+j] +  ptr[rim_+j] + ptr[rip_ + j]);
            k++;
            if (k > 2) k=0;
            nptr[ri + j] = (c[0]+c[1]+c[2])/9.0;
            j++;
        }
    }

    int ri = (row-1)*col;

    memcpy(&nptr[0],ptr,sizeof(MatrixType) * col);
    memcpy(&nptr[ri],&ptr[ri],sizeof(MatrixType) * col);
   
    free(ptr);
    return nptr;
}

                                      
int main(int argc, char **argv){
    int i;
    int p, rank;
    double startTime, finishTime;
    double readTime, writeTime;
    MPI_Request req[3];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    Rank0 startTime = MPI_Wtime();

    // start time set
    // Debug printf("Start At %d\n",rank);
    int row, col;
    int writeRow;
    Rank0  checkMatrixSize(inputFile, &row, &col);
    MatrixType *matrix = malloc(sizeof(matrix)*row*col);
    MatrixType *out    = malloc(sizeof(matrix)*row*col);
    Rank0 {
        readToMatrix1D(matrix ,inputFile,row,col);
        // Debug printf("Matrix Size row : %d || col : %d \n",row ,col);
        writeRow = row;
    }
    // Rank0 readTime = MPI_Wtime();

    // calculate share data
    int Nrow = row/p;
    int rem = row%p;
    int rowIndex = 0;
    int displs[p];
    int sendCount[p];
    int sendSize[p];

    int recvIndex = 0;
    int recvDispls[p];
    int recvCount[p];
    int recvSize[p];

    Rank0 {
        if (p > 1)
            for (i=0; i<p; i++){
                sendCount[i] = i == p-1 || i == 0 ? Nrow + 1 : Nrow +2;
                recvCount[i] = Nrow;
                if (rem > 0){
                    sendCount[i]++;
                    recvCount[i]++;
                    rem--;
                }
                sendSize[i] = sendCount[i] * col;
                displs[i] = i == 0 ? rowIndex*col : (rowIndex-2)*col;
                rowIndex += i == 0 ? sendCount[i] : (sendCount[i]-2);

                recvSize[i] = recvCount[i] * col;
                recvDispls[i] = recvIndex;
                recvIndex += recvSize[i];
            }
        else {
            sendSize[0] = row * col;
            displs[0] = 0;
            sendCount[0] = row;

            *recvSize = *sendSize;
            *recvDispls = *displs;
            *recvCount = *sendCount;
        }   
    }

    MPI_Bcast(displs,p,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sendCount,1,MPI_INT,&row,1,MPI_INT, 0, MPI_COMM_WORLD);
    // Debug printf("rank : %d || displs : %d || row : %d || col : %d\n", rank, displs[rank], row, col);
  
    MatrixType *M1 = malloc(sizeof(MatrixType)*row*col);
    MPI_Scatterv(matrix, sendSize, displs, MPI_FLOAT, M1, row*col, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
    int zi;
    // calculate
    for (i=0; i< Iteration; i++){

        M1 = heatCalOptSSS(M1, row, col);

        if (i < Iteration-1 && p > 1)
        {
            // Debug Rank0 printf("Iteration : %d\n",i);
            zi = (row-2)*col;
            if (rank == 0){
                MPI_Isend(&M1[zi], col, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &req[0]);
                
                MPI_Irecv(&M1[zi + col], col, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &req[0]);
                MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            }

            else if (rank == p-1){
                MPI_Isend(&M1[col], col, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &req[0]);

                MPI_Irecv(&M1[0], col, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &req[0]);
                MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            }

            else {
                MPI_Isend(&M1[col], col, MPI_FLOAT, rank-1, 0,MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&M1[zi], col, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &req[1]);

                MPI_Irecv(&M1[0], col, MPI_FLOAT, rank-1,0, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&M1[zi+col], col, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &req[1]);

                MPI_Wait(&req[0],MPI_STATUS_IGNORE);
                MPI_Wait(&req[1],MPI_STATUS_IGNORE);
            }
        }
    }

    // printf("complete cal @ : %d\n",rank);
    int recvRow;
    MPI_Scatter(recvSize,1,MPI_INT,&recvRow,1,MPI_INT, 0,MPI_COMM_WORLD);

    MPI_Gatherv(rank == 0 ? &M1[0] : &M1[col], recvRow, MPI_FLOAT, out, recvSize, recvDispls, MPI_FLOAT, 0, MPI_COMM_WORLD );
    // Rank0 writeTime = MPI_Wtime();
    Rank0 writeToFile1D(out, outputFile, writeRow, col);
    // Debug Rank0 {

    //     printf("Iteration : %d || rank : %d \n", i, rank);
    //     showMatrix(out, row, col);

    // }

    // finish time set
    Rank0 { finishTime = MPI_Wtime();}
    MPI_Finalize();
    
    Rank0 printf("All Usage Time   : %lf\n",finishTime - startTime);
    // Rank0 printf("read Usage Time  : %lf\n",readTime - startTime);
    // Rank0 printf("write Usage Time : %lf\n",finishTime - writeTime);
    // Rank0 printf("solve Usage Time : %lf\n",writeTime - readTime);


    return 0;
}