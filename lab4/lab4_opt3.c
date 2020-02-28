#include <mpi.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
 
#define MatrixA argv[1]
#define MatrixB argv[2]
#define MatrixOut argv[3]

#define BLK_SIZE 128
#define min(a,b) (((a)<(b))?(a):(b))

static inline double** malloc2d(int row,int col){
    double *data = malloc(col * row * sizeof(double));
    double **arr = malloc(row* sizeof(double));
    int i;
    for (i=0;i<row;i++)
        arr[i] = &(data[col*i]);
    return arr;
}   
static inline double** calloc2d(int row,int col){
    double *data = calloc(col * row , sizeof(double));
    double **arr = calloc(row, sizeof(double));
    int i;
    for (i=0;i<row;i++)
        arr[i] = &(data[col*i]);
    return arr;
}  
static inline double** MatrixMultiply(double** a, double** b, int n,  int l, int m) 
{ 
    double **c = calloc2d(n,m); 
    for (int i = 0; i < n; i++) { 
        for (int k = 0; k < l; k++) { 
            for (int j = 0; j < m; j++) { 
                c[i][j] += a[i][k] * b[k][j]; 
            } 
        } 
    } 
    return c; 
} 

void Matrix_Super(double *C, const double *A, const double *B, int m, int n, int p) // Arow, Bcol, "Acol | Brow"
{	
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
	
	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += Aik * B[p*k+j];		
					}					
}

static inline void writeToFile(double **InMatrix,char *path,int row,int col){
    int i,j;
    FILE *fp = fopen(path,"w");
    fprintf(fp,"%d %d\n",row,col);
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            fprintf(fp,"%.0lf ",InMatrix[i][j]);
        }
        fputs("\n",fp);
    }
    fclose(fp);
}
static inline void readToMatrix(double **matrix, char *path){
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
static inline void checkMatrixSize( char *path,int *row,int *col){
    FILE *fp = fopen(path,"r");
    fscanf(fp,"%d %d",row,col);
    fclose(fp);
} 

// static inline void shareBalance(int maxProcess, int rowA, int colA, int colB, int *sendCounts, int *sendSize,int *sendDispls, int *recvSize, int *recvDispls){
//     int i;
//     int NumRow = rowA/maxProcess;
//     int rem = rowA%maxProcess;
//     int rowIndex = 0;
//     int recvIndex = 0;

//     for(i=0; i < maxProcess; i++){
//         sendCounts[i] = NumRow;
//         if (rem>0){
//             sendCounts[i]++;
//             rem--;
//         }
//         sendSize[i] = sendCounts[i] * colA;
//         recvSize[i] = sendCounts[i] * colB;
//         recvDispls[i] = rowIndex;
//         rowIndex += sendSize[i];
//         recvDispls[i] = recvIndex;
//         recvIndex += recvSize[i];
//     }
// }

int main(int argc, char *argv[]){
    // mpi rank and size variable
    double startTime, endTime;
    int p,id;
    int i,j,k;
    int row,col;
    MPI_Status status;
    MPI_Request req[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (id == 0) startTime = MPI_Wtime();

    int Arow = 0,Acol = 0;
    int Brow = 0,Bcol = 0;
    
    checkMatrixSize(MatrixA,&Arow,&Acol);
    double **matrix1 = malloc2d(Arow,Acol);
    checkMatrixSize(MatrixB,&Brow,&Bcol);
    double **matrix2 = malloc2d(Brow,Bcol);
    col = Bcol;
    row = Arow;

    double **result = malloc2d(Arow,Bcol);
    // opt read file
    // printf("Start readFile\n");
    if(p == 1){
        readToMatrix(matrix1,MatrixA);
        readToMatrix(matrix2,MatrixB);
    }
    else{
        if (id == 0){
            readToMatrix(matrix1,MatrixA);
        }
        else if (id == 1){
            readToMatrix(matrix2,MatrixB);
        }
        MPI_Bcast(&Brow,1,MPI_INT,1,MPI_COMM_WORLD);
        MPI_Bcast(&Bcol,1,MPI_INT,1,MPI_COMM_WORLD);        
        MPI_Ibcast(&matrix2[0][0],Brow*Bcol,MPI_DOUBLE,1,MPI_COMM_WORLD,&req[0]);
    } 
    // printf("Reading complete\n");

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
        // printf("Arow : %d || Acol : %d || Brow : %d || Bcol : %d\n",Arow,Acol,Brow,Bcol);
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
    }
    
    MPI_Bcast(&Acol,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(sendcounts,1,MPI_INT,&Arow,1,MPI_INT,0,MPI_COMM_WORLD);

    double **M1 = malloc2d(Arow,Acol);
    MPI_Scatterv(&matrix1[0][0],sendSize,displs,MPI_DOUBLE,&M1[0][0],Arow*Acol,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double **Mresult = calloc2d(Arow,Bcol);
    if (p > 1) MPI_Wait(&req[0],MPI_STATUS_IGNORE);

    // double **Mresult = MatrixMultiply(M1,matrix2,Arow,Brow,Bcol);
    Matrix_Super(&Mresult[0][0], &M1[0][0], &matrix2[0][0],Arow,Acol,Bcol);
    MPI_Gatherv(&Mresult[0][0],Arow*Bcol,MPI_DOUBLE,&result[0][0],recvSize,recvDispls,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(id == 0){
        writeToFile(result,MatrixOut,row,col);
        endTime = MPI_Wtime();
        printf("Time usage : %lf\n", endTime - startTime);
    }
    MPI_Finalize();
}

