#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define inputFile argv[1]
#define outputFile argv[2]
#define MaxThread atoi(argv[3])

#define Mtype double
#define swap(type,a,b){type _z; _z = a, a=b; b= _z;}

#define Rank0 if(rank == 0)

// change sring to double
static inline double toDouble(const char *p) {
    double out = 0.0;
    unsigned char buffer[39];
    int neg = 0;
    if (*p == '-') {
        neg = 1;
        p++;
    }
    
    // interger part
    int i=0;
    while(*p != '.') out = (out*10.0) + (*p++ - '0');
    
    // faction part
    if (*p == '.') {
        double f = 0.0;
        double n = 1.0;
        p++;
        while (*p >= '0' && *p <= '9') {
            f = (f*10.0) + (*p++ - '0');
            n *= 10.0;
        }
        out += f / n;
    }
    // for nerative number
    if (neg)  out = -out;
    
    return out;
}


static inline void toString(double *arr, int size, int allSize, unsigned char out[], int *len, int header){
    unsigned char buffer[40];
    const int nPoint = 5; // 4 fragtion
    double *p;
    int k = 0;
    if (!header){
        int y = 0;
        while(allSize){
            buffer[y++] = (allSize%10) + '0';
            allSize /= 10;
        }
        int z = y-1;
        while(z >= 0)
            out[k++] = buffer[z--];
        out[k++] = '\n';
    }
    for(p=arr; p<arr+size; p++){
        int ipart = *p;
        int fpart = ((*p - ipart + 0.00005) * 10000);

        // integer part
        int i=1;
        if(ipart){
            buffer[i++] = (ipart%10) + '0';
            ipart /= 10;
            while(ipart){
                buffer[i++] = (ipart%10) + '0';
                ipart /= 10;
            }
        }
        else
            buffer[i++] = '0';
        i--;
        while(i)
            out[k++] = buffer[i--];
        out[k++] = '.';

        // fragtion part
        i = 1;
        do{
            buffer[i++] = (fpart%10) + '0';
            fpart /= 10;
        }while(fpart);
        while(i < nPoint) buffer[i++] = '0';

        i--;
        while(i)
            out[k++] = buffer[i--];
        out[k++] = '\n';
    }
    // all length of char
    *len = k;
}

// read file fucntion read with string and change string to double with my toDouble() function
static inline Mtype *masterRead(char *path, int *size){
    FILE *f = fopen(path,"r");
    unsigned char buffer[39];
    fgets(buffer,sizeof(buffer),f);
    *size = atoi(buffer);
    printf("Array Size : %d\n", *size);
    Mtype *arr = malloc(sizeof(Mtype) * (*size));
    int i;
    
    for(i=0; i < *size; i++){
        fgets(buffer,sizeof(buffer),f);
        arr[i] = toDouble(buffer);
        // printf("%s\n",buffer);
    }
    fclose(f);
    return arr;
}

// write file function write with fwrite. Change double to string
static inline void masterWrite(Mtype *arr, char *path, int size){
    FILE *f = fopen(path,"w");
    int len;
    const int chunkSize = 100000; // 100k number => char size : 2k * 39
    const int bufSize = chunkSize * 39;
    unsigned char buffer[bufSize];

    int nLoop = size / chunkSize;
    int rem = size % chunkSize;

    int i = 0;
    if (nLoop){
        while(i < chunkSize * nLoop){
            // change double to string
            toString(&arr[i], chunkSize, size, buffer, &len,i);
            // write to file
            fwrite(buffer,sizeof(unsigned char),len,f);
            i += chunkSize;
        }
    }

    if(rem){
        // printf("have rem : %d\n",rem);
        toString(&arr[chunkSize*nLoop], rem, size, buffer, &len,i);
        fwrite(buffer,sizeof(unsigned char),len,f);
    }
    fclose(f);
}

void arrShow(Mtype *arr, int size){
    printf("Size of arr : %d\n",size);
    int i;
    for(i=0; i<size; i++){
        printf("%4.4lf\n",arr[i]);
    }
}

static inline int partition(Mtype *arr, int lo,int hi){
    // int num = (rand() % (hi - lo + 1)) + lo; 
    Mtype pivot = arr[lo];
    int i = --lo;
    int j = ++hi;

    for(;;){
        do{i++;}while(arr[i] < pivot);
        do{j--;}while(arr[j] > pivot);
    
        if (i >= j) return j;
        swap(Mtype,arr[i],arr[j]);
    }
}

static inline void quickSort(Mtype *arr, int lo, int hi){   
    if (lo < hi){
        int pivot = partition(arr,lo,hi);
        quickSort(arr, lo, pivot);
        quickSort(arr, pivot+1, hi);
    }
}

static inline void *merge(Mtype *arr1, Mtype *arr2, int size1, int size2){
    int allSize = size1 + size2;
    Mtype *out = malloc(sizeof(Mtype)*allSize);
    int i=0;
    int j=0;
    int k=0;    
    while(i<size1 && j<size2){
        if (arr1[i] < arr2[j]) out[k++] = arr1[i++];
        else out[k++] = arr2[j++];
    }
    while(i<size1) out[k++] = arr1[i++];
    while(j<size2) out[k++] = arr2[j++];
    free(arr1);
    free(arr2);
    return out;
}

int main(int argc, char **argv){
    srand(999);
    // omp_set_num_threads(MaxThread);

    int p,rank;
    double sAlltime;
    double eReadtime;
    double eWritetime;
    double eSequence1, eSequence2;
    double eParalleltime;

    // initial mpi
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int chunk[p];
    int displs[p];
    int arrSize;
    Mtype *arr;
    
    // rank 0 only
    // calculate size of chunk for share data
    Rank0{ 
        sAlltime = MPI_Wtime();
        arr = masterRead(inputFile,&arrSize);
        eReadtime = MPI_Wtime();

        int block = arrSize / p;
        int rem = arrSize % p;

        int i;
        int point = 0;
        for(i=0; i<p; i++){
            chunk[i] = block;
            if (rem > 0){
                chunk[i]++;
                rem--;
            }
            displs[i] = point;
            point += chunk[i];
        }
        eSequence1 = MPI_Wtime();
    }

    // size Array in rank (receive from scatter)
    int privateSize;
    // Start overhead
    // send size of array for malloc
    MPI_Scatter(chunk,1,MPI_INT,&privateSize,1,MPI_INT,0,MPI_COMM_WORLD);    
    Mtype *data = malloc(sizeof(Mtype)*privateSize);                        // not overhead
    // share and receive data from rank 0
    MPI_Scatterv(arr,chunk,displs,MPI_DOUBLE,data,privateSize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // End overhead

    // select pivot 
    int p1 = partition(data,0,privateSize-1);
    // selct 2 pivit from 0 to pi and pi+1 to array size in process
    int p11 = partition(data,0,p1);
    int p12 = partition(data,p1+1,privateSize-1);
    #pragma omp parallel num_threads(MaxThread)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                // printf("thread : %d\n",omp_get_thread_num());
                quickSort(data,0,p11);
            }
            #pragma omp section
            {
                // printf("thread : %d\n",omp_get_thread_num());
                quickSort(data,p11+1,p1);
            }
            #pragma omp section
            {
                // printf("thread : %d\n",omp_get_thread_num());
                quickSort(data,p1+1,p12);
            }
            #pragma omp section
            {
                // printf("thread : %d\n",omp_get_thread_num());
                quickSort(data,p12+1,privateSize-1);
            }
        }
    }

    // MPI_Gatherv(data,privateSize,MPI_DOUBLE,arr,chunk,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    Rank0{
        eParalleltime = MPI_Wtime();

        //navie merge data
        Mtype *arrOut = data;
        int i;
        int size=chunk[0];
        // merge data from mpi process
        for(i=1; i<p;i+=1){
            Mtype *recvData = malloc(sizeof(Mtype) * chunk[i]);
            MPI_Recv(recvData,chunk[i],MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            arrOut = merge(arrOut,recvData,size,chunk[i]);
            size+=chunk[i];
        }
        eSequence2 = MPI_Wtime();

        masterWrite(arrOut, outputFile, size);

        eWritetime = MPI_Wtime();
        
        // calculate time usage
        double read = eReadtime-sAlltime;
        double seq1 = eSequence1-eReadtime;
        double para = eParalleltime-eSequence1;
        double merge = eSequence2-eParalleltime;
        double write = eWritetime-eSequence2;
        double All   = eWritetime - sAlltime;
        double overhead = All - read - seq1 - para - merge - write;

        printf("TimeUsage\n");
        printf(" Read      : %lf\n",read);
        printf(" Sequence  : %lf\n",seq1);
        printf(" Parallel  : %lf\n",para);
        printf(" Merge(seq): %lf\n",merge);
        printf(" Write     : %lf\n",write);
        printf(" All       : %lf\n",All);
        printf(" overhead  : %lf\n",overhead);
        
    }
    else {
        MPI_Send(data,privateSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}