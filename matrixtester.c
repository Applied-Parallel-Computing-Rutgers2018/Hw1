#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#define n 7
#define nsqured n*n

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */ 
#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);
extern void do_block(const int lda, const int M, const int N, const int K, double * restrict A, double *restrict B, double  *restrict C);
extern void printMatrix(const int lda,const double* A );
extern void printMatrixLinear(const int lda,const double* A );
extern void copy_b(int lda, const int K, double *b_src, double *b_dest);
extern void transpose(const double *A, double * B, const int lda);
extern void Direct_Packed_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * restrict B_Block, double * restrict B);
extern void Direct_Blocked_Copy(const int lda, const int N, const int K, const int K_Offset, const int J_Offset, double * restrict Out, double * restrict In);

// void fill (double* p, int nElements, double FillVal)
// {
//   for (int i = 0; i < nElements; ++i)
//     p[i] = FillVal; // Uniformly distributed over [-1, 1]
// }

void fill(double* p, int lda, double FillVal)
{
  for (int i = 0; i < lda; ++i)
  {
    for(int j = 0; j < lda; ++j)
    {
      //printf("%d\n",i + (j*lda));
      p[i + (j*lda)] = FillVal; // Uniformly distributed over [-1, 1]
    }
  }
}

void fillinc (double* p, int lda, double FillVal)
{
  for (int i = 0; i < lda; ++i)
  {
    for(int j = 0; j < lda; ++j)
    {
      p[i + (j*lda)] = FillVal; // Uniformly distributed over [-1, 1]
      FillVal++;
    }
  }
}

/* The benchmarking program */
int main (int argc, char **argv)
{
  printf ("Description:\t%s\n\n", dgemm_desc);

  /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

    double* C = (double*)calloc(3*n*n, sizeof(double));
    //double* Temp = (double*)calloc(3*n*n, sizeof(double));
    //double C [nsqured];

    //double A_Block[(n-4) * (n-4) ];
    
    double A [nsqured];
    double Flag[nsqured];
    double B [nsqured];

    fill(A, n, 2.0);
    fill(Flag, n, 6.66); // canary flags
    fill(B, n, 2.0);

    fill(C,n,0.0);

    printf("INPUT MATRIXES \n");

    printf("Matrix A\n");
    printf("\n");
    printMatrix(n,A);

    printf("\n");

    printf("Matrix B\n");
    printf("\n");

    printMatrix(n,B);

    printf("\n");
    printf("\n");
    printf("\n");
    printf("\n");

    // printf("\n");
    // printf("Before square_dgemm\n");
    // printf("\n");

  	square_dgemm (n, A, B, C);
    //Direct_Blocked_Copy(n, 2, 2, 3, 3,C, B);
    //Transposed_Blocked_Copy(n, 2, 2, 3, 3, C, B);

    //extern void Direct_Blocked_Copy(const int lda, const int N, const int K, const int K_Offset, const int J_Offset, double * restrict Out, double * restrict In);

    //do_block(n, n, n, n, A, B,  C);

    //copy_b(n, n, B, C);
    //Direct_Copy_4(n, n, B, C);

    //transpose(B ,C,n);
    //transpose(B ,Temp,n);
    //do_block(n, n, n, n, A, Temp,  C);

    // copy_b(n, n, Temp, C);
    //Direct_Copy_4(n, n, Temp, C);

    //do_block(n, n, n, n, A, B,  C);


    //Direct_Packed_Blocked_Copy(n, n-1, n, n, 1, 1, C,B);
 //Direct_Packed_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * restrict B_Block, const double * restrict B);
    printf("\n");
    printf("After square_dgemm\n");
    printf("\n");

    // printf("Matrix A\n");
    // printf("\n");
    // printMatrix(n,A);

    // printf("\n");

    // printf("Matrix B\n");
    // printf("\n");

    // printMatrix(n,B);

    printf("\n");
    printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
    printf("Matrix C\n");
    printf("\n");

    printMatrix(n,C);

    free(C);

  return 0;
}

