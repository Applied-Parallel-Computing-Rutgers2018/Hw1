#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#define n 6
#define nsqured n*n

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */ 
#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);
extern void printMatrix(const int lda,const double* A );


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
    //double C [nsqured];
    
    double A [nsqured];
    double B [nsqured];

    fillinc(A, n, 1.0);
    fill(B, n, 1.0);

  	square_dgemm (n, A, B, C);

    printf("\n");
    printf("After square_dgemm\n");
    printf("\n");

    printf("Matrix A\n");
    printf("\n");
    printMatrix(n,A);

    printf("\n");

    printf("Matrix B\n");
    printf("\n");

    printMatrix(n,B);

    printf("\n");
    printf("Matrix C\n");
    printf("\n");

    printMatrix(n,C);

    free(C);

  return 0;
}
