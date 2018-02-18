/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";
// can use -DBLOCK_SIZE to set block size
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#define L2_SIZE_BYTES 8192
#define L2_SQURED L2_SIZE_BYTES/2/sizeof(double)
#endif

#define min(a,b) (((a)<(b))?(a):(b))
void copy_a(int lda, const int K, double *a_src, double *a_dest);
void copy_b(int lda, const int K, double *b_src, double *b_dest);

void avx_basic(int lda, int K, double * restrict a, double * restrict b, double * restrict c);
inline void Transposed_Packed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * restrict A_Block, const double * restrict A);

void square_dgemm (int lda, double* A, double* B, double* C);

#define Test 

#ifdef Test

void printMatrix(const int lda,const double* A );
void printMatrixLinear(const int lda,const double* A );

#endif

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
// static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
// {
//   /* For each row i of A */
//   for (int i = 0; i < M; ++i)
//     /* For each column j of B */ 
//     for (int j = 0; j < N; ++j) 
//     {
//       /* Compute C(i,j) */
//       double cij = C[i+j*lda];
//       for (int k = 0; k < K; ++k)
// 	cij += A[i+k*lda] * B[k+j*lda];
//       C[i+j*lda] = cij;
//     }
// }


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block(const int lda, const int M, const int N, const int K, double * restrict A, double *restrict B, double  *restrict C)
{

  // printf("\n");
  // printf("Inside do block -----------------------------------------------------------------------------------\n");
  // printf("lda: %d M: %d N: %d K: %d\n" , lda, M, N, K );
  // printf("\n");
  // printf("\n");
  // printf("Matrix A\n");-
  // printMatrix(M,A );
  // printf("Matrix B\n");
  // printMatrix(N,B);
  // printf("\n");
  // printf("\n");
  // printf("\n");
  double A_block[M * K], B_block[K * N];
  register double *a_ptr, *b_ptr, *c;

  int Mmax = M - 3;
  const int Nmax = N - 3;
  const int MedgeRemainder = M % 4;

  // loop variables. declared globally  since we need to handle "fringes" after the AVX instructions. 
    int i = 0;
    int j = 0;
    int p = 0;

    /* For each column of B */
    for (j = 0; j < Nmax; j += 4)
    {
        b_ptr = &B_block[j * K]; //start address of each B column
        // copy and transpose B_block
        //copy_b(lda, K, B + j * lda, b_ptr);
        copy_b(lda, K, B + j * lda, b_ptr);
        /* For each row of A */
        for (i = 0; i < Mmax; i += 4)
        {
            a_ptr = &A_block[i * K];
            copy_a(lda, K, A + i, a_ptr);
            c = C + i + j * lda;
            avx_basic(lda, K, a_ptr, b_ptr, c);
        }
    }

      //////// correct for the edges on the side of the matrix 
        /* For each row of A */
        for (; i < M; ++i)  //M is i  or row
        {
            /* For each column of B */
            for (p = 0; p < N; ++p)
            {
                /* Compute C[i,j] */
                register double c_ip = 0; 
                //for each element
                for (int k = 0; k < K; ++k)
                {
                    c_ip +=  A[i+k*lda] * B[k+ p*lda];
                }
                C[i+ p*lda] += c_ip;
            }
          }

        Mmax = M - MedgeRemainder;
        /* For each remaining column of B */
        for (; j < N; ++j)  //N is i  or col
        {
            /* For eveey row of A  not just the fringes*/
            for (i = 0; i < Mmax; ++i)
            {
                /* Compute C[i,j] */
                register double cij = 0; 
                // for each element
                for (int k = 0; k < K; ++k)
                {
                    cij += A[i + k*lda] * B[k + j*lda];
                }
                C[i+j*lda] += cij;
            }
        }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
inline void square_dgemm (int lda, double* A, double* B, double* C)
{

    double A_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
    double B_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
    double C_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];

    int M = 0;
    int K = 0;
    int N = 0;
      /* For each block-column of B */

  //transpose(A, A_Block, lda);
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
      /* Correct block dimensions if block "goes off edge of" the matrix */
       N = min (BLOCK_SIZE, lda-j);
      /* For each block-row of A */ 
    
    for (int k = 0; k < lda; k += BLOCK_SIZE)
    {
     /* Correct block dimensions if block "goes off edge of" the matrix */
        K = min (BLOCK_SIZE, lda-k);
        //direct_Blocked_Copy(lda, BLOCK_SIZE,N,K, k, j,B_Block,B);
        Transposed_Packed_Blocked_Copy(lda, BLOCK_SIZE,N,K,k,j,B_Block,B);

          for (int i = 0; i < lda; i += BLOCK_SIZE)
          {
            /* Correct block dimensions if block "goes off edge of" the matrix */
              M = min (BLOCK_SIZE, lda-i);
              //Transposed_Packed_Blocked_Copy(lda, BLOCK_SIZE,M,K,k,i,A_Block,A);
              //direct_Blocked_Copy(lda, BLOCK_SIZE,N,M, k, j,C_Block,C);
                      /* Accumulate block dgemms into block of C */

              //----------------------------------------------------------------------------------
        	     /* Perform individual block dgemm */
        	     do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            //-------------------------------------------------------------------------------
                 //do_block(lda, M, N, K, A_Block + k + i*lda, B + k + j*lda, C + i + j*lda);

            }

           // write back C_Block to C
        }
    }



}


void avx_basic(int lda, int K, double * restrict a, double * restrict b, double * restrict c);
{
    // adapted from code taken from the power point sent out to the class 
    // Min AVX required 
    // declare 256 bit registers 
    __m256d a0x_1x_a2x_3x,
        bx0, bx1, bx2, bx3,
        c00_10_c20_30,
        c01_11_c21_31,
        c02_12_c22_32,
        c03_13_c23_33;

    double *c01_11_ptr = c + lda;
    double *c02_12_ptr = c01_11_ptr + lda;
    double *c03_13_ptr = c02_12_ptr + lda;

    // load an entire column  
    //256bits is 4 64bit doubles 
    c00_10_c20_30 = _mm256_loadu_pd(c);

    c01_11_c21_31 = _mm256_loadu_pd(c01_11_ptr);

    c02_12_c22_32 = _mm256_loadu_pd(c02_12_ptr);

    c03_13_c23_33 = _mm256_loadu_pd(c03_13_ptr);

    // we need to loop k times to get the correct values
    for (int x = 0; x < K; ++x)
    {
        // load an entire 4 element row of matrix A 
        a0x_1x_a2x_3x = _mm256_loadu_pd(a);

        // move the a pointer to the next row 
        a += 4;

        // load and copy the same element into all 4 positions of the 256 bit vector   
        bx0 = _mm256_broadcast_sd(b++);
        bx1 = _mm256_broadcast_sd(b++);
        bx2 = _mm256_broadcast_sd(b++);
        bx3 = _mm256_broadcast_sd(b++);

        c00_10_c20_30 = _mm256_add_pd(c00_10_c20_30, _mm256_mul_pd(a0x_1x_a2x_3x,bx0));
        //c00_10_c20_30 = _mm256_fmadd_pd(c00_10_c20_30, a0x_1x_a2x_3x,bx0 );

        c01_11_c21_31 = _mm256_add_pd(c01_11_c21_31, _mm256_mul_pd(a0x_1x_a2x_3x,bx1 ));
        //c01_11_c21_31 = _mm256_fmadd_pd(c01_11_c21_31,a0x_1x_a2x_3x,bx1);

        c02_12_c22_32 = _mm256_add_pd(c02_12_c22_32, _mm256_mul_pd(a0x_1x_a2x_3x,bx2));
        //c02_12_c22_32 = _mm256_fmadd_pd(c02_12_c22_32,a0x_1x_a2x_3x,bx2);

        c03_13_c23_33 = _mm256_add_pd(c03_13_c23_33, _mm256_mul_pd(a0x_1x_a2x_3x,bx3));
        //c03_13_c23_33 = _mm256_fmadd_pd(c03_13_c23_33,a0x_1x_a2x_3x,bx3);

    }

    _mm256_storeu_pd(c, c00_10_c20_30);

    // _mm_storeu_pd(c01_11_ptr, c01_11);
    // _mm_storeu_pd((c01_11_ptr + 2), c21_31);

    _mm256_storeu_pd(c01_11_ptr, c01_11_c21_31);

    // _mm_storeu_pd(c02_12_ptr, c02_12);
    // _mm_storeu_pd((c02_12_ptr + 2), c22_32);

    _mm256_storeu_pd(c02_12_ptr, c02_12_c22_32);

    // _mm_storeu_pd(c03_13_ptr, c03_13);
    // _mm_storeu_pd((c03_13_ptr + 2), c23_33);

    _mm256_storeu_pd(c03_13_ptr, c03_13_c23_33);

}


inline void copy_a(int lda, const int K, double *a_src, double *a_dest)
{
    /* For each 4xK block-row of A */
    for (int i = 0; i < K; ++i)
    {
        *a_dest++ = *a_src; //column major, each itero read 4 elements from 4 consecutive row
        *a_dest++ = *(a_src + 1);
        *a_dest++ = *(a_src + 2);
        *a_dest++ = *(a_src + 3);
        a_src += lda;
    }
}

inline void copy_b(int lda, const int K, double *b_src, double *b_dest)
{
    double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;
    b_ptr2 = b_ptr1 + lda;
    b_ptr3 = b_ptr2 + lda;

    for (int i = 0; i < K; ++i)
    {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
        *b_dest++ = *b_ptr2++;
        *b_dest++ = *b_ptr3++;
    }
}


inline void direct_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * B_Block, const double * B)
{

    register int ColOrginal = 0;
    register int ColMapped = 0;
    double * Bptr = B + K_Offset;

    for (int J_load = 0; J_load < N; ++J_load)
    {
      ColOrginal = (J_load + J_Offset) *lda; /// the col from the orginal matrix
      ColMapped = J_load * block; // the newly mapped index in the blocked matrix

      for (int K_load = 0; K_load < K; K_load+=4)
      {
        //j_load = 0
        // B_Block[K_load + ColMapped] = B[K_load + K_Offset + ColOrginal];
        // B_Block[K_load + ColMapped + 1] = B[K_load + K_Offset + ColOrginal + 1];
        // B_Block[K_load + ColMapped + 2] = B[K_load + K_Offset + ColOrginal + 2];
        // B_Block[K_load + ColMapped + 3] = B[K_load + K_Offset + ColOrginal + 3];
        B_Block[K_load + ColMapped]     = Bptr[K_load + ColOrginal];
        B_Block[K_load + ColMapped + 1] = Bptr[K_load + ColOrginal + 1];
        B_Block[K_load + ColMapped + 2] = Bptr[K_load + ColOrginal + 2];
        B_Block[K_load + ColMapped + 3] = Bptr[K_load + ColOrginal + 3];

      } 
    } 
}

inline void Transposed_Packed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * restrict A_Block, const double * restrict A)
{
     double * Aptr = A + i_Offset;

    for(int I_load = 0; I_load < M; ++I_load)
    {
      //int ColOrginal = ((K_load + K_Offset) *lda); /// the col from the orginal matrix
      //int RowMapped = (I_load * block); // the newly mapped index in the blocked matrix
      for (int K_load = 0; K_load < K; K_load+=4)
      {
          // A_Block[K_load + (I_load * block)] =  A[i_Offset + I_load + ((K_load + K_Offset) *lda)];
          // A_Block[K_load + (I_load * block) + 1] =  A[i_Offset + I_load + ((K_load +1  + K_Offset) *lda)];
          // A_Block[K_load + (I_load * block) + 2] =  A[i_Offset + I_load + ((K_load + 2 + K_Offset) *lda)];
          // A_Block[K_load + (I_load * block) + 3] =  A[i_Offset + I_load + ((K_load + 3 + K_Offset) *lda)];
          A_Block[K_load + (I_load * block)]     =  Aptr[I_load + ((K_load + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 1] =  Aptr[I_load + ((K_load +1  + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 2] =  Aptr[I_load + ((K_load + 2 + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 3] =  Aptr[I_load + ((K_load + 3 + K_Offset) *lda)];
      } 

    } // for(int I_load = 0; I_load < M; ++I_load)
}


// inline void transpose(const double *A, double * B, const int lda)
// {
//     int i, j;
//     for (j = 0; j < lda; j++)
//     {
//       int j_lda = j*lda;
//       for (i = 0; i < lda; i++) 
//       {
//           //B[i][j] = A[j][i];
//           B[i + j_lda] = A[j + i * lda];

//       }
//     }
// }


#ifdef Test


void printMatrix(const int lda,const double* A )
{

  //int numElements =lda * lda; 
  //row
  for(int i = 0; i < lda; ++i)
  {
    //col
    for (int j = 0; j < lda; ++j) // each column
    { /* For each block-column of B */
        printf(" %f", A[j + i*lda]);
    }
        printf("\n");
  }
}


void printMatrixLinear(const int lda,const double* A )
{
  //int numElements =lda * lda; 
  //row
  int limit = lda*lda;
  for(int i = 0; i < limit; i++)
  {
        printf(" %f", A[i]);
  }
}

#endif