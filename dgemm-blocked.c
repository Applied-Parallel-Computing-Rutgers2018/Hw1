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

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#define L2_SIZE_BYTES 8192
#define L2_SQURED L2_SIZE_BYTES/2/sizeof(double)
#endif

#define ARRAY(A, i, j) (A)[(j)*lda + (i)]

#define min(a,b) (((a)<(b))?(a):(b))
void copy_a(int lda, const int K, double *a_src, double *a_dest);
void copy_b(int lda, const int K, double *b_src, double *b_dest);
void avx_basic(int lda, int K, double *a, double *b, double *c);
//void sse_basic(int lda, int K, double *a, double *b, double *c);

void square_dgemm (int lda, double* A, double* B, double* C);

void printMatrix(const int lda,const double* A );
void printMatrixLinear(const int lda,const double* A );


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
static inline void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{

  // printf("\n");
  // printf("Inside do block -----------------------------------------------------------------------------------\n");
  // printf("lda: %d M: %d N: %d K: %d\n" , lda, M, N, K );
  // printf("\n");
  // printf("\n");
  // printf("Matrix A\n");
  // printMatrix(M,A );
  // printf("Matrix B\n");
  // printMatrix(N,B);
  // printf("\n");
  // printf("\n");
  // printf("\n");
  double A_block[M * K], B_block[K * N];
  register double *a_ptr, *b_ptr, *c;


  const int Nmax = N - 3;
    int Mmax = M - 3;
  const int MedgeRemainder = M % 4;

    int i = 0, j = 0, p = 0;

    /* For each column of B */
    for (j = 0; j < Nmax; j += 4)
    {
        b_ptr = &B_block[j * K]; //start address of each B column
        // copy and transpose B_block
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
                double c_ip = 0; //ARRAY(C, i, p);
                for (int k = 0; k < K; ++k)
                {
                    c_ip += ARRAY(A, i, k) * ARRAY(B, k, p);
                }
                ARRAY(C, i, p) += c_ip;
            }
          }

        Mmax = M - MedgeRemainder;
        /* For each column of B */
        for (; j < N; ++j)  //N is i  or col
        {
            /* For each row of A */
            for (i = 0; i < Mmax; ++i)
            {
                /* Compute C[i,j] */
                double cij = 0; //ARRAY(C, i, j);
                for (int k = 0; k < K; ++k)
                {
                    cij += ARRAY(A, i, k) * ARRAY(B, k, j);
                }
                ARRAY(C, i, j) += cij;
            }
        }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
inline void square_dgemm (int lda, double* A, double* B, double* C)
{

  int M = 0;
  int K = 0;
  int N = 0;
  /* For each block-column of B */
for (int j = 0; j < lda; j += BLOCK_SIZE)
{
  /* Correct block dimensions if block "goes off edge of" the matrix */
   N = min (BLOCK_SIZE, lda-j);
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
  {
    /* Correct block dimensions if block "goes off edge of" the matrix */
      M = min (BLOCK_SIZE, lda-i);

      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	     /* Correct block dimensions if block "goes off edge of" the matrix */
      	K = min (BLOCK_SIZE, lda-k);

	     /* Perform individual block dgemm */
	     do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
  }
}



}


// inline void sse_basic(int lda, int K, double *a, double *b, double *c)
// {

//     // declare 128 bit registers 
//     __m128d a0x_1x, a2x_3x,
//         bx0, bx1, bx2, bx3,
//         c00_10, c20_30,
//         c01_11, c21_31,
//         c02_12, c22_32,
//         c03_13, c23_33;

//     double *c01_11_ptr = c + lda;
//     double *c02_12_ptr = c01_11_ptr + lda;
//     double *c03_13_ptr = c02_12_ptr + lda;

//     c00_10 = _mm_loadu_pd(c);
//     c20_30 = _mm_loadu_pd(c + 2);

//     c01_11 = _mm_loadu_pd(c01_11_ptr);
//     c21_31 = _mm_loadu_pd(c01_11_ptr + 2);


//     c02_12 = _mm_loadu_pd(c02_12_ptr);
//     c22_32 = _mm_loadu_pd(c02_12_ptr + 2);


//     c03_13 = _mm_loadu_pd(c03_13_ptr);
//     c23_33 = _mm_loadu_pd(c03_13_ptr + 2);

//     for (int x = 0; x < K; ++x)
//     {
//         a0x_1x = _mm_load_pd(a);
//         a2x_3x = _mm_load_pd(a + 2);
//         a += 4;

//         // load and copy into the both position
//         bx0 = _mm_load1_pd(b++);
//         bx1 = _mm_load1_pd(b++);
//         bx2 = _mm_load1_pd(b++);
//         bx3 = _mm_load1_pd(b++);

//         c00_10 = _mm_add_pd(c00_10, _mm_mul_pd(a0x_1x, bx0));
//         c20_30 = _mm_add_pd(c20_30, _mm_mul_pd(a2x_3x, bx0));



//         c01_11 = _mm_add_pd(c01_11, _mm_mul_pd(a0x_1x, bx1));
//         c21_31 = _mm_add_pd(c21_31, _mm_mul_pd(a2x_3x, bx1));


//         c02_12 = _mm_add_pd(c02_12, _mm_mul_pd(a0x_1x, bx2));
//         c22_32 = _mm_add_pd(c22_32, _mm_mul_pd(a2x_3x, bx2));


//         c03_13 = _mm_add_pd(c03_13, _mm_mul_pd(a0x_1x, bx3));
//         c23_33 = _mm_add_pd(c23_33, _mm_mul_pd(a2x_3x, bx3));
//     }

//     _mm_storeu_pd(c, c00_10);
//     _mm_storeu_pd((c + 2), c20_30);
//     _mm_storeu_pd(c01_11_ptr, c01_11);
//     _mm_storeu_pd((c01_11_ptr + 2), c21_31);
//     _mm_storeu_pd(c02_12_ptr, c02_12);
//     _mm_storeu_pd((c02_12_ptr + 2), c22_32);
//     _mm_storeu_pd(c03_13_ptr, c03_13);
//     _mm_storeu_pd((c03_13_ptr + 2), c23_33);
// }


inline void avx_basic(int lda, int K, double *a, double *b, double *c)
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

    // c00_10 = _mm_loadu_pd(c);
    // c20_30 = _mm_loadu_pd(c + 2);
    // load an entire column  
    //256bits is 4 64bit doubles 
    c00_10_c20_30 = _mm256_loadu_pd(c);

    // c01_11 = _mm_loadu_pd(c01_11_ptr);
    // c21_31 = _mm_loadu_pd(c01_11_ptr + 2);

    c01_11_c21_31 = _mm256_loadu_pd(c01_11_ptr);

    // c02_12 = _mm_loadu_pd(c02_12_ptr);
    // c22_32 = _mm_loadu_pd(c02_12_ptr + 2);

    c02_12_c22_32 = _mm256_loadu_pd(c02_12_ptr);
    
    // c03_13 = _mm_loadu_pd(c03_13_ptr);
    // c23_33 = _mm_loadu_pd(c03_13_ptr + 2);

    c03_13_c23_33 = _mm256_loadu_pd(c03_13_ptr);

    for (int x = 0; x < K; ++x)
    {
        // a0x_1x = _mm_load_pd(a);
        // a2x_3x = _mm_load_pd(a + 2);

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
        //c00_10_c20_30 = _mm256_fmadd_pd(c00_10_c20_30, a0x_1x_a2x_3x, );

        c01_11_c21_31 = _mm256_add_pd(c01_11_c21_31, _mm256_mul_pd(a0x_1x_a2x_3x,bx1 ));
        //c01_11_c21_31 = _mm256_fmadd_pd(c01_11_c21_31,a0x_1x_a2x_3x,);

        c02_12_c22_32 = _mm256_add_pd(c02_12_c22_32, _mm256_mul_pd(a0x_1x_a2x_3x,bx2));
        //c02_12_c22_32 = _mm256_fmadd_pd(c02_12_c22_32,a0x_1x_a2x_3x,);

        c03_13_c23_33 = _mm256_add_pd(c03_13_c23_33, _mm256_mul_pd(a0x_1x_a2x_3x,bx3));
        //c03_13_c23_33 = _mm256_fmadd_pd(c03_13_c23_33,a0x_1x_a2x_3x,);

    }

    // _mm_storeu_pd(c, c00_10);
    // _mm_storeu_pd((c + 2), c20_30);

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


// /////////////////////////////////////////////  AVX  ///////////////////////////////////////////////////////
// ///////////// My desktop does not support AVX :( ??????????????????/

// static inline __m256 twolincomb_AVX_8(__m256 A01, const Mat44 &B)
// {
//     __m256 result;
//     result = _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x00), _mm256_broadcast_ps(&B.row[0]));
//     result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x55), _mm256_broadcast_ps(&B.row[1])));
//     result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xaa), _mm256_broadcast_ps(&B.row[2])));
//     result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xff), _mm256_broadcast_ps(&B.row[3])));
//     return result;
// }

// // this should be noticeably faster with actual 256-bit wide vector units (Intel);
// // not sure about double-pumped 128-bit (AMD), would need to check.
// void matmult_AVX_8(Mat44 &out, const Mat44 &A, const Mat44 &B)
// {
//     _mm256_zeroupper();
//     __m256 A01 = _mm256_loadu_ps(&A.m[0][0]);
//     __m256 A23 = _mm256_loadu_ps(&A.m[2][0]);
    
//     __m256 out01x = twolincomb_AVX_8(A01, B);
//     __m256 out23x = twolincomb_AVX_8(A23, B);

//     _mm256_storeu_ps(&out.m[0][0], out01x);
//     _mm256_storeu_ps(&out.m[2][0], out23x);
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






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
