/* 
    this version could get 15.5

    grade will dip if add unfold in the inner loop

    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
*/

/*
#define a(I, J) A[(I) + ((J)) * (lda)]
#define b(I, J) B[(I) + ((J)) * (lda)]
#define c(I, J) C[(I) + ((J)) * (lda)]
*/

#include <mmintrin.h>
#include <emmintrin.h>
#include <string.h>

const char *dgemm_desc = "SSE blocked dgemm.";

#if !defined(BLOCK_SIZE)
//local memory of bridge
#define BLOCK_SIZE 256
#endif

#define unrollloop 4
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define MATRIXELEM(A, i, j) (A)[(j)*lda + (i)]


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k)
            {
                cij += A[i + k * lda] * B[k + j * lda];
            }

            C[i + j * lda] = cij;
        }
}

static void do_block_opt(int lda, int M, int N, int K, double *A, double *B, double *C)
{
    /* For each row i of A */
    int i = 0, j = 0, k = 0;
    static double aik;
    static double temp;

    //printf("curr lad (%d) M (%d) N (%d) K (%d)", lda, M, N, K);

    /* For each column j of B */
    //change the for loop to let the element accessed by row order
    for (j = 0; j < N; ++j)
    {
        for (k = 0; k < K; k++)
        {
            temp = B[k + j * lda];
            for (i = 0; i < M; ++i)
            {
                /* Compute C(i,j) */
                //double cij = C[i + j * n];
                //cij = c(i, j);
                //printf("get cij %ld\n",cij);
                //C[i + j * n] += A[i + k * n] * B[k + j * n];
                C[i + j * lda] += temp * A[i + k * lda];
            }
            //c(i, j) = cij;
        }
    }
}


//copy optimization
static inline void localise_a(int lda, const int K, double *a_src, double *a_dest)
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

static inline void localise_b(int lda, const int K, double *b_src, double *b_dest)
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

void do_block2(int lda, int M, int N, int K, double *A, double *B, double *C)
{
    double A_block[M * K], B_block[K * N];
    double *a_ptr, *b_ptr, *c;

    const int Nmax = N - 3;
    int Mmax = M - 3;
    int fringe1 = M % 4;
    int fringe2 = N % 4;

    //printf("parameter Nmax %d Mmax %d fringe1 %d fringe2 %d\n", Nmax, Mmax, fringe1, fringe2);

    int i = 0, j = 0, p = 0;

    // For each column of B
    for (j = 0; j < Nmax; j += 4)
    {
        b_ptr = &B_block[j * K];
        localise_b(lda, K, B + j * lda, b_ptr);

        for (i = 0; i < Mmax; i += 4)
        {
            a_ptr = &A_block[i * K];
            if (j == 0)
            {
                localise_a(lda, K, A + i, a_ptr);
            }

            c = C + i + j * lda;
            block_sse_4x4(lda, K, a_ptr, b_ptr, c);
        }
    }

    /* Handle "fringes" */
    if (fringe1 != 0)
    {
        /* For each row of A */
        for (; i < M; ++i)
            /* For each column of B */
            for (p = 0; p < N; ++p)
            {
                /* Compute C[i,j] */
                double c_ip = MATRIXELEM(C, i, p);
                for (int k = 0; k < K; ++k)
                {
                    c_ip += MATRIXELEM(A, i, k) * MATRIXELEM(B, k, p);
                }

                MATRIXELEM(C, i, p) = c_ip;
            }
    }
    if (fringe2 != 0)
    {
        Mmax = M - fringe1;
        /* For each column of B */
        for (; j < N; ++j)
            /* For each row of A */
            for (i = 0; i < Mmax; ++i)
            {
                /* Compute C[i,j] */
                double cij = MATRIXELEM(C, i, j);
                for (int k = 0; k < K; ++k)
                {
                    cij += MATRIXELEM(A, i, k) * MATRIXELEM(B, k, j);
                }

                MATRIXELEM(C, i, j) = cij;
            }
    }
}

void block_sse_4x4(int lda, int K, double *a, double *b, double *c)
{
    __m128d a0x_1x, a2x_3x,
        bx0, bx1, bx2, bx3,
        c00_10, c20_30,
        c01_11, c21_31,
        c02_12, c22_32,
        c03_13, c23_33;

    double *c01_11_ptr = c + lda;
    double *c02_12_ptr = c01_11_ptr + lda;
    double *c03_13_ptr = c02_12_ptr + lda;

    c00_10 = _mm_loadu_pd(c);
    c20_30 = _mm_loadu_pd(c + 2);
    c01_11 = _mm_loadu_pd(c01_11_ptr);
    c21_31 = _mm_loadu_pd(c01_11_ptr + 2);
    c02_12 = _mm_loadu_pd(c02_12_ptr);
    c22_32 = _mm_loadu_pd(c02_12_ptr + 2);
    c03_13 = _mm_loadu_pd(c03_13_ptr);
    c23_33 = _mm_loadu_pd(c03_13_ptr + 2);

    for (int x = 0; x < K; ++x)
    {
        a0x_1x = _mm_load_pd(a);
        a2x_3x = _mm_load_pd(a + 2);
        a += 4;

        // load and copy into the both position
        bx0 = _mm_load1_pd(b++);
        bx1 = _mm_load1_pd(b++);
        bx2 = _mm_load1_pd(b++);
        bx3 = _mm_load1_pd(b++);

        c00_10 = _mm_add_pd(c00_10, _mm_mul_pd(a0x_1x, bx0));
        c20_30 = _mm_add_pd(c20_30, _mm_mul_pd(a2x_3x, bx0));
        c01_11 = _mm_add_pd(c01_11, _mm_mul_pd(a0x_1x, bx1));
        c21_31 = _mm_add_pd(c21_31, _mm_mul_pd(a2x_3x, bx1));
        c02_12 = _mm_add_pd(c02_12, _mm_mul_pd(a0x_1x, bx2));
        c22_32 = _mm_add_pd(c22_32, _mm_mul_pd(a2x_3x, bx2));
        c03_13 = _mm_add_pd(c03_13, _mm_mul_pd(a0x_1x, bx3));
        c23_33 = _mm_add_pd(c23_33, _mm_mul_pd(a2x_3x, bx3));
    }

    _mm_storeu_pd(c, c00_10);
    _mm_storeu_pd((c + 2), c20_30);
    _mm_storeu_pd(c01_11_ptr, c01_11);
    _mm_storeu_pd((c01_11_ptr + 2), c21_31);
    _mm_storeu_pd(c02_12_ptr, c02_12);
    _mm_storeu_pd((c02_12_ptr + 2), c22_32);
    _mm_storeu_pd(c03_13_ptr, c03_13);
    _mm_storeu_pd((c03_13_ptr + 2), c23_33);
}

static void do_block_opt2(int lda, int M, int N, int K, double *A, double *B, double *C)
{
    /* For each row i of A */
    int i = 0, j = 0, k = 0;
    static double aik;
    static double temp;

    //printf("curr lad (%d) M (%d) N (%d) K (%d)", lda, M, N, K);

    /* For each column j of B */
    //change the for loop to let the element accessed by row order
    for (j = 0; j < N; ++j)
    {
        for (k = 0; k < K; k++)
        {
            temp = B[k + j * K];
            for (i = 0; i < M; i++)
            {

                C[i + j * lda] += temp * A[i + k * M];
            }
            //c(i, j) = cij;
        }
    }
}
//the underlying struct of the matrix is an array
void reorder(double *localMat, double *originalMat, int startr, int startc, int localM, int localN, int M, int N)
{

    int i = 0;
    for (i = 0; i < localN; i++)
    {
        memcpy(localMat + i * localM, originalMat + startr + (startc + i) * M, localM * sizeof(double));
    }
    return;
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
    int M, N, K;
    int i, j, k;

    //get local memory space
    double *localMatrxA = (double *)malloc(sizeof(double) * BLOCK_SIZE * BLOCK_SIZE);
    double *localMatrxB = (double *)malloc(sizeof(double) * BLOCK_SIZE * BLOCK_SIZE);

    /* For each block-column of B */
    for (j = 0; j < lda; j += BLOCK_SIZE)
    {
        N = min(BLOCK_SIZE, lda - j);
        /* Accumulate block dgemms into block of C */
        for (k = 0; k < lda; k += BLOCK_SIZE)
        {
            K = min(BLOCK_SIZE, lda - k);
            //reorder(localMatrxB, B, k, j, K, N, lda, lda);

            /* For each block-row of A */
            for (i = 0; i < lda; i += BLOCK_SIZE)
            {
                M = min(BLOCK_SIZE, lda - i);
                /* Correct block dimensions if block "goes off edge of" the matrix */

                /* Perform individual block dgemm */
                //do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                //copy data to local space before do this block opt???
                //memcpy(localMatrxA, A + i + k * lda, (M + 1) * (K + 1) * sizeof(double));

                //do_block_opt(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                //void reorder(double *localMat, double *originalMat, int startr, int startc, int localM, int localN, int M, int N)
                reorder(localMatrxA, A, i, k, M, K, lda, lda);
                reorder(localMatrxB, B, k, j, K, N, lda, lda);
                //do_block_opt(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                //printf("original %lf %lf %lf %lf, local %lf %lf %lf %lf\n",
                //       A + i + k * lda, A + i + k * lda + 1, A + i + k * lda + 2, A + i + k * lda + 3,
                //       localMatrxA, localMatrxA + 1, localMatrxA + 2, localMatrxA + 3);
                //do_block_opt2(lda, M, N, K, localMatrxA, localMatrxB, C + i + j * lda);
                if (M <= 3 || N <= 3)
                {

                    do_block_opt2(lda, M, N, K, localMatrxA, localMatrxB, C + i + j * lda);
                }
                else
                {
                    do_block2(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
            }
        }
    }
}
