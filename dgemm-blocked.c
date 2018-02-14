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

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

void transpose(const double *A, double * B, const int lda);
void direct_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * B_Block, const double * B);
//void Transposed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * A_Block, const double * A);

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int block, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */

  for (int j = 0; j < N; ++j) 
  {

      for (int i = 0; i < M; i+=4)
      {
        /* For each column j of B */ 
          /* Compute C(i,j) */
          // unneeded memory access. 
          register double cij = 0;
          // register double cij0 = 0; 
          // register double cij1 = 0; 
          // register double cij2 = 0; 
          // register double cij3 = 0; 

            for (int k = 0; k < K; ++k)
            {
      	       cij += A[i+k*lda] * B[k+j*lda];
              // cij0 += A[k+i*block] * B[k+j*block];
              // cij1 += A[k+((1+i)*block)] * B[k+j*block];
              // cij2 += A[k+((2+i)*block)] * B[k+j*block];
              // cij3 += A[k+((3+i)*block)] * B[k+j*block];

            }

          C[i+j*lda] += cij;
          // C[i+1+j*lda] += cij1;
          // C[i+2+j*lda] += cij2;
          // C[i+3+j*lda] += cij3;


        }

    }
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



inline void direct_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * B_Block, const double * B)
{

    register int ColOrginal = 0;
    register int ColMapped = 0;

    for (int J_load = 0; J_load < N; ++J_load)
    {
      ColOrginal = (J_load + J_Offset) *lda; /// the col from the orginal matrix
      ColMapped = J_load * block; // the newly mapped index in the blocked matrix

      for (int K_load = 0; K_load < K; K_load+=4)
      {
        //j_load = 0
        B_Block[K_load + ColMapped] = B[K_load + K_Offset + ColOrginal];
        B_Block[K_load + ColMapped + 1] = B[K_load + K_Offset + ColOrginal + 1];
        B_Block[K_load + ColMapped + 2] = B[K_load + K_Offset + ColOrginal + 2];
        B_Block[K_load + ColMapped + 3] = B[K_load + K_Offset + ColOrginal + 3];

        //j_load = 1


        //j_load = 2


        //j_load = 3


      } 
    } 
}

inline void Transposed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * A_Block, const double * A)
{


    for(int I_load = 0; I_load < M; ++I_load)
    {
      //int ColOrginal = ((K_load + K_Offset) *lda); /// the col from the orginal matrix
      //int RowMapped = (I_load * block); // the newly mapped index in the blocked matrix
      for (int K_load = 0; K_load < K; K_load+=4)
      {
          A_Block[K_load + (I_load * block)] =  A[i_Offset + I_load + ((K_load + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 1] =  A[i_Offset + I_load + ((K_load +1  + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 2] =  A[i_Offset + I_load + ((K_load + 2 + K_Offset) *lda)];
          A_Block[K_load + (I_load * block) + 3] =  A[i_Offset + I_load + ((K_load + 3 + K_Offset) *lda)];
      } 

    } // for(int I_load = 0; I_load < M; ++I_load)
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{

  double B_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
  double A_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
  //double* buf = (double*)calloc(4*lda*lda, sizeof(double));
  int M = 0;
  int K = 0;
  int N = 0;

  //transpose(A, buf, lda);
  /* For each block-row of A */ 
  for (int j = 0; j < lda; j += BLOCK_SIZE)
  {
    //int j_mul = j*lda;
    N = min (BLOCK_SIZE, lda-j);
    /* Accumulate block dgemms into block of C */
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          K = min (BLOCK_SIZE, lda-k);

          /* For each block-column of B */
          direct_Blocked_Copy(lda, BLOCK_SIZE,N,K, k, j,B_Block,B);

              for (int i = 0; i < lda; i += BLOCK_SIZE)
              {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                M = min (BLOCK_SIZE, lda-i);
                Transposed_Blocked_Copy(lda, BLOCK_SIZE,M,K,k,i,A_Block,A);
            
            	
            	/* Perform individual block dgemm */
                do_block(lda, BLOCK_SIZE, M, N, K, A_Block, B_Block, C + i + j*lda);
              }
        }
  }

}
