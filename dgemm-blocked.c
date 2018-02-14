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
void Transposed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * A_Block, const double * A);

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int block, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
      {
	       //cij += A[i+k*lda] * B[k+j*lda];
        double Aelement = A[k+i*block];
        double Belement = B[k+j*block];;
        cij += Aelement * Belement;
      }
      C[i+j*lda] = cij;
    }
}


inline void transpose(const double *A, double * B, const int lda)
{
    int i, j;
    for (j = 0; j < lda; j++)
    {
      int j_lda = j*lda;
      for (i = 0; i < lda; i++) 
      {
          //B[i][j] = A[j][i];
          B[i + j_lda] = A[j + i * lda];

      }
    }
}



inline void direct_Blocked_Copy(const int lda, const int block, const int N, const int K, const int K_Offset, const int J_Offset, double * B_Block, const double * B)
{
    for (int J_load = 0; J_load < N; ++J_load)
    {
      for (int K_load = 0; K_load < K; ++K_load)
      {
        //B_Block[K_load + J_load*BLOCK_SIZE] = B[k_Block + K_load + J_load*lda];
        B_Block[K_load + J_load * block] = B[K_load + K_Offset + (J_load + J_Offset) *lda];
          //B_Block[Index] =  B[k_Block + Index];
      } 
    } 
}

inline void Transposed_Blocked_Copy(const int lda, const int block, const int M, const int K, const int K_Offset, const int i_Offset, double * A_Block, const double * A)
{
    for(int I_load = 0; I_load < M; ++I_load)
    {
      for (int K_load = 0; K_load < K; ++K_load)
      {
          //A_Block[K_load + I_load * BLOCK_SIZE] =  A[K_load + k_Block + (I_load + j_Block) *lda];
          A_Block[K_load + (I_load * block)] =  A[i_Offset + I_load + ((K_load + K_Offset) *lda)];
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
