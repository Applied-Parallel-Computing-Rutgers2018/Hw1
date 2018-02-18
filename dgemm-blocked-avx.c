/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O2 -mavx
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

//use SSE to process the fringe case

#include <emmintrin.h> //SSE intrinsic
#include <immintrin.h> //AVX intrinsics
#include <stdio.h>
const char* dgemm_desc = "8*4 AVX & SSE optized matrix multiplication";

#define BLOCK_SIZE 128
#define B_buf_size 2048

//width of B matrix temporary buffer size,
//shall larger then the largest benchmark input

#define min( i, j ) ( (i)<(j) ? (i): (j) )

// This subroutine copy an 8 by k block of A
void copy8toA(int k, double *A, int lda, double *A_copy)
{
  int j;
  //A is packed in column-major order
  //simultaneously load 8 rows and iteratively for k columns
  for(j = 0; j<k; j++) 
  {
    double *a_ptr = A+j*lda;
    *A_copy = *a_ptr;
    *(A_copy+1) = *(a_ptr+1);
    *(A_copy+2) = *(a_ptr+2);
    *(A_copy+3) = *(a_ptr+3);
    *(A_copy+4) = *(a_ptr+4);
    *(A_copy+5) = *(a_ptr+5);
    *(A_copy+6) = *(a_ptr+6);
    *(A_copy+7) = *(a_ptr+7);

    A_copy += 8;
  }
}

void copy4toB(int k, double *B, int ldb, double *B_copy)
{
  int i;
  double *b_i0 = B, *b_i1 = B+ldb, *b_i2 = B+2*ldb, *b_i3 = B+3*ldb;

  // B is packed in row-major order
  //i is the row index, and simultaneously load 4 column
  for(i=0; i<k; i++)
  {
    *B_copy++ = *b_i0++;
    *B_copy++ = *b_i1++;
    *B_copy++ = *b_i2++;
    *B_copy++ = *b_i3++;
  }
}

// updates an 8 by 4 block of C at a time using AVX intrinsic
void update8X4(int k, double *A, int lda, double* B, int ldb, double *C, int ldc)
{
  int p;
  __m256d c_00_30, c_01_31, c_02_32, c_03_33, c_40_70, c_41_71, c_42_72, c_43_73,
          a_0p_3p, a_4p_7p,
          b_p0, b_p1, b_p2, b_p3;

  c_00_30 = _mm256_loadu_pd(C);
  c_01_31 = _mm256_loadu_pd(C+ldc);
  c_02_32 = _mm256_loadu_pd(C+ldc*2);
  c_03_33 = _mm256_loadu_pd(C+ldc*3);
  c_40_70 = _mm256_loadu_pd(C+4);
  c_41_71 = _mm256_loadu_pd(C+4+ldc);
  c_42_72 = _mm256_loadu_pd(C+4+ldc*2);
  c_43_73 = _mm256_loadu_pd(C+4+ldc*3);


  for(p=0; p<k; p++)
  {
    // Load A, 8 rows single column
    a_0p_3p = _mm256_loadu_pd(A);
    a_4p_7p = _mm256_loadu_pd(A+4);
    A += lda;
    
    // Load B, single row, 4 columns
    b_p0 = _mm256_broadcast_sd(B);     
    b_p1 = _mm256_broadcast_sd(B+1); 
    b_p2 = _mm256_broadcast_sd(B+2); 
    b_p3 = _mm256_broadcast_sd(B+3); 

    B += 4; 

    // First four rows of C updated once
    c_00_30 = _mm256_add_pd(c_00_30, _mm256_mul_pd(a_0p_3p, b_p0));
    c_01_31 = _mm256_add_pd(c_01_31, _mm256_mul_pd(a_0p_3p, b_p1));
    c_02_32 = _mm256_add_pd(c_02_32, _mm256_mul_pd(a_0p_3p, b_p2));
    c_03_33 = _mm256_add_pd(c_03_33, _mm256_mul_pd(a_0p_3p, b_p3));

    // Last four rows of C updated once
    c_40_70 = _mm256_add_pd(c_40_70, _mm256_mul_pd(a_4p_7p, b_p0));
    c_41_71 = _mm256_add_pd(c_41_71, _mm256_mul_pd(a_4p_7p, b_p1));
    c_42_72 = _mm256_add_pd(c_42_72, _mm256_mul_pd(a_4p_7p, b_p2));
    c_43_73 = _mm256_add_pd(c_43_73, _mm256_mul_pd(a_4p_7p, b_p3));
  }

//store back C
  _mm256_storeu_pd(C, c_00_30);
  _mm256_storeu_pd(C+ldc, c_01_31);
  _mm256_storeu_pd(C+ldc*2, c_02_32);
  _mm256_storeu_pd(C+ldc*3, c_03_33);
  _mm256_storeu_pd(C+4, c_40_70);
  _mm256_storeu_pd(C+4+ldc, c_41_71);
  _mm256_storeu_pd(C+4+ldc*2, c_42_72);
  _mm256_storeu_pd(C+4+ldc*3, c_43_73);
}


void copy3toB(int k, double *B, int ldb, double *B_copy)
{
  int i;
  double *b_i0 = B, *b_i1 = B+ldb, *b_i2 = B+2*ldb;

  // B is packed in row-major form
  //i is the row index, and simultaneously load 3 column
  for(i=0; i<k; i++)
  {
    *B_copy++ = *b_i0++;
    *B_copy++ = *b_i1++;
    *B_copy++ = *b_i2++;
  }
}


//Addressing fringe case, updates an 8 by 4 block of C at a time
void update8X3(int k, double *A, int lda, double* B, int ldb, double *C, int ldc)
{
  int p;
  __m256d c_00_30, c_01_31, c_02_32, c_40_70, c_41_71, c_42_72,
          a_0p_3p, a_4p_7p,
          b_p0, b_p1, b_p2;

  c_00_30 = _mm256_loadu_pd(C);
  c_01_31 = _mm256_loadu_pd(C+ldc);
  c_02_32 = _mm256_loadu_pd(C+ldc*2);
  c_40_70 = _mm256_loadu_pd(C+4);
  c_41_71 = _mm256_loadu_pd(C+4+ldc);
  c_42_72 = _mm256_loadu_pd(C+4+ldc*2);


  for(p=0; p<k; p++)
  {
    // Load A, 8 rows single column
    a_0p_3p = _mm256_loadu_pd(A);
    a_4p_7p = _mm256_loadu_pd(A+4);
    A += lda;
    
    // Load B, single row, 3 columns
    b_p0 = _mm256_broadcast_sd(B);     
    b_p1 = _mm256_broadcast_sd(B+1); 
    b_p2 = _mm256_broadcast_sd(B+2); 

    B += 3; 

    // First four rows of C updated once
    c_00_30 = _mm256_add_pd(c_00_30, _mm256_mul_pd(a_0p_3p, b_p0));
    c_01_31 = _mm256_add_pd(c_01_31, _mm256_mul_pd(a_0p_3p, b_p1));
    c_02_32 = _mm256_add_pd(c_02_32, _mm256_mul_pd(a_0p_3p, b_p2));

    // Last four rows of C updated once
    c_40_70 = _mm256_add_pd(c_40_70, _mm256_mul_pd(a_4p_7p, b_p0));
    c_41_71 = _mm256_add_pd(c_41_71, _mm256_mul_pd(a_4p_7p, b_p1));
    c_42_72 = _mm256_add_pd(c_42_72, _mm256_mul_pd(a_4p_7p, b_p2));
  }

  _mm256_storeu_pd(C, c_00_30);
  _mm256_storeu_pd(C+ldc, c_01_31);
  _mm256_storeu_pd(C+ldc*2, c_02_32);
  _mm256_storeu_pd(C+4, c_40_70);
  _mm256_storeu_pd(C+4+ldc, c_41_71);
  _mm256_storeu_pd(C+4+ldc*2, c_42_72);
}



// This is the subfunction to address fringe cases
void dgemm_fringe(int lda, int m, int n, int k, double *A, double *B, double* C)
{
  if (m == 0 || n == 0) return;

   //Here use SSE intrinsic to calculate 2*2 block
  int r, c, p;
  __m128d c_00_10, c_01_11,
          a_0p_1p,
          b_p0, b_p1;
  for(r = 0; r < m/2*2; r+=2)
  {
    for(c = 0; c < n/2*2; c+=2)
    {
      // This is updating a 2X2 block at (r, c)
      c_00_10 = _mm_loadu_pd(C+r+c*lda);
      c_01_11 = _mm_loadu_pd(C+r+(c+1)*lda);

      // Summing up over k
      for(p = 0; p < k; p++)
      {
        a_0p_1p = _mm_loadu_pd(A+r+p*lda);
        b_p0 = _mm_load1_pd(B+p+c*lda);
        b_p1 = _mm_load1_pd(B+p+(c+1)*lda);

        c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a_0p_1p, b_p0));
        c_01_11 = _mm_add_pd(c_01_11, _mm_mul_pd(a_0p_1p, b_p1));
      }

      _mm_storeu_pd(C+r+c*lda, c_00_10);
      _mm_storeu_pd(C+r+(c+1)*lda, c_01_11);
    }

    //Case of last column, keep using SSE to update 2*1 C matrix
    if (n % 2) 
    {

      c_00_10 = _mm_loadu_pd(C+r+(n-1)*lda);

      for(p = 0; p < k; p++)
      {
        a_0p_1p = _mm_loadu_pd(A+r+p*lda);
        b_p0 = _mm_load1_pd(B+p+c*lda);

        c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a_0p_1p, b_p0));
      }

      _mm_storeu_pd(C+r+(n-1)*lda, c_00_10);
    }
  }

//Case of last row, using basic itero way with unroll the loop by 2
  if (m % 2) 
  {
    double c_0, c_1, a, b_0, b_1;
    
    for(c=0; c<n/2*2; c+=2)
    {
      c_0 = C[m-1+c*lda];
      c_1 = C[m-1+(c+1)*lda];
      for(p=0; p<k; p++)
      {
        a = A[m-1+p*lda];
        b_0 = B[p+c*lda];
        b_1 = B[p+(c+1)*lda];
        c_0 += a * b_0;
        c_1 += a * b_1;
      }
      C[m-1+c*lda] = c_0;
      C[m-1+(c+1)*lda] = c_1;
    }

//Dealing with the one last element
    if (n % 2)
    {
      
      c_0 = C[m-1+(n-1)*lda];
      for(p=0; p<k; p++)
      {
        a = A[m-1+p*lda];
        b_0 = B[p+(n-1)*lda];
        c_0 += a * b_0;
      }
      C[m-1+(n-1)*lda] = c_0;
    }
  }

}

void do_block(int lda, int m, int n, int k, double *A, double *B, double *C, int need_to_packB)
{
  int i, j;
  //Use local variable to pack A and B into smaller block
  //Be sure to assign enough B_buf_size, otherwise, lead to error
  double packedA[m*k], packedB[BLOCK_SIZE*B_buf_size]; 

  for (j=0; j<n/4*4; j+=4) /* n/4*4 operation ensure j stop before hitting fringe  */
  {  // Loop over the columns of C, unrolled by 4
    if (need_to_packB) copy4toB(k, B+j*lda, lda, &packedB[j*k]);
    
    for (i=0; i<m/8*8; i+=8) /* n/8*8 operation ensure i stop before hitting fringe  */
    { // Loop over the rows of C, unrolled by 8 
      if (j == 0) 
      {
          copy8toA(k, A+i, lda, &packedA[i*k]);
      }
      update8X4(k, &packedA[i*k], 8, &packedB[j*k], k, C+i+j*lda, lda);
    }
  }
  
  //Dealing the case that row dimension is odd number
  if (m%8!=0 && n%4 == 0)
  {
    // starting from the row fringe
    int row_index = m/8*8;
    dgemm_fringe(lda, m - row_index, n, k, A+row_index, B, C+row_index);
  }
  //Case of both dimension are odd number
  else if (m%8!=0 && n%4 != 0)
  {
    int row_index = m/8*8, col_index = n/4*4;

    //First addressing column fringe with 8*3 strategy
    if (n%4 == 3)
    {
      double packedB3[BLOCK_SIZE*3];
      copy3toB(k, B+col_index*lda, lda, &packedB3[0]);
      for (i=0; i<m/8*8; i+=8)
      {        /* Loop over the rows of C, unrolled by 8 */
        if (j == 0) copy8toA(k, A+i, lda, &packedA[i*k]);
        update8X3(k, &packedA[i*k], 8, &packedB3[0], k, C+i+j*lda, lda);
      }
    }
    //Else general case
    else dgemm_fringe(lda, row_index, n-col_index, k, A, B+lda*col_index, C+lda*col_index);
    
    // Case of bottom row block and a tailing column block
    dgemm_fringe(lda, m-row_index, n, k, A+row_index, B, C+row_index);
    
  }
  // Here we rule out the case that m%8 == 0 and n%4 != 0 cause that's not possible
  else if (m%8 == 0 && n%4!=0)
  {
    int col_index = n/4*4;

    dgemm_fringe(lda, m, n-col_index, k, A, B+lda*col_index, C+lda*col_index);
  }

}

void square_dgemm(int n, double* A, double* B, double* C)
{
  int i, j, a_col, c_row;

    //A column index
    for (j=0; j<n; j+=BLOCK_SIZE) 
    {
      a_col = min(n-j, BLOCK_SIZE );
      
      //C row index
      for (i=0; i<n; i+=BLOCK_SIZE)  
      {
        c_row = min(n-i, BLOCK_SIZE);
        do_block(n, c_row, n, a_col, A+i+j*n, B+j, C+i, i==0); 
        //Using flag i to load B within do_block           
        //at change of C row
        //to eliminate repeatly loading B, reduce overhead
      }
    }
}

