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

// changes that improved performance
// 1.) transverse one of the matrixies --> convert from row order to column order.
// 2.) Added opm parallel for to the inner loop
// 3.) Set the cach size to equal sqrt(L2_SIZE_BYTES/2/sizeof(double))
// 4.) gcc compiler flags set to -mfma -mavx2 -funroll-loops -ftree-vectorize -ffast-math -fopenmp -O3
// 5.) inlined functions 
// 6.) switched the order of the loops.
// 7.) moved declariations out of the loops so there are performed far less. 
// 8.) reduced the number of multiplys 


#include <immintrin.h>
#include <omp.h>
//#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
// form intel https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf page 299
// tilewidth = L2SizeInBytes/2/TileHeight/Sizeof(element)
//#define L2_SIZE_BYTES 1048576
//#define L2_SIZE_BYTES 6291456
//#define L2_SIZE_BYTES 12582912
//#define L2_SIZE_BYTES 25165824
//#define L2_SIZE_BYTES 262144
//int BLOCK_SIZE = 8192;
#define BLOCK_SIZE 2
#define L2_SIZE_BYTES 8192
#define L2_SQURED L2_SIZE_BYTES/2/sizeof(double)
#endif

 // #define _MM_TRANSPOSE4_PD(row0,row1,row2,row3)                                 //\
 //  {                                                                //\
 //      //__m256d tmp3, tmp2, tmp1, tmp0;                              \
 //                                                                   \
 //    __m256d tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);                    \
 //    __m256d tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);                \
 //    __m256d tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);                    \
 //    __m256d tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);                \
 //                                                                 \
 //    (row0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   \
 //    (row1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   \
 //    (row2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   \
 //    (row3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);   \
 //  }

#define min(a,b) (((a)<(b))?(a):(b))


// void transpose_block_AVX4x4(const double *In, double *Out, const int nN, const int mM, const int ldaSquare,const int block_size);
// void transpose4x4_AVX(const double *InA, double *OutB, const int Dimlda);
// //void transpose_block_AVX4x4(const double *src, double *dst, const int nN, const int blocksize);

// inline void transpose4x4_AVX(const double *InA, double *OutB, const int Dimlda)
// {
//   //https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
//   //__m256d a,b,c,d;
//    // get rid of stupid & and unnessacery mul
//   // have an issue with memory alignme__restrict__ nt!!!!
//   __m256d row0 = _mm256_loadu_pd(InA);
//   __m256d row1 = _mm256_loadu_pd(InA + Dimlda );
//   __m256d row2 = _mm256_loadu_pd(InA + (Dimlda << 1) );
//   __m256d row3 = _mm256_loadu_pd(InA + 3*Dimlda );

//   __m256d tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);                    \
//   __m256d tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);                \
//   __m256d tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);                    \
//   __m256d tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);                \
//                                                                \
//   (row0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   \
//   (row1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   \
//   (row2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   \
//   (row3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);   \

//   _mm256_storeu_pd(OutB, row0);
//   _mm256_storeu_pd(OutB + Dimlda, row1);
//   _mm256_storeu_pd(OutB + (Dimlda << 1), row2);
//   _mm256_storeu_pd(OutB + 3*Dimlda, row3);

// }


// inline void transpose_block_AVX4x4(const double *In, double *Out, const int nN, const int mM, const int ldaSquare,const int block_size)
// {
//     //#pragma omp parallel for
//     for(int i=0; i<nN; i+=block_size)
//     {
//         for(int j=0; j<mM; j+=block_size)
//         {
//             int max_i2 = i+block_size < nN ? i + block_size : nN;
//             int max_j2 = j+block_size < mM ? j + block_size : mM;
//             for(int i2=i; i2<max_i2; i2+=4)
//             {
//                 for(int j2=j; j2<max_j2; j2+=4)
//                 {
//                     transpose4x4_AVX(&In[i2*ldaSquare +j2], &Out[j2*ldaSquare + i2], ldaSquare);
//                 }
//             }
//         }
//     }
// }


// inline void transpose(const double *A, double * B, const int lda)
// {
//     //#pragma unroll(8)
//     int i, j;
//     //#pragma omp parallel for
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

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
inline static void do_block (const int lda, const int M, const int N, const int K, const double* A, const double* B, double* C)
{

  //#pragma omp parallel for
  //#pragma unroll(8)
  /* For each row i of A */
    for (int j = 0; j < N; ++j) 
    {
    //__m128d a,b,c,d;
    int C_index_mul = j*lda;
    double cij;
    /* For each column j of B */ 

        for (int i = 0; i < M; ++i) 
        {
        /* Compute C(i,j) */
        //c = _mm_loadu_pd(C + i + j*lda);

        
          cij = C[i + C_index_mul];
          for (int k = 0; k < K; ++k)
          {

             //FMA instruction
            //https://software.intel.com/en-us/cpp-compiler-18.0-developer-guide-and-reference-mm-fmadd-pd-mm256-fmadd-pd
             // a = _mm_loadu_pd(A+k+i*lda);
             // b = _mm_loadu_pd(B+k+j*lda);
             // d = _mm_fmadd_pd(a, b, c);
        
            // once again A is jumping across the cache. 
           // transposed
            //cij += A[k+i*lda] * B[k+C_index_mul];

            // non transposed
            cij += A[i+k*lda] * B[k+j*lda];

          }
        
       
        //C[i+j*lda] = (double)d[0];

         C[i + C_index_mul] = cij;

      }
  }
}

// static void do_block_L1(const int lda, const int M_L2, const int N_L2, const int K_L2, const double* A, const double* B, double* C)
// {
//   for (int i = 0; i < M_L2; i += BLOCK_SIZE)
//   {
//           // redeclare everything as less as possible
//           int M = min (BLOCK_SIZE, lda-i);
//           int N = 0;
//           int K = 0;
//           /* For each block-column of B */

//           for (int j = 0; j < N_L2; j += BLOCK_SIZE)
//           {
//             /* Accumulate block dgemms into block of C */
//             N = min (BLOCK_SIZE, lda-j);

//             //double cij = C[i+j*lda];

//             for (int k = 0; k < K_L2; k += BLOCK_SIZE)
//             {
//                //Correct block dimensions if block "goes off edge of" the matrix 
//               K = min (BLOCK_SIZE, lda-k);
//         /* Perform individual block dgemm */

//               // A addressed by row so A is incremented by lda plus offset. A is jumping across the cache. 
//               //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
//               //cij += A[i+k*lda] * B[k+j*lda];
//               do_block(lda, M, N, K, buf + i + k*lda, B + k + j*lda, C + i + j*lda);
//               //do_block(lda, M, N, K, buf + k + i*lda, B + k + j*lda, C + i + j*lda);
//             }
//             //C[i+j*lda] = cij;

//           }
//   } // end for (int i = 0; i < lda; i += BLOCK_SIZE)  // END of L!

// }



//128K cache L1 
// 2000 doubles 
// lda is the dimesion of the Matrix.
//Block size is the chunk of memory in doubles 

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, const double* A, const double* B, double* C)
{

  //int BLOCK_SIZE = sqrt(L2_SQURED);
  //int BLOCK_SIZE = L2_SQURED;
  //https://gcc.gnu.org/onlinedocs/gcc/Restricted-Pointers.html  // __restrict__ 

    double B_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
    double A_Block[BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE];
    int M = 0;
    int K = 0;
    int N = 0;


    double x = B_Block[0];

    for (int j_Block = 0; j_Block < lda; j_Block += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      N = min (BLOCK_SIZE, lda-j_Block);
      int j_mul = j_Block*lda;

      for (int k_Block = 0; k_Block < lda; k_Block += BLOCK_SIZE)
      {
        //Correct block dimensions if block "goes off edge of" the matrix 
        K = min (BLOCK_SIZE, lda-k_Block);
        //double * B_Block = (double*)calloc(BLOCK_SIZE*BLOCK_SIZE*N*M, sizeof(double));
        // double * B_Block = (double*)aligned_alloc(8, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
        // ////load a block of Matrix B into memory. Loading by columns 
        for (int J_load = 0; J_load < N; ++J_load)
        {
          for (int K_load = 0; K_load < K; ++K_load)
          {
            //B_Block[K_load + J_load*BLOCK_SIZE] = B[k_Block + K_load + J_load*lda];
            B_Block[K_load + J_load * BLOCK_SIZE] = B[K_load + k_Block + (J_load + j_Block) *lda];
              //B_Block[Index] =  B[k_Block + Index];
          } 
        } 

        printf("\n");
        printf("B_BLOCK\n");
        printf("\n");
        printMatrixLinear(BLOCK_SIZE, B_Block);
        printf("\n");
        printf("\n");
        printMatrix(BLOCK_SIZE,B_Block);
        printf("\n");

        for (int i_Block = 0; i_Block < lda; i_Block += BLOCK_SIZE)
        {
                // redeclare everything as less as possible
          M = min (BLOCK_SIZE, lda-i_Block);
          //double A_Block[(K+1) *(M+1)];
          //double * A_Block = (double*)calloc((K+1)*(M*lda), sizeof(double));
          //double * A_Block = (double*)aligned_alloc(8, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
          ////load a block of Matrix A into memory. Loading by columns into ROWS!
          // The loaded Block is TRANSVERSE OF MATRIX BLOCK A 
          for(int I_load = 0; I_load < M; ++I_load)
          {
            for (int K_load = 0; K_load < K; ++K_load)
            {
                //A_Block[K_load + I_load * BLOCK_SIZE] =  A[K_load + k_Block + (I_load + j_Block) *lda];
                A_Block[K_load + I_load * BLOCK_SIZE] =  A[i_Block + I_load + (K_load + k_Block) *lda];
            } 
          } // for(int I_load = 0; I_load < M; ++I_load)

            printf("\n");
            printf("A_BLOCK\n");
            printf("\n");
            printMatrixLinear(BLOCK_SIZE, A_Block);
            printf("\n");
            printf("\n");
            printMatrix(BLOCK_SIZE,A_Block);
            printf("\n");

          /////////////////// START INNER LOOP !!!!!!!!
          /* Perform individual block dgemm */

                // A addressed by row so A is incremented by lda plus offset. A is jumping across the cache. 
                
                //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
                //do_block(lda, M, N, K, A + i_Block + k_Block*lda, B + k_Block + j_Block*lda, C + i_Block + j_Block*lda);
                //do_block(lda, M, N, K, A_Block + i_Block + k_Block*lda, B_Block + j_Block*BLOCK_SIZE, C + i_Block + j_Block*lda);
                //cij += A[i+k*lda] * B[k+j*lda];
                //do_block(lda, M, N, K, A_Block + k_Block + i_Block*lda, B_Block + k_Block + j_mul, C + i_Block + j_mul);

                // let's try to vectorize by 4 at first!!!!
                for(int j = 0; j<N; j+=2)
                {
                  //int B_index_mul = j*lda;
                  //int C_index_mul = (j+j_Block) *lda;

                  for(int i=0; i<M; i+=2)
                  {
                      // load into local variables. 
                      double C_Partial_0 = 0;
                      double C_Partial_1 = 0;
                      double C_Partial_2 = 0;
                      double C_Partial_3 = 0;
                      // double C_Partial_4 = 0;
                      // double C_Partial_5 = 0;
                      // double C_Partial_6 = 0;
                      // double C_Partial_7 = 0;

                      double Aelement = 0;
                      double Aelement1 = 0;
                      double Aelement2 = 0;
                      double Aelement3 = 0;

                      // perform the Partial sums C = C + A*B
                      for(int k = 0; k < K; ++k)
                      {   //// THIS CAN BE REPLACED WITH FMA instructions!!!!
                        /// cij += A[k+i*lda] * B[k+C_index_mul]; 

                        Aelement = A_Block[k + i*BLOCK_SIZE];
                        Aelement1 = A_Block[k + 1 + i*BLOCK_SIZE];  
                        Aelement2 = A_Block[k + (i+1)*BLOCK_SIZE];
                        Aelement3 = A_Block[k+1 + (i+1)*BLOCK_SIZE];  

                        // Row 0 Col 0,1  
                        C_Partial_0 = Aelement * B_Block[ k + j*BLOCK_SIZE] + Aelement1 * B_Block[ k +1 + j*BLOCK_SIZE];
                        C_Partial_1 = Aelement * B_Block[ k + (j+1)*BLOCK_SIZE] + Aelement1 * B_Block[ k +1 + (j+1)*BLOCK_SIZE];

                        //Row 1 Col 0,1 
                        C_Partial_2 = Aelement2 * B_Block[ k + j*BLOCK_SIZE] + Aelement3 * B_Block[ k + 1 + j*BLOCK_SIZE];
                        C_Partial_3 = Aelement2 * B_Block[ k + (j+1)*BLOCK_SIZE] + Aelement3 * B_Block[ k + 1 + (j+1)*BLOCK_SIZE];


                        // C_Partial_2 += Aelement * B_Block[ k + j +1 *BLOCK_SIZE ];
                        // C_Partial_3 += Aelement1 * B_Block[ k + j +1 * BLOCK_SIZE ];

                        // C_Partial_4 += Aelement * B_Block[ k + j + 2 * BLOCK_SIZE ];
                        // C_Partial_5 += Aelement1 * B_Block[ k + j +2 * BLOCK_SIZE ];

                        // C_Partial_6 += Aelement * B_Block[ k + j+ 3 * BLOCK_SIZE ];
                        // C_Partial_7 += Aelement1 * B_Block[ k + j +3 * BLOCK_SIZE ];

                        printf("A0: %f\n", Aelement);
                        printf("A1: %f\n", Aelement1);
                        printf("B: %f\n", B_Block[k+j*BLOCK_SIZE]);
                        printf("I: %d J: %d K: %d\n", i,j,k);
                        printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_0);
                        printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_1);
                        printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_2);
                        printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_3);
                        // printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_4);
                        // printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_5);
                        // printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_6);
                        // printf(" I: %d J: %d PV: %f\n", i,j, C_Partial_7);

                      } 
                      // Sum elements in block 
                      // C is column major ordered. 
                      // C[i_Block + i + (j+j_Block)*lda ] +=  C_Partial_0;
                      // C[i_Block + i + 1 + (j+j_Block) *lda] += C_Partial_1;
                      
                      // C[i_Block + i + (j+j_Block + 1) *lda] +=  C_Partial_2;
                      // C[i_Block + i +1 + (j+j_Block + 1) *lda] +=  C_Partial_3;

                      // C[i_Block + i + (j+j_Block + 2) *lda] +=  C_Partial_4;
                      // C[i_Block + i + 1 + (j+j_Block + 2) *lda] +=  C_Partial_5;

                      // C[i_Block + i + (j+j_Block + 3) *lda] +=  C_Partial_6 ;
                      // C[i_Block + i + 1 + (j+j_Block + 3) *lda] +=  C_Partial_7;

                  }

                } /// end of the inner loop 

             //free(A_Block);


            //double C_Partial_1_1 = A_Block[k + i*BLOCK_SIZE] * B_Block[k + j*BLOCK_SIZE];
            //double C_Partial_1_2 = A_Block[k + i*BLOCK_SIZE] * B_Block[k+1 + j*BLOCK_SIZE];

            //double C_Partial_2_1 = A_Block[k+1 + i*BLOCK_SIZE] * B_Block[k + j+*BLOCK_SIZE];
            //double C_Partial_2_2 = A_Block[k+1 + i*BLOCK_SIZE] * B_Block[k+1 + j*BLOCK_SIZE];

                          // for (int j = 0; j < N; ++j) 
                          // {
                          // //__m128d a,b,c,d;
                          // int C_index_mul = j*lda;
                          
                          // /* For each column j of B */ 

                          //     for (int i = 0; i < M; ++i) 
                          //     {
                          //     /* Compute C(i,j) */
                          //     //c = _mm_loadu_pd(C + i + j*lda);

                          //       //cij = C[i + C_index_mul];
                          //       double cij = 0;
                          //       for (int k = 0; k < 2; ++k)
                          //       {
                          //          //FMA instruction
                          //         //https://software.intel.com/en-us/cpp-compiler-18.0-developer-guide-and-reference-mm-fmadd-pd-mm256-fmadd-pd
                          //          // a = _mm_loadu_pd(A+k+i*lda);
                          //          // b = _mm_loadu_pd(B+k+j*lda);
                          //          // d = _mm_fmadd_pd(a, b, c);
                              
                          //         // once again A is jumping across the cache. 
                          //        // transposed
                          //         printf(" I: %d J: %d K: %d\n", i,j,k);

                          //         cij += A_Block[k+i*BLOCK_SIZE] * B_Block[k+j*BLOCK_SIZE];

                          //         printf("A: %f\n", A_Block[k+i*BLOCK_SIZE]);
                          //         printf("B: %f\n", B_Block[k+i*BLOCK_SIZE]);
                          //         printf("C: %f\n", cij);
                          //         //cij = A[row e1]* B[col e1] + A[row e2]c* B[col e2]

                                  

                          //         // non transposed
                          //         //cij += A[i+k*lda] * B[k+j*lda];
                          //       }
                          //       printf(" I: %d J: %d V: %f\n", i,j, cij);
                              
                          //     //C[i+j*lda] = (double)d[0];

                          //      C[i + C_index_mul] = cij;

                          //     }
                          //   }



            }   //  for (int i = 0; i < lda; i += BLOCK_SIZE)
              //free(B_Block);

        } // end for (int k = 0; k < lda; k += BLOCK_SIZE)
    }  // end for (int j = 0; j < lda; j += BLOCK_SIZE)

  //free(buf);
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



//References

//http://wgropp.cs.illinois.edu/courses/cs598-s16/lectures/lecture11.pdf

//TIPS 

// use the intel complier

// cache

// Huge pages

// gcc fast math
// https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do

// "#pragma unroll(8)"
// https://en.wikipedia.org/wiki/Loop_nest_optimization

//
//https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX

//FMA instruction
//https://en.wikipedia.org/wiki/FMA_instruction_set
//https://software.intel.com/en-us/cpp-compiler-18.0-developer-guide-and-reference-mm-fmadd-pd-mm256-fmadd-pd

