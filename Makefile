# On Bridges we will check versus your performance versus Intel MKL library's BLAS. 

#CC = gcc
CC= icc 
#OPT = -g
OPT = -O3 -g
#OPT = -g  
#CFLAGS = -Wall -std=gnu99 -ftree-vectorize -funroll-loops -mfma -mavx2 -fstrict-aliasing -ffast-math $(OPT)
#CFLAGS = -Wall -std=gnu99 -mavx2 -mfma -ftree-vectorize -funroll-loops -fstrict-aliasing -ffast-math $(OPT)
CFLAGS = -Wall -std=gnu99 -march=core-avx2 -vec -ftree-vectorize -funroll-loops -fstrict-aliasing $(OPT)
MKLROOT = /opt/intel/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDLIBS = -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lgomp -lpthread -lm -ldl 

targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-blocked-sse test-blocked benchmark-blocked-sse test-blocked-sse
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o matrixtester.o dgemm-blocked-sse.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-sse : benchmark.o dgemm-blocked-sse.o
	$(CC) -o $@ $^ $(LDLIBS)
test-blocked: matrixtester.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
test-blocked-sse: matrixtester.o dgemm-blocked-sse.o
	$(CC) -o $@ $^ $(LDLIBS)
	
%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout

