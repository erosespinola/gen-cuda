CUDADIR = /usr/local/cuda
INC = -I$(CUDADIR)/include
CFLAGS = -O3 -arch=compute_20 -code=sm_20
NVCCFLAGS = -O3 $(CUDAINC) -arch=compute_20 -code=sm_20

CC = $(CUDADIR)/bin/nvcc

TARGETS = gpu-ga

all: $(TARGETS)

gpu-ga: gpu-ga.o
	$(CC) -o $@ $(NVCCLIBS) gpu-ga.o

gpu-ga.o: gpu-ga.cu
	$(CC) -c $(NVCCFLAGS) gpu-ga.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
