CUDADIR = /usr/local/cuda
INC = -I$(CUDADIR)/include
CFLAGS = -O3 -arch=compute_20 -code=sm_20
NVCCFLAGS = -O3 $(CUDAINC) -arch=compute_20 -code=sm_20

GA_INC_DIR= .
GA_LIB_DIR= ./ga

INC_DIRS= -I$(GA_INC_DIR)
LIB_DIRS= -L$(GA_LIB_DIR)

CC = $(CUDADIR)/bin/nvcc

TARGETS = gpu-ga

all: $(TARGETS)

# serial: serial.o
# 	g++ serial.o -o serial $(LIB_DIRS) serial.cpp -lga -lm $(CXX_LIBS)

gpu-ga: gpu-ga.o
	$(CC) -o $@ $(NVCCLIBS) gpu-ga.o

gpu-ga.o: gpu-ga.cu
	$(CC) -c $(NVCCFLAGS) gpu-ga.cu

# serial.o: serial.cpp
# 	g++ serial.o -o serial $(LIB_DIRS) serial.cpp -lga -lm $(CXX_LIBS)

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
