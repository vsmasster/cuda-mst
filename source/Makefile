NVCCFLAGS = --std=c++11 -Wno-deprecated-gpu-targets
NVCCPATH = /usr/local/cuda/bin/nvcc

CXX = g++-5
CXXFLAGS = -std=c++11 -Wall -O2

CUDA_ALG_OBJS = scan.o segreduce.o sort.o
MST_OBJS = cpu_mst.o vi_mst.o dpk_mst.o

main: main.o $(MST_OBJS) $(CUDA_ALG_OBJS)
	$(NVCCPATH) $(NVCCFLAGS) -o main main.o $(MST_OBJS) $(CUDA_ALG_OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<	

%.o: %.cu
	$(NVCCPATH) $(NVCCFLAGS) -Xcompiler "-Wall -O2" -c $<	
	
clean:
	rm *.o
