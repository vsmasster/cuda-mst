#include "cuda.h"
#include "cuda_alg.h"

__global__ 
static void block_scan(int* a, int* b, int n, int* sb)
{
  __shared__ int t[1024];
  
  int i = threadIdx.x;
  int gi = blockIdx.x*1024 + threadIdx.x;
  
  if(gi < n) {
    t[i] = a[gi];
  } else {
    t[i] = 0;
  }
    
  for(int p = 1; p <= 512; p *= 2) {
    __syncthreads();

    int x = (i >= p ? t[i-p] : 0);
    
    __syncthreads();

    t[i] += x;
  }
  
  if(gi < n) {
    b[gi] = t[i];
  }
  
  if(i == 1023) {
    sb[blockIdx.x] = t[i];
  }
}


__global__
static void add(int* a, int* sb, int n)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < n) {
    a[i] += sb[blockIdx.x];
  }
}


void scan(int* a, int* b, int n)
{
  int nb = (n-1)/1024+1;
  int* h_sb;
  int* d_sb;
  
  h_sb = new int[nb];
  cudaErrChk(cudaMalloc(&d_sb, nb*sizeof(int)));
  
  block_scan<<<CB(n)>>>(a, b, n, d_sb);
  cudaErrChk(cudaMemcpy(h_sb, d_sb, nb*sizeof(int), cudaMemcpyDeviceToHost));
  
  for(int i = 0, t = 0; i < nb; ++i) {
    int x = h_sb[i]; 
    h_sb[i] = t;
    t += x;
  } 
  
  cudaErrChk(cudaMemcpy(d_sb, h_sb, nb*sizeof(int), cudaMemcpyHostToDevice));
  add<<<CB(n)>>>(b, d_sb, n);

  delete[] h_sb;
  cudaErrChk(cudaFree(d_sb));
}

