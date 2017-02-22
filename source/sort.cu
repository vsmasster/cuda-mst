#include "cuda.h"
#include "cuda_alg.h"

#include <algorithm>
using namespace std;

__global__ 
static void block_scan(uint64_t* a, int* s, uint64_t p, int n, int* sb)
{
  __shared__ int t[1024];
  
  int i = threadIdx.x;
  int gi = blockIdx.x*1024 + threadIdx.x;
  
  if(gi < n)
    t[i] = !(bool)(a[gi] & p);
  else
    t[i] = 0;
    
  for(int p = 1; p <= 512; p *= 2) {
    __syncthreads();

    int x = (i >= p ? t[i-p] : 0);
    
    __syncthreads();

    t[i] += x;
  }
  
  if(gi < n) {
    s[gi] = t[i];
  }
  
  if(i == 1023)
    sb[blockIdx.x] = t[i];
}


__global__
static void perm(uint64_t* a, uint64_t* b, int* s, int* sb, int n, uint64_t p, int nz)
{
  __shared__ int temp; 
  int i = blockIdx.x*1024 + threadIdx.x;
  
  if(threadIdx.x == 0) {
    temp = sb[blockIdx.x];
  }
  
  __syncthreads();

  if(i < n) {
    if(!(a[i] & p))
      b[ s[i]+temp-1 ] = a[i];
    else
      b[ nz+i-(s[i]+temp) ] = a[i];
  }
}

  

void sort(uint64_t* in, int n)
{
  int nb = CEIL(n, 1024);
  
  uint64_t* a = in;
  uint64_t* b;
  int* s;
  int* d_sb;
  int* h_sb;
  
  cudaErrChk(cudaMalloc(&b, 8*n));
  cudaErrChk(cudaMalloc(&s, 4*n));
  cudaErrChk(cudaMalloc(&d_sb, 4*nb));
  h_sb = new int[nb];
  
  for(int k = 0; k < 64; ++k) {
    uint64_t p = uint64_t(1) << k;
    
    block_scan<<<CB(n)>>>(a, s, p, n, d_sb);
    cudaErrChk(cudaMemcpy(h_sb, d_sb, nb*sizeof(int), cudaMemcpyDeviceToHost));
  
    int nz = 0;
    for(int i = 0; i < nb; ++i) {
      int x = h_sb[i]; 
      h_sb[i] = nz;
      nz += x;
    } 
    
    cudaErrChk(cudaMemcpy(d_sb, h_sb, nb*sizeof(int), cudaMemcpyHostToDevice));
    
    perm<<<CB(n)>>>(a, b, s, d_sb, n, p, nz);
    
    swap(a, b);
  }
  
  cudaErrChk(cudaFree(b));
  cudaErrChk(cudaFree(s));
  cudaErrChk(cudaFree(d_sb));
  delete[] h_sb;
}
