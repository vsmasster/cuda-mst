#include "cuda.h"
#include "cuda_alg.h"

__global__ 
static void clear_res(int* a, int n)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < n) {
    a[i] = -1;
  }
}


__global__ 
static void set_flags(int* id, int n, int* f)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < n) {
    f[i] = (i == 0 || id[i] != id[i-1]) ? 2 : 0;
  }
}


__global__ 
static void block_reduce(int* ga, int* gf, int n)
{
  __shared__ int a[1024];
  __shared__ int r[1024];
  __shared__ int f[1024];
  
  int i = threadIdx.x;
  int gi = blockIdx.x*1024 + threadIdx.x;
  
  if(gi < n) {
    a[i] = ga[gi];
    f[i] = gf[gi];
  } else {  
    a[i] = 0;
    f[i] = (gi == n ? 2 : 1);
  }
  
  r[i] = gi;
    
  for(int p = 1; p <= 512; p *= 2) {
    __syncthreads();

    int j = i-p;
    
    bool fj;
    int aj, rj;
    
    if(j >= 0) {
      fj = f[j];
      aj = a[j];
      rj = r[j];
    } 

    __syncthreads();

    if(j >= 0 && !f[i]) {      
      if(a[i] > aj) {
        a[i] = aj;
        r[i] = rj;
      }
      f[i] = fj;
    }
  }
  
  if(i == 1023 || (f[i+1] & 2)) {
    gf[gi] = r[i];
  }
}

__global__ 
static void fill_res(int* id, int* a, int* p, int* res, int n)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  
  if(i < n) {
    int mid = id[i];
    
    if(i != n-1 && mid == id[i+1])
      return;
    
    int r = p[i];
    int bval = a[r];
    
    i = (i & ~1023) - 1;
    while(i >= 0 && id[i] == mid) {
      int j = p[i];
      int x = a[j];
      if(bval > x) {
        bval = x;
        r = j;
      }
      
      i -= 1024;
    }
    
    res[mid] = r;
  }
}


void segreduce(int* id, int* a, int n, int m, int* res)
{
  int* f;
  cudaErrChk(cudaMalloc(&f, m*4));
  
  set_flags<<<CB(m)>>>(id, m, f);
  
  block_reduce<<<CB(m)>>>(a, f, m);
  
  clear_res<<<CB(n)>>>(res, n);
  fill_res<<<CB(m)>>>(id, a, f, res, m);

  cudaErrChk(cudaFree(f));
}
