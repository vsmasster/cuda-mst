#include "cuda.h"
#include "cuda_alg.h"
#include "mst.h"


#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <ctime>
#include <algorithm>
#include <memory>
#include <vector>

__global__
static void d_copy(int* src, int* dst, int n)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < n) {
    dst[i] = src[i];
  }
}

__global__ 
static void d_fill_ident(int* a, int n)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < n) {
    a[i] = i;
  }
}

__global__
static void d_mark_nonloop_edges(int* ei, int* eu, int* ev, int* vr, int* res, int m)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < m) {
    int j = ei[i];
    int u = vr[eu[j]];
    int v = vr[ev[j]];
    if(u != v) {
      res[i] = 1;
      eu[j] = u;
      ev[j] = v;
    } else {
      res[i] = 0;
    }
  }
}

__global__
static void d_remove_loop_edges(int* ei, int* t, int* res, int m)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < m) {
    int pos = t[i];
    if((i == 0 && pos) || (i && pos != t[i-1])) {
      res[pos-1] = ei[i];
    }
  }
}

__global__
static void d_double_edges(int* ei, int* eu, int* ev, int* n_ei, int* n_eu, int m)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < m) {
    int j = ei[i];
    n_ei[i] = j;
    n_ei[i+m] = -j-1;
    n_eu[i] = eu[j];
    n_eu[i+m] = ev[j];
  }
}

__global__
static void d_fill_vi_fields(int* ei, int* eu, int* ev, int* ew, int* vi_ev, int* vi_ew, int m)
{
  int i = blockIdx.x*1024 + threadIdx.x;
  if(i < m) {
    int j = ei[i];
    if(j < 0) {
      j = -j-1;
      vi_ev[i] = eu[j];
    } else {
      vi_ev[i] = ev[j];
    }
    vi_ew[i] = ew[j];
  }
}

uint64_t dpk_mst(int n, int m, int* eu, int* ev, int* ew)
{
  int rn = n;
  int qn = 2*n;
  int* vr;
  int* ei;
  int* t;
  int* vi_eu;
  int* vi_ev;
  int* vi_ew;
  
  cudaErrChk(cudaMalloc(&vr, n*4));
  cudaErrChk(cudaMalloc(&ei, m*4));
  
  cudaErrChk(cudaMalloc(&t, std::max(qn, m)*4));
  cudaErrChk(cudaMalloc(&vi_eu, qn*4));
  cudaErrChk(cudaMalloc(&vi_ev, qn*4));
  cudaErrChk(cudaMalloc(&vi_ew, qn*4));
  
  d_fill_ident<<<CB(n)>>>(vr, n);
  d_fill_ident<<<CB(m)>>>(ei, m);
  
  d_copy<<<CB(m)>>>(ew, t, m);
  {
    thrust::device_ptr<int> t_ei(ei);
    thrust::device_ptr<int> t_ew(t);
    thrust::sort_by_key(t_ew, t_ew+m, t_ei);
  }
  
  uint64_t ret = 0;
  
  for(int b = 0, s = n; b < m && n > 1; s *= 1) {
    int e = std::min(m, b+s);
    int tm = e-b;
    
    d_mark_nonloop_edges<<<CB(tm)>>>(ei+b, eu, ev, vr, t, tm);
    scan(t, t, tm);
    
    if(tm == 0) {
      b += s;
      continue;
    }
    
    int new_tm;
    cudaErrChk(cudaMemcpy(&new_tm, t+tm-1, 4, cudaMemcpyDeviceToHost));

    if(new_tm > 0) {
      d_remove_loop_edges<<<CB(tm)>>>(ei+b, t, vi_ew, tm);
      tm = new_tm;
    
      d_double_edges<<<CB(tm)>>>(vi_ew, eu, ev, t, vi_eu, tm);
      tm *= 2;
    
      {
        thrust::device_ptr<int> t_ei(t);
        thrust::device_ptr<int> t_eu(vi_eu);
        thrust::sort_by_key(t_eu, t_eu+tm, t_ei);
      }
    
      d_fill_vi_fields<<<CB(tm)>>>(t, eu, ev, ew, vi_ev, vi_ew, tm);
        
    
      ret += vi_mst(n, tm, vi_eu, vi_ev, vi_ew, vr, rn);
    }
    
    b += s;
  }
  
  
  cudaErrChk(cudaFree(vr));
  cudaErrChk(cudaFree(ei));
  
  cudaErrChk(cudaFree(t));
  cudaErrChk(cudaFree(vi_eu));
  cudaErrChk(cudaFree(vi_ev));
  cudaErrChk(cudaFree(vi_ew));
  
  return ret;
}

uint64_t dpk_mst(int n, const std::vector<edge>& edges, int& time_ms)
{
  int m = edges.size();
  
  int* h_eu = new int[m];
  int* h_ev = new int[m];
  int* h_ew = new int[m];
  
  for(int i = 0; i < m; ++i) {
    h_eu[i] = edges[i].u;
    h_ev[i] = edges[i].v;
    h_ew[i] = edges[i].w;
  }
  
  int* d_eu; 
  int* d_ev;
  int* d_ew;
  
  cudaErrChk(cudaMalloc(&d_eu, m*4));
  cudaErrChk(cudaMalloc(&d_ev, m*4));
  cudaErrChk(cudaMalloc(&d_ew, m*4));

  cudaErrChk(cudaMemcpy(d_eu, h_eu, m*4, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(d_ev, h_ev, m*4, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(d_ew, h_ew, m*4, cudaMemcpyHostToDevice));
  
  delete[] h_eu;
  delete[] h_ev;
  delete[] h_ew;
  
  auto start = std::clock();
  uint64_t ret = dpk_mst(n, m, d_eu, d_ev, d_ew);
  time_ms = (int)(1000.0 * (clock() - start) / CLOCKS_PER_SEC);
  
  cudaErrChk(cudaFree(d_eu));
  cudaErrChk(cudaFree(d_ev));
  cudaErrChk(cudaFree(d_ew));

  return ret;
}


