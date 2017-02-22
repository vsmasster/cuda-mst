#include "cuda.h"
#include "cuda_alg.h"
#include "mst.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <ctime>
#include <algorithm>
#include <memory>
#include <vector>
using namespace std;


__global__ 
static void d_zerout(int* a, int n)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < n) {
      a[i] = 0;
   }
}


__global__
static void d_mark(int* in, int* out, int n)
{
   int i = blockIdx.x*1024 + threadIdx.x;
  
   if(i < n) {
      out[in[i]] = 1;
   }
}


__global__ 
static void d_set_parent(int* ei, int* ev, int n, int* res)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < n) {
      int j = ei[i];
      res[i] = (j != -1 ? ev[j] : i); 
   }
}


__global__
static void d_rem_cycles_and_sum(int* ei, int* par, int* ew, int n, int* sb)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   int w = 0;
  
   if(i < n) {
      int j = par[i];
      int k = par[j];
      
      if(i != k || i < j) {
         w = ew[ei[i]];
         ei[i] = j;
      } else {
         ei[i] = i;
      }
   } 
  
   __shared__ int t[1024];
   i = threadIdx.x;
   t[i] = w;

   for(int p = 512; p >= 1; p /= 2) {
      __syncthreads();
    
      if(i < p) {
         t[i] += t[i+p];
      }
   }
  
   if(i == 0) {
      sb[blockIdx.x] = t[0];
   }
} 


__global__
static void d_go_up(int* sa, int* sb, int n, bool* chg)
{
   __shared__ bool temp;
   int i = blockIdx.x*1024 + threadIdx.x;
  
   if(threadIdx.x == 0) {
      temp = false;
   }
  
   __syncthreads();
  
   if(i < n) {
      int j = sa[i];
      int k = sa[j];
      sb[i] = k;
      if(j != k) {
         temp = true;
      }
   }
  
   __syncthreads();
  
   if(threadIdx.x == 0 && temp) {
      *chg = true;
   }
}


__global__
static void d_norm_reprs(int* repr, int* t, int n)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < n) {
      repr[i] = t[repr[i]]-1;
   }
}


__global__ 
static void d_norm_edges(int* eu, int* repr, int* ev, int* ew, uint64_t* res, int m)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < m) {
      int u = repr[eu[i]];
      int v = repr[ev[i]];
      if(u != v) {
         res[i] = ((uint64_t)u << 40) | ((uint64_t)v << 16) | ew[i]; 
      } else {
         res[i] = 0;
      }
   }   
}


#define EU(e) ((int)(e>>40))
#define EV(e) ((int)(e>>16) & ((1<<24)-1))
#define EW(e) ((int)(e & ((1<<16)-1)))


__global__
static void d_mark_lightest_edges(uint64_t* e, int* f, int m)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < m) {
      uint64_t ec = e[i];
      if(ec) {
         uint64_t ep = e[i-1];
         int u1 = EU(ep);
         int v1 = EV(ep);
         int u2 = EU(ec);
         int v2 = EV(ec);
         f[i] = (u1 != u2 || v1 != v2);
      } else f[i] = 0;
   }
}

__global__
static void d_make_new_edges(uint64_t* e, int* s, int* eu, int* ev, int* ew, int m)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < m) {
      int j = s[i];
      if(j != (i ? s[i-1] : 0)) {
         eu[j-1] = EU(e[i]);
         ev[j-1] = EV(e[i]);
         ew[j-1] = EW(e[i]);
      } 
   }
}


__global__
static void d_update_vert_mapping(int* vm, int* trans, int n)
{
   int i = blockIdx.x*1024 + threadIdx.x;
   if(i < n) vm[i] = trans[vm[i]];
}


uint64_t rem_cycles_and_sum(int* ei, int* par, int* ew, int n)
{
   int nb = CEIL(n, 1024);
   int* d_sb;
   int* h_sb;
  
   cudaErrChk(cudaMalloc(&d_sb, 4*nb));
   h_sb = new int[nb];
  
   d_rem_cycles_and_sum<<<CB(n)>>>(ei, par, ew, n, d_sb);
   cudaErrChk(cudaMemcpy(h_sb, d_sb, nb*sizeof(int), cudaMemcpyDeviceToHost));
  
   uint64_t ret = 0;
   for(int i = 0; i < nb; ++i)
      ret += h_sb[i];
  
   cudaErrChk(cudaFree(d_sb));
   delete[] h_sb;
  
   return ret;
}

void find_reprs(int** ta, int** tb, int n)
{
   bool h_chg;
   bool* d_chg;
  
   cudaErrChk(cudaMalloc(&d_chg, 1));
  
   do {
      h_chg = false;
      cudaErrChk(cudaMemcpy(d_chg, &h_chg, 1, cudaMemcpyHostToDevice));
    
      d_go_up<<<CB(n)>>>(*ta, *tb, n, d_chg);
    
      cudaErrChk(cudaMemcpy(&h_chg, d_chg, 1, cudaMemcpyDeviceToHost));
      swap(*ta, *tb);
  
   } while(h_chg);
  
   cudaErrChk(cudaFree(d_chg));
}

// returns new n
static int norm_reprs(int* repr, int* t, int n)
{
   int nn;
  
   d_zerout<<<CB(n)>>>(t, n);
   d_mark<<<CB(n)>>>(repr, t, n);
  
   scan(t, t, n);
   cudaErrChk(cudaMemcpy(&nn, t+n-1, 4, cudaMemcpyDeviceToHost));
  
   d_norm_reprs<<<CB(n)>>>(repr, t, n);
   
   return nn;
}

uint64_t vi_mst(int& n, int m, int* eu, int* ev, int* ew, int* vm /*= NULL*/, int rn /*= 0*/)
{
   int* ta;
   int* tb;
   int* tc;
   uint64_t* te;
  
   cudaErrChk(cudaMalloc(&ta, 4*n));
   cudaErrChk(cudaMalloc(&tb, 4*n));
   cudaErrChk(cudaMalloc(&tc, 4*m));
   cudaErrChk(cudaMalloc(&te, 8*m));
  
   uint64_t ret = 0;
   while(m) {
      segreduce(eu, ew, n, m, ta);
  
      d_set_parent<<<CB(n)>>>(ta, ev, n, tb);
    
      ret += rem_cycles_and_sum(ta, tb, ew, n);
      find_reprs(&ta, &tb, n);
    
      n = norm_reprs(ta, tb, n);
      if(vm) d_update_vert_mapping<<<CB(rn)>>>(vm, ta, rn);
    
      d_norm_edges<<<CB(m)>>>(eu, ta, ev, ew, te, m);
    
      thrust::device_ptr<uint64_t> thrust_ptr(te);
      thrust::sort(thrust_ptr, thrust_ptr+m);
    
      d_mark_lightest_edges<<<CB(m)>>>(te, tc, m);
      scan(tc, tc, m);
      d_make_new_edges<<<CB(m)>>>(te, tc, eu, ev, ew, m);
    
      cudaErrChk(cudaMemcpy(&m, tc+m-1, 4, cudaMemcpyDeviceToHost));
   }  
  
  
   cudaErrChk(cudaFree(ta));
   cudaErrChk(cudaFree(tb));
   cudaErrChk(cudaFree(tc));
   cudaErrChk(cudaFree(te));
    
   return ret;
}

uint64_t vi_mst(int n, const vector<edge>& edges, int& time_ms) 
{
   int m = 2*edges.size();
  
   vector<vector<int>> av(n);
   vector<vector<int>> aw(n);
  
   for(auto e : edges) {
      av[e.u].push_back(e.v);
      aw[e.u].push_back(e.w);
      av[e.v].push_back(e.u);
      aw[e.v].push_back(e.w);
   }

   int* h_eu = new int[m];
   int* h_ev = new int[m];
   int* h_ew = new int[m];
  
   for(int i = 0, b = 0; i < n; ++i) {
      vector<int> tmp(av[i].size(), i);
      copy(tmp.begin(), tmp.end(), h_eu+b);
      copy(av[i].begin(), av[i].end(), h_ev+b);
      copy(aw[i].begin(), aw[i].end(), h_ew+b);
      b += av[i].size();
   }
  
   int* d_eu;
   int* d_ev;
   int* d_ew;

   cudaErrChk(cudaMalloc(&d_eu, m*sizeof(int)));
   cudaErrChk(cudaMalloc(&d_ev, m*sizeof(int)));
   cudaErrChk(cudaMalloc(&d_ew, m*sizeof(int)));

   cudaErrChk(cudaMemcpy(d_eu, h_eu, m*sizeof(int), cudaMemcpyHostToDevice));
   cudaErrChk(cudaMemcpy(d_ev, h_ev, m*sizeof(int), cudaMemcpyHostToDevice));
   cudaErrChk(cudaMemcpy(d_ew, h_ew, m*sizeof(int), cudaMemcpyHostToDevice));
  
   delete[] h_eu;
   delete[] h_ev;
   delete[] h_ew;
  
   auto start = clock();
   uint64_t ret = vi_mst(n, m, d_eu, d_ev, d_ew);
   time_ms = (int)(1000.0 * (clock() - start) / CLOCKS_PER_SEC);
  
   cudaErrChk(cudaFree(d_eu));
   cudaErrChk(cudaFree(d_ev));
   cudaErrChk(cudaFree(d_ew));
  
   return ret;
}

