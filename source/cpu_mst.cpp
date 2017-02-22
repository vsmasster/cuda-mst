#include "mst.h"

#include <ctime>
#include <algorithm>
#include <vector>

static int find(std::vector<int>& p, int x)
{
  if(p[x] != x) {
    p[x] = find(p, p[x]);
  }
  
  return p[x];
}

uint64_t cpu_mst(int n, const std::vector<edge>& edges, int& time_ms)
{
  auto start = clock();
  
  std::vector<edge> es = edges;
  std::sort(es.begin(), es.end(), [](const edge& p, const edge& q) {
    return p.w < q.w;
  });
  
  std::vector<int> p(n);
  for(int i = 0; i < n; ++i)
    p[i] = i;
    
  uint64_t ret = 0;
  for(auto e : es) {
    if(find(p, e.u) != find(p, e.v)) {
      p[p[e.u]] = p[e.v];
      ret += e.w;
    }
  }
  
  time_ms = (int)(1000.0 * (clock() - start) / CLOCKS_PER_SEC);
  
  return ret;
}

