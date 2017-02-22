#include "mst.h"

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

int main()
{
  int n, m;
  std::cin >> n >> m;
  std::vector<edge> edges;
  for (int i = 0; i < m; i++) {
    int u, v, w;
    std::cin >> u >> v >> w;
    edges.push_back({u, v, w});
  }

  int time_ms;
    
  uint64_t vres = cpu_mst(n, edges, time_ms);
  std::cout << "cpu_mst: " << time_ms << "\n";
    
  uint64_t res = vi_mst(n, edges, time_ms);
  // if(res != vres) {
  //   std::cout << "vi_mst incorrect answer\n";
  //   return 0;
  // }
  std::cout << "vi_mst: " << time_ms << "\n";
  
  res = dpk_mst(n, edges, time_ms);
  // if(res != vres) {
  //   std::cout << "dpk_mst incorrect answer\n";
  //   return 0;
  // }
  std::cout << "dpk_mst: " << time_ms << "\n"; 
  
  
  return 0;
}
