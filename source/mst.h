#pragma once

#include <cstdint>
#include <vector>



struct edge {
  int u, v, w;
};

uint64_t cpu_mst(int n, const std::vector<edge>& edges, int& time_ms);

uint64_t vi_mst(int& n, int m, int* eb, int* ev, int* ew, int* vm = 0, int rn = 0);
uint64_t vi_mst(int n, const std::vector<edge>& edges, int& time_ms);

uint64_t dpk_mst(int n, const std::vector<edge>& edges, int& time_ms);


