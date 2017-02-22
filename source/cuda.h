#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda.h>

#define cudaErrChk(statement) { \
                              cudaError_t err = (statement); \
                              if(err != cudaSuccess) { \
                                fprintf(stderr, "%s:%d cuda error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
                                abort(); \
                              } \
                            }
                            

#define CEIL(a, b) (((a)-1)/(b)+1)

#define CB(n) CEIL(n, 1024), 1024 
