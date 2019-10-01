#ifndef _LINEAR_ALGEBRA_CUH
#define _LINEAR_ALGEBRA_CUH

// #include "structures.h"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include <cuda.h>
// #include <stdio.h>
#include "kernels.cuh"

typedef struct argument_la
{
	/* kernel arguments for linear algebra kernels */
	dim3 grid_dim;
	dim3 block_dim;
    int shared_size;
    int max_depth;
} argument_la;

void linear_algebra_bfs(graph g_h, int source);
__global__ void linear_algebra_bfs(graph g_d, int source, int * multiplier_d, int ** working_array, argument_la argument);
__global__ void CSR_multiply_one_BFS(graph g_d, int * multiplier_d, int ** working_array, int level);

#endif
