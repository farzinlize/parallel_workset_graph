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

#ifdef DP
__global__ void linear_algebra_bfs_dp(graph g_d, int source, int * multiplier_d, int * working_array, argument_la argument);
#endif

__global__ void CSR_multiply_reductionAdd(graph g_d, int * multiplier_d, int * working_array);
__global__ void result_and_BFS(graph g_d, int * multiplier_d, int * working_array, int block_count_y, int level);

void linear_algebra_bfs(graph g_h, int source);
void linear_algebra_bfs_scalar(graph g_h, int source);

#endif
