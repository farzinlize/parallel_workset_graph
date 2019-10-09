#ifndef _LINEAR_ALGEBRA_CUH
#define _LINEAR_ALGEBRA_CUH

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
__global__ void spmv_csr_scalar_kernel (graph g_d, int * multiplier_d, int * working_array, int level);
__global__ void set_result (graph g_d, int * multiplier_d, int * working_array);
__global__ void spmv_csr_vector_kernel (graph g_d, int * multiplier_d, int * working_array, int level);

#endif
