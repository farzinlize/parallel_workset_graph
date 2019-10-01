#ifndef _KERNELS_CUH
#define _KERNELS_CUH

#include "structures.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef struct argument
{
	/* covering kernel arguments */
	int covering_block_count;
	int covering_block_size;

	/* reduction add kernel arguments */
	int add_half_full_flag;
	int add_block_count;
	int add_block_size;
	int * add_result;
} argument;

__global__ void one_bfs_B_QU(graph g, queue workset, char * update, int level);
__global__ void one_bfs_B_BM(graph g, char * bitmap_mask, char * update, int level);
__global__ void one_bfs_T_QU(graph g, queue workset, char * update, int level);
__global__ void one_bfs_T_BM(graph g, char * bitmap_mask, char * update, int level);
__global__ void workset_update_QU(char * update, queue workset);
__global__ void workset_update_BM(char * update, char * bitmap_mask);
__global__ void add_kernel(char *a_in, int * out);
__global__ void add_kernel_half(char * a_in, int * out);
__global__ void add_kernel_full(char * a_in, int * out);
__global__ void inital_char_array(char * array, char value);
__global__ void inital_int_array(int * array, int value, int size);

__global__ void T_BM_bfs(graph g_d, int source, char * bitmap_mask, char * update, argument argument);

__device__ void warpReduce(volatile int* sdata, int tid);
__device__ int sum_array_one_thread(int * a_in, int size);

// __global__ void linear_algebra_bfs(graph g_d, int source, int * multiplier_d, int * working_array, argument_la argument);
// __global__ void CSR_multiply_one_BFS(graph g_d, int * multiplier_d, int * working_array, int * result_vector, int level);

#endif