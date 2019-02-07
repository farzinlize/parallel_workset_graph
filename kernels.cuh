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

__global__ void one_bfs_B_QU(graph g, queue workset, char * update, int level);
__global__ void one_bfs_B_BM(graph g, char * bitmap_mask, char * update, int level);
__global__ void one_bfs_T_QU(graph g, queue workset, char * update, int level);
__global__ void one_bfs_T_BM(graph g, char * bitmap_mask, char * update, int level);
__global__ void workset_update_QU(char * update, queue workset);
__global__ void workset_update_BM(char * update, char * bitmap_mask);
__global__ void add_kernel(char *a_in, int * out);
__global__ void inital_char_array(char * array, char value);
__global__ void inital_int_array(int * array, int value, int size);

#endif