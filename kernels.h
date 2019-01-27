#ifndef _KERNELS_H
#define _KERNELS_H

extern "C" {
    #include "structures.h"
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__global__ void one_bfs_B_QU(struct graph * g, struct queue * workset, char * update, int level);
__global__ void one_bfs_B_BM(struct graph * g, char * bitmap_mask, char * update, int level);
__global__ void one_bfs_T_QU(struct graph * g, struct queue * workset, char * update, int level);
__global__ void one_bfs_T_BM(struct graph * g, char * bitmap_mask, char * update, int level);
__global__ void workset_update_QU(char * update, struct queue * workset);
__global__ void workset_update_BM(char * update, char * bitmap_mask);
__global__ void add_kernel(int *a_in);
__global__ void inital_char_array(char * array, char value, int size);
__global__ void inital_int_array(int * array, int value, int size);

#endif