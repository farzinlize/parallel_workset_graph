#ifndef _LINEAR_ALGEBRA_CUH
#define _LINEAR_ALGEBRA_CUH

typedef struct argument_la
{
	/* kernel arguments for linear algebra kernels */
	dim3 grid_dim;
	dim3 block_dim;
    int shared_size;
    int max_depth;
}

void linear_algebra_bfs(graph g_h, int source);

#endif
