#include "kernels.h"

/* ### WORKSET_GEN KERNELS ### */
__global__ void workset_update_BM(char * update, char * bitmap_mask)
{
    int tid = threadIdx.x;
    if (update[tid])    //thread divergence ALERT
    {
        bitmap_mask[tid] = 1;
    } else 
    {
        bitmap_mask[tid] = 0;
    }
}

__global__ void workset_update_QU(char * update, struct queue * workset)
{
    int tid = threadIdx.x;
    if (tid == 0)   //first thread clear the workset (no critical section)
    {
        int queue_clear(workset);
    }
    if (update[tid])
    {
        atomicExch(&workset->items[workset->size], tid);
        atomicAdd(&workset->size, 1);
    }
}


/* ### BFS KERNELS ### */
__global__ void one_bfs_B_QU(struct graph * g, struct queue * workset, char * update, int level)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    if (bid < workset.size) //each block process a workset entry
    {
        int node = workset->items[bid];
        int neighbours_count = g->edge_vector[node+1] - g->edge_vector[node];
        if (tid < neighbours_count)
        {
            //each thread in block process a neighbour of original node of block
            int neighbour = g->edge_vector[node] + tid;
            if (g->node_level_vector[neighbour] > level + 1) {
                g->node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_B_BM(struct graph * g, char * bitmap_mask, char * update, int level)
{
    int tid = threadIdx.x;
    int node = blockIdx.x; //each block process a node
    if (bitmap_mask[node])
    {
        int neighbours_count = g->edge_vector[node+1] - g->edge_vector[node];
        if (tid < neighbours_count)
        {
            //each thread in block process a neighbour of original node of block
            int neighbour = g->edge_vector[node] + tid;
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_T_QU(struct graph * g, struct queue * workset, char * update, int level)
{
    int tid = threadIdx.x;
    if (tid < workset.size) //each thread process a workset entry
    {
        int node = workset->items[tid];
        //visiting neighbours
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++)
        {
            if (g->node_level_vector[neighbour] > level+1)
            {
                g->node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

__global__ void one_bfs_T_BM(struct graph * g, char * bitmap_mask, char * update, int level)
{
    int node = threadIdx.x;
    if (bitmap_mask[node]) //each thread process a node if it's in bitmap_mask
    {
        //visiting neighbours
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++)
        {
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                update[neighbour] = 1;
            }
        }
    }
}

/* ### DECISION KERNELS ### */
__global__ void add_kernel(int *a_in)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
	__syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1)
    {
		if (tid_block < s)
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		__syncthreads();
	}

    if (tid_block == 0)
        a_in[blockIdx.x] = a_s[0];
}