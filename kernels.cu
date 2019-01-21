#include "kernels.h"

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