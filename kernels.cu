#include "structures.h"

__global__ void one_bfs_B_QU(struct graph * g, struct queue * workset, int level)
{
    for (int i = 0; i < workset->size; i++) {
        int node = workset->items[i];
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++) {
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                queue_push(workset, neighbour);
            }
        }
    }
}

__global__ void one_bfs_B_BM(struct graph * g, char * bitmap, int level)
{
    for (int i = 0; i < workset->size; i++) {
        int node = workset->items[i];
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++) {
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                queue_push(workset, neighbour);
            }
        }
    }
}

__global__ void one_bfs_T_QU(struct graph * g, struct queue * workset, int level)
{
    for (int i = 0; i < workset->size; i++) {
        int node = workset->items[i];
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++) {
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                queue_push(workset, neighbour);
            }
        }
    }
}

__global__ void one_bfs_T_BM(struct graph * g, struct queue * workset, int level)
{
    for (int i = 0; i < workset->size; i++) {
        int node = workset->items[i];
        for (int neighbour = g->edge_vector[node]; neighbour < g->edge_vector[node+1]; neighbour_id++) {
            if (g->node_level_vector[neighbour] > level+1) {
                g->node_level_vector[neighbour] = level + 1;
                queue_push(workset, neighbour);
            }
        }
    }
}