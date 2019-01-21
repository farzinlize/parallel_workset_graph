#include "structures.h"


void one_bfs_B_QU(struct graph * g, struct queue * workset, int level)
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

void one_bfs_B_BM(struct graph * g, char * bitmap, int level)
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

void one_bfs_T_QU(struct graph * g, struct queue * workset, int level)
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

void one_bfs_T_BM(struct graph * g, struct queue * workset, int level)
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

void run_bfs(struct graph * g)
{
    struct queue * workset = (workset *) malloc(sizeof(queue));
    g->node_level_vector[0] = 0;
    queue_push(workset, 0);
    int level = 0;
    while (workset->size != 0) {
        int algo = decide();
        if (algo == B_QU) {
            one_bfs_B_QU(g, workset, level++);
        } else if (algo == B_BM) {
            // one_bfs_B_BM(g, workset, level++);
            continue;
        } else if (algo == T_QU) {
            // one_bfs_T_QU(g, workset, level++);
            continue;
        } else if (algo == T_BM) {
            // one_bfs_T_BM(g, workset, level++);
            continue;
        }
    }
}

int main()
{
    return 0;
}