#include "sequential.h"

queue sequential_one_bfs_QU(graph * g, queue workset, int level)
{
    #ifdef DEBUG
    printf("[DEBUG][ONE_BFS] bfs level: %d\n", level);
    #endif

    queue next_workset = construct_queue(g->size);
    for(int workset_index = 0; workset_index < workset.size ; workset_index++){
        int node = workset.items[workset_index];
        int node_neighbour_index = g->node_vector[node];
        int neighbours_count = g->node_vector[node+1] - node_neighbour_index;

        //visiting neighbours
        for (int neighbour_id = 0; neighbour_id < neighbours_count; neighbour_id++){
            int neighbour = g->edge_vector[node_neighbour_index + neighbour_id];
            if (g->node_level_vector[neighbour] > level+1){
                g->node_level_vector[neighbour] = level + 1;
                queue_push(&next_workset, neighbour);
            }
        }
    }
    destroy_queue(workset);
    return next_workset;
}

void sequential_run_bfs_QU(graph * g, int root)
{
    g->node_level_vector = (int *)malloc(sizeof(int)*g->size);
    for(int i=0;i<g->size;i++) g->node_level_vector[i]=INT_MAX;
    queue workset = construct_queue(g->size);
    int level = 0;
    queue_push(&workset, root);
    g->node_level_vector[root] = 0;

    #ifdef DEBUG
    printf("[DEBUG][BFS] workset size after first push: %d\n", workset.size);
    #endif

    while(workset.size != 0){

        #ifdef DEBUG
        printf("[DEBUG][BFS] workset size: %d\n", workset.size);
        #endif

        workset = sequential_one_bfs_QU(g, workset, level++);
    }
}