#ifndef _STRUCTURES_H
#define _STRUCTURES_H
/* TODO: add macro */

#include <cuda.h>
#include "cuda_runtime.h"

/* ### GRAPH ### */
typedef struct graph
{
    /* TODO: add MACRO for creating lightweight graph structre */
    int * node_vector;          //number of nodes plus one for size of edges_vector
    int * edge_vector;          //number of out-going edges
    int * node_level_vector;    //used for BFS
    int * node_distance_vector; //used for SSSP
    int size;                   //number of nodes
} graph;

/* host functions */
graph consturct_graph(char * filename);
void destroy_graph(graph g);
double get_average_out_deg(graph g);

/* device functions */
graph consturct_graph_device_from_file(char * filename);
graph consturct_graph_device(graph g);
void destroy_graph_device(graph g_d);

/* ### QUEUE ### */
typedef struct queue
{
    int * items;
    int size;
} queue;

/* device functions */
queue construct_queue_device_with_source(int max_size, int * source_p);
void destroy_queue_device(queue q);

#endif