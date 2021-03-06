#ifndef _STRUCTURES_H
#define _STRUCTURES_H
/* TODO: add macro */

#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

/* ### GRAPH ### */
typedef struct graph
{
    /* TODO: add MACRO for creating lightweight graph structre */
    int * node_vector;              //number of nodes plus one for size of edges_vector
    int * edge_vector;              //number of out-going edges
    int * node_level_vector;        //used for BFS
    int * node_predecessors_vector; //used for BFS
    int * node_distance_vector;     //used for SSSP
    int size;                       //number of nodes
} graph;

/* host functions */
graph consturct_graph(const char * nodes_file, const char * edges_file);
void destroy_graph(graph g);
double get_average_out_deg(graph g);

/* device functions */
graph consturct_graph_device_from_file(const char * nodes_file, const char * edges_file);
graph consturct_graph_device(graph g);
void destroy_graph_device(graph g_d);

/* ### QUEUE ### */
typedef struct queue
{
    int * items;
    int size;
} queue;

/* host functions */
queue construct_queue(int max_size);
void destroy_queue(queue q);
int queue_push(queue * q, int item);

/* device functions */
queue construct_queue_device_with_source(int max_size, int * source_p);
void destroy_queue_device(queue q);

#endif