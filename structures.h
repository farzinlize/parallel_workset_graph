#ifndef _STRUCTURES_H
#define _STRUCTURES_H
/* TODO: add macro */

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* ### GRAPH ### */
struct graph
{
    /* TODO: add MACRO for creating lightweight graph structre */
    int * node_vector;          //number of nodes plus one for size of edges_vector
    int * edge_vector;          //number of out-going edges
    int * node_level_vector;    //used for BFS
    int * node_distance_vector; //used for SSSP
    int size;                   //number of nodes
};

/* host functions */
struct graph * consturct_graph(char * filename);
void destroy_graph(struct graph * g);
int get_average_out_deg(struct graph * g);

/* device functions */
struct graph * consturct_graph_device_from_file(char * filename);
struct graph * consturct_graph_device(struct graph * g);
void destroy_graph_device(struct graph * g);

/* ### QUEUE ### */
struct queue
{
    int * items;
    int size;
};

/* host functions */
struct queue * construct_queue(int max_size);
void destroy_queue(struct queue * q);
int queue_push(struct queue * workset, int item);
int queue_clear(struct queue * workset);
int queue_get(struct queue * workset, int index, int * item);

/* device functions */
struct queue * construct_queue_device(int max_size);
void destroy_queue_device(struct queue * q);
int queue_push_device(struct queue * workset_d, int item, int * workset_size_h);

#endif