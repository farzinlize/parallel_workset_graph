#include "structures.h"

/* ### GRAPH : HOST ### */
struct graph * consturct_graph(char * filename)
{
    return 0;
}

void destroy_graph(struct graph * g)
{
    return ;
}

int get_average_out_deg(struct graph * g)
{
    return g->node_vector[g->size]; //size of edge_vector
}


/* ### GRAPH : DEVICE ### */
struct graph * consturct_graph_device_from_file(char * filename)
{
    return 0;
}

struct graph * consturct_graph_device(struct graph * g_h)
{
    struct graph * g_d;

    /*    initial graph on device    */
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d, sizeof(struct graph *)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->size, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->node_vector, sizeof(int)*(g_h->size+1)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_d->edge_vector, sizeof(int)*(g_h->node_vector[g->size])));

    /*    transform graph to device  */
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->node_vector, g_h->node_vector, sizeof(int)*(g_h->size+1), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->edge_vector, g_h->edge_vector, sizeof(int)*(g_h->node_vector[g->size]), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(g_d->size, g_h->size, sizeof(int), cudaMemcpyHostToDevice));

    return g_d;
}

void destroy_graph_device(struct graph * g)
{
    return ;
}

/* ### QUEUE : HOST ### */
struct queue * construct_queue(int max_size)
{
    return 0;
}

struct queue * construct_queue_device(int max_size)
{
    return 0;
}

void destroy_queue(struct queue * q)
{
    return ;
}

int queue_push(struct queue * workset, int item)
{
    return 0;
}

int queue_clear(struct queue * workset)
{
    return 0;
}

int queue_get(struct queue * workset, int index, int * item)
{
    return 0;
}


/* ### QUEUE : DEVICE ### */
struct queue * construct_queue_device(int max_size)
{
    struct queue * workset_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&workset_d, sizeof(struct queue *)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&workset_d->items, sizeof(int)*max_size);
    return workset_d;
}

void destroy_queue_device(struct queue * q)
{
    return ;
}

int queue_push_device(struct queue * workset_d, int item, int * workset_size_h)
{
    CUDA_CHECK_RETURN(cudaMemcpy(&workset_d->items[(*workset_size_h)++], &item, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&workset_d->size, workset_size_h, sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}