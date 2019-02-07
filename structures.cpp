#include "structures.h"

/* ### GRAPH : HOST ### */
graph consturct_graph(char * filename)
{
    graph result;
    return result;
}

void destroy_graph(graph g)
{
    return ;
}

double get_average_out_deg(graph g)
{
    /* size of edge_vector indicate summation of out degres */
    return (g.node_vector[g.size]) / g.size;
}


/* ### GRAPH : DEVICE ### */
graph consturct_graph_device_from_file(char * filename)
{
    return consturct_graph_device(consturct_graph(filename)); //TODO: overlap job of reading from file and copy to device
}

graph consturct_graph_device(graph g)
{
    graph g_d;
    g_d.size = g.size;

    /*    initial graph vectors on device    */
    cudaMalloc((void **)&(g_d.node_vector), sizeof(int)*(g.size+1));
    cudaMalloc((void **)&(g_d.edge_vector), sizeof(int)*(g.node_vector[g.size]));

    /*    transform graph vectors to device  */
    cudaMemcpy(g_d.node_vector, g.node_vector, sizeof(int)*(g.size+1), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d.edge_vector, g.edge_vector, sizeof(int)*(g.node_vector[g.size]), cudaMemcpyHostToDevice);

    return g_d;
}

void destroy_graph_device(graph g_d)
{
    cudaFree(g_d.node_vector);
    cudaFree(g_d.edge_vector);

    cudaFree(g_d.node_level_vector); //TODO: level or distance?
}


/* ### QUEUE : DEVICE ### */
queue construct_queue_device_with_source(int max_size, int * source_p)
{
    queue q_d;
    q_d.size = 1;

    /* initial data on device */
    cudaMalloc((void **)&(q_d.items), sizeof(int)*max_size);

    /* add source to items */
    cudaMemcpy(q_d.items, source_p, sizeof(int), cudaMemcpyHostToDevice); //TODO: test it

    return q_d;
}

void destroy_queue_device(queue q_d)
{
    cudaFree(q_d.items);
}