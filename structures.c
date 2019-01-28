#include "structures.h"

/* ### GRAPH : HOST ### */
struct graph * consturct_graph(char * nodes_file, char * edges_file)
{
    struct graph * g = (struct graph *) malloc(sizeof(struct graph));

    /* read nodes */
    FILE *fptr = fopen(nodes_file, "r");
    if (fptr == NULL)
    {
        printf("Cannot open file \n");
        exit(0);
    }

    int node_size, edge_size, size, n, i;
    fscanf(fptr, "%d", &node_size);
    int * nodes = malloc(sizeof(int) * node_size);
    i = 0;
    size = node_size;
    while (size--) {
        fscanf(fptr, "%d", &n);
        nodes[i++] = n;
    }
    fclose(fptr);


    /* read edges */
    fptr = fopen(edges_file, "r");
    if (fptr == NULL)
    {
        printf("Cannot open file \n");
        exit(0);
    }

    fscanf(fptr, "%d", &edge_size);
    int * edges = malloc(sizeof(int) * edge_size);
    i = 0;
    size = edge_size;
    while (size--) {
        fscanf(fptr, "%d", &n);
        edges[i++] = n;
    }
    fclose(fptr);

    g->node_vector = nodes;
    g->edge_vector = edges;
    g->size = node_size;

    return g;
}


void destroy_graph(struct graph * g)
{
    return ;
}

double get_average_out_deg(struct graph * g)
{
    return g->node_vector[g->size] / g->size; //size of edge_vector
}


/* ### GRAPH : DEVICE ### */
struct graph * consturct_graph_device_from_file(char * filename)
{
    return 0;
}

void consturct_graph_device(struct graph * g_h, struct graph ** g_d)
{
    // struct graph * g_d;
    struct graph * g_h2 = (struct graph *)malloc(sizeof(struct graph));
    
    cudaMalloc(&g_h2->node_vector, sizeof(int)*(g_h->size));
    cudaMemcpy(g_h2->node_vector, g_h->node_vector, sizeof(int)*(g_h->size), cudaMemcpyHostToDevice);

    cudaMalloc(&g_h2->edge_vector, sizeof(int)*(g_h->edge_vector[g_h->size]));
    cudaMemcpy(g_h2->edge_vector, g_h->edge_vector, sizeof(int)*(g_h->node_vector[g_h->size]), cudaMemcpyHostToDevice);

    // cudaMemcpy(g_h2->size, g_h->size, sizeof(int), cudaMemcpyHostToDevice);
    g_h2->size = g_h->size;

    cudaMemcpy(*g_d, g_h2, sizeof(struct graph), cudaMemcpyHostToDevice);


    /*    initial graph on device    */
    // cudaMalloc((void **)&g_d, sizeof(struct graph *));
    // cudaMalloc((void **)&g_d->size, sizeof(int));
    // cudaMalloc((void **)&g_d->node_vector, sizeof(int)*(g_h->size+1));
    // cudaMalloc((void **)&g_d->edge_vector, sizeof(int)*(g_h->node_vector[g_h->size]));

    /*    transform graph to device  */
    // cudaMemcpy(g_d->node_vector, g_h->node_vector, sizeof(int)*(g_h->size+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(g_d->edge_vector, g_h->edge_vector, sizeof(int)*(g_h->node_vector[g_h->size]), cudaMemcpyHostToDevice);
    // cudaMemcpy(&g_d->size, &g_h->size, sizeof(int), cudaMemcpyHostToDevice);

    // return g_d;
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
// struct queue * construct_queue_device(int max_size)
// {
//     struct queue * workset_d;
//     cudaMalloc((void **)&workset_d, sizeof(struct queue *));
//     cudaMalloc((void **)&(workset_d->items), sizeof(int)*max_size);
//     return workset_d;
// }

void destroy_queue_device(struct queue * q)
{
    return ;
}

int queue_push_device(struct queue * workset_d, int item, int * workset_size_h)
{
    cudaMemcpy(&workset_d->items[(*workset_size_h)++], &item, sizeof(int), cudaMemcpyHostToDevice);
    return 0;
    cudaMemcpy(&workset_d->size, workset_size_h, sizeof(int), cudaMemcpyHostToDevice);
    return 0;
}