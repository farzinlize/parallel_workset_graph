#include "structures.h"

extern FILE * fileout;

/* ### GRAPH : HOST ### */
graph consturct_graph(const char * nodes_file, const char * edges_file)
{
    /* set and define variables */
	graph result;
    FILE * fptr;
    int node_size, edge_size, readed, i;

    #ifdef CSR_VALIDATION
    /* define flags for validating readed graph as a valid CSR graph */
    /* CSR: Compressed Sparese Row                                   */
    int pre_readed = -1;
    bool first_zero_node_vector = false;
    bool edge_vector_size = false;
    bool ascending_node_vector = true;
    #endif

	/* open nodes file */
	fptr = fopen(nodes_file, "r");
	if (fptr == NULL){
		printf("[ERROR] Cannot open file\n");
		exit(0);
	}

    /* read first line for number of nodes_file items       */
    /* hint: graph size is node_files items count minus one */
	fscanf(fptr, "%d", &node_size);
	result.node_vector = (int *) malloc(sizeof(int) * node_size);
    result.size = node_size-1;
	i = 0;
	while (i < node_size) {
		fscanf(fptr, "%d", &readed);
		result.node_vector[i++] = readed;

        #ifdef CSR_VALIDATION
        if (i == 1 && readed == 0) first_zero_node_vector = true;
        if (readed < pre_readed) ascending_node_vector = false;
        #endif
	}

    #ifdef CSR_VALIDATION
    fprintf(fileout, "[VALIDATE][CSR_GRAPH] first node_vector item must be zero: %s\n", first_zero_node_vector ? "PASS" : "FAIL");
    fprintf(fileout, "[VALIDATE][CSR_GRAPH] node_vector must be ascending: %s\n", ascending_node_vector ? "PASS" : "FAIL");
    #endif

    /* close nodes_file */
	fclose(fptr);


	/* open edges file */
	fptr = fopen(edges_file, "r");
	if (fptr == NULL)
	{
		printf("[ERROR] Cannot open file\n");
		exit(0);
	}

    /* read first line for number of edges_file items */
	fscanf(fptr, "%d", &edge_size);

    #ifdef CSR_VALIDATION
    if (result.node_vector[result.size] == edge_size) edge_vector_size = true;
    fprintf(fileout, "[VALIDATE][CSR_GRAPH] node_vector last item must be equal to edge_vector size: %s\n", edge_vector_size ? "PASS" : "FAIL");
    #endif

	result.edge_vector = (int *) malloc(sizeof(int) * edge_size);
	i = 0;
	while (i < edge_size) {
		fscanf(fptr, "%d", &readed);
		result.edge_vector[i++] = readed;
	}

    /* close edges_file */
	fclose(fptr);

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][GRAPH] graph node count: %d\n", result.size);
    fprintf(fileout, "[DETAIL][GRAPH] graph directed edges count: %d\n", result.node_vector[result.size]);
    #endif

	return result;
}

void destroy_graph(graph g)
{
    free(g.node_vector);
    free(g.edge_vector);

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][FREE] node_vector and edge_vector of host graph destroyed\n");
    #endif
}

double get_average_out_deg(graph g)
{
    /* size of edge_vector indicate summation of out degres */
    #ifdef DETAIL
    double avrage_out_deg = (g.node_vector[g.size]) / g.size;
    fprintf(fileout, "[DETAIL][GRAPH] graph average out degree: %f\n", avrage_out_deg);
    return avrage_out_deg;
    #else
    return (g.node_vector[g.size]) / g.size;
    #endif
}


/* ### GRAPH : DEVICE ### */
graph consturct_graph_device_from_file(const char * nodes_file, const char * edges_file)
{
    return consturct_graph_device(consturct_graph(nodes_file, edges_file)); //TODO: overlap job of reading from file and copy to device
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

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][DEVICE] node_vector and edge_vector deployed on device\n");
    #endif

    return g_d;
}

void destroy_graph_device(graph g_d)
{
    cudaFree(g_d.node_vector);
    cudaFree(g_d.edge_vector);

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][FREE] node_vector and edge_vector of device destroyed\n");
    #endif
}


/* ### QUEUE : HOST ### */
queue construct_queue(int max_size)
{
    queue result;
    result.size = 0;
    result.items = (int *)malloc(sizeof(int)*max_size);
    return result;
}

void destroy_queue(queue q)
{
    free(q.items);
}

int queue_push(queue * q, int item)
{
    q->items[q->size++] = item;
    return 0; //TODO: check for maximum size
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

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][DEVICE] workset queue builed and first item added successfully\n");
    #endif

    return q_d;
}

void destroy_queue_device(queue q_d)
{
    cudaFree(q_d.items);

    #ifdef DETAIL
    fprintf(fileout, "[DETAIL][FREE] workset queue on device destroyed\n");
    #endif
}