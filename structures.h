#ifndef _STRUCTURES_H
#define _STRUCTURES_H
/* TODO: add macro */

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

struct graph * consturct_graph(char * filename);
int get_average_out_deg(struct graph * g);

/* ### QUEUE ### */
struct queue
{
    int * items;
    int size;
};

struct queue * construct_queue(int max_size);
int queue_push(int item);
int queue_clear();
int queue_get(int index, int * item);


#endif