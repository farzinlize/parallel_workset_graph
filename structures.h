#ifndef _STRUCTURES_H
#define _STRUCTURES_H
/* TODO: add macro */

/* ### GRAPH ### */
struct graph
{
    /* TODO: add MACRO for creating lightweight graph structre */
    int * node_vector;
    int * node_level_vector;
    int * node_distance_vector;
    int * edge_vector;
    int size;
};

struct graph consturct_graph(char * filename);
int get_avrage_out_deg(struct graph * g);

/* ### QUEUE ### */
struct queue
{
    int * items;
    int size;
};

struct queue construct_queue();
int queue_push(int item);
int queue_clear();
int queue_get(int index, int * item);


#endif