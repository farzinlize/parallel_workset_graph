#ifndef _DESICION_MAKER_H
#define _DESICION_MAKER_H

#include "structures.h"

#define B_QU 0
#define B_BM 1
#define T_QU 2
#define T_BM 3

#define T1 32
#define T2 2688
/* T3/#num == 6% */

void set_T3(int node_count);

// int decide(double average_outdeg, int workset_size, int * block_count, int * thread_per_block);
int decide(double average_outdeg, int workset_size, int max_neighbour_in_workset, int * block_count, int * thread_per_block);
int next_sample_distance();

#endif