#ifndef _DESICION_MAKER_H
#define _DESICION_MAKER_H

#include "structures.h"

#define B_QU 0
#define B_BM 1
#define T_QU 2
#define T_BM 3

#define T1 32
#define T2 2688

int decide(double average_outdeg, int workset_size);
int next_sample_distance();
void set_T3(int node_count);

#endif