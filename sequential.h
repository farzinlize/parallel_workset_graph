#ifndef _SEQUENTIAL_H
#define _SEQUENTIAL_H

#include "structures.h"
#include <limits.h>

queue sequential_one_bfs_QU(graph * g, queue workset, int level);
void sequential_run_bfs_QU(graph * g, int root);

void sequential_csr_mult(graph g, int * vector, int * result);

#endif