#include "structures.h"
#include "desicion_maker.h"

void run_bfs(struct graph * g)
{
    double avrage_outdeg = get_avrage_out_deg(g);
    struct queue * workset = (struct queue *) malloc(sizeof(struct queue));
    g->node_level_vector[0] = 0;
    queue_push(workset, 0);
    int level = 0;
    while (workset->size != 0) {
        int algo = decide(avrage_outdeg, workset->size);
        if (algo == B_QU) {
            one_bfs_B_QU(g, workset, level++);
        } else if (algo == B_BM) {
            // one_bfs_B_BM(g, workset, level++);
            continue;
        } else if (algo == T_QU) {
            // one_bfs_T_QU(g, workset, level++);
            continue;
        } else if (algo == T_BM) {
            // one_bfs_T_BM(g, workset, level++);
            continue;
        }
    }
}

int main()
{
    return 0;
}