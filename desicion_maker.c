#include "desicion_maker.h"

int T3_value;

void set_T3(int node_count)
{
    T3_value =  0.06*node_count;
}

int decide(double average_outdeg, int workset_size, int max_neighbour_in_workset, int * block_count, int * thread_per_block)
{
    if(workset_size < T2)
    {
        *block_count = workset_size;
        *thread_per_block = max_neighbour_in_workset;
        return B_QU;
    }
    else if (workset_size < T3_value)
    {
        if (average_outdeg < T1)
        {
            *block_count = workset_size / 1024 + 1;
            *thread_per_block = 1024;
            return T_QU;
        }else
        {
            *block_count = workset_size;
            *thread_per_block = max_neighbour_in_workset;
            return B_QU;
        }
    }
    else
    {
        if (average_outdeg < T1)
        {
            return T_BM;
        }else
        {
            return B_BM;
        }
    }    
}

int next_sample_distance()
{
    return 6;
}