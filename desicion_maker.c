#include "desicion_maker.h"

int T3_value;

void set_T3(int node_count)
{
    T3_value =  0.06*node_count;
}

int decide(double average_outdeg, int workset_size)
{
    if(workset_size < T2){
        return B_QU;
    }
    else if (workset_size < T3_value){
        if (average_outdeg < T1){
            return T_QU;
        }else{
            return B_QU;
        }
    }
    else{
        if (average_outdeg < T1){
            return T_BM;
        }else{
            return B_BM;
        }
    }    
}

int next_sample_distance()
{
    return 6;
}
