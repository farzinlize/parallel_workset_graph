#include "report.h"

FILE * fileout;

void initial_fileout()
{
    int version = VERSION;
    char fileout_name[50];
    char extention[20];

    strcpy(extention, "ag");

    #ifdef DP
    strcat(extention, "_dp");
    #endif
    #ifdef DEBUG
    strcat(extention, "_debug");
    #endif
    #ifdef TEST
    strcat(extention, "_test");
    #endif
    #ifdef DETAIL
    strcat(extention, "_detail");
    #endif

    sprintf(fileout_name, "out/%s_%d.out", extention, version);
    fileout = fopen(fileout_name, "wb");
}

void make_compare_file(char * file_name, char * result_1_name, int * result_1, char * result_2_name, int * result_2, int size)
{
    FILE * result_file = fopen(file_name, "wb");
    int max_level_1 = -1, max_level_2 = -1, fault_tree = 0, diffrenet = 0;
    int nodes_in_level_result_1[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int nodes_in_level_result_2[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(int i = 0; i<size; i++){
        if(result_1[i] >= 20 || result_2[i] >= 20){
            if((result_1[i] >= 20 && result_2[i] < 20) || (result_1[i] < 20 && result_2[i] >= 20))
                fault_tree++;
            fprintf(result_file, "node (is not in same tree): %d\t| %s level: %d\t| %s level: %d\n", i, result_1_name, result_1[i], result_2_name, result_2[i]);
            continue;
        }
        if(result_1[i] != result_2[i])    diffrenet++;
        if(result_1[i] > max_level_1)    max_level_1 = result_1[i];
        if(result_2[i] > max_level_2)   max_level_2 = result_2[i];
        nodes_in_level_result_1[result_1[i]]++;
        nodes_in_level_result_2[result_2[i]]++;
        fprintf(result_file, "node: %d\t| %s level: %d\t| %s level: %d\n", i, result_1_name, result_1[i], result_2_name, result_2[i]);
    }
    fprintf(result_file, "----------------------------------------\n");
    fprintf(result_file, "max level in %s run: %d\nmax level in %s run: %d\n", result_1_name, max_level_1, result_2_name, max_level_2);
    fprintf(result_file, "FAULT TREE --> %d\n", fault_tree);
    fprintf(result_file, "NUMBER OF DIFFRENT (between two run) --> %d\n", diffrenet);
    fprintf(result_file, "number of nodes in each level (%s)\n", result_1_name);
    for(int i=0;i<max_level_1;i++)    fprintf(result_file, "level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_1[i]);
    fprintf(result_file, "number of nodes in each level (%s)\n", result_2_name);
    for(int i=0;i<max_level_2;i++)    fprintf(result_file, "level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_2[i]);
    fclose(result_file);

    /* brif summery */
    printf("----------------------------------------\n");
    printf("max level in %s run: %d\nmax level in %s run: %d\n", result_1_name, max_level_1, result_2_name, max_level_2);
    printf("FAULT TREE --> %d\n", fault_tree);
    printf("NUMBER OF DIFFRENT (between two run) --> %d\n", diffrenet);
    printf("number of nodes in each level (%s)\n", result_1_name);
    for(int i=0;i<max_level_1;i++)    printf("level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_1[i]);
    printf("number of nodes in each level (%s)\n", result_2_name);
    for(int i=0;i<max_level_2;i++)    printf("level: %d\tnumber of nodes: %d\n", i, nodes_in_level_result_2[i]);
    fclose(result_file);
}