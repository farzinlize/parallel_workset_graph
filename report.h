#ifndef _REPORT_H_
#define _REPORT_H_

#include <string.h>
#include <stdio.h>

void initial_fileout();
void make_compare_file(char * file_name, char * result_1_name, int * result_1, char * result_2_name, int * result_2, int size);

#endif