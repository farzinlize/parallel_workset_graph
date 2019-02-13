#ifndef _FUZZY_TIMING_
#define _FUZZY_TIMING_

#ifdef WINDOWS
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
int gettimeofday(struct timeval * tp, struct timezone * tzp);
#endif

#ifdef LINUX
#include<time.h>
#include<sys/time.h>
#endif

void set_clock();
double get_elapsed_time();

#endif