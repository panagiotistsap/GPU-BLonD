
#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef PARALLEL
	#include <omp.h>        // omp_get_thread_num(), omp_get_num_threads()
#else
	int omp_get_max_threads();
	int omp_get_num_threads();
	int omp_get_thread_num();
#endif

#ifdef USEFLOATS
	typedef float REAL;
#else
	typedef double REAL;
#endif

#endif // _COMMON_H_