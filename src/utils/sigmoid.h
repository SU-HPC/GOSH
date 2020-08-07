#ifndef _SIGMOID
#define _SIGMOID
#include "type_def.h"
#define SIGMOID_BOUND 6.0
#define SIGMOID_TABLE_SIZE 1024
#define SHARED_MEMORY
const float SIGMOID_RESOLUTION = (2.0f*SIGMOID_BOUND)/SIGMOID_TABLE_SIZE;
void init_sig_table(float*& sigmoid_lookup_table);
__device__ float fastSig(double num, float* table);
#endif
