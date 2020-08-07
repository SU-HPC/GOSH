#include <stdio.h>
#include "sigmoid.h"
#include "align.h"
using namespace std;
void init_sig_table(float*& sigmoid_lookup_table){
#ifdef ALIGNMENT
  sigmoid_lookup_table = new float[SIGMOID_TABLE_SIZE];
#else
  sigmoid_lookup_table =  (float*)aligned_malloc((size_t)(SIGMOID_TABLE_SIZE*sizeof(float)),(size_t)ALIGNMENT);
#endif
  for (int i =0; i<SIGMOID_TABLE_SIZE;i++){
    float x = (-SIGMOID_BOUND+float(i*SIGMOID_RESOLUTION));
    sigmoid_lookup_table[i]=1/(1+exp(-(x)));
  }
}

__device__ float fastSig(double num, float* table){
  if (num>=SIGMOID_BOUND)
    return 1;
  else if(num<-SIGMOID_BOUND)
    return 0;
  int index_in_table = (num+SIGMOID_BOUND)/SIGMOID_RESOLUTION;
  if (index_in_table > 1023 || index_in_table<0){
    printf("num %f sigmoid_bound %f sigmoid_res %f\n", num, SIGMOID_BOUND, SIGMOID_RESOLUTION);
  }
  return table[index_in_table];
}
