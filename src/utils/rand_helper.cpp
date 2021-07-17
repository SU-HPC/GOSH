#include "rand_helper.h"
__host__ __device__ unsigned long randn(void* s) { //xorshf96          //period 2^96-1
  unsigned long t;
  seed* myxyz= (seed*)s;
  myxyz->x ^= myxyz->x << 16;
  myxyz->x ^= myxyz->x >> 5;
  myxyz->x ^= myxyz->x << 1;

  t = myxyz->x;
  myxyz->x = myxyz->y;
  myxyz->y = myxyz->z;
  myxyz->z = t ^ myxyz->x ^ myxyz->y;

  return myxyz->z;
}

template <typename VID>
__device__ __host__ unsigned int sample_node_alias(VID * alias_v, char * alias_p, unsigned int size, unsigned int r1, unsigned long r2){

  unsigned int index = r1 % size; 
  // printf("index %d x %d y %d\n", index, x,y);
  long long res = r2%101 < alias_p[index] ? index : alias_v[index];
  //if (res <0 || res>=size){
    //throw -1;
    //printf("bad index in sample size %d sample %d %lld alias_p[index] %d alias_v[index] %d\n", size, index, res, alias_p[index], alias_v[index]);
  //}
  return res;
}
template <typename VID, typename E_T>
__device__ __host__ unsigned int get_neighbor(VID * d_V, E_T * d_A, VID v, unsigned long rn){
  int num_neighbors = (d_V[v+1]-d_V[v]);
  if (num_neighbors <1){
    return UINT_MAX;
  } else {
    return d_A[d_V[v]+(rn%num_neighbors)];
  }
}
template unsigned int get_neighbor<int, int>(int*, int*, int, unsigned long);
template unsigned int get_neighbor(unsigned int*, unsigned int*, unsigned int, unsigned long);
template unsigned int sample_node_alias(int*, char*, unsigned int, unsigned int, unsigned long);
template unsigned int sample_node_alias(unsigned int*, char*, unsigned int, unsigned int, unsigned long);

