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
