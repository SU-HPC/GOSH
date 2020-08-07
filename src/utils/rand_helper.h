#ifndef _RANDOM_HELPER
#define _RANDOM_HELPER
struct seed{ 
  unsigned long x,y,z; 
};
__host__ __device__ unsigned long randn(void* s); //xorshf96          //period 2^96-1
#endif
