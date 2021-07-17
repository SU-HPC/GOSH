#ifndef _RANDOM_HELPER
#define _RANDOM_HELPER
struct seed{ 
  unsigned long x,y,z; 
};
__host__ __device__ unsigned long randn(void* s); //xorshf96          //period 2^96-1
template <typename VID>
__device__ __host__ unsigned int sample_node_alias(VID * alias_v, char * alias_p, unsigned int size, unsigned int r1, unsigned long r2);
template <typename VID, typename E_T>
__device__ __host__ unsigned int get_neighbor(VID * d_V, E_T * d_A, VID v, unsigned long rn);

#endif
