#ifndef CUDA_FUNCS
#define CUDA_FUNCS
#include <cuda_runtime.h>
#include "type_def.h"
#include "cuda_utils.h"
#define NUM_THREADS 512
#define NUM_BLOCKS 8192

__global__ void Embedding_Kernel(unsigned int d_num_vertices, int d_num_epoch, unsigned int * d_V, unsigned int * d_A,  emb_t * d_embeddings, int d_dim, int d_s, double d_lr, float* d_sigmoid_lookup_table, int ep_start, int total_batches, int alpha, double negative_weight, int WARP_SIZE, int WARPS_PER_BLOCK, int NUM_WARPS);
__global__ void Big_Graphs_Embedding_Kernel(emb_t *source_bin, emb_t* dest_bin, long long vertices_per_part, int num_vertices, int starting_ep, int batch_ep,vid_t* vids, double d_lr, int dim, int neg_s, float* sig_table, int alpha, int source_part_id, int dest_part_id, int WARP_SIZE, int WARPS_PER_BLOCK, int NUM_WARPS);
__global__ void Embedding_Kernel_SP(unsigned int d_num_vertices, unsigned long samples_per_pool, unsigned int* d_sample_array, unsigned int* d_fake,  emb_t * d_embeddings, float d_lr, int dimension, int negative_samples, float negative_weight, int WARPS_PER_BLOCK, int WARP_SIZE, int NUM_WARPS);
#endif
