#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <cstring>
#include "rand_helper.h"
#include "csr.h"
#include "debug.h"

#define NUM_POOLS 2

template <typename VID_T, typename E_T>
class random_walk_training{
  public:
  random_walk_training(int _sample_pool_size, int _shuffle_base, int _sampling_threads, int _deviceID, int negative_samples): max_sample_pool_size(_sample_pool_size), shuffle_base(_shuffle_base), sampling_threads(_sampling_threads), deviceID(_deviceID), negative_samples(negative_samples){
    
    cudaSetDevice(deviceID);
    positive_sample_pools = new VID_T*[NUM_POOLS];
    negative_sample_pools = new VID_T*[NUM_POOLS];
    d_positive_sample_pools = new VID_T*[NUM_POOLS];
    d_negative_sample_pools = new VID_T*[NUM_POOLS];
    for (int i =0; i<NUM_POOLS; i++){
      positive_sample_pools[i] = new VID_T[max_sample_pool_size*2];
      CUDA_CHECK(cudaHostRegister(positive_sample_pools[i], max_sample_pool_size*2*sizeof(VID_T), cudaHostRegisterPortable));

      CUDA_CHECK(cudaMalloc((void**)&(d_positive_sample_pools[i]), sizeof(VID_T)*max_sample_pool_size*2)); 

      negative_sample_pools[i] = new VID_T[max_sample_pool_size*negative_samples];
      CUDA_CHECK(cudaHostRegister(negative_sample_pools[i], max_sample_pool_size*negative_samples*sizeof(VID_T), cudaHostRegisterPortable));

      CUDA_CHECK(cudaMalloc((void**)&(d_negative_sample_pools[i]), sizeof(VID_T)*max_sample_pool_size*negative_samples)); 

      device_pool_full[i] = 0;
      host_pool_full[i] = 0;
    }
    max_sample_pool_size = max_sample_pool_size/(shuffle_base*sampling_threads)*(shuffle_base*sampling_threads);
    max_samples_per_section = max_sample_pool_size/shuffle_base;
    max_samples_per_thread = max_sample_pool_size/sampling_threads;
    max_samples_per_segment = max_samples_per_section/sampling_threads;
    private_sample_pools = new VID_T*[sampling_threads];
    for (int i =0; i<sampling_threads; i++){
      private_sample_pools[i] = new VID_T[max_samples_per_thread*2];
    }
    sampling_stream = new cudaStream_t;
    kernel_stream = new cudaStream_t;
    CUDA_CHECK(cudaStreamCreateWithFlags(sampling_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(kernel_stream, cudaStreamNonBlocking));
    TM_PRINTF(true, "max_sample_pool_size %lu  max_samples_per_section %lu max_samples_per_segment %lu max_samples_per_thread %lu\n", max_sample_pool_size, max_samples_per_section, max_samples_per_segment, max_samples_per_thread);
  }

  template <typename KernelCallerLambda>
  void train_num_samples(size_t num_samples, int lrd_strategy, float starting_learning_rate, int walk_length, int augmentation_distance, CSR<VID_T>* graph, KernelCallerLambda kernel){
    //float learning_rate_ec;
    size_t sample_pool_size = min(max_sample_pool_size, num_samples);
    sample_pool_size = sample_pool_size/(shuffle_base*sampling_threads)*(shuffle_base*sampling_threads);
    size_t samples_per_section = sample_pool_size/shuffle_base;
    size_t samples_per_thread = sample_pool_size/sampling_threads;
    size_t samples_per_segment = samples_per_section/sampling_threads;
    size_t num_rounds = ((float)(num_samples/sample_pool_size));
    //size_t num_rounds = num_samples/sample_pool_size;
    for (int p =0; p < NUM_POOLS; p++){
      host_pool_full[p] = 0;
      device_pool_full[p] = 0;
    }
    omp_set_nested(1);
    #pragma omp parallel num_threads(4)
    {
      #pragma omp single
      {
        #pragma omp task
        { 
          TM_PRINTF(true, "COPIER - start\n");
          copier_task(num_rounds, sample_pool_size);
        }
        #pragma omp task
        {
          TM_PRINTF(true, "SAMPLER - start\n");
          sampler_task(num_rounds, sample_pool_size, graph, walk_length, augmentation_distance, samples_per_section, samples_per_segment, samples_per_thread);
        }
        #pragma omp task
        {
          TM_PRINTF(true, "KERNEL - start\n");
          kernel_dispatched_task(num_rounds, num_samples, sample_pool_size, graph, lrd_strategy, starting_learning_rate, kernel);
        }
      }
    }
    cudaStreamSynchronize(*kernel_stream);
    cudaStreamSynchronize(*sampling_stream);
    cudaDeviceSynchronize();
  }
private:
  int deviceID;
  int negative_samples;
  VID_T **positive_sample_pools, **negative_sample_pools;
  VID_T **d_positive_sample_pools, **d_negative_sample_pools;
  int sampling_threads;
  int device_pool_full[NUM_POOLS], host_pool_full[NUM_POOLS];
  mutex device_pool_mutex[NUM_POOLS], host_pool_mutex[NUM_POOLS];
  condition_variable device_pool_condition_variable_full[NUM_POOLS], device_pool_condition_variable_empty[NUM_POOLS], host_pool_condition_variable_full[NUM_POOLS], host_pool_condition_variable_empty[NUM_POOLS];
  const int shuffle_base = 5;
  size_t max_samples_per_section;
  size_t max_samples_per_thread;
  size_t max_samples_per_segment;
  size_t max_sample_pool_size;
  VID_T** private_sample_pools;
  struct call_back_pool_id {
    int selected_pool;
#ifdef _DEBUG
    int i;
#endif
    random_walk_training<VID_T, E_T>* us;
  };

  cudaStream_t *sampling_stream, *kernel_stream;
  void enqueue(VID_T* queue, int max_queue_size, char& front, char &rear, char &size, unsigned int val){
    if (front == -1){
      front = rear = 0;
      queue[rear ] = val;
    } else {
      rear++;
      rear%=max_queue_size;
      queue[rear] = val;
    }
    size++;
  }

  unsigned int dequeue(VID_T* queue, int max_queue_size, char &front, char &rear, char &size){
    if (size == 0) return UINT_MAX;
    int old_front = front;
    front++;
    front %=max_queue_size;
    size--;
    return queue[old_front];
  }

  void sample_into_pool(VID_T* sample_array, int max_queue_size, int shuffle_base, long long samples_per_thread, long long samples_per_section, int samples_per_segment, int tid, int walk_length, int augmentation_distance, VID_T * V, VID_T* A, VID_T* alias_v, char* alias_p, unsigned long long num_vertices, seed* sd){
    VID_T* private_sample_pool = private_sample_pools[(tid)];
    long long current=0;
    VID_T source, p_sample;
    VID_T * walk_queue = new VID_T[max_queue_size];
    char rear, front, size;
    unsigned long r1 = 10, r2 = 10;
    long long segment_idx, idx_in_segment;
    while (true){
      size = 0;
      rear = -1; front = -1;
      r1 = randn((void*)sd);
      r2 = randn((void*)sd);
      p_sample = sample_node_alias(alias_v, alias_p, num_vertices, r1, r2);
      enqueue(walk_queue, max_queue_size, front, rear, size, p_sample);
      p_sample = get_neighbor(V, A, p_sample, r1);
      if (p_sample != UINT_MAX){
        enqueue(walk_queue, max_queue_size, front, rear, size, p_sample);
      } else {
        continue;
      }
      for (int i =1; size>1;i++){
        if (i<augmentation_distance){
          r1 = randn(sd);
          p_sample = get_neighbor(V, A, p_sample, r1);
          if (p_sample != UINT_MAX){
            enqueue(walk_queue, max_queue_size, front, rear, size, p_sample);
          } else {
            i = walk_length;
          }
        } else {
          source = dequeue(walk_queue, max_queue_size, front, rear, size);
          if (source == UINT_MAX){
            break;
          }
          for (int k =0; k<size && k<augmentation_distance;k++){
            p_sample = walk_queue[(front+k)%max_queue_size];
            segment_idx = (current)%shuffle_base; 
            idx_in_segment = (current++)/shuffle_base;
            private_sample_pool[(segment_idx*samples_per_segment+idx_in_segment)*2] = source;
            private_sample_pool[(segment_idx*samples_per_segment+idx_in_segment)*2+1] = p_sample;
            if (current == samples_per_thread){
              for (long long b =0; b<shuffle_base; b++){
                memcpy(sample_array+b*samples_per_section*2+tid*samples_per_segment*2, private_sample_pool+samples_per_segment*b*2, 2*samples_per_segment*sizeof(VID_T)); 
              }
              return;
            }
          }
          if (i < walk_length){
            r1 = randn(sd); 
            p_sample = get_neighbor(V, A, p_sample, r1);
            if (p_sample != UINT_MAX) {
              enqueue(walk_queue, max_queue_size, front, rear, size, p_sample);
            } else {
              i = walk_length;
            }
          }
        }
      }
    }
    delete [] walk_queue;
  }
  void sampler_task(size_t num_rounds, size_t sample_pool_size, CSR<VID_T>* csr, int walk_length, int augmentation_distance, size_t samples_per_section, size_t samples_per_segment, size_t samples_per_thread){
    cudaSetDevice(deviceID);
    TM_PRINTF(true, "sample_pool_size %lu  samples_per_section %lu samples_per_segment %lu samples_per_thread %lu\n", sample_pool_size, samples_per_section, samples_per_segment, samples_per_thread);
    printf("Execution %ld rounds\n", num_rounds);
    for (int i =0; i< num_rounds;i++){
      int selected_pool = i%NUM_POOLS;
       TM_PRINTF(true, "SAMPLER %d: No locks\nt", i);
      unique_lock<mutex> lock(host_pool_mutex[selected_pool]);
      while (host_pool_full[selected_pool] != 0){
        host_pool_condition_variable_empty[selected_pool].wait(lock);
      }
       TM_PRINTF(true, "SAMPLER %d: locked host_pool_mutex[%d] since host_pool_full[%d]==0\n", i, selected_pool, selected_pool);
      // sampling done here
#pragma omp parallel num_threads(sampling_threads)
      {
#pragma omp single
        {
          for (int y =0 ;y<sampling_threads; y++){
#pragma omp task
            {
              // The sample pool will contain sections, each section is going to contain *num_threads* segments, each for a thread. Threads will split their samples across segments
              // such that each section will contain a bunch of samples from a thread
              seed mySeed;
              mySeed.x = (13*y)+(i*23)+123456789; // the values of the seed are a function of the thread id
              mySeed.y = (13*y)+(i*23)+362436069; // (i wasn't very creative with these functions but I didn't see any decrease in ML performance)
              mySeed.z = (13*y)+(i*23)+521288629;
              //int starting = (y)*(sample_pool_size/omp_get_max_threads());
              //int ending =(y == num_threads-1) ? sample_pool_size :  (y+1)*(sample_pool_size/num_threads);
              sample_into_pool(positive_sample_pools[selected_pool], 6, shuffle_base, samples_per_thread, samples_per_section, samples_per_segment, y, walk_length, augmentation_distance, csr->V, csr->E, csr->v_alias.alias_v, csr->v_alias.alias_p, csr->num_vertices, &mySeed);
            }
          }
        }
      }
#ifdef _DEBUG
              for (int t =0; t<sample_pool_size*2; t++) if (positive_sample_pools[selected_pool][t] <0 ||positive_sample_pools[selected_pool][t] >=csr->num_vertices) TM_PRINTF(true, "BAD SAMPLE BATCH AT %d\n",i); 
#endif
      #pragma omp parallel
      { 
        unsigned int tid = omp_get_thread_num();
        seed mySeed;
        mySeed.x = (13*i)+(23*tid)+123456789; // the values of the seed are a function of the thread id
        mySeed.y = (13*i)+(23*tid)+362436069; // (i wasn't very creative with these functions but I didn't see any decrease in ML performance)
        mySeed.z = (13*i)+(23*tid)+521288629;
        #pragma omp for
        for(unsigned int ff = 0; ff < sample_pool_size*3; ff++){
          negative_sample_pools[selected_pool][ff] = randn((void*)&mySeed)%csr->num_vertices;//;sample_node_alias(csr->ns_alias.alias_v, csr->ns_alias.alias_p, csr->num_vertices, r1, r2);
        }
      }
      host_pool_full[selected_pool] = 1;
      host_pool_condition_variable_full[selected_pool].notify_all();
      TM_PRINTF(true, "SAMPLER %d: set host_pool_full[%d]=1 and released host_pool_mutex[%d]\n", i, selected_pool, selected_pool);
    }
  }


  static void CUDART_CB sample_pool_copied(cudaStream_t event, cudaError_t status,void * data){
    int selected_pool = ((call_back_pool_id*)data)->selected_pool;
#ifdef _DEBUG
    int i = ((call_back_pool_id*)data)->i;
#endif
    random_walk_training<VID_T, E_T>* us = ((call_back_pool_id*)data)->us;
    TM_PRINTF(true, "COPIED - CB %d: No locks\n", i);
    unique_lock<mutex> host_lock(us->host_pool_mutex[selected_pool]);
    TM_PRINTF(true, "COPIED - CB %d: locked host_pool_mutex[%d]\n", i, selected_pool);
    unique_lock<mutex> device_lock(us->device_pool_mutex[selected_pool]);
    TM_PRINTF(true, "COPIED - CB %d: locked device_pool_mutex[%d]\n", i, selected_pool);
    us->host_pool_full[selected_pool] = 0;
    us->device_pool_full[selected_pool] = 2;
    TM_PRINTF(true, "COPIED - CB %d: set host_pool_full[%d] = 0 and device_pool_full[%d] = 2\n", i, selected_pool, selected_pool);
    us->host_pool_condition_variable_empty[selected_pool].notify_all();
    us->device_pool_condition_variable_full[selected_pool].notify_all();
    TM_PRINTF(true, "COPIED - CB %d: released device_pool_mutex[%d] and host_pool_mutex[%d]\n", i, selected_pool, selected_pool);
  }


  void copier_task(size_t num_rounds, size_t sample_pool_size){
    cudaSetDevice(deviceID);
    for (int i = 0; i < num_rounds; i++){
      TM_PRINTF(true, "COPIER %d: No locks\n", i);
      int selected_pool = i%NUM_POOLS;
      unique_lock<mutex> host_lock(host_pool_mutex[selected_pool]);
      while (host_pool_full[selected_pool] != 1){
        host_pool_condition_variable_full[selected_pool].wait(host_lock);
      }
      host_pool_full[selected_pool] = 2;
      TM_PRINTF(true, "COPIER %d: locked host_pool_mutex[%d] since host_pool_full[%d]==1\n", i, selected_pool, selected_pool);
      TM_PRINTF(true, "COPIER %d: set host_pool_full[%d]=2\n", i, selected_pool);
      unique_lock<mutex> device_lock(device_pool_mutex[selected_pool]);
      while (device_pool_full[selected_pool] != 0){
        device_pool_condition_variable_empty[selected_pool].wait(device_lock);
      }
      TM_PRINTF(true, "COPIER %d: locked device_pool_mutex[%d] since device_pool_full[%d]==0\n", i, selected_pool, selected_pool);
      device_pool_full[selected_pool] = 1;
      TM_PRINTF(true, "COPIER %d: set device_pool_full[%d]=1\n", i, selected_pool);
      CUDA_CHECK(cudaMemcpyAsync(d_positive_sample_pools[selected_pool], positive_sample_pools[selected_pool], sample_pool_size*2*sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
      CUDA_CHECK(cudaMemcpyAsync(d_negative_sample_pools[selected_pool], negative_sample_pools[selected_pool], sample_pool_size*negative_samples*sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
      call_back_pool_id *payload = new call_back_pool_id; 
      #ifdef _DEBUG
      payload->i = i; 
      #endif
      payload->selected_pool = selected_pool; payload->us = this;
      CUDA_CHECK(cudaStreamAddCallback(*sampling_stream, sample_pool_copied, (void*)payload, 0));
      TM_PRINTF(true, "COPIER %d: released device_pool_mutex[%d] and host_pool_mutex[%d]\n", i, selected_pool, selected_pool);
      
    }
  }

  static void CUDART_CB sample_pool_used(cudaStream_t event, cudaError_t status,void * data){
    int selected_pool = ((call_back_pool_id*)data)->selected_pool;
    #ifdef _DEBUG
    int i = ((call_back_pool_id*)data)->i;
    #endif
     TM_PRINTF(true, "KERNEL - CB %d: No locks \n", i);
    random_walk_training<VID_T, E_T>* us = ((call_back_pool_id*)data)->us;
    unique_lock<mutex> device_lock(us->device_pool_mutex[selected_pool]);
     TM_PRINTF(true, "KERNEL - CB %d: locked device_pool_mutex[%d]\n", i, selected_pool);
    us->device_pool_full[selected_pool] = 0;
     TM_PRINTF(true, "KERNEL - CB %d: set device_pool_full[%d] = 0\n", i, selected_pool);
    us->device_pool_condition_variable_empty[selected_pool].notify_all();
     TM_PRINTF(true, "KERNEL - CB %d: released device_pool_mutex[%d]\n", i, selected_pool);
  }
  template <typename KernelCallerLambda>
  void kernel_dispatched_task(size_t num_rounds, unsigned long long num_samples, size_t sample_pool_size, CSR<VID_T> * csr, int lrd_strategy, float starting_learning_rate, KernelCallerLambda kernel){
    cudaSetDevice(deviceID);
    float learning_rate_ec;
    for (int i =0; i<num_rounds; i++){
      if (lrd_strategy<2){
        learning_rate_ec = max(float(1-float(i)*1.0/(num_samples/sample_pool_size)), 1e-4f)*(starting_learning_rate);
      } else {
        learning_rate_ec = starting_learning_rate;
      }
      int selected_pool = i%NUM_POOLS;
      TM_PRINTF(true, "KERNEL %d: no locks\n", i);
      unique_lock<mutex> device_lock(device_pool_mutex[selected_pool]);
      while (device_pool_full[selected_pool] != 2){
        device_pool_condition_variable_full[selected_pool].wait(device_lock);
      }
      device_pool_full[selected_pool] = 3;
      TM_PRINTF(true, "KERNEL %d: locked device_pool_mutex[%d] since device_pool_full[%d] == 2\n", i, selected_pool, selected_pool);
      TM_PRINTF(true, "KERNEL %d: set device_pool_full[%d] = 3\n", i, selected_pool);
      kernel(csr, d_positive_sample_pools[selected_pool], d_negative_sample_pools[selected_pool], learning_rate_ec, sample_pool_size, kernel_stream);
      call_back_pool_id *payload = new call_back_pool_id; payload->selected_pool = selected_pool; 
      #ifdef _DEBUG
      payload->i = i; 
      #endif
      payload->us = this;
      CUDA_CHECK(cudaStreamAddCallback(*kernel_stream, sample_pool_used, (void*)payload, 0));
      TM_PRINTF(true, "KERNEL %d: released device_pool_mutex[%d]\n", i, selected_pool);

    }
  }
};

 template class random_walk_training<int, int>;
