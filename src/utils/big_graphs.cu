#include "big_graphs.h"
#include <algorithm>
#include <unordered_set>
#include <set>
#include <atomic>
#include "cuda_profiler_api.h"
//#define _DEBUG
#define MEASURE_TIME
#include "debug.h"
#define PARTS_HARD_LIMIT 20
Node::Node(int id):id(id){
  type = node;
}


void TerminationNode::execute(){
  cudaDeviceSynchronize();
}

bool BigGraphs::use_big_graphs(CSR<vid_t>* a_csr, int emb_dimensions, int device_id){
  size_t gpu_free_memory, gpu_total_memory;
  cudaSetDevice(device_id);
  CUDA_CHECK(cudaMemGetInfo(&gpu_free_memory,&gpu_total_memory));
  unsigned long long size_of_embeddings = a_csr->num_vertices*emb_dimensions*sizeof(emb_t);
  unsigned long long size_of_csr = (a_csr->num_vertices+1+a_csr->num_edges)*sizeof(vid_t);
  if (size_of_embeddings+size_of_csr>gpu_free_memory){
    //printf("Cannot fit the CSR on the device; need to use pools\n");
    //printf("Size of embeddings %lld MB; Size of CSR %lld MB; Free memory on GPU %ld MB\n",size_of_embeddings/1000000,size_of_csr/1000000, gpu_free_memory/1000000);
    return true;
  }
  return false;
}

void BigGraphs::sample_task(int starting_epoch, int round, int pool_set, int main, int sub){
  TM_PRINTF(true, "ST: START :: [%d, %d] pool_set %d round %d\n", main, sub, pool_set, round); 
  unique_lock<mutex> pool_lock(pool_mutex[pool_set][main][sub]);
  while (pool_status[pool_set][main][sub] !=-1){
    pool_empty_cv[pool_set][main][sub].wait(pool_lock);
  }
  TM_PRINTF(true, "ST: LOCKED POOL :: [%d, %d] pool_set %d round %d\n", main, sub, pool_set, round); 
  double _s = omp_get_wtime();
  sample_to_pool(main, sub, h_sample_pools[pool_set][main][sub], starting_epoch, epoch_batch_size);
  TM_PRINTF(true, "ST: DONE:: [%d, %d] pool_set %d round %d\n", main, sub, pool_set, round); 
  double _e = omp_get_wtime();
  pool_status[pool_set][main][sub] = 0;
  pool_full_cv[pool_set][main][sub].notify_all();
}

void EmbeddingNode::set_device(){
  cudaSetDevice(device_id);
}

EmbeddingNode::EmbeddingNode(BigGraphs* bg_o, cudaStream_t * _stream, int _device_id, int id):Node(id), bg(bg_o), stream(_stream), device_id(_device_id){
  cudaSetDevice(device_id);
  this->type = embedding_node;
  record_event = shared_ptr<cudaEvent_t>(new cudaEvent_t, [=](cudaEvent_t* event){
      CUDA_CHECK(cudaEventDestroy(*event));
      delete event;
      });
  CUDA_CHECK(cudaEventCreateWithFlags(record_event.get(),  cudaEventDisableTiming));
}

void EmbeddingNode::record_event_of_node(){
}

void EmbeddingNode::wait_for_events(){
  for (auto event : wait_events){
    CUDA_CHECK(cudaStreamWaitEvent(*stream, *event, 0));
  }
}

KernelNode::KernelNode(BigGraphs* bg_o, cudaStream_t* _stream, int _main_idx, int _sub_idx, int _gpu_part_main, int _gpu_part_sub, int _starting_epoch, int _round, int _device_id, int id): EmbeddingNode(bg_o,_stream, _device_id, id), main_idx(_main_idx), sub_idx(_sub_idx), gpu_part_main(_gpu_part_main), gpu_part_sub(_gpu_part_sub), starting_epoch(_starting_epoch), round(_round){
  this->type  = kernel_node;
}

void KernelNode::execute(){
  set_device();
  int selected_gpu_pool;
  TM_PRINTF(true, "K: START :: [%d, %d]->[%d, %d] round %d id %d\n", main_idx, sub_idx, gpu_part_main, gpu_part_sub, round, id); 
  double s = omp_get_wtime();
  {
    unique_lock<mutex> kernel_lock(bg->kernel_pool_mutex[main_idx][sub_idx]);
    while (bg->kernel_pool_index[main_idx][sub_idx]==-1 || (bg->gpu_pool_status[bg->kernel_pool_index[main_idx][sub_idx]] != 1)){
      bg->kernel_pool_full_cv[main_idx][sub_idx].wait(kernel_lock);
    }
    selected_gpu_pool = bg->kernel_pool_index[main_idx][sub_idx];
  TM_PRINTF(true, "K: LOCKED KERNEL POOL ON %d [%d, %d]->[%d, %d] round %d id %d\n",selected_gpu_pool, main_idx, sub_idx, gpu_part_main, gpu_part_sub, round, id); 
  }
  wait_for_events();
  double e = omp_get_wtime();
  for (double g = bg->time_spent_waiting_kernel ;(e-s != 0) && !bg->time_spent_waiting_kernel.compare_exchange_weak(g, g + (e-s)););
  Big_Graphs_Embedding_Kernel<<<NUM_BLOCKS,NUM_THREADS,bg->size_of_shared_array,*stream>>>(bg->d_embedding_parts[gpu_part_main], bg->d_embedding_parts[gpu_part_sub], bg->vertices_per_part, bg->csr->num_vertices, starting_epoch,       bg->epoch_batch_size,    bg->d_sample_pools[selected_gpu_pool],       bg->learning_rate, bg->dimension, bg->negative_samples,       bg->d_sigmoid_lookup_table,        bg->alpha, main_idx, sub_idx, bg->WARP_SIZE, bg->WARPS_PER_BLOCK, bg->NUM_WARPS);
  TM_PRINTF(true, "K: DISPATCHED KERNEL ON %d [%d, %d]->[%d, %d] round %d id %d\n",selected_gpu_pool, main_idx, sub_idx, gpu_part_main, gpu_part_sub, round, id); 
  kernel_callback_data * kcd = new kernel_callback_data;
  kcd->bg = bg;
  kcd->main_idx = main_idx;
  kcd->sub_idx = sub_idx;
  kcd->selected_gpu_pool = selected_gpu_pool;
  kcd->round = round;
  CUDA_CHECK(cudaEventRecord(*record_event.get(), *stream));
  TM_PRINTF(true, "K: WATED FOR EVENTS ON %d [%d, %d]->[%d, %d] round %d id %d\n",selected_gpu_pool, main_idx, sub_idx, gpu_part_main, gpu_part_sub, round, id); 
  CUDA_CHECK(cudaLaunchHostFunc(*stream, KernelNode::kernel_finished_callback, (void*)kcd));
  TM_PRINTF(true, "K: ADDED CALLBACK ON %d [%d, %d]->[%d, %d] round %d id %d\n",selected_gpu_pool, main_idx, sub_idx, gpu_part_main, gpu_part_sub, round, id); 
}

void CUDART_CB KernelNode::kernel_finished_callback(void* data){
  kernel_callback_data* kcd = (kernel_callback_data*) data;
  int main_idx = kcd->main_idx;
  int sub_idx = kcd->sub_idx;
  BigGraphs* bg = kcd->bg;
  int selected_gpu_pool = kcd->selected_gpu_pool;
  #ifdef _DEBUG
    int round = kcd->round;
  #endif
  delete kcd;
  TM_PRINTF(true, "KC: START:: ON %d [%d, %d] round %d \n",selected_gpu_pool, main_idx, sub_idx, round); 
  unique_lock<mutex> gpu_pool_lock(bg->gpu_pool_mutex[selected_gpu_pool]);
  TM_PRINTF(true, "KC: LOCKED GPU POOL:: ON %d [%d, %d] round %d \n",selected_gpu_pool, main_idx, sub_idx, round); 
  unique_lock<mutex> kernel_lock(bg->kernel_pool_mutex[main_idx][sub_idx]);
  TM_PRINTF(true, "KC: LOCKED GPU POOL:: ON %d [%d, %d] round %d \n",selected_gpu_pool, main_idx, sub_idx, round); 
  bg->gpu_pool_status[selected_gpu_pool] = -1;
  bg->kernel_pool_index[main_idx][sub_idx] = -1;
  bg->kernel_pool_empty_cv[main_idx][sub_idx].notify_all();
}

SampleCopyNode::SampleCopyNode(BigGraphs* bg_o, cudaStream_t* _stream, int m, int s, int r, int _device_id, int id): EmbeddingNode(bg_o, _stream, _device_id, id), main_idx(m), sub_idx(s), pool_set(r){
  this->type = sample_copy_node;
}

void CUDART_CB SampleCopyNode::sample_copy_finished_callback(void* data){
  callback_data * cbd = (callback_data*)data;
  BigGraphs* bg = cbd->bg;
  int main_idx = cbd->main_idx;
  int sub_idx = cbd->sub_idx;
  int gpu_selected_pool = cbd->gpu_selected_pool;
  int pool_set = cbd->pool_set;
  delete cbd;
  TM_PRINTF(true, "SCC: START:: ON %d [%d, %d] pool_set %d \n",gpu_selected_pool, main_idx, sub_idx, pool_set); 
  unique_lock<mutex> gpu_pool_lock(bg->gpu_pool_mutex[gpu_selected_pool]);
  TM_PRINTF(true, "SCC: LOCKED GPU POOL:: ON %d [%d, %d] pool_set %d \n",gpu_selected_pool, main_idx, sub_idx, pool_set); 
  unique_lock<mutex> kernel_pool_lock(bg->kernel_pool_mutex[main_idx][sub_idx]);
  TM_PRINTF(true, "SCC: LOCKED KERNEL POOL:: ON %d [%d, %d] pool_set %d \n",gpu_selected_pool, main_idx, sub_idx, pool_set); 
  unique_lock<mutex> pool_lock(bg->pool_mutex[pool_set][main_idx][sub_idx]);
  TM_PRINTF(true, "SCC: LOCKED POOL:: ON %d [%d, %d] pool_set %d \n",gpu_selected_pool, main_idx, sub_idx, pool_set); 
  bg->pool_status[pool_set][main_idx][sub_idx] = -1;
  bg->gpu_pool_status[gpu_selected_pool] = 1;
  bg->kernel_pool_index[main_idx][sub_idx] = gpu_selected_pool;
  bg->pool_empty_cv[pool_set][main_idx][sub_idx].notify_all();
  bg->kernel_pool_full_cv[main_idx][sub_idx].notify_all();
}

void SampleCopyNode::execute(){
  set_device();
  TM_PRINTF(true, "S: START:: ON [%d, %d] pool_set %d \n",main_idx, sub_idx, pool_set); 
  double start = omp_get_wtime();
  unique_lock<mutex> pool_lock(bg->pool_mutex[pool_set][main_idx][sub_idx]);
  while (bg->pool_status[pool_set][main_idx][sub_idx] != 0){
    bg->pool_full_cv[pool_set][main_idx][sub_idx].wait(pool_lock);
  }
  bg->pool_status[pool_set][main_idx][sub_idx] = 1;
  TM_PRINTF(true, "S: LOCKED POOL:: ON [%d, %d] pool_set %d \n",main_idx, sub_idx, pool_set); 
  double e = omp_get_wtime();
  for (double g =bg->time_spent_waiting_for_sample_pool;(e-start != 0) && !bg->time_spent_waiting_for_sample_pool.compare_exchange_weak(g, g + (e-start)););
  start = omp_get_wtime();
  int selected_gpu_pool=-1;
  while (selected_gpu_pool == -1){
    for (int k =0; k<bg->num_pools_gpu; k++){
      if (bg->gpu_pool_mutex[k].try_lock()){
        if (bg->gpu_pool_status[k]!=-1){
          bg->gpu_pool_mutex[k].unlock();
        } else {
          bg->gpu_pool_status[k] = 0;
          selected_gpu_pool = k;
          break;
        }
      }
    }
  }
  TM_PRINTF(true, "S: LOCKED GPU POOL %d:: ON [%d, %d] pool_set %d \n",selected_gpu_pool, main_idx, sub_idx, pool_set); 
  e = omp_get_wtime();
  for (double g =bg->time_spent_waiting_for_pool_gpus; (e-start != 0) &&!bg->time_spent_waiting_for_pool_gpus.compare_exchange_weak(g, g + (e-start)););
  start = omp_get_wtime();
  unique_lock<mutex> kernel_lock(bg->kernel_pool_mutex[main_idx][sub_idx]);
  while (bg->kernel_pool_index[main_idx][sub_idx] != -1){
    bg->kernel_pool_empty_cv[main_idx][sub_idx].wait(kernel_lock);
  }
  TM_PRINTF(true, "S: LOCKED KERNEL POOL %d:: ON [%d, %d] pool_set %d \n",selected_gpu_pool, main_idx, sub_idx, pool_set); 
  e = omp_get_wtime();
  for (double g =bg->time_spent_in_sc_waiting_for_kernel; (e-start != 0) &&!bg->time_spent_in_sc_waiting_for_kernel.compare_exchange_weak(g, g + (e-start)););
  bg->kernel_pool_index[main_idx][sub_idx] = selected_gpu_pool;
  wait_for_events();
  vid_t* pool = bg->h_sample_pools[pool_set][main_idx][sub_idx];
  unsigned long long copy_amount = sizeof(vid_t)*((pool[0]+pool[1])*2+2);
  if (bg->h_sample_pools[pool_set][main_idx][sub_idx][0] == bg->h_sample_pools[pool_set][main_idx][sub_idx][0] && bg->h_sample_pools[pool_set][main_idx][sub_idx][0] == 0){
    CUDA_CHECK(cudaMemcpyAsync(bg->d_sample_pools[selected_gpu_pool], bg->h_sample_pools[pool_set][main_idx][sub_idx],(size_t)2*sizeof(vid_t), cudaMemcpyHostToDevice, *stream));
  }else {
    CUDA_CHECK(cudaMemcpyAsync(bg->d_sample_pools[selected_gpu_pool], bg->h_sample_pools[pool_set][main_idx][sub_idx],(size_t)copy_amount, cudaMemcpyHostToDevice, *stream));
  }
  TM_PRINTF(true, "S: DISPATCHED COPY %d :: ON [%d, %d] pool_set %d \n",selected_gpu_pool, main_idx, sub_idx, pool_set); 
  callback_data *payload = new callback_data;
  payload->main_idx = main_idx;
  payload->sub_idx = sub_idx;
  payload->gpu_selected_pool = selected_gpu_pool;
  payload->bg = bg;
  payload->pool_set = pool_set;
  CUDA_CHECK(cudaEventRecord(*record_event.get(), *stream));
  TM_PRINTF(true, "S: RECORDED EVENT %d :: ON [%d, %d] pool_set %d \n",selected_gpu_pool, main_idx, sub_idx, pool_set); 
  CUDA_CHECK(cudaLaunchHostFunc(*stream, SampleCopyNode::sample_copy_finished_callback, (void*)payload));
  TM_PRINTF(true, "S: ADDED CALLBACK %d :: ON [%d, %d] pool_set %d \n",selected_gpu_pool, main_idx, sub_idx, pool_set); 
  bg->gpu_pool_mutex[selected_gpu_pool].unlock();
}


PartSwitchNode::PartSwitchNode(BigGraphs* bg_o, cudaStream_t* _stream, int in, int out, int gpu_index, int _device_id, int id): EmbeddingNode(bg_o, _stream, _device_id, id), in_part(in), out_part(out), gpu_part_index(gpu_index){
  this->type = part_switch_node;
}

unsigned long long embedding_bytes_to_copy(BigGraphs * bg, int part){
  unsigned long long num_vertices;
  if (part == bg->num_parts-1){
    num_vertices = bg->csr->num_vertices-((bg->num_parts-1)*bg->vertices_per_part);
  } else {
    num_vertices = bg->vertices_per_part;
  }
  return num_vertices*(unsigned long long)bg->dimension*sizeof(emb_t);
}

void PartSwitchNode::execute(){
  TM_PRINTF(true, "PS: START:: (%d, %d, %d) \n", in_part, out_part, gpu_part_index); 
  set_device();
  wait_for_events();
  if (out_part != -1){
    unsigned long long bytes_to_copy = embedding_bytes_to_copy(bg, out_part);
    CUDA_CHECK(cudaMemcpyAsync(&(bg->h_embeddings[bg->dimension*out_part * bg->vertices_per_part]), bg->d_embedding_parts[gpu_part_index], bytes_to_copy, cudaMemcpyDeviceToHost, *stream));
  TM_PRINTF(true, "PS: DISPATCHED OUT:: (%d, %d, %d) \n", in_part, out_part, gpu_part_index); 
  }
  if (in_part != -1){
    unsigned long long bytes_to_copy = embedding_bytes_to_copy(bg, in_part);
    CUDA_CHECK(cudaMemcpyAsync(bg->d_embedding_parts[gpu_part_index], &(bg->h_embeddings[bg->dimension*in_part*bg->vertices_per_part]), bytes_to_copy, cudaMemcpyHostToDevice, *stream));
  TM_PRINTF(true, "PS: DISPATCHED IN:: (%d, %d, %d) \n", in_part, out_part, gpu_part_index); 
  }
  CUDA_CHECK(cudaEventRecord(*record_event.get(), *stream));
  TM_PRINTF(true, "PS: WAITED EVENT:: (%d, %d, %d) \n", in_part, out_part, gpu_part_index); 
}

void Node::add_edge(shared_ptr<Node>& edge){
  unique_lock<mutex> node_lock(self_mutex);
  outgoing_edges.push_back(edge);
  edge->increment_incoming_edge_counter();
}
void Node::add_weak_edge(shared_ptr<Node>& edge){
  unique_lock<mutex> node_lock(self_mutex);
  weak_outgoing_edges.push_back(edge);
  edge->increment_incoming_edge_counter();
}

void Node::increment_incoming_edge_counter(){
  unique_lock<mutex> node_lock(self_mutex);
  incoming_edge_counter++;
}

void Node::execute(){
  printf("Default node execution - you're doing something wrong\n");
}

node_type Node::get_type(){
  return type;
}

void TaskQueue::execute_node(shared_ptr<Node> node){
  cudaSetDevice(device_id);
  node->execute();
  for (auto edge : node->outgoing_edges){
    unique_lock<mutex> edge_lock(edge->self_mutex);
    if (node->record_event != nullptr){
      edge->wait_events.push_back(node->record_event);
    }
    edge->incoming_edge_counter--;
    if (edge->incoming_edge_counter < 0){
      throw 1;
    }
    if (edge->incoming_edge_counter == 0){ 
      add_to_queue(edge);
    }
  }
  for (auto edge : node->weak_outgoing_edges){
    unique_lock<mutex> edge_lock(edge->self_mutex);
    edge->incoming_edge_counter--;
    if (edge->incoming_edge_counter < 0){
      throw 1;
    }
    if (edge->incoming_edge_counter == 0){ 
      add_to_queue(edge);
    }
  }

  node->outgoing_edges.clear();
}
void TaskQueue::run(){
// thread distributions
// 1 thread to run the "single" region
// local_threads to run the node executions
#pragma omp parallel num_threads(1+local_threads)
  {
    cudaSetDevice(device_id);
#pragma omp single 
    {
      cudaSetDevice(device_id);
      shared_ptr<Node> curr = nullptr;
      bool graph_completed = false;
      while (!graph_completed || node_queue.size()>0){
        {
          unique_lock<mutex> lock(queue_mutex);
          while (node_queue.size() == 0){
            queue_not_empty.wait(lock);
          }
          curr = node_queue.back();
          node_queue.pop_back();
        }
#pragma omp task firstprivate(curr)
        {
          cudaSetDevice(device_id);
          execute_node(curr);
        }
        if (curr->get_type() == termination_node){
          TM_PRINTF(false,"Exit condition met. Ending task queue ...\n");
          graph_completed = true;
        }
      }
      TM_PRINTF(false,"Task queue is done executing.\n");
    }
  }
}

void TaskQueue::add_to_queue(shared_ptr<Node>&& new_node){
  cudaSetDevice(device_id);
  unique_lock<mutex> lock(queue_mutex);
  node_queue.push_front(new_node);
  if (node_queue.size() == 1){
    queue_not_empty.notify_all();
  }
}
void TaskQueue::add_to_queue(shared_ptr<Node>& new_node){
  cudaSetDevice(device_id);
  unique_lock<mutex> lock(queue_mutex);
  node_queue.push_front(new_node);
  if (node_queue.size() == 1){
    queue_not_empty.notify_all();
  }
}


TerminationNode::TerminationNode(int _id): Node(_id){
  this->type = termination_node;
}

BeginningNode::BeginningNode(int id):Node(id){
  this->type = beginning_node;
}
void BeginningNode::execute(){
  printf("Started embedding ...\n");
}

BigGraphs::~BigGraphs(){
  for (int i =0; i<num_parts_gpu;i ++){
    CUDA_CHECK(cudaFree(d_embedding_parts[i]));
  }
  for (int i = 0; i<num_pool_sets; i++){
    for (int j = 0; j<num_parts; j++){
      for (int k = 0; k < j+1; k++){
        CUDA_CHECK(cudaHostUnregister(h_sample_pools[i][j][k]));
        delete [] h_sample_pools[i][j][k];
      }
    }
  }
  for (int i =0; i<num_pools_gpu; i++)
    CUDA_CHECK(cudaFree(d_sample_pools[i]));
  CUDA_CHECK(cudaHostUnregister(h_embeddings));
}

void BigGraphs::initialize_private_members(){
  cudaSetDevice(device_id);
  total_samples = (unsigned long long) csr->num_vertices*(unsigned long long)epochs;
  time_spent_waiting_kernel.store(0);
  time_spent_waiting_for_sample_pool.store(0);
  time_spent_waiting_for_pool_gpus.store(0);
  time_spent_in_sc_waiting_for_kernel.store(0);
  WARP_SIZE = 32;
  if(dimension <= 8){
    WARP_SIZE = 8;
  }
  else if(dimension <= 16){
    WARP_SIZE = 16;
  }
  WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
  NUM_WARPS = WARPS_PER_BLOCK * NUM_BLOCKS;
  size_of_shared_array=dimension*(WARPS_PER_BLOCK)*sizeof(emb_t);
  for (int i =0; i<num_parts_gpu; i++){
    part_streams.push_back(make_shared<cudaStream_t>());
    CUDA_CHECK(cudaStreamCreateWithFlags(part_streams[i].get(), cudaStreamNonBlocking));
  }
  for (int ks = 0; ks < NUM_KERNEL_STREAMS; ks++){
    kernel_streams.push_back(make_shared<cudaStream_t>());
    CUDA_CHECK(cudaStreamCreateWithFlags(kernel_streams[ks].get(),cudaStreamNonBlocking));
  }
  sample_stream = make_shared<cudaStream_t>();
  CUDA_CHECK(cudaStreamCreateWithFlags(sample_stream.get(),cudaStreamNonBlocking));

  calculate_big_graphs_resources(csr->num_vertices, csr->num_edges, num_parts_gpu, num_pools_gpu, dimension, num_parts, vertices_per_part, epoch_batch_size, vids_per_pool);
  ////// allocation
  d_embedding_parts = vector<emb_t*>(num_parts_gpu, nullptr);
  for (int i =0; i<num_parts_gpu; i++){
    CUDA_CHECK(cudaMalloc((void**)&d_embedding_parts[i], vertices_per_part*dimension*sizeof(emb_t)));
  }
  CUDA_CHECK(cudaHostRegister(h_embeddings, csr->num_vertices*dimension*sizeof(emb_t), cudaHostRegisterPortable));


  h_sample_pools = vector<vector<vector<vid_t*>>>(num_pool_sets,vector<vector<vid_t*>>(num_parts));
  for (int i = 0; i<num_pool_sets; i++){
    for (int j = 0; j<num_parts; j++){
      h_sample_pools[i][j] = vector<vid_t*>(j+1);
      for (int k = 0; k < j+1; k++){
        h_sample_pools[i][j][k] = new vid_t[vids_per_pool];
        CUDA_CHECK(cudaHostRegister(h_sample_pools[i][j][k], vids_per_pool*sizeof(vid_t), cudaHostRegisterPortable));
      }
    }
  }

  d_sample_pools = vector<vid_t*>(num_pools_gpu);
  for (int i =0 ;i<num_pools_gpu; i++){
    CUDA_CHECK(cudaMalloc((void**)&d_sample_pools[i], sizeof(vid_t)*vids_per_pool));
  }

  ///// control variables
  terminate_samplers = vector<vector<vector<bool>>>(num_pool_sets,vector<vector<bool>>(num_parts));
  pool_status = vector<vector<vector<int>>>(num_pool_sets,vector<vector<int>>(num_parts));
  pool_empty_cv =vector<vector<vector<condition_variable>>>(num_pool_sets);
  pool_full_cv = vector<vector<vector<condition_variable>>>(num_pool_sets);
  pool_mutex = vector<vector<vector<mutex>>>(num_pool_sets);
  kernel_pool_index = vector<vector<int>>(num_parts);
  kernel_pool_empty_cv = vector<vector<condition_variable>>(num_parts);
  kernel_pool_full_cv = vector<vector<condition_variable>>(num_parts);
  kernel_pool_mutex = vector<vector<mutex>>(num_parts);
  for (int j =0; j<num_parts;j++){
    kernel_pool_index[j] = vector<int>(j+1);
    kernel_pool_empty_cv[j] = vector<condition_variable>(j+1);
    kernel_pool_full_cv[j] = vector<condition_variable>(j+1);
    kernel_pool_mutex[j] = vector<mutex>(j+1);
    for (int k = 0; k < j+1; k++){
      kernel_pool_index[j][k] = -1;
    }
  }
  for (int i = 0; i<num_pool_sets; i++){
    pool_empty_cv[i] = vector<vector<condition_variable>>(num_parts);
    pool_full_cv[i] = vector<vector<condition_variable>>(num_parts);
    pool_mutex[i] = vector<vector<mutex>>(num_parts);
    for (int j = 0; j<num_parts; j++){
      terminate_samplers[i][j] = vector<bool>(j+1);
      pool_status[i][j] = vector<int>(j+1);
      pool_empty_cv[i][j] = vector<condition_variable>(j+1);
      pool_full_cv[i][j] = vector<condition_variable>(j+1);
      pool_mutex[i][j] = vector<mutex>(j+1);
      for (int k = 0; k < j+1; k++){
        terminate_samplers[i][j][k] = false;
        pool_status[i][j][k] = -1;
      }
    }
  }

  gpu_pool_status = vector<int>(num_pools_gpu, -1);
  gpu_pool_mutex = vector<mutex>(num_pools_gpu);
}

void BigGraphs::begin_embedding(){
  cudaSetDevice(device_id);
  num_rounds = max(1ull,epochs/epoch_batch_size);
  omp_set_nested(1);
// thread distributions
// "concurrent_samplers" threads to run the sample_task functions
// 1 thread for section 1
// 1 thread for section 2
#pragma omp parallel num_threads(concurrent_samplers+2)
  {
      cudaSetDevice(device_id);
#pragma omp sections
    {
      cudaSetDevice(device_id);
#pragma omp section
      {
        cudaSetDevice(device_id);
        bool *** dependency_structure = new bool**[num_pool_sets];
        for (int i =0; i<num_pool_sets; i++){
          dependency_structure[i] =new bool*[num_parts];
          for (int j =0; j < num_parts; j++){
            dependency_structure[i][j] = new bool[j+1];
            for (int k = 0; k<j+1; k++) dependency_structure[i][j][k] = 0;
          }
        } 
        for (int i =0;i<num_rounds; i++){
          for (int j =0; j<num_parts; j++){
            for (int k =0; k < j+1; k++){
#pragma omp task firstprivate(i, j, k) depend (inout:dependency_structure[i%num_pool_sets][j][k])
              {
                cudaSetDevice(device_id);
                sample_task(i*epoch_batch_size, i, i%num_pool_sets, j, k); 
              }
            }
          }
        }
        for (int i = 0; i< num_pool_sets; i++){
          for (int j = 0; j < num_parts; j++){
            delete [] dependency_structure[i][j];
          }
          delete [] dependency_structure[i];
        }
        delete [] dependency_structure;
      }
#pragma omp section
      {
#ifdef MEASURE_TIME
        double start = omp_get_wtime();
#endif
        shared_ptr<Node> beginning_node =create_execution_graph();
#ifdef MEASURE_TIME
        double end = omp_get_wtime();
        printf("Generating execution graph took %f seconds\n", end-start);
#endif
        task_queue.add_to_queue(beginning_node);
        beginning_node = nullptr;
        cudaProfilerStart();
#ifdef MEASURE_TIME
        start = omp_get_wtime();
#endif
        task_queue.run();
        for (int j =0; j<num_parts; j++){
          for (int k = 0; k< j+1; k++){
            while (kernel_pool_index[j][k]!=-1);
          }
        }
        for (int i =0; i<num_pools_gpu; i++){
          while (gpu_pool_status[i]!=-1);
        }
#ifdef MEASURE_TIME
        end = omp_get_wtime();
        printf("Embedding took %f seconds\n", end-start);
#endif
        cudaProfilerStop();
      }
    }
  }
  #ifdef MEASURE_TIME
    printf("Time spent at sample copy node waiting for sample pools to fill: %f seconds\n", time_spent_waiting_for_sample_pool.load());
    printf("Time spent at kernel node waiting for samples to be on GPU: %f seconds\n", time_spent_waiting_kernel.load());
    printf("Time spent at sample waiting copy node for gpu_sample_pools to be empty for samples to be moved to them: %f seconds\n", time_spent_waiting_for_pool_gpus.load());
    printf("Time spent at sample waiting copy node for the kernel currently running to finish: %f seconds\n", time_spent_in_sc_waiting_for_kernel.load());
    if (time_finished_samples != -1)
      printf("Time between finishing all the samples and ending embedding: %f seconds\n", omp_get_wtime() - time_finished_samples);
    else {
      printf("Did not run all samples!\n");
    }
  #endif
  printf("Finished the large graph embedding process\n");
  printf("Ran %llu/%llu positive samples\n", ((unsigned long long)epochs * (unsigned long long)csr->num_vertices)-total_samples, ((unsigned long long)epochs * (unsigned long long)csr->num_vertices));
}

BigGraphs::BigGraphs(int _dimension, int _negative_samples, unsigned long long _epochs, double _learning_rate, int _epoch_batch_size, int _alpha, double _negative_weight, int _lrd_strategy, emb_t* _h_embeddings, CSR<vid_t> *_csr, float* _d_sigmoid_lookup_table, int _num_parts_gpu, int _num_pools_gpu, int _concurrent_samplers, int _sampling_threads, int _num_pool_sets, int _task_queue_threads, int _device_id): dimension(_dimension), negative_samples(_negative_samples), epochs(_epochs), learning_rate(_learning_rate), epoch_batch_size(_epoch_batch_size), alpha(_alpha), negative_weight(_negative_weight), lrd_strategy(_lrd_strategy), h_embeddings(_h_embeddings), csr(_csr), d_sigmoid_lookup_table(_d_sigmoid_lookup_table), num_parts_gpu(_num_parts_gpu), num_pools_gpu(_num_pools_gpu), concurrent_samplers(_concurrent_samplers), sampling_threads(_sampling_threads), task_queue(_task_queue_threads, _device_id), num_pool_sets(_num_pool_sets), device_id(_device_id){
#ifdef MEASURE_TIME
  double start = omp_get_wtime();
#endif
  printf("Starting the large graph embedding process ...\n");
  initialize_private_members();
#ifdef MEASURE_TIME
  double end = omp_get_wtime();
  printf("Allocating memory and initializing variables took %f seconds\n", end-start);
#endif
}
void pretty_bytes(char* buf, unsigned long long bytes)
{
    const char* suffixes[7];
    suffixes[0] = "B";
    suffixes[1] = "KB";
    suffixes[2] = "MB";
    suffixes[3] = "GB";
    suffixes[4] = "TB";
    suffixes[5] = "PB";
    suffixes[6] = "EB";
    uint s = 0; // which suffix to use
    double count = bytes;
    while (count >= 1024 && s < 7)
    {
        s++;
        count /= 1024;
    }
    if (count - floor(count) == 0.0)
        sprintf(buf, "%d %s", (int)count, suffixes[s]);
    else
        sprintf(buf, "%.1f %s", count, suffixes[s]);
}
void BigGraphs::print_big_graphs_info(){
  cout << "  Big graphs information:\n";
  cout <<"    Number of embedding parts: " << num_parts << " - Vertices per part: " << vertices_per_part << "\n    Number of sample pools: " << num_pool_sets*(num_parts*(num_parts+1)/2) << " - Vertex IDs per sample pool: " << vids_per_pool << endl;
  cout << "    CPU Memory Usage: \n";
  char buffer[20];
  pretty_bytes(buffer, num_pool_sets*(num_parts*(num_parts+1)/2)*vids_per_pool*sizeof(vid_t));
  cout << "      Sample pools: " << buffer;
  pretty_bytes(buffer, dimension*csr->num_vertices*sizeof(emb_t));
  cout << " - Embedding: " << buffer << endl;
  cout <<"    GPU Memory Usage: \n";
  pretty_bytes(buffer, num_pools_gpu*vids_per_pool*sizeof(vid_t));
  cout << "      Sample pools: " << buffer; 
  pretty_bytes(buffer, num_parts_gpu*vertices_per_part*dimension*sizeof(emb_t));
  cout << " - Embedding: " << buffer << endl;
}
#ifndef _BIG_GRAPHS
void BigGraphs::calculate_big_graphs_resources(long long num_vertices, long long num_edges, int num_parts_gpu, int num_pools_gpu, int dimension, int &num_parts, long long &vertices_per_part, int epoch_batch_size, long long& vids_per_pool){
  vertices_per_part = ceil(num_vertices*1.0/num_parts);
  vids_per_pool = vertices_per_part*2*2*epoch_batch_size;
  print_big_graphs_info();
}
#else
void BigGraphs::calculate_big_graphs_resources(long long num_vertices, long long num_edges, int input_num_parts_gpu, int input_num_pools_gpu, int dimension, int &return_num_parts, long long &return_vertices_per_part, int input_epoch_batch_size, long long & return_vids_per_pool){
  cout << "Calculating memory requirement ...\n";
  size_t gpu_free_memory, gpu_total_memory;
  CUDA_CHECK(cudaMemGetInfo(&gpu_free_memory,&gpu_total_memory));
  cout <<"  GPU total memory: " << gpu_total_memory << " - GPU free memory: " << gpu_free_memory << endl;
  double mbs_to_leave_empty = 100000000; 
  return_num_parts = input_num_parts_gpu;
  do{
    return_num_parts++;
    if (return_num_parts > PARTS_HARD_LIMIT){
      throw -1;
    }
    return_vertices_per_part = ceil(num_vertices*1.0/return_num_parts);
    return_vids_per_pool = return_vertices_per_part*2*2*input_epoch_batch_size+2;
  } while((long long) ((input_num_pools_gpu*((long long) (return_vids_per_pool)*sizeof(vid_t))))+input_num_parts_gpu*(long long) return_vertices_per_part*dimension*sizeof(emb_t)>(long long) gpu_free_memory-mbs_to_leave_empty);
  print_big_graphs_info();

}
#endif


vector<int> BigGraphs::parts_in_gpu_but_not_needed(vector<int> curr_gpu_part_state, vector<kernel_pair>& kernel_order, int index_in_kernel_order){
  vector<int> ret;
  set<int> needed_parts;
  for (int i = index_in_kernel_order + 1; i<index_in_kernel_order+num_parts_gpu; i++){
    auto kernel = kernel_order[i%kernel_order.size()];
    needed_parts.insert(kernel.first);
    needed_parts.insert(kernel.second);
  }
  for (int part : curr_gpu_part_state){
    if (needed_parts.count(part) == 0){
      ret.push_back(part);
    }
  }
  /*if (ret.size() > 2){
        printf("4\n");
    throw 4;
  }
*/
  return ret;
}
bool in_vector(vector<int>v, int val){
  for (auto c: v) if (c == val) return true;
  return false;
}
vector<int> BigGraphs::parts_not_in_gpu(vector<int> curr_gpu_part_state, vector<kernel_pair>& kernel_order, int index_in_kernel_order){
  vector<int> p_set;
  for (int i = index_in_kernel_order +1; i<index_in_kernel_order+num_parts_gpu;i++){
    auto kernel = kernel_order[i%kernel_order.size()];
    if (index_of_part_in_gpu(kernel.first, curr_gpu_part_state) == -1){
      if (!in_vector(p_set, kernel.first))
        p_set.push_back(kernel.first);
    }
    if (index_of_part_in_gpu(kernel.second, curr_gpu_part_state) == -1){
      if (!in_vector(p_set, kernel.second))
        p_set.push_back(kernel.second);
    }
  }
  reverse(p_set.begin(), p_set.end());
  return p_set;
}

int BigGraphs::index_of_part_in_gpu(int part, vector<int>& curr_gpu_state){
  for (int i =0; i<num_parts_gpu; i++){
    if (part == curr_gpu_state[i]) return i;
  }
  return -1;
}

shared_ptr<Node> BigGraphs::create_execution_graph(){
  int starting_epoch = 1;
  vector<kernel_pair> kernel_order;
  for (int j =0; j<num_parts; j++){
    for (int k =0; k< j+1; k++){
      kernel_order.push_back(make_pair(j, k));
    }
  }
  shared_ptr<Node> last_kernel_node_added(nullptr), last_sample_node_added(nullptr), last_part_switch_node(nullptr);
  vector<int> curr_gpu_part_state(num_parts_gpu, -1);
  vector<shared_ptr<Node>> part_swap_nodes_of_curr_gpu_parts(num_parts_gpu, nullptr);
  vector<vector<shared_ptr<Node>>> kernels_to_wait_for_before_swap(num_parts_gpu);
  int node_counter = 1;
  // create the beginning node
  shared_ptr<Node> beginning_node = make_shared<BeginningNode>();
  // first round, move the first num_parts_gpu parts
  last_part_switch_node = beginning_node;
  for (int i =0; i<num_parts_gpu; i++){
    shared_ptr<Node> part_swap_node = make_shared<PartSwitchNode>(this, part_streams[i].get(), i, -1, i, device_id, node_counter++);
    curr_gpu_part_state[i] = i;
    part_swap_nodes_of_curr_gpu_parts[i] = part_swap_node;
    kernels_to_wait_for_before_swap[i].clear();
    last_part_switch_node->add_edge(part_swap_node);
    last_part_switch_node = part_swap_node;
  }
  last_sample_node_added = beginning_node;
  for (int r = 0; r< num_rounds; r++){
    for (int k = 0; k<kernel_order.size(); k++){
      // create a sample node for this kernel
      int main = kernel_order[k].first;
      int sub = kernel_order[k].second;
      shared_ptr<Node> sample_node = make_shared<SampleCopyNode>(this, sample_stream.get(), main, sub, r%num_pool_sets, device_id, node_counter++);
      // make it depend on last sample node
      last_sample_node_added->add_edge(sample_node);
      last_sample_node_added = sample_node;
      // create kernel node
      int main_part_idx = index_of_part_in_gpu(main, curr_gpu_part_state);
      if (main_part_idx == -1) {
        throw 1; 
      }
      int sub_part_idx = index_of_part_in_gpu(sub, curr_gpu_part_state);
      if (sub_part_idx == -1){
       throw 2; 
      }
      shared_ptr<Node> kernel_node = make_shared<KernelNode>(this, kernel_streams[k%NUM_KERNEL_STREAMS].get(), main, sub, main_part_idx, sub_part_idx, starting_epoch, r, device_id, node_counter++);
      // make the sample node of this kernel a dependent of the kernel_node
      sample_node->add_edge(kernel_node);
      // make the kernel node depend on its swaps
      part_swap_nodes_of_curr_gpu_parts[main_part_idx]->add_edge(kernel_node);
      if (main!=sub)
        part_swap_nodes_of_curr_gpu_parts[sub_part_idx]->add_edge(kernel_node);
      // add this kernel to the list nodes that should be depended on in case of a part swap
      kernels_to_wait_for_before_swap[main_part_idx].push_back(kernel_node);
      if (main != sub)
        kernels_to_wait_for_before_swap[sub_part_idx].push_back(kernel_node);
      // make the last kernel node depend on this node
      if (last_kernel_node_added != nullptr){
        last_kernel_node_added->add_weak_edge(kernel_node);
      }
      // set this as the last kernel node created
      last_kernel_node_added = kernel_node;
      // do the switchity switch
      // find the parts in the GPU which are not in the next npg-1 kernels (with rollover)
      auto parts_to_bring = parts_not_in_gpu(curr_gpu_part_state, kernel_order, k);
      // find the parts in in the next npg-1 kernels which are not in the GPU
      auto parts_to_remove = parts_in_gpu_but_not_needed(curr_gpu_part_state, kernel_order, k);
      if (parts_to_bring.size() > parts_to_remove.size()){
        throw 6;
      }
      // for each part we can remove
      for (int part_to_remove : parts_to_remove){
        if (parts_to_bring.size() == 0){
          break;
        }
        // find the part to bring
        int part_to_bring = parts_to_bring.back();
        parts_to_bring.pop_back();
        // find the index of the part to remove
        int gpu_index = index_of_part_in_gpu(part_to_remove, curr_gpu_part_state);
        // create the switch node
        shared_ptr<Node> part_switch_node = make_shared<PartSwitchNode>(this,part_streams[gpu_index].get(), part_to_bring, part_to_remove, gpu_index, device_id, node_counter++);
        //last_part_switch_node->add_edge(part_switch_node);
        last_part_switch_node = part_switch_node;
        // make the switch wait for the kernels the depend on the part being switched out
        for (auto incoming_node : kernels_to_wait_for_before_swap[gpu_index]){
          incoming_node->add_edge(part_switch_node);
        }
        // set this part as the part in gpu_index of the part memory model
        curr_gpu_part_state[gpu_index] = part_to_bring;
        // clear the list of nodes that need to be waited for before before swapping out this part
        kernels_to_wait_for_before_swap[gpu_index].clear();
        // make this the node of the switch in gpu_index
        part_swap_nodes_of_curr_gpu_parts[gpu_index] = part_switch_node;
      } 

    }
    starting_epoch+=epoch_batch_size;
  }
  shared_ptr<Node> termination_node = make_shared<TerminationNode>(node_counter++);
  last_kernel_node_added->add_edge(termination_node);
  return beginning_node;
}

vid_t BigGraphs::get_positive_sample_ppr_host(vid_t source, vid_t starting_id, vid_t end_id, vid_t * V, vid_t * A, void* seed){
  vid_t numNeighbours = V[source+1] - V[source];
  vid_t sample = UINT_MAX;    
  unsigned long randNum = randn(seed);
  if (alpha == 0){
    if (numNeighbours == 0){
      return sample;
    } else {
      source = A[V[source] + (randNum % numNeighbours)];
      if (source <end_id && source>= starting_id) sample = source;
      return sample;
    }
  }
  while (randNum % MAX_ALPHA < alpha){
    if(numNeighbours !=0){
      source = A[V[source] + (randNum % numNeighbours)];
      if (source <end_id && source>= starting_id) sample = source;
      numNeighbours = V[source+1] - V[source];
      randNum = randn(seed);
    }
    else{
      return sample;
    }
  }
  return sample; // this should be return -1 if we want to match our code not verse's
}


void BigGraphs::sample_to_pool(int main_part, int sub_part, vid_t*& pool, int starting_ep, int batch_ep){
  if (total_samples <= 0){ // if I shouldn't sample
    printf("skipping this sampling\n");
    pool[0] = pool[1] = 0;
    if (time_finished_samples == -1) time_finished_samples = omp_get_wtime();
    return;
  }
  int local_sampling_threads = max(1,sampling_threads/concurrent_samplers);
  sample_idx_m.lock();
  int myidx = sample_idx;
  sample_idx++;
  sample_idx_m.unlock();
  vid_t dest_start_id = sub_part*vertices_per_part; // the starting id of acceptable positive and negative samples
  vid_t dest_end_id = (sub_part+1)*vertices_per_part; // the ending id of acceptable positive and negative samples
  if (dest_end_id>=csr->num_vertices){
    dest_end_id = csr->num_vertices; // if this is the last bin, we cannot go over the number of vertices
  }
  vid_t source_start_id = vertices_per_part*main_part;
  vid_t source_end_id = vertices_per_part*(main_part+1); // the highest possible vid to use as a source
  if (source_end_id>=csr->num_vertices){
    source_end_id = csr->num_vertices;
  } 
  vid_t vids_per_thread= ceil((source_end_id-source_start_id)*1.0/local_sampling_threads);
  long long * curr_indeces_s2d = new long long[local_sampling_threads]();
  long long * curr_indeces_d2s = new long long[local_sampling_threads]();
  vid_t **thread_pools = new vid_t*[local_sampling_threads];
#pragma omp parallel num_threads(local_sampling_threads)
  {
    unsigned long long id = omp_get_thread_num();
    thread_pools[id] =new vid_t[(unsigned long long)vids_per_thread*batch_ep*2*(1+(main_part!=sub_part))];
    seed mySeed;
    mySeed.x = (myidx+10+id)*123456789; // the values of the seed are a function of the thread id
    mySeed.y = (myidx+10+id)*362436069; // (i wasn't very creative with these functions but I didn't see any decrease in ML performance)
    mySeed.z = (myidx+10+id)*521288629;
    unsigned long long ep_s = starting_ep;
    unsigned long long ep_e = starting_ep+batch_ep;
    vid_t thread_starting_id = vids_per_thread*id+source_start_id;
    vid_t thread_end_id = thread_starting_id+vids_per_thread;
    if (id == local_sampling_threads-1) thread_end_id = source_end_id;
    vid_t * thread_pool = thread_pools[id];
    long long curr_index_in_pool_s2d;
    curr_index_in_pool_s2d = 0;
    long long curr_index_in_pool_d2s;
    curr_index_in_pool_d2s = 0; 
    vid_t p_sample;
    for (unsigned long long ep = ep_s; ep<ep_e; ep++){
      for (unsigned long long source = thread_starting_id; source<thread_end_id; source++){
        p_sample = get_positive_sample_ppr_host(source, dest_start_id, dest_end_id, csr->V, csr->E, &mySeed);

        if (p_sample!=UINT_MAX){
          thread_pool[curr_index_in_pool_s2d++] = source;
          thread_pool[curr_index_in_pool_s2d++] = p_sample;
        }
      }
    }
    if(main_part!=sub_part) {
      vid_t dest_start_id = main_part*vertices_per_part; // the starting id of acceptable positive and negative samples
      vid_t dest_end_id = (main_part+1)*vertices_per_part; // the ending id of acceptable positive and negative samples
      if (dest_end_id>csr->num_vertices){
        dest_end_id = csr->num_vertices; // if this is the last bin, we cannot go over the number of vertices
      }
      vid_t source_start_id = vertices_per_part*sub_part;
      vid_t source_end_id = vertices_per_part*(sub_part+1); // the highest possible vid to use as a source
      if (source_end_id>csr->num_vertices){
        source_end_id = csr->num_vertices;
      } 
      vid_t vids_per_thread= ceil((source_end_id-source_start_id)*1.0/local_sampling_threads);
      thread_starting_id = vids_per_thread*id+source_start_id;
      thread_end_id = thread_starting_id+vids_per_thread;
      if (id == local_sampling_threads-1) thread_end_id = source_end_id;
      for (int ep = ep_s; ep<ep_e; ep++){
        for (int source = thread_starting_id; source<thread_end_id; source++){
          p_sample = get_positive_sample_ppr_host(source, dest_start_id, dest_end_id, csr->V, csr->E, &mySeed);

          if (p_sample!=UINT_MAX){
            thread_pool[curr_index_in_pool_s2d+curr_index_in_pool_d2s++] = source;
            thread_pool[curr_index_in_pool_s2d+curr_index_in_pool_d2s++] = p_sample;
          }
        }
      } 
    }
    curr_indeces_s2d[id] = curr_index_in_pool_s2d;
    curr_indeces_d2s[id] = curr_index_in_pool_d2s;
  }
  long long prev_index =2;
  long long s2d=0;
  for (unsigned long long i =0; i<local_sampling_threads;i++){
    memcpy(pool+prev_index,thread_pools[i],curr_indeces_s2d[i]*(unsigned long long)sizeof(vid_t));
    prev_index += curr_indeces_s2d[i];
    s2d+= curr_indeces_s2d[i];
  }
  pool[0] = (vid_t)(s2d/2);
  if (main_part!=sub_part){
    long long d2s=0;
    for (unsigned long long i =0; i<local_sampling_threads;i++){
      memcpy(pool+prev_index,thread_pools[i]+curr_indeces_s2d[i],curr_indeces_d2s[i]*(unsigned long long)sizeof(vid_t));
      prev_index += curr_indeces_d2s[i];
      d2s+=curr_indeces_d2s[i];
    }
    pool[1] = (vid_t)d2s/2;
  } else {
    pool[1] =0;
  }
  if (total_samples <= 0){ // if I shouldn't sample
    printf("skipping this sampling\n");
    pool[0] = pool[1] = 0;
    if (time_finished_samples == -1) time_finished_samples = omp_get_wtime();
    return;
  }
  total_samples -= (pool[1]+pool[0]);
#ifdef CHECK_SAMPLES
  bool success = check_sample(csr->num_vertices, vertices_per_part, main_part, sub_part, pool.vids, pool.vids[0], pool.vids[1]);
  if (!success){
    printf("Sampling pool %d %d failed!\n", main_part, sub_part);
  }
#endif
  delete [] curr_indeces_s2d;
  delete [] curr_indeces_d2s;
  for (int  h =0;h <local_sampling_threads;h++){
    delete [] thread_pools[h];
  } 
  delete [] thread_pools;


}

bool u_has_edge_v_h(vid_t u, vid_t v, vid_t* d_V, vid_t* d_A){
  for (int i =0; i< d_V[u+1]-d_V[u];i++){
    if (d_A[i+d_V[u]]==v) return true;
  }
  return false;
}
template <class T>
inline bool within(T num, T lower_inc, T upper_exc){
  if (num>= lower_inc && num<upper_exc) return true;
  else return false;
}

// This function will check the samples generated for a bin to make sure they are within the boundaries of vertices bins
bool check_sample(int num_vertices, int vertices_per_part, int source_id, int dest_id, vid_t* samples_array, long long size_s2d, long long size_d2s){
  size_s2d = samples_array[0];
  size_d2s = samples_array[1];
  samples_array = samples_array +2;
  vid_t source_first_id = source_id*vertices_per_part;
  vid_t source_last_id_exc = (source_id+1)*vertices_per_part;
  if (source_last_id_exc>num_vertices) source_last_id_exc = num_vertices;
  vid_t dest_first_id = dest_id*vertices_per_part;
  vid_t dest_last_id_exc = (dest_id+1)*vertices_per_part;
  if (dest_last_id_exc>num_vertices) dest_last_id_exc = num_vertices;
  long long counter = 0;
  int succ=true;
  int error_counter =0;
  for (; counter<size_s2d*2;counter+=2){
    if (!within(samples_array[counter], source_first_id, source_last_id_exc)){
      succ = false;
      error_counter++;
      if (error_counter==2)
        return false;
    }
    if (!within(samples_array[counter+1], dest_first_id, dest_last_id_exc)){
      succ = false;
      error_counter++;
      if (error_counter==2)
        return false;
    }
  }
  long long offset = counter;
  for (; counter<offset+size_d2s*2;counter+=2){
    if (!within(samples_array[counter], dest_first_id, dest_last_id_exc)){
      succ = false;
      error_counter++;
      if (error_counter==2)
        return false;
    }
    if (!within(samples_array[counter+1], source_first_id, source_last_id_exc)){
      succ = false;
      error_counter++;
      if (error_counter==2)
        return false;
    }
  }
  if (succ){
  }
  else{
  }
  return succ;
}
int calc_sample_priority(int set_num, int num_parts, int main_part, int sub_part){
  return set_num*(num_parts*num_parts)+main_part*num_parts+sub_part+1;
}
