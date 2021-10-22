#ifndef BG_NAME
#define BG_NAME
#include <vector>
#include <atomic>
#include <stdio.h>
#include <omp.h>
#include <cstring>
#include <omp.h>
#include <mutex>
#include <condition_variable>
#include <string>
#include <deque>
#include <chrono>
#include <cuda_runtime.h>
#include "type_def.h"
#include "csr.h"
#include "cuda_utils.h"
#include "rand_helper.h"
#include "gpu_functions.h"
#include <memory>
#include <omp.h>
#define MAX_ALPHA 101
typedef pair<int, int> kernel_pair;

#define FULL_MASK 0xffffffff
enum node_type{
  beginning_node=0,
  termination_node,
  embedding_node,
  kernel_node,
  part_switch_node,
  sample_copy_node,
  node
};
class TaskQueue;
class Node {
  public:
    vector<shared_ptr<Node>> outgoing_edges;
    vector<shared_ptr<Node>> weak_outgoing_edges;
    vector<shared_ptr<cudaEvent_t>> wait_events;
    shared_ptr<cudaEvent_t> record_event;
    int incoming_edge_counter = 0;
    mutex self_mutex;
    node_type type; // describes the type of the node 
    int id;
    Node(int id = 0);
    void add_edge(shared_ptr<Node>& edge); 
    void add_weak_edge(shared_ptr<Node>& edge); 
    void increment_incoming_edge_counter();
    virtual void execute();
    virtual node_type get_type();
    friend class TaskQueue;
};

class TerminationNode: public Node{
  public:
    virtual void execute();
    TerminationNode(int id = -1);
};

class BeginningNode: public Node{
  public:
    BeginningNode(int id = 0);
    virtual void execute();
};

class TaskQueue {
  private:
    deque<shared_ptr<Node>> node_queue;
    condition_variable queue_not_empty;
    mutex queue_mutex;
    int local_threads;
    int device_id;
  public: 
    TaskQueue(int _task_queue_threads, int _device_id): local_threads(_task_queue_threads), device_id(_device_id){}
    void run();
    void add_to_queue(shared_ptr<Node>& new_node);
    void add_to_queue(shared_ptr<Node>&& new_node);
    void execute_node(shared_ptr<Node> node);
};

class BigGraphs{
  public:
    TaskQueue task_queue;
    int num_rounds;
    int device_id;

    //// embedding hyperparameters ////
    int dimension;// input
    int negative_samples; // input
    int epoch_batch_size; // input
    unsigned long long epochs; // input
    double learning_rate; // input
    int alpha; // input
    int lrd_strategy; // input
    double negative_weight; // input
    float* d_sigmoid_lookup_table; //input

    atomic<long long> total_samples;
    double time_finished_samples = -1;
    //// CUDA Kernel metadata
    unsigned long long size_of_shared_array;
    int WARPS_PER_BLOCK;
    int WARP_SIZE;
    int NUM_WARPS;
    int NUM_KERNEL_STREAMS=1;
    //// CUDA streams for exection
    shared_ptr<cudaStream_t> sample_stream; 
    vector<shared_ptr<cudaStream_t>> part_streams;
    vector<shared_ptr<cudaStream_t>> kernel_streams;

    atomic<double> time_spent_waiting_kernel;
    atomic<double> time_spent_waiting_for_sample_pool;
    atomic<double> time_spent_waiting_for_pool_gpus;
    atomic<double> time_spent_in_sc_waiting_for_kernel;

    //// sample workers
    mutex sample_idx_m;
    unsigned int sample_idx=1;
    int concurrent_samplers;
    int sampling_threads;
    //// embedding part ////
    // metadata
    CSR<vid_t>* csr;
    int num_parts_gpu;// input
    long long vertices_per_part;// calculated 
    int num_parts = 4;// calculated
    long long total_embedding_size;// calculated
    // CPU
    emb_t* h_embeddings; 
    // GPU
    vector<emb_t*> d_embedding_parts; 

    //// sample pool ////
    // metadata
    int num_pools_gpu; //input 
    int num_pool_sets; //input
    long long vids_per_pool; 
    // CPU
    vector<vector<vector<vid_t*>>> h_sample_pools; 
    // GPU
    vector<vid_t*> d_sample_pools; 

    //// sample - main function communication ////
    vector<vector<vector<bool>>> terminate_samplers; 

    //// sampler - sample manager communication ////
    vector<vector<vector<condition_variable>>> pool_full_cv; 
    vector<vector<vector<condition_variable>>> pool_empty_cv; 
    vector<vector<vector<mutex>>> pool_mutex; 
    vector<vector<vector<int>>> pool_status; 

    //// sample manager - sample manager - kernel communication ////
    vector<mutex> gpu_pool_mutex; 
    vector<int> gpu_pool_status; 
    vector<vector<int>> kernel_pool_index; 
    vector<vector<mutex>> kernel_pool_mutex; 
    vector<vector<condition_variable>> kernel_pool_full_cv; 
    vector<vector<condition_variable>> kernel_pool_empty_cv; 


    void sample_task(int starting_epoch, int round, int pool_set, int main, int sub); 
    void initialize_private_members();
    void print_big_graphs_info();
    BigGraphs(int _dimension, int _negative_samples, unsigned long long _epochs, double _learning_rate, int _epoch_batch_size, int _alpha, double _negative_weight, int _lrd_strategy, emb_t* _h_embeddings, CSR<vid_t> *_csr, float* _d_sigmoid_lookup_table, int _num_parts_gpu, int _num_pools_gpu, int _concurrent_samplers = 4, int sampling_threads = 16, int _num_pool_sets=2, int task_queue_threads = 8, int device_id=0);
    static bool use_big_graphs(CSR<vid_t>* a_csr, int emb_dimensions, int deviceID);
    void calculate_big_graphs_resources(long long num_vertices, long long num_edges, int num_parts_gpu, int num_pools_gpu, int dimension, int &num_parts, long long &vertices_per_part, int epoch_batch_size, long long& vids_per_pool);
    shared_ptr<Node> create_execution_graph();
    vector<int> parts_not_in_gpu(vector<int> curr_gpu_part_state, vector<kernel_pair>& kernel_order, int index_in_kernel_order);
    int index_of_part_in_gpu(int part, vector<int>& curr_gpu_state);
    vector<int> parts_in_gpu_but_not_needed(vector<int> curr_gpu_part_state, vector<kernel_pair>& kernel_order, int index_in_kernel_order);
    void begin_embedding();
    vid_t get_positive_sample_ppr_host(vid_t source, vid_t starting_id, vid_t end_id, vid_t * V, vid_t * A, void* seed);
    void sample_to_pool(int main_part, int sub_part, vid_t*& pool, int starting_ep, int batch_ep);
    ~BigGraphs();
};

class EmbeddingNode: public Node{
  protected:
    BigGraphs * bg;
    cudaStream_t* stream;
    int device_id;
    void wait_for_events();
    void record_event_of_node();
    void set_device();
  public:
    EmbeddingNode(BigGraphs* bg_o, cudaStream_t * _stream, int _device_id=0, int id = 0); 
};

class KernelNode: public EmbeddingNode {
  protected:
    struct kernel_callback_data{
      BigGraphs* bg;
      int main_idx;
      int sub_idx;
      int selected_gpu_pool;
      int round;
    };
    int starting_epoch;
    int main_idx;
    int sub_idx;
    int gpu_part_main;
    int gpu_part_sub;
    int round;
  public:
    static void kernel_finished_callback(void* data);
    KernelNode(BigGraphs* bg_o, cudaStream_t * _stream, int _main_idx, int _sub_idx, int _gpu_part_main, int _gpu_part_sub, int starting_epoch, int round, int _device_id = 0, int id = kernel_node);
    virtual void execute();
};

class SampleCopyNode: public EmbeddingNode{
  public: 
    struct callback_data {
      int main_idx;
      int sub_idx;
      int gpu_selected_pool;
      int pool_set;
      BigGraphs* bg;
    };
    int main_idx;
    int sub_idx;
    int pool_set;
    static void sample_copy_finished_callback(void* data);
    SampleCopyNode(BigGraphs* bg_o, cudaStream_t* _stream, int m, int s, int pool_s, int _device_id = 0, int id=0);
    virtual void execute();
};

class PartSwitchNode: public EmbeddingNode {
  public:
    int in_part;
    int out_part;
    int gpu_part_index;
    PartSwitchNode(BigGraphs* bg_o, cudaStream_t* _stream,  int in, int out, int gpu_index, int _device_id = 0, int id = part_switch_node);
    virtual void execute();
};
#endif
