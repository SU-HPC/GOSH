#include <iostream>
#include <stdio.h>
#include <cmath>
#include <random>
#include <omp.h>
#include <cstring>
#include <cuda_profiler_api.h>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
using namespace std;
// CSR object
#include "utils/csr.h"
// BigGraphs object
#include "utils/big_graphs.h"
// CUDA_CHECK macro
#include "utils/cuda_utils.h"
// init_sig_table()
#include "utils/sigmoid.h"
// Embedding_Kernel()
#include "utils/gpu_functions.h"
// ArgumentParser object
#include "utils/argparse/argparse.hpp"

// Coarsening parameters
#define LR_REF_NV 100000
float COARSE_STOP_PREC = 0.8;
int COARSE_SC_THRESHOLD_RATIO = 400;
double COARSE_SMOOTHING_RATIO = 0.5;
int COARSE_STOP_THRESHOLD = 1000;
int MIN_VERTICES_IN_GRAPH = 100;

// GPU paramters
int WARP_SIZE = 32;
int WARPS_PER_BLOCK = 32; 
int NUM_WARPS = WARPS_PER_BLOCK * NUM_BLOCKS;

// inputs
bool apply_coarsening = true;
string input_file_name = "input.graph";
string output_file_name = "EMBEDDING.emb";
int n_epochs; // number of training epochs
int directed; // 0 -> input graph is undirected, 1-> input graph is directed, 2-> input graph is in binary CSR format
int dimension = 128; // embedding dimension
int negative_samples = 3; // number of negative samples
double learning_rate = 0.025; // initial learning rate
int binary_output = 0; // 0 -> output is printed out as a string text file , 1 ->output will be printed in binary
int g_alpha = 0; // if (g_alpha==0) then adjacency similarity is used a similarity measure between vertices. 
              // 1 <= g_alpha < 100: Personalized PageRank is used a similarity measure with g_alpha as the damping factor as defined in VERSE: https://arxiv.org/abs/1803.04742
float negative_weight = 1; // the weight of a single negative sample update (negative updates are scaled by this value)
int lrd_strategy = 0; // the strategy used for learning rate decay:
// 0, 1: At every level, learning rate decays linearly from the initial learning rate starting at the first epoch until the last epoch based on the following equation:
//       current_learning_rate =  (max(1-current_epoch/total_epochs), 1e-4f)*initial_learning_rate
//       where current_epoch and total_epoch are the current and total epochs for the curent coarsening level
// 2, 3: The learning rate at the end of a level i is the same as it is on the beginning. No decay
// -----
// 1, 3; initial learning rate for every coarsening level differs based on the following heuristic:
//                                  { lr; if nv[i] < LR_REF_NV
//      learning_decay_at_level_i = { lr / sqrt(nv[i] / LR_REF_NV); otherwise
//      where:
//                 lr = learning_rate
//              nv[i] = number of vertices for the graph at level i
//          LR_REF_NV = tunable hyperparameter 
// 0, 2: initial learning rate at each level is the same as the original learning rate given as input
string emb_strategy = "s-fast"; // epoch distribution strategy across levels
// 'fast': The smallest graph is given half of n_epochs, and the next level is given half of that, and so on
// 's-fast': (COARSE_SMOOTHING_RATIO*n_epochs) epochs are distributed equally acorss levels, while the remainder is distrubuted based on the 'fast' rule
// normal: equal distribution of epochs across levels
// 'accurate': The opposite of fast; the biggest graph is given half of n_epochs and the smaller level is given half of that and so on
// s-accurate: (COARSE_SMOOTHING_RATIO*n_epochs) epochs are distributed equally across levels, while the remainder is distributed based on the 'accurate' rule 
// NOTE: in all of the aforementioned rules, a level is allocated a minimum of 1 epoch.
int num_parts_gpu = 3; // number of embedding partitions to store on the GPU at once
int num_pools_gpu = 4; // number of sample pools to store on the GPU at once
int epoch_batch_size = 5; // the number of samples per vertex in a single batch of samples for big graphs
int sampling_threads = 16; // the number of threads to be used during sampling for big graphs only
int num_sample_pool_sets = 2;
int concurrent_samplers = 4; // number of concurrent sample workers, each of which will use sampling_threads/concurrent_samplers number of threads to sample
int task_queue_threads = 3; // number of threads used by the task queue
int deviceID = 0; // the ID of the GPU to be used in the embedding

bool binary_input; // input graph is in BinaryCSR format 

CSR<vid_t>* csr; // the original input graph in CSR format

vector<pair<int,int>> num_parts; // the num_parts counts for the levels using big graphs

// random number generators for initializing the embedding
minstd_rand gen(std::random_device{}());
uniform_real_distribution<double> dist(0, 1);

void initialize_embeddings(long long num_vertices, int dimensions, emb_t*&embeddings_matrix);
void print_embeddings(emb_t* embeddings, int num_vertices, int dimension, string fname, bool binary);
void apply_dist_strategy(int n_epochs, int coarse_depth, int * & dist, string strategy);
bool set_global_parameters(int argc, const char **argv);
void print_global_parameters();
void read_graph();
void set_gpu_dimensions();
// Projects the embeddings from one level to the one before it
template <class T>
void expand_embeddings(CSR<T> * csr, int num_vertices, int dimension,  emb_t*&embedding_matrix_n, unsigned long long num_elements= 0){
  emb_t * temp_embeddings = embedding_matrix_n;
  // embedding_matrix_n = (emb_t*)aligned_malloc(num_vertices*dimension*sizeof(emb_t), ALIGNMENT);
  if (num_elements == 0){
    embedding_matrix_n = new emb_t[(long long)num_vertices*dimension];
  } else {
    embedding_matrix_n = new emb_t[num_elements];
  }
#pragma omp parallel firstprivate(temp_embeddings, csr, num_vertices, dimension) num_threads(sampling_threads)
  {

    emb_t *embedding_matrix = embedding_matrix_n;
#pragma omp for schedule(guided, 4) 
    for(unsigned long long i = 0; i < num_vertices; i++ ){
      T super = csr->map[i];
      if(super == T(-1)){
        for (unsigned long long j = 0; j < dimension; j++)
          embedding_matrix[i * (unsigned long long)dimension + j] = dist(gen)-0.5;
      }
      else{
        memcpy(embedding_matrix+i*dimension, temp_embeddings+super*dimension, dimension*sizeof(emb_t));
      }
    }
  }
  delete[] temp_embeddings;
}

bool failed = false;


// Takes the addresses of the source bin and the destination bin and the pool to read from 
// in the pool P[u]=v where u is the source vertex and v is the destination vertex 
int main(int argc, char * argv[]){
  float *sigmoid_lookup_table;
  float *d_sigmoid_lookup_table;
  emb_t * h_embeddings=NULL;
  emb_t * d_embeddings=NULL;
  vid_t* d_V;
  vid_t* d_A;
  unsigned long long size_of_shared_array; // shared memory size used by embedding kernels

  if (!set_global_parameters(argc, const_cast<const char**>(argv))){
    return -1;
  }
  print_global_parameters();
  read_graph();
  set_gpu_dimensions();
  cudaSetDevice(deviceID);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceID);
  printf("Device Number: %d\n", deviceID);
  printf("  Device name: %s\n", prop.name);
  cudaDeviceReset();
  init_sig_table(sigmoid_lookup_table);
  CUDA_CHECK(cudaMalloc((void**)&d_sigmoid_lookup_table, SIGMOID_TABLE_SIZE*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_sigmoid_lookup_table, sigmoid_lookup_table, sizeof(float)*SIGMOID_TABLE_SIZE, cudaMemcpyHostToDevice));
  size_of_shared_array=dimension*(WARPS_PER_BLOCK)*sizeof(emb_t);
  bool using_big_graphs = BigGraphs::use_big_graphs(csr, dimension, deviceID);

  cout << endl;
  cout << "GRAPH COARSENING: " << endl;
  cout << "Approach: MultiEdgeCollapse" << endl;
  cout << "--------------------------------------------------------" << endl;
  cout << "\nLEVEL 0:" << endl;
  cout << "Number of vertices: " << csr->num_vertices << "\nNumber of edges: " << csr->num_edges <<endl;
  cout << "Big graphs needed: ";
  cout << boolalpha;
  cout << using_big_graphs << endl;
  vector<bool> big_graphs_needed;
  bool locked_sizes = false;
  bool found_highest=false;
  if (!using_big_graphs){
    found_highest = true;
    locked_sizes = true; 
    CUDA_CHECK(cudaMalloc((void **)&d_embeddings, sizeof(emb_t)* (csr->num_vertices + (WARP_SIZE - (csr->num_vertices % WARP_SIZE))) * dimension));
    CUDA_CHECK(cudaMalloc((void **)&d_V, sizeof(vid_t)*(csr->num_vertices + 1)));
    CUDA_CHECK(cudaMalloc((void **)&d_A, csr->num_edges*sizeof(vid_t)));
    big_graphs_needed.push_back(false);
  }
  else{
    big_graphs_needed.push_back(true);
  }

  double start_time = omp_get_wtime();

  //get coarsed distribution
  vector<CSR<unsigned int> *> graphs; 
  bool switched_to_bg = false;
  graphs.push_back(csr);
  bool keep_coarsening = true;
  if (apply_coarsening){
    for(unsigned int i = 1; keep_coarsening; i++){
      cout << "Coarsening ...\n~~~~\n";
      CSR<unsigned int> * temp = graphs[i - 1]->coarsen_with_multi_edge_collapse("random", i * i * (csr->num_vertices /COARSE_SC_THRESHOLD_RATIO));
      cout << "~~~~\n" << "Number of vertices: " << temp->num_vertices << "\nNumber of edges: " << temp->num_edges <<endl;
      if (!found_highest){
        bool this_graph_needs_big_graphs = BigGraphs::use_big_graphs(temp, dimension, deviceID);
        big_graphs_needed.push_back(this_graph_needs_big_graphs);
        if (using_big_graphs!=this_graph_needs_big_graphs && !locked_sizes){
          found_highest = true;
          locked_sizes=true;
          CUDA_CHECK(cudaMalloc((void **)&d_embeddings, sizeof(emb_t)* (temp->num_vertices + (WARP_SIZE - (temp->num_vertices % WARP_SIZE))) * dimension));
          CUDA_CHECK(cudaMalloc((void **)&d_V, sizeof(vid_t)*(temp->num_vertices + 1)));
          CUDA_CHECK(cudaMalloc((void **)&d_A, temp->num_edges*sizeof(vid_t)));
        }
      } else {
        big_graphs_needed.push_back(false);
      }
      if(temp->num_vertices > COARSE_STOP_PREC * graphs[i - 1]->num_vertices){
        cout << "Graph was not added to the coarsened set and coarsening ended:\nThe graph fails the addition criteria of reducing the number of vertices by more than " << (100-COARSE_STOP_PREC*100) << "% which satisfies a stopping criteria and doesn't satisfy the criteria for adding a graph to the coarsened set. To change this percentage, please change parameter COARSE_STOP_PREC\n";
        keep_coarsening = false;
      }
      else if(temp->num_vertices < COARSE_STOP_THRESHOLD && temp->num_vertices <= MIN_VERTICES_IN_GRAPH){
        cout << "Graph was not added to the coarsened set and coarsening ended:\nThe graph fails the addition criteria of having more than " << MIN_VERTICES_IN_GRAPH << " vertices. To change this value, please change the variable MIN_VERTICES_IN_GRAPH in main.cu\n";
        keep_coarsening = false;
      } else {
          cout << "Big graphs needed: ";
          cout << boolalpha;
          cout << big_graphs_needed.back() << endl;
          cout << "\nadded LEVEL " << i << endl;
          graphs.push_back(temp);
          if (temp->num_vertices < COARSE_STOP_THRESHOLD){
            cout << "Coarsening ended:\nThe stopping criteria \"number of vertices in the last level < " << COARSE_STOP_THRESHOLD <<"\" has been satisfied. To change this value, please change the value of the input parameter COARSE_STOP_THRESHOLD\n";
            keep_coarsening = false;
          }
      }
    }
  }
  cout << "--------------------------------------------------------" << endl;
  cout << endl;
  
  double end_of_coarsening = omp_get_wtime(); 
  //

  //init embeddings and sig table
  cout << "EMBEDDING: " << endl;
  cout << "---------------------------------------------" << endl;

  //get distribution for the epochs
  int * epoch_distribution = new int[graphs.size()];
  apply_dist_strategy(n_epochs, graphs.size(),  epoch_distribution, emb_strategy);
  initialize_embeddings(graphs[graphs.size()-1]->num_vertices, dimension,h_embeddings);
  double learning_rate_e;
  double learning_rate_ec;
  double prev = omp_get_wtime();
  int batch_size_small_graphs=1;
  for(int i = graphs.size()-1; i >= 0; i--){
    if (lrd_strategy == 1 || lrd_strategy == 3){
      if(graphs[i]->num_vertices > LR_REF_NV){
        learning_rate_e = learning_rate / sqrt(graphs[i]->num_vertices / LR_REF_NV);
      }
      else{
        learning_rate_e = learning_rate;
      }
    } else {
      learning_rate_e = learning_rate;
    }

    cout << "Embedding on level: " << i << endl;
    cout << "Number of vertices: " << graphs[i]->num_vertices << endl;
    cout << "Number of edges: " << graphs[i]->num_edges << endl;
    cout << "Density: " << graphs[i]->num_edges / graphs[i]->num_vertices << endl;
    cout << "Number of epochs: " << epoch_distribution[graphs.size() - 1 - i] << endl;
    cout << "Learning Rate: " << learning_rate_e << endl;

    if (!big_graphs_needed[i]){

      cudaMemcpy(d_embeddings, h_embeddings, sizeof(emb_t)*graphs[i]->num_vertices*dimension, cudaMemcpyHostToDevice);
      cudaMemcpy(d_A, graphs[i]->E, sizeof(unsigned int)*graphs[i]->num_edges, cudaMemcpyHostToDevice);
      cudaMemcpy(d_V, graphs[i]->V, sizeof(unsigned int)*(graphs[i]->num_vertices + 1), cudaMemcpyHostToDevice);
      //
      for (int j = 0;j <epoch_distribution[graphs.size() - 1 - i]/batch_size_small_graphs;j++){
        if (lrd_strategy<2){
          learning_rate_ec = (max(1-float(j)*1.0/(epoch_distribution[graphs.size() - 1 - i]/batch_size_small_graphs), 1e-4f))*(learning_rate_e);
        } else {
          learning_rate_ec = learning_rate_e;
        }
        Embedding_Kernel<<<NUM_BLOCKS, NUM_THREADS, size_of_shared_array>>>(graphs[i]->num_vertices, batch_size_small_graphs, d_V, d_A, d_embeddings, dimension, negative_samples, learning_rate_ec, d_sigmoid_lookup_table, j, epoch_distribution[graphs.size() - 1 - i]/batch_size_small_graphs, g_alpha, negative_weight, WARP_SIZE, WARPS_PER_BLOCK, NUM_WARPS); //call for device
      }
      CUDA_CHECK(cudaMemcpy(h_embeddings, d_embeddings, sizeof(emb_t) * graphs[i]->num_vertices * dimension, cudaMemcpyDeviceToHost));      
    } else {
      if (!switched_to_bg){
        CUDA_CHECK(cudaFree(d_embeddings));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_V));
        switched_to_bg = true;
      }
      try{
        BigGraphs bg_embedder(dimension, negative_samples, epoch_distribution[graphs.size() - 1 - i], learning_rate_e, epoch_batch_size, g_alpha, negative_weight, lrd_strategy, h_embeddings, graphs[i], d_sigmoid_lookup_table, num_parts_gpu, num_pools_gpu, concurrent_samplers, sampling_threads, num_sample_pool_sets, task_queue_threads, deviceID ); // cocs samp tq_threads sample_thread
        bg_embedder.begin_embedding(); 
        num_parts.push_back(make_pair(i,bg_embedder.num_parts));
      } catch (int error){
        if (error == -1){
          i = -1;
          failed = true;
          printf("Failed to embed - num_parts was crossing thershold\n");
          break;
        }
      }
    }
    //copy embeddings & csr info


    //expand embeddings
    if(i > 0){ //no extension needed for the original graph
      double s_ex = omp_get_wtime();
      expand_embeddings(graphs[i], graphs[i-1]->num_vertices, dimension, h_embeddings);
      double e_ex = omp_get_wtime();
      cout << "Embedding projection took " << double(e_ex - s_ex)  << " seconds" << endl;
    }

    double coarsed = omp_get_wtime();
    double elapsed_secs = double(coarsed - prev) ;
    cout << "Level took " << elapsed_secs << " seconds" << endl;
    cout << endl;
    prev = coarsed;
  }
  if (failed == true){
    cout << "COARSE " << -1 << " seconds" << endl;

    cout << "TRAIN " << -1 << " seconds" << endl;
    cout << "NUM_PARTS " << endl;
    return 0;
  }
  cout << "--------------------------------------------------------" << endl;
  cout << endl;

  double elapsed_secs_c = double(end_of_coarsening - start_time) ;
  cout << "COARSE " << elapsed_secs_c << " seconds" << endl;

  double end_time = omp_get_wtime();
  double elapsed_secs = double(end_time - end_of_coarsening) ;
  cout << "TRAIN " << elapsed_secs << " seconds" << endl;
  cout << "NUM_PARTS ";
  for (auto level_parts : num_parts)
    cout << level_parts.first << "," << level_parts.second;
  cout << endl;

  cout <<"Printing the embeddings to " << output_file_name.c_str()<< " ...\n";
  if (binary_output != 2)
    print_embeddings(h_embeddings, csr->num_vertices, dimension,output_file_name.c_str(), binary_output); 

  return 0;
}
void initialize_embeddings(long long num_vertices, int dimensions, emb_t*&embeddings_matrix){
#ifndef ALIGNMENT
  embeddings_matrix = new emb_t[num_vertices*dimensions];
#else
  embeddings_matrix =(emb_t*) aligned_malloc((size_t)(num_vertices*dimensions*sizeof(emb_t)),(size_t) ALIGNMENT);
#endif

#pragma omp parallel for schedule(guided,32)
  for (long long i =0;i<num_vertices*dimensions;i++)
    embeddings_matrix[i]=(dist(gen)-0.5)/float(dimensions);
}  

// print embeddings in a text file
void print_embeddings(emb_t* embeddings, int num_vertices, int dimension, string fname, bool binary){
  if (binary) { 
    ofstream output(fname, std::ios::binary);
    int *nv = new int(num_vertices);
    int *d = new int(dimension);
    output.write(reinterpret_cast<char*>(nv),sizeof(int));
    output.write(reinterpret_cast<char*>(d),sizeof(int));
    output.write(reinterpret_cast<char*>(embeddings), (unsigned long long) sizeof(emb_t)*dimension*num_vertices);
  }
  else { 
    string type="w";
    FILE * fout = fopen(fname.c_str(), type.c_str());
    fprintf(fout, "%s ", to_string(num_vertices).c_str());
    fprintf(fout,"%s\n", to_string(dimension).c_str());
    for (int i =0; i<num_vertices;i++){
      fprintf(fout, "%s ", to_string(i).c_str());
      for (int j =0; j<dimension;j++){
        //    fout << embeddings[i*dimension+j]<<" ";
        fprintf(fout,"%s ", to_string(embeddings[i*dimension+j]).c_str());
      }
      fputc('\n',fout);
    }
  }
}

// generates the epoch distribution based on the strategy
void apply_dist_strategy(int n_epochs, int coarse_depth, int * & dist, string strategy){
  if (coarse_depth == 1) {
    dist[0] = n_epochs;
    return;
  } 
  if(strategy == "fast"){
    double per = 1;
    double rate = 0.5;
    for(int i = 0; i < coarse_depth; i++){
      per *= rate;
      if((int) ((double) n_epochs * per) == 0){
        dist[i] = 1;
      }
      else{
        dist[i] = (int) ((double) n_epochs * per);
      }
    }
  }
  else if(strategy == "accurate"){
    double per = 1;
    double rate = 0.5;
    for(int i = 0; i < coarse_depth; i++){
      per *= rate;
      if((int) ((double)n_epochs * per) == 0){
        dist[coarse_depth-i-1] = 1;
      }
      else{
        dist[coarse_depth-i-1] = (int) ((double)n_epochs * per);
      }
    }
  }
  else if(strategy == "s-fast"){
    double per = 1;
    double rate = 0.5;
    for(int i = 0; i < coarse_depth; i++){
      per *= rate;
      dist[i] = ((n_epochs * COARSE_SMOOTHING_RATIO) / coarse_depth) + (int) ((double)n_epochs * (double)((double)1 - COARSE_SMOOTHING_RATIO) * per);
      if(dist[i] == 0){
        dist[i] = 1;
      }
    }
  }
  else if(strategy == "s-accurate"){
    double per = 1;
    double rate = 0.5;
    for(int i = 0; i < coarse_depth; i++){
      per *= rate;
      dist[coarse_depth-i-1] = (n_epochs * COARSE_SMOOTHING_RATIO / coarse_depth) + (int) ((double)n_epochs * (double)((double)1 - COARSE_SMOOTHING_RATIO) * per);
      if(dist[coarse_depth-i-1] == 0){
        dist[coarse_depth-i-1] = 1;
      }
    }
  }
  else{
    for(int i = 0; i < coarse_depth; i++){
      dist[i] = (int) ((double)n_epochs/(double)coarse_depth);
      if(dist[i] == 0){
        dist[i] = 1;
      }
    }
  }
}
void print_global_parameters(){
  printf("Graph: %s\n", input_file_name.c_str());
  printf("Dimension: %d\n", dimension);
  printf("Negative samples: %d\n",negative_samples);
  printf("Epochs: %d\n", n_epochs);
  printf("Epoch batch size: %d\n", epoch_batch_size);
  printf("Negative weight: %f\n", negative_weight);
  printf("Device ID: %d\n", deviceID);
  printf("Directed: %d\n", directed);
  printf("Binary output: %d\n", binary_output);
  printf("Alpha: %d\n", g_alpha);
  printf("-------\n");
  printf("Learning rate: %f\n", learning_rate);
  printf("Learning rate decay strategy: %d\n", lrd_strategy);
  printf("-------\n");
  printf("Strategy: %s\n", emb_strategy.c_str());
  printf("Smoothing Ratio: %f\n", COARSE_SMOOTHING_RATIO);
  printf("-------\n");
  printf("Apply coarsening: %d\n", apply_coarsening);
  printf("Stopping Threshold: %d\n", COARSE_STOP_THRESHOLD);
  printf("Stopping Threshold Precision: %f\n", COARSE_STOP_PREC);
  printf("Matching Threshold: %d\n", COARSE_SC_THRESHOLD_RATIO);
  printf("Minimum Number of Vertices in Graph: %d\n", MIN_VERTICES_IN_GRAPH);
  printf("-------\n");
  printf("Number of embedding parts on the GPU: %d\n", num_parts_gpu);
  printf("Number of sampling pools on the GPU: %d\n", num_pools_gpu);
  printf("Sampling threads: %d\n", sampling_threads);
  printf("Concurrent samplers: %d\n", concurrent_samplers);
  printf("Sample pool sets: %d\n", num_sample_pool_sets);
  printf("Task queue thread: %d\n", task_queue_threads);
}
#ifndef _ARGPARSE

void usage(){
  printf("input format:\ninput_graph dimension negative_samples epochs learning_rate epoch_batch_size alpha strategy smoothing_ratio stopping_threshold stopping_precision matching_threshold_ratio min_vertices_in_graph negative_weight lrd_strategy num_parts_gpu num_pools_gpu sampling_threads concurrent_samplers num_sample_pool_sets task_queue_threads device_id directed binary_output apply_coarsening output_path \n");
  printf("input format:\ninput_graph dimension negative_samples epochs learning_rate epoch_batch_size alpha strategy smoothing_ratio stopping_threshold negative_weight lrd_strategy num_parts_gpu num_pools_gpu sampling_threads concurrent_samplers num_sample_pool_sets task_queue_threads device_id directed binary_output apply_coarsening output_path \n");
}

bool set_global_parameters(int argc, const char **argv){
  if (argc < 21) {
    usage();
    return false;
  }
  input_file_name = argv[1]; 
  dimension=atoi(argv[2]); 
  negative_samples = atoi(argv[3]); 
  n_epochs = atoi(argv[4]); 
  learning_rate = atof(argv[5]); 
  epoch_batch_size = atoi(argv[6]); 
  g_alpha = atoi(argv[7]);
  emb_strategy = string(argv[8]);
  COARSE_SMOOTHING_RATIO = atof(argv[9]);
  COARSE_STOP_THRESHOLD = atoi(argv[10]);
  COARSE_STOP_PREC = atof(argv[11]);
  COARSE_SC_THRESHOLD_RATIO = atoi(argv[12]);
  MIN_VERTICES_IN_GRAPH = atoi(argv[13]);
  negative_weight = atof(argv[14]);
  lrd_strategy = atoi(argv[15]);
  num_parts_gpu = atoi(argv[16]);
  num_pools_gpu = atoi(argv[17]);
  sampling_threads = atoi(argv[18]);
  concurrent_samplers = atoi(argv[19]);
  num_sample_pool_sets = atoi(argv[20]);
  task_queue_threads = atoi(argv[21]);
  deviceID = atoi(argv[22]);
  directed = atoi(argv[23]);
  binary_output = atoi(argv[24]);
  apply_coarsening = atoi(argv[25]);
  output_file_name = string(argv[26]); 

  return true;
}
#else
bool set_global_parameters(int argc, const char **argv){
  argparse::ArgumentParser parser;
  parser.addArgument("-i", "--input-graph", 1, false);
  parser.addArgument("-o", "--output-embeddings", 1, false);
  parser.addArgument("--directed", 1, false);
  parser.addArgument("-e", "--epochs", 1, false);
  parser.addArgument("-d", "--dimension", 1, true);
  parser.addArgument("-a", "--alpha", 1, true);
  parser.addArgument("-s", "--negative-samples", 1, true);
  parser.addArgument("-b", "--binary-output", 0 , true);
  parser.addArgument("--device-id", 1, true);
  parser.addArgument("--negative-weight", 1, true);

  parser.addArgument("--coarsening-stopping-threshold", 1, true);
  parser.addArgument("--coarsening-stopping-precision", 1, true);
  parser.addArgument("--coarsening-matching-threshold-ratio", 1, true);
  parser.addArgument("--coarsening-min-vertices-in-graph", 1, true);
  parser.addArgument("--no-coarsening", 0, true);

  parser.addArgument("--epoch-strategy", 1, true);
  parser.addArgument("--smoothing-ratio", 1, true);

  parser.addArgument("-l", "--learning-rate", 1, true);
  parser.addArgument("--learning-rate-decay-strategy", 1, true);
  
  parser.addArgument("--epoch-batch-size", 1, true);
  parser.addArgument("--sampling-threads", 1, true);
  parser.addArgument("--concurrent-samplers", 1, true);
  parser.addArgument("--task-queue-threads", 1, true);
  parser.addArgument("--num-pools", 1, true);
  parser.addArgument("--num-parts", 1, true);
  parser.addArgument("--num-sample-pool-sets",1, true);

  try {
    parser.parse(argc, argv);
  } catch(...){
    parser.usage();
    return false;
  }

  input_file_name = parser.retrieve<string>("i"); 
  output_file_name =parser.retrieve<string>("o");  
  directed =parser.retrieve<int>("directed"); 
  n_epochs =parser.retrieve<int>("e");  
  if (parser.gotArgument("d"))
    dimension=parser.retrieve<int>("d");
  if (parser.gotArgument("num-pools"))
    num_pools_gpu=parser.retrieve<int>("num-pools");
  if (parser.gotArgument("num-parts"))
    num_parts_gpu=parser.retrieve<int>("num-parts");
  if (parser.gotArgument("no-coarsening"))
    apply_coarsening = false;
  else
    apply_coarsening = true;
  if(parser.gotArgument("s"))
    negative_samples =parser.retrieve<int>("s");  
  if (parser.gotArgument("l"))
    learning_rate =parser.retrieve<double>("l"); 
  if (parser.gotArgument("epoch-batch-size"))
    epoch_batch_size =parser.retrieve<int>("epoch-batch-size");  
  if (parser.gotArgument("a"))
    g_alpha =parser.retrieve<int>("a"); 
  if (parser.gotArgument("epoch-strategy"))
    emb_strategy =parser.retrieve<string>("epoch-strategy"); 
  if (parser.gotArgument("smoothing-ratio"))
    COARSE_SMOOTHING_RATIO =parser.retrieve<float>("smoothing-ratio"); 
  if (parser.gotArgument("coarsening-stopping-threshold"))
    COARSE_STOP_THRESHOLD =parser.retrieve<int>("coarsening-stopping-threshold");  
  if (parser.gotArgument("coarsening-stopping-precision"))
    COARSE_STOP_PREC =parser.retrieve<float>("coarsening-stopping-precision");  
  if (parser.gotArgument("coarsening-matching-threshold-ratio"))
    COARSE_SC_THRESHOLD_RATIO =parser.retrieve<float>("coarsening-matching-threshold-ratio");  
  if (parser.gotArgument("coarsening-min-vertices-in-graph"))
    MIN_VERTICES_IN_GRAPH =parser.retrieve<int>("coarsening-min-vertices-in-graph");  
  if (parser.gotArgument("negative-weight"))
  negative_weight =parser.retrieve<float>("negative-weight"); 
  if (parser.gotArgument("learning-rate-decay-strategy"))
    lrd_strategy =parser.retrieve<int>("learning-rate-decay-strategy");  
  if (parser.gotArgument("sampling-threads"))
    sampling_threads =parser.retrieve<int>("sampling-threads");  
  if (parser.gotArgument("concurrent-samplers"))
    concurrent_samplers =parser.retrieve<int>("concurrent-samplers");  
  if (parser.gotArgument("task-queue-threads"))
    task_queue_threads =parser.retrieve<int>("task-queue-threads");
  if (parser.gotArgument("device-id")) 
  deviceID =parser.retrieve<int>("device-id"); 
  if (parser.gotArgument("b"))
    binary_output = true;
  else
    binary_output = false;
  if (parser.gotArgument("num-sample-pool-sets"))
    num_sample_pool_sets=parser.retrieve<int>("num-sample-pool-sets");
 
  return true;
}

#endif


void read_graph(){
  if (directed == 2){
    printf("Binary: true\n");
    binary_input=true;
  } else {
    binary_input = false;
    printf("Binary: false\n");
  }
  printf("Output: %s\n", output_file_name.c_str());
  if (!binary_input)
    csr = new CSR<vid_t>(input_file_name, directed, false);
  else
    csr = new CSR<vid_t>(input_file_name);
}

void set_gpu_dimensions(){
  WARP_SIZE = 32;
  if(dimension <= 8){
    WARP_SIZE = 8;
  }
  else if(dimension <= 16){
    WARP_SIZE = 16;
  }
  WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
  NUM_WARPS = WARPS_PER_BLOCK * NUM_BLOCKS;
  cout << "NUM_BLOCKS: " << NUM_BLOCKS << " NUM_THREADS: " << NUM_THREADS << endl;
  cout << "WARP_SIZE: " << WARP_SIZE << " WARPS_PER_BLOCK: " << WARPS_PER_BLOCK << " NUM_WARPS: " <<  NUM_WARPS << endl;

}
