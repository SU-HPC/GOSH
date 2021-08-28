//#define _VERBOSE
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <omp.h>

#include "align.h"
#include "rand_helper.h"

#define EDGE_COLLAPSE "edge_collapse"
#define STAR_COLLAPSE "star_collapse"

#define SC_SIM_THRESHOLD 0.01

#define JACCARD "jaccard"
#define RANDOM "random"
#define ORDERED
#define OPTIMIZED

#define ALIGNMENT 128
#define MEASURE_TIME

using namespace std;

/* Helper Funtions */
static unsigned long x = 123456789, y = 362436069, z = 521288629; //seeds for randn

template <class T>
void initialize_array(T *arr, T init, ll size)
{
  for (unsigned long long i = 0; i < size; i++)
  {
    arr[i] = init;
  }
}

template <class T>
void print_array(T *arr, unsigned int size)
{
  cout << "Array: " << endl;
  for (unsigned int i = 0; i < size; i++)
  {
    cout << i << ". " << arr[i] << endl;
  }
  cout << endl;
}

template <class T>
void CSR<T>::skip_mtx_comments(ifstream &file)
{
  string line;
  unsigned long long place = 0;
  while (getline(file, line))
  { //skip comments
    if (line.at(0) == '%' || line.at(0) == '#')
    {
      place = file.tellg();
      continue;
    }
    else
    {
      file.seekg(place);
      break;
    }
  }
}
/* */

template <class T>
CSR<T>::CSR(ll nv, ll ne, T *v, T *e, T *w, T *map) : num_edges(ne), num_vertices(nv), E(e), V(v), W(w), map(map)
{
  sizeOfUnit = sizeof(T);
    if (alias_table_imp ==0){
    construct_ps_alias_table_gv();
    construct_ns_alias_table_gv(0.75);
  } else {
    construct_alias_table_line();
  }
}

template <class T>
CSR<T>::CSR(string fname, int ati): alias_table_imp(ati)
{
  sizeOfUnit = sizeof(T);
  int flag = read_bcsr(fname);
  if (flag != 1)
  {
    cout << "Failed to read CSR from bcsr file.\n";
    throw 1;
  }
   if (alias_table_imp == 0){  
    construct_ps_alias_table_gv();
    construct_ns_alias_table_gv(0.75);
  } else {
    construct_alias_table_line();
  }
}

template <class T>
CSR<T>::CSR(string fname, const bool directed, const bool weighted, bool binary, int ati)
{
  sizeOfUnit = sizeof(T);
  //int flag = mtx2csr(fname);
  int flag;
  if (!binary)
    flag = mtx2csr(fname, directed, weighted);
  else 
    flag = bmtx2csr(fname, directed, weighted);  
  if (flag != 1)
  {
    cout << "Failed to read CSR from mtx file.\n";
    throw 1;
  }
  alias_table_imp = ati;
  if (alias_table_imp == 0){  
    construct_ps_alias_table_gv();
    construct_ns_alias_table_gv(0.75);
  } else {
    construct_alias_table_line();
  }
}

template <class T>
CSR<T>::CSR(CSR &copy) : num_edges(copy.num_edges), num_vertices(copy.num_vertices)
{
#ifndef ALIGNMENT
  V = new T[num_vertices]; // create V array
  E = new T[num_edges];  // create E array
#else
  // V = (T*)_aligned_malloc(num_vertices*sizeof(T),ALIGNMENT);
  // E = (T*)_aligned_malloc(num_edges*sizeof(T),ALIGNMENT);
  V = (T *)aligned_malloc(num_vertices * sizeOfUnit, ALIGNMENT);
  E = (T *)aligned_malloc(num_edges * sizeOfUnit, ALIGNMENT);
#endif
  if (copy.W != NULL)
#ifndef ALIGNMENT
    W = new T[num_edges]; // If weighted create W
#else
    //    W = (T*) _aligned_malloc(num_edges*sizeof(T), ALIGNMENT);
    W = (T *)aligned_malloc(num_edges * sizeOfUnit, ALIGNMENT);
#endif
  // creation of arrays first to check if memory is available. If not an exception will be thrown
  for (unsigned long long i = 0; i < num_vertices; i++)
    V[i] = copy.V[i];

  for (unsigned long long i = 0; i < num_edges; i++)
    E[i] = copy.E[i];

  if (copy.W != NULL)
    for (unsigned long long i = 0; i < num_edges; i++)
      W[i] = copy.W[i];
  v_alias.size = copy.v_alias.size;
  v_alias.alias_p = new float[copy.v_alias.size];
  v_alias.alias_v = new T[copy.v_alias.size];
  for (int i = 0; i < copy.v_alias.size; i++)
  {
    v_alias.alias_p[i] = copy.alias_p[i];
    v_alias.alias_v[i] = copy.alias_v[i];
  }
  ns_alias.size = copy.size;
  ns_alias.alias_p = new float[copy.ns_alias.size];
  ns_alias.alias_v = new T[copy.ns_alias.size];
  for (int i = 0; i < copy.ns_alias.size; i++)
  {
    ns_alias.alias_p[i] = copy.alias_p[i];
    ns_alias.alias_v[i] = copy.alias_v[i];
  }
}

template <class T>
CSR<T>::~CSR()
{
#ifndef ALIGNMENT
  if (V != nullptr)
    delete[] V;
  if (E != nullptr)
    delete[] E;
  if (W != nullptr)
    delete[] W;
#else
  if (V != nullptr)
    aligned_free(V);
  if (E != nullptr)
    aligned_free(E);
  if (W != nullptr)
    aligned_free(W);

#endif
}

template <class T>
T CSR<T>::get_correct_edge_index(T v, T * edge_place)
{
  /*
  T i = V[v];
  for (i; E[i] != -1; i++)
  {
    continue;
  }
  return i;
  */
  T i = V[v] + edge_place[v];
  edge_place[v]++;
  return i;
}

template <class T>
bool CSR<T>::is_weighted() const { return W != NULL; }

template <class T>
int CSR<T>::mtx2csr(string file_path, bool directed, bool weighted)
{
  vector<vector<T>> mtx;
  string line;
  T v, w, weight;

  ifstream file;
  file.open(file_path); //open file
  if (file.fail())    //check if file opened correctly
    return 0;
  skip_mtx_comments(file); //skip comments

  while (getline(file, line))
  { //get the number of edges and vertices
    vector<T> temp;
    stringstream ss(line);
    ss >> v;
    if (v > num_vertices)
    {
      num_vertices = v;
    }
    ss >> w;
    if (w > num_vertices)
    {
      num_vertices = w;
    }
    num_edges++; //update number of edges

    //fill matrix
    temp.push_back(v);
    temp.push_back(w);
    if /*constexpr*/ (weighted)
    { //get weight if the graph is weighted
      ss >> weight;
      temp.push_back(weight);
    }
    mtx.push_back(temp);
    //
  }
  if /*constexpr*/ (!directed)
  { //if undirected graph edge amount is double the line amount
    num_edges *= 2;
  }
  num_vertices++;
  file.clear();
  file.close();
#ifndef ALIGNMENT
  V = new T[num_vertices + 1](); //allocate V
  E = new T[num_edges];       //allocate A
#else
  V = (T *)aligned_malloc((num_vertices + 1) * sizeOfUnit, ALIGNMENT); //allocate V
  memset(V,0,sizeOfUnit*(num_vertices+1));
  E = (T *)aligned_malloc(num_edges * sizeOfUnit, ALIGNMENT);       //allocate A
#endif
  for (T i = 0; i < num_edges; i++)
  {
    E[i] = UINT_MAX;
  }

  for (T i = 0; i < mtx.size(); i++) //get the number of edges per vertex
  {
    V[mtx[i][0] + 1]++;
    if (!directed)
    {
      V[mtx[i][1] + 1]++;
    }
  }

  for (T i = 1; i < num_vertices + 1; i++)
  { //organize V
    V[i] += V[i - 1];
  }

  T * edge_place = new T[num_vertices]();
  for (T i = 0; i < mtx.size(); i++)
  { //get the number of edges per vertex
    E[get_correct_edge_index(mtx[i][0], edge_place)] = mtx[i][1];
    if /*constexpr*/ (weighted)
    {
      W[get_correct_edge_index(mtx[i][0], edge_place)] = mtx[i][2];
    }
    if /*constexpr*/ (!directed)
    {
      E[get_correct_edge_index(mtx[i][1], edge_place)] = mtx[i][0];
      if /*constexpr*/ (weighted)
      { //populate weight array if the graph is weigted
        W[get_correct_edge_index(mtx[i][1], edge_place)] = mtx[i][2];
      }
    }
  }
  return 1;
}


// Creates a CSR from a binary edge list
template <class T>
int CSR<T>::bmtx2csr(string file_path, bool directed, bool weighted)
{
  vector<vector<T>> mtx;
  string line;
  T v, w, weight;
  unsigned long long read_v, read_w, read_weight;
  FILE * file = fopen(file_path.c_str(), "rb");
  // file.open(file_path, "rb"); //open file
  if (file==NULL)    //check if file opened correctly
    return 0;
  // skip_mtx_comments(file); //skip comments
  read_v = fread(&v, sizeof(T), 1, file);
  read_w = fread(&w, sizeof(T), 1, file);
  while (read_v == read_w && read_v == 1)
  { //get the number of edges and vertices
    vector<T> temp;
    // stringstream ss(line);
    // ss >> v;
    if (v > num_vertices)
    {
      num_vertices = v;
    }
    // ss >> w;
    if (w > num_vertices)
    {
      num_vertices = w;
    }
    num_edges++; //update number of edges

    //fill matrix
    temp.push_back(v);
    temp.push_back(w);
    if /*constexpr*/ (weighted)
    { //get weight if the graph is weighted
      // ss >> weight;
      read_weight = fread(&weight, sizeof(T), 1, file);
      if (read_weight != num_edges){
        return 0;
      }
      temp.push_back(weight);
    }
    mtx.push_back(temp);
    //
    read_v = fread(&v, sizeof(T), 1, file);
    read_w = fread(&w, sizeof(T), 1, file);
  }
  if /*constexpr*/ (!directed)
  { //if undirected graph edge amount is double the line amount
    num_edges *= 2;
  }
  num_vertices++;
  // file.clear();
  // file.close();
  fclose(file);
#ifndef ALIGNMENT
  V = new T[num_vertices + 1](); //allocate V
  E = new T[num_edges];       //allocate A
#else
  V = (T *)aligned_malloc((num_vertices + 1) * sizeOfUnit, ALIGNMENT); //allocate V
  memset(V,0,sizeOfUnit*(num_vertices+1));
  E = (T *)aligned_malloc(num_edges * sizeOfUnit, ALIGNMENT);       //allocate A
#endif
  for (T i = 0; i < num_edges; i++)
  {
    E[i] = UINT_MAX;
  }

  for (T i = 0; i < mtx.size(); i++) //get the number of edges per vertex
  {
    V[mtx[i][0] + 1]++;
    if (!directed)
    {
      V[mtx[i][1] + 1]++;
    }
  }

  for (T i = 1; i < num_vertices + 1; i++)
  { //organize V
    V[i] += V[i - 1];
  }

  T * edge_place = new T[num_vertices]();
  for (T i = 0; i < mtx.size(); i++)
  { //get the number of edges per vertex
    E[get_correct_edge_index(mtx[i][0], edge_place)] = mtx[i][1];
    if /*constexpr*/ (weighted)
    {
      W[get_correct_edge_index(mtx[i][0], edge_place)] = mtx[i][2];
    }
    if /*constexpr*/ (!directed)
    {
      E[get_correct_edge_index(mtx[i][1], edge_place)] = mtx[i][0];
      if /*constexpr*/ (weighted)
      { //populate weight array if the graph is weigted
        W[get_correct_edge_index(mtx[i][1], edge_place)] = mtx[i][2];
      }
    }
  }
  return 1;
}


template <class T>
int CSR<T>::read_bcsr(string file_path)
{
  ll *nv = new ll;
  ll *ne = new ll;
  FILE *file = fopen(file_path.c_str(), "rb");
  if (file == NULL)
  {
#ifdef _VERBOSE
    cout << "No such file!" << endl;
#endif
    delete nv;
    delete ne;
    return -1;
  }
  unsigned long long read = fread(nv, sizeof(ll), 1, file);
  if (!read == sizeof(ll) * 1)
  {
#ifdef _VERBOSE
    cout << "Reading the number of vertices failed! Expected to read " << sizeof(ll) << " bytes but only read " << read;
#endif
    delete nv;
    delete ne;
    return -2;
  }
#ifdef _VERBOSE
  cout << "Nummber of vertices is " << *nv << endl;
#endif
  read = fread(ne, sizeof(ll), 1, file);
  if (!read == sizeof(ll) * 1)
  {
#ifdef _VERBOSE
    cout << "Reading the number of edges failed! Expected to read " << sizeof(ll) << " bytes but only read " << read;
#endif
    delete nv;
    delete ne;
    return -3;
  }
#ifdef _VERBOSE
  cout << "Nummber of edges is ss " << *ne << endl;
#endif
  if (feof(file))
  {
#ifdef _VERBOSE
    cout << "file ended\n";
#endif
    delete nv;
    delete ne;
    return -6;
  }
#ifndef ALIGNMENT
  T *V = new T[*nv + 1];
  T *E = new T[*ne];
#else
  T *V = (T *)aligned_malloc((*nv + 1) * sizeOfUnit, ALIGNMENT);     //allocate V
  T *E = (T *)aligned_malloc(*ne * sizeOfUnit, ALIGNMENT);       //allocate A
#endif
  T *weights = NULL;
  /*  int *temp;
  while(!feof(file)){
    fread((void*)A, sizeof(int), 1, file);
    cout << temp<<endl;
  }*/
  read = fread((void *)V, sizeof(T), *nv, file);
  if (read != (*nv))
  {
#ifdef _VERBOSE
    cout << "Reading the vertices array failed! Expected to read " << sizeOfUnit * (*nv) << " bytes but only read " << read;
#endif
    delete nv;
    delete ne;
    delete[] V;
    delete[] E;
    return -4;
  }
  V[*nv] = *ne;
#ifdef _VERBOSE
  cout << "Read the V vector " << endl;
#endif

  read = fread((void *)E, sizeof(T), *ne, file);
  //cout << "Read " << read << " when reading edge vector" << endl;
  if (read != (*ne))
  {
    cout << "Reading the edges array failed! Expected to read " << sizeof(T) * (*ne) << " bytes but only read " << read;
    delete nv;
    delete ne;
    delete[] V;
    delete[] E;
    return -5;
  }
#ifdef _VERBOSE
  cout << "Read the A vector " << endl;
#endif
#ifndef ALIGNMENT
  weights = new T[*ne];
#else
  weights = (T *)aligned_malloc(*ne * sizeOfUnit, ALIGNMENT);
#endif
  read = fread((void *)weights, sizeof(T), *ne, file);
  if (read == 0)
  {
    //cout << "Graph is unweighted\n";
    delete weights;
    weights = NULL;
  }
  else if (read != *ne)
  {
#ifdef _VERBOSE
    cout << "Reading the weights array failed! Expected to read " << sizeof(T) * (*ne) << " bytes but only read " << read << endl;
    cout << "Graph is unweighted";
#endif
    delete nv;
    delete ne;
    delete[] V;
    delete[] E;
    delete weights;
    return -6;
  }
  else
  {
    cout << "Read the W vector \n";
  }

  // Creating the CSR object
  this->V = V;
  this->E = E;
  this->W = weights;
  this->num_edges = *ne;
  this->num_vertices = *nv;
  delete nv;
  delete ne;
  fclose(file);
#ifdef _MOCK
  cout << "V: ";
  for (unsigned long long i = 0; i < csr_ptr->num_vertices + 1; i++)
  {
    cout << csr_ptr->V[i] << ", ";
  }
  cout << "\b\b\b\b\b\b\n";
  cout << "A: ";
  for (unsigned long long i = 0; i < csr_ptr->num_edges; i++)
  {
    cout << csr_ptr->E[i] << ", ";
  }
  cout << "\b\b\b\b\b\b\n";
#endif
  return 1;
}

template <class T>
int CSR<T>::write_bcsr(string ofile_path)
{
  // takes a BCSR formatted graph and wrties it in binary format to a file
  // the V is a pointer to the vertex array, A is a pointer to the edge array, num_vertices is the number of vertices, num_edges is the number of edges
  // returns flags
  FILE *file = fopen(ofile_path.c_str(), "wb");
  if (file == NULL)
  {
#ifdef _VERBOSE
    cout << "Could not create file to write BCSR. Returning code -1\n";
#endif
    return -1;
  }
  long long *numVertices = new long long(num_vertices);
  long long *numedges = new long long(num_edges);
  unsigned long long elements_written;
#ifdef _VERBOSE
  cout << "Writing the number of vertices which is " << num_vertices << endl;
#endif
  elements_written = fwrite(numVertices, sizeof(long long), 1, file);

  if (elements_written != 1)
  {
#ifdef _VERBOSE
    cout << "Could not write the number of vertices correctly. Expected to write " << 1 << " long long but wrote " << elements_written << ". Returning code -2\n";
#endif
    delete numVertices;
    delete numedges;
    return -2;
  }

#ifdef _VERBOSE
  cout << "Writing the number of edges which is " << num_edges << endl;
#endif
  elements_written = fwrite(numedges, sizeof(long long), 1, file);
  if (elements_written != 1)
  {
#ifdef _VERBOSE
    cout << "Could not write the number of edges correctly. Expected to write " << 1 << " long long but wrote " << elements_written << ". Returning code -2\n";
#endif
    delete numVertices;
    delete numedges;
    return -3;
  }
#ifdef _VERBOSE
  cout << "Writing the V array " << endl;
#endif
  elements_written = fwrite(V, sizeof(T), num_vertices, file);
  if (elements_written != num_vertices)
  {
#ifdef _VERBOSE
    cout << "Could not write the array of vertices correctly. Expected to write " << num_vertices << " ints but wrote " << elements_written << ". Returning code -4\n";
#endif
    delete numVertices;
    delete numedges;
    return -4;
  }
#ifdef _VERBOSE
  cout << "Writing the A array " << endl;
#endif
  elements_written = fwrite(E, sizeof(T), num_edges, file);
  if (elements_written != num_edges)
  {
#ifdef _VERBOSE
    cout << "Could not write the array of edges correctly. Expected to write " << num_edges << " ints but wrote " << elements_written << ". Returning code -5\n";
#endif
    delete numVertices;
    delete numedges;
    return -5;
  }
  if (is_weighted())
  {
#ifdef _VERBOSE
    cout << "Writing the A array " << endl;
#endif
    elements_written = fwrite(W, sizeof(T), num_edges, file);
    if (elements_written != num_edges)
    {
#ifdef _VERBOSE
      cout << "Could not write the array of weights correctly. Expected to write " << num_edges << " ints but wrote " << elements_written << ". Returning code -5\n";
#endif
      delete numVertices;
      delete numedges;
      return -6;
    }
  }
#ifdef _VERBOSE
  cout << "Done!";
#endif
  fclose(file);

  delete numVertices;
  delete numedges;
  return 1;
}

template <class T>
CSR<T> *CSR<T>::coarsen_with_multi_edge_collapse(const char * heuristic, unsigned long long sc_threshold)
{
  unsigned long long num_threads = omp_get_max_threads();
#ifdef MEASURE_TIME
  double start = omp_get_wtime();
  double jaccard = 0;
#endif
  unsigned long long SC_THRESHOLD = sc_threshold;
  //float *heuristic_edges = NULL;
  if (heuristic == JACCARD)
  {
    //heuristic_edges = get_jaccard_edges();
#ifdef MEASURE_TIME
    jaccard = omp_get_wtime();
    cout << "Jaccards calculated in " << double(jaccard - start) << " seconds" << endl;
#endif
  }
  else
  {
#ifdef MEASURE_TIME
    jaccard = start;
#endif
  }

#ifdef ORDERED

  //sort vertices according to edge number (count sort)
  T *edge_count = new T[num_vertices + 1]();
  T *mrkr = new T[num_vertices]();
  for (unsigned long long i = 0; i < num_vertices; i++)
  {
    edge_count[(V[i + 1] - V[i]) + 1]++;
  }

  for (unsigned long long i = 1; i <= num_vertices; i++)
  {
    edge_count[i] += edge_count[i - 1];
  }

  T *sorted_ordering = new T[num_vertices];
  initialize_array(sorted_ordering, (T)-1, num_vertices);
  for (unsigned long long i = 0; i < num_vertices; i++)
  {
    T ec = edge_count[V[i + 1] - V[i]];
    sorted_ordering[ec + mrkr[ec]] = i;
    mrkr[ec]++;
  }
  delete[] mrkr;

  unsigned long long num_vertices_c = num_vertices - edge_count[1]; //eleminate isolated edges
  delete[] edge_count;
  //

#endif

  omp_lock_t *vertices_locks = new omp_lock_t[num_vertices];
  for (unsigned long long i = 0; i < num_vertices; i++)
  {
    omp_init_lock(&vertices_locks[i]);
  }
  long long *dont_match = new long long[num_vertices];
  initialize_array(dont_match, (long long)0, num_vertices);

  T *match = new T[num_edges];
  initialize_array(match, (T)-1, num_edges);
  T *matching_vertices = new T[num_vertices];
  initialize_array(matching_vertices, (T)-1, num_vertices);
  unsigned long long matching_vertices_amount = 0;
  unsigned long long *matching_vertices_amount_array = new unsigned long long[num_threads];
  initialize_array(matching_vertices_amount_array, (unsigned long long)0, num_threads);
  T **matching_vertices_arrays = new T *[num_threads];
#pragma omp parallel num_threads(num_threads)
  {
    unsigned long long id = omp_get_thread_num();
    T *&my_matching_vertices_array = matching_vertices_arrays[id];
    my_matching_vertices_array = new T[num_vertices];
    unsigned long long &my_matching_vertices_amount = matching_vertices_amount_array[id];
#pragma omp for schedule(dynamic, 32)
#ifdef ORDERED
    for (T k = 0; k < num_vertices_c /*&& MA < 0.8 * num_vertices_c*/; k++)
    {
      T v = sorted_ordering[num_vertices - 1 - k];
#else
    for (T v = 0; v < num_vertices /*&& MA < 0.8 * num_vertices_c*/; v++)
    {
#endif
      omp_set_lock(&vertices_locks[v]);
      if (dont_match[v] == false)
      {
        dont_match[v] = true;
        my_matching_vertices_array[my_matching_vertices_amount++] = v;
        unsigned long long match_amount = 0;
        for (unsigned long long i = V[v]; i < V[v + 1] && match_amount < SC_THRESHOLD; i++)
        {
          T nbr = E[i];
#ifdef OPTIMIZED
          if (match[V[nbr]] == (T)-1 && V[nbr] < V[nbr + 1] && V[nbr + 1] - V[nbr] < ((num_edges / num_vertices)+1))
          {
#else
          if (match[V[nbr]] == (T)-1 && V[nbr] < V[nbr + 1])
          {
#endif
            if (dont_match[nbr] == false)
            {

              while (!omp_test_lock(&vertices_locks[nbr]))
                {
                  omp_unset_lock(&vertices_locks[v]);
                  omp_set_lock(&vertices_locks[v]);
                }

              if (dont_match[nbr] == false) {
                dont_match[nbr] = true;
                match[V[v] + match_amount] = nbr;
                match[V[nbr]] = v;
                match_amount++;
              }
              
              omp_unset_lock(&vertices_locks[nbr]);
            }

          }
        }
      }
      omp_unset_lock(&vertices_locks[v]);
    }
  }
  for (unsigned long long i = 0; i < num_vertices; i++)
  {
    omp_destroy_lock(&vertices_locks[i]);
  }
  delete[] vertices_locks;
  delete[] dont_match;
#ifdef ORDERED
  delete[] sorted_ordering;
#endif

#ifdef MEASURE_TIME
  double merge_start = omp_get_wtime();
#endif
  for (unsigned long long i = 0; i < num_threads; i++)
  {
    memcpy(matching_vertices + matching_vertices_amount, matching_vertices_arrays[i], matching_vertices_amount_array[i] * sizeof(T));
    matching_vertices_amount += matching_vertices_amount_array[i];
  }

  delete[] matching_vertices_amount_array;
  for (unsigned long long i = 0; i < num_threads; i++)
  {
    delete[] matching_vertices_arrays[i];
  }
  delete[] matching_vertices_arrays;

#ifdef MEASURE_TIME
  double matchings = omp_get_wtime();
  cout << "Matchings calculated in " << double(matchings - jaccard) << " seconds" << endl;
#endif

  T *map = new T[num_vertices];
  initialize_array(map, (T)-1, num_vertices);
#pragma omp prallel for num_threads(num_threads) schedule(guided, 32)
  for (T k = 0; k < matching_vertices_amount; k++)
  {
    T v = matching_vertices[k];

    if (match[V[v]] == (T)-1)
    {
      map[v] = k;
    }
    else
    {
      map[v] = k;
      for (unsigned long long i = 0; i < SC_THRESHOLD && i < (V[v + 1] - V[v]) && match[V[v] + i] != (T)-1; i++)
      {
        map[match[V[v] + i]] = k;
      }
      // new_v_count++;
    }
  }
#ifdef MEASURE_TIME
  double maps = omp_get_wtime();
  cout << "Map generated in " << double(maps - matchings) << " seconds" << endl;
#endif

  //create and return coarsened graph
  unsigned long long *new_e_counters = new unsigned long long[num_threads];
  T **local_n_Es = new T *[num_threads];
  T *n_V = new T[matching_vertices_amount + 1];
  n_V[0] = 0;
#pragma omp parallel num_threads(num_threads)
  {
    unsigned long long id = omp_get_thread_num();
    unsigned long long &local_new_e_count = new_e_counters[id];
    local_new_e_count = 0;
    
    T *local_edge_marker = (T *)aligned_malloc(num_vertices * sizeof(T), ALIGNMENT);
    initialize_array(local_edge_marker, (T)0, num_vertices);

    unsigned long long num_n_v = matching_vertices_amount / num_threads + 1;
    if (matching_vertices_amount % num_threads != 0 && id == num_threads - 1)
      num_n_v += matching_vertices_amount % num_threads;
    
    T *&local_n_E = local_n_Es[id];
    local_n_E = (T *)aligned_malloc(num_edges * sizeof(T), ALIGNMENT);
    unsigned long long start = id * (matching_vertices_amount / num_threads);
    unsigned long long end = (id + 1) * (matching_vertices_amount / num_threads);
    if (matching_vertices_amount % num_threads != 0 && id == num_threads - 1)
      end += matching_vertices_amount % num_threads;

    unsigned long long v = start;
    for (T k = start; k < end; k++)
    {
      T i = matching_vertices[k];
      unsigned long long copied_edge_amount = copy_coarsed_vertex_edges(v, i, 0, local_new_e_count, local_n_E, map, local_edge_marker);
      for (unsigned long long j = 0; j < (V[i + 1] - V[i]) && match[V[i] + j] != UINT_MAX; j++)
      {                                                                          //NEW
        copied_edge_amount = copy_coarsed_vertex_edges(v, match[V[i] + j], copied_edge_amount, local_new_e_count, local_n_E, map, local_edge_marker); //NEW
      }
      local_new_e_count += copied_edge_amount;
      n_V[++v] = local_new_e_count;
    }
    delete[] local_edge_marker;
  }

  T *n_E = new T[num_edges];
  double start_merge = omp_get_wtime();
#pragma omp parallel num_threads(num_threads)
  {
    unsigned long long id = omp_get_thread_num();
    unsigned long long offset = 0;
    for (unsigned long long i = 0; i < id; i++)
    {
      offset += new_e_counters[i];
    }
    unsigned long long start = id * (matching_vertices_amount / num_threads);
    unsigned long long end = (id + 1) * (matching_vertices_amount / num_threads);

    if (matching_vertices_amount % num_threads != 0 && id == num_threads - 1)
      end += (matching_vertices_amount % num_threads);
    for (unsigned long long j = start + 1; j < end + 1; j++)
    {
      n_V[j] += offset;
    }
    memcpy(n_E + offset, local_n_Es[id], new_e_counters[id] * sizeof(T));
  }

#ifdef MEASURE_TIME
  double coarsed = omp_get_wtime();
  cout << "Coarsed graph generated in " << double(coarsed - maps) << " seconds" << endl;
  cout << "Total coarsening time: " << double(coarsed - start) << " seconds" << endl;
#endif

  CSR<T> *coarsed_csr = new CSR<T>(matching_vertices_amount, n_V[matching_vertices_amount], n_V, n_E, NULL, map);

  for (unsigned long long i = 0; i < num_threads; i++)
  {
    delete[] local_n_Es[i];
  }
  delete[] local_n_Es;
  delete[] new_e_counters;
  return coarsed_csr;
}

template <class T>
unsigned long long CSR<T>::copy_coarsed_vertex_edges(T v, T i, unsigned long long copied_edge_amount, T local_new_e_count, T *n_E, T *map, T *marker)
{
  for (unsigned long long j = V[i]; j < V[i + 1]; j++)
  {
    if (V[E[j]] < V[E[j] + 1] && map[E[j]] != v && marker[map[E[j]]] != v + 1)
    {
      marker[map[E[j]]] = v + 1;
      n_E[local_new_e_count + copied_edge_amount] = map[E[j]];
      copied_edge_amount++;
    }
  }
  return copied_edge_amount;
}

/*
template <class T>
bool CSR<T>::is_merged_neighbor_copied(T v, unsigned int copied_edge_amount, T * n_V, T * n_E, T old_edge_map){
  for(unsigned int k = 0; k < copied_edge_amount; k++){
    if(n_E[n_V[v]+k] == old_edge_map){
      return true;
    }
  }
  return false;
}
*/

/**/
template <class T>
float *CSR<T>::get_jaccard_edges()
{
  float *jaccard_edges = new float[num_edges];

  T *e_marker = new T[num_vertices];
  initialize_array(e_marker, (unsigned int)-1, num_vertices);

  for (T i = 0; i < num_vertices; i++)
  { //for each vertex
    //unsigned int edge_amount = V[i + 1] - V[i];

    //mark neighbors
    for (unsigned int j = V[i]; j < V[i + 1]; j++)
    { //for each vertex edge
      unsigned int nbr = E[j];
      e_marker[nbr] = i;
    }
    //

    for (unsigned int j = V[i]; j < V[i + 1]; j++)
    { //for each vertex edge
      unsigned int nbr = E[j];
      if (nbr > i)
      {
        int intersection = 0;
        for (unsigned int k = V[nbr]; k < V[nbr + 1]; k++)
        { //calculate jaccard
          unsigned int nbr2 = E[k];
          if (e_marker[nbr2] == i)
          {
            intersection++;
          }
        }
        jaccard_edges[j] = ((float)intersection) / ((V[i + 1] - V[i]) + (V[nbr + 1] - V[nbr]) - intersection);
      }
      else
      { //already calculated jaccards
        for (unsigned int k = V[nbr]; k < V[nbr + 1]; k++)
        {
          if (E[k] == i)
          {
            jaccard_edges[j] = jaccard_edges[k];
          }
        }
      }
    }
  }
  delete[] e_marker;
  return jaccard_edges;
}

/*
template <class T>
float CSR<T>::get_jaccard(T v, T u, unsigned int v_edge_amount){
  unsigned int intersection_size = 0;
  unsigned int u_edge_amount = V[u+1] - V[u];
  for(unsigned int i = 0; i < v_edge_amount; i++){
    for(unsigned int j = 0; j < u_edge_amount; j++){
      if(E[V[v] + i] == E[V[u] + j]){
  intersection_size++;
      }
    } 
  }
  return (float)(intersection_size)/(float)(v_edge_amount + u_edge_amount - intersection_size);
}
*/

template <class T>
T CSR<T>::get_jaccard_matching(T v, T size, float *jaccard_edges, T *match)
{
  T best = -1;
  float best_val = 0;
  for (T i = 0; i < size; i++)
  {
    T edge = V[v] + i;
    if (jaccard_edges[edge] > best_val && match[E[edge]] == -1)
    {
      best = E[edge];
      best_val = jaccard_edges[edge];
    }
  }
  return best;
}

template <class T>
void CSR<T>::construct_ps_alias_table_gv(){
  v_alias.alias_p = new char[num_vertices];
  v_alias.alias_v = new T[num_vertices];
  v_alias.size = num_vertices;
  double * probs= new double [num_vertices];
  for (int i =0 ; i<num_vertices; i++){
    unsigned long long ne = V[i+1]-V[i]>0 ? V[i+1]-V[i] : 0;
    if (ne>num_edges){
      printf("bad edges %d %lld\n", i, ne);
    }
    probs[i] = ne;
  }
  construct_alias_table_gv(v_alias.alias_v, v_alias.alias_p, probs, v_alias.size);
}

template <class T>
void CSR<T>::construct_ns_alias_table_gv(float factor){
  ns_alias.alias_p = new char[num_vertices];
  ns_alias.alias_v = new T[num_vertices];
  ns_alias.size = num_vertices;
  double * probs = new double[num_vertices];
  for (int i =0; i<num_vertices; i++){
    double ne = V[i+1]-V[i]>0 ? V[i+1]-V[i] : 0;
    probs[i] = pow(ne, factor);
    //if (ne == 0) printf("%f\n", probs[i]);
    //if (i%100 == 0) printf("i %d probs[i] %f\n", i, probs[i]);
  }
  construct_alias_table_gv(ns_alias.alias_v, ns_alias.alias_p, probs, ns_alias.size);
}

template <class T>
void CSR<T>::construct_alias_table_gv(T *& alias_v, char *& alias_p, double * probs, unsigned long long size){
  double norm=0;
  for (int i =0 ; i<num_vertices; i++){
    norm+=probs[i];
  }
  // double norm = 0;
  // for (int i =0; i<num_vertices; i++){
  //   norm+= probs[i];
  // }
  norm = norm/num_vertices;
  for (int i =0; i<num_vertices; i++){
    probs[i]/=norm;
  }

  queue<T> large, little;
  for (int i = 0; i < num_vertices; i++) {
      if (probs[i] < 1)
          little.push(i);
      else
          large.push(i);
  }
  while (!little.empty() && !large.empty()) {
      T i = little.front(), j = large.front();
      little.pop();
      large.pop();
      alias_v[i] = j;
      probs[j] = probs[i] + probs[j] - 1;
      if (probs[j] < 1)
          little.push(j);
      else
          large.push(j);
  }
  // suppress some trunction error
  while (!little.empty()) {
      T i = little.front();
      little.pop();
      alias_v[i] = i;
  }
  while (!large.empty()) {
      T i = large.front();
      large.pop();
      alias_v[i] = i;
  }
    // making probs in the range 100
    for (int i = 0; i<num_vertices; i++){
      alias_p[i] =  int(probs[i]*100)%101;
      if (alias_p[i]>100){
        printf("What even is this %f %d\n", probs[i], alias_p[i] );
      }
    }
    for (int i =0; i<num_vertices; i++){
      if (alias_v[i]>=num_vertices){
        printf("vbad aliasv %d alias %d", i, alias_v[i]);
      }
    }
    delete [] probs;
}


template <class T>
void CSR<T>::construct_alias_table_line()
{
  v_alias.alias_p = new char[num_vertices];
  v_alias.alias_v = new T[num_vertices];
  v_alias.size = num_vertices;

  float *norm_prob = new float[num_vertices];
  T *large_block = new T[num_vertices];
  T *small_block = new T[num_vertices];

  float sum = 0;
  T cur_small_block, cur_large_block;
  T num_small_block = 0, num_large_block = 0;
  T add;
  for (T k = 0; k != num_vertices; k++)
  {
    add = V[k + 1] - V[k] > 0 ? V[k + 1] - V[k] : 0;
    sum += add;
  }
  for (T k = 0; k != num_vertices; k++)
  {
    add = V[k + 1] - V[k] > 0 ? V[k + 1] - V[k] : 0;
    norm_prob[k] = add * num_vertices / sum;
  }

  for (long long k = num_vertices - 1; k >= 0; k--)
  {
    if (norm_prob[k] < 1)
      small_block[num_small_block++] = k; // adding to L
    else
      large_block[num_large_block++] = k;
  }

  while (num_small_block && num_large_block)
  {
    cur_small_block = small_block[--num_small_block];
    cur_large_block = large_block[--num_large_block];
    v_alias.alias_p[cur_small_block] = int(100*norm_prob[cur_small_block]);
    v_alias.alias_v[cur_small_block] = cur_large_block;
    norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
    if (norm_prob[cur_large_block] < 1)
      small_block[num_small_block++] = cur_large_block;
    else
      large_block[num_large_block++] = cur_large_block;
  }

  while (num_large_block)
    v_alias.alias_p[large_block[--num_large_block]] = 1;
  while (num_small_block)
    v_alias.alias_p[small_block[--num_small_block]] = 1;

  delete[] norm_prob;
  delete[] small_block;
  delete[] large_block;
}

/*
template <class T>
T CSR<T>::get_random_matching(T v, T size, T * match){
  for(T i = 0; i < size; i++){
    T edge = V[v] + (randn() % size);
    if(match[E[edge]] == -1){
      return E[edge];
    }
  }
  return -1;
}
*/
