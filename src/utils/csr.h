#ifndef _CSR_H
#define _CSR_H
//#define _VERBOSE
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
using namespace std; 
// class definition of CSR                                                                                                                                                  // This class encapsulates all the data members required for a single weighted or unweighted graph stored in a Compressed Sparse Row matrix format
// It is a templated class where the data type of the vertices IDs is dependant on initialization (for storage optimization purposes) 
typedef long long ll;  
template <class T>   
struct alias_table{
  ll size;
  T* alias_v;
  char* alias_p;
};                
template <class T>
class CSR       
{
  private:
    //static constexpr bool weighted = false;
    //static constexpr bool directed = false;
    int read_bcsr(string fname);  
    //int mtx2csr(char *fname);  
    int mtx2csr(string fname, bool directed, bool weighted);  
    int bmtx2csr(string fname, bool directed, bool weighted);  
    size_t sizeOfUnit;

    void construct_alias_table_gv(T*&, char*&, double*, unsigned long long);
    void construct_alias_table_line();
    void construct_ns_alias_table_gv(float);
    void construct_ps_alias_table_gv();

    void skip_mtx_comments(ifstream &file);

    //jaccard
    float * get_jaccard_edges();
    //float get_jaccard(T v, T u, unsigned int v_edge_amount);
    //
    
    //matchings
    T get_jaccard_matching(T v, T size, float * jaccard_edges, T * match);
    //T get_random_matching(T v, T size, T * match);
    //
    
    //bool is_merged_neighbor_copied(T v, unsigned int copied_edge_amount, T * n_V, T * n_E, T old_edge_map);
    unsigned long long copy_coarsed_vertex_edges(T v, T i, unsigned long long copied_edge_amount, T local_new_e_count, T * n_E, T * map, T * marker);

 public:  
    T *V; // the array of vertices
    T *E; // the array of edges    
    T *W;
    T * map;
    alias_table<T> v_alias, ns_alias;
    ll num_vertices = 0;
    ll num_edges = 0;  
    int alias_table_imp = 0;
    CSR(ll nv, ll ne, T *V, T *E, T *W = NULL, T *map = NULL);
    CSR(CSR &); 
    CSR(string fname, int ati = 0);
    CSR(string fname, bool directed, bool weighted, bool binary = false, int ati=0);
    ~CSR(); 
    bool is_weighted() const;  
    T get_correct_edge_index(T v, T * edge_place);
    int write_bcsr(string fname);
    CSR<T> * coarsen_with_multi_edge_collapse(const char * heuristic, unsigned long long level);
};
#include "csr.cpp"
#endif
