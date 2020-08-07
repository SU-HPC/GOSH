#include <stdlib.h>
#include <cstddef>
#ifndef _ALIGN_FUNCTIONS
#define _ALIGN_FUNCTIONS
#define ALIGNMENT 128
inline void* aligned_malloc(size_t size, size_t align){
    void *result;
    #ifdef _MSC_VER 
    result = _aligned_malloc(size, align);
    #else 
     if(posix_memalign(&result, align, size)) result = 0;
    #endif
    return result;
}
inline void aligned_free(void *ptr) {
    #ifdef _MSC_VER 
        _aligned_free(ptr);
    #else 
      free(ptr);
    #endif

}
#endif
