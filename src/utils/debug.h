#ifdef _DEBUG
#define TM_PRINTF(print_flag ,f_, ...) if (print_flag) printf((f_), ##__VA_ARGS__)
#else
#define TM_PRINTF(print_flag ,f_, ...)
#endif
