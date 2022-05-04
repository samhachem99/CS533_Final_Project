//Simple Linked List Structure For Tracking Pre-Allocated Data

#ifndef PRE_ALLOC_MEM
#define PRE_ALLOC_MEM

#pragma GCC diagnostic ignored "-Wnullability-completeness"

int init_pre_alloc_mem(char *ptr, long num_bytes);
int get_pre_alloc_mem(long num_bytes, char **ptr);
int free_pre_alloc_mem(char *ptr);
int free_all_pre_alloc_mem();

//some simple functions used only for testing
int test_pre_alloc_internal_check();
long test_pre_alloc_get_free();
long test_pre_alloc_get_used();
int test_print_LL();

#endif //PRE_ALLOC_MEM