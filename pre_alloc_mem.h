//Simple Linked List Structure For Tracking Pre-Allocated Data

#ifndef PRE_ALLOC_MEM
#define PRE_ALLOC_MEM

int init_pre_alloc_mem(char *ptr, int num_bytes); //NOTE IF WE'RE USING MORE THAN 4.9 GB NUM_BYTES NEEDS TO CHANGE TO LONG
int get_pre_alloc_mem(int num_bytes, char **ptr);
int free_pre_alloc_mem(char *ptr);
int free_all_pre_alloc_mem();

//some simple test functions
int test_pre_alloc_internal_check();
int test_pre_alloc_get_free();
int test_pre_alloc_get_used();

#endif //PRE_ALLOC_MEM