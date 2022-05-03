//Simple Linked List Structure For Tracking Pre-Allocated Data

#ifndef PRE_ALLOC_MEM
#define PRE_ALLOC_MEM

int init_pre_alloc_mem(char *ptr, int num_bytes);
int get_pre_alloc_mem(int num_bytes, char **ptr);
int free_pre_alloc_mem(char *ptr);
int free_all_pre_alloc_mem();

#endif //PRE_ALLOC_MEM