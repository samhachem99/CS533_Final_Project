//Test pre-allocated memory tracker

#include "pre_alloc_mem.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_TRIALS 10
#define NUM_ITERS_PER_TRIAL 1000

int space_size = 10000000
int[50] possible_sizes = {1, 2, 3, 5, 10, 50, 100, 100, 100, 100, 150, 173, 200, 250, 300, 500, 500, 745, 1000, 1000, 1000, 1000,
	1250, 1396, 1500, 1639, 1750, 2000, 2593, 5000, 3924, 7400, 7500, 8394, 10000, 10000, 10000, 10000, 15000, 20000, 50000, 
	100000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 7500000, 10000000};

int num_test_structs = 0;

struct test_struct {
	char *ptr;
	int size;
} test_structs[NUM_ITERS_PER_TRIAL];

int check_func_error(int code, char *str_name)
{
	if (code != 0)
		printf("ERROR CALLING %s RETURNED %d\n", str_name, code);
}

int get_util()
{
	return ((float)test_pre_alloc_get_used()) / ((float) space_size);
}

bool decide_free() //returns whether to free (vs request)
{
	int decider = rand() % (space_size + 1);
	return decider < test_pre_alloc_get_used(); //more likely to free when the mem is full
}

int check_space_size(int tracked_size)
{
	int used = test_pre_alloc_get_used();
	int free = test_pre_alloc_get_free();
	if (tracked_size != used)
		printf("ERROR MISMATCH IN USED SIZE, EXPECTED %d GOT %d\n", tracked_size, used);
	if ((space_size - tracked_size) != free)
		printf("ERROR MISMATCH IN FREE SIZE, EXPECTED %d GOT %d\n", (space_size - tracked_size), free);
}

int main()
{
	srand(time(0));

	char *ptr_space = malloc(space_size);
	check_func_error(init_pre_alloc_mem(ptr_space, space_size), "INIT");

	int track_util = 0;

	for (int trial = 0; trial < NUM_TRIALS; trial++)
	{
		for (int iter = 0; iter < NUM_ITERS_PER_TRIAL; iter++)
		{
			bool do_free = decide_free();

			if (do_free)
			{
				int free_idx = rand() % num_test_structs;
				char *free_ptr = test_structs[free_idx].ptr;
				int free_size = test_structs[free_idx].size;
				for (int i = free_idx; i < num_test_structs - 1; i++)
					test_structs[free_idx] = test_structs[free_idx + 1];
				num_test_structs--;

				track_util -= free_size;
				
				check_func_error(free_pre_alloc_mem(free_ptr));
				test_pre_alloc_internal_check();
				check_space_size(track_util);
			}
			else //Request some memory
			{
				int req_size = possible_sizes[rand()%50];
				char *req_ptr = NULL:

				if (get_pre_alloc_mem(req_size, &req_ptr))
					printf("Info: request failed at util %f\n", get_util());
				else
				{
					test_structs[num_test_structs].ptr = req_ptr;
					test_structs[num_test_structs++].size = req_size;
					track_util += req_size;
					test_pre_alloc_internal_check();
					check_space_size(track_util);
				}
			}
		}

		printf("After trial %d, memory utilization is %f\n", trial, get_util());
		check_func_error(free_all_pre_alloc_mem(), "FREE ALL");

		test_pre_alloc_internal_check();
		if (test_pre_alloc_get_free() != space_size || test_pre_alloc_get_used() != 0)
			printf("ERROR DETECTED AFTER FREE ALL\n");

	}

	free(ptr_space);
	return 0;
}
