//Test pre-allocated memory tracker

#include "pre_alloc_mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POSSIBLE_SIZES 51
#define NUM_TRIALS 1000
#define NUM_ITERS_PER_TRIAL 10000
#define VERBOSE 0

long space_size = 100000000;
long possible_sizes[NUM_POSSIBLE_SIZES] = {1, 2, 3, 5, 10, 50, 100, 100, 100, 100, 150, 173, 200, 250, 300, 500, 500, 745, 1000, 1000, 1000, 1000,
	1250, 1396, 1500, 1639, 1750, 2000, 2593, 5000, 3924, 7400, 7500, 8394, 10000, 10000, 10000, 10000, 15000, 20000, 50000, 
	100000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 7500000, 10000000, 50000000};

long num_test_structs = 0;

struct test_struct {
	char *ptr;
	long size;
} test_structs[NUM_ITERS_PER_TRIAL];

int check_func_error(int code, char *str_name, unsigned int info)
{
	if (code != 0)
		printf("ERROR CALLING %s RETURNED %d, INFO %#x\n", str_name, code, info);
	return 0;
}

float get_util()
{
	return ((float)test_pre_alloc_get_used()) / ((float) space_size);
}

char decide_free() //returns whether to free (vs request)
{
	long decider = ((long)rand()) % (space_size + 1);
	return decider < (test_pre_alloc_get_used() ? 1 : 0; //more likely to free when the mem is full
}

int check_space_size(long tracked_size)
{
	long used = test_pre_alloc_get_used();
	long free = test_pre_alloc_get_free();
	if (tracked_size != used)
		printf("ERROR MISMATCH IN USED SIZE, EXPECTED %ld GOT %ld\n", tracked_size, used);
	if ((space_size - tracked_size) != free)
		printf("ERROR MISMATCH IN FREE SIZE, EXPECTED %ld GOT %ld\n", (space_size - tracked_size), free);
	return 0;
}

int main()
{
	srand(time(0));

	char *ptr_space = malloc(space_size);
	check_func_error(init_pre_alloc_mem(ptr_space, space_size), "INIT", 0);

	long track_util = 0;

	if (VERBOSE)
	{
		printf("START\n");
		test_print_LL();
	}

	for (int trial = 0; trial < NUM_TRIALS; trial++)
	{
		for (int iter = 0; iter < NUM_ITERS_PER_TRIAL; iter++)
		{
			char do_free = decide_free();

			if (do_free)
			{
				if (num_test_structs)
				{
					long free_idx = ((long)rand()) % num_test_structs;
					char *free_ptr = test_structs[free_idx].ptr;
					long free_size = test_structs[free_idx].size;
					for (int i = free_idx; i < num_test_structs - 1; i++)
						test_structs[i] = test_structs[i + 1];
					num_test_structs--;
					track_util -= free_size;

					check_func_error(free_pre_alloc_mem(free_ptr), "FREE", (unsigned int) free_ptr);
					test_pre_alloc_internal_check();
					check_space_size(track_util);

					if (VERBOSE)
					{
						printf("FREE %#x\n", (unsigned int) free_ptr);
						test_print_LL();
					}
				}
			}
			else //Request some memory
			{
				long req_size = possible_sizes[rand()%NUM_POSSIBLE_SIZES];
				char *req_ptr = NULL;

				if (get_pre_alloc_mem(req_size, &req_ptr))
				{
					if (VERBOSE)
						printf("Info: request size %ld failed at util %f\n", req_size, get_util());
				}
				else
				{
					if (VERBOSE)
					{
						printf("REQ %#x\n", (unsigned int) req_ptr);
						test_print_LL();
					}
					test_structs[num_test_structs].ptr = req_ptr;
					test_structs[num_test_structs++].size = req_size;
					track_util += req_size;
					test_pre_alloc_internal_check();
					check_space_size(track_util);
				}
			}
		}

		printf("After trial %d, memory utilization is %f\n", trial, get_util());
		check_func_error(free_all_pre_alloc_mem(), "FREE ALL", 0);
		check_func_error(init_pre_alloc_mem(ptr_space, space_size), "RE-INIT", 0);
		track_util = 0;
		num_test_structs = 0;

		test_pre_alloc_internal_check();
		if (test_pre_alloc_get_free() != space_size || test_pre_alloc_get_used() != 0)
		{
			printf("ERROR DETECTED AFTER FREE ALL\n");
			test_print_LL();
		}

	}

	free(ptr_space);
	return 0;
}
