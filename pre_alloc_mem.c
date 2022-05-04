//Simple Linked List Structure For Tracking Pre-Allocated Data

#include "pre_alloc_mem.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
	char *ptr;
	char free;
	long size;
	struct Node *next;
} Node;

Node *head = NULL;
int num_nodes = 0;

void free_all_recurse(Node *n)
{
	if (NULL == n)
		return;

	free_all_recurse(n->next);
	free(n); //It's not our responsibility to free n->ptr
}

int init_pre_alloc_mem(char *ptr, long num_bytes)
{
	if (num_bytes <= 0)
		return 1; //fail

	if (head != NULL)
		free_all_pre_alloc_mem();

	head = malloc(sizeof(Node));
	head->ptr = ptr;
	head->free = 1;
	head->size = num_bytes;
	head->next = NULL;
	num_nodes = 1;

	return 0; //success
}

int get_pre_alloc_mem(long num_bytes, char **ptr)
{
	*ptr = NULL;

	if (num_bytes <= 0)
		return 1; //fail

	if (NULL == head)
		return 1; //fail

	//Walk the LL to find smallest big-enough free Node
	Node *best = NULL;
	Node *curr = head;
	while (curr != NULL)
	{
		if (1 == curr->free && curr->size >= num_bytes &&
			(NULL == best || curr->size < best->size))
			best = curr;

		curr = curr->next;
	}

	if (NULL == best)
		return 1; //fail

	if (best->size == num_bytes)
		best->free = 0;
	else
	{
		Node *new_node = malloc(sizeof(Node));
		new_node->ptr = best->ptr + num_bytes;
		new_node->free = 1;
		new_node->size = best->size - num_bytes;
		new_node->next = best->next;
		num_nodes++;
		
		best->free = 0;
		best->size = num_bytes;
		best->next = new_node;

	}

	*ptr = best->ptr;
	return 0; //success
}

int free_pre_alloc_mem(char *ptr) //must pass in the ptr to the beginning of the region
{
	if (NULL == head)
		return 1; //fail

	//Walk the LL to find ptr
	Node *curr = head;
	Node *prev = NULL;
	while (curr != NULL)
	{
		if (curr->ptr == ptr)
			break;

		prev = curr;
		curr = curr->next;
	}

	if (curr == NULL) //not found
		return 1; //fail

	//Case 1: previous and next are both free (combine all 3)
	if (prev && curr->next && prev->free && curr->next->free)
	{
		prev->size += (curr->size + curr->next->size);
		prev->next = curr->next->next;
		free(curr);
		free(curr->next);
		num_nodes -= 2;
		return 0; //success
	}

	//Case 2: combine prev and curr
	if (prev && prev->free)
	{
		prev->size += curr->size;
		prev->next = curr->next;
		free(curr);
		num_nodes--;
		return 0; //success
	}

	//Case 3: combine curr and next
	if (curr->next && curr->next->free)
	{
		curr->next->ptr = curr->ptr;
		curr->next->size += curr->size;
		if (prev)
			prev->next = curr->next;
		else
			head = curr->next;
		free(curr);
		num_nodes--;
		return 0; //success
	}

	//Case 4: Just mark curr free
	curr->free = 1;
	return 0; //success
}

int free_all_pre_alloc_mem()
{
	free_all_recurse(head);
	head = NULL;
	num_nodes = 0;
	return 0; //success
}

//Test Functions
int test_pre_alloc_internal_check()
{
	int num_err = 0, num_nodes_test = 0;

	Node *curr = head, *prev = NULL;
	while (curr != NULL)
	{
		num_nodes_test++;

		if (prev && prev->free && curr->free)
		{
			printf("ERROR: CONSECUTIVE FREE NODES PRESENT\n");
			num_err++;
		}

		prev = curr;
		curr = curr->next;
	}

	if (num_nodes != num_nodes_test)
	{
		printf("ERROR: STRUCTURE EXPECTED %d NODES BUT LIST ONLY HAS %d\n", num_nodes, num_nodes_test);
		num_err++;
	}

	return num_err;
}

long test_pre_alloc_get_free()
{
	long free_size = 0;
	Node *curr = head;
	while (curr != NULL)
	{
		if (curr->free)
			free_size += curr->size;

		curr = curr->next;
	}

	return free_size;
}

long test_pre_alloc_get_used()
{
	long used_size = 0;
	Node *curr = head;
	while (curr != NULL)
	{
		if (!curr->free)
			used_size += curr->size;

		curr = curr->next;
	}

	return used_size;
}

int test_print_LL()
{
	Node *curr = head;
	int idx = 0;
	printf("----------LL----------\n");
	while (curr != NULL)
	{
		if (curr->free)
			printf("NODE %3d: SIZE%8ld, FREE --> %#x\n", idx, curr->size, (unsigned int) curr->ptr);
		else
			printf("NODE %3d: SIZE%8ld, USED --> %#x\n", idx, curr->size, (unsigned int) curr->ptr);

		curr = curr->next;
		idx++;
	}

	printf("--------END LL--------\n");
	return 0;
}
