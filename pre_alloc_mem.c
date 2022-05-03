//Simple Linked List Structure For Tracking Pre-Allocated Data

#include "pre_alloc_mem.h"

typedef struct Node {
	char *ptr;
	bool free;
	int size;
	Node next;
} Node;

Node *head = NULL;

void free_all_recurse(Node *n)
{
	if (NULL == n)
		return;

	free_all_recurse(n->next);
	free(n); //It's not our responsibility to free n->ptr
}

int init_pre_alloc_mem(char *ptr, int num_bytes)
{
	if (head != NULL)
		free_all_pre_alloc_mem();

	head = malloc(sizeof(Node));
	head->ptr = ptr;
	head->free = true;
	head->size = num_bytes;
	head->next = NULL;

	return 0; //success
}

int get_pre_alloc_mem(int num_bytes, char **ptr)
{
	*ptr = NULL;

	if (NULL == head)
		return 1; //fail

	//Walk the LL to find smallest big-enough free Node
	Node *best = NULL;
	Node *curr = head;
	while (curr != NULL)
	{
		if (true == curr->free && curr->size >= num_bytes &&
			(NULL == best || curr->size < best->size))
			best = curr;

		curr = curr->next;
	}

	if (NULL == best)
		return 1; //fail

	if (best->size == num_bytes)
		best->free = false;
	else
	{
		Node *new_node = malloc(sizeof(Node));
		new_node->ptr = best->ptr + num_bytes;
		new_node->free = true;
		new_node->size = best->size - num_bytes;
		new_node->next = best->next;
		
		best->free = false;
		best->size = num_bytes;
		best->next = new_node;
	}

	*ptr = best;
	return 0; //success
}

int free_pre_alloc_mem(char *ptr)
{
	if (NULL == head)
		return 1; //fail

	//Walk the LL to find ptr
	Node *curr = head;
	Node *prev = NULL;
	while (curr != NULL)
	{
		if (curr == ptr)
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
		return 0; //success
	}

	//Case 2: combine prev and curr
	if (prev && prev->free)
	{
		prev->size += curr->size;
		prev->next = curr->next;
		free(curr);
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
		return 0; //success
	}

	//Case 4: Just mark curr free
	curr->free = true;
	return 0; //success
}

int free_all_pre_alloc_mem()
{
	free_all_recurse(head);
	head = NULL;
	return 0; //success
}
