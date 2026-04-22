#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void queue_init(ptr_queue_t *q)
{
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->cv, NULL);
}

void queue_close(ptr_queue_t *q)
{
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->cv);
    pthread_mutex_unlock(&q->mu);
}

void queue_push(ptr_queue_t *q, void *ptr)
{
    qnode_t *n = calloc(1, sizeof(*n));
    if (!n) {
        perror("calloc qnode");
        exit(1);
    }
    n->ptr = ptr;

    pthread_mutex_lock(&q->mu);
    if (q->tail) {
        q->tail->next = n;
    } else {
        q->head = n;
    }
    q->tail = n;
    pthread_cond_signal(&q->cv);
    pthread_mutex_unlock(&q->mu);
}

void *queue_pop(ptr_queue_t *q)
{
    pthread_mutex_lock(&q->mu);

    while (!q->head && !q->closed) {
        pthread_cond_wait(&q->cv, &q->mu);
    }

    if (!q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }

    qnode_t *n = q->head;
    q->head = n->next;
    if (!q->head) {
        q->tail = NULL;
    }

    pthread_mutex_unlock(&q->mu);

    void *p = n->ptr;
    free(n);
    return p;
}
