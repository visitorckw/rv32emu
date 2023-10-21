/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cache.h"
#include "mpool.h"
#include "utils.h"

#define MIN(a, b) ((a < b) ? a : b)
#define HASH(val) \
    (((val) * (GOLDEN_RATIO_32)) >> (32 - (cache_size_bits))) & (cache_size - 1)
/* Limit number of HIR blocks is less than 1% of cache capacity. */
#define HIR_RATE 1

/* THRESHOLD is set to identify hot spots. Once the frequency of use for a block
 * exceeds the THRESHOLD, the JIT compiler flow is triggered.
 */
#define THRESHOLD 32768

#if RV32_HAS(JIT)
#define sys_icache_invalidate(addr, size) \
    __builtin___clear_cache((char *) (addr), (char *) (addr) + (size));
#endif

static uint32_t cache_size, cache_size_bits;
static struct mpool *cache_mp;

struct list_head {
    struct list_head *prev, *next;
};

struct hlist_head {
    struct hlist_node *first;
};

struct hlist_node {
    struct hlist_node *next, **pprev;
};

typedef struct {
    void *value;
    uint32_t key;
    uint32_t frequency;
    int resident;
    int isHIR;
    struct list_head lru_list;
    struct list_head hir_list;
    struct hlist_node ht_list;
} lirs_entry_t;


typedef struct {
    struct hlist_head *ht_list_head;
} hashtable_t;

typedef struct cache {
    struct list_head *lists[2];       /* LRU list and HIR list */
    uint32_t list_size;               /* Number of blocks in cache */
    uint32_t lru_list_size;           /* Number of blocks in LRU list */
    uint32_t num_lir;                 /* Number of LIR blocks */
    uint32_t HIR_block_portion_limit; /* The limit of HIR blocks */

    hashtable_t *map;
    uint32_t capacity;
} cache_t;

static inline void INIT_LIST_HEAD(struct list_head *head)
{
    head->next = head;
    head->prev = head;
}

static inline int list_empty(const struct list_head *head)
{
    return (head->next == head);
}

static inline void list_add(struct list_head *node, struct list_head *head)
{
    struct list_head *next = head->next;

    next->prev = node;
    node->next = next;
    node->prev = head;
    head->next = node;
}

static inline void list_del(struct list_head *node)
{
    struct list_head *next = node->next;
    struct list_head *prev = node->prev;

    next->prev = prev;
    prev->next = next;
}

static inline void list_del_init(struct list_head *node)
{
    list_del(node);
    INIT_LIST_HEAD(node);
}

#define list_entry(node, type, member) container_of(node, type, member)

#define list_first_entry(head, type, member) \
    list_entry((head)->next, type, member)

#define list_last_entry(head, type, member) \
    list_entry((head)->prev, type, member)

#ifdef __HAVE_TYPEOF
#define list_for_each_entry_safe(entry, safe, head, member)                \
    for (entry = list_entry((head)->next, __typeof__(*entry), member),     \
        safe = list_entry(entry->member.next, __typeof__(*entry), member); \
         &entry->member != (head); entry = safe,                           \
        safe = list_entry(safe->member.next, __typeof__(*entry), member))
#else
#define list_for_each_entry_safe(entry, safe, head, member, type) \
    for (entry = list_entry((head)->next, type, member),          \
        safe = list_entry(entry->member.next, type, member);      \
         &entry->member != (head);                                \
         entry = safe, safe = list_entry(safe->member.next, type, member))
#endif

#define INIT_HLIST_HEAD(ptr) ((ptr)->first = NULL)

static inline void INIT_HLIST_NODE(struct hlist_node *h)
{
    h->next = NULL;
    h->pprev = NULL;
}

static inline int hlist_empty(const struct hlist_head *h)
{
    return !h->first;
}

static inline void hlist_add_head(struct hlist_node *n, struct hlist_head *h)
{
    struct hlist_node *first = h->first;
    n->next = first;
    if (first)
        first->pprev = &n->next;

    h->first = n;
    n->pprev = &h->first;
}

static inline bool hlist_unhashed(const struct hlist_node *h)
{
    return !h->pprev;
}

static inline void hlist_del(struct hlist_node *n)
{
    struct hlist_node *next = n->next;
    struct hlist_node **pprev = n->pprev;

    *pprev = next;
    if (next)
        next->pprev = pprev;
}

static inline void hlist_del_init(struct hlist_node *n)
{
    if (hlist_unhashed(n))
        return;
    hlist_del(n);
    INIT_HLIST_NODE(n);
}

#define hlist_entry(ptr, type, member) container_of(ptr, type, member)

#ifdef __HAVE_TYPEOF
#define hlist_entry_safe(ptr, type, member)                  \
    ({                                                       \
        typeof(ptr) ____ptr = (ptr);                         \
        ____ptr ? hlist_entry(____ptr, type, member) : NULL; \
    })
#else
#define hlist_entry_safe(ptr, type, member) \
    (ptr) ? hlist_entry(ptr, type, member) : NULL
#endif

#ifdef __HAVE_TYPEOF
#define hlist_for_each_entry(pos, head, member)                              \
    for (pos = hlist_entry_safe((head)->first, typeof(*(pos)), member); pos; \
         pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))
#else
#define hlist_for_each_entry(pos, head, member, type)              \
    for (pos = hlist_entry_safe((head)->first, type, member); pos; \
         pos = hlist_entry_safe((pos)->member.next, type, member))
#endif

/* Pruning LRU list to ensure the bottom of the LRU list is always an LIR block.
 */
static void prune_lru_list(cache_t *cache)
{
    lirs_entry_t *delete_target;
    while (!list_empty(cache->lists[0])) {
        delete_target =
            list_last_entry(cache->lists[0], lirs_entry_t, lru_list);
        if (!delete_target->isHIR)
            break;
        delete_target->resident = 0;
        list_del_init(&delete_target->lru_list);
        cache->lru_list_size--;
    }
}

cache_t *cache_create(int size_bits)
{
    cache_t *cache = malloc(sizeof(cache_t));
    if (!cache)
        return NULL;
    cache_size_bits = size_bits;
    cache_size = 1 << size_bits;

    for (int i = 0; i < 2; i++) {
        cache->lists[i] = malloc(sizeof(struct list_head));
        INIT_LIST_HEAD(cache->lists[i]);
    }

    cache->map = malloc(sizeof(hashtable_t));
    if (!cache->map) {
        free(cache->lists);
        free(cache);
        return NULL;
    }
    cache->map->ht_list_head = malloc(cache_size * sizeof(struct hlist_head));
    if (!cache->map->ht_list_head) {
        free(cache->map);
        free(cache->lists);
        free(cache);
        return NULL;
    }
    for (uint32_t i = 0; i < cache_size; i++) {
        INIT_HLIST_HEAD(&cache->map->ht_list_head[i]);
    }
    cache->list_size = 0;
    cache->lru_list_size = 0;
    cache->num_lir = 0;
    cache->HIR_block_portion_limit = HIR_RATE / 100.0 * cache_size;
    cache_mp =
        mpool_create(cache_size * sizeof(lirs_entry_t), sizeof(lirs_entry_t));

    cache->capacity = cache_size;
    return cache;
}

void *cache_get(cache_t *cache, uint32_t key)
{
    if (hlist_empty(&cache->map->ht_list_head[HASH(key)]))
        return NULL;

    lirs_entry_t *entry = NULL;
#ifdef __HAVE_TYPEOF
    hlist_for_each_entry (entry, &cache->map->ht_list_head[HASH(key)], ht_list)
#else
    hlist_for_each_entry (entry, &cache->map->ht_list_head[HASH(key)], ht_list,
                          lirs_entry_t)
#endif
    {
        if (entry->key == key)
            break;
    }
    if (!entry || entry->key != key)
        return NULL;

    /* Hit the block in HIR list */
    if (entry->isHIR) {
        assert(!list_empty(&entry->hir_list));
        list_del_init(&entry->hir_list);
    } else {
        assert(list_empty(&entry->hir_list));
        assert(!list_empty(&entry->lru_list));
    }

    cache->lru_list_size -= !list_empty(&entry->lru_list);
    list_del_init(&entry->lru_list);
    list_add(&entry->lru_list, cache->lists[0]);
    cache->lru_list_size++;

    /* If a newly referenced block is HIR with resident status,
     * change it to LIR if the LRU list length is greater than its limit.
     * After that, we need to prune the LRU list to ensure the bottom of the LRU
     * list is always an LIR block.
     */
    if (entry->isHIR && entry->resident) {
        assert(cache->num_lir + cache->HIR_block_portion_limit + 1 >=
               cache->capacity);
        entry->isHIR = 0;
        lirs_entry_t *last_lru =
            list_last_entry(cache->lists[0], lirs_entry_t, lru_list);
        last_lru->isHIR = 1;
        last_lru->resident = 0;
        list_del_init(&last_lru->lru_list);
        cache->lru_list_size--;
        list_add(&last_lru->hir_list, cache->lists[1]);

        prune_lru_list(cache);
    } else if (entry->isHIR) {
        list_add(&entry->hir_list, cache->lists[1]);
    }

    entry->resident = 1;
    entry->frequency++;

    assert(cache->list_size <= cache->capacity);
    assert(cache->lru_list_size <= cache->capacity);
    assert(cache->num_lir + cache->HIR_block_portion_limit <= cache->capacity);

    /* return NULL if cache miss */
    return entry->value;
}

void *cache_put(cache_t *cache, uint32_t key, void *value)
{
    void *delete_value = NULL;

    assert(cache->list_size <= cache->capacity);
    /* check the cache is full or not before adding a new entry */
    if (cache->list_size == cache->capacity) {
        lirs_entry_t *delete_target =
            list_last_entry(cache->lists[1], lirs_entry_t, hir_list);
        cache->list_size--;
        cache->lru_list_size -= !list_empty(&delete_target->lru_list);
        list_del_init(&delete_target->lru_list);
        list_del_init(&delete_target->hir_list);
        hlist_del_init(&delete_target->ht_list);
        delete_value = delete_target->value;
        mpool_free(cache_mp, delete_target);
    }

    assert(cache->list_size < cache->capacity);

    lirs_entry_t *new_entry = mpool_alloc(cache_mp);
    new_entry->key = key;
    new_entry->value = value;
    new_entry->frequency = 0;
    new_entry->resident = 1;
    INIT_LIST_HEAD(&new_entry->lru_list);
    INIT_LIST_HEAD(&new_entry->hir_list);
    /* Add the new block to the top of the LRU list */
    new_entry->isHIR =
        cache->capacity - cache->list_size <= cache->HIR_block_portion_limit;
    list_add(&new_entry->lru_list, cache->lists[0]);
    cache->lru_list_size++;
    if (new_entry->isHIR) {
        list_add(&new_entry->hir_list, cache->lists[1]);
    }
    cache->num_lir += !new_entry->isHIR;
    cache->list_size++;
    hlist_add_head(&new_entry->ht_list, &cache->map->ht_list_head[HASH(key)]);
    assert(cache->list_size <= cache->capacity);
    assert(cache->lru_list_size <= cache->capacity);
    assert(cache->num_lir + cache->HIR_block_portion_limit <= cache->capacity);

    return delete_value;
}

void cache_free(cache_t *cache, void (*callback)(void *))
{
    lirs_entry_t *entry, *safe;
#ifdef __HAVE_TYPEOF
    list_for_each_entry_safe (entry, safe, cache->lists[1], hir_list) {
        if (list_empty(&entry->lru_list))
            list_add(&entry->lru_list, cache->lists[0]);
    }

    list_for_each_entry_safe (entry, safe, cache->lists[0], lru_list)
        callback(entry->value);
#else
    list_for_each_entry_safe (entry, safe, cache->lists[1], hir_list,
                              lirs_entry_t)
        if (list_empty(&entry->lru_list))
            list_add(&entry->lru_list, cache->lists[0]);

    list_for_each_entry_safe (entry, safe, cache->lists[0], lru_list,
                              lirs_entry_t)
        callback(entry->value);
#endif

    mpool_destory(cache_mp);
    free(cache->map->ht_list_head);
    free(cache->map);
    free(cache);
}

#if RV32_HAS(JIT)
bool cache_hot(struct cache *cache, uint32_t key)
{
    if (!cache->capacity || hlist_empty(&cache->map->ht_list_head[HASH(key)]))
        return false;
    lirs_entry_t *entry = NULL;
#ifdef __HAVE_TYPEOF
    hlist_for_each_entry (entry, &cache->map->ht_list_head[HASH(key)], ht_list)
#else
    hlist_for_each_entry (entry, &cache->map->ht_list_head[HASH(key)], ht_list,
                          lirs_entry_t)
#endif
    {
        if (entry->key == key && entry->frequency >= THRESHOLD)
            return true;
    }
    return false;
}
#endif
