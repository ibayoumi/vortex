#include <vx_spawn2.h>
#include "common.h"

// -----------------------------------------------------------------------------
// Feature quantization and CHT key generation
// -----------------------------------------------------------------------------

static inline uint32_t quantize_feature(float value) {
    for (uint32_t i = 0; i < 16; ++i) {
        float threshold = -1.0f + 0.125f * (float)i;

        if (value <= threshold) {
            return i;
        }
    }

    return 16;
}

static inline uint32_t pack_feature_key(
    uint32_t bx,
    uint32_t by,
    uint32_t bz) {

    return bx * 17 * 17 + by * 17 + bz;
}

static inline uint32_t get_feature_key(
    const trace_feature_t& feature) {

    return pack_feature_key(
        quantize_feature(feature.x),
        quantize_feature(feature.y),
        quantize_feature(feature.z));
}

// -----------------------------------------------------------------------------
// Python-compatible MT19937 random number generator
//
// Python random.Random(seed), for a nonnegative 32-bit integer seed, initializes
// MT19937 using init_by_array with the seed as a one-word key.
//
// random() uses the standard 53-bit construction:
//   a = genrand_uint32() >> 5
//   b = genrand_uint32() >> 6
//   (a * 2^26 + b) / 2^53
// -----------------------------------------------------------------------------

#define MT_N 624
#define MT_M 397

typedef struct {
    uint32_t state[MT_N];
    uint32_t index;
} coord_rng_t;

static inline void mt_seed_genrand(
    coord_rng_t* rng,
    uint32_t seed) {

    rng->state[0] = seed;

    for (uint32_t i = 1; i < MT_N; ++i) {
        uint32_t previous = rng->state[i - 1];

        rng->state[i] =
            1812433253U *
            (previous ^ (previous >> 30)) + i;
    }

    rng->index = MT_N;
}

static inline void mt_seed_by_array(
    coord_rng_t* rng,
    uint32_t seed) {

    // Python random.seed(int) uses init_by_array.
    mt_seed_genrand(rng, 19650218U);

    uint32_t i = 1;
    uint32_t j = 0;

    // A nonnegative uint32 seed is represented by one little-endian word.
    const uint32_t key_length = 1;

    for (uint32_t k = MT_N > key_length ? MT_N : key_length;
         k > 0;
         --k) {

        uint32_t previous = rng->state[i - 1];

        rng->state[i] =
            (rng->state[i] ^
             ((previous ^ (previous >> 30)) * 1664525U))
            + seed + j;

        ++i;
        ++j;

        if (i >= MT_N) {
            rng->state[0] = rng->state[MT_N - 1];
            i = 1;
        }

        if (j >= key_length) {
            j = 0;
        }
    }

    for (uint32_t k = MT_N - 1; k > 0; --k) {
        uint32_t previous = rng->state[i - 1];

        rng->state[i] =
            (rng->state[i] ^
             ((previous ^ (previous >> 30)) * 1566083941U))
            - i;

        ++i;

        if (i >= MT_N) {
            rng->state[0] = rng->state[MT_N - 1];
            i = 1;
        }
    }

    rng->state[0] = 0x80000000U;
    rng->index = MT_N;
}

static inline void mt_twist(coord_rng_t* rng) {
    const uint32_t upper_mask = 0x80000000U;
    const uint32_t lower_mask = 0x7fffffffU;
    const uint32_t matrix_a = 0x9908b0dfU;

    for (uint32_t i = 0; i < MT_N; ++i) {
        uint32_t y =
            (rng->state[i] & upper_mask) |
            (rng->state[(i + 1) % MT_N] & lower_mask);

        uint32_t next =
            rng->state[(i + MT_M) % MT_N] ^ (y >> 1);

        if (y & 1U) {
            next ^= matrix_a;
        }

        rng->state[i] = next;
    }

    rng->index = 0;
}

static inline uint32_t mt_next_uint32(coord_rng_t* rng) {
    if (rng->index >= MT_N) {
        mt_twist(rng);
    }

    uint32_t y = rng->state[rng->index++];

    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;

    return y;
}

static inline uint64_t mt_random_numerator(coord_rng_t* rng) {
    uint32_t a = mt_next_uint32(rng) >> 5;
    uint32_t b = mt_next_uint32(rng) >> 6;

    return ((uint64_t)a << 26) | (uint64_t)b;
}

// -----------------------------------------------------------------------------
// COORD scheduler data structures
// -----------------------------------------------------------------------------

typedef struct {
    uint32_t key;
    uint32_t label;
} coord_query_t;

typedef struct {
    uint32_t key;
    uint32_t label;
    uint32_t active;
    uint32_t ready_cycle;
} coord_slot_t;

typedef struct {
    coord_query_t entries[COORD_QNONCOLL_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
} coord_noncoll_queue_t;

typedef struct {
    coord_query_t entries[COORD_QCOLL_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
} coord_coll_queue_t;

static inline void noncoll_push(
    coord_noncoll_queue_t* queue,
    uint32_t key,
    uint32_t label) {

    queue->entries[queue->tail].key = key;
    queue->entries[queue->tail].label = label;

    queue->tail =
        (queue->tail + 1) % COORD_QNONCOLL_CAPACITY;

    ++queue->count;
}

static inline coord_query_t noncoll_pop(
    coord_noncoll_queue_t* queue) {

    coord_query_t result = queue->entries[queue->head];

    queue->head =
        (queue->head + 1) % COORD_QNONCOLL_CAPACITY;

    --queue->count;

    return result;
}

static inline void coll_push(
    coord_coll_queue_t* queue,
    uint32_t key,
    uint32_t label) {

    queue->entries[queue->tail].key = key;
    queue->entries[queue->tail].label = label;

    queue->tail =
        (queue->tail + 1) % COORD_QCOLL_CAPACITY;

    ++queue->count;
}

static inline coord_query_t coll_pop(
    coord_coll_queue_t* queue) {

    coord_query_t result = queue->entries[queue->head];

    queue->head =
        (queue->head + 1) % COORD_QCOLL_CAPACITY;

    --queue->count;

    return result;
}

// -----------------------------------------------------------------------------
// Simulate one motion. The CHT and RNG are shared across motions.
// -----------------------------------------------------------------------------

static inline coord_result_t coord_simulate_motion(
    uint64_t query_begin,
    uint64_t query_end,
    const uint64_t* ordered_query_ids,
    const trace_feature_t* features,
    const uint8_t* labels,
    coord_cht_entry_t* cht,
    coord_rng_t* rng,
    uint32_t num_slots,
    float threshold,
    uint64_t update_threshold) {

    coord_result_t result{};

    const uint32_t num_queries =
        (uint32_t)(query_end - query_begin);

    result.num_queries = num_queries;

    if (num_queries == 0) {
        return result;
    }

    coord_slot_t slots[COORD_MAX_SLOTS];

    for (uint32_t i = 0; i < num_slots; ++i) {
        slots[i].key = 0;
        slots[i].label = 0;
        slots[i].active = 0;
        slots[i].ready_cycle = 0;
    }

    coord_noncoll_queue_t qnoncoll{};
    coord_coll_queue_t qcoll{};

    uint32_t query_cursor = 0;
    uint32_t links_remaining = num_queries;

    uint32_t cycle = 0;
    uint32_t first_two_running = 0;
    uint32_t first_two_checked = 0;

    uint32_t coll_found = 0;
    uint32_t everything_free = 0;

    float completed_predictions = 0.0f;
    uint32_t weighted_prediction_work_units = 0;

    uint32_t classification_attempts = 0;
    uint32_t queries_enqueued = 0;

    while (!coll_found && !everything_free) {

        // ---------------------------------------------------------------------
        // First: complete predictions and dispatch queued predictions.
        // ---------------------------------------------------------------------

        for (uint32_t slot_id = 0;
             slot_id < num_slots;
             ++slot_id) {

            coord_slot_t* slot = &slots[slot_id];

            if (slot->active &&
                slot->ready_cycle <= cycle) {

                completed_predictions += 1.0f;

                if (slot->label == 0) {
                    weighted_prediction_work_units += 400;
                    coll_found = 1;
                } else {
                    weighted_prediction_work_units += 280;
                }

                // Create the CHT entry when the first prediction completes.
                coord_cht_entry_t* entry = &cht[slot->key];

                if (slot->label == 1) {
                    // Consume a random draw for every completed free query,
                    // even if the free counter has already reached 15.
                    uint64_t update_draw =
                        mt_random_numerator(rng);

                    if (update_draw <= update_threshold &&
                        entry->free_count < 15) {
                        ++entry->free_count;
                    }
                } else {
                    if (entry->collision_count < 15) {
                        ++entry->collision_count;
                    }
                }
            }

            if (slot->ready_cycle <= cycle) {

                if (qcoll.count > 0 &&
                    first_two_checked < cycle) {

                    ++first_two_running;

                    if (first_two_running == 1) {
                        first_two_checked =
                            cycle + COORD_PREDICTION_LATENCY;
                    }

                    coord_query_t query = coll_pop(&qcoll);

                    slot->key = query.key;
                    slot->label = query.label;
                    slot->active = 1;
                    slot->ready_cycle =
                        cycle + COORD_PREDICTION_LATENCY;

                } else if (
                    qnoncoll.count == COORD_QNONCOLL_CAPACITY ||
                    (
                        (links_remaining == 0 &&
                         qnoncoll.count > 0) &&
                        first_two_checked < cycle
                    )
                ) {

                    coord_query_t query = noncoll_pop(&qnoncoll);

                    slot->key = query.key;
                    slot->label = query.label;
                    slot->active = 1;
                    slot->ready_cycle =
                        cycle + COORD_PREDICTION_LATENCY;

                } else {
                    slot->key = 0;
                    slot->label = 0;
                    slot->active = 0;
                    slot->ready_cycle = 0;
                }
            }
        }

        // ---------------------------------------------------------------------
        // Second: classify at most one query per cycle.
        // ---------------------------------------------------------------------

        if (query_cursor < num_queries) {
            uint64_t original_query_id =
                ordered_query_ids[query_begin + query_cursor];

            const trace_feature_t feature =
                features[original_query_id];

            const uint32_t label =
                (uint32_t)labels[original_query_id];

            ++classification_attempts;

            const uint32_t key = get_feature_key(feature);

            const coord_cht_entry_t entry = cht[key];

            // A CHT entry exists if either counter is nonzero.
            // An entry with two zero counters is equivalent to a missing
            // entry for classification purposes.
            const uint32_t entry_exists =
                (entry.collision_count != 0 ||
                 entry.free_count != 0);

            if (entry_exists) {
                if ((float)entry.collision_count >
                    (float)entry.free_count * threshold) {

                    if (qcoll.count < COORD_QCOLL_CAPACITY) {
                        coll_push(&qcoll, key, label);
                        ++query_cursor;
                        ++queries_enqueued;
                    }

                } else {
                    if (qnoncoll.count <
                        COORD_QNONCOLL_CAPACITY) {
                        noncoll_push(&qnoncoll, key, label);
                        ++query_cursor;
                        ++queries_enqueued;
                    }
                }

            } else {
                if (qnoncoll.count <
                    COORD_QNONCOLL_CAPACITY) {
                    noncoll_push(&qnoncoll, key, label);
                    ++query_cursor;
                    ++queries_enqueued;
                }
            }
        }

        links_remaining = num_queries - query_cursor;

        // ---------------------------------------------------------------------
        // Third: determine whether all work has drained.
        // ---------------------------------------------------------------------

        if (links_remaining == 0) {
            everything_free = 1;

            for (uint32_t i = 0; i < num_slots; ++i) {
                if (slots[i].ready_cycle > cycle) {
                    everything_free = 0;
                }
            }

            if (qnoncoll.count > 0) {
                everything_free = 0;
            }

            if (qcoll.count > 0) {
                everything_free = 0;
            }
        }

        ++cycle;
    }

    // -------------------------------------------------------------------------
    // Partial credit for predictions still in flight when a collision ends
    // the motion.
    // -------------------------------------------------------------------------

    for (uint32_t i = 0; i < num_slots; ++i) {
        if (slots[i].ready_cycle > cycle) {
            float fraction =
                (float)(
                    COORD_PREDICTION_LATENCY -
                    slots[i].ready_cycle +
                    cycle
                ) / (float)COORD_PREDICTION_LATENCY;

            completed_predictions += fraction;

            // 0.7 * (partial_cycles / 40) = (7 * partial_cycles) / 400.
            const uint32_t partial_cycles =
                COORD_PREDICTION_LATENCY -
                (slots[i].ready_cycle - cycle);
            weighted_prediction_work_units += 7 * partial_cycles;
        }
    }

    result.cycles = cycle;
    result.collision = coll_found;
    result.classification_attempts = classification_attempts;
    result.queries_enqueued = queries_enqueued;
    result.completed_predictions = completed_predictions;
    result.weighted_prediction_work_units = weighted_prediction_work_units;

    return result;
}

// -----------------------------------------------------------------------------
// Main kernel
// -----------------------------------------------------------------------------

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t motion_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    // -------------------------------------------------------------------------
    // Mode 0: Original synthetic AABB collision workload.
    // -------------------------------------------------------------------------

    if (arg->mode == KERNEL_MODE_AABB) {
        if (motion_id >= arg->num_motions) {
            return;
        }

        auto robot_boxes =
            reinterpret_cast<const aabb_t*>(arg->robot_boxes_addr);

        auto obstacle_boxes =
            reinterpret_cast<const aabb_t*>(arg->obstacle_boxes_addr);

        auto results =
            reinterpret_cast<uint32_t*>(arg->results_addr);

        uint32_t collision = 0;

        for (uint32_t pose_id = 0;
             pose_id < arg->poses_per_motion;
             ++pose_id) {

            uint32_t box_idx =
                motion_id * arg->poses_per_motion + pose_id;

            const aabb_t robot = robot_boxes[box_idx];

            for (uint32_t j = 0; j < arg->num_obstacles; ++j) {
                const aabb_t obstacle = obstacle_boxes[j];

                bool overlap =
                    (robot.min_x <= obstacle.max_x) &&
                    (robot.max_x >= obstacle.min_x) &&
                    (robot.min_y <= obstacle.max_y) &&
                    (robot.max_y >= obstacle.min_y) &&
                    (robot.min_z <= obstacle.max_z) &&
                    (robot.max_z >= obstacle.min_z);

                if (overlap) {
                    collision = 1;
                    break;
                }
            }

            if (collision) {
                break;
            }
        }

        results[motion_id] = collision;
        return;
    }

    // -------------------------------------------------------------------------
    // Mode 1: MPNet trace replay and feature-key processing.
    // -------------------------------------------------------------------------

    if (arg->mode == KERNEL_MODE_TRACE) {
        if (motion_id >= arg->num_motions) {
            return;
        }

        auto motion_offsets =
            reinterpret_cast<const uint64_t*>(
                arg->motion_offsets_addr);

        auto pose_offsets =
            reinterpret_cast<const uint64_t*>(
                arg->pose_offsets_addr);

        auto features =
            reinterpret_cast<const trace_feature_t*>(
                arg->features_addr);

        auto labels =
            reinterpret_cast<const uint8_t*>(
                arg->labels_addr);

        auto results =
            reinterpret_cast<trace_result_t*>(
                arg->trace_results_addr);

        uint64_t pose_begin = motion_offsets[motion_id];
        uint64_t pose_end = motion_offsets[motion_id + 1];

        uint32_t collision = 0;
        uint32_t query_count = 0;
        uint32_t key_checksum = 0;

        for (uint64_t pose_id = pose_begin;
             pose_id < pose_end;
             ++pose_id) {

            uint64_t query_begin = pose_offsets[pose_id];
            uint64_t query_end = pose_offsets[pose_id + 1];

            for (uint64_t query_id = query_begin;
                 query_id < query_end;
                 ++query_id) {

                const trace_feature_t feature = features[query_id];

                uint32_t key = get_feature_key(feature);

                key_checksum += key;
                ++query_count;

                if (labels[query_id] == 0) {
                    collision = 1;
                }
            }
        }

        results[motion_id].collision = collision;
        results[motion_id].query_count = query_count;
        results[motion_id].key_checksum = key_checksum;

        return;
    }

    // -------------------------------------------------------------------------
    // Mode 2: COORD scheduler.
    //
    // Exactly one GPU thread processes all selected motions sequentially.
    // This preserves CHT updates and the CPU reference's scheduling order.
    // -------------------------------------------------------------------------

    if (arg->mode == KERNEL_MODE_COORD) {
        if (motion_id != 0) {
            return;
        }

        auto features =
            reinterpret_cast<const trace_feature_t*>(
                arg->features_addr);

        auto labels =
            reinterpret_cast<const uint8_t*>(
                arg->labels_addr);

        auto ordered_query_offsets =
            reinterpret_cast<const uint64_t*>(
                arg->ordered_query_offsets_addr);

        auto ordered_query_ids =
            reinterpret_cast<const uint64_t*>(
                arg->ordered_query_ids_addr);

        auto cht =
            reinterpret_cast<coord_cht_entry_t*>(
                arg->coord_cht_addr);

        auto results =
            reinterpret_cast<coord_result_t*>(
                arg->coord_results_addr);

        // Reset the CHT once at the start of this benchmark.
        for (uint32_t i = 0; i < COORD_CHT_ENTRIES; ++i) {
            cht[i].collision_count = 0;
            cht[i].free_count = 0;
        }

        // Initialize one Python-compatible random stream for the benchmark.
        coord_rng_t rng;
        mt_seed_by_array(&rng, arg->coord_seed);

        for (uint32_t current_motion = 0;
             current_motion < arg->coord_num_motions;
             ++current_motion) {

            const uint64_t query_begin =
                ordered_query_offsets[current_motion];

            const uint64_t query_end =
                ordered_query_offsets[current_motion + 1];

            results[current_motion] =
                coord_simulate_motion(
                    query_begin,
                    query_end,
                    ordered_query_ids,
                    features,
                    labels,
                    cht,
                    &rng,
                    arg->coord_slots,
                    arg->coord_threshold,
                    arg->coord_update_threshold);
        }
    }
}
