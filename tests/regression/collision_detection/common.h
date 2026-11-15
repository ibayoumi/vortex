#ifndef COLLISION_DETECTION_COMMON_H
#define COLLISION_DETECTION_COMMON_H

#include <stdint.h>

// Workload modes.
#define KERNEL_MODE_AABB     0
#define KERNEL_MODE_TRACE    1
#define KERNEL_MODE_COORD    2
#define KERNEL_MODE_GEOMETRY 3

// Jaco geometry workload constants.
#define GEOMETRY_NUM_LINKS 7

// COORD scheduler constants.
#define COORD_CHT_ENTRIES 4913
#define COORD_QNONCOLL_CAPACITY 56
#define COORD_QCOLL_CAPACITY 8
#define COORD_MAX_SLOTS 64
#define COORD_PREDICTION_LATENCY 40

// Axis-aligned bounding box.
typedef struct {
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
} aabb_t;

// One three-dimensional feature vector from the MPNet trace.
typedef struct {
    float x;
    float y;
    float z;
} trace_feature_t;

// Oriented bounding box.
// Rotation is a row-major 3x3 matrix whose columns are the local axes.
typedef struct {
    float center[3];
    float half_extents[3];
    float rotation[9];
} obb_t;

// Per-motion result from the geometry workload.
typedef struct {
    uint32_t collision;
    uint32_t link_collision_mask;
} geometry_result_t;

// Per-motion result produced by the trace-replay kernel.
typedef struct {
    uint32_t collision;
    uint32_t query_count;
    uint32_t key_checksum;
} trace_result_t;

// Per-motion COORD scheduler result.
typedef struct {
    uint32_t cycles;
    uint32_t collision;
    uint32_t num_queries;
    uint32_t classification_attempts;
    uint32_t queries_enqueued;
    float completed_predictions;
    uint32_t weighted_prediction_work_units; // Units of 1/400.
} coord_result_t;

// CHT entry: collision count and free count.
typedef struct {
    uint32_t collision_count;
    uint32_t free_count;
} coord_cht_entry_t;

// Arguments passed from the host to the GPU kernel.
typedef struct {
    // Workload configuration.
    uint32_t mode;
    uint32_t num_motions;
    uint32_t poses_per_motion;
    uint32_t num_obstacles;

    // Existing AABB workload buffers.
    uint64_t robot_boxes_addr;
    uint64_t obstacle_boxes_addr;
    uint64_t results_addr;

    // MPNet trace workload buffers.
    uint64_t motion_offsets_addr;
    uint64_t pose_offsets_addr;
    uint64_t features_addr;
    uint64_t labels_addr;
    uint64_t trace_results_addr;

    // COORD scheduler configuration.
    uint32_t coord_slots;
    uint32_t coord_seed;
    float coord_threshold;
    float coord_update_probability;
    uint32_t coord_num_motions;
    uint32_t coord_num_benchmarks;
    uint64_t coord_update_threshold;

    // COORD scheduler buffers.
    uint64_t benchmark_offsets_addr;
    uint64_t ordered_query_offsets_addr;
    uint64_t ordered_query_ids_addr;
    uint64_t coord_cht_addr;
    uint64_t coord_results_addr;

    // Jaco geometry workload buffers.
    uint64_t obbs_addr;
    uint64_t geometry_obstacles_addr;
    uint64_t geometry_results_addr;
} kernel_arg_t;

#endif
