#ifndef COLLISION_DETECTION_COMMON_H
#define COLLISION_DETECTION_COMMON_H

#include <stdint.h>

// Workload modes.
#define KERNEL_MODE_AABB  0
#define KERNEL_MODE_TRACE 1

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

// Per-motion result produced by the trace-replay kernel.
typedef struct {
    uint32_t collision;
    uint32_t query_count;
    uint32_t key_checksum;
} trace_result_t;

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
} kernel_arg_t;

#endif
