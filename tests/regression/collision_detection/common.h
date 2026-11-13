#ifndef COLLISION_DETECTION_COMMON_H
#define COLLISION_DETECTION_COMMON_H

#include <stdint.h>

// Axis-aligned bounding box.
typedef struct {
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
} aabb_t;

// Arguments passed from the host to the GPU kernel.
typedef struct {
    uint32_t num_queries;
    uint32_t num_obstacles;
    uint64_t robot_boxes_addr;
    uint64_t obstacle_boxes_addr;
    uint64_t results_addr;
} kernel_arg_t;

#endif
