#include <vx_spawn2.h>
#include "common.h"

// Return the original COORD feature-bin index.
//
// This matches:
//   bins = [-1.0, -0.875, ..., 0.875]
//   np.digitize(value, bins, right=True)
//
// The result is in [0, 16].
static inline uint32_t quantize_feature(float value) {
    for (uint32_t i = 0; i < 16; ++i) {
        float threshold = -1.0f + 0.125f * (float)i;

        if (value <= threshold) {
            return i;
        }
    }

    return 16;
}

// Pack three base-17 bin indices into one unique key.
static inline uint32_t pack_feature_key(
    uint32_t bx,
    uint32_t by,
    uint32_t bz) {

    return bx * 17 * 17 + by * 17 + bz;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t motion_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (motion_id >= arg->num_motions) {
        return;
    }

    // ------------------------------------------------------------
    // Mode 0: Original synthetic AABB collision workload.
    // ------------------------------------------------------------
    if (arg->mode == KERNEL_MODE_AABB) {
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

    // ------------------------------------------------------------
    // Mode 1: MPNet trace replay and feature-key processing.
    // ------------------------------------------------------------
    if (arg->mode == KERNEL_MODE_TRACE) {
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

        // Find the pose range belonging to this motion.
        uint64_t pose_begin = motion_offsets[motion_id];
        uint64_t pose_end   = motion_offsets[motion_id + 1];

        uint32_t collision = 0;
        uint32_t query_count = 0;
        uint32_t key_checksum = 0;

        for (uint64_t pose_id = pose_begin;
             pose_id < pose_end;
             ++pose_id) {

            // Find the query range belonging to this pose.
            uint64_t query_begin = pose_offsets[pose_id];
            uint64_t query_end   = pose_offsets[pose_id + 1];

            for (uint64_t query_id = query_begin;
                 query_id < query_end;
                 ++query_id) {

                const trace_feature_t feature = features[query_id];

                uint32_t bx = quantize_feature(feature.x);
                uint32_t by = quantize_feature(feature.y);
                uint32_t bz = quantize_feature(feature.z);

                uint32_t key = pack_feature_key(bx, by, bz);

                // Accumulate a simple checksum for host-side validation.
                // Unsigned overflow is intentional (modulo 2^32).
                key_checksum += key;
                ++query_count;

                // Label 0 means collision; label 1 means collision-free.
                if (labels[query_id] == 0) {
                    collision = 1;
                }
            }
        }

        results[motion_id].collision = collision;
        results[motion_id].query_count = query_count;
        results[motion_id].key_checksum = key_checksum;
    }
}
