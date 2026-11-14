#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t motion_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    // The launch grid may contain more threads than motions.
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

    // Each motion consists of a sequence of discrete poses.
    for (uint32_t pose_id = 0;
         pose_id < arg->poses_per_motion;
         ++pose_id) {

        uint32_t box_idx =
            motion_id * arg->poses_per_motion + pose_id;

        const aabb_t robot = robot_boxes[box_idx];

        // Check the current pose against every obstacle.
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

        // A single colliding pose makes the entire motion collide.
        if (collision) {
            break;
        }
    }

    // One result per motion.
    results[motion_id] = collision;
}
