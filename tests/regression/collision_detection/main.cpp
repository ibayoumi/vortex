#include <vortex2.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {

const char* kernel_file = "kernel.vxbin";
uint32_t num_queries = 16;
uint32_t num_obstacles = 8;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:o:k:h")) != -1) {
        switch (c) {
            case 'n': num_queries = std::atoi(optarg); break;
            case 'o': num_obstacles = std::atoi(optarg); break;
            case 'k': kernel_file = optarg; break;
            case 'h':
                std::cout << "Usage: [-k kernel] [-n queries] "
                             "[-o obstacles] [-h]" << std::endl;
                std::exit(0);
            default:
                std::cout << "Usage: [-k kernel] [-n queries] "
                             "[-o obstacles] [-h]" << std::endl;
                std::exit(1);
        }
    }

    if (num_queries == 0 || num_obstacles == 0) {
        std::cerr << "Number of queries and obstacles must be positive."
                  << std::endl;
        std::exit(1);
    }
}

bool aabb_intersects(const aabb_t& a, const aabb_t& b) {
    return
        (a.min_x <= b.max_x) &&
        (a.max_x >= b.min_x) &&
        (a.min_y <= b.max_y) &&
        (a.max_y >= b.min_y) &&
        (a.min_z <= b.max_z) &&
        (a.max_z >= b.min_z);
}

uint32_t cpu_collision_check(
    const aabb_t& robot,
    const std::vector<aabb_t>& obstacles) {

    for (const auto& obstacle : obstacles) {
        if (aabb_intersects(robot, obstacle)) {
            return 1;
        }
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);

    const uint64_t robot_buf_size =
        static_cast<uint64_t>(num_queries) * sizeof(aabb_t);
    const uint64_t obstacle_buf_size =
        static_cast<uint64_t>(num_obstacles) * sizeof(aabb_t);
    const uint64_t results_buf_size =
        static_cast<uint64_t>(num_queries) * sizeof(uint32_t);

    std::cout << "collision_detection vortex2:"
              << " queries=" << num_queries
              << " obstacles=" << num_obstacles
              << std::endl;

    // ----- Generate deterministic input AABBs -----

    std::vector<aabb_t> h_robots(num_queries);
    std::vector<aabb_t> h_obstacles(num_obstacles);
    std::vector<uint32_t> h_results(num_queries, 0);

    // Obstacles are separated along the x-axis.
    for (uint32_t j = 0; j < num_obstacles; ++j) {
        float center_x = static_cast<float>(j) * 10.0f;

        h_obstacles[j] = {
            center_x - 1.0f, -1.0f, -1.0f,
            center_x + 1.0f,  1.0f,  1.0f
        };
    }

    // ----- Generate robot boxes with geometric edge cases -----

    for (uint32_t i = 0; i < num_queries; ++i) {
        if (i == 0) {
            // Partial overlap with obstacle 0.
            h_robots[i] = {
                -0.5f, -0.5f, -0.5f,
                 0.5f,  0.5f,  0.5f
            };
        } else if (i == 1) {
            // Clearly separated from every obstacle.
            h_robots[i] = {
                 2.0f, -0.5f, -0.5f,
                 3.0f,  0.5f,  0.5f
            };
        } else if (i == 2) {
            // Touching obstacle 0 at its maximum X face.
            h_robots[i] = {
                 1.0f, -0.5f, -0.5f,
                 2.0f,  0.5f,  0.5f
            };
        } else if (i == 3) {
            // Small gap beyond obstacle 0's maximum X face.
            h_robots[i] = {
                 1.001f, -0.5f, -0.5f,
                 2.0f,    0.5f,  0.5f
            };
        } else if (i == 4) {
            // Robot box fully contained inside obstacle 0.
            h_robots[i] = {
                -0.25f, -0.25f, -0.25f,
                 0.25f,  0.25f,  0.25f
            };
        } else if (i == 5) {
            // Robot box fully contains obstacle 0.
            h_robots[i] = {
                -2.0f, -2.0f, -2.0f,
                 2.0f,  2.0f,  2.0f
            };
        } else if (i == 6) {
            // Touching obstacle 0 at its maximum Y face.
            h_robots[i] = {
                -0.5f,  1.0f, -0.5f,
                 0.5f,  2.0f,  0.5f
            };
        } else if (i == 7) {
            // Small gap beyond obstacle 0's maximum Y face.
            h_robots[i] = {
                -0.5f,  1.001f, -0.5f,
                 0.5f,  2.0f,    0.5f
            };
        } else {
            // Retain the alternating overlap/separation pattern.
            float center_x =
                static_cast<float>((i / 2) % num_obstacles) * 10.0f;

            if (i % 2 != 0) {
                center_x += 5.0f;
            }

            h_robots[i] = {
                center_x - 0.5f, -0.5f, -0.5f,
                center_x + 0.5f,  0.5f,  0.5f
            };
        }
    }

    // ----- Open device and create queue -----

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = {
        sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0
    };

    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    // ----- Allocate device buffers -----

    vx_buffer_h robots_buf = nullptr;
    vx_buffer_h obstacles_buf = nullptr;
    vx_buffer_h results_buf = nullptr;

    CHECK(vx_buffer_create(dev, robot_buf_size,
                           VX_MEM_READ, &robots_buf));
    CHECK(vx_buffer_create(dev, obstacle_buf_size,
                           VX_MEM_READ, &obstacles_buf));
    CHECK(vx_buffer_create(dev, results_buf_size,
                           VX_MEM_WRITE, &results_buf));

    // ----- Load kernel -----

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;

    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    // ----- Set kernel arguments -----

    kernel_arg_t kernel_arg{};
    kernel_arg.num_queries = num_queries;
    kernel_arg.num_obstacles = num_obstacles;

    CHECK(vx_buffer_address(robots_buf,
                            &kernel_arg.robot_boxes_addr));
    CHECK(vx_buffer_address(obstacles_buf,
                            &kernel_arg.obstacle_boxes_addr));
    CHECK(vx_buffer_address(results_buf,
                            &kernel_arg.results_addr));

    // ----- Upload inputs and launch kernel -----

    CHECK(vx_enqueue_write(q, robots_buf, 0,
                           h_robots.data(), robot_buf_size,
                           0, nullptr, nullptr));

    CHECK(vx_enqueue_write(q, obstacles_buf, 0,
                           h_obstacles.data(), obstacle_buf_size,
                           0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(
        dev, 1, &num_queries, grid, block));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel = kern;
    li.args_host = &kernel_arg;
    li.args_size = sizeof(kernel_arg);
    li.ndim = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0] = block[0];

    vx_event_h launch_ev = nullptr;
    vx_event_h read_ev = nullptr;

    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));

    // Readback depends on completion of the kernel launch.
    CHECK(vx_enqueue_read(q, h_results.data(), results_buf,
                          0, results_buf_size,
                          1, &launch_ev, &read_ev));

    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    // ----- Verify GPU results against CPU reference -----

    int errors = 0;

    for (uint32_t i = 0; i < num_queries; ++i) {
        uint32_t expected =
            cpu_collision_check(h_robots[i], h_obstacles);

        // Print every query's expected and actual collision flags.
        std::printf(
            "Query %2u: expected=%u actual=%u\n",
            i, expected, h_results[i]);

        if (h_results[i] != expected) {
            ++errors;
        }
    }

    // ----- Cleanup -----

    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    vx_buffer_release(results_buf);
    vx_buffer_release(obstacles_buf);
    vx_buffer_release(robots_buf);

    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);

    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!"
                  << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
