#include <vortex2.h>
#include "common.h"
#include "trace_loader.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
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
const char* trace_file = nullptr;

uint32_t num_motions = 16;
uint32_t poses_per_motion = 8;
uint32_t num_obstacles = 8;

// COORD configuration.
bool coord_mode = false;
bool coord_all_benchmarks = false;
bool coord_benchmark_specified = false;
uint32_t coord_benchmark = 0;
uint32_t coord_slots = 6;
uint32_t coord_seed = 0;
float coord_threshold = 1.0f;
float coord_update_probability = 0.125f;

uint32_t parse_count(const char* value, const char* name) {
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);

    if (value[0] == '-' ||
        end == value ||
        *end != '\0' ||
        parsed == 0 ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Invalid value for " << name << ": "
                  << value << std::endl;
        std::exit(1);
    }

    return static_cast<uint32_t>(parsed);
}

float parse_float(const char* value, const char* name) {
    char* end = nullptr;
    float parsed = std::strtof(value, &end);

    if (end == value || *end != '\0' ||
        !std::isfinite(parsed)) {
        std::cerr << "Invalid value for " << name
                  << ": " << value << std::endl;
        std::exit(1);
    }

    return parsed;
}

void parse_args(int argc, char** argv) {
    int c;

    while ((c = getopt(argc, argv, "n:p:o:k:t:cab:w:s:u:r:h")) != -1) {
        switch (c) {
            case 'n':
                num_motions = parse_count(optarg, "-n");
                break;

            case 'p':
                poses_per_motion = parse_count(optarg, "-p");
                break;

            case 'o':
                num_obstacles = parse_count(optarg, "-o");
                break;

            case 'k':
                kernel_file = optarg;
                break;

            case 't':
                trace_file = optarg;
                break;

            case 'c':
                coord_mode = true;
                break;

            case 'a':
                coord_all_benchmarks = true;
                break;

            case 'b':
                coord_benchmark = parse_count(optarg, "-b") - 1;
                coord_benchmark_specified = true;
                break;

            case 'w':
                coord_slots = parse_count(optarg, "-w");
                break;

            case 's':
                coord_threshold = parse_float(optarg, "-s");
                break;

            case 'u':
                coord_update_probability = parse_float(optarg, "-u");
                break;

            case 'r':
                coord_seed = parse_count(optarg, "-r") - 1;
                break;

            case 'h':
                std::cout
                    << "Usage: [-k kernel] [-n motions] "
                    << "[-p poses_per_motion] [-o obstacles] "
                    << "[-t trace_file] [-c] [-a | -b benchmark] "
                    << "[-w slots] [-s threshold] "
                    << "[-u update_probability] [-r seed] [-h]"
                    << std::endl;
                std::exit(0);

            default:
                std::exit(1);
        }
    }

    if (trace_file == nullptr && poses_per_motion < 3) {
        std::cerr
            << "At least 3 poses per motion are required for "
            << "the intermediate/final collision test cases."
            << std::endl;
        std::exit(1);
    }

    if (coord_mode && trace_file == nullptr) {
        std::cerr << "COORD mode requires a trace file (-t)."
                  << std::endl;
        std::exit(1);
    }

    if (coord_all_benchmarks && !coord_mode) {
        std::cerr << "Option -a requires COORD mode (-c)."
                  << std::endl;
        std::exit(1);
    }

    if (coord_all_benchmarks && coord_benchmark_specified) {
        std::cerr << "Use either -a or -b, not both."
                  << std::endl;
        std::exit(1);
    }

    if (coord_mode &&
        (coord_threshold < 0.0f ||
         coord_update_probability < 0.0f ||
         coord_update_probability > 1.0f ||
         coord_slots == 0 ||
         coord_slots > COORD_MAX_SLOTS)) {
        std::cerr << "Invalid COORD configuration."
                  << std::endl;
        std::exit(1);
    }

    uint64_t total_boxes =
        static_cast<uint64_t>(num_motions) * poses_per_motion;

    if (trace_file == nullptr &&
        total_boxes > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Total number of robot boxes exceeds the "
                  << "32-bit kernel indexing limit."
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

uint32_t cpu_motion_collision_check(
    uint32_t motion_id,
    uint32_t poses_per_motion,
    const std::vector<aabb_t>& robot_boxes,
    const std::vector<aabb_t>& obstacles) {

    for (uint32_t pose_id = 0;
         pose_id < poses_per_motion;
         ++pose_id) {

        uint32_t box_idx =
            motion_id * poses_per_motion + pose_id;

        const aabb_t& robot = robot_boxes[box_idx];

        for (const auto& obstacle : obstacles) {
            if (aabb_intersects(robot, obstacle)) {
                return 1;
            }
        }
    }

    return 0;
}

aabb_t make_box(
    float min_x,
    float min_y,
    float min_z,
    float max_x,
    float max_y,
    float max_z) {

    return {
        min_x, min_y, min_z,
        max_x, max_y, max_z
    };
}

// A clear pose placed between obstacle 0 and obstacle 1.
aabb_t make_clear_pose() {
    return make_box(
        2.0f, -0.5f, -0.5f,
        3.0f,  0.5f,  0.5f
    );
}

// A colliding pose centered on the specified obstacle.
aabb_t make_colliding_pose(uint32_t obstacle_id) {
    float center_x = static_cast<float>(obstacle_id) * 10.0f;

    return make_box(
        center_x - 0.5f, -0.5f, -0.5f,
        center_x + 0.5f,  0.5f,  0.5f
    );
}

void generate_obstacles(std::vector<aabb_t>& obstacles) {
    for (uint32_t j = 0; j < num_obstacles; ++j) {
        float center_x = static_cast<float>(j) * 10.0f;

        obstacles[j] = make_box(
            center_x - 1.0f, -1.0f, -1.0f,
            center_x + 1.0f,  1.0f,  1.0f
        );
    }
}

void generate_motions(std::vector<aabb_t>& robot_boxes) {
    for (auto& box : robot_boxes) {
        box = make_clear_pose();
    }

    for (uint32_t motion_id = 0;
         motion_id < num_motions;
         ++motion_id) {

        uint32_t base = motion_id * poses_per_motion;

        if (motion_id == 0) {
            robot_boxes[base] = make_colliding_pose(0);
        } else if (motion_id == 1) {
            // All poses remain clear.
        } else if (motion_id == 2) {
            robot_boxes[base] = make_box(
                 1.0f, -0.5f, -0.5f,
                 2.0f,  0.5f,  0.5f
            );
        } else if (motion_id == 3) {
            robot_boxes[base] = make_box(
                 1.001f, -0.5f, -0.5f,
                 2.0f,    0.5f,  0.5f
            );
        } else if (motion_id == 4) {
            uint32_t collision_pose = poses_per_motion / 2;
            robot_boxes[base + collision_pose] =
                make_colliding_pose(0);
        } else if (motion_id == 5) {
            robot_boxes[base + poses_per_motion - 1] =
                make_colliding_pose(0);
        } else if (motion_id == 6) {
            uint32_t collision_pose = poses_per_motion - 2;
            robot_boxes[base + collision_pose] =
                make_colliding_pose(0);
        } else if (motion_id == 7) {
            // All poses remain clear.
        } else if (motion_id == 8) {
            robot_boxes[base] = make_box(
                -0.25f, -0.25f, -0.25f,
                 0.25f,  0.25f,  0.25f
            );
        } else if (motion_id == 9) {
            robot_boxes[base] = make_box(
                -2.0f, -2.0f, -2.0f,
                 2.0f,  2.0f,  2.0f
            );
        } else if (motion_id == 10) {
            robot_boxes[base] = make_box(
                -0.5f,  1.0f, -0.5f,
                 0.5f,  2.0f,  0.5f
            );
        } else if (motion_id == 11) {
            robot_boxes[base] = make_box(
                -0.5f,  1.001f, -0.5f,
                 0.5f,  2.0f,    0.5f
            );
        } else {
            if (motion_id % 2 == 0) {
                uint32_t collision_pose =
                    motion_id % poses_per_motion;

                uint32_t obstacle_id =
                    motion_id % num_obstacles;

                robot_boxes[base + collision_pose] =
                    make_colliding_pose(obstacle_id);
            }
        }
    }
}

uint32_t cpu_quantize_feature(float value) {
    for (uint32_t i = 0; i < 16; ++i) {
        float threshold = -1.0f + 0.125f * static_cast<float>(i);

        if (value <= threshold) {
            return i;
        }
    }

    return 16;
}

uint32_t cpu_pack_feature_key(
    uint32_t bx,
    uint32_t by,
    uint32_t bz) {

    return bx * 17 * 17 + by * 17 + bz;
}

trace_result_t cpu_trace_motion_check(
    uint32_t motion_id,
    const mpnet_trace_t& trace) {

    trace_result_t result{};

    uint64_t pose_begin = trace.motion_offsets[motion_id];
    uint64_t pose_end = trace.motion_offsets[motion_id + 1];

    for (uint64_t pose_id = pose_begin;
         pose_id < pose_end;
         ++pose_id) {

        uint64_t query_begin = trace.pose_offsets[pose_id];
        uint64_t query_end = trace.pose_offsets[pose_id + 1];

        for (uint64_t query_id = query_begin;
             query_id < query_end;
             ++query_id) {

            const trace_feature_t& feature =
                trace.features[query_id];

            uint32_t bx = cpu_quantize_feature(feature.x);
            uint32_t by = cpu_quantize_feature(feature.y);
            uint32_t bz = cpu_quantize_feature(feature.z);

            uint32_t key = cpu_pack_feature_key(bx, by, bz);

            result.key_checksum += key;
            ++result.query_count;

            if (trace.labels[query_id] == 0) {
                result.collision = 1;
            }
        }
    }

    return result;
}

} // namespace


void build_coord_ordered_queries(
    const mpnet_trace_t& trace,
    bool all_benchmarks,
    uint32_t benchmark_index,
    uint32_t motion_limit,
    uint64_t& first_motion,
    std::vector<uint64_t>& ordered_query_offsets,
    std::vector<uint64_t>& ordered_query_ids,
    std::vector<uint64_t>& benchmark_motion_offsets) {

    if (!all_benchmarks &&
        benchmark_index >= trace.benchmark_ids.size()) {
        std::cerr << "Invalid COORD benchmark index: "
                  << benchmark_index << std::endl;
        std::exit(1);
    }

    static const uint32_t pose_offsets_order[] = {
        0, 4, 2, 6, 1, 5, 3, 7
    };

    ordered_query_offsets.clear();
    ordered_query_ids.clear();
    benchmark_motion_offsets.clear();

    ordered_query_offsets.push_back(0);

    const uint32_t first_benchmark =
        all_benchmarks ? 0 : benchmark_index;

    const uint32_t end_benchmark =
        all_benchmarks
            ? static_cast<uint32_t>(trace.benchmark_ids.size())
            : benchmark_index + 1;

    uint64_t selected_total = 0;
    first_motion = 0;

    for (uint32_t b = first_benchmark;
         b < end_benchmark && selected_total < motion_limit;
         ++b) {

        const uint64_t benchmark_begin =
            trace.benchmark_offsets[b];

        const uint64_t benchmark_end =
            trace.benchmark_offsets[b + 1];

        const uint64_t available_motions =
            benchmark_end - benchmark_begin;

        const uint64_t remaining_limit =
            motion_limit - selected_total;

        const uint64_t selected_motions =
            available_motions < remaining_limit
                ? available_motions
                : remaining_limit;

        if (selected_motions == 0) {
            continue;
        }

        if (selected_total == 0) {
            first_motion = benchmark_begin;
        }

        // This is a position in the selected workload, not a trace motion ID.
        benchmark_motion_offsets.push_back(selected_total);

        for (uint64_t local_motion = 0;
             local_motion < selected_motions;
             ++local_motion) {

            const uint64_t motion_id =
                benchmark_begin + local_motion;

            const uint64_t first_pose =
                trace.motion_offsets[motion_id];

            const uint64_t last_pose =
                trace.motion_offsets[motion_id + 1];

            const uint64_t num_poses =
                last_pose - first_pose;

            if (num_poses == 0) {
                std::cerr << "Motion " << motion_id
                          << " contains no poses." << std::endl;
                std::exit(1);
            }

            // Process the final pose first.
            const uint64_t final_pose = last_pose - 1;

            for (uint64_t query_id = trace.pose_offsets[final_pose];
                 query_id < trace.pose_offsets[final_pose + 1];
                 ++query_id) {
                ordered_query_ids.push_back(query_id);
            }

            // Process the remaining poses in csp_rearrange order.
            for (uint32_t i = 0; i < 8; ++i) {
                for (uint64_t j = pose_offsets_order[i];
                     j < num_poses - 1;
                     j += 8) {

                    const uint64_t pose_id = first_pose + j;

                    for (uint64_t query_id = trace.pose_offsets[pose_id];
                         query_id < trace.pose_offsets[pose_id + 1];
                         ++query_id) {
                        ordered_query_ids.push_back(query_id);
                    }
                }
            }

            ordered_query_offsets.push_back(
                ordered_query_ids.size());
        }

        selected_total += selected_motions;
    }

    if (selected_total == 0) {
        std::cerr << "COORD workload contains no motions."
                  << std::endl;
        std::exit(1);
    }

    std::cout << "COORD ordered-query workload:" << std::endl;
    if (all_benchmarks) {
        std::cout << "  Benchmarks:      all" << std::endl;
    } else {
        std::cout << "  Benchmark index: " << benchmark_index << std::endl;
        std::cout << "  Benchmark ID:    "
                  << trace.benchmark_ids[benchmark_index] << std::endl;
    }
    std::cout << "  First motion:    " << first_motion << std::endl;
    std::cout << "  Motions:         " << selected_total << std::endl;
    std::cout << "  CHT resets:      "
              << benchmark_motion_offsets.size() << std::endl;
    std::cout << "  Ordered queries: "
              << ordered_query_ids.size() << std::endl;
}

int main(int argc, char** argv) {
    parse_args(argc, argv);

    const uint32_t requested_num_motions = num_motions;

    const bool trace_mode = (trace_file != nullptr);

    // ----- Optionally load the MPNet trace -----

    mpnet_trace_t trace;

    uint64_t coord_first_motion = 0;

    std::vector<uint64_t> h_ordered_query_offsets;
    std::vector<uint64_t> h_ordered_query_ids;
    std::vector<uint64_t> h_coord_benchmark_offsets;

    if (trace_mode) {
        std::string error;

        if (!load_mpnet_trace(trace_file, &trace, &error)) {
            std::cerr << "Trace loading failed: " << error
                      << std::endl;
            return 1;
        }

        if (trace.num_motions() == 0 ||
            trace.num_motions() >
                std::numeric_limits<uint32_t>::max()) {
            std::cerr << "Invalid or unsupported trace motion count."
                      << std::endl;
            return 1;
        }

        num_motions = static_cast<uint32_t>(trace.num_motions());

        if (coord_mode) {
            build_coord_ordered_queries(
                trace,
                coord_all_benchmarks,
                coord_benchmark,
                requested_num_motions,
                coord_first_motion,
                h_ordered_query_offsets,
                h_ordered_query_ids,
                h_coord_benchmark_offsets);

            num_motions =
                static_cast<uint32_t>(h_ordered_query_offsets.size() - 1);
        }

        uint64_t collision_count = 0;
        for (uint8_t label : trace.labels) {
            collision_count += (label == 0);
        }

        std::cout << "MPNet trace loaded successfully:" << std::endl;
        std::cout << "  File:       " << trace_file << std::endl;
        std::cout << "  Version:    " << trace.version << std::endl;
        std::cout << "  Benchmarks: " << trace.benchmark_ids.size() << std::endl;
        std::cout << "  Motions:    " << trace.num_motions() << std::endl;
        std::cout << "  Poses:      " << trace.num_poses() << std::endl;
        std::cout << "  Queries:    " << trace.num_queries() << std::endl;
        std::cout << "  Collisions: " << collision_count << std::endl;
        std::cout << "  Free:       "
                  << trace.num_queries() - collision_count << std::endl;
    }

    // ----- Prepare host-side workload data -----

    const uint64_t num_robot_boxes =
        trace_mode
            ? 0
            : static_cast<uint64_t>(num_motions) * poses_per_motion;

    const uint64_t robot_buf_size =
        num_robot_boxes * sizeof(aabb_t);

    const uint64_t obstacle_buf_size =
        static_cast<uint64_t>(num_obstacles) * sizeof(aabb_t);

    const uint64_t results_buf_size =
        static_cast<uint64_t>(num_motions) *
        (coord_mode
            ? sizeof(coord_result_t)
            : (trace_mode ? sizeof(trace_result_t) : sizeof(uint32_t)));

    std::vector<aabb_t> h_robots;
    std::vector<aabb_t> h_obstacles;
    std::vector<uint32_t> h_results;
    std::vector<trace_result_t> h_trace_results;
    std::vector<coord_result_t> h_coord_results;

    if (coord_mode) {
        h_coord_results.resize(num_motions);
    } else if (trace_mode) {
        h_trace_results.resize(num_motions);

        std::cout << "Trace replay workload:"
                  << " motions=" << num_motions
                  << " poses=" << trace.num_poses()
                  << " queries=" << trace.num_queries()
                  << std::endl;
    } else {
        h_robots.resize(static_cast<size_t>(num_robot_boxes));
        h_obstacles.resize(num_obstacles);
        h_results.resize(num_motions, 0);

        generate_obstacles(h_obstacles);
        generate_motions(h_robots);

        std::cout << "collision_detection vortex2:"
                  << " motions=" << num_motions
                  << " poses_per_motion=" << poses_per_motion
                  << " obstacles=" << num_obstacles
                  << " total_robot_boxes=" << num_robot_boxes
                  << std::endl;
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

    vx_buffer_h motion_offsets_buf = nullptr;
    vx_buffer_h pose_offsets_buf = nullptr;
    vx_buffer_h features_buf = nullptr;
    vx_buffer_h labels_buf = nullptr;

    vx_buffer_h ordered_query_offsets_buf = nullptr;
    vx_buffer_h ordered_query_ids_buf = nullptr;
    vx_buffer_h coord_benchmark_offsets_buf = nullptr;
    vx_buffer_h coord_cht_buf = nullptr;

    if (trace_mode) {
        CHECK(vx_buffer_create(
            dev,
            trace.motion_offsets.size() * sizeof(uint64_t),
            VX_MEM_READ,
            &motion_offsets_buf));

        CHECK(vx_buffer_create(
            dev,
            trace.pose_offsets.size() * sizeof(uint64_t),
            VX_MEM_READ,
            &pose_offsets_buf));

        CHECK(vx_buffer_create(
            dev,
            trace.features.size() * sizeof(trace_feature_t),
            VX_MEM_READ,
            &features_buf));

        CHECK(vx_buffer_create(
            dev,
            trace.labels.size() * sizeof(uint8_t),
            VX_MEM_READ,
            &labels_buf));

        if (coord_mode) {
            CHECK(vx_buffer_create(
                dev,
                h_ordered_query_offsets.size() * sizeof(uint64_t),
                VX_MEM_READ,
                &ordered_query_offsets_buf));

            CHECK(vx_buffer_create(
                dev,
                h_ordered_query_ids.size() * sizeof(uint64_t),
                VX_MEM_READ,
                &ordered_query_ids_buf));

            CHECK(vx_buffer_create(
                dev,
                h_coord_benchmark_offsets.size() * sizeof(uint64_t),
                VX_MEM_READ,
                &coord_benchmark_offsets_buf));

            CHECK(vx_buffer_create(
                dev,
                COORD_CHT_ENTRIES * sizeof(coord_cht_entry_t),
                VX_MEM_READ | VX_MEM_WRITE,
                &coord_cht_buf));
        }

        CHECK(vx_buffer_create(
            dev,
            results_buf_size,
            VX_MEM_WRITE,
            &results_buf));
    } else {
        CHECK(vx_buffer_create(
            dev, robot_buf_size, VX_MEM_READ, &robots_buf));

        CHECK(vx_buffer_create(
            dev, obstacle_buf_size, VX_MEM_READ, &obstacles_buf));

        CHECK(vx_buffer_create(
            dev, results_buf_size, VX_MEM_WRITE, &results_buf));
    }

    // ----- Load kernel -----

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;

    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    // ----- Set kernel arguments -----

    kernel_arg_t kernel_arg{};

    kernel_arg.mode =
        coord_mode
            ? KERNEL_MODE_COORD
            : (trace_mode ? KERNEL_MODE_TRACE : KERNEL_MODE_AABB);

    kernel_arg.num_motions = num_motions;
    kernel_arg.poses_per_motion = poses_per_motion;
    kernel_arg.num_obstacles = num_obstacles;

    if (trace_mode) {
        CHECK(vx_buffer_address(
            motion_offsets_buf, &kernel_arg.motion_offsets_addr));

        CHECK(vx_buffer_address(
            pose_offsets_buf, &kernel_arg.pose_offsets_addr));

        CHECK(vx_buffer_address(
            features_buf, &kernel_arg.features_addr));

        CHECK(vx_buffer_address(
            labels_buf, &kernel_arg.labels_addr));

        if (coord_mode) {
            CHECK(vx_buffer_address(
                ordered_query_offsets_buf,
                &kernel_arg.ordered_query_offsets_addr));

            CHECK(vx_buffer_address(
                ordered_query_ids_buf,
                &kernel_arg.ordered_query_ids_addr));

            CHECK(vx_buffer_address(
                coord_benchmark_offsets_buf,
                &kernel_arg.benchmark_offsets_addr));

            CHECK(vx_buffer_address(
                coord_cht_buf,
                &kernel_arg.coord_cht_addr));

            CHECK(vx_buffer_address(
                results_buf,
                &kernel_arg.coord_results_addr));

            kernel_arg.coord_slots = coord_slots;
            kernel_arg.coord_seed = coord_seed;
            kernel_arg.coord_threshold = coord_threshold;
            kernel_arg.coord_update_probability =
                coord_update_probability;
            kernel_arg.coord_num_motions = num_motions;
            kernel_arg.coord_num_benchmarks =
                static_cast<uint32_t>(h_coord_benchmark_offsets.size());

            kernel_arg.coord_update_threshold =
                static_cast<uint64_t>(
                    static_cast<double>(coord_update_probability) *
                    9007199254740992.0);
        } else {
            CHECK(vx_buffer_address(
                results_buf, &kernel_arg.trace_results_addr));
        }
    } else {
        CHECK(vx_buffer_address(
            robots_buf, &kernel_arg.robot_boxes_addr));

        CHECK(vx_buffer_address(
            obstacles_buf, &kernel_arg.obstacle_boxes_addr));

        CHECK(vx_buffer_address(
            results_buf, &kernel_arg.results_addr));
    }

    // ----- Upload inputs -----

    if (trace_mode) {
        CHECK(vx_enqueue_write(
            q, motion_offsets_buf, 0,
            trace.motion_offsets.data(),
            trace.motion_offsets.size() * sizeof(uint64_t),
            0, nullptr, nullptr));

        CHECK(vx_enqueue_write(
            q, pose_offsets_buf, 0,
            trace.pose_offsets.data(),
            trace.pose_offsets.size() * sizeof(uint64_t),
            0, nullptr, nullptr));

        CHECK(vx_enqueue_write(
            q, features_buf, 0,
            trace.features.data(),
            trace.features.size() * sizeof(trace_feature_t),
            0, nullptr, nullptr));

        CHECK(vx_enqueue_write(
            q, labels_buf, 0,
            trace.labels.data(),
            trace.labels.size() * sizeof(uint8_t),
            0, nullptr, nullptr));

        if (coord_mode) {
            CHECK(vx_enqueue_write(
                q, ordered_query_offsets_buf, 0,
                h_ordered_query_offsets.data(),
                h_ordered_query_offsets.size() * sizeof(uint64_t),
                0, nullptr, nullptr));

            CHECK(vx_enqueue_write(
                q, ordered_query_ids_buf, 0,
                h_ordered_query_ids.data(),
                h_ordered_query_ids.size() * sizeof(uint64_t),
                0, nullptr, nullptr));

            CHECK(vx_enqueue_write(
                q, coord_benchmark_offsets_buf, 0,
                h_coord_benchmark_offsets.data(),
                h_coord_benchmark_offsets.size() * sizeof(uint64_t),
                0, nullptr, nullptr));
        }
    } else {
        CHECK(vx_enqueue_write(
            q, robots_buf, 0,
            h_robots.data(), robot_buf_size,
            0, nullptr, nullptr));

        CHECK(vx_enqueue_write(
            q, obstacles_buf, 0,
            h_obstacles.data(), obstacle_buf_size,
            0, nullptr, nullptr));
    }

    // ----- Launch kernel -----

    uint32_t grid[1], block[1];

    uint32_t launch_work_items = coord_mode ? 1 : num_motions;

    CHECK(vx_device_max_occupancy_grid(
        dev, 1, &launch_work_items, grid, block));

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

    CHECK(vx_enqueue_launch(
        q, &li, 0, nullptr, &launch_ev));

    // ----- Read back results -----

    void* h_result_data =
        coord_mode
            ? static_cast<void*>(h_coord_results.data())
            : (trace_mode
                ? static_cast<void*>(h_trace_results.data())
                : static_cast<void*>(h_results.data()));

    CHECK(vx_enqueue_read(
        q, h_result_data, results_buf,
        0, results_buf_size,
        1, &launch_ev, &read_ev));

    CHECK(vx_event_wait_value(
        read_ev, 1, VX_TIMEOUT_INFINITE));

    // ----- Verify GPU results against CPU reference -----

    int errors = 0;

    if (coord_mode) {
        uint64_t total_queries = 0;
        uint64_t total_cycles = 0;
        uint64_t total_collisions = 0;
        uint64_t total_attempts = 0;
        uint64_t total_enqueued = 0;
        double total_completed_predictions = 0.0;
        uint64_t total_weighted_work_units = 0;

        for (uint32_t motion_id = 0;
             motion_id < num_motions;
             ++motion_id) {

            const coord_result_t& actual =
                h_coord_results[motion_id];

            std::printf(
                "Motion %3u: cycles=%u collision=%u queries=%u "
                "attempts=%u enqueued=%u completed=%.6f weighted=%.6f\n",
                motion_id,
                actual.cycles,
                actual.collision,
                actual.num_queries,
                actual.classification_attempts,
                actual.queries_enqueued,
                actual.completed_predictions,
                static_cast<double>(actual.weighted_prediction_work_units) / 400.0);

            total_queries += actual.num_queries;
            total_cycles += actual.cycles;
            total_collisions += actual.collision;
            total_attempts += actual.classification_attempts;
            total_enqueued += actual.queries_enqueued;
            total_completed_predictions +=
                actual.completed_predictions;
            total_weighted_work_units +=
                actual.weighted_prediction_work_units;
        }

        std::cout << "COORD GPU results (CPU comparison pending):"
                  << " motions=" << num_motions
                  << " queries=" << total_queries
                  << " collisions=" << total_collisions
                  << " avg_cycles="
                  << static_cast<double>(total_cycles) / num_motions
                  << " avg_completed="
                  << total_completed_predictions / num_motions
                  << " avg_weighted_work="
                  << static_cast<double>(total_weighted_work_units) /
                         (400.0 * num_motions)
                  << " avg_attempts="
                  << static_cast<double>(total_attempts) / num_motions
                  << " avg_enqueued="
                  << static_cast<double>(total_enqueued) / num_motions
                  << std::endl;

    } else if (trace_mode) {
        uint64_t total_queries = 0;
        uint64_t total_collisions = 0;

        for (uint32_t motion_id = 0;
             motion_id < num_motions;
             ++motion_id) {

            trace_result_t expected =
                cpu_trace_motion_check(motion_id, trace);

            const trace_result_t& actual =
                h_trace_results[motion_id];

            std::printf(
                "Motion %5u: collision=%u/%u "
                "queries=%u/%u checksum=%u/%u\n",
                motion_id,
                expected.collision, actual.collision,
                expected.query_count, actual.query_count,
                expected.key_checksum, actual.key_checksum);

            if (actual.collision != expected.collision ||
                actual.query_count != expected.query_count ||
                actual.key_checksum != expected.key_checksum) {
                ++errors;
            }

            total_queries += actual.query_count;
            total_collisions += actual.collision;
        }

        std::cout << "Trace verification summary:"
                  << " motions=" << num_motions
                  << " queries_processed=" << total_queries
                  << " motions_with_collision=" << total_collisions
                  << std::endl;

        if (!coord_mode && total_queries != trace.num_queries()) {
            std::cerr << "Total processed query count does not match "
                      << "the trace query count." << std::endl;
            ++errors;
        }
    } else {
        for (uint32_t motion_id = 0;
             motion_id < num_motions;
             ++motion_id) {

            uint32_t expected =
                cpu_motion_collision_check(
                    motion_id,
                    poses_per_motion,
                    h_robots,
                    h_obstacles);

            std::printf(
                "Motion %3u: expected=%u actual=%u\n",
                motion_id, expected, h_results[motion_id]);

            if (h_results[motion_id] != expected) {
                ++errors;
            }
        }
    }

    // ----- Cleanup -----

    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    vx_buffer_release(results_buf);

    if (trace_mode) {
        if (coord_mode) {
            vx_buffer_release(coord_cht_buf);
            vx_buffer_release(coord_benchmark_offsets_buf);
            vx_buffer_release(ordered_query_ids_buf);
            vx_buffer_release(ordered_query_offsets_buf);
        }

        vx_buffer_release(labels_buf);
        vx_buffer_release(features_buf);
        vx_buffer_release(pose_offsets_buf);
        vx_buffer_release(motion_offsets_buf);
    } else {
        vx_buffer_release(obstacles_buf);
        vx_buffer_release(robots_buf);
    }

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

    if (coord_mode) {
        std::cout
            << "COORD execution completed; CPU comparison is pending."
            << std::endl;
    } else {
        std::cout << "PASSED!" << std::endl;
    }

    return 0;
}
