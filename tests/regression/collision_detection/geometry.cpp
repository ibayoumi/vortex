#include <vortex2.h>
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
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

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-6f;

struct mat4_t {
    float m[4][4];
};

mat4_t identity_matrix() {
    mat4_t result{};

    for (uint32_t i = 0; i < 4; ++i) {
        result.m[i][i] = 1.0f;
    }

    return result;
}

mat4_t multiply(const mat4_t& a, const mat4_t& b) {
    mat4_t result{};

    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t k = 0; k < 4; ++k) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }

    return result;
}

// Standard DH transform, matching the Jaco model used by
// trace_generation/pred_trace_generation.py.
mat4_t give_dh(
    float d,
    float r,
    float theta,
    float alpha) {

    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const float ca = std::cos(alpha);
    const float sa = std::sin(alpha);

    mat4_t result = identity_matrix();

    result.m[0][0] = ct;
    result.m[0][1] = -ca * st;
    result.m[0][2] = sa * st;
    result.m[0][3] = r * ct;

    result.m[1][0] = st;
    result.m[1][1] = ca * ct;
    result.m[1][2] = -sa * ct;
    result.m[1][3] = r * st;

    result.m[2][0] = 0.0f;
    result.m[2][1] = sa;
    result.m[2][2] = ca;
    result.m[2][3] = d;

    return result;
}

void set_obb(
    obb_t& obb,
    const mat4_t& transform,
    const float dimensions[3]) {

    for (uint32_t i = 0; i < 3; ++i) {
        obb.center[i] = transform.m[i][3];
        obb.half_extents[i] = dimensions[i] * 0.5f;

        for (uint32_t j = 0; j < 3; ++j) {
            // Row-major matrix. Its columns are the OBB's local axes.
            obb.rotation[i * 3 + j] = transform.m[i][j];
        }
    }
}

void generate_jaco_obbs(
    const float q[7],
    obb_t* obbs) {

    // Full link dimensions from the Jaco model's OBB definitions.
    constexpr float dimensions[7][3] = {
        {0.100f, 0.091f, 0.160f},
        {0.494f, 0.083f, 0.053f},
        {0.250f, 0.064f, 0.100f},
        {0.072f, 0.064f, 0.096f},
        {0.084f, 0.066f, 0.100f},
        {0.092f, 0.116f, 0.110f},
        {0.084f, 0.089f, 0.1535f}
    };

    // Base link: centered above the ground with identity orientation.
    mat4_t base = identity_matrix();
    base.m[2][3] = dimensions[6][2] * 0.5f;
    set_obb(obbs[0], base, dimensions[6]);

    const mat4_t dh01 = give_dh(0.1535f, 0.0f, 0.0f, 0.0f);
    const mat4_t dh1e = give_dh(0.080f, 0.0f, q[1], 0.0f);

    const mat4_t dh12 =
        give_dh(0.1185f, 0.0f, q[1], -kPi * 0.5f);
    const mat4_t dh2e =
        give_dh(-0.029f, 0.206f, q[2], 0.0f);

    const mat4_t dh23 =
        give_dh(0.0f, 0.410f, q[2], 0.0f);
    const mat4_t dh3e =
        give_dh(0.014f, -0.080f, -kPi * 0.5f + q[3],
                -kPi * 0.5f);

    const mat4_t dh34 =
        give_dh(0.01125f, 0.0f, q[3], -kPi * 0.5f);
    const mat4_t dh4e =
        give_dh(0.207f, 0.0f, q[4], 0.0f);

    const mat4_t dh4ed = multiply(
        give_dh(0.050f, 0.0f, 0.0f, 0.61f),
        give_dh(0.0f, 0.014f, -kPi * 0.5f, 0.0f));

    const mat4_t dh45 =
        give_dh(0.207f + 0.0658f, 0.0f, q[4], kPi + 0.9599f);

    const mat4_t dh5e =
        give_dh(-0.028f, 0.01968f, kPi * 0.5f, 0.0f);
    const mat4_t dh5ed =
        give_dh(-0.050f, 0.010f, -q[5], 0.0f);

    const mat4_t dh56 = multiply(
        give_dh(-0.028f, 0.01968f, kPi * 0.5f, 0.0f),
        give_dh(0.0f, 0.0f, -q[5], 0.0f));

    const mat4_t dh6e = multiply(
        give_dh(0.0f, 0.0f, -kPi * 0.5f, 0.9599f),
        give_dh(-0.0658f, -0.0343f, kPi * 0.5f, 0.0f));

    const mat4_t dh6ed =
        give_dh(-0.055f, 0.0f, -q[6], 0.0f);

    const mat4_t transforms[6] = {
        multiply(dh01, dh1e),

        multiply(
            multiply(dh01, dh12),
            dh2e),

        multiply(
            multiply(
                multiply(dh01, dh12),
                dh23),
            dh3e),

        multiply(
            multiply(
                multiply(
                    multiply(dh01, dh12),
                    dh23),
                dh34),
            multiply(dh4e, dh4ed)),

        multiply(
            multiply(
                multiply(
                    multiply(
                        multiply(dh01, dh12),
                        dh23),
                    dh34),
                dh45),
            multiply(dh5e, dh5ed)),

        multiply(
            multiply(
                multiply(
                    multiply(
                        multiply(
                            multiply(dh01, dh12),
                            dh23),
                        dh34),
                    dh45),
                dh56),
            multiply(dh6e, dh6ed))
    };

    for (uint32_t link = 0; link < 6; ++link) {
        set_obb(obbs[link + 1], transforms[link], dimensions[link]);
    }
}

void generate_obstacles(
    uint32_t num_obstacles,
    std::vector<aabb_t>& obstacles) {

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> position(-1.12f, 1.12f);
    std::uniform_real_distribution<float> scale(0.001f, 0.56f);

    obstacles.resize(num_obstacles);

    for (uint32_t i = 0; i < num_obstacles; ++i) {
        const float sx = scale(rng);
        const float tx = position(rng);
        const float sy = scale(rng);
        const float ty = position(rng);
        const float sz = scale(rng);
        const float tz = position(rng);

        obstacles[i].min_x = tx;
        obstacles[i].min_y = ty;
        obstacles[i].min_z = tz;

        obstacles[i].max_x = tx + sx;
        obstacles[i].max_y = ty + sy;
        obstacles[i].max_z = tz + sz;
    }
}

void generate_motions(
    uint32_t num_motions,
    std::vector<obb_t>& obbs) {

    // The first joint is fixed. The remaining joints are sampled
    // deterministically within representative Jaco joint ranges.
    std::mt19937 rng(7);

    std::uniform_real_distribution<float> q1(-kPi, kPi);
    std::uniform_real_distribution<float> q2(-3.97935067f, 0.83775803f);
    std::uniform_real_distribution<float> q3(-4.41568301f, 1.274090357f);
    std::uniform_real_distribution<float> q456(-kPi, kPi);

    obbs.resize(
        static_cast<size_t>(num_motions) * GEOMETRY_NUM_LINKS);

    for (uint32_t motion = 0; motion < num_motions; ++motion) {
        const float q[7] = {
            0.0f,
            q1(rng),
            q2(rng),
            q3(rng),
            q456(rng),
            q456(rng),
            q456(rng)
        };

        generate_jaco_obbs(
            q,
            obbs.data() + static_cast<size_t>(motion) *
                         GEOMETRY_NUM_LINKS);
    }
}

float abs_float(float value) {
    return value < 0.0f ? -value : value;
}

// CPU reference for the same 15-axis OBB-vs-AABB SAT test
// implemented by the GPU kernel.
bool obb_aabb_intersects(
    const obb_t& obb,
    const aabb_t& aabb) {

    const float a[3] = {
        obb.half_extents[0],
        obb.half_extents[1],
        obb.half_extents[2]
    };

    const float b[3] = {
        (aabb.max_x - aabb.min_x) * 0.5f,
        (aabb.max_y - aabb.min_y) * 0.5f,
        (aabb.max_z - aabb.min_z) * 0.5f
    };

    const float aabb_center[3] = {
        (aabb.min_x + aabb.max_x) * 0.5f,
        (aabb.min_y + aabb.max_y) * 0.5f,
        (aabb.min_z + aabb.max_z) * 0.5f
    };

    const float displacement[3] = {
        aabb_center[0] - obb.center[0],
        aabb_center[1] - obb.center[1],
        aabb_center[2] - obb.center[2]
    };

    float rotation[3][3];
    float abs_rotation[3][3];
    float t[3];

    for (uint32_t i = 0; i < 3; ++i) {
        t[i] = 0.0f;

        for (uint32_t j = 0; j < 3; ++j) {
            rotation[i][j] = obb.rotation[j * 3 + i];
            abs_rotation[i][j] =
                abs_float(rotation[i][j]) + kEpsilon;
            t[i] += displacement[j] * rotation[i][j];
        }
    }

    // OBB face normals.
    for (uint32_t i = 0; i < 3; ++i) {
        const float radius_b =
            b[0] * abs_rotation[i][0] +
            b[1] * abs_rotation[i][1] +
            b[2] * abs_rotation[i][2];

        if (abs_float(t[i]) > a[i] + radius_b) {
            return false;
        }
    }

    // AABB face normals.
    for (uint32_t j = 0; j < 3; ++j) {
        const float radius_a =
            a[0] * abs_rotation[0][j] +
            a[1] * abs_rotation[1][j] +
            a[2] * abs_rotation[2][j];

        if (abs_float(displacement[j]) > b[j] + radius_a) {
            return false;
        }
    }

    // Nine cross-product axes: OBB axis i x AABB axis j.
    for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t i1 = (i + 1) % 3;
        const uint32_t i2 = (i + 2) % 3;

        for (uint32_t j = 0; j < 3; ++j) {
            const uint32_t j1 = (j + 1) % 3;
            const uint32_t j2 = (j + 2) % 3;

            const float radius_a =
                a[i1] * abs_rotation[i2][j] +
                a[i2] * abs_rotation[i1][j];

            const float radius_b =
                b[j1] * abs_rotation[i][j2] +
                b[j2] * abs_rotation[i][j1];

            const float projected_distance =
                abs_float(
                    t[i2] * rotation[i1][j] -
                    t[i1] * rotation[i2][j]);

            if (projected_distance > radius_a + radius_b) {
                return false;
            }
        }
    }

    return true;
}

geometry_result_t cpu_geometry_check(
    uint32_t motion,
    uint32_t num_obstacles,
    const std::vector<obb_t>& obbs,
    const std::vector<aabb_t>& obstacles) {

    geometry_result_t result{};

    for (uint32_t link = 0; link < GEOMETRY_NUM_LINKS; ++link) {
        const obb_t& robot_link =
            obbs[static_cast<size_t>(motion) * GEOMETRY_NUM_LINKS + link];

        for (uint32_t obstacle = 0;
             obstacle < num_obstacles;
             ++obstacle) {

            if (obb_aabb_intersects(robot_link, obstacles[obstacle])) {
                result.link_collision_mask |= (1u << link);
                break;
            }
        }
    }

    result.collision = result.link_collision_mask != 0 ? 1u : 0u;
    return result;
}

} // namespace

int run_geometry_mode(
    const char* kernel_file,
    uint32_t num_motions,
    uint32_t num_obstacles) {

    const size_t obbs_buf_size =
        static_cast<size_t>(num_motions) *
        GEOMETRY_NUM_LINKS * sizeof(obb_t);

    const size_t obstacles_buf_size =
        static_cast<size_t>(num_obstacles) * sizeof(aabb_t);

    const size_t results_buf_size =
        static_cast<size_t>(num_motions) * sizeof(geometry_result_t);

    std::vector<obb_t> h_obbs;
    std::vector<aabb_t> h_obstacles;
    std::vector<geometry_result_t> h_results(num_motions);

    generate_motions(num_motions, h_obbs);
    generate_obstacles(num_obstacles, h_obstacles);

    std::cout << "Jaco OBB geometry workload:"
              << " motions=" << num_motions
              << " links_per_motion=" << GEOMETRY_NUM_LINKS
              << " obstacles=" << num_obstacles
              << " total_robot_obbs=" << h_obbs.size()
              << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = {
        sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0
    };

    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h obbs_buf = nullptr;
    vx_buffer_h obstacles_buf = nullptr;
    vx_buffer_h results_buf = nullptr;

    CHECK(vx_buffer_create(
        dev, obbs_buf_size, VX_MEM_READ, &obbs_buf));

    CHECK(vx_buffer_create(
        dev, obstacles_buf_size, VX_MEM_READ, &obstacles_buf));

    CHECK(vx_buffer_create(
        dev, results_buf_size, VX_MEM_WRITE, &results_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;

    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.mode = KERNEL_MODE_GEOMETRY;
    kernel_arg.num_motions = num_motions;
    kernel_arg.num_obstacles = num_obstacles;

    CHECK(vx_buffer_address(
        obbs_buf, &kernel_arg.obbs_addr));

    CHECK(vx_buffer_address(
        obstacles_buf, &kernel_arg.geometry_obstacles_addr));

    CHECK(vx_buffer_address(
        results_buf, &kernel_arg.geometry_results_addr));

    CHECK(vx_enqueue_write(
        q, obbs_buf, 0,
        h_obbs.data(), obbs_buf_size,
        0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    uint32_t launch_work_items = num_motions;

    CHECK(vx_device_max_occupancy_grid(
        dev, 1, &launch_work_items, grid, block));
    std::printf(
        "Geometry launch: motions=%u grid=%u block=%u total_work_items=%u\n",
        num_motions, grid[0], block[0], grid[0] * block[0]);


    vx_launch_info_t li{};

    li.struct_size = sizeof(li);
    li.kernel = kern;
    li.args_host = &kernel_arg;
    li.args_size = sizeof(kernel_arg);
    li.ndim = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0] = block[0];

    int errors = 0;

    for (uint32_t test_case = 0; test_case < 2; ++test_case) {
        const bool positive_case = (test_case == 0);
        const char* case_name =
            positive_case ? "positive-collision" : "negative-collision";

        // Move every obstacle far away.
        for (auto& obstacle : h_obstacles) {
            obstacle = {
                10.0f, 10.0f, 10.0f,
                11.0f, 11.0f, 11.0f
            };
        }

        // In the positive case, overlap the fixed base link.
        if (positive_case && !h_obstacles.empty()) {
            h_obstacles[0] = {
                -0.01f, -0.01f, 0.06f,
                 0.01f,  0.01f, 0.09f
            };
        }

        std::cout << "\nGeometry case: " << case_name << std::endl;

        CHECK(vx_enqueue_write(
            q, obstacles_buf, 0,
            h_obstacles.data(), obstacles_buf_size,
            0, nullptr, nullptr));

        // Reset the result buffer so every case independently verifies
        // that the GPU writes every motion's result.
        for (auto& result : h_results) {
            result.collision = 0xdeadbeef;
            result.link_collision_mask = 0xcafebabe;
        }

        CHECK(vx_enqueue_write(
            q, results_buf, 0,
            h_results.data(), results_buf_size,
            0, nullptr, nullptr));

        vx_event_h launch_ev = nullptr;
        vx_event_h read_ev = nullptr;

        CHECK(vx_enqueue_launch(
            q, &li, 0, nullptr, &launch_ev));

        CHECK(vx_enqueue_read(
            q, h_results.data(), results_buf,
            0, results_buf_size,
            1, &launch_ev, &read_ev));

        CHECK(vx_event_wait_value(
            read_ev, 1, VX_TIMEOUT_INFINITE));

        uint32_t total_collisions = 0;
        uint32_t total_link_collisions = 0;

        for (uint32_t motion = 0; motion < num_motions; ++motion) {
            const geometry_result_t expected =
                cpu_geometry_check(
                    motion, num_obstacles, h_obbs, h_obstacles);

            const geometry_result_t& actual = h_results[motion];

            std::printf(
                "Motion %3u: collision=%u/%u link_mask=0x%02x/0x%02x\n",
                motion,
                expected.collision,
                actual.collision,
                expected.link_collision_mask,
                actual.link_collision_mask);

            if (actual.collision != expected.collision ||
                actual.link_collision_mask != expected.link_collision_mask) {
                ++errors;
            }

            // Also verify that the CPU reference produces the intended
            // deterministic outcome for this test case.
            if (positive_case) {
                if (expected.collision != 1 ||
                    (expected.link_collision_mask & 0x01u) == 0) {
                    ++errors;
                }
            } else if (expected.collision != 0 ||
                       expected.link_collision_mask != 0) {
                ++errors;
            }

            total_collisions += actual.collision;

            for (uint32_t link = 0; link < GEOMETRY_NUM_LINKS; ++link) {
                total_link_collisions +=
                    (actual.link_collision_mask >> link) & 1u;
            }
        }

        std::cout << "Geometry verification summary:"
                  << " case=" << case_name
                  << " motions=" << num_motions
                  << " motions_with_collision=" << total_collisions
                  << " link_collisions=" << total_link_collisions
                  << std::endl;

        vx_event_release(read_ev);
        vx_event_release(launch_ev);
    }

    vx_buffer_release(results_buf);
    vx_buffer_release(obstacles_buf);
    vx_buffer_release(obbs_buf);

    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);

    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors != 0) {
        std::cout << "Found " << errors << " errors!\nFAILED!"
                  << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
