#ifndef COLLISION_DETECTION_TRACE_LOADER_H
#define COLLISION_DETECTION_TRACE_LOADER_H

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

struct mpnet_trace_t {
    uint32_t version = 0;

    std::vector<int32_t> benchmark_ids;

    // Benchmark index -> motion index.
    std::vector<uint64_t> benchmark_offsets;

    // Motion index -> pose index.
    std::vector<uint64_t> motion_offsets;

    // Pose index -> query index.
    std::vector<uint64_t> pose_offsets;

    // One three-dimensional feature vector per query.
    std::vector<trace_feature_t> features;

    // 0 = collision, 1 = collision-free.
    std::vector<uint8_t> labels;

    uint64_t num_motions() const {
        return motion_offsets.empty() ? 0 : motion_offsets.size() - 1;
    }

    uint64_t num_poses() const {
        return pose_offsets.empty() ? 0 : pose_offsets.size() - 1;
    }

    uint64_t num_queries() const {
        return features.size();
    }
};

bool load_mpnet_trace(
    const char* filename,
    mpnet_trace_t* trace,
    std::string* error);

#endif
