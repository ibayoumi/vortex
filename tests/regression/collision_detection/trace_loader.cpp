#include "trace_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {

const char TRACE_MAGIC[8] = {
    'M', 'P', 'N', 'E', 'T', '7', 'D', '\0'
};

static_assert(sizeof(trace_feature_t) == 12, "trace_feature_t must contain exactly three float32 values");

const uint32_t EXPECTED_VERSION = 1;
const uint32_t LINKS_PER_POSE = 7;

template <typename T>
bool read_value(std::ifstream& file, T* value) {
    file.read(
        reinterpret_cast<char*>(value),
        static_cast<std::streamsize>(sizeof(T))
    );

    return file.good();
}

template <typename T>
bool read_array(
    std::ifstream& file,
    uint64_t count,
    std::vector<T>* values) {

    if (count > values->max_size()) {
        return false;
    }

    if (count > static_cast<uint64_t>(
            std::numeric_limits<std::streamsize>::max()) / sizeof(T)) {
        return false;
    }

    values->resize(static_cast<size_t>(count));

    const uint64_t bytes = count * sizeof(T);

    if (bytes != 0) {
        file.read(
            reinterpret_cast<char*>(values->data()),
            static_cast<std::streamsize>(bytes)
        );
    }

    return file.good();
}

bool validate_offsets(
    const std::vector<uint64_t>& offsets,
    uint64_t expected_end,
    const char* name,
    std::string* error) {

    if (offsets.empty()) {
        *error = std::string(name) + " is empty";
        return false;
    }

    if (offsets.front() != 0) {
        *error = std::string(name) + " must start at zero";
        return false;
    }

    if (offsets.back() != expected_end) {
        *error = std::string(name) + " has an invalid final offset";
        return false;
    }

    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            *error = std::string(name) + " is not monotonic";
            return false;
        }
    }

    return true;
}

} // namespace

bool load_mpnet_trace(
    const char* filename,
    mpnet_trace_t* trace,
    std::string* error) {

    if (filename == nullptr || trace == nullptr || error == nullptr) {
        return false;
    }

    *trace = mpnet_trace_t{};
    error->clear();

    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        *error = std::string("Cannot open trace file: ") + filename;
        return false;
    }

    // Determine the file size before allocating memory.
    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < 40) {
        *error = "Trace file is smaller than its 40-byte header";
        return false;
    }

    // Header layout: <8sIIQQQ
    char magic[8];
    uint32_t version = 0;
    uint32_t num_benchmarks = 0;
    uint64_t num_motions = 0;
    uint64_t num_poses = 0;
    uint64_t num_queries = 0;

    if (!file.read(magic, sizeof(magic)) ||
        !read_value(file, &version) ||
        !read_value(file, &num_benchmarks) ||
        !read_value(file, &num_motions) ||
        !read_value(file, &num_poses) ||
        !read_value(file, &num_queries)) {
        *error = "Failed to read trace header";
        return false;
    }

    if (std::memcmp(magic, TRACE_MAGIC, sizeof(magic)) != 0) {
        *error = "Invalid trace magic value";
        return false;
    }

    if (version != EXPECTED_VERSION) {
        std::ostringstream ss;
        ss << "Unsupported trace version: " << version;
        *error = ss.str();
        return false;
    }

    if (num_benchmarks == 0 ||
        num_motions == 0 ||
        num_poses == 0 ||
        num_queries == 0) {
        *error = "Trace header contains a zero count";
        return false;
    }

    // Validate the expected file size before allocating the arrays.
    const uint64_t expected_size =
        40ULL
        + 4ULL * num_benchmarks
        + 8ULL * (static_cast<uint64_t>(num_benchmarks) + 1)
        + 8ULL * (num_motions + 1)
        + 8ULL * (num_poses + 1)
        + 12ULL * num_queries
        + num_queries;

    if (static_cast<uint64_t>(file_size) != expected_size) {
        std::ostringstream ss;
        ss << "Trace file size mismatch: expected "
           << expected_size << " bytes, got " << file_size;
        *error = ss.str();
        return false;
    }

    trace->version = version;

    if (!read_array(file, num_benchmarks, &trace->benchmark_ids)) {
        *error = "Failed to read benchmark IDs";
        return false;
    }

    if (!read_array(
            file, static_cast<uint64_t>(num_benchmarks) + 1,
            &trace->benchmark_offsets)) {
        *error = "Failed to read benchmark offsets";
        return false;
    }

    if (!read_array(
            file, num_motions + 1, &trace->motion_offsets)) {
        *error = "Failed to read motion offsets";
        return false;
    }

    if (!read_array(
            file, num_poses + 1, &trace->pose_offsets)) {
        *error = "Failed to read pose offsets";
        return false;
    }

    if (!read_array(file, num_queries, &trace->features)) {
        *error = "Failed to read query features";
        return false;
    }

    if (!read_array(file, num_queries, &trace->labels)) {
        *error = "Failed to read query labels";
        return false;
    }

    // Validate all three levels of the hierarchy.
    if (!validate_offsets(
            trace->benchmark_offsets,
            num_motions,
            "Benchmark offsets",
            error)) {
        return false;
    }

    if (!validate_offsets(
            trace->motion_offsets,
            num_poses,
            "Motion offsets",
            error)) {
        return false;
    }

    if (!validate_offsets(
            trace->pose_offsets,
            num_queries,
            "Pose offsets",
            error)) {
        return false;
    }

    // Every pose in this dataset contains exactly seven link queries.
    for (size_t i = 1; i < trace->pose_offsets.size(); ++i) {
        const uint64_t queries_in_pose =
            trace->pose_offsets[i] - trace->pose_offsets[i - 1];

        if (queries_in_pose != LINKS_PER_POSE) {
            std::ostringstream ss;
            ss << "Pose " << (i - 1)
               << " has " << queries_in_pose
               << " queries instead of " << LINKS_PER_POSE;
            *error = ss.str();
            return false;
        }
    }

    // Every motion must contain at least one pose.
    for (size_t i = 1; i < trace->motion_offsets.size(); ++i) {
        if (trace->motion_offsets[i] == trace->motion_offsets[i - 1]) {
            std::ostringstream ss;
            ss << "Motion " << (i - 1) << " contains no poses";
            *error = ss.str();
            return false;
        }
    }

    // Validate the recorded collision labels.
    uint64_t collision_count = 0;
    uint64_t free_count = 0;

    for (uint8_t label : trace->labels) {
        if (label == 0) {
            ++collision_count;
        } else if (label == 1) {
            ++free_count;
        } else {
            *error = "Encountered a label other than 0 or 1";
            return false;
        }
    }

    if (collision_count + free_count != num_queries) {
        *error = "Label count does not match query count";
        return false;
    }

    return true;
}
