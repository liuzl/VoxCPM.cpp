#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "test_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;
using Catch::Approx;

namespace voxcpm {
struct AudioVAETestAccess {
    static ggml_tensor* conv(AudioVAE& vae, ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                             int kernel, int stride, int padding, int output_padding) {
        return vae.causal_conv1d(ctx, x, w, nullptr, kernel, stride, 1, padding, output_padding);
    }
    static void set_v2_padding(AudioVAE& vae, bool enabled) { vae.config_.encoder_v2_padding = enabled; }
};
namespace test {

namespace {

const std::string kModelPath = get_model_path();
const std::string kTraceEncodePath = get_trace_path("trace_AudioVAE_encode.jsonl");
const std::string kTraceDecodePath = get_trace_path("trace_AudioVAE_decode.jsonl");
const std::string kVoxCPM2ModelPath = "/root/code/VoxCPM.cpp/models/voxcpm2.gguf";
const std::string kVoxCPM2QuantizedModelPath = "/root/code/VoxCPM.cpp/models/quantized/voxcpm2-q8_0.gguf";
constexpr float kTargetMaxDiff = 2e-4f;
constexpr float kCudaSmokeMaxDiff = 1e-1f;

int cpu_thread_count() {
    if (const char* env = std::getenv("VOXCPM_TEST_THREADS")) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return 2;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::unique_ptr<VoxCPMBackend> try_create_cuda_backend() {
    try {
        return std::make_unique<VoxCPMBackend>(BackendType::CUDA, cpu_thread_count());
    } catch (const std::exception& e) {
        WARN("CUDA backend unavailable, skipping test: " << e.what());
        return nullptr;
    }
}

json load_jsonl_line(const std::string& path, int line_index) {
    std::ifstream file(path);
    REQUIRE(file.is_open());

    std::string line;
    for (int i = 0; i <= line_index; ++i) {
        REQUIRE(std::getline(file, line));
    }
    REQUIRE_FALSE(line.empty());
    return json::parse(line);
}

std::vector<float> flatten_bct_to_tc(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    const json& channels = tensor[0];
    const size_t c = channels.size();
    const size_t t = c > 0 ? channels[0].size() : 0;

    std::vector<float> out;
    out.reserve(t * c);
    for (size_t ci = 0; ci < c; ++ci) {
        REQUIRE(channels[ci].size() == t);
        for (size_t ti = 0; ti < t; ++ti) {
            out.push_back(channels[ci][ti].get<float>());
        }
    }
    return out;
}

std::vector<float> flatten_bt_to_t(const json& tensor) {
    REQUIRE(tensor.is_array());
    REQUIRE(tensor.size() == 1);
    const json& values = tensor[0];
    std::vector<float> out;
    out.reserve(values.size());
    for (const auto& v : values) {
        out.push_back(v.get<float>());
    }
    return out;
}

struct ErrorStats {
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    float rmse = 0.0f;
};

struct TensorStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
};

struct GraphTimingStats {
    double build_ms = 0.0;
    double alloc_ms = 0.0;
    double compute_ms = 0.0;
};

TensorStats compute_tensor_stats(const std::vector<float>& data) {
    REQUIRE_FALSE(data.empty());

    TensorStats stats;
    stats.min_val = std::numeric_limits<float>::max();
    stats.max_val = std::numeric_limits<float>::lowest();

    for (float value : data) {
        stats.min_val = std::min(stats.min_val, value);
        stats.max_val = std::max(stats.max_val, value);
        stats.mean += value;
    }

    stats.mean /= static_cast<float>(data.size());
    return stats;
}

ErrorStats compute_error_stats(const std::vector<float>& actual, const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());

    ErrorStats stats;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        stats.max_abs_diff = std::max(stats.max_abs_diff, static_cast<float>(std::fabs(diff)));
        sum_abs += std::fabs(diff);
        sum_sq += diff * diff;
    }

    const double n = static_cast<double>(actual.size());
    stats.mean_abs_diff = static_cast<float>(sum_abs / n);
    stats.rmse = static_cast<float>(std::sqrt(sum_sq / n));
    return stats;
}

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

TEST_CASE("AudioVAEConfig output sample rate falls back to input sample rate", "[audio_vae][config]") {
    AudioVAEConfig config;

    REQUIRE(config.sample_rate == 16000);
    REQUIRE(config.output_sample_rate() == 16000);

    config.out_sample_rate = 48000;
    REQUIRE(config.output_sample_rate() == 48000);
}

TEST_CASE("AudioVAEConfig keeps encoder and decoder hop lengths separate", "[audio_vae][config]") {
    AudioVAEConfig config;
    config.encoder_rates = {2, 5, 8, 8};
    config.decoder_rates = {8, 6, 5, 2, 2, 2};

    REQUIRE(config.hop_length() == 640);
    REQUIRE(config.decode_hop_length() == 1920);
}

void print_error_stats(const char* label,
                       const std::vector<float>& input,
                       const std::vector<float>& expected,
                       const std::vector<float>& actual,
                       const ErrorStats& stats,
                       float tolerance) {
    const TensorStats input_stats = compute_tensor_stats(input);
    const TensorStats expected_stats = compute_tensor_stats(expected);
    const TensorStats actual_stats = compute_tensor_stats(actual);

    size_t mismatch_count = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            ++mismatch_count;
        }
    }
    const float mismatch_rate = static_cast<float>(mismatch_count) / static_cast<float>(actual.size());

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "elements: " << actual.size() << "\n";
    std::cout << "input range: [" << input_stats.min_val << ", " << input_stats.max_val
              << "], mean=" << input_stats.mean << "\n";
    std::cout << "expected range: [" << expected_stats.min_val << ", " << expected_stats.max_val
              << "], mean=" << expected_stats.mean << "\n";
    std::cout << "actual range: [" << actual_stats.min_val << ", " << actual_stats.max_val
              << "], mean=" << actual_stats.mean << "\n";
    std::cout << "max abs error: " << stats.max_abs_diff << "\n";
    std::cout << "avg abs error: " << stats.mean_abs_diff << "\n";
    std::cout << "rmse: " << stats.rmse << "\n";
    std::cout << "mismatch rate (> " << tolerance << "): " << (mismatch_rate * 100.0f) << "%\n";
}

void print_graph_timing(const char* label, const GraphTimingStats& stats) {
    std::cout << label
              << " graph timing: build=" << stats.build_ms << " ms"
              << ", alloc=" << stats.alloc_ms << " ms"
              << ", compute=" << stats.compute_ms << " ms\n";
}

template <typename PrepareInputFn>
GraphTimingStats run_graph_with_timing(VoxCPMContext& graph_ctx,
                                       VoxCPMBackend& backend,
                                       ggml_tensor* output,
                                       PrepareInputFn&& prepare_input) {
    GraphTimingStats stats;
    ggml_cgraph* graph = graph_ctx.new_graph();
    REQUIRE(graph != nullptr);

    const auto build_start = std::chrono::steady_clock::now();
    graph_ctx.build_forward(graph, output);
    const auto build_end = std::chrono::steady_clock::now();

    const auto alloc_start = build_end;
    backend.alloc_graph(graph);
    const auto alloc_end = std::chrono::steady_clock::now();

    prepare_input();

    const auto compute_start = alloc_end;
    REQUIRE(backend.compute(graph) == GGML_STATUS_SUCCESS);
    const auto compute_end = std::chrono::steady_clock::now();

    stats.build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    stats.alloc_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    stats.compute_ms = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    return stats;
}

}  // namespace

TEST_CASE("AudioVAE preprocess aligns to hop length", "[audio_vae][preprocess]") {
    AudioVAE vae;
    const int sample_rate = vae.config().sample_rate;
    const size_t hop_length = static_cast<size_t>(vae.config().hop_length());

    SECTION("aligned input remains unchanged") {
        std::vector<float> input(hop_length * 5, 0.25f);
        std::vector<float> processed = vae.preprocess(input, sample_rate);
        REQUIRE(processed.size() == hop_length * 5);
        REQUIRE(processed.front() == Approx(0.25f));
        REQUIRE(processed.back() == Approx(0.25f));
    }

    SECTION("unaligned input is right padded") {
        std::vector<float> input(hop_length * 5 + 1, 0.5f);
        std::vector<float> processed = vae.preprocess(input, sample_rate);
        REQUIRE(processed.size() == hop_length * 6);
        REQUIRE(processed[hop_length * 5] == Approx(0.5f));
        REQUIRE(processed.back() == Approx(0.0f));
    }
}

TEST_CASE("AudioVAE loads config and weights from GGUF", "[audio_vae][weights]") {
    if (!file_exists(kModelPath)) {
        WARN("Model file not found, skipping test");
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 4096, 65536);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(vae.config().encoder_dim == 64);
    REQUIRE(vae.config().decoder_dim == 2048);
    REQUIRE(vae.config().latent_dim == 64);
    REQUIRE(vae.config().sample_rate == 44100);
    REQUIRE(vae.config().encoder_rates == std::vector<int>({2, 3, 6, 7, 7}));
    REQUIRE(vae.config().decoder_rates == std::vector<int>({7, 7, 6, 3, 2}));

    REQUIRE(vae.weights().encoder_block_0_weight != nullptr);
    REQUIRE(vae.weights().encoder_fc_mu_weight != nullptr);
    REQUIRE(vae.weights().decoder_model_0_weight != nullptr);
    REQUIRE(vae.weights().decoder_model_1_weight != nullptr);
    REQUIRE(vae.weights().decoder_final_conv_weight != nullptr);
    REQUIRE(vae.weights().encoder_blocks.size() == 5);
    REQUIRE(vae.weights().decoder_blocks.size() == 5);
}

TEST_CASE("AudioVAE loads VoxCPM2 sample-rate conditioning from GGUF", "[audio_vae][weights][voxcpm2][sr-cond]") {
    if (!file_exists(kVoxCPM2ModelPath)) {
        WARN("VoxCPM2 GGUF not found, skipping test");
        return;
    }

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kVoxCPM2ModelPath, weight_ctx, graph_ctx, backend));

    REQUIRE(vae.config().sample_rate == 16000);
    REQUIRE(vae.config().output_sample_rate() == 48000);
    REQUIRE(vae.config().sr_bin_boundaries == std::vector<int>({20000, 30000, 40000}));
    REQUIRE(vae.config().sample_rate_bucket(16000) == 0);
    REQUIRE(vae.config().sample_rate_bucket(20000) == 0);
    REQUIRE(vae.config().sample_rate_bucket(20001) == 1);
    REQUIRE(vae.config().sample_rate_bucket(48000) == 3);

    REQUIRE(vae.weights().decoder_blocks.size() == 6);
    REQUIRE(vae.weights().decoder_blocks.front().sr_cond.scale_embed != nullptr);
    REQUIRE(vae.weights().decoder_blocks.front().sr_cond.bias_embed != nullptr);
    REQUIRE(vae.weights().decoder_blocks.front().sr_cond.scale_embed->ne[0] == 2048);
    REQUIRE(vae.weights().decoder_blocks.front().sr_cond.scale_embed->ne[1] == 4);

    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, 8, vae.config().latent_dim);
    REQUIRE(latent != nullptr);
    ggml_set_input(latent);
    ggml_tensor* audio = vae.decode(graph_ctx, backend, latent);
    REQUIRE(audio != nullptr);
    REQUIRE(vae.last_decode_sr_cond_tensor() != nullptr);
    REQUIRE(vae.last_decode_sr_bucket() == 3);

    const std::vector<float> latent_input(static_cast<size_t>(8 * vae.config().latent_dim), 0.0f);
    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, backend, audio, [&]() {
        backend.tensor_set(latent, latent_input.data(), 0, latent_input.size() * sizeof(float));
        vae.prepare_decode_inputs(backend);
    });

    std::vector<float> actual(static_cast<size_t>(ggml_nelements(audio)), 0.0f);
    backend.tensor_get(audio, actual.data(), 0, actual.size() * sizeof(float));
    REQUIRE(all_finite(actual));
    print_graph_timing("AudioVAE VoxCPM2 sr-cond decode", timing);
}

TEST_CASE("AudioVAE encode matches trace", "[audio_vae][encode][trace]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceEncodePath)) {
        WARN("Encode test dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTraceEncodePath, 0);
    std::vector<float> input_audio = flatten_bt_to_t(trace.at("inputs").at("audio_data"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    std::vector<float> audio_copy = input_audio;
    ggml_tensor* latent = vae.encode(graph_ctx, backend, audio_copy, 44100);
    REQUIRE(latent != nullptr);
    REQUIRE(latent->ne[0] == 216);
    REQUIRE(latent->ne[1] == 64);
    REQUIRE(latent->ne[2] == 1);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, backend, latent, [&]() {
        backend.tensor_set(vae.last_input_tensor(),
                           vae.last_preprocessed_audio().data(),
                           0,
                           vae.last_preprocessed_audio().size() * sizeof(float));
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(latent, actual.data(), 0, actual.size() * sizeof(float));

    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE encode trace output", input_audio, expected, actual, stats, kTargetMaxDiff);
    print_graph_timing("AudioVAE encode", timing);
    INFO("encode max_abs_diff = " << stats.max_abs_diff);
    INFO("encode mean_abs_diff = " << stats.mean_abs_diff);
    INFO("encode rmse = " << stats.rmse);
    REQUIRE(stats.max_abs_diff <= kTargetMaxDiff);
}

TEST_CASE("AudioVAE decode matches trace", "[audio_vae][decode][trace]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceDecodePath)) {
        WARN("Decode test dependencies missing, skipping test");
        return;
    }

    const json trace = load_jsonl_line(kTraceDecodePath, 0);
    const std::vector<float> latent_input = flatten_bct_to_tc(trace.at("inputs").at("z"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMBackend backend(BackendType::CPU, cpu_thread_count());
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, backend));

    ggml_tensor* latent = graph_ctx.new_tensor_3d(GGML_TYPE_F32, 36, 64, 1);
    REQUIRE(latent != nullptr);
    ggml_set_input(latent);
    ggml_tensor* audio = vae.decode(graph_ctx, backend, latent);
    REQUIRE(audio != nullptr);
    REQUIRE(audio->ne[0] == 63504);
    REQUIRE(audio->ne[1] == 1);
    REQUIRE(audio->ne[2] == 1);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, backend, audio, [&]() {
        backend.tensor_set(latent, latent_input.data(), 0, latent_input.size() * sizeof(float));
        vae.prepare_decode_inputs(backend);
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend.tensor_get(audio, actual.data(), 0, actual.size() * sizeof(float));

    const auto [min_it, max_it] = std::minmax_element(actual.begin(), actual.end());
    REQUIRE(*min_it >= -1.001f);
    REQUIRE(*max_it <= 1.001f);

    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE decode trace output", latent_input, expected, actual, stats, kTargetMaxDiff);
    print_graph_timing("AudioVAE decode", timing);
    INFO("decode max_abs_diff = " << stats.max_abs_diff);
    INFO("decode mean_abs_diff = " << stats.mean_abs_diff);
    INFO("decode rmse = " << stats.rmse);
    REQUIRE(stats.max_abs_diff <= kTargetMaxDiff);
}

TEST_CASE("AudioVAE encode CUDA smoke", "[audio_vae][encode][cuda][smoke]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceEncodePath)) {
        WARN("Encode test dependencies missing, skipping test");
        return;
    }

    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    const json trace = load_jsonl_line(kTraceEncodePath, 0);
    std::vector<float> input_audio = flatten_bt_to_t(trace.at("inputs").at("audio_data"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, *backend));

    std::vector<float> audio_copy = input_audio;
    ggml_tensor* latent = vae.encode(graph_ctx, *backend, audio_copy, 44100);
    REQUIRE(latent != nullptr);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, *backend, latent, [&]() {
        backend->tensor_set(vae.last_input_tensor(),
                            vae.last_preprocessed_audio().data(),
                            0,
                            vae.last_preprocessed_audio().size() * sizeof(float));
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend->tensor_get(latent, actual.data(), 0, actual.size() * sizeof(float));

    REQUIRE(all_finite(actual));
    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE encode CUDA output", input_audio, expected, actual, stats, kCudaSmokeMaxDiff);
    print_graph_timing("AudioVAE encode CUDA", timing);
    INFO("encode CUDA max_abs_diff = " << stats.max_abs_diff);
    REQUIRE(stats.max_abs_diff <= kCudaSmokeMaxDiff);
}

TEST_CASE("AudioVAE decode CUDA smoke", "[audio_vae][decode][cuda][smoke]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceDecodePath)) {
        WARN("Decode test dependencies missing, skipping test");
        return;
    }

    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    const json trace = load_jsonl_line(kTraceDecodePath, 0);
    const std::vector<float> latent_input = flatten_bct_to_tc(trace.at("inputs").at("z"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, *backend));

    ggml_tensor* latent = graph_ctx.new_tensor_3d(GGML_TYPE_F32, 36, 64, 1);
    REQUIRE(latent != nullptr);
    ggml_set_input(latent);
    ggml_tensor* audio = vae.decode(graph_ctx, *backend, latent);
    REQUIRE(audio != nullptr);

    const GraphTimingStats timing = run_graph_with_timing(graph_ctx, *backend, audio, [&]() {
        backend->tensor_set(latent, latent_input.data(), 0, latent_input.size() * sizeof(float));
        vae.prepare_decode_inputs(*backend);
    });

    std::vector<float> actual(expected.size(), 0.0f);
    backend->tensor_get(audio, actual.data(), 0, actual.size() * sizeof(float));

    REQUIRE(all_finite(actual));
    const auto [min_it, max_it] = std::minmax_element(actual.begin(), actual.end());
    REQUIRE(*min_it >= -1.01f);
    REQUIRE(*max_it <= 1.01f);

    const ErrorStats stats = compute_error_stats(actual, expected);
    print_error_stats("AudioVAE decode CUDA output", latent_input, expected, actual, stats, kCudaSmokeMaxDiff);
    print_graph_timing("AudioVAE decode CUDA", timing);
    INFO("decode CUDA max_abs_diff = " << stats.max_abs_diff);
    REQUIRE(stats.max_abs_diff <= kCudaSmokeMaxDiff);
}

TEST_CASE("AudioVAE stateful streaming decode CUDA matches trace", "[audio_vae][decode][cuda][streaming]") {
    if (!file_exists(kModelPath) || !file_exists(kTraceDecodePath)) {
        WARN("Decode test dependencies missing, skipping test");
        return;
    }

    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    const json trace = load_jsonl_line(kTraceDecodePath, 0);
    const std::vector<float> latent_input = flatten_bct_to_tc(trace.at("inputs").at("z"));
    const std::vector<float> expected = flatten_bct_to_tc(trace.at("outputs").at("output"));

    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kModelPath, weight_ctx, graph_ctx, *backend));
    REQUIRE(vae.supports_streaming_decode(*backend));

    AudioVAEStreamingDecodeState stream_state;
    REQUIRE(vae.initialize_streaming_decode_state(*backend, stream_state));
    REQUIRE(stream_state.slot_count() > 0);

    constexpr int kTotalPatches = 36;
    constexpr int kChunkPatches = 9;
    constexpr int kLatentDim = 64;
    std::vector<float> streamed;
    streamed.reserve(expected.size());

    for (int patch_offset = 0; patch_offset < kTotalPatches; patch_offset += kChunkPatches) {
        const int chunk_patches = std::min(kChunkPatches, kTotalPatches - patch_offset);
        std::vector<float> chunk(static_cast<size_t>(chunk_patches) * kLatentDim, 0.0f);
        for (int d = 0; d < kLatentDim; ++d) {
            for (int t = 0; t < chunk_patches; ++t) {
                chunk[static_cast<size_t>(d) * chunk_patches + t] =
                    latent_input[static_cast<size_t>(d) * kTotalPatches + patch_offset + t];
            }
        }

        VoxCPMContext chunk_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* latent = chunk_ctx.new_tensor_3d(GGML_TYPE_F32, chunk_patches, kLatentDim, 1);
        REQUIRE(latent != nullptr);
        ggml_set_input(latent);
        ggml_tensor* audio = vae.decode_streaming(chunk_ctx, *backend, latent, stream_state);
        REQUIRE(audio != nullptr);

        ggml_cgraph* graph = chunk_ctx.new_graph();
        REQUIRE(graph != nullptr);
        chunk_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend->alloc_graph(graph);
        backend->tensor_set(latent, chunk.data(), 0, chunk.size() * sizeof(float));
        vae.prepare_decode_inputs(*backend);
        REQUIRE(backend->compute(graph) == GGML_STATUS_SUCCESS);
        stream_state.publish_updates(*backend);

        std::vector<float> chunk_audio(static_cast<size_t>(ggml_nelements(audio)), 0.0f);
        backend->tensor_get(audio, chunk_audio.data(), 0, chunk_audio.size() * sizeof(float));
        streamed.insert(streamed.end(), chunk_audio.begin(), chunk_audio.end());
    }

    REQUIRE(streamed.size() == expected.size());
    REQUIRE(all_finite(streamed));

    const ErrorStats stats = compute_error_stats(streamed, expected);
    print_error_stats("AudioVAE stateful streaming decode CUDA output", latent_input, expected, streamed, stats, kCudaSmokeMaxDiff);
    INFO("stateful streaming decode CUDA max_abs_diff = " << stats.max_abs_diff);
    REQUIRE(stats.max_abs_diff <= kCudaSmokeMaxDiff);
}

TEST_CASE("AudioVAE stateful streaming decode CUDA matches full decode", "[audio_vae][decode][cuda][streaming][voxcpm2]") {
    if (!file_exists(kVoxCPM2QuantizedModelPath)) {
        WARN("VoxCPM2 quantized GGUF not found, skipping test");
        return;
    }

    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);

    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(kVoxCPM2QuantizedModelPath, weight_ctx, graph_ctx, *backend));
    REQUIRE(vae.supports_streaming_decode(*backend));

    const int total_patches = 32;
    const int latent_dim = vae.config().latent_dim;
    std::vector<float> latent_input(static_cast<size_t>(total_patches) * latent_dim, 0.0f);
    for (int d = 0; d < latent_dim; ++d) {
        for (int t = 0; t < total_patches; ++t) {
            latent_input[static_cast<size_t>(d) * total_patches + t] =
                0.1f * std::sin(static_cast<float>(d * 17 + t * 31) * 0.013f);
        }
    }

    ggml_tensor* full_latent = graph_ctx.new_tensor_3d(GGML_TYPE_F32, total_patches, latent_dim, 1);
    REQUIRE(full_latent != nullptr);
    ggml_set_input(full_latent);
    ggml_tensor* full_audio = vae.decode(graph_ctx, *backend, full_latent);
    REQUIRE(full_audio != nullptr);

    run_graph_with_timing(graph_ctx, *backend, full_audio, [&]() {
        backend->tensor_set(full_latent, latent_input.data(), 0, latent_input.size() * sizeof(float));
        vae.prepare_decode_inputs(*backend);
    });

    std::vector<float> expected(static_cast<size_t>(ggml_nelements(full_audio)), 0.0f);
    backend->tensor_get(full_audio, expected.data(), 0, expected.size() * sizeof(float));

    AudioVAEStreamingDecodeState stream_state;
    REQUIRE(vae.initialize_streaming_decode_state(*backend, stream_state));

    constexpr int kChunkPatches = 8;
    std::vector<float> streamed;
    streamed.reserve(expected.size());
    for (int patch_offset = 0; patch_offset < total_patches; patch_offset += kChunkPatches) {
        const int chunk_patches = std::min(kChunkPatches, total_patches - patch_offset);
        std::vector<float> chunk(static_cast<size_t>(chunk_patches) * latent_dim, 0.0f);
        for (int d = 0; d < latent_dim; ++d) {
            for (int t = 0; t < chunk_patches; ++t) {
                chunk[static_cast<size_t>(d) * chunk_patches + t] =
                    latent_input[static_cast<size_t>(d) * total_patches + patch_offset + t];
            }
        }

        VoxCPMContext chunk_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* latent = chunk_ctx.new_tensor_3d(GGML_TYPE_F32, chunk_patches, latent_dim, 1);
        REQUIRE(latent != nullptr);
        ggml_set_input(latent);
        ggml_tensor* audio = vae.decode_streaming(chunk_ctx, *backend, latent, stream_state);
        REQUIRE(audio != nullptr);

        ggml_cgraph* graph = chunk_ctx.new_graph();
        REQUIRE(graph != nullptr);
        chunk_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend->alloc_graph(graph);
        backend->tensor_set(latent, chunk.data(), 0, chunk.size() * sizeof(float));
        vae.prepare_decode_inputs(*backend);
        REQUIRE(backend->compute(graph) == GGML_STATUS_SUCCESS);
        stream_state.publish_updates(*backend);

        std::vector<float> chunk_audio(static_cast<size_t>(ggml_nelements(audio)), 0.0f);
        backend->tensor_get(audio, chunk_audio.data(), 0, chunk_audio.size() * sizeof(float));
        streamed.insert(streamed.end(), chunk_audio.begin(), chunk_audio.end());
    }

    REQUIRE(streamed.size() == expected.size());
    REQUIRE(all_finite(streamed));

    const ErrorStats stats = compute_error_stats(streamed, expected);
    print_error_stats("AudioVAE stateful streaming decode CUDA vs full decode", latent_input, expected, streamed, stats, kCudaSmokeMaxDiff);
    INFO("stateful streaming vs full decode CUDA max_abs_diff = " << stats.max_abs_diff);
    REQUIRE(stats.max_abs_diff <= kCudaSmokeMaxDiff);
}


TEST_CASE("AudioVAE causal strided convolution preserves V1 and V2 alignment", "[audio_vae][padding]") {
    for (int stride : {2, 3, 5, 7, 8}) {
        for (bool v2 : {false, true}) {
            CAPTURE(stride, v2);
            VoxCPMBackend backend(BackendType::CPU, 2);
            VoxCPMContext weights(ContextType::Weights, 32);
            VoxCPMContext graph_ctx(ContextType::Graph, 512, 2048);
            AudioVAE vae;
            const int kernel = stride * 2;
            const int samples = 43;
            auto* weight = weights.new_tensor_3d(GGML_TYPE_F32, kernel, 2, 2);
            auto buffer = backend.alloc_buffer(weights.raw_context(), BufferUsage::Weights);
            std::vector<float> w(kernel * 4), input(samples * 2);
            for (size_t i = 0; i < w.size(); ++i) w[i] = (static_cast<int>(i % 11) - 5) * 0.125f;
            for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 17) - 8) * 0.25f;
            backend.tensor_set(weight, w.data(), 0, w.size() * sizeof(float));
            auto* x = graph_ctx.new_tensor_3d(GGML_TYPE_F32, samples, 2, 1);
            ggml_set_input(x);
            const int padding = (stride + 1) / 2;
            auto* output = AudioVAETestAccess::conv(vae, graph_ctx.raw_context(), x, weight,
                                                    kernel, stride, padding, v2 ? stride % 2 : 0);
            run_graph_with_timing(graph_ctx, backend, output, [&]() {
                backend.tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            });
            // Independent cross-correlation oracle; no GGML layout/padding helpers.
            const int left = v2 ? stride : 2 * ((stride + 1) / 2);
            const int frames = (samples + left - kernel) / stride + 1;
            REQUIRE(output->ne[0] == frames);
            std::vector<float> actual(frames * 2);
            backend.tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            for (int oc = 0; oc < 2; ++oc) {
                for (int t = 0; t < frames; ++t) {
                    float expected = 0;
                    for (int ic = 0; ic < 2; ++ic) {
                        for (int k = 0; k < kernel; ++k) {
                            const int source = t * stride + k - left;
                            if (source >= 0 && source < samples) {
                                expected += input[ic * samples + source] * w[(oc * 2 + ic) * kernel + k];
                            }
                        }
                    }
                    REQUIRE(actual[oc * frames + t] == Approx(expected).margin(1e-5));
                }
            }
            backend.free_buffer(buffer);
        }
    }
}

TEST_CASE("AudioVAE encoder matches official Python with identical GGUF weights", "[audio_vae][encoder-parity]") {
    const char* fixture = std::getenv("VOXCPM_ENCODER_PARITY_JSON");
    if (!fixture) SKIP("Set VOXCPM_ENCODER_PARITY_JSON using scripts/export_encoder_parity.py");
    std::ifstream stream(fixture);
    REQUIRE(stream.is_open());
    const json traces = json::parse(stream);
    REQUIRE(traces.at("schema") == "voxcpm.encoder-parity.v1");
    REQUIRE(traces.at("cases").size() == 8);
    BackendType type = BackendType::CPU;
    if (const char* name = std::getenv("VOXCPM_TEST_BACKEND")) {
        REQUIRE((std::string(name) == "cpu" || std::string(name) == "metal" || std::string(name) == "cuda"));
        if (std::string(name) == "metal") type = BackendType::Metal;
        if (std::string(name) == "cuda") type = BackendType::CUDA;
    }
    VoxCPMBackend backend(type, 2);
    VoxCPMContext weight_ctx(ContextType::Weights, 512);
    VoxCPMContext load_ctx(ContextType::Graph, 512, 2048);
    AudioVAE vae;
    REQUIRE(vae.load_from_gguf(get_model_path(), weight_ctx, load_ctx, backend));
    REQUIRE(vae.config().encoder_v2_padding);
    const int sample_rate = traces.at("sample_rate");
    for (const auto& trace : traces.at("cases")) {
        CAPTURE(trace.at("name"));
        std::vector<float> input = trace.at("audio").get<std::vector<float>>();
        const auto expected = trace.at("latent").get<std::vector<float>>();
        auto run = [&](bool v2) {
            AudioVAETestAccess::set_v2_padding(vae, v2);
            VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);
            auto* latent = vae.encode(graph_ctx, backend, input, sample_rate);
            REQUIRE(latent->ne[0] == trace.at("shape").at(2).get<int>());
            REQUIRE(latent->ne[1] == trace.at("shape").at(1).get<int>());
            run_graph_with_timing(graph_ctx, backend, latent, [&]() {
                backend.tensor_set(vae.last_input_tensor(), vae.last_preprocessed_audio().data(), 0,
                                   vae.last_preprocessed_audio().size() * sizeof(float));
            });
            std::vector<float> result(expected.size());
            backend.tensor_get(latent, result.data(), 0, result.size() * sizeof(float));
            backend.reset_request_state();
            return result;
        };
        const bool v2 = trace.at("v2");
        const auto actual = run(v2);
        REQUIRE(all_finite(actual));
        const auto stats = compute_error_stats(actual, expected);
        std::cout << trace.at("name") << " max_abs=" << stats.max_abs_diff << " rmse=" << stats.rmse;
        if (v2) {
            const auto old = compute_error_stats(run(false), expected);
            std::cout << " legacy_padding_rmse=" << old.rmse;
            REQUIRE(old.rmse > stats.rmse * 10);
        }
        std::cout << "\n";
        CHECK(stats.max_abs_diff < 0.002f);
        CHECK(stats.rmse < 0.0002f);
    }
}

}  // namespace test
}  // namespace voxcpm
