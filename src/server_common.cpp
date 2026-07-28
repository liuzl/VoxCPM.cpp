#include "voxcpm/server_common.h"

#include "voxcpm/audio_io.h"
#include "voxcpm/context.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace voxcpm {

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

int env_int_or_default(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }

    try {
        return std::max(1, std::stoi(raw));
    } catch (const std::exception&) {
        return default_value;
    }
}

std::filesystem::path manifest_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "manifest.json";
}

std::filesystem::path prompt_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "prompt_feat.bin";
}

std::filesystem::path reference_path_for(const std::string& root, const std::string& id) {
    return std::filesystem::path(root) / id / "reference_feat.bin";
}

constexpr int32_t kAudioStartToken = 101;
constexpr int32_t kRefAudioStartToken = 103;
constexpr int32_t kRefAudioEndToken = 104;

enum class PaddingMode {
    Left,
    Right,
};

struct PreparedConditioning {
    std::vector<int32_t> full_text_tokens;
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;
    std::vector<float> feat;
};

size_t skip_ascii_whitespace(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

std::pair<std::string, bool> strip_hifi_control_prefix(const std::string& text) {
    const size_t start = skip_ascii_whitespace(text, 0);
    if (start >= text.size()) {
        return {text, false};
    }

    size_t content_start = std::string::npos;
    size_t close_pos = std::string::npos;
    size_t close_len = 0;
    if (text.compare(start, 1, "(") == 0) {
        content_start = start + 1;
        close_pos = text.find(')', content_start);
        close_len = 1;
    } else if (text.compare(start, 3, "（") == 0) {
        content_start = start + 3;
        close_pos = text.find("）", content_start);
        close_len = 3;
    }
    if (close_pos == std::string::npos) {
        return {text, false};
    }

    const size_t next = skip_ascii_whitespace(text, close_pos + close_len);
    return {text.substr(next), true};
}

void write_binary_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fail("Failed to open file for writing: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        fail("Failed to open file for reading: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(float))) != 0) {
        fail("Invalid prompt feature blob: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float), 0.0f);
    in.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}

void pad_audio_for_patch_alignment(std::vector<float>& audio, size_t patch_len, PaddingMode mode) {
    if (patch_len == 0 || audio.empty() || (audio.size() % patch_len) == 0) {
        return;
    }
    const size_t padding = patch_len - (audio.size() % patch_len);
    if (mode == PaddingMode::Left) {
        audio.insert(audio.begin(), padding, 0.0f);
    } else {
        audio.insert(audio.end(), padding, 0.0f);
    }
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, backend, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    ggml_tensor* patch_major = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), latent));
    graph_ctx.build_forward(graph, patch_major);
    backend.reserve_compute_memory(graph, "server.audio_vae.encode");
    backend.alloc_graph(graph, "server.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int latent_dim = static_cast<int>(latent->ne[1]);
    if (latent_dim != feat_dim) {
        fail("Prompt latent dim mismatch");
    }
    if (total_patches % patch_size != 0) {
        fail("Prompt latent patches are not divisible by patch size");
    }

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * patch_size * feat_dim, 0.0f);
    backend.tensor_get(patch_major, features.data(), 0, features.size() * sizeof(float));
    return features;
}

PreparedConditioning build_conditioning(ChineseCharSplitTokenizer& split_tokenizer,
                                        const std::string& effective_target_text,
                                        const PromptFeatures& prompt,
                                        int patch_size,
                                        int feat_dim) {
    const int prompt_audio_length = prompt.prompt_audio_length;
    const int reference_audio_length = prompt.reference_audio_length;
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const std::string main_text =
        prompt_audio_length > 0 ? prompt.prompt_text + effective_target_text : effective_target_text;

    std::vector<int32_t> text_tokens = split_tokenizer.encode(main_text, false);
    text_tokens.push_back(kAudioStartToken);

    const size_t total_frames = static_cast<size_t>(reference_audio_length) + 2 +
                                static_cast<size_t>(text_tokens.size()) +
                                static_cast<size_t>(prompt_audio_length);
    PreparedConditioning prepared;
    prepared.full_text_tokens.reserve(total_frames);
    prepared.text_mask.reserve(total_frames);
    prepared.feat_mask.reserve(total_frames);
    prepared.feat.reserve(total_frames * frame_stride);

    const auto append_zero_frame = [&]() {
        prepared.feat.insert(prepared.feat.end(), frame_stride, 0.0f);
    };
    const auto append_feat_frames = [&](const std::vector<float>& frames, int frame_count) {
        prepared.feat.insert(prepared.feat.end(),
                             frames.begin(),
                             frames.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(frame_count) * frame_stride));
    };

    if (reference_audio_length > 0) {
        prepared.full_text_tokens.push_back(kRefAudioStartToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();

        for (int i = 0; i < reference_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prompt.reference_feat, reference_audio_length);

        prepared.full_text_tokens.push_back(kRefAudioEndToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    for (int32_t token : text_tokens) {
        prepared.full_text_tokens.push_back(token);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    if (prompt_audio_length > 0) {
        for (int i = 0; i < prompt_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prompt.prompt_feat, prompt_audio_length);
    }

    return prepared;
}

std::vector<float> decode_audio(AudioVAE& audio_vae,
                                VoxCPMBackend& backend,
                                const std::vector<float>& features,
                                int total_patches,
                                int feat_dim) {
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode");
    backend.alloc_graph(graph, "server.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

std::vector<float> decode_audio_from_patch_major_frames(AudioVAE& audio_vae,
                                                        VoxCPMBackend& backend,
                                                        const std::vector<float>& frames,
                                                        int patch_size,
                                                        int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    if (frames.empty() || (frames.size() % frame_stride) != 0) {
        return {};
    }

    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* patch_major = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_dim, total_patches);
    ggml_set_input(patch_major);
    ggml_tensor* latent = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), patch_major));
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph from patch-major frames");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "server.audio_vae.decode.patch_major");
    backend.alloc_graph(graph, "server.audio_vae.decode.patch_major");
    backend.tensor_set(patch_major, frames.data(), 0, frames.size() * sizeof(float));
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode from patch-major frames failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

std::vector<float> decode_audio_from_output_pool(AudioVAE& audio_vae,
                                                 VoxCPMBackend& backend,
                                                 const VoxCPMOutputPool& output_pool,
                                                 int frame_offset,
                                                 int frame_count,
                                                 int patch_size,
                                                 int feat_dim) {
    if (frame_count <= 0) {
        return {};
    }
    ggml_tensor* latent_seq = output_pool.latent_seq();
    if (latent_seq == nullptr) {
        return {};
    }

    const int total_patches = frame_count * patch_size;
    const int patch_offset = frame_offset * patch_size;
    if (patch_offset < 0 || patch_offset + total_patches > output_pool.shape().max_latent_patches * patch_size) {
        return {};
    }
    if (output_pool.shape().feat_dim != feat_dim || output_pool.shape().patch_size != patch_size) {
        fail("Output pool shape does not match AudioVAE decode request");
    }

    static const bool log_chunk_timing = [] {
        const char* raw = std::getenv("VOXCPM_LOG_CHUNK_DECODE_TIMING");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    const auto t0 = std::chrono::steady_clock::now();

    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = output_pool.make_audio_vae_latent_view(graph_ctx.raw_context(), frame_offset, frame_count);
    if (latent == nullptr) {
        return {};
    }
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    const auto t_build = std::chrono::steady_clock::now();
    backend.reserve_compute_memory(graph, "server.audio_vae.decode.output_pool");
    backend.alloc_graph(graph, "server.audio_vae.decode.output_pool");
    const auto t_alloc = std::chrono::steady_clock::now();
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode from output pool failed");
    }
    const auto t_compute = std::chrono::steady_clock::now();

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    if (log_chunk_timing) {
        const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr,
                "[chunk_decode] frames=%d build_ms=%.2f alloc_ms=%.2f compute_ms=%.2f readback_ms=%.2f\n",
                frame_count,
                ms(t0, t_build),
                ms(t_build, t_alloc),
                ms(t_alloc, t_compute),
                ms(t_compute, std::chrono::steady_clock::now()));
    }
    return waveform;
}

int stateful_audio_decode_patch_threshold(const AudioVAE& audio_vae) {
    constexpr int kDefaultCudaAudioDecodePatchThreshold = 2048;
    constexpr int kConditionedCudaAudioDecodePatchThreshold = 1024;
    const bool has_sr_conditioning = std::any_of(audio_vae.weights().decoder_blocks.begin(),
                                                 audio_vae.weights().decoder_blocks.end(),
                                                 [](const DecoderBlockWeights& block) {
                                                     return block.sr_cond.active();
                                                 });
    if (has_sr_conditioning || audio_vae.config().num_decoder_blocks() >= 6) {
        return kConditionedCudaAudioDecodePatchThreshold;
    }
    return kDefaultCudaAudioDecodePatchThreshold;
}

int stateful_audio_decode_max_window_patches(const AudioVAE& audio_vae) {
    constexpr int kDefaultMaxWindowPatches = 1024;
    constexpr int kConditionedMaxWindowPatches = 1024;
    const bool has_sr_conditioning = std::any_of(audio_vae.weights().decoder_blocks.begin(),
                                                 audio_vae.weights().decoder_blocks.end(),
                                                 [](const DecoderBlockWeights& block) {
                                                     return block.sr_cond.active();
                                                 });
    if (has_sr_conditioning || audio_vae.config().num_decoder_blocks() >= 6) {
        return kConditionedMaxWindowPatches;
    }
    return kDefaultMaxWindowPatches;
}

int stateful_audio_decode_chunk_frames(const AudioVAE& audio_vae, int patch_size) {
    const int max_window_frames =
        std::max(1, stateful_audio_decode_max_window_patches(audio_vae) / std::max(1, patch_size));
    const int requested_chunk_frames = env_int_or_default("VOXCPM_AUDIO_DECODE_CHUNK_FRAMES", 0);
    if (requested_chunk_frames > 0) {
        return std::min(requested_chunk_frames, max_window_frames);
    }
    return max_window_frames;
}

bool should_use_stateful_audio_decode(const VoxCPMBackend& backend,
                                      const AudioVAE& audio_vae,
                                      int total_patches) {
    return backend.is_gpu() &&
           total_patches >= stateful_audio_decode_patch_threshold(audio_vae) &&
           audio_vae.supports_streaming_decode(backend);
}

void append_stateful_decoded_chunk(std::vector<float>& waveform,
                                   std::vector<float>& chunk_waveform,
                                   int chunk_start,
                                   int new_frames,
                                   int skip_frames,
                                   int patch_len) {
    const int discard_frames = std::max(0, std::min(new_frames, skip_frames - chunk_start));
    const size_t discard = static_cast<size_t>(discard_frames) * static_cast<size_t>(patch_len);
    if (chunk_waveform.size() > discard) {
        chunk_waveform.erase(chunk_waveform.begin(),
                             chunk_waveform.begin() + static_cast<std::ptrdiff_t>(discard));
    } else {
        chunk_waveform.clear();
    }

    const int keep_frames = std::max(0, new_frames - discard_frames);
    const size_t keep = static_cast<size_t>(keep_frames) * static_cast<size_t>(patch_len);
    if (chunk_waveform.size() > keep) {
        chunk_waveform.resize(keep);
    }
    waveform.insert(waveform.end(), chunk_waveform.begin(), chunk_waveform.end());
}

std::vector<float> decode_audio_stateful_from_patch_major_frames(AudioVAE& audio_vae,
                                                                 VoxCPMBackend& backend,
                                                                 const std::vector<float>& frames,
                                                                 int skip_frames,
                                                                 int chunk_frames,
                                                                 int patch_size,
                                                                 int feat_dim,
                                                                 int patch_len) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    if (frames.empty() || (frames.size() % frame_stride) != 0 || skip_frames < 0 || skip_frames > total_frames) {
        return {};
    }

    AudioVAEStreamingDecodeState stream_state;
    if (!audio_vae.initialize_streaming_decode_state(backend, stream_state)) {
        return {};
    }

    std::vector<float> waveform;
    waveform.reserve(static_cast<size_t>(std::max(0, total_frames - skip_frames)) * static_cast<size_t>(patch_len));
    for (int chunk_start = 0; chunk_start < total_frames; chunk_start += chunk_frames) {
        const int new_frames = std::min(chunk_frames, total_frames - chunk_start);
        const int total_patches = new_frames * patch_size;
        const size_t begin = static_cast<size_t>(chunk_start) * frame_stride;
        const size_t end = static_cast<size_t>(chunk_start + new_frames) * frame_stride;
        std::vector<float> chunk_frames_host(frames.begin() + static_cast<std::ptrdiff_t>(begin),
                                             frames.begin() + static_cast<std::ptrdiff_t>(end));

        VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* patch_major = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_dim, total_patches);
        ggml_set_input(patch_major);
        ggml_tensor* latent = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), patch_major));
        ggml_tensor* audio = audio_vae.decode_streaming(graph_ctx, backend, latent, stream_state);
        if (!audio) {
            fail("Failed to build stateful AudioVAE decode graph from patch-major frames");
        }

        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend.reserve_compute_memory(graph, "server.audio_vae.decode.stateful.patch_major");
        backend.alloc_graph(graph, "server.audio_vae.decode.stateful.patch_major");
        backend.tensor_set(patch_major, chunk_frames_host.data(), 0, chunk_frames_host.size() * sizeof(float));
        audio_vae.prepare_decode_inputs(backend);
        if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
            fail("Stateful AudioVAE decode from patch-major frames failed");
        }
        stream_state.publish_updates(backend);

        std::vector<float> chunk_waveform(static_cast<size_t>(ggml_nelements(audio)));
        backend.tensor_get(audio, chunk_waveform.data(), 0, chunk_waveform.size() * sizeof(float));
        append_stateful_decoded_chunk(waveform, chunk_waveform, chunk_start, new_frames, skip_frames, patch_len);
    }
    return waveform;
}

std::vector<float> decode_audio_stateful_from_output_pool(AudioVAE& audio_vae,
                                                          VoxCPMBackend& backend,
                                                          const VoxCPMOutputPool& output_pool,
                                                          int frame_offset,
                                                          int frame_count,
                                                          int skip_frames,
                                                          int chunk_frames,
                                                          int patch_size,
                                                          int feat_dim,
                                                          int patch_len) {
    if (frame_count <= 0 || skip_frames < 0 || skip_frames > frame_count) {
        return {};
    }
    if (output_pool.shape().feat_dim != feat_dim || output_pool.shape().patch_size != patch_size) {
        fail("Output pool shape does not match stateful AudioVAE decode request");
    }

    AudioVAEStreamingDecodeState stream_state;
    if (!audio_vae.initialize_streaming_decode_state(backend, stream_state)) {
        return {};
    }

    std::vector<float> waveform;
    waveform.reserve(static_cast<size_t>(std::max(0, frame_count - skip_frames)) * static_cast<size_t>(patch_len));
    for (int chunk_start = 0; chunk_start < frame_count; chunk_start += chunk_frames) {
        const int new_frames = std::min(chunk_frames, frame_count - chunk_start);

        VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* latent =
            output_pool.make_audio_vae_latent_view(graph_ctx.raw_context(), frame_offset + chunk_start, new_frames);
        if (latent == nullptr) {
            return {};
        }
        ggml_tensor* audio = audio_vae.decode_streaming(graph_ctx, backend, latent, stream_state);
        if (!audio) {
            fail("Failed to build stateful AudioVAE decode graph from output pool");
        }

        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend.reserve_compute_memory(graph, "server.audio_vae.decode.stateful.output_pool");
        backend.alloc_graph(graph, "server.audio_vae.decode.stateful.output_pool");
        audio_vae.prepare_decode_inputs(backend);
        if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
            fail("Stateful AudioVAE decode from output pool failed");
        }
        stream_state.publish_updates(backend);

        std::vector<float> chunk_waveform(static_cast<size_t>(ggml_nelements(audio)));
        backend.tensor_get(audio, chunk_waveform.data(), 0, chunk_waveform.size() * sizeof(float));
        append_stateful_decoded_chunk(waveform, chunk_waveform, chunk_start, new_frames, skip_frames, patch_len);
    }
    return waveform;
}

void fill_noise(std::vector<float>& noise, int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    noise.resize(static_cast<size_t>(patch_size) * feat_dim);
    for (float& value : noise) {
        value = dist(rng);
    }
}

int decode_step_cap_for_service(BackendType backend_type, int seq_len) {
    constexpr int kShortSeqThreshold = 256;
    constexpr int kLongSeqThreshold = 512;

    if (backend_type == BackendType::CPU) {
        return seq_len > kLongSeqThreshold ? 32 : (seq_len > kShortSeqThreshold ? 48 : 64);
    }
    return seq_len > kLongSeqThreshold ? 64 : (seq_len > kShortSeqThreshold ? 96 : 128);
}

int decode_step_budget_for_request(const SynthesisRequest& request, BackendType backend_type, int seq_len) {
    constexpr int kMaxDecodeSteps = 2000;
    if (request.max_decode_steps < 0) {
        fail("max_decode_steps must be >= 0");
    }
    const int service_cap = request.max_decode_steps > 0
                                ? request.max_decode_steps
                                : decode_step_cap_for_service(backend_type, seq_len);
    return std::min(service_cap, kMaxDecodeSteps);
}

bool should_use_output_pool_timeline(const VoxCPMDecodeState& state, bool has_reference_audio, int seq_len) {
    constexpr int kOutputPoolSeqLimit = 256;
    return state.output_pool != nullptr && state.output_pool->is_initialized() && !has_reference_audio &&
           seq_len <= kOutputPoolSeqLimit;
}

std::vector<float> build_decode_feature_sequence(const std::vector<float>& prompt_feat,
                                                 int prompt_audio_length,
                                                 const std::vector<float>& generated_steps,
                                                 int streaming_prefix_len,
                                                 int patch_size,
                                                 int feat_dim,
                                                 int* prepended_context_frames) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
    }

    std::vector<float> decode_frames;
    decode_frames.reserve(static_cast<size_t>(context_frames) * frame_stride + generated_steps.size());
    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        decode_frames.insert(decode_frames.end(),
                             prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                             prompt_feat.end());
    }
    decode_frames.insert(decode_frames.end(), generated_steps.begin(), generated_steps.end());

    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    return decode_frames;
}

std::vector<float> build_decode_latent_sequence(const std::vector<float>& prompt_feat,
                                                int prompt_audio_length,
                                                const std::vector<float>& generated_steps,
                                                int streaming_prefix_len,
                                                int patch_size,
                                                int feat_dim,
                                                int* prepended_context_frames) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
    }

    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    const int generated_frames = static_cast<int>(generated_steps.size() / frame_stride);
    const int total_frames = context_frames + generated_frames;
    const int total_patches = total_frames * patch_size;
    std::vector<float> latent(static_cast<size_t>(total_patches) * feat_dim, 0.0f);

    auto write_patch_major_frames = [&](const float* frames, int frame_count, int frame_base) {
        if (frames == nullptr || frame_count <= 0) {
            return;
        }
        for (int frame = 0; frame < frame_count; ++frame) {
            for (int patch = 0; patch < patch_size; ++patch) {
                const int time_index = (frame_base + frame) * patch_size + patch;
                for (int d = 0; d < feat_dim; ++d) {
                    const size_t src = (static_cast<size_t>(frame) * patch_size + patch) * feat_dim + d;
                    const size_t dst = static_cast<size_t>(d) * total_patches + time_index;
                    latent[dst] = frames[src];
                }
            }
        }
    };

    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        write_patch_major_frames(prompt_feat.data() + static_cast<std::ptrdiff_t>(context_offset), context_frames, 0);
    }
    write_patch_major_frames(generated_steps.data(), generated_frames, context_frames);
    return latent;
}

void append_stream_frame(std::vector<float>& recent_frames,
                         const std::vector<float>& patch,
                         int max_frames,
                         int patch_size,
                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    recent_frames.insert(recent_frames.end(), patch.begin(), patch.end());
    const size_t max_elems = static_cast<size_t>(max_frames) * frame_stride;
    if (recent_frames.size() > max_elems) {
        recent_frames.erase(recent_frames.begin(),
                            recent_frames.begin() + static_cast<std::ptrdiff_t>(recent_frames.size() - max_elems));
    }
}

}  // namespace

VoiceStore::VoiceStore(std::string root_dir)
    : root_dir_(std::move(root_dir)) {
    std::filesystem::create_directories(root_dir_);
}

bool VoiceStore::has_voice(const std::string& id) const {
    return std::filesystem::exists(manifest_path_for(root_dir_, id)) &&
           std::filesystem::exists(prompt_path_for(root_dir_, id));
}

void VoiceStore::save_voice(const PromptFeatures& features) {
    if (!is_valid_voice_id(features.id)) {
        fail("Invalid voice id");
    }
    const auto dir = std::filesystem::path(root_dir_) / features.id;
    std::filesystem::create_directories(dir);

    json manifest = {
        {"id", features.id},
        {"prompt_text", features.prompt_text},
        {"prompt_audio_length", features.prompt_audio_length},
        {"reference_audio_length", features.reference_audio_length},
        {"sample_rate", features.sample_rate},
        {"patch_size", features.patch_size},
        {"feat_dim", features.feat_dim},
        {"created_at", features.created_at},
        {"updated_at", features.updated_at},
    };

    std::ofstream out(manifest_path_for(root_dir_, features.id));
    if (!out.is_open()) {
        fail("Failed to write voice manifest");
    }
    out << manifest.dump(2);
    write_binary_file(prompt_path_for(root_dir_, features.id), features.prompt_feat);
    if (!features.reference_feat.empty()) {
        write_binary_file(reference_path_for(root_dir_, features.id), features.reference_feat);
    } else {
        std::error_code ec;
        std::filesystem::remove(reference_path_for(root_dir_, features.id), ec);
    }
}

PromptFeatures VoiceStore::load_voice(const std::string& id) const {
    const auto manifest_path = manifest_path_for(root_dir_, id);
    const auto prompt_path = prompt_path_for(root_dir_, id);
    if (!std::filesystem::exists(manifest_path) || !std::filesystem::exists(prompt_path)) {
        fail("Voice not found: " + id);
    }

    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        fail("Failed to read voice manifest");
    }
    json manifest = json::parse(in);
    PromptFeatures features;
    features.id = manifest.at("id").get<std::string>();
    features.prompt_text = manifest.at("prompt_text").get<std::string>();
    features.prompt_audio_length = manifest.at("prompt_audio_length").get<int>();
    features.reference_audio_length = manifest.value("reference_audio_length", 0);
    features.sample_rate = manifest.at("sample_rate").get<int>();
    features.patch_size = manifest.at("patch_size").get<int>();
    features.feat_dim = manifest.at("feat_dim").get<int>();
    features.created_at = manifest.value("created_at", "");
    features.updated_at = manifest.value("updated_at", "");
    features.prompt_feat = read_binary_file(prompt_path);
    if (features.reference_audio_length > 0) {
        const auto ref_path = reference_path_for(root_dir_, id);
        if (!std::filesystem::exists(ref_path)) {
            fail("Reference feature blob missing for voice: " + id);
        }
        features.reference_feat = read_binary_file(ref_path);
    }
    return features;
}

VoiceMetadata VoiceStore::load_metadata(const std::string& id) const {
    const PromptFeatures features = load_voice(id);
    return VoiceMetadata{
        features.id,
        features.prompt_text,
        features.prompt_audio_length,
        features.reference_audio_length,
        features.sample_rate,
        features.patch_size,
        features.feat_dim,
        features.created_at,
        features.updated_at,
    };
}

std::vector<VoiceMetadata> VoiceStore::list_voices() const {
    std::vector<VoiceMetadata> voices;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(root_dir_, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string id = entry.path().filename().string();
        if (!is_valid_voice_id(id)) {
            continue;
        }
        try {
            voices.push_back(load_metadata(id));
        } catch (const std::exception&) {
            // Skip entries with missing or unreadable manifests.
        }
    }
    std::sort(voices.begin(), voices.end(), [](const VoiceMetadata& a, const VoiceMetadata& b) {
        return a.id < b.id;
    });
    return voices;
}

void VoiceStore::delete_voice(const std::string& id) {
    const auto dir = std::filesystem::path(root_dir_) / id;
    if (!std::filesystem::exists(dir)) {
        fail("Voice not found: " + id);
    }
    std::filesystem::remove_all(dir);
}

VoxCPMServiceCore::VoxCPMServiceCore(std::string model_path, BackendType backend_type, int threads)
    : model_path_(std::move(model_path)),
      backend_type_(backend_type),
      threads_(threads) {}

void VoxCPMServiceCore::load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_) {
        return;
    }

    if (!std::filesystem::exists(model_path_) || !std::filesystem::is_regular_file(model_path_)) {
        fail("Model path must point to an existing GGUF file");
    }

    backend_ = std::make_unique<VoxCPMBackend>(backend_type_, threads_);

    // AudioVAE runs on the main backend by default. VOXCPM_VAE_ON_CPU=1
    // routes its graphs to a dedicated CPU backend instead — the escape
    // hatch for the 2026-07-28 AGX incident class (pathological Metal
    // kernels); the rewritten conv_transpose_1d has since passed on-device
    // validation. The VAE weights stay in the main backend's shared buffer,
    // which is host-addressable on Apple Silicon, so nothing is loaded twice.
    const char* vae_on_cpu = std::getenv("VOXCPM_VAE_ON_CPU");
    const bool vae_on_cpu_enabled = vae_on_cpu && *vae_on_cpu && std::strcmp(vae_on_cpu, "0") != 0;
    if (backend_->type() == BackendType::Metal && vae_on_cpu_enabled) {
        vae_backend_ = std::make_unique<VoxCPMBackend>(BackendType::CPU, threads_);
        std::cerr << "AudioVAE backend: cpu (VOXCPM_VAE_ON_CPU=1)\n";
    }

    // Streaming chunk decodes run on a worker thread with their own backend
    // instance (own command queue), overlapping the ~60 ms windowed AudioVAE
    // decode with the next LM step instead of serializing it into the decode
    // loop. VOXCPM_ASYNC_CHUNK_DECODE=0 disables the worker.
    const char* async_chunk = std::getenv("VOXCPM_ASYNC_CHUNK_DECODE");
    const bool async_chunk_enabled = !(async_chunk && *async_chunk && std::strcmp(async_chunk, "0") == 0);
    if (backend_->is_gpu() && !vae_backend_ && async_chunk_enabled) {
        // CPU, not a second GPU queue: on a single GPU another queue does not
        // reduce total GPU work, while the CPU sits idle during the decode
        // loop and chunk jobs arrive only every few steps. Unified memory
        // means the CPU worker reads the pool and the f16 VAE weights from
        // the Metal shared buffers with no copies.
        stream_chunk_backend_ = std::make_unique<VoxCPMBackend>(BackendType::CPU, threads_);
        std::cerr << "Streaming chunk decode: async on cpu\n";
    }

    store_ = std::make_shared<VoxCPMWeightStore>();
    if (!store_->load_from_file(model_path_, *backend_)) {
        fail("Failed to load GGUF: " + model_path_);
    }
    if (!runtime_.load_from_store(store_, *backend_)) {
        fail("Failed to initialize VoxCPM runtime from GGUF");
    }
    if (!audio_vae_.load_from_store(store_)) {
        fail("Failed to initialize AudioVAE from GGUF");
    }

    tokenizer_ = std::make_unique<VoxCPMTokenizer>();
    if (!tokenizer_->load_from_store(*store_)) {
        fail("Failed to load tokenizer metadata from GGUF");
    }
    split_tokenizer_ = std::make_unique<ChineseCharSplitTokenizer>(*tokenizer_);
    loaded_ = true;
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio(const std::string& id,
                                                      const std::string& prompt_text,
                                                      const std::vector<float>& mono_audio,
                                                      int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return encode_prompt_audio_locked(id, prompt_text, mono_audio, sample_rate);
}

PromptFeatures VoxCPMServiceCore::encode_reference_audio(const std::string& id,
                                                         const std::vector<float>& mono_audio,
                                                         int sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return encode_reference_audio_locked(id, mono_audio, sample_rate);
}

PromptFeatures VoxCPMServiceCore::encode_prompt_audio_locked(const std::string& id,
                                                             const std::string& prompt_text,
                                                             const std::vector<float>& mono_audio,
                                                             int sample_rate) {
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int encode_patch_len = patch_size_value * audio_vae_.config().hop_length();
    std::vector<float> resampled = resample_audio_to_rate(mono_audio, sample_rate, audio_vae_.config().sample_rate);
    resampled = trim_audio_silence_vad(resampled, audio_vae_.config().sample_rate);
    pad_audio_for_patch_alignment(resampled, static_cast<size_t>(encode_patch_len), PaddingMode::Left);

    PromptFeatures features;
    features.id = id;
    features.prompt_text = prompt_text;
    features.prompt_feat = extract_prompt_features(audio_vae_,
                                                   audio_vae_backend(),
                                                   resampled,
                                                   audio_vae_.config().sample_rate,
                                                   patch_size_value,
                                                   feat_dim_value);
    features.prompt_audio_length =
        static_cast<int>(features.prompt_feat.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    features.sample_rate = audio_vae_.config().sample_rate;
    features.patch_size = patch_size_value;
    features.feat_dim = feat_dim_value;
    const std::string now = make_timestamp_utc();
    features.created_at = now;
    features.updated_at = now;
    std::cerr << "[voice] encoded id=" << id
              << " prompt_audio_length=" << features.prompt_audio_length
              << " patch_size=" << features.patch_size
              << " feat_dim=" << features.feat_dim
              << " sample_rate=" << features.sample_rate
              << "\n";
    return features;
}

PromptFeatures VoxCPMServiceCore::encode_reference_audio_locked(const std::string& id,
                                                                const std::vector<float>& mono_audio,
                                                                int sample_rate) {
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int encode_patch_len = patch_size_value * audio_vae_.config().hop_length();
    std::vector<float> resampled = resample_audio_to_rate(mono_audio, sample_rate, audio_vae_.config().sample_rate);
    resampled = trim_audio_silence_vad(resampled, audio_vae_.config().sample_rate);
    pad_audio_for_patch_alignment(resampled, static_cast<size_t>(encode_patch_len), PaddingMode::Right);

    PromptFeatures features;
    features.id = id;
    features.reference_feat = extract_prompt_features(audio_vae_,
                                                      audio_vae_backend(),
                                                      resampled,
                                                      audio_vae_.config().sample_rate,
                                                      patch_size_value,
                                                      feat_dim_value);
    features.reference_audio_length =
        static_cast<int>(features.reference_feat.size() / static_cast<size_t>(patch_size_value * feat_dim_value));
    features.sample_rate = audio_vae_.config().sample_rate;
    features.patch_size = patch_size_value;
    features.feat_dim = feat_dim_value;
    const std::string now = make_timestamp_utc();
    features.created_at = now;
    features.updated_at = now;
    std::cerr << "[voice] encoded reference id=" << id
              << " reference_audio_length=" << features.reference_audio_length
              << " patch_size=" << features.patch_size
              << " feat_dim=" << features.feat_dim
              << " sample_rate=" << features.sample_rate
              << "\n";
    return features;
}

SynthesisResult VoxCPMServiceCore::synthesize(const SynthesisRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        fail("Model core is not loaded");
    }
    return synthesize_locked(request);
}

SynthesisResult VoxCPMServiceCore::synthesize_locked(const SynthesisRequest& request) {
    if (request.text.empty()) {
        fail("Text input must not be empty");
    }
    const int patch_size_value = runtime_.config().patch_size;
    const int feat_dim_value = runtime_.config().feat_dim;
    const int decode_patch_len = patch_size_value * audio_vae_.config().decode_hop_length();
    const size_t expected_prompt_feat_size =
        static_cast<size_t>(request.prompt.prompt_audio_length) *
        static_cast<size_t>(patch_size_value) *
        static_cast<size_t>(feat_dim_value);
    const size_t expected_reference_feat_size =
        static_cast<size_t>(request.prompt.reference_audio_length) *
        static_cast<size_t>(patch_size_value) *
        static_cast<size_t>(feat_dim_value);

    if (request.prompt.prompt_audio_length < 0) {
        fail("Voice metadata is invalid: prompt_audio_length must be >= 0");
    }
    if (request.prompt.reference_audio_length < 0) {
        fail("Voice metadata is invalid: reference_audio_length must be >= 0");
    }
    if (request.prompt.patch_size != patch_size_value) {
        fail("Voice metadata patch_size does not match the loaded model");
    }
    if (request.prompt.feat_dim != feat_dim_value) {
        fail("Voice metadata feat_dim does not match the loaded model");
    }
    if (request.prompt.prompt_feat.size() != expected_prompt_feat_size) {
        fail("Voice metadata is inconsistent with stored prompt features");
    }
    if (request.prompt.reference_feat.size() != expected_reference_feat_size) {
        fail("Voice metadata is inconsistent with stored reference features");
    }
    if ((request.prompt.prompt_audio_length == 0) != request.prompt.prompt_text.empty()) {
        fail("Voice metadata is invalid: prompt_text must be provided iff prompt audio is present");
    }
    if (request.retry_badcase_max_times < 1) {
        fail("retry_badcase_max_times must be >= 1");
    }
    if (request.retry_badcase_ratio_threshold <= 0.0f) {
        fail("retry_badcase_ratio_threshold must be > 0");
    }

    // Request boundary reset: cached runtime graphs and backend graph bookkeeping
    // are rebuilt per synthesis request so reused service instances do not carry
    // stale graph state across calls.
    runtime_.reset_request_state();
    backend_->reset_request_state();

    const bool has_prompt_audio = request.prompt.prompt_audio_length > 0;
    const bool has_reference_audio = request.prompt.reference_audio_length > 0;
    std::string effective_text = request.text;
    if (has_prompt_audio) {
        const auto [stripped_text, stripped] = strip_hifi_control_prefix(request.text);
        if (stripped) {
            std::cerr << "[tts] Hi-Fi mode ignores control instructions; stripping the leading parenthesized prefix.\n";
            effective_text = stripped_text;
        }
    }
    bool retry_badcase = request.retry_badcase;
    if (retry_badcase && request.chunk_callback) {
        std::cerr << "[tts] retry_badcase is not supported with streaming chunks; disabling retries.\n";
        retry_badcase = false;
    }
    const PreparedConditioning prepared =
        build_conditioning(*split_tokenizer_, effective_text, request.prompt, patch_size_value, feat_dim_value);
    const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
    std::cerr << "[tts] synth start seq_len=" << prepared.full_text_tokens.size()
              << " prompt_audio_length=" << request.prompt.prompt_audio_length
              << " reference_audio_length=" << request.prompt.reference_audio_length
              << " prompt_feat_size=" << request.prompt.prompt_feat.size()
              << "\n";
    const int prompt_audio_length = request.prompt.prompt_audio_length;
    const int target_text_token_count =
        std::max<int>(1, static_cast<int>(split_tokenizer_->tokenize(effective_text).size()));
    const int natural_max_len =
        std::min(static_cast<int>(target_text_token_count * request.retry_badcase_ratio_threshold + 10.0f), 2000);
    const int decode_step_budget = decode_step_budget_for_request(request, backend_type_, seq_len);
    const int max_len = std::min(natural_max_len, decode_step_budget);
    std::cerr << "[tts] decode budget natural_max_len=" << natural_max_len
              << " service_cap=" << decode_step_budget
              << " max_len=" << max_len
              << (request.max_decode_steps > 0 ? " override=1" : " override=0")
              << "\n";
    constexpr int kMinLen = 2;
    const int max_attempts = retry_badcase ? std::max(1, request.retry_badcase_max_times) : 1;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        VoxCPMDecodeState state = runtime_.prefill(prepared.full_text_tokens,
                                                   prepared.text_mask,
                                                   prepared.feat,
                                                   prepared.feat_mask,
                                                   seq_len,
                                                   request.streaming_prefix_len);
        std::cerr << "[tts] prefill done seq_len=" << seq_len << " attempt=" << (attempt + 1) << "\n";
        const bool use_output_pool_timeline = should_use_output_pool_timeline(state, has_reference_audio, seq_len);

        std::mt19937 rng(request.seed >= 0 ? static_cast<uint32_t>(request.seed) : std::random_device{}());
        std::vector<float> generated_steps;
        if (!use_output_pool_timeline) {
            generated_steps.reserve(static_cast<size_t>(max_len) * patch_size_value * feat_dim_value);
        }
        std::vector<float> noise;
        std::vector<float> stream_recent_frames;
        const size_t frame_stride = static_cast<size_t>(patch_size_value) * feat_dim_value;
        const int context_frames =
            (has_prompt_audio && request.streaming_prefix_len > 1)
                ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                : 0;
        const bool use_fallback_streaming_window = request.chunk_callback && !use_output_pool_timeline;
        if (use_fallback_streaming_window) {
            stream_recent_frames.reserve(static_cast<size_t>(request.streaming_prefix_len) * frame_stride);
        }
        if (use_fallback_streaming_window && context_frames > 0) {
            const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
            stream_recent_frames.insert(stream_recent_frames.end(),
                                        request.prompt.prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                                        request.prompt.prompt_feat.end());
        }

        int pending_stream_frames = 0;
        int emitted_stream_chunks = 0;
        const int stream_cadence = std::max(1, env_int_or_default("VOXCPM_STREAM_CADENCE", 4));
        const int stream_first_emit_frames = std::max(1, env_int_or_default("VOXCPM_STREAM_FIRST_EMIT_FRAMES", 1));

        // Async chunk pipeline: the decode loop only enqueues (frame_offset,
        // recent, pending) descriptors; a single worker thread decodes each
        // window on its own backend queue and invokes chunk_callback in FIFO
        // order. Enqueueing happens after the synchronous LM graph compute, so
        // every frame in the window is fully written before the worker reads
        // it. Falls back to inline decoding without the dedicated backend.
        struct StreamChunkJob {
            int frame_offset;
            int recent;
            int pending;
        };
        std::deque<StreamChunkJob> chunk_jobs;
        std::mutex chunk_mutex;
        std::condition_variable chunk_cv;
        bool chunk_jobs_done = false;
        std::exception_ptr chunk_worker_error;
        std::thread chunk_worker;
        const bool async_chunks = stream_chunk_backend_ != nullptr &&
                                  request.chunk_callback != nullptr &&
                                  state.output_pool != nullptr;
        // The pool object outlives the loop, but ownership of it is moved out
        // of `state` for the duration of every runtime_.decode call — the
        // worker must hold the pointee directly, never read state.output_pool.
        VoxCPMOutputPool* const stream_pool = async_chunks ? state.output_pool.get() : nullptr;
        if (async_chunks) {
            chunk_worker = std::thread([&]() {
                for (;;) {
                    StreamChunkJob job;
                    {
                        std::unique_lock<std::mutex> lock(chunk_mutex);
                        chunk_cv.wait(lock, [&]() { return chunk_jobs_done || !chunk_jobs.empty(); });
                        if (chunk_jobs.empty()) {
                            return;
                        }
                        job = chunk_jobs.front();
                        chunk_jobs.pop_front();
                    }
                    try {
                        std::vector<float> chunk_waveform = decode_audio_from_output_pool(audio_vae_,
                                                                                          *stream_chunk_backend_,
                                                                                          *stream_pool,
                                                                                          job.frame_offset,
                                                                                          job.recent,
                                                                                          patch_size_value,
                                                                                          feat_dim_value);
                        const size_t keep =
                            static_cast<size_t>(decode_patch_len) * static_cast<size_t>(job.pending);
                        if (chunk_waveform.size() > keep) {
                            chunk_waveform.erase(chunk_waveform.begin(),
                                                 chunk_waveform.end() - static_cast<std::ptrdiff_t>(keep));
                        }
                        request.chunk_callback(chunk_waveform);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(chunk_mutex);
                        chunk_worker_error = std::current_exception();
                        return;
                    }
                }
            });
        }
        struct ChunkWorkerGuard {
            std::thread& worker;
            std::mutex& mutex;
            std::condition_variable& cv;
            bool& done;
            ~ChunkWorkerGuard() {
                if (!worker.joinable()) {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    done = true;
                }
                cv.notify_one();
                worker.join();
            }
        } chunk_worker_guard{chunk_worker, chunk_mutex, chunk_cv, chunk_jobs_done};

        const auto finish_chunk_worker = [&]() {
            if (!chunk_worker.joinable()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(chunk_mutex);
                chunk_jobs_done = true;
            }
            chunk_cv.notify_one();
            chunk_worker.join();
            if (chunk_worker_error) {
                std::rethrow_exception(chunk_worker_error);
            }
        };

        const auto emit_pool_stream_chunk = [&]() {
            if (pending_stream_frames <= 0) {
                return;
            }
            const int recent = std::min(request.streaming_prefix_len - 1 + pending_stream_frames,
                                        state.audio_frame_count);
            const int frame_offset = state.audio_frame_count - recent;
            if (recent <= 0) {
                return;
            }
            // The very first chunk decodes inline on the main (GPU) backend:
            // ~60 ms there versus ~150+ ms on the CPU worker directly delays
            // first audio. Later chunks go to the worker, whose latency hides
            // behind playback. The inline decode finishes before any job is
            // enqueued, so callback ordering stays FIFO.
            if (async_chunks && emitted_stream_chunks > 0) {
                {
                    std::lock_guard<std::mutex> lock(chunk_mutex);
                    if (chunk_worker_error) {
                        std::rethrow_exception(chunk_worker_error);
                    }
                    chunk_jobs.push_back(StreamChunkJob{frame_offset, recent, pending_stream_frames});
                }
                chunk_cv.notify_one();
                pending_stream_frames = 0;
                ++emitted_stream_chunks;
                return;
            }
            std::vector<float> chunk_waveform = decode_audio_from_output_pool(audio_vae_,
                                                                              audio_vae_backend(),
                                                                              *state.output_pool,
                                                                              frame_offset,
                                                                              recent,
                                                                              patch_size_value,
                                                                              feat_dim_value);
            const size_t keep = static_cast<size_t>(decode_patch_len) * static_cast<size_t>(pending_stream_frames);
            if (chunk_waveform.size() > keep) {
                chunk_waveform.erase(chunk_waveform.begin(),
                                     chunk_waveform.end() - static_cast<std::ptrdiff_t>(keep));
            }
            request.chunk_callback(chunk_waveform);
            pending_stream_frames = 0;
            ++emitted_stream_chunks;
        };

        for (int step = 0; step < max_len; ++step) {
            fill_noise(noise, patch_size_value, feat_dim_value, rng);
            VoxCPMDecodeOptions decode_options;
            decode_options.export_patch_to_host = !use_output_pool_timeline;
            decode_options.publish_stop_logits_to_output = !use_output_pool_timeline;
            decode_options.publish_patch_to_output = !use_output_pool_timeline;
            decode_options.trust_persistent_state = use_output_pool_timeline;
            VoxCPMDecodeResult result = runtime_.decode(std::move(state),
                                                        noise,
                                                        request.inference_timesteps,
                                                        request.cfg_value,
                                                        decode_options);
            if (!use_output_pool_timeline) {
                generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
            }
            state = std::move(result.output_1);

            if (request.chunk_callback) {
                int recent_frame_count = 0;

                if (use_output_pool_timeline && state.audio_frame_count > 0) {
                    // The windowed chunk decode costs a near-constant ~60 ms per call on
                    // Metal (a ~650-node graph on tiny tensors is dispatch-bound, so the
                    // window size barely matters), so emission is batched. The first
                    // emission waits for VOXCPM_STREAM_FIRST_EMIT_FRAMES (default 1:
                    // fastest first audio); raising it trades speech onset for a large
                    // opening buffer that rides out contention at the start of a reply.
                    // Later chunks go out every VOXCPM_STREAM_CADENCE steps.
                    ++pending_stream_frames;
                    const int cadence = emitted_stream_chunks == 0 ? stream_first_emit_frames : stream_cadence;
                    if (pending_stream_frames >= cadence) {
                        emit_pool_stream_chunk();
                    }
                } else {
                    append_stream_frame(stream_recent_frames,
                                        result.output_0,
                                        request.streaming_prefix_len,
                                        patch_size_value,
                                        feat_dim_value);
                    recent_frame_count = static_cast<int>(stream_recent_frames.size() / frame_stride);
                    if (recent_frame_count > 0) {
                        std::vector<float> chunk_waveform = decode_audio_from_patch_major_frames(audio_vae_,
                                                                                                 audio_vae_backend(),
                                                                                                 stream_recent_frames,
                                                                                                 patch_size_value,
                                                                                                 feat_dim_value);
                        if (chunk_waveform.size() > static_cast<size_t>(decode_patch_len)) {
                            chunk_waveform.erase(chunk_waveform.begin(),
                                                 chunk_waveform.end() - static_cast<std::ptrdiff_t>(decode_patch_len));
                        }
                        // AudioVAE graphs allocate from their own compute arena
                        // (VoxCPMBackend::stage_allocator), so the chunk decode can no
                        // longer resize the arena the cached LM step graphs point into
                        // and the caches stay valid across chunks.
                        request.chunk_callback(chunk_waveform);
                    }
                }
            }

            if (step > kMinLen && result.output_2) {
                break;
            }
        }
        if (request.chunk_callback && use_output_pool_timeline) {
            emit_pool_stream_chunk();
        }
        finish_chunk_worker();
        const int generated_frames = use_output_pool_timeline
                                         ? std::max(0, state.audio_frame_count - prompt_audio_length)
                                         : static_cast<int>(generated_steps.size() / frame_stride);
        std::cerr << "[tts] decode loop done generated_frames="
                  << generated_frames
                  << " attempt=" << (attempt + 1)
                  << "\n";
        if (generated_frames >= max_len && decode_step_budget < natural_max_len) {
            std::cerr << "[tts] decode step budget reached before stop token; increase --max-decode-steps "
                         "if the output is truncated.\n";
        }
        if (retry_badcase && (attempt + 1 < max_attempts) && (
               (generated_frames >= static_cast<int>(target_text_token_count * request.retry_badcase_ratio_threshold))
            || (generated_frames < std::max(1, target_text_token_count))
        )) {
            std::cerr << "[tts] badcase detected: audio_text_ratio="
                      << (static_cast<float>(generated_frames) / static_cast<float>(target_text_token_count))
                      << ", retrying attempt " << (attempt + 2) << "/" << max_attempts << "\n";
            continue;
        }

        int prepended_context_frames = 0;
        const int total_frames = (has_prompt_audio && request.streaming_prefix_len > 1
                                      ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                                      : 0) +
                                 generated_frames;
        const int total_patches = total_frames * patch_size_value;
        if (generated_frames == 0 || total_patches == 0) {
            fail("Model generated no audio patches");
        }

        std::vector<float> waveform;
        std::vector<float> latent;
        if (request.skip_final_waveform && request.chunk_callback) {
            // Streaming consumers already received every frame through
            // chunk_callback; the whole-utterance decode would only recompute
            // audio nobody reads.
            SynthesisResult result;
            result.sample_rate = audio_vae_.config().output_sample_rate();
            result.generated_frames = generated_frames;
            return result;
        }
        const bool use_stateful_final_audio_decode =
            should_use_stateful_audio_decode(audio_vae_backend(), audio_vae_, total_patches);
        const int decode_stateful_chunk_frames =
            stateful_audio_decode_chunk_frames(audio_vae_, patch_size_value);
        if (use_output_pool_timeline &&
            state.audio_frame_count >= prompt_audio_length + generated_frames) {
            const int frame_offset =
                std::max(0, prompt_audio_length - std::min(request.streaming_prefix_len - 1, prompt_audio_length));
            prepended_context_frames = has_prompt_audio && request.streaming_prefix_len > 1
                                           ? std::min(request.streaming_prefix_len - 1, prompt_audio_length)
                                           : 0;
            if (use_stateful_final_audio_decode) {
                waveform = decode_audio_stateful_from_output_pool(audio_vae_,
                                                                  *backend_,
                                                                  *state.output_pool,
                                                                  frame_offset,
                                                                  total_frames,
                                                                  prepended_context_frames,
                                                                  decode_stateful_chunk_frames,
                                                                  patch_size_value,
                                                                  feat_dim_value,
                                                                  decode_patch_len);
                if (waveform.empty()) {
                    fail("Stateful AudioVAE decode from output pool failed");
                }
                prepended_context_frames = 0;
            }
            if (waveform.empty()) {
                waveform = decode_audio_from_output_pool(audio_vae_,
                                                         audio_vae_backend(),
                                                         *state.output_pool,
                                                         frame_offset,
                                                         total_frames,
                                                         patch_size_value,
                                                         feat_dim_value);
            }
        } else {
            if (use_stateful_final_audio_decode) {
                const std::vector<float> decode_frames = build_decode_feature_sequence(request.prompt.prompt_feat,
                                                                                       prompt_audio_length,
                                                                                       generated_steps,
                                                                                       request.streaming_prefix_len,
                                                                                       patch_size_value,
                                                                                       feat_dim_value,
                                                                                       &prepended_context_frames);
                waveform = decode_audio_stateful_from_patch_major_frames(audio_vae_,
                                                                         *backend_,
                                                                         decode_frames,
                                                                         prepended_context_frames,
                                                                         decode_stateful_chunk_frames,
                                                                         patch_size_value,
                                                                         feat_dim_value,
                                                                         decode_patch_len);
                if (waveform.empty()) {
                    fail("Stateful AudioVAE decode from patch-major frames failed");
                }
                prepended_context_frames = 0;
            }
            if (waveform.empty()) {
                latent = build_decode_latent_sequence(request.prompt.prompt_feat,
                                                      prompt_audio_length,
                                                      generated_steps,
                                                      request.streaming_prefix_len,
                                                      patch_size_value,
                                                      feat_dim_value,
                                                      &prepended_context_frames);
                waveform = decode_audio(audio_vae_, audio_vae_backend(), latent, total_patches, feat_dim_value);
            }
        }
        if (has_prompt_audio) {
            const size_t trim = static_cast<size_t>(decode_patch_len) * static_cast<size_t>(prepended_context_frames);
            if (waveform.size() > trim) {
                waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(trim));
            }
        }

        return SynthesisResult{
            std::move(waveform),
            audio_vae_.config().output_sample_rate(),
            generated_frames,
        };
    }

    fail("Retry loop exhausted without producing an accepted sample");
}

int VoxCPMServiceCore::sample_rate() const {
    return audio_vae_.config().output_sample_rate();
}

int VoxCPMServiceCore::patch_size() const {
    return runtime_.config().patch_size;
}

int VoxCPMServiceCore::feat_dim() const {
    return runtime_.config().feat_dim;
}

std::string make_timestamp_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now_time);
#else
    gmtime_r(&now_time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

bool is_valid_voice_id(const std::string& id) {
    if (id.empty()) {
        return false;
    }
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.';
    });
}

}  // namespace voxcpm
