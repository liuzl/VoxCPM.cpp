/**
 * @file localenc.cpp
 * @brief VoxCPM Local Encoder implementation
 */

#include "voxcpm/localenc.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace voxcpm {

namespace {
constexpr float kLocEncBatchMaskNeg = -1.0e9f;
}  // namespace

LocEncModel::~LocEncModel() {
    scratch_kv_cache_.reset();

    if (batch_buffer_) {
        ggml_backend_buffer_free(batch_buffer_);
        batch_buffer_ = nullptr;
    }
    if (batch_ctx_) {
        ggml_free(batch_ctx_);
        batch_ctx_ = nullptr;
    }
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
}

bool LocEncModel::init_scratch_cache(VoxCPMBackend& backend) {
    if (scratch_kv_cache_) {
        return true;
    }

    scratch_kv_cache_ = std::make_unique<MiniCPMKVCache>(
        config().n_layer,
        config().n_kv_heads,
        config().max_length,
        config().head_dim());
    scratch_kv_cache_->init(backend);
    return true;
}

bool LocEncModel::load_from_gguf(const std::string& gguf_path,
                                 VoxCPMContext& weight_ctx,
                                 VoxCPMContext& graph_ctx,
                                 VoxCPMBackend& backend) {
    VOXCPM_UNUSED(weight_ctx);
    VOXCPM_UNUSED(graph_ctx);

    auto store = std::make_shared<VoxCPMWeightStore>();
    if (!store->load_from_file(gguf_path, backend)) {
        return false;
    }
    return load_from_store(store, backend);
}

bool LocEncModel::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                                  VoxCPMBackend& backend) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;
    weights_.in_proj_weight = store->get_tensor("locenc.in_proj.weight");
    weights_.in_proj_bias = store->get_tensor("locenc.in_proj.bias");
    weights_.special_token = store->get_tensor("locenc.special_token");
    if (!weights_.in_proj_weight || !weights_.in_proj_bias || !weights_.special_token) {
        return false;
    }

    feat_dim_ = static_cast<int>(weights_.in_proj_weight->ne[0]);

    if (!encoder_.load_from_store(store, "locenc", backend)) {
        return false;
    }
    if (config().hidden_size != static_cast<int>(weights_.special_token->ne[0])) {
        return false;
    }

    backend_ = &backend;
    return init_scratch_cache(backend);
}

ggml_tensor* LocEncModel::forward_patch(VoxCPMContext& ctx, ggml_tensor* input) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(scratch_kv_cache_ != nullptr);
    VOXCPM_ASSERT(input->ne[1] > 0);

    ggml_context* raw = ctx.raw_context();
    const int64_t n_patches = input->ne[1];
    const int hidden_size = config().hidden_size;

    VOXCPM_ASSERT(n_patches + 1 <= config().max_length);
    VOXCPM_ASSERT(input->ne[0] == feat_dim_ || input->ne[0] == hidden_size);

    scratch_kv_cache_->clear();

    ggml_tensor* projected = input;
    if (input->ne[0] != hidden_size) {
        projected = ggml_mul_mat(raw, weights_.in_proj_weight, input);
        projected = ggml_add(raw, projected, weights_.in_proj_bias);
    }

    ggml_tensor* cls = ggml_reshape_2d(raw, weights_.special_token, hidden_size, 1);
    ggml_tensor* full_input = ggml_concat(raw, cls, projected, 1);
    ggml_tensor* output = encoder_.forward(ctx, full_input, nullptr, *scratch_kv_cache_, false, false);

    return ggml_view_1d(raw, output, hidden_size, 0);
}

bool LocEncModel::ensure_batch_constants(int64_t patch_size, int64_t seq_len) {
    VOXCPM_ASSERT(patch_size > 0);
    VOXCPM_ASSERT(seq_len > 0);
    VOXCPM_ASSERT(backend_ != nullptr);

    if (batch_patch_size_ == patch_size && batch_seq_len_ == seq_len &&
        batch_positions_ != nullptr && batch_attention_mask_ != nullptr) {
        return true;
    }

    if (batch_buffer_) {
        backend_->free_buffer(batch_buffer_);
        batch_buffer_ = nullptr;
    }
    if (batch_ctx_) {
        ggml_free(batch_ctx_);
        batch_ctx_ = nullptr;
    }
    batch_positions_ = nullptr;
    batch_attention_mask_ = nullptr;
    batch_patch_size_ = 0;
    batch_seq_len_ = 0;

    ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 2 + 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    batch_ctx_ = ggml_init(params);
    if (!batch_ctx_) {
        return false;
    }

    const int64_t tokens_per_patch = patch_size + 1;
    const int64_t total_tokens = tokens_per_patch * seq_len;
    batch_positions_ = ggml_new_tensor_1d(batch_ctx_, GGML_TYPE_I32, total_tokens);
    batch_attention_mask_ = ggml_new_tensor_2d(batch_ctx_, GGML_TYPE_F16, total_tokens, total_tokens);
    if (!batch_positions_ || !batch_attention_mask_) {
        return false;
    }

    batch_buffer_ = backend_->alloc_buffer(batch_ctx_, BufferUsage::Weights);
    if (!batch_buffer_) {
        return false;
    }

    std::vector<int32_t> positions(static_cast<size_t>(total_tokens));
    for (int64_t i = 0; i < total_tokens; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i % tokens_per_patch);
    }
    backend_->tensor_set(batch_positions_, positions.data(), 0, positions.size() * sizeof(int32_t));

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg = ggml_fp32_to_fp16(kLocEncBatchMaskNeg);
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(total_tokens) * static_cast<size_t>(total_tokens), neg);
    for (int64_t p = 0; p < seq_len; ++p) {
        const int64_t base = p * tokens_per_patch;
        for (int64_t row = base; row < base + tokens_per_patch; ++row) {
            ggml_fp16_t* row_data = mask.data() + static_cast<size_t>(row) * static_cast<size_t>(total_tokens);
            std::fill(row_data + base, row_data + base + tokens_per_patch, zero);
        }
    }
    backend_->tensor_set(batch_attention_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    batch_patch_size_ = patch_size;
    batch_seq_len_ = seq_len;
    return true;
}

ggml_tensor* LocEncModel::forward_sequence(VoxCPMContext& ctx, ggml_tensor* input) {
    VOXCPM_ASSERT(input != nullptr);
    VOXCPM_ASSERT(backend_ != nullptr);
    VOXCPM_ASSERT(scratch_kv_cache_ != nullptr);
    VOXCPM_ASSERT(ggml_n_dims(input) == 3);
    VOXCPM_ASSERT(input->ne[1] > 0);
    VOXCPM_ASSERT(input->ne[2] > 0);

    ggml_context* raw = ctx.raw_context();
    const int64_t patch_size = input->ne[1];
    const int64_t seq_len = input->ne[2];
    const int hidden_size = config().hidden_size;

    VOXCPM_ASSERT(patch_size + 1 <= config().max_length);
    VOXCPM_ASSERT(input->ne[0] == feat_dim_ || input->ne[0] == hidden_size);

    // Batch all patches through one encoder forward: patches are concatenated
    // along the sequence axis, isolated by a block-diagonal attention mask,
    // with RoPE positions restarting at every patch. This replaces seq_len
    // separate 8-layer forwards (whose tiny per-patch kernels made prefill
    // dispatch-bound) with one wide pass. VOXCPM_LOCENC_SEQUENTIAL=1 restores
    // the per-patch loop.
    const int64_t tokens_per_patch = patch_size + 1;
    const int64_t total_tokens = tokens_per_patch * seq_len;
    const char* force_seq = std::getenv("VOXCPM_LOCENC_SEQUENTIAL");
    const bool use_batched = !(force_seq && *force_seq && std::strcmp(force_seq, "0") != 0) &&
                             total_tokens <= scratch_kv_cache_->max_length() &&
                             ensure_batch_constants(patch_size, seq_len);
    if (!use_batched) {
        return forward_sequence_looped(ctx, input);
    }

    ggml_tensor* projected = input;
    if (input->ne[0] != hidden_size) {
        projected = ggml_mul_mat(raw, weights_.in_proj_weight, input);
        projected = ggml_add(raw, projected, weights_.in_proj_bias);
    }

    ggml_tensor* cls = ggml_reshape_3d(raw, weights_.special_token, hidden_size, 1, 1);
    ggml_tensor* cls_shape = ggml_new_tensor_3d(raw, GGML_TYPE_F32, hidden_size, 1, seq_len);
    ggml_tensor* cls_seq = ggml_repeat(raw, cls, cls_shape);
    ggml_tensor* full_input = ggml_concat(raw, cls_seq, projected, 1);
    ggml_tensor* flat_input = ggml_reshape_2d(raw, full_input, hidden_size, total_tokens);

    ggml_tensor* hidden = encoder_.forward(ctx,
                                           flat_input,
                                           batch_positions_,
                                           *scratch_kv_cache_,
                                           false,
                                           false,
                                           batch_attention_mask_);

    ggml_tensor* cls_out = ggml_view_2d(raw,
                                        hidden,
                                        hidden_size,
                                        seq_len,
                                        static_cast<size_t>(tokens_per_patch) * hidden->nb[1],
                                        0);
    return ggml_cont(raw, cls_out);
}

ggml_tensor* LocEncModel::forward_sequence_looped(VoxCPMContext& ctx, ggml_tensor* input) {
    ggml_context* raw = ctx.raw_context();
    const int64_t patch_size = input->ne[1];
    const int64_t seq_len = input->ne[2];
    const int hidden_size = config().hidden_size;

    ggml_tensor* output = ggml_new_tensor_2d(raw, GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor* sync = nullptr;

    for (int64_t idx = 0; idx < seq_len; ++idx) {
        ggml_tensor* patch_view = ggml_view_2d(raw,
                                               input,
                                               input->ne[0],
                                               patch_size,
                                               input->nb[1],
                                               static_cast<size_t>(idx) * input->nb[2]);
        ggml_tensor* hidden = forward_patch(ctx, patch_view);
        ggml_tensor* out_view = ggml_view_1d(raw,
                                             output,
                                             hidden_size,
                                             static_cast<size_t>(idx) * output->nb[1]);
        ggml_tensor* copied = ggml_cpy(raw, hidden, out_view);
        ggml_tensor* copied_sum = ggml_sum(raw, copied);
        sync = sync ? ggml_add(raw, sync, copied_sum) : copied_sum;
    }

    if (!sync) {
        return output;
    }

    sync = ggml_scale(raw, sync, 0.0f);
    return ggml_add(raw, output, sync);
}

}  // namespace voxcpm
