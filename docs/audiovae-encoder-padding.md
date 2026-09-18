# AudioVAE encoder padding contract

The V2 encoder downsampling convolution uses left padding
`2 * ceil(stride / 2) - stride % 2`. V1 uses `2 * ceil(stride / 2)`.
For odd strides, including the V2 encoder's stride 5, equal output lengths do not
imply equivalent features: the old formula shifts the convolution's input window.

This fix follows the operator correction identified in
[upstream PR #16](https://github.com/bluryar/VoxCPM.cpp/pull/16) and the
[official V2 implementation](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py).
It does not adopt that PR's model-name overrides, latent feature conversion,
request-mode switching or Vulkan changes. The runtime reads `voxcpm_architecture`
from GGUF, selecting V2 only for `voxcpm2`; legacy V1 behavior is retained.
VoxCPM2 exports must include this architecture metadata, as required for reference
registration. `--model-name` remains an API alias.

## Layout and ownership

Torch `[B,C,T]` inputs map to GGML `[T,C,B]`; convolution weights `[OC,IC,K]`
map to GGML `[K,IC,OC]`. Padding is applied on the input time axis before
`im2col -> mul_mat`. The existing final transpose/contiguous conversion is kept.
Residual stride-1 convolutions and decoder/stateful decode paths are unchanged.
Weights still belong to the shared WeightStore. No new inference-time host
materialization or per-module model load is introduced.

## Numerical verification

The model-free test covers odd/even strides with a separate cross-correlation
oracle. Full-encoder verification loads the official Python V1 and V2 encoder
classes and supplies the same encoder weights from one GGUF to both. Two input
lengths and left/right patch padding produce eight synthetic cases. The V1 cases
check the original formula using the same weights; they are not a qualification
of a separate VoxCPM1.5 model.

For a strict comparison, promote the encoder's F16 weights to F32 storage in a
**new test-only GGUF**. Their numerical values do not change; all other tensors
are copied unchanged. This isolates padding from backend activation/accumulation
precision. The original model is never overwritten. The JSON records hashes of
the Python source files used. Keep derived models and traces outside Git.

```sh
uv run --no-project --with torch --with numpy --with pydantic --with gguf \
  python scripts/export_encoder_parity.py \
  --model /path/to/voxcpm2.gguf --python-root /path/to/VoxCPM \
  --output /tmp/encoder-parity.json \
  --encoder-f32-output /tmp/encoder-f32.gguf
cmake --build build --target test_audio_vae test_server_common voxcpm-server
build/tests/test_audio_vae '[padding]'
VOXCPM_MODEL_PATH=/tmp/encoder-f32.gguf \
  VOXCPM_ENCODER_PARITY_JSON=/tmp/encoder-parity.json \
  build/tests/test_audio_vae '[encoder-parity]'
```

The fixed CPU/F32 strict profile must satisfy max absolute error < 0.002 and RMSE < 0.0002.
Each V2 case also verifies that the old padding has at least ten times the RMSE.
Backend arithmetic has a separate precision gap: the normal CPU/F16 runtime and
Metal comparisons did not meet these strict CPU/F32 tolerances, including in the
V1 control. Promoting encoder storage to F32 alone did not close the Metal gap.
Do not interpret passing the promoted-weight check as F16 parity, audible quality,
or CUDA qualification. Do not increase tolerances to hide that gap.

## Existing voice features

New registrations include `encoder_contract` in their manifest and HTTP metadata:
`audiovae-v2-causal-padding-1` or `audiovae-v1-causal-padding-1`.
Both continuation and reference features depend on the encoder. Existing V2
voices without the current tag remain readable, but synthesis rejects them before
using their features. Unknown/incompatible nonempty tags are also rejected.
Legacy untagged V1 voices remain usable. This tag versions the padding contract;
it does not replace model identity or model-weight verification.

Rebuild from the retained **source WAV**, never by shifting latent patches or
editing the tag. Use a different ID, audition it, then explicitly change the
consumer's binding. No production voice or binding is migrated automatically.

```sh
# The server must run this fix with the intended model. Authentication, if needed,
# is read from VOXCPM_API_KEY (or the variable named by --token-env).
python3 scripts/reencode_voice.py \
  --voice-dir /path/to/voices --voice-id old-voice --new-id old-voice-padding1 \
  --url http://127.0.0.1:8080
```

The helper preserves the stored conditioning mode and continuation transcript.
It checks the source file and refuses an identical target ID. It cannot migrate
feature-only voices without source recordings, combined-mode voices or design
profiles whose provenance requires separate handling. Keep the original entry;
there is no automatic overwrite or retry after an uncertain response.

The HTTP regression uses only synthetic audio and checks stale-cache rejection,
new-ID rebuilding, byte-for-byte preservation of the old entry and successful
synthesis from the rebuilt voice:

```sh
python3 tests/test_reference_registration.py --server build/examples/voxcpm-server \
  --model /path/to/voxcpm2.gguf --backend metal
```
