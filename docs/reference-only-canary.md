# Reference-only registration canary

Date: 2026-09-14. Status: API verified; acoustic migration not qualified.

This canary predates the [V2 encoder padding correction](audiovae-encoder-padding.md).
Its acoustic measurements do not qualify features rebuilt with the corrected encoder.

`POST /v1/voices` now exposes the existing reference encoder with
`mode=reference`. Omitted mode preserves continuation registration. The reference
path right-pads audio patches and stores reference features without a transcript;
it does not relabel continuation features. Speech requests, including streaming,
consume the stored mode. See the registration contract in [README](../README.md).

## Validation

- Server utility tests: 8 cases, 82 assertions passed, including reference-only
  feature persistence and legacy continuation readability.
- HTTP checks: invalid modes rejected; default continuation still requires text;
  reference registration without text succeeds and reads back zero prompt frames
  and positive reference frames. Continuation reads back the inverse.
- Runtime-boundary audit has the same unique findings before and after the patch;
  no decode, state ownership, weight-loading or graph implementation changed.
- Isolated CUDA canary: same VoxCPM2 Q4_K / AudioVAE F16 model, same English
  reference recording, same Chinese text containing an English name, CFG 2,
  10 steps, seeds 1234 and 5678. Both modes use the same new binary. Compare
  concatenated text against three independent original chunks. Sixteen batch
  calls plus two full-text streaming calls completed, without synthesis retries.
- Streaming returned nonempty PCM with the expected sample rate and matched
  batch duration and Chinese-only CER for each corresponding full-text sample.

## Observations

| Registration / segmentation | Full normalized ASR CER | Chinese-only ASR CER | Whole-clip speaker cosine |
|---|---:|---:|---:|
| Continuation / full text | 26.8–43.9% | 10.7–17.9% | 0.909–0.931 |
| Continuation / three chunks | 56.1–78.0% | 39.3–67.9% | 0.912–0.926 |
| Reference / full text | 41.5–43.9% | 32.1–35.7% | 0.641–0.725 |
| Reference / three chunks | 19.5% | 0% | 0.794–0.810 |

Reference-only improves Chinese content retention with the original segmentation
in both seeds, but the full-text variant omits the opening sentence in ASR and
speaker similarity declines. It does not reproduce the overall quality of the
previous official-Python reference-only preview. Do not promote either mode or
sentence merging as a universal fix, or change existing voice bindings from this
canary alone. Further work should isolate reference-path numerical parity from
quantization/backend effects using fixed inputs.

ASR can misrecognize names and unclear speech. Chinese-only CER excludes Latin
names and also ignores inserted English, so it is supplementary to full CER and
audio. Speaker cosine uses Resemblyzer against a nonoverlapping same-session
English recording; it is language-sensitive and is not a listening score or a
percentage of likeness. All measured outputs were finite without sample clipping.
This one-text canary is not a long-dialogue or naturalness qualification. Raw
recordings, transcripts, receipts and full logs remain outside Git.

## Reproducible contract checks

The dated acoustic observations above remain a historical canary, not a fresh
quality qualification. The registration contract is exercised independently with
synthetic audio (no private voice fixture):

```sh
cmake --build build --target test_server_common voxcpm-server
build/tests/test_server_common '~[integration]'
python3 tests/test_reference_registration.py --server build/examples/voxcpm-server \
  --model /path/to/voxcpm2.gguf --backend metal
```

Unit checks reject missing, malformed, unknown and non-VoxCPM2 architecture
metadata even when the display name says VoxCPM2. The real-model HTTP check covers
invalid modes, absent/blank continuation transcripts, default compatibility,
reference transcript isolation, separate padding-derived features, duplicate IDs,
source retention, persisted voices after restart, legacy WAV and reference WAV/PCM.
It does not qualify speaker similarity, pronunciation or CUDA performance.
