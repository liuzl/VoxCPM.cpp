#!/usr/bin/env python3
"""Re-encode one saved voice's source WAV under a NEW ID through a running server.

Never edits the old manifest/features, overwrites a voice, or changes a binding.
API credentials are read from an environment variable, never command-line values.
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--new-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-env", default="VOXCPM_API_KEY")
    args = parser.parse_args()
    for voice_id in [args.voice_id, args.new_id]:
        if voice_id in (".", "..") or re.fullmatch(r"[A-Za-z0-9._-]+", voice_id) is None:
            parser.error("voice IDs must be safe single path components")
    if args.new_id == args.voice_id:
        parser.error("--new-id must differ from the source ID")
    directory = Path(args.voice_dir) / args.voice_id
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("design_profile"):
        parser.error("design profiles require explicit provenance migration; this tool only rebuilds registered voices")
    source = manifest.get("source_audio", {})
    if source.get("file") != "ref.wav":
        parser.error("no preserved source WAV; obtain the original recording instead of converting latent features")
    audio = (directory / "ref.wav").read_bytes()
    if not audio or len(audio) != source.get("bytes"):
        parser.error("source WAV is missing, empty, or has a different size than its manifest")
    prompt = manifest.get("prompt_audio_length", 0) > 0
    reference = manifest.get("reference_audio_length", 0) > 0
    if prompt == reference:
        parser.error("expected one stored conditioning mode; empty/combined features require manual migration")
    text = manifest.get("prompt_text", "")
    if prompt and not text.strip():
        parser.error("continuation voice has no transcript")
    mode = "continuation" if prompt else "reference"
    boundary = "voxcpm-reencode-" + uuid.uuid4().hex
    body = bytearray()
    for key, value in {"id": args.new_id, "mode": mode, "text": text if prompt else ""}.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="ref.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    body.extend(audio)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    if token := os.environ.get(args.token_env):
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(args.url.rstrip("/") + "/v1/voices", data=bytes(body), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
            if (response.status != 201 or result.get("id") != args.new_id or not result.get("encoder_contract")
                or (result.get("prompt_audio_length", 0) > 0) != prompt
                or (result.get("reference_audio_length", 0) > 0) != reference):
                raise RuntimeError("unexpected registration response; inspect the new ID before retrying")
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Registration returned HTTP {error.code}; no retry performed. Inspect the new ID before retrying.")
    print(json.dumps({"id": args.new_id, "mode": mode, "encoder_contract": result["encoder_contract"]}))


if __name__ == "__main__":
    main()
