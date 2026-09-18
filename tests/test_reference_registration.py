#!/usr/bin/env python3
"""Real VoxCPM2 HTTP contract regression; synthetic audio is not a voice-quality test."""
import argparse
import io
import json
import math
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        # Deliberately not patch-aligned: left and right padding must differ.
        output.writeframes(b"".join(struct.pack("<h", int(2000 * math.sin(i * 0.0864))) for i in range(3700)))
    with tempfile.TemporaryDirectory(prefix="voxcpm-reference-") as directory:
        root = Path(directory)
        voices = root / "voices"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        def request(path, data=None, content_type="application/json"):
            try:
                with urllib.request.urlopen(urllib.request.Request(base + path, data=data,
                    headers={"Content-Type": content_type}), timeout=60) as response:
                    return response.status, response.read(), response.headers
            except urllib.error.HTTPError as error:
                return error.code, error.read(), error.headers

        def register(voice_id, mode=None, text=None):
            boundary = "reference-contract-boundary"
            body = bytearray()
            for key, value in {"id": voice_id, "mode": mode, "text": text}.items():
                if value is not None:
                    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
            body.extend(audio.getvalue())
            body.extend(f"\r\n--{boundary}--\r\n".encode())
            status, result, _ = request("/v1/voices", bytes(body), "multipart/form-data; boundary=" + boundary)
            return status, json.loads(result)

        def start(log):
            process = subprocess.Popen([
                str(Path(args.server).resolve()), "--model-path", args.model, "--model-name", "test",
                "--voice-dir", str(voices), "--backend", args.backend, "--host", "127.0.0.1",
                "--port", str(port), "--disable-auth", "--inference-timesteps", "4",
                "--max-decode-steps", "128",
            ], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during startup")
                    try:
                        if request("/healthz")[0] == 200:
                            return process
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                raise RuntimeError("server readiness timeout")
            except BaseException:
                stop(process)
                raise

        def stop(process):
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        with open(root / "server.log", "w+") as log:
            process = None
            try:
                process = start(log)
                for voice_id, mode, text in [("invalid", "bogus", None), ("missing", None, None),
                                            ("empty", "continuation", ""), ("blank", "continuation", " \t\r\n")]:
                    status, result = register(voice_id, mode, text)
                    assert status == 400, (voice_id, status, result)
                    assert not (voices / voice_id).exists(), voice_id
                for voice_id, mode, text in [("legacy", None, "你好"), ("explicit", "continuation", "你好"),
                                            ("reference", "reference", None), ("reference_text", "reference", "unused")]:
                    status, result = register(voice_id, mode, text)
                    assert status == 201, (voice_id, status, result)
                    is_reference = mode == "reference"
                    assert (result["prompt_audio_length"] == 0) == is_reference
                    assert (result["reference_audio_length"] > 0) == is_reference
                    assert result["prompt_text"] == ("" if is_reference else text)
                    assert (voices / voice_id / "ref.wav").is_file()
                assert (voices / "legacy/prompt_feat.bin").read_bytes() == (voices / "explicit/prompt_feat.bin").read_bytes()
                ref = (voices / "reference/reference_feat.bin").read_bytes()
                assert ref == (voices / "reference_text/reference_feat.bin").read_bytes()
                assert ref != (voices / "legacy/prompt_feat.bin").read_bytes()
                before = request("/v1/voices/reference")[1]
                assert register("reference", "reference")[0] == 409
                assert request("/v1/voices/reference")[1] == before
                assert len(json.loads(request("/v1/voices")[1])["voices"]) == 4
                stop(process)
                process = start(log)
                assert json.loads(request("/v1/voices/reference")[1])["reference_audio_length"] > 0
                for voice_id, streaming in [("legacy", False), ("reference", False), ("reference", True)]:
                    status, data, headers = request("/v1/audio/speech", json.dumps({
                        "model": "test", "input": "你好。", "voice": voice_id, "response_format": "wav",
                        "stream": streaming, "seed": 7,
                    }).encode())
                    assert status == 200, (voice_id, streaming, status, data[:200])
                    if streaming:
                        assert headers.get("Content-Type").startswith("audio/pcm")
                        assert int(headers["X-Sample-Rate"]) > 0
                        assert len(data) > 0 and len(data) % 4 == 0
                        assert all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", data))
                    else:
                        assert data[:4] == b"RIFF" and len(data) > 44
                print("PASS: registration validation, mode isolation, source retention, restart, legacy WAV, reference WAV/PCM")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:])
                raise
            finally:
                if process is not None:
                    stop(process)


if __name__ == "__main__":
    main()
