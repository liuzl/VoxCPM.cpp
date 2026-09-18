#!/usr/bin/env python3
"""Bounded real-model HTTP regression: one decode step must never certify a reply."""
import argparse
import http.client
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="voxcpm-truncation-") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with open(Path(directory) / "server.log", "w+") as log:
            process = subprocess.Popen([
                str(Path(args.server).resolve()), "--model-path", args.model,
                "--model-name", "test-model", "--voice-dir", str(Path(directory) / "voices"), "--backend", args.backend,
                "--host", "127.0.0.1", "--port", str(port), "--disable-auth",
                "--max-decode-steps", "1", "--inference-timesteps", "1",
            ], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during startup")
                    try:
                        with urllib.request.urlopen(base + "/healthz", timeout=1):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("server readiness timeout")
                        time.sleep(0.1)
                for stream, sse in [(False, False), (True, True), (True, False)]:
                    body = {"model": "test-model", "input": "这是一个不应该在一步解码后被当作完整回答的句子。", "voice": "design",
                            "response_format": "wav", "stream": stream, "seed": 7}
                    if sse:
                        body["stream_format"] = "sse"
                    request = urllib.request.Request(base + "/v1/audio/speech",
                        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    if not stream or sse:
                        try:
                            urllib.request.urlopen(request, timeout=60)
                        except urllib.error.HTTPError as error:
                            assert error.code == 500, (error.code, error.read())
                            assert json.load(error)["error"]["code"] == "synthesis_truncated"
                        else:
                            raise AssertionError("truncated synthesis returned success")
                    else:
                        with urllib.request.urlopen(request, timeout=60) as response:
                            assert response.status == 200
                            try:
                                response.read()
                            except http.client.IncompleteRead:
                                pass
                            else:
                                raise AssertionError("truncated PCM stream ended cleanly")
                print("PASS: WAV and SSE reject truncation; PCM transfer is incomplete")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
