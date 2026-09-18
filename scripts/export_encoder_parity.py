#!/usr/bin/env python3
"""Export synthetic AudioVAE traces using official Python code and identical GGUF weights.

Run with torch, numpy, pydantic and gguf installed. Output contains generated input
and model-derived features; keep it outside Git. This isolates encoder arithmetic,
not model quantization quality or speaker fidelity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import gguf
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder-f32-output", help="Derived GGUF for strict parity: promote only encoder F16 values to F32")
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.model).resolve() or (
        args.encoder_f32_output and Path(args.output).resolve() == Path(args.encoder_f32_output).resolve()
    ):
        raise ValueError("Trace output must not overwrite a model")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    reader = gguf.GGUFReader(args.model)

    def metadata(key, default=None):
        field = reader.fields.get(key)
        if field is None:
            return default
        values = [field.parts[i].tolist() for i in field.data]
        if field.types[0] == gguf.GGUFValueType.STRING:
            return bytes(values[0]).decode()
        if field.types[0] == gguf.GGUFValueType.ARRAY:
            return [value[0] for value in values]
        return values[0][0]

    if metadata("voxcpm_architecture") != "voxcpm2":
        raise ValueError("This comparison requires a VoxCPM2 GGUF with architecture metadata")
    if args.encoder_f32_output:
        destination = Path(args.encoder_f32_output)
        if destination.exists() or destination.resolve() == Path(args.model).resolve():
            raise ValueError("Derived model output must be a new file, never overwrite the source")
        writer = gguf.GGUFWriter(str(destination), metadata("general.architecture"))
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        for tensor in reader.tensors:
            if tensor.name.startswith("audio_vae.encoder."):
                if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                    raise ValueError("Only F32/F16 encoder weights are supported")
                writer.add_tensor(tensor.name, np.array(tensor.data, dtype=np.float32))
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    prefix = "voxcpm_audio_vae_config_"
    strides = metadata(prefix + "encoder_rates")
    patch_size = metadata("voxcpm_patch_size")
    hop = int(np.prod(strides))
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("Missing patch_size")
    weights = {tensor.name.removeprefix("audio_vae.encoder."): tensor for tensor in reader.tensors
               if tensor.name.startswith("audio_vae.encoder.")}
    encoder_hash = hashlib.sha256()
    for name, weight in sorted(weights.items()):
        encoder_hash.update(name.encode() + b"\0")
        encoder_hash.update(np.asarray(weight.data, dtype=np.float32).tobytes())
    report = {"encoder_values_sha256": encoder_hash.hexdigest(), "schema": "voxcpm.encoder-parity.v1", "source_sha256": {}, "torch": torch.__version__,
              "sample_rate": metadata(prefix + "sample_rate"), "cases": []}
    for version in (1, 2):
        filename = "audio_vae_v2.py" if version == 2 else "audio_vae.py"
        source = Path(args.python_root) / "src/voxcpm/modules/audiovae" / filename
        report["source_sha256"][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(f"official_vae_{version}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.CausalEncoder(d_model=metadata(prefix + "encoder_dim"),
            latent_dim=metadata(prefix + "latent_dim"), strides=strides,
            depthwise=bool(metadata(prefix + "depthwise", True)))
        for layer in encoder.modules():
            if hasattr(layer, "weight_g"):
                torch.nn.utils.remove_weight_norm(layer)
        state = {}
        for name, tensor in encoder.state_dict().items():
            weight = weights[name]
            if weight.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                raise ValueError(f"Encoder parity requires F32/F16 weights: {name}")
            state[name] = torch.from_numpy(np.array(weight.data, dtype=np.float32).reshape(tensor.shape))
        encoder.load_state_dict(state, strict=True)
        encoder.eval()
        for samples in (3700, 10421):
            t = np.arange(samples, dtype=np.float32) / report["sample_rate"]
            signal = (0.12 * np.sin(2 * np.pi * 223 * t) + 0.04 * np.sin(2 * np.pi * 617 * t)).astype(np.float32)
            pad = (-samples) % (patch_size * hop)
            for side in ("left", "right"):
                aligned = np.pad(signal, (pad, 0) if side == "left" else (0, pad))
                with torch.inference_mode():
                    hidden = encoder.block(torch.from_numpy(aligned).reshape(1, 1, -1))
                    latent = encoder.fc_mu(hidden).numpy()
                report["cases"].append({"name": f"v{version}-{side}-{samples}", "v2": version == 2,
                    "audio": aligned.tolist(), "shape": list(latent.shape), "latent": latent.flatten().tolist()})
        del encoder, state
    Path(args.output).write_text(json.dumps(report))
    print(f"Exported {len(report['cases'])} cases; V1/V2 use the same GGUF weights to isolate padding semantics")


if __name__ == "__main__":
    main()
