from typing import List, Optional, Sequence
from dataclasses import dataclass, field

import argparse
import json
import torchaudio
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten

from config import ModelConfig


class SeaNetResnetBlock(nn.Module):
    def __init__(self, args: ModelConfig, dim: int):
        super().__init__()
        self.compress = args.compress
        self.dim = dim
        self.hidden = dim // self.compress
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, self.hidden, kernel_size=3, stride=1, bias=False),
            nn.ELU(),
            nn.Conv1d(self.hidden, dim, kernel_size=1, stride=1, bias=False),
        )
        self.shortcut = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)

    def __call__(self, x):
        out = self.shortcut(x) + self.block(x)
        return out


class SEANetEncoder(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.dimension = args.hidden_size // 2
        self.ratios = list(reversed(args.upsampling_ratios))
        self.hop_length = mx.prod(mx.array(self.ratios))

        self.elu = nn.ELU()
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=args.audio_channels,
                out_channels=args.num_filters,
                kernel_size=args.kernel_size,
            ),
            *[SeaNetResnetBlock(args=args, dim=args.num_filters) for _ in range(1)],
            nn.Conv1d(
                in_channels=args.num_filters,
                out_channels=args.num_filters,
                kernel_size=args.kernel_size,
            ),
        )

        def _make_layer(in_channels, out_channels, kernel_size):
            mult = 1
            self.layers = [
                nn.Conv1d(
                    in_channels=mult * in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            ]
            for i, ratio in enumerate(self.ratios):
                for j in range(args.num_residual_layers):
                    self.layers.append(
                        SeaNetResnetBlock(args=args, dim=args.num_filters * mult)
                    )
                    pass

            pass

    def __call__(self, x: mx.array):
        return x


class SEANetDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class ResidualVectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()


class EncodecModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.encoder = SEANetEncoder(args=args)
        # self.quantizer = ResidualVectorQuantizer()
        # self.decoder = SEANetDecoder()

    def __call__(self, x: mx.array):
        return self.decoder(self.encoder(x))


def load_model(model_path):
    weights = mx.load(model_path)
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda x: x.astype(mx.float32), weights)
    with open("weights/config.json", "r") as f:
        config = json.loads(f.read())
        unused = [
            "_name_or_path",
            "architectures",
            "model_type",
            "torch_dtype",
            "transformers_version",
        ]
        for key in unused:
            del config[key]
    model = EncodecModel(ModelConfig(**config))
    # model.update(weights)
    return model


def test_load_input(fp: str):
    wav, sr = torchaudio.load(fp)
    wav = wav[:, : 24000 * 2]
    wav_in = wav.unsqueeze(0)
    wav_in = mx.array(wav_in.numpy())
    return wav_in


def test_encodec_decoder():
    encoder = SEANetEncoder()
    decoder = SEANetDecoder()
    x = mx.random.randn(1, 1, 24000)
    z = encoder(x)
    assert list(z.shape) == [1, 128, 75], z.shape
    y = decoder(z)
    assert y.shape == x.shape, (x.shape, y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encodec inference script")
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/weights.npz",
        help="The path to the model weights",
    )
    parser.add_argument("--seed", type=int, default=42, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    print("[INFO] Loading model from disk.")
    model = load_model(args.model_path)
    print(model)

    print("[INFO] Loading test audio file.")
    wav_in = test_load_input("test_24k.wav")
