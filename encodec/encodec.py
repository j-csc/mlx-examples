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


class EncodecResnetBlock(nn.Module):
    def __init__(
        self,
        args: ModelConfig,
        dim: int,
        kernel_sizes: Sequence[int],
        dilations: Sequence[int],
        true_skip: bool = True,
    ):
        super().__init__()
        self.compress = args.compress
        self.dim = dim
        self.hidden = dim // self.compress
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.block = nn.Sequential(*self._make_layer(self.dim, self.kernel_sizes))
        if true_skip:
            self.shortcut = mx.Identity()
        else:
            self.shortcut = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, dim, kernel_size):
        block = []
        for i, (kernel_size, dilation) in enumerate(
            zip(self.kernel_sizes, self.dilations)
        ):
            in_chs = dim if i == 0 else self.hidden
            out_chs = dim if i == len(self.kernel_sizes) - 1 else self.hidden
            block += [
                nn.ELU(),
                nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                ),
            ]

    def __call__(self, x):
        out = self.shortcut(x) + self.block(x)
        return out


class EncodecEncoder(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.dimension = args.hidden_size // 2
        self.ratios = list(reversed(args.upsampling_ratios))
        self.hop_length = mx.prod(mx.array(self.ratios))

        self.elu = nn.ELU()

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
                    self.layers += [
                        EncodecResnetBlock(args=args, dim=args.num_filters * mult)
                    ]
                    pass

            pass

    def __call__(self, x: mx.array):
        return x


class EncodecDecoder(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args


class EncodecModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.encoder = EncodecEncoder(args=args)
        self.quantizer = ResidualVectorQuantizer(args=args)
        self.decoder = EncodecDecoder(args=args)

    def __call__(self, x: mx.array):
        return self.decoder(self.encoder(x))


def print_weights(weights):
    for k, v in weights.items():
        print(k, v.shape)


def load_model(model_path):
    weights = mx.load(model_path)
    print_weights(weights)
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
    encoder = EncodecEncoder()
    decoder = EncodecDecoder()
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
