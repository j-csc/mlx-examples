import argparse
import numpy as np
from pathlib import Path
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Encodec Pytorch weights to MLX."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/",
        help="The path to the Encodec model. The MLX weights will also be saved there.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    state = torch.load(str(model_path / "pytorch_model.bin"))

    # print weights
    for k, v in state.items():
        print(k, v.shape)

    np.savez(
        str(model_path / "weights.npz"),
        **{k: v.to(torch.float16).numpy() for k, v in state.items()}
    )
