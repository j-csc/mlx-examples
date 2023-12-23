from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    target_bandwidths: List[int] = field(
        default_factory=lambda: [1.5, 3.0, 6.0, 12.0, 24.0]
    )
    sampling_rate: int = 24000
    audio_channels: int = 1
    normalize: Optional[bool] = False
    chunk_length_s: Optional[float] = None
    overlap: Optional[float] = None
    hidden_size: int = 256
    num_filters: int = 32
    num_residual_layers: int = 1
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 5, 4, 2])
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "reflect"
    compress: int = 2
    num_lstm_layers: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 1024
    codebook_dim: Optional[int] = field(default=hidden_size)
    use_conv_shortcut: bool = True
