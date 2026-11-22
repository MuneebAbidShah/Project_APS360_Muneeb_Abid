"""CRNN: CNN feature extractor -> 2-layer BiLSTM -> linear layer, trained with CTC.

Shapes for one batch (B images, padded width W):

    image          (B, 1, 32, W)
    after the CNN  (B, 512, 1, W/4)   height squeezed to 1, width / 4
    as a sequence  (T=W/4, B, 512)    one feature vector per 4-pixel-wide column
    after BiLSTM   (T, B, 2*hidden)   each step sees the whole word (both directions)
    output         (T, B, C)          log-probabilities over C = charset + blank

The output is exactly what nn.CTCLoss wants, and Charset.decode_batch decodes it.
"""
import torch
import torch.nn as nn

from src.config import ModelConfig


def conv_block(c_in, c_out, pool):
    """3x3 conv -> BatchNorm -> ReLU, then optional max-pool.

    pool=(2, 2) halves height and width; pool=(2, 1) halves only the height,
    which keeps more time steps for CTC (long words need many steps).
    """
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
              nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool, pool))
    return nn.Sequential(*layers)


class CRNN(nn.Module):
    # Width shrinks by 2 in the first two blocks only, so T = W // 4.
    WIDTH_DOWNSAMPLE = 4

    def __init__(self, num_classes: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        c = cfg.cnn_channels
        # Height: 32 -> 16 -> 8 -> 4 -> 2 -> 1.  Width: W -> W/2 -> W/4 (then unchanged).
        self.cnn = nn.Sequential(
            conv_block(1,    c[0], (2, 2)),
            conv_block(c[0], c[1], (2, 2)),
            conv_block(c[1], c[2], (2, 1)),
            conv_block(c[2], c[3], (2, 1)),
            conv_block(c[3], c[4], (2, 1)),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.rnn = nn.LSTM(c[4], cfg.lstm_hidden, num_layers=cfg.lstm_layers,
                           bidirectional=True, dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0)
        self.fc = nn.Linear(2 * cfg.lstm_hidden, num_classes)

    def forward(self, x):
        f = self.cnn(x)                  # (B, C, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # (T, B, C): time first, as the LSTM and CTC expect
        f, _ = self.rnn(self.dropout(f))
        logits = self.fc(self.dropout(f))  # (T, B, num_classes)
        return logits.log_softmax(dim=2)

    @classmethod
    def output_lengths(cls, widths: torch.Tensor) -> torch.Tensor:
        """Real (unpadded) number of time steps for each image: its width // 4.
        CTC uses these so it ignores the columns that are only padding."""
        return torch.clamp(widths // cls.WIDTH_DOWNSAMPLE, min=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
