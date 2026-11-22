"""CRNN shapes, CTC lengths, and that the model can actually learn."""
import torch

from src.config import ModelConfig
from src.data.charset import Charset
from src.data.iam import collate_fn
from src.models.crnn import CRNN
from src.training.engine import ctc_loss_on_batch, overfit_one_batch
from src.training.sampler import BucketBatchSampler


def test_output_shape_and_log_probs():
    cs = Charset()
    m = CRNN(cs.num_classes, ModelConfig(lstm_hidden=32, cnn_channels=(8, 16, 16, 16, 32))).eval()
    out = m(torch.randn(3, 1, 32, 100))
    assert out.shape == (25, 3, cs.num_classes)            # T = 100 // 4
    assert torch.allclose(out.exp().sum(2), torch.ones(25, 3), atol=1e-4)  # each step sums to 1


def test_output_lengths_match_cnn():
    m = CRNN(10, ModelConfig(lstm_hidden=16, cnn_channels=(8, 8, 8, 8, 8))).eval()
    for w in (16, 17, 63, 64, 130):
        assert m(torch.zeros(1, 1, 32, w)).shape[0] == CRNN.output_lengths(torch.tensor([w])).item()


def test_ctc_loss_is_finite():
    cs = Charset()
    batch = collate_fn([(torch.randn(1, 32, 80), torch.tensor(cs.encode("hello")), 5, "hello"),
                        (torch.randn(1, 32, 40), torch.tensor(cs.encode("hi")), 2, "hi")])
    m = CRNN(cs.num_classes, ModelConfig(lstm_hidden=32))
    loss, log_probs, lengths = ctc_loss_on_batch(m, batch, torch.device("cpu"))
    assert torch.isfinite(loss) and lengths.tolist() == [20, 10]


def test_bucket_sampler_covers_everything_once():
    widths = [5, 100, 7, 90, 50, 51, 6, 99, 30, 31]
    s = BucketBatchSampler(widths, batch_size=3, seed=1)
    batches = list(s)
    assert sorted(i for b in batches for i in b) == list(range(10))
    s.set_epoch(1)
    assert list(s) != batches or len(batches) == 1         # reshuffled next epoch


class _TinyDS:
    """8 synthetic 'words': distinct bar patterns per character, so they're learnable."""
    def __init__(self, cs):
        self.items, self.cs = ["ab", "ba", "abc", "cab", "a", "bc", "cc", "acb"], cs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        t = self.items[i]
        img = torch.ones(1, 32, 16 * len(t))
        for k, ch in enumerate(t):
            rows = {"a": slice(4, 12), "b": slice(12, 20), "c": slice(20, 28)}[ch]
            img[0, rows, 16 * k + 4: 16 * k + 12] = -1.0
        return img, torch.tensor(self.cs.encode(t)), len(t), t


def test_overfit_one_batch_learns():
    cs = Charset()
    losses, preds = overfit_one_batch(_TinyDS(cs), cs, ModelConfig(lstm_hidden=64, dropout=0.0,
                                      cnn_channels=(16, 32, 32, 32, 64)), steps=150, lr=3e-3)
    assert losses[-1] < 0.25 * losses[0]
    assert sum(r == h for r, h in preds) >= 6               # memorised most of the batch
