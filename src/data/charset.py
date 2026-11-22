"""Character set, label encoding, and greedy CTC decoding.

CTC needs one extra class, the "blank", which means "no character here".
We reserve index 0 for it, so real characters start at index 1.
"""
import string

# Letters, digits and the punctuation that appears in IAM word labels.
# (IAM words never contain a space, since each image is one word.)
DEFAULT_CHARS = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + "!\"#&'()*+,-./:;?"
)

BLANK = 0  # CTC blank index


class Charset:
    def __init__(self, chars: str = DEFAULT_CHARS):
        if len(set(chars)) != len(chars):
            raise ValueError("charset contains duplicate characters")
        self.chars = chars
        # char -> index, starting at 1 because 0 is the blank
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}

    @property
    def num_classes(self) -> int:
        """Output size of the model's last layer (characters + blank)."""
        return len(self.chars) + 1

    def is_valid(self, text: str) -> bool:
        """True if every character of `text` is in the charset."""
        return all(c in self.char_to_idx for c in text)

    def encode(self, text: str) -> list[int]:
        """'cat' -> [3, 1, 20]. Raises KeyError on unknown characters."""
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices) -> str:
        """Plain lookup, no CTC logic. Blanks (0) are skipped."""
        return "".join(self.idx_to_char[int(i)] for i in indices if int(i) != BLANK)

    def ctc_greedy_decode(self, indices) -> str:
        """Turn a per-frame best path into text.

        CTC rule: first merge repeated indices, then remove blanks.
        e.g. [0, 3, 3, 0, 1, 1, 20, 0] -> [3, 1, 20] -> 'cat'
        A blank between two equal letters keeps both: [12, 0, 12] -> 'll'.
        """
        out, prev = [], None
        for i in indices:
            i = int(i)
            if i != prev and i != BLANK:
                out.append(self.idx_to_char[i])
            prev = i
        return "".join(out)

    def decode_batch(self, log_probs) -> list[str]:
        """Greedy-decode model output of shape (T, B, C) into B strings.

        T = time steps (image columns after the CNN), B = batch, C = classes.
        This is the same layout nn.CTCLoss expects.
        """
        best = log_probs.argmax(dim=2)  # (T, B): most likely class per frame
        return [self.ctc_greedy_decode(best[:, b].tolist()) for b in range(best.shape[1])]
