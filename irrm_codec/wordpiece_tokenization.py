"""WordPiece tokenizer for CDR3 sequences, as an alternative to the char-level one.

Requires: pip install tokenizers
Requires a tokenizer.json with special tokens at [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3 —
same ids irrm_codec.tokenization uses. A mismatched file raises ValueError.
"""
from pathlib import Path

from tokenizers import Tokenizer

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID

_EXPECTED_SPECIAL_IDS = {"[PAD]": PAD_ID, "[UNK]": UNK_ID, "[BOS]": BOS_ID, "[EOS]": EOS_ID}


def _validate_special_ids(tokenizer, tokenizer_path):
    """Check PAD/UNK/BOS/EOS ids match, since ForwardModel/InverseModel assume padding_idx=0."""
    for token, expected_id in _EXPECTED_SPECIAL_IDS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"Tokenizer at {tokenizer_path} has {token}={actual_id}, expected {expected_id}."
            )


def load_wordpiece_tokenizer(path) -> Tokenizer:
    """Load and validate a tokenizer.json. Returns the Tokenizer itself — it's picklable,
    so it can go straight into a DataLoader worker."""
    tokenizer = Tokenizer.from_file(str(Path(path)))
    _validate_special_ids(tokenizer, path)
    return tokenizer


def encode_wordpiece(seq, tokenizer, max_len):
    """Encode a CDR3 sequence into exactly max_len ids: truncated or PAD-padded."""
    seq = "" if seq is None else str(seq).strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    ids = tokenizer.encode(seq).ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def decode_wordpiece(token_ids, tokenizer, stop_at_eos=True):
    """Decode ids back to an amino-acid string, stopping at the first EOS by default."""
    ids = list(token_ids)
    if stop_at_eos and EOS_ID in ids:
        ids = ids[: ids.index(EOS_ID)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def wordpiece_vocab_size(tokenizer) -> int:
    """Vocab size, for ForwardModel/InverseModel(vocab_size=...)."""
    return tokenizer.get_vocab_size()


class WordpieceEncodeFn:
    """Binds a tokenizer to encode_wordpiece so it matches datasets.py's encode_fn(seq, max_len)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seq, max_len):
        return encode_wordpiece(seq, self.tokenizer, max_len)
