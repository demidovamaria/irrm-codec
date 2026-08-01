"""WordPiece tokenizer for CDR3 sequences, as an alternative to the char-level one.

Requires: pip install tokenizers
Requires a tokenizer.json with special tokens at [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3 —
same ids irrm_codec.tokenization uses. A mismatched file raises ValueError.
"""
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

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


def filter_vocab_by_max_token_length(tokenizer: Tokenizer, max_token_length: int) -> Tokenizer:
    """Drop vocab entries longer than max_token_length amino acids, keeping special tokens.

    "Length" is the piece's real amino-acid count, i.e. the "##" continuation prefix isn't
    counted. Remaining tokens get contiguous ids (0..k-1), with PAD/UNK/BOS/EOS staying at
    0/1/2/3 since they're always kept and sorted first. Encoding after filtering just falls
    back to shorter surviving pieces (WordPiece's normal longest-match behavior) - it does
    not raise or produce extra [UNK] tokens by itself.

    This does not touch the tokenizer.json file on disk or retrain anything - it only
    rebuilds an in-memory Tokenizer with a smaller vocab for the current run.
    """
    if max_token_length < 1:
        raise ValueError(f"max_token_length must be >= 1, got {max_token_length}.")

    prefix = tokenizer.model.continuing_subword_prefix
    old_vocab = tokenizer.get_vocab()

    kept_tokens = []
    for token, _old_id in sorted(old_vocab.items(), key=lambda kv: kv[1]):
        if token in _EXPECTED_SPECIAL_IDS:
            kept_tokens.append(token)
            continue
        piece = token[len(prefix):] if token.startswith(prefix) else token
        if len(piece) <= max_token_length:
            kept_tokens.append(token)

    new_vocab = {token: new_id for new_id, token in enumerate(kept_tokens)}
    new_model = WordPiece(
        new_vocab,
        unk_token=tokenizer.model.unk_token,
        continuing_subword_prefix=prefix,
        max_input_chars_per_word=tokenizer.model.max_input_chars_per_word,
    )
    filtered = Tokenizer(new_model)
    filtered.normalizer = tokenizer.normalizer
    filtered.pre_tokenizer = tokenizer.pre_tokenizer
    filtered.decoder = tokenizer.decoder
    return filtered


class WordpieceEncodeFn:
    """Binds a tokenizer to encode_wordpiece so it matches datasets.py's encode_fn(seq, max_len)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seq, max_len):
        return encode_wordpiece(seq, self.tokenizer, max_len)