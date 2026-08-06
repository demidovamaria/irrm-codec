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
    """Encode a CDR3 sequence into exactly max_len ids: PAD-padded at the end.

    Raises on overflow (matches encode_wordpiece_anchored and the char tokenizer's
    gap_pad_cdr3) rather than silently truncating - a silently truncated sequence would
    lose real information every epoch without any warning.
    """
    seq = "" if seq is None else str(seq).strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    ids = tokenizer.encode(seq).ids
    if len(ids) > max_len:
        raise ValueError(f"Sequence '{seq}' encodes to {len(ids)} WordPiece tokens, exceeds max_len={max_len}.")
    return ids + [PAD_ID] * (max_len - len(ids))


def encode_wordpiece_anchored(seq, tokenizer, max_len, left_anchor=1, right_anchor=1):
    """Encode a CDR3 sequence, with padding split into the middle instead of the end.

    Keeps the first left_anchor and last right_anchor tokens at fixed offsets from the
    start/end of the buffer, and inserts all padding between them - e.g. for
    left_anchor=1, right_anchor=1: [tok0, PAD, PAD, ..., PAD, tok_{-1}]. The idea: the
    convolutional ForwardModel is position-sensitive, and the N-/C-terminal ends of a CDR3
    are its most conserved regions (see the WordPiece longest-token analysis earlier in
    this project) - with end-padding those ends land on a different absolute position in
    every example (depending on how long the encoded sequence happens to be), while
    anchoring keeps them at a fixed position always, mirroring what gap_pad_cdr3() already
    does for the char tokenizer (there with actual gap characters; here with real PAD_ID,
    still excluded from mean_pool/max_pool by the model's mask).

    Falls back to plain end-padding if there's no room for both anchors plus padding
    (short sequences, or left_anchor+right_anchor >= sequence length).
    """
    seq = "" if seq is None else str(seq).strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    ids = tokenizer.encode(seq).ids
    if len(ids) > max_len:
        raise ValueError(f"Sequence '{seq}' encodes to {len(ids)} WordPiece tokens, exceeds max_len={max_len}.")

    pad_total = max_len - len(ids)
    if pad_total == 0 or len(ids) <= left_anchor + right_anchor:
        return ids + [PAD_ID] * pad_total

    left = ids[:left_anchor]
    right = ids[len(ids) - right_anchor:]
    middle = ids[left_anchor: len(ids) - right_anchor]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return left + [PAD_ID] * pad_left + middle + [PAD_ID] * pad_right + right


def decode_wordpiece(token_ids, tokenizer, stop_at_eos=True):
    """Decode ids back to an amino-acid string, stopping at the first EOS by default."""
    ids = list(token_ids)
    if stop_at_eos and EOS_ID in ids:
        ids = ids[: ids.index(EOS_ID)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def wordpiece_vocab_size(tokenizer) -> int:
    """Vocab size, for ForwardModel/InverseModel(vocab_size=...)."""
    return tokenizer.get_vocab_size()


def _rebuild_tokenizer_with_vocab(tokenizer: Tokenizer, kept_tokens: list) -> Tokenizer:
    """Reassign contiguous ids (0..k-1) for kept_tokens (in their original relative order)
    and build a new Tokenizer sharing everything except the vocab.

    add_special_tokens() is required here: a freshly constructed Tokenizer(model) doesn't
    know which of its vocab entries are "special" (that's tracked separately from
    model.vocab) - without it, tokenizer.decode(ids, skip_special_tokens=True) silently
    stops skipping PAD/UNK/BOS/EOS. Registering strings already present in the model's
    vocab does not create new entries or change their ids.
    """
    new_vocab = {token: new_id for new_id, token in enumerate(kept_tokens)}
    new_model = WordPiece(
        new_vocab,
        unk_token=tokenizer.model.unk_token,
        continuing_subword_prefix=tokenizer.model.continuing_subword_prefix,
        max_input_chars_per_word=tokenizer.model.max_input_chars_per_word,
    )
    filtered = Tokenizer(new_model)
    filtered.normalizer = tokenizer.normalizer
    filtered.pre_tokenizer = tokenizer.pre_tokenizer
    filtered.decoder = tokenizer.decoder
    filtered.add_special_tokens(list(_EXPECTED_SPECIAL_IDS.keys()))
    return filtered


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

    return _rebuild_tokenizer_with_vocab(tokenizer, kept_tokens)


class WordpieceEncodeFn:
    """Binds a tokenizer to an encode_* function matching datasets.py's encode_fn(seq, max_len).

    pad_layout="end" (default) uses encode_wordpiece - unchanged from before this option
    existed. pad_layout="anchored" uses encode_wordpiece_anchored instead.
    """

    def __init__(self, tokenizer, pad_layout="end", left_anchor=1, right_anchor=1):
        if pad_layout not in ("end", "anchored"):
            raise ValueError(f"pad_layout must be 'end' or 'anchored', got {pad_layout!r}.")
        self.tokenizer = tokenizer
        self.pad_layout = pad_layout
        self.left_anchor = left_anchor
        self.right_anchor = right_anchor

    def __call__(self, seq, max_len):
        if self.pad_layout == "end":
            return encode_wordpiece(seq, self.tokenizer, max_len)
        return encode_wordpiece_anchored(seq, self.tokenizer, max_len, self.left_anchor, self.right_anchor)