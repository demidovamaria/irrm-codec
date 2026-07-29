"""Shared --tokenizer-type/--tokenizer-path/--vocab-size CLI wiring for training scripts.

Kept in one place so train_forward.py and train_inverse.py can't drift apart on how a
tokenizer is chosen and validated.
"""
from irrm_codec.tokenization import AA_VOCAB
from irrm_codec.wordpiece_tokenization import (
    WordpieceEncodeFn,
    load_wordpiece_tokenizer,
    wordpiece_vocab_size,
)


def add_tokenizer_args(parser):
    parser.add_argument("--tokenizer-type", choices=["char", "wordpiece"], default="char")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to a WordPiece tokenizer.json. Required when --tokenizer-type=wordpiece.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Optional explicit vocab size. If given, must match the tokenizer's actual "
        "vocab size (25 for char, tokenizer.json's own size for wordpiece) — this is a "
        "sanity check, not an override, since the embedding table must match the tokenizer.",
    )


def resolve_tokenizer(args, logger):
    """Validate args and build (encode_fn, vocab_size, tokenizer_info) for the chosen tokenizer.

    encode_fn is None for the char tokenizer (ForwardDataset/InverseDataset already default
    to char encode() when encode_fn=None) or a WordpieceEncodeFn for wordpiece.
    """
    if args.tokenizer_type == "wordpiece":
        if not args.tokenizer_path:
            raise ValueError("--tokenizer-path is required when --tokenizer-type=wordpiece.")
        tokenizer = load_wordpiece_tokenizer(args.tokenizer_path)
        vocab_size = wordpiece_vocab_size(tokenizer)
        if args.vocab_size is not None and args.vocab_size != vocab_size:
            raise ValueError(
                f"--vocab-size={args.vocab_size} does not match tokenizer's actual "
                f"vocab_size={vocab_size} ({args.tokenizer_path})."
            )
        encode_fn = WordpieceEncodeFn(tokenizer)
    else:
        if args.tokenizer_path:
            raise ValueError("--tokenizer-path was given but --tokenizer-type=char ignores it.")
        vocab_size = len(AA_VOCAB)
        if args.vocab_size is not None and args.vocab_size != vocab_size:
            raise ValueError(f"--vocab-size={args.vocab_size} does not match char vocab_size={vocab_size}.")
        encode_fn = None

    tokenizer_info = {
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": vocab_size,
    }
    logger.info(
        "tokenizer_type=%s vocab_size=%d tokenizer_path=%s",
        tokenizer_info["tokenizer_type"],
        tokenizer_info["vocab_size"],
        tokenizer_info["tokenizer_path"],
    )
    return encode_fn, vocab_size, tokenizer_info
