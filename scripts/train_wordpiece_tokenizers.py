"""Train WordPiece tokenizers at several vocab sizes on the train-only CDR3 corpus.

Input: artifacts/tokenizers/train_cdr3.txt (produced by prepare_wordpiece_corpus.py).
Output per vocab_size, under artifacts/tokenizers/wordpiece_vocab_{N}/:
    tokenizer.json  - full Tokenizer serialization (model+normalizer+pre_tokenizer+decoder)
    vocab.txt       - flat token list, one per line, ordered by id (BERT-style)

vocab_size=100 is a sanity-check only (WordPiece degenerates close to char-level
tokenization at this size) and is marked as such in the training summary; it is
not part of the main vocab_size ablation grid (1000/2000/5000/10000[/20000]).

Special tokens are fixed as [PAD],[UNK],[BOS],[EOS] with [PAD] forced to id=0,
matching ForwardModel/InverseModel's nn.Embedding(padding_idx=0) contract.
"""
import argparse
import json
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from irrm_codec.utils import setup_logging

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
CONTINUING_SUBWORD_PREFIX = "##"
SANITY_CHECK_VOCAB_SIZES = {100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WordPiece tokenizers at multiple vocab sizes.")
    parser.add_argument("--corpus-path", default="artifacts/tokenizers/train_cdr3.txt")
    parser.add_argument("--output-dir", default="artifacts/tokenizers")
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[100, 1000, 2000, 5000, 10000],
    )
    parser.add_argument("--min-frequency", type=int, default=1)
    return parser.parse_args()


def build_trainer(vocab_size: int, min_frequency: int) -> WordPieceTrainer:
    return WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix=CONTINUING_SUBWORD_PREFIX,
    )


def train_single_tokenizer(corpus_path: str, vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix=CONTINUING_SUBWORD_PREFIX)
    trainer = build_trainer(vocab_size, min_frequency)
    tokenizer.train([corpus_path], trainer)
    return tokenizer


def verify_special_tokens(tokenizer: Tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            raise ValueError(f"Special token {token!r} missing from trained vocab.")
    pad_id = vocab["[PAD]"]
    if pad_id != 0:
        raise ValueError(
            f"[PAD] must be id=0 to match ForwardModel/InverseModel padding_idx=0, got id={pad_id}."
        )


def sanity_check_roundtrip(tokenizer: Tokenizer, sample_seqs: list[str], logger: logging.Logger) -> None:
    for seq in sample_seqs:
        encoding = tokenizer.encode(seq)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        if decoded != seq:
            logger.warning(
                "roundtrip mismatch seq=%r decoded=%r tokens=%s", seq, decoded, encoding.tokens
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "train_wordpiece.log")

    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with corpus_path.open("r", encoding="utf-8") as handle:
        all_seqs = [line.strip() for line in handle if line.strip()]
    logger.info("loaded corpus rows=%d path=%s", len(all_seqs), corpus_path)
    sample_seqs = all_seqs[:20]

    results = {}
    for vocab_size in args.vocab_sizes:
        role = "sanity_check" if vocab_size in SANITY_CHECK_VOCAB_SIZES else "main_grid"
        logger.info("training WordPiece vocab_size=%d role=%s", vocab_size, role)
        tokenizer = train_single_tokenizer(str(corpus_path), vocab_size, args.min_frequency)

        verify_special_tokens(tokenizer)
        actual_vocab_size = tokenizer.get_vocab_size()
        if actual_vocab_size != vocab_size:
            # WordPiece counts base-of-word and ##continuation forms of each alphabet
            # character separately, so the floor is roughly 2 * unique_chars + len(SPECIAL_TOKENS).
            # actual can land above OR below the requested vocab_size near that floor.
            logger.warning(
                "requested vocab_size=%d but trainer produced actual_vocab_size=%d "
                "(near-floor mismatch, expected for small vocab_size given alphabet size)",
                vocab_size, actual_vocab_size,
            )

        sanity_check_roundtrip(tokenizer, sample_seqs, logger)

        vocab_dir = output_dir / f"wordpiece_vocab_{vocab_size}"
        vocab_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_path = vocab_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # WordPiece model.save() writes vocab.txt (one token per line, ordered by id)
        saved_files = tokenizer.model.save(str(vocab_dir))
        logger.info("saved model files: %s", saved_files)

        logger.info(
            "saved tokenizer vocab_size_requested=%d actual_vocab_size=%d dir=%s",
            vocab_size, actual_vocab_size, vocab_dir,
        )

        results[str(vocab_size)] = {
            "role": role,
            "requested_vocab_size": vocab_size,
            "actual_vocab_size": actual_vocab_size,
            "tokenizer_dir": str(vocab_dir),
            "tokenizer_path": str(tokenizer_path),
            "vocab_txt_path": str(vocab_dir / "vocab.txt"),
            "pad_id": tokenizer.get_vocab()["[PAD]"],
            "unk_id": tokenizer.get_vocab()["[UNK]"],
            "bos_id": tokenizer.get_vocab()["[BOS]"],
            "eos_id": tokenizer.get_vocab()["[EOS]"],
        }

    summary_path = output_dir / "wordpiece_training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote training summary path=%s", summary_path)


if __name__ == "__main__":
    main()