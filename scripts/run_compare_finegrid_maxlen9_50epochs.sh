#!/bin/bash
python scripts/compare_forward_tokenizers.py \
  --airr-path data/raw/trb_background_100k.tsv \
  --embeddings-path data/raw/trb_background_embeddings.parquet \
  --locus beta --max-len 40 \
  --batch-size 512 --epochs 50 \
  --max-token-len 9 \
  --seeds 1 42 777 \
  --configs char \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_100/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_500/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_1000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_2000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_3000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_4000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_5000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_6000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_7000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_8000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_9000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_10000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_12000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_15000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_17000/tokenizer.json \
            wordpiece:artifacts/tokenizers_2026-08-01_fine_grid/wordpiece_vocab_20000/tokenizer.json \
  --output-root artifacts/compare_forward_tokenizers_2026-08-01_fine_grid_maxlen9_50epochs
