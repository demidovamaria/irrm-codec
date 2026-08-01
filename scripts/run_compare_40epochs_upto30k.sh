#!/bin/bash
python scripts/compare_forward_tokenizers.py \
  --airr-path data/raw/trb_background_100k.tsv \
  --embeddings-path data/raw/trb_background_embeddings.parquet \
  --locus beta --max-len 40 \
  --batch-size 512 --epochs 40 \
  --seeds 1 42 777 \
  --configs char \
            wordpiece:artifacts/tokenizers_1M_integ/wordpiece_vocab_1000/tokenizer.json \
            wordpiece:artifacts/tokenizers_1M_integ/wordpiece_vocab_5000/tokenizer.json \
            wordpiece:artifacts/tokenizers_1M_integ/wordpiece_vocab_10000/tokenizer.json \
            wordpiece:artifacts/tokenizers_1M_integ/wordpiece_vocab_20000/tokenizer.json \
            wordpiece:artifacts/tokenizers_1M_integ/wordpiece_vocab_30000/tokenizer.json \
  --output-root artifacts/compare_forward_tokenizers_background100k_40epochs_upto30k
