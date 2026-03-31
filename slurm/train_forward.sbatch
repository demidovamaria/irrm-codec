#!/usr/bin/env bash
#SBATCH --job-name=irrm-forward
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p slurm/logs

AIRR_PATH="${AIRR_PATH:-/projects/immunestatus/vdjdb/airr_format/trb_background.tsv}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/forward_demo_trb}"
LOCUS="${LOCUS:-beta}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_LEN="${MAX_LEN:-40}"
EMBEDDING_COLUMN="${EMBEDDING_COLUMN:-tcremp_emb}"
CLONE_ID_COL="${CLONE_ID_COL:-clone_id}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "root_dir=$ROOT_DIR"
echo "airr_path=$AIRR_PATH"
echo "embeddings_path=$EMBEDDINGS_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "locus=$LOCUS"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS"

"$PYTHON_BIN" -m irrm_codec.train_forward \
  --airr-path "$AIRR_PATH" \
  --embeddings-path "$EMBEDDINGS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --locus "$LOCUS" \
  --clone-id-col "$CLONE_ID_COL" \
  --embedding-column "$EMBEDDING_COLUMN" \
  --max-len "$MAX_LEN" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --train-fraction "$TRAIN_FRACTION" \
  --val-fraction "$VAL_FRACTION" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --log-interval "$LOG_INTERVAL" \
  --no-progress
