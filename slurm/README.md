`slurm/train_forward.sbatch` launches the same forward training you ran from the notebook, but through `sbatch`.

Default paths in the job already match your current TRB run:

- AIRR: `/projects/immunestatus/vdjdb/airr_format/trb_background.tsv`
- embeddings: `/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet`
- output: `artifacts/forward_demo_trb`
- locus: `beta`

Submit as-is:

```bash
sbatch slurm/train_forward.sbatch
```

Override paths or hyperparameters without editing the file:

```bash
sbatch \
  --export=ALL,OUTPUT_DIR=/home/evlasova/irrm-codec/artifacts/forward_1m_trb,EPOCHS=10,BATCH_SIZE=512,NUM_WORKERS=16,PYTHON_BIN=/home/evlasova/.conda/envs/irrm-codec/bin/python \
  slurm/train_forward.sbatch
```

If your cluster needs a different queue or resources, edit the `#SBATCH` lines at the top of the script:

- `--partition`
- `--gres`
- `--cpus-per-task`
- `--mem`
- `--time`

Logs go to `slurm/logs/`.
