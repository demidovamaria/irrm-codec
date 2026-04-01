import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID, encode


def validate_airr_dataframe(df, max_len=40):
    required_columns = {"junction_aa", "v_call", "j_call", "locus"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Dataframe is empty after filtering.")

    sequence_lengths = []
    unk_sequences = 0
    truncated_sequences = 0
    empty_sequences = 0

    for raw_seq in df["junction_aa"].tolist():
        seq = "" if raw_seq is None else str(raw_seq).strip().upper()
        if not seq:
            empty_sequences += 1
            continue
        sequence_lengths.append(len(seq))
        unk_sequences += int(any(char not in "ACDEFGHIKLMNPQRSTVWY" for char in seq))
        truncated_sequences += int(len(seq) > max_len)

    if empty_sequences:
        raise ValueError(f"Found {empty_sequences} empty or missing sequences.")

    return {
        "num_samples": len(df),
        "num_unique_clone_ids": int(df["clone_id"].nunique()) if "clone_id" in df.columns else int(len(df)),
        "min_length": int(min(sequence_lengths)),
        "max_length": int(max(sequence_lengths)),
        "mean_length": float(np.mean(sequence_lengths)),
        "truncated_fraction": truncated_sequences / len(df),
        "unk_sequence_fraction": unk_sequences / len(df),
        "max_len": max_len,
    }


def validate_dataframe(df, emb_array, max_len=40, emb_dim=9000):
    emb_array = np.asarray(emb_array, dtype=np.float32)
    if emb_array.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {emb_array.shape}.")
    if emb_array.shape[0] != len(df):
        raise ValueError(
            f"Embedding count {emb_array.shape[0]} does not match dataframe length {len(df)}."
        )
    if emb_array.shape[1] != emb_dim:
        raise ValueError(f"Expected embedding dimension {emb_dim}, got {emb_array.shape[1]}.")
    if not np.isfinite(emb_array).all():
        raise ValueError("Embedding array contains NaN or infinite values.")
    stats = validate_airr_dataframe(df, max_len=max_len)
    stats["embedding_dim"] = emb_array.shape[1]
    return stats


class ForwardDataset(Dataset):
    def __init__(self, df, emb_array, max_len=40):
        self.seqs = df["junction_aa"].tolist()
        self.embs = np.asarray(emb_array, dtype=np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx], self.max_len)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "embedding": torch.from_numpy(self.embs[idx]),
            "length": len(tokens),
        }


class InverseDataset(Dataset):
    def __init__(self, df, emb_array, max_len=40):
        self.seqs = df["junction_aa"].tolist()
        self.embs = np.asarray(emb_array, dtype=np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx], self.max_len)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return {
            "embedding": torch.from_numpy(self.embs[idx]),
            "decoder_input": torch.cat(
                [torch.tensor([BOS_ID], dtype=torch.long), token_tensor], dim=0
            ),
            "target": torch.cat(
                [token_tensor, torch.tensor([EOS_ID], dtype=torch.long)], dim=0
            ),
            "length": len(tokens),
        }


class StreamingEmbeddingDataset(IterableDataset):
    def __init__(
        self,
        *,
        task,
        records_by_key,
        selected_keys,
        iter_embedding_batches_fn,
        max_len,
        mean,
        std,
        shuffle=False,
        seed=42,
    ):
        self.task = task
        self.records_by_key = records_by_key
        self.selected_keys = set(selected_keys)
        self.iter_embedding_batches_fn = iter_embedding_batches_fn
        self.max_len = max_len
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.selected_keys)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _make_item(self, key, embedding):
        record = self.records_by_key[key]
        tokens = encode(record["junction_aa"], self.max_len)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        embedding_tensor = torch.from_numpy(embedding)

        if self.task == "forward":
            return {
                "tokens": token_tensor,
                "embedding": embedding_tensor,
                "length": len(tokens),
            }

        return {
            "embedding": embedding_tensor,
            "decoder_input": torch.cat([torch.tensor([BOS_ID], dtype=torch.long), token_tensor], dim=0),
            "target": torch.cat([token_tensor, torch.tensor([EOS_ID], dtype=torch.long)], dim=0),
            "length": len(tokens),
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + self.epoch + worker_id)

        for batch_index, (keys, emb_batch) in enumerate(self.iter_embedding_batches_fn()):
            if batch_index % num_workers != worker_id:
                continue

            matched_pairs = []
            for row_idx, key in enumerate(keys):
                if key in self.selected_keys:
                    matched_pairs.append((key, row_idx))

            if not matched_pairs:
                continue

            if self.shuffle and len(matched_pairs) > 1:
                rng.shuffle(matched_pairs)

            row_indices = [row_idx for _key, row_idx in matched_pairs]
            embeddings = emb_batch[row_indices]
            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D embedding batch, got shape {embeddings.shape}.")
            if embeddings.shape[1] != self.mean.shape[0]:
                raise ValueError(
                    f"Embedding dimension {embeddings.shape[1]} does not match standardizer dimension {self.mean.shape[0]}."
                )
            if not np.isfinite(embeddings).all():
                raise ValueError("Embedding batch contains NaN or infinite values.")

            standardized = ((embeddings - self.mean) / self.std).astype(np.float32, copy=False)
            for (key, _row_idx), embedding in zip(matched_pairs, standardized):
                yield self._make_item(key, embedding)


def collate_forward(batch):
    tokens = torch.nn.utils.rnn.pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    emb = torch.stack([item["embedding"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = tokens.ne(PAD_ID)
    return tokens, mask, emb, lengths


def collate_inverse(batch):
    emb = torch.stack([item["embedding"] for item in batch])
    decoder_input = torch.nn.utils.rnn.pad_sequence(
        [item["decoder_input"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    target = torch.nn.utils.rnn.pad_sequence(
        [item["target"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    target_mask = target.ne(PAD_ID)
    unk_fraction = target.eq(UNK_ID).logical_and(target_mask).float().sum() / target_mask.float().sum()
    return emb, decoder_input, target, lengths, unk_fraction
